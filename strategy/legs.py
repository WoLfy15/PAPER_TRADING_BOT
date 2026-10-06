"""
strategy/legs.py -- turns a sizing decision into concrete option
contracts, respecting what Alpaca's paper options levels actually allow.

This is the module that encodes a constraint worth stating plainly: per
docs.alpaca.markets/us/docs/options-trading, Alpaca's options levels
support Level 1 (covered call / cash-secured put), Level 2 (buy calls,
buy puts), and Level 3 (buy call spreads / buy put spreads via
order_class="mleg", where "all legs must be covered within the same
order" -- naked short legs are rejected even inside a multi-leg order).
There is no level that allows a naked short straddle or strangle.

So the two directions from strategy/sizing.py map to two different,
both fully compliant, structures:

  long_vol  (kelly > 0, buy volatility): a plain long straddle -- one
      single-leg BUY order for the ATM call, one for the ATM put. Level 2.
  short_vol (kelly < 0, sell volatility): an iron condor -- short strikes
      near `short_leg_target_delta`, long wings near `wing_target_delta`,
      submitted as ONE order_class="mleg" order with all four legs so
      Alpaca's "all legs covered in the same order" rule is satisfied by
      construction. Level 3. Every short leg in the structure is paired
      with a further-OTM long leg in the SAME order -- there is no
      uncovered leg at any point, including mid-fill (MLeg orders fill
      as a single unit or not at all).
"""
from __future__ import annotations

import datetime as dt
from us_time import us_today
from dataclasses import dataclass, replace
from typing import Optional

from data.market_data import ChainContract, implied_vol_for, to_option_quote
from pricing_engine.market_data.contract import OptionType
from pricing_engine.models.bsm import BlackScholesMerton
from pricing_engine.risk.scenarios import StressReport, build_stress_report

_bsm = BlackScholesMerton()


@dataclass
class StraddleLegs:
    expiry: dt.date
    call: ChainContract
    put: ChainContract
    price: float     # call mid + put mid, per share
    gamma: float      # per straddle (1 call + 1 put), per-underlying-unit convention
    vega: float


@dataclass
class IronCondorLegs:
    expiry: dt.date
    short_put: ChainContract
    long_put: ChainContract      # further OTM, protects the short put
    short_call: ChainContract
    long_call: ChainContract     # further OTM, protects the short call
    put_width: float
    call_width: float
    net_credit: float            # per share, positive = credit
    max_loss_per_contract: float  # dollars, conservative (ignores credit)
    contracts: int
    stress_report: Optional[StressReport] = None  # v3: requirement 11/19 -- see select_iron_condor_by_cvar


def condor_structure_legs(condor: IronCondorLegs) -> list[tuple[ChainContract, float]]:
    """(contract, signed_qty) pairs at the condor's ACTUAL contract count,
    the generic representation pricing_engine.risk.scenarios.
    _reprice_structure expects -- that function reprices a position's
    FULL size in one pass with no separate contracts multiplier applied
    anywhere else, so pre-scaling here is correct for that one caller.
    Signs match strategy/sizing.py's and execution/order_manager.py's
    convention: short legs negative.

    DO NOT use this for accounting/ledger.py's open_structure/
    close_structure -- those take contracts as their OWN separate
    parameter and multiply by it internally (so they can also work with
    single-contract-unit legs from build_straddle, which has no
    pre-scaled equivalent). Passing this function's ALREADY-scaled
    quantities into the ledger double-counts contracts (confirmed the
    hard way -- an early backtest/engine.py draft did exactly this and
    produced a 20-contract condor priced as if it were 400 contracts).
    Use condor_structure_legs_per_contract for the ledger instead."""
    n = condor.contracts
    return [
        (condor.short_put, -float(n)), (condor.long_put, +float(n)),
        (condor.short_call, -float(n)), (condor.long_call, +float(n)),
    ]


def condor_structure_legs_per_contract(condor: IronCondorLegs) -> list[tuple[ChainContract, float]]:
    """Same legs as condor_structure_legs, at PER-CONTRACT-UNIT signed
    quantities (+/-1, never condor.contracts itself) -- what
    accounting/ledger.py's open_structure/close_structure want, since
    those functions take `contracts` as their own separate argument and
    multiply by it themselves. See condor_structure_legs's docstring for
    why these two are deliberately not the same function."""
    return [
        (condor.short_put, -1.0), (condor.long_put, +1.0),
        (condor.short_call, -1.0), (condor.long_call, +1.0),
    ]


def select_expiry(contracts: list[ChainContract], target_dte_days: int, tolerance_days: int, as_of: Optional[dt.date] = None) -> Optional[dt.date]:
    as_of = as_of or us_today()
    expiries = sorted({c.expiry for c in contracts})
    if not expiries:
        return None
    best = min(expiries, key=lambda e: abs((e - as_of).days - target_dte_days))
    if abs((best - as_of).days - target_dte_days) > tolerance_days:
        return None
    return best


def _delta_of(contract: ChainContract, spot: float, risk_free_rate: float, as_of: Optional[dt.date] = None) -> Optional[float]:
    if contract.alpaca_delta is not None:
        return contract.alpaca_delta
    quote = to_option_quote(contract, spot, risk_free_rate, as_of=as_of)
    iv = implied_vol_for(quote)
    if iv is None or quote.expiry_years <= 0:
        return None
    return _bsm.delta(quote, iv)


def _nearest_by_abs_delta(
    candidates: list[ChainContract], target_abs_delta: float, spot: float, risk_free_rate: float, as_of: Optional[dt.date] = None,
) -> Optional[ChainContract]:
    scored = []
    for c in candidates:
        d = _delta_of(c, spot, risk_free_rate, as_of=as_of)
        if d is None:
            continue
        scored.append((abs(abs(d) - target_abs_delta), c))
    if not scored:
        return None
    return min(scored, key=lambda t: t[0])[1]


def build_straddle(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float, as_of: Optional[dt.date] = None,
) -> Optional[StraddleLegs]:
    same_expiry = [c for c in chain if c.expiry == expiry]
    calls = [c for c in same_expiry if c.option_type == OptionType.CALL]
    puts = [c for c in same_expiry if c.option_type == OptionType.PUT]
    if not calls or not puts:
        return None
    atm_strike = min({c.strike for c in calls} & {c.strike for c in puts}, key=lambda k: abs(k - spot), default=None)
    if atm_strike is None:
        return None
    call = next(c for c in calls if c.strike == atm_strike)
    put = next(c for c in puts if c.strike == atm_strike)
    if call.mid is None or put.mid is None:
        return None

    call_quote = to_option_quote(call, spot, risk_free_rate, as_of=as_of)
    put_quote = to_option_quote(put, spot, risk_free_rate, as_of=as_of)
    call_iv = implied_vol_for(call_quote)
    put_iv = implied_vol_for(put_quote)
    gamma = vega = 0.0
    if call_iv:
        gamma += _bsm.gamma(call_quote, call_iv)
        vega += _bsm.vega(call_quote, call_iv)
    if put_iv:
        gamma += _bsm.gamma(put_quote, put_iv)
        vega += _bsm.vega(put_quote, put_iv)

    return StraddleLegs(expiry=expiry, call=call, put=put, price=call.mid + put.mid, gamma=gamma, vega=vega)


def select_iron_condor_by_cvar(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float,
    short_delta: float, wing_delta_candidates: list[float], max_loss_budget: float, max_contracts: int,
    equity: float, max_structure_cvar_pct: float, cvar_alpha: float = 0.95, cvar_samples: int = 2000,
    as_of: Optional[dt.date] = None,
) -> Optional[IronCondorLegs]:
    """v3, requirement 11: "redesign the iron-condor construction around
    expected credit versus max loss/CVaR and scenario stress, rather
    than choosing wings only from delta." build_iron_condor above still
    does the actual strike selection (short_delta / a single wing_delta,
    unchanged) -- this wraps it: builds a ONE-CONTRACT condor at each
    candidate wing_delta, scores each by real-data-bootstrapped CVaR per
    dollar of credit collected (pricing_engine.risk.scenarios), picks
    the best-scoring width, THEN sizes contracts against BOTH the
    existing max_loss_budget (build_iron_condor's own logic, unchanged)
    AND a portfolio CVaR budget (max_structure_cvar_pct * equity) --
    whichever binds tighter wins, exactly the same "most restrictive cap
    applies" pattern size_long_straddle already uses for gamma/vega.

    wing_delta_candidates: e.g. [0.6, 1.0, 1.6] x the configured
    wing_target_delta -- tighter wings (less protection, more credit,
    more tail risk) vs. wider ones (more protection, less credit, less
    tail risk). Degrades to plain build_iron_condor's behavior if given
    a single-element list equal to the current default.

    as_of: forwarded to every BSM pricing call this makes (directly and
    via build_stress_report's own repricing) -- see to_option_quote's
    docstring for why this matters for anything other than a live
    "today is really today" call site."""
    candidates: list[tuple[float, IronCondorLegs, StressReport]] = []
    for wd in wing_delta_candidates:
        one_lot = build_iron_condor(chain, expiry, spot, risk_free_rate, short_delta, wd, max_loss_budget=1e12, max_contracts=1, as_of=as_of)
        if one_lot is None:
            continue
        report = build_stress_report(condor_structure_legs(one_lot), spot, risk_free_rate, alpha=cvar_alpha, n_samples=cvar_samples, as_of=as_of)
        cvar_per_contract = abs(report.monte_carlo_cvar_dollars)
        credit_per_contract = one_lot.net_credit * 100
        score = credit_per_contract / max(cvar_per_contract, 1.0)  # credit collected per dollar of bootstrapped tail risk
        candidates.append((score, one_lot, report))

    if not candidates:
        return None
    _best_score, best_one_lot, best_report = max(candidates, key=lambda t: t[0])
    cvar_per_contract = max(abs(best_report.monte_carlo_cvar_dollars), 1.0)

    sized = build_iron_condor(
        chain, expiry, spot, risk_free_rate, short_delta,
        wing_delta=_wing_delta_of(best_one_lot, spot, risk_free_rate, as_of=as_of),
        max_loss_budget=max_loss_budget, max_contracts=max_contracts, as_of=as_of,
    )
    if sized is None:
        return None

    cvar_max_contracts = int((max_structure_cvar_pct * equity) // cvar_per_contract)
    final_contracts = max(0, min(sized.contracts, cvar_max_contracts))
    if final_contracts == 0:
        return None
    if final_contracts != sized.contracts:
        sized = replace(sized, contracts=final_contracts)

    final_report = build_stress_report(condor_structure_legs(sized), spot, risk_free_rate, alpha=cvar_alpha, n_samples=cvar_samples, as_of=as_of)
    return replace(sized, stress_report=final_report)


def _wing_delta_of(condor: IronCondorLegs, spot: float, risk_free_rate: float, as_of: Optional[dt.date] = None) -> float:
    """Recovers the effective wing delta a one-lot condor was actually
    built at, so select_iron_condor_by_cvar can re-run build_iron_condor
    at the CHOSEN candidate's width with the real max_loss_budget/
    max_contracts -- build_iron_condor takes wing_delta as an input, not
    an output, so the winning candidate's own long legs are re-measured
    rather than threading an extra return value through build_iron_condor
    itself."""
    call_d = _delta_of(condor.long_call, spot, risk_free_rate, as_of=as_of)
    put_d = _delta_of(condor.long_put, spot, risk_free_rate, as_of=as_of)
    deltas = [abs(d) for d in (call_d, put_d) if d is not None]
    return sum(deltas) / len(deltas) if deltas else 0.05


def build_iron_condor(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float,
    short_delta: float, wing_delta: float, max_loss_budget: float, max_contracts: int,
    as_of: Optional[dt.date] = None,
) -> Optional[IronCondorLegs]:
    same_expiry = [c for c in chain if c.expiry == expiry]
    calls = sorted((c for c in same_expiry if c.option_type == OptionType.CALL), key=lambda c: c.strike)
    puts = sorted((c for c in same_expiry if c.option_type == OptionType.PUT), key=lambda c: c.strike)
    if len(calls) < 2 or len(puts) < 2:
        return None

    otm_calls = [c for c in calls if c.strike >= spot]
    otm_puts = [c for c in puts if c.strike <= spot]
    short_call = _nearest_by_abs_delta(otm_calls, short_delta, spot, risk_free_rate, as_of=as_of)
    short_put = _nearest_by_abs_delta(otm_puts, short_delta, spot, risk_free_rate, as_of=as_of)
    if short_call is None or short_put is None:
        return None

    further_calls = [c for c in otm_calls if c.strike > short_call.strike]
    further_puts = [c for c in otm_puts if c.strike < short_put.strike]
    long_call = _nearest_by_abs_delta(further_calls, wing_delta, spot, risk_free_rate, as_of=as_of) or (
        further_calls[-1] if further_calls else None
    )
    long_put = _nearest_by_abs_delta(further_puts, wing_delta, spot, risk_free_rate, as_of=as_of) or (
        further_puts[-1] if further_puts else None
    )
    if long_call is None or long_put is None:
        return None
    if any(c.mid is None for c in (short_call, long_call, short_put, long_put)):
        return None

    call_width = long_call.strike - short_call.strike
    put_width = short_put.strike - long_put.strike
    net_credit = (short_call.mid - long_call.mid) + (short_put.mid - long_put.mid)
    if net_credit <= 0:
        # A short iron condor should always collect a net credit (the
        # short inner strikes are worth more than the protective outer
        # wings). A non-positive net credit here means the quotes are
        # crossed/stale or the strike selection degenerated -- refuse to
        # build the order rather than submit a structure that pays a
        # debit to sell volatility, which would defeat the whole point.
        return None
    worst_width = max(call_width, put_width)
    max_loss_per_contract = max(worst_width * 100 - net_credit * 100, 0.0)
    if max_loss_per_contract <= 0:
        return None

    contracts = int(max_loss_budget // max_loss_per_contract)
    contracts = max(0, min(contracts, max_contracts))
    if contracts == 0:
        return None

    return IronCondorLegs(
        expiry=expiry, short_put=short_put, long_put=long_put, short_call=short_call, long_call=long_call,
        put_width=put_width, call_width=call_width, net_credit=net_credit,
        max_loss_per_contract=max_loss_per_contract, contracts=contracts,
    )
