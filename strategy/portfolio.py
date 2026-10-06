"""
strategy/portfolio.py -- the "portfolio overlay" track: reads whatever
is actually sitting in the Alpaca paper account right now (options AND
equity, however they got there -- this agent's own prior trades, or
positions the user placed by hand) and turns them into one set of
portfolio Greeks, a drawdown check, and a volatility-targeting scalar.

This is deliberately account-state-driven rather than trusting an
in-memory ledger: every cycle re-derives Greeks from Alpaca's own
positions endpoint, so a restart of the agent (or the FastAPI process)
never loses track of real risk.
"""
from __future__ import annotations

import datetime as dt
from us_time import us_today
import re
from dataclasses import dataclass, field
from typing import Optional

from config import RiskLimits
from pricing_engine.calibration.iv_solver import solve_implied_vol, IVSolverError
from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()

_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ_symbol(symbol: str) -> Optional[tuple[str, dt.date, OptionType, float]]:
    """Standard OCC option symbol, e.g. AAPL240119C00100000 -> ('AAPL',
    date(2024,1,19), CALL, 100.0). Alpaca's option contract symbols
    follow this format; see docs.alpaca.markets/us/docs/options-trading
    for a worked example."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root = m.group("root")
    expiry = dt.date(2000 + int(m.group("yy")), int(m.group("mm")), int(m.group("dd")))
    otype = OptionType.CALL if m.group("cp") == "C" else OptionType.PUT
    strike = int(m.group("strike")) / 1000.0
    return root, expiry, otype, strike


@dataclass
class PositionGreeks:
    symbol: str
    qty: float               # signed: long positive, short negative (contracts, not shares)
    market_value: float
    delta_shares: float       # qty * 100 * per-contract delta
    gamma_shares: float
    vega_dollars: float       # dollars of P&L per 1 vol POINT move (e.g. IV 15%->16%), across the full position
    theta_dollars: float      # per year; divide by 365 for a daily figure in the UI


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    equity_shares_held: float          # underlying shares currently held (the hedge leg)
    option_positions: list[PositionGreeks] = field(default_factory=list)
    net_delta_shares: float = 0.0
    net_gamma_shares: float = 0.0
    net_vega_dollars: float = 0.0
    net_theta_dollars: float = 0.0

    @property
    def net_delta_incl_hedge(self) -> float:
        return self.net_delta_shares + self.equity_shares_held


def _option_position_greeks(pos: dict, spot: float, risk_free_rate: float) -> Optional[PositionGreeks]:
    parsed = parse_occ_symbol(pos["symbol"])
    if parsed is None:
        return None
    _root, expiry, otype, strike = parsed
    expiry_years = max((expiry - us_today()).days, 0) / 365.0
    if expiry_years <= 0:
        return None

    qty = float(pos["qty"])
    side_sign = 1.0 if pos.get("side", "long") == "long" else -1.0
    signed_qty = abs(qty) * side_sign if float(pos.get("qty", 0)) >= 0 else float(pos["qty"])
    # Alpaca returns qty already signed for short equity, but option short
    # positions from a covered-call/spread can come through with a
    # positive qty and side="short" depending on endpoint version -- use
    # whichever sign is present, falling back to side.
    signed_qty = float(pos["qty"]) if float(pos["qty"]) != 0 else signed_qty

    current_price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
    quote = OptionQuote(
        underlying=pos.get("underlying_symbol", pos["symbol"][:6]),
        spot=spot, strike=strike, expiry_years=expiry_years, option_type=otype,
        risk_free_rate=risk_free_rate, market_price=current_price if current_price > 0 else None,
    )
    if quote.market_price is None:
        return None
    try:
        iv = solve_implied_vol(_bsm, quote, quote.market_price).implied_vol
    except IVSolverError:
        return None

    greeks = _bsm.greeks(quote, iv)
    mult = signed_qty * 100
    # v3 FIX: vega_dollars used to read `mult * greeks["vega"]` with no
    # /100 -- the identical units bug fixed in strategy/sizing.py's
    # size_long_straddle, and fixed here in the same pass for the same
    # reason (see that function's inline comment for the full
    # derivation): BlackScholesMerton.vega returns dollars per SHARE for
    # a full 100-POINT vol move, and max_vega_risk_pct is a per-1-POINT
    # budget, so the correct per-contract-per-point figure needs /100.
    return PositionGreeks(
        symbol=pos["symbol"], qty=signed_qty, market_value=float(pos.get("market_value", 0.0)),
        delta_shares=mult * greeks["delta"], gamma_shares=mult * greeks["gamma"],
        vega_dollars=mult * greeks["vega"] / 100, theta_dollars=mult * greeks["theta"],
    )


def build_portfolio_snapshot(account: dict, positions: list[dict], underlying_symbol: str, spot: float, risk_free_rate: float) -> PortfolioSnapshot:
    """v3: takes account/positions as already-fetched dicts rather than a
    client -- agent/loop.py now fetches both concurrently alongside its
    other independent reads (see that module's docstring on the v3
    concurrent-fetch phase) and passes them straight through here,
    instead of this function making its own redundant serial
    get_account()/get_positions() calls after the concurrent phase
    already has the data. Saves 2 sequential round trips every cycle."""

    equity_shares_held = 0.0
    option_greeks: list[PositionGreeks] = []
    for pos in positions:
        if pos.get("asset_class") == "us_equity" and pos.get("symbol") == underlying_symbol:
            equity_shares_held += float(pos.get("qty", 0.0))
        elif pos.get("asset_class") in ("us_option",) or (len(pos.get("symbol", "")) > 10 and parse_occ_symbol(pos["symbol"])):
            parsed = parse_occ_symbol(pos["symbol"])
            if parsed and parsed[0] == underlying_symbol:
                pg = _option_position_greeks(pos, spot, risk_free_rate)
                if pg:
                    option_greeks.append(pg)

    snap = PortfolioSnapshot(
        equity=float(account["equity"]), cash=float(account["cash"]),
        equity_shares_held=equity_shares_held, option_positions=option_greeks,
    )
    snap.net_delta_shares = sum(p.delta_shares for p in option_greeks)
    snap.net_gamma_shares = sum(p.gamma_shares for p in option_greeks)
    snap.net_vega_dollars = sum(p.vega_dollars for p in option_greeks)
    snap.net_theta_dollars = sum(p.theta_dollars for p in option_greeks)
    return snap


SHORT_VOL_RISK_MULTIPLIER = 6.0  # see existing_vol_exposure_notional's docstring


def existing_vol_exposure_notional(option_positions: list[PositionGreeks]) -> tuple[float, float]:
    """(deployed_long_vol_notional, deployed_short_vol_risk_proxy).

    Added so strategy/sizing.py can size an entry against the RISK BUDGET
    STILL REMAINING rather than the full budget every cycle -- see
    decide_sizing's docstring for why this matters. This agent only ever
    puts on long straddles (qty > 0, both legs) or defined-risk short
    iron condors (qty < 0 short strikes, paired long wings -- see
    strategy/legs.py), so a simple sign split on existing positions is a
    reasonable, cheap classifier without needing to reconstruct which
    positions belong to which historical structure:

      long_vol_notional  = sum(market_value) over qty > 0 legs -- the
        premium already paid for straddle-like long exposure. Exact and
        apples-to-apples with strategy/sizing.py's long-vol target
        budget, which is ALSO a premium-dollar figure.

      short_vol_risk_proxy = SHORT_VOL_RISK_MULTIPLIER * sum(|market_value|)
        over qty < 0 legs. NOT apples-to-apples on its own: the target
        short-vol budget is a MAX-LOSS figure (strategy/sizing.py sizes
        a credit structure by its defined risk, not its collected
        premium), while a short leg's own market_value is just the
        premium at risk on THAT leg -- for a real iron condor the max
        loss is (wing width - net credit), typically several times the
        credit alone. Using raw premium unscaled was checked against a
        synthetic existing position while building this and materially
        under-counted deployed risk (a fully-sized 20-lot condor's
        short legs priced at single-digit-dollar premiums against a
        ~$45k max-loss budget) -- meaning the incremental-sizing check
        would rarely bind for the short-vol path in practice, even
        though the book was already fully sized. SHORT_VOL_RISK_MULTIPLIER
        is a fixed heuristic scalar (typical wing-width-to-credit ratio
        for the delta targets this project defaults to,
        config.Settings.short_leg_target_delta=0.16 /
        wing_target_delta=0.05), not a re-derivation of each open
        structure's actual width -- tune it per symbol/strike spacing if
        the chain's typical condors run wider or narrower than that."""
    long_notional = sum(p.market_value for p in option_positions if p.qty > 0)
    short_notional = SHORT_VOL_RISK_MULTIPLIER * sum(abs(p.market_value) for p in option_positions if p.qty < 0)
    return float(long_notional), float(short_notional)


def check_drawdown_kill_switch(equity_now: float, equity_high_water_mark: float, risk: RiskLimits) -> tuple[bool, float]:
    """Mirrors IITfinal.ipynb's hard stop (25% drawdown -> flatten). Returns
    (breached, drawdown_pct). Caller (agent/loop.py) owns persisting the
    high-water mark across cycles and deciding what "flatten" means."""
    if equity_high_water_mark <= 0:
        return False, 0.0
    drawdown = (equity_now - equity_high_water_mark) / equity_high_water_mark
    return drawdown < -risk.max_drawdown_pct, drawdown


def check_loss_circuit_breakers(
    daily_pnl_pct: Optional[float], weekly_pnl_pct: Optional[float], risk: RiskLimits,
) -> tuple[bool, list[str]]:
    """v3, requirement 10: daily/weekly loss circuit breakers, DISTINCT
    from check_drawdown_kill_switch's peak-to-trough measure above -- a
    fast single-day loss can sit well inside a 25% max-drawdown budget
    (e.g. the very first losing day off a fresh high-water mark) and
    still be exactly the kind of single event worth halting new entries
    over immediately, rather than waiting for it to compound into an
    overall drawdown breach. Inputs come from
    agent.state.compute_recent_loss_pcts, itself a rollup of the same
    equity_curve the drawdown check's high-water mark is derived from --
    no separate ledger needed. Returns (breached, reasons); breached
    means agent/loop.py should refuse ALL new entries (both directions,
    unlike the event filter which is short-vol-only -- a large loss is
    evidence the CURRENT risk model or regime read is off, which cuts
    against opening either kind of new position, not just one side)."""
    reasons: list[str] = []
    if daily_pnl_pct is not None and daily_pnl_pct < -risk.max_daily_loss_pct:
        reasons.append(f"Daily loss {daily_pnl_pct:+.2%} breached -max_daily_loss_pct {risk.max_daily_loss_pct:.2%}")
    if weekly_pnl_pct is not None and weekly_pnl_pct < -risk.max_weekly_loss_pct:
        reasons.append(f"Weekly loss {weekly_pnl_pct:+.2%} breached -max_weekly_loss_pct {risk.max_weekly_loss_pct:.2%}")
    return bool(reasons), reasons


def remaining_greek_headroom(snap: PortfolioSnapshot, risk: RiskLimits) -> dict[str, float]:
    """v3, requirement 10: how much delta/gamma/vega/theta capacity is
    LEFT before the portfolio-level hard caps bind, given what's already
    on the book. Values can be negative (already over cap -- e.g. a
    position placed by hand outside this agent, or a cap that was
    tightened after entries were already on). strategy/legs.py uses this
    to shrink a NEW structure's contract count so the resulting NET book
    (existing + new) respects every cap -- gamma/vega joins delta/theta
    here rather than being separately capped only for the long-vol path
    the way v2 had it (size_long_straddle's own gamma/vega check): this
    is now the single place all four caps are enforced, for BOTH
    directions, against the actual current book rather than the new
    structure's greeks considered in isolation."""
    equity = snap.equity if snap.equity > 0 else 1.0
    return {
        "delta_shares": risk.max_delta_risk_pct * equity - abs(snap.net_delta_incl_hedge),
        "gamma_shares": risk.max_gamma_risk_pct * equity - abs(snap.net_gamma_shares),
        "vega_dollars": risk.max_vega_risk_pct * equity - abs(snap.net_vega_dollars),
        "theta_dollars": risk.max_theta_risk_pct * equity - max(0.0, -snap.net_theta_dollars),   # v4.4: positive theta is the short-vol INCOME, not a risk (it used to count |theta|, which blocked even one 7-DTE condor below ~$1.3M of equity)
    }


def vol_target_scale(realized_portfolio_vol: Optional[float], risk: RiskLimits) -> float:
    """Scales new entries down if the book's recent realized vol is
    already running hot relative to the target -- same role as
    IITfinal.ipynb's `vol_scale = target_vol / realized_vol`. Returns 1.0
    (no scaling) until there's enough portfolio return history to
    estimate realized_portfolio_vol.

    v3, requirement 9: target_portfolio_vol is now a BAND
    ([target_portfolio_vol_min, target_portfolio_vol_max], default
    8%-12%), not a single point -- realized vol landing anywhere inside
    the band returns 1.0 (no scaling at all) rather than continuously
    fighting to pull realized vol to one exact number every cycle. Only
    OUTSIDE the band does a scalar apply, pulling toward the nearer
    edge of the band (not all the way to the center) -- same 1.2x
    expansion ceiling as before on the upside, so a quiet-vol stretch
    still can't size up more aggressively than v2 allowed."""
    if not realized_portfolio_vol or realized_portfolio_vol <= 0:
        return 1.0
    if risk.target_portfolio_vol_min <= realized_portfolio_vol <= risk.target_portfolio_vol_max:
        return 1.0
    if realized_portfolio_vol > risk.target_portfolio_vol_max:
        return risk.target_portfolio_vol_max / realized_portfolio_vol
    return min(risk.target_portfolio_vol_min / realized_portfolio_vol, 1.2)



def fit_contracts_to_greek_caps(existing_positions: list[dict], proposed: list[tuple], contracts: int, account: dict,
                                underlying: str, spot: float, risk_free_rate: float, risk: RiskLimits) -> int:
    """v4.4: EXISTING + PROPOSED book against the delta/gamma/vega/theta caps BEFORE any order is sent. `proposed` is
    [(ChainContract, signed_qty_at_`contracts`)]; returns the largest contract count <= `contracts` whose NET book fits
    every cap (0 = does not fit even at one contract)."""
    for k in range(contracts, 0, -1):
        legs = []
        for c, q in proposed:
            mid = ((c.bid + c.ask) / 2) if (c.bid and c.ask) else (c.last or 0.0)
            qty = q * k / contracts
            legs.append({"symbol": c.symbol, "asset_class": "us_option", "qty": qty, "side": "long" if qty > 0 else "short",
                         "avg_entry_price": mid, "current_price": max(mid, 0.01), "market_value": qty * mid * 100})
        snap = build_portfolio_snapshot(account, list(existing_positions) + legs, underlying, spot, risk_free_rate)
        if min(remaining_greek_headroom(snap, risk).values()) >= 0:
            return k
    return 0


def realized_book_vol(equity_curve: list[dict], lookback_days: int = 20, min_days: int = 8) -> Optional[float]:
    """Annualized vol of the account's DAILY equity returns (last equity of each UTC day), or None if too short."""
    import datetime as _dt
    import math
    last: dict = {}
    for row in equity_curve:
        try:
            last[_dt.datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")).date()] = float(row["equity"])
        except Exception:  # noqa: BLE001
            continue
    eq = [last[d] for d in sorted(last)][-(lookback_days + 1):]
    rets = [math.log(b / a) for a, b in zip(eq[:-1], eq[1:]) if a > 0 and b > 0]
    if len(rets) < min_days:
        return None
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(252)
