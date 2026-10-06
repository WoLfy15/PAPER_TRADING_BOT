"""
agent/exits.py -- position exit evaluation logic.

Every cycle, agent/loop.py calls `evaluate_exits()` to scan all open
option positions and check three exit triggers:

  1. TAKE PROFIT: unrealized P&L exceeds a configurable threshold
     (default +50%). Locks in gains before mean-reversion erodes the edge.

  2. STOP LOSS: unrealized P&L drops below a configurable threshold
     (default -35%). Cuts losses to preserve capital -- the drawdown
     kill switch in strategy/portfolio.py is a portfolio-level backstop,
     but position-level stops fire much earlier and more surgically.

  3. NEAR EXPIRY: DTE drops to or below a configurable threshold
     (default 1 day). Avoids pin risk, gamma explosion on 0-DTE, and
     potential exercise/assignment on short legs of iron condors.

P&L is calculated from Alpaca's own `avg_entry_price` vs `current_price`
on each position, so it survives agent restarts -- no in-memory state
needed.

Exit orders flow through the same execution/order_manager.py choke point
and respect the same dry_run gate as every other order in this project.

v3 CHANGE -- exits are now judged per STRUCTURE, not per leg
------------------------------------------------------------
v2 evaluated every option position independently. That is fine for a
single long option, but this agent mostly opens two multi-leg structures
(a long straddle and an iron condor), and for the condor the per-leg rule
is actively wrong: the two long PROTECTIVE WINGS are cheap far-OTM
options that lose value fastest precisely when the trade is going well.
Measured with this project's own BSM pricer for a default-config weekly
condor (TARGET_DTE_DAYS=7) in a perfectly QUIET market -- the best case
for a short-vol trade:

    day 1: long wings -54% / -57%   (past the -35% stop-loss)
           short legs +27% / +27%   (below the +50% take-profit)

i.e. on day one of essentially every WINNING condor the per-leg rule
would order the protective wings closed while both short legs were still
open. Depending on how the broker treats that, the order is either
rejected every cycle (and, since the whole exit loop shared one
try/except, aborts any other exits that cycle) or -- worse -- accepted,
leaving naked short options: exactly the risk the wings exist to remove.

So: legs are grouped into the structures this agent actually opens, each
structure's AGGREGATE P&L is compared to the thresholds, and when a
structure triggers EVERY leg of it is emitted for closing together --
short legs first, so no intermediate state of the close sequence is ever
less covered than the starting position.

  * iron condor (net credit): pnl% = unrealized P&L / net credit
    received, so +50% means "captured half of max profit" -- the
    conventional condor take-profit rule.
  * long straddle (net debit): pnl% = unrealized P&L / total debit paid.
  * anything else (a lone long call, a lone short put, an odd leg
    left behind by a partial fill): evaluated on its own with EXACTLY the
    v2 arithmetic, so single-leg behavior is unchanged.

Grouping only happens when the legs unambiguously match a structure this
agent opens (equal contract counts across the legs, wings on the correct
side of the shorts). Anything that doesn't match falls through to the
per-leg rule rather than being guessed at.

KNOWN LIMITATION, on purpose not changed here: EXIT_STOP_LOSS_PCT=0.35
of NET CREDIT is a tight stop for a condor (a credit structure routinely
swings well past 35% of its credit and recovers). That threshold is the
project's own risk-policy choice, so it is left as configured -- but it
is worth revisiting for credit structures specifically now that it is no
longer being tripped by the wings' natural decay.
"""
from __future__ import annotations

import datetime as dt
from us_time import us_today
import logging
from dataclasses import dataclass
from typing import Optional

from config import RiskLimits
from pricing_engine.market_data.contract import OptionType
from strategy.portfolio import parse_occ_symbol

log = logging.getLogger("alpaca_vol_agent.exits")


@dataclass
class ExitDecision:
    """One decision to close a single option position (leg)."""
    symbol: str           # OCC option symbol
    qty: float            # signed: positive = long, negative = short
    reason: str           # "take_profit" | "stop_loss" | "near_expiry"
    pnl_pct: float        # unrealized P&L as a fraction (e.g. 0.50 = +50%) -- of the STRUCTURE this leg belongs to (v3)
    dte: int              # days to expiry
    avg_entry_price: float
    current_price: float
    market_value: float
    # v3: label of the structure this leg belongs to ("" for a standalone
    # position). agent/loop.py uses it to enforce "if a SHORT leg of a
    # structure fails to close, do NOT go on to close that structure's
    # protective long wings" -- see the exit loop there.
    group: str = ""


@dataclass
class _Leg:
    symbol: str
    qty: float
    entry: float
    current: float
    market_value: float
    strike: float
    otype: OptionType
    expiry: dt.date

    @property
    def valid_prices(self) -> bool:
        return self.entry > 0 and self.current > 0

    @property
    def pnl_dollars_per_share_unit(self) -> float:
        """Signed P&L of this leg per share of ONE contract-unit x qty
        (i.e. (current - entry) * qty). Positive = profit for a long
        that rose OR a short that fell, by construction of the sign of qty."""
        return (self.current - self.entry) * self.qty


def _parse_legs(positions: list[dict], underlying_symbol: str) -> list[_Leg]:
    legs: list[_Leg] = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        parsed = parse_occ_symbol(symbol)
        if parsed is None:
            continue
        root, expiry, otype, strike = parsed
        if root != underlying_symbol:
            continue
        qty = float(pos.get("qty", 0))
        if qty == 0:
            continue
        legs.append(_Leg(
            symbol=symbol, qty=qty, entry=float(pos.get("avg_entry_price", 0)),
            current=float(pos.get("current_price", 0)), market_value=float(pos.get("market_value", 0)),
            strike=strike, otype=otype, expiry=expiry,
        ))
    return legs


def _extract_condors(legs: list[_Leg]) -> tuple[list[list[_Leg]], list[_Leg]]:
    """Pulls unambiguous iron condors out of one expiry's legs: a short
    put + short call with equal contract counts, each with a long wing of
    that same count on the OUTER side (long put below the short put, long
    call above the short call -- nearest such strike). Returns
    (condors, leftover_legs)."""
    remaining = list(legs)
    condors: list[list[_Leg]] = []
    short_puts = sorted((l for l in remaining if l.qty < 0 and l.otype == OptionType.PUT), key=lambda l: -l.strike)
    for sp in short_puts:
        if sp not in remaining:
            continue
        n = abs(sp.qty)
        short_calls = sorted(
            (l for l in remaining if l.qty < 0 and l.otype == OptionType.CALL and abs(l.qty) == n), key=lambda l: l.strike
        )
        for sc in short_calls:
            put_wings = sorted(
                (l for l in remaining if l.qty > 0 and l.otype == OptionType.PUT and l.strike < sp.strike and l.qty == n),
                key=lambda l: -l.strike,
            )
            call_wings = sorted(
                (l for l in remaining if l.qty > 0 and l.otype == OptionType.CALL and l.strike > sc.strike and l.qty == n),
                key=lambda l: l.strike,
            )
            if not put_wings or not call_wings or sc.strike <= sp.strike:
                continue
            group = [put_wings[0], sp, sc, call_wings[0]]
            for leg in group:
                remaining.remove(leg)
            condors.append(group)
            break
    return condors, remaining


def _extract_straddles(legs: list[_Leg]) -> tuple[list[list[_Leg]], list[_Leg]]:
    """A long call + long put at the SAME strike with equal contract
    counts (what plan_straddle_buy opens)."""
    remaining = list(legs)
    straddles: list[list[_Leg]] = []
    for call in [l for l in remaining if l.qty > 0 and l.otype == OptionType.CALL]:
        if call not in remaining:
            continue
        match = next(
            (l for l in remaining if l.qty > 0 and l.otype == OptionType.PUT and l.strike == call.strike and l.qty == call.qty),
            None,
        )
        if match is not None:
            straddles.append([call, match])
            remaining.remove(call)
            remaining.remove(match)
    return straddles, remaining


def _extract_verticals(legs: list[_Leg]) -> tuple[list[list[_Leg]], list[_Leg]]:
    """Two legs, same type, equal counts, one short + one long (a credit or debit vertical spread)."""
    remaining, out = list(legs), []
    for short in sorted((l for l in remaining if l.qty < 0), key=lambda l: l.strike):
        if short not in remaining:
            continue
        n = abs(short.qty)
        cands = [l for l in remaining if l.qty == n and l.otype == short.otype and l.strike != short.strike]
        if cands:
            long_ = min(cands, key=lambda l: abs(l.strike - short.strike))
            out.append([long_, short]); remaining.remove(short); remaining.remove(long_)
    return out, remaining


def _group_pnl_pct(group: list[_Leg], kind: str) -> Optional[float]:
    """Aggregate P&L as a fraction of the structure's basis, or None when
    the basis can't be computed sanely (missing quotes, nonpositive
    credit/debit) -- the caller then treats the group as unevaluable for
    the P&L triggers (near-expiry still applies), the same way v2 treats
    a leg with no usable price."""
    if not all(l.valid_prices for l in group):
        return None
    pnl = sum(l.pnl_dollars_per_share_unit for l in group)
    if kind in ("condor", "vertical"):
        basis = sum(-l.qty * l.entry for l in group)      # net credit received (per share)
    else:                                                  # "straddle"
        basis = sum(l.qty * l.entry for l in group)       # total debit paid (per share)
    if basis <= 0:
        return None
    return pnl / basis


def _single_pnl_pct(leg: _Leg) -> float:
    """EXACTLY v2's per-leg arithmetic (kept verbatim in behavior)."""
    if leg.entry > 0 and leg.current > 0:
        pnl_pct = (leg.current - leg.entry) / leg.entry
    else:
        pnl_pct = 0.0
    if leg.qty < 0:
        pnl_pct = -pnl_pct
    return pnl_pct


def _decide(
    group: list[_Leg], pnl_pct: float, dte: int, risk: RiskLimits, label: str,
    kind: str = "single", peaks: Optional[dict] = None,
) -> list[ExitDecision]:
    tp = {"condor": risk.exit_condor_take_profit_pct, "vertical": risk.exit_condor_take_profit_pct, "straddle": risk.exit_straddle_take_profit_pct}.get(kind, risk.exit_take_profit_pct)
    sl = {"condor": risk.exit_condor_stop_loss_pct, "vertical": risk.exit_condor_stop_loss_pct, "straddle": risk.exit_straddle_stop_loss_pct}.get(kind, risk.exit_stop_loss_pct)
    close_dte = risk.exit_dte_close_days + (1 if kind == "straddle" else 0)   # long premium bleeds fastest in the last days
    peak = 0.0
    if peaks is not None:
        peak = max(peaks.get(label, 0.0), pnl_pct)
        peaks[label] = peak
    reason: Optional[str] = None
    if dte <= close_dte:
        reason = "near_expiry"
    elif pnl_pct >= tp:
        reason = "take_profit"
    elif peak >= risk.exit_trail_activate_pct and pnl_pct <= peak * risk.exit_trail_keep_fraction:
        reason = "trailing_stop"
    elif pnl_pct <= -sl:
        reason = "stop_loss"
    if reason is None:
        return []
    log.info("EXIT %s: %s pnl=%.1f%% peak=%.1f%% dte=%d", reason, label, pnl_pct * 100, peak * 100, dte)
    # Shorts first, then longs: closing a short leg first never leaves the
    # remaining position LESS covered than it started; closing a protective
    # long first would leave a naked short in between.
    ordered = sorted(group, key=lambda l: (0 if l.qty < 0 else 1, l.symbol))
    return [
        ExitDecision(
            symbol=l.symbol, qty=l.qty, reason=reason, pnl_pct=pnl_pct, dte=dte,
            avg_entry_price=l.entry, current_price=l.current, market_value=l.market_value,
            group=label if len(group) > 1 else "",
        )
        for l in ordered
    ]


def evaluate_exits(
    positions: list[dict],
    risk: RiskLimits,
    underlying_symbol: str,
    today: Optional[dt.date] = None,
    peaks: Optional[dict] = None,
) -> list[ExitDecision]:
    """Scan all positions from Alpaca's GET /v2/positions and return an
    ExitDecision for EVERY LEG of each structure (or single position)
    that triggers an exit -- see module docstring for the v3 structure-
    level semantics.

    `positions`: raw list[dict] from AlpacaClient.get_positions().
    `risk`: the RiskLimits with exit thresholds.
    `underlying_symbol`: only evaluate positions on this underlying.
    `today`: override for testing; defaults to date.today().
    """
    if today is None:
        today = us_today()

    legs = _parse_legs(positions, underlying_symbol)
    by_expiry: dict[dt.date, list[_Leg]] = {}
    for leg in legs:
        by_expiry.setdefault(leg.expiry, []).append(leg)

    exits: list[ExitDecision] = []
    for expiry in sorted(by_expiry):
        dte = max((expiry - today).days, 0)
        condors, rest = _extract_condors(by_expiry[expiry])
        straddles, rest2 = _extract_straddles(rest)
        verticals, singles = _extract_verticals(rest2)

        for group in condors:
            pnl = _group_pnl_pct(group, "condor")
            label = f"iron_condor {underlying_symbol} {expiry} strikes {sorted(l.strike for l in group)}"
            if pnl is None:
                # Unevaluable P&L (missing quote / nonpositive credit): only the
                # expiry trigger can apply, same as a v2 leg with no usable price.
                exits.extend(_decide(group, 0.0, dte, risk, label, "condor", peaks))
            else:
                exits.extend(_decide(group, pnl, dte, risk, label, "condor", peaks))
        for group in straddles:
            pnl = _group_pnl_pct(group, "straddle")
            label = f"straddle {underlying_symbol} {expiry} strike {group[0].strike:g}"
            exits.extend(_decide(group, 0.0 if pnl is None else pnl, dte, risk, label, "straddle", peaks))
        for group in verticals:
            pnl = _group_pnl_pct(group, "vertical" if any(l.qty < 0 for l in group) and sum(-l.qty * l.entry for l in group) > 0 else "straddle")
            exits.extend(_decide(group, 0.0 if pnl is None else pnl, dte, risk, f"vertical {underlying_symbol} {expiry} strikes {sorted(l.strike for l in group)}", "vertical", peaks))
        for leg in singles:
            exits.extend(_decide([leg], _single_pnl_pct(leg), dte, risk, f"{leg.symbol} qty={leg.qty:g}"))
    return exits
