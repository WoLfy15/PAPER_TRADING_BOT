"""
execution/order_manager.py -- the only place in this codebase that
builds an Alpaca order payload or calls alpaca_client.submit_order.

Every function here returns the payload it WOULD submit; submit_orders()
is the single choke point that actually calls the network, and it is
gated by settings.dry_run. This split matters for the UI this backend
sits behind: the API can call the `build_*` functions and show the user
exactly what the agent is about to do, then let the user (or an
autonomous loop running with dry_run=False) decide whether it fires.

v2: straddle entries moved from plain MARKET orders to marketable LIMIT
orders (mid + a small buffer). A market order on a 2-3-wide indicative-
feed option quote pays the full spread with no ceiling; a limit at
mid + buffer still fills promptly against a reasonable book while
capping the worst-case slippage -- and the buffer widens when
research/toxicity.py's flow-toxicity read is elevated, i.e. exactly
when a resting order is more likely to get picked off. This is a real
cost reduction, not a cosmetic order-type swap: on a $5 straddle with a
$0.10 spread, a market order can pay the full $0.05/side vs. mid; a 2%
limit buffer caps that at $0.10 while still being marketable against a
normal book.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger("alpaca_vol_agent.order_manager")

from alpaca_client import AlpacaClient
from config import Settings
from data.market_data import executable_price
from strategy.legs import IronCondorLegs, StraddleLegs


@dataclass
class PlannedOrder:
    kind: str            # "straddle_buy" | "iron_condor" | "equity_hedge"
    payloads: list[dict]  # one Alpaca order payload per structure (see plan_straddle_buy's v3 note)
    description: str
    executable_reference_price: Optional[float] = None  # v3: conservative (ask-side) reference used for risk figures, NOT the resting limit


def plan_straddle_buy(
    straddle: StraddleLegs, contracts: int, limit_price_buffer_bps: float = 200.0,
) -> PlannedOrder:
    """v3: ONE order_class='mleg' order (both legs buy_to_open,
    ratio_qty 1:1) instead of v2's two independent single-leg orders --
    requirement 14 ("execute multi-leg structures atomically... to
    eliminate legging risk"). v2's reasoning for two independent orders
    was real (Alpaca's Level 2 permits two independent BUY orders, and
    neither leg is naked) but was a compliance argument, not an
    execution-quality one: two independent marketable orders can still
    fill at very different moments (or one fill and the other miss
    entirely under a fast-moving quote), leaving an unintended naked
    call or put instead of the intended straddle. Alpaca's mleg order
    type supports two BUY legs in the same combo order (it isn't
    restricted to credit spreads), so bundling costs nothing here and
    removes the legging risk outright -- the order fills as the
    straddle or not at all, the same guarantee plan_iron_condor's mleg
    order already gives the condor.

    The limit is a NET DEBIT across both legs (mirroring
    plan_iron_condor's net-credit convention), built from each leg's
    OWN executable ask (requirement 12 -- the price you'd actually pay
    crossing the spread right now) plus limit_price_buffer_bps, not raw
    mid -- mid understates the true cost of actually acquiring a long
    position and would make max-loss/notional figures downstream of
    this function optimistic versus what a marketable order can really
    achieve. `PlannedOrder.executable_reference_price` carries that
    ask-based total separately from the (buffered, i.e. intentionally
    friendlier-than-ask) resting limit, so callers can log/display both
    honestly rather than conflating "what we're resting at" with "what
    it would cost to guarantee a fill."""
    call_ask = executable_price(straddle.call, "buy") or straddle.call.mid or straddle.price / 2.0
    put_ask = executable_price(straddle.put, "buy") or straddle.put.mid or straddle.price / 2.0
    ask_total = call_ask + put_ask
    limit = round(ask_total * (1.0 + limit_price_buffer_bps / 10_000.0), 2)
    # ^ buffer widens ABOVE the ask total here (unlike the old per-leg
    # buffer above MID) because the reference point moved from mid to
    # ask -- resting a BUY combo above the current ask total, not above
    # mid, is what actually gives it a realistic chance of filling
    # promptly while still capping worst-case cost at a known ceiling.
    legs = [
        {"symbol": straddle.call.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        {"symbol": straddle.put.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
    ]
    payload = {
        "order_class": "mleg",
        "qty": str(contracts),
        "type": "limit",
        "limit_price": f"{limit:.2f}",
        "time_in_force": "day",
        "legs": legs,
        "client_order_id": f"volagent-straddle-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind="straddle_buy", payloads=[payload],
        description=(
            f"BUY {contracts}x straddle on {straddle.call.underlying} {straddle.expiry.isoformat()} "
            f"{straddle.call.strike:g} strike (call {straddle.call.symbol} + put {straddle.put.symbol}), "
            f"one atomic mleg order, resting at ${limit:.2f}/combo "
            f"(~${limit * 100 * contracts:,.0f} worst-case notional), "
            f"limit buffer {limit_price_buffer_bps:.0f}bps over the executable ask total ${ask_total:.2f}."
        ),
        executable_reference_price=ask_total,
    )


def plan_iron_condor(
    condor: IronCondorLegs, limit_price: Optional[float] = None, limit_buffer_bps: float = 0.0,
) -> PlannedOrder:
    """One order_class='mleg' order, all four legs covered within the
    order (long wings protect the short strikes) -- compliant with
    Alpaca's Level 3 multi-leg rules, including the "no uncovered legs,
    even mid-fill" restriction, since MLeg orders fill as a single unit.
    limit_buffer_bps: shave this many bps off the theoretical net credit
    to make the resting limit more likely to fill (0 = the original v1
    behavior of resting exactly at net_credit); widened by agent/loop.py
    under elevated flow toxicity, same rationale as the straddle buffer.

    v3, requirement 12: when limit_price is left as None, the credit
    this rests at is now built from each leg's own EXECUTABLE price
    (bid for the two short legs sold, ask for the two long legs bought)
    rather than mid -- condor.net_credit itself (used for the max-loss/
    sizing math in strategy/legs.py) already reflects live mid pricing
    at build time, so this keeps the ORDER's resting price and the
    RISK figures it was sized against from quietly drifting apart when
    the book is wide. Passing limit_price explicitly still overrides
    this entirely, unchanged from v2."""
    if limit_price is not None:
        limit = limit_price
        executable_ref = None
    else:
        short_put_ref = executable_price(condor.short_put, "sell") or condor.short_put.mid or 0.0
        long_put_ref = executable_price(condor.long_put, "buy") or condor.long_put.mid or 0.0
        short_call_ref = executable_price(condor.short_call, "sell") or condor.short_call.mid or 0.0
        long_call_ref = executable_price(condor.long_call, "buy") or condor.long_call.mid or 0.0
        executable_ref = (short_put_ref - long_put_ref) + (short_call_ref - long_call_ref)
        limit = round(executable_ref * (1.0 - limit_buffer_bps / 10_000.0), 2)
    legs = [
        {"symbol": condor.long_put.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        {"symbol": condor.short_put.symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
        {"symbol": condor.short_call.symbol, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
        {"symbol": condor.long_call.symbol, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
    ]
    payload = {
        "order_class": "mleg",
        "qty": str(condor.contracts),
        "type": "limit",
        "limit_price": f"{limit:.2f}",
        "time_in_force": "day",
        "legs": legs,
        "client_order_id": f"volagent-condor-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind="iron_condor", payloads=[payload],
        description=(
            f"SELL {condor.contracts}x iron condor on {condor.short_call.underlying} "
            f"{condor.expiry.isoformat()}: puts {condor.long_put.strike:g}/{condor.short_put.strike:g}, "
            f"calls {condor.short_call.strike:g}/{condor.long_call.strike:g}, "
            f"credit ~${limit * 100 * condor.contracts:,.0f}, "
            f"max loss ~${condor.max_loss_per_contract * condor.contracts:,.0f}."
        ),
        executable_reference_price=executable_ref,
    )


def plan_equity_hedge(symbol: str, trade_shares: float) -> Optional[PlannedOrder]:
    if abs(trade_shares) < 1:
        return None
    qty = int(round(abs(trade_shares)))
    side = "buy" if trade_shares > 0 else "sell"
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day",
        "client_order_id": f"volagent-hedge-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind="equity_hedge", payloads=[payload],
        description=f"{side.upper()} {qty} shares {symbol} to rebalance the delta hedge.",
    )


def plan_position_close(
    symbol: str, qty: float, reason: str, pnl_pct: float = 0.0, dte: int = 0,
) -> Optional[PlannedOrder]:
    """Close an existing option position by selling (long) or buying to close (short).
    `reason` is one of 'take_profit', 'stop_loss', 'near_expiry'.
    `qty` is signed: positive for long positions (sell to close),
    negative for short positions (buy to close)."""
    if abs(qty) < 1:
        return None
    abs_qty = int(round(abs(qty)))
    # Long positions: sell to close. Short positions: buy to close.
    side = "sell" if qty > 0 else "buy"
    intent = "sell_to_close" if qty > 0 else "buy_to_close"
    pnl_str = f"{pnl_pct:+.1%}" if pnl_pct else ""
    reason_label = reason.replace("_", " ").upper()
    payload = {
        "symbol": symbol,
        "qty": str(abs_qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "position_intent": intent,
        "client_order_id": f"volagent-exit-{uuid.uuid4().hex[:12]}",
    }
    return PlannedOrder(
        kind=f"exit_{reason}",
        payloads=[payload],
        description=(
            f"[{reason_label}] {side.upper()} {abs_qty}x {symbol} to close "
            f"(P&L {pnl_str}, {dte} DTE)."
        ),
    )


def submit_orders(client: AlpacaClient, plan: PlannedOrder, settings: Settings) -> dict[str, Any]:
    if settings.dry_run:
        return {"submitted": False, "dry_run": True, "would_submit": plan.payloads, "description": plan.description}
    results = [client.submit_order(p) for p in plan.payloads]
    return {"submitted": True, "dry_run": False, "orders": results, "description": plan.description}


def submit_close_position(client: AlpacaClient, symbol: str, settings: Settings, reason: str) -> dict[str, Any]:
    """Close a position via DELETE /v2/positions/{symbol}. If Alpaca rejects
    with 403 'no available quote' (illiquid / expired option), the position
    is already effectively worthless -- we record it as skipped rather than
    crashing the whole cycle. dry_run gate still applies."""
    if settings.dry_run:
        return {"submitted": False, "dry_run": True, "close_symbol": symbol, "reason": reason}
    try:
        result = client.close_position(symbol)
        return {"submitted": True, "dry_run": False, "close_result": result, "reason": reason}
    except Exception as exc:
        err_str = str(exc)
        # 403 'no available quote' means the option has no bid -- it is
        # effectively worthless (expired or deeply OTM). Log and skip.
        if "40310000" in err_str or "no available quote" in err_str:
            log.warning("close_position %s: no quote (likely worthless) -- skipping close: %s", symbol, exc)
            return {"submitted": False, "dry_run": False, "skipped": True,
                    "reason": reason, "skip_reason": "no_quote_worthless"}
        raise
