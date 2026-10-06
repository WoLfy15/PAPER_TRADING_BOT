"""
execution/reconciliation.py -- requirement 15: "implement a complete
order state machine that continuously reconciles submitted, partial,
filled, cancelled and replaced orders with the actual account
positions, preventing duplicate or unintended exposure."

This is also the direct fix for the "bookings not syncing with the
logs" report: agent/state.py's decision_log only gets a new entry once
per FULL agent cycle (every `--interval` seconds, 900 by default) and
records each order's SUBMISSION response -- a snapshot of "what we just
asked Alpaca to do," not what actually happened to it afterward. An
order can sit open, partially fill, fully fill, or get rejected any
time in the minutes between cycles, and nothing previously went back
and updated the log to say so. api/server.py's new GET /orders calls
reconcile() fresh on every poll (a few REST calls, not a full agent
cycle) specifically so the dashboard's order blotter reflects Alpaca's
REAL current state on its own fast refresh, independent of the much
slower decision-log cadence -- the two panels are now explicitly
different things (live blotter vs. reasoning history) rather than
silently drifting apart while looking like they should agree.

Every payload this project submits is stamped with a client_order_id
of the form "volagent-{kind}-{hex12}" (execution/order_manager.py) --
that prefix is ALL this module uses to recognize "an order we placed"
among whatever else might be in the account's order history (a person
could also be trading the same paper account by hand).
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from alpaca_client import AlpacaClient
from config import Settings

log = logging.getLogger("alpaca_vol_agent.reconciliation")

_KIND_PREFIX = "volagent-"
OPEN_STATUSES = frozenset({"new", "accepted", "pending_new", "partially_filled", "pending_cancel", "pending_replace"})
TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended"})


@dataclass
class ReconciledOrder:
    order_id: str
    client_order_id: str
    kind: str            # "straddle" | "condor" | "hedge" | "exit" | "unknown" -- parsed from client_order_id
    status: str           # Alpaca's own status string, unmodified
    symbol: str
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    submitted_at: Optional[str]
    updated_at: Optional[str]
    age_seconds: float
    is_open: bool
    is_stale: bool         # open AND older than settings.stale_order_max_age_minutes


@dataclass
class ReconciliationResult:
    as_of_utc: str
    open_orders: list[ReconciledOrder] = field(default_factory=list)
    recently_filled: list[ReconciledOrder] = field(default_factory=list)
    stale_orders: list[ReconciledOrder] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    open_notional_by_kind: dict[str, float] = field(default_factory=dict)  # for sizing.py's incremental-budget math

    @property
    def has_open_orders(self) -> bool:
        return len(self.open_orders) > 0


def _kind_of(client_order_id: str) -> str:
    if not client_order_id.startswith(_KIND_PREFIX):
        return "unknown"
    rest = client_order_id[len(_KIND_PREFIX):]
    for known in ("straddle", "condor", "hedge", "exit"):
        if rest.startswith(known):
            return known
    return "unknown"


def _parse_order(o: dict, now: dt.datetime) -> Optional[ReconciledOrder]:
    client_order_id = o.get("client_order_id", "")
    if not client_order_id.startswith(_KIND_PREFIX):
        return None  # not ours -- e.g. a person trading the same paper account by hand
    status = o.get("status", "unknown")
    submitted_at = o.get("submitted_at")
    updated_at = o.get("updated_at") or submitted_at
    age = 0.0
    if updated_at:
        try:
            ts = dt.datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
            age = max((now - ts).total_seconds(), 0.0)
        except ValueError:
            pass
    is_open = status in OPEN_STATUSES
    return ReconciledOrder(
        order_id=o.get("id", ""), client_order_id=client_order_id, kind=_kind_of(client_order_id),
        status=status, symbol=o.get("symbol") or (o.get("legs") or [{}])[0].get("symbol", ""),
        qty=float(o.get("qty") or 0), filled_qty=float(o.get("filled_qty") or 0),
        filled_avg_price=float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
        submitted_at=submitted_at, updated_at=updated_at, age_seconds=age,
        is_open=is_open, is_stale=False,  # is_stale set by the caller, which knows the threshold
    )


def reconcile(
    client: AlpacaClient, settings: Settings, cancel_stale: bool = True, lookback_limit: int = 200,
) -> ReconciliationResult:
    """The one function both agent/loop.py (before sizing new entries --
    requirement 15's "preventing duplicate exposure") and api/server.py's
    GET /orders (dashboard's live blotter) call. Pulls the account's
    recent order history fresh from Alpaca every time (status="all"),
    filters to this project's own orders by client_order_id prefix, and
    classifies each as open/filled/stale.

    cancel_stale: when True (agent/loop.py's default; api/server.py's
    read-only dashboard poll passes False so merely VIEWING the blotter
    never has a side effect), any OPEN order older than
    settings.stale_order_max_age_minutes is cancelled -- a resting limit
    that hasn't filled in that long is either mispriced against a book
    that's moved on, or resting behind a quote that's gone stale itself;
    either way, leaving it open risks the NEXT cycle sizing a fresh
    entry on top of it (double exposure) or the eventual fill landing at
    a price that no longer reflects the edge that justified the trade."""
    now = dt.datetime.now(dt.timezone.utc)
    raw_orders = client.get_orders(status="all", limit=lookback_limit)
    stale_threshold_s = settings.stale_order_max_age_minutes * 60.0

    open_orders: list[ReconciledOrder] = []
    recently_filled: list[ReconciledOrder] = []
    stale_orders: list[ReconciledOrder] = []
    open_notional_by_kind: dict[str, float] = {}

    for raw in raw_orders:
        parsed = _parse_order(raw, now)
        if parsed is None:
            continue
        if parsed.is_open:
            parsed.is_stale = parsed.age_seconds > stale_threshold_s
            open_orders.append(parsed)
            if parsed.is_stale:
                stale_orders.append(parsed)
            price_ref = parsed.filled_avg_price or float(raw.get("limit_price") or 0.0)
            open_notional_by_kind[parsed.kind] = open_notional_by_kind.get(parsed.kind, 0.0) + parsed.qty * price_ref * 100
        elif parsed.status == "filled" and parsed.age_seconds <= 3600:
            recently_filled.append(parsed)

    result = ReconciliationResult(
        as_of_utc=now.isoformat(), open_orders=open_orders, recently_filled=recently_filled,
        stale_orders=stale_orders, open_notional_by_kind=open_notional_by_kind,
    )

    if cancel_stale and stale_orders:
        for o in stale_orders:
            action: dict[str, Any] = {
                "action": "cancel_stale", "order_id": o.order_id, "client_order_id": o.client_order_id,
                "kind": o.kind, "age_seconds": round(o.age_seconds, 1),
            }
            if settings.dry_run:
                action["dry_run"] = True
            else:
                try:
                    client.cancel_order(o.order_id)
                    action["dry_run"] = False
                except Exception as exc:  # noqa: BLE001 -- a cancel racing a fill is expected occasionally
                    action["error"] = str(exc)
                    log.warning("reconcile: failed to cancel stale order %s: %s", o.order_id, exc)
            result.actions_taken.append(action)
            log.info(
                "reconcile: stale %s order %s (%.0fs old) -- %s",
                o.kind, o.order_id, o.age_seconds, "would cancel (dry_run)" if settings.dry_run else "cancelled",
            )

    return result
