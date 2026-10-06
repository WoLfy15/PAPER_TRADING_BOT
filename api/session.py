"""
api/session.py -- the Run / Stop controller behind the dashboard.

One background thread runs agent.loop.run_cycle() for ONE symbol, with
dry_run=False (real orders on the Alpaca PAPER account), every
`interval_seconds`, only while the market is open. Stop is cooperative:
the cycle already in flight finishes (an order can't be safely aborted
half-way), then the loop exits. "Stop & close" additionally cancels this
agent's open orders for the symbol and closes its positions, short legs
first so a protective wing is never closed while its short is still open
(same rule as agent/loop._execute_exits).

State lives in memory: restarting the API server ends the session.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import re
import threading
from collections import deque
from dataclasses import asdict
from typing import Any, Optional

from agent.loop import run_cycle
from alpaca_client import AlpacaClient
from config import SETTINGS

log = logging.getLogger("alpaca_vol_agent.session")

OCC = re.compile(r"^([A-Z.]{1,6})\d{6}[CP]\d{8}$")
MAX_CONSECUTIVE_ERRORS = 5
PAUSE_SECONDS = int(os.getenv("CYCLE_PAUSE_SECONDS", "15"))  # breather between back-to-back cycles (API rate limits)
ACTIVE_STATES = ("running", "waiting_for_market", "stopping")


class SessionBusy(RuntimeError):
    pass


def underlying_of(symbol: Optional[str]) -> str:
    m = OCC.match(symbol or "")
    return m.group(1) if m else (symbol or "")


def _order_symbols(order: dict) -> list[str]:
    syms = [order.get("symbol")] + [leg.get("symbol") for leg in (order.get("legs") or [])]
    return [s for s in syms if s]


class TradingSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._flatten = False
        self.events: deque = deque(maxlen=60)
        self._reset(None, 900)
        self.state = "idle"

    def _reset(self, symbol: Optional[str], interval: int) -> None:
        self.symbol = symbol
        self.interval = interval
        self.state = "running"
        self.started_at: Optional[str] = None
        self.cycles = 0
        self.orders_sent = 0
        self.next_cycle_at: Optional[dt.datetime] = None
        self.last_result: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def _event(self, kind: str, msg: str) -> None:
        self.events.appendleft({"t": dt.datetime.utcnow().isoformat() + "Z", "kind": kind, "msg": msg})
        log.info("session[%s] %s", kind, msg)

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    # ------------------------------------------------------------------ API
    def start(self, symbol: str, interval_seconds: int) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise SessionBusy(f"A session for {self.symbol} is still {self.state}. Stop it first.")
            self._stop.clear()
            self._flatten = False
            self._reset(symbol, interval_seconds)
            self.started_at = dt.datetime.utcnow().isoformat() + "Z"
            self.events.clear()
            self._event("start", f"Started trading {symbol}. Running continuously until you press Stop.")
            self._thread = threading.Thread(target=self._run, name=f"session-{symbol}", daemon=True)
            self._thread.start()
        return self.snapshot()

    def stop(self, flatten: bool = False) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._flatten = flatten
                self.state = "stopping"
                self._event("stop", "Stop requested. Finishing the current cycle" + (", then closing positions." if flatten else "."))
                self._stop.set()
        return self.snapshot()

    def snapshot(self) -> dict:
        nxt = self.next_cycle_at
        return {
            "state": self.state, "symbol": self.symbol, "interval_seconds": self.interval,
            "started_at": self.started_at, "cycles": self.cycles, "orders_sent": self.orders_sent,
            "seconds_to_next": max(0, int((nxt - dt.datetime.utcnow()).total_seconds())) if nxt else None,
            "last_error": self.last_error, "last_result": self.last_result, "events": list(self.events),
        }

    # ----------------------------------------------------------------- loop
    def _run(self) -> None:
        errors = 0
        try:
            SETTINGS.validate()
            client = AlpacaClient(settings=SETTINGS)
            settings = dataclasses.replace(SETTINGS, symbol=self.symbol, dry_run=False)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self._event("error", f"Could not start: {exc}")
            self.state = "error"
            return

        while not self._stop.is_set():
            wait = self.interval
            try:
                if not client.is_market_open():
                    if self.state != "waiting_for_market":
                        self._event("wait", "Market is closed. Waiting for the open.")
                    self.state = "waiting_for_market"
                    self.next_cycle_at = None
                    self._stop.wait(30)
                    continue
                self.state = "running"
                result = asdict(run_cycle(settings, client=client))
                errors = 0
                self.cycles += 1
                self.last_result = result
                sent = [o for o in result.get("planned_orders", []) if o.get("submitted")]
                self.orders_sent += len(sent)
                edge = result.get("composite_edge")
                self._event("cycle", f"Cycle {self.cycles}: {result.get('direction', 'flat')}"
                            + (f", edge {edge:+.2f}" if edge is not None else "") + f", {len(sent)} order(s) sent.")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._event("error", f"Cycle failed ({errors}/{MAX_CONSECUTIVE_ERRORS}): {self.last_error}")
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    self._event("error", "Stopped after 5 failed cycles in a row. Nothing was closed.")
                    self.state = "error"
                    return
                wait = 30
            self.next_cycle_at = dt.datetime.utcnow() + dt.timedelta(seconds=wait)
            self._stop.wait(wait)

        if self._flatten:
            self._close_everything(client)
        self.next_cycle_at = None
        self.state = "stopped"
        self._event("stop", "Stopped.")

    def _close_everything(self, client: AlpacaClient) -> None:
        sym = self.symbol
        try:
            cancelled = 0
            for o in client.get_orders("open"):
                if str(o.get("client_order_id", "")).startswith("volagent-") and any(underlying_of(s) == sym for s in _order_symbols(o)):
                    try:
                        client.cancel_order(o["id"])
                        cancelled += 1
                    except Exception as exc:  # noqa: BLE001
                        self._event("error", f"Could not cancel order {o.get('id')}: {exc}")
            positions = [p for p in client.get_positions() if underlying_of(p.get("symbol")) == sym]
            positions.sort(key=lambda p: float(p.get("qty", 0)))  # shorts first
            short_open, closed = False, 0
            for p in positions:
                qty, is_option = float(p.get("qty", 0)), p.get("asset_class") == "us_option"
                if is_option and qty > 0 and short_open:
                    self._event("error", f"Left {p['symbol']} open: a short leg did not close and this wing still covers it.")
                    continue
                try:
                    client.close_position(p["symbol"])
                    closed += 1
                except Exception as exc:  # noqa: BLE001
                    self._event("error", f"Could not close {p['symbol']}: {exc}")
                    if is_option and qty < 0:
                        short_open = True
            self._event("close", f"Cancelled {cancelled} open order(s), sent close for {closed} of {len(positions)} position(s).")
        except Exception as exc:  # noqa: BLE001
            self._event("error", f"Closing positions failed: {exc}")


SESSION = TradingSession()
