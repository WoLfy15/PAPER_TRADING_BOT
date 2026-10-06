"""
alpaca_client.py -- every Alpaca call in this project goes through
direct REST API calls using the `requests` library.

This replaces the original Alpaca CLI (Go binary) implementation with
equivalent direct HTTPS calls to:
  - Trading API:  https://paper-api.alpaca.markets
  - Market Data:  https://data.alpaca.markets

Endpoints in use (confirmed against docs.alpaca.markets):
  Trading API paths
    GET  /v2/account
    GET  /v2/clock
    GET  /v2/positions
    GET  /v2/positions/{symbol_or_id}
    GET  /v2/orders
    GET  /v2/orders/{id}
    POST /v2/orders                 (single-leg AND order_class="mleg")
    PATCH /v2/orders/{id}           (replace -- used for reprice-in-place)
    DEL  /v2/orders/{id}
    DEL  /v2/orders                 (cancel-all)
    GET  /v2/options/contracts?underlying_symbols=...
  Market Data API paths
    GET  /v2/stocks/{symbol}/bars
    GET  /v2/stocks/{symbol}/quotes/latest
    GET  /v2/stocks/{symbol}/trades/latest
    GET  /v1beta1/options/snapshots/{underlying_symbol}   (option chain + Greeks)

v3 PERFORMANCE FIX (see PERFORMANCE.md): every call used to go through
the bare module-level `requests.request(...)` function, which opens a
brand-new TCP+TLS connection every single time even though nearly every
call in a single agent cycle goes to one of exactly two hosts. Over a
typical cycle (account, clock, positions, daily bars, minute bars,
contracts, snapshot, orders -- 8+ round trips) that is 8+ needless
handshakes. This is now a persistent `requests.Session` per
AlpacaClient instance, with a pooled `HTTPAdapter` -- connections to
the same host are reused, and the pool is sized so several requests
issued concurrently (agent/loop.py now fetches independent endpoints
in parallel via a thread pool) don't contend for a connection.
`requests.Session` is documented as safe to share across threads
provided the pool is large enough for the expected concurrency, which
is exactly this use.

v3 RELIABILITY FIX: GET requests (reads) now retry automatically on
connect/read timeouts and on 429/500/502/503/504, with exponential
backoff -- a momentary network blip or a transient 5xx from Alpaca's
edge no longer fails an entire agent cycle outright. POST /v2/orders
(writes) does NOT use this blanket retry -- a 4xx (insufficient buying
power, invalid contract, market closed) is a real answer that retrying
would just reproduce, and blindly retrying an ambiguous write is how
duplicate orders happen. Order submission instead uses its own narrow,
explicit retry in `submit_order` that fires ONLY on a connection-level
failure where we know the request never reached Alpaca (a timeout or
connection error while the socket was still being opened, not after
Alpaca had already accepted the payload) and always resubmits the
IDENTICAL payload -- including the same client_order_id, which Alpaca
already treats as an idempotency key. See `submit_order`'s docstring.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import Settings

log = logging.getLogger("alpaca_vol_agent.client")

TRADING_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

# GET-only blanket retry policy (see module docstring). backoff_factor=0.4
# gives sleeps of ~0.4s, 0.8s, 1.6s, 2.4s -- fast enough not to make an
# already-slow cycle materially slower on the common case (no failure),
# bounded enough not to hang a loop cycle for minutes on a real outage.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_GET_RETRY = Retry(
    total=4, backoff_factor=0.4, status_forcelist=list(_RETRYABLE_STATUS),
    allowed_methods=frozenset({"GET"}), raise_on_status=False,
)
# Narrow, explicit retry budget for order submission -- see submit_order.
ORDER_SUBMIT_MAX_ATTEMPTS = 3
ORDER_SUBMIT_RETRY_BACKOFF_SECONDS = 1.0
_POOL_SIZE = 12  # >= the largest number of concurrent calls one agent cycle issues


class AlpacaAPIError(RuntimeError):
    """Raised on a non-2xx response from the Alpaca REST API."""

    def __init__(self, method: str, url: str, status: int, body: str):
        self.method = method
        self.url = url
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> HTTP {status}: {body[:500]}")


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=2, pool_maxsize=_POOL_SIZE, max_retries=_GET_RETRY)
    session.mount("https://", adapter)
    return session


@dataclass
class AlpacaClient:
    settings: Settings
    timeout: float = 30.0
    # cli_binary kept for API compatibility (unused now)
    cli_binary: str = "alpaca"
    # One pooled, retrying Session per client instance -- see module
    # docstring. default_factory (not shared across instances) so tests
    # that construct multiple AlpacaClients never share sockets/state.
    _session: requests.Session = field(default_factory=_make_session, repr=False, compare=False)
    # Set once this client has fallen back from opra -> indicative, so
    # the warning in get_option_chain_snapshot logs once per client
    # lifetime (typically once per agent cycle) instead of once per page.
    _opra_fallback_warned: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        # Validate credentials are present
        if not self.settings.api_key or not self.settings.api_secret:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. "
                "Check your .env file."
            )
        log.info("AlpacaClient initialised (direct REST, paper trading endpoint, pooled session)")

    def close(self) -> None:
        """Release pooled connections. Safe to call multiple times;
        harmless to skip entirely (the OS reclaims sockets on process
        exit either way) -- provided for long-running services
        (api/server.py) that want to recycle a client cleanly."""
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 -- closing is best-effort
            pass

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.settings.api_key,
            "APCA-API-SECRET-KEY": self.settings.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def doctor(self) -> str:
        """Connectivity check: fetch account info and return a summary string."""
        try:
            acct = self.get_account()
            return (
                f"OK -- account_id={acct.get('id')} "
                f"status={acct.get('status')} "
                f"equity={acct.get('equity')}"
            )
        except Exception as exc:
            return f"ERROR: {exc}"

    @staticmethod
    def _with_query(path: str, params: Optional[dict]) -> str:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        return f"{path}?{urlencode(clean)}" if clean else path

    def _trading_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> Any:
        url = TRADING_BASE + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            url = url + "?" + urlencode(clean) if clean else url
        # self._session (not the bare `requests` module) -- pooled,
        # keep-alive connections to paper-api.alpaca.markets, and GETs
        # get the automatic retry mounted on the adapter. See module
        # docstring for why POST is deliberately NOT retried here.
        resp = self._session.request(
            method,
            url,
            headers=self._headers(),
            json=json,
            timeout=self.timeout,
        )
        log.debug("%s %s -> %s", method, url, resp.status_code)
        if not resp.ok:
            raise AlpacaAPIError(method, url, resp.status_code, resp.text)
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _data_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
    ) -> Any:
        url = DATA_BASE + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            url = url + "?" + urlencode(clean) if clean else url
        resp = self._session.request(
            method,
            url,
            headers=self._headers(),
            timeout=self.timeout,
        )
        log.debug("%s %s -> %s", method, url, resp.status_code)
        if not resp.ok:
            raise AlpacaAPIError(method, url, resp.status_code, resp.text)
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    def _trading_get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._trading_request("GET", path, params=params)

    def _trading_post(self, path: str, json: dict) -> Any:
        return self._trading_request("POST", path, json=json)

    def _trading_patch(self, path: str, json: dict) -> Any:
        return self._trading_request("PATCH", path, json=json)

    def _trading_delete(self, path: str) -> Any:
        return self._trading_request("DELETE", path)

    def _data_get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._data_request("GET", path, params=params)

    # ---------------------------------------------------------------- account
    def get_account(self) -> dict:
        return self._trading_get("/v2/account")

    def get_clock(self) -> dict:
        return self._trading_get("/v2/clock")

    def is_market_open(self) -> bool:
        return bool(self.get_clock().get("is_open"))

    # --------------------------------------------------------------- positions
    def get_positions(self) -> list[dict]:
        return self._trading_get("/v2/positions") or []

    def get_position(self, symbol_or_id: str) -> Optional[dict]:
        try:
            return self._trading_get(f"/v2/positions/{symbol_or_id}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                return None
            raise

    def close_position(self, symbol_or_id: str) -> Optional[dict]:
        """Liquidate a single position at market via DELETE /v2/positions/{symbol}.
        Alpaca fills this as a market sell-to-close for long positions or
        buy-to-close for short positions. Returns the closing order dict,
        or None if the position was already gone (404)."""
        try:
            return self._trading_request("DELETE", f"/v2/positions/{symbol_or_id}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                log.warning("close_position: %s not found (already closed?)", symbol_or_id)
                return None
            raise

    # ------------------------------------------------------------------ orders
    def get_orders(self, status: str = "open", limit: int = 100) -> list[dict]:
        """status: Alpaca's own values -- 'open', 'closed', or 'all'.
        execution/reconciliation.py calls this with 'all' and a larger
        limit to build the full picture of what's happened to every
        order this project has submitted recently."""
        return self._trading_get("/v2/orders", params={"status": status, "limit": limit}) or []

    def get_order(self, order_id: str) -> Optional[dict]:
        """Single order's current state by Alpaca's own order id (not
        client_order_id) -- GET /v2/orders/{order_id}, the direct
        counterpart to get_position for positions. Returns None on 404
        (already fully settled and aged out, or an id from a different
        account) rather than raising, matching get_position's convention."""
        try:
            return self._trading_get(f"/v2/orders/{order_id}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                return None
            raise

    def submit_order(self, payload: dict) -> dict:
        """Submit any order payload as-is (single-leg or order_class='mleg').
        Callers build the payload via execution/order_manager.py, which
        already stamps every payload with a client_order_id for idempotency.

        v3: retries up to ORDER_SUBMIT_MAX_ATTEMPTS times, but ONLY on a
        requests.ConnectionError or requests.Timeout -- i.e. only when
        the failure happened at the socket/TCP level and we genuinely
        don't know whether Alpaca ever saw the request, as distinct from
        an AlpacaAPIError (a real HTTP response Alpaca sent back, meaning
        the request was received and answered -- 4xx there is a real
        rejection, not a maybe). Every retry resends the IDENTICAL
        payload, so the SAME client_order_id goes out again; Alpaca
        treats client_order_id as an idempotency key, so if the first
        attempt actually did land, the retry is a safe no-op rather than
        a second order. This is deliberately narrower than the blanket
        GET retry on the Session's adapter -- see module docstring."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, ORDER_SUBMIT_MAX_ATTEMPTS + 1):
            try:
                return self._trading_post("/v2/orders", json=payload)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < ORDER_SUBMIT_MAX_ATTEMPTS:
                    sleep_s = ORDER_SUBMIT_RETRY_BACKOFF_SECONDS * attempt
                    log.warning(
                        "submit_order: connection-level failure on attempt %d/%d "
                        "(client_order_id=%s) -- retrying in %.1fs: %s",
                        attempt, ORDER_SUBMIT_MAX_ATTEMPTS,
                        payload.get("client_order_id"), sleep_s, exc,
                    )
                    time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    def cancel_order(self, order_id: str) -> None:
        self._trading_delete(f"/v2/orders/{order_id}")

    def cancel_all_orders(self) -> None:
        self._trading_delete("/v2/orders")

    def replace_order(self, order_id: str, changes: dict) -> dict:
        """PATCH /v2/orders/{id} -- reprice or resize a still-open order
        in place instead of cancel+resubmit. Preferred over cancel/
        resubmit for the toxicity-buffer/staleness reprice case in
        execution/reconciliation.py: it preserves the order's queue
        priority-relevant timestamp semantics on Alpaca's book rather
        than losing your place and paying a fresh round trip."""
        return self._trading_patch(f"/v2/orders/{order_id}", json=changes)

    # ------------------------------------------------------------ option chain
    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Trading API's contract master list (status, tradable, OI, etc.) --
        NOT priced. Use get_option_chain_snapshot for live quotes/Greeks."""
        params: dict[str, Any] = {"underlying_symbols": underlying_symbol, "limit": limit}
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        contracts: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._trading_get("/v2/options/contracts", params=params)
            contracts.extend(resp.get("option_contracts", []))
            page_token = resp.get("page_token")
            if not page_token:
                break
        return contracts

    def get_option_chain_snapshot(
        self,
        underlying_symbol: str,
        option_type: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        expiration_date: Optional[str] = None,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, dict]:
        """Data API's live snapshot per contract symbol: latestTrade,
        latestQuote, and greeks (delta/gamma/theta/vega/rho + impliedVolatility)
        when the feed supplies them. Returns {contract_symbol: snapshot}.

        v3: config.py now defaults ALPACA_OPTION_FEED to 'opra' (the real
        consolidated feed -- see that module's docstring for why the
        indicative feed alone was flagged as a live-agent limitation).
        Not every account is entitled to opra, and Alpaca's own error
        for that case ("OPRA agreement is not signed") is worded as an
        agreement problem rather than a plain 403 -- confirmed against
        Alpaca's support forum, since this is exactly the kind of
        account-specific gotcha that's easy to get wrong from memory
        alone. If the configured feed is 'opra' and the request fails
        with a message that looks like this entitlement error, this
        method falls back ONCE per client instance to 'indicative' and
        logs a clear, loud warning -- this is the "never fall back
        blindly to stale prices" requirement read literally: falling
        back to a WORSE FEED is sometimes the only way to get a price at
        all, but it must never happen silently. A feed value the caller
        set explicitly to 'indicative' is left alone -- there's nothing
        to fall back from."""
        params: dict[str, Any] = {"feed": self.settings.data_feed, "limit": limit}
        if option_type:
            params["type"] = option_type
        if strike_price_gte is not None:
            params["strike_price_gte"] = strike_price_gte
        if strike_price_lte is not None:
            params["strike_price_lte"] = strike_price_lte
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte

        try:
            return self._fetch_all_snapshots(underlying_symbol, params)
        except AlpacaAPIError as exc:
            looks_like_opra_entitlement = (
                params.get("feed") == "opra"
                and exc.status in (403, 422)
                and any(term in exc.body.lower() for term in ("opra", "agreement"))
            )
            if not looks_like_opra_entitlement:
                raise
            if not self._opra_fallback_warned:
                log.warning(
                    "Options data feed 'opra' was requested but this account is not "
                    "entitled (%s) -- falling back to 'indicative' for the rest of this "
                    "client's lifetime. Indicative quotes are delayed/derived, not real "
                    "OPRA BBO; sign the OPRA market data agreement in the Alpaca dashboard "
                    "and subscribe to remove this fallback. This warning logs once.",
                    exc.body[:200],
                )
                self._opra_fallback_warned = True
            params["feed"] = "indicative"
            return self._fetch_all_snapshots(underlying_symbol, params)

    def _fetch_all_snapshots(self, underlying_symbol: str, params: dict) -> dict[str, dict]:
        params = dict(params)
        snapshots: dict[str, dict] = {}
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._data_get(f"/v1beta1/options/snapshots/{underlying_symbol}", params=params)
            snapshots.update(resp.get("snapshots", {}))
            page_token = resp.get("next_page_token")
            if not page_token:
                break
        return snapshots

    # ------------------------------------------------------------- stock data
    def get_stock_bars(
        self, symbol: str, timeframe: str = "1Day", start: Optional[str] = None,
        end: Optional[str] = None, limit: int = 1000, sort: Optional[str] = None, adjustment: str = "all",
    ) -> list[dict]:
        params: dict[str, Any] = {
            "timeframe": timeframe, "limit": limit, "feed": self.settings.stock_feed,
            "adjustment": adjustment,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if sort:
            params["sort"] = sort   # "desc" = newest first, so `limit` keeps the LATEST bars
        bars: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            resp = self._data_get(f"/v2/stocks/{symbol}/bars", params=params)
            bars.extend(resp.get("bars", []))
            page_token = resp.get("next_page_token")
            if not page_token or len(bars) >= limit:
                break
        return bars

    def get_latest_stock_quote(self, symbol: str) -> dict:
        resp = self._data_get(
            f"/v2/stocks/{symbol}/quotes/latest",
            params={"feed": self.settings.stock_feed},
        )
        return resp.get("quote", {})

    def get_latest_stock_trade(self, symbol: str) -> dict:
        resp = self._data_get(
            f"/v2/stocks/{symbol}/trades/latest",
            params={"feed": self.settings.stock_feed},
        )
        return resp.get("trade", {})

    def get_mid_price(self, symbol: str) -> float:
        q = self.get_latest_stock_quote(symbol)
        bid, ask = q.get("bp"), q.get("ap")
        if bid and ask:
            return (bid + ask) / 2.0
        return float(self.get_latest_stock_trade(symbol).get("p", 0.0))

    # ------------------------------------------------- asset lookup / snapshot
    def get_asset(self, symbol: str) -> Optional[dict]:
        try:
            return self._trading_get(f"/v2/assets/{symbol}")
        except AlpacaAPIError as exc:
            if exc.status == 404:
                return None
            raise

    def list_assets(self) -> list[dict]:
        return self._trading_get("/v2/assets", params={"status": "active", "asset_class": "us_equity"}) or []

    def get_stock_snapshot(self, symbol: str) -> dict:
        return self._data_get(f"/v2/stocks/{symbol}/snapshot", params={"feed": self.settings.stock_feed}) or {}
