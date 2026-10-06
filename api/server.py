"""
api/server.py -- FastAPI backend. This is the seam the UI is built
against: every route returns plain JSON (dataclasses via `asdict`),
CORS is wide open for local UI development, and nothing in here
duplicates logic that already lives in agent/loop.py or the strategy/
modules -- the routes are thin.

v3 additions, by the specific request each answers:

  - GET /orders: the LIVE, reconciled order blotter (execution/
    reconciliation.py), polled by the dashboard on its own fast cadence
    (a few seconds), deliberately SEPARATE from GET /decisions (the
    agent's reasoning log, which only gets a new entry once per full
    cycle -- 900s by default). This is the direct fix for "bookings not
    syncing with the logs": the two panels are explicitly different
    things now (live account state vs. reasoning history as of the last
    cycle), each labeled with its own "as of" timestamp, rather than
    implicitly expected to always agree. cancel_stale=False here --
    VIEWING the blotter must never have a side effect; only a running
    agent cycle (agent/loop.py) cancels anything.
  - GET /chart/bars: daily OHLC for the dashboard's candlestick chart.
  - GET /risk: portfolio Greek headroom, loss-circuit-breaker status,
    drawdown, and kill-switch state in one place, refreshed independent
    of a full agent cycle.
  - GET /calendar: upcoming FOMC/CPI/NFP dates (research/event_filter.py)
    for the dashboard's event banner.
  - POST /backtest: now runs backtest/engine.py's real walk-forward
    engine + backtest/walk_forward.py's evaluation report, not v2's
    synthetic-straddle backtest -- see those modules' docstrings for
    exactly what changed and what the results honestly do/don't mean.

v3 fixes:
  - /positions was still calling build_portfolio_snapshot with the OLD
    client-based signature; that function now takes pre-fetched
    account/positions dicts (agent/loop.py's concurrent-fetch speed fix
    made the old per-call signature redundant work) -- fixed here.
  - /run and /run-all used to hand-reconstruct a whole new Settings(...)
    object field by field to override just dry_run/symbol, which meant
    every new Settings field added since that code was written (which is
    most of them, as of v3) silently reverted to its default instead of
    inheriting from SETTINGS -- a real, easy-to-miss bug. Both now use
    dataclasses.replace(SETTINGS, ...), which carries over every field
    except the ones explicitly overridden, by construction.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from us_time import us_today
import threading
import time
from zoneinfo import ZoneInfo
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.session import PAUSE_SECONDS, SESSION, SessionBusy
from agent.loop import run_cycle
from agent.multi_loop import run_multi_cycle
from agent.state import compute_recent_loss_pcts, compute_risk_metrics, load_state
from alpaca_client import AlpacaClient
from config import SETTINGS, Settings
from data.market_data import atm_implied_vol, fetch_chain
from execution.reconciliation import reconcile
from research.event_filter import upcoming_events
from strategy.portfolio import (
    build_portfolio_snapshot, check_drawdown_kill_switch, check_loss_circuit_breakers, remaining_greek_headroom,
)

app = FastAPI(title="Alpaca Volatility Agent", version="0.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static"), name="static")


def _active_symbol(symbol: Optional[str] = None) -> str:
    """Explicit symbol > running session's symbol > SYMBOL from .env."""
    return (symbol or (SESSION.symbol if SESSION.active else None) or SETTINGS.symbol).upper()


_CLIENT: Optional[AlpacaClient] = None
_CLIENT_LOCK = threading.Lock()
_TTL: dict = {}


def _client() -> AlpacaClient:
    """ONE shared client (pooled keep-alive session). It used to build a new client -- a fresh TCP+TLS
    handshake -- on every single request, which is a big part of why the dashboard felt slow."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                SETTINGS.validate()
                _CLIENT = AlpacaClient(settings=SETTINGS)
    return _CLIENT


def _cached(key: str, ttl: float, fn):
    hit = _TTL.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _TTL[key] = (time.time(), val)
    return val


@app.on_event("startup")
def _warm():
    def go():
        for fn in (lambda: search("A"), lambda: _client().get_clock()):
            try:
                fn()
            except Exception:  # noqa: BLE001 -- warm-up only
                pass
    threading.Thread(target=go, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "time_utc": dt.datetime.utcnow().isoformat(), "dry_run": SETTINGS.dry_run, "symbol": SETTINGS.symbol}


@app.get("/status")
def status():
    client = _client()
    account = _cached("acct", 4, client.get_account)
    clock = _cached("clock", 8, client.get_clock)
    return {
        "account": {
            "equity": account.get("equity"), "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "options_trading_level": account.get("options_trading_level"),
            "options_approved_level": account.get("options_approved_level"),
        },
        "market_open": clock.get("is_open"),
        "next_open": clock.get("next_open"), "next_close": clock.get("next_close"),
        "dry_run": SETTINGS.dry_run, "symbol": SETTINGS.symbol,
        "data_feed": SETTINGS.data_feed,
    }


@app.get("/chain")
def chain(symbol: Optional[str] = None, dte: int = Query(default=None)):
    sym = symbol or SETTINGS.symbol
    client = _client()
    spot = _cached("mid:" + sym, 3, lambda: client.get_mid_price(sym))
    end = us_today()
    target_dte = dte if dte is not None else SETTINGS.target_dte_days
    contracts = fetch_chain(
        client, sym,
        expiration_date_gte=end.isoformat(),
        expiration_date_lte=(end + dt.timedelta(days=target_dte + SETTINGS.dte_tolerance_days + 10)).isoformat(),
        min_open_interest=SETTINGS.min_open_interest,
        max_spread_pct=SETTINGS.max_bid_ask_spread_pct, max_quote_age_seconds=SETTINGS.max_quote_age_seconds,
    )
    atm_iv = atm_implied_vol(contracts, spot, 0.045)
    return {
        "symbol": sym, "spot": spot, "atm_implied_vol": atm_iv,
        "contracts": [
            {
                "symbol": c.symbol, "strike": c.strike, "expiry": c.expiry.isoformat(),
                "type": c.option_type.value, "bid": c.bid, "ask": c.ask, "mid": c.mid,
                "open_interest": c.open_interest, "delta": c.alpaca_delta, "iv": c.alpaca_iv,
                "spread_pct": c.spread_pct,
            }
            for c in contracts
        ],
    }


TF_SECONDS = {"1Min": 60, "3Min": 180, "5Min": 300, "15Min": 900, "30Min": 1800, "1Hour": 3600, "3Hour": 10800,
              "4Hour": 14400, "1Day": 86400, "1Week": 604800, "1Month": 2592000}
NY = ZoneInfo("America/New_York")
MAX_START = "2015-06-01T00:00:00Z"


def _rth(bars):
    """(ET datetime, minutes since midnight ET, bar) for bars inside the 9:30-16:00 ET regular session."""
    out = []
    for b in bars:
        t = dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(NY)
        m = t.hour * 60 + t.minute
        if 570 <= m < 960:
            out.append((t, m, b))
    return out


def _agg_hours(bars, n):
    """30-min bars -> n-hour bars aligned to the 9:30 open (1H = 9:30, 10:30 ...; 3H = 9:30, 12:30; 4H = 9:30, 13:30)."""
    groups: dict = {}
    for t, m, b in _rth(bars):
        groups.setdefault((t.date(), (m - 570) // (60 * n)), []).append(b)
    return [{"t": g[0]["t"], "o": g[0]["o"], "h": max(x["h"] for x in g), "l": min(x["l"] for x in g),
             "c": g[-1]["c"], "v": sum(x.get("v") or 0 for x in g)} for _, g in sorted(groups.items())]


@app.get("/chart/bars")
def chart_bars(symbol: Optional[str] = None, timeframe: str = "1Day", count: int = Query(default=500, ge=1, le=10000),
               start: Optional[str] = None, end: Optional[str] = None):
    """Chart bars, oldest first. Intraday = US regular session only (9:30-16:00 ET), split-adjusted.
    start=<ISO>|max -> everything from there; otherwise the latest `count` bars before `end` (used to page older history)."""
    if timeframe not in TF_SECONDS:
        raise HTTPException(status_code=400, detail=f"timeframe must be one of {list(TF_SECONDS)}")
    sym = _active_symbol(symbol)
    hours = {"1Hour": 1, "3Hour": 3, "4Hour": 4}.get(timeframe)
    fetch_tf = "30Min" if hours else timeframe
    sec = TF_SECONDS[fetch_tf]
    need = min(10000, count * hours * 2 if hours else count)
    end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00")) if end else dt.datetime.now(dt.timezone.utc)
    client = _client()
    if start:
        raw = client.get_stock_bars(sym, timeframe=fetch_tf, start=MAX_START if start == "max" else start, end=end, limit=10000, adjustment="split")
    else:
        days = need / (23400 / sec) * 1.5 + 6 if sec < 86400 else need * {86400: 1.5, 604800: 7, 2592000: 31}[sec] + 10
        span_start = (end_dt - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = client.get_stock_bars(sym, timeframe=fetch_tf, start=span_start, end=end, limit=need, sort="desc", adjustment="split")[::-1]
    raw.sort(key=lambda b: b["t"])
    rows = _agg_hours(raw, hours) if hours else [b for _, _, b in _rth(raw)] if sec < 86400 else raw
    if not start:
        rows = rows[-count:]
    return {"symbol": sym, "timeframe": timeframe,
            "bars": [{"t": b.get("t"), "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"), "v": b.get("v")} for b in rows]}


@app.get("/quote/{symbol}")
def quote(symbol: str):
    """Lightweight last-trade tick for the forming candle (cached 1.5s server-side)."""
    sym = symbol.strip().upper()
    tr = _cached("q:" + sym, 1.5, lambda: _client().get_latest_stock_trade(sym)) or {}
    return {"symbol": sym, "price": tr.get("p"), "t": tr.get("t")}


_ASSETS: dict = {"at": 0.0, "rows": []}


@app.get("/search")
def search(q: str = Query(min_length=1, max_length=40)):
    """Ticker or company-name search over Alpaca's tradable US equities (cached 6h)."""
    if not _ASSETS["rows"] or time.time() - _ASSETS["at"] > 6 * 3600:
        rows = _client().list_assets()
        _ASSETS["rows"] = [{"symbol": a["symbol"], "name": a.get("name") or "", "exchange": a.get("exchange")} for a in rows if a.get("tradable")]
        _ASSETS["at"] = time.time()
    needle, upper = q.strip().lower(), q.strip().upper()
    scored = []
    for r in _ASSETS["rows"]:
        name = r["name"].lower()
        if r["symbol"] == upper:
            scored.append((0, r))
        elif r["symbol"].startswith(upper):
            scored.append((1, r))
        elif needle in name:
            scored.append((2 if name.startswith(needle) else 3, r))
    scored.sort(key=lambda x: (x[0], len(x[1]["symbol"])))
    return {"results": [r for _, r in scored[:8]]}


@app.get("/ticker/{symbol}")
def ticker(symbol: str):
    return _cached("ticker:" + symbol.strip().upper(), 5, lambda: _ticker(symbol))


def _ticker(symbol: str):
    """Everything the header strip shows for a symbol, and whether the agent can trade it
    (it trades volatility through options, so a symbol without listed options can't run)."""
    sym = symbol.strip().upper()
    client = _client()
    asset = _cached("asset:" + sym, 3600, lambda: client.get_asset(sym))
    if not asset:
        raise HTTPException(status_code=404, detail=f"{sym} is not a US-listed symbol on Alpaca.")
    snap = client.get_stock_snapshot(sym)
    trade, quote = snap.get("latestTrade") or {}, snap.get("latestQuote") or {}
    day, prev = snap.get("dailyBar") or {}, snap.get("prevDailyBar") or {}
    price = trade.get("p") or day.get("c")
    prev_close = prev.get("c")
    has_options = None
    try:
        res = _cached("opt:" + sym, 3600, lambda: client._trading_get("/v2/options/contracts", params={"underlying_symbols": sym, "limit": 1}))
        has_options = bool(res and res.get("option_contracts"))
    except Exception:  # noqa: BLE001 -- unknown (e.g. account lacks options access) rather than a false "no"
        has_options = None
    return {
        "symbol": sym, "name": asset.get("name"), "exchange": asset.get("exchange"),
        "tradable": bool(asset.get("tradable")), "shortable": bool(asset.get("shortable")),
        "price": price, "prev_close": prev_close,
        "change": (price - prev_close) if price and prev_close else None,
        "change_pct": ((price / prev_close - 1) * 100) if price and prev_close else None,
        "bid": quote.get("bp"), "ask": quote.get("ap"),
        "day_open": day.get("o"), "day_high": day.get("h"), "day_low": day.get("l"), "day_volume": day.get("v"),
        "has_options": has_options,
    }


class SessionStart(BaseModel):
    symbol: str


class SessionStop(BaseModel):
    flatten: bool = False


@app.get("/session")
def session_state():
    return SESSION.snapshot()


@app.post("/session/start")
def session_start(req: SessionStart):
    """Start real (paper-account) trading on one symbol. Refuses symbols the agent can't trade."""
    sym = req.symbol.strip().upper()
    info = ticker(sym)
    if not info["tradable"]:
        raise HTTPException(status_code=422, detail=f"{sym} is not tradable on Alpaca.")
    if info["has_options"] is False:
        raise HTTPException(status_code=422, detail=f"{sym} has no listed options, and this agent trades volatility through options.")
    try:
        return SESSION.start(sym, PAUSE_SECONDS)
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/session/stop")
def session_stop(req: SessionStop = SessionStop()):
    return SESSION.stop(flatten=req.flatten)


@app.get("/orders")
def orders():
    """v3: the LIVE reconciled order blotter -- see module docstring for
    why this is deliberately separate from /decisions. Read-only:
    cancel_stale=False always, regardless of what agent/loop.py itself
    does on its own cycle -- viewing this endpoint never cancels or
    changes anything."""
    client = _client()
    result = reconcile(client, SETTINGS, cancel_stale=False)
    return {
        "as_of_utc": result.as_of_utc,
        "open_orders": [asdict(o) for o in result.open_orders],
        "recently_filled": [asdict(o) for o in result.recently_filled],
        "open_notional_by_kind": result.open_notional_by_kind,
        "note": "Live account state as of this request -- independent of /decisions' once-per-cycle cadence.",
    }


@app.get("/risk")
def risk(symbol: Optional[str] = None):
    """v3: portfolio Greek headroom + loss-circuit-breaker status +
    drawdown + kill-switch, refreshed on every request independent of a
    full agent cycle -- the dashboard's risk panel polls this directly
    rather than waiting for the next /run."""
    client = _client()
    account = _cached("acct", 4, client.get_account)
    positions = _cached("pos", 4, client.get_positions)
    sym = _active_symbol(symbol)
    spot = _cached("mid:" + sym, 3, lambda: client.get_mid_price(sym))
    snap = build_portfolio_snapshot(account, positions, sym, spot, 0.045)
    equity = float(account["equity"])
    state = load_state()
    kill_switch, drawdown_pct = check_drawdown_kill_switch(equity, state.equity_high_water_mark, SETTINGS.risk)
    loss_pcts = compute_recent_loss_pcts(state)
    loss_tripped, loss_reasons = check_loss_circuit_breakers(loss_pcts["daily_pnl_pct"], loss_pcts["weekly_pnl_pct"], SETTINGS.risk)
    headroom = remaining_greek_headroom(snap, SETTINGS.risk)
    return {
        "equity": equity, "equity_high_water_mark": state.equity_high_water_mark,
        "drawdown_pct": drawdown_pct, "kill_switch": kill_switch, "max_drawdown_pct": SETTINGS.risk.max_drawdown_pct,
        "daily_pnl_pct": loss_pcts["daily_pnl_pct"], "weekly_pnl_pct": loss_pcts["weekly_pnl_pct"],
        "loss_breaker_tripped": loss_tripped, "loss_breaker_reasons": loss_reasons,
        "max_daily_loss_pct": SETTINGS.risk.max_daily_loss_pct, "max_weekly_loss_pct": SETTINGS.risk.max_weekly_loss_pct,
        "greeks": {
            "net_delta_shares": snap.net_delta_shares, "net_gamma_shares": snap.net_gamma_shares,
            "net_vega_dollars": snap.net_vega_dollars, "net_theta_dollars": snap.net_theta_dollars,
        },
        "greek_headroom": headroom,
        "greek_limits": {
            "max_delta_risk_pct": SETTINGS.risk.max_delta_risk_pct, "max_gamma_risk_pct": SETTINGS.risk.max_gamma_risk_pct,
            "max_vega_risk_pct": SETTINGS.risk.max_vega_risk_pct, "max_theta_risk_pct": SETTINGS.risk.max_theta_risk_pct,
        },
        "target_portfolio_vol_band": [SETTINGS.risk.target_portfolio_vol_min, SETTINGS.risk.target_portfolio_vol_max],
    }


@app.get("/calendar")
def calendar(horizon_days: int = 21):
    """v3: upcoming FOMC/CPI/NFP dates for the dashboard's event banner
    (research/data/calendar_events.py -- real, source-verified dates,
    see that module's docstring)."""
    return {"events": upcoming_events(us_today(), horizon_days=horizon_days)}


@app.get("/positions")
def positions(symbol: Optional[str] = None):
    client = _client()
    raw_positions = _cached("pos", 4, client.get_positions)
    sym = _active_symbol(symbol)
    all_rows = [{
        "symbol": p.get("symbol"), "asset_class": p.get("asset_class"), "qty": float(p.get("qty") or 0),
        "avg_entry": float(p.get("avg_entry_price") or 0), "price": float(p.get("current_price") or 0),
        "market_value": float(p.get("market_value") or 0), "unrealized_pl": float(p.get("unrealized_pl") or 0),
        "unrealized_plpc": float(p.get("unrealized_plpc") or 0), "delta_shares": None, "gamma_shares": None,
        "vega_dollars": None, "theta_dollars": None,
    } for p in raw_positions]
    for r in all_rows:   # shares: delta = qty, no other Greeks
        if r["asset_class"] != "us_option":
            r.update(delta_shares=r["qty"], gamma_shares=0.0, vega_dollars=0.0, theta_dollars=0.0)
    try:
        account = _cached("acct", 4, client.get_account)
        spot = _cached("mid:" + sym, 3, lambda: client.get_mid_price(sym))
        snap = build_portfolio_snapshot(account, raw_positions, sym, spot, 0.045)
    except Exception as exc:  # noqa: BLE001 -- positions must still show even if the Greeks can't be computed
        return {"all_positions": all_rows, "positions": [], "greeks_error": f"{type(exc).__name__}: {exc}"}
    greeks = {p.symbol: asdict(p) for p in snap.option_positions}
    for r in all_rows:
        g = greeks.get(r["symbol"])
        if g:
            r.update(delta_shares=g["delta_shares"], gamma_shares=g["gamma_shares"], vega_dollars=g["vega_dollars"], theta_dollars=g["theta_dollars"])
    return {
        "all_positions": all_rows,
        "equity": snap.equity, "cash": snap.cash, "equity_shares_held": snap.equity_shares_held,
        "net_delta_shares": snap.net_delta_shares, "net_gamma_shares": snap.net_gamma_shares,
        "net_vega_dollars": snap.net_vega_dollars, "net_theta_dollars": snap.net_theta_dollars,
        "positions": [asdict(p) for p in snap.option_positions],
    }


@app.get("/decisions")
def decisions(limit: int = 50):
    state = load_state()
    return {"equity_high_water_mark": state.equity_high_water_mark, "last_run_utc": state.last_run_utc,
            "decisions": state.decision_log[-limit:]}


@app.get("/metrics")
def metrics():
    """Live risk metrics (Sharpe/Sortino/max-drawdown/win-rate) computed
    from the account equity curve agent/loop.py persists every cycle
    (agent/state.record_equity + compute_risk_metrics). Returns Nones,
    not zeros, until enough cycles have run to say anything -- see
    compute_risk_metrics's own docstring."""
    state = load_state()
    risk_metrics = compute_risk_metrics(state)
    return {
        **risk_metrics,
        "equity_curve": state.equity_curve[-500:],   # capped for a reasonably-sized response
        "equity_high_water_mark": state.equity_high_water_mark,
    }


class BacktestRequest(BaseModel):
    symbol: Optional[str] = None
    years: float = 3.0
    starting_cash: float = 100_000.0
    roll_freq_days: int = 21
    use_bundled_reference_data: bool = False


@app.post("/backtest")
def backtest(req: BacktestRequest = BacktestRequest()):
    """v3: runs backtest/engine.py's real walk-forward engine (walk-
    forward regime + horizon-matched forecast refitting, the SAME
    strategy code the live agent runs, real cash accounting) plus
    backtest/walk_forward.py's evaluation report (fold stability,
    block-bootstrap Sharpe CI, stress scenarios). See both modules'
    docstrings for exactly what's real (the underlying price path and
    the IV level) and what's modeled (bid/ask spread -- there is no
    bundled historical options order book) about the result.

    use_bundled_reference_data=True runs against the S&P-500-derived
    price series already bundled in data/reference/ (useful for a quick
    demo with no network access); the default (False) downloads real
    history for `symbol` via yfinance, which needs network access from
    wherever this API server process is actually running."""
    from backtest.engine import BacktestConfig, run_backtest as _run_backtest
    from backtest.walk_forward import run_walk_forward_report

    sym = req.symbol or SETTINGS.symbol
    try:
        if req.use_bundled_reference_data:
            import pandas as pd
            sp = pd.read_csv("data/reference/sp500_fred.csv", parse_dates=["observation_date"]).set_index("observation_date")["SP500"].dropna()
            price_series = sp / 10.0  # SPX/10 ~= SPY-like level; see data/reference's own docstring in data/historical_iv.py
            cutoff = price_series.index.max() - pd.Timedelta(days=int(req.years * 365.25))
            price_series = price_series.loc[price_series.index >= cutoff]
        else:
            import yfinance as yf
            end = us_today()
            start = end - dt.timedelta(days=int(req.years * 365.25))
            df = yf.download(sym, start=start.isoformat(), end=end.isoformat(), progress=False)
            if df.empty:
                raise HTTPException(status_code=502, detail=f"yfinance returned no data for {sym} -- check network access and the symbol.")
            price_series = df["Close"].dropna()
            if hasattr(price_series, "squeeze"):
                price_series = price_series.squeeze()
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Backtest dependency not installed: {exc}") from exc

    cfg = BacktestConfig(symbol=sym, starting_cash=req.starting_cash, roll_freq_days=req.roll_freq_days)
    try:
        result = _run_backtest(price_series, cfg)
        report = run_walk_forward_report(result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "config": {"symbol": sym, "years": req.years, "starting_cash": req.starting_cash, "roll_freq_days": req.roll_freq_days,
                    "data_source": "bundled_reference" if req.use_bundled_reference_data else "yfinance"},
        "daily_nav": [{"date": str(d), "nav": float(v)} for d, v in result.daily_nav.items()],
        "roll_log": [
            {"date": str(r.date), "regime": r.regime_label, "composite_edge": r.composite_edge,
             "direction": r.direction, "contracts": r.contracts}
            for r in result.roll_log
        ],
        "full_sample": asdict(report.full_sample),
        "folds": [asdict(f) for f in report.folds],
        "median_fold_sharpe": report.median_fold_sharpe, "worst_fold_sharpe": report.worst_fold_sharpe,
        "bootstrap_ci": asdict(report.bootstrap_ci) if report.bootstrap_ci else None,
        "worst_seen_stress": report.worst_seen_stress,
        "acceptance": report.acceptance,
        "notes": report.notes,
    }


class RunRequest(BaseModel):
    dry_run: Optional[bool] = None
    symbol: Optional[str] = None


@app.post("/run")
def run(req: RunRequest = RunRequest()):
    """Runs exactly one agent cycle: fetch -> analyze -> decide ->
    (execute unless dry_run). Returns the full AgentRunResult as JSON --
    this is the single endpoint a UI needs to drive a "Run Agent"
    button. dataclasses.replace (v3) carries over every SETTINGS field
    except the ones explicitly overridden here -- see module docstring
    for the bug this fixes versus hand-reconstructing Settings(...)."""
    settings = dataclasses.replace(
        SETTINGS,
        symbol=req.symbol if req.symbol is not None else SETTINGS.symbol,
        dry_run=req.dry_run if req.dry_run is not None else SETTINGS.dry_run,
    )
    try:
        result = run_cycle(settings)
    except Exception as exc:  # noqa: BLE001 -- surface the real error to the UI rather than a bare 500
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return asdict(result)


class RunAllRequest(BaseModel):
    dry_run: Optional[bool] = None
    symbols: Optional[list[str]] = None  # override watchlist; null = use config


@app.post("/run-all")
def run_all(req: RunAllRequest = RunAllRequest()):
    """Run the agent across every symbol in the watchlist (or a caller-
    supplied subset). Returns per-symbol results + an aggregate summary.
    Each symbol that fails (illiquid chain, not enough bars, etc.) is
    logged as an error without killing the scan for the rest."""
    settings = dataclasses.replace(SETTINGS, dry_run=req.dry_run if req.dry_run is not None else SETTINGS.dry_run)
    try:
        multi = run_multi_cycle(settings, symbols=req.symbols)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return asdict(multi)


@app.get("/watchlist")
def watchlist():
    return {"watchlist": SETTINGS.watchlist}


@app.get("/dashboard")
def dashboard():
    """Serves dashboard.html directly from this API's own origin, so
    opening it needs nothing more than this server running --
    http://localhost:8000/dashboard -- with no separate static-file
    server and no CORS considerations (same origin as every /api call
    the page itself makes). Opening dashboard.html straight from disk
    (file://) also still works: CORS is wide open (see this module's
    top) specifically to support that path too."""
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found next to this project's root.")
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/")
def root():
    return {
        "service": "alpaca-vol-agent", "docs": "/docs", "dashboard": "/dashboard",
        "endpoints": [
            "/health", "/status", "/chain", "/chart/bars", "/search", "/quote/{symbol}", "/ticker/{symbol}", "/session", "/orders", "/risk", "/calendar",
            "/positions", "/decisions", "/metrics", "/watchlist", "/dashboard",
            "POST /run", "POST /run-all", "POST /backtest", "POST /session/start", "POST /session/stop",
        ],
    }
