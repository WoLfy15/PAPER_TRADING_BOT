"""
agent/loop.py -- the orchestration layer. `run_cycle()` is the one
function that ties every other module together into a single pass:

  fetch (Alpaca account, chain, bars, minute bars, open orders)
    -> analyze (regime, vol forecast, SAS, Heston cross-check, flow
       toxicity, event/jump filter, composite signal)
    -> risk-check (drawdown kill switch, daily/weekly loss breakers,
       portfolio Greeks + hard caps, existing exposure incl. still-open
       orders, vol-of-vol + toxicity + event throttles, cost floor)
    -> decide (Kelly sizing -> concrete legs (CVaR-aware condor) ->
       hedge trade with a REAL cross-cycle hedge ratio)
    -> explain (human-readable rationale for every decision)
    -> execute (only if not dry_run) -> reconcile

It returns a single AgentRunResult dataclass with everything the FastAPI
layer (api/server.py) needs to render a UI: the raw signal numbers, the
regime, every planned order (submitted or not), and the hedge decision.
Nothing here talks to the network except through alpaca_client.py, and
nothing submits an order except through execution/order_manager.py.

v3 changes (see CHANGES_V3.md for the full list against the original
21-point spec; this docstring covers what's specific to THIS file):

  - CONCURRENT FETCH PHASE: v2 issued account/mid-price/daily-bars/
    chain/positions/minute-bars/open-orders as 7+ SEQUENTIAL blocking
    HTTP calls -- each one waiting on the last even though none of them
    depend on each other's results. alpaca_client.py's Session is now
    pooled and thread-safe (see that module's docstring); this file
    fires all of them at once via a ThreadPoolExecutor and only then
    proceeds to the genuinely sequential CPU work (regime fit, GARCH
    fit, SAS/Heston calibration, which all need the fetched data) and
    the dependent fetches (chain -> expiry selection -> ATM IV). This
    is the direct fix for cycles "taking time."
  - RECONCILE BEFORE SIZING (execution/reconciliation.py): every cycle
    now reconciles Alpaca's actual open-order state before sizing any
    new entry, folds still-OPEN order notional into the incremental-
    sizing gap (not just settled positions), and cancels anything
    that's been resting past settings.stale_order_max_age_minutes. This
    is what actually prevents a slow-to-fill order from silently
    getting duplicated next cycle -- requirement 15 -- and it's also
    the fix for "bookings not syncing with the logs": the decision log
    only updates once per full cycle, but api/server.py's new GET
    /orders calls this same reconcile() fresh on every dashboard poll.
  - HORIZON-MATCHED FORECAST (research/vol_forecast.forecast_horizon):
    the composite signal's forecast_vol_now now comes from a genuine
    multi-step-ahead forecast matched to the SELECTED expiry's actual
    DTE, not vol_fc.blended_next's in-sample last point (requirement 1).
  - REAL HEDGE RATIO: min_variance_hedge_ratio is now called with the
    REAL cross-cycle option-book/underlying return series
    agent.state.hedge_ratio_inputs persists, not empty lists -- see
    that function's docstring (requirement 16). Previously dead code.
  - EVENT/JUMP FILTER (research/event_filter.py): FOMC/CPI/NFP blackout
    + same-day jump detection, gating/throttling short-vol entries only
    (requirement 8).
  - LOSS CIRCUIT BREAKERS + GREEK HARD CAPS (strategy/portfolio.py):
    daily/weekly loss and an already-breached delta/theta cap now force
    ALL new entries flat, alongside the existing drawdown kill switch
    (requirement 10).
  - CVaR-AWARE CONDOR (strategy.legs.select_iron_condor_by_cvar):
    replaces the plain delta-only build_iron_condor call on the
    short-vol path (requirement 11).
"""
from __future__ import annotations

import datetime as dt
from us_time import in_entry_window, us_now, us_today
import logging
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from agent.exits import ExitDecision, evaluate_exits
from agent.state import (
    compute_recent_loss_pcts, compute_risk_metrics, hedge_ratio_inputs, load_state,
    record_book_snapshot, record_decision, record_equity, save_state,
)
from alpaca_client import AlpacaClient
from pricing_engine.calibration.iv_solver import IVSolverError
from pricing_engine.risk.friction_hedging import leland_cost_bps_of_vega
from config import Settings
from data.market_data import (
    atm_implied_vol, daily_log_returns_from_bars, fetch_chain, implied_vol_for, to_option_quote,
)
from execution.order_manager import plan_equity_hedge, plan_iron_condor, plan_position_close, plan_straddle_buy, submit_close_position, submit_orders
from execution.reconciliation import ReconciliationResult, reconcile
from pricing_engine.market_data.contract import OptionType
from pricing_engine.risk.sas import strike_adjusted_spread
from research.event_filter import EventRisk, assess_event_risk
from research.heston_signal import HestonCrossCheck, assess_heston_cross_check
from research.regime import detect_regime
from research.signal import build_signal
from research.edge import allocation as edge_alloc
from research.edge.forecast import latest_forecast
from strategy.legs import condor_structure_legs
from research.toxicity import FlowToxicity, assess_flow_toxicity
from research.vol_forecast import HorizonForecast, estimate_hurst, forecast_horizon, forecast_volatility
from strategy.hedging import decide_hedge, min_variance_hedge_ratio
from strategy.legs import build_straddle, select_expiry, select_iron_condor_by_cvar
from strategy.portfolio import (
    build_portfolio_snapshot, check_drawdown_kill_switch, check_loss_circuit_breakers,
    existing_vol_exposure_notional, remaining_greek_headroom, fit_contracts_to_greek_caps, realized_book_vol, vol_target_scale,
)
from strategy.sizing import REGIME_SIZE_SCALE, decide_sizing, kelly_fraction as _compute_kelly

log = logging.getLogger("alpaca_vol_agent.loop")

RISK_FREE_RATE_FALLBACK = 0.045
BAR_HISTORY_DAYS = 400
TOXICITY_LOOKBACK_DAYS = 7
STRADDLE_BASE_LIMIT_BUFFER_BPS = 200.0   # 2% above mid -- see execution/order_manager.py's docstring
WING_DELTA_CANDIDATE_MULTIPLIERS = (0.6, 1.0, 1.6)  # narrower/configured/wider -- see select_iron_condor_by_cvar


@dataclass
class AgentRunResult:
    timestamp_utc: str
    symbol: str
    spot: float
    regime_label: str
    regime_confidence: float
    gamma_regime: str
    forecast_vol: float
    live_atm_iv: Optional[float]
    vrp_raw: Optional[float]
    vrp_z: Optional[float]
    sas_mean: Optional[float]
    composite_edge: Optional[float]
    signal_rationale: str
    kelly_fraction: Optional[float]
    direction: str
    portfolio_delta_shares: float
    portfolio_gamma_shares: float
    portfolio_vega_dollars: float
    portfolio_theta_dollars: float
    equity: float
    drawdown_pct: float
    kill_switch: bool
    planned_orders: list[dict[str, Any]] = field(default_factory=list)
    hedge_decision: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    # -- v2 additions --
    heston_cross_check: Optional[dict[str, Any]] = None
    flow_toxicity: Optional[dict[str, Any]] = None
    vol_path_hurst: Optional[float] = None
    deployed_long_vol_notional: float = 0.0
    deployed_short_vol_risk: float = 0.0
    extra_size_multiplier: float = 1.0
    incremental_budget_dollars: float = 0.0
    already_at_target: bool = False
    risk_metrics: dict[str, Optional[float]] = field(default_factory=dict)
    # -- sizing detail (what the agent WOULD deploy; non-zero even with no open positions) --
    risk_budget_dollars: float = 0.0           # |kelly| * capital before Greek caps
    contracts_planned: int = 0                 # straddle contracts that would be bought (long_vol)
    max_structure_loss: float = 0.0            # max risk budget for short-vol structure
    capped_by: list[str] = field(default_factory=list)  # which risk caps actually fired
    regime_size_scale: float = 1.0             # REGIME_SIZE_SCALE applied this cycle
    # -- auto-exit decisions (v2) --
    exit_decisions: list[dict[str, Any]] = field(default_factory=list)
    # -- v3 additions --
    horizon_forecast: Optional[dict[str, Any]] = None     # requirement 1/6: genuine DTE-matched forecast + CI
    event_risk: Optional[dict[str, Any]] = None            # requirement 8
    daily_pnl_pct: Optional[float] = None
    weekly_pnl_pct: Optional[float] = None
    loss_breaker_tripped: bool = False
    loss_breaker_reasons: list[str] = field(default_factory=list)
    greek_headroom: dict[str, float] = field(default_factory=dict)   # requirement 10
    hedge_ratio_used: Optional[float] = None                          # requirement 16 -- the REAL value now, not always 1.0
    reconciliation: Optional[dict[str, Any]] = None                   # requirement 15
    condor_stress_report: Optional[dict[str, Any]] = None             # requirement 11/19
    historical_iv_source: Optional[str] = None                        # requirement 2 -- transparency on what backed the VRP z-score


def _bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    df = df.rename(columns={"c": "close"})
    return df


def _concurrent_fetch(client: AlpacaClient, settings: Settings, end: dt.date, dte_search_end: dt.date) -> dict[str, Any]:
    """v3 speed fix -- see module docstring. Every one of these is
    independent of every other's RESULT (fetching the chain doesn't
    need spot; fetching bars doesn't need the chain; etc.), so they run
    on a shared thread pool against alpaca_client's pooled, thread-safe
    Session instead of one after another. Returns a dict keyed by name
    so the caller's subsequent code reads exactly as it did when these
    were sequential local variables -- only the fetch itself changed."""
    start = end - dt.timedelta(days=BAR_HISTORY_DAYS)
    tox_start = end - dt.timedelta(days=TOXICITY_LOOKBACK_DAYS)
    jobs = {
        "account": lambda: client.get_account(),
        "spot": lambda: client.get_mid_price(settings.symbol),
        "daily_bars": lambda: client.get_stock_bars(settings.symbol, timeframe="1Day", start=start.isoformat(), end=end.isoformat()),
        "minute_bars": lambda: client.get_stock_bars(settings.symbol, timeframe="1Min", start=tox_start.isoformat(), end=end.isoformat(), limit=3000) if settings.enable_toxicity_gate else [],
        "chain": lambda: fetch_chain(
            client, settings.symbol, expiration_date_gte=end.isoformat(), expiration_date_lte=dte_search_end.isoformat(),
            min_open_interest=settings.min_open_interest, max_spread_pct=settings.max_bid_ask_spread_pct,
            max_quote_age_seconds=settings.max_quote_age_seconds,
        ),
        "positions": lambda: client.get_positions(),
        "reconciliation": lambda: reconcile(client, settings, cancel_stale=settings.enable_order_reconciliation) if settings.enable_order_reconciliation else None,
    }
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            results[name] = fut.result()
    return results


def _execute_exits(client: AlpacaClient, settings: Settings, decisions: list[ExitDecision]) -> tuple[list[dict[str, Any]], list[str]]:
    """Submit the close orders agent/exits.evaluate_exits decided on.

    v3: exits are now emitted per STRUCTURE (every leg of a condor or
    straddle together, short legs first -- see agent/exits.py). Two
    consequences handled here that the old flat one-try/except loop got
    wrong:

      1. FAILURE ISOLATION. One failing close no longer aborts every
         other exit that cycle -- each leg gets its own try/except, and a
         failure in one structure never blocks another structure's exits.
      2. NEVER LEAVE A SHORT NAKED. Within a structure, if a SHORT leg
         does not actually get closed (the call raised, or it was
         skipped -- e.g. Alpaca's "no available quote" -- which leaves
         the short OPEN), the remaining protective LONG legs of that same
         structure are deliberately NOT closed: closing the wings while a
         short is still open would leave that short uncovered, which is
         the exact hazard the wings exist to prevent. Shorts are ordered
         first (agent/exits._decide), so this check always runs before any
         wing close is attempted. In dry_run nothing is ever "submitted",
         and that is normal rather than a failure, so it never blocks.
    """
    planned: list[dict[str, Any]] = []
    notes: list[str] = []
    blocked_groups: set[str] = set()
    for ed in decisions:
        tag = ed.reason.replace("_", " ").upper()
        desc = f"[{tag}] CLOSE {ed.qty:g}x {ed.symbol} (P&L {ed.pnl_pct:+.1%}, {ed.dte} DTE)"

        if ed.qty > 0 and ed.group and ed.group in blocked_groups:
            msg = (
                f"Auto-exit [{tag}]: SKIPPED closing protective long {ed.symbol} -- a short leg of the same "
                f"structure ({ed.group}) did not close, and closing the wing alone would leave that short naked."
            )
            notes.append(msg)
            planned.append({"kind": f"exit_{ed.reason}", "description": desc, "submitted": False,
                            "skipped": True, "skip_reason": "short_leg_not_closed"})
            continue

        try:
            # DELETE /v2/positions/{symbol} -- Alpaca's preferred close method; avoids the
            # HTTP 403 "no available quote" a plain market order hits on an option with no live bid.
            result_dict = submit_close_position(client, ed.symbol, settings, ed.reason)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Auto-exit [{tag}] FAILED for {ed.symbol}: {type(exc).__name__}: {exc}")
            log.warning("Auto-exit close failed for %s: %s", ed.symbol, exc, exc_info=True)
            planned.append({"kind": f"exit_{ed.reason}", "description": desc, "submitted": False, "error": str(exc)})
            if ed.qty < 0 and ed.group:
                blocked_groups.add(ed.group)
            continue

        if ed.qty < 0 and ed.group and not settings.dry_run and not result_dict.get("submitted"):
            blocked_groups.add(ed.group)  # e.g. skipped for no quote: the short is still open
        planned.append({"kind": f"exit_{ed.reason}", "description": desc, **result_dict})
        notes.append(
            f"Auto-exit [{tag}]: {ed.symbol} qty={ed.qty:g} P&L={ed.pnl_pct:+.1%} DTE={ed.dte} "
            f"({'SUBMITTED' if not settings.dry_run else 'DRY RUN'})"
        )
    return planned, notes


def _minutes_since(iso: str) -> float:
    try:
        return (us_now() - dt.datetime.fromisoformat(iso)).total_seconds() / 60.0
    except Exception:  # noqa: BLE001 -- empty / unparsable = "long ago"
        return 1e9


def _entry_block_reason(state, settings: Settings) -> str:
    """Pacing gate for NEW entries only (exits and hedge flattening are never blocked)."""
    if not in_entry_window(settings.avoid_open_close_minutes):
        return f"outside the entry window (no entries in the first/last {settings.avoid_open_close_minutes} min of the US session or when closed)"
    sym = settings.symbol
    if _minutes_since(state.last_entry_utc.get(sym, "")) < settings.entry_cooldown_minutes:
        return f"cooldown: {sym} was entered less than {settings.entry_cooldown_minutes} min ago"
    wait_h = settings.after_stop_cooldown_hours if state.last_exit_reason.get(sym) in ("stop_loss", "trailing_stop") else 0.5
    if _minutes_since(state.last_exit_utc.get(sym, "")) < wait_h * 60:
        return f"cooldown: {sym} was closed ({state.last_exit_reason.get(sym)}) less than {wait_h:g} h ago -- not re-entering straight away"
    return ""


_EDGE_BARS: dict = {}


def _edge_history_bars(client, symbol: str, end: dt.date) -> "pd.Series":
    """~6 years of split-adjusted daily log returns for the forecast engine, fetched once per day."""
    key = (symbol, str(end))
    if key not in _EDGE_BARS:
        _EDGE_BARS.clear()
        rows = client.get_stock_bars(symbol, timeframe="1Day", start=(end - dt.timedelta(days=6 * 366)).isoformat(), end=end.isoformat(), limit=10000, adjustment="split")
        px = pd.Series([r["c"] for r in rows], index=pd.to_datetime([r["t"] for r in rows]).tz_localize(None).normalize()).dropna()
        _EDGE_BARS[key] = np.log(px / px.shift(1)).dropna()
    return _EDGE_BARS[key]


def _atm_spread_pct(chain, expiry, spot: float) -> float:
    cs = sorted((c for c in chain if c.expiry == expiry and c.bid and c.ask and c.ask > 0), key=lambda c: abs(c.strike - spot))[:6]
    sp = [(c.ask - c.bid) / ((c.ask + c.bid) / 2) for c in cs if c.ask + c.bid > 0]
    return float(sum(sp) / len(sp)) if sp else float("nan")


def run_cycle(settings: Settings, client: Optional[AlpacaClient] = None) -> AgentRunResult:
    settings.validate()
    client = client or AlpacaClient(settings=settings)
    state = load_state()
    notes: list[str] = []

    end = us_today()
    dte_search_end = end + dt.timedelta(days=settings.target_dte_days + settings.dte_tolerance_days + 5)
    fetched = _concurrent_fetch(client, settings, end, dte_search_end)

    account = fetched["account"]
    spot = fetched["spot"]
    bars = fetched["daily_bars"]
    chain = fetched["chain"]
    raw_positions = fetched["positions"]
    reconciliation: Optional[ReconciliationResult] = fetched["reconciliation"]

    equity = float(account["equity"])
    if state.equity_high_water_mark <= 0:
        state.equity_high_water_mark = equity
    state.equity_high_water_mark = max(state.equity_high_water_mark, equity)
    kill_switch, drawdown_pct = check_drawdown_kill_switch(equity, state.equity_high_water_mark, settings.risk)
    if kill_switch:
        notes.append(
            f"DRAWDOWN KILL SWITCH TRIPPED: {drawdown_pct:.1%} below high-water mark "
            f"(limit {settings.risk.max_drawdown_pct:.0%}). No new entries this cycle."
        )

    loss_pcts = compute_recent_loss_pcts(state)
    loss_tripped, loss_reasons = check_loss_circuit_breakers(loss_pcts["daily_pnl_pct"], loss_pcts["weekly_pnl_pct"], settings.risk)
    if loss_tripped:
        notes.extend(f"LOSS CIRCUIT BREAKER: {r}. No new entries this cycle." for r in loss_reasons)

    if reconciliation is not None and reconciliation.actions_taken:
        for a in reconciliation.actions_taken:
            notes.append(f"Reconciliation: cancelled stale {a['kind']} order {a['order_id']} ({a['age_seconds']:.0f}s old).")

    df = _bars_to_frame(bars)
    if len(df) < 120:
        raise RuntimeError(
            f"only {len(df)} daily bars returned for {settings.symbol} over the last "
            f"{BAR_HISTORY_DAYS} days -- not enough history for the HMM/GARCH fit. "
            "Try a more liquid symbol or widen BAR_HISTORY_DAYS."
        )

    regime = detect_regime(df["close"])
    feat_log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    realized_vol = feat_log_returns.rolling(21).std() * np.sqrt(252)
    realized_vol = realized_vol.dropna()
    aligned_returns = feat_log_returns.loc[realized_vol.index]
    vol_fc = forecast_volatility(aligned_returns, realized_vol)

    # Diagnostic only -- see research/vol_forecast.estimate_hurst's
    # docstring for why this deliberately does NOT feed the forecast
    # blend or sizing. Surfaced in the UI as context, nothing more.
    vol_path_hurst: Optional[float] = None
    with np.errstate(divide="ignore"):
        log_rv = np.log(realized_vol.replace(0, np.nan)).dropna()
    if len(log_rv) >= 40:
        vol_path_hurst = estimate_hurst(log_rv.values)

    available_expiries = sorted({c.expiry for c in chain})
    log.info(
        "Chain fetch: %d contracts, expiries=%s (searched %s to %s, target_dte=%d, tol=%d)",
        len(chain), [str(e) for e in available_expiries],
        end.isoformat(), dte_search_end.isoformat(),
        settings.target_dte_days, settings.dte_tolerance_days,
    )
    expiry = select_expiry(chain, settings.target_dte_days, settings.dte_tolerance_days, as_of=end)
    if expiry is None:
        if not available_expiries:
            raise RuntimeError(
                f"Chain returned 0 contracts for {settings.symbol} between {end} and {dte_search_end}. "
                "Check that the paper account has options data access and that MIN_OPEN_INTEREST "
                "is not filtering everything out."
            )
        # Fallback: use the nearest available expiry and warn. This is common on
        # the indicative feed where only near-term contracts have live snapshots.
        expiry = min(available_expiries, key=lambda e: abs((e - end).days - settings.target_dte_days))
        actual_dte = (expiry - end).days
        notes.append(
            f"DTE fallback: no expiry within {settings.dte_tolerance_days} days of "
            f"{settings.target_dte_days}-day target. Using nearest available: "
            f"{expiry} ({actual_dte} DTE). Available: {[str(e) for e in available_expiries]}."
        )
        log.warning("DTE fallback: using %s (%d DTE) -- indicative feed may only have near-term snapshots.", expiry, actual_dte)

    actual_dte_days = max((expiry - end).days, 1)
    risk_free_rate = RISK_FREE_RATE_FALLBACK
    live_iv = atm_implied_vol([c for c in chain if c.expiry == expiry], spot, risk_free_rate)

    # v3, requirement 1/6: genuine horizon-matched forecast, DTE-matched to
    # the expiry actually selected above -- replaces vol_fc.blended_next
    # (an in-sample point, not a forward forecast) as forecast_vol_now
    # everywhere below. vol_fc itself is kept as a cheap diagnostic (UI
    # context, Heston's own internal reference) since it's already computed.
    horizon_fc: Optional[HorizonForecast] = None
    forecast_vol_now = vol_fc.blended_next
    try:
        horizon_fc = forecast_horizon(aligned_returns, realized_vol, horizon_days=actual_dte_days)
        forecast_vol_now = horizon_fc.blended_expected_vol
    except Exception as exc:  # noqa: BLE001 -- a bad GARCH fit cycle shouldn't kill the whole run; fall back to the diagnostic point
        notes.append(f"Horizon-matched forecast failed ({type(exc).__name__}: {exc}) -- using in-sample point forecast instead.")

    edge_fc: Optional[dict] = None
    try:
        edge_fc = latest_forecast(_edge_history_bars(client, settings.symbol, end), actual_dte_days)
        if edge_fc and edge_fc.get("selected") and edge_fc["selected"] > 0:
            forecast_vol_now = edge_fc["selected"]
            notes.append(f"Forecast engine: research.edge ({edge_fc['chosen']} in the {edge_fc['regime']} regime), {actual_dte_days}-day vol {forecast_vol_now:.1%}.")
    except Exception as exc:  # noqa: BLE001 -- keep the previous horizon forecast if the engine can't run this cycle
        edge_fc = None
        notes.append(f"research.edge forecast unavailable ({type(exc).__name__}: {exc}) -- using the previous horizon forecast.")

    sas_values: list[float] = []
    daily_returns = daily_log_returns_from_bars(bars)
    if live_iv and len(daily_returns) > 30:
        same_expiry = [c for c in chain if c.expiry == expiry]
        near_strikes = sorted({c.strike for c in same_expiry}, key=lambda k: abs(k - spot))[:5]
        for c in same_expiry:
            if c.strike not in near_strikes or c.option_type != OptionType.CALL:
                continue
            quote = to_option_quote(c, spot, risk_free_rate)
            iv = implied_vol_for(quote)
            if iv is None:
                continue
            try:
                sas = strike_adjusted_spread(quote, iv, daily_returns)
                sas_values.append(sas)
            except (ValueError, ZeroDivisionError, IVSolverError, Exception):
                # 0-DTE or deep-ITM quotes often produce stale/crossed prices
                # that can't be bracketed by the IV solver -- skip silently.
                continue
    else:
        notes.append("SAS leg skipped: insufficient live IV or return history this cycle.")

    if live_iv is None:
        notes.append(f"No live ATM IV available for {settings.symbol} {expiry}; signal falls back to VRP-only.")
        live_iv = float(realized_vol.iloc[-1]) * 1.15  # last-resort proxy, clearly flagged

    # --- Heston smile-consistency cross-check + vol-of-vol throttle ---
    heston: Optional[HestonCrossCheck] = None
    if settings.enable_heston_cross_check:
        try:
            heston = assess_heston_cross_check(
                chain, expiry, spot, risk_free_rate,
                live_atm_iv=live_iv, forecast_vol_now=forecast_vol_now,
            )
            if not heston.available:
                notes.extend(f"Heston: {n}" for n in heston.notes)
            elif heston.notes:
                notes.extend(f"Heston: {n}" for n in heston.notes)
        except Exception as exc:  # noqa: BLE001 -- a bad calibration cycle shouldn't kill the whole run
            notes.append(f"Heston cross-check failed unexpectedly: {type(exc).__name__}: {exc}")
            heston = None

    # --- flow-toxicity read off recent minute bars (already fetched concurrently above) ---
    toxicity: Optional[FlowToxicity] = None
    if settings.enable_toxicity_gate:
        try:
            toxicity = assess_flow_toxicity(fetched["minute_bars"])
            if not toxicity.available:
                notes.extend(f"Flow toxicity: {n}" for n in toxicity.notes)
            elif toxicity.regime in ("Elevated", "Toxic"):
                notes.append(
                    f"Flow toxicity: {toxicity.regime} (VPIN percentile {toxicity.percentile_vs_history:.0%} "
                    f"of trailing history) -- sizing throttled x{toxicity.size_multiplier:.2f}, "
                    f"execution limits widened +{toxicity.extra_limit_buffer_bps:.0f}bps."
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Flow-toxicity read failed unexpectedly: {type(exc).__name__}: {exc}")
            toxicity = None

    # --- event/jump filter (requirement 8) -- short-vol only, see that module's docstring ---
    event_risk: Optional[EventRisk] = None
    if settings.enable_event_filter:
        try:
            event_risk = assess_event_risk(
                end, fomc_blackout_days_before=settings.fomc_blackout_days_before,
                fomc_blackout_days_after=settings.fomc_blackout_days_after,
                cpi_nfp_blackout_days_before=settings.cpi_nfp_blackout_days_before,
                cpi_nfp_blackout_days_after=settings.cpi_nfp_blackout_days_after,
                daily_log_returns=feat_log_returns.values,
            )
            if event_risk.reasons:
                notes.extend(f"Event filter: {r}" for r in event_risk.reasons)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Event filter failed unexpectedly: {type(exc).__name__}: {exc}")
            event_risk = None

    _hist = None
    if edge_fc and edge_fc.get("history") is not None:
        _h = edge_fc["history"].dropna()
        _ix = _h.index.intersection(realized_vol.index)
        if len(_ix) >= 30:
            _hist = (_h.loc[_ix], realized_vol.loc[_ix])
    signal = build_signal(
        symbol=settings.symbol,
        regime_label=regime.current_label,
        forecast_vol_now=forecast_vol_now,
        live_atm_iv=live_iv,
        forecast_vol_history=_hist[0] if _hist else vol_fc.blended,
        realized_vol_history=_hist[1] if _hist else realized_vol.loc[vol_fc.blended.index],
        sas_values=sas_values or None,
        heston_term_gap=heston.term_gap if (heston and heston.available) else None,
        heston_confident=bool(heston.data_quality_ok) if heston else True,
    )

    portfolio = build_portfolio_snapshot(account, raw_positions, settings.symbol, spot, risk_free_rate)
    record_book_snapshot(
        state, dt.datetime.now(dt.timezone.utc).isoformat(),
        option_book_value=sum(p.market_value for p in portfolio.option_positions), underlying_price=spot,
    )
    deployed_long_vol_notional, deployed_short_vol_risk = existing_vol_exposure_notional(portfolio.option_positions)
    if reconciliation is not None:
        # v3, requirement 15: fold still-OPEN order notional (not yet a
        # settled position) into the SAME incremental-sizing gap --
        # otherwise a resting order from last cycle that hasn't filled
        # yet is invisible to decide_sizing, and a slow-to-fill entry
        # could get sized again on top of itself.
        deployed_long_vol_notional += reconciliation.open_notional_by_kind.get("straddle", 0.0)
        deployed_short_vol_risk += reconciliation.open_notional_by_kind.get("condor", 0.0)

    greek_headroom = remaining_greek_headroom(portfolio, settings.risk)
    greek_hard_breach = greek_headroom["delta_shares"] < 0 or greek_headroom["theta_dollars"] < 0
    if greek_hard_breach:
        notes.append(
            f"GREEK HARD CAP already breached this cycle (delta headroom "
            f"{greek_headroom['delta_shares']:+.0f} shares, theta headroom "
            f"${greek_headroom['theta_dollars']:+.0f}/yr) -- no new entries until the book is back in budget."
        )

    extra_size_multiplier = 1.0
    book_vol = realized_book_vol(state.equity_curve)
    vt_scale = min(1.0, vol_target_scale(book_vol, settings.risk))        # never size UP past the hard caps
    dd_scale = edge_alloc.drawdown_scale(drawdown_pct, settings.risk.max_drawdown_pct)
    liq_scale = edge_alloc.liquidity_scale(_atm_spread_pct(chain, expiry, spot), getattr(settings, "max_bid_ask_spread_pct", 0.10))
    extra_size_multiplier *= vt_scale * dd_scale * liq_scale
    if extra_size_multiplier < 0.999:
        notes.append(f"Size scaled to {extra_size_multiplier:.0%}: book vol {'n/a' if book_vol is None else f'{book_vol:.1%}'} (target {settings.risk.target_portfolio_vol_min:.0%}-{settings.risk.target_portfolio_vol_max:.0%}) x{vt_scale:.2f}, drawdown {drawdown_pct:.1%} x{dd_scale:.2f}, liquidity x{liq_scale:.2f}.")
    if heston and heston.available:
        extra_size_multiplier *= heston.size_multiplier
    if toxicity and toxicity.available:
        extra_size_multiplier *= toxicity.size_multiplier

    cost_floor_vol_pts: Optional[float] = None
    if settings.enable_cost_floor:
        cost_floor_vol_pts = leland_cost_bps_of_vega(
            sigma=max(live_iv, 1e-3), k=settings.hedge.transaction_cost_bps / 10_000.0, dt=1 / 252,
        ) / 10_000.0

    planned_orders: list[dict[str, Any]] = []
    exit_decisions_raw: list[ExitDecision] = []

    # --- Auto-exit: evaluate open positions for stop-loss / take-profit / near-expiry ---
    # Deliberately NEVER gated by kill_switch/loss_tripped/event_risk/greek_hard_breach
    # above -- those all gate NEW risk-taking; closing existing risk should
    # if anything be EASIER under exactly those conditions, never harder.
    if not portfolio.option_positions:
        state.peak_pnl.clear()
    if settings.enable_auto_exit:
        try:
            exit_decisions_raw = evaluate_exits(raw_positions, settings.risk, settings.symbol, peaks=state.peak_pnl)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Auto-exit evaluation failed: {type(exc).__name__}: {exc}")
            log.warning("Auto-exit evaluation failed: %s", exc, exc_info=True)
        else:
            exit_orders, exit_notes = _execute_exits(client, settings, exit_decisions_raw)
            planned_orders.extend(exit_orders)
            notes.extend(exit_notes)
            if exit_decisions_raw and any(o.get("submitted") for o in exit_orders):
                state.last_exit_utc[settings.symbol] = us_now().isoformat()
                state.last_exit_reason[settings.symbol] = exit_decisions_raw[0].reason

    condor_stress_report = None
    sizing = None
    cooldown_reason = _entry_block_reason(state, settings)
    if cooldown_reason:
        notes.append(f"Entries paused: {cooldown_reason}.")
    block_new_entries = kill_switch or loss_tripped or greek_hard_breach or bool(cooldown_reason)
    if not block_new_entries:
        straddle = build_straddle(chain, expiry, spot, risk_free_rate, as_of=end)
        if straddle is not None:
            sizing = decide_sizing(
                signal.composite_edge, regime.current_label, equity,
                straddle.price, straddle.gamma, straddle.vega, settings.risk,
                deployed_long_vol_notional=deployed_long_vol_notional,
                deployed_short_vol_risk=deployed_short_vol_risk,
                extra_scale=extra_size_multiplier,
                vrp_raw=signal.vrp_raw,
                cost_floor_vol_pts=cost_floor_vol_pts,
                short_vol_extra_scale=event_risk.short_vol_size_multiplier if event_risk else 1.0,
                short_vol_blocked=bool(event_risk.short_vol_blocked) if event_risk else False,
                short_vol_block_reason="; ".join(event_risk.reasons) if event_risk else "",
            )
        else:
            notes.append(f"Could not build an ATM straddle for {settings.symbol} {expiry}; entries skipped.")

        if sizing and "cost_floor" in sizing.capped_by:
            notes.append(
                f"Edge {signal.vrp_raw:+.2%} does not clear the estimated round-trip hedging cost "
                f"({cost_floor_vol_pts:.2%} vol pts at {settings.hedge.transaction_cost_bps:.0f}bps) -- entry skipped."
            )
        elif sizing and sizing.already_at_target:
            notes.append(
                f"Already at target risk budget (${deployed_long_vol_notional + deployed_short_vol_risk:,.0f} "
                f"deployed vs. ${sizing.risk_budget_dollars:,.0f} target this cycle) -- no new entry."
            )

        extra_buffer_bps = toxicity.extra_limit_buffer_bps if (toxicity and toxicity.available) else 0.0

        if sizing and sizing.direction == "long_vol" and settings.enable_long_vol and sizing.contracts > 0 and straddle is not None:
            fit_n = fit_contracts_to_greek_caps(raw_positions, [(straddle.call, float(sizing.contracts)), (straddle.put, float(sizing.contracts))], sizing.contracts,
                                                account, settings.symbol, spot, risk_free_rate, settings.risk)
            if fit_n < sizing.contracts:
                notes.append(f"Pre-trade Greek check (existing + proposed): straddle cut from {sizing.contracts} to {fit_n} contract(s).")
            sizing = dataclasses.replace(sizing, contracts=fit_n)
        if sizing and sizing.direction == "long_vol" and not settings.enable_long_vol:
            notes.append("Long-vol signal ignored: buying straddles is off (ENABLE_LONG_VOL=false) -- premium decays and the long-vol edge is the weaker one.")
        elif sizing and sizing.direction == "long_vol" and sizing.contracts > 0:
            plan = plan_straddle_buy(
                straddle, sizing.contracts,
                limit_price_buffer_bps=STRADDLE_BASE_LIMIT_BUFFER_BPS + extra_buffer_bps,
            )
            result = submit_orders(client, plan, settings)
            if result.get("submitted"):
                state.last_entry_utc[settings.symbol] = us_now().isoformat()
            planned_orders.append({"kind": plan.kind, "description": plan.description, **result})
        elif sizing and sizing.direction == "short_vol" and settings.enable_credit_spreads:
            condor = select_iron_condor_by_cvar(
                chain, expiry, spot, risk_free_rate,
                settings.short_leg_target_delta,
                [settings.wing_target_delta * m for m in WING_DELTA_CANDIDATE_MULTIPLIERS],
                sizing.max_structure_loss, settings.risk.max_contracts_per_leg,
                equity=equity, max_structure_cvar_pct=settings.risk.max_structure_cvar_pct,
                cvar_alpha=settings.risk.cvar_alpha, as_of=end,
            )
            if condor is not None:
                fit_n = fit_contracts_to_greek_caps(raw_positions, condor_structure_legs(condor), condor.contracts, account, settings.symbol, spot, risk_free_rate, settings.risk)
                if fit_n < condor.contracts:
                    notes.append(f"Pre-trade Greek check (existing + proposed): condor cut from {condor.contracts} to {fit_n} contract(s)" + (" -- not even one fits the caps." if fit_n == 0 else "."))
                condor = dataclasses.replace(condor, contracts=fit_n) if fit_n > 0 else None
            if condor is not None:
                if condor.stress_report:
                    condor_stress_report = {
                        "worst_named_loss": condor.stress_report.worst_named_loss,
                        "monte_carlo_cvar_dollars": condor.stress_report.monte_carlo_cvar_dollars,
                        "monte_carlo_alpha": condor.stress_report.monte_carlo_alpha,
                        "leverage_slope": condor.stress_report.leverage_slope_vix_pts_per_pct,
                        "named_scenarios": [
                            {"label": s.label, "pnl_dollars": s.pnl_dollars} for s in condor.stress_report.named_scenarios
                        ],
                    }
                plan = plan_iron_condor(condor, limit_buffer_bps=extra_buffer_bps)
                result = submit_orders(client, plan, settings)
                if result.get("submitted"):
                    state.last_entry_utc[settings.symbol] = us_now().isoformat()
                planned_orders.append({"kind": plan.kind, "description": plan.description, **result})
            else:
                notes.append("Short-vol edge present but no iron condor could be built within the risk/CVaR budget or chain liquidity.")

    hedge_decision = None
    if settings.enable_equity_hedge:
        # v3, requirement 16: h_star now comes from a REAL regression
        # against the cross-cycle option-book-value / underlying-price
        # series agent.state persists every cycle (record_book_snapshot
        # above; hedge_ratio_inputs turns the two curves into the return
        # series min_variance_hedge_ratio's covariance formula needs).
        # v2 called this with `[], []` every single cycle -- literally
        # empty, per that version's own comment "v1 has no cross-cycle
        # P&L series to regress on yet" -- so the regression branch was
        # dead code and the hedge ratio was hard-coded to 1.0 in
        # practice. It still degrades to that same 1.0 fallback for a
        # fresh state file or thin history (see min_variance_hedge_ratio's
        # own min_obs gate) -- this is a strict improvement, not a
        # behavior change for a brand-new deployment.
        opt_returns, stock_returns = hedge_ratio_inputs(state, lookback=20)
        h_star = min_variance_hedge_ratio(opt_returns, stock_returns, fallback_delta=1.0)
        h_star = min(1.0, max(0.5, h_star))   # the regression ratio lives in return-space; never let it over-hedge
        hd = decide_hedge(
            portfolio_option_delta_shares=portfolio.net_delta_shares,
            portfolio_gamma_shares=portfolio.net_gamma_shares,
            current_hedge_shares=portfolio.equity_shares_held,
            regime_label=regime.current_label,
            spot=spot, time_to_expiry=max((expiry - end).days, 1) / 365.0,
            risk_free_rate=risk_free_rate, params=settings.hedge, hedge_ratio_override=h_star,
        )
        if hd.should_trade:
            mins = _minutes_since(state.last_hedge_utc)
            urgent = abs(hd.trade_shares) > 3 * hd.band_half_width or abs(portfolio.net_delta_shares) < 1
            if mins < settings.hedge.rehedge_min_interval_min and not urgent:
                hd.should_trade = False
                hd.reason += f" Skipped: last hedge was {mins:.0f} min ago (minimum {settings.hedge.rehedge_min_interval_min})."
        hedge_decision = {
            "should_trade": hd.should_trade, "target_shares": hd.target_shares,
            "current_shares": hd.current_shares, "trade_shares": hd.trade_shares,
            "band_half_width": hd.band_half_width, "reason": hd.reason,
        }
        if hd.should_trade:
            hedge_plan = plan_equity_hedge(settings.symbol, hd.trade_shares)
            if hedge_plan:
                result = submit_orders(client, hedge_plan, settings)
                if result.get("submitted"):
                    state.last_hedge_utc = us_now().isoformat()
                planned_orders.append({"kind": hedge_plan.kind, "description": hedge_plan.description, **result})
    else:
        h_star = None

    timestamp_utc = dt.datetime.utcnow().isoformat()
    record_equity(state, timestamp_utc, equity)
    risk_metrics = compute_risk_metrics(state)

    # For display: compute Kelly fraction + risk budget from the live signal
    # even when no straddle could be built (e.g. 0-DTE chain at close has
    # stale/crossed quotes). No execution happens -- purely for the UI.
    if sizing is not None:
        display_kelly = sizing.kelly_fraction
        display_budget = sizing.risk_budget_dollars
    else:
        display_kelly = _compute_kelly(
            signal.composite_edge, regime.current_label, settings.risk,
            extra_scale=extra_size_multiplier,
        )
        display_budget = abs(display_kelly) * equity

    result = AgentRunResult(
        timestamp_utc=timestamp_utc,
        symbol=settings.symbol, spot=spot,
        regime_label=regime.current_label, regime_confidence=regime.current_confidence,
        gamma_regime=signal.gamma_regime,
        forecast_vol=forecast_vol_now, live_atm_iv=live_iv,
        vrp_raw=signal.vrp_raw, vrp_z=signal.vrp_z, sas_mean=signal.sas_mean,
        composite_edge=signal.composite_edge, signal_rationale=signal.rationale,
        kelly_fraction=display_kelly,
        direction=sizing.direction if sizing else "flat",
        portfolio_delta_shares=portfolio.net_delta_shares, portfolio_gamma_shares=portfolio.net_gamma_shares,
        portfolio_vega_dollars=portfolio.net_vega_dollars, portfolio_theta_dollars=portfolio.net_theta_dollars,
        equity=equity, drawdown_pct=drawdown_pct, kill_switch=kill_switch,
        planned_orders=planned_orders, hedge_decision=hedge_decision, notes=notes,
        heston_cross_check=_heston_to_dict(heston), flow_toxicity=_toxicity_to_dict(toxicity),
        vol_path_hurst=vol_path_hurst,
        deployed_long_vol_notional=deployed_long_vol_notional, deployed_short_vol_risk=deployed_short_vol_risk,
        extra_size_multiplier=extra_size_multiplier,
        incremental_budget_dollars=sizing.incremental_budget_dollars if sizing else 0.0,
        already_at_target=sizing.already_at_target if sizing else False,
        risk_metrics=risk_metrics,
        risk_budget_dollars=display_budget,
        contracts_planned=sizing.contracts if sizing else 0,
        max_structure_loss=sizing.max_structure_loss if sizing else 0.0,
        capped_by=sizing.capped_by if sizing else [],
        regime_size_scale=REGIME_SIZE_SCALE.get(regime.current_label, 1.0),
        exit_decisions=[
            {"symbol": ed.symbol, "qty": ed.qty, "reason": ed.reason,
             "pnl_pct": ed.pnl_pct, "dte": ed.dte,
             "avg_entry_price": ed.avg_entry_price, "current_price": ed.current_price,
             "market_value": ed.market_value, "group": ed.group}
            for ed in exit_decisions_raw
        ],
        horizon_forecast=_horizon_forecast_to_dict(horizon_fc),
        event_risk=_event_risk_to_dict(event_risk),
        daily_pnl_pct=loss_pcts["daily_pnl_pct"], weekly_pnl_pct=loss_pcts["weekly_pnl_pct"],
        loss_breaker_tripped=loss_tripped, loss_breaker_reasons=loss_reasons,
        greek_headroom=greek_headroom, hedge_ratio_used=h_star,
        reconciliation=_reconciliation_to_dict(reconciliation),
        condor_stress_report=condor_stress_report,
        historical_iv_source=signal.historical_iv_source,
    )

    state.last_run_utc = result.timestamp_utc
    record_decision(state, {
        "timestamp_utc": result.timestamp_utc, "symbol": result.symbol, "regime": result.regime_label,
        "composite_edge": result.composite_edge, "direction": result.direction,
        "orders": [o.get("description") for o in planned_orders], "dry_run": settings.dry_run,
        # v3: order_ids threaded through so the dashboard/decision log can
        # cross-reference a decision-log entry against GET /orders' live
        # reconciled state by id, instead of the two panels having no
        # shared key to join on at all.
        "order_ids": [
            (o.get("orders", [{}])[0] or {}).get("id") for o in planned_orders if o.get("orders")
        ],
    })
    save_state(state)
    return result


def _horizon_forecast_to_dict(hf: Optional[HorizonForecast]) -> Optional[dict[str, Any]]:
    if hf is None:
        return None
    return {
        "horizon_days": hf.horizon_days, "garch_expected_vol": hf.garch_expected_vol,
        "garch_ci_low": hf.garch_ci_low, "garch_ci_high": hf.garch_ci_high,
        "har_expected_vol": hf.har_expected_vol, "blended_expected_vol": hf.blended_expected_vol,
        "blended_ci_low": hf.blended_ci_low, "blended_ci_high": hf.blended_ci_high,
        "har_r_squared": hf.har_r_squared, "notes": hf.notes,
    }


def _event_risk_to_dict(er: Optional[EventRisk]) -> Optional[dict[str, Any]]:
    if er is None:
        return None
    return {
        "short_vol_blocked": er.short_vol_blocked, "short_vol_size_multiplier": er.short_vol_size_multiplier,
        "reasons": er.reasons, "nearest_event": er.nearest_event, "nearest_event_date": er.nearest_event_date,
        "jump_z": er.jump_z, "jump_flagged": er.jump_flagged,
    }


def _reconciliation_to_dict(rr: Optional[ReconciliationResult]) -> Optional[dict[str, Any]]:
    if rr is None:
        return None
    return {
        "as_of_utc": rr.as_of_utc, "n_open_orders": len(rr.open_orders),
        "n_recently_filled": len(rr.recently_filled), "n_stale_cancelled": len(rr.actions_taken),
        "open_notional_by_kind": rr.open_notional_by_kind,
    }


def _heston_to_dict(h: Optional[HestonCrossCheck]) -> Optional[dict[str, Any]]:
    if h is None:
        return None
    return {
        "available": h.available, "n_points": h.n_points, "n_expiries": h.n_expiries,
        "calibration_success": h.calibration_success, "rmse_vol_pts": h.rmse_vol_pts,
        "heston_atm_iv": h.heston_atm_iv, "atm_gap_vol_pts": h.atm_gap_vol_pts,
        "data_quality_ok": h.data_quality_ok, "fair_term_vol": h.fair_term_vol, "term_gap": h.term_gap,
        "feller_ratio": h.feller_ratio, "vol_of_vol_xi": h.vol_of_vol_xi,
        "vol_of_vol_regime": h.vol_of_vol_regime, "size_multiplier": h.size_multiplier, "notes": h.notes,
    }


def _toxicity_to_dict(t: Optional[FlowToxicity]) -> Optional[dict[str, Any]]:
    if t is None:
        return None
    return {
        "available": t.available, "n_bars": t.n_bars, "n_buckets": t.n_buckets,
        "current_vpin": t.current_vpin, "percentile_vs_history": t.percentile_vs_history,
        "regime": t.regime, "size_multiplier": t.size_multiplier,
        "extra_limit_buffer_bps": t.extra_limit_buffer_bps, "notes": t.notes,
    }
