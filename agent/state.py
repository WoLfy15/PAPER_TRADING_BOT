"""
agent/state.py -- the only persistence this project uses: a small JSON
file tracking the account equity high-water mark (for the drawdown kill
switch), a rolling log of recent decisions (for the API's /decisions
endpoint and the UI's activity feed), and -- new in v2 -- a rolling
equity curve, because "what is the agent's Sharpe ratio" was previously
unanswerable from live state at all: v1 persisted the high-water mark
(a single running max) but never a time series, so there was nothing to
compute a live Sharpe/Sortino/drawdown FROM. compute_risk_metrics below
is what api/server.py's new /metrics endpoint and the dashboard's risk
panel actually read.

This is intentionally not a database. Swap it for one before running
this unattended for weeks -- see README "Productionizing" section --
but for a hackathon-length paper-trading loop, a JSON file that's
trivial to inspect (`cat agent_state.json`) is the right amount of
infrastructure.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

STATE_PATH = Path(__file__).resolve().parent.parent / "agent_state.json"
MAX_DECISION_LOG = 200
MAX_EQUITY_CURVE = 5000
TRADING_PERIODS_PER_YEAR = 252


@dataclass
class AgentState:
    equity_high_water_mark: float = 0.0
    last_run_utc: str = ""
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)  # [{"t": iso_ts, "equity": float}, ...]
    # v3: two more parallel curves, same shape as equity_curve, added so
    # strategy/hedging.min_variance_hedge_ratio can finally be called
    # with real cross-cycle data (requirement 16). Before this, agent/
    # loop.py called it as min_variance_hedge_ratio([], [], fallback_delta=1.0)
    # every single cycle -- literally empty lists, by its own comment
    # "v1 has no cross-cycle P&L series to regress on yet" -- so the
    # regression branch of an otherwise-correct function was simply dead
    # code and the hedge ratio was hard-coded to 1.0 delta-for-delta in
    # practice. option_book_curve is the mark-to-market value of ALL
    # option positions (strategy.portfolio.build_portfolio_snapshot's
    # own valuation, already computed every cycle for the Greeks panel
    # -- this just keeps a history of it); underlying_price_curve is the
    # hedge instrument's own price. hedge_ratio_inputs() below turns
    # both into the aligned return series min_variance_hedge_ratio needs.
    option_book_curve: list[dict[str, Any]] = field(default_factory=list)     # [{"t": iso_ts, "value": float}]
    underlying_price_curve: list[dict[str, Any]] = field(default_factory=list)  # [{"t": iso_ts, "price": float}]
    # v4.2: trailing-stop peaks, entry/exit cooldowns, last hedge time (all tolerate older state files)
    peak_pnl: dict[str, float] = field(default_factory=dict)
    last_entry_utc: dict[str, str] = field(default_factory=dict)
    last_exit_utc: dict[str, str] = field(default_factory=dict)
    last_exit_reason: dict[str, str] = field(default_factory=dict)
    last_hedge_utc: str = ""


def load_state() -> AgentState:
    if not STATE_PATH.exists():
        return AgentState()
    try:
        raw = json.loads(STATE_PATH.read_text())
        raw.setdefault("equity_curve", [])
        raw.setdefault("option_book_curve", [])
        raw.setdefault("underlying_price_curve", [])
        return AgentState(**raw)
    except (json.JSONDecodeError, TypeError):
        return AgentState()


def save_state(state: AgentState) -> None:
    state.decision_log = state.decision_log[-MAX_DECISION_LOG:]
    state.equity_curve = state.equity_curve[-MAX_EQUITY_CURVE:]
    state.option_book_curve = state.option_book_curve[-MAX_EQUITY_CURVE:]
    state.underlying_price_curve = state.underlying_price_curve[-MAX_EQUITY_CURVE:]
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2, default=str))


def record_decision(state: AgentState, entry: dict[str, Any]) -> None:
    state.decision_log.append(entry)


def record_equity(state: AgentState, timestamp_utc: str, equity: float) -> None:
    """Appends one point per agent cycle. Deliberately per-CYCLE, not
    per-calendar-day: at the default 15-minute loop interval this
    naturally gives an intraday-resolution curve early on, and
    compute_risk_metrics annualizes correctly either way because it
    infers the periods-per-year from the ACTUAL observed spacing of the
    points rather than assuming daily bars."""
    state.equity_curve.append({"t": timestamp_utc, "equity": float(equity)})


def record_book_snapshot(state: AgentState, timestamp_utc: str, option_book_value: float, underlying_price: float) -> None:
    """v3: companion to record_equity -- see AgentState's own comment on
    option_book_curve/underlying_price_curve for why these exist.
    Called once per cycle from agent/loop.py right alongside
    record_equity, with values strategy.portfolio.build_portfolio_snapshot
    and alpaca_client.get_mid_price already computed that cycle -- no
    new network calls."""
    state.option_book_curve.append({"t": timestamp_utc, "value": float(option_book_value)})
    state.underlying_price_curve.append({"t": timestamp_utc, "price": float(underlying_price)})


def hedge_ratio_inputs(state: AgentState, lookback: int = 20) -> tuple[list[float], list[float]]:
    """Turns the two curves above into the (option_pnl_returns,
    stock_returns) pair strategy.hedging.min_variance_hedge_ratio
    actually wants -- period-over-period CHANGES (not levels), aligned
    by position (both curves are appended together every cycle in
    record_book_snapshot, so index i in one curve is the same cycle as
    index i in the other; no separate timestamp-join is needed provided
    callers always call both together, which agent/loop.py does).
    Returns ([], []) until there are at least 3 aligned points (need at
    least 2 returns to compute anything) -- min_variance_hedge_ratio's
    own min_obs default (20) is what actually gates whether the
    regression is trusted; this just guards against a divide-by-nothing
    on the first couple of cycles after a fresh state file."""
    n = min(len(state.option_book_curve), len(state.underlying_price_curve))
    if n < 3:
        return [], []
    book = [p["value"] for p in state.option_book_curve[-lookback - 1:]][-n:]
    price = [p["price"] for p in state.underlying_price_curve[-lookback - 1:]][-n:]
    m = min(len(book), len(price))
    book, price = book[-m:], price[-m:]
    option_pnl_returns = [b - a for a, b in zip(book, book[1:])]  # dollar P&L deltas, not % -- matches
    stock_returns = [(b - a) for a, b in zip(price, price[1:])]   # min_variance_hedge_ratio's own covariance-of-levels convention
    return option_pnl_returns, stock_returns


def _periods_per_year(timestamps: list[str]) -> float:
    """Infer an annualization factor from the equity curve's own
    observed spacing rather than assuming daily -- a curve logged every
    15 minutes during market hours and a curve logged once a day need
    very different sqrt(N) scalings, and hard-coding 252 (as
    backtest/run_backtest.py correctly does for its genuinely daily
    series) would silently misstate live Sharpe by orders of magnitude."""
    import datetime as dt
    if len(timestamps) < 3:
        return TRADING_PERIODS_PER_YEAR
    try:
        parsed = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps]
    except ValueError:
        return TRADING_PERIODS_PER_YEAR
    deltas = [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return TRADING_PERIODS_PER_YEAR
    median_seconds = sorted(deltas)[len(deltas) // 2]
    trading_day_seconds = 6.5 * 3600  # ~6.5h regular session
    if median_seconds <= trading_day_seconds:
        # multiple observations per trading day (the common case at the
        # default 15-minute loop interval): scale by how many such
        # gaps fit in a trading day, then by 252 trading days/year.
        return max(TRADING_PERIODS_PER_YEAR * trading_day_seconds / median_seconds, 1.0)
    # coarser than one trading session apart: treat the gap as N calendar
    # days and scale 252 trading-days/year by it directly (this is the
    # branch backtest/run_backtest.py's own hard-coded 252 corresponds
    # to, at N=1 -- genuinely once-a-day sampling).
    median_days = median_seconds / 86400.0
    return max(TRADING_PERIODS_PER_YEAR / median_days, 1.0)


def compute_risk_metrics(state: AgentState, min_points: int = 5) -> dict[str, Optional[float]]:
    """Sharpe/Sortino/max-drawdown/win-rate off the live equity_curve.
    Returns Nones (not zeros -- a young agent with 3 logged cycles has
    an UNKNOWN Sharpe, not a zero one) until there's enough history to
    say anything. Same rf=0 convention run_backtest.py already uses for
    the excess-return proxy (returns net of a flat 0% -- both this and
    the backtest treat period-over-period equity return itself as the
    excess return, not re-subtracting a separate risk-free drag that's
    already implicitly inside the account's own cash balance)."""
    points = state.equity_curve
    if len(points) < min_points:
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None, "win_rate": None,
            "n_observations": len(points), "total_return": None,
        }
    equities = [p["equity"] for p in points]
    timestamps = [p["t"] for p in points]
    returns = [(b - a) / a for a, b in zip(equities, equities[1:]) if a > 0]
    if len(returns) < min_points - 1:
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None, "win_rate": None,
            "n_observations": len(points), "total_return": None,
        }

    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    std_r = math.sqrt(var_r)
    periods_per_year = _periods_per_year(timestamps)
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 1e-12 else None

    downside = [r for r in returns if r < 0]
    if len(downside) > 1:
        down_var = sum(r * r for r in downside) / len(downside)
        down_std = math.sqrt(down_var)
        sortino = (mean_r / down_std) * math.sqrt(periods_per_year) if down_std > 1e-12 else None
    else:
        sortino = None

    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1.0 + r)
        peak = max(peak, cum)
        max_dd = min(max_dd, cum / peak - 1.0)

    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    total_return = (equities[-1] - equities[0]) / equities[0] if equities[0] else None

    return {
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd, "win_rate": win_rate,
        "n_observations": len(points), "total_return": total_return,
    }


def compute_recent_loss_pcts(state: AgentState) -> dict[str, Optional[float]]:
    """v3: daily/weekly P&L, bucketed off the SAME equity_curve
    compute_risk_metrics already reads -- no new persistence, just a
    different rollup of what's already there. Feeds
    strategy.portfolio.check_loss_circuit_breakers (requirement 10's
    daily/weekly loss caps, distinct from the existing peak-to-trough
    max_drawdown_pct check: a fast single-day loss can be well inside a
    25% max-drawdown budget and still be exactly the kind of event those
    caps exist to catch quickly, before the NEXT full drawdown
    calculation would). "Today"/"this week" bucket using each point's
    own UTC calendar date -- consistent with how timestamps are stored
    (dt.datetime.utcnow().isoformat() at the call site), so this only
    drifts from the account's actual local trading-day boundary by the
    UTC offset, an acceptable approximation for a circuit breaker whose
    whole job is to fire EARLY and conservatively, not to reconcile to
    the cent with Alpaca's own statement cutoff."""
    import datetime as dt

    points = state.equity_curve
    if len(points) < 2:
        return {"daily_pnl_pct": None, "weekly_pnl_pct": None}
    try:
        parsed = [(dt.datetime.fromisoformat(p["t"].replace("Z", "+00:00")), p["equity"]) for p in points]
    except (ValueError, KeyError):
        return {"daily_pnl_pct": None, "weekly_pnl_pct": None}

    today = parsed[-1][0].date()
    week_start = today - dt.timedelta(days=today.weekday())  # Monday

    todays_points = [eq for ts, eq in parsed if ts.date() == today]
    weeks_points = [eq for ts, eq in parsed if ts.date() >= week_start]
    # "start of day/week" reference = the LAST point strictly before the
    # bucket began, if one exists (so the first cycle of the day is
    # compared against yesterday's close, not against itself -- a
    # same-bucket-only compare would read 0% all day until a second
    # cycle lands). Falls back to the bucket's own first point if the
    # curve doesn't reach further back (e.g. right after a fresh state file).
    prior_day = [eq for ts, eq in parsed if ts.date() < today]
    prior_week = [eq for ts, eq in parsed if ts.date() < week_start]
    day_ref = prior_day[-1] if prior_day else (todays_points[0] if todays_points else None)
    week_ref = prior_week[-1] if prior_week else (weeks_points[0] if weeks_points else None)

    daily_pct = (todays_points[-1] - day_ref) / day_ref if (todays_points and day_ref) else None
    weekly_pct = (weeks_points[-1] - week_ref) / week_ref if (weeks_points and week_ref) else None
    return {"daily_pnl_pct": daily_pct, "weekly_pnl_pct": weekly_pct}
