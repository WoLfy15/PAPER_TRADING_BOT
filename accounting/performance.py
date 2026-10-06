"""
accounting/performance.py -- ported, with one deliberate trim, from
the sister options_pricing_engine project's
real_market_validation/metrics.py (that project's own 10-year real-data
backtest already exercised this function -- see that project's
real_market_validation/backtest.py for the original run this came from).

performance_report is structure-agnostic (works on any daily P&L
series) and is kept as-is. The ORIGINAL file also had a
`monte_carlo_var` function -- deliberately NOT carried over: it assumes
a single, static, unhedged ATM straddle position, which was that
project's own (simpler, single-instrument) strategy. This project
trades either a straddle OR an iron condor, position-specific, and
already has a genuinely more general Monte Carlo/bootstrap tail-risk
tool for that -- pricing_engine/risk/scenarios.monte_carlo_cvar, which
works on ANY structure's actual legs and reuses real bootstrapped
historical (SPY return, VIX change) pairs rather than a flat-vol
simulation. Carrying the narrower function over unchanged would have
looked reusable while silently assuming a strategy this project doesn't
run -- see backtest/walk_forward.py for how the two different Monte
Carlo tools here are actually used together (position-level CVaR at
each roll via scenarios.py; backtest-result-level Sharpe/return CI via
a block bootstrap of the realized daily-return series, which is the
right tool for THAT specific question).

All P&L series in this module are in "points on a 100-notional base"
(a fund starting at 100 that receives these P&L points) -- divide by
100 to read as a decimal return. backtest/walk_forward.py's
report_backtest_performance is the adapter that converts this
project's own dollar-NAV daily_returns into that convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceReport:
    n_days: int
    n_active_days: int
    total_return_pct: float
    annualized_return_pct: float
    annualized_vol_pct: float
    sharpe: float
    sharpe_active_only: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    hit_rate: float
    var_95_hist_pct: float
    var_99_hist_pct: float
    cvar_95_hist_pct: float
    cvar_99_hist_pct: float
    var_95_parametric_pct: float
    var_99_parametric_pct: float
    skew: float
    kurtosis: float


def _drawdown(nav: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(nav)
    return (nav - running_max) / running_max


def performance_report(daily_pnl_points: np.ndarray, trading_days_per_year: int = 252) -> PerformanceReport:
    """daily_pnl_points: array of daily P&L in points on a 100-notional
    base (see backtest.py). NAV is tracked as 100 + cumsum(pnl), i.e. a
    fund starting at 100 that receives these P&L points."""
    pnl = np.asarray(daily_pnl_points, dtype=float)
    n = len(pnl)
    active = pnl[pnl != 0.0]

    nav = 100.0 + np.cumsum(pnl)
    dd = _drawdown(nav)
    max_dd_pct = float(dd.min() * 100.0)

    total_return_pct = float((nav[-1] / 100.0 - 1.0) * 100.0)
    years = n / trading_days_per_year
    annualized_return_pct = float(((nav[-1] / 100.0) ** (1.0 / years) - 1.0) * 100.0) if years > 0 else float("nan")

    ret = pnl / 100.0  # decimal daily return on the 100-notional base
    mean_r, std_r = ret.mean(), ret.std(ddof=1)
    annualized_vol_pct = float(std_r * math.sqrt(trading_days_per_year) * 100.0)
    sharpe = float(mean_r / std_r * math.sqrt(trading_days_per_year)) if std_r > 0 else float("nan")

    if len(active) > 1:
        active_ret = active / 100.0
        sharpe_active = float(active_ret.mean() / active_ret.std(ddof=1) * math.sqrt(trading_days_per_year))
    else:
        sharpe_active = float("nan")

    downside = ret[ret < 0]
    downside_dev = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = float(mean_r / downside_dev * math.sqrt(trading_days_per_year)) if downside_dev and downside_dev > 0 else float("nan")

    calmar = float(annualized_return_pct / abs(max_dd_pct)) if max_dd_pct != 0 else float("nan")
    hit_rate = float((active > 0).mean()) if len(active) else float("nan")

    var95_hist = float(-np.percentile(pnl, 5))
    var99_hist = float(-np.percentile(pnl, 1))
    cvar95_hist = float(-pnl[pnl <= np.percentile(pnl, 5)].mean()) if (pnl <= np.percentile(pnl, 5)).any() else float("nan")
    cvar99_hist = float(-pnl[pnl <= np.percentile(pnl, 1)].mean()) if (pnl <= np.percentile(pnl, 1)).any() else float("nan")

    mu_pts, sigma_pts = pnl.mean(), pnl.std(ddof=1)
    var95_param = float(-(mu_pts + (-1.6448536269514722) * sigma_pts))
    var99_param = float(-(mu_pts + (-2.3263478740408408) * sigma_pts))

    m3 = ((ret - mean_r) ** 3).mean()
    m2 = ((ret - mean_r) ** 2).mean()
    skew = float(m3 / m2 ** 1.5) if m2 > 0 else float("nan")
    m4 = ((ret - mean_r) ** 4).mean()
    kurt = float(m4 / m2 ** 2 - 3.0) if m2 > 0 else float("nan")

    return PerformanceReport(
        n_days=n, n_active_days=len(active),
        total_return_pct=total_return_pct, annualized_return_pct=annualized_return_pct,
        annualized_vol_pct=annualized_vol_pct, sharpe=sharpe, sharpe_active_only=sharpe_active,
        sortino=sortino, max_drawdown_pct=max_dd_pct, calmar=calmar, hit_rate=hit_rate,
        var_95_hist_pct=var95_hist, var_99_hist_pct=var99_hist,
        cvar_95_hist_pct=cvar95_hist, cvar_99_hist_pct=cvar99_hist,
        var_95_parametric_pct=var95_param, var_99_parametric_pct=var99_param,
        skew=skew, kurtosis=kurt,
    )
