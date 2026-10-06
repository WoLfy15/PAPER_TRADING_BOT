"""
backtest/walk_forward.py -- requirement 19: "add comprehensive
walk-forward testing with parameter stability checks, Monte Carlo/
bootstrap confidence intervals, stress scenarios such as SPY -1%, -2%,
-3%, -5% combined with IV shocks, and out-of-sample validation, with
the main acceptance criterion being roughly full-sample OOS Sharpe
>1.25, median walk-forward Sharpe >1.25, worst-fold Sharpe preferably
>0.9-1.0, controlled CVaR and drawdown."

backtest/engine.py's run_backtest is ALREADY walk-forward-correct
throughout a single run (every regime fit and forecast at every roll
uses only data up to that day -- there is no code path in that file
that can see forward). What this module adds on top is the EVALUATION
layer requirement 19 is asking for:

  1. FOLD STABILITY: splits the realized daily-return series into
     chronological chunks and reports Sharpe per chunk -- "worst-fold
     Sharpe" and "median fold Sharpe" read directly off real, disjoint
     sub-periods of the same walk-forward run, not a re-fit.
  2. BOOTSTRAP CI: a block bootstrap (resampling contiguous chunks of
     daily returns, not individual days, to partially preserve
     autocorrelation) of the REALIZED daily-return series, giving a
     genuine confidence interval on the full-sample Sharpe -- how much
     of the measured Sharpe could plausibly be sampling noise.
  3. STRESS: reuses pricing_engine.risk.scenarios' named SPY+/-1/2/3/5%
     (with real fitted IV-shock pairing) against whatever position the
     backtest happened to be holding at each roll, reporting the worst
     one seen across the whole run -- not a separate, disconnected
     stress exercise.

THE ACCEPTANCE THRESHOLDS ABOVE ARE REPORTED, NEVER ASSERTED AS MET.
This function tells you where the numbers actually landed; it does not
decide whether that's good enough to trade. See this module's own
run_walk_forward_report docstring and backtest/engine.py's module
docstring for exactly what is and isn't real about the P&L feeding
these numbers -- a Sharpe measured against a semi-realistic
reconstruction (real underlying, real VIX, modeled spread) is a
genuinely useful number for judging the STRATEGY LOGIC, and a
genuinely different number from what real historical option fills
would have produced. Report it as what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from accounting.performance import PerformanceReport, performance_report
from backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252


@dataclass
class FoldResult:
    fold_index: int
    start_date: str
    end_date: str
    n_days: int
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float


@dataclass
class BootstrapCI:
    point_estimate_sharpe: float
    ci_low_sharpe: float
    ci_high_sharpe: float
    n_bootstrap_samples: int
    block_size_days: int


@dataclass
class WalkForwardReport:
    full_sample: PerformanceReport
    folds: list[FoldResult]
    median_fold_sharpe: float
    worst_fold_sharpe: float
    bootstrap_ci: Optional[BootstrapCI]
    worst_seen_stress: Optional[dict]   # the single worst named-scenario loss across every roll that held a position
    acceptance: dict[str, object] = field(default_factory=dict)   # requirement 19's own criteria, REPORTED not asserted
    notes: list[str] = field(default_factory=list)


def _report_from_returns(daily_returns: pd.Series) -> PerformanceReport:
    """Adapter: this project's own decimal daily returns -> the ported
    performance_report's "points on 100-notional base" convention (see
    accounting/performance.py's docstring)."""
    pnl_points = daily_returns.values * 100.0
    return performance_report(pnl_points, trading_days_per_year=TRADING_DAYS_PER_YEAR)


def fold_stability(daily_returns: pd.Series, n_folds: int = 4) -> list[FoldResult]:
    if len(daily_returns) < n_folds * 20:
        return []  # too little data for n_folds to mean anything -- caller sees an empty list, not a misleading single fold
    # np.array_split on a Series silently returns raw ndarrays (losing
    # the index this function needs for start_date/end_date) -- split by
    # position with .iloc instead, which reliably keeps each chunk a
    # proper Series.
    split_points = np.array_split(np.arange(len(daily_returns)), n_folds)
    out = []
    for i, positions in enumerate(split_points):
        if len(positions) < 5:
            continue
        chunk = daily_returns.iloc[positions[0]:positions[-1] + 1]
        rep = _report_from_returns(chunk)
        out.append(FoldResult(
            fold_index=i, start_date=str(chunk.index[0]), end_date=str(chunk.index[-1]),
            n_days=len(chunk), sharpe=rep.sharpe, total_return_pct=rep.total_return_pct,
            max_drawdown_pct=rep.max_drawdown_pct,
        ))
    return out


def block_bootstrap_sharpe_ci(
    daily_returns: pd.Series, n_bootstrap: int = 2000, block_size_days: int = 10,
    ci_low_pct: float = 5.0, ci_high_pct: float = 95.0, seed: int = 13,
) -> Optional[BootstrapCI]:
    """Block bootstrap (not iid resampling -- daily option-strategy
    returns are autocorrelated within a held position, so resampling
    individual days independently would understate the true sampling
    uncertainty) of the REALIZED daily-return series. Each bootstrap
    sample is built by drawing whole `block_size_days`-long contiguous
    chunks, with replacement, concatenated back to the original length;
    Sharpe is recomputed on each resample. The resulting spread of
    Sharpes is a genuine estimate of how much of the measured Sharpe
    could be sampling noise given THIS MUCH data -- not a claim about
    what future performance will look like."""
    returns = daily_returns.dropna().values
    n = len(returns)
    if n < block_size_days * 5:
        return None
    n_blocks = int(np.ceil(n / block_size_days))
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_bootstrap)
    max_start = n - block_size_days
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block_size_days] for s in starts])[:n]
        mean_r, std_r = sample.mean(), sample.std(ddof=1)
        sharpes[i] = (mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 1e-12 else 0.0

    point = (returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std(ddof=1) > 1e-12 else 0.0
    return BootstrapCI(
        point_estimate_sharpe=float(point), ci_low_sharpe=float(np.percentile(sharpes, ci_low_pct)),
        ci_high_sharpe=float(np.percentile(sharpes, ci_high_pct)), n_bootstrap_samples=n_bootstrap,
        block_size_days=block_size_days,
    )


def worst_stress_seen(result: BacktestResult) -> Optional[dict]:
    """Requirement 19's SPY +/-1/2/3/5% (+ real fitted IV shock) table,
    already computed once per short-vol roll by
    select_iron_condor_by_cvar as part of picking that trade's wing
    width (strategy/legs.py) and carried through on RollRecord.
    stress_report -- this just scans every roll that actually opened a
    condor and reports the single worst named-scenario loss seen across
    the whole run, as a fraction of that roll's equity. Deliberately
    reuses the SAME stress evaluation the backtest already made its
    sizing decision from, rather than a second, separately-computed
    stress pass that could tell a different story from what the
    strategy itself actually saw and acted on."""
    worst_pct: Optional[float] = None
    worst_detail: Optional[dict] = None
    equity_by_date = result.daily_nav
    for record in result.roll_log:
        if record.stress_report is None:
            continue
        equity_at_roll = float(equity_by_date.asof(record.date)) if len(equity_by_date) else result.config.starting_cash
        if not equity_at_roll or equity_at_roll <= 0:
            continue
        for scenario in record.stress_report.named_scenarios:
            pct = scenario.pnl_dollars / equity_at_roll
            if worst_pct is None or pct < worst_pct:
                worst_pct = pct
                worst_detail = {
                    "date": str(record.date), "scenario": scenario.label,
                    "pnl_dollars": scenario.pnl_dollars, "pnl_pct_of_equity": pct,
                    "equity_at_roll": equity_at_roll,
                }
    return worst_detail


def run_walk_forward_report(result: BacktestResult, n_folds: int = 4) -> WalkForwardReport:
    """The one function backtest/run_full_backtest.py (and any user
    script) calls after run_backtest. Puts together everything this
    module builds, plus requirement 19's literal acceptance thresholds
    -- REPORTED, never asserted true; see module docstring."""
    notes: list[str] = []
    full = _report_from_returns(result.daily_returns)
    folds = fold_stability(result.daily_returns, n_folds=n_folds)
    fold_sharpes = [f.sharpe for f in folds if np.isfinite(f.sharpe)]
    median_fold = float(np.median(fold_sharpes)) if fold_sharpes else float("nan")
    worst_fold = float(np.min(fold_sharpes)) if fold_sharpes else float("nan")
    if not folds:
        notes.append(f"Not enough daily-return history for {n_folds} meaningful folds -- fold stability skipped.")

    ci = block_bootstrap_sharpe_ci(result.daily_returns)
    if ci is None:
        notes.append("Not enough daily-return history for a meaningful block bootstrap -- CI skipped.")

    worst_stress = worst_stress_seen(result)
    if worst_stress is None:
        notes.append("No short-vol trades with a captured stress report in this run -- stress-scenario reporting skipped.")

    acceptance = {
        "full_sample_oos_sharpe": full.sharpe,
        "full_sample_oos_sharpe_target": 1.25,
        "full_sample_meets_target": bool(np.isfinite(full.sharpe) and full.sharpe > 1.25),
        "median_walk_forward_sharpe": median_fold,
        "median_meets_target": bool(np.isfinite(median_fold) and median_fold > 1.25),
        "worst_fold_sharpe": worst_fold,
        "worst_fold_target": 0.9,
        "worst_fold_meets_target": bool(np.isfinite(worst_fold) and worst_fold > 0.9),
        "max_drawdown_pct": full.max_drawdown_pct,
        "cvar_95_hist_pct": full.cvar_95_hist_pct,
    }
    notes.append(
        "These targets are requirement 19's own literal numbers, reported for comparison -- they are NOT "
        "validated trading thresholds and meeting them here is not a claim about real-money performance. "
        "See backtest/engine.py's module docstring for exactly what is and isn't real about the P&L this "
        "was measured against (real underlying + real VIX; modeled, not observed, bid/ask)."
    )

    return WalkForwardReport(
        full_sample=full, folds=folds, median_fold_sharpe=median_fold, worst_fold_sharpe=worst_fold,
        bootstrap_ci=ci, worst_seen_stress=worst_stress, acceptance=acceptance, notes=notes,
    )
