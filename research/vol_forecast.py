"""
research/vol_forecast.py -- GARCH(1,1) + HAR-RV blended volatility
forecast, ported from IITfinal.ipynb (Part 1) and generalized off NVDA.

This produces the "what do we think realized vol will be" leg of the
volatility risk premium signal in research/signal.py. The other leg --
"what does the market currently think" -- comes from data/market_data.py
reading Alpaca's live option chain, replacing the notebook's
`realized_vol * 1.15` proxy (explicitly flagged as the #1 limitation in
OPTION-HEDGING-MODEL's own README) with the real thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS_PER_YEAR = 252


@dataclass
class VolForecast:
    garch_conditional_vol: pd.Series   # annualized, in-sample conditional vol
    garch_next_step: float             # annualized, true one-step-ahead forecast
    har_forecast: pd.Series            # annualized HAR-RV blend
    blended: pd.Series                 # 50/50 GARCH + HAR, annualized
    blended_next: float                # today's blended point estimate


@dataclass
class HorizonForecast:
    """requirement 1 ("genuine forward N-day variance forecast... match
    the actual option DTE") and requirement 6's point-forecast half
    ("E[future variance]... rather than a single point forecast" --
    the uncertainty/CI half is `ci_low`/`ci_high` below, both from a
    REAL simulated ensemble, not a parametric assumption bolted on
    after the fact -- see forecast_horizon's docstring).

    Every vol figure here is annualized and horizon-matched: it answers
    "what do we expect average daily vol to run at, over the NEXT
    `horizon_days` trading days" -- the number that's actually
    comparable to an option's implied vol at that same DTE, unlike
    VolForecast.blended_next (today's in-sample filtered level)."""
    horizon_days: int
    garch_expected_vol: float          # E[vol] over the horizon, from the simulated ensemble
    garch_ci_low: float                # 5th percentile across simulated paths
    garch_ci_high: float               # 95th percentile across simulated paths
    har_expected_vol: float            # iterated HAR-OLS forecast, same horizon
    blended_expected_vol: float        # 50/50 garch_expected_vol + har_expected_vol
    blended_ci_low: float              # GARCH CI recentered on the blended point (see docstring)
    blended_ci_high: float
    n_simulations: int
    har_r_squared: float               # in-sample fit quality of the HAR-OLS regression actually used
    notes: list[str] = field(default_factory=list)


def har_rv(realized_vol: pd.Series) -> pd.Series:
    """Heterogeneous Autoregressive RV: 1-day / 1-week / 1-month realized
    vol blend (Corsi 2009), same weights as the original notebook."""
    rv_1d = realized_vol
    rv_1w = realized_vol.rolling(5).mean()
    rv_1m = realized_vol.rolling(22).mean()
    return 0.5 * rv_1d + 0.3 * rv_1w + 0.2 * rv_1m


def forecast_volatility(log_returns: pd.Series, realized_vol: pd.Series) -> VolForecast:
    """log_returns, realized_vol: aligned Series, same index as the price
    history used elsewhere (e.g. research.regime.detect_regime's input).
    realized_vol should already be annualized (21-day rolling std * sqrt(252),
    matching research/regime.py's convention) so HAR and GARCH outputs are
    directly comparable."""
    if len(log_returns) < 60:
        raise ValueError(f"need >=60 return observations for a stable GARCH fit, got {len(log_returns)}")

    returns_pct = log_returns.dropna() * 100  # arch expects returns in %, standard convention
    garch = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch.fit(disp="off")

    garch_cond_vol = garch_fit.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) / 100
    garch_cond_vol.index = returns_pct.index

    one_step = garch_fit.forecast(horizon=1)
    garch_next = float(np.sqrt(one_step.variance.values[-1][0]) * np.sqrt(TRADING_DAYS_PER_YEAR) / 100)

    har = har_rv(realized_vol)
    idx = garch_cond_vol.index.intersection(har.index)
    blended = 0.5 * har.loc[idx] + 0.5 * garch_cond_vol.loc[idx]

    return VolForecast(
        garch_conditional_vol=garch_cond_vol,
        garch_next_step=garch_next,
        har_forecast=har,
        blended=blended,
        blended_next=float(blended.iloc[-1]) if len(blended) else garch_next,
    )


def _fit_har_ols(realized_vol: pd.Series) -> tuple[np.ndarray, float]:
    """Real Corsi (2009) HAR-RV: fit RV_t = b0 + b1*RV_1d_{t-1} +
    b2*RV_1w_{t-1} + b3*RV_1m_{t-1} + eps by OLS, rather than assuming
    v2's fixed 0.5/0.3/0.2 blend weights are the right one-step-ahead
    coefficients (they're a reasonable STATIC blend for a same-day
    read, which is all har_rv() above is used for; a genuine multi-step
    projection needs an actual fitted one-step model to iterate -- see
    _har_multistep_forecast). Returns (coefficients [b0,b1,b2,b3],
    in-sample R^2)."""
    rv = realized_vol.dropna()
    rv_1d = rv.shift(1)
    rv_1w = rv.rolling(5).mean().shift(1)
    rv_1m = rv.rolling(22).mean().shift(1)
    frame = pd.DataFrame({"y": rv, "rv_1d": rv_1d, "rv_1w": rv_1w, "rv_1m": rv_1m}).dropna()
    if len(frame) < 30:
        # Too little history for a trustworthy regression -- fall back to
        # the same fixed weights har_rv() uses, R^2 reported as NaN so
        # callers can tell this branch was taken.
        return np.array([0.0, 0.5, 0.3, 0.2]), float("nan")
    X = np.column_stack([np.ones(len(frame)), frame["rv_1d"], frame["rv_1w"], frame["rv_1m"]])
    y = frame["y"].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return coeffs, r_squared


def _har_multistep_forecast(realized_vol: pd.Series, horizon_days: int) -> tuple[float, float]:
    """Iterates the fitted one-step HAR-OLS model forward horizon_days
    times, feeding each step's own forecast back in as if it were the
    realized value (the standard technique for turning a one-step model
    into a multi-step path -- HAR has no closed-form multi-step formula,
    unlike GARCH's variance-forecast recursion arch computes natively).
    Returns (expected_vol_over_horizon, in_sample_r_squared). The
    returned vol is the average of the horizon_days iterated values,
    matching how garch_expected_vol below averages its own simulated
    path -- both answer "average daily vol over the next N days."."""
    coeffs, r_squared = _fit_har_ols(realized_vol)
    rv = realized_vol.dropna()
    history = list(rv.iloc[-22:].values)  # need the last 22 for the rolling components
    path = []
    for _ in range(max(horizon_days, 1)):
        rv_1d = history[-1]
        rv_1w = float(np.mean(history[-5:]))
        rv_1m = float(np.mean(history[-22:]))
        nxt = float(coeffs[0] + coeffs[1] * rv_1d + coeffs[2] * rv_1w + coeffs[3] * rv_1m)
        nxt = max(nxt, 1e-4)  # HAR-OLS has no positivity constraint -- floor a pathological negative fit
        path.append(nxt)
        history.append(nxt)
    return float(np.mean(path)), r_squared


def forecast_horizon(
    log_returns: pd.Series, realized_vol: pd.Series, horizon_days: int,
    n_simulations: int = 2000, ci_low_pct: float = 5.0, ci_high_pct: float = 95.0,
    seed: int = 7,
) -> HorizonForecast:
    """The horizon-matched, probabilistic replacement for reading
    VolForecast.blended_next as if it were a forward forecast (it was
    always today's in-sample level -- see that field's own comment).
    Called with horizon_days = the option's actual DTE (config.Settings.
    target_dte_days in the live agent; the backtest's own roll_freq_days/
    remaining-DTE at each roll), NOT a fixed constant, so the forecast
    always matches what's actually being traded.

    GARCH side: arch_model's own `method="simulation"` forecast --
    genuinely simulated forward variance PATHS from the fitted GARCH(1,1)
    process (not a parametric assumption bolted on after a point
    forecast). Each simulated path's average daily variance over the
    horizon is one draw from the model's own view of "average vol over
    the next `horizon_days` days"; the mean of those draws is the point
    forecast, and their 5th/95th percentiles (configurable) are a real,
    non-fabricated confidence interval -- requirement 6's "E[future
    variance] plus uncertainty/confidence intervals."

    HAR side: the fitted one-step HAR-OLS model (_fit_har_ols), iterated
    forward horizon_days times (_har_multistep_forecast) -- HAR has no
    native multi-step machinery the way GARCH's variance recursion does,
    so this is the standard "iterate the one-step model" technique
    rather than an ad hoc formula.

    The blended CI is the GARCH ensemble's CI re-centered on the 50/50
    blended point rather than a fresh HAR ensemble (HAR-OLS is a point
    model here, not simulated) -- stated explicitly rather than implied
    to be a joint simulation of both models."""
    if len(log_returns) < 60:
        raise ValueError(f"need >=60 return observations for a stable GARCH fit, got {len(log_returns)}")
    horizon_days = max(int(horizon_days), 1)
    notes: list[str] = []

    returns_pct = log_returns.dropna() * 100
    garch = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch.fit(disp="off")

    sim = garch_fit.forecast(horizon=horizon_days, method="simulation", simulations=n_simulations, reindex=False)
    # sim.simulations.variances shape: (1, n_simulations, horizon_days) -- a plain
    # ndarray in this arch version (unlike sim.variance, which is pandas-wrapped) --
    # percent^2 daily variances.
    sim_variances_pct2 = np.asarray(sim.simulations.variances)[0]  # (n_simulations, horizon_days)
    path_avg_daily_var_pct2 = sim_variances_pct2.mean(axis=1)  # one avg-daily-variance draw per simulated path
    path_annualized_vol = np.sqrt(np.clip(path_avg_daily_var_pct2, 0, None) * TRADING_DAYS_PER_YEAR) / 100.0

    garch_expected_vol = float(np.mean(path_annualized_vol))
    garch_ci_low = float(np.percentile(path_annualized_vol, ci_low_pct))
    garch_ci_high = float(np.percentile(path_annualized_vol, ci_high_pct))

    har_expected_vol, har_r2 = _har_multistep_forecast(realized_vol, horizon_days)
    if np.isnan(har_r2):
        notes.append("HAR-OLS fell back to fixed Corsi weights (insufficient history to fit); R^2 not meaningful.")

    blended_expected_vol = 0.5 * garch_expected_vol + 0.5 * har_expected_vol
    ci_shift = blended_expected_vol - garch_expected_vol
    blended_ci_low = max(garch_ci_low + ci_shift, 1e-4)
    blended_ci_high = garch_ci_high + ci_shift

    return HorizonForecast(
        horizon_days=horizon_days,
        garch_expected_vol=garch_expected_vol, garch_ci_low=garch_ci_low, garch_ci_high=garch_ci_high,
        har_expected_vol=har_expected_vol, blended_expected_vol=blended_expected_vol,
        blended_ci_low=blended_ci_low, blended_ci_high=blended_ci_high,
        n_simulations=n_simulations, har_r_squared=har_r2, notes=notes,
    )


def estimate_hurst(log_realized_vol: "np.ndarray | pd.Series", lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> Optional[float]:
    """Gatheral-Jaisson-Rosenbaum (2018) log-regression Hurst estimator,
    ported from rough_vol_project_v2/code/fbm_simulator.py's
    estimate_hurst (same closed-form: regress log(mean squared lag-k
    increment) on log(k); H = slope / 2).

    DIAGNOSTIC ONLY -- deliberately NOT wired into forecast_volatility's
    blend. Before adding it there, a rough-vol-kernel forecast (weights
    ~ lag^(H-0.5) applied to this same log-RV path) was tested
    out-of-sample against the real SPY price history this project
    already ships (backtest/out/SPY_backtest.csv): it did not improve on
    the existing GARCH+HAR blend -- RMSE got WORSE at every blend
    weight tried, monotonically, with the empirically best weight on the
    rough-vol component landing at 0.0 (i.e. "don't use it"). Two
    reasons, both worth recording rather than silently dropping the
    idea: (1) applying this estimator to an already 21-day-ROLLING
    (i.e. overlapping-averaged) realized-vol series is a methodological
    mismatch with the literature it's from -- GJR's own results are on
    much higher-frequency vol proxies, and rolling-averaging manufactures
    artificial smoothness (measured H ~= 0.57 here on daily-close-only
    SPY data, vs. the ~0.1 the literature reports from intraday data);
    (2) even setting that aside, real roughness estimation is known to
    need higher-frequency (e.g. 5-minute) data than a single daily close
    per day provides, which this project's data source doesn't have.
    Exposed here purely as an informational read (dashboard: "vol path
    roughness") -- H well below ~0.5 would suggest genuine short-memory
    structure worth revisiting as a forecast input LATER, with proper
    higher-frequency data; H close to the ~0.5-0.6 typically seen on
    daily-close-derived series is exactly the "don't trust this as rough
    vol" signal that motivated keeping it out of position sizing."""
    x = np.asarray(log_realized_vol, dtype=float)
    x = x[~np.isnan(x)]
    log_vars, log_lags = [], []
    for lag in lags:
        if lag >= len(x):
            continue
        diffs = x[lag:] - x[:-lag]
        mean_sq = float(np.mean(diffs ** 2))
        if mean_sq <= 0:
            continue
        log_vars.append(np.log(mean_sq))
        log_lags.append(np.log(lag))
    if len(log_lags) < 3:
        return None
    slope = float(np.polyfit(log_lags, log_vars, 1)[0])
    return slope / 2.0


def vrp_zscore(forecast_vol: pd.Series, implied_vol: pd.Series, lookback: int = 60) -> pd.Series:
    """Volatility Risk Premium, z-scored: VRP = forecast - implied. A
    positive z-score means the market is under-pricing volatility relative
    to the model's forecast (options look cheap -- see research/signal.py
    for how this becomes a trade direction)."""
    vrp = forecast_vol - implied_vol
    return (vrp - vrp.rolling(lookback).mean()) / vrp.rolling(lookback).std()
