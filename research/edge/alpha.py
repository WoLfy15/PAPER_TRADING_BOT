"""
Alpha decomposition into four premia (short-vol-favourable = positive), rolling CAUSAL z-scores, and a walk-forward
ridge composite whose weights/error are fitted only on outcomes already realized.

  vrp    = IV_h - forecast_h                              (volatility risk premium vs the selected forecast)
  carry  = (IV_7 - IV_30) - (fc_7 - fc_30)                (market term structure vs the model's own term structure)
  skew   = z(skew_h) - z(realized downside/upside variance asymmetry)   [realized-skew PROXY from daily returns]
  jump   = z(curvature_h) - z(jump intensity)             (wing richness vs realized jump frequency) [PROXY]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def causal_z(x: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    base = x.shift(1)                                   # today's value never enters its own baseline
    return ((x - base.rolling(window, min_periods=min_periods).mean()) / base.rolling(window, min_periods=min_periods).std()).clip(-4, 4)


def realized_asymmetry(ret: pd.Series, w: int = 63) -> pd.Series:
    d, u = (ret.clip(upper=0) ** 2).rolling(w).sum(), (ret.clip(lower=0) ** 2).rolling(w).sum()
    return (d - u) / (d + u)


def jump_intensity(ret: pd.Series, w: int = 63) -> pd.Series:
    sd = ret.pow(2).ewm(alpha=0.06, adjust=False).mean().pow(0.5).shift(1)
    return (ret.abs() > 2.5 * sd).astype(float).rolling(w).mean()


def premia(surface: pd.DataFrame, fc: dict, ret: pd.Series, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fc: {7: Series, 14: Series, 30: Series} of SELECTED forecasts (annualized vol)."""
    ix = surface.index
    raw = pd.DataFrame(index=ix)
    raw["vrp"] = surface[f"iv_{h}d"] - fc[h].reindex(ix)
    raw["carry"] = (surface["iv_7d"] - surface["iv_30d"]) - (fc[7].reindex(ix) - fc[30].reindex(ix))
    raw["skew"], raw["curv"] = surface[f"skew_{h}d"], surface[f"curv_{h}d"]
    raw["asym"], raw["jump"] = realized_asymmetry(ret).reindex(ix), jump_intensity(ret).reindex(ix)
    z = pd.DataFrame(index=ix)
    z["z_vrp"], z["z_carry"] = causal_z(raw["vrp"]), causal_z(raw["carry"])
    z["z_skew"] = causal_z(raw["skew"]) - causal_z(raw["asym"])
    z["z_jump"] = causal_z(raw["curv"]) - causal_z(raw["jump"])
    return raw, z


def composite_alpha(z: pd.DataFrame, net: pd.Series, h: int, refit_every: int = 21, ridge: float = 10.0, min_rows: int = 250):
    """Walk-forward ridge of the realized net outcome on the four z-scores. A forecast made at s is a training row
    only from s+h on. Returns (mu, sigma, weights) where sigma is the residual std of REALIZED rows."""
    X, y, n = z.fillna(0.0).to_numpy(), net.to_numpy(), len(z)
    mu, sg, W = np.full(n, np.nan), np.full(n, np.nan), np.full((n, X.shape[1]), np.nan)
    w, s_res = None, np.nan
    for t in range(min_rows, n):
        if w is None or t % refit_every == 0:
            rows = np.arange(0, t - h + 1)
            rows = rows[~np.isnan(y[rows])]
            if len(rows) >= min_rows:
                A = np.column_stack([np.ones(len(rows)), X[rows]])
                reg = ridge * np.eye(A.shape[1]); reg[0, 0] = 0
                w = np.linalg.solve(A.T @ A + reg, A.T @ y[rows])
                s_res = float((y[rows] - A @ w).std(ddof=1))
        if w is not None:
            mu[t], sg[t], W[t] = w[0] + X[t] @ w[1:], s_res, w[1:]
    return pd.DataFrame({"mu": mu, "sigma": sg}, index=z.index), pd.DataFrame(W, index=z.index, columns=z.columns)


def rolling_ic(mu: pd.Series, net: pd.Series, h: int, window: int = 126) -> pd.Series:
    """Causal rolling Spearman IC of past signals vs the outcomes already realized (alpha-decay monitor)."""
    m, o = mu.shift(h), net
    out = pd.Series(np.nan, index=mu.index)
    mv, ov = m.to_numpy(), o.shift(0).to_numpy()
    nv = net.to_numpy()
    for t in range(window + h, len(mu)):
        a, b = mu.to_numpy()[t - h - window + 1: t - h + 1], nv[t - h - window + 1: t - h + 1]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 40:
            out.iloc[t] = stats.spearmanr(a[ok], b[ok]).statistic
    return out
