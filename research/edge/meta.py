"""Meta-labeling: P(profitable after costs) and E[net P&L], walk-forward, labels used only once realized."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """All in annualized vol points (0.01 = 1 vol point). ASSUMPTIONS to be calibrated against real fills."""
    slip_vp: float = 0.004
    hedge_vp: float = 0.004
    commission_vp: float = 0.001
    spread_mult: float = 1.0          # 2.0 = the "double the spreads" stress

    def cost_vp(self, iv, spread_pct):
        return iv * spread_pct * self.spread_mult + self.slip_vp + self.hedge_vp + self.commission_vp


def net_pnl_proxy(iv: pd.Series, rv_fwd: pd.Series, spread_pct: pd.Series, cm: CostModel) -> pd.Series:
    """Short-vol P&L proxy in vol points: (IV - subsequent realized vol) - round-trip costs. A straddle/vega-style
    proxy used for the signal layer; the trade-level truth comes from the chain backtest."""
    return (iv - rv_fwd) - cm.cost_vp(iv, spread_pct.fillna(spread_pct.median()))


def adaptive_threshold(regime: str, jump_z: float, spread_pct: float, event: bool = False, iv_level: float = 0.15, base: float = 0.55) -> float:
    """Required P(profit): stricter in crisis, with jump risk, wide spreads, event days and high-vol levels."""
    thr = base + {"calm": 0.0, "transition": 0.03, "crisis": 0.10}.get(regime, 0.03)
    thr += min(0.06, 0.02 * max(0.0, 0.0 if pd.isna(jump_z) else jump_z))
    thr += min(0.06, max(0.0, (0.0 if pd.isna(spread_pct) else spread_pct) - 0.03))
    thr += 0.05 if event else 0.0
    thr += 0.04 if iv_level > 0.30 else 0.0
    return float(min(thr, 0.90))


def walk_forward_meta(X: pd.DataFrame, net: pd.Series, primary: pd.Series, h: int, refit_every: int = 21, min_rows: int = 200, C: float = 0.3) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    Xv, nv, prim, n = X.fillna(0.0).to_numpy(), net.to_numpy(), primary.fillna(False).to_numpy().astype(bool), len(X)
    y = (nv > 0).astype(float)
    p, en, sg = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    m, a, b, sres = None, 0.0, 0.0, np.nan
    for t in range(min_rows, n):
        if m is None or t % refit_every == 0:
            rows = np.arange(0, t - h + 1)
            rows = rows[prim[rows] & ~np.isnan(nv[rows])]
            if len(rows) >= min_rows and 0 < y[rows].mean() < 1:
                mu_, sd_ = Xv[rows].mean(0), Xv[rows].std(0) + 1e-9
                clf = LogisticRegression(C=C, max_iter=300).fit((Xv[rows] - mu_) / sd_, y[rows])
                pr = clf.predict_proba((Xv[rows] - mu_) / sd_)[:, 1]
                A = np.column_stack([np.ones(len(rows)), pr])
                coef = np.linalg.lstsq(A, nv[rows], rcond=None)[0]
                a, b, sres, m = coef[0], coef[1], float((nv[rows] - A @ coef).std(ddof=1)), (clf, mu_, sd_)
        if m is not None and prim[t]:
            pt = m[0].predict_proba(((Xv[t] - m[1]) / m[2]).reshape(1, -1))[0, 1]
            p[t], en[t], sg[t] = pt, a + b * pt, sres
    return pd.DataFrame({"p": p, "exp_net": en, "sigma_net": sg}, index=X.index)
