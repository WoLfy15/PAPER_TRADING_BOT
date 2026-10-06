"""Out-of-sample validation: decile study, walk-forward with an untouched holdout, bootstrap, Deflated Sharpe,
PBO (CSCV) and a placebo/permutation test. Bin edges and every fitted quantity come from TRAINING rows only."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def _bins(s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, s, side="right")


def block_bootstrap(x: np.ndarray, stat, block: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng, n = np.random.default_rng(seed), len(x)
    k = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, k)])[:n]
        vals.append(stat(x[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def decile_study(signal, outcome, h: int, edges=None, bins=None, n_bins: int = 10, n_boot: int = 500, seed: int = 0) -> dict:
    """signal = forecast - IV;  outcome = subsequent realized vol - IV.  Monotonic increasing = the thesis holds."""
    df = pd.DataFrame({"s": np.asarray(signal, float), "o": np.asarray(outcome, float)})
    if bins is not None:
        df["b"] = np.asarray(bins)
    else:
        edges = np.quantile(df["s"].dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]) if edges is None else edges
        df["b"] = _bins(df["s"].to_numpy(), edges)
    df = df.dropna()
    tab = df.groupby("b").agg(n=("o", "size"), mean_signal=("s", "mean"), mean_outcome=("o", "mean"))
    rho = float(stats.spearmanr(tab.index, tab["mean_outcome"]).statistic) if len(tab) >= 3 else np.nan
    top, bot = df["b"].max(), df["b"].min()
    arr = df[["b", "o"]].to_numpy()
    def spread(a):
        hi_, lo_ = a[a[:, 0] == top, 1], a[a[:, 0] == bot, 1]
        return float(np.mean(hi_) - np.mean(lo_)) if len(hi_) and len(lo_) else np.nan
    lo, hi = block_bootstrap(arr, spread, block=max(2 * h, 5), n_boot=n_boot, seed=seed)
    ic = float(stats.spearmanr(df["s"], df["o"]).statistic)
    return {"table": tab, "edges": edges, "n": len(df), "decile_rho": rho, "spread": float(spread(arr)),
            "spread_ci": (lo, hi), "ic": ic, "passes": bool(rho >= 0.6 and lo > 0 and ic > 0)}


def walk_forward_study(signal: pd.Series, outcome: pd.Series, h: int, n_folds: int = 5, train_min: int = 504,
                       holdout_frac: float = 0.2, n_bins: int = 10) -> dict:
    """Expanding-window folds with an h-day embargo, then ONE evaluation on a final holdout that no fold, edge or
    threshold ever saw (edges come from all pre-holdout data)."""
    df = pd.DataFrame({"s": signal, "o": outcome}).dropna()
    n = len(df)
    ho = int(n * (1 - holdout_frac))
    dev, hold = df.iloc[:ho], df.iloc[ho + h:]
    cuts = np.linspace(train_min, ho, n_folds + 1).astype(int)
    pooled_s, pooled_o, pooled_b = [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        train, test = dev.iloc[: a - h], dev.iloc[a:b]
        edges = np.quantile(train["s"], np.linspace(0, 1, n_bins + 1)[1:-1])
        pooled_s.append(test["s"].to_numpy()); pooled_o.append(test["o"].to_numpy()); pooled_b.append(_bins(test["s"].to_numpy(), edges))
    oos = decile_study(np.concatenate(pooled_s), np.concatenate(pooled_o), h, bins=np.concatenate(pooled_b))
    frozen = np.quantile(dev["s"], np.linspace(0, 1, n_bins + 1)[1:-1])
    holdout = decile_study(hold["s"], hold["o"], h, edges=frozen) if len(hold) > 60 else None
    return {"oos": oos, "holdout": holdout, "n_dev": len(dev), "n_holdout": len(hold),
            "reject": (not oos["passes"]) or holdout is None or (not holdout["passes"])}


def permutation_pvalue(signal, outcome, h: int, n_perm: int = 1000, seed: int = 0) -> float:
    """Placebo: circularly shift the outcome by a random offset (keeps its autocorrelation, destroys alignment)."""
    s, o = np.asarray(signal, float), np.asarray(outcome, float)
    ok = ~(np.isnan(s) | np.isnan(o))
    s, o = s[ok], o[ok]
    ic0, rng = stats.spearmanr(s, o).statistic, np.random.default_rng(seed)
    null = [stats.spearmanr(s, np.roll(o, int(rng.integers(2 * h, len(o) - 2 * h)))).statistic for _ in range(n_perm)]
    return float((1 + np.sum(np.array(null) >= ic0)) / (1 + n_perm))


def sharpe(r) -> float:
    r = np.asarray(r, float)
    return float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0


def deflated_sharpe(returns, n_trials: int, trial_sr_var: float | None = None) -> float:
    """Bailey & Lopez de Prado. P(true Sharpe > 0 | the best of n_trials tries). Per-period Sharpe throughout."""
    r = np.asarray(returns, float)
    T, sr = len(r), sharpe(r)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-9))
    var = trial_sr_var if trial_sr_var is not None else denom**2 / (T - 1)
    sr0 = 0.0 if n_trials <= 1 else np.sqrt(var) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials) + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def pbo(M: np.ndarray, n_blocks: int = 10) -> float:
    """Probability of Backtest Overfitting via CSCV. M is T x N (rows = periods, cols = strategy variants)."""
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    logits = []
    for c in itertools.combinations(range(n_blocks), n_blocks // 2):
        i_in = np.concatenate([blocks[k] for k in c])
        i_out = np.concatenate([blocks[k] for k in range(n_blocks) if k not in c])
        sr = lambda ix: M[ix].mean(0) / np.maximum(M[ix].std(0, ddof=1), 1e-12)
        best = int(np.argmax(sr(i_in)))
        rank = stats.rankdata(sr(i_out))[best] / (N + 1)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))
