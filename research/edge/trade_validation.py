"""Trade-level validation of the live-replica backtest: nested walk-forward with a frozen final holdout, bootstrap
intervals, Deflated Sharpe, PBO, Holm-adjusted significance, regime reports and hard rejection rules."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from backtest.live_replica import ReplicaConfig, run_replica, summarize
from research.edge.validation import block_bootstrap, deflated_sharpe, pbo


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def objective(res: dict) -> float:
    """Stable net expectancy, Sharpe and drawdown -- NOT raw return: Sharpe - 2*|max drawdown| (needs >= 8 trades)."""
    s = summarize(res)
    return -1e9 if s.get("n_trades", 0) < 8 else s["sharpe"] - 2.0 * abs(s["max_drawdown"])


def nested_walk_forward(prices, snaps, sig, grid: list[dict], settings=None, n_outer: int = 4, holdout_frac: float = 0.2,
                        embargo_days: int = 10, base: ReplicaConfig | None = None) -> dict:
    """Outer folds: for each, pick the config from `grid` that maximises `objective` on TRAINING data only, freeze it,
    score it on the unseen test window. The final holdout is touched exactly once, with a config chosen on all earlier data."""
    base = base or ReplicaConfig()
    dates = [d for d in sig.index if d in snaps and d in prices.index]
    n = len(dates); ho = int(n * (1 - holdout_frac)); first = int(ho * 0.4)
    cuts = np.linspace(first, ho, n_outer + 1).astype(int)
    run = lambda cfg, a, b: run_replica(prices, snaps, sig, cfg, settings, start=dates[a], end=dates[b - 1])
    cfgs = [dataclasses.replace(base, **g) for g in grid]
    folds, pooled_tr, pooled_ret = [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        scores = [objective(run(c, 0, max(a - embargo_days, 1))) for c in cfgs]
        k = int(np.argmax(scores)); res = run(cfgs[k], a, b)
        folds.append({"test_start": dates[a], "test_end": dates[b - 1], "chosen": grid[k], "train_objective": scores[k], **summarize(res)})
        pooled_tr.append(res["trades"]); pooled_ret.append(res["equity"].pct_change().dropna())
    scores = [objective(run(c, 0, ho - embargo_days)) for c in cfgs]
    k = int(np.argmax(scores)); hold = run(cfgs[k], ho, n)
    oos_tr = pd.concat([t for t in pooled_tr if not t.empty], ignore_index=True) if any(not t.empty for t in pooled_tr) else pd.DataFrame()
    return {"folds": pd.DataFrame(folds), "oos_trades": oos_tr, "oos_returns": pd.concat(pooled_ret), "holdout": hold, "holdout_summary": summarize(hold),
            "holdout_config": grid[k], "n_trials": len(grid) * (n_outer + 1), "dates": (dates[0], dates[ho - 1], dates[-1])}


def regime_report(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L split by the volatility regime at entry and by trend (63-day return: bull > +5%, bear < -5%, else sideways).
    Earnings/event days are NOT tagged here (no calendar in the dataset) -- add one to split those out."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.assign(trend=np.where(trades["trend63"] > 0.05, "bull", np.where(trades["trend63"] < -0.05, "bear", "sideways")))
    rows = []
    for col in ("regime", "trend"):
        for k, g in t.groupby(col):
            rows.append({"split": col, "bucket": k, "n": len(g), "win_rate": (g["pnl"] > 0).mean(), "expectancy": g["pnl"].mean(), "total_pnl": g["pnl"].sum(),
                         "exp_per_risk": (g["pnl"] / g["max_loss"]).mean()})
    return pd.DataFrame(rows)


def credibility(trades: pd.DataFrame, returns: pd.Series, n_trials: int, config_returns: pd.DataFrame | None = None, seed: int = 0) -> dict:
    out = {"n_trades": int(len(trades))}
    if len(trades) < 5 or len(returns) < 30:
        return out
    epr = (trades["pnl"] / trades["max_loss"]).to_numpy()
    out["exp_per_risk"] = float(epr.mean())
    out["exp_per_risk_ci"] = block_bootstrap(epr, np.mean, 1, 2000, seed)
    out["t_pvalue"] = float(stats.ttest_1samp(epr, 0.0, alternative="greater").pvalue)
    out["sharpe_ci"] = block_bootstrap(returns.to_numpy(), lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 0 else 0.0, 10, 1000, seed)
    out["deflated_sharpe"] = deflated_sharpe(returns.to_numpy(), n_trials)
    if config_returns is not None and config_returns.shape[1] > 1:
        out["pbo"] = pbo(config_returns.dropna().to_numpy(), n_blocks=8)
    return out


def verdict(summary: dict, cred: dict, stress: dict, holdout: dict, trades: pd.DataFrame, max_risk_pct: float = 0.03) -> tuple[str, list[str]]:
    why = []
    if cred.get("n_trades", 0) < 30:
        why.append(f"too few out-of-sample trades ({cred.get('n_trades', 0)} < 30) for any statistical claim")
    if cred.get("exp_per_risk_ci", (0, 0))[0] <= 0:
        why.append("bootstrap interval of net expectancy per unit of risk includes zero")
    if cred.get("deflated_sharpe", 0.0) < 0.95:
        why.append(f"Deflated Sharpe {cred.get('deflated_sharpe', float('nan')):.2f} < 0.95 after {summary.get('n_trials', '?')} trials")
    if holdout.get("n_trades", 0) == 0 or holdout.get("total_return", 0) <= 0 or holdout.get("sharpe", 0) <= 0:
        why.append("untouched final holdout is not profitable")
    if stress.get("n_trades", 0) == 0 or stress.get("total_return", 0) <= 0:
        why.append("not profitable when spreads are doubled (profit depends on the execution assumption)")
    if summary.get("top5pct_profit_share", 0) and summary["top5pct_profit_share"] > 0.5:
        why.append(f"{summary['top5pct_profit_share']:.0%} of gross profit comes from the top 5% of trades (concentration)")
    if not trades.empty:
        yr = trades.groupby(pd.to_datetime(trades["exit_date"]).dt.year)["pnl"].sum()
        if yr.sum() > 0 and yr.max() / yr.sum() > 0.6:
            why.append("more than 60% of total profit comes from a single calendar year")
    if summary.get("avg_risk_pct_equity", 0) > max_risk_pct:
        why.append(f"average capital at risk per trade {summary['avg_risk_pct_equity']:.1%} > {max_risk_pct:.0%} (profit depends on leverage)")
    return ("REJECT" if why else "PASS"), why
