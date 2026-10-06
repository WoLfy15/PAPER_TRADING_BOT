"""One causal signal table per date: forecasts -> premia -> composite -> meta-label. Used by the backtest AND research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.edge.alpha import composite_alpha, premia, rolling_ic
from research.edge.forecast import build_forecasts
from research.edge.meta import CostModel, net_pnl_proxy, walk_forward_meta


def prepare_signals(prices: pd.Series, surface: pd.DataFrame, cm: CostModel | None = None, h: int = 7, use_garch: bool = True) -> pd.DataFrame:
    cm = cm or CostModel()
    ret = np.log(prices / prices.shift(1)).dropna()
    fc = {k: build_forecasts(ret, k, use_garch) for k in sorted({7, h, 30})}
    raw, z = premia(surface, {k: v["selected"] for k, v in fc.items()}, ret, h)
    iv, spread = surface[f"iv_{h}d"], surface["spread_pct"]
    net = net_pnl_proxy(iv, fc[h]["target"].reindex(surface.index), spread, cm)
    comp, W = composite_alpha(z, net, h)
    ic = rolling_ic(comp["mu"], net, h)
    reg = fc[h]["regime"].reindex(surface.index)
    X = z.assign(spread=spread, jump=raw["jump"], ts_slope=surface.get("ts_slope", 0.0), crisis=(reg == "crisis").astype(float), calm=(reg == "calm").astype(float))
    meta = walk_forward_meta(X, net, comp["mu"] > 0, h)
    sig = pd.concat([z, raw[["vrp", "carry", "skew", "curv", "asym", "jump"]].add_prefix("raw_"), comp.rename(columns={"mu": "mu", "sigma": "sigma_mu"}), W.add_prefix("w_"), meta], axis=1)
    sig["iv"], sig["spread_pct"], sig["fc"], sig["sigma_err"] = iv, spread, fc[h]["selected"].reindex(surface.index), fc[h]["sigma_err"].reindex(surface.index)
    sig["regime"], sig["ic"], sig["net"], sig["chosen"] = reg, ic, net, fc[h]["chosen"].reindex(surface.index)
    return sig
