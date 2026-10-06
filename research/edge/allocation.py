"""Sizing, scaling, ranking and pre-trade Greek checks (pure functions)."""
from __future__ import annotations

import numpy as np


def fractional_kelly(mu: float, sigma_pnl: float, sigma_err: float, frac: float = 0.25, cap: float = 0.02) -> float:
    """Risk budget as a fraction of equity = frac * z, z = edge / sqrt(outcome variance + forecast-error variance)
    (all in the same units, e.g. vol points). Full Kelly on a unit-variance bet is f* = z; we take `frac` of it and
    hard-cap it. More forecast uncertainty => smaller z => smaller size, before any other scaler."""
    if not np.isfinite(mu) or mu <= 0:
        return 0.0
    sd = np.sqrt((sigma_pnl if np.isfinite(sigma_pnl) else 0.0) ** 2 + (sigma_err if np.isfinite(sigma_err) else 0.0) ** 2)
    return float(min(cap, max(0.0, frac * mu / max(sd, 1e-8))))


def drawdown_scale(dd: float, limit: float) -> float:
    """1.0 until 40% of the drawdown limit is used, then linearly to 0 at the limit."""
    lo = 0.4 * limit
    return float(np.clip(1.0 - (dd - lo) / (limit - lo), 0.0, 1.0)) if dd > lo else 1.0


def liquidity_scale(spread_pct: float, max_spread: float) -> float:
    return float(np.clip(1.0 - (spread_pct / max_spread) ** 2, 0.0, 1.0)) if np.isfinite(spread_pct) else 0.5


def alpha_decay_scale(recent_ic: float, floor_ic: float = 0.0, full_ic: float = 0.10) -> float:
    """0 when the recent out-of-sample IC is at/below the floor, 1 at/above full_ic; unknown IC -> 0.5."""
    if not np.isfinite(recent_ic):
        return 0.5
    return float(np.clip((recent_ic - floor_ic) / (full_ic - floor_ic), 0.0, 1.0))


def scale_to_fit(existing: dict, proposed: dict, caps: dict) -> float:
    """Largest s in [0, 1] with |existing + s * proposed| <= cap for EVERY metric (delta, gamma, vega, theta, cvar...)."""
    s = 1.0
    for k, cap in caps.items():
        e, p = existing.get(k, 0.0), proposed.get(k, 0.0)
        if abs(e + p) <= cap:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if abs(e + mid * p) <= cap else (lo, mid)
        s = min(s, lo)
    return s


def rank_and_allocate(opps: list[dict], cvar_budget: float, min_ratio: float, max_per_underlying: float) -> list[dict]:
    """opps: {name, underlying, exp_net, cvar, contracts}. Keep exp_net/cvar >= min_ratio, best ratio first, fund
    until the portfolio CVaR budget and each underlying's concentration cap are used up (contracts scaled down)."""
    good = sorted((o for o in opps if o["exp_net"] > 0 and o["cvar"] > 0 and o["exp_net"] / o["cvar"] >= min_ratio),
                  key=lambda o: o["exp_net"] / o["cvar"], reverse=True)
    used, per, out = 0.0, {}, []
    for o in good:
        per_cvar = o["cvar"]
        room = min(cvar_budget - used, max_per_underlying - per.get(o["underlying"], 0.0))
        n = min(o["contracts"], int(room // per_cvar)) if per_cvar > 0 else 0
        if n >= 1:
            out.append({**o, "contracts": n}); used += n * per_cvar; per[o["underlying"]] = per.get(o["underlying"], 0.0) + n * per_cvar
    return out
