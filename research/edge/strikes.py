"""
Structure selection by the SOURCE of the detected mispricing, and strike optimisation by expected P&L / CVaR under
CAUSAL conditional scenarios (past h-day returns from the same regime) -- not fixed deltas.
Entry prices are realistic: sell at the bid, buy at the ask, plus a slippage share of each leg's half-spread.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

MULT = 100


def forward_scenarios(ret: pd.Series, regimes: pd.Series, h: int, t: int, min_n: int = 150) -> np.ndarray:
    """h-day log returns whose window had ENDED by index t, preferably from the regime in force at t."""
    fwd = ret.rolling(h).sum().shift(-h).to_numpy()
    idx = np.arange(0, t - h + 1)
    idx = idx[~np.isnan(fwd[idx])]
    if len(idx) < 50:
        return np.array([])
    reg = regimes.to_numpy()
    same = idx[reg[idx] == reg[t]]
    return fwd[same if len(same) >= min_n else idx]


def _verticals(q: pd.DataFrame, spot: float, cp: str, widths, min_otm: float, max_otm: float, slip: float) -> list[dict]:
    side = q[q["cp"] == cp].drop_duplicates("strike").set_index("strike").sort_index()
    ks_all = side.index.to_numpy()
    out = []
    for i, ks in enumerate(ks_all):
        otm = (spot - ks) / spot if cp == "P" else (ks - spot) / spot
        if not (min_otm <= otm <= max_otm):
            continue
        for w in widths:
            j = i - w if cp == "P" else i + w
            if j < 0 or j >= len(ks_all):
                continue
            kl = ks_all[j]
            bs, as_, bl, al = side.at[ks, "bid"], side.at[ks, "ask"], side.at[kl, "bid"], side.at[kl, "ask"]
            if bs <= 0 or al <= 0:
                continue
            credit = (bs - slip * (as_ - bs) / 2) - (al + slip * (al - bl) / 2)
            if credit > 0:
                sp = ((as_ - bs) / ((as_ + bs) / 2) + (al - bl) / ((al + bl) / 2)) / 2
                out.append({"cp": cp, "ks": float(ks), "kl": float(kl), "credit": float(credit), "width": float(abs(ks - kl)), "spread": float(sp)})
    return out


def _loss(v: dict, st: np.ndarray) -> np.ndarray:
    return np.clip(v["ks"] - st, 0, v["width"]) if v["cp"] == "P" else np.clip(st - v["ks"], 0, v["width"])


def _score(pnl_ps: np.ndarray, credit: float, spread: float, scen: np.ndarray, comm: float, lam_jump: float, lam_liq: float) -> dict:
    pnl = pnl_ps * MULT - comm
    e = float(pnl.mean())
    k = max(1, int(0.05 * len(pnl)))
    cvar = float(-np.sort(pnl)[:k].mean())
    sd = scen.std() if len(scen) else 0.0
    jm = np.abs(scen) > 3 * sd if sd > 0 else np.zeros(len(scen), bool)
    jump_pen = lam_jump * float(jm.mean()) * max(0.0, float(-pnl[jm].mean())) if jm.any() else 0.0
    liq_pen = lam_liq * credit * MULT * spread
    return {"exp_pnl": e, "cvar": max(cvar, 1e-6), "score": (e - jump_pen - liq_pen) / max(cvar, 1e-6), "spread": spread}


def optimize_structure(structure: str, quotes: pd.DataFrame, spot: float, scen: np.ndarray, widths=(1, 2, 3, 5), min_otm: float = 0.004,
                       max_otm: float = 0.05, slip: float = 0.25, comm_per_contract: float = 0.65, lam_jump: float = 1.0, lam_liq: float = 1.0,
                       top_k: int = 8) -> dict | None:
    """Best candidate of `structure` ('iron_condor' | 'put_vertical' | 'call_vertical' | 'long_straddle') by
    (E[pnl] - jump penalty - liquidity penalty) / CVaR95. Returns dict(legs=[(cp, strike, qty)], credit, exp_pnl, cvar, max_loss, score)."""
    if len(scen) < 50:
        return None
    st = quotes["underlying_price"].iloc[0] * np.exp(scen) if "underlying_price" in quotes else spot * np.exp(scen)
    if structure == "long_straddle":
        k = quotes.iloc[(quotes["strike"] - spot).abs().argmin()]["strike"]
        c, p = (quotes[(quotes["strike"] == k) & (quotes["cp"] == cp)].iloc[0] for cp in ("C", "P"))
        debit = (c["ask"] + slip * (c["ask"] - c["bid"]) / 2) + (p["ask"] + slip * (p["ask"] - p["bid"]) / 2)
        sc = _score(np.abs(st - k) - debit, debit, ((c["ask"] - c["bid"]) / ((c["ask"] + c["bid"]) / 2) + (p["ask"] - p["bid"]) / ((p["ask"] + p["bid"]) / 2)) / 2, scen, 2 * comm_per_contract, lam_jump, lam_liq)
        return {**sc, "structure": structure, "legs": [("C", float(k), 1), ("P", float(k), 1)], "credit": -float(debit), "max_loss": float(debit * MULT)}
    sides = {"put_vertical": ["P"], "call_vertical": ["C"], "iron_condor": ["P", "C"]}[structure]
    cands = {cp: _verticals(quotes, spot, cp, widths, min_otm, max_otm, slip) for cp in sides}
    if any(not v for v in cands.values()):
        return None
    scored = {}
    for cp, vs in cands.items():
        rows = [(v, _score(v["credit"] - _loss(v, st), v["credit"], v["spread"], scen, 2 * comm_per_contract, lam_jump, lam_liq)) for v in vs]
        scored[cp] = sorted(rows, key=lambda r: r[1]["score"], reverse=True)[:top_k]
    best = None
    for combo in itertools.product(*(scored[cp] for cp in sides)):
        vs = [c[0] for c in combo]
        credit = sum(v["credit"] for v in vs)
        sc = _score(credit - sum(_loss(v, st) for v in vs), credit, float(np.mean([v["spread"] for v in vs])), scen, 2 * comm_per_contract * len(vs), lam_jump, lam_liq)
        if best is None or sc["score"] > best["score"]:
            legs = [leg for v in vs for leg in ((v["cp"], v["ks"], -1), (v["cp"], v["kl"], 1))]
            best = {**sc, "structure": structure, "legs": legs, "credit": float(credit), "max_loss": float((max(v["width"] for v in vs) - credit) * MULT)}
    return best


def select_structure(z: dict, weights: dict, mu: float, allow_long_vol: bool = False) -> tuple[str, bool, str]:
    """Pick the structure from the DOMINANT source of the composite edge. Returns (structure, executable, why).
    calendar is recognised but not executable yet (needs two-expiry pricing/execution)."""
    if mu is None or not np.isfinite(mu):
        return "none", False, "no composite edge available"
    if mu < 0:
        return ("long_straddle", True, "implied vol is cheap vs forecast") if allow_long_vol else ("none", False, "long-vol disabled (ENABLE_LONG_VOL=false)")
    contrib = {k: weights.get(k, 0.0) * (z.get(k, 0.0) if np.isfinite(z.get(k, 0.0)) else 0.0) for k in ("z_vrp", "z_carry", "z_skew", "z_jump")}
    top = max(contrib, key=contrib.get)
    if contrib[top] <= 0:
        return "none", False, "no positive source"
    if top == "z_skew":
        return ("put_vertical" if z.get("z_skew", 0) > 0 else "call_vertical"), True, "edge driven by skew richness"
    if top == "z_carry":
        return "calendar", False, "edge driven by term structure (calendar not executable yet)"
    return "iron_condor", True, "edge driven by " + ("level (VRP)" if top == "z_vrp" else "jump/curvature premium")
