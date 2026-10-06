"""
Historical option-chain schema + constant-maturity ATM IV (calls AND puts), point-in-time.

Required columns (one row per quote snapshot): ts, underlying, underlying_price, strike, expiration, cp (C/P),
bid, ask, iv, volume, open_interest. Alpaca's historical API does not provide historical option quotes/IV, so this
dataset must come from a vendor (OPRA via Polygon/Databento/ThetaData, CBOE DataShop, OptionMetrics) as CSV/Parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHAIN_COLUMNS = ["ts", "underlying", "underlying_price", "strike", "expiration", "cp", "bid", "ask", "iv", "volume", "open_interest"]


def load_chain(path: str, underlying: str = "SPY") -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith((".parquet", ".pq")) else pd.read_csv(path)
    missing = [c for c in CHAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"chain dataset is missing columns: {missing}")
    df = df[CHAIN_COLUMNS].copy()
    df["ts"], df["expiration"] = pd.to_datetime(df["ts"]), pd.to_datetime(df["expiration"])
    df["cp"] = df["cp"].astype(str).str.upper().str[0]
    df = df[(df["underlying"] == underlying) & df["cp"].isin(["C", "P"])]
    return df[(df["bid"] >= 0) & (df["ask"] >= df["bid"]) & (df["iv"] > 0)].reset_index(drop=True)


def _spread_pct(e: pd.DataFrame) -> pd.Series:
    return (e["ask"] - e["bid"]) / ((e["ask"] + e["bid"]) / 2)


def _expiry_point(e: pd.DataFrame, spot: float) -> dict | None:
    """ATM IV (mean of the call and put at the strike nearest spot), 95%-moneyness put IV, 105%-moneyness call IV
    and ATM open interest for ONE expiry, from liquid quotes of a single snapshot."""
    both = set(e[e["cp"] == "C"]["strike"]) & set(e[e["cp"] == "P"]["strike"])
    if not both:
        return None
    k = min(both, key=lambda x: abs(x - spot))
    at = e[e["strike"] == k]
    pt = {"atm": float(at["iv"].mean()), "spread": float(_spread_pct(at).mean()), "oi": float(at["open_interest"].sum()), "put95": np.nan, "call105": np.nan}
    for cp, tgt, key in (("P", 0.95, "put95"), ("C", 1.05, "call105")):
        w = e[(e["cp"] == cp) & ((e["strike"] < spot) if cp == "P" else (e["strike"] > spot))]
        if not w.empty:
            r = w.iloc[(w["strike"] - tgt * spot).abs().argmin()]
            if abs(r["strike"] / spot - tgt) <= 0.02:
                pt[key] = float(r["iv"])
    return pt


def _var_interp(pts: list, tgt: int, key: str) -> float:
    lo = [p for p in pts if p[0] <= tgt and not np.isnan(p[1][key])]
    hi = [p for p in pts if p[0] >= tgt and not np.isnan(p[1][key])]
    if not lo or not hi:
        return np.nan
    (d0, a), (d1, b) = lo[-1], hi[0]
    w0, w1 = a[key] ** 2 * d0 / 365, b[key] ** 2 * d1 / 365
    w = w0 if d1 == d0 else w0 + (w1 - w0) * (tgt - d0) / (d1 - d0)
    return float(np.sqrt(w * 365 / tgt))


def build_surface(chain: pd.DataFrame, targets=(7, 14, 30), max_spread_pct: float = 0.25) -> pd.DataFrame:
    """Synchronized constant-maturity surface per date from the LAST snapshot of that day (one decision time, no
    later quotes). Per target: iv_Nd (ATM, calls+puts), skew_Nd (put95 - call105 IV), curv_Nd (wing average - ATM),
    each interpolated in total variance between bracketing expiries (NaN if not bracketed -- never extrapolated).
    Plus ts_slope = iv_30d - iv_7d, spread_pct and liq_oi (ATM open interest, front expiries)."""
    rows = []
    for day, g in chain.assign(day=chain["ts"].dt.normalize()).groupby("day"):
        g = g[g["ts"] == g["ts"].max()]
        spot = float(g["underlying_price"].iloc[0])
        pts = []
        for exp, e in g.groupby("expiration"):
            dte = (exp.normalize() - day).days
            if dte < 1:
                continue
            e = e[(e["bid"] > 0) & (_spread_pct(e) <= max_spread_pct)]
            pt = _expiry_point(e, spot) if len(e) else None
            if pt:
                pts.append((dte, pt))
        pts.sort(key=lambda p: p[0])
        rec = {"date": day, "spread_pct": float(np.mean([p[1]["spread"] for p in pts])) if pts else np.nan,
               "liq_oi": float(np.mean([p[1]["oi"] for p in pts[:3]])) if pts else np.nan}
        for t in targets:
            atm, pu, ca = (_var_interp(pts, t, k) for k in ("atm", "put95", "call105"))
            rec[f"iv_{t}d"], rec[f"skew_{t}d"] = atm, pu - ca
            rec[f"curv_{t}d"] = (pu + ca) / 2 - atm
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date")
    if "iv_30d" in out and "iv_7d" in out:
        out["ts_slope"] = out["iv_30d"] - out["iv_7d"]
    return out


def constant_maturity_iv(chain: pd.DataFrame, targets=(7, 14), max_spread_pct: float = 0.25) -> pd.DataFrame:
    """Back-compat wrapper: ATM IV per target (+ spread_pct)."""
    s = build_surface(chain, targets=targets, max_spread_pct=max_spread_pct)
    return s[[f"iv_{t}d" for t in targets] + ["spread_pct"]]
