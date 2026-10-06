"""SYNTHETIC option-chain generator -- for TESTING THE MECHANICS ONLY. Any result obtained on this data says nothing
about real alpha: the implied vol here is built from realized vol with an arbitrary premium."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


def bsm(S, K, T, sigma, cp, r=0.04):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2) if cp == "C" else K * np.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)


def synth_chain(prices: pd.Series, atm_iv: pd.Series, dtes=(3, 7, 10, 14, 21, 30, 35), seed: int = 0, noise: float = 0.01, weekly: bool = False) -> pd.DataFrame:
    """weekly=True lists the next Friday expiries (40 days) so a contract stays listed until it expires, as in a real chain;
    weekly=False uses fixed DTE offsets (fine for single-day tests, wrong for multi-day backtests)."""
    rng, frames = np.random.default_rng(seed), []
    for d, S in prices.items():
        base = float(atm_iv.get(d, np.nan))
        if not np.isfinite(base):
            continue
        step = max(0.5, round(S * 0.005 * 2) / 2)
        ks = np.arange(np.floor(S * 0.88 / step) * step, S * 1.12, step)
        ts = pd.Timestamp(d).normalize() + pd.Timedelta(hours=15, minutes=45)
        day_dtes = [k for k in range(1, 41) if (pd.Timestamp(d) + pd.Timedelta(days=k)).weekday() == 4] if weekly else dtes
        for dte in day_dtes:
            T, exp = dte / 365, pd.Timestamp(d).normalize() + pd.Timedelta(days=dte)
            m = np.log(ks / S)
            iv = np.maximum(0.05, base * (1 + 0.03 * (np.sqrt(dte / 30) - 1)) * (1 - 1.4 * m + 4.0 * m * m) + rng.normal(0, noise * base, len(ks)))
            for cp in "CP":
                mid = bsm(S, ks, T, iv, cp)
                spr = np.maximum(0.01, 0.015 * mid + 0.01)
                frames.append(pd.DataFrame({"ts": ts, "underlying": "SPY", "underlying_price": S, "strike": ks, "expiration": exp, "cp": cp,
                                            "bid": np.maximum(0.0, np.round(mid - spr / 2, 2)), "ask": np.round(mid + spr / 2, 2), "iv": iv,
                                            "volume": rng.integers(50, 2000, len(ks)), "open_interest": rng.integers(500, 8000, len(ks))}))
    return pd.concat(frames, ignore_index=True)
