"""
Leakage-free rolling realized-vol forecasts.

Contract (enforced by tests/test_edge_pipeline.py): the value at date t is a function of returns up to and
including t ONLY. Targets are the annualized realized vol over the NEXT h days; a target is "known" at date t
only if its window ended by t, i.e. forecasts made at s are scored/used from s+h onward, never earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def forward_rv(ret: pd.Series, h: int) -> pd.Series:
    """Annualized realized vol over (t+1 .. t+h). NaN for the last h dates. Used ONLY as a target."""
    return np.sqrt(ret.pow(2).rolling(h).mean().shift(-h) * ANN)


def trailing_rv(ret: pd.Series, window: int = 21) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(ANN)


def ewma_forecast(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt(ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean() * ANN)


def har_forecast(ret: pd.Series, h: int, refit_every: int = 21, min_train: int = 504) -> pd.Series:
    """HAR on daily squared returns: forward h-day variance ~ r2(1d) + mean r2(5d) + mean r2(22d).
    Re-fit every `refit_every` days on rows whose target has ALREADY been realized (s <= t-h)."""
    r2 = ret.pow(2)
    X = pd.concat([r2, r2.rolling(5).mean(), r2.rolling(22).mean()], axis=1).to_numpy()
    y = r2.rolling(h).mean().shift(-h).to_numpy()
    n, out, beta = len(ret), np.full(len(ret), np.nan), None
    for t in range(n):
        if t >= min_train and (beta is None or (t - min_train) % refit_every == 0):
            s = t - h                                    # last row whose target is realized by t
            rows = np.arange(21, s + 1)
            rows = rows[~np.isnan(y[rows]) & ~np.isnan(X[rows]).any(axis=1)]
            if len(rows) > 100:
                A = np.column_stack([np.ones(len(rows)), X[rows]])
                beta = np.linalg.lstsq(A, y[rows], rcond=None)[0]
        if beta is not None and not np.isnan(X[t]).any():
            out[t] = max(float(beta[0] + X[t] @ beta[1:]), 1e-8)
    return pd.Series(np.sqrt(out * ANN), index=ret.index)


def garch_forecast(ret: pd.Series, h: int, refit_every: int = 63, min_train: int = 756) -> pd.Series:
    """Zero-mean GARCH(1,1), params re-fit every `refit_every` days on data <= t, variance filtered forward
    with the latest params; h-day average variance from the closed-form mean reversion."""
    from arch import arch_model
    r, n = ret.to_numpy(), len(ret)
    out, params, s2 = np.full(n, np.nan), None, None
    for t in range(n):
        if params is not None and s2 is not None and t > 0:
            s2 = params[0] + params[1] * r[t - 1] ** 2 + params[2] * s2     # advance sigma^2_{t-1} -> sigma^2_t
        if t >= min_train and (params is None or (t - min_train) % refit_every == 0):
            try:
                res = arch_model(r[: t + 1] * 100, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off", show_warning=False)
                om, al, be = res.params["omega"] / 1e4, res.params["alpha[1]"], res.params["beta[1]"]
                params = (om, al, min(be, 0.999 - al))
                s2 = float(res.conditional_volatility[-1] ** 2 / 1e4)
            except Exception:  # noqa: BLE001 -- keep the previous params
                pass
        if params is not None and s2 is not None:
            om, al, be = params
            p, nxt = al + be, om + al * r[t] ** 2 + be * s2           # sigma^2_{t+1}
            V = om / (1 - p)
            k = np.arange(h)
            out[t] = np.mean(V + p**k * (nxt - V))
    return pd.Series(np.sqrt(np.maximum(out, 1e-10) * ANN), index=ret.index)


def causal_regimes(ret: pd.Series, window: int = 21, calm_q: float = 0.40, crisis_q: float = 0.90) -> pd.Series:
    """calm / transition / crisis from the EXPANDING percentile of trailing vol (information at t only)."""
    rv = trailing_rv(ret, window)
    pct = rv.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True)
    lab = pd.Series("transition", index=ret.index, dtype=object)
    lab[pct <= calm_q], lab[pct >= crisis_q] = "calm", "crisis"
    lab[pct.isna()] = np.nan
    return lab


def regime_adjust(base: pd.Series, y: pd.Series, regimes: pd.Series, h: int, shrink: float = 50.0) -> pd.Series:
    """Regime-CONDITIONING layer (not an alpha source): scale the base forecast by the shrunk historical
    realized/forecast ratio observed in the same regime, using only ratios already realized by t."""
    ratio, mult = (y / base).replace([np.inf, -np.inf], np.nan), pd.Series(1.0, index=base.index)
    for reg in ("calm", "transition", "crisis"):
        m = (regimes == reg) & ratio.notna()
        cs, cn = ratio.where(m, 0.0).cumsum().shift(h), m.astype(float).cumsum().shift(h)
        est = ((cs + shrink * 1.0) / (cn + shrink)).fillna(1.0)
        mult[regimes == reg] = est[regimes == reg]
    return base * mult


def ensemble_forecast(fcs: dict[str, pd.Series], y: pd.Series, h: int, window: int = 126) -> tuple[pd.Series, pd.DataFrame]:
    """Inverse-rolling-error weights. Errors of forecasts made at s become usable at s+h (shift(h))."""
    names = list(fcs)
    err = pd.DataFrame({k: (np.log(fcs[k]) - np.log(y)) ** 2 for k in names}).shift(h)
    mse = err.rolling(window, min_periods=30).mean()
    inv = 1.0 / mse.clip(lower=1e-6)
    w = inv.div(inv.sum(axis=1), axis=0)
    w = w.where(w.notna().all(axis=1), 1.0 / len(names))
    F = pd.DataFrame(fcs)
    return (F * w).sum(axis=1, min_count=1).where(F.notna().all(axis=1)), w


def regime_selected_forecast(fcs: dict[str, pd.Series], y: pd.Series, regimes: pd.Series, h: int,
                             halflife: float = 126.0, min_obs: int = 40) -> tuple[pd.Series, pd.Series]:
    """Per regime, pick the model with the lowest recency-weighted out-of-sample error OBSERVED IN THAT REGIME.
    Errors of forecasts made at s count from s+h on (shift(h)); the regime tag travels with the error."""
    names = list(fcs)
    F = pd.DataFrame(fcs)
    err = ((np.log(F).sub(np.log(y), axis=0)) ** 2).shift(h)
    reg_k = regimes.shift(h)
    mse, cnt = {}, {}
    for r in ("calm", "transition", "crisis"):
        masked = err.where(np.repeat((reg_k == r).to_numpy()[:, None], err.shape[1], axis=1))
        mse[r] = masked.ewm(halflife=halflife, ignore_na=True).mean().to_numpy()
        cnt[r] = masked.notna().cumsum().to_numpy()
    out, chosen = np.full(len(F), np.nan), np.full(len(F), None, dtype=object)
    Fv, reg_now = F.to_numpy(), regimes.to_numpy()
    for t in range(len(F)):
        if np.isnan(Fv[t]).any() or reg_now[t] not in mse:
            continue
        m, c = mse[reg_now[t]][t], cnt[reg_now[t]][t]
        if (c >= min_obs).all() and not np.isnan(m).any():
            k = int(np.argmin(m)); out[t], chosen[t] = Fv[t, k], names[k]
        else:
            out[t], chosen[t] = Fv[t].mean(), "equal_weight"
    return pd.Series(out, index=F.index), pd.Series(chosen, index=F.index)


def build_forecasts(ret: pd.Series, h: int, use_garch: bool = True) -> pd.DataFrame:
    """All components + fixed ensemble (kept for ablation) + the REGIME-SELECTED forecast (the one used for trading)
    and sigma_err = rolling std of the selected forecast's realized log error (usable only after h days)."""
    y = forward_rv(ret, h)
    comps = {"ewma": ewma_forecast(ret), "har": har_forecast(ret, h)}
    if use_garch:
        comps["garch"] = garch_forecast(ret, h)
    reg = causal_regimes(ret)
    comps["har_regime"] = regime_adjust(comps["har"], y, reg, h)
    ens, _ = ensemble_forecast(comps, y, h)
    sel, chosen = regime_selected_forecast(comps, y, reg, h)
    sig_log = (np.log(sel) - np.log(y)).shift(h).rolling(252, min_periods=60).std()
    out = pd.DataFrame(comps)
    out["ensemble"], out["selected"], out["chosen"] = ens, sel, chosen
    out["sigma_err"], out["target"], out["regime"] = sig_log * sel, y, reg      # sigma_err in annualized vol points
    return out


_LIVE_CACHE: dict = {}


def latest_forecast(ret: pd.Series, h: int) -> dict:
    """What the LIVE agent calls: the same build_forecasts, last row, cached per (last date, h)."""
    key = (str(ret.index[-1]), h, len(ret))
    if key not in _LIVE_CACHE:
        _LIVE_CACHE.clear()
        full = build_forecasts(ret, h)
        row = full.iloc[-1]
        _LIVE_CACHE[key] = {"selected": float(row["selected"]), "sigma_err": float(row["sigma_err"]) if pd.notna(row["sigma_err"]) else None,
                            "chosen": row["chosen"], "regime": row["regime"], "history": full["selected"].tail(60), "ensemble": float(row["ensemble"]) if pd.notna(row["ensemble"]) else None}
    return _LIVE_CACHE[key]
