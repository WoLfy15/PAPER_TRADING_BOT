import numpy as np
import pandas as pd

from research.edge import forecast as F
from research.edge.data_schema import constant_maturity_iv
from research.edge.validation import decile_study, deflated_sharpe, pbo, permutation_pvalue, walk_forward_study


def _returns(n=1500, seed=1):
    rng, vol, out = np.random.default_rng(seed), 0.01, []
    for _ in range(n):
        vol = np.sqrt(2e-6 + 0.09 * (out[-1] if out else 0) ** 2 + 0.88 * vol**2)
        out.append(vol * rng.standard_normal())
    return pd.Series(out, index=pd.bdate_range("2018-01-01", periods=n))


def test_forecasts_do_not_look_ahead():
    """Wrecking every return after day 1100 must not change ANY forecast made on or before day 1100."""
    r = _returns()
    r2 = r.copy()
    r2.iloc[1101:] = r2.iloc[1101:] * 7 + 0.03
    a, b = F.build_forecasts(r, 7), F.build_forecasts(r2, 7)
    cols = ["ewma", "har", "garch", "har_regime", "ensemble", "selected", "sigma_err"]
    pd.testing.assert_frame_equal(a.loc[: r.index[1100], cols], b.loc[: r.index[1100], cols], check_exact=False, rtol=1e-9, atol=1e-12)
    assert a["ensemble"].dropna().shape[0] > 300            # the ensemble actually produced forecasts


def test_forward_target_is_the_future_only():
    r = _returns(300)
    y = F.forward_rv(r, 7)
    assert np.isclose(y.iloc[10], np.sqrt((r.iloc[11:18] ** 2).mean() * 252))
    assert y.iloc[-7:].isna().all()


def test_decile_study_accepts_a_planted_signal_and_rejects_noise():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4000)
    good = decile_study(s, 0.6 * s + rng.standard_normal(4000), h=7)
    bad = decile_study(s, rng.standard_normal(4000), h=7)
    assert good["passes"] and good["decile_rho"] > 0.8 and good["spread_ci"][0] > 0
    assert not bad["passes"]


def test_walk_forward_holdout_rejects_a_signal_that_dies_out_of_sample():
    rng = np.random.default_rng(3)
    n = 3000
    s = pd.Series(rng.standard_normal(n), index=pd.bdate_range("2010-01-01", periods=n))
    o = 0.6 * s + rng.standard_normal(n)
    live = walk_forward_study(s, o, h=7)
    o_dead = o.copy()
    o_dead.iloc[int(n * 0.8):] = rng.standard_normal(n - int(n * 0.8))   # relationship vanishes in the holdout
    dead = walk_forward_study(s, o_dead, h=7)
    assert not live["reject"] and dead["reject"]


def test_permutation_pvalue_separates_signal_from_noise():
    rng = np.random.default_rng(5)
    s = rng.standard_normal(1500)
    assert permutation_pvalue(s, 0.5 * s + rng.standard_normal(1500), 7, n_perm=300) < 0.02
    assert permutation_pvalue(s, rng.standard_normal(1500), 7, n_perm=300) > 0.05


def test_deflated_sharpe_penalises_many_trials_and_pbo_of_pure_noise_is_near_half():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0006, 0.01, 1000)
    assert deflated_sharpe(r, 1) > deflated_sharpe(r, 50)
    assert 0.3 < pbo(rng.standard_normal((1000, 20)), n_blocks=8) < 0.7


def test_constant_maturity_iv_interpolates_variance_using_calls_and_puts():
    ts = pd.Timestamp("2025-03-03 15:45")
    rows = []
    for exp, iv in (("2025-03-08", 0.20), ("2025-03-15", 0.22)):          # 5 and 12 calendar DTE
        for cp in "CP":
            rows.append(dict(ts=ts, underlying="SPY", underlying_price=500.0, strike=500.0, expiration=pd.Timestamp(exp),
                             cp=cp, bid=4.9, ask=5.0, iv=iv, volume=10, open_interest=100))
    out = constant_maturity_iv(pd.DataFrame(rows), targets=(7, 14))
    w = 0.20**2 * 5 / 365 + (0.22**2 * 12 / 365 - 0.20**2 * 5 / 365) * (7 - 5) / (12 - 5)
    assert np.isclose(out["iv_7d"].iloc[0], np.sqrt(w * 365 / 7))
    assert np.isnan(out["iv_14d"].iloc[0])                                 # 14d is not bracketed: no extrapolation


def test_live_latest_forecast_equals_the_backtest_series_value():
    """Same engine live and in research: the live helper on data up to t equals the full-history series at t."""
    r = _returns(1400)
    full = F.build_forecasts(r, 7)
    live = F.latest_forecast(r.iloc[:1250], 7)
    assert np.isclose(live["selected"], full["selected"].iloc[1249], rtol=1e-9)


def test_regime_selection_only_uses_realized_errors_and_names_a_model():
    fc = F.build_forecasts(_returns(1600), 7)
    assert fc["selected"].dropna().shape[0] > 300
    assert set(fc["chosen"].dropna().unique()) <= {"ewma", "har", "garch", "har_regime", "equal_weight"}
