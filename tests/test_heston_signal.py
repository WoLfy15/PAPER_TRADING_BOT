"""
Unit tests for research/heston_signal.py -- run with `pytest tests/ -v`.
Builds a realistic multi-expiry IV surface by round-tripping through the
vendored BSM pricer (price -> mid -> re-solved IV), the same approach
research/heston_signal.py's own module docstring warns is NECESSARY:
tests/test_signal_and_sizing.py's own `_synthetic_chain` fixture (flat
dollar extrinsic value regardless of expiry) is fine for strike-selection
tests but produces an economically nonsensical IV surface (100%+ IVs on
the near-dated leg) that a stochastic-vol calibrator should not be
expected to fit sensibly -- do not reuse it here.
"""
from __future__ import annotations

import datetime as dt

from data.market_data import ChainContract
from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton
from research.heston_signal import MIN_USABLE_POINTS, assess_heston_cross_check

_bsm = BlackScholesMerton()


def _realistic_chain(spot: float, expiries_ivs, r: float = 0.045) -> list[ChainContract]:
    """bid/ask rounded to 4dp, not the usual 2dp: BSM IV inversion on a
    short-dated, moderately-ITM strike is numerically thin on time value,
    so 1-cent price rounding alone was enough to turn a clean 19%
    synthetic input into a ~27% recovered IV outlier when this fixture
    was first written with 2dp rounding -- a real, well-known IV-
    inversion sensitivity (thin extrinsic value / large IV swing per
    cent of price), not something worth re-testing here. 4dp keeps the
    fixture's OWN precision loss below what would meaningfully affect
    the calibration quality this test is actually checking."""
    chain = []
    for expiry, T, atm_iv, skew in expiries_ivs:
        for k in range(70, 131, 5):
            iv = max(atm_iv + skew * (k - spot) + 0.00003 * (k - spot) ** 2, 0.03)
            q = OptionQuote(underlying="TST", spot=spot, strike=float(k), expiry_years=T,
                             option_type=OptionType.CALL, risk_free_rate=r)
            price = _bsm.price(q, iv)
            chain.append(ChainContract(
                symbol=f"TST{expiry.strftime('%y%m%d')}C{int(k * 1000):08d}",
                underlying="TST", strike=float(k), expiry=expiry, option_type=OptionType.CALL,
                open_interest=100, tradable=True, bid=round(price - 0.001, 4), ask=round(price + 0.001, 4), last=price,
                alpaca_iv=None, alpaca_delta=None, alpaca_gamma=None, alpaca_theta=None, alpaca_vega=None,
            ))
    return chain


def test_heston_cross_check_fits_a_realistic_surface():
    spot = 100.0
    today = dt.date.today()
    expiries_ivs = [
        (today + dt.timedelta(days=7), 7 / 365, 0.18, -0.0007),
        (today + dt.timedelta(days=30), 30 / 365, 0.195, -0.0005),
    ]
    chain = _realistic_chain(spot, expiries_ivs)
    result = assess_heston_cross_check(
        chain, expiries_ivs[0][0], spot, 0.045, live_atm_iv=0.18, forecast_vol_now=0.21,
    )
    assert result.available is True
    assert result.n_expiries == 2
    # NOT a tight tolerance: even at 4dp price precision, BSM IV inversion
    # on a moderately-ITM 7-DTE strike (thin extrinsic value relative to
    # intrinsic) recovers a few points off the intended synthetic IV --
    # verified directly while writing this test (K=90 here: 19.00%
    # intended vs. 20.57% recovered) -- a real numerical characteristic
    # of short-dated ITM IV inversion, not a fixture bug. The module's
    # own MAX_ACCEPTABLE_RMSE_VOL_PTS=3.0 tolerance is what actually
    # gates data_quality_ok in production; 0.15 here just confirms the
    # fit stays well within an order of magnitude of a genuinely usable
    # calibration despite that noisy point, not that it's noise-free.
    assert result.rmse_vol_pts is not None and result.rmse_vol_pts < 0.15
    assert result.data_quality_ok is True
    # heston_atm_iv should land close to the raw print for a smooth, well-behaved surface
    assert result.heston_atm_iv is not None and abs(result.heston_atm_iv - 0.18) < 0.03
    assert result.term_gap is not None
    assert result.vol_of_vol_regime in ("Stable", "Elevated", "Fragile")
    assert 0.0 < result.size_multiplier <= 1.0


def test_heston_cross_check_gracefully_skips_a_thin_chain():
    spot = 100.0
    today = dt.date.today()
    expiry = today + dt.timedelta(days=7)
    chain = _realistic_chain(spot, [(expiry, 7 / 365, 0.18, -0.0007)])[: MIN_USABLE_POINTS - 1]
    result = assess_heston_cross_check(chain, expiry, spot, 0.045, live_atm_iv=0.18, forecast_vol_now=0.21)
    assert result.available is False
    assert result.notes  # explains why


def test_heston_cross_check_flags_a_stale_atm_print():
    """If the ATM print disagrees materially with what the rest of the
    smile implies, atm_gap_vol_pts should be large and data_quality_ok
    should reflect that -- this is the module's core data-quality job."""
    spot = 100.0
    today = dt.date.today()
    expiries_ivs = [
        (today + dt.timedelta(days=7), 7 / 365, 0.18, -0.0007),
        (today + dt.timedelta(days=30), 30 / 365, 0.195, -0.0005),
    ]
    chain = _realistic_chain(spot, expiries_ivs)
    # live_atm_iv reported far away from the surface's own ATM level (0.18)
    result = assess_heston_cross_check(
        chain, expiries_ivs[0][0], spot, 0.045, live_atm_iv=0.45, forecast_vol_now=0.21,
    )
    assert result.available is True
    assert result.atm_gap_vol_pts is not None
    assert abs(result.atm_gap_vol_pts) > 0.10
    assert result.data_quality_ok is False
