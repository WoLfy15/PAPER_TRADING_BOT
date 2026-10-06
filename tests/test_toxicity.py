"""
Unit tests for research/toxicity.py -- run with `pytest tests/ -v`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from research.toxicity import MIN_BARS, assess_flow_toxicity


def _bars(prices: np.ndarray, volumes: np.ndarray) -> list[dict]:
    return [{"c": float(p), "v": float(v)} for p, v in zip(prices, volumes)]


def test_toxicity_skips_gracefully_below_min_bars():
    result = assess_flow_toxicity(_bars(np.full(10, 100.0), np.full(10, 1000.0)))
    assert result.available is False
    assert "bars" in result.notes[0]


def test_toxicity_skips_gracefully_on_zero_volume():
    n = MIN_BARS + 10
    result = assess_flow_toxicity(_bars(np.full(n, 100.0), np.zeros(n)))
    assert result.available is False


def test_toxicity_flags_an_injected_burst_as_the_current_reading():
    """A regression test for the causal-rolling-sigma fix (see the
    module docstring): a single-whole-sample-sigma BVC implementation
    scores an injected toxic burst LOWER than quiet baseline noise on
    this exact scenario -- verified while building this module. The
    fix must score the burst as the current (most recent) reading's
    regime, not dilute/invert it."""
    rng = np.random.default_rng(3)
    n = 2000
    price = 100 + np.cumsum(rng.normal(0, 0.015, n))
    volume = rng.lognormal(mean=7.5, sigma=0.6, size=n)

    burst = slice(n - 80, n - 10)
    price_burst = price.copy()
    price_burst[burst] = price[burst] + np.cumsum(np.full(70, 0.08))
    volume_burst = volume.copy()
    volume_burst[burst] = volume[burst] * 4

    normal = assess_flow_toxicity(_bars(price, volume))
    toxic = assess_flow_toxicity(_bars(price_burst, volume_burst))

    assert normal.available and toxic.available
    assert normal.regime == "Normal"
    assert toxic.regime == "Toxic"
    assert toxic.current_vpin > normal.current_vpin
    assert toxic.size_multiplier < normal.size_multiplier
    assert toxic.extra_limit_buffer_bps > normal.extra_limit_buffer_bps


def test_toxicity_false_positive_rate_under_pure_noise_is_low():
    """VPIN's own well-documented noise floor (~0.5, see module
    docstring) means occasional false "Toxic" reads under pure noise are
    expected at a 97th-percentile threshold -- but should stay rare."""
    n = 2000
    regimes = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        price = 100 + np.cumsum(rng.normal(0, 0.015, n))
        volume = rng.lognormal(mean=7.5, sigma=0.6, size=n)
        result = assess_flow_toxicity(_bars(price, volume))
        regimes.append(result.regime)
    counts = Counter(regimes)
    assert counts["Normal"] >= 12  # at most ~1-in-5 false "Elevated"/"Toxic" reads under pure noise
