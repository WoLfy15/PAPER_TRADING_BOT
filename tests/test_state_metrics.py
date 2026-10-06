"""
Unit tests for agent/state.py's v2 equity-curve + risk-metrics additions
-- run with `pytest tests/ -v`.
"""
from __future__ import annotations

import datetime as dt
import random

from agent.state import AgentState, _periods_per_year, compute_risk_metrics, record_equity


def test_compute_risk_metrics_returns_none_not_zero_when_young():
    state = AgentState()
    metrics = compute_risk_metrics(state)
    assert metrics["sharpe"] is None
    assert metrics["n_observations"] == 0

    record_equity(state, dt.datetime(2026, 1, 1).isoformat(), 1_000_000.0)
    record_equity(state, dt.datetime(2026, 1, 1, 0, 15).isoformat(), 1_000_500.0)
    metrics = compute_risk_metrics(state, min_points=5)
    assert metrics["sharpe"] is None  # still below min_points
    assert metrics["n_observations"] == 2


def test_compute_risk_metrics_matches_a_hand_computed_sharpe():
    random.seed(0)
    t0 = dt.datetime(2025, 1, 1, 16, 0, 0)
    equity = 1_000_000.0
    equities = [equity]
    for _ in range(1, 60):
        equity *= 1 + random.gauss(0.0005, 0.01)
        equities.append(equity)

    state = AgentState()
    for i, eq in enumerate(equities):
        record_equity(state, (t0 + dt.timedelta(days=i)).isoformat(), eq)

    metrics = compute_risk_metrics(state)
    returns = [(b - a) / a for a, b in zip(equities, equities[1:])]
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = var_r ** 0.5
    expected_sharpe = (mean_r / std_r) * (252 ** 0.5)
    assert abs(metrics["sharpe"] - expected_sharpe) < 1e-9
    assert -1.0 <= metrics["win_rate"] <= 1.0
    assert metrics["max_drawdown"] <= 0.0


def test_periods_per_year_infers_intraday_vs_daily_spacing():
    t0 = dt.datetime(2026, 1, 1, 14, 0, 0)
    intraday = [(t0 + dt.timedelta(minutes=15 * i)).isoformat() for i in range(20)]
    daily = [(t0 + dt.timedelta(days=i)).isoformat() for i in range(20)]
    assert _periods_per_year(intraday) > 5000
    assert 200 < _periods_per_year(daily) < 300
