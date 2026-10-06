"""
tests/test_api_server.py -- v3's new/fixed FastAPI endpoints, exercised
through FastAPI's own TestClient against a mocked AlpacaClient (no real
network calls). Covers the two real bugs found while building the
dashboard against this file (see api/server.py's module docstring):
/positions calling build_portfolio_snapshot with its old, now-changed
signature, and /run + /run-all silently dropping every Settings field
added since they were written.
"""
from __future__ import annotations

import datetime as dt
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPACA_API_KEY", "PKtest0000000000000")
os.environ.setdefault("ALPACA_SECRET_KEY", "x" * 40)

import pytest
from fastapi.testclient import TestClient

import api.server as server_mod


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_account.return_value = {"equity": "100000", "cash": "50000", "buying_power": "200000"}
    client.get_clock.return_value = {"is_open": True, "next_open": "2026-09-22T13:30:00Z", "next_close": "2026-09-21T20:00:00Z"}
    client.get_positions.return_value = []
    client.get_mid_price.return_value = 500.0
    client.get_orders.return_value = []
    today = dt.date.today()
    client.get_stock_bars.return_value = [
        {"t": (today - dt.timedelta(days=i)).isoformat() + "T00:00:00Z", "o": 500.0, "h": 502.0, "l": 498.0, "c": 501.0, "v": 1_000_000}
        for i in range(30, 0, -1)
    ]
    return client


@pytest.fixture
def api_client(mock_client):
    with patch.object(server_mod, "_client", return_value=mock_client):
        yield TestClient(server_mod.app)


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_positions_uses_new_build_portfolio_snapshot_signature(api_client):
    """Regression test for the exact bug found while wiring the dashboard:
    /positions used to call build_portfolio_snapshot(client, ...) after
    that function's signature changed to take pre-fetched account/
    positions dicts instead -- this would have raised a TypeError."""
    r = api_client.get("/positions")
    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == 100_000.0
    assert body["positions"] == []


def test_orders_endpoint_is_read_only(api_client, mock_client):
    r = api_client.get("/orders")
    assert r.status_code == 200
    assert mock_client.cancel_order.call_count == 0, "GET /orders must never cancel anything -- see its own docstring"


def test_risk_endpoint_reports_greek_headroom(api_client):
    r = api_client.get("/risk")
    assert r.status_code == 200
    body = r.json()
    assert "greek_headroom" in body
    assert body["kill_switch"] is False  # fresh state, no drawdown yet


def test_calendar_endpoint_returns_real_verified_events(api_client):
    r = api_client.get("/calendar?horizon_days=400")
    assert r.status_code == 200
    events = r.json()["events"]
    assert any(e["label"] == "FOMC" for e in events), "should find at least one real FOMC date within 400 days of any date"


def test_chart_bars_endpoint(api_client):
    r = api_client.get("/chart/bars?symbol=SPY&days=30")
    assert r.status_code == 200
    assert len(r.json()["bars"]) == 30


def test_dashboard_route_serves_html(api_client):
    r = api_client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "addCandlestickSeries" in r.text


def test_run_endpoint_carries_over_settings_fields_not_just_symbol_and_dry_run(api_client):
    """Regression test for the exact bug found in /run: it used to
    hand-construct Settings(api_key=..., api_secret=..., data_feed=...,
    symbol=..., dry_run=...) field by field, so every field added to
    Settings after that code was written silently reverted to its
    default instead of inheriting from the real SETTINGS. This asserts
    a non-default field on the real SETTINGS actually reaches run_cycle."""
    import dataclasses

    from api.server import SETTINGS
    custom_settings = dataclasses.replace(SETTINGS, short_leg_target_delta=0.199)

    captured = {}

    def fake_run_cycle(settings, client=None):
        captured["short_leg_target_delta"] = settings.short_leg_target_delta
        from agent.loop import AgentRunResult
        return AgentRunResult(
            timestamp_utc="2026-01-01T00:00:00", symbol="SPY", spot=500.0, regime_label="Range",
            regime_confidence=1.0, gamma_regime="SHORT_GAMMA", forecast_vol=0.15, live_atm_iv=0.15,
            vrp_raw=0.0, vrp_z=0.0, sas_mean=None, composite_edge=0.0, signal_rationale="test",
            kelly_fraction=0.0, direction="flat", portfolio_delta_shares=0.0, portfolio_gamma_shares=0.0,
            portfolio_vega_dollars=0.0, portfolio_theta_dollars=0.0, equity=100000.0, drawdown_pct=0.0,
            kill_switch=False,
        )

    with patch.object(server_mod, "SETTINGS", custom_settings), patch.object(server_mod, "run_cycle", fake_run_cycle):
        client = TestClient(server_mod.app)
        r = client.post("/run", json={"dry_run": True})
        assert r.status_code == 200
        assert captured["short_leg_target_delta"] == pytest.approx(0.199), (
            "a non-default Settings field failed to reach run_cycle -- /run is back to hand-reconstructing Settings"
        )
