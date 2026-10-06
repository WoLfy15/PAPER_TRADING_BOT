"""
tests/test_run_cycle_integration.py -- agent.loop.run_cycle end to end,
against a mocked AlpacaClient and REAL bundled S&P 500 history (so it
needs no network and no credentials).

Until v3 the whole-cycle path had no automated coverage at all -- every
end-to-end check of it was an ad hoc script. These tests pin the wiring
that unit tests can't see: the concurrent fetch phase, the event/loss/
Greek gates, the CVaR-aware condor path, and how auto-exits flow through
to submitted (dry-run) orders.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import agent.state as state_mod
from agent.loop import run_cycle
from config import Settings
from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

SP500_CSV = Path(__file__).resolve().parent.parent / "data" / "reference" / "sp500_fred.csv"
from us_time import us_today
TODAY = us_today()   # the agent runs on the New York date


def _daily_bars():
    sp = pd.read_csv(SP500_CSV, parse_dates=["observation_date"]).set_index("observation_date")["SP500"].dropna()
    recent = sp.iloc[-420:] / 10.0                                   # SPX/10 ~ SPY-like level
    dates = pd.bdate_range(end=TODAY - dt.timedelta(days=1), periods=len(recent))
    bars = [{"t": d.isoformat() + "Z", "o": float(c), "h": float(c * 1.003), "l": float(c * 0.997), "c": float(c), "v": 50_000_000}
            for d, c in zip(dates, recent.values)]
    return bars, float(recent.iloc[-1])


def _minute_bars(spot: float):
    rng = np.random.default_rng(3)
    idx = pd.date_range(end=dt.datetime.now(dt.timezone.utc), periods=390, freq="1min")
    px = spot * (1 + np.cumsum(rng.normal(0, 0.0003, 390)))
    return [{"t": d.isoformat(), "o": float(p), "h": float(p * 1.0005), "l": float(p * 0.9995), "c": float(p), "v": int(rng.integers(1000, 5000))}
            for d, p in zip(idx, px)]


def _chain(spot: float, expiry: dt.date, base_iv: float):
    bsm = BlackScholesMerton()
    meta, snaps = [], {}
    lo, hi = int(spot * 0.85 // 5 * 5), int(spot * 1.15 // 5 * 5) + 5
    for strike in range(lo, hi, 5):
        m = abs(strike - spot) / spot
        for otype, label in ((OptionType.CALL, "call"), (OptionType.PUT, "put")):
            q = OptionQuote(underlying="SPY", spot=spot, strike=float(strike), expiry_years=(expiry - TODAY).days / 365,
                            option_type=otype, risk_free_rate=0.045, market_price=None)
            fair = max(bsm.price(q, base_iv + 0.30 * m), 0.02)
            symbol = f"SPY{expiry.strftime('%y%m%d')}{'C' if label == 'call' else 'P'}{int(strike * 1000):08d}"
            meta.append({"symbol": symbol, "tradable": True, "open_interest": 500, "strike_price": str(strike),
                         "expiration_date": expiry.isoformat(), "type": label})
            snaps[symbol] = {"latestQuote": {"bp": round(fair * 0.97, 2), "ap": round(fair * 1.03, 2),
                                             "t": dt.datetime.now(dt.timezone.utc).isoformat()},
                             "latestTrade": {"p": round(fair, 2)}, "greeks": {}}
    return meta, snaps


def _occ(expiry: dt.date, cp: str, strike: float) -> str:
    return f"SPY{expiry.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


def _position(symbol: str, qty: int, entry: float, current: float) -> dict:
    return {"symbol": symbol, "qty": str(qty), "avg_entry_price": str(entry), "current_price": str(current),
            "market_value": str(current * qty * 100)}


def _condor_positions(spot: float, expiry: dt.date, prices: dict[str, tuple[float, float]], n: int = 4) -> list[dict]:
    r5 = lambda x: round(x / 5) * 5
    return [
        _position(_occ(expiry, "P", r5(spot - 40)), +n, *prices["lp"]),
        _position(_occ(expiry, "P", r5(spot - 25)), -n, *prices["sp"]),
        _position(_occ(expiry, "C", r5(spot + 25)), -n, *prices["sc"]),
        _position(_occ(expiry, "C", r5(spot + 40)), +n, *prices["lc"]),
    ]


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "STATE_PATH", tmp_path / "agent_state.json")
    import agent.loop as loop_mod
    monkeypatch.setattr(loop_mod, "in_entry_window", lambda n: True)   # tests run at any hour; the pacing gate has its own test


def _make_client(base_iv: float, positions: list[dict] | None = None):
    bars, spot = _daily_bars()
    expiry = TODAY + dt.timedelta(days=32)
    meta, snaps = _chain(spot, expiry, base_iv)
    client = MagicMock()
    client.get_account.return_value = {"equity": "100000", "cash": "50000"}
    client.get_mid_price.return_value = spot
    client.get_stock_bars.side_effect = lambda symbol, timeframe="1Day", **kw: _minute_bars(spot) if timeframe == "1Min" else bars
    client.get_option_contracts.return_value = meta
    client.get_option_chain_snapshot.return_value = snaps
    client.get_positions.return_value = positions or []
    client.get_orders.return_value = []
    return client, spot


def _settings(**overrides) -> Settings:
    import dataclasses
    return dataclasses.replace(Settings(api_key="PKtest0000000000000", api_secret="x" * 40, symbol="SPY", dry_run=True), **overrides)


def test_short_vol_cycle_plans_a_cvar_bounded_condor(isolated_state):
    client, _spot = _make_client(base_iv=0.32)               # rich IV vs ~15% forecast => short-vol
    result = run_cycle(_settings(), client=client)

    assert result.direction == "short_vol"
    kinds = [o["kind"] for o in result.planned_orders]
    assert "iron_condor" in kinds
    assert all(o.get("dry_run") for o in result.planned_orders), "dry_run must never submit"
    client.submit_order.assert_not_called()

    # CVaR budget respected (default max_structure_cvar_pct=3% of $100k)
    assert result.condor_stress_report is not None
    assert abs(result.condor_stress_report["monte_carlo_cvar_dollars"]) <= 0.03 * 100_000 + 1e-6

    # v3 fields are populated, and the horizon forecast is matched to the traded DTE
    assert result.horizon_forecast and result.horizon_forecast["horizon_days"] == 32
    assert result.historical_iv_source == "vix_spx_proxy"
    assert result.hedge_ratio_used == 1.0                    # fresh state file => documented fallback
    assert set(result.greek_headroom) == {"delta_shares", "gamma_shares", "vega_dollars", "theta_dollars"}


def test_take_profit_condor_exits_as_one_structure(isolated_state):
    exp = TODAY + dt.timedelta(days=10)
    client, spot = _make_client(base_iv=0.15)
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.06), "sc": (0.517, 0.08), "lc": (0.017, 0.002)}   # structure well past +50%
    client.get_positions.return_value = _condor_positions(spot, exp, prices)

    result = run_cycle(_settings(), client=client)

    assert len(result.exit_decisions) == 4
    assert {d["reason"] for d in result.exit_decisions} == {"take_profit"}
    assert len({d["group"] for d in result.exit_decisions}) == 1 and result.exit_decisions[0]["group"]
    exit_orders = [o for o in result.planned_orders if o["kind"] == "exit_take_profit"]
    assert len(exit_orders) == 4 and all(o["dry_run"] for o in exit_orders)


def test_winning_condor_is_not_stopped_out_on_its_wings(isolated_state):
    """v2's per-leg rule would have ordered the two protective long wings
    closed here (they're down ~55%) while the structure is UP ~26%."""
    exp = TODAY + dt.timedelta(days=10)
    client, spot = _make_client(base_iv=0.15)
    prices = {"lp": (0.007, 0.003), "sp": (0.388, 0.283), "sc": (0.517, 0.377), "lc": (0.017, 0.0078)}
    client.get_positions.return_value = _condor_positions(spot, exp, prices)

    result = run_cycle(_settings(), client=client)
    assert result.exit_decisions == []
    assert not [o for o in result.planned_orders if o["kind"].startswith("exit_")]


def test_loss_circuit_breaker_blocks_new_entries_but_not_exits(isolated_state):
    """A tripped daily-loss breaker must stop NEW risk (both directions)
    while leaving risk-REDUCING exits fully live."""
    import json
    now = dt.datetime.utcnow()
    prior_day = (now - dt.timedelta(days=1)).isoformat()
    state_mod.STATE_PATH.write_text(json.dumps({
        "equity_high_water_mark": 100_000.0, "last_run_utc": prior_day, "decision_log": [],
        "equity_curve": [{"t": prior_day, "equity": 100_000.0}, {"t": now.isoformat(), "equity": 94_000.0}],
        "option_book_curve": [], "underlying_price_curve": [],
    }))
    exp = TODAY + dt.timedelta(days=10)
    client, spot = _make_client(base_iv=0.32)
    client.get_account.return_value = {"equity": "94000", "cash": "50000"}
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.06), "sc": (0.517, 0.08), "lc": (0.017, 0.002)}
    client.get_positions.return_value = _condor_positions(spot, exp, prices)

    result = run_cycle(_settings(), client=client)

    assert result.loss_breaker_tripped is True
    assert result.direction == "flat"
    assert not [o for o in result.planned_orders if o["kind"] in ("iron_condor", "straddle_buy")], "no NEW entry may be planned"
    assert len([o for o in result.planned_orders if o["kind"] == "exit_take_profit"]) == 4, "exits must still fire"
