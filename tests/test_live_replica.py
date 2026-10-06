import dataclasses

import numpy as np
import pandas as pd

from backtest.live_replica import ReplicaConfig, index_chain, run_replica, summarize
from config import SETTINGS
from research.edge.synth import synth_chain


def _world(n=150, seed=4, iv=0.20, vol=0.007):
    rng = np.random.default_rng(seed)
    px = pd.Series(500 * np.exp(np.cumsum(rng.normal(0.0003, vol, n))), index=pd.bdate_range("2025-01-02", periods=n))
    ch = synth_chain(px, pd.Series(iv, index=px.index), seed=seed, weekly=True)
    # a long warm-up of returns so forward_scenarios has >= 50 realized windows
    sig = pd.DataFrame({"mu": 0.02, "p": 0.80, "exp_net": 0.05, "sigma_net": 0.04, "sigma_err": 0.02, "spread_pct": 0.03, "iv": iv, "regime": "calm",
                        "z_vrp": 2.0, "z_carry": 0.0, "z_skew": 0.0, "z_jump": 0.0, "w_z_vrp": 1.0, "w_z_carry": 0.0, "w_z_skew": 0.0, "w_z_jump": 0.0, "ic": 0.10}, index=px.index)
    return px, index_chain(ch), sig


def _settings():
    return dataclasses.replace(SETTINGS, symbol="SPY", dry_run=True)


def test_replica_trades_through_the_live_exit_and_pacing_code_and_books_balance():
    px, snaps, sig = _world()
    sig = sig.iloc[:]                                              # signal on from day 0
    res = run_replica(px, snaps, sig, ReplicaConfig(hedge=False), _settings())
    t, eq = res["trades"], res["equity"]
    assert len(t) >= 1, res["diag"]
    assert set(t["reason"]) <= {"take_profit", "trailing_stop", "stop_loss", "near_expiry", "expired", "end_of_test"}
    assert (t["entry_date"] < t["exit_date"]).all()
    # with hedging off and everything closed, equity change == sum of net trade P&L (fills, commissions included)
    assert np.isclose(eq.iloc[-1] - 100_000.0, t["pnl"].sum(), atol=1e-6)
    # live pacing: a stop-out / trailing exit blocks re-entry for 24h, other exits for 30 min -> never two open at once
    assert not (t["entry_date"].iloc[1:].to_numpy() < t["exit_date"].iloc[:-1].to_numpy()).any()


def test_wider_spreads_hurt_and_a_dead_signal_never_trades():
    px, snaps, sig = _world()
    base = summarize(run_replica(px, snaps, sig, ReplicaConfig(hedge=False), _settings()))
    stress = summarize(run_replica(px, snaps, sig, ReplicaConfig(hedge=False, spread_mult=2.0), _settings()))
    assert stress.get("total_return", 0) <= base["total_return"]
    dead = run_replica(px, snaps, sig.assign(p=0.30), ReplicaConfig(hedge=False), _settings())
    assert dead["trades"].empty and dead["diag"].get("meta_p_below_threshold", 0) > 0


def test_replica_with_hedging_runs_and_respects_the_entry_window_clock():
    px, snaps, sig = _world(50)
    res = run_replica(px, snaps, sig, ReplicaConfig(hedge=True), _settings())
    assert len(res["equity"]) == 49 and np.isfinite(res["equity"]).all()
    from us_time import us_now
    assert abs((us_now().date() - pd.Timestamp.now().date()).days) <= 1      # the simulated clock was released


def test_theta_income_no_longer_blocks_short_vol_but_paying_decay_beyond_the_cap_still_does():
    from config import RiskLimits
    from strategy.portfolio import PortfolioSnapshot, remaining_greek_headroom
    inc = PortfolioSnapshot(equity=100_000, cash=100_000, equity_shares_held=0.0, net_theta_dollars=12_000.0)
    bleed = PortfolioSnapshot(equity=100_000, cash=100_000, equity_shares_held=0.0, net_theta_dollars=-12_000.0)
    assert remaining_greek_headroom(inc, RiskLimits())["theta_dollars"] > 0 > remaining_greek_headroom(bleed, RiskLimits())["theta_dollars"]


def test_realized_book_vol_needs_history_and_is_annualized():
    from strategy.portfolio import realized_book_vol
    rng = np.random.default_rng(0)
    eq = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))
    curve = [{"t": f"2026-09-{d + 1:02d}T20:00:00Z", "equity": float(e)} for d, e in enumerate(eq)]
    assert realized_book_vol(curve[:5]) is None
    assert 0.05 < realized_book_vol(curve) < 0.40


def test_sign_conventions_quiet_market_condors_win_violent_market_loses():
    """Mechanics check: implied vol 20% on a market that barely moves must make money; a violent market must lose."""
    px, snaps, sig = _world(150, iv=0.20, vol=0.0006)
    calm = summarize(run_replica(px, snaps, sig, ReplicaConfig(hedge=False), _settings()))
    px2, snaps2, sig2 = _world(150, iv=0.20, vol=0.03)
    wild = summarize(run_replica(px2, snaps2, sig2, ReplicaConfig(hedge=False), _settings()))
    assert calm["n_trades"] >= 3 and calm["total_return"] > 0 and calm["win_rate"] >= 0.7, calm
    assert wild.get("total_return", 0) < calm["total_return"]
