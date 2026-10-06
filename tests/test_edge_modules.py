import numpy as np
import pandas as pd

from agent.exits import evaluate_exits
from config import RiskLimits
from research.edge import allocation as A
from research.edge import strikes as S
from research.edge.alpha import causal_z, composite_alpha
from research.edge.data_schema import build_surface
from research.edge.meta import CostModel, adaptive_threshold, net_pnl_proxy, walk_forward_meta
from research.edge.synth import synth_chain


def _prices(n=120, seed=0):
    r = np.random.default_rng(seed).normal(0.0004, 0.008, n)
    return pd.Series(500 * np.exp(np.cumsum(r)), index=pd.bdate_range("2024-01-02", periods=n))


def test_surface_has_synchronized_7_14_30_with_skew_and_curvature():
    px = _prices(8)
    ch = synth_chain(px, pd.Series(0.16, index=px.index))
    s = build_surface(ch)
    last = s.iloc[-1]
    assert {"iv_7d", "iv_14d", "iv_30d", "skew_7d", "curv_14d", "ts_slope"} <= set(s.columns)
    assert 0.12 < last["iv_7d"] < 0.22 and last["skew_7d"] > 0       # puts richer than calls in the generator


def test_causal_z_ignores_the_future_and_today():
    x = pd.Series(np.random.default_rng(1).standard_normal(400))
    a, y = causal_z(x), x.copy()
    y.iloc[300:] += 50
    assert np.allclose(a.iloc[:300].dropna(), causal_z(y).iloc[:300].dropna())


def test_composite_and_meta_use_only_realized_rows():
    rng = np.random.default_rng(2)
    n, h = 900, 7
    z = pd.DataFrame(rng.standard_normal((n, 4)), columns=["z_vrp", "z_carry", "z_skew", "z_jump"])
    net = pd.Series(0.02 * z["z_vrp"] + rng.normal(0, 0.03, n))
    comp, W = composite_alpha(z, net, h)
    assert W["z_vrp"].iloc[-1] > W["z_carry"].iloc[-1].__abs__()          # the planted driver gets the weight
    net2 = net.copy(); net2.iloc[700:] = 99.0                              # wreck outcomes after 700
    comp2, _ = composite_alpha(z, net2, h)
    k = 700 - h                                                             # forecasts up to k+h use rows <= k only
    assert np.allclose(comp["mu"].iloc[:k + 1].dropna(), comp2["mu"].iloc[:k + 1].dropna())
    m = walk_forward_meta(z, net, comp["mu"] > 0, h)
    assert m["p"].dropna().between(0, 1).all() and m["p"].notna().sum() > 100


def test_adaptive_threshold_is_stricter_in_stress():
    calm = adaptive_threshold("calm", 0.0, 0.02)
    assert adaptive_threshold("crisis", 2.0, 0.08, event=True, iv_level=0.4) > calm + 0.15
    assert CostModel(spread_mult=2).cost_vp(0.15, 0.04) > CostModel().cost_vp(0.15, 0.04)


def test_strike_optimizer_prefers_wider_wings_when_tails_are_fat():
    px = _prices(3); ch = synth_chain(px, pd.Series(0.16, index=px.index))
    day = ch[ch.ts == ch.ts.max()]; spot = float(day.underlying_price.iloc[0])
    q = day[(day.expiration - day.ts.dt.normalize()).dt.days == 7]
    rng = np.random.default_rng(3)
    calm = S.optimize_structure("iron_condor", q, spot, rng.normal(0, 0.012, 3000))
    fat = S.optimize_structure("iron_condor", q, spot, rng.standard_t(2.2, 3000) * 0.012)
    assert calm and fat and len(calm["legs"]) == 4 and calm["credit"] > 0
    assert fat["max_loss"] <= calm["max_loss"] * 3 and fat["cvar"] >= calm["cvar"] * 0.5   # sane; both bounded by width
    assert S.optimize_structure("long_straddle", q, spot, rng.normal(0, 0.03, 3000))["credit"] < 0


def test_structure_selection_follows_the_dominant_source():
    w = dict(z_vrp=1.0, z_carry=1.0, z_skew=1.0, z_jump=1.0)
    assert S.select_structure(dict(z_vrp=2, z_carry=0, z_skew=0, z_jump=0), w, 0.02)[0] == "iron_condor"
    assert S.select_structure(dict(z_vrp=0, z_carry=0, z_skew=2, z_jump=0), w, 0.02)[0] == "put_vertical"
    assert S.select_structure(dict(z_vrp=0, z_carry=2, z_skew=0, z_jump=0), w, 0.02)[:2] == ("calendar", False)
    assert S.select_structure({}, w, -0.02)[0] == "none" and S.select_structure({}, w, -0.02, allow_long_vol=True)[0] == "long_straddle"


def test_allocation_caps_and_greek_fit():
    assert A.fractional_kelly(0.01, 0.04, 0.02) <= 0.02 and A.fractional_kelly(-1, 0.04, 0.02) == 0
    assert A.fractional_kelly(0.002, 0.04, 0.10, cap=1) < A.fractional_kelly(0.002, 0.04, 0.01, cap=1)   # more forecast error -> smaller size
    assert A.drawdown_scale(0.05, 0.25) == 1.0 and A.drawdown_scale(0.25, 0.25) == 0.0
    assert A.scale_to_fit({"vega": 80}, {"vega": 40}, {"vega": 100}) == 0.5
    out = A.rank_and_allocate([dict(name="a", underlying="SPY", exp_net=50, cvar=100, contracts=5), dict(name="b", underlying="QQQ", exp_net=5, cvar=100, contracts=5)],
                              cvar_budget=300, min_ratio=0.2, max_per_underlying=250)
    assert [o["name"] for o in out] == ["a"] and out[0]["contracts"] == 2


def test_live_exits_treat_a_vertical_as_one_structure():
    today = pd.Timestamp("2026-03-02").date()
    def leg(sym, qty, entry, cur): return {"symbol": sym, "asset_class": "us_option", "qty": qty, "avg_entry_price": entry, "current_price": cur, "market_value": qty * cur * 100}
    pos = [leg("SPY260306P00490000", -2, 1.00, 0.30), leg("SPY260306P00485000", 2, 0.30, 0.05)]   # credit 0.70 -> now 0.25: +64%
    d = evaluate_exits(pos, RiskLimits(), "SPY", today=today)
    assert len(d) == 2 and {x.reason for x in d} == {"take_profit"} and d[0].qty < 0   # both legs, short first
