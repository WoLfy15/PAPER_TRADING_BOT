"""
tests/test_exits.py -- agent/exits.py had NO test coverage in v2, which
is how the per-leg stop-loss on an iron condor's protective wings went
unnoticed (see that module's docstring for the measured numbers). These
tests pin the v3 structure-level behavior AND the v2 single-leg
behavior it deliberately preserves.
"""
from __future__ import annotations

import datetime as dt

import pytest

from agent.exits import evaluate_exits
from config import RiskLimits

TODAY = dt.date(2026, 10, 5)
EXP = dt.date(2026, 10, 12)          # 7 DTE from TODAY
RISK = RiskLimits(exit_condor_stop_loss_pct=0.35)   # the original -35% condor stop, kept so these tests keep their meaning
RISK_V42 = RiskLimits()               # v4.2 defaults: condor stop -100% of credit, TP +50%, trailing stop, dte_close 2


def sym(strike: float, cp: str, expiry: dt.date = EXP, root: str = "SPY") -> str:
    return f"{root}{expiry.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


def pos(symbol: str, qty: float, entry: float, current: float) -> dict:
    return {
        "symbol": symbol, "qty": str(qty), "avg_entry_price": str(entry), "current_price": str(current),
        "market_value": str(current * qty * 100),
    }


def condor(n: int, prices: dict[str, tuple[float, float]], expiry: dt.date = EXP) -> list[dict]:
    """prices: {'lp': (entry, current), 'sp': ..., 'sc': ..., 'lc': ...}"""
    return [
        pos(sym(470, "P", expiry), +n, *prices["lp"]),
        pos(sym(485, "P", expiry), -n, *prices["sp"]),
        pos(sym(515, "C", expiry), -n, *prices["sc"]),
        pos(sym(530, "C", expiry), +n, *prices["lc"]),
    ]


# ------------------------------------------------ v2 single-leg parity ---
def test_single_long_call_take_profit():
    d = evaluate_exits([pos(sym(500, "C"), 2, 5.0, 8.0)], RISK, "SPY", today=TODAY)   # +60%
    assert [(x.reason, round(x.pnl_pct, 2)) for x in d] == [("take_profit", 0.60)]


def test_single_long_call_stop_loss():
    d = evaluate_exits([pos(sym(500, "C"), 2, 5.0, 3.0)], RISK, "SPY", today=TODAY)   # -40%
    assert [(x.reason, round(x.pnl_pct, 2)) for x in d] == [("stop_loss", -0.40)]


def test_single_short_put_profit_sign_is_inverted():
    d = evaluate_exits([pos(sym(480, "P"), -3, 2.0, 0.8)], RISK, "SPY", today=TODAY)  # price -60% => +60% for the short
    assert [(x.reason, round(x.pnl_pct, 2)) for x in d] == [("take_profit", 0.60)]


def test_single_leg_near_expiry():
    d = evaluate_exits([pos(sym(500, "C", TODAY + dt.timedelta(days=1)), 1, 5.0, 5.0)], RISK, "SPY", today=TODAY)
    assert [x.reason for x in d] == ["near_expiry"]


def test_other_underlyings_and_non_options_ignored():
    positions = [pos(sym(500, "C", root="QQQ"), 1, 5.0, 9.0), {"symbol": "SPY", "qty": "100", "avg_entry_price": "500", "current_price": "510", "market_value": "51000"}]
    assert evaluate_exits(positions, RISK, "SPY", today=TODAY) == []


# ----------------------------------------- the regression that mattered ---
def test_winning_condor_wings_do_not_trigger_stop_loss_on_day_one():
    """THE regression. Prices are the measured day-1 values for a default
    weekly condor in a perfectly quiet market (best case for the trade):
    wings -57%/-54% (v2 would stop them out at -35%), shorts only +27%
    (below the +50% take-profit). At the STRUCTURE level the condor is up
    ~26% of its credit -- no exit should fire at all."""
    prices = {"lp": (0.007, 0.003), "sp": (0.388, 0.283), "sc": (0.517, 0.377), "lc": (0.017, 0.0078)}
    d = evaluate_exits(condor(5, prices), RISK, "SPY", today=TODAY)
    assert d == [], f"protective wings must not be stopped out while the structure is winning: {d}"


def test_condor_take_profit_closes_all_four_legs_shorts_first():
    # Shorts decayed most of the way: structure well past +50% of credit.
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.06), "sc": (0.517, 0.08), "lc": (0.017, 0.002)}
    d = evaluate_exits(condor(5, prices), RISK, "SPY", today=TODAY)
    assert len(d) == 4
    assert {x.reason for x in d} == {"take_profit"}
    # ordering: both short legs (qty<0) are submitted before either long wing
    assert [x.qty < 0 for x in d] == [True, True, False, False]
    assert len({round(x.pnl_pct, 6) for x in d}) == 1   # every leg carries the STRUCTURE's pnl


def test_condor_stop_loss_when_the_structure_actually_loses():
    # Underlying ran at the short put. Hand-computed structure P&L (per share x qty 5):
    #   long put  (0.05-0.007)*(+5)  = +0.215      short put  (1.25-0.388)*(-5) = -4.310
    #   short call (0.10-0.517)*(-5) = +2.085      long call  (0.002-0.017)*(+5) = -0.075
    #   net = -2.085 ; credit basis = 5*(0.388+0.517-0.007-0.017) = 4.405  =>  -47.3% (past the -35% stop)
    prices = {"lp": (0.007, 0.05), "sp": (0.388, 1.25), "sc": (0.517, 0.10), "lc": (0.017, 0.002)}
    d = evaluate_exits(condor(5, prices), RISK, "SPY", today=TODAY)
    assert len(d) == 4 and {x.reason for x in d} == {"stop_loss"}
    assert [x.qty < 0 for x in d] == [True, True, False, False]
    assert round(d[0].pnl_pct, 2) == -0.47


def test_condor_loss_just_inside_the_stop_does_not_exit():
    """Boundary partner of the test above: same trade with the short put
    at 1.10 instead of 1.25 nets -1.335 on a 4.405 credit = -30.3%, which
    is inside the -35% stop -- structure-level P&L, not any single leg's
    move (the short put alone is already +183% against us), decides."""
    prices = {"lp": (0.007, 0.05), "sp": (0.388, 1.10), "sc": (0.517, 0.10), "lc": (0.017, 0.002)}
    assert evaluate_exits(condor(5, prices), RISK, "SPY", today=TODAY) == []


def test_condor_near_expiry_closes_all_legs():
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.20), "sc": (0.517, 0.30), "lc": (0.017, 0.001)}
    near = TODAY + dt.timedelta(days=1)
    d = evaluate_exits(condor(2, prices, expiry=near), RISK, "SPY", today=TODAY)
    assert len(d) == 4 and {x.reason for x in d} == {"near_expiry"}


# ----------------------------------------------------------- straddle ---
def test_straddle_is_judged_on_net_pnl_not_each_leg():
    """Call +80%, put -60% -- v2 would emit BOTH a take_profit (call) and
    a stop_loss (put) and close the whole straddle piecemeal. Net of the
    two legs the straddle is only +10% of its debit: hold."""
    positions = [pos(sym(500, "C"), 3, 5.0, 9.0), pos(sym(500, "P"), 3, 5.0, 2.0)]
    assert evaluate_exits(positions, RISK, "SPY", today=TODAY) == []


def test_straddle_stop_loss_on_net_debit():
    positions = [pos(sym(500, "C"), 3, 5.0, 3.0), pos(sym(500, "P"), 3, 5.0, 3.5)]   # -35%... just past: 6.5/10 => -35%
    positions[1]["current_price"] = "3.4"                                            # 6.4/10 => -36%
    d = evaluate_exits(positions, RISK, "SPY", today=TODAY)
    assert len(d) == 2 and {x.reason for x in d} == {"stop_loss"}


# --------------------------------------------- mixed / degenerate books ---
def test_straddle_and_condor_at_same_expiry_are_evaluated_independently():
    """Plausible in the default weekly config (every cycle picks the same
    nearest expiry): a losing straddle must not drag a healthy condor out
    with it, or vice versa."""
    winning_condor = condor(5, {"lp": (0.007, 0.003), "sp": (0.388, 0.283), "sc": (0.517, 0.377), "lc": (0.017, 0.0078)})
    losing_straddle = [pos(sym(500, "C"), 2, 5.0, 2.0), pos(sym(500, "P"), 2, 5.0, 2.0)]   # -60%
    d = evaluate_exits(winning_condor + losing_straddle, RISK, "SPY", today=TODAY)
    assert {x.symbol for x in d} == {sym(500, "C"), sym(500, "P")}
    assert {x.reason for x in d} == {"stop_loss"}


def test_condor_with_unequal_leg_counts_falls_back_to_per_leg_rule():
    """Can't be safely grouped (e.g. a partial fill left the legs uneven)
    -- must not be guessed at. Falls back to v2's per-leg behavior."""
    positions = condor(5, {"lp": (0.007, 0.003), "sp": (0.388, 0.283), "sc": (0.517, 0.377), "lc": (0.017, 0.0078)})
    positions[0]["qty"] = "3"    # long put wing now 3 vs 5 shorts
    d = evaluate_exits(positions, RISK, "SPY", today=TODAY)
    # the ungroupable long wings are judged alone: both are far below -35%
    assert {x.reason for x in d} == {"stop_loss"}


def test_missing_quote_only_allows_near_expiry_trigger():
    prices = {"lp": (0.007, 0.0), "sp": (0.388, 0.283), "sc": (0.517, 0.377), "lc": (0.017, 0.0078)}   # long put has no quote
    assert evaluate_exits(condor(5, prices), RISK, "SPY", today=TODAY) == []
    near = TODAY + dt.timedelta(days=1)
    d = evaluate_exits(condor(5, prices, expiry=near), RISK, "SPY", today=TODAY)
    assert len(d) == 4 and {x.reason for x in d} == {"near_expiry"}


# ---------------------------------------------- agent.loop._execute_exits ---
# The order-submission side of structure-level exits: failure isolation and
# "never leave a short naked".
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.loop import _execute_exits


def _winning_condor_decisions(risk=RISK):
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.06), "sc": (0.517, 0.08), "lc": (0.017, 0.002)}   # take_profit
    return evaluate_exits(condor(5, prices), risk, "SPY", today=TODAY)


def _live_settings():
    return SimpleNamespace(dry_run=False)


def test_all_legs_close_in_order_when_everything_succeeds():
    decisions = _winning_condor_decisions()
    client = MagicMock()
    client.close_position.return_value = {"id": "ok"}
    planned, notes = _execute_exits(client, _live_settings(), decisions)
    closed = [c.args[0] for c in client.close_position.call_args_list]
    assert len(closed) == 4
    # both shorts (485P, 515C) are closed before either wing (470P, 530C)
    assert set(closed[:2]) == {sym(485, "P"), sym(515, "C")} and set(closed[2:]) == {sym(470, "P"), sym(530, "C")}
    assert all(p["submitted"] for p in planned)


def test_short_leg_failure_protects_the_wings():
    """If a short leg's close raises, its structure's protective wings
    must NOT be closed -- that would leave the surviving short naked."""
    decisions = _winning_condor_decisions()
    client = MagicMock()

    def flaky(symbol):
        if symbol == sym(485, "P"):
            raise RuntimeError("HTTP 500 upstream")
        return {"id": "ok"}
    client.close_position.side_effect = flaky

    planned, notes = _execute_exits(client, _live_settings(), decisions)
    closed = {c.args[0] for c in client.close_position.call_args_list}
    assert sym(470, "P") not in closed and sym(530, "C") not in closed, "a wing was closed while a short was still open"
    assert sum("SKIPPED closing protective long" in n for n in notes) == 2
    assert any("FAILED" in n and sym(485, "P") in n for n in notes)


def test_skipped_short_leg_no_quote_also_protects_the_wings():
    """`submit_close_position` returns skipped (not raised) when Alpaca has
    no quote -- the short is STILL OPEN, so this must block the wings too."""
    decisions = _winning_condor_decisions()
    client = MagicMock()

    def no_quote(symbol):
        if symbol == sym(515, "C"):
            raise RuntimeError("HTTP 403 {\"code\":40310000,\"message\":\"no available quote\"}")
        return {"id": "ok"}
    client.close_position.side_effect = no_quote

    _planned, _notes = _execute_exits(client, _live_settings(), decisions)
    closed = {c.args[0] for c in client.close_position.call_args_list}
    assert sym(470, "P") not in closed and sym(530, "C") not in closed


def test_one_structures_failure_does_not_block_another_structure():
    other_expiry = dt.date(2026, 10, 19)
    prices = {"lp": (0.007, 0.001), "sp": (0.388, 0.06), "sc": (0.517, 0.08), "lc": (0.017, 0.002)}
    positions = condor(5, prices) + condor(5, prices, expiry=other_expiry)
    decisions = evaluate_exits(positions, RISK, "SPY", today=TODAY)
    assert len(decisions) == 8

    client = MagicMock()

    def flaky(symbol):
        if symbol == sym(485, "P", EXP):            # only the FIRST condor's short put fails
            raise RuntimeError("boom")
        return {"id": "ok"}
    client.close_position.side_effect = flaky

    _execute_exits(client, _live_settings(), decisions)
    closed = {c.args[0] for c in client.close_position.call_args_list}
    assert {sym(485, "P", other_expiry), sym(515, "C", other_expiry), sym(470, "P", other_expiry), sym(530, "C", other_expiry)} <= closed
    assert sym(470, "P", EXP) not in closed and sym(530, "C", EXP) not in closed


def test_dry_run_never_counts_as_a_failure():
    decisions = _winning_condor_decisions()
    client = MagicMock()
    planned, _notes = _execute_exits(client, SimpleNamespace(dry_run=True), decisions)
    assert len(planned) == 4 and all(p.get("dry_run") for p in planned)
    assert client.close_position.call_count == 0


def test_v42_default_condor_stop_is_wider_than_normal_noise():
    """-47% of credit (the loss in the stop test above) is inside the v4.2 -100% stop: no exit."""
    prices = {"lp": (0.007, 0.05), "sp": (0.388, 1.25), "sc": (0.517, 0.10), "lc": (0.017, 0.002)}
    assert evaluate_exits(condor(5, prices), RISK_V42, "SPY", today=TODAY) == []


def test_trailing_stop_locks_in_gains_after_a_peak():
    prices_up = {"lp": (0.007, 0.004), "sp": (0.388, 0.25), "sc": (0.517, 0.40), "lc": (0.017, 0.010)}   # ~ +30% of credit
    prices_back = {"lp": (0.007, 0.006), "sp": (0.388, 0.36), "sc": (0.517, 0.49), "lc": (0.017, 0.015)}  # gave most of it back
    peaks: dict = {}
    assert evaluate_exits(condor(5, prices_up), RISK_V42, "SPY", today=TODAY, peaks=peaks) == []
    assert max(peaks.values()) > 0.25
    d = evaluate_exits(condor(5, prices_back), RISK_V42, "SPY", today=TODAY, peaks=peaks)
    assert len(d) == 4 and {x.reason for x in d} == {"trailing_stop"}
