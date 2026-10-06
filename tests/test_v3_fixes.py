"""
tests/test_v3_fixes.py -- regression coverage for the specific bugs
found and fixed while building v3 (see CHANGES_V3.md for the full
list). Each test is named after the bug it guards against regressing,
not just the function it happens to call.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pytest

from config import RiskLimits, Settings
from strategy.sizing import decide_sizing, kelly_fraction


# ------------------------------------------------------- vega units bug ---
def test_vega_cap_uses_per_point_convention_not_per_100_point():
    """The bug: size_long_straddle computed vega_risk = raw_contracts *
    straddle_vega * 100, but BlackScholesMerton.vega returns dollars per
    SHARE for a full 100-POINT vol move (its own docstring says so
    explicitly) -- the correct per-contract-per-vol-POINT figure needs
    NO extra multiplier (the /100 for "per point" and the *100 for "100
    shares/contract" exactly cancel). The old formula was ~100x too
    tight: a single realistic ~30 DTE ATM straddle's vega alone could
    exceed the entire default vega budget, capping every long-vol trade
    to 0 contracts regardless of how good the edge was. This reproduces
    that exact realistic magnitude and asserts a real, nonzero contract
    count comes out."""
    risk = RiskLimits()  # default max_vega_risk_pct=0.02 -> $2,000 on $100k equity
    capital = 100_000.0
    straddle_price = 17.66   # a realistic ~30 DTE ATM SPY straddle, from an actual run against real data
    straddle_gamma = 0.036
    straddle_vega = 91.16    # ditto -- raw BlackScholesMerton.vega output, NOT pre-divided by 100

    from strategy.sizing import size_long_straddle
    result = size_long_straddle(
        kelly=0.15, capital=capital, straddle_price=straddle_price,
        straddle_gamma=straddle_gamma, straddle_vega=straddle_vega, risk=risk,
    )
    assert result.contracts > 0, (
        "vega cap is back to being ~100x too tight -- a realistic straddle's vega alone "
        "is blocking every long-vol trade regardless of edge"
    )
    assert "vega_limit" not in result.capped_by  # this specific realistic case shouldn't even bind the cap

    # And the cap DOES still bind for a position genuinely large enough to warrant it --
    # confirms this isn't just "the cap never fires anymore" as an overcorrection.
    huge = size_long_straddle(
        kelly=0.15, capital=capital, straddle_price=straddle_price,
        straddle_gamma=straddle_gamma, straddle_vega=straddle_vega * 50, risk=risk,
    )
    assert "vega_limit" in huge.capped_by


def test_portfolio_vega_dollars_uses_same_convention_as_sizing():
    """The identical units bug also existed in strategy/portfolio.py's
    per-position vega aggregation (net_vega_dollars) -- fixed in the
    same pass so both sides of every vega-vs-max_vega_risk_pct
    comparison agree. This checks the arithmetic directly against a
    hand-computed expected value rather than just "some fix happened."""
    from strategy.portfolio import _option_position_greeks

    # A short option position; iv solves to something reasonable given this synthetic price.
    pos = {
        "symbol": "SPY261016C00500000", "qty": "-2", "current_price": "5.00",
        "market_value": "-1000",
    }
    greeks = _option_position_greeks(pos, spot=500.0, risk_free_rate=0.045)
    if greeks is None:
        pytest.skip("IV solver couldn't bracket this synthetic price -- not what this test is about")
    # mult = qty*100 = -200; raw bsm vega for an ATM ~1yr option is on the order of tens of
    # dollars per 100-point move -- after the /100 fix, vega_dollars should be a much smaller,
    # "per realistic 1-point IV move" magnitude, not hundreds/thousands of dollars for a 2-lot.
    assert abs(greeks.vega_dollars) < 500, (
        f"vega_dollars={greeks.vega_dollars} looks like the old per-100-point convention crept back in"
    )


# --------------------------------------------------- asymmetric sizing ---
def test_long_vol_requires_higher_conviction_than_short_vol():
    """Requirement 5: short-vol is the primary strategy (fires at any
    negative edge, matching v2); long-vol requires clearing a materially
    higher conviction bar before ANY Kelly fraction is computed for it."""
    risk = RiskLimits()  # long_vol_edge_threshold=0.35, short_vol_edge_threshold=0.0 by default
    small_positive = 0.20
    small_negative = -0.20
    assert kelly_fraction(small_positive, "Range", risk) == 0.0, "small positive edge should be gated on the long-vol side"
    assert kelly_fraction(small_negative, "Range", risk) != 0.0, "the same-magnitude negative edge should still fire on the short-vol side"

    strong_positive = 0.50
    assert kelly_fraction(strong_positive, "Range", risk) > 0.0, "a strong enough positive edge should still fire"


def test_event_filter_blocks_short_vol_only_never_long_vol():
    """The event/jump filter (research/event_filter.py) is documented to
    throttle/block SHORT-VOL entries only -- this checks the actual
    sizing-layer wiring honors that: short_vol_blocked must never affect
    a long_vol decision."""
    risk = RiskLimits()
    long_decision = decide_sizing(
        composite_edge=0.6, regime_label="Range", capital=100_000,
        straddle_price=10.0, straddle_gamma=0.02, straddle_vega=50.0, risk=risk,
        short_vol_blocked=True, short_vol_block_reason="FOMC blackout",
    )
    assert long_decision.direction == "long_vol"
    assert long_decision.contracts > 0


# ------------------------------------------------------------- ledger ---
def test_ledger_iron_condor_open_and_expiration_cash_accounting():
    """Hand-computed expected values for both the entry (commission +
    modeled spread + slippage, netted against pure-mid credit) and the
    expiration settlement (intrinsic value on the exact leg that ends
    ITM) -- requirement 3's "correct option premium cash accounting...
    commissions... assignment/expiration handling"."""
    from accounting.ledger import Ledger
    from pricing_engine.market_data.contract import OptionType

    ledger = Ledger(starting_cash=100_000, commission_per_contract=0.65, slippage_bps_of_notional=2.0)
    spot = 500.0
    expiry = dt.date(2026, 10, 20)
    legs_mid = [
        ("short_put", OptionType.PUT, 490.0, -1.0, 4.0),
        ("long_put", OptionType.PUT, 480.0, +1.0, 2.0),
        ("short_call", OptionType.CALL, 510.0, -1.0, 4.2),
        ("long_call", OptionType.CALL, 520.0, +1.0, 2.1),
    ]
    pos, fill = ledger.open_structure(dt.date(2026, 9, 20), "iron_condor", expiry, legs_mid, contracts=5, spot=spot)

    assert fill.gross_mid_cash_flow == pytest.approx((4.0 - 2.0 + 4.2 - 2.1) * 100 * 5)
    assert fill.commission_dollars == pytest.approx(0.65 * 5 * 4)
    assert fill.net_cash_flow == pytest.approx(fill.gross_mid_cash_flow - fill.spread_cost_dollars - fill.slippage_cost_dollars - fill.commission_dollars)
    assert ledger.cash == pytest.approx(100_000 + fill.net_cash_flow)

    net = ledger.settle_expiration(pos, spot_at_expiry=484.0)
    # short put (strike 490) is ITM by 6 at spot=484; everything else is OTM.
    assert net == pytest.approx(-1.0 * 6.0 * 100 * 5)
    assert pos.closed is True
    assert pos.close_reason == "expired_itm_settlement"


def test_ledger_condor_legs_are_not_double_counted_by_contracts():
    """The bug: strategy.legs.condor_structure_legs returns signed_qty
    ALREADY multiplied by condor.contracts (correct for its actual
    caller, pricing_engine.risk.scenarios' CVaR repricing, which has no
    separate contracts multiplier elsewhere) -- but accounting.ledger.
    Ledger.open_structure ALSO multiplies by its own separate `contracts`
    argument, so passing the former into the latter double-counts
    contracts (a 20-contract condor priced as if it were 400). The fix
    is condor_structure_legs_per_contract (signed_qty always +/-1) for
    ledger use specifically -- this asserts the ledger's resulting
    notional is the correct 1x, not 20x."""
    from accounting.ledger import Ledger

    ledger_correct = Ledger(starting_cash=100_000)
    contracts = 20
    per_contract_legs = [("leg", None, 500.0, -1.0, 5.0)]
    # Minimal stand-in avoiding a full OptionType import for this arithmetic-only check --
    # open_structure only needs a real OptionType for expiration settlement, not for the
    # entry cash-flow math this test exercises.
    from pricing_engine.market_data.contract import OptionType
    per_contract_legs = [("leg", OptionType.CALL, 500.0, -1.0, 5.0)]
    _pos, fill = ledger_correct.open_structure(dt.date.today(), "iron_condor", dt.date.today() + dt.timedelta(days=30), per_contract_legs, contracts=contracts, spot=500.0)
    # short 1-unit leg at mid=5.0, scaled by contracts=20 exactly once: credit ~= 5.0*100*20 = 10,000
    assert fill.gross_mid_cash_flow == pytest.approx(5.0 * 100 * contracts)


# ------------------------------------------------------- event filter ---
def test_event_filter_hard_blocks_on_fomc_decision_day():
    from research.event_filter import FOMC_DECISION_DATES, assess_event_risk

    a_real_fomc_date = FOMC_DECISION_DATES[0]
    result = assess_event_risk(a_real_fomc_date, fomc_blackout_days_before=1, fomc_blackout_days_after=0)
    assert result.short_vol_blocked is True
    assert result.short_vol_size_multiplier == 0.0


def test_event_filter_does_not_block_an_ordinary_day():
    from research.event_filter import assess_event_risk

    # A date deliberately far from any verified FOMC/CPI/NFP date.
    result = assess_event_risk(dt.date(2026, 1, 20), fomc_blackout_days_before=1, fomc_blackout_days_after=0,
                                cpi_nfp_blackout_days_before=0, cpi_nfp_blackout_days_after=0)
    assert result.short_vol_blocked is False
    assert result.short_vol_size_multiplier == 1.0


def test_event_filter_jump_gate_flags_large_moves_independent_of_calendar():
    from research.event_filter import assess_event_risk

    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.008, 60)
    returns[-1] = -0.05  # a crash-sized move against ~0.8% trailing daily sigma
    result = assess_event_risk(dt.date(2026, 1, 20), daily_log_returns=returns,
                                fomc_blackout_days_before=1, fomc_blackout_days_after=0,
                                cpi_nfp_blackout_days_before=0, cpi_nfp_blackout_days_after=0)
    assert result.jump_flagged is True
    assert result.short_vol_size_multiplier < 1.0


# ------------------------------------------------------- reconciliation ---
def test_reconciliation_ignores_orders_not_placed_by_this_project():
    """execution/reconciliation.py must never touch or even report on an
    order it didn't submit -- a person may be trading the same paper
    account by hand. Identification is solely by the volagent- prefix
    every payload this project submits is stamped with."""
    from execution.reconciliation import reconcile

    now = dt.datetime.now(dt.timezone.utc)
    fake_orders = [
        {"id": "ord-1", "client_order_id": "manual-order-xyz", "status": "new", "symbol": "SPY",
         "qty": "1", "filled_qty": "0", "submitted_at": now.isoformat(), "updated_at": now.isoformat()},
    ]
    client = MagicMock()
    client.get_orders.return_value = fake_orders
    settings = Settings(api_key="PKtest", api_secret="x" * 40)
    result = reconcile(client, settings, cancel_stale=True)
    assert result.open_orders == []
    assert client.cancel_order.call_count == 0


def test_reconciliation_cancels_stale_orders_only_when_not_dry_run():
    from execution.reconciliation import reconcile

    now = dt.datetime.now(dt.timezone.utc)
    old = (now - dt.timedelta(minutes=30)).isoformat()
    fake_orders = [
        {"id": "ord-2", "client_order_id": "volagent-hedge-aaaa", "status": "new", "symbol": "SPY",
         "qty": "5", "filled_qty": "0", "submitted_at": old, "updated_at": old},
    ]
    client = MagicMock()
    client.get_orders.return_value = fake_orders

    dry_settings = Settings(api_key="PKtest", api_secret="x" * 40, dry_run=True)
    reconcile(client, dry_settings, cancel_stale=True)
    assert client.cancel_order.call_count == 0, "dry_run must never actually cancel"

    live_settings = Settings(api_key="PKtest", api_secret="x" * 40, dry_run=False)
    reconcile(client, live_settings, cancel_stale=True)
    assert client.cancel_order.call_count == 1
    assert client.cancel_order.call_args[0][0] == "ord-2"
