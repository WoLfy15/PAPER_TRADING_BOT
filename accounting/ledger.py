"""
accounting/ledger.py -- requirement 3's cash-accounting half: "rebuild
the backtest with correct option premium cash accounting, commissions,
bid/ask spread, slippage, assignment/expiration handling and daily
NAV." backtest/engine.py is the other half (walk-forward, real
structures); this module is purely the bookkeeping -- it knows nothing
about signals, regimes, or sizing, only cash in, cash out, and what's
currently open.

What "real" means here, stated plainly (see backtest/engine.py's own
docstring for the full picture): the UNDERLYING price path and the
IMPLIED VOL level (real CBOE VIX, via data/historical_iv.py) driving
every mark-to-market in this ledger are genuine historical market data.
The bid/ask SPREAD is not -- there is no bundled historical per-strike
options order book, so `modeled_spread_pct` below is a documented,
literal formula (wider for further-OTM strikes, tighter ATM), not a
real observed spread. Every dollar this ledger moves is explicit about
which side that dollar came from (see LedgerFill's cash_components).

ASSIGNMENT/EXPIRATION: modeled as cash settlement on intrinsic value at
the exact expiry date (max(0, spot-strike) for a call, max(0,
strike-spot) for a put) -- the standard simplifying convention used in
options backtesting generally (ignoring early-exercise risk on ITM
American-style short legs before expiration, which is a real but
second-order effect for this kind of position-level backtest; it does
NOT ignore what expiration itself costs -- an ITM short leg still costs
real money here, on the correct date, at the correct intrinsic value).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()

# Documented spread model (see module docstring): half-spread as a
# fraction of mid, widening with |moneyness| (distance from ATM as a
# fraction of spot). 1.0%-1.5% ATM to ~8% far OTM/ITM is in the
# broad, generally-liquid range real SPY options quote at; this is a
# MODEL, not a fit to observed data, and is exposed as a parameter so
# it can be tightened/widened without touching any other code.
DEFAULT_BASE_HALF_SPREAD_PCT = 0.015
DEFAULT_MONEYNESS_WIDENING = 0.12


def modeled_half_spread_pct(moneyness: float, base_pct: float = DEFAULT_BASE_HALF_SPREAD_PCT, widening: float = DEFAULT_MONEYNESS_WIDENING) -> float:
    return base_pct + widening * min(abs(moneyness), 0.5)


@dataclass
class LedgerLeg:
    label: str            # e.g. "SPY_30D_CALL_505"
    option_type: OptionType
    strike: float
    signed_qty: float      # contracts, +long / -short (NOT x100 -- ledger applies the multiplier)
    entry_fill_price: float  # per-share, the EXECUTABLE price actually paid/received


@dataclass
class LedgerPosition:
    position_id: str
    kind: str              # "straddle" | "iron_condor"
    opened_date: dt.date
    expiry: dt.date
    legs: list[LedgerLeg]
    contracts: int
    open_cash_flow: float   # net $ at entry (already includes commission+slippage), + = received
    closed: bool = False
    close_date: Optional[dt.date] = None
    close_cash_flow: float = 0.0
    close_reason: Optional[str] = None  # "closed_early" | "expired_itm_settlement" | "expired_worthless"


@dataclass
class DailyRecord:
    date: dt.date
    cash: float
    mark_to_market_positions_value: float
    nav: float
    daily_pnl_dollars: float
    daily_return_pct: Optional[float]
    notes: list[str] = field(default_factory=list)


@dataclass
class LedgerFill:
    """What open_structure/close_structure return -- broken out by
    component so a caller (or a UI) can show exactly where the cost
    came from, per requirement 3's explicit separate call-out of
    "commissions, bid/ask spread, slippage" as distinct line items
    rather than one bundled friction number."""
    gross_mid_cash_flow: float     # what it would have cost/paid at pure mid, no friction at all
    spread_cost_dollars: float     # modeled bid/ask cost vs. mid (always a cost, i.e. subtracted)
    slippage_cost_dollars: float
    commission_dollars: float
    net_cash_flow: float           # the actual amount that hit `cash` -- gross - spread - slippage - commission


class Ledger:
    def __init__(
        self, starting_cash: float, commission_per_contract: float = 0.0,
        slippage_bps_of_notional: float = 2.0, base_half_spread_pct: float = DEFAULT_BASE_HALF_SPREAD_PCT,
        moneyness_widening: float = DEFAULT_MONEYNESS_WIDENING,
    ):
        self.cash = starting_cash
        self.commission_per_contract = commission_per_contract
        self.slippage_bps_of_notional = slippage_bps_of_notional
        self.base_half_spread_pct = base_half_spread_pct
        self.moneyness_widening = moneyness_widening
        self.positions: list[LedgerPosition] = []
        self.daily_records: list[DailyRecord] = []
        self._next_id = 0

    def _executable_price(self, mid: float, spot: float, strike: float, side: str) -> tuple[float, float]:
        """Returns (fill_price, spread_cost_dollars_per_share). side:
        'buy' pays mid+half_spread, 'sell' receives mid-half_spread --
        both worse than mid by construction, matching requirement 12's
        "actual executable prices" principle applied to the backtest
        instead of live quotes."""
        moneyness = (strike - spot) / spot if spot > 0 else 0.0
        half_spread_pct = modeled_half_spread_pct(moneyness, self.base_half_spread_pct, self.moneyness_widening)
        half_spread = mid * half_spread_pct
        fill = mid + half_spread if side == "buy" else max(mid - half_spread, 0.01)
        return fill, half_spread

    def open_structure(
        self, date: dt.date, kind: str, expiry: dt.date,
        legs_mid: list[tuple[str, OptionType, float, float, float]],  # (label, type, strike, signed_qty, mid_price)
        contracts: int, spot: float,
    ) -> tuple[LedgerPosition, LedgerFill]:
        """legs_mid's mid_price is per-share theoretical/mid value (from
        BSM at the real, historical IV level backtest/engine.py priced
        it at) -- this function is what turns that mid into an actual
        fill and an actual cash flow, applying spread+slippage+commission."""
        gross_mid = 0.0
        spread_cost = 0.0
        legs: list[LedgerLeg] = []
        for label, otype, strike, signed_qty, mid in legs_mid:
            side = "sell" if signed_qty < 0 else "buy"
            fill_price, half_spread = self._executable_price(mid, spot, strike, side)
            legs.append(LedgerLeg(label=label, option_type=otype, strike=strike, signed_qty=signed_qty, entry_fill_price=fill_price))
            gross_mid += -signed_qty * mid * 100 * contracts       # cash flow AT MID (sign: short=+cash in, long=-cash out)
            spread_cost += half_spread * abs(signed_qty) * 100 * contracts  # spread always costs, both directions

        notional = sum(abs(s) * m * 100 * contracts for _, _, _, s, m in legs_mid)
        slippage_cost = notional * self.slippage_bps_of_notional / 10_000.0
        commission = self.commission_per_contract * contracts * len(legs_mid)
        net_cash_flow = gross_mid - spread_cost - slippage_cost - commission

        self.cash += net_cash_flow
        self._next_id += 1
        position = LedgerPosition(
            position_id=f"pos-{self._next_id}", kind=kind, opened_date=date, expiry=expiry,
            legs=legs, contracts=contracts, open_cash_flow=net_cash_flow,
        )
        self.positions.append(position)
        fill = LedgerFill(
            gross_mid_cash_flow=gross_mid, spread_cost_dollars=spread_cost,
            slippage_cost_dollars=slippage_cost, commission_dollars=commission, net_cash_flow=net_cash_flow,
        )
        return position, fill

    def close_structure(self, position: LedgerPosition, date: dt.date, legs_mid: dict[str, float], spot: float, reason: str = "closed_early") -> LedgerFill:
        """legs_mid: {leg_label: current_mid_price}. Closing REVERSES
        each leg's sign for pricing purposes (a short leg is bought back
        -> pays the spread on the buy side; a long leg is sold -> pays
        the spread on the sell side)."""
        gross_mid = 0.0
        spread_cost = 0.0
        for leg in position.legs:
            mid = legs_mid.get(leg.label)
            if mid is None:
                continue
            closing_side = "buy" if leg.signed_qty < 0 else "sell"  # reverse of how it was opened
            fill_price, half_spread = self._executable_price(mid, spot, leg.strike, closing_side)
            gross_mid += leg.signed_qty * mid * 100 * position.contracts  # closing reverses the original sign convention
            spread_cost += half_spread * abs(leg.signed_qty) * 100 * position.contracts

        notional = sum(abs(l.signed_qty) * (legs_mid.get(l.label) or 0.0) * 100 * position.contracts for l in position.legs)
        slippage_cost = notional * self.slippage_bps_of_notional / 10_000.0
        commission = self.commission_per_contract * position.contracts * len(position.legs)
        net_cash_flow = gross_mid - spread_cost - slippage_cost - commission

        self.cash += net_cash_flow
        position.closed = True
        position.close_date = date
        position.close_cash_flow = net_cash_flow
        position.close_reason = reason
        return LedgerFill(
            gross_mid_cash_flow=gross_mid, spread_cost_dollars=spread_cost,
            slippage_cost_dollars=slippage_cost, commission_dollars=commission, net_cash_flow=net_cash_flow,
        )

    def settle_expiration(self, position: LedgerPosition, spot_at_expiry: float) -> float:
        """Cash-settles every leg at intrinsic value on the expiry date
        (see module docstring for the early-exercise simplification this
        represents). No spread/slippage/commission on expiration itself
        -- there's no order to fill, the exchange settles it. Returns
        the net cash flow, which is ALSO applied to self.cash."""
        net = 0.0
        for leg in position.legs:
            intrinsic = max(spot_at_expiry - leg.strike, 0.0) if leg.option_type == OptionType.CALL else max(leg.strike - spot_at_expiry, 0.0)
            net += leg.signed_qty * intrinsic * 100 * position.contracts
        self.cash += net
        position.closed = True
        position.close_date = position.expiry
        position.close_cash_flow = net
        position.close_reason = "expired_itm_settlement" if abs(net) > 1e-6 else "expired_worthless"
        return net

    def mark_to_market(self, date: dt.date, spot: float, iv: float, risk_free_rate: float, time_to_expiry_fn=None) -> DailyRecord:
        """Prices every OPEN position at real mid (BSM, at the day's
        real spot + real IV level) -- no spread/friction here, mark-to-
        market is a valuation, not a transaction. time_to_expiry_fn
        defaults to (leg.expiry - date).days / 365 if not supplied."""
        positions_value = 0.0
        for pos in self.positions:
            if pos.closed:
                continue
            tte = time_to_expiry_fn(pos.expiry, date) if time_to_expiry_fn else max((pos.expiry - date).days, 0) / 365.0
            for leg in pos.legs:
                if tte <= 0:
                    price = max(spot - leg.strike, 0.0) if leg.option_type == OptionType.CALL else max(leg.strike - spot, 0.0)
                else:
                    q = OptionQuote(underlying="", spot=spot, strike=leg.strike, expiry_years=tte, option_type=leg.option_type, risk_free_rate=risk_free_rate, market_price=None)
                    price = _bsm.price(q, max(iv, 0.01))
                positions_value += leg.signed_qty * price * 100 * pos.contracts

        nav = self.cash + positions_value
        prior_nav = self.daily_records[-1].nav if self.daily_records else nav
        daily_pnl = nav - prior_nav
        daily_return = (daily_pnl / prior_nav) if prior_nav else None
        record = DailyRecord(
            date=date, cash=self.cash, mark_to_market_positions_value=positions_value,
            nav=nav, daily_pnl_dollars=daily_pnl, daily_return_pct=daily_return,
        )
        self.daily_records.append(record)
        return record

    @property
    def open_positions(self) -> list[LedgerPosition]:
        return [p for p in self.positions if not p.closed]
