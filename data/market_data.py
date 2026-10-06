"""
data/market_data.py -- the join between Alpaca's live data and the
pricing_engine's OptionQuote/MarketSmilePoint contracts.

Two Alpaca calls feed every contract:
  - GET /v2/options/contracts (Trading API): the master record -- strike,
    expiry, type, tradable, open_interest. Authoritative for "does this
    contract exist and can I trade it."
  - GET /v1beta1/options/snapshots/{symbol} (Data API): the live quote
    (bid/ask/last) and, when the feed provides it, Alpaca's own Greeks
    and implied vol. Authoritative for "what is it worth right now."

We join on contract symbol. Where Alpaca's own IV/Greeks are present we
keep them for reference, but every OptionQuote is *also* solvable via
pricing_engine.calibration.iv_solver against the mid price -- so the
signal layer never silently depends on one feed's Greek convention.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from alpaca_client import AlpacaClient
from pricing_engine.market_data.contract import MarketSmilePoint, OptionQuote, OptionType
from pricing_engine.calibration.iv_solver import solve_implied_vol, IVSolverError
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()

TRADING_DAYS_PER_YEAR = 252


@dataclass
class ChainContract:
    symbol: str
    underlying: str
    strike: float
    expiry: dt.date
    option_type: OptionType
    open_interest: int
    tradable: bool
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    alpaca_iv: Optional[float]
    alpaca_delta: Optional[float]
    alpaca_gamma: Optional[float]
    alpaca_theta: Optional[float]
    alpaca_vega: Optional[float]
    # v3, requirement 12 ("reject trades with excessive bid/ask width or
    # stale quotes"): the live quote's own timestamp, so a caller can
    # measure staleness. None for contracts priced off `last` (no live
    # quote at all) or when the snapshot's quote timestamp didn't parse.
    quote_timestamp: Optional[dt.datetime] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread_pct(self) -> Optional[float]:
        """(ask - bid) / mid -- None when there's no two-sided quote to
        measure a spread from at all (a last-trade-only contract has an
        unknown spread, not a zero one)."""
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            m = 0.5 * (self.bid + self.ask)
            return (self.ask - self.bid) / m if m > 0 else None
        return None

    def quote_age_seconds(self, as_of: Optional[dt.datetime] = None) -> Optional[float]:
        if self.quote_timestamp is None:
            return None
        as_of = as_of or dt.datetime.now(dt.timezone.utc)
        ts = self.quote_timestamp if self.quote_timestamp.tzinfo else self.quote_timestamp.replace(tzinfo=dt.timezone.utc)
        return max((as_of - ts).total_seconds(), 0.0)


def executable_price(contract: ChainContract, side: str) -> Optional[float]:
    """v3, requirement 12 ("use actual executable prices -- bid for
    sells, ask for buys"). This is the CONSERVATIVE price used for
    RISK figures that feed sizing (a structure's max-loss and net-credit
    in strategy/legs.py, and the backtest's cash accounting) -- it is
    deliberately more pessimistic than the mid-price a resting limit
    order actually targets in execution/order_manager.py, which is a
    legitimate, separate execution TACTIC (try to get filled inside the
    spread) rather than a risk-figure input. side: 'buy' -> ask (what
    you'd pay if you had to cross the spread right now); 'sell' -> bid
    (what you'd receive). Falls back to `last` (with no conservative
    adjustment -- there is no side-specific worse-case number to use
    without a two-sided quote) when one side of the book is missing,
    and to None when there's nothing at all to price from."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if contract.bid and contract.ask and contract.bid > 0 and contract.ask > 0:
        return contract.ask if side == "buy" else contract.bid
    return contract.last


def _years_to_expiry(expiry: dt.date, as_of: Optional[dt.date] = None) -> float:
    as_of = as_of or dt.date.today()
    return max((expiry - as_of).days, 0) / 365.0


def fetch_chain(
    client: AlpacaClient,
    underlying_symbol: str,
    expiration_date_gte: Optional[str] = None,
    expiration_date_lte: Optional[str] = None,
    option_type: Optional[str] = None,
    min_open_interest: int = 0,
    max_spread_pct: Optional[float] = None,
    max_quote_age_seconds: Optional[float] = None,
) -> list[ChainContract]:
    """One call each to the contracts endpoint and the snapshot endpoint,
    joined into a flat list of ChainContract. Contracts with no live quote
    (illiquid, halted) are dropped -- nothing downstream should have to
    special-case a None mid price.

    v3, requirement 12: max_spread_pct / max_quote_age_seconds, when
    given (agent/loop.py and backtest/engine.py pass config.Settings.
    max_bid_ask_spread_pct / max_quote_age_seconds), ALSO drop any
    contract whose live spread is wider than that fraction of mid, or
    whose quote is older than that many seconds -- a wide-or-stale quote
    is worse than no quote at all for sizing a real trade against,
    because unlike a missing quote it looks like real, usable data.
    Left as None (the default), this is a no-op and fetch_chain behaves
    exactly as before -- existing callers/tests are unaffected."""
    contracts_meta = client.get_option_contracts(
        underlying_symbol,
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
    )
    snapshots = client.get_option_chain_snapshot(
        underlying_symbol,
        option_type=option_type,
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
    )

    now = dt.datetime.now(dt.timezone.utc)
    out: list[ChainContract] = []
    for c in contracts_meta:
        if not c.get("tradable", False):
            continue
        oi = int(c.get("open_interest") or 0)
        if oi < min_open_interest:
            continue
        snap = snapshots.get(c["symbol"])
        if not snap:
            continue
        quote = snap.get("latestQuote") or {}
        trade = snap.get("latestTrade") or {}
        greeks = snap.get("greeks") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        last = trade.get("p")
        if not ((bid and ask) or last):
            continue
        quote_ts = None
        if quote.get("t"):
            try:
                quote_ts = dt.datetime.fromisoformat(str(quote["t"]).replace("Z", "+00:00"))
            except ValueError:
                quote_ts = None
        contract = ChainContract(
            symbol=c["symbol"],
            underlying=underlying_symbol,
            strike=float(c["strike_price"]),
            expiry=dt.date.fromisoformat(c["expiration_date"]),
            option_type=OptionType.CALL if c["type"] == "call" else OptionType.PUT,
            open_interest=oi,
            tradable=True,
            bid=bid, ask=ask, last=last,
            alpaca_iv=snap.get("impliedVolatility"),
            alpaca_delta=greeks.get("delta"),
            alpaca_gamma=greeks.get("gamma"),
            alpaca_theta=greeks.get("theta"),
            alpaca_vega=greeks.get("vega"),
            quote_timestamp=quote_ts,
        )
        if max_spread_pct is not None and contract.spread_pct is not None and contract.spread_pct > max_spread_pct:
            continue
        if max_quote_age_seconds is not None:
            age = contract.quote_age_seconds(as_of=now)
            if age is not None and age > max_quote_age_seconds:
                continue
        out.append(contract)
    return out


def to_option_quote(
    contract: ChainContract, spot: float, risk_free_rate: float, dividend_yield: float = 0.0,
    as_of: Optional[dt.date] = None,
) -> OptionQuote:
    """as_of: v3 fix -- previously this always measured time-to-expiry
    against dt.date.today() (see _years_to_expiry's own default), which
    is correct for the live agent (where "today" always IS the real
    date) but silently wrong for a backtest walking through historical
    dates -- a 2022-dated contract's expiry, measured against the REAL
    current date, produces a large NEGATIVE time-to-expiry and a
    downstream BSM ValueError. This surfaced only once backtest/engine.py
    started reusing this exact function instead of a separate bespoke
    pricer (which is the correct fix per requirement 3 -- "trade the
    same structures" -- and exactly why this gap was never exercised
    before). Defaults to None -> dt.date.today(), so every existing
    live-path caller is unaffected; backtest/engine.py passes its own
    walk-forward date explicitly at every call site."""
    return OptionQuote(
        underlying=contract.underlying,
        spot=spot,
        strike=contract.strike,
        expiry_years=_years_to_expiry(contract.expiry, as_of=as_of),
        option_type=contract.option_type,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        market_price=contract.mid,
        bid=contract.bid,
        ask=contract.ask,
        volume=None,
        open_interest=contract.open_interest,
        implied_vol=contract.alpaca_iv,
    )


def implied_vol_for(quote: OptionQuote, fallback_guess: float = 0.25) -> Optional[float]:
    """Solve for IV from the mid price if Alpaca didn't supply one (the
    indicative feed often doesn't). Returns None if unsolvable (crossed
    or stale quote) rather than raising -- callers should skip the strike."""
    if quote.implied_vol and quote.implied_vol > 0:
        return quote.implied_vol
    if not quote.market_price or quote.market_price <= 0 or quote.expiry_years <= 0:
        return None
    try:
        return solve_implied_vol(_bsm, quote, quote.market_price, initial_guess=fallback_guess).implied_vol
    except IVSolverError:
        return None


def build_smile(
    contracts: list[ChainContract], spot: float, risk_free_rate: float, dividend_yield: float = 0.0,
) -> dict[dt.date, list[MarketSmilePoint]]:
    """Group live IVs by expiry into MarketSmilePoint lists, ready for
    pricing_engine.calibration.local_vol_surface.SmileSlice.fit(...)."""
    by_expiry: dict[dt.date, list[MarketSmilePoint]] = {}
    for c in contracts:
        quote = to_option_quote(c, spot, risk_free_rate, dividend_yield)
        iv = implied_vol_for(quote)
        if iv is None:
            continue
        by_expiry.setdefault(c.expiry, []).append(
            MarketSmilePoint(strike=c.strike, expiry_years=quote.expiry_years, implied_vol=iv)
        )
    return by_expiry


def atm_implied_vol(contracts: list[ChainContract], spot: float, risk_free_rate: float) -> Optional[float]:
    """Straddle-relevant single number: average call+put IV at the strike
    closest to spot, across the nearest expiry present in `contracts`."""
    if not contracts:
        return None
    nearest_expiry = min(c.expiry for c in contracts)
    same_expiry = [c for c in contracts if c.expiry == nearest_expiry]
    atm_strike = min({c.strike for c in same_expiry}, key=lambda k: abs(k - spot))
    ivs = []
    for c in same_expiry:
        if c.strike != atm_strike:
            continue
        quote = to_option_quote(c, spot, risk_free_rate)
        iv = implied_vol_for(quote)
        if iv:
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else None


def daily_log_returns_from_bars(bars: list[dict]) -> np.ndarray:
    closes = np.array([b["c"] for b in bars], dtype=float)
    if len(closes) < 2:
        return np.array([])
    return np.diff(np.log(closes))
