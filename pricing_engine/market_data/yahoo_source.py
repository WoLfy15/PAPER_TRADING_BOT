"""
Team A — Phase 1: market data ingestion, Yahoo Finance first (free, no
key). Stub interfaces for CBOE/NASDAQ/Polygon below so Team A can swap
sources without touching any downstream model.

Note: this dev sandbox's network egress is restricted to package
registries (pypi/npm/github), so the live yfinance calls below are not
exercised by the test suite here -- tests use synthetic OptionQuote
fixtures instead. Wire this up and smoke-test it from your own machine
before Week 1's data pipeline is considered done.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List

from pricing_engine.market_data.contract import OptionQuote, OptionType


class MarketDataSource(ABC):
    """Every data source (Yahoo, CBOE, NASDAQ, Polygon) implements this so
    Team A can swap providers without any downstream model noticing."""

    @abstractmethod
    def get_option_chain(self, underlying: str, expiry: date) -> List[OptionQuote]: ...

    @abstractmethod
    def get_spot(self, underlying: str) -> float: ...

    @abstractmethod
    def get_risk_free_rate(self, tenor_years: float) -> float: ...


class YahooFinanceSource(MarketDataSource):
    """pip install yfinance. Free, delayed quotes, no API key -- the
    documented Week-1 starting point."""

    def __init__(self, risk_free_rate_flat: float = 0.045, dividend_yield_flat: float = 0.0):
        self._r_flat = risk_free_rate_flat
        self._q_flat = dividend_yield_flat

    def get_spot(self, underlying: str) -> float:
        import yfinance as yf  # local import: keep this an optional dependency

        ticker = yf.Ticker(underlying)
        fast = ticker.fast_info
        return float(fast["lastPrice"])

    def get_risk_free_rate(self, tenor_years: float) -> float:
        # Placeholder flat rate. Swap for a fetched Treasury curve
        # (e.g. FRED's DGS3MO/DGS1/DGS2) once Team A wires up a curve
        # source -- every model already takes rate as an explicit input,
        # so this is a pure data-layer upgrade with zero blast radius.
        return self._r_flat

    def get_option_chain(self, underlying: str, expiry: date) -> List[OptionQuote]:
        import yfinance as yf

        ticker = yf.Ticker(underlying)
        spot = self.get_spot(underlying)
        chain = ticker.option_chain(expiry.isoformat())
        expiry_years = max((expiry - date.today()).days / 365.0, 1e-6)
        r = self.get_risk_free_rate(expiry_years)

        quotes: List[OptionQuote] = []
        for row in chain.calls.itertuples():
            quotes.append(_row_to_quote(row, underlying, spot, expiry_years, r, self._q_flat, OptionType.CALL))
        for row in chain.puts.itertuples():
            quotes.append(_row_to_quote(row, underlying, spot, expiry_years, r, self._q_flat, OptionType.PUT))
        return quotes


def _row_to_quote(row, underlying, spot, expiry_years, r, q, option_type) -> OptionQuote:
    return OptionQuote(
        underlying=underlying,
        spot=spot,
        strike=float(row.strike),
        expiry_years=expiry_years,
        option_type=option_type,
        risk_free_rate=r,
        dividend_yield=q,
        market_price=float(row.lastPrice) if getattr(row, "lastPrice", None) is not None else None,
        bid=float(row.bid) if getattr(row, "bid", None) not in (None, 0) else None,
        ask=float(row.ask) if getattr(row, "ask", None) not in (None, 0) else None,
        volume=int(row.volume) if getattr(row, "volume", None) is not None else None,
        open_interest=int(row.openInterest) if getattr(row, "openInterest", None) is not None else None,
        implied_vol=float(row.impliedVolatility) if getattr(row, "impliedVolatility", None) is not None else None,
    )


class StubSource(MarketDataSource):
    """Interface placeholder for CBOE / NASDAQ / Polygon -- fill in once
    API keys are available. Keeping these as no-op stubs (rather than not
    writing them at all) means every downstream team can already code
    against `MarketDataSource` on day one."""

    def __init__(self, name: str):
        self.name = name

    def get_option_chain(self, underlying: str, expiry: date) -> List[OptionQuote]:
        raise NotImplementedError(f"{self.name} source not yet wired up")

    def get_spot(self, underlying: str) -> float:
        raise NotImplementedError(f"{self.name} source not yet wired up")

    def get_risk_free_rate(self, tenor_years: float) -> float:
        raise NotImplementedError(f"{self.name} source not yet wired up")
