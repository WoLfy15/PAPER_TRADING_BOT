"""
Team A — base Model interface (Phase 2).
Every pricing model (BSM, Black-76, CRR, Leisen-Reimer, Monte Carlo, PDE,
local-vol MC ...) implements this so Team E's aggregation layer, the
dashboard, and validation/checks.py can treat all models polymorphically.

sigma is passed explicitly rather than read off the quote, so a single
OptionQuote can be repriced under many vol assumptions (market IV, a local
vol surface, a scenario-shocked vol) without mutating anything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pricing_engine.market_data.contract import OptionQuote


class PricingModel(ABC):
    name: str = "base"

    @abstractmethod
    def price(self, quote: OptionQuote, sigma: float) -> float: ...

    @abstractmethod
    def delta(self, quote: OptionQuote, sigma: float) -> float: ...

    @abstractmethod
    def gamma(self, quote: OptionQuote, sigma: float) -> float: ...

    @abstractmethod
    def vega(self, quote: OptionQuote, sigma: float) -> float: ...

    @abstractmethod
    def theta(self, quote: OptionQuote, sigma: float) -> float: ...

    @abstractmethod
    def rho(self, quote: OptionQuote, sigma: float) -> float: ...

    def greeks(self, quote: OptionQuote, sigma: float) -> dict:
        return {
            "price": self.price(quote, sigma),
            "delta": self.delta(quote, sigma),
            "gamma": self.gamma(quote, sigma),
            "vega": self.vega(quote, sigma),
            "theta": self.theta(quote, sigma),
            "rho": self.rho(quote, sigma),
        }


class FiniteDifferenceGreeksMixin:
    """Fallback numerical Greeks for any model that only implements
    `price()` cleanly (trees, PDE grids, Monte Carlo). Also used across the
    whole engine as the Model-Risk-paper-mandated cross-check against each
    model's own analytic Greeks (see validation/checks.py)."""

    def fd_delta(self, quote: OptionQuote, sigma: float, bump: float = 1e-4) -> float:
        up = self.price(quote.with_(spot=quote.spot * (1 + bump)), sigma)
        dn = self.price(quote.with_(spot=quote.spot * (1 - bump)), sigma)
        return (up - dn) / (2 * quote.spot * bump)

    def fd_gamma(self, quote: OptionQuote, sigma: float, bump: float = 1e-3) -> float:
        up = self.price(quote.with_(spot=quote.spot * (1 + bump)), sigma)
        mid = self.price(quote, sigma)
        dn = self.price(quote.with_(spot=quote.spot * (1 - bump)), sigma)
        h = quote.spot * bump
        return (up - 2 * mid + dn) / (h * h)

    def fd_vega(self, quote: OptionQuote, sigma: float, bump: float = 1e-4) -> float:
        up = self.price(quote, sigma + bump)
        dn = self.price(quote, max(sigma - bump, 1e-6))
        return (up - dn) / (2 * bump)

    def fd_theta(self, quote: OptionQuote, sigma: float, bump: float = 1e-4) -> float:
        # theta = -dV/dtau; shrink time-to-expiry and negate
        t_dn = max(quote.expiry_years - bump, 1e-6)
        up = self.price(quote, sigma)
        dn = self.price(quote.with_(expiry_years=t_dn), sigma)
        return (dn - up) / bump

    def fd_rho(self, quote: OptionQuote, sigma: float, bump: float = 1e-4) -> float:
        up = self.price(quote.with_(risk_free_rate=quote.risk_free_rate + bump), sigma)
        dn = self.price(quote.with_(risk_free_rate=quote.risk_free_rate - bump), sigma)
        return (up - dn) / (2 * bump)
