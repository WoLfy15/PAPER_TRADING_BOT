"""
Team B — Phase 5: Black-76.
Options on futures/forwards: same lognormal machinery as BSM but the
underlying (quote.spot, read as F) already carries no cost-of-carry — both
legs discount at r alone. We reuse BSM's norm_cdf/norm_pdf to keep one
source of truth for the normal distribution.
"""
from __future__ import annotations

import math

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.base import PricingModel
from pricing_engine.models.bsm import norm_cdf, norm_pdf


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive for d1/d2")
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


class Black76(PricingModel):
    """quote.spot is read as the forward/futures price F; quote.dividend_yield
    is ignored (the forward convention already embeds carry)."""

    name = "Black76"

    def price(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        if T <= 0:
            intrinsic = max(F - K, 0.0) if quote.option_type == OptionType.CALL else max(K - F, 0.0)
            return intrinsic
        d1, d2 = _d1_d2(F, K, T, sigma)
        disc = math.exp(-r * T)
        if quote.option_type == OptionType.CALL:
            return disc * (F * norm_cdf(d1) - K * norm_cdf(d2))
        return disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

    def delta(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        d1, _ = _d1_d2(F, K, T, sigma)
        disc = math.exp(-r * T)
        if quote.option_type == OptionType.CALL:
            return disc * norm_cdf(d1)
        return -disc * norm_cdf(-d1)

    def gamma(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        d1, _ = _d1_d2(F, K, T, sigma)
        return math.exp(-r * T) * norm_pdf(d1) / (F * sigma * math.sqrt(T))

    def vega(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        d1, _ = _d1_d2(F, K, T, sigma)
        return F * math.exp(-r * T) * norm_pdf(d1) * math.sqrt(T)

    def theta(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        d1, d2 = _d1_d2(F, K, T, sigma)
        disc = math.exp(-r * T)
        decay = -disc * F * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
        if quote.option_type == OptionType.CALL:
            return decay + r * disc * F * norm_cdf(d1) - r * disc * K * norm_cdf(d2)
        return decay - r * disc * F * norm_cdf(-d1) + r * disc * K * norm_cdf(-d2)

    def rho(self, quote: OptionQuote, sigma: float) -> float:
        F, K, T, r = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate
        price = self.price(quote, sigma)
        return -T * price
