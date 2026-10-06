"""
Team B — Phase 3: Black-Scholes-Merton.
European call/put pricing with continuous dividend yield, plus the full
analytic Greek set. This is the foundational model every other module
(local vol surface fitting, SAS, validation) reprices against.
"""
from __future__ import annotations

import math

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.base import PricingModel

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive for d1/d2")
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


class BlackScholesMerton(PricingModel):
    name = "BSM"

    def price(self, quote: OptionQuote, sigma: float) -> float:
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        if T <= 0:
            intrinsic = max(S - K, 0.0) if quote.option_type == OptionType.CALL else max(K - S, 0.0)
            return intrinsic
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
        if quote.option_type == OptionType.CALL:
            return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

    def delta(self, quote: OptionQuote, sigma: float) -> float:
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        if quote.option_type == OptionType.CALL:
            return math.exp(-q * T) * norm_cdf(d1)
        return math.exp(-q * T) * (norm_cdf(d1) - 1.0)

    def gamma(self, quote: OptionQuote, sigma: float) -> float:
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))

    def vega(self, quote: OptionQuote, sigma: float) -> float:
        """Per unit (100%) change in sigma. Divide by 100 for a 'per vol point' convention."""
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)

    def theta(self, quote: OptionQuote, sigma: float) -> float:
        """Per year. Divide by 365 for 'per calendar day'."""
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        d1, d2 = _d1_d2(S, K, T, r, q, sigma)
        term1 = -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
        if quote.option_type == OptionType.CALL:
            return term1 - r * K * math.exp(-r * T) * norm_cdf(d2) + q * S * math.exp(-q * T) * norm_cdf(d1)
        return term1 + r * K * math.exp(-r * T) * norm_cdf(-d2) - q * S * math.exp(-q * T) * norm_cdf(-d1)

    def rho(self, quote: OptionQuote, sigma: float) -> float:
        """Per unit (100%) change in r."""
        S, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
        _, d2 = _d1_d2(S, K, T, r, q, sigma)
        if quote.option_type == OptionType.CALL:
            return K * T * math.exp(-r * T) * norm_cdf(d2)
        return -K * T * math.exp(-r * T) * norm_cdf(-d2)
