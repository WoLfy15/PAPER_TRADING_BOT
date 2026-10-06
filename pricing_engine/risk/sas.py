"""
risk/sas.py — Strike-Adjusted Spread (Zou & Derman, Goldman Sachs 1999).

A one-number cheap/rich metric: SAS(K,T) = market implied vol - a "fair"
implied vol implied purely by the underlying's OWN historical returns,
with no options market needed to build the fair leg. Useful for:
  - flagging strikes where the market smile has priced in more (or less)
    tail risk than the underlying's own history suggests
  - a fair-value fallback for illiquid strikes / names with no listed
    options (Team A's stubbed CBOE/Polygon/NASDAQ feeds can fall back to
    this when a quote is missing or stale)

Method (risk-neutralized historical distribution, RNHD):
  1. Build an empirical distribution of horizon-T returns from daily
     price history (overlapping windows).
  2. Find the minimum-relative-entropy distribution Q that is absolutely
     continuous w.r.t. the empirical P and satisfies the single
     no-arbitrage constraint E_Q[S_T] = forward. The solution has closed
     form Q(S) ~ P(S) * exp(-lambda*S) (an exponential tilt; Zou-Derman's
     appendix derives this via an Arrow-Debreu/exponential-utility
     argument, so the functional form doesn't depend on assuming any
     particular risk aversion -- only the multiplier lambda does, fixed
     by the forward constraint).
  3. Price options under Q; back out the Black-Scholes implied vol that
     would reproduce that price. SAS is the market/history gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton
from pricing_engine.calibration.iv_solver import solve_implied_vol

_bsm = BlackScholesMerton()


@dataclass
class RNHD:
    """Risk-neutralized historical distribution: discrete scenario prices
    with Q-probabilities that reprice the forward exactly."""

    scenario_prices: np.ndarray
    q_probabilities: np.ndarray
    lam: float

    def call_price(self, strike: float, r: float, T: float) -> float:
        payoff = np.maximum(self.scenario_prices - strike, 0.0)
        return math.exp(-r * T) * float(np.dot(self.q_probabilities, payoff))

    def put_price(self, strike: float, r: float, T: float) -> float:
        payoff = np.maximum(strike - self.scenario_prices, 0.0)
        return math.exp(-r * T) * float(np.dot(self.q_probabilities, payoff))


def build_rnhd(
    spot: float,
    daily_log_returns: np.ndarray,
    T: float,
    r: float,
    q: float,
    trading_days_per_year: int = 252,
) -> RNHD:
    horizon_days = max(int(round(T * trading_days_per_year)), 1)
    if len(daily_log_returns) <= horizon_days:
        raise ValueError("not enough historical returns for this horizon")

    cumsum = np.insert(np.cumsum(daily_log_returns), 0, 0.0)
    horizon_returns = cumsum[horizon_days:] - cumsum[:-horizon_days]
    scenario_prices = spot * np.exp(horizon_returns)
    forward = spot * math.exp((r - q) * T)

    n = len(scenario_prices)
    base_p = np.full(n, 1.0 / n)

    def expected_S(lam: float) -> float:
        x = -lam * scenario_prices
        x = x - x.max()  # numerically stable softmax
        w = base_p * np.exp(x)
        w /= w.sum()
        return float(np.dot(w, scenario_prices))

    if not (scenario_prices.min() < forward < scenario_prices.max()):
        raise ValueError(
            "forward lies outside the historical scenario range -- widen "
            "the lookback window or check the r/q/T inputs"
        )

    scale = 1.0 / spot
    lo, hi = -scale, scale
    for _ in range(60):
        if expected_S(lo) >= forward:
            break
        lo *= 2
    else:
        raise ValueError("could not bracket forward on the downside")
    for _ in range(60):
        if expected_S(hi) <= forward:
            break
        hi *= 2
    else:
        raise ValueError("could not bracket forward on the upside")

    lam = brentq(lambda l: expected_S(l) - forward, lo, hi, xtol=1e-12)
    x = -lam * scenario_prices
    x = x - x.max()
    w = base_p * np.exp(x)
    w /= w.sum()
    return RNHD(scenario_prices=scenario_prices, q_probabilities=w, lam=lam)


def strike_adjusted_spread(
    market_quote: OptionQuote,
    market_implied_vol: float,
    daily_log_returns: np.ndarray,
) -> float:
    """SAS = market IV - historical 'fair' IV at the same strike/maturity.
    Positive SAS => the market is pricing this strike richer than recent
    realized history would suggest; negative => cheaper."""
    rnhd = build_rnhd(
        spot=market_quote.spot,
        daily_log_returns=daily_log_returns,
        T=market_quote.expiry_years,
        r=market_quote.risk_free_rate,
        q=market_quote.dividend_yield,
    )
    if market_quote.option_type == OptionType.CALL:
        fair_price = rnhd.call_price(market_quote.strike, market_quote.risk_free_rate, market_quote.expiry_years)
    else:
        fair_price = rnhd.put_price(market_quote.strike, market_quote.risk_free_rate, market_quote.expiry_years)

    result = solve_implied_vol(_bsm, market_quote, fair_price)
    return market_implied_vol - result.implied_vol
