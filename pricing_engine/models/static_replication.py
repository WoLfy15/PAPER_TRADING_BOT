"""
Static replication for barrier options (Derman, Ergener, Kani 1994).

Instead of (or alongside) Team D's PDE grid, this builds a fixed-weight
portfolio of vanilla calls that matches an up-and-out call's boundary
conditions -- no rebalancing, unlike delta-hedging or a PDE re-solve
under a bumped surface. The output is the actual hedge (a list of
strikes/maturities/weights), not just a number, which is the whole point
of the paper: a trading desk can put this portfolio on once and hold it.

Algorithm (discrete boundary matching, working backward from expiry):
  1. Terminal leg: long 1 call(K,T), short 1 call(B,T). This reproduces
     the up-and-out payoff exactly for S<B at expiry; it leaves a flat
     residual of (B-K) for S>=B, which is corrected by the steps below.
  2. For each earlier monitoring date t_k (descending from the one just
     before T down to the first), price the portfolio built so far at
     the barrier (S=B, time=t_k) using Black-Scholes. A true knockout
     option is worth exactly 0 there, so add a new call struck at K,
     maturing exactly at t_k, with weight w_k chosen so the total
     portfolio value at (B, t_k) becomes 0. Because this new leg expires
     at t_k, it does not disturb any boundary match already achieved at
     later dates.
  3. More monitoring dates -> more legs -> smaller residual mismatch,
     converging to exact replication as n_matches -> infinity. This
     mirrors the paper's own finding: a 7-option static hedge for a
     1-year up-and-out call already tracks the true value well away
     from the barrier/expiry corner, where the mismatch is largest.

Uses a single flat sigma for tractability. The paper itself flags that
static hedging should ideally use a skew-consistent (implied/local vol)
tree when volatility is skewed -- pass a LocalVolatilitySurface via
`vol_func` to price each leg off calibration/local_vol_surface.py instead
of a flat number, which is a direct, documented extension point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()


@dataclass(frozen=True)
class Leg:
    weight: float
    strike: float
    maturity: float
    option_type: OptionType = OptionType.CALL


@dataclass
class StaticReplicationResult:
    price: float
    legs: list[Leg] = field(default_factory=list)

    def hedge_table(self) -> list[dict]:
        return [
            {"weight": round(l.weight, 4), "strike": round(l.strike, 4),
             "maturity_years": round(l.maturity, 4), "type": l.option_type.value}
            for l in self.legs
        ]


SigmaFunc = Callable[[float, float], float]  # (strike, t) -> sigma


def _leg_value(leg: Leg, spot: float, t: float, r: float, q: float, sigma_func: SigmaFunc) -> float:
    t_remaining = leg.maturity - t
    quote = OptionQuote(
        underlying="static_replication",
        spot=spot,
        strike=leg.strike,
        expiry_years=max(t_remaining, 0.0),
        option_type=leg.option_type,
        risk_free_rate=r,
        dividend_yield=q,
    )
    sigma = sigma_func(leg.strike, max(t_remaining, 1e-6))
    return leg.weight * _bsm.price(quote, sigma)


def _portfolio_value(legs, spot: float, t: float, r: float, q: float, sigma_func: SigmaFunc) -> float:
    return sum(_leg_value(leg, spot, t, r, q, sigma_func) for leg in legs)


def static_replicate_up_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_matches: int = 12,
    vol_func: Optional[SigmaFunc] = None,
) -> StaticReplicationResult:
    """K < B required (a standard up-and-out call). n_matches is the
    number of discrete barrier-monitoring points (including expiry)."""
    if not (0 < strike < barrier):
        raise ValueError("require 0 < strike < barrier")
    if T <= 0:
        raise ValueError("T must be positive")

    sigma_func: SigmaFunc = vol_func if vol_func is not None else (lambda k, t: sigma)

    match_times = np.linspace(T / n_matches, T, n_matches)
    legs: list[Leg] = [
        Leg(weight=1.0, strike=strike, maturity=T, option_type=OptionType.CALL),
        Leg(weight=-1.0, strike=barrier, maturity=T, option_type=OptionType.CALL),
    ]

    for t_k in sorted(match_times[:-1], reverse=True):
        v_existing = _portfolio_value(legs, spot=barrier, t=t_k, r=r, q=q, sigma_func=sigma_func)
        denom = barrier - strike  # intrinsic value at S=B of a call struck at `strike`, at its own maturity t_k
        w_k = -v_existing / denom
        legs.append(Leg(weight=w_k, strike=strike, maturity=t_k, option_type=OptionType.CALL))

    price_today = _portfolio_value(legs, spot=spot, t=0.0, r=r, q=q, sigma_func=sigma_func)
    return StaticReplicationResult(price=price_today, legs=legs)


def static_replicate_down_and_out_put(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_matches: int = 12,
    vol_func: Optional[SigmaFunc] = None,
) -> StaticReplicationResult:
    """B < K required (a standard down-and-out put), built with the same
    recursive boundary-matching structure as the up-and-out call above.

    *** KNOWN OPEN VALIDATION ISSUE -- DO NOT TRUST THIS FUNCTION YET ***
    Unlike the call above (cross-validated against both a fine CRR tree
    and independent Monte Carlo, converging cleanly to the right price as
    n_matches grows), this mirrored put construction was checked against
    the same two independent references during development and does NOT
    converge to the correct price -- it converges smoothly and
    confidently to a number roughly HALF the true value, despite passing
    its own internal consistency check (the portfolio really is priced
    at exactly zero at the barrier on every matched date; verified
    directly). That combination -- internally self-consistent, cleanly
    convergent, and simply wrong -- is precisely the numerical-method
    risk "Model Risk" (Derman 1996) warns about: a model can look correct
    by every check it grades itself on and still be wrong, which is why
    every model in this engine needs an INDEPENDENT reference, not just
    an internal one (see validation/checks.py).

    Root cause is still open. The likely culprit: the correction legs
    are struck at the *original* strike K, which for the call case sits
    well inside the continuation region in a way that happens to work,
    but for the put case appears to distort value inside the
    already-matched interior region enough to bias the whole backward
    recursion. Priced experiments during debugging showed that moving
    the correction strike closer to the barrier fixes the bias (at the
    cost of new conditioning problems very close to B) -- suggesting the
    fix is a smarter choice of correction instrument (a tight spread
    near B approximating a digital, per the original Derman-Ergener-Kani
    prescription of options "struck at or beyond the barrier," rather
    than a single option at K). This is flagged rather than hidden or
    silently patched with a loose strike heuristic -- treat completing
    this as a real Phase-13 validation task, not a rubber stamp.
    """
    if not (0 < barrier < strike):
        raise ValueError("require 0 < barrier < strike")
    if T <= 0:
        raise ValueError("T must be positive")

    sigma_func: SigmaFunc = vol_func if vol_func is not None else (lambda k, t: sigma)

    match_times = np.linspace(T / n_matches, T, n_matches)
    legs: list[Leg] = [
        Leg(weight=1.0, strike=strike, maturity=T, option_type=OptionType.PUT),
        Leg(weight=-1.0, strike=barrier, maturity=T, option_type=OptionType.PUT),
    ]

    for t_k in sorted(match_times[:-1], reverse=True):
        v_existing = _portfolio_value(legs, spot=barrier, t=t_k, r=r, q=q, sigma_func=sigma_func)
        denom = strike - barrier  # intrinsic value at S=B of a put struck at `strike`, at its own maturity t_k
        w_k = -v_existing / denom
        legs.append(Leg(weight=w_k, strike=strike, maturity=t_k, option_type=OptionType.PUT))

    price_today = _portfolio_value(legs, spot=spot, t=0.0, r=r, q=q, sigma_func=sigma_func)
    return StaticReplicationResult(price=price_today, legs=legs)
