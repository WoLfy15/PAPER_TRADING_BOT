"""
pricing_engine/risk/friction_hedging.py

Every delta-hedge in this project so far (backtest.py, capital_backtest.py,
models/local_vol_mc.py) rehedges to EXACT Black-Scholes delta every day at
zero cost. Real hedging is neither continuous nor free. Two classical,
complementary fixes:

Leland (1985) -- HOW MUCH to charge/reserve for hedging costs, given you
WILL rehedge on a fixed schedule (e.g. daily): replace sigma with an
adjusted sigma_hat in the pricing formula itself, so the theoretical
price already contains the expected transaction-cost drag:

    sigma_hat^2 = sigma^2 * (1 + sqrt(2/pi) * (k / (sigma*sqrt(dt))) * sign(Gamma))

k is the proportional cost per unit of underlying traded (e.g. 0.001 for
10bp); dt is the rehedging interval. sign(Gamma) is with respect to the
HEDGER'S position: a hedger who is short gamma (has sold options and must
buy high/sell low to stay delta-neutral) needs sigma_hat > sigma to
charge enough to cover that cost; a hedger who is long gamma effectively
hedges profitably on average, so sigma_hat < sigma is the fair value THEY
should be willing to pay. leland_adjusted_vol()'s `position` argument
picks the sign explicitly rather than asking the caller to track sign(Gamma)
themselves -- a well-known source of errors in this formula.

Whalley & Wilmott (1997) -- HOW OFTEN to rehedge in the first place: an
asymptotic (small-cost) no-transaction band around Black-Scholes delta,

    H(t) = [ (3/2) * k * S^2 * Gamma^2 * e^{r(T-t)} / gamma_risk ]^(1/3)

derived from a utility-maximization tradeoff between transaction costs
and hedging-error risk. The well-established, robust part of this result
-- and the part this module leans on -- is the cube-root scaling in k
(band width ~ k^(1/3), the signature result of this whole literature,
also found in related work by Hodges-Neuberger and others): transaction
costs matter a LOT even when small, because the optimal response to a
tiny cost is a first-order (not infinitesimal) no-trade region. The exact
leading constant is more convention-dependent across sources; treat it
as the standard textbook constant, not a re-derivation from source.

Applied to the real 10-year backtest in
real_market_validation/friction_hedging_backtest.py -- both changes go
directly onto the strategy backtest.py already validated, not a toy
example.
"""
from __future__ import annotations

import math
from typing import Literal


def leland_adjusted_vol(sigma: float, k: float, dt: float, position: Literal["short", "long"]) -> float:
    """sigma: the model's own (e.g. BSM) volatility. k: proportional
    transaction cost (e.g. 0.0005 = 5bp of notional per trade). dt: the
    rehedging interval in years (1/252 for daily). position: 'short'
    (you've sold the option and are hedging it -- charge MORE, sigma_hat
    > sigma) or 'long' (you've bought it -- it's worth LESS to you net of
    hedging cost, sigma_hat < sigma).

    Degenerates to sigma exactly as k -> 0 (frictionless) -- checked in
    tests/test_friction_hedging.py alongside the sign convention."""
    if sigma <= 0 or dt <= 0:
        raise ValueError("sigma and dt must be positive")
    sign = 1.0 if position == "short" else -1.0
    adjustment = math.sqrt(2.0 / math.pi) * (k / (sigma * math.sqrt(dt))) * sign
    sigma_hat_sq = sigma * sigma * (1.0 + adjustment)
    return math.sqrt(max(sigma_hat_sq, 1e-10))


def leland_cost_bps_of_vega(sigma: float, k: float, dt: float) -> float:
    """A more intuitive-to-read summary than the raw adjusted vol: the
    Leland cost expressed as roughly how many vol points it adds,
    independent of position sign (magnitude only)."""
    hedged = leland_adjusted_vol(sigma, k, dt, "short")
    return (hedged - sigma) * 10000  # in "vol bps"


def whalley_wilmott_band(spot: float, gamma: float, k: float, risk_aversion: float,
                          r: float, time_to_expiry: float) -> float:
    """Half-width of the no-trade band around Black-Scholes delta, in
    delta units (i.e. rehedge only when |current delta - last hedged
    delta| exceeds this). risk_aversion: the trader's local risk-
    aversion parameter (higher = tighter bands, more willing to pay
    transaction costs to stay closer to delta-neutral); there is no
    "correct" universal value -- it is a preference input, calibrated to
    how much tracking error a desk will tolerate, same role a Sharpe-
    ratio target or VaR limit plays elsewhere in this project's capital
    layer (real_market_validation/capital_backtest.py)."""
    if k <= 0:
        return 0.0
    numerator = 1.5 * k * (spot ** 2) * (gamma ** 2) * math.exp(r * time_to_expiry)
    return (numerator / risk_aversion) ** (1.0 / 3.0)
