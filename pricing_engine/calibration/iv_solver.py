"""
Team B — Phase 3: implied volatility solver.
Newton-Raphson (fast, uses analytic vega) with an automatic fall back to
Brent's method (robust, derivative-free) when Newton fails to converge or
steps outside a sane vol range — exactly the kind of "don't trust one
numerical method blindly" discipline the Model Risk paper argues for.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

from pricing_engine.market_data.contract import OptionQuote
from pricing_engine.models.base import PricingModel


class IVSolverError(RuntimeError):
    pass


@dataclass
class IVResult:
    implied_vol: float
    iterations: int
    method: str


def solve_implied_vol(
    model: PricingModel,
    quote: OptionQuote,
    target_price: float,
    initial_guess: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 50,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> IVResult:
    if target_price <= 0:
        raise IVSolverError("target_price must be positive")

    # --- Newton-Raphson using the model's own analytic vega ---
    sigma = initial_guess
    for i in range(1, max_iter + 1):
        try:
            price = model.price(quote, sigma)
            vega = model.vega(quote, sigma)
        except ValueError:
            break
        diff = price - target_price
        if abs(diff) < tol:
            return IVResult(implied_vol=sigma, iterations=i, method="newton")
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= lo or sigma >= hi or not (sigma == sigma):  # NaN guard
            break

    # --- Brent fallback: robust bisection-style search on [lo, hi] ---
    def f(s: float) -> float:
        return model.price(quote, s) - target_price

    try:
        f_lo, f_hi = f(lo), f(hi)
    except ValueError as exc:
        raise IVSolverError(f"model failed to price at solver bounds: {exc}") from exc

    if f_lo * f_hi > 0:
        raise IVSolverError(
            f"target_price {target_price:.6f} not bracketed by vol range "
            f"[{lo}, {hi}] -> prices [{f_lo + target_price:.6f}, {f_hi + target_price:.6f}]. "
            "Likely a stale/crossed quote or an arbitrage violation upstream."
        )

    root, results = brentq(f, lo, hi, xtol=tol, full_output=True)
    return IVResult(implied_vol=root, iterations=results.iterations, method="brent")
