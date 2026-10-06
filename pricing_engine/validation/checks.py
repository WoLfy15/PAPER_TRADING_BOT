"""
validation/checks.py — Phase 13, built directly from Derman's "Model
Risk" (1996) prescriptions rather than generic unit tests:
  - test every model against simple known closed-form solutions first
  - test boundary behavior (e.g. an option model should reduce to a
    forward deep in-the-money)
  - never silently ignore a small numerical discrepancy -- report it
  - check convergence explicitly rather than assuming more steps/paths
    always means "more correct"

Every check returns a CheckResult (pass/fail + measured value +
tolerance) rather than raising, so Team E's dashboard can render a full
validation report instead of a stack trace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.base import PricingModel


@dataclass
class CheckResult:
    name: str
    passed: bool
    measured: float
    tolerance: float
    detail: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: measured={self.measured:.6g} (tol={self.tolerance:.2g}) {self.detail}"


def put_call_parity(
    model: PricingModel, spot: float, strike: float, T: float, r: float, q: float, sigma: float, tol: float = 1e-6
) -> CheckResult:
    """C - P == S*e^{-qT} - K*e^{-rT} for any correct European model."""
    call_q = OptionQuote("chk", spot, strike, T, OptionType.CALL, r, q)
    put_q = OptionQuote("chk", spot, strike, T, OptionType.PUT, r, q)
    lhs = model.price(call_q, sigma) - model.price(put_q, sigma)
    rhs = spot * math.exp(-q * T) - strike * math.exp(-r * T)
    diff = abs(lhs - rhs)
    return CheckResult("put_call_parity", diff < tol, diff, tol, f"lhs={lhs:.6f} rhs={rhs:.6f}")


def no_arbitrage_bounds(
    model: PricingModel, spot: float, strike: float, T: float, r: float, q: float, sigma: float, tol: float = 1e-6
) -> CheckResult:
    """max(0, S e^{-qT} - K e^{-rT}) <= C <= S e^{-qT}."""
    call_q = OptionQuote("chk", spot, strike, T, OptionType.CALL, r, q)
    c = model.price(call_q, sigma)
    lower = max(0.0, spot * math.exp(-q * T) - strike * math.exp(-r * T))
    upper = spot * math.exp(-q * T)
    ok = (lower - tol) <= c <= (upper + tol)
    return CheckResult("no_arbitrage_bounds", ok, c, tol, f"bounds=[{lower:.6f}, {upper:.6f}]")


def deep_itm_forward_limit(
    model: PricingModel, spot: float, T: float, r: float, q: float, sigma: float, tol: float = 1e-3
) -> CheckResult:
    """A call struck near zero should reduce to the forward value, per
    the Model Risk paper's own recommended boundary test."""
    tiny_k = spot * 1e-4
    call_q = OptionQuote("chk", spot, tiny_k, T, OptionType.CALL, r, q)
    c = model.price(call_q, sigma)
    forward_value = spot * math.exp(-q * T) - tiny_k * math.exp(-r * T)
    diff = abs(c - forward_value)
    return CheckResult("deep_itm_forward_limit", diff < tol, diff, tol)


def monotonic_in_strike(
    model: PricingModel, spot: float, strikes: np.ndarray, T: float, r: float, q: float, sigma: float
) -> CheckResult:
    """Call prices must be non-increasing in strike; put prices
    non-decreasing. Flags a smile-fit or calibration bug immediately --
    exactly the kind of "small numerical discrepancy" the paper warns
    against ignoring."""
    strikes = np.sort(strikes)
    call_prices = [model.price(OptionQuote("chk", spot, k, T, OptionType.CALL, r, q), sigma) for k in strikes]
    diffs = np.diff(call_prices)
    worst = float(diffs.max()) if len(diffs) else 0.0
    return CheckResult("monotonic_in_strike", worst <= 1e-8, worst, 1e-8, "max positive step in call price vs K")


def american_geq_european(american_price: float, european_price: float, tol: float = 1e-8) -> CheckResult:
    """American options must never be worth less than their European
    counterpart (early exercise is an option, not an obligation) --
    Team C's CRR/Leisen-Reimer output should be checked against Team B's
    BSM output with this on every run."""
    diff = european_price - american_price
    return CheckResult("american_geq_european", diff <= tol, diff, tol)


def convergence_check(
    step_counts: list[int], prices_by_steps: list[float], reference_price: float, tol: float = 5e-3
) -> CheckResult:
    """Checks that the LAST (finest) numerical price is within tolerance
    of a reference (closed-form or very-fine-grid) price, and that error
    is trending down as steps increase -- not just that it happens to
    land close by chance."""
    errors = [abs(p - reference_price) for p in prices_by_steps]
    final_err = errors[-1]
    trending_down = all(errors[i] >= errors[i + 1] - 1e-9 for i in range(len(errors) - 2)) if len(errors) > 2 else True
    passed = (final_err < tol) and trending_down
    detail = f"errors={['%.5f' % e for e in errors]} trending_down={trending_down}"
    return CheckResult("convergence", passed, final_err, tol, detail)


def run_bsm_suite(model: PricingModel, spot=100.0, T=1.0, r=0.03, q=0.01, sigma=0.2) -> list[CheckResult]:
    """Convenience bundle: everything a new model should pass before it's
    trusted anywhere else in the engine."""
    strikes = np.linspace(spot * 0.5, spot * 1.5, 15)
    return [
        put_call_parity(model, spot, spot, T, r, q, sigma),
        no_arbitrage_bounds(model, spot, spot * 0.9, T, r, q, sigma),
        deep_itm_forward_limit(model, spot, T, r, q, sigma),
        monotonic_in_strike(model, spot, strikes, T, r, q, sigma),
    ]
