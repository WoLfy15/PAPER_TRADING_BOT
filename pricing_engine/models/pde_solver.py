"""
pricing_engine/models/pde_solver.py

Direct finite-difference solution of the Black-Scholes PDE, in log-spot
x = ln(S) (constant-coefficient in x, so a uniform grid is exact rather
than an approximation of the non-uniform S-grid the PDE has in S itself):

    dV/dt + 0.5*sigma^2 d2V/dx2 + (r-q-0.5*sigma^2) dV/dx - r*V = 0

solved backward from T to 0.

Crank-Nicolson (average of explicit and implicit discretization) is
second-order accurate in both time and space and unconditionally stable
-- but CN is well known to ring/oscillate near the non-smooth kink in a
vanilla payoff at the strike unless the first few steps are damped
(Rannacher 1984): this solver takes `rannacher_steps` fully-implicit
(first-order but non-oscillatory) steps immediately after the terminal
payoff before switching to CN for the remainder, which is the standard
fix and is on by default.

Each implicit time step is a tridiagonal linear system, solved via the
Thomas algorithm (O(n) per step, vs. O(n^3) for a generic solver) --
`_thomas_solve`. American exercise is enforced with PSOR (Projected
SOR): the same tridiagonal system, solved iteratively with a
successive-over-relaxation update, projected onto the exercise
constraint V >= intrinsic value after every sweep, repeated to
convergence at every time step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.base import PricingModel, FiniteDifferenceGreeksMixin


def _thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves a tridiagonal system Ax=rhs in O(n). lower[0] and upper[-1]
    are unused (no sub-/super-diagonal entry there) -- kept as full-length
    arrays so all four arrays share one index convention."""
    n = len(diag)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


@dataclass
class PDEGridResult:
    x_grid: np.ndarray       # log-spot grid
    spot_grid: np.ndarray    # exp(x_grid)
    values: np.ndarray       # option value at t=0 on spot_grid
    n_exercise_boundary: int  # American only: grid index nearest the exercise boundary at t=0 (-1 if none/European)


def _boundary_values(is_call: bool, S_hi: float, K: float, r: float, q: float, tau: float) -> tuple[float, float]:
    """Analytic (BSM-consistent) boundary values at time-to-expiry tau, at
    the two edges of the grid -- far more accurate this far out than a
    naive zero-gamma / linear extrapolation."""
    if is_call:
        return 0.0, S_hi * math.exp(-q * tau) - K * math.exp(-r * tau)
    return K * math.exp(-r * tau), 0.0


def price_pde(
    quote: OptionQuote, sigma: float, american: bool = False,
    n_space: int = 400, n_time: int = 400, x_width: float = 4.0,
    rannacher_steps: int = 2, psor_omega: float = 1.5, psor_tol: float = 1e-8, psor_max_iter: int = 200,
) -> PDEGridResult:
    """x_width: half-width of the log-spot grid (default 4 -> roughly
    e^{-4} to e^{+4} times spot, generously wide so the boundary
    condition is accurate near the region that matters). n_space,
    n_time: grid resolution -- see tests/test_pde_solver.py for the
    convergence check that motivates the defaults."""
    S0, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
    is_call = quote.option_type == OptionType.CALL

    x0 = math.log(S0)
    x_min, x_max = x0 - x_width, x0 + x_width
    dx = (x_max - x_min) / n_space
    x = np.linspace(x_min, x_max, n_space + 1)
    S = np.exp(x)
    dt = T / n_time

    payoff = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    V = payoff.copy()

    drift = r - q - 0.5 * sigma * sigma
    a = 0.5 * sigma * sigma / (dx * dx) - 0.5 * drift / dx   # coefficient on V[i-1]
    b = -sigma * sigma / (dx * dx) - r                        # coefficient on V[i]
    c = 0.5 * sigma * sigma / (dx * dx) + 0.5 * drift / dx    # coefficient on V[i+1]

    n_interior = n_space - 1  # grid indices 1..n_space-1; 0 and n_space held at Dirichlet boundary values

    exercise_boundary_idx = -1
    tau = 0.0
    for step in range(n_time):
        tau_next = tau + dt
        theta_cn = 1.0 if step < rannacher_steps else 0.5  # 1.0 = fully implicit (Rannacher damping), 0.5 = Crank-Nicolson

        lo_next, hi_next = _boundary_values(is_call, S[-1], K, r, q, tau_next)

        lower = np.zeros(n_interior)
        diag = np.zeros(n_interior)
        upper = np.zeros(n_interior)
        rhs = np.zeros(n_interior)

        for j in range(n_interior):
            i = j + 1
            lower[j] = -theta_cn * dt * a
            diag[j] = 1.0 - theta_cn * dt * b
            upper[j] = -theta_cn * dt * c
            explicit_term = (1 - theta_cn) * dt * (a * V[i - 1] + b * V[i] + c * V[i + 1]) if theta_cn < 1.0 else 0.0
            rhs[j] = V[i] + explicit_term

        rhs[0] -= lower[0] * lo_next
        rhs[-1] -= upper[-1] * hi_next

        if american:
            interior_payoff = payoff[1:n_space]
            v_interior = V[1:n_space].copy()
            for _ in range(psor_max_iter):
                max_change = 0.0
                for j in range(n_interior):
                    left = v_interior[j - 1] if j > 0 else lo_next
                    right = v_interior[j + 1] if j < n_interior - 1 else hi_next
                    gs_value = (rhs[j] - lower[j] * left - upper[j] * right) / diag[j]
                    sor_value = v_interior[j] + psor_omega * (gs_value - v_interior[j])
                    projected = max(sor_value, interior_payoff[j])
                    max_change = max(max_change, abs(projected - v_interior[j]))
                    v_interior[j] = projected
                if max_change < psor_tol:
                    break
            V[1:n_space] = v_interior
        else:
            V[1:n_space] = _thomas_solve(lower, diag, upper, rhs)

        V[0], V[-1] = lo_next, hi_next
        tau = tau_next

    if american:
        exercised = V <= payoff + 1e-10
        idx = np.where(exercised)[0]
        exercise_boundary_idx = int(idx[np.argmin(np.abs(x[idx] - x0))]) if len(idx) else -1

    return PDEGridResult(x_grid=x, spot_grid=S, values=V, n_exercise_boundary=exercise_boundary_idx)


def price_pde_at_spot(quote: OptionQuote, sigma: float, american: bool = False, **kwargs) -> float:
    """Convenience wrapper: run the full grid and interpolate to the
    quote's own spot -- what PDEModel.price() calls."""
    result = price_pde(quote, sigma, american=american, **kwargs)
    return float(np.interp(quote.spot, result.spot_grid, result.values))


class PDEModel(PricingModel, FiniteDifferenceGreeksMixin):
    """European or American vanilla pricing via Crank-Nicolson/PSOR.
    Conforms to PricingModel so it can be run through validation/
    checks.py's suite exactly like every other model in this project."""
    name = "PDE-CrankNicolson"

    def __init__(self, american: bool = False, n_space: int = 400, n_time: int = 400):
        self.american = american
        self.n_space = n_space
        self.n_time = n_time

    def price(self, quote: OptionQuote, sigma: float) -> float:
        return price_pde_at_spot(quote, sigma, american=self.american, n_space=self.n_space, n_time=self.n_time)

    def delta(self, quote: OptionQuote, sigma: float) -> float:
        return self.fd_delta(quote, sigma)

    def gamma(self, quote: OptionQuote, sigma: float) -> float:
        return self.fd_gamma(quote, sigma)

    def vega(self, quote: OptionQuote, sigma: float) -> float:
        return self.fd_vega(quote, sigma)

    def theta(self, quote: OptionQuote, sigma: float) -> float:
        return self.fd_theta(quote, sigma)

    def rho(self, quote: OptionQuote, sigma: float) -> float:
        return self.fd_rho(quote, sigma)
