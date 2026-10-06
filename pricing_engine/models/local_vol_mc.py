"""
Team D extension — Monte Carlo pricing under a calibrated local
volatility surface (Phase 6, upgraded).

Standard GBM Monte Carlo simulates dS = (r-q)S dt + sigma S dW with a
single flat sigma. Once calibration/local_vol_surface.py has extracted
sigma_loc(S,t) from the market smile, the natural extension -- the one
"The Local Volatility Surface" paper itself uses for its lookback-option
example -- is to simulate under sigma_loc(S_t, t) instead: at every
timestep, look up the local vol at the CURRENT simulated spot and time,
not a single constant. Antithetic variates are kept for variance
reduction, matching Team D's existing engine; both a flat-vol run and a
local-vol run can price the exact same exotic side by side to quantify
the "smile risk" a flat-vol model silently ignores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

VolFunc = Callable[[np.ndarray, float], np.ndarray]  # (S_array, t) -> sigma_array


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


def simulate_paths(
    spot: float,
    r: float,
    q: float,
    T: float,
    n_steps: int,
    n_paths: int,
    vol_func: VolFunc,
    antithetic: bool = True,
    seed: int | None = 7,
) -> np.ndarray:
    """Euler-Maruyama simulation of dS = (r-q)S dt + sigma(S,t) S dW.
    vol_func is vectorized: called once per timestep with the whole
    cross-section of paths, not once per path (this is what makes local
    vol MC fast enough to be usable -- see LocalVolatilitySurface's grid
    interpolant, which is itself vectorized).
    Returns an array of shape (n_paths, n_steps+1) including S0 at t=0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqdt = np.sqrt(dt)

    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    m = z.shape[0]

    paths = np.empty((m, n_steps + 1))
    paths[:, 0] = spot
    for step in range(n_steps):
        t = step * dt
        S_t = paths[:, step]
        sigma_t = np.asarray(vol_func(S_t, t))
        drift = (r - q - 0.5 * sigma_t ** 2) * dt
        diffusion = sigma_t * sqdt * z[:, step]
        paths[:, step + 1] = S_t * np.exp(drift + diffusion)
    return paths


def _mc_result(discounted_payoff: np.ndarray) -> MCResult:
    price = discounted_payoff.mean()
    se = discounted_payoff.std(ddof=1) / np.sqrt(len(discounted_payoff))
    return MCResult(price=float(price), std_error=float(se), n_paths=len(discounted_payoff))


def price_european(
    spot, strike, r, q, T, n_steps, n_paths, vol_func: VolFunc,
    option_type: Literal["call", "put"] = "call", antithetic=True, seed=7,
) -> MCResult:
    paths = simulate_paths(spot, r, q, T, n_steps, n_paths, vol_func, antithetic, seed)
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - strike, 0.0) if option_type == "call" else np.maximum(strike - S_T, 0.0)
    return _mc_result(np.exp(-r * T) * payoff)


def price_asian(
    spot, strike, r, q, T, n_steps, n_paths, vol_func: VolFunc,
    option_type: Literal["call", "put"] = "call", antithetic=True, seed=7,
) -> MCResult:
    paths = simulate_paths(spot, r, q, T, n_steps, n_paths, vol_func, antithetic, seed)
    avg = paths[:, 1:].mean(axis=1)
    payoff = np.maximum(avg - strike, 0.0) if option_type == "call" else np.maximum(strike - avg, 0.0)
    return _mc_result(np.exp(-r * T) * payoff)


def price_lookback(
    spot, r, q, T, n_steps, n_paths, vol_func: VolFunc,
    option_type: Literal["call", "put"] = "call", antithetic=True, seed=7,
) -> MCResult:
    """Floating-strike lookback: call pays S_T - min(S); put pays max(S) - S_T."""
    paths = simulate_paths(spot, r, q, T, n_steps, n_paths, vol_func, antithetic, seed)
    if option_type == "call":
        payoff = paths[:, -1] - paths.min(axis=1)
    else:
        payoff = paths.max(axis=1) - paths[:, -1]
    return _mc_result(np.exp(-r * T) * payoff)


def flat_vol_func(sigma: float) -> VolFunc:
    return lambda S, t: np.full_like(S, sigma)


def local_vol_func(surface) -> VolFunc:
    """surface: a LocalVolatilitySurface with build_lookup_grid(...) already
    called. Vectorized via the grid interpolator."""
    def f(S: np.ndarray, t: float) -> np.ndarray:
        t_c = min(max(t, surface.t_min), surface.t_max)
        pts = np.column_stack([np.clip(S, surface._grid_interp.grid[0][0], surface._grid_interp.grid[0][-1]),
                                np.full_like(S, t_c)])
        return surface._grid_interp(pts)
    return f
