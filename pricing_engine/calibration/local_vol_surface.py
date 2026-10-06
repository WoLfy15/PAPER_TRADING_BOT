"""
Calibration — local volatility surface (the project's connective tissue).

Takes a market implied-vol smile (per-maturity quadratic fits in log-
moneyness) and extracts Dupire's local volatility surface sigma_loc(K,T)
via finite differences on the fitted call-price surface. This
operationalizes the framing in "The Volatility Smile and Its Implied
Tree" (Derman-Kani 1994) and "The Local Volatility Surface"
(Derman-Kani-Zou 1995): implied vol is the options-market analog of
yield-to-maturity; local vol is the analog of the forward rate, and it is
what a model should actually simulate/hedge with, not the implied vol
itself.

Design choice, stated plainly: this deliberately does NOT re-derive the
original Derman-Kani node-by-node combinatorial binomial tree. That
construction has known negative-probability / arbitrage-overwrite edge
cases (documented in the paper itself and in the 1996 trinomial-tree
follow-up) that are easy to get subtly wrong under a time-boxed
implementation -- exactly the numerical-method risk the "Model Risk"
paper warns about. Instead, local vol is extracted directly from
Dupire's PDE in price space:

    sigma_loc(K,T)^2 = [dC/dT + q*C + (r-q)*K*dC/dK] / (0.5 * K^2 * d2C/dK2)

evaluated on a smooth fitted smile surface. This is mathematically
equivalent to Derman-Kani in the continuous-time limit, is how local vol
is actually computed in practice (Gatheral, "The Volatility Surface",
Ch.1-2), and plugs directly into Team D's Monte Carlo engine
(models/local_vol_mc.py) exactly the way the 1995 paper's own lookback-
option worked example does: simulate under sigma_loc(S_t, t) instead of
a single flat sigma.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pricing_engine.market_data.contract import MarketSmilePoint, OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()


@dataclass
class SmileSlice:
    """A single-maturity smile fit as a quadratic in log-moneyness
    k = ln(K/F): iv(k) = a + b*k + c*k^2. This is the simplest smile
    parameterization that is smooth and twice-differentiable everywhere
    (required for Dupire) while still capturing skew (b) and curvature
    (c); swap in SVI here if a production-grade fit is needed later."""

    expiry_years: float
    forward: float
    coeffs: np.ndarray  # [c, b, a], highest degree first (np.polyfit convention)

    @classmethod
    def fit(cls, forward: float, expiry_years: float, points: Sequence[MarketSmilePoint]) -> "SmileSlice":
        if len(points) < 3:
            raise ValueError("need >=3 strikes to fit a quadratic smile slice")
        k = np.array([math.log(p.strike / forward) for p in points])
        iv = np.array([p.implied_vol for p in points])
        coeffs = np.polyfit(k, iv, deg=2)
        return cls(expiry_years=expiry_years, forward=forward, coeffs=coeffs)

    def iv(self, strike: float) -> float:
        k = math.log(strike / self.forward)
        c, b, a = self.coeffs
        return max(a + b * k + c * k * k, 1e-4)

    def total_variance(self, strike: float) -> float:
        v = self.iv(strike)
        return v * v * self.expiry_years


class LocalVolatilitySurface:
    """Builds sigma_loc(K,T) from a set of per-maturity SmileSlice fits.
    Cross-maturity interpolation is done in TOTAL VARIANCE w(K,T) =
    iv(K,T)^2 * T, linearly interpolated between the two bracketing
    maturities -- variance is roughly additive over time, which is the
    standard way to avoid obvious calendar-spread arbitrage in the fit."""

    def __init__(self, spot: float, r: float, q: float, slices: Sequence[SmileSlice]):
        if len(slices) < 2:
            raise ValueError("need >=2 maturity slices to estimate dC/dT")
        self.spot = spot
        self.r = r
        self.q = q
        self.slices = sorted(slices, key=lambda s: s.expiry_years)
        self.t_min = self.slices[0].expiry_years
        self.t_max = self.slices[-1].expiry_years
        self._grid_interp: RegularGridInterpolator | None = None

    def implied_vol(self, strike: float, t: float) -> float:
        t = min(max(t, self.t_min), self.t_max)
        if t <= self.slices[0].expiry_years:
            return self.slices[0].iv(strike)
        if t >= self.slices[-1].expiry_years:
            return self.slices[-1].iv(strike)
        for lo, hi in zip(self.slices[:-1], self.slices[1:]):
            if lo.expiry_years <= t <= hi.expiry_years:
                w_lo, w_hi = lo.total_variance(strike), hi.total_variance(strike)
                frac = (t - lo.expiry_years) / (hi.expiry_years - lo.expiry_years)
                w = w_lo + frac * (w_hi - w_lo)
                return math.sqrt(max(w, 1e-10) / t)
        raise RuntimeError("unreachable: t outside bracketed range")

    def _call_price(self, strike: float, t: float) -> float:
        t = max(t, 1e-6)
        iv = self.implied_vol(strike, t)
        quote = OptionQuote(
            underlying="local_vol_surface",
            spot=self.spot,
            strike=strike,
            expiry_years=t,
            option_type=OptionType.CALL,
            risk_free_rate=self.r,
            dividend_yield=self.q,
        )
        return _bsm.price(quote, iv)

    def local_vol(self, strike: float, t: float, dk_rel: float = 1e-3, dt_abs: float = 1e-4) -> float:
        """Dupire local volatility at (K,T) via central finite differences
        on the fitted call-price surface."""
        K = strike
        dK = max(K * dk_rel, 1e-6)
        t_eval = min(max(t, self.t_min + dt_abs), self.t_max - dt_abs)

        C = self._call_price(K, t_eval)
        C_up_k = self._call_price(K + dK, t_eval)
        C_dn_k = self._call_price(K - dK, t_eval)
        dC_dK = (C_up_k - C_dn_k) / (2 * dK)
        d2C_dK2 = (C_up_k - 2 * C + C_dn_k) / (dK * dK)

        C_up_t = self._call_price(K, t_eval + dt_abs)
        C_dn_t = self._call_price(K, t_eval - dt_abs)
        dC_dT = (C_up_t - C_dn_t) / (2 * dt_abs)

        numerator = dC_dT + self.q * C + (self.r - self.q) * K * dC_dK
        denominator = 0.5 * K * K * d2C_dK2

        if denominator <= 1e-10 or numerator / denominator <= 0:
            # Convexity has numerically vanished (deep wing / coarse fit),
            # or the finite-difference estimate went negative. Fall back to
            # the local implied vol rather than return garbage -- exactly
            # the "never ignore a numerical discrepancy silently, and have
            # a documented fallback" discipline the Model Risk paper asks
            # for. Team E's scenario engine should log/flag these events.
            return self.implied_vol(K, t_eval)
        return math.sqrt(numerator / denominator)

    def build_lookup_grid(
        self, s_min: float, s_max: float, n_s: int = 40, n_t: int = 20
    ) -> "RegularGridInterpolator":
        """Precompute sigma_loc on an (S,t) grid and return a fast
        interpolant. Dupire-via-finite-differences costs ~6 BSM calls per
        (K,T) query; a Monte Carlo engine needs one lookup per path per
        step, so this caching step is what makes models/local_vol_mc.py
        tractable at path counts that matter (real desks do the same)."""
        s_grid = np.linspace(s_min, s_max, n_s)
        t_grid = np.linspace(self.t_min, self.t_max, n_t)
        vals = np.empty((n_s, n_t))
        for i, s in enumerate(s_grid):
            for j, t in enumerate(t_grid):
                vals[i, j] = self.local_vol(s, t)
        interp = RegularGridInterpolator(
            (s_grid, t_grid), vals, bounds_error=False, fill_value=None
        )
        self._grid_interp = interp
        return interp

    def fast_local_vol(self, strike: float, t: float) -> float:
        if self._grid_interp is None:
            raise RuntimeError("call build_lookup_grid(...) first")
        s_clamped = np.clip(strike, self._grid_interp.grid[0][0], self._grid_interp.grid[0][-1])
        t_clamped = np.clip(t, self._grid_interp.grid[1][0], self._grid_interp.grid[1][-1])
        return float(self._grid_interp([[s_clamped, t_clamped]])[0])
