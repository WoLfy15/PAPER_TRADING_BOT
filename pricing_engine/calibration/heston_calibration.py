"""
pricing_engine/calibration/heston_calibration.py

Joint calibration of HestonParams (v0, kappa, theta, xi, rho) to a
cross-sectional market vol surface (any number of maturities/strikes),
in implied-vol space rather than price space -- fitting prices directly
overweights deep ITM options by dollar magnitude and effectively ignores
OTM wings, which is exactly backwards from what a skew calibration needs
to get right.

Optimizer: scipy.optimize.least_squares with method="trf" (Trust Region
Reflective). This is the bounded-constraint sibling of classical
Levenberg-Marquardt -- scipy's literal "lm" method does not support
parameter bounds, and bounds are what keep kappa/theta/xi positive and
rho in (-1,1) without the optimizer wandering into a region where the
characteristic function's complex sqrt/log picks up a different branch.
A soft penalty residual enforces the Feller condition (2*kappa*theta >
xi**2) -- violating it doesn't break pricing (simulate_paths floors
variance regardless), but a calibration that has to lean on that floor
rather than the SDE's own non-negativity is worth flagging, not hiding.

Every model evaluation reprices an ENTIRE maturity slice at once via
heston.price_fft_grid rather than one strike at a time -- this is the
concrete payoff of having built the FFT path: a joint calibration across
several maturities needs on the order of (iterations x maturities) FFT
calls rather than (iterations x maturities x strikes) individual
quadrature calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from pricing_engine.market_data.contract import MarketSmilePoint, OptionQuote, OptionType
from pricing_engine.models.bsm import BlackScholesMerton
from pricing_engine.models.heston import HestonParams, price_fft_grid
from pricing_engine.calibration.iv_solver import solve_implied_vol

_bsm = BlackScholesMerton()

# (v0, kappa, theta, xi, rho)
_LOWER = (1e-4, 1e-3, 1e-4, 1e-3, -0.999)
_UPPER = (4.0, 20.0, 4.0, 5.0, 0.999)


@dataclass
class HestonCalibrationResult:
    params: HestonParams
    rmse_vol_pts: float          # root-mean-square IV error, in vol POINTS (e.g. 0.4 = 0.4 vol pts)
    max_abs_error_vol_pts: float
    n_points: int
    n_iterations: int
    success: bool
    message: str
    feller_ratio: float


def _model_ivs_for_maturity(spot: float, r: float, q: float, T: float,
                             strikes: np.ndarray, params: HestonParams) -> np.ndarray:
    prices = price_fft_grid(spot, strikes, T, r, q, params)
    ivs = np.empty(len(strikes))
    for i, (K, price) in enumerate(zip(strikes, prices)):
        quote = OptionQuote("cal", spot, float(K), T, OptionType.CALL, r, q)
        try:
            ivs[i] = solve_implied_vol(_bsm, quote, max(float(price), 1e-8)).implied_vol
        except Exception:
            ivs[i] = np.nan
    return ivs


def calibrate_heston(
    points: Sequence[MarketSmilePoint], spot: float, r: float, q: float,
    x0: Optional[HestonParams] = None, feller_penalty_weight: float = 5.0,
    max_nfev: int = 400,
) -> HestonCalibrationResult:
    """points may span several maturities -- grouped internally so each
    is priced with a single price_fft_grid call per optimizer iteration.
    Market quotes are calls by convention (put-call parity makes the
    choice immaterial to the fitted vol); pass MarketSmilePoint objects
    exactly as calibration/local_vol_surface.py's SmileSlice.fit expects,
    so the same market data feeds either calibration unchanged."""
    maturities = sorted(set(p.expiry_years for p in points))
    by_maturity = {T: [p for p in points if p.expiry_years == T] for T in maturities}
    strikes_by_T = {T: np.array([p.strike for p in pts]) for T, pts in by_maturity.items()}
    market_iv_by_T = {T: np.array([p.implied_vol for p in pts]) for T, pts in by_maturity.items()}

    if x0 is None:
        atm_var = float(np.median([p.implied_vol for p in points])) ** 2
        x0 = HestonParams(v0=atm_var, kappa=2.0, theta=atm_var, xi=0.5, rho=-0.5)
    x0_vec = np.array([x0.v0, x0.kappa, x0.theta, x0.xi, x0.rho])

    def residuals(x: np.ndarray) -> np.ndarray:
        params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
        resid = []
        for T in maturities:
            model_iv = _model_ivs_for_maturity(spot, r, q, T, strikes_by_T[T], params)
            diff = model_iv - market_iv_by_T[T]
            diff = np.where(np.isnan(diff), 0.5, diff)  # a failed IV inversion is a bad fit, not a free pass
            resid.append(diff)
        resid = np.concatenate(resid)
        feller_gap = 2 * params.kappa * params.theta - params.xi ** 2
        feller_resid = feller_penalty_weight * max(-feller_gap, 0.0)
        return np.concatenate([resid, [feller_resid]])

    result = least_squares(residuals, x0_vec, bounds=(_LOWER, _UPPER), method="trf", max_nfev=max_nfev)
    fitted = HestonParams(v0=result.x[0], kappa=result.x[1], theta=result.x[2], xi=result.x[3], rho=result.x[4])

    pricing_resid = result.fun[:-1]  # drop the Feller penalty row for error reporting
    rmse = float(np.sqrt(np.mean(pricing_resid ** 2)))
    max_err = float(np.max(np.abs(pricing_resid)))

    return HestonCalibrationResult(
        params=fitted, rmse_vol_pts=rmse, max_abs_error_vol_pts=max_err,
        n_points=len(points), n_iterations=result.nfev, success=result.success,
        message=result.message, feller_ratio=fitted.feller_ratio(),
    )
