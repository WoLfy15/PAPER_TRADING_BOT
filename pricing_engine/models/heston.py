"""
pricing_engine/models/heston.py

Heston (1993) stochastic volatility:
    dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa(theta - v_t) dt + xi sqrt(v_t) dW_t^v,   d<W^S,W^v>_t = rho dt

Dupire local volatility (calibration/local_vol_surface.py) reprices today's
smile exactly but says nothing correct about how that smile will *evolve* --
its forward skew flattens unrealistically fast (a well-documented property,
not a bug in this project's implementation). Heston is a genuine second
model of the underlying's risk-neutral dynamics, not just a fit to today's
cross-section: it prices consistently through time because v_t is an actual
state variable, and it produces the term structure and forward-skew
behavior real markets show.

Two independent pricing paths are implemented and cross-checked against
each other, the same "never trust one number" discipline validation/
checks.py already applies elsewhere in this project:

  1. Semi-analytical: the characteristic function of ln(S_T) in the
     numerically stable "Little Trap" form (Albrecher, Mayer, Schoutens &
     Tistaert 2007 -- Heston's own 1993 formula has a branch-cut
     discontinuity in the complex log that this form avoids), priced via
     Carr-Madan (1999) damped Fourier inversion. Two implementations of
     the SAME integral are provided -- direct quadrature (`price_quad`,
     exact for one strike) and FFT (`price_fft_grid`, approximate but
     O(N log N) for an entire strike grid at once, which is what makes
     joint calibration to a full surface tractable).
  2. `simulate_paths`: an Euler full-truncation Monte Carlo simulation of
     the SDE directly. This shares no code and no mathematical machinery
     with the characteristic function above, so agreement between the two
     is real evidence the characteristic function is implemented
     correctly, not a tautology.

See tests/test_heston.py for the actual cross-validation: the xi->0 limit
against closed-form BSM, put-call parity, and quad-vs-FFT-vs-MC agreement.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from pricing_engine.market_data.contract import OptionQuote, OptionType
from pricing_engine.models.base import PricingModel, FiniteDifferenceGreeksMixin


@dataclass(frozen=True)
class HestonParams:
    """v0, theta: variance (not vol) units -- v0=0.04 means a 20% starting
    vol. Feller condition 2*kappa*theta > xi**2 keeps v_t from reaching
    zero under the continuous-time SDE; violating it doesn't break the
    pricing formula (the discretization below floors variance regardless)
    but does mean the model is relying on that floor rather than the
    SDE's own structure to stay non-negative -- calibration.py's
    Feller penalty keeps a fit away from that regime."""
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def feller_ratio(self) -> float:
        """2*kappa*theta / xi**2 -- > 1 satisfies the Feller condition."""
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


def characteristic_function(u: complex, T: float, S0: float, r: float, q: float, p: HestonParams) -> complex:
    """phi(u) = E[exp(iu * ln S_T)] under the risk-neutral measure, in the
    'Little Trap' stable form. Vectorized over u via numpy for the FFT
    path; also called with a plain complex scalar from price_quad."""
    kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta * beta + xi * xi * (1j * u + u * u))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    C = 1j * u * (r - q) * T + (kappa * theta / (xi * xi)) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
    )
    D = (beta - d) / (xi * xi) * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0 + 1j * u * math.log(S0))


def _carr_madan_integrand(u: np.ndarray, k: np.ndarray, T: float, S0: float, r: float, q: float,
                           p: HestonParams, alpha: float) -> np.ndarray:
    """Ψ(u), the damped Fourier transform of the call price in log-strike
    k -- shared by both the quadrature and FFT pricers so they are
    provably evaluating the same integrand, not just similar ones."""
    shifted = u - (alpha + 1) * 1j
    phi = characteristic_function(shifted, T, S0, r, q, p)
    denom = alpha * alpha + alpha - u * u + 1j * (2 * alpha + 1) * u
    return np.exp(-r * T) * phi / denom


def price_quad(quote: OptionQuote, p: HestonParams, alpha: float = 1.5, u_max: float = 200.0) -> float:
    """Exact (to quadrature tolerance) single-strike Carr-Madan price via
    scipy.integrate.quad -- the reference implementation; price_fft_grid
    trades a small, checkable amount of this accuracy for O(N log N)
    speed across a whole strike grid."""
    from scipy.integrate import quad

    S0, K, T, r, q = quote.spot, quote.strike, quote.expiry_years, quote.risk_free_rate, quote.dividend_yield
    k = math.log(K)

    def integrand(u: float) -> float:
        val = _carr_madan_integrand(np.array([u]), np.array([k]), T, S0, r, q, p, alpha)[0]
        return (np.exp(-1j * u * k) * val).real

    integral, _ = quad(integrand, 1e-10, u_max, limit=200)
    call = math.exp(-alpha * k) / math.pi * integral

    if quote.option_type == OptionType.CALL:
        return max(call, 1e-12)
    put = call - S0 * math.exp(-q * T) + K * math.exp(-r * T)  # put-call parity, not a second CF evaluation
    return max(put, 1e-12)


def price_fft_grid(S0: float, K_grid: np.ndarray, T: float, r: float, q: float, p: HestonParams,
                    alpha: float = 1.5, n: int = 4096, eta: float = 0.25) -> np.ndarray:
    """Carr-Madan FFT: call prices for every strike in K_grid at once in
    O(n log n), the speed that makes calibrating to a full surface
    (many strikes x many maturities, every Levenberg-Marquardt iteration)
    tractable. n, eta are the FFT grid size / spacing -- eta*lambda =
    2*pi/n where lambda is the log-strike spacing; defaults give a strike
    range wide enough for the moneyness this project's smiles use."""
    lam = 2 * math.pi / (n * eta)
    b = n * lam / 2
    u = np.arange(n) * eta
    k = -b + lam * np.arange(n)

    psi = _carr_madan_integrand(u, k, T, S0, r, q, p, alpha)
    delta0 = np.zeros(n)
    delta0[0] = 1.0
    simpson = (3.0 - (-1.0) ** np.arange(n) - delta0) / 3.0
    x = np.exp(1j * b * u) * psi * eta * simpson
    y = np.fft.fft(x).real
    call_prices = np.exp(-alpha * k) / math.pi * y

    return np.interp(np.log(K_grid), k, call_prices)


def simulate_paths(S0: float, r: float, q: float, T: float, n_steps: int, n_paths: int,
                    p: HestonParams, seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Euler full-truncation scheme (Lord, Koekkoek & Van Dijk 2010 -- the
    standard fix for the negative-variance problem plain Euler has on the
    CIR variance process): v is floored at 0 wherever the discretization
    would drive it negative, but the DRIFT still uses v itself rather than
    max(v,0), which keeps the scheme's bias smaller than the naive
    "reflect" or "absorb" fixes. Returns terminal spot prices, shape
    (n_paths,) -- used only to cross-validate the characteristic-function
    pricers above; not wired into the daily backtest (that stays on the
    local-vol MC engine, which real-market VIX data can anchor day to day
    in a way five free Heston parameters can't be re-fit for every day
    without overfitting the very history being backtested)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    n_sim = n_paths * 2 if antithetic else n_paths
    n_half = n_paths

    z1 = rng.standard_normal((n_half, n_steps))
    z_ind = rng.standard_normal((n_half, n_steps))
    z2 = p.rho * z1 + math.sqrt(max(1 - p.rho * p.rho, 0.0)) * z_ind
    if antithetic:
        z1 = np.concatenate([z1, -z1], axis=0)
        z2 = np.concatenate([z2, -z2], axis=0)

    v = np.full(n_sim, p.v0)
    x = np.full(n_sim, math.log(S0))
    sqdt = math.sqrt(dt)
    for step in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        x += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqdt * z1[:, step]
        v += p.kappa * (p.theta - v_pos) * dt + p.xi * sqrt_v * sqdt * z2[:, step]
    return np.exp(x)


class HestonModel(PricingModel, FiniteDifferenceGreeksMixin):
    """Wraps a fixed calibrated HestonParams as a PricingModel so it can be
    dropped into validation/checks.py's run_bsm_suite and the rest of the
    engine's polymorphic model handling unchanged. `sigma` in price()/
    greeks() is interpreted as sqrt(v0) -- i.e. it lets a caller shock the
    starting variance level (the direct Heston analogue of shocking sigma
    for BSM) while kappa/theta/xi/rho stay at their calibrated values;
    the model's own base v0 is used whenever sigma is None."""
    name = "Heston"

    def __init__(self, params: HestonParams):
        self.params = params

    def _params_for(self, sigma: float | None) -> HestonParams:
        if sigma is None:
            return self.params
        return HestonParams(v0=sigma * sigma, kappa=self.params.kappa, theta=self.params.theta,
                             xi=self.params.xi, rho=self.params.rho)

    def price(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return price_quad(quote, self._params_for(sigma))

    def delta(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return self.fd_delta(quote, self._resolve_sigma(sigma))

    def gamma(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return self.fd_gamma(quote, self._resolve_sigma(sigma))

    def vega(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return self.fd_vega(quote, self._resolve_sigma(sigma))

    def theta(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return self.fd_theta(quote, self._resolve_sigma(sigma))

    def rho(self, quote: OptionQuote, sigma: float | None = None) -> float:
        return self.fd_rho(quote, self._resolve_sigma(sigma))

    def _resolve_sigma(self, sigma: float | None) -> float:
        return sigma if sigma is not None else math.sqrt(self.params.v0)
