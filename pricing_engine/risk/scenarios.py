"""
pricing_engine/risk/scenarios.py -- requirements 11 and 19:

  11. "redesign the iron-condor construction around expected credit
      versus max loss/CVaR and scenario stress, rather than choosing
      wings only from delta"
  19. "...stress scenarios such as SPY -1%, -2%, -3%, -5% combined with
      IV shocks..."

The named SPY -5%/-3%/-2%/-1%/+1%/+2%/+3%/+5% scenarios are the
literal, explicit ask -- built here (`named_stress_table`). But eight
fixed points is a thin basis for an actual CVaR figure to size or gate
on (a "95th percentile of 8 numbers" is really "the single worst one, a
lot of the time"), so `monte_carlo_cvar` backs the real sizing/limit
decision with something better: real historical (SPY return, VIX point
change) PAIRS, bootstrapped with replacement into a few thousand
resamples, then each resample reprices the structure. This is
deliberately NOT a parametric assumption (no assumed correlation
coefficient, no assumed shock-per-percent formula) -- it's the actual
joint distribution the S&P 500 and VIX have shown, over the lookback
window, reused directly. The same fitted relationship (a slope from
regressing daily VIX POINT changes on daily SPY returns -- the
well-documented "leverage effect": down days come with disproportionately
larger vol increases than up days of the same size, e.g. SPY's ~-3%
day in early August 2024 saw VIX roughly double) is what fills in each
named scenario's IV shock too, so the two tables tell a consistent
story instead of one being data-driven and the other invented.

What this is NOT: a claim that history will repeat with this exact
joint distribution, or a substitute for a real historical options
order-book-level backtest. It's a real, non-fabricated approximation --
strictly better than assuming spot and IV move independently (they
demonstrably don't) or than picking an unvalidated fixed shock number.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.market_data import ChainContract, to_option_quote
from pricing_engine.market_data.contract import OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()
_SP500_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "sp500_fred.csv"
_VIX_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "vix_daily.csv"

NAMED_SPOT_SHOCKS_PCT = (-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05)

StructureLeg = tuple[ChainContract, float]  # (contract, signed quantity -- +1 long, -1 short, per contract)


@dataclass
class ScenarioResult:
    label: str
    spot_shock_pct: float
    iv_shock_pts: float
    pnl_dollars: float


@dataclass
class StressReport:
    named_scenarios: list[ScenarioResult]
    worst_named_loss: float
    monte_carlo_cvar_dollars: float   # expected loss in the worst (1-alpha) tail, bootstrapped from real history
    monte_carlo_alpha: float
    n_bootstrap_samples: int
    leverage_slope_vix_pts_per_pct: float   # the fitted real relationship both tables use
    notes: list[str]


_joint_cache: Optional[pd.DataFrame] = None


def _load_joint_history() -> pd.DataFrame:
    """Real, aligned daily (spy_return, vix_change_pts) pairs from the
    two bundled CSVs (same files data/historical_iv.py already uses for
    the historical IV proxy). Cached at module level -- both files
    together are under 1MB and read once per process, not once per
    structure evaluated."""
    global _joint_cache
    if _joint_cache is not None:
        return _joint_cache
    sp = pd.read_csv(_SP500_CSV_PATH, parse_dates=["observation_date"]).set_index("observation_date")["SP500"].dropna()
    vix = pd.read_csv(_VIX_CSV_PATH, parse_dates=["DATE"]).set_index("DATE")["CLOSE"].dropna()
    spy_return = sp.pct_change()
    vix_change = vix.diff()
    joint = pd.DataFrame({"spy_return": spy_return, "vix_change": vix_change}).dropna()
    _joint_cache = joint
    return joint


def fit_leverage_slope(lookback_years: Optional[float] = 10.0) -> float:
    """OLS slope of daily VIX point-change on daily SPY return, over the
    trailing `lookback_years` of the bundled joint history (None = use
    everything available). A real, computed number -- not asserted --
    typically comes out solidly negative (down days -> VIX up), which is
    exactly the leverage effect this module exists to capture rather
    than assume."""
    joint = _load_joint_history()
    if lookback_years is not None:
        cutoff = joint.index.max() - pd.Timedelta(days=int(lookback_years * 365.25))
        joint = joint.loc[joint.index >= cutoff]
    if len(joint) < 30:
        return -50.0  # degenerate-history fallback; a very rough, clearly-labeled-as-such leverage estimate
    x = joint["spy_return"].values
    y = joint["vix_change"].values
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _reprice_structure(
    legs: list[StructureLeg], spot_now: float, risk_free_rate: float,
    spot_shock_pct: float, iv_shock_pts: float,
) -> float:
    """Dollar P&L of the WHOLE structure if spot moves spot_shock_pct and
    every leg's IV shifts by iv_shock_pts (in vol POINTS, e.g. 0.05 = 5
    points), relative to current mid-market value. Re-solves each leg's
    OWN current IV from its live mid (data/market_data.implied_vol_for
    convention, matching how strategy/legs.py already prices these) so
    the shock is applied on top of that contract's real starting point,
    not a single flat assumed IV for the whole structure."""
    from data.market_data import implied_vol_for

def _reprice_structure(
    legs: list[StructureLeg], spot_now: float, risk_free_rate: float,
    spot_shock_pct: float, iv_shock_pts: float, as_of: Optional[dt.date] = None,
) -> float:
    """Dollar P&L of the WHOLE structure if spot moves spot_shock_pct and
    every leg's IV shifts by iv_shock_pts (in vol POINTS, e.g. 0.05 = 5
    points), relative to current mid-market value. Re-solves each leg's
    OWN current IV from its live mid (data/market_data.implied_vol_for
    convention, matching how strategy/legs.py already prices these) so
    the shock is applied on top of that contract's real starting point,
    not a single flat assumed IV for the whole structure.

    as_of: forwarded to to_option_quote -- see that function's own
    docstring for why this matters outside a live "today is really
    today" call (backtest/engine.py passes its own walk-forward date)."""
    from data.market_data import implied_vol_for

    shocked_spot = spot_now * (1.0 + spot_shock_pct)
    pnl = 0.0
    for contract, signed_qty in legs:
        quote_now = to_option_quote(contract, spot_now, risk_free_rate, as_of=as_of)
        iv_now = implied_vol_for(quote_now)
        if iv_now is None or contract.mid is None:
            continue
        shocked_iv = max(iv_now + iv_shock_pts, 0.01)
        quote_shocked = to_option_quote(contract, shocked_spot, risk_free_rate, as_of=as_of)
        shocked_price = _bsm.price(quote_shocked, shocked_iv)
        pnl += signed_qty * 100 * (shocked_price - contract.mid)
    return float(pnl)


def named_stress_table(
    legs: list[StructureLeg], spot_now: float, risk_free_rate: float, leverage_slope: Optional[float] = None,
    as_of: Optional[dt.date] = None,
) -> list[ScenarioResult]:
    """The literal requirement-19 ask: SPY +/-1/2/3/5%, each paired with
    an IV shock from the REAL fitted leverage slope (fit_leverage_slope)
    rather than an assumed number -- e.g. a slope of -60 VIX-points-per-
    100%-move means a -5% day pairs with a +3.0 point IV shock
    (-60 * -0.05 = +3.0), a +5% day with a -3.0 point one."""
    slope = leverage_slope if leverage_slope is not None else fit_leverage_slope()
    out = []
    for shock_pct in NAMED_SPOT_SHOCKS_PCT:
        iv_shock_pts = (slope * shock_pct) / 100.0  # slope is VIX POINTS (e.g. 15.0 = 15 vol pts) per unit return
        pnl = _reprice_structure(legs, spot_now, risk_free_rate, shock_pct, iv_shock_pts, as_of=as_of)
        out.append(ScenarioResult(
            label=f"SPY {shock_pct:+.0%}", spot_shock_pct=shock_pct, iv_shock_pts=iv_shock_pts, pnl_dollars=pnl,
        ))
    return out


def monte_carlo_cvar(
    legs: list[StructureLeg], spot_now: float, risk_free_rate: float,
    alpha: float = 0.95, n_samples: int = 3000, lookback_years: float = 10.0, seed: int = 11,
    as_of: Optional[dt.date] = None,
) -> tuple[float, int]:
    """Bootstraps n_samples real historical (spy_return, vix_change) PAIRS
    (with replacement, from the trailing lookback_years) -- not a
    parametric draw -- reprices the structure under each, and returns
    (cvar_dollars, n_samples_used). cvar_dollars is the average P&L
    among the worst (1-alpha) fraction of resamples -- a real,
    non-fabricated worst-tail figure, since every individual resample IS
    a real day that actually happened, just possibly re-used or drawn in
    a new order/count. Returns (0.0, 0) if there's not enough joint
    history to bootstrap meaningfully (should not happen given the
    bundled data's real range, but keeps this safe against a stripped-down
    data file)."""
    joint = _load_joint_history()
    cutoff = joint.index.max() - pd.Timedelta(days=int(lookback_years * 365.25))
    windowed = joint.loc[joint.index >= cutoff]
    if len(windowed) < 30:
        return 0.0, 0
    rng = np.random.default_rng(seed)
    draws = windowed.sample(n=n_samples, replace=True, random_state=rng.integers(0, 2**31 - 1))
    pnls = np.array([
        _reprice_structure(legs, spot_now, risk_free_rate, r.spy_return, r.vix_change / 100.0, as_of=as_of)
        for r in draws.itertuples()
    ])
    tail_count = max(int(np.ceil((1 - alpha) * len(pnls))), 1)
    worst = np.sort(pnls)[:tail_count]
    return float(np.mean(worst)), len(pnls)


def build_stress_report(
    legs: list[StructureLeg], spot_now: float, risk_free_rate: float, alpha: float = 0.95, n_samples: int = 3000,
    as_of: Optional[dt.date] = None,
) -> StressReport:
    """The single call site strategy/legs.py and backtest/engine.py both
    use -- everything above this function is building blocks; this is
    the one result object the rest of the codebase actually holds onto.

    as_of: the date every BSM reprice in this report is measured against
    -- None (the default) means "really today" (dt.date.today(), via
    to_option_quote's own default), correct for a live cycle. A backtest
    walking through historical dates MUST pass its own current
    walk-forward date here, or every leg's time-to-expiry is measured
    against the real system clock instead of the simulated one -- see
    to_option_quote's docstring for the concrete failure this caused
    before this parameter existed."""
    notes: list[str] = []
    slope = fit_leverage_slope()
    if slope > 0:
        notes.append(
            "Fitted leverage slope came out POSITIVE (VIX rising WITH SPY) over the configured lookback -- "
            "unusual; falling back to a conservative negative assumption for the shock tables rather than "
            "shocking IV in the empirically-backwards direction."
        )
        slope = -50.0
    named = named_stress_table(legs, spot_now, risk_free_rate, leverage_slope=slope, as_of=as_of)
    worst_named = min((s.pnl_dollars for s in named), default=0.0)
    cvar, n_used = monte_carlo_cvar(legs, spot_now, risk_free_rate, alpha=alpha, n_samples=n_samples, as_of=as_of)
    if n_used == 0:
        notes.append("Bootstrap CVaR unavailable (insufficient bundled joint history) -- using named-scenario worst-case only.")
    return StressReport(
        named_scenarios=named, worst_named_loss=worst_named, monte_carlo_cvar_dollars=cvar,
        monte_carlo_alpha=alpha, n_bootstrap_samples=n_used, leverage_slope_vix_pts_per_pct=slope, notes=notes,
    )
