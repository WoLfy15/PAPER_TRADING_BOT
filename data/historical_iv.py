"""
data/historical_iv.py -- requirement 2: "replace the historical
realized_vol * 1.15 implied-volatility proxy with a real historical
SPY option IV surface/constant-maturity IV dataset."

What this actually ships, stated plainly rather than oversold: real,
independent, CBOE-published VIX daily closes from 1990-01-02 through
2026-08-13 (data/reference/vix_daily.csv -- the same real dataset
already used, unmodified, in the rough_vol_kaggle project's own
real-data validation). VIX *is* a constant-maturity IV dataset in the
literal sense CBOE defines it -- "the 30-day expected volatility of the
S&P 500 index, derived from SPX option prices" -- so for SPY (which
tracks SPX at ~1/10th scale; the two have essentially the same
annualized implied vol in practice) this is a genuine, independently-
sourced, market-observed IV series, not a multiplier applied to the
same realized-vol series already being forecast. That is a real fix
for the specific problem research/signal.py's docstring named: VRP was
being measured as (forecast of realized vol) minus (a CONSTANT times
the SAME realized vol) two numbers that are algebraically related by
construction, not an independent read of what the market was pricing.

What this does NOT ship, also stated plainly: genuine historical
per-strike skew or a genuine multi-tenor term structure (VIX is one
number at one ~30-day point; this dataset has no VIX9D/VIX3M/VIX6M
series alongside it). Rather than invent an unvalidated term-structure
slope formula to make this look more granular than the data actually
supports, the documented convention here is: use the VIX level as-is,
flat across DTE. That has real, honestly-stated fewer degrees of
freedom than a paid vendor's true surface -- `IVSurfaceProvider` (below)
is the seam a real OptionMetrics/ORATS/CBOE DataShop feed plugs into
later without any caller of `historical_iv_proxy` needing to change.

Scope: VIX is an S&P-500-level read. It is used here ONLY for SPY (this
project's default/primary symbol, and the one SPX-vol is actually a
good proxy for). Any other symbol (the WATCHLIST names -- NVDA, AAPL,
etc.) falls back to the old realized_vol * proxy_multiplier convention,
clearly labeled as such in the returned HistoricalIVSeries.source field
-- a broad index vol number is not a substitute for a single stock's
own, often earnings-driven, implied vol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

_VIX_CSV_PATH = Path(__file__).resolve().parent / "reference" / "vix_daily.csv"
_INDEX_PROXY_SYMBOLS = frozenset({"SPY"})  # symbols VIX is a reasonable stand-in for
LEGACY_PROXY_MULTIPLIER = 1.15  # v1/v2's realized_vol * 1.15 -- kept ONLY as the non-index fallback


class IVSurfaceProvider(Protocol):
    """Extension seam for a real paid vendor (OptionMetrics IvyDB, ORATS,
    CBOE DataShop). Anything implementing this can be swapped in for
    `_VixProvider` below without research/signal.py or the backtest
    changing at all -- both only ever call `historical_iv_proxy`."""

    def atm_iv_series(self, symbol: str, dates: pd.DatetimeIndex, target_dte_days: int) -> pd.Series:
        ...


@dataclass
class HistoricalIVSeries:
    values: pd.Series          # annualized IV fraction (0.15 = 15%), indexed like `dates`
    source: str                 # "vix_spx_proxy" | "realized_vol_legacy_proxy"
    note: str


_vix_cache: Optional[pd.Series] = None


def _load_vix_raw() -> pd.Series:
    global _vix_cache
    if _vix_cache is not None:
        return _vix_cache
    if not _VIX_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected bundled VIX history at {_VIX_CSV_PATH} -- if this file was removed, "
            "either restore data/reference/vix_daily.csv or point IVSurfaceProvider at a "
            "real vendor feed."
        )
    df = pd.read_csv(_VIX_CSV_PATH, parse_dates=["DATE"])
    series = df.set_index("DATE")["CLOSE"].astype(float) / 100.0  # VIX points -> annualized fraction
    series = series.sort_index()
    series.name = "vix"
    _vix_cache = series
    return series


def historical_iv_proxy(
    symbol: str, dates: pd.DatetimeIndex, realized_vol: pd.Series, target_dte_days: int = 30,
    surface_provider: Optional[IVSurfaceProvider] = None,
) -> HistoricalIVSeries:
    """The direct drop-in replacement for `realized_vol * 1.15`. `dates`
    and `realized_vol` should be the same aligned index/series callers
    already build (research/signal.py, backtest/engine.py). Forward-
    fills across non-trading-day gaps and, at the series' own start (a
    backtest window beginning before 1990 -- not reachable given the
    data, but handled anyway) uses the legacy proxy for whatever leading
    rows have no VIX print yet, so callers never see NaN they didn't
    already have to handle for `realized_vol` itself.

    A caller-supplied `surface_provider` (a real paid vendor's
    IVSurfaceProvider) takes priority over both the VIX proxy and the
    legacy fallback when given -- this is the upgrade path referenced
    throughout this module's docstring."""
    if surface_provider is not None:
        values = surface_provider.atm_iv_series(symbol, dates, target_dte_days)
        return HistoricalIVSeries(values=values, source="vendor_surface", note="Real vendor IV surface.")

    # Normalize to timezone-naive calendar dates before anything else.
    # Alpaca's own bar timestamps come back RFC-3339 with a trailing "Z"
    # (UTC) -- pandas parses that into a genuinely tz-AWARE index, while
    # the bundled reference CSVs (plain calendar dates, no time-of-day or
    # zone information at all) parse tz-NAIVE. Comparing/unioning the two
    # directly raises (confirmed against a realistic end-to-end run, not
    # just this docstring's say-so) -- what matters here is the calendar
    # DATE, not a specific instant, so this strips tz info rather than
    # trying to convert both sides to match some particular zone.
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    if realized_vol.index.tz is not None:
        realized_vol = realized_vol.copy()
        realized_vol.index = realized_vol.index.tz_localize(None)

    if symbol.upper() not in _INDEX_PROXY_SYMBOLS:
        legacy = realized_vol.reindex(dates) * LEGACY_PROXY_MULTIPLIER
        return HistoricalIVSeries(
            values=legacy, source="realized_vol_legacy_proxy",
            note=(
                f"{symbol} has no bundled index-vol proxy (VIX only stands in for SPY); "
                f"using the v1/v2 realized_vol*{LEGACY_PROXY_MULTIPLIER} approximation for "
                "this symbol only. Plug in a real per-symbol IV surface via surface_provider "
                "for a genuine fix on names outside SPY."
            ),
        )

    vix = _load_vix_raw()
    aligned = vix.reindex(dates.union(vix.index)).sort_index().ffill().reindex(dates)
    n_missing = int(aligned.isna().sum())
    if n_missing:
        # Only reachable for dates before VIX's own 1990-01-02 start.
        legacy_fill = realized_vol.reindex(dates) * LEGACY_PROXY_MULTIPLIER
        aligned = aligned.fillna(legacy_fill)
    return HistoricalIVSeries(
        values=aligned, source="vix_spx_proxy",
        note=(
            "Real CBOE VIX (SPX 30-day constant-maturity implied vol), used as SPY's "
            f"historical IV proxy -- independent of the realized-vol series being forecast. "
            f"{n_missing} of {len(dates)} rows predate VIX's 1990-01-02 start and used the "
            "legacy realized-vol proxy instead. Flat across DTE (no historical term "
            f"structure/skew in this dataset); target_dte_days={target_dte_days} was not "
            "used to adjust the level -- see module docstring."
        ),
    )


def vix_level_asof(as_of) -> Optional[float]:
    """Most recent real VIX close on or before `as_of` (a date-like).
    Used by the dashboard's macro-context panel and by
    backtest/engine.py's stress-scenario IV shocks (requirement 19),
    which shock relative to the REGIME the backtest was actually in,
    not an arbitrary fixed IV level."""
    vix = _load_vix_raw()
    asof_ts = pd.Timestamp(as_of)
    eligible = vix.loc[:asof_ts]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])
