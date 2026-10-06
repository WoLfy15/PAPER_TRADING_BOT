"""
research/event_filter.py -- requirement 8: "add explicit regime/event
filters for FOMC, CPI, NFP and large volatility/jump regimes, reducing
or blocking short-vol exposure around high-risk events."

Two independent checks, both scoped to NEW SHORT-VOL ENTRIES only:

  1. CALENDAR: is today inside a blackout window around a known FOMC
     decision, CPI print, or NFP release (data/calendar_events.py)?
     Two tiers, matching the requirement's own "reducing OR blocking"
     language rather than a single on/off switch: a tight HARD-BLOCK
     window (config.Settings.*_blackout_days_before/after, defaults:
     block the FOMC decision day itself and CPI/NFP print days) and a
     wider ELEVATED window one extra trading day out, which throttles
     size rather than refusing outright.

  2. JUMP: did today's underlying move an unusual number of trailing-
     vol-units, independent of any calendar? This is deliberately a
     SEPARATE, faster-reacting signal from research/regime.py's HMM
     state -- the HMM is fit on a rolling window and only reclassifies
     a regime once the recent-return distribution has actually shifted,
     which lags a single sharp print by design (that's what makes it a
     regime detector rather than a jump detector). A same-day z-score
     of today's log return against trailing realized vol catches the
     print itself, the same day it happens, which is exactly what
     "large volatility/jump regimes" as a distinct item from "regime"
     in the requirement text is asking for.

WHY new entries only, never exits: blocking risk-REDUCING trades around
a risk event is backwards -- if anything, the agent should find it
EASIER to flatten into a catalyst, not harder. agent/loop.py and
backtest/engine.py must never route an exit/close decision through this
gate; both are wired to call assess_event_risk() only immediately
before deciding whether to open a NEW short-vol structure.

Calendar dates carry a confidence flag from data/calendar_events.py --
a VERIFIED date gets the configured (tight) window; an ESTIMATED date
(no verified entry exists for that year/month -- see that module's
docstring on why CPI/NFP 2027 is unpopulated) gets a wider window,
since a less certain date is exactly the case where erring toward
caution costs the least: a few extra blocked cycles around a guessed
date versus trading straight through a real one that was one day off.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.calendar_events import (
    CPI_RELEASE_DATES_VERIFIED,
    FOMC_DECISION_DATES,
    NFP_RELEASE_DATES_VERIFIED,
    cpi_mid_month_estimate,
    first_friday_estimate,
)

# Estimated (unverified) dates get one extra day of blackout on each
# side versus the configured verified-date window -- see module docstring.
ESTIMATED_EXTRA_BLACKOUT_DAYS = 1

# Same-day jump gate (independent of the calendar). abs(today's log
# return) / trailing realized vol's IMPLIED daily sigma, i.e. how many
# "normal days" today's move represents. 3.0 is a standard, conservative
# "this is not an ordinary trading day" threshold (under a normal
# distribution, |z|>3 is a ~0.3% daily event); not fit to any particular
# symbol's history.
JUMP_Z_THRESHOLD = 3.0
JUMP_SIZE_MULTIPLIER = 0.35  # how hard a live jump throttles short-vol sizing
ELEVATED_SIZE_MULTIPLIER = 0.5  # elevated-window (near-miss) throttle


@dataclass
class EventRisk:
    short_vol_blocked: bool           # hard block -- decide_sizing should force short_vol to flat
    short_vol_size_multiplier: float  # 1.0 = no throttle; multiplies into extra_scale otherwise
    reasons: list[str] = field(default_factory=list)
    nearest_event: Optional[str] = None       # human label of the closest calendar event, for the UI
    nearest_event_date: Optional[str] = None  # ISO date
    nearest_event_confidence: Optional[str] = None  # "verified" | "estimated"
    jump_z: Optional[float] = None
    jump_flagged: bool = False


def _nearest(dates: tuple[dt.date, ...], as_of: dt.date) -> Optional[tuple[dt.date, int]]:
    if not dates:
        return None
    best = min(dates, key=lambda d: abs((d - as_of).days))
    return best, (best - as_of).days


def _window_hit(days_away: int, before: int, after: int) -> bool:
    """days_away = event_date - as_of (negative = event already passed
    today). Inside the block window when 0 <= days_away <= before (event
    is upcoming, within `before` days) or -after <= days_away < 0 (event
    just happened, within `after` days), or days_away == 0 (the day of)."""
    if days_away == 0:
        return True
    if days_away > 0:
        return days_away <= before
    return (-days_away) <= after


def _check_one_calendar(
    label: str, verified_dates: tuple[dt.date, ...], estimate_fn, as_of: dt.date,
    days_before: int, days_after: int,
) -> tuple[bool, float, Optional[str]]:
    """Returns (blocked, size_multiplier, note). Checks the verified
    list first; only falls back to the date-math estimate for the
    as_of month if verified has nothing within +/- ~45 days (no point
    estimating a month that's already covered)."""
    nearest = _nearest(verified_dates, as_of)
    confidence = "verified"
    if nearest is None or abs(nearest[1]) > 45:
        # No verified date anywhere near as_of's month -- estimate for
        # as_of's own month AND the next one (an event early next month
        # can still fall inside a wide pre-event window today).
        candidates = []
        for delta_months in (0, 1):
            y, m = as_of.year, as_of.month + delta_months
            if m > 12:
                y, m = y + 1, m - 12
            candidates.append(estimate_fn(y, m))
        nearest = _nearest(tuple(candidates), as_of)
        confidence = "estimated"
    if nearest is None:
        return False, 1.0, None

    event_date, days_away = nearest
    extra = ESTIMATED_EXTRA_BLACKOUT_DAYS if confidence == "estimated" else 0
    hard_before, hard_after = days_before + extra, days_after + extra
    elevated_before, elevated_after = hard_before + 1, hard_after + 1

    if _window_hit(days_away, hard_before, hard_after):
        return True, 0.0, (
            f"{label} on {event_date.isoformat()} ({confidence}) is within the "
            f"hard-block window (+{hard_before}/-{hard_after} days) -- new short-vol "
            f"entries refused."
        )
    if _window_hit(days_away, elevated_before, elevated_after):
        return False, ELEVATED_SIZE_MULTIPLIER, (
            f"{label} on {event_date.isoformat()} ({confidence}) is within the elevated "
            f"window -- short-vol sizing throttled x{ELEVATED_SIZE_MULTIPLIER:.2f}."
        )
    return False, 1.0, None


def _jump_check(daily_log_returns: Optional[np.ndarray], lookback: int = 21) -> tuple[float | None, bool]:
    if daily_log_returns is None or len(daily_log_returns) < lookback + 2:
        return None, False
    today = float(daily_log_returns[-1])
    trailing = daily_log_returns[-(lookback + 1):-1]
    sigma = float(np.std(trailing))
    if sigma <= 1e-9:
        return None, False
    z = today / sigma
    return z, abs(z) >= JUMP_Z_THRESHOLD


def assess_event_risk(
    as_of: dt.date,
    fomc_blackout_days_before: int = 1,
    fomc_blackout_days_after: int = 0,
    cpi_nfp_blackout_days_before: int = 0,
    cpi_nfp_blackout_days_after: int = 0,
    daily_log_returns: Optional[np.ndarray] = None,
) -> EventRisk:
    """Called once per cycle, BEFORE sizing a new short-vol structure --
    never before evaluating an exit. `daily_log_returns` should be the
    same aligned daily log-return series research/vol_forecast.py and
    research/regime.py already build; passing None skips the jump check
    (calendar check still runs)."""
    reasons: list[str] = []
    blocked = False
    multiplier = 1.0
    nearest_label = nearest_date = nearest_conf = None

    for label, dates, estimate_fn, before, after in (
        ("FOMC", FOMC_DECISION_DATES, first_friday_estimate, fomc_blackout_days_before, fomc_blackout_days_after),
        ("CPI", CPI_RELEASE_DATES_VERIFIED, cpi_mid_month_estimate, cpi_nfp_blackout_days_before, cpi_nfp_blackout_days_after),
        ("NFP", NFP_RELEASE_DATES_VERIFIED, first_friday_estimate, cpi_nfp_blackout_days_before, cpi_nfp_blackout_days_after),
    ):
        b, mult, note = _check_one_calendar(label, dates, estimate_fn, as_of, before, after)
        if note:
            reasons.append(note)
            nearest_label = label
        blocked = blocked or b
        multiplier = min(multiplier, mult)

    jump_z, jump_flagged = _jump_check(daily_log_returns)
    if jump_flagged:
        multiplier = min(multiplier, JUMP_SIZE_MULTIPLIER)
        reasons.append(
            f"Same-day jump gate: today's move is {jump_z:+.1f} trailing-sigma units "
            f"(threshold {JUMP_Z_THRESHOLD:.1f}) -- short-vol sizing throttled "
            f"x{JUMP_SIZE_MULTIPLIER:.2f} independent of the calendar."
        )

    if nearest_label:
        nearest = _nearest(
            {"FOMC": FOMC_DECISION_DATES, "CPI": CPI_RELEASE_DATES_VERIFIED, "NFP": NFP_RELEASE_DATES_VERIFIED}[nearest_label],
            as_of,
        )
        if nearest:
            nearest_date = nearest[0].isoformat()
            nearest_conf = "verified"

    return EventRisk(
        short_vol_blocked=blocked, short_vol_size_multiplier=multiplier if not blocked else 0.0,
        reasons=reasons, nearest_event=nearest_label, nearest_event_date=nearest_date,
        nearest_event_confidence=nearest_conf, jump_z=jump_z, jump_flagged=jump_flagged,
    )


def upcoming_events(as_of: dt.date, horizon_days: int = 14) -> list[dict]:
    """All verified FOMC/CPI/NFP dates within horizon_days of as_of --
    for the UI's event-calendar banner, not used in sizing."""
    out = []
    for label, dates in (("FOMC", FOMC_DECISION_DATES), ("CPI", CPI_RELEASE_DATES_VERIFIED), ("NFP", NFP_RELEASE_DATES_VERIFIED)):
        for d in dates:
            days_away = (d - as_of).days
            if 0 <= days_away <= horizon_days:
                out.append({"label": label, "date": d.isoformat(), "days_away": days_away})
    return sorted(out, key=lambda e: e["days_away"])
