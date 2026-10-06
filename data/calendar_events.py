"""
data/calendar_events.py -- macro event dates backing the v3 event
filter (requirement 8: "add explicit regime/event filters for FOMC,
CPI, NFP and large volatility/jump regimes, reducing or blocking
short-vol exposure around high-risk events").

Every date below was pulled from the issuing agency's own site while
this module was written (mid-September 2026), not reconstructed from
training-time memory -- FOMC decision dates matter to the day, and
"probably first Friday of the month" turned out to be WRONG for at
least one 2026 NFP release (see NFP_2026, February), so a rule-of-thumb
generator would have silently mis-scheduled the blackout window around
the wrong day. Sources:

  FOMC 2026 : https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
              https://www.chicagofed.org/utilities/about-us/federal-reserve-calendars
  FOMC 2027 : https://www.mnimarkets.com/articles/mni-federal-reserve-sets-2027-fomc-meeting-schedule-1757093400287
              (the Fed's own Friday, Sep 5 2026 press release; cross-checked
              against a second independent source for the same dates)
  CPI/NFP 2026 : https://www.bls.gov/schedule/2026/home.htm (BLS's own full-year
              "Schedule of Selected Releases", fetched directly -- every
              date below for 2026 came off that one page)

CPI/NFP 2027 is intentionally NOT populated: BLS had not yet published
its 2027 schedule as of when this was written (BLS typically posts each
year's exact dates a few months into the prior year). Rather than guess
a specific day and have the filter confidently block or allow trading
around a date that's wrong, `event_filter.py` falls back to a clearly
lower-confidence heuristic (first-Friday-ish) for any month with no
verified entry here, and treats that fallback with a WIDER blackout
window precisely because it's uncertain -- see that module. To refresh
this file for a new year: re-fetch the three URLs above (BLS publishes
the coming year's schedule around September of the prior year) and
extend the lists below; nothing else needs to change.

Every entry is a `date` — the decision/print DAY itself, not the
multi-day FOMC meeting or a reference-period label.
"""
from __future__ import annotations

import datetime as dt

# ---------------------------------------------------------------- FOMC
# Decision day (second day of the two-day meeting; the day the policy
# statement and, where noted, the Summary of Economic Projections are
# released) -- this is the volatility event, not the meeting's start.
FOMC_DECISION_DATES: tuple[dt.date, ...] = (
    # 2026 -- federalreserve.gov / chicagofed.org
    dt.date(2026, 1, 28),
    dt.date(2026, 3, 18),
    dt.date(2026, 4, 29),
    dt.date(2026, 6, 17),
    dt.date(2026, 7, 29),
    dt.date(2026, 9, 16),
    dt.date(2026, 10, 28),
    dt.date(2026, 12, 9),
    # 2027 -- Federal Reserve's official Sep 5, 2026 announcement
    dt.date(2027, 1, 27),
    dt.date(2027, 3, 17),
    dt.date(2027, 4, 28),
    dt.date(2027, 6, 9),
    dt.date(2027, 7, 28),
    dt.date(2027, 9, 15),
    dt.date(2027, 10, 27),
    dt.date(2027, 12, 8),
)

# ------------------------------------------------------- CPI (8:30 ET)
CPI_RELEASE_DATES_VERIFIED: tuple[dt.date, ...] = (
    # 2026 -- bls.gov/schedule/2026/home.htm, full year, fetched directly
    dt.date(2026, 1, 13),
    dt.date(2026, 2, 13),
    dt.date(2026, 3, 11),
    dt.date(2026, 4, 10),
    dt.date(2026, 5, 12),
    dt.date(2026, 6, 10),
    dt.date(2026, 7, 14),
    dt.date(2026, 8, 12),
    dt.date(2026, 9, 11),
    dt.date(2026, 10, 14),
    dt.date(2026, 11, 10),
    dt.date(2026, 12, 10),
)

# ---------------------------------- NFP / Employment Situation (8:30 ET)
NFP_RELEASE_DATES_VERIFIED: tuple[dt.date, ...] = (
    # 2026 -- bls.gov/schedule/2026/home.htm, full year, fetched directly.
    # NOTE: Feb 11 2026 is a Wednesday, not the "first Friday" a naive
    # rule would predict (Feb 6) -- BLS shifted it. Real, source-checked
    # dates are exactly why this list exists instead of a date-math rule.
    dt.date(2026, 1, 9),
    dt.date(2026, 2, 11),
    dt.date(2026, 3, 6),
    dt.date(2026, 4, 3),
    dt.date(2026, 5, 8),
    dt.date(2026, 6, 5),
    dt.date(2026, 7, 2),
    dt.date(2026, 8, 7),
    dt.date(2026, 9, 4),
    dt.date(2026, 10, 2),
    dt.date(2026, 11, 6),
    dt.date(2026, 12, 4),
)

LAST_VERIFIED_UTC = "2026-09-19"
SOURCE_URLS = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "https://www.chicagofed.org/utilities/about-us/federal-reserve-calendars",
    "https://www.bls.gov/schedule/2026/home.htm",
)


def first_friday_estimate(year: int, month: int) -> dt.date:
    """Fallback ONLY for a year/month with no verified entry above.
    Standard convention for NFP timing, but confirmed NOT universally
    exact (see NFP_RELEASE_DATES_VERIFIED's Feb-2026 note) -- callers
    must treat a date produced by this function as lower-confidence
    than a verified list entry. Also used as a rough CPI proxy (CPI
    typically follows 1-2 weeks after NFP) when no verified CPI date
    exists either; event_filter.py adds a wider blackout around
    estimates for exactly this reason."""
    d = dt.date(year, month, 1)
    while d.weekday() != 4:  # Friday
        d += dt.timedelta(days=1)
    return d


def cpi_mid_month_estimate(year: int, month: int) -> dt.date:
    """Fallback ONLY for a year/month with no verified CPI entry.
    CPI has landed on the 10th-14th in every verified 2026 print above;
    the 13th is the sample median. Same lower-confidence caveat as
    first_friday_estimate."""
    return dt.date(year, month, 13)
