"""US-market clock helpers. The agent runs on a machine that may be in another timezone (e.g. India), where
`date.today()` is a day AHEAD of New York for the last hours of the US session -- which skewed DTE maths
(early 'near expiry' exits). Everything date-sensitive uses these instead."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


_CLOCK: "dt.datetime | None" = None   # set by the backtester so live pacing/cooldown code sees simulated time


def set_clock(now: "dt.datetime | None") -> None:
    global _CLOCK
    _CLOCK = None if now is None else (now.astimezone(NY) if now.tzinfo else now.replace(tzinfo=NY))


def us_now() -> dt.datetime:
    return _CLOCK if _CLOCK is not None else dt.datetime.now(NY)


def us_today() -> dt.date:
    return us_now().date()


def in_entry_window(avoid_minutes: int) -> bool:
    """True on weekdays between (open + N min) and (close - N min) ET."""
    n = us_now()
    m = n.hour * 60 + n.minute
    return n.weekday() < 5 and 570 + avoid_minutes <= m < 960 - avoid_minutes
