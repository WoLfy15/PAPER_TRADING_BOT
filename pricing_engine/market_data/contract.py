"""
Team A — Data & Infrastructure
Shared data contract (Phase 1/2). Every pricing model, calibration routine,
and risk module in this engine consumes and produces OptionQuote objects,
so this file is the one contract every other team codes against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionQuote:
    """One market observation of a single option contract.

    All rates/yields are continuously compounded, act/365, matching the
    tenor of `expiry_years`. This dataclass is intentionally immutable —
    use `.with_(...)` to derive a bumped copy (handy for finite-difference
    Greeks and scenario shocks in Team E's work).
    """

    underlying: str
    spot: float
    strike: float
    expiry_years: float
    option_type: OptionType
    risk_free_rate: float
    dividend_yield: float = 0.0
    market_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_vol: Optional[float] = None

    def __post_init__(self):
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.expiry_years < 0:
            raise ValueError("expiry_years must be non-negative")

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.market_price

    @property
    def log_moneyness(self) -> float:
        """ln(K / F) — the natural coordinate for smile fitting (Gatheral)."""
        return math.log(self.strike / self.forward)

    @property
    def forward(self) -> float:
        return self.spot * math.exp((self.risk_free_rate - self.dividend_yield) * self.expiry_years)

    def with_(self, **overrides) -> "OptionQuote":
        """Return a copy with the given fields overridden. Used to bump spot,
        vol, rate, or time for finite-difference Greeks and scenario shocks
        without ever mutating the original quote."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class MarketSmilePoint:
    """One (strike, maturity, implied vol) observation feeding calibration/
    local_vol_surface.py. Kept separate from OptionQuote because a smile
    fit consumes many quotes at once, cross-sectionally."""

    strike: float
    expiry_years: float
    implied_vol: float
