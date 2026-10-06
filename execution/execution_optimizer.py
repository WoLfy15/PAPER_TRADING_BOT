"""
execution/execution_optimizer.py -- requirement 17: "use the RL
execution project only as an execution optimizer -- deciding when/how
to place and hedge trades -- not as the directional volatility signal,
and validate it against VWAP/TWAP and actual options execution costs
before deployment."

WHAT'S REAL AND ACTIVE HERE: TWAP and VWAP child-order slicing, using
the same pure, environment-agnostic trajectory math the RL execution
project's own baselines.py uses (twap_shares/vwap_shares -- reproduced
here rather than imported, since that project's copy lives inside an
ExecutionSimulator-coupled module tree not meant to be imported as a
library). Given a multi-contract entry, this produces a schedule of
smaller child submissions over the next few minutes instead of one lump
order -- a real, honest execution-quality improvement, validated below
against a "dump it all at once" baseline using this project's OWN real
options cost model (accounting/ledger.py's modeled bid/ask spread).

WHAT'S NOT WIRED UP, AND WHY, STATED PLAINLY: the trained PPO
checkpoints (rl_execution_extension/proj/models/ppo_*.npz) are real,
but they were trained against execution_env.py's OWN simulated
limit-order-book observation space -- order-book imbalance, inventory
fraction, time-remaining, a VPIN-style toxicity read, all specific to
that project's synthetic Hawkes-process equity LOB. There is no
established, validated mapping from "how many SPY option contracts are
left to fill and how wide the quoted spread is right now" into that
exact trained feature space. Building one badly would be worse than not
building one at all: a policy fed features it was never trained to
interpret doesn't fail loudly, it just produces confident, wrong
actions. RLExecutionAdapter below is a real, correct INTERFACE (so the
seam exists and is documented) that raises NotImplementedError with a
specific explanation rather than silently doing something ungrounded --
and it is OFF by default (config.Settings.enable_execution_optimizer)
regardless. Making this real is future work, not a checkbox to fake:
it needs someone to either retrain against an options-specific
execution environment, or carefully hand-build and BACKTEST a feature
mapping before it ever sees a live order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def twap_shares(order_size: float, n_steps: int) -> np.ndarray:
    return np.full(n_steps, order_size / n_steps)


def vwap_shares(order_size: float, n_steps: int) -> np.ndarray:
    """Stylized U-shaped intraday volume profile (heavier at the start
    and end of the slicing window) -- the standard stand-in used
    whenever a real historical intraday volume curve isn't available,
    same convention as the RL project's own baselines.py."""
    x = np.linspace(0, 1, n_steps)
    profile = 1.0 + 1.6 * (2 * x - 1) ** 2
    profile = profile / profile.sum()
    return order_size * profile


@dataclass
class ChildOrderSchedule:
    strategy: str            # "TWAP" | "VWAP" | "lump"
    child_sizes: list[int]    # contracts per child submission, summing to the total
    n_children: int


def build_child_schedule(total_contracts: int, strategy: str = "TWAP", n_children: Optional[int] = None) -> ChildOrderSchedule:
    """Splits a total contract count into child order sizes. n_children
    defaults to min(total_contracts, 4) -- enough to meaningfully spread
    execution without turning a 3-contract entry into three 1-contract
    orders that each pay their own worst-case spread crossing for no
    real benefit. strategy='lump' returns the whole size as one child --
    the baseline this module's own validation compares against."""
    if total_contracts <= 0:
        return ChildOrderSchedule(strategy=strategy, child_sizes=[], n_children=0)
    if strategy == "lump" or total_contracts == 1:
        return ChildOrderSchedule(strategy="lump", child_sizes=[total_contracts], n_children=1)
    n = n_children or min(total_contracts, 4)
    raw = twap_shares(total_contracts, n) if strategy == "TWAP" else vwap_shares(total_contracts, n)
    # Round to whole contracts, giving any rounding remainder to the
    # LAST child so the total always matches total_contracts exactly.
    sizes = np.floor(raw).astype(int)
    sizes[-1] += total_contracts - sizes.sum()
    sizes = [s for s in sizes if s > 0]
    return ChildOrderSchedule(strategy=strategy, child_sizes=sizes, n_children=len(sizes))


class ExecutionOptimizer(Protocol):
    """The seam requirement 17 asks for: something that decides HOW to
    place a trade this project has already decided (direction + size)
    to make -- never WHETHER or WHICH direction. Every implementation
    takes a total contract count and returns a schedule; nothing here
    ever sees or influences signal.composite_edge."""

    def schedule(self, total_contracts: int) -> ChildOrderSchedule:
        ...


@dataclass
class DefaultExecutionOptimizer:
    """Always available, on by default when
    Settings.enable_execution_optimizer is True. TWAP by default --
    VWAP's U-shaped profile is calibrated to a full trading DAY's volume
    curve, which doesn't map cleanly onto slicing a handful of child
    orders over a few minutes (this project's actual order-placement
    cadence); TWAP's flat profile is the honest choice at this
    timescale. strategy='VWAP' is still available and validated below
    for a deployment that widens the slicing window enough for the
    intraday shape to matter."""
    strategy: str = "TWAP"
    n_children: Optional[int] = None

    def schedule(self, total_contracts: int) -> ChildOrderSchedule:
        return build_child_schedule(total_contracts, strategy=self.strategy, n_children=self.n_children)


class RLExecutionAdapter:
    """Real interface, honestly not wired to a live decision path -- see
    module docstring. checkpoint_path is accepted (not ignored) so a
    future implementation has an obvious place to load a real, validated
    checkpoint once one exists for this exact task."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def schedule(self, total_contracts: int) -> ChildOrderSchedule:
        raise NotImplementedError(
            "RLExecutionAdapter has no validated observation-space mapping from options "
            "order state to the trained PPO checkpoint's features -- see this module's "
            "docstring. Use DefaultExecutionOptimizer, or implement and BACKTEST a real "
            "feature mapping before wiring this in."
        )


@dataclass
class ValidationResult:
    strategy: str
    cost_bps_of_notional: float
    n_children: int


def validate_against_lump(
    total_contracts: int, spot: float, strike: float,
    base_half_spread_pct: float = 0.015, impact_bps_per_sqrt_contract: float = 8.0,
) -> list[ValidationResult]:
    """Requirement 17's own "validate it against VWAP/TWAP ... before
    deployment" -- run here using THIS project's real options cost model
    (accounting.ledger.modeled_half_spread_pct) PLUS a square-root
    market-impact term, not the RL project's equity-LOB simulator, since
    the question is specifically "does slicing help for an OPTIONS
    order," which the equity simulator was never built to answer.

    The flat bid/ask spread model alone (as used elsewhere in this
    project's ledger) costs EXACTLY the same total whether one order or
    several smaller ones fill the same total size -- spread cost is
    linear in quantity by construction, so it cannot distinguish
    slicing strategies at all (confirmed empirically before this impact
    term was added: an earlier version of this function using only the
    flat spread returned identical bps-of-notional costs for lump/TWAP/
    VWAP -- a correct property of that model, and a USELESS comparison
    for this specific question). What actually makes slicing help in
    real markets is nonlinear PRICE IMPACT -- a single large order moves
    the quote against itself more than proportionally, which is why
    breaking it up can reduce total cost. `impact_bps_per_sqrt_contract`
    adds the standard, well-documented "impact scales with sqrt(order
    size)" convention (Almgren-Chriss and the broader market-
    microstructure literature) ON TOP OF the flat spread, applied PER
    CHILD ORDER -- so a lump child pays impact on its full size once,
    while several smaller children each pay a smaller per-child impact,
    with sqrt's concavity meaning the sum across children is less than
    one sqrt of the full size. This is a documented, standard
    convention, not a fitted parameter -- 8.0 bps per sqrt-contract is a
    round, illustrative starting point; treat this validation's output
    as directional evidence (does slicing help at all, and roughly how
    much) rather than a precise cost forecast.

    Reported as a single deterministic cost-in-bps-of-notional per
    strategy, not a distribution across simulated trials: a common
    multiplicative quote move between the time a decision is made and
    each child fills would affect the RAW DOLLAR cost of a schedule, but
    cancels out exactly in a bps-of-notional RATIO (both the cost and
    the notional it's measured against scale by the same factor) -- so
    dressing this up with a trial loop and an RNG would have shown a
    distribution with zero actual spread, which is worse than just
    saying so. What genuinely varies here is the SCHEDULE (lump vs. TWAP
    vs. VWAP), not simulated noise around a fixed schedule."""
    from accounting.ledger import modeled_half_spread_pct

    moneyness = (strike - spot) / spot if spot > 0 else 0.0
    half_spread_pct = modeled_half_spread_pct(moneyness, base_half_spread_pct)
    results = []
    for strategy in ("lump", "TWAP", "VWAP"):
        schedule = build_child_schedule(total_contracts, strategy=strategy)
        total_cost_pct_notional = 0.0
        total_qty = sum(schedule.child_sizes) or 1
        for child_qty in schedule.child_sizes:
            impact_pct = impact_bps_per_sqrt_contract * np.sqrt(max(child_qty, 1)) / 10_000.0
            # weight each child's (spread+impact) rate by its share of total quantity,
            # so the result is a genuine quantity-weighted average rate, not a naive sum.
            total_cost_pct_notional += (half_spread_pct + impact_pct) * (child_qty / total_qty)
        results.append(ValidationResult(
            strategy=strategy, cost_bps_of_notional=total_cost_pct_notional * 10_000.0, n_children=schedule.n_children,
        ))
    return results
