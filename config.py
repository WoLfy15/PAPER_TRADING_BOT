"""
config.py -- single source of truth for every tunable in the agent.

Everything is read from the environment (via a .env file in development)
so the same code runs unmodified in a notebook, a container, or behind
the FastAPI service. Nothing here ever hard-codes a key.

Design note: DRY_RUN defaults to True. The agent will compute signals,
size positions, and build order payloads whether or not DRY_RUN is set --
the only thing DRY_RUN gates is the final `alpaca_client.submit_order`
call. This means the UI can always show "what the agent would do" even
before the user is comfortable flipping it live, and it means a code
review of execution/order_manager.py is enough to convince yourself
nothing fires without the flag being explicitly turned off.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("alpaca_vol_agent.config")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so we don't force a python-dotenv dependency.
    Does nothing if the file isn't present; never overrides a variable
    already set in the real environment (real env wins, e.g. in Docker)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


DEFAULT_WATCHLIST = "NVDA,AAPL,TSLA,MSFT,AMZN,GOOGL,META,NFLX,AVGO,AMD,BRK.B,JPM,V,LLY,WMT,COST"


def _list(name: str, default: str) -> list[str]:
    val = os.getenv(name)
    raw = val if val not in (None, "") else default
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class RiskLimits:
    """Portfolio-overlay risk budget. All fractions are of current
    account equity, read live from Alpaca -- not a fixed notional --
    so the budget scales automatically as paper equity moves.

    v3 additions (see CHANGES_V3.md item 9/10/11/19): a real hard-limit
    set on top of the gamma/vega caps and single drawdown kill switch
    v2 already had -- per-trade/daily/weekly loss circuit breakers, a
    delta and theta cap alongside the existing gamma/vega ones, an
    explicit vol-TARGET BAND (8-12%) rather than one point plus a soft
    scalar, an asymmetric long-vol/short-vol edge threshold (the
    project's own research found the long-vol side materially weaker --
    see research/signal.py's ASYMMETRIC_EDGE_THRESHOLDS), and a CVaR
    budget used both to pick iron-condor wing width (strategy/legs.py)
    and as a portfolio-level cap (strategy/portfolio.py)."""

    max_gamma_risk_pct: float = field(default_factory=lambda: _float("MAX_GAMMA_RISK_PCT", 0.0005))
    max_vega_risk_pct: float = field(default_factory=lambda: _float("MAX_VEGA_RISK_PCT", 0.02))
    # v3: delta and theta join gamma/vega as HARD per-cycle caps (requirement
    # 10) -- previously only monitored (strategy/portfolio.py), never enforced
    # at sizing time. delta is expressed as a fraction of equity per $1 spot
    # move (i.e. dollar-delta / equity); theta as a fraction of equity of
    # daily time decay.
    max_delta_risk_pct: float = field(default_factory=lambda: _float("MAX_DELTA_RISK_PCT", 0.15))
    max_theta_risk_pct: float = field(default_factory=lambda: _float("MAX_THETA_RISK_PCT", 0.01))
    target_portfolio_vol: float = field(default_factory=lambda: _float("TARGET_PORTFOLIO_VOL", 0.10))
    # v3: the single point above is now the BAND CENTER; vol_target_scale
    # (strategy/portfolio.py) scales down only outside [min, max], so the
    # agent isn't fighting its own sizing inside a normal range -- see that
    # function's docstring. 0.08-0.12 is requirement 9's literal band.
    target_portfolio_vol_min: float = field(default_factory=lambda: _float("TARGET_PORTFOLIO_VOL_MIN", 0.08))
    target_portfolio_vol_max: float = field(default_factory=lambda: _float("TARGET_PORTFOLIO_VOL_MAX", 0.12))
    max_drawdown_pct: float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", 0.25))
    # v3: daily/weekly/per-trade circuit breakers, distinct from the
    # peak-to-trough drawdown check above -- a 6% day can be well inside a
    # 25% max-drawdown budget and still be exactly the kind of single event
    # this project's own research (see rl_execution_extension) flags as
    # where realized vol accelerates fastest. Checked in
    # strategy/portfolio.check_loss_circuit_breakers, using the daily/
    # weekly P&L rollups accounting/ledger.py now persists every cycle.
    max_daily_loss_pct: float = field(default_factory=lambda: _float("MAX_DAILY_LOSS_PCT", 0.04))
    max_weekly_loss_pct: float = field(default_factory=lambda: _float("MAX_WEEKLY_LOSS_PCT", 0.08))
    max_per_trade_loss_pct: float = field(default_factory=lambda: _float("MAX_PER_TRADE_LOSS_PCT", 0.02))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))
    kelly_clip: float = field(default_factory=lambda: _float("KELLY_CLIP", 0.15))
    # v3: smaller default sizing multipliers (requirement 9: "much smaller
    # sizing multipliers") -- kelly_fraction 0.25->0.20 and kelly_clip
    # 0.15->0.10 tighten every position this project sizes without touching
    # the Kelly MECHANISM itself (still risk.kelly_fraction * composite_edge,
    # clipped). Overridable via env exactly as before.
    long_vol_edge_threshold: float = field(default_factory=lambda: _float("LONG_VOL_EDGE_THRESHOLD", 0.35))
    # v3: asymmetric gate (requirement 5). composite_edge is in [-1, 1]
    # after the tanh squash; short_vol (composite_edge < 0, "sell rich
    # vol") is the primary strategy and fires at any nonzero signed Kelly
    # as before. long_vol (composite_edge > 0, "buy cheap vol") additionally
    # requires |composite_edge| to clear this threshold BEFORE a Kelly
    # fraction is even computed for that side -- see
    # strategy/sizing.kelly_fraction. 0.0 recovers the old symmetric v2
    # behavior exactly, for anyone who wants to A/B against this change.
    short_vol_edge_threshold: float = field(default_factory=lambda: _float("SHORT_VOL_EDGE_THRESHOLD", 0.0))
    max_contracts_per_leg: int = field(default_factory=lambda: _int("MAX_CONTRACTS_PER_LEG", 20))
    # v3: CVaR budget for structure selection (requirements 9/11/19).
    # cvar_alpha is the tail probability (0.95 = worst 5%); a structure
    # whose modeled CVaR at that tail exceeds max_structure_cvar_pct * equity
    # is rejected regardless of what its plain max-loss figure says -- see
    # risk/scenarios.py and strategy/legs.select_iron_condor_by_cvar.
    cvar_alpha: float = field(default_factory=lambda: _float("CVAR_ALPHA", 0.95))
    max_structure_cvar_pct: float = field(default_factory=lambda: _float("MAX_STRUCTURE_CVAR_PCT", 0.03))
    # -- auto-exit thresholds (v2) --
    exit_take_profit_pct: float = field(default_factory=lambda: _float("EXIT_TAKE_PROFIT_PCT", 0.50))
    exit_stop_loss_pct: float = field(default_factory=lambda: _float("EXIT_STOP_LOSS_PCT", 0.35))
    exit_dte_close_days: int = field(default_factory=lambda: _int("EXIT_DTE_CLOSE_DAYS", 2))
    # v4.2: exits judged per structure type. A condor's stop used to be the generic -35% of credit, which normal
    # noise trips constantly (its own docstring flagged this); 100% of credit = the standard "2x credit" stop.
    exit_condor_take_profit_pct: float = field(default_factory=lambda: _float("EXIT_CONDOR_TAKE_PROFIT_PCT", 0.50))
    exit_condor_stop_loss_pct: float = field(default_factory=lambda: _float("EXIT_CONDOR_STOP_LOSS_PCT", 1.00))
    exit_straddle_take_profit_pct: float = field(default_factory=lambda: _float("EXIT_STRADDLE_TAKE_PROFIT_PCT", 0.40))
    exit_straddle_stop_loss_pct: float = field(default_factory=lambda: _float("EXIT_STRADDLE_STOP_LOSS_PCT", 0.30))
    # trailing stop: once a structure has been up >= activate, exit if it gives back half of its best P&L
    exit_trail_activate_pct: float = field(default_factory=lambda: _float("EXIT_TRAIL_ACTIVATE_PCT", 0.25))
    exit_trail_keep_fraction: float = field(default_factory=lambda: _float("EXIT_TRAIL_KEEP_FRACTION", 0.50))


@dataclass(frozen=True)
class HedgeParams:
    """Feeds pricing_engine.risk.friction_hedging -- see that module's
    docstring for the Leland / Whalley-Wilmott derivations."""

    transaction_cost_bps: float = field(default_factory=lambda: _float("TRANSACTION_COST_BPS", 5.0))
    risk_aversion: float = field(default_factory=lambda: _float("HEDGE_RISK_AVERSION", 1.0))
    rehedge_min_interval_min: int = field(default_factory=lambda: _int("REHEDGE_MIN_INTERVAL_MIN", 30))
    min_variance_lookback_days: int = field(default_factory=lambda: _int("MIN_VARIANCE_LOOKBACK_DAYS", 20))


@dataclass(frozen=True)
class Settings:
    # -- Alpaca credentials -- names match the CLI's own env vars exactly
    # (github.com/alpacahq/cli): ALPACA_API_KEY / ALPACA_SECRET_KEY. The CLI
    # itself resolves paper vs. live -- env-sourced credentials default to
    # paper and only route live if ALPACA_LIVE_TRADE is exactly "true".
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    # v3: default changed indicative -> opra (requirement 13). Confirmed
    # against Alpaca's own docs that these are the only two valid values
    # for this parameter; alpaca_client.get_option_chain_snapshot now
    # catches the specific "not entitled" error and falls back to
    # indicative automatically, loudly, once -- so defaulting to the
    # better feed is safe even for an account that turns out not to be
    # subscribed. Explicitly setting ALPACA_OPTION_FEED=indicative in
    # .env still works exactly as before (no fallback logic runs for a
    # feed you asked for on purpose).
    data_feed: str = field(default_factory=lambda: os.getenv("ALPACA_OPTION_FEED", "opra"))
    stock_feed: str = field(default_factory=lambda: os.getenv("ALPACA_STOCK_FEED", "iex"))

    # -- what to trade --
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "SPY"))
    watchlist: list = field(default_factory=lambda: _list("WATCHLIST", DEFAULT_WATCHLIST))
    target_dte_days: int = field(default_factory=lambda: _int("TARGET_DTE_DAYS", 7))
    dte_tolerance_days: int = field(default_factory=lambda: _int("DTE_TOLERANCE_DAYS", 3))
    short_leg_target_delta: float = field(default_factory=lambda: _float("SHORT_LEG_TARGET_DELTA", 0.16))
    wing_target_delta: float = field(default_factory=lambda: _float("WING_TARGET_DELTA", 0.05))

    # -- safety --
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))
    # Separate from dry_run: this gates whether the Alpaca CLI itself is even
    # allowed to see ALPACA_LIVE_TRADE=true (alpaca_client.AlpacaClient._env).
    # Leave false. The CLI defaults to paper regardless, but this is the
    # project's own explicit opt-in on top of that default.
    i_understand_this_is_live: bool = field(default_factory=lambda: _bool("I_UNDERSTAND_THIS_IS_LIVE", False))
    enable_credit_spreads: bool = field(default_factory=lambda: _bool("ENABLE_CREDIT_SPREADS", True))
    enable_equity_hedge: bool = field(default_factory=lambda: _bool("ENABLE_EQUITY_HEDGE", True))
    min_open_interest: int = field(default_factory=lambda: _int("MIN_OPEN_INTEREST", 10))
    # -- v2: cross-checks/gates that were previously vendored code with no
    # caller (research/heston_signal.py, research/toxicity.py) or dead
    # code (pricing_engine.risk.friction_hedging.leland_cost_bps_of_vega).
    # Each degrades gracefully to v1 behavior when disabled or when its
    # own data requirements aren't met this cycle -- see the respective
    # module docstrings.
    enable_heston_cross_check: bool = field(default_factory=lambda: _bool("ENABLE_HESTON_CROSS_CHECK", True))
    enable_toxicity_gate: bool = field(default_factory=lambda: _bool("ENABLE_TOXICITY_GATE", True))
    enable_cost_floor: bool = field(default_factory=lambda: _bool("ENABLE_COST_FLOOR", True))
    enable_auto_exit: bool = field(default_factory=lambda: _bool("ENABLE_AUTO_EXIT", True))
    # -- v3: new cross-checks/gates, each independently toggleable and each
    # degrading gracefully to v2 behavior when disabled -- same pattern the
    # v2 flags above already established.
    enable_event_filter: bool = field(default_factory=lambda: _bool("ENABLE_EVENT_FILTER", True))
    enable_order_reconciliation: bool = field(default_factory=lambda: _bool("ENABLE_ORDER_RECONCILIATION", True))
    enable_probabilistic_vol: bool = field(default_factory=lambda: _bool("ENABLE_PROBABILISTIC_VOL", True))
    enable_execution_optimizer: bool = field(default_factory=lambda: _bool("ENABLE_EXECUTION_OPTIMIZER", False))
    # -- v3: execution-quality knobs (requirement 12) --
    max_bid_ask_spread_pct: float = field(default_factory=lambda: _float("MAX_BID_ASK_SPREAD_PCT", 0.10))
    max_quote_age_seconds: float = field(default_factory=lambda: _float("MAX_QUOTE_AGE_SECONDS", 15.0))
    commission_per_contract: float = field(default_factory=lambda: _float("COMMISSION_PER_CONTRACT", 0.0))
    # -- v3: order-state-machine knobs (requirement 15) --
    stale_order_max_age_minutes: float = field(default_factory=lambda: _float("STALE_ORDER_MAX_AGE_MINUTES", 10.0))
    # -- v3: event-filter blackout widths in trading days (requirement 8) --
    fomc_blackout_days_before: int = field(default_factory=lambda: _int("FOMC_BLACKOUT_DAYS_BEFORE", 1))
    fomc_blackout_days_after: int = field(default_factory=lambda: _int("FOMC_BLACKOUT_DAYS_AFTER", 0))
    cpi_nfp_blackout_days_before: int = field(default_factory=lambda: _int("CPI_NFP_BLACKOUT_DAYS_BEFORE", 0))
    cpi_nfp_blackout_days_after: int = field(default_factory=lambda: _int("CPI_NFP_BLACKOUT_DAYS_AFTER", 0))

    # v4.2: buying straddles is OFF by default (premium decays; this project's own research found the long-vol
    # side the weaker one). Entries are paced: cooldowns after an entry / exit, and no entries in the first and
    # last N minutes of the US session (widest spreads).
    enable_long_vol: bool = field(default_factory=lambda: _bool("ENABLE_LONG_VOL", False))
    entry_cooldown_minutes: int = field(default_factory=lambda: _int("ENTRY_COOLDOWN_MINUTES", 120))
    after_stop_cooldown_hours: float = field(default_factory=lambda: _float("AFTER_STOP_COOLDOWN_HOURS", 24.0))
    avoid_open_close_minutes: int = field(default_factory=lambda: _int("AVOID_OPEN_CLOSE_MINUTES", 15))

    # -- risk --
    risk: RiskLimits = field(default_factory=RiskLimits)
    hedge: HedgeParams = field(default_factory=HedgeParams)

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to .env "
                "and fill in a NEW, dedicated PAPER account's keys from "
                "https://app.alpaca.markets (Paper Trading) -- the hackathon playbook "
                "requires a fresh paper account created for this event, not a reused one."
            )
        if self.i_understand_this_is_live:
            log.warning(
                "I_UNDERSTAND_THIS_IS_LIVE=true -- the Alpaca CLI will be allowed to route "
                "orders to LIVE trading if ALPACA_LIVE_TRADE=true is also set in the "
                "environment. This project is built and validated for paper trading only."
            )


SETTINGS = Settings()
