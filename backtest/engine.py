"""
backtest/engine.py -- requirements 3 and 4:

  3. "rebuild the backtest with correct option premium cash accounting,
      commissions, bid/ask spread, slippage, assignment/expiration
      handling and daily NAV, and make the backtest trade the same
      iron-condor/vol structures that the live agent trades, not a
      synthetic short straddle"
  4. "refit the HMM/regime model strictly in rolling/walk-forward
      fashion to remove look-ahead bias"

WHAT CHANGED FROM v2's backtest/run_backtest.py and run_backtest_v2.py,
concretely: both fit `detect_regime(close)` ONCE on the entire close
series and then looped day-by-day through the resulting labels -- the
regime call on day 1 of a 10-year backtest already "knew" about every
crash and rally in the following 10 years, because the HMM was fit on
the whole series at once. Both also traded a single synthetic
`bs_straddle` helper for BOTH long_vol AND short_vol (short_vol just
went negative on the same straddle) instead of the real
strategy/legs.py structures. And both used `realized_vol * 1.15` as the
implied-vol input. This file fixes all three: the loop below refits
regime + the horizon-matched forecast ONLY at each roll date, using
ONLY data up to that date (walk-forward, by construction -- there is no
code path here that can see forward); it calls the REAL
research/signal.build_signal, strategy/sizing.decide_sizing, and
strategy/legs (build_straddle / select_iron_condor_by_cvar) -- the
EXACT SAME functions agent/loop.py calls for a live cycle, not a
separate reimplementation; and forecast_vol_now comes from real VIX
(data/historical_iv.py) wherever this file needs a historical IV level,
never a multiple of the realized-vol series being forecast.

WHAT THIS STILL IS AND ISN'T, stated as plainly as possible because
overstating this would be a real problem for something that could
influence real trading decisions:

  IS: a genuine walk-forward simulation, on REAL historical data (S&P
  500 levels and CBOE VIX, both bundled in data/reference/ -- see
  data/historical_iv.py's own docstring for exactly what these are and
  aren't), that runs the ACTUAL strategy code, with real explicit cash
  accounting for commissions/spread/slippage/expiration
  (accounting/ledger.py).

  IS NOT: a backtest against a real historical OPTIONS order book.
  There is no bundled multi-year per-strike historical bid/ask dataset
  (that's a paid vendor's product -- OptionMetrics, ORATS, CBOE
  DataShop; data/historical_iv.py's IVSurfaceProvider is the seam one
  plugs into). Every option "quote" this engine trades against is a
  BSM price computed from the real underlying level and the real VIX
  level that day, with a documented, simple skew and a documented,
  simple modeled bid/ask spread (accounting/ledger.py's
  modeled_half_spread_pct) -- not observed market quotes. Sharpe,
  drawdown, and every other number this file computes are real
  arithmetic on the P&L this simulation actually produced, but that
  P&L is only as realistic as the assumptions above. Do not read a
  Sharpe out of this file as a validated, tradeable edge -- read it as
  "how would this exact strategy logic have behaved against this
  semi-realistic reconstruction," which is a meaningfully better
  question to be able to answer than v2's backtest could, but still a
  different question from "what would this actually have earned."
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from accounting.ledger import Ledger, LedgerPosition
from config import RiskLimits, Settings
from data.historical_iv import historical_iv_proxy, vix_level_asof
from data.market_data import ChainContract
from pricing_engine.market_data.contract import OptionType
from pricing_engine.risk.scenarios import StressReport, build_stress_report
from research.event_filter import assess_event_risk
from research.regime import detect_regime
from research.signal import build_signal
from research.vol_forecast import forecast_horizon
from research.edge.forecast import build_forecasts
from strategy.legs import build_straddle, condor_structure_legs_per_contract, select_iron_condor_by_cvar
from strategy.sizing import decide_sizing

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.045
# Documented, simple skew: OTM puts trade at higher IV than OTM calls
# by this many vol points per 1% of moneyness -- the standard equity
# index "smirk" shape, not fit to any specific observed surface.
PUT_SKEW_PTS_PER_PCT_MONEYNESS = 0.15
CALL_SKEW_PTS_PER_PCT_MONEYNESS = 0.05


@dataclass
class BacktestConfig:
    symbol: str = "SPY"
    starting_cash: float = 100_000.0
    roll_freq_days: int = 21             # how often the strategy re-evaluates & can open a new structure
    target_dte_days: int = 30
    strike_spacing: float = 5.0
    n_strikes_each_side: int = 12
    commission_per_contract: float = 0.0
    slippage_bps_of_notional: float = 2.0
    min_history_days: int = 260          # need >=1 trading year before the first regime/GARCH fit
    risk: RiskLimits = field(default_factory=RiskLimits)
    enable_event_filter: bool = True
    enable_credit_spreads: bool = True
    wing_delta_candidates_mult: tuple[float, ...] = (0.6, 1.0, 1.6)
    short_leg_target_delta: float = 0.16
    use_rolling_forecast: bool = True    # v4.3: genuine rolling forecast history (was: today's forecast repeated 60 times)
    wing_target_delta: float = 0.05


@dataclass
class RollRecord:
    date: dt.date
    regime_label: str
    composite_edge: float
    direction: str
    contracts: int
    notes: list[str] = field(default_factory=list)
    stress_report: Optional[StressReport] = None  # set when direction == "short_vol" -- see select_iron_condor_by_cvar


@dataclass
class BacktestResult:
    daily_nav: pd.Series
    daily_returns: pd.Series
    roll_log: list[RollRecord]
    positions: list[LedgerPosition]
    final_cash: float
    config: BacktestConfig


def _synthetic_chain(spot: float, iv_level: float, as_of: dt.date, expiry: dt.date, cfg: BacktestConfig) -> list[ChainContract]:
    """BSM-priced synthetic chain at the real spot + real VIX-derived IV
    level for that date, with a documented simple skew (see module
    docstring). This is the honest stand-in for a real historical option
    chain -- see the module docstring's IS/IS NOT section."""
    out = []
    base_strike = round(spot / cfg.strike_spacing) * cfg.strike_spacing
    for i in range(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1):
        strike = base_strike + i * cfg.strike_spacing
        if strike <= 0:
            continue
        moneyness_pct = (strike - spot) / spot * 100
        for otype in (OptionType.CALL, OptionType.PUT):
            skew = (-PUT_SKEW_PTS_PER_PCT_MONEYNESS * moneyness_pct if otype == OptionType.PUT
                    else CALL_SKEW_PTS_PER_PCT_MONEYNESS * moneyness_pct)
            iv = max(iv_level + skew / 100.0, 0.03)
            from pricing_engine.market_data.contract import OptionQuote
            from pricing_engine.models.bsm import BlackScholesMerton
            tte = max((expiry - as_of).days, 1) / 365.0
            q = OptionQuote(underlying=cfg.symbol, spot=spot, strike=strike, expiry_years=tte, option_type=otype, risk_free_rate=RISK_FREE_RATE, market_price=None)
            fair = max(BlackScholesMerton().price(q, iv), 0.01)
            half_spread = fair * 0.02
            label = f"{cfg.symbol}{expiry.strftime('%y%m%d')}{'C' if otype == OptionType.CALL else 'P'}{int(strike*1000):08d}"
            out.append(ChainContract(
                symbol=label, underlying=cfg.symbol, strike=strike, expiry=expiry, option_type=otype,
                open_interest=1000, tradable=True, bid=round(fair - half_spread, 2), ask=round(fair + half_spread, 2),
                last=round(fair, 2), alpaca_iv=iv, alpaca_delta=None, alpaca_gamma=None, alpaca_theta=None, alpaca_vega=None,
                quote_timestamp=dt.datetime.combine(as_of, dt.time(16, 0), tzinfo=dt.timezone.utc),
            ))
    return out


def _legs_mid_tuples(legs: list[tuple[ChainContract, float]]) -> list[tuple[str, OptionType, float, float, float]]:
    return [(c.symbol, c.option_type, c.strike, qty, c.mid or c.last or 0.0) for c, qty in legs]


def run_backtest(price_series: pd.Series, cfg: Optional[BacktestConfig] = None) -> BacktestResult:
    """price_series: a real, date-indexed daily close series (e.g. the
    bundled S&P 500 level scaled to a SPY-like price, or a real
    yfinance/Alpaca download when run with network access -- this
    function itself has no data-source opinion, it just needs a real
    price history). Walks forward day by day; at each roll date (every
    cfg.roll_freq_days trading days, or when the currently-open
    structure expires, whichever comes first) it evaluates the strategy
    fresh -- with a regime fit and a horizon forecast built from ONLY
    the data up to that day -- exactly mirroring what a live cycle sees."""
    cfg = cfg or BacktestConfig()
    price_series = price_series.dropna().sort_index()
    log_returns_full = np.log(price_series / price_series.shift(1)).dropna()
    edge_fc = None   # causal ensemble forecast for EVERY date (value at t uses data <= t only -- see tests/test_edge_pipeline.py)
    if cfg.use_rolling_forecast:
        try:
            edge_fc = build_forecasts(log_returns_full, cfg.target_dte_days)["ensemble"].dropna()
        except Exception:  # noqa: BLE001 -- short history etc.: fall back to the old single-point forecast
            edge_fc = None

    ledger = Ledger(cfg.starting_cash, cfg.commission_per_contract, cfg.slippage_bps_of_notional)
    roll_log: list[RollRecord] = []
    open_position: Optional[LedgerPosition] = None
    open_position_kind: Optional[str] = None
    days_since_roll = 10**9  # force a roll evaluation on day 1

    dates = price_series.index[cfg.min_history_days:]
    for today in dates:
        today_date = today.date() if hasattr(today, "date") else today
        spot = float(price_series.loc[today])
        iv_level = vix_level_asof(today_date) or float(log_returns_full.loc[:today].tail(21).std() * np.sqrt(252))

        # --- settle/roll the open position if it expired ---
        if open_position is not None and not open_position.closed and open_position.expiry <= today_date:
            ledger.settle_expiration(open_position, spot)
            open_position = None
            open_position_kind = None

        # --- mark to market EVERY day, whether or not anything rolls today ---
        ledger.mark_to_market(today_date, spot, iv_level, RISK_FREE_RATE)

        days_since_roll += 1
        should_evaluate = (open_position is None) or (days_since_roll >= cfg.roll_freq_days)
        if not should_evaluate:
            continue
        days_since_roll = 0

        # ---- WALK-FORWARD: everything below uses ONLY data up to `today` ----
        hist_returns = log_returns_full.loc[:today]
        if len(hist_returns) < cfg.min_history_days:
            continue
        hist_prices = price_series.loc[:today]
        realized_vol = (hist_returns.rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).dropna()
        aligned_returns = hist_returns.loc[realized_vol.index]

        regime = detect_regime(hist_prices.loc[realized_vol.index[0]:])  # fit ONLY on history up to today
        expiry = today_date + dt.timedelta(days=cfg.target_dte_days)
        try:
            hf = forecast_horizon(aligned_returns, realized_vol, horizon_days=cfg.target_dte_days)
            forecast_vol_now = hf.blended_expected_vol
        except Exception:
            forecast_vol_now = float(realized_vol.iloc[-1])

        forecast_hist = pd.Series(forecast_vol_now, index=aligned_returns.index[-60:])
        if edge_fc is not None:
            ef = edge_fc.loc[:today]
            if len(ef) >= 60:
                forecast_vol_now, forecast_hist = float(ef.iloc[-1]), ef.tail(60)
        chain = _synthetic_chain(spot, iv_level, today_date, expiry, cfg)
        signal = build_signal(
            symbol=cfg.symbol, regime_label=regime.current_label, forecast_vol_now=forecast_vol_now,
            live_atm_iv=iv_level, forecast_vol_history=forecast_hist,
            realized_vol_history=realized_vol,
        )

        event_multiplier, event_blocked, event_reasons = 1.0, False, []
        if cfg.enable_event_filter:
            er = assess_event_risk(today_date, daily_log_returns=hist_returns.values)
            event_multiplier, event_blocked, event_reasons = er.short_vol_size_multiplier, er.short_vol_blocked, er.reasons

        # close whatever's open before sizing something new -- one
        # structure at a time in this simplified engine (see BacktestConfig)
        if open_position is not None and not open_position.closed:
            close_chain = {c.symbol: c.mid for c in _synthetic_chain(spot, iv_level, today_date, open_position.expiry, cfg)}
            ledger.close_structure(open_position, today_date, close_chain, spot, reason="rolled")
            open_position, open_position_kind = None, None

        straddle = build_straddle(chain, expiry, spot, RISK_FREE_RATE, as_of=today_date)
        equity_now = ledger.daily_records[-1].nav if ledger.daily_records else cfg.starting_cash
        sizing = None
        if straddle is not None:
            sizing = decide_sizing(
                signal.composite_edge, regime.current_label, equity_now,
                straddle.price, straddle.gamma, straddle.vega, cfg.risk,
                short_vol_extra_scale=event_multiplier, short_vol_blocked=bool(event_blocked),
                short_vol_block_reason="; ".join(event_reasons),
            )

        record = RollRecord(date=today_date, regime_label=regime.current_label, composite_edge=signal.composite_edge, direction="flat", contracts=0)
        if sizing and sizing.direction == "long_vol" and sizing.contracts > 0 and straddle is not None:
            legs_mid = _legs_mid_tuples([(straddle.call, 1.0), (straddle.put, 1.0)])
            open_position, _fill = ledger.open_structure(today_date, "straddle", expiry, legs_mid, sizing.contracts, spot)
            open_position_kind = "straddle"
            record.direction, record.contracts = "long_vol", sizing.contracts
        elif sizing and sizing.direction == "short_vol" and cfg.enable_credit_spreads:
            condor = select_iron_condor_by_cvar(
                chain, expiry, spot, RISK_FREE_RATE, cfg.short_leg_target_delta,
                [cfg.wing_target_delta * m for m in cfg.wing_delta_candidates_mult],
                sizing.max_structure_loss, cfg.risk.max_contracts_per_leg,
                equity=equity_now, max_structure_cvar_pct=cfg.risk.max_structure_cvar_pct, cvar_alpha=cfg.risk.cvar_alpha,
                as_of=today_date,
            )
            if condor is not None:
                legs_mid = _legs_mid_tuples(condor_structure_legs_per_contract(condor))
                open_position, _fill = ledger.open_structure(today_date, "iron_condor", expiry, legs_mid, condor.contracts, spot)
                open_position_kind = "iron_condor"
                record.direction, record.contracts = "short_vol", condor.contracts
                record.stress_report = condor.stress_report
        roll_log.append(record)

    daily_nav = pd.Series({r.date: r.nav for r in ledger.daily_records}).sort_index()
    daily_returns = daily_nav.pct_change().dropna()
    return BacktestResult(
        daily_nav=daily_nav, daily_returns=daily_returns, roll_log=roll_log,
        positions=ledger.positions, final_cash=ledger.cash, config=cfg,
    )
