# v4: trading desk

- Search a ticker or company name, see its price/day range/options availability, press **Run**: the agent trades that symbol on your Alpaca **paper** account (`dry_run=False`) every 5/15/30 min while the market is open. **Stop** halts new orders after the current cycle; **Stop and close positions** also cancels this agent's open orders and closes its positions (short legs first).
- Chart: 1m 3m 5m 15m 30m 1H 3H 4H 1D 1W, candles / Heikin Ashi / OHLC bars / line / area, volume, SMA 20, EMA 50, Bollinger, VWAP, horizontal + trend line tools, OHLC legend, live ticking last candle, times in New York.
- New routes: `GET /search`, `GET /ticker/{symbol}`, `GET /session`, `POST /session/start`, `POST /session/stop`; `GET /chart/bars` now takes `timeframe` + `count`.
- Session state is in memory; restarting the API server ends the session.
- The chart library is served locally from `static/lwc.js` (CDN fallback).

## v4.1
- Run is continuous (pause between cycles: `CYCLE_PAUSE_SECONDS`, default 15). No interval picker.
- Speed: one shared pooled Alpaca client, short server-side caches, ticker search list pre-warmed, chart tick uses a light `/quote` call, older history loads as you scroll left.
- Chart: regular-session US hours only, 9:30 ET-aligned 1H/3H/4H, split-adjusted, red/green candles, ranges 1D 5D 1M 3M 6M 1Y 5Y Max, all times ET.
- Needs `pip install -r requirements.txt` (adds `tzdata`, required for New York time on Windows).

## v4.2 -- trading logic + positions panel
- Entries: straddle buying OFF by default; paced (2 h cooldown after an entry, 24 h after a stop-loss/trailing exit, none in the first/last 15 min of the US session or when closed); max bid/ask spread 25% -> 10%.
- Exits: per-structure thresholds (condor TP +50% / stop -100% of credit, straddle TP +40% / stop -30%), a trailing stop (arms at +25%, exits on giving back half), close at <= 2 DTE (3 for long straddles).
- DTE and expiry maths use the New York date (a machine in India is a day ahead for the last hours of the US session, which triggered premature near-expiry exits).
- Hedge: capped at 100% of option delta; regression ratio clamped to [0.5, 1.0]; REHEDGE_MIN_INTERVAL_MIN is now actually enforced.
- Positions panel: lists every position (shares too) with avg cost, price and P&L; a Greeks failure no longer blanks the table; errors are shown.
- Tests updated for the new defaults, plus new tests for the trailing stop and the wider condor stop (80 pass).

## v4.3 -- phase 1 of the robust-vol-alpha rebuild (research/edge)
- `research/edge/forecast.py`: leakage-free rolling 7D/14D realized-vol forecasts -- EWMA, HAR (refit every 21d on realized targets only), GARCH(1,1) (refit every 63d), a regime-conditioning layer (calm/transition/crisis from expanding percentiles), and an ensemble weighted by rolling out-of-sample error. Tested: wrecking all data after day N changes no forecast made on or before day N.
- `research/edge/data_schema.py`: option-chain schema (ts, underlying, underlying_price, strike, expiration, cp, bid, ask, iv, volume, open_interest), loader/validator, and constant-maturity 7D/14D ATM IV using calls AND puts, interpolated in total variance, with a bid/ask liquidity filter and no extrapolation.
- `research/edge/validation.py`: decile study, walk-forward with an h-day embargo and an untouched final holdout, block bootstrap CI, permutation/placebo test, Deflated Sharpe Ratio, PBO (CSCV).
- `research/edge/run_edge_study.py`: runs the above, per-model ablation, writes `research/edge/out/edge_report_h{7,14}.md`. Verdict is REJECT unless the decile relationship is monotonic out-of-sample AND on the holdout.
- `backtest/engine.py`: the VRP history was today's forecast repeated 60 times; it is now the real rolling forecast (`use_rolling_forecast`).
- NOT done yet (phases 2+): continuous alpha score, HMM redesign, surface/skew/curvature + Heston gating, edge-calibration (P(profit)/EV), dynamic thresholds, portfolio pre-trade Greeks, vol-targeted sizing, strike optimisation, adaptive structures, realistic multileg fills, v4.2 exit/cooldown logic inside the backtester, regime reports, full ablations.

## v4.4 -- regime-conditioned vol relative-value engine (phase 2)
LIVE AGENT
- FIX: the theta cap counted |theta|, so a short-vol structure's theta INCOME was treated as risk. A single 1-lot 7-DTE condor (~$12.5k/yr of theta) breached the 1%-of-equity cap at any account below ~$1.3M. Now only adverse (negative) theta counts.
- Existing + PROPOSED Greeks are checked before every entry (condor and straddle); contracts are cut until the net book fits all four caps (strategy.portfolio.fit_contracts_to_greek_caps). Before, only the existing book was checked.
- Vol targeting (`vol_target_scale`, written long ago but never called) is now applied from the account's own daily equity curve, together with drawdown scaling and liquidity (ATM spread) scaling. Never sizes up past the hard caps.
- The forecast comes from `research/edge/forecast.py` (regime-selected model, same code as the backtest), ~6y of split-adjusted daily bars cached per day; the VRP z-score history uses it too. Falls back to the old forecast with a note if it can't run.
- Exits recognise vertical spreads as one structure (were legged out leg-by-leg).
RESEARCH / BACKTEST
- forecast.py: regime-dependent model selection from rolling out-of-sample errors in the SAME regime (+ sigma_err).
- data_schema.build_surface: synchronized 7/14/30D ATM IV, skew, curvature, term slope (calls + puts).
- alpha.py: VRP / term-structure carry / skew / jump premia with causal z-scores; walk-forward ridge composite; rolling IC (alpha-decay monitor).
- meta.py: meta-labeling (P(profit), E[net]) walk-forward; adaptive threshold (stricter in crisis/jump/wide spreads/event/high vol); cost model.
- strikes.py: structure selection by dominant alpha source; strike optimisation by (E[P&L] - jump & liquidity penalties) / CVaR on causal regime-conditional scenarios, real bid/ask fills.
- allocation.py: edge/uncertainty fractional Kelly with hard caps; drawdown / liquidity / alpha-decay scalers; ranking + CVaR-budget allocation; pre-trade Greek fit.
- backtest/live_replica.py: daily backtest that CALLS the live exit, cooldown/pacing, hedge, risk and Greek code (simulated US clock).
- trade_validation.py + run_replica_study.py: nested walk-forward, frozen holdout, 2x-spread stress, bootstrap, Deflated Sharpe, PBO, Holm, regime report, PASS/REJECT with reasons.
NOT DONE: real historical chain data (needs a vendor file), Heston gating by calibration quality, HMM redesign, calendar execution, alpha-decay scaling live, event/earnings tagging.
