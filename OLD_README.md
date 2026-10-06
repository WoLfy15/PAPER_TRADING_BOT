# alpaca-vol-agent

An autonomous options volatility-trading & hedging agent for Alpaca's
paper trading environment, built for the
[Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)
(lablab.ai x Alpaca, **Options Alpha Agents** track, Aug 28 - Sep 4 2026 15:00 UTC).

This is the **model / backend**, meant to sit behind a UI. Every Alpaca
call goes through the **official Alpaca CLI** (`alpacahq/cli`), run as a
subprocess -- not a bare REST/SDK call -- because the event's team
playbook is explicit about this: *"a bare SDK call alone doesn't
satisfy this"* requirement. See "Tooling compliance" below for exactly
how and why.

## Where this comes from

This project fuses two things that already existed independently:

1. **`OPTION-HEDGING-MODEL`** -- a regime-conditional (HMM) +
   GARCH/HAR-RV volatility-forecasting straddle engine with Kelly
   sizing and min-variance delta hedging (2nd place, IIT Guwahati
   options hackathon). Its own README flagged its #1 limitation
   plainly: *"Currently approximates implied vol as `realized_vol *
   1.15`. Replacing this with actual options chain data would make the
   VRP signal empirically grounded."*
2. **`options_pricing_engine`** -- an MFE-level pricing/risk library
   (BSM, Heston, Dupire local vol, Derman-Ergener-Kani static
   replication, Zou-Derman Strike-Adjusted Spread, Leland/Whalley-
   Wilmott transaction-cost-aware hedging), 55 passing tests, vendored
   here unmodified under `pricing_engine/`.

This project is exactly the fix #1 asked for, wired to Alpaca: the
GARCH/HAR forecast now gets checked against **live, real implied
volatility read straight off Alpaca's option chain**, cross-validated
by an independent, forecast-model-free signal (SAS, built only from
the underlying's own historical returns), and everything downstream
(sizing, hedging, execution) is now real Alpaca paper orders instead of
a backtest loop.

## Tooling compliance -- read this first

The playbook's hard requirement: *"Uses Alpaca's MCP server or CLI (a
bare SDK call alone doesn't satisfy this)."* `alpaca_client.py` routes
**every** Alpaca call -- account, positions, chain data, order
submission, including the multi-leg iron condor -- through
`subprocess.run(["alpaca", "api", METHOD, path, ...])`, i.e. the
official CLI's own documented raw escape hatch
(`alpaca api METHOD <path>`, piping JSON on stdin for POST -- exactly
the pattern in the CLI's own README and `.agents/skills/alpaca-cli/SKILL.md`).
Nothing here calls `requests` or `alpaca-py` against Alpaca directly.

Why the raw `alpaca api` escape hatch instead of the CLI's dedicated
`option` / `data option` / `data bars` subcommands: the CLI is an
*"Alpha Preview"* and its own skill doc says outright *"never rely on
stale documentation when the CLI is installed."* The `api` escape
hatch is the one interface documented as stable, and it lets this
project keep the exact REST paths already verified against
docs.alpaca.markets rather than guessing at a faster-moving flag
surface.

**One open item, flagged rather than hidden**: it isn't confirmed from
public docs whether `alpaca api GET <path>` resolves market-data paths
(`/v2/stocks/...`, `/v1beta1/options/...`) to `data.alpaca.markets`, or
only knows the trading host. Run `alpaca doctor` and one real
`get_stock_bars()` call after installing the CLI (see "How to run"
below) -- if market-data calls 404, swap `_data_get` in
`alpaca_client.py` to shell out to `alpaca data bars` / `alpaca data
option` instead. Every other method's signature is unaffected; that's
the only function that would need to change.

Why not the MCP server instead: the playbook recommends the MCP server
as primary specifically *for an LLM-driven agent (Claude Desktop /
Cursor)*, and calls the CLI path out for *"a custom Python/Node agent
loop"* -- which is exactly what this project already is (the GARCH/HAR
forecast, HMM regime detection, SAS cross-check, Kelly sizing, and
Whalley-Wilmott hedging bands are deterministic Python, not an LLM
making trading decisions). The CLI is the correct-fit tool for this
architecture, not a workaround.

## Architecture

```
config.py                 env-driven settings (ALPACA_API_KEY/ALPACA_SECRET_KEY, matches the CLI's own env vars)
alpaca_client.py            every call shells out to `alpaca api METHOD <path>` (see "Tooling compliance")
pricing_engine/               vendored: BSM, Heston, local vol, SAS, friction hedging, IV solver...
data/market_data.py           Alpaca chain + bars -> pricing_engine's OptionQuote/MarketSmilePoint
research/
  regime.py                    4-state HMM regime detection (Range/Trend/Vol_Expansion/Crash)
  vol_forecast.py                GARCH(1,1) + HAR-RV blended realized-vol forecast
  signal.py                       fuses forecast-vs-live-IV (VRP) with SAS into one composite edge
strategy/
  sizing.py                       fractional-Kelly risk budget, gamma/vega capped
  legs.py                          concrete, Alpaca-compliant option structures (see below)
  hedging.py                       min-variance hedge ratio + Whalley-Wilmott no-trade band
  portfolio.py                      live Greeks aggregated from actual Alpaca positions, drawdown kill switch
execution/order_manager.py     builds Alpaca order payloads (client_order_id stamped); DRY_RUN gates submission
agent/
  loop.py                        run_cycle(): fetch -> analyze -> decide -> explain -> execute
  state.py                        small JSON file: equity high-water mark + decision log (the "reasoning trail")
api/server.py                FastAPI backend for a UI (CORS open, plain JSON)
backtest/run_backtest.py     offline historical validation (yfinance, any symbol)
cli.py                       `python cli.py run|backtest|serve`
tests/                       26 pytest cases for everything new (mocks subprocess -- no live Alpaca calls)
```

## Hackathon track coverage (Options Alpha Agents)

One coherent strategy, not four bolted-together demos. It lands closest
to the playbook's **Option 3 (Volatility-Harvesting Agent)** --
*"sells iron condors/strangles when IV rank is elevated vs. its own
history... closes or hedges when IV compresses"* -- fused with
**Option 4 (Portfolio Hedge Overlay)** -- *"auto-buys protective
puts/collars when drawdown or volatility risk rises."* The playbook
flags Option 3 as *"more impressive, higher build risk"* and
recommends Option 1/2 for a from-scratch 7-day build; the reason to
keep Option 3 here is that it isn't from scratch -- the HMM regime
detection, GARCH/HAR forecasting, SAS cross-check, and Whalley-Wilmott
hedging bands already existed and are now just wired to live Alpaca
data. If the team wants to de-risk further before the deadline, the
straddle-only long-vol path (`direction == "long_vol"` in
`strategy/sizing.py`) is a strict subset that never touches multi-leg
orders at all -- Level 2 only, closer to the playbook's Option 1/2 risk
profile.

- **Volatility trading**: the core loop *is* a VRP (volatility risk
  premium) trade -- forecast realized vol vs. live implied vol.
- **Options alpha**: the VRP read is cross-checked against SAS
  (Zou-Derman), an independent, purely historical-returns-based
  cheap/rich signal, before either is trusted (see `research/signal.py`
  docstring for why using two unrelated methods matters here).
- **Hedging**: every options position is delta-hedged with the
  underlying using a min-variance hedge ratio and a transaction-cost-
  aware Whalley-Wilmott no-trade band (`strategy/hedging.py`), not a
  naive rehedge-to-zero-every-tick.
- **Portfolio overlays**: `strategy/portfolio.py` re-derives portfolio
  Greeks from whatever is *actually* sitting in the Alpaca account
  every cycle (gamma/vega risk caps, a 25%-drawdown kill switch, vol
  targeting) -- overlay logic that works whether the positions came
  from this agent or were placed by hand.

The playbook's competitive-intel note -- *"a real reasoning/risk layer
is a genuine differentiator, most entries will just be 'one model
decides'"* -- is what `AgentRunResult.signal_rationale`,
`.notes`, and `HedgeDecision.reason` are for: every cycle logged in
`agent_state.json` carries the human-readable *why*, not just the
*what*, ready to drive the dashboard's "reasoning trail" the playbook's
Day 5-6 plan calls for.

## A real broker constraint this project designs around

Alpaca's options trading levels (see
[docs.alpaca.markets/us/docs/options-trading](https://docs.alpaca.markets/us/docs/options-trading))
do **not** include a naked short straddle/strangle at any level --
Level 1 is covered calls / cash-secured puts, Level 2 is long calls/
puts, Level 3 is long spreads via `order_class="mleg"` where **every
leg must be covered within the same order**. So:

| Signal says... | Structure | Alpaca level | How |
|---|---|---|---|
| options cheap (long vol) | long straddle | 2 | two single-leg BUY orders |
| options rich (short vol) | iron condor | 3 | one `mleg` order, 4 legs, wings cover the shorts |

`strategy/legs.py`'s docstring has the full reasoning; this is the kind
of constraint that's easy to miss if you design a strategy against a
textbook options-trading algorithm rather than the specific broker API
you're shipping against.

## How to run

Roughly follows the playbook's own Day 1-2 setup:

**1. Create a NEW, dedicated paper account for this event** (the
playbook requires this -- don't reuse an existing one): go to
https://app.alpaca.markets, sign up or add a fresh paper account, and
generate a paper API key + secret.

**2. Install the Alpaca CLI** (required -- this is the tooling
compliance piece above):
```bash
go install github.com/alpacahq/cli/cmd/alpaca@latest
# or: brew install alpacahq/tap/cli
alpaca version          # confirms the binary is on PATH
```

**3. Set credentials** (env vars -- preferred for automation per the
CLI's own docs; nothing touches disk):
```bash
export ALPACA_API_KEY=PK...
export ALPACA_SECRET_KEY=...
alpaca account get --quiet     # sanity check -- should print your paper account JSON
alpaca doctor                  # broader connectivity check
```

**4. Install this project's Python dependencies:**
```bash
cd alpaca_vol_agent
pip install -r requirements.txt
cp .env.example .env
# .env reads the SAME ALPACA_API_KEY / ALPACA_SECRET_KEY you exported above --
# fill them in there too (or just keep them exported in your shell; either works)
```

**5. Run one dry-run agent cycle** (computes everything -- regime,
forecast vol, live IV, SAS, sizing, planned orders, hedge decision --
but `DRY_RUN=true` by default means nothing is actually submitted):
```bash
python cli.py run --once
```
Read the JSON it prints, especially `signal_rationale` and
`planned_orders[*].description` / `would_submit`. This is the "get ONE
full paper options trade working end to end" step the playbook calls
the riskiest part of Day 2-4 -- do this before building anything else.

**6. Once you're comfortable with what it's proposing, go live on
paper** (still 100% paper trading -- this only flips whether orders
actually submit to your paper account):
```bash
python cli.py run --once --live
```

**7. Run the offline backtest** (no Alpaca calls at all, validates the
regime/sizing/hedging machinery on real historical prices via
yfinance):
```bash
python cli.py backtest --symbol SPY --years 3
```

**8. Launch the API for the UI to hit:**
```bash
python cli.py serve --port 8000
# GET  localhost:8000/status     -- account equity, buying power, options level, market clock
# GET  localhost:8000/chain      -- live option chain with bid/ask/OI/delta/IV
# GET  localhost:8000/positions  -- portfolio Greeks from real Alpaca positions
# GET  localhost:8000/decisions  -- recent decision log + equity high-water mark
# POST localhost:8000/run        -- one full agent cycle, returns the complete AgentRunResult
```
`POST /run` is the one button a UI needs: it always computes and
returns the full signal + decision even when `dry_run` is true, so
"what would the agent do right now" is answerable without risking an
order.

**9. Run the tests** (mocked subprocess -- no `alpaca` binary or
network needed for this part):
```bash
pytest tests/ -v
```
26 tests cover everything this project adds on top of the vendored
`pricing_engine/` (which ships its own 55-passing/4-skipped-C++/
1-documented-xfail suite).

**10. For continuous paper trading** (e.g. to have something running
during judging), loop the agent instead of running it once:
```bash
python cli.py run --loop --interval 900   # every 15 minutes
```

## What's still on you for submission

This repo covers the "working prototype" piece. Still needed per the
playbook: deploy `api/server.py` somewhere reachable by URL (Render,
Fly.io, a spare VPS, or an ngrok tunnel for the demo window), build the
minimal dashboard against the `/status`, `/chain`, `/positions`,
`/decisions`, `/run` routes, record the pitch video (5 min max, MP4),
put together the slide deck (PDF), and push this to a **public**
GitHub repo. Happy to help draft the pitch deck outline or a
dashboard once the strategy choice above is confirmed.

## Safety

- `DRY_RUN=true` by default. The agent computes signals, sizes
  positions, and builds real order payloads regardless -- the flag only
  gates the final `alpaca_client.submit_order` call
  (`execution/order_manager.py:submit_orders`).
- The Alpaca CLI itself defaults every env-key-authenticated call to
  paper trading; it only routes live if `ALPACA_LIVE_TRADE=true` is set
  in the environment. `alpaca_client.AlpacaClient._env()` actively
  strips that variable out of the subprocess environment unless
  `config.Settings.i_understand_this_is_live` is explicitly `True` --
  two independent switches (this project's `DRY_RUN` and the CLI's own
  `ALPACA_LIVE_TRADE`) both have to be deliberately flipped before any
  order can reach live trading.
- A 25% account-drawdown kill switch blocks new entries
  (`strategy/portfolio.check_drawdown_kill_switch`), mirroring
  `OPTION-HEDGING-MODEL`'s original hard stop.
- Every submitted order is covered (long straddle) or defined-risk with
  all legs covered in the same multi-leg order (iron condor) -- see
  "A real broker constraint" above.
- Every order carries a `client_order_id` (stamped in
  `execution/order_manager.py`), per the CLI's own idempotency
  guidance -- a retry after an ambiguous failure checks `get_orders()`
  rather than blindly resubmitting.

This is a paper-trading research tool, not investment advice.

## v2 changes: what's new, what's validated, what's deliberately out

Four pieces of previously-vendored code (`pricing_engine`'s Heston
calibrator, `pricing_engine.risk.friction_hedging`'s Leland cost
function) or previously-separate projects (`rl_execution_extension`'s
VPIN, `rough_vol_project_v2`'s Hurst estimator) got wired in here for
real, plus one correctness fix that mattered more than any of them.
Everything below was actually run against real data or a realistic
synthetic scenario before shipping -- see each module's own docstring
for the specific test, not just a claim that it works.

**The highest-impact fix: position-aware sizing
(`strategy/portfolio.existing_vol_exposure_notional` +
`strategy/sizing.decide_sizing`).** v1 re-sized a *fresh* entry against
the full Kelly budget every single cycle with no memory of what was
already on the book. At the default 15-minute loop interval, that meant
re-buying more straddles (or selling more condors) every cycle for as
long as the composite edge stayed positive -- which it typically does
for many consecutive cycles, since regime/vol signals don't flip every
15 minutes. `decide_sizing` now takes the book's already-deployed
notional and only sizes the *remaining gap* to target. Verified
end-to-end against a faked Alpaca client: a flat book sizes a fresh
20-lot condor; the same setup with a partial existing position sizes a
correspondingly smaller order; a position already at or past the target
budget places *no new order at all*. This is a correctness fix, not a
new alpha source -- and it's the one most likely to matter for real P&L,
since unbounded re-entry both over-leverages the book and repeatedly
pays the bid/ask spread for no new edge. Known limitation: the short-vol
side approximates "risk already committed" from short legs' market
value times a fixed multiplier (`SHORT_VOL_RISK_MULTIPLIER`, see that
module's docstring) rather than reconstructing each open structure's
actual defined risk -- reasonable, not exact.

**`research/heston_signal.py`** -- calibrates the vendored Heston model
to the live chain's multi-strike (optionally multi-expiry) smile.
Deliberately *not* a fourth "is IV cheap or rich" signal on its own
(Heston is fit to match the smile it's given, so "model ATM IV vs.
market ATM IV" is close to circular). Two things it actually does: (1) a
data-quality cross-check -- if the smile-consistent fit disagrees
sharply with the raw ATM print, that flags a possibly stale/crossed
quote; (2) a genuine third signal leg, `term_gap` = forecast vol vs.
theta (the model's whole-curve-implied long-run vol level, informed by
skew/term structure, not just the one ATM point) -- *this* feeds
`research/signal.py`'s blend; and a vol-of-vol risk throttle from the
Feller ratio (low ratio / high vol-of-vol = fragile calibrated regime =
size down, regardless of which direction the edge points).

**`research/toxicity.py`** -- a VPIN-style order-flow read, adapted from
`rl_execution_extension`'s tick-level implementation to Alpaca's minute
bars via Bulk Volume Classification (BVC only needs price+volume per
bucket, not trade-level data -- a standard practical adaptation). One
real fix versus a literal port: BVC's standardization now uses a causal
*rolling* sigma instead of one whole-sample sigma. Verified why this
matters with a synthetic quiet-then-toxic-burst series while building
it: the whole-sample-sigma version scored the burst *lower* than quiet
baseline noise (backwards -- a single outlier-heavy window inflates the
one global sigma enough to suppress the standardized signal everywhere,
majority-quiet-buckets included); the rolling-sigma fix correctly peaks
during the burst. Used as an *execution-quality* gate (throttle size,
widen resting limit prices), not a directional signal -- matching how
VPIN is actually used in the literature. Reads its own trailing
percentile rather than an absolute threshold, because VPIN has a
well-documented noise floor around 0.5 even under pure noise (Andersen &
Bondarenko 2014) -- confirmed empirically here too (a 20-seed pure-noise
false positive check at the 97th-percentile "Toxic" cutoff came back
19/20 "Normal").

**Transaction costs, everywhere they were missing.**
`backtest/run_backtest.py` had *no* friction model at all -- every trade
priced at frictionless theoretical value. That's not a minor omission
for this strategy shape: re-running the exact v1 signal/sizing logic
through a lightweight research replica on the real SPY history this
repo ships (since this build environment has neither `arch` nor
`hmmlearn` installed to run the actual pipeline) showed Sharpe moving
from +0.15 to **-0.02** once realistic weekly-roll + daily-rehedge
friction was added at the existing `TRANSACTION_COST_BPS` default
(5bps) -- reproduced through the *actual* production `run_backtest()`
function (yfinance/arch/hmmlearn stubbed at the import boundary only)
with the identical result. `cost_bps=0` recovers the old frictionless
number if you want the comparison. Separately, a Leland transaction-cost
floor (`pricing_engine.risk.friction_hedging.leland_cost_bps_of_vega`,
vendored since v1 but never called) now gates live entries: an edge that
can't clear its own estimated hedging cost is skipped rather than sized.
Read honestly: in a friction-free toy backtest this filter can't show a
benefit *by construction* (it only ever removes trades, and removing a
trade that cost nothing to take never looks good) -- it's shipped on
transaction-cost-theory grounds and because a live paper account *does*
pay real spread/slippage, not because a backtest proved it adds Sharpe.

**Explicitly not done, and why:**
- **`rl_execution_extension`'s PPO execution agent** -- trained on a
  synthetic Hawkes limit-order-book simulator, and Alpaca's REST feed
  doesn't give this project a live L2 book to feed it in production
  anyway. Deploying an RL policy trained on synthetic microstructure
  straight to live paper execution, unvalidated, is exactly the kind of
  overfit risk worth flagging rather than shipping.
- **`rough_vol_project_v2`'s neural rough-Bergomi surface calibrator**
  (PyTorch, trained checkpoints) -- heavy, and this build environment
  doesn't have `torch` to even smoke-test it. The literature-standard,
  lightweight piece (`research.vol_forecast.estimate_hurst`, the
  Gatheral-Jaisson-Rosenbaum log-regression Hurst estimator) *was*
  tested as a forecast input: a rough-vol-kernel forecast built from it
  was compared out-of-sample against the existing HAR-RV baseline on the
  real SPY price history this repo ships. It did not improve on HAR --
  RMSE got *worse* at every blend weight tried, and the empirically-best
  weight on the rough-vol component came out to 0.0 (use HAR alone). Two
  reasons worth recording: applying a roughness estimator to an
  already-21-day-*rolling* (i.e. overlapping-averaged) realized-vol
  series is a methodological mismatch with the literature it's from
  (measured H ~= 0.57 here vs. the ~0.1 the literature reports from
  intraday data); and genuine roughness estimation needs higher-
  frequency data than one daily close per day provides, which this
  project's data source doesn't have. `estimate_hurst` is still exposed
  in the dashboard as a labeled diagnostic (vol-path "roughness")
  because it's informative context -- it just doesn't touch sizing.
- **`local_vol_surface.py` / `static_replication.py` / `pde_solver.py` /
  `black76.py`** -- reviewed; legitimate vendored code, no clearly
  non-redundant live use found within scope. Available for future work.

**Sharpe/risk metrics, end to end.** v1 persisted only a running equity
high-water mark (for the drawdown kill switch) -- no time series, so
live Sharpe was unanswerable from state at all. `agent/state.py` now
logs an equity point every cycle and `compute_risk_metrics` derives
Sharpe/Sortino/max-drawdown/win-rate from it, inferring the correct
annualization factor from the *observed* spacing between points (15-
minute-interval cycles and once-daily logging need very different
`sqrt(N)` scalings; hard-coding 252 the way the backtest correctly does
for its genuinely-daily series would silently misstate a live intraday
Sharpe by orders of magnitude -- cross-checked against a hand-computed
numpy Sharpe on a synthetic daily curve to confirm). Exposed via a new
`GET /metrics` endpoint and a new `POST /backtest` endpoint (runs
`run_backtest()` on demand), both rendered in the rebuilt
`dashboard.html`.

**One pre-existing thing worth flagging, unrelated to any of the
above:** `alpaca_client.py`'s actual code makes direct `requests` calls
against the Alpaca REST API; the "Tooling compliance" section above
describes routing every call through the `alpaca` CLI via
`subprocess`. That was true of an earlier version of this file and the
prose here was never updated when the implementation changed. If the
hackathon's CLI/MCP-only requirement is still in force, this is worth
reconciling one way or the other before submission -- rewrite the prose
to match the code, or route `alpaca_client.py` back through the CLI.
Left as-is here since it's orthogonal to this pass and a decision only
you can make (network access to `alpaca.markets` isn't available from
this build environment either way, so neither path could be tested
end-to-end here).

## Known v1 simplifications (still true in v2 unless noted above)

- **Min-variance hedge ratio** (`strategy/hedging.min_variance_hedge_ratio`)
  needs a persisted history of the book's own option-value returns to
  improve on the 1.0 baseline (fully hedging the aggregated BSM delta).
  `agent/state.py`'s JSON file doesn't yet log per-cycle option P&L --
  add that before relying on the regression branch live.
- **VRP z-score scale** is calibrated against a historical
  `realized_vol * 1.15` proxy (see `research/signal.py`'s
  `_historical_vrp_zscore` docstring) because pulling a daily history of
  Alpaca's *live* chain IV is expensive in the paper environment. The
  live VRP reading itself is always real; only the distribution used to
  judge "how unusual is this reading" uses the proxy.
- **Regime labeling**: `research/regime.py` maps 4 raw HMM states to
  {Range, Trend, Vol_Expansion, Crash} by ranking mean realized vol/
  return rather than trusting GaussianHMM's arbitrary state order --
  robust, but on some symbols/windows two of the four states can
  collapse to the same label (documented in the module).
- **State/persistence** is a single JSON file
  (`agent_state.json`), by design, for a hackathon-length paper-trading
  loop. Swap for a real datastore before running this unattended for
  weeks.
- **Backtest vs. live signal**: `backtest/run_backtest.py` uses the
  original notebook's `realized_vol * 1.15` IV proxy (no cheap way to
  pull years of historical option chains); the *live* agent
  (`agent/loop.py`) does not -- it always reads Alpaca's real chain. Its
  purpose is validating the regime/sizing/hedging machinery on real
  price history, not benchmarking live options-pricing accuracy.
- **`alpaca api` data-host routing** -- see "Tooling compliance" above;
  the one thing to verify locally with `alpaca doctor` once the CLI is
  installed, since it can't be tested from this sandbox (no network
  route to alpaca.markets here).
