"""
cli.py -- single entry point for everything in this project.

    python cli.py run --once              # one agent cycle, respects DRY_RUN in .env
    python cli.py run --once --live       # force live execution for this run only
    python cli.py run --loop --interval 900   # repeat every 15 minutes
    python cli.py backtest --symbol NVDA --years 3
    python cli.py serve --port 8000       # launch the FastAPI backend for a UI
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alpaca_vol_agent.cli")


def _settings_for_run(args: argparse.Namespace):
    from config import SETTINGS

    if not args.live:
        return SETTINGS
    log.warning("Running LIVE (dry_run=False) for this invocation -- real paper orders will submit.")
    # v3 fix: this used to hand-reconstruct Settings(...) field by field
    # (api_key, api_secret, data_feed, ... risk, hedge), which meant every
    # field added to Settings after that list was written -- which is most
    # of them, as of v3 -- silently reverted to its default instead of
    # inheriting from the real SETTINGS. dataclasses.replace carries over
    # every field except the one explicitly overridden here, by
    # construction, so this can't happen again by omission. See
    # api/server.py's identical fix (and its module docstring) for the
    # concrete case that surfaced this.
    return dataclasses.replace(SETTINGS, dry_run=False)


def cmd_run(args: argparse.Namespace) -> None:
    from agent.loop import run_cycle

    settings = _settings_for_run(args)

    def one_cycle():
        result = run_cycle(settings)
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        for note in result.notes:
            log.warning(note)

    if args.loop:
        log.info("Starting agent loop, interval=%ss, dry_run=%s, symbol=%s", args.interval, settings.dry_run, settings.symbol)
        while True:
            try:
                one_cycle()
            except Exception:
                log.exception("agent cycle failed; will retry next interval")
            time.sleep(args.interval)
    else:
        one_cycle()


def cmd_backtest(args: argparse.Namespace) -> None:
    """v3 fix: this used to call backtest.run_backtest.run_backtest --
    v2's synthetic-short-straddle backtest with the realized_vol*1.15 IV
    proxy and a full-history (look-ahead-biased) regime fit -- meaning
    the CLI's `backtest` subcommand was silently bypassing every one of
    the backtest fixes in CHANGES_V3.md items 1-4/11/19 even after they
    were built, since nothing routed this specific entry point to them.
    Now runs backtest.engine's real walk-forward engine +
    backtest.walk_forward's evaluation report -- the SAME path
    api/server.py's POST /backtest uses, and the same strategy code
    (research.signal, strategy.sizing, strategy.legs) the live agent's
    own run_cycle calls. See backtest/engine.py's own module docstring
    for exactly what is and isn't real about the resulting numbers
    before reading a Sharpe off this as a validated result."""
    from backtest.engine import BacktestConfig, run_backtest
    from backtest.walk_forward import run_walk_forward_report

    if args.use_bundled_reference_data:
        import pandas as pd
        sp = pd.read_csv("data/reference/sp500_fred.csv", parse_dates=["observation_date"]).set_index("observation_date")["SP500"].dropna()
        price_series = sp / 10.0  # SPX/10 ~= SPY-like level -- see data/historical_iv.py's docstring
        cutoff = price_series.index.max() - pd.Timedelta(days=int(args.years * 365.25))
        price_series = price_series.loc[price_series.index >= cutoff]
    else:
        import yfinance as yf
        import datetime as dt
        end = dt.date.today()
        start = end - dt.timedelta(days=int(args.years * 365.25))
        df = yf.download(args.symbol, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df.empty:
            raise SystemExit(f"yfinance returned no data for {args.symbol} -- check network access and the symbol, or pass --use-bundled-reference-data for an offline demo.")
        price_series = df["Close"].dropna()
        if hasattr(price_series, "squeeze"):
            price_series = price_series.squeeze()

    cfg = BacktestConfig(symbol=args.symbol, starting_cash=args.capital)
    result = run_backtest(price_series, cfg)
    report = run_walk_forward_report(result)
    print(json.dumps({
        "full_sample": dataclasses.asdict(report.full_sample),
        "folds": [dataclasses.asdict(f) for f in report.folds],
        "median_fold_sharpe": report.median_fold_sharpe, "worst_fold_sharpe": report.worst_fold_sharpe,
        "bootstrap_ci": dataclasses.asdict(report.bootstrap_ci) if report.bootstrap_ci else None,
        "worst_seen_stress": report.worst_seen_stress,
        "acceptance": report.acceptance,
        "notes": report.notes,
    }, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca volatility trading agent -- CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the agent")
    p_run.add_argument("--once", action="store_true", default=True, help="(default) run a single cycle")
    p_run.add_argument("--loop", action="store_true", help="run continuously")
    p_run.add_argument("--interval", type=int, default=900, help="seconds between cycles in --loop mode")
    p_run.add_argument("--live", action="store_true", help="override DRY_RUN=false for this invocation")
    p_run.set_defaults(func=cmd_run)

    p_bt = sub.add_parser("backtest", help="run the offline historical backtest")
    p_bt.add_argument("--symbol", default="SPY")
    p_bt.add_argument("--years", type=float, default=3.0)
    p_bt.add_argument("--capital", type=float, default=1_000_000.0)
    p_bt.add_argument(
        "--use-bundled-reference-data", action="store_true",
        help="run against the S&P-500-derived price series bundled in data/reference/ instead of "
             "downloading via yfinance -- an offline demo path, no network access needed",
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_serve = sub.add_parser("serve", help="launch the FastAPI backend")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
