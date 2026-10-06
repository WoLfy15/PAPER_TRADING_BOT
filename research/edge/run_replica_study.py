"""
Full study on REAL data:  python -m research.edge.run_replica_study --chain spy_chain.parquet --prices spy_daily.csv
  prices csv: date,close (SPY, split-adjusted).  chain: ts, underlying, underlying_price, strike, expiration, cp, bid, ask, iv, volume, open_interest
Runs surface -> causal signals -> nested walk-forward (live-replica backtest) -> frozen holdout -> 2x-spread stress ->
bootstrap / Deflated Sharpe / regime reports -> PASS or REJECT with reasons.  `--synthetic-smoke` runs the SAME code on a
generated chain: that only proves the pipeline executes; its numbers mean nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.live_replica import ReplicaConfig, index_chain, run_replica, summarize
from research.edge.data_schema import build_surface, load_chain
from research.edge.signals import prepare_signals
from research.edge.trade_validation import credibility, nested_walk_forward, regime_report, verdict

GRID = [{"p_base": p, "min_cost_mult": m} for p in (0.55, 0.62) for m in (1.5, 2.5)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain"), ap.add_argument("--prices"), ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--synthetic-smoke", action="store_true"), ap.add_argument("--days", type=int, default=1500)
    a = ap.parse_args()
    if a.synthetic_smoke:
        from research.edge.synth import synth_chain
        raw = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "reference" / "sp500_fred.csv")
        px = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy() / 10, index=pd.to_datetime(raw.iloc[:, 0])).dropna().iloc[-a.days:]
        r = np.log(px / px.shift(1))
        iv = (r.pow(2).ewm(alpha=0.06).mean().pow(0.5) * np.sqrt(252) * 1.15 + 0.01).clip(lower=0.08)
        chain, tag = synth_chain(px, iv, weekly=True), "SYNTHETIC DATA -- mechanics only, numbers are meaningless"
    else:
        chain, tag = load_chain(a.chain), "real option chain"
        d = pd.read_csv(a.prices)
        px = pd.Series(d.iloc[:, 1].to_numpy(), index=pd.to_datetime(d.iloc[:, 0])).dropna()
    px.index = px.index.normalize()
    surface = build_surface(chain)
    sig = prepare_signals(px, surface)
    snaps = index_chain(chain)
    nw = nested_walk_forward(px, snaps, sig, GRID)
    base = dataclasses.replace(ReplicaConfig(), **nw["holdout_config"])
    stress = summarize(run_replica(px, snaps, sig, dataclasses.replace(base, spread_mult=2.0), start=nw["dates"][1]))
    oos = summarize({"trades": nw["oos_trades"], "equity": (1 + nw["oos_returns"]).cumprod() * 100_000}) if not nw["oos_trades"].empty else {"n_trades": 0}
    oos["n_trials"] = nw["n_trials"]
    cred = credibility(nw["oos_trades"], nw["oos_returns"], nw["n_trials"])
    v, why = verdict(oos, cred, stress, nw["holdout_summary"], nw["oos_trades"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    md = [f"# Live-replica study ({tag})", "", f"Dates: {nw['dates'][0].date()} to {nw['dates'][2].date()}; untouched holdout starts after {nw['dates'][1].date()}", "",
          "## Outer folds (config chosen on training data only)", "", nw["folds"].round(3).to_markdown(index=False), "",
          "## Pooled out-of-sample", "", str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in oos.items()}), "", "## Credibility", "", str(cred), "",
          "## Final holdout (frozen config)", "", str(nw["holdout_summary"]), "", "## 2x-spread stress on the holdout", "", str(stress), "",
          "## Regime report (out-of-sample trades)", "", regime_report(nw["oos_trades"]).round(3).to_markdown(index=False) if not nw["oos_trades"].empty else "no trades", "",
          f"## Verdict: **{v}**", ""] + [f"- {w}" for w in why]
    (out / "replica_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
