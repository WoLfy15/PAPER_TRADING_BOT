"""
Forecast-minus-IV edge study (leakage-free). Examples:
  python -m research.edge.run_edge_study --horizon 7 --chain data/spy_chain.parquet          # real chain (the target)
  python -m research.edge.run_edge_study --horizon 7 --iv-csv spy_iv.csv                    # date,iv_7d,iv_14d
  python -m research.edge.run_edge_study --horizon 14 --allow-vix-proxy                      # indicative only
Verdict is REJECT unless the decile relationship is monotonic out-of-sample AND on the untouched holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.edge.data_schema import constant_maturity_iv, load_chain
from research.edge.forecast import build_forecasts
from research.edge.validation import permutation_pvalue, walk_forward_study

REF = Path(__file__).resolve().parents[2] / "data" / "reference"
COMPONENTS = ["ewma", "har", "garch", "har_regime", "ensemble"]


def load_prices(path: str | None) -> pd.Series:
    df = pd.read_csv(path or REF / "sp500_fred.csv")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0])).dropna()
    return s.sort_index()


def load_iv(args, h: int) -> tuple[pd.Series, str]:
    if args.chain:
        return constant_maturity_iv(load_chain(args.chain), targets=(7, 14))[f"iv_{h}d"], f"real option chain ({args.chain})"
    if args.iv_csv:
        d = pd.read_csv(args.iv_csv, index_col=0, parse_dates=True)
        return d[f"iv_{h}d"], f"IV file ({args.iv_csv})"
    if args.allow_vix_proxy:
        v = pd.read_csv(REF / "vix_daily.csv")
        return pd.Series(v["CLOSE"].to_numpy() / 100, index=pd.to_datetime(v["DATE"])), "VIX proxy: 30-day, flat across DTE -- NOT a true 7/14-day implied vol; indicative only"
    raise SystemExit("No implied-vol source. Provide --chain or --iv-csv (real data), or --allow-vix-proxy for an indicative run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7, choices=[7, 14])
    ap.add_argument("--prices"), ap.add_argument("--chain"), ap.add_argument("--iv-csv")
    ap.add_argument("--allow-vix-proxy", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    a = ap.parse_args()
    h = a.horizon
    px = load_prices(a.prices)
    ret = np.log(px / px.shift(1)).dropna()
    iv, src = load_iv(a, h)
    fc = build_forecasts(ret, h)
    iv = iv.reindex(fc.index)
    rows, detail = [], {}
    for comp in COMPONENTS:
        sig, out = fc[comp] - iv, fc["target"] - iv
        r = walk_forward_study(sig, out, h)
        ho = r["holdout"]
        rows.append({"model": comp, "oos_ic": r["oos"]["ic"], "oos_decile_rho": r["oos"]["decile_rho"], "oos_spread": r["oos"]["spread"],
                     "oos_spread_ci_low": r["oos"]["spread_ci"][0], "holdout_ic": ho["ic"] if ho else np.nan,
                     "holdout_decile_rho": ho["decile_rho"] if ho else np.nan, "verdict": "REJECT" if r["reject"] else "PASS"})
        detail[comp] = r
    p_val = permutation_pvalue(fc["ensemble"] - iv, fc["target"] - iv, h)
    table = pd.DataFrame(rows).set_index("model")
    ens = detail["ensemble"]
    md = [f"# Forecast-minus-IV edge study, horizon {h}D", "", f"IV source: **{src}**", f"Sample: {ret.index[0].date()} to {ret.index[-1].date()}  (dev n={ens['n_dev']}, untouched holdout n={ens['n_holdout']})", "",
          "Signal = forecast vol - implied vol.  Outcome = subsequent realized vol - implied vol.  All bins/edges from training data only.", "",
          "## Per-model out-of-sample results (this is also the ablation of the forecast components)", "", table.round(3).to_markdown(), "",
          "## Ensemble: pooled out-of-sample deciles (1 = most negative signal, 10 = most positive)", "", ens["oos"]["table"].round(4).to_markdown(), ""]
    if ens["holdout"]:
        md += ["## Ensemble: untouched holdout deciles", "", ens["holdout"]["table"].round(4).to_markdown(), ""]
    md += [f"Permutation (placebo) p-value of the ensemble IC, whole sample: **{p_val:.3f}**", "",
           f"## Verdict: **{'REJECT' if ens['reject'] else 'PASS'}**",
           "", "REJECT = the monotonic decile relationship did not hold out-of-sample and/or on the holdout." if ens["reject"] else "Monotonic out-of-sample and on the holdout."]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"edge_report_h{h}.md").write_text("\n".join(md))
    (out / f"edge_report_h{h}.json").write_text(json.dumps({"iv_source": src, "table": table.round(4).to_dict("index"), "perm_p": p_val, "verdict": "REJECT" if ens["reject"] else "PASS"}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
