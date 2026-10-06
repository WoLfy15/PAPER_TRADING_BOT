"""
backtest/run_backtest.py -- historical backtest of the fused
regime+GARCH/HAR+Kelly straddle strategy, generalized from
IITfinal.ipynb's NVDA-only backtest to any symbol via yfinance.

This intentionally does NOT hit Alpaca at all -- it's an offline sanity
check you run once before pointing agent/loop.py at a live paper
account, same role the original notebook's backtest played. Because
there's no cheap way to pull a multi-year daily history of live option
chains for a backtest, this keeps the original notebook's honestly-
flagged simplification: implied vol is proxied as `realized_vol * 1.15`
rather than the live-chain-derived number agent/loop.py uses. Treat this
as validating the REGIME + SIZING + HEDGING machinery on real price
history, not as a claim about live options pricing accuracy -- the
live agent's signal (research/signal.py) is materially better-grounded
than this backtest's proxy, by design.

v2: charges an explicit transaction-cost drag (cost_bps of notional on
every hedge share traded, cost_bps*option_cost_multiple of premium on
every straddle contract opened/closed at a roll) -- v1 had NO friction
model at all, pricing every trade at frictionless theoretical
Black-Scholes value. That is not a minor omission for THIS strategy
shape: re-running v1's exact signal/sizing logic through a lightweight
research replica of this same backtest (numpy/pandas only, since this
sandbox has neither `arch` nor `hmmlearn` installed to run the real
pipeline) on the real SPY price history this project ships
(backtest/out/SPY_backtest.csv) showed Sharpe moving from +0.15 to
-0.02 once realistic weekly-roll + daily-rehedge friction was added at
TRANSACTION_COST_BPS's own default (5bps). A strategy this
trade-frequent needs its transaction costs modeled to be read honestly
at all; cost_bps=0.0 recovers the exact v1 (frictionless) behavior if
you want it for comparison.

Usage:
    python -m backtest.run_backtest --symbol NVDA --years 3
    python -m backtest.run_backtest --symbol NVDA --years 3 --cost-bps 0   # v1's frictionless behavior
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.regime import detect_regime, REGIME_LABELS  # noqa: E402
from research.vol_forecast import forecast_volatility, har_rv  # noqa: E402
from strategy.sizing import REGIME_SIZE_SCALE  # noqa: E402
from config import RiskLimits  # noqa: E402


def bs_straddle(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float, float, float, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    return call + put, delta, gamma, vega, theta


@dataclass
class BacktestSummary:
    symbol: str
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    win_rate: float
    final_capital: float
    n_days: int
    total_friction_paid: float
    cost_bps: float


def run_backtest(
    symbol: str, years: float = 3.0, initial_capital: float = 1_000_000.0,
    risk: RiskLimits | None = None, roll_freq_days: int = 7, out_dir: str = "backtest/out",
    cost_bps: float = 5.0, option_cost_multiple: float = 3.0,
) -> BacktestSummary:
    """cost_bps: proportional transaction cost, applied to BOTH the daily
    equity hedge rebalance and straddle contract rolls (option leg cost
    is charged at cost_bps * option_cost_multiple, since listed option
    spreads run wider than the underlying's -- 3x is a conservative
    standard rule of thumb, not fit to this data). 0.0 recovers v1's
    frictionless backtest exactly. Defaults to
    config.HedgeParams.transaction_cost_bps's own default (5bps) so the
    backtest and the live agent's Whalley-Wilmott band assume the same
    cost environment unless told otherwise."""
    risk = risk or RiskLimits()
    data = yf.download(symbol, period=f"{years:.0f}y", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close = data["Close"].dropna()
    if len(close) < 300:
        raise ValueError(f"only {len(close)} bars for {symbol}; need >=300 for a meaningful backtest")

    regime = detect_regime(close)
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = (log_returns.rolling(21).std() * np.sqrt(252)).dropna()
    aligned_returns = log_returns.loc[realized_vol.index]
    vol_fc = forecast_volatility(aligned_returns, realized_vol)

    idx = regime.labels.index.intersection(vol_fc.blended.index)
    frame = pd.DataFrame({
        "close": close.loc[idx], "regime": regime.labels.loc[idx],
        "forecast_vol": vol_fc.blended.loc[idx], "realized_vol": realized_vol.loc[idx],
    }).dropna()
    frame["implied_vol_proxy"] = frame["realized_vol"] * 1.15  # see module docstring
    frame["vrp"] = frame["forecast_vol"] - frame["implied_vol_proxy"]
    frame["vrp_z"] = (frame["vrp"] - frame["vrp"].rolling(60).mean()) / frame["vrp"].rolling(60).std()
    frame = frame.dropna()

    r = 0.045
    k = cost_bps / 10_000.0
    cash = initial_capital
    shares = 0.0
    contracts_held = 0
    trade_log = []
    K = round(frame["close"].iloc[0] / 5) * 5
    expiry_idx = 0
    total_friction = 0.0

    for i, (date, row) in enumerate(frame.iterrows()):
        capital_mark = cash + shares * row["close"]  # pre-trade mark, for the drawdown guard only
        if capital_mark < (1 - risk.max_drawdown_pct) * initial_capital:
            trade_log.append({"date": date, "capital": capital_mark, "regime": row["regime"], "contracts": 0, "friction": 0.0})
            continue

        rolled = (i % roll_freq_days == 0)
        if rolled:
            K = round(row["close"] / 5) * 5
            expiry_idx = i + roll_freq_days
        T = max((expiry_idx - i), 1) / 252.0

        sigma_impl = max(row["implied_vol_proxy"], 1e-3)
        price, delta, gamma, vega, _theta = bs_straddle(row["close"], K, T, r, sigma_impl)

        scale = REGIME_SIZE_SCALE.get(row["regime"], 1.0)
        vol_edge = row["forecast_vol"] ** 2 - sigma_impl ** 2
        vol_var = frame["forecast_vol"].rolling(60).var().reindex(frame.index).loc[date]
        kelly = risk.kelly_fraction * (vol_edge / vol_var) if vol_var and vol_var > 0 else 0.0
        kelly = float(np.clip(kelly, -risk.kelly_clip, risk.kelly_clip))
        gamma_weight = np.tanh(1.2 * row["vrp_z"]) if not np.isnan(row["vrp_z"]) else 0.0

        notional = capital_mark * kelly
        raw_contracts = (notional / (price * 100)) * gamma_weight * scale if price > 0 else 0.0
        gamma_risk = abs(raw_contracts * gamma * 100)
        vega_risk = abs(raw_contracts * vega * 100)
        max_gamma, max_vega = risk.max_gamma_risk_pct * capital_mark, risk.max_vega_risk_pct * capital_mark
        if gamma_risk > max_gamma > 0:
            raw_contracts *= max_gamma / gamma_risk
        if vega_risk > max_vega > 0:
            raw_contracts *= max_vega / vega_risk
        new_contracts = int(np.clip(raw_contracts, -risk.max_contracts_per_leg, risk.max_contracts_per_leg))

        friction = 0.0
        if k > 0 and (rolled or new_contracts != contracts_held):
            # a roll closes the whole old position and opens the whole new
            # one (new strike/expiry); a same-roll resize just trades the
            # delta in contract count -- either way, charge cost on the
            # premium notional that actually changes hands.
            changed_qty = (abs(new_contracts) + abs(contracts_held)) if rolled else abs(new_contracts - contracts_held)
            friction_opt = k * option_cost_multiple * changed_qty * price * 100
            cash -= friction_opt
            friction += friction_opt
        contracts_held = new_contracts

        target_shares = -delta * contracts_held * 100
        trade_shares = target_shares - shares
        if k > 0:
            friction_eq = k * abs(trade_shares) * row["close"]
            cash -= friction_eq
            friction += friction_eq
        cash -= trade_shares * row["close"]
        shares = target_shares
        total_friction += friction

        option_value = contracts_held * price * 100
        capital = cash + shares * row["close"] + option_value

        trade_log.append({
            "date": date, "close": row["close"], "regime": row["regime"], "contracts": contracts_held,
            "shares": shares, "capital": capital, "delta": delta, "gamma": gamma, "vega": vega,
            "friction": friction,
        })

    trade_df = pd.DataFrame(trade_log).set_index("date")
    trade_df["return"] = trade_df["capital"].pct_change()
    excess = trade_df["return"].dropna()
    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0
    downside = excess[excess < 0]
    sortino = float(np.sqrt(252) * excess.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else 0.0
    win_rate = float((excess > 0).mean()) if len(excess) > 0 else 0.0
    cum = (1 + trade_df["return"].fillna(0)).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    cagr = float(cum.iloc[-1] ** (252 / len(cum)) - 1) if len(cum) > 0 else 0.0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(out_path / f"{symbol}_backtest.csv")

    return BacktestSummary(
        symbol=symbol, sharpe=sharpe, sortino=sortino, cagr=cagr, max_drawdown=max_dd,
        win_rate=win_rate, final_capital=float(trade_df["capital"].iloc[-1]) if len(trade_df) else initial_capital,
        n_days=len(trade_df), total_friction_paid=float(total_friction), cost_bps=cost_bps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="round-trip transaction cost in bps; 0 = v1's frictionless behavior")
    args = parser.parse_args()

    summary = run_backtest(args.symbol, years=args.years, initial_capital=args.capital, cost_bps=args.cost_bps)
    print(f"\n================ {summary.symbol} VOL-ARB BACKTEST ================")
    print(f"Sharpe Ratio  : {summary.sharpe:.2f}")
    print(f"Sortino Ratio : {summary.sortino:.2f}")
    print(f"CAGR          : {summary.cagr * 100:.2f}%")
    print(f"Max Drawdown  : {summary.max_drawdown * 100:.2f}%")
    print(f"Win Rate      : {summary.win_rate * 100:.1f}%")
    print(f"Final Capital : ${summary.final_capital:,.0f}")
    print(f"Trading Days  : {summary.n_days}")
    print(f"Cost Assumed  : {summary.cost_bps:.1f}bps  |  Friction Paid: ${summary.total_friction_paid:,.0f}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
