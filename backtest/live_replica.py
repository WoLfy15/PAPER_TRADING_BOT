"""
backtest/live_replica.py -- a daily-stepping backtest that CALLS THE LIVE DECISION CODE rather than re-implementing it:
  exits (take-profit / trailing / stop / near-expiry, incl. vertical spreads) .... agent.exits.evaluate_exits
  entry pacing (cooldowns, no first/last-15-min, after-stop wait) ................. agent.loop._entry_block_reason + us_time clock
  hedge (band, regime multiplier, minimum interval) ............................... strategy.hedging.decide_hedge
  risk (drawdown kill switch, daily/weekly breakers, vol-target band, Greek caps) .. strategy.portfolio.*
  Greeks ........................................................................... strategy.portfolio.build_portfolio_snapshot
Fills use real bid/ask (sell at bid, buy at ask, plus a slip share of the half-spread), commissions, and hedge-share costs.
KNOWN SIMPLIFICATIONS: one structure open at a time; one decision time per day (11:00 ET, quotes = that day's last
snapshot); expiry settles at intrinsic with no assignment/early-exercise modelling; calendar spreads are not executed.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agent.exits import evaluate_exits
from agent.loop import _entry_block_reason, _minutes_since
from agent.state import AgentState
from config import SETTINGS
from research.edge import allocation as A
from research.edge import strikes as S
from research.edge.meta import CostModel, adaptive_threshold
from strategy.hedging import decide_hedge
from strategy.portfolio import (build_portfolio_snapshot, check_drawdown_kill_switch, check_loss_circuit_breakers,
                                remaining_greek_headroom, vol_target_scale)
from us_time import set_clock, us_now

MULT, RF = 100, 0.04


@dataclass
class ReplicaConfig:
    equity0: float = 100_000.0
    target_dte: int = 7
    slip: float = 0.25
    comm: float = 0.65
    share_bps: float = 1.0
    spread_mult: float = 1.0          # 2.0 = stress test: every spread twice as wide
    kelly_frac: float = 0.25
    p_base: float = 0.55
    min_cost_mult: float = 1.5        # expected net vol points must beat this multiple of the cost estimate
    min_ratio: float = 0.05           # expected net P&L / CVaR per contract
    allow_long_vol: bool = False
    hedge: bool = True
    cost: CostModel = field(default_factory=CostModel)


def occ(exp: dt.date, cp: str, k: float) -> str:
    return f"SPY{exp:%y%m%d}{cp}{int(round(k * 1000)):08d}"


def index_chain(chain: pd.DataFrame) -> dict:
    """date -> that day's LAST snapshot (the one decision time per day)."""
    day = chain["ts"].dt.normalize()
    c = chain[chain["ts"] == chain.groupby(day)["ts"].transform("max")]
    return {d: g.reset_index(drop=True) for d, g in c.groupby(day[c.index])}


def _qmap(g: pd.DataFrame, m: float = 1.0) -> dict:
    mid, half = (g["bid"] + g["ask"]) / 2, (g["ask"] - g["bid"]) / 2 * m
    bid, ask = (mid - half).clip(lower=0), mid + half
    return {(e.date(), float(k), c): (float(b), float(a)) for e, k, c, b, a in zip(g["expiration"], g["strike"], g["cp"], bid, ask)}


def _fill(bq: tuple, buy: bool, slip: float) -> float:
    b, a = bq
    h = (a - b) / 2
    return a + slip * h if buy else max(b - slip * h, 0.0)


@dataclass
class _Pos:
    structure: str
    n: int
    expiry: dt.date
    legs: list
    entry_date: pd.Timestamp
    max_loss: float
    meta: dict


def _raw(pos: "_Pos | None", sign_n: bool = True) -> list[dict]:
    if pos is None:
        return []
    return [{"symbol": occ(pos.expiry, l["cp"], l["k"]), "asset_class": "us_option", "qty": float(l["q"] * pos.n), "side": "long" if l["q"] > 0 else "short",
             "avg_entry_price": l["entry"], "current_price": max(l["cur"], 0.01), "market_value": l["q"] * pos.n * l["cur"] * MULT} for l in pos.legs]


def run_replica(prices: pd.Series, snaps: dict, sig: pd.DataFrame, cfg: ReplicaConfig | None = None, settings=None,
                start=None, end=None) -> dict:
    cfg = cfg or ReplicaConfig()
    settings = settings or dataclasses.replace(SETTINGS, symbol="SPY", dry_run=True)
    risk = settings.risk
    ret = np.log(prices / prices.shift(1)).dropna()
    regimes = sig["regime"].reindex(ret.index)
    dates = [d for d in sig.index if d in snaps and d in prices.index and d in ret.index and (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    state = AgentState(); state.equity_high_water_mark = cfg.equity0
    cash, shares, pos = cfg.equity0, 0.0, None
    eq_dates, eq_vals, trades, diag = [], [], [], Counter()
    max_spread_pct = getattr(settings, "max_bid_ask_spread_pct", getattr(risk, "max_bid_ask_spread_pct", 0.10))

    def close(p: _Pos, d, reason: str, qm_fill: dict, spot: float) -> None:
        nonlocal cash, pos
        pnl_comm, settle = 0.0, d.date() >= p.expiry
        total = 0.0
        for l in p.legs:
            if settle:
                px = max(spot - l["k"], 0.0) if l["cp"] == "C" else max(l["k"] - spot, 0.0)
            else:
                bq = qm_fill.get((p.expiry, l["k"], l["cp"]))
                px = _fill(bq, buy=l["q"] < 0, slip=cfg.slip) if bq else max(l["cur"] * (1.05 if l["q"] < 0 else 0.95), 0.0)
                pnl_comm += cfg.comm * p.n
            cash += l["q"] * px * MULT * p.n - (0 if settle else cfg.comm * p.n)
            total += l["q"] * (px - l["entry"]) * MULT * p.n
        total -= p.meta["entry_comm"] + pnl_comm
        trades.append({**p.meta, "entry_date": p.entry_date, "exit_date": d, "structure": p.structure, "contracts": p.n, "max_loss": p.max_loss,
                       "pnl": total, "reason": "expired" if settle else reason})
        state.last_exit_utc["SPY"], state.last_exit_reason["SPY"] = us_now().isoformat(), reason
        pos = None

    for i, d in enumerate(dates):
        set_clock(dt.datetime(d.year, d.month, d.day, 11, 0))
        spot, g, today = float(prices[d]), snaps[d], d.date()
        qm, qf = _qmap(g), _qmap(g, cfg.spread_mult)
        if pos is not None:
            for l in pos.legs:
                q = qm.get((pos.expiry, l["k"], l["cp"]))
                if q:
                    l["cur"] = (q[0] + q[1]) / 2
        equity = cash + shares * spot + (sum(l["q"] * pos.n * l["cur"] * MULT for l in pos.legs) if pos else 0.0)
        state.equity_high_water_mark = max(state.equity_high_water_mark, equity)
        prev = eq_vals[-1] if eq_vals else equity
        # ---- exits: the LIVE evaluator, then close every leg of the structure
        if pos is not None:
            if today >= pos.expiry:
                close(pos, d, "expired", qf, spot)
            else:
                dec = evaluate_exits(_raw(pos), risk, "SPY", today=today, peaks=state.peak_pnl)
                if dec:
                    close(pos, d, dec[0].reason, qf, spot)
                    equity = cash + shares * spot
        if pos is None:
            state.peak_pnl.clear()
        # ---- hedge: the LIVE band logic
        if cfg.hedge and (pos is not None or abs(shares) >= 1):
            sh = [{"symbol": "SPY", "asset_class": "us_equity", "qty": shares, "avg_entry_price": spot, "current_price": spot, "market_value": shares * spot}] if shares else []
            sn = build_portfolio_snapshot({"equity": equity, "cash": cash}, _raw(pos) + sh, "SPY", spot, RF)
            hd = decide_hedge(sn.net_delta_shares, sn.net_gamma_shares, shares, "Crash" if sig["regime"].get(d) == "crisis" else "Trend", spot,
                              max(((pos.expiry - today).days if pos else 1), 1) / 365.0, RF, settings.hedge, hedge_ratio_override=1.0)
            if hd.should_trade:
                urgent = abs(hd.trade_shares) > 3 * hd.band_half_width or abs(sn.net_delta_shares) < 1
                if _minutes_since(state.last_hedge_utc) >= settings.hedge.rehedge_min_interval_min or urgent:
                    tr = float(round(hd.trade_shares))
                    if tr:
                        cash -= tr * spot + abs(tr) * spot * cfg.share_bps / 1e4
                        shares += tr
                        state.last_hedge_utc = us_now().isoformat()
        equity = cash + shares * spot + (sum(l["q"] * pos.n * l["cur"] * MULT for l in pos.legs) if pos else 0.0)
        eq_dates.append(d); eq_vals.append(equity)
        # ---- risk gates (live functions)
        if pos is not None:
            diag["holding"] += 1
            continue
        if i == len(dates) - 1:
            continue                                  # no future day left to manage a new position
        killed, dd = check_drawdown_kill_switch(equity, state.equity_high_water_mark, risk)
        daily = equity / eq_vals[-2] - 1 if len(eq_vals) > 1 else None
        weekly = equity / eq_vals[-6] - 1 if len(eq_vals) > 5 else None
        tripped, _ = check_loss_circuit_breakers(daily, weekly, risk)
        if killed or tripped:
            diag["kill_switch" if killed else "loss_breaker"] += 1
            continue
        block = _entry_block_reason(state, settings)
        if block:
            diag["cooldown"] += 1
            continue
        # ---- signal gates
        s = sig.loc[d]
        if not all(np.isfinite([s.get("mu", np.nan), s.get("p", np.nan), s.get("exp_net", np.nan), s.get("iv", np.nan)])):
            diag["no_signal"] += 1
            continue
        if s["mu"] <= 0 and not cfg.allow_long_vol:
            diag["mu_nonpositive"] += 1
            continue
        thr = adaptive_threshold(str(s["regime"]), s.get("z_jump", 0.0), s["spread_pct"], False, s["iv"], cfg.p_base)
        if s["p"] < thr:
            diag["meta_p_below_threshold"] += 1
            continue
        if s["exp_net"] < cfg.min_cost_mult * dataclasses.replace(cfg.cost, spread_mult=cfg.spread_mult).cost_vp(s["iv"], s["spread_pct"]):
            diag["cost_gate"] += 1
            continue
        zs = {k: s.get(k, 0.0) for k in ("z_vrp", "z_carry", "z_skew", "z_jump")}
        ws = {k: s.get(f"w_{k}", 0.0) for k in zs}
        structure, ok, why = S.select_structure(zs, ws, s["mu"], cfg.allow_long_vol)
        if not ok:
            diag[f"structure:{structure}"] += 1
            continue
        level = structure == "iron_condor"
        eligible = ["iron_condor", "put_vertical", "call_vertical"] if level else [structure]
        dtes = (g["expiration"].dt.normalize() - d).dt.days
        opts = sorted({int(x) for x in dtes if abs(x - cfg.target_dte) <= 2}, key=lambda x: abs(x - cfg.target_dte))
        if not opts:
            diag["no_expiry"] += 1
            continue
        qd = g[dtes == opts[0]]
        qd_f = qd.assign(bid=[qf[(e.date(), float(k), c)][0] for e, k, c in zip(qd["expiration"], qd["strike"], qd["cp"])],
                         ask=[qf[(e.date(), float(k), c)][1] for e, k, c in zip(qd["expiration"], qd["strike"], qd["cp"])])
        scen = S.forward_scenarios(ret, regimes, cfg.target_dte, ret.index.get_loc(d))
        if len(scen) < 50:
            diag["no_scenarios"] += 1
            continue
        f = A.fractional_kelly(s["exp_net"], s.get("sigma_net", np.nan), s.get("sigma_err", np.nan), cfg.kelly_frac, risk.max_per_trade_loss_pct)
        vt = min(1.0, vol_target_scale(float(np.std(np.diff(np.log(eq_vals[-21:]))) * np.sqrt(252)) if len(eq_vals) >= 21 else None, risk))
        scale = vt * A.drawdown_scale(dd, risk.max_drawdown_pct) * A.liquidity_scale(s["spread_pct"], max_spread_pct) * A.alpha_decay_scale(s.get("ic", np.nan))
        budget = equity * f * scale
        opps, cands = [], {}
        for st in eligible:
            c = S.optimize_structure(st, qd_f, spot, scen, slip=cfg.slip, comm_per_contract=cfg.comm)
            if c and c["exp_pnl"] > 0 and c["max_loss"] > 0:
                n0 = int(min(budget // c["max_loss"], risk.max_structure_cvar_pct * equity // c["cvar"]))
                cands[st] = c
                opps.append({"name": st, "underlying": "SPY", "exp_net": c["exp_pnl"], "cvar": c["cvar"], "contracts": n0})
        alloc = A.rank_and_allocate(opps, risk.max_structure_cvar_pct * equity, cfg.min_ratio, risk.max_structure_cvar_pct * equity)
        if not alloc:
            diag["no_candidate_or_ratio_or_size"] += 1
            continue
        pick, c = alloc[0], cands[alloc[0]["name"]]
        expiry = qd["expiration"].iloc[0].date()
        legs = [{"cp": cp, "k": float(k), "q": q, "entry": _fill(qf[(expiry, float(k), cp)], q > 0, cfg.slip), "cur": 0.0} for cp, k, q in c["legs"]]
        for l in legs:
            l["cur"] = sum(qm[(expiry, l["k"], l["cp"])]) / 2
        n, sh = pick["contracts"], ([{"symbol": "SPY", "asset_class": "us_equity", "qty": shares, "current_price": spot, "avg_entry_price": spot, "market_value": shares * spot}] if shares else [])
        while n >= 1:      # existing + proposed Greeks against the live caps, shrinking until the NET book fits
            trial = _Pos(c["structure"], n, expiry, legs, d, 0.0, {})
            hr = remaining_greek_headroom(build_portfolio_snapshot({"equity": equity, "cash": cash}, _raw(trial) + sh, "SPY", spot, RF), risk)
            if min(hr.values()) >= 0:
                break
            n -= 1
        if n < 1:
            diag["greek_cap"] += 1
            continue
        comm = cfg.comm * n * len(legs)
        for l in legs:
            cash -= l["q"] * l["entry"] * MULT * n
        cash -= comm
        pos = _Pos(c["structure"], n, expiry, legs, d, c["max_loss"] * n,
                   {"entry_comm": comm, "regime": s["regime"], "mu": s["mu"], "p": s["p"], "exp_net_vp": s["exp_net"], "why": why, "risk_scale": scale,
                    "spot_entry": spot, "trend63": float(prices[d] / prices.iloc[max(0, prices.index.get_loc(d) - 63)] - 1)})
        state.last_entry_utc["SPY"] = us_now().isoformat()
        diag["entered"] += 1
    if pos is not None and dates:
        d = dates[-1]
        close(pos, d, "end_of_test", _qmap(snaps[d], cfg.spread_mult), float(prices[d]))
        eq_vals[-1] = cash + shares * float(prices[d])
    set_clock(None)
    eq = pd.Series(eq_vals, index=eq_dates, dtype=float)
    return {"trades": pd.DataFrame(trades), "equity": eq, "diag": dict(diag), "hedge_shares_end": shares, "cash_end": cash}


def summarize(res: dict, equity0: float = 100_000.0) -> dict:
    t, eq = res["trades"], res["equity"]
    if t.empty or len(eq) < 3:
        return {"n_trades": 0}
    r = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    pnl, risk = t["pnl"], t["max_loss"]
    top = max(1, int(np.ceil(0.05 * len(t))))
    tot_profit = pnl[pnl > 0].sum()
    k = max(1, int(0.05 * len(pnl)))
    return {"n_trades": int(len(t)), "win_rate": float((pnl > 0).mean()), "expectancy": float(pnl.mean()), "expectancy_per_risk": float((pnl / risk).mean()),
            "profit_factor": float(tot_profit / max(-pnl[pnl < 0].sum(), 1e-9)), "total_return": float(eq.iloc[-1] / equity0 - 1),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0, "max_drawdown": float(dd),
            "trade_cvar5": float(-np.sort(pnl.to_numpy())[:k].mean()), "avg_risk_pct_equity": float((risk / equity0).mean()),
            "top5pct_profit_share": float(np.sort(pnl.to_numpy())[::-1][:top][np.sort(pnl.to_numpy())[::-1][:top] > 0].sum() / tot_profit) if tot_profit > 0 else float("nan"),
            "exit_reasons": t["reason"].value_counts().to_dict()}
