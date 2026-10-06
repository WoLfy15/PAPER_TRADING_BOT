"""
research/regime.py -- 4-state Gaussian HMM regime detection.

Ported from Jenil's OPTION-HEDGING-MODEL (IITfinal.ipynb), generalized
from a hard-coded NVDA pipeline to any symbol's OHLC DataFrame. Same
feature set and state count as the original: [log_return, realized_vol,
vol_of_vol] -> 4 states, labeled by their mean realized vol / return
(Range, Trend, Vol_Expansion, Crash) rather than assuming the fit
happens to land in the original notebook's arbitrary label order --
GaussianHMM's state indices are not stable across refits, so remapping
by empirical characteristics is what makes the labels mean the same
thing every time this is run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

REGIME_LABELS = ("Range", "Trend", "Vol_Expansion", "Crash")

# Same table as the original notebook -- see README for what each
# multiplier does to hedge frequency.
HEDGE_MULTIPLIER = {"Range": 1.2, "Trend": 0.6, "Vol_Expansion": 1.0, "Crash": 1.5}


@dataclass
class RegimeResult:
    labels: pd.Series          # one label per row of the input frame
    probabilities: pd.DataFrame  # per-state posterior, columns = REGIME_LABELS
    current_label: str
    current_confidence: float
    model: GaussianHMM


def _engineer_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": close})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    df["vol_of_vol"] = df["realized_vol"].rolling(21).std()
    return df.dropna()


def _label_states_by_characteristics(model: GaussianHMM, feature_cols: list[str]) -> dict[int, str]:
    """GaussianHMM state 0/1/2/3 are arbitrary post-fit -- map them to
    stable semantic labels by ranking each state's mean [return, vol]:
    highest vol + negative mean return -> Crash; highest vol otherwise ->
    Vol_Expansion; lowest vol -> Range; the remainder -> Trend."""
    ret_idx, vol_idx = feature_cols.index("log_return"), feature_cols.index("realized_vol")
    means = model.means_  # (n_states, n_features), in the *scaled* feature space
    order_by_vol = np.argsort(means[:, vol_idx])  # ascending vol
    labels: dict[int, str] = {}
    labels[int(order_by_vol[0])] = "Range"
    remaining = list(order_by_vol[1:-1])
    # of the two middle-vol states, the one with the more negative mean
    # return is "Trend" only if it's actually trending down/up hard;
    # otherwise both middling states are just "Trend" (directional, not
    # crisis) -- Crash is reserved for the single highest-vol state.
    for idx in remaining:
        labels[int(idx)] = "Trend"
    top = int(order_by_vol[-1])
    labels[top] = "Crash" if means[top, ret_idx] < 0 else "Vol_Expansion"
    return labels


def detect_regime(close: pd.Series, n_states: int = 4, n_iter: int = 1000, random_state: int = 42) -> RegimeResult:
    """close: a daily-close price Series, DatetimeIndex, ascending order.
    Needs at least ~150 observations (after the 21-day rolling warmup) for
    the HMM fit to be meaningful -- shorter histories will fit but the
    regime labels won't be trustworthy."""
    feat = _engineer_features(close)
    feature_cols = ["log_return", "realized_vol", "vol_of_vol"]
    if len(feat) < 100:
        raise ValueError(f"need >=100 rows after feature warmup, got {len(feat)}")

    X = StandardScaler().fit_transform(feat[feature_cols].values)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=random_state)
    model.fit(X)

    state_labels = _label_states_by_characteristics(model, feature_cols)
    hidden_states = model.predict(X)
    posterior = model.predict_proba(X)

    labels = pd.Series([state_labels[s] for s in hidden_states], index=feat.index, name="regime")
    proba_df = pd.DataFrame(posterior, index=feat.index, columns=[state_labels[i] for i in range(n_states)])
    # a symbol's HMM can (rarely) collapse two states into the same label
    # if the fit is degenerate; sum duplicate-labeled columns so the
    # caller always gets exactly REGIME_LABELS columns.
    proba_df = proba_df.T.groupby(level=0).sum().T.reindex(columns=list(REGIME_LABELS), fill_value=0.0)

    current_label = labels.iloc[-1]
    current_confidence = float(proba_df.iloc[-1][current_label])

    return RegimeResult(
        labels=labels, probabilities=proba_df,
        current_label=current_label, current_confidence=current_confidence, model=model,
    )
