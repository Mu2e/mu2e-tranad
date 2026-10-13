"""
TranAD input preparation

Keeps calibration changes out of TranAD and gives it a representative baseline:
  - Separate "condition" metrics (calibration constants, pedestals) from detector response
  - Find calibration epochs (stretches with identical condition values) so recalibrations
    are reported as such instead of being flagged as anomalies
  - Optionally pass condition metrics as changes relative to their epoch (glitch detection)
  - Drop near-constant response features and replace "no fit" sentinel values
  - Choose training / threshold-validation rows that span every calibration epoch
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Metrics that reflect calibration inputs rather than detector response
CONDITION_PATTERNS = ("crvCalibConstants", "crvPedestals")

# Values written when a fit/measurement was not possible (they carry sigma 0 or -1 in DQM)
SENTINELS = {
    "crvPEsMPV_": ("Centroid", 0.0),
    "crvCoincidencesClusters_": ("", -1.0),
}


def split_condition_features(features: List[str]) -> Tuple[List[str], List[str]]:
    """Return (condition_features, response_features)."""
    cond = [f for f in features if any(p in f for p in CONDITION_PATTERNS)]
    resp = [f for f in features if f not in cond]
    return cond, resp


def detect_calibration_epochs(df: pd.DataFrame, condition_cols: List[str],
                              min_rows: int = 20) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Label each row with a calibration epoch: a run of consecutive rows whose condition
    Centroids are identical. Stretches shorter than min_rows (e.g. a glitch where the
    constants read 0) are labelled -1 ("transient").

    Returns (labels, summary DataFrame with one row per epoch).
    """
    cols = [c for c in condition_cols if c.endswith("Centroid")] or condition_cols
    if not cols:
        return np.zeros(len(df), dtype=int), pd.DataFrame()

    key = df[cols].round(4).astype(str).agg("|".join, axis=1)
    block = (key != key.shift()).cumsum().to_numpy()

    labels = np.full(len(df), -1)
    long_blocks = [b for b in pd.unique(block) if (block == b).sum() >= min_rows]
    # Blocks with the same condition values (e.g. before and after a glitch) share an epoch
    epoch_of_key: Dict[str, int] = {}
    for b in long_blocks:
        k = key[block == b].iloc[0]
        epoch_of_key.setdefault(k, len(epoch_of_key))
        labels[block == b] = epoch_of_key[k]

    time = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df else None
    rows = []
    for e in sorted(set(labels)):
        m = labels == e
        r = {"epoch": e, "rows": int(m.sum()),
             "first_run": int(df.loc[m, "run"].min()), "last_run": int(df.loc[m, "run"].max())}
        if time is not None:
            r["start"], r["end"] = time[m].min(), time[m].max()
        for c in cols:
            r[c] = df.loc[m, c].iloc[0] if e >= 0 else np.nan
        rows.append(r)
    return labels, pd.DataFrame(rows)


def prepare_response_features(df: pd.DataFrame, features: List[str],
                              min_distinct: int = 6) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the TranAD input table from response features:
      - drop features with fewer than min_distinct distinct values (histogram edges that
        only move on glitches dominate the reconstruction error otherwise)
      - replace sentinel "no fit" values with the median of valid values and add a
        0/1 '<feature>__valid' indicator

    Returns (feature table, used feature names, dropped feature names).
    """
    out = pd.DataFrame(index=df.index)
    dropped = []
    for f in features:
        vals = pd.to_numeric(df[f], errors="coerce")
        sentinel = next((v for p, (suffix, v) in SENTINELS.items()
                         if p in f and f.endswith(suffix)), None)
        if sentinel is not None:
            ok = vals != sentinel
            if ok.sum() == 0:
                dropped.append(f)
                continue
            out[f] = vals.where(ok, vals[ok].median())
            out[f + "__valid"] = ok.astype(float)
        elif vals.nunique() < min_distinct:
            dropped.append(f)
        else:
            out[f] = vals
    return out, list(out.columns), dropped


def prepare_condition_changes(df: pd.DataFrame, condition_cols: List[str],
                              epochs: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    """
    Express condition metrics as changes relative to their calibration epoch, so TranAD
    sees glitches (e.g. constants reading 0) but not legitimate recalibrations:
      '<feature>__change' = (value - epoch reference) / |epoch reference|
    The reference is the epoch's median; transient rows (epoch -1) use the most recent
    preceding epoch (the next one if none precedes them). Rows inside an epoch get 0.

    Returns (feature table, feature names).
    """
    out = pd.DataFrame(index=df.index)
    if not condition_cols or (epochs < 0).all():
        return out, []
    ref_epoch = pd.Series(np.where(epochs >= 0, epochs, np.nan), index=df.index).ffill().bfill()
    for f in condition_cols:
        vals = pd.to_numeric(df[f], errors="coerce")
        ref = ref_epoch.map(vals[epochs >= 0].groupby(epochs[epochs >= 0]).median())
        scale = ref.abs().where(ref != 0, 1.0)
        out[f + "__change"] = ((vals - ref) / scale).fillna(0.0)
    return out, list(out.columns)


def select_rows(epochs: np.ndarray, mode: str = "stratified", train_ratio: float = 0.3,
                val_ratio: float = 0.1, n_window: int = 4,
                seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick row indices for TranAD training and for threshold validation (disjoint).

    mode:
      stratified    - train_ratio / val_ratio of every calibration epoch (transients excluded)
      random        - uniform random rows over the whole dataset
      chronological - first train_ratio of rows for training, the next val_ratio for validation
    """
    n = len(epochs)
    rng = np.random.default_rng(seed)
    eligible = np.arange(n_window, n)

    if mode == "chronological":
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return np.arange(n_window, n_train), np.arange(n_train, n_train + n_val)

    if mode == "random":
        pools = [eligible]
    elif mode == "stratified":
        pools = [eligible[epochs[eligible] == e] for e in np.unique(epochs) if e >= 0]
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    train, val = [], []
    for pool in pools:
        perm = rng.permutation(pool)
        n_tr = max(1, int(round(len(pool) * train_ratio)))
        n_va = max(1, int(round(len(pool) * val_ratio)))
        train.append(perm[:n_tr])
        val.append(perm[n_tr:n_tr + n_va])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def scale_on_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Standardise with mean/std of the given rows (features constant there get std 1)."""
    mu = X[rows].mean(axis=0)
    sd = X[rows].std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd
