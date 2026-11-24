"""
TranAD input preparation

Keeps calibration changes out of TranAD and gives it a representative baseline:
  - Separate "condition" metrics (calibration constants, pedestals) from detector response
  - Find calibration epochs (stretches with identical condition values) so recalibrations
    are reported as such instead of being flagged as anomalies
  - Optionally pass condition metrics as changes relative to their epoch (glitch detection)
  - Drop near-constant response features and replace "no fit" sentinel values
  - Choose training / threshold-validation rows that span every calibration epoch
  - Measure drift: how far each input's recent median has moved from the training rows
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Metrics that reflect calibration inputs rather than detector response
CONDITION_PATTERNS = ("crvCalibConstants", "crvPedestals")

# Suffix of the 0/1 "measurement was valid" inputs added next to sentinel features
VALID_SUFFIX = "__valid"

# Values written when a fit/measurement was not possible (they carry sigma 0 or -1 in DQM)
SENTINELS = {
    "crvPEsMPV_": ("Centroid", 0.0),
    "crvCoincidencesClusters_": ("", -1.0),
}


def sentinel_value(feature: str):
    """The "no fit" value this feature writes, or None if it has none."""
    return next((v for p, (suffix, v) in SENTINELS.items()
                 if p in feature and feature.endswith(suffix)), None)


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
      - replace sentinel "no fit" values and missing values (NaN) with the median of valid
        values and add a 0/1 '<feature>__valid' indicator

    Returns (feature table, used feature names, dropped feature names).
    """
    out = pd.DataFrame(index=df.index)
    dropped = []
    for f in features:
        vals = pd.to_numeric(df[f], errors="coerce")
        sentinel = sentinel_value(f)
        # Missing values (metric not filled for that subrun) are treated like a sentinel
        if sentinel is not None or vals.isna().any():
            ok = vals.notna()
            if sentinel is not None:
                ok &= vals != sentinel
            if ok.sum() == 0:
                dropped.append(f)
                continue
            out[f] = vals.where(ok, vals[ok].median())
            out[f + VALID_SUFFIX] = ok.astype(float)
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


def row_stats(X: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std of the given rows (features constant there get std 1)."""
    mu = X[rows].mean(axis=0)
    sd = X[rows].std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def scale_on_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Standardise with mean/std of the given rows (features constant there get std 1)."""
    mu, sd = row_stats(X, rows)
    return (X - mu) / sd


def drift_min_points(window: int) -> int:
    """Fewest rows a drift window needs before it says anything."""
    return max(2, window // 2)


def drift_center(Z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-feature median of the standardised training rows: the drift reference."""
    return np.median(Z[rows], axis=0)


def rolling_drift(Z: np.ndarray, center: np.ndarray, window: int) -> np.ndarray:
    """
    Training drift of every row: per feature, the median of the standardised values over
    the `window` rows ending at that row, minus the training median, in units of the
    training std. Medians, so a single anomalous row does not count as drift.

    Rows with fewer than drift_min_points(window) rows behind them are NaN. The online
    evaluator computes the same thing from a point and its history.
    """
    med = pd.DataFrame(Z).rolling(window, min_periods=drift_min_points(window)).median()
    return med.to_numpy() - center


# ---------------------------------------------------------------------------------------
# Optional fixes for mixed datasets (2026-11-15): runs with no CRV data, and inputs scaled
# within detector-configuration periods instead of across all of them. Off by default.
# ---------------------------------------------------------------------------------------

EMPTY_PATTERN = "crvDigisPerChannelAndEvent_"


def no_crv_data_mask(df: pd.DataFrame, tol: float = 1e-9) -> np.ndarray:
    """Rows (subruns) with no CRV digis: every digis-per-channel Centroid is 0.

    Such rows put zeros into every response metric. Left in, they dominate the spread the
    inputs are scaled with and many drift alarms; they are better caught by this rule.
    """
    cols = [c for c in df.columns if EMPTY_PATTERN in c and c.endswith("Centroid")]
    if not cols:
        return np.zeros(len(df), dtype=bool)
    vals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    return (vals <= tol).all(axis=1).to_numpy()


def load_periods(path: str) -> pd.DataFrame:
    """Configuration periods from a CSV with columns first_run,last_run,label."""
    p = pd.read_csv(path)
    missing = {"first_run", "last_run", "label"} - set(p.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return p.sort_values("first_run").reset_index(drop=True)


def period_labels(runs: np.ndarray, periods: pd.DataFrame) -> np.ndarray:
    """Index of the period each run falls in, -1 if none."""
    runs = np.asarray(runs, dtype=int)
    labels = np.full(len(runs), -1)
    for i, p in periods.iterrows():
        labels[(runs >= p.first_run) & (runs <= p.last_run)] = i
    return labels


def robust_stats(X: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median and 1.4826 x MAD of the given rows (falls back to std, then 1, when 0)."""
    sub = X[rows]
    med = np.median(sub, axis=0)
    mad = 1.4826 * np.median(np.abs(sub - med), axis=0)
    sd = sub.std(axis=0)
    scale = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
    return med, scale


def scale_by_groups(X: np.ndarray, groups: np.ndarray, rows: np.ndarray,
                    min_rows: int = 10, floor_frac: float = 0.05,
                    clip: float = 20.0) -> np.ndarray:
    """Standardise each group (period) with robust stats of its own training rows.

    Groups with fewer than min_rows training rows, and rows in no group (-1), use the
    statistics of all training rows. A group's scale is floored at floor_frac of the global
    scale (an input that is almost constant within a period would otherwise reach z ~ 1e5
    and make training diverge), and the result is clipped to +-clip: a fault still sits at
    the clip, far beyond normal rows.
    """
    rows = np.asarray(rows)
    g_med, g_scale = robust_stats(X, rows)
    Z = (X - g_med) / g_scale
    for g in np.unique(groups):
        if g < 0:
            continue
        gr = rows[groups[rows] == g]
        if len(gr) < min_rows:
            continue
        med, scale = robust_stats(X, gr)
        scale = np.maximum(scale, floor_frac * g_scale)
        m = groups == g
        Z[m] = (X[m] - med) / scale
    return np.clip(Z, -clip, clip)


def clean_rows(Z: np.ndarray, rows: np.ndarray, zmax: float = 5.0,
               exclude_cols: np.ndarray = None) -> np.ndarray:
    """Training rows with no input beyond zmax (robust units): a proxy for known-good data."""
    rows = np.asarray(rows)
    A = np.abs(Z[rows])
    if exclude_cols is not None:
        A = A[:, ~exclude_cols]
    return rows[(A <= zmax).all(axis=1)]
