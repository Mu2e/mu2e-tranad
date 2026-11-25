"""
Model bundle: what a trained pipeline run leaves behind for online evaluation

Training is offline (run_pipeline.py --save-model); scoring is one point at a time
inside the DQM driver, by the `tranad` evaluator in Mu2e/DQM
(python/dqm/evaluate/tranad.py). The bundle carries everything the second needs from the
first, frozen at training time:
  - PCA     - feature list, StandardScaler mean/scale, components, error threshold
  - TranAD  - input columns, sentinel fill values, scaling, weights, threshold
  - drift   - training medians of the reference inputs (TranAD's, else PCA's), window, sigma

Arrays are stored as numpy and the TranAD weights as a numpy state_dict, so loading
needs neither sklearn pickles nor a matching torch build.

The bundle's keys are an interface to Mu2e/DQM: renaming or reshaping one needs the
matching change in dqm/evaluate/tranad.py, released together with this package.
"""

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import tranad_prep

FORMAT = 1
VALID_SUFFIX = tranad_prep.VALID_SUFFIX

# The model used when none is named: the CRV model trained on crvreco,kpp_crv,file,006-001
# (see models/README.md). MU2E_TRANAD_MODEL overrides it.
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "crv_tranad_006-001.pkl"


def default_model_path() -> str:
    """The bundle to score with when no model is given: $MU2E_TRANAD_MODEL, else DEFAULT_MODEL."""
    return os.environ.get("MU2E_TRANAD_MODEL") or str(DEFAULT_MODEL)


def inputs_from_frame(df: pd.DataFrame, inputs: List[str], fill: Dict[str, float],
                      mean: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    The model input matrix for every row of df, built the way the model was trained:
    a "no fit" feature (one in `fill`) has its sentinel or missing values replaced by the
    training median, and its '<feature>__valid' input is 0 there; a missing value in any
    other input is set to that input's training mean (0 once scaled). Returns the matrix
    and, per metric, how many missing values were set to the mean.
    """
    base = [f[:-len(VALID_SUFFIX)] if f.endswith(VALID_SUFFIX) else f for f in inputs]
    absent = sorted(set(base) - set(df.columns))
    if absent:
        raise ValueError(f"the model needs {len(absent)} metric(s) this data does not have "
                         f"(trained on another source?): {absent}")
    X = np.empty((len(df), len(inputs)))
    filled = {}
    for j, (f, b) in enumerate(zip(inputs, base)):
        vals = pd.to_numeric(df[b], errors="coerce")
        if b in fill:
            ok = vals.notna()
            sentinel = tranad_prep.sentinel_value(b)
            if sentinel is not None:
                ok &= vals != sentinel
            X[:, j] = ok.astype(float) if f.endswith(VALID_SUFFIX) else vals.where(ok, fill[b])
        else:
            n = int(vals.isna().sum())
            if n:
                filled[b] = n
            X[:, j] = vals.fillna(float(mean[j]))
    return X, filled


def save_bundle(path: str,
                df: pd.DataFrame,
                active_features: List[str],
                scaler,
                model_results: dict,
                tranad_features: str,
                tranad_cols: Optional[List[str]],
                X_tranad_raw: Optional[np.ndarray],
                train_idx: np.ndarray,
                source: Optional[str] = None,
                drift: Optional[dict] = None) -> dict:
    """
    Write the bundle for this training run and return it. `drift` holds 'center' (one
    training median per reference input), 'window' and 'sigma', as in run_pipeline.
    """
    if tranad_features == "changes":
        # Condition changes are relative to a calibration epoch found over the whole
        # dataset; a single online point has no such reference yet.
        raise ValueError("--save-model supports --tranad-features response or all, not changes")

    pca = model_results["pca_detector"]
    bundle = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "rows": int(len(df)),
        "runs": [int(df["run"].min()), int(df["run"].max())] if "run" in df else None,
        "pca": {
            "features": list(active_features),
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "components": pca.pca_.components_,
            "center": pca.pca_.mean_,
            "threshold": float(pca.reconstruction_error_threshold_),
        },
        "tranad": None,
        "drift": None,
    }

    tranad = model_results.get("tranad_detector")
    if tranad is not None:
        if X_tranad_raw is None:  # 'all': the same StandardScaler matrix as PCA
            inputs, mu, sd, fill = list(active_features), scaler.mean_, scaler.scale_, {}
        else:
            inputs = list(tranad_cols)
            mu, sd = tranad_prep.row_stats(X_tranad_raw, train_idx)
            fill = {}
            for f in inputs:
                if f + VALID_SUFFIX in inputs:
                    vals = pd.to_numeric(df[f], errors="coerce")
                    fill[f] = float(vals[vals != tranad_prep.sentinel_value(f)].median())
        bundle["tranad"] = {
            "inputs": inputs,
            "mean": np.asarray(mu, dtype=float),
            "std": np.asarray(sd, dtype=float),
            "fill": fill,
            "n_window": int(tranad.n_window),
            "arch": {"d_model": tranad.d_model, "nhead": tranad.nhead,
                     "d_ff": tranad.d_ff, "num_layers": tranad.num_layers},
            "state_dict": {k: v.detach().cpu().numpy()
                           for k, v in tranad.model.state_dict().items()},
            "threshold": float(tranad.threshold),
        }

    if drift is not None:
        bundle["drift"] = {"center": np.asarray(drift["center"], dtype=float),
                           "window": int(drift["window"]), "sigma": float(drift["sigma"])}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✓ Model bundle saved: {path}"
          f" (PCA {len(active_features)} features"
          f"{', TranAD %d inputs' % len(bundle['tranad']['inputs']) if bundle['tranad'] else ', no TranAD'})")
    return bundle


def save_weights(path: str, model_results: dict, tranad_cols, settings: dict) -> Optional[dict]:
    """Write the trained TranAD weights alone, for runs whose options save_bundle can't
    represent (per-period scaling, --drop-empty, --train-clean, 'changes' features).
    The file is a record of the trained model, not something the online evaluator loads."""
    tranad = model_results.get("tranad_detector")
    if tranad is None:
        return None
    weights = {
        "kind": "tranad_weights",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": dict(settings),
        "inputs": list(tranad_cols) if tranad_cols is not None else None,
        "n_window": int(tranad.n_window),
        "arch": {"d_model": tranad.d_model, "nhead": tranad.nhead,
                 "d_ff": tranad.d_ff, "num_layers": tranad.num_layers},
        "state_dict": {k: v.detach().cpu().numpy()
                       for k, v in tranad.model.state_dict().items()},
        "threshold": float(tranad.threshold),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(weights, f)
    print(f"✓ TranAD weights saved: {path} (not loadable by the online evaluator: "
          f"its options need per-period scaling or row filtering)")
    return weights


def load_bundle(path: str) -> dict:
    """Read a bundle written by save_bundle."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("format") != FORMAT:
        raise ValueError(f"{path}: bundle format {bundle.get('format')!r}, expected {FORMAT}")
    return bundle
