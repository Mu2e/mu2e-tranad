"""
Model bundle: what a trained pipeline run leaves behind for online evaluation

Training is offline (run_pipeline.py --save-model); scoring is one point at a time
inside the DQM driver (evaluator.py). The bundle carries everything the second needs
from the first, frozen at training time:
  - PCA     - feature list, StandardScaler mean/scale, components, error threshold
  - TranAD  - input columns, sentinel fill values, scaling, weights, threshold

Arrays are stored as numpy and the TranAD weights as a numpy state_dict, so loading
needs neither sklearn pickles nor a matching torch build.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import tranad_prep

FORMAT = 1
VALID_SUFFIX = "__valid"


def save_bundle(path: str,
                df: pd.DataFrame,
                active_features: List[str],
                scaler,
                model_results: dict,
                tranad_features: str,
                tranad_cols: Optional[List[str]],
                X_tranad_raw: Optional[np.ndarray],
                train_idx: np.ndarray,
                source: Optional[str] = None) -> dict:
    """Write the bundle for this training run and return it."""
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✓ Model bundle saved: {path}"
          f" (PCA {len(active_features)} features"
          f"{', TranAD %d inputs' % len(bundle['tranad']['inputs']) if bundle['tranad'] else ', no TranAD'})")
    return bundle


def load_bundle(path: str) -> dict:
    """Read a bundle written by save_bundle."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("format") != FORMAT:
        raise ValueError(f"{path}: bundle format {bundle.get('format')!r}, expected {FORMAT}")
    return bundle
