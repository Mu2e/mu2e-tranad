"""
The `crv_ml` evaluator: score each new DQM point with the trained pipeline models

Plugs dqm_pipeline into the Mu2e DQM evaluation framework (Mu2e/DQM,
python/dqm/evaluate). Training stays offline in run_pipeline.py --save-model; this
class only loads that bundle and judges one point at a time, so it is cheap enough for
a gpvm cron job (CPU, a few ms per point; the cost is importing torch).

    algo:   crv_ml
    config: crv-tranad-v1        # bump on every retrain: it is part of alarm identity
    sid:    115
    min_time: 2026-11-04         # go-live date: don't alarm on the whole back history
    commit: false

    pars:
      model: /exp/mu2e/app/users/.../crv_ml_v1.pkl   # from --save-model
      detectors: [tranad]        # and/or pca
      level: warn                # severity of a finding
      threshold_scale: 1.0       # multiply the trained thresholds (>1 = fewer findings)
      top_causes: 5              # metrics named as causes, by reconstruction error

Subjects:

    tranad_outlier   the point is poorly reconstructed by TranAD from the ones before it
    pca_outlier      the point is far from the PCA subspace of the training data

Run it with python -m dqm_pipeline.dqm_driver <config.yaml>, which is dqm-driver with
this algorithm registered.
"""

import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dqm.errors import DqmConfigError
from dqm.evaluate.base import LEVELS, Evaluator, Finding, Point

from .bundle import VALID_SUFFIX, load_bundle
from .tranad_prep import sentinel_value

LOGGER = logging.getLogger("dqm.evaluate.crv_ml")

__all__ = ["MLEvaluator"]

DETECTORS = ("tranad", "pca")


class MLEvaluator(Evaluator):
    """Flag points the trained TranAD / PCA models find anomalous."""

    algo = "crv_ml"
    subjects = ("tranad_outlier", "pca_outlier")
    lookback = 4  # replaced by the bundle's TranAD window in configure()

    def configure(self) -> None:
        path = self.pars.get("model")
        if not path:
            raise DqmConfigError("crv_ml: pars.model must name a bundle from --save-model")
        try:
            self._bundle = load_bundle(str(path))
        except Exception as exc:
            raise DqmConfigError(f"crv_ml: cannot load model {path}: {exc}") from exc

        detectors = self.pars.get("detectors", ["tranad"])
        if not isinstance(detectors, (list, tuple)) or not detectors \
                or any(d not in DETECTORS for d in detectors):
            raise DqmConfigError(f"crv_ml: detectors must be a non-empty list from {DETECTORS}")
        self._detectors = tuple(detectors)
        pca = self._bundle["pca"]
        if "pca" in self._detectors and len(pca["components"]) >= len(pca["features"]):
            # Every component kept: reconstruction is exact and the "error" is rounding
            raise DqmConfigError(
                f"crv_ml: the PCA model keeps all {len(pca['components'])} components, so its "
                f"reconstruction error is numerical noise; retrain with fewer components")

        level = self.pars.get("level", LEVELS["warn"])
        if isinstance(level, str):
            if level not in LEVELS:
                raise DqmConfigError(f"crv_ml: level must be one of {sorted(LEVELS)} or 0-3")
            level = LEVELS[level]
        if level not in range(4):
            raise DqmConfigError(f"crv_ml: level must be 0-3, got {level!r}")
        self._level = int(level)

        try:
            self._scale = float(self.pars.get("threshold_scale", 1.0))
            self._top = int(self.pars.get("top_causes", 5))
        except (TypeError, ValueError) as exc:
            raise DqmConfigError(f"crv_ml: {exc}") from exc
        if self._scale <= 0 or self._top < 1:
            raise DqmConfigError("crv_ml: threshold_scale must be > 0 and top_causes >= 1")

        self._model = None
        if "tranad" in self._detectors:
            self._model = self._load_tranad()
            self.lookback = self._bundle["tranad"]["n_window"]
        else:
            self.lookback = 0

        LOGGER.info("crv_ml: model %s (trained %s on %s, %s rows), detectors %s",
                    path, self._bundle["created"], self._bundle["source"],
                    self._bundle["rows"], ",".join(self._detectors))

    def _load_tranad(self):
        t = self._bundle["tranad"]
        if t is None:
            raise DqmConfigError("crv_ml: the bundle has no TranAD model; use detectors: [pca]")
        # Imported here so a PCA-only driver never needs torch or anldq
        from .tranad_model import TRANAD_ANL_AVAILABLE, TranAD, TranADConfig, torch
        if not TRANAD_ANL_AVAILABLE:
            raise DqmConfigError("crv_ml: TranAD needs torch and anldq "
                                 "(source dqm_pipeline/setup_env.sh)")
        model = TranAD(input_dims=len(t["inputs"]),
                       arch_config=TranADConfig(n_window=t["n_window"], **t["arch"]))
        model.load_state_dict({k: torch.from_numpy(v) for k, v in t["state_dict"].items()})
        model.eval()
        self._torch = torch
        return model

    # -- inputs -----------------------------------------------------------

    @staticmethod
    def _columns(point: Point) -> Dict[str, int]:
        """Pipeline column name -> vid. The pipeline names metrics groupx_subgroup_namex."""
        return {name.replace("/", "_"): vid for vid, name in point.names.items()}

    def _vector(self, point: Point, inputs: Sequence[str],
                fill: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[int], Optional[str]]:
        """
        The model input row for this point, the vid behind each input, and the first
        input that could not be filled (None if all were).
        """
        cols = self._columns(point)
        x, vids = [], []
        for f in inputs:
            base = f[:-len(VALID_SUFFIX)] if f.endswith(VALID_SUFFIX) else f
            vid = cols.get(base)
            m = point.get(vid) if vid is not None else None
            if m is None:
                return None, [], base
            if base in fill:
                # A "no fit" feature: a sentinel or a flagged measurement is invalid,
                # replaced by the training median and marked in the __valid input
                ok = m.ok and m.valuex != sentinel_value(base)
                v = float(ok) if f.endswith(VALID_SUFFIX) else (m.valuex if ok else fill[base])
            elif not m.ok:
                return None, [], base
            else:
                v = m.valuex
            x.append(v)
            vids.append(vid)
        return np.asarray(x, dtype=float), vids, None

    # -- detectors --------------------------------------------------------

    def _tranad(self, point: Point, history: Sequence[Point]):
        """Per-input squared errors, the vids behind them, and the threshold; or None."""
        t = self._bundle["tranad"]
        n = t["n_window"]
        if len(history) < n:
            LOGGER.debug("%s: %d point(s) of history, TranAD needs %d -- skipping",
                         point.label, len(history), n)
            return None
        rows = []
        for p in list(history[-n:]) + [point]:
            x, vids, missing = self._vector(p, t["inputs"], t["fill"])
            if x is None:
                LOGGER.info("%s: TranAD skipped, %s missing or flagged at %s",
                            point.label, missing, p.label)
                return None
            rows.append(x)
        torch = self._torch
        X = torch.from_numpy((np.stack(rows) - t["mean"]) / t["std"]).float()
        with torch.no_grad():
            _, x2 = self._model(X[:n].unsqueeze(0), X[n].unsqueeze(0).unsqueeze(1))
        err = ((x2.reshape(-1) - X[n]) ** 2).numpy()
        return err, err.mean(), vids, t["threshold"]

    def _pca(self, point: Point):
        p = self._bundle["pca"]
        x, vids, missing = self._vector(point, p["features"], {})
        if x is None:
            LOGGER.info("%s: PCA skipped, %s missing or flagged", point.label, missing)
            return None
        z = (x - p["mean"]) / p["scale"] - p["center"]
        err = (z - z @ p["components"].T @ p["components"]) ** 2
        return err, err.sum(), vids, p["threshold"]

    def _finding(self, subject: str, point: Point, scored) -> Optional[Finding]:
        err, score, vids, threshold = scored
        threshold *= self._scale
        if score <= threshold:
            return None
        # The worst-reconstructed metrics are the causes; a sentinel feature and its
        # __valid flag share a vid, so keep each vid once
        top, seen = [], set()
        for i in np.argsort(err)[::-1]:
            if vids[i] not in seen:
                seen.add(vids[i])
                top.append(i)
            if len(top) == self._top:
                break
        detail = [{"metric": point.name(vids[i]), "vid": vids[i],
                   "value": point[vids[i]].valuex, "error": float(err[i])} for i in top]
        LOGGER.info("%s: %s score %.4g > %.4g (x%.1f) -- worst %s",
                    point.label, subject, score, threshold, score / threshold,
                    ", ".join("/".join(d["metric"].split("/")[-2:]) for d in detail[:2]))
        note = {"score": float(score), "threshold": threshold, "top": detail,
                "model": {"created": self._bundle["created"], "source": self._bundle["source"]}}
        return self.finding(subject, *[point.cause(d["vid"], level=self._level) for d in detail],
                            note=json.dumps(note))

    def evaluate(self, point: Point, history: Sequence[Point] = ()) -> Sequence[Finding]:
        if not point.names:
            # Without names nothing maps onto the trained features
            LOGGER.warning("%s: no metric names in the point -- skipping", point.label)
            return []
        findings = []
        if "tranad" in self._detectors:
            scored = self._tranad(point, history)
            if scored is not None:
                findings.append(self._finding("tranad_outlier", point, scored))
        if "pca" in self._detectors:
            scored = self._pca(point)
            if scored is not None:
                findings.append(self._finding("pca_outlier", point, scored))
        return [f for f in findings if f is not None]
