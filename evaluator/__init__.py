"""Evaluator: the `crv_ml` evaluator for the Mu2e DQM driver framework (Mu2e/DQM).

  ml_evaluator  - MLEvaluator: scores each new DQM point with a trained bundle
  dqm_driver    - dqm-driver with crv_ml registered:
                  python -m mu2e_tranad.evaluator.dqm_driver <config.yaml>

Needs the Mu2e/DQM python package (dqm.evaluate) on PYTHONPATH.
"""

from .ml_evaluator import MLEvaluator

__all__ = ["MLEvaluator"]
