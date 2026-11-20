"""
mu2e-tranad: CRV DQM anomaly detection

A modular pipeline for CRV detector anomaly detection using multiple statistical
and machine learning approaches (Z-Score, PCA, TranAD).

Subpackages:
  reading/    - Fetch a DQM source from the DQM MCP server, drop static columns,
                load and validate the CSV (dqm_mcp, export_from_dqm, filter_from_dqm,
                ingestion)
  training/   - Feature selection and scaling, TranAD input preparation, the Z-Score,
                PCA and TranAD models, and the frozen model bundle (preprocessing,
                tranad_prep, models, tranad_model, bundle)
  evaluator/  - The `crv_ml` evaluator for the Mu2e DQM driver framework (ml_evaluator,
                dqm_driver); needs the Mu2e/DQM python package
  analysis/   - Diagnostics and plots of the results (diagnostics, visualization)

run_pipeline.py runs reading -> training -> reports -> analysis (see README.md).
"""

__version__ = "0.1.0"
__author__ = "Sophie"

# evaluator is not imported here: it needs the Mu2e/DQM package (dqm.evaluate)
from . import reading, training, analysis

__all__ = ["reading", "training", "analysis"]
