"""
DQM Anomaly Detection Pipeline

A modular pipeline for CRV detector anomaly detection using multiple statistical
and machine learning approaches (Z-Score, PCA, TranAD).

Phases (orchestrated by run_pipeline.py; see README.md):
  0. Fetch         - Download a DQM source from the DQM MCP server (dqm_mcp.py)
  1. Ingestion     - Load filtered CSV, validate timestamps, handle timezones
  2. Preprocessing - Active-feature selection and scaling
  2b. TranAD prep  - Condition/response split, calibration epochs, training rows
  3. Models        - Z-Score, PCA, and TranAD anomaly detection
  4. Reports       - anomaly_report_*.csv
  5. Diagnostics   - diagnostics.py
  6. Plots         - visualization.py
"""

__version__ = "0.1.0"
__author__ = "Sophie"

from . import ingestion, preprocessing, models, diagnostics

__all__ = ["ingestion", "preprocessing", "models", "diagnostics"]
