"""Training: turn the time series into model input and fit the detectors.

  preprocessing  - active-feature selection and scaling
  tranad_prep    - condition/response split, calibration epochs, training rows
  models         - Z-Score, PCA and TranAD anomaly detectors
  tranad_model   - TranAD wrapper around anldq (CrossExperimentalAIDQM/src)
  bundle         - frozen model file written by --save-model, read by the evaluator
"""
