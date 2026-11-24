"""
Phase 3: Anomaly Detection Models

Implementations of three complementary approaches:
  1. Z-Score    - Statistical thresholding (simple, interpretable)
  2. PCA        - Principal Component Analysis (multivariate outliers)
  3. TranAD     - Transformer-based Anomaly Detection (deep learning, for sequences)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .tranad_model import TranADDetector, TRANAD_ANL_AVAILABLE
except ImportError:
    TRANAD_ANL_AVAILABLE = False
    TranADDetector = None
    print("⚠️  PyTorch/ANL TranAD not available.")


class ZScoreDetector:
    """
    Statistical anomaly detection using Z-scores.
    
    Detects records where any feature deviates more than threshold standard deviations
    from the mean.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Args:
            threshold: Number of standard deviations to consider anomalous (default: 3σ)
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'ZScoreDetector':
        """Fit scaler on training data."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies.
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if self.mean_ is None:
            raise ValueError("Must call fit() first")
        
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-8))
        anomalies = (z_scores > self.threshold).any(axis=1).astype(int)
        return anomalies
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return max Z-score for each record."""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-8))
        return z_scores.max(axis=1)


class PCADetector:
    """
    PCA-based anomaly detection.
    
    Detects multivariate outliers by computing reconstruction error from PCA components.
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        """
        Args:
            n_components: Number of PCA components (None = auto)
            variance_threshold: Variance to retain if n_components=None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca_ = None
        self.reconstruction_error_threshold_ = None
    
    def fit(self, X: np.ndarray, contamination: float = 0.05) -> 'PCADetector':
        """
        Fit PCA and determine reconstruction error threshold.
        
        Args:
            X: Training data
            contamination: Expected fraction of outliers
        """
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X)
        
        # Compute reconstruction errors on training data
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Set threshold as upper percentile
        self.reconstruction_error_threshold_ = np.percentile(errors, (1 - contamination) * 100)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies based on reconstruction error.
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if self.pca_ is None:
            raise ValueError("Must call fit() first")
        
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        anomalies = (errors > self.reconstruction_error_threshold_).astype(int)
        return anomalies
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error for each record."""
        X_reconstructed = self.pca_.inverse_transform(self.pca_.transform(X))
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        return errors


class TranADDetectorWrapper:
    """Removed - using TranADDetector directly from tranad_model.py"""
    pass


def run_ensemble_detection(X: np.ndarray, 
                          df: pd.DataFrame,
                          active_features: List[str],
                          voting_threshold: float = 2.0,
                          include_tranad: bool = True,
                          threshold_percentile: float = 95,
                          X_tranad: np.ndarray = None,
                          tranad_train_idx: np.ndarray = None,
                          tranad_val_idx: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Run ensemble of Z-Score, PCA, and TranAD detectors.
    
    Args:
        X: Scaled feature matrix (n_samples, n_features)
        df: DataFrame for context/timestamps
        active_features: List of feature names
        voting_threshold: Detectors >= this value → anomaly
        include_tranad: Whether to include TranAD in ensemble
        threshold_percentile: TranAD threshold percentile (default: 95)
        X_tranad: Separate TranAD input matrix (default: X)
        tranad_train_idx: Rows TranAD trains on (default: first 30%, chronological)
        tranad_val_idx: Held-out rows for the TranAD threshold (default: training rows)
        
    Returns:
        Tuple of (ensemble_predictions, results_dict)
    """
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION - ENSEMBLE")
    print("=" * 70)
    
    # Z-Score detector
    z_detector = ZScoreDetector(threshold=3.0)
    z_detector.fit(X)
    z_pred = z_detector.predict(X)
    z_scores = z_detector.anomaly_scores(X)
    
    # PCA detector
    pca_detector = PCADetector()
    pca_detector.fit(X, contamination=0.05)
    pca_pred = pca_detector.predict(X)
    pca_scores = pca_detector.anomaly_scores(X)
    
    # TranAD detector (using official ANL implementation)
    tranad_pred = np.zeros_like(z_pred)
    tranad_scores = np.zeros_like(z_scores)
    tranad_losses = []
    tranad_batch_losses = []
    tranad_threshold = None
    tranad_detector = None

    if include_tranad and TRANAD_ANL_AVAILABLE:
        try:
            # Use official ANL TranAD with Genesis notebook parameters
            tranad_detector = TranADDetector(
                n_window=4,
                d_model=32,
                nhead=4,
                d_ff=64,
                num_layers=2,
                learning_rate=0.001,
                epochs=50,
                contamination=0.05,
                threshold_percentile=threshold_percentile,
                train_ratio=0.3
            )
            X_t = X if X_tranad is None else X_tranad
            tranad_detector.fit(X_t, train_idx=tranad_train_idx, val_idx=tranad_val_idx)
            tranad_pred = tranad_detector.predict(X_t)
            tranad_scores = tranad_detector.anomaly_scores(X_t)
            tranad_losses = tranad_detector.training_losses
            tranad_threshold = tranad_detector.threshold  # Capture actual threshold
            print(f"  TranAD threshold captured: {tranad_threshold:.8f}")
        except Exception as e:
            print(f"⚠️  TranAD training failed: {e}")
            tranad_detector = None
            tranad_pred = np.zeros_like(z_pred)
            tranad_scores = np.zeros_like(z_scores)
            tranad_losses = []
            tranad_batch_losses = []
    
    # Voting ensemble (Z-Score + PCA + TranAD)
    votes = z_pred + pca_pred + tranad_pred
    ensemble_pred = (votes >= voting_threshold).astype(int)
    
    n_anomalies = ensemble_pred.sum()
    anomaly_rate = n_anomalies / len(ensemble_pred) * 100
    
    print(f"\nZ-Score Detector:")
    print(f"  Anomalies found:     {z_pred.sum()} ({z_pred.sum()/len(z_pred)*100:.2f}%)")
    print(f"  Max Z-score:         {z_scores.max():.2f}")
    
    print(f"\nPCA Detector:")
    print(f"  Anomalies found:     {pca_pred.sum()} ({pca_pred.sum()/len(pca_pred)*100:.2f}%)")
    print(f"  Max reconstruction error: {pca_scores.max():.2f}")
    
    if include_tranad and TRANAD_ANL_AVAILABLE:
        print(f"\nTranAD Detector:")
        print(f"  Anomalies found:     {tranad_pred.sum()} ({tranad_pred.sum()/len(tranad_pred)*100:.2f}%)")
        print(f"  Max reconstruction error: {tranad_scores.max():.2f}")
    
    print(f"\nEnsemble (voting threshold={voting_threshold}):")
    print(f"  Anomalies found:     {n_anomalies} ({anomaly_rate:.2f}%)")
    
    results = {
        'ensemble_predictions': ensemble_pred,
        'z_score_predictions': z_pred,
        'z_score_scores': z_scores,
        'pca_predictions': pca_pred,
        'pca_scores': pca_scores,
        'tranad_predictions': tranad_pred,
        'tranad_scores': tranad_scores,
        'tranad_losses': tranad_losses,
        'tranad_threshold': tranad_threshold,  # Add actual threshold
        'pca_detector': pca_detector,      # fitted, for saving a model bundle
        'tranad_detector': tranad_detector,
        'voting_scores': votes,
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
    }
    
    if tranad_threshold is not None:
        print(f"\nThreshold stored in results: {tranad_threshold:.8f}")
    
    return ensemble_pred, results


def score_with_bundle(X: np.ndarray,
                      df: pd.DataFrame,
                      model_bundle: dict,
                      voting_threshold: float = 2.0) -> Tuple[np.ndarray, Dict]:
    """
    Score every row with a saved model bundle (training/bundle.py) instead of training.

    TranAD and PCA use the bundle's weights, scaling and thresholds, frozen at training
    time. Z-score has no saved model: it is computed on this data, as when training. A PCA
    model that keeps every component reconstructs exactly, so it is skipped (scores 0).

    Returns (ensemble_predictions, results) with the keys of run_ensemble_detection, plus
    'tranad_inputs' (the scaled TranAD input matrix, or None) for the drift check.
    """
    from .bundle import inputs_from_frame

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION - SAVED MODEL")
    print("=" * 70)

    z_detector = ZScoreDetector(threshold=3.0)
    z_detector.fit(X)
    z_pred = z_detector.predict(X)
    z_scores = z_detector.anomaly_scores(X)

    p = model_bundle["pca"]
    pca_pred = np.zeros(len(X), dtype=int)
    pca_scores = np.zeros(len(X))
    if len(p["components"]) >= len(p["features"]):
        print(f"PCA: the model keeps all {len(p['components'])} components, so its "
              f"reconstruction error is rounding noise; skipped")
    else:
        Xp, _ = inputs_from_frame(df, p["features"], {}, p["mean"])
        Zp = (Xp - p["mean"]) / p["scale"] - p["center"]
        C = p["components"]
        pca_scores = ((Zp - Zp @ C.T @ C) ** 2).sum(axis=1)
        pca_pred = (pca_scores > p["threshold"]).astype(int)

    t = model_bundle["tranad"]
    tranad_pred = np.zeros(len(X), dtype=int)
    tranad_scores = np.zeros(len(X))
    tranad_threshold, tranad_detector, X_tranad = None, None, None
    if t is None:
        print("TranAD: the model has none")
    elif not TRANAD_ANL_AVAILABLE:
        print("⚠️  TranAD skipped: PyTorch not available (source mu2e_tranad/setup_env.sh)")
    else:
        Xr, filled = inputs_from_frame(df, t["inputs"], t["fill"], t["mean"])
        for col, n in filled.items():
            print(f"  {col}: {n} missing value(s) set to the training mean")
        X_tranad = (Xr - t["mean"]) / t["std"]
        tranad_detector = TranADDetector.from_bundle(t)
        tranad_detector.score(X_tranad)
        tranad_pred = tranad_detector.predict(X_tranad)
        tranad_scores = tranad_detector.anomaly_scores(X_tranad)
        tranad_threshold = tranad_detector.threshold
        print(f"TranAD: {len(t['inputs'])} inputs, window {t['n_window']}, "
              f"threshold {tranad_threshold:.8f} (from training)")

    votes = z_pred + pca_pred + tranad_pred
    ensemble_pred = (votes >= voting_threshold).astype(int)
    n_anomalies = ensemble_pred.sum()
    anomaly_rate = n_anomalies / len(ensemble_pred) * 100

    print(f"\nZ-Score Detector:   {z_pred.sum()} anomalies ({z_pred.mean() * 100:.2f}%)")
    print(f"PCA Detector:       {pca_pred.sum()} anomalies ({pca_pred.mean() * 100:.2f}%)")
    print(f"TranAD Detector:    {tranad_pred.sum()} anomalies ({tranad_pred.mean() * 100:.2f}%)")
    print(f"Ensemble (>= {voting_threshold}): {n_anomalies} anomalies ({anomaly_rate:.2f}%)")

    results = {
        'ensemble_predictions': ensemble_pred,
        'z_score_predictions': z_pred,
        'z_score_scores': z_scores,
        'pca_predictions': pca_pred,
        'pca_scores': pca_scores,
        'tranad_predictions': tranad_pred,
        'tranad_scores': tranad_scores,
        'tranad_losses': [],
        'tranad_threshold': tranad_threshold,
        'pca_detector': None,
        'tranad_detector': tranad_detector,
        'tranad_inputs': X_tranad,
        'voting_scores': votes,
        'n_anomalies': n_anomalies,
        'anomaly_rate': anomaly_rate,
    }
    return ensemble_pred, results
