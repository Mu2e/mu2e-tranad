"""
DQM Pipeline Runner

Master orchestrator for the complete anomaly detection pipeline:
  Phase 1: Ingestion   - Load and validate data
  Phase 2: Preprocessing - Feature scaling and selection
  Phase 3: Models      - Run anomaly detection (Z-Score, PCA, TranAD)
  Phase 4: Export      - Generate reports and visualizations
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from .reading import ingestion
from .reading.filter_from_dqm import filter_static_columns
from .training import bundle, preprocessing, models, tranad_prep
from .analysis.visualization import DQMVisualizer
from .analysis.diagnostics import AnomalyDiagnostics


def create_report(df: pd.DataFrame,
                 active_features: list,
                 model_results: dict,
                 output_path: str = "anomaly_report.csv") -> None:
    """
    Generate anomaly detection report.
    Uses TranAD as primary anomaly predictions, includes all three methods for evaluation.
    
    Args:
        df: Original DataFrame
        active_features: List of features used
        model_results: Results from ensemble detection (contains all predictions)
        output_path: Path to save CSV report
    """
    report_df = df.copy()
    # Primary anomaly column: TranAD predictions
    report_df['anomaly'] = model_results['tranad_predictions']
    # Individual detector predictions for evaluation
    report_df['z_score_anomaly'] = model_results['z_score_predictions']
    report_df['pca_anomaly'] = model_results['pca_predictions']
    report_df['tranad_anomaly'] = model_results['tranad_predictions']
    report_df['ensemble_voting_score'] = model_results['voting_scores']
    # Anomaly scores from each detector
    report_df['z_score_max'] = model_results['z_score_scores']
    report_df['pca_reconstruction_error'] = model_results['pca_scores']
    report_df['tranad_reconstruction_error'] = model_results['tranad_scores']
    
    # Save full report
    report_df.to_csv(output_path, index=False)
    print(f"\n✓ Report saved: {output_path}")
    
    # Summarize anomalies (TranAD-based)
    anomalies_df = report_df[report_df['anomaly'] == 1].copy()
    
    if len(anomalies_df) > 0:
        print(f"\nAnomalies ({len(anomalies_df)} records):")
        print("-" * 70)
        
        anomaly_path = output_path.replace('.csv', '_anomalies_only.csv')
        anomalies_df.to_csv(anomaly_path, index=False)
        print(f"\n✓ Anomalies report: {anomaly_path}")
    else:
        print("\n✓ No anomalies detected!")


def fetch_from_dqm(source_str: str, output_dir: str, groupx: str = "CRV",
                   start_time_after_iso_utc: str = None,
                   end_time_before_iso_utc: str = None) -> str:
    """
    Pull a DQM source from the DQM MCP server, drop static columns, and save it
    as a CSV in output_dir. Returns the path of the filtered CSV.
    """
    from .reading.dqm_mcp import fetch_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = source_str.replace(",", "_")

    print("\n[PHASE 0] FETCH FROM DQM MCP")
    print("-" * 70)
    df = fetch_matrix(source_str, groupx=groupx,
                      start_time_after_iso_utc=start_time_after_iso_utc,
                      end_time_before_iso_utc=end_time_before_iso_utc)
    raw_path = output_dir / f"dqm_{tag}.csv"
    df.to_csv(raw_path, index=False)

    df_clean, dropped = filter_static_columns(df)
    filtered_path = output_dir / f"dqm_{tag}_filtered.csv"
    df_clean.to_csv(filtered_path, index=False)
    print(f"✓ Dropped {len(dropped)} static columns")
    print(f"✓ Saved: {raw_path}")
    print(f"✓ Saved: {filtered_path}")
    return str(filtered_path)


def _prepare_training(df_clean, X_scaled, response_feats, condition_feats, epochs,
                      tranad_features, train_mode, seed, scale_by, periods_path,
                      train_clean, threshold_mode):
    """
    TranAD inputs and training rows for --retrain (Phase 2b). May add 'config_period' to
    df_clean. Returns (X_tranad_raw, tranad_cols, train_idx, val_idx, X_tranad, X_drift).
    """
    if tranad_features in ("response", "changes"):
        tranad_df, tranad_cols, dropped = tranad_prep.prepare_response_features(df_clean, response_feats)
        print(f"TranAD features: {len(tranad_cols)} (response metrics + validity flags); "
              f"excluded {len(condition_feats)} condition, dropped {len(dropped)} near-constant: {dropped}")
        if tranad_features == "changes":
            change_df, change_cols = tranad_prep.prepare_condition_changes(
                df_clean, condition_feats, epochs)
            tranad_df = pd.concat([tranad_df, change_df], axis=1)
            n_changed = int((change_df != 0).any(axis=1).sum())
            print(f"  + {len(change_cols)} condition-change features "
                  f"(relative to calibration epoch; {n_changed} rows deviate)")
        X_tranad_raw = tranad_df.to_numpy(dtype=float)
    elif tranad_features == "all":
        X_tranad_raw = None
        tranad_cols = None
    else:
        raise ValueError(f"Unknown tranad_features: {tranad_features}")

    train_idx, val_idx = tranad_prep.select_rows(epochs, mode=train_mode, seed=seed)
    print(f"TranAD rows: {train_mode} -> {len(train_idx)} train, {len(val_idx)} threshold-validation")
    X_tranad = (X_scaled if X_tranad_raw is None
                else tranad_prep.scale_on_rows(X_tranad_raw, train_idx))
    X_drift = X_tranad  # drift keeps the global scaling: its job is to see period changes
    if scale_by == "period":
        if X_tranad_raw is None or not periods_path:
            raise ValueError("--scale-by period needs --periods and response/changes features")
        periods = tranad_prep.load_periods(periods_path)
        groups = tranad_prep.period_labels(df_clean["run"].to_numpy(), periods)
        df_clean["config_period"] = [periods.label[g] if g >= 0 else "" for g in groups]
        X_tranad = tranad_prep.scale_by_groups(X_tranad_raw, groups, train_idx)
        print(f"TranAD inputs scaled within {len(periods)} configuration periods "
              f"({int((groups < 0).sum())} rows outside every period use the global scale)")
    elif scale_by != "global":
        raise ValueError(f"Unknown scale_by: {scale_by}")
    if train_clean:
        valid_cols = np.array([c.endswith(tranad_prep.VALID_SUFFIX) for c in tranad_cols])
        before = len(train_idx)
        train_idx = tranad_prep.clean_rows(X_tranad, train_idx, train_clean, valid_cols)
        print(f"Clean training rows (no input beyond {train_clean} sigma): {len(train_idx)} of {before}")
    if threshold_mode not in ("heldout", "train"):
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
    return X_tranad_raw, tranad_cols, train_idx, val_idx, X_tranad, X_drift


def _add_drift(df_clean, Z, drift_cols, center, drift_window, drift_sigma):
    """Phase 3b: training-drift columns on df_clean, from inputs Z and training medians."""
    shift = tranad_prep.rolling_drift(Z, center, drift_window)
    shift[:, [c.endswith(tranad_prep.VALID_SUFFIX) for c in drift_cols]] = 0.0
    abs_shift = np.nan_to_num(np.abs(shift), nan=0.0)
    df_clean['drift_max_shift'] = abs_shift.max(axis=1)
    df_clean['drift_worst_feature'] = np.array(drift_cols)[abs_shift.argmax(axis=1)]
    df_clean['drift_n_features'] = (abs_shift > drift_sigma).sum(axis=1)
    df_clean['training_drift'] = (df_clean['drift_n_features'] > 0).astype(int)
    drifted = df_clean['training_drift'] == 1
    print(f"\nTraining drift (median of {drift_window} rows vs training, > {drift_sigma} sigma): "
          f"{int(drifted.sum())} rows ({100 * drifted.mean():.1f}%)")
    if drifted.any():
        first = df_clean[drifted].iloc[0]
        print(f"  first at run {first['run']}"
              f"{':%s' % first['subrun'] if 'subrun' in df_clean else ''}, worst "
              f"{first['drift_worst_feature']} ({first['drift_max_shift']:.1f} sigma)")


def main(csv_path: str,
         output_dir: str = "./results",
         iqr_multiplier: float = 1.5,
         voting_threshold: float = 2.0,
         threshold_percentile: float = 99.5,
         generate_plots: bool = True,
         apply_iqr_filtering: bool = False,
         tranad_features: str = "response",
         train_mode: str = "stratified",
         threshold_mode: str = "heldout",
         seed: int = 0,
         drift_window: int = 50,
         drift_sigma: float = 3.0,
         save_model: str = None,
         auto_save: bool = True,
         source: str = None,
         drop_empty: bool = False,
         scale_by: str = "global",
         periods_path: str = None,
         train_clean: float = None,
         model: str = None,
         retrain: bool = False) -> None:
    """
    Run complete pipeline.

    By default the data is scored with a saved model (`model`, else the default CRV model,
    training/bundle.py default_model_path()) and nothing is trained. With retrain=True the
    detectors are trained on this data instead, and the trained model is saved.

    Args:
        csv_path: Path to filtered CSV
        output_dir: Directory for output reports
        model: Model bundle to score with (default: the packaged 006-001 CRV model, or
            $MU2E_TRANAD_MODEL). Not used with retrain.
        retrain: Train on this data instead of loading a model. The options marked
            "training only" below need it.

    Training only (retrain=True): tranad_features, train_mode, threshold_mode,
    threshold_percentile, seed, scale_by, periods_path, train_clean, save_model, auto_save.
    In scoring mode these come from the model.

    Args (continued):
        iqr_multiplier: IQR multiplier for outlier detection (only used if apply_iqr_filtering=True)
        voting_threshold: Voting threshold for ensemble
        threshold_percentile: TranAD threshold percentile (default: 99.5)
        generate_plots: Whether to generate visualization plots
        apply_iqr_filtering: Whether to apply IQR-based outlier removal (default: False to preserve anomalies)
        tranad_features: 'response' (exclude calibration/pedestal metrics, drop near-constant
            features, handle sentinels), 'changes' (response + calibration/pedestal metrics as
            changes relative to their calibration epoch) or 'all' (same scaled matrix as Z-Score/PCA)
        train_mode: TranAD training rows: 'stratified' (every calibration epoch),
            'random', or 'chronological' (first 30%, original behaviour)
        threshold_mode: 'heldout' (percentile of held-out rows) or 'train' (of training rows)
        seed: Random seed for row selection
        drift_window: Rows over which the training-drift median is taken
        drift_sigma: Flag training drift when a feature's window median has moved more
            than this many training standard deviations from its training median
        save_model: If set, write the trained models here for the online DQM evaluator
            (see training/bundle.py; read by dqm/evaluate/tranad.py in Mu2e/DQM)
        auto_save: Without save_model, still save the trained model in output_dir
            (tranad_model_<timestamp>.pkl, a full bundle; or tranad_weights_<timestamp>.pkl,
            weights only, when the options are ones the bundle can't represent)
        source: DQM source the data came from, recorded in the model bundle
        drop_empty: Set aside subruns with no CRV digis before modelling; they are written
            to no_crv_data.csv and flagged there instead of being scored
        scale_by: How TranAD inputs are standardised: 'global' (mean/std of all training
            rows, the default) or 'period' (robust median/MAD within each detector-
            configuration period from periods_path)
        periods_path: CSV of configuration periods (first_run,last_run,label)
        train_clean: If set, leave out of TranAD training any row with an input beyond this
            many robust sigma within its period (a proxy for known-good data)
    """
    model_bundle = None
    if retrain:
        if model:
            raise ValueError("--model loads a trained model; with --retrain, name the new "
                             "model's file with --save-model")
        if save_model and (scale_by != "global" or drop_empty or train_clean):
            raise ValueError("--save-model supports only the default scaling for now "
                             "(the online evaluator has no per-period scaling yet)")
    else:
        training_only = {"--tranad-features": (tranad_features, "response"),
                         "--train-mode": (train_mode, "stratified"),
                         "--threshold-mode": (threshold_mode, "heldout"),
                         "--threshold-percentile": (threshold_percentile, 99.5),
                         "--seed": (seed, 0), "--scale-by": (scale_by, "global"),
                         "--periods": (periods_path, None), "--train-clean": (train_clean, None),
                         "--save-model": (save_model, None)}
        given = [flag for flag, (value, default) in training_only.items() if value != default]
        if given:
            raise ValueError(f"training-only option(s) given: {', '.join(given)}. Add --retrain "
                             f"to train, or leave them out to score with the saved model")
        model = model or bundle.default_model_path()
        model_bundle = bundle.load_bundle(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = output_dir / "plots" if generate_plots else None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("DQM ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    print(f"Start time: {timestamp}")
    print(f"CSV input:  {csv_path}")
    print(f"Output dir: {output_dir}")
    if retrain:
        print("Mode:       retrain (train the detectors on this data)")
    else:
        print(f"Mode:       score with saved model {model}")
        print(f"            (trained {model_bundle['created']} on {model_bundle['source']}, "
              f"{model_bundle['rows']} rows, runs {model_bundle['runs']})")
    print("=" * 70 + "\n")
    
    # =========================================================================
    # PHASE 1: INGESTION
    # =========================================================================
    print("\n[PHASE 1] INGESTION")
    print("-" * 70)
    df, feature_cols = ingestion.ingest_data(csv_path)
    
    # =========================================================================
    # PHASE 2: PREPROCESSING
    # =========================================================================
    print("\n[PHASE 2] PREPROCESSING")
    print("-" * 70)
    df_clean, active_features, X_scaled, scaler = preprocessing.preprocess_pipeline(
        df, feature_cols, iqr_multiplier=iqr_multiplier, apply_iqr_filtering=apply_iqr_filtering
    )
    
    # =========================================================================
    # PHASE 2b: TRANAD INPUTS & CALIBRATION EPOCHS
    # =========================================================================
    print("\n[PHASE 2b] TRANAD INPUTS & CALIBRATION EPOCHS")
    print("-" * 70)
    df_clean = df_clean.reset_index(drop=True)
    if drop_empty:
        empty = tranad_prep.no_crv_data_mask(df_clean)
        df_clean.loc[empty].to_csv(output_dir / "no_crv_data.csv", index=False)
        print(f"No CRV data: {int(empty.sum())} rows in {df_clean.loc[empty, 'run'].nunique()} runs "
              f"set aside (no_crv_data.csv)")
        df_clean = df_clean.loc[~empty].reset_index(drop=True)
        X_scaled = X_scaled[~empty]
    condition_feats, response_feats = tranad_prep.split_condition_features(active_features)
    epochs, epoch_summary = tranad_prep.detect_calibration_epochs(df_clean, condition_feats)
    df_clean['calib_epoch'] = epochs
    if not epoch_summary.empty:
        epoch_summary.to_csv(output_dir / "calibration_epochs.csv", index=False)
        print(f"Calibration epochs (condition metrics: {len(condition_feats)}):")
        print(epoch_summary[["epoch", "rows", "first_run", "last_run"] +
                            [c for c in ("start", "end") if c in epoch_summary]].to_string(index=False))
        print("  (epoch -1 = short transients, e.g. constants reading 0; excluded from training)")

    if retrain:
        X_tranad_raw, tranad_cols, train_idx, val_idx, X_tranad, X_drift = _prepare_training(
            df_clean, X_scaled, response_feats, condition_feats, epochs, tranad_features,
            train_mode, seed, scale_by, periods_path, train_clean, threshold_mode)

    # =========================================================================
    # PHASE 3: MODELS (ANOMALY DETECTION)
    # =========================================================================
    print("\n[PHASE 3] ANOMALY DETECTION")
    print("-" * 70)
    if retrain:
        anomaly_pred, model_results = models.run_ensemble_detection(
            X_scaled, df_clean, active_features, voting_threshold=voting_threshold,
            threshold_percentile=threshold_percentile,
            X_tranad=X_tranad, tranad_train_idx=train_idx,
            tranad_val_idx=val_idx if threshold_mode == "heldout" else None
        )
        # Training drift, measured on the inputs the online evaluator will see: TranAD's
        # (standardised on its training rows), else PCA's (fitted on every row)
        if model_results.get("tranad_detector") is not None:
            Z, drift_cols, drift_rows = X_drift, tranad_cols or active_features, train_idx
        else:
            Z, drift_cols, drift_rows = X_scaled, active_features, np.arange(len(X_scaled))
        drift_center = tranad_prep.drift_center(Z, drift_rows)
    else:
        anomaly_pred, model_results = models.score_with_bundle(
            X_scaled, df_clean, model_bundle, voting_threshold=voting_threshold)
        # Drift against the training medians saved with the model, on the same inputs
        drift_center = None
        d = model_bundle.get("drift")
        if d is not None:
            t = model_bundle["tranad"]
            if t is not None and model_results["tranad_inputs"] is not None:
                Z, drift_cols = model_results["tranad_inputs"], t["inputs"]
            elif t is None:
                p = model_bundle["pca"]
                Xp, _ = bundle.inputs_from_frame(df_clean, p["features"], {}, p["mean"])
                Z, drift_cols = (Xp - p["mean"]) / p["scale"], p["features"]
            if t is None or model_results["tranad_inputs"] is not None:
                drift_center = np.asarray(d["center"], dtype=float)
        if drift_center is None:
            print("\nTraining drift: not measured (no drift reference in the model, "
                  "or TranAD could not run)")

    if drift_center is not None:
        _add_drift(df_clean, Z, drift_cols, drift_center, drift_window, drift_sigma)

    if retrain:
        bundle_ok = (scale_by == "global" and not drop_empty and not train_clean
                     and tranad_features != "changes")
        if not save_model and auto_save and bundle_ok:
            save_model = str(output_dir / f"tranad_model_{timestamp}.pkl")
        if not save_model and auto_save:
            bundle.save_weights(output_dir / f"tranad_weights_{timestamp}.pkl", model_results,
                                tranad_cols,
                                settings={"source": source or csv_path,
                                          "tranad_features": tranad_features,
                                          "train_mode": train_mode, "threshold_mode": threshold_mode,
                                          "threshold_percentile": threshold_percentile, "seed": seed,
                                          "drop_empty": drop_empty, "scale_by": scale_by,
                                          "periods": periods_path, "train_clean": train_clean})
        if save_model:
            bundle.save_bundle(save_model, df_clean, active_features, scaler, model_results,
                               tranad_features, tranad_cols, X_tranad_raw, train_idx,
                               source=source or csv_path,
                               drift={"center": drift_center, "window": drift_window,
                                      "sigma": drift_sigma})

    # =========================================================================
    # PHASE 4: REPORTING
    # =========================================================================
    print("\n[PHASE 4] REPORTING")
    print("-" * 70)
    report_path = output_dir / f"anomaly_report_{timestamp}.csv"
    create_report(df_clean, active_features, model_results, str(report_path))
    
    # Add TranAD anomaly column to df_clean for diagnostics and visualization
    df_clean = df_clean.copy()
    df_clean['anomaly'] = model_results['tranad_predictions']
    
    # =========================================================================
    # PHASE 5: DIAGNOSTICS
    # =========================================================================
    print("\n[PHASE 5] DIAGNOSTICS")
    print("-" * 70)
    diagnostics = AnomalyDiagnostics(df_clean, active_features)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics.export_diagnostics(
        model_results['z_score_predictions'],
        model_results['pca_predictions'],
        model_results['tranad_predictions'],
        anomaly_pred,
        diagnostics_dir
    )
    
    # =========================================================================
    # PHASE 6: VISUALIZATION (OPTIONAL)
    # =========================================================================
    if generate_plots:
        print("\n[PHASE 6] VISUALIZATION")
        print("-" * 70)
        visualizer = DQMVisualizer(output_dir=str(plot_dir))
        
        # Plot training curve if available
        if 'tranad_losses' in model_results and len(model_results.get('tranad_losses', [])) > 0:
            visualizer.plot_tranad_training(
                model_results['tranad_losses'],
                detector_name='TranAD'
            )
        
        # Generate all other plots using TranAD predictions
        visualizer.generate_all_plots(
            df_clean,
            active_features,
            model_results['z_score_scores'],
            model_results['pca_scores'],
            model_results['tranad_scores'],
            anomaly_col='anomaly',
            tranad_losses=None,  # Already plotted above
            anomaly_predictions=model_results['tranad_predictions'],
            model_results=model_results  # Pass model results to use actual thresholds
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Records processed:   {len(df)}")
    print(f"Records after clean: {len(df_clean)}")
    print(f"Features analyzed:   {len(active_features)}")
    print(f"Model:               {'retrained on this data' if retrain else model}")
    tranad_anomalies = int(model_results['tranad_predictions'].sum())
    ensemble_anomalies = int(anomaly_pred.sum())
    print(f"TranAD Anomalies:    {tranad_anomalies} ({100*tranad_anomalies/len(df_clean):.2f}%)")
    print(f"Ensemble Anomalies:  {ensemble_anomalies} (for reference, see diagnostics)")
    print(f"Output dir:          {output_dir}")
    if generate_plots:
        print(f"Plot dir:            {plot_dir}")
    print("=" * 70 + "\n")


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="DQM Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default the data is scored with the saved 006-001 CRV model (models/), with no training.
Add --retrain to train the detectors on this data instead.

Examples:
  python -m mu2e_tranad --csv data.csv                       # score with the default model
  python -m mu2e_tranad --csv data.csv --model my_model.pkl  # score with another model
  python -m mu2e_tranad --dqm-source crvreco,kpp_crv,file,006-001 --retrain \\
      --save-model models/crv_tranad_006-001_v2.pkl            # train a new model
        """
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train the detectors on this data instead of scoring with a saved model. "
             "Needed for the training options (--tranad-features, --train-mode, ...)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model bundle to score with (default: $MU2E_TRANAD_MODEL, else "
             "models/crv_tranad_006-001.pkl). Not used with --retrain"
    )
    
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=str,
        help="Path to filtered DQM CSV file"
    )
    source.add_argument(
        "--dqm-source",
        type=str,
        help="Fetch data from the DQM MCP server for this source, "
             "given as process,stream,aggregation,version (e.g. crvreco,kpp_crv,file,004-000)"
    )
    parser.add_argument(
        "--dqm-group",
        type=str,
        default="CRV",
        help="DQM value group to fetch with --dqm-source (default: CRV)"
    )
    parser.add_argument(
        "--after",
        type=str,
        help="With --dqm-source: only intervals starting after this ISO UTC time"
    )
    parser.add_argument(
        "--before",
        type=str,
        help="With --dqm-source: only intervals ending before this ISO UTC time"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for reports (default: ./results)"
    )
    parser.add_argument(
        "--iqr",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--voting-threshold",
        type=float,
        default=2.0,
        help="Voting threshold for ensemble (default: 2.0)"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.5,
        help="TranAD threshold percentile (default: 99.5)"
    )
    parser.add_argument(
        "--tranad-features",
        choices=["response", "changes", "all"],
        default="response",
        help="TranAD inputs: 'response' excludes calibration/pedestal metrics and near-constant "
             "features (default); 'changes' adds calibration/pedestal metrics as changes "
             "relative to their calibration epoch; 'all' uses every active feature"
    )
    parser.add_argument(
        "--train-mode",
        choices=["stratified", "random", "chronological"],
        default="stratified",
        help="TranAD training rows: sampled from every calibration epoch (default), "
             "random over all data, or the first 30%% (original behaviour)"
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["heldout", "train"],
        default="heldout",
        help="Set the TranAD threshold on held-out rows (default) or on the training rows"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for TranAD row selection (default: 0)"
    )
    parser.add_argument(
        "--drift-window",
        type=int,
        default=50,
        help="Rows over which training drift is measured (default: 50)"
    )
    parser.add_argument(
        "--drift-sigma",
        type=float,
        default=3.0,
        help="Flag training drift beyond this many training standard deviations (default: 3.0)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        help="With --retrain: write the trained models to this file, for the online DQM "
             "evaluator (tranad, in Mu2e/DQM). Default: saved anyway, as "
             "<output>/tranad_model_<timestamp>.pkl"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="With --retrain: don't save the trained model"
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Set aside subruns with no CRV digis (written to no_crv_data.csv) before modelling"
    )
    parser.add_argument(
        "--scale-by",
        choices=["global", "period"],
        default="global",
        help="Standardise TranAD inputs over all training rows (default) or within each "
             "detector-configuration period given by --periods (robust median/MAD)"
    )
    parser.add_argument(
        "--periods",
        type=str,
        help="CSV of configuration periods: first_run,last_run,label"
    )
    parser.add_argument(
        "--train-clean",
        type=float,
        help="Leave out of TranAD training rows with any input beyond this many robust sigma"
    )
    parser.add_argument(
        "--apply-iqr",
        action="store_true",
        help="Apply IQR-based outlier filtering in preprocessing (default: False to preserve anomalies)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate visualization plots (default: True)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization plots"
    )
    
    args = parser.parse_args()
    generate_plots = not args.no_plots
    
    try:
        csv_path = args.csv
        if args.dqm_source:
            csv_path = fetch_from_dqm(args.dqm_source, args.output, groupx=args.dqm_group,
                                      start_time_after_iso_utc=args.after,
                                      end_time_before_iso_utc=args.before)
        main(
            csv_path=csv_path,
            output_dir=args.output,
            iqr_multiplier=args.iqr,
            voting_threshold=args.voting_threshold,
            threshold_percentile=args.threshold_percentile,
            generate_plots=generate_plots,
            apply_iqr_filtering=args.apply_iqr,
            tranad_features=args.tranad_features,
            train_mode=args.train_mode,
            threshold_mode=args.threshold_mode,
            seed=args.seed,
            drift_window=args.drift_window,
            drift_sigma=args.drift_sigma,
            save_model=args.save_model,
            auto_save=not args.no_save_model,
            source=args.dqm_source,
            drop_empty=args.drop_empty,
            scale_by=args.scale_by,
            periods_path=args.periods,
            train_clean=args.train_clean,
            model=args.model,
            retrain=args.retrain
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
