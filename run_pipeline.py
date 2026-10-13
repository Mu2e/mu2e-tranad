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

from . import ingestion, preprocessing, models, tranad_prep
from .visualization import DQMVisualizer
from .diagnostics import AnomalyDiagnostics
from .filter_from_dqm import filter_static_columns


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
    from .dqm_mcp import fetch_matrix

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
         seed: int = 0) -> None:
    """
    Run complete pipeline.
    
    Args:
        csv_path: Path to filtered CSV
        output_dir: Directory for output reports
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
    """
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
    condition_feats, response_feats = tranad_prep.split_condition_features(active_features)
    epochs, epoch_summary = tranad_prep.detect_calibration_epochs(df_clean, condition_feats)
    df_clean['calib_epoch'] = epochs
    if not epoch_summary.empty:
        epoch_summary.to_csv(output_dir / "calibration_epochs.csv", index=False)
        print(f"Calibration epochs (condition metrics: {len(condition_feats)}):")
        print(epoch_summary[["epoch", "rows", "first_run", "last_run"] +
                            [c for c in ("start", "end") if c in epoch_summary]].to_string(index=False))
        print("  (epoch -1 = short transients, e.g. constants reading 0; excluded from training)")

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
    else:
        raise ValueError(f"Unknown tranad_features: {tranad_features}")

    train_idx, val_idx = tranad_prep.select_rows(epochs, mode=train_mode, seed=seed)
    print(f"TranAD rows: {train_mode} -> {len(train_idx)} train, {len(val_idx)} threshold-validation")
    X_tranad = (X_scaled if X_tranad_raw is None
                else tranad_prep.scale_on_rows(X_tranad_raw, train_idx))
    if threshold_mode not in ("heldout", "train"):
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    # =========================================================================
    # PHASE 3: MODELS (ANOMALY DETECTION)
    # =========================================================================
    print("\n[PHASE 3] ANOMALY DETECTION")
    print("-" * 70)
    anomaly_pred, model_results = models.run_ensemble_detection(
        X_scaled, df_clean, active_features, voting_threshold=voting_threshold,
        threshold_percentile=threshold_percentile,
        X_tranad=X_tranad, tranad_train_idx=train_idx,
        tranad_val_idx=val_idx if threshold_mode == "heldout" else None
    )
    
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
Examples:
  python -m dqm_pipeline --csv data.csv
  python -m dqm_pipeline --dqm-source crvreco,kpp_crv,file,004-000
  python -m dqm_pipeline --csv data.csv --output ./results --iqr 1.5 --plots
        """
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
            seed=args.seed
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
