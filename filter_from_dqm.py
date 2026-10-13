#!/usr/bin/env python3
"""
Filter static/uninformative columns (e.g., constant 1.0) from DQM Anomaly Dataset.
"""

import sys
import pandas as pd

INPUT_FILE = "dqm_crv_anomaly_dataset_004-000.csv"
OUTPUT_CLEAN_FILE = "dqm_crv_anomaly_dataset_004-000_filtered.csv"


def filter_static_columns(df: pd.DataFrame):
    """Drop feature columns that are constant (incl. all 1.0). Returns (df_clean, dropped_cols)."""
    dropped_cols = []
    kept_cols = []

    # Preserve metadata columns regardless of variance
    meta_cols = {"run", "subrun", "timestamp"}

    for col in df.columns:
        if col in meta_cols:
            kept_cols.append(col)
            continue

        # Convert column to numeric for accurate checking
        numeric_series = pd.to_numeric(df[col], errors="coerce")

        # Check if the column is entirely 1.0 (or within floating-point tolerance)
        is_all_ones = (numeric_series == 1.0).all()

        # Check if the column is completely static/constant across all rows
        is_constant = numeric_series.nunique(dropna=False) <= 1

        if is_all_ones or is_constant:
            dropped_cols.append(col)
        else:
            kept_cols.append(col)

    return df[kept_cols], dropped_cols


def clean_dqm_matrix(input_csv: str, output_csv: str):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")

    df_clean, dropped_cols = filter_static_columns(df)
    kept_cols = list(df_clean.columns)

    print("\n--- Cleaning Summary ---")
    print(f"Kept columns ({len(kept_cols)}): {kept_cols}")
    print(f"\nDropped static/uninformative columns ({len(dropped_cols)}):")
    for col in dropped_cols:
        val_sample = df[col].iloc[0] if not df.empty else "N/A"
        print(f"  - {col} (constant value: {val_sample})")

    # Save cleaned matrix
    df_clean.to_csv(output_csv, index=False)
    print(
        f"\n✓ Saved cleaned matrix ({df_clean.shape[0]} rows, {df_clean.shape[1]} cols) → {output_csv}"
    )


if __name__ == "__main__":
    try:
        clean_dqm_matrix(INPUT_FILE, OUTPUT_CLEAN_FILE)
    except FileNotFoundError:
        print(
            f"Error: Could not find '{INPUT_FILE}'. Run the extraction script first.",
            file=sys.stderr,
        )
