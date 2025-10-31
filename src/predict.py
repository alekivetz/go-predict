# ==========================================================
# Make Predictions (using MLflow-logged models)
# ==========================================================
# This script loads trained models directly from MLflow instead of local .joblib files.
# It reads configuration from a YAML file, segments the test data by MSRP, 
# loads the correct MLflow model for each segment, and saves predictions.
# ==========================================================

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict

import mlflow
import mlflow.sklearn


# ==========================================================
# Command-line Arguments
# ==========================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Run GoAuto predictions using YAML config (MLflow models)."
    )

    # Path to the config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/predict_config.yaml",
        help="Path to prediction YAML config file."
    )

    # Optional runtime overrides (these can override YAML paths)
    parser.add_argument("--data_dir", type=str, help="Override data directory path.")
    parser.add_argument("--test_filename", type=str, help="Override test CSV filename.")
    parser.add_argument("--predictions_filename", type=str, help="Override predictions output filename.")

    # MLflow parameters
    parser.add_argument("--parent-run-id", type=str, required=True,
                        help="Parent MLflow run id from training (contains nested segment runs).")
    parser.add_argument("--tracking-uri", type=str, default=None,
                        help="Optional MLflow tracking URI (overrides YAML/env).")
    return parser.parse_args()


# ==========================================================
# Configuration Loader
# ==========================================================
def load_cfg(path: str) -> dict:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ==========================================================
# Helper Functions
# ==========================================================
def safe_segment_name(label) -> str:
    """
    Clean up a segment label to make it a safe model name for MLflow artifact paths.
    Example: 'Low (<33.6k)' -> 'Low_lt33.6k'
    """
    s = str(label)
    return (s.replace(" ", "_")
             .replace("(", "").replace(")", "")
             .replace("<", "lt").replace(">", "gt")
             .replace("-", "_").replace("/", "_")
             .replace("=", ""))


def resolve_segments_from_children(parent_run_id: str) -> Dict[str, str]:
    """
    Retrieve all child runs for the given parent MLflow run ID.
    Returns a mapping: {segment_label -> child_run_id}
    Each child run corresponds to one segment model logged in MLflow.
    """
    runs = mlflow.search_runs(
        experiment_ids=None,
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        output_format="pandas",
    )
    if runs is None or runs.empty:
        raise RuntimeError(f"No child runs found for parent run id '{parent_run_id}'.")

    mapping = {}
    for _, row in runs.iterrows():
        seg = None
        # Try to get the segment label from the params
        if "params.segment" in row and pd.notna(row["params.segment"]):
            seg = str(row["params.segment"])
        else:
            # Fallback: infer from run name (e.g. "segment=Low (<33.6k)")
            rn = str(row.get("tags.mlflow.runName", ""))
            if rn.startswith("segment="):
                seg = rn.split("segment=", 1)[1]

        if seg:
            mapping[seg] = row["run_id"]

    if not mapping:
        raise RuntimeError("Could not infer segment->run_id mapping from child runs.")
    return mapping


# ==========================================================
# Core Steps
# ==========================================================
def load_data_and_segment(test_file: str, msrp_bins, msrp_labels, right_closed: bool) -> pd.DataFrame:
    """Load the test dataset and assign MSRP-based segments."""
    print(f"Loading test data from {test_file}...")
    try:
        df_test = pd.read_csv(test_file)
    except FileNotFoundError as e:
        raise SystemExit(f"Error: Test file not found at {test_file}.") from e

    if "msrp" not in df_test.columns:
        raise ValueError("Column 'msrp' not found in test data; required for segmentation.")

    # Segment cars into MSRP bins
    df_test["msrp_segment"] = pd.cut(
        df_test["msrp"], bins=msrp_bins, labels=msrp_labels, right=right_closed
    )
    print("Test data loaded and segmented successfully.")
    return df_test


def generate_predictions(df_test: pd.DataFrame,
                         selected_features: list,
                         msrp_labels: list,
                         seg_to_run: Dict[str, str],
                         predictions_file: str):
    """
    Load MLflow models for each MSRP segment and generate predictions.
    Saves final predictions as a CSV file.
    """
    # Ensure test data contains all necessary features
    missing = [c for c in selected_features if c not in df_test.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in test data: {missing}")

    # Prepare a placeholder Series for predictions
    y_pred_test = pd.Series(index=df_test.index, dtype=float)

    for segment in msrp_labels:
        mask = (df_test["msrp_segment"] == segment)
        if not mask.any():
            continue  # Skip if no rows belong to this segment

        seg_name = safe_segment_name(segment)
        child_run_id = seg_to_run.get(str(segment))
        if child_run_id is None:
            raise RuntimeError(f"No MLflow child run found for segment '{segment}'.")

        # Construct MLflow model URI (matches how it was logged during training)
        model_uri = f"runs:/{child_run_id}/models/{seg_name}"

        # Load model from MLflow
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded MLflow model for segment '{segment}' from {model_uri}")

        # Generate predictions for this segment
        X_test_seg = df_test.loc[mask, selected_features]
        y_pred = model.predict(X_test_seg)
        y_pred_test.loc[mask] = y_pred

    # Save combined predictions
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    df_predictions = pd.DataFrame({
        "days_on_market_pred": y_pred_test.round().astype("Int64")
    })
    df_predictions.to_csv(predictions_file, index=True)
    print(f"\n✅ Predictions saved to {predictions_file}.")


# ==========================================================
# Main Entry Point
# ==========================================================
def main():
    args = get_args()
    cfg = load_cfg(args.config)

    # --- Configure MLflow tracking URI ---
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # --- Resolve paths from YAML / CLI ---
    data_dir = args.data_dir or cfg["paths"]["data_dir"]
    test_filename = args.test_filename or cfg["paths"].get("test_filename", "Go_auto_test_data.csv")
    predictions_filename = args.predictions_filename or cfg["paths"].get("predictions_filename", "predictions.csv")

    test_file = os.path.join(data_dir, test_filename)
    predictions_file = os.path.join(data_dir, predictions_filename)

    # --- Load feature & segmentation settings ---
    selected_features = cfg["features"]["selected"]
    msrp_bins = cfg["segmentation"]["msrp_bins"]
    # Replace textual "inf" with real infinity values
    msrp_bins = [float("inf") if (isinstance(b, str) and b.lower() in {".inf", "inf", "infty"}) else b
                 for b in msrp_bins]
    msrp_labels = cfg["segmentation"]["msrp_labels"]
    right_closed = cfg["segmentation"].get("right_closed", False)

    # --- Summary of configuration ---
    print("\n===== PREDICT CONFIG (MLflow) =====")
    print(f"Config file:          {args.config}")
    print(f"Data directory:       {data_dir}")
    print(f"Test file:            {test_file}")
    print(f"Predictions file:     {predictions_file}")
    print(f"Features (n={len(selected_features)}): {selected_features}")
    print(f"MSRP bins:            {msrp_bins}")
    print(f"MSRP labels:          {msrp_labels}")
    print(f"right_closed:         {right_closed}")
    print(f"Parent run id:        {args.parent_run_id}")
    print(f"MLflow URI:           {mlflow.get_tracking_uri()}")
    print("===================================\n")

    # --- Load and segment test data ---
    df_test = load_data_and_segment(
        test_file=test_file,
        msrp_bins=msrp_bins,
        msrp_labels=msrp_labels,
        right_closed=right_closed
    )

    # --- Get child run IDs for each MSRP segment ---
    seg_to_run = resolve_segments_from_children(args.parent_run_id)

    # --- Predict using MLflow models ---
    generate_predictions(
        df_test=df_test,
        selected_features=selected_features,
        msrp_labels=msrp_labels,
        seg_to_run=seg_to_run,
        predictions_file=predictions_file
    )


# ==========================================================
# Script Entry
# ==========================================================
if __name__ == "__main__":
    main()
