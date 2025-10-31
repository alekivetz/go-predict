import os
import argparse
import yaml
import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def safe_segment_name(label) -> str:
    s = str(label)
    return (s.replace(" ", "_")
             .replace("(", "").replace(")", "")
             .replace("<", "lt").replace(">", "gt")
             .replace("-", "_").replace("/", "_")
             .replace("=", ""))

def resolve_segments_from_children(parent_run_id: str) -> Dict[str, str]:
    runs = mlflow.search_runs(experiment_ids=None,
                              filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                              output_format="pandas")
    if runs is None or runs.empty:
        raise RuntimeError(f"No child runs found for parent run id '{parent_run_id}'.")
    mapping = {}
    for _, row in runs.iterrows():
        seg = None
        if "params.segment" in row and pd.notna(row["params.segment"]):
            seg = str(row["params.segment"])
        else:
            rn = str(row.get("tags.mlflow.runName", ""))
            if rn.startswith("segment="):
                seg = rn.split("segment=", 1)[1]
        if seg:
            mapping[seg] = row["run_id"]
    if not mapping:
        raise RuntimeError("Could not infer segment->run_id mapping from child runs.")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLflow-logged segment models on test data.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--parent-run-id", type=str, required=True,
                        help="Parent MLflow run id from training (contains nested segment runs).")
    parser.add_argument("--tracking-uri", type=str, default=None,
                        help="Optional MLflow tracking URI (overrides YAML/env).")
    parser.add_argument("--test-filename", type=str, default="Go_auto_test_data.csv",
                        help="Name of test CSV file under paths.data_dir.")
    parser.add_argument("--log-to-mlflow", action="store_true",
                        help="If set, logs evaluation metrics to a new MLflow run.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # ---- MLflow URI ----
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # ---- Paths / Features / Segmentation ----
    data_dir = cfg["paths"]["data_dir"]
    test_file = os.path.join(data_dir, args.test_filename)
    selected_features = cfg["features"]["selected"]
    target_col = cfg.get("target", "days_on_market")  # fallback to your current default

    seg_cfg = cfg["segmentation"]
    msrp_bins = seg_cfg["msrp_bins"]
    msrp_bins = [float("inf") if (isinstance(b, str) and b.lower() in {".inf", "inf", "infty"}) else b
                 for b in msrp_bins]
    msrp_labels = seg_cfg["msrp_labels"]
    right_closed = seg_cfg.get("right_closed", False)

    # ---- Load test data ----
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    df_test = pd.read_csv(test_file)

    needed = set(selected_features + [target_col, "msrp"])
    missing = [c for c in needed if c not in df_test.columns]
    if missing:
        raise ValueError(f"Missing columns in test data: {missing}")

    df_test["msrp_segment"] = pd.cut(df_test["msrp"],
                                     bins=msrp_bins,
                                     labels=msrp_labels,
                                     right=right_closed)

    y_true_all = df_test[target_col]

    # ---- Resolve child runs and load models ----
    seg_to_run = resolve_segments_from_children(args.parent_run_id)

    preds = pd.Series(index=df_test.index, dtype=float)

    # Optional MLflow eval run
    run_ctx = mlflow.start_run(run_name="evaluation") if args.log_to_mlflow else None
    try:
        for seg_label in msrp_labels:
            mask = (df_test["msrp_segment"] == seg_label)
            if not mask.any():
                continue

            seg_name = safe_segment_name(seg_label)
            child_run_id = seg_to_run.get(str(seg_label))
            if child_run_id is None:
                raise RuntimeError(f"No child run found for segment '{seg_label}'.")

            model_uri = f"runs:/{child_run_id}/models/{seg_name}"
            model = mlflow.sklearn.load_model(model_uri)

            X_seg = df_test.loc[mask, selected_features]
            y_pred_seg = model.predict(X_seg)
            preds.loc[mask] = y_pred_seg

            # Segment metrics
            y_true_seg = y_true_all.loc[mask]
            mse = mean_squared_error(y_true_seg, y_pred_seg)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_true_seg, y_pred_seg)
            mae = mean_absolute_error(y_true_seg, y_pred_seg)

            print(f"Segment '{seg_label}': MSE={mse:.2f}, RMSE={rmse:.2f}, "
                  f"MAE={mae:.2f}, R²={r2:.4f} (N={len(y_true_seg)})")

            if run_ctx:
                # Prefix metrics with segment name
                prefix = f"{seg_name}"
                mlflow.log_metric(f"{prefix}_mse", mse)
                mlflow.log_metric(f"{prefix}_rmse", rmse)
                mlflow.log_metric(f"{prefix}_mae", mae)
                mlflow.log_metric(f"{prefix}_r2", r2)

        # Combined metrics
        mask_all = preds.notna()
        y_true_combined = y_true_all.loc[mask_all]
        y_pred_combined = preds.loc[mask_all]

        mse_c = mean_squared_error(y_true_combined, y_pred_combined)
        rmse_c = float(np.sqrt(mse_c))
        r2_c = r2_score(y_true_combined, y_pred_combined)
        mae_c = mean_absolute_error(y_true_combined, y_pred_combined)

        print(f"\nCombined: MSE={mse_c:.2f}, RMSE={rmse_c:.2f}, "
              f"MAE={mae_c:.2f}, R²={r2_c:.4f} (N={len(y_true_combined)})")

        if run_ctx:
            mlflow.log_metric("combined_mse", mse_c)
            mlflow.log_metric("combined_rmse", rmse_c)
            mlflow.log_metric("combined_mae", mae_c)
            mlflow.log_metric("combined_r2", r2_c)

    finally:
        if run_ctx:
            mlflow.end_run()


if __name__ == "__main__":
    main()
