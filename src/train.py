"""
train.py — GoAuto model training with MLflow logging

- Trains one RandomForestRegressor per MSRP segment using configs/train_config.yaml
- Supports CLI overrides for paths
- Logs:
    * Full YAML config (as flattened params + artifact)
    * Metrics (R², RMSE, MAE) per segment
    * Trained models (artifacts + MLflow model)
- Enables mlflow.sklearn.autolog() for extra automatic logging
"""

import os
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- MLflow ---
import mlflow
import mlflow.sklearn


# ===============================
# Helpers
# ===============================
def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_args():
    parser = argparse.ArgumentParser(
        description="Train RandomForest models per MSRP segment (GoAuto) using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",  # run from repo root: python src/train.py
        help="Path to training configuration YAML file.",
    )
    # runtime overrides
    parser.add_argument("--data_dir", type=str, help="Override data directory path.")
    parser.add_argument("--train_filename", type=str, help="Override training CSV filename.")
    parser.add_argument("--model_dir", type=str, help="Override model directory path.")
    return parser.parse_args()


def safe_segment_name(label) -> str:
    """Make a filesystem-safe segment name."""
    s = str(label)
    return (
        s.replace(" ", "_")
         .replace("(", "").replace(")", "")
         .replace("<", "lt").replace(">", "gt")
         .replace("-", "_").replace("/", "_")
         .replace("=", "")
         .replace(".", "_") # Avoid invalid MLflow run names
    )


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts to a single level so mlflow.log_params can accept them.
    Example: {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            # Only log simple types as params; cast lists to string.
            if isinstance(v, (list, tuple, set)):
                items[new_key] = str(list(v))
            else:
                items[new_key] = v
    return items


# ===============================
# Data loading
# ===============================
def load_ready_data(train_file: str,
                    msrp_bins,
                    msrp_labels,
                    right_closed: bool,
                    selected_features):
    """
    Load preprocessed CSV created by preprocess.py, ensure required columns exist,
    and (if needed) compute 'msrp_segment'.
    """
    print(f"Loading training data from {train_file}")
    try:
        df = pd.read_csv(train_file)
    except FileNotFoundError as e:
        raise SystemExit(
            f"ERROR: Training file not found at {train_file}. "
            f"Make sure preprocess.py produced this file."
        ) from e

    # Validate required columns
    needed = set(selected_features + ["msrp", "days_on_market"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed data: {missing}")

    # Add msrp_segment only if preprocess didn't create it
    if "msrp_segment" not in df.columns:
        df["msrp_segment"] = pd.cut(
            df["msrp"], bins=msrp_bins, labels=msrp_labels, right=right_closed
        )
        print("Added 'msrp_segment' based on MSRP bins/labels from config.")
    else:
        print("'msrp_segment' already present — not recomputing.")

    return df


# ===============================
# Training
# ===============================
def train_and_save_per_segment(df_train: pd.DataFrame,
                               model_dir: str,
                               selected_features,
                               msrp_labels,
                               best_params: dict):
    """
    Train one RandomForestRegressor per MSRP segment and save to disk.
    Also logs params, metrics, and artifacts to MLflow (nested runs).
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving trained models to {model_dir} ...")

    for segment in msrp_labels:
        seg_df = df_train[df_train["msrp_segment"] == segment]
        seg_name = safe_segment_name(segment)

        if seg_df.empty:
            print(f"No data for segment '{segment}'. Skipping...")
            continue

        X = seg_df[selected_features]
        y = seg_df["days_on_market"]

        # --------- Train & log (nested MLflow run) ----------
        with mlflow.start_run(run_name=f"segment={segment}", nested=True):
            # parameters (explicit)
            mlflow.log_param("segment", str(segment))
            mlflow.log_param("n_rows", int(len(seg_df)))
            mlflow.log_param("n_features", int(len(selected_features)))
            for k, v in best_params.items():
                mlflow.log_param(f"rf__{k}", v)

            # model fit
            model = RandomForestRegressor(**best_params)
            model.fit(X, y)

            # quick in-sample metrics (replace with a true holdout if available)
            preds = model.predict(X)
            r2 = r2_score(y, preds)
            rmse = float(np.sqrt(mean_squared_error(y, preds)))
            mae = mean_absolute_error(y, preds)

            # metrics
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # save locally + log as artifacts and MLflow model
            fname = f"rf_model_{seg_name}.joblib"
            fpath = os.path.join(model_dir, fname)
            joblib.dump(model, fpath)
            mlflow.log_artifact(fpath)
            mlflow.sklearn.log_model(model, name=f'rf_model_{seg_name}')

            print(f"✅ Segment '{segment}': trained on {len(seg_df)} rows "
                  f"(r2={r2:.3f}, rmse={rmse:.3f}, mae={mae:.3f}) → {fpath}")

    print("🎉 Training complete.")


# ===============================
# Main
# ===============================
def main():
    args = get_args()

    # Load config
    cfg = load_yaml(args.config)

    # Resolve paths (CLI overrides YAML)
    try:
        data_dir = args.data_dir or cfg["paths"]["data_dir"]
        train_filename = args.train_filename or cfg["paths"]["train_filename"]
        model_dir = args.model_dir or cfg["paths"]["model_dir"]
    except KeyError as e:
        raise KeyError(f"Missing 'paths' key in config for {e}") from e

    train_file = os.path.join(data_dir, train_filename)

    # Other settings
    try:
        selected_features = cfg["features"]["selected"]
        msrp_bins = cfg["segmentation"]["msrp_bins"]
        msrp_labels = cfg["segmentation"]["msrp_labels"]
        right_closed = cfg["segmentation"].get("right_closed", False)
        best_params = cfg["model"]["best_params"]
    except KeyError as e:
        raise KeyError(f"Missing required key in config: {e}") from e

    # ---- MLflow configuration ----
    # Priority: env vars > YAML > default local ./mlruns
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri"))
    experiment = mlflow_cfg.get("experiment", "GoAuto")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    # Enable autologging (still keep explicit logs for clarity)
    mlflow.sklearn.autolog(log_models=True)

    # Summary
    print("\n===== TRAIN CONFIG =====")
    print(f"Config file:       {args.config}")
    print(f"Data directory:    {data_dir}")
    print(f"Training file:     {train_file}")
    print(f"Model directory:   {model_dir}")
    print(f"Features (n={len(selected_features)}): {selected_features}")
    print(f"MSRP bins:         {msrp_bins}")
    print(f"MSRP labels:       {msrp_labels}")
    print(f"right_closed:      {right_closed}")
    print(f"RF params:         {best_params}")
    print(f"MLflow URI:        {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {experiment}")
    print("========================\n")

    # Parent MLflow run
    with mlflow.start_run(run_name="GoAuto_RF_training"):
        # Log entire config as artifact + flattened params
        mlflow.log_dict(cfg, "train_config.yaml")
        mlflow.log_params(flatten_dict(cfg))

        # Pipeline
        df_train = load_ready_data(
            train_file=train_file,
            msrp_bins=msrp_bins,
            msrp_labels=msrp_labels,
            right_closed=right_closed,
            selected_features=selected_features,
        )

        train_and_save_per_segment(
            df_train=df_train,
            model_dir=model_dir,
            selected_features=selected_features,
            msrp_labels=msrp_labels,
            best_params=best_params,
        )


if __name__ == "__main__":
    main()
