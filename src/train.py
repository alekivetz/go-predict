"""
GoPredict Model Training
==============================================================
This module trains one Random Forest Regressor per MSRP segment
using configs/train_config.yaml.

Key features:
    - Logs into Dagshub MLflow:
        - Flattened YAML config (as params) + full YAML as artifact
        - Per-segment metrics: R2, MAE, RMSE
        - Per-segment model .joblib files as plain artifacts
    - Uses parent run + nested child runs (one per MSRP segment)

Usage (CLI):
    python -m src.train --config configs/train_config.yaml 
"""

# ============================================================
# =============== 1. IMPORTS & CONFIGURATION =================
# ============================================================

# --- Standard Libraries ---
import argparse
import os
import tempfile
import shutil
from typing import Dict, Any

# --- Third-party Libraries ---
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
from mlflow.sklearn import save_model  
from mlflow.models.signature import infer_signature


# --- Internal Imports ---
from src.utils.helper_functions import load_yaml_config, configure_logging, safe_segment_name

loggers = configure_logging()   
logger = loggers['train']
logger.info('Imported train.py and initialized "train" logger.')

# ============================================================
# =============== 2. MLFLOW CONFIG ===========================
# ============================================================

"""
Get the tracking URI from environment variable or use default.
Default is localhost:5000 (useful if we run MLflow directly),
but inside the docker-compose we override it.
"""
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment('GoAuto_RF_training')

logger.info(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
logger.info('MLFLOW experiment set to GoAuto_RF_training.')

# ============================================================
# =============== 3. CONFIG HELPERS ==========================
# ============================================================

def get_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description='Train Random Forest models per MSRP segment using YAML config.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to the YAML config file. Default: configs/train_config.yaml'
    )

    # Optional overrides
    parser.add_argument('--data-dir', type=str, help='Override data directory.')
    parser.add_argument('--train_filename', type=str, help='Override training CSV filename.')
    parser.add_argument('--model_dir', type=str, help='Override model directory path.')
    parser.add_argument('--save', action='store_true', help='Save trained models locally.')

    args = parser.parse_args()
    logger.info(f'Command-line arguments: {args}')
    return args

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary so mlflow.log_params can accept them."""

    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v if not isinstance(v, (list, tuple, set)) else str(list(v))
    return items


# ============================================================
# =============== 4. DATA LOADING ============================
# ============================================================

def load_ready_data(
        train_file: str,
        msrp_bins, 
        msrp_labels,
        right_closed: bool,
        selected_features: list
    ) -> pd.DataFrame:
    """
    Load preprocessed CSV created by preprocess.py, ensure required columns
    exist, and (if needed) compute 'msrp_segment'.
    """
    logger.info(f'Loading training data from {train_file}...')
    try:
        df = pd.read_csv(train_file)
        logger.info(f'Data loaded successfully from {train_file}. Shape: {df.shape}')
    except FileNotFoundError as e:  
        logger.error(f'Failed to load data from {train_file}: {e}')
        raise SystemExit(
            f'ERROR: Training data file {train_file} not found.'
        ) from e
    
    needed = set(selected_features + ['msrp', 'days_on_market'])
    missing = [col for col in needed if col not in df.columns]
    if missing:
        logger.error(f'Missing required columns: {missing}')
        raise ValueError(f'Missing required columns: {missing}')
    
    if 'msrp_segment' not in df.columns:
        df['msrp_segment'] = pd.cut(
            df['msrp'], bins=msrp_bins, labels=msrp_labels, right=right_closed
        )
        logger.info(f'MSRP segment column created based on MSRP bins/labels from config.')
    else:
        logger.info(f'MSRP segment column already exists.')

    return df


# ============================================================
# =============== 5. MODEL TRAINING PER SEGMENT ==============
# ============================================================

def train_and_save_per_segment(
        df_train: pd.DataFrame,
        model_dir: str,
        selected_features: list,
        msrp_labels: list,
        best_params: dict,
        args,
        preprocess
    ):
    """
    Train one Random Forest Regressor per MSRP segment and save/log results.
    Uses a nested MLflow run for each segment. Models are logged as plain artifacts. 
    """
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f'Model directory created: {model_dir}')

    # If we save local copies, ensure the models/ folder exists
    if args.save:
        os.makedirs('models', exist_ok=True)
        logger.info('Local model saving enabled. Models will be saved to models/ folder.')
    
    logger.info('Starting training per MSRP segment...')

    # Get active parent run ID for MLflow
    parent_run = mlflow.active_run()
    parent_id = parent_run.info.run_id if parent_run else None

    for segment in msrp_labels:
        seg_df = df_train[df_train['msrp_segment'] == segment]
        if seg_df.empty:
            logger.warning(f'No data for segment {segment}. Skipping...')
            continue

        seg_name = safe_segment_name(segment)
        X = seg_df[selected_features]
        # Convert integer columns to float64 for MLflow error
        X = X.astype({col: 'float64' for col in X.select_dtypes(include='int').columns})
        y = seg_df['days_on_market']

        logger.info(f'Training segment {segment} on {X.shape[0]} rows...')

        # Start nested MLflow run per segment
        with mlflow.start_run(
            run_name=f'segment_{seg_name}', 
            nested=True, 
            tags={'mlflow.parentRunId': parent_id, 'segment': seg_name}):
            
            # Parameters
            mlflow.log_param('segment', str(segment))
            mlflow.log_param('n_rows', X.shape[0])
            mlflow.log_param('n_features', X.shape[1])
            for k, v in best_params.items():
                mlflow.log_param(f'rf_{k}', v)

            # Preprocess and model pipeline
            pipeline = Pipeline([
                ('preprocess', preprocess),
                ('model', RandomForestRegressor(**best_params))
            ])
            pipeline.fit(X, y)

            # In-sample metrics
            preds = pipeline.predict(X)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rmse = mean_squared_error(y, preds) ** 0.5

            # Log metrics   
            mlflow.log_metric('r2', r2)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('rmse', rmse)

            # Build and log MLflow model artifact
            signature = infer_signature(X, preds)

            # Build and upload the MLflow model folder as artifacts
            temp_dir = tempfile.mkdtemp()
            local_model_dir = os.path.join(temp_dir, 'model_dir')
            save_model(
                sk_model=pipeline,
                path=local_model_dir,
                signature=signature,
                input_example=X.head(3)
            )

            # If the user chose to save the model locally (via --save flag)
            if args.save:
                # Create a file path
                local_save_path = os.path.join('models', f'{seg_name}_model.joblib')
                joblib.dump(pipeline, local_save_path)
                logger.info(f'Local model saved to {local_save_path}')
            
            # Upload the folder to this run under Artifacts/model/
            mlflow.log_artifacts(local_model_dir, artifact_path='model')

            # Store model URI
            run_id = mlflow.active_run().info.run_id
            model_uri = f'runs:/{run_id}/artifacts/model'
            mlflow.log_param('model_uri', model_uri)
            mlflow.log_text(model_uri, 'MODEL_URI.txt')

            # Clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True) 
            logger.info(
                f'Segment {segment} complete! | '
                f'Metrics: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f} | '
                'Model artifacts saved to models/'
            )

    logger.info('All MSRP segments processed successfully. Training complete!')


# ============================================================
# =============== 6. MAIN ENTRY POINT ========================
# ============================================================

def main():
    logger.info('=== GoPredict training script started ===')
    args = get_args()
    config = load_yaml_config(args.config)

    # Resolve paths (CLI overrides config YAML) 
    data_dir = args.data_dir or config['paths']['data_dir']
    train_filename = args.train_filename or config['paths']['train_filename']
    model_dir = args.model_dir or config['paths']['model_dir']
    train_file = os.path.join(data_dir, train_filename)

    # Load other settings
    selected_features = config['features']['selected']
    msrp_bins = config['segmentation']['msrp_bins']
    msrp_labels = config['segmentation']['msrp_labels']
    right_closed = config['segmentation'].get('right_closed', False)
    best_params = config['model']['best_params']

    # Configuration summary
    logger.info(
        '\n=== TRAIN CONFIG SUMMARY ===\n'
        f'Config file:          {args.config}\n'
        f'Data directory:       {data_dir}\n'
        f'Training file:        {train_file}\n'
        f'Model directory:      {model_dir}\n'
        f'Selected features:    {selected_features}\n'
        f'MSRP bins:            {msrp_bins}\n'
        f'MSRP labels:          {msrp_labels}\n'
        f'Right-closed:         {right_closed}\n'
        f'Best hyperparameters: {best_params}\n'
        f'MLflow tracking URI:  {mlflow.get_tracking_uri()}\n'
        '==============================='
    )


    # Parent run: full log config, then train per MSRP segment
    with mlflow.start_run(run_name='GoAuto_RF_training') as parent_run:
        parent_id = parent_run.info.run_id
        logger.info(f'Parent run ID: {parent_id}')
        mlflow.log_param('parent_run_id', parent_id)

        try:
            mlflow.log_dict(config, 'train_config.json')
        except Exception as e:
            logger.warning(f'Skipping artifact upload due to error: {e}')  
        mlflow.log_params(flatten_dict(config))

        # Load dataset and fit preprocessing
        df_train = load_ready_data(
            train_file, msrp_bins, msrp_labels, right_closed, selected_features
        )

        # Build preprocessing pipeline
        possible_cats = ['model', 'make']
        CATEGORICAL = [c for c in possible_cats if c in df_train.columns and c in selected_features]
        NUMERICAL = [c for c in selected_features if c not in CATEGORICAL]

        preprocess = ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                ('numerical', 'passthrough', NUMERICAL)
            ]  
        )

        # Fit the preprocessing pipeline 
        X_all = df_train[selected_features]
        preprocess.fit(X_all)


        # Save preprocessing pipeline
        os.makedirs('data/processed', exist_ok=True)
        pipeline_path = 'data/processed/preprocessing_pipeline.pkl'
        joblib.dump(preprocess, pipeline_path)
        logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

        try:
            mlflow.log_artifact(pipeline_path, artifact_path='preprocess')
        except Exception as e:
            logger.warning(f"Skipping pipeline artifact upload due to: {e}")

        # Log preprocessing artifact to MLflow
        try: 
            mlflow.log_artifact('data/processed/preprocessing_pipeline.pkl', artifact_path='preprocess')
            logger.info('Logged preprocessing pipeline as MLflow artifact under preprocess/')

        except Exception as e:
            logger.warning(f'Skipping artifact upload due to error: {e}')

        # Train models by MSRP segment
        train_and_save_per_segment(
            df_train, model_dir, selected_features, msrp_labels, best_params, args, preprocess
        )

        logger.info('=== GoPredict training script completed! ===')

# ============================================================
# =============== 7. EXECUTION GUARD =========================
# ============================================================

if __name__ == '__main__':
    main()
