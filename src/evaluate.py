"""
GoPredict Model Evaluation Script
==============================================================
This module evaluates trained MLflow models (segmented by MSRP)
using the test dataset.

Key features:
    - Loads configuration and segmentation setup from YAML.
    - Retrieves MLflow child runs associated with a parent training run.
    - Loads each segment model, predicts on the corresponding test subset.
    - Computes and logs performance metrics (MSE, RMSE, MAE, R²).
    - Optionally logs metrics back to MLflow for tracking.

Usage (CLI):
    python -m src.evaluate --parent-run-id <MLFLOW_RUN_ID> [--log-to-mlflow]

"""

# ============================================================
# =============== 1. IMPORTS & CONFIGURATION =================
# ============================================================

# --- Standard Libraries ---
import argparse
import os
from typing import Dict

# --- Third-party Libraries ---
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from src.utils.helper_functions import load_yaml_config, configure_logging, safe_segment_name

loggers = configure_logging()   
logger = loggers['evaluate']
logger.info('Imported evaluate.py and initialized "evaluate" logger.')


# ============================================================
# =============== 2. CLI ARGUMENT PARSER =====================
# ============================================================

def get_args(): 
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate GoPredict MLflow models per MSRP segment.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to the YAML config file. Default: configs/train_config.yaml'
        )
    parser.add_argument(
        '--parent-run-id',
        type=str,
        required=True,
        help='Parent MLflow run id from training (contains nested segment runs).'
        )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='Optional MLflow tracking URI (overrides YAML/env).'
        )
    parser.add_argument(
        '--log-to-mlflow',
        action='store_true',
        help='Log metrics to a new MLflow run.'
        )

    args = parser.parse_args()
    logger.info(f'Evaluation CLI arguments: {args}')
    return args

# ============================================================
# =============== 3. HELPER FUNCTIONS ========================
# ============================================================

def resolve_segments_from_children(parent_run_id: str) -> Dict[str, str]:
    """
    Retreive all MLflow child runs associated with a parent training run.
    """

    logger.info(f'Resolving segments from children of {parent_run_id}...')

    runs = mlflow.search_runs(
        experiment_ids=None,
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
        output_format='pandas'
    )
    if runs is None or runs.empty:
        logger.error(f'No child runs found for parent run {parent_run_id}.')
        raise RuntimeError(f'No child runs found for parent run {parent_run_id}.')
    
    mapping = {}
    for _, row in runs.iterrows():
        seg = None

        # Try to get the segment label from the params
        if 'params.segment' in row and pd.notna(row['params.segment']):
            seg = str(row['params.segment'])
        else:
            # Fallback: infer from run name
            rn = str(row.get('tags.mlflow.runName', ''))
            if rn.startswith('segment='):
                seg = rn.split('segment=', 1)[1]
    
        if seg:
            mapping[seg] = row['run_id']
    
    if not mapping:
        logger.error('Could not infer segment names from child runs.')
        raise RuntimeError('Could not infer segment names from child runs.')
    
    logger.info(f'Resolved segment to run mapping for {len(mapping)} segments: {mapping}')
    return mapping

# ============================================================
# =============== 4. MAIN ENTRY POINT ========================
# ============================================================

def main():
    """
    Evaluate per-segment models and compute metrics.
    """
    
    args = get_args()
    config = load_yaml_config(args.config)

    # MLflow setup
    mlflow_config = config.get('mlflow', {})
    # tracking_uri = args.tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or mlflow_config.get('tracking_uri')
    tracking_uri = (
        args.tracking_uri
        or os.getenv('MLFLOW_TRACKING_URI')
        or f"file://{os.path.abspath('mlruns')}"
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('GoAuto_RF_training')
        logger.info(f'MLflow tracking URI set to {tracking_uri}')

    # Paths and parameters
    data_dir = config['paths']['data_dir']
    test_file = os.path.join(data_dir, config['paths']['test_filename'])
    selected_features = config['features']['selected']  
    target = config['features']['target']

    msrp_bins = config['segmentation']['msrp_bins']
    # Replace textual inf with real infinity values
    msrp_bins = [float(b) if b != '.inf' else np.inf for b in msrp_bins]
    msrp_labels = config['segmentation']['msrp_labels']
    right_closed = config['segmentation'].get('right_closed', False)

    # Load data
    try:
        df_test = pd.read_csv(test_file)
        logger.info(f'Data loaded successfully from {test_file}. Shape: {df_test.shape}')
    except Exception as e:
        logger.error(f'Failed to load data from {test_file}: {e}')
        raise SystemExit(f'ERROR: Test data file {test_file} not found.') from e
    
    if target not in df_test.columns:
        raise ValueError(f'Column "{target}" not found in test data.')
    
    df_test['msrp_segment'] = pd.cut(
        df_test['msrp'], bins=msrp_bins, labels=msrp_labels, right=right_closed)
    
    # Retreive MLflow child runs
    seg_to_run = resolve_segments_from_children(args.parent_run_id)

    # Evaluate per segment
    preds = pd.Series(index=df_test.index, dtype='float64')
    y_true = df_test[target]

    # Optionally log this evaluation as a new MLflow run
    run_ctx = mlflow.start_run(run_name='GoAuto_RF_evaluation') if args.log_to_mlflow else None
    try: 
        for seg_label in msrp_labels:
            mask = (df_test['msrp_segment'] == seg_label)
            if not mask.any():
                logger.warning(f'No rows found for segment {seg_label} in test data. Skipping...')
                continue

            seg_name = safe_segment_name(seg_label)
            child_run_id = seg_to_run.get(seg_name) or seg_to_run.get(seg_label)
            if child_run_id is None:
                logger.error(f'No MLflow child run found for segment {seg_label}.')
                raise RuntimeError(f'No MLflow child run found for segment {seg_label}.')
            
            # Load the model for this segment
            model_uri = f'runs:/{child_run_id}/model'
            model = mlflow.sklearn.load_model(model_uri)

            # Predict on this segment subset
            X_seg = df_test.loc[mask, selected_features]
            y_pred_seg = model.predict(X_seg)
            preds.loc[mask] = y_pred_seg

            # Compute metrics
            y_true_seg = y_true.loc[mask]
            mse = mean_squared_error(y_true_seg, y_pred_seg)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_seg, y_pred_seg)
            r2 = r2_score(y_true_seg, y_pred_seg)

            logger.info(f'Segment {seg_label} metrics: MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}')

            # Optionally log metrics to MLflow  
            if run_ctx:
                # Prefix with segment name
                prefix = f'{seg_name}'
                mlflow.log_metric(f'{prefix}_mse', mse)
                mlflow.log_metric(f'{prefix}_rmse', rmse)
                mlflow.log_metric(f'{prefix}_mae', mae)
                mlflow.log_metric(f'{prefix}_r2', r2)
        
        # Combined metrics across all segments  
        mask_all = preds.notna()
        y_true_combined = y_true.loc[mask_all]
        y_pred_combined = preds.loc[mask_all]

        mse_combined = mean_squared_error(y_true_combined, y_pred_combined)
        rmse_combined = np.sqrt(mse_combined)
        mae_combined = mean_absolute_error(y_true_combined, y_pred_combined)
        r2_combined = r2_score(y_true_combined, y_pred_combined)

        logger.info(f'Combined metrics: MSE={mse_combined:.3f}, RMSE={rmse_combined:.3f}, MAE={mae_combined:.3f}, R²={r2_combined:.3f}')

        if run_ctx:
            # Log metrics to MLflow
            mlflow.log_metric('mse', mse_combined)
            mlflow.log_metric('rmse', rmse_combined)
            mlflow.log_metric('mae', mae_combined)
            mlflow.log_metric('r2', r2_combined)

    finally:
        if run_ctx:
            mlflow.end_run()

# ============================================================
# =============== 5. EXECUTION GUARD =========================
# ============================================================

if __name__ == '__main__':
    main()
