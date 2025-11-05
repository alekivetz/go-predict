"""
GoPredict Model Prediction Script
==============================================================
This module generates predictions for vehicle listings using 
MLflow-logged models.

Key features:
    - Loads trained models from MLflow (not from local .joblib files)
    - Segments the test dataset based on MSRP bins
    - Loads the corresponding MLflow model for each segment
    - Generates and saves predictions as a CSV file

Usage (CLI):
    python -m src.predict --parent-run-id <MLFLOW_RUN_ID> 
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

import mlflow
import mlflow.sklearn

# --- Internal Imports ---
from src.utils.helper_functions import load_yaml_config, configure_logging, safe_segment_name

loggers = configure_logging()   
logger = loggers['predict']
logger.info('Imported predict.py and initialized "predict" logger.')


# ============================================================
# =============== 2. CLI ARGUMENT PARSER =====================
# ============================================================

def get_args():
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(
        description='Run GoPredict predictions using YAML config and MLflow models.'
    )

    # YAML config file
    parser.add_argument(
        '--config',
        type=str,
        default='configs/predict_config.yaml',
        help='Path to the YAML config file. Default: configs/predict_config.yaml'
    )

    # Optional overrides
    parser.add_argument('--data-dir', type=str, help='Override data directory.')
    parser.add_argument('--test_filename', type=str, help='Override test CSV filename.')
    parser.add_argument('--predictions_filename', type=str, help='Override predictions output filename.')

    # MLflow tracking and model references
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

    args = parser.parse_args()
    logger.info(f'Prediction CLI arguments: {args}')    
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
# =============== 4. CORE FUNCTIONS ==========================
# ============================================================

def load_data_and_segment(
        test_file: str,
        msrp_bins,
        msrp_labels,
        right_closed: bool,
    ) -> pd.DataFrame:
    """
    Load the test dataset and assign MSRP-based segments.
    """
    logger.info(f'Loading test data from {test_file}...')

    try:
        df_test = pd.read_csv(test_file)
        logger.info(f'Data loaded successfully from {test_file}. Shape: {df_test.shape}')
    except FileNotFoundError as e:
        logger.error(f'Failed to load data from {test_file}: {e}')
        raise SystemExit(f'ERROR: Test data file {test_file} not found.') from e
    
    if 'msrp' not in df_test.columns:
        logger.error('Column "msrp" not found in test data - required for MSRP segmentation.')  
        raise ValueError('Column "msrp" not found in test data - required for MSRP segmentation.')
    
    # Assign segments based on MSRP bins    
    df_test['msrp_segment'] = pd.cut(
        df_test['msrp'], bins=msrp_bins, labels=msrp_labels, right=right_closed
    )

    logger.info(f'MSRP segment column created based on MSRP bins/labels from config.')
    return df_test


def generate_predictions(
        df_test: pd.DataFrame,
        selected_features: list,
        msrp_labels: list,
        seg_to_run: Dict[str, str], 
        predictions_file: str
    ):
    """
    Load MLflow models for each MSRP segment, generate predictions, and save them.    
    """
    logger.info('Starting prediction generation for all segments...')

    # Validate presence of all required features
    missing = [col for col in selected_features if col not in df_test.columns]
    if missing:
        logger.error(f'Missing required features in test data: {missing}')
        raise ValueError(f'Missing required features in test data: {missing}')
    
    # Prepare a placeholder series for predictions
    y_pred_test = pd.Series(index=df_test.index, dtype='float64')

    for segment in msrp_labels:
        mask = (df_test['msrp_segment'] == segment) 
        if not mask.any():
            logger.warning(f'No rows found for segment {segment} in test data. Skipping...') 
            continue

        seg_name = safe_segment_name(segment)
        child_run_id = seg_to_run.get(seg_name) or seg_to_run.get(segment)
        if child_run_id is None:
            logger.error(f'No MLflow child run found for segment {segment}.')
            raise RuntimeError(f'No MLflow child run found for segment {segment}.')
        
        # Construct MLflow model URI
        model_uri = f'runs:/{child_run_id}/model'

        # Load model from MLflow
        logger.info(f'Loading MLflow model for segment {segment} from {model_uri}...')

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f'MLflow model for segment {segment} loaded successfully.')

        # Generate predictions for this segment
        X_test_seg = df_test.loc[mask, selected_features]
        y_pred = model.predict(X_test_seg)
        y_pred_test.loc[mask] = y_pred
        logger.info(f'Generated predictions for segment {segment} successfully on {mask.sum()} rows.')

    # Save combined predictions
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    df_predictions = pd.DataFrame({
        'days_on_market_pred': y_pred_test.round().astype('int64'), 
    })

    df_predictions.to_csv(predictions_file, index=True)
    logger.info(f'Predictions saved to {predictions_file}.')


# ============================================================
# =============== 5. MAIN ENTRY POINT ========================
# ============================================================

def main():
    """
    Main execution flow for model predictions using MLflow-logged models.
    """

    args = get_args()   
    config = load_yaml_config(args.config)

    # Configure MLflow tracking URI
    mlflow_config = config.get('mlflow', {})
    # tracking_uri = args.tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or mlflow_config.get('tracking_uri')
    
    # Configure MLflow tracking URI (hardcoded to match training)
    tracking_uri = (
        args.tracking_uri
        or os.getenv('MLFLOW_TRACKING_URI')
        or f"file://{os.path.abspath('mlruns')}"
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('GoAuto_RF_training')
        logger.info(f'MLflow tracking URI set to {tracking_uri}')
    else:
        logger.info(f'Using existing MLflow tracking URI: {mlflow.get_tracking_uri()}')

    
    # Resolve paths and parameters
    data_dir = args.data_dir or config['paths']['data_dir']
    test_filename = args.test_filename or config['paths']['test_filename']
    predictions_filename = args.predictions_filename or config['paths']['predictions_filename']

    test_file = os.path.join(data_dir, test_filename)
    predictions_file = os.path.join(data_dir, predictions_filename)

    # Load features and segmentation settings
    selected_features = config['features']['selected']
    msrp_bins = config['segmentation']['msrp_bins']
    # Replace textual inf with real infinity values
    msrp_bins = [float(b) if b != '.inf' else np.inf for b in msrp_bins]
    msrp_labels = config['segmentation']['msrp_labels']
    right_closed = config['segmentation'].get('right_closed', False)

    # Display config summary
    logger.info(
        '\n=== PREDICTION CONFIG SUMMARY ===\n'
        f'Config file:          {args.config}\n'
        f'Data directory:       {data_dir}\n'
        f'Test file:            {test_file}\n'
        f'Predictions file:     {predictions_file}\n'
        f'Selected features:    {selected_features}\n'
        f'MSRP bins:            {msrp_bins}\n'
        f'MSRP labels:          {msrp_labels}\n'
        f'Right-closed:         {right_closed}\n'
        f'MLflow tracking URI:  {mlflow.get_tracking_uri()}\n'
        f'Parent run ID:        {args.parent_run_id}\n'  
        "====================================="
    )

    # Load and segment the test data
    df_test = load_data_and_segment(
        test_file, msrp_bins, msrp_labels, right_closed
    )

    # Retreive MLflow child runs
    seg_to_run = resolve_segments_from_children(args.parent_run_id)

    # Generate predictions
    generate_predictions(
        df_test, selected_features, msrp_labels, seg_to_run, predictions_file
    )

    logger.info('=== GoPredict prediction script completed! ===')

# ============================================================
# =============== 6. EXECUTION GUARD =========================
# ============================================================

if __name__ == '__main__':
    main()
