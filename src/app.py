"""
GoPredict REST API
==============================================================
This is a flask-bassed REST API that serves three regression models
to predict vehicle market duration (days_on_market) by MSRP segment.

Segments:
    - Low (< 33.6k)
    - Medium (33.6k–61.5k)
    - High (> 61.5k)

Key features:
    - Health check endpoint
    - Home endpoint with input examples
    - Model prediction endpoints (/v1/predict and /v2/predict)
    - Swagger documentation via Flasgger

"""

# ============================================================
# =============== 1. IMPORTS & CONFIGURATION =================
# ============================================================

import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import psutil, threading, time

from src.utils.helper_functions import configure_logging

loggers = configure_logging()   
logger = loggers['api']
logger.info('Imported api.py and initialized "api" logger.')

# Initialize
app = Flask(__name__) 

# ============================================================
# =============== 2. PROMETHEUS ==============================
# ============================================================

# Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'GoPredict ML API', version='1.0.0', app_name='go-predict-api')

# Custom metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of ML predictions made',
    ['model_version', 'prediction_result', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Time spent processing ML predictions (seconds)',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

memory_usage_gauge = Gauge('app_memory_usage_bytes', 'Memory usage of the GoPredict API')
cpu_usage_gauge = Gauge('app_cpu_usage_percent', 'CPU usage of the GoPredict API')

def monitor_system():
    """Background thread to monitor system resources every 15 seconds."""
    while True:
        try:
            process = psutil.Process(os.getpid())
            memory_usage_gauge.set(process.memory_info().rss)
            cpu_usage_gauge.set(process.cpu_percent(interval=1))
        except Exception:
            pass  # Ignore errors in background monitoring
        time.sleep(15)

# ============================================================
# =============== 3. FLASK APP CONFIGURATION =================
# ============================================================

# Configure Flasgger for API documentation
swagger = Swagger(app, template={
    "info": {
        "title": "GoPredict REST API",
        "description": "A lightweight API for predicting Days on Market (DOM) using segmented machine learning models.",
        "version": "1.0.0"
    }
})


# ============================================================
# =============== 4. LOAD MODELS AND ARTIFACTS ===============
# ============================================================

# Global placeholders for models and pipeline
models = {}
pipeline = None
model_v1 = None
model_v2 = None

def load_models_from_gcs():
    """Downloads models and pipeline from GCS on first request."""
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
    import tempfile, joblib, os, google.auth

    BUCKET_NAME = "go-predict-artifacts-go-predict"
    GCS_PATH = "gopredict"

    try:
        client = storage.Client.from_service_account_json('/secrets/gcp.json')
        logger.info('Authenticated with GCP with service account JSON.')
    except DefaultCredentialsError as e:
        logger.error(f"GCP credentials not found: {e}")
        raise

    tmp_dir = tempfile.mkdtemp()
    logger.info(f"📁 Temporary folder created at {tmp_dir}")

    # -------------------------------------------------------------------
    # FIXED DOWNLOAD FUNCTION
    # -------------------------------------------------------------------
    def download(blob_name, local_path):
        """Download a file from GCS to a local path with error handling."""
        try:
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path, timeout=60)
            logger.info(f"✅ Downloaded {blob_name} → {local_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download {blob_name}: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------
    # Download pipeline
    # -------------------------------------------------------------------
    local_pipeline = os.path.join(tmp_dir, "preprocessing_pipeline.pkl")
    download(f"{GCS_PATH}/data/processed/preprocessing_pipeline.pkl", local_pipeline)
    pipeline = joblib.load(local_pipeline)

    # -------------------------------------------------------------------
    # Download models
    # -------------------------------------------------------------------
    models = {}
    model_blobs = {
        "Low": f"{GCS_PATH}/models/Low_lt33_6k_model.joblib",
        "Medium": f"{GCS_PATH}/models/Medium_33_6k_61_5k_model.joblib",
        "High": f"{GCS_PATH}/models/High_gt61_5k_model.joblib",
    }

    for seg, blob_path in model_blobs.items():
        try:
            local_model = os.path.join(tmp_dir, f"{seg}.joblib")
            logger.info(f"Downloading {seg} model from GCS...")
            download(blob_path, local_model)
            models[seg] = joblib.load(local_model)
            logger.info(f"✅ Loaded {seg} model from GCS.")
        except Exception as e:
            logger.error(f"❌ Failed to load {seg} model from GCS: {e}", exc_info=True)

    logger.info(f"✅ All models and pipeline successfully loaded from GCS ({len(models)} models).")
    return pipeline, models


def preload_models():
    global pipeline, models, model_v1, model_v2
    logger.info("Pre-loading models from GCS at startup...")
    try:
        pipeline, models = load_models_from_gcs()
        model_v1 = models["High"]
        model_v2 = models["Medium"]
        logger.info("Models successfully loaded from GCS.")
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")
        pipeline, models = None, {}

# ============================================================
# =============== 5. FEATURE DEFINITIONS =====================
# ============================================================

# Core features required by the API for prediction
REQUIRED_FEATURES = [
    "month_listed", "number_price_changes", "discount",
    "model", "mileage", "price_imputed", "make",
    "msrp", "years_on_market", "wheelbase_from_vin"
]

# Data type classification for validation
NUMERICAL_FEATURES = [
    "month_listed", "number_price_changes", "discount",
    "mileage", "price_imputed", "msrp",
    "years_on_market", "wheelbase_from_vin"
]

CATEGORICAL_FEATURES = [
    "model", "make"
]


# ============================================================
# =============== 6. HELPER FUNCTIONS ========================
# ============================================================

def assign_segment(msrp: float) -> str:
    """Determine MSRP segment based on thresholds."""
    if msrp < 33650:
        return 'Low'
    elif msrp <= 61500:
        return 'Medium'
    else:
        return 'High'
    

def validate_input(data):
    """Ensure all required features exist and are of the right type."""
    missing = [feat for feat in REQUIRED_FEATURES if feat not in data]
    if missing:
        return f'Missing required features: {", ".join(missing)}', 400

    for k, v in data.items():
        if k in CATEGORICAL_FEATURES and not isinstance(v, str):    
            return f'Feature "{k}" must be a string.', 400
        if k in REQUIRED_FEATURES and k not in CATEGORICAL_FEATURES:
            if not isinstance(v, (int, float)):
                return f'Feature "{k}" must be a number.', 400
    return None, 200

# ============================================================
# =============== 7. ROUTES: HEALTH & HOME ===================
# ============================================================

@app.route('/health', methods=['GET'])  
def health_check():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: API is alive and running.
        schema:
          id: health_status
          properties:
            status:
              type: string
              example: "ok"
    """
    return jsonify({'status': 'ok'})


@app.route('/gopredict_home', methods=['GET'])
def home():
    """
    Home Endpoint
    Provides documentation and expected JSON input format.
    ---
    responses:
      200:
        description: API documentation.
        schema:
          id: home_page
          properties:
            message:
              type: string
            endpoints:
              type: object
            example_input:
              type: object
    """
    example_input = {
        'month_listed': 1,
        'number_price_changes': 0,
        'discount': 0,
        'model': 'Civic',
        'mileage': 42000,
        'price_imputed': 29000.0,
        'make': 'Honda',
        'msrp': 36000,
        'years_on_market': 7,
        'wheelbase_from_vin': 106.3
    }

    return jsonify({
        'message': 'Welcome to the GoPredict API!',
        'api_documentation': 'Visit /apidocs for the interactive Swagger UI.',
        'endpoints': {
            'health': '/health',
            'home': '/gopredict_home',
            'predict': 'predict'
        },
        'segments': {
            'Low': 'MSRP < 33.6k',
            'Medium': 'MSRP 33.6k–61.5k',
            'High': 'MSRP > 61.5k'
        },
        'example_input': example_input
    })

# ============================================================
# =============== 8. PREDICTION ENDPOINTS ====================
# ============================================================

def predict_with_model(model, version_label, data):
    """Reusable helper for v1/v2 endpoints."""
    if not data:
        return jsonify(error='No input data.'), 400
    
    records = data if isinstance(data, list) else [data]
    preds = []

    # Start timer 
    start_time = time.time()
    try: 
        for record in records:
            msg, code = validate_input(record)
            if msg:
                return jsonify(error=msg), code
            df = pd.DataFrame([record], columns=REQUIRED_FEATURES)
            pred = float(model.predict(df)[0])
            preds.append({
                'model_version': version_label,
                'predicted_days_on_market': pred
            })
        
        # Success metrics
        prediction_counter.labels(
            model_version=version_label,
            prediction_result='success',
            status='success'
        ).inc()

        duration = time.time() - start_time
        prediction_latency.labels(model_version=version_label).observe(duration)

        return jsonify({
            'total_records': len(preds),
            'predictions': preds
        }), 200
    except Exception as e:
        # Record failure metrics
        prediction_counter.labels(
            model_version=version_label,
            prediction_result='failure',
            status='error'
        ).inc()
        logger.error(f'Prediction failed for version {version_label}: {e}')
        return jsonify(error=str(e)), 500


# High segment model - best performing
@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """
    Version 1 Prediction Endpoint
    ---
    tags:
      - prediction
    description: Predicts days_on_market using model_v1 (best performing segment model).
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            month_listed:         {type: integer, example: 8}
            number_price_changes: {type: integer, example: 1}
            discount:             {type: number,  example: 4200.0}
            model:                {type: string,  example: "Corolla"}
            mileage:              {type: number,  example: 31000}
            price_imputed:        {type: number,  example: 45000.0}
            make:                 {type: string,  example: "Toyota"}
            msrp:                 {type: number,  example: 48000.0}
            years_on_market:      {type: integer, example: 5}
            wheelbase_from_vin:   {type: number,  example: 106.5}
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input data
      500:
        description: Models or pipeline not loaded
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify(error='Invalid input data.'), 400
    return predict_with_model(model_v1, "v1", data)

# Medium segment model - second-best performing
@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """
    Version 2 Prediction Endpoint
    ---
    tags:
      - prediction
    description: Predicts days_on_market using model_v2 (second-best performing segment model).
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            month_listed:         {type: integer, example: 8}
            number_price_changes: {type: integer, example: 1}
            discount:             {type: number,  example: 4200.0}
            model:                {type: string,  example: "Corolla"}
            mileage:              {type: number,  example: 31000}
            price_imputed:        {type: number,  example: 45000.0}
            make:                 {type: string,  example: "Toyota"}
            msrp:                 {type: number,  example: 48000.0}
            years_on_market:      {type: integer, example: 5}
            wheelbase_from_vin:   {type: number,  example: 106.5}
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input data
      500:
        description: Models or pipeline not loaded
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify(error='Invalid input data.'), 400
    return predict_with_model(model_v2, "v2", data)

# Overall prediction - automatically selects the correct model based on MSRP
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction Endpoint
    Predicts days_on_market using the appropriate MSRP segment model.
    ---
    tags: 
      - prediction  
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body    
        name: body
        required: true
        description: >
          Provide one or more vehicle records in JSON format.
          The correct model (Low/Medium/High) will be selected automatically.
        schema:
          type: object
          properties:
            month_listed:         {type: integer, example: 8}
            number_price_changes: {type: integer, example: 1}
            discount:             {type: number,  example: 4200.0}
            model:                {type: string,  example: "Corolla"}
            mileage:              {type: number,  example: 31000}
            price_imputed:        {type: number,  example: 45000.0}
            make:                 {type: string,  example: "Toyota"}
            msrp:                 {type: number,  example: 48000.0}
            years_on_market:      {type: integer, example: 5}
            wheelbase_from_vin:   {type: number,  example: 106.5}
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input data
      500:
        description: Models or pipeline not loaded
    """

    # Ensure models were loaded at startup
    if not models or pipeline is None:
        return jsonify(error='Models or pipeline not loaded.'), 500
    
    # Parse JSON input
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify(error='Invalid input data.'), 400

    if not data:
        return jsonify(error='No input data.'), 400
    
    # Normalize input
    records = data if isinstance(data, list) else [data]
    predictions = []

    # Iterate over records and predict
    for record in records:
        msg, code = validate_input(record)
        if msg:
            prediction_counter.labels(
                model_version='auto',
                prediction_result='invalid_input',
                status='error'
            ).inc()
            return jsonify(error=msg), code

        seg = assign_segment(record['msrp'])
        model = models.get(seg)
        if model is None:
            prediction_counter.labels(
                model_version=seg,
                prediction_result='no_model',
                status='error'
            ).inc()
            return jsonify(error=f'No model found for segment "{seg}".'), 500

        df = pd.DataFrame([record], columns=REQUIRED_FEATURES)

        start_time = time.time()
        try:
            pred = float(model.predict(df)[0])
            prediction_counter.labels(
                model_version=seg,
                prediction_result='success',
                status='success'
            ).inc()
            prediction_latency.labels(model_version=seg).observe(time.time() - start_time)
        except Exception as e:
            prediction_counter.labels(
                model_version=seg,
                prediction_result='error',
                status='error'
            ).inc()
            return jsonify(error=str(e)), 500

        predictions.append({
            'segment': seg,
            'predicted_days_on_market': pred
        })

    return jsonify({
        'total_records': len(predictions),
        'predictions': predictions
    }), 200


# ============================================================
# =============== 9. MAIN ENTRY POINT ========================
# ============================================================

if __name__ == '__main__':
    # Preload models from GCS
    logger.info("Preloading models from GCS...")
    preload_models()

    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()

    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting GoPredict API on port {port}...")
    app.run(host='0.0.0.0', port=port)