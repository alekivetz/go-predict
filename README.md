# GoPredict: Vehicle Resale Days-on-Market Prediction

## Description

A production-ready machine learning project designed to predict how many days a vehicle will remain on the market before being sold. The system uses a segmented regression approach optimized for high-MSRP vehicles and includes a complete MLOps workflow with preprocessing, training, prediction, model tracking, and containerized deployment.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-3.63.0-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.1-0194E2.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Swagger](https://img.shields.io/badge/Swagger-85EA2D?style=flat&logo=swagger&logoColor=black)](https://swagger.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://go-predict.streamlit.app)

---

## Live Demo

| Service | URL |
| :--- | :--- |
| GoPredict API | https://gopredict-api-5258735120.us-central1.run.app |
| Swagger UI | https://gopredict-api-5258735120.us-central1.run.app/apidocs |
| Streamlit App | https://go-predict.streamlit.app |

---

## Overview

GoPredict predicts days-on-market (DOM) for used vehicles using a segmented Random Forest model. The project includes:

- A modular ML pipeline (preprocessing → training → evaluation → prediction)
- YAML-based configurations
- CLI scripts for all pipeline stages
- Virtual environment support
- MLflow for experiment tracking
- DVC for managing data and model artifacts
- Flask REST API with Swagger UI
- Fully containerized deployment (ML Application + MLflow server)
- Deployed to GCP Cloud Run
- Interactive Streamlit front end

---

## Dataset and Data Source

- **Source:** Dataset provided by Go Auto, one of the largest vehicle dealership networks in Canada.
- **Location:** Edmonton and surrounding areas.
- **Key Features:** mileage, msrp, model, dealer, price changes, discount, etc.
- Dataset not included in repo due to confidentiality. Use `dvc pull` to retrieve artifacts if you have access.

---

## Project Objectives

- Conduct Exploratory Data Analysis (EDA) to understand the dataset and detect trends.
- Develop a regression model to predict days on the market.
- Deploy the model as a REST API.
- Containerize the full system (ML App + MLflow).
- Deploy to GCP Cloud Run with a live Streamlit front end.

---

## Project Structure

All commands must be executed from the project root directory `gopredict`.

```
gopredict/
├── configs/                 # YAML configs for preprocess/train/predict
├── data/                    # Raw & processed datasets (DVC tracked)
├── docs/                    # Documentation, notebooks, screenshots
├── grafana/                 # Grafana monitoring configuration
├── logs/                    # Application logs
├── mlruns/                  # MLflow experiment tracking artifacts
├── models/                  # Model artifacts (DVC tracked)
├── prometheus/              # Prometheus monitoring configuration
├── secrets/                 # Secret configs (not committed)
├── src/
│   ├── app.py               # Flask API
│   ├── preprocess.py        # CLI preprocessing
│   ├── train.py             # CLI training
│   ├── predict.py           # CLI batch/single predictions
│   ├── evaluate.py          # CLI evaluation
│   └── utils/               # Logging, helpers
├── streamlit/               # Streamlit front end
├── .gcloudignore
├── .gitignore
├── API_Documentation.md
├── cloudbuild.yaml          # GCP Cloud Build configuration
├── docker-compose.yml
├── Dockerfile.mlapp         # ML Application container
├── Dockerfile.mlflow        # MLflow Tracking container
├── requirements.txt
└── README.md
```

---

## API Documentation

This project is served via a REST API deployed on GCP Cloud Run. For complete details on all available endpoints, see the [API Documentation](./API_Documentation.md).

For a live interactive API specification, visit the Swagger UI:
`https://gopredict-api-5258735120.us-central1.run.app/apidocs`

---

## Model and Technical Implementation

### Architecture

The final model employs a segmented Random Forest Regressor to address non-linearities across different price points.

| Component | Detail |
| :--- | :--- |
| Model Type | Segmented Random Forest Regressor |
| Segmentation | 3 Tiers (Low, Medium, High) based on MSRP |
| Feature Count | 10 Selected Features (including engineered features like `month_listed`, `discount`) |
| Key Hyperparameters | `n_estimators=350`, `max_depth=35`, `min_samples_split=5` |

### Technologies and Tools

- **Core:** Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)
- **API:** Flask, Swagger UI
- **Front End:** Streamlit
- **Cloud:** GCP Cloud Run, Cloud Build
- **Monitoring:** Prometheus, Grafana
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Containerization:** Docker
- **Visualization:** Power BI
- **Version Control:** GitHub

---

## Pipeline Execution Guide

Navigate to the project root directory `gopredict` and execute the scripts sequentially.

### Step 1 - Install Dependencies

```sh
pip install -r requirements.txt
```

### Step 2 - Pull Models and Pipelines (DVC)

```sh
dvc pull
```

This retrieves the preprocessing pipeline, target encoders, `model_v1.pkl`, and `model_v2.pkl`.

### Step 3 - Run the ML Pipeline

```sh
python src/preprocess.py
python src/train.py
python src/predict.py
python src/evaluate.py
```

---

## Docker Containerization

This project includes two containers:

1. ML Application Container (Flask API + prediction logic)
2. MLflow Tracking Container (experiment logs and artifacts)

```sh
# Build containers
docker build -f Dockerfile.mlapp -t gopredict-mlapp .
docker build -f Dockerfile.mlflow -t gopredict-mlflow .

# Run both containers
docker-compose up --build

# Stop containers
docker-compose down
```

Local services:

| Service | URL |
| :--- | :--- |
| GoPredict API | `http://localhost:5000` |
| Swagger UI | `http://localhost:5000/apidocs` |
| MLflow UI | `http://localhost:5001` |

---

## Dashboard

The Power BI dashboard visualizes inventory aging patterns across three dimensions: vehicle age, price tier, and mileage. It includes interactive filtering by make, custom tooltip pages, and a summary table on each page.

Screenshots are available in [`docs/screenshots/`](./docs/screenshots/).

---

## Key Findings

**Newer Isn't Always Faster**
0-1 year vehicles have the lowest sell-through rate at 64%. Older inventory moves more reliably, with 10+ year vehicles closing at 76%.

**Low Mileage, Long Wait**
Sub-25K km vehicles sit the longest, with 31% aging past 60 days and a sell-through rate of just 64%.

**Luxury Takes Its Time**
Premium and Luxury segments average 64 to 68 days on market and are the least likely to sell within 30 days.

**Mileage Ceiling Matters Less Than You'd Think**
Vehicles with 150K+ km match mid-mileage sell-through rates at 72%, suggesting buyers are less deterred by high mileage than expected.