# GoPredict: Vehicle Resale Days-on-Market Prediction 🚗

## Description
A production-ready machine learning project designed to predict how many days a vehicle will remain on the market before being sold. The system uses a segmented regression approach optimized for high-MSRP vehicles and includes a complete MLOps workflow with preprocessing, training, prediction, model tracking, and containerized deployment.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-3.63.0-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.1-0194E2.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Swagger](https://img.shields.io/badge/Swagger-85EA2D?style=flat&logo=swagger&logoColor=black)](https://swagger.io/)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

## Overview
GoPredict predicts days-on-market (DOM) for used vehicles using a segmented Random Forest model.
The project includes:

- 🏗 A modular ML pipeline (preprocessing → training → evaluation → prediction)
- 📦 YAML-based configurations
- 🧪 CLI scripts for all pipeline stages
- 🐍 Virtual environment support
- 📊 MLflow for experiment tracking
- 📦 DVC for managing data and model artifacts
- 🌐 Flask REST API with Swagger UI
- 🐳 Fully containerized deployment (ML Application + MLflow server)

##  Dataset and Data Source

* **Source:** Dataset provided by Go Auto, one of the largest vehicle dealership networks in Canada.
* **Location:** Edmonton and surrounding areas.
* **Key Features:** mileage, msrp, model, dealer, price changes, discount, etc.
* ⚠ Dataset not included in repo due to confidentiality.
Use dvc pull to retrieve artifacts if you have access.

---

## Project Objectives

* Conduct Exploratory Data Analysis (EDA) to understand the dataset and detect trends.
* Develop a regression model to predict days on the market.
* Deploy the model as a REST API
* Containerize the full system (ML App + MLflow)

### Project Structure

This project follows a standard machine learning repository layout. **All commands must be executed from the Project Root Directory gopredict.**

```bash
    gopredict/
    ├── configs/                 # YAML configs for preprocess/train/predict
    ├── data/                    # Raw & processed datasets (DVC tracked)
    ├── docs/                    # Documentation + notebooks
    ├── models/ (DVC)            # Model artifacts
    ├── src/
    │   ├── app.py               # Flask API
    │   ├── preprocess.py        # CLI preprocessing
    │   ├── train.py             # CLI training
    │   ├── predict.py           # CLI batch/single predictions
    │   ├── evaluate.py          # CLI evaluation
    │   └── utils/               # Logging, helpers
    ├── tests/                   # Automated tests
    ├── Dockerfile.mlapp         # ML Application container
    ├── Dockerfile.mlflow        # MLflow Tracking container
    ├── docker-compose.yml       # Multi-container orchestration
    ├── requirements.txt
    └── README.md         
```

---

## API Documentation

This project is served via a REST API. For complete details on installation, running, and all available endpoints, please see the **[API Documentation](../API_Documentation.md)**.

For a live, interactive API specification (Swagger UI), run the server and navigate to:
`http://127.0.0.1:5000/apidocs/`

---

## Model and Technical Implementation

### Architecture
The final model employs a **Segmented Random Forest Regressor** to address non-linearities across different price points.

| Component | Detail |
| :--- | :--- |
| **Model Type** | Segmented Random Forest Regressor |
| **Segmentation** | 3 Tiers (Low, Medium, High) based on **MSRP** |
| **Feature Count** | **10 Selected Features** (including engineered features like `month_listed`, `discount`) |
| **Key Hyperparameters** | `n_estimators=350`, `max_depth=35`, `min_samples_split=5` |

### Technologies and Tools
* **Core:** Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)
* **Analysis:** Jupyter Notebook
* **Version Control:** GitHub
* **Model Persistence:** `joblib`
* **Visualization:** Power BI


## ⚡ Pipeline Execution Guide

To run the full pipeline, navigate to the **Project Root Directory gopredict** and execute the scripts sequentially:

### Step 1 - Install Dependencies
pip install -r requirements.txt

### Step 2 - Pull Models and Pipelines (DVC)
dvc pull

This retrieves:

- preprocessing pipeline
- target encoders
- model_v1.pkl
- model_v2.pkl

### Step 3 - Run the ML Pipeline

**Preprocess**
```sh
python src/preprocess.py
```

**Train**
```sh
python src/train.py
```

**Predict**
```sh
python src/predict.py
```

**Evaluate**
```sh
python src/evaluate.py
```
---

## 🐳 Docker Containerization

This project includes two containers:

1. ML Application Container (Flask API + prediction logic)
2. MLflow Tracking Container (for experiment logs & artifacts)

Both containers run together using Docker Compose

🧱  **Build the ML Application Container (Flask API)**
```sh
docker build -f Dockerfile.mlapp -t gopredict-mlapp .
```

🧱 **Build the MLflow Tracking Container**
```sh
docker build -f Dockerfile.mlflow -t gopredict-mlflow .
```

🧩  **Run Both Containers With Docker Compose**
```sh
docker-compose up --build
```
**Stop the containers**
```sh
docker-compose down
```

This will:

- Start MLflow on its assigned port
- Start GoPredict API on its assigned port
- Create an internal Docker network for communication

Services available:

| Service           | URL                              |
| ----------------- | -------------------------------- |
| **GoPredict API** | `http://localhost:5000`          |
| **Swagger UI**    | `http://localhost:5000/apidocs`  |
| **MLflow UI**     | `http://localhost:5001` |

---

## Team Members

* 👤 Aquiles Escarra
* 👤 Angela Lekivetz
* 👤 Komaljeet Kaur
* 👤 Victoriia Biaragova

---