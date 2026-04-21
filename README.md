# GoPredict: Vehicle Resale Days-on-Market Prediction

A production-ready machine learning project predicting how many days a used vehicle will remain on the market before being sold. Built in collaboration with Go Auto, one of Canada's largest dealership networks, through NorQuest College. The system uses a segmented Random Forest regression model achieving R² of 0.82 to 0.89 across three MSRP tiers, deployed as a REST API on GCP Cloud Run with an interactive Streamlit front end and a Power BI dashboard analyzing 91,000 sold listings.

**Business question:** What factors drive days-on-market for used vehicles, and can we predict how long a specific vehicle will sit before selling?

---

## Live Demo

| Service | URL |
| :--- | :--- |
| GoPredict API | https://gopredict-api-5258735120.us-central1.run.app |
| Swagger UI | https://gopredict-api-5258735120.us-central1.run.app/apidocs |
| Streamlit App | https://go-predict.streamlit.app |

---

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

## Tech Stack

| Layer | Tool |
| :--- | :--- |
| Modeling | Python, scikit-learn, pandas, NumPy |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | Flask, Swagger UI |
| Front End | Streamlit |
| Cloud | GCP Cloud Run, Cloud Build |
| Containerization | Docker |
| Monitoring | Prometheus, Grafana |
| Visualization | Power BI |
| Version Control | GitHub |

---

## Dataset and Data Source

- **Source:** Dataset provided by Go Auto, one of the largest vehicle dealership networks in Canada.
- **Location:** Edmonton and surrounding areas.
- **Period:** May 2023 to July 2024
- **Size:** 91,171 sold vehicle listings
- **Key Features:** mileage, msrp, model, dealer, price changes, discount, listing month, model year
- Dataset not included in repo due to confidentiality. Use `dvc pull` to retrieve artifacts if you have access.

---

## Project Objectives

- Conduct Exploratory Data Analysis (EDA) to understand the dataset and detect trends.
- Develop a segmented regression model to predict days on market across MSRP tiers.
- Deploy the model as a REST API on GCP Cloud Run.
- Containerize the full system (ML App + MLflow).
- Build an interactive Streamlit front end for live predictions.
- Analyze inventory aging patterns through a Power BI dashboard.

---

## Model

### Architecture

The final model uses a segmented Random Forest Regressor to address non-linearities across different price points. Rather than training a single model on the full dataset, vehicles are segmented into three MSRP tiers and a separate model is trained per tier. This approach captures the distinct market dynamics between economy, mid-range, and luxury vehicles.

| Component | Detail |
| :--- | :--- |
| Model Type | Segmented Random Forest Regressor |
| Segmentation | 3 Tiers (Low, Medium, High) based on MSRP |
| Tier Boundaries | Low: under $33,600 / Medium: $33,600 to $61,500 / High: over $61,500 |
| Feature Count | 10 selected features including engineered features `month_listed` and `discount` |
| Key Hyperparameters | `n_estimators=350`, `max_depth=35`, `min_samples_split=5` |

### Performance

| Segment | R² |
| :--- | :--- |
| Low (under $33,600) | 0.82 |
| Medium ($33,600 to $61,500) | 0.86 |
| High (over $61,500) | 0.89 |

Note: The Streamlit app uses a simplified feature set limited to inputs a user can realistically provide (make, model, price, MSRP, mileage, model year, listing month). Performance metrics for the deployed model differ from the full evaluation above.

---

## Dashboard

The Power BI dashboard analyzes 91,171 sold vehicle listings from May 2023 to July 2024 across three dimensions: vehicle age, price tier, and mileage. Each page includes sold listing volume, days-on-market comparison (median and average), percentage sold over 60 days, sell-through rate, a summary table, and interactive filtering by make. Custom tooltip pages show top 5 brands by sell-through rate on hover.

Screenshots are available in [`docs/screenshots/`](./docs/screenshots/).

### Page 1 - Vehicle Age

Analyzes how vehicle age affects days on market and sell-through rate across four age groups: 0-1 years, 2-5 years, 6-10 years, and 10+ years. Newer vehicles (0-1 years) dominate volume at 48K listings but have the worst sell-through rate, while older vehicles move more reliably despite lower demand.

### Page 2 - Price Tier

Compares inventory aging across four price tiers: Economy, Mid, Premium, and Luxury. Mid-tier vehicles account for the highest volume at 38K listings. Premium and Luxury segments show the longest days on market and the highest percentage of listings aging past 60 days.

### Page 3 - Mileage Tier

Examines the relationship between mileage and market performance across four tiers: 0-25K km, 25-75K km, 75-150K km, and 150K+ km. Low-mileage vehicles have the highest volume but the worst sell-through rate, while high-mileage vehicles perform comparably to mid-mileage ones.

---

## Key Findings

**Newer Isn't Always Faster**: 0-1 year vehicles have the lowest sell-through rate at 64%. Older inventory moves more reliably, with 10+ year vehicles closing at 76%.

**Low Mileage, Long Wait**: Sub-25K km vehicles sit the longest, with 31% aging past 60 days and a sell-through rate of just 64%.

**Luxury Takes Its Time**: Premium and Luxury segments average 64 to 68 days on market and are the least likely to sell within 30 days.

**Mileage Ceiling Matters Less Than You'd Think**: Vehicles with 150K+ km match mid-mileage sell-through rates at 72%, suggesting buyers are less deterred by high mileage than expected.

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

## How to Run

### Prerequisites

- Python 3.12
- Docker
- DVC
- GCP account (for deployment)

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

### Step 4 - Run Locally with Docker

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