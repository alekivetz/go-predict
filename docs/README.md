# GoPredict: Vehicle Resale Days-on-Market Prediction 🚗

## Description
**GoPredict** is a Machine Learning model designed to predict how many days a vehicle will stay on the market before being sold. It leverages features such as age, price, and mileage to help dealerships optimize inventory management and implement dynamic pricing strategies.

## Team Members

* 👤 Aquiles Escarra
* 👤 Angela Lekivetz
* 👤 Komaljeet Kaur
* 👤 Victoriia Biaragova

## Project Objectives

* Conduct Exploratory Data Analysis (EDA) to understand the dataset and detect trends.
* Develop a regression model to predict days on the market.
* Optimize inventory management for dealerships.
* Visualize results using Power BI and an APP.

##  Dataset and Data Source

* **Source:** Dataset provided by Go Auto, one of the largest vehicle dealership networks in Canada.
* **Location:** Edmonton and surrounding areas.
* **Key Features:** Vehicle age, mileage, price, dealership location.

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

### Project Structure

This project follows a standard machine learning repository layout. **All commands must be executed from the Project Root Directory gopredict.**

```bash
    gopredict/
    ├── data/                # Stores datasets (raw, processed)
    │   ├── raw/
    │   ├── processed/
    ├── models/              # Saved or checkpointed model files
    ├── configs/             # Configuration YAMLs for pipeline components
    ├── notebooks/           # Jupyter notebooks for exploration/demos
    ├── src/
    │   ├── train.py         # Script to train your model
    │   ├── predict.py       # Script to make predictions
    │   ├── preprocess.py    # Data preprocessing logic
    │   ├── evaluate.py      # Model evaluation script
    ├── docs/                # Documentation (README)
    ├── requirements.txt     # Project dependencies
    └── .gitignore           
```

## Pipeline Execution Guide

To run the full pipeline, navigate to the **Project Root Directory gopredict** and execute the scripts sequentially:

### Step 1: Install Dependencies
pip install -r requirements.txt

### Step 2: Preprocessing and Data Split (`preprocess.py`)
python src/preprocess.py 

### Step 3: Training the Segmented Models (`train.py`)
python src/train.py

### Step 4: Generating Predictions (`predict.py`)
python src/predict.py

### Step 5: Evaluating Performance (`evaluate.py`)
python src/evaluate.py
