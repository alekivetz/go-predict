"""
GoPredict Data Preprocessing Module
==============================================================
This module performs data loading, cleaning, feature engineering, encoding,
and splitting for the GoPredict project.

It provides:
    - Functions for imputing outliers and cleaning datasets
    - Creation of engineered features
    - Train/test split and encoding logic
    - CLI support to preprocess raw CSV data and save artifacts

Usage (CLI):
    python -m src.preprocess --input data/raw/CBB_Listings.csv
    
"""

# ============================================================
# =============== 1. IMPORTS & CONFIGURATION =================
# ============================================================

# --- Standard Libraries ---
import argparse
import logging
import os
from pathlib import Path    
from typing import Tuple

# --- Third-party Libraries ---
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Internal Imports ---
from src.utils.helper_functions import load_yaml_config

# ============================================================
# =============== 2. LOGGING SETUP ===========================
# ============================================================

# Configure global logging format for uniform runtime information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================
# =============== 3. DATA LOADING ============================
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads GoPredict data from a CSV file.
    """
    filepath = Path(filepath)
    try:
        df = pd.read_csv(filepath)
        logger.info(f'Data loaded successfully from {filepath}. Shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Failed to load data from {filepath}: {e}')
        raise

# ============================================================
# =============== 4. CLEANING & IMPUTATION ===================
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the vehicle dataset by:
        - Convering data types - objects to categorical
        - Removing duplicate records based on VIN and mileage
    """

    df = df.copy()

    # Loop through all columns and convert object to categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Keep only the record with lowest days_on_market per unique VIN and mileage combination
    df = df.loc[df.groupby(['vin', 'mileage'], observed=True)['days_on_market'].idxmin()].reset_index(drop=True)    

    logger.info(f'Data types converted and duplicates removed. Shape: {df.shape}')
    return df

def impute_wheelbase_outliers(df: pd.DataFrame) -> pd.DataFrame:    
    """
    Imputes outliers and zero values in the wheelbase column with the median of the entire dataset.
    """

    # Ensure 'wheelbase_from_vin' is numeric and convert non-numeric values to NaN
    df['wheelbase_from_vin'] = pd.to_numeric(df['wheelbase_from_vin'], errors='coerce')

    # Calculate Q1, Q3, IQR, and bounds for the entire dataset
    Q1 = df['wheelbase_from_vin'].quantile(0.25)
    Q3 = df['wheelbase_from_vin'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculate the global median
    global_median = df['wheelbase_from_vin'].median()

    # Impute outliers and zero values with the global median
    df['wheelbase_from_vin'] = np.where(
        (df['wheelbase_from_vin'] < lower_bound) |
        (df['wheelbase_from_vin'] > upper_bound) |
        (df['wheelbase_from_vin'] == 0),
        global_median,
        df['wheelbase_from_vin']
    )

    logger.info(f'Wheelbase outliers imputed.')
    return df

def impute_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute outlier prices using median within different levels, ensuring at least 3 vehicles per group.
    """

    df = df.copy()

    # Binary flag potential outliers
    df['price_flagged'] = (
        ((df['msrp'] < 150000) & (df['price'] > 250000)) | # Overpriced vehicles
        (df['price'] < 2000) | # Extremely low price
        ((df['model_year'].isin([2023, 2024])) & (df['price'] < df['msrp'])) # 2023-2024 below MSRP
    ).astype(int)

    # Initialize imputed price
    df['price_imputed'] = df['price']

    # Compute median within each segment, ensuring groups have more than 3 vehicles
    grouped_median_full = (
        df[df['price'] > 1000]
        .groupby(['make', 'model', 'model_year'], observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby(['make', 'model', 'model_year'], observed=True)['price']
        .median()
    )

    grouped_median_model = (
        df[df['price'] > 1000]
        .groupby(['make', 'model'], observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby(['make', 'model'], observed=True)['price']
        .median()
    )

    grouped_median_make = (
        df[df['price'] > 1000]
        .groupby(['make'], observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby(['make'], observed=True)['price']
        .median()
    )

    # Overall median for fallback
    overall_median = df[df['price'] > 0]['price'].median()

    # Apply median by segment
    def fill_price(row):
        if row['price_flagged'] == 1:  # Only replace outliers and flagged prices
            key_full = (row['make'], row['model'], row['model_year'])
            key_model = (row['make'], row['model'])
            key_make = row['make']

            if key_full in grouped_median_full and grouped_median_full[key_full] > 0:
                return grouped_median_full[key_full]  # Use full match if valid
            elif key_model in grouped_median_model and grouped_median_model[key_model] > 0:
                return grouped_median_model[key_model]  # Use model-level median if valid
            elif key_make in grouped_median_make and grouped_median_make[key_make] > 0:
                return grouped_median_make[key_make]  # Use brand-level median if valid
            else:
                return overall_median  # Use overall median as the last fallback
        return row['price_imputed'] 
    
    df['price_imputed'] = df.apply(fill_price, axis=1)

    logger.info(f'Price outliers imputed.')
    return df

def impute_mileage_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and impute mileage outliers within each model year using the IQR method. 
    """

    df = df.copy()
    # Finding mileage outliers by grouping model_year
    for year, group in df.groupby('model_year'):
        df['mileage'] = df['mileage'].astype(float)

        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(group['mileage'], 25)
        Q3 = np.percentile(group['mileage'], 75)

        # Compute Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Compute median mileage for the model year (excluding outliers)
        median_mileage = group[(group['mileage'] >= lower_bound) & (group['mileage'] <= upper_bound)]['mileage'].median()

        # Replace outliers with the median mileage of their respective model year
        df.loc[(df['model_year'] == year) & ((df['mileage'] < lower_bound) | (df['mileage'] > upper_bound)), 'mileage'] = median_mileage

    logger.info(f'Mileage outliers imputed.')
    return df

# ============================================================
# =============== 5. FEATURE ENGINEERING =====================
# ============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the vehicle dataset.
        - 'discount': Difference between MSRP and imputed price.
        - 'years_on_market': Vehicle age relative to 2025.
        - 'month_listed': Month extracted from the listing date.
    """

    df = df.copy()

    # Discount
    df['discount'] = df['msrp'] - df['price_imputed']

    # Years on market
    df['years_on_market'] = 2025 - df['model_year'] 

    # Month listed
    df['listing_first_date'] = pd.to_datetime(df['listing_first_date'])
    df['month_listed'] = df['listing_first_date'].dt.month

    logger.info(f'Feature engineering applied.')
    return df

# ============================================================
# =============== 6. ENCODING & SPLITTING ====================
# ============================================================

def target_encode_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        categorical_features: list
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    """
    Target encode categorical features. 
    Also return the per-column mapping dicts (category to mean target)
    and the global mean for unseen categories.
    """

    X_train_encoded = X_train.copy()
    X_test_encoded  = X_test.copy()
    global_mean = float(y_train.mean())

    encoding_maps = {}

    for col in categorical_features:
        # Build mapping on TRAIN ONLY using ORIGINAL string categories
        mean_encoding = y_train.groupby(X_train[col], observed=True).mean()
        encoding_maps[col] = mean_encoding.to_dict()

        # Apply to train/test
        X_train_encoded[col] = X_train[col].map(mean_encoding)
        X_test_encoded[col]  = X_test[col].map(mean_encoding).fillna(global_mean)

    logger.info(f'Target encoding applied to {len(categorical_features)} categorical features.')

    return X_train_encoded, X_test_encoded, encoding_maps, global_mean

def drop_unecessary_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Drop columns that are not necessary for the model.
    """

    df = df.copy()
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    logger.info(f'Dropped {len(cols_to_drop)} unnecessary columns.')
    return df

def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataset into features and target variables.
    """

    X = df.drop(columns=target)
    y = df[target]

    logger.info(f'Data split into features and target variables. Shapes: {X.shape}, {y.shape}') 
    return X, y

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,   
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(f'Dataset split into training and testing sets. Shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')
    return X_train, X_test, y_train, y_test

# ============================================================
# =============== 7. MAIN PIPELINE FUNCTION ==================
# ============================================================

def preprocess_pipeline(filepath: str, config: dict) -> Tuple:
    """
    Complete preprocessing pipeline for the GoPredict project.
    Executes data loading, cleaning, imputation, feature engineering,
    encoding, and splitting.
    """

    logger.info('Staring the preprocessing pipeline...')

    # Extract configuration parameters
    categorical_features    = config['features']['categorical']
    drop_columns            = config['features']['drop']
    target                  = config['features']['target']
    test_size               = config['split']['test_size']
    random_state            = config['split']['random_state']

    # Load data
    df = load_data(filepath)

    # Clean data
    df = clean_data(df)

    # Handle missing values
    df = impute_wheelbase_outliers(df)
    df = impute_price_outliers(df)
    df = impute_mileage_outliers(df)

    # Feature engineering
    df = feature_engineering(df)

    # Drop unnecessary columns
    df = drop_unecessary_columns(df, drop_columns)

    # Split features and target
    X, y = split_features_target(df, target)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    # Encode categorical features
    X_train, X_test, encoding_maps, global_mean = target_encode_features(X_train, X_test, y_train, categorical_features)

    logger.info('Preprocessing pipeline completed.')
    return X_train, X_test, y_train, y_test, encoding_maps, global_mean

# ============================================================
# =============== 8. SAVE PREPROCESSED DATA ==================
# ============================================================

def save_preprocessed_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    output_dir: str
) -> dict:
    """
    Saves the preprocessed data to disk.
    """

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}

    # === Save training data CSV ===
    train_csv_path = os.path.join(output_dir, 'Go_auto_train_data.csv')
    train_df = pd.concat([X_train, y_train.rename('days_on_market')], axis=1)
    train_df.to_csv(train_csv_path, index=False)
    saved_paths['train_csv'] = train_csv_path
    logger.info(f'Preprocessed training CSV exported to: {train_csv_path}')

    # === Save test data CSV ===
    test_csv_path = os.path.join(output_dir, 'Go_auto_test_data.csv')
    test_df = pd.concat([X_test, y_test.rename('days_on_market')], axis=1)
    test_df.to_csv(test_csv_path, index=False)
    saved_paths['test_csv'] = test_csv_path
    logger.info(f'Preprocessed test CSV exported to: {test_csv_path}')

    # === Save numpy version ===
    data_path = os.path.join(output_dir, 'gopredict_preprocessed.npy')
    np.save(data_path, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})
    saved_paths['data'] = data_path
    logger.info(f'Preprocessed GoPredict data saved to: {data_path}')

    return saved_paths


# ============================================================
# =============== 9. CLI ENTRY POINT =========================
# ============================================================

def main():
    """
    Main function for Command Line Interface (CLI) preprocessing.   
    """

    config = load_yaml_config('configs/preprocess_config.yaml')

    parser = argparse.ArgumentParser(
        description='Preprocess GoPredict data for training and testing.'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the raw GoPredict data CSV file.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=config['data']['processed_path'],
        help=f'Directory to save the preprocessed data to. Default: {config["data"]["processed_path"]}'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=config['split']['test_size'],
        help=f'Test size for the train/test split. Default: {config["split"]["test_size"]}'
    )

    args = parser.parse_args()

    # Run preprocessing pipeline
    logger.info(f'Preprocessing data from {args.input}...')

    X_train, X_test, y_train, y_test, encoding_maps, global_mean = preprocess_pipeline(args.input, config)

    # Save target-encoding mappings and global mean
    enc_bundle = {
        'global_mean': float(global_mean),
        'encoders': encoding_maps,
        'categorical_features': config['features']['categorical']
    }
    encoders_path = os.path.join(args.output_dir, 'target_encoders.pkl')
    joblib.dump(enc_bundle, encoders_path)
    logger.info(f'Target encoders saved to: {encoders_path}')

    # Save preprocessed data
    saved_paths = save_preprocessed_data(X_train, X_test, y_train, y_test, args.output_dir)

    print('\n' + '='*60)
    print('Preprocessing completed successfully!')
    print('Saved files:')
    for key, path in saved_paths.items():
        print(f'  {key}: {path}')
    print('='*60 + '\n')

    logger.info('Preprocessing completed!.')

# ============================================================
# =============== 10. MODULE ENTRY POINT =====================
# ============================================================

if __name__ == '__main__':
    main()
