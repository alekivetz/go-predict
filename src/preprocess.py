"""
Data preprocessing module for the GoPredict project.
Handles data loading, cleaning, feature engineering, encoding, and splitting.
"""

# Standard library imports
import argparse
import logging
import os
from typing import Optional, Tuple

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load configuration from YAML file
def load_config(path: str = "configs/preprocess_config.yaml") -> dict:
    """
    Loads preprocessing configuration parameters from a YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Configure the logging system to include timestamps, module names, and log levels.
# This provides consistent and detailed runtime information across the entire pipeline.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load GoPredict data from a CSV file.

    Args:
        filepath: Path to the CSV file containing the data.

    Returns:
        DataFrame containing the data.
    """
    try: 
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise


def clean_vehicle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the vehicle dataset by:
    - Converting object columns to categorical types.
    - Removing duplicate records based on VIN and mileage, 
      retaining only the record with the lowest 'days_on_market' per VIN-mileage pair.

    Args: 
        df: DataFrame containing the vehicle data.

    Returns:
        DataFrame containing the cleaned data.

    """

    df = df.copy()

    # Loop through all columns and convert 'object' type to 'category'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    # Keep only the record with the lowest "days_on_market" per unique VIN and mileage
    df= df.loc[df.groupby(["vin", "mileage"], observed=True)["days_on_market"].idxmin()].reset_index(drop=True)

    logger.info(f'Data types converted and duplicates removed. Shape: {df.shape}')

    return df

    
def impute_wheelbase_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes outliers and zero values in 'wheelbase_from_vin' with the median of the entire dataset.

    Args:
        df: DataFrame containing the vehicle data.
    
    Returns:
        DataFrame with wheelbase outliers imputed.
    """

    df = df.copy()

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
    
    Args:
        df: DataFrame containing the vehicle data.
    
    Returns:
        DataFrame with price outliers imputed.
    """

    df = df.copy()

    # Binary flag potential outliers
    df["price_flagged"] = (
        ((df["msrp"] < 150000) & (df["price"] > 250000)) |  # Overpriced vehicles
        (df["price"] < 2000) |  # Extremely low price
        ((df["model_year"].isin([2023, 2024])) & (df["price"] < df["msrp"]))  # 2023-2024 below MSRP
    ).astype(int)
    
    # Initialize imputed price
    df['price_imputed'] = df['price']
    
    # Compute median within each segment, ensuring groups have more than 3 vehicles
    grouped_median_full = (
        df[df["price"] > 1000]
        .groupby(["make", "model", "model_year"], observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby(["make", "model", "model_year"], observed=True)["price"]
        .median()
    )

    grouped_median_model = (
        df[df["price"] > 1000]
        .groupby(["make", "model"], observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby(["make", "model"], observed=True)["price"]
        .median()
    )

    grouped_median_make = (
        df[df["price"] > 1000]
        .groupby("make", observed=True)
        .filter(lambda x: len(x) > 3)
        .groupby("make", observed=True)["price"]
        .median()
    )

    # Compute overall median price as the last fallback
    overall_median = df[df["price"] > 0]["price"].median()

    # Apply median by segment
    def fill_price(row):
        if row["price_flagged"] == 1:  # Only replace outliers and flagged prices
            key_full = (row["make"], row["model"], row["model_year"])
            key_model = (row["make"], row["model"])
            key_make = row["make"]

            if key_full in grouped_median_full and grouped_median_full[key_full] > 0:
                return grouped_median_full[key_full]  # Use full match if valid
            elif key_model in grouped_median_model and grouped_median_model[key_model] > 0:
                return grouped_median_model[key_model]  # Use model-level median if valid
            elif key_make in grouped_median_make and grouped_median_make[key_make] > 0:
                return grouped_median_make[key_make]  # Use brand-level median if valid
            else:
                return overall_median  # Use overall median as the last fallback
        return row["price_imputed"]  # Keep original value if not flagged

    df["price_imputed"] = df.apply(fill_price, axis=1)

    logger.info(f'Price outleirs imputed.')

    return df


def impute_mileage_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and impute mileage outliers within each model year using the IQR method,
    grouped by model year. 

    Args:
        df: DataFrame containing the vehicle data.

    Returns:
        DataFrame with mileage outliers imputed.
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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the vehicle dataset.
        - 'discount': Difference between MSRP and imputed price.
        - 'years_on_market': Vehicle age relative to 2025.
        - 'month_listed': Month extracted from the listing date.

    Args:
        df: DataFrame containing the vehicle data.

    Returns:
        DataFrame with feature engineering applied.
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


def target_encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply target encoding to multiple categorical variables using only training data
    to avoid data leakage. Unseen categories in the test set are replaced with the
    global target mean.

    Args: 
        X_train: Training feature set.
        X_test: Test feature set.
        y_train: Target variable for the training set.
        categorical_features: List of feature names to encode.

    Returns:
        Tuple of encoded training and test dataframes
    """

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    global_mean = y_train.mean()

    for col in categorical_features:
        # Mean target value per category (from training data only)
        mean_encoding = y_train.groupby(X_train[col], observed=True).mean()

        # Apply encoding to train/test sets
        X_train_encoded[col] = X_train[col].map(mean_encoding)
        X_test_encoded[col] = X_test[col].map(mean_encoding).fillna(global_mean)

    logger.info(f'Target encoding applied to {len(categorical_features)} categorical features.')

    return X_train_encoded, X_test_encoded


def drop_unecessary_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Drop columns that are not necessary for the model.

    Args: 
        df: DataFrame containing the vehicle data.
        drop_cols: List of columns to drop
    
    Returns:
        DataFrame with unnecessary columns dropped.
    """

    df = df.copy()
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    logger.info(f'Unnecessary columns dropped: {cols_to_drop}')

    return df


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target variables.
    
    Args:
        df: DataFrame containing the vehicle data.
        target: Target column
    
    Returns:
        Tuple of (features, target)
    """

    X = df.drop(columns=target)
    y = df[target]

    logger.info(f'Data split into features and target. Shapes: {X.shape}, {y.shape}')

    return X, y


def split_train_test(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test sets.

    Args:
        X: DataFrame containing the features.
        y: Series containing the target variable.
        test_size: Proportion of the dataset to use for testing.
        random_state: Random state for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(f'Data split into train and test sets. Shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath: str, config: dict) -> Tuple:
    """
    Complete preprocessing pipeline for the GoPredict project.
    Executes data loading, cleaning, imputation, feature engineering,
    encoding, and train/test splitting.

    Args:
        filepath: Path to the raw data file.
        config: Configuration dictionary
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """

    logger.info(f'Starting preprocessing pipeline...')  

    # Extract relevant settings from config
    CATEGORICAL_FEATURES = config["features"]["categorical"]
    DROP_COLUMNS         = config["features"]["drop"]
    TARGET               = config["features"]["target"]
    TEST_SIZE            = config["split"]["test_size"]
    RANDOM_STATE         = config["split"]["random_state"]

    # Load data
    df = load_data(filepath)

    # Clean and prepare data
    df = clean_vehicle_data(df)

    # Handle missing values
    df = impute_mileage_outliers(df)
    df = impute_price_outliers(df)
    df = impute_wheelbase_outliers(df)

    # Feature engineering
    df = feature_engineering(df)

    # Drop unnecessary columns
    df = drop_unecessary_columns(df, DROP_COLUMNS)

    # Split features and target
    X, y = split_features_target(df, TARGET)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)

    # Encode categorical features
    X_train, X_test = target_encode_features(X_train, X_test, y_train, CATEGORICAL_FEATURES)

    logger.info(f'Preprocessing pipeline completed.')

    return X_train, X_test, y_train, y_test


def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str 
) -> dict:
    """
    Save preprocessed GoPredict data to disk.

    Args:
        X_train: Training feature set.
        X_test: Test feature set.
        y_train: Target variable for the training set.
        y_test: Target variable for the test set.
        output_dir: Directory to save the data to.

    Returns:
        Dictionary of saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}

    # Save data as numpy files
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    data_path = os.path.join(output_dir, 'gopredict_preprocessed.npy')
    np.save(data_path, data_dict)
    saved_paths['data'] = data_path
    logger.info(f"Preprocessed GoPredict data saved to: {data_path}")

    return saved_paths


def main():
    """
    Main function for Command Line Interface (CLI) preprocessing. 
    """

    # Load configuration
    config = load_config()

    parser = argparse.ArgumentParser(description='Preprocess CBB data for GoPredict model training.')

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the raw CSV file.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=config["data"]["processed_path"],
        help=f"Directory to save preprocessed data (default: {config['data']['processed_path']})"
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=config["split"]["test_size"],
        help=f"Proportion of the dataset to use for testing (default: {config['split']['test_size']})"
    )


    args = parser.parse_args()

    # Run preprocessing pipeline
    logger.info(f'Preprocessing data from {args.input}...')

    X_train, X_test, y_train, y_test = preprocess_pipeline(args.input, config)

    # Save preprocessed data
    saved_paths = save_preprocessed_data(X_train, X_test, y_train, y_test, args.output_dir)

    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nSaved files:")
    for key, path in saved_paths.items():
        print(f"  - {key}: {path}")
    print("="*60)
    
    logger.info("Preprocessing completed!")


if __name__ == '__main__':
    main()
