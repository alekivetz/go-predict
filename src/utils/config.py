"""
Configuration file for the GoPredict project.
"""

import os

# Paths
# Get the project root directory by goijg up three levels from the current file's location. 
# os.path.abspath(__file__) returns the absolute path of the current file.
# Each os.path.dirname() call moves one level up in the directory hierarchy.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.path.join() combines multiple directory paths into a single path.
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs')

# Model parameters
# Random state controls random number generation for reproducibility. 
# Test size is the proportion of the data to use for testing - 20% is standard.
RANDOM_STATE = 42   
TEST_SIZE = 0.2     

DATA_FILENAME = 'CBB_Listings.csv'

# Feature columns

CATEGORICAL_FEATURES = ['make', 'model', 'wheelbase_from_vin',]

NUMERICAL_FEATURES = ['mileage', 'price', 'msrp', 'number_price_changes',]

TARGET = 'days_on_market'

DROP_COLUMNS = [
    'listing_id', 'listing_heading', 'listing_type', 'listing_url', 'listing_first_date', 
    'dealer_id', 'dealer_name', 'dealer_street', 'dealer_city', 'dealer_province', 
    'dealer_postal_code', 'dealer_url', 'dealer_email', 'dealer_phone', 'dealer_type', 
    'stock_type', 'vehicle_id', 'vin', 'uvc',  'model_year', 'series', 'style', 'certified',
    'has_leather', 'has_navigation', 'exterior_color', 'exterior_color_category', 
    'interior_color', 'interior_color_category', 'price_analysis', 'drivetrain_from_vin', 
    'engine_from_vin', 'transmission_from_vin', 'fuel_type_from_vin', 
    'price_history_delimited', 'distance_to_dealer', 'location_score', 'listing_dropoff_date']

