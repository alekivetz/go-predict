# Define MSRP bins and labels
msrp_labels = ['Low (<33.6k)', 'Medium (33.6k-61.5k)', 'High (>61.5k)']

# Define the best hyperparameters for each segment
param_grid_segment = {
    'Low (<33.6k)': {'n_estimators': 400, 'min_samples_split': 15, 'min_samples_leaf': 3, 'max_features': 0.8, 'max_depth': 10},
    'Medium (33.6k-61.5k)': {'n_estimators': 400, 'min_samples_split': 15, 'min_samples_leaf': 3, 'max_features': 0.8, 'max_depth': 10},
    'High (>61.5k)': {'n_estimators': 400, 'min_samples_split': 15, 'min_samples_leaf': 3, 'max_features': 0.8, 'max_depth': 10}
}

# Feature selection
chosen_features = [
    'make_encoded',
    'model_encoded',
    'price',
    'model_year',
    'price_mileage_interaction',
    'listing_month',
    'mileage'
]

# All columns
all_features = [
    'mileage', 
    'msrp', 
    'make', 
    'model', 
    'model_year', 
    'listing_month', 
    'price', 
    'price_mileage_interaction',
    'make_encoded', 
    'model_encoded', 
    'msrp_binned'
]

# Selected columns for printing
selected_columns = [
    'make',
    'model',
    'msrp',
    'price',
    'model_year',
    'mileage'
]

# Months
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December']