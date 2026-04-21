import numpy as np
import pandas as pd
import streamlit as st
from constants import *

# Preprocessing function for user input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df_train = st.session_state.df_train

    # Apply mean encoding
    for col in ['make', 'model']:
        df[f'{col}_encoded'] = df[col].map(st.session_state[f'{col}_encoder']).astype(float)

    global_mean = df_train['days_on_market'].mean()
    df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_mean)

    df['price_mileage_interaction'] = df['price'] * df['mileage']

    return df[all_features]

# Predict function
def predict_days_on_market(user_input):
    segment = user_input['segment']
    model = st.session_state.models.get(segment)

    if not model:
        raise ValueError(f"No model found for segment: {segment}")

    input_df = preprocess_input(user_input)
    prediction = model.predict(input_df[chosen_features])[0]
    return prediction

def calculate_distance(row, user_input_values):
    return np.sqrt(np.sum((row.values - user_input_values) ** 2))

def find_closest_match(user_input):
    df_test = st.session_state.df_test
    input_df = preprocess_input(user_input)
    user_input_values = input_df[chosen_features].iloc[0].values  # Convert to numpy array

    # Calculate the Euclidean distance
    distances = df_test[chosen_features].apply(lambda row: calculate_distance(row, user_input_values), axis=1)

    # Find the closest match index
    closest_match_index = distances.idxmin()

    # Get the actual 'days_on_market' value for the closest match
    closest_actual = df_test.loc[closest_match_index, 'days_on_market']

    # Get details for closest row
    closest_match_details = df_test.loc[closest_match_index]
    return closest_actual, closest_match_details

def filter_closest(user_input):
    df_train = st.session_state.df_train

    # Make df of user input
    input_df = preprocess_input(user_input)

    # Merge matches from df_test
    filtered_df = df_train.merge(input_df, on=['make', 'model'], how='inner')

    return input_df, filtered_df

