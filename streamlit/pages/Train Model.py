import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from load_data import encode_data
from display_images import * 
from constants import *

# Set up Streamlit page
st.set_page_config(page_title='Train Model - Days on Market Predictor', page_icon='📊', layout='wide')
display_images()
st.title('📊 Train Model')
st.subheader("Train Predictive Models by MSRP Segment")

st.write("""
    This page trains a separate Random Forest model for each MSRP segment to predict how many days a vehicle will remain on the market. 
    After training, key performance metrics (MSE, MAE, and R²) are displayed to evaluate model accuracy.
""")
st.subheader('Tips for Best Results')
st.info("""
    * Click the **Train Model** button to begin training. 
    * If the model has already been trained, the metrics will be displayed automatically. 
    * You can retrain the model at any time by clicking the **Train Model** button again.
""")


# Get model data
encode_data()
df = st.session_state.df
df_train = st.session_state.df_train
df_test = st.session_state.df_test

# Fix predict session state
st.session_state.predicted_days = None

# Check if models are stored in session state 
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.mse_combined = None
    st.session_state.mae_combined = None
    st.session_state.r2_combined = None
    st.session_state.actual_all = []
    st.session_state.predicted_all = []
    st.session_state.y_pred_test = pd.Series()

# Train and predict if model hasn't been trained yet
if st.button('Train Model') or not st.session_state.models:

    actual_all = []
    predicted_all = []
    y_pred_test = pd.Series(index=df_test.index, dtype=float)

    for segment in msrp_labels:
        df_train_seg = df_train[df_train['msrp_binned'] == segment]
        df_test_seg = df_test[df_test['msrp_binned'] == segment]

        X_train_seg = df_train_seg[chosen_features]
        y_train_seg = df_train_seg['days_on_market']
        X_test_seg = df_test_seg[chosen_features]
        y_test_seg = df_test_seg['days_on_market']

        params = param_grid_segment[segment]
        
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42
        )
        
        model.fit(X_train_seg, y_train_seg)
        y_pred = model.predict(X_test_seg)
        y_pred_test.loc[df_test_seg.index] = y_pred

        actual_all.extend(y_test_seg)
        predicted_all.extend(y_pred)

        mse = mean_squared_error(y_test_seg, y_pred)
        mae = mean_absolute_error(y_test_seg, y_pred)
        r2 = r2_score(y_test_seg, y_pred)

        # Display results for each segment
        col1, col2 = st.columns(2)
        with col1:
            st.success(f'✅ **Segment:** {segment} - **Train Size:** {len(df_train_seg)} - **Test Size:** {len(df_test_seg)}')

        with col2:
            col3, col4, col5 = st.columns(3)
            col3.metric('MSE', f'{mse:.2f}')
            col4.metric('MAE', f'{mae:.2f}')
            col5.metric('R²', f'{r2:.2f}')

        # Store model in session state
        st.session_state.models[segment] = model

    # Calculate and display combined model results
    valid_idx = y_pred_test.dropna().index
    mse_combined = mean_squared_error(df_test.loc[valid_idx, 'days_on_market'], y_pred_test.loc[valid_idx])
    mae_combined = mean_absolute_error(df_test.loc[valid_idx, 'days_on_market'], y_pred_test.loc[valid_idx])
    r2_combined = r2_score(df_test.loc[valid_idx, 'days_on_market'], y_pred_test.loc[valid_idx])
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f'✅ **Combined Model by MSRP** - **Train Size:** {len(df_train)} - **Test Size:** {len(df_test)}')

    with col2:
        col3, col4, col5 = st.columns(3)
        col3.metric('MSE', f'{mse_combined:.2f}')
        col4.metric('MAE', f'{mae_combined:.2f}')
        col5.metric('R²', f'{r2_combined:.2f}')

    # Store session state vars
    st.session_state.mse_combined = mse_combined
    st.session_state.mae_combined = mae_combined
    st.session_state.r2_combined = r2_combined
    st.session_state.actual_all = actual_all
    st.session_state.predicted_all = predicted_all
    st.session_state.y_pred_test = y_pred_test

else:
    # Model already trained
    if st.session_state.models:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f'✅ **Model Already Trained** - **Train Size:** {len(df_train)} - **Test Size:** {len(df_test)}')

        with col2:
            col3, col4, col5 = st.columns(3)
            col3.metric('MSE', f'{st.session_state.mse_combined:.2f}')
            col4.metric('MAE', f'{st.session_state.mae_combined:.2f}')
            col5.metric('R²', f'{st.session_state.r2_combined:.2f}')
col1, col2 = st.columns(2)

# Generate plot
with col1:  
    generate_scatter_plot(st.session_state.actual_all, df_test, st.session_state.y_pred_test)

