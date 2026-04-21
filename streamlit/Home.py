import pandas as pd 
import streamlit as st 
from load_data import load_data
from display_images import display_images

st.set_page_config(page_title='Days on Market Predictor', page_icon='🚗', layout='wide')

display_images()

st.title('🚗 Days on Market Predictor')
st.write('This app predicts the number of days a vehicle will remain on the market based on various factors, using machine learning techniques and historical sales data from Go Auto dealerships in Edmonton.')

st.subheader('Key Features:')
st.info("""
    * Train a Model: Train a machine learning model using historical vehicle data. 
    * Market Prediction: Predict how long a vehicle will remain on the market based on input features.
""")

st.subheader("How to Use the App:")
st.info("""
1. Use the **Train Model** section to train the predictive model based on the historical data.
2. Go to the **Predict** section to input a vehicle's details and predict how long it will stay on the market.
""")

# Load dataset automatically
load_data()
df = st.session_state.df

st.subheader('Sample Vehicle Data')
st.write(df[['make', 'model', 'model_year', 'mileage', 'price', 'msrp', 'days_on_market']].head())

st.subheader("Credits")
st.write("""
This app uses historical CBB data for the Edmonton area, provided by GoAuto. 
\nModels are built using machine learning techniques such as Random Forest.
""")

st.sidebar.success('Select a page above.')

