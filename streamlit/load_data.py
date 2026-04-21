import pandas as pd
import streamlit as st
from io import StringIO
import requests

def load_data():
    if 'df' not in st.session_state:
        # Fetch the content using requests
        url = st.secrets['data_url']
        response = requests.get(url)
        
        if response.status_code == 200:
            # Load the CSV content into a pandas dataframe using StringIO
            df = pd.read_csv(StringIO(response.text))
            df['price'] = df['price_imputed']
            df = df.drop(columns=['price_imputed', 'segment'])
            st.session_state.df = df
        else:
            st.error(f"Failed to load data. HTTP Status: {response.status_code}")
        # # If the data isn't already in session_state, load it
        # df = pd.read_csv(st.secrets['data_url'])
        # df['price'] = df['price_imputed']
        # df = df.drop(columns=['price_imputed', 'segment'])
        # st.session_state.df = df

def encode_data():
    if 'df' not in st.session_state:
        load_data()
    
    df = st.session_state.df
    if 'df_train' not in st.session_state:
        # Interaction features
        df['price_mileage_interaction'] = df['price'] * df['mileage']

        # Encoding categorical features
        for col in ['make', 'model']:
            st.session_state[f'{col}_encoder'] = df['days_on_market'].groupby(df[col]).mean()
            df[f'{col}_encoded'] = df[col].map(st.session_state[f'{col}_encoder']).astype(float)

        global_mean = df['days_on_market'].mean()
        df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_mean)
        # Define MSRP bins
        msrp_bins = [0, 33650, 61500, float('inf')]
        msrp_labels = ['Low (<33.6k)', 'Medium (33.6k-61.5k)', 'High (>61.5k)']

        df['msrp_binned'] = pd.cut(df['msrp'], bins=msrp_bins, labels=msrp_labels)
        
        # Split data into train and test sets
        shuffle_df = df.sample(frac=1)
        train_size = int(0.2 * len(shuffle_df))

        df_train = shuffle_df[:train_size].copy()
        df_test = shuffle_df[train_size:].copy()

        # Store the data in session_state
        st.session_state.df_train = df_train
        st.session_state.df_test = df_test
