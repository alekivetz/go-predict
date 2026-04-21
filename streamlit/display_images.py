from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt 
from constants import *
import seaborn as sns
import numpy as np
import os

def display_images():

# Get the current script's directory to ensure relative paths work regardless of where you run the script
    script_dir = os.path.dirname(__file__)  # This gets the directory of the current script

    image1 = Image.open(os.path.join(script_dir, 'images', 'NorQuest-Logo.png'))
    image2 = Image.open(os.path.join(script_dir, 'images', 'GoAuto-Logo.png'))
    col1, col2 = st.columns([8, 1])
    with col1:
        st.image(image1.resize((237, 60)))
    with col2:
        st.empty()
        st.image(image2.resize((145, 80)))

# Function to generate scatter plot
def generate_scatter_plot(actual_all, df_test, y_pred_test, filters=None):

    colors = {'Low (<33.6k)': 'blue', 'Medium (33.6k-61.5k)': 'green', 'High (>61.5k)': 'orange'}

    # Begin graph
    plt.figure(figsize=(8, 8))
    if len(actual_all) > 0:
        plt.plot([0, max(actual_all)], [0, max(actual_all)], color='red', linestyle='--', label='Perfect Prediction (y=x)')
    plt.xlabel('Actual Days on Market')
    plt.ylabel('Predicted Days on Market')

    # Filter the df_test to only include the specified values
    if filters is not None:
        df_filtered = df_test[
            (df_test['make'].isin(filters['make'])) & 
            (df_test['model'].isin(filters['model'])) & 
            (df_test['price'] >= filters['price'][0]) & (df_test['price'] <= filters['price'][1]) &
            (df_test['model_year'] >= filters['model_year'][0]) & (df_test['model_year'] <= filters['model_year'][1]) &
            (df_test['mileage'] >= filters['mileage'][0]) & (df_test['mileage'] <= filters['mileage'][1]) &
            (df_test['msrp'] >= filters['msrp'][0]) & (df_test['msrp'] <= filters['msrp'][1]) &
            (df_test['listing_month'].isin(filters['listing_month']))
        ]

        y_pred_filtered = y_pred_test.loc[df_filtered.index]
        title = 'Predicted vs. Actual Days on Market: Filtered'

        for segment in msrp_labels:
            df_test_seg = df_filtered[df_filtered['msrp_binned'] == segment]
            y_test_seg = df_test_seg['days_on_market']
            y_pred_seg = y_pred_filtered.loc[df_test_seg.index]
            plt.scatter(y_test_seg, y_pred_seg, color=colors[segment], alpha=0.5, s=10, label=segment)
    else:
        title = 'Predicted vs. Actual Days on Market'
        for segment in msrp_labels:
            df_test_seg = df_test[df_test['msrp_binned'] == segment]
            y_test_seg = df_test_seg['days_on_market']
            y_pred_seg = y_pred_test.loc[df_test_seg.index]
            plt.scatter(y_test_seg, y_pred_seg, color=colors[segment], alpha=0.5, s=10, label=segment)
    
    plt.title(title)
    if st.session_state.predicted_days is not None:
        plt.scatter(st.session_state.predicted_days, st.session_state.predicted_days, marker='D', color='black', s=80, zorder=5, edgecolor='white', label=f'User Predicted: {st.session_state.predicted_days:.0f} days')

    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
