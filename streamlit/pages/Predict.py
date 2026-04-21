import pandas as pd 
import streamlit as st 
from display_images import *
from constants import *
from process_data import *

# -------------------
# Streamlit Interface
# -------------------

st.set_page_config(page_title='Predict - Days on Market Predictor', page_icon='🗓️', layout='wide')
display_images()
st.title('🗓️ Predict')

st.subheader("About the Days on Market Predictor")

st.write("""
    This tool allows you to predict the number of days a vehicle will remain on the market before being sold. By entering a few details about the vehicle, such as the make, model, price, and mileage, the tool will generate a prediction based on historical data.

    The prediction takes into account various factors such as:
    - Vehicle's **make** and **model**
    - **Price** and **MSRP** (Manufacturer Suggested Retail Price)
    - **Mileage** and **model year**
    - The **month** the vehicle is listed

    Once the prediction is made, the tool will provide:
    - The **predicted days on market**
    - An **estimated range** based on the model's confidence interval
""")

st.subheader('Tips for Best Results')
st.info("""
    - **Provide Accurate Data**: Ensure that the vehicle details you enter are accurate and representative of the vehicle you're analyzing.
    - **Use Full Price Range**: The model is sensitive to pricing data. If you are unsure of the exact price, try to input a range for better results.
    - **Adjust Filters**: You can use the filter options below the prediction to refine the dataset and visualize the impact of different variables like price, mileage, or model year.
    - **Explore the Graph**: After making the prediction, the graph will display how the model's prediction compares to actual market data, with the ability to filter based on your selected criteria.

    If you're not getting the results you're expecting, try adjusting the input values or exploring the graph filters to see how different factors affect the prediction.
""")

# Check for predicted days to avoid resetting
if 'predicted_days' not in st.session_state:
    st.session_state.predicted_days = None
    st.session_state.predicted_range = None
    st.session_state.success_message = None
    st.session_state.success_range = None

filters = None

if 'models' in st.session_state:
    # Datasets
    df, df_train, df_test = st.session_state.df, st.session_state.df_train, st.session_state.df_test

    # Model
    models = st.session_state.models

    st.subheader('**Input Vehicle Details**')
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        make_options = sorted(df_train['make'].unique())
        make_model_dict = {make: df_train[df_train['make'] == make]['model'].unique() for make in make_options}

        make_input = st.selectbox("Make", make_options, key="input_make")
    with col2:
        model_options = sorted(make_model_dict.get(make_input, []))
        model_input = st.selectbox("Model", model_options, key="input_model")

        # Other inputs
    with col3:
        msrp_input = st.number_input('MSRP', min_value=0, max_value=400000, value=0, step=500, key="input_msrp")
    with col4:    
        price_input = st.number_input('Price', min_value=0, max_value=400000, value=0, step=500, key="input_price")
    with col5:    
        model_year_input = st.number_input("Model Year", min_value=1980, max_value=2023, value=1980, key="input_model_year")
    with col6:
        mileage_input = st.number_input("Mileage (in km)", min_value=0, max_value=500000, value=0, step=500, key="input_mileage")
    with col7:    
        month_name_input = st.selectbox("Month Listed", months, key="input_month")
        listing_month_input = months.index(month_name_input) + 1

    # MSRP segmentation
    if msrp_input < 33600: 
        segment_input = 'Low (<33.6k)'
    elif msrp_input < 61500:
        segment_input = 'Medium (33.6k-61.5k)'
    else:
        segment_input = 'High (>61.5k)'

     # Construct input dict
    user_input = {
        'make': make_input,
        'model': model_input,
        'segment': segment_input,
        'price': price_input,
        'model_year': model_year_input,
        'mileage': mileage_input,
        'listing_month': listing_month_input,
        'msrp': msrp_input,
        'msrp_binned': segment_input
        }

    # Prediction button
    if st.button('Predict Days on Market'):
        try:
            predicted_days = predict_days_on_market(user_input)
            # Store prediction and confidence interval in session state
            st.session_state.predicted_days = predicted_days
            st.session_state.predicted_range = (predicted_days - st.session_state.mae_combined, 
                                                predicted_days + st.session_state.mae_combined)
            
            st.session_state.success_message = f'✅ **Predicted Days on Market:** {predicted_days:.0f} days'
            st.session_state.success_range = f'📉 _Estimated range:_ {st.session_state.predicted_range[0]:.0f} – {st.session_state.predicted_range[1]:.0f} days'
                        
        except ValueError as e:
            st.error(f"❌ {e}")

    if st.session_state.predicted_days is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f'✅ **Predicted Days on Market:** {st.session_state.predicted_days:.0f} days')
        with col2:    
            st.success(f'📉 _Estimated range:_ {st.session_state.predicted_range[0]:.0f} – {st.session_state.predicted_range[1]:.0f} days')

    # Graph and filters        
    col8, col9 = st.columns([5, 4])
    with col9:
        st.subheader('**Filter Plot**')

        # Set default values first
        selected_makes = sorted(df_test['make'].unique())
        selected_models = sorted(df_test['model'].unique())
        price_range = (0, 400000)
        model_year_range = (1980, 2023)
        mileage_range = (0, 500000)
        msrp_range = (0, 400000)
        listing_months = list(range(1, 13))  # Jan to Dec

        # Optional Make Filter
        show_make_selector = st.checkbox('Show Make Filter')
        if show_make_selector:
            selected_makes = st.multiselect('Select Make(s)', options=sorted(df_train['make'].unique()), key='filter_make')

        # Optional Model Filter
        show_model_selector = st.checkbox('Show Model Filter')
        if show_model_selector:
            make_model_dict = {
                make: sorted(df_train[df_train['make'] == make]['model'].unique())
                for make in selected_makes
            }
            filtered_model_options = sorted({model for models in make_model_dict.values() for model in models})
            selected_models = st.multiselect('Select Model(s)', options=filtered_model_options, key='filter_model')

        # Optional Price
        show_price_slider = st.checkbox('Show Price Range Filter')
        if show_price_slider:
            price_range = st.slider('Select Price Range', 0, 400000, (0, 400000), 500, key='filter_price')

        # Optional Model Year
        show_year_slider = st.checkbox('Show Model Year Range Filter')
        if show_year_slider:
            model_year_range = st.slider('Select Model Year Range', 1980, 2023, (1980, 2023), key='filter_model_year')

        # Optional Mileage
        show_mileage_slider = st.checkbox('Show Mileage Range Filter')
        if show_mileage_slider:
            mileage_range = st.slider('Select Mileage Range (in km)', 0, 500000, (0, 500000), 500, key='filter_mileage')

        # Optional MSRP
        show_msrp_slider = st.checkbox('Show MSRP Range Filter')
        if show_msrp_slider:
            msrp_range = st.slider('Select MSRP Range', 0, 400000, (0, 400000), 500, key='filter_msrp')

        # Optional Month
        show_month_selector = st.checkbox('Show Month Filter')
        if show_month_selector:
            selected_months = st.multiselect('Select Listing Month(s)', options=months, key='filter_months')
            listing_months = [months.index(month) + 1 for month in selected_months]

        # MSRP Segmentation
        msrp_avg = (msrp_range[0] + msrp_range[1]) / 2
        if msrp_avg < 33600: 
            segment_filter = 'Low (<33.6k)'
        elif msrp_avg < 61500:
            segment_filter = 'Medium (33.6k-61.5k)'
        else:
            segment_filter = 'High (>61.5k)'

        filters = {
            'make': selected_makes,
            'model': selected_models,
            'price': price_range,
            'model_year': model_year_range,
            'mileage': mileage_range,
            'msrp': msrp_range,
            'listing_month': listing_months,
            'segment': segment_filter,
            'msrp_binned': segment_filter
        }

    with col8:
        generate_scatter_plot(
            st.session_state.actual_all,
            st.session_state.df_test,
            st.session_state.y_pred_test,
            filters
        )        

else:
    st.error('Please train the model first.')
