# GoPredict REST API

The **GoPredict REST API** is a Flask-based service for predicting vehicle market duration (**Days on Market**) using segmented machine learning models trained on Go Auto data. The models are divided into three MSRP segments: **Low (<33.6k)**, **Medium (33.6k–61.5k)**, and **High (>61.5k)**. The API automatically selects the correct model based on the input vehicle’s MSRP and returns the predicted number of days a vehicle will remain on the market.  

The application is built with **Flask** and documented using **Flasgger**, which provides an interactive Swagger UI at `/apidocs`. It includes three primary endpoints: `/health` for API status checks, `/gopredict_home` for endpoint details and example input, and `/predict` for generating predictions. Models are stored in the `models/` directory and loaded when the API starts. A preprocessing pipeline file is also saved for reproducibility but is not used during runtime since preprocessing is already embedded in the trained models.  

To run the API:  
1. Activate your virtual environment: `source .venv/bin/activate`  
2. Install dependencies: `pip install -r requirements.txt`  
3. Start the application: `python -m src.app`  
Once running, you can access the API home at [http://127.0.0.1:5000/gopredict_home](http://127.0.0.1:5000/gopredict_home) and documentation at [http://127.0.0.1:5000/apidocs](http://127.0.0.1:5000/apidocs).  

**Endpoints**  
- **GET /health** – Confirms that the API is running and returns a JSON status message.  
- **GET /gopredict_home** – Displays API information, MSRP segment thresholds, and sample input.  
- **POST /predict** – Accepts vehicle attributes in JSON format, determines the correct segment model, and returns predicted days on market.  
- **POST /v1/predict** – Uses the best-performing segment model (Medium segment, R² = 0.961).  
- **POST /v2/predict** – Uses the second-best segment model (High segment, R² = 0.963).  


**Example input:**  
{  
&nbsp;&nbsp;"month_listed": 5,  
&nbsp;&nbsp;"number_price_changes": 2,  
&nbsp;&nbsp;"discount": 3500.0,  
&nbsp;&nbsp;"model": "Civic",  
&nbsp;&nbsp;"mileage": 42000,  
&nbsp;&nbsp;"price_imputed": 29000.0,  
&nbsp;&nbsp;"make": "Honda",  
&nbsp;&nbsp;"msrp": 36000,  
&nbsp;&nbsp;"years_on_market": 7,  
&nbsp;&nbsp;"wheelbase_from_vin": 106.3  
}  

**Example output:**  
{  
&nbsp;&nbsp;"predictions": [  
&nbsp;&nbsp;&nbsp;&nbsp;{ "segment": "Medium", "predicted_days_on_market": 278.03 }  
&nbsp;&nbsp;],  
&nbsp;&nbsp;"total_records": 1  
}  

Each MSRP segment model is trained independently and stored as a `.joblib` file. The `assign_segment()` function determines which model to use for each prediction. Input validation ensures all required features are present and correctly typed. Common errors, such as missing fields, invalid JSON, or unavailable models, return descriptive messages with appropriate HTTP status codes (400 for invalid data, 500 for missing models).  

**Swagger UI** automatically documents all endpoints based on the Flasgger configuration. This project demonstrates the practical deployment of machine learning models through RESTful API design, containerized development practices, and reproducible experiment tracking.  
