🌍 AI-Driven Dynamic AQI Forecaster

An end-to-end, dynamic Machine Learning pipeline that predicts hyperlocal Air Quality Index (AQI) and PM2.5 levels for any city on Earth. It uses live meteorological data, satellite imagery, and on-the-fly model training to generate accurate 72-hour forecasts.

✨ Key Innovations

📍 Predict-Anywhere Architecture: Bypasses the need for static, pre-collected datasets. Instantly geocodes any requested city and fetches highly localized environmental data.

🧠 On-the-Fly Machine Learning: Downloads the last 90 days of weather and pollution data and trains a custom XGBoost time-series model in seconds, explicitly tailoring the AI to that specific city's micro-climate.

📈 Seamless 72-Hour Forecasting: Integrates 3-day weather forecasts to predict future PM2.5 levels, mathematically converting the raw particle mass into the standard US EPA 0-500 Air Quality Index (AQI).

🛰️ Satellite NO2 Heatmaps: Leverages Google Earth Engine and Sentinel-5P satellite data to visualize hyperlocal pollution hotspots (like highways and industrial zones) that drive the AI's predictions.

🏗️ System Architecture

The Data Pipeline: Uses the Open-Meteo API to grab historical weather, historical AQI, and 3-day future weather forecasts.

Feature Engineering: Extracts temporal features (hour, day of week) and lag features so the model can learn from daily traffic patterns and pollution retention.

The ML Engine: An XGBoost Regressor is trained instantly on the 90-day dataset.

The Dashboard: A custom dark-themed Streamlit UI displays continuous Plotly area charts (merging historical and predicted data) and Folium-rendered satellite maps.

📁 Repository Structure

aqi_predictor/
│
├── data_pipeline/         # Handles all external API requests
│   ├── geocoder.py        # Converts city names to coordinates
│   └── open_meteo.py      # Fetches historical & forecast weather/AQI
│
├── ml_engine/             # Handles data processing and ML
│   ├── preprocessor.py    # Feature engineering for time-series data
│   └── forecaster.py      # XGBoost training and prediction logic
│
├── .streamlit/            
│   └── config.toml        # Custom dark-mode UI theme configuration
│
├── app.py                 # Main Streamlit dashboard and UI logic
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation


🚀 Step-by-Step Windows Installation & Setup

Follow these exact steps to run the project locally on a Windows machine in a clean virtual environment.

Step 1: Clone the Repository

Open your Command Prompt or PowerShell and run:

git clone [https://github.com/codesdred/AQI-Predictor.git](https://github.com/codesdred/AQI-Predictor.git)
cd AQI-Predictor


Step 2: Create and Activate a Virtual Environment

This ensures Python installs the packages securely in this folder without affecting your main system.

python -m venv venv
venv\Scripts\activate


(You should now see (venv) at the start of your terminal line).

Step 3: Install Required Dependencies

Install all required libraries (Streamlit, XGBoost, Pandas, Earth Engine, etc.):

pip install -r requirements.txt


Step 4: Configure Google Earth Engine (Free / Non-Commercial)

To view the Sentinel-5P satellite heatmaps, you must authenticate a Google Cloud account. It is 100% free.

Go to the Google Cloud Console and create a new Project.

Go to the Earth Engine API Registration page: https://console.cloud.google.com/earth-engine/configuration

Click Enable API or Register.

When asked for the use case, strictly select Unpaid, Noncommercial, or Academic/Research to bypass any billing requirements.

Once registered, return to your terminal (with the venv still activated) and run:

earthengine authenticate --auth_mode=notebook


A link will appear in your terminal. Click it, log into your Google account, check ALL permission boxes, and click Continue.

Copy the Authorization Code Google gives you, paste it into your terminal, and press Enter.

(Note: The project ID in app.py must match the Project ID you just registered in Google Cloud).

Step 5: Launch the Application

Start the Streamlit server:

streamlit run app.py


Your browser will automatically open to http://localhost:8501.

🎯 Future Scope

Deep Learning Integration: Implement CNN-LSTM spatio-temporal models for complex city-grid forecasting.

Alert System: Add Twilio/SMTP integration to send SMS/Email warnings when a city's forecasted AQI crosses the "Unhealthy" (150+) threshold.

Extended Satellite Data: Incorporate MODIS Aerosol Optical Depth data for enhanced particle tracking.
