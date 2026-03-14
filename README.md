# 🌍 AI-Driven Dynamic AQI Forecaster

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-239B56?style=for-the-badge)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Satellite%20Data-F29C1F?style=for-the-badge&logo=google&logoColor=white)

An end-to-end, dynamic Machine Learning pipeline that predicts hyperlocal Air Quality Index (AQI) and PM2.5 levels for **any city on Earth**. It uses live meteorological data, satellite imagery, and on-the-fly model training to generate accurate 72-hour forecasts.

---

## ✨ Key Innovations

* 📍 **Predict-Anywhere Architecture:** Bypasses the need for static, pre-collected datasets. Instantly geocodes any requested city and fetches highly localized environmental data.
* 🧠 **On-the-Fly Machine Learning:** Downloads the last 90 days of weather and pollution data and trains a custom XGBoost time-series model *in seconds*, explicitly tailoring the AI to that specific city's micro-climate.
* 📈 **Seamless 72-Hour Forecasting:** Integrates 3-day weather forecasts to predict future PM2.5 levels, mathematically converting the raw particle mass into the standard US EPA 0-500 Air Quality Index (AQI).
* 🛰️ **Satellite NO2 Heatmaps:** Leverages Google Earth Engine and Sentinel-5P satellite data to visualize hyperlocal pollution hotspots (like highways and industrial zones) that drive the AI's predictions.

---

## 🏗️ System Architecture

1. **The Data Pipeline:** Uses the `Open-Meteo API` to grab historical weather, historical AQI, and 3-day future weather forecasts. 
2. **Feature Engineering:** Extracts temporal features (hour, day of week) and lag features so the model can learn from daily traffic patterns and pollution retention.
3. **The ML Engine:** An `XGBoost Regressor` is trained instantly on the 90-day dataset.
4. **The Dashboard:** A custom dark-themed `Streamlit` UI displays continuous Plotly area charts (merging historical and predicted data) and Folium-rendered satellite maps.

---

## 📁 Repository Structure

```text
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
🚀 Installation & Setup
We recommend using a virtual environment to run this project cleanly.

1. Clone the repository
Bash
git clone [https://github.com/YOUR_USERNAME/AQI-Predictor.git](https://github.com/YOUR_USERNAME/AQI-Predictor.git)
cd AQI-Predictor
2. Create and activate a virtual environment
For Windows:

Bash
python -m venv venv
venv\Scripts\activate
For Mac/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
Bash
pip install -r requirements.txt
4. Authenticate Google Earth Engine (Required for Satellite Maps)
This project uses Google Earth Engine for Sentinel-5P satellite data. Run the following command in your terminal and follow the browser prompts to log into your Google Cloud account:

Bash
earthengine authenticate --auth_mode=notebook
Note: Ensure your Google account is linked to a valid Google Cloud Project with the Earth Engine API enabled.

5. Run the Application
Bash
streamlit run app.py
🎯 Future Scope
Deep Learning Integration: Implement CNN-LSTM spatio-temporal models for complex city-grid forecasting.

Alert System: Add Twilio/SMTP integration to send SMS/Email warnings when a city's forecasted AQI crosses the "Unhealthy" (150+) threshold.

Extended Satellite Data: Incorporate MODIS Aerosol Optical Depth data for enhanced particle tracking.


*(Just remember to change `YOUR_USERNAME` in the `git clone` link under the Installation section to your actual GitHub username!)*

If you have a few minutes before judging begins, I highly recommend practicing a 60-second
