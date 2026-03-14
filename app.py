import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import ee

# Import our custom modules
from data_pipeline.geocoder import get_coordinates
from data_pipeline.open_meteo import fetch_training_data, fetch_forecast_weather
from ml_engine.preprocessor import prepare_training_data, prepare_forecast_data
from ml_engine.forecaster import train_aqi_model, predict_future_aqi

# --- AQI MATHEMATICAL CONVERSION ---
def calculate_aqi(pm25):
    """Converts raw PM2.5 concentration (µg/m³) to standard US EPA AQI (0-500)"""
    if pm25 < 0: return 0
    breakpoints = [
        (0.0, 12.0, 0, 50),        
        (12.1, 35.4, 51, 100),     
        (35.5, 55.4, 101, 150),    
        (55.5, 150.4, 151, 200),   
        (150.5, 250.4, 201, 300),  
        (250.5, 500.4, 301, 500)   
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return int(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    return 500 

# UI Config
st.set_page_config(page_title="Hyperlocal AQI", layout="wide", page_icon="🌍")

# --- HEADER SECTION ---
st.title("🌍 AI-Driven Dynamic AQI Forecaster")
st.markdown("**Predicting hyper-local Air Quality Index using live meteorological data, satellite imagery, and on-the-fly Machine Learning.**")

# --- SESSION STATE SETUP ---
if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False
if "target_city" not in st.session_state:
    st.session_state.target_city = "Rourkela"

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209945.png", width=100) 
st.sidebar.header("🔍 Search Location")
city_input = st.sidebar.text_input("Target City:", st.session_state.target_city)

if st.sidebar.button("Generate Forecast", type="primary", use_container_width=True):
    st.session_state.run_pipeline = True
    st.session_state.target_city = city_input

st.sidebar.markdown("---")
st.sidebar.subheader("💡 What is AQI?")
st.sidebar.info(
    "The Air Quality Index (AQI) is a yardstick that runs from 0 to 500. "
    "The higher the AQI value, the greater the level of air pollution and the greater the health concern. "
    "Our AI predicts raw PM2.5 particle mass and mathematically converts it to this global standard."
)

# --- MAIN LOGIC ---
if st.session_state.run_pipeline:
    active_city = st.session_state.target_city
    
    with st.spinner(f"Initiating Global Data Fetch & Training ML Model for {active_city}..."):
        
        # 1. Geocoding
        location = get_coordinates(active_city)
        if not location:
            st.error("City not found. Please try another name.")
            st.stop()
            
        lat, lon = location['lat'], location['lon']
        
        # 2. Fetch Data
        hist_df = fetch_training_data(lat, lon, past_days=90)
        forecast_weather_df = fetch_forecast_weather(lat, lon, forecast_days=3)
        
        if hist_df is None or forecast_weather_df is None:
            st.error("Failed to fetch data from Open-Meteo API.")
            st.stop()

        # 3. Train Model
        X_train, y_train, clean_hist_df = prepare_training_data(hist_df)
        model, mae = train_aqi_model(X_train, y_train)
        
        # 4. Predict Future
        X_forecast, clean_forecast_df = prepare_forecast_data(forecast_weather_df)
        future_predictions = predict_future_aqi(model, X_forecast)
        
        clean_forecast_df['Predicted_PM25'] = future_predictions
        clean_forecast_df['Predicted_AQI'] = clean_forecast_df['Predicted_PM25'].apply(calculate_aqi)
        clean_hist_df['AQI'] = clean_hist_df['pm2_5'].apply(calculate_aqi)

        # --- DASHBOARD UI ---
        st.success(f"✅ Successfully locked onto **{location['name']}**")
        
        # KEY METRICS
        current_aqi = clean_hist_df['AQI'].iloc[-1]
        next_24h_avg_aqi = clean_forecast_df['Predicted_AQI'].iloc[0:24].mean()
        
        current_pm25 = clean_hist_df['pm2_5'].iloc[-1]
        next_24h_avg_pm25 = clean_forecast_df['Predicted_PM25'].iloc[0:24].mean()

        st.markdown("### 📊 Air Quality Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current AQI", f"{current_aqi:.0f}", delta="Live")
        col2.metric("Predicted 24h Avg AQI", f"{next_24h_avg_aqi:.0f}", delta=f"{next_24h_avg_aqi - current_aqi:.0f} from current", delta_color="inverse")
        col3.metric("Current PM2.5", f"{current_pm25:.1f} µg/m³", delta="Live")
        col4.metric("Predicted 24h Avg PM2.5", f"{next_24h_avg_pm25:.1f} µg/m³", delta=f"{next_24h_avg_pm25 - current_pm25:.1f} from current", delta_color="inverse")

        st.markdown("---")
        
        # CHARTS AREA (Fixed Timeline Leakage)
        st.subheader(f"📈 Past 48 Hours & 72-Hour Prediction Forecast")
        with st.expander("How to read this chart"):
            st.write("This chart maps the actual recorded AQI from the past 48 hours (gray) directly into the AI's predicted AQI for the next 72 hours (green).")
            
        # Cleanly split the timeline so historical data doesn't overlap the forecast
        forecast_start_time = clean_forecast_df['time'].iloc[0]
        true_historical = clean_hist_df[clean_hist_df['time'] < forecast_start_time]
        
        recent_hist = true_historical.tail(48).copy()
        recent_hist['Timeline'] = 'Historical (Actual)'
        recent_hist['Chart_AQI'] = recent_hist['AQI']

        forecast_plot = clean_forecast_df.copy()
        forecast_plot['Timeline'] = 'Forecast (Predicted)'
        forecast_plot['Chart_AQI'] = forecast_plot['Predicted_AQI']

        combined_df = pd.concat([
            recent_hist[['time', 'Chart_AQI', 'Timeline']],
            forecast_plot[['time', 'Chart_AQI', 'Timeline']]
        ])

        fig = px.area(
            combined_df, 
            x='time', 
            y='Chart_AQI', 
            color='Timeline',
            labels={'Chart_AQI': 'Air Quality Index (AQI)', 'time': 'Date & Time'},
            color_discrete_map={'Historical (Actual)': '#78909C', 'Forecast (Predicted)': '#00E676'}
        )
        fig.add_hline(y=100, line_dash="dash", annotation_text="Moderate Limit", line_color="yellow")
        fig.add_hline(y=150, line_dash="dot", annotation_text="Unhealthy Threshold", line_color="#FF5252")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)") 
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # SATELLITE MAP AREA
        col_map_text, col_map_legend = st.columns([3, 1])
        with col_map_text:
            st.subheader("🛰️ Hyperlocal Satellite Pollution Heatmap")
            st.markdown("*(Powered by Sentinel-5P Satellite via Google Earth Engine)*")
        with col_map_legend:
            # Custom HTML/CSS Legend
            st.markdown("""
            <div style='background-color: #1E2130; padding: 10px; border-radius: 8px; border: 1px solid #333;'>
                <div style='margin-bottom: 5px; font-size: 14px; color: #FAFAFA; text-align: center; font-weight: bold;'>NO2 Density Legend</div>
                <div style='display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: #aaa;'>
                    <span>Low</span>
                    <div style='height: 12px; width: 60%; background: linear-gradient(to right, black, blue, purple, cyan, green, yellow, red); border-radius: 6px;'></div>
                    <span>High</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("What am I looking at? (Satellite Data Explained)"):
            st.write("""
            **Why Nitrogen Dioxide (NO2)?** While our AI predicts the overall AQI based on PM2.5, this satellite map displays NO2. NO2 is a highly localized gas produced primarily by burning fossil fuels (vehicle exhaust and industrial power plants). 
            
            By mapping NO2, we can visually identify the exact geographical "hotspots" (like busy highways or factory districts) that are generating the pollution driving our AQI predictions. 
            """)

        try:
            # Make sure your project ID here is correct
            ee.Initialize(project='ee-sanjeeb0509dash2004')
            m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB dark_matter") 
            
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterDate(ee.Date(pd.Timestamp.now() - pd.Timedelta(days=30)), ee.Date(pd.Timestamp.now())) \
                .select('tropospheric_NO2_column_number_density') \
                .mean()
            
            vis_params = {
                'min': 0,
                'max': 0.0002,
                'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
            }
            
            map_id_dict = ee.Image(dataset).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='NO2 Heatmap',
                overlay=True,
                control=True,
                opacity=0.5
            ).add_to(m)
            
            st_folium(m, width=1200, height=500, returned_objects=[])
            
        except Exception as e:
            st.error(f"⚠️ Satellite Map Error: {e}")
            m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB dark_matter")
            folium.Marker([lat, lon], popup=active_city).add_to(m)
            st_folium(m, width=1200, height=500, returned_objects=[])