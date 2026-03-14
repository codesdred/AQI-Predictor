# data_pipeline/open_meteo.py
import requests
import pandas as pd

def fetch_training_data(lat, lon, past_days=90):
    """
    Fetches historical weather and AQI data to train the model on the fly.
    Uses the standard forecast API with past_days to avoid Archive API lag.
    """
    # UPDATED: Using standard API with &past_days=90 instead of the Archive API
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days={past_days}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&timezone=auto"
    
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&past_days={past_days}&hourly=pm2_5&timezone=auto"

    try:
        # Fetch and parse weather
        w_res = requests.get(weather_url).json()
        if 'hourly' not in w_res:
            print(f"Weather API Error: {w_res}") # Better error logging
            return None
            
        weather_df = pd.DataFrame(w_res['hourly'])
        weather_df['time'] = pd.to_datetime(weather_df['time'])

        # Fetch and parse AQI
        a_res = requests.get(aqi_url).json()
        if 'hourly' not in a_res:
            print(f"AQI API Error: {a_res}") # Better error logging
            return None
            
        aqi_df = pd.DataFrame(a_res['hourly'])
        aqi_df['time'] = pd.to_datetime(aqi_df['time'])

        # Merge them on the 'time' column
        merged_df = pd.merge(weather_df, aqi_df, on='time')
        merged_df = merged_df.dropna() # Drop rows with missing data
        
        return merged_df
        
    except Exception as e:
        print(f"Error fetching training data: {e}")
        return None

def fetch_forecast_weather(lat, lon, forecast_days=3):
    """
    Fetches the weather forecast for the next few days to feed into our model.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&forecast_days={forecast_days}&timezone=auto"
    
    try:
        res = requests.get(url).json()
        if 'hourly' not in res:
             print(f"Forecast API Error: {res}")
             return None
             
        forecast_df = pd.DataFrame(res['hourly'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        return forecast_df
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return None