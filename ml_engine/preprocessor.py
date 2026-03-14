# ml_engine/preprocessor.py
import pandas as pd

def engineer_features(df):
    """
    Takes raw dataframe and extracts time-based features.
    Works for both training data and future forecast data.
    """
    df = df.copy()
    
    # Extract temporal features from the datetime column
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def prepare_training_data(merged_df):
    """
    Prepares the X (features) and y (target) for training.
    """
    # 1. Engineer the time features
    df = engineer_features(merged_df)
    
    # 2. Define our features and target
    # Note: Open-Meteo returns 'pm2_5' as the target variable
    features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                'precipitation', 'hour', 'day_of_week', 'month', 'is_weekend']
    target = 'pm2_5'
    
    # Drop any rows where PM2.5 might be NaN from the API
    df = df.dropna(subset=[target] + features)
    
    X = df[features]
    y = df[target]
    
    return X, y, df

def prepare_forecast_data(forecast_df):
    """
    Prepares the future weather data to be fed into the trained model.
    """
    df = engineer_features(forecast_df)
    features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                'precipitation', 'hour', 'day_of_week', 'month', 'is_weekend']
    
    # Fill any missing future weather data with the previous hour's data (forward fill)
    df[features] = df[features].ffill() 
    
    X_forecast = df[features]
    return X_forecast, df