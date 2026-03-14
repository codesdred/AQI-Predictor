# ml_engine/forecaster.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_aqi_model(X, y):
    """
    Trains an XGBoost model on the location's historical data.
    """
    # Split data chronologically (don't shuffle time-series data!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize the model 
    # (Kept lightweight so it trains instantly when a user searches a city)
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    
    # Train it
    model.fit(X_train, y_train)
    
    # Quick evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MAE: {mae:.2f}, R2 Score: {r2:.2f}")
    
    return model, mae

def predict_future_aqi(model, X_forecast):
    """
    Uses the trained model to predict future PM2.5 levels.
    """
    predictions = model.predict(X_forecast)
    
    # Ensure no negative PM2.5 predictions (models can sometimes dip below 0)
    predictions = [max(0, pred) for pred in predictions]
    
    return predictions