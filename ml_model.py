import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# In-memory registry to store trained models per target metric
SAVED_MODELS = {}

def create_lag_features(data, target_col, lags=7):
    """
    Constructs a pure 7-day rolling window dataset using only target observations,
    completely isolating features from metadata or timestamps (utcTimeMillis).
    """
    df = pd.DataFrame({target_col: data[target_col]})
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    
    # Drop initial NaN rows created by shifting
    df = df.dropna().reset_index(drop=True)
    
    # X strictly contains lag_1 through lag_7
    X = df[[f'lag_{i}' for i in range(1, lags + 1)]]
    y = df[target_col]
    return X, y

def train_satellite_model(data_path, target_column="clock_error"):
    """
    Loads data, isolates time-series lag features, trains a Gradient Boosting
    regressor, saves the model in memory, and returns evaluation metrics.
    """
    df = pd.read_csv(data_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X, y = create_lag_features(df, target_column, lags=7)
    
    # Sequential 80-20 train-test split for time-series validity
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=3, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Store trained model in persistent cache
    SAVED_MODELS[target_column] = model
    
    # Calculate performance metrics on test set
    y_pred = model.predict(X_test)
    y_mean = np.mean(y_test)
    
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_mean) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_actual": y_test.tolist(),
        "test_predictions": y_pred.tolist()
    }

def predict_day_8(last_7_days, target_column="clock_error"):
    """
    Accepts an input array of exactly 7 sequence values and generates 
    the 8th-day forecast value.
    """
    if target_column not in SAVED_MODELS:
        raise ValueError(f"Model for '{target_column}' is not trained yet.")
        
    model = SAVED_MODELS[target_column]
    
    # Match input vector directly to the 7 trained lag feature columns
    input_df = pd.DataFrame([last_7_days], columns=[f'lag_{i}' for i in range(1, 8)])
    prediction = model.predict(input_df)[0]
    
    return round(float(prediction), 4)