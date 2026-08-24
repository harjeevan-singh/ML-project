import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SAVED_MODELS = {}
SCALERS = {}

def engineer_features_from_sequence(seq):
    """Transforms a 7-day time-series sequence into rich trend, momentum, and statistical features."""
    seq = np.array(seq, dtype=float)
    feats = {}
    
    # 1. Base Lags (lag_1 = Day 7, lag_7 = Day 1)
    for i in range(1, 8):
        feats[f'lag_{i}'] = seq[-i]
        
    # 2. Velocity and Acceleration (1st & 2nd Order Differences)
    feats['diff_1'] = seq[-1] - seq[-2]
    feats['diff_2'] = seq[-2] - seq[-3]
    feats['accel'] = feats['diff_1'] - feats['diff_2']
    
    # 3. Aggregations & Moving Averages
    feats['rolling_mean_3'] = np.mean(seq[-3:])
    feats['rolling_mean_7'] = np.mean(seq)
    feats['rolling_std_7'] = np.std(seq)
    feats['rolling_min_7'] = np.min(seq)
    feats['rolling_max_7'] = np.max(seq)
    
    # Exponential Weighted Moving Average (EWMA)
    weights = np.exp(np.linspace(-1.0, 0.0, 7))
    weights /= weights.sum()
    feats['ema_7'] = np.sum(seq * weights)

    df_feats = pd.DataFrame([feats])
    ordered_cols = [f'lag_{i}' for i in range(1, 8)] + [
        'diff_1', 'diff_2', 'accel',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
        'rolling_min_7', 'rolling_max_7', 'ema_7'
    ]
    return df_feats[ordered_cols]

def build_dataset_from_series(series, lags=7):
    """Constructs sliding window matrix from target error values."""
    X_list, y_list = [], []
    for i in range(lags, len(series)):
        window = series.iloc[i-lags:i].values
        target = series.iloc[i]
        X_list.append(engineer_features_from_sequence(window))
        y_list.append(target)
        
    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list)
    return X, y

def train_satellite_model(data_path, target_column="clock_error"):
    """Trains a scaled Random Forest + Gradient Boosting ensemble model optimized for high R² and low MAE."""
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    series = df[target_column].dropna().reset_index(drop=True)
    X, y = build_dataset_from_series(series, lags=7)
    
    # Chronological Train/Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimized High-Performance Estimators
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    
    gbr.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    
    SAVED_MODELS[target_column] = {"gbr": gbr, "rf": rf}
    SCALERS[target_column] = scaler
    
    # Weighted Ensemble Predictions
    p_gbr = gbr.predict(X_test_scaled)
    p_rf = rf.predict(X_test_scaled)
    y_pred = 0.5 * p_gbr + 0.5 * p_rf
    
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_actual": y_test.tolist(),
        "test_predictions": y_pred.tolist()
    }

def predict_day_8(last_7_days, target_column="clock_error"):
    """Predicts Day 8 satellite error value using calibrated feature scaling."""
    if target_column not in SAVED_MODELS:
        raise ValueError(f"Model for '{target_column}' is not trained yet.")
        
    gbr = SAVED_MODELS[target_column]["gbr"]
    rf = SAVED_MODELS[target_column]["rf"]
    scaler = SCALERS[target_column]
    
    raw_feats = engineer_features_from_sequence(last_7_days)
    scaled_feats = scaler.transform(raw_feats)
    
    pred_gbr = gbr.predict(scaled_feats)[0]
    pred_rf = rf.predict(scaled_feats)[0]
    
    ensemble_pred = 0.5 * pred_gbr + 0.5 * pred_rf
    return round(float(ensemble_pred), 4)