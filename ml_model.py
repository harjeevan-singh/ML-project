import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SAVED_MODELS = {}
SCALERS = {}

def engineer_features_from_sequence(seq):
    """Transforms a 7-day time-series window into trend and momentum metrics."""
    seq = np.array(seq, dtype=float)
    feats = {}
    
    # 1. Base Lags (lag_1 = Day 7, lag_7 = Day 1)
    for i in range(1, 8):
        feats[f'lag_{i}'] = seq[-i]
        
    # 2. Differences & Acceleration
    feats['diff_1'] = seq[-1] - seq[-2]
    feats['diff_2'] = seq[-2] - seq[-3]
    feats['accel'] = feats['diff_1'] - feats['diff_2']
    
    # 3. Rolling Aggregations
    feats['rolling_mean_3'] = np.mean(seq[-3:])
    feats['rolling_mean_7'] = np.mean(seq)
    feats['rolling_std_7'] = np.std(seq)
    feats['rolling_min_7'] = np.min(seq)
    feats['rolling_max_7'] = np.max(seq)
    
    # Exponential Moving Average
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
    """Constructs differenced targets (delta prediction) to remove drift bias."""
    X_list, y_list, y_diff_list = [], [], []
    series_vals = series.values
    
    for i in range(lags, len(series_vals)):
        window = series_vals[i-lags:i]
        target = series_vals[i]
        # Train model on step change (delta) rather than unstationary magnitude
        delta = target - window[-1]
        
        X_list.append(engineer_features_from_sequence(window))
        y_list.append(target)
        y_diff_list.append(delta)
        
    X = pd.concat(X_list, ignore_index=True)
    y_actual = np.array(y_list)
    y_diff = np.array(y_diff_list)
    return X, y_actual, y_diff

def train_satellite_model(data_path, target_column="clock_error"):
    """Trains Gradient Boosting + Random Forest ensemble on step-differenced GNSS series."""
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    series = df[target_column].dropna().reset_index(drop=True)
    X, y_actual, y_diff = build_dataset_from_series(series, lags=7)
    
    # Train/Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_diff_train, y_diff_test = y_diff[:split], y_diff[split:]
    y_actual_test = y_actual[split:]
    last_lags_test = X_test['lag_1'].values
    
    # MinMax Feature Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Estimators
    gbr = GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=4, random_state=42)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    
    gbr.fit(X_train_scaled, y_diff_train)
    rf.fit(X_train_scaled, y_diff_train)
    
    SAVED_MODELS[target_column] = {"gbr": gbr, "rf": rf}
    SCALERS[target_column] = scaler
    
    # Reconstruct Day 8 Absolute Prediction from Predicted Delta
    pred_deltas = 0.55 * gbr.predict(X_test_scaled) + 0.45 * rf.predict(X_test_scaled)
    y_pred_actual = last_lags_test + pred_deltas
    
    mae = float(mean_absolute_error(y_actual_test, y_pred_actual))
    rmse = float(np.sqrt(mean_squared_error(y_actual_test, y_pred_actual)))
    r2 = float(r2_score(y_actual_test, y_pred_actual))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_actual": y_actual_test.tolist(),
        "test_predictions": y_pred_actual.tolist()
    }

def predict_day_8(last_7_days, target_column="clock_error"):
    """Predicts absolute 8th-day value from historical input vector."""
    if target_column not in SAVED_MODELS:
        raise ValueError(f"Model for '{target_column}' is not trained yet.")
        
    gbr = SAVED_MODELS[target_column]["gbr"]
    rf = SAVED_MODELS[target_column]["rf"]
    scaler = SCALERS[target_column]
    
    raw_feats = engineer_features_from_sequence(last_7_days)
    scaled_feats = scaler.transform(raw_feats)
    
    pred_delta = 0.55 * gbr.predict(scaled_feats)[0] + 0.45 * rf.predict(scaled_feats)[0]
    predicted_val = last_7_days[-1] + pred_delta
    
    return round(float(predicted_val), 4)