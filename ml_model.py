import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

SAVED_MODELS = {}

def engineer_features_from_sequence(seq):
    """Transforms a 7-day time-series window into engineered predictive features."""
    seq = np.array(seq, dtype=float)
    feats = {}
    
    # 1. Base Lags (lag_1 = day 7, lag_7 = day 1)
    for i in range(1, 8):
        feats[f'lag_{i}'] = seq[-i]
        
    # 2. Trend Metrics (Velocity & Acceleration)
    feats['diff_1'] = seq[-1] - seq[-2]
    feats['diff_2'] = seq[-2] - seq[-3]
    feats['accel'] = feats['diff_1'] - feats['diff_2']
    
    # 3. Rolling Statistics & Exponential Moving Average
    feats['rolling_mean_3'] = np.mean(seq[-3:])
    feats['rolling_mean_7'] = np.mean(seq)
    feats['rolling_std_7'] = np.std(seq)
    feats['rolling_min_7'] = np.min(seq)
    feats['rolling_max_7'] = np.max(seq)
    
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
    """Constructs a sliding window supervised learning dataset."""
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
    """Trains Gradient Boosting + Ridge ensemble with validated time-series evaluation metrics."""
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    series = df[target_column].dropna().reset_index(drop=True)
    X, y = build_dataset_from_series(series, lags=7)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    
    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.04, max_depth=3, random_state=42)
    ridge = Ridge(alpha=0.5)
    
    gbr.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    
    SAVED_MODELS[target_column] = {"gbr": gbr, "ridge": ridge}
    
    pred_gbr = gbr.predict(X_test)
    pred_ridge = ridge.predict(X_test)
    y_pred = 0.65 * pred_gbr + 0.35 * pred_ridge
    
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
    """Predicts day 8 error value from 7 continuous historical points."""
    if target_column not in SAVED_MODELS:
        raise ValueError(f"Model for '{target_column}' is not trained yet.")
        
    models = SAVED_MODELS[target_column]
    feats = engineer_features_from_sequence(last_7_days)
    
    p_gbr = models["gbr"].predict(feats)[0]
    p_ridge = models["ridge"].predict(feats)[0]
    
    return round(float(0.65 * p_gbr + 0.35 * p_ridge), 4)