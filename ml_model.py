import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

SAVED_MODELS = {}

def remove_outliers_iqr(series):
    """Smooths telemetry phase spikes and clock jump outliers using Interquartile Range."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)

def engineer_features_from_sequence(seq):
    """
    Transforms a 7-day sequence array into a rich feature vector containing 
    lags, velocity/acceleration differences, rolling stats, and EMA weights.
    """
    seq = np.array(seq, dtype=float)
    feats = {}
    
    # 1. Base Lags (lag_1 = most recent x7, lag_7 = oldest x1)
    for i in range(1, 8):
        feats[f'lag_{i}'] = seq[-i]
        
    # 2. 1st & 2nd Order Differences (Velocity and Acceleration trends)
    feats['diff_1'] = seq[-1] - seq[-2]
    feats['diff_2'] = seq[-2] - seq[-3]
    feats['diff_2nd'] = feats['diff_1'] - feats['diff_2']
    
    # 3. Rolling Window Statistics
    feats['rolling_mean_3'] = np.mean(seq[-3:])
    feats['rolling_mean_7'] = np.mean(seq)
    feats['rolling_std_7'] = np.std(seq)
    feats['rolling_min_7'] = np.min(seq)
    feats['rolling_max_7'] = np.max(seq)
    feats['rolling_range_7'] = feats['rolling_max_7'] - feats['rolling_min_7']
    
    # 4. Exponential Moving Average (EMA) weighting
    weights = np.exp(np.linspace(-1.0, 0.0, 7))
    weights /= weights.sum()
    feats['ema_7'] = np.sum(seq * weights)

    df_feats = pd.DataFrame([feats])
    ordered_cols = [f'lag_{i}' for i in range(1, 8)] + [
        'diff_1', 'diff_2', 'diff_2nd',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
        'rolling_min_7', 'rolling_max_7', 'rolling_range_7', 'ema_7'
    ]
    return df_feats[ordered_cols]

def build_dataset_from_series(series, lags=7):
    """Constructs the complete supervised training matrix from input time-series data."""
    series = remove_outliers_iqr(series).reset_index(drop=True)
    X_list = []
    y_list = []
    
    for i in range(lags, len(series)):
        window = series.iloc[i-lags:i].values
        target = series.iloc[i]
        feat_df = engineer_features_from_sequence(window)
        X_list.append(feat_df)
        y_list.append(target)
        
    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list)
    return X, y

def train_satellite_model(data_path, target_column="clock_error"):
    """
    Trains a Gradient Boosting + Ridge ensemble using TimeSeriesSplit cross-validation
    and stores the fitted models in memory for prediction.
    """
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    target_series = df[target_column].dropna()
    X, y = build_dataset_from_series(target_series, lags=7)
    
    # 5-fold TimeSeriesSplit cross-validation evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        gbr_temp = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42)
        ridge_temp = Ridge(alpha=1.0)
        
        gbr_temp.fit(X_tr, y_tr)
        ridge_temp.fit(X_tr, y_tr)
        
        p_gbr = gbr_temp.predict(X_val)
        p_ridge = ridge_temp.predict(X_val)
        val_pred = 0.6 * p_gbr + 0.4 * p_ridge
        cv_scores.append(np.mean(np.abs(y_val - val_pred)))

    # Final production train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    
    final_gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42)
    final_ridge = Ridge(alpha=1.0)
    
    final_gbr.fit(X_train, y_train)
    final_ridge.fit(X_train, y_train)
    
    # Store ensemble pair in memory
    SAVED_MODELS[target_column] = {
        "gbr": final_gbr,
        "ridge": final_ridge
    }
    
    # Ensemble predictions (60% Gradient Boosting + 40% Ridge Regression)
    pred_gbr = final_gbr.predict(X_test)
    pred_ridge = final_ridge.predict(X_test)
    y_pred = 0.6 * pred_gbr + 0.4 * pred_ridge
    
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    y_mean = np.mean(y_test)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_mean) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_mae": float(np.mean(cv_scores)) if cv_scores else mae,
        "test_actual": y_test.tolist(),
        "test_predictions": y_pred.tolist()
    }

def predict_day_8(last_7_days, target_column="clock_error"):
    """Generates 8th-day error prediction using the trained ensemble models."""
    if target_column not in SAVED_MODELS:
        raise ValueError(f"Model for '{target_column}' is not trained yet.")
        
    model_dict = SAVED_MODELS[target_column]
    gbr = model_dict["gbr"]
    ridge = model_dict["ridge"]
    
    input_feats = engineer_features_from_sequence(last_7_days)
    
    pred_gbr = gbr.predict(input_feats)[0]
    pred_ridge = ridge.predict(input_feats)[0]
    
    ensemble_pred = 0.6 * pred_gbr + 0.4 * pred_ridge
    return round(float(ensemble_pred), 4)