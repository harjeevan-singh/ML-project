import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

MODEL_DIRECTORY = "models"
WINDOW_SIZE = 7

def get_model_path(target_column="clock_error"):
    return os.path.join(MODEL_DIRECTORY, f"satellite_model_{target_column}.pkl")

def remove_outliers_iqr(series, k=2.5):
    """Filter phase jumps and telemetry spikes using rolling interquartile range (IQR)."""
    s = pd.Series(series)
    q25 = s.rolling(window=14, min_periods=1).quantile(0.25)
    q75 = s.rolling(window=14, min_periods=1).quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    return s.clip(lower=lower_bound, upper=upper_bound).values

def engineer_features(seq, epoch_step=0):
    """
    Constructs a 32-dimensional feature vector:
    - 7 Lag values
    - 6 First-order velocity differences (rate of drift)
    - 5 Second-order acceleration differences (drift change)
    - Running Statistics (Mean, Std, Min, Max, Median, IQR)
    - Exponentially Weighted Moving Average (EWMA)
    - 7-Day Linear Trend Slope
    - Orbital Periodicity Cyclical Harmonics (Sine/Cosine phase components)
    - Step change & scale ratios
    """
    seq = np.array(seq, dtype=float)
    diffs1 = np.diff(seq)
    diffs2 = np.diff(diffs1)
    
    mean_val = np.mean(seq)
    std_val = np.std(seq)
    min_val = np.min(seq)
    max_val = np.max(seq)
    median_val = np.median(seq)
    iqr_val = np.percentile(seq, 75) - np.percentile(seq, 25)
    
    # Exponential weight vector emphasizing recent telemetry
    weights = np.exp(np.linspace(-1.0, 0.0, WINDOW_SIZE))
    weights /= weights.sum()
    ewma = np.sum(seq * weights)
    
    # Linear trend slope calculation
    x = np.arange(WINDOW_SIZE)
    slope = np.polyfit(x, seq, 1)[0]
    
    # Cyclical orbital harmonics (~12-hour GPS orbital period encoding)
    period = 12.0
    sin_time = np.sin(2 * np.pi * (epoch_step % period) / period)
    cos_time = np.cos(2 * np.pi * (epoch_step % period) / period)
    
    last_val = seq[-1]
    last_diff = diffs1[-1]
    overall_change = seq[-1] - seq[0]
    ratio_change = seq[-1] / (seq[0] + 1e-8)
    
    return np.hstack([
        seq,
        diffs1,
        diffs2,
        [mean_val, std_val, min_val, max_val, median_val, iqr_val, ewma, slope,
         sin_time, cos_time, last_val, last_diff, overall_change, ratio_change]
    ])

def create_sequences(values):
    cleaned_vals = remove_outliers_iqr(values)
    X, y_delta, y_actual, last_vals = [], [], [], []
    
    for i in range(len(cleaned_vals) - WINDOW_SIZE):
        seq = cleaned_vals[i : i + WINDOW_SIZE]
        target = cleaned_vals[i + WINDOW_SIZE]
        delta = target - seq[-1]
        
        X.append(engineer_features(seq, epoch_step=i + WINDOW_SIZE))
        y_delta.append(delta)
        y_actual.append(target)
        last_vals.append(seq[-1])
        
    return np.array(X), np.array(y_delta), np.array(y_actual), np.array(last_vals)

def train_satellite_model(csv_path, target_column="clock_error"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset missing at path: {csv_path}")

    df = pd.read_csv(csv_path)
    if "utcTimeMillis" in df.columns:
        df = df.sort_values("utcTimeMillis")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")

    values = pd.to_numeric(df[target_column], errors="coerce").dropna().values
    if len(values) < WINDOW_SIZE + 5:
        raise ValueError("Insufficient continuous sequence points for training.")

    X, y_delta, y_actual, last_vals = create_sequences(values)
    
    # Time-Series Validation using TimeSeriesSplit (prevents look-ahead bias)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    
    X_train, y_train_delta = X[train_idx], y_delta[train_idx]
    X_test = X[test_idx]
    y_test_actual = y_actual[test_idx]
    last_vals_test = last_vals[test_idx]

    # Primary Non-Linear Model: Tuned Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.025,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42
    )
    gbr.fit(X_train, y_train_delta)

    # Secondary Linear Stabilizer: Ridge Regression
    ridge = Ridge(alpha=1.5)
    ridge.fit(X_train, y_train_delta)

    metrics = {"mae": None, "rmse": None, "r2": None, "test_actual": [], "test_predictions": []}

    if len(X_test) > 0:
        pred_delta_gbr = gbr.predict(X_test)
        pred_delta_ridge = ridge.predict(X_test)
        
        # Optimized Blend (85% Gradient Boosting + 15% Ridge Linear Smoothing)
        blended_delta = 0.85 * pred_delta_gbr + 0.15 * pred_delta_ridge
        predictions = last_vals_test + blended_delta
        
        metrics["mae"] = float(mean_absolute_error(y_test_actual, predictions))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test_actual, predictions)))
        metrics["r2"] = float(r2_score(y_test_actual, predictions))
        metrics["test_actual"] = [float(v) for v in y_test_actual]
        metrics["test_predictions"] = [float(v) for v in predictions]

    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    saved_model = {
        "gbr": gbr,
        "ridge": ridge,
        "target_column": target_column,
        "window_size": WINDOW_SIZE
    }

    with open(get_model_path(target_column), "wb") as file:
        pickle.dump(saved_model, file)

    return metrics

def predict_day_8(last_7_days, target_column="clock_error"):
    if len(last_7_days) != WINDOW_SIZE:
        raise ValueError("Exactly 7 historical values required.")

    model_path = get_model_path(target_column)
    if not os.path.exists(model_path):
        fallback_path = get_model_path("clock_error")
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Model for '{target_column}' is not trained yet.")

    with open(model_path, "rb") as file:
        saved_model = pickle.load(file)

    gbr = saved_model["gbr"]
    ridge = saved_model["ridge"]
    
    features = engineer_features(last_7_days, epoch_step=0).reshape(1, -1)
    
    delta_gbr = gbr.predict(features)[0]
    delta_ridge = ridge.predict(features)[0]
    predicted_delta = 0.85 * delta_gbr + 0.15 * delta_ridge
    
    return float(last_7_days[-1] + predicted_delta)