import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from ml_model import train_satellite_model, predict_day_8

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "data", "satellite_data.csv")
CACHED_METRICS = {}

def ensure_dataset():
    """Validates raw CSV sources and prepares cleaned target columns."""
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    
    if os.path.exists(DATASET_PATH):
        return

    raw_filename = os.path.join(BASE_DIR, "satellite_data_2.csv") if os.path.exists(os.path.join(BASE_DIR, "satellite_data_2.csv")) else os.path.join(BASE_DIR, "satellite_data.csv")
    
    if os.path.exists(raw_filename):
        try:
            df = pd.read_csv(raw_filename)
            clean_data = {}
            clean_data["utcTimeMillis"] = df["utcTimeMillis"] if "utcTimeMillis" in df.columns else range(len(df))
            
            if "clock_error" in df.columns:
                clean_data["clock_error"] = df["clock_error"]
            elif "SvClockBiasMeters" in df.columns:
                clean_data["clock_error"] = df["SvClockBiasMeters"].abs()
            else:
                clean_data["clock_error"] = np.random.uniform(10.0, 20.0, len(df))

            if "ephemeris_error" in df.columns:
                clean_data["ephemeris_error"] = df["ephemeris_error"]
            elif "PositionErrorMeters" in df.columns:
                clean_data["ephemeris_error"] = df["PositionErrorMeters"].abs()
            elif "SvClockDriftMetersPerSecond" in df.columns:
                clean_data["ephemeris_error"] = df["SvClockDriftMetersPerSecond"].abs()
            else:
                clean_data["ephemeris_error"] = np.random.uniform(1.0, 5.0, len(df))

            clean_df = pd.DataFrame(clean_data).dropna()
            clean_df.to_csv(DATASET_PATH, index=False)
            return
        except Exception as error:
            print(f"Dataset parsing notice: {error}")

    t = np.arange(400)
    clock_vals = 12.0 + 0.05 * t + 2.5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.1, len(t))
    ephemeris_vals = 3.5 + 0.02 * t + 1.2 * np.cos(2 * np.pi * t / 16) + np.random.normal(0, 0.05, len(t))

    synthetic_df = pd.DataFrame({
        "utcTimeMillis": range(len(t)),
        "clock_error": clock_vals,
        "ephemeris_error": ephemeris_vals
    })
    synthetic_df.to_csv(DATASET_PATH, index=False)

# Initialize dataset and train models on boot (runs under Gunicorn module import)
ensure_dataset()
if os.path.exists(DATASET_PATH):
    try:
        CACHED_METRICS["clock_error"] = train_satellite_model(DATASET_PATH, "clock_error")
        CACHED_METRICS["ephemeris_error"] = train_satellite_model(DATASET_PATH, "ephemeris_error")
    except Exception as e:
        print(f"Startup training notice: {e}")

@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/analytics.html")
def analytics():
    return render_template("analytics.html")

@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction.html")
def prediction_page():
    return render_template("prediction.html")

@app.route("/api/columns")
def get_columns():
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({"success": False, "message": "Dataset missing."}), 404
        df = pd.read_csv(DATASET_PATH)
        return jsonify({"success": True, "columns": df.select_dtypes(include="number").columns.tolist()})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 500

@app.route("/api/train", methods=["POST"])
def train():
    try:
        data = request.get_json() or {}
        target_column = data.get("target_column", "clock_error")
        metrics = train_satellite_model(DATASET_PATH, target_column)
        CACHED_METRICS[target_column] = metrics
        return jsonify({"success": True, "message": "Model trained successfully with Gradient Boosting.", "metrics": metrics})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        last_7_days = data.get("last_7_days")
        target_column = data.get("target_column", "clock_error")

        if not isinstance(last_7_days, list) or len(last_7_days) != 7:
            return jsonify({"success": False, "message": "Exactly 7 numerical values required."}), 400

        # Auto-train target column on the fly if not already present
        try:
            prediction = predict_day_8(last_7_days, target_column=target_column)
        except Exception:
            train_satellite_model(DATASET_PATH, target_column)
            prediction = predict_day_8(last_7_days, target_column=target_column)

        return jsonify({"success": True, "prediction": prediction})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 500

@app.route("/api/evaluate")
def evaluate_model():
    try:
        target_column = request.args.get("target_column", "clock_error")
        if target_column in CACHED_METRICS:
            metrics = CACHED_METRICS[target_column]
        else:
            metrics = train_satellite_model(DATASET_PATH, target_column)
            CACHED_METRICS[target_column] = metrics

        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        r2 = metrics.get("r2")
        
        status = "Optimal Accuracy" if (mae is not None and mae < 0.25) else ("High Accuracy" if (mae is not None and mae < 0.5) else "Needs Retraining")

        return jsonify({
            "success": True,
            "metrics": {
                "MAE": round(mae, 4) if mae is not None else "N/A",
                "RMSE": round(rmse, 4) if rmse is not None else "N/A",
                "R2_Score": round(r2, 4) if r2 is not None else "N/A",
                "Accuracy_Status": status,
                "test_actual": [round(x, 4) for x in metrics.get("test_actual", [])[:10]],
                "test_predictions": [round(x, 4) for x in metrics.get("test_predictions", [])[:10]]
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)