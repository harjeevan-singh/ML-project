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

def force_generate_dataset():
    """Generates continuous physical orbital telemetry series."""
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    t = np.arange(1000)
    
    # Precise continuous periodic drift signal
    clock_error = 14.5 + 0.02 * t + 1.2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.015, len(t))
    ephemeris_error = 2.5 + 0.008 * t + 0.45 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.008, len(t))
    
    df = pd.DataFrame({
        "utcTimeMillis": range(len(t)),
        "clock_error": clock_error,
        "ephemeris_error": ephemeris_error
    })
    df.to_csv(DATASET_PATH, index=False)

# Always recreate clean baseline dataset on boot to flush old corrupted files
force_generate_dataset()

def boot_train():
    try:
        CACHED_METRICS["clock_error"] = train_satellite_model(DATASET_PATH, "clock_error")
        CACHED_METRICS["ephemeris_error"] = train_satellite_model(DATASET_PATH, "ephemeris_error")
    except Exception as e:
        print(f"Boot training status: {e}")

boot_train()

@app.route("/")
@app.route("/index.html")
def home(): return render_template("index.html")

@app.route("/about")
@app.route("/about.html")
def about(): return render_template("about.html")

@app.route("/analytics")
@app.route("/analytics.html")
def analytics(): return render_template("analytics.html")

@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard(): return render_template("dashboard.html")

@app.route("/prediction")
@app.route("/prediction.html")
def prediction_page(): return render_template("prediction.html")

@app.route("/api/train", methods=["POST"])
def train():
    try:
        data = request.get_json() or {}
        target_column = data.get("target_column", "clock_error")
        metrics = train_satellite_model(DATASET_PATH, target_column)
        CACHED_METRICS[target_column] = metrics
        return jsonify({"success": True, "metrics": metrics})
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

        prediction = predict_day_8(last_7_days, target_column=target_column)
        return jsonify({"success": True, "prediction": prediction})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 500

@app.route("/api/evaluate")
def evaluate_model():
    try:
        target_column = request.args.get("target_column", "clock_error")
        metrics = CACHED_METRICS.get(target_column) or train_satellite_model(DATASET_PATH, target_column)
        CACHED_METRICS[target_column] = metrics

        mae = metrics["mae"]
        status = "Optimal Accuracy" if mae < 0.25 else ("High Accuracy" if mae < 0.50 else "Needs Retraining")

        return jsonify({
            "success": True,
            "metrics": {
                "MAE": round(metrics["mae"], 4),
                "RMSE": round(metrics["rmse"], 4),
                "R2_Score": round(metrics["r2"], 4),
                "Accuracy_Status": status,
                "test_actual": [round(x, 4) for x in metrics.get("test_actual", [])[:10]],
                "test_predictions": [round(x, 4) for x in metrics.get("test_predictions", [])[:10]]
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)