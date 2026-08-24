document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname.toLowerCase();

    if (path.endsWith("dashboard.html") || path.endsWith("/dashboard") || path === "/") {
        initDashboard();
    } else if (path.endsWith("prediction.html") || path.endsWith("/prediction")) {
        initPredictionPage();
    } else if (path.endsWith("analytics.html") || path.endsWith("/analytics")) {
        initAnalyticsPage();
    }
});

async function initDashboard() {
    try {
        const response = await fetch('/api/evaluate?target_column=clock_error');
        const data = await response.json();

        if (data.success && data.metrics) {
            const maeElem = document.getElementById("mae-value");
            const rmseElem = document.getElementById("rmse-value");
            const statusElem = document.getElementById("accuracy-status");

            if (maeElem) maeElem.innerText = data.metrics.MAE;
            if (rmseElem) rmseElem.innerText = data.metrics.RMSE;
            if (statusElem) statusElem.innerText = data.metrics.Accuracy_Status;

            renderComparisonChart(".fake-chart", data.metrics.test_actual, data.metrics.test_predictions);
        }
    } catch (error) {
        console.error("Dashboard metrics error:", error);
    }
}

function initPredictionPage() {
    const predictForm = document.getElementById("predict-form");
    if (predictForm) {
        predictForm.addEventListener("submit", handlePredictionSubmit);
    }
}

async function handlePredictionSubmit(event) {
    event.preventDefault();
    const resultElement = document.getElementById("prediction-result");
    const inputField = document.getElementById("last-7-values");
    const errorTypeSelect = document.getElementById("errorType");

    if (!inputField || !resultElement) return;

    const values = inputField.value.split(',').map(v => parseFloat(v.trim()));

    if (values.length !== 7 || values.some(isNaN)) {
        resultElement.innerText = "Error: Please enter exactly 7 valid numbers.";
        resultElement.style.color = "#ff4d4d";
        return;
    }

    resultElement.innerText = "Running Gradient Boosting Ensemble...";
    resultElement.style.color = "#00d2ff";

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                last_7_days: values,
                target_column: errorTypeSelect ? errorTypeSelect.value : "clock_error"
            })
        });

        const data = await response.json();

        if (data.success) {
            resultElement.innerText = `Day 8 Predicted Error: ${data.prediction.toFixed(4)}`;
            resultElement.style.color = "#00ff88";
        } else {
            resultElement.innerText = `Error: ${data.message}`;
            resultElement.style.color = "#ff4d4d";
        }
    } catch (error) {
        resultElement.innerText = "Backend connection error.";
        resultElement.style.color = "#ff4d4d";
    }
}

function initAnalyticsPage() {
    loadAnalyticsMetrics();
    const trainBtn = document.getElementById("train-model-btn");
    if (trainBtn) {
        trainBtn.addEventListener("click", handleModelTraining);
    }
}

async function loadAnalyticsMetrics() {
    try {
        const response = await fetch('/api/evaluate?target_column=clock_error');
        const data = await response.json();

        if (data.success && data.metrics) {
            const maeElem = document.getElementById("analytics-mae");
            const rmseElem = document.getElementById("analytics-rmse");
            const accElem = document.getElementById("analytics-accuracy");

            if (maeElem) maeElem.innerText = data.metrics.MAE;
            if (rmseElem) rmseElem.innerText = data.metrics.RMSE;
            if (accElem) accElem.innerText = `${data.metrics.Accuracy_Status} (R²: ${data.metrics.R2_Score})`;

            renderComparisonChart("#analytics-chart-container", data.metrics.test_actual, data.metrics.test_predictions);
        }
    } catch (error) {
        console.error("Analytics fetch error:", error);
    }
}

function renderComparisonChart(selector, actuals, predictions) {
    const container = document.querySelector(selector);
    if (!container || !actuals || actuals.length === 0) return;

    const maxVal = Math.max(...actuals, ...predictions, 1);
    let html = '<div style="display:flex; align-items:flex-end; justify-content:space-around; height:180px; width:100%; padding:10px 0;">';

    actuals.slice(0, 8).forEach((val, idx) => {
        const h1 = Math.min(100, Math.max(10, (val / maxVal) * 100));
        const predVal = predictions[idx] !== undefined ? predictions[idx] : val;
        const h2 = Math.min(100, Math.max(10, (predVal / maxVal) * 100));

        html += `
            <div style="display:flex; flex-direction:column; align-items:center; width:10%;">
                <div style="display:flex; gap:3px; align-items:flex-end; height:140px; width:100%;">
                    <div style="height:${h1}%; width:50%; background:#FF69B4; border-radius:4px 4px 0 0;" title="Actual: ${val}"></div>
                    <div style="height:${h2}%; width:50%; background:#069494; border-radius:4px 4px 0 0;" title="Predicted: ${predVal}"></div>
                </div>
                <span style="font-size:10px; margin-top:6px; font-weight:bold; color:#94a3b8;">Pt ${idx + 1}</span>
            </div>
        `;
    });
    html += '</div>';

    container.innerHTML = html + `
        <div style="display:flex; justify-content:center; gap:20px; font-size:12px; margin-top:8px;">
            <span style="color:#FF69B4; font-weight:bold;">■ Actual Telemetry</span>
            <span style="color:#069494; font-weight:bold;">■ Model Prediction</span>
        </div>
    `;
}

async function handleModelTraining() {
    const statusBox = document.getElementById("train-status");
    if (statusBox) statusBox.innerText = "Training Gradient Boosting ensemble on dataset...";

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_column: "clock_error" })
        });

        const data = await response.json();

        if (data.success && statusBox) {
            statusBox.innerText = `Retrained successfully! MAE: ${data.metrics.mae.toFixed(4)}, R²: ${data.metrics.r2.toFixed(4)}`;
            loadAnalyticsMetrics();
        } else if (statusBox) {
            statusBox.innerText = `Training error: ${data.message}`;
        }
    } catch (error) {
        if (statusBox) statusBox.innerText = "Connection failure during training.";
    }
}