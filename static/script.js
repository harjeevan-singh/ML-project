document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname.toLowerCase();
    if (path.includes("dashboard")) initDashboard();
    else if (path.includes("prediction")) initPredictionPage();
    else if (path.includes("analytics")) initAnalyticsPage();
});

async function initDashboard() {
    try {
        const response = await fetch('/api/evaluate?target_column=clock_error');
        const data = await response.json();
        if (data.success && data.metrics) {
            document.getElementById("mae-value").innerText = data.metrics.MAE;
            document.getElementById("rmse-value").innerText = data.metrics.RMSE;
            document.getElementById("accuracy-status").innerText = data.metrics.Accuracy_Status;
            renderComparisonChart(".fake-chart", data.metrics.test_actual, data.metrics.test_predictions);
        }
    } catch (e) { console.error("Dashboard error:", e); }
}

function initPredictionPage() {
    const predictForm = document.getElementById("predict-form");
    if (predictForm) predictForm.addEventListener("submit", handlePredictionSubmit);
}

async function handlePredictionSubmit(event) {
    event.preventDefault();
    const resultElement = document.getElementById("prediction-result");
    const messageElement = document.getElementById("predictionMessage");
    const inputField = document.getElementById("last-7-values");
    const errorTypeSelect = document.getElementById("errorType");

    const values = inputField.value.split(',').map(v => parseFloat(v.trim()));
    if (values.length !== 7 || values.some(isNaN)) {
        resultElement.innerText = "Error";
        messageElement.innerText = "Provide 7 numerical values separated by commas.";
        return;
    }

    resultElement.innerText = "...";
    try {
        const targetCol = errorTypeSelect ? errorTypeSelect.value : "clock_error";
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ last_7_days: values, target_column: targetCol })
        });
        const data = await response.json();
        if (data.success) {
            resultElement.innerText = data.prediction.toFixed(4);
            messageElement.innerText = `Day 8 predicted successfully.`;
        } else {
            resultElement.innerText = "Error";
            messageElement.innerText = data.message;
        }
    } catch (e) {
        resultElement.innerText = "Failed";
        messageElement.innerText = "Server communication failure.";
    }
}

async function initAnalyticsPage() {
    loadAnalyticsMetrics();
    const trainBtn = document.getElementById("train-model-btn");
    if (trainBtn) trainBtn.addEventListener("click", handleModelTraining);
}

async function loadAnalyticsMetrics() {
    try {
        const response = await fetch('/api/evaluate?target_column=clock_error');
        const data = await response.json();
        if (data.success && data.metrics) {
            document.getElementById("analytics-mae").innerText = data.metrics.MAE;
            document.getElementById("analytics-rmse").innerText = data.metrics.RMSE;
            document.getElementById("analytics-accuracy").innerText = `${data.metrics.Accuracy_Status} (R²: ${data.metrics.R2_Score})`;
            renderComparisonChart("#analytics-chart-container", data.metrics.test_actual, data.metrics.test_predictions);
        }
    } catch (e) { console.error("Analytics fetch error:", e); }
}

function renderComparisonChart(selector, actuals, predictions) {
    const container = document.querySelector(selector);
    if (!container || !actuals || actuals.length === 0) return;
    const maxVal = Math.max(...actuals, ...predictions, 1);
    let html = '<div style="display:flex; align-items:flex-end; justify-content:space-around; height:180px; width:100%; padding:10px 0;">';

    actuals.slice(0, 8).forEach((val, idx) => {
        const h1 = Math.min(100, Math.max(10, (val / maxVal) * 100));
        const pVal = predictions[idx] !== undefined ? predictions[idx] : val;
        const h2 = Math.min(100, Math.max(10, (pVal / maxVal) * 100));

        html += `
            <div style="display:flex; flex-direction:column; align-items:center; width:10%;">
                <div style="display:flex; gap:3px; align-items:flex-end; height:140px; width:100%;">
                    <div style="height:${h1}%; width:50%; background:#FF69B4; border-radius:4px 4px 0 0;" title="Actual: ${val}"></div>
                    <div style="height:${h2}%; width:50%; background:#069494; border-radius:4px 4px 0 0;" title="Predicted: ${pVal}"></div>
                </div>
                <span style="font-size:10px; margin-top:6px; color:#94a3b8;">P${idx + 1}</span>
            </div>`;
    });
    container.innerHTML = html + '</div>';
}

async function handleModelTraining() {
    const statusBox = document.getElementById("train-status");
    if (statusBox) statusBox.innerText = "Training model...";
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_column: "clock_error" })
        });
        const data = await response.json();
        if (data.success && statusBox) {
            statusBox.innerText = `Retrained! MAE: ${data.metrics.mae.toFixed(4)}, R²: ${data.metrics.r2.toFixed(4)}`;
            loadAnalyticsMetrics();
        }
    } catch (e) { if (statusBox) statusBox.innerText = "Training connection error."; }
}