from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from twilio.rest import Client

app = Flask(__name__)
app.secret_key = "SK4fe5f16babd2aa415c62d6fc7ea10503"

# ======================================================
#                 TWILIO CONFIG (NO ENV VARS)
# ======================================================

ACCOUNT_SID = "AC7d247a2a98284b168aa079c8e50d7742"
AUTH_TOKEN  = "60d3b89f5e16d948be7eeee2ed117413"
TWILIO_FROM = "+19202676633"

print("DEBUG: FINAL Twilio SID Used =", ACCOUNT_SID)
print("DEBUG: FINAL Twilio FROM Used =", TWILIO_FROM)

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Global variables
latest_result = None
alert_status_message = None

# ======================================================
#               MODEL CONFIG (SAME AS TRAINING)
# ======================================================

NUMERIC_COLS = [
    "temperature", "humidity", "rainfall", "water_pH",
    "turbidity", "bacteria_level", "population_density",
    "reported_cases"
]
NUM_NUMERIC_FEATURES = len(NUMERIC_COLS)
REPORTED_CASES_INDEX = NUMERIC_COLS.index("reported_cases")
GRU_INPUT_SIZE = 10
GRU_OUT_SIZE = 7

# ======================================================
#                    REGRESSION MODEL
# ======================================================

class GRURegression(nn.Module):
    def __init__(self, input_size=GRU_INPUT_SIZE, hidden_size=128, num_layers=2, out_size=GRU_OUT_SIZE, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

reg_model = GRURegression()
reg_model.load_state_dict(torch.load("gru_regression_model.pth", map_location="cpu"))
reg_model.eval()

# Load scalers + encoders for regression
reg_scaler = joblib.load("regression_forecast_scaler.pkl")
reg_region_enc = joblib.load("regression_region_encoder.pkl")
reg_district_enc = joblib.load("regression_district_encoder.pkl")

# ======================================================
#                CLASSIFICATION MODEL
# ======================================================

class RiskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

clf_model = RiskClassifier(10)
clf_model.load_state_dict(torch.load("gru_classifier_model.pth", map_location="cpu"))
clf_model.eval()

clf_scaler = joblib.load("classification_risk_scaler.pkl")
clf_region_enc = joblib.load("classification_region_encoder.pkl")
clf_district_enc = joblib.load("classification_district_encoder.pkl")
clf_label_enc = joblib.load("classification_risk_label_encoder.pkl")

# ======================================================
#                     ROUTES
# ======================================================

@app.route("/", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username")
    password = request.form.get("password")

    if username == "admin" and password == "admin123":
        return redirect(url_for("data_entry_page"))

    return "<h2>Invalid Credentials</h2><p><a href='/'>Go back</a></p>"

# ======================================================
#               DATA ENTRY (GET + POST)
# ======================================================

@app.route("/data-entry", methods=["GET", "POST"])
def data_entry_page():
    global latest_result, alert_status_message

    if request.method == "POST":
        alert_status_message = None

        # ----- Raw inputs (normal numbers, not normalized) -----
        region = request.form["region"]
        district = request.form["district"]

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])
        water_pH = float(request.form["water_pH"])
        turbidity = float(request.form["turbidity"])
        bacteria_level = float(request.form["bacteria_level"])
        population_density = float(request.form["population_density"])
        diarrhea_cases = float(request.form["diarrhea_cases"])
        vomiting_cases = float(request.form["vomiting_cases"])

        reported_cases = diarrhea_cases + vomiting_cases

        numeric_vals = [
            temperature, humidity, rainfall, water_pH,
            turbidity, bacteria_level, population_density, reported_cases
        ]

        latest_result = {}

        # ======================================================
        #                 CLASSIFICATION PIPELINE
        # ======================================================
        region_clf = clf_region_enc.transform([region])[0]
        district_clf = clf_district_enc.transform([district])[0]

        # Build a DataFrame row with correct columns
        clf_row = numeric_vals + [region_clf, district_clf]
        df_clf_input = pd.DataFrame(
            [clf_row],
            columns=NUMERIC_COLS + ["region_enc", "district_enc"]
        )

        # Scale ONLY numeric columns (same as in training)
        df_clf_input[NUMERIC_COLS] = clf_scaler.transform(df_clf_input[NUMERIC_COLS])

        X_clf = df_clf_input.values.astype(np.float32)

        with torch.no_grad():
            logits = clf_model(torch.tensor(X_clf))
            pred_idx = int(torch.argmax(logits, dim=1).item())
            risk_label = clf_label_enc.inverse_transform([pred_idx])[0]

        # ======================================================
        #                 REGRESSION PIPELINE (GRU)
        # ======================================================
        region_reg = reg_region_enc.transform([region])[0]
        district_reg = reg_district_enc.transform([district])[0]

        reg_row = numeric_vals + [region_reg, district_reg]
        df_reg_input = pd.DataFrame(
            [reg_row],
            columns=NUMERIC_COLS + ["region_enc", "district_enc"]
        )

        # Scale numeric cols for regression
        df_reg_input[NUMERIC_COLS] = reg_scaler.transform(df_reg_input[NUMERIC_COLS])

        # Create a 30-day sequence by repeating the same scaled row
        df_seq = pd.concat([df_reg_input] * 30, ignore_index=True)

        seq_array = df_seq.values.astype(np.float32)
        seq_tensor = torch.tensor(seq_array).unsqueeze(0)  # (1, 30, 10)

        with torch.no_grad():
            forecast_scaled = reg_model(seq_tensor).numpy().flatten()

        # Inverse transform reported_cases from standardized scale
        rep_mean = reg_scaler.mean_[REPORTED_CASES_INDEX]
        rep_scale = np.sqrt(reg_scaler.var_[REPORTED_CASES_INDEX])

        forecast_real = (forecast_scaled * rep_scale) + rep_mean
        forecast_list = [float(max(0.0, x)) for x in forecast_real]

        # ======================================================
        #                 STORE RESULT FOR DASHBOARD
        # ======================================================
        latest_result = {
            "risk_label": risk_label,
            "predicted_class": pred_idx,
            "forecast": forecast_list,
            "inputs": {
                "region": region,
                "district": district,
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "water_pH": water_pH,
                "turbidity": turbidity,
                "bacteria_level": bacteria_level,
                "population_density": population_density,
                "diarrhea_cases": diarrhea_cases,
                "vomiting_cases": vomiting_cases,
                "reported_cases": reported_cases,
            },
        }

        return redirect(url_for("dashboard_page"))

    return render_template("data_entry.html")

# ======================================================
#                  SEND ALERT SMS
# ======================================================

@app.route("/send-alert", methods=["POST"])
def send_alert():
    global latest_result, alert_status_message

    phone = request.form.get("phone")

    if not latest_result:
        alert_status_message = "❌ No prediction available to send."
        return redirect(url_for("dashboard_page"))

    if latest_result["risk_label"] != "High":
        alert_status_message = "⚠ Risk is not High. SMS not allowed."
        return redirect(url_for("dashboard_page"))

    msg = f"""
🚨 HIGH WATERBORNE DISEASE RISK ALERT 🚨

Region: {latest_result['inputs']['region']}
District: {latest_result['inputs']['district']}
Risk Level: HIGH
"""

    try:
        twilio_client.messages.create(
            body=msg,
            from_=TWILIO_FROM,
            to=phone
        )
        alert_status_message = "✔ Alert sent successfully!"
    except Exception as e:
        alert_status_message = "❌ SMS Error: " + str(e)

    return redirect(url_for("dashboard_page"))

# ======================================================
#                     DASHBOARD
# ======================================================

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template(
        "dashboard.html",
        latest=latest_result["inputs"] if latest_result else None,
        risk_level=latest_result["risk_label"] if latest_result else None,
        forecast=latest_result["forecast"] if latest_result else [],
        alert_status=alert_status_message
    )

# ======================================================
#                   RUN SERVER
# ======================================================

if __name__ == "__main__":
    import os

    # Check if running on Render (Render sets the PORT variable)
    port = os.environ.get("PORT")

    if port:
        # Running on Render
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # Running locally
        app.run(debug=True)

