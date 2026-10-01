import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}

    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f in FEATURES}
            df = pd.DataFrame([values], columns=FEATURES)

            scaled = scaler.transform(df)
            prob = model.predict_proba(scaled)[0][1]

            if prob < 0.33:
                level, color = "Low risk", "green"
            elif prob < 0.66:
                level, color = "Moderate risk", "orange"
            else:
                level, color = "High risk", "red"

            result = {"prob": round(prob * 100, 1), "level": level, "color": color}
        except (ValueError, KeyError):
            result = {"error": "Please fill in all fields with valid numbers."}

    return render_template("index.html", result=result, values=values)

if __name__ == "__main__":
    app.run(debug=True)