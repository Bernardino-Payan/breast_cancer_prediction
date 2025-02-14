from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load trained model and scaler
MODEL_PATH = os.path.join("models", "breast_cancer_knn.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form
        features = [float(request.form[f'feature_{i}']) for i in range(9)]
        features_scaled = scaler.transform([features])  # Scale input

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
