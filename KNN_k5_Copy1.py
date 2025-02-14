from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("models/breast_cancer_knn.pkl")
scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Breast Cancer Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

