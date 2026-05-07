from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    scaled_data = scaler.transform(features)

    prediction = model.predict(scaled_data)

    result = "Fraud" if prediction[0] == 1 else "Not Fraud"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)