from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

# --- Configuration & Model Loading ---
MODEL_FILENAME = 'loan_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
MODEL = None
SCALER = None

# Features must be in the exact order the model was trained on
MODEL_FEATURES = ['Credit_Score', 'ApplicantIncome', 'LoanAmount', 'Gender']

def load_ml_objects():
    """Loads the trained model and scaler into memory once."""
    global MODEL, SCALER
    try:
        if os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME):
            MODEL = load(MODEL_FILENAME)
            SCALER = load(SCALER_FILENAME)
            print(f"--- SUCCESS: ML Model and Scaler loaded ---")
        else:
            print(f"--- ERROR: Model files missing. Run 'train_model.py' first ---")
    except Exception as e:
        print(f"--- CRITICAL ERROR: Could not load ML objects: {e} ---")

# Load model on startup
with app.app_context():
    load_ml_objects()

# --- Internal Prediction Logic ---

def preprocess_and_predict(data):
    """Prepares raw input data and runs it through the ML model."""
    if MODEL is None or SCALER is None:
        return "Model not initialized", 0.0

    try:
        # 1. Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=MODEL_FEATURES)

        # 2. Preprocessing: Encode Gender (Male=1, Female=0)
        # This MUST match the encoding used in train_model.py
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)

        # 3. Preprocessing: Scale numerical features
        numerical_cols = ['Credit_Score', 'ApplicantIncome', 'LoanAmount']
        scaled_values = SCALER.transform(input_df[numerical_cols])

        # 4. Reconstruct feature array [ScaledNumerical, EncodedGender]
        final_features = np.concatenate((scaled_values, input_df[['Gender']].values), axis=1)

        # 5. Execute Prediction
        prediction = MODEL.predict(final_features)[0]
        prediction_proba = MODEL.predict_proba(final_features)[0]

        # Map 1 to Approved, 0 to Rejected
        result = "Approved" if prediction == 1 else "Rejected"
        confidence = round(prediction_proba[prediction] * 100, 2)

        return result, confidence

    except Exception as e:
        print(f"Prediction error: {e}")
        return str(e), 0.0

# --- API Routes ---

@app.route('/')
def home():
    """Serves the web interface from templates/index.html."""
    try:
        return render_template('index.html')
    except Exception:
        # Fallback if index.html is missing
        return jsonify({
            "status": "Online",
            "message": "API is running, but index.html was not found in the templates folder.",
            "endpoint": "/predict (POST)"
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive loan details and return a prediction."""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded on server."}), 503

    try:
        raw_data = request.get_json()

        # Extract and format incoming JSON
        processed_input = {
            'Credit_Score': int(raw_data.get('credit_score', 0)),
            'ApplicantIncome': int(raw_data.get('income', 0)),
            'LoanAmount': int(raw_data.get('loan_amount', 0)),
            'Gender': raw_data.get('gender', 'Male')
        }

        result, confidence = preprocess_and_predict(processed_input)

        if result in ["Approved", "Rejected"]:
            return jsonify({
                "status": "success",
                "prediction": result,
                "confidence": confidence
            })
        else:
            return jsonify({"status": "error", "message": result}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Running on default port 5000
    app.run(debug=True, port=5000)