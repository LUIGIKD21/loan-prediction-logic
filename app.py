from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from joblib import load
from datetime import datetime
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

# --- DATABASE CONFIGURATION (LOCAL + CLOUD) ---
DB_URL = os.environ.get('DATABASE_URL')

# Fix for Render: SQLAlchemy requires 'postgresql://'
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)


LOCAL_DB = 'postgresql://postgres:admin123@localhost:5432/loan_db'

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL or LOCAL_DB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# --- DATABASE MODEL ---
class PredictionRecord(db.Model):
    __tablename__ = 'loan_history'
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    income = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    credit_score = db.Column(db.Integer)
    dti = db.Column(db.Float)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# --- INITIALIZE DATABASE TABLES ---
# This runs when Gunicorn loads the app, ensuring tables exist on Render
with app.app_context():
    try:
        db.create_all()
        print("Successfully connected to Database and verified tables.")
    except Exception as e:
        print(f"Database initialization error: {e}")

# --- LOAD ML COMPONENTS ---
try:
    MODEL = load('loan_model.joblib')
    SCALER = load('scaler.joblib')
    COLUMNS = load('model_columns.joblib')
    print("AI Model Components Loaded Successfully.")
except Exception as e:
    print(f"Warning: Could not load model files: {e}")


# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1. Prepare Input Data
        input_df = pd.DataFrame([{
            'age': int(data.get('age', 0)),
            'years_employed': float(data.get('years_employed', 0)),
            'annual_income': int(data.get('income', 0)),
            'credit_score': int(data.get('credit_score', 0)),
            'current_debt': int(data.get('current_debt', 0)),
            'loan_amount': int(data.get('loan_amount', 0)),
            'debt_to_income_ratio': float(data.get('dti', 0)),
            'occupation_status': data.get('occupation', 'Other'),
            'loan_intent': data.get('intent', 'Personal')
        }])

        # 2. Encode and Align
        input_encoded = pd.get_dummies(input_df)
        input_final = input_encoded.reindex(columns=COLUMNS, fill_value=0)

        # 3. Scale
        num_cols = ['age', 'years_employed', 'annual_income', 'credit_score',
                    'current_debt', 'loan_amount', 'debt_to_income_ratio']
        input_final[num_cols] = SCALER.transform(input_final[num_cols])

        # 4. Predict
        pred_idx = MODEL.predict(input_final)[0]
        status = "Eligible" if pred_idx == 1 else "Not Eligible"

        # Convert np.float64 to native Python float
        conf_raw = np.max(MODEL.predict_proba(input_final)) * 100
        conf = float(round(conf_raw, 2))

        # 5. Save to Database
        new_record = PredictionRecord(
            age=int(data.get('age', 0)),
            income=float(data.get('income', 0)),
            loan_amount=float(data.get('loan_amount', 0)),
            credit_score=int(data.get('credit_score', 0)),
            dti=float(data.get('dti', 0)),
            prediction=status,
            confidence=conf
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({"status": "success", "prediction": status, "confidence": conf})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/admin')
def admin_dashboard():
    try:
        records = PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).limit(50).all()
        return render_template('admin.html', records=records)
    except Exception as e:
        return f"Database error: {e}", 500


# --- LOCAL RUN (Ignored by Gunicorn) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)