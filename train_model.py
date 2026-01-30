import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sqlalchemy import create_engine

# --- SETTINGS & CONFIGURATION ---
# Replace 'your_password' with your actual PostgreSQL password
DB_PASSWORD = 'admin123'
DB_URL = f'postgresql://postgres:{DB_PASSWORD}@localhost:5432/loan_db'

FEATURES = ['age', 'years_employed', 'annual_income', 'credit_score',
            'current_debt', 'loan_amount', 'debt_to_income_ratio',
            'occupation_status', 'loan_intent']
TARGET = 'loan_status'


def train_and_save_model():
    print("--- Starting ML Pipeline: PostgreSQL Integration ---")

    # 1. Load the 2025 Dataset
    try:
        df = pd.read_csv('Loan_approval_data_2025.csv')
        print(f"Successfully loaded {len(df)} rows of data.")
    except FileNotFoundError:
        print("ERROR: Loan_approval_data_2025.csv not found in folder.")
        return

    # 2. Seed to PostgreSQL (Data Governance Step)
    # This proves you can move data between files and enterprise databases
    try:
        engine = create_engine(DB_URL)
        df.to_sql('raw_training_data', engine, if_exists='replace', index=False)
        print("SUCCESS: Raw training data seeded to PostgreSQL table 'raw_training_data'.")
    except Exception as e:
        print(f"WARNING: Database sync failed. (Check if Postgres is running) | Error: {e}")

    # 3. Preprocessing: Encoding Categories
    # Converts 'Employed'/'Student' into 0s and 1s the model can understand
    X = pd.get_dummies(df[FEATURES], columns=['occupation_status', 'loan_intent'])
    y = df[TARGET]

    # Critical: Save the column order so app.py doesn't get confused
    dump(list(X.columns), 'model_columns.joblib')

    # 4. Preprocessing: Scaling Numbers
    # Ensures 'Income' (large numbers) doesn't drown out 'DTI' (small numbers)
    scaler = StandardScaler()
    num_cols = ['age', 'years_employed', 'annual_income', 'credit_score',
                'current_debt', 'loan_amount', 'debt_to_income_ratio']
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # 5. Training: Random Forest Classifier
    # We use 200 trees and a depth of 12 for high accuracy without overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=42
    )

    print("Training the Random Forest model...")
    model.fit(X_train, y_train)

    # 6. Evaluate & Save
    accuracy = model.score(X_test, y_test)
    print(f"TRAINING COMPLETE. Model Accuracy: {accuracy * 100:.2f}%")

    # Save the 'brains' of the app
    dump(model, 'loan_model.joblib')
    dump(scaler, 'scaler.joblib')
    print("Files saved: loan_model.joblib, scaler.joblib, model_columns.joblib")


if __name__ == '__main__':
    train_and_save_model()