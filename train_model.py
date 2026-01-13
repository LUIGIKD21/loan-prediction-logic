import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os


# Define the features the model expects (must match frontend form inputs)
FEATURES = ['Credit_Score', 'ApplicantIncome', 'LoanAmount', 'Gender']
TARGET = 'Loan_Status'
MODEL_FILENAME = 'loan_model.joblib'
SCALER_FILENAME = 'scaler.joblib'

def create_dummy_data():
    """
    Creates a simple, small dataset
    'Loan_Status': 1 = Approved, 0 = Rejected
    """
    np.random.seed(42)
    data_size = 200

    # Generate synthetic data
    credit_scores = np.random.randint(550, 850, size=data_size)
    incomes = np.random.randint(20000, 150000, size=data_size)
    loan_amounts = np.random.randint(5000, 500000, size=data_size)
    genders = np.random.choice(['Male', 'Female'], size=data_size)

    # Simple logic for "Approval":
    # Approved if Credit Score > 650 AND Loan Amount < (Income * 5)
    loan_status = []
    for i in range(data_size):
        if credit_scores[i] > 650 and loan_amounts[i] < (incomes[i] * 5):
            loan_status.append(1)
        else:
            loan_status.append(0)

    data = {
        'Credit_Score': credit_scores,
        'ApplicantIncome': incomes,
        'LoanAmount': loan_amounts,
        'Gender': genders,
        'Loan_Status': loan_status
    }
    return pd.DataFrame(data)

def train_and_save_model():
    """Trains the Random Forest model and saves the necessary objects."""
    print("Starting Model Training for MVP...")

    # 1. Load Data
    df = create_dummy_data()

    # 2. Preprocessing: Encode Gender (Male=1, Female=0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Separate features and target
    X = df[FEATURES]
    y = df[TARGET]

    # 3. Scaling Numerical Features (Credit Score, Income, Loan Amount)
    scaler = StandardScaler()
    numerical_cols = ['Credit_Score', 'ApplicantIncome', 'LoanAmount']

    # Fit the scaler on numerical columns
    X_scaled_values = scaler.fit_transform(X[numerical_cols])

    # Recombine scaled data and the encoded gender feature for training
    X_final = np.concatenate((X_scaled_values, X[['Gender']].values), axis=1)

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # 5. Train Model (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # 7. Save Model and Scaler (The MVP Deliverable)
    dump(model, MODEL_FILENAME)
    dump(scaler, SCALER_FILENAME)

    print(f"\nSUCCESS: '{MODEL_FILENAME}' and '{SCALER_FILENAME}' have been created.")
    print("You can now run 'app.py' to start the web server.")

if __name__ == '__main__':
    train_and_save_model()