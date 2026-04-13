"""
Wine Quality Regression Model Training Script
Author: Reetam Dey - 2022BCS0120
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Dataset URL (UCI Wine Quality - Red Wine)
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def load_data():
    """Load Wine Quality dataset from UCI repository."""
    print("Loading Wine Quality dataset...")

    # Download dataset from UCI
    df = pd.read_csv(DATASET_URL, sep=';')

    print(f"Dataset loaded successfully with {df.shape[0]} samples and {df.shape[1]} features")
    return df


def preprocess_data(df):
    """Preprocess data: split into features and target, then train-test split."""
    print("Preprocessing data...")

    # Features (all columns except 'quality')
    X = df.drop('quality', axis=1)
    # Target (quality score)
    y = df['quality']

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train RandomForestRegressor model."""
    print("Training RandomForestRegressor model...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model using MSE and R2 Score."""
    print("Evaluating model...")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print required logs
    print(f"MSE: {mse}")
    print(f"R2: {r2}")

    return mse, r2


def save_artifacts(model, mse, r2):
    """Save model and metrics to files."""
    print("Saving model and metrics...")

    # Save model as model.pkl
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl")

    # Save metrics as metrics.json
    metrics = {
        "mse": mse,
        "r2": r2
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to metrics.json")


def main():
    """Main function to run the training pipeline."""
    print("=" * 50)
    print("Wine Quality Regression Training Pipeline")
    print("=" * 50)

    # Load data
    df = load_data()

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    mse, r2 = evaluate_model(model, X_test, y_test)

    # Save artifacts
    save_artifacts(model, mse, r2)

    print("=" * 50)
    print("Run completed by: Reetam Dey - 2022BCS0120")
    print("=" * 50)


if __name__ == "__main__":
    main()
