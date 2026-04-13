# Wine Quality MLOps

**Wine Quality Regression with GitHub Actions CI/CD**

## Project Overview

This project implements a complete MLOps pipeline for predicting wine quality using machine learning. It includes automated training, evaluation, and CI/CD integration with GitHub Actions.

## Author

**Reetam Dey** - 2022BCS0120

## Dataset

The project uses the **Wine Quality (Red)** dataset from the UCI Machine Learning Repository:
- **Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- **Samples:** 1,599
- **Features:** 11 physicochemical properties
- **Target:** Quality score (0-10)

### Features
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

## Project Structure

```
wine-quality-mlops/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml    # GitHub Actions CI/CD workflow
├── train.py                   # Training script
├── requirements.txt           # Python dependencies
├── model.pkl                  # Trained model (generated)
├── metrics.json               # Evaluation metrics (generated)
└── README.md                  # Project documentation
```

## Model

- **Algorithm:** Random Forest Regressor
- **Parameters:** 100 estimators
- **Train/Test Split:** 80/20

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | ~0.30 |
| R² Score | ~0.54 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python train.py
```

This will:
1. Load the Wine Quality dataset
2. Preprocess and split the data
3. Train the Random Forest model
4. Evaluate and save metrics
5. Generate `model.pkl` and `metrics.json`

## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:

1. **Train Job:**
   - Triggers on push to `main` branch
   - Sets up Python environment
   - Installs dependencies
   - Runs training script
   - Uploads model artifacts

2. **Report Job:**
   - Depends on successful training
   - Downloads model artifacts
   - Displays evaluation metrics

## Dependencies

- scikit-learn
- pandas
- numpy
- joblib

