# ML Project 1: Iris Flower Classifier

A simple machine learning project that trains a Logistic Regression model
to classify iris flowers into 3 species from 4 measurements.

## What it does
- Loads the Iris dataset (150 flowers, 4 features)
- Splits the data: 80% training, 20% testing
- Trains a Logistic Regression model
- Evaluates accuracy on unseen test data
- Saves the trained model with joblib

## Result
Test accuracy: 1.0

## How to run
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

## Tech
Python, scikit-learn, joblib