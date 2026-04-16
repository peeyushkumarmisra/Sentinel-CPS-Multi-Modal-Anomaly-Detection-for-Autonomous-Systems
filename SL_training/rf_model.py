"""
File: rf_model.py

This script implements a Random Forest Classifier as part of a Supervised Learning pipeline,
handling training, evaluation, and model serialization.
    1. run_rf:
        * Loads and scales the dataset using utility functions.
        * Initializes a RandomForestClassifier with 100 estimators.
        * Measures the execution time for fitting the model to the training data.
        * Generates predictions on the test set and calculates performance metrics
          (Accuracy, Precision, Recall, and Confusion Matrix).
        * Serializes and saves the trained model as 'rf_model.pkl' for future use.
        * Returns the calculated metrics and the total training time.
"""

import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from SL_training.sl_utils import load_csv_and_scale, get_metrics

def run_rf():
    x_train, x_test, y_train, y_test = load_csv_and_scale() # Loading data
    model = RandomForestClassifier(n_estimators=100, random_state=47) # Loading model
    start_time = time.time() # Starting time
    model.fit(x_train, y_train) # Fitting into model
    end_time = time.time() # End Time
    training_time = end_time - start_time # Total Time
    y_pred = model.predict(x_test) # Prediction
    acc, prec, rec, cm = get_metrics(y_test, y_pred) # Getting metric score
    
    # Saving Model
    model_filename = 'SL_training/rf_model.pkl'
    joblib.dump(model, model_filename)
    return acc, prec, rec, cm, training_time

if __name__ == "__main__":
    run_rf()