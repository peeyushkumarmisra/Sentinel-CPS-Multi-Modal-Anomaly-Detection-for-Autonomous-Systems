"""
File: svm_model.py

This script implements a Support Vector Machine (SVM) classifier using a Radial Basis Function (RBF)
kernel to perform supervised classification.
    1. run_svm:
        * Loads and standardizes the input features and target labels via the load_csv_and_scale helper.
        * Initializes an SVC model with an 'rbf' kernel for non-linear decision boundaries.
        * Tracks the time taken to train the model on the training dataset.
        * Performs inference on the test set to generate predictions.
        * Computes classification performance metrics including Accuracy, Precision, Recall,
          and a Confusion Matrix.
        * Saves the trained SVM model to disk as 'svm_model.pkl'.
        * Returns the performance metrics and total training duration.
"""

import time
import joblib
from sklearn.svm import SVC
from SL_training.sl_utils import load_csv_and_scale, get_metrics

def run_svm():
    x_train, x_test, y_train, y_test = load_csv_and_scale() # Loading data
    model = SVC(kernel='rbf', random_state=47) # Loading model
    start_time = time.time() # Starting time
    model.fit(x_train, y_train) # Fitting into model
    end_time = time.time() # End Time
    training_time = end_time - start_time # Total Time
    y_pred = model.predict(x_test) # Prediction
    acc, prec, rec, cm = get_metrics(y_test, y_pred) # Getting metric score
    
    # Saving Model
    model_filename = 'SL_training/svm_model.pkl'
    joblib.dump(model, model_filename)
    return acc, prec, rec, cm, training_time

if __name__ == "__main__":
    run_svm()