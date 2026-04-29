"""
File: train_gmm.py
"""

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMTrainer:
    def __init__(self, n_clusters, covariance_type, seed):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.seed = seed
        self.model = GaussianMixture(
            n_components=self.n_clusters, 
            covariance_type=self.covariance_type, 
            random_state=self.seed, 
            n_init=10 # Run 10 times with different initializations to find the best fit
        )

    def train(self, x_train):
        print(f"Features loaded successfully. Shape: {x_train.shape}")
        print(f"Fitting GMM with {self.n_clusters} components...")
        self.model.fit(x_train) # Training
        # Checking convergence
        if self.model.converged_:
            print(f"GMM converged in {self.model.n_iter_} iterations.")
        else:
            print("GMM did not converge. Consider increasing max_iter.")
        return self.model

    def predict(self, X): # Predicting with conf
        probs = self.model.predict_proba(X)
        conf = np.max(probs, axis=1)
        cluster_assignments = np.argmax(probs, axis=1)
        return cluster_assignments, conf, probs

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"GMM model saved to: {path}")