"""
File: train_usl_model.py

This script acts as the master orchestrator for the Unsupervised Learning (USL) workflow,
sequentially executing the training, feature extraction, and clustering stages.
    1. train_cae_step:
        * Initializes and executes the CAETrainer to learn compressed image representations.
        * Saves the pre-trained weights to be used as a deep feature extractor.
    2. extract_features_step:
        * Triggers the FeaturePipeline to combine CAE and traditional CV features.
        * Performs PCA reduction and encodes ground-truth labels for performance evaluation.
    3. train_gmm_step:
        * Trains a Gaussian Mixture Model on the fused feature matrix.
        * Uses the Hungarian algorithm (via map_clusters_to_truth) to align unsupervised cluster IDs 
            with ground-truth classes for accuracy assessment.
    4. train_dec_step:
        * Orchestrates the Deep Embedding Clustering training process, including BIRCH
        * initialization and self-supervised refinement.
        * Returns aligned predictions and calculates total execution time for the neural clustering phase.
    5. run_usl_pipeline:
        * The high-level entry point that connects all steps in the pipeline, ensuring data flows
            correctly from raw images to evaluated cluster assignments.
"""

import time
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import your classes and utils
from USL_training.cae_model import CAETrainer
from feature_pipeline import FeaturePipeline
from train_gmm import GMMTrainer
from train_dec import DECTrainer
from USL_training.usl_utils import map_clusters_to_truth, scan_images

def train_cae_step(data_dir="data/fantasy_dataset", model_path="models/cae_feature_ex.pt"):
    """Function 1: Trains the Convolutional Autoencoder from scratch."""
    print("\n[USL] Training CAE Model...")
    paths = list(scan_images(data_dir).values())
    trainer = CAETrainer(latent_dim=128, epochs=20, lr=1e-3, batch_size=64)
    trainer.prepare_loaders(paths)
    trainer.train()
    trainer.save(model_path)
    return model_path

def extract_features_step(base_dir="data/fantasy_dataset", n_components=15):
    """Function 2: Runs the Deep + CV feature extraction and PCA reduction."""
    print("\n[USL] Extracting and Fusing Features...")
    pipeline = FeaturePipeline(base_dir=base_dir, n_components=n_components)
    final_df = pipeline.run()
    
    # Process ground truth for downstream evaluation
    le = LabelEncoder()
    true_labels = le.fit_transform(final_df['Asset Class'])
    return true_labels, le.classes_

def train_gmm_step(true_labels, n_clusters=5):
    """Function 3: Trains GMM, returns aligned predictions and execution time."""
    print("\n[USL] Training GMM...")
    X_train = np.load("models/training_features.npy")
    trainer = GMMTrainer(n_components=n_clusters)
    
    start_time = time.time()
    gmm_model = trainer.train()
    trainer.save()
    exec_time = time.time() - start_time
    
    preds_raw = gmm_model.predict(X_train)
    preds_aligned = map_clusters_to_truth(true_labels, preds_raw)
    return preds_aligned, exec_time

def train_dec_step(true_labels, n_clusters=5):
    """Function 4: Trains Hybrid DEC, returns aligned predictions and execution time."""
    print("\n[USL] Training DEC...")
    X_train = np.load("models/training_features.npy")
    trainer = DECTrainer(input_dim=X_train.shape[1], n_clusters=n_clusters)
    
    start_time = time.time()
    y_pseudo = trainer.generate_pseudo_labels(X_train)
    trainer.prepare_loaders(X_train, y_pseudo)
    trainer.train()
    trainer.save()
    exec_time = time.time() - start_time
    
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(torch.tensor(X_train, dtype=torch.float32).to(trainer.device))
        preds_raw = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
        
    preds_aligned = map_clusters_to_truth(true_labels, preds_raw)
    return preds_aligned, exec_time

def run_usl_pipeline():
    """Master function to execute all 4 steps sequentially and evaluate."""
    BASE_DIR = "data/fantasy_dataset"
    
    # 1. Train CAE
    train_cae_step(data_dir=BASE_DIR)
    
    # 2. Extract Features
    true_labels, class_names = extract_features_step(base_dir=BASE_DIR)
    
    return true_labels, class_names