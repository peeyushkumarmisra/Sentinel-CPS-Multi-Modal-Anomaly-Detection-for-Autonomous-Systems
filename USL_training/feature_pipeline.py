"""
File: feature_pipeline.py

This script manages the end-to-end feature engineering workflow, combining deep learning
latent vectors with traditional CV descriptors and applying dimensionality reduction.
  1. FeaturePipeline:
    * init: Initializes the pipeline with paths, PCA configurations, and detects hardware 
            acceleration (GPU/CPU).
    * _scan:    Identifies all images in the directory and extracts asset class labels based on 
                file naming conventions.
    * _load_image:  A robust loader that handles image reading, color conversion, and resizing
                    with a fallback for corrupted files.
    * _cae_features:    Batch-processes images through a pre-trained CAE encoder to extract 
                        high-level spatial features.
    * _reduce:  Fits a StandardScaler and PCA to a specific feature set, storing the 
                transformation parameters for future inference.
    * run: The main execution logic that coordinates scanning, extraction, PCA reduction,
            horizontal feature fusion, and final standardization. It also saves the
            'pca_bundle.pkl'artifact.
"""

import os
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from USL_training.cae_model import CAEmodel
from USL_training.usl_utils import scan_images
from USL_training.cv_features import CVFeatureExtractor



class FeaturePipeline:
    """
    Master pipeline for Unsupervised Learning feature extraction.
    Loads images, extracts deep (CAE) and traditional (CV) features, reduces 
    their dimensionality via PCA, and fuses them into a single standardized vector.
    
    Crucially, it saves the PCA and Scaler states so the inference pipeline 
    can process new, unseen data using the exact same mathematical boundaries.
    """

    def __init__(self, base_dir, cae_model_path="cae_feature_ex.pt",
                 n_components=15, img_size=80, artifact_save_path="models/pca_bundle.pkl"):
        
        self.base_dir       = base_dir
        self.cae_model_path = cae_model_path
        self.n_components   = n_components
        self.img_size       = img_size
        self.artifact_path  = artifact_save_path
        
        # Auto-detect GPU for faster deep learning extraction
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_extractor   = CVFeatureExtractor()
        
        # State dictionary to hold our fitted sklearn objects for future inference
        self.pipeline_state = {'reducers': {}, 'final_scaler': StandardScaler()}

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _scan(self):
        """Scans the directory and maps file paths to their respective asset classes."""
        path_map = scan_images(self.base_dir)
        records = [
            # Assumes naming convention like: "pump_001.png" -> extracts "pump"
            {'image_name': f, 'Asset Class': f.rsplit('.', 1)[0].rsplit('_', 1)[0]}
            for f in path_map
        ]
        return pd.DataFrame(records), path_map

    def _load_image(self, path):
        """Safely loads, converts, and resizes a single image."""
        img = cv2.imread(path) if path else None
        
        # Fallback for corrupt data to prevent pipeline crashes
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.img_size, self.img_size))

    def _cae_features(self, df, path_map):
        """Passes all images through the pre-trained Convolutional Autoencoder."""
        # 1. Preprocess images into PyTorch-friendly formats (C, H, W) scaled [0, 1]
        images = np.stack([
            np.transpose(self._load_image(path_map.get(n)), (2, 0, 1)).astype(np.float32) / 255.0
            for n in df['image_name']
        ])
        
        # 2. Batch the data to prevent Out-Of-Memory (OOM) errors on the GPU
        loader = DataLoader(TensorDataset(torch.tensor(images)), batch_size=256, shuffle=False)
        
        # 3. Load the model strictly for evaluation (no gradient tracking)
        model = CAEmodel(latent_dim=128).to(self.device)
        model.load_state_dict(torch.load(self.cae_model_path, map_location=self.device, weights_only=True))
        model.eval()
        
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                # Extract the 128-dim latent vector from the encoder half of the CAE
                latents.append(model.get_features(batch.to(self.device)).cpu().numpy())
                
        return np.vstack(latents)

    def _reduce(self, feature_name, features):
        """
        Standardizes and applies PCA to a single feature set.
        Saves the fitted Scaler and PCA objects to the pipeline state.
        """
        # Always scale before PCA, otherwise variables with larger ranges dominate the components
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        
        # Ensure we don't ask for more components than we have features
        actual_components = min(self.n_components, scaled.shape[1])
        pca = PCA(n_components=actual_components)
        reduced = pca.fit_transform(scaled)
        
        # Store these exact fitted instances so inference can use .transform() instead of .fit_transform()
        self.pipeline_state['reducers'][feature_name] = {'scaler': scaler, 'pca': pca}
        
        return reduced

    # ── Public Interface ───────────────────────────────────────────────────────

    def run(self):
        """Executes the pipeline, saves artifacts, and returns the final DataFrame."""
        df, path_map = self._scan()
        print(f"\nFEATURE PIPELINE | {len(df)} images in '{self.base_dir}'")

        # Step 1: Extract Deep Features
        cae_feat = self._cae_features(df, path_map)
        print(f"  CAE features    : {cae_feat.shape}")

        # Step 2: Extract Traditional Computer Vision Features
        cv_feat = self.cv_extractor.extract_batch([path_map.get(n) for n in df['image_name']])
        print(f"  CV feature sets : {list(cv_feat.keys())}")

        # Step 3: Dimensionality Reduction (PCA) for each individual feature block
        reduced_blocks = [
            self._reduce('cae', cae_feat),
            self._reduce('cfd', cv_feat['cfd']),
            self._reduce('hum', cv_feat['hum']),
            self._reduce('hsv', cv_feat['hsv']),
            self._reduce('lbp', cv_feat['lbp'])
        ]

        # Step 4: Horizontal stacking of all 5 reduced arrays into one massive feature matrix
        fused = np.hstack(reduced_blocks)
        
        # Step 5: Final standardization over the combined matrix to ensure downstream ML models behave well
        fused = self.pipeline_state['final_scaler'].fit_transform(fused)
        np.save(self.features_save_path, fused)
        print(f"  [Artifact Saved]: {self.features_save_path}")

        # Step 6: Save the mathematical parameters (Scalers/PCA) for the inference scripts
        os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
        joblib.dump(self.pipeline_state, self.artifact_path)
        print(f"  [Artifact Saved]: {self.artifact_path} (CRITICAL FOR INFERENCE)")

        # Step 7: Attach to the dataframe for easy tracking/saving
        df["fused_features"] = list(fused)
        print(f"  Final tensor    : {fused.shape}")
        
        return df

if __name__ == "__main__":
    # Test execution
    pipeline = FeaturePipeline(base_dir="data/fantasy_dataset")
    final_df = pipeline.run()