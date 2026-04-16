"""
File: inferance_pipeline.py

This script defines a high-performance inference engine used to evaluate images
in real-time, specifically designed for integration with Reinforcement Learning (RL) environments.
  1. ClusteringNN:
    * init:     Defines a simple multi-layer perceptron (MLP) architecture to map fused features 
                into cluster assignments.
    * forward:  Executes the pass through the linear layers and ReLU activations to produce
                cluster logits.
  2. SecurityEvaluator:
    * init:     Loads all necessary pre-trained artifacts (CAE weights, PCA/Scaler bundle, 
                GMM model, and DEC model) into memory for fast access.
    * _preprocess:  Handles raw image loading, color conversion, and formatting into both a
                    Numpy array (for CV) and a PyTorch tensor (for Deep Learning).
    * _extract_and_fuse:    Coordinates the multi-modal feature extraction. It transforms new data
                            using the saved PCA and Scaling parameters (ensuring consistency 
                            with training) and horizontally stacks them.
    * evaluate: The primary public method. It takes an image path, generates the fused feature
                vector, and returns cluster predictions and confidence scores from both the 
                Gaussian Mixture Model (GMM) and Deep Embedded Clustering (DEC).
"""

import os
import cv2
import torch
import joblib
import numpy as np
import torch.nn as nn

# Import your custom feature extractors
from USL_training.cae_model import CAEmodel
from USL_training.cv_features import CVFeatureExtractor

# ==========================================
# 1. DEC ARCHITECTURE (Inference Mode)
# ==========================================
class ClusteringNN(nn.Module):
    """Redefined here so the RL environment doesn't need to import training scripts."""
    def __init__(self, input_dim, n_clusters=5):
        super(ClusteringNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_clusters)
        )
    def forward(self, x):
        return self.encoder(x)

# ==========================================
# 2. THE INFERENCE PIPELINE
# ==========================================
class SecurityEvaluator:
    """
    Ultra-fast inference class for the RL environment.
    Loads all artifacts into memory once, providing split-second evaluations.
    """
    def __init__(self, 
                 cae_path="models/cae_feature_ex.pt",
                 pca_bundle_path="models/pca_bundle.pkl",
                 gmm_path="models/gmm_model.pkl",
                 dec_path="models/dec_model.pt",
                 img_size=80, 
                 n_clusters=5):
        
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_extractor = CVFeatureExtractor()

        print("Initializing Security Evaluator... Loading Artifacts.")

        # 1. Load PCA & Scaler Bundle
        if not os.path.exists(pca_bundle_path):
            raise FileNotFoundError(f"Missing {pca_bundle_path}")
        bundle = joblib.load(pca_bundle_path)
        self.reducers = bundle['reducers']
        self.final_scaler = bundle['final_scaler']
        
        # Calculate the final fused dimension dynamically from the scaler
        fused_dim = self.final_scaler.mean_.shape[0]

        # 2. Load CAE Encoder
        self.cae = CAEmodel(latent_dim=128).to(self.device)
        self.cae.load_state_dict(torch.load(cae_path, map_location=self.device, weights_only=True))
        self.cae.eval() # CRITICAL: Disables dropout/batchnorm for inference

        # 3. Load GMM
        self.gmm = joblib.load(gmm_path)

        # 4. Load DEC Model
        self.dec = ClusteringNN(input_dim=fused_dim, n_clusters=n_clusters).to(self.device)
        self.dec.load_state_dict(torch.load(dec_path, map_location=self.device, weights_only=True))
        self.dec.eval()
        
        print("All models loaded and ready for RL environment queries.")

    # ── Internal Math ────────────────────────────────────────────────────────

    def _preprocess(self, image_path):
        """Loads and formats the image for both CV and PyTorch."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"RL Environment Error: Cannot read {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        
        # Format for PyTorch (1, C, H, W)
        tensor_img = np.transpose(img_resized, (2, 0, 1)).astype(np.float32) / 255.0
        tensor_img = torch.tensor(tensor_img).unsqueeze(0).to(self.device)
        
        return img, tensor_img

    def _extract_and_fuse(self, raw_img, tensor_img):
        """Runs the deep + traditional extraction and applies saved PCA states."""
        # 1. Deep Feature Extraction
        with torch.no_grad():
            cae_feat = self.cae.get_features(tensor_img).cpu().numpy()

        # 2. Traditional CV Feature Extraction
        cv_feats = {
            'cfd': np.array([self.cv_extractor.cfd(raw_img)]),
            'hum': np.array([self.cv_extractor.hum(raw_img)]),
            'hsv': np.array([self.cv_extractor.hsv(raw_img)]),
            'lbp': np.array([self.cv_extractor.lbp(raw_img)])
        }

        # 3. Dimensionality Reduction (using the exact parameters saved during training)
        reduced_blocks = []
        for name, feats in [('cae', cae_feat), ('cfd', cv_feats['cfd']), 
                            ('hum', cv_feats['hum']), ('hsv', cv_feats['hsv']), 
                            ('lbp', cv_feats['lbp'])]:
            
            scaler = self.reducers[name]['scaler']
            pca = self.reducers[name]['pca']
            # Notice we use .transform(), NOT .fit_transform()
            reduced_blocks.append(pca.transform(scaler.transform(feats)))

        # 4. Final Fusion and Scaling
        fused = np.hstack(reduced_blocks)
        return self.final_scaler.transform(fused)

    # ── Public RL Interface ──────────────────────────────────────────────────

    def evaluate(self, image_path):
        """
        The main method called by the RL environment.
        Returns a dictionary containing predictions and confidences from both models.
        """
        raw_img, tensor_img = self._preprocess(image_path)
        features = self._extract_and_fuse(raw_img, tensor_img)

        # GMM Prediction
        gmm_probs = self.gmm.predict_proba(features)[0]
        gmm_cluster = np.argmax(gmm_probs)
        gmm_conf = np.max(gmm_probs)

        # DEC Prediction
        features_t = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.dec(features_t)
            dec_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        dec_cluster = np.argmax(dec_probs)
        dec_conf = np.max(dec_probs)

        return {
            "gmm": {"cluster": int(gmm_cluster), "confidence": float(gmm_conf)},
            "dec": {"cluster": int(dec_cluster), "confidence": float(dec_conf)}
        }

# ==========================================
# 3. TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Test the pipeline on a single image to ensure no crashes
    TEST_IMAGE = "data/fantasy_dataset/test_asset.png" # Replace with a real image path
    
    if os.path.exists(TEST_IMAGE):
        evaluator = SecurityEvaluator()
        result = evaluator.evaluate(TEST_IMAGE)
        
        print("\n--- Industrial Asset Security Audit ---")
        print(f"Target: {TEST_IMAGE}")
        print(f"GMM Pipeline -> Cluster: {result['gmm']['cluster']}, Confidence: {result['gmm']['confidence']:.2%}")
        print(f"DEC Pipeline -> Cluster: {result['dec']['cluster']}, Confidence: {result['dec']['confidence']:.2%}")
    else:
        print("Provide a valid TEST_IMAGE path to run the standalone test.")