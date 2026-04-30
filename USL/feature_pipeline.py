"""
File: feature_pipeline.py
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

from USL.cae_model import CAEmodel
from USL.cv_features import CVFeatureExtractor
from USL.usl_utils import scan_images, ReproducibilityManager


class FeaturePipeline:
    def __init__(self, img_dir=None, n_components=None, seed=None, model_path=None, is_train=True):
        self.is_train     = is_train
        self.base_dir       = img_dir
        self.model_path     = model_path 
        self.seed           = seed
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_extractor   = CVFeatureExtractor()
        ReproducibilityManager.reproducible(self.seed)
        if self.is_train: # For Training
            self.n_components   = n_components
            self.pipeline_state = {'reducers': {}, 'final_scaler': StandardScaler()}
        else: # For Inferance
            self.pipeline_state = joblib.load(os.path.join(self.model_path, "extracted_pca_features.pkl"))
            print(f"Feature Pipeline loaded from: {self.model_path}")

    def scan(self): # Scan and maps image name to asset_class (ground truth)
        path_map = scan_images(self.base_dir)
        records = []
        for f in path_map:
            name_without_ext = f.rsplit('.', 1)[0]
            asset_class = name_without_ext.rsplit('_', 1)[0]
            entry = {'image_name': f, 'asset_class': asset_class}
            records.append(entry)
        df = pd.DataFrame(records)
        return df, path_map
    
    def load_images(self, img_paths):
        processed = []
        for path in img_paths:
            img = cv2.imread(path) if path else None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            processed.append(tensor)    
        images = np.stack(processed)
        loader = DataLoader(TensorDataset(torch.tensor(images)), batch_size=256, shuffle=False)
        cae = CAEmodel(latent_dim=128).to(self.device)
        cae.load_state_dict(torch.load(os.path.join(self.model_path, "cae_feature_ex.pt"),
                                       map_location=self.device, weights_only=True))
        cae.eval()
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                latents.append(cae.get_features(batch.to(self.device)).cpu().numpy())
        return np.vstack(latents)

    def process_block(self, name, features):
        if not self.is_train:
            r = self.pipeline_state['reducers'][name]
            return r['pca'].transform(r['scaler'].transform(features))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        pca = PCA(n_components=min(self.n_components, scaled.shape[1]))
        reduced = pca.fit_transform(scaled)
        self.pipeline_state['reducers'][name] = {'scaler': scaler, 'pca': pca}
        return reduced

    def run_feature_pipeline(self, image_paths=None): 
        if self.is_train:
            df, path_map = self.scan()
            image_paths = [path_map.get(n) for n in df['image_name']]
        # Extracting Features
        cae_feat = self.load_images(image_paths)
        cv_feat = self.cv_extractor.extract_batch(image_paths)
        # Reducing and Fusing
        reduced_blocks = [
            self.process_block('cae', cae_feat),
            self.process_block('cfd', cv_feat['cfd']),
            self.process_block('hsv', cv_feat['hsv']),
            self.process_block('lbp', cv_feat['lbp'])
        ]
        fused = np.hstack(reduced_blocks) 
        # Final Scaling & Output
        if not self.is_train:
            return self.pipeline_state['final_scaler'].transform(fused)
        fused = self.pipeline_state['final_scaler'].fit_transform(fused)
        np.save(os.path.join(self.model_path, "training_features.npy"), fused)
        joblib.dump(self.pipeline_state, os.path.join(self.model_path, "extracted_pca_features.pkl"))
        df["fused_features"] = list(fused)
        print(f"Training Data Saved! Final tensor shape: {fused.shape}")
        return df