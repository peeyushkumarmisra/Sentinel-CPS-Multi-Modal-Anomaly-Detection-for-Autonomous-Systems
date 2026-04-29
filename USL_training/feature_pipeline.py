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

from USL_training.cae_model import CAEmodel
from USL_training.cv_features import CVFeatureExtractor
from USL_training.usl_utils import scan_images, ReproducibilityManager



class FeaturePipeline:
    def __init__(self, img_dir, n_components, seed, model_path):
        self.base_dir       = img_dir
        self.model_path     = model_path 
        self.n_components   = n_components
        self.seed           = seed
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_extractor   = CVFeatureExtractor()
        self.pipeline_state = {'reducers': {}, 'final_scaler': StandardScaler()}
        ReproducibilityManager.reproducible(self.seed)

    def scan(self): # Scan and maps image name to asset_class (ground truth)
        path_map = scan_images(self.base_dir)
        records = []
        for f in path_map:
            name_without_ext = f.rsplit('.', 1)[0]
            asset_class = name_without_ext.rsplit('_', 1)[0]
            entry = {
            'image_name': f,
            'asset_class': asset_class
            }
            records.append(entry)
        df = pd.DataFrame(records)
        return df, path_map

    def cae_features(self, df, path_map):
        processed_images = []
        for name in df['image_name']:
            path = path_map.get(name)
            img = cv2.imread(path) if path else None
            if img is None:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0 # Converting to PyTorch format
            processed_images.append(img_tensor)
        images = np.stack(processed_images)
        loader = DataLoader(TensorDataset(torch.tensor(images)), batch_size=256, shuffle=False)
        # Loading the model for evaluation (no grad)
        model = CAEmodel(latent_dim=128).to(self.device)
        cae_model_path = os.path.join(self.model_path, "cae_feature_ex.pt")
        model.load_state_dict(torch.load(cae_model_path, map_location=self.device, weights_only=True))
        model.eval()
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                # Extracting from the encoder half of the CAE
                features = model.get_features(batch.to(self.device))
                latents.append(features.cpu().numpy())
        return np.vstack(latents)

    def reducer(self, feature_name, features): # Applying PCA
        # Scaling (to reduce dominance of larger one)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        # Ensuring we don't ask for more components than we have features
        actual_components = min(self.n_components, scaled.shape[1])
        pca = PCA(n_components=actual_components)
        reduced = pca.fit_transform(scaled)
        # Storing so inference can use .transform() instead of .fit_transform()
        self.pipeline_state['reducers'][feature_name] = {'scaler': scaler, 'pca': pca}
        return reduced

    def run_feature_pipeline(self): 
        df, path_map = self.scan()
        # Extracting Features
        cae_feat = self.cae_features(df, path_map)
        cv_feat = self.cv_extractor.extract_batch([path_map.get(n) for n in df['image_name']])
        print(f"CAE features    : {cae_feat.shape}")
        print(f"CV feature sets : {list(cv_feat.keys())}")
        # Applying Dimensionality Reduction (PCA)
        reduced_blocks = [
            self.reducer('cae', cae_feat),
            self.reducer('cfd', cv_feat['cfd']),
            self.reducer('hsv', cv_feat['hsv']),
            self.reducer('lbp', cv_feat['lbp'])
        ]
        # Numpy Feature (for training)
        fused = np.hstack(reduced_blocks) # Stacking of all into one massive feature matrix
        fused = self.pipeline_state['final_scaler'].fit_transform(fused)
        numpy_features_path = os.path.join(self.model_path, "training_features.npy")
        np.save(numpy_features_path, fused) # Saving
        print(f"Numpy Features Saved: {numpy_features_path}")

        # Pickle file for feature (for inferance)
        feature_model_path = os.path.join(self.model_path, "extracted_pca_features.pkl")
        joblib.dump(self.pipeline_state, feature_model_path)
        print(f"Feature Model Saved: {feature_model_path}")
        df["fused_features"] = list(fused) # Makinng Dataframe
        print(f" Final tensor : {fused.shape}") 
        return df