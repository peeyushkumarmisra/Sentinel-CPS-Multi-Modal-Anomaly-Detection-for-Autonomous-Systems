# train_usl_model.py


import os
import time
import random
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import your classes and utils
from USL_training.cae_model import CAETrainer
from USL_training.feature_pipeline import FeaturePipeline
from USL_training.train_gmm import GMMTrainer
from USL_training.train_dec import DECTrainer
from USL_training.usl_utils import map_clusters_to_truth, scan_images, evaluate_and_plot_usl

def set_global_determinism(seed=47):
    # Lock Python, NumPy, and PyTorch seeds for 100% reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class USLTraining:
    MODELS = {
        "GMM": GMMTrainer,
        "DEC": DECTrainer
    }

    def __init__(self, data_dir="data/fantasy_dataset", n_clusters=5, seed=47):
        self.data_dir = data_dir
        self.n_clusters = n_clusters
        self.seed = seed
        os.makedirs("models", exist_ok=True)

    def train_cae(self, model_path="USL_training/cae_feature_ex.pt"): 
        print("\n[USL] Training CAE Model...")
        paths = list(scan_images(self.data_dir).values())
        trainer = CAETrainer(latent_dim=128, epochs=20, lr=1e-3, batch_size=64)
        trainer.prepare_loaders(paths)
        trainer.train()
        trainer.save(model_path)
        return model_path

    def extract_features(self, n_components=15): 
        print("\n[USL] Extracting and Fusing Features...")
        pipeline = FeaturePipeline(base_dir=self.data_dir, n_components=n_components)
        final_df = pipeline.run()
        # Process ground truth for downstream evaluation
        le = LabelEncoder()
        true_labels = le.fit_transform(final_df['Asset Class'])
        return true_labels, le.classes_

    def run_model(self, name, true_labels, X_train):
        ModelClass = self.MODELS[name]
        print(f"\n[USL] Training {name}...")
        
        start_time = time.process_time()
        
        if name == "GMM":
            trainer = ModelClass(n_components=self.n_clusters)
            gmm_model = trainer.train()
            trainer.save()
            
            exec_time = time.process_time() - start_time
            preds_raw = gmm_model.predict(X_train)
            
        else: # DEC
            trainer = ModelClass(input_dim=X_train.shape[1], n_clusters=self.n_clusters)
            y_pseudo = trainer.generate_pseudo_labels(X_train)
            trainer.prepare_loaders(X_train, y_pseudo)
            trainer.train()
            trainer.save()
            
            exec_time = time.process_time() - start_time
            
            trainer.model.eval()
            with torch.no_grad():
                logits = trainer.model(torch.tensor(X_train, dtype=torch.float32).to(trainer.device))
                preds_raw = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
                
        preds_aligned = map_clusters_to_truth(true_labels, preds_raw)
        print(f"✅ {name} Training finished in {exec_time:.5f}s")
        
        return preds_aligned, exec_time

    def train_usl_models(self):
        """Master function to execute all 4 steps sequentially and evaluate."""
        # 1. Train CAE
        self.train_cae()
        
        # 2. Extract Features
        true_labels, class_names = self.extract_features()
        
        # Load the newly extracted features 
        X_train = np.load("USL_training/training_features.npy")
        
        # 3. Train GMM
        gmm_pred, gmm_time = self.run_model("GMM", true_labels, X_train)
        
        # 4. Train DEC
        dec_pred, dec_time = self.run_model("DEC", true_labels, X_train)
        
        # 5. Evaluate and Plot automatically
        evaluate_and_plot_usl(true_labels, gmm_pred, dec_pred, class_names, gmm_time, dec_time)
        
        print("\n[USL] Pipeline Complete.")
        return {
            "GMM": {"pred": gmm_pred, "time": gmm_time},
            "DEC": {"pred": dec_pred, "time": dec_time}
        }


if __name__ == "__main__":
    set_global_determinism(47)
    usl_trainer = USLTraining(data_dir="data/fantasy_dataset", n_clusters=5, seed=47)
    usl_trainer.train_usl_models()