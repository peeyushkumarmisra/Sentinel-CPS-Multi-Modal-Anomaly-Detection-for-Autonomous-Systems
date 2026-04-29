# train_usl_model.py


import os
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from USL_training.cae_model import CAETrainer
from USL_training.feature_pipeline import FeaturePipeline
from USL_training.gmm import GMMTrainer
from USL_training.dec import DECTrainer
from USL_training.usl_utils import map_clusters_to_truth, scan_images, plot_loss_curve, evaluate_and_plot_usl, ReproducibilityManager



class USLTraining:
    MODELS = {
        "GMM": GMMTrainer,
        "DEC": DECTrainer
    }

    def __init__(self, seed, data_dir, plot_dir, model_dir):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.seed = seed

    def train_cae(self):
        print("\nStarting CAE Training...")
        paths = list(scan_images(self.data_dir).values())
        assert paths, f"No images found in the specified directory: {self.data_dir}"
        trainer = CAETrainer(latent_dim=128, epochs=20, seed=self.seed)
        trainer.prepare_loaders(paths)
        training_history = trainer.train()
        # Saving Model
        cae_model_dir = os.path.join(self.model_dir, "cae_feature_ex.pt")
        trainer.save(cae_model_dir)
        # Saving Plot
        cae_plot_path = os.path.join(self.plot_dir, "cae_loss_curve.jpeg")
        plot_loss_curve(training_history, "CAE Reconstruction Loss", "MSE", cae_plot_path)

    def extract_features(self): 
        print("\nExtracting Features...")
        pipeline = FeaturePipeline(img_dir=self.data_dir, n_components = 4, 
                                   seed=self.seed, model_path =self.model_dir)
        final_df = pipeline.run_feature_pipeline()
        le = LabelEncoder() # Process ground truth for downstream evaluation
        true_labels = le.fit_transform(final_df['asset_class'])
        return true_labels, le.classes_

    def run_model(self, name, true_labels, x_train):
        ModelClass = self.MODELS[name]
        print(f"\nTraining USL Model - {name}...")
        start_time = time.process_time()
        if name == "GMM":
            trainer = ModelClass(n_clusters=5, covariance_type='spherical', seed=self.seed)
            trainer.train(x_train)
            exec_time = time.process_time() - start_time
            gmm_model_path = os.path.join(self.model_dir, "gmm_model.pkl")
            trainer.save(gmm_model_path)
            preds_raw, confidences, _ = trainer.predict(x_train)   
        else: # DEC
            trainer = ModelClass(input_dim=x_train.shape[1], n_clusters=5, epochs = 80, seed=self.seed)
            y_pseudo = trainer.generate_pseudo_labels(x_train) # Pseudo-Labels via BIRCH
            trainer.prepare_loaders(x_train, y_pseudo)
            history = trainer.train()
            exec_time = time.process_time() - start_time
            dec_model_path = os.path.join(self.model_dir, "dec_model.pt")
            trainer.save(dec_model_path)
            dec_plot_path = os.path.join(self.plot_dir, "dec_loss_curve.jpeg")
            plot_loss_curve(history, "BIRCH-Initialized Hybrid DEC","Loss (CrossEntropy -> KL-Divergence)", dec_plot_path )
            # Inference to get predictions and confidences
            trainer.model.eval()
            preds_raw, confidences, _ = trainer.predict(x_train)
        preds_aligned = map_clusters_to_truth(true_labels, preds_raw)
        print(f"{name} Training finished in {exec_time:.5f}s")
        return preds_aligned, confidences, exec_time

    def train_usl_models(self):
        self.train_cae() # Training CAE
        true_labels, class_names = self.extract_features() # Extracting Features
        numpy_features_path = os.path.join(self.model_dir, "training_features.npy")
        x_train = np.load(numpy_features_path) # Loading Extracted numpy Features
        gmm_pred, gmm_conf, gmm_time = self.run_model("GMM", true_labels, x_train)
        dec_pred, dec_conf, dec_time = self.run_model("DEC", true_labels, x_train)
        plot_path = os.path.join(self.plot_dir, "usl_model_comparison.jpeg")
        evaluate_and_plot_usl(true_labels = true_labels, class_names = class_names, plot_path = plot_path,
                              gmm_preds = gmm_pred, dec_preds = dec_pred,
                              gmm_time = gmm_time, dec_time = dec_time,
                              gmm_conf = gmm_conf, dec_conf = dec_conf
                              )
        print("\n USL Training is Complete.")
        return {
            "GMM": {"pred": gmm_pred, "time": gmm_time, "conf": gmm_conf},
            "DEC": {"pred": dec_pred, "time": dec_time, "conf": dec_conf}
        }