"""
File: train_dec.py

This script implements Deep Embedding Clustering (DEC) using a hybrid training strategy.
It bootstraps cluster assignments using BIRCH and refines them through self-supervised
KL-divergence minimization.
  1. ClusteringNN:
    * init: Defines a Multi-Layer Perceptron (MLP) architecture that maps fused input features 
            to cluster space.
    * forward:  Passes input data through linear layers and ReLU activations to generate cluster logits.
  2. DECTrainer:
    * init: Initializes hyperparameters, hardware device selection, and the ClusteringNN model.
    * _get_target_distribution: Computes the auxiliary target distribution (P) by squaring soft 
                                assignments (Q) to sharpen cluster centroids and increase model confidence.
    * generate_pseudo_labels:   Utilizes the BIRCH algorithm to create initial "hard" cluster
                                labels used for the model warm-up phase.
    * prepare_loaders:  Handles data splitting and initializes PyTorch DataLoaders for training
                        and validation.
    * train:    Executes a two-phase training loop:
    * Warm-up (Epochs 0-14):    Supervised learning using Cross-Entropy loss against BIRCH
                                pseudo-labels.
    * Refinement (Epochs 15+):  Self-supervised learning using KL-Divergence to soften boundaries
                                and improve cluster purity.
    * save: Serializes the trained model weights to the specified disk location.
"""



import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import Birch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from USL_training.usl_utils import plot_loss_curve

# ==========================================
# 1. NETWORK ARCHITECTURE
# ==========================================
class ClusteringNN(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) used to refine traditional cluster boundaries.
    Maps the PCA-reduced feature space into probabilistic cluster assignments.
    """
    def __init__(self, input_dim, n_clusters):
        super(ClusteringNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_clusters)
        )

    def forward(self, x):
        return self.encoder(x)

# ==========================================
# 2. DEC TRAINER LOGIC
# ==========================================
class DECTrainer:
    """
    Handles the BIRCH pseudo-label generation and the hybrid training loop 
    (Warm-up -> Self-Supervised Refinement) for Deep Embedding Clustering.
    """
    def __init__(self, input_dim, n_clusters=5, epochs=80, lr=0.0005, batch_size=64):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClusteringNN(input_dim=input_dim, n_clusters=n_clusters).to(self.device)
        
    def _get_target_distribution(self, q):
        """
        Calculates the auxiliary target distribution (p) for the KL-Divergence loss.
        Squares the soft predictions to sharpen them, forcing the network to become more confident.
        """
        weight = q**2 / (q.sum(0) + 1e-10) 
        return (weight.t() / (weight.sum(1) + 1e-10)).t()

    def generate_pseudo_labels(self, X):
        """Uses BIRCH to generate hard boundary labels for the neural network warm-up."""
        print("Running BIRCH to generate pseudo-labels for warm-up phase...")
        birch = Birch(n_clusters=self.n_clusters)
        return birch.fit_predict(X)

    def prepare_loaders(self, X, y_pseudo):
        """Splits the data and creates PyTorch DataLoaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_pseudo, test_size=0.2, random_state=42, stratify=y_pseudo
        )

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True
        )
        
        # Validation tensors are loaded directly to device for speed
        self.X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        
        history = {"train_loss": [], "val_loss": []}
        print(f"Training Hybrid DEC: 15 Epochs CE Warm-Up -> {self.epochs - 15} Epochs KL Self-Supervised")

        for epoch in range(self.epochs):
            # --- TRAINING PHASE ---
            self.model.train()
            epoch_train_loss = 0
            
            for batch_x, batch_y in self.train_loader: 
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                logits = self.model(batch_x)
                q = torch.softmax(logits, dim=1)
                
                # HYBRID SWITCH
                if epoch < 15:
                    # Warm-up: Learn BIRCH's highly accurate hard boundaries
                    loss = ce_loss_fn(logits, batch_y)
                else:
                    # DEC Refinement: Soften the boundaries for probabilistic uncertainty
                    p = self._get_target_distribution(q).detach()
                    loss = F.kl_div((q + 1e-10).log(), p, reduction='batchmean')
                
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # --- VALIDATION PHASE ---
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.X_val_t)
                val_q = torch.softmax(val_logits, dim=1)
                
                if epoch < 15:
                    val_loss = ce_loss_fn(val_logits, self.y_val_t).item()
                else:
                    val_p = self._get_target_distribution(val_q).detach()
                    val_loss = F.kl_div((val_q + 1e-10).log(), val_p, reduction='batchmean').item()
                    
                history["val_loss"].append(val_loss)
                
            # Logging
            if (epoch + 1) % 10 == 0:
                phase = "Warm-Up (CE)" if epoch < 15 else "Self-Supervised (KL)"
                print(f"Epoch [{epoch+1}/{self.epochs}] | {phase} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        return history

    def save(self, save_path="models/dec_model.pt"):
        """Saves the PyTorch weights."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[Artifact Saved] DEC model successfully saved to: {save_path}")

# ==========================================
# 3. EXECUTION LOGIC
# ==========================================
if __name__ == "__main__":
    FEATURES_PATH = "models/training_features.npy"
    MODEL_SAVE_PATH = "models/dec_model.pt"
    PLOT_SAVE_PATH = "graphs/dec_loss_curve.png"
    TARGET_CLUSTERS = 5

    print("\n=== STARTING DEC TRAINING ===")
    
    # 1. Load Data
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Cannot find {FEATURES_PATH}. Run feature_pipeline.py first.")
    X_train = np.load(FEATURES_PATH)
    print(f"Loaded fused features: {X_train.shape}")
    
    # 2. Initialize Trainer
    trainer = DECTrainer(input_dim=X_train.shape[1], n_clusters=TARGET_CLUSTERS)
    
    # 3. Bootstrap Pseudo-Labels via BIRCH
    y_pseudo = trainer.generate_pseudo_labels(X_train)
    trainer.prepare_loaders(X_train, y_pseudo)
    
    # 4. Train Hybrid Model
    history = trainer.train()
    trainer.save(MODEL_SAVE_PATH)
    
    # 5. Plot Loss
    plot_loss_curve(
        history_dict=history, 
        title="BIRCH-Initialized Hybrid DEC Convergence", 
        ylabel="Loss (CrossEntropy -> KL-Divergence)", 
        save_path=PLOT_SAVE_PATH
    )
    print("=== DEC TRAINING COMPLETE ===\n")