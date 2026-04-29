"""
File: train_dec.py
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import Birch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from USL_training.usl_utils import ReproducibilityManager, seed_worker


# MODEL ARCHITECTURE
class DECmodel(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(DECmodel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_clusters)
        )

    def forward(self, x):
        return self.encoder(x)


# TRAINING
class DECTrainer:  # Warm-up -> Self-Supervised Refinement for Deep Embedding Clustering
    def __init__(self, input_dim=None, n_clusters=None, epochs=None, seed=None, model_path=None, is_train=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_train = is_train
        if self.is_train: # For Training
            self.n_clusters = n_clusters
            self.epochs = epochs
            self.seed = seed
            self.model = DECmodel(input_dim=input_dim, n_clusters=self.n_clusters).to(self.device)
            ReproducibilityManager.reproducible(self.seed)
        else: # For Inference
            self.seed   = None
            self.epochs = None
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            extracted_input_dim = state_dict['encoder.0.weight'].shape[1]
            self.n_clusters = state_dict['encoder.4.weight'].shape[0]
            self.model = DECmodel(input_dim=extracted_input_dim, n_clusters=self.n_clusters).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"DEC loaded from: {model_path}  (input_dim={extracted_input_dim}, n_clusters={self.n_clusters})")
        
    def get_target_distribution(self, q): # Calculates the auxiliary target distribution (p) for the KL-Divergence loss.
        weight = q**2 / (q.sum(0) + 1e-10) 
        return (weight.t() / (weight.sum(1) + 1e-10)).t()

    def generate_pseudo_labels(self, X): # Uses BIRCH to generate boundary labels for warm-up
        print("Running BIRCH to generate pseudo-labels for warm-up phase...")
        birch = Birch(n_clusters=self.n_clusters)
        return birch.fit_predict(X)

    def prepare_loaders(self, x, y_pseudo):
        X_train, X_val, y_train, y_val = train_test_split(
            x, y_pseudo, test_size=0.2, random_state=self.seed, stratify=y_pseudo
        )
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
                          batch_size=64, 
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g
        )
        # For Validation
        self.X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        ce_loss_fn = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": []}
        print(f"Training DEC: 15 Epochs CE Warm-Up -> {self.epochs - 15} Epochs")
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            for batch_x, batch_y in self.train_loader: 
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                q = torch.softmax(logits, dim=1)
                # Switching to Self-Supervised Refinement after 15 epoch
                if epoch < 15:
                    loss = ce_loss_fn(logits, batch_y) # Warming up (Learn BIRCH's pseudo_labels)
                else:
                    # DEC Refinement
                    p = self.get_target_distribution(q).detach()
                    loss = F.kl_div((q + 1e-10).log(), p, reduction='batchmean')
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            history["train_loss"].append(avg_train_loss)
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.X_val_t)
                val_q = torch.softmax(val_logits, dim=1)
                # Switching
                if epoch < 15:
                    val_loss = ce_loss_fn(val_logits, self.y_val_t).item()
                else:
                    val_p = self.get_target_distribution(val_q).detach()
                    val_loss = F.kl_div((val_q + 1e-10).log(), val_p, reduction='batchmean').item()
                history["val_loss"].append(val_loss)
            if (epoch + 1) % 10 == 0:
                phase = "Warm-Up" if epoch < 15 else "Self-Supervised"
                print(f"Epoch [{epoch+1}/{self.epochs}] | {phase} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        return history
    
    def predict(self, x):   
        self.model.eval()
        with torch.no_grad():
            # Processing input and get Softmax probabilities
            x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        # Extracting hard assignments with max
        cluster_assignments = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return cluster_assignments, confidences, probs

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"DEC model saved to: {path}")
