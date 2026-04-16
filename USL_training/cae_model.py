"""
File: cae_model.py

This script defines and trains a Convolutional Autoencoder (CAE) to compress images into a 
latent space for feature extraction.
  1. CAEmodel:
    * init: Initializes the encoder (4 convolutional layers + 1 linear) and the decoder 
            (1 linear layer + 4 transpose convolutional layers).
    * get_features: Performs a partial forward pass through the encoder only to return the 
                    compressed latent representation.
    * forward: Executes a full forward pass (Encode -> Decode) to produce a reconstructed image.
  2. _ImageDataset:
    * init: Stores image paths and target dimensions.
    * len: Returns the total number of images in the dataset.
    * getitem: Loads an image from disk, converts BGR to RGB, resizes it, scales pixels to 
                [0, 1], and returns a PyTorch tensor.
  3. CAETrainer:
    * init: Sets up training hyperparameters and initializes the CAEmodel on the available device (CPU/GPU).
    * prepare_loaders: Performs an 80/20 train/validation split and creates PyTorch DataLoaders.
    * train: Orchestrates the training process over multiple epochs, managing the MSE loss, 
                Adam optimizer, and learning rate scheduler.
    * _run_epoch: Internal helper that processes a single pass (training or validation) through the dataset.
    * save: Exports the model's state dictionary (weights) to a specified file path.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from USL_training.usl_utils import scan_images, plot_loss_curve

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class CAEmodel(nn.Module):
    """
    Convolutional Autoencoder (CAE) designed to compress images into a lower-dimensional 
    latent space and reconstruct them, forcing the model to learn the most important features.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # ENCODER: Compresses the image (80x80x3) down to a flat vector of size `latent_dim`.
        # Stride=2 acts as a downsampling mechanism, halving the spatial dimensions each layer.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), 
            nn.ReLU(True),
            nn.Flatten(),
            # The 5x5 comes from halving 80x80 four times (80 -> 40 -> 20 -> 10 -> 5)
            nn.Linear(128 * 5 * 5, latent_dim)
        )
        
        # DECODER (Linear): Expands the latent vector back into a tensor shape for convolutions.
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * 5 * 5), 
            nn.ReLU(True)
        )
        
        # DECODER (ConvTranspose): Upsamples the compressed tensor back to the original image size (80x80x3).
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), 
            # Sigmoid ensures output pixel values are strictly between 0 and 1, matching our input scaling.
            nn.Sigmoid()
        )

    def get_features(self, x):
        """
        Utility method used during inference (in feature_pipeline.py).
        It stops at the encoder, returning the compressed representation.
        """
        with torch.no_grad():
            return self.encoder(x)

    def forward(self, x):
        """Standard forward pass used during training: Encode -> Decode -> Reconstruct."""
        latent = self.encoder(x)
        # Reshape the flat vector back into a 4D tensor (Batch, Channels, Height, Width)
        unflattened = self.decoder_linear(latent).view(-1, 128, 5, 5)
        return self.decoder_conv(unflattened)


# ==========================================
# 2. DATASET HANDLING
# ==========================================
class _ImageDataset(Dataset):
    """Private dataset class to handle image loading, resizing, and tensor conversion."""
    def __init__(self, paths, img_size):
        self.paths = paths
        self.img_size = img_size

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            raise ValueError(f"Cannot read: {self.paths[idx]}")
            
        # Convert BGR (OpenCV default) to RGB, then resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Transpose from (H, W, Channels) to PyTorch's expected (Channels, H, W)
        # Scale pixel values from [0, 255] to [0.0, 1.0] for stable neural network training
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.tensor(img)


# ==========================================
# 3. TRAINING ORCHESTRATOR
# ==========================================
class CAETrainer:
    """Encapsulates the model, device, data loaders, and the training loop logic."""
    def __init__(self, latent_dim=128, epochs=50, lr=1e-4, batch_size=64, img_size=80):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Automatically select GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CAEmodel(latent_dim).to(self.device)

    def prepare_loaders(self, image_paths):
        """Creates an 80/20 train/validation split and initializes DataLoaders."""
        ds = _ImageDataset(image_paths, self.img_size)
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        print(f"Dataset Split → {n_train} training images / {n_val} validation images")

    def train(self):
        """Executes the training loop across all epochs and manages learning rate."""
        # Mean Squared Error compares the reconstructed image pixels to the original image pixels
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Reduces learning rate if validation loss stops improving, preventing overshooting
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            # Run training phase
            tl = self._run_epoch(self.train_loader, criterion, optimizer, is_train=True)
            # Run validation phase
            vl = self._run_epoch(self.val_loader, criterion, is_train=False)
            
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            
            scheduler.step(vl)
            print(f"Epoch [{epoch}/{self.epochs}] | Train Loss: {tl:.6f} | Val Loss: {vl:.6f}")

        return history

    def _run_epoch(self, loader, criterion, optimizer=None, is_train=True):
        """Handles the forward/backward passes for a single epoch."""
        self.model.train(is_train)
        total_loss = 0.0
        
        # Enable gradients only during training to save memory during validation
        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            # Iterating directly over the loader returns the image tensors natively
            for imgs in loader:
                imgs = imgs.to(self.device)
                
                if is_train:
                    optimizer.zero_grad()
                    
                # Forward pass: reconstruct images and calculate loss
                reconstructed = self.model(imgs)
                loss = criterion(reconstructed, imgs)
                
                # Backward pass: compute gradients and update weights
                if is_train:
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item() * imgs.size(0)
                
        return total_loss / len(loader.dataset)

    def save(self, path):
        """Saves the trained model weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model successfully saved → {path}")


# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/fantasy_dataset"
    MODEL_PATH = "models/cae_feature_ex.pt"
    PLOT_PATH = "graphs/cae_loss_curve.png"

    # Scan for images using the utility function
    paths_dict = scan_images(DATA_DIR)
    paths = list(paths_dict.values())
    assert paths, f"No images found in the specified directory: {DATA_DIR}"

    # Initialize and run the trainer
    trainer = CAETrainer(latent_dim=128, epochs=20, lr=1e-3, batch_size=64)
    trainer.prepare_loaders(paths)
    
    print("\nStarting CAE Training...")
    training_history = trainer.train()
    
    trainer.save(MODEL_PATH)
    plot_loss_curve(training_history, "CAE Reconstruction Loss", "MSE", PLOT_PATH)