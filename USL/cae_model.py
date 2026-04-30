"""
File: cae_model.py
"""


import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from USL.usl_utils import ReproducibilityManager, seed_worker


# MODEL ARCHITECTURE
class CAEmodel(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # ENCODER
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
        
        # DECODER (Linear)
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * 5 * 5), 
            nn.ReLU(True)
        )
        
        # DECODER (Upsamplling)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), 
            nn.Sigmoid() # Ensuring output pixels are between 0 and 1
        )

    def get_features(self, x): # It stops at the encoder and returns the compressed thing
        with torch.no_grad():
            return self.encoder(x)

    def forward(self, x):
        latent = self.encoder(x)
        # Reshape the vector back into a 4D tensor (Batch, Channels, Height, Width)
        unflattened = self.decoder_linear(latent).view(-1, 128, 5, 5)
        return self.decoder_conv(unflattened)



# DATASET HANDLING
class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            raise ValueError(f"Cannot read: {self.paths[idx]}")
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Transpose from (H, W, Channels) to PyTorch's expected (Channels, H, W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0 # Scale pixel values [0, 255] -> [0.0, 1.0]
        return torch.tensor(img)



# TRAINING 
class CAETrainer:
    def __init__(self, latent_dim, epochs, seed):
        self.epochs = epochs
        self.seed = seed
        ReproducibilityManager.reproducible(self.seed) # Fixing Seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CAEmodel(latent_dim).to(self.device)

    def prepare_loaders(self, img_paths):
        ds = ImageDataset(img_paths)
        batch_size=64
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
        self.train_loader = DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       worker_init_fn=seed_worker,
                                       generator=g
                                       )
        self.val_loader = DataLoader(val_ds, 
                                     batch_size=batch_size, 
                                     shuffle=False, 
                                     num_workers=4,
                                     worker_init_fn=seed_worker,
                                     generator=g
                                     )
        print(f"Dataset Split → {n_train} training images / {n_val} validation images")

    def train(self):
        # Using Mean Squared Error to evaluate the reconstructed image
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        # For Plot
        history = {"train_loss": [], "val_loss": []}
        # Main Loop
        for epoch in range(1, self.epochs + 1):
            # Run training phase
            tl = self.run_epoch(self.train_loader, criterion, optimizer, is_train=True)
            # Run validation phase
            vl = self.run_epoch(self.val_loader, criterion, is_train=False)
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            scheduler.step(vl)
            print(f"Epoch [{epoch}/{self.epochs}] | Train Loss: {tl:.6f} | Val Loss: {vl:.6f}")
        return history

    def run_epoch(self, loader, criterion, optimizer=None, is_train=True):
        """Handles the forward/backward passes for a single epoch."""
        self.model.train(is_train)
        total_loss = 0.0
        # Enable gradients only during training
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
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
                # Loss
                total_loss += loss.item() * imgs.size(0)
        return total_loss / len(loader.dataset)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
