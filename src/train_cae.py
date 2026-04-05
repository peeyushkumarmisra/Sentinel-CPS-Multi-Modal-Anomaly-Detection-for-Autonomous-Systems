import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from cae_model import CAEmodel
from torch.utils.data import Dataset, DataLoader, random_split



# DATA PREPARATION
class ImageDataset(Dataset):
    def __init__(self, image_paths, img_size=80):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.tensor(img)

def get_image_paths(base_dir):
    paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png')):
                paths.append(os.path.join(root, file))
    return paths



# TRAINING
def train_cae(model, train_loader, val_loader, epochs: int = 50, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )  
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        
        # Training
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            imgs = batch.to(device) 
            optimizer.zero_grad()
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * imgs.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch.to(device)
                reconstructed = model(imgs)
                loss = criterion(reconstructed, imgs)
                running_val_loss += loss.item() * imgs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
    return model, history



# EXECUTING & SAVING 
if __name__ == "__main__":
    DATA_DIRECTORY = "data/fantasy_dataset"  
    MODEL_SAVE_PATH = "models/cae_feature_ex.pt"
    BATCH_SIZE = 64
    EPOCHS = 20 
    
    print("Scanning for images...")
    image_paths = get_image_paths(DATA_DIRECTORY)
    if len(image_paths) == 0:
        print(f"Warning: No images found in {DATA_DIRECTORY}. Exiting")
        exit()
        
    print(f"Found {len(image_paths)} images. Preparing Dataset...")
    dataset = ImageDataset(image_paths, img_size=80)
    
    # Train and Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Split complete: {train_size} training images, {val_size} validation images.") 

    cae_model = CAEmodel(latent_dim=128)
    trained_model, training_history = train_cae(cae_model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3)
    
    # Saving Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[Artifact Saved] Inference model successfully saved to: {MODEL_SAVE_PATH}")
    
    # Plotting
    print("Generating validation loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(training_history["train_loss"], label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(training_history["val_loss"], label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.title('CAE Reconstruction Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = 'graphs/cae_loss_curve.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Loss curve saved as '{plot_path}' - Ready for your report.")