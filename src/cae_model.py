# CAE ARCHITECTURE
import torch
import torch.nn as nn

class CAEmodel(nn.Module):
    def __init__(self, latent_dim=128):
        super(CAEmodel, self).__init__()      
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, latent_dim)
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * 5 * 5),
            nn.ReLU(True)
        )    
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def get_features(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def forward(self, x):
        latent = self.encoder(x) 
        x_unflatten = self.decoder_linear(latent).view(-1, 128, 5, 5) 
        reconstructed = self.decoder_conv(x_unflatten) 
        return reconstructed