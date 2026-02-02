import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import VisionTransformer


class ChangeFormer(nn.Module):
    """
    ChangeFormer: A Transformer-based model for change detection
    Based on: Bandara, W.G.C., & Patel, V.M. (2022). A Transformer-Based Siamese Network for Change Detection.
    """

    def __init__(self, in_channels=3, embed_dim=256, num_heads=8, num_layers=4, patch_size=16):
        super(ChangeFormer, self).__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))

        # Transformer encoder for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder for change map generation
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # Siamese network structure
        self.encoder_ref = self.encoder
        self.encoder_tar = self.encoder

    def forward(self, img1, img2):
        # Extract features from both images
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        # Compute difference
        diff = torch.abs(feat1 - feat2)

        # Generate change map
        change_map = self.decoder(diff)

        return change_map

    def extract_features(self, x):
        # Patch embedding
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add position embedding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer encoding
        x = self.encoder(x)

        # Reshape to feature map
        Hp = H // 16
        Wp = W // 16
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)

        return x


# Training script for ChangeFormer
def train_changeforner(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, (img1, img2, mask) in enumerate(train_loader):
            img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(img1, img2)

            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img1, img2, mask in val_loader:
                    img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
                    pred = model(img1, img2)
                    val_loss += criterion(pred, mask).item()

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss / len(train_loader):.4f}, '
                  f'Val Loss: {val_loss / len(val_loader):.4f}')

    return model