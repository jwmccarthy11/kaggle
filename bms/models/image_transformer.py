import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEmbed(nn.Module):
    """Embed text."""

    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        x = self.embed(x)
        return x


class ImageEmbed(nn.Module):
    """Embed patches of an image."""

    def __init__(self, img_dim, patch_dim=16, in_channels=1, embed_dim=512):
        super().__init__()
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.n_patches = (img_dim // patch_dim)**2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_dim,
            stride=patch_dim
        )

    def forward(self, x):
        x = self.proj(x)        # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)        # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (n_samples, n_patches, embed_dim)
        return x


class PositionalEncoder(nn.Module):
    """Encode position of embedded inputs."""

    def __init__(self, in_features, p_drop=0.1, max_len=5000):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        pe = torch.zeros(max_len, in_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, in_features, 2).float() * (-math.log(10000.0) / in_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.drop(x)
        return x


