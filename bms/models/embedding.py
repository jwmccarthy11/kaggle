import torch
import torch.nn as nn


class TextEmbed(nn.Module):
    """Embed text."""

    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)


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