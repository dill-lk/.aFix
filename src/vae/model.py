"""
model.py — Variational Autoencoder for .aFix S2 Latent Texture Field.

Architecture
============

Encoder
-------
Input  : B × 3 × H × W  (RGB image tiles, H = W = 512 by default)
Output : mean, log_var   each B × 4 × (H/4) × (W/4)   (latent spatial res = 128²)

  Conv2d(3  → 64,  3×3, stride=1, pad=1) → ReLU → BN
  Conv2d(64 → 128, 3×3, stride=2, pad=1) → ReLU → BN   [256×256]
  Conv2d(128→ 256, 3×3, stride=2, pad=1) → ReLU → BN   [128×128]
  Conv2d(256→ 256, 3×3, stride=1, pad=1) → ReLU → BN
  Conv2d(256→ 8,   1×1)                               (4 mean + 4 log_var channels)

Decoder
-------
Input  : B × 4 × (H/4) × (W/4)
Output : B × 3 × H × W

  Conv2d(4 → 256, 1×1)
  ConvTranspose2d(256→ 256, 3×3, stride=1, pad=1) → ReLU → BN
  ConvTranspose2d(256→ 128, 4×4, stride=2, pad=1) → ReLU → BN   [256×256]
  ConvTranspose2d(128→  64, 4×4, stride=2, pad=1) → ReLU → BN   [512×512]
  Conv2d(64 → 3, 3×3, pad=1) → Sigmoid

Latent format
-------------
The encoder produces float16 latent tensors stored as the `LAT_` chunk:
  sub_format = 0x01  (VAE, as opposed to 0x02 for DCT)
  width, height, channels = 128, 128, 4 (f16 values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block used inside the encoder bottleneck."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(self.conv1(x)), inplace=True)


# ── Encoder ───────────────────────────────────────────────────────────────────

class AfixEncoder(nn.Module):
    """
    Encodes a 512×512 RGB tile into a 128×128×4 latent (mean, log_var).

    Parameters
    ----------
    latent_channels : int
        Number of latent channels (default 4, per spec).
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.latent_channels = latent_channels

        self.down = nn.Sequential(
            ConvBnRelu(3,   64,  3, stride=1, padding=1),   # 512×512
            ConvBnRelu(64,  128, 3, stride=2, padding=1),   # 256×256
            ConvBnRelu(128, 256, 3, stride=2, padding=1),   # 128×128
            ConvBnRelu(256, 256, 3, stride=1, padding=1),   # 128×128
            ResBlock(256),
            ResBlock(256),
        )
        # Project to mean + log_var simultaneously.
        self.proj = nn.Conv2d(256, latent_channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor   B × 3 × H × W, values in [0, 1]

        Returns
        -------
        mean    : B × latent_channels × (H/4) × (W/4)
        log_var : B × latent_channels × (H/4) × (W/4)
        """
        h = self.down(x)
        out = self.proj(h)
        mean, log_var = out.chunk(2, dim=1)
        return mean, log_var

    @staticmethod
    def reparameterise(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mean + ε·exp(0.5·log_var)."""
        if not mean.requires_grad:
            return mean  # deterministic at inference time
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


# ── Decoder ───────────────────────────────────────────────────────────────────

class AfixDecoder(nn.Module):
    """
    Decodes a 128×128×4 latent tensor back to a 512×512 RGB tile.

    Parameters
    ----------
    latent_channels : int
        Must match the encoder (default 4).
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=1),
            ConvBnRelu(256, 256, 3, stride=1, padding=1),   # 128×128
            ResBlock(256),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 256×256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 512×512
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor   B × latent_channels × (H/4) × (W/4)

        Returns
        -------
        recon : torch.Tensor   B × 3 × H × W, values in [0, 1]
        """
        return self.up(z)


# ── Full VAE ──────────────────────────────────────────────────────────────────

class AfixVAE(nn.Module):
    """
    Full Variational Autoencoder for .aFix S2 compression.

    Encoder : RGB tile → (mean, log_var) latent
    Decoder : latent → reconstructed RGB tile
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = AfixEncoder(latent_channels)
        self.decoder = AfixDecoder(latent_channels)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        recon   : reconstructed image B × 3 × H × W
        mean    : latent mean
        log_var : latent log-variance
        """
        mean, log_var = self.encoder(x)
        z = AfixEncoder.reparameterise(mean, log_var)
        recon = self.decoder(z)
        return recon, mean, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling — returns mean latent (for inference)."""
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor to an RGB image."""
        return self.decoder(z)

    @property
    def latent_channels(self) -> int:
        return self.encoder.latent_channels


# ── Convenience ───────────────────────────────────────────────────────────────

def load_vae(checkpoint_path: str, device: str = "cpu") -> AfixVAE:
    """Load a trained VAE from a checkpoint file."""
    model = AfixVAE()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    # Smoke test.
    model = AfixVAE()
    x = torch.randn(1, 3, 512, 512)
    recon, mean, log_var = model(x)
    print(f"Input  : {x.shape}")
    print(f"Latent : {mean.shape}  (mean)  {log_var.shape}  (log_var)")
    print(f"Recon  : {recon.shape}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params : {params:.2f} M")
