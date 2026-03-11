"""
loss.py — Composite training loss for the .aFix VAE.

Implements the loss formula from the spec:

    L_total = λ1·L_structural + λ2·L_neural + λ3·L_pixel + λ4·L_kl

Where:
    L_structural — edge/gradient perceptual loss (SSIM on Sobel maps)
    L_neural     — perceptual feature loss via a VGG-style feature extractor
    L_pixel      — pixel-level L1 reconstruction loss
    L_kl         — KL divergence regularisation on the latent distribution

Default λ values: λ1=0.5, λ2=0.3, λ3=0.1, λ4=0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Structural loss (edge / gradient SSIM) ────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Computes the L1 loss between the Sobel edge maps of the target and
    reconstruction.  Encourages the VAE to preserve perceptually important
    structural edges (matching the Canny-based S1 layer).
    """

    def __init__(self):
        super().__init__()
        # Fixed Sobel kernels — not learned.
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           requires_grad=False).view(1, 1, 3, 3)
        ky = kx.transpose(-2, -1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to greyscale luma.
        luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        gx = F.conv2d(luma, self.kx, padding=1)
        gy = F.conv2d(luma, self.ky, padding=1)
        return (gx ** 2 + gy ** 2).sqrt()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(recon), self._edges(target))


# ── Neural perceptual loss ────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Feature-matching loss using the first three ReLU feature maps of a
    lightweight VGG-like feature extractor.

    Penalises differences in texture and structure at multiple scales without
    requiring a full VGG-16 (too heavy for the encoder side).
    """

    def __init__(self):
        super().__init__()
        # A lightweight 3-block feature network.
        self.features = nn.Sequential(
            # Block 1 (output stride /1)
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2 (output stride /2)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3 (output stride /4)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Freeze these weights — they act as a fixed feature extractor.
        for p in self.features.parameters():
            p.requires_grad_(False)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_r = self.features(recon)
        feat_t = self.features(target)
        return F.mse_loss(feat_r, feat_t)


# ── KL divergence ─────────────────────────────────────────────────────────────

def kl_divergence(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Analytical KL divergence between q(z|x) = N(mean, exp(log_var)) and p(z) = N(0,I).

    Returns the mean KL across the batch.
    """
    return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())


# ── Saliency-weighted L1 ──────────────────────────────────────────────────────

def saliency_l1(recon: torch.Tensor, target: torch.Tensor,
                saliency: torch.Tensor) -> torch.Tensor:
    """
    Pixel-level L1 loss weighted by per-pixel saliency W_s.

    High-saliency (subject) pixels incur a larger penalty, encouraging the
    VAE to allocate more bits to the foreground.

    Parameters
    ----------
    recon    : B × 3 × H × W  reconstructed image
    target   : B × 3 × H × W  original image
    saliency : B × 1 × H × W  saliency weight in [0, 1] from SaliencyNet
    """
    diff = (recon - target).abs()
    # Boost weight in salient regions (1 + W_s doubles the penalty).
    weight = 1.0 + saliency
    return (diff * weight).mean()


# ── Composite loss ────────────────────────────────────────────────────────────

class AfixVAELoss(nn.Module):
    """
    Composite loss for training the .aFix VAE.

    L_total = λ1·L_structural + λ2·L_neural + λ3·L_pixel + λ4·L_kl

    Parameters
    ----------
    lambda_structural : float   weight for edge/gradient loss  (default 0.5)
    lambda_neural     : float   weight for perceptual loss     (default 0.3)
    lambda_pixel      : float   weight for pixel L1 loss       (default 0.1)
    lambda_kl         : float   weight for KL divergence       (default 0.1)
    use_saliency      : bool    weight pixel loss by W_s        (default True)
    """

    def __init__(self,
                 lambda_structural: float = 0.5,
                 lambda_neural: float     = 0.3,
                 lambda_pixel: float      = 0.1,
                 lambda_kl: float         = 0.1,
                 use_saliency: bool        = True):
        super().__init__()
        self.l_struct    = lambda_structural
        self.l_neural    = lambda_neural
        self.l_pixel     = lambda_pixel
        self.l_kl        = lambda_kl
        self.use_saliency = use_saliency

        self.edge_loss       = SobelEdgeLoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        saliency: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute and return the composite loss components.

        Returns
        -------
        dict with keys: total, structural, neural, pixel, kl
        """
        l_struct = self.edge_loss(recon, target)
        l_neural = self.perceptual_loss(recon, target)

        if self.use_saliency and saliency is not None:
            l_pixel = saliency_l1(recon, target, saliency)
        else:
            l_pixel = F.l1_loss(recon, target)

        l_kl = kl_divergence(mean, log_var)

        total = (self.l_struct * l_struct
                 + self.l_neural * l_neural
                 + self.l_pixel  * l_pixel
                 + self.l_kl     * l_kl)

        return {
            "total":       total,
            "structural":  l_struct,
            "neural":      l_neural,
            "pixel":       l_pixel,
            "kl":          l_kl,
        }


if __name__ == "__main__":
    vae_loss = AfixVAELoss()
    recon   = torch.rand(2, 3, 64, 64)
    target  = torch.rand(2, 3, 64, 64)
    mean    = torch.randn(2, 4, 16, 16)
    log_var = torch.randn(2, 4, 16, 16)
    sal     = torch.rand(2, 1, 64, 64)

    losses = vae_loss(recon, target, mean, log_var, sal)
    for k, v in losses.items():
        print(f"  {k:12s} : {v.item():.4f}")
