"""
saliency_model.py — Lightweight saliency estimation network for .aFix W_s generation.

Architecture (MobileNet-inspired depthwise separable convolutions)
=============

Input  : B × 3 × H × W  (RGB image, any resolution)
Output : B × 1 × H × W  (per-pixel saliency weight in [0, 1])

  Stem     : Conv2d(3→16, 3×3, stride=2)  + BN + ReLU6
  Block 1  : DWSConv(16→32,  stride=2)  → [H/4,  W/4]
  Block 2  : DWSConv(32→64,  stride=2)  → [H/8,  W/8]
  Block 3  : DWSConv(64→128, stride=2)  → [H/16, W/16]
  Block 4  : DWSConv(128→256,stride=2)  → [H/32, W/32]
  ASPP     : multi-scale context aggregation (dilated convolutions)
  Decoder  : bilinear × 4 → Conv(256→64) → bilinear × 8 → Conv(64→1) + Sigmoid

Total parameters: ~0.7 M (designed to run on CPU in < 50 ms for 512×512 inputs).

Training target
===============
Ground-truth saliency maps from the SALICON or MIT1003 datasets, or derived from
semantic segmentation masks (subject pixels → 1.0, background → 0.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class DWSConv(nn.Module):
    """Depthwise separable convolution: depthwise → pointwise."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branches = nn.ModuleList([
            # 1×1 convolution (rate=1)
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            # Dilated 3×3 at rate 6
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            # Dilated 3×3 at rate 12
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            # Global average pooling branch
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
        ])
        # Fuse 4 branches.
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        outs = []
        for branch in self.branches:
            y = branch(x)
            # Global avg pool branch: upsample back to feature map size.
            if y.shape[2] == 1:
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))


# ── Saliency network ──────────────────────────────────────────────────────────

class SaliencyNet(nn.Module):
    """
    Lightweight saliency estimation network.

    Produces a W_s per-pixel saliency weight map in [0, 1] for use in
    the non-linear quantisation formula:
        C = ∮ (S_v · W_s) + (T_n · W_p)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder backbone (mobile-style) ──────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.block1 = DWSConv(16,  32,  stride=2)   # /4
        self.block2 = DWSConv(32,  64,  stride=2)   # /8
        self.block3 = DWSConv(64,  128, stride=2)   # /16
        self.block4 = DWSConv(128, 256, stride=2)   # /32

        # ── Multi-scale context ───────────────────────────────────────────────
        self.aspp = ASPP(256, 128)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Upsample /32 → /4 and fuse with block1 skip connection.
        self.up_conv = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # /4 → /1
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor   B × 3 × H × W, values in [0, 1]

        Returns
        -------
        sal : torch.Tensor   B × 1 × H × W, saliency in [0, 1]
        """
        h_in, w_in = x.shape[2], x.shape[3]

        # Encode.
        s = self.stem(x)       # /2
        b1 = self.block1(s)    # /4  — kept as skip connection
        b2 = self.block2(b1)   # /8
        b3 = self.block3(b2)   # /16
        b4 = self.block4(b3)   # /32

        # Multi-scale context at /32.
        ctx = self.aspp(b4)

        # Upsample to /4 and fuse with skip.
        ctx_up = F.interpolate(ctx, size=b1.shape[2:], mode="bilinear", align_corners=False)
        fused = self.up_conv(torch.cat([ctx_up, b1], dim=1))

        # Upsample to input resolution.
        out = F.interpolate(fused, size=(h_in, w_in), mode="bilinear", align_corners=False)
        return self.head(out)


# ── Convenience ───────────────────────────────────────────────────────────────

def load_saliency_model(checkpoint_path: str, device: str = "cpu") -> SaliencyNet:
    """Load a trained SaliencyNet from a checkpoint file."""
    model = SaliencyNet()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    model = SaliencyNet()
    x = torch.randn(1, 3, 512, 512)
    sal = model(x)
    print(f"Input      : {x.shape}")
    print(f"Saliency   : {sal.shape}  min={sal.min():.3f} max={sal.max():.3f}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters : {params:.2f} M")
