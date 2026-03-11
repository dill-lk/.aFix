"""
segmentation_model.py — Semantic segmentation network for .aFix OBJM chunk generation.

Architecture (FCN / DeepLab-style with a MobileNet encoder)
============

Input  : B × 3 × H × W  (RGB image, any resolution)
Output : B × num_classes × H × W  (per-pixel class logits, pre-softmax)

Default classes (matches the OBJM manifest categories):
  0 — background
  1 — subject  (people, animals, primary objects)
  2 — overlay  (UI elements, text, watermarks)
  3 — sky
  4 — ground

The output `category` field in the OBJM chunk is derived from the argmax of
the per-pixel softmax probabilities.

Training target
===============
Any semantic segmentation dataset with compatible labels (e.g. COCO-Stuff,
ADE20K with label remapping, or a custom dataset of .aFix-relevant categories).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from saliency_model import DWSConv, ASPP


# ── Segmentation head ─────────────────────────────────────────────────────────

class SegHead(nn.Module):
    """Lightweight segmentation head: 1×1 conv → bilinear upsample."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, out_size: tuple) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ── Segmentation network ──────────────────────────────────────────────────────

#: Default OBJM category names, ordered by class index.
OBJM_CATEGORIES: List[str] = ["background", "subject", "overlay", "sky", "ground"]


class SegmentationNet(nn.Module):
    """
    Semantic segmentation network for .aFix OBJM chunk generation.

    Shares the same MobileNet-style encoder as :class:`SaliencyNet` so both
    can run on a single shared backbone to save computation.

    Parameters
    ----------
    num_classes : int
        Number of semantic categories (default 5 — see OBJM_CATEGORIES).
    """

    def __init__(self, num_classes: int = len(OBJM_CATEGORIES)):
        super().__init__()
        self.num_classes = num_classes
        self.categories  = OBJM_CATEGORIES[:num_classes]

        # ── Shared encoder backbone ──────────────────────────────────────────
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
        )
        self.block1 = DWSConv(16,  32,  stride=2)   # /4
        self.block2 = DWSConv(32,  64,  stride=2)   # /8
        self.block3 = DWSConv(64,  128, stride=2)   # /16
        self.block4 = DWSConv(128, 256, stride=2)   # /32

        # ── Multi-scale context ───────────────────────────────────────────────
        self.aspp = ASPP(256, 128)

        # ── Decoder with skip connections ────────────────────────────────────
        # /32 → /8 (skip from block2)
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # /8 → /4 (skip from block1)
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = SegHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor   B × 3 × H × W, values in [0, 1]

        Returns
        -------
        logits : torch.Tensor   B × num_classes × H × W (pre-softmax)
        """
        h_in, w_in = x.shape[2], x.shape[3]

        s  = self.stem(x)
        b1 = self.block1(s)    # /4
        b2 = self.block2(b1)   # /8
        b3 = self.block3(b2)   # /16
        b4 = self.block4(b3)   # /32

        ctx = self.aspp(b4)

        # Upsample /32 → /8 + skip from b2.
        up1 = F.interpolate(ctx, size=b2.shape[2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, b2], dim=1))

        # Upsample /8 → /4 + skip from b1.
        up2 = F.interpolate(up1, size=b1.shape[2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat([up2, b1], dim=1))

        return self.head(up2, (h_in, w_in))

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return B × H × W integer class map (argmax of logits)."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def predict_objects(self, x: torch.Tensor, min_area_frac: float = 0.005) -> list:
        """
        Run segmentation and return a list of detected semantic objects in the
        OBJM manifest format.

        Parameters
        ----------
        x             : B=1 image tensor
        min_area_frac : minimum fraction of total pixels for a region to be
                        included in the manifest

        Returns
        -------
        list of dicts compatible with ObjectManifest.objects
        """
        import numpy as np

        logits = self.forward(x)             # 1 × C × H × W
        probs  = logits.softmax(dim=1)       # 1 × C × H × W
        mask   = logits.argmax(dim=1)[0]     # H × W

        h, w = mask.shape
        total = h * w
        min_area = int(total * min_area_frac)

        objects = []
        for cls_idx, cls_name in enumerate(self.categories):
            region = (mask == cls_idx)
            if region.sum() < min_area:
                continue

            ys, xs = region.nonzero(as_tuple=True)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            conf = float(probs[0, cls_idx][region].mean())

            category = "subject" if cls_idx == 1 else \
                       "overlay" if cls_idx == 2 else "background"

            objects.append({
                "id":         f"{cls_name}_{cls_idx}",
                "label":      cls_name,
                "category":   category,
                "bbox":       [x0, y0, x1 - x0, y1 - y0],
                "confidence": round(conf, 4),
            })

        return objects


# ── Convenience ───────────────────────────────────────────────────────────────

def load_segmentation_model(checkpoint_path: str, device: str = "cpu") -> SegmentationNet:
    """Load a trained SegmentationNet from a checkpoint file."""
    model = SegmentationNet()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    model = SegmentationNet()
    x = torch.randn(1, 3, 512, 512)
    logits = model(x)
    print(f"Input   : {x.shape}")
    print(f"Logits  : {logits.shape}  (classes: {model.categories})")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params  : {params:.2f} M")
