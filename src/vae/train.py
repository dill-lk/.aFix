"""
train.py — Training script for the .aFix VAE (+ optional saliency model).

Usage
-----
    python train.py --data /path/to/images --epochs 100 --batch-size 8

The script expects `--data` to point to a directory containing JPEG/PNG images.
Images are randomly cropped to 512×512 tiles for training.

Checkpoints are saved to ./checkpoints/ every 5 epochs and at the end of
training.  The best checkpoint (lowest total validation loss) is saved as
`./checkpoints/vae_best.pt`.

After training, run `export_onnx.py` to export the encoder and decoder to ONNX
for use by the Rust encoder/decoder.
"""

import argparse
import os
import math
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import AfixVAE
from saliency_model import SaliencyNet
from loss import AfixVAELoss


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageTileDataset(Dataset):
    """
    Loads images from a directory and randomly crops 512×512 tiles.
    Images smaller than 512×512 are resized up first.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    TILE_SIZE = 512

    def __init__(self, root: str, augment: bool = True):
        self.paths = [
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(f"No images found in {root!r}")

        t = [transforms.ToTensor()]
        if augment:
            t = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                       saturation=0.1, hue=0.02),
            ] + t
        self.transform = transforms.Compose(t)
        self.crop = transforms.RandomCrop(self.TILE_SIZE)
        self.resize = transforms.Resize(self.TILE_SIZE)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        # Ensure minimum size.
        if min(img.size) < self.TILE_SIZE:
            img = self.resize(img)
        img = self.crop(img)
        return self.transform(img)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = ImageTileDataset(args.data,      augment=True)
    val_ds   = ImageTileDataset(args.val or args.data, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    print(f"Train: {len(train_ds)} images   Val: {len(val_ds)} images")

    # ── Models ────────────────────────────────────────────────────────────────
    vae = AfixVAE().to(device)

    # Optional frozen saliency model for W_s.
    saliency_model = None
    if args.saliency_ckpt and os.path.exists(args.saliency_ckpt):
        from saliency_model import load_saliency_model
        saliency_model = load_saliency_model(args.saliency_ckpt, str(device))
        print(f"Loaded saliency model from {args.saliency_ckpt}")

    # ── Loss + optimiser ──────────────────────────────────────────────────────
    criterion = AfixVAELoss(
        lambda_structural=args.lambda_structural,
        lambda_neural=args.lambda_neural,
        lambda_pixel=args.lambda_pixel,
        lambda_kl=args.lambda_kl,
        use_saliency=(saliency_model is not None),
    ).to(device)

    optimizer = optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume from checkpoint if requested.
    start_epoch = 0
    best_val_loss = math.inf
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        vae.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_val_loss", math.inf)
        print(f"Resumed from epoch {start_epoch}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        vae.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} train",
                          leave=False):
            batch = batch.to(device)

            # Saliency W_s (optional).
            saliency = None
            if saliency_model is not None:
                with torch.no_grad():
                    saliency = saliency_model(batch)

            recon, mean, log_var = vae(batch)
            losses = criterion(recon, batch, mean, log_var, saliency)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["total"].item()

        train_loss /= len(train_dl)

        # ── Validate ──────────────────────────────────────────────────────────
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} val",
                              leave=False):
                batch = batch.to(device)
                saliency = None
                if saliency_model is not None:
                    saliency = saliency_model(batch)
                recon, mean, log_var = vae(batch)
                losses = criterion(recon, batch, mean, log_var, saliency)
                val_loss += losses["total"].item()
        val_loss /= len(val_dl)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:4d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={elapsed:.1f}s")

        # ── Checkpoints ───────────────────────────────────────────────────────
        state = {
            "epoch":         epoch,
            "model":         vae.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(args.checkpoint_dir, "vae_best.pt"))
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"vae_epoch{epoch+1:04d}.pt")
            torch.save(state, path)

    # Save final checkpoint.
    torch.save(state, os.path.join(args.checkpoint_dir, "vae_final.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {args.checkpoint_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the .aFix VAE")
    p.add_argument("--data",              required=True, help="Training image directory")
    p.add_argument("--val",               default=None,  help="Validation image directory (defaults to --data)")
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--workers",           type=int,   default=4)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--resume",            default=None,  help="Resume from checkpoint path")
    p.add_argument("--checkpoint-dir",    default="./checkpoints")
    p.add_argument("--save-every",        type=int,   default=5)
    p.add_argument("--saliency-ckpt",     default=None,  help="Path to trained SaliencyNet checkpoint")
    p.add_argument("--lambda-structural", type=float, default=0.5)
    p.add_argument("--lambda-neural",     type=float, default=0.3)
    p.add_argument("--lambda-pixel",      type=float, default=0.1)
    p.add_argument("--lambda-kl",         type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
