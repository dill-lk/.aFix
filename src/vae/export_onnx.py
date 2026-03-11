"""
export_onnx.py — Export trained .aFix VAE models to ONNX for use in Rust.

Usage
-----
    python export_onnx.py --vae-ckpt checkpoints/vae_best.pt \\
                          --saliency-ckpt checkpoints/saliency_best.pt \\
                          --seg-ckpt checkpoints/seg_best.pt \\
                          --output-dir ./onnx_models

Outputs
-------
    vae_encoder.onnx       — AfixEncoder (RGB tile → latent mean)
    vae_decoder.onnx       — AfixDecoder (latent → RGB tile)
    saliency.onnx          — SaliencyNet (RGB → W_s)
    segmentation.onnx      — SegmentationNet (RGB → class logits)

ONNX input/output names and dynamic axes are set up so the Rust
`ort` (ONNX Runtime) crate can call them with any batch size and
any spatial resolution (the decoder always operates at the latent
resolution that the encoder produces).

The Rust encoder loads `vae_encoder.onnx` at runtime if it finds
it next to the binary (searched in the same directory as the
executable, then `~/.afix/models/`).
"""

import argparse
import os

import torch

from model             import AfixVAE, AfixEncoder, AfixDecoder
from saliency_model    import SaliencyNet
from segmentation_model import SegmentationNet


def export_vae(vae: AfixVAE, out_dir: str, tile_size: int = 512) -> None:
    """Export encoder and decoder as separate ONNX graphs."""
    vae.eval()
    H = W = tile_size
    L = tile_size // 4  # latent spatial size

    # ── Encoder ───────────────────────────────────────────────────────────────
    enc_path = os.path.join(out_dir, "vae_encoder.onnx")
    dummy_img = torch.zeros(1, 3, H, W)
    torch.onnx.export(
        vae.encoder,
        (dummy_img,),
        enc_path,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["mean", "log_var"],
        dynamic_axes={
            "image":   {0: "batch", 2: "height", 3: "width"},
            "mean":    {0: "batch", 2: "lat_h",  3: "lat_w"},
            "log_var": {0: "batch", 2: "lat_h",  3: "lat_w"},
        },
    )
    print(f"  Encoder  → {enc_path}")

    # ── Decoder ───────────────────────────────────────────────────────────────
    dec_path = os.path.join(out_dir, "vae_decoder.onnx")
    dummy_lat = torch.zeros(1, vae.latent_channels, L, L)
    torch.onnx.export(
        vae.decoder,
        (dummy_lat,),
        dec_path,
        export_params=True,
        opset_version=17,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch", 2: "lat_h",  3: "lat_w"},
            "image":  {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"  Decoder  → {dec_path}")


def export_saliency(model: SaliencyNet, out_dir: str) -> None:
    model.eval()
    sal_path = os.path.join(out_dir, "saliency.onnx")
    dummy = torch.zeros(1, 3, 512, 512)
    torch.onnx.export(
        model,
        (dummy,),
        sal_path,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["saliency"],
        dynamic_axes={
            "image":    {0: "batch", 2: "height", 3: "width"},
            "saliency": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"  Saliency → {sal_path}")


def export_segmentation(model: SegmentationNet, out_dir: str) -> None:
    model.eval()
    seg_path = os.path.join(out_dir, "segmentation.onnx")
    dummy = torch.zeros(1, 3, 512, 512)
    torch.onnx.export(
        model,
        (dummy,),
        seg_path,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"  Segment  → {seg_path}")


def load_model(cls, path: str, device: str = "cpu"):
    m = cls()
    state = torch.load(path, map_location=device)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print("Exporting ONNX models…")

    if args.vae_ckpt:
        vae = load_model(AfixVAE, args.vae_ckpt)
        export_vae(vae, args.output_dir, tile_size=args.tile_size)
    else:
        print("  (no --vae-ckpt supplied — exporting with random weights)")
        vae = AfixVAE()
        export_vae(vae, args.output_dir, tile_size=args.tile_size)

    if args.saliency_ckpt:
        sal = load_model(SaliencyNet, args.saliency_ckpt)
        export_saliency(sal, args.output_dir)
    elif args.export_all:
        print("  (no --saliency-ckpt supplied — exporting with random weights)")
        export_saliency(SaliencyNet(), args.output_dir)

    if args.seg_ckpt:
        seg = load_model(SegmentationNet, args.seg_ckpt)
        export_segmentation(seg, args.output_dir)
    elif args.export_all:
        print("  (no --seg-ckpt supplied — exporting with random weights)")
        export_segmentation(SegmentationNet(), args.output_dir)

    print(f"\nDone. Place the *.onnx files next to `afix-convert` or in")
    print(f"~/.afix/models/ so the Rust encoder can find them at runtime.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export .aFix neural models to ONNX")
    p.add_argument("--vae-ckpt",      default=None, help="Trained VAE checkpoint (.pt)")
    p.add_argument("--saliency-ckpt", default=None, help="Trained SaliencyNet checkpoint (.pt)")
    p.add_argument("--seg-ckpt",      default=None, help="Trained SegmentationNet checkpoint (.pt)")
    p.add_argument("--output-dir",    default="./onnx_models")
    p.add_argument("--tile-size",     type=int, default=512)
    p.add_argument("--export-all",    action="store_true",
                   help="Export all models with random weights even if no checkpoint is given")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
