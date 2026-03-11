# .aFix Neural Network Models (`src/vae/`)

This directory contains the PyTorch implementations of the three neural
networks that power the `.aFix` codec:

| File | Purpose |
|---|---|
| `model.py` | **AfixVAE** — Variational Autoencoder (S2 Latent Texture Field) |
| `saliency_model.py` | **SaliencyNet** — Per-pixel saliency map W_s |
| `segmentation_model.py` | **SegmentationNet** — Semantic segmentation for OBJM |
| `loss.py` | Composite VAE loss (spec formula) |
| `train.py` | Training script |
| `export_onnx.py` | Export to ONNX for Rust runtime |
| `test_models.py` | Architecture shape tests (no GPU required) |

---

## Architecture overview

```
RGB tile (512×512×3)
        │
        ▼
  ┌─────────────┐
  │  AfixVAE    │   AfixEncoder → mean/log_var (128×128×4)
  │  Encoder    │   AfixDecoder ← latent z     (128×128×4)
  └─────────────┘
        │                          stored as LAT_ chunk (sub-format 0x01)
        ▼
  ┌─────────────┐
  │ SaliencyNet │   per-pixel W_s ∈ [0,1]    → non-linear quantisation
  └─────────────┘   (MobileNet-style, ~0.7 M params)
        │
        ▼
  ┌──────────────────┐
  │ SegmentationNet  │   per-pixel class logits → OBJM manifest
  └──────────────────┘   (FCN/DeepLab-style, ~1.2 M params)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training the VAE

```bash
python train.py \
    --data /path/to/image/dataset \
    --val  /path/to/val/images \
    --epochs 100 \
    --batch-size 8 \
    --device cuda
```

Optional: supply a pre-trained `SaliencyNet` checkpoint to enable
saliency-weighted pixel loss (better foreground sharpness):

```bash
python train.py \
    --data /path/to/images \
    --saliency-ckpt checkpoints/saliency_best.pt \
    --epochs 200
```

Checkpoints are saved to `./checkpoints/`.

---

## Exporting to ONNX

After training, export all models to ONNX for use by the Rust encoder:

```bash
python export_onnx.py \
    --vae-ckpt       checkpoints/vae_best.pt \
    --saliency-ckpt  checkpoints/saliency_best.pt \
    --seg-ckpt       checkpoints/seg_best.pt \
    --output-dir     ./onnx_models
```

Place the resulting `*.onnx` files next to the `afix-convert` binary
**or** in `~/.afix/models/`.  The Rust encoder will detect them
automatically and switch from DCT-based S2 to VAE-based S2.

---

## Running architecture tests

```bash
python -m pytest test_models.py -v
```

These tests run without GPU or trained weights.

---

## Loss function

The composite loss matches the spec's quantisation formula:

```
L_total = λ1·L_structural + λ2·L_neural + λ3·L_pixel + λ4·L_kl
```

| Term | Description | Default λ |
|---|---|---|
| `L_structural` | Sobel edge SSIM — preserves S1 skeleton accuracy | 0.5 |
| `L_neural` | Perceptual feature matching (VGG-style) | 0.3 |
| `L_pixel` | Saliency-weighted L1 pixel loss (W_s) | 0.1 |
| `L_kl` | KL divergence — regularises latent distribution | 0.1 |

---

## Integration with the Rust encoder

The Rust encoder (`src/encoder`) currently uses DCT-based S2 compression
(sub-format byte `0x02`).  When ONNX model files are present at runtime,
it will switch to VAE-based S2 (sub-format byte `0x01`) for higher quality.

The sub-format byte in the `LAT_` chunk header tells the decoder which
codec was used:
- `0x01` — VAE latents (float16 tensor, 128×128×4)
- `0x02` — DCT tile compression (current default)
