# .aFix Tri-Stream Engine — Architecture Deep Dive

This document provides a detailed description of the three data streams that make up every `.aFix` file.

## S1 — Geometric Skeleton

The Geometric Skeleton is the first and fastest layer to load. It is a **resolution-independent vector representation** of the image's structural content.

### Encoding

1. Canny edge detection is applied to the source image at multiple scales.
2. Detected contours are fit to **cubic B-Spline curves** using least-squares minimisation.
3. Control points are quantised to 16-bit fixed-point coordinates.
4. Delta-encoded control point sequences are entropy-compressed using ANS.

### Decoding

- Requires only a basic vector rasteriser (equivalent to an SVG renderer).
- Renders in under 1 ms on any modern device.
- Provides crisp, zero-pixelation outlines at any zoom level.

## S2 — Latent Texture Field

The Latent Texture Field encodes the "style" and "texture" of the image as a compact neural representation.

### Encoding

- A quantised VAE encoder maps each 512×512 pixel tile to a **128×128×4 float16 tensor**.
- Per-channel scale factors and zero-points are stored in the META chunk.
- The latent tensor is zstd-compressed before writing to the `LAT_Z` chunk.

### Decoding

- A localised VAE decoder (ONNX, ~800 KB) synthesises full-resolution texture from the latent tensor.
- Runs on WebGPU compute shaders or hardware NPU for sub-100 ms decode.
- The synthesised texture is composited over the S1 vector skeleton.

## S3 — Parity Residual

The Parity Residual is an **optional** chunk that enables lossless fidelity.

### Encoding

- `Residual = original_pixels − VAE_decode(S2_latents)`
- Computed in YCbCr colour space to align with HVS sensitivity.
- Compressed with HEVC Intra or AV1 Still coding for maximum efficiency.

### Decoding

- Applied as an additive correction over the S2 synthesis.
- When present, the final image is pixel-perfect (PSNR ≥ 60 dB vs. source).

## Progressive Rendering

The three-layer design enables a unique progressive rendering experience:

```
t=0ms   → S1 decoded → Vector skeleton visible (crisp outlines)
t=50ms  → S2 decoded → Full colour, AI texture
t=100ms → S3 decoded → Lossless correction applied
```

This is superior to JPEG's top-down progressive scan and PNG's Adam7 interlacing because **every intermediate state is visually meaningful**.
