# afix-encoder (Yaka-Core)

Rust-based encoder for the `.aFix` format. Part of Phase 1 ("Yaka-Core Engine") of the development roadmap.

## Architecture

- **Edge Detection:** Canny multi-scale edge detector.
- **B-Spline Fitting:** Least-squares cubic B-Spline curve fitting (S1).
- **VAE Encoder:** Quantised variational autoencoder for neural latents (S2).
- **Residual Encoder:** HEVC Intra / AV1 Still residual encoder (S3).
- **Saliency Model:** MobileNet-SSD saliency weight map generation.
- **Atom Packer:** ANS entropy coder + zstd envelope.

## Status

🚧 **Phase 1 — In Development**

See the root [SPEC.md](../../SPEC.md) for the binary format this encoder must produce.
