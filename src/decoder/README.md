# afix-decoder (WASM / WebGPU)

JavaScript/WASM decoder for the `.aFix` format. Provides zero-dependency, 12 KB runtime compatibility for browsers without native `.aFix` support.

## Architecture

- **Header Parser:** Validates magic bytes, reads `ATOM_MAP`.
- **S1 Rasteriser:** SVG-compatible vector rasteriser for the Geometric Skeleton.
- **S2 VAE Decoder:** ONNX-based VAE decoder running on WebGPU compute shaders.
- **S3 Residual Applier:** Additive residual correction (lossless mode).
- **Depth Renderer:** WebGPU parallax effect renderer using the `DPTH` chunk.
- **Provenance Verifier:** C2PA Ed25519 signature verification.

## Integration

```html
<script type="module" src="afix-decoder.js"></script>
<afix-img src="photo.afix"></afix-img>
```

## Status

🚧 **Phase 1 — In Development**

See the root [SPEC.md](../../SPEC.md) for the binary format this decoder must read.
