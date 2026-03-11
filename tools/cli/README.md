# afix-convert CLI

Command-line tool for converting JPEG, PNG, and WebP files to `.aFix`.

## Usage

```bash
# Basic conversion (Web Lossy profile)
afix-convert input.jpg output.afix

# Lossless conversion
afix-convert --profile web-lossless input.png output.afix

# Batch conversion of a directory
afix-convert --batch ./images/ --output ./afix-images/

# Spatial (AR/VR) profile with depth estimation
afix-convert --profile spatial --depth-model auto input.jpg output.afix

# Full profile with provenance signing
afix-convert --profile full --sign ./my-key.pem input.jpg output.afix

# View file info
afix-info photo.afix
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--profile` | Encoding profile (`web-lossy`, `web-lossless`, `spatial`, `trusted`, `full`) | `web-lossy` |
| `--quality` | Neural latent quality (0–100) | `85` |
| `--semantic-model` | Segmentation model (`auto`, `fast`, `accurate`, `none`) | `auto` |
| `--depth-model` | Depth estimation model (`auto`, `none`) | `none` |
| `--sign` | Path to Ed25519 private key for C2PA signing | — |
| `--batch` | Input directory for batch conversion | — |
| `--output` | Output file or directory | — |

## Status

🚧 **Phase 1 — In Development**
