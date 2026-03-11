# Proof of Origin — The .aFix Trust Layer

## Overview

Every `.aFix` file contains a **C2PA 2.0-compliant cryptographic signature** stored in the `SIGB` chunk. This makes `.aFix` the first image format to fight Deepfakes at the file-format level.

## What Gets Signed

| Field | Description |
|-------|-------------|
| **Claim Generator** | Tool name and version that created/modified the file |
| **Ingredient Hashes** | SHA-256 of every source asset used |
| **Actions Log** | Immutable list of all edits (crop, AI upscale, colour grade, etc.) |
| **Hard Binding** | Manifest hash bound to the PAYLOAD CRC chain |

## Hardware Provenance Sources

### Camera

```json
{
  "generatorType": "camera",
  "sensorSerial": "SNS-XXXX-YYYY",
  "lens": "24-70mm f/2.8",
  "gps": { "lat": 37.7749, "lon": -122.4194 }
}
```

### Generative AI

```json
{
  "generatorType": "generative_ai",
  "model": "StableDiffusion-XL-1.0",
  "modelVersion": "1.0.0",
  "seed": 42,
  "promptHash": "sha256:abc123..."
}
```

## Verifying Provenance in JavaScript

```js
const img = document.getElementById('myImage');
const manifest = await img.getProvenance();

if (manifest.isTampered) {
  console.warn('⚠️  This image has been modified without a valid signature chain.');
} else {
  console.log('✅ Provenance verified');
  console.log('Source:', manifest.generatorType);
  console.log('Edit history:', manifest.editHistory);
}
```

## Cryptographic Details

- **Signature Algorithm:** Ed25519
- **Signed Payload:** SHA-256 of the serialised C2PA claim
- **Key Storage:** Hardware Secure Enclave (camera) / TPM (desktop tools)
- **Trust Anchors:** `.aFix Foundation` root certificate + device manufacturer CAs

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Pixel-level manipulation | Residual (S3) hash bound to signature |
| Re-signed with forged key | Certificate chain verification against Foundation root CA |
| Metadata stripping | Signature covers `PAYLOAD` CRC chain; stripping `SIGB` flags `isTampered = true` |
| Adversarial latents | VAE model signatures verified before decode (see SPEC §9.3) |
