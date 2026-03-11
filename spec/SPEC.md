# THE .aFix PROTOCOL — TECHNICAL WHITE PAPER

**Subject:** High-Fidelity Neural-Vector Image Container  
**Lead Architect:** Jinuk Chanthusa  
**Version:** 1.0.4-B  
**Status:** Protocol Definition Phase  
**Classification:** Open Standard (Proprietary Core)

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Binary Specification](#2-binary-specification)
3. [Tri-Stream Engine Detail](#3-tri-stream-engine-detail)
4. [Semantic Addressability](#4-semantic-addressability)
5. [Spatial Parallax & Z-Depth](#5-spatial-parallax--z-depth)
6. [Proof of Origin](#6-proof-of-origin)
7. [Mathematical Foundation](#7-mathematical-foundation)
8. [Codec Pipeline](#8-codec-pipeline)
9. [Security Considerations](#9-security-considerations)
10. [Conformance & Versioning](#10-conformance--versioning)

---

## 1. Architectural Overview

The `.aFix` format discards the "Pixel Grid" philosophy of the 1990s. It treats an image as a **Dynamic Scene** — a container that synchronises three disparate data streams into a single perceptual experience.

### 1.1 Design Principles

1. **Resolution Independence** — No fixed pixel dimensions; the image scales infinitely from the vector skeleton.
2. **Progressive Disclosure** — Each LOD layer independently decodable; browsers render something meaningful within the first kilobyte.
3. **Semantic Completeness** — Every object in the scene carries a label, enabling programmatic manipulation without external metadata.
4. **Trust by Default** — Cryptographic provenance is baked into every file, not bolted on after the fact.

### 1.2 The Tri-Stream Engine

| Stream | Name | Role |
|--------|------|------|
| **S1** | Geometric Skeleton | Resolution-independent vector layer. Defines shapes, edges, and silhouettes using B-Spline curves. Sharp at any scale. |
| **S2** | Latent Texture Field | A 128×128×4 neural tensor representing texture "style." NPU-accelerated VAE synthesises micro-details on decode. |
| **S3** | Parity Residual | High-frequency delta between S2 synthesis and original sensor data. Required for lossless fidelity mode. |

```
┌─────────────────────────────────────────────────────┐
│                   .aFix Container                   │
├──────────┬──────────────────────────┬───────────────┤
│  HEADER  │       ATOM MAP           │   PAYLOAD     │
│  28 B    │       144 B              │   Variable    │
├──────────┴──────────────────────────┴───────────────┤
│  S1: Geometric Skeleton  (LOD-0 / VEC_B)  ~2 KB     │
│  S2: Latent Texture Field (LOD-1 / LAT_Z) 50 KB+    │
│  S3: Parity Residual      (LOD-2 / RES_P) Variable  │
│  META: Semantic Object Manifest           Variable  │
│  DEPTH: 16-bit Z-Map                      Variable  │
│  SIG: C2PA Cryptographic Signature        Variable  │
└─────────────────────────────────────────────────────┘
```

---

## 2. Binary Specification

### 2.1 File Header

| Byte Offset | Identifier | Size | Function |
|-------------|------------|------|----------|
| `0x00–0x04` | `AFIXK` | 5 B | Magic Number. ASCII `A`, `F`, `I`, `X` + sentinel `K` (hex `41 46 49 58 4B`). |
| `0x05–0x08` | `VSN_` | 4 B | Protocol Version. Packed as `MAJOR.MINOR.PATCH.FLAG` (1 byte each). |
| `0x09–0x20` | `DESC` | 24 B | Global Dimensions. Two IEEE 754 doubles (width, height) + 7 B reserved. |
| `0x21–0xB0` | `ATOM_MAP` | 144 B | 6 × 24-byte pointers: {stream_id, byte_offset, byte_length, checksum}. |
| `0xB1–EOF` | `PAYLOAD` | Var | Encrypted, entropy-coded bitstream (see §2.3). |

### 2.2 Atom Chunk Format

Each atom chunk inside `PAYLOAD` follows this structure:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
┌─────────────────────────────────────────────────────────────────┐
│                    Chunk ID (4 bytes ASCII)                     │
├─────────────────────────────────────────────────────────────────┤
│                    Chunk Length (4 bytes, uint32 LE)            │
├─────────────────────────────────────────────────────────────────┤
│                    Flags (2 bytes)                              │
├─────────────────────────────────────────────────────────────────┤
│                    Reserved (2 bytes)                           │
├─────────────────────────────────────────────────────────────────┤
│                    Data (Chunk Length bytes)                    │
│                    ...                                          │
├─────────────────────────────────────────────────────────────────┤
│                    CRC-32 (4 bytes)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Registered Chunk IDs:**

| ID | Type | Description |
|----|------|-------------|
| `META` | JSON/BSON | Creator, Licence, Semantic Object Tags. |
| `VEC_` | Binary | S1 — B-Spline vector skeleton (LOD-0). |
| `LAT_` | Binary | S2 — Neural Latent tensor (LOD-1). |
| `RES_` | Binary | S3 — Parity Residual (LOD-2, optional). |
| `DPTH` | Binary | 16-bit unsigned depth map. |
| `SIGB` | Binary | C2PA cryptographic signature block. |
| `OBJM` | JSON/BSON | Semantic Object Manifest. |

### 2.3 Entropy Coding & Encryption

- The `PAYLOAD` region uses **ANS (Asymmetric Numeral Systems)** entropy coding.
- Optional AES-256-GCM encryption per chunk (flag bit 0 in chunk flags).
- The file as a whole supports an outer **zstd** compression envelope (magic byte flag in `VSN_`).

### 2.4 Magic Number

```
Hex:   41 46 49 58 4B
ASCII: A  F  I  X  K   (0x414649584B)
```

The trailing `K` byte (`0x4B`) serves as a version epoch sentinel and distinguishes `.aFix` from other `AFIX`-prefixed proprietary formats.

---

## 3. Tri-Stream Engine Detail

### 3.1 S1 — Geometric Skeleton (VEC_B)

- Encoded using **cubic B-Spline** curves for silhouettes and edges.
- Supplemented by a **Signed Distance Field (SDF)** texture for smooth anti-aliasing.
- Typical size: **1–4 KB** for a full-scene image.
- Decoder requirement: basic vector rasteriser (SVG-level capability).

**Encoding Steps:**

1. Run Canny edge detection on the source image.
2. Fit B-Spline curves to detected contours (least-squares fitting, tolerance ε = 0.5 px).
3. Quantise control points to 16-bit fixed-point coordinates.
4. Entropy-code the control point deltas.

### 3.2 S2 — Latent Texture Field (LAT_Z)

- Stores a **128×128×4** tensor (float16) per 512×512 tile.
- Encoded by a **quantised VAE encoder** (trained on diverse image corpora).
- Decoder uses a **localised VAE decoder** (ONNX model, ~800 KB, NPU-accelerated).
- The `LAT_Z` identifier suffix `_Z` indicates zstd-compressed latents.

**Latent Quantisation:**

```
latent_q = round(latent / scale_factor) + zero_point
scale_factor  ∈ [0.001, 0.1]   (per-channel, stored in META chunk)
zero_point    ∈ [-128, 127]    (int8)
```

### 3.3 S3 — Parity Residual (RES_P)

- Optional chunk; omitting it yields the **Perceptual Lossy** profile.
- Stores `original_pixels − VAE_decode(S2_latents)` in YCbCr colour space.
- Residual is further compressed with **HEVC Intra** or **AV1 Still** coding.
- Including S3 enables the **Lossless Professional** profile (PSNR ≥ 60 dB).

---

## 4. Semantic Addressability

### 4.1 Object Manifest Schema

The `OBJM` chunk stores a BSON document conforming to the following schema:

```json
{
  "version": "1.0",
  "objects": [
    {
      "id": "sky",
      "label": "sky",
      "category": "background",
      "mask_rle": "<run-length encoded bitmask>",
      "bbox": [0, 0, 1920, 400],
      "confidence": 0.97
    },
    {
      "id": "face_0",
      "label": "human_face",
      "category": "subject",
      "mask_rle": "<...>",
      "bbox": [760, 200, 400, 500],
      "confidence": 0.99,
      "landmarks": {
        "left_eye":  [860, 310],
        "right_eye": [960, 310],
        "nose":      [910, 380],
        "mouth":     [910, 450]
      }
    }
  ]
}
```

### 4.2 JavaScript API (afix-view Web Component)

```html
<afix-img id="myImage" src="photo.afix"></afix-img>
```

```js
const img = document.getElementById('myImage');

// Style a named semantic layer
img.layers['sky'].style.filter = 'sepia(1)';

// Hide a layer
img.layers['face_0'].style.display = 'none';

// Replace a layer's texture
img.layers['sky'].src = 'new-sky.afix#sky';

// Event: fired when all LOD layers are decoded
img.addEventListener('afix:ready', () => console.log('fully decoded'));
```

### 4.3 CSS Custom Properties

```css
afix-img::layer(sky) {
  filter: blur(4px) brightness(0.8);
}

afix-img::layer(face_0) {
  transform: scale(1.05);
}
```

---

## 5. Spatial Parallax & Z-Depth

### 5.1 Depth Map Format

- Stored in the `DPTH` chunk as a **16-bit unsigned integer** array.
- Dimensions: same as the logical image width × height.
- Value range: `0` = closest to camera, `65535` = furthest.
- Compressed with lossless **PNG-style DEFLATE** filtering before entropy coding.

### 5.2 Parallax Rendering

The `afix-view` decoder exposes the depth map to the WebGPU pipeline, enabling:

- **Tilt-Parallax** on mobile (DeviceMotion API → vertex shader displacement).
- **Stereoscopic 3D** for VR headsets (left/right eye views generated from single `.aFix` file).
- **Synthetic Bokeh** — depth-based circle-of-confusion rendering in post.

### 5.3 AR/VR Integration

On Apple Vision Pro and Meta Quest, the `.aFix` decoder registers as a native **spatial media type**, allowing the OS to treat `.aFix` files as volumetric objects without any additional application code.

---

## 6. Proof of Origin

### 6.1 C2PA Compliance

Every `.aFix` file contains a `SIGB` chunk that is a fully-conformant **C2PA 2.0 Manifest**:

- **Claim Generator:** Records the tool that created or modified the file.
- **Ingredient Hashes:** SHA-256 hashes of every source asset used.
- **Actions Log:** Immutable list of all edits (crop, colour grade, AI upscale, etc.).
- **Hard Binding:** The manifest hash is bound to the `PAYLOAD` CRC chain.

### 6.2 Hardware Provenance

| Source | Provenance Data Logged |
|--------|------------------------|
| Camera | Sensor serial number, lens EXIF, GPS (optional) |
| Generative AI | Model name, version, random seed, prompt hash |
| Scan | Scanner serial, ICC profile |

### 6.3 Trust Verification

```js
const manifest = await img.getProvenance();
console.log(manifest.isTampered);      // false
console.log(manifest.generatorType);   // "camera" | "generative_ai" | "scan"
console.log(manifest.editHistory);     // [{action, tool, timestamp}, ...]
```

---

## 7. Mathematical Foundation

### 7.1 Weighted Loss Function

The encoder is trained with a composite loss that mirrors the Human Visual System (HVS):

$$L_{\text{total}} = \lambda_1 L_{\text{structural}} + \lambda_2 L_{\text{neural}} + \lambda_3 L_{\text{pixel}}$$

| Term | Formula | Weight (default) |
|------|---------|-----------------|
| $L_{\text{structural}}$ | Edge-aware gradient loss | $\lambda_1 = 0.1$ |
| $L_{\text{neural}}$ | Perceptual VGG-16 feature loss | $\lambda_2 = 0.7$ |
| $L_{\text{pixel}}$ | MSE in YCbCr space | $\lambda_3 = 0.2$ |

In **bandwidth-priority mode** ($\lambda_2 \to 1.0$), S3 is dropped entirely and $\lambda_1$ absorbs structural correctness.

### 7.2 Non-Linear Quantisation

$$C = \oint (S_v \cdot W_s) + (T_n \cdot W_p)$$

| Symbol | Definition |
|--------|-----------|
| $S_v$ | Structural Vector data (S1 bitstream) |
| $W_s$ | Saliency weight map — foreground subject emphasis |
| $T_n$ | Neural Latent tensor (S2) |
| $W_p$ | Perceptual weight — HVS visibility mask |

The saliency map $W_s$ is generated by a lightweight **MobileNet-SSD** model running at encode time. Background regions with $W_s < 0.05$ receive maximum quantisation (up to 95% bit reduction), while subject pixels with $W_s > 0.8$ are encoded near-losslessly.

### 7.3 B-Spline Curve Fitting

For an ordered set of edge points $\{p_i\}$, the encoder fits a degree-3 (cubic) uniform B-Spline $C(t)$:

$$C(t) = \sum_{i=0}^{n} N_{i,3}(t) \cdot P_i$$

where $N_{i,3}(t)$ are the cubic B-Spline basis functions and $\{P_i\}$ are the computed control points. Fitting tolerance is $\varepsilon \leq 0.5$ pixels RMS.

---

## 8. Codec Pipeline

### 8.1 Encoder Pipeline

```
Source Image (JPEG/PNG/RAW)
        │
        ▼
┌─────────────────┐
│  Pre-processing  │  Colour space → YCbCr, normalisation
└────────┬────────┘
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────┐                      ┌──────────────────┐
│  Edge Detection  │                      │  Saliency Model  │
│  + B-Spline Fit  │                      │  (MobileNet-SSD) │
│  → S1 / VEC_B   │                      └────────┬─────────┘
└────────┬────────┘                               │ W_s map
         │                                        ▼
         │                             ┌──────────────────┐
         │                             │  VAE Encoder     │
         │                             │  → S2 / LAT_Z    │
         │                             └────────┬─────────┘
         │                                      │
         │                             ┌────────▼─────────┐
         │                             │  Residual Calc   │
         │                             │  original - S2   │
         │                             │  → S3 / RES_P    │
         │                             └────────┬─────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────────────────────────────────────────────┐
│        Atom Packer + Entropy Coder (ANS + zstd)          │
│        + C2PA Signing + Depth Map (DPTH)                 │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
                   .aFix output file
```

### 8.2 Decoder Pipeline (WASM/WebGPU)

```
.aFix file
    │
    ▼
Header Parse + Atom Map
    │
    ├── Immediate → S1 (VEC_B) → Rasterise vector skeleton (CPU, <1 ms)
    │
    ├── Async     → S2 (LAT_Z) → VAE Decode via WebGPU/NPU (~50 ms)
    │                         → Composite over S1
    │
    └── Optional  → S3 (RES_P) → Apply residual correction
                              → Final lossless composite
```

### 8.3 Progressive Rendering States

| State | Data Loaded | Visual Quality |
|-------|------------|---------------|
| `loading` | Header only | Dimensions known, blank canvas |
| `skeleton` | S1 (VEC_B) | Sharp vector outline, no texture |
| `textured` | S2 (LAT_Z) | Full colour, AI-synthesised texture |
| `lossless` | S3 (RES_P) | Pixel-perfect fidelity |

---

## 9. Security Considerations

### 9.1 Parsing

- All chunk length fields must be validated against the total file size before allocation.
- The decoder must reject files where `ATOM_MAP` pointers overlap or exceed `PAYLOAD` bounds.
- Maximum chunk size: 2 GB (uint32 limit). Decoders should enforce a configurable cap (default: 512 MB).

### 9.2 Cryptographic Provenance

- The `SIGB` chunk uses **Ed25519** signatures over the SHA-256 hash of the serialised C2PA claim.
- Private keys for camera/AI provenance are held in hardware secure enclaves (TPM/Secure Enclave).
- Signature verification is mandatory for "Trusted" display mode; optional for "Standard" mode.

### 9.3 Neural Decode Safety

- VAE decoder models are distributed as signed ONNX files.
- Model signatures are verified against the `.aFix Foundation` root certificate before execution.
- Decoders must sandbox model execution to prevent adversarial latent attacks.

---

## 10. Conformance & Versioning

### 10.1 Profiles

| Profile | S1 | S2 | S3 | DPTH | SIGB | Target Use |
|---------|----|----|----|----|------|------------|
| **Web Lossy** | ✓ | ✓ | ✗ | Optional | Optional | Consumer web |
| **Web Lossless** | ✓ | ✓ | ✓ | Optional | Optional | Design/print |
| **Spatial** | ✓ | ✓ | Optional | ✓ | Optional | AR/VR |
| **Trusted** | ✓ | ✓ | Optional | Optional | ✓ | Journalism/legal |
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | Professional |

### 10.2 Version Negotiation

The `VSN_` field encodes `MAJOR.MINOR.PATCH.FLAG`:

- **MAJOR** bump: breaking binary format change (new decoder required).
- **MINOR** bump: new optional chunk types (backwards-compatible).
- **PATCH** bump: codec algorithm improvements (same format, better quality).
- **FLAG** byte: reserved for extension flags (compression envelope, encryption).

### 10.3 MIME Type & File Extension

| Property | Value |
|----------|-------|
| File Extension | `.afix` |
| MIME Type | `image/afix` |
| UTI (Apple) | `org.afix-foundation.afix` |
| Magic Bytes | `41 46 49 58 4B` |

---

*© 2026 The .aFix Foundation. This specification is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. SDK implementations (`libafix`) are MIT-licensed.*
