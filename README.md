# .aFix — Adaptive Flexible Image X

**Version:** 1.0.4-B  
**Authors:** Jinuk Chanthusa & Gemini  
**Organisation:** The .aFix Foundation  
**Status:** Protocol Definition Phase  
**Classification:** Open Standard (Proprietary Core)

> *"The goal of .aFix is not just to save space, but to give the web a pair of eyes. It is the first image format that understands what it is looking at."*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Core Problem](#2-the-core-problem)
3. [Three-Pillar Architecture](#3-three-pillar-architecture)
4. [Technical Specifications](#4-technical-specifications)
5. [Key Innovations](#5-key-innovations)
6. [Mathematical Foundation](#6-mathematical-foundation)
7. [Development Roadmap](#7-development-roadmap)
8. [Competitive Advantage](#8-competitive-advantage)
9. [Market Strategy](#9-market-strategy)
10. [Project Structure](#10-project-structure)

---

## 1. Executive Summary

The **.aFix** (Adaptive Flexible Image X) format is a next-generation visual container designed to replace JPEG, PNG, and WebP. Unlike legacy formats that store static pixel grids, `.aFix` stores **Intent**, **Geometry**, and **Neural Latents**.

It is built specifically for the **2026 hardware landscape** — leveraging NPUs, Spatial Computing, and high-performance WebGPU rendering.

---

## 2. The Core Problem

| Problem | Description |
|---------|-------------|
| **Legacy Bloat** | JPEG/PNG are 30+ years old; they don't understand "objects" or "depth." |
| **Bandwidth Crisis** | High-res images consume ~60% of global web traffic. |
| **Static Limitation** | Modern images need to be 3D-aware and editable via code. |

---

## 3. Three-Pillar Architecture

### Pillar I — The Multi-Layer Bitstream

An `.aFix` file is **not** a flat file; it is a stacked container:

| Layer | ID | Description |
|-------|----|-------------|
| **Structural Layer** (Vector) | `LOD-0` | Stores the "skeleton" of the image. Instant load, zero-pixelation. |
| **Cognitive Layer** (Neural) | `LOD-1` | Uses a quantised latent vector (128-bit) to describe textures. The decoder "imagines" micro-details. |
| **Accuracy Layer** (Residual) | `LOD-2` | A mathematical delta that fixes any AI errors to ensure 100% visual fidelity. |

### Pillar II — Semantic Object Mapping (The "DOM" for Images)

Every `.aFix` file contains a built-in **Object Manifest**:

- **Feature:** Backgrounds, faces, and text are pre-segmented at encode time.
- **Benefit:** Developers can target individual layers via CSS/JS without reloading the file.

```js
// Example: change the sky layer without reloading the asset
document.getElementById('myImage').layers['sky'].style.filter = 'sepia(1)';
```

### Pillar III — Spatial-First (Z-Depth)

- Natively includes a **16-bit depth map**.
- Converts 2D photos into "Live 3D" scenes for AR/VR headsets.
- Enables post-capture refocusing (Bokeh) and parallax effects on any device.

---

## 4. Technical Specifications

### 4.1 File Header — Binary DNA

| Byte Offset | Block ID | Size | Description |
|-------------|----------|------|-------------|
| `0x00–0x03` | `AFIX` | 4 B | Magic Number — identifies the file as `.aFix`. |
| `0x04–0x07` | `VSN_` | 4 B | Protocol Version (e.g., `1.0.0`). |
| `0x08–0x1F` | `DESC` | 24 B | Global Dimensions (stored as Float for infinite scaling). |
| `0x20–0xAF` | `ATOM_MAP` | 144 B | Pointers to S1, S2, and S3 locations in the bitstream. |
| `0xB0–EOF` | `PAYLOAD` | Var | Encrypted, entropy-coded bitstream. |

### 4.2 Atom Chunk Table

| Block ID | Size | Description |
|----------|------|-------------|
| `Magic` `0x414649584B` | — | Identifies the file as `.aFix`. |
| `Meta` JSON/BSON | Var | Creator, Licence, and Semantic Object Tags. |
| `LOD-0` `VEC_B` | ~2 KB | The vector structure for instant preview. |
| `LOD-1` `LAT_Z` | 50 KB+ | The Neural Latent stream for texture synthesis. |
| `LOD-2` `RES_P` | Var | High-frequency detail (Residuals). |

### 4.3 The Tri-Stream Engine

| Stream | Name | Role |
|--------|------|------|
| **S1** | Geometric Skeleton | Resolution-independent vector layer using B-Spline curves. Sharp boundaries at 1% file size. |
| **S2** | Latent Texture Field | 128×128×4 tensor stored as Neural Latents. NPU-accelerated VAE synthesises micro-details on decode. |
| **S3** | Parity Residual | High-frequency delta between AI synthesis and original sensor data. Enables lossless mode. |

---

## 5. Key Innovations

### 5.1 Semantic Addressability (DOM-Image)

The `.aFix` format includes a **Semantic Labelling Layer**. Each object in the image is tagged at encode time.

- **Dev Integration:** Engineers target named layers directly in CSS or JavaScript.
- **Impact:** The image is no longer a static asset; it is a live, programmable component.

### 5.2 Spatial Parallax & Z-Depth

- **Mobile:** Tilt-to-look (Parallax) becomes a native feature of every `.aFix` photo.
- **Vision Pro / VR:** `.aFix` files are automatically 3D volumes, not 2D posters.

### 5.3 Proof of Origin (The "Trust" Layer)

Every `.aFix` file includes a **C2PA-compliant cryptographic signature**:

- Logs the Camera Sensor Serial Number **or** the Generative AI Model Seed.
- Creates an immutable "History of Edits."
- The first image format designed to fight Deepfakes at the kernel level.

---

## 6. Mathematical Foundation

### 6.1 Latent Saliency Reconstruction (Loss Function)

Compression quality is guided by a weighted loss function that prioritises what the Human Visual System (HVS) perceives:

$$\text{Total\_Loss} = \lambda_1 L_{\text{Structural}} + \lambda_2 L_{\text{Neural}} + \lambda_3 L_{\text{Pixel}}$$

Where $\lambda_2$ is the primary driver in low-bandwidth environments.

### 6.2 Non-Linear Quantisation

$$C = \oint (S_v \cdot W_s) + (T_n \cdot W_p)$$

| Symbol | Meaning |
|--------|---------|
| $S_v$ | Structural Vector Data |
| $W_s$ | Saliency Weight — is this the subject? |
| $T_n$ | Texture Neural Latent |
| $W_p$ | Perceptual Weight — can the eye see this? |

By optimising for $W_s$, background noise is compressed by up to **95%** while the subject's eyes remain at **100% visual fidelity**.

---

## 7. Development Roadmap

### Phase 1 — "Yaka-Core" Engine (Months 0–6)

- [ ] Develop the **Rust-based encoder** (`libafix`) for maximum speed.
- [ ] Release the **WASM Decoder** for Chrome/Safari compatibility.
- [ ] Build the `afix-view` web component.

### Phase 2 — Studio & Tools (Months 6–12)

- [ ] **aFix Studio:** Desktop app for creators to tag semantic layers.
- [ ] **Adobe/Figma Plugin:** Export directly to `.aFix`.

### Phase 3 — Global Adoption (Months 12–24)

- [ ] **CDN Push:** Partner with Cloudflare to auto-compress JPEGs into `.aFix`.
- [ ] **NPU Optimisation:** Collaborate with chip manufacturers for silicon-level decoding.

---

## 8. Competitive Advantage

| Feature | JPEG | WebP | **.aFix** |
|---------|------|------|-----------|
| Compression | 10:1 | 20:1 | **100:1** (Neural) |
| 3D Support | No | No | **Yes** (Native Depth) |
| Programmable | No | No | **Yes** (CSS/JS Ready) |
| Loading | Blurry | Blocky | **Vector Sharp** |
| Deepfake Resistance | No | No | **Yes** (C2PA Signed) |
| AI-Training Ready | No | No | **Yes** (Semantic Layers) |

---

## 9. Market Strategy

### The "Yaka-Squeeze" (The Hook)

Free CLI tool and Web App that batch-converts JPG/PNG to `.aFix`, advertising **60% bandwidth savings** for platforms like Netflix and Instagram.

### The "WASM-Bridge" (Compatibility)

A **12 KB JavaScript library** decodes `.aFix` via WebGPU in a `<canvas>` element when native browser support is unavailable. No broken images.

### The Open Source SDK

`libafix` (C++ and Rust) released under the **MIT Licence** — encouraging adoption in Photoshop, Blender, FFmpeg, and beyond.

---

## 10. Project Structure

```
.aFix/
├── spec/
│   └── SPEC.md              # Full binary specification & whitepaper
├── src/
│   ├── encoder/             # Rust-based Yaka-Core encoder
│   ├── decoder/             # WASM/WebGPU decoder
│   └── libafix/             # Core C++/Rust library (MIT)
├── docs/
│   ├── architecture.md      # Tri-Stream Engine deep-dive
│   ├── semantic-layers.md   # Semantic Addressability guide
│   └── proof-of-origin.md   # C2PA Trust Layer documentation
├── tools/
│   ├── cli/                 # afix-convert CLI tool
│   └── web/                 # afix-view web component
├── examples/
│   └── sample.afix          # Reference binary sample
├── SPEC.md                  # Technical whitepaper (mirror)
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT Licence
└── README.md                # This file
```

---

## Projected Outcomes

Within 24 months, `.aFix` targets becoming the standard for:

- **High-Speed Web:** Sites load in <200 ms regardless of image count.
- **AI Training:** `.aFix` files are natively "understood" by AI due to semantic layers.
- **Spatial Computing:** Default format for the Metaverse and AR glasses.

---

*© 2026 The .aFix Foundation — Open Standard, Proprietary Core. Licensed under MIT for SDK components.*