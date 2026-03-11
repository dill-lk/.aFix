# afix-view Web Component

The `<afix-img>` custom element — a drop-in replacement for `<img>` that natively renders `.aFix` files in any browser via WASM + WebGPU.

## Installation

```bash
npm install afix-view
```

Or via CDN (12 KB gzipped):

```html
<script type="module" src="https://cdn.afix.io/afix-view@1.0.0/afix-view.min.js"></script>
```

## Usage

```html
<afix-img src="photo.afix" alt="A beautiful photo"></afix-img>

<!-- With responsive sizes -->
<afix-img src="photo.afix" width="800" height="600" loading="lazy"></afix-img>

<!-- Fallback for very old browsers (no WebGPU) -->
<afix-img src="photo.afix">
  <img slot="fallback" src="photo.jpg" alt="Fallback">
</afix-img>
```

## Attributes

| Attribute | Description |
|-----------|-------------|
| `src` | Path to `.afix` file |
| `alt` | Accessible description |
| `width` / `height` | Explicit dimensions |
| `loading` | `lazy` or `eager` (default: `eager`) |
| `parallax` | Enable depth parallax on device tilt (boolean) |
| `lod` | Maximum LOD to decode (`0`=skeleton, `1`=textured, `2`=lossless) |

## Events

| Event | Description |
|-------|-------------|
| `afix:skeleton` | S1 (vector skeleton) decoded and rendered |
| `afix:textured` | S2 (neural texture) decoded and composited |
| `afix:ready` | All requested LOD layers decoded |
| `afix:error` | Decode or network error |

## Status

🚧 **Phase 1 — In Development**
