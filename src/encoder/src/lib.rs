//! # afix-encoder (Yaka-Core)
//!
//! Converts JPEG, PNG, and other raster images into the `.aFix` format.
//!
//! ## Pipeline
//!
//! ```text
//! Source Image (JPEG/PNG)
//!     │
//!     ▼
//! Pre-processing  (YCbCr, normalisation)
//!     │
//!     ├── JPEG Preview (PREV)            → instant display on legacy viewers
//!     │
//!     ├── Edge Detection + B-Spline Fit → S1 / VEC_
//!     │
//!     ├── Saliency + VAE Encode         → S2 / LAT_
//!     │
//!     └── Residual Calculation          → S3 / RES_  (lossless profiles)
//!         │
//!         ▼
//! Atom Packer + CRC → .aFix output
//! ```

use std::io::{Seek, Write};
use std::path::Path;

use image::DynamicImage;
use libafix::{AfixError, AfixFile, Chunk, ChunkId, Profile, Result, manifest::{ObjectManifest, SemanticObject}};

// ── Public API ────────────────────────────────────────────────────────────────

/// Configuration for an encode operation.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Encoding profile (determines which chunks are written).
    pub profile: Profile,
    /// Neural latent quality, 0–100 (higher = more latent data, better quality).
    pub quality: u8,
    /// Whether to auto-detect semantic objects and write an `OBJM` chunk.
    pub semantic: bool,
    /// Whether to embed a JPEG preview (`PREV` chunk) for backward compatibility.
    ///
    /// When `true` (the default), a down-sampled JPEG is written as the very
    /// first chunk inside the `PAYLOAD`.  Legacy tools that do not understand
    /// `.aFix` can extract this chunk and display it directly.  New decoders
    /// show it instantly as a "loading" frame before the neural layers arrive.
    pub preview: bool,
    /// JPEG quality for the embedded preview, 1–100 (default: 60).
    pub preview_quality: u8,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            profile: Profile::WebLossy,
            quality: 85,
            semantic: true,
            preview: true,
            preview_quality: 60,
        }
    }
}

/// Encode a source image file into an `.aFix` file at `output_path`.
pub fn encode_file<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    options: &EncodeOptions,
) -> Result<()> {
    let img = image::open(input_path.as_ref()).map_err(|e| {
        AfixError::InvalidChunkData(format!("cannot open source image: {e}"))
    })?;
    let mut file = std::fs::File::create(output_path.as_ref())
        .map_err(AfixError::Io)?;
    encode_image(&img, &mut file, options)
}

/// Encode a [`DynamicImage`] into an `.aFix` file written to `writer`.
pub fn encode_image<W: Write + Seek>(
    img: &DynamicImage,
    writer: W,
    options: &EncodeOptions,
) -> Result<()> {
    let width = img.width() as f64;
    let height = img.height() as f64;

    let mut afix = AfixFile::new(width, height, options.profile);

    // ── META chunk ────────────────────────────────────────────────────────────
    let meta = build_meta_chunk(options);
    afix.add_chunk(meta);

    // ── PREV — JPEG preview (first chunk for legacy compatibility) ────────────
    // Placed immediately after META so that legacy tools encounter it as early
    // as possible when scanning the PAYLOAD sequentially.
    if options.preview {
        let prev_data = encode_preview(img, options.preview_quality)?;
        afix.add_chunk(Chunk { id: ChunkId::Preview, flags: 0, data: prev_data });
    }

    // ── S1 — VEC_ (Geometric Skeleton) ───────────────────────────────────────
    let s1_data = encode_s1(img, options.quality);
    afix.add_chunk(Chunk { id: ChunkId::Vec, flags: 0, data: s1_data });

    // ── S2 — LAT_ (Latent Texture Field) ─────────────────────────────────────
    let s2_data = encode_s2(img, options.quality);
    afix.add_chunk(Chunk { id: ChunkId::Lat, flags: 0, data: s2_data });

    // ── S3 — RES_ (Parity Residual) — lossless profiles only ─────────────────
    if options.profile.requires_residual() {
        let s3_data = encode_s3(img, options.quality);
        afix.add_chunk(Chunk { id: ChunkId::Res, flags: 0, data: s3_data });
    }

    // ── OBJM — Semantic Object Manifest ──────────────────────────────────────
    if options.semantic {
        let manifest = detect_objects(img);
        let json = manifest
            .to_chunk_data()
            .map_err(|e| AfixError::InvalidChunkData(e.to_string()))?;
        afix.add_chunk(Chunk { id: ChunkId::ObjManifest, flags: 0, data: json });
    }

    afix.write(writer)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build the META chunk JSON payload.
fn build_meta_chunk(options: &EncodeOptions) -> Chunk {
    let meta = serde_json::json!({
        "version": "1.0",
        "creator": "afix-encoder/1.0.0 (Yaka-Core)",
        "profile": options.profile.to_string(),
        "quality": options.quality,
        "latent_scale_factors": [0.018, 0.018, 0.018, 0.018],
        "latent_zero_points": [0, 0, 0, 0]
    });
    Chunk {
        id: ChunkId::Meta,
        flags: 0,
        data: meta.to_string().into_bytes(),
    }
}

/// Encode the S1 Geometric Skeleton (`VEC_` chunk).
///
/// This produces a compact binary representation of the image's structural
/// content by:
/// 1. Converting to greyscale and applying a Sobel edge filter.
/// 2. Sampling strong edge pixels and storing their coordinates as delta-coded
///    16-bit fixed-point pairs.
///
/// A full production encoder would fit B-Spline curves here; this
/// implementation stores the raw edge-pixel list as a portable baseline that
/// conforms to the `VEC_` chunk format.
fn encode_s1(img: &DynamicImage, _quality: u8) -> Vec<u8> {
    let grey = img.to_luma8();
    let (w, h) = (grey.width(), grey.height());

    // Simple Sobel edge detection.
    let mut edges: Vec<(u16, u16)> = Vec::new();
    let threshold: u16 = 30;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let gx: i32 = sobel_gx(&grey, x, y);
            let gy: i32 = sobel_gy(&grey, x, y);
            let mag = ((gx * gx + gy * gy) as f64).sqrt() as u16;
            if mag > threshold {
                edges.push((x as u16, y as u16));
            }
        }
    }

    // Pack as: [count: u32 LE] [x0: u16 LE, y0: u16 LE, x1: u16 LE, ...]
    let mut out = Vec::with_capacity(4 + edges.len() * 4);
    let count = edges.len() as u32;
    out.extend_from_slice(&count.to_le_bytes());
    for (x, y) in &edges {
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
    }
    out
}

fn sobel_gx(img: &image::GrayImage, x: u32, y: u32) -> i32 {
    let p = |dx: i32, dy: i32| img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as i32;
    -p(-1, -1) + p(1, -1) - 2 * p(-1, 0) + 2 * p(1, 0) - p(-1, 1) + p(1, 1)
}

fn sobel_gy(img: &image::GrayImage, x: u32, y: u32) -> i32 {
    let p = |dx: i32, dy: i32| img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as i32;
    -p(-1, -1) - 2 * p(0, -1) - p(1, -1) + p(-1, 1) + 2 * p(0, 1) + p(1, 1)
}

/// Encode the S2 Latent Texture Field (`LAT_` chunk).
///
/// A production encoder uses a quantised VAE; this implementation downsamples
/// the image to 128×128, converts to float16 (stored as f32 for portability),
/// and packs the result as a raw tensor.  The tensor dimensions and element
/// size are stored in a 12-byte header so decoders can reconstruct the data.
fn encode_s2(img: &DynamicImage, quality: u8) -> Vec<u8> {
    // Determine latent spatial resolution based on quality.
    let lat_size: u32 = if quality >= 90 { 128 } else if quality >= 60 { 64 } else { 32 };
    let channels: u32 = 4; // RGBA

    let resized = img.resize_exact(lat_size, lat_size, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();

    // Header: [width: u32 LE][height: u32 LE][channels: u32 LE]
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&lat_size.to_le_bytes());
    out.extend_from_slice(&lat_size.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());

    // Normalised pixel values stored as f32 (stand-in for float16 latents).
    for pixel in rgba.pixels() {
        for &channel_val in pixel.0.iter() {
            let normalised: f32 = (channel_val as f32 / 255.0) * 2.0 - 1.0;
            out.extend_from_slice(&normalised.to_le_bytes());
        }
    }
    out
}

/// Encode the S3 Parity Residual (`RES_` chunk).
///
/// Stores the difference between the original pixel data and the S2 synthesis
/// in YCbCr space. A production encoder uses HEVC Intra or AV1 Still; here we
/// store the raw residuals as a placeholder that preserves format correctness.
fn encode_s3(img: &DynamicImage, _quality: u8) -> Vec<u8> {
    // For lossless mode, store the full original image as PNG bytes so a
    // decoder can reconstruct the pixel-perfect original.  In a production
    // encoder this would be `original - VAE_decode(S2_latents)`.
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"RES1"); // sub-format identifier
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    // Store raw RGB bytes as residual placeholder.
    out.extend_from_slice(rgb.as_raw());
    out
}

/// Produce a basic `ObjectManifest` by dividing the image into coarse regions.
///
/// A production encoder uses a MobileNet-SSD segmentation model. This
/// heuristic divides the image into top/bottom halves and assigns "sky" and
/// "ground" labels, giving the manifest meaningful (if approximate) content
/// for demonstration and testing purposes.
fn detect_objects(img: &DynamicImage) -> ObjectManifest {
    let w = img.width() as f64;
    let h = img.height() as f64;
    let half_h = h / 2.0;

    ObjectManifest {
        version: "1.0".into(),
        objects: vec![
            SemanticObject {
                id: "sky".into(),
                label: "sky".into(),
                category: "background".into(),
                mask_rle: None,
                bbox: Some([0.0, 0.0, w, half_h]),
                confidence: Some(HEURISTIC_DETECTION_CONFIDENCE),
                landmarks: None,
            },
            SemanticObject {
                id: "ground".into(),
                label: "ground".into(),
                category: "background".into(),
                mask_rle: None,
                bbox: Some([0.0, half_h, w, half_h]),
                confidence: Some(HEURISTIC_DETECTION_CONFIDENCE),
                landmarks: None,
            },
        ],
    }
}

/// Confidence score used by the heuristic region detector.
/// A production encoder replaces this with real segmentation model scores.
const HEURISTIC_DETECTION_CONFIDENCE: f64 = 0.80;

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use libafix::AfixFile;
    use std::io::{Cursor, Seek, SeekFrom};

    fn synthetic_image(w: u32, h: u32) -> DynamicImage {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            Rgb([(x * 255 / w) as u8, (y * 255 / h) as u8, 128u8])
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn encode_web_lossy_roundtrip() {
        let img = synthetic_image(64, 64);
        let mut buf = Cursor::new(Vec::new());
        let opts = EncodeOptions { profile: Profile::WebLossy, quality: 85, semantic: true };
        encode_image(&img, &mut buf, &opts).expect("encode failed");

        buf.seek(SeekFrom::Start(0)).unwrap();
        let parsed = AfixFile::read(&mut buf).expect("parse failed");
        assert_eq!(parsed.header.dimensions.width, 64.0);
        assert_eq!(parsed.header.dimensions.height, 64.0);
        assert!(parsed.get_chunk(ChunkId::Meta).is_some());
        assert!(parsed.get_chunk(ChunkId::Vec).is_some());
        assert!(parsed.get_chunk(ChunkId::Lat).is_some());
        assert!(parsed.get_chunk(ChunkId::Res).is_none(), "lossless chunk must not appear");
        assert!(parsed.get_chunk(ChunkId::ObjManifest).is_some());
    }

    #[test]
    fn encode_web_lossless_has_residual() {
        let img = synthetic_image(32, 32);
        let mut buf = Cursor::new(Vec::new());
        let opts = EncodeOptions { profile: Profile::WebLossless, quality: 90, semantic: false };
        encode_image(&img, &mut buf, &opts).expect("encode failed");

        buf.seek(SeekFrom::Start(0)).unwrap();
        let parsed = AfixFile::read(&mut buf).expect("parse failed");
        assert!(parsed.get_chunk(ChunkId::Res).is_some(), "residual chunk must be present");
    }

    #[test]
    fn meta_chunk_is_valid_json() {
        let img = synthetic_image(16, 16);
        let mut buf = Cursor::new(Vec::new());
        let opts = EncodeOptions::default();
        encode_image(&img, &mut buf, &opts).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let parsed = AfixFile::read(&mut buf).unwrap();
        let meta_chunk = parsed.get_chunk(ChunkId::Meta).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&meta_chunk.data).unwrap();
        assert_eq!(json["version"], "1.0");
        assert_eq!(json["profile"], "web-lossy");
    }

    #[test]
    fn objm_chunk_has_two_objects() {
        let img = synthetic_image(64, 64);
        let mut buf = Cursor::new(Vec::new());
        let opts = EncodeOptions { profile: Profile::WebLossy, quality: 85, semantic: true };
        encode_image(&img, &mut buf, &opts).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let parsed = AfixFile::read(&mut buf).unwrap();
        let objm_chunk = parsed.get_chunk(ChunkId::ObjManifest).unwrap();
        let manifest = libafix::ObjectManifest::from_chunk_data(&objm_chunk.data).unwrap();
        assert_eq!(manifest.objects.len(), 2);
    }
}
