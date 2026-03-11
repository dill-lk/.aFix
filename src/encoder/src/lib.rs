//! # afix-encoder (Yaka-Core)
//!
//! Converts JPEG, PNG, and other raster images into the `.aFix` format.
//!
//! ## Encode pipeline
//!
//! ```text
//! Source Image (JPEG/PNG)
//!     │
//!     ▼
//! Pre-processing  (RGB → YCbCr, normalisation)
//!     │
//!     ├── JPEG Preview (PREV)            → instant display on legacy viewers
//!     │
//!     ├── Canny edge detection
//!     │   + B-Spline curve fitting       → S1 / VEC_
//!     │
//!     ├── Gradient saliency map (W_s)
//!     │   + DCT tile compression         → S2 / LAT_
//!     │                                    (VAE ONNX if model file present)
//!     │
//!     ├── S2 reconstruction error        → S3 / RES_  (lossless profiles)
//!     │
//!     └── Semantic object detection      → OBJM
//!         (saliency-guided region scoring)
//!         │
//!         ▼
//! Atom Packer + CRC-32 → .aFix output
//! ```

use std::io::{Cursor, Seek, Write};
use std::path::Path;

use image::DynamicImage;
use libafix::{
    AfixError, AfixFile, Chunk, ChunkId, Profile, Result,
    manifest::{ObjectManifest, SemanticObject},
};

pub mod bspline;
pub mod canny;
pub mod dct;
pub mod saliency;

use bspline::{fit_splines, serialise_splines};
use canny::canny;
use dct::{decode_dct, encode_dct};
use saliency::compute_saliency;

// ── Public API ────────────────────────────────────────────────────────────────

/// Configuration for an encode operation.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Encoding profile (determines which chunks are written).
    pub profile: Profile,
    /// Neural latent quality, 0–100 (higher = more detail, larger file).
    pub quality: u8,
    /// Whether to auto-detect semantic objects and write an `OBJM` chunk.
    pub semantic: bool,
    /// Whether to embed a JPEG preview (`PREV` chunk) for backward compatibility.
    ///
    /// When `true` (the default), a JPEG is written as the second chunk in the
    /// `PAYLOAD` (right after `META`).  Legacy tools that do not understand
    /// `.aFix` can extract this chunk and display it directly.  New decoders
    /// show it instantly as a "loading" frame while the neural layers decode.
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
    let mut file = std::fs::File::create(output_path.as_ref()).map_err(AfixError::Io)?;
    encode_image(&img, &mut file, options)
}

/// Encode a [`DynamicImage`] into an `.aFix` file written to `writer`.
pub fn encode_image<W: Write + Seek>(
    img: &DynamicImage,
    writer: W,
    options: &EncodeOptions,
) -> Result<()> {
    let width  = img.width()  as f64;
    let height = img.height() as f64;

    let mut afix = AfixFile::new(width, height, options.profile);

    // ── META chunk ────────────────────────────────────────────────────────────
    afix.add_chunk(build_meta_chunk(options));

    // ── PREV — JPEG preview (placed early for legacy tool compatibility) ───────
    if options.preview {
        let prev_data = encode_preview(img, options.preview_quality)?;
        afix.add_chunk(Chunk { id: ChunkId::Preview, flags: 0, data: prev_data });
    }

    // ── Shared pre-processing: greyscale luma + saliency map ──────────────────
    let grey     = img.to_luma8();
    let grey_buf = grey.as_raw().as_slice();
    let sal_map  = compute_saliency(grey_buf, img.width(), img.height(), 0.3);

    // ── S1 — VEC_ : Canny edge detection + B-Spline fitting ──────────────────
    let (low_t, high_t) = canny_thresholds(options.quality);
    let edge_map = canny(grey_buf, img.width(), img.height(), low_t, high_t);
    let splines  = fit_splines(&edge_map, 8);
    let s1_data  = serialise_splines(&splines);
    afix.add_chunk(Chunk { id: ChunkId::Vec, flags: 0, data: s1_data });

    // ── S2 — LAT_ : saliency-weighted DCT compression ─────────────────────────
    let rgb    = img.to_rgb8();
    let s2_data = encode_dct(rgb.as_raw(), img.width(), img.height(), options.quality, &sal_map);
    afix.add_chunk(Chunk { id: ChunkId::Lat, flags: 0, data: s2_data.clone() });

    // ── S3 — RES_ : parity residual (lossless profiles only) ─────────────────
    if options.profile.requires_residual() {
        let s3_data = encode_residual(img, &s2_data)?;
        afix.add_chunk(Chunk { id: ChunkId::Res, flags: 0, data: s3_data });
    }

    // ── OBJM — Semantic Object Manifest ──────────────────────────────────────
    if options.semantic {
        let manifest = detect_objects(img, &sal_map);
        let json = manifest
            .to_chunk_data()
            .map_err(|e| AfixError::InvalidChunkData(e.to_string()))?;
        afix.add_chunk(Chunk { id: ChunkId::ObjManifest, flags: 0, data: json });
    }

    afix.write(writer)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn build_meta_chunk(options: &EncodeOptions) -> Chunk {
    let meta = serde_json::json!({
        "version": "1.0",
        "creator": "afix-encoder/1.0.0 (Yaka-Core)",
        "profile": options.profile.to_string(),
        "quality": options.quality,
        "s2_codec": "dct",
        "latent_scale_factors": [0.018, 0.018, 0.018, 0.018],
        "latent_zero_points": [0, 0, 0, 0]
    });
    Chunk { id: ChunkId::Meta, flags: 0, data: meta.to_string().into_bytes() }
}

fn encode_preview(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    const MAX_PREVIEW: u32 = 512;
    let preview = if img.width() > MAX_PREVIEW || img.height() > MAX_PREVIEW {
        img.thumbnail(MAX_PREVIEW, MAX_PREVIEW)
    } else {
        img.clone()
    };

    let rgb = preview.to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode(rgb.as_raw(), rgb.width(), rgb.height(), image::ColorType::Rgb8.into())
        .map_err(|e| AfixError::InvalidChunkData(format!("JPEG preview encode error: {e}")))?;
    Ok(buf.into_inner())
}

fn canny_thresholds(quality: u8) -> (f32, f32) {
    let q = quality as f32 / 100.0;
    let high = 80.0 - q * 50.0;
    let low  = high * 0.4;
    (low, high)
}

fn encode_residual(img: &DynamicImage, s2_data: &[u8]) -> Result<Vec<u8>> {
    let rgb_orig = img.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let (rgb_s2, dw, dh) = decode_dct(s2_data)
        .ok_or_else(|| AfixError::InvalidChunkData("failed to decode S2 for residual".into()))?;

    let mut out = Vec::new();
    out.extend_from_slice(b"RES2");
    out.extend_from_slice(&(w as u32).to_le_bytes());
    out.extend_from_slice(&(h as u32).to_le_bytes());

    let dw = dw as usize;
    let dh = dh as usize;
    for py in 0..h {
        for px in 0..w {
            for ch in 0..3usize {
                let orig = rgb_orig.as_raw()[(py * w + px) * 3 + ch] as i16;
                let rec  = if px < dw && py < dh {
                    rgb_s2[(py * dw + px) * 3 + ch] as i16
                } else {
                    orig
                };
                out.extend_from_slice(&(orig - rec).to_le_bytes());
            }
        }
    }
    Ok(out)
}

fn detect_objects(img: &DynamicImage, sal: &saliency::SaliencyMap) -> ObjectManifest {
    let w  = img.width();
    let h  = img.height();
    let cw = (w / 3).max(1);
    let ch = (h / 3).max(1);

    const SUBJECT_THRESHOLD: f32 = 0.55;
    let mut objects = Vec::new();
    let mut subj_id = 0u32;

    for grid_y in 0..3u32 {
        for grid_x in 0..3u32 {
            let rx = grid_x * cw;
            let ry = grid_y * ch;
            let mean_sal = sal.region_mean(rx, ry, cw, ch);

            let (id, label, category) = if mean_sal >= SUBJECT_THRESHOLD {
                subj_id += 1;
                (format!("subject_{subj_id}"), "subject".to_string(), "subject".to_string())
            } else {
                (
                    format!("region_{}_{}", grid_x, grid_y),
                    "background".to_string(),
                    "background".to_string(),
                )
            };

            objects.push(SemanticObject {
                id,
                label,
                category,
                mask_rle: None,
                bbox: Some([rx as f64, ry as f64, cw as f64, ch as f64]),
                confidence: Some(mean_sal as f64),
                landmarks: None,
            });
        }
    }

    ObjectManifest { version: "1.0".into(), objects }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use libafix::AfixFile;
    use std::io::{Cursor, SeekFrom, Seek};

    fn synthetic_image(w: u32, h: u32) -> DynamicImage {
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            Rgb([(x * 255 / w) as u8, (y * 255 / h) as u8, 128u8])
        });
        DynamicImage::ImageRgb8(buf)
    }

    fn encode_to_afix(img: &DynamicImage, opts: &EncodeOptions) -> AfixFile {
        let mut buf = Cursor::new(Vec::new());
        encode_image(img, &mut buf, opts).expect("encode failed");
        buf.seek(SeekFrom::Start(0)).unwrap();
        AfixFile::read(&mut buf).expect("parse failed")
    }

    #[test]
    fn prev_chunk_present_by_default() {
        let afix = encode_to_afix(&synthetic_image(32, 32), &EncodeOptions::default());
        assert!(afix.get_chunk(ChunkId::Preview).is_some());
    }

    #[test]
    fn prev_chunk_is_valid_jpeg() {
        let afix = encode_to_afix(&synthetic_image(64, 64), &EncodeOptions::default());
        let prev = afix.get_chunk(ChunkId::Preview).unwrap();
        assert_eq!(&prev.data[0..3], &[0xFF, 0xD8, 0xFF], "must start with JPEG magic");
    }

    #[test]
    fn prev_chunk_absent_when_disabled() {
        let opts = EncodeOptions { preview: false, ..Default::default() };
        let afix = encode_to_afix(&synthetic_image(32, 32), &opts);
        assert!(afix.get_chunk(ChunkId::Preview).is_none());
    }

    #[test]
    fn all_mandatory_chunks_present() {
        let afix = encode_to_afix(&synthetic_image(64, 64), &EncodeOptions::default());
        assert!(afix.get_chunk(ChunkId::Meta).is_some());
        assert!(afix.get_chunk(ChunkId::Vec).is_some());
        assert!(afix.get_chunk(ChunkId::Lat).is_some());
        assert!(afix.get_chunk(ChunkId::Res).is_none(), "no RES in lossy");
        assert!(afix.get_chunk(ChunkId::ObjManifest).is_some());
    }

    #[test]
    fn lossless_profile_has_residual() {
        let opts = EncodeOptions { profile: Profile::WebLossless, quality: 90, semantic: false, ..Default::default() };
        let afix = encode_to_afix(&synthetic_image(32, 32), &opts);
        assert!(afix.get_chunk(ChunkId::Res).is_some());
    }

    #[test]
    fn dimensions_preserved() {
        let afix = encode_to_afix(&synthetic_image(48, 36), &EncodeOptions::default());
        assert_eq!(afix.header.dimensions.width, 48.0);
        assert_eq!(afix.header.dimensions.height, 36.0);
    }

    #[test]
    fn meta_has_dct_codec_field() {
        let afix = encode_to_afix(&synthetic_image(16, 16), &EncodeOptions::default());
        let meta = afix.get_chunk(ChunkId::Meta).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&meta.data).unwrap();
        assert_eq!(json["s2_codec"], "dct");
    }

    #[test]
    fn vec_chunk_parseable_by_bspline_module() {
        let afix = encode_to_afix(&synthetic_image(64, 64), &EncodeOptions::default());
        let vec_chunk = afix.get_chunk(ChunkId::Vec).unwrap();
        assert!(bspline::deserialise_splines(&vec_chunk.data).is_some());
    }

    #[test]
    fn lat_chunk_has_dct_sub_format() {
        let afix = encode_to_afix(&synthetic_image(32, 32), &EncodeOptions::default());
        let lat = afix.get_chunk(ChunkId::Lat).unwrap();
        assert_eq!(lat.data[0], 0x02, "LAT_ sub-format byte must be 0x02 (DCT)");
    }

    #[test]
    fn objm_has_nine_regions() {
        let afix = encode_to_afix(&synthetic_image(64, 64), &EncodeOptions::default());
        let objm = afix.get_chunk(ChunkId::ObjManifest).unwrap();
        let manifest = libafix::ObjectManifest::from_chunk_data(&objm.data).unwrap();
        assert_eq!(manifest.objects.len(), 9, "3×3 grid should yield 9 OBJM regions");
    }
}
