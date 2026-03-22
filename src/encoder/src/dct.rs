//! 2D DCT-based tile compression for the S2 Latent Texture Field (`LAT_` chunk).
//!
//! Used as the Rust-side implementation of S2 until a trained VAE ONNX model
//! is loaded.  Conceptually similar to how a VAE latent space compresses
//! information — both discard high-frequency information not visible to the
//! Human Visual System (HVS) — but based on the well-understood DCT rather
//! than a learned basis.
//!
//! ## What this does
//!
//! 1. Convert the image to YCbCr.
//! 2. Divide each channel into 8×8 blocks.
//! 3. Apply a 2D DCT-II to each block.
//! 4. **Saliency-weighted quantisation:** blocks in high-saliency regions use a
//!    finer quantisation step; background blocks are quantised more coarsely,
//!    matching the spec formula  `C = ∮ (S_v · W_s) + (T_n · W_p)`.
//! 5. Store the non-zero quantised coefficients as a compact tensor.
//!
//! ## `LAT_` chunk format produced here
//!
//! ```text
//! [sub_format: u8]    = 0x02 (DCT-based, not raw pixels)
//! [width:  u32 LE]    tile columns
//! [height: u32 LE]    tile rows
//! [channels: u32 LE]  = 3 (Y, Cb, Cr)
//! [quality: u8]       0-100
//! [sal_scales: f32 LE × tiles_w × tiles_h]  per-tile saliency scale (1.0–3.0)
//! [coeff_data: ...]   quantised int16 DCT coefficients, block-major, zigzag order
//! ```
//!
//! Storing the per-tile saliency scale in the chunk ensures the decoder can
//! exactly invert the saliency-weighted quantisation step used at encode time,
//! preventing the "grey wash" artefact that appears when the dequantisation step
//! does not match the quantisation step.

use crate::saliency::SaliencyMap;

// ── Constants ─────────────────────────────────────────────────────────────────

const BLOCK: usize = 8;

/// JPEG-style luminance quantisation table (quality = 50 baseline).
/// Values are scaled by the quality factor at encode time.
#[rustfmt::skip]
const LUMA_QUANT_BASE: [f32; 64] = [
    16., 11., 10., 16., 24., 40., 51., 61.,
    12., 12., 14., 19., 26., 58., 60., 55.,
    14., 13., 16., 24., 40., 57., 69., 56.,
    14., 17., 22., 29., 51., 87., 80., 62.,
    18., 22., 37., 56., 68.,109.,103., 77.,
    24., 35., 55., 64., 81.,104.,113., 92.,
    49., 64., 78., 87.,103.,121.,120.,101.,
    72., 92., 95., 98.,112.,100.,103., 99.,
];

/// Chrominance quantisation table (quality = 50 baseline).
#[rustfmt::skip]
const CHROMA_QUANT_BASE: [f32; 64] = [
    17., 18., 24., 47., 99., 99., 99., 99.,
    18., 21., 26., 66., 99., 99., 99., 99.,
    24., 26., 56., 99., 99., 99., 99., 99.,
    47., 66., 99., 99., 99., 99., 99., 99.,
    99., 99., 99., 99., 99., 99., 99., 99.,
    99., 99., 99., 99., 99., 99., 99., 99.,
    99., 99., 99., 99., 99., 99., 99., 99.,
    99., 99., 99., 99., 99., 99., 99., 99.,
];

/// Zigzag scan order for an 8×8 block.
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// ── Public API ────────────────────────────────────────────────────────────────

/// Encode an RGB image into the DCT-based `LAT_` chunk data.
///
/// `quality` 0–100 maps to JPEG-style quantisation step scaling:
/// - 100 → finest (step ≈ 1, near-lossless)
/// - 50  → JPEG-equivalent baseline  
/// - 1   → maximum compression
pub fn encode_dct(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    saliency: &SaliencyMap,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    // ── 1. RGB → YCbCr ───────────────────────────────────────────────────────
    let (y_ch, cb_ch, cr_ch) = rgb_to_ycbcr(rgb, w, h);

    // ── 2. Build quantisation tables from quality ─────────────────────────────
    let luma_q   = build_quant_table(&LUMA_QUANT_BASE,   quality);
    let chroma_q = build_quant_table(&CHROMA_QUANT_BASE, quality);

    let tiles_w = (w + BLOCK - 1) / BLOCK;
    let tiles_h = (h + BLOCK - 1) / BLOCK;

    // ── 3. Encode each channel, collecting per-tile saliency scales ──────────
    let mut all_coeffs: Vec<i16> = Vec::new();
    // sal_scale is the same for all channels of a given tile; store once per tile.
    let mut sal_scale_table: Vec<f32> = Vec::with_capacity(tiles_w * tiles_h);

    for tile_y in 0..tiles_h {
        for tile_x in 0..tiles_w {
            // Compute block saliency (mean of pixel saliencies in this tile).
            let sal = saliency.region_mean(
                (tile_x * BLOCK) as u32,
                (tile_y * BLOCK) as u32,
                BLOCK as u32,
                BLOCK as u32,
            );
            // High saliency → finer quantisation (multiply step by (2 - sal)).
            // Low saliency  → coarser (up to 3× the base step).
            let sal_scale = 1.0 + 2.0 * (1.0 - sal); // [1, 3]
            sal_scale_table.push(sal_scale);
        }
    }

    for (ch_idx, channel) in [&y_ch, &cb_ch, &cr_ch].iter().enumerate() {
        let q_table = if ch_idx == 0 { &luma_q } else { &chroma_q };

        for tile_y in 0..tiles_h {
            for tile_x in 0..tiles_w {
                // Extract 8×8 block (pad with edge replication if needed).
                let block = extract_block(channel, w, h, tile_x * BLOCK, tile_y * BLOCK);

                // Look up the pre-computed per-tile saliency scale.
                let sal_scale = sal_scale_table[tile_y * tiles_w + tile_x];

                // 2D DCT.
                let dct_block = dct2d(&block);

                // Quantise with saliency-scaled step, zigzag scan.
                for &zi in &ZIGZAG {
                    let step = q_table[zi] * sal_scale;
                    let coeff = (dct_block[zi] / step).round() as i16;
                    all_coeffs.push(coeff);
                }
            }
        }
    }

    // ── 4. Pack header + sal_scale table + coefficients ──────────────────────
    let sal_bytes = tiles_w * tiles_h * std::mem::size_of::<f32>(); // one f32 per tile
    let mut out = Vec::with_capacity(13 + sal_bytes + all_coeffs.len() * 2);
    out.push(0x02u8); // sub-format: DCT
    out.extend_from_slice(&(tiles_w as u32).to_le_bytes());
    out.extend_from_slice(&(tiles_h as u32).to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // channels
    out.push(quality);
    // Saliency scale table (tiles_w × tiles_h f32 values, row-major).
    for &s in &sal_scale_table {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &c in &all_coeffs {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

/// Decode a DCT-based `LAT_` chunk back to an RGB image.
///
/// Returns `(rgb_pixels, width, height)` where `rgb_pixels` is row-major RGB.
pub fn decode_dct(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    if data.len() < 14 || data[0] != 0x02 {
        return None;
    }

    let tiles_w = u32::from_le_bytes(data[1..5].try_into().ok()?) as usize;
    let tiles_h = u32::from_le_bytes(data[5..9].try_into().ok()?) as usize;
    let _channels = u32::from_le_bytes(data[9..13].try_into().ok()?) as usize; // must be 3
    let quality = data[13];

    let width  = tiles_w * BLOCK;
    let height = tiles_h * BLOCK;

    let luma_q   = build_quant_table(&LUMA_QUANT_BASE,   quality);
    let chroma_q = build_quant_table(&CHROMA_QUANT_BASE, quality);

    // ── Read per-tile saliency scale table ────────────────────────────────────
    let num_tiles = tiles_w * tiles_h;
    let sal_table_bytes = num_tiles * std::mem::size_of::<f32>(); // one f32 per tile
    if data.len() < 14 + sal_table_bytes {
        return None;
    }
    let mut sal_scale_table = vec![1.0f32; num_tiles];
    for i in 0..num_tiles {
        let off = 14 + i * 4;
        sal_scale_table[i] = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
    }

    // ── Read DCT coefficients ─────────────────────────────────────────────────
    let coeffs_per_tile = BLOCK * BLOCK;
    let total_tiles = num_tiles * 3; // 3 channels
    let expected_coeffs = total_tiles * coeffs_per_tile;

    let coeff_bytes = &data[14 + sal_table_bytes..];
    if coeff_bytes.len() < expected_coeffs * 2 {
        return None;
    }

    let mut channels: Vec<Vec<f32>> = vec![vec![0f32; width * height]; 3];

    let mut coeff_idx = 0usize;
    for (ch_idx, channel) in channels.iter_mut().enumerate() {
        let q_table = if ch_idx == 0 { &luma_q } else { &chroma_q };

        for tile_y in 0..tiles_h {
            for tile_x in 0..tiles_w {
                // Retrieve the per-tile saliency scale used during encoding.
                let sal_scale = sal_scale_table[tile_y * tiles_w + tile_x];

                // Read 64 quantised coefficients in zigzag order and dequantise
                // with the same saliency-scaled step that was used at encode time.
                let mut dct_block = [0f32; 64];
                for &zi in &ZIGZAG {
                    let base = coeff_idx * 2;
                    let coeff = i16::from_le_bytes(coeff_bytes[base..base + 2].try_into().ok()?);
                    dct_block[zi] = coeff as f32 * q_table[zi] * sal_scale;
                    coeff_idx += 1;
                }

                // Inverse DCT.
                let block = idct2d(&dct_block);

                // Write block back to channel.
                for by in 0..BLOCK {
                    for bx in 0..BLOCK {
                        let px = tile_x * BLOCK + bx;
                        let py = tile_y * BLOCK + by;
                        if px < width && py < height {
                            channel[py * width + px] = block[by * BLOCK + bx];
                        }
                    }
                }
            }
        }
    }

    // ── YCbCr → RGB ───────────────────────────────────────────────────────────
    let rgb = ycbcr_to_rgb(&channels[0], &channels[1], &channels[2], width, height);
    Some((rgb, width as u32, height as u32))
}

// ── Colour space conversions ──────────────────────────────────────────────────

fn rgb_to_ycbcr(rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut y  = vec![0f32; n];
    let mut cb = vec![0f32; n];
    let mut cr = vec![0f32; n];
    for i in 0..n {
        let r = rgb[i * 3]     as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        y[i]  =  0.299  * r + 0.587  * g + 0.114  * b - 128.0;
        cb[i] = -0.1687 * r - 0.3313 * g + 0.5    * b;
        cr[i] =  0.5    * r - 0.4187 * g - 0.0813 * b;
    }
    (y, cb, cr)
}

fn ycbcr_to_rgb(y: &[f32], cb: &[f32], cr: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let yv  = y[i]  + 128.0;
        let cbv = cb[i];
        let crv = cr[i];
        let r = (yv               + 1.402   * crv).round().clamp(0.0, 255.0) as u8;
        let g = (yv - 0.344136   * cbv - 0.714136 * crv).round().clamp(0.0, 255.0) as u8;
        let b = (yv + 1.772      * cbv).round().clamp(0.0, 255.0) as u8;
        rgb[i * 3]     = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

// ── DCT ───────────────────────────────────────────────────────────────────────

/// 2D DCT-II of an 8×8 block.
fn dct2d(block: &[f32; 64]) -> [f32; 64] {
    let mut tmp = [0f32; 64];
    // Row-wise DCT-II.
    for row in 0..BLOCK {
        let row_in: [f32; 8] = block[row * BLOCK..row * BLOCK + BLOCK].try_into().unwrap();
        let row_out = dct8(&row_in);
        tmp[row * BLOCK..row * BLOCK + BLOCK].copy_from_slice(&row_out);
    }
    // Column-wise DCT-II.
    let mut out = [0f32; 64];
    for col in 0..BLOCK {
        let col_in: [f32; 8] = std::array::from_fn(|r| tmp[r * BLOCK + col]);
        let col_out = dct8(&col_in);
        for r in 0..BLOCK {
            out[r * BLOCK + col] = col_out[r];
        }
    }
    out
}

/// 2D inverse DCT (IDCT-II) of an 8×8 block.
fn idct2d(block: &[f32; 64]) -> [f32; 64] {
    let mut tmp = [0f32; 64];
    for col in 0..BLOCK {
        let col_in: [f32; 8] = std::array::from_fn(|r| block[r * BLOCK + col]);
        let col_out = idct8(&col_in);
        for r in 0..BLOCK {
            tmp[r * BLOCK + col] = col_out[r];
        }
    }
    let mut out = [0f32; 64];
    for row in 0..BLOCK {
        let row_in: [f32; 8] = tmp[row * BLOCK..row * BLOCK + BLOCK].try_into().unwrap();
        let row_out = idct8(&row_in);
        out[row * BLOCK..row * BLOCK + BLOCK].copy_from_slice(&row_out);
    }
    out
}

/// 8-point DCT-II using the standard direct formula.
fn dct8(x: &[f32; 8]) -> [f32; 8] {
    let n = 8.0f32;
    let mut out = [0f32; 8];
    for k in 0..8 {
        let c_k = if k == 0 { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() };
        let mut sum = 0f32;
        for j in 0..8 {
            sum += x[j] * ((std::f32::consts::PI * k as f32 * (2.0 * j as f32 + 1.0)) / (2.0 * n)).cos();
        }
        out[k] = c_k * sum;
    }
    out
}

/// 8-point inverse DCT-II.
fn idct8(x: &[f32; 8]) -> [f32; 8] {
    let n = 8.0f32;
    let mut out = [0f32; 8];
    for j in 0..8 {
        let mut sum = 0f32;
        for k in 0..8 {
            let c_k = if k == 0 { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() };
            sum += c_k * x[k] * ((std::f32::consts::PI * k as f32 * (2.0 * j as f32 + 1.0)) / (2.0 * n)).cos();
        }
        out[j] = sum;
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_quant_table(base: &[f32; 64], quality: u8) -> [f32; 64] {
    // JPEG quality factor mapping: quality 50 → scale 1.0 (baseline table).
    // quality 100 → scale 0.0 → all steps clamped to minimum 1.
    // quality  1  → scale 50.0 → maximum compression.
    let q = quality.clamp(1, 100) as f32;
    let scale = if q < 50.0 { 50.0 / q } else { (100.0 - q) / 50.0 };
    let mut table = [0f32; 64];
    for (i, &b) in base.iter().enumerate() {
        table[i] = (b * scale).clamp(1.0, 255.0);
    }
    table
}

fn extract_block(channel: &[f32], w: usize, h: usize, bx: usize, by: usize) -> [f32; 64] {
    let mut block = [0f32; 64];
    for dy in 0..BLOCK {
        for dx in 0..BLOCK {
            let px = (bx + dx).min(w - 1);
            let py = (by + dy).min(h - 1);
            block[dy * BLOCK + dx] = channel[py * w + px];
        }
    }
    block
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saliency::compute_saliency;

    fn make_gradient_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.push((x * 255 / w) as u8);
                rgb.push((y * 255 / h) as u8);
                rgb.push(128u8);
            }
        }
        rgb
    }

    #[test]
    fn dct_idct_roundtrip_is_lossless() {
        let block: [f32; 64] = std::array::from_fn(|i| i as f32 - 32.0);
        let dct   = dct2d(&block);
        let back  = idct2d(&dct);
        for (a, b) in block.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.5, "IDCT(DCT(x)) should recover x, got {a} vs {b}");
        }
    }

    #[test]
    fn encode_decode_colour_preserved() {
        let (w, h) = (32u32, 32u32);
        let rgb = make_gradient_rgb(w, h);
        let grey: Vec<u8> = rgb.chunks(3).map(|p| p[0] / 3 + p[1] / 3 + p[2] / 3).collect();
        let sal = compute_saliency(&grey, w, h, 0.3);

        let encoded = encode_dct(&rgb, w, h, 90, &sal);
        let (decoded, dw, dh) = decode_dct(&encoded).expect("decode failed");

        assert_eq!(dw, w);
        assert_eq!(dh, h);

        // With quality=90 and correct saliency-weighted dequantisation, RMSE should be small.
        let mut total_err = 0f64;
        for (a, b) in rgb.iter().zip(decoded.iter()) {
            total_err += (*a as f64 - *b as f64).powi(2);
        }
        let rmse = (total_err / rgb.len() as f64).sqrt();
        assert!(rmse < 15.0, "RMSE too high at quality=90: {rmse:.2}");
    }

    #[test]
    fn high_quality_smaller_error_than_low() {
        let (w, h) = (16u32, 16u32);
        let rgb = make_gradient_rgb(w, h);
        let grey: Vec<u8> = rgb.chunks(3).map(|p| p[0] / 3 + p[1] / 3 + p[2] / 3).collect();
        let sal = compute_saliency(&grey, w, h, 0.3);

        let enc_hi = encode_dct(&rgb, w, h, 95, &sal);
        let enc_lo = encode_dct(&rgb, w, h, 30, &sal);

        let mse = |enc: &[u8]| -> f64 {
            let (dec, _, _) = decode_dct(enc).unwrap();
            rgb.iter().zip(dec.iter()).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / rgb.len() as f64
        };
        assert!(mse(&enc_hi) < mse(&enc_lo), "high quality should have lower error");
    }

    /// Regression test for the "grey wash" bug.
    ///
    /// A pure-black image must decode to (near-)black regardless of saliency,
    /// not to a mid-grey as happened when the dequantisation step did not
    /// include the saliency scale that was used at encode time.
    #[test]
    fn pure_black_decodes_to_black() {
        let (w, h) = (16u32, 16u32);
        let black_rgb = vec![0u8; (w * h * 3) as usize];
        // Use a flat (all-zero) saliency map → sal_scale will be 3 for all tiles,
        // which was the worst-case trigger for the grey-wash bug.
        let grey = vec![0u8; (w * h) as usize];
        let sal = compute_saliency(&grey, w, h, 0.3);

        let encoded = encode_dct(&black_rgb, w, h, 85, &sal);
        let (decoded, _, _) = decode_dct(&encoded).expect("decode failed");

        // All decoded pixels must be close to black (≤10 per channel).
        for (i, &v) in decoded.iter().enumerate() {
            assert!(
                v <= 10,
                "pixel[{}] = {v} — expected near-black, got grey (grey-wash regression)",
                i
            );
        }
    }
}
