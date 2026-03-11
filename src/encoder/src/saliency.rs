//! Gradient-based saliency map for non-linear quantisation (W_s).
//!
//! Generates a per-pixel saliency weight in `[0, 1]` where values close to
//! **1.0** mark the foreground subject (high detail preserved) and values close
//! to **0.0** mark background (aggressively compressed).
//!
//! The formula from the spec:
//! ```text
//! C = ∮ (S_v · W_s) + (T_n · W_p)
//! ```
//! This module produces `W_s`.
//!
//! ## Algorithm
//!
//! 1. Compute the Sobel gradient magnitude in YCbCr luminance.
//! 2. Apply a multi-scale approach: average gradient at 1× and 2× scale.
//! 3. Add a centre-bias (Gaussian distance from image centre).
//! 4. Normalise to `[0, 1]`.

/// Per-pixel saliency weight map.
pub struct SaliencyMap {
    pub width: u32,
    pub height: u32,
    /// Row-major `f32` values in `[0, 1]`. `1.0` = max salience (subject).
    pub weights: Vec<f32>,
}

impl SaliencyMap {
    /// Look up the saliency weight at pixel `(x, y)`.
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.weights[(y * self.width + x) as usize]
    }

    /// Average saliency over a rectangular region (used for bounding-box scoring).
    pub fn region_mean(&self, x: u32, y: u32, rw: u32, rh: u32) -> f32 {
        let x1 = x.min(self.width);
        let y1 = y.min(self.height);
        let x2 = (x + rw).min(self.width);
        let y2 = (y + rh).min(self.height);
        if x1 >= x2 || y1 >= y2 {
            return 0.0;
        }
        let mut sum = 0f32;
        let mut count = 0u32;
        for py in y1..y2 {
            for px in x1..x2 {
                sum += self.get(px, py);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }
}

/// Compute a saliency map from a greyscale `u8` buffer (row-major).
///
/// `centre_bias` in `[0, 1]` blends a Gaussian distance-from-centre term
/// into the gradient saliency.  Use `0.3` as a good default.
pub fn compute_saliency(pixels: &[u8], width: u32, height: u32, centre_bias: f32) -> SaliencyMap {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    // ── Multi-scale gradient magnitude ────────────────────────────────────────
    let grad1 = gradient_magnitude(pixels, w, h);
    let downsampled = downsample2(pixels, w, h);
    let grad2_small = gradient_magnitude(&downsampled, w / 2, h / 2);
    let grad2 = upsample2(&grad2_small, w / 2, h / 2, w, h);

    // Blend scales.
    let mut sal: Vec<f32> = (0..n).map(|i| 0.6 * grad1[i] + 0.4 * grad2[i]).collect();

    // ── Centre bias ───────────────────────────────────────────────────────────
    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            // Centre bias: pixels near the centre get a small lift.
            let bias = 1.0 - (dist / max_dist).powi(2);
            sal[y * w + x] = sal[y * w + x] * (1.0 - centre_bias) + bias * centre_bias;
        }
    }

    // ── Normalise ─────────────────────────────────────────────────────────────
    let max = sal.iter().cloned().fold(0f32, f32::max).max(1e-6);
    for v in sal.iter_mut() {
        *v /= max;
    }

    SaliencyMap { width, height, weights: sal }
}

// ── Internals ─────────────────────────────────────────────────────────────────

fn gradient_magnitude(pixels: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dx: isize, dy: isize| {
                pixels[(y as isize + dy) as usize * w + (x as isize + dx) as usize] as f32
            };
            let gx = -p(-1, -1) + p(1, -1) - 2.0 * p(-1, 0) + 2.0 * p(1, 0) - p(-1, 1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);
            mag[y * w + x] = (gx * gx + gy * gy).sqrt() / 360.0; // normalise to ~[0,1]
        }
    }
    mag
}

fn downsample2(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut dst = vec![0u8; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let sy = (y * 2).min(h - 1);
            let sx = (x * 2).min(w - 1);
            // 2×2 average.
            let mut sum = src[sy * w + sx] as u32;
            let mut cnt = 1u32;
            if sx + 1 < w { sum += src[sy * w + sx + 1] as u32; cnt += 1; }
            if sy + 1 < h { sum += src[(sy + 1) * w + sx] as u32; cnt += 1; }
            if sx + 1 < w && sy + 1 < h { sum += src[(sy + 1) * w + sx + 1] as u32; cnt += 1; }
            dst[y * nw + x] = (sum / cnt) as u8;
        }
    }
    dst
}

fn upsample2(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let sy = ((y * sh) / dh).min(sh - 1);
            let sx = ((x * sw) / dw).min(sw - 1);
            dst[y * dw + x] = src[sy * sw + sx];
        }
    }
    dst
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_image_returns_centre_bias_only() {
        let (w, h) = (32u32, 32u32);
        let pixels = vec![128u8; (w * h) as usize];
        let sal = compute_saliency(&pixels, w, h, 0.5);
        // No gradient → all variation from centre bias. Centre pixel > corner.
        let centre = sal.get(w / 2, h / 2);
        let corner = sal.get(0, 0);
        assert!(centre > corner, "centre ({centre}) should be more salient than corner ({corner})");
    }

    #[test]
    fn high_contrast_region_is_salient() {
        let (w, h) = (64u32, 64u32);
        let mut pixels = vec![128u8; (w * h) as usize];
        // Place a high-contrast pattern in the top-left quadrant.
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                pixels[(y * w + x) as usize] = if (x + y) % 4 < 2 { 0 } else { 255 };
            }
        }
        let sal = compute_saliency(&pixels, w, h, 0.0);
        let active_region = sal.region_mean(1, 1, w / 2 - 2, h / 2 - 2);
        let inactive_region = sal.region_mean(w / 2, h / 2, w / 2 - 1, h / 2 - 1);
        assert!(
            active_region > inactive_region,
            "high-contrast region ({active_region:.3}) should be more salient than uniform region ({inactive_region:.3})"
        );
    }

    #[test]
    fn saliency_values_in_unit_range() {
        let (w, h) = (16u32, 16u32);
        let pixels: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let sal = compute_saliency(&pixels, w, h, 0.3);
        for &v in &sal.weights {
            assert!((0.0..=1.0).contains(&v), "saliency {v} out of [0,1]");
        }
    }
}
