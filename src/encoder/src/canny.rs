//! Full Canny edge detector for S1 Geometric Skeleton generation.
//!
//! Pipeline:
//! 1. Gaussian blur (σ = 1.4, 5×5 kernel)
//! 2. Sobel gradient magnitude & direction
//! 3. Non-maximum suppression
//! 4. Double threshold (high / low)
//! 5. Edge tracking by hysteresis

/// Output of the Canny detector: a flat `width × height` bitmask where
/// `true` means a confirmed strong edge pixel.
pub struct EdgeMap {
    pub width: u32,
    pub height: u32,
    /// Row-major. `pixels[y * width + x]` is `true` for an edge pixel.
    pub pixels: Vec<bool>,
}

impl EdgeMap {
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Run the full Canny pipeline on a greyscale `width × height` buffer.
///
/// `pixels` must be row-major `u8` values (one byte per pixel).
/// `high_thresh` and `low_thresh` are gradient-magnitude thresholds in the
/// same units as the Sobel output (0–255 scale × √2 ≈ 0–360).
pub fn canny(
    pixels: &[u8],
    width: u32,
    height: u32,
    low_thresh: f32,
    high_thresh: f32,
) -> EdgeMap {
    let n = (width * height) as usize;
    let w = width as usize;
    let h = height as usize;

    // ── 1. Gaussian blur ──────────────────────────────────────────────────────
    let blurred = gaussian_blur(pixels, w, h);

    // ── 2. Sobel gradients ────────────────────────────────────────────────────
    let mut mag = vec![0f32; n];
    let mut dir = vec![0u8; n]; // quantised direction: 0=H, 1=45°, 2=V, 3=135°

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = sobel_gx(&blurred, x, y, w);
            let gy = sobel_gy(&blurred, x, y, w);
            mag[y * w + x] = (gx * gx + gy * gy).sqrt();

            // Quantise angle to 4 directions (0, 45, 90, 135 degrees).
            let angle = gy.atan2(gx).to_degrees();
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };
            dir[y * w + x] = if angle < 22.5 || angle >= 157.5 {
                0
            } else if angle < 67.5 {
                1
            } else if angle < 112.5 {
                2
            } else {
                3
            };
        }
    }

    // ── 3. Non-maximum suppression ────────────────────────────────────────────
    let mut nms = vec![0f32; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let m = mag[y * w + x];
            let (n1, n2) = match dir[y * w + x] {
                0 => (mag[y * w + x - 1], mag[y * w + x + 1]),            // horizontal
                1 => (mag[(y - 1) * w + x + 1], mag[(y + 1) * w + x - 1]), // 45°
                2 => (mag[(y - 1) * w + x], mag[(y + 1) * w + x]),        // vertical
                _ => (mag[(y - 1) * w + x - 1], mag[(y + 1) * w + x + 1]), // 135°
            };
            if m >= n1 && m >= n2 {
                nms[y * w + x] = m;
            }
        }
    }

    // ── 4 + 5. Double threshold + hysteresis ─────────────────────────────────
    let mut edge = vec![EdgeState::None; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let v = nms[y * w + x];
            if v >= high_thresh {
                edge[y * w + x] = EdgeState::Strong;
            } else if v >= low_thresh {
                edge[y * w + x] = EdgeState::Weak;
            }
        }
    }

    // Propagate: weak pixels connected (8-connected) to a strong pixel become strong.
    // Use an iterative flood from strong pixels.
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if edge[y * w + x] == EdgeState::Weak && has_strong_neighbour(&edge, x, y, w) {
                    edge[y * w + x] = EdgeState::Strong;
                    changed = true;
                }
            }
        }
    }

    let pixels_out: Vec<bool> = edge.iter().map(|e| *e == EdgeState::Strong).collect();
    EdgeMap { width, height, pixels: pixels_out }
}

// ── Internals ─────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum EdgeState {
    None,
    Weak,
    Strong,
}

/// 5×5 Gaussian blur with σ ≈ 1.4.
fn gaussian_blur(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    // Kernel (approximated, sum = 256): row-separable 5-tap [1, 4, 6, 4, 1].
    let kernel = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let ksum = 16.0f32;
    let mut tmp = vec![0f32; w * h];
    let mut out = vec![0f32; w * h];

    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            let mut wsum = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let xi = x as isize + ki as isize - 2;
                if xi >= 0 && xi < w as isize {
                    acc += src[y * w + xi as usize] as f32 * kv;
                    wsum += kv;
                }
            }
            tmp[y * w + x] = acc / wsum;
        }
    }

    // Vertical pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            let mut wsum = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let yi = y as isize + ki as isize - 2;
                if yi >= 0 && yi < h as isize {
                    acc += tmp[yi as usize * w + x] * kv;
                    wsum += kv;
                }
            }
            out[y * w + x] = acc / wsum;
        }
    }

    let _ = ksum; // suppress unused warning
    out
}

fn sobel_gx(src: &[f32], x: usize, y: usize, w: usize) -> f32 {
    let p = |dx: isize, dy: isize| src[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
    -p(-1, -1) + p(1, -1) - 2.0 * p(-1, 0) + 2.0 * p(1, 0) - p(-1, 1) + p(1, 1)
}

fn sobel_gy(src: &[f32], x: usize, y: usize, w: usize) -> f32 {
    let p = |dx: isize, dy: isize| src[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
    -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)
}

fn has_strong_neighbour(edge: &[EdgeState], x: usize, y: usize, w: usize) -> bool {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;
            if edge[ny * w + nx] == EdgeState::Strong {
                return true;
            }
        }
    }
    false
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_border_image(w: u32, h: u32) -> Vec<u8> {
        // White image with a black rectangle inside — produces clear edges.
        let mut img = vec![200u8; (w * h) as usize];
        let (bx, by, bw, bh) = (w / 4, h / 4, w / 2, h / 2);
        for y in by..by + bh {
            for x in bx..bx + bw {
                img[(y * w + x) as usize] = 20;
            }
        }
        img
    }

    #[test]
    fn detects_rectangle_edges() {
        let (w, h) = (64u32, 64u32);
        let img = make_border_image(w, h);
        let edges = canny(&img, w, h, 20.0, 60.0);
        let edge_count = edges.pixels.iter().filter(|&&e| e).count();
        // A 32×32 inner rectangle has ~4 edges of length 32 ≈ 128 edge pixels.
        assert!(edge_count > 50, "expected >50 edge pixels, got {edge_count}");
    }

    #[test]
    fn uniform_image_has_no_edges() {
        let (w, h) = (32u32, 32u32);
        let img = vec![128u8; (w * h) as usize];
        let edges = canny(&img, w, h, 20.0, 60.0);
        let edge_count = edges.pixels.iter().filter(|&&e| e).count();
        assert_eq!(edge_count, 0, "uniform image should have no edges");
    }
}
