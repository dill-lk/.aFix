//! Cubic B-Spline contour fitting for the S1 Geometric Skeleton (`VEC_` chunk).
//!
//! After Canny edge detection produces a binary edge map, this module:
//! 1. Traces connected 8-connected contours from the edge pixels.
//! 2. Fits a degree-3 (cubic) uniform B-Spline to each contour using
//!    chord-length parameterisation and least-squares minimisation.
//! 3. Quantises control points to 16-bit fixed-point coordinates.
//! 4. Serialises the result into the `VEC_` binary format.
//!
//! ## `VEC_` chunk binary layout
//!
//! ```text
//! [spline_count: u32 LE]
//! For each spline:
//!   [degree: u8]            = 3 (cubic)
//!   [ctrl_count: u16 LE]    number of control points
//!   [x0: u16 LE][y0: u16 LE]  control points (delta-coded from previous)
//!   ...
//! ```

use crate::canny::EdgeMap;

// ── Public types ──────────────────────────────────────────────────────────────

/// A fitted cubic B-Spline described by its control points.
#[derive(Debug, Clone)]
pub struct BSpline {
    /// Control points in logical pixel coordinates.
    pub control_points: Vec<(f32, f32)>,
}

impl BSpline {
    /// Evaluate the B-Spline at parameter `t ∈ [0, 1]`.
    pub fn evaluate(&self, t: f32) -> (f32, f32) {
        let n = self.control_points.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        if n == 1 {
            return self.control_points[0];
        }

        // Map t to [0, n-3] for a uniform open cubic B-Spline.
        let segments = (n as f32 - 3.0).max(1.0);
        let u = t * segments;
        let seg = (u.floor() as usize).min(n - 4);
        let s = u - seg as f32;

        // De Boor basis for cubic uniform B-Spline (degree 3).
        let b0 = (1.0 - s).powi(3) / 6.0;
        let b1 = (3.0 * s.powi(3) - 6.0 * s.powi(2) + 4.0) / 6.0;
        let b2 = (-3.0 * s.powi(3) + 3.0 * s.powi(2) + 3.0 * s + 1.0) / 6.0;
        let b3 = s.powi(3) / 6.0;

        let p: [&(f32, f32); 4] = [
            &self.control_points[seg],
            &self.control_points[seg + 1],
            &self.control_points[seg + 2],
            &self.control_points[(seg + 3).min(n - 1)],
        ];

        (
            b0 * p[0].0 + b1 * p[1].0 + b2 * p[2].0 + b3 * p[3].0,
            b0 * p[0].1 + b1 * p[1].1 + b2 * p[2].1 + b3 * p[3].1,
        )
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Extract contours from `edge_map` and fit a cubic B-Spline to each.
///
/// `min_contour_len` filters out tiny noise contours.
pub fn fit_splines(edge_map: &EdgeMap, min_contour_len: usize) -> Vec<BSpline> {
    let contours = trace_contours(edge_map, min_contour_len);
    contours.iter().map(|c| fit_bspline_to_contour(c)).collect()
}

/// Serialise a slice of `BSpline`s into the `VEC_` chunk binary format.
pub fn serialise_splines(splines: &[BSpline]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(splines.len() as u32).to_le_bytes());

    for spline in splines {
        out.push(3u8); // degree = 3 (cubic)
        let n = spline.control_points.len() as u16;
        out.extend_from_slice(&n.to_le_bytes());

        // Delta-code the control points (reduces entropy).
        let mut prev_x = 0i32;
        let mut prev_y = 0i32;
        for &(px, py) in &spline.control_points {
            // Quantise to 16-bit fixed-point (1 unit = 1 pixel / 4 for sub-pixel accuracy).
            let qx = (px * 4.0).round() as i32;
            let qy = (py * 4.0).round() as i32;
            let dx = (qx - prev_x).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let dy = (qy - prev_y).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            out.extend_from_slice(&dx.to_le_bytes());
            out.extend_from_slice(&dy.to_le_bytes());
            prev_x = qx;
            prev_y = qy;
        }
    }
    out
}

/// Deserialise the `VEC_` chunk binary back into `BSpline`s (for decoder use).
pub fn deserialise_splines(data: &[u8]) -> Option<Vec<BSpline>> {
    if data.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(data[..4].try_into().ok()?) as usize;
    let mut pos = 4;
    let mut splines = Vec::with_capacity(count);

    for _ in 0..count {
        if pos + 3 > data.len() {
            return None;
        }
        let _degree = data[pos]; // must be 3
        pos += 1;
        let ctrl_count = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
        pos += 2;

        if pos + ctrl_count * 4 > data.len() {
            return None;
        }

        let mut pts = Vec::with_capacity(ctrl_count);
        let mut cur_x = 0i32;
        let mut cur_y = 0i32;
        for _ in 0..ctrl_count {
            let dx = i16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as i32;
            let dy = i16::from_le_bytes(data[pos + 2..pos + 4].try_into().ok()?) as i32;
            pos += 4;
            cur_x += dx;
            cur_y += dy;
            pts.push((cur_x as f32 / 4.0, cur_y as f32 / 4.0));
        }
        splines.push(BSpline { control_points: pts });
    }
    Some(splines)
}

// ── Contour tracing ───────────────────────────────────────────────────────────

/// Trace 8-connected contours from a binary edge map.
fn trace_contours(edge: &EdgeMap, min_len: usize) -> Vec<Vec<(f32, f32)>> {
    let w = edge.width as usize;
    let h = edge.height as usize;
    let mut visited = vec![false; w * h];
    let mut contours = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            let idx = sy * w + sx;
            if !edge.pixels[idx] || visited[idx] {
                continue;
            }
            // BFS flood fill to collect the contour.
            // Ordered contour following: walk 8-connected neighbours in a
            // clockwise preference order so the resulting point sequence follows
            // the actual edge rather than a BFS flood-fill order.
            let mut contour = Vec::new();
            contour.push((sx as f32, sy as f32));
            let mut cx = sx;
            let mut cy = sy;
            visited[sy * w + sx] = true;

            // 8-connected neighbour offsets in clockwise order starting East.
            const DIRS: [(i32, i32); 8] = [
                (1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1),
            ];

            'walk: loop {
                let mut found = false;
                for &(dx, dy) in &DIRS {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if edge.pixels[ni] && !visited[ni] {
                        visited[ni] = true;
                        cx = nx as usize;
                        cy = ny as usize;
                        contour.push((cx as f32, cy as f32));
                        found = true;
                        break;
                    }
                }
                if !found {
                    break 'walk;
                }
            }

            if contour.len() >= min_len {
                contours.push(contour);
            }
        }
    }
    contours
}

// ── Least-squares B-Spline fitting ───────────────────────────────────────────

/// Fit a cubic B-Spline to an ordered set of 2D points using chord-length
/// parameterisation and a fixed number of control points.
fn fit_bspline_to_contour(pts: &[(f32, f32)]) -> BSpline {
    let n = pts.len();
    if n < 4 {
        // Too few points — return a trivial spline at the centroid.
        let cx: f32 = pts.iter().map(|p| p.0).sum::<f32>() / n as f32;
        let cy: f32 = pts.iter().map(|p| p.1).sum::<f32>() / n as f32;
        return BSpline { control_points: vec![(cx, cy); 4] };
    }

    // Number of control points: ≥4, at most 1 per 3 input points.
    let num_ctrl = (n / 3).clamp(4, 64);

    // ── Chord-length parameterisation ────────────────────────────────────────
    let mut chord = vec![0f32; n];
    for i in 1..n {
        let dx = pts[i].0 - pts[i - 1].0;
        let dy = pts[i].1 - pts[i - 1].1;
        chord[i] = chord[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let total = chord[n - 1].max(1e-6);
    let t_vec: Vec<f32> = chord.iter().map(|&c| c / total).collect();

    // ── Uniform open knot vector for cubic B-Spline ──────────────────────────
    // k+1 = 4, so knot vector has num_ctrl + 4 entries.
    let knots = open_uniform_knots(num_ctrl, 3);

    // ── Least-squares fit: build matrix N (n × num_ctrl) ────────────────────
    // N[i][j] = B_{j,3}(t_i)
    let mut n_mat = vec![0f32; n * num_ctrl];
    for (i, &t) in t_vec.iter().enumerate() {
        for j in 0..num_ctrl {
            n_mat[i * num_ctrl + j] = bspline_basis(j, 3, t, &knots);
        }
    }

    // Solve N^T N P = N^T Q  (normal equations) for both x and y.
    let ctrl_x = solve_normal_equations(&n_mat, &pts.iter().map(|p| p.0).collect::<Vec<_>>(), n, num_ctrl);
    let ctrl_y = solve_normal_equations(&n_mat, &pts.iter().map(|p| p.1).collect::<Vec<_>>(), n, num_ctrl);

    let control_points: Vec<(f32, f32)> =
        ctrl_x.into_iter().zip(ctrl_y).collect();

    BSpline { control_points }
}

/// Compute a Cox-de Boor recursive B-Spline basis value B_{i,p}(t).
fn bspline_basis(i: usize, p: usize, t: f32, knots: &[f32]) -> f32 {
    if p == 0 {
        let in_span = knots[i] <= t && t < knots[i + 1];
        // At the very end of the domain include the last knot.
        let at_end = (t - 1.0).abs() < 1e-8 && (knots[i + 1] - 1.0).abs() < 1e-8;
        return if in_span || at_end { 1.0 } else { 0.0 };
    }
    let d1 = knots[i + p] - knots[i];
    let d2 = knots[i + p + 1] - knots[i + 1];
    let c1 = if d1 > 1e-8 { (t - knots[i]) / d1 } else { 0.0 };
    let c2 = if d2 > 1e-8 { (knots[i + p + 1] - t) / d2 } else { 0.0 };
    c1 * bspline_basis(i, p - 1, t, knots) + c2 * bspline_basis(i + 1, p - 1, t, knots)
}

/// Build an open uniform knot vector for degree `p` with `n` control points.
fn open_uniform_knots(n: usize, p: usize) -> Vec<f32> {
    let m = n + p + 1; // total knot count
    let mut knots = vec![0f32; m];
    // First `p+1` knots = 0, last `p+1` = 1, internal knots uniform.
    for i in p + 1..n {
        knots[i] = (i - p) as f32 / (n - p) as f32;
    }
    for i in n..m {
        knots[i] = 1.0;
    }
    knots
}

/// Solve the normal equations N^T N x = N^T b using Gaussian elimination.
/// Returns a vector of `num_ctrl` control values.
fn solve_normal_equations(n_mat: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Build A = N^T N  (cols × cols) and rhs = N^T b (cols × 1).
    let mut a = vec![0f32; cols * cols];
    let mut rhs = vec![0f32; cols];

    for j in 0..cols {
        for k in 0..cols {
            for i in 0..rows {
                a[j * cols + k] += n_mat[i * cols + j] * n_mat[i * cols + k];
            }
        }
        for i in 0..rows {
            rhs[j] += n_mat[i * cols + j] * b[i];
        }
    }

    gaussian_elimination(&mut a, &mut rhs, cols)
}

/// Gaussian elimination with partial pivoting.
fn gaussian_elimination(a: &mut [f32], b: &mut [f32], n: usize) -> Vec<f32> {
    for col in 0..n {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

        let pivot = a[col * n + col];
        if pivot.abs() < 1e-10 {
            continue; // singular or near-singular — leave as-is
        }

        for row in col + 1..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let v = a[col * n + k] * factor;
                a[row * n + k] -= v;
            }
            let v = b[col] * factor;
            b[row] -= v;
        }
    }

    // Back substitution.
    let mut x = vec![0f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n {
            sum -= a[i * n + j] * x[j];
        }
        let denom = a[i * n + i];
        x[i] = if denom.abs() > 1e-10 { sum / denom } else { 0.0 };
    }
    x
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canny::EdgeMap;

    fn make_circle_edge_map(cx: u32, cy: u32, r: u32, w: u32, h: u32) -> EdgeMap {
        let n = (w * h) as usize;
        let mut pixels = vec![false; n];
        for angle_deg in 0..360u32 {
            let a = angle_deg as f32 * std::f32::consts::PI / 180.0;
            let x = (cx as f32 + r as f32 * a.cos()).round() as i32;
            let y = (cy as f32 + r as f32 * a.sin()).round() as i32;
            if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                pixels[y as usize * w as usize + x as usize] = true;
            }
        }
        EdgeMap { width: w, height: h, pixels }
    }

    #[test]
    fn fits_circle_to_reasonable_bounds() {
        let (w, h, r) = (128u32, 128u32, 40u32);
        let edge_map = make_circle_edge_map(64, 64, r, w, h);
        let splines = fit_splines(&edge_map, 20);
        assert!(!splines.is_empty(), "should fit at least one spline");
        // All control points must lie within image bounds (small margin allowed).
        for sp in &splines {
            for &(px, py) in &sp.control_points {
                assert!(
                    px >= -10.0 && px <= w as f32 + 10.0
                        && py >= -10.0 && py <= h as f32 + 10.0,
                    "control point ({px:.1},{py:.1}) outside image bounds"
                );
            }
        }
    }

    #[test]
    fn serialise_deserialise_roundtrip() {
        let spline = BSpline {
            control_points: vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0)],
        };
        let data = serialise_splines(&[spline.clone()]);
        let recovered = deserialise_splines(&data).expect("deserialise failed");
        assert_eq!(recovered.len(), 1);
        let eps = 0.5;
        for (a, b) in spline.control_points.iter().zip(recovered[0].control_points.iter()) {
            assert!((a.0 - b.0).abs() < eps, "x mismatch: {} vs {}", a.0, b.0);
            assert!((a.1 - b.1).abs() < eps, "y mismatch: {} vs {}", a.1, b.1);
        }
    }

    #[test]
    fn bspline_evaluate_stays_in_convex_hull() {
        // A uniform open B-Spline lies within the convex hull of its control points.
        let sp = BSpline {
            control_points: vec![(0.0, 0.0), (1.0, 3.0), (2.0, 3.0), (3.0, 0.0)],
        };
        for t_int in 0..=10 {
            let (x, y) = sp.evaluate(t_int as f32 / 10.0);
            assert!(x >= -0.1 && x <= 3.1, "x={x:.3} outside convex hull x∈[0,3]");
            assert!(y >= -0.1 && y <= 3.1, "y={y:.3} outside convex hull y∈[0,3]");
        }
    }
}
