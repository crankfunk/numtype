//! Sum-reduce: full (mirrors the `axis === undefined` branch of `sumRuntime`)
//! and per-axis (mirrors the rest of `sumRuntime`).

use crate::shape::{checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult, KernelError};

/// Sum every element, in data order. Mirrors `sumRuntime`'s `axis ===
/// undefined` branch: `for (i=0;i<data.length;i++) total += data[i]`.
/// Infallible — a flat data slice has no shape to be invalid.
pub fn sum_all(data: &[f64]) -> f64 {
    let mut total = 0.0f64;
    for &v in data {
        total += v;
    }
    total
}

/// Sum-reduce along one axis (negative axes count from the end). Mirrors
/// `sumRuntime`'s axis branch exactly (same loop order/accumulation).
pub fn sum_axis(shape: &[u32], data: &[f64], axis: i32) -> KResult<(Vec<u32>, Vec<f64>)> {
    checked_element_count(shape)?;
    let rank = shape.len() as i32;
    let norm_axis = if axis < 0 { rank + axis } else { axis };
    if norm_axis < 0 || norm_axis >= rank {
        return Err(KernelError::ShapeIncompatible);
    }
    let norm_axis = norm_axis as usize;

    let mut out_shape: Vec<u32> = Vec::with_capacity(shape.len() - 1);
    out_shape.extend_from_slice(&shape[..norm_axis]);
    out_shape.extend_from_slice(&shape[norm_axis + 1..]);

    let strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let out_size = checked_element_count(&out_shape)?;
    let axis_dim = shape[norm_axis];
    let axis_stride = strides[norm_axis];

    let mut out = vec![0f64; out_size as usize];
    for out_flat in 0..out_size {
        let idx = unravel(out_flat, &out_shape, &out_strides);
        let mut base_offset: u32 = 0;
        let mut out_axis = 0usize;
        for in_axis in 0..shape.len() {
            if in_axis == norm_axis {
                continue;
            }
            base_offset += idx[out_axis] * strides[in_axis];
            out_axis += 1;
        }
        let mut total = 0.0f64;
        for a in 0..axis_dim {
            let off = base_offset + a * axis_stride;
            total += data.get(off as usize).copied().unwrap_or(0.0);
        }
        out[out_flat as usize] = total;
    }
    Ok((out_shape, out))
}

/// Kern 03: strided full sum. **The determinism trap this phase exists to
/// get right:** accumulation runs in the view's *logical row-major order*
/// (flat 0..size, unravel, strided offset), NOT in memory order — float
/// addition is order-sensitive, and on a transposed view the two orders
/// differ. For contiguous metadata the strided offset equals the flat index,
/// so this reproduces [`sum_all`]'s `data[0], data[1], …` order (and
/// therefore its bits) exactly. Fallible, unlike [`sum_all`]: shape and
/// caller strides must validate first.
pub fn sum_all_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64]) -> KResult<f64> {
    let size = checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;

    let logical_strides = compute_strides(shape);
    let mut total = 0.0f64;
    for flat in 0..size {
        let idx = unravel(flat, shape, &logical_strides);
        let mut off: u32 = offset;
        for i in 0..shape.len() {
            off += idx[i] * strides[i];
        }
        total += data.get(off as usize).copied().unwrap_or(0.0);
    }
    Ok(total)
}

/// Kern 03: strided generalization of [`sum_axis`] — same loop order and
/// accumulation order, with `compute_strides(shape)` replaced by caller
/// strides and the base offset added. The contiguous original stays
/// untouched (frozen v1 baseline).
pub fn sum_axis_strided(
    shape: &[u32],
    strides: &[u32],
    offset: u32,
    data: &[f64],
    axis: i32,
) -> KResult<(Vec<u32>, Vec<f64>)> {
    checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;
    let rank = shape.len() as i32;
    let norm_axis = if axis < 0 { rank + axis } else { axis };
    if norm_axis < 0 || norm_axis >= rank {
        return Err(KernelError::ShapeIncompatible);
    }
    let norm_axis = norm_axis as usize;

    let mut out_shape: Vec<u32> = Vec::with_capacity(shape.len() - 1);
    out_shape.extend_from_slice(&shape[..norm_axis]);
    out_shape.extend_from_slice(&shape[norm_axis + 1..]);

    let out_strides = compute_strides(&out_shape);
    let out_size = checked_element_count(&out_shape)?;
    let axis_dim = shape[norm_axis];
    let axis_stride = strides[norm_axis];

    let mut out = vec![0f64; out_size as usize];
    for out_flat in 0..out_size {
        let idx = unravel(out_flat, &out_shape, &out_strides);
        let mut base_offset: u32 = offset;
        let mut out_axis = 0usize;
        for in_axis in 0..shape.len() {
            if in_axis == norm_axis {
                continue;
            }
            base_offset += idx[out_axis] * strides[in_axis];
            out_axis += 1;
        }
        let mut total = 0.0f64;
        for a in 0..axis_dim {
            let off = base_offset + a * axis_stride;
            total += data.get(off as usize).copied().unwrap_or(0.0);
        }
        out[out_flat as usize] = total;
    }
    Ok((out_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_all_basic() {
        assert_eq!(sum_all(&[1.0, 2.0, 3.0, 4.0]), 10.0);
        assert_eq!(sum_all(&[]), 0.0);
    }

    #[test]
    fn sum_axis_cube() {
        // cube [2,3,4] filled 1..24, sum axis 1 -> [2,4], matches demo.ts's
        // showcase example structurally (same shape/axis).
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let (shape, out) = sum_axis(&[2, 3, 4], &data, 1).unwrap();
        assert_eq!(shape, vec![2, 4]);
        // cube[0][.][.] rows: [1..4],[5..8],[9..12] -> col sums [15,18,21,24]
        assert_eq!(&out[0..4], &[15.0, 18.0, 21.0, 24.0]);
        // cube[1][.][.] rows: [13..16],[17..20],[21..24] -> [51,54,57,60]
        assert_eq!(&out[4..8], &[51.0, 54.0, 57.0, 60.0]);
    }

    #[test]
    fn sum_axis_negative() {
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let (shape_pos, out_pos) = sum_axis(&[2, 3, 4], &data, 1).unwrap();
        let (shape_neg, out_neg) = sum_axis(&[2, 3, 4], &data, -2).unwrap();
        assert_eq!(shape_pos, shape_neg);
        assert_eq!(out_pos, out_neg);
    }

    #[test]
    fn sum_axis_out_of_range() {
        let err = sum_axis(&[2, 3], &[0.0; 6], 5).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
        let err_scalar = sum_axis(&[], &[1.0], 0).unwrap_err();
        assert_eq!(err_scalar, KernelError::ShapeIncompatible);
    }

    #[test]
    fn sum_axis_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = sum_axis(&shape, &[0.0], 0).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn sum_axis_size_zero() {
        let (shape, out) = sum_axis(&[0, 3], &[], 0).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    // --- Kern 03: sum_all_strided / sum_axis_strided ------------------------

    #[test]
    fn sum_all_strided_contiguous_matches_sum_all() {
        let data: Vec<f64> = (0..24).map(|i| (i as f64) * 0.1 + 0.01).collect();
        let plain = sum_all(&data);
        let strided = sum_all_strided(&[2, 3, 4], &[12, 4, 1], 0, &data).unwrap();
        assert_eq!(plain.to_bits(), strided.to_bits());
    }

    /// The order-sensitivity guard: on a transposed view, logical-order
    /// accumulation must equal summing the *materialized* transpose in flat
    /// order — bit-for-bit — and (for these deliberately mixed-magnitude
    /// values) genuinely differs from memory-order accumulation.
    #[test]
    fn sum_all_strided_transposed_view_uses_logical_order() {
        // Absorption pattern: memory order hits 1e100 + 1.0 (the 1.0 is
        // absorbed) before cancelling; logical order cancels 1e100 - 1e100
        // FIRST, then adds 1.0. Memory-order total: 0.0; logical: 1.0.
        // Memory order is d0..d5; logical view order is d0,d3,d1,d4,d2,d5.
        let data: Vec<f64> = vec![1e100, 1.0, 0.0, -1e100, 0.0, 0.0];
        // View [3,2] with strides [1,3] = transpose of contiguous [2,3].
        // Materialize that transpose by hand, sum flat (the reference order).
        let mut materialized = vec![0f64; 6];
        for i in 0..3 {
            for j in 0..2 {
                materialized[i * 2 + j] = data[i + 3 * j];
            }
        }
        let reference = sum_all(&materialized);
        let strided = sum_all_strided(&[3, 2], &[1, 3], 0, &data).unwrap();
        assert_eq!(reference.to_bits(), strided.to_bits());
        // Non-vacuity: memory-order summation gives DIFFERENT bits here, so
        // the test above really pins the order, not just the value.
        assert_ne!(sum_all(&data).to_bits(), strided.to_bits());
    }

    #[test]
    fn sum_all_strided_offset_and_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(sum_all_strided(&[2], &[1], 2, &data).unwrap(), 7.0);
        let err = sum_all_strided(&[2], &[1], 3, &data).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
        // Rank-0 scalar view: reads exactly data[offset].
        assert_eq!(sum_all_strided(&[], &[], 1, &data).unwrap(), 2.0);
        // Size-0 view sums to 0.0 without reading.
        assert_eq!(sum_all_strided(&[0, 3], &[3, 1], 0, &[]).unwrap(), 0.0);
    }

    #[test]
    fn sum_axis_strided_contiguous_matches_sum_axis() {
        let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let (shape_c, out_c) = sum_axis(&[2, 3, 4], &data, 1).unwrap();
        let (shape_s, out_s) = sum_axis_strided(&[2, 3, 4], &[12, 4, 1], 0, &data, 1).unwrap();
        assert_eq!(shape_c, shape_s);
        assert_eq!(out_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   out_s.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    /// Summing a transposed view along axis 0 == summing the base along
    /// axis 1 (values; the per-element accumulation order is identical too:
    /// both walk that axis in increasing index order).
    #[test]
    fn sum_axis_strided_transposed_view() {
        let data: Vec<f64> = (1..=6).map(|i| i as f64).collect(); // [[1,2,3],[4,5,6]]
        let (shape_t, out_t) = sum_axis_strided(&[3, 2], &[1, 3], 0, &data, 0).unwrap();
        assert_eq!(shape_t, vec![2]);
        // A^T summed over its rows = column sums of A^T = row sums... check:
        // A^T = [[1,4],[2,5],[3,6]]; axis 0 -> [1+2+3, 4+5+6] = [6, 15].
        assert_eq!(out_t, vec![6.0, 15.0]);
        let (_, out_base) = sum_axis(&[2, 3], &data, 1).unwrap();
        assert_eq!(out_base, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_axis_strided_errors() {
        let data = vec![0.0; 6];
        let err = sum_axis_strided(&[2, 3], &[3, 1], 0, &data, 5).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
        let err_oob = sum_axis_strided(&[2, 3], &[4, 1], 0, &data, 0).unwrap_err();
        assert_eq!(err_oob, KernelError::StridesOutOfBounds);
    }
}
