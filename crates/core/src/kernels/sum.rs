//! Sum-reduce: full (mirrors the `axis === undefined` branch of `sumRuntime`)
//! and per-axis (mirrors the rest of `sumRuntime`).

use crate::shape::{checked_element_count, compute_strides, unravel, KResult, KernelError};

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
}
