//! Broadcasting elementwise add. Mirrors `elementwiseBinary` in
//! `spike/src/runtime.ts` (specialized to `op = (x, y) => x + y`; the ABI
//! only needs `add`, so the generic `op` callback isn't ported).

use crate::shape::{
    align_to_rank, aligned_effective_strides, broadcast_shape, checked_element_count, compute_strides,
    effective_strides, unravel, validate_strided_bounds, KResult,
};

/// `a_shape`/`b_shape` and `a_data`/`b_data` are validated independently
/// before broadcasting (so an "outer product" broadcast blowup — see
/// `shape::tests::broadcast_shape_outer_product_blowup_detected_by_caller`
/// — is still caught via `checked_element_count` on the *output* shape).
pub fn add(a_shape: &[u32], a_data: &[f64], b_shape: &[u32], b_data: &[f64]) -> KResult<(Vec<u32>, Vec<f64>)> {
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;

    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let rank = out_shape.len();
    let size = checked_element_count(&out_shape)?;

    let a_strides = compute_strides(a_shape);
    let b_strides = compute_strides(b_shape);
    let (a_shape_al, a_strides_al) = align_to_rank(a_shape, &a_strides, rank);
    let (b_shape_al, b_strides_al) = align_to_rank(b_shape, &b_strides, rank);
    let a_eff = effective_strides(&a_shape_al, &a_strides_al);
    let b_eff = effective_strides(&b_shape_al, &b_strides_al);

    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0f64; size as usize];

    // Offset accumulation below is `u32` (mirroring the ABI's address
    // domain). Safe from wraparound: each term `idx[i] * eff[i]` is either
    // 0 (broadcast axis) or bounded by the *source* operand's own element
    // count (already validated to fit u32 above), and the well-formed
    // row-major strides guarantee the total never exceeds that bound
    // either — see shape.rs module docs.
    for flat in 0..size {
        let idx = unravel(flat, &out_shape, &out_strides);
        let mut a_off: u32 = 0;
        let mut b_off: u32 = 0;
        for i in 0..rank {
            let ix = idx[i];
            a_off += ix * a_eff[i];
            b_off += ix * b_eff[i];
        }
        let a_val = a_data.get(a_off as usize).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_off as usize).copied().unwrap_or(0.0);
        out[flat as usize] = a_val + b_val;
    }
    Ok((out_shape, out))
}

/// Kern 03: strided generalization of [`add`] — operands carry
/// caller-supplied strides and a base element offset instead of being
/// assumed contiguous. Same loop structure, same iteration and accumulation
/// order as [`add`] (which itself mirrors `elementwiseBinary` in
/// `runtime.ts`), so a contiguous-metadata call is bit-identical to the
/// contiguous kernel by construction. The contiguous original stays
/// untouched (frozen v1 baseline — see docs/kern-03-strided-spec.md).
#[allow(clippy::too_many_arguments)]
pub fn add_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;
    validate_strided_bounds(a_shape, a_strides, a_offset, a_data.len() as u32)?;
    validate_strided_bounds(b_shape, b_strides, b_offset, b_data.len() as u32)?;

    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let rank = out_shape.len();
    let size = checked_element_count(&out_shape)?;

    let a_eff = aligned_effective_strides(a_shape, a_strides, rank);
    let b_eff = aligned_effective_strides(b_shape, b_strides, rank);

    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0f64; size as usize];

    // Offset accumulation stays u32: every partial sum is bounded by the
    // operand's own validated max reach (validate_strided_bounds above), so
    // wraparound is structurally unreachable — same argument as in `add`,
    // with the validation now explicit because strides are caller input.
    for flat in 0..size {
        let idx = unravel(flat, &out_shape, &out_strides);
        let mut a_off: u32 = a_offset;
        let mut b_off: u32 = b_offset;
        for i in 0..rank {
            let ix = idx[i];
            a_off += ix * a_eff[i];
            b_off += ix * b_eff[i];
        }
        let a_val = a_data.get(a_off as usize).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_off as usize).copied().unwrap_or(0.0);
        out[flat as usize] = a_val + b_val;
    }
    Ok((out_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::KernelError;

    #[test]
    fn same_shape() {
        let (shape, data) = add(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &[2, 2], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn trailing_broadcast() {
        // [2,3] + [3] -> [2,3], matches spike/demo.ts's showcase example.
        let (shape, data) = add(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3], &[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn interior_one_broadcast() {
        // [3,1,5] + [3,4,5] -> [3,4,5]
        let a: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let (shape, data) = add(&[3, 1, 5], &a, &[3, 4, 5], &b).unwrap();
        assert_eq!(shape, vec![3, 4, 5]);
        // Spot-check one element by hand: out[i][j][k] = a[i][0][k] + b[i][j][k]
        // i=1,j=2,k=3 -> a_off = 1*5+3=8 (a[1][3]=8.0), b_off=1*20+2*5+3=33 (33.0)
        let out_idx = 1 * 20 + 2 * 5 + 3;
        assert_eq!(data[out_idx], 8.0 + 33.0);
    }

    #[test]
    fn rank0_scalar_broadcast() {
        let (shape, data) = add(&[], &[5.0], &[3], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(data, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn size_zero_array() {
        let (shape, data) = add(&[0, 3], &[], &[3], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn incompatible_shapes() {
        let err = add(&[2, 3], &[0.0; 6], &[2, 4], &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn rank_too_large() {
        let shape = vec![1u32; 33];
        let err = add(&shape, &[0.0], &[1], &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    // --- Kern 03: add_strided ---------------------------------------------

    /// Strided with contiguous metadata must be bit-identical to the
    /// contiguous kernel (the "always strided" TS routing relies on this).
    #[test]
    fn strided_contiguous_metadata_matches_contiguous_kernel() {
        let a: Vec<f64> = (0..6).map(|i| i as f64 * 1.25).collect();
        let b: Vec<f64> = (0..3).map(|i| i as f64 * -0.5).collect();
        let (shape_c, data_c) = add(&[2, 3], &a, &[3], &b).unwrap();
        let (shape_s, data_s) = add_strided(&[2, 3], &[3, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape_c, shape_s);
        assert_eq!(data_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   data_s.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    /// A transposed view of A ([2,3] buffer read as [3,2] with strides
    /// [1,3]) added to a contiguous [3,2] — checked against the hand-
    /// materialized transpose.
    #[test]
    fn strided_transposed_view_operand() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // [3,2] contiguous
        let (shape, data) = add_strided(&[3, 2], &[1, 3], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape, vec![3, 2]);
        // A^T = [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0]);
    }

    /// Nonzero base offset: a rank-1 window into the middle of a buffer.
    #[test]
    fn strided_offset_window() {
        let buf = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (shape, data) = add_strided(&[3], &[1], 2, &buf, &[3], &[1], 0, &buf).unwrap();
        assert_eq!(shape, vec![3]);
        // buf[2..5] + buf[0..3]
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }

    /// Broadcast axis (dim 1) with a nonzero stride must contribute stride 0.
    #[test]
    fn strided_broadcast_dim_one() {
        let a = vec![100.0, 200.0]; // [2,1] with strides [1,1] (transposed [1,2])
        let b = vec![1.0, 2.0, 3.0]; // [3]
        let (shape, data) = add_strided(&[2, 1], &[1, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![101.0, 102.0, 103.0, 201.0, 202.0, 203.0]);
    }

    #[test]
    fn strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let err = add_strided(&[2, 3], &[3, 2], 0, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
        let err_off = add_strided(&[2, 3], &[3, 1], 1, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err_off, KernelError::StridesOutOfBounds);
    }

    #[test]
    fn strided_size_zero_and_rank_zero() {
        let (shape, data) = add_strided(&[0, 3], &[3, 1], 0, &[], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
        let (shape0, data0) = add_strided(&[], &[], 0, &[5.0], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape0, vec![3]);
        assert_eq!(data0, vec![6.0, 7.0, 8.0]);
    }
}
