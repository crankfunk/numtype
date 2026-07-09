//! 2-D matmul core + batch-dim broadcasting. Mirrors the batch/core part of
//! `matmulRuntime` in `spike/src/runtime.ts` (lines computing `batchOut`
//! through the `mi`/`ni`/`ki` triple loop). Per spec, **1-D promotion and
//! the final squeeze stay TS-side** — this kernel requires both operands to
//! already be rank >= 2 (promoted) and returns the *unsqueezed*
//! `[...batch, m, n]` shape; the caller (backend.ts) squeezes exactly as
//! `matmulRuntime` does after calling this.

use crate::shape::{
    align_to_rank, aligned_effective_strides, broadcast_shape, checked_element_count, compute_strides,
    effective_strides, product, unravel, validate_strided_bounds, KResult, KernelError,
};

pub fn matmul(a_shape: &[u32], a_data: &[f64], b_shape: &[u32], b_data: &[f64]) -> KResult<(Vec<u32>, Vec<f64>)> {
    // Contract: caller has already promoted any 1-D operand to rank 2.
    // A rank < 2 operand here means the ABI contract was violated (or, for
    // a from-scratch fuzz/negative test, a deliberately malformed call) —
    // bucketed as "shape incompatible" since none of the other categories
    // fit and it is, structurally, an invalid shape for this op.
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(KernelError::ShapeIncompatible);
    }
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;

    let m = a_shape[a_shape.len() - 2];
    let k1 = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k1 != k2 {
        return Err(KernelError::ShapeIncompatible);
    }

    let batch_a = &a_shape[..a_shape.len() - 2];
    let batch_b = &b_shape[..b_shape.len() - 2];
    let batch_out = broadcast_shape(batch_a, batch_b)?;
    let batch_rank = batch_out.len();

    let a_strides_full = compute_strides(a_shape);
    let b_strides_full = compute_strides(b_shape);

    let (a_batch_shape_al, a_batch_strides_al) =
        align_to_rank(batch_a, &a_strides_full[..a_strides_full.len() - 2], batch_rank);
    let (b_batch_shape_al, b_batch_strides_al) =
        align_to_rank(batch_b, &b_strides_full[..b_strides_full.len() - 2], batch_rank);
    let a_batch_eff = effective_strides(&a_batch_shape_al, &a_batch_strides_al);
    let b_batch_eff = effective_strides(&b_batch_shape_al, &b_batch_strides_al);

    let a_row_stride = a_strides_full[a_strides_full.len() - 2];
    let a_col_stride = a_strides_full[a_strides_full.len() - 1];
    let b_row_stride = b_strides_full[b_strides_full.len() - 2];
    let b_col_stride = b_strides_full[b_strides_full.len() - 1];

    let mut out_full_shape = batch_out.clone();
    out_full_shape.push(m);
    out_full_shape.push(n);
    // batch_a.len(), batch_b.len() <= 30 (since a_shape/b_shape <= MAX_RANK
    // and each has 2 axes reserved for m/k, n/k), so batch_rank <= 30 and
    // out_full_shape.len() <= 32 always — never independently RankTooLarge.
    debug_assert!(out_full_shape.len() <= crate::shape::MAX_RANK);
    let out_size = checked_element_count(&out_full_shape)?;

    let out_strides_full = compute_strides(&out_full_shape);
    let out_batch_strides = &out_strides_full[..batch_rank];
    let batch_strides_plain = compute_strides(&batch_out);

    let batch_size = product(&batch_out); // safe: batch_size <= out_size, already validated above
    let mut out = vec![0f64; out_size as usize];

    for b_idx in 0..batch_size {
        let multi = unravel(b_idx, &batch_out, &batch_strides_plain);
        let mut a_batch_off: u32 = 0;
        let mut b_batch_off: u32 = 0;
        let mut out_batch_off: u32 = 0;
        for i in 0..batch_rank {
            let ix = multi[i];
            a_batch_off += ix * a_batch_eff[i];
            b_batch_off += ix * b_batch_eff[i];
            out_batch_off += ix * out_batch_strides[i];
        }
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f64;
                for ki in 0..k1 {
                    let a_idx = a_batch_off + mi * a_row_stride + ki * a_col_stride;
                    let b_idx2 = b_batch_off + ki * b_row_stride + ni * b_col_stride;
                    let a_val = a_data.get(a_idx as usize).copied().unwrap_or(0.0);
                    let b_val = b_data.get(b_idx2 as usize).copied().unwrap_or(0.0);
                    sum += a_val * b_val;
                }
                out[(out_batch_off + mi * n + ni) as usize] = sum;
            }
        }
    }

    Ok((out_full_shape, out))
}

/// Kern 03: strided generalization of [`matmul`] — same contract (operands
/// arrive promoted to rank >= 2, output is the unsqueezed `[...batch, m, n]`
/// shape), but row/col/batch strides come from the caller instead of being
/// computed from the shape. Same triple-loop order and accumulation order as
/// [`matmul`], so contiguous metadata reproduces it bit-identically. The
/// contiguous original stays untouched (frozen v1 baseline).
#[allow(clippy::too_many_arguments)]
pub fn matmul_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(KernelError::ShapeIncompatible);
    }
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;
    validate_strided_bounds(a_shape, a_strides, a_offset, a_data.len() as u32)?;
    validate_strided_bounds(b_shape, b_strides, b_offset, b_data.len() as u32)?;

    let m = a_shape[a_shape.len() - 2];
    let k1 = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k1 != k2 {
        return Err(KernelError::ShapeIncompatible);
    }

    let batch_a = &a_shape[..a_shape.len() - 2];
    let batch_b = &b_shape[..b_shape.len() - 2];
    let batch_out = broadcast_shape(batch_a, batch_b)?;
    let batch_rank = batch_out.len();

    let a_batch_eff = aligned_effective_strides(batch_a, &a_strides[..a_strides.len() - 2], batch_rank);
    let b_batch_eff = aligned_effective_strides(batch_b, &b_strides[..b_strides.len() - 2], batch_rank);

    let a_row_stride = a_strides[a_strides.len() - 2];
    let a_col_stride = a_strides[a_strides.len() - 1];
    let b_row_stride = b_strides[b_strides.len() - 2];
    let b_col_stride = b_strides[b_strides.len() - 1];

    let mut out_full_shape = batch_out.clone();
    out_full_shape.push(m);
    out_full_shape.push(n);
    debug_assert!(out_full_shape.len() <= crate::shape::MAX_RANK);
    let out_size = checked_element_count(&out_full_shape)?;

    let out_strides_full = compute_strides(&out_full_shape);
    let out_batch_strides = &out_strides_full[..batch_rank];
    let batch_strides_plain = compute_strides(&batch_out);

    let batch_size = product(&batch_out);
    let mut out = vec![0f64; out_size as usize];

    for b_idx in 0..batch_size {
        let multi = unravel(b_idx, &batch_out, &batch_strides_plain);
        let mut a_batch_off: u32 = a_offset;
        let mut b_batch_off: u32 = b_offset;
        let mut out_batch_off: u32 = 0;
        for i in 0..batch_rank {
            let ix = multi[i];
            a_batch_off += ix * a_batch_eff[i];
            b_batch_off += ix * b_batch_eff[i];
            out_batch_off += ix * out_batch_strides[i];
        }
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f64;
                for ki in 0..k1 {
                    let a_idx = a_batch_off + mi * a_row_stride + ki * a_col_stride;
                    let b_idx2 = b_batch_off + ki * b_row_stride + ni * b_col_stride;
                    let a_val = a_data.get(a_idx as usize).copied().unwrap_or(0.0);
                    let b_val = b_data.get(b_idx2 as usize).copied().unwrap_or(0.0);
                    sum += a_val * b_val;
                }
                out[(out_batch_off + mi * n + ni) as usize] = sum;
            }
        }
    }

    Ok((out_full_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_d_core() {
        // M1 = [[1,2,3],[4,5,6]] (2x3), M2 = [[7,8],[9,10],[11,12]] (3x2)
        // Expected (from spike/demo.ts's hand-checked comment):
        // row0: [58, 64], row1: [139, 154]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let (shape, data) = matmul(&[2, 3], &a, &[3, 2], &b).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn batch_matmul_no_broadcast() {
        // batch=2, each 2x2 @ 2x2
        let a = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // two identity-like 2x2s
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (shape, data) = matmul(&[2, 2, 2], &a, &[2, 2, 2], &b).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        // batch0: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
        // batch1: 2I @ [[5,6],[7,8]] = [[10,12],[14,16]]
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn batch_broadcast() {
        // a: [1,2,2] (batch dim broadcasts), b: [3,2,2] -> out batch = 3
        let a = vec![1.0, 2.0, 3.0, 4.0]; // single 2x2, broadcast over 3 batches
        let b: Vec<f64> = (0..12).map(|i| i as f64 + 1.0).collect(); // 3 distinct 2x2s
        let (shape, data) = matmul(&[1, 2, 2], &a, &[3, 2, 2], &b).unwrap();
        assert_eq!(shape, vec![3, 2, 2]);
        // batch0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
        assert_eq!(&data[0..4], &[7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn inner_dim_mismatch() {
        let err = matmul(&[2, 3], &[0.0; 6], &[4, 2], &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn rank_below_two_rejected() {
        let err = matmul(&[3], &[1.0, 2.0, 3.0], &[3, 2], &[0.0; 6]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn rank_too_large_operand() {
        let mut shape = vec![1u32; 31];
        shape.push(2); // rank 32 already at cap
        let mut shape2 = shape.clone();
        shape2.push(1); // rank 33: over cap
        let err = matmul(&shape2, &[0.0], &[2, 2], &[0.0; 4]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    // --- Kern 03: matmul_strided ------------------------------------------

    #[test]
    fn strided_contiguous_metadata_matches_contiguous_kernel() {
        let a: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let b: Vec<f64> = (0..6).map(|i| i as f64 * 1.5).collect();
        let (shape_c, data_c) = matmul(&[2, 3], &a, &[3, 2], &b).unwrap();
        let (shape_s, data_s) = matmul_strided(&[2, 3], &[3, 1], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape_c, shape_s);
        assert_eq!(data_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   data_s.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    /// Logical A = [[1,2,3],[4,5,6]] supplied as a transposed VIEW over its
    /// column-major storage [1,4,2,5,3,6] (shape [2,3], strides [1,2]) —
    /// must reproduce the classic hand-checked result.
    #[test]
    fn strided_transposed_view_operand() {
        let a_store = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let (shape, data) = matmul_strided(&[2, 3], &[1, 2], 0, &a_store, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    /// Nonzero base offset selects the second 2x2 matrix out of a buffer of
    /// two — equivalent to a (future) slicing view.
    #[test]
    fn strided_offset_selects_submatrix() {
        let buf = vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0]; // [I, M]
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        let (shape, data) = matmul_strided(&[2, 2], &[2, 1], 4, &buf, &[2, 2], &[2, 1], 0, &ident).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![5.0, 6.0, 7.0, 8.0]);
    }

    /// Batch matmul where the batch axis itself is strided (a transposed
    /// batch view): buffer [2,2,2] read with batch/rows swapped.
    #[test]
    fn strided_batch_axes() {
        // Buffer holds [[a00,a01],[a10,a11]] x 2 batches contiguous [2,2,2].
        let buf: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        // View: shape [2,2,2], strides permuted so batch axis is the MIDDLE
        // storage axis: view[b][i][j] = buf[i][b][j] -> strides [2,4,1].
        let (shape, data) =
            matmul_strided(&[2, 2, 2], &[2, 4, 1], 0, &buf, &[2, 2], &[2, 1], 0, &ident).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        // view batch0 = [[0,1],[4,5]], batch1 = [[2,3],[6,7]]
        assert_eq!(data, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let b = vec![0.0; 6];
        let err = matmul_strided(&[2, 3], &[3, 2], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    #[test]
    fn strided_rank_below_two_rejected() {
        let err = matmul_strided(&[3], &[1], 0, &[1.0, 2.0, 3.0], &[3, 2], &[2, 1], 0, &[0.0; 6]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }
}
