//! WASM parity S4 (docs/wasm-parity-argmax-spec.md): index-of-maximum
//! reductions over a strided view — full (`argmax_all_strided`) and per-axis
//! (`argmax_axis_strided`). Structural twins of
//! [`crate::kernels::sum::sum_all_strided`] / [`crate::kernels::sum::sum_axis_strided`]:
//! same validation order, same iteration order, same offset algebra, same
//! `data.get(off).copied().unwrap_or(0.0)` read convention. Exactly two things
//! differ (spec D3): the accumulation `total += v` is replaced by the
//! challenger comparison below plus index bookkeeping, and there are two new
//! empty-reduction branches `sum` has no need for (summing zero elements is
//! `0.0`, a well-defined answer; taking the argmax of zero candidates is not).
//!
//! **Total order — PINNED, not re-derived (docs/op-w1-argmax-topk-spec.md D4,
//! implemented in `beatsMax`, spike/src/runtime.ts:555).** NaN counts as
//! MAXIMAL (NumPy `argmax` behavior); among several NaNs the FIRST index wins;
//! on a value tie (`0.0` vs `-0.0` included — compared with plain `>`, never a
//! bitwise or total-order comparison) the FIRST index wins. An element beats
//! the running maximum iff `(el.is_nan() && !max.is_nan()) || el > max`, the
//! literal transliteration of the TS predicate. Rust's `f64` `>` and
//! `f64::is_nan` are IEEE-754 exactly as JS's `>` and `Number.isNaN` are, so
//! the two predicates agree elementwise — that is the ARGUMENT; the PROOF is
//! differential (`spike/tests-runtime/resident.test.ts`, M1).
//!
//! **The result is a LOGICAL index, never a memory offset (the tragende
//! semantic point).** `argmax_all_strided` returns the index into the view's
//! own logical row-major flattening — the `flat` counter, never the computed
//! `off`. On a transposed or sliced view the two genuinely diverge; that is
//! the index-domain form of the accumulation-order trap
//! [`crate::kernels::sum::sum_all_strided`]'s doc comment describes, and
//! `argmax_all_strided_transposed_view_uses_logical_index` below pins it with
//! an explicit non-vacuity assertion. For the axis form the index runs along
//! the axis (`0..axis_dim`) and is therefore stride-independent by
//! construction.
//!
//! **`unwrap_or(0.0)` convention — load-bearing here in a way it is not for
//! `sum` (spec D3, Baustein-0 finding D2).** For `sum` the fallback `0.0` is
//! the additive neutral: a genuine no-op that could only ever mask a memory
//! error. For `argmax` the same `0.0` is a COMPETING CANDIDATE in the total
//! order — it could win or lose, and would silently corrupt the returned
//! index. The path is unreachable: both entry points call
//! `validate_strided_bounds` (crates/core/src/shape.rs:165-180) before the
//! loop, which proves `offset + Σ (dim_i − 1)·stride_i < data_len` for every
//! reachable element. Correctness — not merely memory safety, as for `sum` —
//! now rests on that guarantee.

use crate::shape::{checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult, KernelError};

/// Does `el` beat the running maximum `max`? The pinned total order (module
/// doc): a strict "does the NEW element win" challenger check, never `>=`, so
/// ties leave the first-seen index in place. Byte-for-byte the predicate of
/// `beatsMax` (spike/src/runtime.ts).
fn beats_max(el: f64, max: f64) -> bool {
    (el.is_nan() && !max.is_nan()) || el > max
}

/// Read the element at LOGICAL row-major position `flat` of a strided view —
/// the exact offset algebra `sum_all_strided`'s own inner loop performs
/// inline, factored out only so the seed read (`flat == 0`) and the challenger
/// reads share one code path, mirroring `argmaxRuntime`'s `maxVal = data[0]` +
/// `for (i = 1; …)` structure.
fn read_logical(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], logical_strides: &[u32], flat: u32) -> f64 {
    let idx = unravel(flat, shape, logical_strides);
    let mut off: u32 = offset;
    for i in 0..shape.len() {
        off += idx[i] * strides[i];
    }
    data.get(off as usize).copied().unwrap_or(0.0)
}

/// Index of the maximum element of a strided view, in the view's LOGICAL
/// row-major flattening (module doc). Mirrors `argmaxRuntime`'s `axis ===
/// undefined` branch: seed from logical element 0, then challenge with
/// elements `1..size`. Returned as `f64` because the whole library is f64-only
/// and every index here stays far below `2^53` (a `u32` is exactly
/// representable in f64 without rounding).
///
/// A size-0 view has no maximum and no index to return:
/// `KernelError::ShapeIncompatible`. `WNDArray.argmax` prevalidates that case
/// in TS (so the pinned message stem matches `argmaxRuntime`'s word for word)
/// — this branch is defense in depth, pinned by its own test below.
pub fn argmax_all_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64]) -> KResult<f64> {
    let size = checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;
    if size == 0 {
        return Err(KernelError::ShapeIncompatible);
    }

    let logical_strides = compute_strides(shape);
    let mut max_idx: u32 = 0;
    let mut max_val = read_logical(shape, strides, offset, data, &logical_strides, 0);
    for flat in 1..size {
        let v = read_logical(shape, strides, offset, data, &logical_strides, flat);
        if beats_max(v, max_val) {
            max_val = v;
            max_idx = flat;
        }
    }
    Ok(max_idx as f64)
}

/// Index of the maximum along `axis` (negative axes count from the end) of a
/// strided view — same loop order, same out-shape construction and same base-
/// offset algebra as [`crate::kernels::sum::sum_axis_strided`], with the
/// accumulation replaced by the challenger comparison. Each output element is
/// the index ALONG THE AXIS (`0..axis_dim`), so it is stride-independent.
///
/// Validation order mirrors `argmaxRuntime` exactly: axis range first
/// (`ShapeIncompatible`), then a zero-length axis (`ShapeIncompatible` —
/// no candidates, no index). Note that a size-0 OUTPUT is perfectly valid and
/// must NOT error: reducing `[0, 3]` along axis 1 yields an empty `[0]`
/// result, exactly as in TS.
pub fn argmax_axis_strided(
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
    if shape[norm_axis] == 0 {
        return Err(KernelError::ShapeIncompatible);
    }

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
        let mut max_idx: u32 = 0;
        let mut max_val = data.get(base_offset as usize).copied().unwrap_or(0.0);
        for a in 1..axis_dim {
            let off = base_offset + a * axis_stride;
            let v = data.get(off as usize).copied().unwrap_or(0.0);
            if beats_max(v, max_val) {
                max_val = v;
                max_idx = a;
            }
        }
        out[out_flat as usize] = max_idx as f64;
    }
    Ok((out_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_all_strided_contiguous() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // Hand reference: the maximum 9.0 sits at flat index 5.
        assert_eq!(argmax_all_strided(&[2, 4], &[4, 1], 0, &data).unwrap(), 5.0);
        assert_eq!(argmax_all_strided(&[8], &[1], 0, &data).unwrap(), 5.0);
    }

    /// The index-domain counterpart of `sum_all_strided`'s order guard: on a
    /// transposed view the answer must be the index into the view's LOGICAL
    /// flattening, which here provably differs from the memory-order index.
    #[test]
    fn argmax_all_strided_transposed_view_uses_logical_index() {
        // Base buffer, [2,3] row-major: [[0,9,2],[3,4,5]].
        // Memory-order argmax is 1 (the 9.0 sits at data[1]).
        let data = vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0];
        // Materialize the transpose by hand ([3,2] = [[0,3],[9,4],[2,5]]) and
        // take ITS flat argmax as the reference — the 9.0 is now at index 2.
        let mut materialized = vec![0f64; 6];
        for i in 0..3 {
            for j in 0..2 {
                materialized[i * 2 + j] = data[i + 3 * j];
            }
        }
        let reference = argmax_all_strided(&[6], &[1], 0, &materialized).unwrap();
        // View [3,2] with strides [1,3] = transpose of the contiguous [2,3].
        let strided = argmax_all_strided(&[3, 2], &[1, 3], 0, &data).unwrap();
        assert_eq!(reference, strided);
        assert_eq!(strided, 2.0);
        // Non-vacuity: the MEMORY-order index is a genuinely different number
        // here, so this pins the LOGICAL index, not merely "an" index.
        let memory_order = argmax_all_strided(&[6], &[1], 0, &data).unwrap();
        assert_ne!(memory_order, strided);
        assert_eq!(memory_order, 1.0);
    }

    #[test]
    fn argmax_all_strided_offset_and_bounds() {
        let data = vec![7.0, 1.0, 2.0, 3.0];
        // Offset window over data[1..4] = [1,2,3] -> max at logical index 2.
        assert_eq!(argmax_all_strided(&[3], &[1], 1, &data).unwrap(), 2.0);
        let err = argmax_all_strided(&[3], &[1], 2, &data).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    #[test]
    fn argmax_all_strided_rank_zero_is_index_zero() {
        let data = vec![10.0, 20.0];
        assert_eq!(argmax_all_strided(&[], &[], 0, &data).unwrap(), 0.0);
        // A rank-0 view reads exactly data[offset] and still answers 0.
        assert_eq!(argmax_all_strided(&[], &[], 1, &data).unwrap(), 0.0);
    }

    #[test]
    fn argmax_all_strided_size_zero_is_shape_incompatible() {
        let err = argmax_all_strided(&[0, 3], &[3, 1], 0, &[]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
        let err_flat = argmax_all_strided(&[0], &[1], 0, &[]).unwrap_err();
        assert_eq!(err_flat, KernelError::ShapeIncompatible);
    }

    #[test]
    fn argmax_all_strided_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = argmax_all_strided(&shape, &shape, 0, &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn argmax_all_strided_nan_is_maximal_and_first_nan_wins() {
        // A NaN beats every finite value, including +Inf.
        let data = vec![1.0, f64::INFINITY, f64::NAN, 2.0];
        assert_eq!(argmax_all_strided(&[4], &[1], 0, &data).unwrap(), 2.0);
        // Several NaNs: the FIRST one wins (challenger check is strict).
        let many_nan = vec![1.0, f64::NAN, 5.0, f64::NAN];
        assert_eq!(argmax_all_strided(&[4], &[1], 0, &many_nan).unwrap(), 1.0);
        // Without any NaN, +Inf wins.
        let no_nan = vec![1.0, f64::INFINITY, 2.0];
        assert_eq!(argmax_all_strided(&[3], &[1], 0, &no_nan).unwrap(), 1.0);
    }

    #[test]
    fn argmax_all_strided_ties_keep_the_first_index() {
        // Plain value tie.
        let tie = vec![5.0, 5.0, 5.0];
        assert_eq!(argmax_all_strided(&[3], &[1], 0, &tie).unwrap(), 0.0);
        // +0.0 / -0.0 tie: `>` is false in BOTH directions, so the first index
        // stands regardless of sign order (never `Object.is`/`total_cmp`).
        let zeros_pos_first = vec![0.0, -0.0];
        assert_eq!(argmax_all_strided(&[2], &[1], 0, &zeros_pos_first).unwrap(), 0.0);
        let zeros_neg_first = vec![-0.0, 0.0];
        assert_eq!(argmax_all_strided(&[2], &[1], 0, &zeros_neg_first).unwrap(), 0.0);
    }

    #[test]
    fn argmax_axis_strided_contiguous() {
        // [[1,7,3],[9,2,4]] -> axis 0: [1,0,1]; axis 1: [1,0]
        let data = vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0];
        let (shape0, out0) = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, 0).unwrap();
        assert_eq!(shape0, vec![3]);
        assert_eq!(out0, vec![1.0, 0.0, 1.0]);
        let (shape1, out1) = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, 1).unwrap();
        assert_eq!(shape1, vec![2]);
        assert_eq!(out1, vec![1.0, 0.0]);
    }

    #[test]
    fn argmax_axis_strided_negative_axis_matches_positive() {
        let data = vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0];
        let (shape_pos, out_pos) = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, 1).unwrap();
        let (shape_neg, out_neg) = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, -1).unwrap();
        assert_eq!(shape_pos, shape_neg);
        assert_eq!(out_pos, out_neg);
    }

    /// Axis reduction of a transposed VIEW equals the corresponding axis
    /// reduction of the base — the axis-form counterpart of
    /// `sum_axis_strided_transposed_view`.
    #[test]
    fn argmax_axis_strided_transposed_view() {
        let data = vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0]; // [[1,7,3],[9,2,4]]
        // A^T is [3,2] with strides [1,3]: [[1,9],[7,2],[3,4]].
        let (shape_t, out_t) = argmax_axis_strided(&[3, 2], &[1, 3], 0, &data, 0).unwrap();
        assert_eq!(shape_t, vec![2]);
        // Columns of A^T are the rows of A: max index of [1,7,3] is 1, of
        // [9,2,4] is 0.
        assert_eq!(out_t, vec![1.0, 0.0]);
        let (_, out_base) = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, 1).unwrap();
        assert_eq!(out_t, out_base);
    }

    #[test]
    fn argmax_axis_strided_offset_window() {
        // Buffer [5,?]: use rows 1.. of a [3,2] base -> shape [2,2], offset 2.
        let data = vec![0.0, 0.0, 8.0, 1.0, 2.0, 9.0]; // rows: [0,0] [8,1] [2,9]
        let (shape, out) = argmax_axis_strided(&[2, 2], &[2, 1], 2, &data, 1).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(out, vec![0.0, 1.0]);
    }

    #[test]
    fn argmax_axis_strided_zero_length_axis_is_shape_incompatible() {
        let err = argmax_axis_strided(&[0, 3], &[3, 1], 0, &[], 0).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    /// A size-0 OUTPUT is valid and must not error (the axis itself is
    /// non-empty): `[0,3]` reduced along axis 1 is an empty `[0]`.
    #[test]
    fn argmax_axis_strided_size_zero_output_is_ok() {
        let (shape, out) = argmax_axis_strided(&[0, 3], &[3, 1], 0, &[], 1).unwrap();
        assert_eq!(shape, vec![0]);
        assert_eq!(out, Vec::<f64>::new());
    }

    #[test]
    fn argmax_axis_strided_axis_out_of_range_and_rank_too_large() {
        let data = vec![0.0; 6];
        let err = argmax_axis_strided(&[2, 3], &[3, 1], 0, &data, 5).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
        let err_scalar = argmax_axis_strided(&[], &[], 0, &[1.0], 0).unwrap_err();
        assert_eq!(err_scalar, KernelError::ShapeIncompatible);
        let big = vec![1u32; 33];
        let err_rank = argmax_axis_strided(&big, &big, 0, &[0.0], 0).unwrap_err();
        assert_eq!(err_rank, KernelError::RankTooLarge);
        let err_oob = argmax_axis_strided(&[2, 3], &[4, 1], 0, &data, 0).unwrap_err();
        assert_eq!(err_oob, KernelError::StridesOutOfBounds);
    }

    #[test]
    fn argmax_axis_strided_nan_and_ties() {
        // Rows: [1, NaN, 3] -> NaN wins at 1; [NaN, NaN, 0] -> first NaN at 0;
        // [5, 5, 5] -> first index 0; [0.0, -0.0, -0.0] -> first index 0.
        let data = vec![
            1.0,
            f64::NAN,
            3.0,
            f64::NAN,
            f64::NAN,
            0.0,
            5.0,
            5.0,
            5.0,
            0.0,
            -0.0,
            -0.0,
        ];
        let (shape, out) = argmax_axis_strided(&[4, 3], &[3, 1], 0, &data, 1).unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0]);
    }
}
