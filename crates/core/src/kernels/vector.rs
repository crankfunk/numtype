//! Kern 07: `dot`/`norm_sq` reductions — the two new kernels the embedding/
//! RAG op surface needs (`norm`/`cosineSimilarity` are TS-side scalar
//! compositions over these, per the spec: `sqrt`/`*`// `/` are all IEEE-exact,
//! so they need no kernel of their own).
//!
//! `dot_strided`: 1-D inner product of two strided rank-1 views — single
//! accumulator, strictly ascending index, `acc += a[i] * b[i]`, seed `0.0`
//! (bit-identity law). Rank != 1 on either operand, or unequal length, is
//! `ShapeIncompatible` (status 1) — matmul-style promotion/broadcast is
//! explicitly out of scope (spec: "Batch/broadcast dot ... matmul already
//! covers matrix/batch cases").
//!
//! `norm_sq_strided`: sum of squares over EVERY element of a strided view (any
//! rank), in the view's LOGICAL row-major order — mirrors
//! `sum::sum_all_strided`'s determinism contract exactly (same reasoning: for
//! a transposed view, logical order differs from memory order, and float
//! addition is order-sensitive) but accumulates `v * v` instead of `v`.

use crate::shape::{checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult, KernelError};

/// 1-D inner product of two strided rank-1 views. Both operands must be rank
/// 1 with equal length (checked_element_count first, for the MAX_RANK/
/// overflow defense-in-depth every kernel here applies to its own operands
/// independently — see `add::add_strided` for the same pattern); a rank != 1
/// or length mismatch is `ShapeIncompatible`. A size-0 vector pair is valid
/// and returns `0.0` (the empty accumulation — matches `np.dot([], [])`).
#[allow(clippy::too_many_arguments)]
pub fn dot_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<f64> {
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;
    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(KernelError::ShapeIncompatible);
    }
    if a_shape[0] != b_shape[0] {
        return Err(KernelError::ShapeIncompatible);
    }
    validate_strided_bounds(a_shape, a_strides, a_offset, a_data.len() as u32)?;
    validate_strided_bounds(b_shape, b_strides, b_offset, b_data.len() as u32)?;

    let n = a_shape[0];
    let a_stride = a_strides[0];
    let b_stride = b_strides[0];
    let mut acc = 0.0f64;
    for i in 0..n {
        let a_val = a_data.get((a_offset + i * a_stride) as usize).copied().unwrap_or(0.0);
        let b_val = b_data.get((b_offset + i * b_stride) as usize).copied().unwrap_or(0.0);
        acc += a_val * b_val;
    }
    Ok(acc)
}

/// Sum of squares over every element of a strided view (any rank <=
/// `MAX_RANK`), accumulated in LOGICAL row-major order — same iteration
/// (`unravel` over `compute_strides(shape)`, offset via caller strides) as
/// `sum::sum_all_strided`, just squaring each value before accumulating.
/// Size-0 views sum to `0.0` without reading (matches `sum_all_strided`).
pub fn norm_sq_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64]) -> KResult<f64> {
    let size = checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;

    let logical_strides = compute_strides(shape);
    let mut acc = 0.0f64;
    for flat in 0..size {
        let idx = unravel(flat, shape, &logical_strides);
        let mut off: u32 = offset;
        for i in 0..shape.len() {
            off += idx[i] * strides[i];
        }
        let v = data.get(off as usize).copied().unwrap_or(0.0);
        acc += v * v;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::compute_strides;

    // --- dot_strided ---------------------------------------------------------

    #[test]
    fn dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let got = dot_strided(&[3], &[1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(got, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn dot_size_zero_is_zero() {
        let got = dot_strided(&[0], &[1], 0, &[], &[0], &[1], 0, &[]).unwrap();
        assert_eq!(got, 0.0);
    }

    #[test]
    fn dot_strided_window() {
        // a-window: buf[1..4] = [1,2,3]; b: step-2 view over [10,20,30,40,50] -> [10,30,50]
        let a_buf = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let b_buf = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let got = dot_strided(&[3], &[1], 1, &a_buf, &[3], &[2], 0, &b_buf).unwrap();
        assert_eq!(got, 1.0 * 10.0 + 2.0 * 30.0 + 3.0 * 50.0);
    }

    #[test]
    fn dot_length_mismatch_is_shape_incompatible() {
        let err = dot_strided(&[3], &[1], 0, &[0.0; 3], &[4], &[1], 0, &[0.0; 4]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn dot_rank_errors() {
        let err_a = dot_strided(&[2, 3], &[3, 1], 0, &[0.0; 6], &[6], &[1], 0, &[0.0; 6]).unwrap_err();
        assert_eq!(err_a, KernelError::ShapeIncompatible);
        let err_b = dot_strided(&[6], &[1], 0, &[0.0; 6], &[2, 3], &[3, 1], 0, &[0.0; 6]).unwrap_err();
        assert_eq!(err_b, KernelError::ShapeIncompatible);
        let err_scalar = dot_strided(&[], &[], 0, &[1.0], &[1], &[1], 0, &[1.0]).unwrap_err();
        assert_eq!(err_scalar, KernelError::ShapeIncompatible);
    }

    #[test]
    fn dot_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = dot_strided(&shape, &shape, 0, &[0.0], &[1], &[1], 0, &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn dot_strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 3];
        let err = dot_strided(&[3], &[2], 0, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    // --- norm_sq_strided -------------------------------------------------------

    #[test]
    fn norm_sq_basic() {
        let data = vec![3.0, 4.0]; // norm_sq = 9+16 = 25 (norm = 5)
        let got = norm_sq_strided(&[2], &[1], 0, &data).unwrap();
        assert_eq!(got, 25.0);
    }

    #[test]
    fn norm_sq_rank0_and_size_zero() {
        assert_eq!(norm_sq_strided(&[], &[], 0, &[5.0]).unwrap(), 25.0);
        assert_eq!(norm_sq_strided(&[0, 3], &[3, 1], 0, &[]).unwrap(), 0.0);
    }

    /// The order-sensitivity pin (same idiom as `sum::sum_all_strided`'s own
    /// absorption-pattern test), adapted to squares. Since `norm_sq` only
    /// ever accumulates NON-NEGATIVE terms, the classic large-plus-large-
    /// negative CANCELLATION trick (which `sum_all_strided`'s own test
    /// relies on) does not transfer — squaring destroys sign, so there is no
    /// exact-zero cancellation available. Order-sensitivity for a sum of
    /// non-negative floats still exists (plain non-associativity/rounding of
    /// `+`), just not via a hand-derivable "absorption story" — the values
    /// below were found by exhaustive random search in Node (see this
    /// phase's implementation notes) specifically screening for `memory-order
    /// sum-of-squares != logical-order sum-of-squares`, then verified
    /// independently in JS: mem=13926483699150020, log=13926483699150016.
    #[test]
    fn norm_sq_strided_transposed_view_uses_logical_order() {
        let data: Vec<f64> = vec![60590962.0, -54635165.0, -9.0, 85265572.0, -6.0, -7.0];
        // View [3,2] with strides [1,3] = transpose of contiguous [2,3].
        let mut materialized = vec![0f64; 6];
        for i in 0..3 {
            for j in 0..2 {
                materialized[i * 2 + j] = data[i + 3 * j];
            }
        }
        // Reference: square-then-sum the MATERIALIZED (logical-order) data,
        // via the same accumulation loop as norm_sq_strided over a
        // contiguous view (offset 0, natural strides).
        let mat_strides = compute_strides(&[3, 2]);
        let reference = norm_sq_strided(&[3, 2], &mat_strides, 0, &materialized).unwrap();
        let strided = norm_sq_strided(&[3, 2], &[1, 3], 0, &data).unwrap();
        assert_eq!(reference.to_bits(), strided.to_bits());

        // Non-vacuity: accumulating squares in MEMORY order (flat 0..6 over
        // `data` directly, ignoring the view's logical order) must differ in
        // bits from the logical-order result, so this test really pins the
        // order rather than merely the value.
        let memory_order_strides = compute_strides(&[6]);
        let memory_order = norm_sq_strided(&[6], &memory_order_strides, 0, &data).unwrap();
        assert_ne!(memory_order.to_bits(), strided.to_bits());
    }

    #[test]
    fn norm_sq_strided_offset_and_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(norm_sq_strided(&[2], &[1], 2, &data).unwrap(), 9.0 + 16.0);
        let err = norm_sq_strided(&[2], &[1], 3, &data).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    #[test]
    fn norm_sq_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = norm_sq_strided(&shape, &shape, 0, &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    // --- cross-kernel parity: dot(a,a) === norm_sq(a) ---------------------------

    #[test]
    fn dot_of_a_with_itself_matches_norm_sq() {
        let a = vec![1.5, -2.25, 3.0, 0.0, -7.75];
        let dot_aa = dot_strided(&[5], &[1], 0, &a, &[5], &[1], 0, &a).unwrap();
        let strides = compute_strides(&[5]);
        let ns = norm_sq_strided(&[5], &strides, 0, &a).unwrap();
        assert_eq!(dot_aa.to_bits(), ns.to_bits());
    }
}
