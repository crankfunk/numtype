//! Kern 07: broadcasting elementwise sub/mul/div. Structural mirror of
//! `add.rs`'s `add_strided` — same validation order, same iteration order,
//! same offset algebra — generalized over a binary `f64` op via a
//! monomorphized generic (no dynamic dispatch, no vtable): each of
//! `sub_strided`/`mul_strided`/`div_strided` compiles to its own specialized
//! loop, bit-identical to a hand-written copy of `add_strided` with `+`
//! replaced by the respective operator. `div_strided` is pure IEEE 754 (no
//! zero checks, no special-casing): `x / 0.0` is a signed infinity, `0.0 /
//! 0.0` is NaN — exactly Rust's/WASM's `f64` division semantics (see
//! docs/kern-07-elementwise-vector-spec.md, "div is pure IEEE 754").

use crate::shape::{
    aligned_effective_strides, broadcast_shape, checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult,
};

/// Private generic core: identical loop structure to
/// [`crate::kernels::add::add_strided`], with the binary op abstracted over
/// `F`. Monomorphized per caller ([`sub_strided`]/[`mul_strided`]/
/// [`div_strided`] below) — no dynamic dispatch, no trait object.
#[allow(clippy::too_many_arguments)]
fn binary_strided<F: Fn(f64, f64) -> f64>(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
    op: F,
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

    // Same u32-offset-accumulation safety argument as add_strided: every
    // partial sum is bounded by the operand's own validated max reach
    // (validate_strided_bounds above), so wraparound is structurally
    // unreachable.
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
        out[flat as usize] = op(a_val, b_val);
    }
    Ok((out_shape, out))
}

/// Broadcasting elementwise subtract over two strided views: `a - b`.
#[allow(clippy::too_many_arguments)]
pub fn sub_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    binary_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data, |x, y| x - y)
}

/// Broadcasting elementwise multiply over two strided views: `a * b`.
#[allow(clippy::too_many_arguments)]
pub fn mul_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    binary_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data, |x, y| x * y)
}

/// Broadcasting elementwise divide over two strided views: `a / b`. Pure
/// IEEE 754 (see module doc comment) — no zero checks, no throws.
#[allow(clippy::too_many_arguments)]
pub fn div_strided(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    binary_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data, |x, y| x / y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::KernelError;

    // --- sub_strided --------------------------------------------------------

    #[test]
    fn sub_same_shape() {
        let (shape, data) = sub_strided(&[2, 2], &[2, 1], 0, &[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1], 0, &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![-9.0, -18.0, -27.0, -36.0]);
    }

    #[test]
    fn sub_trailing_broadcast() {
        // [2,3] - [3] -> [2,3]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let (shape, data) = sub_strided(&[2, 3], &[3, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![-9.0, -18.0, -27.0, -6.0, -15.0, -24.0]);
    }

    #[test]
    fn sub_rank0_scalar_broadcast() {
        let (shape, data) = sub_strided(&[], &[], 0, &[5.0], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(data, vec![4.0, 3.0, 2.0]);
    }

    #[test]
    fn sub_size_zero_array() {
        let (shape, data) = sub_strided(&[0, 3], &[3, 1], 0, &[], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn sub_incompatible_shapes() {
        let err = sub_strided(&[2, 3], &[3, 1], 0, &[0.0; 6], &[2, 4], &[4, 1], 0, &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn sub_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = sub_strided(&shape, &shape, 0, &[0.0], &[1], &[1], 0, &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    /// Transposed view operand: A^T (view of [2,3] as [3,2], strides [1,3])
    /// minus a contiguous [3,2].
    #[test]
    fn sub_transposed_view_operand() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // [3,2] contiguous
        let (shape, data) = sub_strided(&[3, 2], &[1, 3], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape, vec![3, 2]);
        // A^T = [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![1.0 - 10.0, 4.0 - 20.0, 2.0 - 30.0, 5.0 - 40.0, 3.0 - 50.0, 6.0 - 60.0]);
    }

    #[test]
    fn sub_offset_window() {
        let buf = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (shape, data) = sub_strided(&[3], &[1], 2, &buf, &[3], &[1], 0, &buf).unwrap();
        assert_eq!(shape, vec![3]);
        // buf[2..5] - buf[0..3] = [2,3,4] - [0,1,2]
        assert_eq!(data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn sub_broadcast_dim_one() {
        let a = vec![100.0, 200.0]; // [2,1] strides [1,1]
        let b = vec![1.0, 2.0, 3.0]; // [3]
        let (shape, data) = sub_strided(&[2, 1], &[1, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![99.0, 98.0, 97.0, 199.0, 198.0, 197.0]);
    }

    #[test]
    fn sub_strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let err = sub_strided(&[2, 3], &[3, 2], 0, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    // --- mul_strided ---------------------------------------------------------

    #[test]
    fn mul_same_shape() {
        let (shape, data) = mul_strided(&[2, 2], &[2, 1], 0, &[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1], 0, &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn mul_trailing_broadcast() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let (shape, data) = mul_strided(&[2, 3], &[3, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn mul_size_zero_array() {
        let (shape, data) = mul_strided(&[0, 3], &[3, 1], 0, &[], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn mul_incompatible_shapes() {
        let err = mul_strided(&[2, 3], &[3, 1], 0, &[0.0; 6], &[2, 4], &[4, 1], 0, &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn mul_transposed_view_operand() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let (shape, data) = mul_strided(&[3, 2], &[1, 3], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(data, vec![10.0, 80.0, 60.0, 200.0, 150.0, 360.0]);
    }

    #[test]
    fn mul_broadcast_dim_one() {
        let a = vec![10.0, 20.0]; // [2,1]
        let b = vec![1.0, 2.0, 3.0]; // [3]
        let (shape, data) = mul_strided(&[2, 1], &[1, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn mul_strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let err_off = mul_strided(&[2, 3], &[3, 1], 1, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err_off, KernelError::StridesOutOfBounds);
    }

    // --- div_strided -----------------------------------------------------------
    // Pure IEEE 754: no zero checks. Explicitly pins the documented divergence
    // from NumPy (which additionally warns; we don't) at the VALUE level.

    #[test]
    fn div_same_shape() {
        let (shape, data) = div_strided(&[2, 2], &[2, 1], 0, &[10.0, 20.0, 30.0, 40.0], &[2, 2], &[2, 1], 0, &[2.0, 4.0, 5.0, 8.0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn div_trailing_broadcast() {
        let a = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let b = vec![2.0, 4.0, 5.0];
        let (shape, data) = div_strided(&[2, 3], &[3, 1], 0, &a, &[3], &[1], 0, &b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![5.0, 5.0, 6.0, 20.0, 12.5, 12.0]);
    }

    #[test]
    fn div_by_zero_is_signed_infinity_not_an_error() {
        let (shape, data) = div_strided(&[4], &[1], 0, &[1.0, -1.0, 0.0, 42.0], &[4], &[1], 0, &[0.0, 0.0, 0.0, 2.0]).unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(data[0], f64::INFINITY);
        assert_eq!(data[1], f64::NEG_INFINITY);
        assert!(data[2].is_nan()); // 0.0 / 0.0
        assert_eq!(data[3], 21.0);
    }

    #[test]
    fn div_negative_zero_signedness() {
        // 0 / -2 -> -0.0, distinguished from +0.0 via to_bits (matches the
        // differential suite's Object.is standard).
        let (shape, data) = div_strided(&[1], &[1], 0, &[0.0], &[1], &[1], 0, &[-2.0]).unwrap();
        assert_eq!(shape, vec![1]);
        assert_eq!(data[0].to_bits(), (-0.0f64).to_bits());
        assert_ne!(data[0].to_bits(), (0.0f64).to_bits());
    }

    #[test]
    fn div_size_zero_array() {
        let (shape, data) = div_strided(&[0, 3], &[3, 1], 0, &[], &[3], &[1], 0, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn div_incompatible_shapes() {
        let err = div_strided(&[2, 3], &[3, 1], 0, &[0.0; 6], &[2, 4], &[4, 1], 0, &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn div_transposed_view_operand() {
        let a = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // [[10,20,30],[40,50,60]]
        let b = vec![2.0, 4.0, 5.0, 8.0, 10.0, 12.0]; // [3,2] contiguous
        let (shape, data) = div_strided(&[3, 2], &[1, 3], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap();
        assert_eq!(shape, vec![3, 2]);
        // A^T = [[10,40],[20,50],[30,60]]
        assert_eq!(data, vec![10.0 / 2.0, 40.0 / 4.0, 20.0 / 5.0, 50.0 / 8.0, 30.0 / 10.0, 60.0 / 12.0]);
    }

    #[test]
    fn div_strided_out_of_bounds_is_status_4() {
        let a = vec![1.0; 6];
        let err = div_strided(&[2, 3], &[3, 2], 0, &a, &[3], &[1], 0, &a).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }
}
