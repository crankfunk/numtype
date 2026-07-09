//! Broadcasting elementwise add. Mirrors `elementwiseBinary` in
//! `spike/src/runtime.ts` (specialized to `op = (x, y) => x + y`; the ABI
//! only needs `add`, so the generic `op` callback isn't ported).

use crate::shape::{
    align_to_rank, broadcast_shape, checked_element_count, compute_strides, effective_strides, unravel, KResult,
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
}
