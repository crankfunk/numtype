//! Kern 03: gather a strided view into a fresh contiguous row-major buffer —
//! the workhorse behind `WNDArray.contiguous()` and the copy-out path for
//! non-contiguous views. Structurally the same walk as `transpose` (which is
//! a materialize specialized to reversed strides), generalized to arbitrary
//! caller-supplied strides plus a base offset. The logical shape is
//! unchanged: only the layout becomes contiguous.

use crate::shape::{checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult};

pub fn materialize(shape: &[u32], strides: &[u32], offset: u32, data: &[f64]) -> KResult<(Vec<u32>, Vec<f64>)> {
    let size = checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;

    let logical_strides = compute_strides(shape);
    let mut out = vec![0f64; size as usize];
    for flat in 0..size {
        let idx = unravel(flat, shape, &logical_strides);
        let mut off: u32 = offset;
        for i in 0..shape.len() {
            off += idx[i] * strides[i];
        }
        out[flat as usize] = data.get(off as usize).copied().unwrap_or(0.0);
    }
    Ok((shape.to_vec(), out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::transpose::transpose;
    use crate::shape::KernelError;

    /// Materializing a transposed view must equal the transpose kernel's
    /// copy bit-for-bit (same gather, expressed via strides).
    #[test]
    fn transposed_view_matches_transpose_kernel() {
        let data: Vec<f64> = (0..24).map(|i| (i as f64) * 0.7 - 3.3).collect();
        let (t_shape, t_data) = transpose(&[2, 3, 4], &data).unwrap();
        // Transposed view of [2,3,4]: shape [4,3,2], strides reversed [1,4,12].
        let (m_shape, m_data) = materialize(&[4, 3, 2], &[1, 4, 12], 0, &data).unwrap();
        assert_eq!(t_shape, m_shape);
        assert_eq!(t_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   m_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    /// Contiguous metadata: materialize degenerates to an identity copy.
    #[test]
    fn contiguous_is_identity_copy() {
        let data = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let (shape, out) = materialize(&[2, 3], &[3, 1], 0, &data).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn offset_window() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (shape, out) = materialize(&[2, 2], &[2, 1], 1, &data).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn edge_cases_and_bounds() {
        // Rank-0 scalar view.
        let (shape0, out0) = materialize(&[], &[], 0, &[42.0]).unwrap();
        assert_eq!(shape0, Vec::<u32>::new());
        assert_eq!(out0, vec![42.0]);
        // Size-0 view.
        let (shape_e, out_e) = materialize(&[3, 0], &[0, 1], 0, &[]).unwrap();
        assert_eq!(shape_e, vec![3, 0]);
        assert_eq!(out_e, Vec::<f64>::new());
        // Out of bounds -> status 4.
        let err = materialize(&[2, 3], &[3, 1], 1, &[0.0; 6]).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
        // Rank cap still enforced.
        let shape = vec![1u32; 33];
        let strides = vec![0u32; 33];
        let err_rank = materialize(&shape, &strides, 0, &[0.0]).unwrap_err();
        assert_eq!(err_rank, KernelError::RankTooLarge);
    }
}
