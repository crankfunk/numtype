//! Reverse every axis (mirrors `transposeRuntime` in `spike/src/runtime.ts`).

use crate::shape::{checked_element_count, compute_strides, unravel, KResult};

pub fn transpose(shape: &[u32], data: &[f64]) -> KResult<(Vec<u32>, Vec<f64>)> {
    let size = checked_element_count(shape)?;
    let rank = shape.len();
    let mut out_shape = shape.to_vec();
    out_shape.reverse();
    let in_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);

    let mut out = vec![0f64; size as usize];
    for flat in 0..size {
        let out_idx = unravel(flat, &out_shape, &out_strides);
        let mut in_offset: u32 = 0;
        for i in 0..rank {
            let original_axis = rank - 1 - i;
            in_offset += out_idx[i] * in_strides[original_axis];
        }
        out[flat as usize] = data.get(in_offset as usize).copied().unwrap_or(0.0);
    }
    Ok((out_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_d() {
        // A = [[1,2,3],[4,5,6]] (2x3) -> transpose -> (3x2)
        // [[1,4],[2,5],[3,6]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (shape, data) = transpose(&[2, 3], &a).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn three_d() {
        let a: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let (shape, data) = transpose(&[2, 3, 4], &a).unwrap();
        assert_eq!(shape, vec![4, 3, 2]);
        // spot check: original[i][j][k] -> transposed[k][j][i]
        // original[1][2][3] = 1*12+2*4+3 = 23 -> transposed[3][2][1]
        let out_strides = [6, 2, 1]; // compute_strides([4,3,2])
        let out_off = 3 * out_strides[0] + 2 * out_strides[1] + 1 * out_strides[2];
        assert_eq!(data[out_off], 23.0);
    }

    #[test]
    fn rank0_scalar() {
        let (shape, data) = transpose(&[], &[42.0]).unwrap();
        assert_eq!(shape, Vec::<u32>::new());
        assert_eq!(data, vec![42.0]);
    }

    #[test]
    fn size_zero() {
        let (shape, data) = transpose(&[0, 3], &[]).unwrap();
        assert_eq!(shape, vec![3, 0]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn rank_too_large() {
        let shape = vec![1u32; 33];
        let err = transpose(&shape, &[0.0]).unwrap_err();
        assert_eq!(err.status(), 2);
    }
}
