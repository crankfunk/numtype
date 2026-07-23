//! WASM parity S1 (docs/wasm-parity-scalar-spec.md): scalar elementwise ops
//! (`add`/`sub`/`mul`/`div` with a constant `f64` operand) over a strided
//! view. Each op is a one-liner over the S0 `unary_strided` generic core
//! (`kernels::sqrt`) — no new iteration kernel, no duplication (M1
//! structural identity: there is exactly one unary strided kernel).
//!
//! Bit parity is a COROLLAR of already-frozen facts, not a new empirical
//! claim (spec's "tragende Beobachtung"): IEEE 754 §5.4.1 / ECMA-262 both
//! demand the correctly-rounded result for `+`/`-`/`*`/`/` — exactly the
//! property the frozen binary kernels (`nt_add_strided` & co., Kern 07)
//! already prove bit-identical to the naive JS reference. Scalar `x op s` is
//! the `y := s` (constant) specialization of the same binary op, so it
//! inherits the same bit-identity. Operand order is pinned: sub is
//! `data[i] - s` (NOT `s - data[i]`), div is `data[i] / s` (NOT `s /
//! data[i]`) — exactly `scalarElementwiseRuntime`'s own closures
//! (spike/src/runtime.ts:816-833). No guard on the scalar: every
//! finite/non-finite `f64` is valid, IEEE propagation only (`x/0 -> +/-Inf`,
//! `0/0 -> NaN`, signed zero/Infinity propagate per the standard).

use crate::kernels::sqrt::unary_strided;
use crate::shape::KResult;

/// Elementwise `data[i] + s` over a strided view (shape-preserving, no
/// broadcast). Operand order: `data[i] + s` (commutative, but stated for
/// symmetry with sub/div below).
pub fn scalar_add_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], s: f64) -> KResult<(Vec<u32>, Vec<f64>)> {
    unary_strided(shape, strides, offset, data, |x| x + s)
}

/// Elementwise `data[i] - s` over a strided view (shape-preserving, no
/// broadcast). Operand order PINNED: `data[i] - s`, NOT `s - data[i]`.
pub fn scalar_sub_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], s: f64) -> KResult<(Vec<u32>, Vec<f64>)> {
    unary_strided(shape, strides, offset, data, |x| x - s)
}

/// Elementwise `data[i] * s` over a strided view (shape-preserving, no
/// broadcast). Operand order: `data[i] * s` (commutative, stated for
/// symmetry with sub/div above/below).
pub fn scalar_mul_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], s: f64) -> KResult<(Vec<u32>, Vec<f64>)> {
    unary_strided(shape, strides, offset, data, |x| x * s)
}

/// Elementwise `data[i] / s` over a strided view (shape-preserving, no
/// broadcast). Operand order PINNED: `data[i] / s`, NOT `s / data[i]`. Pure
/// IEEE 754: no zero-divisor guard — `x/0 -> +/-Infinity`, `0/0 -> NaN`.
pub fn scalar_div_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], s: f64) -> KResult<(Vec<u32>, Vec<f64>)> {
    unary_strided(shape, strides, offset, data, |x| x / s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{compute_strides, KernelError};

    // -------------------------------------------------------------------
    // (i) Shared strided machinery, proven once via `add` (same generic core
    // as sqrt — same-shape, transposed view, offset window, size-0 array,
    // rank-too-large, and the fast-path-vs-general-path equivalence proof on
    // a transposed view, including special-value inputs).
    // -------------------------------------------------------------------

    #[test]
    fn add_same_shape() {
        let (shape, data) = scalar_add_strided(&[2, 2], &[2, 1], 0, &[1.0, 2.0, 3.0, 4.0], 10.0).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![11.0, 12.0, 13.0, 14.0]);
    }

    /// Transposed view operand: A^T (view of [2,3] as [3,2], strides [1,3]).
    #[test]
    fn add_transposed_view_operand() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let (shape, data) = scalar_add_strided(&[3, 2], &[1, 3], 0, &a, 100.0).unwrap();
        assert_eq!(shape, vec![3, 2]);
        // A^T = [[1,4],[2,5],[3,6]] + 100
        assert_eq!(data, vec![101.0, 104.0, 102.0, 105.0, 103.0, 106.0]);
    }

    #[test]
    fn add_offset_window() {
        let buf = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (shape, data) = scalar_add_strided(&[3], &[1], 2, &buf, 1.0).unwrap();
        assert_eq!(shape, vec![3]);
        // buf[2..5] = [2,3,4] + 1
        assert_eq!(data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn add_size_zero_array() {
        let (shape, data) = scalar_add_strided(&[0, 3], &[3, 1], 0, &[], 5.0).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn add_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = scalar_add_strided(&shape, &shape, 0, &[0.0], 1.0).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn add_strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let err = scalar_add_strided(&[2, 3], &[3, 2], 0, &a, 1.0).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    /// Kern-11-style equivalence proof (mirrors sqrt.rs's own
    /// `sqrt_fast_path_matches_general_path_transposed_view`): a `[3,2]`
    /// buffer `raw`'s transposed VIEW (shape `[2,3]`, strides `[1,3]`, offset
    /// 0) reads the exact same logical values as a separately-materialized
    /// `[2,3]` contiguous buffer. Adding a scalar to the materialized buffer
    /// hits the new fast path (natural strides, offset 0); adding through the
    /// transposed view fails only the natural-strides check, forcing the
    /// untouched general `unravel`-based loop. Includes special-value inputs
    /// (NaN, negative, ±0) so the fast path is proven not to diverge on IEEE
    /// edge propagation.
    #[test]
    fn add_fast_path_matches_general_path_transposed_view() {
        // raw = [3,2] row-major: [[0,5],[-3,-0.0],[7,f64::NAN]]
        let raw = vec![0.0, 5.0, -3.0, -0.0, 7.0, f64::NAN];
        // Materialized transpose of `raw`, shape [2,3], natural strides:
        // [[0,-3,7],[5,-0.0,NaN]]
        let materialized = vec![0.0, -3.0, 7.0, 5.0, -0.0, f64::NAN];
        let shape = [2u32, 3u32];
        let natural_strides = compute_strides(&shape);
        let transposed_strides = [1u32, 2u32];
        let s = 2.5;

        let (fast_shape, fast_data) = scalar_add_strided(&shape, &natural_strides, 0, &materialized, s).unwrap();
        let (gen_shape, gen_data) = scalar_add_strided(&shape, &transposed_strides, 0, &raw, s).unwrap();

        assert_eq!(fast_shape, gen_shape);
        assert_eq!(fast_shape, shape.to_vec());
        for i in 0..fast_data.len() {
            let f = fast_data[i];
            let g = gen_data[i];
            if f.is_nan() {
                assert!(g.is_nan(), "index {i}: fast=NaN but general={g}");
            } else {
                assert_eq!(f.to_bits(), g.to_bits(), "index {i}: fast={f} general={g}");
            }
        }
    }

    // -------------------------------------------------------------------
    // (ii) Op-specific arithmetic, incl. operand-order proofs and div edges.
    // -------------------------------------------------------------------

    #[test]
    fn add_arithmetic() {
        let (_, data) = scalar_add_strided(&[1], &[1], 0, &[3.0], 5.0).unwrap();
        assert_eq!(data, vec![8.0]);
    }

    #[test]
    fn sub_arithmetic_and_operand_order() {
        // sub(5,3) == 2 (basic), sub(3,5) == -2 (proves data - s, not s - data).
        let (_, data_a) = scalar_sub_strided(&[1], &[1], 0, &[5.0], 3.0).unwrap();
        assert_eq!(data_a, vec![2.0]);
        let (_, data_b) = scalar_sub_strided(&[1], &[1], 0, &[3.0], 5.0).unwrap();
        assert_eq!(data_b, vec![-2.0]);
    }

    #[test]
    fn mul_arithmetic() {
        let (_, data) = scalar_mul_strided(&[1], &[1], 0, &[4.0], 2.5).unwrap();
        assert_eq!(data, vec![10.0]);
    }

    #[test]
    fn div_arithmetic_and_edges() {
        // div(6,2) == 3 (basic).
        let (_, data) = scalar_div_strided(&[1], &[1], 0, &[6.0], 2.0).unwrap();
        assert_eq!(data, vec![3.0]);

        // div(1,0) == +Inf, div(-1,0) == -Inf, div(0,0) is NaN — pure IEEE,
        // no zero-divisor guard.
        let (_, pos_inf) = scalar_div_strided(&[1], &[1], 0, &[1.0], 0.0).unwrap();
        assert_eq!(pos_inf[0], f64::INFINITY);
        let (_, neg_inf) = scalar_div_strided(&[1], &[1], 0, &[-1.0], 0.0).unwrap();
        assert_eq!(neg_inf[0], f64::NEG_INFINITY);
        let (_, nan) = scalar_div_strided(&[1], &[1], 0, &[0.0], 0.0).unwrap();
        assert!(nan[0].is_nan());
    }

    // -------------------------------------------------------------------
    // (iii) One special-value edge test per op.
    // -------------------------------------------------------------------

    #[test]
    fn add_special_value_edges() {
        // NaN scalar propagates.
        let (_, nan_scalar) = scalar_add_strided(&[1], &[1], 0, &[1.0], f64::NAN).unwrap();
        assert!(nan_scalar[0].is_nan());
        // x + (-0.0) is bit-correct (x unchanged for finite nonzero x).
        let (_, plus_neg_zero) = scalar_add_strided(&[1], &[1], 0, &[5.0], -0.0).unwrap();
        assert_eq!(plus_neg_zero[0], 5.0);
        // (-0.0) + (-0.0) == -0.0, bit-exact.
        let (_, neg_zero_sum) = scalar_add_strided(&[1], &[1], 0, &[-0.0], -0.0).unwrap();
        assert_eq!(neg_zero_sum[0].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn sub_special_value_edges() {
        let (_, nan_scalar) = scalar_sub_strided(&[1], &[1], 0, &[1.0], f64::NAN).unwrap();
        assert!(nan_scalar[0].is_nan());
        // 0.0 - 0.0 == +0.0 (not -0.0), bit-exact IEEE rule.
        let (_, zero_sub) = scalar_sub_strided(&[1], &[1], 0, &[0.0], 0.0).unwrap();
        assert_eq!(zero_sub[0].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn mul_special_value_edges() {
        let (_, nan_scalar) = scalar_mul_strided(&[1], &[1], 0, &[1.0], f64::NAN).unwrap();
        assert!(nan_scalar[0].is_nan());
        // Infinity * 0 is NaN.
        let (_, inf_times_zero) = scalar_mul_strided(&[1], &[1], 0, &[f64::INFINITY], 0.0).unwrap();
        assert!(inf_times_zero[0].is_nan());
        // (-0.0) * finite-positive == -0.0, bit-exact.
        let (_, neg_zero_mul) = scalar_mul_strided(&[1], &[1], 0, &[-0.0], 3.0).unwrap();
        assert_eq!(neg_zero_mul[0].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn div_special_value_edges() {
        let (_, nan_scalar) = scalar_div_strided(&[1], &[1], 0, &[1.0], f64::NAN).unwrap();
        assert!(nan_scalar[0].is_nan());
        // Infinity / Infinity is NaN.
        let (_, inf_over_inf) = scalar_div_strided(&[1], &[1], 0, &[f64::INFINITY], f64::INFINITY).unwrap();
        assert!(inf_over_inf[0].is_nan());
    }
}
