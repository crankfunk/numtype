//! WASM parity S0 (docs/wasm-parity-sqrt-spec.md): elementwise square root
//! over a strided view. Structural unary counterpart of
//! `kernels::elementwise::binary_strided` — same validation order, same
//! iteration order, same offset algebra, generalized over a UNARY `f64` op
//! instead of a binary one (no broadcast: output shape == input shape).
//!
//! `f64::sqrt` is the IEEE-754 correctly-rounded square root, so it is
//! bit-identical to `Math.sqrt` (ECMA-262 `sec-math.sqrt` demands the same
//! correctly-rounded contract) — the spec's Baustein-0 scratch probe
//! confirmed this empirically over 30,028 inputs (all special edges plus the
//! full exponent spectrum) before this file was written; the differential
//! test suite (`spike/tests-runtime/elementwise.test.ts`,
//! `special-values.test.ts`) confirms it again against the real kernel.

use crate::shape::{aligned_effective_strides, checked_element_count, compute_strides, unravel, validate_strided_bounds, KResult};

/// Generic core: unary counterpart of
/// [`crate::kernels::elementwise::binary_strided`] — identical structure, ONE
/// operand, `F: Fn(f64) -> f64` instead of a binary op.
///
/// WASM parity S1 (docs/wasm-parity-scalar-spec.md, D2/M4): widened from
/// private `fn` to `pub(crate) fn` so `kernels::scalar` can reuse it for the
/// four scalar ops instead of duplicating the iteration kernel — a purely
/// additive visibility change (grants access, removes nothing); the
/// monomorphization at THIS file's own `|x| x.sqrt()` call site is
/// unaffected, so `sqrt_strided`'s codegen/behavior is unchanged (confirmed
/// by a clean-rebuild byte-identical artifact hash, spec's Baustein-0 probe).
pub(crate) fn unary_strided<F: Fn(f64) -> f64>(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], op: F) -> KResult<(Vec<u32>, Vec<f64>)> {
    checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;

    // Shape-preserving (no broadcast): output shape is exactly the input
    // shape.
    let out_shape = shape;
    let rank = out_shape.len();
    let size = checked_element_count(out_shape)?;

    // Kern-11 contiguous fast path — see `binary_strided`'s identical block
    // (crates/core/src/kernels/elementwise.rs) for the full rationale.
    // Offset 0 and natural strides ⇒ `flat == effective offset` for every
    // element, so the general loop below reduces to `out[i] = op(data[i])`.
    // Checked AFTER all validations above, so error semantics are
    // unchanged. `validate_strided_bounds` guarantees `data.len() >= size`
    // here, so the direct slice indexing is in-bounds by construction.
    if offset == 0 && strides == compute_strides(shape) {
        let n = size as usize;
        let src = &data[..n];
        let mut out = vec![0f64; n];
        for i in 0..n {
            out[i] = op(src[i]);
        }
        return Ok((out_shape.to_vec(), out));
    }

    let eff = aligned_effective_strides(shape, strides, rank);
    let out_strides = compute_strides(out_shape);
    let mut out = vec![0f64; size as usize];

    // Same u32-offset-accumulation safety argument as binary_strided: every
    // partial sum is bounded by the operand's own validated max reach
    // (validate_strided_bounds above), so wraparound is structurally
    // unreachable.
    for flat in 0..size {
        let idx = unravel(flat, out_shape, &out_strides);
        let mut off: u32 = offset;
        for i in 0..rank {
            off += idx[i] * eff[i];
        }
        let val = data.get(off as usize).copied().unwrap_or(0.0);
        out[flat as usize] = op(val);
    }
    Ok((out_shape.to_vec(), out))
}

/// Elementwise square root over a strided view (shape-preserving, no
/// broadcast). `f64::sqrt` is the IEEE-754 correctly-rounded square root —
/// see the module doc comment for the M1 bit-parity argument.
pub fn sqrt_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64]) -> KResult<(Vec<u32>, Vec<f64>)> {
    unary_strided(shape, strides, offset, data, |x| x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::KernelError;

    #[test]
    fn sqrt_same_shape() {
        let (shape, data) = sqrt_strided(&[2, 2], &[2, 1], 0, &[1.0, 4.0, 9.0, 16.0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Transposed view operand: A^T (view of [2,3] as [3,2], strides [1,3]).
    #[test]
    fn sqrt_transposed_view_operand() {
        let a = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]; // [[1,4,9],[16,25,36]]
        let (shape, data) = sqrt_strided(&[3, 2], &[1, 3], 0, &a).unwrap();
        assert_eq!(shape, vec![3, 2]);
        // A^T = [[1,16],[4,25],[9,36]]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn sqrt_offset_window() {
        let buf = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let (shape, data) = sqrt_strided(&[3], &[1], 2, &buf).unwrap();
        assert_eq!(shape, vec![3]);
        // buf[2..5] = [4,9,16]
        assert_eq!(data, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn sqrt_size_zero_array() {
        let (shape, data) = sqrt_strided(&[0, 3], &[3, 1], 0, &[]).unwrap();
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn sqrt_rank_too_large() {
        let shape = vec![1u32; 33];
        let err = sqrt_strided(&shape, &shape, 0, &[0.0]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn sqrt_strided_out_of_bounds_is_status_4() {
        let a = vec![0.0; 6];
        let err = sqrt_strided(&[2, 3], &[3, 2], 0, &a).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    /// Kern-11-style equivalence proof (docs/kern-11-elementwise-fastpath-spec.md's
    /// D4 argument, unary form): a `[3,2]` buffer `raw`'s transposed VIEW
    /// (shape `[2,3]`, strides `[1,3]`, offset 0) reads the exact same
    /// logical values as a separately-materialized `[2,3]` contiguous
    /// buffer. Square-rooting the materialized buffer hits the new fast
    /// path (natural strides, offset 0); square-rooting through the
    /// transposed view fails only the natural-strides check (shape and
    /// offset are identical), forcing the untouched general `unravel`-based
    /// loop. Includes special-value inputs (NaN, negative, ±0) so the fast
    /// path is also proven not to diverge on IEEE edge propagation.
    #[test]
    fn sqrt_fast_path_matches_general_path_transposed_view() {
        // raw = [3,2] row-major: [[0,5],[-3,-0.0],[7,f64::NAN]]
        let raw = vec![0.0, 5.0, -3.0, -0.0, 7.0, f64::NAN];
        // Materialized transpose of `raw`, shape [2,3], natural strides:
        // [[0,-3,7],[5,-0.0,NaN]]
        let materialized = vec![0.0, -3.0, 7.0, 5.0, -0.0, f64::NAN];
        let shape = [2u32, 3u32];
        let natural_strides = compute_strides(&shape);
        // `raw` is a [3,2] buffer (natural strides [2,1]); reading it as a
        // [2,3] transposed view swaps both shape and strides -> [1,2].
        let transposed_strides = [1u32, 2u32];

        let (fast_shape, fast_data) = sqrt_strided(&shape, &natural_strides, 0, &materialized).unwrap();
        let (gen_shape, gen_data) = sqrt_strided(&shape, &transposed_strides, 0, &raw).unwrap();

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

        // Sanity: hand-derived expected values, including the special ones.
        assert_eq!(fast_data[0].to_bits(), 0.0f64.to_bits()); // sqrt(+0) == +0
        assert!(fast_data[1].is_nan()); // sqrt(-3.0)
        assert_eq!(fast_data[2], 7.0f64.sqrt());
        assert_eq!(fast_data[3], 5.0f64.sqrt());
        assert_eq!(fast_data[4].to_bits(), (-0.0f64).to_bits()); // sqrt(-0) == -0
        assert!(fast_data[5].is_nan()); // sqrt(NaN)
    }

    /// IEEE special-value edges pinned directly at the kernel level (M1,
    /// docs/wasm-parity-sqrt-spec.md "tragende Beobachtung"): `sqrt(-0) ==
    /// -0` (bit-exact, Object.is-equivalent), `sqrt(4) == 2`, `sqrt(neg)` is
    /// NaN, `sqrt(+Inf) == +Inf`, `sqrt(NaN)` is NaN, and a genuine
    /// subnormal input square-roots into a normal finite positive value
    /// (no flush-to-zero).
    #[test]
    fn sqrt_special_values() {
        let subnormal = f64::MIN_POSITIVE / 2.0; // genuine subnormal (< 2.2250738585072014e-308)
        assert!(subnormal > 0.0 && subnormal < f64::MIN_POSITIVE, "sanity: constructed value must be a positive subnormal");
        let data = vec![-0.0, 4.0, -1.0, f64::INFINITY, f64::NAN, subnormal];
        let shape = [data.len() as u32];
        let strides = compute_strides(&shape);
        let (out_shape, out) = sqrt_strided(&shape, &strides, 0, &data).unwrap();
        assert_eq!(out_shape, vec![data.len() as u32]);

        assert_eq!(out[0].to_bits(), (-0.0f64).to_bits(), "sqrt(-0) must be -0, bit-exact");
        assert_eq!(out[1], 2.0, "sqrt(4) == 2");
        assert!(out[2].is_nan(), "sqrt(negative) is NaN");
        assert_eq!(out[3], f64::INFINITY, "sqrt(+Inf) == +Inf");
        assert!(out[4].is_nan(), "sqrt(NaN) is NaN");
        assert!(out[5].is_finite() && out[5] > 0.0, "sqrt of a positive subnormal is a normal positive finite value");
        assert_eq!(out[5].to_bits(), subnormal.sqrt().to_bits(), "subnormal input: sqrt matches direct f64::sqrt bit-for-bit");
    }
}
