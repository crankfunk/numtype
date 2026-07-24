//! WASM parity S5 (docs/wasm-parity-topk-spec.md): top-`k` values + indices
//! of a rank-1 strided view. A literal transliteration of `topkRuntime`
//! (spike/src/runtime.ts:689-791) — same validation order, same comparator,
//! same size-`k` bounded max-heap of "badness" (O(n log k)), same final
//! O(k log k) sort of the held indices, and the same "re-read `values[i]`
//! from the ORIGINAL data, never from the heap" data flow.
//!
//! **The algorithm is NOT the M1 obligation (spec, "Die geerbte Annahme
//! ... ist zu korrigieren").** `compare_values` combined with the trailing
//! `a_idx - b_idx` tiebreak is a STRICT TOTAL ORDER on the pairwise-distinct
//! logical indices `0..n-1`: the comparator returns 0 in exactly two cases
//! (both NaN, or numerically equal — `+0.0`/`-0.0` included, since plain
//! `>`/`<` are false in both directions), and in both of them the index
//! tiebreak is non-zero. So there is exactly ONE correct top-`k` index set in
//! exactly ONE order, and any correct selection algorithm must produce it.
//! Bit-identity to the TS reference therefore rests on exactly two things:
//!
//!  1. the ORDER PREDICATE being the same predicate (transliterated below —
//!     Rust's `f64` `>`/`<` and `f64::is_nan` are IEEE-754 exactly as JS's
//!     `>`/`<` and `Number.isNaN` are), and
//!  2. `values[i] = data[indices[i]]` staying a pure ELEMENT COPY that never
//!     routes a value through arithmetic (see `topk_strided`'s final loop).
//!
//! That is categorically unlike `sum`, where the accumulation ORDER changes
//! the result bits and the kernel must mirror the reference's loop shape:
//! `topk` performs NO floating-point arithmetic at all. The heap is chosen
//! here because it is the measured-better algorithm
//! (docs/op-topk-selection-ergebnisse.md) and lies in front of us directly
//! transliterable — not because M1 demands it. Consequently the tests below
//! pin only the OBSERVABLE result, never heap internals (sift order, root
//! choice, intermediate states).
//!
//! **The returned indices are LOGICAL indices `0..n-1`, never memory
//! offsets** — the same trap [`crate::kernels::argmax`]'s module doc
//! describes, sharpened here: a memory-index confusion would corrupt not just
//! `indices` but, through `values[i] = data[indices[i]]`, the VALUES too. The
//! strided read `data[offset + i * strides[0]]` happens inside `read_logical`
//! below and nowhere else; `topk_strided_strided_view_uses_logical_index`
//! pins it with an explicit non-vacuity assertion.
//!
//! **`unwrap_or(0.0)` convention — doubly load-bearing here (spec D3.6).**
//! For `sum` the fallback `0.0` is the additive neutral, a genuine no-op. For
//! `topk` the same `0.0` is BOTH a competing candidate in the total order AND
//! a value that would be copied into the output — so it is non-neutral twice
//! over. The path is unreachable: `validate_strided_bounds`
//! (crates/core/src/shape.rs:165-180) runs before any read and proves
//! `offset + (n − 1)·strides[0] < data.len()`. Correctness — not merely
//! memory safety — rests on that guarantee.

use crate::shape::{checked_element_count, validate_strided_bounds, KResult, KernelError};

/// `topk`'s selection-order comparator over VALUES only — the literal
/// transliteration of `topkCompareValues` (spike/src/runtime.ts:645): NaN
/// entries sort first; among two non-NaN entries the larger value sorts
/// first; a tie (two NaNs, or two numerically equal values — `0.0`/`-0.0`
/// included via plain `>`/`<`, never a bitwise or total-order comparison) is
/// left to the caller's index tiebreak. Returns a negative number when `a`
/// should sort before `b`, mirroring the JS `Array.prototype.sort`
/// comparator contract the TS original is written against.
fn compare_values(a: f64, b: f64) -> i32 {
    let a_nan = a.is_nan();
    let b_nan = b.is_nan();
    if a_nan && b_nan {
        return 0;
    }
    if a_nan {
        return -1;
    }
    if b_nan {
        return 1;
    }
    if a > b {
        return -1;
    }
    if a < b {
        return 1;
    }
    0
}

/// The full order: `compare_values` first, ties broken by ASCENDING logical
/// index — the transliteration of the TS `cmp` closure's
/// `topkCompareValues(aVal, bVal) || aIdx - bIdx` (runtime.ts:718-719). The
/// subtraction is done in `i64` so it is exact for every `u32` index pair,
/// exactly as the TS original's f64 subtraction is exact for every index
/// below `2^53`.
fn cmp(a_val: f64, a_idx: u32, b_val: f64, b_idx: u32) -> i64 {
    let by_value = compare_values(a_val, b_val);
    if by_value != 0 {
        return by_value as i64;
    }
    a_idx as i64 - b_idx as i64
}

/// Read the element at LOGICAL index `i` of a rank-1 strided view. THE ONLY
/// place this kernel touches `data` — see the module doc on why a memory-vs-
/// logical index confusion here would corrupt both outputs. The `u32`
/// arithmetic cannot overflow for `i < n`: `validate_strided_bounds` has
/// already proven `offset + (n − 1)·stride < data.len() <= u32::MAX`.
fn read_logical(data: &[f64], offset: u32, stride: u32, i: u32) -> f64 {
    data.get((offset + i * stride) as usize).copied().unwrap_or(0.0)
}

/// Top-`k` values + indices of a rank-1 strided view, in the pinned total
/// order (module doc). Returns `(values, indices)`; `indices` carries f64-
/// encoded LOGICAL indices, because this library is f64-only throughout and
/// every index here stays far below `2^53`.
///
/// Validation order mirrors `topkRuntime` (runtime.ts:694-703) exactly:
/// rank first, then `k` against the vector's length. (`topkRuntime`'s middle
/// check — `k` a non-negative integer — has no counterpart in the `u32` ABI
/// domain, where it holds by construction.) A rank other than 1, or `k > n`,
/// is `KernelError::ShapeIncompatible`; `WNDArray.topk` prevalidates both in
/// TS so the thrown message stems match `topkRuntime`'s word for word, and
/// these branches are defense in depth. `k = 0` is VALID and yields two empty
/// outputs — never an error, on any receiver including a size-0 one.
pub fn topk_strided(shape: &[u32], strides: &[u32], offset: u32, data: &[f64], k: u32) -> KResult<(Vec<f64>, Vec<f64>)> {
    checked_element_count(shape)?;
    validate_strided_bounds(shape, strides, offset, data.len() as u32)?;
    if shape.len() != 1 {
        return Err(KernelError::ShapeIncompatible);
    }
    let n = shape[0];
    if k > n {
        return Err(KernelError::ShapeIncompatible);
    }
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let stride = strides[0];

    // Size-`k` max-heap of "badness": the root holds the WORST of the `k`
    // currently held elements under the shared total order, so a candidate
    // that ranks BEFORE the root evicts it and resifts. Two parallel vectors
    // (values + source indices) rather than a heap of pairs — the observable
    // semantics and the O(n log k) complexity are the binding parts, not the
    // representation (runtime.ts:708-716).
    let ks = k as usize;
    let mut heap_val = vec![0f64; ks];
    let mut heap_idx = vec![0u32; ks];
    let mut size: usize = 0;

    for i in 0..n {
        let v = read_logical(data, offset, stride, i);
        if size < ks {
            heap_val[size] = v;
            heap_idx[size] = i;
            // sift up (runtime.ts:721-737)
            let mut j = size;
            while j > 0 {
                let parent = (j - 1) / 2;
                if cmp(heap_val[j], heap_idx[j], heap_val[parent], heap_idx[parent]) > 0 {
                    heap_val.swap(j, parent);
                    heap_idx.swap(j, parent);
                    j = parent;
                } else {
                    break;
                }
            }
            size += 1;
        } else if cmp(v, i, heap_val[0], heap_idx[0]) < 0 {
            heap_val[0] = v;
            heap_idx[0] = i;
            // sift down (runtime.ts:739-756)
            let mut j = 0usize;
            loop {
                let l = 2 * j + 1;
                let r = 2 * j + 2;
                let mut worst = j;
                if l < size && cmp(heap_val[l], heap_idx[l], heap_val[worst], heap_idx[worst]) > 0 {
                    worst = l;
                }
                if r < size && cmp(heap_val[r], heap_idx[r], heap_val[worst], heap_idx[worst]) > 0 {
                    worst = r;
                }
                if worst == j {
                    break;
                }
                heap_val.swap(j, worst);
                heap_idx.swap(j, worst);
                j = worst;
            }
        }
    }

    // Final O(k log k) sort of the held indices under the SAME order, with
    // every value RE-READ from the view (never a cached heap value) — the
    // transliteration of runtime.ts:776-781. `sort_by` is a comparison sort
    // and the order is a strict total order on distinct indices, so its
    // result is the unique correct one regardless of the algorithm it uses
    // internally (module doc).
    let mut order: Vec<u32> = heap_idx[..size].to_vec();
    order.sort_by(|&ia, &ib| {
        cmp(
            read_logical(data, offset, stride, ia),
            ia,
            read_logical(data, offset, stride, ib),
            ib,
        )
        .cmp(&0)
    });

    // `values[i]` is a PURE ELEMENT COPY out of the view — never a heap
    // value, never the result of any arithmetic (runtime.ts:785-789 does
    // exactly this and says why). That is what makes a NaN's exact bit
    // payload survive: an f64 load/store preserves payloads, only arithmetic
    // may canonicalize.
    let mut values = vec![0f64; size];
    let mut indices = vec![0f64; size];
    for (i, &src) in order.iter().enumerate() {
        indices[i] = src as f64;
        values[i] = read_logical(data, offset, stride, src);
    }
    Ok((values, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact comparison — strictly stronger than `==` for NaN payloads
    /// and `+0.0`/`-0.0`, both of which this kernel's order is defined on.
    fn assert_bits_eq(actual: &[f64], expected: &[f64], ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "{ctx}: element {i}: got {a} (0x{:x}) want {e} (0x{:x})", a.to_bits(), e.to_bits());
        }
    }

    #[test]
    fn topk_contiguous_matches_hand_reference() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // Descending: 9(5), 6(7), 5(4), 4(2), 3(0), 2(6), 1(1), 1(3).
        let (values, indices) = topk_strided(&[8], &[1], 0, &data, 3).unwrap();
        assert_bits_eq(&values, &[9.0, 6.0, 5.0], "top3 values");
        assert_bits_eq(&indices, &[5.0, 7.0, 4.0], "top3 indices");
    }

    /// The index-domain counterpart of `argmax`'s own view guard: on a
    /// strided view the answer must be indices into the view's LOGICAL
    /// flattening, which here provably differs from the memory offsets.
    #[test]
    fn topk_strided_view_uses_logical_index() {
        // Base buffer of 6; the view takes every 2nd element: [10, 30, 50].
        let data = vec![10.0, 999.0, 30.0, 998.0, 50.0, 997.0];
        let (values, indices) = topk_strided(&[3], &[2], 0, &data, 2).unwrap();
        assert_bits_eq(&values, &[50.0, 30.0], "strided view values");
        // LOGICAL indices 2 and 1 — the memory offsets would be 4 and 2.
        assert_bits_eq(&indices, &[2.0, 1.0], "strided view indices");
        // Non-vacuity: reading the same buffer contiguously gives a
        // genuinely different answer, so this pins the LOGICAL index, not
        // merely "an" index. It also proves the VALUES would be wrong under
        // a memory-index confusion (999/998, not 50/30).
        let (values_flat, indices_flat) = topk_strided(&[6], &[1], 0, &data, 2).unwrap();
        assert_bits_eq(&values_flat, &[999.0, 998.0], "contiguous control values");
        assert_bits_eq(&indices_flat, &[1.0, 3.0], "contiguous control indices");
        assert_ne!(indices, indices_flat);
        assert_ne!(values, values_flat);
    }

    #[test]
    fn topk_offset_window() {
        let data = vec![7.0, 1.0, 2.0, 3.0];
        // Window over data[1..4] = [1, 2, 3] -> top2 is 3 (logical 2), 2 (logical 1).
        let (values, indices) = topk_strided(&[3], &[1], 1, &data, 2).unwrap();
        assert_bits_eq(&values, &[3.0, 2.0], "offset window values");
        assert_bits_eq(&indices, &[2.0, 1.0], "offset window indices");
        // Out-of-bounds offset is rejected before any read.
        assert_eq!(topk_strided(&[3], &[1], 2, &data, 2).unwrap_err(), KernelError::StridesOutOfBounds);
    }

    #[test]
    fn topk_k_zero_is_empty_never_an_error() {
        let data = vec![1.0, 2.0, 3.0];
        let (values, indices) = topk_strided(&[3], &[1], 0, &data, 0).unwrap();
        assert!(values.is_empty() && indices.is_empty());
        // n = 0, k = 0: still valid, still empty.
        let (v0, i0) = topk_strided(&[0], &[1], 0, &[], 0).unwrap();
        assert!(v0.is_empty() && i0.is_empty());
    }

    #[test]
    fn topk_k_equals_n_is_the_whole_vector_sorted() {
        let data = vec![2.0, 5.0, 1.0, 4.0, 3.0];
        let (values, indices) = topk_strided(&[5], &[1], 0, &data, 5).unwrap();
        assert_bits_eq(&values, &[5.0, 4.0, 3.0, 2.0, 1.0], "k=n values");
        assert_bits_eq(&indices, &[1.0, 3.0, 4.0, 0.0, 2.0], "k=n indices");
    }

    #[test]
    fn topk_k_one_is_the_maximum() {
        let data = vec![2.0, 5.0, 1.0, 4.0, 3.0];
        let (values, indices) = topk_strided(&[5], &[1], 0, &data, 1).unwrap();
        assert_bits_eq(&values, &[5.0], "k=1 values");
        assert_bits_eq(&indices, &[1.0], "k=1 indices");
    }

    #[test]
    fn topk_nan_sorts_first_ascending_index_among_nans() {
        // A single NaN outranks +Inf.
        let one = vec![1.0, f64::INFINITY, f64::NAN, 2.0];
        let (_, idx_one) = topk_strided(&[4], &[1], 0, &one, 2).unwrap();
        assert_bits_eq(&idx_one, &[2.0, 1.0], "single NaN first, then +Inf");
        // Several NaNs: among themselves by ASCENDING index, all ahead of
        // every real number.
        let many = vec![5.0, f64::NAN, 7.0, f64::NAN, 6.0];
        let (_, idx_many) = topk_strided(&[5], &[1], 0, &many, 4).unwrap();
        assert_bits_eq(&idx_many, &[1.0, 3.0, 2.0, 4.0], "NaNs by ascending index, then 7, then 6");
    }

    #[test]
    fn topk_signed_zero_tie_breaks_by_ascending_index() {
        // `+0.0` and `-0.0` are EQUAL under plain `>`/`<`, so only the index
        // decides — and the returned VALUES keep their own signs bit-exactly.
        let data = vec![-0.0, 0.0, -1.0];
        let (values, indices) = topk_strided(&[3], &[1], 0, &data, 2).unwrap();
        assert_bits_eq(&indices, &[0.0, 1.0], "signed-zero tie -> ascending index");
        assert_bits_eq(&values, &[-0.0, 0.0], "signed-zero values keep their own bits");
        let flipped = vec![0.0, -0.0, -1.0];
        let (values_f, indices_f) = topk_strided(&[3], &[1], 0, &flipped, 2).unwrap();
        assert_bits_eq(&indices_f, &[0.0, 1.0], "reversed signed-zero tie -> still ascending index");
        assert_bits_eq(&values_f, &[0.0, -0.0], "reversed signed-zero values keep their own bits");
    }

    #[test]
    fn topk_value_ties_break_by_ascending_index_including_across_the_k_boundary() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (values, indices) = topk_strided(&[5], &[1], 0, &data, 3).unwrap();
        assert_bits_eq(&values, &[5.0, 5.0, 5.0], "all-equal values");
        // The k boundary runs THROUGH the tie group: only the index tiebreak
        // decides which three survive.
        assert_bits_eq(&indices, &[0.0, 1.0, 2.0], "all-equal -> the three lowest indices");
    }

    #[test]
    fn topk_infinities() {
        let data = vec![f64::NEG_INFINITY, 0.0, f64::INFINITY, -1.0];
        let (values, indices) = topk_strided(&[4], &[1], 0, &data, 4).unwrap();
        assert_bits_eq(&values, &[f64::INFINITY, 0.0, -1.0, f64::NEG_INFINITY], "infinity ordering values");
        assert_bits_eq(&indices, &[2.0, 1.0, 3.0, 0.0], "infinity ordering indices");
    }

    #[test]
    fn topk_rank_not_one_is_shape_incompatible() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(topk_strided(&[2, 3], &[3, 1], 0, &data, 2).unwrap_err(), KernelError::ShapeIncompatible);
        assert_eq!(topk_strided(&[], &[], 0, &data, 1).unwrap_err(), KernelError::ShapeIncompatible);
        // …even for k = 0, where the result would otherwise be trivially empty.
        assert_eq!(topk_strided(&[2, 3], &[3, 1], 0, &data, 0).unwrap_err(), KernelError::ShapeIncompatible);
    }

    #[test]
    fn topk_k_greater_than_n_is_shape_incompatible() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(topk_strided(&[3], &[1], 0, &data, 4).unwrap_err(), KernelError::ShapeIncompatible);
        assert_eq!(topk_strided(&[0], &[1], 0, &[], 1).unwrap_err(), KernelError::ShapeIncompatible);
    }

    #[test]
    fn topk_rank_too_large() {
        let shape = vec![1u32; 33];
        assert_eq!(topk_strided(&shape, &shape, 0, &[0.0], 1).unwrap_err(), KernelError::RankTooLarge);
    }

    /// A non-canonical NaN payload must come back byte-identical: the value
    /// path is a pure load/store, never arithmetic (module doc, point 2).
    #[test]
    fn topk_preserves_a_non_canonical_nan_payload() {
        let weird = f64::from_bits(0x7ff8_0000_cafe_babe);
        assert!(weird.is_nan());
        assert_ne!(weird.to_bits(), f64::NAN.to_bits());
        let data = vec![1.0, weird, 2.0];
        let (values, indices) = topk_strided(&[3], &[1], 0, &data, 3).unwrap();
        assert_eq!(values[0].to_bits(), 0x7ff8_0000_cafe_babe, "the exact payload must survive the copy");
        assert_bits_eq(&indices, &[1.0, 2.0, 0.0], "NaN first, then 2, then 1");
    }
}
