//! Shape/stride primitives shared by every kernel. Each function here is a
//! direct, line-by-line port of the corresponding helper in
//! `spike/src/runtime.ts` — same iteration order, same edge-case handling —
//! so that floating-point accumulation order (and therefore bit-identity)
//! matches exactly. See that file for the TS original of each function
//! (docstrings below point at the specific TS function mirrored).
//!
//! Rank is capped at 32 (`MAX_RANK`) per the ABI spec. All dims/strides/
//! offsets are `u32`, matching the WASM32 address space and the ABI's
//! little-endian `u32` shape arrays.

/// Maximum supported rank (spec: "rank ≤ 32").
pub const MAX_RANK: usize = 32;

/// The three failure categories the ABI can report (status codes 1..3;
/// 0 is success and isn't represented here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// Status 1: operand shapes are not compatible for this op (broadcast
    /// mismatch, matmul inner-dim mismatch, out-of-range reduce axis, or a
    /// promotion-contract violation for matmul).
    ShapeIncompatible,
    /// Status 2: a supplied shape's rank exceeds `MAX_RANK`.
    RankTooLarge,
    /// Status 3: the element count (or byte size) implied by a shape does
    /// not fit in `u32` — includes the caller-supplied output buffer length
    /// not matching the kernel's independently computed output length,
    /// which doubles as a safety guard against writing past the buffer and
    /// as the differential-testing signal for TS/Rust shape-logic
    /// divergence (see ABI doc in docs/kern-01-ergebnisse.md).
    SizeOverflow,
    /// Status 4 (Kern 03): a strided operand's reachable index range
    /// (`offset + Σ (dim_i − 1)·stride_i`) exceeds its buffer length — or
    /// overflows entirely. Caller-supplied strides are the first ABI input
    /// whose validity the kernels cannot derive themselves, so they are
    /// bounds-checked before any data access (see `validate_strided_bounds`).
    StridesOutOfBounds,
}

impl KernelError {
    pub fn status(self) -> u32 {
        match self {
            KernelError::ShapeIncompatible => 1,
            KernelError::RankTooLarge => 2,
            KernelError::SizeOverflow => 3,
            KernelError::StridesOutOfBounds => 4,
        }
    }
}

pub type KResult<T> = Result<T, KernelError>;

/// Validate a shape's rank and compute its element count, checking for
/// overflow in both the element-count (`u32`) and byte-size (`elements * 8`,
/// also `u32`) domains. This is the single validation entry point every
/// kernel calls on every operand shape *before* doing any shape arithmetic
/// with it — mirrors nothing in `runtime.ts` directly (TS numbers don't
/// overflow at these magnitudes) but is required because Rust offsets here
/// are `u32`.
pub fn checked_element_count(shape: &[u32]) -> KResult<u32> {
    if shape.len() > MAX_RANK {
        return Err(KernelError::RankTooLarge);
    }
    let mut acc: u64 = 1;
    for &d in shape {
        acc *= d as u64;
        if acc > u32::MAX as u64 {
            return Err(KernelError::SizeOverflow);
        }
    }
    if acc * 8 > u32::MAX as u64 {
        return Err(KernelError::SizeOverflow);
    }
    Ok(acc as u32)
}

/// Product of a shape's dims, no validation. Only ever called after the
/// shape has already passed `checked_element_count` (directly, or because
/// it's a sub-shape of one that has — see call sites), so overflow is
/// structurally unreachable; mirrors `product` in runtime.ts.
pub fn product(shape: &[u32]) -> u32 {
    shape.iter().fold(1u32, |acc, &d| acc.wrapping_mul(d))
}

/// Row-major (C-contiguous) strides for a shape. Mirrors `computeStrides`.
/// Safety precondition: `shape` must already have passed
/// `checked_element_count` (or be a subshape of one that has) so the
/// running product never overflows `u32`.
pub fn compute_strides(shape: &[u32]) -> Vec<u32> {
    let mut strides = vec![0u32; shape.len()];
    let mut acc: u32 = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.wrapping_mul(shape[i]);
    }
    strides
}

/// Decode a flat row-major index into per-axis indices. Mirrors `unravel`.
/// Only ever called with `flat < product(shape)`, which (see `compute_strides`
/// docs) guarantees every `strides[i]` used here is nonzero whenever the
/// surrounding loop actually runs (a zero stride can only arise from a later
/// dim being 0, which forces the total product — and hence the loop bound —
/// to 0 too).
pub fn unravel(flat: u32, shape: &[u32], strides: &[u32]) -> Vec<u32> {
    shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| (flat / strides[i]) % dim)
        .collect()
}

/// NumPy broadcast of two shapes (mirrors `runtimeBroadcastShape`). Does NOT
/// itself re-check rank — callers must have already validated both `a` and
/// `b` individually via `checked_element_count` (which guarantees
/// `max(a.len(), b.len()) <= MAX_RANK` here, since both operands are
/// individually `<= MAX_RANK`).
pub fn broadcast_shape(a: &[u32], b: &[u32]) -> KResult<Vec<u32>> {
    let rank = a.len().max(b.len());
    let mut result = vec![0u32; rank];
    for i in 0..rank {
        let ai = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bi = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if ai == bi || ai == 1 || bi == 1 {
            result[rank - 1 - i] = if ai == 1 { bi } else { ai };
        } else {
            return Err(KernelError::ShapeIncompatible);
        }
    }
    Ok(result)
}

/// Pad a shape/strides pair with leading (dim=1, stride=0) axes up to
/// `rank`. Mirrors `alignToRank`.
pub fn align_to_rank(shape: &[u32], strides: &[u32], rank: usize) -> (Vec<u32>, Vec<u32>) {
    let pad = rank - shape.len();
    let mut out_shape = vec![1u32; pad];
    out_shape.extend_from_slice(shape);
    let mut out_strides = vec![0u32; pad];
    out_strides.extend_from_slice(strides);
    (out_shape, out_strides)
}

/// Effective strides for indexing under broadcasting: any axis of size 1 in
/// the (rank-aligned) source contributes stride 0, regardless of its
/// naturally-computed stride. Mirrors `effectiveStrides`.
pub fn effective_strides(aligned_shape: &[u32], aligned_strides: &[u32]) -> Vec<u32> {
    aligned_shape
        .iter()
        .zip(aligned_strides.iter())
        .map(|(&d, &s)| if d == 1 { 0 } else { s })
        .collect()
}

/// Kern 03: validate a strided operand before any data access. Checks that
/// the highest element index the (shape, strides, offset) triple can ever
/// reach — `offset + Σ (dim_i − 1)·stride_i` — fits inside `data_len`.
/// All arithmetic is checked `u64`; any overflow is reported as
/// `StridesOutOfBounds` too (caller-supplied strides are untrusted input).
/// A logically size-0 view (any dim == 0) never reads and passes vacuously,
/// matching the ABI's existing "zero-length is valid regardless of pointer"
/// rule. Rank must already have been validated via `checked_element_count`;
/// `strides.len() == shape.len()` is the ABI contract (same `rank` argument
/// covers both arrays).
pub fn validate_strided_bounds(shape: &[u32], strides: &[u32], offset: u32, data_len: u32) -> KResult<()> {
    if shape.iter().any(|&d| d == 0) {
        return Ok(());
    }
    let mut max_reach: u64 = offset as u64;
    for (&d, &s) in shape.iter().zip(strides.iter()) {
        let term = (d as u64 - 1)
            .checked_mul(s as u64)
            .ok_or(KernelError::StridesOutOfBounds)?;
        max_reach = max_reach.checked_add(term).ok_or(KernelError::StridesOutOfBounds)?;
    }
    if max_reach >= data_len as u64 {
        return Err(KernelError::StridesOutOfBounds);
    }
    Ok(())
}

/// Kern 03: effective strides for broadcasting with *caller-supplied*
/// strides — `align_to_rank` + `effective_strides` in one step, mirroring
/// what the contiguous kernels do with computed strides. Leading padding
/// axes and size-1 axes both contribute stride 0.
pub fn aligned_effective_strides(shape: &[u32], strides: &[u32], rank: usize) -> Vec<u32> {
    let (shape_al, strides_al) = align_to_rank(shape, strides, rank);
    effective_strides(&shape_al, &strides_al)
}

// --- Kern 06: allocation-free counterparts ---------------------------------
//
// `nt_matmul_blocked_partial`'s entire call graph must be allocation-free
// (docs/kern-06-threads-spec.md, ABI addition contract) — each worker calls
// this kernel independently, and a heap allocation inside a per-worker WASM
// call is exactly the kind of thing that's easy to overlook (the Kern-06
// feasibility spike's own audit found `nt_sum_all_strided` allocating
// internally via `unravel` despite an innocent signature — see that spec's
// "Feasibility grounding" section). These are line-by-line no-alloc twins of
// `compute_strides`/`unravel`/`broadcast_shape`/`aligned_effective_strides`
// above: same arithmetic, same iteration order, but writing into a
// caller-owned `[u32; MAX_RANK]` stack buffer and returning the logical
// length instead of allocating a `Vec`. The originals are untouched
// (existing callers — `matmul`/`matmul_strided`/`matmul_blocked`/`sum_axis*`
// — keep using the allocating versions; additive duplication per this
// crate's own precedent, not a refactor).
//
// **Freeze gate.** Gated `#[cfg(any(not(target_arch = "wasm32"),
// target_feature = "atomics"))]` — same condition, same rationale, as
// `kernels::matmul_blocked`'s own "Freeze gate" doc comment: present for
// native `cargo test` and the threads wasm32 build, absent from the plain
// wasm32 build. These four functions are never `#[no_mangle]` (so never a
// WASM export by themselves), but empirically their mere presence in the
// crate — even fully unreachable, since only the gated `matmul_blocked_partial`
// call graph uses them — still shifted the compiled `.wasm`'s bytes before
// this gate was added (observed directly: a clean `pnpm build:wasm` rebuild
// hashed differently with these functions present-but-unreferenced vs.
// absent, even though the exported-symbol SET was already identical either
// way). Gating them out entirely removes any dependence on LTO/dead-code
// elimination fully erasing unreferenced code from the final artifact.

/// No-alloc twin of [`compute_strides`]: writes `shape.len()` row-major
/// strides into `out[0..shape.len()]` and returns that length. Same overflow
/// precondition as `compute_strides` (`shape` already passed
/// `checked_element_count`, or is a subshape of one that has).
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
pub fn compute_strides_into(shape: &[u32], out: &mut [u32; MAX_RANK]) -> usize {
    let len = shape.len();
    let mut acc: u32 = 1;
    for i in (0..len).rev() {
        out[i] = acc;
        acc = acc.wrapping_mul(shape[i]);
    }
    len
}

/// No-alloc twin of [`unravel`]: writes `shape.len()` per-axis indices into
/// `out[0..shape.len()]` and returns that length. Same precondition as
/// `unravel` (`flat < product(shape)`).
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
pub fn unravel_into(flat: u32, shape: &[u32], strides: &[u32], out: &mut [u32; MAX_RANK]) -> usize {
    let len = shape.len();
    for i in 0..len {
        out[i] = (flat / strides[i]) % shape[i];
    }
    len
}

/// No-alloc twin of [`broadcast_shape`]: writes the broadcast shape into
/// `out[0..rank]` and returns `rank` (`= max(a.len(), b.len())`), or the same
/// `ShapeIncompatible` error the allocating version returns. Same
/// precondition (`a`/`b` individually already rank-validated, so
/// `rank <= MAX_RANK`).
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
pub fn broadcast_shape_into(a: &[u32], b: &[u32], out: &mut [u32; MAX_RANK]) -> KResult<usize> {
    let rank = a.len().max(b.len());
    for i in 0..rank {
        let ai = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bi = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if ai == bi || ai == 1 || bi == 1 {
            out[rank - 1 - i] = if ai == 1 { bi } else { ai };
        } else {
            return Err(KernelError::ShapeIncompatible);
        }
    }
    Ok(rank)
}

/// No-alloc twin of [`aligned_effective_strides`] (itself `align_to_rank` +
/// `effective_strides` fused): writes `rank` effective strides into
/// `out[0..rank]` and returns `rank`. Padding axes (the first
/// `rank - shape.len()` slots) always get stride 0 (dim 1 by construction);
/// remaining axes get 0 if their own dim is 1, else their caller-supplied
/// stride — identical rule to the allocating version.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
pub fn aligned_effective_strides_into(shape: &[u32], strides: &[u32], rank: usize, out: &mut [u32; MAX_RANK]) -> usize {
    let pad = rank - shape.len();
    for slot in out.iter_mut().take(pad) {
        *slot = 0;
    }
    for i in 0..shape.len() {
        out[pad + i] = if shape[i] == 1 { 0 } else { strides[i] };
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_element_count_ok() {
        assert_eq!(checked_element_count(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_element_count(&[]).unwrap(), 1); // rank-0 scalar
        assert_eq!(checked_element_count(&[0, 5]).unwrap(), 0); // size-0 array
    }

    #[test]
    fn checked_element_count_rank_too_large() {
        let shape = vec![1u32; 33];
        assert_eq!(checked_element_count(&shape), Err(KernelError::RankTooLarge));
        let shape32 = vec![1u32; 32];
        assert!(checked_element_count(&shape32).is_ok());
    }

    #[test]
    fn checked_element_count_overflow() {
        // Single dim already at the u32 boundary: element count fits u32,
        // but byte size (*8) overflows.
        assert_eq!(checked_element_count(&[u32::MAX]), Err(KernelError::SizeOverflow));
        // Two dims whose product exceeds u32::MAX outright.
        assert_eq!(checked_element_count(&[70_000, 70_000]), Err(KernelError::SizeOverflow));
    }

    #[test]
    fn compute_strides_matches_runtime_ts() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[]), Vec::<u32>::new());
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn unravel_matches_runtime_ts() {
        let shape = vec![2, 3];
        let strides = compute_strides(&shape);
        assert_eq!(unravel(0, &shape, &strides), vec![0, 0]);
        assert_eq!(unravel(1, &shape, &strides), vec![0, 1]);
        assert_eq!(unravel(3, &shape, &strides), vec![1, 0]);
        assert_eq!(unravel(5, &shape, &strides), vec![1, 2]);
    }

    #[test]
    fn broadcast_shape_basic_and_interior_ones() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[3, 1, 5], &[3, 4, 5]).unwrap(), vec![3, 4, 5]);
        assert_eq!(broadcast_shape(&[1], &[5]).unwrap(), vec![5]);
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_shape_incompatible() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), Err(KernelError::ShapeIncompatible));
    }

    #[test]
    fn broadcast_shape_outer_product_blowup_detected_by_caller() {
        // Broadcasting can grow the output beyond either operand's own size
        // ("outer product" case) — broadcast_shape itself doesn't check
        // this (callers must run checked_element_count on the result).
        let out = broadcast_shape(&[100_000, 1], &[1, 100_000]).unwrap();
        assert_eq!(out, vec![100_000, 100_000]);
        assert_eq!(checked_element_count(&out), Err(KernelError::SizeOverflow));
    }

    #[test]
    fn validate_strided_bounds_contiguous_and_transposed() {
        // Contiguous [2,3]: strides [3,1], max reach 1*3+2*1 = 5 < 6.
        assert!(validate_strided_bounds(&[2, 3], &[3, 1], 0, 6).is_ok());
        // Transposed view [3,2]: strides [1,3], same buffer, same max reach.
        assert!(validate_strided_bounds(&[3, 2], &[1, 3], 0, 6).is_ok());
        // One element short: max reach 5 >= 5.
        assert_eq!(
            validate_strided_bounds(&[2, 3], &[3, 1], 0, 5),
            Err(KernelError::StridesOutOfBounds)
        );
        // Offset pushes the reach past the end.
        assert_eq!(
            validate_strided_bounds(&[2, 3], &[3, 1], 1, 6),
            Err(KernelError::StridesOutOfBounds)
        );
        assert!(validate_strided_bounds(&[2, 3], &[3, 1], 1, 7).is_ok());
    }

    #[test]
    fn validate_strided_bounds_edge_cases() {
        // Rank-0 scalar reads exactly data[offset].
        assert!(validate_strided_bounds(&[], &[], 0, 1).is_ok());
        assert_eq!(validate_strided_bounds(&[], &[], 1, 1), Err(KernelError::StridesOutOfBounds));
        // Size-0 views never read: valid regardless of strides/offset.
        assert!(validate_strided_bounds(&[0, 3], &[999, 999], 123, 0).is_ok());
        // u64 accumulation overflow from garbage strides -> status 4, not wraparound.
        let shape = vec![u32::MAX; 3];
        let strides = vec![u32::MAX; 3];
        assert_eq!(
            validate_strided_bounds(&shape, &strides, 0, u32::MAX),
            Err(KernelError::StridesOutOfBounds)
        );
    }

    #[test]
    fn aligned_effective_strides_pads_and_zeroes() {
        // [3] with stride [1] aligned to rank 3 -> [0,0,1]; a size-1 axis
        // with a real stride is zeroed for broadcasting.
        assert_eq!(aligned_effective_strides(&[3], &[1], 3), vec![0, 0, 1]);
        assert_eq!(aligned_effective_strides(&[1, 5], &[5, 1], 2), vec![0, 1]);
    }

    #[test]
    fn align_and_effective_strides() {
        let strides = compute_strides(&[3]);
        let (shape_al, strides_al) = align_to_rank(&[3], &strides, 3);
        assert_eq!(shape_al, vec![1, 1, 3]);
        assert_eq!(strides_al, vec![0, 0, 1]);
        let eff = effective_strides(&shape_al, &strides_al);
        assert_eq!(eff, vec![0, 0, 1]);
    }

    // --- Kern 06: no-alloc twins agree with the allocating originals -------

    #[test]
    fn compute_strides_into_matches_compute_strides() {
        for shape in [vec![2u32, 3, 4], vec![], vec![5u32], vec![1u32; 32]] {
            let want = compute_strides(&shape);
            let mut buf = [0u32; MAX_RANK];
            let len = compute_strides_into(&shape, &mut buf);
            assert_eq!(len, shape.len());
            assert_eq!(&buf[..len], want.as_slice());
        }
    }

    #[test]
    fn unravel_into_matches_unravel() {
        let shape = vec![2u32, 3, 4];
        let strides = compute_strides(&shape);
        for flat in 0..24u32 {
            let want = unravel(flat, &shape, &strides);
            let mut buf = [0u32; MAX_RANK];
            let len = unravel_into(flat, &shape, &strides, &mut buf);
            assert_eq!(len, shape.len());
            assert_eq!(&buf[..len], want.as_slice());
        }
    }

    #[test]
    fn broadcast_shape_into_matches_broadcast_shape_ok_and_err() {
        let cases: &[(&[u32], &[u32])] = &[
            (&[2, 3], &[3]),
            (&[3, 1, 5], &[3, 4, 5]),
            (&[1], &[5]),
            (&[], &[2, 3]),
        ];
        for &(a, b) in cases {
            let want = broadcast_shape(a, b).unwrap();
            let mut buf = [0u32; MAX_RANK];
            let len = broadcast_shape_into(a, b, &mut buf).unwrap();
            assert_eq!(len, want.len());
            assert_eq!(&buf[..len], want.as_slice());
        }
        let mut buf = [0u32; MAX_RANK];
        assert_eq!(broadcast_shape_into(&[2, 3], &[2, 4], &mut buf), Err(KernelError::ShapeIncompatible));
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), Err(KernelError::ShapeIncompatible));
    }

    #[test]
    fn aligned_effective_strides_into_matches_aligned_effective_strides() {
        let cases: &[(&[u32], &[u32], usize)] = &[(&[3], &[1], 3), (&[1, 5], &[5, 1], 2), (&[], &[], 0), (&[4, 1, 6], &[6, 999, 1], 3)];
        for &(shape, strides, rank) in cases {
            let want = aligned_effective_strides(shape, strides, rank);
            let mut buf = [0u32; MAX_RANK];
            let len = aligned_effective_strides_into(shape, strides, rank, &mut buf);
            assert_eq!(len, want.len());
            assert_eq!(&buf[..len], want.as_slice());
        }
    }
}
