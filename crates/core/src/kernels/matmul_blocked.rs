//! Kern 04: blocked + packed + SIMD128 (f64x2) matmul core. Same contract as
//! [`super::matmul::matmul_strided`] (operands pre-promoted to rank >= 2,
//! caller-supplied strided quadruples, batch broadcasting, unsqueezed
//! `[...batch, m, n]` output, status codes 1/2/3/4 via the same
//! `validate_strided_bounds` call) — a from-scratch GEMM-style kernel that
//! replaces the naive strided triple loop with the classic pack + block
//! recipe, addressing the cache-access-pattern bottleneck Kern 03's bench
//! measured directly (a transposed operand's column-wise k-loop reads lose
//! ~30% at n >= 256; see docs/kern-03-ergebnisse.md).
//!
//! ## The bit-identity law (docs/kern-04-simd-blocking-spec.md) — how this
//! stays bit-for-bit equal to [`super::matmul::matmul`] /
//! [`super::matmul::matmul_strided`] despite blocking and vectorizing:
//!
//! 1. **Packing is pure data movement.** [`pack_a_tile`]/[`pack_b_tile`] only
//!    gather operand elements (via the caller's strides/offset, exactly like
//!    [`super::matmul::matmul_strided`]'s reads) into contiguous scratch —
//!    no arithmetic, so no effect on any computed bit.
//! 2. **Per output element, one accumulator, strictly ascending k, forever.**
//!    The (m, n, k) tile loops are nested `mb { kb { pack A; nb { pack B;
//!    microkernel } } }`. For a *fixed* output element, that visits k-blocks
//!    in ascending order (the `kb` loop is outermost of the two loops that
//!    touch k), and every visit **reads the element's current value straight
//!    out of `out[]`** (0.0 for the very first k-block — `out` is
//!    zero-initialized exactly like the unblocked kernels) **and writes the
//!    updated value back** before the next k-block reads it. A value
//!    round-tripped through `out[]` between k-blocks is bit-for-bit the same
//!    value it would be sitting in a register the whole time (both are the
//!    same IEEE-754 f64 bits — WASM has no x87-style extended-precision
//!    intermediate to lose); so this is bit-identical to one flat
//!    `for k in 0..K { out[idx] += a*b }` loop, merely chunked. This is
//!    exactly what the spec's law calls "k-tiles in ascending order,
//!    accumulating into the same chain" — never a fresh local-zero partial
//!    sum added post-hoc (that WOULD reassociate and is explicitly banned).
//! 3. **Vectorization is only ever across two *adjacent output columns*.**
//!    [`accumulate_pair`] (wasm) runs ONE `f64x2` accumulator for the
//!    `(nj, nj+1)` column pair; each lane independently performs exactly the
//!    scalar sequence (splat one `a` value, multiply, add) that
//!    [`accumulate_pair`] (native, scalar) performs with two independent
//!    `f64` accumulators — WASM SIMD128's `f64x2.add`/`f64x2.mul` are
//!    specified as lane-wise IEEE-754 ops with the same rounding as scalar
//!    (verified against the WASM SIMD proposal text: "lane-wise versions of
//!    the existing scalar WebAssembly operations... all operations use the
//!    default roundTiesToEven rounding mode" — no fusion, no
//!    implementation-defined behavior). A leftover odd column (tile
//!    remainder, or an odd matrix dimension) falls to [`accumulate_single`],
//!    the same scalar sequence a lane would have run.
//! 4. **No FMA, no relaxed-simd.** Only `f64x2_mul` + `f64x2_add` (two
//!    roundings, matching scalar `a*b` then `+=`); `f64x2_relaxed_madd` and
//!    friends are banned outright — the relaxed-simd proposal explicitly
//!    permits implementation-defined per-engine results, which would break
//!    cross-engine determinism (see KB
//!    `cross-plattform-float-determinismus-verifizieren`).
//!
//! ## SIMD mechanics (docs-first, verified against doc.rust-lang.org
//! `core::arch::wasm32` + the WASM SIMD proposal text before writing this):
//! `f64x2_splat`/`f64x2_mul`/`f64x2_add`/`f64x2_extract_lane`/`f64x2` (the
//! two-scalar constructor) are all **safe** functions (not `unsafe fn`) —
//! confirmed empirically on this toolchain (rustc 1.95.0): with `simd128`
//! enabled crate-wide via `-C target-feature=+simd128` (this crate's
//! `.cargo/config.toml`), they compile and run from ordinary safe code with
//! no `unsafe` block anywhere in this module (Rust 1.86+ stabilized calling
//! safe `#[target_feature]` functions from code that provably has the
//! feature enabled — here, unconditionally, since the whole module is
//! `cfg(target_arch = "wasm32")`-gated and the crate always builds wasm32
//! with `+simd128`). `v128_load`/`v128_store` exist and are `unsafe fn`
//! (raw-pointer deref) but are deliberately NOT used here — the safe `f64x2`
//! two-scalar constructor reads the already-bounds-checked packed slice
//! directly, sidestepping any pointer/alignment reasoning entirely (WASM
//! SIMD128 loads are unaligned-safe by spec anyway, but this avoids the
//! question altogether).
//!
//! ## Coverage split (spec, stated honestly)
//! Native `cargo test` runs the `#[cfg(not(target_arch = "wasm32"))]` scalar
//! micro-step — it validates the blocking/packing/tiling logic and the
//! ascending-k accumulation argument above, natively, against
//! [`super::matmul::matmul_strided`]. It does **not** exercise a single WASM
//! SIMD instruction (core::arch::wasm32 doesn't even compile off-target).
//! The TS differential suite (`spike/tests-runtime/blocked.test.ts`) running
//! against the real compiled `.wasm` is the only gate for the SIMD path
//! itself. The `compile_error!` guard below is the loud proof the `.wasm`
//! was actually built with `simd128` codegen, not a silent scalar fallback
//! (confirmed empirically: omitting `-C target-feature=+simd128` does NOT
//! fail to compile the intrinsic calls — LLVM legalizes `v128` ops to
//! scalarized code on wasm32 even without the feature, i.e. exactly the
//! silent-fallback failure mode this guard exists to catch).
//!
//! ## Runtime baseline
//! WASM SIMD128 is on by default in Node >= 16.4 and all evergreen browsers
//! (2021+); an engine without it fails loudly at module instantiation
//! (validation error), never silently.

use crate::shape::{
    aligned_effective_strides, broadcast_shape, checked_element_count, compute_strides, product, unravel,
    validate_strided_bounds, KResult, KernelError,
};

#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!(
    "matmul_blocked requires the simd128 target feature on wasm32: set \
     rustflags = [\"-C\", \"target-feature=+simd128\"] under \
     [target.wasm32-unknown-unknown] in a repo-root .cargo/config.toml, AND \
     invoke cargo with its current working directory at (or under) that \
     root — Cargo's config-file discovery is cwd-based, not \
     --manifest-path-based (verified empirically, see that file's comment)."
);

/// M-tile (row panel) size: rows of A packed together per (mb, kb) block.
/// N-tile (column panel) size: must stay a multiple of 2 for the f64x2
/// micro-step to tile it evenly (an odd leftover column within a tile still
/// works via the scalar tail, but keeping NC even avoids paying that tail on
/// every full tile). K-tile size: bounds the packed-panel working set so it
/// stays cache-resident regardless of the operand's full K.
///
/// Chosen by rough measurement (raw-ABI timing of `nt_matmul_blocked` at
/// n=512/1024, see docs/kern-04-ergebnisse.md for the full table) — no
/// BLIS-grade auto-tuning; "clearly better and understood" per spec. A first
/// guess of (64, 64, 256) measured 48.2ms/370.0ms at n=512/1024; shrinking
/// all three consistently helped (smaller K-panels in particular — packed
/// A/B tiles staying well inside L1/L2 beats a larger single pack-then-reuse
/// window), bottoming out around (32, 32, 32) at 38.8ms/310.9ms — roughly a
/// 15-20% further improvement over the first guess, and still a ~2.1-2.7x
/// win over the unblocked Kern-03 scalar-strided kernel across n=64..1024
/// (see the bench table). Values much smaller (16) or larger (128, or K
/// >= 256) both measured worse.
const MC: usize = 32;
const NC: usize = 32;
const KC: usize = 32;

/// Gather `A[mb..mb+mc, kb..kb+kc]` (via the operand's own strides/offset —
/// contiguous and transposed-view operands take the identical path here)
/// into a fresh contiguous row-major `mc x kc` buffer. Pure data movement:
/// no arithmetic, so no effect on any bit the microkernel later produces.
fn pack_a_tile(
    a_data: &[f64],
    a_base_off: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    mb: usize,
    mc: usize,
    kb: usize,
    kc: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; mc * kc];
    for mi in 0..mc {
        let row_base = a_base_off + ((mb + mi) as u32) * a_row_stride;
        for kj in 0..kc {
            let idx = row_base + ((kb + kj) as u32) * a_col_stride;
            out[mi * kc + kj] = a_data.get(idx as usize).copied().unwrap_or(0.0);
        }
    }
    out
}

/// Gather `B[kb..kb+kc, nb..nb+nc]` into a fresh contiguous row-major
/// `kc x nc` buffer — same pure-data-movement contract as [`pack_a_tile`].
/// Row-major over `nc` so two adjacent columns (`nj`, `nj+1`) at a fixed `k`
/// are adjacent in memory, which is exactly what the f64x2 microkernel loads.
fn pack_b_tile(
    b_data: &[f64],
    b_base_off: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    kb: usize,
    kc: usize,
    nb: usize,
    nc: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; kc * nc];
    for kj in 0..kc {
        let row_base = b_base_off + ((kb + kj) as u32) * b_row_stride;
        for nj in 0..nc {
            let idx = row_base + ((nb + nj) as u32) * b_col_stride;
            out[kj * nc + nj] = b_data.get(idx as usize).copied().unwrap_or(0.0);
        }
    }
    out
}

/// Accumulate ONE k-block's contribution for the adjacent output-column pair
/// `(nj, nj+1)` into `out[o0]`/`out[o1]`, which already hold the running
/// total from prior k-blocks (or 0.0 for the very first — see module doc
/// law #2). WASM: one `f64x2` accumulator across the whole `kc` range —
/// each lane reproduces exactly the scalar sequence the `not(wasm32)` arm
/// below runs independently per lane.
#[cfg(target_arch = "wasm32")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_pair(a_row: &[f64], b_pack: &[f64], nc: usize, nj: usize, kc: usize, out: &mut [f64], o0: usize, o1: usize) {
    use std::arch::wasm32::{f64x2, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat};
    let mut acc = f64x2(out[o0], out[o1]);
    for kj in 0..kc {
        let av = f64x2_splat(a_row[kj]);
        let bv = f64x2(b_pack[kj * nc + nj], b_pack[kj * nc + nj + 1]);
        acc = f64x2_add(acc, f64x2_mul(av, bv));
    }
    out[o0] = f64x2_extract_lane::<0>(acc);
    out[o1] = f64x2_extract_lane::<1>(acc);
}

/// Native (host) equivalent of the wasm [`accumulate_pair`] above: two
/// independent scalar accumulators, each running the identical arithmetic
/// sequence (ascending `kj`, one mul then one add per step) its wasm
/// counterpart's matching lane would run. Validated by `cargo test` against
/// [`super::matmul::matmul_strided`]; NOT a stand-in for the SIMD path
/// itself (see module doc "Coverage split").
#[cfg(not(target_arch = "wasm32"))]
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_pair(a_row: &[f64], b_pack: &[f64], nc: usize, nj: usize, kc: usize, out: &mut [f64], o0: usize, o1: usize) {
    let mut acc0 = out[o0];
    let mut acc1 = out[o1];
    for kj in 0..kc {
        let a_val = a_row[kj];
        acc0 += a_val * b_pack[kj * nc + nj];
        acc1 += a_val * b_pack[kj * nc + nj + 1];
    }
    out[o0] = acc0;
    out[o1] = acc1;
}

/// Scalar tail: one leftover output column (odd tile/matrix remainder). Same
/// accumulate-across-k-blocks-via-`out[]` pattern as [`accumulate_pair`],
/// just one lane — used identically on wasm and native.
#[inline]
fn accumulate_single(a_row: &[f64], b_pack: &[f64], nc: usize, nj: usize, kc: usize, out: &mut [f64], o: usize) {
    let mut acc = out[o];
    for kj in 0..kc {
        acc += a_row[kj] * b_pack[kj * nc + nj];
    }
    out[o] = acc;
}

/// Run the microkernel over one packed `(mc x kc) x (kc x nc)` tile,
/// accumulating this k-block's contribution into `out[]` at the tile's
/// global `(mb, nb)` position (row width `out_row_width`, batch-slice base
/// `out_base_off`).
#[allow(clippy::too_many_arguments)]
fn micro_tile(
    mc: usize,
    nc: usize,
    kc: usize,
    a_pack: &[f64],
    b_pack: &[f64],
    out: &mut [f64],
    out_base_off: u32,
    out_row_width: u32,
    mb: usize,
    nb: usize,
) {
    for mi in 0..mc {
        let a_row = &a_pack[mi * kc..mi * kc + kc];
        let out_row_off = out_base_off + ((mb + mi) as u32) * out_row_width;
        let mut nj = 0usize;
        while nj + 2 <= nc {
            let o0 = (out_row_off + (nb + nj) as u32) as usize;
            let o1 = (out_row_off + (nb + nj + 1) as u32) as usize;
            accumulate_pair(a_row, b_pack, nc, nj, kc, out, o0, o1);
            nj += 2;
        }
        if nj < nc {
            let o = (out_row_off + (nb + nj) as u32) as usize;
            accumulate_single(a_row, b_pack, nc, nj, kc, out, o);
        }
    }
}

/// The blocked 2-D core for a single batch slice: `mb { kb { pack A; nb {
/// pack B; microkernel } } }` — see module doc law #2 for why this loop
/// order (kb ascending, nested inside a fixed mb, wrapping nb) preserves
/// per-output-element ascending-k accumulation. A packs once per `(mb, kb)`
/// (reused across every `nb` in that block); B repacks once per `(mb, kb,
/// nb)` — some redundant B-repacking across `mb` blocks is accepted
/// (research-spike-grade, not BLIS-grade packing reuse) since its cost is
/// `O(m/MC)` relative to the O(m*k*n) total work, i.e. amortized by `MC`.
#[allow(clippy::too_many_arguments)]
fn matmul_2d_blocked(
    m: u32,
    k: u32,
    n: u32,
    a_data: &[f64],
    a_base_off: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_data: &[f64],
    b_base_off: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    out: &mut [f64],
    out_base_off: u32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    if m == 0 || n == 0 {
        return; // k==0 needs no special case: out[] is already zero-initialized.
    }
    let out_row_width = n as u32;

    let mut mb = 0usize;
    while mb < m {
        let mc = MC.min(m - mb);
        let mut kb = 0usize;
        while kb < k {
            let kc = KC.min(k - kb);
            let a_pack = pack_a_tile(a_data, a_base_off, a_row_stride, a_col_stride, mb, mc, kb, kc);
            let mut nb = 0usize;
            while nb < n {
                let nc = NC.min(n - nb);
                let b_pack = pack_b_tile(b_data, b_base_off, b_row_stride, b_col_stride, kb, kc, nb, nc);
                micro_tile(mc, nc, kc, &a_pack, &b_pack, out, out_base_off, out_row_width, mb, nb);
                nb += NC;
            }
            kb += KC;
        }
        mb += MC;
    }
}

/// Kern 04: blocked + packed + SIMD128 generalization of
/// [`super::matmul::matmul_strided`] — identical contract and identical
/// batch-broadcast handling (this function's batch loop is a deliberate,
/// expected duplicate of `matmul_strided`'s, matching this file's existing
/// precedent of duplication over shared code paths); only the 2-D core
/// differs (blocked+packed+vectorized instead of the naive triple loop).
#[allow(clippy::too_many_arguments)]
pub fn matmul_blocked(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
) -> KResult<(Vec<u32>, Vec<f64>)> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(KernelError::ShapeIncompatible);
    }
    checked_element_count(a_shape)?;
    checked_element_count(b_shape)?;
    validate_strided_bounds(a_shape, a_strides, a_offset, a_data.len() as u32)?;
    validate_strided_bounds(b_shape, b_strides, b_offset, b_data.len() as u32)?;

    let m = a_shape[a_shape.len() - 2];
    let k1 = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k1 != k2 {
        return Err(KernelError::ShapeIncompatible);
    }

    let batch_a = &a_shape[..a_shape.len() - 2];
    let batch_b = &b_shape[..b_shape.len() - 2];
    let batch_out = broadcast_shape(batch_a, batch_b)?;
    let batch_rank = batch_out.len();

    let a_batch_eff = aligned_effective_strides(batch_a, &a_strides[..a_strides.len() - 2], batch_rank);
    let b_batch_eff = aligned_effective_strides(batch_b, &b_strides[..b_strides.len() - 2], batch_rank);

    let a_row_stride = a_strides[a_strides.len() - 2];
    let a_col_stride = a_strides[a_strides.len() - 1];
    let b_row_stride = b_strides[b_strides.len() - 2];
    let b_col_stride = b_strides[b_strides.len() - 1];

    let mut out_full_shape = batch_out.clone();
    out_full_shape.push(m);
    out_full_shape.push(n);
    debug_assert!(out_full_shape.len() <= crate::shape::MAX_RANK);
    let out_size = checked_element_count(&out_full_shape)?;

    let out_strides_full = compute_strides(&out_full_shape);
    let out_batch_strides = &out_strides_full[..batch_rank];
    let batch_strides_plain = compute_strides(&batch_out);

    let batch_size = product(&batch_out);
    let mut out = vec![0f64; out_size as usize];

    for b_idx in 0..batch_size {
        let multi = unravel(b_idx, &batch_out, &batch_strides_plain);
        let mut a_batch_off: u32 = a_offset;
        let mut b_batch_off: u32 = b_offset;
        let mut out_batch_off: u32 = 0;
        for i in 0..batch_rank {
            let ix = multi[i];
            a_batch_off += ix * a_batch_eff[i];
            b_batch_off += ix * b_batch_eff[i];
            out_batch_off += ix * out_batch_strides[i];
        }
        matmul_2d_blocked(
            m,
            k1,
            n,
            a_data,
            a_batch_off,
            a_row_stride,
            a_col_stride,
            b_data,
            b_batch_off,
            b_row_stride,
            b_col_stride,
            &mut out,
            out_batch_off,
        );
    }

    Ok((out_full_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::matmul_strided;

    fn strides_of(shape: &[u32]) -> Vec<u32> {
        compute_strides(shape)
    }

    fn bits(data: &[f64]) -> Vec<u64> {
        data.iter().map(|v| v.to_bits()).collect()
    }

    // --- packing unit tests --------------------------------------------

    #[test]
    fn pack_a_tile_contiguous() {
        // A = [[1,2,3],[4,5,6]] (2x3), strides [3,1]. Packing the full
        // (mc=2, kc=3) tile from (mb=0, kb=0) must reproduce it verbatim.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let packed = pack_a_tile(&a, 0, 3, 1, 0, 2, 0, 3);
        assert_eq!(packed, a);
    }

    #[test]
    fn pack_a_tile_strided_view_and_offset() {
        // Transposed view of A=[[1,2,3],[4,5,6]] stored column-major as
        // [1,4,2,5,3,6] (shape [2,3] read with strides [1,2]); packing a
        // sub-tile (mb=1,mc=1,kb=1,kc=2) must select A[1][1..3] = [5,6].
        let store = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let packed = pack_a_tile(&store, 0, 1, 2, 1, 1, 1, 2);
        assert_eq!(packed, vec![5.0, 6.0]);

        // Nonzero base offset selects a later sub-buffer (same shape [2,2],
        // contiguous strides [2,1], base offset 4 into an [I, M] buffer).
        let buf = vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0];
        let packed_off = pack_a_tile(&buf, 4, 2, 1, 0, 2, 0, 2);
        assert_eq!(packed_off, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn pack_b_tile_contiguous_and_partial() {
        // B = [[7,8],[9,10],[11,12]] (3x2), strides [2,1]. Full pack.
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let packed = pack_b_tile(&b, 0, 2, 1, 0, 3, 0, 2);
        assert_eq!(packed, b);
        // Partial tile: kb=1,kc=2, nb=1,nc=1 -> column 1, rows 1..3 = [10,12].
        let packed_partial = pack_b_tile(&b, 0, 2, 1, 1, 2, 1, 1);
        assert_eq!(packed_partial, vec![10.0, 12.0]);
    }

    // --- seeded bit-equivalence vs matmul_strided -----------------------
    // Deterministic LCG (test-only, zero deps) — enough variety to cross
    // MC/NC/KC tile boundaries with remainders.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            let bits = self.next();
            let sign = if bits & 1 == 1 { -1.0 } else { 1.0 };
            let int_part = ((bits >> 1) % 1000) as f64;
            let frac_part = ((bits >> 11) % 1000) as f64 / 1000.0;
            sign * (int_part + frac_part)
        }
    }

    fn gen_data(lcg: &mut Lcg, n: usize) -> Vec<f64> {
        (0..n).map(|_| lcg.next_f64()).collect()
    }

    fn assert_bit_eq(a: &[f64], b: &[f64], ctx: &str) {
        assert_eq!(bits(a), bits(b), "{ctx}: bit mismatch");
    }

    /// Core equivalence: for a grid of (m,k,n) values straddling MC/NC/KC
    /// (below, exactly at, one past, ~2-3x, and a prime/odd remainder),
    /// contiguous operands, matmul_blocked must match matmul_strided
    /// bit-for-bit.
    #[test]
    fn blocked_equals_strided_contiguous_grid() {
        let dims = [1usize, 2, 63, 64, 65, 127, 128, 129, 200, 255, 256, 257, 300];
        let mut lcg = Lcg(0x4b45524e5f303421);
        let mut cases = 0;
        for &m in dims.iter().take(6) {
            for &k in dims.iter().skip(2).take(6) {
                for &n in dims.iter().skip(4).take(5) {
                    let a = gen_data(&mut lcg, m * k);
                    let b = gen_data(&mut lcg, k * n);
                    let a_shape = [m as u32, k as u32];
                    let b_shape = [k as u32, n as u32];
                    let a_strides = strides_of(&a_shape);
                    let b_strides = strides_of(&b_shape);
                    let (shape_ref, data_ref) =
                        matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
                    let (shape_got, data_got) =
                        matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
                    assert_eq!(shape_ref, shape_got, "m={m} k={k} n={n}: shape");
                    assert_bit_eq(&data_ref, &data_got, &format!("m={m} k={k} n={n}"));
                    cases += 1;
                }
            }
        }
        assert!(cases >= 100, "expected >=100 grid cases, got {cases}");
    }

    #[test]
    fn blocked_equals_strided_transposed_view_operands() {
        // Both operands as transposed views (column-major storage), dims
        // spanning tile boundaries on both m/k (A) and k/n (B) sides.
        let mut lcg = Lcg(0x5649455721544553);
        for &(m, k, n) in &[(65usize, 130usize, 33usize), (129, 64, 200), (33, 257, 65)] {
            let a_logical = gen_data(&mut lcg, m * k); // logical A[m][k], row-major
            let b_logical = gen_data(&mut lcg, k * n);
            // Store A transposed (k x m) so that a strided VIEW with shape
            // [m,k] and strides [1,m] reads back a_logical exactly (mirrors
            // strided.test.ts's involution trick, adapted natively).
            let mut a_store = vec![0.0; m * k];
            for mi in 0..m {
                for ki in 0..k {
                    a_store[ki * m + mi] = a_logical[mi * k + ki];
                }
            }
            let mut b_store = vec![0.0; k * n];
            for ki in 0..k {
                for nj in 0..n {
                    b_store[nj * k + ki] = b_logical[ki * n + nj];
                }
            }
            let a_shape = [m as u32, k as u32];
            let a_view_strides = [1u32, m as u32];
            let b_shape = [k as u32, n as u32];
            let b_view_strides = [1u32, k as u32];

            let a_natural_strides = strides_of(&a_shape);
            let b_natural_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) = matmul_strided(
                &a_shape,
                &a_natural_strides,
                0,
                &a_logical,
                &b_shape,
                &b_natural_strides,
                0,
                &b_logical,
            )
            .unwrap();
            let (shape_got, data_got) = matmul_blocked(
                &a_shape,
                &a_view_strides,
                0,
                &a_store,
                &b_shape,
                &b_view_strides,
                0,
                &b_store,
            )
            .unwrap();
            assert_eq!(shape_ref, shape_got, "view case m={m} k={k} n={n}");
            assert_bit_eq(&data_ref, &data_got, &format!("view case m={m} k={k} n={n}"));
        }
    }

    #[test]
    fn blocked_equals_strided_k_equals_one_and_n_equals_one() {
        let mut lcg = Lcg(0x4b3d315f4e3d315f);
        for &(m, k, n) in &[(70usize, 1usize, 70usize), (70, 70, 1), (1, 1, 1), (300, 1, 1)] {
            let a = gen_data(&mut lcg, m * k);
            let b = gen_data(&mut lcg, k * n);
            let a_shape = [m as u32, k as u32];
            let b_shape = [k as u32, n as u32];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got, "m={m} k={k} n={n}");
            assert_bit_eq(&data_ref, &data_got, &format!("m={m} k={k} n={n}"));
        }
    }

    #[test]
    fn blocked_equals_strided_batch_and_broadcast_batch() {
        let mut lcg = Lcg(0x4241544348212121);
        // No-broadcast batch: [3,65,70] @ [3,70,66].
        {
            let (bsz, m, k, n) = (3usize, 65usize, 70usize, 66usize);
            let a = gen_data(&mut lcg, bsz * m * k);
            let b = gen_data(&mut lcg, bsz * k * n);
            let a_shape = [bsz as u32, m as u32, k as u32];
            let b_shape = [bsz as u32, k as u32, n as u32];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got);
            assert_bit_eq(&data_ref, &data_got, "batch no-broadcast");
        }
        // Broadcast batch: [1,65,70] @ [4,70,66] -> batch 4.
        {
            let (m, k, n) = (65usize, 70usize, 66usize);
            let a = gen_data(&mut lcg, m * k); // batch=1 operand
            let b = gen_data(&mut lcg, 4 * k * n);
            let a_shape = [1u32, m as u32, k as u32];
            let b_shape = [4u32, k as u32, n as u32];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got);
            assert_bit_eq(&data_ref, &data_got, "batch broadcast");
        }
    }

    #[test]
    fn blocked_equals_strided_size_zero() {
        let mut lcg = Lcg(0x53495a455f303030);
        // m=0
        {
            let a: Vec<f64> = vec![];
            let b = gen_data(&mut lcg, 5 * 4);
            let a_shape = [0u32, 5];
            let b_shape = [5u32, 4];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got);
            assert_bit_eq(&data_ref, &data_got, "m=0");
        }
        // k=0 (all outputs must be exactly 0.0, not merely close)
        {
            let a: Vec<f64> = vec![];
            let b: Vec<f64> = vec![];
            let a_shape = [70u32, 0];
            let b_shape = [0u32, 66];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got);
            assert_bit_eq(&data_ref, &data_got, "k=0");
            assert!(data_got.iter().all(|&v| v == 0.0), "k=0 outputs must all be exactly 0.0");
        }
        // n=0
        {
            let a = gen_data(&mut lcg, 70 * 65);
            let b: Vec<f64> = vec![];
            let a_shape = [70u32, 65];
            let b_shape = [65u32, 0];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (shape_ref, data_ref) =
                matmul_strided(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let (shape_got, data_got) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(shape_ref, shape_got);
            assert_bit_eq(&data_ref, &data_got, "n=0");
        }
    }

    // --- status paths (1/2/4) — same statuses as matmul_strided ----------

    #[test]
    fn status_1_inner_dim_mismatch() {
        let err = matmul_blocked(&[2, 3], &[3, 1], 0, &[0.0; 6], &[4, 2], &[2, 1], 0, &[0.0; 8]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn status_1_rank_below_two_rejected() {
        let err = matmul_blocked(&[3], &[1], 0, &[1.0, 2.0, 3.0], &[3, 2], &[2, 1], 0, &[0.0; 6]).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn status_2_rank_too_large_operand() {
        let mut shape = vec![1u32; 31];
        shape.push(2); // rank 32 already at cap
        let mut shape2 = shape.clone();
        shape2.push(1); // rank 33: over cap
        let strides2 = strides_of(&shape2);
        let err = matmul_blocked(&shape2, &strides2, 0, &[0.0], &[2, 2], &[2, 1], 0, &[0.0; 4]).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn status_4_strides_out_of_bounds() {
        let a = vec![0.0; 6];
        let b = vec![0.0; 6];
        let err = matmul_blocked(&[2, 3], &[3, 2], 0, &a, &[3, 2], &[2, 1], 0, &b).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    /// The two-D core itself must reproduce the classic hand-checked
    /// result (same fixture as matmul.rs's `two_d_core`), pinning it
    /// independent of the equivalence-vs-matmul_strided tests above.
    #[test]
    fn hand_checked_two_d_core() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_strides = strides_of(&[2, 3]);
        let b_strides = strides_of(&[3, 2]);
        let (shape, data) = matmul_blocked(&[2, 3], &a_strides, 0, &a, &[3, 2], &b_strides, 0, &b).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
