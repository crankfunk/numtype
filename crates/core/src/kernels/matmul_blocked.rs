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
    // ==== Kern 06: matmul_blocked_partial equivalence grid ==================
    //
    // `matmul_blocked_partial` must reproduce `matmul_blocked`'s output
    // bit-for-bit for ANY partition of the flat row space into partial calls
    // (the parallel bit-identity law's "MC-aligned split points are a
    // performance recommendation, never a correctness requirement"). Every
    // test below drives the partial kernel through `run_partial_calls`,
    // which starts the output buffer at NaN sentinels — any row the chosen
    // split pattern fails to cover would surface as a loud NaN-vs-real-value
    // mismatch in `assert_bit_eq`, not a silent pass.

    /// Run one or more `matmul_blocked_partial` calls (a "split pattern") into
    /// a fresh `out_len`-element buffer seeded with NaN, so any row a split
    /// pattern fails to cover is visibly wrong (NaN bits vs. real bits)
    /// rather than accidentally matching leftover zeros.
    #[allow(clippy::too_many_arguments)]
    fn run_partial_calls(
        a_shape: &[u32],
        a_strides: &[u32],
        a_offset: u32,
        a_data: &[f64],
        b_shape: &[u32],
        b_strides: &[u32],
        b_offset: u32,
        b_data: &[f64],
        out_len: usize,
        splits: &[(u32, u32)],
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; out_len];
        for &(rs, re) in splits {
            let status = matmul_blocked_partial(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data, &mut out, rs, re);
            assert!(status.is_ok(), "partial call [{rs},{re}) over out_len={out_len} failed: {status:?}");
        }
        out
    }

    /// Generate several partition patterns of `[0, total)` deliberately
    /// straddling the MC=32 tile boundary: a single range, leading/trailing
    /// EMPTY ranges (correctness, not just performance, per the spec law), an
    /// MC-aligned 2-way split, an MC-misaligned 2-way split, a generic
    /// misaligned 2-way split, and a many-range (~4 chunk) split.
    fn split_patterns(total: u32) -> Vec<Vec<(u32, u32)>> {
        let mut patterns = vec![vec![(0, total)]];
        if total > 0 {
            patterns.push(vec![(0, 0), (0, total)]);
            patterns.push(vec![(0, total), (total, total)]);
        }
        if total >= 2 {
            let mid = total / 2;
            patterns.push(vec![(0, mid), (mid, total)]);
        }
        if total > 32 {
            patterns.push(vec![(0, 32), (32, total)]);
        }
        if total > 33 {
            patterns.push(vec![(0, 33), (33, total)]);
        }
        if total >= 4 {
            let step = (total / 4).max(1);
            let mut ranges = vec![];
            let mut cur = 0u32;
            while cur < total {
                let next = (cur + step).min(total);
                ranges.push((cur, next));
                cur = next;
            }
            patterns.push(ranges);
        }
        patterns
    }

    /// Core equivalence: for a grid of (m,k,n) straddling MC/NC/KC (below, at,
    /// one past, and well past the 32 boundary) and every split pattern from
    /// `split_patterns`, `matmul_blocked_partial` (union of calls) must equal
    /// one `matmul_blocked` call, bit-for-bit.
    #[test]
    fn partial_equals_blocked_grid_various_splits() {
        let dims = [1usize, 2, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129];
        let mut lcg = Lcg(0x5041525449414c21);
        let mut cases = 0;
        for &m in dims.iter().take(5) {
            for &k in dims.iter().skip(3).take(3) {
                for &n in dims.iter().skip(5).take(3) {
                    let a = gen_data(&mut lcg, m * k);
                    let b = gen_data(&mut lcg, k * n);
                    let a_shape = [m as u32, k as u32];
                    let b_shape = [k as u32, n as u32];
                    let a_strides = strides_of(&a_shape);
                    let b_strides = strides_of(&b_shape);
                    let (_shape_ref, data_ref) =
                        matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
                    let total_rows = m as u32; // batch_size = 1
                    for splits in split_patterns(total_rows) {
                        let data_got =
                            run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, data_ref.len(), &splits);
                        assert_bit_eq(&data_ref, &data_got, &format!("m={m} k={k} n={n} splits={splits:?}"));
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 100, "expected >=100 grid*split cases, got {cases}");
    }

    #[test]
    fn partial_equals_blocked_transposed_view_operands() {
        let mut lcg = Lcg(0x5649455721504152);
        for &(m, k, n) in &[(65usize, 130usize, 33usize), (129, 64, 200)] {
            let a_logical = gen_data(&mut lcg, m * k);
            let b_logical = gen_data(&mut lcg, k * n);
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
            let (_shape_ref, data_ref) = matmul_strided(
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
            for splits in split_patterns(m as u32) {
                let data_got = run_partial_calls(
                    &a_shape,
                    &a_view_strides,
                    0,
                    &a_store,
                    &b_shape,
                    &b_view_strides,
                    0,
                    &b_store,
                    data_ref.len(),
                    &splits,
                );
                assert_bit_eq(&data_ref, &data_got, &format!("view case m={m} k={k} n={n} splits={splits:?}"));
            }
        }
    }

    #[test]
    fn partial_equals_blocked_batch_no_broadcast_and_broadcast() {
        let mut lcg = Lcg(0x4241544348504152);
        // No-broadcast batch: [3,65,70] @ [3,70,66] — total_rows = 3*65 = 195,
        // split patterns deliberately straddle batch boundaries (65-row
        // batches), including one call spanning parts of two batches.
        {
            let (bsz, m, k, n) = (3usize, 65usize, 70usize, 66usize);
            let a = gen_data(&mut lcg, bsz * m * k);
            let b = gen_data(&mut lcg, bsz * k * n);
            let a_shape = [bsz as u32, m as u32, k as u32];
            let b_shape = [bsz as u32, k as u32, n as u32];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (_shape_ref, data_ref) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let total_rows = (bsz * m) as u32;
            let mut patterns = split_patterns(total_rows);
            // One call spanning the tail of batch 0 + all of batch 1 + head
            // of batch 2 (cross-batch-boundary partial calls, not just
            // batch-aligned ones).
            patterns.push(vec![(50, 130), (0, 50), (130, total_rows)]);
            for splits in patterns {
                let data_got =
                    run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, data_ref.len(), &splits);
                assert_bit_eq(&data_ref, &data_got, &format!("batch no-broadcast splits={splits:?}"));
            }
        }
        // Broadcast batch: [1,65,70] @ [4,70,66] -> batch 4.
        {
            let (m, k, n) = (65usize, 70usize, 66usize);
            let a = gen_data(&mut lcg, m * k);
            let b = gen_data(&mut lcg, 4 * k * n);
            let a_shape = [1u32, m as u32, k as u32];
            let b_shape = [4u32, k as u32, n as u32];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (_shape_ref, data_ref) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            let total_rows = (4 * m) as u32;
            for splits in split_patterns(total_rows) {
                let data_got =
                    run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, data_ref.len(), &splits);
                assert_bit_eq(&data_ref, &data_got, &format!("batch broadcast splits={splits:?}"));
            }
        }
    }

    #[test]
    fn partial_equals_blocked_size_zero() {
        let mut lcg = Lcg(0x53495a455f504152);
        // m=0: total_rows = 0 -> only the empty range is valid.
        {
            let a: Vec<f64> = vec![];
            let b = gen_data(&mut lcg, 5 * 4);
            let a_shape = [0u32, 5];
            let b_shape = [5u32, 4];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (_shape_ref, data_ref) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(data_ref.len(), 0);
            let data_got =
                run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, 0, &[(0, 0)]);
            assert_eq!(data_got.len(), 0);
        }
        // k=0: every output must be EXACTLY 0.0, not merely close, for every
        // split pattern (the zero-fill-only path, no accumulation at all).
        {
            let a: Vec<f64> = vec![];
            let b: Vec<f64> = vec![];
            let a_shape = [70u32, 0];
            let b_shape = [0u32, 66];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (_shape_ref, data_ref) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert!(data_ref.iter().all(|&v| v == 0.0));
            for splits in split_patterns(70) {
                let data_got =
                    run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, data_ref.len(), &splits);
                assert_bit_eq(&data_ref, &data_got, &format!("k=0 splits={splits:?}"));
                assert!(data_got.iter().all(|&v| v == 0.0), "k=0 outputs must all be exactly 0.0, splits={splits:?}");
            }
        }
        // n=0: total_rows = m (>0) but every row has 0 columns.
        {
            let a = gen_data(&mut lcg, 70 * 65);
            let b: Vec<f64> = vec![];
            let a_shape = [70u32, 65];
            let b_shape = [65u32, 0];
            let a_strides = strides_of(&a_shape);
            let b_strides = strides_of(&b_shape);
            let (_shape_ref, data_ref) =
                matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b).unwrap();
            assert_eq!(data_ref.len(), 0);
            for splits in split_patterns(70) {
                let data_got =
                    run_partial_calls(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, 0, &splits);
                assert_eq!(data_got.len(), 0, "n=0 splits={splits:?}");
            }
        }
    }

    // --- status paths (1/2/4 same as matmul_blocked, plus new 3 for range) --

    #[test]
    fn partial_status_1_inner_dim_mismatch() {
        let mut out = vec![0.0; 8];
        let err = matmul_blocked_partial(&[2, 3], &[3, 1], 0, &[0.0; 6], &[4, 2], &[2, 1], 0, &[0.0; 8], &mut out, 0, 2).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn partial_status_1_rank_below_two_rejected() {
        let mut out = vec![0.0; 6];
        let err = matmul_blocked_partial(&[3], &[1], 0, &[1.0, 2.0, 3.0], &[3, 2], &[2, 1], 0, &[0.0; 6], &mut out, 0, 1).unwrap_err();
        assert_eq!(err, KernelError::ShapeIncompatible);
    }

    #[test]
    fn partial_status_2_rank_too_large_operand() {
        let mut shape = vec![1u32; 31];
        shape.push(2);
        let mut shape2 = shape.clone();
        shape2.push(1); // rank 33: over cap
        let strides2 = strides_of(&shape2);
        let mut out = vec![0.0; 4];
        let err = matmul_blocked_partial(&shape2, &strides2, 0, &[0.0], &[2, 2], &[2, 1], 0, &[0.0; 4], &mut out, 0, 1).unwrap_err();
        assert_eq!(err, KernelError::RankTooLarge);
    }

    #[test]
    fn partial_status_4_strides_out_of_bounds() {
        let a = vec![0.0; 6];
        let b = vec![0.0; 6];
        let mut out = vec![0.0; 4];
        let err = matmul_blocked_partial(&[2, 3], &[3, 2], 0, &a, &[3, 2], &[2, 1], 0, &b, &mut out, 0, 2).unwrap_err();
        assert_eq!(err, KernelError::StridesOutOfBounds);
    }

    /// New status path for this phase: an invalid row range (`row_start >
    /// row_end`, or `row_end` past the true `batch_size * m` bound) reuses
    /// status 3 (`SizeOverflow`) — the boundary `row_end == batch_size * m`
    /// itself must stay valid (exclusive upper bound, per the ABI contract).
    #[test]
    fn partial_status_3_row_range_invalid() {
        let a_shape = [4u32, 3];
        let b_shape = [3u32, 5];
        let a_strides = strides_of(&a_shape);
        let b_strides = strides_of(&b_shape);
        let a = vec![0.0; 12];
        let b = vec![0.0; 15];
        let mut out = vec![0.0; 20]; // 4*5
        let err_order = matmul_blocked_partial(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, &mut out, 3, 1).unwrap_err();
        assert_eq!(err_order, KernelError::SizeOverflow);
        let err_past_end =
            matmul_blocked_partial(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, &mut out, 0, 5).unwrap_err();
        assert_eq!(err_past_end, KernelError::SizeOverflow);
        // Exactly at the true bound (row_end == total_rows == 4): valid.
        let ok = matmul_blocked_partial(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, &mut out, 0, 4);
        assert!(ok.is_ok());
    }

    #[test]
    fn partial_status_3_out_len_mismatch() {
        let a_shape = [2u32, 3];
        let b_shape = [3u32, 2];
        let a_strides = strides_of(&a_shape);
        let b_strides = strides_of(&b_shape);
        let a = vec![0.0; 6];
        let b = vec![0.0; 6];
        let mut out = vec![0.0; 3]; // wrong: should be 4 (2*2)
        let err = matmul_blocked_partial(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, &mut out, 0, 2).unwrap_err();
        assert_eq!(err, KernelError::SizeOverflow);
    }

    /// Hand-checked pin, split across two partial calls straddling the
    /// single row boundary — must reproduce the classic fixture exactly.
    #[test]
    fn partial_hand_checked_two_d_core_split() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_strides = strides_of(&[2, 3]);
        let b_strides = strides_of(&[3, 2]);
        let data = run_partial_calls(&[2, 3], &a_strides, 0, &a, &[3, 2], &b_strides, 0, &b, 4, &[(0, 1), (1, 2)]);
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }
}

// Kern 06: imports exclusive to the row-partial core below — same
// freeze-preserving gate as that code (see its own comment for the full
// rationale); kept as a separate `use` so the STABLE wasm32 build (which
// never compiles that code) doesn't even see an unused-import warning.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
use crate::shape::{aligned_effective_strides_into, broadcast_shape_into, compute_strides_into, unravel_into, MAX_RANK};

// --- Kern 06: allocation-free row-partial core -----------------------------
//
// `nt_matmul_blocked_partial` (docs/kern-06-threads-spec.md) lets a worker
// thread compute an arbitrary contiguous slice of the FLAT OUTPUT ROW SPACE
// `[0, batch_size * m)` (batch index = row / m, row within batch = row % m)
// independently of every other worker, writing directly into the caller's
// pre-allocated output buffer instead of returning a fresh `Vec`. The
// parallel bit-identity law (spec) requires every output element's entire
// k-accumulation chain to run on exactly one thread and forbids any partial-
// sum recombination — a row-range split satisfies this trivially because
// each output ROW (and therefore every element in it) is assigned to exactly
// one partial call by construction of the caller's partition, and each
// call's own accumulation is the SAME `mb { kb { pack A; nb { pack B;
// microkernel } } }` loop nest as `matmul_2d_blocked` above (same
// `pack_a_tile`/`pack_b_tile` arithmetic reproduced by the `_into` variants
// below — pure data movement, see module doc law #1 — and the SAME
// `accumulate_pair`/`accumulate_single`/`micro_tile` functions, reused
// verbatim, not duplicated, since those are not frozen against caller
// addition). MC-aligned split points are a performance recommendation only:
// an arbitrary (even MC-misaligned) row range is computed via the identical
// per-output-element ascending-k accumulator chain regardless of where the
// `mb` tile loop happens to start/stop, so bit-identity holds for ANY
// partition (proven by the equivalence test grid below, not merely assumed).
//
// **Allocation-freedom (mechanical, proven by `crates/core/tests/
// zero_alloc.rs`).** The full call graph below never touches the heap:
// `checked_element_count`/`product`/`validate_strided_bounds` already don't
// allocate (they only read slices); `broadcast_shape_into`/
// `aligned_effective_strides_into`/`compute_strides_into`/`unravel_into`
// (shape.rs, Kern 06) are the no-alloc twins of the `Vec`-returning helpers
// `matmul_blocked` uses above, writing into fixed `[u32; MAX_RANK]` stack
// buffers instead; `pack_a_tile_into`/`pack_b_tile_into` below write into
// fixed `[f64; MC * KC]` / `[f64; KC * NC]` stack buffers (8 KiB each, since
// MC = NC = KC = 32) instead of allocating a `Vec` per tile the way
// `pack_a_tile`/`pack_b_tile` do. No `Vec`, `String`, `Box`, or any other
// heap type appears anywhere in this section's functions or their callees.
//
// **Freeze gate (Acceptance Criterion 1 + Freeze-Disziplin, hard
// requirement).** `nt_matmul_blocked_partial` is a `#[no_mangle] extern "C"`
// function — on `wasm32-unknown-unknown` that unconditionally makes it a
// WASM *export*, regardless of whether anything else calls it, so its mere
// presence in a compilation changes the produced `.wasm`'s bytes (confirmed
// empirically before adding this gate: `pnpm build:wasm`'s artifact hash and
// export count both changed the moment this code was added ungated). Since
// `pnpm build:wasm` (stable toolchain) and `pnpm build:wasm:threads`
// (pinned nightly, `+atomics`) compile the exact same `crates/core/src`
// source tree, the four functions below (and their ABI wrapper in `abi.rs`)
// are gated `#[cfg(any(not(target_arch = "wasm32"), target_feature =
// "atomics"))]`: present for native builds (so `cargo test` — always native,
// no `--target` — exercises them fully) and for the threads wasm32 build
// (`-C target-feature=+...,+atomics,...`, per docs/kern-06-threads-spec.md),
// absent for the plain wasm32 build (`.cargo/config.toml` sets only
// `+simd128`, never `+atomics`) — which is exactly the STABLE frozen
// baseline. Verified: `pnpm build:wasm`'s artifact is byte-identical
// before/after this phase with this gate in place (see the results doc's
// hash comparison).

/// No-alloc twin of [`pack_a_tile`]: same pure-data-movement gather (see
/// module doc law #1 — no arithmetic, so identical to the allocating version
/// bit-for-bit), writing into a caller-owned `[f64; MC * KC]` stack buffer at
/// `out[mi * kc + kj]` instead of a fresh `Vec`. Caller must have `mc <= MC`
/// and `kc <= KC` (always true here: both are `.min(MC)`/`.min(KC)` at every
/// call site) so `mi * kc + kj < MC * KC` never runs off the buffer.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
#[allow(clippy::too_many_arguments)]
fn pack_a_tile_into(
    a_data: &[f64],
    a_base_off: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    mb: usize,
    mc: usize,
    kb: usize,
    kc: usize,
    out: &mut [f64; MC * KC],
) {
    for mi in 0..mc {
        let row_base = a_base_off + ((mb + mi) as u32) * a_row_stride;
        for kj in 0..kc {
            let idx = row_base + ((kb + kj) as u32) * a_col_stride;
            out[mi * kc + kj] = a_data.get(idx as usize).copied().unwrap_or(0.0);
        }
    }
}

/// No-alloc twin of [`pack_b_tile`] — same contract as [`pack_a_tile_into`],
/// writing into a caller-owned `[f64; KC * NC]` stack buffer.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
#[allow(clippy::too_many_arguments)]
fn pack_b_tile_into(
    b_data: &[f64],
    b_base_off: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    kb: usize,
    kc: usize,
    nb: usize,
    nc: usize,
    out: &mut [f64; KC * NC],
) {
    for kj in 0..kc {
        let row_base = b_base_off + ((kb + kj) as u32) * b_row_stride;
        for nj in 0..nc {
            let idx = row_base + ((nb + nj) as u32) * b_col_stride;
            out[kj * nc + nj] = b_data.get(idx as usize).copied().unwrap_or(0.0);
        }
    }
}

/// The blocked 2-D core restricted to a LOCAL row range `[row_start, row_end)`
/// within one batch slice's own `[0, m)` rows (global batch offset already
/// folded into `out_base_off` by the caller, exactly like
/// [`matmul_2d_blocked`]'s `mb`/`kb`/`nb` tile loop nest — the only
/// structural difference is that the outer `mb` loop is bounded by
/// `[row_start, row_end)` instead of `[0, m)`, and this function explicitly
/// zero-fills its assigned rows first (the full-output core relies on its
/// caller's fresh `vec![0f64; out_size]`; a row-partial call writes into a
/// buffer it does NOT own exclusively, so it must zero only ITS rows, never
/// touching rows other workers own). `row_end <= m` is the caller's
/// responsibility (upheld by `matmul_blocked_partial` below, which derives
/// `row_start`/`row_end` here from the intersection of the caller's global
/// range with this batch's own row span).
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
#[allow(clippy::too_many_arguments)]
fn matmul_2d_blocked_rows(
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
    row_start: usize,
    row_end: usize,
    a_pack_buf: &mut [f64; MC * KC],
    b_pack_buf: &mut [f64; KC * NC],
) {
    let (k, n) = (k as usize, n as usize);
    if row_start >= row_end || n == 0 {
        return;
    }
    let out_row_width = n as u32;
    for row in row_start..row_end {
        let row_off = (out_base_off + (row as u32) * out_row_width) as usize;
        out[row_off..row_off + n].fill(0.0);
    }
    if k == 0 {
        return; // already zeroed above — matches matmul_2d_blocked's k==0 short circuit
    }

    let mut mb = row_start;
    while mb < row_end {
        let mc = MC.min(row_end - mb);
        let mut kb = 0usize;
        while kb < k {
            let kc = KC.min(k - kb);
            pack_a_tile_into(a_data, a_base_off, a_row_stride, a_col_stride, mb, mc, kb, kc, a_pack_buf);
            let mut nb = 0usize;
            while nb < n {
                let nc = NC.min(n - nb);
                pack_b_tile_into(b_data, b_base_off, b_row_stride, b_col_stride, kb, kc, nb, nc, b_pack_buf);
                micro_tile(mc, nc, kc, &a_pack_buf[..mc * kc], &b_pack_buf[..kc * nc], out, out_base_off, out_row_width, mb, nb);
                nb += NC;
            }
            kb += KC;
        }
        mb += MC;
    }
}

/// Kern 06: allocation-free row-partial generalization of [`matmul_blocked`]
/// — same shape contract and validation (rank >= 2, `checked_element_count`,
/// `validate_strided_bounds`, inner-dim match; each call validates
/// independently, exactly as if it were the only caller), but computes and
/// writes only the output rows in the half-open range `[row_start, row_end)`
/// over the flat row space `[0, batch_size * m)` into the caller-supplied
/// `out` buffer (already sized to the FULL output length, like
/// `nt_matmul_blocked`'s `out_len` — workers write disjoint slices of the
/// SAME buffer). `row_start > row_end` or `row_end > batch_size * m` is
/// status 3 (`SizeOverflow`, reused — no new status codes), as is `out.len()`
/// not matching this call's own independently computed output length (same
/// differential-testing convention as every other kernel's `out_len` check).
/// An empty range (`row_start == row_end`) is a valid no-op — the "MC-aligned
/// split points are a performance recommendation, never a correctness
/// requirement" law from the spec, tested directly by the equivalence grid
/// below.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
#[allow(clippy::too_many_arguments)]
pub fn matmul_blocked_partial(
    a_shape: &[u32],
    a_strides: &[u32],
    a_offset: u32,
    a_data: &[f64],
    b_shape: &[u32],
    b_strides: &[u32],
    b_offset: u32,
    b_data: &[f64],
    out: &mut [f64],
    row_start: u32,
    row_end: u32,
) -> KResult<()> {
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
    let mut batch_out_buf = [0u32; MAX_RANK];
    let batch_rank = broadcast_shape_into(batch_a, batch_b, &mut batch_out_buf)?;
    let batch_out = &batch_out_buf[..batch_rank];

    let mut a_batch_eff_buf = [0u32; MAX_RANK];
    let mut b_batch_eff_buf = [0u32; MAX_RANK];
    aligned_effective_strides_into(batch_a, &a_strides[..a_strides.len() - 2], batch_rank, &mut a_batch_eff_buf);
    aligned_effective_strides_into(batch_b, &b_strides[..b_strides.len() - 2], batch_rank, &mut b_batch_eff_buf);
    let a_batch_eff = &a_batch_eff_buf[..batch_rank];
    let b_batch_eff = &b_batch_eff_buf[..batch_rank];

    let a_row_stride = a_strides[a_strides.len() - 2];
    let a_col_stride = a_strides[a_strides.len() - 1];
    let b_row_stride = b_strides[b_strides.len() - 2];
    let b_col_stride = b_strides[b_strides.len() - 1];

    // out_full_shape = [...batch_out, m, n] — same rank argument as
    // matmul_blocked (batch_a.len(), batch_b.len() <= MAX_RANK - 2, so
    // batch_rank <= MAX_RANK - 2 and out_rank <= MAX_RANK always).
    let out_rank = batch_rank + 2;
    debug_assert!(out_rank <= MAX_RANK);
    let mut out_shape_buf = [0u32; MAX_RANK];
    out_shape_buf[..batch_rank].copy_from_slice(batch_out);
    out_shape_buf[batch_rank] = m;
    out_shape_buf[batch_rank + 1] = n;
    let out_size = checked_element_count(&out_shape_buf[..out_rank])?;
    if out.len() as u32 != out_size {
        return Err(KernelError::SizeOverflow);
    }

    // Row-range validation over the TRUE (unbounded-safe) flat row count
    // `batch_size * m`. `product(batch_out)` (used by matmul_blocked's own
    // batch loop) wraps in u32 by design — sound there only because its
    // result is never used when it WOULD wrap (m==0 or n==0 collapses the
    // batch loop to all-no-op iterations regardless of the wrapped count).
    // Here the row-range check is the one place that count is used directly
    // as a correctness-relevant bound, so it is computed via saturating u64
    // arithmetic instead: `row_start`/`row_end` are `u32` (< 2^32 <<
    // u64::MAX), so a saturated total can never wrongly reject a legitimate
    // caller-supplied range — saturation only ever produces a total that is
    // either exact or astronomically larger than any u32, never smaller.
    let mut total_rows: u64 = m as u64;
    for &d in batch_out {
        total_rows = total_rows.saturating_mul(d as u64);
    }
    if (row_start as u64) > (row_end as u64) || (row_end as u64) > total_rows {
        return Err(KernelError::SizeOverflow);
    }
    if row_start == row_end {
        return Ok(()); // empty range: valid no-op, zero-fills nothing
    }

    let out_strides_full = {
        let mut buf = [0u32; MAX_RANK];
        compute_strides_into(&out_shape_buf[..out_rank], &mut buf);
        buf
    };
    let out_batch_strides = &out_strides_full[..batch_rank];
    let batch_strides_plain = {
        let mut buf = [0u32; MAX_RANK];
        compute_strides_into(batch_out, &mut buf);
        buf
    };

    // `m > 0` is guaranteed here: row_start < row_end <= total_rows, so
    // total_rows > 0, so (since total_rows = m * product(batch_out)) m > 0.
    let start_batch = row_start / m;
    let end_batch_inclusive = (row_end - 1) / m;

    let mut a_pack_buf = [0f64; MC * KC];
    let mut b_pack_buf = [0f64; KC * NC];
    let mut multi_buf = [0u32; MAX_RANK];

    for b_idx in start_batch..=end_batch_inclusive {
        let batch_row_lo = (b_idx as u64) * (m as u64);
        let batch_row_hi = batch_row_lo + (m as u64);
        let lo = (row_start as u64).max(batch_row_lo) as u32;
        let hi = (row_end as u64).min(batch_row_hi) as u32;
        let local_start = (lo as u64 - batch_row_lo) as usize;
        let local_end = (hi as u64 - batch_row_lo) as usize;

        unravel_into(b_idx, batch_out, &batch_strides_plain[..batch_rank], &mut multi_buf);
        let mut a_batch_off: u32 = a_offset;
        let mut b_batch_off: u32 = b_offset;
        let mut out_batch_off: u32 = 0;
        for i in 0..batch_rank {
            let ix = multi_buf[i];
            a_batch_off += ix * a_batch_eff[i];
            b_batch_off += ix * b_batch_eff[i];
            out_batch_off += ix * out_batch_strides[i];
        }

        matmul_2d_blocked_rows(
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
            out,
            out_batch_off,
            local_start,
            local_end,
            &mut a_pack_buf,
            &mut b_pack_buf,
        );
    }

    Ok(())
}
