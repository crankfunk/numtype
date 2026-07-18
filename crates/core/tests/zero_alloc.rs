//! Kern 06: mechanical zero-allocation proof for `matmul_blocked_partial`'s
//! entire call graph (docs/kern-06-threads-spec.md, ABI addition contract).
//!
//! **Why a dedicated integration-test binary, not a `#[cfg(test)] mod` in
//! `matmul_blocked.rs`.** Cargo compiles every file under `tests/` into its
//! own separate test binary/process (Cargo Book, "Tests" chapter — each
//! integration test file is its own crate). A `#[global_allocator]` here
//! therefore governs allocations only within THIS process, not the crate's
//! unit-test binary or the wasm32 release artifact. That isolation matters
//! because `cargo test`'s default harness runs `#[test]` functions on a
//! pool of worker threads, and a process-global allocation counter would
//! otherwise be polluted by *unrelated* tests allocating concurrently on
//! other threads — a real race, not a hypothetical one, since the crate's
//! unit-test binary alone already has 100+ tests. Putting both measurements
//! into ONE `#[test]` function in a file with no other test function removes
//! that race structurally: there is no other test in this process to
//! interleave with. (A per-thread counter via `thread_local!` would be the
//! usual alternative, but the spec explicitly forbids `thread_local!`
//! anywhere in this crate — see `abi.rs`'s module doc — because
//! `__tls_base` is only initialized in the winning instance under
//! multi-instantiation; this file sidesteps the question entirely by not
//! needing thread-local state at all.)
//!
//! Residual contamination handling: a `#[global_allocator]` counter can still
//! catch an allocation made by the test harness's own bookkeeping (e.g. stdout
//! capture plumbing) on another thread DURING the measurement window — observed
//! on CI (Linux, shared load) as a sporadic non-zero delta, though the property
//! itself (the call graph is allocation-free) holds. That contamination is
//! TRANSIENT (it inflates some measurements, not others); a genuine allocation
//! inside the measured call graph would be INHERENT (present in every run). The
//! zero-alloc assertion below therefore measures the same call N times and
//! checks the MINIMUM delta — filtering the transient harness noise structurally
//! without weakening the proof (a real allocation keeps min >= 1). The
//! dedicated-binary structure additionally rules out cross-test contamination.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use numtype_core::kernels::matmul_blocked::{matmul_blocked, matmul_blocked_partial};
use numtype_core::shape::compute_strides;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

/// Both checks live in ONE test function (see module doc: no other test in
/// this binary, so no cross-thread contamination of the global counter).
#[test]
fn matmul_blocked_partial_is_zero_alloc_matmul_blocked_is_not() {
    // Build test operands BEFORE any measurement window starts (Vec
    // construction here is expected to allocate and must not be counted).
    let (m, k, n) = (70usize, 65usize, 66usize); // straddles the MC/KC/NC=32 tile boundary
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.25 - 3.0).collect();
    let a_shape = [m as u32, k as u32];
    let b_shape = [k as u32, n as u32];
    let a_strides = compute_strides(&a_shape);
    let b_strides = compute_strides(&b_shape);
    let mut out = vec![0.0f64; m * n];

    // --- Zero-allocation proof: matmul_blocked_partial's core call graph ---
    // Measure the SAME call N times and check the MINIMUM delta. A real
    // allocation inside matmul_blocked_partial would be INHERENT — present in
    // EVERY measurement (min >= 1). The residual contamination this global
    // counter is exposed to (the harness's own bookkeeping / stdout-capture
    // plumbing allocating on another thread during the tiny measurement window —
    // see module doc; observed on CI/Linux under shared load as a sporadic
    // delta of a few) is TRANSIENT: it inflates some measurements but not
    // others, so the minimum filters it out structurally without weakening the
    // proof.
    const N: usize = 16;
    let mut min_delta = usize::MAX;
    for _ in 0..N {
        let before_partial = alloc_count();
        let status = matmul_blocked_partial(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b, &mut out, 0, m as u32);
        let after_partial = alloc_count();
        assert!(status.is_ok(), "matmul_blocked_partial call itself must succeed: {status:?}");
        min_delta = min_delta.min(after_partial - before_partial);
    }
    assert_eq!(
        min_delta, 0,
        "matmul_blocked_partial's call graph must be allocation-free (min delta over {N} runs = {min_delta})"
    );

    // --- Non-vacuity: the SAME counting allocator, over the SAME shapes,
    // via matmul_blocked (which DOES allocate — Vec-returning helpers,
    // pack_a_tile/pack_b_tile's fresh Vec per tile, the output Vec itself) —
    // must observe a NON-ZERO delta. If this assertion ever failed, it would
    // mean the counting allocator itself is broken (e.g. wired to the wrong
    // global, or optimized away), which would silently invalidate the zero
    // result above too. This is exactly the failure mode a "prove the meter
    // works" non-vacuity check exists to catch.
    let before_blocked = alloc_count();
    let result = matmul_blocked(&a_shape, &a_strides, 0, &a, &b_shape, &b_strides, 0, &b);
    let after_blocked = alloc_count();
    assert!(result.is_ok(), "matmul_blocked call itself must succeed");
    assert!(
        after_blocked - before_blocked > 0,
        "sanity/non-vacuity: matmul_blocked must allocate (delta={}) — proves the counting allocator actually observes allocations",
        after_blocked - before_blocked
    );

    // Cross-check while we're here: same shapes/data, partial (one full-range
    // call) vs. the allocating original must agree bit-for-bit too — this
    // integration binary is a fresh crate instance, so this is an
    // independent confirmation of the equivalence law already covered in
    // depth by matmul_blocked.rs's own unit test grid.
    let (_shape_ref, data_ref) = result.unwrap();
    assert_eq!(
        data_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "matmul_blocked_partial (single full-range call) must match matmul_blocked bit-for-bit"
    );
}
