//! Hand-rolled `extern "C"` ABI over WASM linear memory. No wasm-bindgen: the
//! only external interface is native WASM export machinery
//! (`#[no_mangle] pub extern "C" fn`), which the `wasm32-unknown-unknown`
//! target exposes automatically as WASM module exports, alongside the
//! module's linear `memory` (also auto-exported by this target — verified
//! empirically: a stock `cargo build --target wasm32-unknown-unknown
//! --release` with no linker flags produces a module whose `instance.exports`
//! already includes `memory`).
//!
//! ## Calling convention
//!
//! - Every pointer is a `u32` byte offset into the module's own `memory`
//!   export (WASM32 addresses are 32-bit).
//! - A **shape** argument is a pair `(ptr, rank)`: `rank` little-endian `u32`
//!   values at `ptr`, one per axis (row-major shape, same as `NDArray.shape`).
//! - A **data** argument is a pair `(ptr, len)`: `len` `f64` values at `ptr`
//!   (row-major, contiguous — same layout as `NDArray.data`, a `Float64Array`).
//! - Every op returns a `u32` **status code**: `0` ok, `1` shape-incompatible,
//!   `2` rank > 32, `3` size overflow (see `shape::KernelError` for exactly
//!   which condition maps to which code, including the deliberate reuse of
//!   `3` for "caller-supplied output length doesn't match the kernel's own
//!   computed output length" — see that type's doc comment), `4` strides out
//!   of bounds (Kern 03, strided entry points only — see below).
//!
//! ## Strided entry points (Kern 03)
//!
//! The `nt_*_strided` exports (plus `nt_materialize`) generalize their
//! contiguous originals to **views**: an operand becomes the quadruple
//! `(shape ptr/rank, strides ptr, offset, data ptr/len)` where `strides` is
//! `rank` little-endian `u32` *element* strides at its own pointer, `offset`
//! is a `u32` base *element* offset into the buffer, and `len` is the **full
//! buffer** length in elements (not the view's logical size — the bounds
//! check needs the real allocation size). Outputs are always freshly
//! allocated, contiguous, row-major; strided outputs don't exist in this
//! ABI. Caller-supplied strides are the first ABI input the kernels cannot
//! derive themselves, so every strided operand is validated (max reachable
//! index inside `len`, checked-u64 arithmetic) *before* any data access —
//! failure is status `4`. The contiguous entry points and kernels are
//! deliberately untouched (frozen v1 baseline, see
//! docs/kern-03-strided-spec.md).
//!
//! `nt_matmul_blocked` (Kern 04) reuses this exact quadruple convention and
//! status set — it is a drop-in alternative entry point for matmul only,
//! see `kernels::matmul_blocked` for the blocked/packed/SIMD128 core and
//! docs/kern-04-simd-blocking-spec.md for the design. `nt_matmul_strided`
//! itself stays untouched (frozen Kern-03 baseline, kept as the measurable
//! "before" and the native equivalence reference for the new kernel).
//!
//! **Defense-in-depth hardening (post-Kern-04):** the six entry points above
//! (`nt_add_strided`, `nt_matmul_strided`, `nt_sum_all_strided`,
//! `nt_sum_axis_strided`, `nt_matmul_blocked`, `nt_materialize`) additionally
//! validate their caller-declared `rank` and every `(ptr, len)` operand/
//! output pair *before* constructing any Rust slice over them — one step
//! earlier than the existing `validate_strided_bounds` check above, which
//! only runs once shape/strides/data are already slices. A `rank` exceeding
//! `shape::MAX_RANK` returns status `2` (`RankTooLarge`); a
//! `(ptr, len, elem_size)` triple whose implied byte region would violate
//! `from_raw_parts`'s safety contract (byte size > `isize::MAX`, or
//! `ptr + bytes` wrapping past the 32-bit address space) returns status `3`
//! (`SizeOverflow`) — both statuses already exist for exactly these failure
//! categories, just checked earlier and directly against the raw ABI
//! arguments. This closes a gap where a garbage-`rank`/garbage-`len` caller
//! could make `read_slice`/`read_slice_mut` build an invalid slice — UB by
//! `from_raw_parts`'s own contract the moment the slice is constructed, no
//! dereference required. No legitimate caller observes a behavior change:
//! every rank a real `NDArray` can have already satisfies `rank <=
//! MAX_RANK`, and every buffer `nt_alloc` can return already satisfies both
//! region checks (see `validate_rank`/`validate_region` below for the
//! detailed argument). The v1 entry points (`nt_add`, `nt_matmul`,
//! `nt_sum_all`, `nt_sum_axis`, `nt_transpose`, `nt_fill`) deliberately keep
//! the original trust-the-caller contract with no such prevalidation — they
//! are the frozen performance baseline and are out of scope for this pass.
//! - **The caller (TS) computes output shapes and allocates the output
//!   buffer** (via `nt_alloc`) *before* calling an op; the op writes into
//!   that buffer and returns status only (it does not report its own
//!   shape/length back — TS already knows what it asked for). This is the
//!   deliberate TS/Rust shape-logic duplication the spec calls a
//!   differential-testing feature: if either side's shape math is wrong,
//!   the buffer size and/or the visible values diverge from the reference,
//!   not just the status code.
//! - On any non-zero status the output buffer's contents are **unspecified
//!   and must not be read** — no partial-result contract is offered.
//! - A `ptr`/`len` (or `ptr`/`rank`) pair with `len`/`rank == 0` is valid
//!   regardless of the pointer's numeric value — this ABI never
//!   dereferences memory for a zero-length shape/data argument (see
//!   `read_slice`/`read_slice_mut` below for why that matters).
//!
//! ## Allocator
//!
//! `nt_alloc`/`nt_free` sit directly on `std::alloc::{alloc, dealloc}` with
//! an 8-byte-alignment convention for every allocation (safe for both `f64`
//! data buffers and `u32` shape buffers). No allocator crate: `std` on
//! `wasm32-unknown-unknown` already provides a global allocator backed by
//! `memory.grow`, confirmed by testing this module directly (an allocation
//! larger than the module's initial memory does trigger growth and returns
//! a valid pointer). `nt_alloc(0)` and any allocation failure both return
//! `0` — the caller must never allocate a real object of size 0 and must
//! treat pointer `0` as "no allocation" (never dereference it). Rust's
//! `dealloc` requires the *exact same* `(size, align)` layout used at
//! `alloc` time, so `nt_free`'s `bytes` argument must be the exact value
//! passed to the paired `nt_alloc` call — the loader/backend must track
//! this per allocation (it already does, since it computes every buffer's
//! byte length itself to do the copy-in/copy-out).

use crate::kernels;
use crate::shape::{KResult, KernelError, MAX_RANK};

const STATUS_OK: u32 = 0;

fn status_of(err: KernelError) -> u32 {
    err.status()
}

/// Build a shared slice over `len` `T`s at `ptr`. `std::slice::from_raw_parts`
/// requires a non-null, aligned pointer *even for a zero-length slice*
/// (documented Rust safety requirement) — but `ptr == 0` is exactly the
/// sentinel `nt_alloc` returns for a zero-byte request, and TS legitimately
/// calls into this ABI with zero-length shape/data arguments (rank-0
/// scalars, size-0 arrays — both are explicit spec edge cases). Special-case
/// `len == 0` to a real empty slice literal instead, so the null pointer is
/// never hand to `from_raw_parts`.
///
/// # Safety
/// For `len > 0`, `ptr` must denote `len` valid, correctly aligned,
/// initialized `T`s in this module's own linear memory, stable for the
/// duration of the borrow.
unsafe fn read_slice<'a, T>(ptr: u32, len: u32) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr as *const T, len as usize) }
    }
}

/// Mutable counterpart of `read_slice` — same zero-length special case.
///
/// # Safety
/// For `len > 0`, `ptr` must denote `len` valid, correctly aligned `T`s,
/// exclusively borrowed for the duration of the borrow (no other live
/// reference into the same bytes).
unsafe fn read_slice_mut<'a, T>(ptr: u32, len: u32) -> &'a mut [T] {
    if len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len as usize) }
    }
}

/// Defense-in-depth (post-Kern-04 hardening): validate a caller-declared
/// `rank` *before* it is ever used as the length of a shape/strides slice.
/// Reuses `shape::MAX_RANK` and `KernelError::RankTooLarge` (status 2) — the
/// exact constant and status the kernels themselves already enforce for an
/// over-rank shape via `shape::checked_element_count`. A legitimate caller
/// never observes a behavior change: every real `NDArray` already satisfies
/// `rank <= MAX_RANK`, so this only makes the rejection happen earlier —
/// before `rank` is ever handed to `read_slice`/`read_slice_mut` as a slice
/// length, rather than after a kernel-internal shape check.
fn validate_rank(rank: u32) -> KResult<()> {
    if rank as usize > MAX_RANK {
        return Err(KernelError::RankTooLarge);
    }
    Ok(())
}

/// Defense-in-depth: validate a `(ptr, len, elem_size)` region *before* it is
/// ever handed to `read_slice`/`read_slice_mut`. `std::slice::from_raw_parts[_mut]`
/// require the resulting slice's total byte size not to exceed `isize::MAX`
/// — on `wasm32-unknown-unknown`, `isize` is 32-bit, so that bound is
/// `i32::MAX` — and constructing a slice that violates it is immediate UB by
/// that function's own safety contract, with no dereference required at all.
/// Two checks, both against checked `u64` arithmetic (no wraparound risk: a
/// `u32` `len`/`ptr` times a small constant `elem_size` fits comfortably in
/// `u64`):
/// - `bytes = len as u64 * elem_size as u64` must fit in `i32::MAX`.
/// - `ptr as u64 + bytes` must not exceed the 32-bit address space (`2^32`)
///   — the region must not run off the end of linear memory.
///
/// Both violations reuse `KernelError::SizeOverflow` (status 3) — the same
/// status the kernels already return when a shape's byte size doesn't fit
/// `u32` (`shape::checked_element_count`); this is the same failure
/// category, just checked earlier and directly against the raw ABI
/// arguments instead of shape-derived ones. No new ABI status codes.
///
/// A legitimate caller never observes a behavior change: every `(ptr, len)`
/// pair TS ever passes either came from `nt_alloc` — which can only return a
/// pointer/size that `std::alloc::Layout::from_size_align` accepted, itself
/// requiring the size to fit `isize::MAX`, well within a 32-bit address
/// space starting at a real allocation — or is a shape/strides length
/// already bounded by `validate_rank`. This only ever rejects a region no
/// `nt_alloc` result could have produced, i.e. caller-side corruption or a
/// garbage argument.
///
/// `len == 0` always passes trivially (`bytes == 0`, so both checks are
/// vacuous regardless of `ptr`), matching `read_slice`/`read_slice_mut`'s
/// own "a zero-length pair is valid regardless of pointer" rule.
fn validate_region(ptr: u32, len: u32, elem_size: u32) -> KResult<()> {
    let bytes = (len as u64) * (elem_size as u64);
    if bytes > i32::MAX as u64 {
        return Err(KernelError::SizeOverflow);
    }
    if (ptr as u64) + bytes > 1u64 << 32 {
        return Err(KernelError::SizeOverflow);
    }
    Ok(())
}

/// Return the first `Err` found in `checks`, in order, or `None` if every
/// check passed. A small combinator so each hardened entry point can list
/// its rank/region checks as a flat, auditable array up front instead of a
/// chain of nested `if let Err`. All the checks in `checks` are pure
/// numeric comparisons over already-computed `u32` arguments (no pointer is
/// ever dereferenced by `validate_rank`/`validate_region`), so evaluating
/// every element of the array before this function inspects them is always
/// safe, regardless of which one(s) fail.
fn first_error(checks: &[KResult<()>]) -> Option<KernelError> {
    for check in checks {
        if let Err(e) = check {
            return Some(*e);
        }
    }
    None
}

/// Allocate `bytes` bytes (8-byte aligned) in linear memory, returning a
/// pointer, or `0` on a zero-size request or allocation failure.
#[no_mangle]
pub extern "C" fn nt_alloc(bytes: u32) -> u32 {
    if bytes == 0 {
        return 0;
    }
    let layout = match std::alloc::Layout::from_size_align(bytes as usize, 8) {
        Ok(l) => l,
        Err(_) => return 0,
    };
    let ptr = unsafe { std::alloc::alloc(layout) };
    ptr as u32
}

/// Free a pointer previously returned by `nt_alloc`. `bytes` MUST be the
/// exact value passed to that `nt_alloc` call. A `ptr` of `0` (or `bytes`
/// of `0`) is a no-op, matching `nt_alloc`'s "no allocation" sentinel.
#[no_mangle]
pub extern "C" fn nt_free(ptr: u32, bytes: u32) {
    if ptr == 0 || bytes == 0 {
        return;
    }
    let layout = match std::alloc::Layout::from_size_align(bytes as usize, 8) {
        Ok(l) => l,
        Err(_) => return,
    };
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
}

/// Broadcasting elementwise add. `out_data_ptr`/`out_len` must be a buffer
/// TS has already sized to the broadcast output shape it computed.
#[no_mangle]
pub extern "C" fn nt_add(
    a_shape_ptr: u32,
    a_rank: u32,
    a_data_ptr: u32,
    a_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_data_ptr: u32,
    b_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Safety: every pointer/len pair here denotes a region of this module's
    // own linear memory that the caller (TS, via the ABI contract) promises
    // is valid, correctly typed, non-overlapping between distinct
    // arguments, and stable for the duration of this call (no `nt_alloc`/
    // `nt_free`/`memory.grow` happens *during* a kernel call — only
    // between calls).
    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_len) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_len) };

    match kernels::add::add(a_shape, a_data, b_shape, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// 2-D matmul core + batch broadcasting. Both operands must already be
/// rank >= 2 (1-D promotion happens TS-side, before this call); the output
/// is the unsqueezed `[...batch, m, n]` shape (TS squeezes afterward).
#[no_mangle]
pub extern "C" fn nt_matmul(
    a_shape_ptr: u32,
    a_rank: u32,
    a_data_ptr: u32,
    a_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_data_ptr: u32,
    b_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_len) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_len) };

    match kernels::matmul::matmul(a_shape, a_data, b_shape, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Sum every element of `data` down to a single `f64`, written to
/// `out_data_ptr` (a 1-element buffer). Always succeeds.
#[no_mangle]
pub extern "C" fn nt_sum_all(data_ptr: u32, len: u32, out_data_ptr: u32) -> u32 {
    let data: &[f64] = unsafe { read_slice(data_ptr, len) };
    let total = kernels::sum::sum_all(data);
    let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, 1) };
    dst[0] = total;
    STATUS_OK
}

/// Sum-reduce along `axis` (may be negative, counting from the end).
#[no_mangle]
pub extern "C" fn nt_sum_axis(
    shape_ptr: u32,
    rank: u32,
    data_ptr: u32,
    len: u32,
    axis: i32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, len) };

    match kernels::sum::sum_axis(shape, data, axis) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Reverse every axis (copy, not a view — WASM32-unknown-unknown has no
/// strided view concept exposed here; see spec "out of scope").
#[no_mangle]
pub extern "C" fn nt_transpose(shape_ptr: u32, rank: u32, data_ptr: u32, len: u32, out_data_ptr: u32, out_len: u32) -> u32 {
    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, len) };

    match kernels::transpose::transpose(shape, data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 03: broadcasting elementwise add over two strided views. See the
/// "Strided entry points" section of the module doc for the operand
/// quadruple convention; output contract matches `nt_add`.
#[no_mangle]
pub extern "C" fn nt_add_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc): rank first, so a
    // garbage rank reports status 2 rather than being caught downstream by
    // the region checks (which would misreport it as status 3).
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    // Safety: same caller contract as nt_add (valid, typed, non-overlapping
    // regions of this module's own memory, stable during the call); the
    // strides arrays are `rank` u32s each, per the strided ABI convention.
    // rank and every (ptr, len) region above have already been
    // prevalidated above, so from_raw_parts[_mut]'s safety contract (byte
    // size <= isize::MAX, no address-space wraparound) cannot be violated
    // here by a garbage caller argument.
    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::add::add_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 03: matmul over two strided views. Same promotion contract as
/// `nt_matmul` (operands arrive rank >= 2; TS squeezes afterward); the axis
/// added by 1-D promotion carries stride 0 by TS convention.
#[no_mangle]
pub extern "C" fn nt_matmul_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc): rank first, so a
    // garbage rank reports status 2 rather than being caught downstream by
    // the region checks (which would misreport it as status 3).
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::matmul::matmul_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 03: full sum of a strided view, accumulated in the view's logical
/// row-major order (order-sensitive — see `kernels::sum::sum_all_strided`).
/// Fallible, unlike `nt_sum_all`: caller strides must validate first.
#[no_mangle]
pub extern "C" fn nt_sum_all_strided(
    shape_ptr: u32,
    rank: u32,
    strides_ptr: u32,
    offset: u32,
    data_ptr: u32,
    data_len: u32,
    out_data_ptr: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc). Output is implicit
    // len=1 (a single f64), matching the unconditional `read_slice_mut(...,
    // 1)` below.
    if let Err(e) = validate_rank(rank) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(shape_ptr, rank, 4),
        validate_region(strides_ptr, rank, 4),
        validate_region(data_ptr, data_len, 8),
        validate_region(out_data_ptr, 1, 8),
    ]) {
        return status_of(e);
    }

    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let strides: &[u32] = unsafe { read_slice(strides_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, data_len) };

    match kernels::sum::sum_all_strided(shape, strides, offset, data) {
        Ok(total) => {
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, 1) };
            dst[0] = total;
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 03: sum-reduce a strided view along `axis` (may be negative).
#[no_mangle]
pub extern "C" fn nt_sum_axis_strided(
    shape_ptr: u32,
    rank: u32,
    strides_ptr: u32,
    offset: u32,
    data_ptr: u32,
    data_len: u32,
    axis: i32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc).
    if let Err(e) = validate_rank(rank) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(shape_ptr, rank, 4),
        validate_region(strides_ptr, rank, 4),
        validate_region(data_ptr, data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let strides: &[u32] = unsafe { read_slice(strides_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, data_len) };

    match kernels::sum::sum_axis_strided(shape, strides, offset, data, axis) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 04: blocked + packed + SIMD128 generalization of `nt_matmul_strided`
/// — identical signature and contract (see
/// `kernels::matmul_blocked::matmul_blocked`'s doc comment for the
/// bit-identity argument). `resident.ts`'s `matmul()` routes here
/// unconditionally (no size-based dispatch this phase).
#[no_mangle]
pub extern "C" fn nt_matmul_blocked(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc): rank first, so a
    // garbage rank reports status 2 rather than being caught downstream by
    // the region checks (which would misreport it as status 3).
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::matmul_blocked::matmul_blocked(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 03: gather a strided view into a contiguous row-major buffer of the
/// same logical shape (`out_len` must equal the view's element count).
#[no_mangle]
pub extern "C" fn nt_materialize(
    shape_ptr: u32,
    rank: u32,
    strides_ptr: u32,
    offset: u32,
    data_ptr: u32,
    data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc).
    if let Err(e) = validate_rank(rank) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(shape_ptr, rank, 4),
        validate_region(strides_ptr, rank, 4),
        validate_region(data_ptr, data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let strides: &[u32] = unsafe { read_slice(strides_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, data_len) };

    match kernels::materialize::materialize(shape, strides, offset, data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Fill `out_len` elements at `out_data_ptr` with `value`. Always succeeds.
#[no_mangle]
pub extern "C" fn nt_fill(out_data_ptr: u32, out_len: u32, value: f64) -> u32 {
    let result = kernels::fill::fill(out_len, value);
    let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
    dst.copy_from_slice(&result);
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nt_alloc(0)` must return the `0` sentinel, never a real pointer.
    #[test]
    fn alloc_zero_is_sentinel() {
        assert_eq!(nt_alloc(0), 0);
    }

    /// `nt_free` on the zero-sentinel is always a no-op (never calls
    /// `dealloc` on pointer 0).
    #[test]
    fn free_zero_is_noop() {
        nt_free(0, 0);
        nt_free(0, 8); // also a no-op per the documented contract (ptr==0)
    }

    /// `read_slice`/`read_slice_mut`'s whole reason to exist: `len == 0`
    /// must produce an empty slice *without* calling `from_raw_parts[_mut]`
    /// — even when `ptr` is the null/sentinel value `0`. This is safe and
    /// meaningful to test on any host (no real memory is touched).
    ///
    /// NOTE — deliberately NOT tested here: an end-to-end `nt_alloc` ->
    /// write -> `nt_*` op -> read -> `nt_free` round trip through the
    /// public ABI. `nt_alloc` returns `u32` (correct: WASM32 addresses are
    /// genuinely 32-bit), but `cargo test` runs this crate's `rlib`
    /// *natively* on a 64-bit host, where `std::alloc::alloc` returns real
    /// 64-bit pointers — `ptr as u32` truncates them, and casting the
    /// truncated value back to a pointer produces a wild address. An
    /// earlier version of this test module did exactly that round trip and
    /// crashed the test binary with SIGSEGV (confirmed via `lldb`: the
    /// faulting address, e.g. `0xd3000a00`, is recognizably a
    /// zero-extended 32-bit value, not a real heap pointer — see
    /// docs/kern-01-ergebnisse.md for the full note). This isn't a bug in
    /// the ABI (on the real wasm32 target, `u32` pointers are exact and
    /// correct); it's a case where the ABI's own contract makes it
    /// *only* meaningfully testable by actually running inside wasm32 —
    /// which the differential TS suite does, against the real compiled
    /// `.wasm`, including explicit rank-0/size-0 cases.
    #[test]
    fn read_slice_zero_length_never_dereferences_null() {
        let f: &[f64] = unsafe { read_slice(0, 0) };
        assert_eq!(f, &[] as &[f64]);
        let u: &[u32] = unsafe { read_slice(0, 0) };
        assert_eq!(u, &[] as &[u32]);
    }

    #[test]
    fn read_slice_mut_zero_length_never_dereferences_null() {
        let f: &mut [f64] = unsafe { read_slice_mut(0, 0) };
        assert_eq!(f, &mut [] as &mut [f64]);
    }

    // --- defense-in-depth prevalidation helpers -----------------------------

    #[test]
    fn validate_rank_accepts_max_rank_rejects_above() {
        assert!(validate_rank(MAX_RANK as u32).is_ok());
        assert_eq!(validate_rank(MAX_RANK as u32 + 1), Err(KernelError::RankTooLarge));
        assert_eq!(validate_rank(u32::MAX), Err(KernelError::RankTooLarge));
    }

    #[test]
    fn validate_region_zero_len_always_ok_regardless_of_ptr() {
        assert!(validate_region(u32::MAX, 0, 8).is_ok());
    }

    #[test]
    fn validate_region_byte_size_boundary() {
        // Exactly isize::MAX (== i32::MAX on wasm32) bytes: ok. One more: not.
        assert!(validate_region(0, i32::MAX as u32, 1).is_ok());
        assert_eq!(validate_region(0, (i32::MAX as u32) + 1, 1), Err(KernelError::SizeOverflow));
    }

    #[test]
    fn validate_region_address_space_wraparound_boundary() {
        // A region ending exactly at 2^32 is allowed; one byte further wraps
        // past the 32-bit address space and is rejected.
        assert!(validate_region(u32::MAX - 7, 1, 8).is_ok());
        assert_eq!(validate_region(u32::MAX - 6, 1, 8), Err(KernelError::SizeOverflow));
    }

    // --- defense-in-depth prevalidation, per hardened entry point -----------
    //
    // NOTE: these garbage-rank/garbage-len cases are host-safe for the exact
    // reason the caveat above `read_slice_zero_length_never_dereferences_null`
    // rules out an end-to-end round trip: `cargo test` runs this crate
    // natively on a 64-bit host, where a real `u32` pointer produced by
    // truncating a 64-bit `nt_alloc` allocation would be wild if
    // dereferenced. Every value chosen below makes prevalidation
    // (`validate_rank`/`validate_region`) fail and return *before* the
    // function ever reaches a `read_slice`/`read_slice_mut` call — so no
    // pointer here is ever turned into a slice, let alone dereferenced. All
    // pointers are the sentinel `0`, which is only safe to pass here because
    // it is guaranteed to never be dereferenced. Do NOT repurpose these
    // tests to exercise a path that passes prevalidation and reaches a real
    // kernel call — that would reintroduce the same native-host hazard the
    // NOTE above already documents.

    #[test]
    fn add_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_add_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn add_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_add_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn matmul_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_matmul_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn matmul_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_matmul_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn sum_all_strided_garbage_rank_is_status_2() {
        assert_eq!(nt_sum_all_strided(0, u32::MAX, 0, 0, 0, 0, 0), KernelError::RankTooLarge.status());
    }

    #[test]
    fn sum_all_strided_garbage_len_is_status_3() {
        assert_eq!(nt_sum_all_strided(0, 0, 0, 0, 0, u32::MAX, 0), KernelError::SizeOverflow.status());
    }

    #[test]
    fn sum_axis_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_sum_axis_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn sum_axis_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_sum_axis_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn matmul_blocked_garbage_rank_is_status_2() {
        assert_eq!(
            nt_matmul_blocked(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn matmul_blocked_garbage_len_is_status_3() {
        assert_eq!(
            nt_matmul_blocked(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn materialize_garbage_rank_is_status_2() {
        assert_eq!(nt_materialize(0, u32::MAX, 0, 0, 0, 0, 0, 0), KernelError::RankTooLarge.status());
    }

    #[test]
    fn materialize_garbage_len_is_status_3() {
        assert_eq!(nt_materialize(0, 0, 0, 0, 0, u32::MAX, 0, 0), KernelError::SizeOverflow.status());
    }
    #[test]
    fn matmul_blocked_partial_garbage_rank_is_status_2() {
        assert_eq!(
            nt_matmul_blocked_partial(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn matmul_blocked_partial_garbage_len_is_status_3() {
        assert_eq!(
            nt_matmul_blocked_partial(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }
}

/// Kern 06: allocation-free row-partial generalization of
/// `nt_matmul_blocked` — identical 14-argument operand/output convention
/// (see `kernels::matmul_blocked::matmul_blocked_partial`'s doc comment for
/// the row-range/bit-identity contract), plus a half-open range
/// `[row_start, row_end)` over the flat output row space
/// `[0, batch_size * m)`. Each worker calls this independently with its own
/// disjoint row range against the SAME shared output buffer; prevalidation
/// is identical to the other hardened entry points (rank first, then every
/// `(ptr, len)` region) — the row range itself is validated inside the
/// kernel (needs shape-derived `batch_size`/`m`, which aren't known yet at
/// this point), reusing status 3 (`SizeOverflow`, no new status codes).
///
/// **Freeze gate.** `#[no_mangle] extern "C"` makes this an unconditional
/// WASM export the moment it's compiled for `wasm32-unknown-unknown` —
/// gated `#[cfg(any(not(target_arch = "wasm32"), target_feature =
/// "atomics"))]` for exactly the reason `kernels::matmul_blocked`'s own
/// "Freeze gate" doc comment explains: present for native `cargo test` and
/// the threads wasm32 build (`+atomics`), absent from the plain wasm32
/// build (`pnpm build:wasm`, `+simd128` only) — which is the frozen
/// baseline `numtype_core.wasm` must stay byte-identical to.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn nt_matmul_blocked_partial(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
    row_start: u32,
    row_end: u32,
) -> u32 {
    // Defense-in-depth prevalidation (see module doc): rank first, so a
    // garbage rank reports status 2 rather than being caught downstream by
    // the region checks (which would misreport it as status 3).
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };
    let out: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };

    match kernels::matmul_blocked::matmul_blocked_partial(
        a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data, out, row_start, row_end,
    ) {
        Ok(()) => STATUS_OK,
        Err(e) => status_of(e),
    }
}

// ---------------------------------------------------------------------------
// Kern 07 (docs/kern-07-elementwise-vector-spec.md): elementwise sub/mul/div
// + dot/norm_sq vector reductions. Appended strictly after every
// pre-existing item in this file (freeze discipline: `nt_matmul_blocked_partial`
// above is the last pre-existing item; nothing above this comment is
// touched). NOT cfg-gated (unlike `nt_matmul_blocked_partial`) — these ops
// belong in BOTH the plain artifact (`pnpm build:wasm`) and the threads
// artifact, so the plain artifact's hash necessarily changes this phase
// (Kern-04 precedent, disclosed in the spec's freeze section).
// ---------------------------------------------------------------------------

/// Kern 07: broadcasting elementwise subtract over two strided views
/// (`a - b`). Identical operand-quadruple convention, prevalidation order,
/// and status set to `nt_add_strided` — see that function's doc comment.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn nt_sub_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::elementwise::sub_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 07: broadcasting elementwise multiply over two strided views
/// (`a * b`). Identical convention to `nt_sub_strided`/`nt_add_strided`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn nt_mul_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::elementwise::mul_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 07: broadcasting elementwise divide over two strided views
/// (`a / b`). Identical convention to `nt_sub_strided`/`nt_add_strided`.
/// Pure IEEE 754 (see `kernels::elementwise` module doc) — never returns a
/// non-zero status for a zero divisor; `x/0.0`, `0.0/0.0` flow through as
/// signed infinity / NaN in the output data.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn nt_div_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
    out_len: u32,
) -> u32 {
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, out_len, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::elementwise::div_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok((_out_shape, out_data)) => {
            if out_data.len() as u32 != out_len {
                return KernelError::SizeOverflow.status();
            }
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, out_len) };
            dst.copy_from_slice(&out_data);
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 07: 1-D inner product of two strided rank-1 views. Output is
/// implicit len=1 (a single f64), matching `nt_sum_all_strided`'s
/// convention (no `out_len` parameter). `dot_strided` itself validates
/// rank==1/equal-length as `ShapeIncompatible` (status 1) — this
/// prevalidation only covers rank-too-large (status 2) and region/size
/// overflow (status 3), same division of responsibility as every other
/// hardened entry point here.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub extern "C" fn nt_dot_strided(
    a_shape_ptr: u32,
    a_rank: u32,
    a_strides_ptr: u32,
    a_offset: u32,
    a_data_ptr: u32,
    a_data_len: u32,
    b_shape_ptr: u32,
    b_rank: u32,
    b_strides_ptr: u32,
    b_offset: u32,
    b_data_ptr: u32,
    b_data_len: u32,
    out_data_ptr: u32,
) -> u32 {
    if let Err(e) = validate_rank(a_rank).and(validate_rank(b_rank)) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(a_shape_ptr, a_rank, 4),
        validate_region(a_strides_ptr, a_rank, 4),
        validate_region(a_data_ptr, a_data_len, 8),
        validate_region(b_shape_ptr, b_rank, 4),
        validate_region(b_strides_ptr, b_rank, 4),
        validate_region(b_data_ptr, b_data_len, 8),
        validate_region(out_data_ptr, 1, 8),
    ]) {
        return status_of(e);
    }

    let a_shape: &[u32] = unsafe { read_slice(a_shape_ptr, a_rank) };
    let a_strides: &[u32] = unsafe { read_slice(a_strides_ptr, a_rank) };
    let a_data: &[f64] = unsafe { read_slice(a_data_ptr, a_data_len) };
    let b_shape: &[u32] = unsafe { read_slice(b_shape_ptr, b_rank) };
    let b_strides: &[u32] = unsafe { read_slice(b_strides_ptr, b_rank) };
    let b_data: &[f64] = unsafe { read_slice(b_data_ptr, b_data_len) };

    match kernels::vector::dot_strided(a_shape, a_strides, a_offset, a_data, b_shape, b_strides, b_offset, b_data) {
        Ok(total) => {
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, 1) };
            dst[0] = total;
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// Kern 07: sum of squares over every element of a strided view (any rank),
/// accumulated in LOGICAL row-major order. Output implicit len=1, same
/// convention as `nt_sum_all_strided` (which this otherwise mirrors
/// exactly, one operand instead of two, `v*v` instead of `v`).
#[no_mangle]
pub extern "C" fn nt_norm_sq_strided(
    shape_ptr: u32,
    rank: u32,
    strides_ptr: u32,
    offset: u32,
    data_ptr: u32,
    data_len: u32,
    out_data_ptr: u32,
) -> u32 {
    if let Err(e) = validate_rank(rank) {
        return status_of(e);
    }
    if let Some(e) = first_error(&[
        validate_region(shape_ptr, rank, 4),
        validate_region(strides_ptr, rank, 4),
        validate_region(data_ptr, data_len, 8),
        validate_region(out_data_ptr, 1, 8),
    ]) {
        return status_of(e);
    }

    let shape: &[u32] = unsafe { read_slice(shape_ptr, rank) };
    let strides: &[u32] = unsafe { read_slice(strides_ptr, rank) };
    let data: &[f64] = unsafe { read_slice(data_ptr, data_len) };

    match kernels::vector::norm_sq_strided(shape, strides, offset, data) {
        Ok(total) => {
            let dst: &mut [f64] = unsafe { read_slice_mut(out_data_ptr, 1) };
            dst[0] = total;
            STATUS_OK
        }
        Err(e) => status_of(e),
    }
}

/// New test module for this phase (Kern 07) — deliberately SEPARATE from the
/// pre-existing `mod tests` above `nt_matmul_blocked_partial` (freeze
/// discipline: never insert into that module). Same garbage-rank/garbage-len
/// prevalidation pattern as its own "defense-in-depth prevalidation, per
/// hardened entry point" section — every pointer here is the sentinel `0`,
/// safe only because prevalidation is guaranteed to reject before any
/// `read_slice`/`read_slice_mut` call is reached (see that section's NOTE
/// in the pre-existing module for the full native-host-safety argument).
#[cfg(test)]
mod kern07_abi_tests {
    use super::*;

    #[test]
    fn sub_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_sub_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn sub_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_sub_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn mul_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_mul_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn mul_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_mul_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn div_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_div_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn div_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_div_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn dot_strided_garbage_rank_is_status_2() {
        assert_eq!(
            nt_dot_strided(0, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            KernelError::RankTooLarge.status()
        );
    }

    #[test]
    fn dot_strided_garbage_len_is_status_3() {
        assert_eq!(
            nt_dot_strided(0, 0, 0, 0, 0, u32::MAX, 0, 0, 0, 0, 0, 0, 0),
            KernelError::SizeOverflow.status()
        );
    }

    #[test]
    fn norm_sq_strided_garbage_rank_is_status_2() {
        assert_eq!(nt_norm_sq_strided(0, u32::MAX, 0, 0, 0, 0, 0), KernelError::RankTooLarge.status());
    }

    #[test]
    fn norm_sq_strided_garbage_len_is_status_3() {
        assert_eq!(nt_norm_sq_strided(0, 0, 0, 0, 0, u32::MAX, 0), KernelError::SizeOverflow.status());
    }
}

