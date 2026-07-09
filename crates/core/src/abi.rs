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
use crate::shape::KernelError;

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
    // Safety: same caller contract as nt_add (valid, typed, non-overlapping
    // regions of this module's own memory, stable during the call); the
    // strides arrays are `rank` u32s each, per the strided ABI convention.
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
}
