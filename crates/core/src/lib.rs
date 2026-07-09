//! NumType Kern 01 — from-scratch Rust kernels for `NDArray`'s number
//! crunching, compiled to WASM with a hand-rolled `extern "C"` ABI (no
//! wasm-bindgen, no wasm-pack, no external crates — `[dependencies]` is
//! empty by design).
//!
//! - [`shape`]: shared shape/stride primitives, ported line-by-line from
//!   `spike/src/runtime.ts` so floating-point accumulation order (and
//!   therefore bit-identity with the naive TS reference) matches exactly.
//! - [`kernels`]: pure, natively-unit-testable functions on slices — one
//!   module per op (add, matmul, sum, transpose, fill).
//! - [`abi`]: thin `#[no_mangle] pub extern "C"` wrappers exposing the
//!   kernels (plus the allocator) as WASM exports. See that module's doc
//!   comment for the full calling convention.
//!
//! `crate-type = ["cdylib", "rlib"]`: `cdylib` so `cargo build --target
//! wasm32-unknown-unknown --release` produces a `.wasm`; `rlib` so `cargo
//! test` can run the kernel-layer unit tests natively.

pub mod abi;
pub mod kernels;
pub mod shape;
