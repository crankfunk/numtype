# NumType

> NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors.

An n-dimensional array library for TypeScript with **compile-time shape checking** — matmul, broadcasting, and reduction shape mismatches appear as editor squiggles while you type, not as runtime crashes in production. Gradual by design: literal dimensions are checked statically, dynamic (`number`) dimensions degrade gracefully to runtime checks. Planned backend: from-scratch Rust/WASM kernels.

**Status:** private research project. Done and verified: the type layer (Spike 01 — broadcast/matmul/reduce at the type level, gradual, errors at the offending argument), from-scratch Rust/WASM kernels bit-identical to the naive TS reference (Kern 01), zero-copy residency (Kern 02 — `WNDArray` lives in WASM linear memory, full dispose/GC lifecycle), and strided views (Kern 03 — O(1) `transpose()`, refcounted shared buffers, strided kernels, `contiguous()` escape hatch). See `docs/` for the competitive analysis, per-phase specs, and results; `FOLLOWUPS.md` for the backlog (next up: SIMD128 + blocking for matmul).
