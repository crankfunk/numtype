# NumType

> NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors.

An n-dimensional array library for TypeScript with **compile-time shape checking** — matmul, broadcasting, and reduction shape mismatches appear as editor squiggles while you type, not as runtime crashes in production. Gradual by design: literal dimensions are checked statically, dynamic (`number`) dimensions degrade gracefully to runtime checks. Planned backend: from-scratch Rust/WASM kernels.

**Status:** private research project — Spike 01 (type-level shape system). See `docs/` for the competitive analysis and spike spec.
