# NumType

NumPy-like n-dimensional array library: TypeScript type-level shape checking + (later) from-scratch Rust/WASM kernels. Research project — the explicit goal is probing the limits of what's feasible.

## Hard constraints (user-set, 2026-07-09)

- **No external libraries.** All kernels and all type machinery written from scratch. Dev tooling (typescript, later test runners) is allowed; product/runtime dependencies are not. Never suggest pulling in `ndarray`/`faer`/BLAS bindings etc.
- Brand name: **NumType** (npm package name stays lowercase `numtype`).
- Private repo for now; planned open-source release if the approach proves out. Write code/README/spec docs in English (OSS-facing); internal research notes may be German.
- Research fan-outs stay small: targeted agents (≤3), no broad sweeps.

## USP (defensible form — sources in docs/wettbewerbsanalyse-und-usp.md)

NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors — gradual, with a `number`-dim escape hatch for dynamic shapes. Python provably cannot do this statically today; in TS it is newly tractable but unproven at scale. That gap is the project.

## Current phase

Kern 01: from-scratch Rust/WASM kernels behind the existing NDArray API — spec in docs/kern-01-wasm-core-spec.md. No wasm-bindgen/wasm-pack: hand-rolled `extern "C"` ABI + hand-written dual-target loader; naive TS runtime stays as the differential-test reference (bit-identical, transcendental-free v1). Spike 01 (type layer) is done and committed — spec in docs/spike-01-type-layer-spec.md, results incl. verification addendum in docs/spike-01-ergebnisse.md.

## Toolchain note (2026-07-09)

Installed TypeScript is **7.0.2** — the native (Go) compiler generation, now `latest` on npm (6.0 is still `beta`). All researched recursion/instantiation limits below were documented for TS 5.x; treat them as hypotheses to verify empirically on 7.x, not as facts. `--extendedDiagnostics` works on 7.0.2.

## Key TS limits to respect (researched on TS 5.x, sourced in docs/wettbewerbsanalyse-und-usp.md §4)

- ~100 instantiation depth non-tail-recursive; ~1000 tail-recursive → write ALL recursive types accumulator/tail-recursive.
- Global ~5M type-instantiation budget per compilation (TS5, version-fragile) — track via `pnpm check:diag`.
- Tuple-length arithmetic is fine for *ranks* (small ints), never for large *dimension values* (no type-level products → reshape/flatten deferred, see FOLLOWUPS.md).
- Use `const` type parameters (TS 5.0+) so callers never need `as const`.
- Shape errors must surface at the offending argument with the shapes named in the message; hovers must show clean resolved tuples like `NDArray<[2, 4]>`.
