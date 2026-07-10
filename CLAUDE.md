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

Between phases — next candidates per FOLLOWUPS.md: small hardenings (v1 OOM path, test-list guard, ABI pre-validation), threads (own phase), or type-level follow-ons now unlocked by Kern 05's digit arithmetic (index bounds-checks, reshape/flatten products). Done, verified, and committed: Spike 01 (type layer — docs/spike-01-*), Kern 01 (from-scratch kernels behind a hand-rolled `extern "C"` ABI, bit-identical to the naive TS reference — docs/kern-01-*), Kern 02 (zero-copy residency incl. fromArray Float64Array overload — docs/kern-02-*), Kern 03 (strided views: O(1) transpose, refcounted buffers, strided ABI entry points + status 4, `contiguous()` — docs/kern-03-*), Kern 04 (blocked+packed+SIMD128 matmul, bit-identical under the "bit-identity law": vectorize only ACROSS output elements, ascending-k single accumulator chain, no FMA/relaxed-simd — docs/kern-04-*; 2.1–3.25× over the Kern-03 scalar kernel, Kern-03's view-matmul penalty erased by packing), Kern 05 (slicing: O(1) slice views with the first nonzero offsets, zero Rust changes; type layer = gradual core rules + statically computed slice dims via from-scratch digit-string arithmetic in spike/src/slice-literal.ts — docs/kern-05-*). The naive TS runtime remains the correctness reference; the v1 copy-based backend remains the frozen performance baseline (its kernels/entry points stay byte-for-byte untouched). Build note: wasm builds need the repo-root `.cargo/config.toml` simd128 rustflag, and cargo config discovery is CWD-based — run all commands from the repo root (a compile_error! guard fires if the flag is lost). Every phase follows: binding spec doc → implementation → fresh-context verification → results doc with post-verification addendum → KB capture → commit.

## Commands

`pnpm check` (types) · `pnpm test:core` (v1 differential, 791) · `pnpm test:resident` (+`:gc` with --expose-gc) · `pnpm demo` (all three backends, asserted equal) · `pnpm bench:scaling` / `bench:chain` / `bench:strided` / `bench:blocked` / `bench:slice` · `cargo test --manifest-path crates/core/Cargo.toml`. Note: test scripts use EXPLICIT file lists in package.json — new test files must be added there manually (guard pending, see FOLLOWUPS.md).

## Toolchain note (2026-07-09)

Installed TypeScript is **7.0.2** — the native (Go) compiler generation, now `latest` on npm (6.0 is still `beta`). All researched recursion/instantiation limits below were documented for TS 5.x; treat them as hypotheses to verify empirically on 7.x, not as facts. `--extendedDiagnostics` works on 7.0.2.

## Obligatory workflow: capture findings in the coding-kb (user-mandated, 2026-07-09)

Every substantial slice of work (spike, phase, verification pass, benchmark) ends with a knowledge capture —
it is part of the Definition of Done, not optional:

1. **In-project:** findings go into the phase's results doc (`docs/*-ergebnisse.md`) — grounded in commands
   actually run, honest about failures and gaps (see the docs' "honesty rule").
2. **Cross-project:** every *general* lesson (disproven assumption, non-trivial gotcha, failed→working
   approach, transferable technique) is upserted into the coding-kb Obsidian vault as an atomic note —
   revise a related existing note rather than duplicating; correct notes that turned out wrong or imprecise
   (e.g. replace single-point figures with measured ranges). Follow the vault's
   `90-Meta/Capture-Workflow.md` (template, `status: seedling`, tags, `projekte: [numtype]`).
3. **Wire it:** link the note into the fitting MOC(s), then rebuild the graph (root op, command in
   Capture-Workflow.md) and verify the new note's link edges via a `coding-kb` query.
4. Before starting non-trivial work in a known domain, **consult** the KB first (`find` → `neighbors` →
   `read`) — do not re-derive what is already written down.

## Key TS limits to respect (researched on TS 5.x, sourced in docs/wettbewerbsanalyse-und-usp.md §4)

- ~100 instantiation depth non-tail-recursive; ~1000 tail-recursive → write ALL recursive types accumulator/tail-recursive.
- Global ~5M type-instantiation budget per compilation (TS5, version-fragile) — track via `pnpm check:diag`.
- Tuple-length arithmetic is fine for *ranks* (small ints), never for large *dimension values*. Since Kern 05, arithmetic over large literal dims IS available via digit-string types (`spike/src/slice-literal.ts`: subtraction + comparison, O(digit-count) recursion) — extendable (products for reshape/flatten would be O(digits²), see FOLLOWUPS.md), but never fall back to tuple-length-per-value.
- Use `const` type parameters (TS 5.0+) so callers never need `as const`.
- Shape errors must surface at the offending argument with the shapes named in the message; hovers must show clean resolved tuples like `NDArray<[2, 4]>`.
