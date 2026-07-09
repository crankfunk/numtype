# NumType — Kern 01: From-Scratch Rust/WASM Kernels (Spec)

Date: 2026-07-09 · Status: in progress

## Why (intent)

Spike 01 proved the type layer. Kern 01 replaces the naive TS runtime's number-crunching with **from-scratch
Rust kernels compiled to WASM** — no `ndarray`, no `faer`, no `wasm-bindgen`, no `wasm-pack`: a hand-rolled
`extern "C"` ABI over WASM linear memory plus a hand-written dual-target TS loader. This is the research goal
(probing what "everything from scratch" costs and teaches), and it keeps the product at **zero runtime
dependencies** on both sides. The naive TS runtime stays untouched as the **reference implementation** for
differential testing.

## Scope

### In scope

1. **Rust crate** `crates/core/` — `Cargo.toml` with an **empty `[dependencies]` table** (hard requirement),
   `crate-type = ["cdylib", "rlib"]` (cdylib for WASM, rlib so `cargo test` works), pinned via
   `rust-toolchain.toml` (channel `1.95.0`). Release profile: `opt-level = 3`, `lto = true`,
   `codegen-units = 1`, `panic = "abort"`.
   - **Kernel layer** (pure Rust functions on slices — unit-testable natively): elementwise add with NumPy
     broadcasting, matmul (2-D core + batch broadcasting; 1-D promotion stays TS-side, exactly as
     `spike/src/runtime.ts` does it), sum (full and per-axis), transpose (axis-reversal copy), fill.
     All f64, contiguous row-major, rank ≤ 32.
   - **ABI layer** (thin `#[no_mangle] pub extern "C"` wrappers): `nt_alloc(bytes: u32) -> u32`,
     `nt_free(ptr: u32, bytes: u32)`, `nt_add`, `nt_matmul`, `nt_sum_all`, `nt_sum_axis`, `nt_transpose`,
     `nt_fill`. Shapes are passed as little-endian `u32` arrays in linear memory (ptr + rank), data as `f64`
     arrays (ptr + len). **TS computes output shapes** (the logic already exists in `runtime.ts`); Rust
     kernels **re-validate** inputs and return a status code: `0` ok, `1` shape-incompatible, `2` rank > 32,
     `3` size overflow. The deliberate TS/Rust duplication of shape logic is a differential-testing feature.
   - Allocator: implement `nt_alloc`/`nt_free` on top of Rust's global allocator (`Vec`-leak /
     `Vec::from_raw_parts` reclaim, or `std::alloc::alloc`/`dealloc` with a stored layout convention —
     implementor's choice, documented). No allocator crates.
2. **TS loader + backend** `spike/src/wasm/`:
   - `loader.ts`: dual-target, hand-written. Node: `fs/promises.readFile` + `WebAssembly.instantiate`;
     browser: `fetch` + `instantiateStreaming` (feature-detect via `typeof process`). Exposes
     `async initCore(): Promise<Core>` with typed exports.
   - `backend.ts`: op wrappers used by a WASM-backed twin of the runtime: copy operands into wasm scratch
     memory, run kernel, copy result out, free scratch. **v1 is copy-in/copy-out by design** — zero-copy
     resident buffers (FinalizationRegistry, view-detach handling) are v2 (FOLLOWUPS). Measure the copy
     overhead instead of engineering around it prematurely.
   - **Memory rule (hard):** never cache a `Float64Array`/`Uint32Array` view across any call that may
     allocate — `memory.grow` detaches views. Always re-derive views from `memory.buffer` at use time.
3. **Determinism contract:** Rust kernels must use **the same iteration/accumulation order** as
   `spike/src/runtime.ts` (read it first; mirror its loop orders exactly). v1 kernels use only `+` and `*`
   (IEEE-defined in WASM) — no transcendentals. Therefore differential tests assert **bit-identical** results
   (compare via `Float64Array`→`BigUint64Array` bits or `Object.is`), not epsilon.
4. **Tests:**
   - `cargo test` (native) on the kernel layer: per-kernel unit tests incl. edges (rank-0 via TS-side
     semantics where applicable, broadcast with interior 1s, batch matmul with broadcast batch dims, axis
     bounds, empty/size-0 arrays).
   - Differential suite `spike/tests-runtime/*.test.ts` using **`node:test` + `node:assert`** (built-in —
     zero new dependencies): a hand-written seeded PRNG (splitmix64, same algorithm in TS) generates ≥100
     cases per op across ranks 0–4, dims 1–8, including broadcast-1 placements and batch matmul; assert
     WASM ≡ naive TS **bitwise**. Negative paths: incompatible shapes → status ≠ 0 → TS throws with the
     shapes named in the message.
   - Type layer untouched: `pnpm check` stays green.
5. **Bench (informational):** matmul `[256,256] × [256,256]` — wall-clock naive TS vs WASM (incl. copies),
   printed by a small script (`pnpm bench:core`), numbers recorded in the findings doc. No threshold gate.
6. **Build wiring:** `pnpm build:wasm` = `cargo build --manifest-path crates/core/Cargo.toml --target
   wasm32-unknown-unknown --release` + copy the `.wasm` to `spike/src/wasm/numtype_core.wasm`.
   The `.wasm` is a **build artifact**: add `*.wasm` to `.gitignore` (KB lesson: checked-in wasm is not
   byte-stable across hosts and panics embed file:line). Scripts that need it (`test:core`, `bench:core`,
   demo) must chain the build (KB lesson: wasm must exist before tsc/tests run).
7. **Demo:** extend `spike/demo.ts` to run the same showcase ops on both backends and print both, asserting
   equality inline (so `pnpm demo` fails loudly on divergence).
8. **Findings** `docs/kern-01-ergebnisse.md` (English): ABI documentation (exports, layouts, status codes),
   measured bench numbers, gotchas hit (with evidence), any deviation from this spec with reasons.

### Out of scope (v2+, FOLLOWUPS)

Zero-copy resident buffers + FinalizationRegistry ownership; strided (non-materialized) kernels; SIMD
(wasm simd128); threads; transcendental ops (would break bit-parity — needs its own determinism decision);
dtype system beyond f64; publishing layout / renaming `spike/`.

## Acceptance criteria

- `cargo test --manifest-path crates/core/Cargo.toml` green; `[dependencies]` empty.
- `pnpm build:wasm` produces the artifact from a clean target dir.
- `pnpm test:core` green: all differential cases **bit-identical**, negative paths throw with shapes named.
- `pnpm check` green (type layer untouched); `pnpm demo` runs both backends, results equal, hand-checkable
  output unchanged.
- `pnpm bench:core` prints naive-vs-WASM matmul numbers; recorded in findings doc.
- No new entries in `package.json` dependencies/devDependencies; no crates in `[dependencies]`.

## Honesty rule

Same as Spike 01: nothing faked or silently weakened. If bit-identity fails for a case class, do not relax
to epsilon silently — document the exact case, the bit difference, and the root cause (that finding would be
research gold, not failure).
