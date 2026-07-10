# NumType — Kern 06: Threads (Spec)

Binding spec for the threads phase. Deviations require an explicit note in the results doc
(`docs/kern-06-ergebnisse.md`), same contract as Kern 01–05.

## Why (intent)

Kern 04 made the matmul kernel fast on one thread (blocked + packed + SIMD128, 2.1–3.25× over the
Kern-03 scalar kernel). The remaining large, orthogonal lever for the compute-bound matmul is data
parallelism across cores. The research question this phase answers: **can a hand-rolled (no
wasm-bindgen, no emscripten, no external libraries) WASM threading substrate deliver parallel
speedup while preserving NumType's bit-identity contract?** The feasibility spike (2026-07-10,
session worktree; PoC copies preserved in the session scratchpad under `spike-threads-poc/`) proved
the substrate end-to-end; this phase productizes it for `matmul` only.

## The parallel bit-identity law (extends the Kern-04 law, non-negotiable)

**Parallelize only ACROSS output elements: every output element's entire accumulation chain runs on
exactly one thread.** Work is split by *output rows* (flat row index over `batch × m`); no split may
divide any single element's ascending-k chain, and no partial-sum recombination across threads is
ever allowed (that would reassociate floating-point addition and break bit-identity). Under this
law, bit-identity holds **by construction** for *any* partition of the row space and *any* worker
count — the WASM threads memory model defines races strictly over overlapping accesses
(threads proposal, "Relaxed Memory Model"), and disjoint output regions never overlap. Worker
count and split boundaries are performance knobs only; tests must prove the bits do not vary with
either.

## Feasibility grounding (what is proven vs. assumed)

Empirically proven by the spike (Node v24.16.0, nightly rustc 1.99.0-nightly af3d95584 2026-07-09):

- Stable Rust **cannot** link this (`rust-lld: --shared-memory is disallowed by std-*.o because it
  was not compiled with 'atomics' or 'bulk-memory' features`); nightly + `-Z build-std=std,panic_abort`
  works with the exact flag set in "Toolchain & build" below. Upstream this path is tracked, still
  unstabilized, and documented as such (rust-lang/rust#77839).
- Multi-instantiation of one module over one shared memory is safe: `rust-lld` synthesizes
  `__wasm_init_memory` with an atomic CAS guard (winner initializes data segments, losers wait) —
  documented in WebAssembly/tool-conventions `Linking.md` and confirmed by 3× racing
  double-instantiations in the spike.
- **Per-instance shadow stacks are a design assumption, not a specified protocol** (tool-conventions
  `BasicCABI.md` has an open TODO here): every instantiation starts with the same baked-in
  `__stack_pointer` init value. The spike demonstrated real cross-instance stack corruption without
  per-instance stacks (negative control) and correct results with them: main allocates a stack
  region per worker via `nt_alloc` before spawning; the worker sets the exported mutable global
  `__stack_pointer.value = stackTop` (16-byte aligned downward) before its first call.
- `Atomics.wait` works (blocking) on the Node **main** thread — a synchronous main-thread API is
  possible in Node. Browsers forbid main-thread blocking; that is a deployment note, not a target.
- Shared `WebAssembly.Memory.grow` never detaches, but a cached `.buffer` reference silently keeps
  its old length. The existing "never cache views" rule is therefore **load-bearing without a loud
  failure backstop** on the threads path: every view must be rebuilt from `memory.buffer` at point
  of use, including all `Int32Array` cells used with `Atomics`.
- `WebAssembly.Module` and shared `WebAssembly.Memory` survive structured clone to
  `worker_threads` (no per-worker recompilation needed).
- **Allocation-freedom must be proven by source audit of the full call graph, never by ABI
  signature** — the spike found `nt_sum_all_strided` allocates internally (`unravel` returns a
  `Vec`) despite its innocent signature. Only `nt_sum_all` is allocation-free today.

## Scope

### In scope

1. **Toolchain & build.** New script `build:wasm:threads` producing
   `spike/src/wasm/numtype_core_threads.wasm`:
   - Pinned toolchain `nightly-2026-07-09` (the empirically proven one) with `rust-src` component
     and the `wasm32-unknown-unknown` target. The script fails early with the exact
     `rustup toolchain install nightly-2026-07-09 --component rust-src --target wasm32-unknown-unknown`
     instruction if the toolchain is missing.
   - Exact flags (RUSTFLAGS replaces the config-file flags, so `+simd128` must be carried along —
     the `compile_error!` guard stays satisfied):
     `-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals`
     `-C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=1073741824`
     `-C link-arg=--export=__stack_pointer`, built with
     `rustup run nightly-2026-07-09 cargo build … --release -Z build-std=std,panic_abort` from the
     repo root, with a **separate target dir** (`--target-dir crates/core/target-threads`) so the
     stable baseline build and its incremental cache are never clobbered.
   - The existing `build:wasm` script and `numtype_core.wasm` artifact remain byte-for-byte
     untouched (frozen baseline); `test:core`/`test:resident`/`demo` keep running on stable exactly
     as before.
   - **Constraint decision (documented, deliberate):** `-Z build-std` compiles the standard library
     from source, which pulls std's *own* vendored dependencies (dlmalloc, compiler_builtins, …).
     These are dependencies of std itself — present in every Rust build, normally precompiled — not
     product dependencies of numtype-core (its `Cargo.lock` stays unchanged). The "no external
     libraries" constraint is read as: no product/runtime dependencies; std and its internals were
     always in the trusted base. A stable-toolchain `no_std` rewrite is a separate research option
     (goes to FOLLOWUPS, not this phase).
2. **ABI addition (additive only): `nt_matmul_blocked_partial`.** Same 14-argument operand
   convention as `nt_matmul_blocked`, plus `row_start: u32, row_end: u32` — a half-open range over
   the **flat output row space** `[0, batch_size · m)` (batch index = `row / m`, row within slice =
   `row % m`). Contract:
   - Runs the same prevalidation as the six hardened entry points (rank/region), then the same
     shape validation as `matmul_blocked` (each worker validates independently), then **zero-fills
     exactly its assigned output rows** and computes them with the *identical* blocked/packed/SIMD
     micro-kernels (`pack_a_tile`-equivalent, `accumulate_pair`, `accumulate_single` — the
     arithmetic sequence per output element must be untouched).
   - Invalid range (`row_start > row_end` or `row_end > batch_size · m`) → status 3 (reuse
     `SizeOverflow`; no new status codes). `out_len` is the FULL output length, as in
     `nt_matmul_blocked`; workers write disjoint row slices of the same buffer.
   - Equivalence law: for **any** partition of `[0, batch_size · m)` into ranges, the union of one
     partial call per range (each zero-filling and computing exactly its own rows) must produce an
     output buffer bit-identical to one `nt_matmul_blocked` call. MC-aligned split points are a
     performance recommendation, never a correctness requirement.
   - **The entire call graph of `nt_matmul_blocked_partial` must be allocation-free.** Existing
     shape helpers that allocate (`broadcast_shape`, `compute_strides`, `unravel`,
     `aligned_effective_strides`) get additive no-alloc variants writing into fixed
     `[u32; MAX_RANK]` buffers; pack buffers become fixed-size stack arrays
     (`[f64; MC*KC]` / `[f64; KC*NC]`, 8 KiB each — MC/NC/KC are compile-time constants). Existing
     helpers and kernels stay untouched (frozen callers).
   - **Mechanical zero-allocation proof (binding):** a `#[cfg(test)]` counting `#[global_allocator]`
     wrapper in the crate; a native test asserts the allocation count is **zero** across a
     `matmul_blocked_partial` core call and **non-zero** across a `matmul_blocked` call
     (non-vacuity). Plus the source-audit comment listing the partial kernel's call graph.
3. **Threaded runtime layer (TS, additive):** `spike/src/wasm/threaded.ts` + a worker bootstrap
   module.
   - `initThreadedCore(workers?: number)` → loads `numtype_core_threads.wasm`, creates
     `WebAssembly.Memory({ initial, maximum, shared: true })` matching the module's declared import
     limits (verified loudly at instantiation), instantiates on main, synthesizes a `CoreExports`
     object (`{ ...exports, memory }`) so the **unchanged** `WNDArray` class from `resident.ts`
     works over it (lifecycle, views, slicing, non-matmul ops all stay single-threaded main-thread
     paths), spawns the persistent worker pool.
   - Pool: default size `max(1, min(workers ?? availableParallelism() - 1, 8))`. Per worker, main
     allocates (via `nt_alloc`, before spawning) a 1 MiB stack region and a job-control block; the
     worker instantiates the same module over the same memory, sets `__stack_pointer`, and enters a
     job loop.
   - Job protocol (binding requirements; cell layout is implementer's choice): persistent workers;
     job dispatch and completion signaled via `Atomics` store/notify + wait on `Int32Array` cells
     in the shared wasm memory (views rebuilt at point of use, never cached); the main-thread API
     is **synchronous** via blocking `Atomics.wait` (Node-only, by design); the kernel status of
     each partial call is transported through the control block and checked on main; a worker
     dying mid-job (`error`/`exit` events, wait timeouts + liveness check) surfaces as a thrown
     `Error` on main within finite time — never a hang. `dispose()` signals workers to quit,
     awaits their exit, and frees stacks + control blocks (leak-testable via the existing free
     counter).
   - `threadedMatmul(a, b)` on WNDArrays bound to the threaded core: mirrors the resident matmul
     semantics exactly (1-D promotion, batch broadcast, final squeeze — reuse resident's logic;
     a small additive export from `resident.ts` is allowed iff the resident suite stays green
     unchanged), allocates the output via `nt_alloc` on main, splits the flat row space into
     MC-aligned contiguous chunks round-robined over the pool, dispatches
     `nt_matmul_blocked_partial` jobs, waits, and returns a `WNDArray`. Empty row spaces / size-0
     shapes short-circuit on main without dispatching.
4. **Test-list guard extension:** new `test:threaded` script (runs `build:wasm:threads` + explicit
   file list). The guard's rule (a) becomes "exactly one of test:core / test:resident /
   test:threaded"; rules (b) and (c) unchanged. Guard test updated in the same commit.
5. **Gates (new):**
   - `cargo test`: partial-kernel equivalence vs `matmul_blocked` for a grid of `(m, k, n, splits)`
     straddling MC boundaries (aligned AND deliberately misaligned splits, single-range,
     many-range, empty-range), batch + broadcast-batch + size-0 cases, status paths (row range →
     3, plus the standard 1/2/4), and the zero-allocation proof from item 2.
   - `pnpm test:threaded` (`spike/tests-runtime/threaded.test.ts`): differential bit-identity of
     `threadedMatmul` against BOTH `nt_matmul_blocked` (single-threaded, same artifact) and the
     naive `runtime.ts` reference, across worker counts {1, 2, 3, 4} × {contiguous, transposed
     views, batch, broadcast batch, k=0, size-0, odd/prime dims}; identical bits for different
     worker counts on the same inputs; pool lifecycle (dispose → workers exited, free-counter
     plateau, second pool works); sequential reuse (many matmuls on one pool).
   - `pnpm bench:threaded` (`spike/bench-core/threaded.ts`): n ∈ {256, 512, 1024}, workers ∈
     {1, 2, 4, 8} vs single-threaded `nt_matmul_blocked`, correctness-gated before timing (bits
     equal), reporting ranges over repetitions, dispatch overhead measured at small n (n=64) and
     reported honestly even (especially) if threads lose there.
   - Existing gates must stay untouched: `pnpm check` clean, `test:core` 817, `test:resident`
     2318+2 skips, cargo suite green, `pnpm demo` bit-identical — all still on stable.

### Out of scope

- Browser execution (COOP/COEP: `Cross-Origin-Opener-Policy: same-origin` +
  `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`), HTTPS, feature-detect via
  `self.crossOriginIsolated`; main-thread blocking `Atomics.wait` is forbidden there → a browser
  port needs an async dispatch variant. Documented here; not built or tested this phase).
- Threading any op other than matmul (elementwise/sum are memory-bound; measure first — existing
  FOLLOWUPS item).
- Packing-buffer reuse / BLIS-grade packing (existing FOLLOWUPS item; unchanged by this phase).
- Work stealing / dynamic scheduling / auto-tuned worker counts (static split only).
- `no_std`/stable-toolchain path (new FOLLOWUPS item), `thread_local!` anywhere in the crate
  (forbidden — `__tls_base` is only initialized in the winning instance; a source comment in
  `abi.rs` must record this landmine), demo changes.

## Acceptance criteria

1. `pnpm build:wasm:threads` produces the threads artifact on the pinned nightly; `pnpm build:wasm`
   and its artifact remain byte-for-byte unchanged (`git diff` empty on frozen paths; stable-only
   gates never require nightly).
2. All existing gates green and unchanged in count: check clean, cargo (old tests) green,
   `test:core` 817, `test:resident` 2318+2, demo bit-identical.
3. New cargo tests green, including the mechanical zero-allocation proof (zero for partial core,
   non-zero for `matmul_blocked` — non-vacuous).
4. `pnpm test:threaded` green: bit-identity across worker counts and against both references,
   lifecycle clean (no leaked workers or buffers; test process exits by itself).
5. `pnpm bench:threaded` runs and reports honest numbers (speedup at large n is *expected* but not
   an acceptance gate; correctness gating and honest reporting are).
6. Guard updated; every new test file registered in exactly one list.
7. No modification to: v1 backend/entry points, `read_slice`/`read_slice_mut`, existing kernels
   (`matmul_blocked` internals may be *refactored into shared allocation-free helpers* ONLY if the
   compiled single-threaded artifact stays bit-identical in behavior — the differential suites are
   the proof — and the source diff keeps the frozen v1 files untouched; prefer additive
   duplication over refactoring where in doubt, matching the crate's existing precedent).
8. Fresh-context verification (spec vs. diff) before "done"; results doc with post-verification
   addendum; FOLLOWUPS updated (threads item out; no_std option in); KB capture; commit.

## Honesty rule

The results doc reports what was actually run and measured, including: exact toolchain/flag set;
any deviation from this spec; scaling numbers as ranges with the machine named; cases where threads
LOSE (small n, dispatch overhead) reported as prominently as wins; the per-instance stack-pointer
mechanism labeled as an empirically validated convention (tool-conventions TODO), not a spec
guarantee; risks the tests do not cover (worker crash mid-job recovery paths actually exercised?
browser untested). Claims of bit-identity must cite the differential runs that prove them.
