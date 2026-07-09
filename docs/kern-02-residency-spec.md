# NumType — Kern 02: Zero-Copy Residency (Spec)

Date: 2026-07-09 · Status: in progress

## Why (intent)

Kern 01 proved the from-scratch kernels bit-identical at ~1.3–2.5× over naive TS *including* per-op
copy-in/copy-out; the scaling bench showed copy bandwidth is the growing cost share at large sizes (add 1024²
drops from ~2.4× to ~1.8–2.3×). Kern 02 removes that overhead: **array data lives resident in WASM linear
memory**, ops run pointer-to-pointer, and copies happen only at explicit boundaries (`fromArray` in,
`toArray`/`toNestedArray` out). The naive TS runtime and the v1 copy-based backend both stay untouched — the
first remains the correctness reference, the second the performance baseline v2 must beat.

## Scope

### In scope

1. **Residency layer** `spike/src/wasm/resident.ts`:
   - `class WNDArray<S extends Shape>` — the resident twin of `NDArray<S>`, reusing the **same type-level
     machinery** (`Guard`/`OkShape` — export these two type aliases from `spike/src/ndarray.ts`, a type-only
     change; do not otherwise touch the type layer) and the same `const` type parameters. Surface: static
     `zeros`/`ones`/`fromArray`, methods `add`/`matmul`/`sum`/`transpose`, plus `toArray(): Float64Array`
     (copy-out), `toNestedArray()`, `dispose(): void`, `readonly shape`, and a `disposed` getter.
   - Data lives as `(ptr, len)` into the core's memory. **Never store a typed-array view** — derive views
     fresh from `core.memory.buffer` at every read/write (memory.grow detaches; KB/kern-01 hard rule).
   - Shape marshalling per call stays copy-based (tiny u32 arrays — not worth residency); operand DATA is
     never copied for an op.
   - Ops allocate a fresh output buffer in wasm, call the kernel pointer-to-pointer, and wrap the result in
     a new `WNDArray` — inputs remain valid and owned by the caller. Outputs never alias inputs (kernels
     assume non-overlapping out; document this invariant).
2. **Lifecycle (the hard part — get this exactly right):**
   - `dispose()`: frees the wasm allocation, marks the handle disposed, **unregisters** from the
     FinalizationRegistry (unregister token) so the backstop can never double-free. Second `dispose()` is a
     safe no-op.
   - `FinalizationRegistry` as GC backstop: if a `WNDArray` is collected without `dispose()`, its allocation
     is freed. Held value must contain ptr/bytes only — NEVER a reference to the `WNDArray` itself (that
     would prevent collection).
   - **Use-after-dispose throws** a clear `Error` naming the operation and "disposed" — on ops, `toArray`,
     `toNestedArray`. Never read/write freed wasm memory.
   - **Error paths leak-free:** if a kernel returns non-zero status, the already-allocated output buffer is
     freed before throwing; input handles stay valid and usable.
3. **Rust crate: expected unchanged.** The kernels already operate on raw pointers. If a change turns out to
   be necessary, document exactly why in the findings doc (honesty rule) — do not silently extend the ABI.
4. **Tests** (`spike/tests-runtime/resident.test.ts`, `resident-lifecycle.test.ts`; node:test, zero deps,
   reuse `prng.ts`/`assert-helpers.ts`):
   - **Differential:** ≥100 seeded cases per op (add/matmul/sum/transpose): resident path
     (`fromArray → op → toArray`) vs naive TS reference — **bit-identical** (kernels are unchanged; this
     verifies the new plumbing: ptr lifecycle, marshalling, view discipline).
   - **Chained residency:** multi-op chains (e.g. add → matmul → transpose → sum) staying resident
     throughout, compared bit-identically against the same chain on the naive reference; intermediate
     results disposed along the way.
   - **Lifecycle:** use-after-dispose throws (op, toArray); double-dispose is a no-op; a failing op (shape
     mismatch → thrown Error) leaves inputs usable and leaks nothing.
   - **Leak plateau (deterministic):** after a warmup phase, run ≥1000 op+dispose cycles and assert
     `core.memory.buffer.byteLength` reaches a plateau (identical at checkpoints, e.g. cycle 100 vs 1000).
     If the allocator's reuse behavior makes an exact plateau unachievable, do NOT delete the test — assert
     the strongest bound that holds and document the observed behavior with evidence.
   - **GC backstop (best-effort, honestly labeled):** if `globalThis.gc` is available (run that one test
     file via `node --expose-gc --test`), drop references, force GC, and assert the registry freed the
     allocation (observable via the memory-plateau or an instrumented free counter). If not available,
     skip with an explicit note — never fake it. The deterministic part (unregister-on-dispose logic) must
     be tested regardless.
5. **Bench:**
   - Extend `spike/bench-core/scaling.ts` with a third series: **WASM resident** (operands pre-resident,
     per-op timing excludes boundary copies) — same methodology (seeded, bit-identity gate, adaptive
     batch-timed reps).
   - New `spike/bench-core/chain.ts`: a chain of k=8 ops (mixed add/matmul) at n=128/256/512, comparing
     naive TS vs v1 copy-based vs resident end-to-end (boundary copies included once at the ends for the
     resident series — that is the realistic usage pattern). Wire `pnpm bench:chain`.
6. **Demo:** small resident section in `spike/demo.ts` (same showcase ops, resident, asserted equal inline,
   explicit dispose at the end).
7. **Findings** `docs/kern-02-ergebnisse.md` (English): lifecycle design decisions (registry semantics,
   dispose contract), measured bench numbers (scaling + chain, vs v1 and naive), gotchas with evidence,
   deviations with reasons.

### Out of scope (unchanged from kern-01 FOLLOWUPS)

Strided kernels; SIMD/threads; transcendentals; dtypes beyond f64; converting the naive `NDArray` itself to
residency; making `WNDArray` the default public API (that's a library-phase decision).

## Acceptance criteria

- `pnpm check` green (type layer untouched except the type-only `Guard`/`OkShape` export).
- `cargo test` green and the crate diff is empty (or any change documented per honesty rule).
- `pnpm test:core` (v1 suite) still 791/791 — v2 must not regress v1.
- New resident tests green: differential ≥100/op bit-identical, chains bit-identical, all lifecycle cases,
  leak plateau asserted.
- `pnpm demo` runs all three paths (naive, v1 wasm, resident) with equal results.
- `pnpm bench:scaling` shows the resident series; `pnpm bench:chain` runs; numbers recorded in the findings
  doc with an explicit v2-vs-v1 delta statement (did residency deliver what the copy-overhead measurements
  predicted?).
- Zero new dependencies (both sides).

## Honesty rule

Same as Spike 01 / Kern 01. Specifically here: if the leak plateau or the GC backstop cannot be asserted as
strongly as specced, document the strongest true statement with evidence instead of weakening silently; if
residency does NOT deliver the predicted win at some sizes, that number is a finding, not a failure.
