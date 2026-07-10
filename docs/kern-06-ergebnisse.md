# NumType — Kern 06: Threads (Results)

Spec: `docs/kern-06-threads-spec.md`. Everything below is grounded in commands actually run in the
implementing session (2026-07-10, Node v24.16.0, 8-core Apple-Silicon host); the honesty rule
applies throughout.

## Summary against acceptance criteria

1. **Threads artifact builds on the pinned nightly; baseline untouched** — MET. `pnpm
   build:wasm:threads` builds `numtype_core_threads.wasm` on `nightly-2026-07-09` (rustc
   `14cae6813`); `pnpm build:wasm` reproduces a byte-identical `numtype_core.wasm` (SHA256
   `a6622a59bc331517294f070507dfd75f8a557cee64ece431e2c847abf538ab2a` before and after, from clean
   rebuilds). Stable gates never require nightly.
2. **Existing gates unchanged** — MET: check clean, `test:core` 817, `test:resident` 2318+2 skips,
   demo bit-identical.
3. **New cargo tests incl. mechanical zero-allocation proof** — MET: cargo 110 (was 92); the
   counting-`#[global_allocator]` integration test measures delta = 0 allocations across a
   `matmul_blocked_partial` core call and delta > 0 across `matmul_blocked` on identical shapes
   (non-vacuous).
4. **`test:threaded` green, bit-identity, clean lifecycle** — MET: 58/58 (after the verification
   round, see addendum; 55 pre-fix), multiple consecutive runs, process exits by itself.
5. **Honest bench** — MET, see table below (including where threads LOSE).
6. **Guard updated** — MET: rule (a) is now "exactly one of test:core / test:resident /
   test:threaded"; all test files registered exactly once; negative demo re-run.
7. **Frozen paths untouched** — MET, with a real gotcha (see below): zero removed/modified lines in
   `abi.rs`/`matmul_blocked.rs`/`shape.rs` (additions strictly appended), v1 files and other
   kernels zero-diff, artifact hash identical.
8. **Process** — this doc + addendum, FOLLOWUPS updated, KB capture, commit (same commit as the
   code, per project convention).

## What was built

- **Feasibility spike first** (session worktree; PoC scripts preserved in the session scratchpad):
  proved stable Rust cannot link `--shared-memory` (precompiled std lacks atomics — exact
  `rust-lld` error captured), the pinned-nightly + `-Z build-std=std,panic_abort` flag set works,
  multi-instantiation over one shared memory is safe (`__wasm_init_memory` CAS guard, raced 3×),
  and — the load-bearing empirical result — **cross-instance stack corruption is real** without
  per-instance stacks (demonstrated with a synthetic stack-heavy module) and fully resolved by
  allocating a stack region per worker and setting the exported `__stack_pointer` before the first
  call. `Atomics.wait` blocks fine on the Node main thread (sync API possible).
- **Rust (additive, cfg-gated `any(not(wasm32), target_feature = "atomics")` so the stable
  artifact is bit-unaffected):** allocation-free shape-helper twins (`*_into` variants over
  `[u32; MAX_RANK]` buffers), no-alloc pack twins (`[f64; MC*KC]`/`[f64; KC*NC]` stack arrays),
  `matmul_2d_blocked_rows` (row-range-bounded core **reusing** `micro_tile`/`accumulate_pair`/
  `accumulate_single` verbatim — the per-element arithmetic sequence is untouched, which is the
  whole bit-identity argument), `matmul_blocked_partial`, and the `nt_matmul_blocked_partial`
  entry point (prevalidated like the six hardened entry points; invalid row range → status 3).
- **Build:** `scripts/build-wasm-threads.sh` — toolchain pre-check with actionable install
  message, exact RUSTFLAGS (incl. `+simd128` since env RUSTFLAGS replaces the config file's),
  separate `--target-dir crates/core/target-threads` (also a new `.gitignore` entry — the existing
  pattern didn't cover it).
- **TS:** `threaded-protocol.ts` (shared Atomics control-block layout, sequence-number handshake —
  single source of truth for main and worker), `threaded-worker.ts` (persistent worker bootstrap:
  instantiate over the shared memory, set `__stack_pointer` to its 16-byte-aligned region, job
  loop), `threaded.ts` (`initThreadedCore(workers?, matmulTimeoutMs?)`, `ThreadedPool`,
  `threadedMatmul(pool, a, b)`, poisoned-pool error semantics — see addendum). The unchanged
  `WNDArray` class from `resident.ts` runs over the threads core (only additive exports plus one
  TS-erased visibility change there; resident suite count unchanged).
- **Tests/bench:** `threaded.test.ts` (differential bit-identity vs both `nt_matmul_blocked` and
  the naive `runtime.ts` across worker counts {1,2,3,4} on identical inputs, views/batch/broadcast/
  size-0/odd dims, aligned+misaligned splits; lifecycle incl. worker-death paths; process
  self-exits), `bench-core/threaded.ts` (correctness-gated timings).

## Bench (`pnpm bench:threaded`, 7 reps after 2 warmups, medians; speedup vs single-threaded `nt_matmul_blocked`)

| n | workers=1 | workers=2 | workers=4 | workers=8 |
|---|---|---|---|---|
| 256 | 1.23× | 1.86× | 3.00× | 3.46× |
| 512 | **0.75×** | 1.31× | 2.15× | 2.86× |
| 1024 | 1.01× | 1.98× | 3.37× | 4.42× |

Honest reading: real scaling at large n (2.86–4.42× with 8 workers at n=512/1024), but the
workers=1 configuration — pure protocol overhead by construction — genuinely LOSES at n=512
(0.75×), and n=64 is a wash at best (workers=1: 0.94×; nominal wins at 2/4/8 sit inside heavily
overlapping ranges). Threading pays off for large matmuls only; the single-threaded path remains
the right default for small ones. No size-based auto-routing was built (out of scope).

Measurement follow-up (same host, idle, post-commit re-run): 2 workers 1.91–1.96×, 4 workers
3.29–3.61×, 8 workers 3.97–4.29× across n=256/512/1024 — and workers=1 measured 1.01–1.02× at
every large n, so the 0.75× reading above did not reproduce and was most likely concurrent host
load during the original run, not protocol cost. The n=64 workers=1 loss (0.88×) persists: at
small sizes the dispatch round trip remains a real, measured cost.

## Gotchas (with evidence — each cost real debugging time or was caught by a layer of the process)

1. **Line shifts alone break byte-identity of untouched functions.** Inserting new code above
   existing code changed the compiled bytes of *unmodified* functions: Rust embeds `file:line`
   panic locations (`#[track_caller]`, array indexing) for downstream code whose line numbers
   moved. Cost a real hash-mismatch investigation; fixed by appending all new declarations strictly
   after all pre-existing content (verified: pristine source prefix byte-identical, artifact hash
   restored). Consequence for every future phase touching shared frozen-adjacent files.
2. **Allocation-freedom is a call-graph property, not a signature property.** The spike found
   `nt_sum_all_strided` allocates internally (`unravel` returns a `Vec`) despite its innocent
   signature; only `nt_sum_all` was worker-safe as-is. This finding shaped the spec's audit rule
   and the mechanical zero-alloc proof.
3. **Per-instance shadow stacks are a convention, not a spec** (tool-conventions has an open TODO).
   The spike's negative control made the risk concrete: without per-instance stacks, concurrent
   main+worker computation produced wrong results on BOTH sides (`24095456949966` vs expected
   `23232614400000` in the synthetic module). Every worker gets a dedicated `nt_alloc`'d region;
   `__stack_pointer` is set before the first call.
4. **Shared-memory grow fails silently where non-shared fails loudly.** A cached `memory.buffer`
   reference keeps its stale length forever (no detach). The existing never-cache-views rule is
   therefore load-bearing without a crash backstop on the threads path; all Atomics cells and data
   views are rebuilt from `memory.buffer` at point of use (grep-verified in review).
5. **Worker `exit`/`error` events cannot interrupt a blocking `Atomics.wait` loop** — Node only
   delivers them after the synchronous call chain returns. Crash detection is therefore
   deadline-based (configurable timeout), proven with a throwaway timing harness before the design
   was fixed. Platform fact, documented in `threaded.ts`.

## Deviations from spec (with reasons)

- `threadedMatmul(pool, a, b)` instead of `threadedMatmul(a, b)`: a WNDArray's `core` reference
  cannot identify "its" pool; matches the existing `WNDArray.zeros(core, shape)` precedent.
- `initThreadedCore(workers?, matmulTimeoutMs?)`: second parameter added so crash tests don't need
  a real 30 s deadline.
- Crash detection deadline-based, not event-based (gotcha 5 — platform constraint, not a choice).
- New Rust code cfg-gated (not spec-mandated wording, but required to satisfy criterion 1).
- Zero-alloc proof lives in a dedicated integration-test binary: cargo's parallel test threads
  would contaminate a shared allocation counter (identified before it could flake, not after).
- Pinned `nightly-2026-07-09` resolves to rustc commit `14cae6813` (2026-07-08), not the floating
  `nightly` alias's `af3d95584` the spike used — normal rustup dated-channel behavior; both build.

## Open issues

- Wait-loop-timeout error path shares the poison mechanism with the (tested) dispatch-time path
  but has no dedicated deterministic test (constructing one without sleep-flakiness is hard; the
  shared-code-path argument is documented). See addendum for the verifier's assessment.
- `MC=32` duplicated between Rust and TS for split alignment (performance-only: drift would
  produce suboptimal splits, never wrong bits).
- `MaxListenersExceededWarning` in `test:threaded` (Node-internal accounting above 10 workers per
  process) — cosmetic, free-counter plateaus confirm real cleanup; deliberately not suppressed.
- Browser execution untested (out of scope; COOP/COEP + async dispatch documented in the spec).
- `no_std`/stable-toolchain path and threading further ops: FOLLOWUPS.

## Post-verification addendum (2026-07-10)

Independent fresh-context verification (brainroute:verify) ran in three rounds against the spec:

1. **Round 1** re-ran every gate itself (all confirmed, including the cfg-gate check via
   `WebAssembly.Module.exports()`: the stable artifact does NOT export
   `nt_matmul_blocked_partial`, the threads artifact does) and found **one confirmed major bug**
   the implementation and its 55 tests had missed: a **use-after-free in the error paths** of
   `dispatchAndRun` — on a mid-dispatch/mid-wait throw, shared scratch/output buffers were freed
   while an earlier-dispatched, still-alive worker could still write into them. The verifier
   proved it empirically with an external instrumentation harness (freed pointer reissued by the
   next same-size `nt_alloc`; the still-running worker's completion signal landed measurably after
   the free). The existing crash tests structurally couldn't see it (dead worker always at index 0
   of a 2-worker pool). Also flagged: the spec's "16-byte aligned" stack claim was not enforced
   (`nt_alloc` guarantees 8).
2. **Fix**: poisoned-pool semantics. A non-zero kernel status (all workers healthy) frees only
   after the wait loop confirms every dispatched worker reached DONE, then throws (sync). A dead
   or timed-out worker instead poisons the pool permanently (fast-fail on further use), throws
   immediately, and **defers** the buffer frees to an async cleanup that runs only after
   `Promise.all(worker.terminate())` settles (Node's contract: resolved only once the thread has
   actually stopped). The deferred free is observable via `getPoisonCleanupFreeCount()` and pinned
   by a deterministic test. `stackTop` is now `alignDown16`'d, unit- and end-to-end-tested.
3. **Round 2** confirmed both fixes: it re-ran its original external repro 3× against the fixed
   code (throw immediate; counter unmoved; an immediate same-size `nt_alloc` returns a
   **different** pointer than the still-in-use output buffer; all 5 in-flight buffers freed only
   inside the deferred cleanup) and separately exercised the wait-timeout poison branch with a
   genuinely concurrent later-index worker. Verdict: **meets the spec**; no new bugs; one
   low-severity test-coverage gap (no dedicated in-suite regression test for the timeout branch)
   plus a nit (no disposed-pool guard in `threadedMatmul`).
4. **Final round** closed both: a dedicated wait-timeout regression test (2 workers, both on
   ~2.2 s jobs vs a 150 ms deadline — 14.8× margin, calibrated empirically; a slower host only
   widens the margin, so the test cannot flake toward false-negative) and
   `assertNotDisposed()` in `threadedMatmul`. Final state: **`test:threaded` 60/60, three
   consecutive runs, zero flakes**; all other gates unchanged (check clean, cargo 110,
   test:core 817, test:resident 2318+2, demo bit-identical, artifact hash `a6622a59…` identical).

Honest residue after verification: `poison()`'s idempotency guard is structurally unreachable in
the current control flow and untested directly (documented in code); the
`MaxListenersExceededWarning` remains (cosmetic); browser execution remains untested (out of
scope). The verification found a real concurrency bug that self-review and 55 passing tests did
not — another data point for the fresh-context-verify discipline.

## Follow-up addendum: size-based auto-routing (2026-07-10)

The FOLLOWUPS item deferred out of this phase's scope — `threadedMatmul` routes small problems to
the single-threaded `nt_matmul_blocked` call on the main thread instead of through the pool — is
now implemented, with the threshold **measured, not guessed**, per the item's own requirement.

**What was built.** `threadedMatmul(pool, a, b, opts?)` computes the call's work volume
(batch·m·k·n, from the same `planMatmul` result it already needed) and routes it to
`a.matmul(b)` — the unchanged `WNDArray` method over the pool's own core — when the volume is
below `THREADED_MATMUL_MIN_POOL_WORK` (exported from `threaded.ts`); at/above it dispatches
through the pool as before. Both routes are bit-identical by construction (the pool path IS
row-partitioned `nt_matmul_blocked`; parallel bit-identity law above). The lifecycle contract is
deliberately size-independent: a disposed/poisoned pool refuses every call, including ones the
router would have run on main. `opts.minPoolWork` overrides per call (`0` = force pool,
`Infinity` = force main); the differential/crash tests pin the pool route with `0` (they would
otherwise silently stop exercising worker dispatch), and new tests observe the route actually
taken via the workers' `postedSeq` counters.

**Measurement** (`pnpm bench:crossover`, new: `spike/bench-core/threaded-crossover.ts`). Timed
unit = `threadedMatmul` + result dispose, operands created once per (case, pool) — i.e. exactly
the decision the router faces, both routes on the SAME threads core. Grid: square n=8…256,
volume-matched non-square (wide-n / deep-k / tall-m at ~128³ and ~256³ volumes), batched cases;
worker counts {2, 4, 8}; adaptively batch-timed samples with ~30 ms tier-up warmup per
(case, pool, route) — a first run with only 2 warmup calls showed V8's per-module wasm tiering
as a systematic main-route spread across pools. Bit-identity gate (both routes vs `runtime.ts`)
before any timing. Host near-idle (load ~2.5 from the interactive desktop session itself; two
runs, findings agree on every case both runs contain).

| volume (Mops) | representative case | pool/main (w=2 / w=4 / w=8) | verdict |
|---|---|---|---|
| 0.0005 (n=8) | 8×8 @ 8×8 | 4.46 / 14.72 / 21.03 | main wins BIG |
| 0.004–0.03 (n=16…32) | square, tall-tiny [256,8]@[8,8] | 1.84–13.37 | main wins |
| 0.11 (n=48) | 48×48 @ 48×48 | 0.99 / 1.06 / 1.16 | wash |
| 0.26 (n=64) | 64×64 @ 64×64 | 0.67 / 0.70 / 0.76 (run 1: 0.85/0.55/0.91) | pool wins, both runs |
| 0.88–16.8 (n=96…256 + non-square + batch) | all | 0.22–1.02 (one outlier: 1.30, see below) | pool wins or ties |

Pool dispatch overhead itself measures ~13–40µs per call (grows with worker count — one
Atomics post/wake round trip per worker, including empty-range workers). The worst above-threshold
cases are single-MC-block shapes (rows < 32 → one active worker), which essentially tie
(0.97–1.02) — with one disclosed outlier: run 2's deep-k/8-rows cell at workers=8 read 1.30,
while run 1 read 0.99 for the same cell and the fresh-context verifier's independent re-run read
1.00 — a one-off host-noise spike, not reproduced, but it sits outside the tie band and is
reported rather than hidden. The volume criterion never loses badly above the cut, while below it
the pool loses up to ~21×.

**Chosen threshold: `262_144` (= 64³)** — the top of the 0.11–0.26 Mops indifference band, i.e.
the smallest measured volume at which the pool reliably won in both runs keeps its win
(`>=` dispatches through the pool). The asymmetric risk motivates cutting high: tiny calls
through the pool lose up to ~21×, a band-edge call on main loses ≤ a few percent.

**Honest correction of this doc's own Series B reading.** Series B (and the "n=64 loss persists"
measurement follow-up above) baselined end-to-end calls — `fromArray` marshalling, `toArray`,
disposes included — against the STABLE core. On the router's actual comparison (call-only, same
threads core) the pool already wins at n=64; the small-n "threads lose" readings above are a
property of that other measurement unit (stable-core baseline + marshalling dilution), not of
dispatch overhead at n=64. Dispatch overhead does lose — decisively — but at n≤48, sizes Series B
never measured. `bench:threaded`'s Series B now pins its worker rows to the pool route explicitly
(`{minPoolWork: 0}`) and carries an auto-routed row for the record.

**Caveats.** The constant is calibrated on the reference machine only (per-host variation
untested — acceptable for a research-grade constant; the override parameter exists). n=48 was
measured in one run only (its routing either way costs ≤16 % at worst, inside the wash band).
The main route pays a second `planMatmul` (once in the router, once inside `matmul()`) — O(rank)
array work, nanoseconds against a ≥µs matmul, accepted for zero duplication of the frozen-adjacent
`matmul()` method.

Gates after the change: `pnpm check` clean; `test:threaded` 65/65 (60 + 5 new auto-routing tests)
three consecutive runs; `test:core` 817 unchanged (test-list guard: the new tests live in the
already-listed `threaded.test.ts`); `bench:threaded` re-run with the pool route pinned — Series A
scaling reproduced (1.92–1.98× / 2.66–3.52× / 3.54–4.35× at 2/4/8 workers across n=256/512/1024;
host carried the interactive desktop session, ~1 core), Series B this run even showed workers≥2
WINNING end-to-end at n=64 (1.19–1.24×; only workers=1 — pure protocol overhead by construction —
lost at 0.84×), further corroborating that the older n=64 readings sat inside the noise of a
different comparison; auto row 1.19× (routes to the pool, as designed at exactly-threshold volume).
