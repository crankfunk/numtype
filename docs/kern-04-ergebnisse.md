# NumType — Kern 04: SIMD128 + Blocked Matmul (Results)

Date: 2026-07-10 · Spec: docs/kern-04-simd-blocking-spec.md · Status: complete, independently verified

Process note: this phase was executed by a delegated implementation agent (Sonnet 5, xhigh) against the
binding spec, then passed **two independent review layers**: an orchestrator code review (bit-identity
argument re-checked line-by-line, all gates re-run) and a fresh-context verification agent (which
re-derived the bit-identity argument from scratch, disassembled the built `.wasm`, and deliberately
tripped the build guard). All numbers below come from commands actually run on 2026-07-10.

## Summary against acceptance criteria

- `pnpm check` — clean. `cargo test` — **76/76** (was 63; +13: packing units, blocked≡strided equivalence
  grid of 180 shape combinations crossing tile boundaries, view operands, k=1/n=1, batch + broadcast
  batch, size-0 with exact-0.0 assertion, statuses 1/2/4, hand-checked fixture).
- `pnpm test:core` — **791/791** (frozen v1). `pnpm test:resident` — **1578 + 2 honest GC skips**
  (was 1412+2; +166 new `blocked.test.ts` differential cases; every pre-existing case now exercises the
  blocked kernel through `WNDArray.matmul()` and passed unchanged). `pnpm test:resident:gc` — 2/2.
  `pnpm demo` — all three backends bit-identical.
- `pnpm bench:blocked` — three runs recorded below, including the honest small-size datapoint.
- Frozen code byte-for-byte untouched (`git diff` empty on `matmul.rs`, `backend.ts`, `runtime.ts`,
  `demo.ts`; `nt_matmul_strided` kept intact as the measurable baseline). Zero new dependencies.
- The `.wasm` demonstrably contains SIMD: real `f64x2.mul`/`f64x2.add`/`f64x2.splat` opcodes confirmed by
  disassembly (wabt, dev tooling), zero `relaxed`/`madd`/`fma` opcodes anywhere in the binary.

## What was built

`crates/core/src/kernels/matmul_blocked.rs` (behind the new `nt_matmul_blocked` export, same strided
quadruple convention and status codes as `nt_matmul_strided`): a from-scratch GEMM-style kernel —
M/N/K tiling, per-tile packing into contiguous scratch, and an f64x2 micro-step. `WNDArray.matmul()`
routes to it unconditionally; everything else (v1 backend, Kern-03 kernels, type layer) is untouched.
Zero `unsafe` in the new module.

**How it stays bit-identical (the phase's core design):**
- Vectorization only ACROSS output elements: one f64x2 accumulator covers two adjacent output columns;
  each lane performs exactly the scalar sequence (splat-a, mul, add — two roundings, no FMA). Odd
  columns fall to a scalar tail with the same sequence.
- Per output element, one accumulator chain in strictly ascending k: each k-block reads the running value
  out of `out[]` and writes it back — a memory round-trip of an f64 is bit-exact (WASM has no extended
  precision), so k-tiling merely chunks the flat k-loop without reassociating it. Never a fresh local
  partial sum added post-hoc (that would reassociate; banned by the spec's law).
- Packing is pure data movement via the operand's own strides/offset — contiguous and view operands take
  the identical path (this is what erases the Kern-03 view penalty).
- relaxed-simd is banned (implementation-defined results would break cross-engine determinism).

**Tile sizes MC = NC = KC = 32**, chosen by rough raw-ABI measurement (first guess 64/64/256 measured
48.2 ms / 370 ms at n=512/1024; sweeping showed smaller tiles winning consistently; 32/32/32 landed at
38.8–43.0 ms / 310.9–328.8 ms — a further ~15–20 %). Research-grade, not BLIS-grade; the sweep is
documented in the kernel's doc comment.

## Bench (`pnpm bench:blocked`, three runs, 2026-07-10)

**Series A — headline: blocked+SIMD vs the Kern-03 scalar strided kernel (contiguous operands):**

| n | Kern-03 strided | blocked+SIMD | speedup (3 runs) |
|---|---|---|---|
| 64 | ~181 µs | ~86 µs | 2.09–2.11× |
| 128 | ~1.52 ms | ~0.63 ms | 2.40–2.43× |
| 256 | ~12.3 ms | ~5.3 ms | 2.30–2.47× |
| 512 | ~115 ms | ~39 ms | 2.95–3.25× |
| 1024 | ~996 ms | ~362 ms | 2.75–3.21× |

The advantage grows with n, as the cache-pressure model predicts. Against naive TS the blocked kernel is
~3.9–4.7×. (v1 and Kern-03-strided remain within noise of each other, consistent with Kern 03's Series B.)

**Series B — the Kern-03 prediction on record: CONFIRMED.** Kern 03 measured transposed-view operands
~30 % SLOWER than materializing at n ≥ 256 (scalar kernel, cache-hostile strided k-loop). Through the
packing kernel the view path is now as fast or faster than `contiguous()`-then-matmul at every size and
in every run: 1.28–1.76× (n=128), 1.14× (n=256), 1.06–1.08× (n=512). Packing normalizes the access
pattern tile-wise regardless of source layout — the "materialize before hot matmuls" guidance from
kern-03-ergebnisse.md is **obsolete as of Kern 04** (kept there with a pointer here).

**Series C — small-size honesty:** n=8/16/32 win clearly (1.14–1.97×). **n=4 is at the noise floor**
(~1 µs total): one of three runs showed 0.41×, two showed ~1.0–1.1× — inconclusive noise, not a
systematic regression, but reported rather than rounded up. No threshold dispatch needed on this
evidence.

## Gotchas (with evidence)

- **Silent scalarization is real:** omitting `-C target-feature=+simd128` does NOT fail compilation —
  LLVM legalizes `v128` ops into scalar code silently (verified empirically both ways). The
  `compile_error!` guard in the kernel module is what makes the flag loss loud; the verifier
  independently tripped it.
- **Cargo config discovery is cwd-based, not `--manifest-path`-based** (verified empirically with a
  throwaway layout, and again by the verifier building from an outside cwd): the repo-root
  `.cargo/config.toml` only applies when cargo runs from (under) the repo root. The guard's error message
  documents this; it also revalidates the session rule "run all commands from the repo root".
- **Safe SIMD without `unsafe`:** the `f64x2(a, b)` two-scalar constructor (safe) replaces the `unsafe`
  `v128_load` pointer intrinsic; LLVM fuses the two adjacent packed-slice reads into a single vector load
  anyway (disassembly-confirmed). Zero `unsafe` in the new module.
- **Stale doc comments after tuning:** the tile-size sweep left one test-file header naming the first-guess
  tile sizes; caught in orchestrator review. Constants referenced in prose want a single source of truth.

## Deviations from spec (with reasons)

- B-panels are repacked once per (mb, kb, nb) — redundant across mb blocks. Explicitly within the spec's
  "research-spike-grade, not BLIS-grade" allowance; cost amortized by MC. Recorded as a future perf lever
  in FOLLOWUPS.md (packing-buffer reuse), not a correctness issue.
- Safe `f64x2` constructor instead of `v128_load` (see gotchas) — flagged by the executor rather than
  silently substituted; strictly an improvement (no `unsafe`), adopted.
- No other deviations. Threshold dispatch correctly omitted (out of scope; Series C shows it unneeded).

## Open issues

Tracked in FOLLOWUPS.md: threads/COOP-COEP (the remaining third of the original SIMD/threads/blocking
bullet), packing-buffer reuse, SIMD for elementwise/sum kernels (measure first — they are memory-bound),
plus the carried-over items (v1 OOM path, ABI rank/len pre-validation, special-value injection).

## Post-verification addendum (2026-07-10)

Fresh-context verification (brainroute:verify) returned **"Kern 04 meets its spec"** with the explicit
verdict that the bit-identity argument is **"SOUND, and correctly implemented"** — independently
re-derived (loop-nesting analysis including the specific buggy orderings it ruled out, the memory
round-trip argument, lane pairing, packing-bounds argument vs `validate_strided_bounds`), plus two
artifact-level checks beyond the spec's gate list: disassembly of the built `.wasm` (real f64x2 opcodes,
zero relaxed/fma) and deliberately tripping the `compile_error!` guard from an outside cwd. Findings and
resolution:

1. *Stale error string* (minor): `resident.ts`'s matmul failure message still named `nt_matmul_strided`
   after the entry-point swap — fixed, suites re-run green (1578+2).
2. *FOLLOWUPS bullet* (nit): the combined SIMD/threads/blocking item — split at commit time (this commit);
   threads remain as their own open item.
3. *Per-tile `Vec` allocation / redundant B-repack* (nit): acknowledged future perf lever, added to
   FOLLOWUPS.md.
