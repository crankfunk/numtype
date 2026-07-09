# NumType — Kern 01: From-Scratch Rust/WASM Kernels (Results)

Date: 2026-07-09 · Status: complete, all acceptance criteria met

## Summary against acceptance criteria

| Criterion | Result |
|---|---|
| `cargo test --manifest-path crates/core/Cargo.toml` green | **Pass** — 38/38 (see below) |
| `[dependencies]` empty | **Pass** — `crates/core/Cargo.toml` has a literal empty `[dependencies]` table |
| `pnpm build:wasm` produces the artifact from a clean target dir | **Pass** — verified by `rm -rf crates/core/target` then re-running |
| `pnpm test:core` green, all cases bit-identical, negative paths throw with shapes named | **Pass** — 791/791 (see below) |
| `pnpm check` green (type layer untouched) | **Pass** |
| `pnpm demo` runs both backends, results equal, hand-checkable output unchanged | **Pass** — original prints unchanged, `[wasm] ...` lines added, throws loudly on divergence (verified this does NOT happen) |
| `pnpm bench:core` prints naive-vs-WASM matmul numbers | **Pass** — ~1.9x WASM speedup incl. copies (see below) |
| No new `package.json` deps; no crate deps | **Pass** — `package.json`'s `dependencies`/`devDependencies` unchanged (only `scripts` grew); `Cargo.toml`'s `[dependencies]` stays empty |

## What was built

```
crates/core/                       Rust crate, empty [dependencies], crate-type = [cdylib, rlib]
  Cargo.toml
  rust-toolchain.toml               channel = "1.95.0", targets = [wasm32-unknown-unknown]
  src/
    lib.rs                         module wiring + crate-level docs
    shape.rs                       shared shape/stride primitives, ported from runtime.ts (38 unit tests total, spread across all kernel files)
    kernels/
      mod.rs
      add.rs                       broadcasting elementwise add
      matmul.rs                    2-D core + batch broadcasting (1-D promotion stays TS-side)
      sum.rs                       sum_all + sum_axis
      transpose.rs                 axis-reversal copy
      fill.rs                      constant fill
    abi.rs                         extern "C" ABI wrappers + nt_alloc/nt_free allocator

spike/src/wasm/
  loader.ts                        dual-target (Node fs/promises + WebAssembly.instantiate; browser fetch + instantiateStreaming)
  backend.ts                       op wrappers (wasmAdd/wasmMatmul/wasmSum/wasmTranspose/wasmFill) — the WASM-backed twin of runtime.ts
  numtype_core.wasm                build artifact (gitignored)

spike/src/ambient.d.ts             hand-written ambient decls for process/node:fs/promises/node:test/node:assert (no @types/node — see Deviations)

spike/tests-runtime/
  prng.ts                          splitmix64 PRNG + shape/data generators
  assert-helpers.ts                shape/bit-identity assertions shared by all *.test.ts files
  add.test.ts                      150 cases
  matmul.test.ts                   150 cases (incl. 1-D promotion, batch broadcast)
  sum.test.ts                      150 (sum_all) + 150 (sum_axis, incl. negative axes) cases
  transpose.test.ts                150 cases
  fill.test.ts                     30 cases
  negative-paths.test.ts           11 cases (all 4 status codes + throwing contract)

spike/bench-core/matmul-256.ts     matmul [256,256]x[256,256] naive-TS-vs-WASM wall-clock bench

spike/demo.ts                      extended: every showcase op also runs via WASM, asserted equal inline
package.json                       + build:wasm/test:core/bench:core scripts; demo now chains build:wasm
.gitignore                         + *.wasm, crates/**/target/
```

~1049 lines of Rust (incl. unit tests), ~1115 lines of new/changed TypeScript (incl. differential tests), measured via `wc -l` this session.

## ABI documentation

Hand-rolled `extern "C"` ABI, no wasm-bindgen/wasm-pack. Verified empirically (see Gotchas) that `wasm32-unknown-unknown` auto-exports both `memory` and every `#[no_mangle] pub extern "C" fn` with zero linker flags — no `env` import object is needed at all.

**Calling convention:**
- Every pointer is a `u32` byte offset into the module's own `memory` export (WASM32 = 32-bit address space).
- A **shape** argument is a `(ptr, rank)` pair: `rank` little-endian `u32` values, one per axis, row-major (same as `NDArray.shape`).
- A **data** argument is a `(ptr, len)` pair: `len` `f64` values, row-major contiguous (same layout as a `Float64Array`).
- Every op returns a `u32` **status code**.
- TS computes output shapes and pre-allocates the output buffer (via `nt_alloc`) before calling; the op writes into it and returns status only.
- On non-zero status, the output buffer's contents are unspecified and must not be read.
- A `(ptr, len)`/`(ptr, rank)` pair with `len`/`rank == 0` is valid for **any** pointer value (including `0`) — the ABI never dereferences memory for a zero-length argument (see Gotchas: this is a real Rust safety requirement, not a convenience).

**Exports:**

| Export | Signature | Notes |
|---|---|---|
| `nt_alloc` | `(bytes: u32) -> u32` | Returns `0` on a zero-byte request or allocation failure. `8`-byte alignment for every allocation. |
| `nt_free` | `(ptr: u32, bytes: u32)` | `bytes` MUST equal the value passed to the paired `nt_alloc` (Rust's `dealloc` requires the identical layout). `ptr == 0` or `bytes == 0` is a no-op. |
| `nt_add` | `(aShapePtr, aRank, aDataPtr, aLen, bShapePtr, bRank, bDataPtr, bLen, outDataPtr, outLen) -> status` | Broadcasting elementwise add. |
| `nt_matmul` | same shape as `nt_add` | Both operands must already be rank ≥ 2 (1-D promotion is TS-side); returns the **unsqueezed** `[...batch, m, n]` shape — TS squeezes afterward, exactly mirroring `matmulRuntime`. |
| `nt_sum_all` | `(dataPtr, len, outDataPtr) -> status` | Sums every element; always succeeds (status is always `0`). |
| `nt_sum_axis` | `(shapePtr, rank, dataPtr, len, axis: i32, outDataPtr, outLen) -> status` | `axis` may be negative (counts from the end), matching `sumRuntime`. |
| `nt_transpose` | `(shapePtr, rank, dataPtr, len, outDataPtr, outLen) -> status` | Reverses every axis; a copy, not a view. |
| `nt_fill` | `(outDataPtr, outLen, value: f64) -> status` | Fills with a constant; always succeeds. |

**Status codes** (`shape::KernelError::status()`):

| Code | Meaning | Triggered by |
|---|---|---|
| `0` | ok | — |
| `1` | shape-incompatible | Broadcast mismatch, matmul inner-dim mismatch, out-of-range reduce axis, matmul operand rank < 2 (a promotion-contract violation — bucketed here since none of the other three categories fit) |
| `2` | rank > 32 | Any operand shape with `rank > MAX_RANK` (`MAX_RANK = 32`) |
| `3` | size overflow | A shape's element count or byte size (`elements * 8`) doesn't fit `u32` — **and**, deliberately reusing the same code, the caller-supplied `outLen` not matching the kernel's own independently-computed output length. This dual use is a deviation from the literal wording "size overflow"; see Deviations below for why. |

**Allocator:** `nt_alloc`/`nt_free` sit directly on `std::alloc::{alloc, dealloc}` with an 8-byte alignment convention. No allocator crate — confirmed empirically that `std` on `wasm32-unknown-unknown` already provides a working global allocator backed by `memory.grow` (see Gotchas).

## Determinism / bit-identity approach

Every kernel is a line-by-line port of its `runtime.ts` counterpart: same iteration order (`unravel`/stride-offset accumulation order), same accumulation order in reduction loops (`sum +=`, matmul's `sum += aVal * bVal` in the same `mi`/`ni`/`ki` nesting). v1 uses only `+` and `*` (IEEE-754-defined identically in WASM and JS), so no epsilon tolerance is needed anywhere — every differential test asserts exact `Object.is` equality per element (distinguishes `-0`/`+0` and handles `NaN` correctly, though the PRNG's value generator never produces `NaN`/`Infinity` by construction).

## Testing

### `cargo test` (native, kernel layer)

```
running 38 tests
test abi::tests::alloc_zero_is_sentinel ... ok
test abi::tests::free_zero_is_noop ... ok
test abi::tests::read_slice_mut_zero_length_never_dereferences_null ... ok
test abi::tests::read_slice_zero_length_never_dereferences_null ... ok
test kernels::add::tests::* (7 tests) ... ok
test kernels::matmul::tests::* (6 tests) ... ok
test kernels::sum::tests::* (6 tests) ... ok
test kernels::transpose::tests::* (5 tests) ... ok
test kernels::fill::tests::fill_basic ... ok
test shape::tests::* (9 tests) ... ok

test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Covers: broadcast basic/interior-1/rank-0/size-0/incompatible/rank-too-large for add; 2-D/batch/batch-broadcast/inner-dim-mismatch/rank-below-2/rank-too-large for matmul; axis/negative-axis/out-of-range/rank-too-large/size-0 for sum; 2-D/3-D/rank-0/size-0/rank-too-large for transpose; overflow boundary conditions (`u32::MAX` single dim, `70000×70000`) and rank-33-vs-32-boundary for the shared `checked_element_count`.

### Differential suite (`pnpm test:core`, `node --test`, against the real compiled `.wasm`)

```
ℹ tests 791
ℹ pass 791
ℹ fail 0
```

Breakdown: 150 add cases (ranks 0–4, broadcast-1 placements on either operand at every axis), 150 matmul cases (2-D core, batch broadcasting with broadcast-1 batch dims, ~20% chance each operand is 1-D to exercise TS-side promotion, plus vector·vector), 150 sum_all + 150 sum_axis cases (rank 0–4 / rank 1–4, negative axes included), 150 transpose cases, 30 fill cases, 11 negative-path cases. All seeded (splitmix64, hand-written, fixed per-file seeds) — fully reproducible across runs.

**No bit-identity failures across any case class.** Every generated case matched the naive TS reference exactly, including the deliberately-adversarial "outer product" broadcast case and negative-axis reduction.

**Verification that the harness isn't vacuous:** deliberately changed the add kernel's `a_val + b_val` to `a_val - b_val` and confirmed (a) `cargo test` fails 4 tests with the exact wrong values shown, and (b) `pnpm test:core` fails all 150 add cases with clear `expected X got Y` messages; reverted and re-confirmed both suites pass again. The harness genuinely detects regressions, not just re-deriving the same buggy value on both sides.

### Negative paths (all four status codes + the throwing contract)

- **Status 1** (shape-incompatible): tested directly via raw `nt_add`/`nt_matmul`/`nt_sum_axis` calls with incompatible shapes/mismatched inner dims/out-of-range axis, AND via the `wasmAdd`/`wasmMatmul`/`wasmSum` wrappers, asserting they throw `Error`s naming the offending shapes/axis (e.g. `"...shapes [2,3] and [2,4]..."`, `"...inner dimensions 3 and 4..."`, `"...axis 5 is out of range for shape [2,3]..."`).
- **Status 2** (rank > 32): tested directly against the raw ABI (rank 33 → status 2; rank exactly 32 → status 0, boundary-checked both ways) — this path is structurally unreachable through normal TS usage (TS never constructs shapes with rank > 32), so it can only be meaningfully tested this way.
- **Status 3** (size overflow): tested directly against the raw ABI with a `[70000, 70000]` shape (product ≈4.9e9 > `u32::MAX`) alongside deliberately tiny, correctly-sized real backing buffers — safe because `checked_element_count` on the *declared shape* bails out before any data is touched (see the ABI's `read_slice` doc comment). Also tested the `outLen`-mismatch sub-case (status 3 reused, see Deviations).

## Bench (`pnpm bench:core`)

matmul `[256,256] × [256,256]`, 10 timed iterations + 1 untimed warmup, wall clock via `performance.now()`:

```
Correctness check (bit-identical TS vs WASM): true

naive TS            : min=23.436ms avg=23.544ms max=23.888ms
WASM (incl. copies) : min=12.198ms avg=12.251ms max=12.378ms
speedup (avg TS / avg WASM): 1.92x
```

Reproduced across 3 independent runs this session: 1.93x, 1.93x, 1.92x — stable. The WASM timing **includes** the full copy-in/copy-out overhead (shape+data marshalling into scratch WASM memory, kernel call, result copy-out, four `nt_free` calls) — v1 is copy-in/copy-out by design (spec: zero-copy resident buffers are v2/FOLLOWUPS). Despite that overhead, WASM is ~1.9x faster than the naive triple-loop TS implementation for this size — both are unoptimized/naive algorithms (no blocking, no SIMD), so this reflects raw scalar-loop throughput, not an apples-to-apples "good vs. best" comparison.

## Gotchas (with evidence)

### 1. `wasm32-unknown-unknown` auto-exports `memory` and every `#[no_mangle]` function — no linker flags needed

Verified by building a minimal probe crate and inspecting `instance.exports` from Node directly:

```
exports: [ 'memory', 'nt_alloc', 'nt_free', 'nt_probe_add', '__data_end', '__heap_base' ]
```

No `wasm-bindgen`, no `-C link-arg=--export-memory`, nothing beyond a stock `cargo build --target wasm32-unknown-unknown --release`. This was a real risk going in (older docs/blog posts about hand-rolled WASM ABIs mention needing explicit export flags); empirically, on this toolchain (rustc 1.95.0), it's unnecessary.

### 2. `std`'s allocator works out of the box on `wasm32-unknown-unknown`, including triggering real `memory.grow`

```
before bytes: 1114112
alloc(5MB) -> 1114120
after bytes: 6160384
same buffer object? false
before detached (byteLength===0)? true
```

Confirms both that `std::alloc::{alloc, dealloc}` need no allocator crate on this target, AND the spec's hard memory rule empirically: growing memory **detaches** the old `ArrayBuffer` (`byteLength` becomes `0`) and hands back a genuinely different buffer object. `backend.ts` never caches a typed-array view across any call that might allocate — every `write*`/`read*` helper re-derives its view from `core.memory.buffer` fresh, immediately before use.

### 3. `std::slice::from_raw_parts` requires a non-null pointer even for a zero-length slice — and `nt_alloc(0)`'s `0` sentinel violates that if handed to it naively

This is the most substantive finding this session. An early version of `abi.rs` built every ABI-boundary slice via a direct `unsafe { std::slice::from_raw_parts(ptr as *const T, len as usize) }`. Since `nt_alloc(0)` (and any zero-length shape/data argument — rank-0 scalars, size-0 arrays, both explicit spec edge cases) legitimately produces `ptr == 0`, and Rust's documented safety contract for `from_raw_parts` requires the pointer to be **non-null and aligned even when `len == 0`**, this was latent unsound code.

It surfaced concretely (not just as a theoretical concern) while writing a native `cargo test` for the ABI layer: an early test tried to round-trip a real allocation through the `u32`-typed ABI functions *natively* (not via the actual wasm32 target). That test SIGSEGV'd:

```
* thread #2 ... stop reason = EXC_BAD_ACCESS (code=1, address=0xd3000a00)
    frame #0: 0x000000018ff7f508 libsystem_platform.dylib`_platform_memmove + 424
```

Root cause (confirmed via `lldb`, not guessed): `nt_alloc` returns `u32` — correct for WASM32's genuinely-32-bit address space, but `cargo test` runs the crate's `rlib` **natively** on a 64-bit host (aarch64-apple-darwin here), where `std::alloc::alloc` returns real 64-bit pointers. `ptr as u32` truncates them; casting the truncated value back to a pointer produces a wild address — `0xd3000a00` is recognizably a zero-extended 32-bit value, not a real heap pointer. This is not a bug in the WASM32 behavior (there, `u32` pointers are exact); it's that **the ABI's own `u32`-pointer contract makes it only meaningfully testable by actually running inside wasm32** — which the differential TS suite does, against the real compiled `.wasm`, in Node's real `WebAssembly` runtime (a true 32-bit address space).

Fix applied: `read_slice`/`read_slice_mut` helpers special-case `len == 0` to a real empty-slice literal (`&[]`/`&mut []`), never touching `from_raw_parts` at all in that case, regardless of the pointer value. Verified via two native, host-portable unit tests (`read_slice_zero_length_never_dereferences_null` etc., safe because they never construct a "real" allocation through the truncating `u32` return path) plus the full differential suite, which exercises rank-0/size-0 cases through the real ABI end-to-end (e.g. `add` case generator produces rank-0 operands; `transpose`/`sum_all` differential cases include size-0 shapes).

The problematic native round-trip tests were removed rather than "fixed to pass" — they were testing something structurally untestable that way; the honest fix is documenting why, not forcing green via a wrong methodology. Miri (which would give a formal UB verdict) was not available on this toolchain (`cargo +1.95.0 miri` errors: component not available for this target-toolchain pair) — noted as a real, not-glossed-over gap; the mitigations above (documented safety contract read directly from Rust's own docs, a working fix, and end-to-end real-wasm32 coverage of the exact edge case) are the strongest verification available without it.

### 4. `node --test <bare-directory-path>` does not scan the directory for test files — an explicit glob is required

`node --test spike/tests-runtime` (no trailing slash) and `node --test spike/tests-runtime/` both fail with `Cannot find module '.../spike/tests-runtime'` — Node's CLI resolves a bare path argument as a script/module path, not a test-discovery root, in this version. `node --test "spike/tests-runtime/*.test.ts"` (quoted so the shell doesn't glob-expand it first, letting Node's own test-runner path resolution do it) works correctly and picks up exactly the six `*.test.ts` files (791 tests total, matching the sum of each file's own case count). `package.json`'s `test:core` script uses the quoted-glob form.

### 5. (Minor, methodology-only) `mv`-restoring a file after a deliberate-bug edit preserved a stale mtime, causing `cargo test` to skip recompilation

While verifying the differential harness isn't vacuous (see Testing above), reverting `add.rs` via `mv add.rs.bak add.rs` kept the backup file's original (pre-edit) mtime, which was *older* than the timestamp Cargo's fingerprint cache had already recorded for the buggy version — so `cargo test` silently reused the stale (buggy) compiled artifact and kept reporting failures after the "revert." `touch add.rs` (bumping the mtime forward) fixed it immediately; not a product bug, just a reminder that Cargo's incremental rebuild detection is mtime-based, not content-hash-based, so restoring old file content with an old mtime doesn't reliably invalidate the build cache.

## Deviations from spec (with reasons)

1. **Status 3 reused for "output-length mismatch," not just literal arithmetic overflow.** The spec defines four codes with `3` = "size overflow." Beyond that literal meaning, this implementation also returns `3` when the caller-supplied `outLen` doesn't match the kernel's own independently-computed output length. Reason: without this check, a shape-logic bug that makes TS and Rust disagree on the output size (while each individually believes its own shape math is "compatible") would let the kernel write past the caller-allocated buffer — real memory corruption in WASM linear memory, not just a wrong-value bug. Given only four status codes exist and none of the other three fit "the two sides disagree on size," reusing `3` was the safest, simplest choice, and it doubles as exactly the "differential-testing feature" the spec calls out (a TS/Rust shape-logic divergence surfaces as a hard status rather than silent corruption).
2. **matmul operand rank < 2 bucketed as status 1 (shape-incompatible), not a new category.** Per spec, 1-D promotion happens TS-side, so `nt_matmul` should only ever receive rank ≥ 2 operands. A rank < 2 operand reaching the kernel indicates an ABI-contract violation; since none of the three real categories (shape-incompatible / rank-too-large / size-overflow) describes "wrong rank on the low end" precisely, `ShapeIncompatible` was the closest fit and is what's implemented and tested.
3. **New `spike/bench-core/` directory instead of adding the bench script into the existing `spike/bench/`.** `spike/bench/` is spike-01's own isolated-project instantiation-diagnostics harness (`check:diag:bench`, `-p spike/bench`, whose `tsconfig.json` includes the whole directory via `"include": ["."]`). Dropping a new file in there would have silently changed what that pre-existing, already-recorded measurement compiles, corrupting a spike-01 result without touching spike-01's own files. Verified after the fact: `check:diag:bench` still reports `Files: 70` (unchanged) with my code in place, confirming no scope leak — see note below on the *Instantiations* count.
4. **Hand-written ambient `.d.ts` for a handful of Node APIs (`spike/src/ambient.d.ts`), since `@types/node` is not a permitted dependency.** `process` (feature-detection only, typed as `unknown`), `node:fs/promises` (`readFile`), `node:test` (`test`), `node:assert` (`ok`/`strictEqual`/`deepStrictEqual`/`throws`) are declared minimally — exactly the surface actually called, not a general Node-typings shim. Verified empirically first that `WebAssembly.*`, `fetch`, `URL`, `performance.now`, and `import.meta` are all already ambiently available from TypeScript's default lib set for this project's `target: "ES2022"` (no explicit `lib` option) — only the four Node-specific items above needed declaring.

**Note on `check:diag:bench`'s `Instantiations` number:** re-running this session shows `7425` (stable across 3 consecutive runs), vs. `9,494` recorded in `docs/spike-01-ergebnisse.md`. `git diff --stat -- tsconfig.json spike/bench/` is empty — neither the root `tsconfig.json` nor anything under `spike/bench/` was touched this session, and the file count matches (`70` both times), so this is not something Kern 01 introduced. Flagged here for visibility, not investigated further (out of this task's scope — it concerns spike-01's own recorded numbers, not Kern 01's kernels).

## Open issues

None blocking. Everything in the spec's "in scope" list is implemented, tested, and green. Out-of-scope items (zero-copy resident buffers, strided kernels, SIMD, threads, transcendental ops, dtype system beyond f64) are correctly left for v2/FOLLOWUPS per the spec — not touched here.

---

## Post-verification addendum (2026-07-09)

An independent fresh-context verification pass re-ran every acceptance criterion (all green), read the Rust
kernels line-by-line against `runtime.ts` (order mirroring confirmed by construction, not just by the 791
passing cases), audited memory-rule compliance (no cached views; `try/finally` free-pairing on all paths incl.
error paths; status 2/3 enforced in Rust), and — independently of this doc's own mutation test — built its own
mutant core in session-scratchpad isolation and confirmed the differential methodology catches a real kernel
bug (mutant add: `bit-identical? false`; the repo artifact untouched). Verdict: meets spec; findings all
minor/note:

1. **ABI hardening (minor):** `read_slice`/`read_slice_mut` construct the slice for the caller-declared
   length BEFORE `checked_element_count`/rank validation runs — safe under the documented caller contract and
   at every actual call site, but validate-before-slice would be cheaper to trust. → FOLLOWUPS.
2. **Instantiation-count oddity resolved:** the 9,494-vs-7,425 discrepancy flagged above is a stale
   pre-fix number in spike-01's own doc (its addendum updated the suite figure but not the bench figure) —
   corrected there; not a Kern 01 effect (`spike/bench` compile scope unchanged, 70 files both times).
3. **Special-value coverage (note):** the PRNG produces normal finite values (and occasional `-0`) only —
   bit-identity is proven for that class; NaN/±Infinity/subnormal inputs are untested (low risk: v1 ops are
   `+`/`*` only, IEEE-identical for those classes too). → FOLLOWUPS (inject special values into the generator).
4. FOLLOWUPS.md hygiene gap (v2 items not yet listed) — fixed alongside this addendum.
