# NumType research notes — reading guide

These are the project's research notes, written *as the work happened*, not polished after
the fact. Every substantial slice followed the same loop: **binding spec → implementation →
independent fresh-context verification → results doc with a post-verification addendum**.
The `*-spec.md` / `*-ergebnisse.md` pairs are that loop's paper trail ("Ergebnisse" =
results). The results docs follow an honesty rule: every claim is grounded in commands
actually run, failures and open gaps are recorded rather than smoothed over, and each ends
with what the independent verifier found — including the findings that were wrong or
inconvenient.

**Language note:** most notes are in German — they are internal research artifacts, kept
deliberately (translating them would launder the evidence). File names, code, commit
messages, and everything user-facing are English. The competitive analysis carries a full
English appendix. If a specific document matters to you, open an issue and we'll translate
the relevant part.

## Start here

| Document | What it is |
|---|---|
| [roadmap.md](roadmap.md) | The release roadmap with per-item status — the project's table of contents. |
| [projekt-log.md](projekt-log.md) | The full chronological narrative, slice by slice. |
| [wettbewerbsanalyse-und-usp.md](wettbewerbsanalyse-und-usp.md) | Why the project exists: competitive landscape + the defensible USP. **Appendix A is a full English survey** of every NumPy-like JS/TS library (verified against live registry/GitHub data). |
| [../COVENANT.md](../COVENANT.md) | The standing product contract: the invariants every slice is verified against (bit-identity, "never wrong, only incomplete", frozen baseline, browser safety, zero deps). |

## If you only read three documents

1. **[kern-05-ergebnisse.md](kern-05-ergebnisse.md)** — where the digit-string arithmetic
   was born: dimensions as decimal digit strings instead of tuple lengths, making
   `1000 − 100 = 900` computable by the type checker in O(digit count) instead of O(value).
2. **[spike-02-ergebnisse.md](spike-02-ergebnisse.md)** — the editor-latency evidence: a
   headless LSP harness against the native TS 7 language server, 0.04–0.08 ms median
   hovers — the hard "is this usable at all?" gate, passed with three orders of magnitude
   of headroom.
3. **[kern-04-ergebnisse.md](kern-04-ergebnisse.md)** — the "bit-identity law": how a
   blocked + packed + SIMD128 matmul stays bit-identical to a naive JS reference
   (vectorize only across output elements, ascending-k accumulation, no FMA).

## By interest

### The type layer (TypeScript story)

In reading order:

- [spike-01-type-layer-spec.md](spike-01-type-layer-spec.md) / [-ergebnisse](spike-01-ergebnisse.md) — broadcast/matmul/reduce as gradual types; the USP-risk spike.
- [kern-05-slicing-spec.md](kern-05-slicing-spec.md) / [-ergebnisse](kern-05-ergebnisse.md) — O(1) slice views + the digit-string arithmetic breakthrough.
- [spike-03-index-bounds-spec.md](spike-03-index-bounds-spec.md) / [-ergebnisse](spike-03-ergebnisse.md) — compile-time bounds checks for literal indices (negative indices included — comparison suffices, no signed arithmetic needed).
- [spike-04-shape-products-spec.md](spike-04-shape-products-spec.md) / [-ergebnisse](spike-04-ergebnisse.md) — schoolbook multiplication over digit strings (`reshape`/`flatten` shape products), incl. the MAX_SAFE_INTEGER cap that keeps verdicts *never wrong*.
- [spike-06-range-literals-spec.md](spike-06-range-literals-spec.md) / [-ergebnisse](spike-06-ergebnisse.md) — negative range bounds + literal steps via compare/subtract/clamp and a schoolbook long division.
- [spike-05-variance-design-spec.md](spike-05-variance-design-spec.md) / [-ergebnisse](spike-05-ergebnisse.md) — why the enforced-covariant read view must be computation-free (TS2636), and the method-shorthand bivariance loophole.
- [phase-d-vorarbeiten-spec.md](phase-d-vorarbeiten-spec.md) / [v1](phase-d-vorarbeiten-v1-ergebnisse.md), [v2](phase-d-vorarbeiten-v2-ergebnisse.md), [v3](phase-d-vorarbeiten-v3-ergebnisse.md) + [union-axis-mini-spec.md](union-axis-mini-spec.md) / [-ergebnisse](union-axis-mini-ergebnisse.md) — union soundness: how union dims/shapes/axes degrade to no-claim instead of lying, deep-readonly shapes, the browser smoke test.
- [spike-02-editor-latency-spec.md](spike-02-editor-latency-spec.md) / [-ergebnisse](spike-02-ergebnisse.md) — the LSP measurement harness (now a hard CI gate).

### The from-scratch WASM core (Rust story)

- [kern-01-wasm-core-spec.md](kern-01-wasm-core-spec.md) / [-ergebnisse](kern-01-ergebnisse.md) — hand-rolled `extern "C"` ABI (no wasm-bindgen), first bit-identical kernels.
- [kern-02-residency-spec.md](kern-02-residency-spec.md) / [-ergebnisse](kern-02-ergebnisse.md) — zero-copy residency; honest finding: the copies were never the bottleneck.
- [kern-03-strided-spec.md](kern-03-strided-spec.md) / [-ergebnisse](kern-03-ergebnisse.md) — strided views, O(1) transpose, refcounted buffers.
- [kern-04-simd-blocking-spec.md](kern-04-simd-blocking-spec.md) / [-ergebnisse](kern-04-ergebnisse.md) — blocked+packed+SIMD128 matmul under the bit-identity law.
- [kern-06-threads-spec.md](kern-06-threads-spec.md) / [-ergebnisse](kern-06-ergebnisse.md) — hand-rolled threading substrate (shared-memory artifact, worker pool over Atomics), the "parallel bit-identity law", and a use-after-free the verifier caught.
- [kern-07-elementwise-vector-spec.md](kern-07-elementwise-vector-spec.md) / [-ergebnisse](kern-07-ergebnisse.md) — elementwise family + dot/norm/cosine.
- [kern-08-reshape-flatten-spec.md](kern-08-reshape-flatten-spec.md) / [-ergebnisse](kern-08-ergebnisse.md), [kern-09-keepdims-spec.md](kern-09-keepdims-spec.md) / [-ergebnisse](kern-09-keepdims-ergebnisse.md) — runtime reshape/flatten and keepdims.
- [kern-10-special-values-spec.md](kern-10-special-values-spec.md) / [-ergebnisse](kern-10-special-values-ergebnisse.md) — IEEE-754 special values in the differential suite; key finding: the SIMD matmul preserves subnormals, proven catchable by mutation.
- [kern-11-elementwise-fastpath-spec.md](kern-11-elementwise-fastpath-spec.md) / [-ergebnisse](kern-11-elementwise-fastpath-ergebnisse.md) — measurement-driven perf: SIMD elementwise measured NO-GO (memory-bound), the real lever was a per-element allocation — 13–17× on the contiguous path, bit-identical.
- [phase-c-threads-scoping.md](phase-c-threads-scoping.md) — why threads stay a Node-only opt-in for v0 (no stable-toolchain path exists today; browser gated on COOP/COEP).

### Productization & process

- [item-10-backend-api-spec.md](item-10-backend-api-spec.md) / [-ergebnisse](item-10-backend-api-ergebnisse.md) — the explicit, browser-safe `backend()` opt-in.
- [item-11-api-paket-spec.md](item-11-api-paket-spec.md) / [s1](item-11-s1-ergebnisse.md), [s2](item-11-s2-ergebnisse.md), [s3](item-11-s3-ergebnisse.md) — API cut + the TS7-emit/packaging pipeline + checked package gates.
- [item-12-ci-spec.md](item-12-ci-spec.md) / [-ergebnisse](item-12-ergebnisse.md) — the 8-job CI incl. the frozen-artifact-hash gate and the editor-latency hard gate.
- [infra-01-stress-split.md](infra-01-stress-split.md) — why the instantiation-budget corpus is split (realistic vs. digit-stress), and the non-vacuity proofs.
- [verify-runde-template.md](verify-runde-template.md) — the verification protocol: every slice gets a spec-conformance verifier *and* an adversarial verifier (plus a covenant check), with pre-implementation spec review.

## Conventions

- **Spec first, then results.** Specs are binding and versioned; deviations are disclosed
  and owner-confirmed, never silent.
- **Verification is adversarial and independent.** Fresh-context verifiers try to break
  the work; their findings (including refuted ones) are part of the results docs.
- **Numbers are pinned.** Type-checker instantiation counts, test counts, and the WASM
  artifact hash are exact pins checked in CI — the docs record how each pin moved and why.
