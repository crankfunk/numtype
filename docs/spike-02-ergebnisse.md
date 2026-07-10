# NumType — Spike 02: Editor Latency of the Type Layer (Results)

Date: 2026-07-10 · Status: complete, independently verified (see addendum) · Toolchain: TypeScript
7.0.2 (native/Go generation, `serverInfo: "typescript-go"`), Node v24, pnpm

Per `docs/spike-02-editor-latency-spec.md`. Every number below was produced by a command run in this
session (`pnpm bench:editor`, two full runs; raw outputs preserved); commands are shown so numbers
are reproducible, not asserted.

## Headline verdict

**The roadmap-A1 hard gate PASSES — by about three orders of magnitude.** Warm hover on inferred
shape types measures **0.04–0.08 ms median** (gate: ≤ 100 ms) at every one of 19 positions across
all five workloads, in both runs; the edit→diagnostics loop measures **1.4–1.6 ms median** (gate:
≤ 500 ms). The type layer — including Kern 05's digit-string slice arithmetic under multi-digit
borrow chains — is nowhere near being an editor-latency problem at realistic scale. The honest
scope of that claim (small projects, LSP wire time only) is spelled out under "What this does NOT
show".

## Method (implemented per spec)

- Headless, hand-rolled LSP client (JSON-RPC over stdio, Content-Length framing, zero
  dependencies) driving the **native TS 7.0.2 server** (`<exe> --lsp --stdio`, exe resolved via
  `typescript/lib/getExePath.js`) — the same wire protocol the official VS Code extension speaks.
- Server→client requests (`workspace/configuration`, `client/registerCapability`) are answered —
  an unanswered one stalls hover indefinitely (verified during research; encoded in the harness).
- Requests strictly serialized; per-request 30 s timeout that aborts loudly, never hangs.
- **Correctness gate on every timed sample** (never time a wrong result): a hover sample counts
  only if its text contains the expected inferred shape (e.g. `NDArray<[45526]>` — expectations
  hand-computed in the generator and independently re-checked); an M3 sample counts only if the
  toggled diagnostic (code 2741 at the toggle line) actually appears/disappears. All 19×2 hover
  positions, 5 completion positions and both toggle directions passed their gates in both runs —
  no wrong/empty response was ever timed.
- Workloads W1–W5 generated deterministically (`spike/bench-dx/gen-workloads.ts`), importing the
  REAL type layer (`spike/src/ndarray.ts`); per-workload self-contained tsconfigs. Instantiation
  counts per isolated workload project tie this spike back to Spike 01's `check:diag` proxy.
- 3 warmup + 20 timed samples per position/direction; median + min–max; fresh server per workload
  (M1 measured on that fresh instance); host load recorded in-run (~2.3–3.1 1-min load, the
  interactive desktop session).

Reproduce: `pnpm bench:editor` (generator + harness; workloads dir is gitignored and regenerated).

## Results (run 1 / run 2)

**M2 — warm hover (hard gate ≤ 100 ms, 2× ceiling 200 ms): PASS everywhere.** Medians 0.04–0.08 ms
across all 19 positions in both runs; worst single sample 0.51 ms (W5 dynamic-batch position,
run 1). Chain length has no visible effect (L=1 vs L=100: both ~0.05–0.08 ms) — consistent with
the checker resolving these types once and answering warm hovers from cache. Selection:

| position | run 1 med | run 2 med |
|---|---|---|
| W1 L=100 final (`NDArray<[8]>`) | 0.05 ms | 0.06 ms |
| W2 rank=16 broadcast | 0.05 ms | 0.06 ms |
| W3 dim=65536 after 12 slices (`NDArray<[45526]>`) | 0.05 ms | 0.06 ms |
| W5 dynamic-batch (`NDArray<[number, 8]>`) | 0.08 ms | 0.08 ms |

**M3 — warm edit→diagnostics toggle (hard gate ≤ 500 ms): PASS.** W4 single-token toggle
(matmul inner dim 3↔5, full-text `didChange` + pull diagnostic): → fixed 1.47/1.39 ms median,
→ broken 1.60/1.56 ms median (max outlier 5.41 ms, run 1). Measured on W4 only — the only
spec-defined toggle target.

**M1 — cold project load (informational):** `initialize` 5.8–6.7 ms (one 46.6 ms outlier: the
first workload of run 1, most plausibly first-load page-cache effects on the native exe — the two
runs genuinely disagree here and that is reported, not averaged away). didOpen → pull diagnostics
21–46 ms. First hover after load 1.1–7.3 ms.

**M4 — completion (informational):** 0.07–0.11 ms median, always 9 items (the `NDArray` member
surface).

**Instantiation counts** (per isolated workload project, `--extendedDiagnostics`; identical in
both runs — generation is deterministic):

| workload | instantiations | note |
|---|---|---|
| W1 op chains (L up to 100) | 10,452 | |
| W2 broadcast (rank up to 16) | 12,074 | |
| W3 slice digit arithmetic | 41,515 | highest — matches expectation that Kern-05 arithmetic is the priciest machinery |
| W4 error file | 10,454 | 2 deliberate type errors |
| W5 mixed ~222 LOC | 15,417 | |

All far below the ~5M global budget — these workloads represent *realistic* use, not adversarial
stress (Spike 01 covered the ceilings).

## Deviations from spec (with reasons)

- **Push diagnostics (`publishDiagnostics`) were never observed for opened `.ts` files** in either
  run (case-insensitive URI matching, 40+ toggle round trips) — the 7.0.2 server appears
  pull-only for source files in this mode, contradicting the spec's research note that push "is
  also emitted". M1's push column honestly reports `n/a`; pull timing carries the metric.
  (Verification nuance: push notifications DO fire for the project's `tsconfig.json` — just
  never for the source files this spike times.)
- **M3 on W4 only:** the spec's workload section defines a toggle target only for W4; no
  synthetic toggle sites were invented for the other workloads.
- Valid workloads show 4–6 diagnostics on open — all severity-4 "declared but never read" hints
  (code 6133), a property of straight-line generated code, not errors; W4 has exactly its 2
  designed severity-1 errors.

## What this does NOT show (limitations, per the honesty rule)

- **Editor-process overhead is excluded.** This measures LSP wire latency (what the language
  server contributes). VS Code's extension host, rendering, and debouncing sit on top; they are
  the same for every TS library and not NumType-specific, but total perceived latency is larger
  than these numbers.
- **Small projects.** Workload projects are ~72 files (mostly libs) / ~57 kLOC total. The result
  supports "the type layer is fast in realistic use", NOT "hover stays sub-ms in a 5,000-file
  monorepo" — project scale stresses the server independently of NumType's types.
- **Warm ≠ first-interaction:** cold first-diagnostics is ~20–50 ms here; still far under any
  gate, but the sub-0.1 ms figures are steady-state cache hits, which is what repeated editor
  interaction actually is.
- The 46.6 ms vs 6.7 ms first-`initialize` disagreement between runs is unexplained beyond the
  page-cache hypothesis and left as an honest open point (informational metric, no gate).

## Consequences

- **Roadmap Phase A1 gate: PASS.** No type-design changes needed for editor latency at realistic
  scale; Phase A can proceed to the type-level follow-ons (bounds checks, reshape/flatten).
- The harness is stable and cheap (~1.2 s per full run, wall-clock measured during verification;
  an earlier ~30 s estimate in this doc's draft was wrong and was caught by the verifier) — a
  candidate CI gate (FOLLOWUPS).
- Side effect worth keeping: `spike/src/ambient.d.ts` gained minimal, doc-commented shims
  (global `Buffer`, `node:fs` sync, `node:path`, `node:url`, `node:child_process`) — the project
  still has no `@types/node`; the shims model only what the harness actually calls.

## Post-verification addendum (2026-07-10)

Independent fresh-context verification (brainroute:verify) confirmed acceptance criteria 1–5 and
the results above. Highlights of what it verified independently rather than re-ran:

- **The sub-0.1 ms hover claim was reproduced with a from-scratch probe** (own LSP client, own
  framing, own clock — none of the harness's code): 0.0566 ms median at the harness's own W1
  L=100 position, and **0.0524 ms at a mid-chain position the harness never warms** (correct
  typed answer `NDArray<[8, 8]>`) — ruling out "only the pre-warmed position is cached".
- The W3 digit-arithmetic expectations were hand-recomputed independently (1024→8, 4096→2084,
  65536→45526, multi-axis→[900, 3000]) — all match; the W4 toggle's code-2741 diagnostic is
  genuinely the matmul inner-dim error ("inner dimensions 3 and 5 do not match"), flipping 2↔1
  as designed.
- Gates re-run: `pnpm check` clean, `test:core` 817/817, `bench:editor` PASS with **identical
  instantiation counts (4/4 runs now)** — notably on a host with 1-min load ~10.5, so the gate
  verdict is robust to host load.
- The ambient.d.ts shims were isolated experimentally: **zero instantiation cost** (0→0 on a
  dummy program); the main program's `check:diag` growth since Spike 01 (26,250 → 54,700) is
  attributable to Kern 02–06 code growth, not to this spike. The `tsconfig.json` exclude was
  shown to be load-bearing (without it, W4's deliberate errors would fail `pnpm check`).
- Two findings were folded into this doc (the wrong ~30 s runtime figure; the tsconfig.json push
  nuance). Honest residue, flagged low-severity/low-confidence and left as-is: if a correctness
  gate aborts a run mid-workload, that workload's `tsc --lsp` child is not explicitly shut down
  (abort means process exit; the server exits on stdin EOF — never observed in 4 clean runs).
