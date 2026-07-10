# NumType — Spike 02: Editor Latency of the Type Layer (Spec)

Date: 2026-07-10 · Status: in progress

## Why (intent)

NumType's USP **is** the editor experience: shape errors as squiggles while you type, hovers
showing inferred shapes (`NDArray<[2, 4]>`). Spike 01 measured the type layer only through the
`tsc --extendedDiagnostics` proxy (batch compile time and instantiation counts); the roadmap
(docs/roadmap.md, Phase A1) names real editor latency as the **hard release gate**: if
hover/diagnostics become sluggish on realistic op chains, the type design must react before
anything else is built. This spike measures it. Honest negative results are as valuable as
passing gates — a failed gate redirects Phase A, it does not get argued away.

## Measured foundation (primary-source research, 2026-07-10)

- `typescript@7.0.2` (installed, native/Go generation) ships one binary (`tsc`, a Node wrapper
  around a per-platform native exe resolved via `lib/getExePath.js`). The language server starts
  as **`tsc --lsp --stdio`** and speaks **LSP**; the `initialize` response identifies
  `serverInfo: {"name":"typescript-go","version":"7.0.2"}` and advertises `hoverProvider`,
  `completionProvider`, and **pull diagnostics** (`diagnosticProvider`); push
  `textDocument/publishDiagnostics` is also emitted after `didOpen`. (Verified live on the
  installed package; Microsoft's own benchmark infra drives the same entry point:
  microsoft/typescript-benchmarking, `lsp-startup` scenario.)
- The official VS Code integration is a genuine LSP client (`vscode-languageclient` over stdio,
  microsoft/typescript-go `_extension/src/client.ts`) — so a headless LSP harness measures the
  same wire protocol the editor uses. What it does NOT measure: VS Code extension-host and
  rendering overhead (documented limitation, see Scope).
- Pitfall (verified empirically during research): the server issues server→client requests
  (`client/registerCapability`, `workspace/configuration`); a client that does not answer them
  stalls hover indefinitely. The harness MUST answer them.
- `typescript@7.0.2` bundles **no** legacy `tsserver.js`; a TS 5.x comparison would need a
  side-by-side install (declared optional stretch, not part of this spike's gates).

## What is measured (metrics)

All timings are client-observed wall-clock from request write to usable response read, over the
stdio LSP channel, requests strictly serialized (the server is multithreaded; we measure
editor-like single-interaction latency, not throughput).

- **M1 — cold project load:** `initialize` roundtrip, and `didOpen` of the workload file →
  first full diagnostics for it (pull `textDocument/diagnostic`; the first push
  `publishDiagnostics` is recorded too if it arrives earlier). One cold measurement per fresh
  server process per workload.
- **M2 — warm hover (the USP interaction):** `textDocument/hover` at predefined positions whose
  expected hover text is an inferred shape type. Warm = after warmup requests at that position.
- **M3 — warm incremental diagnostics (the squiggle loop):** `didChange` toggling a single
  token that flips the file between "one deliberate shape error" and "clean" → pull
  `textDocument/diagnostic` response reflecting the change. Measured in both directions.
- **M4 — completion (informational, no gate):** `textDocument/completion` at a member-access
  position on an `NDArray` value.

**Correctness gate before any timing** (analog of the bench rule "never time a wrong result"):
a hover sample only counts if its content contains the expected shape text (e.g. `[2, 4]`);
an M3 sample only counts if the diagnostics actually appear/disappear as expected; expected
error positions must produce the expected `code`/message-fragment. Wrong/empty responses fail
the run loudly.

## Workloads (generated, importing the REAL type layer)

A dedicated harness project (own directory + `tsconfig.json` consistent with the spike's
compiler settings) importing `spike/src` types — measured against the real product types, not
simplified stand-ins. Generated deterministically by a checked-in generator:

- **W1 op-chain sweep:** chains of `matmul`/`add`/`transpose`/`sum` of length
  L ∈ {1, 5, 10, 25, 50, 100} over literal shapes (hover at the chain end).
- **W2 rank/broadcast sweep:** broadcast-heavy `add` across ranks r ∈ {2, 4, 8, 16} with
  mixed size-1 axes.
- **W3 slice/digit-arithmetic stress:** many `slice()` calls with literal bounds on large
  literal dims (1024, 4096, 65536) — Kern 05's digit-string arithmetic is our own highest-risk
  type machinery.
- **W4 error file:** valid code surrounding deliberate shape errors (matmul inner-dim mismatch,
  broadcast conflict) — M3's toggle target.
- **W5 realistic mixed file:** ~200–400 LOC resembling an embedding-pipeline consumer
  (creation, slicing, matmul, reductions, a `number`-dim gradual boundary).

## Gates (our definition of "acceptable", stated up front)

- **Hard gate (roadmap A1):** warm hover (M2) median ≤ 100 ms at every defined position of
  every workload, AND warm incremental diagnostics (M3) median ≤ 500 ms per direction per
  workload. No single workload median may exceed 2× its gate.
- **Informational (recorded, no gate):** M1 cold load, M4 completion, instantiation counts via
  the existing `check:diag` for the same workload files (ties Spike-01's proxy metric to the
  real one).
- Rationale: 100 ms is the standard perceived-as-instant threshold for direct-manipulation
  feedback; sub-500 ms keeps the edit→squiggle loop comfortably inside "responsive" for
  background diagnostics. Gates are OUR definitions — the results doc reports raw numbers so
  readers can apply their own.

## Measurement discipline (same rules as the perf benches)

Deterministic workload generation (no timestamps/randomness in generated code); serialized
requests; per-position warmup before timed samples; N ≥ 20 timed samples for M2/M3, median +
min–max reported (never single points); host-state note (load average) in the results doc;
**two full runs** — findings must agree in direction, disagreements are reported, not averaged
away. The harness is hand-rolled (JSON-RPC framing over stdio, ~small) — no external LSP client
dependency, consistent with the no-external-libraries constraint (dev-tooling exception is for
compilers/test runners, and a from-scratch client is also what keeps the measurement
transparent).

## Scope

- **In:** the harness (`spike/bench-dx/`), workload generator, `pnpm bench:editor` script,
  measurements on the reference machine, results doc with gate verdicts.
- **Out:** VS Code extension-host/render overhead (limitation, documented); TS 5.x tsserver
  side-by-side (optional stretch — only if the harness lands early, never at the cost of run
  quality); ANY fix/optimization of the type layer (follow-up items derived from findings);
  editor-latency CI automation (candidate follow-up if the harness proves stable).

## Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Harness completes an LSP session against the installed 7.0.2 server, answering `workspace/configuration` / `client/registerCapability`; failures are loud, never hangs (global per-request timeout). |
| 2 | Correctness gates enforced before timing (hover text contains expected shape; error toggle observed) — a run with a wrong response aborts with a named position. |
| 3 | All workloads W1–W5 measured for M1–M4, cold and warm separated, two full runs. |
| 4 | Gate verdict (pass/fail per workload, hard gate overall) stated explicitly in the results doc, with raw medians/ranges and host-state note. |
| 5 | `check:diag` instantiation counts recorded for the same workload files (proxy↔real linkage). |
| 6 | Results doc (`docs/spike-02-ergebnisse.md`) follows the honesty rule (commands shown, failures/gaps named), incl. the "what this does not measure" limitation. |
| 7 | Fresh-context verification before "done"; KB capture of general lessons; FOLLOWUPS updated (item ticked; discovered follow-ups entered). |
