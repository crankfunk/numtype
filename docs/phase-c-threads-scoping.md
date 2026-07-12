# Phase C — Threads platform scoping (decision doc, no build)

Date: 2026-07-12. Owner decision recorded. This is a **scoping/decision** slice
(roadmap Phase C, Items 8–10): no code was written, no build run. The constraint
facts below come from two fresh-context research passes (one on browser web-platform
constraints, one on the Rust stable/no_std toolchain question), grounded in primary
sources; confidence levels are carried through. Facts not independently re-verified by
the author are flagged as such — the honesty rule applies.

## The question

Roadmap Items 8 & 9 asked whether to, before packaging:
- **Item 8** — port the threads path to the browser (COOP/COEP, `crossOriginIsolated`,
  async dispatch instead of blocking `Atomics.wait` on the main thread), **or**
- **Item 9** — get the threads artifact off the pinned `nightly-2026-07-09` +
  `-Z build-std` onto a stable/`no_std` toolchain,

**or** deliberately defer threading for v0 as an explicitly-experimental, Node-only
opt-in. Item 10 (backend-choice API) was blocked on the outcome, because a browser
async-dispatch port would change the layer's public signature.

## Decision (owner, 2026-07-12): defer — Option 1

**Threads stay a Node-only, explicitly-experimental opt-in for v0. Neither the browser
port (Item 8) nor a stable/`no_std` path (Item 9) is built now. Next build slice is
Item 10 (backend-choice API).** This satisfies the release gate as written ("threads
either browser-capable *or* cleanly delimited as Node-only/experimental").

Rationale: the two research strands converge on the same answer.

### Item 9 finding — there is no way off the pinned nightly today
- `-Z build-std` is nightly-only (`-Z` flags never exist on stable). The two 2026
  merged build-std RFCs (**3874**, **3875**) **explicitly do not cover rebuilding std
  with different target-features** (i.e. `+atomics`); 3874's stated goal is a drop-in
  replacement of the prebuilt std, feature/flag customization is deferred to a future,
  not-yet-drafted proposal. "Wait for build-std to stabilize" is therefore a *non-answer*
  to this specific need, not merely slow. (High confidence, RFC text read directly.)
  Sources: rfcs#3874 (merged 2026-04-15), rfcs#3875 (merged 2026-05-04),
  rust-project-goals 2026 build-std page.
- `wasm32-wasip1-threads` **does** ship a precompiled atomics-enabled std on the stable
  channel (verified against the actual `channel-rust-stable.toml` distribution
  manifests, true for ≥2 years) — but it is a **dead end**: it implements the
  **withdrawn** wasi-threads WebAssembly proposal; `thread::spawn` is **currently broken
  on stable/beta** (rust-lang/rust#146721, open since 2025-09-18); no browser implements
  WASI; Node's `node:wasi` has no `wasi_thread_spawn`. Adopting it would require a
  hand-rolled WASI host shim broader than our current bare `extern "C"` ABI. (High
  confidence on the decisive points.)
- `no_std` **probably does not escape** `-Z build-std`: `core` pulls in
  `compiler_builtins`, shipped as precompiled object code **without** atomics, which
  should trip the same `--shared-memory is disallowed` linker wall. **This is the one
  open question** — reasoned inference, no wasm32-specific primary source found either
  way. Resolvable by a ~30-min throwaway experiment (a `no_std` + bump-allocator crate
  built with `+atomics` *without* `-Z build-std`); deferred to FOLLOWUPS, not v0-critical
  (the browser blocker below stands regardless of its outcome).
- **Our pinned-nightly + build-std approach is the ecosystem standard**, not an
  idiosyncratic workaround: `wasm-bindgen-rayon` and `wasm_thread` (the reference Rust
  wasm-threads projects) do exactly the same and recommend a *fixed* nightly. Dated
  nightlies are durable (snapshots from Jan 2023 still resolve today, 3.5 yrs;
  rust-infra prunes only overwritten pointer files after 3 months, never dated
  snapshots) → a pinned date is fully reproducible.

### Item 8 finding — the browser port is feasible but its value is gated
- The port itself is **bounded** against our code: the lifecycle (`initThreadedCore`,
  `dispose`, `poison`) is already async; only the hot path (`threadedMatmul`, a
  deliberately *synchronous* `Atomics.wait`-blocking call, threaded.ts:706) must go
  sync→async via a **coordinator worker** (sync `Atomics.wait` is legal inside a Worker,
  forbidden on the browser main thread where it throws `TypeError`); plus a thin
  platform shim (Web Worker + `fetch` vs. `worker_threads` + `readFile`, and an explicit
  `.terminate()` because browser `Worker.onerror` does not auto-terminate like Node).
  The handshake layer (`threaded-protocol.ts`) is already the shared core. Highest-risk
  item: shipping a worker file **from a library** — Vite library-mode worker bundling is
  broken, esbuild has no worker support → a prebuilt worker file would be needed.
- **But the value is gated on deployment the library cannot control.** SharedArrayBuffer
  requires `self.crossOriginIsolated === true`, which requires COOP+COEP **HTTP response
  headers set by the consuming app's server** — no JS API, no `<meta>` equivalent; a
  library cannot ship them. Enableable on Vercel/Netlify/Cloudflare/AWS/Next.js;
  **impossible on GitHub Pages** (4 yrs unresolved), in sandboxes (CodeSandbox/
  StackBlitz), on plain CDNs, or when the app is embedded cross-origin; COEP
  additionally breaks third-party resources (fonts/analytics/iframes) lacking CORP. A
  meaningful fraction of NumType's target audience cannot turn it on.
- Every comparable project (ffmpeg.wasm `-mt` builds, ONNX Runtime Web, Pyodide) ships
  threading as a **feature-detected opt-in with a single-threaded fallback, never the
  default** — the pattern we are adopting by deferring. (Browser web-platform facts:
  high confidence; MDN/caniuse/web.dev primary sources.)

## What "Node-only" means for usability (the clarifying point)

"Node-only" applies **only to the threads path** — one optional perf feature for one
case — **not to the library.** NumType has three layers; only the smallest is affected:

| Layer | What | Node | Browser | Needs COOP/COEP? |
|---|---|---|---|---|
| Type level (the USP) | shape checks, editor squiggles, static slice/reshape dims | ✅ | ✅ | — (compile-time only) |
| Standard runtime | *all* ops: add/sub/mul/div, matmul (blocked+SIMD), sum, transpose, slice, reshape, dot/norm/cosine | ✅ | ✅ | ❌ no |
| Threads (opt-in) | *only* the parallel multi-worker matmul (`nt_matmul_blocked_partial`) | ✅ | ❌ | ✅ yes |

- The **type level** (what NumType sells) is pure TS compile-time — platform-agnostic,
  untouched by any of this.
- The **standard runtime** covers every operation and is already browser-capable:
  `loader.ts` is dual-target (Node `readFile` / browser `fetch` +
  `instantiateStreaming`, feature-detected via `typeof process`), and the crate exports
  **no imports / no WASI**, so single-threaded WASM needs **no** cross-origin isolation —
  it runs everywhere, GitHub Pages included. (Architecture verified in code; **honest
  caveat: browser-untested** — all tests run in Node. A browser smoke test belongs to
  packaging, Phase D, and is COOP-free/independent of the threads question.)
- A browser user therefore loses **no functionality** — only the multi-worker speedup
  (2.86–4.42× at n≥512) on **large** matmuls. Single-threaded blocked+SIMD matmul
  (2.1–3.25×) remains. And the multi-core matmul win is niche even in-browser: the
  primary embedding/RAG use case is dominated by `dot`/`norm`/`cosine` (vector
  reductions), for which there is **no** threads kernel at all (threading parallelizes
  only matmul).

## Consequences

- **Item 10 (backend-choice API) is the next build slice**, and is *simplified* by this
  decision: keeping the stable backends synchronous (no browser-threads async rewrite)
  means one `NDArray` surface, stable artifact as the default everywhere, threads as an
  environment-detected opt-in (Node + threads artifact present). Item 10 begins with a
  spec/design; the open design fork (single auto-choosing surface vs. explicit backends)
  goes to the owner before the spec is fixed.
- **Release labeling:** the threads add-on must be documented as "experimental,
  Node-only, requires a pinned nightly to *build* (prebuilt `.wasm` shipped — end users
  need no Rust)". The nightly dependency is a CI/contributor/publish-time concern, not a
  user runtime concern. (Note: `.wasm` artifacts are currently `.gitignore`d; bundling
  them into the package is Phase-D work, Item 11.)

## Deferred to FOLLOWUPS (not v0-critical)
- The `no_std`-escapes-`build-std` ~30-min experiment (resolves the one open Item-9
  question).
- The browser threads port itself (Item 8) — revisit only if real demand for in-browser
  multi-core matmul from COOP/COEP-capable consumers appears.
- Browser smoke test of the standard runtime (Phase D packaging; COOP-free).

## Honesty residue
- All constraint facts are from fresh-context research, not the author's own
  primary-source verification, except where re-checked in code (loader dual-target,
  threaded.ts sync hot path, Node-only file set, `.wasm` gitignored).
- Open/unverified: whether `no_std` truly escapes `build-std` for wasm+atomics (§Item 9);
  exact WebKit stance on Document-Isolation-Policy; CSP `'wasm-unsafe-eval'` need for
  `WebAssembly.instantiate` in a worker; several Emscripten/ffmpeg.wasm issue citations'
  July-2026 currency. None are load-bearing for the defer decision.
