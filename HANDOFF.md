# Handoff — 2026-07-10 (nach Kern 04)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Fünf Phasen implementiert, **unabhängig verifiziert** und committet (Branch `main`; Remote: `github.com/crankfunk/numtype`, privat — Kern-03-Commits gepusht, Kern-04-Commit lokal, Push steht aus):

- **Typ-Ebene** (Spike 01) · **Rust/WASM-Kerne** (Kern 01, handgerolltes ABI, bit-identisch) · **Zero-Copy-Residenz** (Kern 02) · **Strided Views** (Kern 03, O(1)-Transpose, Refcount, Status 4).
- **Blocked+Packed+SIMD128-Matmul** (Kern 04, diese Session): `nt_matmul_blocked` (f64x2), bit-identisch unter dem „bit-identity law" (Vektorisierung nur QUER zu Output-Elementen; eine Akkumulator-Kette pro Element in aufsteigender k-Ordnung via Memory-Roundtrip; Packing = reine Datenbewegung; kein FMA/relaxed-simd). **2,1–3,25×** über Kern-03-Skalar (wächst mit n), Kern-03-View-Malus durch Packing eliminiert (View jetzt 1,06–1,76× schneller als materialize-first), Kleinst-Größen ok (n=4 = Noise-Floor). Tiles MC=NC=KC=32 (gemessen). Null `unsafe` im neuen Modul.

Gate-Stand (alle grün): `pnpm check` · cargo 76/76 · `test:core` 791/791 · `test:resident` 1578 (+2 GC-Skips) · `test:resident:gc` 2/2 · Demo bit-identisch.

## In dieser Session erledigt
- Kern 03 committet & gepusht (`428cdca`, `62504d7`).
- **Kern 04 nach korrigiertem Routing**: Spec hier (Fable), **Implementierung delegiert an brainroute:deep** (Sonnet 5 xhigh, spec-verankerter Prompt), dann zwei Review-Schichten: eigenes Diff-Review (Bit-Identitäts-Argument Zeile für Zeile) + Fresh-Context-Verify („meets its spec"; Bit-Identitäts-Argument unabhängig neu hergeleitet, `.wasm` disassembliert, compile_error!-Guard absichtlich ausgelöst). Ein Minor-Befund (stale Error-String) gefixt.
- Bench 3 Läufe; Kern-03-Vorhersage („Packing eliminiert View-Malus") explizit bestätigt und die überholte Guidance in kern-03-ergebnisse.md als superseded markiert.
- KB: 2 neue Notizen (`simd-blocking-ohne-bit-identitaet-zu-brechen`, `wasm-simd128-stille-skalarisierung-cargo-config-cwd`), Views-Notiz revidiert (Malus war kernel-abhängig), 2 MOCs verdrahtet, Graph rebuilt, Kanten verifiziert.
- Prozess-Memory: Delegations-Doktrin für Fable-Sessions als Feedback-Memory festgehalten (Ausführung nach unten, Ausnahmen explizit begründen).

## Offen / in Arbeit
Nichts halbfertig. **Kern-04-Commit lokal — Push nicht beauftragt.**

## Nächste Schritte (Kandidaten, FOLLOWUPS.md)
1. **Slicing am TS-Surface** — ABI unterstützt Offsets seit Kern 03; Typ-Ebenen-Frage (Slice-Shapes) ist der interessante Teil.
2. **Kleine Härtungen als Aufwärm-Scheibe**: OOM-Pfad v1-`backend.ts`, `test:core`-Listen-Guard, ABI-rank/len-Prävalidierung (betrifft inzwischen 6 strided Einstiegspunkte).
3. **Threads** (COOP/COEP, SharedArrayBuffer) — eigene, größere Phase.
4. Perf-Hebel notiert: Packing-Buffer-Reuse; SIMD für elementwise/sum erst nach Messung (memory-bound).

## Bekannte Probleme / Stolperfallen
- Alles aus den letzten Handoffs gilt weiter (TS-7-Limits, Invarianz/`AnyWNDArray`, explizite Test-Listen, TypedArray-View-Regel, Reihenfolge-Disziplin, Free-Counter-Isolation, `contiguous()` kopiert immer).
- **Neu (Kern 04):** Ohne `+simd128` skalarisiert LLVM v128-Ops STILL (kein Build-Fehler) — der `compile_error!`-Guard in `matmul_blocked.rs` fängt das. **Cargo-Config-Discovery ist cwd-basiert**: alle Befehle vom Repo-Root ausführen, sonst fehlt das rustflag (Guard feuert dann — gewollt).
- **Neu (Kern 04):** Native `cargo test` deckt den SIMD-Pfad NICHT (skalarer Micro-Step gleicher Rundungsfolge); das Gate für SIMD ist die TS-Differentialsuite gegen das echte `.wasm`. Coverage-Split ist im Kernel-Doc dokumentiert.
- Prozess: nach `cd` in den Vault zurück-cd-en (pnpm/cargo lösen sonst falsch auf).

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse**: `docs/{spike-01,kern-01..kern-04}-*.md` (je Spec + Ergebnisse mit Verifikations-Addendum), Backlog `FOLLOWUPS.md`.
- **Code**: `spike/src/wasm/` (`loader.ts`, `backend.ts` v1 frozen, `resident.ts`), `crates/core/` (`abi.rs`, `shape.rs`, `kernels/` inkl. `matmul_blocked.rs`), `.cargo/config.toml` (simd128-rustflag), Tests `spike/tests-runtime/` (neu: `blocked.test.ts`), Benches `spike/bench-core/` (neu: `blocked.ts`).
- **Befehle**: `pnpm check` · `pnpm test:core` · `pnpm test:resident` (+`:gc`) · `pnpm demo` · `pnpm bench:scaling` / `bench:chain` / `bench:strided` / `bench:blocked` · `cargo test --manifest-path crates/core/Cargo.toml` — alles vom Repo-Root.
