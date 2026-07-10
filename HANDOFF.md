# Handoff — 2026-07-10, Session-Ende (nach Härtungen + Kern 06 Threads)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Sieben Phasen + drei Härtungen implementiert, **unabhängig verifiziert, committet und gepusht** (Branch `main` = `origin/main` = `2df8906`; Remote: `github.com/crankfunk/numtype`, privat):

- **Spike 01** Typ-Ebene · **Kern 01** from-scratch Kernels (bit-identisch, handgerolltes ABI) · **Kern 02** Zero-Copy-Residenz · **Kern 03** Strided Views · **Kern 04** blocked+packed+SIMD128-Matmul (2,1–3,25×) · **Kern 05** Slicing (O(1)-Views + statische Slice-Dims via Digit-Arithmetik) · **Härtungen** (v1-OOM-Pfad mit Mock-Allocator-Beweis, Test-Listen-Guard, ABI-rank/len-Prävalidierung unter Scope (a)) · **Kern 06** Threads (handgerolltes Shared-Memory-Substrat ohne wasm-bindgen, allokationsfreier `nt_matmul_blocked_partial`, worker_threads-Pool mit Atomics-Handshake, Poison-Fehlersemantik; ~1,9×/3,3–3,6×/4,0–4,3× mit 2/4/8 Workern ab n=256, Verluste unter n≈100 ehrlich dokumentiert).

Gate-Stand (alle grün, von der Session final selbst gelaufen): `pnpm check` · cargo 110/110 · `test:core` 817 · `test:resident` 2318+2 Skips · `test:resident:gc` 2/2 · `test:threaded` 60/60 (3× stabil) · Demo bit-identisch · Baseline-Artefakt-Hash `a6622a59…` unverändert.

## In dieser Session erledigt
- **Drei kleine Härtungen** (Commit 79a00be): backend.ts-Scratch-Listen-Muster + Mock-Allocator-Test mit Mutation-Check; test-scripts-guard.test.ts; `validate_rank`/`validate_region` vor Slice-Konstruktion in den sechs nicht eingefrorenen Einstiegspunkten (v1 bewusst trust-the-caller, Freeze nicht aufgeweicht).
- **Kern 06 Threads** (Commit 240b13e): Feasibility-Spike (Worktree, empirisch: stable kann `--shared-memory` nicht linken; Stack-Kollisions-Negativkontrolle) + Primärquellen-Recherche → bindende Spec → Implementierung (brainroute:deep) → **drei Verify-Runden**: Runde 1 fand ein echtes Use-after-free im Fehlerpfad (Puffer freigegeben, während frühere Worker noch schrieben) → Fix: Poison-Semantik mit bis nach `worker.terminate()` aufgeschobener, per Zähler beobachtbarer Freigabe → Runde 2 re-bestätigt mit externem Repro → Finale Runde: Timeout-Zweig-Regressionstest (14,8×-Marge) + Disposed-Guard.
- Bench-Nachtrag (03c5c67): der committete workers=1-Verlust (0,75× bei n=512) war Host-Last-Rauschen — Idle-Lauf misst 1,01–1,02×; n=64-Verlust bleibt real.
- Prozessfix (43878a6): versehentlich committeter Agent-Worktree-Gitlink entfernt, `.claude/worktrees/` in .gitignore.
- FOLLOWUPS: Threads + Härtungs-Items ausgetragen; NEU: no_std/stable-Pfad, Browser-Port, Auto-Weiche (2df8906), weitere-Ops-threaden.
- KB: 6 neue Notizen + 3 Upserts (validate-before-slice, Mock-Allocator-Leak-Beweis, Threads-Substrat, Zeilenshift/track_caller, Allokationsfreiheit=Call-Graph, Poison-statt-free; SIMD-Notiz um Parallel-Gesetz erweitert), MOCs verdrahtet, Graph gebaut, Kanten verifiziert.

## Offen / in Arbeit
Nichts halbfertig, Arbeitsbaum clean, alles gepusht. Doku-/Memory-Updates dieser Handoff-Runde (README Kern-06-Satz, CLAUDE.md Freeze-Disziplin-Abschnitt, diese Datei) sind beim Lesen dieser Notiz committet.

## Nächste Schritte (priorisiert)
1. **Aufwärm-Scheibe: Auto-Weiche** (FOLLOWUPS): `threadedMatmul` routet kleine Probleme auf den single-threaded `nt_matmul_blocked` — Schwelle MESSEN (Break-even zwischen n=64 und n=256), Kriterium Arbeitsvolumen batch·m·k·n, nicht Zeilenzahl. Klein, kein Phasen-Doc nötig; Fresh-Context-Verify + FOLLOWUPS-Austrag bleiben Pflicht.
2. **Kür-Kandidat A — Typ-Ebenen-Folgethemen** (durch Kern 05 freigeschaltet): Index-Bounds-Check literaler Indizes (Digit-`Compare` existiert; Entscheidung Compile-Fehler vs. Runtime-only) und/oder reshape/flatten-Produkte (O(Stellen²), Budget-Entscheidung — `check:diag` beobachten).
3. **Kür-Kandidat B — Perf-Hebel:** Packing-Buffer-Wiederverwendung im blocked matmul (bekannter Verify-Nit, hebt vermutlich auch die 8-Worker-Effizienz ~50 %), oder SIMD elementwise (erst messen, memory-bound).
4. Größer/später: Browser-Port des Threads-Pfads (COOP/COEP, async Dispatch), no_std/stable-Pfad, Spezialwerte im Differential-Generator.

## Bekannte Probleme / Stolperfallen
- **Freeze-Disziplin verschärft (Kern 06, steht in CLAUDE.md):** Neuer Code in geteilten Dateien (abi.rs, matmul_blocked.rs, shape.rs) nur ANS ENDE anhängen — Zeilenshift ändert `#[track_caller]`-Bytes unveränderter Funktionen. Freeze-Beweis = Artefakt-Hash aus Clean-Rebuild, nicht Diff-Optik.
- **Threads-Build:** gepinnter `nightly-2026-07-09` (+rust-src), Install-Befehl steht in scripts/build-wasm-threads.sh; env RUSTFLAGS ERSETZT config-Flags (`+simd128` immer mitführen); eigenes target-dir. Nur `test:threaded`/`bench:threaded` brauchen nightly, alles andere stable.
- **Threads-Runtime:** `thread_local!` im Crate verboten (`__tls_base`-Landmine); Views/Atomics-Zellen nie cachen (shared grow scheitert STILL — alte Länge, kein Detach); Worker-Events feuern nicht während synchronem Atomics.wait → Crash-Detection ist deadline-basiert; Pool-Fehlersemantik: poisoned + aufgeschobene Freigabe (getPoisonCleanupFreeCount), nie free solange ein Thread schreiben könnte.
- **Test-Listen:** drei Explizitlisten (core/resident/threaded), Guard erzwingt genau-eine; gc ⊆ resident.
- **Unverändert gültig:** Cargo-Config ist CWD-basiert → alles vom Repo-Root; TS-7-Fallen (CLAUDE.md); v1 + Kern-03/04-Kernels eingefroren; `normalizeSliceSpecs` Differential-Blind-Spot; MC=32 ist in Rust und TS dupliziert (nur Perf-relevant).
- **Bench-Ehrlichkeit:** Threads-Zahlen von unbelasteter Maschine nehmen (der 0,75×-Ausreißer im Ergebnisdoc war Parallellast — Nachtrag steht drin).

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse:** `docs/{spike-01,kern-01..kern-06}-*.md` (Kern 06 mit dreistufigem Verifikations-Addendum), `docs/wettbewerbsanalyse-und-usp.md`, Backlog `FOLLOWUPS.md`.
- **Code:** Typ-Ebene `spike/src/`; WASM-Schicht `spike/src/wasm/` (`backend.ts` v1 FROZEN, `resident.ts` v2, NEU `threaded.ts`/`threaded-worker.ts`/`threaded-protocol.ts`); Rust `crates/core/` (`abi.rs`, `shape.rs`, `kernels/`, NEU `tests/zero_alloc.rs`); Build `scripts/build-wasm-threads.sh` + `.cargo/config.toml`.
- **Befehle:** `pnpm check`/`check:diag` · `pnpm test:core` (817) · `pnpm test:resident` (+`:gc`) · `pnpm test:threaded` (60, nightly!) · `pnpm demo` · `pnpm bench:{scaling,chain,strided,blocked,slice,threaded}` · `cargo test --manifest-path crates/core/Cargo.toml` (110) — alles vom Repo-Root.
