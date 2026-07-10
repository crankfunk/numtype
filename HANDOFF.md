# Handoff — 2026-07-10, Session-Ende (Auto-Weiche · Roadmap · Spike 02 · Spike 03)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Alles committet und gepusht (`main` = `origin/main` = `63ea082` vor dieser Handoff-Runde; Remote: `github.com/crankfunk/numtype`, privat). Seit dem letzten Handoff sind gelandet:

- **Kern-06-Follow-up: Auto-Weiche** — `threadedMatmul` routet nach Arbeitsvolumen batch·m·k·n; Schwelle GEMESSEN: `THREADED_MATMUL_MIN_POOL_WORK = 262_144` (= 64³) via neuem `bench:crossover`. Überraschungsbefund ehrlich dokumentiert: „Threads verlieren bei n=64" war ein Artefakt der Serie-B-Messeinheit (End-to-End vs. STABLE-Core); auf der echten Router-Vergleichsbasis gewinnt der Pool ab n=64, Main erst bei n≤48 (bis ~21×).
- **docs/roadmap.md** — Weg zu einem möglichen v0.1 research preview: Phasen A (USP absichern) → B (Op-Surface) → C (Plattform) → D (Paketierung), mit Release-Gates. FOLLOWUPS bleibt Backlog-Hauptbuch, Roadmap ordnet nur.
- **Spike 02: Editor-Latenz** — hartes Roadmap-A1-Gate BESTANDEN: handgerollter headless LSP-Harness (`pnpm bench:editor`, null Dependencies) gegen den nativen TS-7.0.2-Server; Warm-Hover 0,04–0,08 ms Median (Gate 100 ms), Edit→Diagnose 1,4–1,6 ms (Gate 500 ms), deterministisch, PASS auch unter Host-Last.
- **Spike 03: Compile-Zeit-Bounds-Checks** — literale OOB-Integer-Indizes an `slice()` sind Compile-Fehler am Argument, wortgleich mit dem Runtime-Throw; negative Literale via Vergleich (keine signierte Arithmetik nötig); nie-falsch-nur-unvollständig (wide/dynamisch/1.5/1e21/Misch-Unions → Runtime-Backstop); 174/174-Paritäts-Grid; Maschinerie-Budget 1,036×.
- **README-Highlight** — Digit-Arithmetik prominent, Neuheits-Claim VOR dem Schreiben verifiziert und sauber gescoped (Technik hat Prior Art: ts-arithmetic, verlinkt; neu ist die Anwendung im gradualen Shape-System).

Gate-Stand (zuletzt gelaufen): `pnpm check` clean · `check:diag` 68 141 Instantiations (54,7k Baseline + Spike-03-Maschinerie 1,036× + neue Typ-Tests) · `test:core` 817 · `test:resident` 2319+2 Skips · `test:threaded` 65/65 (×3) · `bench:editor` PASS · cargo 110 · Baseline-Artefakt-Hash `a6622a59…` unverändert (null Rust-Änderungen in allen vier Scheiben).

## In dieser Session erledigt
- Auto-Weiche implementiert + `bench:crossover` (Commit 21ac0cd; Tests pinnen Routen via `{minPoolWork: 0/∞}`, Route beobachtbar über `postedSeq`; Lifecycle größenunabhängig).
- FOLLOWUPS-Item Backend-Wahl = Platzierungsentscheidung (51105ab; Per-Call-Routing zwischen Cores als Sackgasse dokumentiert).
- Roadmap geschrieben (3ebd598).
- Spike 02 komplett: Spec, Harness (`spike/bench-dx/`), Workload-Generator, zwei Läufe, Verify, Ergebnisdoc (d3bda22).
- Spike 03 komplett: Spec → Implementierung (`LiteralIndexBounds`, `ValidateSpecsAcc`-Erweiterung) → 174er-Paritäts-Grid → Fresh-Context-Verify CONFIRMED → Ergebnisdoc mit Addendum (fa45bc9).
- README-Highlight mit verifiziertem Prior-Art-Status (63ea082).
- KB: Notizen `runtime-fehler-auf-typ-ebene-heben` (neu), `typ-ebenen-arithmetik-digit-strings` (Spike-03-Nachtrag + Prior-Art-Status), `js-wasm-benchmarks-jit-zustand-und-crossover` (Entscheidungsbaseline + Tier-up-Warmup), `typescript-7-nativen-lsp-headless-treiben` (Spike 02); MOCs verdrahtet, Graph gebaut, Kanten verifiziert.
- CLAUDE.md: Spike 03 in Done, Next aktualisiert, neue TS-Gotchas (Union-Distributions-Falle, Ein-Diagnose-pro-Call, Test-Widening).

## Offen / in Arbeit
Nichts halbfertig. CLAUDE.md/HANDOFF.md dieser Handoff-Runde sind beim Lesen dieser Notiz committet.

## Nächste Schritte (priorisiert, = Roadmap Phase A Rest)
1. **reshape/flatten-Produkte auf Typ-Ebene** (FOLLOWUPS): Schulbuch-Multiplikation über Digit-Strings wäre O(Stellen²) — Budget-Entscheidung via `check:diag` (Baseline jetzt 68,1k mit den Spike-03-Tests); Runtime-`reshape` selbst ist Phase-B-Arbeit. Vorher KB `typ-ebenen-arithmetik-digit-strings` lesen.
2. **Varianz-Design entscheiden (A3, vor jedem API-Freeze):** `NDArrayView<out S>` (kovariante Read-only-View ohne guard-tragende Methoden) vs. Spike-Lösung `AnyNDArray` — Grundlagen in docs/spike-01-ergebnisse.md (Addendum, two-of-three rule) + KB `ts-varianz-two-of-three-bei-typlevel-guards`.
3. Danach Phase B (Op-Surface entlang der Use Cases: elementwise-Familie, dot/Norm/Cosine, Runtime-reshape/flatten, keepdims; Spezialwerte im Differential-Generator).
4. Kür/parallel: Packing-Buffer-Reuse (Perf, hebt vermutlich 8-Worker-Effizienz), `bench:editor` als CI-Gate (FOLLOWUPS).

## Bekannte Probleme / Stolperfallen
- **Freeze-Disziplin (CLAUDE.md):** Neuer Code in abi.rs/matmul_blocked.rs/shape.rs nur ANS ENDE; Freeze-Beweis = Artefakt-Hash aus Clean-Rebuild.
- **Threads-Build:** gepinnter `nightly-2026-07-09` (+rust-src) nur für `test:threaded`/`bench:threaded`/`bench:crossover`; env RUSTFLAGS ERSETZT config-Flags (`+simd128` mitführen); `thread_local!` verboten; Views/Atomics-Zellen nie cachen; Pool-Fehlersemantik poisoned + aufgeschobene Freigabe.
- **Threads-Tests/Benches pinnen Routen explizit** — seit der Auto-Weiche würden kleine Shapes sonst still auf Main laufen (`FORCE_POOL = {minPoolWork: 0}` in threaded.test.ts/bench-core).
- **Typ-Ebene (neu, Spike 03):** Union-Distributions-Falle beim Klassifizieren von `Compare`-Ergebnissen (nur tuple-gewrappte Subset-Checks); Runtime-Fehlerpfad-Tests brauchen bewusst geweitete Literale (`5 as number`); TS7-nativ zeigt nur EINE Diagnose pro Call/Pass bei mehreren invaliden Argumenten.
- **bench-dx:** `spike/bench-dx/workloads/` ist generiert + gitignored und aus dem Root-tsconfig AUSGESCHLOSSEN (tragend: W4 enthält absichtliche Fehler); Generator deterministisch halten (keine Zeitstempel).
- **Bench-Ehrlichkeit:** unbelastete Maschine; Crossover-Messungen auf der tatsächlich wählbaren Alternative baselinen (KB-Notiz); µs-Benches brauchen Batch-Timing + per-Modul-Tier-up-Warmup.
- **Unverändert:** cargo-Config CWD-basiert → alles vom Repo-Root; Test-Explizitlisten mit Guard; v1 + Kern-03/04-Kernels eingefroren; `normalizeSliceSpecs`-Differential-Blind-Spot; MC=32 dupliziert (nur Perf).

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse:** `docs/{spike-01..03,kern-01..06}-*.md` · `docs/roadmap.md` · `docs/wettbewerbsanalyse-und-usp.md` · Backlog `FOLLOWUPS.md`.
- **Code:** Typ-Ebene `spike/src/` (`slice-literal.ts` = Digit-Arithmetik + `LiteralIndexBounds`, `slice.ts` = Guard); WASM `spike/src/wasm/` (`backend.ts` v1 FROZEN, `resident.ts`, `threaded.ts` mit Auto-Weiche); DX-Harness `spike/bench-dx/`; Rust `crates/core/`.
- **Befehle (alle vom Repo-Root):** `pnpm check`/`check:diag` · `pnpm test:core` (817) · `pnpm test:resident` (+`:gc`) · `pnpm test:threaded` (65, nightly!) · `pnpm demo` · `pnpm bench:{scaling,chain,strided,blocked,slice,threaded,crossover}` · `pnpm bench:editor` (~1,2 s, kein Rust nötig) · `cargo test --manifest-path crates/core/Cargo.toml` (110).
