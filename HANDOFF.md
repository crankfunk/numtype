# Handoff — 2026-07-10, Session-Ende (nach Kern 05, Fokus: kleine Härtungen)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Sechs Phasen implementiert, **unabhängig verifiziert, committet und gepusht** (Branch `main` = `origin/main` = `ad502d6`; Remote: `github.com/crankfunk/numtype`, privat):

- **Spike 01** Typ-Ebene · **Kern 01** from-scratch Kernels (bit-identisch, handgerolltes ABI) · **Kern 02** Zero-Copy-Residenz · **Kern 03** Strided Views (O(1)-Transpose, Refcount, Status 4) · **Kern 04** blocked+packed+SIMD128-Matmul (2,1–3,25×, bit-identisch unterm „bit-identity law") · **Kern 05** Slicing (O(1)-Views mit Nonzero-Offsets, null Rust-Änderungen; Typ-Ebene: graduelle Kern-Regeln + statisch berechnete Slice-Dims via Digit-String-Arithmetik, Budget 1,59× bei 3×-Gate).

Gate-Stand (alle grün): `pnpm check` (+ `check:diag`: 46.855 Instantiations) · cargo 76/76 · `test:core` 791/791 · `test:resident` 2316 (+2 ehrliche GC-Skips) · `test:resident:gc` 2/2 · Demo bit-identisch inkl. Slice-Showcase.

## In dieser Session erledigt
- **Kern 03** (in-Session implementiert), **Kern 04 + Kern 05** (Implementierung an brainroute:deep delegiert, je zwei Review-Schichten: eigenes Diff-Review + Fresh-Context-Verify, alle Verdikte „meets its spec", alle Befunde geschlossen). Je Phase: bindende Spec → Impl → Verify → Ergebnisdoc mit Addendum → KB-Capture → Commit. Alles gepusht.
- Ehrliche Kernbefunde dokumentiert: Kern-03-View-Malus beim Matmul (~30 % ab n=256) → von Kern 04 per Packing eliminiert (Vorhersage explizit bestätigt); Kern-05-Digit-Arithmetik vom Verifier mit 25 Adversarial-Fällen gegen echtes tsc und NumPy-Semantik gegen echtes NumPy bestätigt.
- KB: 5 neue Notizen + 4 Upserts (Views/Zugriffsmuster, logische Ordnung, SIMD-ohne-Bit-Bruch, stille Skalarisierung/cwd-Config, Digit-Arithmetik; Argument-Guard um Rest-Param-Variante erweitert), MOCs verdrahtet, Graph verifiziert.
- Prozess: Delegations-Doktrin für Fable-Sessions als Feedback-Memory verankert (Ausführung nach unten delegieren, Ausnahmen explizit begründen).

## Offen / in Arbeit
Nichts halbfertig, Arbeitsbaum clean, alles gepusht.

## Nächste Schritte (Fokus: kleine Härtungen als Aufwärm-Scheibe, dann Kür)
1. **OOM-Härtung v1-`backend.ts`** (Verify-Befund Kern 02): Shape-Scratch-Allokationen liegen vor dem Output-`nt_alloc` außerhalb von try/finally → Leak, falls dieser fehlschlägt. Vorlage existiert: `resident.ts`' Scratch-Liste + einzelnes `finally` (seit Kern 03, deckt auch OOM zwischen Marshalling-Allokationen). Ehrlichkeits-Hinweis: echter OOM ist im Test kaum deterministisch erzwingbar — struktureller Umbau + alle Suiten grün ist der realistische Beweis, so dokumentieren.
2. **Explizitlisten-Guard für Test-Scripts**: Meta-Test, der die `node --test`-Dateilisten in package.json gegen `spike/tests-runtime/*.test.ts` abgleicht (jede Testdatei muss in genau einer Liste stehen). Henne-Ei beachten: der Guard-Test selbst muss in die Liste. Damit stirbt der wiederkehrende Footgun.
3. **ABI-Härtung rank/len-Prävalidierung** (Verify-Befund Kern 01; seit Kern 03 sechs strided Einstiegspunkte betroffen): **zuerst Scope-Entscheidung nötig** — die v1-Einstiegspunkte sind als Baseline byte-für-byte eingefroren. Optionen: (a) nur die NICHT eingefrorenen strided/blocked Einstiegspunkte härten, (b) Freeze-Ausnahme für semantisch neutrale Härtung, bewiesen durch unveränderte 791/791 + Bench-Stichprobe. Empfehlung: (a), Freeze-Disziplin nicht aufweichen.
4. Danach Kür laut FOLLOWUPS: Typ-Ebenen-Folgethemen (Index-Bounds-Check — Digit-`Compare` existiert; reshape/flatten-Produkte O(Stellen²); Stretch-Erweiterung negative Literale/steps≠1) ODER Threads (eigene große Phase) ODER Perf-Hebel (Packing-Reuse, SIMD elementwise erst nach Messung).
- Prozess für die Härtungen: kein volles Phasen-Doc nötig (Aufwärm-Scheibe, kein „Kern 06"), aber Fresh-Context-Verify vor „fertig" und ehrliche Commit-Message bleiben Pflicht; FOLLOWUPS-Items austragen.

## Bekannte Probleme / Stolperfallen
- **TS 7.0.2**: Tail-Rekursion ~1000, TS5112 bei expliziten Datei-Argumenten (`--ignoreConfig`/scoped `-p`), `allowImportingTsExtensions`; Instantiations-Budget via `pnpm check:diag` tracken.
- **Typ-Ebene**: `NDArray`/`WNDArray` sind invariant → `AnyNDArray`/`AnyWNDArray`; wide types VOR Tupel-Rekursion abfangen; `never` als Fehler-Sentinel „gelingt" still → String-Sentinel; Optional-Property-Inferenz gegen `{}` liefert `unknown` → Required-Property-Pattern; bedingte Typen auf Rest-Params = TS2370 → homomorpher Mapped Type; nie Tupel-Längen-Arithmetik über Dim-WERTE (Digit-Strings nutzen, `spike/src/slice-literal.ts`).
- **Build/Umgebung**: simd128-rustflag lebt in Repo-Root-`.cargo/config.toml`, Cargo-Config-Discovery ist CWD-basiert → ALLE Befehle vom Repo-Root (compile_error!-Guard feuert sonst — gewollt); ohne Flag skalarisiert LLVM still. Nach `cd` in den Obsidian-Vault zurück-cd-en (pnpm löst sonst fremdes package.json auf). pnpm immer, nie npm.
- **Tests**: Explizitlisten-Footgun (bis Guard existiert); Free-Counter-Assertions müssen Handles isolieren (Op-Output-dispose bewegt den Counter synchron); GC-Tests ehrlich gaten; nie TypedArray-Views über Allokationen cachen (memory.grow detacht).
- **Disziplin**: v1-Kernels/-Einstiegspunkte + Kern-03/04-Kernels sind eingefrorene Baselines (git diff leer); Bit-Identität = Reihenfolge spiegeln (bei Views: LOGISCHE Ordnung); `normalizeSliceSpecs` ist geteilter Differential-Blind-Spot (Fixture-Tabelle pinnt ihn); `contiguous()` kopiert immer (bewusst).
- **Prozess**: Implementierung substanzieller Phasen an brainroute:deep delegieren (Spec/Review/Verify bleiben beim Orchestrator); Delegations-Prompts brauchen Umgebungsregeln (nie /tmp, Repo-Root-CWD, Scratchpad); Commit-Hashes nie in Dateien, die im selben Commit landen; KB-Capture ist Pflicht (CLAUDE.md).

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse**: `docs/{spike-01,kern-01..kern-05}-*.md` (je Spec + Ergebnisse mit Verifikations-Addendum), `docs/wettbewerbsanalyse-und-usp.md`, Backlog `FOLLOWUPS.md`.
- **Code**: Typ-Ebene `spike/src/` (`dim/broadcast/matmul/reduce/slice/slice-literal.ts`, naive Referenz `runtime.ts`, `ndarray.ts`); WASM `spike/src/wasm/` (`loader.ts`, `backend.ts` v1 FROZEN, `resident.ts` v2 mit Views/Slicing); Rust `crates/core/` (`abi.rs`, `shape.rs`, `kernels/` inkl. `matmul_blocked.rs`, `materialize.rs`); `.cargo/config.toml` (simd128). Tests `spike/tests/` (Typ-Ebene, läuft via check) + `spike/tests-runtime/` (Explizitlisten!); Benches `spike/bench-core/`.
- **Befehle**: `pnpm check` / `check:diag` · `pnpm test:core` · `pnpm test:resident` (+`:gc`) · `pnpm demo` · `pnpm bench:{scaling,chain,strided,blocked,slice}` · `cargo test --manifest-path crates/core/Cargo.toml` — alles vom Repo-Root; `.wasm` wird automatisch vor Tests/Demo gebaut.
