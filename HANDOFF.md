# Handoff — 2026-07-10 (nach Kern 05)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Sechs Phasen implementiert, **unabhängig verifiziert** und committet (Branch `main`; Remote: `github.com/crankfunk/numtype`, privat — bis Kern 04 gepusht, Kern-05-Commit lokal, Push steht aus):

- **Typ-Ebene** (Spike 01) · **Kerne** (Kern 01) · **Residenz** (Kern 02) · **Strided Views** (Kern 03) · **Blocked+SIMD-Matmul** (Kern 04).
- **Slicing** (Kern 05, diese Session): `slice(...specs)` auf beiden Klassen (NumPy-Basic-Slicing minus negative Steps; Indizes werfen bei OOB, Ranges clampen, step ≥ 1). `WNDArray.slice` = O(1)-View — erste Nonzero-Offsets, end-to-end durch alle strided Kernels + blocked Matmul; **null Rust-Änderungen** (ABI war seit Kern 03 bereit). Typ-Ebene: Kern-Regeln (Integer droppt Achse statisch, `null` erhält Literal-Dim, Range degradiert ehrlich, Fehler-am-Argument bei zu vielen Specs via homomorphem ErrorTuple — `Guard` ist auf Rest-Params illegal, TS2370) **plus gelandetes Stretch-Goal**: statisch berechnete Slice-Dims via selbstgebauter Digit-String-Arithmetik (`spike/src/slice-literal.ts`; `NDArray<[1024]>.slice({start:100,stop:1000})` hovert als `NDArray<[900]>`). Budget: 46.855 Instantiations = 1,59× Baseline (Gate ≤3×), Zeit flach.

Gate-Stand (alle grün): `pnpm check` · cargo 76/76 (Crate-Diff leer) · `test:core` 791/791 · `test:resident` 2316 (+2 GC-Skips) · `test:resident:gc` 2/2 · Demo inkl. Slice-Showcase.

## In dieser Session erledigt
- Kern 04 gepusht (`109d1aa`).
- **Kern 05 nach dem Delegations-Muster**: Spec hier (inkl. NumPy-Fixture-Tabelle + Stretch-Gates), Implementierung bei brainroute:deep, eigenes Review (Digit-Arithmetik Zeile für Zeile; 2 Borrow-Chain-Typ-Pins ergänzt: 1000−1, 100−99), Fresh-Context-Verify: **„meets its spec"**, Digit-Arithmetik mit 25 eigenen Adversarial-Fällen gegen echtes tsc als korrekt bestätigt, NumPy-Semantik gegen echtes NumPy geprüft, Baseline im isolierten Worktree re-gemessen. Beide Minor-Befunde (ungetestete Number.isInteger-Pfade; Broadcast-zwischen-verschieden-geslicten-Operanden ungedeckt) mit +31 Tests geschlossen.
- KB: neue Notiz `typ-ebenen-arithmetik-digit-strings` (inkl. der drei Fallen: never-Sentinel, Optional-Property→unknown, TS2370), Upsert der Argument-Guard-Notiz um die Rest-Parameter-Variante, MOC verdrahtet, Graph rebuilt + Kanten verifiziert.

## Offen / in Arbeit
Nichts halbfertig. **Kern-05-Commit lokal — Push nicht beauftragt.**

## Nächste Schritte (Kandidaten, FOLLOWUPS.md)
1. **Kleine Härtungen als Aufwärm-Scheibe**: OOM-Pfad v1-`backend.ts`, `test:core`-Listen-Guard, ABI-rank/len-Prävalidierung (6 strided Einstiegspunkte).
2. **Typ-Ebenen-Folgethemen, durch Kern 05 freigeschaltet**: Bounds-Check literaler Indizes (Digit-`Compare` existiert), Stretch-Erweiterung (negative Literale = vorzeichenbehaftete Addition; steps ≠ 1 = ceil-Division), `reshape`/`flatten` (Produkte = O(Stellen²) — jetzt Design-Frage, kein Limit).
3. **Threads** (COOP/COEP, SharedArrayBuffer) — eigene, größere Phase.

## Bekannte Probleme / Stolperfallen
- Alles aus den letzten Handoffs gilt weiter (TS-7-Limits; Invarianz/`AnyWNDArray`; explizite Test-Listen; TypedArray-View-Regel; Bit-Identitäts-Disziplin; Free-Counter-Isolation; simd128-Guard + CWD-basierte Cargo-Config → alles vom Repo-Root; nach Vault-`cd` zurück-cd-en).
- **Neu (Kern 05, Typ-Ebene):** `never` als Fehler-Sentinel in Typ-Helpern „gelingt" still (never extends alles) → String-Sentinel nutzen. Optional-Property-Inferenz gegen `{}` liefert `unknown`, nicht `undefined` → Required-Property-Pattern. Bedingte Typen auf Rest-Parametern: TS2370 bricht die Deklaration → homomorpher Mapped Type.
- **Neu (Kern 05, Runtime):** Nicht-Integer-Slice-Komponenten werden eagerly geworfen (`Number.isInteger`) — fraktionale Offsets/Strides würden strided Reads still korrumpieren.
- `normalizeSliceSpecs` ist von beiden Backends geteilt (dokumentierter Differential-Blind-Spot) — Semantik ist über die Fixture-Tabelle direkt gepinnt, nicht übers Differential.

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse**: `docs/{spike-01,kern-01..kern-05}-*.md` (je Spec + Ergebnisse mit Verifikations-Addendum), Backlog `FOLLOWUPS.md`.
- **Code**: Typ-Ebene `spike/src/` (`slice.ts` + `slice-literal.ts` neu; `dim/broadcast/matmul/reduce.ts` unverändert), Runtime `runtime.ts` (+ Normalizer/sliceRuntime), `spike/src/wasm/` (`resident.ts` + slice-View), `crates/core/` (seit Kern 04 unverändert), Tests `spike/tests{,-runtime}/` (neu: `slice.test-d.ts`, `slice.test.ts`), Benches `spike/bench-core/` (neu: `slice.ts`).
- **Befehle**: `pnpm check` (+`check:diag` für Instantiations-Budget) · `pnpm test:core` · `pnpm test:resident` (+`:gc`) · `pnpm demo` · `pnpm bench:*` · `cargo test --manifest-path crates/core/Cargo.toml` — alles vom Repo-Root.
