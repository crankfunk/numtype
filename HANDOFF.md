# Handoff — 2026-07-10 (nach Kern 03)

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Vier Phasen sind implementiert, **unabhängig verifiziert** und committet (Branch `main`, Arbeitsbaum clean; Remote: `github.com/crankfunk/numtype`, privat — Kern-03-Commit ist noch NICHT gepusht):

- **Typ-Ebene** (Spike 01): `Broadcast`/`MatMul`/`ReduceAxis`/`Transpose` im Typsystem, graduell, Shape-Fehler am schuldigen Argument.
- **Rust/WASM-Kerne** (Kern 01): handgerolltes `extern "C"`-ABI (kein wasm-bindgen, leere `[dependencies]`), bit-identisch zur naiven TS-Referenz.
- **Zero-Copy-Residenz** (Kern 02 + fromArray-Follow-up): `WNDArray` lebt im WASM-Speicher; dispose/FinalizationRegistry-Lifecycle adversarial verifiziert.
- **Strided Views** (Kern 03, diese Session): `transpose()` ist O(1)-View (geteilter refgezählter Buffer, Shape+Strides revers); fünf strided ABI-Einstiegspunkte + Status 4 (Strides-Bounds-Validierung); alle Resident-Ops laufen über die strided Kernels (Routing gemessen gratis, 0,90–1,03×); `contiguous()`-Escape-Hatch; v1-Kernels byte-für-byte eingefroren. Docs: `docs/kern-03-strided-spec.md` + `-ergebnisse.md` (mit Verifikations-Addendum).

Gate-Stand (alle grün): `pnpm check` · cargo 63/63 · `test:core` 791/791 · `test:resident` 1412 (+2 ehrliche GC-Skips) · `test:resident:gc` 2/2 · Demo läuft alle drei Backends bit-identisch inkl. View-Showcase.

## In dieser Session erledigt
- Kern 03 komplett nach Phasenmuster: bindende Spec → Implementierung → Fresh-Context-Verify (brainroute:verify: „meets its spec", 0 critical/major; alle 4 minor/nit-Befunde behoben, u. a. Raw-ABI-Status-4-Test nachgezogen) → Ergebnisdoc → KB-Capture → Commit `428cdca`.
- **Ehrlicher Kernbefund (Bench, 3 Läufe):** Views gewinnen ~2× bei consume-once (`Aᵀ.sum()`), verlieren aber ~30 % vor dem Matmul ab n≥256 (strided k-Loop-Reads teurer als die gesparte O(n²)-Kopie — deshalb packt BLAS). Guidance: `contiguous()` vor heißen Matmuls.
- Determinismus-Falle gelöst: `sum_all` über Views akkumuliert in LOGISCHER Row-Major-Ordnung; nicht-vakuös gepinnt (Absorptions-Muster 1e100/−1e100 — die erste „gemischte Magnituden"-Testdatenwahl unterschied die Ordnungen NICHT, der assert_ne-Guard fing das).
- OOM-Scratch-Leak in resident.ts nebenbei geschlossen (Scratch-Liste + einzelnes finally); v1-Hälfte bleibt als FOLLOWUP.
- KB-Capture: 2 neue Notizen (`zero-copy-views-zugriffsmuster-entscheidet`, `float-reduktion-ueber-views-logische-ordnung`), 2 Upserts (FinalizationRegistry-Idiom um Refcount-Sharing erweitert; Differentialtest-Notiz verlinkt), 2 MOCs verdrahtet, Graph rebuilt, Link-Kanten verifiziert.

## Offen / in Arbeit
Nichts halbfertig. Kern-03-Commit lokal — **Push steht aus** (nicht explizit beauftragt).

## Nächste Schritte
1. **SIMD128 + Blocking fürs Matmul** — der nächste reale Performance-Sprung; Kern 03 Serie A ist der direkte Beleg, dass Speicherzugriffsmuster (nicht Architektur) der Matmul-Flaschenhals sind. Blocking/Packing adressiert genau den gemessenen ~30 %-View-Verlust.
2. Kleine Härtungen: OOM-Pfad v1-`backend.ts`, `test:core`-Listen-Guard, ABI-rank/len-Validierung vor Slice-Konstruktion (betrifft seit Kern 03 fünf weitere Einstiegspunkte).
3. Bei Gelegenheit: Slicing am TS-Surface (ABI unterstützt Offsets bereits — kein ABI-Bruch nötig).

## Bekannte Probleme / Stolperfallen
- Alles aus dem letzten Handoff gilt weiter (TS 7.0.2-Limits; `AnyNDArray`/`AnyWNDArray` wegen Invarianz; explizite Test-Dateilisten in package.json; nie TypedArray-Views über Allokationen cachen; Reihenfolge-Disziplin für Bit-Identität; v1 transzendentenfrei).
- **Neu (Kern 03):** Free-Counter-Assertions müssen Handles isolieren — Op-Output-`dispose()` bewegt `getResidentFreeCount()` synchron mit (zweimal in Tests reingefallen, dokumentiert in kern-03-ergebnisse.md).
- **Neu (Kern 03):** `contiguous()` kopiert IMMER (auch wenn schon kontiguierlich) — bewusste Ownership-Entscheidung, nicht optimieren ohne die Lifecycle-Implikationen zu bedenken.
- Prozess: Nach `cd` in den Obsidian-Vault (KB-Upserts) das Zurück-`cd` nicht vergessen — `pnpm` löst sonst gegen ein fremdes package.json im Vault-Baum auf und schlägt mit verwirrenden Deps-Fehlern fehl (in dieser Session passiert).
- Prozess: Delegations-Prompts an Subagenten brauchen Umgebungsregeln (nie /tmp; Session-Scratchpad nutzen); Commit-Hashes nie in Dateien schreiben, die im selben Commit landen.

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse**: `docs/{spike-01,kern-01,kern-02,kern-03}-*.md` (je Spec + Ergebnisse mit Verifikations-Addendum), `docs/wettbewerbsanalyse-und-usp.md`, Backlog `FOLLOWUPS.md`.
- **Code**: `spike/src/` (Typ-Ebene + naive Referenz `runtime.ts`), `spike/src/wasm/` (`loader.ts`, `backend.ts` v1 eingefroren, `resident.ts` v2+Views), `crates/core/` (Rust, zero deps: `abi.rs`, `shape.rs`, `kernels/` inkl. `materialize.rs`), Tests `spike/tests-runtime/` (neu: `strided.test.ts`, `strided-lifecycle.test.ts`), Benches `spike/bench-core/` (neu: `strided.ts`).
- **Befehle**: `pnpm check` · `pnpm test:core` · `pnpm test:resident` (+`:gc`) · `pnpm demo` · `pnpm bench:scaling` / `bench:chain` / `bench:strided` · `cargo test --manifest-path crates/core/Cargo.toml`. Build-Kette: `.wasm` wird von den Scripts automatisch vor Tests/Demo gebaut.
