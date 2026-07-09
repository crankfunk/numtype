# Handoff — 2026-07-10

## Aktueller Stand
NumType (Forschungsprojekt: typsichere n-dim Arrays — TS-Typ-Ebene + from-scratch Rust/WASM-Kerne). Drei Phasen sind implementiert, **unabhängig verifiziert** und committet (Branch `main`, 6 Commits, Arbeitsbaum clean, **kein Remote** — lokal-privat):

- **Typ-Ebene** (Spike 01): `Broadcast`/`MatMul`/`ReduceAxis`/`Transpose` im Typsystem, graduell (`number`-Dims und `number[]`-Rang degradieren ehrlich), Shape-Fehler am schuldigen Argument mit benannten Shapes.
- **Rust/WASM-Kerne** (Kern 01): handgerolltes `extern "C"`-ABI (kein wasm-bindgen, leere `[dependencies]`), **bit-identisch** zur naiven TS-Referenz (791 Differentialfälle), ~1,3–2,5× über naiv-TS.
- **Zero-Copy-Residenz** (Kern 02 + fromArray-Follow-up): `WNDArray` lebt im WASM-Speicher, dispose/FinalizationRegistry-Lifecycle adversarial verifiziert, Leak-Plateau nicht-vakuös bewiesen; `fromArray` akzeptiert `Float64Array` direkt (Chain-Defizit eliminiert).

Gate-Stand (alle grün): `pnpm check` · cargo 38/38 · `test:core` 791/791 · `test:resident` 678 (+1 ehrlicher GC-Skip) · Demo läuft alle drei Backends bit-identisch.

## In dieser Session erledigt
- Wettbewerbsanalyse & USP (docs/wettbewerbsanalyse-und-usp.md): Lücke real & unbesetzt; „NumType : NumPy = TypeScript : JavaScript".
- Spike 01 inkl. Verifikation; Kernbefund „two-of-three" (sound degradation / Fehler-am-Argument / Top-Typ-Zuweisbarkeit → max. zwei; `AnyNDArray`).
- Kern 01 inkl. Scaling-Bench (kein Small-Op-Crossover; JIT-Zustand verschiebt Ratios).
- Kern 02 inkl. Chain-Bench-Root-Cause (`Array.from`-Boxing ~100× teurer als memcpy) und fromArray-Overload-Fix (Defizit 0,76–0,91× → 1,00–1,01×; memcpy-Kopien waren nie das Problem).
- KB-Capture als Pflicht-Workflow in CLAUDE.md verankert; 12 Notizen in der coding-kb (TS-Typ-Ebene, WASM-ABI, Determinismus/Benchmarks, Lifecycle, Test-Methodik).

## Offen / in Arbeit
Nichts halbfertig. Bewusst zurückgestellt (alles in FOLLOWUPS.md): OOM-Härtung beider Backends, `test:core`-Listen-Guard, strided Kernels, SIMD128/Threads/Blocking, Transzendenten-Determinismus-Entscheidung, `reshape`/`flatten` (Typ-Ebenen-Produkte), Varianz-Design (`NDArrayView<out S>`), echte Editor-Latenz-Messung, npm-Name `numtype` reservieren (war am 2026-07-09 frei).

## Nächste Schritte
1. **Strided Kernels** (Views ohne Materialisierung) — erster echter Payoff des Residenz-Speichermodells; als „Kern 03" mit bindender Spec nach etabliertem Muster.
2. **SIMD128 + Blocking fürs Matmul** — der nächste reale Performance-Sprung (naive Skalar-Loops sind die aktuelle Grenze, nicht die Architektur).
3. Kleine Härtungen aus der Kern-02-Verifikation (OOM-Pfad, Listen-Guard) — gut als Aufwärm-Scheibe.

## Bekannte Probleme / Stolperfallen
- **TS 7.0.2** (nativer Go-Compiler): Limits empirisch bestätigt (~1000 tail-rekursiv, Rang 999 ✓/1000 ✗); TS5112 bei expliziten Datei-Argumenten → `--ignoreConfig` oder scoped `-p`; `allowImportingTsExtensions` nötig (Node 24 führt `.ts` direkt aus).
- **`NDArray<Shape>`/`WNDArray<Shape>` sind KEINE impliziten Supertypen** (invariant, „two-of-three", docs/spike-01-ergebnisse.md Addendum) → `AnyNDArray`/`AnyWNDArray` verwenden; Invarianz ist per @ts-expect-error gepinnt.
- **Test-Scripts nutzen explizite Dateilisten** in package.json — neue Testdateien dort manuell eintragen, sonst laufen sie still nie.
- **Nie TypedArray-Views über Allokationen hinweg cachen** (`memory.grow` detacht) — immer frisch aus `memory.buffer` ableiten.
- **Reihenfolge-Disziplin für Bit-Identität**: Rust-Kernels spiegeln `runtime.ts`-Schleifen exakt; v1 bleibt transzendentenfrei.
- Prozess: Delegations-Prompts an Subagenten brauchen Umgebungsregeln (nie /tmp — hängt den Lauf; Session-Scratchpad nutzen); KB-Capture ist Pflicht (CLAUDE.md); Commit-Hashes nie in Dateien schreiben, die im selben Commit landen.

## Wichtige Dateien & Befehle
- **Specs & Ergebnisse**: `docs/{spike-01,kern-01,kern-02}-*.md` (je Spec + Ergebnisse mit Verifikations-Addendum), `docs/wettbewerbsanalyse-und-usp.md`, Backlog `FOLLOWUPS.md`.
- **Code**: `spike/src/` (Typ-Ebene `dim/broadcast/matmul/reduce.ts` + naive Referenz `runtime.ts` + `ndarray.ts`), `spike/src/wasm/` (`loader.ts`, `backend.ts` v1, `resident.ts` v2), `crates/core/` (Rust, zero deps, `abi.rs` + `kernels/`), Tests `spike/tests-runtime/`, Benches `spike/bench-core/`.
- **Befehle**: `pnpm check` · `pnpm test:core` · `pnpm test:resident` (+ `pnpm test:resident:gc`) · `pnpm demo` · `pnpm bench:scaling` · `pnpm bench:chain` · `cargo test --manifest-path crates/core/Cargo.toml`. Build-Kette: `.wasm` wird von den Scripts automatisch vor Tests/Demo gebaut.
