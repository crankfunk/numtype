# Item 11 / S1 — Ergebnisse (Typ-Vorarbeiten: sum-Overload-Umbau + slice-literal-Umbenennung)

**Stand:** erledigt & DREIfach verifiziert (Spec CONFIRMED + adversarial HÄLT + covenant-verify
kein Verstoß), 2026-07-17. Bindende Spec: `docs/item-11-api-paket-spec.md` (Abschnitte T1a/T1b
+ Baustein-0-Addendum). Ausgangscommit b51c1a7.

> **Ehrlichkeitsregel:** Jede Zahl/Aussage hier ist in einem tatsächlich gelaufenen Kommando
> verankert (Haupt-Tree-Läufe des Orchestrators + drei unabhängige Verifier-Läufe in isolierten
> Worktrees). Der Worktree-Kontaminations-Vorfall (unten) wird offen berichtet, nicht geglättet.

## Was S1 gemacht hat

### T1a — `sum`-Overload-Umbau (COVENANT-M2-Verstoß geschlossen, BEIDE Facetten)

Der M2-Verstoß: `a.sum(u)` mit `u: 0|undefined` bzw. `a.sum(0, kd)` mit `kd: true|undefined` war
konfident FALSCH, weil TS `undefined` an einem OPTIONALEN Parameter aus der Inferenz streift, bevor
`ReduceAxis` es sieht. Fix = die optionalen Parameter durch **Overloads nach Argument-Anzahl**
ersetzen (0-Arg / 1-Arg axis-required / 2-Arg axis+keepdims-both-required), sodass in der
Mehr-Argument-Form kein optionaler Parameter mehr existiert; Impl-Signatur-Rückgabetyp auf
`NDArray<any>`/`WNDArray<any>` (extern unsichtbar, Guard/M3 unberührt). KD-2 erweitert zusätzlich
`reduce.ts`s `KeepDims`-Constraint an `ApplyAt`/`ResolveAndApply`/`ReduceAxis` von `boolean` auf
`boolean | undefined`, damit die ungestrippt ankommende `keepdims`-Union über die vorhandenen
`KeepDims extends true`-Conditionals zu einer **ehrlichen Shape-Union** distribuiert.

Ausgänge (verifiziert, beide Surfaces NDArray + WNDArray):
- `a.sum(u)`/`u:0|undefined` → **`readonly number[]`** (Degradation zu no-claim, never-wrong).
- `a.sum(0, kd)`/`kd:true|undefined` → **`readonly [3] | readonly [1,3]`** (ehrliche Union).
- `a.sum(undefined, kd)`/`kd:true|undefined` → **`readonly [] | readonly [1,1]`**.
- Die vier kanonischen Formen unverändert; `sum(9)` bleibt Compile-Fehler AM Argument mit
  wortgleichem Stamm `reduce: axis 9 is out of range for shape [2,3] (rank 2)` (M3 erhalten).
- Der frühere `UA_GAP`-Sentinel-Pin ist **umgekehrt** (`UA_AXIS_CLOSED` + neue `UA_KEEP_CLOSED`/
  `UA_KEEP_CLOSED_FULL` + WNDArray-Zwillinge) und bewacht künftig die Schließung.

**Disclosed Deviation (Owner-bestätigt 2026-07-17):** die Impl-Signatur bekommt eine geänderte
Rückgabetyp-Annotation (`NDArray<OkShape<…>>` → `NDArray<any>`) — nicht strikt insertion-only, aber
Änderung AM `sum`-Member selbst (analog Kern-09-D3). `reduce.ts` ist keine frozen-baseline-Datei;
die Erweiterung ist rückwärtskompatibel (alle bestehenden `.sum(axis, keepdims)`-Aufrufe nutzen
reines `boolean`).

### T1b — Umbenennung `slice-literal.ts` → `literal-arithmetic.ts`

Die Datei enthält seit Spike 04 die gesamte Digit-Arithmetik (Subtraktion, Addition, Multiplikation
`LiteralShapeProduct`, Long Division `DivCeil`) + Literal-Klassifikatoren, nicht mehr nur
Slice-Logik. `git mv` (Historie erhalten, `similarity index 99%`), 9 brechende Import-Sites + 19
Kommentar-/Codegen-Referenzen aktualisiert, Dateikopf-Traceability-Vermerk (nennt den alten Namen).
Kein Import-Zyklus eingeführt (cycles-Gate zusammensetzungs-identisch).

## Gate-Ergebnisse (Haupt-Tree-Läufe + von A/B im sauberen Worktree bestätigt)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (root + stress + browser) | EXIT 0 |
| `pnpm test:core` | 818 pass, 0 fail |
| `pnpm test:resident` | 4278 pass, 2 skip (GC-Backstop), 0 fail |
| `pnpm demo` | grün, alle drei Backends bit-identisch |
| `cargo test` | 161 pass, 0 fail (ZERO Rust-Änderungen) |
| Artefakt-Hash | `0b9df4f1…519c7d` — **byte-identisch** zum Kern-11-Pin |
| `graph-a-lama query lint` (covenant-s1) | 0 errors, 0 warnings |
| `cycles --kind import` | keine neue SCC mit dim.ts/literal-arithmetic.ts |
| grep `slice-literal` in spike/ | genau 1 (der Traceability-Vermerk) |
| `bench:editor` | PASS, alle 7 Workloads, keine Hover-Regression |

**Pins (NEUER Baseline nach S1):** `check:diag` **179'986 @ 132** (Baseline 178'865; Δ+1'121 —
aber T1b ist eine Datei-UMBENENNUNG → Sort-Position-Änderung → Order-Noise bis ±~2'000; NICHT als
reine Typkost gegen 178'865 verrechenbar, neuer Baseline-Kandidat). `check:diag:stress` **102'877
@ 82** (Baseline 102'182; Δ+695, Dateizahl konstant, nicht bisektiert). `check:diag:browser`
**2'142 @ 75** (unverändert).

## Post-Verification-Addendum (drei unabhängige Fresh-Context-Verifier, 2026-07-17)

Volle A+B+C-Runde nach `docs/verify-runde-template.md`, alle in isolierten Worktrees (Patch von
b51c1a7).

- **Baustein A (Spec-Konformität, `brainroute:verify`): CONFIRMED.** Alle Binding-Entscheidungen
  T1a 1-6 / T1b 1-5 gegen den Diff geprüft, alle Gates frisch mit realen Zahlen (s.o.),
  Order-Noise-Vorbehalt korrekt angewandt. **Eigener Mutant:** `reduce.ts` `KeepDims` zurück auf
  `boolean` → `pnpm check` EXIT 1 (TS2344 an ndarray.ts:412 + resident.ts:805, an der
  Klassensignatur, beide Surfaces) → Coverage nicht-vakuös, robust gefangen. M3 über eigene Probe
  bytegenau bestätigt.
- **Baustein B (adversarial, `brainroute:deep`): HÄLT.** M3-Fehlermeldung über 10 Fälle geprüft
  (`sum(9/-9/5/2)` Rang 2, `sum(1)` Rang 1, `sum(0)` Rang 0, 2-Arg-Formen, WNDArray-Zwillinge, OOB
  + keepdims-Union gleichzeitig) — ALLE TS2345 mit wortgleichem Stamm AM Argument, nie generisches
  TS2769. keepdims-Union bei Rang 1/3/size-0/negativ-Achse: alle ehrlich. **Drei chirurgische
  Mutanten:** Constraint-Revert bricht am Deklarationsort; `AllOnes`-Zweig-Tuple-Wrap kippt genau
  `UA_KEEP_CLOSED_FULL`; `ApplyAt`-Zweig-Tuple-Wrap kippt 4 Pins (NDArray + WNDArray + bestehender
  dyn-boolean-Pin) — Pins strukturell scharf. Overload-Reihenfolge-Vertauschung: grün (Arity
  diskriminiert eindeutig, kein toter Pin).
- **Baustein C (covenant-verify): KEIN VERSTOSS.** S1, M1, M2, M3, M4, M5 halten inhaltlich, kein
  Nicht-Ziel berührt, Hash byte-identisch, M5-Importkanten unverändert (nur der Type-only-Import-Pfad
  umbenannt). Zwei **Vertragstext-Drift-Befunde** → in COVENANT v3 eingearbeitet (M2-Anker
  `slice-literal.ts`→`literal-arithmetic.ts`, M2-Notiz „offen"→„geschlossen").

### Adressierte Verify-Befunde

- **D1 (A, minor, Spec-Text):** Spec T1a-1 sagte irrig „Compile-Fehler" für die axis-Facette; die
  Impl macht Degradation zu `readonly number[]` (never-wrong Menü (i), der sauberere Ausgang) —
  Rest eines früheren Entwurfs. **KORRIGIERT** im Spec-Text.
- **D2 / 6a (A+B, nit, Spec-Text):** T1b-2 „grep == 0" vs. dem mandatierten 1-Treffer-Vermerk —
  Selbstwiderspruch der Spec. **KORRIGIERT** auf „== genau 1 (Traceability-Vermerk)".
- **C M2-Drift (mittel):** COVENANT v2 beschrieb den Verstoß als offen. **ADRESSIERT:** COVENANT
  v2→v3 (Owner-bestätigt), M2-Notiz „geschlossen", Anker umbenannt, Changelog.
- Keine der drei Befunde ist ein Code-Defekt; die Implementierung wurde in allen Fällen bestätigt.

### Prozessbefund: Worktree-Kontamination (Orchestrator-Fehler, Lektion)

Die drei Verifier-Aufträge gaben keinen pro-Agent-eindeutigen Worktree-Pfad vor → Baustein B und C
landeten im selben `worktrees/s1`, gegenseitige Kontamination (streu `_probe_m3.test-d.ts`,
`reduce.ts`-Reversion). **Neutralisiert:** A erkannte es selbst und wiederholte alles in einem
eindeutig benannten Worktree (`s1-verifyA-<ts>-<pid>`); B ist durch die EXAKTE Übereinstimmung seiner
Messzahlen mit A's sauberen Zahlen (179'986 / 102'877 / 818 / 4278 / 161 / Hash) kreuzvalidiert; C's
Verdikt ist diff-basiert (kontaminations-unabhängig). Kein Verifier-Ergebnis kompromittiert; der
Haupt-Tree blieb durchgehend unberührt (mehrfach per `git status` belegt). **Lektion (KB):**
parallele Verifier IMMER mit erzwungen eindeutigem Worktree-Pfad (PID/Timestamp) beauftragen.

## Nächste Schritte

Item 11 / S2 (Emit-/Paket-Pipeline: dist-Build, exports-Map, `.wasm`-Bundling, Post-Emit-Rewrite,
package.json-Metadaten) — eigene A+B+C-Runde. Danach S3 (Zero-dep-Guard + Paket-Smoke-Tests).
