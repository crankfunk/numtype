# Op-Scheibe W5: `item` — Ergebnisse

Status: **abgeschlossen** (Eskalationsleiter Stufe 3; Zahlen hier sind die im Haupt-Tree
tatsächlich gemessenen). Spec: docs/op-w5-item-spec.md (Version 2 + Baustein-0-Addendum F1-F8).
Datum: 2026-07-21.

## Summary

Die letzte Op-Scheibe aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md W5/F3 —
ein Skalar-Read aus der Score-Matrix) ist umgesetzt: `NDArray.item(...indices)`, NumPys direkter
Skalar-Accessor. D1: **NDArray-only, kein WASM-Kernel, kein Kernel überhaupt** (M1 v5,
kernel-los — reiner strided Read über `computeStrides`). Alle Gates grün, Hash byte-identisch
(keine Rust-Änderung). Absolut-Gate (≤ +6.000) mit **+5.873** knapp, aber sauber eingehalten —
der D6-Kostenmechanismus dahinter ist selbst ein Befund (siehe unten). Die Baustein-0-Skizze
(F1-F8) war vollständig kompilierbar und semantisch korrekt; die einzige substanzielle Abweichung
von der Erst-Umsetzung war eine nachträgliche BUDGET-getriebene Kürzung der Typ-Pins in
`ndarray.test-d.ts` (siehe „Abweichungen" unten) — keine Design-Korrektur.

## Umgesetzte Form je D-Punkt

- **D1 (Scope):** `item(...indices): number` auf `NDArray`, VOLLE Indizierung (ein Index je
  Achse). Rang 0: `item()` (null Argumente). Kein Partial-Indexing, kein Setter, kein `at`-Alias.
- **D2 (Typ-Ebene, verbindliche v2-Form aus dem Addendum):** `ItemGuard<S, Idx>` (vector.ts,
  APPENDED) DIREKT als Rest-Parameter-Typ (F1 — `Guard<>` kollabiert dort zu TS2370):
  - `ItemMark<D, I, Axis, S>` — pro Position, DREI Gates in fester Reihenfolge: `IsUnion<I>`
    zuerst (W4-Lektion + F6: `LiteralIndexBounds`s Union-Verhalten ist bereits konservativer als
    sein eigener Doc-Kommentar behauptet), dann `IsDotFormStep<\`${I}\`>` (F5 — NICHT in
    `LiteralIndexBounds` selbst abgedeckt, braucht den einen `export`-Zusatz in
    literal-arithmetic.ts), dann `LiteralIndexBounds<I, D>` = `"out"`.
  - `ItemFoldAcc<S, Idx, FullS, Acc>` — S-GETRIEBEN (F2): rekurriert über `S`, füllt bei
    erschöpftem `Idx` mit dem `number`-Marker weiter, sodass der deklarierte Rückgabe-Tupeltyp
    IMMER exakt `S["length"]` Elemente hat — Arity-Fehler (beide Richtungen) landen dadurch
    NATIV als TS2554 (F3), nicht als Custom-Message (es gibt architektonisch keine Position für
    ein fehlendes Argument).
  - `ItemGuard<S, Idx>` — zwei Wide-Type-Gates VOR der Rekursion: `RankUnknowable<S>` (dynamischer
    Rang / Mixed-Rank-Union) und `IsDynamicRank<Idx>` (F4-Fix — ohne dieses Gate bricht ein
    Spread-Aufruf `item(...arr)` mit TS2556, da der Rest-Parameter sonst zu einem festen Tupel
    gezwungen würde).
- **D3 (Runtime):** `itemRuntime(shape, data, indices)` APPENDED in runtime.ts — Arity-Check
  (eigener, runtime-only Stem, F3), pro Achse NumPy-Negativ-Normalisierung + Bounds-Check
  (Stems WORTGLEICH zu `ItemMark`s Type-Level-Stems, siehe unten), dann Offset-Summe über
  `computeStrides`. Size-0-Dims brauchen keinen Sonderfall (jeder Index ist automatisch OOB).
  NaN/±0 werden bit-exakt durchgereicht (direkter `Float64Array`-Read, keine Arithmetik).
- **D4 (Methode):** `item<const Idx extends readonly number[]>(...indices: ItemGuard<S, Idx>):
  number` als Klassenkörper-Append NACH `sqrt` (insertion-only, `NDArray`s Privat-Konstruktor-
  Klasse bleibt sonst unverändert).
- **D5 (Tests):** 16 neue Runtime-Tests in scalar-mean.test.ts (Rang 0-3, negative Parität, alle
  drei Stems direkt UND über die öffentliche API, size-0-OOB, NaN/-0-Bits, 200 randomisierte
  Flat-Index-Differential-Fälle über 5 Formen × 50 Trials, transponierter + geslicter Empfänger)
  + 14 Typ-Pins in ndarray.test-d.ts (10 `Expect<Equal<...>>` + 4 `@ts-expect-error`, s.u.).

## Stem-Wortlaute (M3: wortgleich, Typ vs. Runtime)

| Kante | Typ-Ebene (`ItemMark`, vector.ts) | Runtime (`itemRuntime`, runtime.ts) |
|---|---|---|
| Nicht-Integer | `item: index ${I} for axis ${Axis} is not an integer (shape ${ShowShape<S>})` | `item: index ${raw} for axis ${axis} is not an integer` |
| Out-of-Bounds | `item: index ${I} is out of bounds for axis ${Axis} with dim ${D} (shape ${ShowShape<S>})` | `item: index ${raw} is out of bounds for axis ${axis} with dim ${d}` |
| Arity (nur Runtime, F3) | — (nativ TS2554 am Call) | `item: expected ${shape.length} indices (got ${indices.length})` |

Der STEM (Präfix bis `dim ${D}` bzw. `an integer`) ist wortgleich zwischen Typ- und
Runtime-Nachricht; der Typ hängt zusätzlich `(shape [...])` für Editor-Kontext an — dieselbe
Stem-vs-Editor-Kontext-Konvention, die `slice.ts`s `IndexOutOfBoundsMessage`/`StepInvalidMessage`
relativ zu `normalizeAxisSpec`s eigenen Throws bereits etabliert hat (bewusste, konsistente
Design-Entscheidung, keine Spec-Vorgabe — D3 hatte das offen gelassen).

## Pin-Deltas (2× gemessen, deterministisch)

Baseline (frischer Worktree, `af6e7b0`): **195.481 @ 137 · 105.758 @ 82 · 2.142 @ 75**
(exakt reproduziert, siehe unten).

| Korpus | Baseline | Final (1.) | Final (2.) | Delta | Gate |
|---|---|---|---|---|---|
| `check:diag` (Haupt) | 195.481 @ 137 | 201.354 @ 137 | 201.354 @ 137 | **+5.873** | ≤ +6.000 → PASS |
| `check:diag:stress` | 105.758 @ 82 | 106.398 @ 82 | 106.398 @ 82 | +640 (ungated, attribuiert unten) | — |
| `check:diag:browser` | 2.142 @ 75 | 2.142 @ 75 | 2.142 @ 75 | +0 | — |

**Attribution (empirisch dekomponiert, Backup-Kopie/Bisektion, kein Mutant):**
- Nur `spike/src/*`-Änderungen (literal-arithmetic.ts Export + vector.ts/runtime.ts/ndarray.ts
  Appends): **+623** — nahe an der Baustein-0-Verifier-Messung (+712; Differenz durch die
  konkrete Reihenfolge der finalen Doc-Kommentare/Imports, order-noise-typisch).
- `spike/tests-runtime/scalar-mean.test.ts` (16 neue Tests, viele mit realen `item()`-Calls über
  WIDE, nicht-literale `number`-Argumente in Schleifen — jeder Call-Site zwingt `ItemGuard` zur
  Auswertung auch bei dynamischen Indizes): **+2.824**.
- `spike/tests/ndarray.test-d.ts` (14 Typ-Pins): **+2.426**.
- Stress-Korpus-Delta (+640) stammt AUSSCHLIESSLICH aus den geteilten `spike/src`-Änderungen
  (kein stress-eigenes File berührt) — plausibel etwas höher als der Haupt-Korpus-Src-Anteil
  (+623), da `tests-stress` andere, digit-lastigere Nachbar-Typen im selben Kompilierdurchlauf
  cached/lädt (Order-Noise, keine echte Mehrkosten-Quelle).

**D6-Befund (neu, nicht in der Baustein-0-Skizze sichtbar):** Eine EINZELNE
`Expect<Equal<ItemGuard<S, Idx>, HandGeschriebenerTyp>>`-Message-Equality-Pin gegen einen
strukturell ähnlichen (aber nicht identischen) Handtyp kostet in Isolation **≈1.700-1.750**
Instantiations — eine Größenordnung über einem bloßen Referenz-Zugriff auf `ItemGuard<...>`
(≈80) oder einem Self-Compare (`Equal<X, X>`, ≈100-110) oder einem Compare gegen `unknown`
(≈100, da `tsc`s Assignability-Check bei `unknown` nicht die volle Struktur der Quelle
normalisieren muss). Isoliert gemessen (Backup-Kopie-Bisektion, 5 identische `ItemGuard`-Pins vs.
1 kombinierter Pin): Ein einziger `ItemGuard`-basierter Pin, der ZWEI Fehler-Positionen in EINEM
Tupel gleichzeitig prüft (`ItemGuard<[2,3], [0.5, 3]>` — Achse 0 Dot-Form UND Achse 1
Out-of-Bounds in einem Call), bewies beide Stems für ≈1/5 der Kosten von fünf separaten
Ein-Kanten-Pins (5.020 → geschätzt ~1.700 für den kombinierten Pin). Diese Konsolidierungstechnik
(„ein Guard-Vergleich, mehrere Fehlerpositionen gleichzeitig, F7s Ein-Diagnose-pro-Call-Fakt
ausnutzend") ist der Grund, warum das finale `ndarray.test-d.ts` NUR EINE
`ItemGuard`-Message-Equality-Assertion trägt statt der ursprünglich geplanten fünf — siehe
„Abweichungen" unten.

## Gate-Tabelle

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Root + stress + browser, dreifach) | grün | 0 |
| `check:diag` 2× | 201.354 @ 137 beide Male | — |
| `check:diag:stress` | 106.398 @ 82 (ungated, attribuiert) | — |
| `check:diag:browser` | 2.142 @ 75 (unverändert) | — |
| `test:core` | **1.588/1.588** (Baseline 1.572 + 16 neue W5-Tests) | 0 |
| `test:resident` | 4.278/4.280 (2 skipped, Pin unverändert) | 0 |
| `cargo test` | 161/161 (Pin unverändert, keine Rust-Änderung) | 0 |
| `check:freeze` | Hash byte-identisch `0b9df4f1...` (Pin unverändert) | 0 |
| `bench:editor` 2× | Hard CI Gate: PASS beide Male, uniform **+628** über alle 7 Workloads (Pins aktualisiert) | 0 |
| `graph-a-lama query lint` | 0 errors, 0 warnings | 0 |
| `test:example` (RAG-Demo) | grün, alle 8 Queries + Pooling-Check bestehen | 0 |
| GFM (`~~`-Scan als `<del>`-Proxy) | 0 Treffer in Spec + diesem Doc | — |

## Abweichungen (offengelegt, keine stillen Änderungen)

1. **Typ-Pin-Anzahl in `ndarray.test-d.ts` budgetgetrieben reduziert.** Die Erst-Umsetzung folgte
   der Addendum-Skizze wörtlich (5 separate `ItemGuard`-Message-Equality-Pins: positive/negative
   OOB auf Achse 0, OOB auf Achse 1, Dot-Form, valider negativer Durchlass) und maß **+11.563**
   Gesamt-Delta — fast das Doppelte des ≤ +6.000-Gates. Bisektion (Backup-Kopie, keine
   Mutanten-Notwendigkeit — reine additive Entfernung) zeigte: der Quellcode selbst kostet nur
   +623; die Typ-Pin-Datei allein kostete +9.066 der übrigen +10.940. Der D6-Befund oben
   (`Equal`-Message-Vergleiche gegen `ItemGuard` sind pro Pin ≈1.700-1.750 teuer, nicht die paar
   Hundert, die andere Ops-Pins typischerweise kosten) wurde erst durch diese Messung entdeckt.
   **Reaktion:** Pins auf das Nötigste konsolidiert — EIN kombinierter Message-Equality-Pin
   (zwei Fehlerpositionen gleichzeitig) statt fünf, die Mixed-Rank-S-Kante über die bloße
   `ItemGuard`-Typebene statt über eine Klasseninstanzen-Union getestet (günstiger, gleiche
   Aussagekraft), ein `@ts-expect-error`-Real-Call-Reject-Trio statt vier. **Coverage-Auswirkung:**
   T3 („Typ-Pins decken jede D2-Kante ab") bleibt erfüllt — jede Kante (Arity beide Richtungen,
   OOB positiv, OOB negativ, Dot-Form, gültiges negatives Literal, wide Rang, Union-Index,
   Mixed-Rank-S, dynamischer Spread) hat mindestens einen Pin; nur die REDUNDANTEN
   Mehrfach-Belege pro Kante (z. B. OOB auf zwei verschiedenen Achsen separat UND nochmal per
   `@ts-expect-error`) wurden dedupliziert. Kein D2-Kanten-Verlust, nur Konsolidierung. Dies ist
   eine Abweichung von der WÖRTLICHEN Baustein-0-Skizze (die die genaue Pin-AUFTEILUNG nicht
   bindend spezifizierte, nur die abzudeckenden Kanten), keine Abweichung von D2/D5 selbst.
2. **Stem-Formulierung (D3s offene Frage) entschieden:** D3 ließ offen, ob der Runtime-Bounds-
   Stem die Achsen-Position tragen soll. Entscheidung: JA, symmetrisch zum Typ-Stem (siehe
   Tabelle oben) — dieselbe Stem-vs-Editor-Kontext-Konvention wie `slice.ts`/`normalizeAxisSpec`.
   Konsistent gepinnt, in beide Test-Ebenen eingearbeitet.
3. **`itemRuntime`s Arity-Stem-Wortlaut** (D3, „von dir festgelegt"): `item: expected ${R}
   indices (got ${M})` — gewählt in Analogie zu den bestehenden Slice-/Stack-Stems (Verb +
   Objekt + Klammer-Ist-Wert).

## Nächste Schritte

Kein Release in dieser Scheibe (Nicht-Ziel). Die Roadmap-Wunschliste aus
docs/dogfooding-rag-ergebnisse.md ist mit W5 vollständig abgearbeitet (W1-W5). FOLLOWUPS trägt
zwei neue Einträge: `item`-WASM-Parität (M1-v5-Disclosure) und das Aufsplitten von
scalar-mean.test.ts (D6-Mandat).

## Post-Verification-Addendum (2026-07-21)

Verify-Runde Stufe 3 (kompakt), A+B+C parallel + covenant-check/lint (grün).
Baustein 0 hatte den Guard-auf-Rest-Param-Blocker (TS2370) vor dem Bau gefangen und
die verbindliche ItemGuard-Form geliefert.

- **Baustein A: CONFIRMED.** Alle Kanten nach der Budget-Konsolidierung selbst
  enumeriert — keine verloren; beide Mutanten beißen (Negativ-Normalisierung: 3 Tests
  rot; OOB-Pin-Flip: TS2578); alle Zahlen reproduziert. Korrigiert nach A-Befunden:
  Pin-Aufschlüsselung war falsch angegeben (tatsächlich 6 Equal + 6 @ts-expect-error
  + 2 Must-Compile; nach der Schließungsrunde +1 Policy-Pin), GFM-Scope-Doku ergänzt
  (A prüfte selbst alle 7 .md — 0 Treffer), irreführender Fast-Path-Testkommentar
  korrigiert (es existiert kein Fast-Path-Split; transpose() materialisiert).
- **Baustein B: „teilweise widerlegt" — Implementierung hielt, zwei META-Behauptungen
  fielen:** (1) IsUnion-Pre-Gate war coverage-tot (Suite grün ohne ihn); (2) die
  F6-Prämisse war faktisch falsch (LiteralIndexBounds<5|9,3> = „out" — Spike-03-
  Union-Disziplin trägt). Auflösung: Pre-Gate bleibt als bewusste POLICY-Angleichung
  an reduce.ts (ALL-invalide Unions degradieren uniform; die vollere Ablehnung wäre
  ebenfalls sound — dokumentierte Vollständigkeits-Abwägung), neuer Policy-Pin
  macht ihn load-bearing (Mutant: exakt 1 Zeile rot). Gehalten: Flat-Offset-
  Differential, -d-Grenzkante (NumPy-Parität beidseitig), beide Runtime-Mutanten
  (3 bzw. 4 Tests rot), TS2554-Kanten, F7-Ein-Diagnose-Reproduktion,
  Union-no-claim-Verhalten empirisch.
- **Baustein C: ein niedriger Befund — die M3-Spannung ist ein WIEDERHOLUNGSMUSTER**
  (W4 Compile-only-Stem, W5 TS2554-Arity): Praxis weicht disclosed vom wörtlichen
  „Stems wortgleich" ab, der Vertragstext kennt die Ausnahme-Klasse nicht →
  als M3-Präzisierungs-Kandidat für Covenant v6 in FOLLOWUPS (bündelbar mit dem
  M2-Scope-Kandidaten aus W2). Alles andere hält (M2-Ketten, Export-Neutralität,
  M1-v5-Bedingung).
- **Finale Zahlen (nach Schließungsrunde, je 2×):** check:diag **201,455 @ 137**
  (+5,974 zur Baseline 195,481 — Gate ≤ +6,000 mit 26 Instantiations Rest-Puffer;
  der Policy-Pin kostete +101) · stress 106,398 @ 82 · browser 2,142 @ 75 ·
  test:core 1,588 · Hash byte-identisch. Gate-Nähe als Beobachtung notiert:
  künftige test-d-Ergänzungen zu item sollten die ≈1,700-pro-ItemGuard-Equal-Pin-
  Kostenklasse kennen (FOLLOWUPS-Messbefund).
