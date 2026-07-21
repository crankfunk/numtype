# Op-Scheibe W4: `stack` — Ergebnisse

Status: **abgeschlossen NACH Verify-Runde-Fix** (Eskalationsleiter Stufe 3; Zahlen hier sind die
im Haupt-Tree tatsächlich gemessenen, kein Vorab-Stand — inkl. des unten dokumentierten
Baustein-B-Fixes). Spec: docs/op-w4-stack-spec.md (Version 2 + Baustein-0-Addendum F1-F8).
Datum: 2026-07-21.

## Summary

Die vierte Op-Scheibe aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md
W4/F5 — der selbstgebaute `embedMatrix`-Zeilen-Flatten-Helper in
`examples/rag-demo/embedding.ts`) ist umgesetzt: `NDArray.stack(rows)` — eine Matrix aus N
unabhängig berechneten Rang-1-Zeilenvektoren, `np.stack`/`np.array([...])`-Affinität. D1:
**NDArray-only, kein WASM-Kernel** — dieselbe bewusste, COVENANT-v5-gedeckte Surface-Asymmetrie
wie W1/W2/W3. Alle Gates grün, Hash byte-identisch (keine Rust-Änderung), Absolut-Gate
(Haupt-Pin ≤ +8.000) mit deutlichem Spielraum (+4.841 nach dem Verify-Fix). Die F5-Schließung
ist bewiesen: `stack` reproduziert `embedMatrix`s eigenen row-major `Float64Array#set`-bei-
Offset-Algorithmus byte-identisch. ZWEI echte Typ-Fehler wurden gefangen und behoben — einer
während der eigenen Umsetzungs-Verifikation (siehe „Baustein-0-Fund während der Umsetzung"
unten), einer während der Verify-Runde selbst (Baustein B, BLOCKER-Klasse, siehe „F-ADV-1"
unten) — die Erst-Skizze aus dem Addendum hatte beide Kanten nicht abgedeckt.

## Umgesetzte Form je D-Punkt

- **D1 (Scope & Name):** Statische Methode `NDArray.stack(rows)`, NUR Rang-1-Zeilen gleicher
  Länge → Rang-2-Ergebnis `[N, D]`. Kein allgemeines Achsen-stack, kein `axis`-Parameter, kein
  `concat`/`vstack`/`hstack` (Nicht-Ziele). `fromRows` nur als Doc-Erwähnung.
- **D2 (Typ-Ebene, Schichtung F1):** `StackCheck<Shapes>` + `StackShape<Shapes>` APPENDED in
  vector.ts, operieren auf `readonly Shape[]` (nie `NDArray` — Zyklus-Vermeidung). Enthält:
  - `StackFold` (Tupel-Pfad): Tail-rekursiver Head/Rest-Fold mit Index-Tracking via
    Tuple-Length (`Seen["length"]`), Akkumulator `Dim | "none"` mit dem F4-Tupel-Wrapped-Gate
    (`[Acc] extends ["none"]` + `Extract<Acc, Dim>`).
  - `StackDimMerge`: Wide-Sentinel-Merge (F6, CompatDim-Präzedenz ohne Broadcast-1-Sonderfall)
    — ein dynamischer/Union-Dim auf IRGENDEINER Seite weitet monoton auf `number`, sonst
    `DimEq` (nach Filtern — Baustein-0-Fund 9 bestätigt).
  - `StackCheckArray`/`StackShapeArray` (Array-Pfad, F5): non-distributiver Rang-Check gegen
    `Shapes[number]` als Ganzes (F7: uniform falscher Rang wird abgelehnt, da auch das leere
    Array wirft), `ArrayRowD` (eigene naked-Generic für erzwungene Distribution, F8) +
    `IsUnion`-Filter für Union-Element-Typen → `readonly [number, number]`.
  - Drei gepinnte Message-Template-Typen (`StackEmptyMessage`/`StackRankMessage`/
    `StackLengthMismatchMessage`), wortgleich zur Runtime.
- **D2 Forts. (Homomorpher Unwrap, F2):** `RowShapesOf<Rows>` in ndarray.ts — EIN benannter
  homomorpher Mapped Type (`{ [I in keyof Rows]: UnwrapRow<Rows[I]> }`), NICHT die naive
  `Rows[number] extends NDArray<infer S> ? S : never`-Form (kollabiert zu `never` auf
  heterogenen Tupeln wegen `NDArray`s Invarianz-Marker). Faktorisiert in EINEN Typ statt
  zweimal inline (Baustein-0-Messung: halbiert die Kosten ≈1.428→801 in der Skizze).
- **D3 (Runtime):** `stackRuntime(rows: ReadonlyArray<{shape, data}>)` APPENDED in runtime.ts —
  EIN Links-nach-rechts-Durchlauf (Rang vor Länge PRO Zeile, dieselbe Reihenfolge wie der
  Typ-Fold), drei gepinnte Stems, dann `Float64Array#set`-Zeilenkopie (exakt `embedMatrix`s
  Algorithmus). NaN-Bits byte-erhalten (reiner Typed-Array-Copy). D=0 valide.
- **D4 (Methode):** `static stack<const Rows extends readonly NDArray<any>[]>(rows: Guard<...>):
  NDArray<OkShape<...>>` als Insertion NACH `fromArray` im statischen Block (Baustein-0-
  Empfehlung übernommen — stack ist konzeptionell ein Konstruktor, liest nie `this`).
  `const Rows` erhält Tupel-Literale ohne `as const` (empirisch gepinnt).
- **D5 (Tests):** 8 neue Runtime-Tests (Stem-Pins über `stackRuntime` UND die öffentliche API
  via dynamischer-Rang-Zeilen — dieselbe "widen-past-the-guard"-Technik wie `mean(5)`s
  Achsen-Pin; 1/2/3-Zeilen; D=0; NaN-Payload-Bits via `bitsOf`; F5-Rückprobe; Large-N-Smoke
  5000×8; Aliasing-Isolation) + **22 Typ-Pins in ndarray.test-d.ts** (18 benannte `Expect<
  Equal<...>>` + 4 `@ts-expect-error`; korrigierte Zählung nach F-ADV-3, s.u. — ursprünglich
  16 aus der Erst-Umsetzung, +6 aus dem Verify-Runde-Fix unten).
- **D6 (Gates & Pins):** siehe Tabellen unten.

## F5-Schließungs-Beweis (byte-identisch)

`embedMatrix`s eigener Algorithmus (`examples/rag-demo/embedding.ts`: `Float64Array#set` an
`row * dims`) wurde LOKAL nachgebaut (nicht importiert — das Beispielpaket ist bewusst
außerhalb des `spike/`-Kompilationsgraphen, ein Import hätte `check:diag`s Dateizahl und damit
den Pin kontaminiert). 7 randomisierte Zeilen à 12 Dimensionen: `rebuiltEmbedMatrix(...)` vs.
`NDArray.stack(rows).data` — `assertShapeEqual` UND `assertDataBitIdentical` — **PASS**. Damit
ist die in der Spec benannte Friction (F5, docs/dogfooding-rag-ergebnisse.md) strukturell
geschlossen: derselbe Zeilen-Kopier-Algorithmus, jetzt hinter einer typsicheren `NDArray`-API.

## Baustein-0-Fund während der Umsetzung (real, nicht hypothetisch)

Ein Scratch-Probe (isolierter `tsc`-Lauf gegen einen Symlink auf `spike/src`, außerhalb des
Repos) deckte auf: `RowShapesOf<Rows>`s naiv inline geschriebener homomorpher Mapped Type
(`Rows[I] extends NDArray<infer S> ? S : never`) liefert für ein ARRAY mit
UNION-Element-Typ (`readonly (NDArray<[3]>|NDArray<[4]>)[]`, der F8-Testfall) `readonly
[number, never]` statt der erwarteten `readonly [number, number]` — derselbe
Invarianz-Kollaps-Mechanismus wie F2, aber INNERHALB der homomorphen Mapped-Type-eigenen
Element-Auswertung für Arrays (nicht bei Tupel-Positionen, wo F2s Fix bereits greift). Ursache:
für ein Array wertet TS den Element-Typ-Ausdruck EINMAL gegen den (hier: Union-)Elementtyp aus,
non-distributiv, da `Rows[I]` an dieser Stelle ein Indexed-Access-Typ ist, kein naked
Type-Parameter. Fix: `UnwrapRow<R>` als EIGENE Generic mit eigenem naked Parameter (derselbe
"extra Generic erzwingt Distribution"-Kunstgriff wie `ArrayRowD` in vector.ts), dann
`RowShapesOf<Rows> = { [I in keyof Rows]: UnwrapRow<Rows[I]> }`. Nach dem Fix liefert der Probe
`readonly [number, number]` — verifiziert, siehe Typ-Pin `STACK_ARRAY_UNION` in
ndarray.test-d.ts. Dies ist der Grund, warum `RowShapesOf` in der finalen Form eine
Zwei-Typen-Kaskade ist (`UnwrapRow` + `RowShapesOf`), nicht der ursprünglich in der Spec
skizzierte Ein-Typ-Ausdruck — eine LEGITIME, während der eigenen Verifikation gefundene und
geschlossene Kante, kein Abweichen von der Spec-Absicht (die Spec verlangt explizit "nie
confidently wrong"; der Bug hätte STILL eine falsche — `never` statt `number` — Kante erzeugt).

## F-ADV-1: Verify-Runde-Fund (Baustein B, BLOCKER-Klasse, M2-Verstoß) — gefixt

Die Verify-Runde (Baustein B) fand einen ZWEITEN, unabhängigen echten Bug, den weder Baustein 0
noch die eigene Umsetzungs-Verifikation gefangen hatten: `NDArray.stack([fixed, row])` mit
`row: NDArray<[3]> | NDArray<[4]>` — eine GEWÖHNLICHE Union über einen Ternary, keine
`stack`-spezifische Konstruktion — kompilierte OHNE Fehler mit dem KONFIDENTEN Ergebnis
`readonly [2, 3]`, warf aber zur Laufzeit `stack: row length mismatch`. Ein echter M2-Verstoß
(confidently wrong).

**Root Cause** (von B nachverfolgt, hier reproduziert): An der Position `Rows[I] extends
NDArray<infer S> ? S : never` in `UnwrapRow` ist `R` (bzw. das originale `Rows[I]`-Argument, an
UnwrapRow übergeben) ein NAKED type parameter INNERHALB von UnwrapRows eigenem Rumpf — anders
als der F2/F8-Indexed-Access-Fall, der genau DIESE Distribution braucht (F8s Array-Element-
Union), distribuiert diese Union HIER auch für eine TUPEL-Position, deren eigener Typ zufällig
eine Union ist. Der betroffene Tupel-Slot wird `readonly [3] | readonly [4]`. In `StackFold`
distribuiert `Head extends readonly [infer D]` WEITER darüber (`Head` bleibt ein naked
Typ-Variable über die gesamte Rekursionskette hinweg) — der Fold GABELT SICH in zwei parallele
Fortsetzungen (eine mit D=3, eine mit D=4), die je nach übrigem Zeilen-Set UNTERSCHIEDLICH
verifizieren können. Endergebnis: `StackFold`s Gesamtresultat wird eine GEMISCHTE Union
`3 | ShapeError<...>`. Zwei nachgelagerte Mechanismen verschleiern das dann: (1) `Guard`s
Tupel-Wrapped-Check `[Result] extends [ShapeError<...>]` lehnt nur UNIFORME Error-Unions ab —
eine gemischte passiert durch; (2) `StackShape`s `Extract<StackFold<...>, Dim>` wirft den
`ShapeError`-Zweig still weg und behält das (potenziell falsche) Literal. Bei symmetrisch
vertauschter Zeilen-Reihenfolge identisch; bei beidseitigem Mismatch entsteht zusätzlich eine
UNION zweier verschiedener Fehler-Stems statt eines sauberen.

**Fix** (Hauspolitik D-V1.3/`ReduceAxis`-Präzedenz, in `StackFold`, vector.ts): ein
`IsUnion<Head>`-Gate DIREKT NACH dem `RankUnknowable<Head>`-Gate und VOR dem naked
`Head extends readonly [infer D]`-Match. Position ist load-bearing — identisch zur bereits
dokumentierten `ReduceAxis`-Lektion (reduce.ts, Union-Axis-Mini-Scheibe D-A.2): sobald die
Ausführung den naked Check erreicht, ist eine Union bereits member-für-member distribuiert; ein
Filter danach sähe nur noch einzelne Mitglieder, nie die Union als Ganzes. Ein Union-Head weitet
jetzt UNIFORM auf den wide-Sentinel `number` — genau wie der `RankUnknowable`-Zweig — statt zu
distribuieren; NIE ein `ShapeError`, selbst im Doppel-Mismatch-Fall (bewusste, einfachere
Degradation statt eines Beweises "jedes Union-Mitglied schlägt fehl" — der Laufzeit-Backstop
bleibt maßgeblich). `UnwrapRow`s erzwungene Distribution selbst bleibt UNVERÄNDERT (für F8s
Array-Pfad weiterhin nötig und dort bereits sicher, da der Array-Pfad komplett getrennt über
`StackCheckArray`/`StackShapeArray` läuft, niemals durch `StackFold`).

**Sechs neue Pins** (ndarray.test-d.ts): `STACK_UNION_ROW_REPRO`/`_REV` (Bs exakter Repro, beide
Zeilen-Reihenfolgen, → `readonly [2, number]`), `STACK_UNION_ROW_DOUBLE_MISMATCH` (beidseitiger
Mismatch → `readonly [2, number]`, keine Stem-Union), `STACK_DIM_MERGE_WIDE`/`_REV`
(F-ADV-2-Schließung: direkte `StackDimMerge`-Wide-Abdeckung über `NDArray<[number]>`, nicht nur
indirekt über `RankUnknowable`), `STACK_ARRAY_MIXED_RANK_UNION` (Verify-C-Lücke: Array-Element-
Union verschiedener Ränge, EMPIRISCH per Scratch-Probe geprüft — `RankUnknowable` feuert bereits
auf `Shapes[number]["length"]` = `1 | 2`, Ergebnis `readonly [number, number]`, bestätigt Cs
Typraum-Vorhersage).

**Mutations-Selbstprobe** (Backup-Kopie-Verfahren, NIE `git checkout`): `vector.ts` nach
`/tmp` kopiert (MD5 vorher `4aef7bf7…`), den neuen `IsUnion<Head>`-Zweig testweise entfernt →
`pnpm tsc --noEmit` schlägt mit GENAU 4 Fehlern fehl (Exit 1): `STACK_UNION_ROW_REPRO` UND
`_REV` werden `false`, der `stackDoubleMismatch`-Aufruf selbst wird abgelehnt (TS2741, fehlendes
`__shapeError`), UND `STACK_UNION_ROW_DOUBLE_MISMATCH`s `Equal`-Check wird `false` — die neuen
Pins sind nicht-vakuös, kein anderer Pin ist betroffen. Restore aus der Backup-Kopie, `diff`
gegen die Kopie **identisch** (0 Byte Unterschied), MD5 nachher wieder `4aef7bf7…` — byte-exakte
Wiederherstellung bewiesen, danach `pnpm tsc --noEmit`/`pnpm check`/`test:core` erneut grün.

**F-ADV-3 (Pin-Zählung korrigiert):** die ursprüngliche Zählung in diesem Dokument ("19 neue
Typ-Pins") war falsch — die tatsächliche Erst-Umsetzung hatte **16** (12 benannte `Expect<
Equal<...>>` + 4 `@ts-expect-error`, nachgezählt via `grep`). Mit den sechs neuen Verify-Runde-
Pins oben: **22 gesamt (18 benannte + 4 `@ts-expect-error`)**.

## Pin-Protokoll (D6, 2× deterministisch je Messpunkt)

Baseline (frischer `git worktree` von HEAD `cc1443b`, `pnpm install --frozen-lockfile`, je 2×
gemessen) **exakt reproduziert**: `190.640 @ 137` (Haupt) / `104.900 @ 82` (stress) /
`2.142 @ 75` (browser) — deckungsgleich mit dem CLAUDE.md-Pin.

| Messpunkt | Instantiations | Δ zur Baseline |
|---|---|---|
| Baseline root (frischer Worktree, 2×) | 190.640 | — |
| Final root VOR Verify-Fix (Haupt-Tree, 2×) | 194.545 | +3.905 |
| **Final root NACH Verify-Fix (Haupt-Tree, 2×)** | **195.481** | **+4.841** |
| Baseline stress (frischer Worktree, 2×) | 104.900 | — |
| Final stress VOR Verify-Fix (Haupt-Tree, 2×) | 105.752 | +852 |
| **Final stress NACH Verify-Fix (Haupt-Tree, 2×)** | **105.758** | **+858** |
| Baseline browser (frischer Worktree, 2×) | 2.142 | — |
| **Final browser (Haupt-Tree, 2×, unverändert durch den Fix)** | **2.142** | **±0** |

Absolut-Gate (≤ +8.000) klar eingehalten (+4.841 = ≈61 % des Budgets, immer noch deutlicher
Spielraum). Kein neues File (D2-D5 sind reine Appends/Insertions in bestehende, bereits
registrierte Dateien) → kein empty-then-fill-Schritt nötig, keine Order-Noise-Komponente.
**Stress-Delta-Attribution:** +858 stress vs. +4.841 root — derselbe "Klassen-Fläche wächst,
jeder `NDArray<S>`-Verweis im Korpus kostet marginal mehr"-Mechanismus wie bei W1s argmax/topk
(neuer STATISCHER Member rippelt über JEDE `NDArray`-Instantiierung im jeweiligen Korpus, nicht
nur an der Aufrufstelle) — die Skizzen-Schätzung aus dem Addendum (+801 Root/+1.441 stress) lag
in der Größenordnung richtig, die tatsächliche Vollimplementierung (mit allen F1-F8-Kanten,
Message-Templates, Doc-Kommentaren — Letztere kosten nichts, aber die zusätzliche
Typ-Maschinerie schon, PLUS dem Verify-Runde-Fix: ein zusätzlicher `IsUnion<Head>`-Zweig in
`StackFold`) liegt etwas höher am Root und niedriger am stress-Korpus als die Skizze; alle Werte
werden hier als tatsächlich gemessen berichtet, nicht die Skizzen-Schätzung. Der Verify-Fix
selbst kostete +936 Root/+6 stress (194.545→195.481 / 105.752→105.758) — plausibel, da er sowohl
neue Typ-Maschinerie (den `IsUnion`-Zweig) als auch sechs neue Typ-Pins hinzufügt.

## Gates (alle frisch gemessen, Haupt-Tree, dieser Stand)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (3-Verbund: root + stress + browser) | 0 Fehler | 0 |
| `check:diag` | **195.481 @ 137** (2× deterministisch, NACH Verify-Fix) | 0 |
| `check:diag:stress` | **105.758 @ 82** (2× deterministisch, NACH Verify-Fix) | 0 |
| `check:diag:browser` | 2.142 @ 75 (2× deterministisch, Δ0, unverändert durch den Fix) | 0 |
| `test:core` | **1.572/1.572** (Baseline 1.564 + 8 neu — Verify-Fix ist rein typseitig, keine neuen Runtime-Tests) | 0 |
| `test:resident` | 4.278 pass + 2 skip (unverändert) | 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161 passed (unverändert, kein Rust berührt) | 0 |
| `check:freeze` | Hash byte-identisch `0b9df4f1…2519c7d` (auch nach dem Fix) | 0 |
| `bench:editor` | Hard-CI-Gate PASS NACH ZWEI Pin-Updates (s.u.) | 0 |
| `graph-a-lama query lint` | 0 Findings, 0 Errors/Warnings (auch nach dem Fix, Graph neu gebaut: 181 Dateien/1.264 Symbole) | 0 |
| `pnpm test:example` | PASS, numtype@0.1.1 (Registry-Major.Minor-Match), Demo-Assertions grün | 0 |
| `pnpm test:package` | PASS, 3 Runtime-Tests + Konsumenten-Typ-Smoke | 0 |

**`bench:editor`-Pin-Abweichung Nr. 1 (Erst-Umsetzung, 2× gemessen, byte-identisch):** UNIFORM
**+845** auf ALLEN sieben Workloads (w1-w7) — derselbe Klassen-Fläche-Ripple-Mechanismus wie W1s
argmax/topk (ein neuer, generischer statischer Member vergrößert JEDES `NDArray<S>`-Vorkommen im
LSP-Harness-Korpus geringfügig). Anders als W2s Overload-Reorder-Fund differenziert diese
Verschiebung NICHT zwischen der absichtlich-fehlerhaften w4-Workload und den übrigen —
`stack` fügt keinem der Overloads hinzu, gegen die w4s zwei absichtliche Fehler auflösen.

**`bench:editor`-Pin-Abweichung Nr. 2 (Verify-Runde-Fix, 2× gemessen, byte-identisch):** ein
zusätzlicher UNIFORMER **+6** auf allen sieben Workloads — der neue `IsUnion<Head>`-Zweig in
`StackFold` ist minimale zusätzliche Typ-Maschinerie (eine Konditional-Verzweigung), der bisher
kleinste Ripple einer Op-Scheiben-Nachmessung. `INSTANTIATION_PINS` in
`spike/bench-dx/editor-latency.ts` zweimal aktualisiert:

| Workload | Vor W4 (post-W2-Verify-B) | Nach Erst-Umsetzung | Nach Verify-Fix (final) |
|---|---|---|---|
| w1 | 26.290 | 27.135 | **27.141** |
| w2 | 28.099 | 28.944 | **28.950** |
| w3 | 59.239 | 60.084 | **60.090** |
| w4 | 26.453 | 27.298 | **27.304** |
| w5 | 31.744 | 32.589 | **32.595** |
| w6 | 32.914 | 33.759 | **33.765** |
| w7 | 25.462 | 26.307 | **26.313** |

## Tests im Detail

**Runtime (`spike/tests-runtime/scalar-mean.test.ts`, W4-Block, 8 neue Tests):** kein
WASM-Gegenpart (D1, wie W1/W2/W3). Zwei Stem-Pin-Tests (`stackRuntime` direkt UND über die
öffentliche API via dynamischer-Rang-Zeilen — die `mean(5)`-Technik, kein unsicherer Cast
nötig); 1/2/3-Zeilen-Erfolgsfälle; D=0; ein Byte-exakter NaN-Payload-Test (`bitsOf`, nicht
`Object.is` — mirroring special-values.test.ts's Transpose-Fixture, stack ist auch ein reiner
Movement-Op); die F5-Rückprobe (rebuiltEmbedMatrix, s.o.); Large-N-Smoke (5.000 Zeilen à 8,
Stichproben an erster/mittlerer/letzter Zeile); Aliasing-Isolation (frischer Buffer, Zeilen
unverändert, Ergebnis-Mutation schlägt nicht auf Eingaben durch — die W3-Lektion).

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, +22 Pins gesamt — 16 aus der Erst-Umsetzung + 6 aus
dem Verify-Runde-Fix, F-ADV-3 korrigiert):** exakte `[2,3]`/`[3,4]`-Formen; Längen-Mismatch als
`@ts-expect-error` AM Argument + Message-Equality-Pin; Rang≠1-Tupel-Member als `@ts-expect-error`
+ Message-Pin; leeres Tupel-Literal als `@ts-expect-error` + Message-Pin (F3); Array-Input →
`[number, 3]` (F5); Array mit dynamischem Dim → `[number, number]` (F6); Array mit uniform
falschem literalen Rang als `@ts-expect-error` (F7); Array mit Union-Element-Typ →
`[number, number]` (F8, fing den Baustein-0-Fund oben); RankUnknowable-Member sowohl als
Tupel-Position (`[2, number]` — N bleibt literal, D weitet) als auch als Array-Element
(`[number, number]`) — nie confidently wrong; `const`-Tupel-Inferenz ohne `as const`
(Positiv-Pin, inline-Literal); D=0-Formen. Aus dem Verify-Runde-Fix (F-ADV-1, s.o.): Bs exakter
Union-Row-Repro in beiden Zeilen-Reihenfolgen, der Doppel-Mismatch-Union-Fall, direkte
`StackDimMerge`-Wide-Abdeckung (F-ADV-2-Schließung, beide Reihenfolgen), Array-Element-Union
verschiedener Ränge (Verify-C-Lücke, empirisch geprüft).

## Abweichungen

Der real gemessene Root-Pin-Delta (+4.841 final) und der Stress-Pin-Delta (+858 final) weichen
von der Baustein-0-Skizzen-Schätzung (+801/+1.441) ab — beide Richtungen (Root höher, stress
niedriger als geschätzt). Kein Verstoß gegen das Absolut-Gate (≤ +8.000, hier +4.841), keine
Owner-Eskalation nötig (die Spec selbst sagt "Messung entscheidet", D6). Ursache plausibel: die
volle Implementierung deckt alle acht F1-F8-Kanten samt drei Message-Template-Typen ab, während
die Baustein-0-Skizze eine schlankere Machbarkeitsprobe war; der Verify-Runde-Fix trägt weitere
+936 Root/+6 stress bei.

ZWEI während der eigenen Verifikation (nicht vom Owner, nicht ungeprüft aus der Spec
übernommen) gefundene und geschlossene Typ-Bugs — kein Abweichen von der Spec-Absicht, sondern
deren korrekte Erfüllung (M2: nie confidently wrong): (1) `RowShapesOf`s Array-Union-Kollaps
(während der eigenen Umsetzungs-Verifikation gefunden, s.o.); (2) `StackFold`s
Tupel-Positions-Union-Distribution (F-ADV-1, von der Verify-Runde/Baustein B gefunden — dieser
zweite Bug entging der eigenen Verifikation, da die dort verwendeten Testfälle nie einen
GEWÖHNLICHEN Ternary/Union-typisierten Zeilen-Wert probierten, nur explizit konstruierte
`declare const`-Unions für den Array-Pfad).

## Post-Verification-Addendum

_(vom Orchestrator nach der Verify-Runde zu ergänzen.)_

## Post-Verification-Addendum (2026-07-21)

Verify-Runde Stufe 3, A+B+C parallel + covenant-check/lint (grün). Baustein 0 hatte
vorab mit kompilierender Skizze drei HOCH-Befunde in verbindliche Formen überführt
(F1 Schichtung, F2 Invarianz-Kollaps → homomorpher Mapped Type, F5 Array≠Tupel).

- **Baustein A: CONFIRMED.** Alle D/T/F-Punkte am Diff; beide Pflicht-Mutanten beißen
  (Offset: 6/8 rot; Tupel-Homogenisierung: TS2578 exakt); den selbstgefixten F8-Bug
  UNABHÄNGIG in-repo reproduziert — nachdem eine Standalone-tsconfig-Probe das
  Gegenteil suggeriert hatte (dokumentierte Verifier-Warnung: TS-Verhaltensproben
  außerhalb der exakten Projekt-tsconfig sind für distributive Conditional-Fragen
  unzuverlässig; in-repo per Backup-Mutant verifizieren). Zwei Nits akzeptiert
  (StackRankMessageArray als offengelegte Compile-only-Stem-Ausnahme; GFM-Gate ist
  manuelle Konvention).
- **Baustein B: fand einen ECHTEN BLOCKER (F-ADV-1, M2-Verstoß) — in-Slice gefixt
  und von B re-verifiziert (FIX BESTÄTIGT).** Union-Row-Typ aus gewöhnlichem
  Branching (`cond ? zeros([4]) : zeros([3])`) kompilierte mit konfidentem
  [2,3]-Claim und warf zur Laufzeit: UnwrapRows erzwungene Distribution (für F8
  nötig) distribuierte AUCH an Tupel-Positionen; der Fold forkte zu
  `Dim | ShapeError`, Guards uniform-only-Tuple-Wrap ließ die Misch-Union passieren,
  Extract verschluckte den Error-Zweig. Die Baustein-0-Scope-Annahme („distribuiert
  natürlich, out of scope") war falsch. Fix: `IsUnion<Head>`-Gate in StackFold VOR
  dem naked Destructure (RankUnknowable-Nachbarposition, reduce.ts-Präzedenz) →
  Union-Slot degradiert uniform zu wide, auch im Doppel-Mismatch (Runtime-Backstop
  maßgeblich; löst auch die sekundäre Stem-Union-Kante). 6 neue Pins (Repro beide
  Reihenfolgen, Doppel-Mismatch, direkte StackDimMerge-wide-Pins = F-ADV-2-Schließung,
  Array-Mixed-Rank-Union = Verify-C-Kante); Nicht-Vakuität per Gate-Entfernungs-Mutant
  (exakt 4 Pins rot, MD5-byte-exakte Wiederherstellung); Bs Original-Mutant wird jetzt
  von 3 Pins DIREKT gefangen. B-Re-Check: Typ-Hover ehrlich [2, number] (Kontroll-
  @ts-expect-error auf den alten Literal-Claim feuert), Runtime wirft in allen vier
  Szenarien korrekt. F-ADV-3 (Pin-Zählung) korrigiert: final 22 (18 benannte + 4
  @ts-expect-error).
- **Baustein C: keine Verstöße.** M2-Ketten am Diff bis zum Ende verfolgt (inkl.
  Zwei-Durchgänge-Distributionsanalyse); M3-Stems zeichenweise; M1-v5-Bedingung
  erfüllt; die von C ehrlich als unverifiziert gemeldete Kante (Array-Element-Union
  verschiedener Ränge) wurde in der Fix-Runde empirisch bestätigt und gepinnt.
- **Finale Zahlen (nach Fix, je 2×):** check:diag **195,481 @ 137** (+4,841 zur
  Baseline 190,640, Gate ≤ +8,000) · stress **105,758 @ 82** (+858, Static-Member-
  Ripple attribuiert) · browser 2,142 @ 75 · test:core 1,572 · bench:editor PASS
  (zweite Pin-Anpassung +6 uniform) · Hash byte-identisch.
