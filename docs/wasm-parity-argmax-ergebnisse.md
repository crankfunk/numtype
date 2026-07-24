# WASM-Parität S4: `argmax` — Ergebnisse

Scheibe: WASM-Parität S4 (vierte der Kampagne S0–S5) · Datum: 2026-07-24 ·
Bindende Spec: [wasm-parity-argmax-spec.md](wasm-parity-argmax-spec.md) **v3** ·
Eskalationsleiter: Stufe 3 (Baustein 0 + Verify-Katalog A+B+C).

**Ehrlichkeitsregel dieser Dokumente:** jede Zahl stammt aus einem tatsächlich
ausgeführten Kommando; nicht Ausgeführtes wird als nicht ausgeführt benannt, nicht
weggelassen. Verdikte, die sich als falsch erwiesen, bleiben mit ihrer Korrektur stehen.

## Ergebnis in einem Satz

`WNDArray.argmax` (und damit der threaded-Pfad) existiert, bit-identisch zur TS-Referenz
`argmaxRuntime` bewiesen, über einen neuen Rust/WASM-Kernel — der erste echte neue Kernel
seit S1; der Freeze-Hash bewegt sich legitim von `8255821b…` auf `eba6ba7a…`, dreifach
belegt.

## Arbeitsregel-11-Verdikt: KERNEL (und warum das keine Formsache war)

Arbeitsregel 11 verlangt vor jedem neuen Kernel die Frage nach einer Komposition bereits
verifizierter Kernel. Die beiden Vorgängerscheiben hatten sie bejaht (S2 `mean` =
`sum ∘ scalar_div`; S3 `stack` = N × `nt_materialize`, `item` = reiner strided TS-Read).
Für `argmax` lautet die Antwort **nein**:

- **Komposition ausgeschlossen.** Der Kernel-Bestand enthält keine Vergleichs-,
  Maximum- oder Index-produzierende Operation. Baustein 0 hat das nicht geglaubt, sondern
  über alle Dateien in `crates/core/src/kernels/` gegrept und bestätigt.
- **Kernel-los in TS** (das S3-`item`-Muster: Schleife über `core.memory.buffer`) wurde
  ausdrücklich als Alternative in die Spec geschrieben, damit Baustein 0 sie kippen kann.
  Er hat sie nicht gekippt: `item` war ein O(1)-Skalar-Read, `argmax` ist eine O(N)-
  Reduktion — dieselbe Klasse wie `dot` und `norm_sq`, die beide einen Kernel haben und
  damit den Präzedenzfall „Reduktionen bekommen einen Kernel" setzen.

Konsequenz: **M1 bindet neu**, Arbeitsregel 10 greift (zwei neue `CoreExports`-Member),
und der Freeze-Beweis muss nach dem S0/S1-Muster geführt werden.

## Was gebaut wurde

**Rust** — neue Datei `crates/core/src/kernels/argmax.rs` (361 Zeilen) mit
`argmax_all_strided` / `argmax_axis_strided`, schleifen-strukturgleich zu
`sum_all_strided` / `sum_axis_strided`. Gegenüber den `sum`-Vorbildern ändern sich genau
zwei Dinge: die Akkumulation wird durch den Challenger-Vergleich plus Index-Buchführung
ersetzt, und es kommen die beiden neuen Leer-Zweige hinzu (`size == 0` bzw.
`axis_dim == 0` → `ShapeIncompatible`), die `sum` nicht braucht. `beats_max` ist die
wörtliche Transliteration von `beatsMax` (runtime.ts:555), die Totalordnung stammt
unverändert aus [op-w1-argmax-topk-spec.md](op-w1-argmax-topk-spec.md) D4.

Zwei ABI-Einstiegspunkte `nt_argmax_all_strided` (7 Parameter) /
`nt_argmax_axis_strided` (9 Parameter), signatur-identisch zu den `sum`-Zwillingen,
**strikt ans Dateiende** von `abi.rs` angehängt.

**Der tragende Semantik-Punkt:** `nt_argmax_all_strided` liefert den Index in die
**logische** row-major-Abflachung der View, nie den berechneten Speicher-Offset. Auf
transponierten oder geslicten Views fallen beide auseinander. Das ist dieselbe Falle, die
`sum_all_strided`s Doc-Kommentar für die Akkumulations*reihenfolge* beschreibt, hier in
der Index-Domäne — und sie ist load-bearing, siehe Bausteins A Mutant unten.

**TypeScript** — `WNDArray.argmax` als Instanz-Methode mit drei Overloads (niladisch →
`number` nach der `dot`/`norm`-Präzedenz; Achsen-Form und keepdims-Form →
`WNDArray<OkShape<ReduceAxis<…>>>` nach dem `sum`/`mean`-Muster), plus ein privater Helfer
`argmaxAllBuf`, den beide Voll-Reduktions-Formen teilen. Insertion-only ans Ende des
Klassenkörpers. Die drei Fehler-Stämme werden **in TS prävalidiert, vor jeder Allokation**,
damit sie wortgleich zu `argmaxRuntime` sind.

**Keine Facaden-Änderung** — anders als S3, wo `stack` als *statische* Methode für
Paketkonsumenten sonst unerreichbar gewesen wäre. `argmax` ist eine Instanz-Methode auf
einem Handle, das Konsumenten ohnehin halten; Baustein A hat das nicht nur gelesen,
sondern über `NDArray.backend("wasm")` end-to-end nachgefahren.

## Freeze-Beweis (M4), dreiteilig — und zweifach unabhängig nachgestellt

| Schritt | Ergebnis |
|---|---|
| ① Pre-Edit-Clean-Rebuild auf `3fe47ae` | reproduziert den **alten** Pin `8255821bb1fb42b0367296cc9f64886a4e72968fcc3290086e7ab24309739176` exakt |
| ② Additive-only-Dekomposition am Diff | `abi.rs` ein einziger Hunk am Dateiende: **138 Insertions, 0 Deletions**; `mod.rs` 3/0; `argmax.rs` neu; `shape.rs`/`matmul_blocked.rs`/`sum.rs` byte-unverändert |
| ③ Neuer Pin aus Clean-Rebuild | `eba6ba7ac85d15a814fd027392a81c7450d885d2048d7efd6694b7e8370988bb`, zweimal byte-identisch |

Schritt ① ist der Teil, der den Beweis trägt: er zeigt, dass die Hash-Änderung von *dieser
Scheibe* kommt und nicht von der Umgebung. **Baustein A hat beide Richtungen selbst
nachgebaut** (Rust-Änderungen weggestasht → alter Pin; zurück → neuer Pin, zweimal),
**Baustein B ebenfalls** (`git stash push -u` nur auf `crates/`). Damit liegt der
Freeze-Beweis in drei unabhängigen Ausführungen vor.

Threads-Artefakt `3ad9c49b4d6d768f2a2c7e82175e24c1ba34acdf4d336c3c97c55e76c9621df5` —
bewusst **kein** persistierter Pin (S0/S1-Präzedenz); `test:threaded` beweist die
Bit-Identität zum stable Core verhaltensseitig.

## Pin-Protokoll (D8)

Gemessen im frischen Worktree, im Haupt-Baum reproduziert. **Dateizahl bleibt 140 →
null Order-Noise**, wie vorhergesagt (kein neues File, alle Tests als Anhänge).

| Stufe | Inhalt | Wert | Δ |
|---|---|---|---|
| 0 | Baseline `3fe47ae` | 226.690 | — |
| ① | `CoreExports`-Member + beide Mock-Stubs | 226.690 | **0** |
| ② | `WNDArray.argmax` + privater Helfer | 227.148 | +458 |
| ③ | Test-Anhänge (resident / lifecycle / threaded) | 229.265 | +2.117 |
| ④ | Typ-Pins | 229.647 | +382 |
| ⑤ | Verify-Runden-Nachträge (V1/V2/memory.grow) | **229.828** | +181 |

**Gesamt +3.138 gegen das vorregistrierte Gate von ≤ +8.000 — mit 4.862 Luft gehalten.**
Vom Orchestrator selbst nachgemessen: `Files: 140`, `Instantiations: 229828`, Exit 0.

Stufe ① bestätigt den Kampagnengewinn aus S0/D10 erneut: ein neuer `CoreExports`-Member
kostet **+0** statt +7, weil der `Omit<ThreadedCoreExports,"memory">` → Direkt-Cast-Fix
die `keyof`-getriebene Generic-Neuauflösung an der Wurzel beseitigt hat. Bei n=2 exakt
bestätigt (nach n=4 in S1 und n=0 in S2).

Weitere Pins: `check:diag:stress` **115.934 @ 82** (Δ+436) · `check:diag:browser`
**2.142 @ 75** (Δ0) · `bench:editor` **uniform +436** über alle acht Workloads, zweifach
byte-identisch gemessen, neue Pins `{w1 37454, w2 39287, w3 70429, w4 37609, w5 42908,
w6 44102, w7 36658, w8 44347}`. Die Uniformität ist diesmal perfekt (anders als S3, wo w8
eine eigene `stack`-Aufrufstelle hatte) — kein Workload hat eine `argmax`-Aufrufstelle,
der Effekt ist reine Klassen-Surface-Ripple. Der identische Wert +436 auf `stress`, das
`spike/src` direkt kompiliert, bestätigt die Attribution unabhängig.

Testzahlen: `test:core` 1591 (Δ0) · `test:resident` **5866+2** (Δ+369) · `test:threaded`
**127** (Δ+13) · `cargo` **204+1** (Δ+20) · `test:browser` 4 · `test:package` 3.

## Das Absolut-Gate wurde vor der Messung angehoben — mit offengelegter Herleitung

Die Spec v1 registrierte ≤ +6.000, begründet mit S2/`mean` = +1.500. Baustein 0 hat diese
Begründung als zu optimistisch markiert: S2s **realisierte** Gesamtkosten waren nicht
+1.500, sondern **+5.689** — der View-Coverage-Nachtrag (selbst ein Verify-Befund) kostete
allein +4.189. S4 fordert diese View-Fälle von Anfang an und hat zusätzlich eine
Rust/ABI/`CoreExports`-Schicht, die S2 nicht hatte.

Das Gate wurde daraufhin in v2 auf ≤ +8.000 angehoben — **vor jeder Messung**, mit der
Herleitung im Spec-Text, und die STOPP-Klausel blieb unverändert. Das ist ausdrücklich
etwas anderes als S3, wo ein gerissenes Gate nachträglich Owner-abgenommen angehoben
wurde. Der realisierte Wert (+3.138) liegt unter beiden Zahlen — die alte Registrierung
hätte also auch gehalten. Das ändert nichts daran, dass die v1-Begründung sachlich falsch
war und die Korrektur richtig.

## Verify-Runde

### Baustein 0 (vor der ersten Codezeile)

Zwei substanzielle Befunde, die sonst mitten in der Arbeit aufgeschlagen wären:

**F1 (MAJOR) — Arbeitsregel 10 in CLAUDE.md war faktisch falsch.** Sie nannte
`backend-oom.test.ts` als „das EINZIGE strukturell getippte `CoreExports`-Literal im
Repo". Es gibt ein zweites: `resident-lifecycle.test.ts:601`, mit eigenem lokalem
`notImplemented`-Helfer. Empirisch reproduziert statt gelesen: zwei Dummy-Member ins
Interface → **beide** Dateien werfen TS2739 bei Exit 1, während `check:diag` weiterhin
eine plausible `Instantiations: 231240`-Zeile druckt. Das ist die Arbeitsregel-6-Falle
(ein Gate, das seine eigene Messlauf-Gesundheit nicht mitprüft) in Reinform. Die
Regel-Korrektur wurde Teil dieser Scheibe.

**D1 (MAJOR-ish) — die Orakel-Methodik für View-/keepdims-Fälle war unspezifiziert, und
das Repo trug dafür zwei unvereinbare Präzedenzfälle.** S2/`mean` benutzt `keepDimsShape`
als Shape-Orakel; `NDArray.argmax`s eigener Test lehnt genau das als zirkulär ab (der
Helfer wird von der geprüften Funktion selbst aufgerufen) und prüft stattdessen
strukturelle Invarianten. Beide haben recht — für verschiedene Fragen. Die Spec v1 wählte
keine von beiden, dieselbe Lückenform, die S2 ein eigenes Nachtrags-Addendum gekostet
hatte.

Aufgelöst als dreiteilige bindende Regel: **Daten** gegen `argmaxRuntime`, mit vorab
gelesenem `view.toArray()` (die S2-Lektion: die Referenz nie aus dem Basis-Puffer neu
herleiten); **keepdims-Shape** über strukturelle Invarianten statt `keepDimsShape`; plus
ein **orakelfreier Cross-Surface-Shape-Pin** gegen die materialisierte `NDArray`-Form.

Dazu drei kleinere Korrekturen: die `unwrap_or(0.0)`-Konvention ist bei `argmax` nicht so
harmlos wie bei `sum` (dort additives Neutrales, hier ein echter Vergleichs-Kandidat — die
**Korrektheit**, nicht nur die Speichersicherheit, hängt damit an
`validate_strided_bounds`; als Pflicht-Doc-Kommentar verankert); die
`keepDimsShape`-Call-Site-Zählung; und eine zu enge Beschreibung des Kernel-Diffs.

Positiv verifiziert und damit load-bearing statt bloß argumentiert: die vollständige
Overload-Form wurde in einer Scratch-Kopie end-to-end typgeprüft (alle Formen, alle
Degradationskanten, Fehler am Argument, `tsc --noEmit` Exit 0), und beide Baselines
wurden exakt reproduziert.

### Baustein A — Spec-Konformität: konform

Alle D1–D10 einzeln geprüft, alle T1–T8 abgehakt, **jede berichtete Zahl exakt
reproduziert** — ohne eine einzige Abweichung. Freeze-Beweis in beide Richtungen selbst
nachgebaut. Die sechs vom Implementierer gemeldeten Abweichungen wurden einzeln geprüft und
alle als tragfähig beurteilt (u. a.: die Lifecycle-Tests liegen in
`resident-lifecycle.test.ts` statt wie in D7 gruppiert in `resident.test.ts` — dort steht
die Zähl-Mock-Infrastruktur, und S2/S3 taten es genauso; der zusätzliche private Helfer
`argmaxAllBuf` ist DRY-Konsolidierung und verletzt keine Disziplin).

**Bausteins A eigener Mutant war die wertvollste Einzelmessung der Runde.** Statt die
Totalordnung anzugreifen (die der Implementierer schon mutiert hatte), traf A den
logischen-vs-Speicher-Index — genau den Punkt, den D3 als tragende Semantik markiert:
`max_idx` wurde auf den Speicher-Offset statt auf den logischen Zähler gesetzt.

Ergebnis: 3 cargo-Tests fielen, darunter exakt die dafür geschriebene
Nicht-Vakuitäts-Assertion `argmax_all_strided_transposed_view_uses_logical_index`
(erwartete 2.0, bekam 1.0). Und **40 von 1607 JS-Tests — ausschließlich die vier
View-Klassen** über alle Achsen-/keepdims-Kombinationen plus 14 Spezialwert-Fälle mit
`view=true`. **Kein einziger contiguous-Fall fiel**, was mathematisch erwartbar ist (bei
natürlichen Strides und Offset 0 gilt `off == flat`). Das ist der direkte empirische
Beleg, dass Arbeitsregel 12 (View-Coverage) genau die Fehlerklasse fängt, für die sie
eingeführt wurde — und dass eine reine contiguous-Suite hier blind gewesen wäre.

### Baustein B — adversarial: eine reale Lücke, die alle Gates grün ließ

**V2 (MAJOR) — der Kernel-Fehlerpfad war von keinem Test abgedeckt.** `WNDArray.argmax`
gibt bei `status !== 0` den frisch allozierten Ausgabepuffer frei, bevor es wirft — eine
Garantie, die der Doc-Kommentar des Moduls ausdrücklich behauptet. B hat den `freeBuf`-
Aufruf entfernt und die komplette Suite laufen lassen: **0 von 334 argmax-Tests, 0 von 7
Lifecycle-Tests, 0 von 1 threaded-Test** fielen. Mit einem eigenen zählenden Core-Wrapper,
der Status 1 erzwingt, wies B nach, dass der unmutierte Code sauber bilanziert, der
mutierte aber ein reales 16-Byte-Leck hinterlässt. Kein committeter Test erzwang je einen
Kernel-Fehlschlag für diese Op — obwohl S3 für `stack` genau so einen Präzedenz-Test hat.

**V1 (MAJOR-ish) — eine Nicht-Vakuitäts-Assertion, die überclaimt.** Der Leck-Test
behauptete, 500 Aufrufe ohne Speicherwachstum seien eine „unabhängige Bestätigung", weil
ein geleaktes 8-Byte-Ergebnis irgendwann eine Seitengrenze reißen würde. Rechnung:
500 × 8 = 4.000 Byte gegen 64-KiB-Seiten — es bräuchte 8.192 Iterationen, über 16× so
viele. Die Assertion war für genau das Leck, das sie adressierte, arithmetisch unfähig.
Der echte Schutz war der Ledger-Test daneben, der den Mutanten auch fing. **Diese
Fehlerklasse war in S3 bereits gelernt worden — und wurde ausgerechnet in dem Test
reproduziert, der sie verhindern sollte.**

**`memory.grow` mitten im Aufruf — gescheiterter Angriff, aber keine Regressionsabdeckung.**
B baute den Fall selbst (`[4, 2000000]`, Wachstum nachweislich *während* des
`argmax(0)`-Aufrufs, nicht im vorangehenden `fromArray`) und maß **0 von 2.000.000
Abweichungen**. Die Disziplin „Views strikt nach der letzten Allokation ableiten" hält.
Es existierte aber kein committeter Test, obwohl die harte Repo-Regel (`memory.buffer` nie
cachen — mit shared memory schlägt das *still* fehl) genau diese Klasse adressiert.

**Zur Tie-Blindheit hat B die Diagnose des Implementierers verschärft statt bestätigt.**
Der hatte berichtet, seine 240 randomisierten Differentialfälle hätten den eigenen
Pflicht-Mutanten (`>` → `>=`) nicht gefangen, mit der Erklärung „Zufalls-Floats erzeugen
praktisch nie Gleichstände". B präzisierte: die Daten stammen aus einer **stetigen**
Verteilung, ein Gleichstand ist dort ein **Maß-Null-Ereignis** — die 0/240 sind eine
mathematische Gewissheit der Testkonstruktion, kein Pech. Und das Spezialwert-Raster fängt
es nur **zufällig** (5/60), weil sein Generator mit 35 % pro Element aus einem kleinen
diskreten Pool zieht; ob eine Kollision entsteht, hängt am Seed statt an einer
Konstruktion. Verlässlich fangen es nur die absichtlich konstruierten Rust-Unit-Tests. Die
Lücke sitzt damit genau in der Cross-Language-Schicht, in der M1s Vertrag lebt.

### Baustein C — Vertrag: keine Verstöße, keine neue Auslegungsfrage

Die Invarianten-Zuordnung war vollständig; C fand keine übersehene berührte Invariante.
M4 prüfte C strukturell am Diff und **legte offen, was es nicht prüfte**: den Hash-Rebuild
selbst nicht, weil ein `cargo build` Artefakte schreibt und sein Mandat read-only ist —
die richtige Grenzziehung, der Rebuild-Beweis lag bei A und B.

C's inhaltlich wertvollster Beitrag betrifft **S5, nicht S4**: In FOLLOWUPS steht als
COVENANT-v6-Kandidat, dass M1s Text den NaN-Payload-Vorbehalt nicht nennt. C stellte fest,
dass der für `argmax` **strukturell gar nicht greifen kann** — die Ausgabe ist stets ein
ganzzahliger Index, nie ein kopierter Datenwert, ein NaN-Payload kann also nicht
durchschlagen. Bei `topk` (S5) wird er dagegen relevant, weil dort
`values[i] === data[indices[i]]` byte-exakt gilt. Das entschärft den v6-Kandidaten hier
und verschiebt ihn präzise dorthin, wo er beißt.

### Arbeitsregel 13: die Hover-Norm wurde dreifach gemessen

S3 hatte gelehrt, dass eine Hover-/Diagnose-Norm gemessen und nicht gelesen werden muss —
dort übersahen drei Leser des Quelltexts eine M3-Verletzung, die nur der Verifier fand,
der den echten LSP-Server startete. In dieser Scheibe haben **der Implementierer, Baustein
A und Baustein B je unabhängig** eine echte `tsc --lsp --stdio`-Messung mit `sum` als
Kontrollpunkt gefahren, C zusätzlich eine vierte. Alle vier: saubere aufgelöste Tupel an
allen Formen (`argmax(1, true)` → `WNDArray<[2, 1, 4]>`, `argmax(-1)` → `WNDArray<[2, 3]>`,
`argmax()` → `number`), **keine Alias-Leckage**. Die S3-Fehlerklasse ist hier by
construction abwesend, weil D2 den Rückgabetyp ausschreibt statt einen Top-Level-Alias in
Rückgabeposition einzuführen.

B hat zusätzlich die `@ts-expect-error`-Positionen spaltengenau verifiziert statt
angenommen: `rw.argmax(5)` → `(3,11)`, und Zeichen 11 ist exakt die `5`.

## Befund-Schließung nach der Verify-Runde

| Befund | Auflösung | Nicht-Vakuitäts-Beweis |
|---|---|---|
| V2 Kernel-Fehlerpfad | zwei neue Tests in `resident-lifecycle.test.ts`, ein Mock je Einstiegspunkt erzwingt Status ≠ 0 | **je Zweig getrennt mutiert**: `freeBuf` bei resident.ts:1775 entfernt → genau 1 Fehlschlag (`assertLedgerBalanced`, all_strided-Test), Achsen-Test blieb grün; `freeBuf` bei :1905 entfernt → genau 1 Fehlschlag (Achsen-Test), niladischer blieb grün |
| V1 überclaimende Assertion | **Auflösung (b)**: Anspruch im Kommentar zurückgenommen auf „ergänzende Plausibilitätsprüfung, der Ledger-Test ist der Beweis" | Rechnung im Kommentar; Präzedenz im selben File (`stack SUCCESS path`) |
| `memory.grow` | committeter Regressionstest mit eigenem frischem Core; `fromArray` verursacht beweisbar **null** Wachstum, `argmax(0)` erzwingt es | Bit-Identität gegen `argmaxRuntime`, Ergebnis 13.337/86.663 auf zwei Indizes verteilt (kein degeneriertes Signal) |
| Regel-10-Prüfkommando | `grep -A1` — die Folgezeile diskriminiert selbst (`...name,` = Spread, exempt; echtes Member = exhaustiv) | am finalen Baum verifiziert: genau 2 exhaustive, 2 Spread |

Dass **jeder Mutant genau einen** Test fällte und der jeweils andere Zweig grün blieb, ist
das entscheidende Signal: ein schlampiger Test hätte beide oder keinen gefällt.

Die Befund-Schließung berührte **kein Rust** (vom Orchestrator nachgeprüft: `crates/`-Diff
unverändert 138/0, 361/0, 3/0) und ließ `resident.ts` byte-identisch (SHA-256-Rundreise
bewiesen, Diff weiterhin 185 Insertions / 0 Deletions).

**Auf eine dritte Verify-Runde wurde bewusst verzichtet.** Die Nachträge sind
Test-Additionen plus ein Kommentar und eine Regel-Zeile; kein Produktivcode wurde
geändert, jeder neue Test trägt seinen eigenen Mutanten-Beweis mit benannter Assertion,
und der volle Gate-Block lief erneut grün. Eine weitere Runde wäre Über-Verifikation.
Die Entscheidung steht hier, statt still getroffen zu werden.

## Ehrliche Einschränkung der M1-Behauptung (Baustein-B-Befund C1)

M1 verlangt Bit-Identität zur TS-Referenz. Die ist über contiguous, alle vier
View-Klassen, alle drei Formen und das Spezialwert-Raster bewiesen — **mit einer
benennbaren Ausnahme an einer degradierten Eingabe**, die B empirisch reproduziert hat:

Eine **nicht-ganzzahlige Achse** (nur über `as unknown as number` erreichbar, kein
typsicherer Compile-Pfad) divergiert zwischen den Flächen. Ursache, von B präzise
zurückverfolgt: JS behält den Float und trunkiert erst später innerhalb von
`Array.prototype.slice`s `ToIntegerOrInfinity`; die WASM-ABI macht dagegen `ToInt32`
**vor** Rusts eigener Negativ-Achsen-Arithmetik. Bei `axis = 1.5` liefert `NDArray`
degenerierte Nullen, `WNDArray` eine echte, andere Reduktion; bei `-1.5` wirft `WNDArray`
`status 3`, während `NDArray` nicht wirft.

**Das ist vorbestehend, nicht von dieser Scheibe eingeführt:** B hat live reproduziert,
dass derselbe Mechanismus für `WNDArray.sum` seit S1 gilt, und die Wurzel
(`sumRuntime`/`argmaxRuntime` akzeptieren non-integer-Achsen klaglos) ist als
FOLLOWUPS-Item seit der W1-Scheibe offen und bewusst zurückgestellt. D4 verlangt
ausdrücklich Struktur-Gleichheit zu `sum`, die Scheibe **erbt** die Lücke also korrekt,
statt sie zu erzeugen. Der FOLLOWUPS-Eintrag wird um die bisher unbekannte
WASM-ABI-Trunkierungs-Dimension ergänzt.

Es bleibt trotzdem richtig: die M1-Behauptung ist an dieser einen degradierten Eingabe
nicht wörtlich unqualifiziert wahr. Das steht hier, statt weggerundet zu werden.

## Weitere offengelegte Grenzen

- **Baustein B hat die gestufte `check:diag`-Dekomposition nicht unabhängig nachgerechnet**
  (dafür wären mehrere isolierte Compiles nötig gewesen); es hat das Gesamt-Delta und die
  Gate-Einhaltung verifiziert, woran die Annahme-Entscheidung hängt. Baustein A hat die
  Zahlen reproduziert, die Stufung stammt vom Implementierer.
- **Der Rust-Achsen-Grenzcheck ist vom Produkt-Surface aus unerreichbar** (B/V3): eine
  Mutation von `norm_axis >= rank` zu `> rank` fällt in cargo sofort auf (Panic), aber
  0 von 334 TS-Tests — weil TS die Achse immer vorvalidiert. Kein Defekt, aber benannt:
  eine *stille* Divergenz zwischen der TS-Prüfung und der Kernel-Grenze wäre für das
  M1-Differential unsichtbar.
- **`test:example` wurde von Baustein A und B nicht ausgeführt** (es installiert das
  publizierte npm-Paket aus der Registry und prüft Registry-Drift, berührt diesen lokalen
  Diff also nicht). Der Implementierer und die Befund-Schließungs-Runde haben es
  ausgeführt, Exit 0.
- **Ein mechanisches GFM-Gate existiert im Repo nicht** (vorbestehend); die geänderten
  `.md` wurden per Grep manuell auf `~~`-Strikethroughs geprüft.

## FOLLOWUPS-Kandidaten aus dieser Scheibe

- **ABI-Dispatch-Abdeckung** (erweitert das bestehende Item um zwei Exporte): die
  cargo-Tests rufen `argmax_{all,axis}_strided` direkt, nie durch `nt_argmax_*_strided`.
  Hier eng begrenzt, weil die beiden Einstiegspunkte verschiedene Aritäten haben und eine
  Verwechslung nicht kompilieren würde — die Klasse besteht aber.
- **Duplizierter keepdims-Invarianten-Helfer** zwischen `resident.test.ts` und
  `threaded.test.ts` (bewusst: getrennte Korpora, keine Cross-Test-File-Import-Konvention).
- **Non-integer-Achse:** das bestehende Item um die WASM-ABI-Trunkierungs-Dimension
  ergänzen (siehe oben).
- **`WasmBackend`/`ThreadedBackend` exponieren weiterhin nur Konstruktion + `stack`** —
  korrekt nach D1, aber die Asymmetrie bleibt bestehen.

## Gate-Block (Endstand, alle Exit 0)

`pnpm check` (Dreier-Verbund) · `check:diag` **229.828 @ 140** · `check:diag:stress`
**115.934 @ 82** · `check:diag:browser` **2.142 @ 75** · `test:core` **1591** ·
`test:resident` **5866+2** · `test:threaded` **127** · `test:browser` **4** ·
`test:package` **3** · `test:example` · `cargo test` **204+1** · `check:freeze`
**`eba6ba7a…`** · `bench:editor` Hard Gate PASS · `graph-a-lama query lint` 0/0 · `demo`.
