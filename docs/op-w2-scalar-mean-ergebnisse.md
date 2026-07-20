# Op-Scheibe W2: Skalar-Overloads + `mean` — Ergebnisse

Status: **abgeschlossen, dreifach verifiziert; Zahlen im Haupttext = Stand VOR der
Verify-Runde** (finale Zahlen nach F1-Fix im Post-Verification-Addendum am Ende). Alt:
(Post-Verification-Addendum folgt — steht laut
Eskalationsleiter/Stufe-3-Verify-Plan noch aus, siehe „Offene Punkte" unten).
Spec: docs/op-w2-scalar-mean-spec.md (Version 3, inkl. Baustein-0- und Verify-Addendum).
Datum: 2026-07-21.

## Summary

Die zweite Op-Scheibe aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md, W2/F2 —
Skalar-Wrap-Workaround `fromArray([1],[2])`, HANDOFF-Erwartung „mean" als granularere Lücke) ist
umgesetzt: `NDArray.add`/`sub`/`mul`/`div` bekommen je einen Skalar-Overload (`x.div(2)` liest sich
als „durch 2 teilen", shape-erhaltend — auch bei Rang 0), und `mean` (Overloads 0/1/2 exakt nach
`sum`-Muster) ist neu. D1: **NDArray-only, kein WASM-Kernel** — dieselbe bewusste, dokumentierte
Surface-Asymmetrie wie W1s argmax/topk, jetzt COVENANT-v5-gedeckt (M1-Präzisierung „kernel-lose
Referenz-Ops zulässig"). Alle D8-Gates grün; das Absolut-Gate (Haupt-Pin-Wachstum ≤ +10,000) hält
mit deutlichem Spielraum (+5,762); der sum/n-vs-sum\*(1/n)-Diskriminator ist NICHT-VAKUÖS bewiesen
(zwei unabhängige, echte Beispiele, siehe unten) — der bindende Determinismus-Punkt D5 ist damit
tatsächlich geprüft, nicht nur behauptet.

## Umgesetzte Form je D-Punkt

- **D1 (Surface & Scope):** Nur `NDArray`. Vier Skalar-Overloads (add/sub/mul/div) + neue Methode
  `mean`. Kein `rsub`/`rdiv`, kein `.scale()`-Alias (Spec-Nicht-Ziele eingehalten).
- **D2 (Skalar-Semantik, shape-erhaltend, v2):** `x.div(s): NDArray<S>` — auch bei Rang 0 bleibt
  `[]` `[]` (Kontrast zum alten `[1]`-Wrap-Workaround, der `[] → [1]` gemacht hätte — explizit als
  eigener Runtime- UND Typ-Pin bewiesen, nicht nur behauptet). Kein Guard am Skalar (jeder
  endliche/nicht-endliche `number` ist valide, IEEE-Propagation). Die v2-Ausnahme — ein
  UNION-Argument über die Overload-Grenze (`number | NDArray<B>`) wird von TS als Ganzes
  abgelehnt (TS2769) — ist als dokumentierte Kante gepinnt (`@ts-expect-error`, Präzedenz
  `NDArray.backend(kind)`), inklusive eines funktionierenden Narrowing-Workarounds
  (`typeof x === "number" ? … : …`), per Mutationsprobe non-vakuös bewiesen.
- **D3 (Skalar-Runtime):** `scalarElementwiseRuntime(op, data, s): Float64Array`, APPENDED in
  runtime.ts — ein String-Op-Dispatcher (Baustein-0-Nit-4-Freistellung genutzt), elementweise
  `data[i] op s` aufsteigend, frisches Float64Array. Der Pflicht-Differential (Rang ≥ 1,
  byte-identisch zum `[1]`-Wrap-Weg über `elementwiseBinary`) ist erfüllt: 160 randomisierte Fälle
  über alle vier Ops, Ränge 1–3, mit Spezialwert-Injektion (NaN/±Infinity/±0/Subnormals via
  `genDataSpecial`/`nextF64Special`), alle byte-exakt (`assertDataBitIdentical`).
- **D4 (`mean`-Form):** Overloads exakt wie `sum` (niladisch → `NDArray<OkShape<ReduceAxis<S,
  undefined, false>>>`; `(axis)`; `(axis, keepdims)`) — `ReduceAxis`/reduce.ts UNVERÄNDERT
  wiederverwendet. KEIN argmax-artiger `arguments.length`-Sonderfall nötig: anders als `argmax()`
  (das bei 0 Argumenten einen bloßen `number` zurückgibt) geben ALLE `mean`-Overloads
  `NDArray<...>` zurück, also gibt es keine „echte 0-Arg-Form vs. 1-Arg-Form mit Wert `undefined`"-
  Verwechslungsgefahr — die W1-Lektion (Bug: `axisNum === undefined` statt `arguments.length ===
  0`) trifft hier strukturell nicht zu; die Implementierung ist deshalb ein reiner Bauplan-Klon von
  `sum()`s eigenem Rumpf (nur `sumRuntime` → `meanRuntime` ersetzt).
- **D5 (`mean`-Runtime, Reihenfolge gepinnt):** `meanRuntime` = `sumRuntime`-Aufruf, dann GENAU
  EINE Division pro Output-Element durch `n` (`shape[normAxis]` bei Achse, `product(shape)` bei
  `undefined`) — NICHT `sum * (1/n)`. Achsen-Validierung/Throw-Stem ist WORTGLEICH `sumRuntime`s
  eigener Throw, weil `meanRuntime` diese Validierung gar nicht selbst macht, sondern komplett an
  `sumRuntime` delegiert (die Wortgleichheit folgt aus der Delegation, nicht aus einer
  Duplizierung). size-0 (leerer Empfänger oder size-0-Achse): Summe 0, `n=0` → `0/0 = NaN`, KEIN
  Throw (NumPy-konform) — beide Reduktionspfade explizit getestet.
- **D6-v2 (Datei-Disziplin, erzwungene Overload-Ausnahme):** Die vier Bestandsmethoden
  add/sub/mul/div wurden ediert (TS2394 verbietet Overload-Signaturen vor einer body-tragenden
  Deklaration — keine Alternative ohne D1/D2 zu brechen). Form je Methode: (1) die bestehende
  generische Signaturzeile wurde BODYLOS (Zeileninhalt sonst identisch — nur `{` → `;`), (2) neue
  Skalar-Overload-Signatur `(s: number): NDArray<S>` ergänzt, (3) neue union-typisierte
  Implementierungssignatur, deren Rumpf die ORIGINALE Logik (die drei Zeilen `const o = …`,
  `elementwiseBinary(…)`, `return new NDArray(…)`) BYTE-IDENTISCH in den `else`-Zweig verschiebt —
  am `git diff` bewiesen: diese drei Zeilen erscheinen für alle vier Methoden als reine
  Kontextzeilen (kein `+`/`-` davor), siehe Byte-Erhaltungs-Nachweis unten. `mean` ist eine reine
  Klassenkörper-Insertion nach `topk`. runtime.ts: reiner Append (zwei neue Funktionen, strikt
  nach dem letzten Bestandscode). Import-Zeilen in ndarray.ts erweitert (`meanRuntime`,
  `scalarElementwiseRuntime`) — gleiche Auslegung wie W1. Kein Byte in vector.ts/reduce.ts/dim.ts/
  literal-arithmetic.ts/index.ts, kein neues File unter spike/src.
- **D7 (Tests):** `spike/tests-runtime/scalar-mean.test.ts` (NEU, 482 Tests, in test:core
  registriert) + 19 neue Type-Pins in ndarray.test-d.ts (17 benannte `Expect<Equal<...>>` + 2
  `@ts-expect-error`-Direktiven — Zählung korrigiert, siehe „Tests im Detail" unten für die
  Aufschlüsselung und die bewusste Abweichung von der „≈4-6"-Schätzung des Spec-Addendums).
  Details unten.
- **D8 (Gates & Pins):** Siehe Pin-Protokoll- und Gate-Tabelle.

## Diskriminator-Beweis (D5, non-vakuös — echte Zahlen)

Der bindende Punkt D5 verlangt `sum/n` statt `sum*(1/n)`. Ein Pin dieser Entscheidung ist nur dann
etwas wert, wenn die beiden Formeln in f64 tatsächlich unterschiedliche Werte liefern — beide
Konstruktionen unten BEWEISEN das erst (Precondition-Assertion im Test), bevor sie den Pin selbst
prüfen:

**Volle Reduktion (n=49, sum=5):**

```
5 / 49        = 0.10204081632653061   (D5-gepinnte Formel — das ist, was meanRuntime liefert)
5 * (1 / 49)  = 0.1020408163265306    (die VERWORFENE Formel — eine andere letzte Nachkommastelle)
```

Konstruiert als 49-elementiges Array (Index 0 = 5, 48 Nullen) — die Summe ist exakt 5 (keine
Rundung in der Summation selbst, per `sumRuntime`-Gegenprobe im Test verifiziert), `mean()`
liefert exakt `0.10204081632653061` (per `Object.is` gepinnt), NICHT `0.1020408163265306`.

**Achsen-Fall (shape=[4,49], axis=1, Zeilensummen [5, 9, 1, 2]):**

```
Zeile 0: sum=5  → 5/49 = 0.10204081632653061   vs.  5*(1/49) = 0.1020408163265306    → WEICHT AB
Zeile 1: sum=9  → 9/49 = 0.1836734693877551    vs.  9*(1/49) = 0.18367346938775508   → WEICHT AB
Zeile 2: sum=1  → 1/49 = 0.02040816326530612   vs.  1*(1/49) = 0.02040816326530612   → GLEICH
Zeile 3: sum=2  → 2/49 = 0.04081632653061224   vs.  2*(1/49) = 0.04081632653061224   → GLEICH
```

Genau 2 von 4 Zeilen diskriminieren — die Spec-Warnung „nicht jedes Beispiel diskriminiert" ist
damit selbst am eigenen Testfall bewiesen, nicht nur zitiert. `mean(1)` liefert für jede Zeile
exakt `sum/n` (per `Object.is` gepinnt); für die zwei diskriminierenden Zeilen wird zusätzlich
explizit assertiert, dass das Ergebnis NICHT der verworfenen Formel entspricht.

## Pin-Protokoll (D8, gestufte Attribution)

Baseline (frischer `git worktree` von HEAD `9bd67ce`) **exakt reproduziert**: `184,330 @ 136` ·
stress `103,719 @ 82` · browser `2,142 @ 75` (Exit 0, alle drei, `pnpm check` im Worktree ebenfalls
grün).

| Messpunkt | Dateien | Instantiations | Δ zum Vorpunkt | Attribution |
|---|---|---|---|---|
| Baseline | 136 | 184,330 | — | frischer Worktree |
| ① runtime.ts (`scalarElementwiseRuntime`+`meanRuntime`) + ndarray.ts (D6-v2-Overloads + `mean`) | 136 | 185,204 | **+874** | Klassen-Surface-Wachstum (4 neue Skalar-Overloads + neue `mean`-Methode) |
| ②a + neues Testfile `scalar-mean.test.ts`, LEER (`export {}`), registriert | 137 | 187,404 | **+2,200** | reiner Order-Noise (Datei-Hinzufügen reshuffelt die Fresh-vs-Cached-Partition, CLAUDE.md-dokumentiertes Phänomen) |
| ②b + Testfile GEFÜLLT (482 Tests) | 137 | 189,368 | **+1,964** | echte Typkosten des Testfiles selbst |
| **final** + ndarray.test-d.ts-Pins (9 neue) | 137 | **190,092** | **+724** | Typ-Pin-Katalog |

**Gesamtwachstum (final − Baseline): 190,092 − 184,330 = +5,762** — deutlich innerhalb des
Absolut-Gates ≤ +10,000. Determinismus: final zweimal gemessen, byte-identisch (190,092 beide
Male).

**stress**: Baseline `103,719 @ 82` → final **`104,900 @ 82` (Δ +1,181)** — derselbe Mechanismus
wie W1s stress-Ripple (+842): `spike/tests-stress` importiert `spike/src` direkt und instanziiert
`NDArray<S>` intensiv; die gewachsene Klassen-Member-Fläche (vier neue Overload-Signaturen + eine
neue generische Methode) kostet dort marginal mehr Auflösungsarbeit PRO Instanziierung, auch ohne
dass ein einziger stress-Test die neuen Ops tatsächlich aufruft. Datei-Anzahl unverändert (82,
`spike/tests-runtime` ist nicht Teil dieses Korpus), zweimal deterministisch reproduziert (bei
Stufe ① und nochmal am Ende).
**browser**: Baseline `2,142 @ 75` → final **`2,142 @ 75` (exakt, Δ 0)** — zweimal deterministisch
reproduziert, wie schon bei W1.

## bench:editor: Vorher/Nachher (D8-erlaubte Verschiebung)

Sechs der sieben Workloads verschoben sich UNIFORM um **+1,181** (w1/w2/w3/w5/w6/w7); w4 (die
Datei mit den zwei absichtlichen Typfehlern) verschob sich um **+1,220** — eine echte, doppelt
reproduzierte, attribuierte Abweichung: w4s `ShapeError`/`Guard`-Diagnosepfade lösen sich gegen
das jetzt größere add/div-Overload-Set anders auf als die fehlerfreien Workloads. Nicht ins Gate
optimiert (Code nicht verändert, um den alten Wert zu erzwingen). Beide Messungen byte-identisch:

| Workload | vorher (nach W1) | nachher (W2) | Δ |
|---|---|---|---|
| w1 | 25,109 | 26,290 | +1,181 |
| w2 | 26,918 | 28,099 | +1,181 |
| w3 | 58,058 | 59,239 | +1,181 |
| w4 | 25,270 | 26,490 | **+1,220** |
| w5 | 30,563 | 31,744 | +1,181 |
| w6 | 31,733 | 32,914 | +1,181 |
| w7 | 24,281 | 25,462 | +1,181 |

`INSTANTIATION_PINS` in `spike/bench-dx/editor-latency.ts` aktualisiert (der einzige erlaubte
Bestandsdatei-Edit dort, D8) — reine Daten-/Kommentar-Änderung, `check:diag`-neutral verifiziert
(190,092 unverändert vor/nach dem Pin-Update). Latenz-Mediane und das Correctness-Gate blieben
PASS, unter dem 2x-Ceiling. Hard-CI-Gate nach dem Update: **PASS**, Exit 0.

## Gates (alle frisch gemessen, dieser Commit)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (3-Verbund) | 0 Fehler root+stress+browser | 0 |
| `check:diag` | 190,092 @ 137 (2× deterministisch) | 0 |
| `check:diag:stress` | 104,900 @ 82 (2× deterministisch, Δ+1,181 offen berichtet) | 0 |
| `check:diag:browser` | 2,142 @ 75 (2× deterministisch, exakt) | 0 |
| `test:core` | **1,334/1,334** (852 Baseline + 482 neu) | 0 |
| `test:resident` | 4,278 pass + 2 skip (unverändert) | 0 |
| `cargo test` | 161 passed (unverändert, kein Rust berührt) | 0 |
| `check:freeze` | Hash byte-identisch `0b9df4f1…2519c7d` | 0 |
| `bench:editor` | Hard-Gate PASS nach Pin-Update (vor Update: FAIL, erwartungsgemäß) | 0 |
| `graph-a-lama query lint` | 0 Findings, 0 Errors/Warnings | — |
| `pnpm test:example` | PASS, weiterhin numtype@0.1.1 (Registry-Major.Minor-Match) | 0 |
| `pnpm demo` | PASS (zusätzliche Absicherung, nicht D8-Pflicht) | 0 |

## Tests im Detail

**Runtime (`spike/tests-runtime/scalar-mean.test.ts`, 482 Tests):** wie W1 kein WASM-Gegenpart
(D1) — Coverage kombiniert: einen expliziten Op×Rang(0/1/2)×Spezialwert-Katalog (12 gruppierte
Tests, je Kombination gegen den nativen IEEE-Operator selbst als unabhängige Referenz — dieselbe
Konvention, die `add.test.ts`s eigener `(x,y)=>x+y`-Referenz schon zugrunde liegt); 160
randomisierte `[1]`-Wrap-Äquivalenz-Fälle (byte-exakt via `assertDataBitIdentical`, Rang 1–3, mit
NaN/±Infinity/±0/Subnormal-Injektion); einen expliziten Rang-0-Kontrast-Test gegen den ALTEN
`[1]`-Wrap-Workaround (beweist D2s Motivation als echten Lauf, nicht nur als Behauptung); den
nicht-vakuösen sum/n-Diskriminator (voll + Achse, s.o.); 300 randomisierte `mean`-Cross-Checks
gegen einen unabhängig geschriebenen Brute-Force-„Summe-dann-Teilen"-Referenzcode (eigener
Stride-/Unravel-Algorithmus, nicht `runtime.ts`s `computeStrides`/`unravel`); `mean`-von-empty →
NaN auf beiden Reduktionspfaden (voll UND Achse), mit explizitem Kontrast zu `argmax`, das auf
demselben Input wirft; eine Wort-für-Wort-Stem-Gleichheitsprobe (`meanRuntime`s und `sumRuntime`s
Achsen-Fehler werden BEIDE gefangen und verglichen — sowohl direkt als auch über die
`NDArray.mean()`-Klassenmethode); `mean(undefined, true)`.

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, +19 Pins — 17 benannte Typen + 2
`@ts-expect-error`):** Skalar-Overload-Gruppe (8 benannt + 1 Direktive = 9): `div(2)`-Shape-Erhalt
exakt (Rang-2, Rang-0, wide/dynamischer Rang, Readonly-S); Union-über-Grenze als
`@ts-expect-error` (D2 v2, Präzedenz `backend()`) plus ein funktionierender Narrowing-Workaround
(beide Zweige `SCALAR_NARROW_NUM`/`SCALAR_NARROW_ARR` mit `Equal` gepinnt); der alte `[1]`-Wrap-
Workaround-Pfad kompiliert weiter; `div(nd)` bleibt generisch/unbeeinflusst. `mean`-Gruppe (9
benannt + 1 Direktive = 10): niladisch → `NDArray<[]>`-artig wie `sum` (D7 explizit gefordert,
eigener Pin `MEAN_FLAT`), plus vier Basis-Wiring-Sanity-Pins (positive Achse, keepdims, negative
Achse — `MEAN_AXIS1`/`MEAN_AXIS1_KEEP`/`MEAN_NEG`), vier Degradations-Wiring-Pins exakt nach
Spec-Vorgabe (dynamische Achse, Mixed-Rank, Union-Achse, keepdims-Union —
`MEAN_DYN_AXIS`/`MEAN_MIXED_RANK`/`MEAN_UNION_AXIS`/`MEAN_KEEP_UNION`), ein OOB-`@ts-expect-error`
plus dessen Message-Gleichheits-Pin (`MEAN_AXIS_OOB_MSG`, `Guard<ReduceAxis<…>,…>` + `Equal<>`,
wortgleich zum `reduce:`-Stem) — re-litigiert NICHT die 15-Pin-Union-Achsen-Familie der
`sum`-Maschinerie. **Bewusste Abweichung von der „≈4-6"-Schätzung** des Spec-Addendums: die
`mean`-Gruppe liegt bei 10 statt 4–6, weil D7 den niladischen Pin EXPLIZIT UND SEPARAT verlangt
(nicht Teil der „4-6"-Degradationsfacetten) und weil „argmax-Muster" — wörtlich als Vorbild benannt
— selbst rund zehn Pins (positiv + Degradation + OOB-Message) trägt, nicht nur vier; die vier von
der Spec namentlich verlangten Degradationsfacetten (dyn axis/union axis/mixed rank/keepdims-union)
sind alle enthalten, die zusätzlichen Pins sind additive Basis-Sanity, keine Wiederholung der
15-Pin-Familie. Beide `@ts-expect-error`-Direktiven per Mutationsprobe (Direktive entfernt →
echter TS2769/TS2345 mit der vorhergesagten Message) non-vakuös bewiesen, danach exakt
wiederhergestellt (`diff` bestätigt Byte-Identität zur Vor-Mutations-Version).

## Byte-Erhaltungs-Nachweis der vier D6-v2-Body-Verschiebungen

Geprüft via `git diff -- spike/src/ndarray.ts`: für jede der vier Methoden (add/sub/mul/div)
erscheinen die drei Zeilen der ursprünglichen Rumpf-Logik (`const o = other as unknown as
NDArray<B>;`, der `elementwiseBinary(...)`-Aufruf, das `return new NDArray<OkShape<...>>(...)`)
im Diff ALS REINE KONTEXTZEILEN — kein `+`/`-` davor, also byte-für-byte identisch zum
Vorher-Zustand, nur an eine neue Position (in den `else`-Zweig der neuen union-typisierten
Implementierungssignatur) verschoben. Ergänzend `pnpm check` grün direkt nach der Konvertierung
(vor jeder weiteren Änderung) — bestätigt, dass alle Bestands-Call-Sites (add.test.ts, sum.test.ts,
alle `.test-d.ts`-Dateien, `spike/demo.ts`, `examples/rag-demo/main.ts`, `spike/tests-browser/
smoke.test.ts`) weiterhin kompilieren, ohne dass eine einzige davon einen Skalar-Aufruf verwendet
(alle bestehenden `.add()`/`.sub()`/`.mul()`/`.div()`-Aufrufe im Repo sind NDArray-Argument-Form,
per `grep` verifiziert).

## README/FOLLOWUPS/CLAUDE.md

README: die W1-Notiz („TypeScript-runtime surface only") im „What's implemented"-Abschnitt bekommt
einen eigenständigen Folge-Absatz für die W2-Ops — die bit-for-bit-Zeile im Usage-Codeblock bleibt
unangetastet und weiterhin korrekt (der Block ruft `.mul(y)`/`.div(scale)` nur mit NDArray-
Argumenten auf, nie mit einem Skalar). FOLLOWUPS: das W1-Paritätsitem („argmax/topk auf WNDArray/
Threaded nachziehen") ist um einen W2-Nachtrag erweitert (Skalar-Overload + `mean` auf `WNDArray`
fehlen ebenso). CLAUDE.md: Status-Einzeiler + „Aktuelle Pins & Gates" auf die neuen IST-Zahlen
aktualisiert.

## Offene Punkte

1. **Post-Verification-Addendum steht aus** — laut Eskalationsleiter (Stufe 3, substanzielle
   Scheibe) folgt der volle Verify-Katalog (A: Spec-Konformität, B: adversarial, C:
   covenant-verify) als separater Schritt; dieses Dokument wird dann um ein Addendum ergänzt
   (Konvention wie bei W1).
2. Kein `rsub`/`rdiv`, kein `.scale()`-Alias, keine `WNDArray`/Threaded-Parität — bewusst außerhalb
   des Scopes (Nicht-Ziele, Spec); FOLLOWUPS-Eintrag oben trackt Letzteres.
3. w4s bench:editor-Sonderdelta (+1,220 statt +1,181) ist attribuiert, aber nicht tiefer
   root-caused (welche genaue Guard-Diagnosepfad-Verzweigung die 39 zusätzlichen Instantiations
   kostet) — für ein 0,15-Promille-Detail am Rand eines informationellen Diagnosewerts nicht
   weiter verfolgt; kein Korrektheits- oder Gate-Risiko.

## Post-Verification-Addendum (2026-07-21)

Verify-Runde nach Template, Stufe 3 (A+B+C parallel) + covenant-check/lint (grün).
Baustein 0 hatte VOR dem Bau den TS2394-Blocker (Overload-vor-Body unmöglich) und die
Spiegel-Ambiguität gefangen (Spec v2).

- **Baustein A: CONFIRMED.** Alle D/T-Punkte; drei Pflicht-Mutanten beißen (÷n→*(1/n):
  67/482 rot inkl. Diskriminator-Assertion; op-Vertausch: 44 rot; Typ-Pin-Mutant: Exit 1);
  Byte-Erhaltung der vier Body-Verschiebungen am Diff bestätigt; Pin-Zahl-Abweichung
  (10 Wiring-Pins) als korrekt bewertet (argmax-Präzedenz zählt selbst 12).
- **Baustein B: HÄLT-mit-Befunden — mit EINEM echten MAJOR (F1), behoben:** Der
  Overload-Umbau ließ beim häufigsten Fehlerfall (simpler Broadcast-Mismatch, kein Union)
  die Shape-Message hinter dem Skalar-Decoy verschwinden („not assignable to type
  'number'") — TS meldet den Fehler des LETZTEN Overload-Kandidaten, und der war der
  Skalar. KEIN bestehender Pin fing das (@ts-expect-error prüft nur Fehler-Existenz).
  **Fix:** Deklarations-Reihenfolge getauscht (Skalar ZUERST, Guard-Träger ZULETZT) in
  allen vier Methoden — die `__shapeError`-Message erscheint wieder am Argument (nested
  unter dem unvermeidbaren TS2769-Kopf; Rest-Preis echter Overloads, offengelegt).
  **Neuer Pin:** „diagnostic quality (F1 pin)"-Test in scalar-mean.test.ts kompiliert
  eine Wegwerf-Fixture AUSSERHALB des Repos mit dem echten tsc und assertiert den
  Broadcast-Stem im Output; Nicht-Vakuität bewiesen (Reihenfolge geflippt → Test rot).
  Dafür drei schmale ambient.d.ts-Shims ergänzt (mkdtempSync/tmpdir/spawnSync, Haus-
  Disziplin) + .href-Form für fileURLToPath. Übrige B-Angriffe hielten (eigener
  200-Shape-mean-Differential, Spezialwert-Bits, 4/5 Mutanten gefangen — der fünfte
  beweisbar äquivalent via ES2019-Sort-Stabilität, Friction-Rückprobe bit-identisch).
- **Baustein C: kein Verstoß.** M1-v5-Bedingung erfüllt (FOLLOWUPS-Paritätslücke
  erweitert — erste Anwendung des neuen Wortlauts); M3-Byte-Verschiebung bestätigt;
  EIN Grenzfall (niedrig, „nicht entscheidbar"): die Union-über-Overload-Grenze-
  Ablehnung vs. M2-WORTLAUT — außerhalb der M2-Anker (keine Shape-Guard-Maschinerie),
  backend()-Präzedenz, KEINE neue Drift-Klasse → als Covenant-v6-Wortlaut-Kandidat in
  FOLLOWUPS getrackt, nicht still aufgelöst.
- **Prozess-Zwischenfall (offengelegt):** Beim Revert eines Orchestrator-Mutanten wurde
  versehentlich `git checkout -- spike/src/ndarray.ts` ausgeführt — bei uncommitteter
  Scheiben-Arbeit destruktiv (Datei fiel auf HEAD zurück; exakt der Fall, den die
  Template-Regel „Mutanten als revertierter EDIT, nie checkout" verhindert). Recovery:
  der W2-Implementierungs-Agent re-applizierte seine Änderungen byte-genau aus seinem
  Kontext (inkl. F1-Fix), verifizierte die Byte-Erhaltung erneut am Diff und bestätigte
  den F1-Pin als grün. Lektion in die KB kapturiert.
- **Finale Zahlen (nach F1-Fix + Pin-Test + ambient-Shims; je 2× deterministisch):**
  check:diag **188,563 @ 137** (= +4,233 zur W1-Baseline 184,330; der Haupttext-Wert
  190,092 war der Vor-Verify-Stand — Reihenfolgen-Tausch + Diagnose-Pin-Test + Shims
  netto −1,529) · stress **104,900 @ 82** (unverändert) · browser **2,142 @ 75** ·
  test:core **1,335** (+1 Diagnose-Pin) · bench:editor PASS (nur w4 26490→26453, 2×
  deterministisch, Pin aktualisiert) · Hash byte-identisch · lint 0/0 · test:example
  grün · GFM 0.
