# Op-Scheibe W1: `argmax` / `topk` — bindende Spec

Status: **bindend** (Owner-Auftrag 2026-07-20: „bau die erste Op-Scheibe: W1 argmax/topk";
Evidenzgrundlage: docs/dogfooding-rag-ergebnisse.md W1/F4 — zweimal aufgetreten, null Ersatz
in der Surface).
Version: 4 (v2 nach Baustein 0 + Owner-Entscheid „Exporte ergänzen"; v3 = T2-Korrektur
nach Implementierungs-Befund; v4 = D3-RankUnknowable-Korrektur nach Verify-B F2, siehe
Nachträge im Addendum) ·
Datum: 2026-07-20 · Eskalationsleiter: **Stufe 3** (voller Verify-Katalog).

## Ziel & Warum

Die erste Op aus der Dogfooding-Wunschliste: Ranking-Primitiven, mit denen eine
Retrieval-Anwendung die NDArray-Welt nicht mehr verlassen muss. Referenz-Friction:
examples/rag-demo/main.ts (Query-Ranking + Mean-Pooling-Max-Suche). Es gilt die
Produkt-Identität: Typ-Ebene „never wrong, only incomplete" (M2), Fehler am Argument mit
Runtime-wortgleichem Stem (M3), Zero-Dep (Z1).

## Berührte Covenant-Invarianten

- **M2** (Anker dim.ts, literal-arithmetic.ts, sym:Guard, sym:OkShape): neue Guards/Shapes
  folgen exakt den etablierten Degradationsregeln (Union-Filter VOR naked-extends,
  RankUnknowable-Gate, wide→no-claim, tuple-wrapped Checks). Keine neue konfident-falsche
  Kante.
- **M3** (sym:Guard, sym:ShowShape): Compile-Fehler erscheinen AM Argument; Message-Stems
  wortgleich zum Runtime-Throw; Klassen-Hover bleiben `NDArray<[…]>`.
- **M1**: NICHT berührt — diese Scheibe fügt KEINEN WASM-Kernel hinzu; runtime.ts erhält
  reine Referenz-Funktionen ohne Kernel-Gegenstück (M1 bindet Kernel an Referenz, nicht
  umgekehrt). Artefakt-Hash-Pin muss byte-identisch bleiben (`check:freeze`).
- **M4**: NICHT berührt (kein Byte Rust). **M5**: NICHT berührt (keine node:*-Imports).
- **Z1**: NICHT berührt (keine Dependencies). **Z2**: neue Testdatei liegt im Root-Korpus →
  automatisch von `pnpm check` gedeckt.

## Bindende Entscheidungen

- **D1 — Surface-Scope:** NUR die JS-Klasse `NDArray` (spike/src/ndarray.ts). WNDArray/
  WasmBackend/Threaded erhalten die Ops in dieser Scheibe NICHT (kein Kernel, keine
  Kopier-Implementierung) — bewusste, dokumentierte Surface-Asymmetrie; FOLLOWUPS-Eintrag
  „argmax/topk auf WNDArray nachziehen" wird Teil der Scheibe. `argsort` ist NICHT im
  Scope (Evidenz war top-k, nicht Vollsortierung; Nicht-Ziel unten).
- **D2 — API-Form `argmax`:** exakt das `sum`-Muster (Arity-Overloads 0/1/2, KEINE
  optionalen Parameter in den Mehr-Argument-Formen — Item-11/M2-Lektion), mit EINER
  bewussten Abweichung: die niladische Form gibt `number` zurück (nicht `NDArray<[]>`).
  - `argmax(): number` — Index ins ROW-MAJOR-FLATTENING über alle Elemente (NumPy
    `np.argmax(a)` ohne axis). Niladisch = kein Guard möglich (norm()-Präzedenz,
    ndarray.ts:440-446); size-0 ist reiner Runtime-Throw, kein Compile-Claim.
    Scalar-Consumer-Präzedenz: dot/norm/cosineSimilarity geben bereits `number`.
  - `argmax<const Axis>(axis: Guard<ReduceAxis<S, Axis>, Axis>): NDArray<OkShape<ReduceAxis<S, Axis, false>>>`
  - `argmax<const Axis, const KeepDims extends boolean | undefined>(axis, keepdims): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>`
  - `ReduceAxis`/`ReduceAxisKeepDims` werden UNVERÄNDERT wiederverwendet (reduce.ts wird
    NICHT editiert — der Union-Axis-Filter dort ist positionskritisch und bleibt unberührt).
    Ergebnis-DATEN sind f64-Integral-Indizes (numtype ist f64-only; dokumentiert im Doc-Kommentar).
- **D3 — API-Form `topk`:** Rang-1-only (DotCheck-Familie).
  `topk<const K extends number>(k: Guard<TopkCheck<S, K>, K>): { values: NDArray<OkShape<TopkShape<S, K>>>; indices: NDArray<OkShape<TopkShape<S, K>>> }`
  (torch.topk-Form: values UND indices — Retrieval braucht beide; es gibt kein gather).
  - `TopkCheck<S, K>` + `TopkShape<S, K>` werden an **spike/src/vector.ts** APPENDED
    (DotCheck-Familie, kein neues Source-File). Wiederverwendung der EXISTIERENDEN
    Digit-Maschinerie aus literal-arithmetic.ts — keine Neu-Herleitung von Arithmetik.
    **v2 (Baustein-0-BLOCKER, Owner-entschieden):** Die benötigten Primitiven (`Compare`
    u. a.) sind dort bisher NICHT exportiert. Auflösung: literal-arithmetic.ts erhält
    **reine Export-Sichtbarkeits-Edits** — `export ` wird bestehenden `type`-Zeilen
    vorangestellt, MINIMALES Set (nur was TopkCheck wirklich braucht; finales Set im
    Ergebnisse-Doc listen), NULL sonstige Änderungen an der Datei (semantisch neutral;
    die Datei ist M2-Anker → covenant-verify prüft den Diff mit). D5 ist entsprechend
    angepasst. **WARNUNG (Baustein-0-MAJOR, verbindlich):** `LiteralIndexBounds` darf
    für die k-Bounds NICHT verwendet werden — es hat für topk in BEIDEN Richtungen die
    falsche Semantik (klassifiziert k=D als „out" (Index-Semantik i<d) und negatives k
    als „in" (NumPy-Negativ-Normalisierung)); empirisch bewiesen. Die Akzeptanz-Pins
    „k=D valide" und „negatives k abgelehnt" (T3) machen diesen Fehlgriff mechanisch
    unüberlebbar.
  - Compile-Fehler AM `k`-Argument (Spike-03-Idiom) NUR für garantierte Runtime-Throws:
    Rang(S) ≠ 1 bei literalem Rang (DotCheck-Präzedenz: der Fehler eines
    Empfänger-Problems erscheint am Argument); literales k mit Dot-Form (non-integer,
    sign-agnostisch), negatives literales k, literales k > literale Dim D. Grenzfälle:
    k = 0 und k = D sind VALID (kein Fehler).
  - Degradation (M2), **v4-korrigiert:** wide `number`-k → `readonly [number]`; Union-k →
    `readonly [number]` (no-claim, spiegelt die Union-AXIS-Policy — Begründung wie
    reduce.ts: der Filter läuft, bevor ein ShapeError entsteht); `RankUnknowable<S>` →
    **UNIFORM no-claim für den GESAMTEN Check** (auch Dot-Form-/Negativ-k werden dort
    nicht mehr statisch beansprucht — das ist die D-V1.3-Hauspolitik aller sieben
    Rank-Gates: unknowable Empfänger degradieren uniform, keine Partial-Claims; die
    v1–v3-Formulierung „D-unabhängige Checks bleiben prüfbar" widersprach dieser Politik
    und ist gestrichen; Runtime-Backstop bleibt maßgeblich, dokumentierender Policy-Pin
    in ndarray.test-d.ts); wide Dim D → kein k-Bounds-Claim; Shape-Fehler-Kanten nie
    „confidently wrong". MAX_SAFE_INTEGER-Kappe der Digit-Maschinerie gilt (jenseits →
    no-claim). Der `IsUnion<K>`-Erstfilter ist defensiv UND semantisch notwendig für
    Mixed-Unions (ein `-1|2` darf weder akzeptiert-mit-Claim noch abgelehnt werden);
    seine POSITION ist — anders als bei reduce.ts — durch die aktuelle Pin-Suite nicht
    als load-bearing bewiesen (Verify-B F2, dokumentiert akzeptiert).
- **D4 — Runtime-Semantik (gepinnt, alle Formen):**
  - **Totalordnung:** NaN gilt als MAXIMAL (NumPy-argmax-Verhalten); bei mehreren NaN
    gewinnt der ERSTE Index; bei Wert-Gleichheit (inkl. `0 === -0`, kein Object.is)
    gewinnt der ERSTE Index. Ein Element schlägt das aktuelle Maximum gdw.
    `(isNaN(el) && !isNaN(max)) || el > max`.
  - `argmax()` niladisch: size 0 → Throw, Stem `argmax: attempt to get argmax of an empty
    array` (in Anlehnung an NumPy). Rang 0 (`[]`, 1 Element) → `0`.
  - `argmax(axis)`: Achsen-Normalisierung/Validierung EXAKT wie sumRuntime — der Throw-Stem
    ist WORTGLEICH `reduce: axis ${axis} is out of range for shape [${shape}] (rank
    ${rank})` (runtime.ts:222), weil der wiederverwendete ReduceAxis-Guard genau diesen
    Stem als Compile-Message trägt (M3). Achsen-Dim 0 → Throw mit dem empty-Stem.
    keepdims via bestehendem `keepDimsShape` (runtime.ts:532) — dessen dokumentierter
    Kontrakt „Call-Sites prävalidieren" wird eingehalten (Validierung VOR dem Aufruf;
    v2-Korrektur: es existieren bereits DREI Call-Sites (ndarray.ts:419,
    resident.ts:835/857), argmax wird die vierte — der offene FOLLOWUPS-Mini
    „defensiver Achsen-Assert" bleibt offen und unverschärft, keine Edits an
    bestehenden runtime.ts-Funktionen).
  - `topk(k)`: Runtime-Validierung mit Compile-wortgleichen Stems: Rang ≠ 1 → Throw
    (Stem-Familie analog assertVectorPair, eigener `topk`-Prefix); k non-integer /
    negativ / k > Länge → Throw. k = 0 → leere `[0]`-Arrays (valid). Ordnung: Indizes
    sortiert nach (NaN zuerst — untereinander nach aufsteigendem Index, dann Wert
    absteigend, dann Index aufsteigend); `values[i] === data[indices[i]]` (byte-exakt,
    inkl. NaN-Payload-Durchreichung via Float64Array-Kopie).
- **D5 — Datei-Disziplin (Pin-Schutz + Freeze), v2:** KEIN neues Source-File unter
  spike/src (Appends an runtime.ts + vector.ts, Insertion-only-Append ans Ende des
  NDArray-Klassenkörpers in ndarray.ts; NULL Edits an bestehenden Membern/Funktionen/
  Typen; reduce.ts wird NICHT editiert, nur importiert). AUSNAHME (v2, Owner-entschieden):
  literal-arithmetic.ts erhält AUSSCHLIESSLICH `export `-Präfixe vor bestehenden
  `type`-Deklarationen (minimales Set, keine Zeile sonst verändert — der Diff dieser
  Datei zeigt NUR Zeilen, deren einzige Änderung das vorangestellte `export ` ist).
  index.ts (Barrel) bleibt UNVERÄNDERT (DotCheck ist dort auch nicht exportiert —
  Präzedenz; TopkCheck/TopkShape bleiben intern, die Methoden-Signaturen tragen sie).
  NDArrayView bleibt unberührt (Hausregel: nie S-konsumierende Member).
- **D6 — Tests:**
  - Runtime: EIN neues File `spike/tests-runtime/argmax-topk.test.ts`, registriert in der
    test:core-Explizitliste (test-scripts-guard). Pflichtfälle: Rang 1/2/3, negative Achse,
    keepdims (Shape UND Daten-Identität zur non-keepdims-Form), NaN-Fälle (einzeln, mehrere,
    NaN-vs-NaN-Reihenfolge), ±0-Tie, Wert-Ties (erster Index), size-0-Throws (niladisch,
    Achsen-Dim 0), Wortgleichheits-Pins der Throw-Stems, topk: k=0/k=D/k>D/non-integer/
    negativ/NaN-Ordnung/Tie-Ordnung/values-indices-Konsistenz, transponierte/gesliceste
    Empfänger (Materialisierungs-Annahme der JS-Klasse explizit absichern).
  - Typ-Ebene: Pins werden an das BESTEHENDE `spike/tests/ndarray.test-d.ts` APPENDED
    (kein zweites neues File): exakte Ergebnis-Tupel für literale Shapes, `argmax()` →
    `number`, Union-/wide-/Mixed-Rank-Degradationen (Spiegel der sum-Pins inkl.
    keepdims-`boolean`-Union-Ehrlichkeit), topk-Pins ([k]-Shape, k>D/negativ/1.5 als
    `@ts-expect-error` AM Argument, wide k → `[number]`, Union-k-Degradation,
    Rang≠1-Ablehnung, MAX_SAFE_INTEGER-Kante), Hover-Sauberkeit via `Equal<…>`-Pins.
- **D7 — Pins & Budget (Mess-Hausregeln gelten):**
  - Baseline VOR der Implementierung im frischen Worktree des HEAD bestätigen
    (Erwartung: 187,918 @ 135 · stress 102,877 @ 82 · browser 2,142 @ 75).
  - Das EINE neue Testfile verschiebt den Haupt-Pin um Order-Noise → **empty-then-fill**:
    erst leeres `export {}`-File registrieren und messen (= Order-Noise-Anteil), dann
    füllen und messen (= echte Typkosten). Stress-/Browser-Pins müssen EXAKT halten
    (deren File-Sets sind unberührt). **v2 (Edit-Kosten-Attribution, Hausregel):** die
    Edits an bestehenden Dateien werden GESTUFT gemessen (Messpunkte: ① runtime.ts+
    ndarray.ts-Appends inkl. export-Präfixe, ② vector.ts-Typmaschinerie, ③ test-d-Pins);
    volle Kommentar-Kontrollproben nur, falls ein Messpunkt einen unerklärlichen Sprung
    zeigt. Erwartete Kostenbasis der k-Bounds nach Owner-Entscheid: bare `Compare`-Sites
    (Spike-03-Größenordnung, einige hundert Instantiations pro Site) — NICHT die
    schwerere LiteralRangeDim-Maschinerie.
  - **Vorregistriertes Absolut-Gate (Spike-04-Regel):** Gesamtwachstum des Haupt-Pins
    (Maschinerie + alle neuen Pins, inkl. Order-Noise) ≤ **+12,000 Instantiations**
    (≈ 0,24 % des 5M-Budgets); `bench:editor`-Latenz-Medians bleiben unter dem
    bestehenden 2x-Ceiling; Correctness-Workloads werfen weiter. Bei Riss: STOPP,
    Befund an den Owner, kein „Optimieren ins Gate".
  - `bench:editor`-Instantiation-Pins (W1–W7) dürfen sich durch die ndarray.ts-Surface-
    Erweiterung verschieben: neue Werte deterministisch doppelt messen, Pins im Harness
    aktualisieren, Vorher/Nachher im Ergebnisse-Doc. KEIN neuer Workload in dieser
    Scheibe (FOLLOWUPS-Kandidat: W8 = argmax/topk-Hover).
- **D8 — Gates der Scheibe:** `pnpm check` (Dreier-Verbund) · `check:diag`(+stress/browser,
  Pin-Protokoll D7) · `test:core` (inkl. neuem File; Zahl steigt von 822) · `test:resident`
  (unberührt grün) · `cargo test` (unberührt) · `check:freeze` (Hash-Pin byte-identisch) ·
  `bench:editor` (Pin-Protokoll D7) · `graph-a-lama query lint` · `pnpm test:example`
  (Example bleibt auf 0.1.1 — die neuen Ops erscheinen dort erst nach einem künftigen
  Release; Tripwire bleibt grün, weil Major.Minor unverändert) · GFM-Gate auf allen
  neuen/geänderten .md.
- **D9 — Sprache:** Code, Kommentare, Tests, Commit: Englisch. Spec + Ergebnisse-Doc:
  Deutsch. `≈` statt `~`.

## Akzeptanzkriterien

- **T1:** Alle D8-Gates grün mit berichteten Exit-Codes; Artefakt-Hash-Pin unverändert.
- **T2 (v3 korrigiert):** Pin-Protokoll vollständig: Baseline-Reproduktion,
  empty-then-fill-Zerlegung, neue Pins (Haupt/stress/browser) dokumentiert;
  Absolut-Gate ≤ +12,000 eingehalten. v1/v2 verlangten „stress/browser EXAKT
  unverändert" — das war FALSCH spezifiziert: die File-Sets dieser Korpora sind zwar
  unberührt, aber sie instanziieren die NDArray-Klasse, deren Member-Surface diese
  Scheibe erweitert — Content-Edits geteilter Quellen sind laut Mess-Hausregel echte,
  legitime Typkosten, und stress/browser sind „ungated by design" (CLAUDE.md).
  Korrigierte Anforderung: ein stress-/browser-Delta ist akzeptabel, wenn es
  DETERMINISTISCH reproduziert, per Bisektion der Scheibe ATTRIBUIERT und im
  Ergebnisse-Doc ausgewiesen ist; der neue Wert wird der Pin.
- **T3:** Typ-Pins beweisen: (a) exakte Shapes für literale Fälle auf allen drei Formen,
  (b) JEDE Degradationskante endet in no-claim, nie in einem falschen Literal-Claim,
  (c) `@ts-expect-error`-Pins sitzen AM Argument (Position empirisch geprüft),
  (d) Message-Stems compile ⇄ runtime wortgleich gepinnt (reduce-Stem für argmax-axis,
  topk-Stems für topk).
- **T4:** Runtime-Semantik-Pins: NaN-maximal/first-wins, ±0-Tie, first-index-Ties,
  empty-Throws, topk-Totalordnung, `values[i] === data[indices[i]]` byte-exakt —
  jeweils als Test mit Mutations-Nachweisbarkeit (Verify-A-Mutant).
- **T5:** Datei-Disziplin am Diff bewiesen: runtime.ts/vector.ts nur Appends (Diff zeigt
  ausschließlich Additionen NACH dem letzten Bestandscode), ndarray.ts-Klassenkörper
  insertion-only, reduce.ts/literal-arithmetic.ts/index.ts byte-unverändert, kein neues
  File unter spike/src.
- **T6:** test-scripts-guard grün mit registriertem neuen Testfile; FOLLOWUPS-Eintrag
  „WNDArray-Parität argmax/topk" angelegt; Doc-Platzierung nach Hausregel (CLAUDE.md
  Einzeiler + Pins, Projekt-Log-Append, Ergebnisse-Doc mit Addendum).
- **T7 (v2 präzisiert):** README-Erwähnung von argmax/topk NICHT im bestehenden
  Usage-Block — der trägt die Behauptung „Every operation below is exercised bit-for-bit
  against the WASM backend" (README.md:125-126), die für argmax/topk FALSCH wäre (D1:
  kein Kernel). Stattdessen: eigener kurzer Absatz/Unterabschnitt mit explizitem
  Hinweis „TypeScript-runtime surface only (no WASM kernel yet)", englisch; oder —
  falls kein sauberer Ort existiert — README unverändert lassen und die Entscheidung
  im Ergebnisse-Doc festhalten. Die bit-for-bit-Zeile darf nicht falsch werden.

## Nicht-Ziele

- Kein `argsort`, kein `gather`/`take`, kein `topk` mit Achsen-Parameter, kein
  Compile-Claim für size-0-Empfänger, keine WNDArray-/Threaded-Parität (FOLLOWUPS),
  kein WASM-Kernel (keine M1-Fläche), kein neuer bench:editor-Workload, kein Release/
  Publish (Example-Update folgt erst mit dem nächsten Release).
- Keine Edits an bestehenden Typen/Funktionen — insbesondere reduce.ts' positionskritischem
  Union-Filter und keepDimsShape.

## Verify-Plan (Stufe 3)

Baustein 0 (brainroute:deep, adversarial gegen DIESE Spec, inkl. empirischer Proben in
Scratch/Worktree: ReduceAxis-Wiederverwendbarkeit für argmax-Signaturen, TopkCheck-Skizze
gegen die echte Digit-Maschinerie, Overload-Auflösung, Message-Stem-Behauptung
runtime.ts:222) VOR der Implementierung. Danach A (Spec-Konformität, alle Gates, eigener
Mutant) + B (adversarial: Grenzfälle jenseits der Spec, Mutanten breit, Mess-Randbedingungen,
Hover-Qualität) + C (covenant-verify: M2/M3-Kanten am Diff, M1/M4-Unberührtheit,
Export-Edits an der M2-Anker-Datei) parallel; Aufträge aus docs/verify-runde-template.md.
Ergebnisse-Doc mit Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-20)

Verifier: brainroute:deep, frischer Kontext, empirische Typ-Proben im Scratch-Worktree
(Commit 46ce403; Haupt-Tree unberührt, worktree sauber entfernt). Befunde und Auflösung:

1. **BLOCKER (hoch):** Die v1-Annahme „TopkCheck importiert Compare & Co. aus
   literal-arithmetic.ts" war FALSCH — keiner der nötigen Typen ist exportiert
   (nur 7 Exports existieren; Import-Probe: 4× TS2459). **Owner-Entscheid 2026-07-20:
   Exporte ergänzen** (reine `export `-Präfixe, minimales Set) — in D3/D5 eingearbeitet;
   die Alternativen (LiteralRangeDim-Clamp-Komposition; k-Bounds streichen) wurden
   verworfen.
2. **MAJOR (hoch):** `LiteralIndexBounds` wäre der naheliegende, aber FALSCHE Baustein
   für die k-Bounds (k=D → „out", −1 → „in"; beides empirisch bewiesen). Als
   verbindliche Warnung + Akzeptanz-Pins in D3/T3 verankert.
3. **MINOR:** keepDimsShape hat bereits drei Call-Sites (nicht zwei) — D4 korrigiert.
4. **MINOR:** T7-Falle: README-Usage-Block trägt eine bit-for-bit-WASM-Behauptung, die
   argmax/topk nicht erfüllen — T7 präzisiert (eigener Absatz mit Caveat oder gar keine
   README-Änderung).
5. **MINOR/Nit:** Edit-Kosten-Attribution fehlte in D7 — gestufte Messpunkte ergänzt;
   Kostenbasis-Schätzung auf den Owner-entschiedenen Pfad (bare Compare) umgestellt.
**Nachtrag v3 (Implementierungs-Befund, 2026-07-20):** Die Implementierung legte offen,
dass `check:diag:stress` um +842 wandert (deterministisch; bisektiert: argmax +469,
topk +373 — Klassen-Member-Surface-Ripple in einem Korpus, dessen File-Set unberührt
ist). Der Implementierer hat korrekt NICHT ins Gate optimiert, sondern berichtet. T2
war in diesem Punkt falsch spezifiziert und ist in v3 korrigiert (Delta akzeptabel wenn
deterministisch + attribuiert + dokumentiert; stress/browser sind ungated by design).
Browser hielt exakt (2,142 @ 75). Neuer stress-Pin: 103,719 @ 82.

**Nachtrag v4 (Verify-Runde A/B/C, 2026-07-20):** Verify-B (F2) bewies empirisch, dass
`TopkCheck`s RankUnknowable-Kurzschluss AUCH die D-unabhängigen k-Checks überspringt —
die Implementierung folgt damit der D-V1.3-Hauspolitik (uniform degrade), die
v1–v3-Spec-Formulierung war die Abweichung. v4 korrigiert D3 auf die implementierte
(und haus-konsistente) Form; ein dokumentierender Policy-Pin (negatives k auf
Mixed-Rank-Empfänger kompiliert als no-claim) wurde ergänzt. Verify-B zeigte außerdem,
dass die POSITION des IsUnion<K>-Erstfilters (anders als bei reduce.ts) von keiner
Pin-Kante erzwungen wird — als dokumentierte Akzeptanz festgehalten, nicht als
Load-Bearing-Behauptung. Alle weiteren A/B/C-Befunde und ihre Auflösung: Ergebnisse-Doc,
Post-Verification-Addendum.

6. **Positiv verifiziert (empirisch):** argmax-Overload-Auflösung inkl. aller
   Degradationskanten (niladisch → number; `argmax(undefined)` → `NDArray<[]>`-Form;
   Union-/Dynamik-Degradation; out-of-range-Fehler AM Argument mit byte-gleichem
   `reduce:`-Stem zu runtime.ts:222); DotCheck-Präzedenz trägt den Rang≠1-Fehler ans
   k-Argument; JS-NDArray ist nach transpose()/slice() immer materialisiert-contiguous
   (transposeRuntime/sliceRuntime allozieren frisch); LiteralReshapeDimInvalid ist der
   korrekte exportierte Baustein für negativ/Dot-Form-k; Baseline-Pins im Worktree exakt
   reproduziert (187,918 @ 135 · 102,877 @ 82 · 2,142 @ 75); bench:editor-Pins sind ein
   editierbares INSTANTIATION_PINS-Literal (editor-latency.ts:692-699); ReduceAxis
   kollisionsfrei wiederverwendbar.
