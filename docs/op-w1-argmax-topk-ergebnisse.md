# Op-Scheibe W1: `argmax`/`topk` — Ergebnisse

Status: **abgeschlossen, dreifach verifiziert** (Post-Verification-Addendum am Ende).
Spec: docs/op-w1-argmax-topk-spec.md (Version 4). Datum: 2026-07-20.

## Summary

Die erste Op aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md, W1/F4 — argmax
trat zweimal auf, null Ersatz in der Surface) ist umgesetzt: `NDArray.argmax()` (Arity 0/1/2,
Overload-Muster von `sum` übernommen) und `NDArray.topk(k)` (Rang-1-only, `{values, indices}`
im `torch.topk`-Stil). D1: **NDArray-only** — kein WASM-Kernel, keine `WNDArray`/Threaded-Parität
in dieser Scheibe (bewusste, dokumentierte Surface-Asymmetrie, FOLLOWUPS-Eintrag unten). Alle
D8-Gates grün; der Haupt-`check:diag`-Pin zeigt sogar ein NETTO-DECREASE (Order-Noise-Effekt,
CLAUDE.md-präzedenzkonform); EIN Spec-Abweichungsbefund (stress-Korpus-Pin nicht exakt gehalten,
siehe unten) wird offen berichtet statt stillschweigend hingenommen.

## Umgesetzte Form je D-Punkt

- **D1 (Surface-Scope):** Nur `spike/src/ndarray.ts`. Kein Byte in `resident.ts`/`threaded.ts`.
  `argsort` nicht im Scope.
- **D2 (`argmax`):** Overloads exakt wie Spec — `argmax(): number`,
  `argmax<Axis>(axis): NDArray<OkShape<ReduceAxis<S,Axis,false>>>`,
  `argmax<Axis,KeepDims>(axis,keepdims): NDArray<OkShape<ReduceAxis<S,Axis,KeepDims>>>`.
  `ReduceAxis`/reduce.ts **unverändert** wiederverwendet (Import, keine Edits). Implementierung
  unterscheidet die ECHTE niladische Form (`arguments.length === 0`) von `argmax(undefined[, keepdims])`
  (1-/2-Arg-Form mit Achsenwert `undefined`) — siehe „Befund während der Implementierung" unten,
  ein während der Test-Erstellung gefundener und gefixter Bug.
- **D3 (`topk`):** `TopkCheck<S,K>` + `TopkShape<S,K>` an vector.ts appended. Finales Export-Set
  aus literal-arithmetic.ts: **`Compare`, `NonNegDigits`** (minimal, nur `export`-Präfixe, siehe
  Diff unten). `LiteralReshapeDimInvalid<[K]>` (bereits exportiert) klassifiziert negativ/
  Dot-Form-k in EINEM Check (Kern-08-Wiederverwendung, keine neue Arithmetik). `LiteralIndexBounds`
  wie gewarnt NICHT verwendet. `IsUnion<K>` wird UNCONDITIONAL zuerst geprüft (mirror der
  Union-Achsen-Policy in reduce.ts) — noch vor dem Rang-Check, exakt wie in der Spec-Warnung
  beschrieben.
- **D4 (Runtime-Semantik):** `argmaxRuntime`/`topkRuntime` an runtime.ts appended.
  `beatsMax(el,max) = (isNaN(el)&&!isNaN(max)) || el>max` — die gepinnte Totalordnung, geteilt
  von argmax und topks Comparator. Achsen-Normalisierung/Validierung ruft dieselbe Fehlermeldung
  wie `sumRuntime` (per Test WORTGLEICH bewiesen, nicht nur behauptet — siehe Test unten).
  Achsen-Dim-0 wirft zusätzlich (anders als sum) den empty-Stem. topk: Reihenfolge
  Rang→k-Gültigkeit→Bounds, `values[i] = data[indices[i]]` Float64Array-Element-Kopie (NaN-Payload
  erhalten).
- **D5 (Datei-Disziplin):** Siehe Diff-Nachweis unten — runtime.ts/vector.ts zeigen ausschließlich
  Additionen nach dem letzten Bestandscode (vector.ts brauchte dafür einen ZWEITEN,
  eigenständigen `import`-Block statt die bestehende Zeile zu erweitern — sonst hätte T5s
  „nur Additionen"-Anspruch nicht gehalten). ndarray.ts: Klassenkörper insertion-only (die zwei
  neuen Methoden ans Ende), die Import-ZEILEN am Dateikopf wurden erweitert (nicht Teil des
  Klassenkörpers, T5 verlangt dort nur „Klassenkörper insertion-only" — Präzedenz: `sum`s eigene
  runtime.ts-Importzeile ist über mehrere Kerne hinweg genauso gewachsen). reduce.ts/index.ts
  byte-unverändert. Kein neues File unter spike/src.
- **D6 (Tests):** `spike/tests-runtime/argmax-topk.test.ts` (30 Tests, in test:core registriert)
  + Appends an `spike/tests/ndarray.test-d.ts` (37 neue Typ-Pins). Details unten.
- **D7 (Pins & Budget):** Siehe Pin-Protokoll-Tabelle.
- **D8 (Gates):** Alle grün, siehe Gate-Tabelle.
- **D9 (Sprache):** Code/Kommentare/Tests Englisch, dieses Doc Deutsch, `≈` statt `~` (hier nicht
  gebraucht).

## Finales Export-Set (literal-arithmetic.ts)

Reiner `export`-Präfix-Diff, keine sonstige Zeile verändert (git diff, vollständig):

```diff
-type Compare<A extends string, B extends string> = LenCompare<A, B> extends infer LC
+export type Compare<A extends string, B extends string> = LenCompare<A, B> extends infer LC
...
-type NonNegDigits<T> = [T] extends [number]
+export type NonNegDigits<T> = [T] extends [number]
```

Zwei neue Exporte (vorher 7, jetzt 9): `Compare`, `NonNegDigits`. `IsUnion`/
`LiteralReshapeDimInvalid` waren bereits exportiert und werden nur importiert, nicht verändert.

## Pin-Protokoll (D7 v2, gestufte Attribution)

Baseline (frischer `git worktree` von HEAD 46ce403) **exakt reproduziert**:
`187,918 @ 135` · stress `102,877 @ 82` · browser `2,142 @ 75` (Exit 0, alle drei).

| Messpunkt | Dateien | Instantiations | Δ zum Vorpunkt | Attribution |
|---|---|---|---|---|
| Baseline | 135 | 187,918 | — | frischer Worktree |
| ① runtime.ts (argmaxRuntime+topkRuntime) + ndarray.ts (nur `argmax`) + literal-arithmetic-Exporte | 135 | 188,383 | **+465** | Runtime-Funktionen (fast kostenlos) + `argmax`-Overloads (reine ReduceAxis-Wiederverwendung) |
| ② + vector.ts-Maschinerie (`TopkCheck`/`TopkShape`) + ndarray.ts `topk`-Methode | 135 | 188,726 | **+343** | die neue Digit-Maschinerie (`Compare`+`NonNegDigits`-Sites) — in der erwarteten Spike-03-Größenordnung, NICHT LiteralRangeDim-schwer |
| ③a + NEUES Testfile, leer (`export {}`), registriert | 136 | 179,186 | **−9,540** | reiner Order-Noise (Datei-Hinzufügen reshuffelt die Fresh-vs-Cached-Partition, CLAUDE.md-dokumentiertes Phänomen — hier deutlich größer als das „±≈2,000"-Beispiel, aber dieselbe Mechanik) |
| ③b + Testfile GEFÜLLT (30 Tests, generische Helfer, `genShape`/`bruteArgmax`/`bruteTopk`) | 136 | 182,249 | **+3,063** | echte Typkosten des Testfiles selbst (Rest-Param-Inferenz, Rng-Typen, viele NDArray-Instantiierungen) |
| **final** + test-d.ts-Pins (37 neue) | 136 | **184,225** | **+1,976** | Typ-Pin-Katalog (T3) |

**Gesamtwachstum (final − Baseline, inkl. Order-Noise): 184,225 − 187,918 = −3,693** — eine
NETTO-ABNAHME, weit innerhalb des Absolut-Gates ≤ +12,000 (tatsächlich negativ). Determinismus:
final zweimal gemessen, byte-identisch (184,225 beide Male).

**stress**: Baseline 102,877 @ 82 → final **103,719 @ 82 (Δ +842)** — **Abweichung von T2s
„EXAKT unverändert"**, siehe „Offener Befund" unten.
**browser**: Baseline 2,142 @ 75 → final **2,142 @ 75 (exakt, Δ 0)** — T2 hier erfüllt.

## Befund stress-Korpus-Pin (+842) — AUFGELÖST per Spec v3 (T2-Korrektur)

> Auflösung (Orchestrator, nach Verify-Runde): T2s „stress EXAKT unverändert" war falsch
> spezifiziert — Content-Edits der geteilten Klassen-Surface sind laut Mess-Hausregel
> echte, legitime Typkosten, und stress/browser sind „ungated by design" (CLAUDE.md).
> Das Delta ist deterministisch (2× hier + 2× Verify-A + 2× Verify-B) und vollständig
> attribuiert (unten). Neuer stress-Pin: **103,719 @ 82**. Der ursprüngliche
> Befund-Text bleibt darunter unverändert als Primärquelle stehen.

`check:diag:stress` bewegte sich um **+842** (102,877 → 103,719 @ 82 Files, zweimal deterministisch
reproduziert). Root Cause bisektiert (temporärer Revert + Re-Messung, danach exakt wiederhergestellt
— `diff` gegen die volle Fassung bestätigt): **`argmax` allein +469, `topk`s inkrementeller Beitrag
+373** (469+373=842, exakt). Ursache: `NDArray`s Klassen-Member-Fläche wuchs um zwei überladene
generische Methoden — der stress-Korpus instanziiert `NDArray<S>` intensiv mit großen/vielen
literalen Shapes (Digit-Arithmetik-Grenzfälle), und JEDE solche Instanziierung muss jetzt
zusätzlich die neuen Overload-Signaturen auflösen, auch wenn kein einziger stress-Test
`argmax`/`topk` tatsächlich AUFRUFT. Das ist dieselbe Klasse Ripple, die `bench:editor`s W1–W7
(siehe unten) gleichmäßig traf — nur dass D7 diese Verschiebung für bench:editor EXPLIZIT erlaubt,
für stress/browser aber „EXAKT unverändert" verlangte (T2). Diese Annahme trägt empirisch NICHT
für stress (wohl aber für browser, das exakt hielt — vermutlich weil sein Korpus `NDArray<S>`
seltener/mit kleineren Shapes instanziiert).

**Nicht optimiert, nicht stillschweigend akzeptiert:** Der Code wurde NICHT verkleinert, um den
alten Pin-Wert zu erzwingen (verboten: „kein Optimieren ins Gate"). Der Betrag ist klein (≈0,8 %
des stress-Korpus, keine Rust-/M1-Fläche), aber die Owner/Verify-Runde sollte explizit
entscheiden, ob (a) der stress-Pin einfach aktualisiert wird (analog zu bench:editor, mit
demselben Ripple-Argument) oder (b) das als tolerierte, aber dokumentierte Abweichung von
Baustein 0s Annahme steht. Beide check:diag:stress/:browser-Zahlen sind laut CLAUDE.md
„ungated by design" (kein automatisches CI-Fail) — nur die SPEC (T2) hatte hier straffer
formuliert, als die Realität hergab.

## bench:editor: Vorher/Nachher (D7-erlaubte Verschiebung)

Alle sieben Workloads verschoben sich UNIFORM um **+804** (zweimal gemessen, byte-identisch),
Latenz/Correctness-Gate unverändert PASS:

| Workload | vorher | nachher | Δ |
|---|---|---|---|
| w1 | 24,305 | 25,109 | +804 |
| w2 | 26,114 | 26,918 | +804 |
| w3 | 57,254 | 58,058 | +804 |
| w4 | 24,466 | 25,270 | +804 |
| w5 | 29,759 | 30,563 | +804 |
| w6 | 30,929 | 31,733 | +804 |
| w7 | 23,477 | 24,281 | +804 |

`INSTANTIATION_PINS` in `spike/bench-dx/editor-latency.ts` aktualisiert (der einzige erlaubte
Bestandsdatei-Edit dort, D7). Hard-CI-Gate nach dem Update: **PASS**, Exit 0.

## Gates (alle frisch gemessen, dieser Commit)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (3-Verbund) | 0 Fehler root+stress+browser | 0 |
| `check:diag` | 184,225 @ 136 (s.o.) | 0 |
| `check:diag:stress` | 103,719 @ 82 (Δ+842, offen berichtet) | 0 |
| `check:diag:browser` | 2,142 @ 75 (exakt) | 0 |
| `test:core` | **852/852** (822 Baseline + 30 neu) | 0 |
| `test:resident` | 4,278 pass + 2 skip (unverändert) | 0 |
| `cargo test` | 161 passed (unverändert, kein Rust berührt) | 0 |
| `check:freeze` | Hash byte-identisch `0b9df4f1…2519c7d` | 0 |
| `bench:editor` | Hard-Gate PASS nach Pin-Update | 0 |
| `graph-a-lama query lint` | 0 Findings, 0 Errors/Warnings | — |
| `pnpm test:example` | PASS, weiterhin numtype@0.1.1 | 0 |

## Tests im Detail

**Runtime (`spike/tests-runtime/argmax-topk.test.ts`, 30 Tests):** Da D1 keine WASM-Gegenseite
vorsieht, gibt es keinen Differential-Partner — Coverage kombiniert stattdessen: handgerechnete
Fixtures für die semantischen Pins (NaN-maximal, First-Index-Wins bei Werte-/±0-/NaN-Ties, so
konstruiert, dass ein plausibler Mutant — `>=` statt `>`, Last-statt-First-Index — den Test
kippen würde); eine Wort-für-Wort-Stem-Gleichheitsprobe (`argmaxRuntime`s Achsen-Fehler und
`sumRuntime`s eigener Throw werden BEIDE gefangen und ihre `.message`-Strings direkt verglichen —
nicht zweimal von Hand abgetippt); unabhängig geschriebene Brute-Force-Referenzen (`bruteArgmax`/
`bruteTopk`, andere Code-Form als runtime.ts) über ≈150 Zufallsfälle je Op inkl. NaN-Injektion
(`genDataSpecial`); strukturelle keepdims-Invarianten (mirrors `keepdims.test.ts`s Nicht-Vakuität);
transponierte/gesliceste Empfänger (Materialisierungs-Annahme, Daten unabhängig via
`transposeRuntime`/`sliceRuntime` hergeleitet, nicht durch erneuten `.transpose()`/`.slice()`-Aufruf).

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, +37 Pins):** exakte Tupel für alle drei Formen;
`argmax()` → `number`; `argmax(undefined)` vs. echtes `argmax()` (1-Arg- vs. 0-Arg-Overload,
D2-Unterscheidung — siehe Bugfix unten) korrekt getrennt gepinnt; Degradationen (dynamische Achse,
Mixed-Rank, Union-Achse, Union-`keepdims`) — dieselbe `ReduceAxis`-Maschinerie wie `sum`, hier
NUR die Verdrahtung bewiesen, nicht die Regel neu hergeleitet; topk: `k=0`/`k=D`-Valid-Grenzen,
`k<0`/`k=1.5`/`k>D` als `@ts-expect-error` AM `k`-Argument (Position empirisch geprüft — je ein
Mutant „Direktive entfernt" zeigte den ECHTEN TS2345, mit exakt der vorhergesagten Message),
Rang≠1-Ablehnung (rank-0 UND rank-2), wide/Union-`k`-Degradation (inkl. eines gemischt-invaliden
Union-`k`), MAX_SAFE_INTEGER-Kante (`9007199254740991` bleibt noch ein echter, provabler Fehler;
`1e21` degradiert korrekt zu no-claim), vier Message-Gleichheits-Pins via `Guard<TopkCheck<…>,…>`
+ `Equal<>` (rank/negativ/dot-form/bounds — direkter Beweis der Compile-Message, nicht nur „ein
Fehler existiert irgendwo"). Nicht-Vakuität der `@ts-expect-error`-Pins UND der `Equal`-Pins durch
gezielte Mutationsproben verifiziert (falscher erwarteter Shape → TS2344; entfernte Direktive →
echter TS2345 mit der vorhergesagten Message), danach exakt wiederhergestellt (`diff` bestätigt).

## Befund während der Implementierung: `argmax(undefined, true)` Bug (gefunden + gefixt)

Beim Schreiben der Runtime-Tests fiel auf: die erste Implementierung prüfte
`axisNum === undefined`, um die niladische (`number`-zurückgebende) Form von der 1-/2-Arg-Form
zu unterscheiden — das ist FALSCH, weil `argmax(undefined, true)` (2 Argumente, Achsenwert
`undefined`) denselben Check trifft wie das echte `argmax()` (0 Argumente) und still das
`keepdims`-Argument verwarf (fiel in den `number`-Zweig statt `NDArray<[1,1,1]>` zu bauen).
TypeScripts eigene Overload-Auflösung unterscheidet nach ARGUMENT-ANZAHL an der Call-Site, nicht
nach dem Wert — die Implementierung muss das auch tun. Fix: `arguments.length === 0` statt
`axisNum === undefined`. Bewiesen via Test (`argmax(undefined, true)` → `NDArray<[1,1,1]>`,
Daten identisch zu `argmax()`), reproduzierbar VOR dem Fix rot (`assertShapeEqual`: „actual is
not iterable", weil eine `number` statt eines `NDArray` zurückkam).

## Befund: JIT-Nichtdeterminismus bei handkonstruierten Nicht-kanonischen NaN-Payloads (Test-Gotcha, kein Produktbug)

Beim Bau des `values[i] === data[indices[i]]`-byte-exakt-Tests mit einer bewusst NICHT-kanonischen
NaN-Payload (`0x7FF800000000DEAD` statt der üblichen `0x7FF8000000000000`) zeigte sich: der
bestehende `bitsOf`-Helfer (`assert-helpers.ts`, `new BigUint64Array(new Float64Array([x]).buffer)`)
liefert für DIESEN speziellen Payload nach mehreren vorherigen Aufrufen IM SELBEN Testfile
gelegentlich die kanonische statt die tatsächliche Payload zurück — bisektiert (isolierte
Repro-Skripte mit/ohne die vorangehenden `bitsOf`-Aufrufe) auf eine V8-JIT-Tier-Eigenheit von
`new Float64Array([x])` (Array-Literal-Konstruktion), NICHT auf `topkRuntime`s eigene
`Float64Array`-Element-Kopie (die separat, per direktem `DataView` über den BACKING BUFFER
verifiziert, IMMER korrekt ist — 5/5 Läufe). Der Test wurde auf einen lokalen `bitsAt`-Helfer
umgestellt (`DataView` direkt über `arr.buffer`, keine Array-Literal-Zwischenkonstruktion) — seither
5/5 (und alle 30 Tests über 5 volle Testfile-Läufe) deterministisch grün. `assert-helpers.ts`
selbst wurde NICHT geändert (kein bestehender Nutzer dort konstruiert absichtlich
nicht-kanonische Payloads). Kandidat für eine KB-Notiz (falls der Owner das für allgemein genug
hält): „hand-constructed non-canonical NaN payloads round-tripped through `new Float64Array([x])`
sind unter V8 nicht JIT-tier-stabil — lies stattdessen direkt aus dem Backing-Buffer, wenn ein
Test einen EXAKTEN NaN-Bitpattern-Claim braucht."

## README

**T7 erfüllt via eigenem kurzen Absatz**, NICHT im bestehenden Usage-Code-Block (der die
„exercised bit-for-bit against the WASM backend"-Behauptung trägt, die für argmax/topk FALSCH
wäre — D1: kein Kernel). Ort: der bereits bestehende „What's implemented"-Abschnitt (reine
Op-Surface-Prosa, kein Code-Beispiel, keine bit-for-bit-Aussage) bekommt einen neuen,
eigenständigen Satz direkt nach der Op-Liste:

> **`argmax`/`topk`** (ranking primitives — index of the maximum element, and the top-`k`
> values + indices of a 1-D vector) are also available on `NDArray`, but — unlike every op
> listed above — **TypeScript-runtime surface only, no WASM kernel yet**: a deliberate,
> disclosed surface asymmetry, not an oversight.

Die bit-for-bit-Zeile selbst bleibt unangetastet (weiterhin korrekt für die dort demonstrierten
Ops).

## FOLLOWUPS

Ein neuer Eintrag: „argmax/topk auf WNDArray/Threaded nachziehen" (D1-Surface-Asymmetrie, siehe
FOLLOWUPS.md).

## Offene Punkte

1. stress-Pin-Abweichung — AUFGELÖST (Spec v3, Abschnitt oben; neuer Pin 103,719 @ 82).
2. Post-Verification-Addendum — ERLEDIGT (Addendum unten; A CONFIRMED, B HÄLT-mit-
   Befunden, C kein Verstoß).
3. Kein `-1`-artiges Auto-`k`, kein `argsort`, kein `topk`-mit-Achse — bewusst außerhalb des
   Scopes (Nicht-Ziele, Spec). Nachgelagerte FOLLOWUPS aus der Verify-Runde: WNDArray/
   Threaded-Parität (inkl. Partial-Selection-Perf-Notiz, B-F5), Runtime-Achsen-Integer-Guard
   (B-F1, vorbestehend), COVENANT-M1-Präzisierung für kernel-lose Referenz-Ops (C-Empfehlung).

## Post-Verification-Addendum (2026-07-20)

Verify-Runde nach docs/verify-runde-template.md, Stufe 3 (A + B + C parallel, drei frische
Kontexte) + mechanisches covenant-check.sh/lint (grün, 0 Befunde). Baustein 0 lief VOR der
Implementierung (Spec-Addendum: 1 BLOCKER — fehlende Exporte — Owner-entschieden aufgelöst,
1 MAJOR-Warnung LiteralIndexBounds, die die Implementierung nachweislich befolgt hat).

- **Baustein A (Spec-Konformität): CONFIRMED.** Alle D1–D9/T1–T7 einzeln bestätigt; alle
  Gates frisch + doppelt gemessen (deterministisch); Attributions-Tabelle nachgerechnet
  (Summen exakt). Zwei Pflicht-Mutanten beißen: (a) NaN-Regel in beatsMax entfernt →
  test:core 848/852, exakt die 4 erwarteten Assertions rot, Revert hash-identisch;
  (b) k=D-Valid-Pin auf k=D+1 → pnpm check Exit 1 mit wortgleicher Bounds-Message.
  Selbstbefunde nachgeprüft: arguments.length-Diskriminierung korrekt (Test + Pin),
  NaN-Payload-Testhelfer 5× deterministisch. Minors: Import-Zeilen-Auslegung von D5
  (offen deklariert; künftige Specs sollten die Grenze Klassenkörper/Importzeilen
  explizit ziehen), Ergebnisse-Doc-Versionsverweis (behoben), roadmap.md-Ergänzung als
  dokumentiertes Ermessen akzeptiert.
- **Baustein B (adversarial): HÄLT-mit-Befunden.** Gehalten: eigener 220-Shape-Differential
  (≈2450 Fälle, 0 Abweichungen), eigene Totalordnungs-Proben (NaN/±0/Infinity/Rang 0/
  size-0), Stem-Wortgleichheit char-für-char, Baseline unabhängig im frischen Worktree
  reproduziert, bench:editor 2× byte-identisch, 4/5 breite Mutanten gefangen (der 5. —
  values aus stabil sortiertem Array statt data[indices[i]] — ist beweisbar äquivalent,
  ES2019-Sort-Stabilität; redundante Abdeckung, kein Bug), Friction-Rückprobe: der
  F4-Workaround aus examples/rag-demo ist mit topk(2) real ersetzt, literale [2]-Shapes.
  Befunde: **F1 (MODERAT, vorbestehend)** non-integer-Achse → stille Falschwerte in
  sumRuntime, von argmaxRuntime D4-treu reproduziert → FOLLOWUPS-Mini (kein In-Slice-Fix,
  Bestandscode-Disziplin). **F2 (NIEDRIG-MODERAT)** TopkChecks RankUnknowable-Kurzschluss
  überspringt auch D-unabhängige k-Checks — die Implementierung folgt damit der
  D-V1.3-Hauspolitik (uniform degrade); die Spec-Formulierung war die Abweichung →
  Spec v4 korrigiert D3, dokumentierender Policy-Pin ergänzt (TOPK_MIXEDRANK_NEG_K_*);
  die Position des IsUnion<K>-Erstfilters ist (anders als bei reduce.ts) von keiner
  Pin-Kante erzwungen — dokumentiert akzeptiert. **F3 (niedrig)** TopkShape ohne
  RankUnknowable-Check ist korrekt-per-Konstruktion (Kommentar begründet; kein Fix).
  **F4 (Nit)** Doku-Drift v2/v3 → behoben (dieses Doc auf v4 synchronisiert).
  **F5 (Nit)** topk-Vollsortierung O(n log n) → FOLLOWUPS-Notiz am Paritäts-Item.
  **F6** war ein Endianness-Fehler in Bs EIGENER Probe (transparent berichtet, kein
  Produktbefund).
- **Baustein C (covenant-verify): kein Verstoß.** M1 hält per Vertragstext (gerichtete
  Aussage über existierende Kerne; kernel-lose Referenz-Ops nicht verboten) — mit
  expliziter Owner-Empfehlung zur M1-Präzisierung vor W2–W5 (FOLLOWUPS). M2 an allen
  neuen Kanten bis in die Degradationsketten zurückverfolgt; Export-Präfixe semantisch
  neutral bestätigt; M3-Stems wortgleich; M4/M5/Z1/Z2/Nicht-Ziele sauber. Ein niedriger
  Erb-Befund (Whole-Shape-Union-Kante, geteiltes vorbestehendes FOLLOWUPS-Thema).
- **Post-Verify-Deltas dieser Abschluss-Runde:** zwei Policy-Pins in ndarray.test-d.ts
  (F2) → Haupt-Pin final **184,330 @ 136** (2× deterministisch; +105 zu 184,225),
  stress/browser unverändert 103,719 @ 82 / 2,142 @ 75; `pnpm check` grün (ein
  Namenskollisions-Zwischenstand `mixedRankRecv` → `topkMixedRankRecv` sofort behoben).
