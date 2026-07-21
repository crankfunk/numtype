# Op-Scheibe W4: `stack` — bindende Spec

Status: **bindend** (Owner-Programm 2026-07-20: W2–W5 gesetzt; Evidenz
docs/dogfooding-rag-ergebnisse.md W4/F5 — selbstgebauter `embedMatrix`-Flatten-Helper
der RAG-Demo).
Version: 2 (nach Baustein 0 — verbindliche Typ-Formen aus der verifizierten Skizze; Addendum unten) · Datum: 2026-07-21 · Eskalationsleiter: **Stufe 3**.
Covenant: v5 (kernel-lose Referenz-Ops zulässig, Paritätslücke in FOLLOWUPS).

## Ziel & Warum

`NDArray.stack([row0, row1, …])` — eine Matrix aus N unabhängig berechneten
Zeilenvektoren, der NumPy-Reflex `np.stack`/`np.array([...])`, den F5 als real, aber
niedrigprior belegt hat. Mit literalen Tupeln entstehen statische `[N, D]`-Shapes
(Tupel-LÄNGE = Rang-Arithmetik über kleine Ints — erlaubt laut TS-Limits-Regeln;
das ist die erste echte neue Typ-Ebenen-Substanz seit W1).

## Berührte Covenant-Invarianten

- **M2** (Anker dim.ts/Guard/OkShape): neue Guard-/Shape-Typen mit den etablierten
  Degradationsregeln — nie confidently wrong. **M3**: Fehler AM Argument,
  Stems wortgleich zum Runtime-Throw. **M1 (v5)**: kernel-lose Referenz (Append,
  FOLLOWUPS-Nachtrag W4). **M4/M5/Z1/Z2**: unberührt.

## Bindende Entscheidungen

- **D1 — Scope & Name:** STATISCHE Methode `NDArray.stack(rows)` — NUR Rang-1-Zeilen
  gleicher Länge → Rang-2-Ergebnis `[N, D]` (die F5-Evidenzform). KEIN allgemeines
  Achsen-stack beliebiger Shapes (keine Evidenz — Nicht-Ziel), kein `axis`-Parameter,
  kein `concat`. Name `stack` (np.stack-Affinität); `fromRows` nur als Doc-Erwähnung.
- **D2 — Typ-Ebene (Kern der Scheibe):** `StackCheck<Rows>` + `StackShape<Rows>`,
  APPENDED an spike/src/vector.ts (Rang-1-Familie; kein neues File).
  - TUPEL-Input mit uniform-literalem D (`[NDArray<[3]>, NDArray<[3]>]`) →
    `readonly [2, 3]` (N = Tupellänge via `Rows["length"]` — kleine Ints, erlaubte
    Rang-Arithmetik; D via Element-Extraktion).
  - Tupel mit BEWEISBAR verschiedenen literalen D (`[3]` vs `[4]`) → ShapeError AM
    Argument (garantierter Runtime-Throw), Stem wortgleich zur Runtime.
  - Tupel mit Rang≠1-Member bei literalem Rang → ShapeError am Argument.
  - Array-Input `readonly NDArray<[3]>[]` (Länge unbekannt) → ehrlich
    `readonly [number, 3]`; wide D → `readonly [number, number]`.
  - Degradation nach Hauspolitik: Union-Element-Typen, RankUnknowable-Member,
    wide D, LEERES Tupel-Literal (`[]` → garantierter Throw → ShapeError; leeres
    ARRAY zur Laufzeit = Runtime-Backstop), Mixed-Fälle → no-claim statt
    Falsch-Claim; tuple-wrapped Checks; `IsUnion`-Filter wo distributiv gefährlich.
    Die uniforme Degradations-Richtung im Zweifel (D-V1.3-Präzedenz).
  - Wiederverwendung: DimEq/IsDynamicDim/RankUnknowable/ShowShape aus dim.ts (alle
    exportiert); KEINE neuen literal-arithmetic-Exporte erwartet (reiner
    Dim-Vergleich, keine Arithmetik).
- **D3 — Runtime:** `stackRuntime(rows: ReadonlyArray<{ shape: readonly number[];
  data: Float64Array }>): { shape: number[]; data: Float64Array }` APPENDED in
  runtime.ts: Validierung (≥1 Zeile; jede Zeile Rang 1; alle Längen gleich) mit
  gepinnten Stems (`stack: expected at least one row` · `stack: expected 1-D rows
  (got shape […] at index i)` · `stack: row length mismatch (expected D, got X at
  index i)` — exakte Stems legt der Implementierer fest, Compile wortgleich);
  dann Row-major-Kopie (`Float64Array#set` mit Zeilen-Offset — exakt der
  embedMatrix-Algorithmus der Demo). NaN-Payloads byte-erhalten (Movement-Op).
  D=0-Zeilen sind VALIDE (`[[],[]]` → `[2, 0]`).
- **D4 — Methode:** `static stack<…>(rows: Guard<StackCheck<…>, …>): NDArray<OkShape<
  StackShape<…>>>` als Insertion NACH `fromArray` im statischen Block (bei den
  Konstruktoren — Auffindbarkeit) ODER am Klassenende (Implementierer-Wahl nach
  Insertion-Disziplin-Abwägung: neuer statischer Member kollidiert mit nichts;
  W1-Auslegung erlaubt Insertion an semantisch passender Stelle, solange kein
  Bestands-Member editiert wird — die Wahl im Ergebnisse-Doc begründen).
  `const`-Typparameter, damit Tupel-Literale ohne `as const` literal bleiben.
  Doc-Kommentar: np.stack-Bezug, Degradations-Kanten, F5-Evidenz, Surface-Hinweis.
- **D5 — Tests:** Runtime: W4-Block APPENDED an spike/tests-runtime/
  scalar-mean.test.ts (KEIN neues File; Kopf-Hinweis ergänzen): Stems wortgleich,
  2/3/1-Zeilen-Fälle, D=0, Längen-Mismatch-Throw, Rang≠1-Throw, leer-Throw,
  NaN-Payload-Bits durch stack, Round-trip gegen embedMatrix-Muster (flatten-Helper
  nachgebaut → byte-identisch — die F5-Schließungs-Rückprobe), großes N Smoke.
  Typ-Pins APPEND an ndarray.test-d.ts: exakte `[2,3]`/`[3,4]`-Formen, Mismatch/
  Rang≠1/leeres-Tupel als @ts-expect-error AM Argument mit Message-Equality-Pins,
  Array-Input → `[number, D]`, wide → `[number, number]`, Union-/RankUnknowable-
  Degradationen, Hover-Sauberkeit (Equal).
- **D6 — Gates & Pins:** wie W3-D5; Baseline 190,640 @ 137 · 104,900 @ 82 ·
  2,142 @ 75; kein neues File → keine Order-Noise; Absolut-Gate ≤ **+8,000**
  (echte neue Typ-Maschinerie mit Tupel-Rekursion über Rows — teurer als W3,
  billiger als W1s Digit-Arithmetik; Messung entscheidet); stress-Ripple möglich
  (statischer Member — beobachten, 2× + attribuieren); Mutanten NUR per
  Backup-Kopie-Verfahren. Sprache/GFM wie immer.

## Akzeptanzkriterien

- **T1:** Alle Gates grün (Standard-Katalog), Hash byte-identisch, Exit-Codes.
- **T2:** Pins 2× deterministisch, Absolut-Gate ≤ +8,000, Deltas attribuiert.
- **T3:** Typ-Pins decken JEDE D2-Kante; Fehler-Pins sitzen AM Argument mit
  wortgleichen Stems; keine confidently-wrong-Kante (Verify prüft adversarial,
  insbesondere Tupel-vs-Array-Inferenz und Mischformen).
- **T4:** Runtime-Pins inkl. NaN-Payload-Bits, D=0, F5-Rückprobe byte-identisch;
  Verify-Mutanten beweisen Nicht-Vakuität.
- **T5:** Datei-Disziplin (vector.ts/runtime.ts Appends, ndarray.ts Insertion, kein
  neues File außer Ergebnisse-Doc); FOLLOWUPS-Nachtrag W4; Doc-Platzierung nach
  Hausregel; README-Op-Notiz um stack ergänzt.

## Nicht-Ziele

Kein allgemeines stack höherer Ränge/mit axis, kein concat/vstack/hstack, kein
unstack/split, kein WNDArray/Threaded (FOLLOWUPS), kein Kernel, kein Release.

## Verify-Plan (Stufe 3)

Baustein 0 (brainroute:deep — Schwerpunkte: `const`-Tupel-Inferenz statischer
Methoden EMPIRISCH (bleiben `[NDArray<[3]>, NDArray<[3]>]`-Literale ohne as const
Tupel? Inferiert `Rows["length"]` das Literal N?), Tupel-vs-Array-Unterscheidung auf
Typ-Ebene (readonly array vs tuple — wie unterscheidet StackShape?), DimEq-Reuse-
Annahme, Stem-Design, Absolut-Gate-Plausibilität, Insertion-Ort D4) VOR der
Implementierung. Danach A+B+C parallel. Ergebnisse-Doc mit Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-21)

Verifier: brainroute:deep, frischer Scratch-Worktree, KOMPILIERENDE Skizze gebaut und
typ- wie laufzeitseitig verifiziert (inkl. echter Node-Ausführung von stackRuntime,
NaN-Bits, embedMatrix-Parität). Befunde → verbindliche v2-Formen:

1. **F1 (hoch): Schichtung.** StackCheck/StackShape operieren auf
   `Shapes extends readonly Shape[]` (vector.ts importiert nie NDArray —
   Zyklus-Gefahr, dim.ts-Kommentar-Präzedenz); der NDArray→Shape-Unwrap ist ein
   EIGENER benannter Typ `RowShapesOf<Rows>` in ndarray.ts.
2. **F2 (hoch, BLOCKER-Klasse): Invarianz-Kollaps.** `Rows[number] extends
   NDArray<infer S>` liefert auf heterogenen Tupeln NEVER (invarianter
   __variance-Marker + nicht-distributiver Indexed-Access; isoliert reproduziert).
   VERBINDLICH: homomorpher Mapped Type
   `{ [I in keyof Rows]: Rows[I] extends NDArray<infer S> ? S : never }` —
   pro Position konkret, nie Union.
3. **F3 (hoch): Leer-Tupel-Falle.** `never extends NDArray<infer S>` ist wahr →
   Fallback auf den Constraint statt never. VERBINDLICH: `Shapes["length"] extends 0`
   -Gate VOR jeder Element-Extraktion (Rows["length"] ist bei []-Literal exakt 0).
4. **F4: Akkumulator-Narrowing.** Conditional-Zweige narrowen Typ-Parameter nicht —
   `[Acc] extends ["none"]`-Tupel-Check + `Extract<Acc, Dim>` an der DimEq-Stelle.
5. **F5 (hoch): Array≠Tupel.** Tupel-Rekursion matcht echte Arrays NIE (empirisch:
   kein Head/Rest-Match) — eigener Array-Pfad, gated über
   `number extends Shapes["length"]`, mit direkter Element-D-Extraktion (ArrayRowD).
6. **F6 (entschieden): wide-D-Semantik = CompatDim-Präzedenz.** EIN dynamischer Dim
   auf IRGENDEINER Zeile weitet das GESAMTE D auf number (wide-Sentinel im Fold) —
   nie stilles Verbleiben beim Literal der Nachbarzeilen.
7. **F7 (entschieden): Array-Input mit uniform beweisbar falschem literalen Rang
   (`readonly NDArray<[2,3]>[]`) wird ABGELEHNT** — sound, weil auch das leere
   Array wirft („expected at least one row"): jeder Aufruf dieses statischen Typs
   ist ein garantierter Throw. Unknowable-Rang-Elemente → Pass/no-claim.
8. **F8 (Kante, pinnen): Array mit Union-Element-Typ** (`readonly (NDArray<[3]>|
   NDArray<[4]>)[]`) → per IsUnion-Filter auf dem Element-Typ VOR der Extraktion
   zu `readonly [number, number]` degradieren (no-claim); als Pin verankern.
9. **Positiv verifiziert:** const-Tupel-Inferenz trägt (heterogen bleibt Tupel,
   ohne const widened zu Array — const ist load-bearing); Guard am rows-Argument
   wortgleich (EINE Signatur → W2-F1-Lektion greift nicht, bestätigt); DimEq
   reicht nach Filtern; D=0-Zeilen konstruieren anstandslos; Stems nach
   <op>:-Konvention; embedMatrix-Parität + NaN-Bits real ausgeführt;
   Skizzen-Kosten GEMESSEN: Root +801 / stress +1,441 / browser 0 bei konstanter
   Dateizahl (Faktorisierung von RowShapesOf in EINEN benannten Typ halbierte die
   Kosten ≈1,428→801 — Empfehlung übernommen); Insertion-Ort: Statics-Block nach
   fromArray (stack ist konzeptionell ein Konstruktor). Import-Konventionen je
   Datei beachten (vector.ts: separates import-Statement je Append; ndarray.ts:
   den EINEN runtime.ts-Block erweitern).
