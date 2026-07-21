# Op-Scheibe W5: `item` — bindende Spec

Status: **bindend** (Owner-Programm 2026-07-20: W2–W5 gesetzt; Evidenz
docs/dogfooding-rag-ergebnisse.md W5/F3 — Skalar-Read aus der Score-Matrix, als
Beobachtung mit niedrigster Priorität; letzter Wunschlisten-Platz vor dem
0.2.0-Bündel).
Version: 3 (v2 nach Baustein 0; v3 = F6-Faktenkorrektur + Policy-Pin nach Verify-B, Nachtrag im Addendum) · Datum: 2026-07-21 · Eskalationsleiter: **Stufe 3** (kompakt skaliert —
W3-Größenordnung).
Covenant: v5.

## Ziel & Warum

`similarities.item(qi, docIdx)` statt `slice(qi).slice(docIdx)` bzw. Flat-Index-
Arithmetik über `.data` — der direkte Skalar-Read, NumPy `x.item(i, j)`. Schließt den
letzten Wunschlisten-Platz; danach ist der komplette Friction-Log der RAG-Demo
abgearbeitet.

## Berührte Covenant-Invarianten

- **M2** (Anker Guard/OkShape/literal-arithmetic.ts): der Rückgabetyp ist immer
  `number` — die EINZIGE Typ-Ebenen-Aufgabe ist die Ablehnung garantierter
  Runtime-Throws (falsche Arity bei literalem Rang; literale Indizes außerhalb
  literaler Dims). **Hier ist `LiteralIndexBounds` die RICHTIGE Semantik** (|i| ≤ d
  mit NumPy-Negativ-Normalisierung — exakt das Spike-03-Regelwerk; der W1-Warnhinweis
  betraf topk, wo die Index-Semantik falsch war). Wide/Union/unknowable → kein
  Compile-Claim, Runtime-Backstop.
- **M3**: Fehler AM Argument (bzw. an der fehlerhaften Argument-POSITION, soweit TS
  das bei Rest-Parametern hergibt — Baustein 0 klärt die erreichbare Form; bekannte
  TS7-Grenze: EINE Diagnose pro Call bei mehreren invaliden Positionen, per Typ-Pin
  statt Squiggle-Zählung verankern), Stems wortgleich zur Runtime.
- **M1 (v5)**: kernel-los (kein Kernel — reiner strided Read; FOLLOWUPS-Nachtrag W5).
- **M4/M5/Z1/Z2**: unberührt.

## Bindende Entscheidungen

- **D1 — Scope:** `item(...indices): number` auf `NDArray` — VOLLE Indizierung (genau
  ein Index je Achse, Rückgabe Skalar). Rang 0: `item()` (null Argumente) → das eine
  Element. KEIN Partial-Indexing (das ist `slice()`), kein Setter, kein `at`-Alias
  (ein Name; `at` kollidiert mental mit Array.prototype.at-Semantik einer EINZELNEN
  Achse).
- **D2 — Typ-Ebene:** `ItemCheck<S, Idx>` (Ablage: vector.ts-Append ODER — falls
  Baustein 0 es sauberer findet — slice.ts-Nachbarschaft; KEIN neues File). Kanten:
  Arity ≠ Rang bei literalem Rang → ShapeError (Stem wortgleich); literaler Index ×
  literale Dim → LiteralIndexBounds-Klassifikation, „out" → ShapeError (inkl.
  negativer Literale — Spike-03-Regeln: Dot-Form-Literale sind garantierte Throws);
  wide Rang/wide Dim/wide Index/Union-Index → Pass (no-claim; Distribution per
  IsUnion-Gates VOR naked Checks — die W4-Lektion gilt verbindlich für JEDEN neuen
  Fold/Check). Rückgabetyp konstant `number` (keine Shape-Berechnung).
- **D3 — Runtime:** `itemRuntime(shape, strides‑frei via computeStrides, data,
  indices): number` APPENDED in runtime.ts: Arity-Check (Stem), je Achse
  Negativ-Normalisierung (`i < 0 → d + i`), Bounds-Check (Stem mit Achse/Index/Dim),
  dann Offset-Summe über computeStrides; size-0-Dims sind durch den Bounds-Check
  automatisch unerreichbar (jeder Index ist OOB — kein Sonderfall nötig, Test belegt).
  NaN/±0 werden als Wert exakt durchgereicht (trivial — direkter Read).
- **D4 — Methode:** `item<const Idx extends readonly number[]>(...indices:
  Guard<ItemCheck<S, Idx>, Idx>): number` als Klassenkörper-Append nach `sqrt`.
  Doc-Kommentar: NumPy-item-Bezug, Spike-03-Bounds-Reuse, F3-Evidenz, dot/norm-
  Skalar-Präzedenz, Surface-Hinweis.
- **D5 — Tests:** Runtime: W5-Block ans Ende von scalar-mean.test.ts (Kopf-Hinweis;
  KEIN neues File): Rang 0/1/2/3, negative Indizes (NumPy-Parität), alle
  Throw-Stems wortgleich, size-0-Dim-OOB, NaN/−0-Durchreichung (Bits), Parität gegen
  `.data`-Flat-Index-Rechnung über 200+ Zufallsfälle, transponierte/gesliceste
  Empfänger. Typ-Pins an ndarray.test-d.ts: valide literale Voll-Indizierung
  kompiliert (Rückgabe number), Arity-Fehler (@ts-expect-error + Message-Pin),
  OOB-Literal positiv UND negativ (@ts-expect-error + Message-Pin), gültiges
  negatives Literal kompiliert, wide/Union/Mixed-Rank → kompiliert (no-claim),
  Dot-Form-Index abgelehnt.
- **D6 — Gates & Pins:** Standard-Katalog; Baseline 195,481 @ 137 · 105,758 @ 82 ·
  2,142 @ 75; kein neues File; Absolut-Gate ≤ **+6,000** (Bounds-Digit-Maschinerie
  pro Pin-Site à la Spike 03, aber wenige Sites); Deltas 2× + attribuiert;
  bench:editor-Pins bei Bedarf 2× + aktualisieren; Mutanten NUR per Backup-Kopie.
  Nach W5: FOLLOWUPS-Notiz „scalar-mean.test.ts ist zum W2–W5-Sammelbecken gewachsen —
  Aufsplitten als eigene Mini-Scheibe MIT empty-then-fill-Protokoll" (Discoverability-
  Nit aus W3-Baustein-0, jetzt aktenkundig machen).

## Akzeptanzkriterien

- **T1:** Alle Gates grün, Hash byte-identisch, Exit-Codes berichtet.
- **T2:** Pins 2× deterministisch, Gate ≤ +6,000, attribuiert.
- **T3:** Typ-Pins decken jede D2-Kante; W4-Lektion nachweislich angewandt
  (IsUnion-Gates vor naked Checks — Verify-B greift genau diese Stelle an).
- **T4:** Runtime-Pins inkl. Flat-Index-Differential und Stems; Mutanten-Nachweis.
- **T5:** Datei-Disziplin; FOLLOWUPS-Nachträge (W5-Parität + Testdatei-Split-Notiz);
  Doc-Platzierung; README-Op-Notiz um item ergänzt.

## Nicht-Ziele

Kein Setter/`set`, kein Partial-Indexing, kein `at`, kein Fancy-Indexing, kein
WNDArray/Threaded (FOLLOWUPS), kein Kernel, kein Release in dieser Scheibe.

## Verify-Plan (Stufe 3, kompakt)

Baustein 0 (brainroute:deep, kompakt): (a) Guard auf REST-Parameter empirisch — wo
erscheint der Fehler (am Call? an der Position?), trägt die `...indices:
Guard<…, Idx>`-Form überhaupt (Rest-Param-Typ muss Array-artig bleiben — kollidiert
der ShapeError-Objekt-Typ mit dem Rest-Spread? ggf. alternative Form: Guard um das
TUPEL als einzelnes Argument-Muster oder never-Rest — empirisch entscheiden und
verbindliche Form liefern); (b) LiteralIndexBounds-Export-Status + exakte Semantik
gegen die W5-Bedürfnisse (Spike-03-Quelle lesen); (c) Union/Dot-Form-Kanten;
(d) Arity-Check-Form (`Idx["length"] extends S["length"]` — Mixed-Rank-Fallen);
(e) Gate-Plausibilität. Danach A+B+C parallel (kompakt; B greift die
Rest-Param-Guard-Form und die W4-Lektion-Anwendung an). Ergebnisse-Doc mit Addendum,
dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-21)

Verifier: brainroute:deep, Scratch-Worktree, verifizierte kompilierende Form geliefert
und GEMESSEN (+712 bei Gate ≤ +6,000; Baseline 195,481 exakt reproduziert). Befunde →
verbindliche v2-Formen:

1. **F1 (BLOCKER, sicher):** D4s `Guard<>` auf dem Rest-Parameter ist ein permanenter
   TS2370 AN DER DEKLARATION (Guard kollabiert im Fehlerfall zu einem Nicht-Array-
   Objekt; Rest-Params müssen Array-artig bleiben). slice() nutzt deshalb seit
   Spike 03 die Fold-Form. VERBINDLICH: `ItemGuard<S, Idx>` DIREKT als
   Rest-Param-Typ (SliceSpecsGuard-Parallele), per-Position `ItemMark` mit
   `{__shapeError}`-ELEMENT-Ersatz — die vom Verifier gelieferte, kompilierte Form
   ist maßgeblich (inkl. `I extends number`-Stilangleichung, F8).
2. **F2 (sicher):** Der Fold ist S-GETRIEBEN (rekurriert über S, füllt number-Marker
   wenn Idx erschöpft) — eine Idx-getriebene slice-Kopie akzeptiert Under-Arity
   still (dort gewollt: Partial-Indexing; hier falsch).
3. **F3 (Spec-Korrektur, erzwungene Mechanik):** Arity-Verstöße erscheinen als
   TS-NATIVES TS2554 („Expected N arguments, got M") — für fehlende Argumente
   existiert architektonisch keine Position für eine Custom-Message. M3-Auslegung
   v2: Custom-Stems wortgleich für Bounds + Dot-Form (verifiziert, exakte Spalte);
   Arity compile-seitig via TS2554 (immer noch AM Call, zählgenau — als
   dokumentierte Ausnahme wie StackRankMessageArray), runtime-seitig mit eigenem
   gepinnten Stem. Rang-0-Verhalten verifiziert (item() 0-stellig; item(0) → TS2554).
4. **F4 (Regression gefunden+gefixt):** Ohne Dynamic-Length-Gate bricht
   `m.item(...dynArr)` (TS2556) — slice() hat das Gate; VERBINDLICH:
   `IsDynamicRank<Idx>`-Gate (dim.ts-Reuse, kein neuer Typ). Spread-Fall wandert
   in den D5-Testplan (Typ-Pin: kompiliert, no-claim).
5. **F5 (Scope-Klärung):** Dot-Form-Ablehnung ist NICHT in LiteralIndexBounds
   (Spike-03-Out-of-scope; 1.5 → „unknown" = silent pass) — braucht den Export von
   `IsDotFormStep` (EIN export-Präfix in literal-arithmetic.ts; gleiche Owner-
   gedeckte Edit-Klasse wie W1s Compare/NonNegDigits).
6. **F6 (informativ):** LiteralIndexBounds' Union-Verhalten ist konservativer als
   sein Doc-Kommentar (uniform-invalide Union → „unknown", nicht „out") — der
   IsUnion-Pre-Gate in ItemMark ist damit doppelt begründet (W4-Lektion + F6).
7. **F7:** TS7-Ein-Diagnose-Regel reproduziert für item — Pins statt Squiggles
   (war bereits Spec-Text).
8. **Positiv:** LiteralIndexBounds exportiert + Semantik passt (Negativ-
   Normalisierung „in" für -1 auf d=3, „out" für 3 auf d=3); Ablage vector.ts
   (Stil-Fit, kein topologischer Tiebreaker); Baseline exakt; Delta 2× deterministisch.

**Nachtrag v3 (Verify-Runde, 2026-07-21):** Verify-B widerlegte ZWEI Meta-Behauptungen
(die Implementierung selbst hielt): (1) Der IsUnion-Pre-Gate in ItemMark war durch
keinen Pin load-bearing — die gesamte Suite blieb bei Entfernung grün. (2) Die
F6-Prämisse des Baustein-0 war FALSCH: `LiteralIndexBounds<5|9, 3>` ist „out", nicht
„unknown" — die Spike-03-Union-Disziplin (tuple-wrapped Subset-Check über uniforme
Verdikte) funktioniert exakt wie dort dokumentiert; F6 ist hiermit korrigiert.
Konsequenz: Der Pre-Gate ist KEINE Notwendigkeit, sondern eine bewusste
POLICY-Angleichung an den reduce.ts-Achsen-Union-Präzedenzfall (auch ALL-invalide
Unions degradieren zu no-claim, statt der volleren, gleichfalls soliden Ablehnung,
die LiteralIndexBounds allein liefern würde) — jetzt dokumentiert und durch den
neuen Policy-Pin `itemUnionIdxUniformInvalid: 5|9` LOAD-BEARING gemacht
(Gate-Entfernungs-Mutant: exakt diese eine Zeile wird rot; Backup-Kopie-Verfahren).
Dazu: irreführender „fast path"-Testkommentar korrigiert (A- und B-Fund),
Selbstbericht-Pin-Aufschlüsselung korrigiert (A-Fund).
