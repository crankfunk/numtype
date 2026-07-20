# Op-Scheibe W2: Skalar-Overloads + `mean` — bindende Spec

Status: **bindend** (Owner-Auftrag 2026-07-20: W2–W5 sind gesetzt; Evidenz
docs/dogfooding-rag-ergebnisse.md W2/F2 — Skalar-Wrap-Workaround `fromArray([1],[2])`,
HANDOFF-Erwartung „mean" als granularere Lücke identifiziert).
Version: 3 (v2 nach Baustein 0; v3 = D2-Reihenfolgen-Korrektur nach Verify-B F1, Nachtrag im Addendum) · Datum: 2026-07-20/21 ·
Eskalationsleiter: **Stufe 3** (voller Verify-Katalog).
Covenant: **v5** (kernel-lose Referenz-Ops explizit zulässig, Paritätslücke in FOLLOWUPS).

## Ziel & Warum

`x.div(2)` soll sich wie „durch 2 teilen" lesen (heute: `[1]`-Wrap-Trick), und `mean`
soll als Komposition fast gratis mitkommen (Summe + Skalar-Division). Zweite Op-Scheibe
der Wunschliste; gleiche Produkt-Identität wie W1 (M2/M3), gleiche Surface-Entscheidung
(NDArray-only, Parität in FOLLOWUPS — jetzt v5-gedeckt).

## Berührte Covenant-Invarianten

- **M1 (v5):** kernel-lose Referenz-Ergänzungen in runtime.ts — zulässig per v5-Zusatz;
  FOLLOWUPS-Paritäts-Item wird um W2-Ops erweitert. Artefakt-Hash byte-identisch.
- **M2/M3:** Skalar-Overloads erzeugen KEINE neue Fehler-Fläche (ein `number`-Skalar ist
  immer valide — kein Guard, keine Degradationskanten); `mean` erbt ReduceAxis' Guards
  und Stems 1:1 (wortgleicher `reduce:`-Stem, M3 wie bei sum/argmax).
- **M4/M5/Z1/Z2:** unberührt (kein Rust, keine node:*-Imports, keine Dependencies;
  neues Testfile liegt im Root-Korpus).

## Bindende Entscheidungen

- **D1 — Surface & Scope:** NUR `NDArray`. Ops: `add`/`sub`/`mul`/`div` erhalten je
  einen Skalar-Overload `(s: number)`, plus neue Methode `mean` (Overloads 0/1/2 exakt
  nach sum-Muster). KEIN `rsub`/`rdiv` (keine Evidenz; `2/x` bleibt Nutzer-Sache),
  KEIN `.scale()`-Alias (die vier Overloads decken alles).
- **D2 — Skalar-Semantik = SHAPE-ERHALTEND (NumPy-Skalar, nicht [1]-Broadcast):**
  `x.div(s)` gibt `NDArray<S>` zurück — auch für Rang 0 (`[]` bleibt `[]`; der
  [1]-Wrap-Workaround hätte `[] → [1]` gemacht, das wäre NumPy-falsch). Typ-Ebene:
  Overload `div(s: number): NDArray<S>` VOR dem generischen `div<B>(other: …)`-Overload.
  Kein Guard am Skalar (jeder endliche/nicht-endliche number ist valide; NaN/±Infinity
  propagieren nach IEEE — keine Sonderbehandlung, Test pinnt das).
  **v2 (Baustein 0, dokumentierte Kante):** Ein UNION-Argument über die Overload-Grenze
  (`x: number | NDArray<[3]>`) wird von TS als Ganzes ABGELEHNT (TS2769), auch wenn
  jeder Member einzeln valide wäre — inhärente Eigenschaft echter Overloads, exakter
  Codebase-Präzedenzfall `NDArray.backend(kind)` (ndarray.ts:336-342) mit dokumentiertem
  Narrowing-Ausweg (`typeof x === "number" ? … : …`, empirisch verifiziert). Bewusst
  akzeptiert, weil eine Union-Parameter-Signatur das D2-Präzisionsziel bricht; im
  Doc-Kommentar der Methoden mit backend()-Verweis offenlegen. M3-Randnotiz: bei
  teil-invaliden Unions erscheint die Shape-Message nur im nested Overload-Detail —
  dokumentiert, keine Änderung.
- **D3 — Skalar-Runtime:** EINE neue Referenz-Funktion, APPENDED in runtime.ts:
  `scalarElementwiseRuntime(op: "add"|"sub"|"mul"|"div", data: Float64Array, s: number):
  Float64Array` — elementweise `data[i] op s` in aufsteigender Index-Reihenfolge,
  frisches Float64Array, Shape unberührt (die Methode reicht `this.shape` durch).
  BEWUSST kein Umweg über den binären Broadcast-Pfad (kein [1]-Temp, keine
  Shape-Änderung bei Rang 0). Differential-Pflichttest: für Rang ≥ 1 ist das Ergebnis
  byte-identisch zum bestehenden `[1]`-Wrap-Weg (`x.div(NDArray.fromArray([1],[s]))`)
  — beweist Semantik-Äquivalenz dort, wo beide Wege existieren; für Rang 0
  dokumentiert der Test die gewollte Differenz ([] bleibt []).
- **D4 — `mean`-Form:** Overloads exakt wie `sum` (niladisch → `NDArray<OkShape<
  ReduceAxis<S, undefined, false>>>`; `(axis)`; `(axis, keepdims)`) — ReduceAxis/
  ReduceAxisKeepDims UNVERÄNDERT wiederverwendet, reduce.ts wird nicht editiert.
  KEINE argmax-artige number-Sonderform (mean IST eine Reduktion wie sum, bleibt
  chainable; sum-Präzedenz schlägt dot-Präzedenz).
- **D5 — `mean`-Runtime (Reihenfolge GEPINNT):** `meanRuntime` = `sumRuntime`-Aufruf,
  dann pro Output-Element GENAU EINE Division durch n (n = `shape[normAxis]` bei
  Achse, Gesamtelementzahl bei undefined) — NICHT `sum * (1/n)` (andere Rundung; die
  Wahl „÷n pro Element" ist die bindende Determinismus-Entscheidung, Test pinnt sie
  gegen ein Handbeispiel, dessen `sum/n ≠ sum*(1/n)` in f64 — der Implementierer MUSS
  so ein Beispiel konstruieren, z. B. via n=49 o. ä., sonst ist der Pin vakuös).
  Achsen-Validierung/Throw-Stem wortgleich sumRuntime (identischer reduce-Stem, M3);
  keepdims via keepDimsShape nach eigener Prävalidierung (fünfte Call-Site, Kontrakt
  eingehalten). size-0: leere Summe 0, n = 0 → `0/0 = NaN` — NumPy-konform (mean of
  empty → NaN), KEIN Throw; explizit getestet und im Doc-Kommentar offengelegt.
- **D6 — Datei-Disziplin, v2:** wie W1 inkl. der dort verifizierten Auslegung: runtime.ts
  nur Appends; `mean` als neuer Member = reine Klassenkörper-Insertion; Import-Zeilen-
  Erweiterungen zulässig und im Diff ausgewiesen. **AUSNAHME (Baustein-0-BLOCKER,
  erzwungene TS-Mechanik ohne Alternative):** Die vier BESTEHENDEN Methoden
  add/sub/mul/div MÜSSEN für die Skalar-Overloads editiert werden — TS verbietet
  Overload-Signaturen vor einer body-tragenden Deklaration (TS2394, empirisch
  reproduziert). Verbindliche Edit-Form (vom Verifier gebaut und grün kompiliert, alle
  Bestands-Call-Sites erhalten): je Methode wird (1) die bestehende generische Signatur
  zur BODYLOSEN Overload-Signatur (Zeileninhalt sonst unverändert), (2) die neue
  Skalar-Overload-Signatur ergänzt, (3) eine neue union-typisierte Implementierungs-
  Signatur eingefügt, deren Body die ORIGINALE Logik BYTE-IDENTISCH in den else-Zweig
  verschiebt (Skalar-Zweig via `typeof === "number"`). Verify-A prüft die
  Byte-Erhaltung der verschobenen Logik explizit am Diff. KEINE Edits an vector.ts/
  reduce.ts/dim.ts/literal-arithmetic.ts/index.ts. KEIN neues Source-File unter
  spike/src.
- **D7 — Tests:** Runtime: EIN neues File `spike/tests-runtime/scalar-mean.test.ts`
  (test:core-Liste registrieren): alle vier Skalar-Ops × Rang 0/1/2 × Spezialwerte
  (NaN/±Inf-Skalar, NaN im Array, ±0), Äquivalenz-Differential zu [1]-Wrap (Rang ≥ 1,
  byte-exakt via Bit-Vergleich), Rang-0-Shape-Erhalt, mean: Handbeispiel-Paritäts-Pin
  (sum/n-vs-sum*(1/n)-Diskriminator), Achsen-/negative-Achsen-/keepdims-Fälle
  (Shape UND Daten), mean-von-empty → NaN, Stem-Wortgleichheit. Typ-Pins: APPEND an
  ndarray.test-d.ts: `div(2)` → exakt `NDArray<[2,3]>`-Erhalt (Equal-Pin), Rang-0-Pin,
  wide-Shape-Erhalt, Readonly-S-Erhalt; **v2 (Baustein-0-MAJOR präzisiert):**
  mean-Typ-Pins nach dem ARGMAX-PräZEDENZ als WIRING-Pins (≈4–6: dynamische Achse,
  Union-Achse, Mixed-Rank, keepdims-Union — beweisen die Verdrahtung an ReduceAxis,
  re-litigieren NICHT die 15er-UA-Pin-Familie der sum-Maschinerie); WNDArray-Zwillinge
  sind strukturell unmöglich (WNDArray hat weder add/sub/mul/div noch mean — explizit
  festgehalten, keine WUA-Spiegel). Overload-Nicht-Interferenz-Pins: `div(nd)` weiter
  generisch, `div(fromArray([1],[s]))`-Workaround-Pfad erreichbar, Union-über-Grenze
  per @ts-expect-error als dokumentierte Ablehnung gepinnt (D2 v2).
- **D8 — Gates & Pins:** identisch W1-D8 (alle Gates, GFM, lint, test:example);
  Pin-Protokoll: Baseline-Reproduktion im frischen Worktree (184,330 @ 136 ·
  103,719 @ 82 · 2,142 @ 75), empty-then-fill fürs neue Testfile, gestufte Messpunkte
  (① runtime+ndarray, ② test-d), stress-/browser-Deltas akzeptabel wenn
  deterministisch + attribuiert (W1-T2-v3-Regel von Anfang an); Absolut-Gate:
  Haupt-Pin-Wachstum ≤ **+10,000**; bench:editor-Pins bei Abweichung doppelt messen +
  aktualisieren. Sprache/`≈`-Regeln wie immer.

## Akzeptanzkriterien

- **T1:** Alle Gates grün (Exit-Codes berichtet), Artefakt-Hash byte-identisch.
- **T2:** Pin-Protokoll vollständig (Baseline, empty-then-fill, Stufen, Attribution);
  Absolut-Gate eingehalten; Deltas deterministisch (Doppelmessung).
- **T3:** Typ-Pins: Shape-Erhalt exakt (inkl. Rang 0 `[]`), mean-Degradationen
  spiegeln sum vollständig, Overload-Nicht-Interferenz gepinnt; keine neue
  konfident-falsche Kante (Verify prüft adversarial).
- **T4:** Runtime-Pins: [1]-Wrap-Äquivalenz byte-exakt (Rang ≥ 1), Rang-0-Erhalt,
  sum/n-Reihenfolgen-Diskriminator NICHT-VAKUÖS (Beispiel unterscheidet die beiden
  Formeln nachweislich), mean-empty → NaN, Stems wortgleich.
- **T5:** Datei-Disziplin am Diff (D6); test-scripts-guard grün; FOLLOWUPS-Paritäts-
  Item um W2-Ops erweitert.
- **T6:** Doc-Platzierung nach Hausregel; README: die W1-Notiz („TypeScript-runtime
  surface only") wird um die W2-Ops erweitert, bit-for-bit-Zeile bleibt wahr.

## Nicht-Ziele

Kein rsub/rdiv/scale-Alias, kein WNDArray/Threaded (FOLLOWUPS-Erweiterung), keine
Kernel, kein `var`/`std` (spätere Kandidaten, keine Demo-Evidenz), kein Release in
dieser Scheibe, keine Edits an Bestands-Typen/-Funktionen.

## Verify-Plan (Stufe 3)

Baustein 0 (brainroute:deep) VOR der Implementierung — Schwerpunkte: Overload-Auflösung
number-vs-NDArray empirisch (inkl. Union-Argument und Literal-number-Argument `div(2)`
mit `const`-Typparameter-Interferenz?), Broadcast-vs-shape-erhaltend-Entscheidung gegen
NumPy-Semantik, sum/n-Diskriminator-Konstruierbarkeit, ReduceAxis-Reuse für mean,
Testplan-Lücken. Danach A + B + C parallel (Template-Bausteine). Ergebnisse-Doc mit
Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-20/21)

Verifier: brainroute:deep, frischer Kontext, empirische Proben im Scratch-Worktree
(HEAD 9bd67ce; Haupt-Tree unberührt). Befunde und Auflösung:

1. **BLOCKER (hoch, zweifach empirisch):** Die v1-D6-Annahme „Skalar-Overloads =
   Insertion-only" war FALSCH — TS2394 verbietet Overload-Signaturen vor body-tragenden
   Deklarationen; die vier bestehenden Signaturzeilen MÜSSEN editiert werden. Kein
   Richtungs-Spielraum (Alternativen brechen D1 bzw. D2) → als erzwungene, exakt
   umrissene D6-v2-Ausnahme eingearbeitet (bodylose Overloads + byte-identisch
   verschobener Body; vom Verifier grün kompiliert, Bestands-Call-Sites erhalten).
2. **MAJOR (hoch):** „mean-Pins spiegeln sum" war mehrdeutig (argmax-Präzedenz = 4
   Wiring-Pins vs. wörtlicher 15er-Spiegel) → D7 v2 entscheidet: Wiring-Pins nach
   argmax-Präzedenz; WNDArray-Zwillinge strukturell unmöglich, explizit festgehalten.
3. **MINOR:** Union-über-Overload-Grenze wird als Ganzes abgelehnt (TS2769) — kein
   neues M2-Loch, exakter Präzedenzfall `NDArray.backend()` mit dokumentiertem
   Narrowing-Ausweg → D2 v2 benennt Kante + Präzedenz; Nit: nested Diagnostik bei
   teil-invaliden Unions dokumentiert.
4. **Nits:** Dispatcher-Form von scalarElementwiseRuntime als bewusste Stil-Wahl des
   Implementierers freigestellt (String-Op vs. vier Closures); Item-11-Querverweis in
   D2 gestrichen (kein optionaler Parameter im Spiel).
5. **Positiv verifiziert (empirisch):** korrigierte Overload-Form kompiliert grün mit
   allen Bestands-Call-Sites (inkl. [1]-Wrap-Workaround, Rang 0, wide, Readonly-S,
   NaN/±Inf-Skalare); [1]-Wrap-Äquivalenz byte-identisch für Rang ≥ 1, Rang-0-Divergenz
   bestätigt D2s Motivation; sum/n-Diskriminator konstruierbar (n=49-Beispiel + echter
   Achsen-Fall mit 2/4 abweichenden Elementen — nicht jedes Beispiel diskriminiert,
   Spec-Warnung berechtigt); size-0 → NaN in beiden Reduktionspfaden; keepDimsShape
   wird fünfte Call-Site (4 existieren); Pins matchen CLAUDE.md exakt; +10,000-Gate
   plausibel (W1-Referenz ≈53 Instantiations/Pin); ReduceAxis-Reuse kollisionsfrei;
   WNDArray hat heute KEINE der fünf Methoden (keine neue Asymmetrie-Klasse).

**Nachtrag v3 (Verify-Runde, 2026-07-21):** Verify-B bewies einen MAJOR: Die
Overload-REIHENFOLGE ist diagnose-tragend — TS meldet den Fehler des LETZTEN Kandidaten;
mit dem Skalar-Overload zuletzt verschwand die Shape-Message des häufigsten Fehlerfalls
hinter dem number-Decoy. D2 verlangt seit v3 verbindlich: Skalar-Overload ZUERST,
generischer Guard-Träger ZULETZT; gepinnt durch den Diagnose-Qualitäts-Test in
scalar-mean.test.ts (Message-INHALT via echtem tsc-Lauf auf Außer-Repo-Fixture; drei
schmale ambient.d.ts-Shims dafür ergänzt). Rest-Preis offengelegt: TS2769-Kopfzeile
bleibt (inhärent bei echten Overloads), die wortgleiche Message steht in der ersten
Detail-Zeile am Argument.
