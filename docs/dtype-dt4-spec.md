# dt4 — Vergleiche, `where`, `any`/`all` — bindende Spec (Stufe 3b)

**Version:** v1.1 (2026-10-06) · **Status:** Owner-abgenommen (A1, A2, A3 i–iv, 2026-10-05; A4–A6 nach
Baustein 0, 2026-10-06), Baustein 0 gelaufen (kein Blocker, A2 empirisch: kein Test betroffen) →
**implementierungsreif** (v11-Wortlaut s. A3)
**Stufe:** 3b — neue Typ-Maschinerie (`CompareCheck`, drei-Wege-Broadcast für `where`),
Covenant-Änderung (v11, A3), kein Kernel.
**Berührte Invarianten:** M2 (Vergleichs-/`where`-Prüfung, Union/`any`-Gate), M3 (Meldungen am
Argument, Message-Parität), M1 (neue kernel-lose Referenzen, Tracking), Z1.
**Grundlage:** docs/dtype-design-spec.md v2.1 (E5, D4 Union-Gate, D5 Zeilen „Vergleiche" und
„`where`/`any`/`all`", D7, D8 bool) · dt2/dt3 auf `main` (Muster; G1 `IsAnyDType`) · Prototyp
`proto/dtype` Stufe 6 (`gt`, Array-Form, +236).

## Ziel

Die bool-Welt wird erzeugbar und verbrauchbar: sechs Vergleiche liefern bool-Masken, `where` wählt
mit einer Maske zwischen zwei Arrays, `any`/`all` reduzieren. Damit ist der Kreis aus D7 geschlossen
(bool entsteht aus Vergleichen, wird von `where`/`any`/`all`/`sum`/`mean` verbraucht). Keine
Veröffentlichung vor dt5.

## Fakten am Code (2026-10-05, `main` = `14d7294`)

- Es gibt heute keinen Vergleich, kein `where`, kein `any`/`all` auf `NDArray` (`git grep`).
- `Promote<A, B>` (ndarray.ts) lehnt bool mit `BOOL_ARITHMETIC_MESSAGE` ab („dtype 'bool' has no
  arithmetic — use astype()…") — für einen Vergleich ist der Wortlaut falsch.
- `Broadcast<A, B>` (broadcast.ts:44) ist zweistellig; einen drei-Wege-Broadcast gibt es nicht.
- Statische Präzedenzen: `NDArray.fromArray`, `NDArray.stack` (ndarray.ts:819/945).
- Prototyp `gt`: `CompareCheck<S, B, D, Dd>` = Shape zuerst, dann `Promote<D, Dd>`; Ergebnis
  `NDArray<OkShape<Broadcast<S, B>>, "bool">`; ohne Skalar-Form.
- Hausregel (a) greift: dt4 fügt nur NEUE Ops hinzu → insertion-only, kein bestehender Member wird
  geändert. Erwartet: kein bestehender Test ändert sich (Baustein 0 belegt das empirisch).

## Umfang

- **V1 Vergleiche** `gt`, `ge`, `lt`, `le`, `eq`, `ne`, je zwei Überladungen (Skalar zuerst,
  Array-Form zuletzt — Arbeitsregel 2):
  - Array-Form: Guard `CompareCheck<S, B, D, Dd>` = Shape-Fehler (`Broadcast`) zuerst, dann die
    dtype-Prüfung: `IsAnyDType` → `IsUnion` (kein Anspruch) → bool-Operand → eigene Meldung (A3 i) →
    sonst gültig. Rückgabe `NDArray<OkShape<Broadcast<S, B>>, "bool">` — der Ergebnis-dtype ist
    IMMER bool, auch bei Union-Operanden (das ist ein korrekter Anspruch, kein Degradieren nötig).
  - Skalar-Form: `NDArray<S, "bool">`; bool-Empfänger → abgelehnt (M3-v9-Ausnahme: der Editor zeigt
    die generische TS2769, die Laufzeit die eigene Meldung).
  - Laufzeit: Vergleich der Werte als JS-Zahlen. Das ist für jede Paarung exakt der Vergleich im
    promoteten Typ (int32 und float32 sind in f64 exakt darstellbar; Vergleiche runden nie).
    **Skalar bei float32** wird vorher per `Math.fround` gerundet (A3 iv, NEP-50 „weak scalar",
    konsistent zu D6); int32 mit nicht-ganzzahligem Skalar ist erlaubt und exakt (kein Ergebnis in
    int32, also keine D6-Sperre). IEEE: jeder Vergleich mit NaN ist `false`, außer `ne` → `true`;
    `-0` und `+0` sind gleich. Ergebnis `Uint8Array` mit 0/1.
- **V2 `NDArray.where(mask, a, b)`** (statisch, wie `np.where`): `mask` muss bool sein; Ergebnis-
  Shape = drei-Wege-Broadcast von `mask`, `a`, `b`. Ergebnis-dtype (A5): bool × bool → bool (reine
  Auswahl — in dt4 der Weg, Masken zu kombinieren); numerisch × numerisch → `Promote<Da, Db>`;
  bool gemischt mit numerisch → Ablehnung mit EIGENER `where`-Meldung (nicht der Arithmetik-
  Meldung). Nur Array-Operanden (A3 iii).
  - **Guard je Argument (v1.1, Baustein 0 D1, M3):** der Masken-Guard greift nur, wenn die Shape-
    Prüfung besteht; der `a`-Guard nur, wenn Shape und Maske bestehen; der `b`-Guard trägt Shape
    und dtype-Paar. So steht jeder Fehler am fehlerhaften Argument UND die Reihenfolge Shape →
    Maske → `a`/`b` gilt zur Compile- wie zur Laufzeit (ein einzelner Guard auf `b` meldete
    Masken-Fehler am dritten Argument). Gemessen +306 gegenüber der Ein-Guard-Form.
  - Laufzeit: Werte aus `a`/`b` nach dem Ergebnis-dtype konvertiert (float32 per `fround`, int32
    exakt). **Bei gleichem dtype ein reiner Kopiervorgang; für float32 über eine `Uint32Array`-
    Sicht auf denselben Buffer** (A6; float64, int32 und bool erhalten ihre Bits schon bei
    elementweiser Zuweisung — für float64 von Baustein 0 über 1 Mio. sNaN-Muster gemessen) —
    elementweise Zuweisung in ein `Float32Array` macht dagegen aus einem
    signalisierenden NaN ein ruhiges (`0x7fa00001` → `0x7fe00001`, vom Orchestrator reproduziert).
    Damit gilt die M1-v6-Datenbewegungs-Zusage bit-exakt inkl. sNaN; per Test mit `0x7fa00001`.
  - Offengelegt (Baustein 0 D4): die Shape-Meldung eines Dreifach-Konflikts nennt die Zwischen-
    Shape aus Maske ⊕ `a` (Compile und Laufzeit gleich). D5: mit `number`-Dims bleiben manche
    garantiert fehlerhaften Tripel ohne Compile-Fehler (Unvollständigkeit, nie falsche Ablehnung).
- **V3 `any`/`all`** mit denselben drei Überladungen wie `sum` (0-arg, Achse, Achse + keepdims) und
  demselben `ReduceAxis`-Guard; Rückgabe `NDArray<…, "bool">` (Rang 0 bei `any()`, wie `sum()`).
  Semantik: `any` = mindestens ein Element ≠ 0, `all` = alle ≠ 0, NaN zählt als ≠ 0 (identisch zu
  `astype("bool")`, D3). size-0: `any` → false, `all` → true (NumPy). Alle dtypes (A3 ii).
- **V4 Maschinerie:** `CompareCheck`, `WhereCheck` (drei-Wege-Broadcast + Masken-Prüfung + `Promote`)
  auf Modulebene, neue Meldungskonstante(n) an `runtime.ts` ANGEHÄNGT, Typ liest sie per `typeof`
  (eine Quelle). Neue Referenzfunktionen nur angehängt.
- **V5 Tests und Pins — Pin-Strategie verbindlich auf ALIAS-Ebene (A4, Baustein 0 T1):**
  `CompareCheck`/`WhereCheck` (und der Skalar-Operand-Typ) werden direkt gepinnt — alle 16
  dtype-Paare, `where`-Stichprobe, Shape-zuerst, `any`/Union/breite dtypes — plus drei Aufruf-Pins
  je Op für die Verdrahtung, plus Pins für: `where`-Ergebnis-dtype (inkl. f32/f32 → float32, i32/i32
  → int32, bool/bool → bool), Skalar-Union-Gate, „Ergebnis bool auch bei Union-Operanden"
  (Mutant `IsUnion<D> ? DType : "bool"`). Grund (gemessen): die `IsAnyDType`-/`IsUnion`-Gates sind
  auf Aufrufebene verhaltensgleich redundant — Aufruf-Pins können ihre Mutanten nicht töten (volle
  Aufrufmatrix 13/20, Alias-Ebene 17/20, mit den Zusatz-Pins voraussichtlich 20/20) und kosten mehr
  (+9,042 statt +3,641). Laufzeitseitig: Vergleichstabelle 4 × 4 dtypes × 6 Ops, Skalar-Formen
  inkl. float32-`fround`-Fall (ein Wert x mit `fround(x) ≠ x`, der das Ergebnis kippt), int32 mit
  `2.5`; NaN/±0/±Inf je Op; Broadcasting über gemischte dtypes; `where` mit drei verschiedenen
  Shapes, gemischten dtypes, NaN-Payload-Erhalt bei gleichem dtype, Maske nicht-bool abgelehnt;
  `any`/`all` mit Achsen, keepdims, size-0, NaN; Operanden aus `transpose()`/`slice(...wideSpecs(…))`
  (Regel 16; `NDArray` kennt keine Strides-Views, Regel 12 greift hier nicht); Prüfreihenfolge
  Shape → Maske → `a`/`b` als Alias-Pin, tsc-Fixture UND Laufzeittest (Baustein 0 T2; ein Mutant
  „dtype vor Shape" überlebte die Aufrufmatrix); Dreifach-Negativfall, bei dem nur die Kombination
  aller drei Shapes scheitert, plus Argument-Permutationen (T4); `where` mit sNaN `0x7fa00001`
  (A6); Skalar-`fround`-Fälle `f32[fround(0.1)].gt(0.1)` → false, `f32[16777216].eq(16777217)` →
  true, `f32[Infinity].eq(3.4028235677973366e38)` → true (Skalar rundet zu Infinity — als Pin und
  Doku-Satz), `i32.gt(2.5)`, `i32.eq(NaN)`, `i32.lt(2**40)` (T6); Union-/`any`-Pins (`AnyNDArray`, `NDArray<S, DType>`, Unions mit
  bool) — je mit Mutant-Nachweis; Diagnose-INHALTE per tsc-Fixture (Regel 2); Hovers per LSP
  (Regel 13): `f32.gt(i32)` → `NDArray<[3], "bool">`, `NDArray.where(m, f32, i32)` →
  `"float64"`, `m.any()` → `NDArray<[], "bool">`. Mutanten auf einer Kopie von `spike/src`
  außerhalb des Repos. Neue Tests nur an bestehende Dateien (Dateiset-Gate).
- **Doku (Baustein 0 D7):** in generischen Helfern mit offenem `S`/`D` lehnen `where` und die
  Array-Form der Vergleiche ab wie `add`/`div` (bekannte Grenze); die Skalar-Form kompiliert.
- **V6 Covenant v11** (A3) und FOLLOWUPS (neue kernel-lose Referenzen nach M1 v5; `where` mit
  Skalaren, logische Ops `and`/`or`/`not` als Kandidaten; **vorbestehend:** `transposeDtyped`/
  `sliceDtyped` machen aus float32-sNaN ruhige NaNs — dieselbe Lücke wie A6, nicht in dt4 gefixt);
  Nachtrag in docs/dtype-design-spec.md D5/D7: `any`/`all` akzeptieren alle dtypes (A3 ii).

**Nicht in dt4:** logische Verknüpfungen von Masken (`and`/`or`/`not`/`xor`) · `where` mit Skalaren
(A3 iii) · Vergleiche zwischen bool-Arrays (A3 i) · `WNDArray`/WASM (dt6+) · `sqrt`/`argmax`/`topk`/
`stack` (dt5) · README (Regel 19).

## Owner-Abnahmen vor Baustein 0

**A1 — neue Oberfläche (Hausregel (a), insertion-only):** an `NDArray` neu `gt`/`ge`/`lt`/`le`/`eq`/
`ne` (je 2 Überladungen + Implementierung), statisch `where`, `any`/`all` (je 3 Überladungen +
Implementierung). Modulebene neu `CompareCheck`, `WhereCheck` (+ Hilfstyp für den drei-Wege-
Broadcast, falls nötig). `runtime.ts` nur Anhänge. Kein bestehender Member wird geändert.

**A2 — bestehende Tests:** keine Änderung erwartet (nur neue Ops). Findet Baustein 0 doch einen
betroffenen Test, kommt er vor der Umsetzung zur Abnahme.

**A4–A6 (nach Baustein 0, ENTSCHIEDEN 2026-10-06, je nach Empfehlung):** A4 Pin-Strategie auf
Alias-Ebene, Gate bleibt +9,000 · A5 `where` bool × bool → bool erlaubt, bool gemischt → eigene
Meldung · A6 Gleich-dtype-`where` über Ganzzahl-Sicht (sNaN-erhaltend), vorbestehende transpose/
slice-Lücke nach FOLLOWUPS.

**A3 — vier Design-Entscheidungen (ENTSCHIEDEN 2026-10-05: alle vier nach Empfehlung):**
- **(i) bool-Operand bei Vergleichen.** D5 lehnt ihn ab. Die vorhandene Meldung spricht von
  „arithmetic" und passt nicht. **Empfehlung: eigene Meldung** (z. B. „dtype 'bool' cannot be
  compared — use astype() to convert first"), wortgleich Compile/Laufzeit. Alternative: die
  Arithmetik-Meldung wiederverwenden (billiger, aber irreführend — M3).
- **(ii) `any`/`all` auf numerischen dtypes.** D5 sagt „nur bool-Maske". Dann bräuchten die
  niladischen 0-arg-Formen auf float/int eine dauerhafte Laufzeit-Sperre — ein neuer Fall neben v10
  (dort sperren niladische Ops bool, hier sperrten sie alles AUSSER bool). **Empfehlung: erlauben**
  (x ≠ 0, NaN → wahr, exakt `astype("bool")` aus D3, wie NumPy). Kein Konflikt mit E5 (E5 verbietet
  Arithmetik AUF bool, nicht das Lesen numerischer Werte als Wahrheitswert), keine Lücke, kein
  Covenant-Text für diesen Punkt. Alternative: nur bool + v11-Text für die zweite Sperr-Richtung.
- **(iii) `where`-Form.** **Empfehlung: statisch `NDArray.where(mask, a, b)`, nur Arrays** — wie
  `np.where`, und die Maske ist Argument, also bekommt sie eine Compile-Prüfung am Argument (als
  Methode auf der Maske säße die Prüfung am Empfänger). Skalare (`where(m, a, 0)`) verdoppeln die
  Überladungen (vier Kombinationen) → FOLLOWUPS. Alternative: Skalar-Formen gleich mitnehmen.
- **(iv) float32 gegen Skalar.** **Empfehlung: Skalar per `fround`** (NEP 50, konsistent zu D6:
  float32 behält D, der Skalar wird float32). Folge: `f32.gt(0.1)` vergleicht mit `fround(0.1)`,
  nicht mit 0.1. Alternative: exakter Vergleich mit dem f64-Skalar (anders als NumPy).

Covenant-Wortlaut (v11) — v1.1 um M3 und die Masken-Ablehnung erweitert (Baustein 0 C1),
**Owner-Bestätigung des erweiterten Wortlauts ausstehend**:

> **M2 · Präzisierung v11 — Vergleiche und `where` (Owner-entschieden 2026-10-05/06):** die Prüfungen
> der Vergleiche (`CompareCheck`) und von `where` (`WhereCheck`) sind dtype-Maschinerie nach
> v8–v10: `any` und Unions degradieren zu keinem Anspruch, bevor eine Ablehnung greift; der
> Ergebnis-dtype eines Vergleichs ist bool unabhängig von den Operanden (ein gültiger Anspruch
> auch bei Union-Operanden). Garantiert abgelehnt werden: ein bool-Operand eines Vergleichs, eine
> nicht-bool-Maske von `where`, eine Mischung aus bool und numerisch in `a`/`b` von `where`.
> Meldungen und Laufzeit-Tabellen stammen aus einer Quelle. Anker ergänzt: `sym:CompareCheck`,
> `sym:WhereCheck`.
>
> **M3 · Präzisierung v11:** die benannte Ausnahme dtype-Skalar (v9) gilt auch für einen
> bool-Empfänger in der Skalar-Form eines Vergleichs (dort scheitert ein Vergleich, keine
> Arithmetik). Bei `where` steht jede Ablehnung am fehlerhaften Argument (Maske, `a` oder `b`),
> in der Reihenfolge Shape → Maske → `a`/`b`, zur Compile- wie zur Laufzeit.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit, JEDER mit Stopp)

1. **Commit A:** V1 + V4-Anteil Vergleiche (ohne Tests). Messen. **Stopp** bei Δ > **+2,500**.
2. **Commit B:** V2 + V3 + V4-Rest. Messen. **Stopp** bei Δ kumuliert > **+4,000**.
3. **Commit C1:** Laufzeittests. Messen. **Stopp** bei Δ kumuliert > **+6,000**.
4. **Commit C2:** Typ-Pins + FOLLOWUPS (Covenant-Text vorab als eigener Commit). Messen gegen das Gate.
5. **Commit D:** bench:editor-Re-Pin — sicher nötig (Baustein 0: alle 8 Pins verschieben sich
   uniform um ≈ +2,845 = stress-Δ), zweimal gemessen, im Projekt-Log erklärt.

**An jedem Stopp — auch am Gate nach C2 — gilt: Stand committen, berichten, nicht weiterbauen.
Abdeckung (Tests, Pins) wird NIE gestrichen, um unter eine Grenze zu kommen** (Lehre aus dt3: 8
gestrichene Pins kosteten 18 überlebende Typ-Mutanten).

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +9,000** gegen 250,319 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert. Herleitung, offen: Prototyp `gt` (eine Op, nur Array-Form) +236; dt4 hat 6 Ops × 2
  Überladungen + `where` + `any`/`all` mit 3 Überladungen → Implementierung grob +2,000 bis +3,500;
  Tests/Pins kosteten in dt2 +3,837, in dt3 +6,609 — dt4 hat die größte Testmatrix bisher. Baustein 0
  misst einen Entwurf; liegt dessen Hochrechnung weit daneben, wird das Gate VOR der Umsetzung mit
  dem Owner neu gesetzt (vorregistriert ist es erst mit der Implementierung). **Baustein 0 hat
  gemessen:** nur Vergleiche +1,662, voller Entwurf +2,909 (Ein-Guard-`where`; Guard je Argument
  +306 mehr), C1-Probe +705, Alias-Pins + C1-Probe gesamt +7,191 (mit Zusatz-Pins geschätzt
  +7,500 bis +8,000). Die volle Aufrufmatrix hätte allein +9,042 gekostet (A4). Gate und Stopps
  bleiben.
- stress/browser berichten · bench:editor: Pins nur mit erklärter Verschiebung neu setzen, zweimal
  messen · Freeze-Hash unverändert (kein Rust) · alle bestehenden Tests unverändert grün · Lint auf
  frisch gebautem Graph · Autorenprüfung vor jedem Push (Arbeitsregel 22).

## Verify

Baustein 0 (`brainroute:deep`, eigener Worktree) vor dem Code: A2 empirisch (kein bestehender Test
betroffen?), Entwurfsmessung mit Hochrechnung, `CompareCheck`/`WhereCheck` ohne Loch (`any`, Unions
mit bool, `AnyNDArray`), drei-Wege-Broadcast gegen die Laufzeit, A3-Optionen im Editor (LSP). Danach
A + B + C parallel (A auf deep). B bekommt ausdrücklich: Vergleichs-Semantik gegen eine unabhängige
Referenz (NaN, ±0, float32-Skalar-Rundung), `where`-Bit-Identität inkl. NaN-Payload, Union-/`any`-
Löcher (G1-Klasse), Mutanten an Nachbar-Ops. Jeder Agent-Auftrag trägt Arbeitsregel 22 und die
Anweisung, den Worktree zuerst per `git merge --ff-only` auf den Zielstand zu bringen.

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-10-05/06)

`brainroute:deep`, eigener Worktree auf `12b2102`, Entwurf aller Ops (insertion-only belegt:
ndarray.ts +203/−0, runtime.ts +129/−0), echter tsc 7.0.2 und `tsc --lsp --stdio`. **Kein Blocker,
keine falsche Code-Annahme. A2 empirisch:** test:core 1653/1653, `pnpm check`, resident, package grün
auf dem Entwurf — kein bestehender Test betroffen. Baseline 250,319 @ 140 reproduziert, Freeze
unverändert. **Semantik** gegen unabhängige Referenzen: 750 Shape-Tripel (Typ-Urteil = Laufzeit),
4,000 `where`-, 3,871 Vergleichs-, 4,000 `any`/`all`-Fälle, 0 Abweichungen; NaN/±0/±Inf je Op.
**Befunde und Einarbeitung:** D1 `where`-Fehler am falschen Argument → Guard je Argument (V2);
D2 float32-sNaN wird bei elementweiser Kopie ruhig → Ganzzahl-Sicht (A6), vorbestehende Lücke in
transpose/slice nach FOLLOWUPS; D3 bool in `where` → A5; D4/D5/D7 offengelegt; D6/T1 Gate-Mutanten
auf Aufrufebene nicht tötbar → Alias-Pins (A4); T2–T6 in V5; Budget-Messung im Gate-Abschnitt;
Commit D fest; C1 v11 um M3 und Masken-Ablehnung erweitert; C2 Nachtrag in der Design-Spec.
Hovers wie erwartet (`f32.gt(i32)` → `NDArray<[3], "bool">`, `where(m, f32, i32)` → `"float64"`,
`m.any()` → `NDArray<[], "bool">`); Rang 900 ohne TS2589.
