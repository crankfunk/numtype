# dt4 — Vergleiche, `where`, `any`/`all` — bindende Spec (Stufe 3b)

**Version:** v1 (2026-10-05) · **Status:** Owner-abgenommen (A1, A2, A3 i–iv je nach Empfehlung, 2026-10-05) → Baustein 0
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
- **V2 `NDArray.where(mask, a, b)`** (statisch, wie `np.where`): `mask` muss bool sein, `a`/`b`
  numerisch; Ergebnis-Shape = drei-Wege-Broadcast von `mask`, `a`, `b`; Ergebnis-dtype
  `Promote<Da, Db>` (bool in `a`/`b` → Ablehnung wie bei Arithmetik). Nur Array-Operanden (A3 iii).
  Prüfreihenfolge zur Compile- und Laufzeit gleich: Shape, dann Masken-dtype, dann `a`/`b`-dtype.
  Laufzeit: Werte aus `a`/`b` nach dem Ergebnis-dtype konvertiert (float32 per `fround`, int32
  exakt) — bei gleichem dtype ein reiner Kopiervorgang, NaN-Payload erhalten (M1 v6
  Datenbewegung; per Test belegt).
- **V3 `any`/`all`** mit denselben drei Überladungen wie `sum` (0-arg, Achse, Achse + keepdims) und
  demselben `ReduceAxis`-Guard; Rückgabe `NDArray<…, "bool">` (Rang 0 bei `any()`, wie `sum()`).
  Semantik: `any` = mindestens ein Element ≠ 0, `all` = alle ≠ 0, NaN zählt als ≠ 0 (identisch zu
  `astype("bool")`, D3). size-0: `any` → false, `all` → true (NumPy). Welche dtypes: A3 ii.
- **V4 Maschinerie:** `CompareCheck`, `WhereCheck` (drei-Wege-Broadcast + Masken-Prüfung + `Promote`)
  auf Modulebene, neue Meldungskonstante(n) an `runtime.ts` ANGEHÄNGT, Typ liest sie per `typeof`
  (eine Quelle). Neue Referenzfunktionen nur angehängt.
- **V5 Tests und Pins:** Vergleichstabelle 4 × 4 dtypes × 6 Ops (Typ + Laufzeit), Skalar-Formen
  inkl. float32-`fround`-Fall (ein Wert x mit `fround(x) ≠ x`, der das Ergebnis kippt), int32 mit
  `2.5`; NaN/±0/±Inf je Op; Broadcasting über gemischte dtypes; `where` mit drei verschiedenen
  Shapes, gemischten dtypes, NaN-Payload-Erhalt bei gleichem dtype, Maske nicht-bool abgelehnt;
  `any`/`all` mit Achsen, keepdims, size-0, NaN; Views per `wideSpecs` (Regel 16) mit
  Klassen-Assertion (Regel 12); Union-/`any`-Pins (`AnyNDArray`, `NDArray<S, DType>`, Unions mit
  bool) — je mit Mutant-Nachweis; Diagnose-INHALTE per tsc-Fixture (Regel 2); Hovers per LSP
  (Regel 13): `f32.gt(i32)` → `NDArray<[3], "bool">`, `NDArray.where(m, f32, i32)` →
  `"float64"`, `m.any()` → `NDArray<[], "bool">`. Mutanten auf einer Kopie von `spike/src`
  außerhalb des Repos.
- **V6 Covenant v11** (A3) und FOLLOWUPS (neue kernel-lose Referenzen nach M1 v5; `where` mit
  Skalaren, logische Ops `and`/`or`/`not` als Kandidaten).

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

Covenant-Wortlaut (v11), mit A3 abgenommen:

> **M2 · Präzisierung v11 — Vergleiche und `where` (Owner-entschieden 2026-10-05):** die Prüfungen der
> Vergleiche (`CompareCheck`) und von `where` (`WhereCheck`) sind dtype-Maschinerie nach v8–v10:
> `any` und Unions degradieren zu keinem Anspruch, bevor eine bool-Ablehnung greift; der
> Ergebnis-dtype eines Vergleichs ist bool unabhängig von den Operanden (ein gültiger Anspruch
> auch bei Union-Operanden). Meldungen und Laufzeit-Tabellen stammen aus einer Quelle. Anker
> ergänzt: `sym:CompareCheck`, `sym:WhereCheck`.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit, JEDER mit Stopp)

1. **Commit A:** V1 + V4-Anteil Vergleiche (ohne Tests). Messen. **Stopp** bei Δ > **+2,500**.
2. **Commit B:** V2 + V3 + V4-Rest. Messen. **Stopp** bei Δ kumuliert > **+4,000**.
3. **Commit C1:** Laufzeittests. Messen. **Stopp** bei Δ kumuliert > **+6,000**.
4. **Commit C2:** Typ-Pins + FOLLOWUPS (Covenant-Text vorab als eigener Commit). Messen gegen das Gate.
5. **Commit D:** bench:editor-Re-Pin, falls die Klassen-Surface die Pins verschiebt (zweimal gemessen,
   Verschiebung im Projekt-Log erklärt).

**An jedem Stopp — auch am Gate nach C2 — gilt: Stand committen, berichten, nicht weiterbauen.
Abdeckung (Tests, Pins) wird NIE gestrichen, um unter eine Grenze zu kommen** (Lehre aus dt3: 8
gestrichene Pins kosteten 18 überlebende Typ-Mutanten).

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +9,000** gegen 250,319 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert. Herleitung, offen: Prototyp `gt` (eine Op, nur Array-Form) +236; dt4 hat 6 Ops × 2
  Überladungen + `where` + `any`/`all` mit 3 Überladungen → Implementierung grob +2,000 bis +3,500;
  Tests/Pins kosteten in dt2 +3,837, in dt3 +6,609 — dt4 hat die größte Testmatrix bisher. Baustein 0
  misst einen Entwurf; liegt dessen Hochrechnung weit daneben, wird das Gate VOR der Umsetzung mit
  dem Owner neu gesetzt (vorregistriert ist es erst mit der Implementierung).
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
