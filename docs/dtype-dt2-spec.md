# dt2 — Promotion und elementweise Arithmetik — bindende Spec (Stufe 3b)

**Version:** v1 (2026-09-26) · **Status:** Owner-abgenommen 2026-09-26 (A1–A3) → Baustein 0 läuft
**Stufe:** 3b — neue Typ-Maschinerie-Klasse (`Promote`) und Covenant-Änderung (v9).
**Berührte Invarianten:** M2 (Promotion, Skalar-Regel), M3 (Meldungen, benannte Skalar-Ausnahme),
M1 (neue kernel-lose Referenzen, Tracking), Z1.
**Grundlage:** docs/dtype-design-spec.md v2.1 (D4 Promotion inkl. Union-Gate, D5 Ergebnis-dtypes,
D6 Skalar-Regel, D7 bool, D8 float32/int32-Rechenregeln) · docs/dtype-design-ergebnisse.md (Prototyp
Stufen 3–4, NO-GO der Ein-Signatur) · dt1 auf `main` (docs/dtype-dt1-ergebnisse.md). Vorlage-Code:
lokaler Branch `proto/dtype`, Stufen 3 (`90e7aae`) und 4 (`c301965`).

## Ziel

`add`, `sub`, `mul`, `div` rechnen für float64, float32 und int32 in jeder Kombination, mit den
Promotionsregeln aus D4/D5 — zur Compile-Zeit berechnet, zur Laufzeit identisch. bool bleibt für
Arithmetik dauerhaft gesperrt (E5, `astype` verlangen). Alle übrigen rechnenden Ops bleiben wie in
dt1 gesperrt. Keine Veröffentlichung vor dt5.

## Umfang

- **P1 `Promote<A, B>`** (D4): Union-Gate ZUERST (`IsUnion<A>`/`IsUnion<B>` → `DType`, kein
  Anspruch), dann die Tabelle — Typ und Laufzeit aus EINER Quelle (`PROMOTE_NUMERIC` in
  `runtime.ts`, der Typ liest sie per `typeof`, Prototyp-Muster). bool mit irgendwas → Ablehnung mit
  eigener Meldung (`BOOL_ARITHMETIC_MESSAGE`, wortgleich Compile/Laufzeit).
- **P2 Array ⊕ Array** für `add`/`sub`/`mul`: Ergebnis `NDArray<Broadcast<S, B>, Promote<D, Dd>>`;
  `div`: Gleitkomma immer (float32/float32 → float32, sonst float64). Laufzeit (D8): float32 in f64
  rechnen und nach JEDER Operation `Math.fround`; int32 `(a + b) | 0`, `(a - b) | 0`,
  `Math.imul(a, b)` (Zweierkomplement-Wrap). Neue Referenzfunktionen an `runtime.ts` ANGEHÄNGT; die
  bestehenden float64-Pfade bleiben byte-gleich und werden für float64 ⊕ float64 weiter benutzt.
- **P3 Skalar-Überladungen** (D6): float32/float64 behalten D (Skalar bei float32 per `fround`);
  int32 `add`/`sub`/`mul` behalten int32 — ein Literal mit Punkt (`2.5`) ist ein Compile-Fehler
  (vorhandenes `IsDotFormStep`), ein nicht-ganzzahliger Laufzeitwert wirft mit derselben Meldung;
  `div` auf int32 → float64. Überladungen bleiben ZWEI (Array-Überladung zuletzt) — die
  Ein-Signatur-Form ist per Messung ausgeschlossen (+8,495, NO-GO).
- **P4 bool:** Arithmetik auf bool (als Empfänger oder Argument) bleibt gesperrt, aber jetzt mit der
  dauerhaften Meldung „use astype()" statt der dt1-Übergangsmeldung.
- **P5 Tests und Pins:** Promotionstabelle vollständig (Typ-Pins + Laufzeit, beides gegen die eine
  Quelle), float32 bit-identisch gegen eine unabhängige `Float32Array`-Referenz, int32-Wrap an
  ±2^31 gegen `BigInt.asIntN(32, …)`, Broadcasting über gemischte dtypes, Union-Gate-Pins
  (`NDArray<S, DType>`, `"bool" | "int32"`), Skalar-Regel an den Rändern (`2.0`, `-0`, `1e3`,
  `2.5`, wide `number`), Diagnose-INHALTE per tsc-Fixture (Arbeitsregel 2), Hovers per LSP
  (Regel 13).
- **P6 Covenant v9** und FOLLOWUPS (neue kernel-lose Referenzen, M1 v5).

**Nicht in dt2:** Reduktionen (`sum`/`mean`/`matmul`/`dot`/`norm`, dt3), Vergleiche/`where`/
`any`/`all` (dt4), `sqrt`/`argmax`/`topk`/`stack` (dt5), `WNDArray` und WASM (dt6+).

## Owner-Abnahmen vor Baustein 0

**A1 — Methodenliste nach Hausregel (b):** an `NDArray` die Signaturen und Körper von `add`, `sub`,
`mul`, `div` (je Skalar- und Array-Überladung). Sonst nur neue Typen auf Modulebene
(`Promote`, Hilfstypen) und Anhänge an `runtime.ts`.

**A2 — vier bestehende Tests aus dt1 ändern sich zwangsläufig**, weil dt2 genau die Sperre aufhebt,
die sie prüfen (in `spike/tests-runtime/scalar-mean.test.ts`):
1. „add/sub/mul/div: scalar form throws the locked message …" und
2. „add/sub/mul/div: array form throws the locked message …" → werden durch Tests der neuen
   Semantik ERSETZT; die bool-Fälle darin bleiben als Sperr-Tests mit der dauerhaften Meldung.
3. „a locked two-operand op with TWO DIFFERENT non-float64 dtypes …" und
4. „DTypeLockPair (F2 pin) …" benutzen `add` nur als Beispiel einer gesperrten Zwei-Operanden-Op →
   ziehen auf `matmul` um (bleibt bis dt3 gesperrt); Absicht und Nachweis-Mechanik unverändert.
Dazu die Pins in `ndarray.test-d.ts`, die `add` auf einem Union-dtype als „kein Anspruch" festhalten,
falls sich ihr Ergebnistyp durch `Promote` verschiebt (vor der Umsetzung per Probe zu klären —
Baustein 0).

**A3 — Covenant-Wortlaut (Version 9):**

> **M3 · Präzisierung v9 — benannte Ausnahme dtype-Skalar:** scheitert ein SKALAR-Argument an
> einer dtype-Regel (Arithmetik auf bool, nicht-ganzzahliges Literal auf int32), zeigt der Editor
> die generische TS2769 der letzten Überladung statt der eigenen Meldung; die Laufzeit wirft die
> eigene. Begründung gemessen: die einzige korrekte Alternative (eine Signatur mit
> `IsUnion`-Gate) kostete +8,495 Instantiations (docs/dtype-design-ergebnisse.md). Im Quelltext an
> den Überladungen dokumentiert (Präzedenz W4/W5).
>
> **M2 · Präzisierung v9 — Promotion:** die dtype-Promotion (`Promote`) ist Teil der
> dtype-Maschinerie nach v8: eine Union als dtype eines Operanden degradiert das Ergebnis zu
> `DType` (kein Anspruch), bevor die Tabelle greift; Typ- und Laufzeit-Tabelle stammen aus einer
> Quelle. Anker ergänzt: `sym:Promote`.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit)

1. **Commit A:** P1 + P2 (ohne Tests). check:diag messen. **Stopp**, wenn Δ gegen 237,098 schon
   > +3,500 — Owner-Rückmeldung statt Weiterbauen.
2. **Commit B:** P3 + P4. Messen.
3. **Commit C:** P5, die A2-Änderungen, P6-FOLLOWUPS.

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +6,000** gegen 237,098 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert (Tests an bestehende Dateien anhängen). Herleitung, offen: im Prototyp kosteten
  Promotion + `add` −466 und die Skalar-Regel +1,798 — aber in einem anderen Maschinerie-Kontext und
  nur für `add`; dt2 fädelt vier Ops. Der Stopp nach Commit A fängt eine Fehlschätzung ab.
- stress/browser berichten · bench:editor: die Workloads benutzen `add` — Pins nur bei erklärter
  Verschiebung neu setzen, zweimal messen · Freeze-Hash unverändert · alle übrigen bestehenden Tests
  unverändert grün (außer A2) · Lint auf frisch gebautem Graph.

## Verify

Baustein 0 vor dem Code, danach A + B + C parallel (A auf deep — der erste standard-Lauf in dt1
lehnte ab). B bekommt ausdrücklich: Bit-Identität float32/int32 gegen unabhängige Referenzen,
Union-Gate-Löcher (die Blocker-Klasse aus Baustein 0 der Design-Scheibe), Broadcasting über gemischte
dtypes, float64-Pfade byte- und bitgleich zu vorher.
