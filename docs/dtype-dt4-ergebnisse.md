# dt4 — Vergleiche, `where`, `any`/`all` — Ergebnisse

**Spec:** docs/dtype-dt4-spec.md v1.1 (Stufe 3b) · **Datum:** 2026-10-06 … 14 · **Commits:** 12b2102 …
74b7295 auf `main` (unveröffentlicht; Release nach dt5) · **Covenant:** v11, Nachträge v12.

## Ergebnis

Sechs Vergleiche `gt`/`ge`/`lt`/`le`/`eq`/`ne` (Skalar- und Array-Form) liefern bool-Masken —
Ergebnis-dtype immer `"bool"`, auch bei Union-Operanden; Werte werden als JS-Zahlen verglichen (für
jede Paarung exakt), ein float32-Skalar wird vorher per `fround` gerundet (NEP 50). Statisch
`NDArray.where(mask, a, b)` (nur Arrays): Ergebnis-Shape aus dem drei-Wege-Broadcast, dtype bool ×
bool → bool, numerisch × numerisch → `Promote`, bool gemischt → eigene Meldung; Guard je Argument,
jede Ablehnung am fehlerhaften Argument in der Reihenfolge Shape → Maske → `a`/`b` (Shape-Fehler am
letzten Argument, M3 v12). Bei gleichem dtype ist `where` reine Datenbewegung: float32 über eine
`Uint32Array`-Sicht, float64 über `BigUint64Array` — NaN-Bitmuster inkl. signalisierender NaNs bleiben
erhalten. `any`/`all` (3 Überladungen wie `sum`) auf allen dtypes (≠ 0, NaN wahr), komponiert aus
`sumRuntime`. Neue Typen `CompareCheck`, `CompareScalarOperand`, `WhereCheck`; drei
Meldungskonstanten in `runtime.ts`. Hausregel (a) eingehalten: null gelöschte Zeilen in bestehendem
Code oder bestehenden Tests.

## Zahlen

| Gate | vorher | nachher |
|---|---|---|
| check:diag Root | 250,319 @ 140 | **264,285 @ 140** (Δ+13,966; Gate +9,000 — Überschreitung Owner-akzeptiert, s. u.) |
| Messpunkt A (Vergleiche) | — | 251,981 (+1,662; Stopp +2,500) |
| Messpunkt B (`where`, `any`/`all`) | — | 254,140 (+3,821; Stopp +4,000 — s. Addendum) |
| Messpunkt C1 (Laufzeittests) | — | 255,111 (+4,792; Stopp +6,000) |
| Messpunkt C2 (Typ-Pins) | — | 262,804 (+12,485; Gate +9,000 → **gestoppt**) |
| Fix-Runde F1 (Pins) / F2 (`BigUint64Array`) | — | 264,250 / 264,285 |
| check:diag:stress | 119,263 | 123,138 |
| check:diag:browser | 2,142 | 2,142 |
| bench:editor | `{41,121; 43,041; 73,798; 41,184; 46,820; 47,437; 40,034; 48,107}` | `{44,996; 46,916; 77,673; 45,059; 50,695; 51,312; 43,909; 51,982}` |
| test:core | 1,653 | 1,668 |
| resident / package / Freeze-Hash / cargo | 6,155+2 / 3 / `2a54d9fd…` / 222+1 | unverändert |

**bench:editor:** zwei Verschiebungen, je uniform über alle 8 Workloads: +3,844 durch die neue
Klassen-Surface (= stress-Δ, kein Workload ruft die neuen Ops), dann +31 durch den neuen
`whereTyped`-Körper in F2 (reiner Wertcode, ohne Typänderung — der Stand nach F1 hielt die alten Pins
exakt).

**Wo die Kosten liegen:** Implementierung +3,821, Laufzeittests +971, Typ-Pins +7,693 (C2) und
+1,446 (F1). Wie in dt2 und dt3 liegt die Überschreitung vollständig in Tests und Pins.

## Post-Verification-Addendum

- **Baustein 0:** kein Blocker, A2 empirisch (kein bestehender Test betroffen). Befunde
  eingearbeitet (v1.1): `where`-Fehler standen am falschen Argument → Guard je Argument; float32-sNaN
  wird bei elementweiser Kopie ruhig → `Uint32Array`-Sicht; bool in `where` → bool × bool erlaubt;
  Gate-Mutanten auf Aufrufebene nicht tötbar → Alias-Pins. Budget-Hochrechnung +7,191 (inkl. C1-Probe).
- **Umsetzung:** Stopp nach C2 korrekt ausgelöst (+12,485 gegen +9,000), nichts gestrichen.
  **Abweichung bei B:** die erste vollständige Messung lag bei +4,150 (Stopp +4,000); der
  Implementierer formulierte vor dem Commit zwei Implementierungsdetails billiger (Blatt von
  `WhereDType` direkt über `PROMOTE_NUMERIC`, Rückgabetyp über `WhereDType` statt `WhereCheck`) und kam
  auf +3,821. Owner: akzeptiert, Regel geschärft (Arbeitsregel 21: der Stopp gilt für die erste
  vollständige Messung). Baustein A und B bestätigten unabhängig, dass beide Formen für jeden
  Aufruf gleich urteilen (B: 7,371 `where`-Aufrufe, 0 Unterschiede).
- **Warum Baustein 0 das Budget unterschätzte:** die schlanke Pin-Menge war mit drei Aufrufen je Op
  gemessen; umgesetzt wurden vier je Op plus deutlich mehr `where`-Aufrufe und 15 zusätzliche
  Mutanten (35 statt 20).
- **Baustein A:** MERGE; alle Messpunkte reproduziert, 52 eigene Mutanten. Drei überlebende
  Union-Mutanten (`ge`/`ne`/`lt`).
- **Baustein B:** MERGE-AFTER-FIXES; rund 41,000 Laufzeitfälle gegen eigene Referenzen ohne
  Produktfehler. **Major:** die Ergebnis-Shape der Array-Form war bei keinem der sechs Vergleiche
  gepinnt — ein falscher Shape-Anspruch wäre unbemerkt geblieben. Dazu Pin-Lücken bei `any`/`all`
  (dritte Überladung) und der Überladungs-Reihenfolge für `ge`/`lt`/`le`/`eq`; float64-NaN-Erhalt nur
  auf V8 belegt.
- **Baustein C:** kein Covenant-Verstoß; zwei Textlücken (→ v12), zwei vorbestehende Befunde
  (→ FOLLOWUPS).
- **Fix-Runde F1–F3 und Gegencheck (verify): alles geschlossen.** 40 Mutanten getötet (Shape × 6,
  Union-bool × 18, `any`/`all`, Reihenfolge × 4). F2 kopiert float64 über `BigUint64Array`. Ehrlicher
  Rest: der Mutant „ohne Sicht" ist auf V8 nicht tötbar (V8 erhält die Bits auch so; 2,4 Mio.
  Payloads geprüft) — die Sicht sichert die Zusage für Engines ab, die NaN kanonisieren dürfen. So
  steht es auch im Code-Kommentar.
- **Owner-Entscheidungen:** Überschreitung zuerst +12,485, mit der Fix-Runde insgesamt **+13,966**
  akzeptiert; Struktur-Folgepunkt „Typ-Pins in eigenen Korpus" (FOLLOWUPS).
- **Integration:** die 8 Commits wurden auf `main` (mit v12) umgesetzt; einziger Konflikt in
  FOLLOWUPS (beide Einträge behalten). Der Code ist byte-gleich zum geprüften Stand `1d3fc79`
  (`git diff --stat` zeigt nur COVENANT.md und FOLLOWUPS.md).
