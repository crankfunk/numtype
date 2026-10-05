# dt3 — Reduktionen — Ergebnisse

**Spec:** docs/dtype-dt3-spec.md v1.1 (Stufe 3b) · **Datum:** 2026-10-05 · **Commits:** 71fb2f5 …
403aeed auf `main` (unveröffentlicht; Release nach dt5) · **Covenant:** v10.

## Ergebnis

`sum`, `mean`, `matmul`, `dot`, `norm` und `cosineSimilarity` rechnen für float64/float32/int32,
`sum`/`mean` zusätzlich für bool (Zählen bzw. Anteil). Ergebnis-dtype der Reduktionen per
`ReduceDType` (float32 → float32, sonst float64; Typ und Laufzeit aus `REDUCE_DTYPE`), `matmul`/
`dot`/`cosineSimilarity` über das wiederverwendete `PromoteDiv` (O1: float32 nur bei zwei
float32-Operanden). Der float64-Rechenweg ist für alle konvertierten Eingaben die BESTEHENDE
Referenzfunktion; neu ist nur der float32-Weg (`fround` je Produkt und Summe, aufsteigend, Seed +0,
`mean = fround(sum32 / n)`). `norm()` auf bool wirft dauerhaft nur zur Laufzeit (A3, M2 v10).
`DTypeLockPair` ist gelöscht. Hover: `i32.sum()` → `NDArray<[], "float64">`, `f32.matmul(f32)` →
`NDArray<[2, 4], "float32">`, `f32.matmul(i32)` → `NDArray<[2, 4], "float64">`.

## Zahlen

| Gate | vorher | nachher |
|---|---|---|
| check:diag Root | 243,818 @ 140 | **250,319 @ 140** (Δ+6,501; Gate +5,000 um 1,501 überschritten — Owner-akzeptiert 2026-10-05, s. u.) |
| Messpunkt A (R1+R2) | — | 243,129 (−689; Stopp +1,500) |
| Messpunkt B (R3–R5) | — | 243,710 (−108; Stopp +2,500) |
| Messpunkt C1 (A2 + Laufzeittests) | — | 245,213 (+1,395; Stopp +2,000) |
| Messpunkt C2 (Typ-Pins, gekürzt) | — | 248,617 (+4,799) |
| nach Fix-Runde G1–G3 | — | 250,319 (+6,501) |
| check:diag:stress | 119,393 | 119,263 |
| check:diag:browser | 2,142 | 2,142 |
| bench:editor | `{41,165; 43,171; 73,928; 41,206; 46,751; 47,567; 40,131; 48,125}` | `{41,121; 43,041; 73,798; 41,184; 46,820; 47,437; 40,034; 48,107}` |
| test:core | 1,632 | 1,653 |
| resident / package / Freeze-Hash / cargo | 6,155+2 / 3 / `2a54d9fd…` / 222+1 | unverändert |

**bench:editor:** Commit A spart fast uniform −321 bis −354 (die `DTypeLock`-Bedingung fiel aus acht
Overloads der geteilten Klassen-Surface), Commit B kostet uniform +224 (der `PromoteDiv`-Guard ist
eine längere Kette als `DTypeLockPair`) plus einen Aufruf-Anteil bei Workloads mit
`matmul`/`dot`/`cosineSimilarity`. w5 ist der `matmul`-schwerste Workload, sein B-Anteil (+409)
übersteigt die A-Ersparnis, daher als einziger +69. Kein Pro-Aufruf-Preis (Super-Additivität).

**Wo die Kosten liegen:** die Implementierung ist billiger als vorher (−108). Die gesamte
Überschreitung kommt aus Tests und Typ-Pins: C1 +1,503, C2 +3,404, Fix-Runde G2 +1,687.

## Post-Verification-Addendum

- **Baustein 0:** kein Blocker, A2 empirisch exakt (fünf Tests), Entwurfsmessung deckungsgleich mit
  der späteren Umsetzung. Präzisierungen D1–D7 in v1.1 (Re-Pin-Commit, geteiltes C, Prüfreihenfolge
  Shape vor bool auch zur Laufzeit, Meldungsgleichheit f32/f64, Offenlegungen).
- **Abweichung in der Umsetzung:** der erste C2-Stand lag bei +5,618. Der Implementierer strich vor
  dem Commit 8 Typ-Pins, um unter das Gate zu kommen, statt zu stoppen — ein Stopp war vorregistriert
  nur nach A/B/C1, nicht für C2. Die Folge sah erst Baustein B: 18 Typ-Mutanten überlebten.
- **Baustein A:** MERGE-AFTER-FIXES; alle Messpunkte per Checkout reproduziert; ein überlebender
  Laufzeit-Mutant (`fround(sum32 * (1/n))` im float32-`mean`, R2 verbietet ihn ausdrücklich). Die
  Behauptung in der Message von Commit C1, beide Formen seien für n ≤ 2^24 „beweisbar gleich", ist
  **falsch**: Gegenbeispiel `[x, 0 × 783]` mit `x = fround(-5.022606823353154e-38)` — die Formen
  liegen im Subnormalbereich eine ulp auseinander (vom Orchestrator reproduziert). Baustein B hatte
  denselben Mutanten über 2,97 Mio. Zufallsfälle als äquivalent eingestuft — Zufallsfuzz trifft
  Subnormal-Ties praktisch nie.
- **Baustein B:** HÄLT; rund 4,3 Mio. Vergleiche gegen eigene Referenzen ohne Fehlschlag, 91
  Mutanten, davon 18 überlebende Typ-Mutanten (alle in der Lücke der gestrichenen Pins). Bekannte
  Grenze bestätigt, keine Regression: generische Funktionen mit offenem dtype lehnen `matmul`/`dot`
  ab (wie `add`/`div` seit dt2).
- **Baustein C:** kein Covenant-Verstoß.
- **Fix-Runde G1–G3 und Gegencheck (verify): alle drei geschlossen.** G1 Testfall mit dem
  Subnormal-Gegenbeispiel; G2 die 8 Pins zurück plus gezielte Pins, alle 18 Mutanten unabhängig
  getötet; G3 Doc-Kommentar zur f32-Absorption präzisiert. Owner-Entscheidung: Überschreitung um
  1,501 akzeptiert. Die Message von `db93621` spricht schon von „accepted by the owner" — zu dem
  Zeitpunkt war nur die Richtung (Pins zurück, Überschreitung benennen) entschieden, die Zahl selbst
  wurde danach vorgelegt und akzeptiert.
- **Bewusst stehen gelassen:** veraltete Erwähnungen von `DTypeLockPair` in ndarray.ts:192/:1028,
  runtime.ts:1701 und im G4-Test (vorbestehende Zeilen; Hausregel (b) und append-only). Sie
  beschreiben die Lage vor dt3.
