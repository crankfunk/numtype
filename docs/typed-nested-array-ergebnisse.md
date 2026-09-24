# 0b — Typisiertes `toNestedArray` — Ergebnisse

**Spec:** docs/typed-nested-array-spec.md v2 (Stufe 3b) · **Datum:** 2026-09-24 ·
**Commits:** `0f9cb6c` (Umsetzung), `fcbea01` (Verify-Nachtrag), COVENANT v7 (Präzisierung M3)
**Ehrlichkeitsregel:** jede Zahl unten stammt aus einem in dieser Scheibe ausgeführten Lauf;
Offenes steht als offen da.

## Ergebnis in einem Satz

`NDArray<S>.toNestedArray()` und `WNDArray<S>.toNestedArray()` liefern jetzt einen am Rang
berechneten Typ (`[2, 3]` → `number[][]`, `[]` → `number`, `[number, 3]` → `number[][]`) statt
`unknown`; bei statisch unbestimmtem Rang den exportierten rekursiven Typ `NestedValue`. Die
View `NDArrayView` bleibt bei `unknown` (TS2636 unter `out S`, am echten Interface gemessen).

## Was gebaut wurde

- `spike/src/ndarray.ts`: `NestedValue`, `NestedArray<S>` (exportiert) und privat `NestedOfRank`
  auf Modulebene — `never`-Gate zuerst, dann das bestehende `RankUnknowable` (dim.ts:82), dann ein
  tail-rekursiver Rang-Akkumulator über `S["length"]`.
- Beide Klassen: nur Signatur + Rückgabe-Cast geändert (Owner-bestätigte Abweichung von der
  insertion-only-Hausregel), Laufzeitkörper unverändert.
- `spike/src/index.ts`: type-only Re-Export von `NestedArray`/`NestedValue`.
- Pins in `spike/tests/ndarray.test-d.ts`: View bleibt `unknown`; beide Klassen gegen Rang 0–3,
  `[number, 3]`, `number[]`, `Shape`, Unions gemischten und gleichen Rangs, `never`, Rest-Tupel,
  `AnyNDArray`/`AnyWNDArray`; NDArray≡WNDArray; indizierter Zugriff unter
  `noUncheckedIndexedAccess` (`number | undefined`, per Gegenmutante nicht-vakuös belegt).
- Beide Konsumenten-Smokes pinnen `.toNestedArray()[0][1]` als `number` ohne Cast; die lokalen
  Typen in `package-smoke.test.ts` nachgezogen. 4 neue Laufzeittests (Rang 0 ist wirklich eine
  Zahl, size-0 ist `[]`, je Klasse); die View-Gleichheit war durch bestehende Tests mit
  assertierter View-Klasse schon abgedeckt.

## Zahlen

| Gate | vorher | nachher | Δ |
|---|---|---|---|
| check:diag Root | 226,220 @ 140 | **227,405 @ 140** | +1,185 (Umsetzung +1,154, Nachtrag-Pin +31) — Gate ≤ +4,000 |
| check:diag:stress | 116,220 @ 82 | 116,279 @ 82 | +59 |
| check:diag:browser | 2,142 @ 75 | 2,142 @ 75 | 0 |
| bench:editor W1–W8 | … | `{37800, 39633, 70775, 37955, 43254, 44448, 37004, 44693}` | uniform **+60** |
| test:resident | 6151+2 | 6155+2 | +4 |
| test:core / threaded / browser / package / cargo | 1591 / 139 / 4 / 3 / 222+1 | unverändert | 0 |
| Freeze-Hash | `2a54d9fd…` | `2a54d9fd…` | unverändert |

Root-Werte von Implementierer, Baustein A und Baustein B je unabhängig reproduziert (B zweimal
pro Seite). Dateiset in allen Korpora unverändert (kein Order-Noise).

**Hover (echter `tsc --lsp --stdio`, von Implementierer, A und B unabhängig gemessen):**
`(method) NDArray<[2, 3]>.toNestedArray(): number[][]`, Rang 0 `number`, `[number, 3]` →
`number[][]`, `number[]` → `NestedValue`, identisch auf `WNDArray`; Op-Kette
`a.matmul(b).toNestedArray()` → `number[][]`; Kontrollpunkt `NDArray<[2, 3]>` sauber.

## Offengelegte Grenzen

- **Rang-Grenze 999:** ab Rang 999 TS2589 (Tail-Rekursions-Limit des Akkumulators); die übrige
  Shape-Maschinerie bricht bei 1024. Baustein B hat bestätigt, dass Kompositionen mit
  `transpose`/`sum` die Grenze nicht weiter senken.
- **+59 vs. +60:** erstmals weichen bench:editor (+60 uniform) und check:diag:stress (+59) um 1
  ab. Ausgeschlossen: Dateiset-Änderung, direkte Nutzung in einer abweichenden Datei.
  Eingegrenzt (Baustein B): die Korpora sind verschiedene Datei-Schließungen (stress hat drei
  zusätzliche Wurzeln), was eine korpus-spezifische fresh-vs-cached-Verschiebung plausibel
  macht. **Nicht** bis zur einzelnen Instanziierung isoliert (bräuchte `--generateTrace`).
  Konsequenz für die Mess-Regeln: „bench:editor-Δ = stress-Δ" ist eine Faustregel, kein Gesetz.
- Im Rumpf einer noch generischen Funktion hovert `NestedArray<S>` unaufgelöst — gewöhnliches
  TS-Verhalten für offene Typparameter.

## Post-Verification-Addendum (2026-09-24)

- **Baustein 0** (vor dem Code): ein Blocker (Union gleichen Rangs → `number[][] | number[][]`)
  und fünf kleinere Befunde, alle in Spec v2 eingearbeitet (Addendum dort).
- **Baustein A** (Spec-Konformität): MERGE-AFTER-FIXES. Alle Gates exakt reproduziert; eigener
  Mutant (Akkumulator-Start `number[]`) von 12+ Pins gefangen. Befund MAJOR: der in D7 zugesagte
  Pin unter `noUncheckedIndexedAccess` fehlte → geschlossen in `fcbea01` (erste Fassung des
  Pins nutzte `?.`, das `undefined` flag-unabhängig hinzufügt — vakuös; auf `[0]![1]` korrigiert
  und per Gegenmutante belegt).
- **Baustein B** (adversarial): MERGE. 5/5 Mutanten gefangen (u. a. vertauschte Zweige, die bei
  dynamischem Rang `number` behauptet hätten; Rang-0-Laufzeitkontrakt; lokale Typen im
  Package-Smoke — deren vermutete Vakuität widerlegt). M2 gegen echte Op-Ketten gehalten, inkl.
  dynamischem `keepdims: boolean` (korrekt → `NestedValue`).
- **Baustein C** (Covenant): M1/M2/M4/M5/S1/Z1/Z2 halten. **M3: Spannung** — `NestedValue` in
  Rückgabe-Position verletzt den v6-Wortlaut, obwohl ein rekursiver Typ nicht auflösbar ist.
  **Owner-Entscheidung 2026-09-24: COVENANT v7** präzisiert M3 (rekursive, exportierte Aliase
  konform; Wrapper und auflösbare generische Aliase weiterhin Verstöße).
- **Re-Verifikation gegen v7** (frischer `covenant-verify`): M2 und M3 **halten** — `NestedValue`
  ist genuin rekursiv (ndarray.ts:257) und type-only aus index.ts exportiert; `NestedArray<S>`
  ist kein Wrapper, sondern löst für konkrete `S` auf. Die Klausel selbst ist präzise und
  schließt den S3-Fall aus. Zwei Nebenbefunde: die v7-Klausel stand mitten in der v6-Liste
  (ans Ende von M3 verschoben); offene Typparameter in generischen Rümpfen benennt der Text
  nicht ausdrücklich — bewusst NICHT eigenmächtig ergänzt, als v8-Kandidat in FOLLOWUPS.
- **Lint** auf frisch gebautem Graph: 0 Fehler, 0 Warnungen (s. Commit).
