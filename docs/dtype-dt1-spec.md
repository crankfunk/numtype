# dt1 — dtype-Kern auf `NDArray` — bindende Spec (Stufe 3b)

**Version:** v1 (2026-09-26) · **Status:** Owner-abgenommen 2026-09-26 (A1 Methodenliste, A2 zwei
benannte Test-Änderungen, A3 Covenant-v8-Wortlaut) → Baustein 0 läuft
**Stufe:** 3b — Covenant-Änderung (M3 v8, M2-Erweiterung) und neue Typ-Maschinerie auf `main`.
**Berührte Invarianten:** M2, M3 (Änderung), M1 (Tracking), M5, Z1.
**Grundlage:** docs/dtype-design-spec.md v2.1 (bindet D1–D10) · docs/dtype-design-ergebnisse.md
(Prototyp-Messung, Defekte, Grenzen) · Prototyp-Code auf lokalem Branch `proto/dtype` (059d852),
wiederverwendbar.

## Ziel

`NDArray<S, D>` mit den vier dtypes wird auf `main` eingeführt — Speicher, Erzeugung, Umwandlung,
Auslesen und die dtype-NEUTRALEN Ops laufen für jeden dtype; alle rechnenden Ops bleiben für
D ≠ float64 gesperrt, bis dt2–dt5 sie freischalten. Bestehender Code, der keinen dtype nennt,
verhält sich exakt wie heute. **Keine Veröffentlichung** vor dt5 (Arbeitsregel 19).

## Umfang

- **K1 Typparameter und Speicher** (Design D1/D2): `DType`, `DataOf<D>`, `NDArray<S, D = "float64">`,
  Feld `readonly dtype: D`, `data: DataOf<D>`, `toArray(): DataOf<D>`, Varianz-Marker
  `__varianceD` (Prototyp-Befund: macht die Invarianz in `D` bewusst).
- **K2 Erzeugung und Umwandlung** (D3): `fromArray` mit Typed-Array-Ableitung und `{ dtype }`,
  `zeros`/`ones(shape, dtype?)`, `astype` mit den D3-Regeln (strikte Throws).
- **K3 dtype-neutrale Ops für jeden dtype** (D5, „D unverändert"): `transpose`, `slice`, `reshape`,
  `flatten`, `item` (bool → `boolean`), `toNestedArray` (`NestedArray<S, D>`, bool-Blätter
  `boolean`, `NestedBoolValue` rekursiv exportiert wie `NestedValue`), `toJSON` (`data` als
  `number[]` bzw. `boolean[]`), `toString`/inspect. Dafür typ-generische Laufzeit-Varianten der
  Bewegungs-Referenzfunktionen, **angehängt** an `runtime.ts`; die bestehenden float64-Funktionen
  bleiben byte-gleich. Damit entfallen die Laufzeit-Sperren für die niladischen Bewegungs-Ops
  (`transpose`, `flatten`) schon in dt1.
- **K4 Sperren** (O2 a) für alle rechnenden Ops bei D ≠ float64: `add`/`sub`/`mul`/`div`
  (Array- und Skalar-Form), `matmul`, `dot`, `cosineSimilarity`, `sum`, `mean`, `argmax`, `topk`,
  `stack`, `sqrt`, `norm`. Guard-Meldung am Argument, wortgleich zur Laufzeit; Ops ohne
  Argumentposition (`sqrt`, `norm`, 0-arg `mean`/`argmax`) nur zur Laufzeit (Grenze 2 der
  Ergebnisse — per M2-Erweiterung unten gedeckt).
- **K5 Defekte aus dem Prototyp:** `AnyNDArray = NDArray<any, any>` (F1); Test mit zwei
  VERSCHIEDENEN Nicht-float64-dtypes an einer gesperrten Zwei-Operanden-Op (F4).
- **K6 Messwerkzeug:** `spike/bench-dx/gen-workloads.ts` erwartet Hovers als `NDArray<[8]>` —
  auf die neue Form `NDArray<[8], "float64">` anpassen (eine Formatfunktion), danach Pins neu.
- **K7 Covenant** (unten) und FOLLOWUPS: die neuen kernel-losen Referenzfunktionen als
  Paritätslücke eintragen (M1 v5).

**Nicht in dt1:** Promotion und gemischte Arithmetik (dt2), Reduktionen (dt3), Vergleiche/`where`
(dt4), `sqrt`/`argmax`/`topk`/`stack` für andere dtypes (dt5), `WNDArray` und WASM (dt6+).
`WNDArray` bleibt einparametrig und float64; `NDArrayView` bleibt dtype-frei.

## Owner-Abnahmen vor Baustein 0

**A1 — Methodenliste nach Hausregel (b)** (bestehende Member dürfen geändert werden): an `NDArray`
Konstruktor, `zeros`, `ones`, `fromArray`, `stack`, `add`, `sub`, `mul`, `div`, `matmul`, `sum`,
`mean`, `dot`, `norm`, `cosineSimilarity`, `transpose`, `slice`, `reshape`, `flatten`, `argmax`,
`topk`, `sqrt`, `item`, `toNestedArray`, `toJSON`, `toString`, inspect. Neu: `dtype`, `astype`,
`toArray`, `__varianceD`. In `runtime.ts` nur Anhänge. `backend()` und `strides` unverändert.

**A2 — zwei benannte Änderungen an bestehenden Test-Artefakten** (Hausregel (b) verbietet sie
sonst): (1) der Pin `NDAnyTop` in `spike/tests/ndarray.test-d.ts` weitet sich von `NestedValue` auf
`NestedValue | NestedBoolValue`, weil `AnyNDArray` jetzt jeden dtype umfasst (zwingende Folge von
K5); (2) die Hover-Erwartungen des Editor-Messwerkzeugs (K6). Beide ändern keine
Verhaltensaussage über float64-Code.

**A3 — Covenant-Wortlaut** (Version 8):

> **M3 · Präzisierung v8 — dtype im Klassen-Hover:** Klassen mit dtype-Parameter zeigen im Hover
> die aufgelöste Shape als Tupel UND den dtype als String-Literal (`NDArray<[2, 3], "float64">`),
> auch wenn er dem Default entspricht (TypeScript blendet Default-Typargumente nicht aus —
> gemessen). Klassen ohne dtype-Parameter (derzeit `WNDArray`) bleiben bei `WNDArray<[2, 3]>`.
>
> **M3 · Präzisierung v8 — benannte Ausnahme dtype-Skalar:** scheitert ein SKALAR-Argument an
> einer dtype-Regel (Arithmetik auf bool, nicht-ganzzahliges Literal auf int32), zeigt der Editor
> die generische TS2769 der letzten Überladung statt der eigenen Meldung; die Laufzeit wirft die
> eigene. Begründung gemessen: die einzige korrekte Alternative (Ein-Signatur mit
> `IsUnion`-Gate) kostete +8,495 Instantiations. Im Quelltext an den Überladungen dokumentiert
> (Präzedenz W4/W5).
>
> **M2 · Präzisierung v8 — dtype-Maschinerie:** M2 gilt auch für die dtype-Maschinerie
> (`Promote`, die dtype-Sperren, die Laufzeit-Tabellen in `runtime.ts`): Ablehnung nur für
> garantierte Laufzeit-Throws; Union- und breite dtypes degradieren zu `DType` (kein Anspruch).
> Übergangsweise nur zur Laufzeit gesperrte Ops ohne Argumentposition gelten als „unvollständig,
> nicht falsch", solange die Laufzeit-Meldung wortgleich zur Sperrmeldung ist und die Sperre in
> FOLLOWUPS getrackt wird. Anker ergänzt: `sym:Promote`, `sym:DTypeLock`.

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +12,000** gegen 227,405 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert (Tests in bestehende Dateien anhängen; eine neue Datei nur mit empty-then-fill-
  Zerlegung). Herleitung: Prototyp-Kern (Stufen 1, 2, 7, Marker) ≈ +5,500 plus Test-/Pin-Anteil
  im Prototyp-Verhältnis ≈ +4,000, plus Reserve für K3.
- stress/browser berichten · Freeze-Hash unverändert · alle bestehenden Tests unverändert grün
  (außer A2) · neue Tests: Rundreisen je dtype, `astype`-Ränder (aus der Prototyp-Probe),
  Bewegungs-Ops je dtype inkl. Views mit assertierter Klasse (Regel 12), Sperr-Meldungen per
  LSP (Regel 13) · `bench:editor` nach K6 neu gepinnt, Latenz gemessen (im Prototyp nicht
  möglich) · Lint auf frisch gebautem Graph.

## Verify

Baustein 0 vor dem Code, dann A + B + C parallel. B bekommt ausdrücklich: M2 an K3 (bewahren die
Bewegungs-Ops den dtype zur Laufzeit wirklich?), K4-Vollständigkeit (fehlt eine Op?), `AnyNDArray`-
Flüsse, float64-Code bitgleich zu vorher.
