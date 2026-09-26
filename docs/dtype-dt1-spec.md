# dt1 — dtype-Kern auf `NDArray` — bindende Spec (Stufe 3b)

**Version:** v2 (2026-09-26) · **Status:** Owner-abgenommen (A1–A3), Baustein 0 gelaufen, drei
A-Punkte vom Owner nachentschieden 2026-09-26 → **implementierungsreif**

**Änderungslog v1 → v2 (Baustein 0, Addendum am Ende):** A1 um das Feld `data` ergänzt (Owner);
A3 ohne die zwei dt2-Sätze (dtype-Skalar-Ausnahme, Anker `sym:Promote` → wandern in die
dt2-Covenant-Änderung) und mit befristeter `stack`-Ausnahme (Owner); K3 präzisiert (Allokation
in `transpose`/`slice`, `reshape`/`flatten` inline in ndarray.ts, `toJSON` bool); K6 zwei Stellen;
Vorprüfung und Budget-Messpunkt als erste Umsetzungsschritte.
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
  `number[]` bzw. `boolean[]`), `toString`/inspect. **v2 präzisiert (Baustein 0):**
  (a) `transposeRuntime`/`sliceRuntime` (runtime.ts) allozieren ihre Ausgabe fest als
  `Float64Array` — dt1 hängt typ-generische Zwillinge an, die die Ausgabe in DERSELBEN
  Typed-Array-Klasse wie die Eingabe anlegen (Muster: `zerosData`/`convertToDType` des Prototyps);
  die bestehenden Funktionen bleiben byte-gleich. Dafür gibt es KEINE Prototyp-Vorlage — erster
  Umsetzungsschritt ist die Vorprüfung unten. (b) `reshape`/`flatten` kopieren inline in
  ndarray.ts (`new Float64Array(this.data)`) — Änderung dort (A1 deckt es). (c) `toJSON` liest
  heute `Array.from(this.data)`; bei bool muss `v !== 0` zu `boolean` werden (sonst M2: Typ sagt
  `boolean[]`, Laufzeit liefert 0/1). NDArray hat keine Views: alle Bewegungs-Ops kopieren. Damit entfallen die Laufzeit-Sperren für die niladischen Bewegungs-Ops
  (`transpose`, `flatten`) schon in dt1.
- **K4 Sperren** (O2 a) für alle rechnenden Ops bei D ≠ float64: `add`/`sub`/`mul`/`div`
  (Array- und Skalar-Form), `matmul`, `dot`, `cosineSimilarity`, `sum`, `mean`, `argmax`, `topk`,
  `stack`, `sqrt`, `norm`. Guard-Meldung am Argument, wortgleich zur Laufzeit; Ops ohne
  Argumentposition (`sqrt`, `norm`, 0-arg `mean`/`argmax`) nur zur Laufzeit (Grenze 2 der
  Ergebnisse — per M2-Erweiterung unten gedeckt).
- **K5 Defekte aus dem Prototyp:** `AnyNDArray = NDArray<any, any>` (F1); Test mit zwei
  VERSCHIEDENEN Nicht-float64-dtypes an einer gesperrten Zwei-Operanden-Op (F4).
- **K6 Messwerkzeug:** `spike/bench-dx/gen-workloads.ts` erwartet Hovers als `NDArray<[8]>` — an
  ZWEI Stellen: `fmtShape` (Z. 55) und die feste Erwartung `"NDArray<readonly number[]>"` (Z. 610,
  W7). Beide auf die neue Form mit dtype anpassen, danach Pins neu.
- **K7 Covenant** (unten) und FOLLOWUPS: die neuen kernel-losen Referenzfunktionen als
  Paritätslücke eintragen (M1 v5).

**Nicht in dt1:** Promotion und gemischte Arithmetik (dt2), Reduktionen (dt3), Vergleiche/`where`
(dt4), `sqrt`/`argmax`/`topk`/`stack` für andere dtypes (dt5), `WNDArray` und WASM (dt6+).
`WNDArray` bleibt einparametrig und float64; `NDArrayView` bleibt dtype-frei.

## Owner-Abnahmen vor Baustein 0

**A1 — Methodenliste nach Hausregel (b)** (bestehende Member dürfen geändert werden): an `NDArray`
Konstruktor, `zeros`, `ones`, `fromArray`, `stack`, `add`, `sub`, `mul`, `div`, `matmul`, `sum`,
`mean`, `dot`, `norm`, `cosineSimilarity`, `transpose`, `slice`, `reshape`, `flatten`, `argmax`,
`topk`, `sqrt`, `item`, `toNestedArray`, `toJSON`, `toString`, inspect, **das Feld `data`** (Typ
`DataOf<D>` — v2, Owner-nachentschieden 2026-09-26). Neu: `dtype`, `astype`,
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
> **M3 · Präzisierung v8 — befristete Ausnahme `stack`:** lehnt `stack` eine Zeile wegen ihres
> dtypes ab, zeigt der Editor bis zur Scheibe dt5 die native TS-Strukturmeldung statt einer
> eigenen; die Laufzeit wirft die eigene Sperrmeldung. Entfällt mit dt5 (dann trägt `stack` jeden
> dtype). Im Quelltext an `stack` dokumentiert.
>
> **M2 · Präzisierung v8 — dtype-Maschinerie:** M2 gilt auch für die dtype-Maschinerie
> (`Promote`, die dtype-Sperren, die Laufzeit-Tabellen in `runtime.ts`): Ablehnung nur für
> garantierte Laufzeit-Throws; Union- und breite dtypes degradieren zu `DType` (kein Anspruch).
> Übergangsweise nur zur Laufzeit gesperrte Ops ohne Argumentposition gelten als „unvollständig,
> nicht falsch", solange die Laufzeit-Meldung wortgleich zur Sperrmeldung ist und die Sperre in
> FOLLOWUPS getrackt wird. Anker ergänzt: `sym:DTypeLock`.

*(v2: die dtype-Skalar-Ausnahme der M3 und der Anker `sym:Promote` gehören in die dt2-Covenant-
Änderung — in dt1 existiert weder `Promote` noch die Skalar-Regel, beides wäre ungeprüft.)*

## Reihenfolge der Umsetzung (v2)

1. **Vorprüfung K3-Allokation:** typ-generische `transpose`/`slice`-Zwillinge für alle vier dtypes
   bauen und gegen die float64-Ergebnisse (umgewandelt) sowie gegen den Eingabe-dtype prüfen —
   bevor der Rest entsteht.
2. **Budget-Messpunkt:** nach K1 + K2 + K4 + K5 (ohne Tests) check:diag messen. Liegt der Wert
   schon über +8,000 (weniger als 4,000 Reserve für Tests und K3), STOPP und Owner-Rückmeldung
   statt Weiterbauen — die Herleitung der Grenze stützte sich auf addierte Prototyp-Stufen, was
   die Mess-Regeln (Super-Additivität) nicht tragen.
3. Rest von K3, K6, K7, Tests. Hinweis aus dem Prototyp: eine gesperrte Skalar-Überladung ohne
   eigenen Typparameter scheitert an TS2394 — Platzhalter `<DD extends DType = D>` nötig.

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

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-09-26)

`brainroute:deep`, eigener Worktree am `main` 1f161d9. **Blocker:** das Feld `data` fehlte in A1
(→ Owner, freigegeben). **Major:** K3-Allokation ohne Prototyp-Vorlage (`transposeRuntime`/
`sliceRuntime` allozieren `Float64Array`); `toJSON` bool → M2-Risiko; Budget-Herleitung addiert
Stufen aus einem anderen Maschinerie-Kontext (→ Messpunkt). **Minor:** `reshape`/`flatten` ohne
runtime.ts-Funktion; `stack` mit nativer Meldung (→ befristete Ausnahme, Owner); zwei dt2-Sätze in
A3 (→ nach dt2, Owner); K6 zwei Stellen; Test-/Pin-Anteil der Herleitung nicht nachvollziehbar.
**Hält:** A2 empirisch bestätigt — Prototyp-Kern + `AnyNDArray`-Fix bricht genau den einen Pin
`NDAnyTop`, keinen weiteren (check, core 1591, resident 6155+2 grün); die A1-Liste ist bis auf
`data` vollständig (29 Member abgeglichen); NDArray hat keine Views; `backend()`/`strides` sicher;
Baseline 227,405 und Prototyp-Stand 234,385 exakt reproduziert. **Nebenbefund:** der
`AnyNDArray`-Fix war im Prototyp nie gebaut — Baustein 0 hat ihn als Erster geprüft.
