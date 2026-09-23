# 0b — Typisiertes `toNestedArray` — bindende Spec (Stufe 3b)

**Version:** v1.1 (2026-09-24) · **Status:** Owner-Richtung abgenommen 2026-09-24 (alle drei
Entscheidungen wie empfohlen, s. unten) → Baustein 0 läuft
**Stufe:** 3b (CLAUDE.md, Eskalationsleiter): neue Typ-Maschinerie auf einer öffentlichen
Signatur + Umkehr einer gepinnten Entscheidung (D-V2.2/B5). Voller Katalog: Baustein 0 vor dem
Code, danach A + B + C parallel, Lint im Gate-Block, Ergebnis-Doc mit Addendum.
**Berührte Covenant-Invarianten:** **M2** (never wrong, only incomplete), **M3** (saubere Hovers,
seit v6 inkl. Methoden-Rückgabetypen). Nicht berührt: M1/M4 (kein Rust, Laufzeit unverändert),
M5, Z1, Z2, S1.
**Anlass:** Stand-Review 2026-09-23, Befund F6: der einzige Weg, Daten verschachtelt
herauszuholen, ist untypisiert — ausgerechnet in einer Bibliothek, deren USP statische Typen sind.

## Ziel

`NDArray<S>.toNestedArray()` und `WNDArray<S>.toNestedArray()` liefern einen am **Rang** von
`S` berechneten Typ statt `unknown`: `[2, 3]` → `number[][]`, `[]` → `number`,
`[number, 3]` → `number[][]`. Wo der Rang statisch nicht feststeht, ein ehrlicher rekursiver
Typ statt einer Behauptung.

## Vorab-Befunde (2026-09-24, Probe in Scratch gegen tsc 7.0.2, nicht im Repo)

Eine Wegwerf-Probe mit dem geplanten Typ hat zwei Designfragen VOR der Spec entschieden. Die
Probe ist informell (Arbeitsregel 8) — Baustein 0 und Verify messen am echten Code nach:

1. **`NDArrayView<out S>` kann den Typ NICHT tragen.** Ein rechnender Rückgabetyp unter `out S`
   wirft TS2636 („`NestedArray<sub-S>` is not assignable to `NestedArray<super-S>`") —
   dieselbe Klasse wie Spike 05 (`Transpose`). → D3.
2. **Der Methoden-Hover löst sauber auf** (echter `tsc --lsp --stdio`, Regel 13):
   `(method) Arr<[2, 3]>.toNestedArray(): number[][]`; Variablen `const n: number[][]`,
   Rang 0 `number`, `[number, 3]` → `number[][]`. Anders als S3s `StackResultOf` bleibt der
   Alias NICHT stehen, weil die Auflösung in einem nicht-generischen Array-Typ endet. Nur der
   dynamische Fall zeigt einen Namen: `toNestedArray(): NestedValue` — ein rekursiver Typ ist
   nicht expandierbar. → D4.

## Design (D1–D7)

- **D1 — der Typ.** Zwei exportierte Aliase auf Modulebene in `spike/src/ndarray.ts` (keine
  neue Datei: eine neue Datei kostet ±≈7.000 Order-Noise):
  ```ts
  export type NestedValue = number | NestedValue[];
  export type NestedArray<S extends Shape> =
    RankUnknowable<S> extends true ? NestedValue      // dim.ts:82, wiederverwendet
    : S extends readonly [] ? number
    : S extends readonly [unknown, ...infer R extends Shape] ? NestedArray<R>[]
    : NestedValue;
  ```
  **Nur der Rang zählt, nie die Dim-Werte** — `number[]` statt Länge-N-Tupel. Tupel pro Dim-Wert
  wäre die verbotene Tupellängen-Arithmetik über große Dimensionen (CLAUDE.md, TS-Limits) und
  würde bei `[1000, 1000]` das Budget sprengen. Tail-rekursiv über den Rang (Tiefe = Rang ≤ ≈100
  unkritisch; der Rang-1024-Cliff der Scale-Probe betrifft ohnehin schon die Shape-Typen).
  `RankUnknowable` schließt sowohl dynamischen Rang (`number[]`) als auch Unions über die
  Länge (`[2] | [2, 3]`) ab, **bevor** destrukturiert wird (Arbeitsregel 3). Eine Union
  gleichen Rangs (`[2, 3] | [4, 5]`) ergibt korrekt `number[][]`.
- **D2 — Signaturen der Klassen.** `NDArray.toNestedArray(): NestedArray<S>` und
  `WNDArray.toNestedArray(): NestedArray<S>`; Laufzeit unverändert, ein Cast an der Rückgabe
  (`as NestedArray<S>`), wie die Ops es mit `OkShape` tun. **Abweichung von der Hausregel
  „TS-Klassenkörper insertion-only"**: die Signatur eines BESTEHENDEN Members ändert sich. Das
  ist der Zweck der Scheibe und nicht vermeidbar (ein zweiter Member daneben wäre eine
  API-Dublette). Zur Owner-Bestätigung vorgelegt (s. „Entscheidungen").
- **D3 — `NDArrayView` bleibt `unknown`.** Belegt durch Vorab-Befund 1; `implements` erlaubt
  einen engeren Rückgabetyp auf den Klassen. Der View-Doc-Kommentar erklärt, warum.
- **D4 — `NestedValue` wird aus `index.ts` exportiert** (type-only), `NestedArray` ebenfalls —
  damit der einzige benannte Hover-Fall für Konsumenten nachschlagbar und benennbar ist.
- **D5 — M2-Kanten.** Jede Variante muss „nie falsch" sein: Rang 0 liefert zur Laufzeit wirklich
  eine Zahl (beide Klassen: `data[offset] ?? 0`); size-0-Shapes (`[0, 3]`) liefern `[]` —
  typkorrekt als `number[][]`. **Zu klären in Baustein 0:** `AnyNDArray` (`NDArray<any>`),
  `never`-Shapes und `Shape` selbst (`readonly number[]`) — erwartet `NestedValue`, nie `never`
  oder `number` (FOLLOWUPS kennt `never`-Verdikte als vorbestehende Klasse).
- **D6 — Pins neu.** Die drei `Equal<…, unknown>`-Pins (spike/tests/ndarray.test-d.ts:216-218)
  werden ersetzt, ihr Zweck (Drift zwischen den Deklarationen fangen) bleibt:
  View-Pin bleibt `unknown`; NDArray und WNDArray je gegen dieselben exakten Erwartungen
  (Rang 0/1/2/3, `[number, 3]`, `number[]`, `Shape`, `[2]|[2,3]`, `[2,3]|[4,5]`, `AnyNDArray`)
  plus ein Pin, dass beide Klassen für dasselbe `S` denselben Typ liefern. Keine neue Testdatei.
- **D7 — Konsumenten-Nachweis.** Im `consumer-strict`-Smoke: `NDArray.fromArray([2, 3], …)
  .toNestedArray()[0]![1]` typcheckt als `number | undefined` (mit
  `noUncheckedIndexedAccess`) bzw. `number` (ohne) — ohne Cast. README: erst mit dem Release
  eintragen (Arbeitsregel 19).

## Nicht-Ziele

Typisierung der View (`NDArrayView`) · Tupel mit Dim-Längen · Änderung der Laufzeit ·
`toArray()`/`data` · jede Op-Änderung · dtype (eigene Scheibe danach — `NestedArray` wird dort
ein zweites Argument brauchen; dieses Design sperrt das nicht).

## Gates (Absolut-Grenzen, VOR der Messung registriert)

- `pnpm check` Exit 0 · check:diag Root **≤ +4.000** gegen 226.220 @ 140 (Dateiset bleibt 140).
  Begründung: ein generischer Member-Rückgabetyp rippelt über die Klassen-Surface
  (Mechanismus 3); die Probe war ohne echten Korpus nicht aussagekräftig.
- stress/browser berichten · `bench:editor`: Pins nur bei uniformer, dem stress-Δ gleicher
  Verschiebung neu setzen · Freeze-Hash unverändert `2a54d9fd…` · alle Testzahlen gleich oder
  um die neuen gewachsen · `test:package` inkl. beider Konsumenten-Smokes grün · Lint auf frisch
  gebautem Graph.
- **LSP-Messung Pflicht** (M3, Regel 13): Methoden-Hover und Variablen-Hover für Rang 0/2,
  `[number, 3]` und dynamisch, auf NDArray UND WNDArray, dazu der Klassen-Hover `NDArray<[2, 3]>`
  als Kontrollpunkt.

## Testplan

- Typebene (D6) in `spike/tests/ndarray.test-d.ts`, Diagnose-Inhalte, nicht nur Existenz.
- Laufzeit: Rang 0 liefert `typeof === "number"` auf beiden Klassen; size-0 liefert `[]`;
  eine View (Transpose) liefert dieselbe Verschachtelung wie ihr materialisiertes Gegenstück
  (Klasse per Regel 12 assertiert).
- Mutanten (A und B je eigene): u. a. `NestedArray<R>[]` → `NestedArray<R>` (Rang um eins falsch),
  `RankUnknowable`-Gate entfernt (Union-Rang → falsche Behauptung, M2-Bruch).

## Entscheidungen für den Owner (vor Baustein 0) — ENTSCHIEDEN 2026-09-24

Owner: (1) Abweichung D2 bestätigt · (2) `NestedValue` · (3) nicht allein veröffentlichen,
mit dem nächsten Minor (0.4.0) bündeln.


1. **Hausregel-Abweichung D2** (Signatur eines bestehenden Members ändert sich): bestätigen?
2. **Dynamischer Fall:** `NestedValue` (rekursiv, nützlich, im Hover benannt) — oder `unknown`
   (wie heute, kein neuer Name)? Empfehlung: `NestedValue`.
3. **Release:** Die Verengung von `unknown` kann bestehenden Konsumenten-Code brechen (z. B. ein
   Cast `as string` wird zum Fehler) → nach der SemVer-Policy ein Minor, also 0.4.0 — nicht
   allein veröffentlichen, sondern mit dem nächsten Paket bündeln?
