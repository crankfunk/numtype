# dt5 — `sqrt`, `argmax`, `topk`, `stack` auf allen dtypes — Ergebnisse

**Spec:** docs/dtype-dt5-spec.md v1.1 (Stufe 3b) · **Datum:** 2026-10-14 · **Commits:** 296e1f6 …
8c0887d auf `main` (unveröffentlicht; Release 0.4.0 als nächster Schritt) · **Covenant:** v13.

## Ergebnis

Die dtype-Umsetzung auf `NDArray` ist abgeschlossen: keine Op ist mehr übergangsweise für einen dtype
gesperrt; `DTypeLock`, `lockedOpMessage` und `assertFloat64Locked` sind gelöscht.
- **`sqrt`:** float64 → float64, float32 → float32 (`Math.fround(Math.sqrt(x))`, korrekt gerundet),
  int32 → float64; bool wirft dauerhaft nur zur Laufzeit (`BOOL_ARITHMETIC_MESSAGE`, Präzedenz
  `norm()`), Rückgabe-dtype `SqrtDType` (bool → `never`, `any`/Union → `DType`).
- **`argmax`/`topk`:** alle dtypes inkl. bool; Achsen-`argmax` und `topk().indices` liefern int32 —
  **auch für float64** (E6, Verhaltensänderung gegenüber 0.3.0; `WNDArray` bleibt bis dt6 float64).
  `topk`-Werte bit-exakt über Ganzzahl-Sichten (float32 `Uint32Array`, float64 `BigUint64Array`).
- **`stack`:** alle Zeilen gleicher dtype → `NDArray<…, D>`; verschiedene dtypes → eigene Meldung am
  Argument, wortgleich zur Laufzeit, Shape-Fehler zuerst; Zeilen mit `any`/Union-dtype werden
  übersprungen (`StackDTypeCheck`). Kopie per `TypedArray#set` — ECMA-262
  SetTypedArrayFromTypedArray kopiert bei gleichem Elementtyp byteweise, also bitgenau inkl. NaN-Payload
  (Owner-Entscheidung nach C1, ersetzt die in F4 vorgesehenen Ganzzahl-Sichten). Die M3-v8-Ausnahme
  (native `__varianceD`-Meldung) ist entfallen.

## Zahlen

| Gate | vorher | nachher |
|---|---|---|
| check:diag Root | 246,763 @ 140 | **249,296 @ 140** (Δ+2,533; Gate +2,500 um 33 überschritten — Owner-akzeptiert) |
| Messpunkt A (`argmax`/`topk`/`sqrt`) | — | 246,065 (−698; Stopp +500) |
| Messpunkt B (`stack`, Aufräumen) | — | 247,759 (+996; Stopp +1,800) |
| Messpunkt C1 (Laufzeittests) | — | 249,296 (+2,533; Stopp +2,500 → **gestoppt**) |
| C1b / C2 / D | — | Root unverändert 249,296 |
| check:diag:dtype | 67,108 @ 81 | 75,868 @ 81 (+8,760, Typ-Pins; kein Gate) |
| check:diag:stress | 123,138 | 123,106 |
| check:diag:browser | 2,142 | 2,142 |
| bench:editor | `{44,996; 46,916; 77,673; 45,059; 50,695; 51,312; 43,909; 51,982}` | `{44,964; 46,884; 77,641; 45,027; 50,663; 51,280; 43,877; 52,201}` |
| test:core | 1,668 | 1,677 |
| resident / threaded / browser / package / Freeze / cargo | 6,155+2 / 139 / 4 / 3 / `2a54d9fd…` / 222+1 | unverändert |

**bench:editor:** w1–w7 uniform −32 (= stress-Δ, Klassen-Surface). w8 +219 = −32 + 202 (der einzige
`stack`-Aufruf: Fold über die Zeilen-dtypes) + 49 (sechs `topk`-Aufrufe) — Auslassungsmessung des
Implementierers; die Summe ist von A und B reproduziert.

**Root-Überschreitung:** die Implementierung kostet +996, die Laufzeittests +1,537 (Baustein 0 hatte
für diesen Schritt nur ~+132 geschätzt); der größte Posten ist die tsc-Fixture mit Thunk-Arrays (~+891).
Erstmals seit infra-02 lag die Überschreitung nicht an Typ-Pins — die liegen im dtype-Korpus.

## Post-Verification-Addendum

- **Baustein 0:** ein Blocker — `SqrtDType` ohne `IsAnyDType`-Gate behauptete für `AnyNDArray.sqrt()`
  falsch `never` (M2); behoben vor dem Code. A2 empirisch korrigiert (`argmax-topk.test.ts` unberührt;
  betroffen `resident.test.ts:2452` und der Import in `scalar-mean.test.ts`, ohne den die ganze Datei
  nicht lädt). Gate von +5,500 auf +2,500 gestrafft (gemessen +1,031 im Entwurf). v13 nachgeschärft.
- **Umsetzung:** Stopp nach C1 korrekt ausgelöst (+33), nichts gekürzt; Owner akzeptierte und
  entschied `TypedArray#set` für `stack`. Kein Commit mit rotem Baum (kompilier-notwendige A2-Edits im
  selben Commit). Abweichungen, von A bewertet: A2.2 in Commit A statt B (zwingend); der Array-Guard
  im Fold (T-5) gestrichen — variadische Tupel prüfen jetzt ihre festen Zeilen (sound, gepinnt);
  `SqrtDType` inline; dt1-K7 in FOLLOWUPS nur zur M2-Hälfte geschlossen (der M1-Teil bleibt bis dt6).
- **Baustein A:** MERGE; alle 7 Messpunkte reproduziert, je Commit `pnpm check` grün; Clean-Rebuild
  Freeze-Hash, `cargo test`, `demo`; ECMA-262-Regel im Original nachgelesen; 19 eigene Mutanten.
- **Baustein B:** HÄLT; `sqrt` float32 gegen eigene BigInt-Referenz über 15,1 Mio. Fälle (alle
  Subnormalen; echte Rundungs-Ties existieren für float32-`sqrt` nicht), `argmax` 44,544, `topk` 23,838,
  `stack` 147 Fälle — null Abweichungen; 57 Mutanten ohne echten Überlebenden.
- **Baustein C:** kein Covenant-Verstoß.
- **Bewusst hingenommen und offengelegt:**
  - **Folgemeldungen nach `sqrt()` über bool** (`NDArray<S, never>`, das zur Laufzeit ohnehin wirft):
    breiter als die drei in A6 genannten — Zuweisung an jeden Typ mit festem dtype (`AnyNDArray`,
    `NDArray<S, "float32">`, `NDArray<S, any>`) zeigt die native `__varianceD`-Meldung; die
    Skalar-Formen von `add`/`sub`/`mul`/`div` und der Vergleiche zeigen die generische TS2769 (wie die
    Skalar-Ausnahme aus M3 v9). Kein falscher Anspruch; v13 deckt die `__varianceD`-Fälle allgemein.
  - **`stack` mit Union-/`any`-Zeile vor einem Konflikt:** der Editor nennt die erste DEFINITE Zeile,
    die Laufzeit die erste Zeile — beide lehnen ab, der Meldungsstamm ist gleich, die genannten dtypes
    können abweichen (die Laufzeit kennt die Union-Information nicht).
  - **Variadische Tupel:** Zeilen im Rest bleiben ungeprüft (Laufzeit prüft).
  - Mutanten „ohne `Math.fround`" (äquivalent: der Float32Array-Store rundet identisch, konstruiert
    belegt) und „`gatherBits` float64 ohne Sicht" (auf V8 nicht beobachtbar) überleben ehrlich.
- **Nacharbeit (FOLLOWUPS):** `toFloat64` in `argmaxTyped`/`topkTyped` ist überflüssig (die
  float64-Orakel indizieren nur), kostet O(n) Speicher für Nicht-float64; E6-Folgen für die
  0.4.0-Release-Notes (README.md:188, examples/rag-demo, `Int32Array` passt nicht in `fromArray`).
