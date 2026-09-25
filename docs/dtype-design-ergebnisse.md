# dtype-Design — Ergebnisse (Prototyp-Messung)

**Spec:** docs/dtype-design-spec.md v2.1 (Stufe 3b) · **Datum:** 2026-09-25 ·
**Prototyp:** Branch `proto/dtype` (1496518..059d852 auf `adb4a0c`), NICHT gemergt.
**Ehrlichkeitsregel:** jede Zahl stammt aus einem in dieser Scheibe ausgeführten Lauf; Baustein A
hat Baseline, Stufe 7, Varianz-Stufe und Endstand in frischen Worktrees exakt reproduziert,
Baustein B die beiden Rückgänge (−466, −3,324).

## Ergebnis in einem Satz

Volles dtype auf `NDArray` ist baubar und in der Maschinerie bezahlbar (+6,980 für den Kern,
hochgerechnet ≈ +16,700 für die ganze Oberfläche, ≈ 7 %); die Laufzeit ist korrekt (18
adversariale Proben, 0 Fehler), aber drei Versprechen der Spec sind mit TypeScript 7 so NICHT
einlösbar, und zwei Defekte müssen vor dem Rollout behoben werden.

## Messung (Root check:diag, Baseline 227,405 @ 140)

| Stufe | Wert | Δ |
|---|---|---|
| 1 bloßer Parameter | 227,944 | +539 |
| 2 Speicher/Erzeugung/`astype` | 232,765 | +4,821 |
| 3 `Promote` + `add` (Array) | 232,299 | −466 |
| 4 Skalar-Regel D6 | 234,097 | +1,798 |
| 5 `sum` | 234,015 | −82 |
| 6 `gt` | 234,251 | +236 |
| 7 Sperren übrige Ops | 237,709 | +3,458 |
| Varianz-Marker `__varianceD` | 234,385 | −3,324 (allein durch die eine Deklaration, per Kommentar-Kontrolle belegt) |
| 8a leere Testdatei | 238,185 @ 141 | +3,800 (Order-Noise) |
| 8b Laufzeittests | 243,289 | +5,104 |
| 8c Typ-Pins | 247,136 | +3,847 |

**Gate ≤ +8,000:** die Maschinerie allein (Stufen 1–7 + Marker) +6,980 — darunter. Mit Tests und
Pins +19,731 — darüber. Der Spec-Wortlaut schließt die Test-Stufe NICHT aus (Baustein A) →
Owner-Entscheidung. stress +2,577, browser Δ0, bench:editor-Instantiations uniform +2,600…2,900.
**bench:editor-Latenz nicht gemessen:** der Harness bricht ab, weil er die alte Hover-Form
(`NDArray<[8]>`) als Korrektheits-Beweis erwartet — muss beim Rollout angepasst werden.

**Hochrechnung:** 6,980 / 41 Signaturen ≈ 170,2 je berührte Signatur × 57 übrige + 6,980
≈ **16,684** (≈ 7,3 %). Annahmen offen: linear; gemeinsame Maschinerie schon bezahlt (eher
niedriger); `WNDArray`/Backends qualitativ anders (eher höher); nur „durchfädeln + sperren",
keine volle Op-Semantik. Signaturen per LSP `documentSymbol` gezählt (98 statt der informellen 67
— andere Zählregel; TS 7 exponiert im npm-Paket keine klassische Compiler-API).

**Hover (echter LSP, von A und B bestätigt):** `NDArray<[2, 3], "float64">`; float32 ⊕ int32 →
`NDArray<[3], "float64">`; int32-`sum` → `NDArray<[], "float64">`; `gt` → `NDArray<[3], "bool">`;
Union-Operand → `NDArray<[3], DType>` — alle aufgelöst, kein Alias sichtbar.

## Was hält

- **Union-Gate** (Baustein-0-Blocker behoben): `any`, `never`, `DType`, `"bool" | "int32"`,
  `astype`/`fromArray` mit breitem dtype, `sum`/`gt` mit breitem Operanden — überall ehrliche,
  breite Aussage, nie falsch-eng (Baustein B, 8 Pins).
- **Laufzeit:** float32 bit-identisch zu unabhängiger `Float32Array`-Referenz; int32-Wrap gegen
  `BigInt.asIntN` über 2,000 Zufallspaare; `astype`-Ränder (−0, ±0,5, 2^31, −2^31−1, NaN, ±Inf);
  bool-Summe über 10 Mio. Elemente; `gt` mit NaN — alles korrekt.
- **Bestehender float64-Code:** alle 1,591 Core- und 6,155 Resident-Tests unverändert grün, kein
  bestehender Test geändert; Paket-Smokes grün; `WNDArray`/WASM unberührt; Freeze-Hash gleich.
- **Varianz in `D`:** der Marker ändert kein Verhalten (6 Richtungen identisch), macht die
  Invarianz aber bewusst statt zufällig.

## Was die Spec so nicht einlösen kann

1. **Eigene Fehlermeldung bei Skalar-Fehlbenutzung (O2 a).** `boolArr.add(1)` und `i32.add(2.5)`
   zeigen nur die generische TS2769-Meldung der letzten Überladung. Baustein B hat drei
   Alternativen gebaut: eine Ein-Signatur-Variante zeigt die eigenen Meldungen (+1,215), öffnet
   aber ein neues M2-Loch (Union zweier Instanzen mit gültigem und ungültigem Shape wird mit
   falsch-engem Typ akzeptiert). Eine `IsUnion`-gesicherte Variante ist ungetestet.
2. **Niladische Ops sperren** (`sqrt`, `norm`, `transpose`, `flatten`, 0-arg `mean`/`argmax`): kein
   Argument, an dem ein Guard sitzen kann; zur Laufzeit wird geworfen, der Typ behauptet aber
   `NDArray<S, "float64">`. Ein Marken-Rückgabetyp meldet sich erst bei späterer Nutzung, nicht
   am Aufruf. Beim vollständigen Rollout entfällt das Problem (dann unterstützt jede Op jeden dtype).
3. **dtype-generische Funktionen** (`function f<D extends DType>(x: NDArray<S, D>) { x.add(1) }`)
   kompilieren nicht, auch wenn die Schranke bool ausschließt — Über-Ablehnung durch
   aufgeschobene bedingte Typen (Baustein B, F3). Dasselbe gilt heute schon für generische Shapes
   über `Broadcast`; dtype verdoppelt die Fläche.

## Defekte, vor dem Rollout zu beheben

- **`AnyNDArray` blieb `NDArray<any>`** und bedeutet damit „beliebige Shape, nur float64" — D1
  verlangt `NDArray<any, any>` (Baustein B, F1; der Fix berührt genau einen Pin).
- **Testlücke:** keine Probe mit zwei VERSCHIEDENEN Nicht-float64-dtypes an einer gesperrten
  Zwei-Operanden-Op — die Prüfreihenfolge in `DTypeLockPair` ist per Mutant unbewacht (F4).
- Covenant-Buchhaltung: neue kernel-lose Referenzfunktionen in FOLLOWUPS führen (M1 v5); die
  Skalar-Maskierung im Quelltext (nicht nur im Test) als benannte M3-Ausnahme dokumentieren.

## Post-Verification-Addendum (2026-09-25)

- **Baustein 0:** 1 Blocker (Union-Gate), 1 Major (`this`-Sperre) — vor dem Bau eingearbeitet.
- **Baustein A:** alle Zahlen bestätigt; eigener Mutant (Promotionstabelle) doppelt gefangen;
  Befunde: Gate-Lesart (Owner), Hausregel-Abweichung im Klassenkörper (inzwischen per Owner-
  Entscheidung 2026-09-25 als Fall (b) der angepassten Hausregel geregelt — kein bestehender Test
  geändert, erfüllt), Rechenfehler in der Hochrechnung (16,681 → 16,684).
- **Baustein B:** Design „tragfähig mit Nachbesserungen"; Befunde s. oben; 4 von 5 Mutanten
  gefangen (der überlebende = F4).
- **Baustein C:** kein akuter Verstoß (Branch nicht gemergt). Beim Rollout betroffen: M3 (Hover →
  v8-Entwurf D10), M3-Meldungsparität (Maskierung als benannte Ausnahme), M1-Tracking, M2-Anker
  (D7a). Lücken in den Entwürfen: D10 und D7a decken die Maskierung, die niladischen Ops und den
  `stack`-Fall nicht ab.

## Owner-Entscheidungen nach der Verify-Runde (2026-09-25)

1. **Gate:** das Design wird an der Maschinerie gemessen (+6,980 ≤ +8,000 — erfüllt); Test- und
   Pin-Kosten bekommen je Umsetzungs-Scheibe eine eigene, vorregistrierte Grenze, die sie einschließt.
2. **Skalar-Maskierung:** ein zeitlich begrenzter Lösungsversuch (`IsUnion`-gesicherte
   Ein-Signatur-Variante, von Baustein B als ungetestet benannt); scheitert er, wird die generische
   TS2769-Meldung als benannte, im Quelltext dokumentierte M3-Ausnahme übernommen (Präzedenz W4/W5).
3. **dtype-generische Funktionen:** als bekannte Grenze dokumentiert (README-Hinweis beim Release),
   wieder aufgreifen erst auf Nachfrage von Nutzern.

## Lösungsversuch Skalar-Maskierung (2026-09-25): NO-GO

Zeitlich begrenzter Versuch nach Owner-Entscheidung 2. Kandidat: `add` als EINE Signatur mit
`IsUnion<Other>`-Gate vor der Unterscheidung Skalar/Array (Code:
`docs/assets/dtype/add-single-signature-candidate.diff`, nur `ndarray.ts`, gegen 059d852).
**Korrektheit bestanden:** eigene Meldungen am Argument, wortgleich zur Laufzeit, für
`boolArr.add(1)`, `i32.add(2.5)`, `i32.add(boolArr)` und Shape-Fehler; alle Union-Argumente
(Instanz-Unions, `number | NDArray`, `2 | 2.5`) abgelehnt statt falsch-eng; alle guten Pfade mit
unverändertem Hover; alle bestehenden Tests unverändert grün (core 1,624, resident 6,155+2).
**Kosten durchgefallen:** check:diag 247,136 → 255,631 (**+8,495**, nach einer Cache-Teilung, die nur
646 zurückholte; naiv +9,141) — mehr als die gesamte bisherige dtype-Maschinerie. Ursache: `add` ist
die heißeste Aufrufstelle des Korpus (u. a. ~30-fach verkettet in `limits.test-d.ts`); die
Ein-Signatur-Form wertet an JEDER Aufrufstelle echte bedingte Typen aus, wo die Überladungsauswahl
praktisch kostenlos ist. stress Δ−7.
**Folge (Owner-Rückfall, vorab entschieden):** die generische TS2769-Meldung bei Skalar-Fehlbenutzung
wird eine **benannte M3-Ausnahme**, im Quelltext an den Überladungen dokumentiert (Präzedenz W4/W5),
und in den M3-v8-Entwurf aufgenommen. Die Laufzeit wirft weiterhin die eigene Meldung.
