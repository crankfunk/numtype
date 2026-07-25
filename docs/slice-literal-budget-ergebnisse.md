# `slice`-Literal-Budget — Ergebnisse der Gegenprüfung

**Datum:** 2026-07-25 · **Basis:** `2d38f67` · **Anlass:** FOLLOWUPS-Eintrag „`slice`-Literal-
Kosten als Budget-Hebel", der aus **einer** Messung in WASM-Parität S5 stammte. Der Owner hat
verlangt, die Behauptung an einem zweiten Korpus gegenzuprüfen, **bevor** daraus eine Hausregel
wird. Diese Prüfung hat sie in zwei Punkten widerlegt.

**Ehrlichkeitsregel dieses Dokuments:** jede Zahl unten stammt aus einem Lauf in einem frischen
`git worktree`, mit geprüftem Exit-Code und geprüfter Fehlerzahl. Wo nichts gemessen wurde,
steht das da. Der Haupt-Working-Tree wurde für keine Messung angefasst.

## Ergebnis in drei Sätzen

Die S5-Zahl reproduziert exakt und ist damit ein viertes Mal bestätigt — aber sie ist ein
**Ausreißer**, kein Merkmal ihrer Blockklasse: der strukturell identische S4/argmax-Block
kostet nur ein Fünftel. Von den +3.926 des S5-Blocks sind nur **57 % slice-getrieben**, nicht
der volle Betrag, wie die Notiz nahelegte. Und der eigentliche neue Befund ist ein
**methodischer**: diese Kosten sind stark super-additiv, weshalb die in der Notiz implizit
verwendete Pro-Aufrufstelle-Rechnung gar nicht definiert ist.

## Messaufbau

Baseline **237.379 @ 140**, im frischen Worktree gemessen und **zweimal reproduziert**, Exit 0,
null Diagnosen — identisch zum gepinnten Wert. Dateiset bleibt in **allen** Varianten bei 140
(es wird keine Datei angelegt oder gelöscht), daher ist **kein Messpunkt von Order-Noise
betroffen**. Kontrolle „nur der `SliceSpecInput`-Typimport, keine Aufrufstelle geändert":
**Δ0** — der Import selbst kostet nichts.

Der Eingriff ist überall derselbe: die Spec-Argumente werden so übergeben, dass `Specs` als
Array statt als Tupel inferiert wird. Damit greift in `slice.ts` der `IsDynamicLength`-Zweig,
und `SliceSpecsGuard` wie `SliceShape` nehmen beide den No-Claim-Pfad.

## Befund 1 — die S5-Zahl hält, ihre Verallgemeinerung nicht

| Messung | inst | Δ |
|---|---|---|
| **Ausbau** S5/topk-Vier-View-Block (`resident.test.ts` 1948–2057) | 233.453 | **−3.926** |
| **Ausbau** S4/argmax-View-Block (`resident.test.ts` 1332–1433) | 236.645 | **−734** |

233.453 ist auf die Ziffer der Wert, den in S5 drei Instanzen unabhängig gemessen hatten. Der
argmax-Block ist der strukturelle Zwilling — ~100 Zeilen, dieselben vier View-Klassen,
dieselben drei Slice-Formen (`{step}`, `{start}`, `null,{start},null`) — und kostet **Faktor
5,3 weniger**. Aus einer Einzelbeobachtung war eine Aussage über eine Codeklasse geworden;
die hält nicht.

**Warum der Unterschied so groß ist, ist ungeklärt.** Gemessen ist nur, dass die zwei
rank-1-Stellen des S5-Blocks (Zeilen 1966/1985) zusammen 1.881 tragen und damit die teuersten
sind. Ein Mechanismus dafür ist **nicht bewiesen** und wird hier nicht behauptet.

## Befund 2 — Slices sind auch in S5 nur gut die Hälfte

| Messung | inst | Δ |
|---|---|---|
| S5-Block: nur die 5 Slice-Specs gewidet (Block bleibt) | 235.147 | −2.232 |
| S5-Block komplett ausgebaut | 233.453 | −3.926 |

**2.232 von 3.926 = 57 %.** Die restlichen 1.694 sind gewöhnliche Testinhalts-Kosten
(`topk`-Aufrufe, `transpose`, Assertions). Die Formulierung „+3.926 … getrieben von
literal-argumentigen `.slice()`-Aufrufstellen" las sich, als sei der volle Betrag
slice-getrieben.

**Folge für die abgeleitete Marge:** Die Notiz sagte, mit der Hausregel hätte S5s Gate-Marge
statt 449 bei ≈4.400 gelegen. Das verwechselt Ausbau des Blocks mit Widen seiner Specs.
Richtig: 7.551 − 2.232 = 5.319 gegen ≤8.000, Marge **≈2.681**. Immer noch eine
Versechsfachung — aber 39 % weniger als notiert.

## Befund 3 — die Kosten sind super-additiv; Einzelsite-Attribution existiert nicht

Das ist der eigentlich neue Befund. Die drei strukturgleichen View-Blöcke, einzeln und
gemeinsam gemessen:

| Messung | inst | Δ |
|---|---|---|
| S2/mean-Block, 3 Specs | 236.830 | −549 |
| S3/item-Block, 3 Specs | 236.800 | −579 |
| S4/argmax-Block, 3 Specs | 236.830 | −549 |
| **dieselben 9 Specs gemeinsam** | 234.345 | **−3.034** |
| alle 15 Stellen in `resident.test.ts` | 230.996 | −6.383 |
| alle 32 Stellen über 7 Dateien | 225.599 | **−11.780** |

Die Summe der drei Einzelmessungen ist 1.677, gemeinsam sind es **3.034** — Faktor 1,8. Über
alle 32 Stellen liegt das Ergebnis rund 3.200 über der Teilsummen-Erwartung.

Noch deutlicher wird es an den Einzelstellen des S5-Blocks:

| einzeln gewidete Stelle(n) | inst | Δ |
|---|---|---|
| 1966 + 1985 (rank-1, einfache Range-Specs) | 235.498 | −1.881 |
| 2034 + 2036 (verkettete Slices) | 237.197 | −182 |
| 2008 (Zeile einer transponierten Matrix) | 237.290 | −89 |
| **2087 (Special-Value-Block)** | 237.586 | **+207** |

Eine einzelne Stelle zu widen kann den Zähler **erhöhen** (2087, zweimal reproduziert). Das
ist derselbe Fresh-vs-Cached-Partitionsmechanismus, den die Mess-Regeln bisher nur für
Order-Noise beim Hinzufügen von Dateien beschrieben — hier tritt er **innerhalb eines fixen
File-Sets** auf, ohne dass eine Datei dazukommt.

**Das erklärt einen früheren Befund nachträglich.** `docs/wasm-parity-topk-ergebnisse.md`
dokumentierte unter „Eine Zahl bleibt ungeklärt", dass die Literal-Mehrkosten *einer*
Aufrufstelle dreimal gemessen wurden und drei verschiedene Werte ergaben (≈122 / ≈149 / ≈95).
Das war kein Messfehler und kein Sondenartefakt: **eine solche Zahl existiert nicht.**

## Befund 4 — die Prämisse der Hausregel hält

Zwei unabhängige Belege, beide mechanisch geprüft statt behauptet:

1. **Die sieben Zieldateien enthalten null Typ-Ebenen-Assertionen** (`Equal<`, `Expect<`,
   `expectType`, `assertType` — je 0 Treffer). Die literalen Specs kaufen dort nichts.
2. **`spike/tests/slice.test-d.ts:8-11` sagt es selbst:** `WNDArray` teile „the exact same
   type machinery … so pinning it once via `NDArray` covers both; `WNDArray.slice`'s own
   correctness is covered by the runtime differential suite instead". Die Typ-Ebenen-Coverage
   liegt also nachweislich woanders und geht nicht verloren.

Verhaltensneutralität, gemessen auf der vollen Umstellung: `pnpm check` (drei Legs) Exit 0 ·
`test:core` **1591 pass / 0 fail** · `test:resident` **6122 pass / 0 fail / 2 skipped** —
beide exakt die gepinnten Zahlen. `test:threaded` (139) ist **nicht gemessen** (braucht die
pinned nightly); zwei der 32 Stellen liegen dort und sind damit noch ungeprüft.

## Befund 5 — die Schreibweise wurde gemessen, nicht gewählt

| Form | inst | Δ |
|---|---|---|
| A · lokale `const specs: readonly SliceSpecInput[]`, 2 Zeilen/Stelle | 225.628 | −11.751 |
| B · Inline-Cast `.slice(...([…] as readonly SliceSpecInput[]))` | 225.684 | −11.695 |
| **C · geteilter Helfer `.slice(...wideSpecs(…))`** | **225.599** | **−11.780** |

Der Abstand ist klein (85 über 32 Stellen), aber C gewinnt auf allen drei Achsen gleichzeitig:
billigste Variante, kürzeste Aufrufstelle, und die einzige, die das *Warum* am Aufrufort
benennt. Details in `docs/slice-literal-budget-spec.md`, D2.

## Was der Umbau NICHT anfasst, und was das kostet

`spike/tests-runtime/slice.test.ts` bleibt außen vor: dort sind die literalen Specs die
Aussage (Semantik-Tabelle mit danebenstehendem Erwartungswert; die vorhandenen
`5 as number`-Widenings markieren gerade die bewusste Typ-/Laufzeit-Grenze). Die Datei kostet
insgesamt **10.683** — gemessen per empty-then-fill (Inhalt durch `export {}` ersetzt, damit
das Dateiset bei 140 bleibt und kein Order-Noise entsteht). Das ist eine **Obergrenze** für
das, was dort liegen bleibt; wie viel davon literal-getrieben ist, wurde **nicht** gemessen.

## Offengelegte Grenzen

- Der Faktor 5,3 zwischen S5- und S4-Block ist **beschrieben, nicht erklärt**.
- Die 10.683 von `slice.test.ts` sind eine Obergrenze der Gesamtkosten der Datei, **keine**
  Messung ihres literalen Anteils.
- `test:threaded` ist auf der Umstellung noch **nicht gelaufen**.
- Alle Messungen stammen von **einer** Instanz (dieser Session) auf **einem** Host. Die
  Verify-Runde der Umsetzungs-Scheibe muss sie unabhängig nachstellen; die Baseline und die
  Zahl 233.453 sind bereits vierfach bzw. durch drei frühere Instanzen gedeckt, die
  **übrigen Zahlen dieses Dokuments nicht**.

## Was daraus folgt

Die Hausregel ist es wert — aber aus einem anderen Grund als notiert. Für eine künftige
Arbeitsregel-12-Scheibe liegt der marginale Gewinn bei rund 550–2.200, nicht bei „52 % der
Scheibe". Der große Hebel ist die **einmalige** Umstellung aller 32 Stellen: −11.780 = 4,96 %
des Root-Korpus, deutlich mehr als die Summe der Einzelscheiben-Ersparnisse. Genau deshalb ist
sie eine eigene Scheibe wert — die bindende Spec liegt als
`docs/slice-literal-budget-spec.md` v1 vor.
