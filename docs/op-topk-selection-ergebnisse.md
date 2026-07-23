# topk-Selektion: Phase-1-Messergebnisse

Spec: [docs/op-topk-selection-spec.md](op-topk-selection-spec.md) v6 · Datum: 2026-07-22 ·
Status: **Messung abgeschlossen, Umsetzung steht aus** (eigene Phase mit eigener Verify-Runde).

**Ehrlichkeitsregel dieses Dokuments:** Jede Zahl stammt aus einem Kommando, das gelaufen ist,
mit geprüftem Exit-Code. Was nicht verifiziert ist, steht als nicht verifiziert da. Die
Prozess-Fehler dieser Scheibe stehen mit drin, nicht nur ihre Ergebnisse.

## Das Verdikt

**PURE HEAP** — unbedingte Ersetzung, kein Hybrid, keine Schwelle.

Mechanisch aus D6 berechnet, nicht von Hand gelesen. Beide Messläufe kamen unabhängig zum
selben Verdikt; der verdikt-tragende pessimistische Zusammenzug bestätigt es. Im gesamten
92-Zellen-Raster gibt es **keine einzige duale Verletzung**, 57 Zellen sind duale Gewinne, und
alle 27 Schwellen-Kandidaten sind sicher — also `t* = 1,0`, die sichere Zone umfasst jede
mögliche Eingabe. Genau dafür sieht D6 den verzweigungsfreien Ausgang vor.

| Klasse | Zellen | Bedeutung |
|---|---|---|
| Gewinn (dual) | 57 | relativ mindestens 15 % schneller UND absolut mindestens 10 µs |
| nur relativ Gewinn | 9 | relativ schneller, absolut zu klein zum Zählen |
| nur absolut Gewinn | 9 | absolut relevant, relativ unter 15 % |
| neutral | 9 | keins von beidem |
| nur relativ Verletzung | 1 | über 15 % langsamer, aber absolut unter 10 µs |
| nur absolut Verletzung | 7 | absolut relevant langsamer, relativ unter 15 % |
| **Verletzung (dual)** | **0** | kommt im gesamten Raster nicht vor |

## Die Größenordnung

Der Gewinn ist dort am größten, wo die Op tatsächlich benutzt wird — wenige Größte aus vielen
Elementen:

| n | k | heute (Sortierung) | Heap | Verhältnis |
|---|---|---|---|---|
| 1.000.000 | 1 | 280,2 ms | 3,78 ms | 0,014 (**Faktor 74**) |
| 1.000.000 | 50 | 279,0 ms | 3,79 ms | 0,014 |
| 100.000 | 1 | 23,1 ms | 0,38 ms | 0,016 |
| 10.000 | 10 | 1,78 ms | 40,2 µs | 0,023 |
| 1.000 | 10 | 128,1 µs | 5,5 µs | 0,046 |

Das entspricht der Erwartung: Der alte Weg sortiert eine Million Elemente, um zehn zu
liefern; der neue führt eine Zehnerliste mit.

## Die ehrliche Kehrseite

**Sieben Zellen sind absolut langsamer**, und das Verdikt nimmt sie bewusst in Kauf, weil sie
relativ unter der 15-%-Toleranz bleiben:

| n | k | k/n | Verhältnis | absolute Verschlechterung |
|---|---|---|---|---|
| 10.000 | 10.000 | 1,00 | 1,028 | +49,7 µs |
| 100.000 | 85.000 | 0,85 | 1,004 | +99,5 µs |
| 100.000 | 95.000 | 0,95 | 1,021 | +470 µs |
| 100.000 | 100.000 | 1,00 | 1,052 | +1,21 ms |
| 1.000.000 | 900.000 | 0,90 | 1,016 | +4,61 ms |
| 1.000.000 | 950.000 | 0,95 | 1,044 | +12,28 ms |
| 1.000.000 | 1.000.000 | 1,00 | 1,050 | **+13,95 ms** |

Einsatz ab `k/n = 0,85`, Maximum bei `n = 1.000.000, k = n` — dem entartetsten Fall, in dem
der Heap nie etwas verwirft und trotzdem am Ende alles sortiert. Relativ sind das 5,0 %, und
genau deshalb zählt es nach dem dualen Kriterium (Owner-Entscheidung 2026-07-22) nicht als
Verletzung. Es ist trotzdem ein realer, gemessener Preis: **Wer `topk(n)` aufruft — heute der
einzige Weg, in dieser Bibliothek überhaupt zu sortieren —, zahlt ihn.** Das ist der Grund,
warum der `sort`/`argsort`-Kandidat in FOLLOWUPS nach dieser Messung einen (schwachen) Beleg
hat, den er vorher nicht hatte.

## Der wichtigste Methodik-Befund: die Vorab-Sondage war um mehr als eine Größenordnung daneben

Die informelle Sondage, die diese ganze Scheibe ausgelöst und vier Runden Regelbau begründet
hat, wurde **live auf derselben Maschine erneut ausgeführt** — nicht aus der Erinnerung
zitiert. Sie reproduziert ihre eigenen Zahlen exakt:

| Fall | Sondage | disziplinierte Messung | Abweichung |
|---|---|---|---|
| `n = 1e6, k = n` | 0,60 (Heap ca. **67 % langsamer**) | 1,050 (Heap **5,0 % langsamer**) | Faktor 13 |
| `n = 1e6, k = n/2` | ca. 6 % langsamer | **24,3 % schneller** | Vorzeichenwechsel |

Die qualitative Form überlebt (das Verhältnis steigt monoton, je näher `k` an `n` rückt), die
Zahlen nicht. Die Ursachen wurden im Quelltext der Sondage belegt, nicht vermutet:

1. **Aufwärmtiefe.** Die Sondage wärmt mit genau zwei Aufrufen und misst dann fünf
   Wiederholungen. Die Messung wärmt adaptiv (30 ms oder 64 Aufrufe) und durchläuft die
   Größen aufsteigend im selben Prozess — die heißen inneren Funktionen des Heaps haben bei
   der teuren letzten Zelle bereits Tausende Aufrufe hinter sich.
2. **Vorherige Aufruf-Polymorphie.** Die Sondage fährt unmittelbar davor über 20.000
   Korrektheits-Fuzz-Fälle mit stark wechselnden Größen im selben Prozess — genau die
   JIT-Kontamination, vor der die KB-Notiz zu JS-Benchmarks warnt.
3. **Datenrepräsentation.** Die Sondage nutzt gewöhnliche JS-Arrays für den Heap, die Messung
   typisierte Arrays.

Welcher der drei Effekte dominiert, wurde NICHT isoliert — das bräuchte ein eigenes Experiment.
Alle drei sind konkret vorhanden und je für sich ausreichend, um eine Verzerrung in diese
Richtung zu erklären.

**Das ist derselbe Mechanismus wie in Kern 06**, wo eine saubere Messung einen früheren Befund
als Artefakt der Messeinheit entlarvte. Die Lehre ist unbequem: Eine schnelle Sondage kann eine
umfangreiche, sorgfältige Folgearbeit auf eine Zahl gründen, die einer disziplinierten Messung
nicht standhält — und je aufwendiger die Folgearbeit, desto weniger fällt es auf, weil niemand
mehr an die Ausgangszahl zurückdenkt.

## Reihenfolge-Sensitivität

Vier von 92 Zellen in Lauf 1 (null in Lauf 2) zeigten relative Unterschiede zwischen den beiden
Messreihenfolgen von bis zu 111 %. Alle vier liegen bei `n ≤ 1.000` mit Gesamtzeiten unter
10 µs; die absolute Schwankung beträgt Bruchteile einer Mikrosekunde. **Keine davon berührt das
Verdikt** — alle liegen tief in der bereits sicheren, bereits gewinnenden Zone, unter jeder
Lesart weit von der Entscheidungsgrenze entfernt. Das Zwei-Reihenfolgen-Protokoll hat damit
genau geleistet, wofür es gedacht war: die Aufwärm-Abhängigkeit sichtbar machen, statt sie
unbemerkt in eine Zahl einfließen zu lassen.

## Bit-Identität

**184 Prüfungen** (92 Zellen mal zwei Läufe), jede vor jeder Zeitmessung, jede gegen die echte,
unveränderte `topkRuntime` als Referenz. Verglichen wurden Werte UND Indizes, elementweise mit
`Object.is` (unterscheidet `+0` von `-0`) plus Längengleichheit. **Null Abweichungen.**

Das ist die empirische Schicht über dem konstruktiven Argument: Weil der bestehende Vergleicher
Gleichstände nach Index auflöst, ist „die k größten in Vergleicher-Reihenfolge" eindeutig
bestimmt, und jedes korrekte Auswahlverfahren muss dasselbe liefern. Der Beweis gilt dem
Entwurf, der Test dem geschriebenen Code.

**Offengelegte Grenze:** Die Phase-1-Messdaten enthalten bewusst keine Spezialwerte (kein NaN,
keine Unendlichkeiten, keine Duplikat-Häufungen). Phase 1 musste nur zeigen, dass der Kandidat
auf den GEMESSENEN Daten identisch liefert. Der Nachweis über NaN-Bitmuster, Gleichstände und
Randfälle gehört in den Differentialtest der Umsetzungsphase.

## Pins

**Neuer Root-Pin: 206.801 @ 140 Dateien** (vorher 199.877 @ 139). Zerlegt nach der
Mess-Hausregel, in einem frischen Worktree gemessen und im Haupt-Baum exakt reproduziert:

| Stufe | Instantiations | Dateien | Delta |
|---|---|---|---|
| Basis (ohne Bench-Datei) | 199.877 | 139 | — |
| leerer `export {}`-Platzhalter | 206.488 | 140 | **+6.611** (reines Reihenfolge-Rauschen) |
| befüllt (echtes Bench-Skript) | 206.801 | 140 | **+313** (echte Typkosten) |

`check:diag:stress` 106.398 @ 82 und `check:diag:browser` 2.142 @ 75 blieben exakt unverändert.

**Befund über die Messinfrastruktur, der über diese Scheibe hinausgeht:** Das in CLAUDE.md
dokumentierte Order-Noise-Band von „bis zu ca. ±2.000" ist damit **empirisch widerlegt** —
eine einzige LEERE Datei verschob den Zähler um +6.611, zweifach reproduziert. Das Band wird
entsprechend korrigiert. Konsequenz für die Spec-Vorgabe „höchstens +2.000" (D12): Sie reißt
formal, obwohl der inhaltliche Anteil mit +313 winzig ist. **Owner-Entscheidung 2026-07-22:**
Das Gate gilt für den DEKOMPONIERTEN Anteil, nicht für die Gesamtverschiebung — genau dafür
existiert das empty-then-fill-Protokoll. Die Spec-Formulierung war unpräzise und wird
nachgeschärft.

## Mess-Randbedingungen

Zwei vollständige Läufe über 92 Zellen in beiden Reihenfolgen, ein Prozess, deterministisch
erzeugte Daten je Lauf (Seeds `0x544f504b00000001` / `...002`). Maschine: Apple M3, macOS,
Node v24.16.0. **Maschinenlast vor dem Start 3,81 / 4,15 / 6,48**, nach Abschluss 2,76 / 3,27 /
4,86 — eine normal genutzte Arbeitsmaschine, kein dedizierter Benchmark-Host. **Wall-Clock:
163,3 s + 162,7 s = 326,0 s.** Das vorregistrierte Budget von 30 Minuten wurde mit ca. 18 %
Auslastung eingehalten.

## Gate-Block

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` | sauber | 0 |
| `pnpm check:diag` | 206.801 @ 140 | 0 |
| `pnpm check:diag:stress` | 106.398 @ 82 (unverändert) | 0 |
| `pnpm check:diag:browser` | 2.142 @ 75 (unverändert) | 0 |
| `pnpm test:core` | 1588 / 1588 | 0 |
| `pnpm bench:editor` | acht Pins exakt, Hard-Gate PASS | 0 |
| `graph-a-lama query lint` | 0 Befunde | — |

`spike/src/runtime.ts` ist byte-identisch unberührt — die Umsetzung ist bewusst nicht Teil
dieser Phase.

## Prozess-Bilanz: vier Runden Regel-Reparatur, bevor eine Zahl existierte

Die Entscheidungsregel D6 wurde viermal gebrochen, bevor sie messen durfte — jedes Mal von
einem unabhängigen Fresh-Context-Verifier, jedes Mal an einer anderen Stelle. **Zwei der
gebrochenen Fassungen stammten vom Orchestrator selbst.**

| Runde | Gegen | Befunde |
|---|---|---|
| Baustein 0 | v2 | Zulässigkeitsprüfung ignorierte die Größen, an denen der Gewinn erwartet wird; eine Klausel widersprach sich im selben Satz; ein undefinierter Fall ließ die Regel abbrechen |
| gezieltes Gegenlesen | v3 | kein Verdikt in 6,5 % der Raster; zwei Verdikte für dieselben Zahlen in 43,6 %; „reiner Heap" trotz null gemessenem Gewinn |
| Simulator-Gegenbeweis | v4 | „hohler Hybrid" — Gewinn-Existenz und Schwellenbildung entkoppelt, in 29,4 % der Hybrid-Verdikte eine Netto-Regression |
| Frontier-Zweitmeinung | v5 | zwei Läufe mandatiert, keiner als verdikt-tragend benannt (39,6 % divergierende Lauf-Paare); rein relative Schwellen machten das Verdikt rausch-fragil |

Jeder einzelne dieser Fehler hätte nach der Messung ein mechanisch berechnetes, eindeutig
aussehendes Verdikt geliefert — also als „die Zahlen sagen es doch" durchgehen können. Die
Regel selbst hätte nie signalisiert, dass sie falsch ist.

**Die eigentliche Ausbeute dieser Scheibe ist deshalb nicht die Optimierung** (die heute
nachweislich niemand spürt), sondern die Erkenntnis, wie brüchig vorregistrierte
Entscheidungsregeln sind, wenn man sie nicht mechanisch gegen synthetische Ergebnisse
durchspielt, bevor echte Zahlen existieren. Eine Regel, die man nur liest, wirkt eindeutig; eine
Regel, die man als Skript gegen tausende konstruierte Raster laufen lässt, verrät ihre Lücken.

## Offengelegte Lücken und Prozess-Fehler

1. **Der Exit-Code des Messprozesses wurde nicht literal erfasst.** Der Lauf wurde als
   Hintergrundprozess gestartet und über das Verschwinden der Prozess-ID überwacht. Der
   saubere Logverlauf (keine Fehlerausgabe, vollständiger Durchlauf bis zur Verdikt-Zeile) ist
   starkes, aber nicht gleichwertiges Indiz. Vom Ausführenden selbst offengelegt.
2. **Ein Exit-Code-Fehler wurde unterwegs gefunden und korrigiert:** Die ersten Gate-Läufe
   waren durch `tail` gepiped, womit dessen Exit-Code gemessen wurde statt der des Kommandos.
   Selbst bemerkt, alle Gates neu erhoben — die Tabelle oben stammt aus dem korrigierten Lauf.
3. **Zwei Agenten beendeten ihren Zug auf einer Absicht** statt auf einem Ergebnis („ich warte,
   bis der Hintergrundlauf fertig ist"). Beide Male existierten die Befunde bereits und wären
   ohne Nachfassen unberichtet geblieben — dieselbe Klasse wie in der Scale-Probe.
4. **Zwei Stellen der Spec waren beim Umsetzen unterspezifiziert** und wurden offengelegt statt
   still entschieden: die Zusammenzug-Formel für das absolute Delta auf REIHENFOLGEN-Ebene (die
   Spec regelt sie nur auf Lauf-Ebene), und die Übertragung des Zwei-Reihenfolgen-Protokolls
   auf den Drei-Kandidaten-Vergleich in Stage B (nie ausgeführt, da kein Hybrid).
5. **Eine Behauptung in FOLLOWUPS wurde durch diese Messung widerlegt** und ist korrigiert: Der
   `sort`-Eintrag argumentierte, ein Hybrid löse den `k`-nahe-`n`-Fall automatisch. Es gibt
   keinen Hybrid.

## Was als Nächstes ansteht

Die Umsetzung (D8) ist eine eigene Phase mit eigener Verify-Runde: `topkRuntime` in-place durch
den Heap ersetzen (Abweichung von der Append-Konvention ist in der Spec vorab genehmigt und
begründet), die heutige Sortierung wörtlich als Orakel in den Test verschieben, und der
Differentialtest über NaN-Bitmuster, `+0`/`-0`, Gleichstände und Randfälle (`k = 0`, `k = n`,
`n = 1`). Erst dort bindet die Bit-Identität als Testpflicht statt als Messvoraussetzung.

---

# Phase-2-Umsetzung (PURE HEAP / D8) — ERLEDIGT 2026-07-23

Status: **Umsetzung abgeschlossen**, Verify-Runde A+B+C durchgeführt (alle drei grün). Jede Zahl
unten stammt aus einem Kommando mit geprüftem Exit-Code.

## Was umgesetzt wurde

`topkRuntime` (spike/src/runtime.ts) wurde **an Ort und Stelle** von der Vollsortierung
(O(n log n) über alle n Indizes) auf einen **größenbeschränkten Max-Heap** (O(n log k))
umgestellt — der mechanisch aus D6 berechnete Ausgang PURE HEAP (t* = 1,0). Die Konstruktion ist
der in Phase 1 über 184 Zellen bit-identisch bewiesene Heap-Kandidat B aus
spike/bench-core/topk-selection.ts, wortgetreu portiert (zwei parallele typisierte Arrays für
Werte + Quellindizes; Wurzel = schlechtestes gehaltenes Element; Ersetzen-bei-Verbesserung +
Resift; finale O(k log k)-Sortierung mit DEMSELBEN Comparator-Ausdruck; `values[i]` aus dem
ursprünglichen `data`-Array re-gelesen, nie aus einem Heap-Wert — D2 Punkt 4).

- `topkCompareValues` bleibt **byte-unverändert** und wird wiederverwendet.
- Die In-Place-Ersetzung bricht bewusst runtime.ts' Append-Konvention — **in der Spec vorab
  genehmigt** (Absatz „Vorab-Genehmigung der In-Place-Abweichung" vor D8). Weder COVENANT M4
  (Anker nur die drei Rust-Dateien) noch die Frozen-baseline-Disziplin betreffen die
  freistehende Funktion `topkRuntime`.
- `NDArray.topk`s Signatur, Aufrufer, Validierungsverhalten und jede beobachtbare Ausgabe bleiben
  unverändert.

## Test-Migration, Differentialtest, Mutant (D11)

Die heutige Full-Sort-Fassung lebt **wörtlich** als test-lokales Orakel (`topkOracleFullSort` +
`topkOracleCompareValues`) in spike/tests-runtime/argmax-topk.test.ts weiter — bewusst KEINE
unabhängig geschriebene Referenz (das ist `bruteTopk`, unverändert daneben). Drei neue Tests:

1. **Feste Grenzfälle:** n=0/k=0, n=1/k=0, n=1/k=1, all-NaN-Vektor (diverse k), all-equal-Vektor,
   +0/-0-Gleichstand-Fixture, ±Infinity gemischt, k=n-1, k=n.
2. **Randomisierter Differentialtest:** 300 Fälle, n über 0..40, k gleichverteilt über 0..n,
   `genDataSpecial` (NaN/±0/±Inf/Subnormale); Nicht-Vakuität (trifft NaN- und Gleichstands-Pfade)
   assertet. Pro Fall indices deepStrictEqual, values bit-identisch (Object.is).
3. **Exakte nicht-kanonische NaN-Payload** (`0x7ff800000000beef`) direkt über DataView in den
   Backing-Buffer geschrieben (nie Array-Literal — der in D11 mandatierte JIT-Fallstrick),
   byte-identisch durch Heap UND Orakel via `bitsAt` geprüft.

**Pflicht-Mutant (T5):** Eviction-Vergleich `cmp(v,i,rv,ri) < 0` → `> 0` in der Heap-Schleife →
`node --test spike/tests-runtime/argmax-topk.test.ts` **exit 1, 4/33 rot** (darunter BEIDE neuen
Differentialtests). Revert per Backup-Kopie (cp nach /tmp, zurück), diff-Beweis byte-identisch
(SHA `d125527…`), erneuter Lauf **33/33 grün, exit 0**. Kein `git checkout` auf uncommittete
Arbeit (harte Arbeitsregel 1).

## Gate-Block

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Dreier-Verbund) | sauber | 0 |
| `pnpm check:diag` | **206.854 @ 140** (Δ+53 gg. 206.801 @ 140; ≤ +3.000-Gate) | 0 |
| `pnpm check:diag:stress` | 106.398 @ 82 (**Δ0**) | 0 |
| `pnpm check:diag:browser` | 2.142 @ 75 (**Δ0**) | 0 |
| `pnpm test:core` | **1591 / 1591** (1588 + 3 neu) | 0 |
| `pnpm test:resident` | 4278 pass, 0 fail (2 skipped) | 0 |
| `cargo test` | 161 passed | 0 |
| `pnpm check:freeze` | Hash byte-identisch `0b9df4f1…` (kein Rust berührt) | 0 |
| `pnpm bench:editor` | 8 Pins Δ0, Hard-Gate PASS | 0 |
| `pnpm test:example` | 8 Queries + Demo grün | 0 |
| `graph-a-lama query lint` | 0 Befunde (S1) | — |

## Pins (Δ-Zerlegung)

Baseline im frischen Worktree @ d9dd5a0 (alte Full-Sort, `heapVal`-Count 0) reproduziert:
**206.801 @ 140**. Nach der Scheibe: **206.854 @ 140**. Da KEINE Datei hinzukommt (Dateiset
bleibt 140), gibt es **kein Order-Noise** — der **Δ+53 ist reine Typkosten** des
runtime.ts-Körpertauschs plus der Testdatei-Erweiterung. Weit unter dem D12-Gate von +3.000.
stress/browser exakt Δ0, bench:editor alle 8 Pins Δ0 — die öffentliche NDArray-Fläche ist
unberührt. Neuer Root-Pin: **206.854 @ 140**.

## Post-Verification-Addendum

Verify-Runde Stufe 3, drei Fresh-Context-Verifier parallel (A/B je im eigenen `isolation:
worktree` mit appliziertem Patch, um Mutanten-Kollision im geteilten Haupt-Baum zu vermeiden; C
read-only). **Alle drei grün, kein Blocker/Major/Minor.**

- **Baustein A (Spec + alle Gates frisch + eigener Mutant) — CONFIRMED.** D8/D2/D11/T6 einzeln
  gegen den Diff konform (hohe Konfidenz). Baseline 206.801 → 206.854 (Δ+53) selbst reproduziert;
  alle Gates frisch grün (stress/browser Δ0, bench:editor 8 Pins Δ0, test:core 1591, test:resident
  4278, cargo 161, check:freeze byte-identisch, graph frisch gebaut + lint 0). Eigener Mutant —
  Off-by-one `size < k` → `size <= k`, verschieden vom bereits gefangenen — von BEIDEN neuen
  Differentialtests plus zwei Bestandstests gefangen (Nicht-Vakuität aus zwei Testgenerationen),
  byte-identisch revertiert ohne `git checkout`. Zwei Nits: Doc-Platzierung (D13) war nicht Teil
  des Diffs (bewusst, hiermit erledigt); ein harmloser System-Reminder aus dem eigenen Revert.
- **Baustein B (adversarial) — keine Blocker/Major/Minor.** Unabhängig geschriebene Full-Sort-
  Referenz (andere Code-Form: decorate-sort-undecorate, eigener PRNG), 3517 Fälle inkl. vier NaNs
  mit vier distinkten nicht-kanonischen Payloads gleichzeitig, dichte Duplikat-Cluster,
  ausschließlich-Spezialwert-Vektoren, geslicte NDArray-Empfänger (9/9) → **0 Fehler**. 6/6
  angeforderte Mutantenklassen (siftUp/siftDown, Parent-Formel, Kinder-Off-by-one,
  Vergleichsrichtung, Sortierrichtung, Kapazitätsgrenze) vom committeten Suite gefangen. Zwei
  zusätzliche Mutanten **beweisbar semantisch inert** (KEINE Testlücken): (3a) siftDowns
  Kinder-Labels vertauscht = order-unabhängiger Max-von-3-Scan; **(7) Root-Replace `< 0` → `<= 0`
  inert, weil der Scan-Index im Ersetzungs-Zweig immer strikt größer ist als jeder gehaltene Index
  — `cmp` wird an dieser Stelle nie exakt 0.** Letzteres ist eine schöne unabhängige Bestätigung
  der Korrektheit; der Tiebreak `|| (aIdx - bIdx)` ist dort toter-aber-harmloser Code und bleibt
  bewusst identisch zur bit-identisch bewiesenen Bench-Konstruktion (nicht wegoptimiert).
  Messbedingungen selbst reproduziert (Δ+53, stress/browser Δ0, keine Korpus-Kontamination),
  Typ-Fläche unberührt.
- **Baustein C (covenant-verify) — keine Befunde.** S1 (lint 0 Errors, Regel `covenant-s1` ↔
  Invariante S1 1:1, Diff fügt keinen runtime→Test/Demo-Import hinzu), M1 (topk hat weiterhin
  keinen WASM-Kernel — `grep -rn topk crates/core/src/` null Treffer —, Bit-Identitäts-PFLICHT
  bindet also laut v5-Präzisierung nicht; Paritätslücke in FOLLOWUPS getrackt), M2/M3/M4/M5/Z1/Z2
  eigenständig gegen den Diff verifiziert — alle unberührt. Nicht-Ziele gewahrt: der Diff
  implementiert exakt den `t*=1,0`-Ausgang (unbedingter Heap, kein Per-Call-Routing), nicht den
  verworfenen Hybrid. Die Append-Abweichung korrekt als außerhalb des Covenant-Mandats an Baustein
  A verwiesen.

**Merge-Fazit:** kein Blocker/Major, keine Code-Änderung nötig. Die einzige offene Aktion (die
D13-Doc-Platzierung, von A als Nit benannt) ist mit dieser Doc-Aktualisierung + den Updates in
roadmap.md/CLAUDE.md/FOLLOWUPS.md erledigt.
