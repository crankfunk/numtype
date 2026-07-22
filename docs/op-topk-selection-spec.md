# `topk`-Selektion: Vollsortierung vs. größenbeschränkter Heap — messen vor entscheiden (bindende Spec)

Status: **bindend** (Owner-Auftrag 2026-07-21: bekannter algorithmischer Defekt aus
docs/op-w1-argmax-topk-ergebnisse.md, Verify-B-Befund F5 — `topkRuntime` sortiert heute das
GESAMTE Array, um die k größten Elemente zu finden).
Version: 6 — siehe Änderungslog am Ende. Kurzbegründung für v6: Eine Frontier-Zweitmeinung fand
einen von allen drei Vorrunden übersehenen Blocker (zwei mandatierte Messläufe, keiner als
verdikt-tragend benannt — 39,6 % der Lauf-Paare divergierten im Verdikt) und zeigte, dass die
rein relativen Schwellen die Regel rausch-fragil machen (eine Sub-Mikrosekunden-Zelle bei
`n = 100` konnte 50-fach-Gewinne bei `n = 10^6` verwerfen). v6 führt das duale Kriterium
(relativ UND absolut) für Verletzung und Gewinn ein, erweitert die Kandidatenmenge um die real
gemessenen `k/n`-Werte und legt den verdikt-tragenden Zusammenzug beider Läufe fest.
Kurzbegründung für v5: der Simulator-Gegenbeweis der
v4-Regel fand einen NEUEN Blocker („hohler Hybrid" — Gewinn-Existenz und Schwellenbildung waren
entkoppelt, wodurch ein Hybrid entstehen konnte, der nirgends besser und in seiner eigenen
Heap-Zone schlechter ist als der Status quo). v5 koppelt beides: erst die sichere Zone
bestimmen, dann den Gewinn INNERHALB dieser Zone verlangen — aus drei Schritten werden zwei.
Kurzbegründung für v4: ein gezieltes zweites
Gegenlesen NUR der Entscheidungsregel D6 (die v3-Neufassung war frisch und unbegutachtet) fand
erneut zwei Blocker und einen inhaltlichen Bruch; v4 ordnet die Prüfschritte um (Gewinn-Gate
ganz nach vorn) und ersetzt die iterative Schwellen-Absenkung durch eine geschlossene Form.
Details im zweiten Addendum. Kurzbegründung für v3: Baustein 0 (adversarial gegen v2)
bestätigte das Fundament (Totalordnung/Bit-Identität, Messdesign D3), fand aber D6 DREIFACH
gebrochen — alle drei Blocker hängen am selben Übergang (Zulässigkeitsprüfung →
Crossover-Bestimmung) und wurden für v3 nicht gepatcht, sondern D6 wurde komplett neu
geschrieben. Details im neuen Abschnitt „Adversariale Spec-Verifikation (Addendum, Baustein
0)" weiter unten.
Datum: 2026-07-21 · Eskalationsleiter: **Stufe 3** (substanzielle Scheibe, M1-Anker
`spike/src/runtime.ts` in mindestens zwei der drei möglichen Ausgänge berührt — voller
Verify-Katalog A+B+C; siehe Verify-Plan unten für die ausgangsabhängige Feinjustierung).
Covenant: v5.

## Ziel & Warum — und was diese Scheibe NICHT ist

`NDArray.topk(k)` (Op-Scheibe W1, `spike/src/ndarray.ts:811-820`) ruft `topkRuntime`
(`spike/src/runtime.ts:669-696`) auf. Die heutige Implementierung baut ein Index-Array über
ALLE `n` Elemente und sortiert es vollständig (`order.sort(...)`, runtime.ts:686) — O(n log n)
plus ein n-elementiges Array geboxter JS-Zahlen — um danach nur die ersten `k` Einträge zu
behalten. Für kleines `k` (der in dieser Bibliothek einzig vorgesehene Anwendungsfall — `torch
.topk`/`np.argpartition`-artige Ranking-Primitiven sind per Definition für `k ≪ n` gedacht) ist
das asymptotisch unnötige Arbeit — ABER asymptotische Unnötigkeit ist keine Konstantenaussage.
Ob ein größenbeschränkter Heap (O(n log k)) in DIESER konkreten JS-Engine, bei DIESEN
realistischen Größen, tatsächlich schneller ist, ist eine empirische Frage, keine, die sich aus
der Big-O-Notation allein beantworten lässt — genau das ist der Punkt, den v1 übersprungen
hatte.

**Ehrlichkeits-Rahmen (Owner-Vorgabe, jetzt geschärft):** Dieser Fix ist **Handwerkspflege, kein
Engpass-Fix**. NumType ist eine typsichere TS-Bibliothek für Shape-geprüfte Arrays —
realistische Nutzung bewegt sich (siehe die eigene Dogfooding-Referenz `examples/rag-demo`) im
Bereich weniger Dutzend bis weniger Tausend Elemente. Weder diese Spec noch das spätere
Ergebnisse-Doc dürfen einen Nutzerbedarf suggerieren, den es nicht gibt. **Diese Scheibe hat
KEINEN bekannten Nutzerbedarf** — sie existiert, weil ein dokumentierter algorithmischer Defekt
(Verify-B F5) offen im FOLLOWUPS steht, nicht weil irgendjemand je auf einen Millisekunden-Ruckler
gestoßen ist. Daraus folgt zwingend: **„wir lassen es, wie es ist" ist ein zulässiger UND
vollständiger Ausgang dieser Scheibe**, wenn die Messung zeigt, dass der Gewinn im realistischen
Bereich klein ist oder der Preis (Komplexität, Verlangsamung im `k`-nahe-`n`-Regime) hoch — nicht
nur theoretisch zulässig, sondern in „Phase 2" unten als GLEICHRANGIGER, mechanisch erreichbarer
Ausgang neben „reiner Heap" und „Hybrid" verankert. Würde diese Spec das nicht so behandeln, wäre
die Messung eine Formalie, die nur das Ergebnis nachträglich rechtfertigt, das sowieso gewollt
war — genau das soll diese Spec verhindern.

Diese Scheibe ist außerdem **kein WASM-Kernel, keine `WNDArray`-Parität, keine Ausweitung auf
`argmax`**. Sie ändert — in JEDEM der drei möglichen Ausgänge — höchstens den
SELEKTIONSALGORITHMUS innerhalb der bestehenden, NDArray-only TS-Referenzfunktion `topkRuntime`.
Signatur, Aufrufer (`ndarray.ts:811-820`), Validierungsverhalten und jede beobachtbare Ausgabe
bleiben in JEDEM Ausgang unverändert.

## Berührte Covenant-Invarianten

- **M1** (Anker `crates/core/src/`, `spike/src/runtime.ts` — COVENANT.md:11-17): `runtime.ts`
  ist ein M1-Anker. Phase 1 (die Messung) berührt ihn NICHT (der Bench importiert `topkRuntime`
  unverändert, siehe D2 unten). Phase 2 berührt ihn NUR bei den Ausgängen „reiner Heap" und
  „Hybrid" (siehe D8/D9), NICHT bei „Status Quo" (D10). In JEDEM Fall gilt: `topk` hat KEINEN
  WASM-Kernel (unverändert durch diese Scheibe) — M1s Bit-Identitäts-PFLICHT bindet damit in
  keinem Ausgang für `topk`. Bit-Identität zur heutigen Semantik bleibt trotzdem in JEDEM
  code-ändernden Ausgang eine eigenständige, strengere Owner-Anforderung dieser Scheibe (Die
  tragende Beobachtung unten), nicht M1-Erfüllung.
- **M2/M3**: NICHT berührt, in keinem Ausgang — kein Byte an `spike/src/vector.ts`
  (`TopkCheck`/`TopkShape`), `spike/src/reduce.ts` oder `spike/src/dim.ts` ändert sich, weil
  `topkRuntime`s UND `NDArray.topk`s Signatur in jedem Ausgang unverändert bleiben.
- **M4** (Anker `abi.rs`/`matmul_blocked.rs`/`shape.rs`): NICHT berührt — kein Rust-Byte ändert
  sich, `check:freeze`-Hash-Pin muss byte-identisch bleiben.
- **M5/Z1**: NICHT berührt. Jeder vermessene/gebaute Kandidat ist vollständig from-scratch
  (Hard Constraint „No external libraries", CLAUDE.md:15) — keine neue Abhängigkeit.
- **Z2**: Der neue Bench (Phase 1, D4) liegt unter `spike/bench-core/`, das per
  `tsc --noEmit --listFilesOnly` (empirisch geprüft, siehe Vorab-Probe unten) bereits Teil des
  ROOT-`tsc`-Korpus ist — `pnpm check`/`check:diag` erfassen die neue Datei automatisch. Dies
  gilt UNABHÄNGIG vom späteren Ausgang, da Phase 1 in jedem Fall durchgeführt wird.

## Die tragende Beobachtung: der heutige Comparator ist eine strikte Totalordnung

Dieses Fundament gilt für JEDEN möglichen Ausgang gleichermaßen — es beweist, dass ein
KORREKTER Kandidat (Heap oder Hybrid) bit-identisch zur heutigen Vollsortierung sein MUSS, egal
ob er am Ende tatsächlich verbaut wird oder nicht. Ohne diesen Beweis wäre selbst die MESSUNG in
Phase 1 wertlos, da ein Heap-Kandidat, der eine andere (aber vielleicht plausibel aussehende)
Auswahl trifft, gar nicht erst korrekt verglichen werden dürfte. Baustein 0 hat diese Kette
unabhängig und HÄRTER geprüft als die eigene Vorab-Probe dieser Spec (269.984 paarweise
Antisymmetrie-Checks, 1.500 Shuffle-Invarianz-Trials, mehrere distinkte NaN-Payloads, 3.031
Fuzz-Fälle gegen den Heap-Kandidaten — null Abweichungen; Details im Addendum unten). Das
Fundament steht.

`topkRuntime` sortiert heute mit (runtime.ts:686):

```ts
order.sort((i, j) => topkCompareValues(data[i] ?? 0, data[j] ?? 0) || i - j);
```

`topkCompareValues` (runtime.ts:645-654) klassifiziert genau nach D4 der W1-Spec (NaN zuerst,
dann absteigend nach Wert, `0`/`-0` via `>`/`<`, nie `Object.is`):

```ts
function topkCompareValues(a: number, b: number): number {
  const aNaN = Number.isNaN(a);
  const bNaN = Number.isNaN(b);
  if (aNaN && bNaN) return 0;
  if (aNaN) return -1;
  if (bNaN) return 1;
  if (a > b) return -1;
  if (a < b) return 1;
  return 0;
}
```

`topkCompareValues` gibt `0` zurück GENAU DANN, wenn zwei Werte in D4s Ordnung gleichrangig sind
(beide NaN, oder wertgleich inkl. `0`/`-0`). In diesem Fall entscheidet `|| i - j` — und weil
`order` (runtime.ts:685) eine Permutation der PAARWEISE VERSCHIEDENEN Indizes `0..n-1` ist,
vergleicht `.sort()` niemals einen Index mit sich selbst: für jedes im Comparator tatsächlich
aufgerufene Paar gilt `i ≠ j`, also `i - j ≠ 0`. Der volle Ausdruck
`topkCompareValues(...) || (i - j)` liefert damit für JEDES Paar verschiedener Indizes einen
von Null verschiedenen, eindeutigen Verdikt — nie einen echten Gleichstand. Das ist per
Definition eine **strikte Totalordnung** auf den Indizes `0..n-1` (Antisymmetrie: aus `cmp(i,j)
< 0` folgt `cmp(j,i) > 0`, da beide Summanden bei Vertauschung ihr Vorzeichen wechseln;
Transitivität: die Ordnung ist eine lexikographische Kombination zweier totaler Ordnungen —
NaN-Klasse/Wert, dann Index — und lexikographische Kombinationen totaler Ordnungen sind
transitiv). Diese Konstruktion — `|| i - j` als expliziter, deterministischer Tiebreak — ist im
Code selbst kein Zufall: sie macht das Ergebnis unabhängig von der (in JS vor ES2019 nicht
garantierten, und selbst danach für diesen Zweck nicht benötigten) Stabilität von
`Array.prototype.sort`.

**Konsequenz (das Fundament dieser Spec):** Weil die Ordnung strikt total ist, gibt es zu jedem
`k` GENAU EINE Menge von `k` Indizes, die „die k größten Elemente in Comparator-Reihenfolge"
bilden — es gibt keinen Grenzfall, in dem zwei verschiedene, gleich gültige Top-k-Mengen
existieren könnten. Jeder Algorithmus, der diese eindeutige Menge korrekt bestimmt —
Vollsortierung, größenbeschränkter Heap, oder ein Hybrid aus beiden — liefert zwangsläufig
DIESELBE Indexmenge in DERSELBEN Ausgabereihenfolge. Bit-Identität zwischen JEDEM korrekten
Kandidaten und der heutigen Vollsortierung ist damit **konstruktiv beweisbar** — eine
Eigenschaft der Ordnung, nicht des Auswahlverfahrens. Das ist die Grundlage, auf der Phase 1
überhaupt „Kandidat A vs. Kandidat B" sinnvoll vergleichen kann (beide MÜSSEN, korrekt
implementiert, dasselbe Ergebnis liefern — der Vergleich ist rein eine Frage der LAUFZEIT, nie
der Korrektheit) und auf der ein späterer Differentialtest (D11) bit-identisches Verhalten
beweisen wird, falls ein code-änderender Ausgang eintritt.

---

## Phase 1 — Messung (bindend, zuerst; keine Implementierungsentscheidung)

Zweck: die tatsächliche Laufzeitcharakteristik von Vollsortierung vs. Heap über ein
vorregistriertes, konkretes Raster ermitteln — kalibriert, nicht geschätzt, in Anlehnung an das
Muster, das `THREADED_MATMUL_MIN_POOL_WORK` bereits im Repo etabliert hat
(`spike/bench-core/threaded-crossover.ts`, docs/kern-06-ergebnisse.md, Abschnitt
„Follow-up addendum: size-based auto-routing").

- **D1 — Scope der Messung:** Ausschließlich ein neuer, informativer Bench (kein Gate). KEIN
  Byte an `spike/src/runtime.ts`, `ndarray.ts`, `vector.ts` ändert sich in Phase 1 — der Bench
  importiert `topkRuntime` UNVERÄNDERT aus `runtime.ts` als Kandidat A und definiert Kandidat B
  (Heap) sowie — bedingt, siehe D7 — Kandidat C (Hybrid) ausschließlich BENCH-LOKAL.
- **D2 — Kandidaten:**
  - **Kandidat A („Sort", heutiges Verhalten):** die bestehende, unveränderte, importierte
    `topkRuntime` (runtime.ts:669-696) — KEINE Kopie, das echte Produktionsverhalten.
  - **Kandidat B („Heap"):** größenbeschränkter Binär-Heap der festen Kapazität `k`, O(n log k),
    bench-lokal implementiert (keine Produktionsdatei berührt). Beschreibung (unverändert aus
    v1 übernommen — das ist die Konstruktion, die Phase 1 tatsächlich BAUEN und vermessen muss,
    nicht mehr die für die Produktion GESETZTE Form):
    1. `k = 0`: sofortiger Rückgabepfad, keine Heap-Operation.
    2. Scan `i = 0..n-1` aufsteigend: Kandidat `(data[i], i)`. Solange der Heap `< k` Elemente
       hält: einfügen. Ist er voll: gegen die Heap-WURZEL vergleichen (die Wurzel hält das
       SCHLECHTESTE der aktuell gehaltenen `k` Elemente) — rankt der Kandidat per Totalordnung
       VOR der Wurzel, wird die Wurzel ersetzt (Resift); sonst verworfen.
    3. Nach dem Scan hält der Heap exakt die (laut obiger Beobachtung eindeutige) Top-k-Menge.
    4. Die `k` gehaltenen Paare werden EIN letztes Mal sortiert — mit DEMSELBEN
       Comparator-Ausdruck `topkCompareValues(valA, valB) || (idxA - idxB)`, den auch Kandidat A
       verwendet (Wiederverwendung, keine unabhängig hergeleitete „äquivalente" Neuformulierung)
       — O(k log k). `values[i]` wird per direktem Element-Read aus dem URSPRÜNGLICHEN
       `data`-Array gesetzt (nicht aus einem zwischengespeicherten Heap-Wert), identisch zum
       Daten-Fluss von Kandidat A, damit keine neue Kopier-Stelle für NaN-Payloads entsteht.
    5. `k = n`: kein Sonderfall — der Heap füllt sich bis zum Schluss, kein Element wird
       verworfen, Ergebnis ist die volle sortierte Folge (das erwartbar ungünstigste Regime für
       Kandidat B relativ zu Kandidat A, siehe die Vorab-Probe unten).
    Heap-Invariante/Repräsentation: die WURZEL hält das schlechteste Element (Max-Heap der
    Schlechtigkeit bezüglich derselben Totalordnung); Knoten-Repräsentation (Objekt-Array vs.
    parallele typisierte Arrays) ist dem Bench-Autor freigestellt, solange die beobachtbare
    Semantik (Bit-Identität) und die Komplexitätsklasse (O(n log k)) eingehalten sind.
  - **Kandidat C („Hybrid"):** siehe D7 — wird NUR bedingt gebaut und vermessen, NACHDEM Phase 2
    (D6) einen Schwellenwert-Kandidaten geliefert hat, nicht vorab.
- **D3 — Mess-Design (verbindlich, konkretes Raster statt „verschiedene Größen"):**
  - **`n`-Reihe (fünf Werte, deckt realistisch bis Stresstest ab):**
    `N_VALUES = [100, 1_000, 10_000, 100_000, 1_000_000]`.
  - **`k`-Reihe je `n`, aus ZWEI Quellen vereinigt, dedupliziert, sortiert, auf `k ≤ n`
    geklemmt:**
    - `ABSOLUTE_K = [1, 5, 10, 20, 50]` — realistische, von `n` UNABHÄNGIGE `k`-Werte (die
      tatsächliche Nutzungsform: ein Top-10/Top-20 unabhängig davon, wie groß der Vektor ist).
    - `RATIO_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9,
      0.95, 1.0]`, aufgelöst zu `k = round(n * frac)` je `n` — eine über alle `n` VERGLEICHBARE
      `k/n`-Serie, die den Crossover-Bereich abbildet. Dichte: erhöht zwischen 0,1 und 0,5 (laut
      der informellen Vorab-Probe unten der plausibelste Umschlagbereich) UND zwischen 0,7 und
      1,0 (**v3-Ergänzung nach Baustein-0-Befund**: die v2-Fassung hatte hier eine Lücke von 0,3
      — die größte Schrittweite der gesamten Reihe, genau im laut Vorab-Probe steilsten Bereich
      — ohne Schließung hätte eine Schwelle einen NIE vermessenen Bereich als heap-sicher
      ausweisen können; `0.8`/`0.85`/`0.9`/`0.95` schließen sie). Dichte ist eine
      Grid-Design-Entscheidung, KEINE Vorwegnahme des Ergebnisses.
    - Beispiel vollständig aufgelöst für `n = 1_000`: `ABSOLUTE_K` liefert `{1, 5, 10, 20, 50}`;
      `RATIO_FRACTIONS` liefert `{10, 20, 50, 100, 150, 200, 300, 400, 500, 700, 800, 850, 900,
      950, 1000}`; Vereinigung (dedupliziert) = `{1, 5, 10, 20, 50, 100, 150, 200, 300, 400,
      500, 700, 800, 850, 900, 950, 1000}` — 17 Zellen. Analog für jedes andere `n` (die exakte
      Formel, nicht die einzeln ausgeschriebenen Listen, ist die verbindliche Vorgabe — beide
      Konstanten `ABSOLUTE_K`/`RATIO_FRACTIONS` werden WÖRTLICH als benannte Konstanten im
      Bench-Skript geführt, keine abweichenden Werte).
  - **Reihenfolgen-Pflicht (Owner-Vorgabe, gegen Aufwärm-Bias):** JEDE Rasterzelle wird in BEIDEN
    Reihenfolgen gemessen — Reihenfolge A = [Sort zuerst warm/messen, dann Heap],
    Reihenfolge B = [Heap zuerst warm/messen, dann Sort] — im SELBEN Prozess, auf DENSELBEN
    (einmal pro Zelle generierten, für beide Reihenfolgen wiederverwendeten) Operanden. Für
    jeden Kandidaten wird aus den zwei resultierenden Medianen (einer je Reihenfolge) das
    KONSERVATIVERE Verhältnis für die Entscheidungsregel (Phase 2) verwendet: `ratio = max(
    heapMedian_A / sortMedian_A, heapMedian_B / sortMedian_B)` — die größere (für den Heap
    ungünstigere) der beiden Reihenfolgen-Ablesungen zählt, damit eine für den Heap günstige
    Aufwärm-Reihenfolge ihn nicht künstlich besser aussehen lässt, als er ist. Beide
    Einzelwerte werden trotzdem vollständig berichtet (nicht nur das Maximum) — eine Differenz
    zwischen den zwei Reihenfolgen jenseits von 10 % relativ ist selbst ein zu berichtender
    Befund (Aufwärm-Sensitivität dieser Zelle), kein stillschweigend weggemitteltes Rauschen.
  - **Warmup/Timing-Protokoll — identisch zum etablierten `measureRange`-Muster aus
    `spike/bench-core/threaded-crossover.ts:107-144`, nicht neu erfunden:** pro (Kandidat,
    Zelle, Reihenfolgen-Position) adaptives Tier-up-Warmup bis `elapsed ≥ 30ms` ODER
    `warmupCalls ≥ 64` (was zuerst eintritt), danach `N_REPS = 7` batch-getimte Samples mit
    einer aus der Warmup-Rate abgeleiteten Batch-Größe (`min(512, max(1, ceil(2 / estMs)))`).
    Berichtet werden Median UND volle Min-Max-Spanne (nie ein Einzelpunkt) — exakt die in der
    coding-kb-Notiz „js-wasm-benchmarks-jit-zustand-und-crossover" festgehaltene Disziplin
    („JIT-Zustand ist Teil der Messung… Headline-Zahlen als Bereich… nie als Ein-Punkt-Wert";
    „µs-Benches: pro Modul/Route warmlaufen, 2 Warmup-Calls ließen das zuerst gemessene Modul
    teils in Liftoff zurück — sichtbar als systematischer Spread"). Für sehr teure Zellen
    (`n = 1_000_000`, Kandidat A nahe `k = n`: Einzelaufruf ca. 280-290 ms laut Vorab-Probe)
    kann die 30-ms-Grenze bereits nach EINEM Warmup-Call erreicht sein — das ist eine bekannte,
    im Präzedenzfall akzeptierte Eigenschaft des adaptiven Protokolls (kein zusätzlicher
    Mindest-Call-Boden wird eingeführt, um vom etablierten Muster nicht abzuweichen), aber die
    berichtete Min-Max-Spanne macht eine dadurch entstehende Instabilität SICHTBAR statt sie zu
    verstecken.
  - **Bit-Identitäts-Gate VOR jeder Zeitmessung**, für JEDEN Kandidaten einzeln, an JEDER Zelle
    (nicht nur stichprobenartig) — `indices` exakt gleich, `values` bit-identisch (Wiederverwendung
    der Totalordnungs-Argumentation oben: eine Abweichung hier ist ein Kandidaten-BUG, keine
    zulässige Perf-Variante, und die Zeitmessung dieser Zelle wird nicht durchgeführt, sondern
    der Bench bricht mit einer benannten Fehlermeldung ab — „nie ein falsches Ergebnis timen",
    exakt die Disziplin von `slice.ts:52-61`/`threaded-crossover.ts:182-188`).
  - **Host-Zustand:** wie beim Präzedenzfall dokumentiert berichten (Systemlast, Anzahl Läufe) —
    mindestens EIN vollständiger Zweitlauf des gesamten Rasters zur Stabilitätsprobe (der
    Präzedenzfall lief zweimal und verglich; dieselbe Disziplin gilt hier).
  - **Welcher Lauf das Verdikt trägt (v6, Blocker-Fix der Frontier-Zweitmeinung — bis v5
    ungeregelt):** Zwei mandatierte Läufe drucken zwei Verdikt-Zeilen, und v5 sagte nirgends,
    welche zählt. In einer Monte-Carlo-Probe mit plausibler Messstreuung wichen **39,6 % der
    Lauf-Paare im Verdikt voneinander ab** — damit wäre genau die Ermessensentscheidung NACH
    der Messung entstanden, die diese ganze Vorregistrierung verhindern soll. Bindend ist
    deshalb: **Das Verdikt wird auf dem PESSIMISTISCHEN Zusammenzug beider Läufe berechnet** —
    je Zelle das Maximum der `ratio`-Werte und das jeweils ungünstigere absolute Delta über
    beide Läufe. Es gibt damit nur EIN Verdikt, nicht zwei, und es ist per Konstruktion das
    vorsichtigere. Weichen die Einzel-Lauf-Verdikte voneinander ab, ist das ein
    BERICHTSPFLICHTIGER Befund im Ergebnisse-Doc (er zeigt, dass die Messung an der
    Entscheidungsgrenze nicht stabil ist), aber kein Entscheidungsproblem mehr.
- **D4 — Bench-Datei & Script:** `spike/bench-core/topk-selection.ts`. KEIN WASM-Bezug (kein
  `initCore()`, kein `WNDArray`, kein `pnpm build:wasm`-Vorlauf — reiner TS-zu-TS-Vergleich).
  Neuer npm-Script-Eintrag `"bench:topk": "node spike/bench-core/topk-selection.ts"` (kein
  `build:wasm`-Präfix). Kein CI-Gate (wie `bench:scaling`/`bench:chain`/etc., nur `bench:editor`
  ist hart) — Datenquelle fürs Ergebnisse-Doc. **Verbindlich: das Skript druckt am Ende eine
  MECHANISCH berechnete Verdikt-Zeile**, die Phase 2s Entscheidungsregel (D6) direkt auf die
  gemessenen Zahlen anwendet und exakt einen der drei Ausgänge benennt (`PURE HEAP` /
  `HYBRID (threshold=…)` / `STATUS QUO (leave as is)`) — analog zu
  `editor-latency.ts`s `enforceHardGate`-Verdikt-Druck und `threaded-crossover.ts`s
  `POOL`-Markierung. Zweck: die Entscheidung folgt aus dem Skript selbst, nicht aus einer
  nachträglichen menschlichen Lektüre der Rohzahlen, die post-hoc einen anderen Ausgang
  rechtfertigen könnte, als die vorregistrierte Regel tatsächlich verlangt.

## Phase 2 — Entscheidungsregel (bindend als REGEL, nicht als Ergebnis)

Diese Regel wird JETZT festgelegt, BEVOR Phase 1 läuft — sie ist mechanisch auf die
Phase-1-Ausgabe anwendbar, ohne dass irgendeine Ermessensentscheidung nach der Messung nötig
wäre. Kein in dieser Regel verwendeter Schwellenwert ist eine Schätzung der Größe, die die
Messung selbst ermitteln soll (der k/n-Kreuzungspunkt bleibt vollständig offen) — die einzige
vorab fixierte Zahl ist eine Toleranzschwelle (D6), in Anlehnung an den
`THREADED_MATMUL_MIN_POOL_WORK`-Präzedenzfall gewählt und unten begründet.

- **D5 — Realistisches Spektrum (Definition, NUR NOCH für Berichterstattung — s. u.):** alle
  gemessenen Rasterzellen mit `n ≤ 10_000` (drei der fünf `n`-Werte: 100, 1.000, 10.000) —
  direkt aus dem Ehrlichkeits-Rahmen abgeleitet („weniger Dutzend bis weniger Tausend
  Elemente"). **v3-Korrektur (Baustein-0-Befund 1, s. Addendum):** dieses Spektrum gate(t) in
  D6 NICHTS mehr — die v2-Fassung ließ die Zulässigkeitsprüfung für „reiner Heap" NUR auf diesem
  Teilraster laufen, wodurch eine unbedingte Ersetzung `n = 100.000`/`1.000.000` blind ersetzt
  hätte, ohne dort je geprüft worden zu sein. D6 prüft jetzt in JEDEM Schritt das GESAMTE
  Raster. Diese Definition bleibt ausschließlich für die Ergebnisse-Doc-Erzählung nützlich (was
  für reale Nutzung zählt), nicht für die Entscheidung selbst.
- **D6 — Die Entscheidungsregel (v3, komplett neu formuliert — Baustein 0 fand v2s Fassung
  DREIFACH gebrochen, s. Addendum unten für die drei Blocker):**

  **Tragende Vereinfachung (der Kern dieser Neufassung):** der Hybrid-Schwellenwert ist EINE
  EINZIGE Konstante `TOPK_HEAP_MAX_K_OVER_N_RATIO` — unterhalb läuft der Heap, oberhalb die
  heutige Sortierung. Die Fehlerfolgen sind ASYMMETRISCH: ein zu HOHER Wert ist gefährlich (der
  Heap liefe dort, wo er tatsächlich verliert — eine echte Regression). Ein zu NIEDRIGER Wert
  kostet dagegen NUR entgangene Beschleunigung und ist NIE schlechter als der heutige Zustand,
  weil oberhalb der Schwelle exakt das heutige, unveränderte Verfahren läuft. Aus dieser
  Asymmetrie folgt zwingend: ein MINIMUM über alle gemessenen `n` ist immer sicher, unabhängig
  davon, wie sehr die einzelnen Kreuzungspunkte streuen. Die Streuung ist damit ein BEFUND, der
  berichtet wird — kein Entscheidungs-Gate. v2 verwechselte „eine `n`-abhängige Schwellen-
  FUNKTION" (tatsächlich unerwünschte Komplexität) mit „eine einzige, konservative Schwellen-
  KONSTANTE" (einfach und sicher) und leitete daraus einen Stabilitäts-Gate ab, der so nicht
  gebraucht wird — er entfällt in v3 ersatzlos.

  **Ausführungsreihenfolge (v5 — zwei Schritte statt drei):** Die Regel besteht seit v5 aus
  genau ZWEI Prüfungen: erst wird die sichere Zone bestimmt, dann wird gefragt, ob in DIESER
  Zone überhaupt ein Gewinn liegt. Diese Kopplung ist der Kern der v5-Korrektur (Befund
  „hohler Hybrid" des zweiten D6-Gegenlesens, s. Addendum): v4 prüfte die Gewinn-Existenz
  über das GESAMTE Raster, bestimmte die Schwelle aber unabhängig davon — beide Kriterien
  waren strukturell entkoppelt. Lag der einzige gemessene Gewinn bei großem `k/n` und eine
  isolierte Regression bei kleinem `k/n`, zog v4 die Schwelle unter die Regression und lieferte
  einen Hybrid, dessen Heap-Zone NIRGENDS gewinnt (dort nur die verträglichen, aber langsameren
  Zellen) und dessen einziger echter Gewinn permanent im Sort-Zweig landet — also gegenüber
  „nichts tun" ausschließlich eine Regression, nirgends ein Vorteil. Mechanisch eindeutig
  berechnet, ohne jede Lesart-Divergenz, und trotzdem falsch. Im gezielt positions-entkoppelten
  Fuzz waren **29,4 % aller v4-Hybrid-Verdikte** beweisbar solche Netto-Regressionen.

  **Was als Verletzung und was als Gewinn zählt (v6 — duales Kriterium, Owner-Entscheidung
  2026-07-22).** Beide Gates arbeiten mit **relativ UND absolut**, nicht nur relativ:
  - Eine Zelle ist eine **Verletzung**, wenn `ratio > 1.15` UND das absolute Zeitdelta
    `heapMedian − sortMedian` größer als `ABS_RELEVANCE_US` ist.
  - Eine Zelle ist ein **Gewinn**, wenn `ratio ≤ 1/1.15` UND das absolute Zeitdelta
    `sortMedian − heapMedian` mindestens `ABS_RELEVANCE_US` beträgt.

  `ABS_RELEVANCE_US = 10` (Mikrosekunden), vorregistriert. **Begründung und Abgrenzung zur
  Hausregel:** Dies ist eine RELEVANZ-Schwelle, keine Schätzung der Größe, die die Messung
  ermitteln soll (der Crossover bleibt vollständig offen) — dieselbe Kategorie wie D7s
  bereits begründete „5 % relativ ODER 50 µs absolut". D6 wählt bewusst den strengeren
  absoluten Wert, weil D6 entscheidungstragend ist und D7 nur informativ. Der Anlass ist ein
  gemessener Befund der Frontier-Zweitmeinung: Ohne absolutes Kriterium hängt das Verdikt an
  einem UND über ca. 22 Zellen am Zonenboden, deren kleinste bei `n = 100` im Bereich von
  Bruchteilen einer Mikrosekunde liegen. Eine einzige solche Zelle bei `1,16` statt `1,15`
  konnte gemessene 50-fach-Gewinne bei `n = 10^6` verwerfen — das Verdikt hing damit eher an
  Messvarianz als an Algorithmus-Wahrheit. Verschärfend: Das Verhältnis ist per Konstruktion
  das MAXIMUM über beide Messreihenfolgen (also einseitig nach oben verzerrt), und D3 selbst
  erwartet Reihenfolgen-Spreads bis 10 % als normal — die Hälfte des 15-%-Bands war damit
  durch einkalkuliertes Rauschen belegt. Das duale Kriterium beendet zugleich einen internen
  Widerspruch der Spec: D7 begründet für seinen eigenen Check bereits wörtlich, warum bei
  kleinen `n` eine feste µs-Grenze aussagekräftiger ist als ein Prozentsatz — nur das
  entscheidungstragende Gate hatte dieses Kriterium bis v5 nicht.

  1. **Sichere Zone bestimmen (rein aus der Sicherheitsbedingung, ohne jeden Gewinn-Bezug).**
     Berechne für JEDE gemessene Rasterzelle (alle fünf `n`, alle `k` — `ABSOLUTE_K` UND
     `RATIO_FRACTIONS`, D3) das `ratio` (Maximum über beide Reihenfolgen, D3) und das absolute
     Zeitdelta. Dann ist `t*` der **größte Wert aus der Kandidatenmenge, für den das GESAMTE
     Raster `t`-sicher ist** — wobei „`t`-sicher" heißt: KEINE gemessene Zelle (beide Familien,
     alle `n`) mit `k/n ≤ t` ist eine Verletzung im obigen dualen Sinn.

     **Kandidatenmenge (v6 erweitert):** `{0} ∪ RATIO_FRACTIONS ∪ {alle im Raster tatsächlich
     vorkommenden k/n-Werte}`. Die Erweiterung um die real gemessenen Verhältnisse ist
     selbst-kalibrierend und braucht keine geratene Zahl: Sie stellt sicher, dass für JEDE
     gemessene Zelle ein Kandidat existiert, der die Zone exakt vor dieser Zelle enden lässt.
     v5s Menge endete nach unten bei `0,01`, wodurch eine Verletzung UNTERHALB dieser Fraktion
     (z. B. bei `k/n = 0,005`) die Zone zwangsweise auf `0` nullte, obwohl eine kleinere,
     sichere Zone existiert hätte — ein reines Auflösungs-Artefakt der Kandidatenmenge, kein
     Messbefund. Die Toleranzschwelle **15 %** bleibt unverändert (in
     Anlehnung an die Kern-06-Präzedenzmessung gewählt: deren eigene „wash"-Zellen — Fälle, die
     als praktisch gleichauf galten, bevor eine Route bevorzugt wurde — lagen bei Verhältnissen
     bis 1,16, siehe docs/kern-06-ergebnisse.md-Tabelle, Zeile „0.11 (n=48)"; dieses Band
     entstand dort aus Thread-Dispatch-Rauschen, ist also eine Größenordnungs-ANALOGIE, keine
     direkt übertragbare Messung — **15 %** ist folglich keine freie Erfindung, aber auch keine
     strikt hergeleitete Zahl). `t*` existiert IMMER, weil `0` immer Kandidat und immer
     `t`-sicher ist (die Zone `k/n ≤ 0` ist leer).
  2. **Nutzen-Prüfung INNERHALB der sicheren Zone.** Existiert mindestens eine gemessene Zelle
     mit `k/n ≤ t*`, die ein **Gewinn im dualen Sinn** ist (relativ `ratio ≤ 1/1.15`, also der
     Heap mindestens 15 % schneller, UND absolut mindestens `ABS_RELEVANCE_US` gespart)? Das
     absolute Kriterium gilt hier aus demselben Grund wie bei der Verletzung: Ein „Gewinn" von
     Nanosekunden bei `n = 100` ist ein Rausch-Artefakt und darf den kompletten
     Implementierungs-, Orakel- und Differentialtest-Apparat (D8/D11) nicht auslösen.
     - **NEIN** → **Ausgang „Status Quo"** (D10), FERTIG. Kein bewährter Sortier-Code wird
       ersetzt, wenn in der Zone, die der Heap überhaupt bedienen dürfte, kein Gewinn gemessen
       wurde. Das deckt beide früher getrennten Fälle mechanisch ab: `t* = 0` (keine nichtleere
       sichere Zone) fällt automatisch hierher, weil eine leere Zone keine Gewinn-Zelle
       enthalten kann; und der „hohle Hybrid" fällt hierher, weil sein Gewinn außerhalb der
       Zone liegt.
     - **JA und `t*` deckt jede gemessene Zelle ab** (also `t* = 1.0`, weil die Zelle `k = n`
       existiert — das gesamte Raster ist sicher, die Zone umfasst jede mögliche Eingabe) → **Ausgang „reiner Heap"** (D8), FERTIG.
       Eine Schwelle, die jede mögliche Eingabe einschließt, IST die verzweigungsfreie
       Ersetzung; sie wird als solche implementiert, nicht als Hybrid mit einer immer wahren
       Bedingung.
     - **JA, sonst** → **Ausgang „Hybrid"** (D9) mit `TOPK_HEAP_MAX_K_OVER_N_RATIO = t*`.

     **Wichtiger Hinweis für die Auswertung (unverändert gültig):** die `ABSOLUTE_K`-Zellen
     (`k` = 1 bis 50, unabhängig von `n`) SIND der realistische Anwendungsfall dieser
     Bibliothek (`topk(5)` auf einer Embedding-Matrix, unabhängig davon, wie groß die Matrix
     ist) — gewinnt der Heap DORT nennenswert, ist ein Hybrid sinnvoll, selbst wenn `t*`
     numerisch klein ausfällt (ein kleiner `k/n`-Schwellenwert deckt genau diese Zellen ab,
     weil `k/n` für kleines festes `k` bei großem `n` ohnehin winzig ist — „klein" ist hier
     kein Zeichen geringer Nützlichkeit). Genau deshalb prüft Schritt 2 die Gewinn-Existenz
     INNERHALB der Zone: diese Zellen liegen dort, ein Gewinn bei großem `k/n` dagegen nicht.

     **Warum diese geschlossene Form (v4-Korrektur, Befunde 1, 2 und 4 des D6-Gegenlesens):**
     Sie ersetzt die v3-Konstruktion (Kreuzungspunkte `c(n)` je `n`, Minimum, dann iteratives
     „Absenken" mit Nachvalidierung) VOLLSTÄNDIG durch eine einzige, ordnungsunabhängige
     Definition. Damit fallen drei Fehler der v3-Fassung zugleich weg:
     - **Kein undefiniertes Verdikt (v3-Befund 1):** `0` ist immer in der Kandidatenmenge und
       immer `t`-sicher (die Zone `k/n ≤ 0` ist leer), also existiert der „größte sichere `t`"
       IMMER. Der Fall „nur eine `ABSOLUTE_K`-Zelle verletzt, kein `c(n)` definiert" (der die
       v3-Regel abbrechen ließ) landet hier sauber auf `t = 0` → Status Quo.
     - **Keine Doppel-Lesart am unteren Rand (v3-Befund 2):** es gibt kein iteratives „senke
       auf den nächstkleineren Wert unterhalb dieser Zelle", das ohne Kandidaten unterspezifiziert
       wäre — die geschlossene Form prüft schlicht alle Kandidaten und nimmt den größten sicheren.
     - **Beide Zell-Familien sind eingeschlossen (v3-Befund 4):** die `t`-sicher-Bedingung läuft
       über JEDE Zelle mit `k/n ≤ t`, gleich ob sie aus `ABSOLUTE_K` oder `RATIO_FRACTIONS`
       stammt — die separate Nachvalidierung entfällt, weil sie in die Definition eingebaut ist.
       Eine `ABSOLUTE_K`-Regression bei kleinem `k/n` senkt `t` genau so weit, dass diese Zelle
       oberhalb der Schwelle (im Sort-Pfad) landet.
     Terminierung ist trivial (endliche Kandidatenmenge, ein Durchlauf). Die Streuung der
     per-`n`-Kreuzungspunkte `max/min` wird weiterhin VOLLSTÄNDIG im Ergebnisse-Doc berichtet
     (informativer Befund über die `n`-Abhängigkeit) — sie ENTSCHEIDET NICHTS (v2s
     Stabilitätsklausel bleibt ersatzlos entfallen: das größte sichere `t` ist unabhängig von
     der Streuung sicher, weil oberhalb von `t` in jedem Fall das unveränderte Sortier-Verfahren
     läuft).

- **D7 — Hybrid als dritter VERMESSENER Kandidat (Stage B, nur falls D6 auf „Hybrid" führt):**
  Führt die Regel auf Hybrid, wird `spike/bench-core/topk-selection.ts` (D4) im SELBEN Lauf
  automatisch um eine zweite Stufe erweitert (kein separates Skript, kein manueller
  Zwischenschritt — Baustein 0 fand hierfür keine technische Hürde und hält es für robuster als
  ein manueller Zwischenschritt): ein dritter Kandidat `hybridSelect(data, k, n,
  thresholdRatio)` — die triviale Verzweigung `k / n <= thresholdRatio ? Kandidat-B-Pfad :
  Kandidat-A-Pfad` — wird mit dem in D6 Schritt 1 bestimmten, ganzraster-sicheren
  `thresholdRatio` instanziiert und über DIESELBEN Rasterzellen (D3) nach DEMSELBEN Protokoll
  (beide Reihenfolgen, Warmup, `N_REPS`, Bit-Identitäts-Gate) vermessen. Berichtet wird für jede
  Zelle der ABSOLUTE Dispatch-Mehraufwand `hybridMedian − min(heapMedian, sortMedian)` (in µs,
  wie beim Kern-06-Präzedenzfall, der den reinen Pool-Dispatch-Overhead ebenfalls absolut in µs
  auswies, 13-40µs, statt ihn als vernachlässigbar anzunehmen). Toleranzgate (informativ, kein
  Blocker, aber zu berichten): Mehraufwand `≤ 5 % relativ ODER ≤ 50 µs absolut` (das Maximum aus
  beiden — bei sehr kleinen `n` ist eine feste µs-Grenze aussagekräftiger als ein Prozentsatz
  eines bereits sub-Mikrosekunden-Aufrufs, bei sehr großen `n` umgekehrt). Eine Überschreitung
  ist kein Scheitern der Scheibe, sondern ein zu dokumentierender Befund — die Verzweigung
  selbst bleibt in JEDEM Fall der einzige praktikable Hybrid-Mechanismus (eine komplexere
  Dispatch-Logik wäre wieder die in D6 ausgeschlossene Eskalation).

---

## Umsetzungs-Konsequenzen je Ausgang (vollständig ausformuliert — die spätere Implementierung improvisiert nichts)

Owner-Entscheidung 1 (in-place ersetzen statt anhängen) und Owner-Entscheidung 2 (heutiges
Verfahren wörtlich als Test-Orakel) aus der ursprünglichen Auftragslage bleiben in JEDEM
code-ändernden Ausgang gültig — sie betreffen die FORM einer späteren Implementierung, nicht die
Wahl des Verfahrens, die jetzt Phase 2 trifft.

**Vorab-Genehmigung der In-Place-Abweichung (verbindlich, gilt für D8 UND D9 — v3-Ergänzung
nach Baustein-0-Befund 5, s. Addendum):** `docs/verify-runde-template.md:72` nennt `runtime.ts`
EXPLIZIT als append-only-Datei, die Baustein A per „Disziplin-Prüfung am Diff" mechanisch
kontrolliert („append-only-Dateien (runtime.ts, gefrorene Rust-Dateien) nur Additionen;
TS-Klassenkörper insertion-only bzw. bestätigte Abweichung") — und aus GENAU diesem Template
wird der spätere Baustein-A-Auftrag instanziiert. Ohne einen Vermerk HIER würde ein
Baustein-A-Verifier den In-Place-Edit an `topkRuntime` zu Recht als Verstoß flaggen. Diese Spec
IST hiermit die vom Template selbst vorgesehene „bestätigte Abweichung" für `topkRuntime` —
VORAB, IN DER SPEC SELBST, disclosed UND bestätigt, nicht erst nachträglich zu rechtfertigen.
Inhaltliche Begründung (unverändert): weder COVENANT M4 (Anker ausschließlich die drei
Rust-Dateien, COVENANT.md:38-41) noch CLAUDE.md „Frozen-baseline discipline" (Rust-
Panic-Metadaten, TS-Klassenkörper mit privaten Konstruktoren — beides nicht `topkRuntime`, eine
freistehende exportierte Funktion) verbieten die Abweichung inhaltlich; die Konvention ist für
`runtime.ts` ein selbst gewähltes Hausmuster, dessen Zweck (Verhalten für bestehende Aufrufer
erhalten) hier durch die Totalordnungs-Argumentation plus den Differentialtest (D11) BEWEISBAR
erfüllt wird — direkter als ein bloß angehängter, nie wieder benutzter Zwillingscode es täte.
`topkCompareValues` (runtime.ts:645-654) bleibt in JEDEM Ausgang UNVERÄNDERT.

- **D8 — Ausgang „reiner Heap":** `topkRuntime` (runtime.ts:669-696) wird AN ORT UND STELLE
  durch Kandidat B (D2) ersetzt (Vorab-Genehmigung s. o.). `topkCompareValues` bleibt
  unverändert und wird vom neuen Code wiederverwendet. `topkRuntime`s Dokumentationskommentar
  wird durch einen aktualisierten Kommentar ersetzt (neuer Algorithmus, Verweis auf diese Spec
  inkl. Version, Verweis auf das Test-Orakel). Siehe D11 für Test-Migration/Differentialtest/
  Mutant, gültig für diesen Ausgang.
- **D9 — Ausgang „Hybrid":** `topkRuntime`s Körper wird AN ORT UND STELLE (Vorab-Genehmigung
  s. o.) durch eine Verzweigung ersetzt: `const useHeap = k / n <= TOPK_HEAP_MAX_K_OVER_N_RATIO;
  return useHeap ? <Heap-Pfad> : <Sort-Pfad>;` (Guard für `n = 0`: die Division wird nur
  ausgeführt, wenn `n > 0` — bei `n = 0` ist nach der bestehenden Validierung ohnehin nur `k = 0`
  erreichbar, beide Pfade liefern dort identisch leere Arrays, die Verzweigung ist für diesen
  Fall irrelevant und darf auf einen der beiden Pfade fest verdrahtet sein).
  `TOPK_HEAP_MAX_K_OVER_N_RATIO` wird als benannte, exportierte Konstante in `runtime.ts`
  geführt (Namens-/Sichtbarkeits-Präzedenz: `THREADED_MATMUL_MIN_POOL_WORK`, exportiert aus
  `threaded.ts`), Wert = das in D6 Schritt 1 (geschlossene Form, ganzraster-sicher)
  bestimmte `thresholdRatio`. Der
  SORT-Zweig ist wörtlich der heutige Code (Kandidat A, unverändert übernommen) — für ihn ist
  Bit-Identität trivial (er IST der heutige Code). Der HEAP-Zweig ist Kandidat B (D2) — für ihn
  gilt dieselbe Bit-Identitäts-Pflicht wie im „reiner Heap"-Ausgang. Selbst wenn die gemessene
  Schwelle numerisch klein ausfällt, ist der Hybrid sinnvoll (D6 Schritt 2s Hinweis: kleine
  `ABSOLUTE_K`-Zellen bei großem `n` sind der realistische Anwendungsfall und liegen ohnehin weit
  unterhalb jeder plausiblen Schwelle). **Testbarkeit beider Zweige OHNE Signaturänderung:** da
  der Schwellenwert nach Phase 1/2 eine BEKANNTE Konstante ist, genügen gewöhnliche kleine
  `(n, k)`-Paare klar auf jeder Seite der Schwelle (z. B. `n=10, k=1` trifft garantiert den
  Heap-Zweig; `n=10, k=9` trifft garantiert den Sort-Zweig, sofern die gemessene Schwelle
  `< 0.9` liegt — der genaue Beleg-Fall wird nach der tatsächlichen Schwelle gewählt) — ANDERS
  als beim `THREADED_MATMUL_MIN_POOL_WORK`-Präzedenzfall wird KEIN Test-Override-Parameter
  gebraucht (dort war die Notwendigkeit, für einen Testfall echte große Matrizen aufzubauen,
  der Grund für `opts.minPoolWork`; hier ist ein `k/n`-Verhältnis frei wählbar, ohne dass
  absolute Größen teuer würden) — `topkRuntime`s Signatur bleibt dadurch unverändert, exakt wie
  in jedem anderen Ausgang. Siehe D11 für Test-Migration/Differentialtest/Mutant, ERWEITERT um
  eine Zweig-Abdeckungs-Pflicht (unten).
- **D10 — Ausgang „Status Quo":** `topkRuntime` wird NICHT verändert — kein Byte an
  `spike/src/runtime.ts` oder irgendeiner anderen Produktionsdatei. Ausgelöst mechanisch (D6
  Schritt 2, NEIN-Zweig), wenn in der ganzraster-sicheren Zone `k/n ≤ t*` KEINE Zelle
  mindestens 15 % schneller ist. Das deckt seit v5 drei früher getrennt behandelte Fälle
  mechanisch ab: der Heap gewinnt nirgends im Raster; `t*` ist `0` (keine nichtleere sichere
  Zone existiert); und der Gewinn liegt zwar im Raster, aber AUSSERHALB der sicheren Zone
  (der „hohle Hybrid", v4-Blocker). D8/D9/D11 entfallen vollständig (keine
  Implementierung, keine Orakel-Migration, kein Differentialtest, kein Mutant — es gibt nichts,
  gegen das zu differenzieren wäre). Das FOLLOWUPS.md-Item (D13) bleibt mit einem Nachtrag
  bestehen, der den gemessenen Befund festhält („Heap gemessen; kein Gewinn im dualen Sinn INNERHALB der sicheren Zone — sei es, weil er nirgends ≥15 % im
  gesamten Raster gewinnt, weil die sichere Zone leer war, oder weil sein Gewinn außerhalb der Zone lag — bewusst nicht
  verbaut, Zahlen in docs/op-topk-selection-ergebnisse.md"), damit ein künftiger Bearbeiter
  nicht denselben Weg erneut ohne Kenntnis dieser Messung beschreitet. Das Ergebnisse-Doc (D13)
  ist in diesem Ausgang die VOLLSTÄNDIGE Deliverable der Scheibe — die Messung selbst IST das
  Ergebnis, kein Nebenprodukt einer unterbliebenen Implementierung.
- **D11 — Test-Migration, Differentialtest, Mutant (gilt für „reiner Heap" UND „Hybrid", NICHT
  für „Status Quo"):**
  - Die HEUTIGE Implementierung von `topkRuntime` — Validierung, `order.sort(...)`,
    Ergebnis-Aufbau — wird WÖRTLICH (Copy, keine Umformulierung) als unexportierte, test-lokale
    Funktion in `spike/tests-runtime/argmax-topk.test.ts` (bereits in `test:core` registriert,
    package.json:55 — KEINE Änderung an den Explizitlisten nötig) verschoben, ebenso eine
    wörtliche Kopie von `topkCompareValues`. Namensvorschlag:
    `topkOracleCompareValues`/`topkOracleFullSort`. Ausschließlich Testinfrastruktur, kein
    Aufrufer aus Produktionscode. Bei „Hybrid" ist das Orakel WEITERHIN sinnvoll — nicht nur
    für den Heap-Zweig, sondern auch als Referenz, die beweist, dass der SORT-Zweig (der
    heutige Code, jetzt hinter einer Verzweigung) sich durch die Umstrukturierung selbst nicht
    verändert hat. Ihr Zweck bleibt: eine vom NEUEN Code UNABHÄNGIGE, exakt der alten Semantik
    entsprechende Referenz — bewusst KEINE „unabhängig geschriebene" Referenz wie `bruteTopk`
    (das bereits existiert, argmax-topk.test.ts:83-99, und unverändert weiterläuft).
  - Ein neuer Testblock in derselben Datei stellt `topkOracleFullSort` gegen den echten,
    importierten `topkRuntime` gegenüber: randomisierter Differentialtest über mindestens
    150-300 Fälle (Größenordnung der bestehenden `bruteTopk`-Probe, argmax-topk.test.ts:418-431),
    `n` über einen Bereich inkl. 0 und kleiner Werte, `k` gleichverteilt über `0..n`, unter
    Wiederverwendung von `genDataSpecial`/`SPECIAL_VALUES` (prng.ts) für
    NaN-/±0-/Infinity-/Subnormal-Injektion; feste Grenzfälle (`n=0,k=0`; `n=1,k=0`; `n=1,k=1`;
    ein Vektor aus ausschließlich NaN; ein Vektor aus ausschließlich demselben Wert; ein
    `+0`/`-0`-Gleichstand-Fixture; `k=0`; `k=n`; `k=n-1`). Bei „Hybrid": die Fallmenge muss
    NACHWEISLICH beide Zweige treffen (mindestens ein fester Fall klar unterhalb, einer klar
    oberhalb `TOPK_HEAP_MAX_K_OVER_N_RATIO`, mit einem Kommentar, der die gewählten `(n,k)`
    gegen die Konstante explizit einordnet — kein Verlass auf zufällige Fallabdeckung für diese
    Eigenschaft).
  - Pro Fall: `indices`-Arrays exakt gleich (`assert.deepStrictEqual`), `values`-Arrays
    bit-identisch (Wiederverwendung von `assertScalarBitIdentical`, argmax-topk.test.ts:56-61).
  - **JIT-Fallstrick (bereits einmal in dieser Datei dokumentiert und in der Vorab-Probe dieser
    Spec unabhängig reproduziert, siehe unten):** jeder Testfall, der eine NICHT-kanonische
    NaN-Payload exakt prüfen will, MUSS die Test-Daten über die bestehende `bitsAt`-artige
    direkte-`DataView`-über-den-Backing-Buffer-Technik aufbauen (argmax-topk.test.ts:391-394),
    NIEMALS über eine Array-Literal-Konstruktion. Mindestens EIN Test dieser Art ist Pflicht,
    exakt nach dem bestehenden Muster (argmax-topk.test.ts:396-416).
  - **Mutanten-Nachweisbarkeit (Pflicht, während Baustein A zu erbringen):** mindestens EIN
    gezielter Mutant muss den Differentialtest zum Kippen bringen. Kandidaten: (1)
    Vergleichsrichtung beim Ersetzen-Test im Heap umgekehrt; (2) Sift-Richtung vertauscht; (3)
    Off-by-one an der Heap-Kapazitätsgrenze; (4) Endsortierung nutzt einen anderen
    Comparator-Ausdruck als der laufende Wurzel-Vergleich; bei „Hybrid" zusätzlich (5) die
    Verzweigungsrichtung invertiert (`<=` zu `>` oder umgekehrt) — muss GENAU an Fällen nahe der
    Schwelle kippen, ein Beleg, dass die Zweig-Abdeckungs-Pflicht oben tatsächlich beide Pfade
    trifft. Nachweislich gefangen (roter Test, benannte Assertion), danach revertiert
    (Backup-Kopie-Verfahren, harte Arbeitsregel (1), CLAUDE.md:151-153 — niemals
    `git checkout`/`git restore` im Haupt-Working-Tree).

---

## D12 — Pins & Budget (Mess-Hausregeln gelten, CLAUDE.md:143-150)

- Erwarteter Ausgangswert laut CLAUDE.md (Stand HEAD 83148dd, VOR jeder Messung im frischen
  Worktree zu reproduzieren, nicht ungeprüft zu übernehmen): `check:diag` **199,877 @ 139
  Files** · `check:diag:stress` **106,398 @ 82** · `check:diag:browser` **2,142 @ 75**.
- **Phase 1 (immer, unabhängig vom Ausgang):** die neue Bench-Datei (D4) bewegt den Haupt-Pin
  um Order-Noise + geringen echten Inhalt (keine `NDArray`/generische Instanziierung — reine
  `(shape, data, k)`-Funktionen). Absolut-Gate für Phase 1 allein: **≤ +2,000** Instantiations
  **auf dem DEKOMPONIERTEN Anteil** — also auf der Differenz zwischen leerem Platzhalter und
  befüllter Datei, NICHT auf der Gesamtverschiebung des Pins.

  **v6.1-Nachschärfung (Owner-Entscheidung 2026-07-22, nach dem realen Phase-1-Lauf):** Die
  ursprüngliche Formulierung ließ offen, worauf sich die Zahl bezieht, und orientierte sich an
  einer Order-Noise-Spanne, die inzwischen widerlegt ist. Gemessen: Die eine neue Datei bewegte
  den Pin um **+6.924 gesamt**, davon **+6.611 reines Order-Noise** (eine LEERE `export
  {}`-Datei, zweifach reproduziert) und nur **+313 echte Typkosten**. Die Gesamtverschiebung
  hätte das Gate formal gerissen, obwohl der inhaltliche Anteil weit darunter liegt. Da
  Order-Noise per Definition keine Typkosten sind und das empty-then-fill-Protokoll genau
  dafür existiert, gatet die Regel den dekomponierten Anteil. Die in CLAUDE.md dokumentierte
  Spanne wurde entsprechend von „±≈2,000" auf „±≈7,000" korrigiert — ein Befund über die
  Messinfrastruktur, der jede künftige dateihinzufügende Scheibe betrifft. `check:diag:stress`/
  `:browser`/`bench:editor` (alle acht Workload-Pins,
  spike/bench-dx/editor-latency.ts:813-822 — `w1:27904, w2:29713, w3:60853, w4:28067,
  w5:33358, w6:34528, w7:27076, w8:34943`) sollten EXAKT unverändert bleiben (Δ0) — Phase 1
  berührt `NDArray`s öffentliche Fläche nicht.
- **Phase 2, Ausgang „reiner Heap" oder „Hybrid" (bedingt):** zusätzliches Absolut-Gate
  **≤ +3,000** Instantiations für den `runtime.ts`-Körpertausch plus die Testdatei-Erweiterung
  (D11) — beide ohne neue Typ-Deklarationen (nur Funktionskörper-/Testinhalt), daher enger als
  die W-Scheiben-Gates (+6,000 bis +12,000), die echte neue generische Typ-Maschinerie
  einführten. `check:diag:stress`/`:browser`/`bench:editor` sollten AUCH hier EXAKT unverändert
  bleiben (Δ0) — weder `topkRuntime`s noch `NDArray.topk`s Signatur ändert sich in irgendeinem
  Ausgang. Ein Abweichen von Δ0 (Phase 1 ODER Phase 2) ist eine ECHTE, zu meldende Überraschung,
  kein erwartbares Klassen-Surface-Ripple wie bei den W-Scheiben.
- **Phase 2, Ausgang „Status Quo":** kein zusätzliches Gate — es gibt keine Implementierung, die
  ein Budget verbrauchen könnte.
- Gestufte Messung wie bei den W-Scheiben: Baseline im frischen Worktree, dann Δ je Schritt
  (Bench-Datei leer/gefüllt, ggf. `runtime.ts`-Tausch, ggf. Testdatei-Erweiterung).
- `check:freeze`-Hash muss in JEDEM Ausgang byte-identisch bleiben (kein Rust berührt).

## D13 — Doc-Platzierung / FOLLOWUPS-Schließung

Nach Abschluss, AUSGANGSABHÄNGIG: (a) FOLLOWUPS.md Zeile 73 (der bestehende, offene
`[ ]`-Eintrag „argmax/topk auf WNDArray/Threaded nachziehen") bekommt einen Nachtrag mit dem
GEMESSENEN Befund und dem gewählten Ausgang — bei „reiner Heap"/„Hybrid": ein künftiger Kernel
sollte diesen Algorithmus spiegeln; bei „Status Quo": der Nachtrag hält fest, dass ein Heap
GEMESSEN und bewusst NICHT verbaut wurde (kein Gewinn im dualen Sinn innerhalb der sicheren Zone — nirgends ≥15 % im Raster, leere Zone, oder
der größte ganzraster-sichere Schwellenwert war `0`), mit Verweis auf die Zahlen, damit niemand
denselben Weg blind wiederholt. (b) `docs/roadmap.md`, Abschnitt „Post-Roadmap"
(docs/roadmap.md:246-276), neuer kurzer Absatz mit dem Ausgang. (c) CLAUDE.md „Status"/„Aktuelle
Pins & Gates" nach der Hausregel (CLAUDE.md:217-221) — nur Einzeiler + IST-Zahlen. Volles
Narrativ IN JEDEM Ausgang (auch „Status Quo") in `docs/op-topk-selection-ergebnisse.md` mit
Post-Verification-Addendum.

## D14 — Sprache

Code/Kommentare/Tests/Commit-Message: Englisch (Hard Constraint, CLAUDE.md:17). Spec +
Ergebnisse-Doc: Deutsch. „ca." statt der Tilde (kein `<del>`-GFM-Risiko).

## Akzeptanzkriterien

- **T1 (Phase 1, immer):** Bench-Datei liefert vollständige, vorschriftsmäßige Messungen
  (Raster D3 vollständig, beide Reihenfolgen, Warmup-Protokoll, Bit-Identitäts-Gate, Min-Max
  UND Median berichtet, mindestens ein Zweitlauf) und druckt die mechanische Verdikt-Zeile
  (D4). `pnpm bench:topk` läuft fehlerfrei durch.
- **T2 (Phase 2, immer):** Die Entscheidungsregel (D6) wird NACHVOLLZIEHBAR auf die
  tatsächlichen Zahlen angewendet — im Ergebnisse-Doc wird die Regelanwendung Schritt für
  Schritt gezeigt (nicht nur das Endergebnis genannt), inkl. der Kandidatenprüfung der
  geschlossenen Form (D6 Schritt 1: welche `t` sicher waren, welche nicht und an welcher Zelle
  sie scheiterten), der Zellen, die als Verletzung bzw. Gewinn im DUALEN Sinn zählten (und
  welche nur eines der beiden Kriterien erfüllten — das ist die interessante Information über
  die Rausch-Nähe der Entscheidung), sowie der Verdikte der EINZELNEN Läufe neben dem
  pessimistisch zusammengezogenen Verdikt. **v6-Bereinigung:** Die frühere Berichtspflicht für
  `c(n)`-Werte und deren Streuung entfällt — `c(n)` (der per-`n`-Kreuzungspunkt) war eine
  Konstruktion der v3-Regel und existiert seit v4 nicht mehr; die Berichtspflicht hatte sie
  als hängende Referenz überlebt (Befund der Frontier-Zweitmeinung).
- **T3 (nur „reiner Heap"/„Hybrid"):** Konstruktive Totalordnungs-Argumentation im
  Ergebnisse-Doc erneut nachvollzogen UND empirisch durch den D11-Differentialtest bestätigt.
- **T4 (nur „reiner Heap"/„Hybrid"):** Pin-Protokoll (D12) vollständig, Gates eingehalten oder
  Abweichung dem Owner vorgelegt; Δ0-Erwartung für stress/browser/bench:editor explizit
  verifiziert.
- **T5 (nur „reiner Heap"/„Hybrid"):** Differentialtest deckt jede in D11 genannte Kategorie ab;
  mindestens ein Mutant nachweislich gefangen, danach revertiert mit `git status`-Beweis. Bei
  „Hybrid" zusätzlich: Zweig-Abdeckungs-Pflicht erfüllt (mindestens ein fester Fall je Zweig,
  explizit gegen die Konstante eingeordnet).
- **T6 (nur „reiner Heap"/„Hybrid"):** Datei-Disziplin: `topkRuntime`s Körper UND
  Dokumentationskommentar ersetzt (In-Place, mit der Vorab-Genehmigung vor D8/D9 begründet,
  Verweis auf diese Spec inkl. Version); `topkCompareValues` byte-unverändert;
  `argmaxRuntime`/`beatsMax`/`ndarray.ts`/`vector.ts`/`reduce.ts`/`dim.ts` byte-unverändert;
  keine neue Datei unter `spike/src`; `test-scripts-guard.test.ts` bleibt grün ohne
  Explizitlisten-Änderung.
- **T7 (nur „Status Quo"):** Ergebnisse-Doc dokumentiert den vollständigen Messbefund, die
  Regelanwendung, und die FOLLOWUPS-Schließung (D13) — ohne dass irgendein Produktionscode
  angefasst wurde; `git status` zeigt außerhalb von Docs keine Änderung.
- **T8 (immer):** Doc-Platzierung (D13) vollständig für den TATSÄCHLICH gewählten Ausgang.

## Nicht-Ziele

Kein WASM-Kernel für `topk` (M1 bindet weiterhin nicht, in keinem Ausgang), keine
`WNDArray`/Threaded-Parität (FOLLOWUPS, unverändert offen), keine Erweiterung des Fixes auf
`argmax` (kein vergleichbarer Defekt dort), kein `argsort`, keine Änderung an `topk`s
öffentlicher Signatur/Fehlerverhalten/Guard-Typebene in irgendeinem Ausgang, keine Behauptung
eines gemessenen Nutzerbedarfs, **keine Vorwegnahme des Algorithmus vor der Messung** (der
zentrale Unterschied zu v1) — insbesondere KEINE Implementierung von D8 oder D9, bevor Phase 1
tatsächlich durchgeführt und Phase 2 tatsächlich mechanisch angewendet wurde. Keine
Ermessens-Korrektur der Entscheidungsregel NACH Kenntnis der Zahlen („die Regel hätte eigentlich
X sagen sollen") — eine Regeländerung nach der Messung ist nur als bewusste, dem Owner
vorgelegte Spec-Revision mit Versionserhöhung zulässig, nie als stille Neuinterpretation.

## Gate-Block / Definition of Done

**Phase 1 (immer):** `pnpm check` (Dreier-Verbund) · `check:diag`(+stress/browser, Pin-Protokoll
D12) · `pnpm bench:topk` (neu, liefert Rasterdaten + mechanisches Verdikt) · `graph-a-lama query
lint` · GFM-Gate auf allen neuen/geänderten `.md` (keine Tilden/Strikethrough, harte
Arbeitsregel (4), CLAUDE.md:158-159).

**Phase 2, zusätzlich NUR bei „reiner Heap"/„Hybrid":** `test:core` (inkl. erweiterter
`argmax-topk.test.ts`, alle bisherigen 30 Assertions weiterhin grün plus die neuen Differential-/
Mutanten-Tests) · `test:resident` (unberührt grün) · `cargo test` (unberührt) · `check:freeze`
(Hash-Pin byte-identisch) · `bench:editor` (alle acht Pins Δ0) · `pnpm test:example` (unberührt).

**Phase 2, bei „Status Quo":** keine zusätzlichen Gates — Phase-1-Gates allein sind die
Definition of Done.

## Verify-Plan (Stufe 3)

**Baustein 0 hat für diese Spec bereits stattgefunden** (gegen v2, adversarial) und drei
Blocker in D6 plus zwei weitere Befunde gefunden — vollständig dokumentiert im Addendum-Abschnitt
unten, in v3 aufgelöst. Die folgende Liste dokumentiert den ursprünglich erteilten Auftrag als
Referenz für künftige Runden (z. B. eine leichte Re-Prüfung von v3s neuem D6, falls gewünscht) —
sie instanziiert denselben Auftragstyp erneut, nicht als offene Aufgabe für DIESE Version:

- **Messdesign (D3):** ist das Raster tatsächlich dicht genug, um einen Crossover zuverlässig
  zu lokalisieren, ohne ihn zwischen zwei Rasterpunkten zu verpassen? Ist die
  Beide-Reihenfolgen-Konstruktion tatsächlich geeignet, Aufwärm-Bias aufzudecken (empirische
  Probe: an mindestens einer Zelle absichtlich NUR eine Reihenfolge fahren und zeigen, dass das
  Ergebnis empfindlich auf die Reihenfolge reagiert, dann mit beiden Reihenfolgen den
  Unterschied auflösen)? Hält das `measureRange`-Warmup-Protokoll für die TEUERSTEN Zellen
  (`n=1_000_000`, `k` nahe `n`, ca. 280 ms/Aufruf laut Vorab-Probe) tatsächlich, oder braucht es
  hier doch einen Mindest-Call-Boden, den D3 bewusst nicht eingeführt hat?
- **Entscheidungsregel (D6):** ist sie WIRKLICH mechanisch anwendbar, ohne eine Lücke, in der
  ein Mensch improvisieren müsste? Testet Baustein 0 die Regel an mindestens drei ERFUNDENEN
  Zahlenreihen (eine, die klar auf „reiner Heap" führt, eine, die klar auf „Hybrid" führt, eine
  dritte, die auf „Status Quo" führt, weil der Heap nirgends 15 % gewinnt) durch — bricht die
  Regel an einer Kombination, die diese Spec nicht bedacht hat (z. B. `ratio` exakt `1.15`,
  `c(n)` für ein `n` undefiniert, eine `ABSOLUTE_K`-Zelle, die die Ganzraster-Validierung zum
  vollständigen Erschöpfen aller Fraktionen zwingt)?
- **Toleranzzahl (15 %, 5 %/50 µs):** hält die Anlehnung an den Kern-06-Präzedenzfall
  (docs/kern-06-ergebnisse.md) einer wörtlichen Nachprüfung stand — sind die zitierten Zahlen
  (1,16 als „wash"-Obergrenze) tatsächlich das, was das Dokument sagt, und ist die Analogie
  (Thread-Dispatch-Rauschen vs. JS-JIT-Selektionskosten) als GRÖSSENORDNUNG, nicht als
  übertragbare Messung, klar genug ausgewiesen?
- **In-Place-Abweichungs-Begründung (D8/D9):** RESOLVED in v3 — Baustein-0-Befund 5 (Addendum)
  fand die fehlende Vorab-Genehmigung gegen `docs/verify-runde-template.md:72`; v3 trägt jetzt
  einen expliziten, gemeinsamen Vorab-Genehmigungs-Absatz vor D8/D9. Eine künftige Runde sollte
  trotzdem gegenlesen, ob die Formulierung dort für einen Baustein-A-Verifier tatsächlich
  ausreicht, den Edit nicht zu flaggen.
- **Bench-Design (D4):** „keine WASM-Abhängigkeit" gegen das tatsächliche `bench-core/`-Muster
  prüfen; ist die automatische Stage-B-Auslösung (D7) im selben Skriptlauf technisch sauber
  umsetzbar (kein zirkulärer Bau-vor-Messen-Zwang)?

Nach Owner-Freigabe: **Phase 1 wird ausgeführt** (kann delegiert werden, ist aber selbst kein
Verify-Gegenstand im A/B/C-Sinn, sondern die Messung selbst — ihre Qualität wird über die
D3-Kriterien geprüft, nicht über einen Diff). **Phase 2** wird auf die Zahlen angewendet. Je
nach Ausgang:
- „Status Quo": ein einzelner `brainroute:verify`-Durchlauf genügt (prüft Messqualität +
  Regelanwendung + Doc, kein Diff an Produktionscode).
- „reiner Heap"/„Hybrid": voller Katalog A (Spec-Konformität, alle Gates frisch, eigener Mutant)
  + B (adversarial: Grenzfälle jenseits der Spec — `n` extrem klein/1, dichte Wert-Cluster,
  wiederholte `topk`-Aufrufe auf transponierten/geslicten Empfängern analog dem bestehenden
  Materialisierungs-Test argmax-topk.test.ts:437-448, Mess-Randbedingungen der Bench) + C
  (covenant-verify: M1-Anker-Berührung ohne M1-Bindung korrekt eingeordnet? M2/M3/M4/M5/Z1/Z2
  sauber unberührt?) parallel; Aufträge aus docs/verify-runde-template.md.

Ergebnisse-Doc mit Post-Verification-Addendum in JEDEM Ausgang, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-21)

Verifier: `brainroute:deep`, adversarial gegen v2 dieser Spec. **Fundament BESTÄTIGT**, härter
geprüft als die eigene Vorab-Probe dieser Spec: 269.984 paarweise Antisymmetrie-Checks der
Totalordnung, 1.500 Shuffle-Invarianz-Trials (dieselbe Top-k-Menge unabhängig von der
Scan-Reihenfolge), mehrere distinkte NaN-Payloads geprüft, 3.031 Fuzz-Fälle des
Heap-Kandidaten gegen das Orakel — NULL Abweichungen. Das Messdesign (D3) hält ebenfalls.

**D6 dagegen war DREIFACH gebrochen — alle drei Blocker hängen am selben Übergang
(Zulässigkeitsprüfung → Crossover-Bestimmung), deshalb für v3 komplett neu geschrieben statt
gepatcht:**

1. **BLOCKER: Die Pure-Heap-Zulässigkeitsprüfung ignorierte strukturell genau die Größen, an
   denen der Gewinn erwartet wird.** v2s Schritt 1 prüfte nur `n ≤ 10.000`; bestand die
   Prüfung, ersetzte D8 UNBEDINGT für ALLE `n` — Schritt 2/3 (die einzige Stelle, die
   `n=100.000/1.000.000` ansah) lief dann nie. Empirisch belegt mit einem synthetischen
   Raster, das auf dem realistischen Spektrum sauber blieb und bei `k` nahe `n` auf
   Verhältnis 1,58 stieg: Verdikt „PURE HEAP", Verlust nie geprüft. **Aufgelöst:** Schritt 1
   prüft jetzt das GESAMTE Raster (alle fünf `n`), weil eine unbedingte, verzweigungsfreie
   Ersetzung bei jeder Größe gilt und ihre Zulässigkeit deshalb auch bei jeder gemessenen
   Größe geprüft werden muss.
2. **BLOCKER: Die Nicht-Monotonie-Klausel widersprach sich selbst.** Sie schrieb „nimm den
   LETZTEN Wechsel" vor und behauptete im selben Satz, die spätere/heap-freundlichere Lesart
   werde bewusst NICHT gewählt — die vorgeschriebene Aktion WAR aber genau diese Lesart (ein
   späterer `c(n)` erlaubt MEHR `k/n`-Raum für den Heap, nicht weniger). Zusätzlich
   kollidierte das mit der eigenen Definition von „konservativ" neun Zeilen weiter unten
   (dort = früher/sort-freundlicher). **Aufgelöst:** `c(n)` ist jetzt explizit der ERSTE
   Übergang (kleinste Fraktion mit `ratio > 1.15`) — die sichere Lesart, weil eine früh
   gemessene schlechte Zelle nie in die als heap-sicher deklarierte Zone unterhalb der
   Schwelle gerät.
3. **BLOCKER: `c(n)` konnte undefiniert bleiben, ohne dass der weitere Ablauf dafür
   spezifiziert war.** Fiel Schritt 1 wegen EINES schlechten `n` in den Sonst-Zweig, während
   ein anderes `n` die Schwelle nie überschritt, war `max/min` nicht berechenbar und die Regel
   brach ab. Empirisch reproduziert. **Aufgelöst:** die gesamte Stabilitäts-/`max/min`-
   Gating-Logik entfällt ERSATZLOS (s. „tragende Vereinfachung" in D6) — ein `n` ohne
   definiertes `c(n)` wird protokolliert und aus der Schwellenbildung ausgeschlossen, mit
   einem expliziten Konsistenz-Check (mindestens ein `n` muss ein definiertes `c(n)` liefern;
   Verletzung bricht ab und meldet, statt zu entscheiden).

**Gemeinsame Fehlursache:** v2 verwechselte „eine `n`-abhängige Schwellen-FUNKTION" (tatsächlich
unerwünschte Komplexität) mit „eine einzige, konservative Schwellen-KONSTANTE" (einfach und
sicher). Weil ein zu NIEDRIGER Schwellenwert nie schlechter als der heutige Zustand ist
(oberhalb der Schwelle läuft exakt das heutige Verfahren) und ein zu HOHER Wert gefährlich ist,
ist ein MINIMUM über alle gemessenen `n` immer sicher — unabhängig davon, wie sehr die
Kreuzungspunkte streuen. Die Streuung wird jetzt berichtet, nicht gegated.

**Weitere Befunde:**

4. **MAJOR: Rasterlücke 0,7-1,0.** Die größte Schrittweite der gesamten `RATIO_FRACTIONS`-Reihe
   lag genau im laut Vorab-Probe steilsten Bereich — ohne Schließung hätte eine Schwelle einen
   nie vermessenen Bereich als heap-sicher ausweisen können. **Aufgelöst:** `0.8`, `0.85`,
   `0.9`, `0.95` ergänzt (D3).
5. **MAJOR: die In-Place-Abweichung hatte keine ausdrückliche Vorab-Genehmigung IN der Spec.**
   Das COVENANT/CLAUDE.md-Argument (D8/D9) hält bei wörtlicher Prüfung, aber eine DRITTE
   einschlägige Quelle blieb unberücksichtigt: `docs/verify-runde-template.md:72` nennt
   `runtime.ts` EXPLIZIT als append-only-Datei, die Baustein A per „Disziplin-Prüfung am Diff"
   mechanisch kontrolliert — und aus genau diesem Template wird der spätere Baustein-A-Auftrag
   instanziiert. Ohne einen Vermerk hätte ein Baustein-A-Verifier den Edit zu Recht als
   Verstoß geflaggt. **Aufgelöst:** die Abweichung ist jetzt als disclosed UND vorab bestätigt
   für `topkRuntime` ausgeschrieben (neuer gemeinsamer Absatz vor D8/D9), unter ausdrücklicher
   Berufung auf die vom Template selbst vorgesehene Ausnahme („bestätigte Abweichung").
6. **Nit:** offene Frage zur Stage-B-Automatisierung — keine technische Hürde gefunden, die
   automatische Auslösung gilt als robuster als ein manueller Zwischenschritt. **Aufgelöst:**
   offene Frage entfernt, D7 bleibt bei der automatischen Variante.
7. **Nit:** die 15-%-Herleitung war sprachlich zu stark („hergeleitet aus") — das
   Kern-06-Band entstand aus Thread-Dispatch-Rauschen, ist also eine Größenordnungs-Analogie,
   keine übertragbare Messung. **Aufgelöst:** „in Anlehnung an" statt „hergeleitet aus"
   überall im Dokument.

**Eigener Zusatz dieser Neufassung (über den Baustein-0-Auftrag hinaus, beim Umbau selbst
gefunden — nicht Teil der ursprünglichen vier Owner-Schritte, daher hier gesondert
ausgewiesen):** die Schwelle wird ausschließlich aus `RATIO_FRACTIONS`-Zellen gebildet (D6
Schritt 3), aber Schritt 1 prüft das GESAMTE Raster inklusive `ABSOLUTE_K`. Eine Regression, die
NUR in einer `ABSOLUTE_K`-Zelle sichtbar wird (plausibel bei großem `n`, wo `k/n` für kleine
feste `k` UNTERHALB der kleinsten `RATIO_FRACTIONS`-Fraktion liegt), würde Schritt 1 korrekt zum
Scheitern bringen, aber von der reinen `RATIO_FRACTIONS`-Schwellenbildung nicht erfasst — der
resultierende Hybrid könnte diese Zelle weiterhin unbemerkt dem Heap zuweisen, ausgerechnet in
der Zellklasse, die D6 Schritt 2 selbst als „der eigentliche realistische Anwendungsfall"
benennt. D6 Schritt 4 bekommt deshalb eine nachträgliche Ganzraster-Validierung mit
Ratchet-down-Mechanismus (Details dort, inkl. eines definierten Terminierungsfalls: Erschöpfen
aller Fraktionen ohne Erfolg → „Status Quo, mit offenem Befund"). Eine kleine, eigenständig
begründete Erweiterung, dem Owner hiermit zur Kenntnis vorgelegt — keine stille Änderung der
vier vorgegebenen Schritte.

## Empirische Vorab-Probe (Scratch, informell — dient NUR der Grid-Kalibrierung, ersetzt Phase 1 NICHT)

Durchgeführt unter
`/private/tmp/claude-501/-Users-marvinmuegge-Documents-CODE-numtype/f37d652e-e07c-4dcd-8d62-e4aeac2dfd0b/scratchpad/topkspec/probe.mjs`
(reines Node-Skript, KEIN Teil dieses Repos, Haupt-Working-Tree unberührt). Zweck:
Feasibility der Kandidaten-Konstruktion (D2) UND sinnvolle Rasterdichte (D3) vorab prüfen —
**ausdrücklich KEIN Ersatz für Phase 1** (informell, ein Prozess, keine Beide-Reihenfolgen-
Disziplin, kein `measureRange`-Warmup-Protokoll für die Zeitmessung selbst). Die in v1 aus
dieser Probe gezogene ALGORITHMUS-Entscheidung wurde mit v2 zurückgenommen — die Probe bleibt
als Kalibrierungsgrundlage stehen, nicht als Beleg.

1. **Äquivalenz-Fuzzing:** eine wörtliche Kopie der heutigen `topkRuntime`/`topkCompareValues`
   (Oracle) gegen eine Kandidaten-Heap-Implementierung exakt nach D2 dieser Spec, über 20.000
   randomisierte Fälle plus feste Grenzfälle plus zwei größere Fälle — **Ergebnis: 20.017
   Fälle, alle `indices`/`values` bit-identisch, keine einzige Abweichung.** Dieser Teil der
   Probe bleibt uneingeschränkt gültig (er belegt NUR Korrektheit, keine Performance) und
   stützt weiterhin die Aussage „ein korrekter Heap-Kandidat ist bit-identisch zur
   Vollsortierung" unabhängig vom späteren Ausgang — von Baustein 0 unabhängig und HÄRTER
   bestätigt (s. Addendum oben).
2. **Nicht-kanonische-NaN-Payload-Probe:** bestätigt unabhängig, dass Array-Literal-Konstruktion
   (`new Float64Array([...])`) für nicht-kanonische NaN-Payloads NACH hinreichend vielen
   vorherigen Float64Array-Operationen im selben Prozess JIT-instabil ist (nach ca. 50.000
   vorangehenden Konstruktionen normalisierte `new Float64Array([1, weirdNaN, 3, 2])` silent
   auf die kanonische Payload; direktes `DataView.setBigUint64` auf den bereits allozierten
   Backing-Buffer blieb über mehrere Wiederholungen deterministisch korrekt) — begründet die
   verbindliche `bitsAt`-Vorgabe in D11.
3. **Informelle Timing-Sondierung (NICHT die Phase-1-Messung — diente ausschließlich der
   Grid-Dichte-Wahl in D3, insbesondere der erhöhten `RATIO_FRACTIONS`-Dichte zwischen 0,1 und
   0,5):** bei `n=1e6` lag die alte Vollsortierung bei ca. 284-286 ms unabhängig von `k`; die
   Heap-Kandidatin lag bei `k∈{1,10,100}` bei ca. 4,5-4,8 ms, bei `k=n/2` bei ca. 303 ms und bei
   `k=n` bei ca. 451 ms. Bei `n=1e5` dasselbe Muster in kleinerem Maßstab. **Diese Zahlen sind
   NICHT die Phase-1-Messung** (andere Disziplin, kein Beide-Reihenfolgen-Protokoll, kein
   `measureRange`-Warmup) — sie dienen ausschließlich der Rasterkalibrierung. Insbesondere darf
   aus ihnen KEIN Vorgriff auf D6s Ausgang gelesen werden.

## Offene Fragen (für Baustein 0 / den Owner)

Frühere offene Fragen sind durch die v2→v3-Entwicklung erledigt: der `k≈n`-Akzeptanz-Frage
(v1) beantwortet jetzt D6 selbst mechanisch; die Heap-Knoten-Repräsentation (v1/v2) bleibt eine
dem Bench-Autor freigestellte Detailfrage, nicht mehr spec-relevant; die Rasterlücke nahe
`k/n=1` ist geschlossen (D3, Baustein-0-Befund 4); die Stage-B-Automatisierung ist entschieden
(D7, Baustein-0-Befund 6); die In-Place-Abweichungs-Begründung trägt jetzt eine ausdrückliche
Vorab-Genehmigung (Baustein-0-Befund 5, s. Addendum). Es verbleibt EINE genuin offene Frage:

1. **Absolut-Gate-Höhe (D12, ≤ +2,000 Phase 1 / ≤ +3,000 Phase 2):** eine begründete Schätzung,
   keine Messung — sollte an einer echten Worktree-Baseline plausibilisiert werden, bevor
   Phase 1 ausgeführt wird, und bei Bedarf korrigiert (wie bei W1s Baustein-0-Praxis). Diese
   Frage konnte Baustein 0s v2-Runde nicht abschließend beantworten, da sie erst gegen den
   TATSÄCHLICHEN `runtime.ts`-Diff messbar ist, den es vor Phase 2s Ausgang noch nicht gibt.

## Änderungslog

- **v3 (2026-07-21):** Baustein 0 (adversarial gegen v2) bestätigte das Fundament (Totalordnung/
  Bit-Identität — 269.984 Antisymmetrie-Checks, 1.500 Shuffle-Trials, 3.031 Heap-Fuzz-Fälle,
  null Abweichungen — sowie das Messdesign D3), fand aber D6 DREIFACH gebrochen: (1) die
  Pure-Heap-Zulässigkeitsprüfung testete nur das realistische Spektrum, ersetzte aber
  unbedingt für ALLE `n` — `n=100.000/1.000.000` liefen nie durch die einzige Prüfung, die sie
  angesehen hätte (empirisch: synthetisches Raster, Verhältnis 1,58 bei `k` nahe `n`, dennoch
  „PURE HEAP"); (2) die Nicht-Monotonie-Klausel widersprach sich selbst (schrieb den letzten
  Übergang vor, behauptete im selben Satz das Gegenteil gewählt zu haben, und kollidierte mit
  der eigenen Konservativ-Definition); (3) `c(n)` konnte für alle `n` gleichzeitig undefiniert
  bleiben, ohne dass die Regel das auffing (`max/min` bricht dann ab). D6 wurde deshalb
  KOMPLETT NEU geschrieben statt gepatcht, um eine gemeinsame Fehlursache zu beheben (v2
  verwechselte eine `n`-abhängige Schwellen-FUNKTION mit einer einzigen, konservativen
  Schwellen-KONSTANTE — Letztere ist wegen der asymmetrischen Fehlerfolgen, ein zu niedriger
  Wert kostet nur entgangene Beschleunigung, ein zu hoher ist eine echte Regression, immer sicher
  als reines Minimum über alle `n`, unabhängig von deren Streuung; die Streuung wird jetzt
  berichtet statt gegated). Neue Regel: Schritt 1 (reiner Heap) prüft jetzt das GESAMTE Raster,
  nicht nur das realistische Spektrum; ein neuer, vorgezogener Schritt 2 (Meaningful-Win/Status-
  Quo, mechanisch: kein Cell mit ≥15 % Gewinn irgendwo) ersetzt die alte Instabilitätsklausel als
  Auslöser für „Status Quo"; Schritt 3s Crossover `c(n)` nimmt jetzt den ERSTEN statt den letzten
  Übergang; Schritt 4 (Schwelle = Minimum über definierte `c(n)`) bekommt eine neue, eigenständig
  begründete Ganzraster-Validierung mit Ratchet-down-Mechanismus (schließt eine beim Umbau selbst
  gefundene Lücke: die Schwelle wird nur aus `RATIO_FRACTIONS`-Zellen gebildet, aber Schritt 1
  prüft auch `ABSOLUTE_K` — eine isolierte `ABSOLUTE_K`-Regression würde sonst unbemerkt in der
  Hybrid-Zone landen). Zwei weitere Befunde eingearbeitet: die `RATIO_FRACTIONS`-Rasterlücke
  zwischen 0,7 und 1,0 (der größte Schritt der Reihe, im laut Vorab-Probe steilsten Bereich) ist
  mit `0.8`/`0.85`/`0.9`/`0.95` geschlossen (D3); die In-Place-Abweichung (D8/D9) trägt jetzt
  einen ausdrücklichen, gemeinsamen Vorab-Genehmigungs-Absatz unter Berufung auf
  `docs/verify-runde-template.md:72`, das `runtime.ts` explizit als von Baustein A mechanisch
  geprüfte append-only-Datei nennt — eine dritte, zuvor übersehene Quelle. Kleinere Punkte:
  offene Frage zur Stage-B-Automatisierung zugunsten der automatischen Variante aufgelöst (D7,
  keine technische Hürde gefunden); die 15-%-Schwellen-Herleitung sprachlich abgeschwächt
  („in Anlehnung an" statt „hergeleitet aus" — das Kern-06-Band ist eine Größenordnungs-Analogie
  aus Thread-Dispatch-Rauschen, keine übertragbare Messung). Neuer Abschnitt „Adversariale
  Spec-Verifikation (Addendum, Baustein 0)" dokumentiert alle Befunde vollständig. Offene Fragen
  auf eine einzige verbleibende reduziert (Absolut-Gate-Höhe).
- **v2 (2026-07-21):** Owner-Entscheidung „erst messen, dann entscheiden" (Auslöser: der in
  v1s Vorab-Probe berichtete `k≈n`-Befund war eine informelle Sondage, kein kalibriertes
  Messergebnis — die Hausregel verbietet geratene Schwellen, Präzedenzfall
  `THREADED_MATMUL_MIN_POOL_WORK`/`bench:crossover`). Struktur-Änderungen: (1) die Scheibe
  zerfällt in Phase 1 (bindende Messung, D1-D4: konkretes `n`/`k`-Raster, Beide-Reihenfolgen-
  Pflicht gegen Aufwärm-Bias, `measureRange`-Warmup-Protokoll aus `threaded-crossover.ts`
  übernommen) und Phase 2 (bindende, VORAB festgelegte Entscheidungsregel, D5-D7); (2) v1s
  D2/D3 (Heap als gesetzte Implementierungsentscheidung) sind zu Kandidaten-BESCHREIBUNGEN
  degradiert (Teil von Phase-1-D2), keine Entscheidung mehr; (3) Hybrid ist jetzt ein dritter,
  bedingt vermessener Kandidat (D7) statt einer ausgeschlossenen Nicht-Ziel-Option; (4) drei
  mögliche Ausgänge — „reiner Heap" (D8), „Hybrid" (D9), „Status Quo/wie es ist" (D10) — sind
  alle vollständig ausformuliert; „Status Quo" ist explizit als vollwertiges, owner-mandiertes
  Ergebnis benannt. Owner-Entscheidungen 1 (in-place) und 2 (Orakel im Test) bleiben inhaltlich
  unverändert gültig, jetzt an D8/D9/D11 verankert. Bit-Identität und die
  Totalordnungs-Argumentation bleiben das unveränderte Fundament für JEDEN Kandidaten. v2s
  eigene Entscheidungsregel (D6) erwies sich in der Baustein-0-Runde als dreifach gebrochen —
  s. v3-Eintrag oben.
- **v6 (2026-07-22):** Frontier-Zweitmeinung (vom Owner angefordert, nachdem die Regel dreimal
  hintereinander an unabhängigen Verifiern gescheitert war; sie prüfte ausdrücklich eine
  Fassung, die vom Orchestrator selbst stammte und noch von niemandem begutachtet war). Sie
  bestätigte v5 als erstmals **formal vollständig und deterministisch** — 0 Abbrüche, 0
  Selbst-Divergenzen über 5.000 Raster, alle Gegenbeispiele der drei Vorrunden korrekt gelöst,
  Monotonie unabhängig nachgewiesen. Drei neue Befunde, alle in v6 aufgelöst:
  **(1) BLOCKER — zwei Läufe, kein verdikt-tragender benannt.** D3 mandatiert einen zweiten
  vollständigen Lauf, D4 lässt DAS Skript die Verdikt-Zeile drucken — zwei Läufe drucken also
  zwei Verdikte, und weder D6 noch der Testplan sagten, welches zählt. Monte-Carlo mit
  plausibler Streuung: **39,6 % der Lauf-Paare divergierten** (meist ein anderes `t*`,
  vereinzelt Status-Quo-Flips). Genau die verbotene Ermessensentscheidung nach der Messung.
  v6-Fix: Das Verdikt wird auf dem PESSIMISTISCHEN Zusammenzug beider Läufe berechnet (je Zelle
  das ungünstigere Ergebnis); Divergenz der Einzel-Verdikte ist ein berichtspflichtiger Befund,
  kein Entscheidungsproblem.
  **(2) MAJOR — rein relative Toleranz macht die Regel rausch-fragil.** Für jedes `t* > 0`
  mussten alle ca. 22 Zellen am Zonenboden unter 1,15 bleiben; die kleinsten davon liegen bei
  `n = 100` im Sub-Mikrosekunden-Bereich. Gegenbeispiel: Faktor-50-Gewinne an allen großen
  Zellen, EINE Zelle bei `1,16` statt `1,15`, absolutes Delta unter einer Mikrosekunde →
  Verdikt „nichts tun". Verschärfend: Das `ratio` ist per Konstruktion das Maximum über beide
  Reihenfolgen (einseitig nach oben verzerrt), und D3 erwartet selbst Spreads bis 10 % als
  normal — die Hälfte des Toleranzbands war durch einkalkuliertes Rauschen belegt. Dazu ein
  interner Widerspruch: D7 begründet für seinen eigenen Check bereits, warum bei kleinen `n`
  eine feste µs-Grenze aussagekräftiger ist als ein Prozentsatz.
  **(3) MAJOR — dasselbe gilt für das Gewinn-Gate.** Ein „Gewinn" im Nanosekunden-Bereich bei
  `n = 100` konnte den kompletten Implementierungs- und Differentialtest-Apparat auslösen.
  **v6-Auflösung (Owner-Entscheidung 2026-07-22):** duales Kriterium für BEIDE Gates — eine
  Zelle zählt erst als Verletzung bzw. Gewinn, wenn sie relativ UND absolut relevant ist
  (`ABS_RELEVANCE_US = 10`, dieselbe Kategorie wie D7s bereits begründete 50 µs, bewusst
  strenger, weil D6 entscheidungstragend ist). Dazu wird die Kandidatenmenge um die im Raster
  real vorkommenden `k/n`-Werte erweitert — selbst-kalibrierend, ohne geratene Zahl, und sie
  behebt ein Auflösungs-Artefakt: v5s Menge endete nach unten bei `0,01`, wodurch eine
  Verletzung darunter die Zone zwangsweise nullte, obwohl eine kleinere sichere Zone existiert
  hätte. Als MINOR mitbereinigt: hängende `c(n)`-Referenzen in der Berichtspflicht (die
  Konstruktion starb mit v4) und der Status-Quo-Nachtragstext, der den dritten, von v5 neu
  hierher gerouteten Fall (Gewinn außerhalb der Zone) nicht abdeckte.
  **Zuschnitt-Urteil der Zweitmeinung, festgehalten:** Messen statt raten ist richtig, die
  Einzelkonstanten-Hybrid-Form ist verteidigbar und repo-präzedent, `k/n` ist für die
  Sicherheitsseite die richtige Achse (konservativ; die `n`-Abhängigkeit des wahren Crossovers
  kostet nur entgangenen Gewinn). Kein Neuzuschnitt nötig — die drei Fixe erhalten die
  Konstruktion.
- **v5 (2026-07-22):** Simulator-Gegenbeweis der v4-Regel (derselbe Verifier, sein Skript aus
  der v3-Runde auf v4 umgestellt, dieselben Gegenbeispiele plus 4.000 frische Fuzz-Raster).
  **Bestätigt:** alle drei v3-Blocker sind durch v4 sauber gelöst — F1 und B3 liefern jetzt
  eindeutig `STATUS_QUO` statt Abbruch bzw. Doppel-Lesart, F4/F4b liefern `STATUS_QUO` statt
  des unbegründeten `PURE_HEAP`, F3 kommt ohne Iteration zur selben Schwelle wie vorher mit.
  Im Fuzz: 0 Abbrüche, 0 Lesart-Divergenzen (v3: 261 bzw. 1.743). Die Monotonie der
  geschlossenen Form ist ANALYTISCH bewiesen, nicht nur gefuzzt: die sicheren `t` bilden immer
  ein zusammenhängendes Präfix, „der größte sichere `t`" und „der größte, unterhalb dessen alles
  sicher ist" sind beweisbar dieselbe Zahl — die beim Umbau offene Löcher-Frage ist damit
  strukturell erledigt, nicht bloß empirisch unauffällig.
  **Neuer Blocker („hohler Hybrid"), der in v3 nicht auftreten konnte, weil v3 nie bis zu einer
  sauberen Schwellenbildung kam:** v4s Schritt 1 (existiert IRGENDWO ein Gewinn?) und Schritt 3
  (Schwelle rein aus der Sicherheitsbedingung) waren strukturell entkoppelt — der Gewinn, der
  Schritt 1 passieren ließ, musste nicht in der Zone liegen, die Schritt 3 dem Heap zuwies.
  Minimaler Beweis: genau eine Gewinn-Zelle bei `k/n = 0,9`, genau eine Regression bei
  `k/n = 0,02`, sonst überall verträgliche 1,05 — v4 liefert `HYBRID(0,01)`, dessen Heap-Zone
  durchgehend 5 % LANGSAMER ist und dessen einziger Gewinn permanent im Sort-Zweig landet.
  Gegenüber „nichts tun" ausschließlich eine Regression, mechanisch eindeutig berechnet, ohne
  jede Divergenz — und trotzdem falsch. Im positions-entkoppelten Fuzz: 29,4 % aller
  Hybrid-Verdikte beweisbar solche Netto-Regressionen, weitere 32,7 % erfassen nur einen Teil
  der gemessenen Gewinne.
  **v5-Auflösung (Empfehlung des Verifiers, übernommen):** Die beiden Kriterien werden
  gekoppelt und die Reihenfolge umgedreht — Schritt 1 bestimmt die sichere Zone `t*` rein aus
  der Sicherheitsbedingung (ohne jeden Gewinn-Bezug), Schritt 2 verlangt einen Gewinn INNERHALB
  dieser Zone. Aus drei Schritten werden zwei, und drei früher getrennt behandelte
  Status-Quo-Fälle (nirgends ein Gewinn; `t* = 0`; Gewinn außerhalb der Zone) fallen mechanisch
  in denselben NEIN-Zweig. Der Ausgang „reiner Heap" ist seit v5 kein eigener Prüfschritt mehr,
  sondern der Spezialfall `t* = 1.0` (die Zone umfasst jede mögliche Eingabe, die Verzweigung
  wäre immer wahr und entfällt). Als Randbefund dokumentiert und NICHT geändert: „Existenz
  eines Gewinns" bleibt auch in v5 ein binäres Kriterium — ein Raster mit genau einem
  marginalen Gewinn und sonst durchgehend knapp verträglichen Zellen besteht die Prüfung. Das
  ist eine inhärente Eigenschaft jeder binären Toleranzschwelle, kein eigener Fehler dieser
  Regel; es wird im Ergebnisse-Doc benannt, falls die realen Zahlen so ausfallen.
- **v4 (2026-07-22):** Zweites, gezieltes Gegenlesen NUR der Entscheidungsregel D6 (die
  v3-Neufassung war frisch und von niemandem geprüft; der Verifier baute D6 als Skript nach und
  ließ es gegen ca. 14 gezielte und ca. 4.000 zufällige synthetische Raster laufen — und fand
  dabei zunächst einen Nichtdeterminismus in seiner EIGENEN Simulation, korrigierte ihn und
  wiederholte alle Läufe; alle Zahlen unten stammen aus dem korrigierten Lauf). Drei Befunde,
  alle in v4 aufgelöst:
  **(1) Blocker — kein Verdikt.** Lagen die einzigen Schwellen-Verletzungen ausschließlich in
  `ABSOLUTE_K`-Zellen, war `c(n)` für JEDES `n` undefiniert; v3s Konsistenz-Check brach dann ab,
  statt einen der drei benannten Ausgänge zu liefern — ein vierter, unbenannter Zustand, im
  Widerspruch zu D4s eigenem Vertrag „exakt einer der drei Ausgänge". Reproduziert; im Fuzz
  in ca. 6,5 % aller Raster (in der wash-band-artigen Verteilung 39 %).
  **(2) Blocker — zwei Verdikte für dieselben Zahlen.** v3s iterative Absenk-Anweisung („senke
  auf den nächstkleineren gemessenen Wert UNTERHALB dieser Zelle") ist unterspezifiziert, wenn
  die verletzende Zelle unterhalb der kleinsten gemessenen Fraktion liegt — der vom Spec-Text
  selbst als plausibel benannte Fall. Wörtliche Lesart: nicht ausführbar, kein Verdikt.
  Wohlwollende Lesart: Status Quo. Divergenz in ca. 43,6 % der gefuzzten Raster.
  **(3) Major — Verdikt widerspricht der eigenen Begründung.** v3s Schritt 1 prüfte nur „nirgends
  schlechter als 1,15", nie „irgendwo besser". Ein Raster, in dem der Heap überall exakt 1,10-fach
  langsamer und NIRGENDS schneller ist, lieferte trotzdem „reiner Heap" — also die unbedingte
  Ersetzung von produktionsgetestetem Sortier-Code ohne jeden gemessenen Vorteil. Das
  widerspricht der Asymmetrie-Begründung der Regel selbst (sie setzt stillschweigend voraus,
  dass es eine Beschleunigung zu entgehen gibt) und dem Ehrlichkeits-Rahmen der Scheibe.
  **v4-Auflösung:** (a) Die Meaningful-Win-Prüfung wandert an die ERSTE Stelle — ohne irgendwo
  gemessenen echten Gewinn wird nichts ersetzt, auch nicht unbedingt (löst 3). (b) Die
  Kreuzungspunkt-Konstruktion samt iterativer Absenkung und separater Nachvalidierung wird
  VOLLSTÄNDIG durch eine geschlossene Form ersetzt: der größte Wert `t` aus
  `RATIO_FRACTIONS ∪ {0}`, für den jede Zelle mit `k/n ≤ t` die Toleranz einhält; `t = 0` bedeutet
  Status Quo. Das löst 1 und 2 zugleich (`0` ist immer Kandidat und immer sicher, also existiert
  der größte sichere `t` IMMER; und es gibt keine iterative Anweisung mehr, die ohne Kandidaten
  unterspezifiziert wäre) und macht zusätzlich die in v3 separat angebaute Ganzraster-Validierung
  überflüssig, weil beide Zell-Familien in der Definition selbst eingeschlossen sind.
  Nicht kaputt und bestätigt: die Terminierung der v3-Schleife, ihre Ordnungsunabhängigkeit, und
  dass keine gemessen schlechte Zelle je als heap-sicher ausgewiesen wurde, wo die Validierung
  überhaupt erreicht wurde. Als Minor dokumentiert: die Verzweigung selbst kostet minimal etwas,
  „oberhalb läuft exakt das heutige Verfahren" ist auf Code-Ebene wahr und auf Uhr-Ebene eine
  Näherung — bereits durch D7s Dispatch-Mehraufwand-Gate gemessen und dort als informativ
  deklariert.
- **v1 (2026-07-21):** Erstfassung — algorithmische Entscheidung für einen größenbeschränkten
  Heap, Owner-Auftrag, Design/Test-/Mess-Plan; durch v2 in ihrer Entscheidungs-Vorwegnahme
  zurückgenommen, in ihrer Korrektheits-/Test-/Doc-Disziplin (Totalordnungs-Argumentation,
  Orakel-Migration, Differentialtest, JIT-NaN-Payload-Fund) vollständig übernommen.
