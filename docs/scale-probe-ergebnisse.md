# Scale-Probe: Ergebnisse

Spec: [docs/scale-probe-spec.md](scale-probe-spec.md) v2 (+v2.1-Nit) · Datum: 2026-07-21 ·
Eskalationsstufe 3 (voller Verify-Katalog A+B+C).

**Ehrlichkeitsregel dieses Dokuments** (Hausregel aller `docs/*-ergebnisse.md`): Jede Zahl
stammt aus einem Kommando, das tatsächlich gelaufen ist, mit geprüftem Exit-Code. Was nicht
gemessen wurde, steht als nicht gemessen da. Fehlschläge, Grenzen und die eigenen
Prozess-Verstöße stehen mit drin, nicht nur die Ergebnisse.

## Worum es ging

Die README trug seit dem ersten Release die Qualifikation „**newly tractable, unproven at
scale**" — die letzte Aussage des Projekts ohne Messung dahinter. Diese Scheibe ersetzt das
Wort *unproven* durch gemessenes Terrain: ein generierter Sweep über vier vorregistrierte
Achsen, kalibriert an der echten Op-Mischung von `examples/rag-demo`, mit einem dauerhaft
gepinnten Sentinel im bestehenden harten Editor-Gate.

Ein gefundener Cliff war ausdrücklich als **Ergebnis** definiert, nicht als Scheitern — die
Spec ist so gebaut, dass an keiner Stelle ein Anreiz entsteht, schlechte Zahlen zu verstecken
(§1, G3).

## Die zentrale Aussage

**Die interaktive Latenz hält überall.** Über alle 34 messbaren Sweep-Punkte hinweg lag der
Median des warmen Hovers zwischen **0,044 ms und 0,106 ms** — bei einem 250-Datei-Projekt
genauso wie bei einer 10.000-Glieder-Kette (2,68 Mio. Instantiations) oder bei Rang 896.
Das ist dieselbe Größenordnung wie die in Spike 02 an ca. 72-Datei-Projekten gemessene
(0,04–0,08 ms), also praktisch unbeeinflusst von der Projektgröße.

**Die Kosten der Skala landen auf dem KALTSTART und der Batch-Prüfzeit, nicht auf dem Tippen.**
Der Kaltstart (Datei öffnen bis erste Diagnostik) reicht von 1,3 ms bis **10,2 s** — je nach
Achse. Das ist die nutzer-erlebbare Grenze, und sie ist eine andere als die, die man
intuitiv erwartet hätte.

## Achse (a) — Korpusgröße

Zwei Unter-Serien (D19): `same` = alle Dateien nutzen dieselben literalen Shapes,
`distinct` = jede Datei eigene, nie wiederkehrende.

| N Dateien | same | distinct | Kaltstart same/distinct | warmer Hover (Median) |
|---|---|---|---|---|
| 1 | 33.811 | 34.008 | 1,62 / 1,41 ms | 0,084 / 0,079 ms |
| 10 | 52.432 | 77.816 | 1,56 / 1,47 ms | 0,079 / 0,068 ms |
| 25 | 52.432 | 135.657 | 1,46 / 1,59 ms | 0,071 / 0,044 ms |
| 50 | 52.432 | 234.684 | 1,42 / 1,35 ms | 0,071 / 0,066 ms |
| 100 | 52.432 | 429.202 | 1,79 / 1,33 ms | 0,090 / 0,060 ms |
| 250 | 52.709 | 998.879 | 1,45 / 1,50 ms | 0,067 / 0,062 ms |

**Zwei Befunde.** Erstens: Bei wiederholten Shapes **plateaut** der Instantiation-Zähler ab
ca. 10 Dateien vollständig — 250 Dateien kosten exakt so viel wie 10. TypeScript dedupliziert
identische Typinstanziierungen projektweit. Zweitens: Bei verschiedenen Shapes wächst er
**linear mit fester Grundlast**, nicht überproportional. Marginalkosten je zusätzlicher Datei
über die Segmente: 4.868 → 3.856 → 3.961 → 3.890 → **3.798** — flach bis leicht *fallend*.
Faktor `distinct/same` bei N=250: **19,0x**.

Der Kaltstart bleibt über die gesamte Achse bei **1,3–1,8 ms** — die Dateizahl bewegt ihn
praktisch nicht. Die inkrementelle Tipp-Latenz (M3-Toggle, nur an den Spitzenpunkten, D22)
liegt bei ca. **4,8 ms** in beiden Unter-Serien.

**Warum beide Serien nötig waren:** Hätte die Probe nur `same` gemessen — die naheliegende
Konstruktion, und exakt das Muster des bestehenden W5-Workloads —, stünde hier eine
waagerechte Linie und darunter „skaliert mühelos", ohne dass jemand falsch gemessen hätte.
Der Befund stammt aus der Frontier-Zweitmeinung zur Spec und wurde vor dem Bau eingearbeitet
(Spec-Addendum, D19).

## Achse (b) — Ketten-Tiefe

Zwei Unter-Serien (D20): `fix` = feste `[8,8]`-Shape über die ganze Kette (die bestehende
W1-Konstruktion), `variable` = pro Schritt frische, nie wiederkehrende Shapes.

| L Glieder | fix | variable | Prüfzeit fix/variable | Kaltstart fix/variable |
|---|---|---|---|---|
| 100 | 27.904 | 53.704 | 0,016 / 0,022 s | 2,41 / 21,0 ms |
| 500 | 27.904 | 159.704 | 0,015 / 0,106 s | 12,7 / 106 ms |
| 1.000 | 27.904 | 292.204 | 0,041 / 0,239 s | 41,8 / 252 ms |
| 2.500 | 27.904 | 689.704 | 0,226 / 0,797 s | 247 / 923 ms |
| 5.000 | 27.904 | 1.352.204 | 0,843 / 2,498 s | 999 ms / 3,05 s |
| 10.000 | 27.904 | 2.677.204 | 3,302 / 8,193 s | 3,74 s / **10,24 s** |

`fix` bleibt über alle sieben Punkte **exakt konstant bei 27.904** — derselbe Mechanismus wie
bei Achse (a): jeder Kettenschritt instanziiert denselben Typ, TypeScript rechnet ihn einmal.
`variable` wächst **exakt linear mit 265,0 Instantiations pro Kettenglied**, über alle sechs
Segmente hinweg auf die Stelle konstant.

Kettenlänge stresst also **nicht** die Rekursionstiefe des Typsystems, sondern reinen
Durchsatz. Auch die adversariale Konstruktion (nie wiederkehrende Shapes, gebaut vom
Verify-Baustein B als eigener Angriff) zeigt keine versteckte Tiefen-Kopplung — nur lineare
Kosten. Der warme Hover bleibt bei 0,056–0,081 ms, selbst bei 2,68 Mio. Instantiations; der
Kaltstart dagegen erreicht 10,2 s, und der Speicher 968 MB.

## Achse (c) — Rang

| Rang | Instantiations | Prüfzeit | Speicher | Kaltstart | warmer Hover |
|---|---|---|---|---|---|
| 16 | 28.758 | 0,016 s | 40 MB | 4,95 ms | 0,069 ms |
| 32 | 31.062 | 0,018 s | 42 MB | 7,04 ms | 0,069 ms |
| 64 | 38.742 | 0,018 s | 46 MB | 12,7 ms | 0,069 ms |
| 128 | 66.390 | 0,035 s | 60 MB | 31,8 ms | 0,082 ms |
| 256 | 170.838 | 0,108 s | 112 MB | 105 ms | 0,082 ms |
| 512 | 576.342 | 0,413 s | 289 MB | 424 ms | 0,106 ms |
| 768 | 1.243.990 | 0,988 s | 601 MB | 1,02 s | 0,094 ms |
| 896 | 1.676.118 | 1,352 s | 772 MB | 1,47 s | 0,097 ms |
| **1024** | **2.170.608 bis Abbruch** | 1,855 s | 973 MB | — | — |

**Hier ist das Wachstum echt überproportional.** Marginalkosten je Rang zwischen den
Verdopplungsschritten: 144 → 240 → 432 → 816 → 1.584 → 2.608 — sie verdoppeln sich mit jeder
Verdopplung des Rangs. Das ist der strukturelle Unterschied zu den Achsen (a) und (b):
`BroadcastAcc` rekursiert genuin einmal pro Tupel-Position INNERHALB eines einzelnen Aufrufs,
während Ketten- und Dateischritte unabhängige, cachebare Aufrufe sind.

**Bei Rang 1024 bricht `tsc` ab:** `error TS2589: Type instantiation is excessively deep,
possibly infinite.` Konsistent mit der dokumentierten ca. 1000er-Grenze für tail-rekursive
Formen. Der letzte durchlaufende Punkt ist Rang 896.

**Einordnung, ehrlich:** Reale NumPy-artige Arrays haben einstelligen Rang. Der Cliff ist ein
Befund über die MASCHINE, nicht über den Anwendungsfall — er wird hier berichtet, weil das
Projekt behauptet, seine Grenzen zu kennen, nicht weil er jemanden trifft.

**Der Cliff ist eine Falsch-Ablehnungs-Grenze, nicht nur eine Kostengrenze** (D25.1): An
diesem Punkt lehnt der Checker **gültigen Code** ab, und die Degradations-Disziplin des
Projekts („wide statt falsch", COVENANT M2) kann dort strukturell nicht greifen, weil der
Checker vor jeder Degradations-Logik stirbt. Das ist keine Änderung dieser Scheibe — die
Grenze existierte immer, unbenannt. Sie liegt jetzt als v6-Präzisierungskandidat in
FOLLOWUPS; still mitentschieden wurde nichts.

## Achse (d) — Kaltstart im Groß-Korpus

Achse (d) ist keine eigene Zahlenreihe, sondern die an jedem Punkt der Achsen (a)-(c)
miterfasste Kaltstart-Messung (D7). Die Schlagzeilen-Datenpunkte:

| Spitzenpunkt | Kaltstart |
|---|---|
| Achse (a), 250 Dateien | **1,5 ms** |
| Achse (c), Rang 896 | **1,47 s** |
| Achse (b), 10.000 Glieder, feste Shape | **3,74 s** |
| Achse (b), 10.000 Glieder, variable Shapes | **10,24 s** |

Die Dateizahl ist also der harmloseste Skalierungsfaktor, die Länge einer einzelnen Datei der
teuerste. Das ist die praktisch verwertbare Erkenntnis dieser Probe: Ein Projekt darf viele
Dateien haben; eine einzelne Datei sollte keine zehntausend verketteten Operationen tragen.

## Mess-Randbedingungen

Ein einziger, vollständiger Lauf (`pnpm bench:scale`, Exit 0), **nicht** aus mehreren Läufen
zusammengesetzt — Baustein A hatte gezeigt, dass Kaltstart-Zeiten zwischen Sitzungen spürbar
schwanken (er maß 10,1 s, wo der Implementierer 9,1 s gemessen hatte). Maschine: Apple M3,
8 Kerne, 16 GB, macOS 26.5.2 arm64, Node v24.16.0, TypeScript 7.0.2. **Maschinenlast beim
Start: 3,46 / 3,11 / 3,13** (mehrere parallele Agent-Sessions; ruhiger war die Maschine zu
diesem Zeitpunkt nicht zu bekommen), am Ende 3,27 / 3,10 / 3,12 — stabil über den Lauf.
Wall-Clock der Messschleife: **68,51 s**, gesamter Befehl 77 s. Gate G1 (30 Minuten) mit
ca. 3,8 % Auslastung eingehalten, keine Sweep-Punkt-Reduktion nötig.

35 Punkte: **34 ok, 1 cliff (erwartet), 0 other-failure.**

Warme Hover-Werte sind Mediane aus 20 gezeiteten Samples nach 3 Warmups je Position; einzelne
Maximalwerte liegen deutlich höher (bis 19,3 ms bei b-variable-10000) — Ausreißer eines
belasteten Hosts, deshalb werden Mediane berichtet und die Spannen mitgeführt.

## Was BEWIESEN wurde, dass es prüft (Nicht-Vakuität)

- **T1** — Werden die echten Ops gemessen oder nur Gerüst? Mutant, der die Op-Ketten der
  generierten Dateien durch Kommentare ersetzt: `a-distinct-25` bricht von 135.657 auf 27.451
  ein (ca. −80 %). Von Baustein A unabhängig wiederholt, Revert per Backup-Kopie mit
  `diff`-Beweis.
- **T2** — Ist die Cliff-Erkennung hartkodiert? Mutant mit invertierter Logik klassifiziert
  den sauber kompilierenden Punkt `c-16` fälschlich als Cliff. Zusätzlich hat Baustein B die
  Klassifikationslogik verbatim extrahiert und mit synthetischen Fehlerobjekten durchgespielt:
  echter TS2589 → cliff; SIGKILL → other-failure; Heap-OOM ohne Signalfeld → other-failure;
  unerwarteter Nicht-TS2589-Typfehler → other-failure; und der adversariale Fall eines
  signal-getöteten Prozesses, dessen abgeschnittene Ausgabe zufällig „excessively deep"
  enthält → **weiterhin other-failure** (die Signalprüfung läuft vor der Textprüfung).
- **T3** — Fängt das Korrektheits-Gate falsche Anzeigen? **Hier lag der einzige echte Defekt
  dieser Scheibe** (siehe unten).
- **T7** — Läuft der Sweep nach einem Cliff weiter? Baustein B hat ein Manifest gebaut, in dem
  der echte Cliff-Punkt an ERSTER Stelle steht, und den echten Runner end-to-end laufen lassen:
  Cliff korrekt klassifiziert, beide Folgepunkte danach normal gemessen.
- **T8** — Sind die Unter-Serien wirklich verschieden? Die Generator-Selbstkontrollen wurden
  per Mutation als wirksam bewiesen (erzwungene Dim-Kollision → sofortiger Wurf; neun
  zusätzliche `mean()`-Aufrufe → Verletzung der Op-Mix-Bindung, Exit 1). Beide Reverts per
  Backup-Kopie mit `diff`-Beweis.
- **T5** — Bewegt der Sentinel die bestehenden Pins? `bench:editor` zweifach: w1-w7 exakt
  `{27904, 29713, 60853, 28067, 33358, 34528, 27076}`, w8 = 34943, Hard-Gate PASS.
- **T4** — Der Root-Pin wurde per empty-then-fill zerlegt (siehe unten).

## Der Defekt, den die Verify-Runde fand — und seine Behebung

Baustein B bewies mechanisch: Die Korrektheitsprüfung der Rang-Achse war **vakuös**. Weil
`tsc` lange Tupel in der Hover-Anzeige kürzt, hatte die Erstumsetzung den Vergleich auf die
ersten sechs Dimensionen verkürzt — und zwar **bedingungslos für jeden Rang**, auch für die
kleinen, wo gar nicht gekürzt wird. Der Verifier extrahierte die echte Prüffunktion und fütterte
ihr eine Antwort, die an Position 0-5 stimmt und ab Position 6 durchgehend falsch ist: kein
Fehlschlag. Für die gesamte Achse konnte also keine Falschheit jenseits von Position 5 je
auffallen.

Empirisch nachgemessen (nicht geraten), wie `tsc` 7.0.2 tatsächlich kürzt: Rang 16/64/128
werden **voll** angezeigt; ab Rang 256 erscheint die Form
`NDArray<[<167 Dims>, ... N more ..., <1 Dim>]>` — Präfix 167, Suffix 1, und
`167 + N + 1` rekonstruiert den wahren Rang exakt (167+88+1 = 256, 167+344+1 = 512,
167+728+1 = 896).

Die Behebung prüft dementsprechend: Volltreffer-Vergleich, wo nicht gekürzt wird; bei Kürzung
Vergleich von sichtbarem Präfix UND sichtbarem Suffix plus Rekonstruktion der Gesamtlänge aus
dem Kürzungsvermerk. Die Werte 167/1 sind bewusst **nicht** hartkodiert, sondern werden aus
dem Text gelesen — ein anderes Kürzungsfenster einer künftigen TS-Version bricht die Prüfung
nicht.

Nachgewiesen wirksam: **acht Mutationen** (falsche Dimension am Anfang, am Ende, Rang zu kurz,
Rang zu lang — je für einen ungekürzten und einen gekürzten Rang) scheiterten jeweils hart mit
`CORRECTNESS GATE FAILED`; **zwei Kontrollläufe** ohne Mutation liefen sauber durch. Die
Mutationen liefen gegen den echten Code-Pfad inklusive echtem Sprachserver, nicht gegen eine
Nachbildung; mutiert wurde ausschließlich das generierte, gitignorierte Manifest, dessen
Wiederherstellung per Regeneration und `diff` als byte-identisch belegt ist.

**Ehrlich mitgeführte Grenze:** Eine falsche Dimension strikt innerhalb des verborgenen
Mittelteils, die die Gesamtlänge nicht ändert, bleibt unsichtbar — die Information steht nicht
im Hover-Text, auch nicht für einen Menschen, der dasselbe Tooltip liest. Das steht als
Kommentar im Code.

## Pins

- **Root `check:diag`: 199.877 @ 139 Dateien** (vorher 201.455 @ 137). Zerlegt nach D3/T4,
  von Baustein A unabhängig aus einem frischen Worktree reproduziert: zwei leere
  `export {}`-Platzhalter → **199.045 @ 139** (Order-Noise-Anteil **−2.410**, reines
  Reihenfolge-Rauschen durch zwei zusätzliche Dateien); befüllt → **199.820** (echte Typkosten
  **+775**); die Hover-Gate-Reparatur danach **+57** (Dateizahl unverändert, also kein
  Order-Noise, sondern echte Kosten des neuen `expectedDims`-Felds und der Parse-Funktionen).
  Nebenbefund von Baustein A: Der Beitrag der w8-Sentinel-Änderungen zu diesem Korpus ist
  **exakt null** — seine Messung ohne diese Änderungen traf denselben Wert.
- **`check:diag:stress` 106.398 @ 82** und **`check:diag:browser` 2.142 @ 75** — unverändert,
  mechanisch garantiert durch Include-Isolation.
- **`bench:editor`: acht Pins**, w1-w7 unverändert auf den V0-Werten, **w8 = 34.943** neu.
- **Artefakt-Hash** unverändert — diese Scheibe fasst keinen Rust-Code an.

## Prozess-Verstöße und Grenzen, offengelegt

1. **Die Chain-Vorab-Probe der Spec-Erstfassung lief auf einem Exit-1-Programm** (ohne
   `ambient.d.ts`) und wurde neben einer sauberen Probe als wechselseitig bestätigend
   präsentiert — ein Verstoß gegen die eigene Exit-Code-Hausregel. Der Befund selbst überlebt
   (beide Verifier haben ihn sauber reproduziert: exakt konstant 27.701 bzw. im Endlauf
   27.904), aber der Verstoß wird benannt statt geheilt (D25.3).
2. **Die Kontrollpunkte vergleichen erst nach V0 saubere Zahlen.** Vor der Vorab-Scheibe V0
   (Commit `c18aa7f`) liefen alle sieben Editor-Workloads gegen Programme mit sieben
   unaufgelösten TS2591-Diagnosen; `enforceHardGate` liest `hadTypeErrors` nirgends, deshalb
   fiel es nie auf. Die alten Pins waren Zahlen aus fehlerhaften Programmen. Die Latenzwerte
   waren davon nicht betroffen (der Sprachserver-Pfad ist ein anderer), die im README
   publizierte Hover-Aussage hält auf der sauberen Basis unverändert (D25.2).
3. **Der Implementierer beschrieb die Achse-(a)-Kurve als „deutlich überproportional zu N".**
   Das tragen die eigenen Zahlen nicht — beide Verifier haben unabhängig nachgerechnet und
   „linear mit fester Grundlast" belegt. Die Formulierung war nie committet; sie wäre in die
   publizierte Aussage gewandert. Berichtet, weil ein Beinahe-Fehler dieser Art in die
   Prozess-Historie gehört, nicht nur seine Korrektur.
4. **Ein Agent beendete seinen Turn auf einer Absicht** („I'll wait for the background task")
   statt auf einem Ergebnis; der Report wurde per Wiederaufnahme nachgefordert. Die
   Sabotage-Nachweise existierten, waren aber ungemeldet — ohne Nachfrage wären sie als
   unbelegt in dieses Dokument eingegangen.
5. **Nicht gemessen:** die exakte Kürzungsschwelle zwischen Rang 128 und 256; das Verhalten auf
   `ubuntu-latest` (der Sweep läuft per Owner-Entscheidung nur lokal, nur der Sentinel läuft in
   CI); ein echter Speicher-Kill (nur per verbatim extrahierter Logik mit synthetischen
   Fehlerobjekten geprüft); der volle 35-Punkte-Lauf durch Baustein B (er fuhr eine reduzierte
   Variante plus viele isolierte Einzelprüfungen).
6. **Der Sweep-Korpus läuft nur auf Zuruf, nicht in CI.** Das weicht wörtlich von COVENANT Z2
   ab; `covenant-verify` hat es als Abweichung mittlerer Schwere benannt. Owner-Entscheidung
   2026-07-21: Vertragstext in v6 präzisieren (benannte dritte Gate-Klasse für
   on-demand-Mess-Korpora), nicht die Abweichung dauerhaft dulden. FOLLOWUPS trägt den
   Kandidaten; das v6-Bündel steht damit bei vier.

## Post-Verification-Addendum

Drei unabhängige Fresh-Context-Verifier, parallel, gegen Commit `28690dc`:

**Baustein A (Spec-Konformität): CONFIRMED** für die Implementierung. Alle Gate-Zahlen
unabhängig reproduziert, teils bit-identisch; die empty-then-fill-Zerlegung komplett neu aus
einem frischen Worktree nachgebaut mit identischem Ergebnis; eigener Pflicht-Mutant bestätigte
T1. Ein MAJOR-Befund: die „überproportional"-Charakterisierung (siehe Verstoß 3). Ein
Blocker-Level-Hinweis: die Scheibe war zu diesem Zeitpunkt nicht abgeschlossen, weil Ergebnis-
Doc und README-Ersetzung fehlten — dieses Dokument schließt den ersten Teil.

**Baustein B (adversarial): kein Blocker, ein MAJOR** — das vakuöse Hover-Gate (oben, behoben).
Dazu ein selbst gestellter und ausgeräumter Einwand: In der `distinct`-Konstruktion wachsen mit
dem Dateiindex auch die Zahlenwerte, die Kurve könnte also wachsende Magnituden statt wachsender
Vielfalt messen. Er baute eine Gegen-Konstruktion mit fest begrenzten Größen und maß
3.812 / 3.711 Instantiations pro Datei — praktisch identisch zu den 3.798–3.961 der echten
Konstruktion. Der Confound existiert, verfälscht das Ergebnis aber nicht. Weitere Angriffe ohne
Fund: Generator-Determinismus (zwei Läufe byte-identisch), Reproduzierbarkeit der
Instantiation-Zahlen (exakt), Korpus-Ausschluss (`--listFilesOnly` findet keinen Sweep-Pfad),
`.gitignore`-Wirksamkeit, N=0 (degradiert sauber zu other-failure, Sweep läuft weiter).

**Baustein C (Covenant): keine Verletzung, zwei Textlücken mittlerer Schwere** — Z2 (siehe
Verstoß 6) und M2 (der Rang-Cliff, siehe Achse c). Beide als v6-Kandidaten dokumentiert, keine
still mitentschieden. S1, M1, M3, M4, M5, Z1 unberührt; `graph-a-lama query lint` 0 Befunde.

## Gate-Block (Abschlusslauf)

| Gate | Exit | Ergebnis |
|---|---|---|
| `pnpm check` | 0 | sauber (Dreifach-Verbund) |
| `pnpm check:diag` | 0 | 199.877 @ 139 |
| `pnpm check:diag:stress` | 0 | 106.398 @ 82 |
| `pnpm check:diag:browser` | 0 | 2.142 @ 75 |
| `pnpm test:core` | 0 | 1588 / 1588 |
| `pnpm bench:editor` | 0 | acht Pins exakt, Hard-Gate PASS |
| `pnpm bench:scale` | 0 | 34 ok / 1 cliff / 0 other-failure, 68,51 s |
| `graph-a-lama query lint` | 0 | 0 Befunde |
