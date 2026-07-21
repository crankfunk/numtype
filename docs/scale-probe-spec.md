# Scale-Probe: Checker-Budget & Editor-Latenz im Maßstab — bindende Spec

Status: **bindend** (Owner-Programm 2026-07-21: Fortsetzung von Spike 02 — die Scale-Probe
überführt die letzte unbelegte README-Aussage, „unproven at scale", in gemessenes Terrain).
Version: **2** (v1 = Erstfassung; v2 nach Baustein 0 + Frontier-Zweitmeinung, beide
Befundsätze gemerged, Owner-Entscheidungen vom 2026-07-21 eingearbeitet — vollständige
Herleitung im Addendum „Adversariale Spec-Verifikation" am Ende) · Datum: 2026-07-21 ·
Eskalationsleiter: **Stufe 3** (substanzielle Scheibe mit bindender Spec — voller
Verify-Katalog A+B+C parallel + `graph-a-lama query lint` im Gate-Block).
Covenant: v5.

## Ziel & Warum

README §5 (`README.md:259`) sagt wörtlich: **„'TypeScript can do this' means: newly
tractable, unproven at scale."** Das Wort *unproven* ist die letzte unbelegte Behauptung
des Projekts — jede andere USP-Qualifikation ist inzwischen an eine Messung gebunden
(Slice-Arithmetik-Budget, Bounds-Check-Budget, Editor-Hover-Latenz — alle drei
`README.md:268-270`, aus Spike 01/02/03). Diese Scheibe schreibt NUR die bindende Spec
für die Messung, die *unproven* durch eine gemessene Aussage ersetzt: ein realistischer
GROSS-Korpus, Checker-Budget und Editor-Latenz gemessen, Kurven statt Vermutungen, entlang
vier vorregistrierter Achsen (Owner-Entscheidung 1): Korpusgröße, Op-Ketten-Tiefe, Rang,
Editor-Latenz kalt im Groß-Korpus.

## 1. Ziel, Nicht-Ziele, Abgrenzung

**Ziel:** Vier vorregistrierte Achsen (Korpusgröße, Chain-Tiefe, Rang, Editor-Latenz kalt)
werden über einen synthetisch generierten, aber am echten Dogfooding-Code (`examples/rag-demo`)
kalibrierten Sweep gemessen. Ergebnis ist eine Kurve pro Achse (Instantiations/Checkzeit/
Latenz gegen den Sweep-Parameter) plus EIN gepinnter Sentinel-Workload im bestehenden
`bench:editor`-Hartgate, der die Aussage vor stillem Verrotten schützt. Ein gefundener Cliff
(z. B. „Type instantiation is excessively deep" bei Rang X) ist ein **Ergebnis dieser
Scheibe**, kein Scheitern — die Spec erzeugt an keiner Stelle einen Anreiz, schlechte Zahlen
zu verstecken oder Sweep-Punkte nachträglich so zu wählen, dass die Kurve gut aussieht. Die
Sweep-Punkte unten sind VOR jeder Messung fixiert (Owner-Entscheidung 1/2); eine spätere
Scheibe entscheidet, ob ein gefundener Cliff behoben, dokumentiert oder als Nicht-Ziel
markiert wird — das ist NICHT Teil dieser Messung.

**Nicht-Ziele (Owner-Entscheidung 4, wörtlich):**
- Ob TS 7.0.2 überhaupt einen harten ca. 5M-Instantiation-Deckel hat, wird NICHT geprüft.
  Die Zahl (`CLAUDE.md` „Key TS limits", `docs/wettbewerbsanalyse-und-usp.md:43`, sourced auf
  TS5/Issue #53514) darf im Ergebnis-Doc NUR als unverifizierte TS-5.x-Literaturreferenz
  erscheinen — keine Aussage der Art „wir nutzen X % des 5M-Budgets" wird aus dieser Scheibe
  abgeleitet, weil das Budget selbst nie gemessen wurde und laut CLAUDE.md „versionsfragil"
  ist.
- Keine Aufbereitung als Blog-Post-Rohmaterial.
- Kein Refactoring, keine neuen Ops, keine Runtime-/Rust-/WASM-Änderung (Scope-Disziplin,
  Intentions-Kontext). Ein gefundener Cliff wird NICHT in dieser Scheibe behoben.
- Kein Fix am `pnpm check`-Verbund; an den bestehenden Workloads W1-W7 nur die in **V0**
  (siehe direkt unten) beschlossene Mess-Basis-Reparatur, sonst nur eine additive
  Erweiterung um genau einen Sentinel (§6).

**V0 — bindende VORBEDINGUNG (Owner-Entscheidung 2026-07-21, v2):** Vor dem ersten
Sweep-Lauf läuft eine eigene, kleine Vorab-Scheibe (Eskalationsstufe 2), die die Mess-Basis
repariert: `spike/src/ambient.d.ts` wird in BEIDE tsconfig-Ausgaben von `gen-workloads.ts`
aufgenommen (das geteilte `workloadsDir/tsconfig.json`, `gen-workloads.ts:632`, UND die
per-Workload-Isolate, `gen-workloads.ts:648`), w1-w7 werden dekomponiert neu gepinnt, und
die im README publizierte Hover-Zahl (`README.md:268-270`) wird auf der sauberen Basis
re-verifiziert. **Begründung:** Ohne V0 entsteht der Sentinel w8 (§6) im selben defekten
Generator-Pfad — der EINZIGE kontinuierliche Wächter der neuen publizierten Aussage würde
also auf einem Programm mit sieben unaufgelösten TS2591-Diagnosen gemessen und gepinnt; und
die Kontrollpunkte in §2.2/§2.3 („reproduziert den heutigen Pin") verglichen sonst zwei
verschiedene Mess-Populationen, was die projekteigene Populationsregel verbietet. Beide
Verifier haben den Delta unabhängig auf **+135 Instantiations** gemessen (klein und per
empty-then-fill dekomponierbar). Pin-Stabilität auf defekter Basis ist kein Wert. Diese
Scheibe beginnt NICHT, bevor V0 committet und grün ist.

**Abgrenzung zu Spike 02:** Spike 02 maß Editor-Latenz an *realistischen* Workloads
(ca. 72 Dateien / ca. 57 kLOC Gesamtprojekt, Chain-Länge bis 100, Rang bis 16 — die
`INSTANTIATION_PINS` in `spike/bench-dx/editor-latency.ts:775-783` sind der heutige Stand:
`{w1: 27769, w2: 29578, w3: 60718, w4: 27932, w5: 33223, w6: 34393, w7: 26941}`, gemessen
nach Op-Scheibe W5). Diese Scheibe fährt exakt dieselbe Methodik (hand-gerollter LSP-Client,
Korrektheits-Gate vor jeder Zeitmessung, `check:diag`-Instantiation-Zählung) über deren
GRENZEN hinaus.

## 2. Die vier Achsen mit vorregistrierten Sweep-Punkten

Alle vier Achsen nutzen dieselbe Methodik wie `spike/bench-dx/editor-latency.ts`: M1 (kalt),
M2 (warmer Hover), M4 (Completion, informativ) je Sweep-Punkt; M3 (Toggle) nur dort, wo ein
Sweep-Punkt einen echten Fehler-Toggle trägt (siehe §9, Nicht-Vakuität). Jeder Sweep-Punkt ist
ein eigenständiges, isoliertes `tsconfig.<id>.json`-Programm nach dem Muster von
`spike/bench-dx/gen-workloads.ts:643-648` — Details in §4.

### 2.1 Achse (a) — Korpusgröße (N Call-Sites / N Dateien)

**Konstruktion:** N unabhängige Dateien, jede ca. so groß wie das bestehende W5
(„realistic mixed consumer", `spike/bench-dx/gen-workloads.ts:381` `W5_LAYER_COUNT = 42`,
Ziel 200-400 LOC, `gen-workloads.ts:668`), jede mit der kalibrierten Op-Mischung aus §5, ALLE
gemeinsam als EIN Mehrdatei-Programm typgeprüft (ein gemeinsames `tsconfig.json` mit
`include: ["*.ts"]`, mirror von `gen-workloads.ts:632`). Das misst, ob Checkzeit/Speicher
linear oder überlinear mit der Dateizahl eines wachsenden Konsumenten-Projekts wächst — die
Frage, die „wachsendes OSS-Projekt mit Nutzern" (CLAUDE.md-Status) konkret aufwirft.

**Sweep-Punkte:** `{1, 10, 25, 50, 100, 250}` Dateien — **in ZWEI getrennten Unter-Serien**
(bindend seit v2, siehe D19).

**D19 — Shape-Diversität ist bindend (v2, aus der Frontier-Zweitmeinung).** Achse (a) wird
zweimal gefahren: **(a-gleich)** alle Dateien nutzen dieselben literalen Shapes, **(a-distinkt)**
jede Datei nutzt eigene, nirgends wiederkehrende literale Shapes. Grund: TS dedupliziert
identische Typinstanziierungen projektweit, weshalb die Op-Mischung allein (§5/D10) die
Kurve NICHT bestimmt. Gemessen (Zweitmeinung, N Dateien à ca. 17 Op-Sites, echter Typ-Layer,
alle Läufe Exit 0): identische Shapes `5 → 53.052`, `25 → 58.032`, `50 → 64.257` (marginal
ca. **250** Instantiations/Datei) gegen distinkte Shapes `5 → 56.642`, `25 → 132.976`,
`50 → 222.141` (marginal ca. **3.680/Datei — Faktor ca. 15**). Eine Ein-Serien-Messung auf
wiederholten Shapes hätte eine fast flache Korpus-Kurve produziert, ohne dass irgendjemand
falsch misst — das ist der teuerste vermeidbare Fehler dieser Scheibe. Das bestehende
Vorbild W5 (42× fixe `[8,8]`-Layer, `gen-workloads.ts:381,413-415`) ist maximal
cache-freundlich und darf NICHT unbesehen als Größenvorbild für die Dims übernommen werden.
Reale Projekte liegen zwischen beiden Serien (Embedding-Dims wiederholen sich, vieles
andere nicht) — deshalb werden beide als Best Case / Worst Case berichtet, nie zu einer
Zahl gemittelt (§7/G4). Der Kontrast selbst ist ein publizierbares Ergebnis.

- `1`: Kontrollpunkt, reproduziert strukturell W5.
- `10`/`25`/`50`: Progression über die Größenordnung typischer kleiner-bis-mittlerer
  TS-Konsumenten-Projekte.
- `100`: bewusst gewählt, weil er der GRÖSSE des heutigen numtype-Root-Korpus selbst
  entspricht (137 Dateien, CLAUDE.md „Aktuelle Pins & Gates" — 100 ist die nächstliegende
  runde Zahl darunter, ein direkt vergleichbarer, quellenverankerter Referenzpunkt: „ein
  Konsumenten-Projekt in der Größenordnung von numtype selbst").
- `250`: bewusster Stress-Punkt, ca. 1,8× der Größe des Projekts selbst — ein „großes,
  wachsendes OSS-Konsumenten-Projekt"-Szenario.

**Vermuteter Cliff:** keine Instantiation-Explosion erwartet (jede Datei ist unabhängig,
keine geteilte Rekursion über Dateigrenzen) — der vermutete Engpass ist Checker-DURCHSATZ
(Parse+Bind+Check-Zeit linear in Dateizahl × Dateigröße) und M1-Kaltstart (Projekt-Discovery
über N Dateien). Eine überlineare Kurve hier wäre der eigentliche Fund.

### 2.2 Achse (b) — Chain-Tiefe über die heutigen 100 hinaus

**Vorab-Probe-Befund (siehe „Was empirisch geprobt wurde" unten), der diese Achse
umformuliert:** Eine Vorab-Probe mit der EXAKTEN Konstruktion des bestehenden W1
(`gen-workloads.ts:145-184`: alternierend `matmul`/`add` über die FIXE Shape `[8,8]`, N
separate `const`-Statements) zeigt, dass die Instantiation-Zahl bei L=100/500/1000/2500/5000
KONSTANT bleibt (27.775 in allen fünf Läufen) — TS cached die identische Instantiierung
`MatMul<[8,8],[8,8]>`/`Broadcast<[8,8],[8,8]>` über alle Call-Sites hinweg, weil jeder Schritt
dieselben literalen Operanden-Shapes hat. Eine ZWEITE Probe mit derselben Op-Folge als EIN
verschachtelter Ausdruck (`x.matmul(w).matmul(w)…`, kein Zwischenschritt-Name) zeigt dasselbe:
konstant 27.452 Instantiations bis N=2000, NIE „excessively deep". Chain-Tiefe (in dieser
Konstruktion) stresst also NICHT die dokumentierte Rekursionstiefen-Grenze (ca. 100
nicht-tail-rekursiv / ca. 1000 tail-rekursiv) — sie stresst reine Datei-DURCHSATZ-Kosten
(Checkzeit wuchs ca. linear mit der Statement-Zahl: 0,016 s bei L=100 auf 0,903 s bei L=5000,
isolierter `tsc`-Batch-Lauf, keine LSP-Zeit). Diese Achse bleibt darum bewusst eine
DURCHSATZ-Achse, nicht die Rekursionstiefen-Achse — letztere wird ehrlich durch Achse (c)
abgedeckt, wo die Rekursion GENUIN pro Rang (pro Tupel-Position, INNERHALB eines einzelnen
Aufrufs) statt pro Chain-Schritt läuft (siehe `BroadcastAcc`, `spike/src/broadcast.ts:20-35`,
explizit als Akkumulator-Form für TS-Tail-Call-Elimination geschrieben). Diese Umdeutung ist
KEINE Abweichung von Owner-Entscheidung 1 (alle vier Achsen bleiben, „Chain-Tiefe" bleibt
Achse (b)) — sie präzisiert nur, WAS an dieser Achse tatsächlich gemessen wird, mit echten
Zahlen belegt statt behauptet.

**Konstruktion:** identisch zu W1 (`gen-workloads.ts:145-184`) fortgesetzt — alternierend
`matmul(w)`/`add(bias)` über feste `[8,8]`-Shapes, ein Hover an der letzten Zeile.

**Sweep-Punkte:** `{100, 250, 500, 1000, 2500, 5000, 10000}` — **in ZWEI Unter-Serien**
(bindend seit v2, siehe D20).

**D20 — Die shape-variierende Kette ist bindende Unter-Serie (v2, ersetzt §14 Frage 4).**
Achse (b) wird zweimal gefahren: **(b-fix)** die bestehende W1-Konstruktion mit fester
`[8,8]`-Shape, und **(b-variabel)** eine Kette mit pro Schritt frischen, nie wiederkehrenden
Operanden-Shapes, sodass keine zwei `MatMul<>`-Instanziierungen der Kette identisch sind.
Beide Verifier haben (b-variabel) unabhängig voneinander gebaut und gemessen: **lineares**
Wachstum, kein Cliff — `100 → 53.729`, `500 → 159.729`, `1000 → 292.229`, `2000 → 557.229`,
marginal konstant ca. **265 Instantiations pro Kettenglied** (die zweite Messreihe liegt
durchgehend 25 Instantiations darunter, reine Fixture-Differenz, identische Marginale).
Warum bindend statt „im Ergebnis-Doc erwähnen": Die publizierte Achse-(b)-Aussage wäre auf
(b-fix) allein **„konstant"** — und das gilt nur für die künstlichste denkbare Konstruktion.
Die ehrliche Aussage lautet „linear, ca. 265 Instantiations pro Kettenglied", und die trägt
nur (b-variabel). Die strukturelle Umdeutung dieser Achse (Ketten akkumulieren KEINE
Rekursionstiefe — der Cliff sitzt auf Achse (c)) wird durch (b-variabel) nicht widerlegt,
sondern gestärkt: auch die adversariale Konstruktion zeigt keine versteckte
Tiefen-Kopplung, sondern sauber lineare Kosten.

- `100`: Kontrollpunkt, reproduziert den heutigen W1-L=100-Pin.
- `250`…`5000`: Progression, an den Vorab-Probe-Messpunkten ausgerichtet (direkt
  reproduzierbar/vergleichbar).
- `10000`: Extrapolationspunkt — bei linearer Fortsetzung der Vorab-Probe-Kurve
  (ca. 0,00018 s/Statement) läge die ISOLIERTE Checkzeit bei ca. 1,8 s; das eigentlich
  interessante ist, ob die ECHTE LSP-M1-Kaltmessung (Editor-Öffnen, nicht nur `tsc --noEmit`)
  bei dieser Größenordnung in einen für Nutzer spürbaren Bereich (Sekunden) kippt.

**Vermuteter Cliff:** kein Instantiation-Wall (siehe Vorab-Probe); der vermutete Cliff ist
M1 (Kalt-Diagnostik) bei sehr großen Einzeldateien — die editor-erlebbare Schwelle, nicht das
Typ-Budget.

### 2.3 Achse (c) — Rang über die heutigen 16 hinaus

**Vorab-Code-Prüfung (Owner-Auflage):** Es existiert KEINE Rang-Obergrenze im Typ-Layer oder
im Default-`NDArray`-Laufzeitpfad — `grep -rn "MAX_RANK" spike/src/*.ts` liefert null
Treffer; `runtime.ts` (z. B. `spike/src/runtime.ts:219-222`) nutzt überall `shape.length`
ungeprüft, keine Rang-Grenze. Der Default-`NDArray`-Pfad rechnet in reinem JS (README.md:224,
COVENANT M5) — Rang ist dort effektiv unbegrenzt darstellbar. Es EXISTIERT eine harte
Rang-Obergrenze, aber NUR im Rust/WASM-ABI: `MAX_RANK = 32` (`crates/core/src/shape.rs:13`),
durchgesetzt in `validate_rank` (`crates/core/src/abi.rs:159`), NUR erreichbar über
`backend("wasm")`/`backend("threaded")`. Da diese Scheibe ausschließlich den TYP-Layer und
Editor-Latenz misst (keine Backend-Ausführung, keine `backend()`-Aufrufe in den generierten
Workloads — reine `NDArray.zeros([...]).add(...)`-artige Konstrukte wie in W2), greift diese
32er-Grenze NICHT ein; Sweep-Punkte über 32 hinaus sind typtechnisch voll gültig, wären aber
über `backend("wasm")` zur Laufzeit abgelehnt. Das wird im Ergebnis-Doc als ausdrücklicher,
belegter Vorbehalt geführt: „diese Ränge sind im TYP-Layer/Default-Backend gültig; ein Nutzer,
der `backend('wasm')` wählt, ist auf Rang ≤ 32 begrenzt — eine andere, in dieser Scheibe nicht
gemessene Grenze."

**Konstruktion:** identisch zu W2 (`gen-workloads.ts:193-235`) fortgesetzt — Broadcast-`add()`
mit alternierender Größe-1-Achse (Achse k: `dimValue = 2 + (k % 5)` auf A wenn k gerade, auf B
wenn k ungerade), garantiert sauber broadcastbar per Konstruktion.

**Sweep-Punkte:** `{16, 32, 64, 128, 256, 512, 768, 896, 1024}`.

- `16`: Kontrollpunkt, reproduziert den heutigen W2-Rang=16-Pin.
- `32`: **doppelt markiert** — sowohl die Fortsetzung der Verdopplungs-Reihe als auch der
  WASM-ABI-`MAX_RANK`-Referenzpunkt (siehe oben); im Ergebnis-Doc mit Label „= WASM-ABI
  MAX_RANK" versehen, OHNE dass das die Typ-Layer-Messung an dieser Stelle verändert.
- `64`…`512`: Verdopplungs-Progression, konsistent mit dem bestehenden Sweep-Stil (W2 nutzt
  bereits `{2,4,8,16}`, hier fortgesetzt).
- `768`/`896`/`1024`: **kein Verdopplungsschritt, sondern eine gezielte Bracketing-Serie um
  einen in der Vorab-Probe TATSÄCHLICH GEFUNDENEN Cliff** (siehe unten) — platziert um die
  dokumentierte ca. 1000-Tail-Rekursions-Grenze (CLAUDE.md „Key TS limits", TS PR #45711).

**Vorab-Probe-Befund (echter, reproduzierter Cliff):** Dieselbe W2-Broadcast-Konstruktion,
gegen den echten Typ-Layer typgeprüft (siehe „Was empirisch geprobt wurde" unten), zeigt eine
klar überlineare Instantiation-Kurve — `16→28.646`, `32→30.966`, `64→38.678`, `128→66.390`,
`256→170.966`, `512→576.726` (Checkzeit 0,021 s → 0,404 s) — und bei `768→1.243.990`
(Checkzeit 1,146 s) **kippt sie bei `1024` tatsächlich in einen echten Fehler**: `tsc` bricht
mit *„Type instantiation is excessively deep, possibly infinite"* ab (2.170.608 Instantiations
bis zum Abbruch, Speicherverbrauch ca. 973 MB). Das ist die harte, dokumentierte
TS-Rekursionsgrenze (`BroadcastAcc` ist als Akkumulator/Tail-Rekursion geschrieben,
`broadcast.ts:10-15` — konsistent mit der ca. 1000-Iterationen-Grenze für tail-rekursive
Formen). **Dieser Fund ist der stärkste Beleg der gesamten Vorab-Probe**: ein echter,
reproduzierbarer Cliff existiert, er liegt in der erwarteten Größenordnung, und Rang (nicht
Chain-Tiefe) ist die Achse, die ihn trifft — weil `BroadcastAcc` GENUIN einmal pro
Tupel-Position innerhalb eines EINZIGEN Aufrufs rekursiert, während Chain-Schritte
unabhängige, cachebare Aufrufe sind (siehe 2.2). Rang 1024 hat NULL praktische Relevanz für
reale NumPy-artige Arrays (typische Ränge liegen bei ≤ 10) — der Fund ist ein ehrliches
Ergebnis über die MASCHINE, nicht über den Nutzungsfall.

### 2.4 Achse (d) — Editor-Latenz KALT im Groß-Korpus

**Design-Entscheidung:** Achse (d) ist KEINE eigenständige fünfte Sweep-Zahlenreihe, sondern
eine QUERSCHNITTS-Metrik, die an JEDEM Sweep-Punkt der Achsen (a)-(c) zusätzlich erfasst wird
— genau wie M1 im bestehenden `editor-latency.ts` bereits an JEDEM Workload (nicht nur einem
speziellen) gemessen wird (`measureWorkload`, `editor-latency.ts:377-435`: M1 läuft auf einem
FRISCHEN Server-Prozess vor M2-M4, für jeden Eintrag im Manifest). D7 (diese Entscheidung):
**M1 (Cold: `initialize`-Roundtrip, `didOpen` → erste Pull-Diagnostik, Load-Proof-Hover) wird
an JEDEM Sweep-Punkt aller drei anderen Achsen gemessen** (frischer `tsc --lsp --stdio`-Prozess
pro Punkt, exakt wie M1 heute), mit dem oberen Ende jeder Achse (Achse a: 250 Dateien; Achse
b: 10.000 Chain-Schritte; Achse c: Rang 1024/der letzte NICHT abstürzende Punkt) als
Schlagzeilen-Datenpunkt für „Editor-Latenz kalt im Groß-Korpus" — konsistent mit der
wörtlichen Owner-Formulierung „kalt IM Groß-Korpus" (an der Spitze jeder Achse, nicht als
vierte unabhängige Größe). Diese Konstruktion vermeidet eine künstliche vierte
Sweep-Zahlenreihe, deren Punkte nur eine Neuformulierung der ersten drei wären.

## 3. Korpus-Platzierung und Pin-Schutz

**D1 — Ablage:** Der volle Sweep-Korpus liegt unter `spike/bench-dx/scale-workloads/`,
generiert von einem NEUEN Skript `spike/bench-dx/gen-scale-workloads.ts` (Schwesterdatei von
`gen-workloads.ts`, gleiches Muster). Analog zum bestehenden Präzedenzfall
`spike/bench-dx/workloads/` (generiert, gitignored, per `.gitignore`-Kommentar „Spike 02:
generated editor-latency workload files … regenerates on every run" und per Root-`tsconfig.json`
`exclude`-Eintrag geschützt, `tsconfig.json:14`) wird `spike/bench-dx/scale-workloads/`
**generiert + gitignored + aus dem Root-Korpus exkludiert**:
- NEUE `.gitignore`-Zeile `spike/bench-dx/scale-workloads/` (gleicher Kommentar-Stil).
- NEUER Root-`tsconfig.json`-`exclude`-Eintrag `"spike/bench-dx/scale-workloads"` (fünfter
  Eintrag neben den bestehenden vier, `tsconfig.json:14`).

**D2 — Beweis, dass die Root-/Stress-/Browser-Pins sich NICHT durch den SWEEP-KORPUS bewegen:**
Der `exclude`-Eintrag bedeutet, dass `pnpm check`/`check:diag` den generierten Sweep-Korpus
NIE sehen — dieselbe mechanische Garantie, die heute schon `spike/bench-dx/workloads/` schützt
(verifiziert: `tsconfig.json:13-14` listet `spike/bench-dx/workloads` bereits im `exclude`;
der neue Eintrag ist strukturell identisch). Der Beweis ist NICHT „behauptet", sondern
mechanisch: ein `exclude`-Pfad kann laut TS-Semantik nicht in ein `include: ["spike"]`-Programm
hineinrutschen, unabhängig davon, wie groß der generierte Baum wird.

**D3 — Beweis für die ZWEI NEUEN QUELLDATEIEN (Generator + Runner):** Anders als der generierte
OUTPUT landen die beiden neuen SKRIPTE selbst (`gen-scale-workloads.ts`, `scale-latency.ts`,
§4) in `spike/bench-dx/` — einem Verzeichnis, das VOM Root-`include: ["spike"]` erfasst wird
und NICHT im `exclude` steht (die vier/fünf Excludes zielen auf Unterverzeichnisse, nicht auf
`spike/bench-dx/` selbst — `editor-latency.ts`/`gen-workloads.ts` sind heute schon Teil des
Root-Korpus). Das bedeutet: **die beiden neuen Dateien VERSCHIEBEN den Root-Pin 201.455 @ 137**
(CLAUDE.md „Aktuelle Pins & Gates") — das ist eine ERWARTETE, keine versehentliche Änderung.
Nach den Mess-Regeln (CLAUDE.md „Mess-Regeln (tragend)") ist diese Verschiebung per
**empty-then-fill-Protokoll** zu dekomponieren: zwei leere `export {}`-Platzhalterdateien
zuerst anlegen + `check:diag` messen (isoliert Order-Noise, ±ca. 2.000, von echten
Typkosten), DANN befüllen + erneut messen (die Differenz ist die echte Typkosten-Komponente).
Der neue Root-Pin wird 2× deterministisch gemessen und im Ergebnis-Doc mit genau dieser
Dekomposition dokumentiert — kein Pin wird ohne diesen Beweis „einfach neu gesetzt". Analog für
`check:diag:stress`/`check:diag:browser`: BEIDE Korpora importieren NICHTS aus
`spike/bench-dx/` (verifiziert: `spike/tests-stress/tsconfig.json` inkludiert nur `"."` +
`../src/ambient.d.ts`; `spike/tests-browser` ist eine eigene, unabhängige `.emit`-Struktur) —
ihre Pins (106.398 @ 82 / 2.142 @ 75) bleiben unberührt, mechanisch garantiert durch dieselbe
Include-Isolation, kein Messbedarf.

**D4 — Committet oder generiert?** Der Sweep-KORPUS (die eigentlichen Workload-`.ts`-Dateien
unter `scale-workloads/`) ist **generiert + gitignored**, NICHT committet — Determinismus des
Generators (D9) macht das sicher reproduzierbar, exakt das etablierte Muster von
`spike/bench-dx/workloads/`. Die GENERATOR-/RUNNER-SKRIPTE selbst SIND committeter Quellcode
(wie `gen-workloads.ts`/`editor-latency.ts` es sind).

**Z2-Konsequenz (Covenant, ausführlich in §11):** Die Analogie zum Präzedenzfall
`spike/bench-dx/workloads/` trägt für „kein committeter generierter Korpus rottet, weil es
keinen committeten Korpus gibt" — aber sie BRICHT an einem Punkt: `spike/bench-dx/workloads/`
wird bei JEDEM `pnpm bench:editor`-Lauf regeneriert und durchgecheckt, und dieser Lauf ist Teil
des HARTEN CI-Gates `editor-gate` (`.github/workflows/ci.yml:156-168`) — bei JEDEM Push/PR.
Der volle Sweep-Korpus dagegen läuft laut Owner-Entscheidung 2 **NICHT in CI**, nur on-demand.
Das heißt: sollte der GENERATOR selbst kaputtgehen (z. B. still leere Dateien erzeugen), würde
das NICHT durch einen kontinuierlichen Lauf auffallen — anders als bei `gen-workloads.ts`. Der
mechanische Ersatz dafür ist der gepinnte Sentinel (§6, LÄUFT in CI) plus der
Nicht-Vakuitäts-Beweis in §9 (beweist EINMALIG, bei Fertigstellung, dass der Generator echten
Code erzeugt) — das schließt die Lücke NICHT vollständig (kein Dauer-Backstop für den vollen
Sweep-Generator selbst), sondern reduziert sie auf ein bewusst akzeptiertes, offengelegtes
Residualrisiko (Owner-Entscheidung 2 nimmt genau das explizit in Kauf: „Der volle Sweep ist ein
eigener Befehl, NICHT in CI").

## 4. Generator + Runner: Design und Wiederverwendungs-Entscheidung

**D5 — Wiederverwendungs-Entscheidung: DUPLIZIEREN, nicht extrahieren, nicht parametrisieren.**
Der Sweep braucht denselben hand-gerollten LSP-Client (`createLspClient`,
`editor-latency.ts:149-274`, JSON-RPC/Content-Length-Framing über stdio, Server→Client-Request-
Beantwortung für `workspace/configuration`) und dieselben Stats-/Narrowing-Helfer (`statsOf`,
`hoverText`, `diagnosticItems`, `editor-latency.ts:102-307`). Drei Optionen standen zur Wahl:

- **(a) Duplizieren** in eine neue Datei `spike/bench-dx/scale-latency.ts`.
- **(b) Extrahieren** in ein geteiltes Modul (z. B. `spike/bench-dx/lsp-harness.ts`), das
  BEIDE Skripte importieren.
- **(c) Parametrisieren** des bestehenden `editor-latency.ts`/`gen-workloads.ts` um einen
  Sweep-Modus.

**Entscheidung: (a).** Begründung, mit explizitem Risiko-Abwägen gegen das bestehende harte
Gate (Owner-Auftrag):

1. `editor-latency.ts`/`gen-workloads.ts` tragen den EXAKT-Match-Instantiation-Pin-Katalog
   w1-w7, der `check:diag`-artig hart gegated ist (`enforceHardGate`,
   `editor-latency.ts:785-832`). Diese Scheibe braucht an DIESEN Dateien **NULL Änderungen**
   für den Sweep-Teil (nur EINE additive Änderung für den Sentinel, §6 — strukturell getrennt).
   Eine Extraktion (Option b), selbst als reine, verhaltensfreie Code-Verschiebung, macht den
   Diff dieser gate-kritischen Dateien nicht-leer und verlangt damit den vollen
   Verifikations-Aufwand (2× Neu-Messung, Baustein-A/B-Prüfung) für eine Datei, die für DAS
   eigentliche Ziel dieser Scheibe (die Sweep-Messung) gar nicht angefasst werden muss.
2. Das Projekt hat dieses Muster bereits: `spike/src/dim.ts:49-60` dupliziert `IsUnion`
   BEWUSST statt es aus `literal-arithmetic.ts` zu importieren, mit exakt derselben
   Begründungsform („Duplicated here — NOT imported — because … a reverse import would risk
   a … cycle" — hier: weil eine Extraktion Risiko für ein FIXES, hart gegatetes File erzeugt,
   das keinen Grund hat, sich zu bewegen). Direktes, quellenverankertes Präzedenzfall-Zitat.
3. Der LSP-Client ist protokollstabil (JSON-RPC/Content-Length über stdio, `tsc --lsp --stdio`)
   — die Duplizierungskosten sind eine begrenzte, einmalige Fläche (ca. 150-200 Zeilen), kein
   wiederkehrendes Wartungsrisiko.

**Prüfbares Kriterium (T-Kriterium, siehe auch §12):** `git diff` auf `editor-latency.ts` und
`gen-workloads.ts` zeigt nach dieser Scheibe **ausschließlich** die additive Sentinel-Änderung
(§6: ein neuer `buildW8()`-Aufruf + ein neuer `INSTANTIATION_PINS`-Eintrag) — keine Zeile der
bestehenden LSP-Client-/Stats-/Narrowing-Logik wird berührt. Die w1-w7-Pins bleiben exakt
`{27769, 29578, 60718, 27932, 33223, 34393, 26941}` — das ist der Beweis, stärker als eine
erneute Messung, weil die Dateien schlicht nicht editiert wurden. Eine spätere Konsolidierung
(Extraktion, sobald sich `scale-latency.ts` als stabil erwiesen hat) ist ein FOLLOWUPS-Kandidat,
NICHT Teil dieser Scheibe.

**D6 — Neue Dateien:**
- `spike/bench-dx/gen-scale-workloads.ts` — Generator, Muster identisch zu
  `gen-workloads.ts` (deterministisch, `WORKLOAD_COMPILER_OPTIONS` wird DUPLIZIERT, siehe
  §14 Frage 1), erzeugt **pro Sweep-Punkt ein eigenes Unterverzeichnis** mit seinen
  `.ts`-Dateien, einer echt benannten `tsconfig.json` und einem gemeinsamen `manifest.json`
  auf der Ebene darüber.

**D21 — Ein Unterverzeichnis mit echter `tsconfig.json` pro Sweep-Punkt (bindend, v2, aus
der Frontier-Zweitmeinung).** Ein per-Punkt-Isolat namens `tsconfig.<achse>-<punkt>.json`
isoliert NUR den `tsc -p`-Zählpfad (`measureInstantiations`, `editor-latency.ts:557-587`) —
für die LATENZ-Messung ist es wirkungslos: Der Sprachserver discovert das Projekt einer
geöffneten Datei über die nächstgelegene Datei, die **`tsconfig.json` heißt**
(`gen-workloads.ts:3-7,628-632`; M1 = `didOpen` + Pull-Diagnostik auf dem discoverten
Projekt, `editor-latency.ts:400-435`). Lägen alle Sweep-Punkte flach in EINEM Verzeichnis
unter einem gemeinsamen `include: ["*.ts"]`, lüde die M1-Messung des kleinsten Punkts das
Programm ALLER Punkte — inklusive der Rang-1024-Datei, die den Checker abbricht. Die
gemessene „Kalt-Latenz bei N=10" wäre dann in Wahrheit die Kalt-Latenz des gesamten Sweeps.
Deshalb: ein Verzeichnis pro Sweep-Punkt, jeweils mit eigener `tsconfig.json` (die
`files`/`include`-Liste nennt genau die Dateien dieses Punkts plus `../../../src/ambient.d.ts`
nach D7). Beide Vorgänger-Verifier haben dieses Loch nicht gesehen; es ist rein
konstruktiv und vor dem Bau billig zu schließen.
- `spike/bench-dx/scale-latency.ts` — Runner, Muster identisch zu `editor-latency.ts` (M1/M2/M4
  je Sweep-Punkt, M3 nur bei Sweep-Punkten mit Toggle-Ziel — siehe §9), OHNE `enforceHardGate`
  (informativ, siehe §7) — druckt Tabellen + schreibt `scale-workloads/results.json`
  (gitignored) für die Ergebnis-Doc-Autorin.

**D7 (Baustein-0-Pflicht):** Sowohl der Generator als auch der Runner MÜSSEN das Vorab-Proben-
Ergebnis aus §2/§3 übernehmen: **jedes generierte per-Sweep-Punkt-`tsconfig.<id>.json`
MUSS `spike/src/ambient.d.ts` explizit in seiner `files`-Liste führen** (siehe „Was empirisch
geprobt wurde" unten — ohne diesen Eintrag schlägt der Typcheck an `process`/
`node:fs/promises`/`node:worker_threads`/`node:os` in `wasm/backend-api.ts`, `wasm/loader.ts`,
`wasm/threaded.ts` fehl, weil `NDArray` diese Module real bzw. typ-only importiert,
`spike/src/ndarray.ts:53,55`). Dies ist eine BINDENDE Design-Vorgabe (kein optionaler Hinweis)
— das bestehende Muster in `gen-workloads.ts:648` (`files: [w.fileName]`, KEIN
`ambient.d.ts`-Eintrag) darf NICHT unbesehen kopiert werden; siehe „Beobachtungen außerhalb des
Scope" für die Frage, ob das bestehende Muster selbst betroffen ist.

**D8 — Befehlsname & Ausgabe:** `pnpm bench:scale`, definiert als
`"node spike/bench-dx/gen-scale-workloads.ts && node spike/bench-dx/scale-latency.ts"`
(Muster identisch zu `bench:editor`, `package.json:70`). Ausgabe: Konsolen-Tabellen im Stil von
`printM1Table`/`printM2Table`/`printInstantiationTable` (`editor-latency.ts:593-650`), je
Achse gruppiert (KEINE gemischte Tabelle über Achsen hinweg — siehe §7 Populationsregel), plus
ein `results.json`-Dump (gitignored, unter `scale-workloads/`) mit allen Rohsamples für die
Ergebnis-Doc-Autorin zum Transkribieren echter Zahlen. **`results.json` erfasst je Sweep-Punkt
auch den Speicherverbrauch** („Memory used" aus dem `--extendedDiagnostics`-Footer, v2): am
Cliff lag er bei ca. 973 MB, eine L=2000-Kette bei ca. 228 MB — Speicher bindet in dieser
Messung potenziell VOR der Zeit, und ein Sweep-Punkt, der auf einem kleineren Rechner per OOM
stirbt statt sauber „excessively deep" zu melden, muss an dieser Zahl erkennbar sein (§13,
Baustein B). **NICHT** in `pnpm check`/CI (Owner-
Entscheidung 2).

**D9 — Determinismus:** wie `gen-workloads.ts` (Datei-Header-Kommentar, Zeile 15-19): kein
`Date.now()`/`Math.random()`/Host-Zustand im generierten OUTPUT — Wiederholungsläufe erzeugen
byte-identische `.ts`/`.json`-Dateien. Begründung identisch zum bestehenden Muster: macht den
Sweep-Korpus sicher gitignorebar/regenerierbar, und macht `results.json`-Diffs zwischen zwei
Läufen auf demselben Rechner aussagekräftig (Abweichungen sind dann echtes Mess-Rauschen, nicht
Generator-Rauschen).

## 5. Der Eich-Anker

**Owner-Entscheidung 3** verlangt: die Op-Mischung pro generierter Call-Site wird aus dem
echten Demo-Code abgeleitet und BELEGT dokumentiert — nicht geschätzt. Folgende Zahlen sind
`grep`-verifiziert gegen `examples/rag-demo/main.ts` (nur ECHTE Code-Aufrufe gezählt; Erwähnungen
in Kommentaren/Docstrings und ein Textvorkommen innerhalb eines Assert-Message-Strings,
`main.ts:154`, sind ausgeschlossen):

| Op | reale Aufrufe | Zeilen |
|---|---|---|
| `.item(` | **9** | 127(×2), 128, 134, 151, 211, 212, 213, 249 |
| `NDArray.fromArray(` | 10 | 42, 91, 187(×2), 210, 224, 243(×2), 246(×2) |
| `.slice(` | 4 | 125, 156, 157, 208 |
| `.cosineSimilarity(` | 3 | 158, 208, 225 |
| `.matmul(` | 2 | 110, 243 |
| `.mul(` | 2 | 50, 100 |
| `.div(` | 2 | 62, 105 |
| `.sum(` | 2 | 51, 101 |
| `.sqrt(` | 2 | 58, 103 |
| `.reshape(` | 2 | 61, 104 |
| `.topk(` | 2 | 126, 210 |
| `.dot(` | 2 | 159, 246 |
| `.mean(` | 1 | 196 |
| `.transpose(` | 1 | 110 |
| `NDArray.stack(` | 1 | 187 |

Summe (ohne `fromArray`, das ein Konstruktor statt eine Op ist): **35** reale Op-Aufrufe.
`item` ist mit 9/35 (ca. 26 %) die dominante Einzel-Op — konsistent mit
`docs/op-w5-item-spec.md`s Einordnung von `item` als letztem, aber real genutztem
Wunschlisten-Eintrag. Reihenfolge-Ableitung für die generierten Call-Sites: `item` > `slice` >
`cosineSimilarity` > {`matmul`,`mul`,`div`,`sum`,`sqrt`,`topk`,`dot`} > {`mean`,`transpose`,
`stack`}.

**D10 — Umsetzung in den generierten Op-Mix:** Jede generierte „realistische" Datei (Achse a,
und die Chain-Konstruktion aus Achse b, sofern sie NICHT die bewusst reine
matmul/add-Konstruktion von W1 fortsetzt) reproduziert diese relative Häufigkeits-Ordnung
NÄHERUNGSWEISE — exakte Reproduktion der 9:10:4:3:2:2:2:2:2:2:2:1:1:1-Verhältnisse bei
KLEINEN Sweep-Punkten (z. B. 1-10 Dateien) ist unrealistisch; die Bindung ist „die drei
häufigsten Ops (`item`, `fromArray`, `slice`) erscheinen in jeder generierten Datei mindestens
so oft wie die am seltensten genutzten (`mean`, `transpose`, `stack`)", geprüft per
Selbstbericht des Generators (zählt seine eigenen erzeugten Aufrufe und druckt die
Verhältnis-Tabelle neben den Sweep-Ergebnissen). Achse (c) (Rang) und die W1-Fortsetzung von
Achse (b) bleiben bei der bestehenden reinen `add`/`matmul`-Konstruktion (§2.2/§2.3) — der
Eich-Anker gilt für Achse (a) (Korpusgröße), wo „realistischer Op-Mix" der Sinn der Achse ist,
nicht für die gezielt isolierenden Stress-Konstruktionen von (b)/(c).

## 6. Der gepinnte Sentinel (`w8`)

**D11 — Warum ein Sentinel:** Owner-Entscheidung 2: „Zusätzlich EIN Scale-Workload im
bestehenden `bench:editor`-Gate mit exaktem Instantiation-Pin, damit die publizierte Aussage
nicht still verrottet." Der volle Sweep läuft nur on-demand (§3, D4-Konsequenz) — ohne einen
Sentinel im HARTEN, kontinuierlich laufenden `editor-gate`-CI-Job (`.github/workflows/
ci.yml:156-168`, jeder Push/PR) könnte eine künftige Änderung die „im Maßstab tragfähig"-Aussage
unbemerkt brechen, bis irgendwann jemand den vollen Sweep von Hand erneut fährt.

**D12 — Konkrete Form:** `w8` wird als achter Eintrag NEBEN w1-w7 in `gen-workloads.ts`
generiert (`buildW8()`, angehängt an die bestehende `workloads`-Liste,
`gen-workloads.ts:626`) — EIN realistisches, aber gegenüber dem heutigen Maximum (W2 Rang 16)
verdoppeltes Szenario: **Broadcast-`add()` bei Rang 24** (zwischen dem heutigen Maximum 16 und
der WASM-ABI-`MAX_RANK`-Grenze 32 — ein bewusst „nächster realistischer Schritt", kein
Stress-Extremum) **gefolgt von einer kurzen, am Eich-Anker orientierten Op-Folge**
(`.item()`/`.slice()`/`.topk()` — die drei häufigsten Nicht-Konstruktor-Ops aus §5) auf einem
`[8, 24]`-artigen Score-Matrix-Muster, das die RAG-Demo-Form (`similarities.slice(qi)`,
`.topk(2)`, `.item(qi, docIdx)`, `main.ts:125-151`) direkt spiegelt. Zielgröße 80-150 LOC
(zwischen W4s Kleingröße und W5s 200-400 LOC, damit der 2×-Latenz-Ceiling und die
CI-Laufzeit-Auswirkung klein bleiben, §7/§12). Exakte literale Dimensionen sind
Implementierer-Wahl (Baustein 0 kann sie verbindlich festlegen) — die STRUKTUR (Rang 24 +
item/slice/topk-Folge) ist bindend.

**D22 — w8 trägt ein Toggle-Ziel (bindend, v2, aus der Frontier-Zweitmeinung).** Der
Produkt-Claim des Projekts ist Editor-Feedback **beim Tippen**; die dafür einschlägige
Metrik ist M3 (`didChange` → Diagnostik, `editor-latency.ts:456-512`). In v1 trug KEIN
Sweep-Punkt und auch der Sentinel kein Toggle-Ziel — die inkrementelle Latenz wäre also im
gesamten Vorhaben ungemessen geblieben, obwohl der Harness sie kann. Deshalb bekommen
**w8 UND der oberste Punkt der Achse (a)** (250 Dateien, beide Unter-Serien) je ein
Toggle-Ziel nach dem etablierten W4/W6-Mechanismus (zwei Volltext-Zustände, gegattet auf
das exakte (Zeile, Diagnose-Code)-Paar). Die 2×-Ceiling-Schwelle für M3 (`TOGGLE_GATE_MS`
= 500, also 1000 ms hart) gilt für w8 wie für die bestehenden Toggle-Workloads.

**D13 — Pin-Mechanik:** Nach Implementierung wird w8s Instantiation-Zahl 2× deterministisch
gemessen und als neuer Eintrag `w8: <Zahl>` in `INSTANTIATION_PINS`
(`editor-latency.ts:775-783`) ergänzt — nach demselben Muster, das dort für w1-w7 bereits
etabliert ist (Kommentarblock „Re-measured … (Op-Scheibe …): …", `editor-latency.ts:687-774`).
Da `measureWorkload`/`printM1Table`/…/`enforceHardGate` bereits GENERISCH über
`manifest.workloads` iterieren (kein Workload-spezifischer Code-Zweig existiert für w1-w7 —
verifiziert durch Lesen von `editor-latency.ts:377-536`, `652-832`), erfordert die Aufnahme von
w8 in die M1/M2/M4/Hard-Gate-Messung **keine neue Logik**, nur den einen neuen Manifest-Eintrag
(aus `gen-workloads.ts`) + den einen neuen Pin-Eintrag.

**D14 — Beweis, dass w8 die w1-w7-Pins NICHT bewegt:** Jeder Workload bekommt sein EIGENES
isoliertes `tsconfig.<id>.json` mit `files: [entry.fileName]` (Einzeldatei-Isolation,
`gen-workloads.ts:643-648`) — w8s Instantiation-MESSUNG (`measureInstantiations`,
`editor-latency.ts:557-587`) läuft also in einem eigenen `tsc`-Prozess, der NUR `w8-*.ts` +
(nach D7) `ambient.d.ts` sieht, nicht w1-w7s Dateien. Die einzige gemeinsame Datei ist das
gemeinsame `workloadsDir/tsconfig.json` (`include: ["*.ts"]`, `gen-workloads.ts:632`), das NUR
für die LSP-Server-Projekt-Discovery beim Öffnen einer Datei greift (M1-M4-Messung, nicht die
Instantiation-Zählung) — ein LSP-Server, der `w8-scale.ts` öffnet, lädt zwar alle
Geschwister-Dateien ins selbe TS-Programm (das ist bereits heute für w1-w7 untereinander der
Fall), aber das ändert NICHT deren `check:diag`-artige Instantiation-ZÄHLUNG, die exklusiv über
die isolierten Einzeldatei-`tsconfig`s läuft. w1-w7s M1-M4-LATENZ-Werte könnten sich durch ein
größeres gemeinsames LSP-Projekt geringfügig verschieben (mehr Dateien im selben Serverprozess)
— das ist die eine tolerierte, im Ergebnis-Doc zu berichtende Nebenwirkung, KEINE
Instantiation-Pin-Verschiebung; das T-Kriterium (§4/§12) bezieht sich explizit nur auf die
Instantiation-Pins, nicht auf Latenz-Mediane (die ohnehin nur am 2×-Ceiling, nicht exakt,
gegated sind, `enforceHardGate`, `editor-latency.ts:792-805`).

**D15 — CI-Laufzeit-Auswirkung:** `editor-gate` hat heute `timeout-minutes: 15`
(`ci.yml:159`); ein achter Workload mit M1 (1 Kaltmessung) + M2 (ca. 3 Hover-Positionen ×
23 Samples) + M4 (23 Samples) + M3 (Toggle, D22) fügt, an der Größenordnung
der bestehenden Workloads gemessen (Spike-02-Ergebnisse: Gesamtlauf ca. 1,2 s lokal für SIEBEN
Workloads inkl. `check:diag`-Läufen), geschätzt **weniger als 1 Sekunde** zur lokalen Laufzeit
hinzu — der 15-Minuten-CI-Timeout hat massiven Puffer; keine Timeout-Anpassung nötig. Diese
Schätzung wird im Ergebnis-Doc durch die tatsächlich gemessene Vorher/Nachher-Laufzeit ersetzt
(kein Ratespiel im Endergebnis, nur hier als Vorab-Einschätzung).

## 7. Vorregistrierte Gates

Nach Hausregel (CLAUDE.md „Qualitätssicherung", Budget-Gate-Lektion aus Spike 04): Gates auf
ABSOLUTE, nutzer-erlebbare Schwellen legen — **niemals** auf eine Schätzung der Zahl, die
gerade erst gemessen werden soll. Populationen werden NICHT gemischt (kein Mittelwert über
realistische Achse-(a)-Punkte und Stress-Achse-(c)-Punkte).

**G1 — Sweep-Wall-Time-Budget (absolut):** Der VOLLSTÄNDIGE Sweep über alle Achsen darf auf
der Referenzmaschine (§8) **30 Minuten Wall-Clock** nicht überschreiten. Begründung: (a)
deutlich unter dem höchsten bestehenden CI-Job-Timeout des Projekts (`test-threaded`,
45 min, `ci.yml:107`, als oberer Referenzpunkt für „was dieses Projekt als vertretbar
akzeptiert"), (b) läuft on-demand (nicht CI), sodass eine Person, die den Befehl manuell
startet, bis zu ca. 30 min tolerieren kann, ohne dass Iteration unpraktikabel wird — vergleichbar
mit `bench:crossover`, ebenfalls ein manuelles, nicht CI-gegatetes Kalibrierungs-Skript. **Bei
Überschreitung:** Sweep-Punkte REDUZIEREN (z. B. den obersten Stress-Punkt einer Achse
weglassen), dokumentiert im Ergebnis-Doc mit Begründung — NIEMALS die Budget-Latte selbst
verschieben, um einen zu langsamen Lauf nachträglich als „fertig" zu deklarieren.

**G2 — Sentinel (`w8`): harter Gate-Katalog wie w1-w7.** Exakter Instantiation-Pin-Match
(`enforceHardGate`), 2×-Latenz-Ceiling für M2 (200 ms) — dieselbe Disziplin wie die
bestehenden sieben, weil w8 eine STABILE, dauerhafte Erweiterung des bestehenden Gates ist,
kein Sweep-Punkt, der etwas Unbekanntes messen soll (§6 hat die Zahl bereits VOR der Messung
strukturell fixiert; nur der exakte Instantiation-WERT wird nachträglich als Pin
festgeschrieben — konsistent mit der Hausregel, weil der Pin die REGRESSION schützt, nicht die
Achse-c-Cliff-Suche vorwegnimmt).

**G3 — Voller Sweep: NUR Korrektheits-Gates hart, KEINE Performance-Schwellen hart.** Für jeden
Sweep-Punkt gilt dieselbe „nie ein falsches Ergebnis timen"-Disziplin wie im bestehenden Harness
(`assertHoverCorrect`, `editor-latency.ts:367-375`: der Hover-Text MUSS die erwartete Shape
enthalten). Für Achse-(c)-Punkte, an denen `tsc` erwartbar mit „excessively deep" abbricht
(Punkte ≥ der gefundenen Cliff-Region, siehe §2.3), ist das ABBRECHEN selbst das erwartete,
korrekte Ergebnis — der Runner MUSS diesen Fall explizit als „Cliff getroffen" klassifizieren
und LOGGEN, statt ihn als Skript-Fehler zu werfen (siehe §9, Nicht-Vakuität: ein Skript, das
bei JEDEM Rang lautlos „OK" meldet, wäre vakuös). Es gibt **keinen** harten Pass/Fail-Schwellwert
auf Latenz oder Instantiation-Zahl für die Sweep-Punkte selbst — genau diese Zahlen SIND das
Messergebnis.

**D23 — Cliff-Isolation ist Mechanismus, nicht nur Absicht (bindend, v2, aus Baustein 0).**
G3 fordert das Ergebnis („Cliff getroffen" klassifizieren statt werfen), v1 benannte aber
keinen Mechanismus — und D5 („1:1 duplizieren") bringt genau den falschen mit: im Vorbild
wirft `assertHoverCorrect` synchron (`editor-latency.ts:367-375`), die Workload-Schleife hat
KEIN try/catch um den einzelnen `measureWorkload`-Aufruf (`editor-latency.ts:861-869`), und
der einzige Fänger `main().catch(...)` rethrowt (Zeile 887) — das beendet den Prozess. Der
Cliff-Abbruch kommt hart und schnell (ca. 1,9 s, TS2589 + TS2769, Exit 1). Unverändert
übernommen hieße das: der ERSTE Cliff-Treffer reißt alle noch nicht gemessenen Sweep-Punkte
ALLER Achsen desselben Laufs mit. Bindend ist deshalb: (1) **try/catch pro Sweep-Punkt** —
ein fehlgeschlagener Punkt wird als Ergebniszeile („Cliff", mit `tsc`-Diagnose-Code und
Exit-Code) erfasst, der Sweep läuft weiter; (2) die Cliff-Klassifikation erfolgt **vor** dem
korrektheitsgegateten Hover, aus der `tsc`-Diagnose des Punkts, nicht aus einem geworfenen
LSP-Fehler; (3) ein Punkt, der aus einem ANDEREN Grund scheitert (OOM, Timeout, Absturz des
Serverprozesses), wird als eigene Kategorie geführt und NIE als „Cliff" gezählt.

**G4 — Populationsregel:** Ergebnis-Tabellen gruppieren strikt je Achse (wie
`printM2Table`/`printInstantiationTable`, `editor-latency.ts:604-650`, bereits je Workload
gruppieren) — kein „Durchschnitt über alle Sweep-Punkte", kein gemischtes Diagramm, das
realistische (Achse a bei kleinem N) und Stress-Punkte (Achse c bei Rang 1024) in eine Kennzahl
verrechnet.

## 8. Mess-Randbedingungen

- **Hardware (diese Session, Vorab-Probe):** Apple M3, 8 Kerne, 16 GB RAM, macOS 26.5.2
  (Build 25F84), arm64, Node v24.16.0 (`.nvmrc` = 24), TypeScript 7.0.2 (nativ/Go-Generation,
  `node_modules/.bin/tsc --version` → „Version 7.0.2"). Die tatsächliche Scale-Probe-Messung
  MUSS ihre eigene Host-Zeile (`uptime`, wie `hostLoadLine()`, `editor-latency.ts:313-319`)
  protokollieren — diese Angabe hier ist die Umgebung der VORAB-Probe, nicht automatisch
  identisch mit dem späteren Mess-Lauf (andere Session, evtl. anderer Host-Load).
- **Frischer `git worktree` für Baselines:** wie CLAUDE.md „Mess-Hausregel" — Root-/Stress-/
  Browser-Pin-Vergleiche VOR und NACH dieser Scheibe ausschließlich in einem frischen
  `git worktree add <scratch> <commit>`, NIE per `git stash` (lässt untracked Dateien liegen,
  kontaminiert den Korpus).
- **IMMER Exit-Code + volle Fehlerausgabe prüfen**, nie nur die Kennzahl greppen — insbesondere
  bei Achse-(c)-Punkten, wo ein NICHT-Null-Exit (der „excessively deep"-Fall) das ERWARTETE
  Ergebnis sein kann; der Runner muss diesen Fall vom „Skript ist kaputt"-Fall unterscheiden
  (siehe G3/§9).
- **Keine Vergleiche über Korpus-Grenzen hinweg** (Root vs. Stress vs. Browser vs.
  Scale-Sweep) — jede Zahl gilt nur für ihr eigenes, fixes File-Set.
- **Order-Noise-Regel:** jede neue/umbenannte Datei verschiebt den Instantiation-Counter um bis
  zu ±ca. 2.000 (reines Reihenfolge-Rauschen, keine Typkosten) — gilt für D3 (die zwei neuen
  Root-Korpus-Dateien) und wird per empty-then-fill dekomponiert.
- **macOS-arm64 vs. ubuntu-latest (CI):** Der Sentinel (w8, §6) läuft im bestehenden
  `editor-gate`-Job auf `ubuntu-latest` (`ci.yml:158`) — die w1-w7-Pins sind bereits heute
  plattformübergreifend als EXAKT reproduzierbar dokumentiert (Kommentar
  `editor-latency.ts:690-692`: „Cross-platform stability is checked by the first CI run"). Der
  volle Sweep (§2-§5) läuft NUR lokal/on-demand, NICHT in CI (Owner-Entscheidung 2) — die
  Diskrepanz-Frage stellt sich für ihn nicht; sollte der Sweep später doch in CI wandern (nicht
  Teil dieser Scheibe), wäre eine erste Ubuntu-Messung fällig, exakt wie beim
  `check-freeze-hash.mjs`-Muster (plattform-gelabelte Pin-Menge statt Einzelpin).
- **Sample-Schema:** N ≥ 20 timed Samples je Position/Richtung nach 3 Warmups (Muster
  `WARMUP_SAMPLES = 3` / `TIMED_SAMPLES = 20`, `editor-latency.ts:49-50`) für M2/M4; M1 bleibt
  ein Einzelwert pro frischem Serverprozess (wie heute); Median + Min-Max berichten, nie ein
  Einzelwert.

## 9. Testplan mit Nicht-Vakuitäts-Beweisen

**T1 — Beweis, dass der Generator echten, typprüfbaren Code erzeugt (nicht leer/trivial):**
Ein Mutant im Runner, der die Op-Aufrufe durch No-Op-Kommentare ersetzt (Backup-Kopie-Verfahren,
siehe unten), MUSS die Instantiation-Zahlen einbrechen lassen (auf einen Bruchteil des
gemessenen Werts) — belegt, dass die reale Op-Maschinerie tatsächlich gemessen wird, nicht nur
Importe/Boilerplate.

**T2 — Beweis, dass Achse-(c)-Cliff-Erkennung nicht vakuös ist:** Ein Mutant, der einen
KÜNSTLICH REDUZIERTEN Rang-Sweep-Punkt (z. B. 8 statt 1024) fälschlich als „excessively deep"
klassifiziert (invertierte Erkennungs-Logik), MUSS den Nicht-Vakuitäts-Test rot werden lassen —
zeigt, dass die Cliff-Klassifikation echte `tsc`-Ausgabe prüft, nicht geraten/hartkodiert ist.

**T3 — Beweis, dass die Korrektheits-Gates (M2 Hover-Text-Match) echte Fehlklassifikation
fangen:** wie im bestehenden Harness (`assertHoverCorrect`) — ein Mutant, der das erwartete
Shape-Substring durch ein FALSCHES ersetzt, MUSS den Lauf mit „CORRECTNESS GATE FAILED" abbrechen
(bereits etablierter Mechanismus, hier nur auf die neuen Sweep-Workloads übertragen und einmal
gegen mindestens einen Achse-a-, einen Achse-b- und einen Achse-c-Punkt bewiesen).

**T4 — Beweis für die empty-then-fill-Dekomposition (D3):** die zwei leeren
Platzhalterdateien + die befüllten Endversionen werden BEIDE gemessen und im Ergebnis-Doc als
zwei Zahlen (Order-Noise-Anteil, echter Typkosten-Anteil) berichtet — kein einzelner
Vorher/Nachher-Delta-Wert ohne diese Aufschlüsselung.

**T5 — Beweis für D14 (w8 bewegt w1-w7 nicht):** w1-w7s Instantiation-Pins werden NACH der
w8-Ergänzung 2× gemessen und MÜSSEN exakt `{27769, 29578, 60718, 27932, 33223, 34393, 26941}`
bleiben — jede Abweichung ist ein BLOCKER (nicht nur ein Befund), weil sie D14s mechanisches
Argument widerlegen würde.

**T6 — Mutanten-Revert-Disziplin (harte Arbeitsregel, CLAUDE.md):** JEDER Mutant für T1/T2 wird
NUR per inversem Edit oder Backup-Kopie-Restore (`cp` nach `/private/tmp/…/scratchpad`, zurück,
`diff`-Beweis) revertiert — **niemals** `git checkout`/`git restore` auf den generierten
Dateien oder den neuen Skripten, solange dort uncommittete Arbeit liegt. Da
`spike/bench-dx/scale-workloads/` gitignored ist, betrifft das primär die beiden neuen
QUELLDATEIEN (`gen-scale-workloads.ts`, `scale-latency.ts`) während ihrer Entstehung.

**T7 — Beweis, dass der Sweep NACH einem Cliff weiterläuft (v2, D23).** T2 beweist, dass die
Cliff-KLASSIFIKATION nicht vakuös ist — aber nicht, dass danach noch etwas gemessen wird. Der
Beweis: ein Lauf, bei dem ein FRÜHER Sweep-Punkt künstlich zum Cliff gemacht wird (Mutant per
Backup-Kopie-Verfahren, T6), MUSS anschließend alle nachfolgenden Punkte aller Achsen normal
messen und im `results.json` ausweisen. Bleibt die Ergebnisliste nach dem künstlichen Cliff
leer, ist D23 nicht umgesetzt — Blocker.

**T8 — Beweis, dass die beiden Unter-Serien wirklich verschieden sind (v2, D19/D20).** Die
Shape-Diversitäts-Serien wären wertlos, wenn der Generator versehentlich in beiden dieselben
Dims erzeugt. Beweis: (a-gleich) und (a-distinkt) MÜSSEN bei gleichem N klar verschiedene
Instantiation-Zahlen liefern (erwartete Größenordnung: Faktor mehrere, siehe D19); zusätzlich
prüft der Generator-Selbstbericht, dass die Menge der literalen Dims in (a-distinkt) über
alle Dateien PAARWEISE disjunkt ist. Gleiche Zahlen in beiden Serien = der Generator baut in
Wahrheit einmal dasselbe, Blocker. Analog für (b-fix) gegen (b-variabel): konstant gegen
linear wachsend.

## 10. Der Abschluss-Deliverable

**D16 — Exakte Fundstellen (verifiziert, `pfad:zeile`):**
- `README.md:259-262`, Qualifikation 2: *„'TypeScript can do this' means: newly tractable,
  unproven at scale.' No existing TS library has delivered general compile-time shape checking
  with broadcasting and reductions; the prior art stops at literal-dimension matmul. NumType is
  not validating a *proven* technique — it is probing the limit. That is the point of the
  research."*
- `README.md:268-270`, der Mess-Absatz direkt danach: *„The type-checker cost is measured, not
  hoped: the slice arithmetic costs [ca.]1.59x instantiations, the bounds checks [ca.]1.036x, and
  hover latency measured against the native TS 7 language server is 0.04-0.08 ms median — about
  three orders of magnitude under a 100 ms editor gate."* (Original nutzt an dieser Stelle das
  Tilde-Zeichen für „ungefähr" — hier durch „[ca.]" ersetzt, um das projekteigene
  Markdown-Verbot in DIESER Spec zu wahren; die tatsächliche README-Bearbeitung darf ihre
  eigene, README-eigene Konvention fortführen, das ist nicht Gegenstand dieser Spec-Regel.)
- `docs/wettbewerbsanalyse-und-usp.md:59`, „Drei Qualifikationen", Punkt 2 (wortgleiches Zitat
  inkl. Markdown-Auszeichnung): **„TypeScript kann es"** heißt: *neu machbar, im Maßstab
  unbewiesen*. Für ein Forschungsprojekt ist genau das der Reiz — wir validieren keine bekannte
  Technik, wir loten die Grenze aus.

**D17 — Bindende Ersetzungsregel:** Die neue Formulierung an ALLEN DREI Stellen darf
ausschließlich aus tatsächlich in dieser Scheibe GEMESSENEN Zahlen ableitbar sein — kein Wort
darf über das hinausgehen, was §2-§9 tatsächlich gemessen haben. Die neue Formulierung MUSS
die ehrlich mitgeführten Grenzen der Messung nennen (mindestens: welche Achsen-Obergrenzen
tatsächlich getestet wurden, ob/wo ein Cliff gefunden wurde und bei welcher Größenordnung, dass
„unproven at scale" NICHT durch „proven at ALL scales" ersetzt wird, sondern durch eine
Aussage der Form „gemessen bis Größenordnung X, Cliff bei Y gefunden/nicht gefunden").

**Terminologie-Klarstellung (v2):** „Diese Scheibe" meint durchgängig den vollen Bogen
Spec → Implementierung → Messung → Ergebnis-Doc → README/USP-Ersetzung. Die Ersetzung ist
also der ABSCHLUSS dieser Scheibe (Owner-Entscheidung 4), nicht die einer späteren — sie
kann nur naturgemäß erst formuliert werden, wenn die Messung vorliegt. Diese Spec bindet
DASS, WOMIT und MIT WELCHEM GELTUNGSBEREICH ersetzt werden darf, nicht den genauen Wortlaut.

**D24 — Geltungsbereich der neuen Aussage: Konsumenten-Skala, explizit gescoped
(Owner-Entscheidung 2026-07-21, v2).** Die projekteigene Erstdefinition des
Skalierungsrisikos ist ZWEIGLEISIG: Konsumenten-Skala (Ränge, Ketten, Dateien, IDE-Latenz)
UND API-Flächen-Skala (`docs/wettbewerbsanalyse-und-usp.md:9` wörtlich: „Skaliert die
Typ-Maschinerie auf eine NumPy-große API-Fläche…", wortgleich in Anhang A, Zeile 165). Diese
Scheibe misst ausschließlich die ERSTE. Bindend ist deshalb: die neue Formulierung wird
**ausdrücklich auf die Konsumenten-Skala gescoped** und **benennt die API-Flächen-Frage
ausdrücklich als weiterhin offen** — sie darf nicht so gelesen werden können, als sei
„at scale" insgesamt erledigt. Eine stille Verengung des Begriffs wäre die unehrlichste
mögliche Form dieser Ersetzung. (Verworfene Alternative: eine gekennzeichnete Extrapolation
aus den vorhandenen W-Serien-Ripple-Zahlen — Owner hat die klarere Scope-Ansage vorgezogen.)

**D25 — Drei Offenlegungs-Pflichten im Ergebnis-Doc (v2).**
1. **Der Cliff ist eine Falsch-Ablehnungs-Grenze, nicht nur eine Kostengrenze.** „Type
   instantiation is excessively deep" bei Rang 1024 ist ein Compile-Fehler auf GÜLTIGEM
   Code; die Degradations-Disziplin („wide statt falsch", Covenant M2, README:95) kann dort
   strukturell nicht greifen, weil der Checker selbst stirbt. Die publizierte Formulierung
   muss die Garantie entsprechend scopen („bis Rang ca. 768/896 gemessen; jenseits lehnt der
   Checker selbst gültigen Code ab"); die Frage, ob COVENANT M2 dafür eine v6-Präzisierung
   braucht, geht als Kandidat in FOLLOWUPS und an Baustein C — sie wird NICHT still
   mitentschieden.
2. **Die Kontrollpunkte vergleichen nach V0 saubere Zahlen** — vor V0 hätten sie zwei
   Mess-Populationen verglichen (alt: mit sieben unaufgelösten TS2591-Diagnosen; neu:
   sauber). Das Ergebnis-Doc nennt V0 als Vorbedingung und die alten Zahlen als das, was
   sie waren.
3. **Die Chain-Vorab-Probe der Spec-Erstfassung lief auf einem Exit-1-Programm** (ohne
   `ambient.d.ts`) und wurde neben einer sauberen Probe als wechselseitig bestätigend
   präsentiert. Der Befund überlebt (beide Verifier haben ihn sauber reproduziert: exakt
   konstant 27.701), aber die Verletzung der eigenen Exit-Code-Hausregel wird offengelegt,
   nicht stillschweigend geheilt.

## 11. Covenant-Abgleich

**Berührte Invarianten:**

- **Z2** (`package.json`) — HAUPT-Berührung, ausführlich in §3 „Z2-Konsequenz" behandelt. Der
  Präzedenzfall `spike/bench-dx/workloads/` (generiert, gitignored, exkludiert) trägt für „kein
  committeter Korpus rottet, weil keiner existiert" — bricht aber an der Kontinuitäts-Frage:
  `spike/bench-dx/workloads/` wird bei JEDEM `pnpm bench:editor`-CI-Lauf regeneriert+geprüft,
  der volle Scale-Sweep-Korpus NUR on-demand (Owner-Entscheidung 2). Der Sentinel (§6, D11-D15)
  ist die mechanische Antwort auf GENAU diese Lücke für die publizierte USP-AUSSAGE selbst
  (schützt die Regressions-Baseline kontinuierlich), schließt aber NICHT die schwächere,
  disclosed Restlücke „der Sweep-GENERATOR selbst könnte zwischen zwei manuellen Läufen
  kaputtgehen, ohne dass CI das bemerkt" — dieser Rest wird EINMALIG durch T1 (Nicht-Vakuität
  bei Fertigstellung) adressiert, nicht dauerhaft durch ein Gate. Diese Einordnung ist eine
  disclosed, Owner-Entscheidung-2-konforme Abweichung, KEIN stiller Normbruch — sie geht
  unverdünnt in den `covenant-verify`-Auftrag (Baustein C, §13).
- **M2** — NICHT berührt: diese Scheibe fügt KEINE neue Typ-Guard-/Shape-Maschinerie hinzu; der
  Sentinel (w8) kombiniert ausschließlich BESTEHENDE, bereits M2-konforme Ops
  (`add`/`item`/`slice`/`topk`) in einer neuen Kombination.
- **M1/M3/M4/M5/Z1** — NICHT berührt: keine neuen Kernel, keine neuen Fehlermeldungen, keine
  Rust-/ABI-Änderung, keine `node:*`-Eager-Imports jenseits dessen, was `gen-workloads.ts`/
  `editor-latency.ts` bereits tun (identisches Muster), keine neue `package.json`-Dependency.
- **S1** — NICHT berührt: `spike/bench-dx/` ist kein Laufzeit-Quellverzeichnis; die neuen
  Skripte importieren aus `spike/src/`, nie umgekehrt (gleiche Richtung wie
  `editor-latency.ts`/`gen-workloads.ts` bereits).

- **M2 — berührt (Ergänzung v2), aber nicht verletzt.** Die Scheibe fügt keine Typ-Maschinerie
  hinzu; sie legt aber eine Grenze der M2-Garantie frei: am Rang-Cliff lehnt der Checker
  GÜLTIGEN Code ab („Type instantiation is excessively deep"), und die Degradation zu no-claim
  kann dort strukturell nicht greifen, weil der Checker selbst abbricht. Das ist kein neuer
  Normbruch dieser Scheibe (die Grenze existiert seit jeher), sondern ein bisher unbenannter
  Geltungsbereich von M2. Behandlung: Offenlegungspflicht im Ergebnis-Doc (D25.1),
  v6-Kandidat in FOLLOWUPS, ausdrückliche Frage an Baustein C — NICHT still mitentschieden.

**Eskalationsstufe:** **Stufe 3** (substanzielle Scheibe, bindende Spec) — voller Katalog
A+B+C parallel + `graph-a-lama query lint` im Gate-Block (§12/§13), wie im
Intentions-Kontext bereits vorgegeben.

## 12. Gate-Block / Definition of Done

**Vorbedingung:** V0 (§1) ist committet und grün, BEVOR der erste Sweep-Lauf stattfindet.

Vor der „fertig"-Meldung dieser Scheibe (Spec → Implementierung → Messung → Ergebnis-Doc →
README/USP-Ersetzung, siehe Terminologie-Klarstellung in §10) müssen grün sein:

- `pnpm check` (Dreifach-Verbund) — Root-Pin nach D3s empty-then-fill-Dekomposition neu
  gesetzt und dokumentiert; Stress-/Browser-Pins unverändert (mechanisch bewiesen, kein
  Messbedarf, §3/§8).
- `pnpm check:diag` / `check:diag:stress` / `check:diag:browser` — Zahlen wie oben.
- `pnpm test:core` — unverändert grün (diese Scheibe berührt keine Laufzeit-Tests).
- `pnpm bench:editor` — **w1-w7 EXAKT unverändert** (T5), w8 neu gepinnt (D13), 2×-Latenz-
  Ceiling für alle acht PASS.
- `pnpm bench:scale` — läuft durch (Exit 0 oder dokumentierter, ERWARTETER Cliff-Abbruch bei
  Achse-c-Extrempunkten, siehe G3), G1 (30-min-Budget) eingehalten oder Sweep-Reduktion
  dokumentiert.
- `graph-a-lama query lint` — 0 Befunde (S-Invarianten-Gate, mechanisch).
- Git-Status: NUR die geplanten neuen/geänderten Dateien (`gen-scale-workloads.ts`,
  `scale-latency.ts`, `.gitignore`-Zeile, `tsconfig.json`-Exclude-Zeile,
  `gen-workloads.ts`-Ergänzung, `editor-latency.ts`-Pin-Ergänzung, Ergebnis-Doc,
  FOLLOWUPS-Einträge) — `spike/bench-dx/scale-workloads/` selbst NICHT im Diff (gitignored).

## 13. Verify-Plan

Nach `docs/verify-runde-template.md`, Stufe 3.

**Baustein 0 (VOR der Implementierung, `brainroute:deep`, frischer Kontext):** Design brechen,
NICHT bestätigen. Schwerpunkte, aus den Design-Entscheidungen dieser Spec abgeleitet:
- **D5 (Duplizieren vs. Extrahieren) empirisch nachprüfen:** stimmt die Risikoabwägung? Gibt es
  einen dritten, hier übersehenen Weg mit geringerem Risiko UND weniger Duplizierung?
- **D7 (ambient.d.ts-Pflicht) am echten Code verifizieren:** Reproduziert der Verifier die in
  dieser Spec dokumentierte Vorab-Probe (TS2591 ohne `ambient.d.ts`, sauber mit) UNABHÄNGIG,
  mit einem eigenen Scratch-Aufbau? Klärt DABEI auch die offene Frage aus „Beobachtungen
  außerhalb des Scope" (ob das bestehende `gen-workloads.ts`-Muster real betroffen ist).
- **D12/D13/D14 (Sentinel) gegen den echten `editor-latency.ts`/`gen-workloads.ts`-Code
  prüfen:** trägt die Behauptung „`measureWorkload`/Hard-Gate sind bereits generisch über
  `manifest.workloads`" wirklich (Datei:Zeile-genau nachlesen, nicht nur dieser Spec glauben)?
- **G1s 30-Minuten-Budget plausibilisieren:** mit den Vorab-Probe-Zahlen (§2.3: Rang 1024
  isoliert ca. 2 s Checkzeit) hochrechnen, ob der VOLLE Sweep (alle Achsen × alle Punkte × M1
  kalter Serverstart pro Punkt) realistisch unter 30 min bleibt, oder ob die Spec hier
  nachschärfen muss (z. B. Sweep-Punkte weiter reduzieren) — BEVOR gebaut wird.
- **Covenant-Abgleich (§11) prüfen:** verletzt die Spec selbst eine Invariante, ist die
  Z2-Einordnung haltbar, ist die Eskalationsstufe korrekt?
- **Achse-(b)-Umdeutung (§2.2) hinterfragen:** ist die Vorab-Probe wirklich repräsentativ genug
  (nur zwei Konstruktionen getestet, `[8,8]`-Fixshape), um Achse (b) komplett von der
  Rekursionstiefen-Frage zu entkoppeln? Sollte eine DRITTE, echt nicht-cachebare Chain-
  Konstruktion (variierende Shapes pro Schritt) ergänzt werden?

**Baustein A (Spec-Konformität, `brainroute:verify`):** jede D-Entscheidung einzeln gegen den
Diff prüfen; alle Gates aus §12 frisch ausführen; eigener Mutant (Pflicht) gegen die neuen
Skripte; Disziplin-Prüfung: `editor-latency.ts`/`gen-workloads.ts`-Diff enthält NUR die in D5/
D12/D13 erlaubten additiven Änderungen.

**Baustein B (adversarial):** Grenzfälle jenseits der Spec — was passiert bei Achse-a N=0? Bei
einem Sweep-Punkt, dessen `tsc`-Prozess wegen Speicher/Timeout crasht statt sauber
„excessively deep" zu melden (Achse c, Rang nahe 1024 — der Vorab-Probe-Speicherverbrauch lag
bei ca. 973 MB, auf einem Rechner mit weniger RAM könnte das OOM statt „excessively deep"
sein)? Mutanten breit statt tief an Nachbarstellen (z. B. der `.gitignore`-Eintrag, die
`tsconfig.json`-Exclude-Zeile). Messrandbedingungen angreifen (ist die Root-Pin-Baseline vor
dieser Scheibe wirklich sauber in einem frischen Worktree gemessen?).

**Baustein C (`covenant:covenant-verify`):** COVENANT.md wörtlich + Diff + die in §11 benannten
Invarianten-IDs + `graph-a-lama query lint`-Output — insbesondere die Z2-Einordnung (§3/§11)
OHNE die Bewertung dieser Spec selbst mitzugeben (der Agent urteilt allein aus Vertrag + Diff).

## 14. Offene Entscheidungen — Auflösungsstand v2

**Alle fünf Fragen der Erstfassung sind beantwortet** (Baustein 0, Frontier-Zweitmeinung,
Owner). Die Liste darunter bleibt als v1-Historie stehen, die Auflösung ist bindend:

1. **`WORKLOAD_COMPILER_OPTIONS`: DUPLIZIEREN** — konsistent mit D5s Duplizierungs-Entscheidung
   für den LSP-Client; `gen-workloads.ts` war nie als stabiles Modul gedacht, und ein Import
   machte den Diff einer gate-kritischen Datei ohne Not nicht-leer.
2. **Sentinel-Dimensionen: bleiben Implementierer-Wahl** — beide Verifier stufen das als reine
   Detailfrage ein; bindend bleibt die STRUKTUR (Rang 24 + item/slice/topk + Toggle-Ziel, D12/D22).
3. **G1s 30-Minuten-Budget: bestätigt, eher zu großzügig** — nicht hochgerechnet, sondern
   gemessen: Achse-a-Spitzenpunkt (250 Dateien, ein Programm) 0,093 s, Achse-b-Spitzenpunkt
   (L=10000) 3,6 s Checkzeit. Keine Nachschärfung. Vorbehalt: diese Messungen liefen auf der
   cache-freundlichen Variante — mit D19/D20 (zwei Unter-Serien) verdoppelt sich der Umfang
   ungefähr, was das Budget weiterhin komfortabel einhält.
4. **Dritte Chain-Konstruktion: JA, bindend** — siehe D20. Die v1-Empfehlung („nein, mit
   Verweis auf den empirischen Befund") ist damit überstimmt: sie hätte „konstant" publiziert,
   wo „linear, ca. 265 Instantiations pro Kettenglied" die ehrliche Aussage ist.
5. **Das bestehende `gen-workloads.ts`-Muster IST betroffen** — unabhängig reproduziert
   (7× TS2591, Exit 1), und zwar in BEIDEN tsconfig-Ausgaben. Auflösung: V0 (§1), Owner-Entscheidung
   „vorher reparieren". Der entscheidende Zusatzbefund der Zweitmeinung: es existiert kein
   `@types/node` im Repo (bewusste Zero-Dep-Entscheidung, dokumentiert im `ambient.d.ts`-Header)
   — erst das macht die Scratch-Reproduktionen für das echte Repo beweiskräftig.

### v1-Fassung der offenen Fragen (Historie)

1. **`WORKLOAD_COMPILER_OPTIONS`-Wiederverwendung:** Soll `gen-scale-workloads.ts` diese
   Konstante aus `gen-workloads.ts` RE-EXPORTIEREN/importieren (DRY, aber ein Import aus einer
   Datei, die selbst nicht als „stabiles Modul" gedacht war) oder — konsistent mit D5s
   Duplizierungs-Entscheidung für den LSP-Client — ebenfalls duplizieren? Diese Spec bindet die
   LSP-Client-Frage (D5), lässt diese kleinere, risikoärmere Frage (eine reine Konstante, kein
   Prozess-/Zustands-Code) für Baustein 0 offen.
2. **Exakte literale Dimensionen des Sentinels (`w8`):** D12 bindet Struktur (Rang 24 +
   item/slice/topk-Folge, 80-150 LOC), nicht die exakten Zahlen. Baustein 0 oder der
   Implementierer legt sie fest; die Spec verlangt nur, dass sie NACH der Kalibrierungs-Tabelle
   in §5 gewählt werden.
3. **G1s 30-Minuten-Budget:** siehe Baustein-0-Auftrag oben — falls die Hochrechnung zeigt, dass
   der volle Sweep (alle Achsen × Punkte × M1-Kaltstart) das Budget strukturell sprengt, muss
   VOR der Implementierung entweder das Budget (mit Owner-Bestätigung) angepasst oder die
   Sweep-Punkte weiter reduziert werden.
4. **Dritte Chain-Konstruktion für Achse (b):** siehe Baustein-0-Auftrag oben — soll eine
   echt-nicht-cachebare Variante (variierende Shape pro Kettenschritt) ergänzt werden, um die
   Rekursionstiefen-Frage nicht komplett an Achse (c) allein zu delegieren? Diese Spec empfiehlt
   „nein, mit Verweis auf den empirischen Befund", entscheidet es aber nicht endgültig — der
   Owner kann hier gegensteuern.
5. **Bestehendes `gen-workloads.ts`-Muster (`files: [w.fileName]`, kein `ambient.d.ts`):** ist
   das ein reales, bisher unbemerktes Problem der PRODUKTIONSMESSUNG (w1-w7) selbst, oder habe
   ich (diese Spec-Autorin) etwas übersehen, das es im echten Setup doch vermeidet? Siehe
   „Beobachtungen außerhalb des Scope" — NICHT Teil dieser Scheibe zu beheben, aber Baustein 0
   sollte es einmal gegen den echten `pnpm bench:editor`-Lauf verifizieren, weil es die
   Glaubwürdigkeit der w1-w7-Zahlen betrifft, auf denen diese Scheibe aufbaut.

## Was empirisch vorab geprobt wurde (Vorab-Proben, KEINE Messergebnisse dieser Scheibe)

Alle Proben liefen AUSSCHLIESSLICH im Scratch
(`/private/tmp/claude-501/.../scratchpad/scale-probe-pretest/`), importierten den echten
Typ-Layer über einen ABSOLUTEN Pfad (read-only), und haben NICHTS im Repo verändert. Nutzt
den installierten, projekt-gepinnten `tsc` (`node_modules/.bin/tsc`, Version 7.0.2) über
`execFileSync(..., ["--noEmit", "--extendedDiagnostics", "-p", ...])`, isoliert pro Programm
via `files: [...]`-Tsconfig (Muster identisch zu `gen-workloads.ts`).

1. **Rang-Sweep (W2-Muster, Broadcast-`add()` mit alternierender Größe-1-Achse), MIT
   `ambient.d.ts` in `files`:** `16→28.646`, `32→30.966`, `64→38.678`, `128→66.390`,
   `256→170.966`, `512→576.726` (Checkzeit 0,021 s → 0,404 s), `768→1.243.990`
   (Checkzeit 1,146 s), **`1024` → Abbruch mit „Type instantiation is excessively deep,
   possibly infinite"** (2.170.608 Instantiations bis zum Abbruch, ca. 973 MB
   Speicherverbrauch). Ein echter, reproduzierbarer Cliff, konsistent mit der dokumentierten
   ca. 1000-Tail-Rekursions-Grenze — direkte Grundlage für §2.3s Sweep-Punkte `{768, 896,
   1024}`.
2. **Chain-Längen-Sweep (W1-Muster, alternierend `matmul`/`add` über feste `[8,8]`-Shape, N
   separate Statements), OHNE `ambient.d.ts`:** `L=100/500/1000/2500/5000` liefern ALLE exakt
   `27.775` Instantiations (konstant), Checkzeit wächst ca. linear (0,016 s → 0,903 s). Zeigt:
   TS cached identische Typinstanzen über Call-Sites hinweg — Chain-Länge in dieser
   Konstruktion stresst Datei-Durchsatz, nicht Rekursionstiefe. Grundlage für §2.2s
   Umformulierung.
3. **Verschachtelter Einzelausdruck (`x.matmul(w).matmul(w)…`, N=50 bis 2000), MIT
   `ambient.d.ts`:** ALLE Läufe liefern konstant `27.452` Instantiations, NIE „excessively
   deep", selbst bei N=2000. Bestätigt Befund 2 aus einer zweiten, strukturell anderen
   Konstruktion (ein einziger verschachtelter Ausdruck statt N Statements) — verstärkt die
   Umformulierung von Achse (b).
4. **`ambient.d.ts`-Notwendigkeit isoliert:** ein Programm ohne `ambient.d.ts` in `files`
   schlägt mit TS2591 („Cannot find name 'process'"/„'node:fs/promises'"/…) an
   `spike/src/wasm/backend-api.ts`, `loader.ts`, `threaded.ts` fehl (real bzw. typ-only von
   `ndarray.ts` importiert, `ndarray.ts:53,55`); dasselbe Programm MIT `ambient.d.ts` in
   `files` kompiliert sauber. Direkte, empirisch bewiesene Grundlage für D7.

## Beobachtungen außerhalb des Scope

- **Möglicherweise betroffen: das bestehende `spike/bench-dx/gen-workloads.ts`s
  Per-Workload-`tsconfig.<id>.json`** (`files: [entry.fileName]`, OHNE `ambient.d.ts`,
  `gen-workloads.ts:648`, genutzt von `measureInstantiations`, `editor-latency.ts:557-587`) hat
  strukturell dieselbe Form wie die Vorab-Probe, die OHNE `ambient.d.ts` zuverlässig an
  `process`/`node:fs/promises`/… scheiterte (siehe „Was empirisch geprobt wurde", Punkt 4). Ich
  habe dies NICHT gegen den echten `pnpm bench:editor`-Lauf verifiziert (die Aufgabenstellung
  untersagt Schreibzugriffe im Haupt-Working-Tree außer der einen Spec-Datei, und
  `gen-workloads.ts` auszuführen hätte Dateien unter `spike/bench-dx/workloads/` erzeugt/
  überschrieben). Falls das Muster real betroffen ist, würden die w1-w7-`hadTypeErrors`-Flags
  vermutlich fälschlich `true` für ALLE Workloads sein (nicht nur w4, das absichtlich Fehler
  trägt) — `enforceHardGate` (`editor-latency.ts:785-832`) prüft `hadTypeErrors` an KEINER
  Stelle, sodass ein solcher Zustand das Hard-Gate nicht rot färben würde, nur die kosmetische
  Notiz in `printInstantiationTable` verfälschen könnte. Die tatsächlichen Instantiation-ZAHLEN
  wären davon vermutlich UNBEEINFLUSST (die fehlenden Ambient-Deklarationen sind in
  Node-spezifischen Dateien, die mit `NDArray<S>`-Instanziierung nichts zu tun haben) — aber das
  ist eine Vermutung, keine Messung. Nichts an dieser Spec hängt davon ab (D7 macht die
  RICHTIGE Vorgabe für die NEUEN Skripte unabhängig vom Ausgang dieser Frage); als Frage 5 in
  §14 an Baustein 0 weitergereicht, NICHT selbst behoben.
- **`docs/wettbewerbsanalyse-und-usp.md` §4 nennt weitere, für diese Scheibe interessante, aber
  bewusst NICHT gemessene TS-Grenzen** (z. B. `#47481`s „4.5s-27s Compile-Zeit bei großen
  String-Literal-Unions") — keine dieser Zahlen wird in dieser Scheibe nachgemessen; nur als
  Beobachtung, dass die Zahl KEIN Sweep-Ziel dieser Spec ist (String-Literal-Unions kommen im
  numtype-Typ-Layer strukturell nicht in der hier relevanten Form vor).

## Adversariale Spec-Verifikation (Addendum, 2026-07-21)

Die Erstfassung (v1) durchlief zwei unabhängige Fresh-Context-Läufe, beide read-only gegen den
echten Code, beide mit eigenen empirischen Proben ausschließlich im Scratch (der
Haupt-Working-Tree blieb unberührt, per `git status --porcelain -uall` von beiden belegt):

1. **Baustein 0** (adversarialer Spec-Verifier, Auftrag aus `docs/verify-runde-template.md`):
   Auftrag war, das Design zu brechen.
2. **Frontier-Zweitmeinung** (vom Owner angefordert, anti-anchoring: eigenes Soll-Design VOR
   der Spec-Lektüre schriftlich fixiert, Baustein-0-Report erst danach gelesen).

**Was BEIDE unabhängig bestätigt haben:** den Rang-Cliff (Randpunkte `768 → 1.243.990` und der
1024er-Abbruch byte-identisch zur Erstfassung; zusätzlich `896 → 1.676.118`, also Cliff bei
1024, nicht bei 896); die Chain-Konstanz auf fixer Shape (sauber reproduziert, inklusive
L=10000 direkt gemessen statt extrapoliert); die `ambient.d.ts`-Notwendigkeit (D7); das
G1-Budget (gemessen statt hochgerechnet); die Code-Annahmen zu `enforceHardGate`,
`INSTANTIATION_PINS`, `measureWorkload`, `measureInstantiations` und der generischen
Manifest-Iteration (D13/D14 halten); die Covenant-Einordnung inklusive der Z2-Analyse.

**Befunde, die in v2 eingearbeitet wurden:**

| Herkunft | Befund | Auflösung |
|---|---|---|
| Zweitmeinung (BLOCKER) | Achse (a) hätte ohne Dims-Vorgabe Cache-Treffer statt Skalierung gemessen (Faktor ca. 15 zwischen identischen und distinkten Shapes, gemessen) | **D19** — zwei bindende Unter-Serien |
| Baustein 0 + Zweitmeinung (MAJOR) | Die shape-variierende Kette wächst linear (ca. 265 Instantiations/Glied); „konstant" gilt nur für die künstlichste Konstruktion | **D20** — bindende Unter-Serie, ersetzt §14 Frage 4 |
| Baustein 0 (MAJOR) | Der erste Cliff-Treffer hätte den gesamten restlichen Sweep mitgerissen (kein try/catch, `main().catch` rethrowt) | **D23** + **T7** |
| Zweitmeinung (MAJOR) | Die LSP-Projekt-Discovery isoliert die Sweep-Punkte nicht — ein `tsconfig.<achse>-<punkt>.json` wird nie discovert | **D21** — Unterverzeichnis pro Punkt mit echter `tsconfig.json` |
| Baustein 0 + Zweitmeinung (MAJOR) | Die generierten tsconfigs führen kein `ambient.d.ts` — auch das geteilte, über das die publizierte Hover-Zahl gemessen wurde | **V0** (Owner: vorher reparieren) |
| Zweitmeinung (MAJOR) | Kein Sweep-Punkt maß die inkrementelle Tipp-Latenz (M3), obwohl der Produkt-Claim „while you type" lautet | **D22** — Toggle-Ziel für w8 und den obersten Achse-(a)-Punkt |
| Zweitmeinung (MAJOR) | Die API-Flächen-Hälfte der projekteigenen Risiko-Definition fehlt ganz | **D24** (Owner: Konsumenten-Skala, API-Fläche explizit als offen benannt) |
| Zweitmeinung (MAJOR/minor) | Der Cliff ist eine Falsch-Ablehnungs-Grenze auf gültigem Code, nicht nur eine Kostengrenze — Geltungsbereich von M2 | **D25.1** + §11 + v6-Kandidat |
| Zweitmeinung (MINOR) | Speicher bindet potenziell vor der Zeit (ca. 973 MB am Cliff) | **D8** — `results.json` erfasst „Memory used" |
| Zweitmeinung (MINOR) | Vorab-Probe 2 der Erstfassung lief auf einem Exit-1-Programm | **D25.3** — Offenlegungspflicht |
| Baustein 0 (NIT/MINOR) | `abi.rs:159` ist tatsächlich Zeile 158; D5s Argument 1 leicht zu optimistisch gewichtet (die Root-Pin-Neumessung fällt wegen D3 ohnehin an) | korrigiert bzw. hier festgehalten; D5 selbst hält, ein risikoärmerer vierter Weg wurde von beiden nicht gefunden |

**Wo die beiden Verifier sich widersprachen (Owner-entschieden, nicht still aufgelöst):**

- **Schweregrad des `ambient.d.ts`-Befunds.** Baustein 0: „blockiert nicht, disclosen,
  FOLLOWUPS". Zweitmeinung: fix-before, tragendes Argument war das von Baustein 0 übersehene
  Detail, dass der Sentinel w8 im selben defekten Generator-Pfad entsteht und damit der einzige
  Dauerwächter der neuen Aussage auf einem fehlerbehafteten Programm gepinnt würde.
  **Owner-Entscheidung 2026-07-21: vorher reparieren** → V0.
- **Verbindlichkeit der dritten Chain-Konstruktion.** Baustein 0: „erwähnen oder aufnehmen".
  Zweitmeinung: bindende Unter-Serie, weil sie die publizierte Aussage von „konstant" auf
  „linear" ändert. **Übernommen in der stärkeren Form** → D20.

**Was v2 NICHT ändert:** die vier Achsen als Zerlegung (Owner-Entscheidung 1), die
Sentinel-Idee und ihre Pin-Mechanik (D11-D15), die Duplizierungs-Entscheidung (D5), die
Korpus-Platzierung und den Pin-Schutz (D1-D4), die Gate-Philosophie (§7) und den Eich-Anker
(§5). Beide Verifier haben diese Teile geprüft und für tragfähig befunden.
