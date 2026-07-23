# NumType

> **Note for contributors:** This file is the maintainer's internal working memory
> for an agent-assisted workflow (Claude Code). It references private tooling and
> plugins (`brainroute`, `graph-a-lama`, `coding-kb`, `covenant`, …) that are not
> part of this repository, and it mixes English with German research notes.
> Nothing in here is required to build, test, or use NumType — see
> [README.md](README.md) and [docs/](docs/) instead. The commands in the
> "Commands" section work for everyone, though.

NumPy-like n-dimensional array library: TypeScript type-level shape checking + (later) from-scratch Rust/WASM kernels. Research project — the explicit goal is probing the limits of what's feasible.

## Hard constraints (user-set, 2026-07-09)

- **No external libraries.** All kernels and all type machinery written from scratch. Dev tooling (typescript, later test runners) is allowed; product/runtime dependencies are not. Never suggest pulling in `ndarray`/`faer`/BLAS bindings etc.
- Brand name: **NumType** (npm package name stays lowercase `numtype`).
- Public repo + npm release since 2026-07-19 (v0.1.0). **All user-/public-facing text is English** (owner-set 2026-07-19): commit messages, tag/release notes, issue/PR text, README/spec docs, code comments, error messages, GitHub/npm metadata. Internal process & research docs (docs/ research notes, projekt-log, FOLLOWUPS, COVENANT, HANDOFF) may stay German — disclosed as such in the README. Chat with the owner stays German.
- Research fan-outs stay small: targeted agents (≤3), no broad sweeps.

## USP (defensible form — sources in docs/wettbewerbsanalyse-und-usp.md)

NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors — gradual, with a `number`-dim escape hatch for dynamic shapes. Python provably cannot do this statically today; in TS it is newly tractable but unproven at scale. That gap is the project.

## Status (v0.1.0 — released)

**numtype@0.2.0 ist live auf npm (2026-07-21 — „the wishlist release": alle fünf
Dogfooding-Ops argmax/topk · Skalar-Overloads+mean · sqrt · stack · item; davor
0.1.0/0.1.1 am 2026-07-19), das Repo ist public, Tags `v0.1.0`/`v0.1.1`/`v0.2.0`,
Apache-2.0. Das Example läuft auf 0.2.0 als Vorher/Nachher-Showcase
(FRICTION→RESOLVED-Kommentare, F→W-Tabelle in der Example-README);
Drop-in-Kompatibilität 0.1.1→0.2.0 asserted bewiesen.** Roadmap-Phasen A–D und Items 1–14 sind komplett —
**die Roadmap ist durchgespielt** (Phase-C-Items 8/9 — Browser-Threads/no_std — bewusst
deferred). Item 14 schloss 2026-07-19: Demo-GIF, SemVer-Sektion, Reading Guide (docs/README.md),
Launch-Blog-Post (https://marvinmuegge.com/notes/teaching-the-checker-arithmetic/).
**Richtung (Owner-entschieden 2026-07-20): wachsendes OSS-Projekt mit Nutzern.** Reihenfolge:
(1) Launch-Post streuen — der README-Playground-Link (ATA lädt die Typen von npm; live
verifiziert, Playground läuft TS 6.0.3) ist das Erlebe-es-selbst-Asset für HN/Reddit-Kommentare;
(2) Dogfooding-Scheibe — **ERLEDIGT 2026-07-20** (examples/rag-demo konsumiert das
veröffentlichte Paket aus der Registry, CI-Job `example`, kuratierte Op-Wunschliste W1–W5 in
docs/dogfooding-rag-ergebnisse.md: argmax/topk > Skalar-Overloads > sqrt > stack > item;
dreifach verifiziert; Z2-Frage Owner-entschieden 2026-07-20 = Option (b): Registry-Tripwire
`scripts/check-example-registry-drift.mjs` in test:example + CI-Job, erzwingt den
Example-Dep-Bump je Release mechanisch); (3) Scale-Probe als bindende Spec („unproven at
scale" → gemessen). Erste Op-Scheibe aus der Wunschliste: **W1 (argmax/topk) — ERLEDIGT
2026-07-20, dreifach verifiziert** (docs/op-w1-argmax-topk-spec.md v4 /-ergebnisse.md;
NDArray-only, kein WASM-Kernel — FOLLOWUPS: WNDArray/Threaded-Parität + M1-Präzisierungs-
Empfehlung; der stress-Pin-Ripple +842 ist als legitime Klassen-Surface-Typkosten
akzeptiert und neu gepinnt, Spec-v3-Korrektur). Zweite Op-Scheibe: **W2 (Skalar-Overloads
add/sub/mul/div + mean) — ERLEDIGT 2026-07-21** (docs/op-w2-scalar-mean-spec.md v2
/-ergebnisse.md; D6-v2-Overload-Umbau der vier Bestandsmethoden, `mean` neu nach
sum-Muster, NDArray-only/kein WASM-Kernel wie W1 — FOLLOWUPS-Paritätsitem erweitert; der
stress-Pin-Ripple +1,181 ist derselbe Klassen-Surface-Mechanismus wie W1, akzeptiert und
neu gepinnt). Dritte Op-Scheibe: **W3 (`sqrt`) — ERLEDIGT 2026-07-21** (docs/op-w3-sqrt-spec.md v1
/-ergebnisse.md; niladisch, shape-erhaltend, NDArray-only/kein WASM-Kernel wie W1/W2 —
IEEE-754-korrekt-gerundet wie `+`/`-`/`*`/`/`, daher vom Transzendenten-Nicht-Ziel
ausgenommen; F1-Schließung (Teilkette + volle L2-Normalisierung) byte-identisch gegen die
alte Hand-Loop-Formulierung aus examples/rag-demo/main.ts bewiesen; stress/browser-Pins
unverändert, kein Klassen-Surface-Ripple diesmal). Vierte Op-Scheibe: **W4 (`stack`) —
ERLEDIGT 2026-07-21, inkl. Verify-Runde-Fix** (docs/op-w4-stack-spec.md v2 + Baustein-0-
Addendum F1-F8 /-ergebnisse.md; `NDArray.stack(rows)` baut `[N, D]` aus N Rang-1-Zeilen,
NDArray-only/kein WASM-Kernel wie W1-W3; F5-Schließung — `embedMatrix`s Zeilen-Flatten-Helper
— byte-identisch bewiesen; ZWEI echte Typfunde selbst gefangen und geschlossen: (1) während
der eigenen Umsetzungs-Verifikation, Array-Union-Element-Kollaps in `RowShapesOf` (F2-
verwandt); (2) während der Verify-Runde (Baustein B, BLOCKER-Klasse M2-Verstoß), Tupel-
Positions-Union-Distribution in `StackFold` — `IsUnion<Head>`-Gate vor dem naked Match
ergänzt, `ReduceAxis`-Positions-Präzedenz, Nicht-Vakuität per Mutations-Probe bewiesen;
Klassen-Surface-Ripple wie W1/W2, `bench:editor`-Pins zweimal neu gesetzt: +845 uniform,
dann +6 uniform aus dem Fix). Fünfte und letzte Op-Scheibe der Wunschliste: **W5 (`item`) —
ERLEDIGT 2026-07-21** (docs/op-w5-item-spec.md v2 + Baustein-0-Addendum F1-F8
/-ergebnisse.md; `NDArray.item(...indices)` — voller Skalar-Read, `ItemGuard<S, Idx>` direkt
als Rest-Parameter-Typ (F1, Guard<>-Wrapper wäre TS2370), S-getriebener Fold (F2), Arity
nativ über TS2554 (F3), Spread-Gate via `IsDynamicRank` (F4-Fix); M1 v5: kernel-los per
Design (reiner strided Read, kein Kernel zu parallelisieren) — damit ist die komplette
Dogfooding-Wunschliste W1-W5 abgearbeitet. D6-Befund: `Equal<ItemGuard<...>>`-Message-
Pins kosteten pro Pin ≈1,700 Instantiations — die Erst-Umsetzung überschritt das
+6,000-Gate fast um das Doppelte, budgetgetriebene Pin-Konsolidierung (ein kombinierter
Zwei-Positionen-Pin statt fünf Einzel-Pins) brachte es auf +5,873; FOLLOWUPS trackt sowohl
den Kostenmechanismus als auch das Aufsplitten von scalar-mean.test.ts, das jetzt W2-W5
sammelt).
**Scale-Probe ERLEDIGT 2026-07-21** (docs/scale-probe-spec.md v2 /-ergebnisse.md): Der dritte und
letzte Punkt der Owner-Reihenfolge ist damit abgearbeitet — „unproven at scale" ist aus README
und USP-Doc verschwunden und durch gemessene Zahlen ersetzt (Konsumenten-Skala explizit gescoped,
API-Flächen-Skala ausdrücklich als offen benannt, Owner-Entscheidung). Kernbefund: warmer Hover
0,04–0,11 ms über ALLE 34 messbaren Punkte, die Skalenkosten landen auf dem Kaltstart (1,5 ms bei
250 Dateien, 10,2 s bei einer 10.000-Glieder-Kette); linear in Dateizahl und Kettenlänge,
überproportional im Rang, harter Cliff bei Rang 1024 (TS2589 auf gültigem Code). Vorab-Scheibe
**V0** (c18aa7f) reparierte die Mess-Basis (generierte tsconfigs ohne ambient.d.ts → 7x TS2591 in
allen Workloads; `enforceHardGate` liest `hadTypeErrors` nie). Prozess-Bilanz: Baustein 0 + eine
Frontier-Zweitmeinung änderten das Design VOR dem Bau an fünf Stellen (ohne sie hätte Achse (a)
Cache-Treffer statt Skalierung gemessen — Faktor 19); Verify-B fand ein VAKUÖSES Hover-Gate auf
der Rang-Achse (behoben, 8 Mutationen belegen die Wirksamkeit); zwei Verifier widerlegten
unabhängig die „überproportional"-Charakterisierung der Datei-Achse, bevor sie publiziert wurde.
**topk-Selektion KOMPLETT (Messung 2026-07-22, Umsetzung ERLEDIGT 2026-07-23)**
(docs/op-topk-selection-spec.md v6 /-ergebnisse.md): Verdikt **reiner Heap**, mechanisch aus der
vorregistrierten Regel berechnet — null duale Verletzungen im 92-Zellen-Raster, 57 Gewinn-Zellen,
`n = 1e6, k = 1` von 280 ms auf 3,8 ms (Faktor 74); Kehrseite offengelegt: sieben Zellen ab
`k/n = 0,85` absolut langsamer, max. +13,95 ms. **Phase 2 (Umsetzung):** `topkRuntime` in-place
durch den größenbeschränkten Max-Heap (O(n log k)) ersetzt, bit-identisch zur alten Full-Sort
(Orakel-Umzug + Differentialtest über 300+ Fälle inkl. exakter nicht-kanonischer NaN-Payload),
voller Verify-Katalog A+B+C alle grün (Root-Pin 206.854 @ 140, Δ+53 reine Typkosten; stress/browser/
bench:editor Δ0). NDArray-only, kein WASM-Kernel (M1 bindet nicht); künftiger `nt_topk`-Kernel
spiegelt den Heap (FOLLOWUPS). In-Place-Bruch der runtime.ts-Append-Konvention in der Spec vorab
genehmigt.
**WASM-Parität-Kampagne S0–S5 gestartet (Owner-entschieden 2026-07-23): `WNDArray`/threaded ziehen
die W1–W5-Ops nach. S0 (sqrt) ERLEDIGT 2026-07-23** (docs/wasm-parity-sqrt-spec.md v3
/-ergebnisse.md): Rust/WASM-Kernel `nt_sqrt_strided` + niladische `WNDArray.sqrt()`, threaded-Parität
automatisch (dasselbe Crate); M1 bindet erstmals für eine der neuen Ops und ist dreifach belegt
(Baustein-0-Vorab-Probe 30.028 Fälle, committeter Differentialtest, Baustein-B-BigInt-Oracle über
102.281 Elemente — je 0 Abweichungen), voller Verify A+B+C grün, netto **−4** check:diag.
Kampagnen-Gewinn aus einem Umsetzungs-Befund: der `Omit<ThreadedCoreExports,"memory">` →
direkter-Cast-Fix in threaded.ts (D10, laufzeit-identisch, vierfach belegt) beseitigt die
`keyof`-getriebene Generic-Neuauflösung an der Wurzel — jede Folge-Scheibe kostet auf diesem
Mechanismus **+0 statt +7** (vierter Kostenmechanismus, s. Mess-Regeln). S1–S5 offen (FOLLOWUPS).
**Zwei Prozess-Lehren, wertvoller als die Optimierung selbst:** (1) Die informelle Vorab-Sondage
lag um mehr als eine Größenordnung daneben (0,60 gegen gemessene 1,050 bei `k = n`; bei `k = n/2`
sogar mit falschem Vorzeichen) — live nachgestellt, Ursachen im Sondage-Quelltext belegt
(2 Aufwärm-Aufrufe, JIT-Kontamination durch 20.000 vorherige Fuzz-Fälle, JS- statt typisierte
Arrays); derselbe Mechanismus wie in Kern 06. (2) Die vorregistrierte Entscheidungsregel wurde
VIERMAL gebrochen, bevor sie messen durfte — zwei der Fassungen stammten vom Orchestrator selbst;
gefunden ausnahmslos dadurch, dass Verifier sie als Skript nachbauten und gegen tausende
synthetische Raster laufen ließen statt sie zu lesen.
FOLLOWUPS-Minis nebenher; Trusted Publishing optional (Fakten in FOLLOWUPS). **COVENANT-v6-Bündel
steht bei vier Kandidaten** — reif für eine eigene kleine Vertrags-Scheibe.
Repo-Härtung aktiv seit 2026-07-20: Rulesets `protect-main` (kein Force-Push/Delete auf main —
gilt auch für den Owner; bewusste Ausnahme nur via Ruleset-Deaktivierung) +
`protect-release-tags` (`v*` unverrückbar). README trägt seit 2026-07-20 eine
numpy-ts-Abgrenzungs-FAQ (re-verifiziert: dort weiterhin keine Shape-Level-Typen).
Der naive TS-Runtime bleibt die Korrektheits-Referenz; das v1-Copy-Backend bleibt die eingefrorene
Performance-Baseline (Kernels/Einstiegspunkte byte-für-byte unberührt).

Jede Phase folgt: bindende Spec → Implementierung → Fresh-Context-Verify → Ergebnisse-Doc mit
Post-Verification-Addendum → KB-Capture → Commit.

Wo was steht: [docs/roadmap.md](docs/roadmap.md) (Item-Status) · `docs/*-spec.md` +
`docs/*-ergebnisse.md` (Scheiben-Details, Primärquellen) ·
[docs/projekt-log.md](docs/projekt-log.md) (das vollständige historische Narrativ, früher in
dieser Datei) · FOLLOWUPS.md (zurückgestellte Arbeit) · HANDOFF.md (lokal, untracked —
Session-Zustand).

## Aktuelle Pins & Gates (IST-Zahlen; Historie im Projekt-Log)

- **Artefakt-Hash** (Clean-Rebuild, SHA256 von `spike/src/wasm/numtype_core.wasm`):
  `24a048c767f3949ad0a8747cecccc0e25e25bdad859c5deb45e218a39d70cea2` (seit WASM-Parität S0/sqrt
  2026-07-23, von `0b9df4f1…` Kern 11 — der neue `nt_sqrt_strided`-Kernel ändert den Hash legitim;
  additive-only-Dekomposition, Baustein A hat den Pre-Edit-Rebuild == altem Pin selbst reproduziert.
  Threads-Artefakt `9743338d…`, bewusst KEIN persistierter Pin — test:threaded beweist seine
  Bit-Identität zum stable Core. CI-Gate `check:freeze` mit plattform-gelabelter Pin-Menge).
- **check:diag** Haupt-Pin **206,850 @ 140 Files** (nur Root-Korpus; seit WASM-Parität S0/sqrt
  2026-07-23, von 206,854 @ 140 — **Δ−4**: die sqrt-Typkosten minus der threaded.ts-Omit-Ersparnis
  (D10), Dateiset unverändert 140, von allen drei Verifiern reproduziert; Details
  docs/wasm-parity-sqrt-ergebnisse.md). **Vorheriger Stand: 206,854 @ 140** (seit der topk-Umsetzung
  2026-07-23, von 206,801 @ 140 — **Δ+53 reine Typkosten** des in-place-Heap-Körpertauschs plus
  der Testdatei-Erweiterung, KEIN Order-Noise (Dateiset unverändert 140, keine neue Datei);
  Baseline im frischen Worktree reproduziert, im Haupt-Baum bestätigt und von allen drei Verifiern
  unabhängig nachgemessen. Details docs/op-topk-selection-ergebnisse.md, Phase-2-Abschnitt).
  Vorheriger Stand: **206,801 @ 140** (seit der topk-Messung 2026-07-22, von 199,877 @ 139 —
  zerlegt: **+6,611 reines Order-Noise** durch die eine neue Datei, **+313 echte Typkosten** des
  Bench-Skripts; im frischen Worktree gemessen, im Haupt-Baum exakt reproduziert). Davor:
  **199,877 @ 139** (seit der Scale-Probe, von
  201,455 @ 137 — die Dateizahl steigt um die zwei neuen bench-dx-Skripte, der WERT sinkt: per
  empty-then-fill zerlegt in −2,410 Order-Noise (zwei zusätzliche Dateien verschieben die
  Prüfreihenfolge) und +775 echte Typkosten, plus +57 aus der Hover-Gate-Reparatur; Baustein A
  hat die Zerlegung unabhängig aus einem frischen Worktree reproduziert und nebenbei belegt,
  dass der w8-Sentinel zu diesem Korpus exakt 0 beiträgt. Details in
  docs/scale-probe-ergebnisse.md). Historie: **201,455 @ 137** war der W5-Stand, von 195,481 —
  Δ+5,873, Aufschlüsselung in docs/op-w5-item-ergebnisse.md — davon nur +623 Quellcode, der Rest
  Test-/Typ-Pin-Kosten; enthält den D6-Befund „`Equal<ItemGuard<...>>`-Message-Pins sind pro
  Pin ≈1,700 teuer", FOLLOWUPS trackt weitere Untersuchung) · **check:diag:stress 106,239 @ 82**
  (seit WASM-Parität S0/sqrt 2026-07-23, Δ−159 aus der threaded.ts-Omit-Ersparnis, D10; davor
  106,398 @ 82 seit W5, von 105,758 — Δ+640, ausschließlich aus den geteilten spike/src-Änderungen, kein
  stress-eigenes File berührt) · **check:diag:browser 2,142 @ 75** (unverändert seit W1,
  stress/browser ungated by design, `pnpm check` compoundet alle drei).
- **Testzahlen:** test:core 1591 · test:resident 4345+2 · test:threaded 75 (+4 sqrt-Parität +2 Spezialwerte) ·
  test:browser 4 · test:package 3 + Typ-Smoke · cargo 169 (+8 sqrt-Kernel) · test:example (Registry-Install +
  Example-Typcheck + 8 asserted Queries).
- **Editor-Gate:** `bench:editor` W1–**W8** — Instantiation-Pins exact-match hart (seit
  **WASM-Parität S0/sqrt 2026-07-23** uniform **−159** neu gesetzt = `{w1 27.745, w2 29.554,
  w3 60.694, w4 27.908, w5 33.199, w6 34.369, w7 26.917, w8 34.784}`; Grund ist der threaded.ts-Omit-Fix
  (D10), der die `keyof`-getriebene Generic-Fixkosten aus JEDEM Workload entfernt — davor seit V0
  uniform +135 über die 7 Altworkloads plus w8 als Scale-Sentinel (Rang 24 + item/slice/topk, mit
  Toggle-Ziel — schützt die publizierte Skalen-Aussage im Dauerbetrieb, weil der volle Sweep nur
  on-demand läuft)), Latenz am 2x-Ceiling, Correctness wirft. **V0 (2026-07-21)** war eine Mess-BASIS-Reparatur, keine
  Klassen-Surface-Änderung: `gen-workloads.ts` erzeugte tsconfigs ohne
  `spike/src/ambient.d.ts`, weshalb jedes Workload-Programm mit 7x TS2591 lief (das Repo hat
  bewusst kein `@types/node`); `enforceHardGate` liest `hadTypeErrors` nirgends, darum fiel es
  nie auf. Latenz-Mediane und `check:diag` (201.455 @ 137, Δ0) blieben unberührt, die im
  README publizierte Hover-Aussage 0,04–0,08 ms hält auf der sauberen Basis.
- **Mess-Regeln (tragend):** Der Instantiation-Counter ist CHECK-ORDER-abhängig — Pins sind nur
  für ein FIXES File-Set exakt. Datei hinzufügen/umbenennen (selbst ein leeres `export {}`)
  verschiebt den Wert um **bis zu ±≈7,000** (Order-Noise, keine Typkosten; die früher hier
  dokumentierte Spanne „±≈2,000" ist seit der topk-Messung 2026-07-22 empirisch WIDERLEGT —
  eine einzige LEERE `export {}`-Datei bewegte den Zähler um **+6,611**, zweifach reproduziert,
  während die echten Typkosten derselben Datei bei +313 lagen) → per empty-then-fill
  dekomponieren; Datei-EDITS können echte Typkosten sein → bisektieren mit gleichlanger
  Kommentar-Kontrollprobe. Dritter Mechanismus (W1/W2): geteilte KLASSEN-SURFACE rippelt auch
  in Korpora mit unberührtem File-Set (generische Member/Overloads; niladische Member nicht).
  **Vierter Mechanismus (WASM-Parität S0, 2026-07-23): `keyof`-getriebene Generic-Alias-Neuauflösung**
  — ein zusätzliches Mitglied auf einer Schnittstelle, über die ein `Omit`/`Pick`/`Exclude`/Mapped-Type
  `keyof` frisch auflöst (hier `Omit<ThreadedCoreExports,"memory">` in threaded.ts:413), kostet +N
  Instantiations pro Member, fix (arity-unabhängig), kumulativ, uniform in JEDER Kompilation, die die
  Schnittstelle kompiliert — sichtbar auch bei niladischen Membern (anders als der dritte Mechanismus).
  Bei WASM-Parität war es +7/Member (`tsc --generateTrace` isoliert; kontrollierte Varianten:
  0/1/8-Parameter alle +7, zwei Member +14, Member auf einem standalone-Interface +0). Reduzierbar/
  vermeidbar durch Umgehen des `Omit` (S0/D10 tat das → +0/Member kampagnenweit).
  Nie über den Root/Stress-Split hinweg vergleichen. Baselines nur im frischen `git worktree`
  messen; immer Exit-Code + Fehlerausgabe prüfen.
- **Harte Arbeitsregeln (aus der W-Serie, 2026-07-21):** (1) Mutanten-Revert im Haupt-Tree NUR
  als inverser Edit oder Backup-Kopie-Restore (`cp` nach /tmp, zurück, `diff`-Beweis) — NIE
  `git checkout`/`git restore` auf Dateien mit uncommitteter Arbeit (zerstört die Scheibe;
  Recovery-Pfad: Implementierer-Agent per SendMessage re-applizieren lassen). (2) Bei
  Overload-Sets trägt der ZULETZT deklarierte Kandidat die Fehlerdiagnose — Guard-Träger immer
  zuletzt; Diagnose-INHALTE pinnen (tsc auf Außer-Repo-Fixture), nicht nur Fehler-Existenz.
  (3) Neue Typ-Folds: IsUnion-Gate VOR jeder naked Destrukturierung (Misch-Verdikt-Unions
  rutschen durch uniform-only-Guards; W4-M2-Blocker). (4) Markdown: auch ABSICHTLICHE
  `~~…~~`-Strikethroughs reißen das 0-`<del>`-GFM-Gate — ganz vermeiden. (5) Shell: der
  nvm-chpwd-Hook (`nvm use` bei .nvmrc-Fund, ~/.zshrc) wirft in der AGENTEN-Shell bei
  `cd` ins Repo Exit 3 — Ursache ist dort uninitialisiertes nvm (findet sein
  Versions-Lager nicht; die Meldung „not yet installed" ist irreführend, v24 IST
  installiert; im interaktiven Owner-Terminal läuft der Hook normal). Darum:
  `cd <root> 2>/dev/null; …` (Semikolon) oder `git -C`; `&&` direkt nach cd bricht
  die Kette ab. Kein `nvm install` als „Fix" — löst es nicht (2026-07-21 verifiziert).
  Zwei weitere zsh-Fallen (beide 2026-07-23 in CI-Watchern getroffen): **`status` ist read-only**
  (Alias für `$?`) — `status=$(…)` wirft „read-only variable", nie an `status` zuweisen; und der
  Pipe-Exit-Array heißt **`$pipestatus[1]`**, nicht `${PIPESTATUS[0]}` (bash-Name, in zsh leer). Für
  „auf CI warten" `gh run watch <run-id> --exit-status` statt einer eigenen Poll-Schleife.
- **Arbeitsregeln aus der Scale-/topk-Serie (2026-07-22):** (6) **Ein Gate muss die GESUNDHEIT
  seines eigenen Messlaufs mitprüfen, nicht nur dessen Kennzahl.** `enforceHardGate` pinnte über
  Monate Instantiation-Zahlen aus Programmen, die mit 7x TS2591 fehlschlugen, weil es
  `hadTypeErrors` nie las — und eine daraus abgeleitete Aussage stand bereits publiziert in der
  README. Wer eine Kennzahl pinnt, pinnt im selben Gate Exit-Code/Fehleranzahl mit; sonst ist
  der Pin ein Stabilitätsbeweis für ein kaputtes Setup. (7) **Eine vorregistrierte
  Entscheidungsregel wird als Skript nachgebaut und gegen synthetische Ergebnisse gefuzzt,
  BEVOR echte Zahlen existieren.** Die topk-Regel wurde in vier Runden achtmal gebrochen (kein
  Verdikt; zwei Verdikte für dieselben Zahlen; „ersetzen" trotz null Gewinn; ein Hybrid, der
  schlechter ist als Nichtstun; zwei Läufe ohne verdikt-tragenden benannt) — jeder Fehler hätte
  danach ein mechanisch berechnetes, eindeutig AUSSEHENDES Verdikt geliefert. Gelesene Regeln
  wirken eindeutig; durchgespielte verraten ihre Lücken. Zwei der gebrochenen Fassungen stammten
  vom Orchestrator selbst — Selbstprüfung ersetzt den frischen Kontext hier nicht. (8)
  **Informelle Sondagen sind keine Messungen und dürfen keine Folgearbeit begründen.** Die
  topk-Sondage lag bei `k = n` um Faktor 13 daneben und bei `k = n/2` mit falschem VORZEICHEN —
  Ursachen: 2 Aufwärm-Aufrufe statt adaptivem Warmup, JIT-Kontamination durch vorherige
  Fuzz-Läufe im selben Prozess, JS- statt typisierte Arrays. Dasselbe Muster wie Kern 06. Je
  aufwendiger die Folgearbeit, desto weniger denkt jemand an die Ausgangszahl zurück. (9)
  **Delegations-Regel:** Agenten, die einen langlaufenden Hintergrundprozess starten, beenden
  ihren Turn regelmäßig auf „ich warte, bis es fertig ist" statt auf einem Ergebnis (zweimal in
  einer Session passiert). Die Befunde existieren dann, sind aber unberichtet. Entweder den Lauf
  selbst im Haupt-Loop überwachen oder den Agenten per SendMessage gezielt zum Nachbericht
  auffordern — nie annehmen, dass ein gestarteter Lauf auch dokumentiert wurde.

## Commands

`pnpm check` — Typ-Verbund, DREIfach: Root + `spike/tests-stress` + `spike/tests-browser`
(Nicht-Vakuität aller Legs per Korruptions-Tests bewiesen; Z2) · `pnpm check:diag` /
`check:diag:stress` / `check:diag:browser` — Instantiation-Messung je Korpus (Pins oben) ·
`pnpm test:core` — v1-Differential + Meta-Guards · `pnpm test:resident` (+`:gc` mit
`--expose-gc`) · `pnpm test:threaded` — baut BEIDE Artefakte (vergleicht bit-identisch gegen den
STABLE-Core), braucht stable 1.95.0 UND die pinned nightly-2026-07-09 (+rust-src; Install-Command
in `scripts/build-wasm-threads.sh`) · `pnpm test:browser` — Playwright/Chromium-Smoke, ≈3 s; der
Wrapper emittiert IMMER frisch — **nie `playwright test` direkt aufrufen** (mtime-Freshness-Guard
wirft sonst); Erstinstallation `pnpm exec playwright install chromium` · `pnpm test:package` —
baut `build:dist`, dann Emit-Präzisions-Gate (`scripts/check-dist-emit.mjs`) + Laufzeit-Smoke
gegen `dist/index.js` + Konsumenten-Typ-Smoke; braucht KEIN nightly; läuft auch als
`prepublishOnly` bei jedem `npm publish` · `pnpm build:dist` — publizierbares Paket nach `dist/`
(gitignored, kommt via `files` in den Tarball) · `pnpm demo` — alle drei Backends, asserted
equal · `pnpm bench:scaling|chain|strided|blocked|slice|threaded|elementwise` ·
`bench:crossover` (kalibriert die Auto-Weiche, nightly) · `bench:editor` (LSP-Harness, ≈1,2 s,
7 Workloads — hartes Gate) · `cargo test --manifest-path crates/core/Cargo.toml`.

Build-Note: wasm-Builds brauchen das repo-root `.cargo/config.toml` (simd128-Rustflag), und
Cargo-Config-Discovery ist CWD-basiert — **alle Commands aus dem Repo-Root** (ein
`compile_error!`-Guard feuert sonst). Test-Skripte nutzen EXPLIZITE File-Listen in package.json —
neue Testdateien manuell eintragen; `test-scripts-guard.test.ts` (in test:core) failt bei
unregistrierten/doppelt gelisteten/fehlenden Dateien, Invariante (d) deckt auch
`spike/tests-browser` ab.

## Frozen-baseline discipline (hard, since Kern 06)

New code in files shared with frozen kernels/entry points (abi.rs, matmul_blocked.rs, shape.rs) must be APPENDED strictly after all pre-existing content — mere line shifts change `#[track_caller]` panic-location metadata and thus the compiled bytes of UNTOUCHED functions. The binding freeze proof is the artifact hash from a clean rebuild (SHA256 of spike/src/wasm/numtype_core.wasm), not a plus-lines-only git diff. Pin from Kern 11 (superseded 2026-07-23 by WASM-parity S0/sqrt's `24a048c767f3949ad0a8747cecccc0e25e25bdad859c5deb45e218a39d70cea2` — the new `nt_sqrt_strided` kernel changes the hash via the same additive-only decomposition): `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` (Kern 11 added a genuine contiguous fast path INSIDE `add_strided`/`binary_strided` themselves — not just new exports like Kern 07 — so the hash legitimately changes again: proof decomposes into (1) pre-edit clean rebuild reproduced the prior Kern-07 pin `7a65d800…` exactly, (2) the new fast path is guarded by a post-validation, post-broadcast condition (`a_shape==b_shape && offsets==0 && both natural strides`) with the pre-existing general `unravel`-based loop otherwise byte-for-byte unchanged, (3) behavioral pins green (`test:core` 817, `test:resident` 4265+2, `cargo` 159 incl. 2 new fast-path-vs-general-path equivalence tests), (4) a mutation proof (`a[i].max(0.0)` in both fast paths) broke 3/159 cargo tests and 158/4267 TS tests, revert restored full green, (5) new hash becomes the pin — docs/kern-11-elementwise-fastpath-ergebnisse.md). Kern-07 lesson still applies generally: a phase that legitimately ADDS exports (not just kernel-body changes) also changes the hash by the same decomposition; append-only is an artifact-bytes argument — TS CLASS BODIES with private constructors take the intent-preserving equivalent instead: insertion-only diffs, zero edits to pre-existing members; specs should say which discipline applies to which file. Threads-path extras: env RUSTFLAGS REPLACES the config-file rustflags (always carry `+simd128`); the threads artifact builds via scripts/build-wasm-threads.sh on pinned nightly-2026-07-09 (+rust-src) with its own target-dir; `thread_local!` is forbidden in the crate (`__tls_base` only initializes in the winning instance); never cache memory.buffer/views — with shared memory this fails SILENTLY (stale length, no detach).

## Toolchain note (2026-07-09)

Installed TypeScript is **7.0.2** — the native (Go) compiler generation, now `latest` on npm (6.0 is still `beta`). All researched recursion/instantiation limits below were documented for TS 5.x; treat them as hypotheses to verify empirically on 7.x, not as facts. `--extendedDiagnostics` works on 7.0.2.

## Obligatory workflow: capture findings in the coding-kb (user-mandated, 2026-07-09)

Every substantial slice of work (spike, phase, verification pass, benchmark) ends with a knowledge capture —
it is part of the Definition of Done, not optional:

1. **In-project:** findings go into the phase's results doc (`docs/*-ergebnisse.md`) — grounded in commands
   actually run, honest about failures and gaps (see the docs' "honesty rule").
2. **Cross-project:** every *general* lesson (disproven assumption, non-trivial gotcha, failed→working
   approach, transferable technique) is upserted into the coding-kb Obsidian vault as an atomic note —
   revise a related existing note rather than duplicating; correct notes that turned out wrong or imprecise
   (e.g. replace single-point figures with measured ranges). Follow the vault's
   `90-Meta/Capture-Workflow.md` (template, `status: seedling`, tags, `projekte: [numtype]`).
3. **Wire it:** link the note into the fitting MOC(s), then rebuild the graph (root op, command in
   Capture-Workflow.md) and verify the new note's link edges via a `coding-kb` query.
4. Before starting non-trivial work in a known domain, **consult** the KB first (`find` → `neighbors` →
   `read`) — do not re-derive what is already written down.
5. **Doc-Platzierung (Hausregel seit 2026-07-19):** Eine fertige Scheibe aktualisiert
   docs/roadmap.md (Item-Status), die Sektionen „Status" + „Aktuelle Pins & Gates" HIER (nur
   Einzeiler + IST-Zahlen) und hängt ihr volles Narrativ an docs/projekt-log.md an — nie mehr
   als Absatz-Append in diese Datei. CLAUDE.md trägt Regeln + aktuellen Stand, das Log die
   Historie.

## Key TS limits to respect (researched on TS 5.x, sourced in docs/wettbewerbsanalyse-und-usp.md §4)

- ≈100 instantiation depth non-tail-recursive; ≈1000 tail-recursive → write ALL recursive types accumulator/tail-recursive.
- Global ≈5M type-instantiation budget per compilation (TS5, version-fragile) — track via `pnpm check:diag`.
- Tuple-length arithmetic is fine for *ranks* (small ints), never for large *dimension values*. Since Kern 05, arithmetic over large literal dims IS available via digit-string types (`spike/src/slice-literal.ts`: subtraction + comparison O(digit-count); since Spike 04 also addition + schoolbook multiplication O(digits²), surfaced as `LiteralShapeProduct<S>`; since Spike 06 also ceil long division `DivCeil`, ≤10 bounded trial-mults per digit) — never fall back to tuple-length-per-value. Comparison alone unlocks bounds checks incl. NEGATIVE literals (Spike 03), and compare+subtract+clamp covers negative-index NORMALIZATION too (Spike 06 — no signed arithmetic exists or is needed yet) — check whether a use case needs only a verdict before building arithmetic. Products must cap at MAX_SAFE_INTEGER: beyond it, `${infer N extends number}` double-rounds through float64 to a WRONG literal — degrade to `number` instead.
- Template-form classification facts (empirical, Spikes 03–06): `${N}` renders plain digits below 1e21 (`${1e20}` = "100000000000000000000"; ≥ 1e21 → "1e+21" — exponent forms are unprovable, NO claim ever); dot-form (contains ".", no "e") is a PROVEN non-integer regardless of sign (−1.5 included) — a guaranteed throw wherever the runtime checks Number.isInteger. Spec-drafting lesson (Spike 06): a spike that MOVES a supported-literal boundary must expect pre-existing pins of the OLD boundary to be re-expressed intent-preservingly — plan for it, don't demand "all old tests untouched".
- TS 7.0.2 native charges instantiations for merely DECLARING template-literal-heavy generic aliases that nothing references (Spike-04 bisection: hundreds per digit-pattern helper, `IsUnion` +0; ≈+2k per spike, observed twice, Spikes 03/04) — "unused machinery is free" is empirically false. Budget-gate lesson (Spike 04): pre-register ABSOLUTE affordability gates (budget share, wall time, editor gate) and let measurements set the regression pins (current pins since Item 10: main `check:diag` 175,634 @ 132 files = ≈3.5% of budget — realistic/semantic corpus only; progression 173,716 Infra01 → 174,213 Kern09 → 174,391 Kern10 → 172,392 Kern11 (a DECREASE from adding one bench file — mechanism PINNED 2026-07-12: the instantiation counter is CHECK-ORDER-dependent, so adding ANY file, even an empty `export {}`, reshuffles the fresh-vs-cached partition of shared recursive instantiations and moves the total ±≈2,000 depending on the file's sort-position; the pin is exact only for a FIXED file set — file-adding slices carry order-noise, decompose via empty-then-fill; retroactively explains Infra-01's super-additive removal) — plus `check:diag:stress` 94,597 for the ≥13-digit/cap stress cases in spike/tests-stress; the two corpora are measured separately by design, never compare deltas across the split; `pnpm check` compounds BOTH so nothing rots); never gate on a guess of the quantity the spike exists to measure, and never blend mandated stress probes with realistic sites into one gated mean.
- When classifying results of distributive type helpers (e.g. digit `Compare`): union inputs distribute and can yield union verdicts — a naive `extends "lt"` check misclassifies `2 | 7`. Accept a verdict only via tuple-wrapped SUBSET checks (`[C] extends [("eq" | "gt")]` = uniform), else degrade to no-claim. Also: lifting a runtime error to compile time makes your own runtime error-path tests stop compiling — widen deliberately (`5 as number`) + add a parity test.
- TS 7.0.2 native reports only ONE diagnostic per call per compile pass when MULTIPLE arguments of the same call are invalid (general checker behavior, affects too-many-specs and bounds errors alike); the guard TYPE flags all positions — pin via type-level assertions, not by counting squiggles.
- Variance (Spike 05): an `out S` annotation is enforced ABSTRACTLY (sub-S/super-S with type variables) — factually-monotone type functions like `Transpose` still fail it (TS2636), so an enforced-covariant view must be computation-free. AND the enforcement has a syntax loophole: METHOD-SHORTHAND parameters are checked bivariantly (`resizeTo(s: S): void` compiles under `out`; only the property-style `resizeTo: (s: S) => void` errors) — house rule: `NDArrayView` never gains an `S`-consuming member, future function members property-style. `readonly shape: S` does not deep-freeze tuple elements (`nd.shape[0] = 99` type-checks — known latent hole, FOLLOWUPS).
- Use `const` type parameters (TS 5.0+) so callers never need `as const`.
- Shape errors must surface at the offending argument with the shapes named in the message; hovers must show clean resolved tuples like `NDArray<[2, 4]>`.

## Qualitätssicherung, modellunabhängig (Owner-Mandat 2026-07-12, nach Kern-09-Review)

Diese Regeln gelten in JEDER Session — sie existieren, weil eine Opus-Session die
Substanz hielt, aber globale standing orders und Prozess-Feinheiten verlor
(Findings F1–F3, docs/kern-09-keepdims-ergebnisse.md):

- **Nicht-Fable-Session?** Zu Beginn substanzieller Arbeit den Skill
  `brainroute:fable-doctrine` laden. Substanzielle Aufgaben per brainroute
  klassifizieren und delegieren, mit Router-Ansage („Router → tier · Modell ·
  Effort" + Einzeiler warum).
- **Strukturfragen zuerst über den Graph** (`graph-a-lama`: outline/def/usages/
  callers), Datei-Reads nur bei echtem Ganzdatei-Bedarf.
- **Mess-Hausregel:** Baselines/Pins (check:diag etc.) nur in einem frischen
  `git worktree` des Zielcommits messen — `git stash` lässt untracked Dateien
  liegen und kontaminiert den Korpus. IMMER Exit-Code + Fehlerausgabe prüfen,
  nie nur die Kennzahl greppen. Diagnosen über Umgebung/Messungen gelten als
  unverifiziert, bis ein isolierter Kontrolllauf sie bestätigt.
- **Abweichungs-Eskalation:** Jede Abweichung von einer Hausregel wird VOR der
  Implementierung dem Owner vorgelegt (nicht nur in der Spec dokumentiert).
  Standard ist „disclosed + confirmed", nicht „disclosed".
- **Hintergrund-Agenten fassen den Haupt-Working-Tree nie an** — kein `git
  stash`/`git checkout` dort; Messungen und Mutanten in eigenen worktrees/
  Scratch-Kopien.
- **Spec-Verifikation VOR der Implementierung:** Jede bindende Spec durchläuft
  nach der Owner-Richtungsabnahme EINEN adversarialen Fresh-Context-Verifier
  (`brainroute:deep`) GEGEN die Spec — Design brechen, die Code-Annahmen der Spec
  am echten Code prüfen, Design-Löcher/Testplan-Lücken/Freeze-Behauptungen — BEVOR
  eine Zeile Code entsteht. Befunde mergen, Design-Blocker mit dem Owner in die
  Spec einarbeiten (Richtungsänderungen abnehmen lassen), erst dann implementieren.
  Auftrag aus docs/verify-runde-template.md „Baustein 0", nicht frei formulieren.
  (Owner-Mandat 2026-07-12, nach der Item-10-Spec-Review, die drei Blocker vor dem
  Bau fing — u. a. eine falsche Code-Annahme, in die sonst blind gebaut worden wäre.)
- **Verify-Runde = IMMER zwei Verifier:** einer prüft gegen die Spec
  (Konformität, coverage-first, eigener Mutant), einer arbeitet adversarial
  (aktiv brechen: Grenzfälle, Messrandbedingungen, Mutanten auch abseits der
  Spec). Aufträge aus docs/verify-runde-template.md, nicht frei formulieren.
- **Covenant (seit v1, Commit 4db74e0):** COVENANT.md ist der stehende
  Produkt-Vertrag (S1, M1–M5, Z1–Z2 + Nicht-Ziele). Vor jeder „fertig"-Meldung
  einer substanziellen Scheibe zusätzlich: `graph-a-lama query lint` als
  mechanisches Gate im Gate-Block (S-Invarianten; Exit 1 = roter Test) und EIN
  `covenant:covenant-verify`-Agent PARALLEL zu Baustein A/B (dritter frischer
  Kontext, sieht nur COVENANT.md + Diff + berührte IDs + Lint-Output; Auftrag
  aus verify-runde-template.md Baustein C); Verdikt ins
  Post-Verification-Addendum. Scheiben-Specs und Delegations-Prompts benennen
  die berührten Invarianten-IDs; Baustein 0 gleicht jede neue Spec gegen den
  Vertrag ab. Spec-Änderungen NUR mit Owner-Bestätigung + Version-Bump +
  Changelog — nie still, auch wenn die Spec „offensichtlich veraltet" wirkt.
  **Eskalationsleiter (Owner-bestätigt 2026-07-13) — nie vorsichtshalber den
  vollen Katalog fahren:** Stufe 0 trivial (Typo/Kommentar/Doku-only, kein
  Anker berührt) → nichts. Stufe 1 klein & anker-frei → nur das mechanische
  Lint (läuft ohnehin im Gate-Block mit). Stufe 2 Anker berührt, aber keine
  Scheibe (FOLLOWUPS-Minis) → Lint immer; covenant-verify nur, wenn die
  Invariante INHALTLICH tangiert sein könnte, sonst Ein-Satz-Begründung im
  Commit (dokumentiertes Ermessen). Stufe 3 substanzielle Scheibe (bindende
  Spec) → voller Katalog A+B+C parallel + Lint im Gate-Block. Dieselbe Leiter
  galt für A/B schon immer (Zwei-Verifier-Regel nur für substanzielle
  Scheiben, Baustein 0 nur für bindende Specs).
