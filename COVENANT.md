# Covenant — NumType
<!-- covenant:version 7 -->

## Invarianten

### Strukturell (mechanisch geprüft)
- **S1** · Runtime-Quellcode importiert nie aus Test-, Bench- oder Demo-Verzeichnissen.
  → Regel `covenant-s1` · Anker: `spike/src/`

### Semantisch (geprüft per Verify)
- **M1** · Jeder WASM-Kern ist bit-identisch zur naiven TS-Referenz (`runtime.ts`), auch für
  IEEE-Spezialwerte; Optimierungen nur unter dem Bit-Identity-Law (Vektorisierung nur QUER zu
  Output-Elementen, aufsteigende k-Akkumulation, kein FMA/relaxed-simd). Kernel-lose
  Referenz-Ops (Referenzfunktion in `runtime.ts` ohne WASM-Kern-Gegenstück) sind zulässig,
  solange die Paritätslücke in FOLLOWUPS getrackt ist; M1 bindet in dem Moment, in dem ein
  Kernel für die Op entsteht. (Präzisierung v5 — Präzedenzfall W1 argmax/topk.)
  · **Präzisierung v6 — NaN-Payloads:** bit-exakt gilt IMMER für finite Werte, ±0 und ±Inf.
  Für Kerne, die ihr Ergebnis durch Gleitkomma-ARITHMETIK erzeugen (`add`/`sub`/`mul`/`div`/
  `sqrt`/`sum`/…), zählt NaN nur als Wert-Klasse — der Payload ist implementierungsdefiniert
  (WASM-spec-konform). Für Kerne, die Werte OHNE Arithmetik weiterreichen (reiner
  Element-Kopiervorgang/Datenbewegung: `topk`, `stack`, `item`, der Kern `nt_transpose` —
  nicht zu verwechseln mit `WNDArray.transpose()`, das eine kernel-lose O(1)-View ist), gilt
  Bit-Identität einschließlich des exakten NaN-Payloads und ist per eigenem, benanntem Test
  zu belegen. (Präzedenzfälle: sqrt/div/mul auf der einen, topk/stack/item auf der anderen
  Seite; S5 lieferte den Gegenfall, der die Grenze sichtbar machte.)
  · **Präzisierung v6 — komponierte Ops ohne eigenen Kern:** eine WASM-berechnete Op, die aus
  bereits bit-identisch bewiesenen Kernen KOMPONIERT ist, erbt deren Pflicht; ein eigener
  Kern-Beweis ist nicht nötig, solange ein direkter Differentialtest gegen die
  Referenz-Runtime existiert. Der Preis ist eine eigene Testpflicht für den Lebenszyklus des
  Zwischenergebnisses (ein Leck dort ist für Korrektheits-Assertions unsichtbar).
  (Präzedenzfälle: `mean` = `sum ∘ scalar_div`, `stack` = N × `nt_materialize`.)
  · **Präzisierung v6 — resident, aber ohne WASM-Aufruf berechnet:** eine residente Op, die
  vollständig in TS berechnet wird (kein `nt_*`-Aufruf) und bit-identisch gegen die Referenz
  bewiesen ist, ist KEINE kernel-lose Referenz-Op im Sinne des v5-Satzes — es existiert keine
  Paritätslücke, also auch keine FOLLOWUPS-Trackingpflicht. (Anlass-Präzedenzfall: `item`.
  Die Klausel gilt allgemein, nicht nur dort: `WNDArray.transpose()` und `.slice()` teilen die
  Eigenschaft „kein `nt_*`-Aufruf" schon länger als O(1)-Views — sie wurden nur nie durch die
  M1-Brille der Paritäts-Kampagne betrachtet.)
  Anker: `crates/core/src/`, `spike/src/runtime.ts`
- **M2** · Typ-Ebene „never wrong, only incomplete": Compile-Ablehnung nur für garantierte
  Runtime-Throws; wide/Union/dynamischer Rang degradieren zu no-claim — nie ein
  konfident-falscher Claim.
  · **Präzisierung v6 — Geltungsbereich:** M2 ist eine Aussage über die
  **Shape-Guard-Maschinerie**. Generische TS-Overload-/Signatur-Auflösung außerhalb der
  Guard-Typen ist nicht M2-Gegenstand (Präzedenzfall W2: `x: number | NDArray<[3]>` wird als
  Union über die Overload-Grenze mit TS2769 abgelehnt, obwohl jeder Member für sich valide
  wäre — ohne jede Beteiligung von `Guard`/`OkShape`).
  · **Präzisierung v6 — Rekursionsgrenze:** die Garantie gilt, solange der Checker die
  Berechnung überhaupt ABSCHLIESST. Jenseits der TS-Rekursionsgrenze ist die Ablehnung eine
  Compiler-Grenze, keine Aussage der Guard-Maschinerie (gemessen, Scale-Probe: Rang 768 und
  896 passieren, Rang 1024 bricht mit TS2589 auf GÜLTIGEM Code ab). Praktisch irrelevant für
  reale Arrays, aber die publizierte „never wrong"-Aussage benennt ihren Bereich ehrlich.
  Anker: `spike/src/dim.ts`, `spike/src/literal-arithmetic.ts`, `sym:Guard`, `sym:OkShape`
  · GESCHLOSSEN in Item 11 / S1 (2026-07-17): der `Literal|undefined`-Verstoß durch OPTIONALE
  Parameter (`sum`s `axis`/`keepdims`) ist behoben. Der `sum`-Overload-Umbau (Overloads nach
  Argument-Anzahl 0/1/2 — keine optionalen Parameter mehr in der Mehr-Argument-Form — plus
  `reduce.ts`-`KeepDims`-Erweiterung auf `boolean | undefined`) verhindert das
  `undefined`-Stripping: `a.sum(u)`/`u:0|undefined` degradiert jetzt zu no-claim
  (`readonly number[]`), `a.sum(0,kd)`/`kd:true|undefined` zu einer ehrlichen Shape-Union
  (`readonly [3] | readonly [1,3]`). Der frühere `UA_GAP`-Sentinel-Pin ist umgekehrt
  (`UA_AXIS_CLOSED` + `UA_KEEP_CLOSED`/`WUA_*`, spike/tests/ndarray.test-d.ts) und bewacht
  künftig die Schließung. Dreifach verifiziert (Spec CONFIRMED + adversarial HÄLT +
  covenant-verify kein Verstoß, 2026-07-17).
- **M3** · Shape-Fehler erscheinen AM fehlerhaften Argument, Message-Stamm wortgleich zum
  Runtime-Throw; Klassen-Hover bleiben saubere Tupel (`NDArray<[2, 3]>`). „Klassen-Hover"
  meint die Typ-Parameter-Anzeige der Klasse; Member-Hover (z. B. `.shape`) dürfen
  readonly-Modifier tragen. (Präzisierung v5 — schließt die seit D-V2 offene
  Auslegungsfrage.)
  · **Präzisierung v6 — Methoden-Rückgabetypen (Owner-entschieden 2026-07-25):** die
  Hover-Norm gilt AUCH für konsumentenseitig erreichbare Methoden-Rückgabetypen. Ein
  Top-Level-Alias in RÜCKGABE-Position, den die Quick Info nicht auflöst, ist ein Verstoß;
  ein Alias in TYP-ARGUMENT-Position, der aufgelöst wird (z. B.
  `WNDArray<OkShape<Broadcast<S, B>>>`), bleibt konform. Damit ist die S3-Einstufung
  rückwirkend bestätigt — der dortige Fix war Pflicht, nicht Kür. Berührt eine Scheibe diese
  Fläche, ist eine **LSP-Messung mit Kontrollpunkt Pflichtbestandteil des Verify-Katalogs**
  (Arbeitsregel 13): Quelltext-Lektüre findet diese Klasse nachweislich nicht.
  · **Präzisierung v6 — native Diagnosen als Ausnahme-Klasse:** Message-Stämme sind
  wortgleich, WO der Code die Message formuliert. TS-native Diagnosen (Arity via TS2554,
  Overload-Auflösung) und Compile-only-Rejections ohne Runtime-Gegenstück sind benannte, im
  Code dokumentierte Ausnahmen (Präzedenzfälle W4/W5).
  · **Präzisierung v6 — Cross-Surface-Parität als eigene Achse:** M3 adressiert
  Typ-vs-Runtime INNERHALB einer Fläche. Tragen mehrere Flächen dieselbe Op mit bewusst
  duplizierter Validierungslogik, gilt zusätzlich: die Runtime-Message-Stämme sind ZWISCHEN
  den Flächen wortgleich, per String-Gleichheitstest abgesichert (Präzedenzfall S3 —
  `NDArray` gegen `WNDArray`).
  · **Präzisierung v7 — rekursive Aliase (Owner-entschieden 2026-09-24):** die v6-Präzisierung zu
  Methoden-Rückgabetypen (oben) zielt auf Aliase, die die Quick Info auflösen KÖNNTE und deren Name die eigentliche
  Typstruktur verdeckt (Präzedenzfall S3, `StackResultOf<…>`). Ein Alias ist davon
  ausgenommen und konform, wenn er (a) **genuin rekursiv** ist — seine Definition referenziert
  sich selbst, eine endliche Expansion existiert also nicht, der Name IST die ausgeschriebene
  Form — und (b) **aus dem Paket-Einstiegspunkt exportiert** ist, sodass Konsumenten ihn
  nachschlagen und benennen können. Nicht ausgenommen sind Aliase, die einen rekursiven Typ
  nur UMHÜLLEN (ein nicht-rekursiver Wrapper um einen rekursiven Kern muss weiterhin auflösen)
  sowie generische Aliase, deren Instanziierung für konkrete Typargumente auflösbar wäre. Die
  LSP-Mess-Pflicht bleibt: die Messung belegt, dass für konkrete Typargumente GENAU dieser
  rekursive Name erscheint und sonst aufgelöste Typen. (Präzedenzfall 0b: `NestedValue =
  number | NestedValue[]` als Rückgabetyp von `toNestedArray()` bei statisch unbestimmtem Rang;
  bei bekanntem Rang hovert dieselbe Methode aufgelöst, z. B. `number[][]`.)
  Anker: `sym:Guard`, `sym:ShowShape`
- **M4** · Frozen Baseline: v1-Kerne/-Einstiegspunkte bleiben byte-unberührt; der bindende
  Freeze-Beweis ist der Artefakt-Hash aus einem Clean-Rebuild; abi.rs/matmul_blocked.rs/shape.rs
  nur append-only.
  · **Klarstellung v6 — Geltungsbereich (Owner-entschieden 2026-07-25):** M4 deckt
  AUSSCHLIESSLICH die drei genannten Rust-Anker. Die „insertion-only"-Disziplin für
  TS-Klassenkörper ist eine **Hausregel** (CLAUDE.md), keine Vertragsinvariante — M4s
  Begründung ist ein Artefakt-BYTE-Argument (eine Zeilenverschiebung ändert
  `#[track_caller]`-Panic-Location-Metadaten und damit die kompilierten Bytes UNBERÜHRTER
  Funktionen), und dieser Mechanismus existiert in TypeScript nicht.
  · **Präzisierung v6 — Host-Reproduzierbarkeit des Freeze-Beweises (Owner-entschieden
  2026-07-25):** „Clean-Rebuild" heißt **host-UNABHÄNGIGER** Clean-Rebuild — dieselbe
  Toolchain-Version baut auf jeder unterstützten Plattform dasselbe Artefakt. Die Eigenschaft
  war seit Item 12 mechanisch in Kraft (der CI-Job `freeze` läuft auf `ubuntu-latest` und
  prüft gegen einen auf macOS gesetzten Pin), aber nie benannt; im S5-Abschluss erwies sie
  sich als load-bearing — sie war das einzige Gate, das einen Host-Pfad im Artefakt fangen
  konnte, nachdem ein neu benutztes std-Generikum (`core::slice::sort`) einen unmapped
  Toolchain-Pfad inklusive Benutzername eingeschleppt hatte. Lokal war das unsichtbar, jeder
  andere Gate-Block war grün. **Neu festgeschrieben wird die Konsequenz:** weicht eine
  Plattform ab, die vorher identisch baute, ist das ein **BEFUND** — kein Anlass für einen
  zweiten Pin. Die Pro-Plattform-Menge in `scripts/check-freeze-hash.mjs` ist ausschließlich
  für eine echte Erstplattform gedacht; ihr Hinweistext bietet den bequemen Weg an, und genau
  der wäre hier falsch gewesen.
  Anker: `crates/core/src/abi.rs`, `crates/core/src/kernels/matmul_blocked.rs`, `crates/core/src/shape.rs`
- **M5** · Der Default-`NDArray`-Pfad ist browser-sicher: kein eager Laden von node:*-Builtins;
  Threads ausschließlich als explizites Node-only-Opt-in hinter `backend("threaded")`
  (type-only-Imports und dynamisches `import()` nach Env-Check sind erlaubt).
  Anker: `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`

## Zusagen (Verhalten, das erhalten bleibt)
- **Z1** · Zero-Dependency-Runtime: das Paket bekommt nie ein `dependencies`-Feld; Kernels und
  Typ-Maschinerie bleiben from scratch (Dev-Tooling ist erlaubt).
  Anker: `package.json`
- **Z2** · `pnpm check` bleibt der Verbund aller QUELLTEXT-Typ-Korpora (root + stress +
  browser + künftige) — kein Quelltext-Korpus rottet ungeprüft. Typchecks gegen ein
  BAUERGEBNIS (`dist/`, das beim reinen `noEmit`-Check nicht existiert — z.B. der Item-11/S3
  Konsumenten-Smoke `spike/tests-package/consumer/`) laufen stattdessen im Paket-Testlauf
  (`test:package`, nach `build:dist`) — auch sie rotten nicht, liegen aber bewusst in einem
  ANDEREN, build-abhängigen Gate. Reine Quelltext-Smokes bleiben in `pnpm check` (z.B. der
  Laufzeit-Smoke `package-smoke.test.ts`, dessen dist-Import dynamisch/untypisiert ist).
  · **Präzisierung v6 — dritte Gate-Klasse: on-demand-Mess-Korpora (Owner-entschieden
  2026-07-21):** Korpora, die ausschließlich der MESSUNG dienen und deren voller Lauf zu teuer
  oder auf geteilter CI-Hardware zu instabil für exakte Pins ist (Präzedenzfall:
  `spike/bench-dx/scale-workloads/`), dürfen aus `pnpm check` und aus der CI ausgenommen
  bleiben — **unter der Bedingung, dass ein dauerhaft gepinnter Stellvertreter im CI
  mitläuft** (hier der Sentinel-Workload `w8` im Job `editor-gate`). Verworfen wurden: die
  Abweichung nur offenzulegen (Text und Praxis liefen weiter auseinander) und den vollen
  Sweep in die CI zu nehmen (gemessen 68,51 s Messschleife / 77 s Gesamtbefehl, plus auf
  geteilter CI-Hardware zu stark schwankende Kaltstartzeiten für exakte Pins).
  Anker: `package.json`

## Nicht-Ziele
- Kein NumPy-Vollklon (keine 400 Ops), kein GPU/autograd, keine DataFrames.
- Kein Per-Call-Routing zwischen Backend-Cores (dokumentierte Sackgasse, Kern 06).
- Kein Browser-Port des Threads-Pfads in v0 (COOP/COEP-gated; Owner-Option 1).
- Keine transzendenten Ops ohne eigene Determinismus-Entscheidung (brechen Bit-Parität).

## Änderungslog
- v7 (2026-09-24) · **M3: rekursive Aliase in Rückgabe-Position präzisiert.** Anlass:
  `covenant-verify`-Befund (Baustein C) der Scheibe 0b (typisiertes `toNestedArray`): bei
  statisch unbestimmtem Rang hovert der Rückgabetyp als `NestedValue` — wörtlich ein
  „nicht aufgelöster Alias in Rückgabe-Position" nach v6, obwohl ein rekursiver Typ
  grundsätzlich keine endliche Auflösung hat. Die v6-Klausel entstand an S3, wo ein
  AUFLÖSBARER Alias die Struktur verdeckte; diesen Unterschied kannte der Wortlaut nicht.
  Owner-Entscheidung 2026-09-24: Wortlaut präzisieren (Option A) statt den Rückgabetyp auf
  `unknown` zurückzunehmen (Option B, hätte die Owner-Entscheidung „`NestedValue`" der
  0b-Spec revidiert). Eng gefasst (rekursiv UND exportiert; Wrapper und auflösbare generische
  Aliase bleiben Verstöße; LSP-Messung bleibt Pflicht) — die Norm-Absicht von v6 ist
  unverändert, die S3-Einstufung bleibt gültig.
- v6 (2026-07-25) · **Sammel-Präzisierung: zwölf über die Kampagnen W1–W5 und S0–S5
  aufgelaufene Auslegungsfragen in einem Zug geschlossen.** Alle stammen aus
  `covenant-verify`- bzw. Baustein-0-Befunden und waren in FOLLOWUPS getrackt; keine davon war
  ein offener Normbruch, alle waren Lücken im WORTLAUT gegenüber gelebter Praxis.
  **M1** (drei): NaN-Payload-Grenze zwischen arithmetik-erzeugenden und wert-weiterreichenden
  Kernen (S5 lieferte den Gegenfall, der die Grenze sichtbar machte); komponierte Ops ohne
  eigenen Kern erben die Pflicht ihrer Bausteine (mean, stack); resident-ohne-WASM-Aufruf ist
  keine kernel-lose Referenz-Op und hat keine Trackingpflicht (item).
  **M2** (zwei): Geltungsbereich auf die Shape-Guard-Maschinerie gescopt (W2-Grenzfall);
  Rekursionsgrenze als Bereichs-Aussage benannt (Rang-1024-Cliff, Scale-Probe).
  **M3** (drei): Methoden-Rückgabetypen ausdrücklich eingeschlossen — **Owner-Entscheidung
  2026-07-25, die die S3-Einstufung rückwirkend bestätigt** (der Vertrag beschreibt damit die
  Norm, die das Projekt nachweislich hält: es hat zweimal danach gehandelt und einmal echtes
  Budget dafür bezahlt); native TS-Diagnosen und Compile-only-Rejections als benannte
  Ausnahme-Klasse (W4/W5); Cross-Surface-Message-Parität als eigene Achse (S3).
  **M4** (zwei, beide Owner-entschieden 2026-07-25): M4 deckt nur die drei Rust-Anker — die
  TS-Insertion-only-Disziplin bleibt bewusst Hausregel, weil M4s Artefakt-Byte-Argument in
  TypeScript keinen Gegenmechanismus hat. UND: „Clean-Rebuild" heißt host-unabhängiger
  Clean-Rebuild; weicht eine Plattform ab, die vorher identisch baute, ist das ein Befund und
  kein Anlass für einen zweiten Pin. Diese zweite Klausel kam als Letzte hinzu — sie war beim
  Schreiben aufgefallen, wurde aber bewusst NICHT eigenmächtig eingebaut, sondern dem Owner
  vorgelegt und auf dessen Entscheidung aufgenommen (ein Vertrag darf nicht dadurch wachsen,
  dass dem Schreibenden noch etwas einfällt).
  **Z2** (eine): on-demand-Mess-Korpora als dritte Gate-Klasse, abgesichert durch einen
  dauerhaft gepinnten Stellvertreter im CI (Owner-entschieden bereits 2026-07-21).
  **Zwei der zehn sind mehr als Klarstellung und werden hier ausdrücklich als solche
  benannt** (Befund des v6-Verifiers — die erste Fassung dieses Logs zählte nur eine):
  (a) die **M3-Erweiterung** auf Rückgabetypen macht eine bisher freiwillig gehaltene
  Qualitätsnorm verbindlich, inklusive LSP-Mess-Pflicht; (b) die **Z2-Erweiterung**
  legalisiert eine Praxis, die vom bisherigen Wortlaut wörtlich abwich — das
  Scale-Probe-Ergebnisdoc sagt das selbst so („Der Sweep-Korpus läuft nur auf Zuruf, nicht in
  CI. Das weicht wörtlich von COVENANT Z2 ab"). Gegenargument, das die Einordnung als
  Präzisierung stützt und der Vollständigkeit halber hier steht: der ausgenommene Korpus ist
  generiert und gitignored, nie handgeschriebener Quelltext — insofern eine natürliche
  Verallgemeinerung der schon bestehenden Bauergebnis-Ausnahme (`dist/` → `test:package`) und
  keine neue Klasse ungeprüften Quelltexts. Die übrigen zehn schreiben auf, was ohnehin galt —
  bei der M4-Host-Reproduzierbarkeit mit einer benennenswerten Nuance: die EIGENSCHAFT galt
  schon (der CI-Job prüfte seit Item 12 Linux gegen einen macOS-Pin), aber die daraus
  folgende REGEL („Abweichung = Befund, nicht Pin-Anlass") stand nirgends und wird hier zum
  ersten Mal festgeschrieben.
- v5 (2026-07-20) · M1: präzisiert — kernel-lose Referenz-Ops sind zulässig (Paritätslücke
  in FOLLOWUPS getrackt; M1 bindet ab Kernel-Existenz). Anlass: covenant-verify-Empfehlung
  der Op-Scheibe W1 (argmax/topk = erster Präzedenzfall), VOR den Folge-Scheiben W2–W5
  geklärt. M3: präzisiert — „Klassen-Hover" = Typ-Parameter-Anzeige; Member-Hover dürfen
  readonly-Modifier tragen (schließt den seit D-V2/2026-07-13 offenen Auslegungs-Mini,
  Baustein-C-Befund V2). Beide Normen inhaltlich unverändert (Klarstellung gelebter
  Praxis); Owner-bestätigt 2026-07-20.
- v4 (2026-07-17) · Z2: präzisiert — der Verbund `pnpm check` deckt alle QUELLTEXT-Typ-Korpora;
  Typchecks gegen ein Bauergebnis (`dist/`, existiert beim noEmit-Check nicht) laufen bewusst
  im Paket-Testlauf (`test:package`), nicht in `pnpm check`. Anlass: covenant-verify-Befund
  der Item-11/S3-Scheibe (der Konsumenten-Typ-Smoke braucht das gebaute `dist/index.d.ts`);
  Owner-entschieden „Norm präzisieren statt Korpus erzwingen". Norm-Absicht unverändert (kein
  Korpus rottet ungeprüft).
- v3 (2026-07-17) · M2: der unter v2 dokumentierte offene Verstoß (Literal|undefined via
  OPTIONALE Parameter, `sum`s `axis`/`keepdims`) ist GESCHLOSSEN — Item 11 / S1
  (`sum`-Overload-Umbau nach Argument-Anzahl + `reduce.ts`-`KeepDims`-Erweiterung, KD-2),
  dreifach verifiziert; M2-Anker `slice-literal.ts` → `literal-arithmetic.ts` (Datei in T1b
  umbenannt). Norm unverändert (der Verstoß war stets ein Norm-Bruch, jetzt behoben);
  Owner-bestätigt.
- v2 (2026-07-13) · M2: bekannter offener Verstoß dokumentiert (Literal|undefined via
  optionale Parameter, UA_GAP-Sentinel, Item-11-Frist) — Norm unverändert; Anlass:
  covenant-verify-Befund der Union-Axis-Mini-Scheibe, Owner-Entscheidung „dokumentieren
  statt Norm konditionieren".
- v1 (2026-07-13) · Erstfassung (Phase-D-Vorarbeiten, vor V2/Item 11).
