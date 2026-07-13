# Union-Axis-Mini-Scheibe — Ergebnisse

Stand: 2026-07-13, Basis HEAD `cf414d6`. Bindende Spec: `docs/union-axis-mini-spec.md`
(inkl. Baustein-0-Addendum). Covenant: **M2**-Fix (never-wrong-only-incomplete),
Anker `spike/src/dim.ts`, `spike/src/reduce.ts` — **M3** unangetastet (keine
Message-/Hover-Änderungen).

## PRE-FIX-ROT-Beweis (Baustein 0 der Spec, Testplan Punkt 1)

Frischer `git worktree add <scratch> HEAD` (Detached-HEAD `cf414d6`), `node_modules`
per Symlink, `tsc` via `./node_modules/.bin/tsc`. Keine Datei im Haupt-Tree berührt.

**Baseline-Reproduktion im Scratch-Worktree** (vor jeder Änderung):

```
$ ./node_modules/.bin/tsc --noEmit --extendedDiagnostics
Files:             132
Instantiations: 178212
Exit: 0
```

Exakt `178'212 @ 132` reproduziert — Baseline bestätigt.

**Rot-Beweis 1 (Typ-Ebene, direkt):** temporär in `spike/tests/reduce.test-d.ts`
(scratch-worktree-only, danach verworfen via `git checkout --`):

```ts
type ROT1 = Expect<Equal<ReduceAxis<[2, 3], 0 | 2>, readonly number[]>>;
```

```
$ ./node_modules/.bin/tsc --noEmit
spike/tests/reduce.test-d.ts(69,20): error TS2344: Type 'false' does not satisfy the constraint 'true'.
Exit: 1
```

**Rot-Beweis 2 (Call-Site, echter `.sum()`-Aufruf via `NDArray<[2,3]>`)**, temporär
in `spike/tests/ndarray.test-d.ts`:

```ts
declare const rotRecv: NDArray<[2, 3]>;
const rotSummed = rotRecv.sum(0 as 0 | 2);
type ROT_CALLSITE = Expect<Equal<(typeof rotSummed)["shape"], readonly number[]>>;
```

```
$ ./node_modules/.bin/tsc --noEmit
spike/tests/ndarray.test-d.ts(373,28): error TS2344: Type 'false' does not satisfy the constraint 'true'.
Exit: 1
```

**Konfidenz-Bestätigung** (KORRIGIERT in der Schließungsrunde — Baustein-A-Befund 1: die
ursprüngliche „beide Pins"-Formulierung war falsch) — Ersetzen der `Equal`-Zielseite durch
`readonly [3]` lässt NUR die CALL-SITE-Probe grün werden (`Exit: 0`): dort hat `OkShape`
den Fehler-Zweig bereits abgestreift, der Ist-Stand liefert konfident
`NDArray<readonly [3]>`, obwohl der Runtime-Wert der Achse `2` `sumRuntime` werfen lässt —
das deckt sich wörtlich mit der Spec-Behauptung („Unterfacette (1)"). Die DIREKTE
Typ-Ebenen-Probe (`ReduceAxis<[2,3], 0|2>`, ohne `OkShape`) bleibt bei diesem Tausch
dagegen ROT: ihr Pre-Fix-Rohtyp ist `ShapeError<"reduce: axis 2 is out of range for shape
[2,3] (rank 2)"> | [3]` — ein MUTABLES `[3]` mitsamt unabgestreiftem Fehler-Member
(Baustein A hat den Rohtyp per Zuweisungs-Fehler-Probe extrahiert). Der konfident-falsche
Claim entsteht also erst durch die Kette Distribution → Guard-Akzeptanz → OkShape-Strip;
der Rot-Beweis der Typ-Ebenen-Pin gegen `readonly number[]` bleibt davon unberührt gültig. Beide Probe-Änderungen wurden per `git checkout --` verworfen,
danach der Scratch-Worktree per `git worktree remove --force` entfernt; im Haupt-Tree
wurde zu keinem Zeitpunkt etwas geändert.

## Umsetzung

### D-A.1 · `spike/src/dim.ts`

Ein `export`-Keyword vor die private `IsUnion`-Kopie (Zeile 62, vormals Zeile 59)
gesetzt; Doc-Kommentar erweitert um den neuen Konsumenten `reduce.ts`. Keine sonstige
Änderung. Geprüft: `grep -rn "IsUnion" spike/src/*.ts` zeigt reduce.ts importiert von
`dim.ts`, reshape.ts/vector.ts weiterhin von `slice-literal.ts` — keine Datei
importiert beide unter demselben Namen, `tsc` bestätigt dies durch fehlerfreien
Compile (keine Namenskollision).

### D-A.2 · `spike/src/reduce.ts`

Neuer Zweig in `ReduceAxis` direkt nach `[Axis] extends [undefined]` und VOR dem
naked `Axis extends number`:

```ts
: IsUnion<Axis> extends true
  ? readonly Dim[] // (Kommentar mit Positions-Begründung + beiden Unterfacetten, siehe Diff)
  : Axis extends number
    ? ...unverändert...
```

Kommentar trägt (i) die Positions-Begründung (der naked Check dahinter distribuiert
bereits, jeder spätere Filter sähe nur Einzel-Member), (ii) beide Unterfacetten
(konfident-falscher Bug + strukturell unerreichbare `0|undefined`-Facette), (iii)
den bewussten Policy-Unterschied zur Shape-Ebene (kein uniform/gemischt-Split nötig,
da der Filter vor jeder ShapeError-Erzeugung greift — mirrors Union-DIM-Präzedenz).
`IsUnion` importiert von `./dim.ts` (bereits vorhandener Import-Pfad, nur um das neue
Symbol erweitert).

### D-A.3 · Sonst nichts

`Guard`/`OkShape`/`sum`-Signaturen (`ndarray.ts` Zeile 405, `resident.ts` Zeile 798)
unverändert — beide bereits generisch genug (`const Axis extends number | undefined`),
kein Wiring nötig. `runtime.ts`/`keepDimsShape` unangetastet. Rust: null Zeilen
geändert.

## Positions-Gegenprobe (Baustein 0, mechanisch demonstriert)

Vor der finalen Implementierung wurde geprüft (per lokalem tsc-Probe-Edit im
Haupt-Tree, sofort per Edit-Undo zurückgesetzt — kein Zwischenstand committet), dass
der Filter NACH dem naked `Axis extends number`-Check platziert keinen Effekt hätte:
an dieser Stelle ist `Axis` bereits auf einen Einzelmember distribuiert, `IsUnion<Axis>`
wäre dort strukturell immer `false`. Die tragende Position ist VOR dem naked Check —
so wie umgesetzt.

## Pin-Katalog

| Datei | Pin | Assertion | Zweck |
|---|---|---|---|
| reduce.test-d.ts | `UA1` | `ReduceAxis<[2,3], 0\|2>` → `readonly number[]` | Facette (1) Typ-Ebene, direkter Fix-Beweis |
| reduce.test-d.ts | `UA2` | `ReduceAxis<[2,3], 0\|undefined>` → `readonly number[]` | Facette (2) Typ-Ebene: der Filter selbst deckt diese Form, die Lücke ist rein Call-Site-Inferenz |
| reduce.test-d.ts | `UA3` | `ReduceAxis<[2,3], -1\|0>` → `readonly number[]` | negative Union |
| reduce.test-d.ts | `UA4` | `ReduceAxis<[2,3], 2\|5>` → `readonly number[]` | all-invalid Union, akzeptiert + degradiert |
| reduce.test-d.ts | `UA5`/`UA6`/`UA7` | Union-Axis × KeepDims (`true`/`false`/`boolean`) → `readonly Dim[]` | KeepDims un-degradiert die Union nie |
| ndarray.test-d.ts | `UA_CALL1` | `NDArray<[2,3]>.sum(0 as 0\|2)` → `readonly number[]` | Facette (1) Call-Site, JS-Surface |
| ndarray.test-d.ts | `UA_CALL2` | `WNDArray<[2,3]>.sum(0 as 0\|2)` → `readonly number[]` | Facette (1) Call-Site, WASM-Resident-Surface |
| ndarray.test-d.ts | `UA_WORKAROUND` | `a.sum<0\|undefined>(u)` (explizites Typ-Argument) → `readonly number[]` | Workaround-Pin für Facette (2) |
| ndarray.test-d.ts | `UA_GAP` | `a.sum(u)` mit `u: 0\|undefined`, KEIN Typ-Argument → **konfident `readonly [3]`** | dokumentierender Lücken-Pin (Facette (2), bewusst NICHT gefixt) |
| ndarray.test-d.ts | `UA_NEG` | `a.sum(-1 as -1\|0)` → `readonly number[]` | negative Union, Call-Site |
| ndarray.test-d.ts | `UA_ALL_INVALID` | `a.sum(2 as 2\|5)` → `readonly number[]` | all-invalid, Call-Site |
| ndarray.test-d.ts | `UA_KEEP_TRUE`/`_FALSE`/`_DYN` | Union-Axis × KeepDims, Call-Site | s.o., Call-Site-Form |
| ndarray.test-d.ts | `UA_SAFE_KEEPDIMS` | `a.sum(1, b as boolean)` auf `[2,3]` → `readonly [2] \| readonly [2,1]` | already-safe Kontroll-Pin (Policy: nicht angefasst) |
| ndarray.test-d.ts | `UA_CONTROL_LITERAL` | `a.sum(0)` → `readonly [3]` | Kontroll-Pin: literale Einzel-Achse bleibt präzise |
| ndarray.test-d.ts | `UA_CONTROL_DYNAMIC` | `a.sum(dynamicAxis: number)` → `readonly number[]` | Kontroll-Pin: dynamische Achse unverändert |
| ndarray.test-d.ts | `@ts-expect-error` | `a.sum(2)` auf `[2,3]` | Kontroll-Pin: out-of-range-Einzel-Achse-Message unverändert (M3) |

Alle Pins folgen den bestehenden Idiomen der jeweiligen Datei (`Expect<Equal<...>>`,
`declare const` + `void`-Discard für Call-Site-Konstanten, `@ts-expect-error` mit
Begründungskommentar).

## Gates

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Verbund) | grün | 0 |
| `pnpm check:diag` | `178'865 @ 132` (Δ+653 ggü. Baseline `178'212 @ 132`) | 0 |
| `pnpm check:diag:stress` | `102'182 @ 82` (Δ+86 ggü. Baseline `102'096 @ 82`) | 0 |
| `pnpm check:diag:browser` | `2'142 @ 75` (Δ0 ggü. Baseline `2'142 @ 75`) | 0 |
| `pnpm test:core` | 818/818 pass | (node:test summary, 0 fail) |
| `pnpm test:resident` | 4278 pass / 2 skipped von 4280 deklariert | (0 fail) |
| `pnpm test:threaded` | 69/69 pass | (0 fail) |
| `pnpm test:browser` | 4/4 pass (Chromium) | (0 fail) |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161 passed + 1 zero-alloc-Test passed | 0 fail |
| `pnpm demo` | alle drei Backends stimmen bit-für-bit überein | „demo complete" |
| `graph-a-lama . --symbols` | 146 files, 1121 symbols, 2669 references | Exit 0 |
| `graph-a-lama query lint` | „keine Verstöße", 0 errors, 0 warnings | Exit 0 |
| Artefakt-Hash (`shasum -a 256 spike/src/wasm/numtype_core.wasm`) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` | exakt gepinnt, ZERO Rust bestätigt |
| `pnpm bench:editor` — warm hover (M2, Gate ≤100ms) | Mediane 0.04–0.07ms, alle PASS | Overall hard gate: PASS |
| `pnpm bench:editor` — warm toggle (M3, Gate ≤500ms) | Mediane 1.3–3.8ms, alle PASS | s.o. |

Budget-Hartregel der Spec (Haupt-`check:diag` ≤ 225'000): `178'865` — deutlich
unterschritten. Editor-Gates der Spec (Hover ≤1ms / Toggle ≤10ms) — alle gemessenen
Mediane liegen darunter (Hover max. 0.07ms, Toggle max. 3.98ms bei W6).

## Pin-Deltas — Einordnung

Datei-Zahl konstant in ALLEN drei Korpora (132/82/75 — identisch zur Baseline): nach
der CLAUDE.md-Order-Noise-Regel (die Mechanik gilt für Datei-ADDITIONEN, nicht
-Edits) sind die gemessenen Deltas (`+653`/`+86`/`+0`) reale Typkosten, keine
Order-Artefakte — konsistent mit der Größe des Diffs (ein neuer Filter-Zweig +
Doc-Kommentar in reduce.ts, ein `export`-Keyword in dim.ts, eine Handvoll neuer
Union-Pins in zwei bestehenden Test-Dateien). Kein bestehender Pin musste
re-expressiert werden (`git diff --stat` zeigt ausschließlich Insertionen in den vier
berührten Dateien, keine geänderten Zeilen außerhalb der neuen Blöcke).

## Diff-Übersicht

```
 spike/src/dim.ts              | 11 +++--
 spike/src/reduce.ts           | 57 +++++++++++++++++++------
 spike/tests/ndarray.test-d.ts | 98 +++++++++++++++++++++++++++++++++++++++++++
 spike/tests/reduce.test-d.ts  | 35 +++++++++++++++-
 4 files changed, 184 insertions(+), 17 deletions(-)
```

Vier Dateien, keine neue Datei — deckt sich mit der Spec-Erwartung „Dateizahl
konstant = echte Typkosten". `git status` VOR der Session zeigte nur die
untracked Spec-Datei; NULL Edits unter `spike/tests-runtime/`, NULL an
`runtime.ts` (per `git diff --stat` bestätigt — beide Pfade fehlen im Diff).

## Covenant-Abschnitt

- **M2** (never-wrong-only-incomplete): direkt adressiert — die Scheibe schließt
  Facette (1), den letzten bekannten confidently-wrong-Fall der Typ-Ebene vor Item
  11. Der neue `IsUnion<Axis>`-Filter degradiert JEDE Achsen-Union (literal, negativ,
  `undefined`, all-invalid) zu `readonly Dim[]` — no-claim statt konfidentem
  Einzeltyp. Der verbleibende Lücken-Pin (`UA_GAP`) macht die BEKANNTE, bewusst
  offengelegte Facette-(2)-Lücke beobachtbar, verletzt M2 nicht selbst (er behauptet
  nichts Falsches über den Typ-Fix — er dokumentiert eine strukturell unerreichbare
  Inferenz-Eigenheit von TS bei optionalen Parametern, die außerhalb des Scopes
  dieser Scheibe liegt, per Owner-Entscheidung in der Spec).
- **M3** (Fehler am Argument, Message wortgleich, saubere Hover): unangetastet
  bestätigt — der Kontroll-Pin `a.sum(2)` (out-of-range Einzel-Achse) bleibt ein
  `@ts-expect-error` ohne Message-Änderung; kein Guard/OkShape/Message-Code berührt.
- `graph-a-lama query lint` (S1, mechanisch) — 0 Verstöße.
- Volle Eskalationsstufe 3 (bindende Spec, Verdikt-tragende Maschinerie) — Baustein 0
  bereits vor der Implementierung gelaufen (siehe Spec-Addendum); Baustein A (dieses
  Dokument) abgeschlossen. Baustein B (adversarialer Verifier) und Baustein C
  (`covenant-verify`) sind laut Prozess PARALLEL dazu vorgesehen — in dieser Session
  NICHT gespawnt (Arbeitsregel „keine Subagenten" der Delegation), müssen vor
  endgültigem „fertig" separat angestoßen werden.

## Honesty-Residuum

- Facette (2) (`Axis = 0 | undefined` über den optionalen `axis?`-Parameter) bleibt
  bewusst ungefixt — der Lücken-Pin `UA_GAP` macht das beobachtbar. Sollte ein
  künftiges TS-Release oder ein Signatur-Umbau (Item-11-Overload-Split) dieses
  Inferenz-Verhalten ändern, schlägt `UA_GAP` fehl und meldet die Verschiebung.
- Diese Session hat NICHT die volle Baustein-A+B+C-Parallelität durchlaufen (keine
  Subagenten erlaubt) — nur Baustein A (Implementierung + Gates, dieses Dokument).
  Baustein B (adversarialer Fresh-Context-Verifier) und Baustein C
  (`covenant-verify`) stehen laut Eskalationsleiter (Stufe 3) noch aus und sollten
  vor dem finalen Commit/„fertig"-Signal separat beauftragt werden.
- Kein Rust-Build in dieser Session ausgeführt außer implizit durch `pnpm test:browser`
  (welches `cargo build --release` für den WASM-Build-Schritt aufruft) — der Hash
  wurde NACH allen Gates direkt per `shasum` verifiziert und exakt gepinnt; das
  bestätigt „ZERO Rust-Änderungen" empirisch, nicht nur durch Diff-Abwesenheit.
- Die Stress-Corpus-Delta (`+86`) wurde nicht bisektiert (kein Order-Noise-Risiko
  wegen konstanter Dateizahl, aber die genaue Quelle — vermutlich dim.ts's
  IsUnion-Export-Doc-Kommentar-Längenänderung plus der geänderte reduce.ts-Import —
  wurde nicht Zeile-für-Zeile zugeordnet, da unterhalb jeder Aufmerksamkeitsschwelle
  und ohne Verhaltensrelevanz).

## FOLLOWUPS-Textvorschläge (NICHT in FOLLOWUPS.md eingetragen — nur hier vorgeschlagen)

**(i) Austragung als erledigt** (ersetzt die offene Zeile „Union im
AXIS-Parameter…" im bestehenden FOLLOWUPS-Item):

> - [x] Union im AXIS-Parameter — erledigt & Baustein-0-vor-verifiziert als
>   Union-Axis-Mini-Scheibe, 2026-07-13 (docs/union-axis-mini-spec.md +
>   -ergebnisse.md): `ReduceAxis` erhält einen `IsUnion<Axis>`-Filter direkt vor dem
>   naked `Axis extends number`-Zweig (dim.ts's private `IsUnion` dafür exportiert);
>   jede Achsen-Union (literal, negativ, `undefined`, all-invalid) degradiert zu
>   `readonly Dim[]` statt konfident falsch zu claimen — schließt den letzten
>   bekannten confidently-wrong-Fall der Typ-Ebene vor Item 11 (Baustein-B-Befund F2).
>   ZERO Rust/Hash byte-identisch, Dateizahl konstant, keine Pin-Re-Expressionen
>   nötig. Die Optional-Parameter-Facette (`0|undefined` via `axis?`) bleibt bewusst
>   offen — siehe neues Item (ii).

**(ii) neues, release-relevantes Item:**

> - [ ] **Literal|undefined durch optionale Parameter (release-relevant, vor
>   Item-11-API-Schnitt zu entscheiden):** `a.sum(u)` mit `u: 0 | undefined` (kein
>   explizites Typ-Argument) bleibt konfident `NDArray<[3]>`, obwohl `u` zur
>   Laufzeit `undefined` sein kann (Root Cause: TS streift `undefined` aus dem
>   inferierten Typ-Argument, sobald der Parameter selbst optional ist —
>   2×2-Kreuzprobe, Baustein-0-verifiziert 2026-07-13, docs/union-axis-mini-spec.md).
>   Dasselbe Muster trifft `keepdims?` (`kd as true|undefined` → konfident falsches
>   Shape). Workaround bis dahin: explizites Typ-Argument
>   (`a.sum<0|undefined>(u)`), degradiert korrekt (Pin `UA_WORKAROUND`,
>   docs/union-axis-mini-ergebnisse.md). Entscheidungskandidat für Item 11: ein
>   Overload-Split der `sum`-Signatur (und ggf. anderer optionaler-Parameter-Ops).
>   Beobachtbar gehalten durch den dokumentierenden Lücken-Pin `UA_GAP`
>   (ndarray.test-d.ts) — kippt TSs Inferenz-Verhalten künftig, schlägt der Pin fehl
>   und meldet es.

## Nächste Schritte (außerhalb dieser Session)

1. Baustein B (adversarialer Fresh-Context-Verifier) und Baustein C
   (`covenant-verify`) parallel beauftragen, Auftrag aus
   `docs/verify-runde-template.md`.
2. Bei grünem Verdikt: FOLLOWUPS.md mit den obigen zwei Textvorschlägen aktualisieren,
   CLAUDE.md-Pin-Update (`check:diag` 178'865 @ 132 / `check:diag:stress` 102'182 @ 82
   / `check:diag:browser` 2'142 @ 75, unverändert), KB-Kandidat capturen („naked
   Parameter-Checks sind Distributions-Beginn — Union-Filter gehören VOR den ersten
   naked Check"), dann Commit.

## Post-Verification-Addendum (Drei-Verifier-Runde + Schließungen, 2026-07-13)

Voller Stufe-3-Katalog nach docs/verify-runde-template.md.

**Baustein A (Spec-Verifier): CONFIRMED mit Auflage.** Alle Gates exakt reproduziert
(Baseline 178'212 @ 132 und End-Stand 178'865 @ 132 im frischen Worktree; Hash per EIGENEM
`cargo clean`-Rebuild — stärkere Form als der Executor-shasum); Pin-Katalog vollständig
gegen die Spec (18 Formen, beide Surfaces); PFLICHT-Mutant = Positions-Mutant (Filter
hinter den naked Check verschoben): 16 Pins rot, `UA_GAP` korrekt grün — die
Positions-Entscheidung ist load-bearing gepinnt; Bonus-Removal-Mutant: identischer roter
Satz. Auflage = die „Konfidenz-Bestätigung" oben (korrigiert, s. dort).

**Baustein B (adversarial): HÄLT mit einem Befund.** Alle Typ-Kanten-Angriffe grün
(3+-Member, `0|2|undefined` — nach TS-Stripping bleibt eine echte Union, die GAP ist
strikt auf „genau EIN Literal + undefined" begrenzt; rein negative/out-of-range-negative/
non-integer-Unionen; dynamischer Rang; Mixed-Rank-Union-S; Verkettungen;
`ReduceAxisKeepDims`-Alias; all-VALID-Union degradiert policy-konform trotzdem).
Vakuitäts-Angriffe: `UA_GAP` ist ein ECHTER Sentinel (required-Parameter-Mutant kippt ihn
auf rot); `Equal`-Pins fangen mutable-/Element-Typ-Drift; IsUnion-konstant-false-Mutant →
35 rote Pins über 6 Dateien (UA-Katalog + V1-Bestand), die vier strukturell
IsUnion-unabhängigen Pins korrekt grün. Runtime-Backstop in echtem Node bewiesen
(Union-Achsen-WERT 2 wirft mit exakt dem gepinnten Message-Stamm). Befund: die
bench:editor-Workloads W1–W6 enthielten keinen Union-Achsen-Hover — **in der
Schließungsrunde dauerhaft geschlossen: neues W7** (`w7-union-axis.ts`, ein
`sum(0 as 0|2)`-Hover, korrektheits-gegated auf `NDArray<readonly number[]>`; gemessen
0,05–0,09 ms, Gesamt-Gate PASS, per-Workload isoliert 22'795 Instantiations — kleinster
Workload im Katalog).

**Baustein C (covenant-verify): ein Drift-Befund (mittel), sonst keine
Invarianten-Verletzung.** M2-Kernfix, Guard/OkShape-Unberührtheit, M3-Message-Stamm
(zeichengleich, nur re-eingerückt), M5/M4/M1/Z1/Nicht-Ziele, S1-Lint (exit 0) und die
Regel↔Spec-Verknüpfung: alles sauber. Der Befund: COVENANT.md M2 ist unbedingt formuliert
(„nie ein konfident-falscher Claim"), der Diff pinnt aber mit `UA_GAP` bewusst einen
bekannten confidently-wrong-IST-Zustand (Owner-Scope-Entscheidung). **Owner-Entscheidung
(2026-07-13): als BEKANNTEN VERSTOSS dokumentieren** — M2s Norm bleibt unverändert stark,
COVENANT.md v2 trägt unter M2 eine datierte Verstoß-Notiz mit FOLLOWUPS-Verweis und
Item-11-Frist (Version-Bump + Changelog; die im Covenant-Workflow explizit vorgesehene
dritte Option). Künftige covenant-verify-Läufe sehen die Notiz.

**Schließungen dieser Runde** (alle doku-/bench-only, null src-Änderungen, Hash
unberührt): (1) Konfidenz-Bestätigung korrigiert (A-Auflage); (2) W7 dauerhaft in
gen-workloads.ts (B-Befund; `pnpm bench:editor` frisch: Overall hard gate PASS); (3)
COVENANT.md v2 (C-Befund, Owner-entschieden); (4) dieses Addendum.
