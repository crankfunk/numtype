# Union-Axis-Mini-Scheibe — bindende Spec

Stand: 2026-07-13, HEAD cf414d6 (nach Phase-D-Vorarbeiten V1–V3). Herkunft: Baustein-B-Befund
F2 der V1-Verify-Runde (FOLLOWUPS-Item „Union im AXIS-Parameter", release-relevant markiert) —
der LETZTE bekannte confidently-wrong-Fall der Typ-Ebene vor Item 11. Covenant: dies ist ein
**M2**-Fix; berührte Anker `spike/src/reduce.ts` (via dim.ts-Import), `spike/src/dim.ts`.
Eskalationsleiter: Stufe 3 (bindende Spec, Verdikt-tragende Maschinerie) → Baustein 0 + voller
Katalog A+B+C, trotz kleinem Diff.

## Problem, am Code verankert (Stand nach V1)

`ReduceAxis<S, Axis, KeepDims>` (reduce.ts:80–96): der erste Zweig `[Axis] extends
[undefined]` ist tuple-wrapped, aber der zweite — `Axis extends number` — ist NAKED und
distribuiert über eine Achsen-Union, BEVOR irgendein späterer Filter die Union als Ganzes
sehen könnte. Zwei Unterfacetten:

1. **`sum(0 as 0|2)` auf `NDArray<[2,3]>`** (Baustein-B-F2, empirisch via echtem Call + LSP):
   Distribution liefert `ResolveAndApply<S,0> | ShapeError<axis 2 out of range>`; die
   V1-tuple-wrapped `Guard` akzeptiert die gemischte Union korrekt (Policy Zeile 2), aber
   `OkShape` streift den Fehler-Zweig → **konfident `NDArray<[3]>`**, während der Runtime-Wert
   `2` `sumRuntime` werfen lässt.
2. **`Axis = 0 | undefined`** — Baustein-0-VERIFIZIERT (2026-07-13) und **per
   Owner-Entscheidung AUS DEM SCOPE genommen**: das Phänomen ist real, aber über den
   OPTIONALEN `axis?`-Parameter strukturell unerreichbar für jeden ReduceAxis-Filter — TS
   streift `undefined` bei der Inferenz aus der Argument-Union (2×2-Kreuzprobe: einzig die
   Parameter-Optionalität entscheidet; `Axis` wird als `0` inferiert, die Optionalität selbst
   akzeptiert das `undefined`-Argument). Der einzige idiomatische Fix wäre ein Overload-Split
   der `sum`-Signaturen; derselbe Root-Cause trifft auch `keepdims?` (`kd as true|undefined`
   → konfident falsches Shape — Baustein-0-Neufund). Die GESAMTE
   „Literal|undefined-durch-optionale-Parameter"-Familie geht als EIN release-relevantes
   FOLLOWUPS-Item zur Item-11-API-Schnitt-Entscheidung (Workaround bis dahin: explizites
   Typ-Argument — `a.sum<0|undefined>(u)` degradiert nach dem Fix korrekt, wird als
   Workaround-Pin festgehalten). Die Kopf-Behauptung dieser Spec ist entsprechend zu lesen:
   die Scheibe schließt Facette (1); die Optional-Parameter-Familie bleibt eine BEKANNTE,
   OFFENGELEGTE Lücke.

## Bindende Policy

Eine Union im AXIS-Parameter — JEDE Union: literale Member (`0|2`), mit `undefined`
(`0|undefined`), negative Member (`-1|0`), auch ALL-invalid (`2|5` auf Rang 2) — verhält sich
EXAKT wie die dynamische Achse (`number`): Degradation zu `readonly Dim[]`, kein Verdikt,
Runtime-Backstop (`sumRuntime` wirft bei konkret invalider Achse). Das folgt der
V1-Owner-Policy „Union → no-claim" einschließlich ihres Präzedenzfalls: auch all-invalid wird
NICHT statisch abgelehnt (wie Union-DIMS in `CompatDim` — dokumentierte Unvollständigkeit,
kein Fehler). **KeepDims-Union (`boolean`)** bleibt UNGEFILTERT: die natürliche Distribution
hat keinen Fehler-Zweig und liefert die korrekte Ergebnis-Union (z. B.
`sum(1, b as boolean)` → beide Varianten) — wird als already-safe GEPINNT, nicht umgebaut.

## Bindende Entscheidungen

- **D-A.1** · `dim.ts`: die private `IsUnion`-Kopie (dim.ts:59, V1) wird EXPORTIERT (ein
  `export`-Keyword; Doc-Kommentar erwähnt den neuen Konsumenten; der historische
  slice-literal.ts-Export und dessen Konsumenten bleiben unberührt).
- **D-A.2** · `ReduceAxis` erhält einen neuen Filter-Zweig DIREKT nach `[Axis] extends
  [undefined]` und VOR dem naked `Axis extends number`:
  `IsUnion<Axis> extends true ? readonly Dim[] : …` — die Position ist tragend (nach dem
  naked Check sähe jeder Filter nur noch Einzel-Member); der Code-Kommentar trägt diese
  Begründung + beide Unterfacetten. Deckt (1) und (2) in einem Zweig.
- **D-A.3** · KEINE Änderungen an `Guard`/`OkShape`/den `sum`-Signaturen/runtime.ts/Rust.
  `keepDimsShape` bleibt unberührt (Runtime sieht nur konkrete Achsen).

## Testplan (repro-first, bindend)

1. **PRE-FIX-ROT** im Scratch-Worktree von HEAD mit dem finalen Pin-Set: Unterfacette (1)
   muss gegen den IST-Stand fehlschlagen (konfidenter Einzeltyp statt Degradation); volle
   tsc-Ausgabe ins Ergebnisdoc.
2. **Pflicht-Pins** (in BESTEHENDE Dateien — reduce.test-d.ts, ndarray.test-d.ts; keine
   neuen Root-Korpus-Dateien): `sum(0 as 0|2)` → `NDArray<readonly number[]>` auf NDArray
   UND WNDArray; direkter Typ-Ebenen-Pin `ReduceAxis<[2,3], 0|undefined>` → degradiert
   (die Typ-Ebenen-Form deckt der Filter — Baustein-0-bewiesen) + Workaround-Pin
   `a.sum<0|undefined>(u)` → degradiert; **dokumentierender Lücken-Pin**: die realistische
   Aufrufform `a.sum(u)` mit `u: 0|undefined` bleibt konfident `NDArray<[3]>` — als
   IST-Zustand gepinnt mit Kommentar auf das FOLLOWUPS-Item (der Pin macht die Lücke
   beobachtbar: kippt TSs Inferenz-Verhalten, meldet er sich); negative Union (`-1|0`);
   all-invalid (`2|5` auf `[2,3]`) → akzeptiert + degradiert (dokumentierte
   Unvollständigkeit); Union-Axis × keepdims (true, false, und `boolean`);
   KeepDims-`boolean`-already-safe-Pin bei LITERALER Achse (exakt `readonly [2] | readonly
   [2,1]` für `sum(1, b as boolean)` auf `[2,3]` — Baustein-0-gemessen); KONTROLL-Pins:
   literale Einzel-Achse bleibt präzise (`sum(0)` → `NDArray<[3]>`), dynamische Achse
   (`number`) unverändert, out-of-range-EINZEL-Achse behält den wortgleichen ShapeError am
   Argument (M3 — bestehende Pins dürfen NICHT re-expressiert werden müssen; falls doch,
   einzeln listen und begründen).
3. Bestehende Suiten by construction unberührt: NULL Edits unter spike/tests-runtime/, NULL
   an runtime.ts.

## Gates (pre-registriert)

`pnpm check` (Verbund) grün; test:core 818 / test:resident 4280 / test:browser 4/4 /
test:threaded 69 / cargo 161 / demo unverändert grün; `graph-a-lama . --symbols && graph-a-lama
query lint` exit 0 (S1); Artefakt-Hash exakt
`0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` (ZERO Rust); Budget hart:
Haupt-`check:diag` ≤ 225'000, bench:editor warm-hover ≤ 1 ms / Toggle ≤ 10 ms. Pins: Baseline
im frischen cf414d6-Worktree muss 178'212 @ 132 / 102'096 @ 82 / 2'142 @ 75 exakt
reproduzieren; End-Stand messen, Deltas berichten (Erwartung: klein — ein Filter-Zweig + eine
Handvoll Union-Pins; Dateizahl konstant = echte Typkosten).

## Verifikation & Artefakte

**Baustein 0 GELAUFEN (2026-07-13, Addendum):** Facette (1) CONFIRMED inkl. Fix- und
Positions-Gegenprobe (Filter NACH dem naked Check ist wirkungslos — Distribution bereits
passiert, mechanisch demonstriert); IsUnion-Export Δ exakt 0, keine Namenskollision; voller
Korpus bleibt clean (keine Pin-Re-Expressionen nötig); KeepDims-`boolean` already-safe
bestätigt (`readonly [2] | readonly [2,1]`); `never`-Axis unverändert (matcht den
vorbestehenden undefined-Tuple-Check); kein weiterer Axis-Konsument außer sum; keine
V2-Interaktion. BLOCKER gefunden und per Owner-Entscheidung (Scope-Reduktion) aufgelöst:
Facette (2) ist über den optionalen Parameter strukturell unerreichbar (TS-Inferenz streift
`undefined`; 2×2-Kreuzprobe), Neufund keepdims-Analog — beides als
Optional-Parameter-Familie → FOLLOWUPS (Item-11-Entscheidung). Dokumentierter Nit: zwei
Union-Politiken koexistieren bewusst (Shape-Ebene: uniforme Fehler-Union lehnt ab via Guard;
Axis-Ebene: no-claim wie Union-Dims — der Filter greift VOR jeder ShapeError-Erzeugung).
Danach Executor, dann Baustein A + B + C parallel (Stufe 3). Ergebnisdoc `docs/union-axis-mini-ergebnisse.md` mit
Post-Verification-Addendum (drei Verdikte); FOLLOWUPS-Austragung des Union-Axis-Items;
CLAUDE.md-Pin-Update; KB-Kandidat: „naked Parameter-Checks sind Distributions-Beginn —
Union-Filter gehören VOR den ersten naked Check" (Ergänzung der Guard-/Union-Notizen); Commit.
