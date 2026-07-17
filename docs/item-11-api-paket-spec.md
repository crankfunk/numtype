# Item 11 — API-Konsolidierung + Paketschnitt — bindende Spec

Stand: 2026-07-17, vor Baustein-0-Verifikation. Owner-Richtung (Chat, 2026-07-17,
drei Gabeln entschieden — Abschnitt „Owner-Gabeln"): aus dem Forschungs-Verzeichnis
`spike/` wird ein publizierbares npm-Paket mit **einem** öffentlichen Surface. Diese
Spec bindet die Scheibe; sie wird in **drei sequenziellen Sub-Scheiben** umgesetzt
(S1 → S2 → S3), jede durchläuft einzeln die volle Hausdisziplin (Implementierung →
Zwei-Verifier-Runde nach docs/verify-runde-template.md + Baustein C covenant-verify →
Ergebnisdoc → KB-Capture → Commit).

Roadmap-Kontext: docs/roadmap.md Item 11 (Phase D). Diese Scheibe zieht sechs
FOLLOWUPS-Punkte ein (Overload-Split der `sum`-Optional-Parameter, `slice-literal.ts`-
Umbenennung, `exports`-Map/`.wasm`-Bundling, d.ts-Hover-Qualität, npm-Name-Sicherung,
Zero-dep-Guard-Test).

## Berührte Covenant-Invarianten (COVENANT.md v2)

Diese Scheibe ist **Stufe 3** der Covenant-Eskalationsleiter (substanzielle Scheibe,
bindende Spec) → voller Katalog A+B+C je Sub-Scheibe + Lint im Gate-Block.

- **M2** (never wrong) — S1 schließt den dokumentierten offenen `Literal|undefined`-
  Verstoß per `sum`-Overload-Umbau, **BEIDE Facetten** (axis + keepdims, KD-2). Anker
  `sym:Guard`, `spike/src/reduce.ts` (KeepDims-Constraint-Erweiterung), `spike/src/ndarray.ts`,
  `spike/src/wasm/resident.ts`. **Positiv berührt** (schließt den v2-M2-Verstoß vollständig;
  erfordert Owner-bestätigte COVENANT-Aktualisierung v2 → v3 mit Version-Bump, s. u.).
- **M3** (Fehler am Argument, saubere Hovers) — die emittierten `.d.ts` sind Teil des
  Produkts; Hover-Qualität (`NDArray<[2, 3]>`, `sum`-Signatur) muss durch den Emit
  erhalten bleiben. Anker `sym:Guard`, `sym:ShowShape`.
- **M4** (Frozen Baseline, Artefakt-Hash) — S2 **kopiert** `numtype_core.wasm` ins
  Paket, ändert es nie; Freeze-Hash `0b9df4f1…519c7d` muss byte-identisch bleiben.
  Keine Sub-Scheibe fasst `crates/` an.
- **M5** (Default-Pfad browser-sicher, Threads Node-only-Opt-in) — zentral: die
  `exports`-Map + der Barrel-Emit dürfen den Default-`NDArray`-Pfad nicht mit eager
  `node:*`-Imports kontaminieren. Anker `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`.
- **S1** (Runtime importiert nie aus Test/Bench/Demo) — mechanisch via
  `graph-a-lama query lint` im Gate-Block jeder Sub-Scheibe (die Umbenennung + das
  Paket-Layout dürfen keine solche Kante einführen).
- **Z1** (Zero-Dependency-Runtime) — S3 macht den Claim zum geprüften Gate; die
  Paket-Metadaten dürfen nie ein `dependencies`-Feld bekommen. Anker `package.json`.
- **Z2** (`pnpm check` = Verbund ALLER Typ-Korpora) — S2 fügt eine Emit-Strecke hinzu;
  `pnpm check` muss weiterhin nichts ungeprüft rotten lassen. Anker `package.json`.

Nicht-Ziele-Abgleich (COVENANT.md): kein NumPy-Vollklon, kein Per-Call-Routing, kein
Browser-Threads-Port, keine transzendenten Ops — diese Scheibe fügt **keine** Ops und
**keine** neuen Backends hinzu, sie konsolidiert das bestehende Surface. Konform.

## Gemeinsame Rahmenbedingungen (alle drei Sub-Scheiben)

- **Zero Rust.** Keine Sub-Scheibe fasst `crates/` an. Freeze-Beweis in der starken
  Form: Artefakt-Hash nach `pnpm build:wasm` byte-identisch zum Pin
  `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`. Die im Paket
  landende `numtype_core.wasm` ist eine byte-identische Kopie (per `shasum` gegen den
  Pin und gegen `spike/src/wasm/numtype_core.wasm` geprüft).
- **Pins** (Ausgangsstand vor S1): `check:diag` Haupt **178'865 @ 132**,
  `check:diag:stress` **102'182 @ 82**, `check:diag:browser` **2'142 @ 75**. Je
  Sub-Scheibe im frischen `git worktree` des jeweiligen Ausgangscommits neu gemessen
  (Mess-Hausregel), Exit-Code + Fehlerausgabe geprüft, nie nur die Kennzahl gegrept.
  **Order-Noise-Regel (hart):** S1 **benennt eine Root-Korpus-Datei um**
  (`slice-literal.ts`) = Datei-Entfernung + Datei-Hinzufügung → der `check:diag`-Pin
  nach S1 ist ein **NEUER Baseline**, NICHT gegen 178'865 vergleichbar (±~2'000
  Sortier-Order-Noise, s. CLAUDE.md „Mechanismus GEPINNT"). Deshalb wird S1 intern in
  zwei Schritte dekomponiert (erst Overload-Split als reiner EDIT sauber messen, dann
  Umbenennung als File-Set-Änderung mit neuem Baseline). S2/S3 fügen dem ROOT-Korpus
  nach Möglichkeit keine Dateien hinzu (dist-Emit lebt außerhalb aller tsconfig-
  Korpora; neue Guard-/Smoke-Tests sind Runtime-Tests, keine `.test-d.ts` im Root).
- **Reihenfolge S1 → S2 → S3, strikt sequenziell.** S2 baut auf den umbenannten
  Dateien und den Overload-Split-Signaturen aus S1 auf; S3 testet das in S2 gebaute
  Paket. Kein Parallelisieren (alle drei berühren dieselbe Emit-/Surface-Fläche).
- **Alle Kommandos vom Repo-Root** (cargo-Config-Discovery ist CWD-basiert).
- **Gate-Block je Sub-Scheibe** (alle grün vor „fertig"): `pnpm check` (Dreier-,
  nach S2 ggf. Vierer-Verbund), `pnpm test:core`, `pnpm test:resident`, `cargo test`,
  `pnpm demo`, `graph-a-lama query lint` (Exit 1 = roter Test), Artefakt-Hash gegen
  Pin. `pnpm test:threaded` nur, wenn die Sub-Scheibe den Threads-Pfad berührt.

---

## Sub-Scheibe S1 — Typ-Vorarbeiten (Overload-Split + Umbenennung)

Zwei unabhängige, rein TS-seitige Refactorings. Kein `crates/`, kein `.wasm`, keine
Paket-Metadaten. Berührt M2 (positiv), M3.

### T1a — `sum`-Overload-Umbau (M2-Verstoß-Schließung, beide Facetten)

> **STATUS 2026-07-17: festgezurrt (KD-2, Owner-entschieden).** Der ursprüngliche
> 3-Overload-Vorschlag kompilierte nicht (TS2394 unter `__variance`) und seine naive
> Reparatur verletzte M3. Ein fokussierter Machbarkeits-Spike hat eine tragfähige Struktur
> gefunden und am echten Code verifiziert (Addendum unten); die keepdims-Facette wird per
> KD-2 (Owner-Wahl) mit-geschlossen. Die finale Struktur ist unten eingearbeitet.

**Problem, am Code verankert.** `sum` trägt heute optionale Parameter:
`spike/src/ndarray.ts:405-408` und `spike/src/wasm/resident.ts:798-801`:

```ts
sum<const Axis extends number | undefined = undefined, const KeepDims extends boolean = false>(
  axis?: Guard<ReduceAxis<S, Axis>, Axis>,
  keepdims?: KeepDims,
): NDArray<OkShape<ReduceAxis<S, Axis, KeepDims>>>
```

`ReduceAxis` (spike/src/reduce.ts:80) filtert Union-Achsen bereits korrekt
(`IsUnion<Axis>` → `readonly Dim[]`, Union-Axis-Mini-Scheibe). Der verbleibende
Verstoß ist strukturell **im optionalen Parameter**: TS streift `undefined` aus einem
`Literal | undefined`-Argument an einem OPTIONALEN Parameter, bevor `ReduceAxis` es
sieht — `a.sum(u)` mit `u: 0 | undefined` bleibt konfident `NDArray<[3]>`, obwohl der
Laufzeitwert `undefined` alles reduziert (docs/union-axis-mini-ergebnisse.md; COVENANT
v2 M2-Notiz; FOLLOWUPS „Literal|undefined durch optionale Parameter"). Gleicher
Root-Cause trifft `keepdims` (`kd: true | undefined`). Beobachtbar über den
`UA_GAP`-Sentinel-Pin (`spike/tests/ndarray.test-d.ts`).

**Fix-Richtung (Machbarkeits-Spike-verifiziert, Owner-entschieden KD-2, 2026-07-17).**
Zwei Facetten, beide geschlossen. Der Machbarkeits-Spike (Addendum unten) hat die finale
Struktur am echten `ndarray.ts` + `resident.ts` verifiziert (alle Constraints grün, nur
der `UA_GAP`-Sentinel kippt erwartungsgemäß rot).

**Kern-Mechanik.** Der optionale Parameter ist die Ursache: TS streift `undefined` aus
einem `Literal | undefined`-Argument an einem OPTIONALEN Parameter. Fix = die Fälle über
Overloads nach *Argument-Anzahl* schneiden, sodass in der Zwei-Argument-Form KEIN
Parameter mehr optional ist. Zwei Nebenbefunde des Spikes, load-bearing:
- **Verschmelzen, nicht spalten:** `undefined` und `number` gehören in EINE Overload
  (`Axis extends number | undefined`, required), nicht in zwei getrennte. Eine separate
  `axis: undefined`-Overload (der ursprüngliche Spec-Vorschlag) bricht mit **TS2394**,
  sobald der `__variance`-Marker (ndarray.ts) da ist — ihr konkreter Rückgabetyp lässt
  sich unter Invarianz nicht gegen die generische Impl-Signatur beweisen.
- **`any` an der Impl-Signatur:** die (extern unsichtbare) Implementierungssignatur bekommt
  Rückgabetyp `NDArray<any>` — der einzige Weg, TS2394 unter `__variance` auch für die
  Null-Argument-Overload aufzulösen. **Parameter-Maschinerie inkl. `Guard` bleibt
  unverändert** → die OOB-Fehlermeldung (M3) ist unberührt, weil sie an der Overload-2/3-
  Parameterseite erzwungen wird, die Caller die Impl-Signatur nie sehen.

**Finale Struktur (KD-2, beide Facetten):** drei Overloads nach Argument-Anzahl —
`sum()` (0 Args, volle Reduktion), `sum(axis)` (1 Arg, keepdims fest `false`),
`sum(axis, keepdims)` (2 Args, BEIDE required, `KeepDims extends boolean | undefined`) —
plus die unveränderte Impl-Signatur mit `NDArray<any>`-Rückgabe. **KD-2 erweitert
zusätzlich `spike/src/reduce.ts`:** die `KeepDims`-Constraint an `ApplyAt`/`ResolveAndApply`/
`ReduceAxis` von `boolean` auf `boolean | undefined`, damit die ungestrippt ankommende
`keepdims`-Union über die bereits vorhandenen nackten `KeepDims extends true`-Conditionals
von selbst korrekt zu einer **ehrlichen Shape-Union** distribuiert (kein `IsUnion`-Filter
nötig). Die exakten Signatur-Zeilen zurrt die Impl fest; Akzeptanzkriterium ist die
Constraint-Tabelle unten (Machbarkeits-Spike hat sie erfüllt).

Die vier kanonischen Aufrufformen (Rückgabetypen unverändert): `sum()` → `NDArray<[]>`,
`sum(axis)` → `ReduceAxis<S, Axis, false>`, `sum(axis, keepdims)` → `ReduceAxis<S, Axis,
KeepDims>`, `sum(undefined, keepdims)` → `AllOnes<S>` bei `KeepDims=true`.

**Bindende Anforderungen T1a:**
1. **M2-Ziel, BEIDE Facetten (das eigentliche Warum):**
   - *axis:* `a.sum(u)` mit `u: 0 | undefined` (ohne Typargument) ist nicht mehr konfident
     `NDArray<[3]>` — Implementierungs-Ausgang (verifiziert, korrigiert ggü. Entwurf):
     **Degradation zu no-claim** `readonly number[]` (never-wrong, Menü (i) — NICHT ein
     Compile-Fehler; der frühe Entwurf sagte irrig „Compile-Fehler", A-Verify-Befund D1); der
     `UA_GAP`-Sentinel-Pin kippt und wird **umgekehrt** (er bewacht künftig die Schließung,
     nicht die Lücke).
   - *keepdims:* `a.sum(0, kd)` mit `kd: true | undefined` ist nicht mehr konfident
     `NDArray<[1,3]>` — KD-2-Ausgang: **ehrliche Union** `readonly [3] | readonly [1, 3]`
     (degradiert wie axis, kein Fehler). Analog `a.sum(undefined, kd)` → `readonly [] |
     readonly [1, 1]`.
2. **Keine Regression der vier Aufrufformen** — jede behält ihren korrekten Rückgabetyp,
   auf BEIDEN Surfaces (NDArray + WNDArray, im Spike beide real verifiziert). Bestehende
   Positiv-Pins bleiben grün oder werden intent-erhaltend re-expressiert.
3. **Guard/M3 erhalten** — `sum(9)` auf Rang 2 bleibt ein Compile-Fehler am Argument,
   Message-Stamm **wortgleich** zum Runtime-Throw (`reduce: axis 9 is out of range for
   shape [2,3] (rank 2)`); Spike-verifiziert byte-gleich zur Baseline. Kein generisches
   TS2769 ohne den Stamm.
4. **Hover-Qualität (M3)** — Klassen-Hover bleibt `NDArray<[2, 3]>`; die Overloads dürfen
   die Hover-Anzeige nicht verschlechtern (LSP-Harness `bench:editor` prüfen).
5. **Insertion-Disziplin — DISCLOSED DEVIATION (Owner-bestätigt 2026-07-17).** Der Fix ist
   NICHT strikt insertion-only: die Impl-Signatur bekommt eine geänderte Rückgabetyp-
   Annotation (`NDArray<OkShape<…>>` → `NDArray<any>`, EIN Token). Das ist eine Änderung
   AM `sum`-Member selbst (dem Ziel der Scheibe), nicht an einem fremden Member — analog
   zur owner-bestätigten Kern-09-D3-Abweichung (ein neuer Param kann nicht insertion-only
   sein). `reduce.ts`s `KeepDims`-Constraint-Erweiterung ist ebenfalls ein EDIT an
   bestehenden Typen (rückwärtskompatibel: alle bestehenden `.sum(axis, keepdims)`-Aufrufe
   nutzen reines `boolean`, keiner bricht — Spike-verifiziert). `reduce.ts` ist KEINE
   frozen-baseline-Datei; die Artefakt-append-only-Regel gilt nicht.
6. **Explizit-Typ-Argument-Pin** — `a.sum<0|undefined>(u)` bleibt korrekt (degradiert;
   Spike-verifiziert grün).

**Testabdeckung T1a:** aktualisierte/neue `.test-d.ts`-Assertions (in BESTEHENDEN Dateien,
kein File-Add) auf beiden Surfaces: die vier Aufrufformen positiv, die axis-`Literal|
undefined`-Form (M2, jetzt Compile-Fehler → der `UA_GAP`-Sentinel umgekehrt), die
keepdims-`true|undefined`-Form (M2, jetzt ehrliche Union), OOB-Literal weiterhin am
Argument abgelehnt mit wortgleichem Stamm (M3), dynamische Achse/`number` degradiert,
Union-Achse degradiert (Union-Axis-Mini-Pins grün), dynamisches `boolean`-keepdims
degradiert (bestehend, unverändert). **Vollständiger Root-`tsc --noEmit`-Lauf ist Pflicht-
Gate** (Baustein-0-Lehre: der bestehende `keepdims.test.ts`-Union-Achsen-Test `[0,1,
undefined]` MUSS grün bleiben — Spike-verifiziert). Runtime-Verhalten unverändert
(Overloads sind type-only) → Parität-Check, dass `sum()`/`sum(0)`/`sum(0,true)`/
`sum(undefined,true)` dieselben Laufzeit-Shapes liefern wie vorher.

### T1b — Umbenennung `slice-literal.ts` → `literal-arithmetic.ts`

**Begründung.** Seit Spike 04 beherbergt die Datei die gesamte Digit-String-Arithmetik
(Subtraktion, Vergleich, Addition, Schulbuch-Multiplikation `MulDigits`, Long-Division
`DivCeil`) plus die Literal-Klassifikatoren (`LiteralIndexBounds`, `LiteralStepInvalid`,
`LiteralReshapeDimInvalid`, `LiteralShapeProduct`, `IsUnion`) — sie ist nicht mehr
slice-spezifisch (FOLLOWUPS). Zielname `literal-arithmetic.ts` (deckt Arithmetik +
Literal-Klassifikation ab; Alternative `digit-arithmetic.ts` verworfen, da auch
Nicht-Digit-Klassifikatoren enthalten sind).

**Umbenennungs-Fläche (Baustein-0-korrigiert, vollständig verifiziert 2026-07-17).**
Die ursprüngliche Fläche übersah drei Korpora — hier die vollständige:

- **Import-Sites, die den Typecheck brechen (MÜSSEN aktualisiert werden), 9 gesamt:**
  - `spike/src/**` (5): `reshape.ts:33`, `slice.ts:59`, `ndarray.ts:43`, `vector.ts:32`,
    `wasm/resident.ts:75`.
  - `spike/tests/**` (3, **ROOT-`pnpm check`-Korpus** — `tsconfig.json` include `["spike"]`
    schließt `spike/tests` NICHT aus): `reshape.test-d.ts:12`, `slice.test-d.ts:14`,
    `product.test-d.ts:11`. **Das war die Baustein-0-Lücke** — ohne diese drei sprengt
    `git mv` den Gate-Block.
  - `spike/tests-stress/**` (1, **stress-Korpus**, Teil von `pnpm check`):
    `product-stress.test-d.ts:12`.
- **Kommentar-/Doc-Referenzen (brechen nichts, für Korrektheit aktualisieren), 19:**
  - `spike/src/**` (11): `reshape.ts` ×5, `dim.ts` ×3, `slice.ts` ×2, `vector.ts` ×1.
  - `spike/tests/**` (3): `reshape.test-d.ts` ×2, `slice.test-d.ts` ×1.
  - `spike/bench-dx/**` (5, reine Kommentare/Codegen-Doc-Strings, KEINE Importe):
    `gen-workloads.ts` ×4, `workloads/w3-slice.ts` ×1. `w3-slice.ts` wird von
    `gen-workloads.ts` GENERIERT (regeneriert beim nächsten `pnpm bench:editor`), also
    ist die `gen-workloads.ts`-Aktualisierung die Quelle.
- **Historisch unverändert:** die 18 `docs/`-Referenzen + Root-Markdown (`HANDOFF.md`,
  `COVENANT.md`, `FOLLOWUPS.md`, `CLAUDE.md`) — FOLLOWUPS-Konvention: Doc-Verweise auf
  historische Namen sind ok. Ein einzeiliger Vermerk am Dateikopf der umbenannten Datei
  nennt den alten Namen für die Rückverfolgbarkeit.

**Bindende Anforderungen T1b:**
1. `git mv spike/src/slice-literal.ts spike/src/literal-arithmetic.ts` (Historie erhalten).
2. Alle 9 Import-Pfade + die 14 src/tests-Kommentar-Referenzen + die 5 bench-dx-Referenzen
   aktualisiert; kein `slice-literal`-String verbleibt in `spike/**` als funktionale/Import-
   Referenz — grep über den GANZEN `spike/`-Baum == **genau 1** (nur der bewusste
   Traceability-Vermerk im Dateikopf der umbenannten Datei, der den alten Namen für die
   Rückverfolgbarkeit nennt; Präzisierung ggü. dem Entwurf-„==0", A/B-Verify-Befund D2/6a).
   `docs/` + Root-Markdown ausgenommen.
3. **Zyklus-Gate (Baustein-0-präzisiert).** `graph-a-lama query cycles --kind import`
   ist **schon jetzt nicht leer** — es existiert eine benigne Item-10-Import-SCC
   (`ndarray.ts` ↔ `wasm/backend-api.ts` ↔ `wasm/resident.ts` ↔ `wasm/threaded.ts`;
   jede Rück-Kante ist `import type` oder der gepinnte dynamische
   `await import("./wasm/threaded.ts")` ndarray.ts:352 — kein eager Value-Zyklus, kein
   M5/S1-Verstoß). Gate deshalb NICHT „bleibt leer", sondern: **die `cycles`-Liste ist
   vorher/nachher zusammensetzungs-identisch; kein NEUER Zyklus beteiligt `dim.ts`/
   `literal-arithmetic.ts`** (Diff der Listen, nicht Leerheit). `dim.ts` hält bewusst
   eine PRIVATE `IsUnion`-Kopie (dim.ts:49/53/60), weil `literal-arithmetic.ts` von
   `dim.ts` importiert — die Umbenennung darf diese Richtung nicht umkehren.
4. **Order-Noise-Isolierung:** T1b wird als LETZTER Schritt von S1 ausgeführt; der
   `check:diag`-Pin danach ist der neue S1-Baseline (dokumentiert als „Umbenennungs-
   bedingte File-Set-Änderung, nicht gegen 178'865 vergleichbar").
5. **Test-Script-Guard:** die Umbenennung berührt keine `.test.ts`-Datei-Liste (die
   umbenannte Datei ist eine `src/`-Datei; die drei `.test-d.ts` sind Typ-Tests, stehen
   in keiner package.json-Test-Liste); `test-scripts-guard.test.ts` bleibt grün.

**S1-Ausgangs-/ Zielcommit:** Ausgang HEAD (b51c1a7). Ein Commit „Item 11 / S1".

---

## Sub-Scheibe S2 — Emit-/Paket-Pipeline

Baut das publizierbare Paket aus `spike/src/**`. Berührt M3, M4, M5, Z2, S1.
Kein `crates/`. Die drei verifizierten Blocker werden hier gefixt.

### Die drei verifizierten Blocker (Ist-Zustand, empirisch belegt 2026-07-17)

- **Blocker 1 — `.d.ts` behalten `.ts`-Import-Endungen.** Emit-Probe (ganzer
  `spike/src`-Baum, TS 7.0.2, `rewriteRelativeImportExtensions: true`,
  `declaration: true`): die `.js`-Emission schreibt korrekt `from "./ndarray.js"`, die
  `.d.ts` emittiert aber `from "./dim.ts"` (verifiziert: `dist/index.js` vs.
  `dist/index.d.ts` in der Scratch-Probe). Ein Konsument bekäme „Cannot find module
  './dim.ts'". `rewriteRelativeImportExtensions` deckt in 7.0.2 die `.d.ts`-Emission
  nicht ab.
- **Blocker 2 — Worker-URL zeigt auf `.ts`.** `new URL("./threaded-worker.ts",
  import.meta.url)` (threaded.ts:121) bleibt im Emit wortgleich (verifiziert:
  `dist/wasm/threaded.js:88`). `rewriteRelativeImportExtensions` fasst String-Literale
  in `new URL(...)` nicht an. Im Paket (nur `.js`) findet der Worker-Spawn die Datei
  nicht. Betrifft nur den Threads-Pfad.
- **Blocker 3 — `node:worker_threads` in der internen `.d.ts`.** `dist/wasm/
  threaded.d.ts:1` importiert `import { Worker } from "node:worker_threads"` top-level
  (für das interne `PoolWorker`-Interface). Umfang (bricht das den Standard-Konsumenten,
  der nur `NDArray` nutzt und `type ThreadedBackend` transitiv lädt?) ist **empirisch zu
  charakterisieren** (s. D-S2.3).

### Binding-Entscheidungen S2

**D-S2.1 — dist-Emit-Konfiguration.** Neue `tsconfig` (Kandidat-Name
`spike/tsconfig.dist.json` oder repo-root `tsconfig.build.json`): `declaration: true`,
`rewriteRelativeImportExtensions: true`, `module: "ESNext"`, `target: "ES2022"`,
`moduleResolution: "bundler"`, `noEmit: false`, `outDir` = paket-`dist/`, `rootDir`
= `spike/src`, `include: ["spike/src/**/*.ts", "spike/src/ambient.d.ts"]`. Verifiziert
(Scratch-Probe, EXIT 0): dieser Emit erzeugt `.js`+`.d.ts` für alle 17 Value-Module +
Barrel; `ambient.d.ts` erzeugt **keinen** Output (Node-Shims lecken nicht ins Paket).

**D-S2.2 — Post-Emit-Rewrite (Blocker 1 + 2 gemeinsam).** Ein zero-dep Node-Skript
(Kandidat `scripts/postbuild-dist.mjs`) läuft nach `tsc` über `dist/**` und schreibt
relative `.ts`-Endungen in **Import/Export-Statements** (`from "./x.ts"`) und in
`new URL("./x.ts", …)`/`import("./x.ts")`-Formen zu `.js` um — in `.js` UND `.d.ts`.
Präzision ist bindend: NUR relative Pfade (`./`/`../`) mit `.ts`-Endung in genau diesen
syntaktischen Positionen; KEINE Umschreibung in Doc-Kommentaren, Strings mit `.ts` in
Prosa, oder Nicht-Modul-Kontexten. Baustein 0 + die S2-Verify prüfen die Präzision an
einem Diff der emittierten Dateien (insb. `threaded.js:88` → `.js`, `index.d.ts` →
alle `.js`). Alternative erwogen und verworfen: Source-Imports direkt auf `.js`
umstellen — bricht den `node --test spike/tests-runtime/*.test.ts`-Direktlauf (Node
lädt die `.ts`-Sources zur Testzeit; `./x.js` existiert dann nicht). Der Rewrite hält
Source `.ts` (Test-Direktlauf + `pnpm check`) und Paket `.js` getrennt.

**D-S2.3 — Blocker 3, minimal-invasiv (Baustein-0-charakterisiert).** Baustein 0 hat
den Umfang empirisch geklärt (Mini-Konsument, nur `import { NDArray }`, vier
Konfigurationen): **A** (`skipLibCheck:true`, kein `@types/node`) → **EXIT 0**; **B**
(`skipLibCheck:false`, kein `@types/node`) → **bricht** (`TS2591: Cannot find name
'node:worker_threads'`); **C** (`skipLibCheck:true` + `@types/node`) → EXIT 0; **D**
(`skipLibCheck:false` + `@types/node` + explizit `types:["node"]`) → EXIT 0. Fix-Rangfolge
(billigster tragfähiger zuerst):
- (a) **nichts** — der Standard-Konsument mit dem in Konsumenten-Projekten üblichen
  Default (`skipLibCheck:true`, von Vite/Next/CRA-Templates gesetzt) checkt sauber
  (Config A, EXIT 0 verifiziert). Für ein Node-only-experimentelles Feature ist der
  interne `node:worker_threads`-Import in `threaded.d.ts` damit akzeptabel (dokumentiert).
  **Primärkandidat.**
- (c) den `Worker`-Import in `threaded.ts` so umbauen, dass der Typ nicht in die
  öffentlich erreichbare `.d.ts` leakt — falls (a) nicht genügen soll (z.B. Anspruch,
  auch `skipLibCheck:false`-Konsumenten zu bedienen).
- **VERWORFEN (Baustein-0-Befund): `@types/node` als devDependency.** Kategorienfehler —
  **devDependencies eines publizierten Pakets werden bei Konsumenten NIE installiert**
  (Standard-npm/pnpm-Semantik); ob `numtype` selbst `@types/node` als devDep führt, hat
  null Einfluss auf das `node_modules` des Konsumenten. Zusatzbefund: selbst ein
  Konsument, der `@types/node` SELBST installiert, braucht in TS 7.0.2 ein explizites
  `types:["node"]` (Config C/D), automatische `@types`-Aufnahme greift nicht.

**Bindend:** der Standard-`NDArray`-Konsument (kein Threads, browser-sichtbar) darf durch
`dist/index.d.ts` NICHT gezwungen werden, `@types/node` zu installieren oder Threads-Typen
aufzulösen (M5). Der Ausgang wird als Entscheidung gepinnt (Erwartung: (a)).

**D-S2.4 — `.wasm`-Bundling.** Der Build kopiert `spike/src/wasm/numtype_core.wasm` nach
`dist/wasm/numtype_core.wasm` (modul-relativ zu `dist/wasm/loader.js`, wegen
`new URL("./numtype_core.wasm", import.meta.url)`). Byte-Identität gegen den Freeze-Pin
verifiziert (`shasum -a 256`). Das **Threads-Artefakt** `numtype_core_threads.wasm`
wird **NICHT** kopiert (Owner-Gabel: Option 2). Der Threads-JS-Code + `.d.ts` sind Teil
des normalen Emit (im Paket vorhanden); `NDArray.backend("threaded")` wirft ohne
Artefakt die bereits gepinnte Meldung `NDArray.backend("threaded"): threaded backend
requires Node with the threads artifact (threads artifact not found)`
(ndarray.ts:350 + backend-api.ts:53) — genau das Option-2-Verhalten, kein neuer Code
nötig, nur ein Test, der diese Meldung im Paket-Kontext bestätigt (S3).

**D-S2.5 — `package.json`-Metadaten (ein Barrel-Einstieg).** Owner-Gabel: **ein**
Export `"."`.
```jsonc
"exports": { ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js" } },
"main": "./dist/index.js",        // Legacy-Resolver
"module": "./dist/index.js",
"types": "./dist/index.d.ts",
"files": ["dist"],                 // .wasm ist gitignored → MUSS über files/dist rein
"sideEffects": false               // Kandidat — S2 prüft VOLLSTÄNDIG vor dem Setzen
```
Zu `sideEffects: false`: Baustein 0 hat die Top-Level-Statements aller 18 Module
stichprobenartig gescannt (`loader.ts:142` `WASM_URL`, `resident.ts` `FinalizationRegistry`,
`threaded.ts` `WASM_URL`/`WORKER_URL`) — keiner mutiert globalen Zustand bei ungenutztem
Modul, der Kandidat ist plausibel; **S2 geht alle 18 Module vor dem Setzen vollständig
durch** (nicht nur Stichprobe).

`"type": "module"` bleibt. **`"private": true` bleibt bis Item 13/Release** (schützt vor
versehentlichem `npm publish` während Item 11/12); die Publish-Metadaten werden gesetzt,
aber das Paket ist noch nicht publish-scharf. Ergänzende Metadaten-Felder (`description`
existiert, `keywords`, `repository`, `license`, `author`) — `license` als FELD gesetzt,
die LICENSE-DATEI + finale Rechtewahl ist Item 13 (Release-Mechanik); Kandidat MIT.
Build-Script `pnpm build:dist` (Kandidat): `tsc -p <dist-tsconfig> && node
scripts/postbuild-dist.mjs && <copy numtype_core.wasm>`.

**D-S2.6 — Z2-Verbund.** Falls S2 eine neue tsconfig einführt, die typgeprüft werden
soll (die dist-tsconfig emittiert, prüft aber auch): entscheiden, ob `pnpm check` sie
aufnimmt oder ob der bestehende Root-Check die Sources bereits abdeckt (er tut es —
`spike/src/**` ist im Root-`include`). Der dist-Emit ist ein BUILD, kein zusätzlicher
CHECK — `pnpm check` bleibt der Verbund der noEmit-Korpora; nichts rottet, weil die
Sources schon im Root-Check sind. Bindend: keine Regression von Z2.

**S2-Testabdeckung:** primär über S3 (Paket-Smoke). In S2 selbst: der Emit läuft grün
(EXIT 0), der Post-Emit-Rewrite-Diff ist präzise (kein Prosa-Treffer), die `.wasm`-Kopie
ist byte-identisch, alle Gate-Block-Kommandos grün, Freeze-Hash unverändert. Der
Threads-Pfad-Gate (`test:threaded`) läuft, weil die Emit-Pipeline `threaded.ts` berührt
(Worker-URL-Rewrite) — die gebauten `.ts`-Sources bleiben aber die Testgrundlage
(Direktlauf), also prüft `test:threaded` das unveränderte Source-Verhalten.

**S2-Ausgangscommit:** S1-Commit. Ein Commit „Item 11 / S2".

---

## Sub-Scheibe S3 — Zero-dep-Guard + Paket-Smoke

Macht die Paketierungs-Claims zu geprüften Gates. Berührt Z1, M5, M3. Kein `crates/`.

**D-S3.1 — Zero-dep-Guard-Test.** Ein mechanischer Test (Runtime, Kandidat
`spike/tests-runtime/zero-dep-guard.test.ts`, in `test:core`-Liste): `JSON.parse`
der `package.json`, Assertion, dass **kein `dependencies`-Feld** existiert (bzw. leer)
— `devDependencies` sind erlaubt. Nicht-Vakuität: der Test fällt, wenn man testweise
ein `dependencies`-Feld einfügt (in der Verify-Runde als Mutant belegt). Deckt Z1 als
Gate ab.

**D-S3.2 — Paket-Smoke, zwei Ebenen.**
1. **Laufzeit-Smoke:** nach `pnpm build:dist` importiert ein frischer Node-Prozess aus
   `dist/index.js` (NICHT aus `spike/src`) und führt eine Grundoperation aus
   (`NDArray.fromArray(...).add(...)` o.ä.), Assertion gegen das erwartete Ergebnis.
   Beweist: das emittierte + rewritten Paket ist konsumierbar, die `.wasm` lädt über
   den paket-relativen Pfad, Blocker 2 (falls Threads berührt) und der Loader-Pfad
   funktionieren. Der `backend("threaded")`-ohne-Artefakt-Fall wirft die gepinnte
   Meldung (D-S2.4) — als Assertion.
2. **Typ-Smoke:** ein Mini-Konsumenten-`.ts` (`import { NDArray } from "<dist>"` bzw.
   relativ auf `dist/index.d.ts`) wird gegen `dist/index.d.ts` typegecheckt, mit
   Konsumenten-typischen Defaults (`skipLibCheck: true`). Beweist Blocker 1 gefixt
   (keine `.ts`-Modul-Fehler) und M3 (Hover — mindestens: ein Shape-Fehler erscheint am
   Argument, ein gültiger Aufruf resolved sauber). Deckt D-S2.3 (a) als Gate ab.

**D-S3.3 — Einordnung ins Test-Harness.** Die Smoke-Tests brauchen ein gebautes
`dist/` — sie laufen NICHT im Standard-`node --test`-Lauf (der die `.ts`-Sources direkt
lädt), sondern hinter einem eigenen Script (Kandidat `pnpm test:package`), das zuerst
`build:dist` fährt. `test-scripts-guard.test.ts` wird ggf. um die Invariante erweitert,
dass Paket-Smoke-Tests in `test:package` gelistet sind und in keiner `node`-Test-Liste
stehen (analog zur Browser-Invariante (d)). Order-Noise: Paket-Smoke-Tests sind
Runtime-Tests (`.test.ts`), keine `.test-d.ts` im Root-Korpus → kein `check:diag`-Move
aus S3 (zu verifizieren).

**S3-Ausgangscommit:** S2-Commit. Ein Commit „Item 11 / S3".

---

## Owner-Gabeln (ENTSCHIEDEN, 2026-07-17, Chat)

1. **Scheiben-Zuschnitt: drei Sub-Scheiben sequenziell** (S1 Typ-Vorarbeiten, S2 Emit-/
   Paket-Pipeline, S3 Zero-dep-Guard + Smoke), je eigene A+B+C-Verify-Runde. Begründung:
   sauberste Fehler-Attribution bei sechs unabhängigen Arbeitssträngen; Muster wie
   Phase-D-Vorarbeiten (eine Spec, drei verifizierte Sub-Scheiben).
2. **Threads im v0-Paket: Option 2 — Code ja, Artefakt nein.** Threads-JS/`.d.ts` im
   Paket + Worker-URL-Fix (Blocker 2, billig); die `numtype_core_threads.wasm` NICHT im
   Tarball; `backend("threaded")` meldet ohne Artefakt klar „aus Source bauen".
   Begründung: entkoppelt den Release von der angepinnten nightly-Toolchain (Publish-Zeit-
   Abhängigkeit statt Endnutzer-Laufzeit); der Threads-Nutzen (nur Node, nur große
   Matrizen) rechtfertigt das Release-Risiko von Option 1 nicht; Roadmap hat Threads für
   v0 ohnehin als experimentelles Node-only-Opt-in eingestuft.
3. **Öffentlicher Einstieg: ein Barrel** (`"."` = `numtype`), re-exportiert alles wie
   `index.ts` heute (NDArray + WasmBackend als Wert, ThreadedBackend type-only, alle
   Typen). Begründung: ein Import-Pfad, simpel; der Default-`NDArray`-Pfad bleibt
   browser-sicher (kein eager `node:*`), Subpath-Trennung (`numtype/backend`) unnötige
   exports-Fläche für v0.

## Nicht in dieser Scheibe (bewusst)

- **npm-Name sichern.** `numtype` ist verifiziert frei (Registry-404 am 2026-07-17,
  ebenso `@numtype/core`). Die Sicherung = `npm publish` = **Owner-Aktion bei Release
  (Item 13)**; kein Impl-Schritt hier (Publish ist irreversibel/state-changing, wird
  nicht autonom ausgeführt). In der Spec vermerkt.
- **LICENSE-Datei + Rechtewahl** — Item 13 (Release-Mechanik). S2 setzt nur das
  `license`-FELD als Kandidat.
- **CI / GitHub Actions** (Freeze-Hash-Gate, bench:editor als Gate) — Item 12.
- **WebKit/Firefox-Browser-Smoke** — FOLLOWUPS, v0 Chromium-only (Spec-Entscheidung V3).
- **`-1`-Dim-Inferenz in `reshape`** — FOLLOWUPS, eigene Scheibe.

## COVENANT-Konsequenz (M2-Notiz, erfordert Owner-Bestätigung)

S1/T1a schließt den in COVENANT.md v2 unter M2 dokumentierten offenen Verstoß
(`Literal|undefined` durch optionale Parameter) **VOLLSTÄNDIG — beide Facetten** (axis:
Compile-Fehler; keepdims/KD-2: ehrliche Union). Nach erfolgreicher T1a-Verifikation:
COVENANT.md-Aktualisierung mit **Version-Bump (v2 → v3) + Changelog** — die M2-Notiz
(inkl. UA_GAP-Sentinel-Beschreibung) wandert von „bekannter offener Verstoß" zu
„geschlossen in Item 11 / S1" (der Sentinel-Pin wird umgekehrt: bewacht künftig die
Schließung). **Nur mit Owner-Bestätigung**, nie still (Covenant-Regel). Wird als Teil
des S1-Abschlusses vorgelegt. (Owner hat die Richtung KD-2 + die `NDArray<any>`-Insertion-
Abweichung bereits am 2026-07-17 bestätigt; die formale v3-Textänderung folgt bei S1-Landung.)

## Verifikation

- **Baustein 0 (Spec-Verifier, adversarial, VOR Implementierung von S1):** EIN
  `brainroute:deep`, frischer Kontext, Auftrag aus docs/verify-runde-template.md
  Baustein 0. Schwerpunkte: (1) die Code-Annahmen dieser Spec (alle Zeilenanker, insb.
  `sum`-Signaturen, `ReduceAxis`-Semantik, die drei Blocker empirisch am Emit
  reproduzieren); (2) die Overload-Auflösung von T1a EMPIRISCH in einem Scratch-Worktree
  (löst der Split die `Literal|undefined`-Form wirklich never-wrong auf? bricht er eine
  der vier Aufrufformen? Hover?); (3) die Post-Emit-Rewrite-Präzision (D-S2.2 —
  falsch-positive Treffer in Prosa?); (4) Blocker-3-Umfang (D-S2.3 — bricht der Standard-
  Konsument?); (5) Covenant-Abgleich (M2/M3/M4/M5/Z1/Z2/S1). Alle Typ-/Emit-Proben in
  eigenen Scratch-Worktrees, Haupt-Tree unberührt (git-status-Beweis).
- **Je Sub-Scheibe:** volle Runde A (Spec-Konformität, eigener Mutant) + B (adversarial,
  breite Mutanten) + C (covenant-verify, parallel; Input COVENANT.md + Diff + berührte
  IDs + Lint-Output). Merge, jeden major+ adressieren, Post-Verification-Addendum.

## Ergebnis-Artefakte je Sub-Scheibe

`docs/item-11-s{1,2,3}-ergebnisse.md` mit Post-Verification-Addendum (alle Verdikte),
KB-Capture der generalisierbaren Lektionen (Kandidaten: T1a „verschmolzene required-
Overload + `any`-Impl-Rückgabe gegen Optional-Parameter-`undefined`-Stripping unter
Invarianz — M2-Muster"; S2 „rewriteRelativeImportExtensions deckt in TS 7.0.2 die
.d.ts-Emission + String-URL-Literale NICHT ab → Post-Emit-Rewrite"; S2 „ambient.d.ts
leckt nicht in declaration-Emit"; S2 „devDependencies helfen einem Paket-Konsumenten
nie"), FOLLOWUPS-Austragungen (Overload-Split-Item BEIDE Facetten, slice-literal-
Umbenennung, Zero-dep-Guard, npm-Name-Status), CLAUDE.md-Pin-Updates (mit Order-Noise-
Vermerk für S1), Commit je Sub-Scheibe.

---

## Adversariale Spec-Verifikation (Addendum, Baustein 0 + T1a-Machbarkeits-Spike, 2026-07-17)

Zwei `brainroute:deep`-Läufe, je frischer Kontext, alle Proben in eigenen
Scratch-Worktrees gegen TS 7.0.2, Haupt-Tree unberührt (git-status-Beweise; ein
cwd-Vorfall im zweiten Lauf erzeugte kurz sechs leere Dateien im Haupt-Tree — vom Agenten
sofort gelöscht, vom Orchestrator per eigenem `git status` als bereinigt verifiziert).

**Baustein 0 (Spec-Verifier).** Bestätigt: alle drei Emit-Blocker empirisch reproduziert
(B1 `.d.ts` behält `.ts`; B2 Worker-URL `.ts`; B3 `node:worker_threads` in
`threaded.d.ts`), die Code-Annahmen (sum-Signaturen, ReduceAxis, checkThreadedEnv,
ambient.d.ts-kein-Leak, D-S2.1-Emit EXIT 0, Rewrite-Präzision, `.wasm`-Gitignore, Z2)
halten. **Eingearbeitete Blocker/Korrekturen:**
- **T1a-Struktur des Erstentwurfs kompiliert nicht** (TS2394 unter `__variance`); ihre
  naive Reparatur verletzt M3 + bricht Pin + `keepdims.test.ts`. → T1a neu entworfen (s.
  T1a-Abschnitt + Machbarkeits-Spike unten), Owner-Richtungsabnahme KD-2.
- **T1b-Umbenennungsfläche unvollständig** (übersah `spike/tests/*.test-d.ts` im
  Root-Korpus + `gen-workloads.ts`). → T1b-Fläche vollständig korrigiert (9 brechende
  Import-Sites).
- **Zyklus-Gate falsch** („bleibt leer" — ist schon jetzt nicht leer, benigne Item-10-SCC).
  → auf Diff-Formulierung präzisiert.
- **D-S2.3 Fix-Option (b) Kategorienfehler** (devDeps helfen Konsumenten nie); Blocker-3-
  Umfang empirisch geklärt (Config A `skipLibCheck:true` = EXIT 0). → (b) verworfen, (a)
  als Primärkandidat.
- `sideEffects:false` nur stichprobengeprüft → S2-Anforderung „vollständig".

**T1a-Machbarkeits-Spike (2 Runden).** Verdikt: **eine tragfähige Struktur existiert**,
am echten `ndarray.ts` + `resident.ts` verifiziert, alle zehn axis-Constraints grün inkl.
M3 (OOB-Message byte-gleich zur Baseline), inkl. des bestehenden `keepdims.test.ts`-Union-
Achsen-Musters, mit dem `UA_GAP`-Sentinel als einzigem (erwartetem) roten Pin. Kern-
Mechanik: (1) `undefined`+`number` in EINE required Overload verschmelzen (nicht spalten —
Spaltung bricht TS2394 unter `__variance`); (2) Impl-Signatur-Rückgabetyp auf `NDArray<any>`
(extern unsichtbar, Guard/M3 unberührt). Zweite Runde (keepdims-Facette): mit reiner
axis-Struktur bleibt `sum(0, kd)`/`kd: true|undefined` offen (konfident `readonly [1,3]`);
KD-2 schließt sie via `reduce.ts`-`KeepDims`-Constraint-Erweiterung (`boolean` →
`boolean|undefined`) zu ehrlicher Union `readonly [3] | readonly [1,3]`. **Owner-Wahl KD-2
(2026-07-17)**, `NDArray<any>`-Insertion-Abweichung offengelegt + bestätigt. Alternative
KD-1 (Compile-Fehler statt Union, ohne `reduce.ts`-Änderung) dokumentiert verworfen.
