# Phase-D-Vorarbeiten — Scheibe V3 (Browser-Smoke-Test): Ergebnisse

Spec: `docs/phase-d-vorarbeiten-spec.md` (Scheibe V3, D-V3.1–D-V3.5 + Gates), bereits
Baustein-0-verifiziert (Addendum am Ende der Spec). Datum: 2026-07-12. Alles unten ist in
tatsächlich in dieser Session ausgeführten Kommandos verankert; Fehlschläge/Überraschungen
werden benannt, nicht übermalt.

**Update (2026-07-12, nach der Zwei-Verifier-Runde):** Baustein A (CONFIRMED mit Auflage)
und Baustein B (HÄLT mit Befunden) haben vier Befunde erbracht. Drei sind jetzt IN-SLICE
geschlossen (F1 Stale-`.emit`-Guard, F2 Streaming-Pfad-Assertion, F3
`playwright.config.ts`-Typprüfung — Abschnitt „In-Slice-Schließungen nach der Verify-Runde"),
der vierte (F4) ist ein dokumentierter Akzeptanz-Punkt. Der ursprüngliche „Pins"-Abschnitt
enthielt außerdem eine FALSCHE Attribution (das +78-Delta wurde fälschlich dem
Kern-11-Order-Mechanismus zugeschrieben, ohne eigene Bisektion) — korrigiert unten unter
„Baustein-A-Befund 1", nicht stillschweigend überschrieben. Alle Gates wurden nach den
Schließungen frisch erneut gelaufen (Abschnitt „Post-Schließungs-Gates").

## Was gebaut wurde

Beweist die bislang ungeprüfte Architektur-Behauptung „das Standard-Surface (`NDArray`) und
`NDArray.backend('wasm')` laufen im Browser, COOP/COEP-frei" — erstmals in einem echten
Chromium statt per Node-`moduleLoadList`-Trace (Item 10). **Null `spike/src`-Änderungen
außer der vorgeschriebenen `node:http`-Ambient-Deklaration** (D-V3.5 eingehalten).

- **`spike/src/ambient.d.ts`** (angehängt, D-V3.3-Vorbedingung): `declare module "node:http"`
  — minimales, scoped Shim (`createServer`, `listen(port, hostname, cb)`, `.address()`,
  `close(cb)`, Response `writeHead`/`end`) im selben Muster wie die acht bestehenden
  `node:*`-Deklarationen dort.
- **`spike/tests-browser/tsconfig.json`** (neu, D-V3.2, Infra-01-Muster): standalone,
  Compiler-Optionen inline vom Root gespiegelt, `include: [".", "../src/ambient.d.ts"]`,
  `exclude: ["./.emit"]`. Das ist der dritte `pnpm check`-Verbund-Leg.
- **`spike/tests-browser/tsconfig.emit.json`** (neu, D-V3.3, BUILD-ONLY — nicht Teil von
  `pnpm check`): emittiert den echten Produkt-Baum (`spike/src/**/*.ts`) nach
  `spike/tests-browser/.emit/` via `rewriteRelativeImportExtensions` (TS 7.0.2, empirisch
  vorab verifiziert — siehe „Docs-first" unten), `rootDir: ".."`, `outDir: "./.emit"` —
  bewahrt die Baumstruktur (`spike/src/wasm/loader.ts` → `.emit/src/wasm/loader.js`), damit
  `WASM_URL`s modul-relatives `new URL("./numtype_core.wasm", import.meta.url)` nach der
  Emission weiter neben der `.wasm`-Datei landet.
- **`spike/tests-browser/server.ts`** (neu): minimaler `node:http`-Static-File-Server, MIME
  `application/wasm` für `.wasm`, `text/javascript` für `.js`, sonst
  `application/octet-stream`; synthetisiert eine leere HTML-Seite für extensionlose Pfade
  (kein eingecheckter `index.html` nötig). **Keine COOP/COEP-Header** — das ist der Punkt des
  Tests.
- **`spike/tests-browser/smoke.test.ts`** (neu, D-V3.4): 4 Playwright-Tests gegen einen in
  `test.beforeAll`/`afterAll` verwalteten Server, jeder mit frischer `page`:
  1. Umgebungsbeweis (`typeof process === "undefined"`, `crossOriginIsolated === false`).
  2. Op-Matrix-Differential komplett IN der Seite (`page.evaluate`, dynamischer
     `import()` der emittierten `ndarray.js`): `fromArray`/`zeros`/`ones`; `add`/`sub`/`mul`/
     `div` (gleiche Shape) + ein Broadcast-Fall (`[2,3]+[3]`); `matmul` (resident `matmul()`
     ruft IMMER `nt_matmul_blocked` — kein separater Pfad nötig, resident.ts:709); `sum` mit
     `axis` UND `keepdims`; `transpose`-View; Offset-Slice-View (`.slice(1)`, echter
     Nicht-Null-Offset — resident.ts' `slice()`-Mechanik verifiziert); `reshape`
     View-Zweig (kontiguierlich) UND Materialize-Zweig (über eine transponierte Quelle
     erzwungen — `isContiguous()` schlägt für jede Rang-≥2-Transposition fehl,
     resident.ts:328); `dot`/`norm`/`cosineSimilarity`; ein Spezialwerte-Sample (NaN, −0,
     +Inf, ein echtes Subnormal `5e-324`, −Inf) durch `add` UND `matmul` (letzteres der
     Kern-10-Concern: SIMD-blocked matmul darf Subnormals nicht flushen). Vergleich:
     `Object.is` pro Element (Muster `assert-helpers.ts`s `assertDataBitIdentical` —
     unterscheidet +0/−0, behandelt NaN als gleiche Werte-KLASSE), inline repliziert (nicht
     importiert — der Comparator muss als Quelltext in die Seite verschifft werden;
     `spike/tests-runtime/` ist ein separater, Node-only-Korpus).
  3. Streaming-Pfad: `typeof WebAssembly.instantiateStreaming === "function"` UND der
     `.wasm`-Response trägt `content-type: application/wasm` (via `page.waitForResponse`,
     nicht vakuös — Mutationsbeweis unten).
  4. `NDArray.backend("threaded")` wirft mit dem gepinnten Message-Stamm.
- **`playwright.config.ts`** (neu, Repo-Root, D-V3.2): `testDir: "./spike/tests-browser"`,
  nur `chromium`-Projekt (v0-Scope), kein `webServer`-Eintrag (der Server lebt im Test-Fixture
  selbst, wie gefordert).
- **`spike/tests-runtime/test-scripts-guard.test.ts`** (erweitert, D-V3.1): ein viertes
  Invariante — jede `spike/tests-browser/*.test.ts`-Datei muss in `test:browser` gelistet
  sein UND darf NIE in `test:core`/`test:resident`/`test:threaded` landen (beide Richtungen
  geprüft, plus die Symmetrie „kein `tests-runtime`-Basename in `test:browser`"). Bleibt Teil
  von `test:core`.
- **`package.json`**: `check` → Dreier-Verbund; neu `check:diag:browser` (Symmetrie zu
  `check:diag:stress`, nicht gegated); neu `test:browser` (Build+Emit+Copy+Playwright als
  EINE explizite Kommandokette, gleiche Kette-von-`pnpm`-Schritten-Konvention wie `demo`).
  `@playwright/test` als devDependency.
- **`tsconfig.json`** (Root): `exclude` um `spike/tests-browser` ergänzt.
- **`.gitignore`**: `spike/tests-browser/.emit/`, `playwright-report/`, `test-results/`,
  `blob-report/`, `playwright/.cache/`.

## Entscheidungen mit Begründung

- **Emission+Kopie als eigene `test:browser`-Kommandokette, NICHT als Playwright
  `globalSetup`** (D-V3.3 offene Wahl): einfacher zu verifizieren (jeder Schritt ist ein
  eigenes, isoliert testbares Kommando — `rm -rf .emit && tsc -p tsconfig.emit.json && mkdir
  -p … && cp … && playwright test`), keine zusätzliche Kopplung an Playwrights
  Lifecycle-API, und spiegelt die bereits etablierte Projekt-Konvention (`demo`/`test:core`
  sind selbst `pnpm build:wasm && node …`-Ketten). Ein `globalSetup` wäre nur bei mehreren
  Playwright-Projekten/-Runs mit unterschiedlichen Fixtures im Vorteil — hier gibt es genau
  einen Testlauf.
- **Kein separates `harness.ts`/`index.html` mit `<script type="module">`.** Ursprünglich
  geplant, dann verworfen: `page.evaluate` kann selbst dynamisch `import()`ieren (funktioniert
  in jedem modernen Browser, auch klassisch, solange die Seite auf einem echten
  http(s)-Origin läuft) — das reduziert die bewegten Teile auf EINE Datei
  (`smoke.test.ts`) statt zwei Compile-Ziele, und die Ergebnis-Rückgabe über den
  `page.evaluate`-Returnwert (strukturiert serialisiert) ist robuster als eine
  `window.__RESULTS__`-Bridge. Die synthetisierte Blank-HTML in `server.ts` existiert nur
  noch, damit `page.goto()` einen echten Origin lädt (für sinnvolle
  `crossOriginIsolated`/`typeof process`-Assertions und einen korrekten `import()`-Base-URL).
- **Lose lokale `ArrLike`/`NDArrayNS`-Interfaces statt der echten `Guard`/`Shape`-Generics.**
  Der Test beweist RUNTIME-Parität und den rohen Lade-Pfad, nicht die
  Compile-Zeit-Shape-Checking-USP (die ist vollständig in `spike/tests/*.test-d.ts`
  abgedeckt); die echte generische Maschinerie in einem dynamisch getypten
  `page.evaluate`-Payload nachzubilden hätte reine Reibung ohne zusätzliche Coverage erzeugt.
- **`test:browser` reicht explizit `spike/tests-browser/smoke.test.ts` an `playwright test`**
  statt Playwright per `testDir` frei entdecken zu lassen — konsistent mit der bestehenden
  Haus-Konvention „explizite Dateilisten in `package.json`" (test:core/test:resident/
  test:threaded), die der Guard-Test genau in diesem Muster prüft.
- **Kein eigenes `check:diag:browser`-Gate** (nur Pin, wie `check:diag:stress`) — die Spec
  fordert für V3 kein hartes Browser-Budget, nur „erweiterter Verbund grün".

## Docs-first (vor der Implementierung verifiziert)

`rewriteRelativeImportExtensions` (TS 7.0.2) an der Primärquelle (`tsc --help --all`) UND
empirisch in einem Scratch-Projekt geprüft, bevor Produktcode angefasst wurde:
- `--help --all`: „Rewrite '.ts', '.tsx', '.mts', and '.cts' file extensions in relative
  import paths to their JavaScript equivalent in output files." — Default `false`.
- `--help --all` zu `allowImportingTsExtensions`: „Requires '--moduleResolution bundler' and
  either '--noEmit' or '--emitDeclarationOnly' to be set." — das schließt es für die
  Emissions-tsconfig aus (die will echten JS-Output, nicht nur `noEmit`/Declarations).
- Scratch-Probe (`a.ts` importiert `./b.ts` statisch UND dynamisch): mit
  `rewriteRelativeImportExtensions: true` und **ohne** `allowImportingTsExtensions`
  kompiliert `tsc` exit 0 und emittiert `import { B } from "./b.js"` sowie
  `await import("./b.js")` — beide Formen korrekt umgeschrieben, keine Diagnose. Das ist
  exakt das Muster, das `spike/src/ndarray.ts`s dynamischer
  `await import("./wasm/threaded.ts")` braucht.
- Playwright-Konfiguration: nicht aus dem Gedächtnis geraten, sondern direkt aus dem
  installierten Pakets eigenen `.d.ts` gelesen (`node_modules/.pnpm/playwright@1.61.1/
  node_modules/playwright/types/test.d.ts`) — `testDir`/`testMatch`/`testIgnore`/
  `projects`/`webServer`/`globalSetup`/`expect(actual, message)` alle dort verifiziert;
  Default-`testMatch` ist `**/*.@(spec|test).?(c|m)[jt]s?(x)` — `smoke.test.ts` matcht ohne
  weitere Konfiguration.
- `res.end(uint8Array)` (Node `node:http`, kein `Buffer`-Wrap nötig): empirisch per
  eigenständigem Node-Skript bestätigt (echter HTTP-Roundtrip, Bytes + Content-Type
  byte-identisch geprüft), bevor `server.ts` geschrieben wurde.

## Gates (echte Läufe, volle Outputs unten zusammengefasst)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm test:browser` | 4/4 „passed (0.9–1.1s)"; Gesamtlaufzeit inkl. cargo/tsc/Playwright-Start **2.88s real** (Budget < ~60s) | 0 |
| `pnpm check` (Dreier-Verbund) | clean, 3 Läufe: 0.64s / 0.74s / 0.75s | 0 |
| D-V3.2 Korruption (i): Typfehler in `smoke.test.ts` | `pnpm check` → `error TS2322` in `spike/tests-browser/smoke.test.ts(46,7)`, root-only `tsc --noEmit` allein **blind** (exit 0) — bewiesene Isolations-Asymmetrie; revertiert, `pnpm check` wieder grün | 1 → 0 → 0 |
| D-V3.2 Korruption (ii): Typfehler in `spike/tests/ndarray.test-d.ts` (Root-Korpus) | `pnpm check` → `error TS2322` an der eingefügten Zeile; revertiert via `git checkout --`, `pnpm check` wieder grün | 1 → 0 |
| D-V3.4.3 Mutationsbeweis: `.wasm`-MIME → `application/octet-stream` | `pnpm test:browser`: **2 von 4 Tests rot** (op-matrix, streaming) mit `TypeError: Failed to execute 'compile' on 'WebAssembly': Incorrect response MIME type. Expected 'application/wasm'.`; revertiert, alle 4 wieder grün | 1 → 0 |
| Guard-Erweiterung, Negativbeweis A: `smoke.test.ts` aus `test:browser` entfernt | `test-scripts-guard.test.ts` rot, nennt `smoke.test.ts` als unregistriert; revertiert (Datei-Diff exakt gleich wie vor der Mutation) | fail → pass |
| Guard-Erweiterung, Negativbeweis B: `smoke.test.ts` zusätzlich in `test:core` eingefügt | `test-scripts-guard.test.ts` rot, nennt `smoke.test.ts` als fälschlich in den Node-Listen; revertiert | fail → pass |
| `pnpm test:core` | **818/818** (817 Baseline **+1** — der neue Guard-Test) | 0 |
| `pnpm test:resident` | 4277 pass + 2 skip = 4279 (unverändert) | 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161/161 (unverändert) | 0 |
| `pnpm demo` | „TS, WASM v1, and WASM resident all agree on every showcase op" | 0 |
| `pnpm test:threaded` (nicht im V3-Pflichtgate, zusätzlich gelaufen) | 69/69 (unverändert) | 0 |
| Artefakt-Hash (`pnpm build:wasm` → `shasum -a 256`) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — exakt der Pin (inkrementeller Cargo-Build, „Finished … in 0.05s" — crates/ in dieser Scheibe nicht angefasst) | 0 |

## Pins

**Baseline-Reproduktion** (frischer `git worktree --detach HEAD` von Commit `8b39c15`,
`node_modules` per Symlink, `./node_modules/.bin/tsc` direkt aufgerufen um `pnpm`s
Sync-Check/TTY-Purge-Prompt zu umgehen):
- `check:diag`: **175.634 @ 132 Dateien** — exakt reproduziert.
- `check:diag:stress`: **103.882 @ 82 Dateien** — exakt reproduziert.

### Baustein-A-Befund 1 — Δ+78-Attribution korrigiert

Die ursprüngliche Fassung dieses Abschnitts ordnete das Haupt-Pin-Delta (+78) dem bereits
in Kern 11 gepinnten Check-Order-Mechanismus zu, ausdrücklich **ohne eigene Bisektion**
("nicht separat neu-bisektiert"). **Das war eine falsche Ferndiagnose — plausibel
klingend, aber nie am echten V3-Diff verifiziert.** Baustein A (Zwei-Verifier-Runde) hat
das Delta vollständig bisektiert: jede Änderungskomponente isoliert in einem frischen
Worktree gemessen, Baseline `175.634 @ 132` exakt reproduziert.

| Isolierte Änderung | `check:diag` | Delta ggü. Baseline |
|---|---|---|
| Baseline (Commit `8b39c15`) | 175.634 @ 132 | — |
| NUR `ambient.d.ts`s `node:http`-Deklaration | 175.634 @ 132 | **+0** |
| NUR `tsconfig.json`s Exclude-Erweiterung (`spike/tests-browser`) | 175.634 @ 132 | **+0** |
| Beide zusammen (`ambient.d.ts` + Exclude) | 175.634 @ 132 | **+0** |
| NUR `spike/tests-browser/`-Verzeichnis-Präsenz (Dateien liegen auf der Platte, root-`tsconfig` excludiert sie) | 175.634 @ 132 | **+0** |
| Kontrollprobe: 83 Zeilen reiner Kommentar an derselben Stelle in `test-scripts-guard.test.ts` wie die echte Guard-Erweiterung | 175.634 @ 132 | **+0** |
| NUR die echte `test-scripts-guard.test.ts`-Erweiterung (Invariante (d), das vierte Guard-Testpaar für `spike/tests-browser/`) | **175.712 @ 132** | **+78** |
| Voller V3-Diff (alle obigen Komponenten zusammen) | 175.712 @ 132 | **+78** |

**Schlussfolgerung:** Das Δ+78 ist ECHTER Instanziierungs-Mehraufwand der neuen
Guard-Logik selbst (die zusätzlichen `Set`-Operationen, Array-Methoden und String-Typen
der vierten Invariante ziehen eigene Instanziierungen) — **kein** Order-Noise-Artefakt.
Die Kontrollprobe (83 Zeilen reiner Kommentar an exakt derselben Stelle, Δ+0) schließt
"jede Textänderung an dieser Position im Datei-Graphen bewegt den Zähler" als Erklärung
aus: es ist spezifisch der neue TypeScript-Code, nicht seine Position. Der Datei-ANZAHL-Pin
bleibt bei 132 in allen Zeilen (kein neues Root-Korpus-File — `tsconfig.json`s
`exclude`-Erweiterung hält `spike/tests-browser` vollständig draußen, wie spezifiziert).

**End-Stand** (Haupt-Tree, `git status` vorher gezeigt — nach den F1–F3-Schließungen unten;
siehe „Geänderte/neue Dateien" für den exakten Korpus dieser Messung):

| Pin | Wert | Delta ggü. Baseline | Delta ggü. der ursprünglichen V3-Messung (vor F1–F3) |
|---|---|---|---|
| `check:diag` (Haupt) | **175.712 @ 132 Dateien** | **+78** (bisektiert, s. o. — echte Kosten) | +0 (F1–F3 fassen keine Root-Korpus-Datei an) |
| `check:diag:stress` | **103.882 @ 82 Dateien** | **+0** | +0 |
| `check:diag:browser` | **2.142 @ 75 Dateien** | — (kein Vorgänger-Pin vor V3) | **+602 Instanziierungen, +2 Dateien** (war 1.540 @ 73) |

Das `check:diag:browser`-Delta (+602 Instanziierungen, +2 Dateien) ist die erwartete
Kostenzunahme aus den drei Schließungen (Details siehe „In-Slice-Schließungen" unten):
+1 Datei `spike/tests-browser/emit-freshness.ts` (F1, neu), +1 Datei `playwright.config.ts`
neu ins `tests-browser`-Korpus aufgenommen (F3 — vorher außerhalb jedes `tsc`-Korpus; ihr
`defineConfig`/`devices`-Import aus `@playwright/test` zieht selbst generische Maschinerie),
plus die zusätzlichen typisierten Zeilen in `smoke.test.ts` (F1s `beforeAll`-Aufruf, F2s
`addInitScript`-Callback mit einem `Parameters<typeof WebAssembly.instantiateStreaming>`-Cast).
Kein hartes Budget-Gate für `check:diag:browser` vorregistriert (wie schon in der
ursprünglichen V3-Spec); Haupt- und Stress-Pin bleiben unverändert, da F1–F3 keine Root- oder
Stress-Korpus-Datei anfassen. Beide Haupt-/Stress-Werte liegen himmelweit unter jedem
denkbaren Budget-Gate.

## In-Slice-Schließungen nach der Verify-Runde

Die Zwei-Verifier-Runde (Baustein A: CONFIRMED mit Auflage; Baustein B: HÄLT mit
Befunden) hat vier Befunde erbracht. Drei sind hier IN-SLICE geschlossen (F1, F2, F3 —
alle test-only, null `spike/src`-Änderungen, Artefakt-Hash unverändert, per Clean-Rebuild
bestätigt); der vierte (F4) ist ein bewusster Akzeptanz-Punkt, kein Fix.

### F1 (MAJOR, Baustein B) — Stale-`.emit`-Freshness-Guard

**Befund:** `pnpm test:browser` (der Wrapper) emittiert bei jedem Lauf frisch (`rm -rf
.emit && tsc -p tsconfig.emit.json`), aber ein DIREKTER `pnpm exec playwright test
spike/tests-browser/smoke.test.ts` (ohne den Wrapper) serviert einen vorhandenen, beliebig
alten `.emit/`-Baum kommentarlos. Bewiesener False-Pass: ein mutiertes
`spike/src/wasm/resident.ts`, dessen `.emit/`-Kompilat noch die alte (korrekte) Version
widerspiegelt, lässt alle 4 Tests grün durchlaufen, obwohl die tatsächliche Quelle kaputt
ist.

**Fix:** neue Datei `spike/tests-browser/emit-freshness.ts` (`assertEmitFresh`), aufgerufen
aus `smoke.test.ts`s `test.beforeAll` VOR dem Server-Start. Kein separat geschriebener
Stempel: da `tsc -p tsconfig.emit.json` bei jedem Lauf JEDE Ausgabedatei frisch schreibt
(kein inkrementeller/composite Build konfiguriert, und der Wrapper löscht `.emit/` vorher
immer), trägt jede emittierte Datei bereits implizit einen "wann emittiert"-Stempel in ihrer
eigenen mtime. Die Prüfung vergleicht die maximale mtime über `spike/src/**/*.ts` (rekursiv,
inkl. `ambient.d.ts` — der Glob matcht auch `*.d.ts`-Dateien, ist also legitim mit
einbezogen) gegen die mtime der einen konkreten emittierten Datei, die der Test tatsächlich
lädt (`.emit/src/ndarray.js`). Fehlt die Referenzdatei ganz ODER ist eine Quelldatei neuer
als sie, wirft `assertEmitFresh` mit einer Meldung, die den korrekten Aufruf nennt
("... run `pnpm test:browser`, do not invoke playwright directly ..."). `spike/src/
ambient.d.ts` bleibt unangetastet — nur der bereits etablierte "lokaler Cast über den
dynamischen `node:fs/promises`-Import"-Trick aus `test-scripts-guard.test.ts` wird für
`readdir`/`stat` wiederverwendet, die das Ambient-Shim nicht deklariert.

**Catchability-Nachweis** (Scratch-Kopie, rsync ohne `node_modules/.git/target/.emit`,
`node_modules` symlinked; Haupt-Tree während der gesamten Prozedur unangetastet — `git
status` im Haupt-Tree vor und nach jeder Mutation geprüft, zeigt ausschließlich die
vorbestehenden V3-Slice-Änderungen):

1. Sauberer `pnpm test:browser`-Lauf in der Scratch-Kopie → 4/4 grün, `.emit/` frisch
   (mtime `23:41:46`).
2. `spike/src/wasm/resident.ts`s `ones()` NACH dem Emit mutiert (`nt_fill(buf.ptr, len, 1)`
   → `nt_fill(buf.ptr, len, 2)`; Quelldatei-mtime danach `23:41:57`, > `.emit`s mtime).
3. DIREKTER `pnpm exec playwright test spike/tests-browser/smoke.test.ts` (Wrapper umgangen)
   → **alle 4 Tests ROT**, jeder mit exakt: `Error: spike/tests-browser/.emit is stale: a
   file under spike/src/**/*.ts was modified after the last emission — run \`pnpm
   test:browser\` (it always re-emits fresh), do not invoke playwright directly against a
   stale .emit.` — geworfen in `emit-freshness.ts:83`, aus `smoke.test.ts`s `beforeAll`.
4. Mutation revertiert (Backup-Diff bestätigt identisch zum Original) → Wrapper (`pnpm
   test:browser`) erneut gelaufen → 4/4 grün (Fall „a": frischer Re-Emit mit
   zurückgesetzter Quelle).
5. Mutation ERNEUT angewendet, diesmal NICHT revertiert → Wrapper (`pnpm test:browser`)
   gelaufen → 3/4 grün, 1 ROT: die Op-Matrix-Differential mit `Error: ones: data mismatch
   ref=[1,1,1,1,1,1] got=[2,2,2,2,2,2]` (Fall „b": der Wrapper emittiert frisch, die
   Freshness-Guard selbst ist also trivial grün, aber die Differential-Matrix fängt die
   echte Verhaltensänderung ab — genau das von der Spec verlangte „beide Fälle zeigen").
6. Mutation final revertiert (Backup-Diff bestätigt identisch); Scratch-Kopie danach
   gelöscht.

### F2 (MAJOR, Baustein B) — Streaming-Pfadwahl assertieren

**Befund:** `if (false && !isNode && …)` in `loader.ts:158` (Scratch-Mutation aus Baustein
B) blieb unbemerkt — Test 3 (D-V3.4.3) prüfte nur die EXISTENZ von
`WebAssembly.instantiateStreaming` und die `.wasm`-Response-MIME, nicht die tatsächlich
GENOMMENE Pfadwahl. Mit der Mutation läuft der Loader über den ArrayBuffer-Fallback (der
ebenfalls per `fetch(WASM_URL)` lädt und damit zufällig dieselbe MIME sieht) — beide
Alt-Assertions bleiben grün, obwohl der Streaming-Zweig nie erreicht wird.

**Fix (test-only, KEINE `loader.ts`-Änderung):** `page.addInitScript` in Test 3 installiert
VOR jeder Seiten-Skriptausführung (Playwright-Garantie: läuft vor jedem Skript der Seite)
einen zählenden Wrapper um `WebAssembly.instantiateStreaming`, der an die echte
Implementierung delegiert (Verhalten unverändert) und die Aufrufzahl in
`window.__ntStreamingCalls` mitzählt; danach navigiert der Test erneut (`page.goto`), damit
der Wrapper für den folgenden `import()`-Lauf aktiv ist. Neue Assertion:
`streamingCallCount >= 1` (nicht `=== 1` — die Nicht-Vakuität ist der Punkt, kein exakter
Call-Count-Pin).

**Catchability-Nachweis** (dieselbe Scratch-Kopie/Disziplin wie F1; Haupt-Tree unangetastet):

1. `spike/src/wasm/loader.ts:158` mutiert: `if (!isNode && typeof
   WebAssembly.instantiateStreaming === "function")` → `if (false && !isNode && typeof
   WebAssembly.instantiateStreaming === "function")`.
2. Wrapper (`pnpm test:browser`, re-emittiert frisch — hier geht es NICHT um Staleness, F1s
   Guard ist also trivial grün) → **3/4 grün, 1 ROT**: exakt Test 3, mit `Error:
   expect(received).toBeGreaterThanOrEqual(expected) … Expected: >= 1 Received: 0` an genau
   der neuen `streamingCallCount`-Zeile — die beiden VORHANDENEN Assertions (Existenz, MIME)
   blieben unverändert GRÜN in demselben Lauf, was B's ursprünglichen Befund exakt
   bestätigt (sie allein reichen nicht, um die Mutation zu fangen).
3. Mutation revertiert (Backup-Diff bestätigt identisch) → Wrapper erneut → 4/4 grün.

### F3 (MINOR, Baustein B, bereits offengelegt) — `playwright.config.ts` typprüfen

**Befund:** Die Datei lag außerhalb jedes `tsc`-Korpus dieses Projekts (drei Legs: root,
stress, browser) — ein vorsätzlicher Typfehler dort blieb von `pnpm check` unbemerkt.

**Fix:** `spike/tests-browser/tsconfig.json`s `include` um `"../../playwright.config.ts"`
erweitert (relativ von `spike/tests-browser/` aus: zwei Verzeichnisebenen hoch zum
Repo-Root).

**Catchability-Nachweis** (Haupt-Tree direkt, sofort revertiert — die im Auftrag
vorgesehene Alternative für Test-/Config-Dateien statt Scratch-Kopie, da
`playwright.config.ts` kein `spike/src`-Produktcode ist; Inhaltsdiff/Grep als
Revert-Beweis):

1. Baseline: `pnpm check` grün (Exit 0).
2. `const F3_MUTATION_TYPE_ERROR: number = "not a number";` in `playwright.config.ts`
   eingefügt.
3. `pnpm check` → **Exit 1**: `playwright.config.ts(16,7): error TS2322: Type 'string' is
   not assignable to type 'number'.` — geworfen vom `tests-browser`-Leg. Kontrollprobe:
   root-only `tsc --noEmit` allein bleibt **blind** (Exit 0) — bestätigt exakt die im Befund
   behauptete Isolationslücke, jetzt geschlossen.
4. Mutation entfernt; Grep auf den Mutations-String liefert keinen Treffer mehr (Datei
   sauber) → `pnpm check` → Exit 0.

**Pin-Bewegung:** `check:diag:browser` bewegt sich von **1.540 @ 73** (vor F1–F3) auf
**2.142 @ 75** (danach) — Δ+602 Instanziierungen, Δ+2 Dateien (`emit-freshness.ts` neu +
`playwright.config.ts` neu im Korpus; siehe Pins-Abschnitt oben für die Zuordnung).

### F4 (Baustein A, bewusster Akzeptanz-Punkt, KEIN Fix)

`dataEqual` (die `Object.is`-pro-Element-Vergleichsfunktion, inline in `smoke.test.ts`s
Op-Matrix-Differential repliziert) ist selbst ungetestet — es gibt keinen eigenen Test, der
z. B. beweist, dass `dataEqual` +0/−0 korrekt unterscheidet oder NaN korrekt als gleiche
Werte-Klasse behandelt. **Akzeptiert, konsistent mit der bestehenden Projektpraxis:** die
Node-seitige Schwesterfunktion `assertDataBitIdentical`
(`spike/tests-runtime/assert-helpers.ts`), von der `dataEqual` das Vergleichs-Idiom
übernimmt, ist ebenfalls ungetestet — beide Comparators sind Test-Infrastruktur, keine
Produktlogik, und ihre Korrektheit wird indirekt durch jeden Test bewiesen, der sie auf
bekannt-gleiche/bekannt-verschiedene Daten anwendet (jeder heute grüne Differential-Test IST
ein Beleg, dass `dataEqual` bekannt-gleiche Daten als gleich erkennt; die
Mutationsbeweise oben — z. B. F1s `ones()`-Mutation — belegen zusätzlich, dass es
bekannt-verschiedene Daten als verschieden erkennt). Kein neuer FOLLOWUPS-Eintrag für diesen
Punkt (bewusst gleichbehandelt mit dem bestehenden Node-seitigen Präzedenzfall, nicht neu
aufgemacht).

## Post-Schließungs-Gates (frisch, nach F1–F3)

Alle folgenden Läufe fanden NACH den drei Schließungen statt, im Haupt-Tree, volle
Outputs/Exit-Codes geprüft (nicht nur die Kennzahl gegrept):

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm test:browser` (3× für Stabilität) | Lauf 1: 4/4 „passed (1.0s)"; Lauf 2: 4/4 „passed (689ms)"; Lauf 3: 4/4 „passed (677ms)" | 0 / 0 / 0 |
| `pnpm check` (Dreier-Verbund) | clean | 0 |
| `pnpm test:core` | **818/818** (unverändert ggü. der ursprünglichen V3-Messung) | 0 |
| `pnpm test:resident` | 4277 pass + 2 skip = 4279 (unverändert) | 0 |
| `cargo test --manifest-path crates/core/Cargo.toml` | 161/161 (unverändert) | 0 |
| `pnpm demo` | „TS, WASM v1, and WASM resident all agree on every showcase op" | 0 |
| Artefakt-Hash, INKREMENTELL (`pnpm build:wasm` → `shasum -a 256`) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — exakt der Pin | 0 |
| Artefakt-Hash, CLEAN REBUILD (`cargo clean` → `pnpm build:wasm` → `shasum -a 256`) | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — exakt der Pin, STARKE Form (löst das Honesty-Residuum unten auf) | 0 |
| `pnpm check:diag` | **175.712 @ 132** — unverändert ggü. der ursprünglichen V3-Messung (Erwartung erfüllt: F1–F3 fassen keine Root-Korpus-Datei an) | 0 |
| `pnpm check:diag:stress` | **103.882 @ 82** — unverändert | 0 |
| `pnpm check:diag:browser` | **2.142 @ 75** — NEU gemessen, s. Pin-Bewegung unter F3 oben | 0 |

## Geänderte/neue Dateien (git status vor der End-Stand-Messung)

```
 M .gitignore
 M package.json
 M pnpm-lock.yaml
 M spike/src/ambient.d.ts
 M spike/tests-runtime/test-scripts-guard.test.ts
 M tsconfig.json
?? docs/phase-d-vorarbeiten-v3-ergebnisse.md
?? playwright.config.ts
?? spike/tests-browser/   (server.ts, smoke.test.ts, tsconfig.json, tsconfig.emit.json,
   emit-freshness.ts — NEU, F1-Schließung)
```

`pnpm-lock.yaml`: nur `@playwright/test` + transitive `playwright`/`playwright-core`/
`fsevents` (optional, darwin) — 38 Zeilen, keine Überraschungen.

## Offene Punkte / FOLLOWUPS

- **Browser-Smoke-Test-Item** (aus der Spec, auszutragen): erledigt für Chromium/v0. WebKit/
  Firefox bleiben FOLLOWUPS (Owner-Scope-Entscheidung, nicht in dieser Scheibe).
- Kein Bundler-Repräsentativitätstest — bewusst (Non-Goal laut Spec; das ist Item 11).
- ~~`check:diag:browser`s +78-Haupt-Delta wurde NICHT per Ablation bisektiert~~ — **erledigt:**
  Baustein A hat die volle Bisektion nachgeholt (s. „Pins" oben, „Baustein-A-Befund 1"); das
  Delta ist bewiesener echter Instanziierungs-Mehraufwand der Guard-Erweiterung, keine
  Vermutung mehr.
- Playwright-Report-/Cache-Verzeichnisse sind gitignored, aber NICHT automatisch aufgeräumt
  von `test:browser` selbst (nur `.emit/` wird bei jedem Lauf frisch `rm -rf`t) — harmlos
  (gitignored), aber ein `test-results/`-Ordner sammelt sich lokal an; bewusst nicht
  "gefixt", da außerhalb des V3-Scopes (keine Spec-Anforderung).
- F4 (`dataEqual` selbst ungetestet) bleibt bewusst ohne eigenen FOLLOWUPS-Eintrag — s.
  „In-Slice-Schließungen" oben, akzeptierter Punkt, konsistent mit dem Node-seitigen
  Präzedenzfall.

## Honesty-Residuum

- Die Op-Matrix-Differential-Suite in `smoke.test.ts` ist eine SMOKE-Suite (feste,
  handverlesene kleine Fixtures), keine PRNG-Fuzzing-Suite wie `spike/tests-runtime/`s
  Differential-Tests — das ist bewusst (Non-Goal: keine Perf-/Bundler-Repräsentativität,
  Ziel ist der Lade-Pfad + eine reale Stichprobe pro Op, nicht erschöpfende Coverage; die
  erschöpfende Differential-Coverage existiert bereits Node-seitig).
- ~~Das `+78`-Haupt-Pin-Delta wird dem Kern-11-Order-Mechanismus zugeordnet, aber NICHT
  Bit-für-Bit durch eine eigene Ablation bewiesen~~ — **aufgelöst:** die Zuordnung war
  FALSCH (siehe „Pins" → „Baustein-A-Befund 1"); Baustein A hat bisektiert und den echten
  Ursprung (die Guard-Erweiterung selbst) bewiesen. Festgehalten hier als Beispiel dafür,
  dass eine plausibel klingende, aber unverifizierte Ferndiagnose in einem Ergebnisdoc
  falsch sein kann — genau wofür die Zwei-Verifier-Runde existiert.
- ~~Der Artefakt-Hash-Lauf war ein INKREMENTELLER `cargo build` … ein unabhängiger
  Clean-Rebuild-Beweis bleibt Aufgabe der nachgelagerten Verifier-Runde~~ — **aufgelöst:**
  diese Schließungsrunde hat einen `cargo clean` + Clean-Rebuild gefahren (s.
  „Post-Schließungs-Gates"), Hash byte-identisch zum Pin in der STARKEN Form.
- Die F1/F2-Mutationsbeweise wurden ausschließlich in einer rsync-Scratch-Kopie gefahren
  (nie im Haupt-Tree) und danach gelöscht — nicht als eigene, reproduzierbare Testdatei im
  Repo hinterlegt (wie in der Spec/im Auftrag vorgesehen: Mutationsbeweise an
  `spike/src`-Dateien sind Einmal-Nachweise, keine Dauer-Suite).
