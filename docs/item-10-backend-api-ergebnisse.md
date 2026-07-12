# Item 10 — Backend-Wahl-API (Ergebnisse)

Spec: `docs/item-10-backend-api-spec.md` (verifiziert + angepasst). Datum: 2026-07-12.
Alles unten ist in tatsächlich ausgeführten Kommandos dieser Session verankert; die
Honesty-Rule gilt durchgehend. Diese Scheibe war die erste unter der neuen Work-Ethic-
Regel **„Spec-Verifikation VOR der Implementierung"** (CLAUDE.md + verify-runde-template.md
Baustein 0) — die Verify-Historie (unten) belegt ihren Wert.

## Was gebaut wurde

Eine reine TS-Fassade über die bereits verifizierten Klassen — **null Rust, Artefakt-Hash
byte-identisch** (`0b9df4f1…`, per Clean-Rebuild von zwei unabhängigen Verifiern + dem
Orchestrator bestätigt). Owner-Richtung: **Option B** (JS-`NDArray` bleibt Default und trägt
den USP überall; WASM/Threads = explizites opt-in-Backend) + **Fassade (ii)**.

- **`spike/src/wasm/backend-api.ts`** (neu): `WasmBackend` (kapselt `CoreExports` aus
  `initCore()`, reicht an die bestehenden `WNDArray.*(core, …)`-Statics durch; `fromArray`/
  `zeros`/`ones`/`dispose`), `BackendKind`/`ThreadedBackendOptions`, und `checkThreadedEnv()`
  (D2 env-Detektion: Node + Artefakt-Erreichbarkeit, mit test-only `artifactUrl`-Override).
- **`spike/src/wasm/threaded.ts`** (angehängt): `ThreadedBackend` (kapselt `ThreadedPool`;
  `matmul(a,b,opts?)` → `threadedMatmul(this.pool, …)` inkl. Auto-Weiche; async `dispose`).
- **`spike/src/ndarray.ts`** (Insertion): `static backend(kind, opts?)`-Overloads (D1); der
  `"threaded"`-Zweig ruft `checkThreadedEnv()` **vor** `await import("./wasm/threaded.ts")`,
  sodass der browser-sichere Default-Pfad die statischen node-Imports von `threaded.ts` nie
  statisch zieht (D2).
- **`spike/src/index.ts`** (angehängt): `WasmBackend` (value), `BackendKind`/
  `ThreadedBackendOptions` (types), `ThreadedBackend` **type-only** (Wert bleibt hinter der
  dynamischen-Import-Grenze).
- **Tests:** `backend-api.test.ts` (→ `test:resident`), `backend-api-threaded.test.ts`
  (→ `test:threaded`, weil nur dort `build:wasm:threads` läuft), `backend-api.test-d.ts`
  (Overload-Auflösung, Typen). `package.json`-Listen aktualisiert (test-scripts-guard grün).

Interop (D4) nutzt vorhandene Member (`WNDArray.toArray()` ⇄ `NDArray.data`/`fromArray`) —
**verify-bestätigt sicher**, weil `NDArray` immer materialisiert/contiguous ist. D5:
v1-`backend.ts` unberührt. `NDArrayView`-Konformität (ex-D3) ist bewusst **nicht** in Scope
(→ Spike-05-Followup, s. u.).

## Offengelegte, owner-abgenommene Abweichung

`spike/tests-stress/tsconfig.json` erhielt `ambient.d.ts` in seinem `include`. Grund: D1
(`backend()` static auf `NDArray`) zieht node-berührenden Code (`backend-api.ts`/`loader.ts`
+ type-only `threaded.ts`) in jede `NDArray`-importierende Datei — der Stress-Korpus
erreicht diese Node-Ambient-Deklarationen jetzt erstmals. Ohne den include-Eintrag: `tsc -p
spike/tests-stress` Exit 1, 7 Fehler (Baustein A empirisch). Fix = eine include-Zeile (macht
Stress konsistent mit dem root-Korpus) + `assert.rejects` in `ambient.d.ts` (async-Zwilling
von `throws`, für den env-Throw-Test). Kein neuer Dependency, kein Produktcode. **Owner
abgenommen 2026-07-12** (D1 beibehalten; die Alternative — `backend()` als freie Funktion —
wäre ein API-Ergonomie-Downgrade für einen Budget-Vorteil, den wir nicht brauchen).

## Gates (echte Läufe)

| Gate | Ergebnis |
|---|---|
| `pnpm check` (Verbund root+stress) | clean, exit 0 |
| `pnpm test:core` | 817/817, exit 0 |
| `pnpm test:resident` | 4279 total (4277 pass, 2 skip); **+12** vs. Vor-Item-10-Baseline 4267 (4265+2)¹ — exit 0 |
| `pnpm test:threaded` | 69/69 (+4 Fassade, M6-Erweiterung im Dispose-Test), exit 0 |
| `pnpm demo` | „TS, WASM v1, WASM resident all agree", exit 0 |
| `cargo test` | 161/161, unberührt, exit 0 |
| Artefakt-Hash | `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — byte-identisch (Clean-Rebuild) |
| `pnpm check:diag` | **175.634 @ 132 Dateien** (neuer Pin) |
| `pnpm check:diag:stress` | **103.882 @ 82 Dateien** (neuer Pin) |

¹ Reporting-Korrektur (Baustein-A-Befund): der Executor meldete für die Impl allein „+13" —
die echte isolierte Baseline (3bce153) ist **4267** total (4265 pass + 2 skip). Die Impl
brachte +11 (die 11 `backend-api.test.ts`-Tests) auf 4278; die M3-Härtung +1 (der „D2
ordering"-Test) auf 4279 = **+12 gesamt**. Die „+13" war ein Off-by-2 gegen die alte pass-
statt total-Zahl. `fail 0`/`skip 2` in jedem Lauf selbst verifiziert.

## check:diag-Attribution (Baustein-B-Ablation, Korrektur)

Der `check:diag:stress`-Anstieg (94.597 → 103.882, +9.285) wurde per Ablation zerlegt (Baustein
B): der Struktureffekt der neuen Dateien ist **netto −13.600** (check-order-abhängiger Zähler,
Kern-11-Mechanismus), überdeckt von **+22.885 realen Kosten** — `threaded.ts`s Generics
(`ThreadedPool`, `threadedMatmul<S,B>`, `Guard<MatMul>`) werden für JEDE `NDArray`-importierende
Stress-Datei erstmals typgeprüft. Das ist eine **inhärente Folge von D1** (backend static auf
NDArray), kein Fehler. **Korrektur** ggü. der ursprünglichen Commit-Attribution: der dominante
Treiber ist `threaded.ts`s Generics, nicht der triviale `ambient.d.ts`-include. Beide Korpora
unter Budget (root 3,5 % < 4 %; Stress ungegattet) → kein Gate-Bruch.

## Verify-Historie (der Kern-Prozess dieser Scheibe)

**Baustein 0 — adversariale Spec-Verifikation VOR der Implementierung** (`brainroute:deep`,
Owner-mandatiert): fand DREI Blocker gegen den echten Code, alle vor dem Bau eingearbeitet:
(1) `WNDArray.strides` ist ein Feld, keine Methode → `NDArrayView`-Konformität (ex-D3) aus
Item 10 heraus; (2) `threaded.ts` importiert node-Module statisch → dynamischer Import hinter
env-Prüfung (D2); (3) `test:resident` baut nur `build:wasm` → ThreadedBackend-Test in eigene
`test:threaded`-Datei. **Ohne diese Runde wäre blind in Blocker (1) gebaut worden.**

**Baustein A — Spec-Verifier** (`brainroute:verify`): D1–D6 alle HALTEN, gegen den Diff
verankert; Freeze per echtem Clean-Rebuild bestätigt; Abweichung empirisch als notwendig+minimal
bewiesen; eigener Mutant nicht-vakuös gefangen. Ein Reporting-Befund (+11 statt +13, s. o.),
ein informational Follow-up (test:resident ohne Test-Timeout).

**Baustein B — adversarialer Verifier** (`brainroute:deep`): Browser-Sicherheit **empirisch**
per ESM-Loader-Trace bewiesen (Default/`backend("wasm")` laden `threaded.ts`/`worker_threads`
nie; erst `backend("threaded")` mit Artefakt, nach der env-Prüfung). Grenzfälle halten. Fand
ZWEI **major Test-Coverage-Lücken** (Code korrekt, Tests vakuös) — genau der Wert des
adversarialen Zweiten, den Baustein As eigener Mutant nicht traf:
- **M3**: die D2-Reihenfolge (env-Check vor dyn. Import) war nicht testgeschützt.
- **M6**: `ThreadedBackend.dispose()` fehlendes `await` wurde nicht gefangen (Test prüfte nur
  das synchron gesetzte `isDisposed`-Flag).

## Härtung nach der Verify-Runde (beide major-Lücken geschlossen, Non-Vakuität bewiesen)

- **M3** — neuer Test „D2 ordering" in `backend-api.test.ts`: prüft via `process.moduleLoadList`-
  Delta (um einen `process`-geblankten `backend("threaded")`-Aufruf), dass `threaded.ts`/
  `worker_threads` bei fehlgeschlagenem env-Check NICHT lädt. **Wichtiger Zusatzfund
  (Executor):** in der naiven Platzierung (nach dem message-stem-Test) war der Test SELBST
  vakuös — der vorherige `backend("threaded")`-Aufruf verschmutzt unter dem Mutanten das
  Modul-Cache (`before=1, after=1` → fälschlich grün). Fix: als ERSTER Test des env-Abschnitts
  platziert (im Code kommentiert). Non-Vakuität bewiesen: Mutant (Import vor env-Check) →
  `test:resident` rot (`1 !== 0`) → revert → grün.
- **M6** — `ThreadedBackend.dispose()`-Test um `getThreadedPoolFreeCount()`-Plateau erweitert
  (analog zum WASM-Dispose-Test; Delta empirisch `+1` für `workers:1`). Non-Vakuität: Mutant
  (`await` entfernt) → `test:threaded` rot (`4 !== 3`) → revert → grün.
- **Doku-Fixes:** `WasmBackend.dispose()`-JSDoc präzisiert (setzt nur ein Flag, kein
  `core`-Teardown); `ThreadedBackend`-JSDoc um Poisoned-Pool-Verhalten (matmul wirft,
  non-matmul-Ops bleiben nutzbar); `backend()`-JSDoc um die Literal-`kind`-Anforderung.

## Offene Punkte / FOLLOWUPS
- `test:resident` ohne harten Test-Timeout (Baustein-A-Befund): Item 10 ist der erste Pfad
  mit echtem Datei-I/O im Default-Testlauf; ein hängender env-Check würde CI hängen statt
  failen → FOLLOWUP.
- `NDArrayView`-Konformität der WASM/Threads-Backends bleibt der Spike-05-Followup (jetzt mit
  dem konkreten `strides`-Feld-vs-Methode-Blocker dokumentiert).
- `Backend.from(view)`-Interop-Kür → FOLLOWUP (nicht v0).
- M3-Test-Platzierung: künftige `backend("threaded")`-Tests in `backend-api.test.ts` müssen
  NACH dem „D2 ordering"-Test stehen (Modul-Cache-Verschmutzung im selben Prozess; im Code
  kommentiert).

## Honesty-Residuum
- Die Browser-Sicherheit ist für native ESM-Dynamic-Import-Semantik bewiesen, **nicht** für ein
  Bundler-Pipeline-Verhalten (kein Bundler-Test existiert — `.wasm`-Bundling ist Phase D/Item 11).
- Der M3-Regressionstest nutzt `process.moduleLoadList` (internes, aber lange stabiles Node-API);
  dokumentiert im Test.
