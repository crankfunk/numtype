# Item 10 — Backend-Wahl-API (Binding Spec)

*Phase C, Item 10. Datum: 2026-07-12. Baut auf der Phase-C-Threads-Entscheidung
(docs/phase-c-threads-scoping.md) auf.*

> **Status: VERIFIZIERT + ANGEPASST, BEREIT FÜR IMPLEMENTIERUNG (2026-07-12).**
> Owner-Richtung (Option B + Fassade ii + vier Detailpunkte) abgenommen. Die
> adversariale Spec-Verifikation (frischer `brainroute:deep`, Owner-mandatiert, VOR
> der Implementierung) fand DREI Blocker; alle wurden am Code bestätigt und
> eingearbeitet — siehe „Adversariale Spec-Verifikation (Addendum)". Diese Runde
> (Spec-Verify VOR Impl) ist seit 2026-07-12 verbindliche Work Ethic (CLAUDE.md +
> verify-runde-template.md „Baustein 0").

## Ausgangslage

Das heutige **öffentliche Surface** (`spike/src/index.ts`) exportiert
ausschließlich `NDArray` (reine JS-Runtime) + die Typ-Ebene (`Shape`, `Broadcast`,
`MatMul`, `ReduceAxis`, `NDArrayView`, …). **Nicht** exportiert und heute rein
intern: `WNDArray` (WASM-resident, `spike/src/wasm/resident.ts`), `initCore`
(`loader.ts`), `initThreadedCore`/`ThreadedPool`/`threadedMatmul`
(`threaded.ts`), die v1-`backend.ts` (`spike/src/wasm/backend.ts`).

Die zwei Klassen-Surfaces sind Typ-Ebenen-Spiegelungen (identische
`Guard`/`OkShape`/`Broadcast`/`MatMul`/`ReduceAxis`), unterscheiden sich aber in
Runtime **und** Ergonomie:

| | `NDArray` | `WNDArray` |
|---|---|---|
| Rechnet in | reinem JS (`elementwiseBinary`, JS-Closures) | WASM (zero-copy resident) |
| Erzeugung | `fromArray(shape, values)` | `fromArray(**core**, shape, values)` |
| Init | keine | `core` via `await initCore()` (async) |
| Speicher | GC | `dispose()` (manuell) |
| Läuft | überall, synchron | überall (WASM); Threads-`core` Node-only |

(Ausnahme von der „exakten Spiegelung": `strides` ist bei `NDArray` eine Methode,
bei `WNDArray` ein Feld — s. Addendum/Blocker 1.)

`ThreadedPool` (`threaded.ts`) kapselt einen `core: ThreadedCoreExports` (⊂
`CoreExports`); `WNDArray` läuft über `pool.core`, der parallele Matmul ist die
freie Funktion `threadedMatmul(pool, a, b, opts?)` mit der bereits gemessenen
größenbasierten Auto-Weiche (`THREADED_MATMUL_MIN_POOL_WORK = 262_144`).

## Owner-Entscheidungen (Richtung, vor dieser Spec getroffen)

1. **Option B — explizite Backends** (2026-07-12): `NDArray` (JS) bleibt der
   Default und trägt den USP überall; WASM ist ein **explizites, opt-in
   Performance-Backend**; Threads sind ein **env-detektiertes Opt-in** darauf.
   *Nicht* automatische Backend-Wahl (A) — die WASM-Asymmetrie (async-Init +
   `dispose`) ließe sich nur durch Leak-Risiko (GC-Finalization für WASM-Speicher)
   oder Opferung der synchronen API verstecken, für einen Default, der den USP
   ohnehin nicht braucht (die Typ-Ebene ist backend-agnostisch).
2. **Fassade (ii)** (2026-07-12): ein `Backend`-Objekt kapselt den `core` und
   heilt die Ergonomie-Wunde (`core`-Parameter an jeder Erzeugung); Threads unter
   demselben Backend-Begriff statt einer Sonder-API.

## Kern-Einsicht (warum das überschaubar und risikoarm ist)

Alle Runtime-Bausteine existieren und sind differentiell verifiziert (Kern 01–11).
Item 10 baut **keine neue Runtime, keinen Kernel, keine Rust-Änderung** — es ist
eine reine TS-Fassade über bestehende, bit-identisch bewiesene Klassen. Damit:

> **Null Rust-Änderungen → Artefakt-Hash byte-identisch** (starke Freeze-Form, wie
> Kern 08/09; Pin `0b9df4f1…`). Der JS-Default (`NDArray`) bleibt unverändert.

## Binding-Entscheidungen

### D1 — Fassade-Form: `NDArray.backend(kind, opts?)`

```ts
type BackendKind = "wasm" | "threaded";

// statische Overloads auf NDArray — ein entdeckbarer Einstiegspunkt:
static backend(kind: "wasm"): Promise<WasmBackend>;
static backend(kind: "threaded", opts?: ThreadedBackendOptions): Promise<ThreadedBackend>;
```

- **async**, weil die WASM-Instanziierung (`initCore`/`initThreadedCore`) async
  ist. Der JS-Default-Pfad (`NDArray.fromArray`) bleibt **synchron** — nur wer
  ein WASM-Backend anfordert, tritt in die async-Welt ein.
- **`static` auf `NDArray`** statt freier `createBackend`-Funktion: ein
  entdeckbarer Namespace-Anker; präzise Rückgabe je `kind` via Overload
  (Overload-Auflösung empirisch bestätigt, s. Addendum).
- **Backend-Objekt** exponiert die Erzeugung: `fromArray<const S>(shape, values)`,
  `zeros<const S>(shape)`, `ones<const S>(shape)` → geben `WNDArray<S>` zurück
  (die **bestehende** Klasse, unverändert; das Backend kapselt den `core` und
  reicht ihn an die vorhandenen `WNDArray.fromArray(core, …)`-Statics durch).
- **`dispose()` bleibt sichtbar** an den erzeugten `WNDArray`-Instanzen — die
  Fassade versteckt WASM-Speicherverwaltung nicht (ehrlicher Perf-Preis).
  Zusätzlich `backend.dispose()` gibt den `core`/Pool frei.

### D2 — Threads-Integration: Pool-gekapselt, paralleler Matmul als Backend-Methode

`ThreadedBackend` kapselt einen `ThreadedPool`; die erzeugten Arrays sind **normale
`WNDArray`** über `pool.core` (ihre Instanzmethoden — `add`/`sum`/… — laufen wie
heute single-threaded auf dem Main-Thread). Der **parallele** Matmul wird als
**Backend-Methode** angeboten:

```ts
class ThreadedBackend {
  matmul<S, B>(a: WNDArray<S>, b: Guard<MatMul<S,B>, WNDArray<B>>, opts?): WNDArray<OkShape<MatMul<S,B>>>;
  // delegiert an threadedMatmul(this.pool, a, b, opts) inkl. Auto-Weiche
}
```

- **Begründung** (Kern-06-Präzedenz): „a `WNDArray`'s `core` reference cannot
  identify *its* pool" — eine `WNDArray`-Instanz kennt ihren Pool nicht, deshalb
  ist `threadedMatmul` eine freie Funktion mit explizitem `pool`. Der parallele
  Pfad an der **Instanz** (`a.matmul(b)`) aufzuhängen, erzwänge einen
  `WNDArray.matmul`-Umbau (frozen-adjacent, Kern 04/06) oder eine pool-aware
  Subklasse — beides teurer und riskanter als eine Backend-Methode.
- **Bewusst offengelegte Inkonsistenz (Owner-abgenommen):** paralleler Matmul via
  `backend.matmul(a, b)`, nicht `a.matmul(b)`. Vertretbar, weil **nur Matmul**
  parallelisiert ist (die Vektor-Ops des Haupt-Use-Case haben keinen Threads-
  Kernel) — dokumentierter Preis, kein versteckter.
- **Cross-Backend-Operanden:** `backend.matmul(a, b)` verlangt, dass `a`, `b` über
  DENSELBEN Pool/`core` liegen. Der bestehende `WNDArray`-Guard `assertSameCore`
  (resident.ts) fängt fremd-core-Operanden bereits — der Testplan deckt das für die
  neue Fassade explizit ab (Addendum: Mechanismus existiert, war nur ungetestet).
- **env-Detektion + dynamischer Import (Verify-Blocker 2):** `threaded.ts`
  importiert `node:os`/`node:fs/promises`/`node:worker_threads` **statisch** am
  Top-Level (Z. 82–84). Ein statischer Import in `ndarray.ts` würde den
  browser-sicheren `NDArray`-Default kontaminieren. Deshalb lädt
  `backend("threaded")` die Threads-Schicht **dynamisch**
  (`await import("./wasm/threaded.ts")`) **hinter** der env-Prüfung (Muster wie
  `loader.ts`' `await import("node:fs/promises")`); der `NDArray`-Default und
  `backend("wasm")` berühren `threaded.ts` **nie**. Die env-Prüfung
  (`typeof process` / Node + Artefakt-Erreichbarkeit) läuft **vor** dem dynamischen
  Import, damit der gepinnte Fehlertext erreichbar bleibt (statt an einem
  knallenden node-Import zu scheitern). Fehlt Node/Artefakt → Wurf mit dem
  gepinnten Stem (Detailentscheidung 4), **kein stiller Fallback**.

### D3 — `NDArrayView`-Konformität: AUS ITEM 10 HERAUS (Verify-Blocker 1)

Die adversariale Verifikation kippte die ursprüngliche D3: `WNDArray.strides` ist
bereits ein öffentliches `readonly`-**FELD** (`resident.ts:273`), keine Methode;
`NDArrayView` verlangt `strides(): number[]` (**Methode**, `ndarray.ts:128`), und
`NDArray` hat `strides()` als Methode (`ndarray.ts:353`). `WNDArray implements
NDArrayView<S>` kollidiert also (TS2300/TS2416, dreifach reproduziert), und der
naheliegende Fix (Feld→Methode bzw. umbenennen) bricht 8 Assertions in 3
bestehenden grünen Testdateien. Das deckt eine **vorbestehende
`strides`-Signatur-Inkonsistenz** zwischen `NDArray` (Methode) und `WNDArray`
(Feld) auf, deren Harmonisierung eine **eigene** Design-Entscheidung mit
Test-Fallout ist.

**Entscheidung (Owner, 2026-07-12):** Die `NDArrayView`-Konformität der
WASM/Threads-Backends wird **aus Item 10 herausgenommen** und bleibt der bereits
offene **Spike-05-Followup** (FOLLOWUPS, „NDArrayView auf WNDArray/threaded
ausdehnen") — jetzt mit dem konkreten `strides`-Blocker + der
Signatur-Harmonisierungs-Frage dokumentiert. Item 10 liefert die
**Backend-Fassade** (D1/D2/D4/D5/D6); die einheitliche kovariante Read-Abstraktion
über alle Backends ist nicht kritisch für die Backend-Wahl und folgt separat.

### D4 — Interop zwischen Backends (minimal, via vorhandene Member)

- `WNDArray → NDArray`: `WNDArray.toArray()` (existiert; liefert die logische
  row-major-Kopie) → `NDArray.fromArray(shape, float64)`.
- `NDArray → WasmBackend`: `NDArray.data` (public `readonly`) → `backend.fromArray(shape,
  nd.data)`. **Verify-bestätigt sicher:** `NDArray` ist immer materialisiert/
  contiguous (`strides()` berechnet `computeStrides(shape)` neu, `transpose`/`slice`
  materialisieren), also ist `.data` stets die logische row-major-Sicht — **kein
  stiller Interop-Datenfehler**.
- Beide Wege existieren mit vorhandenen Membern → **keine neuen Interop-Member
  zwingend**. Eine Bequemlichkeits-`Backend.from(view)` ist **FOLLOWUP**
  (Detailentscheidung 3), nicht v0.

### D5 — v1-`backend.ts`: bleibt intern und frozen

Wie Kern-09 D4: `backend.ts` ist **keine öffentliche Surface** — v1-Differential-
Twin/Baseline in Tests + `demo.ts`. Item 10 exponiert es **nicht** und fasst es
**nicht** an (frozen performance baseline). Die öffentlichen Backends sind
ausschließlich das JS-`NDArray` (Default) und die WASM/Threads-Fassade über
`WNDArray`.

### D6 — Freeze & Scope

- **Null Rust** → Artefakt-Hash `0b9df4f1…` byte-identisch (per Clean-Rebuild
  bestätigen).
- `index.ts`: neue Exports **anhängen** (Barrel). `ndarray.ts`: `static backend`
  als Insertion (private Konstruktor, kein EOF-Append). `resident.ts`:
  insertion-only (falls die Fassade dort Helfer braucht; die `WNDArray`-Klasse
  selbst bleibt in Item 10 unverändert — `NDArrayView` ist raus).
- **Scope-Grenze:** Item 10 = Backend-Wahl-API-Form + öffentliche Exposition der
  Fassade. **Nicht** in Scope: `NDArrayView`-Konformität der WASM/Threads-Backends
  (→ Spike-05-Followup, s. D3); finaler Paketschnitt (`exports`-Map,
  `.wasm`-Bundling, `d.ts`-Hover-Qualität, `slice-literal.ts`-Umbenennung) → Phase
  D / Item 11; `-1`-reshape, transzendente Ops, die vorbestehende Union-Guard-
  Latenz (eigene Scheiben, FOLLOWUPS).

## Testplan

### Typ-Ebene — `spike/tests/*.test-d.ts` (Insertion)
- `(await NDArray.backend("wasm")).fromArray([2,3], …)` ⇒ `WNDArray<[2,3]>`;
  `zeros([2,4])`/`ones(...)` korrekt getypt.
- `ThreadedBackend.matmul<S,B>` trägt denselben `Guard<MatMul<S,B>>`/`OkShape`
  wie `WNDArray.matmul` (Fehler am Argument bei Shape-Mismatch; ein
  `@ts-expect-error`-Fall).
- (`NDArrayView`-Konformität ist NICHT mehr Teil von Item 10 — s. D3.)

### Runtime-Differential — WASM-Backend: `spike/tests-runtime/backend-api.test.ts`
(in `package.json` **`test:resident`**-Liste eintragen — baut `build:wasm`; der
test-scripts-guard erzwingt die Listung):
- **Fassaden-Äquivalenz:** `backend("wasm").fromArray/zeros/ones` erzeugt Arrays,
  die (shape + `toArray()`-Bytes) **bit-identisch** zur direkten
  `WNDArray.fromArray(core, …)`-Erzeugung UND zur naiven `NDArray`-Referenz sind.
- **Lifecycle:** `backend.dispose()` gibt `core` frei (Free-Count-Plateau,
  nicht-vakuös); Nutzung eines disposed Backends wirft.
- **env-Detektion (negativ):** `backend("threaded")` wirft mit dem gepinnten Stem,
  wenn Node/Artefakt fehlt (mockbar — Muster wie `backend-oom.test.ts`); dieser
  Fall braucht das Threads-Artefakt NICHT und bleibt daher hier.
- **Interop:** `NDArray ↔ WNDArray` round-trip (`toArray`/`.data`/`fromArray`)
  shape + bytes bit-identisch; inkl. eines materialisierten `transpose`/`slice`
  (belegt D4 „immer row-major").
- **Cross-Backend-Guard:** zwei `WasmBackend`-Instanzen (getrennte `core`s); ein
  Array aus Backend A in eine Op mit einem Array aus Backend B → `assertSameCore`
  wirft mit erwarteter Meldung.

### Runtime-Differential — Threads-Backend: `spike/tests-runtime/backend-api-threaded.test.ts`
(in `package.json` **`test:threaded`**-Liste eintragen — baut `build:wasm:threads`;
**Verify-Blocker 3:** `test:resident` baut das Threads-Artefakt NICHT):
- **`ThreadedBackend.matmul`** bit-identisch zu `WNDArray.matmul` **und** naiver
  Referenz — über der Auto-Weichen-Schwelle und darunter (`opts.minPoolWork`
  `0`/`Infinity` explizit, wie `threaded.test.ts`, damit beide Routen geprüft
  werden).
- **Lifecycle:** `backend.dispose()` gibt Pool frei (disposed/poisoned-Semantik
  unverändert, Kern-06-Verträge); Cross-Pool-Operanden werfen.

### Bestehende Suiten: unverändert grün
`test:core` (817), `test:resident` (+Fälle), `test:threaded` (+Fälle), `pnpm demo`
(all-agree), `cargo` (161, unberührt), `check` (Verbund), Artefakt-Hash
**`0b9df4f1…` identisch** (Clean-Rebuild).

## Gate-Erwartung
- `pnpm check` (Verbund) clean.
- `check:diag`: Fassade-Overloads + Backend-Typen verschieben den Haupt-Pin
  (realistische Sites → Haupt-Korpus, nicht `tests-stress`). **Neu pinnen, Delta im
  Ergebnisdoc ausweisen** (kein Gate-Bruch; Affordability trivial < 4 % Budget).
  Achtung: die neue(n) Testdatei(en) tragen bis ~±2 000 Order-Rauschen
  (check-order-abhängiger Zähler, Kern-11-Pin) — bei Attributions-Bedarf
  empty-then-fill.
- Artefakt-Hash byte-identisch → Clean-Rebuild bestätigen.

## Abgenommene Detailentscheidungen (Owner, 2026-07-12)
1. **D2-Konsistenz:** paralleler Matmul via `backend.matmul(a, b)` — **abgenommen**.
2. **`ThreadedBackendOptions`:** `{ workers?, matmulTimeoutMs?, minPoolWork? }`
   an `initThreadedCore`/`threadedMatmul` durchgereicht — **abgenommen**.
3. **Interop-Kür `Backend.from(view)`:** **→ FOLLOWUP** (nicht v0). In `FOLLOWUPS.md`
   eingetragen.
4. **Fehler-Meldungs-Stem** der env-Detektion (D2), **gepinnt** — wortgleich
   Runtime ⇄ Test:
   `NDArray.backend("threaded"): threaded backend requires Node with the threads artifact (<reason>)`
   wobei `<reason>` die konkrete fehlende Bedingung nennt (z. B. `not running under
   Node` bzw. `threads artifact not found`).

## Adversariale Spec-Verifikation (Addendum, 2026-07-12)

Ein frischer `brainroute:deep`-Verifier prüfte diese Spec adversarial gegen den
echten Code (Owner-mandatiert, VOR der Implementierung; verify-runde-template.md
„Baustein 0"). Ergebnis: DREI Blocker, alle am Code (durch den Orchestrator)
bestätigt und eingearbeitet:

1. **`WNDArray.strides` ist ein Feld, keine Methode** (`resident.ts:273`); Kollision
   mit `NDArrayView.strides()` (Methode) → **D3 aus Item 10 heraus**, →
   Spike-05-Followup.
2. **`threaded.ts` importiert node-Module statisch** (Z. 82–84) → `backend("threaded")`
   lädt die Threads-Schicht **dynamisch hinter der env-Prüfung** (→ D2).
3. **`test:resident` baut nur `build:wasm`** (`package.json:16`); der
   ThreadedBackend-Test braucht `build:wasm:threads` → **eigene Datei in
   `test:threaded`** (→ Testplan).

**Bestätigt HÄLT:** D1-Overload-Mechanik (empirisch, Exit 0); D4-Interop (`NDArray`
immer materialisiert → `.data` logisch row-major, kein stiller Fehler); der
Cross-Backend-Guard (`assertSameCore` existiert; Testplan deckt ihn jetzt ab).

## Definition of Done
Spec (dies, verifiziert + angepasst) → Implementierung → **ZWEI Fresh-Context-
Verifier** (Spec-konform coverage-first + adversarial; Aufträge aus
`docs/verify-runde-template.md` Bausteine A + B) → `docs/item-10-backend-api-ergebnisse.md`
(grounded, Honesty-Rule, Post-Verify-Addendum) → **KB-Capture** (falls allgemeine
Lektion) → **Commit + Push** (Push separat bestätigen). Alle Befehle vom Repo-Root.
