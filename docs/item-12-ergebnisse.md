# Item 12 — Qualitäts-Portfolio + CI (Ergebnisse)

Stand: 2026-07-18. Bindende Spec: `docs/item-12-ci-spec.md`. Roadmap Phase D, Item 12.

**Ehrlichkeitsregel:** Jede Zahl unten stammt aus einem in dieser Runde tatsächlich
ausgeführten Kommando (Output + Exit-Code geprüft, nicht nur die gegrepte Kennzahl).
Die naive Erst-Implementierung war NICHT korrekt — die lokale Verifikation fing zwei
echte Bugs, die vor der Meldung „fertig" behoben wurden (§4).

## 1. Zusammenfassung

NumType bekommt eine GitHub-Actions-CI, die alle lokalen Gates maschinell reproduziert
(hartes Release-Gate: „CI reproduziert alle Gates inkl. Freeze-Hash"). Zusätzlich:
`bench:editor` wird von einem reinen Report zu einem echten Gate mit Exit-Code gehärtet
(D5), die Test-Timeouts (F6) werden geschlossen, und die S1-Covenant-Invariante bekommt
ein zweites, CI-taugliches (zero-dep) Gate neben dem lokalen `graph-a-lama`-Lint.

Prozess: bindende Spec → Baustein-0-Spec-Verify (fresh context) → Implementierung → volle
A+B+C-Verify-Runde. **Zum Zeitpunkt dieses Docs: Implementierung fertig, lokal vollständig
grün verifiziert; die A+B+C-Verify-Runde steht noch aus (§7 / Addendum).**

## 2. Was gebaut wurde (8 Komponenten)

| Datei | Zweck |
|---|---|
| `rust-toolchain.toml` | Pinnt rustc stable **1.95.0** + wasm-Target → reproduzierbarer Freeze-Hash lokal UND in CI (D1). Threads-Build unberührt (`rustup run nightly-2026-07-09`). |
| `.nvmrc` | Node **24** (D1). |
| `package.json` | `packageManager: pnpm@11.6.0`, Script `check:freeze`, `--test-timeout=120000` in test:core/resident/package/threaded (D6), S1-Guard in test:core-Liste registriert. Keine neue Dependency (Z1). |
| `scripts/check-freeze-hash.mjs` | Zero-dep Freeze-Gate: SHA-256 des `.wasm` gegen eine **Menge** plattform-gelabelter Pins; wholesale-replace-Disziplin (D4). |
| `spike/bench-dx/editor-latency.ts` | Additiv: Instantiation-Pins W1–W7 (exakt), Latenz-2x-Ceiling-Exit, `process.exitCode`; Kommentar-Fix W4→W4+W6 (D5, Baustein-0-F1). Messpfade unberührt. |
| `spike/src/ambient.d.ts` | Minimal um `readdirSync(recursive)` erweitert (für den S1-Guard; from-scratch-Shim-Disziplin, kein @types/node). |
| `spike/tests-runtime/s1-import-guard.test.ts` | Zero-dep CI-Gate, spiegelt `graph-a-lama.rules.json` `covenant-s1`; **Specifier-Auflösung Pflicht** (D7, Baustein-0-D-1: sonst strukturell vakuös). |
| `.github/workflows/ci.yml` | 8 parallele Jobs auf ubuntu-latest; Trigger `push:[main]` + `pull_request` (keine Doppelläufe, Baustein-0-D-2); minimale Toolchain je Job; nightly-Install im threaded-Job (D2/D3). |

## 3. Lokale Verifikation — alle Gates grün (grounded)

| Gate | Ergebnis | Exit |
|---|---|---|
| `pnpm check` (Verbund root+stress+browser) | grün (nach beiden Fixes) | 0 |
| `pnpm test:core` | **821** pass / 0 fail (S1-Guard läuft, test-scripts-guard grün, zero-dep-guard grün) | 0 |
| `pnpm test:resident` | **4280** (4278 pass / 0 fail / 2 skip) | 0 |
| `pnpm test:package` | **3** pass / 0 fail, `check-dist-emit: OK` | 0 |
| `pnpm test:browser` | **4 passed** | 0 |
| `pnpm test:threaded` | **69** pass / 0 fail (nightly-2026-07-09 build-std baute) | 0 |
| `cargo test` | **161** passed / 0 failed | 0 |
| `pnpm demo` | „TS, WASM v1, and WASM resident all agree" | 0 |
| `pnpm check:freeze` | Hash == Pin `0b9df4f1…519c7d` | 0 |
| `pnpm bench:editor` | „Hard CI gate: PASS" | **0** |
| `graph-a-lama query lint` | „keine Verstöße, 0 errors" | 0 |

**Freeze byte-identisch:** `pnpm build:wasm` → `shasum -a 256 spike/src/wasm/numtype_core.wasm`
= `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d` — identisch zum Pin
(M4 hält, kein Rust berührt).

**Instantiation-Pins (grounded gemessen, macos-arm64 / tsc 7.0.2):** w1=24305, w2=26114,
w3=57254, w4=24466, w5=29759, w6=30929, w7=23477. Same-platform deterministisch
(Baustein-0: 3×/2× byte-identisch).

## 4. Von der Verifikation gefundene + behobene Bugs (Ehrlichkeitsregel)

1. **S1-Guard nutzte undeklarierte node-APIs.** Der erste Wurf importierte `readdirSync`,
   `relative`, `sep`, `node:assert/strict` — die from-scratch node-Shim `spike/src/ambient.d.ts`
   (kein `@types/node`, zero-dep-Ethos) deklariert nur genutzte APIs, diese fehlten →
   `pnpm check` rot (TS2305/TS2591/TS2345). Fix: Guard neu geschrieben (nur deklarierte APIs
   + eigene repo-relative Prefix-Strip-Berechnung statt `relative`/`sep`, `assert.ok` statt
   `assert/strict`), Shim minimal um `readdirSync(recursive)` erweitert.
2. **`process.exitCode` scheiterte an `process: unknown`.** Die Shim typisiert `process`
   bewusst als `unknown` (feature-detect only) → `process.exitCode = 1` = TS18046. Fix:
   lokaler Cast `(process as { exitCode?: number })`, die Shim-Aussage unberührt.

## 5. Nicht-Vakuität der neuen Gates (Pflicht-Mutanten, alle rot)

Methode: cp-Backup in den Scratchpad (NICHT `git checkout` — die Änderungen sind
uncommitted), mutieren, testen, zurückkopieren; `git diff --stat` am Ende sauber.

| Mutant | Ergebnis | Exit |
|---|---|---|
| (a) Latenz-Ceiling drastisch senken | M2-Hover „median 0.06ms > 0.00001ms exceeds 2x ceiling" | 1 |
| (b) Instantiation-Pin w1 verstellen (24305→24306) | „instantiations = 24305, pin = 24306 (delta -1)" | 1 |
| (c) verbotener Import in `spike/src/ndarray.ts` | „ndarray.ts:1 imports '../tests-runtime/mutant' **-> spike/tests-runtime/mutant**" (Auflösung greift, D-1-Fix wirkt) | 1 |
| (d) falscher Freeze-Pin | „check-freeze-hash: FAIL … matches NONE" | 1 |
| (e) hängender Test mit `--test-timeout=2000` (D6-Nicht-Vakuität, Spec §6) | „test timed out after 2000ms" | 1 |

**Zusätzliche adversariale Selbst-Checks** (Baustein-B-Angriffsflächen, selbst gefahren
während der Agent-Dispatch gestört war — kein Ersatz für den fresh-context-Verifier, aber
Zusatz-Coverage): M3-Toggle-only-Ceiling feuert auf **W4 UND W6** (M2 unberührt); S1-Guard
fängt `export {x} from` + tief-verschachtelte `../../`-Importe + ist comment-aware;
check:freeze robust bei fehlendem wasm; Instantiation-Pins über 2 Läufe byte-identisch;
`pnpm install --frozen-lockfile` grün mit dem neuen `packageManager`-Feld.

## 6. Geklärte Befunde

- **W1 „type errors by design" (alle 7 Workloads, nicht nur W4/W6 wie Baustein 0 annahm):**
  Es sind `node:`-Modulauflösungsfehler (`process`, `node:fs/promises`, `node:os` in
  `spike/src/wasm/{backend-api,loader,threaded}.ts`) in der isolierten workload-tsconfig
  ohne node-Typen — orthogonal zur Instantiation-Zahl (die volle src-Maschinerie wird
  geladen: 79 Files, ~24k–57k Instantiations). Der Harness fängt sie am `hadTypeErrors`-Pfad.

## 7. Offen

- **Erster echter CI-Lauf** (Definition-of-Done): klärt empirisch, ob der Linux-Freeze-Hash
  vom macOS-Pin abweicht (D4 erwartet das als actionable, nicht als kaputtes CI → Linux-Hash
  als zweiten Plattform-Pin eintragen) und ob die Instantiation-Pins plattform-stabil sind.
- **nightly-Version-Sync (Verify-B MINOR-5, FOLLOWUP):** `ci.yml` und
  `scripts/build-wasm-threads.sh` halten die pinned nightly (`nightly-2026-07-09`) nur per
  Kommentar synchron; ein künftiger Drift würde erst im CI-threaded-Job auffallen. Kein
  aktueller Bug (beide stimmen überein), reines Wartungsrisiko.
- **`cargo`-Job installiert den wasm32-Target mit (Verify-A/B, akzeptierter Nit):**
  `rust-toolchain.toml`s `targets` gilt toolchain-weit, `rustup show` installiert ihn auch im
  reinen `cargo`-Job (der ihn nicht braucht) — nur Download-Zeit, kein Korrektheitsproblem,
  nah am schon akzeptierten D-5-Nit (Spec §9).
- **`.nvmrc`-nvm-Warnung (kosmetisch, lokal):** `24` löst lokal eine harmlose nvm-Warnung
  aus (kein `24`-Alias installiert); in CI korrekt (setup-node löst `24` → neueste 24.x).

## 8. Post-Verification-Addendum

Volle A+B+C-Verify-Runde durchgeführt 2026-07-18 (drei fresh-context Agenten parallel; der
Agent-Dispatch war zuvor länger durch eine Anthropic-seitige Classifier-Störung blockiert,
lief dann durch). **Die fresh-context-Runde hat ihren Wert bewiesen: Verifier B fand zwei
konkret exploitierbare S1-Guard-Bypässe, die die lokale Selbst-Verifikation VERPASST hatte
(geerbter blinder Fleck) — genau der Grund für das Zwei-Verifier-Mandat.**

**Verdikte:**
- **A (Spec-Konformität): CONFORM.** Alle D1–D7 konform; alle Gates frisch grün mit echten
  Exit-Codes (exakt matchend zu §3, inkl. Freeze-Hash + 7 Instantiation-Pins); eigener Mutant
  in 3 Varianten rot; git status sauber. Befunde: (A-1, minor) §5-Tabelle listete den
  Timeout-Mutant nicht → **ergänzt** (Zeile (e)); (A-2, cosmetic) cargo-Job installiert
  wasm-Target mit → **dokumentiert** (§7, akzeptierter Nit).
- **B (adversarial): Löcher gefunden (D7), behoben.** D1–D6 halten unter Beschuss (Toggle-
  W4+W6, Freeze, Timeout, `rustup show` installiert Compiler+Target empirisch bestätigt,
  frozen-lockfile, setup-node „24", pnpm/action-setup). **D7 hielt nicht:** zwei mehrzeilige
  Import-Bypässe (MAJOR) + ein same-line-Blockkommentar-False-Positive (MEDIUM).
- **C (covenant): kein Verstoß.** M4/Z1/Z2/S1 gehalten; `graph-a-lama lint` selbst ausgeführt
  (0 Findings), Regel↔Spec-Verknüpfung `covenant-s1 ↔ S1` bestätigt; kein Nicht-Ziel; keine
  neue Dependency; `ambient.d.ts`-Erweiterung importfreie Typdeklaration, Z1-konform.

**In-Slice-Schließungen der B-Befunde (verifiziert):**
- **MAJOR-1/-2 (mehrzeilige Import-Bypässe):** Der S1-Guard wurde von zeilenweise auf einen
  **string-aware Text-Scanner** umgebaut (`stripComments`-Zustandsmaschine ersetzt Kommentare
  layout-erhaltend durch Blanks, behält String-Literale; die Specifier-Regexes laufen text-
  weit, `\s` matcht Newlines). Verifiziert: der mehrzeilige dynamische **und** statische
  verbotene Import werden jetzt gefangen (je exit 1, „ndarray.ts:1 imports '../tests-runtime/
  bypassN' -> spike/tests-runtime/bypassN"). Neuer Selbst-Test pinnt beide Formen + zwei
  Kommentar-Fälle (Nicht-Vakuität) — test:core jetzt **822** pass / 0 fail.
- **MEDIUM-3 (same-line-Blockkommentar-False-Positive):** dieselbe string-aware Strip-Logik
  behandelt `/* … */` auf einer Zeile korrekt → verifiziert grün (kein false positive).
- **MINOR-4 (Stale-JSDoc-Kommentar):** die zweite „W4 only"-Stelle im Datei-Kopf von
  editor-latency.ts ist auf „W4 and W6" korrigiert.
- **MINOR-6 (threaded-Timeout):** `test-threaded` job `timeout-minutes` 30 → 45 (Headroom für
  den kalten build-std-Erstlauf).
- **MINOR-5 (nightly-Sync):** als FOLLOWUP dokumentiert (§7), Wartungsrisiko ohne aktuellen Bug.

**Nachweis nach den Fixes (grounded):** `pnpm check` grün · `test:core` 822/0 · B-Bypässe 1/2
jetzt rot, MEDIUM-3 grün · `bench:editor` exit 0 · Freeze-Hash unverändert `0b9df4f1…519c7d`
(kein Rust berührt) · `graph-a-lama lint` 0 Verstöße · git status sauber (3 modified + 7
untracked, keine Reste). Die Guard-Fixes berühren nur `s1-import-guard.test.ts` (test:core),
`editor-latency.ts` (nur Kommentar) und `ci.yml` (nur Job-Timeout) — die übrigen Gate-Korpora
(resident/package/browser/threaded/cargo/demo) sind davon unberührt und bleiben wie in §3.

**Gesamt: Item 12 ist verifiziert.** Alle drei Verdikte adressiert; die einzigen
substanziellen Befunde (D7-Bypässe) sind behoben und der Fix gegen die exakten Verifier-B-
Mutanten gegengeprüft.
