# Item 12 — Qualitäts-Portfolio + CI (bindende Spec)

Stand: 2026-07-17. Roadmap Phase D, Item 12. Nachfolger von Item 11 (HEAD 69ab47a).
Diese Spec ist bindend im Sinne des Covenant-Regimes (CLAUDE.md „Qualitätssicherung");
sie durchläuft nach der Owner-Richtungsabnahme **Baustein 0** (adversarialer
Spec-Verifier), dann Implementierung, dann die volle Verify-Runde A+B+C.

## 1. Ziel & Kontext

NumType hat seit Item 11 alle Gates lokal grün (`pnpm check`-Verbund, test:core/
resident/package/browser/threaded, cargo, demo, Freeze-Hash, bench:editor). Es fehlt
die **maschinelle Reproduktion dieser Gates in CI** — das ist ein hartes Release-Gate
(roadmap.md §Release-Gates: „CI reproduziert alle Gates inkl. Freeze-Hash"). Item 12
baut die GitHub-Actions-Pipeline, härtet `bench:editor` zu einem echten Gate (Exit-Code
statt nur Druck) und schließt die offenen Timeout-Followups (F6).

**Kein** Teil von Item 12 (Nicht-Ziele, s. §7): Vitest-Migration (Owner-Entscheidung
2026-07-17: bei `node --test` bleiben), Release/Publikation (Item 13), WebKit/Firefox-
Browser (bleibt Chromium-only), Coverage-Reports, Deploy.

## 2. Umgebungsfakten (verifiziert 2026-07-17, isolierter Kontrolllauf)

- Node **v24.16.0**, pnpm **11.6.0**, rustc/cargo **stable 1.95.0** (`59807616e 2026-04-14`).
- **Kein `rust-toolchain.toml`** — stable ist heute floating (ein `rustup update` würde
  den lokalen Freeze-Build verschieben). nightly-2026-07-09 lokal installiert (Threads).
- Remote `github.com/crankfunk/numtype`, **privat** (→ Actions-Minuten kosten; bei
  späterem Public-Release sind Standard-Runner-Minuten frei).
- WASM-Artefakt aktuell = Freeze-Pin `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`.
- `.gitignore` dokumentiert für `*.wasm`: **„not byte-stable across hosts"** — ein
  starkes Vor-Signal, dass ein Linux-Runner einen anderen Hash als der macOS-Pin
  produziert (prägt D4).
- **Freeze-Hash entsteht auf macos-arm64 / rustc 1.95.0.** `bench:editor`-Harness
  ([spike/bench-dx/editor-latency.ts](../spike/bench-dx/editor-latency.ts)) MISST +
  DRUCKT alles, setzt aber **keinen Exit-Code** bei Latenz-Überschreitung
  (`printGateVerdict` L652 druckt nur PASS/FAIL); nur die Correctness-Gates (Hover-Shape)
  werfen hart. **Es existieren keine assertierten Instantiation-Pins im Harness** — die
  Instantiation-Zahlen sind dokumentiert (Docs), aber nicht als Code-Konstanten verankert.
- `graph-a-lama` ist ein lokal cargo-installiertes Binary (`~/.cargo/bin/graph-a-lama`),
  **nicht trivial CI-installierbar**. S1-Regel (`graph-a-lama.rules.json`): forbidden
  import `^spike/src/` → `^spike/(tests|bench|demo)`.

## 3. Berührte COVENANT-Invarianten (v4)

Item 12 ist eine **reine Infrastruktur-Scheibe**: sie ändert **keinen** Runtime-Quellcode,
**keinen** Rust/Kernel, **keine** Typ-Ebene. Damit:

- **M4 (Frozen Baseline)** — *zentral berührt, aber nur REPRODUZIERT, nicht geändert.*
  Der Freeze-Check ist die CI-Reproduktion von M4. Da kein Rust angefasst wird, bleibt
  der lokale Artefakt-Hash byte-identisch (`0b9df4f1…`). Anker: keine Änderung an
  abi.rs/matmul_blocked.rs/shape.rs.
- **Z2 (`pnpm check` = Verbund aller Quelltext-Korpora)** — *reproduziert.* CI fährt den
  `pnpm check`-Dreier-Verbund + den build-gated `test:package`.
- **Z1 (Zero-Dependency-Runtime)** — *geschützt.* CI fügt keine Runtime-Deps hinzu
  (CI-YAML ist keine Paket-Dependency; neue Scripts sind zero-dep Node/`.mjs`). Der
  bestehende `zero-dep-guard` (in test:core) läuft in CI mit.
- **S1 (Runtime importiert nie aus Test/Bench/Demo)** — *berührt (geprüft).* S. D9.
- **M1/M2/M3/M5** — *nicht direkt berührt, aber ihre Gates werden reproduziert*
  (test:core ⇒ M1-Bit-Identität; check ⇒ M2/M3-Typ-Ebene; backend-api-Tests ⇒ M5).

Kein Nicht-Ziel wird implementiert. Kein Anker-Code (M4-Frozen-Dateien) wird verändert.

## 4. Binding-Entscheidungen

### D1 — Toolchain-Pinning (Owner-abgenommen 2026-07-17: rust-toolchain.toml)

Erstelle **`rust-toolchain.toml`** im Repo-Root:
```toml
[toolchain]
channel = "1.95.0"
targets = ["wasm32-unknown-unknown"]
```
Begründung: Der Freeze-Hash ist Teil des Produkt-Vertrags (M4). Ohne Pinning driftet der
lokale UND der CI-Build, sobald rustc-stable fortschreitet → der Freeze-Beweis würde
grundlos brechen. `1.95.0` (nicht `"stable"`) fixiert exakt die Version, mit der der Pin
`0b9df4f1…` entstand. **Freeze-Behauptung:** Der Owner baut lokal bereits mit 1.95.0
(verifiziert) → sein lokaler Hash bleibt `0b9df4f1…` unverändert; die Datei fixiert ihn nur
gegen künftige `rustup update`. Der Threads-Build bleibt unberührt: `build-wasm-threads.sh`
überschreibt die Toolchain explizit via `rustup run nightly-2026-07-09` — `rust-toolchain.toml`
setzt nur die *default*-Toolchain, die `rustup run` überstimmt.

Node/pnpm werden in CI gepinnt: Node **24** (`.nvmrc` = `24` + `actions/setup-node`),
pnpm über `corepack`/`pnpm/action-setup` an das `packageManager`-Feld (neu in package.json)
gebunden; alle Installs mit **`--frozen-lockfile`** (reproduziert exakt tsc 7.0.2 + Playwright).

### D2 — Runner

**`ubuntu-latest`** für ALLE Jobs (Owner-Präferenz; billig; bei späterem Public-Repo frei).
Der Freeze-Job akzeptiert bewusst, dass der Linux-Hash vom macOS-Pin abweichen kann (D4).

### D3 — Job-Graph & Trigger

Trigger (Owner-abgenommen 2026-07-17 nach Baustein-0-Befund D-2 — Doppellauf-Falle):
```yaml
on:
  push:
    branches: [main]
  pull_request:
```
`push` nur auf `main`, `pull_request` für alle Branches → keine Doppelläufe (ein In-Repo-PR-
Commit feuert sonst push+pull_request für denselben Commit; die per-ref-Concurrency
collabiert das nicht, verschiedene `github.ref`). main-Pushes und alle PRs (inkl. späterer
Fork-PRs beim OSS-Release) werden je genau einmal geprüft. `concurrency`-Gruppe pro
Workflow+Ref mit `cancel-in-progress`, um veraltete Läufe desselben Refs zu canceln. Jobs
laufen **parallel & unabhängig** (fail-independent), jeder mit minimaler Toolchain:

| Job | Kommando(s) | Toolchain | Anmerkung |
|-----|-------------|-----------|-----------|
| `check` | `pnpm check` (Dreier-Verbund) | node+tsc | kein Rust; schnellstes Gate |
| `cargo` | `cargo test --manifest-path crates/core/Cargo.toml` | rust-stable (native) | kein wasm-Target nötig |
| `test-node` | `pnpm test:core` + `test:resident` + `test:package` | rust-stable + wasm | baut wasm/dist |
| `test-browser` | `pnpm test:browser` | rust-stable + wasm + Playwright/Chromium | `playwright install --with-deps chromium` |
| `test-threaded` | `pnpm test:threaded` | **nightly-2026-07-09 + rust-src** + `-Z build-std` | teuerster Job; cache Cargo-Registry |
| `freeze` | `pnpm build:wasm` + `pnpm check:freeze` | rust-stable + wasm | D4 |
| `editor-gate` | `pnpm bench:editor` | node+tsc | D5 |
| `demo` | `pnpm demo` | rust-stable + wasm | billiges End-to-End-Gate (assert equal) |

Alle Jobs: `actions/checkout`, Toolchain-Setup, Caching (Cargo-Registry/-target + pnpm-Store)
wo es lohnt. `pnpm install --frozen-lockfile` nur in Jobs mit Node-Bedarf — **nicht** im
reinen `cargo`-Job (Baustein-0-Befund D-4: dort ist der Node-Toolchain-Install reine
Verschwendung). Alle Kommandos vom Repo-Root.

### D4 — Freeze-Hash-Check (Owner-abgenommen 2026-07-17: Verschmelzung 2⊕3)

Neues **`scripts/check-freeze-hash.mjs`** (zero-dep Node), das SHA-256 von
`spike/src/wasm/numtype_core.wasm` berechnet und gegen eine **Menge bekannter,
plattform-gelabelter Pins** prüft; grün, wenn der Hash *einer* davon ist, sonst
`exit 1` mit prominentem Ist-Hash. Startmenge:
```
macos-arm64 / rustc 1.95.0 : 0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d
```
Neues Script `check:freeze` in package.json. **Erste-Lauf-Empirie (dokumentiert, keine
Wette):** Der erste CI-Lauf auf ubuntu zeigt, ob Linux == macOS-Hash. (a) identisch →
`.gitignore`-Kommentar „not byte-stable across hosts" war konservativ, ein Hash genügt;
(b) abweichend (wahrscheinlich) → Linux-x64-Hash als zweiten gelabelten Pin eintragen
(Ein-Zeilen-Nachtrag). **M4-Schutz hält in beiden Fällen:** jede Änderung an den
eingefrorenen Kernen matcht KEINEN Pin. Das Script ist auch lokal (`pnpm check:freeze`,
macOS) grün, weil `0b9df4f1` in der Menge ist.

**Pin-Mengen-Disziplin (Baustein-0-Befund D-3):** die Menge wird bei jeder LEGITIMEN
Baseline-Änderung (künftige M4-berührende Scheibe) **wholesale ersetzt, nie appended** —
ein appendierter, veralteter Plattform-Pin könnte einen späteren versehentlichen Regress
auf einen alten Kernel-Zustand fälschlich durchwinken und M4 still untergraben. Die Menge
enthält nur die Hashes des AKTUELLEN Baseline-Zustands, je Plattform genau einen.

Der Ausgang des ersten CI-Laufs (der tatsächliche Linux-Hash) wird ins
Post-Verification-Addendum der Ergebnis-Doku geschrieben — nicht angenommen.

**KB-Beleg (coding-kb `reproducible-wasm-build-plattform-gepinnt`, evergreen):** `.wasm`
ist aus zwei anderen Projekten empirisch **nicht cross-host byte-stabil** (macOS-arm64 ≠
linux-x64), u.a. wegen eingebetteter `file:line`-Panic-Locations → Option 3 (hart gegen
den macOS-Pin auf ubuntu) würde mit hoher Sicherheit rot; die Menge-Lösung ist der
robuste Umgang. Nuance: die dort genannte Divergenzquelle **wasm-opt/binaryen entfällt bei
numtype** (`build:wasm` = reiner `cargo build`, kein Post-Processing), der erste Linux-Lauf
bleibt also empirisch offen. Das dort *bewährte* Muster ist ein **kanonischer Build auf
EINER Plattform** (Linux via Docker lokal, CI rebuildt nativ) — eine schwerere,
M4-Referenz-berührende Alternative (Docker-Pflicht lokal, Pin-Wechsel = COVENANT-Änderung).
Für Item 12 bleibt die minimal-invasive Menge-Lösung gewählt; die kanonische-Plattform-
Umstellung ist ein bewusster **FOLLOWUP** (keine Item-12-Blocker-Sache).

**KB-Beleg (coding-kb `hooks-erinnern-statt-erzwingen`):** mechanisch Prüfbares gehört als
CI-Script mit Exit-Code, **fail-closed** — trägt D5 (bench:editor-Exit) und D7 (S1-Guard).

### D5 — `bench:editor` als hartes Gate (Spike-02-Code additiv erweitern)

Owner-Wahl: **Correctness + Instantiation hart, Latenz mit 2x-Ceiling.** Erweitere
[editor-latency.ts](../spike/bench-dx/editor-latency.ts) **additiv** (keine bestehende
Logik ändern):

1. **Exit-Code.** `main()` sammelt einen Gate-Verdikt und setzt `process.exitCode = 1`
   bei Verletzung. Heute wirft nur ein Correctness-Fehler; der Latenz-Verdikt wird nur
   gedruckt. Neu: der Prozess endet non-zero, wenn irgendein Gate fällt.
2. **Latenz-Gate = 2x-Ceiling.** Verletzung, wenn ein M2-Hover-Median > `HOVER_GATE_MS*2`
   (200 ms) ODER ein M3-Toggle-Median > `TOGGLE_GATE_MS*2` (1000 ms). Der bestehende
   `gate2xPass` wird bereits berechnet — nur an den Exit-Code koppeln. (Das strenge
   1x-Gate bleibt informational im Druck; auf geteilter CI-Hardware ist nur das
   2x-Ceiling nicht-flaky, bei lokal ~3 Größenordnungen Luft.)
3. **Correctness-Gate.** Bleibt hart (wirft schon in `assertHoverCorrect`/M3-Toggle) —
   das ist der eigentliche USP-Beweis (jeder Hover zeigt die erwartete Shape).
4. **Instantiation-Pins (neu).** Eine committete Pin-Struktur (W1…W7 → erwartete
   `Instantiations`-Zahl), exakter Vergleich (die Zahlen sind laut Spike 02
   deterministisch, 4/4 Läufe identisch). Bei Abweichung: Ist-Wert prominent ausgeben +
   Gate-Verletzung. Die Pins werden im Zuge der Impl LOKAL gemessen und eingetragen.
   **Cross-Platform-Risiko** (analog D4, aber geringer — reine Typ-Logik ohne Codegen):
   falls der erste CI-Lauf abweicht, zeigt die Ist-Wert-Ausgabe es; Reaktion (Pin-Update
   oder plattform-Menge wie D4) im Addendum dokumentiert.

Kein Freeze-Anker berührt (editor-latency.ts ist Bench-Infra, kein Runtime/Rust). Der
Artefakt-Hash bleibt byte-identisch.

### D6 — Test-Timeouts (FOLLOWUPS F6)

Ergänze in den package.json-Testscripts ein hartes **`--test-timeout`** für alle
`node --test`-Läufe (test:core, test:resident, test:package, test:threaded — nicht
test:browser, das über Playwrights eigene Timeouts läuft). Wert: **`--test-timeout=120000`**
(120 s pro Test — großzügig genug für Build+viele Fälle, fängt aber echte Hänger wie den
`checkThreadedEnv`-file-I/O-ohne-Timeout-Pfad, F6). Rein additiv; Verhalten für grüne
Tests unverändert.

### D7 — S1-covenant-Lint in CI (Owner-abgenommen 2026-07-17: ja)

Ein kleiner **zero-dep S1-Import-Guard** als CI-tauglicher Test, der die
`graph-a-lama.rules.json`-Regel `covenant-s1` spiegelt: er scannt `spike/src/**/*.ts` nach
`from"…"`/`import(…)`/`import"…"`-Specifiern, **löst jeden relativen Specifier gegen das
Verzeichnis der importierenden Datei zu einem repo-relativen Pfad auf** und prüft den
AUFGELÖSTEN Pfad gegen `^spike/(tests|bench|demo)` (`exit 1` bei Treffer, benennt
Datei:Zeile). **Auflösung ist Pflicht (Baustein-0-Befund D-1):** alle src-Imports sind
relativ (`./`/`../`), ein roher Specifier beginnt NIE mit `spike/` — ein Match gegen den
rohen String wäre strukturell vakuös (grün für immer, egal was importiert wird), genau die
falsche Sicherheit, die die Nicht-Vakuitäts-Disziplin verhindern soll. Der Scan muss auch
`import type … from`, `export … from`, mehrzeilige und dynamische `import()`-Formen
erfassen (die `covenant-s1`-Regel gilt für JEDE Import-Kante). Die `rules.json` bleibt die
**kanonische** Quelle; der Guard verweist darauf. Nicht-Vakuität per Mutant (verbotener
Import → rot). Der `graph-a-lama query lint` bleibt zusätzlich das **lokale** Gate (das
externe Binary ist nicht CI-installierbar).

## 5. Datei-Disziplin

- **Neu:** `.github/workflows/ci.yml`, `rust-toolchain.toml`, `.nvmrc`,
  `scripts/check-freeze-hash.mjs`, (D7) `spike/tests-runtime/s1-import-guard.test.ts` +
  Registrierung, ggf. `spike/bench-dx/editor-pins.json` (oder Inline-Konstante).
- **Additiv erweitert:** `spike/bench-dx/editor-latency.ts` (nur Anhang/Exit-Logik, keine
  Änderung bestehender Messpfade), `package.json` (Scripts: `check:freeze`, Timeout-Flags,
  `packageManager`-Feld; devDeps unverändert außer ggf. nichts Neues).
- **Unberührt (Freeze):** jeglicher Rust-Code, `spike/src/**` Runtime, alle
  M4-Anker-Dateien. Artefakt-Hash bleibt byte-identisch — **zu beweisen** durch
  `pnpm build:wasm` + `check:freeze` in der Verify-Runde.
- test-scripts-guard: neue Testdateien (D7) müssen global eindeutig benannt und in genau
  einer Liste registriert sein (Item-11-S3-F4-Lektion: Basename-Kollision vermeiden).

## 6. Test- & Verifikationsplan

**Lokal prüfbar (vor Commit):**
- Alle Gates frisch grün: `pnpm check` (Verbund), test:core/resident/package/browser/
  threaded, cargo, demo, `pnpm check:freeze`, `pnpm bench:editor` (jetzt mit Exit-Code).
- Freeze byte-identisch: `pnpm build:wasm` → `check:freeze` grün (Hash == `0b9df4f1…`).
- **Nicht-Vakuität bench:editor-Gate** (Pflicht-Mutanten): (i) Latenz-Ceiling künstlich
  herabsetzen → Gate rot / exit 1, **je gegen einen M2-Hover UND einen M3-Toggle-Fall**
  (Baustein-0-Befund F1: es gibt ZWEI toggle-Workloads, W4 UND W6 — der Code-Kommentar
  editor-latency.ts:697 „M3 on W4 only" ist seit Kern 08 veraltet; die additive Exit-Logik
  deckt beide Toggle-Workloads ab, der veraltete Kommentar wird in-slice korrigiert, reine
  Kommentar-Änderung ohne Logik-Berührung); (ii) einen Instantiation-Pin verstellen → rot
  mit Ist-Wert; (iii) Correctness-Wurf existiert bereits und terminiert HEUTE schon mit
  exit 1 (Baustein-0-Befund F5). Alle revertiert, `git status`-Beweis.
- **Nicht-Vakuität Test-Timeout:** ein künstlich hängender Test → Timeout greift, exit≠0.
- **Nicht-Vakuität S1-Guard (D7):** verbotener Import in eine spike/src-Datei → Guard rot.
- **Nicht-Vakuität Freeze-Menge:** ein-Byte-Mutation am .wasm (oder falscher Pin) →
  `check:freeze` rot.

**Nur durch echten CI-Lauf prüfbar (dokumentierte Grenze):** Die GitHub-Actions-YAML-
Orchestrierung (Job-Graph, Toolchain-Setup, Caching, nightly-Install) läuft erst auf
GitHub. Lokal verifiziert wird jedes einzelne **Gate-Kommando**; die YAML selbst via
sorgfältigem Review (+ optional `act`, falls verfügbar) und dem **ersten echten
Push-Lauf**. Der erste Lauf ist Teil der Definition-of-Done: sein Ausgang (grün/rot pro
Job, Linux-Freeze-Hash, ob Instantiation-Pins plattform-stabil sind) wird ins
Post-Verification-Addendum geschrieben — nicht angenommen. **Ein initial-ROTER Freeze-Job
(Linux-Hash ≠ macOS-Pin) ist ein ERWARTETES, actionable Ergebnis (D4/KB-Beleg), KEIN
kaputtes CI** — die Reaktion (Linux-x64-Hash als zweiten gelabelten Pin eintragen, dann
grün) ist Teil der DoD. Dasselbe gilt für einen initial abweichenden Instantiation-Pin
(plattform-Differenz, dann Pin-Update). Beide Reaktionen werden im Addendum dokumentiert.

**Verify-Runde (Stufe 3, voller Katalog):** A (Spec-Konformität, eigener Mutant),
B (adversarial: bricht das Editor-Gate, Timeout-Randfälle, YAML-Fallen wie fehlende
`--frozen-lockfile`/falsche Job-Deps, Freeze-Menge umgehbar?), C (covenant-verify:
hält der Diff M4/Z1/Z2/S1?). Alle mit eindeutigen Worktree-Pfaden.

## 7. Nicht-Ziele (explizit)

- **Keine Vitest-Migration** (Owner 2026-07-17: `node --test` bleiben — zero-dep, grün).
- **Kein Release/Publikation** (npm-Name, LICENSE, `private:false` → Item 13).
- **Kein WebKit/Firefox** in CI (bleibt Chromium-only, D-V3.1).
- **Keine Coverage-Reports, kein Deploy, kein Auto-Merge/Release-Bot.**
- **Keine Runtime-/Kernel-/Typ-Ebenen-Änderung** — Item 12 ist reine Infra.

## 8. Offene Punkte für Baustein 0 / Owner-Richtungsabnahme

*Richtungsabnahme abgeschlossen 2026-07-17: D1 = rust-toolchain.toml (1.95.0); D2 = ubuntu;
D3 = alle Gates bei push+PR; D4 = Freeze-Menge (2⊕3); D5 = Correctness+Instantiation hart,
Latenz 2x-Ceiling; node --test bleibt (keine Vitest-Migration); D7 = zero-dep S1-Guard in CI.*

Für **Baustein 0** zu prüfen:
- Baut `test:threaded` in CI überhaupt (nightly-Install +
   build-std + Cache)? Setzt `actions/setup-node@v4` mit `.nvmrc` Node 24 korrekt? Ist der
   Latenz-2x-Ceiling-Exit korrekt an den bestehenden `gate2xPass` gekoppelt? Sind die
   Instantiation-Pins wirklich deterministisch (kein Nichtdeterminismus im Harness)?
   Verletzt irgendein neuer Import die S1-Regel? Ist `--frozen-lockfile` mit dem aktuellen
   Lockfile konsistent (pnpm 11.6 ⇄ committetes `pnpm-lock.yaml`)?

## 9. Adversariale Spec-Verifikation (Addendum)

Baustein 0 durchgeführt 2026-07-17 (`brainroute:deep`, Sonnet 5 / xhigh, frischer Kontext;
read-only + eindeutig benannter Scratch-Worktree für empirische Checks, Haupt-Tree
nachweislich unberührt). Kein Design-Blocker in der Richtung; die Spec wurde in fünf
Punkten präzisiert, eine Richtungsänderung owner-abgenommen.

**Eingearbeitete Befunde:**
- **F1 (major)** — Es gibt ZWEI toggle-Workloads (W4 UND W6, seit Kern 08), nicht nur W4;
  der Code-Kommentar `editor-latency.ts:697` ist veraltet, `measureWorkload` gated M3
  generisch auf `if (entry.toggle)`. D5 ist generisch formuliert (deckt beide ab) →
  **§6-Testplan präzisiert** (Latenz-Mutant gegen M2 UND M3; Kommentar-Fix in-slice).
- **D-1 (Blocker-Kandidat, S1)** — Ein S1-Guard, der das Pattern gegen den ROHEN relativen
  Specifier testet, wäre strukturell vakuös (kein src-Import beginnt mit `spike/`) →
  **D7 präzisiert**: Specifier-Auflösung gegen das importierende Verzeichnis ist Pflicht;
  alle Import-Formen (type/re-export/mehrzeilig/dynamisch) erfassen.
- **D-2 (major, Richtungsänderung)** — `on:[push,pull_request]` → Doppelläufe bei
  In-Repo-PRs. **Owner-abgenommen: `push: branches:[main]` + `pull_request`** → D3.
- **D-3 (medium)** — Pin-Menge muss wholesale ersetzt, nie appended werden (sonst
  M4-Loch) → **D4 präzisiert**.
- **D-4 (minor, Effizienz)** — `cargo`-Job braucht kein `pnpm install` → **D3 präzisiert**.

**Akzeptierte Nits (nicht adressiert, dokumentiert):** D-5 (test-node baut wasm bis zu 3×
im selben Job — Cargo-Incremental macht 2./3. billig, kein Korrektheitsproblem); die
additive 2x-Ceiling-Exit-Logik dupliziert notwendig einen Teil der `printGateVerdict`-
Iteration (append-only-Disziplin verbietet Editieren der bestehenden Funktion).

**Empirisch bestätigt (F2–F11, für den Bau verlässlich):** `gate2xPass` vorberechnet,
`printGateVerdict` setzt keinen Exit-Code, keine Instantiation-Pins existieren heute (F2);
Instantiation-Zahlen same-platform deterministisch, 3×/2× wiederholt byte-identisch —
w1=24305, w4=24466, w6=30929, w7=23477 (F3, cross-platform bleibt offen wie geframt);
execFileSync-Fehlerpfad bei Typ-Fehlern hat `.status=1`+`.stdout` (F4); Correctness-Wurf
terminiert HEUTE schon mit exit 1 (F5); `--test-timeout` echt+per-test, kein Konflikt mit
`build:wasm &&`-Präfix (F6); die neuen Dateien kollidieren nicht mit `.gitignore`/existieren
nicht (F7); lokaler Hash == Pin `0b9df4f1…` (F8); cargo-Job native ohne wasm-Target (F9);
**nightly-2026-07-09 noch gehostet (HTTP 200), ubuntu-Runner haben rustup/cargo,
`playwright install --with-deps chromium` läuft auf bare ubuntu, `--frozen-lockfile` key't
nicht auf scripts/packageManager (F10)**; der neue S1-Test ist automatisch von
`tsconfig.json` `include:["spike"]` typgeprüft (F11); M4/Z1/Z2 unberührt, kein Nicht-Ziel
implementiert. Ein Rest-Punkt ruht auf Allgemeinwissen statt Doku-Check (setup-node
`node-version-file` mit bloßem `"24"` — bei der Impl absichern).
