# NumType

> **Note for contributors:** This file is the maintainer's internal working memory
> for an agent-assisted workflow (Claude Code). It references private tooling and
> plugins (`brainroute`, `graph-a-lama`, `coding-kb`, `covenant`, …) that are not
> part of this repository, and it mixes English with German research notes.
> Nothing in here is required to build, test, or use NumType — see
> [README.md](README.md) and [docs/](docs/) instead. The commands in the
> "Commands" section work for everyone, though.

NumPy-like n-dimensional array library: TypeScript type-level shape checking + from-scratch Rust/WASM kernels. Research project — the explicit goal is probing the limits of what's feasible.

**Diese Datei trägt nur Regeln + IST-Stand** (verschlankt 2026-09-23, Owner-Entscheidung). Historie:
[docs/projekt-log.md](docs/projekt-log.md) (Scheiben-Narrative) ·
[docs/claude-md-archiv-2026-09-23.md](docs/claude-md-archiv-2026-09-23.md) (wortgetreue Vorfassung,
789 Zeilen, inkl. aller Pin-Historien und Regel-Herleitungen) · `docs/*-spec.md` + `docs/*-ergebnisse.md`
(Primärquellen) · FOLLOWUPS.md (zurückgestellte Arbeit) · HANDOFF.md (lokal, untracked).
**Budget: diese Datei bleibt unter ~250 Zeilen.** Neues Narrativ gehört ins Log, nie hierher.

## Hard constraints (user-set, 2026-07-09)

- **No external libraries.** All kernels and all type machinery written from scratch. Dev tooling is allowed; product/runtime dependencies are not. Never suggest `ndarray`/`faer`/BLAS bindings etc.
- Brand name: **NumType** (npm package name lowercase `numtype`).
- Public repo + npm since 2026-07-19. **All user-/public-facing text is English** (commits, tags, release notes, issues/PRs, README/spec docs, code comments, error messages). Internal process/research docs may stay German. Chat with the owner stays German.
- Research fan-outs stay small: targeted agents (≤3), no broad sweeps.

## USP (sources in docs/wettbewerbsanalyse-und-usp.md)

NumType is to NumPy what TypeScript is to JavaScript: shape errors become editor errors — gradual, with a `number`-dim escape hatch for dynamic shapes. Consumer-scale is measured (Scale-Probe 2026-07-21); **API-surface scale is still unproven** — that is the open research question.

## Status (IST, 2026-09-23)

- **npm:** `numtype@0.3.0` (2026-09-24, „parity and polish"; davor 0.2.0 am 2026-07-21). Registry-Tarball nach dem Publish verifiziert (Integrität, Inhalt), Beispiel läuft unverändert auf 0.3.0. Tags `v0.1.0`/`v0.1.1`/`v0.2.0`/`v0.3.0`, Apache-2.0, Repo public, Rulesets `protect-main` + `protect-release-tags`.
- **Aktive Roadmap (Owner-entschieden 2026-09-23, docs/roadmap.md „Roadmap ab 2026-09-23"):**
  0a Release 0.3.0 (ERLEDIGT 2026-09-24) → 0b typisiertes `toNestedArray` (ERLEDIGT 2026-09-24, unveröffentlicht → 0.4.0) → dtype-Design (ERLEDIGT 2026-09-25: Entscheidungen + Prototyp-Messung, docs/dtype-design-ergebnisse.md; Prototyp auf lokalem Branch `proto/dtype`) → dtype-Umsetzung: **dt1 ERLEDIGT 2026-09-26** (Kern auf `NDArray`, unveröffentlicht), **als Nächstes: dt2** (Promotion,
  gemischte Arithmetik, Skalar-Regel) … dt5 → 2 Op-Umfang „die ersten zehn Minuten" → 3 API-Flächen-Skala +
  Strukturumbau → 1 Verbreitung (bewusst ans Ende gestellt).
- **Geparkt:** Klassifikation der View-Test-Restmenge (Spec v2.1 + Skripte als WIP committet;
  Selbsttest 12 rot — FOLLOWUPS). Wird erst nach Phase 0/2 wieder aufgenommen, falls überhaupt.

## Aktuelle Pins & Gates (IST; Historie im Archiv/Log)

- **Freeze-Hash** (Clean-Rebuild, SHA256 `spike/src/wasm/numtype_core.wasm`): `2a54d9fdba55e4e88a9d54cb3b01e111c2717abf13017f778b90accd5cff87e4` (seit S5/topk). Threads-Artefakt bewusst ohne Pin — test:threaded beweist Bit-Identität. CI-Gate `check:freeze`; die Byte-Identität ist cross-host (macOS-arm64 = linux-x64).
- **check:diag** Root **237,098 @ 140** · **stress 118,562 @ 82** · **browser 2,142 @ 75** (seit dt1/dtype-Kern 2026-09-26: Root Δ+9,693, stress Δ+2,283) (stress/browser ungated, `pnpm check` compoundet alle drei).
- **bench:editor** W1–W8 exact-match: `{w1 40,219, w2 42,160, w3 73,097, w4 40,374, w5 45,785, w6 46,736, w7 39,300, w8 47,184}` (seit dt1; Hover-Erwartungen tragen jetzt den dtype; „editor-Δ = stress-Δ" ist Faustregel, kein Gesetz); Latenz am 2x-Ceiling.
- **Tests:** test:core 1618 · test:resident 6155+2 · test:threaded 139 · test:browser 4 · test:package 3 + zwei Konsumenten-Typ-Smokes (`consumer` skipLibCheck:true, `consumer-strict` skipLibCheck:false ohne @types/node) · cargo 222+1 · test:example (Registry-Install + 8 asserted Queries).
- Alle Werte am 2026-09-23 im frischen Worktree reproduziert (Toolchain: node 24.16, pnpm 11.6, tsc 7.0.2, rustc 1.95.0, nightly-2026-07-09).

## Mess-Regeln (tragend)

- Der Instantiation-Counter ist **CHECK-ORDER-abhängig** — Pins sind nur für ein fixes File-Set exakt. Datei hinzufügen/umbenennen verschiebt um **bis zu ±≈7,000** (Order-Noise) → per empty-then-fill dekomponieren. Datei-EDITS können echte Kosten sein → bisektieren mit gleichlanger Kommentar-Kontrollprobe.
- Weitere Mechanismen: geteilte **Klassen-Surface** rippelt in alle Korpora (generische Member/Overloads); **`keyof`-getriebene Alias-Neuauflösung** (`Omit`/`Pick` über ein wachsendes Interface: +N pro Member, uniform — S0/D10 hat ihn in threaded.ts umgangen); **Super-Additivität** innerhalb eines fixen File-Sets (nur Alles-oder-nichts-Messungen sind aussagekräftig, eine Pro-Aufrufstelle-Zahl existiert nicht).
- Nie über den Root/Stress-Split vergleichen. Baselines nur im frischen `git worktree` des Zielcommits (nie `git stash`). IMMER Exit-Code + Fehlerausgabe prüfen, nie nur die Kennzahl greppen.

## Arbeitsregeln (Herleitungen im Archiv)

1. Mutanten-Revert im Haupt-Tree nur als inverser Edit oder Backup-Kopie-Restore mit `diff`-Beweis — NIE `git checkout`/`git restore` auf Dateien mit uncommitteter Arbeit.
2. Overload-Sets: der ZULETZT deklarierte Kandidat trägt die Fehlerdiagnose — Guard-Träger zuletzt; Diagnose-INHALTE pinnen (tsc auf Außer-Repo-Fixture).
3. Neue Typ-Folds: IsUnion-Gate VOR jeder naked Destrukturierung.
4. Markdown: keine `~~…~~`-Strikethroughs (reißt das 0-`<del>`-GFM-Gate).
5. Shell (Agenten): der nvm-chpwd-Hook wirft bei `cd <repo> && …` Exit 3 → `cd <root> 2>/dev/null; …` oder `git -C`. zsh: `status` ist read-only; Pipe-Exits in `$pipestatus[1]`. Auf CI warten: `gh run watch <id> --exit-status`.
6. Ein Gate prüft die GESUNDHEIT seines Messlaufs mit (Exit-Code/Fehleranzahl), nicht nur die Kennzahl.
7. Vorregistrierte Entscheidungsregeln als Skript nachbauen und gegen synthetische Ergebnisse fuzzen, BEVOR echte Zahlen existieren.
8. Informelle Sondagen sind keine Messungen und begründen keine Folgearbeit.
9. Agenten mit Hintergrundprozessen enden oft auf „ich warte" statt auf einem Bericht → selbst überwachen oder per SendMessage den Bericht anfordern.
10. Jeder neue `CoreExports`-Member braucht einen `notImplemented(...)`-Stub in jedem EXHAUSTIV getippten Mock (nicht in Spread-Mocks). Prüfen: `grep -rn --include='*.ts' -A1 ': CoreExports = {' .` — Folgezeile `...x,` = Spread, sonst exhaustiv. Fehlender Stub → TS2739 bei trotzdem plausibler Kennzahl.
11. Vor jedem neuen Kernel prüfen, ob die Op eine KOMPOSITION verifizierter Kernel ist → dann komponieren (Freeze-Hash unberührt); Preis: expliziter Leck-/Lebenszyklus-Test des Zwischenergebnisses.
12. Residente Op-Tests treffen VIEWS (transponiert/geslict/offset/komponiert), und jeder View-Fall ASSERTIERT seine Klasse direkt nach der Konstruktion: exakter Shape + exakter Strides-Vektor + Offset (Ungleichungen genügen nicht).
13. Hover-/Diagnose-Normen werden per echtem `tsc --lsp --stdio` GEMESSEN, nicht gelesen (Alias in Rückgabe-Position bleibt im Hover namentlich stehen).
14. Neues std-Generikum im Kernel (Sortierung, Formatierung, Collections) → `strings -a <artefakt> | grep -E "/Users/|/home/|rustup"` muss LEER sein. Weicht eine vorher identische Plattform ab, ist das ein Befund, kein Pin-Anlass.
15. Vollständigkeitsprüfungen nutzen eine ANDERE Auffindungsmethode als die geprüfte Arbeit (Restmengen-Invariante statt erneuter Stichwortsuche).
16. Laufzeittests bauen Views mit `wideSpecs(...)` (`spike/tests-runtime/assert-helpers.ts`), wenn der Empfänger statisch bekannten Rang hat; Ausnahme `slice.test.ts`.
17. `node --test` druckt Fehlschläge doppelt — über eindeutige Testnamen zählen, nie `grep -c` auf Rohoutput.
18. README gehört zur Doc-Platzierung, sobald eine Scheibe ändert, WO eine Op läuft; Behauptungen empirisch gegen `spike/src/index.ts` prüfen (`grep -n "TypeScript-runtime only\|no WASM kernel" README.md`).
19. **README auf `main` darf nichts als verfügbar bewerben, was nicht auf npm ist** (Review-Befund 2026-09-23: die S0–S5-Parität stand in der README, npm 0.2.0 hatte sie nicht). Neue Features in der README als „on main, unreleased" markieren oder erst mit dem Release eintragen.
20. **Tests, die zur Laufzeit Dateien in `spike/src` schreiben** (Mutanten-Nachweise), brauchen ein Namensmuster, das alle tsconfig-Globs über `spike/src` ausschließen (heute `__*-mutant-*-tmp*.ts`), und einen eindeutigen Namen je Lauf — sonst bricht eine bei hartem Abbruch liegengebliebene Kopie `check:diag` und landet im npm-Paket (dt1, 2026-09-26).
21. **Budget-Stopp-Punkte** in Specs sind nur wirksam als eigener Commit VOR dem nächsten Baustein — misst ein Agent erst nach allem, ist der Stopp wirkungslos (dt1).

22. **Agenten setzen NIE `git config`** (auch nicht „nur im Worktree": ohne `--worktree` schreibt es in die gemeinsame `.git/config` und gilt für ALLE Worktrees und `main`). **Vor jedem Push:** `git log --format='%an <%ae>' origin/main..main | sort -u` darf nur `crankfunk <45401993+crankfunk@users.noreply.github.com>` zeigen. (Vorfall 2026-09-26: ein Agent setzte `scratch <scratch@local>` repo-weit, 16 Commits gingen so auf GitHub; per Owner-Entscheidung umgeschrieben.)

## Commands

`pnpm check` (Typ-Verbund Root + stress + browser) · `pnpm check:diag` / `:stress` / `:browser` ·
`pnpm test:core` · `pnpm test:resident` (+`:gc`) · `pnpm test:threaded` (braucht stable 1.95.0 UND
nightly-2026-07-09 + rust-src; Install in `scripts/build-wasm-threads.sh`) · `pnpm test:browser`
(Wrapper emittiert frisch — **nie `playwright test` direkt**) · `pnpm test:package` (baut `build:dist`,
Emit-Gate + Laufzeit- + Konsumenten-Typ-Smoke; läuft als `prepublishOnly`) · `pnpm build:dist` ·
`pnpm check:freeze` · `pnpm demo` · `pnpm bench:*` · `bench:editor` (hartes Gate) ·
`cargo test --manifest-path crates/core/Cargo.toml`.

Alle Commands aus dem Repo-Root (Cargo-Config-Discovery ist CWD-basiert; simd128 via
`.cargo/config.toml`). Test-Skripte nutzen EXPLIZITE File-Listen in package.json — neue Testdateien
eintragen; `test-scripts-guard.test.ts` failt bei unregistrierten/doppelten/fehlenden Dateien.

## Frozen-baseline discipline (hard)

New code in files shared with frozen kernels/entry points (abi.rs, matmul_blocked.rs, shape.rs, …) is APPENDED strictly after all pre-existing content (in abi.rs: after the last test module) — line shifts change `#[track_caller]` panic metadata and thus bytes of untouched functions. The binding proof is the artifact hash from a clean rebuild, three-part: pre-edit rebuild reproduces the old pin, diff is additive-only, new pin from two identical clean rebuilds. TS class bodies take the intent-preserving equivalent (house rule, not covenant — the byte argument does not exist in TS; its value is reviewability): **(a) ordinary new ops** are insertion-only, zero edits to pre-existing members. **(b) planned API-wide migrations** (e.g. dtype: every method gains a type parameter) may edit pre-existing members IF the spec names them in advance and the owner approves before implementation — and then **no pre-existing test may be modified, only new ones added**, so the untouched old tests prove existing behavior is unchanged (Owner-entschieden 2026-09-25; Präzedenz 0b und dtype-Prototyp). Threads path: env RUSTFLAGS replaces config rustflags (always carry `+simd128`); `thread_local!` forbidden in the crate; never cache `memory.buffer`/views (fails silently with shared memory).

## Toolchain note

TypeScript is **7.0.2** (native/Go). Limits below were researched on TS 5.x — hypotheses to verify on 7.x. `--extendedDiagnostics` works.

## Key TS limits (sourced in docs/wettbewerbsanalyse-und-usp.md §4)

- ≈100 instantiation depth non-tail-recursive, ≈1000 tail-recursive → all recursive types accumulator/tail-recursive. Global ≈5M instantiation budget per compilation — track via `check:diag` (Root-Pin ≈4.5 %).
- Tuple-length arithmetic only for *ranks*. Large literal dims use digit-string arithmetic (`spike/src/literal-arithmetic.ts`: add/sub/compare/mul/`DivCeil`); products cap at MAX_SAFE_INTEGER, beyond → degrade to `number`.
- `${N}` renders plain digits below 1e21; exponent forms → no claim ever. Dot-form = proven non-integer.
- Declaring unused template-literal-heavy generic aliases COSTS instantiations on TS 7 — unused machinery is not free.
- Distributive helpers yield union verdicts → accept only via tuple-wrapped subset checks (`[C] extends ["eq"|"gt"]`), else no-claim. Lifting a runtime error to compile time breaks your own runtime error tests → widen deliberately (`5 as number`) + parity test.
- TS 7 reports only ONE diagnostic per call when several arguments are invalid → pin via type-level assertions, not squiggle counts.
- Variance: `out S` is enforced abstractly (computing type functions fail TS2636); method-shorthand params are bivariant → `NDArrayView` never gains an `S`-consuming member, function members property-style.
- `const` type parameters, so callers never need `as const`. Shape errors surface at the offending argument with both shapes named; hovers show clean tuples like `NDArray<[2, 4]>`.

## Knowledge capture (user-mandated)

Substantielle Scheiben enden mit: Ergebnisse im Projekt (Ergebnis-Doc bzw. Log-Abschnitt, ehrlich über Lücken) → allgemeine Lehren als atomare Notiz in den coding-kb-Vault (`90-Meta/Capture-Workflow.md`; bestehende Notizen revidieren statt duplizieren) → in MOCs verlinken, Graph neu bauen, per `coding-kb`-Query prüfen. Vor Arbeit in bekannter Domäne erst die KB konsultieren. **Doc-Platzierung:** fertige Scheibe → docs/roadmap.md (Status) + hier Status/Pins (nur Einzeiler + Zahlen) + Narrativ an docs/projekt-log.md + README nach Regel 18/19.

## Qualitätssicherung (Owner-Mandat 2026-07-12, kalibriert 2026-09-23)

- **Nicht-Fable-Session:** zu Beginn substanzieller Arbeit `brainroute:fable-doctrine` laden; per brainroute klassifizieren und delegieren, mit Router-Ansage.
- **Strukturfragen zuerst über den Graph** (`graph-a-lama` outline/def/usages/callers).
- **Abweichung von einer Hausregel → VOR der Implementierung dem Owner vorlegen** („disclosed + confirmed").
- **Hintergrund-Agenten fassen den Haupt-Working-Tree nie an**; jeder MUTIERENDE Verifier bekommt einen eigenen Worktree (KB `parallele-mutierende-verifier-worktree-patch`: `git diff > slice.patch` außerhalb des Repos, `git apply` je Worktree, node_modules symlinken). Vor jedem Mutanten `git diff -- <quellpfad>` prüfen; unerklärte Diskrepanzen zwischen identischen Läufen = Kontaminationsverdacht.
- **Covenant:** COVENANT.md (v8) ist der stehende Produktvertrag. `graph-a-lama query lint` läuft im Gate-Block mit. Spec-Änderungen nur mit Owner-Bestätigung + Version-Bump + Changelog.

### Eskalationsleiter — nie vorsichtshalber den vollen Katalog fahren

| Stufe | Wann | Pflicht |
|---|---|---|
| **0** trivial | Typo, Kommentar, Doku-only, kein Anker berührt | nichts |
| **1** klein, anker-frei | kleine Code-Änderung ohne Covenant-Anker | Gate-Block + Lint |
| **2** Mini mit Anker | FOLLOWUPS-Mini, berührt einen Anker | Lint; `covenant-verify` nur bei INHALTLICHER Berührung, sonst Ein-Satz-Begründung im Commit |
| **3a** Routine-Scheibe *(neu 2026-09-23)* | Neue Op/API-Polish/Release **nach etabliertem Muster**: keine neue Typ-Maschinerie-KLASSE, kein neuer/geänderter WASM-Kernel (Freeze-Hash bleibt), keine Mess-/Forschungsfrage, keine Covenant-Änderung | **Kurz-Spec** (≤ ~1 Seite: Ziel, Nicht-Ziele, berührte Anker, Gates mit Absolut-Grenzen, Testplan) → Owner-Abnahme → Umsetzung → **EIN kombinierter Verifier** (Konformität + adversarial, eigener Worktree, Auftrag „Baustein R" in docs/verify-runde-template.md) + `covenant-verify` nur bei inhaltlich berührtem Anker. **Kein Baustein 0**, kein eigenes Ergebnis-Doc (Abschnitt im Projekt-Log genügt). |
| **3b** volle Scheibe | Neue Typ-Maschinerie-Klasse, Kernel dazu/geändert (M1 bindet, Freeze-Hash bewegt sich), Mess-/Forschungsscheibe, Covenant-Änderung, oder 3a mit Blocker-Befund | Bindende Spec → **Baustein 0** (`brainroute:deep`, vor dem Code) → Umsetzung → **A + B + C parallel** (Aufträge aus docs/verify-runde-template.md) + Lint → Ergebnis-Doc mit Post-Verification-Addendum |

**Hochstufen, nie still herunterstufen:** Findet der 3a-Verifier einen Blocker, wird die Scheibe zu 3b. Die Stufe steht in der Spec und im Commit; im Zweifel entscheidet der Owner.
