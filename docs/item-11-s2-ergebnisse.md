# Item 11 / S2 — Ergebnisse (Emit-/Paket-Pipeline)

**Stand:** erledigt & DREIfach verifiziert (Spec CONFIRMED + adversarial 2 Befunde behoben +
covenant-verify kein Verstoß), 2026-07-17. Bindende Spec: `docs/item-11-api-paket-spec.md`
(Abschnitt S2, D-S2.1 bis D-S2.6). Ausgangscommit 48ee440 (S1). ZERO Source-Änderungen.

> **Ehrlichkeitsregel:** Jede Zahl/Aussage in einem tatsächlich gelaufenen Kommando verankert.
> Zwei adversariale Befunde (B) + ein Spec-Prämissen-Befund (A) offen berichtet und adressiert.

## Was S2 gemacht hat

Baut aus `spike/src/**` ein publizierbares npm-Paket nach `dist/`. Neue Dateien:
`tsconfig.build.json` (dist-Emit-Config: `declaration`, `rewriteRelativeImportExtensions`,
`noEmit:false`, `outDir:dist`, `rootDir:spike/src`), `scripts/postbuild-dist.mjs` (zero-dep
Post-Emit-Rewrite), package.json-Publish-Metadaten (`exports`/`main`/`module`/`types`/`files`/
`sideEffects`/`keywords`/`license`/`repository`), `.gitignore` (`/dist/`). Build-Script
`pnpm build:dist`.

**Die drei Emit-Blocker (empirisch fixiert):**
- Blocker 1 (`.d.ts` behalten `.ts`-Endungen) + Blocker 2 (`new URL("./threaded-worker.ts")`) →
  vom Post-Emit-Rewrite behoben (`.ts`→`.js` in import/export/`import()`/`new URL`-Positionen,
  in `.js` + `.d.ts`), die `.wasm`-URLs (`numtype_core.wasm`, `numtype_core_threads.wasm`)
  bleiben unberührt.
- Blocker 3 (`node:worker_threads` in `threaded.d.ts`) → Fix (a): der Standard-Konsument mit
  `skipLibCheck:true` (Vite/Next/CRA-Default) checkt sauber (EXIT 0 verifiziert). Kein Zwang zu
  `@types/node`. `@types/node`-als-devDep war ein Kategorienfehler (Baustein 0).

## Gate-Ergebnisse

| Gate | Ergebnis |
|---|---|
| `pnpm build:dist` | EXIT 0 (baut wasm + emittiert 18 Module + rewritet + kopiert .wasm) |
| Emit-Präzision | 0 verbleibende relative `.ts`-Modul-Referenzen in dist; `.wasm`-URLs unberührt |
| `ls dist/ambient*` | kein Leak (ambient.d.ts erzeugt keinen Output) |
| `.wasm`-Hash (dist + src) | `0b9df4f1…519c7d` — byte-identisch zum Pin |
| `pnpm pack --dry-run` | enthält `dist/wasm/numtype_core.wasm` (gitignore-Fallstrick greift nicht), NICHT `*_threads.wasm`, keine spike/docs/tests/.map |
| `pnpm check` (Dreier) | EXIT 0 |
| `git diff 48ee440 -- spike/src` | 0 Zeilen — ZERO Source-Änderungen |
| `test:core`/`test:resident`/`test:threaded`/`cargo`/`demo` | 818 / 4280 / 69 / 161 / grün (unverändert) |
| Konsumenten-Import (fremdes CWD) | `dist/index.js` lädt, `sum()`=21 / `sum(0)`=[5,7,9] / `sum(1,true)`=[[6],[15]] |
| M5 (Browser-Sicherheit) | Default-`NDArray`-Pfad zieht kein eager `node:*` (Laufzeit-Trace + statisch bestätigt) |

## Post-Verification-Addendum (drei Fresh-Context-Verifier, eindeutige Worktree-Pfade)

Diesmal mit **erzwungen eindeutigen PID-Worktree-Pfaden** (KB-Lektion aus S1) — keine
Kontamination, alle drei Läufe sauber isoliert.

- **Baustein A (Spec-Konformität): CONFIRMED.** D-S2.1 bis D-S2.6 konform; alle Gates grün mit
  realen Zahlen; check:diag im Worktree 179'986 @ 132 (= S1-Baseline, NULL S2-Root-Korpus-Kost).
  Eigener Mutant (Rewrite-`from`-Pattern entfernt) → Emit-Grep fängt ihn (39 verbliebene `.ts`).
  **MAJOR-Befund (Spec-Prämisse, nicht Impl):** Blocker 1 bricht KEINEN realen Konsumenten unter
  TS 7.0.2 (bundler+nodenext) — der Rewrite bleibt defensiv-korrekt (Publish-Konvention), die
  Spec-Begründung präzisiert. **minor:** `author`-Feld fehlt (→ bewusst Item 13); der Emit-Grep
  ist die einzige Absicherung für Blocker 1 (Coverage-Konsequenz dokumentiert).
- **Baustein B (adversarial): 2 Befunde, beide BEHOBEN.** M5/Rewrite-Präzision/Freeze/Gate-Block
  halten vollständig (Idempotenz, `.wasm`-URLs unberührt, Tarball-Hygiene nach clean).
- **Baustein C (covenant-verify): KEIN VERSTOSS.** Z1 (kein `dependencies`-Feld), Z2 (check
  bleibt Dreier-Verbund, dist-Emit ist BUILD nicht CHECK), M5 (browser-sicher, Import-Struktur
  unverändert), M4 (Hash byte-identisch), M3 (keine Typ-Änderung), S1 (lint 0) — alle halten,
  keine Anker-Drift. Methodischer Hinweis: der `node:`-Substring-Filter im moduleLoadList-Trace
  ist auf Node v24 blind (Builtins ohne `node:`-Präfix) — für künftige M5-Traces auf konkrete
  Builtin-Namen filtern (KB-Kandidat).

### Adressierte Befunde

- **B-1 (MEDIUM, latent): Prosa-Fehlrewrite.** Der Regex-Rewrite konnte JSDoc-/Kommentar-Prosa,
  die zufällig `from "./x.ts"`/`import("./x.ts")`/`new URL("./x.ts")` enthält, fälschlich
  umschreiben (heute kein echter Treffer, aber Risiko im stark kommentierten Code). **BEHOBEN:**
  `rewrite` ist jetzt kommentar-bewusst (zero-dep Block-Kommentar-Tracker, überspringt `/* */`/
  `/** */`/`//`-Zeilen). Verifiziert per Prosa-Immunität-Test (echte `rewrite`-Funktion gegen
  Code+Prosa-Misch: Code→`.js`, Prosa→unverändert `.ts`, PASS). Skript dafür testbar gemacht
  (`export function rewrite`, Haupt-Walk hinter run-if-main-Guard).
- **B-2 (MEDIUM-HIGH): `build:dist` räumte `dist/` nicht auf.** Stale-/Fremddateien überlebten
  und landeten im Tarball — besonders heikel nach der S1-Umbenennung. **BEHOBEN:** `rm -rf dist &&`
  vor dem tsc-Emit. Verifiziert (Leftover-Datei nach Rebuild weg).
- **A-2 (minor-moderate): `sideEffects:false` übersah `threaded-worker.ts:175 void main()`** —
  ein echter Top-Level-Effekt (aber die Datei wird nie ESM-importiert, nur als Worker-URL
  geladen). **BEHOBEN:** ehrlich als `sideEffects: ["**/threaded-worker.js"]` markiert.
- **A-1 (MAJOR, Spec-Prämisse):** Blocker-1-Begründung — Spec-Text präzisiert (Rewrite bleibt,
  defensiv-korrekt). Kein Code-Defekt.
- **A-3 (minor): `author`** — Spec präzisiert (bewusste Item-13-Deferral wie LICENSE).
- **Eigener Bug beim Fix:** der run-if-main-Guard warf bei `node -e`-Import (`process.argv[1]`
  undefined) — von meinem eigenen Prosa-Test gefangen, per `process.argv[1] &&`-Guard behoben.

## Nächste Schritte

Item 11 / S3 (Zero-dep-Guard-Test + Paket-Smoke, zwei Ebenen) — eigene A+B+C-Runde. Der
Emit-Präzisions-Grep muss dort als Pflicht-Gate mit (A-1-Coverage-Konsequenz: der Konsumenten-
Smoke fängt eine Blocker-1-Rewrite-Regression NICHT).
