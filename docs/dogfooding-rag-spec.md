# Dogfooding-Scheibe: RAG-Demo auf numtype — bindende Spec

Status: **bindend** (Owner-Richtung: HANDOFF 2026-07-20, „Nächste Schritte" Punkt 2,
Owner-bestätigte Reihenfolge; Session-Auftrag 2026-07-20).
Version: 3 (v2 nach Baustein 0; v3 = D6-Präzisierung nach Verify-B, siehe
Addendum-Nachtrag) · Datum: 2026-07-20

## Ziel & Warum

Erste echte Konsumenten-Anwendung auf dem **veröffentlichten** Paket `numtype@0.1.1`:
eine kleine, deterministische Embedding-/Retrieval-Demo (RAG-Kern ohne LLM-Teil).
Der eigentliche Deliverable ist NICHT die Demo, sondern der **Friction-Log** und die
daraus **begründete Op-Wunschliste** — Forschungsdaten für die Frage „welche Ops
braucht ein realer Nutzer zuerst?" (erwartete Kandidaten laut HANDOFF: mean,
concat/stack, argmax/topk; Transzendente sind Covenant-gegated, siehe Nicht-Ziele).
Sekundärnutzen: ein sichtbares, lauffähiges `examples/`-Verzeichnis für OSS-Besucher.

## Berührte Covenant-Invarianten

- **Z1** (Anker `package.json`): Root-package.json erhält NUR ein neues Script
  (`test:example`) — KEIN `dependencies`-Feld, keine Runtime-Dependency. Die Demo hat
  ihre EIGENE package.json mit `numtype` als Dependency; das ist ein Konsument, nicht
  das Paket.
- **Z2** (Anker `package.json`): Der neue Korpus `examples/rag-demo` ist QUELLTEXT im
  Repo, typechecked aber gegen das aus der Registry INSTALLIERTE Paket
  (`node_modules/numtype/dist/*.d.ts`) — ein Bauergebnis, das beim reinen
  `noEmit`-Check nicht existiert. Einordnung daher analog Item-11/S3
  (Konsumenten-Smoke, Covenant v4-Präzisierung): NICHT in `pnpm check`, sondern in
  einem eigenen install-abhängigen Gate `pnpm test:example` + CI-Job. Der Korpus
  rottet nicht (T5). Falls covenant-verify das als Drift wertet → Owner-Entscheidung,
  keine stille Auflösung.
- **S1** (Anker `spike/src/`): `examples/` ist ein neues Demo-Verzeichnis → die
  mechanische Regel `covenant-s1` in graph-a-lama.rules.json wird um `examples/` als
  verbotenes Import-Ziel erweitert (`to`-Regex: `^(spike/(tests|bench|demo)|examples/)`).
  Der INVARIANTEN-TEXT ändert sich
  nicht („Test-, Bench- oder Demo-Verzeichnisse" deckt examples semantisch bereits) —
  kein Covenant-Version-Bump, nur Regel-Abdeckung.
- M1–M5: unberührt (kein Byte in `spike/src/`, `crates/`, Artefakten).

## Bindende Entscheidungen

- **D1 — Ort & Konsum-Form:** `examples/rag-demo/` im Repo (nicht eigenes Repo —
  OSS-Sichtbarkeit), mit EIGENER `package.json` (`"private": true`,
  `"numtype": "^0.1.1"` — in 0.x erlaubt `^` nur Patches, passend zur
  README-SemVer-Politik „Minor = breaking") und EIGENEM `tsconfig.json`
  (strict-Flags wie Root). Eigenes `pnpm-lock.yaml` wird committet
  (Reproduzierbarkeit; CI installiert `--frozen-lockfile`). Kein
  pnpm-Workspace-Verbund mit dem Repo — der Beispiel-Ordner ist bewusst ein
  eigenständiger Konsument, wie ihn ein Nutzer per Copy-Paste hätte.
  **v2 (Baustein-0-Blocker):** Der Ordner erhält ein committetes
  `pnpm-workspace.yaml` mit `minimumReleaseAge: 0` — pnpm 11 blockt sonst per
  Default (≈24h-Fenster) die Installation frisch publizierter Versionen
  (`ERR_PNPM_MINIMUM_RELEASE_AGE_VIOLATION`, empirisch bewiesen inkl.
  Cold-HOME-Isolation); das würde das CI-Gate nach JEDEM künftigen Release
  brechen. Die Datei macht den Ordner zugleich explizit zum eigenen
  Install-Root (stoppt jede Aufwärts-Workspace-Suche). Nebenwirkung bewusst
  akzeptiert: das Fenster entfällt auch für `typescript` (devDep) — beide Deps
  sind lockfile-gepinnt.
- **D2 — Korpus-Isolation (Pin-Schutz):** Root-`tsconfig.json` (include `["spike"]`)
  bleibt UNVERÄNDERT; keine neue Datei unter `spike/`. Damit ist das File-Set aller
  drei gemessenen Korpora fix → `check:diag` muss den Pin **187,918 @ 135 Files**
  EXAKT reproduzieren (T2). Kein neuer Root-Korpus.
- **D3 — Demo-Inhalt (deterministisch, self-checking, zero-dep):** Hardcodierter
  englischer Mini-Korpus (≈16 Kurzdokumente), Embeddings from scratch (gehashte
  Zeichen-Trigramme, TF-gewichtet, L2-normalisiert; reine TS-Funktionen, keine
  externe Lib/API — einzige Dependency ist `numtype` selbst, plus `typescript` als
  devDep für den Typcheck). Pipeline MUSS diese numtype-Surface real exercieren:
  `fromArray` (Matrix-Aufbau), `matmul`, `reshape` und/oder `transpose`,
  `sum(axis)`, elementwise (`mul`/`div` mit Broadcast), `slice`,
  `dot`/`cosineSimilarity` (Kreuz-Check am Top-Treffer). Shapes fließen als
  LITERALE (const-Dims, z. B. N=16, D=256), sodass Hover echte Tupel zeigen
  (Baustein-0-verifiziert: `const`-gebundene Dims inferieren ohne `as const`
  literale Tupel durch den `const S`-Typparameter); mindestens zwei
  `@ts-expect-error`-Pins beweisen Shape-Fehler am Argument auf
  Konsumenten-Seite (Baustein-0-verifiziert gegen die echten dist-Typen für
  MatMul- UND DotCheck-Guards). Läuft via `node <datei>.ts` (Node-Type-Stripping;
  Repo nutzt dasselbe Muster, CI läuft Node 24 via .nvmrc), Ergebnis-Assertions
  mit `node:assert/strict` (≥ 5 Queries mit erwartetem Top-1-Dokument).
  **v2 (Baustein 0):** Die Assertions prüfen zusätzlich eine MARGIN — Top-1-Score
  minus Top-2-Score ≥ eine im Code gepinnte Schwelle (Richtwert 0.03) — damit
  Near-Ties des 16-Dokumente-Trigramm-Korpus nicht als vermeintlich stabile
  Top-1-Treffer durchrutschen; die tatsächlichen Scores werden im Demo-Output
  mitgedruckt.
- **D4 — Friction-Log-Methode (Kern-Deliverable):** JEDE Stelle, an der eine
  natürlich gewünschte Op fehlt oder unergonomisch ist, wird IM MOMENT DES BAUENS
  geloggt: (a) Intent (NumPy-Idiom), (b) tatsächlich geschriebener Workaround
  (Datei:Zeile im Example), (c) Kosten (Zeilen/Lesbarkeit/Typ-Ebenen-Verlust —
  degradiert der Workaround von literalen Shapes zu wide?), (d) abgeleiteter
  Wunschlisten-Kandidat. Die Demo wird NATÜRLICH geschrieben (erst die naheliegende
  numtype-Formulierung versuchen, dann Workaround) — Friction wird nicht durch
  vorauseilende JS-Umgehung versteckt (z. B. Normalisierung NICHT still in der
  Embedding-Funktion erledigen, sondern die Matrix-Formulierung versuchen).
- **D5 — Op-Wunschliste (begründet, priorisiert):** je Kandidat: Evidenz
  (Friction-Log-Referenz), NumPy-Analog, Skizze Typ-Ebene (Shape-Funktion +
  Degradationsregeln nach M2), Runtime-Implikation (naive TS-Referenz + WASM-Kern
  unter M1-Bit-Parität nötig? sqrt ist IEEE-exakt = deterministisch-safe;
  exp/log/sin = Covenant-Non-Goal bis zur Determinismus-Entscheidung — solche
  Kandidaten werden GELISTET aber als gegated markiert), Priorität aus
  Friction-Häufigkeit × Workaround-Kosten. Ablage in
  `docs/dogfooding-rag-ergebnisse.md` (deutsch, interne Forschungsnotiz).
- **D6 — Gates & CI:** Root-Script
  `test:example` = `pnpm -C examples/rag-demo install --frozen-lockfile`
  + Typcheck (`tsc --noEmit` im Example) + Demo-Lauf (Assertions). Neuer CI-Job
  `example` (Node-only, kein Rust, kein Root-`pnpm install`), bestehende Jobs
  byte-unverändert. **v2 (Baustein 0):** der Job setzt
  `cache-dependency-path: examples/rag-demo/pnpm-lock.yaml` (Default wäre das
  Root-Lockfile → toter Cache). Bewusste Randbedingung: das Gate hängt an der
  npm-Registry (installiert das VERÖFFENTLICHTE Paket) — genau das ist der
  Dogfooding-Zweck; Alternative (pack+install des lokalen Stands) verworfen,
  weil sie nicht das testet, was Nutzer erleben. **v2, akzeptiertes Restrisiko
  (Baustein-0 MAJOR):** Der `^0.1.1`-Pin lässt das Example nach einem künftigen
  Breaking-Minor still auf der alten Version weiterlaufen (Gate bleibt grün,
  Example wird unrepräsentativ) → Release-Prozess-Pflicht „Example-Dependency
  bei jedem Release mitbumpen" wandert nach FOLLOWUPS (T9).
- **D7 — README-Sichtbarkeit:** Haupt-README erhält einen kurzen Verweis auf
  `examples/rag-demo` (englisch); das Example bekommt eine eigene englische
  README.md (was es zeigt, wie man es startet, was man im Editor hovern sollte).
- **D8 — Sprache:** Alles unter `examples/`, der CI-Job, README-Zeile,
  Commit-Message: Englisch. Spec, Friction-Log, Ergebnisse-Doc: Deutsch (interne
  Forschungsnotizen, README-disclosed). Markdown-Gate: keine einzelnen `~`
  (GFM-Strikethrough-Falle) — Prüfung via `gh api markdown` auf allen neuen/
  geänderten Markdown-Dateien.
- **D9 — Keine neuen Ops in dieser Scheibe:** Die Wunschliste ist der Output; ihre
  Umsetzung (auch „nur mean") ist eine SPÄTERE Scheibe mit eigener Spec. Die Demo
  arbeitet ausschließlich mit der 0.1.1-Surface + dokumentierten Workarounds.

## Akzeptanzkriterien

- **T1:** `pnpm test:example` exit 0: frozen Install + Example-Typcheck + Demo-Lauf
  mit ≥ 5 deterministisch asserteten Retrieval-Ergebnissen.
- **T2:** `pnpm check` grün UND `pnpm check:diag` reproduziert **187,918 @ 135**
  exakt (File-Set unberührt; Exit-Code + Fehlerausgabe geprüft, nicht nur Kennzahl).
- **T3:** `pnpm test:core` grün (test-scripts-guard akzeptiert den neuen
  Script-Eintrag; keine `*.test.ts` unter `examples/` — Demo-Dateien heißen bewusst
  nicht `*.test.ts`).
- **T4:** Jeder Wunschlisten-Eintrag referenziert ≥ 1 konkrete Workaround-Stelle
  (examples/…:Zeile) + NumPy-Analog + M1/M2-Einschätzung; transzendente Kandidaten
  tragen explizit den Gate-Vermerk (Covenant-Non-Goal, FOLLOWUPS Z. „Transzendente
  Ops").
- **T5:** CI-Workflow enthält den neuen Job `example`; `git diff` von ci.yml zeigt
  bestehende Jobs unverändert (nur Additionen).
- **T6:** `graph-a-lama query lint` exit 0 mit der um `examples/` erweiterten
  covenant-s1-Regel (nach Graph-Rebuild); Regel↔Spec-Linkage intakt (Invarianten-
  Text unverändert).
- **T7:** Mindestens zwei `@ts-expect-error`-Shape-Fehler-Pins im Example
  kompilieren (d. h. der Fehler ERSCHEINT am Argument — Konsumenten-seitige
  M2/M3-Evidenz gegen dist-Typen).
- **T8:** GFM-Gate: `gh api markdown` über alle neuen/geänderten .md-Dateien
  liefert 0 `<del>` (einmaliges Scheiben-Gate, bewusst OHNE CI-Verdrahtung —
  Konvention wie bisher, kein Markdown-Lint-Job existiert).
- **T9:** Doc-Platzierung nach Hausregel: roadmap unberührt (kein Roadmap-Item),
  CLAUDE.md nur Status-Einzeiler, volles Narrativ als Append an
  docs/projekt-log.md, FOLLOWUPS-Eintrag mit Verweis auf die Wunschliste.

## Nicht-Ziele

- Keine neuen Ops, keine Signatur-/Typ-Ebenen-Änderungen, kein Byte unter
  `spike/src/`, `crates/`, `dist/`-Toolchain.
- Keine Transzendenten-/Determinismus-Entscheidung (bleibt Covenant-Non-Goal).
- Kein npm-Publish, keine COVENANT.md-Änderung, kein LLM-/Generierungs-Teil im
  RAG (Retrieval only — der „G"-Teil braucht externe Modelle und ist nicht der
  Forschungsgegenstand).
- Keine Performance-Aussagen (Mini-Korpus; Benchmarks sind nicht Zweck der Demo).

## Verify-Plan (Eskalationsleiter: Stufe 3)

Baustein 0 (adversarialer Spec-Verifier, brainroute:deep) VOR der Implementierung;
nach der Implementierung Verify-Runde A (Spec-Konformität + Gates + eigener Mutant,
z. B. erwarteten Top-1-Index korrumpieren → test:example muss rot werden) + B
(adversarial: Determinismus über Läufe, Registry-/Install-Randbedingungen,
tsconfig-Leckagen, Sprach-/Markdown-Gates) + C (covenant-verify: Z1/Z2/S1 am Diff)
parallel; Aufträge aus docs/verify-runde-template.md. Ergebnisse-Doc mit
Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-20)

Verifier: brainroute:deep, frischer Kontext, mit empirischen Proben (echte Registry,
Scratch-Konsument, Wegwerf-Worktree; Haupt-Tree unberührt). Befunde und Auflösung:

1. **BLOCKER (bestätigt, hoch):** pnpm-11-Default `minimumReleaseAge` (≈24h) blockt
   `--frozen-lockfile`-Installs frisch publizierter Versionen
   (`ERR_PNPM_MINIMUM_RELEASE_AGE_VIOLATION`, dreifach reproduziert inkl.
   Cold-HOME). → In D1 eingearbeitet: committetes `pnpm-workspace.yaml` mit
   `minimumReleaseAge: 0` im Example-Ordner. Keine Richtungsänderung
   (Registry-Konsum bleibt); Mechanik-Entscheidung im Rahmen der Owner-Richtung.
2. **MAJOR (mittel):** `^0.1.1` rottet still nach künftigem Breaking-Minor (Gate
   bleibt grün auf alter Version). → Akzeptiert + FOLLOWUPS-Pflichteintrag
   „Example-Dep bei jedem Release bumpen" (D6/T9).
3. **MINOR (hoch):** `cache: pnpm` ohne `cache-dependency-path` → toter Cache.
   → In D6 eingearbeitet.
4. **MINOR (plausibel):** Near-Tie-Risiko der Top-1-Assertions. → Margin-Pflicht
   in D3 eingearbeitet.
5. **Nits:** S1-Regex präzisiert (oben); T8 als Einmal-Gate klargestellt; Node-
   Versionsangabe auf das .nvmrc-Faktum (Node 24) gestützt.
6. **Positiv verifiziert:** alle API-Annahmen (fromArray flach, sum-Overloads,
   dot/norm/cosineSimilarity, Broadcast-mul/div); Korpus-Isolation von examples/
   (Pin-Schutz strukturell); test-scripts-guard scannt nur drei spike-Verzeichnisse
   und benannte Scripts; kein Workspace-Join ohne pnpm-workspace.yaml; Z1 hält;
   `const`-Dim-Literal-Inferenz ohne `as const`; `@ts-expect-error`-Pins tragen
   gegen die echten dist-Typen (MatMul + DotCheck); .gitignore deckt
   examples-node_modules bereits ab.

### Nachtrag v3 (nach Verify-Runde, Baustein-B-MAJOR — 2026-07-20)

Die D6-/CI-Kommentar-Behauptung „kein Root-`pnpm install`" war empirisch FALSCH:
`pnpm test:example` (Aufruf eines beliebigen Root-Scripts) installiert auf einem
kalten Runner implizit die kompletten Root-devDependencies nach (pnpm-11-Verhalten,
von Baustein B zweifach in Cold-HOME-Worktrees reproduziert) — ungecacht, weil der
Cache bewusst am Example-Lockfile hängt. Auflösung: Der CI-Job ruft die drei
Example-Steps DIREKT auf (`pnpm -C examples/rag-demo install --frozen-lockfile` /
`run check` / `run demo`) statt über das Root-Script; das Root-Script
`test:example` bleibt als lokale Convenience bestehen (lokal existiert
node_modules ohnehin). CI-Job-Kommentar entsprechend korrigiert. Kleinbefunde
mit-adressiert: `engines: node >=22.18` im Example-package.json (README-Claim
maschinenlesbar), Friction-Inline-Nummern an die F-Nummern des Ergebnisse-Docs
angeglichen, ein `~`→`≈` im Ergebnisse-Doc.
