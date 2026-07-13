# Verify-Runde — Auftrags-Template (Owner-Mandat 2026-07-12)

Jede substanzielle Scheibe endet mit einer Verify-Runde aus **zwei** Fresh-Context-
Verifiern (Anthropic-Befund: separate Fresh-Context-Verifier schlagen Selbstkritik;
Kern-09-Befund: ein einzelner Spec-Verifier erbt die blinden Flecken seines
Auftraggebers — der adversariale Zweite ist die Gegenmaßnahme). Aufträge aus diesem
Template instanziieren, nicht frei formulieren: der Kern-09-Pin-Messfehler (F2)
steckte in einem frei formulierten Auftrag.

## Gemeinsame Pflichtregeln (in BEIDE Aufträge kopieren)

1. **Kein Griff in den Haupt-Working-Tree**: kein `git stash`, kein `git checkout`,
   keine Datei-Modifikation dort. Messungen an anderen Commits ausschließlich in
   einem frischen `git worktree add <scratch> <commit>` (node_modules per Symlink);
   Mutanten ausschließlich in einer Scratch-Kopie oder als sofort revertierter Edit
   mit `git status`-Beweis am Ende.
2. **Messungen vollständig berichten**: voller Output inkl. Exit-Code und
   Fehlerausgabe — nie nur die gegrepte Kennzahl. Eine „saubere Baseline" ist eine
   zu BEWEISENDE Randbedingung (`git status` des Messkorpus zeigen), keine Annahme.
3. **Jede Behauptung verankern** in einem selbst ausgeführten Kommando oder einer
   selbst gelesenen Datei. Fehlschlagende Tests als fehlschlagend berichten, mit
   Output. Nie „sollte funktionieren".
4. **Coverage-first**: jeden Befund mit Schweregrad (blocker/major/minor/nit) und
   Konfidenz berichten; gefiltert wird downstream, nicht beim Verifier.
5. **Scope**: nur die benannte Scheibe. Vorbestehende Probleme kurz als
   „out of scope, pre-existing" notieren, nicht als Blocker behandeln.
6. **Alle Kommandos vom Repo-Root** (cargo-Config-Discovery ist CWD-basiert).

## Baustein 0 — adversarialer Spec-Verifier (VOR der Implementierung)

Läuft nach der Owner-Richtungsabnahme einer bindenden Spec, BEVOR Code entsteht
(CLAUDE.md „Spec-Verifikation VOR der Implementierung", Owner-Mandat 2026-07-12).
EIN `brainroute:deep`-Agent, frischer Kontext. Es existiert kein Diff — Gegenstand
ist die SPEC gegen den ECHTEN bestehenden Code. Auftrag = das Design brechen, nicht
bestätigen. Zusätzlich zu den gemeinsamen Pflichtregeln:

- **Code-Annahmen der Spec verifizieren**: jede Behauptung der Spec über bestehende
  Symbole/Signaturen/Dateien am Code prüfen (gelesene Datei:Zeile). Eine falsche
  Annahme kippt eine Binding-Entscheidung → Blocker (genau der Item-10-Fund:
  `WNDArray.strides` ist ein Feld, keine Methode).
- **Covenant-Abgleich** (falls COVENANT.md existiert): verletzt die Spec SELBST eine
  Invariante oder implementiert sie ein Nicht-Ziel? Die berührten Invarianten-IDs
  benennen (sie wandern in die Spec und in die Delegations-Prompts der Scheibe).
- **Design-Löcher**: API-Form/Overload-Auflösung, Cross-Fall-Operanden (falsche
  Kombinationen), Fehler-/Env-Detektionspfade (erreichbar? statische vs. dynamische
  Imports, Plattform-Kontamination), Lifecycle-/Dispose-Fallen, Nachbar-Effekte.
- **Typ-Ebene adversarial**: Varianz (`out S`, Method-Shorthand-Bivarianz), Union-/
  Degradationskanten, Signatur-Kollisionen mit bestehenden Membern — bei Bedarf
  EMPIRISCH in einer Scratch-Kopie / eigenem worktree typchecken (nie im Haupt-Tree).
- **Testplan- & Freeze-Lücken**: fehlende Fälle (Rang 0, size-0, Views, Fehlerpfade,
  Lifecycle); baut der zugewiesene Test-Task überhaupt das nötige Artefakt (z. B.
  `build:wasm:threads` vs. `build:wasm`)? Hält die Freeze-Behauptung (fügt IRGENDEIN
  Teil Rust/ABI-Bedarf hinzu)?
- **Deliverable**: Report nach Kategorien (falsche Code-Annahmen / Design-Löcher /
  Testplan-Freeze-Lücken / Nits), je Befund Schweregrad + Konfidenz + Verankerung;
  die geprüften Annahmen, die HALTEN, knapp. KEINE Gates ausführen (es gibt keine
  Impl). Der Report IST die Deliverable — nicht auf einer Absicht/Nebensache enden.

Danach: der Orchestrator merged die Befunde, arbeitet Design-Blocker mit dem Owner
in die Spec ein (Richtungsänderungen abnehmen lassen), pinnt ein „Adversariale
Spec-Verifikation (Addendum)" in die Spec, DANN erst Implementierung.

## Baustein A — Spec-Verifier („entspricht es der Spec?")

Auftrag enthält zusätzlich:
- Die bindende Spec (`docs/<phase>-spec.md`) als Ground Truth; Intentions-Kontext
  (das Warum) mitgeben. Jede Binding-Entscheidung (D1…Dn) einzeln gegen den Diff
  prüfen; Spec⇄Impl-Drift ist ein eigener Befund.
- **Alle Gates frisch ausführen** und die echten Zahlen berichten (check-Verbund,
  test:core, test:resident, cargo, demo, ggf. test:threaded; Artefakt-Hash gegen
  den dokumentierten Pin).
- Disziplin-Prüfung am Diff: append-only-Dateien (runtime.ts, gefrorene
  Rust-Dateien) nur Additionen; TS-Klassenkörper insertion-only bzw. bestätigte
  Abweichung; Verhaltenserhaltung für Alt-Aufrufer.
- **Eigener Mutant (Pflicht)**: an einer selbst gewählten Stelle einen gezielten
  Fehler einbauen, beweisen, dass die neue Testabdeckung ihn fängt (welche
  Assertion?), revertieren, `git status`-Beweis. Bleibt die Suite grün → Befund
  der Stufe major/blocker (vakuöse Tests).

## Baustein B — adversarialer Verifier („wo bricht es trotzdem?")

Auftrag enthält zusätzlich (KEINE Spec-Konformitätsprüfung wiederholen — Auftrag
ist das Brechen):
- **Grenzfälle jenseits der Spec**: Rang 0, size-0-Dims, dynamische (nicht-literale)
  Achsen/Argumente auf dem Runtime-Pfad, Views (transponiert, offset-verschoben,
  komponiert), Fehlerpfad-Wortlaute Runtime ⇄ Compile.
- **Mutanten breit statt tief**: mehrere kleine Mutanten an Stellen, die die Spec
  NICHT erwähnt (z. B. Nachbar-Methoden, geteilte Helper, beide Zweige einer
  Fallunterscheidung) — erbt bewusst nicht die Fehlerhypothesen des Implementierers.
- **Messrandbedingungen angreifen**: Ist die Baseline wirklich sauber (untracked
  Dateien im Korpus?), ist der Vergleichskorpus derselbe, ist der Exit-Code 0,
  ist die Zahl über Läufe stabil?
- **Typ-Ebene adversarial**: Union-/Degradations-Kanten (dynamische Literale,
  boolean statt true/false, `number` statt Literal), Hover-Qualität der neuen
  Signaturen, `@ts-expect-error`-Positionen wirklich am Argument.

## Baustein C — covenant-verify (Vertrags-Dimension; gilt, solange COVENANT.md existiert)

Pflicht nur auf Stufe 3 der Covenant-Eskalationsleiter (substanzielle Scheibe mit
bindender Spec — Leiter in CLAUDE.md, „Qualitätssicherung"); auf Stufe 2 (Anker
berührt, keine Scheibe) nur bei inhaltlicher Tangierung, sonst Ein-Satz-Begründung im
Commit. Läuft PARALLEL zu A und B als DRITTER frischer Kontext mit disjunkter Frage:
„hält der Diff den STEHENDEN Vertrag?" — nicht die Scheiben-Spec (das ist A), nicht die
Korrektheit (das ist B). Dispatch über den `covenant:covenant-verify`-Agenten mit
GENAU diesem Input: (a) COVENANT.md wörtlich, (b) `git diff` der Scheibe, (c) die als
berührt identifizierten Invarianten-IDs, (d) `graph-a-lama query lint`-Output + knappe
Graph-Slices (`neighbors`/`impact`) der geänderten Knoten. KEINE Bewertung der eigenen
Arbeit mitgeben — der Agent urteilt allein aus Spec + Diff.

Arbeitsteilung (kein Doppeln): Verstöße gegen S-Invarianten stellt AUSSCHLIESSLICH
das mechanische `graph-a-lama query lint` fest — es läuft als eigenes Gate im
Gate-Block jeder Scheibe (Exit 1 = wie ein roter Test); der Agent prüft bei
S-Invarianten nur die Regel↔Spec-Verknüpfung (jede `covenant-*`-Regel ↔ ihr
Spec-Eintrag). Drift-Befunde — auch in der Richtung „die Spec ist das Veraltete" —
gehen unverdünnt an den Owner: Code-Fix ODER Spec-Änderung mit Version-Bump +
Changelog entscheidet der Owner, nie eine stille Anpassung.

## Abnahme

Alle Reports liegen vor (A + B, plus C in Covenant-Projekten) → Befunde mergen, jeden
major+ Befund adressieren oder begründet als akzeptiert dokumentieren; Ergebnisdoc
erhält ein Post-Verification-Addendum mit ALLEN Verdikten. Erst dann Commit.
