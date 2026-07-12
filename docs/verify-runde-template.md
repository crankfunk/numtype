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

## Abnahme

Beide Reports liegen vor → Befunde mergen, jeden major+ Befund adressieren oder
begründet als akzeptiert dokumentieren; Ergebnisdoc erhält ein
Post-Verification-Addendum mit BEIDEN Verdikten. Erst dann Commit.
