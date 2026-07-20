# Op-Scheibe W3: `sqrt` — bindende Spec

Status: **bindend** (Owner-Programm 2026-07-20: W2–W5 gesetzt; Evidenz
docs/dogfooding-rag-ergebnisse.md W3/F1 — zweifacher Bruch der natürlichen
L2-Normalisierungs-Kette in der RAG-Demo).
Version: 1 · Datum: 2026-07-21 · Eskalationsleiter: **Stufe 3** (voller Katalog;
erwartbar die kleinste W-Scheibe — die Verifier-Aufträge skalieren entsprechend).
Covenant: v5 (kernel-lose Referenz-Ops zulässig, Paritätslücke in FOLLOWUPS).

## Ziel & Warum

`corpusSumSquares.sqrt()` statt Hand-Loop über `.data` — der letzte fehlende Schritt
der Kette `mul → sum(axis) → sqrt → reshape → div`, die seit W2 ansonsten vollständig
in numtype formulierbar ist.

## Berührte Covenant-Invarianten

- **M1 (v5):** kernel-lose Referenz-Funktion in runtime.ts (Append); FOLLOWUPS-
  Paritäts-Item wird um sqrt erweitert; Hash byte-identisch.
- **Nicht-Ziel „Transzendente":** `sqrt` fällt NICHT darunter — IEEE 754 verlangt für
  sqrt KORREKTE RUNDUNG (wie +,−,×,÷), anders als für exp/log/sin; `Math.sqrt` und
  `f64::sqrt` sind damit bit-deterministisch. Diese Einordnung wurde bereits vom
  covenant-verify der Dogfooding-Scheibe explizit als konform bestätigt (W3-Eintrag der
  Wunschliste). exp/log/sin bleiben AUSSEN.
- **M2/M3:** keine neue Fehler-Fläche — `sqrt()` ist niladisch (norm()-Präzedenz: kein
  Argument, kein Guard, kein Compile-Claim); negative Inputs liefern NaN (IEEE), kein
  Throw. **M5/Z1/Z2/M4:** unberührt.

## Bindende Entscheidungen

- **D1 — Scope:** NUR `sqrt(): NDArray<S>` auf `NDArray` (shape-erhaltend, jede
  Rang-Stufe inkl. Rang 0). KEINE weiteren Unary-Ops in dieser Scheibe (abs/neg wären
  gleichfalls exakt, haben aber keine Demo-Evidenz — Nicht-Ziel; die Runtime-Form
  unten macht spätere Ergänzungen billig). Kein generisches `.map(fn)` (nicht
  kernel-spiegelbar, Determinismus-Hintertür — Wunschlisten-Begründung W3).
- **D2 — Runtime:** Append in runtime.ts: unäre Referenz in der W2-Dispatcher-Form
  (`unaryElementwiseRuntime(op: "sqrt", data): Float64Array` oder funktional
  äquivalente Einzel-Funktion — Stil analog scalarElementwiseRuntime, Implementierer-
  Wahl): elementweise `Math.sqrt(data[i])` aufsteigend, frisches Float64Array.
  GEPINNTE IEEE-Kanten (Tests): `sqrt(-0) === -0` (IEEE!), `sqrt(-x) → NaN` für x>0,
  `sqrt(NaN) → NaN`, `sqrt(Infinity) → Infinity`, Subnormals exakt (Bit-Vergleich
  gegen `Math.sqrt`-Referenz — die Op-Definition IST `Math.sqrt`, der Test prüft
  Durchreichung, nicht Mathematik), size-0 → leeres Array.
- **D3 — Methode:** `sqrt(): NDArray<S>` als Klassenkörper-Append nach `mean`
  (Insertion-only — neuer Member-Name, kein Overload-Thema, keine D6-W2-Problematik).
  Doc-Kommentar: IEEE-Exaktheits-Begründung (Abgrenzung zum Transzendenten-Non-Goal),
  norm()-Verweis (niladisch = kein Guard), NaN-für-negativ-Offenlegung,
  Surface-Hinweis (NDArray-only, no WASM kernel yet). NDArrayView bleibt unberührt.
- **D4 — Tests:** Runtime-Tests als klar abgesetzter W3-Block APPENDED an das
  BESTEHENDE spike/tests-runtime/scalar-mean.test.ts (KEIN neues File — File-Set
  bleibt fix, Pin-Delta = reine Inhaltskosten; die Datei ist bereits in test:core
  registriert, der Dateikopf-Kommentar wird um den W3-Block-Hinweis ergänzt):
  D2-Kanten-Matrix, Rang 0/1/2, Shape-Erhalt, transponierte/gesliceste Empfänger,
  randomisierter Bit-Differential gegen direkte Math.sqrt-Schleife, Kette
  `mul→sum(1)→sqrt→reshape→div` als End-to-End-Rückprobe gegen die alte
  Hand-Loop-Formulierung (byte-identisch — das ist die Friction-F1-Schließung).
  Typ-Pins APPENDED an ndarray.test-d.ts (klein): exakter Shape-Erhalt (Equal),
  Rang 0, wide, Readonly-S, dynamic rank.
- **D5 — Gates & Pins:** wie W2-D8; Baseline = aktuelle Pins (188,563 @ 137 ·
  104,900 @ 82 · 2,142 @ 75), KEIN neues File → kein empty-then-fill nötig, Deltas
  je 2× deterministisch + attribuiert; Absolut-Gate Haupt-Pin ≤ **+3,000** (kleinste
  Scheibe; Typ-Ebene ist ein einziger shape-erhaltender Member); stress-Ripple
  akzeptabel wie gehabt; bench:editor-Pins bei Abweichung 2× messen + aktualisieren.
  Sprache/`≈`/GFM wie immer. **Mutanten-Regel verschärft (Lehre aus W2):**
  Haupt-Tree-Mutanten NUR mit vorheriger Backup-Kopie (`cp` nach /tmp) und Restore
  aus der Kopie — NIE `git checkout`/`git restore`.

## Akzeptanzkriterien

- **T1:** Alle Gates grün (check/diag×3/test:core/test:resident/cargo/freeze/
  bench:editor/lint/test:example/GFM), Hash byte-identisch, Exit-Codes berichtet.
- **T2:** Pin-Protokoll: Deltas 2× deterministisch, Absolut-Gate ≤ +3,000 eingehalten,
  stress/browser-Werte berichtet.
- **T3:** IEEE-Kanten-Pins vollständig (−0/NaN/−x/Inf/Subnormal-Bits/size-0) und per
  Verify-Mutant nachweisbar nicht-vakuös; F1-Schließungs-Rückprobe byte-identisch.
- **T4:** Typ-Pins: Shape-Erhalt exakt inkl. Rang 0/wide/Readonly-S; keine neue
  Degradationskante (es gibt keine).
- **T5:** Datei-Disziplin: runtime.ts Append, ndarray.ts Klassenkörper-Insertion,
  KEIN neues File, test-scripts-guard grün; FOLLOWUPS-Paritäts-Item um sqrt erweitert;
  Doc-Platzierung nach Hausregel (CLAUDE.md-Zahlen, Projekt-Log-Append, README-Notiz
  der W1/W2-Ops um sqrt ergänzt — bit-for-bit-Zeile bleibt wahr).

## Nicht-Ziele

Kein abs/neg/square/rsqrt (keine Evidenz), kein `.map(fn)`, keine transzendenten Ops,
kein WNDArray/Threaded (FOLLOWUPS), kein Kernel, kein Release in dieser Scheibe.

## Verify-Plan (Stufe 3, klein skaliert)

Baustein 0 (brainroute:deep, KOMPAKT — Schwerpunkte: Namens-/Symbolkollisionen `sqrt`
im Repo, Math.sqrt-IEEE-Exaktheits-Behauptung gegen Primärquellen-Stand der KB/Spec,
Test-Append-Entscheidung D4 gegen den test-scripts-guard und die Dateikopf-Semantik,
Absolut-Gate-Plausibilität) VOR der Implementierung. Danach A+B+C parallel mit
entsprechend kompakten Aufträgen. Ergebnisse-Doc mit Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-21)

Verifier: brainroute:deep, kompakt skaliert, zwei Scratch-Worktrees (Haupt-Tree
unberührt). **Kein Blocker — alle sechs Prüfpunkte halten**, zwei mit harter Evidenz:
(1) IEEE-Abgrenzung primärquellen-verankert: ECMA-262 `sec-math.sqrt` definiert
Schritt 4 über die exakte 𝔽-Rundung (wie +,−,×,÷); JEDE transzendente Math-Methode
trägt stattdessen wörtlich „implementation-approximated" — sqrts Ausschluss aus dem
Transzendenten-Non-Goal ist damit belegt, nicht behauptet; empirisch alle D2-Kanten
in Node 24 bit-exakt bestätigt (inkl. sqrt(-0) === -0). (2) Absolut-Gate GEMESSEN:
Baseline 188,563 exakt reproduziert; Probe-Worktree mit D3/D4-förmigem Append =
188,587 → Delta +24 (Typ-Ebenen-Anteil; Runtime-Testblock-Kosten folgen bei der
Implementierung). Weiter: keine sqrt-Symbolkollisionen repo-weit; test-scripts-guard
inhaltsblind (Append sicher); Typ-Probe mit 5 Shape-Pins grün inkl. Chaining und
nicht-vakuösem sqrt(1)-Fehler-Pin; M1-v5-FOLLOWUPS-Mechanik (Nachtrag-Muster am
Paritäts-Item) etabliert. Offen markiert (unkritisch): f64::sqrt-Rust-Seite
(erst ab Kernel bindend), Testblock-Anteil am Pin.
