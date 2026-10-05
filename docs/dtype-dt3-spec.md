# dt3 — Reduktionen — bindende Spec (Stufe 3b)

**Version:** v1.1 (2026-10-05) · **Status:** Owner-abgenommen (A1, A2, A3 = Option (a), 2026-10-05), Baustein 0
gelaufen (kein Blocker, A2 empirisch exakt, Präzisierungen eingearbeitet) → **implementierungsreif**
**Stufe:** 3b — Covenant-Änderung (v10, A3); neue dtype-Regel auf bestehender Maschinerie-Klasse
(`ReduceDType` nach dem `Promote`-Muster), kein Kernel.
**Berührte Invarianten:** M2 (Reduktions-dtype, Union/`any`-Gate, niladische bool-Sperre), M3
(Meldungen an `matmul`/`dot`/`cosineSimilarity`), M1 (neue kernel-lose Referenzen, Tracking), Z1.
**Grundlage:** docs/dtype-design-spec.md v2.1 (E3, O1, D5 Zeilen `matmul`/`dot`/`sum`/`mean`/
`norm`/`cosineSimilarity`, D7, D8) · dt2 auf `main` (docs/dtype-dt2-ergebnisse.md, insbesondere G1:
`IsAnyDType` vor jedem Union-Gate mit Indexed-Access-Blatt). Prototyp-Messung Stufe 5 (`sum`):
−82 (docs/dtype-design-ergebnisse.md).

## Ziel

`sum`, `mean`, `matmul`, `dot`, `norm` — und `cosineSimilarity` (s. A1) — rechnen für float64,
float32 und int32; `sum`/`mean` zusätzlich für bool (Zählen bzw. Anteil, D7). Ergebnis-dtypes
zur Compile-Zeit berechnet, zur Laufzeit identisch, Typ- und Laufzeit-Tabelle aus EINER Quelle.
Nach dt3 bleiben nur `sqrt`, `argmax`, `topk`, `stack` für Nicht-float64 gesperrt (dt5). Keine
Veröffentlichung vor dt5.

## Fakten am Code (2026-10-05, gelesen an `126dee0`)

- Alle sechs Ops tragen heute die dt1-Sperre: `sum`/`mean` per `DTypeLock` im `ReduceAxis`-Guard
  (0-arg-Overloads niladisch, nur `assertFloat64Locked`), `matmul`/`dot`/`cosineSimilarity` per
  `DTypeLockPair` (ndarray.ts:461), `norm()` niladisch nur zur Laufzeit.
- `DTypeLockPair` hat genau diese drei Nutzer (ndarray.ts:1212/1267/1304). Nach dt3: **null**.
- Die O1-Tabelle (float32⊕float32 → float32, jede andere numerische Kombination → float64, bool →
  Ablehnung) ist **wortgleich `PromoteDiv`/`PROMOTE_DIV`** aus dt2 — inklusive `IsAnyDType`- und
  Union-Gate.
- Referenzen: `sumRuntime`, `meanRuntime` (= `sum/n`, S2-Pin), `matmulRuntime`, `dotRuntime`,
  `normSqRuntime` — alle `Float64Array`, aufsteigende Akkumulation, Seed `+0`.

## Umfang

- **R1 `ReduceDType<D>`** (E3) für `sum`/`mean`: float32 → float32, float64/int32/bool → float64.
  Laufzeit-Tabelle `REDUCE_DTYPE` an `runtime.ts` ANGEHÄNGT, der Typ liest sie per `typeof`
  (Muster `PROMOTE_DIV`). Gates in der dt2-Reihenfolge: `IsAnyDType` → `IsUnion` → Tabelle; `any`
  und Unions degradieren zu `DType` (kein Anspruch). Keine bool-Ablehnung — bool ist hier gültig.
- **R2 `sum`/`mean`** auf allen vier dtypes: Rückgabe `NDArray<OkShape<ReduceAxis<…>>,
  ReduceDType<D>>`; die `DTypeLock`-Komponente im `ReduceAxis`-Guard entfällt (Guard wie vor dt1).
  Laufzeit:
  - **float64-Rechenweg** (float64, int32, bool): Eingaben exakt nach f64 (int32 → f64 exakt, bool
    0/1), dann die BESTEHENDE `sumRuntime`/`meanRuntime` unverändert. float64-Eingaben werden nicht
    kopiert — der Pfad ist für float64 byte-gleich zu heute.
  - **float32-Rechenweg** (D8): `acc = Math.fround(acc + x)`, aufsteigend, Seed `+0`, je
    Ausgabeelement; `mean` = `Math.fround(sum32 / n)` (f64-Division des f32-Werts durch die exakte
    Ganzzahl `n`, dann `fround` — eindeutig definiert und in WASM als `f64.div` + `f32.demote`
    nachbaubar; Präzedenz `sum/n`, nie `sum*(1/n)`).
  - Die Validierungs-/Fehlertexte des f32-Rechenwegs (Achse außerhalb des Bereichs u. a.) sind
    wortgleich zu denen der bestehenden f64-Funktionen und werden je Fehler gegeneinander getestet
    (v1.1, Baustein 0 D5: der f32-Weg muss sie kopieren, weil die alten Funktionen byte-gleich
    bleiben).
  - Offengelegt: eine int32-Summe in f64 ist exakt, solange jede Teilsumme ≤ 2^53 bleibt (sicher
    bis 2^22 Elemente); darüber deterministisch gerundet, nie Wrap. bool-`sum` zählt, bool-`mean`
    ist der Anteil. size-0 wie heute (`sum` → 0, `mean` → NaN) in jedem dtype.
- **R3 `matmul`/`dot`/`cosineSimilarity`** (O1): der Guard prüft `PromoteDiv<D, Dd>` statt
  `DTypeLockPair` (Shape-Fehler zuerst, wie bei `div`); bool als Empfänger oder Argument → eigene
  Meldung `BOOL_ARITHMETIC_MESSAGE` am Argument, wortgleich zur Laufzeit. `matmul` →
  `NDArray<OkShape<MatMul<S, B>>, OkDType<PromoteDiv<D, Dd>>>`; `dot`/`cosineSimilarity` →
  `number` wie heute. **`PromoteDiv` wird wiederverwendet, keine neue Tabelle** (sie ist O1
  wortgleich; ein zweiter Name kostete Instantiations ohne Inhalt). Laufzeit:
  - Rechen-dtype = `promoteDTypeDiv(D, Dd)`. **float64**: beide Operanden exakt nach f64, dann die
    BESTEHENDE `matmulRuntime`/`dotRuntime`/Kosinus-Formel unverändert. Offengelegt (O1): int32-
    Produkte jenseits 2^53 werden in f64 gerundet, nie gewrappt — NumPy wrappt hier. Dasselbe gilt
    für die Quadrate in `norm()`/`cosineSimilarity` auf int32 (v1.1, Baustein 0 D6).
  - **Prüfreihenfolge zur Laufzeit = Compile-Zeit** (v1.1, Baustein 0 D4): Shape-Fehler zuerst,
    dann die bool-Ablehnung — sind beide falsch, wirft die Laufzeit die Shape-Meldung, wie der
    Editor sie zeigt. Per Test gepinnt.
  - **float32** (beide float32): je Ausgabeelement `acc = fround(acc + fround(a * b))`, k
    aufsteigend, Seed `+0`, kein FMA. `cosineSimilarity` = `fround(num / fround(fround(√nsqA) ·
    fround(√nsqB)))` mit `num`, `nsq` aus dem f32-Rechenweg (gleiche Komposition wie heute).
- **R4 `norm()`**: Rückgabe `number`. float64/int32: f64-Rechenweg (`Math.sqrt(normSqRuntime(f64))`,
  bestehende Funktion). float32: `fround(Math.sqrt(nsq32))` mit `nsq32 = fround(acc + fround(v·v))`
  aufsteigend. **bool: siehe A3.**
- **Doc-Kommentare** der Ops nennen die Folge der aufsteigenden f32-Akkumulation (v1.1, Baustein 0
  D7: n = 2^24+8 Einsen ergeben sum 16,777,216 und mean 0.99999952 — bewusst, paarweise Summation
  ist Nicht-Ziel).
- **R5 Aufräumen:** `DTypeLockPair` wird gelöscht (null Nutzer, toter Code; ob die Löschung
  messbar spart, misst Baustein 0 — ungenutzte generische Aliase sind auf TS 7 nicht gratis,
  CLAUDE.md „Key TS limits"). `DTypeLock` und
  `assertFloat64Locked` bleiben (argmax/topk/sqrt/stack bis dt5).
- **R6 Tests und Pins:** Ergebnis-dtype-Tabelle vollständig (Typ-Pins + Laufzeit, beides gegen
  `REDUCE_DTYPE`/`PROMOTE_DIV`), 0-arg/Achse/keepdims je dtype; float64-Pfade gegen die alten
  Funktionen als Orakel (auch für konvertierte int32/float32/gemischte Eingaben); float32
  bit-identisch gegen eine UNABHÄNGIGE `Float32Array`-Zuweisungs-Referenz inkl. ±0, ±Inf, NaN,
  Subnormalen und f32-Überlauf nach Inf, den f64 nicht hätte; int32 große Werte (Summe > 2^31 ohne
  Wrap; `matmul`/`dot` mit Produkten > 2^53 gegen ein BigInt-Orakel mit f64-Rundung); bool
  Zählen/Anteil; Views per `wideSpecs` (Arbeitsregel 16) mit Klassen-Assertion (Regel 12);
  Union-/`any`-Pins (`NDArray<S, DType>`, `"float32" | "int32"`, `AnyNDArray`) für `sum`/`mean`/
  `matmul`, je mit Mutant-Nachweis für `IsAnyDType` und das Union-Gate in `ReduceDType`;
  Diagnose-INHALTE per tsc-Fixture (Arbeitsregel 2); Hovers per LSP (Regel 13): `i32.sum()` →
  `NDArray<[], "float64">`, `f32.mean(0)`, `f32.matmul(f32)` → `"float32"`, `f32.matmul(i32)` →
  `"float64"`. **Ergänzt v1.1 (Baustein 0):** Rang 0 und size-0 je dtype (inkl. `matmul` mit
  k = 0, leere `cosineSimilarity` → NaN); Klasse von `data` (`Float32Array` bei float32) bei jeder
  Reduktion und keepdims; gemischte dtypes über `matmul`-Batch-Broadcasting und 1-D-Promotion in
  beide Richtungen; BigInt-Orakel für int32 rundet je Schritt (Produkt, dann Summe, je auf f64);
  f32-`cosineSimilarity` mit nsq-Unterlauf zu 0 und -Überlauf zu Inf; f64-Orakel für
  `norm`/`cosineSimilarity` im Test nachgebaut (die alte Formel steht inline in der Methode);
  `norm()` auf bool nicht-leer und size-0, dazu LSP-Pin „keine Diagnose“; Prüfreihenfolge (D4) und
  Meldungsgleichheit f32/f64 (D5). Der Union-Gate-Mutant in `ReduceDType` beweist PRÄZISIONS-
  Konsistenz (ohne Gate verteilt sich die Tabelle korrekt über die Union), keine Soundness — im
  Test so benennen; der `IsAnyDType`-Mutant ist der Soundness-Nachweis (G1-Klasse). Mutanten
  bevorzugt auf einer Kopie von `spike/src` AUSSERHALB des Repos (umgeht Arbeitsregel 20 ganz).
- **R7 Covenant v10** (A3), FOLLOWUPS (neue kernel-lose Referenzen nach M1 v5; dt1-K7-Eintrag:
  die niladischen Lücken von `sum()`/`mean()`/`norm()` schließen sich für float32/int32).

**Nicht in dt3:** `WNDArray`/WASM/Threaded (D9, dt6+) · Vergleiche, `where`, `any`/`all` (dt4) ·
`sqrt`/`argmax`/`topk`/`stack` (dt5) · paarweise Summation oder andere Genauigkeits-Algorithmen
(Bit-Identity-Law: aufsteigend) · README (Regel 19, erst mit dem Release).

## Owner-Abnahmen vor Baustein 0

**A1 — Methodenliste nach Hausregel (b):** an `NDArray` Signaturen und Körper von `sum` (4
Overloads), `mean` (4), `matmul`, `dot`, `norm`, **`cosineSimilarity`** — Letztere steht nicht in
der Roadmap-Liste, aber D5 führt sie mit `norm`, sie ist aus `dot`/`norm` komponiert und hätte nach
dt3 als einzige Op ohne Scheibe gesperrt bleiben müssen. Auf Modulebene: neu `ReduceDType`,
gelöscht `DTypeLockPair` (R5). In `runtime.ts` nur Anhänge (`REDUCE_DTYPE`, `reduceDType`, die
dtype-generischen Dispatcher und f32-Rechenwege). Bestehende Referenzfunktionen byte-gleich.

**A2 — bestehende Tests, die sich zwangsläufig ändern** (alle in
`spike/tests-runtime/scalar-mean.test.ts`; Baustein 0 bestätigt die Liste empirisch, wie in dt2):
1. „matmul/dot/cosineSimilarity: throw the locked message …" → durch Tests der neuen Semantik
   ERSETZT; die bool-Fälle bleiben als Sperr-Tests mit `BOOL_ARITHMETIC_MESSAGE`.
2. „sum/mean: both the 0-arg and axis-bearing forms throw …" → ersetzt (keine Sperre mehr, auch
   nicht für bool).
3. „sqrt/norm: niladic locked ops throw …" → die `norm`-Zeilen entfallen, `sqrt` bleibt bis dt5;
   `norm` auf bool bekommt einen eigenen Test nach A3.
4. „a locked two-operand op with TWO DIFFERENT non-float64 dtypes …" → **entfällt ersatzlos**:
   nach dt3 gibt es keine gesperrte Zwei-Operanden-Op mehr, und die bool-Meldung nennt keinen
   dtype, also gibt es keine Reihenfolge mehr zu beweisen.
5. „DTypeLockPair (F2 pin) …" → der Mutanten-Teil entfällt mit seinem Gegenstand (R5); der
   Cross-Layer-Teil (Editor zeigt `lockedOpMessage` wortgleich) zieht auf `argmax(0)` um (bleibt
   per `DTypeLock` gesperrt bis dt5 — dort zieht er erneut um oder entfällt).

**A3 — `norm()` auf bool und Covenant v10.** `norm()` ist niladisch: es gibt keine Argumentposition
für eine Compile-Ablehnung, und O2(b) (`this`-Parameter) ist ausgeschlossen. dt5 trifft dasselbe bei
`sqrt()` auf bool. Optionen:
- **(a) dauerhafte Laufzeit-Sperre — ENTSCHIEDEN 2026-10-05** — konsequent zu E5/D7 („Arithmetik braucht
  `astype`"); `norm()` auf bool kompiliert, wirft zur Laufzeit `BOOL_ARITHMETIC_MESSAGE`. M2 v8
  deckt das heute nur ÜBERGANGSWEISE ab; dauerhaft braucht es den Text unten.
- (b) erlauben: `norm()` auf bool = √(Anzahl true), wie NumPy (`linalg.norm` castet bool nach
  float). Kein Covenant-Text nötig, aber ein Abweichen von E5, und dt5 muss für `sqrt` dieselbe
  Frage neu stellen.

Wortlaut bei (a):

> **M2 · Präzisierung v10 — Reduktionen (Owner-entschieden 2026-10-05):** der Ergebnis-dtype einer
> Reduktion (`ReduceDType`: float32 → float32, sonst float64) und die Promotion von
> `matmul`/`dot`/`cosineSimilarity` (`PromoteDiv`, O1) sind dtype-Maschinerie nach v8/v9: `any` und
> Unions degradieren zu `DType`, Typ- und Laufzeit-Tabelle stammen aus einer Quelle. Eine
> NILADISCHE Op, die bool dauerhaft ablehnt (ab dt3 `norm()`), lehnt nur zur Laufzeit ab, mit
> `BOOL_ARITHMETIC_MESSAGE`; sie gilt dauerhaft als „unvollständig, nicht falsch" und ist im
> Quelltext an der Op dokumentiert — anders als die übergangsweisen Sperren nach v8 braucht sie
> keinen FOLLOWUPS-Eintrag, weil nichts nachzuziehen ist. Anker ergänzt: `sym:ReduceDType`.

(v1.1: der Halbsatz zu FOLLOWUPS ist auf Empfehlung von Baustein 0 ergänzt — v10 ersetzt sonst das
„übergangsweise" aus v8 stillschweigend. Owner-Bestätigung des ergänzten Wortlauts ausstehend.)

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit)

1. **Commit A:** R1 + R2 (ohne Tests). check:diag messen, Zahl in die Commit-Message. **Stopp**,
   wenn Δ gegen 243,818 > **+1,500** — Owner-Rückmeldung statt Weiterbauen.
2. **Commit B:** R3 + R4 + R5. Messen. **Stopp**, wenn Δ kumuliert > **+2,500**.
3. **Commit C1** (v1.1, Baustein 0 D2 — der Testanteil ist der eigentliche Kostenblock und
   braucht einen eigenen Messpunkt): die A2-Änderungen + Laufzeittests aus R6. Messen. **Stopp**,
   wenn Δ kumuliert > **+2,000** (für die Typ-Pins hat Baustein 0 rund +3,400 gemessen).
4. **Commit C2:** die Typ-Pins aus R6 + R7 (FOLLOWUPS; Covenant-Text als eigener Commit vorab,
   Muster dt2). Messen gegen das Gate.
5. **Commit D:** bench:editor-Re-Pin (v1.1, Baustein 0 D1: der Entwurf reißt alle 8 Pins, w5 mit
   umgekehrtem Vorzeichen; Präzedenz `854dc2c`), zweimal gemessen, jede Verschiebung im Projekt-Log
   erklärt — insbesondere w5.
Zwischen B und C1 dürfen genau die A2-Tests rot sein; gepusht wird erst nach D.

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +5,000** gegen 243,818 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert (Tests an bestehende Dateien anhängen). Herleitung, offen: die Maschinerie ist
  Wiederverwendung (`PromoteDiv`) plus eine Tabelle nach bekanntem Muster, `DTypeLock`/
  `DTypeLockPair` verschwinden aus sechs Signaturen (Prototyp `sum`: −82); den Hauptteil erwarten
  wir aus Tests und Pins (dt2: +3,837 zwischen Commit B und dem Stand vor der Fix-Runde). Die
  Stopps nach A und B fangen eine Fehlschätzung der Implementierung ab. Baustein 0 hat gemessen:
  Entwurf R1+R2 −689, R1–R5 −108 (zweimal identisch; Löschung `DTypeLockPair` allein −10), plus
  rund 90 Typ-Pins +3,394.
- stress/browser berichten · bench:editor: Pins nur bei erklärter Verschiebung neu setzen, zweimal
  messen · Freeze-Hash unverändert (kein Rust) · alle übrigen bestehenden Tests unverändert grün
  (außer A2) · Lint auf frisch gebautem Graph · vor jedem Push Autorenprüfung (Arbeitsregel 22).

## Verify

Baustein 0 (`brainroute:deep`, eigener Worktree) vor dem Code: A2 empirisch vollständig, Entwurfs-
messung, `ReduceDType`-Gates ohne Loch (`any`, Unions, `AnyNDArray`), D8-Formeln für f32 gegen
eine unabhängige Referenz, A3-Verhalten im Editor (LSP). Danach A + B + C parallel (A auf deep).
B bekommt ausdrücklich: f32-Bit-Identität gegen unabhängige Referenz, float64-Pfade bitgleich zu
vorher, Union-/`any`-Löcher (G1-Klasse), int32-Großwerte ohne Wrap, Achsen-Reduktionen über Views.
Jeder Agent-Auftrag trägt Arbeitsregel 22 (nie `git config`).

## Adversariale Spec-Verifikation (Addendum, Baustein 0, 2026-10-05)

`brainroute:deep`, eigener Worktree, Entwurf von R1–R5 für alle sechs Ops gebaut (Patch außerhalb
des Repos), echter tsc 7.0.2 und `tsc --lsp --stdio`. **Kein Blocker, keine falsche Code-Annahme.**
Baseline reproduziert (243,818 @ 140, stress 119,393, browser 2,142, bench:editor, Freeze).
**A2 empirisch exakt:** test:core auf dem Entwurf 1627/1632, genau die fünf benannten Tests rot;
resident 6155+2, package 3/3, `pnpm check` grün, kein `.test-d.ts`-Pin bricht. **Messung:** R1+R2
−689, R1–R5 −108, stress −130, browser ±0, Freeze unverändert; bench:editor reißt alle 8 Pins (D1).
**Typebene:** Tabelle, Degradation (`AnyNDArray`, `DType`, drei Unions), Overload-Auflösung von
`sum`/`mean` inkl. `0 | undefined` und `true | undefined`, `matmul`-Batch-Broadcast und
1-D-Promotion mit gemischten dtypes, Fehler am Argument mit Shape vor bool, Hovers wie in R6
erwartet. **D8:** 45,608 Fälle gegen eine `Float32Array`-Zuweisungsreferenz und rund 796,000 gegen
eine exakte BigInt-Rundung, null Abweichungen; `mean = fround(sum32 / n)` ist für n > 2^24
eindeutig und in WASM nachbaubar (`fround(n)` als Divisor wäre es nicht). **A3:** `boolArr.norm()`
ohne Diagnose, wirft zur Laufzeit (auch size-0); keine Kollision mit M2/M3. **Eingearbeitet:**
D1 Re-Pin-Commit, D2 Commit C geteilt mit Stopp, D3 Mutanten-Benennung, D4 Prüfreihenfolge
Laufzeit = Editor, D5 Meldungsgleichheit f32/f64, D6 int32-Offenlegung für `norm`/
`cosineSimilarity`, D7 Doc-Kommentar zur f32-Sättigung, Testplan-Ergänzungen, v10-Halbsatz.
**Nicht übernommen:** D8 (Rest-Parameter-Sperre für `norm()` auf bool) — ungemessen, kostet
Hover-Klarheit, Owner hat (a) entschieden.
