# dt3 — Reduktionen — bindende Spec (Stufe 3b)

**Version:** v1 (2026-10-05) · **Status:** Owner-abgenommen (A1, A2, A3 = Option (a), 2026-10-05) → Baustein 0
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

## Fakten am Code (2026-10-05, `main` = `126dee0`)

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
    Produkte jenseits 2^53 werden in f64 gerundet, nie gewrappt — NumPy wrappt hier.
  - **float32** (beide float32): je Ausgabeelement `acc = fround(acc + fround(a * b))`, k
    aufsteigend, Seed `+0`, kein FMA. `cosineSimilarity` = `fround(num / fround(fround(√nsqA) ·
    fround(√nsqB)))` mit `num`, `nsq` aus dem f32-Rechenweg (gleiche Komposition wie heute).
- **R4 `norm()`**: Rückgabe `number`. float64/int32: f64-Rechenweg (`Math.sqrt(normSqRuntime(f64))`,
  bestehende Funktion). float32: `fround(Math.sqrt(nsq32))` mit `nsq32 = fround(acc + fround(v·v))`
  aufsteigend. **bool: siehe A3.**
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
  `"float64"`.
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
> Quelltext an der Op dokumentiert. Anker ergänzt: `sym:ReduceDType`.

## Reihenfolge der Umsetzung (Arbeitsregel 21 — jeder Messpunkt ein eigener Commit)

1. **Commit A:** R1 + R2 (ohne Tests). check:diag messen, Zahl in die Commit-Message. **Stopp**,
   wenn Δ gegen 243,818 > **+1,500** — Owner-Rückmeldung statt Weiterbauen.
2. **Commit B:** R3 + R4 + R5. Messen. **Stopp**, wenn Δ kumuliert > **+2,500**.
3. **Commit C:** R6, die A2-Änderungen, R7 (FOLLOWUPS; Covenant-Text als eigener Commit vorab,
   Muster dt2). Zwischen B und C dürfen genau die A2-Tests rot sein; gepusht wird erst nach C.

## Gates (vorregistriert, Absolutwerte)

- check:diag Root **≤ +5,000** gegen 243,818 @ 140 **einschließlich Tests und Pins**, Dateiset
  unverändert (Tests an bestehende Dateien anhängen). Herleitung, offen: die Maschinerie ist
  Wiederverwendung (`PromoteDiv`) plus eine Tabelle nach bekanntem Muster, `DTypeLock`/
  `DTypeLockPair` verschwinden aus sechs Signaturen (Prototyp `sum`: −82); den Hauptteil erwarten
  wir aus Tests und Pins (dt2: +3,837 zwischen Commit B und dem Stand vor der Fix-Runde). Die
  Stopps nach A und B fangen eine Fehlschätzung der Implementierung ab.
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
