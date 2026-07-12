# Kern 11 — elementwise contiguous Fast-Path — Ergebnisse

*Phase B, Item 7 (Perf-Kür, fokussierter Scope). Datum: 2026-07-12. Spec:
docs/kern-11-elementwise-fastpath-spec.md. Zweifach verifiziert (Spec-Verifier
CONFIRMED + adversarialer Verifier HÄLT).*

## Honesty rule
Jede Zahl hier ist ein selbst (oder vom Verifier unabhängig) ausgeführtes
Kommando, kein „sollte". Wo etwas nur teilweise gilt, steht es teilweise da.

## Summary
`add_strided` (add.rs) und `binary_strided` (elementwise.rs, deckt sub/mul/div)
bekommen je einen **Contiguous-Fast-Path**: wenn beide Operanden dieselbe Shape
haben (kein Broadcast), offset 0 sind und natürliche C-contiguous Strides tragen,
gilt `flat == a_off == b_off` für jedes Element — der allgemeine Loop reduziert
sich dann auf `out[i] = op(a[i], b[i])`, und die per-Element-`unravel`-Allokation
(shape.rs:105, `.collect()` in einen `Vec` — >1 Mio. Allokationen bei 1024×1024)
entfällt. **Gemessen 13–17× schneller** auf dem contiguous Hotpath, **bit-identisch**
zum allgemeinen Pfad (inkl. Spezialwerte).

Der Fast-Path ist die MESSGETRIEBENE Wahl: SIMD-elementwise (der ursprüngliche
Item-7-Kandidat) ist gemessen memory-bound und wertlos; die `unravel`-Allokation
war der eigentliche Hebel (Messgrundlage in der Spec, „Messgrundlage").

## Was gebaut wurde
- **Bedingung** (identisch in beiden Funktionen, NACH allen Validierungen —
  Fehler-Semantik Status 1/2/3/4 byte-für-byte unverändert):
  `a_shape == b_shape && a_offset == 0 && b_offset == 0 && a_strides ==
  compute_strides(a_shape) && b_strides == compute_strides(b_shape)`. Trifft sie
  zu: einmal `&a_data[..size]`/`&b_data[..size]` slicen (ein Bounds-Check statt
  size-viele → LLVM vektorisiert), dann flacher Loop. Sonst der bestehende
  allgemeine `unravel`-Loop UNVERÄNDERT.
- **Bit-Identität, bewiesen (nicht gehofft):** für natürliche Strides ist die
  Mixed-Radix-Zerlegung von `unravel` per Konstruktion `idx·natural_strides ==
  flat` für JEDES flat — der adversariale Verifier hat das mathematisch
  nachgezogen (strukturell airtight, kein False-Positive der Bedingung möglich,
  unabhängig von size-0/size-1/Rang). Empirisch: zwei committete cargo-Tests je
  Datei vergleichen `.to_bits()` von Fast-Path- vs. general-path-Zwilling
  (offset-Fenster bzw. transponierter View, mit NaN/±Inf/±0).
- **`compute_strides`, NICHT die `_into`-Twins:** die no-alloc-Twins
  (`compute_strides_into`/`unravel_into`) sind `#[cfg(… atomics)]`-gegated und im
  Plain-wasm32-Artefakt gar nicht kompiliert — ihre bloße Präsenz verschöbe Bytes
  (shape.rs:208-220). Der Fast-Path nutzt das allozierende `compute_strides` (eine
  Allokation pro Call ist irrelevant — das Problem war per-ELEMENT); der
  Fast-Path-Loop braucht ohnehin kein `unravel`.
- **v1 unberührt:** `add()` (nicht `add_strided`) quelltextlich byte-identisch
  (39/39 Zeilen, direkt gedifft HEAD vs. Working); abi.rs/matmul_blocked.rs/
  shape.rs/vector.rs-Diffs leer.

## Freeze-Zeremonie (vierteilig, re-pinnt den Artefakt-Hash)
1. Pre-Edit Clean-Rebuild (isolierter Kern-10-Worktree e64aed6) reproduziert den
   ALTEN Pin `7a65d80062865a5e88952ce3cfbdd974b642f6d3f4b293e3f3b39afad16885d8`
   exakt — von Executor UND beiden Verifiern unabhängig.
2. Post-Edit Clean-Rebuild → NEUER Pin
   `0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`,
   reproduzierbar (auch nach `#[cfg(test)]`-Ergänzungen unverändert — Test-Code
   leckt nicht ins Release-wasm, selbst geprüft).
3. Verhaltens-Pins grün (siehe Gates).
4. Mutations-Beweise: Executor (`a[i].max(0.0)` in beiden Fast-Paths) → 3/159
   cargo + 158/4267 TS rot; Spec-Verifier (Argument-Swap `op(b,a)`) → 5 cargo +
   45 TS rot; adversarial (4 injizierte + 3 eigene Tests). Alle Reverts grünen.

**Freeze-Präzisierung (adversarialer Befund 1, wichtig für die Ehrlichkeit):**
Ein `wasm2wat`-Funktionsdiff zeigt, dass PRAKTISCH JEDE Funktion im Crate —
inklusive des „unberührten" v1 `add()`, matmul, transpose, sum — Byte-Shifts hat.
Der Diff besteht in JEDEM Fall AUSSCHLIESSLICH aus `i32.const`-Operanden
(Panic-Location-Datenzeiger, konstant um ~96 B verschoben durch eine neue
`memcmp`-Hilfsfunktion, die der Compiler für die drei neuen Slice-`==`-Vergleiche
generiert) — **null Opcode-/Logikänderungen**. Konsequenz: „v1 add unberührt"
ist eine QUELL- und LOGIK-Aussage, KEINE Artefakt-Byte-Aussage. Genau deshalb IST
der Freeze-Beweis ein Ganz-Artefakt-Clean-Rebuild-Hash und keine
Per-Funktions-Byte-Behauptung — der WAT-Diff (nur `i32.const`, null Opcodes) ist
starke Zusatzevidenz, dass nirgends Logik wanderte, nicht eine Schwäche.

## Gates (alle mit Exit-Code geprüft; Executor + beide Verifier unabhängig)
- `pnpm check` (Verbund root+stress) — Exit 0.
- `pnpm test:core` — 817/817.
- `pnpm test:resident` — 4267 total, 4265 pass, 2 skip, 0 fail.
- `cargo test` — **161** passed (157 Kern-10-Basis + 2 Executor-Pfad-Äquivalenz
  + 2 adversarial-Follow-up, s. u.), 0 fail.
- `pnpm check:diag` — **172,392 @ 128 Files** (s. Anomalie unten).
  `check:diag:stress` — 94,597 unverändert.
- `pnpm demo` — alle drei Backends stimmen überein (sub/mul/div treffen jetzt den
  Fast-Path).
- Artefakt-Hash `0b9df4f1…` reproduzierbar.

## Zwei-Verifier-Runde (docs/verify-runde-template.md)
- **Spec-Verifier: CONFIRMED.** D1–D6 wortgleich PASS, v1 byte-identisch, beide
  Hash-Reproduktionen selbst nachvollzogen, alle Gates exakt, eigener
  Argument-Swap-Mutant nicht-vakuum.
- **Adversarialer Verifier: HÄLT.** Bit-Identität mathematisch bewiesen + 4
  Mutanten + 3 eigene Grenzfall-Tests; check:diag-Anomalie in Isolation
  reproduziert; Bench-Routing + Bit-Gate-Schärfe bestätigt; Freeze via WAT-Diff
  seziert. Drei verwertbare Befunde:
  1. Freeze-Doku-Präzisierung (oben eingearbeitet).
  2. **Coverage-Lücke, in dieser Scheibe geschlossen:** `cargo test` allein fing
     die *Kanonizität* der Guard-Bedingung nicht — ein Mutant `a_strides ==
     b_strides` (statt `== compute_strides(a_shape)`) überlebt alle 159 alten
     cargo-Tests (nur die TS-`strided.test.ts` fing ihn). Zwei gleich-transponierte
     Operanden würden den Fast-Path fälschlich triggern. → neuer cargo-Test
     `binary_strided_fast_path_requires_natural_not_just_equal_strides`.
  3. **Coverage-Lücke, in dieser Scheibe geschlossen:** Fast-Path bei
     gleicher-Shape + size-0-Dim bzw. size-1-Dims in Nicht-Trailing-Position war
     ungetestet (Differential-Generator schließt Dim 0 aus; bestehende
     size-0-cargo-Tests nutzen UNGLEICHE Shapes → Broadcast, nie den Fast-Path).
     Bewiesen korrekt. → neuer cargo-Test
     `binary_strided_fast_path_size_zero_and_interleaved_size_one`.

Beide neuen Tests sind test-only (`#[cfg(test)]`), Artefakt-Hash danach unverändert
`0b9df4f1…` (selbst per Rebuild geprüft) — kein Freeze-/check:diag-Einfluss.

## check:diag-Anomalie — Mechanismus GEPINNT (2026-07-12, Nachtrag)
Das Hinzufügen der Bench-Datei `spike/bench-core/elementwise.ts` SENKT `check:diag`
von 174,391 @ 127 auf **172,392 @ 128** — ein Rückgang trotz +1 Datei. Zunächst als
„Mechanismus opak" dokumentiert; auf Owner-Wunsch nachträglich per kontrollierter
Bisektion (Scratch-Kopie, eine Variable pro Schritt) **aufgeklärt**:

**Der Zähler ist reihenfolge-abhängig, nicht inhalts- oder anzahl-abhängig.** Belege:
- **Inhalts-unabhängig:** eine LEERE `export {}`-Datei an derselben Stelle erzeugt
  den Sprung fast vollständig (−2,043); der echte Bench-Inhalt addiert nur +44.
  Zerlegung der −1,999: **−2,043 Reihenfolge-Effekt + 44 echte Typ-Kosten.**
- **Reihenfolge-abhängig (decisive):** dieselbe leere Datei, nur umbenannt, gibt
  −2,034 (`aaaa…`, früh in der sortierten Datei-Liste) vs. −304 (`zzzz…`, spät). Der
  EINZIGE Unterschied ist die Position → die Prüf-Reihenfolge.
- **Nicht-monoton:** zwei leere Dateien (−357) < eine früh einsortierte (−2,034).
- Corroboration: der `Types`-Zähler fiel bei der leeren Datei um −1,321 — auch die
  Anzahl *distinkter* erzeugter Typen ist reihenfolge-abhängig.

**Warum (TS-Interna):** Der Checker memoisiert Typ-Instanziierungen global pro
Kompilation. Die teuren Typen hier sind rekursiv (Digit-Arithmetik in
slice-literal.ts, `Broadcast`, `ReduceAxis`). Referenzieren mehrere Dateien
denselben geteilten Typ, zählt die ZUERST geprüfte ihn frisch (samt aller
Rekursions-Instanziierungen darunter); spätere treffen den Cache. Eine zusätzliche
Datei verschiebt die Datei-Reihenfolge → die Frisch-vs-gecacht-Partition der
geteilten Instanziierungen verschiebt sich → der Gesamtwert wandert, in beide
Richtungen, mit Betrag abhängig davon, WO die neue Datei im Sortlauf landet. Das
pinnt rückwirkend auch die Infra-01-Beobachtung (−69,730 beim Entfernen, super-
additiv = dieselbe Reihenfolge-Sensitivität). Nicht per `--generateTrace` bis auf
die konkreten Typen heruntergebrochen (Typ-IDs sind pro Lauf erzeugungsreihenfolge-
vergeben → nur über Struktur matchbar, fiddelig, ändert die Schlussfolgerung nicht).

**Konsequenz für die Pin-Disziplin:** Der Pin bleibt exakt/deterministisch für ein
FESTES Datei-Set (voller Tripwire für Nicht-Datei-Änderungen). Aber jede Scheibe,
die Dateien HINZUFÜGT/ENTFERNT, trägt einen Reihenfolge-Rauschterm bis ~±2,000, der
NICHT den echten Typ-Kosten entspricht — kleine Deltas über solche Scheiben nicht
als „Kosten X" lesen. Budget-Verfolgung (~3,4 % von 5M) unberührt; ein echter
Regress wäre ≫2,000 und monoton. Saubere Pro-Scheiben-Attribution beim Hinzufügen
einer Datei: erst LEER hinzufügen + messen (= Reihenfolge-Anteil), dann füllen +
messen (= echte Inhalts-Kosten). Neuer Haupt-Pin: **172,392 @ 128** (real,
deterministisch, coverage-neutral).

## Bench (`pnpm bench:elementwise`)
n=1024, add/sub/mul/div, contiguous (Fast-Path) vs. non-contiguous (transponiert,
general path) IM SELBEN Binary — fast/general **13,01×–16,25×** (adversarial
reproduziert; Executor 12,8–14,7×). Bit-Identitäts-Gate läuft VOR jedem Timing
(1-ULP-Mutation in die Referenz → sofortiger Throw). vs. naive TS: 19–38×.

## Scope / bewusst deferred (in FOLLOWUPS)
- **`unravel_into`-Generalisierung** (der breitere Hebel): `unravel` durch das
  nicht-allozierende `unravel_into` über ALLE strided Kernel (auch sum/strided
  matmul/broadcast) ersetzen. Payoff für den general-Case ist HERGELEITET, nicht
  gemessen (der Contiguous-Fast-Path gewinnt teils auch durch Vektorisierung, die
  `unravel_into` allein nicht brächte). Eigene, eigens gemessene Scheibe.
- **SIMD elementwise: gemessen NO-GO** (memory-bound, arithm. Intensität ≈0,042
  Flop/B; `div` so schnell wie `mul`). FOLLOWUPS-Item entsprechend geschlossen.
- **Packing-Buffer-Reuse im matmul, Facette A: gemessen 3,3 %** — für die
  Freeze-Zeremonie zu wenig, NO-GO für v0. Facette B (redundantes B-Repack)
  unvermessen.
