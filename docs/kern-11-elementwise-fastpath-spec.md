# Kern 11 — Elementwise Contiguous Fast-Path (Binding Spec)

*Phase B, Item 7 (Perf-Kür). Datum: 2026-07-12. GO auf Basis der Item-7-Messung
(read-only + `deep`-A/B, siehe Mess-Report unten). Owner-Entscheidung: fokussiert
= Contiguous-Fast-Path; die breite `unravel_into`-Generalisierung ist bewusst
deferred (FOLLOWUPS).*

## Messgrundlage (warum GO)

Die Item-7-Eröffnungsmessung (throwaway-Prototypen in Scratch-Kopie,
bit-identitäts-gegated) ergab:
- **SIMD elementwise: NO-GO** — memory-bound (arithm. Intensität ≈0,042 Flop/B;
  `div` misst so schnell wie `mul` → ALU wartet auf Speicher). SIMD adressiert
  ALU-Durchsatz, nicht Bytes/Element → ≈0 Gewinn.
- **Packing-Reuse (matmul, Facette A): NO-GO für v0** — real, bit-identisch,
  aber nur ~3,3 % bei n=512/1024; für die Freeze-Zeremonie zu wenig.
- **Der eigentliche Hebel (dieser Scheibe):** `binary_strided`/`add_strided`
  rufen pro Ausgabe-Element `unravel` auf, und `unravel` (shape.rs) **alloziert
  pro Aufruf einen frischen `Vec<u32>`** — >1 Mio. Kurzlebige Allokationen für
  ein 1024×1024-Array. Ein Contiguous-Fast-Path (unravel ganz überspringen)
  misst **~8–10× schneller** (mul 12,9→1,65 ms; div 12,7→1,31 ms), bit-identisch
  gegen die naive Referenz.

## Ziel

Für den mit Abstand häufigsten elementwise-Fall — **beide Operanden
kontiguiert, gleiche Shape, Offset 0 (kein Broadcast, keine Transposition,
kein Offset)** — die per-Element-`unravel`-Allokation durch einen flachen
`out[i] = op(a[i], b[i])`-Loop ersetzen. Der genau der Embedding/RAG-Hotpath
des Projekts. Bit-für-bit identisch zur bestehenden allgemeinen Schleife.

## Binding-Entscheidungen

### D1 — Scope: die ganze Resident-elementwise-Familie, NUR die strided Pfade

Fast-Path bekommen **`add_strided`** (crates/core/src/kernels/add.rs, ~Z. 62)
UND **`binary_strided`** (crates/core/src/kernels/elementwise.rs → sub/mul/div).
Dupliziert je Datei (Codebase-Präzedenz Kern 04: „duplication over shared code
paths"; add.rs spiegelt elementwise.rs ohnehin line-for-line). 3 von 4 der
Familie zu fixen wäre inkohärent (add ist die häufigste Op).

**NICHT angefasst:** das v1 `add`/`nt_add` (add.rs ~Z. 14) — dokumentiert
eingefrorene Performance-Baseline, kein Optimierungsziel (CLAUDE.md: v1-Backend
bleibt byte-for-byte). Es bleibt langsam; das ist gewollt. Beide Pfade matchen
weiterhin die naive Referenz → alle Differentialtests bleiben grün.

**Deferred (FOLLOWUPS, NICHT in dieser Scheibe):** die breite Wurzel-Lösung
`unravel` → `unravel_into` (nicht-allozierend, Kern-06-Muster) über den
ALLGEMEINEN strided/broadcast-Pfad und andere Kernel (sum etc.). Deren
General-Case-Payoff ist hergeleitet, nicht gemessen; eigene Scheibe.

### D2 — Fast-Path-Bedingung (einmal pro Call geprüft, O(rank), nie per Element)

Fast-Path nur wenn ALLE gelten:
- `a_shape == b_shape` (⇒ `out_shape == a_shape`, kein Broadcast),
- `a_offset == 0 && b_offset == 0`,
- `a_strides == compute_strides(a_shape)` und `b_strides == compute_strides(b_shape)`
  (natürliche C-contiguous Strides — kein transponierter/gestrideter View).

Wenn erfüllt: `for i in 0..size { out[i] = op(a_data[i], b_data[i]); }`. Sonst
der bestehende allgemeine Loop, **unverändert**. Die Prüfung darf eine
(einmalige) `compute_strides`-Allokation pro Call kosten — irrelevant (das
Problem war per-ELEMENT-Allokation).

**Reihenfolge im Code:** Die Fast-Path-Bedingung wird NACH den bestehenden
Validierungen (`checked_element_count`, `validate_strided_bounds`,
`broadcast_shape`) geprüft — so bleibt die Fehler-Semantik (Status 1/2/3/4,
Reihenfolge, Meldungen) byte-für-byte identisch für alle Eingaben.

**Bounds-Garantie (verifiziert an shape.rs):** `validate_strided_bounds`
erzwingt `max_reach < data_len` mit `max_reach = offset + Σ(dim_i−1)·stride_i`.
Für contiguous offset-0 same-shape ist `max_reach = size−1` → `data_len ≥ size`
für BEIDE Operanden. Also ist `a_data[i]`/`b_data[i]` für `i∈0..size` garantiert
in-bounds und bit-identisch zum allgemeinen Pfad (dessen
`.get(off).copied().unwrap_or(0.0)` den 0.0-Fallback hier nie erreicht). Für
Vektorisierung + panic-Freiheit ohne per-Element-Bounds-Check bevorzugt EINMAL
slicen und dann iterieren, z. B. `let a=&a_data[..size as usize]; let
b=&b_data[..size as usize]; for i in 0..size as usize { out[i]=op(a[i],b[i]); }`
(oder `out.iter_mut().zip(a).zip(b).for_each(...)`) — ein Bounds-Check statt
size-viele, LLVM vektorisiert das.

**FREEZE-FALLE — nicht die `_into`-Twins nutzen:** Die Bedingung nutzt
`compute_strides` (allozierend, im Plain-wasm32-Artefakt immer vorhanden; eine
Allokation pro Call ist irrelevant — das Problem war per-ELEMENT). NIEMALS
`compute_strides_into`/`unravel_into` verwenden: die sind
`#[cfg(any(not(target_arch="wasm32"), target_feature="atomics"))]`-gegated und
im Plain-wasm32-Artefakt GAR NICHT kompiliert — schon ihre bloße Präsenz verschob
früher die Bytes (shape.rs:208-220, empirisch belegt). Der Fast-Path-Loop
braucht ohnehin KEIN unravel (er rechnet `out[i]=op(a[i],b[i])` direkt).

### D3 — Bit-Identität ist die kritische Eigenschaft (NICHT verhandelbar)

Für contiguous same-shape offset-0 bildet die allgemeine Schleife jedes
`flat` auf `a_off = flat`, `b_off = flat` ab (idx·natürliche_strides = flat).
Also ist `out[flat] = op(a_data[flat], b_data[flat])` — **exakt** was der
Fast-Path rechnet, in derselben flat-Reihenfolge. Bit-für-bit identisch,
inklusive Spezialwerte (NaN/±Inf/±0/Subnormals — reine Datenbewegung + eine
IEEE-Op, keine Reassoziation, keine Reihenfolgeänderung). **Falls ein Test eine
Divergenz zeigt: STOPP, Root Cause, Owner — NIEMALS den Test abschwächen oder
den Fast-Path „ungefähr" lassen.**

### D4 — Freeze-Zeremonie (Artefakt-Hash bricht → vierteiliger Beweis + Re-Pin)

Diese Scheibe ändert Kernel-Bytes → der Artefakt-Hash `7a65d800…` bricht
legitim (wie Kern 07). Beweis-Dekomposition:
1. **Pre-Edit clean rebuild reproduziert den ALTEN Pin** `7a65d80062865a5e88952ce3cfbdd974b642f6d3f4b293e3f3b39afad16885d8`
   (`cargo clean` + `pnpm build:wasm`, Hash prüfen) — Baseline etabliert.
2. **Verhaltenserhaltung bewiesen:** alle Differentialtests grün (add/sub/mul/div
   inkl. contiguous same-shape, die jetzt den Fast-Path treffen), special-values-
   Suite grün, plus ein NEUER expliziter „beide-Pfade-stimmen-überein"-Test
   (dieselben logischen Daten einmal contiguous = Fast-Path, einmal als
   non-contiguous View = allgemeiner Pfad → beide bit-identisch zur Referenz).
3. **Mutations-Beweis (Nicht-Vakuität):** den Fast-Path mutieren (z. B. falscher
   Index `a_data[i+1]`, oder Bedingung fälschlich immer-true bei Broadcast) →
   Differential-/special-values-Suite MUSS fallen; Revert MUSS grünen. In
   Scratch/Isolation, nie das echte Artefakt überschreiben.
4. **Neuer Hash wird der neue Pin** — CLAUDE.md + alle Pin-Referenzen (Commands-
   Zeile, Frozen-Discipline-Absatz, docs) nachziehen.

`track_caller`-Hinweis: `binary_strided` steht VOR sub/mul/div_strided → ein
In-Place-Branch verschiebt deren Zeilen (bytes ändern sich, Verhalten nicht) —
das ist bei einer INTENTIONAL-CHANGE-Phase erwartet und vom vierteiligen Beweis
gedeckt (nicht der reine Additions-Fall). In add.rs steht `add_strided` NACH dem
v1 `add` → das v1 `add` bleibt unverschoben (nur Tests folgen, nicht im
Release).

### D5 — Committer Micro-Benchmark (der Gewinn kommt on-record)

Neu: `spike/bench-core/elementwise.ts` + `bench:elementwise`-Skript
(package.json), Standard-Harness (Bit-Identitäts-Gate VOR Timing, gewärmter JIT,
adaptive Reps, Bereiche statt Einzelpunkte). Misst add/sub/mul/div auf großen
kontigierten gleich-geformten Arrays (z. B. [1024,1024]) — Referenz⇄resident,
plus (Ehrlichkeit) einen gestrideten/nicht-contiguous Fall, der den Fast-Path
NICHT trifft (zeigt, dass der allgemeine Pfad unverändert langsam bleibt = die
deferred breite Lösung ist sichtbar begründet). Kein neues Test-Gate; die Zahlen
gehen ins Ergebnisdoc. (Falls `bench:*` einen Guard/Explizitliste hat: eintragen.)

### D6 — Scope-Disziplin

Kein SIMD (gemessen wertlos). Kein Packing-Reuse (separater NO-GO). Keine
`unravel_into`-Generalisierung (deferred). Keine Änderung am v1-Backend. Keine
neue Op. Nur der Fast-Path in add_strided + binary_strided + der Bench.

## Testplan

- **Bestehende Differentialsuiten** (test:core add.test.ts, test:resident
  elementwise.test.ts, special-values.test.ts) müssen grün bleiben — ihre
  contiguous-same-shape-Fälle treffen jetzt den Fast-Path und prüfen ihn implizit
  gegen die naive Referenz.
- **Neuer expliziter Pfad-Äquivalenztest** (cargo, in add.rs + elementwise.rs
  test-mod): fast-path-Eingabe vs. logisch identische general-path-Eingabe
  (z. B. via transponiertem Roundtrip oder offset-Window) → `assert_eq!` auf
  Bits. Beweist, dass beide Pfade dasselbe liefern.
- **cargo** Gesamtzahl steigt um die neuen Unit-Tests; berichten.

## Gate-Erwartung

- `pnpm check` (Verbund) clean.
- `pnpm test:core` (817 + neue add-Pfad-Tests), `test:resident` (4265+2 + neue),
  `cargo` (157 + neue) — alle grün.
- **Artefakt-Hash ändert sich** (neuer Pin, vierteiliger Beweis) — Pre-Edit-
  Reproduktion des alten `7a65d800…` zwingend zeigen.
- `check:diag` unbewegt (kein Typ-Ebenen-Change; Bench ist Wertcode) — falls doch
  minimal, neu pinnen. Mess-Hausregel beachten.
- `pnpm demo` all-agree (elementwise läuft dort ggf. nicht — egal).
- `bench:elementwise` zeigt den ~8–10×-Gewinn auf dem contiguous-Pfad on-record.

## Definition of Done

Spec (dies) → Implementierung (`deep`, freeze-sensibel) → **Zwei-Verifier-Runde**
(Spec + adversarial, docs/verify-runde-template.md; adversarial bekommt
Bit-Identität des Fast-Path + Freeze-Zeremonie + „greift der Fast-Path nur wenn
er darf" als Angriffsfläche) → docs/kern-11-elementwise-fastpath-ergebnisse.md
(grounded, Honesty-Rule, beide Verdikte, neuer Pin mit Delta, Bench-Zahlen) →
KB-Upsert (bit-identische-Differentialtests-Notiz: Fast-Path-Bit-Identität als
Facette; ggf. neue Notiz „per-Element-Allokation in Hot-Loops") → FOLLOWUPS
(breites unravel_into als eigene gemessene Scheibe; Packing-Reuse-A 3,3 % als
gemessen-NO-GO dokumentiert) → Commit + Push. Alle Befehle vom Repo-Root.
