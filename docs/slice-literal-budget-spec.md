# Mini-Scheibe: `slice`-Literal-Budget — bindende Spec

**Version:** v1 (2026-07-25)
**Status:** entworfen, wartet auf Owner-Richtungsabnahme → dann Baustein 0
**Vorgeschichte:** FOLLOWUPS „`slice`-Literal-Kosten als Budget-Hebel" ·
Gegenprüfung `docs/slice-literal-budget-ergebnisse.md` (Messteil, bereits gelaufen) ·
korrigierter Ursprungsbefund `docs/wasm-parity-topk-ergebnisse.md`, Nachtrag 2

## Ziel in einem Satz

Literal-argumentige `.slice()`-Aufrufstellen in **Laufzeit**-Testdateien auf eine gewidete
Schreibweise umstellen, weil sie dort die volle `SliceSpecsGuard`/`SliceShape`-Maschinerie
bezahlen, ohne dass irgendeine Zusicherung davon abhängt — gemessen **−11.780
Instantiations = 4,96 % des Root-Korpus**.

## Warum das eine eigene Scheibe ist

Der Umbau editiert **Bestandstestdateien**. Das ist die bewusste Disziplin-Entscheidung,
die der Owner beim Übergeben ausdrücklich verlangt hat (siehe D3) — sie gehört nicht
nebenbei in eine Op-Scheibe.

## Was VORAB schon gemessen ist (kein Neuland mehr)

Alle Zahlen im frischen `git worktree` auf `2d38f67`, Exit-Code und Fehlerzahl je geprüft,
Dateiset durchgehend 140 (also kein Order-Noise). Baseline **237.379 @ 140**, zweimal
reproduziert.

| Größe | vorher | nachher | Δ |
|---|---|---|---|
| `check:diag` (Root) | 237.379 @ 140 | **225.599 @ 140** | **−11.780** |
| `check:diag:stress` | 116.053 @ 82 | 116.053 @ 82 | 0 |
| `check:diag:browser` | 2.142 @ 75 | 2.142 @ 75 | 0 |
| `bench:editor` (8 Pins) | `{37573, 39406, 70548, 37728, 43027, 44221, 36777, 44466}` | identisch | 0, Gate PASS |
| `test:core` | 1591 / 0 fail | 1591 / 0 fail | 0 |
| `test:resident` | 6122 + 2 skipped | 6122 + 2 skipped | 0 |
| Freeze-Hash | `2a54d9fd…` | unberührt (kein Rust-File) | 0 |

**Noch nicht gemessen:** `test:threaded` (139) — braucht die pinned nightly. Zwei der 32
Stellen liegen in `threaded.test.ts`. **Pflicht in der Umsetzung.**

## D1 — Geltungsbereich: 32 Stellen in 7 Dateien

Betroffen ist genau die Menge „Aufrufstelle von `NDArray.slice`/`WNDArray.slice` mit
literalen Spec-Argumenten in einer Datei ohne Typ-Ebenen-Assertionen".

| Datei | Zeilen (Stand `2d38f67`) | n |
|---|---|---|
| `spike/tests-runtime/resident.test.ts` | 501, 521, 544, 658, 681, 706, 1369, 1391, 1416, 1966, 1985, 2008, 2034, 2036, 2087 | 15 |
| `spike/tests-runtime/elementwise.test.ts` | 267, 272, 293, 298, 416, 433 | 6 |
| `spike/tests-runtime/keepdims.test.ts` | 197, 202, 224, 231 | 4 |
| `spike/tests-runtime/scalar-mean.test.ts` | 641, 1141 | 2 |
| `spike/tests-runtime/reshape.test.ts` | 183, 221 | 2 |
| `spike/tests-runtime/threaded.test.ts` | 1179, 1429 | 2 |
| `spike/tests-runtime/backend-api.test.ts` | 236 | 1 |

Alle sieben Dateien haben **null** Treffer auf `Equal<` / `Expect<` / `expectType` /
`assertType` — mechanisch geprüft, das ist die tragende Voraussetzung.

**Nicht betroffen, geprüft:** `resident-lifecycle.test.ts` (16 Treffer, alle
`Array.prototype.slice` auf einfachen Arrays) und `argmax-topk.test.ts` (die zwei
NDArray-Stellen benutzen bereits den gewideten Spread `.slice(...specs)`).

## D2 — Die Schreibweise: geteilter `wideSpecs(...)`-Helfer

Drei Kandidaten wurden **gemessen**, nicht nach Geschmack gewählt:

| Form | inst | Δ |
|---|---|---|
| A · lokale `const specs: readonly SliceSpecInput[]` (2 Zeilen/Stelle) | 225.628 | −11.751 |
| B · Inline-Cast `.slice(...([…] as readonly SliceSpecInput[]))` | 225.684 | −11.695 |
| **C · geteilter Helfer `.slice(...wideSpecs(…))`** | **225.599** | **−11.780** |

**Gewählt: C.** Sie ist die billigste, die kürzeste (eine Zeile, keine erfundenen lokalen
Namen) und die einzige, die das *Warum* an jeder Aufrufstelle sichtbar macht — was für eine
Hausregel zählt, der künftige Beiträge folgen sollen. Der Helfer wird an `spike/tests-runtime/
assert-helpers.ts` **angehängt** (Insertion, kein Edit an Bestandsinhalt):

```ts
/** Widen a literal slice spec list so `SliceSpecsGuard`/`SliceShape` take the
 * no-claim path. Runtime tests check RUNTIME behaviour; the type-level claims
 * live in `spike/tests/slice.test-d.ts`. Paying the literal machinery here
 * buys nothing and is measurably expensive. */
export function wideSpecs(...specs: readonly SliceSpecInput[]): readonly SliceSpecInput[] {
  return specs;
}
```

Aufrufstelle vorher/nachher:

```ts
const view = w.slice({ step: 2 }, null);              // vorher
const view = w.slice(...wideSpecs({ step: 2 }, null)); // nachher
```

**Umsetzungshinweis:** Dateien, die bereits aus `./assert-helpers.ts` importieren, bekommen
`wideSpecs` in den **bestehenden** Import gemergt, keinen zweiten Import-Statement.

## D3 — Disziplin-Entscheidung: Edit an Bestandstests (offenzulegen, Owner-Abnahme)

**Die Regel, um die es geht, greift hier nicht.** Die Append-/Insertion-only-Disziplin des
Projekts ist ein **Artefakt-Byte-Argument** und gilt für `crates/core` (abi.rs,
matmul_blocked.rs, shape.rs) sowie sinngemäß für TS-**Klassenkörper** mit privatem
Konstruktor. `spike/tests-runtime/*.test.ts` fällt unter keine von beiden; eine geschriebene
Regel „Bestandstests nie editieren" existiert nicht.

**Präzedenzfall im Projekt:** Spike 06 hielt fest, dass eine Scheibe, die eine Literal-Grenze
verschiebt, damit rechnen muss, bestehende Pins **absichtserhaltend neu zu formulieren** —
„plan for it, don't demand 'all old tests untouched'".

**Die Entscheidung:** 32 Zeilen in 7 Bestands-Testdateien werden editiert. Jede Änderung ist
eine reine Typ-Ebenen-Widening; der erzeugte JS-Code ist bis auf den Helferaufruf identisch,
die Semantik unverändert. Beleg ist, dass **jede Testzahl exakt gleich bleibt** (siehe
Gate-Tabelle) — nicht bloß „grün", sondern zahlengleich.

## D4 — Was bewusst NICHT umgestellt wird

`spike/tests-runtime/slice.test.ts` (43 literale Stellen) bleibt **unangetastet**. Dort sind
die literalen Specs **tragend**: sie kodieren die Semantik-Tabelle aus
`docs/kern-05-slicing-spec.md` Zeile für Zeile (`base.slice(1, { start: 1 })` mit
danebenstehendem erwarteten Ergebnis) und sie markieren die bewusste Typ-/Laufzeit-Grenze —
die dortigen `5 as number` / `{ step: 0 as number }`-Widenings existieren gerade deshalb, weil
die Typ-Ebene das Literal sonst ablehnen würde. Ein pauschales Widening würde diese Grenze
verwischen.

**Was dabei liegen bleibt, offengelegt:** die Datei kostet insgesamt **10.683** (per
empty-then-fill obenbegrenzt). Wie viel davon literal-getrieben ist, ist **nicht gemessen**.
Bleibt als FOLLOWUPS-Eintrag stehen.

## D5 — Der ehrliche Preis des Umbaus

Nach dem Widening ist der Rückgabetyp der betroffenen Views dynamisch. Konkret heißt das:
ein Tippfehler an einer dieser 32 Stellen (etwa drei Specs für einen Rang-2-Empfänger) wird
**nicht mehr beim Kompilieren** abgefangen, sondern erst beim Testlauf als Throw. Für eine
Laufzeit-Testdatei ist das vertretbar — sie wird ohnehin ausgeführt, und die betroffenen
Blöcke prüfen ihre Voraussetzungen bereits zur Laufzeit nach (`assert.deepStrictEqual([...
(view.shape as readonly number[])], [6], "precondition: …")`, also mit explizit
weggeworfenem Literaltyp). Aber es ist eine Verlagerung, keine Nullkosten, und sie gehört
benannt.

## D6 — Hausregel, die daraus wird

Aufnahme in CLAUDE.md als **Arbeitsregel 16**:

> **Laufzeittests bauen Views mit `wideSpecs(...)`, nicht mit literalen Specs.** Ein
> literales Slice-Spec in einer `*.test.ts` bezahlt die volle `SliceSpecsGuard`/
> `SliceShape`-Maschinerie, ohne dass eine Zusicherung davon abhängt — die Typ-Ebenen-Pins
> liegen in `spike/tests/*.test-d.ts`. Ausnahme: `slice.test.ts` selbst (dort sind die
> Literale die Aussage). **Und die Kosten sind super-additiv:** aus „Block X kostet N" folgt
> NICHT „Aufrufstelle kostet N/k"; eine einzelne Stelle isoliert zu messen kann das Vorzeichen
> umkehren. Nur Alles-oder-nichts-Messungen sind aussagekräftig.

## Vorregistrierte Gates (Absolutwerte, nicht Schätzungen)

Die Scheibe **senkt** den Zähler, ein Obergrenzen-Gate ist also gegenstandslos. Stattdessen
wird auf **Exaktheit** vorregistriert — jede Abweichung ist ein Befund:

- `check:diag` = **225.599 @ 140** exakt. Abweichung ⇒ Stopp, Ursache klären.
- `check:diag:stress` = 116.053 @ 82 · `check:diag:browser` = 2.142 @ 75 · `bench:editor` =
  die acht Bestands-Pins, alle **Δ0**.
- `check:freeze` = `2a54d9fd…` unverändert (kein Rust berührt), `cargo test` 222+1 unverändert.
- `test:core` 1591 · `test:resident` 6122+2 · `test:threaded` **139** · `test:browser` 4 ·
  `test:package` 3 — alle **zahlengleich**.
- `graph-a-lama query lint` Exit 0.

## Nicht-Vakuitäts-Pflicht (Mutant)

Der Umbau ist per Konstruktion verhaltensneutral — genau deshalb kann er **stillschweigend
einen Test entschärfen**. Pflicht: ein Mutant, der beweist, dass die umgestellten Stellen
noch beißen. Konkret in einer der vier View-Klassen den Spec verfälschen (`{ step: 2 }` →
`{ step: 1 }`) und belegen, dass benannte Tests fehlschlagen; Revert per **Backup-Kopie mit
`diff`-Beweis**, nie `git checkout` (Arbeitsregel 1).

## Berührte Covenant-Invarianten

- **S1** (Runtime-Quellcode importiert nie aus Testverzeichnissen) — nicht berührt, es wird
  ausschließlich in `spike/tests-runtime/` editiert. Mechanisch durch das Lint gedeckt.
- **M1** — nicht berührt (kein Kernel, kein Rust, kein Verhalten).
- **M3** (saubere Hover/Diagnosen) — nicht berührt: es ändert sich kein Typ auf der
  Konsumenten-API, nur der lokale Typ von Testvariablen. Arbeitsregel 13 („eine Hover-Norm
  wird gemessen, nicht gelesen") greift deshalb **nicht** — falls Baustein 0 das anders sieht,
  ist eine LSP-Messung nachzuziehen.
- **M4** (Freeze) — nicht berührt, kein Rust-File im Diff. Negative Assertion: der Hash
  **darf sich nicht bewegen**.

Eskalationsstufe: **Stufe 2–3**. Vorschlag: Baustein 0 gegen diese Spec + **eine**
Verify-Runde (A gegen die Spec, B adversarial) + Lint im Gate-Block; Baustein C
(covenant-verify) nur, falls Baustein 0 eine Invariante als inhaltlich tangiert einstuft.
Owner entscheidet.

## Doku-Pflichten am Ende

`docs/slice-literal-budget-ergebnisse.md` (Post-Verification-Addendum) · `docs/roadmap.md`
(falls Item-Status betroffen) · CLAUDE.md Sektionen „Aktuelle Pins & Gates" (neuer
check:diag-Pin) + Arbeitsregel 16 · FOLLOWUPS (Eintrag schließen, den `slice.test.ts`-Rest
als neuen Eintrag öffnen) · `docs/projekt-log.md` (volles Narrativ) · KB-Capture.
**README nicht betroffen** — es ändert sich nichts daran, wo eine Op läuft (Prüfkommando aus
der Hausregel trotzdem laufen lassen).

## Ein-Block-Revert

Der gesamte Umbau ist ein einziger, rein mechanischer Diff: 32 Aufrufstellen + 7
Import-Zeilen + eine Insertion in `assert-helpers.ts`. Rückbau = `git revert` des einen
Commits; der einzige Pin, der zurückwandert, ist `check:diag` auf 237.379 @ 140.
