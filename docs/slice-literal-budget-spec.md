# Mini-Scheibe: `slice`-Literal-Budget — bindende Spec

**Version:** v2 (2026-07-25)
**Status:** Owner-Richtungsabnahme erteilt (D1 Scope 32, D2 Form C, D3 ja, Stufe 2–3 ohne
Baustein C) · Baustein 0 gelaufen, Befunde eingearbeitet → bereit für die Implementierung

**Änderungslog v1 → v2** (alle Punkte aus der adversarialen Spec-Verifikation, keiner davon
eine Richtungsänderung — der Geltungsbereich der 32 Stellen bleibt unverändert):
- **D1** bekommt die fehlende **Empfänger-Bedingung**; drei Stellen, die dem v1-Wortlaut
  genügten, aber nichts einbringen, sind jetzt ausdrücklich ausgeschlossen und belegt.
- **D2** ergänzt den in v1 vergessenen `SliceSpecInput`-Import im Helfer-Rezept.
- **D4** korrigiert die zu pauschale Charakterisierung von `slice.test.ts`.
- **D6/Arbeitsregel 16** erbt die Empfänger-Bedingung aus D1.
- **Die Nicht-Vakuitäts-Pflicht** benennt jetzt konkrete, geprüfte Stellen — die v1-Fassung
  („eine der vier View-Klassen") hätte mit 41 % Wahrscheinlichkeit einen Scheinbeweis
  geliefert.
- **Neu: D7** — ein vorbestehender Testqualitäts-Befund, den Baustein 0 nebenbei aufdeckte.

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

**`test:threaded` (139):** in v1 als ungemessen offengelassen (braucht die pinned nightly; zwei
der 32 Stellen liegen in `threaded.test.ts`). **Baustein 0 hat es nachgeholt: 139/139 pass,
Exit 0**, auf der vollen 32-Stellen-Migration inklusive echtem `cargo build`. Das ist eine
Fremdmessung — sie bleibt **Pflicht in der Umsetzung**, wird dort aber voraussichtlich nur
bestätigt.

## D1 — Geltungsbereich: 32 Stellen in 7 Dateien

Betroffen ist genau die Menge „Aufrufstelle von `NDArray.slice`/`WNDArray.slice` mit
literalen Spec-Argumenten **auf einem Empfänger mit statisch bekanntem Rang**, in einer Datei
ohne Typ-Ebenen-Assertionen".

**Die Empfänger-Bedingung ist tragend** (v2-Ergänzung, Baustein-0-Befund). `spike/src/slice.ts`
prüft in **beiden** Einstiegspunkten `RankUnknowable<S>` **vor** jeder Betrachtung von `Specs`
(`SliceShape` Zeile 110, `SliceSpecsGuard` Zeile 225). Steht der Empfänger auf einer dynamisch
getippten Shape (`number[]`), nimmt die Maschinerie den No-Claim-Pfad also **schon vorher** —
literale Specs kosten dort nichts, und ein `wideSpecs()` darum wäre reine Mühe. Drei Stellen in
`resident.test.ts` (925, 1145, 2216) genügen dem v1-Wortlaut, fallen aber genau darunter;
**gemessen: ihr Widening bringt Δ−35 über drei Stellen** (237.344 @ 140), gegen einen
Durchschnitt von ≈368 pro echter Stelle. Sie sind **nicht** Teil der 32.

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
`Array.prototype.slice` auf einfachen Arrays) · `argmax-topk.test.ts` (die zwei NDArray-Stellen
benutzen bereits den gewideten Spread `.slice(...specs)`) · `vector.test.ts` (3 Treffer, alle
schon `...specs` über eine `SliceSpec[]`-typisierte Variable) · `s1-import-guard.test.ts`
(2 Treffer, beides `String.prototype.slice`). Die letzten beiden fehlten in v1 und sind hier
nur der Vollständigkeit halber ergänzt — an der Menge ändert sich nichts.

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
import type { SliceSpecInput } from "../src/slice.ts"; // v2: in v1 vergessen

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

`spike/tests-runtime/slice.test.ts` bleibt **unangetastet**. v1 nannte alle 43 Stellen pauschal
„tragend"; das war zu grob. Die Datei ist eine **Mischung aus drei Klassen** (Baustein-0-Befund),
und jede fällt aus einem eigenen Grund heraus:

1. **Echte Semantik-Pins** — kodieren die Tabelle aus `docs/kern-05-slicing-spec.md` Zeile für
   Zeile (`base.slice(1, { start: 1 })` mit danebenstehendem erwarteten Ergebnis). Hier ist das
   Literal die Aussage.
2. **Bewusste Typ-/Laufzeit-Grenzfälle** — die vorhandenen `5 as number` /
   `{ step: 0 as number }`-Widenings existieren gerade deshalb, weil die Typ-Ebene das Literal
   sonst ablehnen würde. Ein pauschales Widening verwischt genau die Grenze, die sie markieren.
3. **Bereits dynamische Stellen** — teils `...specs`-Spreads (die randomisierten
   Differentialtests), teils dynamische Empfänger wie Zeile 718–724, die per Kommentar
   ausdrücklich „the runtime backstop" prüfen. Die kosten nach D1 ohnehin nichts.

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

## D7 — Vorbestehender Testqualitäts-Befund (NICHT Gegenstand dieser Scheibe)

Baustein 0 fand nebenbei etwas, das mit dem Umbau nichts zu tun hat, aber wichtiger sein
könnte als er: **an 13 der 32 Stellen ist das Test-Orakel selbstreferentiell.** Die
Helferfunktionen `assertMeanViewMatches`, `assertScalarOpMatches` und (an Stellen ohne
Precondition) `assertTopkMatches` bilden ihre Referenz aus dem `.toArray()` **des bereits
konstruierten Views**. Wird der Spec verfälscht, entsteht ein anderer View — aber Referenz und
Kandidat rechnen beide konsistent über diesen anderen View weiter, und die Assertion kann den
Unterschied strukturell nie sehen.

Betroffen: `resident.test.ts` 501, 521, 544, 658, 681, 706, 1369, 1391, 1416 (die S2/mean-,
S3/item- und S4/argmax-View-Blöcke), 2087; `elementwise.test.ts` 416, 433;
`backend-api.test.ts` 236 (dort vergleicht der Test tautologisch gegen sich selbst).

**Was das praktisch heißt:** Die Tests sind nicht falsch — `mean`/`argmax`/`item` werden dort
korrekt gegen die naive Referenz geprüft. Verloren ist die **Coverage-Aussage**: dass der
Empfänger die View-Klasse ist, die der Testname behauptet. Ausgerechnet Arbeitsregel 12
(„Residente Op-Tests müssen VIEWS treffen") stützt sich auf diese Blöcke. Der S5/topk-Block ist
die Ausnahme — dort hat der Autor Precondition-Assertions gesetzt
(`assert.notStrictEqual(view.describe().strides[0], 1, "precondition: …")`), und genau die
machen ihn mutationssensitiv.

**Befund ist selbst nachgestellt** (`resident.test.ts:501`, `{step:2}` → `{step:1}`:
6122 pass / 0 fail). **Er ist vorbestehend und wird durch die Migration weder verursacht noch
verschlimmert** — er besteht mit und ohne `wideSpecs` identisch. Deshalb: **out of scope für
diese Scheibe**, geht als eigener FOLLOWUPS-Eintrag raus (Vorschlag: die neun Blöcke bekommen
die Precondition-Assertions, die S5 schon hat). Innerhalb dieser Scheibe wirkt er nur als
Einschränkung des Mutations-Orts (siehe Nicht-Vakuitäts-Pflicht).

## D6 — Hausregel, die daraus wird

Aufnahme in CLAUDE.md als **Arbeitsregel 16**:

> **Laufzeittests bauen Views mit `wideSpecs(...)`, nicht mit literalen Specs.** Ein
> literales Slice-Spec in einer `*.test.ts` **auf einem Empfänger mit statisch bekanntem Rang**
> bezahlt die volle `SliceSpecsGuard`/`SliceShape`-Maschinerie, ohne dass eine Zusicherung davon
> abhängt — die Typ-Ebenen-Pins liegen in `spike/tests/*.test-d.ts`. **Die Empfänger-Bedingung
> gehört dazu:** steht der Empfänger auf einer dynamischen Shape (`number[]`), greift
> `RankUnknowable` schon vor der Spec-Prüfung, das Literal kostet nichts, und ein `wideSpecs()`
> wäre verschwendete Mühe (gemessen: Δ−35 über drei solche Stellen). Ausnahme:
> `slice.test.ts` selbst (dort sind die Literale die Aussage bzw. markieren die Typ-/
> Laufzeit-Grenze). **Und die Kosten sind super-additiv:** aus „Block X kostet N" folgt
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
noch beißen; Revert per **Backup-Kopie mit `diff`-Beweis**, nie `git checkout`
(Arbeitsregel 1).

**v1s Anweisung („in einer der vier View-Klassen den Spec verfälschen") war unbrauchbar** und
hätte mit hoher Wahrscheinlichkeit einen **Scheinbeweis** geliefert. Grund ist D7: an 13 der 32
Stellen ist das Test-Orakel selbstreferentiell, ein verfälschter Spec ist dort strukturell
unsichtbar. Selbst nachgestellt: `resident.test.ts:501` von `{ step: 2 }` auf `{ step: 1 }`
gedreht — was aus dem gestrideten View einen contiguous macht, also genau die Eigenschaft
zerstört, für die der Block existiert — ergibt **6122 pass / 0 fail**, kein einziger Test merkt
es.

**Bindend: der Mutant läuft an mindestens zwei Stellen mit nachgewiesen unabhängigem Orakel.**
Zulässige Kandidaten (Baustein 0 hat sie direkt mutations-getestet, je 3 benannte Fehlschläge):

- `resident.test.ts:2036` — verketteter topk-View, hat eine eigene Precondition-Assertion.
- `threaded.test.ts:1429` — Referenz kommt aus `topkRuntime` über die unabhängigen Rohdaten.
- alternativ `elementwise.test.ts:267` — Referenz ist ein separat gebautes `NDArray`.

Die Stellen aus D7 sind als Mutations-Ort **ausgeschlossen**.

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

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-25)

EIN `brainroute:deep`-Agent, frischer Kontext, gegen v1 dieser Spec und den echten Code;
Auftrag aus `docs/verify-runde-template.md` „Baustein 0". Der Verifier hat die Migration in
einem eigenen Worktree tatsächlich gebaut, alle Gates gefahren und **sieben Mutanten** laufen
lassen.

**Verdikt: kein Blocker.** Der Geltungsbereich der 32 Stellen ist im praktischen Ergebnis
korrekt — der vorregistrierte Zielwert 225.599 @ 140 wurde unabhängig exakt reproduziert,
ebenso stress/browser Δ0 und die volle Verhaltensparität inklusive `test:threaded` 139/0.

**Zwei Befunde mit Schweregrad MAJOR, beide in v2 eingearbeitet:**
1. D1s schriftliches Kriterium fehlte die **Empfänger-Bedingung** — die Liste war richtig, die
   Herleitungsregel nicht. Hätte über D6 als Dauerregel ins Projekt wandern können.
2. Die **Nicht-Vakuitäts-Pflicht war ortsabhängig** und hätte bei naiver Ortswahl einen
   Scheinbeweis geliefert (13 der 32 Stellen sind mutations-blind). Daraus wurde D7.

**Selbst nachgeprüft, weil beide Befunde folgenreich sind** (Fremdbefunde gelten hier nicht
ungeprüft): Der Vakuitäts-Befund ist bestätigt — `resident.test.ts:501` von `{step:2}` auf
`{step:1}`, also gestridet → contiguous, ergibt 6122 pass / 0 fail. Die
Empfänger-Bedingung ist bestätigt, **die Zahl des Verifiers dazu jedoch nicht**: er berichtete
Δ−1 für die drei dynamischen Stellen, gemessen sind **Δ−35** (237.344 @ 140). Der Unterschied
ändert die Schlussfolgerung nicht (−35 gegen ≈368 pro echter Stelle), ist aber ein weiterer
Beleg dafür, dass Einzelsite-Zahlen in diesem Zähler nicht belastbar sind (Befund 3 der
Ergebnis-Doku).

**Übernommen ohne eigene Nachprüfung** (klein, plausibel, verankert): die drei
Dokumentationslücken (fehlender `SliceSpecInput`-Import im Helfer-Rezept, `vector.test.ts` und
`s1-import-guard.test.ts` in der Ausschlussliste, die Dreiteilung von `slice.test.ts`) sowie
`test:threaded` 139/0. Alle vier stehen als Fremdmessung markiert.
