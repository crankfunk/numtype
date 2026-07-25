# WASM-Parität S5: `topk` — Ergebnisse

Scheibe: WASM-Parität S5, **die letzte der Kampagne S0–S5** · Datum: 2026-07-25 ·
Bindende Spec: [wasm-parity-topk-spec.md](wasm-parity-topk-spec.md) **v2** ·
Eskalationsleiter: Stufe 3 (Baustein 0 + Verify-Katalog A+B+C).

**Ehrlichkeitsregel:** jede Zahl stammt aus einem tatsächlich ausgeführten Kommando; nicht
Ausgeführtes wird benannt, nicht weggelassen. Verdikte, die sich als falsch erwiesen,
bleiben mit ihrer Korrektur stehen.

## Ergebnis in einem Satz

`WNDArray.topk` existiert, bit-identisch zu `topkRuntime` bewiesen — **inklusive
byte-exakter NaN-Payloads, dem ersten Fall der Kampagne, in dem echte Datenwerte durch
einen neuen Kernel wandern**; damit trägt `WNDArray` alle fünf Ops und die README hat keine
„TypeScript-runtime only"-Ausnahme mehr.

## Der tragende Befund: die geerbte Annahme über S5 war falsch

`FOLLOWUPS.md` und `CLAUDE.md` trugen seit der topk-Selektions-Scheibe zwei Sätze: ein
künftiger `nt_topk`-Kernel **solle den Heap-Algorithmus spiegeln**, und S5 sei die
**härteste M1 der Kampagne**. Beides ist widerlegt.

Der Doc-Kommentar von `topkRuntime` (runtime.ts:676-687) enthielt das Gegenargument bereits:
`topkCompareValues` plus der Indextiebreak `|| (idxA - idxB)` ist eine **strikte
Totalordnung** auf den paarweise verschiedenen Indizes `0..n-1`. Es gibt daher genau eine
korrekte top-`k`-Menge in genau einer Reihenfolge — Heap und Vollsortierung MÜSSEN
übereinstimmen. Daraus folgt:

> **Bit-Identität hängt hier nicht vom Algorithmus ab**, sondern nur (a) von der exakten
> Transliteration des Ordnungs-Prädikats und (b) davon, dass `values[i] = data[indices[i]]`
> ein reiner Element-Kopiervorgang bleibt, der nie durch Arithmetik läuft.

Das ist **kategorial anders als bei `sum`**, wo die Akkumulationsreihenfolge die Bits
tatsächlich ändert und der Kernel die Schleifenordnung spiegeln MUSS. Bei `topk` findet
überhaupt keine Gleitkomma-Arithmetik statt.

Die praktische Konsequenz stand als bindende Festlegung in der Spec: **die Tests dürfen
sich nicht auf Heap-Interna festlegen** (Sift-Reihenfolge, Wurzelwahl, Zwischenzustände),
sonst pinnen sie eine Implementierung statt eines Vertrags und verhindern künftige
legitime Optimierungen ohne Korrektheitsgewinn.

**Zwei unabhängige Verifier haben das Argument geprüft, mit verschiedenen Methoden:**
Baustein 0 analytisch (Trichotomie für jedes Paar durchgeprüft — der Komparator liefert nur
bei zwei NaN oder numerischer Gleichheit inkl. `±0` eine 0, und dort ist der Indextiebreak
bei verschiedenen Indizes nie 0; Transitivität aus der Standardkonstruktion „totale
Präordnung + strikter Tiebreak"; IEEE-754-Übereinstimmung von Rusts/WASMs `f64`-Vergleichen
mit den JS-Operatoren). Baustein B empirisch, und mit einem **nützlichen Korollar**: an
jeder Heap-Vergleichsstelle sind die verglichenen Indizes strukturell verschieden, also
kann der Komparator dort nie exakt 0 liefern — womit `<`-vs-`<=`-Mutationen an genau diesen
Stellen **beweisbar äquivalente Mutanten** sind und keine Testlücke anzeigen. Das erspart
künftigen Verifiern die Jagd auf einen nicht existierenden Bug.

Der Kernel implementiert trotzdem denselben größenbeschränkten Heap (O(n log k)) — weil er
transliterierbar vorlag und kein Zusatzrisiko entsteht, nicht weil M1 es verlangt.

## Der echte M1-Risikopunkt: NaN-Payloads

`topk` gibt echte **Datenwerte** zurück (nicht nur Indizes wie `argmax`). Damit greift
erstmals in der Kampagne der NaN-Payload-Vorbehalt, den Baustein C in S4 präzise hierhin
verschoben hatte: `values[i] === data[indices[i]]` gilt byte-exakt inklusive
nicht-kanonischer Bitmuster.

Der Beweis wurde in drei Stufen geführt:

1. **Baustein 0, vorab:** der bestehende Repo-Präzedenzfall deckt nur den 2D-Transpose ab.
   Er hat zusätzlich eine **Rang-1, gestridete, Offset > 0**-Sicht mit
   nicht-kanonischem NaN durch `nt_materialize` geschickt — strukturell genau das
   Lesemuster des künftigen Kernels — und die Payload kam byte-exakt zurück. Ehrlich
   abgegrenzt als Analogie-Evidenz, kein Test des noch nicht existierenden Kernels.
2. **Implementierung:** ein eigener, benannter Test statt eines Nebeneffekts — zwei
   verschiedene nicht-kanonische Muster roh per `DataView` gesetzt, Ankunft direkt aus
   `core.memory.buffer` verifiziert, byte-exakte Rückgabe inklusive korrekter Ordnung; ein
   zweiter Test über den **gestrideten** Lesepfad; ein dritter über den threads-Artefakt auf
   allen vier Pool-Größen; plus ein cargo-Test im Kernel selbst.
3. **Damit ist Analogie durch direkte Evidenz an `nt_topk_strided` ersetzt.**

## Was gebaut wurde

**Rust** — neue Datei `crates/core/src/kernels/topk.rs` (377 Zeilen): `compare_values`
(wörtliche Transliteration von `topkCompareValues`), `cmp` mit Indextiebreak in `i64`,
`read_logical` als **einziger** Datenzugriff, `topk_strided` = größenbeschränkter Max-Heap
plus Abschluss-Sortierung, `values[i]` als reiner Element-Copy aus `data`. Ein neuer
ABI-Einstiegspunkt `nt_topk_strided`, **ein** Hunk am Dateiende von `abi.rs`.

**TypeScript** — `WNDArray.topk` als Instanz-Methode mit einer einzigen Signatur, exakter
Spiegel von `NDArray.topk`; `TopkCheck`/`TopkShape` unverändert wiederverwendet (zweite
Call-Site, `vector.ts` byte-unberührt). Keine Facaden-Änderung (Instanz-Methode, anders als
S3s statisches `stack`).

**Lifecycle** — der strukturell neue Teil: erstmals gibt eine Op **zwei** residente Handles
zurück. Beide Ausgabepuffer werden vor dem Kernel allokiert; scheitert die zweite
Allokation, wird die erste im `catch` freigegeben; bei Kernel-Status ≠ 0 werden beide
freigegeben; Scratch immer im `finally`. **Beide Fehlerpfade waren ab Tag 1 Testpflicht**
(die S4-Lektion, wo genau dieser Pfad von 0 von 334 Tests abgedeckt war) — und beide sind
einzeln mutations-bewiesen, je genau ein benannter Test.

## Freeze-Beweis (M4), dreiteilig — dreifach unabhängig nachgestellt

| Schritt | Ergebnis |
|---|---|
| ① Pre-Edit-Clean-Rebuild auf `6d730b2` | reproduziert den alten Pin `eba6ba7ac85d15a814fd027392a81c7450d885d2048d7efd6694b7e8370988bb` exakt |
| ② Additive-only am Diff | `abi.rs` **ein** Hunk `@@ -1733,0 +1734 @@` **nach** dem Testmodul `mod s4_argmax_abi_tests`, 0 Deletions; `mod.rs` 3/0; `topk.rs` neu; `shape.rs`/`matmul_blocked.rs`/`sum.rs`/`sqrt.rs`/`argmax.rs`/`scalar.rs`/`vector.rs` byte-unverändert |
| ③ Neuer Pin | `146afdf629694318a5dcca87c5bb980ae6280e875ae5d9b5ddef045a00c0c324`, zweimal byte-identisch |

Baustein A und Baustein B haben beide Richtungen je unabhängig nachgestellt (A in getrennten
detached Worktrees mit isolierten target-dirs; B per Backup-Kopie und manuellem Revert auf
den `git show HEAD:…`-Inhalt, da `git checkout` auf uncommittete Arbeit verboten ist). Der
Orchestrator hat den Endstand bestätigt.

**Der Append-Punkt war eine Spec-Korrektur wert.** v1 nannte `nt_argmax_axis_strided` als
„das letzte Item" von `abi.rs` — falsch: das letzte Item ist ein Testmodul. Die Datei hat
ein durchgängiges, nie gebrochenes Muster (jede Phase hängt Realcode nach dem *jeweils
aktuellen* Dateiende an, dann ihr eigenes Testmodul); ein wörtlich gelesenes v1 hätte
erstmals Realcode **vor** ein bestehendes Testmodul gespleißt. Für den Freeze-Hash
folgenlos (`#[cfg(test)]` fällt im Release-wasm-Build weg), aber musterbrechend.

## Pin-Protokoll (D8) — und der eigentliche Kostenbefund

Baseline im frischen Worktree reproduziert: **229.828 @ 140, Exit 0**.

| Stufe | Inhalt | Wert | Δ |
|---|---|---|---|
| 0 | Baseline | 229.828 | — |
| ① | `CoreExports`-Member + beide Mock-Stubs | 229.828 | **0** |
| ② | `WNDArray.topk`-Methode | 229.958 | +130 |
| ③ | Test-Anhänge | 236.916 | **+6.958** |
| ④ | Typ-Pins | **237.379** | +463 |

**Gesamt +7.551 gegen das vorregistrierte ≤ +8.000 — gehalten, aber mit nur 449 Marge**,
dem engsten Stand der Kampagne. Dateizahl unverändert 140, null Order-Noise. Stufe ①
bestätigt den S0/D10-Kampagnengewinn ein fünftes Mal (+0 statt +7 pro `CoreExports`-Member).

**Der Kostentreiber ist nicht `topk`.** Die Op selbst kostet +130. Der
**Vier-View-Klassen-Block** (110 Zeilen, den Arbeitsregel 12 verlangt) kostet allein
**+3.926** — 52 % der ganzen Scheibe. Getrieben nicht von `topk`, sondern von den
**literal-argumentigen `.slice()`-Aufrufstellen** darin, die je die volle
`SliceSpecsGuard`/`SliceShape`-Maschinerie zahlen. Die Zahl ist **dreifach unabhängig
bestätigt**: Implementierer, Baustein A und Baustein B haben je den Block ausgebaut und
gemessen, alle drei kamen auf exakt 3.926 (233.453 @ 140 ohne den Block).

Das ist der verwertbarste Nebenbefund der Scheibe: für *Laufzeit*-Tests kauft ein literales
Slice-Spec nichts, kostet aber die volle Typ-Maschinerie. Eine Hausregel „View-Aufbau in
Laufzeittests mit gewidenten Specs" würde künftige Arbeitsregel-12-Blöcke deutlich billiger
machen (FOLLOWUPS).

**Weitere Pins:** `check:diag:stress` **116.053 @ 82** (Δ+119) · `check:diag:browser`
**2.142 @ 75** (Δ0) · `bench:editor` **uniform +119**, neu gepinnt
`{w1 37.573, w2 39.406, w3 70.548, w4 37.728, w5 43.027, w6 44.221, w7 36.777, w8 44.466}`,
zweimal byte-identisch. Der identische Wert +119 auf `stress` (das `spike/src` direkt
kompiliert) bestätigt die Attribution unabhängig: reine Klassen-Surface-Ripple aus **einer**
Signatur, entsprechend kleiner als S4s +436 aus drei Overloads.

Testzahlen: `test:core` 1591 (Δ0) · `test:resident` **6122+2** (Δ+256) · `test:threaded`
**139** (Δ+12) · `cargo` **222+1** (Δ+18) · `test:browser` 4 · `test:package` 3.

### Eine Zahl bleibt ungeklärt, und das steht hier statt einer Scheinpräzision

Die Literal-vs-dynamisch-Mehrkosten einer `topk`-Aufrufstelle wurden **dreimal gemessen**
und ergaben **drei verschiedene Werte**: Baustein 0 als `NDArray`-Proxy **≈122**, der
Implementierer am echten `WNDArray.topk` **≈149**, Baustein A mit eigener Sonde **≈95**.

Alle drei liegen in derselben Größenordnung und stützen dieselbe Schlussfolgerung — der
Aufschlag ist gegenüber dem Gate irrelevant und **nirgends in der Nähe von S3s
+6.705/Site**, worauf die v1-Budgetdisziplin sich gestützt hatte. Die Streuung selbst ist
der Befund: der Instantiation-Zähler ist **sichtbar sensitiv gegenüber dem umgebenden
Boilerplate der Sonde** (Baustein 0 mass in einer neuen Datei und zahlte deren Fixkosten
mit). Es wird deshalb **keine der drei Zahlen zur Wahrheit erklärt**; dokumentiert ist die
Spanne **≈95–149** mit dem benannten Mechanismus. Die v1-Disziplin („Differentialtests
müssen `k` dynamisch verwenden") war überstreng und wurde in v2 auf eine Vorsicht mit
Messpflicht zurückgestuft.

## Verify-Runde

### Baustein 0 (vor der ersten Codezeile)

Beide Schwerpunkte hielten dem Angriff stand (Totalordnung analytisch, NaN-Payload
empirisch mit geschlossener Präzedenzfall-Lücke — beides oben). Ein **MAJOR**: der
abi.rs-Append-Punkt (oben). Dazu drei Präzisierungen — die Grenzgruppe im
Gleichstands-Raster muss auch einmal aus **mehreren NaN** bestehen (anderer
Komparator-Zweig), wide-`k` degradiert zu `readonly [number]`, und die M3-Abgrenzung für
Methoden-Rückgabetypen — plus zwei praktische Warnungen, die je einen Fehlversuch
ersparten: der neue `CoreExports`-Member gehört nach `loader.ts` (ein Merge-Block in
`resident.ts` ist ein harter **TS2440**, der Verifier lief selbst hinein und reproduzierte
es isoliert), und **TypeScript 7.0.2 nativ exportiert die klassische Compiler-API nicht**
(`Object.keys(ts).length === 2`) — die LSP-Messung muss über das Rohprotokoll laufen.

Und ein Befund gegen die Spec, der eine Vorsichtsmaßnahme als überstreng entlarvte: die
Literal-Kosten wurden **gemessen statt geschätzt** (oben).

### Baustein A — Spec-Konformität: konform, keine Blocker

Alle D1–D10 und T1–T8 abgehakt, **alle 15 Gates grün mit reproduzierten Exit-Codes**, jede
berichtete Zahl exakt reproduziert. Freeze-Beweis in beide Richtungen selbst nachgebaut.

A's eigener Mutant saß bewusst in einer anderen Sprache und Fehlerklasse als der des
Implementierers: er entfernte im TS-Lifecycle das Freigeben des ersten Ausgabepuffers beim
Scheitern der **zweiten** Allokation. Ergebnis: exakt **1 von 40** Lifecycle-Tests fiel, mit
benanntem Assert und der 16-Byte-Leck-Bilanz im Output. Zusätzlich hat A den Mutanten des
Implementierers unabhängig nachgestellt und **exakt dieselben Fangzahlen** erhalten
(3 cargo / 20 resident / 4 threaded).

Beide quantitativen Behauptungen hat A durch Ausbau nachgerechnet: der View-Block kostet
exakt 3.926 (52,0 %), der memory.grow-Test exakt Δ0.

**Zur Import-Zeile urteilt A: zulässig**, mit drei Gründen — D6 skopiert „insertion-only"
ausdrücklich auf den *Klassenkörper*, die Import-Zeile steht auf Modulebene; ein
`import type` wird beim Kompilieren gelöscht und kann das Verhalten eines Bestandsmembers
strukturell nicht ändern; und S3 hat sogar **zwei** solche Zeilen verbreitert und eine volle
Drei-Verifier-Runde passiert (verifiziert). Nit: die Hausregel sollte das einmal explizit
sagen, damit die Frage nicht jede Scheibe neu auftaucht.

**A hat eine eigene Lücke offengelegt statt sie zu überspielen:** von den zwei
D5-Fehlerpfaden hat A nur einen selbst mutiert, für den anderen stützte sich das Urteil auf
Code-Lektüre und strukturelle Symmetrie. Baustein B hat ihn unabhängig geschlossen.

### Baustein B — adversarial: die Scheibe hält, ein Befund

**Beide D5-Fehlerpfade sind jetzt einzeln mutations-bewiesen** (B hat den zweiten
nachgeholt), je genau ein benannter Test, mit byte-genauer Ledger-Bilanz statt
Seitengranularität.

**Der Befund (niedrig, geschlossen): ein Testkommentar behauptete mehr, als sein Test
beweist.** `nt_topk_strided` validiert zwei Ausgabe-Regionen getrennt; der bestehende
cargo-Test trug den Kommentar, ein kaputtes `out_len` auf dem *indices*-Puffer werde
„exactly like" eines auf dem values-Puffer abgelehnt. Der Aufruf setzte aber beide Pointer
auf `0` mit demselben `out_len` — der erste Check greift immer zuerst. B hat einen
diskriminierenden Fall konstruiert und per Mutant belegt: entfernt man den zweiten Check,
fällt **nur** der neue Test, der alte bleibt grün.

**Das ist das dritte Vorkommnis derselben Klasse in dieser Kampagne** (S3: Leck-Test drei
Größenordnungen zu grobkörnig; S4: eine „unabhängige Bestätigung", die 500 × 8 Byte gegen
64-KiB-Seiten hielt) — eine Assertion, die wie Abdeckung aussieht. Geschlossen: der
diskriminierende Test ist ergänzt (mutations-bewiesen), und der überclaimende Kommentar
wurde auf das zurückgezogen, was er wirklich prüft.

**B hat die Tie-Blindheits-Lektion präzisiert und damit korrigiert.** Der Implementierer
berichtete „0 von 405 randomisierten Fällen". B misst genauer: **0 von 140 Fällen aus
stetiger Verteilung** — die Spezialwert-Fälle fangen den Mutanten dagegen gelegentlich
(6 von 66), weil sie aus einem **diskreten** Wertevorrat ziehen. Die Lektion lautet also
nicht „randomisiert ist blind", sondern „**stetig verteilt** ist blind, diskret gezogen ist
unzuverlässig, konstruiert ist zuverlässig". Alle **11 von 11** konstruierten
Gleichstands-Fälle fingen ihn.

Weitere Angriffe, die hielten: Aliasing-Mutant (244 Tests fielen), Validierungs-Reihenfolge,
Freeze-Dreiteilung, LSP-Hover, Budget-Bisektion, README-Empirie. `k = Infinity` war korrekt,
aber ohne benannte Assertion — ergänzt.

### Baustein C — Vertrag: keine Verstöße, drei Auslegungsfragen

C hat drei Fragen an den Owner formuliert, zwei davon mit fertigem Wortlaut:

1. **Der M1-NaN-Payload-v6-Kandidat wird durch diese Scheibe NICHT geschlossen, sondern
   geschärft** — C hat meine Fragestellung korrigiert. Der Kandidatentext nennt den
   Vorbehalt „Payload implementierungsdefiniert **für Arithmetik-Ergebnisse**". `topk` liegt
   auf der *gegenüberliegenden* Seite: keine Arithmetik, Payload-Erhaltung bewiesen. Die
   Scheibe liefert also den fehlenden **Gegenfall, der die Scope-Grenze erst sichtbar
   macht** — ohne ihn würde ein v6-Leser den „NaN nur als Wert-Klasse"-Satz auf alle Kernel
   anwenden. Wortlaut-Vorschlag: arithmetik-erzeugende Kernel (NaN nur als Wert-Klasse) von
   wert-weiterreichenden Kerneln (Bit-Identität inkl. exaktem Payload) trennen.
2. **M3 und der Objekt-Rückgabetyp — eine Inkonsistenz in der eigenen Projekt-Historie.**
   C bestätigt die wörtliche Lesart (ein Methoden-Rückgabetyp fällt nicht unter M3s
   Wortlaut, COVENANT.md:70 definiert „Klassen-Hover" als Typ-Parameter-Anzeige der Klasse),
   zeigt dann aber: **in S3 wurde ein strukturell identisches Problem als echte
   M3-Verletzung gewertet, für +5.498 Instantiations gefixt und das Gate eigens von +8.000
   auf +13.000 angehoben.** Entweder war die S3-Einstufung zu weit, oder M3 sollte den Fall
   decken, weil das Projekt ihn real als bindend behandelt und dafür bezahlt hat.
   Owner-Entscheidung, Wortlaut-Vorschlag liegt vor.
3. **Die TS-Insertion-only-Disziplin hat keine vertragliche Ankerdeckung** — M4 nennt nur
   drei Rust-Dateien; die `resident.ts`-Regel lebt ausschließlich in CLAUDE.md. Kein
   Verstoß, aber v6 könnte klären, ob sie vertraglich gemeint ist.

C hat außerdem sauber offengelegt, was es **nicht** geprüft hat (Freeze-Reproduktion,
tatsächliche Bit-Identität, alle Gate-Zahlen — alles buildpflichtig und außerhalb eines
read-only-Mandats).

### Arbeitsregel 13: dreifach unabhängig gemessen

Der Implementierer, Baustein A und Baustein B haben je eine echte `tsc --lsp --stdio`-Messung
mit einer Bestandsmethode als Kontrollpunkt gefahren. Alle drei: der Rückgabetyp hovert als
aufgelöstes Objekt-Literal `{ values: WNDArray<[3]>; indices: WNDArray<[3]>; }`, **kein
Alias-Leck** — die S3-Fehlerklasse ist by construction abwesend, weil `NDArray.topk`s
inline Objekt-Literal gespiegelt wurde statt ein Top-Level-Alias eingeführt.

## Befund-Schließung nach der Verify-Runde

| Befund | Auflösung | Beweis |
|---|---|---|
| B: überclaimender Testkommentar am zweiten ABI-Regionscheck | diskriminierender cargo-Test ergänzt, Kommentar zurückgezogen | Mutant (zweiter `validate_region` → `Ok(())`): **nur** der neue Test fällt, der alte bleibt grün; Revert per Backup-Kopie, SHA-256 identisch |
| B: `k = Infinity` ohne benannte Assertion | Cross-Surface-Assertion ergänzt, `k` dynamisch (`as never`) | Stem `topk: k must be a non-negative integer (got Infinity)`, beide Flächen wortgleich |

Kosten der Schließung: `check:diag` **Δ0** (der Rust-Test liegt außerhalb des TS-Korpus,
die TS-Assertion nutzt `as never` und berührt die `TopkCheck`-Maschinerie nicht),
`bench:editor` Δ0 auf allen acht Pins, Freeze-Hash unverändert, `cargo` +1. Die Gate-Marge
blieb bei 449.

**Auf eine dritte Verify-Runde wurde bewusst verzichtet** — Test-Additionen ohne
Produktivcode-Änderung, je mit eigenem Mutanten-Beweis, voller Gate-Block erneut grün. Die
Entscheidung steht hier, statt still getroffen zu werden.

## Offengelegte Grenzen

- **`k = -0` liefert eine Shape `[-0]` statt `[0]`** (B, empirisch). Vorbestehend und
  identisch auf `NDArray.topk` seit W1 (derselbe `[kNum] as unknown as OkShape<…>`-Cast),
  von dieser Scheibe unverändert übernommen; Cross-Surface-Parität hält also. Funktional
  irrelevant (`allocBytes`' `ptr === 0 && bytes !== 0`-Guard behandelt `-0` korrekt).
  FOLLOWUPS.
- **Der zweite ABI-Regionscheck ist vom legitimen TS-Aufrufpfad unerreichbar** —
  `WNDArray.topk` allokiert beide Puffer stets frisch. Der ergänzte Test sichert eine
  defense-in-depth-Zusicherung, keinen erreichbaren Bug.
- **Baustein B hat `test:example` nicht neu ausgeführt** (Registry-Install, von dieser
  Scheibe unberührt); Implementierer, Baustein A und die Schließungsrunde haben es
  ausgeführt, Exit 0.
- **Die Literal-Kosten-Zahl bleibt eine Spanne** (≈95–149), siehe oben.
- **A hat nur einen der zwei D5-Pfade selbst mutiert**; den zweiten hat B geschlossen.
  Beide sind damit belegt, aber von verschiedenen Instanzen.

## FOLLOWUPS-Kandidaten

- **`slice`-Literal-Kosten als Budget-Hebel** (der wertvollste): literal-argumentige
  `.slice()`-Aufrufstellen in *Laufzeit*-Testblöcken zahlen die volle
  `SliceSpecsGuard`/`SliceShape`-Maschinerie pro Site — hier +3.926 für 110 Zeilen, 52 % der
  Scheibe. Für Laufzeittests kauft das nichts. Eine Hausregel „View-Aufbau in Laufzeittests
  mit gewidenten Specs" würde künftige Arbeitsregel-12-Blöcke deutlich billiger machen.
- **Hausregel-Wortlaut präzisieren** (A's Nit): „insertion-only" für TS-Klassenkörper
  explizit auf den Klassenkörper skopieren, Modulebenen-`import type`-Verbreiterung
  ausdrücklich erlauben — sonst taucht die Frage jede Scheibe neu auf.
- **`k = -0`-Shape-Anzeige** (vorbestehend, beide Flächen).
- **Drei COVENANT-v6-Kandidaten** aus Baustein C (siehe oben), zwei mit fertigem Wortlaut.

## Gate-Block (Endstand, alle Exit 0)

`pnpm check` (Dreier-Verbund) · `check:diag` **237.379 @ 140** · `check:diag:stress`
**116.053 @ 82** · `check:diag:browser` **2.142 @ 75** · `test:core` **1591** ·
`test:resident` **6122+2** · `test:threaded` **139** · `test:browser` **4** ·
`test:package` **3** · `test:example` · `cargo test` **222+1** · `check:freeze`
**`146afdf6…`** · `bench:editor` Hard Gate PASS · `graph-a-lama query lint` 0/0 · `demo`.

## Damit ist die Kampagne S0–S5 abgeschlossen

`WNDArray` und der threaded-Pfad tragen jetzt alle fünf Dogfooding-Ops: `sqrt`, die vier
Skalar-Overloads, `mean`, `item`, `stack`, `argmax`, `topk`. Die README trägt keine
„TypeScript-runtime only"-Ausnahme mehr — empirisch gegen `spike/src/index.ts` verifiziert,
nicht nur gelesen.

## Nachtrag: der CI-Fund nach dem Commit (2026-07-25)

Der erste CI-Lauf des committeten Stands `6fc2d47` war in **acht von neun Jobs** grün und
**rot auf `freeze`** — lokal war alles grün gewesen. Der Befund gehört hierher, weil er die
Scheibe inhaltlich korrigiert hat.

**Ursache, belegt:** `order.sort_by(...)` war der **erste Gebrauch von `core::slice::sort`
im gesamten Crate** (per Grep über `crates/core/src/` verifiziert — jeder andere Kernel,
auch `argmax` und `matmul`, nutzt ausschließlich Handschleifen). Damit landeten erstmals die
Panic-Sites von Rusts Sortier-Maschinerie im Artefakt. Panic-Sites tragen ihren
Quelldateipfad als String im Binary; zwei davon waren sauber auf `/rustc/<commit-hash>/…`
remapped, **einer nicht**:

```
/Users/marvinmuegge/.rustup/toolchains/1.95.0-aarch64-apple-darwin/…/slice/sort/stable/quicksort.rs
```

Auf Linux lautet derselbe Pfad `/home/runner/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/…`
— andere Bytes, anderer Hash. Ein Rerun desselben Commits lieferte exakt denselben
Linux-Hash: **deterministisch host-abhängig**, kein Build-Nichtdeterminismus.

**Zwei Folgen, beide real:** (1) die cross-host-Byte-Identität war gebrochen, die für alle
vorherigen Kernel galt und in FOLLOWUPS als empirisch bestätigt dokumentiert war;
(2) `build:dist` kopiert genau dieses Artefakt in den npm-Tarball — der Pfad inklusive
Benutzername wäre mitveröffentlicht worden.

**Was NICHT passiert ist, geprüft statt angenommen:** `numtype@0.2.0` frisch aus der
Registry gezogen und durchsucht — **null** Treffer auf `/Users/` oder `rustup`, drin ist
genau der eine remappte `/rustc/…/raw_vec/mod.rs`. Ebenso die gesamte git-Historie: es wurde
**nie** ein `.wasm` committet (`git log --all --diff-filter=A -- '*.wasm'` ist leer,
`.gitignore:9` greift), und `git grep` über `git rev-list --all` findet in keiner getrackten
Datei einen Host-Pfad. Der Commit `6fc2d47` enthält nur Quelltext, Tests und Doku; die eine
artefaktbezogene Zeile ist ein SHA-256, kein Pfad. Der Leak existierte ausschließlich in der
lokalen Binärdatei.

**Fix (Owner-entschieden): Ursache entfernen statt kaschieren.** Die naheliegende Abkürzung
— den Linux-Hash als zweiten Plattform-Pin eintragen, was `check-freeze-hash.mjs` sogar
ausdrücklich anbietet — hätte ein Artefakt festgeschrieben, das nicht reproduzierbar ist und
einen Home-Pfad trägt. Stattdessen ersetzt ein **selbst geschriebener In-Place-Heapsort**
die std-Sortierung:

- Er läuft auf dem **bereits vorhandenen** Max-Heap der „Schlechtigkeit" und nutzt dessen
  Sift-Down-Routine (in eine gemeinsame Funktion extrahiert, Verhalten unverändert).
- **O(k log k) bleibt erhalten.** Eine Insertion Sort war ausdrücklich ausgeschlossen: bei
  `k = n` — ein legitimer, getesteter Aufruf — wäre der Kernel quadratisch geworden.
- Der temporäre `order`-Vec entfällt vollständig, **keine zusätzliche Allokation**.
- **Dass Heapsort instabil ist, spielt nachweislich keine Rolle** — genau diese Scheibe hat
  bewiesen, dass die Ordnung eine strikte Totalordnung auf paarweise verschiedenen Indizes
  ist, es also gar keine Gleichstände gibt, zwischen denen Stabilität entscheiden könnte.
  Der tragende Befund der Spec zahlt sich hier direkt aus.

**Verifikation:** `grep` über `crates/core/src/` findet kein `sort_by`/`sort_unstable`/
`.sort(` mehr (Exit 1) · `strings -a` auf dem Artefakt findet **null** Host-Pfade (Exit 1),
übrig bleibt genau der remappte `/rustc/…/raw_vec/mod.rs` — **dasselbe Pfad-Profil wie das
publizierte 0.2.0** · Artefakt **106.647 → 94.358 Bytes** (−11,5 %) · **die Testinhalte
wurden nicht angefasst** und sind unverändert grün (cargo 222+1, test:resident 6122+2,
test:threaded 139), inklusive Gleichstands-Raster, vier View-Klassen und NaN-Payload ·
alle TS-seitigen Pins Δ0 (check:diag, stress, browser, bench:editor) · neuer Freeze-Pin
`2a54d9fdba55e4e88a9d54cb3b01e111c2717abf13017f778b90accd5cff87e4`, doppelt clean-rebuilt.

**Der Mutanten-Beweis ist ungewöhnlich scharf:** die Vergleichsrichtung in `sift_down`
gedreht fällte 8 von 18 topk-cargo-Tests, darunter namentlich
`topk_k_equals_n_is_the_whole_vector_sorted`. Bei `k = n` treten **null** Evictions auf (der
Heap füllt sich exakt bis `size == k == n`, der Eviction-Zweig wird nie betreten) — dieser
Test durchläuft also ausschließlich den **neuen** Sortierpfad. Sein Fehlschlag beweist, dass
speziell die neue Sortierung nicht-vakuös getestet ist, nicht nur die Selektion darum herum.

**Die eigentliche Lehre steht als Arbeitsregel 14 in CLAUDE.md:** lokal war der Fehler
unsichtbar — jeder Gate-Block war grün. Gefunden hat ihn ausschließlich der cross-host
laufende CI-`freeze`-Job, dessen Plattform-Unabhängigkeit bis dahin ein unbemerkter
Nebeneffekt war und sich hier als load-bearing erwies. Und: dass ein Prüfskript einen
bequemen Ausweg anbietet („neuer Plattform-Pin"), heißt nicht, dass er hier gemeint ist —
wo eine Plattform vorher identisch baute, ist die Abweichung ein Befund und kein Pin-Anlass.
