# WASM-Parität S5: `topk` auf `WNDArray`/threaded — bindende Spec

Status: **bindend** (Owner-Auftrag 2026-07-25: „jetzt S5 (topk)") ·
Version: **2** (nach Baustein 0 — D6-Append-Punkt korrigiert, Budget-Disziplin mit
gemessener Zahl hinterlegt, M3-Abgrenzung präzisiert, zwei Testplan-Präzisierungen;
Addendum am Dateiende) · Datum: 2026-07-25 · Eskalationsleiter: **Stufe 3** (Baustein 0
vor der ersten Codezeile, Verify-Katalog A+B+C).

**Letzte Scheibe der Kampagne S0–S5.** Vorgänger: S0 `sqrt`, S1 Skalar-Overloads,
S2 `mean`, S3 `item`+`stack`, S4 `argmax`. Semantische Primärquellen, die hier ZITIERT
und nicht neu hergeleitet werden: [op-w1-argmax-topk-spec.md](op-w1-argmax-topk-spec.md)
D3/D4 (API-Form + Totalordnung) und
[op-topk-selection-ergebnisse.md](op-topk-selection-ergebnisse.md) (der gemessene
Heap-Ersatz der JS-Selektion).

## Ziel & Warum

`NDArray.topk(k)` existiert seit W1 nur auf der naiven JS-Klasse. Diese Scheibe schließt
die letzte Paritätslücke der Kampagne: `WNDArray.topk` (und damit automatisch der
threaded-Pfad), bit-identisch zur TS-Referenz `topkRuntime`.

## Arbeitsregel-11-Prüfung — **Verdikt: KERNEL**

Wie bei S4 und aus denselben Gründen: der Kernel-Bestand enthält keine Vergleichs-,
Sortier-, Auswahl- oder Index-produzierende Operation, aus der sich `topk` komponieren
ließe (S4s Grep-Befund gilt unverändert, `argmax` hat daran nichts geändert — es gibt
keinen Kernel, der eine Rangfolge liefert). Die kernel-lose TS-Variante scheidet aus
demselben Grund aus wie bei `argmax`: `topk` ist eine O(n log k)-Selektion über residente
Daten, dieselbe Klasse wie `sum`/`dot`/`norm_sq`/`argmax`, die alle einen Kernel haben.

**Konsequenz:** M1 bindet neu, der Freeze-Hash bewegt sich legitim von `eba6ba7a…`
(S0/S1/S4-Muster), Arbeitsregel 10 greift (ein neuer `CoreExports`-Member → Stubs in
**beiden** Mock-Dateien, S4-Korrektur).

## Die geerbte Annahme „härteste M1 der Kampagne" ist zu korrigieren

FOLLOWUPS.md und CLAUDE.md tragen seit der topk-Selektions-Scheibe den Satz, ein künftiger
`nt_topk`-Kernel **solle den Heap-Algorithmus spiegeln**, und bezeichnen S5 als die
härteste M1 der Kampagne. Diese Einschätzung stammt aus der Zeit vor der Phase-2-Umsetzung.
Der Doc-Kommentar von `topkRuntime` (runtime.ts:676-687) enthält inzwischen selbst das
Argument, das sie entkräftet:

> `topkCompareValues` kombiniert mit `|| (idxA - idxB)` ist eine **strikte Totalordnung**
> auf den paarweise verschiedenen Indizes `0..n-1`. Es gibt daher **genau eine** korrekte
> top-`k`-Indexmenge in **genau einer** Reihenfolge — Heap und Full-Sort MÜSSEN
> übereinstimmen.

Daraus folgt die für diese Scheibe tragende Unterscheidung:

- **Bit-Identität hängt NICHT vom Algorithmus ab**, sondern ausschließlich (a) von der
  Ordnungs-Prädikat-Transliteration und (b) davon, dass `values[i] = data[indices[i]]` ein
  reiner Element-**Kopiervorgang** bleibt, der nie durch Arithmetik läuft.
- Das ist **kategorial anders als bei `sum`**, wo die Akkumulationsreihenfolge die Bits
  tatsächlich ändert und der Kernel deshalb die Schleifenordnung der Referenz spiegeln
  MUSS. Bei `topk` findet **überhaupt keine Gleitkomma-Arithmetik statt**.

**Bindende Festlegung:** der Kernel implementiert denselben größenbeschränkten Heap
(O(n log k)) wie die JS-Referenz — weil er der gemessen bessere Algorithmus ist und die
Referenz direkt transliterierbar vorliegt, also kein Zusatzrisiko entsteht. Aber die
Heap-Struktur ist **NICHT die M1-Verpflichtung**. Diese Unterscheidung ist testrelevant:
die Tests dürfen sich **nicht** auf Heap-Interna festlegen (Sift-Reihenfolge, Wurzelwahl,
Zwischenzustände), sondern prüfen ausschließlich das beobachtbare Ergebnis gegen
`topkRuntime`. Ein Test, der Heap-Interna pinnt, verhinderte künftige legitime
Optimierungen ohne Korrektheitsgewinn.

**Was dadurch NICHT leichter wird — der echte M1-Risikopunkt liegt woanders:** weil
`values` echte **Datenwerte** trägt (anders als `argmax`, das nur Indizes zurückgibt),
greift hier zum ersten Mal in der Kampagne der **NaN-Payload-Vorbehalt** wirklich
(Baustein-C-Befund aus S4, FOLLOWUPS-v6-Kandidat). `values[i] === data[indices[i]]` gilt
byte-exakt inklusive nicht-kanonischer NaN-Bitmuster. Ein f64-Load/Store in WASM erhält
Payloads (nur Arithmetik darf kanonisieren), und der Repo-Präzedenzfall existiert
(transpose-Payload-Fixture, Kern 10) — **das ist eine zu beweisende Behauptung, keine
Annahme** (T3).

## Berührte Covenant-Invarianten (v5)

- **M1** — bindet neu. Bit-Identität zu `topkRuntime` inkl. NaN-Payloads (siehe oben).
- **M3** — `TopkCheck` wird unverändert wiederverwendet (zweite Call-Site); Compile-Fehler
  erscheint AM `k`-Argument, auch für ein Rang-Problem des Empfängers (DotCheck-Präzedenz);
  Message-Stämme wortgleich. Rückgabetyp ist ein **inline Objekt-Literal-Typ**, kein
  Top-Level-Alias in Rückgabeposition (S3/Arbeitsregel-13-Lektion) — `NDArray.topk`
  (ndarray.ts:811-819) macht es bereits genau so. **Arbeitsregel 13 greift → LSP-Messung
  ist Pflichtbestandteil des Verify-Katalogs.**
  **v2-Abgrenzung (Baustein-0-Befund, wichtig für die Bewertung eines Hover-Fundes):**
  COVENANT v5 definiert „Klassen-Hover" ausdrücklich als die **Typ-Parameter-Anzeige der
  Klasse** und erlaubt Member-Hovern sogar readonly-Modifier (COVENANT.md:70). Ein
  METHODEN-RÜCKGABETYP fällt damit strenggenommen **nicht** unter M3s Wortlaut. Die Wahl
  „inline Objekt-Literal statt Top-Level-Alias" ist also eine aus S3 gelernte, freiwillig
  gehaltene Qualitätsnorm ÜBER M3 hinaus, kein Vertragsgebot. Konsequenz: findet die
  LSP-Messung eine Unschärfe im Rückgabetyp-Hover, wird sie **nicht automatisch als
  M3-Verstoß** gewertet, sondern als **neunter COVENANT-v6-Kandidat** geführt (analog zum
  bestehenden M3-Cross-Surface-Kandidaten). Die Unterscheidung „Vertragsbruch" vs. „geerbte
  Best Practice" gehört sauber ins Ergebnis-Doc.
- **M4** — append-only; Freeze-Beweis dreiteilig (D7).
- **M2** — keine neue Typ-Maschinerie (`TopkCheck`/`TopkShape`/`Guard`/`OkShape`
  unverändert, `vector.ts` wird NICHT editiert).
- **M5/Z1/Z2** — nicht berührt.

## Bindende Entscheidungen

### D1 — Surface-Scope

`WNDArray.topk` als **Instanz-Methode**, eine einzige Signatur (keine Overloads).
Threaded-Parität automatisch (dasselbe Crate), per Test belegt. **Keine Facaden-Änderung**
(`WasmBackend`/`ThreadedBackend` bleiben unberührt) — dieselbe Begründung wie S4/argmax:
`topk` ist eine Instanz-Methode auf einem Handle, das Konsumenten ohnehin halten, anders
als S3s statisches `stack`.

### D2 — API-Form (exakter Spiegel von `NDArray.topk`, ndarray.ts:811)

```ts
topk<const K extends number>(
  k: Guard<TopkCheck<S, K>, K>,
): { values: WNDArray<OkShape<TopkShape<S, K>>>; indices: WNDArray<OkShape<TopkShape<S, K>>> };
```

- Rang-1-only (DotCheck-Familie); ein Rang-Problem des EMPFÄNGERS wird AM `k`-Argument
  gemeldet — das ist die etablierte Hauspolitik, keine Neuerfindung.
- `TopkCheck`/`TopkShape` werden **unverändert** wiederverwendet (zweite Call-Site).
- Die Rückgabe sind **zwei unabhängige, residente Handles**, die der Aufrufer selbst
  disponiert. Das ist der erste Fall der Kampagne mit ZWEI Ergebnis-Handles aus einem
  Aufruf; die Lebenszyklus-Folgen stehen in D5.
- `indices` trägt f64-integrale Indizes (numtype ist f64-only), im Doc-Kommentar zu
  benennen wie auf `NDArray.topk`/`argmax`.

### D3 — Kernel-Design

Ein neuer ABI-Einstiegspunkt, **strikt ans Ende von `abi.rs` angehängt**:

```rust
#[no_mangle] pub extern "C" fn nt_topk_strided(
    shape_ptr: u32, rank: u32, strides_ptr: u32, offset: u32,
    data_ptr: u32, data_len: u32, k: u32,
    out_values_ptr: u32, out_indices_ptr: u32, out_len: u32,
) -> u32;
```

Rechenkern in **neuer Datei** `crates/core/src/kernels/topk.rs`. Struktur:

1. **Ordnungs-Prädikat** — wörtliche Transliteration von `topkCompareValues`
   (runtime.ts:645) plus dem `|| (idxA - idxB)`-Indextiebreak (runtime.ts:719). NaN zuerst,
   dann absteigender Wert, dann aufsteigender Index. Rusts `f64`-`>`/`<` und `f64::is_nan`
   entsprechen den JS-Operatoren exakt (`0.0 > -0.0` ist beidseitig falsch, NaN-Vergleiche
   beidseitig falsch) — **die Bit-Identität folgt daraus, wird aber nicht behauptet,
   sondern differentiell bewiesen** (T3).
2. **Selektion** — größenbeschränkter Max-Heap „der Schlechtigkeit" (Wurzel = schlechtestes
   der `k` gehaltenen Elemente), O(n log k), Transliteration von runtime.ts:714-774.
3. **Abschluss-Sortierung** der gehaltenen `k` Indizes mit demselben Prädikat
   (runtime.ts:780-781), dann `values[i] = data[…]` als **reiner Element-Kopiervorgang**,
   **niemals** über einen Zwischenwert aus dem Heap und **niemals** durch Arithmetik
   (runtime.ts:785-789 tut genau das und begründet es).
4. **Strided Lesen:** das Element zum logischen Index `i` ist
   `data[offset + i * strides[0]]`. **Die zurückgegebenen `indices` sind LOGISCHE Indizes
   `0..n-1`, nie Speicher-Offsets** — dieselbe Falle wie bei `argmax` (S4/D3), hier
   zusätzlich mit der Verschärfung, dass ein verwechselter Index über `values[i] =
   data[indices[i]]` auch die WERTE falsch machen würde.
5. **`k = 0`:** zwei leere Ausgaben, kein Fehler. **Rang ≠ 1 / `k` ungültig / `k > n`:**
   `KernelError::ShapeIncompatible` (von TS aus unerreichbar, weil D4 prävalidiert; per
   cargo-Test gepinnt, defense in depth).
6. **Pflicht-Doc-Kommentar** zur `unwrap_or(0.0)`-Konvention, falls sie übernommen wird —
   dieselbe Asymmetrie-Warnung wie in S4/argmax.rs (der Fallback wäre hier ein
   Vergleichs-Kandidat UND ein kopierter Wert, also doppelt nicht-neutral).

### D4 — Runtime-Semantik: Fehlerpfade, TS-prävalidiert

`WNDArray.topk` prüft **vor jeder Allokation und vor dem Kernel-Aufruf**, in **exakt der
Reihenfolge** von `topkRuntime` (runtime.ts:694-703), mit wortgleichen Stämmen:

| Reihenfolge | Bedingung | Stem |
|---|---|---|
| 1 | `shape.length !== 1` | `topk: expected a 1-D vector (got shape [${shape}])` |
| 2 | `!Number.isInteger(k) \|\| k < 0` | `topk: k must be a non-negative integer (got ${k})` |
| 3 | `k > n` | `topk: k=${k} exceeds the vector length ${n}` |

Die Reihenfolge ist load-bearing: sie ist dieselbe, die `TopkCheck` auf der Typ-Ebene
prüft, und ein Vertauschen wäre für einen Aufruf, der zwei Bedingungen gleichzeitig
verletzt, beobachtbar. Ein Kernel-Status ≠ 0 wirft weiterhin die etablierte
`wasm resident nt_topk_strided: status …`-Form. `this.assertLive("topk")` als erste
Anweisung.

### D5 — Lifecycle (der heikelste Teil dieser Scheibe)

Erstmals gibt eine Op **zwei** residente Handles zurück. Bindend:

- Beide Ausgabepuffer werden allokiert, **bevor** der Kernel läuft. Schlägt die **zweite**
  Allokation fehl, muss die **erste** freigegeben werden, bevor der OOM-Fehler propagiert
  (das etablierte Scratch-Listen-Muster).
- Bei Kernel-Status ≠ 0 werden **beide** Ausgabepuffer freigegeben, bevor geworfen wird.
- Im Erfolgspfad gehen beide Puffer je an ein eigenes `WNDArray.fresh`-Handle über; sie
  dürfen **niemals** denselben Puffer teilen oder aliasen.
- Shape-/Strides-Scratch wird wie überall im `finally` freigegeben.
- `k = 0`: `allocBytes(core, 0)` ist zulässig und wirft nicht (`allocBytes` prüft
  `ptr === 0 && bytes !== 0`, resident.ts:87) — die beiden `[0]`-Handles sind gültig.

**Testpflicht ab Tag 1, nicht als Verify-Nachtrag** (die S4/V2-Lektion: genau dieser Pfad
war dort von 0 von 334 Tests abgedeckt und nur per erzwungenem Kernel-Fehlschlag
sichtbar): ein Mock, der `nt_topk_strided` auf Status ≠ 0 zwingt, plus ein Mock, der die
ZWEITE Allokation fehlschlagen lässt — je mit exakter Alloc/Free-Bilanz.

### D6 — Datei-Disziplin

| Datei | Disziplin |
|---|---|
| `crates/core/src/kernels/topk.rs` | **NEU** |
| `crates/core/src/kernels/mod.rs` | ein Append |
| `crates/core/src/abi.rs` | **ein Append strikt ans tatsächliche Dateiende** — siehe die v2-Korrektur direkt unter dieser Tabelle |
| `spike/src/wasm/loader.ts` | ein neuer `CoreExports`-Member |
| `spike/tests-runtime/backend-oom.test.ts` **und** `resident-lifecycle.test.ts` | je ein `notImplemented`-Stub (Arbeitsregel 10 in der **korrigierten** Fassung: es sind ZWEI exhaustive Literale; Spread-basierte Mocks brauchen keine) |
| `spike/src/wasm/resident.ts` | **insertion-only** in den Klassenkörper |
| `spike/src/runtime.ts`, `vector.ts`, `reduce.ts`, `index.ts`, `threaded.ts`, Facaden | **UNVERÄNDERT** |

**v2-Korrektur des abi.rs-Append-Punkts (Baustein-0-Befund, MAJOR).** v1 nannte
`nt_argmax_axis_strided` „das heutige letzte Item" — das ist falsch. Das letzte Item der
Datei ist `#[cfg(test)] mod s4_argmax_abi_tests` (Zeilen 1710-1733; die Datei endet
buchstäblich mit dessen schließender Klammer, per Hex-Dump belegt). `nt_argmax_axis_strided`
ist nur die letzte **reale Funktion**.

Der Unterschied ist keine Haarspalterei, weil die Datei ein **durchgängiges, bisher nie
gebrochenes Muster** hat: jede Phase hängt ihren Realcode nach dem **jeweils aktuellen
Dateiende** an — das zufällig immer ein Testmodul ist — und danach ihr **eigenes** neues
Testmodul (`nt_matmul_blocked_partial` nach `mod tests`; Kern-07-Funktionen danach; dann
`mod kern07_abi_tests`; S0/S1/S4-Funktionen danach; dann `mod s4_argmax_abi_tests`).
Ein wörtlich gelesenes v1 hätte `nt_topk_strided` erstmals **vor** ein bestehendes
Testmodul gespleißt.

**Bindend für S5:** `nt_topk_strided` wird **nach** `mod s4_argmax_abi_tests` angehängt,
ein etwaiges eigenes `mod s5_topk_abi_tests` danach. Für den Freeze-Hash ist das folgenlos
(`#[cfg(test)]` fällt im `--release --target wasm32-unknown-unknown`-Build weg und
verschiebt keine Artefakt-Bytes; Baustein 0 hat das geprüft) — die Korrektur dient der
Musterkonsistenz und dem nächsten Leser.

Kein neues Source-File unter `spike/src`, **kein neues Testfile** (Anhänge an bestehende
Dateien) → kein Order-Noise.

**Warnhinweis (Baustein-0-Selbstfund, kein Spec-Fehler):** der neue `CoreExports`-Member
gehört in `loader.ts`, wo alle bisherigen Merge-Blöcke liegen. Ein `interface
CoreExports`-Block in `resident.ts` (das den Typ nur importiert) ist ein harter
**TS2440** — Baustein 0 ist genau da hineingelaufen und hat es isoliert reproduziert.

### D7 — Testplan

**Rust (cargo, in `topk.rs`):** contiguous gegen Hand-Referenz · **strided/transponierte
View** mit Nicht-Vakuitäts-Assertion (Daten so wählen, dass ein Speicher-statt-logisch-Index
nachweislich ein anderes Ergebnis liefert) · Offset · `k = 0` · `k = n` · `k = 1` ·
NaN-Ordnung (einzeln, mehrere → aufsteigender Index) · ±0-Tie · Wert-Ties → aufsteigender
Index · ±Inf · Rang ≠ 1 → `ShapeIncompatible` · `k > n` → `ShapeIncompatible` ·
`n = 0, k = 0`.

**TS-Differential (`spike/tests-runtime/resident.test.ts`, Anhang) — M1-Beweis:**
- Randomisiertes Differential `WNDArray.topk` vs. `topkRuntime` über contiguous **und die
  vier View-Klassen** (transponiert/geschnitten/offset-verschoben/zusammengesetzt —
  Arbeitsregel 12; bei Rang 1 heißt das insbesondere: geslicte Vektoren mit Stride > 1 und
  Offset > 0), über ein `k`-Raster inkl. `0`, `1`, `n/2`, `n`. Vergleich der **Bits**
  beider Ausgaben, nicht `===`.
- **KONSTRUIERTES Gleichstands-Raster, nicht nur Zufallsdaten** — die S4-Lektion, hier
  verschärft: `topk`s Ordnung hat DREI Tie-Ebenen (NaN-vs-NaN nach Index; gleicher Wert
  nach Index; `+0`/`-0` als gleich behandelt). Randomisierte f64-Ziehungen aus stetiger
  Verteilung erreichen Gleichstände mit Wahrscheinlichkeit 0 und sind für diese
  Fehlerklasse **beweisbar blind**. Pflicht: Vektoren mit vielen Duplikaten, mehreren NaN,
  gemischten `+0`/`-0`, und Fälle, in denen die k-Grenze MITTEN durch eine Gruppe
  gleicher Werte läuft (dort entscheidet allein der Index-Tiebreak). **v2-Präzisierung
  (Baustein 0):** die Grenzgruppe muss **auch** einmal aus **mehreren NaN** bestehen, nicht
  nur aus numerisch gleichen Werten — das ist im Komparator ein anderer Zweig
  (`aNaN && bNaN` statt `a === b`) und fällt weg, wenn man „Gleichstand" nur an Zahlen
  denkt. Vorlage im Repo vorhanden: `spike/tests-runtime/argmax-topk.test.ts` (W1) hat
  bereits All-NaN-Vektoren, Payload-Differentiale und den `topkOracleFullSort`.
- **NaN-Payload byte-exakt:** nicht-kanonische NaN-Bitmuster ins Eingabe-Array schreiben
  (roher `DataView`-Zugriff auf den WASM-Speicher) und beweisen, dass `values` sie
  **unverändert** zurückgibt. Das ist der M1-Risikopunkt dieser Scheibe (siehe oben) und
  ein eigener, benannter Test — kein Nebeneffekt eines anderen.
- **`values[i]` ⇄ `indices[i]`-Konsistenz** byte-exakt gegen den Eingabe-Vektor.
- Fehlerpfade: die drei Stämme, **cross-surface wortgleich** `NDArray` ⇄ `WNDArray`;
  Reihenfolge-Pin für einen Aufruf, der zwei Bedingungen gleichzeitig verletzt.
- **Lifecycle (D5):** exakte Alloc/Free-Bilanz im Erfolgspfad (zwei Handles überleben,
  Scratch nicht); erzwungener Kernel-Status ≠ 0 → beide Ausgaben freigegeben; erzwungener
  Fehlschlag der ZWEITEN Allokation → erste freigegeben. Je mutations-bewiesen.

**Threaded (`threaded.test.ts`, Anhang):** threaded-vs-stable-Parität nach dem
S0/S4-Muster inkl. Spezialwerten.

**Typ-Ebene (`spike/tests/ndarray.test-d.ts`, Anhang):** exakte Ergebnis-Shapes für
literale `K`; Degradationskanten (wide `number`-k → `readonly [number]` — Baustein 0 hat
den exakten Typ gemessen, v1 schrieb informell `[number]` —, Union-k → no-claim,
`RankUnknowable` → uniform no-claim) — Spiegel der bestehenden `NDArray.topk`-Pins;
`@ts-expect-error` AM `k`-Argument für negatives k, k > D und Rang ≠ 1.

**Pflicht-Mutant** (Implementierer): den Index-Tiebreak invertieren
(`idxA - idxB` → `idxB - idxA`). Der Katalog muss ihn fangen — wenn nur das
Gleichstands-Raster ihn fängt und nicht die Zufallsfälle, ist das der erwartete Befund und
gehört berichtet. Revert **nur** per Backup-Kopie + SHA-256-Beweis.

### D8 — Pins, Budget, Gates

**Baseline** (HEAD `6d730b2`, im frischen `git worktree` zu bestätigen):
`check:diag` **229.828 @ 140** · stress **115.934 @ 82** · browser **2.142 @ 75** ·
Freeze `eba6ba7ac85d15a814fd027392a81c7450d885d2048d7efd6694b7e8370988bb` ·
`bench:editor` `{w1 37.454, w2 39.287, w3 70.429, w4 37.609, w5 42.908, w6 44.102,
w7 36.658, w8 44.347}` · test:core 1591 · test:resident 5866+2 · test:threaded 127 ·
cargo 204+1.

**Vorregistriertes Absolut-Gate: ≤ +8.000 Instantiations** auf dem Haupt-Pin.
Herleitung: S4 realisierte +3.138 mit DREI Overloads und zwei `CoreExports`-Membern; S5 hat
**eine** Signatur und **einen** Member, also weniger Klassen-Surface.

**v2 — die Literal-Kosten sind GEMESSEN, nicht mehr aus S3 geschätzt (Baustein 0).** v1
befürchtete den S3-`StackFold`-Mechanismus (literal-typisierte Aufrufstellen zahlen die
volle Digit-Maschinerie PRO Site, dort +6.705) und leitete daraus eine harte Disziplin ab.
Baustein 0 hat den Mechanismus per empty-then-fill über einen Proxy gemessen (`NDArray.topk`,
dieselbe unveränderte `TopkCheck`/`TopkShape`-Maschinerie): eine leere Datei kostet −1.301
Order-Noise, eine Datei mit EINER literalen Aufrufstelle +360 echte Kosten, eine mit EINER
dynamischen +238. **Der literal-spezifische Aufschlag ist also ≈+122 pro Aufrufstelle —
rund 55× billiger als S3.** Die v1-Sorge war unbegründet.

Deshalb v2: die Disziplin wird von einer harten Restriktion auf eine **billige Vorsicht mit
Messpflicht** zurückgestuft. Literal-typisierte `topk`-Aufrufstellen sind zulässig, auch in
den Differentialtests; sie werden aber im Ergebnis-Doc gezählt, und Messstufe ③/④ weist
ihren Anteil aus. Bleibt der Aufschlag bei ≈+122/Site, ist das bei jeder realistischen
Anzahl gegenüber dem ≤+8.000-Gate irrelevant. **Caveat, ehrlich:** gemessen wurde über
`NDArray` als Proxy, nicht am finalen `WNDArray`-Code — die Messstufen bestätigen es oder
widerlegen es.

**Mess-Protokoll:** kein neues File → kein Order-Noise erwartet; Deltas **gestuft**
attribuiert (① Kernel/ABI/CoreExports + beide Mock-Stubs, ② `WNDArray.topk`-Methode,
③ Test-Anhänge, ④ Typ-Pins). Exit-Code und Fehlerausgabe jedes Messlaufs berichten
(Arbeitsregel 6). Bei Riss: **STOPP, Befund an den Owner, kein „Optimieren ins Gate".**

**Freeze-Beweis dreiteilig** (S0/S1/S4-Muster): ① Pre-Edit-Clean-Rebuild reproduziert
`eba6ba7a…` exakt, ② additive-only-Dekomposition am Diff, ③ neuer Hash aus Clean-Rebuild
wird der Pin (`scripts/check-freeze-hash.mjs` + CLAUDE.md).

**Gate-Block:** `pnpm check` · `check:diag`(+`:stress`/`:browser`) · `test:core` ·
`test:resident` · `test:threaded` · `test:browser` · `test:package` · `test:example` ·
`cargo test` · `check:freeze` · `bench:editor` · `graph-a-lama query lint` · GFM.

### D9 — README (Hausregel 5)

Diese Scheibe ändert, WO `topk` läuft. Nach S4 ist `topk` die **einzige** verbleibende
„TypeScript-runtime only"-Op; nach S5 gibt es **keine** mehr, und die Kampagne ist
abgeschlossen. Prüf-Kommando `grep -n "TypeScript-runtime only\|no WASM kernel" README.md`;
die Behauptungen danach **empirisch** gegen `spike/src/index.ts` verifizieren
(Wegwerf-Skript mit absoluten Importen), nicht nur lesen. Die Formulierung muss
berücksichtigen, dass die Ausnahmeliste damit leer wird — ein übriggebliebener
Ausnahme-Satz wäre dann selbst die Drift.

### D10 — Sprache

Code/Kommentare/Tests/Commit/README: **Englisch**. Spec + Ergebnisse-Doc: Deutsch.
`≈` statt `~`, keine `~~`-Strikethroughs.

## Akzeptanzkriterien

- **T1:** Alle D8-Gates grün mit Exit-Codes; Freeze-Beweis dreiteilig, neuer Pin gesetzt.
- **T2:** Pin-Protokoll vollständig, Deltas gestuft attribuiert, Gate ≤ +8.000 eingehalten
  (oder STOPP). Die Literal-vs-dynamisch-Disziplin aus D8 ist am Diff nachweisbar.
- **T3 (M1):** Bit-Identität `WNDArray.topk` ⇄ `topkRuntime` über contiguous + vier
  View-Klassen × `k`-Raster × **konstruiertes Gleichstands-Raster**, 0 Abweichungen, mit
  berichteter Fallzahl. **NaN-Payload-Erhalt als eigener, benannter Test bewiesen.**
  Threaded-Parität separat belegt.
- **T4 (M3):** Compile-Ablehnung AM `k`-Argument (Spaltenposition empirisch geprüft);
  die drei Stämme cross-surface wortgleich; Validierungs-REIHENFOLGE gepinnt;
  **LSP-Messung** mit Bestandsmethode als Kontrollpunkt belegt saubere Hover — inklusive
  des Objekt-Rückgabetyps (`{ values: WNDArray<[3]>; indices: WNDArray<[3]> }`).
- **T5 (M4):** Datei-Disziplin am Diff bewiesen.
- **T6 (D5):** Beide Fehlerpfade (Kernel-Status ≠ 0; zweite Allokation schlägt fehl)
  getestet und je mutations-bewiesen; Erfolgspfad-Bilanz exakt.
- **T7:** Pflicht-Mutant gefangen, mit benannten Tests; Revert per Backup-Kopie.
- **T8:** README nach D9; Doc-Platzierung nach Hausregel 5; **die Kampagne S0–S5 wird in
  CLAUDE.md/roadmap/FOLLOWUPS als ABGESCHLOSSEN geführt**, und der M1-NaN-Payload-
  v6-Kandidat wird mit dem hier gewonnenen Befund aktualisiert.

## Nicht-Ziele

- Kein `argsort`/`sort` (eigener FOLLOWUPS-Kandidat), kein `topk` mit Achsen-Parameter,
  kein `gather`/`take`.
- **Keine Performance-Aussage und kein Benchmark.** Die Scheibe stellt Parität her. Der
  Heap wird gewählt, weil er transliterierbar vorliegt, nicht weil hier Geschwindigkeit
  gemessen oder behauptet würde.
- Keine Facaden-Änderung, keine Edits an `runtime.ts`/`vector.ts`, kein neuer
  `bench:editor`-Workload, kein Release/Publish.
- Keine Auflösung der COVENANT-v6-Kandidaten (eigene Vertrags-Scheibe) — S5 liefert nur
  den Befund zum NaN-Payload-Kandidaten.

## Verify-Plan (Stufe 3)

**Baustein 0** (`brainroute:deep`, adversarial, VOR der ersten Codezeile), mit einem
ausdrücklichen Schwerpunkt: **das Totalordnungs-Argument angreifen.** Ist die Ordnung
wirklich total und strikt auf `0..n-1` (NaN-vs-NaN? `+0`/`-0`? gleiche Werte?), und folgt
daraus wirklich, dass der Algorithmus für die Bit-Identität irrelevant ist? Wenn dieses
Argument fällt, fällt die ganze M1-Begründung dieser Scheibe. Zusätzlich: die
NaN-Payload-Behauptung (erhält ein f64-Load/Store durch den WASM-Kernel wirklich
nicht-kanonische Payloads? empirisch prüfen, nicht aus der Spec ableiten); alle
Code-Annahmen (abi.rs-Dateiende, `allocBytes`-size-0-Verhalten, `TopkCheck`-Wiederverwendung
ohne Edit, Overload-/Signatur-Auflösung empirisch typchecken, Rückgabe-Objekt-Hover);
Lifecycle-Löcher bei zwei Ausgaben; Testplan-Lücken; Freeze-Behauptung; die
Literal-vs-dynamisch-Budgetdisziplin auf Plausibilität.

Danach **A + B + C parallel** (Aufträge aus [verify-runde-template.md](verify-runde-template.md)),
Ergebnisse-Doc mit Post-Verification-Addendum, KB-Capture, Commit.

**Praktischer Vorab-Hinweis für die LSP-Messung (Baustein-0-Fund, spart einen Fehlversuch):**
die installierte TypeScript-7.0.2-Native-Distribution exportiert die klassische
Compiler-API **nicht** mehr — `Object.keys(ts).length === 2` sowohl über ESM-Default-Import
als auch über CJS-`require`; `ts.createLanguageService`/`ts.sys` existieren nicht. Die
Hover-Messung muss über das Rohprotokoll von `tsc --lsp --stdio` laufen (die Harness dafür
existiert seit Spike 02 in `spike/bench-dx/editor-latency.ts`), nicht über das
`typescript`-npm-Paket.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-25)

Verifier: `brainroute:deep`, frischer Kontext, alle Proben in einem isolierten Worktree
(sauber entfernt, Haupt-Baum unverändert bestätigt).

**Schwerpunkt 0 — das Totalordnungs-Argument HÄLT** (hohe Konfidenz). Der Verifier hat die
Ordnung unabhängig neu hergeleitet statt die Spec-Prosa nachzubeten: Trichotomie ist für
jedes Paar erfüllt (der Komparator gibt nur in zwei Fällen 0 zurück — beide NaN, oder
numerisch gleich inkl. `+0`/`-0` —, und dort ist `idxA - idxB` bei paarweise verschiedenen
Indizes nie 0, die im f64-Safe-Integer-Bereich für jede realistische Arraygröße exakt
sind); Transitivität folgt aus der Standard-Konstruktion „totale Präordnung + strikter
Tiebreak"; Rusts/WASMs `f64`-Vergleiche und `is_nan` sind IEEE-754-konform und verhalten
sich bitgenau wie die JS-Operatoren. Code-Ebene bestätigt: `values[i] = data[srcIdx]`
(runtime.ts:788) liest frisch aus `data`, nie aus dem Heap-Array. **Damit ist die
Umkehrung der geerbten Projekt-Annahme („härteste M1", „muss den Heap spiegeln")
belastbar.**

**Schwerpunkt 1 — die NaN-Payload-Behauptung HÄLT empirisch**, und der Verifier hat dabei
eine Lücke im Präzedenzfall geschlossen: die bestehende Repo-Fixture deckt nur den
2D-Transpose-Fall ab. Er hat zusätzlich eine **Rang-1, gestridete, Offset > 0**-Sicht mit
einem nicht-kanonischen NaN durch `nt_materialize` geschickt — strukturell genau das
Lesemuster, das `nt_topk_strided` braucht — und die Payload kam byte-exakt zurück.
Ehrlich abgegrenzt: das ist starke Analogie-Evidenz am selben Kernel, kein Test von
`nt_topk_strided` selbst (der existiert noch nicht) — T3 bleibt also zu Recht als zu
beweisende Pflicht stehen.

**MAJOR — D6s Append-Punkt war faktisch falsch** (oben in D6 korrigiert): das letzte Item
von `abi.rs` ist ein Testmodul, nicht die letzte reale Funktion; ein wörtlich gelesenes v1
hätte erstmals in der Kampagne Realcode vor ein bestehendes Testmodul gespleißt. Für den
Freeze-Hash folgenlos (`#[cfg(test)]` fällt im Release-wasm-Build weg, vom Verifier
geprüft), aber musterbrechend.

**Gemessen statt geschätzt:** die Literal-vs-dynamisch-Kosten von `TopkCheck` liegen bei
≈+122 Instantiations pro Aufrufstelle, nicht bei S3s +6.705 — die v1-Budgetdisziplin war
überstreng und ist in D8 zurückgestuft.

**Weitere Befunde eingearbeitet:** die Grenzgruppe im Gleichstands-Raster muss auch einmal
aus mehreren NaN bestehen (anderer Komparator-Zweig, D7); wide-`number`-k degradiert zu
`readonly [number]`, nicht `[number]` (D7); die M3-Abgrenzung für Methoden-Rückgabetypen
(D2); der TS-7.0.2-Compiler-API-Hinweis (oben); und ein Warnhinweis, dass der neue
`CoreExports`-Member nach `loader.ts` gehört — ein Merge-Block in `resident.ts` ist ein
harter TS2440, in den der Verifier selbst hineinlief und den er isoliert reproduziert hat.

**Positiv verifiziert (empirisch, nicht gelesen):** `allocBytes`-size-0-Verhalten ·
`TopkCheck`/`TopkShape` als zweite Call-Site ohne jeden Edit (echter Methodenkörper in
einer Worktree-Kopie kompiliert, `tsc --strict`: literal `K=3` → exakt
`{values: WNDArray<[3]>; indices: WNDArray<[3]>}`, `K=0`/`K=5` korrekt, dynamisches `k`
degradiert sauber) · Fehlerposition **an der Spalte des `k`-Arguments** für alle drei
Fehlerklassen, mit exakt den erwarteten Stämmen · Arbeitsregel 10 in der korrigierten
S4-Fassung (genau zwei exhaustive Literale; die zwei Spread-Overlays brauchen nichts) ·
Reihenfolge der drei Fehlerstämme · D9-Prämisse (`topk` ist heute wirklich die einzige
verbliebene Ausnahme in der README) · Baselines `check:diag` **229.828 @ 140** (Exit 0),
cargo **204+1** (Exit 0), Freeze-Hash **`eba6ba7a…`** aus eigenem Clean-Rebuild
byte-identisch.

**Bewusst nicht reproduziert (offen an Baustein A):** stress-/browser-Pins,
`bench:editor`-Pins, `test:resident`/`test:threaded`-Zahlen.
