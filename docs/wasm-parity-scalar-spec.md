# WASM-Parität S1 — Skalar-Overloads `add`/`sub`/`mul`/`div` auf `WNDArray`/threaded (bindende Spec)

Status: **bindend** (Owner-Richtungsabnahme 2026-07-23: WASM-Backend-Parität der W1–W5-Ops
nachziehen; S1 ist die zweite Scheibe der Serie, nach dem S0-Pilot `sqrt`, und wiederverwendet
dessen komplett etablierte „neuer Kernel"-Pipeline).
Version: 2 · Datum: 2026-07-23 · Eskalationsleiter: **Stufe 3** (substanzielle Scheibe, M1- UND
M4-Anker berührt — voller Verify-Katalog A+B+C). Covenant: v5. **v2:** Baustein-0-Befunde eingearbeitet
(F1: WNDArray-Diagnose-Qualitätstest jetzt VERPFLICHTEND — UW4 schützt nicht gegen die W2-F1-Regression,
empirisch bewiesen; F2: `backend-oom.test.ts`-Mock-Stubs als Pflicht-Verdrahtung in D1/D7/T2; F3:
kuratierte `div(0)`/`sub`-Ordnungs-Fixtures ergänzt; F4: threaded.ts-Zeilencitat korrigiert). Alle
Kern-Design-Annahmen von Baustein 0 empirisch bestätigt (Clean-Rebuild-Hash bei `pub(crate)` unverändert,
+0/Member bei n=4, 36k-Differential 0 Mismatches). Details im Addendum am Ende.
Roadmap: zweite Scheibe der WASM-Parität-Serie (S1 von S0–S5: sqrt → **Skalar-Overloads** → mean →
item/stack → argmax → topk).

## Ziel & Warum — und was diese Scheibe NICHT ist

`NDArray.add`/`sub`/`mul`/`div` tragen seit W2 je einen **Skalar-Overload** (`x.div(2)` liest sich
als „durch 2 teilen", shape-erhaltend — auch bei Rang 0). Diese Overloads existieren heute NUR auf
der naiven JS-Klasse; `WNDArray` (der WASM-residente Zwilling) hat zwar die vier Array-Array-Methoden
add/sub/mul/div (routen durch die binären `nt_*_strided`-Kernel), aber KEINEN Skalar-Overload — wer
einen residenten `WNDArray` durch einen Skalar teilen will, muss heute `x.div(WNDArray.fromArray([1],
[s]))` broadcasten oder nach JS kopieren. Diese Scheibe schließt die Lücke: vier from-scratch
Rust/WASM-Kernel (`nt_scalar_{add,sub,mul,div}_strided`) plus die vier `WNDArray`-Skalar-Overloads,
sodass residente Daten elementweise `data[i] op s` in-WASM verrechnet werden, ohne pro Op nach JS zu
kopieren und ohne einen `[1]`-Broadcast-Umweg.

**Warum Skalar-Overloads als S1 (easy-first-Begründung):** Nach dem `sqrt`-Pilot ist die Pipeline
(neues Kernel-File → ABI-Eintrag → `CoreExports` → `WNDArray`-Methode → M1-Differentialtest →
Freeze-Re-Pin → threaded-Parität) komplett etabliert. Skalar `+`/`-`/`*`/`/` ist **noch einfacher
bit-paritätsfähig als sqrt**: es ist keine transzendente Op, sondern reine IEEE-754-Arithmetik — und
zwar exakt dieselbe Arithmetik, die die bereits EINGEFRORENEN binären Kernel (`nt_add_strided` &
Co., Kern 07) schon bit-identisch bewiesen haben. Der Kernel selbst ist ein Einzeiler über den in S0
gebauten, generischen `unary_strided`-Kern (`|x| x op s`). S1 ist damit die kleinstmögliche
Erweiterung der etablierten Pipeline um eine Op mit einem SKALAREN Parameter — die einzige neue
Zutat gegenüber sqrt.

**Ehrlichkeits-Rahmen (Owner-Vorgabe, wie bei W2/W3/S0):** Diese Scheibe hat **keinen gemessenen
Nutzerbedarf** — niemand ist auf eine WASM-Skalar-Wand gestoßen. Es ist Vollständigkeits-/
Symmetrie-Arbeit (beide Motoren sollen dasselbe können). Das Ergebnisse-Doc darf keinen Nutzerbedarf
suggerieren, den es nicht gibt. „Wir lassen die Skalar-Ops JS-only" bliebe ein zulässiger Zustand;
die Scheibe existiert, weil der Owner die Parität als OSS-Vollständigkeitsziel entschieden hat.

Diese Scheibe ist **kein** Threaded-Pool-Kernel (der Pool routet weiterhin NUR matmul — die
Skalar-Ops laufen auf dem residenten Core, im stable WIE im threaded Artefakt, da beide dasselbe
Crate sind), **keine** Änderung an `NDArray.add`/`sub`/`mul`/`div`/`scalarElementwiseRuntime`
(bereits fertig, bleibt die M1-Referenz), **keine** andere Op (mean/item/stack/argmax/topk folgen als
S2–S5), **kein** `rsub`/`rdiv` (Skalar-Sub ist `data[i] − s`, Skalar-Div ist `data[i] / s` — exakt
wie die JS-Seite; `s − x` / `s / x` bleiben Nutzer-Sache, W2-Nicht-Ziel).

## Die tragende Beobachtung: Skalar-Bit-Parität ist ein KOROLLAR der eingefrorenen binären Kernel

`sqrt` (S0) brauchte eine empirische Vorab-Probe (30.028 Fälle), weil seine Bit-Parität zwar a-priori
plausibel, aber noch nie GEMESSEN war. Für die Skalar-Ops ist die Bit-Parität **kein neuer Claim,
sondern ein Korollar bereits bewiesener Fakten:**

1. **Die IEEE-Arithmetik `x op s` ist bit-identisch zwischen Rust-`f64` und JS-`number`** — für
   `+`/`-`/`*`/`/` fordert IEEE 754 §5.4.1 (und ECMA-262) das **korrekt gerundete** Ergebnis, es gibt
   genau EIN zulässiges Bitmuster pro Operandenpaar, unabhängig von der Implementierung, ohne FMA.
   Das ist **exakt** die Eigenschaft, die die eingefrorenen binären Kernel `nt_add_strided`/
   `nt_sub_strided`/`nt_mul_strided`/`nt_div_strided` (Kern 07) schon committed bit-identisch zur
   naiven JS-Referenz (`(x,y)=>x+y` usw.) belegen. Skalar `x op s` ist der Spezialfall `y := s`
   (konstant) desselben binären `x op y` — dieselbe IEEE-Operation, dasselbe Ergebnis-Bitmuster.
2. **Der strided Iterationskern `unary_strided` ist bereits bewiesen** (S0/sqrt: `unary_strided` ist
   das unäre Gegenstück zu `binary_strided`, shape-erhaltend, kein Broadcast — strukturell einfacher).
   Die Skalar-Kernel binden nur eine andere Closure (`|x| x op s`) an denselben Kern.

Bit-Parität Skalar = (Iterations-Korrektheit von `unary_strided`, S0-bewiesen) × (IEEE-Arithmetik-
Bit-Parität von `x op s`, Kern-07-bewiesen). **Baustein 0 verifiziert dieses Korollar-Argument** (sind
die binären Kernel wirklich als bit-identisch committed? ist die Arithmetik wirklich identisch?), und
der committete Differentialtest (D6) bestätigt es am tatsächlichen Kernel erneut — eine 30k-Vorab-
Probe wie bei sqrt ist NICHT nötig (kein neuer empirischer Claim). Für IEEE-Spezialwerte gilt
dieselbe Disziplin wie bei den binären Kernel: finite/`±0`/`±Inf` byte-exakt, NaN nur als Wert-Klasse
(`Object.is`/`assertDataBitIdentical` — arithmetische NaN-Payloads sind implementierungsdefiniert,
M1-Vorbehalt).

## Berührte Covenant-Invarianten

- **M1 (Anker `crates/core/src/`, `spike/src/runtime.ts` — COVENANT.md:11-17): BINDET JETZT.** Der
  v5-Zusatz „M1 bindet in dem Moment, in dem ein Kernel für die Op entsteht" greift: diese Scheibe
  ERZEUGT die vier Skalar-Kernel, also wird Bit-Identität zu `scalarElementwiseRuntime` (runtime.ts:
  816) von einer FOLLOWUPS-Paritätslücke zur harten Covenant-Pflicht. Die Kernel müssen bit-identisch
  zur naiven Referenz sein, auch für IEEE-Spezialwerte, unter dem Bit-Identity-Law (Vektorisierung nur
  QUER zu Output-Elementen — bei einer elementweisen Op trivial erfüllt; kein FMA, kein relaxed-simd).
  Siehe „tragende Beobachtung" oben: Bit-Parität ist hier ein Korollar der bereits eingefrorenen
  binären Kernel. Die W2-Paritätslücke für die Skalar-Overloads (FOLLOWUPS:73) schließt für die vier
  add/sub/mul/div-Ops mit dieser Scheibe (`mean` bleibt offen → S2).
- **M4 (Anker `abi.rs`/`matmul_blocked.rs`/`shape.rs`): berührt (abi.rs), append-only gewahrt.** Der
  `check:freeze`-Artefakt-Hash ÄNDERT sich legitim (vier neue Kernel im Binary — Kern-07/11/S0-
  Präzedenz: eine Scheibe, die echte Exports/Kernel HINZUFÜGT, ändert den Hash). Der Freeze-Beweis
  dekomponiert (D7): (1) Pre-Edit-Clean-Rebuild reproduziert den aktuellen Pin `24a048c7…` (S0/sqrt)
  exakt; (2) der neue Code ist rein ADDITIV — neues File `kernels/scalar.rs`, vier `nt_scalar_*_
  strided` strikt ans ENDE von `abi.rs` angehängt, `pub mod scalar;` ans Ende von `kernels/mod.rs`,
  ZERO Edits an bestehenden Funktionskörpern/Zeilenpositionen in `abi.rs`/`matmul_blocked.rs`/
  `shape.rs`; (3) behaviorale Pins grün; (4) Mutations-Beweis; (5) neuer Hash wird der Pin.
  `matmul_blocked.rs`/`shape.rs` byte-unberührt.
  - **Ein-Token-Edit an `kernels/sqrt.rs` (offengelegt):** Damit `scalar.rs` den generischen
    `unary_strided`-Kern wiederverwenden kann, wird dessen Sichtbarkeit von `fn` auf `pub(crate) fn`
    erweitert (D2). Das ist eine rein ADDITIVE Sichtbarkeits-Erweiterung (gewährt Zugriff, entfernt
    nichts) und ändert **weder das Codegen noch das Verhalten von `sqrt_strided`** (Sichtbarkeit ist
    in Rust eine reine Namensauflösungs-/Privacy-Eigenschaft; die Monomorphisierung von
    `unary_strided` an der `|x| x.sqrt()`-Aufrufstelle ist identisch). `sqrt.rs` ist ein NEUES
    S0-File, **nicht** eines der drei frozen-append-only-Files (`abi.rs`/`matmul_blocked.rs`/
    `shape.rs`) und trägt kein `#[track_caller]` — die Append-only-Disziplin gilt für es nicht; der
    Freeze-Beweis ist ohnehin der Hash-Dekompositions-Beweis, nicht ein Byte-Diff von `sqrt.rs`.
    Belegt wird die Unversehrtheit von `sqrt` dadurch, dass ALLE sqrt-Cargo-Tests und der committete
    sqrt-JS-Differentialtest grün bleiben. **Baustein 0 stresst diesen Punkt gezielt.**
- **M5 (Anker `spike/src/ndarray.ts`, `spike/src/wasm/threaded.ts`): UNBERÜHRT.** Kein Rust-Import in
  `ndarray.ts`, kein eager `node:*`-Import; die Kernel sind WASM, `WNDArray`-Skalar-Ops reine
  residente-Memory-Arbeit, der Default-`NDArray`-Pfad bleibt unangefasst. `threaded.ts` wird in dieser
  Scheibe NICHT editiert — der S0/D10-`Omit`→direkter-Cast-Fix steht schon; `ThreadedCoreExports
  extends CoreExports` erbt die vier neuen Deklarationen automatisch, ohne dass eine `threaded.ts`-
  Zeile fällt. Threaded bleibt Node-only-Opt-in hinter `backend("threaded")`.
- **Z1 (Anker `package.json`): unberührt.** Die Kernel sind from-scratch Rust; keine neue
  Abhängigkeit, kein `dependencies`-Feld.
- **Z2 (Anker `package.json`): unberührt.** Alle neuen Tests hängen an BESTEHENDE Korpus-Dateien an
  (`elementwise.test.ts`/`special-values.test.ts` in `test:resident`, `threaded.test.ts` in
  `test:threaded`, `ndarray.test-d.ts` im Root-Typkorpus, Rust-Tests im neuen Kernel-File in
  `cargo test`) — **keine neue Datei, keine Explizitlisten-Änderung in package.json,
  `test-scripts-guard` bleibt grün, KEIN Order-Noise** (Datei-Set unverändert).
- **M2/M3: unberührt.** Die Skalar-Overloads erzeugen KEINE neue Fehler-Fläche (ein `number`-Skalar
  ist immer valide — kein `Guard`, keine Degradationskante). Die vorhandene Shape-Fehler-Maschinerie
  der Array-Array-Form bleibt unverändert; die D2-v3-Overload-Reihenfolge (Skalar zuerst,
  Guard-Träger zuletzt) ERHÄLT die bestehenden WNDArray-Shape-Fehler-Pins (ndarray.test-d.ts UW1–UW4)
  — genau die W2-F1-Lektion, hier proaktiv eingebaut.

---

## Design-Entscheidungen

- **D1 — Scope.** Genau ein neues Kernel-File plus seine Verdrahtung und die vier
  WNDArray-Overload-Erweiterungen: (a) `crates/core/src/kernels/scalar.rs` (NEU) mit vier `pub`
  Funktionen `scalar_{add,sub,mul,div}_strided`, jede ein Einzeiler über den S0-`unary_strided`; (b)
  in `spike/src/wasm/resident.ts`: die vier Bestandsmethoden add/sub/mul/div bekommen je einen
  Skalar-Overload, plus ein neuer privater Marshalling-Helfer `scalarOp`. Verdrahtung: vier
  `nt_scalar_*_strided`-ABI-Einträge (abi.rs), vier `CoreExports`-Deklarationen (neuer vierter
  Merge-Block, loader.ts), `pub mod scalar;` (kernels/mod.rs), `unary_strided` → `pub(crate)`
  (sqrt.rs, M4-Notiz oben), plus **vier `notImplemented(...)`-Stub-Einträge im hand-getippten
  `CoreExports`-Mock in `spike/tests-runtime/backend-oom.test.ts:94-118`** — das EINZIGE strukturell
  getippte `CoreExports`-Literal im Repo; ohne die Stubs bricht `pnpm check`/`check:diag` mit TS2739
  (S0 tat dasselbe für `nt_sqrt_strided`, Z. 117 — dort undokumentiert; Baustein-0-Befund F2,
  empirisch reproduziert). Threaded-Parität ist AUTOMATISCH (dasselbe Crate → die Kernel sind in
  beiden Artefakten; die WNDArray-Skalar-Ops laufen auf jedem Core) und wird NUR getestet, nicht eigens
  verdrahtet. Kein Byte an `NDArray`/`scalarElementwiseRuntime`/`vector.ts`/`reduce.ts`/`dim.ts`/
  `elementwise.rs`/`matmul_blocked.rs`/`shape.rs`. Keine andere Op.

- **D2 — Rust-Kernel (`kernels/scalar.rs`, neues File — freeze-sauber via Wiederverwendung).** Die
  vier Skalar-Kernel binden je eine Skalar-erfassende Closure an den in S0 gebauten generischen
  `unary_strided<F: Fn(f64) -> f64>` (kernels/sqrt.rs) — **kein neuer Iterationskern, keine
  Duplizierung** (M1-strukturelle Identität: es gibt genau EINEN unären strided Kern, bereits
  bewiesen):
  ```rust
  use crate::kernels::sqrt::unary_strided; // pub(crate) — additive Sichtbarkeit, M4-Notiz
  pub fn scalar_add_strided(shape, strides, offset, data, s: f64) -> KResult<(Vec<u32>, Vec<f64>)> {
      unary_strided(shape, strides, offset, data, |x| x + s)
  }
  // sub: |x| x - s   ·   mul: |x| x * s   ·   div: |x| x / s
  ```
  **Operandenordnung gepinnt:** Sub ist `data[i] − s` (NICHT `s − data[i]`), Div ist `data[i] / s`
  (NICHT `s / data[i]`) — exakt `scalarElementwiseRuntime`s eigene Closures (runtime.ts:816-833). Kein
  Guard am Skalar (jeder endliche/nicht-endliche `f64` ist valide; `x/0 → ±Inf`, `0/0 → NaN`,
  signed-zero/Inf propagieren nach IEEE — keine Sonderbehandlung, Test pinnt das). `unary_strided`s
  contiguous fast path / strided general path werden 1:1 mitgenutzt (in S0 bewiesen).
  - **Cargo-Tests** im File-eigenen `#[cfg(test)] mod tests` (nach dem sqrt.rs-Muster): (i) die
    geteilte strided-Maschinerie über EINE Op (z. B. add) — same-shape, transponierte View, offset
    window, size-0-Array, rank-too-large, contiguous-fast-path-vs-general-path-Äquivalenz auf einer
    transponierten View (inkl. Spezialwert-Inputs); (ii) die op-spezifische Arithmetik je Op —
    `add(3,5)==8`, `sub(5,3)==2` UND Ordnung (`sub(3,5)==-2`, beweist `data−s`), `mul(4,2.5)==10`,
    `div(6,2)==3` UND `div(1,0)==+Inf`, `div(-1,0)==-Inf`, `div(0,0).is_nan()`; (iii) ein
    Spezialwert-Kanten-Test je Op (NaN-Skalar propagiert, `x + (-0.0)` bit-korrekt).

- **D3 — ABI-Einträge (vier `nt_scalar_*_strided`, strikt ans ENDE von abi.rs angehängt — M4
  append-only).** Signatur = die 8-Parameter-`nt_sqrt_strided`-Form PLUS ein `scalar: f64`-Parameter,
  eingeschoben zwischen dem Input-Operanden-Block (`data_ptr`, `data_len`) und dem Output-Block
  (`out_data_ptr`, `out_len`):
  ```rust
  pub extern "C" fn nt_scalar_add_strided(
      shape_ptr: u32, rank: u32, strides_ptr: u32, offset: u32,
      data_ptr: u32, data_len: u32, scalar: f64,
      out_data_ptr: u32, out_len: u32,
  ) -> u32 { ... }   // sub/mul/div analog
  ```
  **f64-ABI-Parameter-Präzedenz:** `nt_fill(out_data_ptr, out_len, value: f64)` (abi.rs:698) übergibt
  bereits einen `f64`-Skalar direkt durch die ABI — `f64` ist ein nativer wasm-Typ, kein Pointer
  nötig. Jeder Body ist STRUKTURELL identisch zu `nt_sqrt_strided` (abi.rs:1361): `validate_rank` →
  `validate_region` ×4 (shape, strides, data, out) → `read_slice` ×3 (shape, strides, data) →
  `kernels::scalar::scalar_add_strided(shape, strides, offset, data, scalar)` → bei `Ok`: Output-
  Längen-Check + `read_slice_mut` + `copy_from_slice` + `STATUS_OK`, bei `Err`: `status_of`. Kein
  Zeilenshift an bestehendem abi.rs-Inhalt.

- **D4 — `CoreExports` (loader.ts).** Die vier `nt_scalar_*_strided` werden als **NEUER, vierter
  `export interface CoreExports {}`-Merge-Block am echten Dateiende** deklariert (nach dem S0/sqrt-
  Block Z. 270-283 — TS-Interface-Merging; keine bestehende Zeile angefasst, exakt die Konvention der
  Kern-07- und S0-Blöcke). Neun-Parameter-Signatur je Member, `scalar: number` zwischen `dataLen` und
  `outDataPtr`. `ThreadedCoreExports extends CoreExports` (threaded.ts:165) erbt die vier
  Deklarationen automatisch; der Loader spreadet die echten Instance-Exports (threaded.ts:~421, Kommentar
  ab Z. 413 — Baustein-0-Korrektur F4; seit S0/D10 direkter Cast statt `Omit` → **+0 Instantiations pro
  neuem Member** auf bench:editor/stress, bei n=4 re-validiert).

- **D5 — WNDArray-Skalar-Overloads + `scalarOp`-Helfer (resident.ts, erzwungene Overload-Edit-
  Ausnahme + Klassenkörper-Insertion).**
  - **Overload-Umbau der vier Bestandsmethoden (D6-v2-Muster von W2, erzwungen durch TS2394):** TS
    verbietet Overload-Signaturen vor einer body-tragenden Deklaration — die vier bestehenden
    Signaturzeilen MÜSSEN editiert werden. Form je Methode (Reihenfolge **D2-v3, W2-F1-Lektion:
    Skalar-Overload ZUERST, generischer Guard-Träger ZULETZT**, damit die Broadcast-Shape-Message des
    häufigsten Fehlerfalls nicht hinter dem `number`-Decoy verschwindet):
    ```ts
    add(s: number): WNDArray<S>;                                                         // Skalar zuerst
    add<B extends Shape>(other: Guard<Broadcast<S, B>, WNDArray<B>>): WNDArray<OkShape<Broadcast<S, B>>>; // Guard-Träger zuletzt (bodylos)
    add<B extends Shape>(
      other: number | Guard<Broadcast<S, B>, WNDArray<B>>,
    ): WNDArray<OkShape<Broadcast<S, B>>> | WNDArray<S> {                                // Implementierungssignatur
      if (typeof other === "number") return this.scalarOp("add", other);
      /* ORIGINALER Array-Array-Body BYTE-IDENTISCH hierher verschoben */
    }
    ```
    Die bestehende generische Signaturzeile wird bodylos (`{` → `;`, sonst Zeileninhalt unverändert),
    die neue Skalar-Overload-Signatur davor ergänzt, eine union-typisierte Implementierungssignatur
    eingefügt, deren Body die ORIGINALE Array-Array-Logik (assertLive/assertSameCore/
    runtimeBroadcastShape/scratch-`finally`/`nt_*_strided`-Call/`WNDArray.fresh`) BYTE-IDENTISCH in den
    else-Zweig verschiebt. Verify-A prüft die Byte-Erhaltung explizit am `git diff` (die verschobenen
    Zeilen erscheinen als reine Kontextzeilen). Analog sub/mul/div.
  - **Privater `scalarOp`-Helfer (reine Klassenkörper-Insertion, kein Edit an Bestandsmembern):**
    marshalt EINMAL (statt 4× dupliziert) und wählt den Kernel-Einstiegspunkt per 4-Wege-`switch` auf
    den Op-Schlüssel — spiegelt `scalarElementwiseRuntime`s eigene Dispatcher-Form. Mechanik gespiegelt
    von `sqrt()` (resident.ts:1219), erweitert um den Skalar-Parameter:
    ```ts
    private scalarOp(op: "add" | "sub" | "mul" | "div", s: number): WNDArray<S> {
      this.assertLive(op);
      const outLen = product(this.shape);
      const scratch: ScratchBuf[] = [];
      try {
        const shapeBuf = writeU32Array(this.core, this.shape);   scratch.push(shapeBuf);
        const stridesBuf = writeU32Array(this.core, this.strides); scratch.push(stridesBuf);
        const outDataBuf = allocBytes(this.core, outLen * 8);
        const kernel =
          op === "add" ? this.core.nt_scalar_add_strided
          : op === "sub" ? this.core.nt_scalar_sub_strided
          : op === "mul" ? this.core.nt_scalar_mul_strided
          : this.core.nt_scalar_div_strided;
        const status = kernel(
          shapeBuf.ptr, this.shape.length, stridesBuf.ptr, this.offset,
          this.buf.ptr, this.buf.lenElems, s, outDataBuf.ptr, outLen,
        );
        if (status !== 0) {
          freeBuf(this.core, outDataBuf);
          throw new Error(`wasm resident nt_scalar_${op}_strided: status ${status} for shape [${this.shape.join(",")}]`);
        }
        return WNDArray.fresh<S>(this.core, [...this.shape] as unknown as S, outDataBuf.ptr, outLen);
      } finally {
        for (const buf of scratch) freeBuf(this.core, buf);
      }
    }
    ```
    Die vier WASM-Exports sind kontextfreie Funktionen (kein JS-`this`) — die Zuweisung an `kernel`
    ohne Bindung ist korrekt. Rückgabetyp `WNDArray<S>` (shape-erhaltend, `outShape === this.shape`,
    auch bei Rang 0). **Disziplin:** resident.ts ist ein TS-Klassenkörper mit privatem Konstruktor →
    die Overload-Edits sind die erzwungene Ausnahme (byte-erhaltend), der `scalarOp`-Helfer ist reine
    Insertion; kein sonstiger Bestandsmember angefasst.

- **D6 — M1-Differentialtest + threaded-Parität + Typ-Pins + Mutant.**
  - **Residenter M1-Test (an `spike/tests-runtime/elementwise.test.ts` angehängt — bestehende Datei,
    keine Listenänderung):** je Op × (i) contiguous, (ii) transponierte View, (iii) geslicte View,
    (iv) Offset-Fenster, (v) rank-0, (vi) size-0-Dim: `wnd.op(s)` gegen `scalarElementwiseRuntime(op,
    wnd.toArray(), s)` (Referenz über die logisch-geordneten View-Daten, `.toArray()` VOR der Op
    gewonnen — dieselbe Technik wie sqrt/die strided-View-Tests). Bit-Vergleich
    `assertDataBitIdentical` (Object.is — finite/`±0`/`±Inf` exakt, NaN als Klasse). **Pflicht:**
    mindestens ein VIEW-Fall je Op (beweist den strided general path). **Zusätzlicher Äquivalenz-
    Differential (Rang ≥ 1):** `wnd.op(s)` byte-identisch zu `wnd.op(WNDArray.fromArray([1] as const,
    [s]))` (der bestehende Broadcast-Pfad) — beweist Semantik-Äquivalenz dort, wo beide Wege existieren
    (Rang-0-Divergenz `[]` bleibt `[]` bewusst NICHT über diesen Pfad geprüft, da `[1]`-Broadcast dort
    `[]→[1]` machen würde — genau die Motivation der shape-erhaltenden Semantik).
  - **Spezialwerte (an `special-values.test.ts` angehängt):** ein randomisierter `genDataSpecial`-
    Raster-Block nach der Datei-eigenen Konvention (Header „injects SPECIAL_VALUES into every op that
    has a resident kernel"; strukturell wie der sqrt-Block aus S0/C-2): ca. 60 Fälle je Op oder
    kombiniert, `genShape(rng, 0, 4)` (Rang 0–4) + `genDataSpecial` fürs Array UND ein Spezialwert-
    Skalar (NaN/`±Inf`/`±0`/Subnormal via `nextF64Special`), `wnd.op(s)` gegen `scalarElementwise
    Runtime` bit-identisch (NaN als Klasse). **KEIN** byte-exakter-NaN-Payload-Fixture (Skalar-Ops
    sind Arithmetik, M1s Payload-Vorbehalt).
    **v2-Ergänzung (Baustein-0-Befund F3):** Der Zufalls-Raster deckt einen Skalar von genau `0`/`-0`
    nur mit ca. 3,5 % pro Fall ab (`specialProb`-Default × 1/|SPECIAL_VALUES|) → über 60 Fälle ca.
    12 % Chance, dass der TS-Differential-Layer `x.div(0)` NIE trifft. Die Rust-cargo-Tests (D2) pinnen
    `div`-durch-0/0-0 explizit, also ist die Kernel-Korrektheit nicht in Gefahr — aber der TS-Layer
    braucht Belt-and-Suspenders. Pflicht deshalb: **kuratierte WNDArray-Fixtures** `x.div(0)` (→ `±Inf`
    je Vorzeichen), `x.div(-0)`, und ein `sub`-Ordnungs-Fixture (`x.sub(s)` = `data−s`, NICHT `s−data`)
    — jeweils WNDArray-vs-`scalarElementwiseRuntime` bit-identisch, nach dem Muster der bestehenden
    kuratierten add/mul-Fixtures in `special-values.test.ts` (~Z. 451-540).
  - **Threaded-Parität (an `spike/tests-runtime/threaded.test.ts` angehängt):** je Op auf dem THREADED
    Core bit-identisch zum STABLE Core (das etablierte threaded-vs-stable-Differential um die vier
    Skalar-Ops erweitern; deckt die automatische Parität ab). Mindestens ein View-Fall und — S0/C-2-
    Lektion — mindestens ein `genDataSpecial`-Fall je Op (contiguous + View), damit die „bit-identisch
    inkl. IEEE-Spezialwerte"-Aussage direkt auf dem threaded Artefakt belegt ist.
  - **Typ-Pins (an `spike/tests/ndarray.test-d.ts` angehängt):** WNDArray-Skalar-Overload-Gruppe,
    spiegelt die NDArray-W2-Pins als zweite Call-Site derselben bewiesenen Form: `wnd.div(2)` →
    exakt `WNDArray<[2,3]>`-Erhalt (`Expect<Equal<…>>`), Rang-0 (`WNDArray<[]>`), wide/dynamischer Rang
    (Shape-Erhalt), Readonly-S-Erhalt; `wnd.div(nd)` bleibt generisch/unbeeinflusst (Array-Form-Nicht-
    Interferenz); Union-über-Overload-Grenze (`number | WNDArray<B>`) als `@ts-expect-error`
    (dokumentierte TS2769-Kante, Präzedenz wie W2/`backend()`). **M3-Erhalt:** die bestehenden
    WNDArray-Shape-Fehler-Pins UW1–UW4 (ndarray.test-d.ts:328-366) bleiben unverändert grün — die
    D2-v3-Reihenfolge (Guard-Träger zuletzt) ist genau dafür da. **Ein realer-tsc-Diagnose-
    Qualitätstest für WNDArray ist VERPFLICHTEND (Baustein-0-Befund F1, empirisch bewiesen):** Der
    Typ-Ebenen-Pin UW4 (ndarray.test-d.ts:349-351) prüft `Guard<Broadcast<…>>` als eigenständigen
    Typ-Alias, NIE über einen tatsächlichen `.add()`-Aufruf — er bleibt darum auch bei VERTAUSCHTER
    Overload-Reihenfolge (der W2-F1-Regression) grün, und die volle `check:diag` läuft mit der kaputten
    Reihenfolge fehlerfrei durch (Baustein 0 hat beides live reproduziert: korrekte Reihenfolge →
    `cannot broadcast shapes …`-Message überlebt; vertauscht → kollabiert zu `… not assignable to type
    'number'`; Root-`check:diag` exit 0 in beiden Fällen). UW1–UW4 bieten also NULL Regressionsschutz
    gegen genau die Fehlerklasse, die die D2-v3-Reihenfolge verhindern soll. Pflicht ist deshalb: (1)
    ein „diagnostic quality"-Test (an `elementwise.test.ts` oder `special-values.test.ts` angehängt),
    der eine Wegwerf-Fixture AUSSERHALB des Repos mit dem echten `tsc` kompiliert
    (`spawnSync`-Muster von `scalar-mean.test.ts`s F1-Pin, Z. ~509-556) und assertiert, dass die
    Broadcast-Shape-Message-CONTENT durch `WNDArray.add`s Overload-Set am Argument erscheint;
    Nicht-Vakuität per Reihenfolgen-Flip bewiesen; (2) ein neuer `@ts-expect-error`-Typ-Pin in
    ndarray.test-d.ts, der die überladene Skalar-Methode tatsächlich mit einem shape-inkompatiblen
    WNDArray-Argument AUFRUFT (etwas, das UW1–UW4 nie tun). Falls S0 bereits eine WNDArray-tsc-Fixture
    trägt, wird sie um den Broadcast-Fall erweitert statt dupliziert.
  - **Pflicht-Mutant (T5, während Baustein A):** eine gezielte Kernel-Mutation, die den M1-
    Differentialtest kippt (Kandidat: `|x| x + s` → `|x| x - s` im add-Kernel, oder `|x| x - s` →
    `|x| s - x` im sub-Kernel — beweist die Operandenordnung, oder die fast-path-Bedingung invertiert).
    Nachweislich gefangen (benannte Assertion), Revert per **Backup-Kopie** (NIE `git checkout` auf
    uncommittete Arbeit — harte Arbeitsregel 1), `git status`-Beweis.

- **D7 — Pins, Budget, Freeze.**
  - **`check:freeze`-Hash RE-PINS.** Freeze-Beweis vollständig (M4 oben): Pre-Edit-Clean-Rebuild ==
    `24a048c7…`, additive-only-Diff (neues scalar.rs + vier abi.rs-Anhänge + kernels/mod.rs-Anhang +
    `pub(crate)` an sqrt.rs — Letzteres verhaltens-/codegen-neutral), behaviorale Pins grün (inkl.
    aller sqrt-Tests, die die `unary_strided`-Wiederverwendung unversehrt belegen), Mutations-Beweis,
    neuer Hash = Pin (plattform-gelabelt, `scripts/check-freeze-hash.mjs`). BEIDE Artefakte re-pinnen
    (stable + threads); `test:threaded` beweist ihre Bit-Identität zueinander weiter.
  - **`check:diag`:** Beitrags-Zerlegung: (1) vier neue `CoreExports`-Member — auf bench:editor/stress
    **+0/Member** (S0/D10-Omit-Fix), auf check:diag Root kleine echte Typkosten; (2) die vier
    WNDArray-Overload-Edits = **Klassen-Surface-Wachstum** (vier neue Overload-Signaturen + der
    `scalarOp`-Member) → dritter Mechanismus (Klassen-Surface-Ripple, generische Member rippeln auch
    in Korpora mit unberührtem File-Set); zweite Call-Site der bereits in W2 bewiesenen Skalar-
    Overload-Typ-Form, also KEINE neue Maschinerie; (3) Test-Anhänge (elementwise/special-values/
    threaded/ndarray.test-d) tragen reale Typkosten — aber **KEIN Order-Noise** (keine neue Datei,
    Datei-Set unverändert). Absolut-Gate: **Haupt-check:diag-Wachstum ≤ +6.000** (W2-Referenz für
    NDArray-Skalar-Overloads war +5.762 gross / +4.233 netto; S1 ist NDArray-analog plus vier
    CoreExports-Member, aber ohne neue Datei/Order-Noise und ohne `mean`). **Baustein-0-Messbefund
    (empirisch):** die vier `CoreExports`-Member + backend-oom-Stubs ALLEIN kosten check:diag Root
    **Δ0** (206.850) und stress **Δ0** (106.239), alle 8 bench:editor-Pins exakt — der S0/D10-Omit-Fix
    skaliert nachweislich über n=1 hinaus (bei n=4 re-validiert). Das messbare Root-Wachstum kommt
    fast ausschließlich aus dem WNDArray-Overload-Umbau (Baustein 0 maß +205 für EINE Methode → grobe
    Erwartung ca. +820 für vier + `scalarOp` + Testinhalt + Typ-Pins, klar unter dem +6.000-Gate).
    **Mess-Gültigkeit (harte Arbeitsregel 6, Baustein-0-Befund F2):** `check:diag` ist erst NACH dem
    `backend-oom.test.ts`-Stub-Fix (D1) gültig — ein ungefixter Korpus scheitert mit TS2739, druckt
    aber weiterhin eine plausibel aussehende Instantiations-Zeile (Baustein 0 sah `206850` NEBEN dem
    Fehler). Jede Messung prüft Exit-Code UND Fehleranzahl mit, nie nur die Kennzahl greppen. Gestufte
    Messung: Baseline im frischen Worktree, dann Δ nach (①) Rust+ABI+loader+resident.ts-Quelle (inkl.
    backend-oom-Stubs), (②) Test-Anhänge, (③) Typ-Pins. **stress/browser:** Ripple erwarten und deterministisch attribuieren (stress importiert
    `spike/src` direkt → WNDArray-Klassen-Surface-Ripple wie W2s +1.181; browser kompiliert
    threaded.ts nicht, aber ggf. resident.ts — messen; W2/S0-Erfahrung: browser blieb Δ0). Deltas
    doppelt messen (Determinismus). `bench:editor` (8 Pins): messen; falls die WNDArray-Klassen-
    Surface die Workloads bewegt, neu setzen (Doppelmessung, Ergebnisse-Doc dekomponiert). Erwartung
    aus dem Omit-Fix: die vier CoreExports-Member allein bewegen bench:editor NICHT (+0); eine
    Bewegung käme allein aus dem WNDArray-Klassen-Surface-Ripple, falls Workloads WNDArray
    instanziieren.
  - **Test-/cargo-Zahlen:** `test:resident`, `test:threaded`, `cargo` steigen um die neuen Fälle;
    exakte Deltas ins Ergebnisse-Doc.
  - Gestufte Messung: Baseline im frischen Worktree, immer Exit-Code + Fehlerausgabe prüfen.

- **D8 — Doc-Platzierung.** Ergebnisse-Doc `docs/wasm-parity-scalar-ergebnisse.md` (volles Narrativ +
  Post-Verification-Addendum). FOLLOWUPS: die Kampagnen-Zeile (S1 ERLEDIGT), das W2-Paritätsitem (die
  Skalar-Overload-Lücke für add/sub/mul/div schließt; `mean` bleibt → S2). `docs/roadmap.md`
  WASM-Parität-Sektion: S1 erledigt. CLAUDE.md „Status" + „Aktuelle Pins & Gates": neuer Artefakt-Hash,
  test:resident/threaded/cargo-Zahlen, check:diag/bench:editor-Pins, S1 erledigt (nur Einzeiler +
  IST-Zahlen). Vollnarrativ an `docs/projekt-log.md` anhängen (Hausregel).

- **D9 — Sprache.** Code/Kommentare/Tests/Commit-Message: Englisch (Hard Constraint). Spec +
  Ergebnisse-Doc: Deutsch. „ca." statt Tilde, keine Strikethroughs (GFM-Gate, harte Arbeitsregel 4).

## Akzeptanzkriterien

- **T1:** `kernels/scalar.rs` liefert `scalar_{add,sub,mul,div}_strided`, jede über `unary_strided`
  (`|x| x op s`, Operandenordnung `data−s`/`data/s`); alle File-eigenen cargo-Tests grün, inkl. der
  geteilten strided-Maschinerie (fast-vs-general auf View) und der op-spezifischen Arithmetik-Kanten
  (`div` durch 0/0-0, `sub`-Ordnung).
- **T2:** vier `nt_scalar_*_strided` ans Ende von abi.rs angehängt (9-Parameter, `scalar: f64`),
  `CoreExports` um vier Member erweitert (neuer vierter Merge-Block), `pub mod scalar;` angehängt,
  `unary_strided` `pub(crate)`, vier `notImplemented(...)`-Stubs im `backend-oom.test.ts`-Mock (F2)
  — kein Zeilenshift an bestehendem abi.rs/matmul_blocked.rs/shape.rs-Inhalt.
- **T3:** vier WNDArray-Skalar-Overloads (Reihenfolge Skalar-zuerst/Guard-Träger-zuletzt), der
  Array-Array-Body je Methode byte-identisch in den else-Zweig verschoben (am Diff bewiesen); privater
  `scalarOp`-Helfer als Insertion; keine bestehende Member editiert außer den vier erzwungenen
  Overload-Umbauten.
- **T4 (M1):** Residenter Differentialtest deckt je Op contiguous + mindestens eine View +
  rank-0/size-0 + Spezialwert-Raster + kuratierte `div(0)`/`div(-0)`/`sub`-Ordnungs-Fixtures (F3) ab,
  bit-identisch zu `scalarElementwiseRuntime` (NaN als Klasse); `[1]`-Broadcast-Äquivalenz (Rang ≥ 1)
  byte-identisch; threaded-vs-stable-Parität je Op grün, inkl. Spezialwerte.
- **T4b (M3-Diagnose, F1):** ein realer-tsc-Diagnose-Qualitätstest beweist, dass die Broadcast-Shape-
  Message durch `WNDArray.add`s Overload-Set am Argument erscheint (Nicht-Vakuität per Reihenfolgen-
  Flip), plus ein `@ts-expect-error`-Typ-Pin, der die überladene Skalar-Methode mit shape-inkompatiblem
  WNDArray-Argument tatsächlich AUFRUFT.
- **T5 (Mutant):** mindestens ein Kernel-Mutant nachweislich vom M1-Test gefangen, danach revertiert
  (Backup-Kopie, `git status`-Beweis).
- **T6 (Freeze):** `check:freeze` re-gepinnt mit vollständigem Dekompositions-Beweis (Pre-Edit-Hash
  `24a048c7…` reproduziert, additive-only inkl. der verhaltens-neutralen `pub(crate)`-Erweiterung,
  neuer Hash); beide Artefakte; alle sqrt-Tests bleiben grün (Wiederverwendung unversehrt).
- **T7 (Gates/Pins):** Gate-Block grün; `check:diag`-Δ dekomponiert (CoreExports-Member / WNDArray-
  Klassen-Surface / Test-Inhalt), Absolut-Gate ≤ +6.000 eingehalten; stress/browser-Deltas
  deterministisch attribuiert; bench:editor-Pins gemessen und (falls bewegt) neu gesetzt.
- **T8 (Docs):** Doc-Platzierung (D8) vollständig.

## Nicht-Ziele

Kein Threaded-Pool-Kernel für die Skalar-Ops (der Pool bleibt matmul-only; die Ops laufen auf dem
residenten Core in beiden Artefakten), keine Änderung an `NDArray.add/sub/mul/div`/
`scalarElementwiseRuntime`/`NDArrayView`, keine andere Paritäts-Op (mean/item/stack/argmax/topk sind
S2–S5), kein `rsub`/`rdiv`/`.scale()`-Alias, keine Änderung an der öffentlichen Signatur/Semantik der
Skalar-Overloads auf der NDArray-Seite, keine Behauptung eines gemessenen Nutzerbedarfs, kein
Browser-Port des Threads-Pfads.

## Gate-Block / Definition of Done

`pnpm check` (Dreier-Verbund) · `check:diag`(+stress/browser, Pin-Protokoll D7) · `pnpm test:core` ·
`pnpm test:resident` (inkl. neuer Skalar-M1-Tests) · `pnpm test:threaded` (baut beide Artefakte,
Skalar-Parität) · `cargo test --manifest-path crates/core/Cargo.toml` (inkl. neuer Skalar-Kernel-
Tests) · `pnpm check:freeze` (NEUER Hash-Pin, beide Artefakte) · `pnpm bench:editor` (8 Pins, ggf. neu
gesetzt) · `pnpm test:example` (unberührt) · `graph-a-lama query lint` · GFM-Gate auf allen neuen/
geänderten `.md`.

## Verify-Plan (Stufe 3)

**Baustein 0 (vor dem Bau, gegen DIESE Spec, adversarial — `brainroute:deep`): DURCHGEFÜHRT
2026-07-23, kein Bau-Blocker — Befunde im Addendum unten, v2-Merge oben.** Code-Annahmen der
Spec am echten Code prüfen — insbesondere: (a) ist `unary_strided` wirklich der richtige, bereits
bewiesene Kern, und ist `pub(crate)` codegen-/verhaltens-neutral für `sqrt_strided` (kein Zeilenshift/
keine Monomorphisierungs-Änderung)? (b) sind die binären Kernel (`nt_add_strided` & Co.) wirklich als
bit-identisch zur JS-Referenz committed — trägt das Korollar-Argument, oder braucht S1 doch eine
eigene empirische Vorab-Probe? (c) ist die abi.rs-Einreihung append-only-verträglich und die
9-Parameter-`f64`-Signatur ABI-legal (nt_fill-Präzedenz belastbar)? (d) erbt `ThreadedCoreExports`
die vier Member wirklich ohne threaded.ts-Edit, und hält der S0/D10-Cast das +0/Member? (e) ist die
D2-v3-Overload-Reihenfolge korrekt gegen die bestehenden WNDArray-Shape-Pins (UW1–UW4) — genügt der
Typ-Ebenen-Pin, oder braucht WNDArray einen realen-tsc-Diagnose-Test wie W2? **[GEKLÄRT F1: UW4 genügt
NICHT, Diagnose-Test ist Pflicht.]** (f) ist der `scalarOp`-
Kernel-Select (`this.core.nt_scalar_*_strided` ohne Bindung) korrekt, und sind die Exports wirklich
kontextfrei? (g) Testplan-Lücken (fehlt eine Spezialwert-/Ordnungs-Kante? deckt der threaded-Test die
Parität nicht-vakuär ab?); (h) Covenant-Abgleich (M1 korrekt als bindend, M4-Freeze-Beweis inkl.
`pub(crate)`-Notiz tragfähig, M5 wirklich unberührt?). Befunde mergen, Design-Blocker mit dem Owner in
die Spec einarbeiten (Richtungsänderungen abnehmen lassen), „Adversariale Spec-Verifikation
(Addendum)" hier anhängen, DANN Implementierung.

**Nach der Implementierung:** voller Katalog — **A** (Spec-Konformität pro D, alle Gates frisch,
eigener Mutant, Byte-Erhaltung der vier Body-Verschiebungen am Diff), **B** (adversarial: Grenzfälle
jenseits der Spec — Skalar-Ops auf mehrfach-komponierten Views, ausschließlich-Spezialwert-Buffer,
Spezialwert-Skalar × Spezialwert-Daten, size-0/rank-0, Operandenordnung, breite Kernel-Mutanten;
Diagnose-Qualität der Shape-Message bei Broadcast-Mismatch — die W2-F1-Klasse aktiv angreifen; Mess-/
Freeze-Randbedingungen), **C** (`covenant:covenant-verify`: M1 jetzt bindend korrekt? M4-Hash-Re-Pin
inkl. `pub(crate)`-Additivität sauber? M5/Z1/Z2 unberührt?) — parallel, isoliert (mutierende Verifier
je eigener Worktree + Slice-Patch, read-only C im Haupt-Baum). Aufträge aus
docs/verify-runde-template.md. Ergebnisse-Doc mit Post-Verification-Addendum, dann Commit.

## Adversariale Spec-Verifikation (Addendum, Baustein 0 — 2026-07-23)

Verifier: `brainroute:deep`, adversarial gegen v1 dieser Spec, read-only im Haupt-Baum, empirische
Proben in eigenem `git worktree` (`/private/tmp/.../s1-verify`, node_modules aus dem Haupt-Baum
symlinkt, danach `git worktree remove`) + einer tsc-Fixture außerhalb des Repos. **Der Haupt-Working-
Tree wurde nie angefasst** (`git status` vor/nach clean). **Verdikt: kein Blocker gegen den Bau.**

**Alle Kern-Design-Annahmen HALTEN — mehrere empirisch, nicht nur argumentativ:**
- **`unary_strided` + `pub(crate)` (a):** `unary_strided` ist in sqrt.rs:20 privat (Prämisse korrekt).
  Nach Widening auf `pub(crate)` + `cargo clean` + Full-Rebuild war der Artefakt-Hash **byte-identisch
  `24a048c7…`** (codegen-neutral GEMESSEN, nicht „sollte"), alle 8 sqrt-cargo-Tests grün. sqrt.rs ist
  korrekt KEINE der drei frozen-append-only-Dateien, kein `#[track_caller]`.
- **M1-Korollar (b):** die binären Kernel sind committed bit-identisch zur JS-Referenz
  (special-values.test.ts:104-198, 60 Spezialwert-Fälle je add/sub/mul/div via
  `assertDataBitIdentical`). Als Backstop baute der Verifier den echten `nt_scalar_add_strided` und
  fuhr einen **36.324-Fälle-JS-vs-WASM-Differential (alle SPECIAL_VALUES-Paare + 2.000 Zufalls-f64):
  0 Mismatches.** Keine 30k-Vorab-Probe nötig — Korollar trägt.
- **ABI (c):** `nt_fill(…, value: f64)` (abi.rs:698) ist ein realer f64-Direkt-Param-Präzedenz;
  abi.rs endet genau bei `nt_sqrt_strided` (1361-1399, Datei = 1400 Zeilen), Append-Punkt sauber. Der
  Verifier baute `nt_scalar_add_strided` end-to-end (wasm32-Build grün) + Node-Smoke: Export da,
  `3+5=8` via gebundenem UND `10+(-2.5)=7.5` via UNGEBUNDENEM Call → validiert auch (f).
- **CoreExports/Threaded + +0/Member (d):** `ThreadedCoreExports extends CoreExports` (threaded.ts:165),
  automatische Vererbung bestätigt. **+0/Member bei n=4 re-validiert:** nach dem F2-Fix check:diag
  206.850 (Δ0), stress 106.239 (Δ0), alle 8 bench:editor-Pins exakt.
- **(g)/(h):** View/Offset/rank-0/size-0-Konventionen sound; M4-Freeze-Dekomposition empirisch
  bestätigt; M5 unberührt (S1 macht NULL threaded.ts-Edits, anders als S0/D10); Z1/Z2 sauber; M2s
  Union-über-Grenze-Kaveat repliziert (vorbestehend, W2-v6-Kandidat), nicht neu.

**Zwei MAJOR-Befunde (in v2 eingearbeitet):**
- **F1 — WNDArray-Diagnose-Qualitätstest ist Pflicht (empirisch bewiesen).** v1 hielt ihn für optional
  (UW4 genüge). FALSCH: UW4 (ndarray.test-d.ts:349-351) prüft `Guard<Broadcast<…>>` als Standalone-
  Typ-Alias, nie über einen echten `.add()`-Aufruf. Der Verifier baute D5s Overload auf `add`, flippte
  die Reihenfolge und lief echtes tsc: korrekte Reihenfolge → `cannot broadcast shapes [2,3] and [4]…`
  überlebt; vertauscht → kollabiert zu `… not assignable to type 'number'`; **Root-`check:diag` exit 0
  in BEIDEN Fällen** — UW1–UW4 bieten null Schutz gegen die W2-F1-Regression. D6/T4b jetzt: realer-tsc-
  Diagnose-Test (scalar-mean-F1-Muster) + aufrufender `@ts-expect-error`-Pin, Pflicht.
- **F2 — `backend-oom.test.ts`-Mock-Stubs sind Pflicht-Verdrahtung (empirisch, Arbeitsregel-6-Falle).**
  `backend-oom.test.ts:94-118` ist das EINZIGE hand-getippte `CoreExports`-Literal. Ohne vier
  `notImplemented(...)`-Stubs scheitert `check:diag` mit **TS2739**, druckt aber weiter `206850` neben
  dem Fehler — wer nur die Instantiations-Zeile greppt, meldet Δ0 aus einem NICHT kompilierenden
  Korpus. S0 fügte den Stub für `nt_sqrt_strided` (Z. 117) still hinzu. D1/D7/T2 jetzt explizit.

**Minor/Nit (in v2 berücksichtigt):** F3 — Skalar-`0`-Deckung im Zufalls-Raster nur ca. 3,5 %/Fall →
kuratierte `div(0)`/`div(-0)`/`sub`-Ordnungs-Fixtures Pflicht (D6/T4). F4 — threaded.ts-Cast bei ~Z.
421 (Kommentar ab 413), Zitat korrigiert (D4).

## Änderungslog

- **v2 (2026-07-23):** Baustein 0 (adversarial gegen v1) fand keinen Bau-Blocker und bestätigte alle
  Kern-Design-Annahmen empirisch (Clean-Rebuild-Hash bei `pub(crate)` unverändert, +0/Member bei n=4,
  36k-Differential 0 Mismatches, end-to-end-wasm-Build + ungebundener Export-Call). Zwei MAJOR-Befunde
  eingearbeitet: F1 (WNDArray-Diagnose-Qualitätstest jetzt VERPFLICHTEND — UW4 schützt nicht gegen die
  W2-F1-Regression, live reproduziert), F2 (`backend-oom.test.ts`-Mock-Stubs als Pflicht-Verdrahtung,
  TS2739-Arbeitsregel-6-Falle). F3 (kuratierte `div(0)`/`sub`-Ordnungs-Fixtures) und F4 (threaded.ts-
  Zeilencitat) berücksichtigt. Addendum-Abschnitt dokumentiert alle Befunde + Probe-Ergebnisse. Keine
  Richtungsänderung (alle Befunde sind Testplan-/Verdrahtungs-Verschärfungen im genehmigten Design).
- **v1 (2026-07-23):** Erstfassung nach Owner-Richtungsabnahme (WASM-Parität S1 = Skalar-Overloads,
  nach dem S0/sqrt-Muster). Zentrale Design-Wahl: die vier Skalar-Kernel als Einzeiler über den
  S0-`unary_strided` (Wiederverwendung statt Duplizierung; erfordert dessen `pub(crate)`-Erweiterung,
  in M4 als verhaltens-neutral offengelegt). M1-Bit-Parität als Korollar der eingefrorenen binären
  Kernel begründet (keine 30k-Vorab-Probe nötig, anders als S0). D2-v3-Overload-Reihenfolge (W2-F1-
  Lektion) proaktiv eingebaut.
