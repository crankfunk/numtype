# FOLLOWUPS

Bewusst zurückgestellte Arbeit. Eintragen beim Zurückstellen, austragen wenn es in einem Commit landet.

- [x] Rust/WASM-Kern (from scratch) — v1 implementiert, unabhängig verifiziert & committet als Kern 01 (Commit „Kern 01: from-scratch Rust/WASM kernels…"; docs/kern-01-wasm-core-spec.md, 38 cargo- + 791 Differentialtests bit-identisch, ~1,9× vs. naiv inkl. Kopien)
- [x] Kern v2: Zero-Copy-Residenz — implementiert, unabhängig verifiziert & committet als Kern 02 (Commit „Kern 02: zero-copy residency…"; docs/kern-02-residency-spec.md; WNDArray, 674 Resident-Tests, Leak-Plateau nicht-vakuös bewiesen)
- [x] Resident-API-Grenze: `fromArray`-Overload für `Float64Array` — erledigt 2026-07-10 (beide Klassen symmetrisch erweitert, API-Spiegelung erhalten; Chain-Defizit 0,76–0,91× → 1,00–1,01× eliminiert; ehrlicher Restbefund: memcpy-Kopien waren nie teuer, der Residenz-Vorteil ggü. v1 ist end-to-end ~1,0–1,05×; docs/kern-02-ergebnisse.md Follow-up-Abschnitt)
- [ ] OOM-Härtung beider WASM-Backends (v1 `backend.ts` + `resident.ts`): Shape-Scratch-Allokationen liegen vor dem Output-`nt_alloc` außerhalb von try/finally → Leak, falls dieser fehlschlägt (Verify-Befund Kern 02; präexistentes v1-Muster, nur bei nahezu erschöpftem WASM-Speicher triggerbar)
- [ ] Guard gegen den `test:core`-Explizitlisten-Footgun: neue v1-Testdateien in spike/tests-runtime/ liefen sonst still nie (Meta-Test, der die Liste gegen das Verzeichnis abgleicht, oder Unterverzeichnis-Split)
- [ ] Kern v2: strided Kernels (Transpose/Views ohne Materialisierung)
- [ ] Kern v2+: WASM-SIMD128, Threads (COOP/COEP-Implikationen), Blocking fürs Matmul
- [ ] Transzendente Ops (exp/sin/…): eigene Determinismus-Entscheidung nötig — brechen Bit-Parität zur JS-Referenz (libm-Differenzen); vgl. KB `cross-plattform-float-determinismus-verifizieren`
- [ ] ABI-Härtung: rank/len-Validierung VOR Slice-Konstruktion in `read_slice`/`read_slice_mut` (Verify-Befund Kern 01, defense-in-depth)
- [ ] Differential-Generator: Spezialwerte injizieren (NaN, ±Infinity, ±0, Subnormals) — Bit-Identität bisher nur für normale endliche Werte belegt
- [ ] `reshape`/`flatten` auf Typ-Ebene: bräuchte Produkte großer Literal-Dims — jenseits der Tupel-Arithmetik (~1000er-Grenze). Design-Entscheidung nötig: dynamisch degradieren vs. Spezialfälle
- [ ] Editor-Latenz real messen (VS Code tsserver), nicht nur der tsc-`extendedDiagnostics`-Proxy
- [ ] Alternative Error-Surfacing-Patterns systematisch vergleichen, falls der Spike nur eines implementiert
- [ ] Test-/Qualitäts-Portfolio (Vitest etc.) einrichten, sobald echter Runtime-Code entsteht
- [x] KB-Capture nach Spike-Abschluss: empirische TS-Limits & tragfähige Typ-Patterns in die Wissensbasis — erledigt 2026-07-09 (4 Notizen: wide-type-Wache, two-of-three-Varianz, Argument-Guard, TS7-Empirie)
- [ ] npm-Namensverfügbarkeit `numtype` prüfen/reservieren vor OSS-Release (Scope-Fallback: `@numtype/core`) — Stand 2026-07-09: Name ist frei (Registry „Not found")
- [ ] Varianz-Design für die Bibliotheksphase: `NDArray<Shape>` ist mit sounder Rang-Degradierung + Argument-Guards invariant („two-of-three rule", docs/spike-01-ergebnisse.md Addendum). Spike-Lösung: `AnyNDArray`. Sauberer Kandidat: kovariante Read-only-View (`NDArrayView<out S>`) ohne guard-tragende Methoden
- [ ] `keepdims` im Runtime-`sum()` nachziehen — Typ-Ebene (`ReduceAxisKeepDims`) existiert und ist getestet, Runtime-Parameter fehlt
