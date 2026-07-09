# FOLLOWUPS

Bewusst zurückgestellte Arbeit. Eintragen beim Zurückstellen, austragen wenn es in einem Commit landet.

- [ ] Rust/WASM-Kern (from scratch) — erst nach erfolgreichem Typ-Spike (docs/spike-01-type-layer-spec.md)
- [ ] `reshape`/`flatten` auf Typ-Ebene: bräuchte Produkte großer Literal-Dims — jenseits der Tupel-Arithmetik (~1000er-Grenze). Design-Entscheidung nötig: dynamisch degradieren vs. Spezialfälle
- [ ] Editor-Latenz real messen (VS Code tsserver), nicht nur der tsc-`extendedDiagnostics`-Proxy
- [ ] Alternative Error-Surfacing-Patterns systematisch vergleichen, falls der Spike nur eines implementiert
- [ ] Test-/Qualitäts-Portfolio (Vitest etc.) einrichten, sobald echter Runtime-Code entsteht
- [x] KB-Capture nach Spike-Abschluss: empirische TS-Limits & tragfähige Typ-Patterns in die Wissensbasis — erledigt 2026-07-09 (4 Notizen: wide-type-Wache, two-of-three-Varianz, Argument-Guard, TS7-Empirie)
- [ ] npm-Namensverfügbarkeit `numtype` prüfen/reservieren vor OSS-Release (Scope-Fallback: `@numtype/core`) — Stand 2026-07-09: Name ist frei (Registry „Not found")
- [ ] Varianz-Design für die Bibliotheksphase: `NDArray<Shape>` ist mit sounder Rang-Degradierung + Argument-Guards invariant („two-of-three rule", docs/spike-01-ergebnisse.md Addendum). Spike-Lösung: `AnyNDArray`. Sauberer Kandidat: kovariante Read-only-View (`NDArrayView<out S>`) ohne guard-tragende Methoden
- [ ] `keepdims` im Runtime-`sum()` nachziehen — Typ-Ebene (`ReduceAxisKeepDims`) existiert und ist getestet, Runtime-Parameter fehlt
