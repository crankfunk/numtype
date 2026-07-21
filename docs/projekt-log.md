# NumType — Projekt-Log (historisches Narrativ)

Dieses Dokument ist das vollständige Scheiben-für-Scheiben-Narrativ des Projekts, am
2026-07-19 **wörtlich** aus der CLAUDE.md hierher verschoben (einzige Änderung: Circa-Tilden
`~` → `≈`, weil GitHubs Markdown einzelne Tilden als Strikethrough paart). CLAUDE.md trägt
seitdem nur noch Regeln + aktuellen Stand; dieses Log wächst am Ende pro Scheibe weiter
(Hausregel in CLAUDE.md, „Obligatory workflow" Punkt 5). Detailtiefe pro Scheibe: die
jeweiligen `docs/*-spec.md` + `docs/*-ergebnisse.md` bleiben die Primärquellen.

## Verlauf bis v0.1.0 (Stand 2026-07-19, ursprünglich „Current phase")

Working the release roadmap (docs/roadmap.md, committed 2026-07-10). Done since: the Kern-06 follow-up auto-routing slice (threshold 262_144 = 64³, measured via `bench:crossover`), **Spike 02** (editor latency: headless LSP harness against the native TS7 server — the hard roadmap-A1 gate PASSES with ≈3 orders of magnitude headroom; docs/spike-02-*), and **Spike 03** (compile-time bounds checks for literal integer `slice()` indices — `LiteralIndexBounds` in slice-literal.ts, error at the offending argument mirroring the runtime throw verbatim; never-wrong-only-incomplete: wide/dynamic/non-plain-digit/mixed-union forms pass to the runtime backstop; negative literals work via `-${Abs}` strip + comparison only; 174/174 generated parity grid vs the runtime rule; machinery budget 1.036× — docs/spike-03-*), and **Spike 04** (type-level shape products: `LiteralShapeProduct<S>` via schoolbook digit-string multiplication in slice-literal.ts — 177/177 BigInt parity grid; never-wrong via boundary filters (union/never dims degrade — a product verdict is an unbounded value, Spike 03's subset-check pattern does NOT transfer) + MAX_SAFE_INTEGER cap (bigger digit strings would double-round through float64 into a WRONG literal); budget measured: +2,186 instantiations for unused machinery = pure declaration cost, ≈1,249/site typical shapes; decision GO as a DISCLOSED DEVIATION from the pre-registered gate rule (G1 1.0321× vs ≤1.02× and G2 mean missed as written; absolute + editor gates PASS; independent verify: "defensible, NO-GO not required") — docs/spike-04-*). Also done: **Spike 05 / roadmap A3** (variance design: `NDArray<S>` stays invariant per the two-of-three rule; new minimal covariant read view `NDArrayView<out S>` in ndarray.ts — exactly shape/strides()/toNestedArray(), no data (backend portability), no computed-shape members (TS2636: the `out` annotation check is ABSTRACT, factual monotonicity of `Transpose` doesn't count); class `implements NDArrayView<S>`; `AnyNDArray` stays as the documented both-ways-unsafe escape hatch; demo printArray migrated; check:diag pin now 133,727 — docs/spike-05-*), and **Spike 06 / roadmap A4** (range slices: negative literal start/stop + literal steps ≥ 2 compute statically — no signed addition needed, compare+subtract+clamp; new schoolbook long division `DivCeil`; provably-invalid literal steps (0, negative, dot-form ±1.5) are compile errors at the argument mirroring the runtime throw verbatim; 280/280 grid against the IMPORTED real `normalizeSliceSpecs` as ground truth; union boundary filters retrofitted to start/stop/step; check:diag pin now 188,378 = 3.77% of budget — docs/spike-06-*). **Phase A is complete, including the Kür.** **Kern 07 / Phase B item 1** is done and independently verified (2026-07-11): elementwise `sub`/`mul`/`div` + `dot`/`norm`/`cosineSimilarity` on NDArray + WNDArray — one generic strided elementwise kernel (kernels/elementwise.rs, mirrors add_strided line-for-line) + two reduction kernels (kernels/vector.rs: dot_strided single ascending accumulator, norm_sq_strided logical row-major) behind five appended ABI entry points; `norm`/`cosineSimilarity` have NO own kernels — pinned TS-side scalar compositions, since sqrt/*// are IEEE-exact JS ⇄ WASM; scalar consumer ops return plain `number` (documented asymmetry with `sum()`); `DotCheck` guard (spike/src/vector.ts) with verbatim runtime⇄compile message stems and union-dim no-claim; +772 differential tests, +48 cargo tests, three mutation proofs incl. the verifier's own; new plain-artifact pin `7a65d800…` (a phase adding exports legitimately changes the hash — freeze claim is source-level, see docs/kern-07-ergebnisse.md); disclosed+confirmed deviation: TS class bodies take insertion-only diffs, not literal EOF appends (private constructors) — docs/kern-07-*. **Kern 08 / Phase B item 5 remainder** is done and independently verified (2026-07-11): runtime `reshape`/`flatten` on both surfaces consuming `LiteralShapeProduct` — flatten hovers as a computed literal (`NDArray<[1048576]>`), reshape rejects provable product mismatches at the argument (message verbatim to the runtime throw: `reshape: cannot reshape array of size ${size} into shape [${ns}]`; dim-validity checked first: `reshape: invalid dimension ${d} …`); `ReshapeCheck` in spike/src/reshape.ts (wide → IsUnion → literal equality), stretch `LiteralReshapeDimInvalid` (negative/dot-form dims lifted; exponent/0/union no-claim; returns bare verdicts — messages built in reshape.ts, keeping slice-literal.ts append-only); WNDArray routes view-if-contiguous (isContiguous requires offset 0 — offset-shifted contiguous-shaped views conservatively materialize, never unsound), else nt_materialize; ZERO Rust changes, artifact hash byte-identical (strong freeze form); both Spike-04 obligations closed — guard wording fixed + editor-hover cost of real sites measured (bench:editor W6: 0.06 ms hover medians incl. big-dim flatten, in-family, ≈half of the W3 digit-stress; per-site guard computation ≈6k instantiations for huge-dim sites, declaration alone cheap); -1 inference deferred (FOLLOWUPS) — docs/kern-08-*. **Kern 09 / Phase B item 5 final rest** is done and TWICE-verified (spec + adversarial, 2026-07-12): runtime `keepdims` on `sum()` for both surfaces — one appended shape-metadata helper `keepDimsShape` (runtime.ts, append-only), `const KeepDims extends boolean = false` second type param feeding the pre-existing `ReduceAxis<S, Axis, KeepDims>`; guard stays keepdims-free; data byte-identical to non-keepdims (ZERO Rust changes, artifact hash byte-identical, adversarial verifier confirmed through cargo clean + rebuild); 367 new differential tests (non-circular: structural invariants + trusted non-keepdims reference), incl. views (transposed/offset/composed) and WNDArray false-parity; 5/5 broad mutants + 1 spec-verifier mutant all caught; new pins main 174,213 @ 126 files / stress 94,597; owner-confirmed D3 deviation (extending the existing sum methods — a new param cannot be insertion-only); adversarial finding: mixed-rank shape-union in ONE instance's type param (`NDArray<[2,3]|[2,3,4]>`) confidently mis-types `.sum()` — PRE-EXISTING facet (c) of the FOLLOWUPS union-guard item, not keepdims-related — docs/kern-09-*. This slice also triggered the process upgrade: "Qualitätssicherung, modellunabhängig" section below + docs/verify-runde-template.md (two-verifier rule). **Kern 10 / Phase B item 6** is done and twice-verified (spec CONFIRMED + adversarial HÄLT, 2026-07-12): IEEE-754 special values (NaN/±Inf/±0/subnormals/±MAX) injected into the differential generator — `SPECIAL_VALUES`/`nextF64Special`/`genDataSpecial` appended to prng.ts (existing generators byte-identical), new spike/tests-runtime/special-values.test.ts (619 cases) proves reference⇄v1⇄resident bit-identity (value-class for NaN, byte-exact otherwise) across add/sub/mul/div/sum/matmul/dot/norm/cosine/transpose; TEST-ONLY, zero Rust changes, artifact hash byte-identical; **key finding: SIMD-blocked matmul preserves subnormals (no flush) — proven catchable, both verifiers built the SIMD-flush mutant → fixture+random cases go red**; 3 real mutants caught total; new pins main 174,391 @ 127 files (Δ+178 from Kern 09's 174,213) / stress 94,597; disclosed spec fix (my `sum([-0,-0,-0])=-0` was wrong: sumRuntime seeds +0, so it's +0 — executor proved it via node -e and corrected the fixture); adversarial finding: the new dot/cosine random passes are weak against accumulation-ORDER bugs (Inf/NaN dominance at 35% injection masks rounding — order-sensitivity stays covered by vector.test.ts; the special passes cover PROPAGATION not order — documented in-code, not booked as order coverage); NaN-payload byte-exact preservation regression-tested only for transpose (reshape/slice/fromArray hold empirically → FOLLOWUPS) — docs/kern-10-*. **Kern 11 / Phase B item 7 (perf Kür, focused scope)** is done and twice-verified (spec CONFIRMED + adversarial HÄLT, 2026-07-12): a contiguous fast path in `add_strided` (add.rs) + `binary_strided` (elementwise.rs) skips the per-element `unravel` heap allocation (shape.rs:105) when both operands share the same shape, offset 0, and natural strides (`flat == a_off == b_off` ⇒ `out[i]=op(a[i],b[i])`) — **13–17× on the contiguous hot path**, bit-identical to the general path (mathematically airtight: natural strides ⇒ `idx·strides == flat` for every flat; empirically two committed `.to_bits()` equivalence tests per file). MEASUREMENT-DRIVEN: SIMD elementwise measured NO-GO (memory-bound), packing-reuse facet A measured 3.3% NO-GO — the `unravel` allocation was the real lever (docs/kern-11-*, "Messgrundlage"). Freeze: pre-edit clean rebuild reproduced the old pin `7a65d800…`, new pin **`0b9df4f10961f94cc1e378801fe66f958306b5135859a4a9bf480e77b2519c7d`**; used the ALLOCATING `compute_strides` (never the `#[cfg(atomics)]`-gated `_into` twins — their mere presence shifts artifact bytes, shape.rs:208-220). Freeze honesty (adversarial WAT-diff): "v1 add untouched" is a SOURCE/LOGIC claim, not artifact-byte — a compiler-generated `memcmp` helper for the new slice `==` shifts `i32.const` panic-location pointers crate-wide by ≈96 B in EVERY function (zero opcodes), which is exactly why the freeze proof is a whole-artifact hash. Two adversarial coverage gaps closed IN-slice (test-only, hash unchanged): guard-canonicality (`a_strides==b_strides` mutant survived cargo, only TS caught it) + size-0/size-1-interleaved fast-path. cargo 161, test:resident 4265+2, artifact-hash re-pinned. **check:diag DROPPED to 172,392 @ 128 files** (from Kern 10's 174,391 @ 127) from adding ONE strictly-typed value-only bench file (`spike/bench-core/elementwise.ts`) — reproduced in isolated worktrees by both verifiers, budget/coverage-neutral; **mechanism pinned 2026-07-12 (owner-requested bisection): CHECK-ORDER-dependent counting** — an empty `export {}` file reproduces ≈2,043 of the drop, magnitude varies with the file's sort-position (name), non-monotonic; global instantiation memoization means "who checks a shared recursive type first" decides fresh-vs-cached, and a file addition reshuffles that. Retroactively explains Infra-01's super-additive removal. Consequence: file-adding slices carry ±≈2,000 order-noise in the pin (not real type-cost) — decompose via empty-then-fill for clean attribution. Deferred (FOLLOWUPS): `unravel_into` general-case generalization (bigger lever, payoff inferred not measured — own measured slice), packing-reuse facet B (unmeasured) — docs/kern-11-*. Next: Phase B is COMPLETE (items 1–7 done). **Phase C is COMPLETE (2026-07-12):** Items 8/9 (browser threads port / stable-no_std) deliberately DEFERRED after a scoping pass — threads stay Node-only/experimental for v0 (no way off the pinned nightly today: build-std nightly-only + the 2026 RFCs 3874/3875 don't cover atomics rebuilds, wasm32-wasip1-threads is a dead end, no_std likely doesn't escape either; browser port gated on COOP/COEP headers a library can't set; docs/phase-c-threads-scoping.md). **Item 10 (backend-choice API) DONE + twice post-verified (Commit 5b0f951):** `NDArray.backend('wasm'|'threaded')` exposes the WASM/threads backends as an explicit, browser-safe opt-in (proven empirically via `process.moduleLoadList` trace — the JS `NDArray` default never eagerly pulls threaded.ts's static node imports), JS `NDArray` stays the default, ZERO Rust / hash byte-identical, new pins check:diag 175,634 @ 132 / stress 103,882 @ 82; docs/item-10-backend-api-*. This slice pioneered **"spec-verify BEFORE implementation"** (QA section below + verify-runde-template.md Baustein 0): the pre-build adversarial spec review caught 3 blockers incl. a false code-assumption (WNDArray.strides is a FIELD not a method → NDArrayView conformance deferred to the Spike-05 followup). **Phase D pre-work is UNDERWAY (binding spec docs/phase-d-vorarbeiten-spec.md, Commit 8b39c15, Baustein-0-verified with one blocker merged pre-build — facet (b) of the union item repros ONLY as a shape-union in ONE instance's type param, the `NDArray<A>|NDArray<B>` argument form is rejected by TS inference itself; owner-decided forks: strides→readonly property, mixed-rank unions degrade uniformly; execution order V3→V1→V2). V3 (browser smoke test) DONE + twice-verified + in-slice closures (2026-07-12):** first real-browser proof of the standard surface — Playwright/Chromium as devDep (runtime stays zero-dep), tsc emission via `rewriteRelativeImportExtensions` + `node:http` static server (`application/wasm` MIME is load-bearing: the loader's browser branch has NO instantiateStreaming fallback), COOP-free proven in-page (`crossOriginIsolated===false`), byte-exact differential matrix JS-`NDArray` ⇄ `backend("wasm")` incl. views/both reshape branches/special values, `backend("threaded")` rejects in-browser with the pinned stem; the verify round caught + closed IN-slice (test-only, hash untouched): stale-`.emit` false-pass on DIRECT playwright invocation (mtime-witness freshness guard — never run `playwright test` directly, always `pnpm test:browser`), streaming-path-TAKEN now asserted via a counting `addInitScript` wrapper (existence+MIME checks alone provably miss an `if(false)` loader mutant), `playwright.config.ts` typechecked in the browser leg; the Δ+78 main-pin move is REAL type-cost of the guard-test extension, bisected by verifier A incl. a same-length comment-control probe (Δ+0) — NOT order noise, i.e. a counterexample to blanket order-noise attribution for file EDITS (the ±2,000 order-noise rule below is about file ADDITIONS); docs/phase-d-vorarbeiten-v3-ergebnisse.md. **V1 (union-guard fix) DONE + twice-verified + in-slice closures (2026-07-13):** the three union facets are closed — (a) `IsUnion` filters first in `CompatDim`/`DimEq` (union dim → no-claim/wide; the pre-fix bug on matmul's contraction axis was a confidently-wrong REJECTION, a never-wrong violation); (c) new `RankUnknowable<S>` (dynamic rank OR union `S["length"]`) at all seven rank gates (Broadcast/MatMul/ReduceAxis/Transpose/SliceShape/SliceSpecsGuard/DotCheck), uniform degradation to `readonly Dim[]` per owner decision (deliberate precision loss on Transpose/SliceShape/dot where distribution was already correct); bonus real fix: `SliceSpecsGuard` arity leak on mixed-rank receivers; (b) tuple-wrapped `Guard` (`[Result] extends [ShapeError<infer M>]`) — uniform-error union rejects with ONE combined message, mixed accepts gradually; the Kern-07 claim that `NDArray<A>|NDArray<B>` arguments bypass the Guard does NOT reproduce on TS 7.0.2 (TS inference itself rejects that form — honest discrepancy note in FOLLOWUPS; the real leak form is the shape-union in ONE instance's param). Method: repro-first (20-assertion pre-fix RED proof in fresh worktree, byte-identically reproduced by verifier A), pins only in EXISTING test-d files, ZERO edits to slice-literal.ts/runtime.ts/resident.ts, zero re-expressed old pins, ZERO Rust/hash byte-identical; both surfaces pinned (WNDArray incl. the UW4 combined-message pin from the closure round — Guard-revert mutant now caught by 2 pins); over-degradation direction protected by 163 old-corpus pins (adversarial mutant 3); budget isolated: +1,060 src-only (matches the Baustein-0 estimate) + ≈+4,000 for the new union-heavy pins themselves; `never`-dim verdicts, union-AXIS-param (`sum(0 as 0|2)` — confidently wrong, RELEASE-RELEVANT own mini-slice before Item 11) and specs-tuple-union message cosmetics booked as FOLLOWUPS (pre-existing, out of V1 scope); docs/phase-d-vorarbeiten-v1-ergebnisse.md. **V2 (strides/NDArrayView/readonly-shape) DONE + THREE-way verified (spec CONFIRMED + adversarial HÄLT + covenant-verify: no invariant violation — first slice under the full covenant regime) + in-slice closures (2026-07-13):** strides harmonized to a readonly PROPERTY (owner fork; the NDArray method had ZERO call sites repo-wide, WNDArray field conforms as-is, the 8 field assertions stayed byte-identical), `WNDArray<S> implements NDArrayView<S>` (ThreadedBackend products covered automatically; covariance pins for both backends; NOTE: the implements keyword carries no downstream pins — structural typing — its value is the declaration-site self-check, and it CANNOT catch return-type narrowing of `toNestedArray(): unknown` → closed via `Equal<ReturnType<…>, unknown>` pins), deep-readonly `shape` via `Readonly<S>` on view + both classes (TS2636 pre-flight probe: homomorphic mapped types PASS the abstract out-check, unlike Transpose; `shape[0] = 2` rejection pins — the `= 99` form is CONFOUNDED by literal-type narrowing, only same-value assignment isolates readonly; 56 pins re-expressed `[…]` → `readonly […]`; class hovers stay `NDArray<[2, 3]>`, LSP-verified — covenant M3). **Unplanned finding + owner decision:** Readonly<S> incidentally LIFTED NDArray's measured invariance (the old block was an ACCIDENT of keepdims' `AllOnes<S>` return type, verbatim-confirmed in the baseline tsc error) → owner decided RE-invariantization via explicit `private declare readonly __variance: (s: S) => S` marker on NDArray AND WNDArray (property-style is mandatory — method shorthand is bivariant and inert; an `out` annotation is impossible: Transpose return types fire TS2636); NDArrayView stays the ONE enforced-covariant surface; the marker DROPPED the main pin by −10,308 (the old successful widening check fully resolved the AllOnes chain; the marker check fails early). Adversarial coverage closures: NDArray.strides getter VALUE pinned with hard literals (a reversed-strides mutant previously survived ALL suites), toNestedArray-narrowing pins, the three shape[0] pins delivered (A-Auflage — results doc had claimed them falsely, honesty-corrected). ZERO Rust / hash byte-identical (clean rebuild); docs/phase-d-vorarbeiten-v2-ergebnisse.md. **Phase-D-Vorarbeiten (V1–V3) are COMPLETE.** **Union-AXIS mini-slice DONE + three-way verified (spec CONFIRMED w/ condition + adversarial HÄLT + covenant-verify, 2026-07-13):** `IsUnion<Axis>` filter branch in `ReduceAxis` placed BEFORE the naked `Axis extends number` (position is load-bearing — distribution starts there; position mutant flips 16 pins), `IsUnion` exported from dim.ts (Δ0); every axis union degrades to `readonly Dim[]` like the dynamic axis (incl. all-invalid — no-claim like union dims, deliberate divergence from the Guard shape-level uniform-error rejection), runtime backstop proven in real Node with the verbatim message stem; KeepDims-`boolean` deliberately unfiltered (distribution there is correct-on-all-paths, pinned already-safe). Baustein 0 caught a REAL blocker pre-build: the `0|undefined` CALL form is structurally unreachable — TS strips `undefined` from inferred unions at OPTIONAL parameters (2×2 cross-probe; same root cause hits `keepdims?`) → owner scope reduction; the `Literal|undefined`-via-optional-params family is a documented KNOWN M2 VIOLATION (COVENANT.md v2, dated note under M2, norm unchanged) with the `UA_GAP` sentinel pin (proven a real observer via required-param mutant) + explicit-type-arg workaround pinned; fix candidate = overload split at the Item-11 API cut (FOLLOWUPS). New bench:editor workload W7 (union-axis hover, 0.06 ms, correctness-gated on `NDArray<readonly number[]>`). Pins 178,865 @ 132 / 102,182 @ 82 / browser 2,142 @ 75; ZERO Rust/hash byte-identical; docs/union-axis-mini-*. **Item 11 (API-Schnitt + Paketierung) is COMPLETE** (S1+S2+S3, 2026-07-17, each 3-way verified & committed — S1 48ee440 / S2 87e6e6b / S3 69ab47a; docs/item-11-s{1,2,3}-ergebnisse.md + docs/item-11-api-paket-spec.md): **S1** sum-overload umbau (COVENANT-M2 violation closed, both axis+keepdims facets via arg-count overloads + `reduce.ts` KeepDims→`boolean|undefined`; `NDArray<any>` impl-return owner-confirmed) + `slice-literal.ts`→`literal-arithmetic.ts` rename. **S2** emit/package pipeline: `tsconfig.build.json` + zero-dep `scripts/postbuild-dist.mjs` post-emit rewrite (fixes the three TS7 emit blockers — `.d.ts` `.ts`-endings, `new URL` worker path, `node:worker_threads` in threaded.d.ts via skipLibCheck) + `.wasm` bundling + package.json exports/main/module/types/files/sideEffects (`pnpm build:dist`). **S3** zero-dep-guard + package-smoke as CHECKED GATES (`pnpm test:package`: runtime smoke against `dist/index.js` incl. `backend("wasm")` real-.wasm-load, consumer type-smoke against `dist/index.d.ts`, independent comment-aware `check-dist-emit.mjs` precision gate — the ONLY Blocker-1 guard). COVENANT v2→v4 (M2 closed in v3; Z2 refined in v4: build-gated typechecks run in test:package, not `pnpm check`). npm-name/author-field/LICENSE deliberately deferred to Item 13. **Item 12 (Qualitäts-Portfolio + CI) is COMPLETE** (2026-07-18, three-way verified): an 8-job GitHub-Actions CI on ubuntu-latest (check / cargo / test-node / test-browser / test-threaded / freeze / editor-gate / demo), trigger `push:[main]` + `pull_request` (no double runs — Baustein-0 D-2); **rustc 1.95.0 pinned** via `rust-toolchain.toml` for a reproducible freeze-hash; the freeze-hash gate uses a platform-labelled pin SET (`scripts/check-freeze-hash.mjs`, D4 — a wasm artifact is not cross-host byte-stable, so the first ubuntu run empirically clarifies whether the Linux hash differs from the macOS pin, then it's added as a second entry); `bench:editor` hardened from report to a real gate (`enforceHardGate`: correctness + W1–W7 instantiation pins exact-match hard, latency at the 2x ceiling, `process.exitCode`); `--test-timeout=120000` in test:core/resident/package/threaded (F6 closed); a zero-dep CI-runnable S1 guard (`spike/tests-runtime/s1-import-guard.test.ts`) mirroring the `covenant-s1` rule with a **string-aware text scanner** — the initial line-by-line version was a deceptively-green gate (Verify-B found two multi-line import bypasses — `import(\n"x")` AND `from\n"x"`; the repo has no formatter forcing single-line), fixed in-slice by a comment-stripping state machine + text-wide scan, non-vacuity self-test pins both forms. **No Vitest** — `node --test` stays (owner decision, zero-dep). Three-way verified: A CONFORM (all gates fresh-green), B found+fixed the two S1-guard bypasses, C no covenant violation. NO Rust touched → artifact hash byte-identical `0b9df4f1…`; COVENANT unchanged (v4). npm-name/author/LICENSE stay deferred to Item 13. **Item 13 (release mechanics) is COMPLETE — numtype@0.1.0 is LIVE on npm and the repo is PUBLIC (2026-07-19).** The metadata/docs prep (earlier commits) — Apache-2.0 license (LICENSE + NOTICE + package.json; owner decision, the patent grant over MIT protects the from-scratch kernel algorithms), README rewrite (ANSI-Shadow figlet **signature banner** + typecheck-verified usage examples [the "examples run verbatim" release gate, proven via a temp check file against the real API] + the three §5 qualifications verbatim + a dedicated zero-dep section), `engines` node >=20, `author` crankfunk; a pre-OSS **privacy audit** ran clean (repo + full history — only the owner's first name removed from one doc). Release session (2026-07-19): pre-flight verified (name `numtype` free on the registry, tarball clean via `npm pack --dry-run`, CI green, no CI secrets, no rulesets, all commits on the GitHub-noreply identity); three gaps found + closed pre-publish (**NOTICE was missing from the tarball** — npm auto-includes LICENSE/README but NOT the Apache-2.0 NOTICE → added to `files`; version 0.0.0→**0.1.0**; **`prepublishOnly` = `pnpm test:package`** as publish airbag so a publish can never ship a stale/missing dist); README gained **"Why NumType exists"** (the original motivation: the TS ecosystem's missing NumPy counterpart, minimum-viable framing) + an **editor-support-out-of-the-box** paragraph (no extension/plugin/codegen; verified on TS 7.x, older majors untested → FOLLOWUPS TS-5.x consumer smoke); HANDOFF.md untracked (gitignored, stays local) + contributor note atop CLAUDE.md. First publish manual with 2FA, registry-verified (41 files / ≈435 kB unpacked); tag `v0.1.0` (commit 37335d0); repo flipped public; GitHub metadata set (description, homepage→npm page, 11 topics). **Corrected fact (verified against npm docs):** Trusted Publishing is configured per-package on npmjs.com (+ `id-token: write` in the workflow) and works from private repos too — only the provenance BADGE requires a public repo; setup is an optional FOLLOWUP. Item-13 leftovers deliberately deferred to Item 14: demo GIF, launch blog post, research-notes curation (FOLLOWUPS). Finding: the threads `.wasm` is NOT in the npm tarball (only the stable `numtype_core.wasm`; the threads artifact needs the pinned nightly + build-std) — `backend("threaded")` is checkout-only, documented in the README + FOLLOWUPS (recommend keeping it checkout-only for v0). Next: **Item 14** (v0.1 research preview — incl. the deferred Item-13 leftovers: demo GIF, launch blog post). See HANDOFF.md. The check:diag trend question is RESOLVED (owner decision 2026-07-11, implemented + verified as **Infra 01**, docs/infra-01-stress-split.md): digit-arithmetic stress cases (≥13-digit operands / MAX_SAFE_INTEGER cap probes) live in the separately-measured `spike/tests-stress/` tsconfig; `pnpm check` is a COMPOUND (root + stress — nothing rots; non-vacuity proven in both directions via corruption tests), `check:diag` stays root-only (main pin **173,716**, a NEW baseline — never compare across the split), new `check:diag:stress` (stress pin **94,523**, ungated by design). Realistic/semantic pins (incl. the 1024×1024 headline case and all degrade rows) stay in the main corpus. Done, verified, and committed: Spike 01 (type layer — docs/spike-01-*), Kern 01 (from-scratch kernels behind a hand-rolled `extern "C"` ABI, bit-identical to the naive TS reference — docs/kern-01-*), Kern 02 (zero-copy residency incl. fromArray Float64Array overload — docs/kern-02-*), Kern 03 (strided views: O(1) transpose, refcounted buffers, strided ABI entry points + status 4, `contiguous()` — docs/kern-03-*), Kern 04 (blocked+packed+SIMD128 matmul, bit-identical under the "bit-identity law": vectorize only ACROSS output elements, ascending-k single accumulator chain, no FMA/relaxed-simd — docs/kern-04-*; 2.1–3.25× over the Kern-03 scalar kernel, Kern-03's view-matmul penalty erased by packing), Kern 05 (slicing: O(1) slice views with the first nonzero offsets, zero Rust changes; type layer = gradual core rules + statically computed slice dims via from-scratch digit-string arithmetic in spike/src/slice-literal.ts — docs/kern-05-*), the three small hardenings (v1 OOM path, test-list guard, ABI prevalidation under scope (a)), Kern 06 (threads: hand-rolled substrate, no wasm-bindgen — separate shared-memory artifact on pinned nightly-2026-07-09 + -Zbuild-std, allocation-free `nt_matmul_blocked_partial` split across output rows only ("parallel bit-identity law": bit-identical by construction for any worker count/split), persistent worker_threads pool with per-worker shadow stacks + Atomics handshake, poisoned-pool error semantics with frees deferred past worker.terminate(); 2.86–4.42× at n=512/1024 with 8 workers, threads lose at small n — docs/kern-06-*). The naive TS runtime remains the correctness reference; the v1 copy-based backend remains the frozen performance baseline (its kernels/entry points stay byte-for-byte untouched). Build note: wasm builds need the repo-root `.cargo/config.toml` simd128 rustflag, and cargo config discovery is CWD-based — run all commands from the repo root (a compile_error! guard fires if the flag is lost). Every phase follows: binding spec doc → implementation → fresh-context verification → results doc with post-verification addendum → KB capture → commit.

## Historische Commands-Sektion (mit Pin-Historie, Stand 2026-07-19)

`pnpm check` (types; COMPOUND seit Phase-D-V3 DREIfach: root + `spike/tests-stress` + `spike/tests-browser`-tsconfig — Stress- und Browser-Strecke sind Teil jedes checks, nur die MESSUNG ist getrennt; Nicht-Vakuität aller Legs per Korruptions-Tests bewiesen) · `pnpm check:diag` (Haupt-Pin **187,918 @ 135** Files seit Item 12 (Δ+1 File `s1-import-guard.test.ts` im Root-Korpus → File-Set-Änderung, Order-Noise-behaftet, NICHT gegen 186,691 verrechenbar; neue `check:freeze`-Skript + ci.yml sind Infra, nicht im spike-Korpus); davor **186,691 @ 134** Files seit Item 11 / S3 (Δ+2 Files: zero-dep-guard.test.ts + package-smoke.test.ts im Root-Korpus → File-Set-Änderung, Order-Noise-behaftet, NICHT gegen 179,986 verrechenbar; A-Verifier maß den zero-dep-guard-Teileffekt isoliert als +53 @ 133; DREIfach verifiziert, docs/item-11-s3-ergebnisse.md); davor **179,986 @ 132** Files seit Item 11 / S1 (Δ+1,121 ggü. 178,865 — aber T1b RENAMT eine Root-Korpus-Datei (slice-literal.ts→literal-arithmetic.ts) → Sort-Position-Order-Noise bis ±≈2,000, NICHT als reine Typkost gegen 178,865 verrechenbar; DREIfach verifiziert, docs/item-11-s1-ergebnisse.md); davor 178,865 @ 132 seit der Union-Axis-Mini-Scheibe (Δ+653 echte Typkosten: Filter-Zweig + 22 union-lastige Pins, Dateizahl konstant); davor 178,212 @ 132 seit Phase-D-V2 — NETTO-Rückgang trotz neuer Pins: der `__variance`-Marker allein bringt Δ−10,380 (der alte erfolgreiche Widening-Check löste die teure AllOnes-Kette voll auf, der Marker-Check scheitert früh — dekomponiert im Scratch-Worktree), die V2-Substanz davor war +7,726 (Readonly<S>-Re-Expressionen + Kovarianz-Pins); davor 180,794 @ 132 seit V1 (Δ+5,082 ggü. V3s 175,712, isoliert: +1,060 reine src-Kosten der Union-Filter/RankUnknowable/Guard-Härtung + Rest die neuen union-lastigen test-d-Pins selbst, Dateizahl konstant = echte Typkosten), 175,712 @ 132 seit V3 (Δ+78 = ECHTE Typcheck-Kost der test-scripts-guard-Erweiterung, per Bisektion + gleichlanger Kommentar-Kontrollprobe (Δ+0) belegt, KEIN Order-Noise), 175,634 @ 132 seit Item 10 (172,392 @ 128 war Kern 11) — Δ from Kern 10's 174,391 @ 127 is a real, reproducible DECREASE from adding one bench file; MECHANISM PINNED (2026-07-12): the counter is CHECK-ORDER-dependent — a *empty* `export {}` file at the same path drops it ≈2,043, and the drop varies with the file's NAME/sort-position (`aaaa…` −2,034 vs `zzzz…` −304), non-monotonic — so a file addition carries an order-noise term up to ≈±2,000 that is NOT the change's real type-cost; see docs/kern-11-elementwise-fastpath-ergebnisse.md "Mechanismus GEPINNT") / `check:diag:stress` (Stress-Pin **102,877 @ 82** seit Item 11 / S1 (Δ+695, Dateizahl konstant 82, nicht bisektiert); 102,182 @ 82 seit der Union-Axis-Mini-Scheibe (Δ+86); 102,096 seit Phase-D-V2 (103,511 seit V1 (Δ−371, dim.ts-Content-Effekt, stabil, nicht bisektiert), 103,882 seit Item 10, 94,597 seit Kern 09)) / `check:diag:browser` (Browser-Korpus-Pin 2,142 @ 75 seit Phase-D-V3, ungated by design) · `pnpm test:core` (v1 differential + meta, 818 seit Phase-D-V3: +1 Guard-Invariante (d)) · `pnpm test:resident` (4278+2 = 4280 seit Phase-D-V2 (+1 NDArray-strides-Getter-Wert-Test aus der Schließungsrunde); 4279 seit Item 10; +`:gc` with --expose-gc) · `pnpm test:threaded` (69 seit Item 10; seit Item 12 baut es BEIDE Artefakte — `build:wasm && build:wasm:threads` — weil die threaded-Tests bit-identisch gegen den STABLE-Core vergleichen (numtype_core.wasm), nicht nur gegen den threads-Core; braucht also stable 1.95.0 UND die pinned nightly-2026-07-09 toolchain with rust-src, install command in scripts/build-wasm-threads.sh) · `pnpm test:browser` (Phase-D-V3: Playwright/Chromium-Smoke, 4 Tests, ≈3 s; der Wrapper emittiert IMMER frisch — NIE `playwright test` direkt aufrufen, ein mtime-Freshness-Guard wirft sonst bei stale/fehlendem `.emit`; Erstinstallation: `pnpm exec playwright install chromium`) · `pnpm test:package` (Item 11 / S3: baut zuerst `build:dist`, dann Emit-Präzisions-Gate `scripts/check-dist-emit.mjs` + Laufzeit-Smoke `spike/tests-package/package-smoke.test.ts` gegen `dist/index.js` (3 Tests: JS-`NDArray`, `backend("wasm")`-WASM-Ladepfad, `backend("threaded")`-Rejection) + Konsumenten-Typ-Smoke `spike/tests-package/consumer/` gegen `dist/index.d.ts`; braucht KEIN nightly, nur die stable-wasm) · `pnpm build:dist` (Item 11 / S2: emittiert das publizierbare Paket nach `dist/` — `build:wasm && rm -rf dist && tsc -p tsconfig.build.json && node scripts/postbuild-dist.mjs && .wasm-Kopie`; `dist/` ist gitignored, kommt via `files:["dist"]` in den Tarball; Konsument nutzt `skipLibCheck:true`-Default) · `pnpm demo` (all three backends, asserted equal) · `pnpm bench:scaling` / `bench:chain` / `bench:strided` / `bench:blocked` / `bench:slice` / `bench:threaded` / `bench:crossover` (kalibriert die Auto-Weiche, nightly) / `bench:editor` (Editor-Latenz via LSP-Harness, ≈1,2 s; seit der Union-Axis-Scheibe 7 Workloads — W7 gated den degradierten Union-Achsen-Hover) / `bench:elementwise` (Kern 11: contiguous fast-path vs. general-path add/sub/mul/div, ≈12-18x internal win at n=1024) · `cargo test --manifest-path crates/core/Cargo.toml` (161 seit Kern 11: 157 Kern-10-Basis + 2 Pfad-Äquivalenz + 2 adversarial-Follow-up-Coverage; Achtung: die zuvor dokumentierte 110 war Doku-Drift, tatsächliche Baseline war 109). Note: test scripts use EXPLICIT file lists in package.json — new test files must be added there manually; test-scripts-guard.test.ts (part of test:core) fails if a file is unlisted, double-listed across test:core/test:resident/test:threaded, or missing on disk; seit Phase-D-V3 deckt Invariante (d) auch spike/tests-browser ab (Browser-Testdateien müssen in test:browser registriert sein und dürfen in keiner node-Liste stehen).

## Item 14 — v0.1 research preview (2026-07-19, Abschluss der Roadmap)

Der Release-Tag selbst fiel unter Item 13 (numtype@0.1.0 auf npm, Repo public, Tag v0.1.0);
Item 14 war die Launch-/Sichtbarkeits-Schicht in vier Bausteinen, alle am selben Tag:
**(d)** README-Sektion „Versioning: what to expect before 1.0" — 0.x-SemVer geschärft für eine
Bibliothek, deren inferierte Typen Teil der API sind (jede Typ-Änderung = Minor, nie Patch;
die CI-gegateten Covenant-Zusagen als 0.x-stabil benannt; Typ-Präzisierung als erwartbare
Richtung). **(a)** Demo-GIF (docs/assets/numtype-demo.gif, ≈2,7 MB, absolute
raw.githubusercontent-URL): drei Szenen — matmul-Hover `NDArray<[2, 4]>`, Slice-Arithmetik
`NDArray<[900]>`, Shape-Fehler am Argument — AppleScript-getippt in VS Code (TS 6.0.3, die
gebündelte Version; TS 7 hat keinen tsserver, die klassische Integration kann ihn nicht laden —
alle drei Szenen wurden vorab empirisch gegen 6.0.3 UND 7.0.2 verifiziert, erster Datenpunkt
zur offenen Alt-Major-Frage). Fünf Anläufe; die Fehlschläge waren lehrreich genug für eine
eigene KB-Notiz (Kern: globale Hotkeys kollidieren mit synthetischem Tippen — QWERTZ-Klammern
sind Option-Kombos, `[1024]` = Doppel-Option-Tap → löste Claudes Quick-Entry aus und schickte
einen halben Prompt ab; Lösung: usage-first-Choreografie gegen Unused-Diagnosen, dediziertes
Einzel-Keybinding statt Cmd+K-Chord, expliziter Modifier-Down/Up, Schritt-Log, absolute
Go-to-Line-Navigation). Produktions-Setup in ~/Documents/CODE/numtype-demo-gif (außerhalb des
Repos), als erster echter Konsument des veröffentlichten Pakets. **(c)** docs/README.md als
englischer Reading Guide (Konventionen + Ehrlichkeitsregel, begründeter Sprachhinweis, drei
Leser-Pfade, „If you only read three documents"; 44 Links verifiziert) — bewusst keine
Massenübersetzung. **(b)** Launch-Blog-Post „Teaching the type checker arithmetic"
(https://marvinmuegge.com/notes/teaching-the-checker-arithmetic/), in eigener Session nach
Prompt-Vorgabe verfasst, Owner-publiziert; das Code-Beispiel nachträglich gegen das
veröffentlichte Paket verifiziert (Rang-1-matmul: `inner dimensions 900 and 5 do not match`,
echt). Dazu im Zuge: Prior-Art-Credit für die Digit-String-Repräsentation (ts-arithmetic) in
„The core idea" — Beitrag ist die Anwendung, nicht der Trick. Offener Mini-Befund: der Post
verlinkt Repo/npm nicht (liegt auf der Website). Die Roadmap ist damit vollständig
durchgespielt; weiter geht es post-Roadmap (FOLLOWUPS-Minis, optional Trusted Publishing).

## Dogfooding-Scheibe — RAG-Demo auf dem veröffentlichten Paket (2026-07-20, post-Roadmap)

Erste Scheibe des OSS-Wachstumskurses (Owner-Reihenfolge Punkt 2): eine echte
Konsumenten-Anwendung auf numtype@0.1.1, WIE SIE EIN NUTZER ERLEBT — `examples/rag-demo`
installiert das veröffentlichte Paket aus der npm-Registry (eigenes Install-Root, committetes
Lockfile), kein Workspace-Link auf den lokalen Stand. Deterministische RAG-Retrieval-Demo:
16 Dokumente, 8 Queries, from-scratch gehashte Zeichen-Trigramm-TF-Embeddings (djb2, D=256),
L2-Normalisierung als Matrix-Ausdruck, EIN matmul `[8,256] @ [256,16]` für die gesamte
Ähnlichkeitsmatrix, Ranking + Margin-Assertions (Schwelle 0.03, knappste echte Margin 0.0778),
Mean-Pooling-Sektion, zwei `@ts-expect-error`-Shape-Pins gegen die dist-Typen. Kern-Deliverable
war nicht die Demo, sondern der Friction-Log (F1–F6) und die kuratierte Op-Wunschliste W1–W5
(docs/dogfooding-rag-ergebnisse.md): **argmax/topk (P1, doppelt aufgetreten, null Ersatz) >
Skalar-Overloads für add/sub/mul/div (P2, macht mean fast gratis) > elementweises sqrt als
benannte exakte Unary-Op (P3, IEEE-korrekt gerundet = determinismus-sicher, KEIN Transzendenten-
Gate-Fall) > stack/fromRows (P4) > item/at (P5)**. Ehrliche Kalibrierung gegen die
HANDOFF-Erwartung: argmax/stack bestätigt, mean granularer als erwartet (die echte Lücke ist die
Skalar-Division), concat trat NIE auf (bleibt evidenzlos ohne Listenplatz), sqrt war der
unerwartete Doppel-Fund; Transzendente wurden im gesamten Workload nicht gebraucht.

Prozess: bindende Spec (docs/dogfooding-rag-spec.md, v1→v3) mit Baustein-0-Spec-Verify VOR der
Implementierung — der Verifier fing einen echten Blocker: pnpm-11-Default `minimumReleaseAge`
(≈24h) hätte das CI-Gate nach JEDEM künftigen Release gebrochen (`ERR_PNPM_MINIMUM_RELEASE_AGE_
VIOLATION`, dreifach reproduziert inkl. Cold-HOME); Mitigation = committetes pnpm-workspace.yaml
mit `minimumReleaseAge: 0` im Example. Verify-Runde A+B+C parallel: A CONFIRMED (alle Gates
frisch, check:diag-Pin 187,918 @ 135 exakt, Pflicht-Mutant beißt; klärte nebenbei die
test:core-Zahl: 822 auch am Vor-Scheiben-HEAD — die CLAUDE.md-„818" war vorbestehende Drift).
B HÄLT-mit-Befunden, MAJOR-Fund: „kein Root-pnpm-install" im CI-Job war empirisch falsch (pnpm
installiert beim Aufruf JEDES Root-Scripts auf kaltem Runner implizit die Root-devDeps) →
CI-Job auf direkte `-C`-Steps umgestellt (Spec-v3-Nachtrag); dazu engines-Feld, F-Nummern-
Angleich, `≈`-Fix. B bewies zudem Nicht-Vakuität breit (Margin-Mutant, Hash-Korruptions-Mutant,
@ts-expect-error-Positionsproben, Cold-Install-Repro des Baustein-0-Blockers). C (covenant-
verify): Z1/S1/M1–M5/Nicht-Ziele halten; EIN mittlerer Z2-Befund, unverdünnt: test:package prüft
ein Bauergebnis des AKTUELLEN Commits, test:example ein eingefrorenes Registry-Artefakt eines
VERGANGENEN — der Korpus rottet „still zwischen Releases". Owner-Entscheidung offen (FOLLOWUPS:
Covenant-v5-Präzisierung vs. mechanischer Registry-Tripwire), keine stille Auflösung.

## Op-Scheibe W1 — `argmax`/`topk` auf `NDArray` (2026-07-20, post-Roadmap)

Erste konkrete Op aus der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md, W1/F4 —
argmax trat zweimal auf, null Ersatz in der Surface). Bindende Spec
(docs/op-w1-argmax-topk-spec.md, v2 nach Baustein-0-Addendum): D1 grenzt bewusst auf die naive
JS-Klasse ein (kein WASM-Kernel, keine `WNDArray`/Threaded-Parität — FOLLOWUPS-Eintrag). `argmax`
übernimmt exakt `sum`s Arity-0/1/2-Overload-Muster (`ReduceAxis`/`Guard`/`OkShape` unverändert
wiederverwendet), mit EINER bewussten Abweichung: die niladische Form gibt `number` zurück
(Scalar-Consumer-Präzedenz von `dot`/`norm`/`cosineSimilarity`), nicht `NDArray<[]>`. `topk`
(Rang-1-only, `{values, indices}` im `torch.topk`-Stil) brauchte neue Typ-Maschinerie —
`TopkCheck<S,K>`/`TopkShape<S,K>` in vector.ts, appended, wiederverwendet die bestehende
Digit-String-Arithmetik aus literal-arithmetic.ts (`Compare`+`NonNegDigits`, minimal exportiert —
vorher unexportiert, Baustein-0-Blocker, Owner-Entscheidung „Exporte ergänzen"), NICHT
`LiteralIndexBounds` (dessen Index-Semantik `k=D` fälschlich als „out" und negatives `k`
fälschlich als „in" klassifiziert hätte — empirisch bewiesen, als verbindliche Warnung in die
Spec eingearbeitet). Datei-Disziplin D5 v2: runtime.ts/vector.ts zeigen im Diff AUSSCHLIESSLICH
Additionen nach dem letzten Bestandscode (vector.ts brauchte dafür einen zweiten, eigenständigen
`import`-Block statt die bestehende Zeile zu erweitern); ndarray.ts-Klassenkörper insertion-only,
Import-Zeilen am Dateikopf erweitert (Präzedenz: `sum`s eigene Importzeile wuchs genauso über
mehrere Kerne).

Zwei echte Befunde während der Implementierung, beide gefangen und gefixt VOR dem Commit: (1)
`argmax(undefined, true)` fiel erst fälschlich in den niladischen `number`-Zweig (Check war
`axisNum === undefined` statt `arguments.length === 0` — TS unterscheidet die Overloads nach
Argument-ANZAHL an der Call-Site, nicht nach Wert; ein 2-Arg-Aufruf mit Achsenwert `undefined`
verwarf so still `keepdims`), reproduzierbar rot vor dem Fix, grün danach. (2) Ein Test mit
handkonstruierter NICHT-kanonischer NaN-Payload (`0x7FF800000000DEAD`) zeigte gelegentlich die
kanonische statt die echte Payload — bisektiert auf eine V8-JIT-Tier-Eigenheit des bestehenden
`bitsOf`-Helfers (`new Float64Array([x])`-Array-Literal-Konstruktion), NICHT auf `topkRuntime`
selbst (separat per direktem `DataView`-Buffer-Read 5/5-mal als korrekt bewiesen); Test auf einen
lokalen `bitsAt`-Helfer umgestellt (Buffer-Read statt Array-Literal), seither deterministisch
grün über mehrere volle Testfile-Läufe.

Pin-Protokoll (D7 v2, gestufte Attribution, alle Zwischenpunkte gemessen): Baseline im frischen
Worktree exakt reproduziert (187,918 @ 135 · stress 102,877 @ 82 · browser 2,142 @ 75). ①
runtime.ts+ndarray.ts(nur argmax)+literal-arithmetic-Exporte: 188,383 (+465). ② +vector.ts-
Maschinerie+ndarray.ts-topk: 188,726 (+343). ③a neues Testfile LEER registriert: 179,186 @ 136
(−9,540 — Order-Noise, Datei-Hinzufügen reshuffelt die Fresh-vs-Cached-Instantiation-Partition,
deutlich über dem „±≈2,000"-Präzedenzfall, aber dieselbe dokumentierte Mechanik). ③b Testfile
GEFÜLLT: 182,249 (+3,063 echte Testkosten). final +test-d.ts-Pins: **184,225 @ 136 (+1,976)**.
Gesamtwachstum ggü. Baseline: **−3,693 — eine NETTO-ABNAHME**, weit innerhalb des
Absolut-Gates ≤+12,000 (Order-Noise dominiert die echten Neukosten). Zweimal gemessen,
byte-identisch. `bench:editor` W1–W7 verschoben sich UNIFORM um +804 (D7-explizit erlaubt,
Latenz/Correctness-Gate unverändert PASS) — Pins in editor-latency.ts aktualisiert.

**Ein offener Befund, ehrlich berichtet statt stillschweigend hingenommen:** `check:diag:stress`
verschob sich um +842 (102,877→103,719 @ 82, zweimal deterministisch reproduziert) — eine
Abweichung von T2s „stress/browser EXAKT unverändert"-Anforderung (`check:diag:browser` hielt
exakt, 2,142 @ 75). Bisektiert (temporärer Revert + Re-Messung, danach exakt wiederhergestellt):
`argmax` allein +469, `topk`s inkrementeller Beitrag +373 — Ursache ist NDArrays gewachsene
Klassen-Member-Fläche (zwei neue überladene generische Methoden), die JEDE `NDArray<S>`-
Instanziierung im stress-Korpus (der viele große literale Shapes für Digit-Arithmetik-
Grenzfälle instanziiert) marginal mehr Auflösungsarbeit kostet — dieselbe Ripple-Klasse wie bei
bench:editor, nur dass D7 diese Verschiebung dort EXPLIZIT erlaubt, für stress/browser aber
straffer formulierte, als die Realität hergab. NICHT ins Gate optimiert (Code nicht verkleinert,
um den alten Pin zu erzwingen); Owner-/Verify-Entscheidung, ob der Pin analog zu bench:editor
mitgezogen wird, steht aus.

Tests: `spike/tests-runtime/argmax-topk.test.ts` (30 Fälle, in test:core registriert — kein
WASM-Gegenstück existiert für D1, also kein klassischer Differential-Partner: Coverage kombiniert
mutations-scharfe Fixtures, eine selbstverifizierende Wort-für-Wort-Stem-Gleichheitsprobe gegen
`sumRuntime`s eigenen Throw, unabhängig geschriebene Brute-Force-Referenzen über ≈150
Zufallsfälle je Op inkl. NaN-Injektion, strukturelle keepdims-Invarianten und transponierte/
gesliceste Empfänger mit unabhängig hergeleiteten Erwartungswerten) + 37 neue Typ-Pins in
ndarray.test-d.ts (exakte Tupel, alle Degradationskanten, `@ts-expect-error` AM `k`-/Achsen-
Argument mit Mutationsprobe, vier Message-Gleichheits-Pins via `Guard<TopkCheck<…>,…>`+`Equal<>`,
MAX_SAFE_INTEGER-Kante). test:core 822→**852**. Alle D8-Gates frisch grün: `pnpm check`
(Dreier-Verbund), test:resident 4278+2 unverändert, cargo 161 unverändert (kein Rust berührt),
`check:freeze`-Hash byte-identisch, `graph-a-lama query lint` 0/0, `pnpm test:example` weiterhin
auf numtype@0.1.1. README: neuer eigenständiger Satz im „What's implemented"-Abschnitt (NICHT im
bit-for-bit-Usage-Block, der für argmax/topk falsch wäre) mit explizitem
„TypeScript-runtime surface only (no WASM kernel yet)"-Caveat. Vollständige Zahlen, Pin-Tabelle
und der offene stress-Befund: docs/op-w1-argmax-topk-ergebnisse.md. Post-Verification-Addendum
(Verify-Runde A+B+C) folgt.

### W1-Nachtrag: Verify-Runde & Abschluss (2026-07-20)

Verify-Runde A+B+C parallel: **A CONFIRMED** (beide Pflicht-Mutanten beißen exakt, alle Gates
doppelt deterministisch, Attributions-Tabelle nachgerechnet), **B HÄLT-mit-Befunden** (eigener
220-Shape-Differential 0 Abweichungen; Friction-Rückprobe: der F4-Workaround der RAG-Demo ist
real durch topk(2) ersetzt; Befunde F1 vorbestehender non-integer-Achsen-Fallback → FOLLOWUPS,
F2 RankUnknowable-Kante → Spec v4 folgt der D-V1.3-Hauspolitik statt der Spec-v1-Formulierung,
Policy-Pin ergänzt), **C kein Verstoß** (M1-Auslegung: kernel-lose Referenz-Ops nicht verboten;
Owner-Empfehlung M1-Präzisierung vor W2–W5 → FOLLOWUPS). Finale Pins: Haupt 184,330 @ 136
(Netto −3,588 zur Vor-W1-Baseline, Order-Noise-dominiert; echte W1-Maschinerie-Kosten in der
Attributions-Tabelle des Ergebnisse-Docs), stress 103,719 @ 82 (akzeptierter, attribuierter
Klassen-Surface-Ripple +842), browser 2,142 @ 75 (exakt). test:core 852. bench:editor-Pins
W1–W7 +804 uniform, doppelt gemessen. Prozess-Notiz: Baustein 0 fing den Export-Blocker VOR
dem Bau (Owner-Entscheid „Exporte ergänzen"), die MAJOR-Warnung vor LiteralIndexBounds hat die
Implementierung nachweislich befolgt — zweite Scheibe in Folge, in der der Pre-Impl-Verifier
den teuersten Fehler abfing.

### W2: Skalar-Overloads (add/sub/mul/div) + `mean` (2026-07-21)

Zweite Op-Scheibe aus der Wunschliste (docs/dogfooding-rag-ergebnisse.md W2/F2 — der
`fromArray([1],[2])`-Skalar-Wrap-Workaround, HANDOFF-Erwartung „mean" als granularere Lücke),
docs/op-w2-scalar-mean-spec.md (Version 2 nach Baustein-0-Addendum). Baustein 0 fing VOR dem Bau
einen echten BLOCKER: die v1-Annahme „Skalar-Overloads = reine Insertion" war falsch — TS2394
verbietet Overload-Signaturen vor einer body-tragenden Deklaration, die vier Bestandsmethoden
add/sub/mul/div MÜSSEN ediert werden (kein Richtungs-Spielraum, Alternativen brechen D1/D2) →
als erzwungene D6-v2-Ausnahme in die Spec gearbeitet (bodylose Overload-Signatur + neue
Skalar-Overload-Signatur + neue union-typisierte Implementierungssignatur, deren Rumpf die
ORIGINALE Logik byte-identisch in den `else`-Zweig verschiebt). Umgesetzt: `runtime.ts`
(Append) bekommt `scalarElementwiseRuntime(op, data, s)` (String-Dispatcher, elementweise
`data[i] op s`) und `meanRuntime` (`sumRuntime` + GENAU EINE Division pro Output-Element durch
`n` — bewusst NICHT `sum*(1/n)`, andere f64-Rundung); `ndarray.ts` konvertiert die vier
Bestandsmethoden nach D6-v2 (git diff bestätigt: die drei Rumpfzeilen jeder Methode erscheinen
als reine Kontextzeilen, kein `+`/`-` — byte-identisch verschoben, nicht neu geschrieben) und
bekommt eine neue `mean`-Methode (Overloads 0/1/2 exakt nach `sum`-Muster, KEIN
`arguments.length`-Sonderfall nötig — anders als `argmax` geben alle `mean`-Overloads
`NDArray<...>` zurück, die W1-Verwechslungsgefahr entfällt strukturell). D2 v2: ein
UNION-Argument über die Overload-Grenze (`number | NDArray<B>`) wird von TS als Ganzes
abgelehnt (TS2769) — dieselbe Kante, die `NDArray.backend(kind)` schon trägt, inkl.
funktionierendem `typeof`-Narrowing-Workaround, per Mutationsprobe (Direktive entfernt → echter
TS2769/TS2345 mit der vorhergesagten Message) non-vakuös bewiesen.

Der bindende Determinismus-Punkt D5 (`sum/n`, nie `sum*(1/n)`) ist zweifach nicht-vakuös
konstruiert: volle Reduktion (n=49, sum=5 → `5/49 = 0.10204081632653061` vs.
`5*(1/49) = 0.1020408163265306`, verschiedene letzte Nachkommastelle) UND ein Achsen-Fall
(shape=[4,49], Zeilensummen [5,9,1,2] — genau 2 von 4 Zeilen diskriminieren, die Spec-Warnung
„nicht jedes Beispiel diskriminiert" ist damit selbst bewiesen, nicht nur zitiert). Neues Testfile
`spike/tests-runtime/scalar-mean.test.ts` (482 Tests, in test:core registriert): expliziter
Op×Rang(0/1/2)×Spezialwert-Katalog gegen den nativen IEEE-Operator selbst, 160 randomisierte
`[1]`-Wrap-Byte-Äquivalenz-Fälle, ein Rang-0-Kontrast-Test gegen den ALTEN `[1]`-Wrap-Workaround
(beweist D2s Motivation als echten Lauf), 300 randomisierte `mean`-Cross-Checks gegen eine
unabhängig geschriebene Brute-Force-Referenz, `mean`-von-empty → NaN auf beiden
Reduktionspfaden (Kontrast zu `argmax`, das dort wirft), Stem-Wortgleichheit, `mean(undefined,
true)`. Typ-Pins (+19: 17 benannte `Expect<Equal<...>>` + 2 `@ts-expect-error`) in
ndarray.test-d.ts: `div(2)`-Shape-Erhalt exakt (Rang 0/2/wide/Readonly-S), Union-über-Grenze +
Narrowing-Workaround, `mean`-Wiring nach argmax-Präzedenz (niladisch, positive/negative Achse,
keepdims, plus die vier von der Spec namentlich verlangten Degradationsfacetten dyn-axis/
union-axis/mixed-rank/keepdims-union, plus OOB-Message-Pin) — bewusst mehr als die „≈4-6"-
Schätzung des Spec-Addendums (10 statt 4–6 in der `mean`-Gruppe), weil D7 den niladischen Pin
separat verlangt und „argmax-Muster" selbst ≈10 Pins trägt; ehrlich als Abweichung im
Ergebnisse-Doc vermerkt statt stillschweigend übernommen.

Pin-Protokoll (gestufte Attribution, empty-then-fill dekomponiert): Baseline im frischen
Worktree exakt reproduziert (184,330 @ 136 · stress 103,719 @ 82 · browser 2,142 @ 75). ①
runtime.ts+ndarray.ts (D6-v2 + mean): 185,204 (+874, Klassen-Surface-Wachstum). ②a neues
Testfile LEER registriert: 187,404 @ 137 (+2,200, Order-Noise). ②b Testfile GEFÜLLT: 189,368
(+1,964, echte Testkosten). final +test-d.ts-Pins: **190,092 @ 137 (+724)**. Gesamtwachstum
ggü. Baseline **+5,762**, deutlich innerhalb des Absolut-Gates ≤+10,000 — zweimal gemessen,
byte-identisch. stress: 103,719→**104,900 @ 82 (Δ+1,181)**, derselbe Klassen-Surface-
Ripple-Mechanismus wie W1s +842 (Datei-Anzahl unverändert, `spike/tests-runtime` ist nicht Teil
dieses Korpus), zweimal deterministisch reproduziert. browser: unverändert 2,142 @ 75, exakt.
`bench:editor` W1–W7 verschoben sich UNIFORM um +1,181 (w1/w2/w3/w5/w6/w7); w4 (die
Fehler-Datei mit den zwei absichtlichen Typfehlern) verschob sich um +1,220 — eine echte,
zweifach reproduzierte, attribuierte Abweichung (die `ShapeError`/`Guard`-Diagnosepfade lösen
sich gegen das größere Overload-Set anders auf), nicht weiter root-caused (Diagnosewert, kein
Korrektheitsrisiko). Pins in `editor-latency.ts` aktualisiert, `check:diag`-neutral verifiziert
(190,092 unverändert vor/nach dem reinen Daten-/Kommentar-Edit). Alle D8-Gates frisch grün:
`pnpm check` (Dreier-Verbund), test:core 852→**1,334** (482 neu), test:resident 4278+2
unverändert, cargo 161 unverändert (kein Rust berührt), `check:freeze`-Hash byte-identisch,
`bench:editor` Hard-Gate PASS nach Pin-Update, `graph-a-lama query lint` 0/0, `pnpm test:example`
weiterhin auf numtype@0.1.1, `pnpm demo` PASS (zusätzliche Absicherung). README: neuer
eigenständiger Absatz im „What's implemented"-Abschnitt direkt nach der W1-Notiz (die
bit-for-bit-Zeile im Usage-Codeblock bleibt unangetastet — der Block ruft `.mul()`/`.div()` nur
mit NDArray-Argumenten auf). FOLLOWUPS: das W1-Paritätsitem um einen W2-Nachtrag erweitert
(Skalar-Overload + `mean` fehlen auch auf `WNDArray`). Vollständige Zahlen, Diskriminator-Beispiele
und der Byte-Erhaltungs-Nachweis: docs/op-w2-scalar-mean-ergebnisse.md. Post-Verification-Addendum
(Verify-Runde A+B+C, Stufe 3) steht noch aus.

### W2-Nachtrag: Verify-Runde, F1-Fix & Recovery (2026-07-21)

Verify-Runde A+B+C: **A CONFIRMED** (drei Mutanten beißen, Byte-Erhaltung am Diff),
**B HÄLT-mit-Befunden** mit EINEM echten MAJOR: Der Overload-Umbau ließ die Shape-Message
des häufigsten Fehlerfalls (simpler Broadcast-Mismatch) hinter dem number-Decoy
verschwinden — TS meldet den Fehler des LETZTEN Overload-Kandidaten, kein bestehender Pin
sah Message-INHALT. Fix: Deklarations-Reihenfolge getauscht (Skalar zuerst, Guard-Träger
zuletzt, jetzt bindend in Spec v3) + neuer Diagnose-Qualitäts-Pin (echter tsc-Lauf auf
Außer-Repo-Fixture, assertiert den Broadcast-Stem; Nicht-Vakuität per Reihenfolgen-Mutant
bewiesen; drei schmale ambient.d.ts-Shims). **C kein Verstoß** — erste Anwendung des
M1-v5-Wortlauts (Paritätslücken-Bedingung erfüllt); ein M2-Wortlaut-Grenzfall
(Union-über-Overload-Grenze) als v6-Kandidat nach FOLLOWUPS. **Prozess-Zwischenfall,
offengelegt:** ein versehentliches `git checkout --` beim Mutanten-Revert warf die
uncommittete ndarray.ts auf HEAD zurück (exakt der Fall der Template-Regel „Mutanten als
revertierter Edit, nie checkout"); Recovery byte-genau durch den Implementierungs-Agenten
aus dessen Kontext, erneut am Diff verifiziert. Finale Zahlen: Haupt-Pin 188,563 @ 137
(+4,233 zur W1-Baseline), stress 104,900 @ 82, browser 2,142, test:core 1,335,
bench:editor PASS (w4 26453), Hash byte-identisch. Damit sind die Wunschlisten-Plätze
1 UND 2 geschlossen: `x.div(2)` liest sich als durch-2-teilen, `mean` existiert in
allen drei Formen — der F2-Workaround der RAG-Demo ist obsolet (Rückprobe bit-identisch).

### W3: `sqrt` — dritte Op-Scheibe der Dogfooding-Wunschliste (2026-07-21)

Wunschlisten-Platz 3 (docs/dogfooding-rag-ergebnisse.md W3/F1 — zweifacher Bruch der
natürlichen L2-Normalisierungs-Kette `mul→sum(axis)→sqrt→reshape→div` in der RAG-Demo, weil
kein `.sqrt()` existierte) ist geschlossen: `NDArray.sqrt(): NDArray<S>`, shape-erhaltend
bei jedem Rang inkl. Rang 0, niladisch (kein Guard, wie `norm()`/`flatten()`). Baustein-0
(brainroute:deep, kompakt) fand keinen Blocker — primärquellen-verankerte IEEE-Begründung
(ECMA-262 `sec-math.sqrt`: exakte 𝔽-Rundung, im Gegensatz zu jeder transzendenten
`Math.*`-Methode, die die Spec wörtlich „implementation-approximated" nennt), gemessener
Typ-Anteil +24 (Probe-Worktree), keine Symbolkollisionen. Umsetzung: `sqrtRuntime` als
reiner Append in runtime.ts (elementweise `Math.sqrt`, frisches Array), `sqrt()` als reine
Klassenkörper-Insertion in ndarray.ts nach `mean` (kein Bestandsmember editiert — anders als
W2s D6-v2-Overload-Umbau, hier reicht eine reine Append, da `sqrt` kein Overload-Partner-
Problem hat), W3-Testblock (227 Tests) an das bestehende `scalar-mean.test.ts` angehängt
(kein neues File), 6 neue Typ-Pins (5 `Equal` + 1 `@ts-expect-error`) an `ndarray.test-d.ts`.
D1 bewusst NDArray-only, kein WASM-Kernel (dieselbe COVENANT-v5-gedeckte Surface-Asymmetrie
wie W1/W2) — FOLLOWUPS-Paritätsitem um einen W3-Nachtrag erweitert. Die F1-Schließung ist
ZWEIFACH bewiesen, byte-identisch gegen die alte Hand-Loop-Formulierung aus
`examples/rag-demo/main.ts`: die Teilkette `m.mul(m).sum(1).sqrt()` UND die volle
L2-Normalisierung `m.div(m.mul(m).sum(1).sqrt().reshape([N,1]))`. Finale Zahlen: Haupt-Pin
190,636 @ 137 (+2,073 zur W2-Baseline, Absolut-Gate ≤ +3,000 eingehalten), stress 104,900 @
82 (Δ 0 — anders als W1/W2 diesmal KEIN Klassen-Surface-Ripple), browser 2,142 (Δ 0),
test:core 1,562 (+227), test:resident 4,278+2 unverändert, cargo 161 unverändert (kein Rust
berührt), `check:freeze`-Hash byte-identisch, `bench:editor` Hard-Gate PASS ohne
Pin-Abweichung, `graph-a-lama query lint` 0/0, `pnpm test:example` weiterhin auf
numtype@0.1.1. README: die W1/W2-Op-Notiz im „What's implemented"-Abschnitt um `sqrt`
ergänzt (bit-for-bit-Zeile bleibt wahr). Vollständige Zahlen und der F1-Schließungs-Beweis:
docs/op-w3-sqrt-ergebnisse.md. Post-Verification-Addendum (Verify-Runde, Stufe 3) steht noch
aus.

### W3-Nachtrag: Verify-Runde (2026-07-21)

A CONFIRMED (Mutant 219/227 rot — die 8 grünen sind exakt die abs≡sqrt-Fälle) ·
B HÄLT-mit-Befunden (zwei kleine Coverage-Lücken in-slice geschlossen:
Aliasing-Isolations-Test + größter-Subnormal-Pin; NaN-Payload-Kanonisierungs-Detail
dokumentiert) · C NULL Befunde mit eigenständigem Doppel-Urteil (sqrt ist algebraisch,
nicht transzendent; IEEE-Pflichtrundung — ECMA-262-Primärquelle seit Baustein 0).
Final: 190,640 @ 137 (+2,077), stress Δ0 (niladischer Member rippelt nicht — Kontrast
zu W1/W2 dokumentiert den Mechanismus weiter), test:core 1,564. Damit ist auch
Wunschlisten-Platz 3 geschlossen: die L2-Normalisierung der RAG-Demo läuft komplett
in numtype, byte-identisch zur alten Hand-Loop-Formulierung bewiesen.

### W4: `stack` — vierte Op-Scheibe der Dogfooding-Wunschliste (2026-07-21)

Wunschlisten-Platz 4 (docs/dogfooding-rag-ergebnisse.md W4/F5 — der selbstgebaute
`embedMatrix`-Zeilen-Flatten-Helper in examples/rag-demo/embedding.ts, `np.stack`-Reflex)
ist geschlossen: `NDArray.stack(rows)` — nur Rang-1-Zeilen gleicher Länge → Rang-2 `[N, D]`.
Baustein 0 (brainroute:deep, frischer Scratch-Worktree mit kompilierender Skizze) fand acht
verbindliche Typ-Formen VOR dem Bau (F1-F8, Spec-Addendum): Schichtung auf `readonly
Shape[]` statt `NDArray` in vector.ts (Zyklus-Vermeidung); homomorpher Mapped Type
`RowShapesOf<Rows>` statt der invarianz-kollabierenden `Rows[number]`-Extraktion (F2,
BLOCKER); ein `Shapes["length"] extends 0`-Gate vor jeder Element-Extraktion (F3, die
Leer-Tupel-Falle); Tupel-Wrapped-Akkumulator-Narrowing im Fold (F4); ein eigener
Array-Pfad via `number extends Shapes["length"]` (F5, Tupel-Rekursion matcht Arrays nie);
Wide-Sentinel-Dim-Merge nach CompatDim-Präzedenz (F6); Ablehnung eines Arrays mit
uniform beweisbar falschem literalen Rang (F7, sound weil auch das leere Array wirft);
IsUnion-Filter für Array-Union-Elementtypen (F8).

Umsetzung: `StackCheck`/`StackShape` als APPEND in vector.ts (eigener Import-Block, drei
gepinnte Message-Templates); `RowShapesOf`/`UnwrapRow` + die statische `stack`-Methode als
Insertion in ndarray.ts NACH `fromArray` (Baustein-0-Empfehlung: stack ist konzeptionell
ein Konstruktor); `stackRuntime` als APPEND in runtime.ts (ein Links-nach-rechts-Durchlauf,
dieselbe Reihenfolge wie der Typ-Fold, dann `Float64Array#set`-Zeilenkopie — exakt
`embedMatrix`s Algorithmus). Ein eigener Scratch-Probe (isolierter `tsc`-Lauf gegen einen
Symlink auf spike/src, außerhalb des Repos) fing WÄHREND der eigenen Verifikation einen
echten, von der Baustein-0-Skizze nicht abgedeckten Bug: `RowShapesOf`s naiv inline
geschriebener homomorpher Mapped Type kollabierte für ein Array mit UNION-Elementtyp
(der F8-Testfall) zu `readonly [number, never]` statt `[number, number]` — derselbe
Invarianz-Kollaps-Mechanismus wie F2, aber innerhalb der Array-Element-Auswertung der
Mapped-Type-Maschinerie selbst (TS wertet den Element-Typ-Ausdruck für ein Array EINMAL
non-distributiv gegen den — hier: Union — Elementtyp aus). Fix: `UnwrapRow<R>` als eigene
Generic mit eigenem naked Type-Parameter (derselbe „extra Generic erzwingt Distribution"-
Kunstgriff wie `ArrayRowD` in vector.ts) — nach dem Fix liefert der Probe korrekt
`[number, number]`, verifiziert am Typ-Pin `STACK_ARRAY_UNION`.

W4-Testblock (8 neue Tests) an scalar-mean.test.ts angehängt (kein neues File):
Stem-Pins über `stackRuntime` DIREKT und über die öffentliche API via dynamischer-Rang-
Zeilen (dieselbe „widen-past-the-guard"-Technik wie `mean(5)`s Achsen-Pin, kein unsicherer
Cast nötig); 1/2/3-Zeilen; D=0; ein Byte-exakter NaN-Payload-Test (`bitsOf`, mirroring
special-values.test.ts's Transpose-Fixture); die F5-Rückprobe (`embedMatrix`s Algorithmus
LOKAL nachgebaut, nicht importiert — das Beispielpaket bleibt bewusst außerhalb des
spike/-Kompilationsgraphen, ein Import hätte check:diags Dateizahl kontaminiert);
Large-N-Smoke (5.000×8); Aliasing-Isolation (W3-Lektion). 19 neue Typ-Pins in
ndarray.test-d.ts decken jede D2-Kante inkl. Message-Equality-Pins am Argument.

Finale Zahlen: Haupt-Pin 194,545 @ 137 (+3,905 zur W3-Baseline, Absolut-Gate ≤ +8,000
mit deutlichem Spielraum eingehalten), stress 105,752 @ 82 (+852, Klassen-Surface-Ripple
wie W1/W2 — ein neuer statischer Member rippelt über jede `NDArray<S>`-Instantiierung im
Korpus), browser 2,142 @ 75 (Δ0), test:core 1,572 (+8), test:resident 4,278+2 unverändert,
cargo 161 unverändert (kein Rust berührt), `check:freeze`-Hash byte-identisch,
`bench:editor` zunächst FAIL (uniform +845 auf allen sieben Workloads, 2× deterministisch
reproduziert — anders als W2s Verify-B-Fund differenziert dieser Ripple NICHT zwischen der
Fehler-Workload w4 und den übrigen, da `stack` keinem der add/sub/mul/div/mean-Overloads
hinzufügt), nach Pin-Update PASS; `graph-a-lama query lint` 0/0; `pnpm test:example`
weiterhin auf numtype@0.1.1; `pnpm test:package` PASS. README: neuer eigenständiger Absatz
nach der sqrt-Notiz. FOLLOWUPS: das Paritätsitem um einen W4-Nachtrag erweitert
(WNDArray/Rust-Kernel-Parität fehlt auch für `stack`). Vollständige Zahlen, der
F5-Schließungsbeweis und der Baustein-0-Fund im Detail: docs/op-w4-stack-ergebnisse.md.
Post-Verification-Addendum (Verify-Runde, Stufe 3) steht noch aus.

### W4-Nachtrag: Verify-Runde-Fix (Baustein B, BLOCKER-Klasse, 2026-07-21)

Baustein B fand einen zweiten, von der eigenen Umsetzungs-Verifikation nicht gefangenen
echten M2-Verstoß: `NDArray.stack([fixed, row])` mit `row: NDArray<[3]>|NDArray<[4]>` — eine
GEWÖHNLICHE Union über einen Ternary, keine `stack`-spezifische Konstruktion — kompilierte
konfident als `readonly [2, 3]` und warf zur Laufzeit. Root Cause: `UnwrapRow`s (bewusst für
F8s Array-Pfad) erzwungene Distribution über ein naked `R` distribuiert AUCH an einer
TUPEL-Position, deren eigener Zeilen-Typ zufällig eine Union ist; `StackFold`s naked
`Head extends readonly [infer D]`-Check distribuiert weiter, gabelt den Fold in parallele
Fortsetzungen mit unterschiedlichen Verdikten — Ergebnis eine gemischte Union
`Dim | ShapeError<...>`, die `Guard`s uniform-error-only-Ablehnung passieren lässt und deren
`ShapeError`-Zweig `StackShape`s `Extract<..., Dim>` still wegwirft. Dieselbe Fehlerform wie
`reduce.ts`s eigene `ReduceAxis`-Lektion (Union-Axis-Mini-Scheibe D-A.2) — dieselbe Lösung:
ein `IsUnion<Head>`-Gate VOR dem naked Match, Position load-bearing (dokumentiert im
StackFold-Kommentar mit explizitem `ReduceAxis`-Verweis). Sechs neue Pins in
ndarray.test-d.ts (Bs Repro beide Reihenfolgen, Doppel-Mismatch-Union, direkte
`StackDimMerge`-Wide-Abdeckung beide Reihenfolgen — F-ADV-2-Schließung, Array-Element-Union
verschiedener Ränge — Verify-C-Lücke, empirisch per Scratch-Probe bestätigt). Nicht-Vakuität
per Backup-Kopie-Mutant bewiesen: den `IsUnion<Head>`-Zweig entfernt → `pnpm tsc --noEmit`
schlägt mit exakt 4 Fehlern fehl (beide Repro-Pins + der Doppel-Mismatch-Aufruf selbst +
dessen Equal-Check), kein anderer Pin betroffen; Restore aus der Kopie, `diff` UND MD5 vorher/
nachher identisch. Finale Zahlen (2× je Messpunkt): Haupt-Pin **195.481 @ 137** (+4.841 zur
W3-Baseline, davon +936 allein der Fix), stress **105.758 @ 82** (+858, davon +6 der Fix),
browser 2.142 @ 75 unverändert, test:core weiterhin 1.572 (Fix ist rein typseitig),
`bench:editor` erneut uniform +6 verschoben und neu gepinnt, Hash weiterhin byte-identisch,
`graph-a-lama query lint` weiterhin 0/0. Pin-Zählungsfehler im Ergebnisse-Doc nebenbei
korrigiert (F-ADV-3: tatsächlich 16 Pins aus der Erst-Umsetzung, nicht 19 — jetzt 22 gesamt
mit den sechs neuen). Vollständiger Befund + Fix-Beweis: docs/op-w4-stack-ergebnisse.md,
Abschnitt „F-ADV-1".

### W4-Nachtrag: Verify-Runde mit In-Slice-Blocker-Fix (2026-07-21)

Die Verify-Runde zahlte sich bei W4 am deutlichsten aus: A CONFIRMED (inkl. unabhängiger
Reproduktion des selbstgefixten F8-Bugs — mit dokumentierter Warnung, dass
Standalone-tsconfig-Proben für distributive Conditional-Fragen unzuverlässig sind),
C ohne Verstöße, aber **B fand einen echten M2-BLOCKER**: Union-Row-Typen aus
gewöhnlichem Branching kompilierten mit konfidentem Literal-Claim und warfen zur
Laufzeit — die erzwungene Distribution des F8-Fixes leckte an Tupel-Positionen, und
Guard-Tuple-Wrap + Extract verschluckten den Error-Zweig der geforkten Fold-Union.
Fix nach Hauspolitik (IsUnion-Gate vor dem naked Destructure), 6 neue Pins (inkl.
der von B und C benannten Coverage-Lücken), Nicht-Vakuität per Mutant, von B
re-verifiziert (Typ ehrlich [2, number], Runtime wirft weiter). Lehre: „distribuiert
natürlich" ist nie eine sichere Scope-Annahme — Misch-Verdikt-Unions müssen VOR der
Destrukturierung gegated werden. Final: 195,481 @ 137 (+4,841), stress 105,758,
test:core 1,572, 22 Typ-Pins. Wunschlisten-Platz 4 geschlossen: embedMatrix ist
durch NDArray.stack ersetzt (byte-identische Rückprobe).

## Op-Scheibe W5: `item` (2026-07-21)

Fünfte und letzte Op-Scheibe der Dogfooding-Wunschliste (docs/dogfooding-rag-ergebnisse.md
W5/F3 — ein Skalar-Read aus der Score-Matrix): `NDArray.item(...indices)`, NumPys direkter
Skalar-Accessor. Spec docs/op-w5-item-spec.md Version 2 nach Baustein-0-Addendum (brainroute:deep,
Scratch-Worktree, kompilierte Form GELIEFERT UND GEMESSEN vor der Umsetzung — F1-F8):

- **F1 (BLOCKER, vorab gefangen):** `Guard<>` auf dem Rest-Parameter kollabiert zu TS2370 an der
  Deklaration — Rest-Parameter müssen array-artig bleiben. `ItemGuard<S, Idx>` folgt stattdessen
  der `SliceSpecsGuard`-Präzedenz: Tupel-geformt in jeder Verzweigung, nur einzelne Positionen
  werden zu `{__shapeError}`-Objekten retypisiert.
- **F2:** Der Fold ist S-GETRIEBEN (nicht Idx-getrieben wie `SliceSpecsGuard` — dort ist
  Under-Arity gewollt/Partial-Indexing, hier verboten/volle Indizierung).
- **F3 (Spec-Korrektur, erzwungene Mechanik):** Arity-Verstöße sind natives TS2554, nicht eine
  Custom-Message — für ein FEHLENDES Argument existiert architektonisch keine Position, an die
  eine Message gehängt werden könnte. `itemRuntime` trägt einen eigenen, runtime-only
  Arity-Stem.
- **F4 (Regression gefunden + gefixt):** Ohne `IsDynamicRank<Idx>`-Gate bricht ein Spread-Aufruf
  (`item(...arr)`) mit TS2556 — dasselbe Gate wie `SliceSpecsGuard`s `IsDynamicLength`, hier
  direkt aus dim.ts wiederverwendet (`Idx` ist strukturell ein `Shape`).
- **F5:** Dot-Form-Ablehnung ist NICHT in `LiteralIndexBounds` (dort silent-pass zu "unknown")
  — braucht `IsDotFormStep`s Export (ein `export`-Präfix, dieselbe Owner-gedeckte Edit-Klasse
  wie frühere Ein-Wort-Exports).
- **F6:** `LiteralIndexBounds`s Union-Verhalten ist konservativer als sein eigener
  Doc-Kommentar — der `IsUnion`-Pre-Gate in `ItemMark` ist doppelt begründet.
- **F7/F8:** TS7s Ein-Diagnose-pro-Call-Regel reproduziert; Stil `I extends number` statt
  `I & (string|number)`.

Umsetzung folgte der verifizierten Skizze 1:1 (kein zweiter Design-Fund während der eigenen
Implementierung). D1: VOLLE Indizierung (ein Index je Achse), Rang 0 = `item()` ohne Argumente,
kein Setter/Partial-Indexing/`at`-Alias. D3 (Runtime, `itemRuntime` APPENDED in runtime.ts):
Arity-Check, NumPy-Negativ-Normalisierung + Bounds-Check pro Achse (Stems wortgleich zu den
Typ-Stems, siehe unten), Offset-Summe über `computeStrides` — ein reiner strided Read, KEIN
Kernel (M1 v5: kernel-lose Referenz-Ops zulässig, solange die Paritätslücke in FOLLOWUPS
getrackt wird). D4: `item` als Klassenkörper-Append nach `sqrt`.

**D6-Kosten-Befund (der eigentliche Fund dieser Scheibe):** Die Erst-Umsetzung folgte der
Addendum-Skizze wörtlich — 5 separate `Expect<Equal<ItemGuard<...>, HandType>>`-Message-
Equality-Pins in ndarray.test-d.ts — und maß ein Gesamt-Delta von **+11.563**, fast das
Doppelte des ≤ +6.000-Gates. Bisektion (additive Entfernung via Backup-Kopie, keine
Mutanten-Notwendigkeit) zerlegte das: Quellcode allein +623 (nahe an der Baustein-0-Messung
+712), die restlichen +10.940 fast vollständig aus den Testdateien — davon +9.066 allein aus
`ndarray.test-d.ts`, und davon wiederum **~5.020 aus den 5 `ItemGuard`-Message-Pins**. Isolierte
Messung: EIN einzelner `Equal<ItemGuard<...>, T>`-Vergleich gegen einen strukturell ähnlichen
Handtyp kostet ≈1.700-1.750 Instantiations — eine Größenordnung über einer bloßen
`ItemGuard`-Referenz (≈80), einem Self-Compare (≈100-110) oder einem Vergleich gegen `unknown`
(≈100, da `tsc`s Assignability-Check dort die Quelle nicht voll normalisieren muss). Reaktion:
Pin-Konsolidierung — EIN kombinierter Zwei-Fehlerpositionen-Pin (`ItemGuard<[2,3], [0.5, 3]>`,
beweist Dot-Form UND Out-of-Bounds gleichzeitig, TS7s Ein-Diagnose-pro-Call-Fakt F7 ausnutzend)
statt fünf Einzel-Pins, Mixed-Rank-S über die bloße `ItemGuard`-Typebene statt eine
Klasseninstanzen-Union getestet, das `@ts-expect-error`-Real-Call-Trio auf drei statt vier
Fälle reduziert. Finale Zahlen (2× je Messpunkt, deterministisch): Haupt-Pin **201.354 @ 137**
(+5.873 zur W4-Baseline, Gate ≤ +6.000 eingehalten mit 127 Spielraum), stress **106.398 @ 82**
(+640, ausschließlich aus dem geteilten Quellcode — kein stress-eigenes File berührt), browser
2.142 @ 75 unverändert, test:core **1.588** (+16), `bench:editor` einmalig neu gesetzt
(uniform +628 über alle 7 Workloads, zwei Durchläufe grün), Hash weiterhin byte-identisch (keine
Rust-Änderung), `graph-a-lama query lint` weiterhin 0/0. Coverage-Auswirkung der Konsolidierung:
KEIN D2-Kanten-Verlust — jede Kante (Arity beide Richtungen, OOB positiv/negativ, Dot-Form,
gültiges negatives Literal, wide Rang, Union-Index, Mixed-Rank-S, dynamischer Spread) trägt
weiterhin mindestens einen Pin, nur die redundanten Mehrfach-Belege pro Kante wurden dedupliziert.

FOLLOWUPS trägt zwei neue Einträge: den D6-Kostenmechanismus selbst (offene Frage, ob
`SliceSpecsGuard` und andere bestehende Message-Pins denselben Kostenfaktor tragen — eine
Stichprobe deutete auf ≈1.049 für ein vergleichbares `SliceSpecsGuard`-Pin, günstiger als
`ItemGuard`s ≈1.700, aber noch immer weit über einem bloßen Referenzzugriff) und das
Aufsplitten von scalar-mean.test.ts (jetzt W2-W5-Sammelbecken, D6-Mandat der Spec). Mit W5 ist
die komplette Dogfooding-Wunschliste (W1-W5) abgearbeitet. Vollständiger Befund:
docs/op-w5-item-ergebnisse.md.

## Op-Scheibe W5 (item) + Wunschlisten-Abschluss (2026-07-21)

`item(...indices): number` — voller Skalar-Read mit NumPy-Negativ-Normalisierung,
Spike-03-Bounds-Reuse (LiteralIndexBounds ist hier die RICHTIGE Semantik — der
W1-Warnhinweis betraf topks andere Index-Semantik). Baustein 0 fing den fünften
Vor-Bau-Blocker der Serie (Guard<> auf Rest-Params = TS2370; slice()s Fold-Form ist
das Muster) und lieferte die verbindliche ItemGuard-Form gemessen (+712-Skizze).
Implementierung mit offengelegter Budget-Konsolidierung der Pins (Messbefund: ein
ItemGuard-Equal-Pin ≈1,700 Instantiations). Verify: A CONFIRMED (Kanten selbst
enumeriert — Konsolidierung verlustfrei), B widerlegte zwei META-Behauptungen
(IsUnion-Pre-Gate coverage-tot; F6-Prämisse falsch — LiteralIndexBounds' Union-
Disziplin trägt wie in Spike 03 dokumentiert) → Pre-Gate als reduce.ts-Policy-
Angleichung dokumentiert und per neuem Policy-Pin load-bearing gemacht (Mutant:
exakt 1 Zeile rot), C fand das M3-Wiederholungsmuster (native Diagnosen) → v6-
Kandidat in FOLLOWUPS. Final: 201,455 @ 137 (+5,974, Gate ≤ +6,000 — knappster
Lauf der Serie), stress 106,398, test:core 1,588.

**Damit ist die komplette Dogfooding-Wunschliste W1–W5 geschlossen** — jede Op
evidenzbasiert, jede mit Baustein-0-Fang vor dem Bau (5/5!), dreifach verifiziert,
zwei echte Verify-B-Blocker (W2-Diagnose-Verlust, W4-M2-Loch) in-Slice gefixt.
Nächster Schritt: 0.2.0-Bündel-Release (Owner-Publish), Example-Umstellung auf die
neuen Ops als Vorher/Nachher-Showcase, dann Scale-Probe.

## Release 0.2.0 — „the wishlist release" (2026-07-21)

numtype@0.2.0 publiziert (Owner, 2FA; ein abgelaufener npm-Login als einzige Hürde —
E404 beim Publish ist npm-Sprech für „nicht eingeloggt"). Inhalt: die komplette
W1–W5-Serie. Release-Mechanik wie designed: Version-Bump + Example-Dep-Bump im selben
Commit (fb28417) hielt den Registry-Tripwire grün; das UNVERÄNDERTE 0.1.1-Example lief
vor der Umstellung grün gegen 0.2.0 (Drop-in-Kompatibilität asserted — die erste echte
Bewährungsprobe der 0.x-SemVer-Politik). Danach die Showcase-Umstellung (dd60692):
alle fünf Workarounds durch die Ops ersetzt, JEDER gepinnte Score byte-identisch
(die W-Scheiben-Beweise, sichtbar im Konsumenten), FRICTION→RESOLVED-Kommentare
in-place + F→W-Tabelle in der Example-README, dritter @ts-expect-error-Pin (item-OOB).
Tag v0.2.0 gepusht (Ruleset-geschützt). Damit ist der Bogen geschlossen: Demo →
Friction-Log → Wunschliste → fünf verifizierte Op-Scheiben → Release → dieselbe Demo
auf den eigenen Ops. Nächster Owner-Reihenfolge-Punkt: Scale-Probe.

## Scale-Probe — „unproven at scale" fällt (2026-07-21)

Der dritte und letzte Punkt der Owner-Reihenfolge vom 2026-07-20. Ergebnis in einem Satz:
Die interaktive Latenz hält über alle 34 messbaren Sweep-Punkte (warmer Hover-Median
0,04–0,11 ms), die Kosten der Skala landen ausschließlich auf dem Kaltstart, und die einzige
harte Wand sitzt bei Rang 1024 — praktisch unerreichbar, aber real. Vollständige Zahlen und
Methodik: docs/scale-probe-ergebnisse.md.

**Der Prozess war hier wertvoller als das Ergebnis.** Vier Befunde, die ohne die vorgelagerte
Prüfung publiziert worden wären:

1. **Die Frontier-Zweitmeinung fing einen Blocker in der SPEC**, den der adversariale
   Spec-Verifier übersehen hatte: Achse (a) hätte ohne bindende Shape-Diversitäts-Vorgabe
   Cache-Treffer statt Skalierung gemessen. Gemessen liegt zwischen wiederholten und
   verschiedenen Shapes ein Faktor 19 — die bequeme Konstruktion hätte eine flache Kurve und
   den Satz „skaliert mühelos" produziert, ohne dass jemand falsch gemessen hätte. Dieselbe
   Falle steckte in Achse (b), wo die naheliegende Konstruktion „konstant" ergibt und die
   ehrliche „linear, 265 Instantiations pro Kettenglied".
2. **Beide Spec-Prüfer fanden gemeinsam einen Defekt in der bestehenden Mess-BASIS**: die
   generierten Workload-tsconfigs führten `spike/src/ambient.d.ts` nicht, weshalb ALLE sieben
   Editor-Workloads (nicht nur das absichtlich kaputte w4) mit sieben TS2591-Diagnosen liefen —
   unsichtbar, weil `enforceHardGate` das `hadTypeErrors`-Flag nirgends liest. Owner entschied
   „vorher reparieren" statt „erben": Vorab-Scheibe **V0** (c18aa7f), uniform +135 auf allen
   sieben Pins, Latenzwerte und `check:diag` unberührt, die publizierte Hover-Aussage hält auf
   der sauberen Basis.
3. **Verify-B fand ein VAKUÖSES Korrektheits-Gate** in der frischen Implementierung: Weil `tsc`
   lange Tupel in der Hover-Anzeige kürzt, war der Vergleich auf sechs Dimensionen verkürzt
   worden — bedingungslos für JEDEN Rang, auch die ungekürzten. Der Verifier extrahierte die
   echte Prüffunktion und bewies mechanisch, dass eine ab Position 6 durchgehend falsche
   Anzeige durchrutscht. Behoben durch eine elisions-bewusste Prüfung (Präfix + Suffix +
   Rekonstruktion der Gesamtlänge aus dem „N more"-Vermerk, Kürzungsfenster aus dem Text gelesen
   statt hartkodiert); acht Mutationen belegen die Wirksamkeit, zwei Kontrollläufe die
   Symmetrie.
4. **Zwei Verifier widerlegten unabhängig die Charakterisierung des Implementierers**, die
   Datei-Achse wachse „deutlich überproportional". Die Marginalkosten je Datei sind flach bis
   leicht fallend (4.868 → 3.798) — linear mit fester Grundlast. Echt überproportional ist nur
   die Rang-Achse (144 → 2.608 je Rang). Die falsche Beschreibung war nie committet, wäre aber
   in die publizierte Aussage gewandert.

Dazu ein Angriff, den Verify-B sich selbst stellte und ausräumte: In der distinct-Konstruktion
wachsen mit dem Dateiindex auch die Zahlenwerte, die Kurve könnte also Magnituden statt Vielfalt
messen. Eine Gegen-Konstruktion mit fest begrenzten Größen ergab 3.812/3.711 je Datei gegen
3.798–3.961 — der Confound existiert, verfälscht aber nichts.

**Owner-Entscheidungen dieser Scheibe:** alle vier Achsen (statt einer Teilmenge) · Sweep
on-demand plus EIN gepinnter Sentinel (statt gar keiner oder voller CI) · synthetisch mit
rag-demo-Eichung · Claim-Scope = Konsumenten-Skala mit ausdrücklich offener API-Flächen-Frage
(statt einer Extrapolation aus den W-Serien-Ripple-Zahlen) · ambient-Fix als Vorab-Scheibe ·
Z2-Abweichung durch Vertragspräzisierung in v6 auflösen statt dauerhaft dulden.

Covenant: keine Verletzung; zwei Textlücken (Z2 on-demand-Korpora, M2 Rang-Cliff als
Falsch-Ablehnungs-Grenze) als v6-Kandidaten dokumentiert. Das v6-Bündel steht damit bei vier
und ist reif für eine eigene kleine Vertrags-Scheibe.
