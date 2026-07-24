# MACVM interpreter throughput — S6 baseline

Recorded per `sprint_s06_detail.md` §Benchmarks' procedure (SPRINTS
standing rule 3: **tracking, not gating** — these numbers are not part of
any test's pass/fail criteria).

## Environment

- Host: Apple M4, 10 cores, macOS (Darwin 25.5.0, arm64)
- Build: `cargo build --release` (rustc 1.96.0)
- Date: 2026-07-02

## Procedure

`MACVM_TRACE=count` (prints total bytecodes dispatched at exit) plus
`/usr/bin/time -p` (wall clock) around each of 5 runs per benchmark;
`world/bench/fib.mst` (fib 25), a 30-variant of the same script, and
`world/bench/sieve.mst` (10 iterations, size 8190, expected count 1899).
Bytecode counts were byte-for-byte identical across all 5 runs of every
benchmark (determinism confirmed, per the procedure's requirement).

## Results

| Benchmark | Result | Bytecodes (median = all 5) | Median wall (traced) | bc/s (traced) |
|---|---|---|---|---|
| fib(25) | 75025 | 2,677,644 | 0.08 s | ~33.5M bc/s |
| fib(30) | 832040 | 29,625,095 | 0.90 s | ~32.9M bc/s |
| sieve ×10 | 1899 | 5,672,753 | 0.17 s | ~33.4M bc/s |

fib(30) wall time (0.90 s traced) is well under SPEC §13's `< 2 s` gate.

## `MACVM_TRACE=count` overhead

The three benchmarks above cluster tightly around **~33M bc/s with the
counter enabled** — noticeably below SPEC §13 row 1's 50M bc/s target.
Re-running fib(30) *without* `MACVM_TRACE=count` gives a median wall time
of **0.55 s** for the same 29,625,095 bytecodes: **~53.9M bc/s**, above
target. The counter itself (`sprint_s06_detail.md`'s own estimate: "cost ≈
1 add/dispatch, acceptable") measurably costs more than that in practice
on this build — a ~40% slowdown, not the "1 add" the doc's estimate
assumed. This is worth another look in a later throughput-focused sprint
(S10/S14/S15 per the SPRINTS doc), but is out of scope here: S6 is a
library sprint, not an interpreter-optimization one.

## Pass/fail against SPEC §13 row 1 (tracking only)

- fib(30) < 2 s: **PASS** (0.55 s untraced, 0.90 s traced — both well
  under).
- ≥ 50M bc/s: **PASS untraced** (~53.9M bc/s), **FAIL traced** (~33M
  bc/s) — the gap is attributable to the counter overhead noted above,
  not to per-bytecode dispatch cost. Since the procedure as written
  measures wall time *with* the counter active, the honest reading of
  this baseline is "fails as measured, passes with the counter removed
  from the hot path" — recorded for whoever picks up interpreter
  throughput work later.

# S10 tier-1 JIT — perf marker

Recorded per `tests_s10.md`'s "Perf marker" procedure (SPRINTS standing
rule 3: **tracking, not gating**). `world/bench/arith.mst`'s
`sumTo: 5_000_000` — a send-free, once-compiled smi arithmetic kernel
(`SmiArith Add`, the inlined `to:do:`'s `SmiCmpBr`, `Poll` at the loop
back-edge) — timed via `millisecondClock` after two small warm-up calls
through the same call site (so the compile itself never lands inside the
timed window), under `MACVM_JIT=off` vs `MACVM_JIT=threshold=1`, via
`just bench-s10` (`--release`). The gate WARNS below 5x and FAILS only
below 2x (an architectural-mistake tripwire, not a perf gate — gate item
3 of tests_s10.md's acceptance gate).

| Date | Commit | interp_ms | jit_ms | ratio |
|---|---|---|---|---|
| 2026-07-03 | 177abf1 | 1221 | 9 | 135.66x |
| 2026-07-03 | 353db27 | 1233 | 10 | 123.30x |

# S11 D8 bridge — allocation cost of the pre-S12 GC bridge

Recorded per `tests_s11.md`'s "Bridge accounting" stress/negative test
(SPRINTS standing rule 3: **tracking, not gating** for `bridge_old_allocs`
itself — `gc_under_compiled` IS gating: `just bridge-stats-s11` fails the
run if it's ever nonzero). The full world test suite, under
`MACVM_GC_STRESS=full:64` combined with `MACVM_JIT=threshold=1` (the same
combination `gate-s11` stress-tests), traced with `MACVM_TRACE=gc`.
`bridge_old_allocs` is every allocation the D8 bridge diverted old-direct
because a compiled frame was live (`compiled_depth > 0`) — non-moving,
so it costs old-gen space no scavenge can ever reclaim until S12 deletes
the whole bridge. `gc_under_compiled` is the number of times a
scavenge/full-GC actually ran while `compiled_depth > 0` — i.e. the
bridge failing to hold; must always read 0.

| Date | Commit | bridge_old_allocs | gc_under_compiled |
|---|---|---|---|
| 2026-07-04 | 7ac7b53 | 110 | 0 |

# S11 dispatch — perf marker (adapted, see world/bench/dispatch.mst's own doc)

Recorded per `tests_s11.md`'s gate item 4 ("Dispatch micro-benchmark"),
ADAPTED: the literal 3-class polymorphic design that file sketches cannot
compile at all under S11's as-built eligibility gate (`mono_smi_inline_send`
rejects any non-super send whose IC guard isn't `SmallInteger`, monomorphic
or not — see `world/bench/dispatch.mst`'s own header and
`sprint_s11_detail.md`'s STEP-10 NOTES for the full reasoning). This instead
times `world/bench/dispatch.mst`'s `runLoop: 5_000_000` — arith.mst's own
`sumTo:` shape with its inlined `+` replaced by a REAL super-send dispatch
per iteration (D4.6: the one non-arithmetic, non-`basicNew` send a compiled
method may contain) — under `MACVM_JIT=off` vs `threshold=1`, via
`just bench-s11` (`--release`). Same warn<5x/fail<2x tripwire as
`bench-s10` (tracking, not gating).

A smaller ratio than `bench-s10`'s ~130x is the EXPECTED, honest result: a
real send still costs a real dispatch even compiled (unlike inlined
arithmetic, which erases the cost entirely) — this benchmark measures that
cost, it doesn't erase it.

| Date | Commit | interp_ms | jit_ms | ratio |
|---|---|---|---|---|
| 2026-07-04 | 7ac7b53 | 1834 | 472 | 3.88x |
| 2026-07-04 | abe4f2e | 110 | 0 |
| 2026-07-04 | cdfab6a | 110 | 0 |
| 2026-07-04 | a1e57ac | 110 | 0 |
| 2026-07-04 | 04e774b | (bridge deleted) | 110 |

# S15 A6/A7 — Richards/DeltaBlue perf recording

Recorded per `tests_s15.md` T5's procedure: `world/bench/bench.list`
(`richards.mst`, `deltablue.mst`) run through the shared `Bench.mst`
harness (3 discarded warmups + median-of-outer, timed via
`millisecondClock`, excludes genesis/world load) under `MACVM_JIT=off` vs
`threshold=1` vs `threshold=1000`, via `scripts/perf.sh --release`.

## 2026-07-06 (commit f62a1e4) — Richards t=1 CORRECT for the first time

BUG D root cause 4 turned out to be two distinct bugs (OSR uninit frame
slots + c2i >5-arg marshaling overflow — see tests/repros/README.md and
f62a1e4's own message), both fixed. Richards now completes CORRECTLY
under `threshold=1` (23246/9297 golden values) — previously it could not
complete under any JIT threshold at all.

| Benchmark | interp_ms | jit t=1 | jit t=1000 | best/interp |
|---|---|---|---|---|
| richards | 204 | 193 | **blocked** (mid-threshold wrong-answer) | 1.1x (t=1) |
| deltablue | 208 | 119 | **blocked** (BUG C) | 1.7x (t=1) |

- **richards t=1 = 1.1x is a PERF gap now, not a correctness gap**: the
  run is correct but trap-heavy (the `work kind` two-way branch keeps one
  arm trapping — 60k+ uncommon traps observed per run), so most of the
  win is eaten by deopt/reexecute churn plus interpreted fallback for the
  7-arg-send creator methods the new eligibility cap declines. T5's
  "Richards ratio ≥ 5.0" gate therefore remains unmet — but the remaining
  work is optimization (trap-site healing / poly-arm compilation), not
  bug-fixing. `it_perf_s15.rs` still should not be written yet: it would
  fail on day one for perf reasons.
- **The mid-threshold silent wrong-answer (Richards t=100..20000,
  DeltaBlue t=1000 — very likely BUG C's own band)**: results are correct
  interpreted, at t≤10, and at t≥100000 (OSR-only compilation), but wrong
  whenever invocation-triggered compilation lands MID-run — the
  early-exit fraction tracks the threshold precisely (at t=20000 the
  scheduler dies ~92% through, exactly where `queuePacket:` crosses
  20000 invocations). Documented as the next investigation; blocks only
  the t=1000 columns above.
- Also known, pre-existing (stash-bisected, not from these fixes): the
  BUG D repro fails under `MACVM_GC_STRESS=1` + `threshold=1`
  (`doesNotUnderstand: size`), while passing under `MACVM_DEOPT_STRESS`.

## 2026-07-06 regression A/B: the f62a1e4 fixes are perf-neutral

Question asked and answered with a true A/B (release builds of HEAD vs
d65d1dd — the commit immediately before the fixes — same machine, runs
interleaved): did the root-cause-4 fixes cost performance?

| Benchmark | Mode | pre-fix (d65d1dd) | post-fix (HEAD) |
|---|---|---|---|
| arith | off / t=1 / t=1000 | 1253 / 6 / 9 ms | 1262 / 6 / 9 ms |
| dispatch | off / t=1 / t=1000 | 1844 / 26 / 11 ms | 1838 / 26 / 11 ms |
| sieve | off | 85 ms | 85 ms |
| sieve | t=1 / t=1000 | **SIGSEGV (exit 139)** | 84 / 85 ms, correct |
| richards | off / t=1 | 204 / DNU abort | 205 / 194 ms correct |
| deltablue | off / t=1 | 209 / 119 ms | 209 / 120 ms |

Identical to the millisecond on every benchmark that ran before — which
also directly measures the only hot-path cost the fixes added (the two
extra register-pair spills per runtime-stub call: no observable effect).
And the A/B surfaced something the record didn't yet know: the PRE-fix
build SIGSEGVs on sieve under BOTH JIT thresholds in release — the OSR
uninitialized-slot bug again, cured by the same fix. Net: nothing slower,
two benchmarks (sieve JIT, richards t=1) went from crashing to correct.

# Dual-arm branch storm — sieve is the canonical repro (2026-07-08)

Representative-benchmark sweep after the primitive-shim work (release,
`--world world`; richards/deltablue self-timed, fib/factorial self-timed
via `millisecondClock`, arith/dispatch process-timed incl. ~10 ms boot):

| Benchmark | interp (off) | jit t=1 | speedup | shape |
|---|---|---|---|---|
| arith (`sumTo: 5M`) | 1280 ms | ~10 ms | >100x | tight fused smi loop |
| fib(32) | 1637 ms | 15 ms | ~109x | recursion + dispatch + fused arith |
| factorial 20! x200k | 6989 ms | 890 ms | ~7.9x | smi `*` + overflow→LargeInteger |
| factorial 500! x300 | 72325 ms | 4386 ms | ~16.5x | bignum multiply + allocation |
| deltablue (10x10) | 213 ms | 113 ms | ~1.9x | constraint solver |
| dispatch | 1870 ms | ~20 ms | ~90x | send/IC dispatch |
| **sieve (8190 x10)** | **87 ms** | **88 ms** | **~1.0x** | **dual-arm branch storm** |

**CORRECTION (2026-07-08, via `MACVM_TRACE=deopt`/`MACVM_DBG_REEXEC`/
`MACVM_DBG_IR` — an earlier draft of this entry crowned sieve the "dual-arm"
repro; the debugger overturned that. Recorded honestly.):**

- **Sieve is NOT a balanced-branch storm.** Its `threshold=1` flatness (65
  deopts) is a *compile-cold* artifact: at `threshold=1` the method compiles
  before its loop body has run, so its send ICs are all `Empty`, and the
  compiler lowers `Empty`-IC sends to `Untaken → UncommonTrap` (dead-code
  speculation) — then the loop runs and they all trap. Across the threshold
  sweep sieve is flat at EVERY setting (94-96 ms) and at `threshold=2000`
  has only **30 deopts** — no storm. Its high-threshold flatness is
  compile-timing on a short (~95 ms) workload, not speculation. Not the
  repro we thought.

- **The real speculation storm is Richards**, and it is NOT in
  `processWork:` (weekend_work.md Gap 1's guess, also eyeballed) and NOT a
  balanced boolean branch. The debugger pins it to **`addInput:checkPriority:`
  bci=21** — a `GuardKlass { obj, expect } fail → UncommonTrap` (block1
  @bci8 → block10 @bci21 in the warm IR). It is a **receiver-klass guard on
  a mono-inlined send** (S14 step 4b/5): the compiler inlined a
  `priority`/`packetPending:` accessor betting one Task subclass, but
  Richards runs four (Idle/Worker/Handler/Device) through this method, so the
  guard fails ~half the calls. **160,555 of 160,674 deopts** are this one
  site, and it is **threshold-independent** (826k at t=1, 160k at t=2000 —
  the steady-state storm survives full warmup because the site is genuinely
  polymorphic, not cold).

- **Richards is ~2.4×, not 1.1×.** The "1.1×" on record was measured at
  `threshold=1` (the cold-compile worst case, 826k deopts). Warmed up
  realistically (`threshold=2000`): **off 207 ms vs 85 ms = 2.4×**, deopts
  160k. The benchmark harness's `threshold=1` convention systematically
  understates the JIT. (Threshold sweep: sieve 94/95/96/95 ms at
  off/1/100/2000; richards 207/191/91/85 ms.)

- **The fix** is still the "detect an over-deopting speculation site, then
  de-speculate" shape, but "de-speculate" here = **stop mono-inlining that
  send; dispatch it polymorphically** (a real send, or S14 step 6's existing
  `DominantWithSlowPath`), NOT "compile both branch arms." Gate on Richards
  `addInput:checkPriority:` (deopts at bci=21 → ~0). Open puzzle: it already
  recompiles 18× (nm 38/41/42) and still storms — the recompiler isn't
  switching this site to poly; that is the thing to fix.

## RESOLVED (2026-07-09, commit a2bfd8b): the IC stomp in activate_method

The "open puzzle" above cracked the case: the recompiler wasn't switching
the site to poly because the IC never STAYED poly. `interpreter::send::
activate_method`'s over-threshold path unconditionally rewrote the caller's
IC to Mono-compiled(current receiver klass) on every dispatch —
`ic_transition` would upgrade Mono(A)→Poly[A,B] and the very next
over-threshold dispatch stomped it back to Mono(B). The IC ping-ponged
between Mono states forever: `snapshot_profile`'s tag-only hash never
changed (8,501 "profile unchanged" declines, 0 recompiles in one warm run),
and each customized compile baked whichever klass was last stomped in as a
mono-inline KlassGuard whose fail-edge trap then fired on ~every other
call. Proven with the in-tree debugger: at every reexecution the receiver's
klass EQUALED the live IC guard (the interpreter never missed!) while the
baked pool word held a different klass — a Mono→Mono re-key that
`ic_transition` cannot produce; the only writer capable was the stomp.

Fix (one gated seed in `activate_method`): seed the IC only from Empty or
same-klass Mono; never downgrade Poly/Mega or re-key a different-klass
Mono. The preserved Poly tag lets the EXISTING recompile machinery re-lower
the send (DominantWithSlowPath / plain Call) — no new mechanism needed.

| Benchmark | interp (off) | jit t=1 | jit t=2000 | best/interp |
|---|---|---|---|---|
| richards (before) | 208 | 191 (826k deopts) | 85 (160,674 deopts) | 2.4x |
| **richards (after)** | 208 | **18** (30k deopts, 58 recompiles) | **13** (**2 deopts**, 1 recompile) | **16x** |
| deltablue (before) | 208 | 113 | — | 1.9x |
| **deltablue (after)** | 208 | — | **62** | **3.4x** |
| sieve | 88 | — | 93 (count 1899 correct) | ~1.0x (separate: threshold=1 cold-compile artifact + short workload) |

Correctness: Bench's own checkResult (error: on mismatch) passed on every
run; full test suite green (19 binaries); stress matrix over 4,609 world
tests × {GC_STRESS=1, GC_STRESS=full:64, DEOPT_STRESS=64} × threshold=1 —
0 failures. The S15 T5 gate ("richards ≥ 5.0") is now PASSED at 16x.

## 2026-07-09 (S24 A1, commits 979daf0..db378fa) — compiled closures land

First slice of closure compilation: standalone block bodies compile
(`by_block` registry, closure calling convention, compiled-block NLR
origination, root-is_block deopt materializer). Numbers via
`scripts/perf.sh --release` (t=1/t=1000 columns) on the A1 code:

| benchmark | interp (ms) | jit t=1 | jit t=1000 | best/interp |
|---|---|---|---|---|
| richards | 218 | 19 | 13 | 16.8x |
| deltablue | 229 | 43 | 55 | **5.3x** |

- **DeltaBlue gate (>=5.0x) PASSED already at A1** — the design expected
  this to need A2/A3. Block-iteration backbone (do:/detect:-style bodies)
  no longer interprets. Was 3.4x at the IC-stomp fix (a2bfd8b).
- richards 16.8x: the >=16x no-regression gate holds (blocks there are
  the S14-ELIDED kind; A1 must not and did not perturb the splices).
- arith 1447 -> 11ms (131x), fib/sieve unchanged — gate 3 noise band.
- Interpreted tail (MACVM_TRACE=count, deltablue warm t=2000 vs off):
  14,316,872 / 102,901,203 = **0.139** (pre-A1 doc baseline 0.163;
  gate 2 target <0.10 is A3's job). The NEW per-method attribution
  (`bytecodes-by-method:` lines) shows the residue is entirely the
  A3-target creator methods (constraintsConsuming:do:, makePlan:, ...)
  — ZERO `[block]` entries in the top 40: A1 did exactly its share.
- Correctness: world suite byte-identical vs interpreter at t=1 AND
  t=200, plain and under {GC_STRESS=1, GC_STRESS=full:64,
  DEOPT_STRESS=64}; deltablue+richards under the same matrix at t=200
  all green. Two real bugs found by the benchmark half of the matrix:
  the PIC duplicate-klass GC corruption (FIXED, db378fa — also the true
  cause of the "cache exhaustion" abort) and the t=1-only stale-slot
  (task #125, pre-existing, full dossier in tests/repros/README.md
  entry 9).
- Measurement policy note: threshold=1 stays the DIFFERENTIAL oracle
  (compile everything, compare bytes); it is NOT a perf configuration —
  cold compiles get no feedback-driven code. Stress and perf runs now
  also gate on threshold=200 (metric-driven compiles), release builds,
  all modes in parallel.

## 2026-07-09 (S24 A2, commit 0401df7) — direct value-family dispatch

Compiled `value`-family sends now tail-jump straight to the block nmethod
via a shared per-argc dispatch stub (`by_block` probe), replacing the c2i
adapter + nested interpreter activation. `scripts/perf.sh --release`:

| benchmark | interp (ms) | jit t=1 | jit t=1000 | best/interp |
|---|---|---|---|---|
| richards | 204 | 19 | 12 | 17.0x |
| deltablue | 213 | 34 | 41 | **6.3x** |

- **DeltaBlue 5.3x -> 6.3x** — the warm run dropped 43->34ms purely from
  the value-dispatch fast path. `MACVM_TRACE=stats` on deltablue t=200:
  `value_dispatch_hits=1741710 value_dispatch_fallbacks=1516` — 99.9% of
  `value:` sends in compiled methods tail-jump; the 1516 fallbacks are cold
  warmup before each block compiles. (My pre-implementation caution that A2
  might barely fire before A3 was wrong: deltablue's compiled constraint
  methods send `value:` heavily.)
- richards ~17x (noise vs A1's 16.8x — its blocks are the S14-elided kind,
  few standalone `value:` sites); arith/fib/sieve unchanged.
- Interpreted tail UNCHANGED from A1's 0.139: A2 changes how compiled
  `value:` sites dispatch, not which methods compile. A3 (compiling the
  closure-creating orchestrators) is what closes the tail toward <10%.
- Correctness: world byte-identical vs interpreter at t=1 AND t=200, plain
  and under {GC_STRESS=1, DEOPT_STRESS=64}; benches x 3 stress modes x
  t=200 all green; cargo test 833/0.

## S24 A3 — compiled closure creation (A3a escaping non-ctx, A3b materialize Context)

2026-07-10, commits 96faa0a (A3a), fb01b7a (A3b), 70b5513 + 9de470b (review
remediation). Release, MacBook arm64, `world/bench/*` via `Bench run:`.

| benchmark | interp (ms) | jit t=200 | jit t=1000 | best/interp |
|---|---|---|---|---|
| richards | 204 | 20 | 13 | 15.7x (held) |
| deltablue | 214 | 33 | 33 | **6.5x** |

- **deltablue 6.3x -> 6.5x** and, far more important, the A3b tail methods
  (constraintsConsuming:do:, addConstraintsConsuming:to:, printOn:) now
  COMPILE — closure-creating methods with captured temps get a real
  materialized Context. T5 >=5.0 gate: PASSED.
- **The benchmark run was the detector for two release-observable bugs the
  world differential missed** (both fixed, see tests/repros/):
  1. `rt_alloc_slow` still enforced D7's Slots-only contract and allocated
     `nis` words ignoring the site size — a Closure/Context Alloc overflowing
     eden returned a too-short object and the continuation corrupted the
     neighbor (deltablue DNU #value: under real allocation pressure; every
     small repro stayed on the inline fast path). Latent since A3a.
  2. a has_ctx method whose FIRST bytecode is a loop header re-ran the
     block-0 prologue every iteration (per-iteration Context snapshots vs the
     interpreter's ONE shared Context — silent wrong answers). Now declined.
- Remaining deltablue tail at t=200: 8.9M interpreted bytecodes, led by
  ScaleConstraint>>execute (39.8%) + recalculate (13.0%) — the next
  eligibility targets (B-phase).
- Correctness: world byte-identical off vs t=200 (release), plain and under
  {GC_STRESS=1, GC_STRESS=full:64, DEOPT_STRESS=64}; loop-header +
  tiny-eden repros green under DEOPT_STRESS/GC_STRESS; 633 lib + full
  integration suites green.
- Process note: fb01b7a's post-commit code review (8 finder angles) filed 10
  findings; the two live ones were fixed same-day and the CtxLoc::None
  bci-fingerprint (three finders converged) was re-keyed to ctx-vreg
  liveness before it could bite under organic NotEntrant deopts.

## S24 B-phase L1 — stale PIC-c2i heal (e3a3f00)

2026-07-10. The B-phase understand pass (3-reader + adversarial-verify
workflow) found the deltablue tail was a DISPATCH FREEZE, not eligibility: PIC
pairs baked (klass -> c2i) before the callee compiled never upgrade. One lazy
re-key arm in rt_interpret_call's upgrade hook:

| benchmark | interp (ms) | jit t=200 | jit t=1000 | best/interp |
|---|---|---|---|---|
| richards | 204 | 12 | 12 | **17.0x** |
| deltablue | 214 | 11 | 11 | **19.5x** |

- deltablue **6.5x -> 19.5x** (33 -> 11ms); richards 20 -> 12ms at t=200 (its
  t=1000 record now holds at ALL realistic thresholds). c2i_pic_rekeys=19.
- Remaining deltablue tail (2.8M interpreted bytecodes): the sub-threshold
  DRIVER methods (projectionTest: 32%, chainTest: 24%, makePlan: 15% — all
  called ~103x < 200 and OSR-ineligible because they contain closures,
  driver.rs:750). L2 target: extend the OSR envelope to closure-bearing
  methods. B1-B4 (block-arg inlining wideners) mapped from the design doc
  thereafter.
- Correctness: world byte-identical off vs t=200 plain + all three stress
  modes; deltablue correct under DEOPT_STRESS (re-key x invalidation churn).

## S24 L2 steps 1-2 — counters bit fix + trigger unification (d609251)

2026-07-10. The L2 design pass (3-reader + 3-design panel + judge workflow)
CORRECTED the premise: 3 of deltablue's 4 tail drivers have zero closures and
already OSR-compile — the tail was sub-threshold CALL entry (calls never
consult by_key until the invocation counter crosses). Fix, user-decided
policy: "the loop counters have detected in a different way that the method
containing the loop is hot; the method is now hot" — a by_key install
saturates the invocation counter to the threshold, unifying the two profile
triggers with zero new dispatch state.

| benchmark | interp (ms) | jit t=200 | best/interp |
|---|---|---|---|
| richards | 204 | **6** | **34.0x** |
| deltablue | 214 | **7** | **30.6x** |

- deltablue **19.5x -> 30.6x**, richards **17x -> 34x** (its loop methods
  were also OSR-earned + call-starved). trigger_unifications=3 per bench.
- Also fixed en route (found by the design pass's read phase): the
  COUNTERS_COMPILE_DISABLED_BIT sat inside the S15 loop-counter field —
  loopy NoPermanent methods re-attempted compilation every 10k backedges
  forever (unguided re-compilation). Now bit 33; tripwire test pins it.
- Remaining deltablue tail (1.74M bc): pre-first-OSR warmup of the drivers
  (~55%) + AbstractConstraint>>inputsKnown: (14.5%, loop-free — the B1/B3
  block-arg-inlining flagship). Envelope steps 3-6 (OSR for closure-bearing
  methods via Context adoption) next per the design; measured by the new
  ctxloop.mst when built.
- Correctness: world byte-identical off vs t=200 plain + all three stress
  modes; benches correct under stress; new send-based integration test
  proves a sub-threshold call enters an OSR-earned nmethod (<50 dispatched
  bytecodes vs ~800 interpreted).

S24 arc summary to date: interp -> 6-7ms on both flagship benches
(A1 5.3x -> A2 6.3x -> A3 6.5x -> L1 19.5x -> L2 30.6x deltablue;
richards 16x -> 17x -> 34x).

## S24 L2 steps 3-5+7 — OSR closure envelope: phases A+B (51401dc, c0c51cd, d01a67a)

2026-07-10. The envelope proper, per osr_closure_design.md. Phase A: non-ctx
closure-bearing methods OSR (phantom-temp packing proven + T7/T9 tripwires).
Phase B: has_ctx materialize-form OSR via **Context ADOPTION** — one transfer
pair, zero codegen changes; identity is the soundness story (pre-OSR closures,
post-OSR AllocClosures, and deopt all share the ONE Context); the elided form
declines (osr_declined_elided_ctx, the R1 evidence counter). Step 5: OSR
compiles inherit the key's version (MAX_VERSIONS accumulates across re-arms).

| benchmark | interp (ms) | jit t=200 | best/interp |
|---|---|---|---|
| **ctxloop** (new) | 134 | **1** | **134x** |
| deltablue | 214 | 7 | 30.6x (held) |
| richards | 204 | 7 | ~30x (held) |

- ctxloop is the envelope's own gate: has_ctx + escaping accumulator closure
  + 100k-iteration loop, called ONCE — only OSR can tier it, and its
  checkResult is SELF-VERIFYING for adoption (per-iteration-snapshot
  semantics would answer wrong). One run composes the whole L2 pipeline:
  osr_entries=1, osr_ctx_adopted=1, trigger_unifications=1.
- Tests: adoption-identity flagship (plain + DEOPT_STRESS + GC_STRESS),
  phase-A shapes x stress, elided-decline pinned in-process, tripwires
  T1/T4/T6/T7/T8/T9. World byte-identical off vs {t=1,t=200} + stress.
- Remaining from the design: step 6's gate-s24-l2 justfile recipe + debug
  transfer-buffer verifier (hardening, deferred); B1/B3 (inputsKnown:) is
  the next deltablue-tail item.

## S24 B5 — multi-BB block splicing + B3 self-devirt (b587e8e … f4953c1)

2026-07-11. The payoff layer: conditional-NLR blocks (branch + `^` in one arm)
now splice at direct value sends AND at block-arg sites, and self-receiver
block-arg sends devirtualize per the customization klass so the flagship
`self inputsDo: [...]` shape (Poly IC) compiles.

| benchmark | interp (ms) | jit t=200 | best/interp |
|---|---|---|---|
| deltablue | 214 | **4** | **53.5x** |
| richards | 204 | 6-7 | ~30x (held) |
| ctxloop | 134 | 1 | 134x (held) |

- Arc: A1 5.3x → L1 19.5x → L2 30.6x → B5 step 5b 42.8x (5ms) → B3 53.5x (4ms).
- Step 5b closed a pre-existing deopt-materializer stale-read hazard (allocations
  reordered behind all dead-frame reads via a deferred fixup phase) and a
  recompile deopt-storm (snapshot_profile now recurses into grafted blocks).
- B3: AV::Slf narrow self-provenance in escape ≡ convert's receiver==self_vreg;
  the resolution mode rides in the map so both passes resolve one callee; a
  klass-sensitive NoPermanent maps to NoRetryLater (never poisons the shared
  method's method-wide compile_disabled bit). Guard-free splice, lookup-shaped
  inline dep on (rcvr_klass, selector).
- deltablue interpreted tail: inputsKnown: (14.6%) left the interpreted-tail
  list; makePlan:/chainTest:/projectionTest: (loop-driver bodies) now dominate.
- Gate every step: world byte-identical off vs {t=1,t=200} + GC_STRESS=1/full:64
  + DEOPT_STRESS=64; full lib+integration suites; clippy+fmt.

## S24 — OSR cold-send provenance: no Untaken→Trap under OSR (sieve 90ms→9ms)

2026-07-11. Debugger-driven (DBG_IR ladder) root-cause of the sieve deopt-thrash
found via the JIT-coverage census. The three hotness signals — invocation
counter, loop counter, OSR — OR together to TRIGGER a compile (trigger
unification, L2 step 2), but each certifies a different region as profiled:
the invocation signal the whole body, the loop/OSR signal only the loop's
executed part. The S14-step-3 `Untaken→Trap` speculation assumed whole-body
coverage but fired under loop/OSR provenance too, so a not-yet-reached send
(cold IC only because the OSR-entered loop hadn't exited yet) was trapped and
deopted the instant the loop exited — an endless OSR-compile→trap→interpret
cycle. Fix: `convert` gains `osr`; under OSR the four `decide→Trap` cold-send
sites emit a plain `CallSend` instead (`cold_send_traps() == !osr`). "Reliable,
not merely fast" (the Strongtalk lesson — fast-but-crashes is a demo).

| bench | before | after |
|---|---|---|
| **sieve** ms | ~90 | **9 (~10×)** |
| sieve deopts | 30 | **0** |
| sieve OSR entries | 30 (re-thrash) | **1** (stays compiled) |
| sieve JIT coverage | 4.5% | **97.1%** |
| deltablue / richards / ctxloop | 4 / 6 / 1 | unchanged |

- Whole class of "hot work lives inside loops, method called once" (OSR-only
  methods) goes from ~5% to ~97% compiled.
- Gate: 635 lib + 101 it_tier1 + all 16 integration suites; world differential
  IDENTICAL off vs t=200/t=1/GC_STRESS=1/GC_STRESS=full:64/DEOPT_STRESS=64;
  clippy+fmt clean. Adversarial review (3 lenses) SOUND — CallSend at an
  Untaken site is the always-correct general lowering (the trap was only a
  feedback-warming speculation); all 20 UncommonTrap sites classified, gate
  complete; the reverse perf-cliff (cold error paths as CallSends under OSR)
  negligible, hits no frame/spill limit.

## Multi-VM workers — ParallelMandel scalability (2026-07-15)

The multi-Smalltalk-worker capstone (`docs/multi-smalltalk-worker.md`, worlds
47/48): the live zooming Mandelbrot with every frame computed in parallel
bands by 4 worker VMs (each its own heap + tier-1 JIT on its own OS thread),
the primary VM only assembling bands (via `send:onReply:` continuations, MOP
deep-copy messages) and blitting complete frames.

- **~2.65 CPUs of sustained utilization with 4 workers** (measured on-screen,
  release GUI, Demos → "Mandelbrot — parallel workers") — visibly faster than
  the single-VM `MandelZoom` dive. The gap to 4.0 is the honest price of the
  model: the primary's band assembly + pickle/unpickle copies + the serial
  blit, plus band-boundary imbalance (interior-heavy bands finish last).
- Headless gate (`parallel_mandel_computes_a_full_frame_across_worker_vms`):
  full 320×240 frame with every band verified computed by its worker — 1.29 s
  in a debug build (workers at `Threshold(10)`; the JIT-off first cut needed
  ~8 s+ *per band*, the usual reminder that compute workers must warm their
  own JIT).

## Dynamic compiled-code coverage — arith/richards/deltablue/sieve (2026-07-19)

The README's "98.6–99.8% of executed bytecode-work runs as compiled native
code" headline, reproduced fresh (this exact figure previously existed only
as an unreferenced measurement, not written down here — this section closes
that gap). Methodology (drift-immune: exact dispatch counts, not wall clock,
so no A/B interleaving or throttling caveats apply): `MACVM_TRACE=count`'s
`bytecodes: N` total, `MACVM_JIT=off` (every bytecode interpreted — total
work) vs `MACVM_JIT=threshold=200` (compiled code never touches
`vm.bytecode_count`, so the printed total is exactly the *interpreted
remainder* — startup/warmup plus anything still running cold):

```sh
MACVM_JIT=off             MACVM_TRACE=count target/release/macvm run world/bench/<name>.mst --world world
MACVM_JIT=threshold=200   MACVM_TRACE=count target/release/macvm run world/bench/<name>.mst --world world
```

| bench | total (off) | interpreted remainder (t=200) | still interpreted | moved to compiled |
|---|---|---|---|---|
| arith | 75,008,560 | 158,007 | 0.21% | **99.79%** |
| richards | 137,183,981 | 262,796 | 0.19% | **99.81%** |
| deltablue | 102,900,884 | 1,482,697 | 1.44% | **98.56%** |
| sieve | 5,180,807 | 150,485 | 2.90% | **97.10%** |

Matches the earlier (2026-07-11, HEAD 31f86af) measurement of the same four
benchmarks to within rounding — the S24/OSR work since then hasn't regressed
it, and sieve holds at its post-fix 97.1% (the pre-fix figure was 4.5%; see
the OSR cold-send section above). Range: **98.6–99.8%** — the README's own
figure, now reproducible from this file alone.

---

## WINVM vs Cog (Pharo) — the Windows baseline

**The standing performance target for WINVM is: faster than Cog** — the
production Smalltalk JIT, a more meaningful yardstick for this VM than C.
Run `scripts/cog-bench.sh` (Pharo headless lives in `E:/cog`, outside the
repo; setup instructions in the script header).

First measurement, 2026-07-22 — i7-12700, Pharo 13.0 (Cog/Spur x64,
build 4c3e4714cc) vs WINVM @ d28baa4, `threshold=20`. Identical workload
bodies (`scripts/cog-bench.st` mirrors `BenchmarkDashboard`), identical
protocol: each timing covers 10 inner reps; cold first, then median of 6
warm; results checksum-verified on both VMs. Warm ms per 10 reps:

| bench | Cog | WINVM | WINVM/Cog |
|---|---|---|---|
| arith | 49-50 | 39 | **0.8 — faster** |
| dict | 16-17 | 13 | **0.8 — faster** |
| fib | 135-183 | 194-196 | 1.1-1.4 |
| sieve | <1 | ~~28-29~~ 3 | ~~>=30~~ **~3 (smi-speculation fix, same day)** |
| alloc | 17-34 | 138-141 | **4-8 — the other loss** |
| richards | 33 | 81 | 2.5 |
| deltablue | <1 | 7 | **>=7** |

Reading, with causes separated by confidence:

- **arith, dict: already faster than Cog.** The smi fast path and the
  customized-hash recompile fix (d920dd5) are doing their jobs.
- **alloc, 4-8x behind — cause VERIFIED.** `Association key: i value:
  last` compiles to a `CallSend` chain whose `basicNew` is a primitive
  shim: every allocation crosses into Rust (`rt_call_primitive`). Cog
  inlines allocation entirely in machine code. `Ir::Alloc`'s inline eden
  bump exists and works — the `basicNew`-send path just never reaches
  it. The fix is IR-level (lower a Mono `basicNew`/`new` send on a
  statically-known klass to `Ir::Alloc`) and would benefit the Mac too.
- **sieve — DIAGNOSED AND FIXED same day (28ms -> 3ms).** The
  hypothesis above was wrong: the array ops and most arithmetic DID
  inline. The real cause: OSR fired mid-first-call (fill loop's 8190
  backedges + ~1800 inner-sweep iterations crossed the 10k threshold)
  BEFORE the loop-tail sites `count := count + 1` and the outer
  increment had ever executed — their ICs were Empty, so they compiled
  as full CallSends inside the hottest loop (~18k sends/run), and being
  plain sends they never trapped, so no recompile ever healed them.
  Fix: Cog-parity speculation — an Empty IC whose selector's
  SmallInteger implementation is a SMI_INLINE primitive lowers to the
  guarded inline op anyway (`smi_special_target`). Wrong speculation
  costs one reexecute-trap + one recompile (the customized profile hash
  sees the warmed IC); all-smi sites never deopt at all. The remaining
  ~3x vs Cog is loop code quality (spill-all across the back-edge Poll),
  not send overhead.
- **fib, 1.1-1.4x behind:** send-heavy recursion; plausibly the same
  send-overhead story as sieve's loop overhead, at smaller magnitude.

Richards and DeltaBlue are baselined via `scripts/mst2st.py`, which
translates `world/41a` to Pharo chunk format on the fly (the .mst stays
the single source of truth; Pharo's own `Variable` system class is
renamed `DBVariable` in the translation). Both macro results are
checksum-asserted on both VMs. The macro story matches the micro one:
both are send/alloc-heavy, and the two residual losses — allocation
crossing into Rust, and spill-all send overhead — are exactly what they
exercise. fib reached PARITY (175 vs 181) after the smi-speculation fix.

Still not baselined: floats, anything block-heavy.


## 2026-07-22 — alloc: DIAGNOSED AND FIXED (145ms -> 52ms; 13ms with a Cog-sized nursery)

The verified cause above was right about the disease but wrong about the
site. The fused `Ir::Alloc` DID land in each constructor's own nmethod —
and changed nothing, because small constructors are SPLICED into their
callers, and neither the nonleaf nor the CFG splicer had an alloc arm:
the spliced body's `basicNew` demoted to a generic `CallSend`. On x64,
prim 23 has no shim, so that send linked to a c2i adapter — one full
interpreter round trip (~46ns) per object. Counter-instrumented run of
the alloc microbenches: **24,000,000 interpreted `basicNew` sends**,
deopt_count=0, total scavenge time 4ms. The entire gap was c2i.

Fix (dca37b9), two halves shared with the Mac:

- Customized `self basicNew` (gc_alloc_gap.md cost 1): a body whose
  receiver klass is a statically-proven metaclass recovers the sole
  instance from the globals namespace and lowers to `Ir::Alloc`, target
  resolved against the live world (no IC warmth), invalidation kept
  sound by a `(metaclass, selector)` inline dep.
- `alloc_site_klass_on`: the alloc gate generalized to a spliced
  callee's own IC table; both splicers now fuse in-body `basicNew`, and
  their `PushGlobal` arms track `const_class` like the root.

| bench | before | after | after + MACVM_EDEN=32768 | Cog |
|---|---|---|---|---|
| alloc (warm) | 137-145 | 52 | **13** | 16-17 |
| deltablue (warm) | 6-7 | 5 | 5 | <1 |

Callee-shaped microbench (`^Association basicNew` called 2M times):
89ms -> 5ms. The remaining default-eden gap is nursery geometry
(gc_alloc_gap.md cost 2, upstream's item): the 4MB eden forces ~122
scavenges/run; 32MB makes WINVM FASTER than Cog on its own alloc bench.
richards is unchanged (~90 warm) — its loss is not allocation.

## 2026-07-22 (later) — nursery default, special selectors, and the honest scoreboard

Three more changes toward "at least as fast as Cog everywhere":

1. **Default nursery 4 -> 32 MiB** (`layout::default_eden_for`, capped at a
   quarter of the reservation). alloc warm 53 -> 13 at defaults.
2. **Identity `==`/`~~` lowers to a raw pointer compare** (`Ir::RefCmpVal`,
   no guard — the frontend pins identity non-redefinable) and **boolean
   `not` lowers to a guarded flip** (`Ir::BoolNot`, reexecute-trap fail
   edge, canonical-body-verified, inline-dep-pinned). The richards send
   census showed ~130k `==` sends + ~90k `not` activations per run —
   selectors Cog never sends at all. richards warm 85 -> 41-44.
3. **F7**: entry-defined slots skip the prologue nil-fill (non-OSR).

Same-session head-to-head (machine drifts — Cog's own arith read 48, 116,
and 50 across three sessions today; only same-session pairs mean anything):

| bench | WINVM | Cog | |
|---|---|---|---|
| arith | 36 | 50 | WIN |
| dict | 22 | 15-33 | flapping both sides; quiet-state pair was 13 vs 16 (WIN) |
| alloc | 16 | 22 | WIN |
| fib | 204 | 167 | ~1.2x behind (noisy) |
| richards | 44 | 33 | ~1.3x behind (was 2.6x) |
| sieve | 3 | <1 | behind |
| deltablue | 5 | <1 | behind |

Remaining gap analysis: richards' residual is per-activation spill-all
(F3c — register-resident oops across safepoints with oop maps covering
registers — the one structural project left). sieve/deltablue absolute
numbers are near timer granularity; their residual is the same loop code
quality story. Everything cheaper than F3c is now implemented.

## 2026-07-22 (final) — pinned cores, honest clocks, and the real scoreboard

Two measurement bugs had been corrupting every comparison on this machine:
the P/E-core lottery (fixed: `MACVM_BENCH_CPU=perf` pins the VM to a
detected performance core at HIGH priority; cog-bench.sh pins BOTH VMs to
the same logical CPU), and Pharo's millisecond clock quantizing to the
15.6 ms Windows timer tick — **Cog's "sub-millisecond" sieve and deltablue
were artifacts**; measured with its microsecond clock they are 4 ms each.

Pinned, microsecond-clocked, same-session (warm ms per x10 reps):

| bench | WINVM | Cog | verdict |
|---|---|---|---|
| arith | 36 | 48 | **WIN 1.3x** |
| sieve | 3 | 4 | **WIN** |
| alloc | 14 | 18 | **WIN** |
| dict | 12 | 11 | parity |
| deltablue | 4 | 4 | parity |
| richards | 33 | 29 | ~1.15x behind |
| fib | 207 | 152-201 | ~1.2x behind (flappiest bench on both VMs) |

Day's arc: alloc was 8x behind, richards 2.6x, "sieve 3x" and
"deltablue 6x" — the last two never real. What remains is a ~1.15-1.2x
send-activation residual on the two deep-call benchmarks, which is F3c
(register-resident oops across safepoints) plus fib's pure call chain.

## 2026-07-24 — upstream sync features 1-3: richards inverts (27 vs 33)

The MACVM-side sync (docs/upstream_sync_2026-07-24.md) staged per feature,
benched before/after each (pinned, same-session pairs, warm ms per x10):

| bench | baseline | +alloc-group | +BoolNot | +F7-whitelist | Cog (same runs) |
|---|---|---|---|---|---|
| arith | 34 | 34 | 35 | 35 | 49-50 |
| dict | 13 | 12 | 12 | 12 | 16-17 |
| alloc | 15 | 14 | 14 | 14 | 17 |
| **richards** | **34** | **30** | **28** | **27** | **32-34** |
| sieve | 3 | 3 | 3 | 3 | <1* (real: 4, see 07-22) |
| deltablue | 4 | 4 | 4 | 4 | <1* (real: 4, see 07-22) |
| fib | 195 | 208 | 209 | 211 | 133-183 (flappiest on both) |

*Cog's harness on this side still quantizes to the ms tick (every value a
x1000 multiple); the 2026-07-22 microsecond-clock session measured its
sieve/deltablue at 4 ms each, i.e. parity with ours.

What the stages actually were, after review showed most of "feature 1" and
the eden work had already been replicated here (baseline alloc was already
15 ms — the historical 138 ms table predates 9cb272e-era work):

1. **Alloc-group residue** (5d79c27 docs, 86aec53 test re-arms, 8704792
   pooled-arg smi guard + eden clamp). richards 34 -> 30.
2. **BoolNot fires at last** (9cb272e findings 1+2, left in
   x64_codegen_perf.md by the MACVM port's verification pass): the
   canonical-flip check's dead `n1 >= len` conjunct dropped; the missing
   successors() trap edge added. First time any `not` site compiled
   inline. richards 30 -> 28.
3. **F7 entry-scan whitelist** (finding 3): correctness, not perf —
   richards 27 within noise of 28. Four-way suite green including
   deopt-stress.

Scoreboard after: **ahead on arith (1.4x), dict (1.4x), alloc (1.2x),
richards (1.2x)**; sieve/deltablue parity against Cog's real clock; fib
remains the flappy outlier (~1.2-1.4x behind, pure call-chain — the F3c /
frameless-x64 territory). The 07-22 session's "~1.15x richards residual"
is now inverted. Frameless emission (Mac F0-F3) stays arm64-only; its x64
port is the top remaining lever for fib and the named follow-up.

## 2026-07-24 — F6b: no-barrier propagation through Move (dart124 lessons item 8)

The Dart-1.24.3 extraction's cheapest item, audited first: F6 already
elides smi/old-const stores and the runtime barrier's three early-outs
bound what's left, so the remaining compile-time gap was values LAUNDERED
through `Move`s — merge shapes (richards' `destination:`) whose arms are
inlined constants. The no-barrier pass is now a monotone fixpoint with
`Move` propagating its source's verdict; a listing test pins both sides
(a Param-valued store keeps the card `shr`, Move-of-ConstSmi elides it).

Same-session pinned pair (threshold=20): richards 28 → 27 warm, all other
benches flat. Honest reading: noise-adjacent — the elided sequence was
already early-outing at runtime; taken because it is free, principled,
and richards has inverted on a millisecond before.

Gates: 735 lib tests; world differential off vs t=200 byte-identical
(5860 run, 0 failed) plain and under GC_STRESS=1 / GC_STRESS=full:64 /
DEOPT_STRESS=64; clippy clean on changed lines. **Measurement note for
the record: stress differentials must run the RELEASE binary — the debug
build under GC_STRESS=1 did not finish one pair in 40 minutes; release
runs each pair in ~3 s.**

Remaining item-8 slice (deferred, documented): fresh-`Alloc` receiver
elision for constructor init runs — fold into the F3c slow-path
restructuring, which touches the same Poll/alloc-slow sites.

## 2026-07-24 — PIC arm counts + count-proven dominance (dart124 items 2+3, slice 1)

The counts substrate: the poly pairs array gains a smi count tail
(`[k1,m1,…,c1..c4]`, layout.rs; SPEC §4.3 updated), bumped only by the
interpreter's row-7 hit — the unoptimized tier is the profiler, compiled
code never counts (Dart's cost model). `read_poly` returns cases
count-descending (stable vs first-seen); `snapshot_into` hashes poly
recursion in klass-raw order so count DRIFT cannot flap the profile hash;
reverification carries counts through compaction. `decide_with_budget`
retires the "first-seen, trusted only at len==2" pin: a dominant inlines
at ANY arity past an evidence floor (16 samples, 34% share), and an
under-sampled 2-case site now honestly declines. BoolNot's poly walk
fixed to the pairs region (`len()/2` would have read counts as klasses).

Bench pair (same session, t=20): FLAT — richards 26→27 (its band today),
fib 205→191 (its documented flappiness), rest identical. The instructive
negative: richards' hot poly sites (schedule-loop predicates) are
flat-BY-KLASS with a SHARED target — four Task klasses, one TaskState
method — so no arm clears 34% and by-klass dominance correctly declines.
The unlock for those sites is slice 2: duplicate-target dedup (one
spliced body behind a multi-klass guard chain) and/or CHA guard-free
devirt (lessons item 4), both of which key on the TARGET, not the klass.

Gates: 739 lib tests (4 new: count bump/reverify-carry, arity-3 dominant,
flat-4 declines, under-sampled declines); world differential off vs t=200
byte-identical plain + GC_STRESS=1 + full:64 + DEOPT_STRESS=64 (release);
it_tier1's poly/dominant tests pass with count-seeded evidence. Also this
session: it_tier1 COMPILES ON WINDOWS for the first time (is_osr fields,
native_sp x64 asm — 467bd12); the suite's first-ever x64 run dies at
c2i_adapter_dispatches_to_interpreted_method (FOREIGN pc 0, pre-existing;
tracked as its own porting task).

## 2026-07-24 — same-target poly splice + the dead-tail unlock: richards 27 → 22 (dart124 items 2+3, slice 2)

Slice 2 proper: a poly site whose arms ALL resolve to one method (richards'
schedule loop: four Task klasses, one TaskState/TCB implementation) now
splices the shared body ONCE behind a klass-MEMBERSHIP guard — new
`Ir::GuardKlassIn` (one klass load, hottest-first compare chain, both
emitters), decision `InlineDecision::SameTargetPoly` (no share floor —
flatness by klass is irrelevant when the target is unanimous; smi-seen
sites excluded), fail edge = the same rejoining real send, one
`(klass, selector)` dep per seen klass. `MACVM_TRACE=sametarget` prints
each decision.

**The trace immediately caught something bigger:** `#link` and `#identity`
declined with `spliceable=false blocks=2` — every mst-compiled method
carries the frontend's implicit `^self` tail as a DEAD trailing block, so
`try_inline_leaf`'s `blocks.len() == 1` precondition (and the dominant
path's dry-run copy of it) rejected every real-world accessor.
`DominantWithSlowPath` had NEVER fired outside hand-built tests. The
splice walker already breaks at the first Return, so the fix is the
predicate, not the walker: require entry-block-Return (trailing blocks
are unreachable by construction). This also lets every MONO accessor
inline take the cheap leaf splicer instead of the CFG machinery.

| bench | before (slice-1) | after | confirm runs |
|---|---|---|---|
| richards | 27 (26-28 band all day) | **22** | 22, 22, 24 |
| deltablue | 4 | 3-4 (band edge) | 4, 4 |
| arith | 34 | 33-35 | — |
| others | — | flat | — |

**First real perf movement of the dart124 arc — richards ~19%,** and vs
Cog's same-machine 32-34 the scoreboard now reads ~1.4-1.5× AHEAD on the
benchmark that was 2.6× behind on 2026-07-22.

Gates: 742 lib tests (3 new decision tests: flat-4 same-target inlines,
smi-case declines, under-sampled declines); it_tier1
`poly_same_target_inlines_membership_guard` end-to-end (4 seen siblings
via the membership fast path, the UNSEEN fifth via the rejoining send,
all interpreter-identical, 4 deps, 1 IC site); world differential off vs
t=200 byte-identical plain + GC_STRESS=1 + full:64 + DEOPT_STRESS=64
(release); dominant-path test still green.

Known residue, next lever (slice 3): the TaskState PREDICATES
(`isTaskHoldingOrWaiting` etc.) are genuinely multi-block leaves (fused
`or:`/`and:` branches) and still decline — same-target needs the CFG
splicer, not just the leaf splicer. richards' remaining gap to the
measured 4.6 ms/×10 ceiling (dart 1.24.3, RESULTS.md in dart_origins) is
that plus F3c.

## 2026-07-24 — same-target CFG graft: the predicates splice, richards 22 → 21 (dart124 items 2+3, slice 3)

The slice-2 residue, closed: `SameTargetPoly`'s decision gate widened from
`is_leaf` to `is_leaf || is_inline_eligible_cfg` (the Mono `Inline` arm's
own ladder), and the lowering gained a CFG leg — `GuardKlassIn` fronts a
guard-free `try_inline_cfg` graft; the graft's own continuation block is
claimed as a stub that moves the graft result into the shared `dst` and
jumps to OUR rejoin, so the fast (graft) and slow (real send) paths both
enter it with `dst` written. `MACVM_TRACE=sametarget` on richards now
reads `#isTaskHoldingOrWaiting arms=4 leg=cfg blocks=8` — the fused
or:/and:/not predicate grafts whole, its inner `not` fusing to `BoolNot`
against the callee's own warm ICs.

The e2e test's ORGANIC warm-up (round-robin interpreted probes, richards'
own access pattern) exposed a counting gap: the mono→poly upgrade and the
poly-append dispatches never counted themselves, and mono-era hits are
invisible — a sequential warm-up left the site at 12 samples, under the
16 floor. Rows 6 and 9 now seed the triggering arm's count at 1 (that
dispatch IS a hit); mono-era history stays honestly uncounted.

richards 21/22/21 (slice-2 same-session baseline 22/22/24); others flat.
**Day cumulative: richards 28 → 21 (−25%), vs Cog's 32-34 → ~1.55×
AHEAD.** Gates: 742 lib; it_tier1 poly suite ×5 green including the new
multi-block-predicate e2e (source-compiled or:/not/and: body, both branch
arms exercised through the graft, unseen fifth sibling through the
rejoining send, organic PIC counts); world differential off vs t=200
byte-identical plain + GC_STRESS=1 + full:64 + DEOPT_STRESS=64 (release).

Remaining richards decomposition: F3c (spill-all across safepoints — the
slow-path SaveLiveRegisters blueprint, lessons item 1) is now the
dominant residual on the road to the measured 4.6 ms/×10 cross-VM
ceiling.
