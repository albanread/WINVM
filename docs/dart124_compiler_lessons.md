# What the Dart 1.24.3 optimizing compiler does that WINVM doesn't — a ported-lessons playbook

**Why this document.** 2026-07-24: the WINVM/Cog benchmark suite, ported
workload-for-workload to Dart 1.x and run on the last optionally-typed VM
(1.24.3 x64, Dec 2017) on this machine, same pinned-P-core protocol —
methodology and raw logs in `e:\dart_origins\bench\RESULTS.md`:

| bench | Dart 1.24.3 | WINVM | gap |
|---|---|---|---|
| arith | 9.6 | 35 | 3.6× |
| fib | 71 | 211 | 3.0× |
| sieve | 0.69 | 3 | 4.3× |
| dict | 1.00 | 12 | 12× |
| alloc | 7.4 | 14 | 1.9× |
| richards | 4.6 | 27 | 5.9× |
| deltablue | 0.65 | 4 | 6.2× |

Dart 1.x runs the *same semantics* WINVM does — Smi/Mint/Bigint tower
(`object.h:6486/6571/6616`), every call/field/operator a dynamic dispatch
with `noSuchMethod`, type feedback from ICs only (annotations ignored).
The gap is compiler depth, which means every item below is portable in
principle. Source: `e:\dart_origins\sdk-1.24.3\runtime\vm\` (all file:line
refs below are into that tree). Ranked by expected leverage against the
table above.

---

## 1. Registers stay live across safepoints — the F3c blueprint (richards, arith, fib)

**The finding that matters most: Dart 1.x never teaches the GC to read
register files.** The contract is inverted — any instruction whose slow
path can reach a safepoint *saves its own live registers first*:

- Each instruction carries a `LocationSummary` with a `RegisterSet` of
  live registers (`locs->live_registers()`).
- `FlowGraphCompiler::RecordSafepoint` (`flow_graph_compiler.cc:728-790`)
  builds the frame's GC bitmap as **spill area + appended live-register
  section**, one tagged/untagged bit per saved register
  (`bitmap->Set(bitmap->Length(), locs->live_registers()->IsTagged(reg))`).
- The out-of-line slow path (`SlowPathCode`, e.g.
  `CheckStackOverflowSlowPath`, `intermediate_language_x64.cc:2650`) calls
  `SaveLiveRegisters`/`RestoreLiveRegisters` (`flow_graph_compiler.cc:724,812`)
  around the runtime call — pushes exactly that register set into the
  bitmap-described area.

So the hot path — including the loop back-edge poll — is `cmp; jcc` to an
out-of-line stub, with **zero spills**. The GC only ever walks stack
slots; registers reach it pre-spilled by the rare path that needed it.

**WINVM today:** spill-all across safepoints; PERF.md 2026-07-22 names
per-activation spill-all as the richards residual, and F3c ("register-
resident oops across safepoints with oop maps covering registers") as the
one structural project left.

**Literal change:** F3c does *not* need register-aware stack walking.
Give each Ir op that can slow-path a live-register set from the register
allocator; emit `Poll`, alloc-slow, and guard-fail paths as out-of-line
stubs that push/pop that set into a bitmap-extension area of the existing
frame map. The GC keeps its stack-slot walker unchanged. This converts
F3c from a GC project into a codegen+metadata project — much smaller.

**Expect:** the single largest richards/arith/fib lever. Dart's 3-cycle
arith iteration is this plus item 6.

## 2. A budgeted, frequency-ranked, *recursive* inliner (everything; fib, deltablue first)

`flow_graph_inliner.cc:28-93` is the complete policy, numbers included:

| knob | value | meaning |
|---|---|---|
| `inlining_depth_threshold` | **6** | inline through 6 nested calls |
| `inlining_size_threshold` | 25 | always inline callees ≤25 IR instrs |
| `inline_getters_setters_smaller_than` | 10 | accessors always inlined |
| `inlining_callee_size_threshold` | 80 | never inline callees >80 |
| `inlining_callee_call_sites_threshold` | 1 | callees with ≤1 call inside: always |
| `inlining_constant_arguments_*` | 60/200 | bigger budget when an arg is constant |
| `inlining_hotness` | 10% | site must be ≥10% of the function's hottest site count |
| `inlining_recursion_depth_threshold` | **1** | self-recursive calls inline one level |
| `max_inlined_per_depth` | 500 | per-round cap |
| `deoptimization_counter_inlining_threshold` | 12 | stop inlining a callee that keeps deopting |

Sites are *ranked by real per-site call counts* (`CallSites`,
`flow_graph_inliner.cc:219-315` — `call->CallCount()` from the IC,
normalized to `ratio` against the hottest site) and processed as a
worklist, re-collecting call sites inside freshly inlined bodies until
depth/budget exhausts.

**WINVM today:** mono-inline of small bodies + constructor/block splicing;
7-arg-send creators decline; no recursion inlining; no depth iteration.

**Literal change:** turn the splicer into a worklist inliner with these
ten knobs (start with Dart's values verbatim), fed by per-arm counts in
the ICs (add a count word to Mono/Poly arms). `recursion_depth=1` alone
halves fib's real call count.

## 3. Polymorphic inlining — inline *all* the arms (richards, deltablue)

`PolymorphicInliner` (`flow_graph_inliner.cc:458,1417-1902`): for a poly
site, try to inline **every** feedback-seen target (`TryInliningPoly`),
share bodies when two class-ids hit the same method
(`CheckInlinedDuplicate`), then `BuildDecisionGraph` (line 1652) — a
class-id compare chain (contiguous cid *ranges* collapse to one range
check) dispatching into the inlined bodies, with a megamorphic-call
fallback arm for unseen classes.

**WINVM today:** `DominantWithSlowPath` — the dominant klass inlines, all
others take a real send. Richards runs four Task subclasses through
`addInput:`/`processWork:`; the 2026-07-08 storm analysis showed exactly
this site.

**Literal change:** extend `DominantWithSlowPath` to N guarded inlined
arms (N=4 covers Richards' task set) with the shared-body dedup and the
send fallback. Klass-id ranges: if sibling subclasses get adjacent klass
ids at world load, four guards collapse toward one unsigned range check —
worth a look at world genesis ordering.

## 4. CHA — drop the guard entirely when the hierarchy proves it (richards, deltablue)

`cha.cc`: if a selector has a single concrete implementation under the
receiver's class subtree, the optimizer devirtualizes/inlines **with no
receiver check at all**, registering a dependency; loading a class that
adds an override invalidates the code (lazy deopt for on-stack frames).

**WINVM today:** every inline carries a KlassGuard (the guard *is* the
deopt-storm surface the IC-stomp saga was about). The dependency plumbing
already exists — the customized-`basicNew` work registers
`(metaclass, selector)` inline deps with NotEntrant invalidation.

**Literal change:** at compile time, walk the live world's subclass tree
for the guarded klass; if the inlined method has no override below the
receiver's static-or-guarded type, emit the body guard-free and register
the existing dep keyed `(klass-subtree, selector)`. Smalltalk-legal:
`become:`/method (re)definition already funnels through the invalidation
path. Guards remaining after items 3+4 should be rare in richards.

## 5. Deopt *reasons* recorded into the IC, consulted on recompile (kills storm classes)

Every deopt stamps *why* into the call site's ICData:
`ICData::AddDeoptReason/HasDeoptReason` (`object.h:1950-1955`). The next
compile *reads* them — `jit_optimizer.cc:685-750`: "saw
`kDeoptBinarySmiOp` → compile the Mint path this time", "saw
`kDeoptBinaryMintOp` → don't try Mint unboxing again". Plus a
function-level budget: `FLAG_max_deoptimization_counter_threshold`
(`compiler.cc:65,231`) → `SetIsOptimizable(false)` (`object.cc:3113`).

**WINVM today:** profile-hash recompiles + hand-fixed storm sites (the
a2bfd8b IC-stomp fix, the OSR cold-send provenance fix). The *general*
mechanism — per-site reason memory steering the next compile — doesn't
exist; each storm has been a bespoke investigation.

**Literal change:** add a reason byte (bitset) to IC entries; every
reexecute-trap records its reason id; the compiler's speculation gates
(`smi_special_target`, Untaken→Trap, mono-inline, BoolNot) each check
"have I already deopted here for this reason" before re-speculating.
This is the principled version of the "detect an over-deopting site, then
de-speculate" shape PERF.md called for.

## 6. Range analysis: induction variables → checks deleted (sieve, dict, arith)

`flow_graph_range_analysis.cc`: pattern-matches simple induction
variables (line 52ff), computes symbolic ranges, then
`EliminateRedundantBoundsChecks` (line 32) removes `CheckArrayBound`s the
range proves (line 266ff), and range-proven smi arithmetic drops its
overflow deopts.

**WINVM today:** no range pass; sieve's inner loop pays a bounds check
per store and smi overflow checks per `+`; PERF.md's "loop code quality"
residual. Dart runs the identical sieve at 0.69 vs 3.

**Literal change:** a minimal version pays first: recognize the fused
`to:do:` induction pattern (WINVM already fuses the iteration), derive
`i ∈ [lo, hi]`, and delete in-range `Ir` bounds checks and overflow
checks on `i`-derived arithmetic inside the loop. Skip the symbolic
general case initially — the benchmarks' loops are all the simple
pattern.

## 7. Load forwarding + LICM (deltablue, richards)

`redundancy_elimination.cc`: alias-classed CSE of field/array loads and
hoisting of loop-invariant loads *and checks* out of loops. Inlining
(items 2-4) is what makes this pay: once accessors are inlined, repeated
`packet.link`/`walkStrength` reads become forwardable loads.

**WINVM change:** after the inliner lands, a basic same-alias-class
load-forwarding + loop-invariant hoist over the fused loop body. Do this
*after* items 2-4; before them there's little to forward.

## 8. Write-barrier elision (sieve, alloc, richards)

`intermediate_language.h:3695/3721/4104`: `StoreInstanceField` and
`StoreIndexed` carry `ShouldEmitStoreBarrier()`; the compiler elides the
barrier when `CanValueBeSmi()` is false-for-barrier-purposes (smi/bool/
null stores need no barrier) and for stores into objects proven
freshly-allocated since the last safepoint.

**WINVM check:** sieve stores `true`/`false` into an Array per iteration;
richards flips boolean ivars constantly. If those stores emit remembered-
set barriers today, this is a same-day win: immediates never need one.

## 9. Megamorphic sites: hash-cache stub, never the interpreter (dict)

`EmitMegamorphicInstanceCall` (`flow_graph_compiler.cc:1176,1938`): a
mega site compiles to a probe of a global (selector, cid) → entry hash
table in machine code — no interpreter round-trip. WINVM's c2i adapter at
cold/mega sites was exactly the alloc-gap disease. Give Mega ICs a
compiled probe stub as the terminal state.

## 10. Cheap and cheerful: edge-count block layout + cold code out of line

`block_scheduler.cc`: unoptimized code counts branch edges
(`edge_counters`); `FLAG_reorder_basic_blocks` lays out optimized code by
those counts — hot paths fall through, guard-fail/trap arms land at the
function tail. WINVM's interpreter profile could seed the same layout;
uncommon-trap arms should never sit inside a hot loop body.

---

## Tiering appendix

Dart 1.x: one usage counter per function, incremented on entry *and* loop
back-edge; optimize at `optimization_counter_threshold=30000`
(`flag_list.h:119`), OSR from the same counter. WINVM's t=200 with
trigger unification is philosophically the same machine — no change
urged; the constants differ because the counters count different events.

Not worth porting now: the background compiler thread (throughput
identical, only jank differs), allocation sinking + materialization
(deltablue's next 20%, big machinery), DBC. Frameless leaf entries
(`Intrinsifier`, entry-before-frame fast paths) become newly attractive
*after* item 1, and connect to the parked frameless-x64 work — same
territory as fib's remaining gap.

## Provenance

Benchmarks: `e:\dart_origins\bench\RESULTS.md` (protocol, wall-clock
cross-validation, checked-mode tax). Source browsed at the `1.24.3` tag
worktree `e:\dart_origins\sdk-1.24.3` (plus the 1.0.0.3 tree in
`e:\dart_origins\sdk` for the 2013 baseline — `cha.cc` and
`flow_graph_inliner.cc` are already present there). The lineage claim
"same semantics, deeper compiler" is measured, not asserted: same golden
check values on all seven workloads, on this machine, this week.
