# JIT Deopt-Storm Fixes

Status: implemented and committed. Two storm fixes — `4c575c8` (inlined cold
sends) and `b4d55a8` (numeric-fusion constant operands) — plus two supporting
fixes from the same investigation, `3e5d430` (threshold=1 guard) and `24c4a33`
(recompile snapshot follows grafts). This document is the authority for *what a
deopt storm is*, *the three classes found and fixed*, and *how to detect the
next one*.

## 1. What a deopt storm is

MACVM's tier-1 compiler is speculative (S14): it compiles a method against the
type feedback its inline caches have collected, guarding every speculation with
an **uncommon trap** (`brk #0xDE00`). A trap DEOPTIMIZES — it reconstructs the
interpreter frame(s), re-executes the trapping operation interpreted, and returns
the result to the compiled caller as if the method had returned normally. This is
correct by construction: a wrong guess never produces a wrong answer, it just
falls back to the interpreter.

Traps are meant to be **rare and self-correcting**. The recompile-on-trap loop
(`runtime::recompile::note_uncommon_trap`) is the corrector: after
`UNCOMMON_TRAP_LIMIT` (2) traps through one nmethod, it re-snapshots the method's
feedback profile; if the profile CHANGED (the trap warmed a cold IC), it
recompiles against the new truth and retires the old code; if the profile is
UNCHANGED, it declines ("profile unchanged") and re-arms the counter.

A **deopt storm** is the pathological case: a compiled method traps on
*essentially every call*, *forever*, because

1. the trapped operation is on the method's hot path (so it fires every call), **and**
2. the recompile loop cannot heal it — every re-snapshot reads the *same* profile
   and declines, so the method stays at v0 with the trap still compiled.

A storming method is **slower than interpreting** — every call pays the full
deopt-materialize-reinterpret-return cost on top of the compiled prologue — while
looking "compiled" in every coverage metric. The answer stays correct (the deopt
guarantees that), which is exactly why storms hide: nothing fails, it's just slow.

## 2. Detecting storms

Two tools, both added/used by this investigation:

**The named deopt trace.** `MACVM_TRACE=deopt` prints one line per trap. As of
`b4d55a8` each line also names the trapping method's `Klass>>selector`
(`runtime::deopt.rs`, behind the trace gate — zero cost when tracing is off):

```
[deopt] nm=NmethodId(80) #ScaleConstraint>>markInputs: pc=… bci=4 reexecute=true frames=2 result=false
```

**The scaling test — the decisive signal.** Warmup deopts happen once (a cold
method traps a few times, recompiles, settles) so their count is FIXED regardless
of workload size. A storm's count SCALES linearly with iteration count. Run a
workload at N and 2N iterations and diff per-method:

```sh
# aggregate deopts by method for a workload run R times
run() { printf '%d timesRepeat: [ BenchmarkDashboard %s ].\n' "$2" "$1" >/tmp/sw.mst
  MACVM_JIT=threshold=10 MACVM_TRACE=deopt ./target/release/macvm run /tmp/sw.mst --world world 2>&1 \
    | grep -cE 'nm=NmethodId'; }
run benchDeltaBlue 20   # e.g. 24
run benchDeltaBlue 40   # 24 (flat = warmup, healthy) vs ~2x (scaling = STORM)
```

**The full-suite sweep.** The widest exercise is the world test suite (6061
tests). It is not loaded by plain `--world`; build a combined list of
`world/world.list` then `world/tests/tests.list` (absolute paths — `load_world`
joins each line to the list's dir, and an absolute path overrides the join), and
loading the final `99_run_all.mst` runs the whole suite as top-level doits:

```sh
{ grep -vE '^#|^$' world/world.list       | sed "s|^|$PWD/world/|";
  grep -vE '^#|^$' world/tests/tests.list | sed "s|^|$PWD/world/tests/|"; } >/tmp/fw/world.list
printf 'nil.\n' >/tmp/noop.mst
MACVM_JIT=threshold=10 MACVM_TRACE=deopt ./target/release/macvm run /tmp/noop.mst --world /tmp/fw 2>&1 \
  | sed -nE 's/.*nm=NmethodId\([0-9]+\) (.+) pc=.*/\1/p' | sort | uniq -c | sort -rn
```

A method contributing hundreds/thousands of deopts to a single suite run, with a
matching count of `recompile declined (profile unchanged)` lines, is a storm.
(Note: extract the identity as `(.+) pc=`, not `[^ ]+` — metaclass methods like
`Foo class>>bar` contain a space and a non-space pattern silently drops them.)

## 3. Storm class A — inlined cold sends (`4c575c8`)

**Symptom.** `ScaleConstraint>>markInputs:` deopted ~102×/run steadily at every
realistic threshold (10/shipped, ≤100), declining recompilation 51×/run, stuck at
v0 forever. deltablue was 1.34× slower at the shipped threshold than it should be.

**Root cause.** `markInputs:` is `self inputsDo: [:in | in mark: mk]`. Compiled
customized for `ScaleConstraint`, it inlines `ScaleConstraint>>inputsDo:` (its own
override), whose body is `(direction == #forward) ifTrue: […] ifFalse: […]`. When
the customized `markInputs:` compiled, that inlined `==` site's IC was still
**Empty** (cold), so the compiler read `SiteFeedback::Untaken` and lowered the
whole `==`+`ifTrue:ifFalse:` to a single bare `UncommonTrap` — the S14 "don't
compile code that never runs" speculation. But the branch runs every call.

Why it can't heal: a ROOT send's trap re-executes in THIS method and warms THIS
method's own IC, so the recompile snapshot sees the change. An INLINED send's trap
re-executes in the CALLEE and warms the CALLEE's IC — which the caller's snapshot
cannot reliably observe, so it declines forever. `EqualityConstraint>>markInputs:`
escaped only by luck: it inlines the SHARED `BinaryConstraint>>inputsDo:`, kept
warm by every other binary constraint.

**Fix.** `compiler::ir::Translator::inlined_cold_send_traps()` returns `false` —
the inlined companion to the existing OSR guard `cold_send_traps() == !osr`. The
three inlined cold-send lowering sites (inlined leaf/nonleaf body, block-arg
in-callee graft, spliced block body) now emit a plain `CallSend` instead of a
trap; the ROOT site keeps its (healable) trap. Same "reliability over speculation"
principle. A genuinely-cold inlined path just compiles as an unreached dispatch —
a few bytes, zero runtime cost — instead of a trap that might storm.

**Result.** markInputs: deopts 102→0; deltablue total deopts (thr=20) 392→23 (all
transient warmup); declines 51→0; deltablue 58→43 ms at thr=10 (1.34×);
byte-identical result (224874).

## 4. Storm class B — numeric fusion on a wrong-type constant operand (`b4d55a8`)

A different mechanism: a numeric fuse fires on a compile-time constant of the
WRONG numeric type, emitting an operand guard that fails on every call. The
literal is invariant, so the recompile snapshot never changes → declines forever.
Two instances dominated the world-suite sweep (1284 total deopts before the fix).

### 4a. `aDouble < <smi-literal>` — the big one

**Symptom.** `Double>>truncated` (`^self < 0 ifTrue: [self ceiling] ifFalse:
[self floor]`) deopted **1198×/suite run — 93% of all suite deopts.** `Double>>abs`
shared the root.

**Root cause.** The float-compare fuse (`is_double_inlinable`, gated on a
mono-Double *receiver*, never checking the arg) lowers `self < 0` to an `FCmpBr`
that `FUnbox`es BOTH operands:

```
ConstSmi VReg2 <- 0
FUnbox   VReg3 <- self    (Double — ok)
FUnbox   VReg4 <- VReg2   (the SmallInteger 0 — a tagged int is NOT a boxed
                           Double, so this unbox ALWAYS fails → trap, every call)
```

**Fix.** In the double-fuse (`compiler::ir`, the `is_double_inlinable` arm): if
the arg operand is a compile-time smi constant (detected via `code.last()` being
its `ConstSmi`), materialize it as a float constant — `FConst { bits: (v as
f64).to_bits() }` — instead of the doomed unbox. `0` becomes `0.0`. This is
byte-identical to the interpreter fallback the deopt was already taking:
`Double>>< aNumber` is `<primitive: 104> ^self < aNumber asDouble`, i.e. the same
smi→f64 coercion (same precision loss for a >2^53 smi). The hot Double-vs-Double
path is untouched — its arg is not a `ConstSmi`, so it falls through to the unbox.

### 4b. `aSmallInteger * <Double-literal>` — the mirror

**Symptom.** `SmallInteger>>asFloat` (`^self * 1.0`) scaled ~1 deopt/call
(N=40→32, N=80→72). Smaller in absolute terms (`smi * Double-const` is a rarer
pattern than `float < 0`) but a real steady-state storm.

**Root cause.** The smi-arith fuse (`is_smi_inlinable`, gated on a mono-smi
*receiver*) lowers `self * 1.0` to a `SmiArith { op: Mul }` whose operand guard
requires both to be smis; the pooled `1.0` (a Double) fails it every call.

**Fix.** `compiler::ir::smi_fuse_arg_is_pooled` — decline the smi-fuse when the
arg is a **pooled** literal (a `ConstPool` load; smi literals are immediate
`ConstSmi`s, never pooled, so a pooled arg is always a non-smi that needs
coercion). The send falls through to a plain `CallSend`, whose shim runs
`SmallInteger>>*`'s own coercing bytecode fallback with no trap. The hot integer
path (`x + 1`, `x < n` — smi-const or smi-var args) is untouched.

**Result (4a + 4b).** Suite deopts 1284→72; non-healing declines 607→2; both
storms→0. A float-compare-heavy loop (300k `Double < smi` truncations) 738→92 ms
(**8.0×**); deltablue control unchanged (80→81 ms) = zero regression; world suite
6061/0.

## 5. Supporting fixes from the same investigation

**`3e5d430` — `MACVM_JIT=threshold=1` is refused.** threshold=1 compiles a method
on its first call, before any inlined callee has warmed, so it measures
cold-compile deopts rather than steady-state speed — it is never a valid
measurement or production config. `runtime::vm_state::parse_jit` now clamps `1` up
to `JIT_THRESHOLD_FLOOR` (20) with a warning; `2`+ are honored unchanged.

**`24c4a33` — the recompile snapshot follows grafted callees.**
`compiler::feedback::snapshot_into` recurses through a method's grafted callee ICs
(depth-capped, visited-set) so a trap that warms only a grafted callee's IC still
changes the caller's profile and can trigger a recompile. This is the general
machinery that lets *some* inlined storms heal; class A above is where it is
provably insufficient, which is why class A needed the compile-time fix in §3
rather than relying on healing.

## 6. Results summary

| storm | pattern | fix | before → after |
|---|---|---|---|
| `ScaleConstraint>>markInputs:` | inlined cold send → Untaken trap | de-speculate inlined cold sends | 102 → 0 deopts/run; deltablue 1.34× |
| `Double>>truncated` | `Double < <smi>` → FUnbox(smi) | smi const → `FConst` | 1198 → 0 deopts/suite; float-cmp 8× |
| `SmallInteger>>asFloat` | `smi * <Double>` → SmiArith(Double) | decline fuse on pooled arg | 32/loop → 0 |

Full world suite: 1284 → 72 deopts, 607 → 2 non-healing declines, 6061/0 tests.
Debug suite (non-GUI) 738/0; release suite 756 real tests pass. All results
byte-identical.

## 7. Finding the next one

Every storm in this document has the same fingerprint: **a compiled method whose
deopt count grows with the workload, and a matching stream of `recompile declined
(profile unchanged)`.** To sweep for more:

1. Run the workload (or the full suite, §2) under `MACVM_JIT=threshold=10
   MACVM_TRACE=deopt` at two iteration counts.
2. Aggregate deopts by `Klass>>selector`; any method whose count SCALES (not
   fixed) is a storm.
3. `MACVM_DBG_IR=<selector>` (debug build) dumps the IR + machine listing — look
   for a bare `UncommonTrap` where a real op should be (class A), or an
   `FUnbox`/`SmiArith` on a constant of the wrong type (class B).
4. The fix is almost always compile-time: emit the correct operation for the known
   constant/inlined case, keeping the hot path's codegen identical.
