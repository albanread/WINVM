# MACVM Debugger Design — HALT and PROBE

**Status:** Design v2 implemented — DBG0-DBG3 complete and gated (crash
dossier; HALT + mixed-tier backtrace; a64 disassembler) and DBG5 PROBE live
mode complete (MACVM_TRACE=calls/oops, MACVM_STEP_CALLS); DBG4 (W-debugger)
deferred. See §6 for per-wave status. v2 changes: PROBE's walkback is a NEW
raw-fp-seeded walker (the vanilla walker provably panics at any async
mid-nmethod crash — §4.2); fault-hardening rules added (§4.5 — catch_unwind
does not catch SIGSEGV); the "static state only" claim corrected to the
x28-recovery design plus a dedicated non-destructive probe trampoline (§4.1);
sigaltstack added; PIC/mega/adapter range accessors and non-panicking
nearest-below pc lookups added to the build list (§1); a deopt/compile ring
buffer and a fatal-DNU mini-dossier added to DBG0's scope (§4.2, §6); stale
symbol references fixed (`resignal_fatal` → `restore_default_and_return`,
"R3 compile service" → the frontend's existing doit path).
**Prereqs read:** SPEC §5/§8.7/§16.3, APPS.md §3 (R1–R5) & §5.7, tracing_debugging.md,
`src/runtime/{frames,deopt,vm_state}.rs`, `src/codecache/{deopt_trap,nmethod,flush}.rs`,
`src/compiler/scopes.rs`, `src/interpreter/mod.rs`
**Fills in:** the deferred **R5 debugger** primitive group (APPS.md §3) and the
`W-debugger` sprint row (SPRINTS.md:293), plus a new low-level tier the plan
did not yet have.

---

## 0. One principle, two debuggers

MACVM runs the same Smalltalk method in two engines — tier-0 bytecode
interpreter and tier-1 compiled nmethod — and moves activations between them
through deopt. A debugger has to pick which representation it shows the user,
and the right answer differs by audience:

| | **HALT** (level 1) | **PROBE** (level 2) |
|---|---|---|
| For | Smalltalk programmers | MACVM/compiler developers |
| Debugs | *the program*, as bytecode + source | *the VM's own output*, as machine code |
| Unit shown | activation / bci / source range | nmethod / native pc / register |
| Strategy | **deoptimize to tier-0, step bytecode** | **freeze the physical frame, never deopt** |
| Trigger | high-level breakpoint, `halt`, DNU/`error:` | crash (`brk`, SIGSEGV/SIGBUS in code cache), low-level breakpoint |
| Trust model | trusts the VM (heap, walker, deopt) | trusts as little as possible — the VM is the suspect |

The split follows from one observation: **deopt destroys the evidence.** For a
user chasing their own logic bug, deopting to bytecode is exactly right — the
interpreter is the semantic reference, and stepping it is stepping the
language. For a VM developer chasing a miscompile (BUG D…), the compiled
frame's registers, spill slots, and instruction stream *are* the bug; the
moment you materialize interpreter frames you've replaced the crime scene with
the VM's own *story* about the crime scene — reconstructed by the very
machinery under suspicion. So HALT's first act is deopt, and PROBE's first
rule is *never* deopt.

Both run **inside the VM process** — no external debugger, no ptrace, no
second process. HALT is (eventually) Smalltalk-in-the-image over R5
primitives, Strongtalk-style; PROBE is Rust inside `macvm` itself, entered
from the signal path.

---

## 1. What already exists (the substrate)

The design deliberately adds **no new architecture** — every mechanism below
is in the tree today and gets reused, not rebuilt:

| Mechanism | Where | Debugger use |
|---|---|---|
| Frame walk, innermost-first, mixed-tier | `frames.rs:269` `walk_frames` → `FrameView::{Interpreted, Compiled{fp,ret_pc,nm}, Adapter, CallStub}` | both: the backtrace is this walk, verbatim |
| Frame layout (plain oop array, FP links, serials) | SPEC §5.1; `interpreter/stack.rs` | HALT: per-frame receiver/temps/operand-stack reads are the existing `Frame` accessors |
| Scope descs + PcDescs (per-safepoint virtual-frame chain, per-slot `ValueLoc`) | `scopes.rs:400,466,629` (`ScopeDescData`, `PcDesc`, `DecodedScope`), `nmethod.rs:292` `bci_at` | HALT: deopt input. PROBE: *read-only* map of where the compiler claims every value lives — diffable against reality |
| Full deopt: materialize interpreter frames, resume mid-method | `deopt.rs:264` `deoptimize_frame` → `DeoptResume`; `interpreter/mod.rs:296` `interpret_active`; `dispatch_from(method, bci)` enters at any bci | HALT: "debug this compiled activation" = exactly this path |
| Lazy deopt of mid-stack frames | `flush.rs:77,98` `make_not_entrant{,_lazy}`; `frames.rs:553` `redirect_returns_into_nm`; `vm_state.rs:661` `pending_deopts` | HALT: breakpointed method with live compiled activations → they convert on return, no stack surgery |
| Back-edge validity poll | SPEC §8.7; DeoptStats `Poll` reason | HALT: a breakpointed method spinning in a compiled loop still comes down at the next back edge |
| `brk #imm16` trap namespace + SIGTRAP handler + trampoline escape | `deopt_trap.rs:76,300` (`0xDE00` trap / `0xDE01` stress / `0xDE02` assert) | PROBE: the entry path; new immediates below |
| Per-method `compile_disabled` bit | `method.rs:147,154`; `driver.rs:462` | HALT: pin a method to tier-0 while it has breakpoints |
| Per-bytecode hook point in the dispatch loop | `interpreter/mod.rs:377-383` (trace/count checks already live there) | HALT: the breakpoint/step check goes in the same slot |
| Bytecode disassembler, pinned format | `bytecode/disasm.rs`, ISA.md | both: method view |
| Stack traces, compiled-frame aware | `error:`/DNU paths (tracing_debugging.md) | superseded by the walker-backed backtrace |
| Source registry (`Smalltalk methodSources`, A16) | SPEC §16.4 | HALT: source pane; needs a bci↔source-range map added (§5.7 APPS.md already calls for it) |
| Mirror architecture + GuiHost seam | SPEC §16.2/16.3, APPS.md §3 | HALT: the UI story |
| Heap verify, stack-write auditor, `MACVM_DBG_OOP` | `memory/verify.rs`, `stack.rs` (MACVM_DBG_STACK_WRITES), tracing_debugging.md | PROBE: forensic checks run at the frozen point |

What does **not** exist and must be built: a breakpoint store, a step machine,
the R5 primitives, SIGSEGV/SIGBUS handlers, a machine-code disassembler, a
bci↔source map, and the two command loops. The substrate audit added six more
items the original list missed, all small but all load-bearing for DBG0:

- **A raw-`(fp, pc)`-seeded native walker.** `walk_frames` starts from
  `tier_links.last()` + the anchor, and the anchor is CLEARED by
  `emit_stub_epilogue` on every stub exit — so at an async crash inside an
  nmethod's own code it panics immediately at `assert_ne!("no anchor is
  set")` (frames.rs:347). The loop PROBE needs exists as the private
  `Mode::NativeStep`; a public, validated-read variant seeded from the
  captured ucontext `__fp`/`__pc` must be added to frames.rs.
- **Non-panicking nearest-below pc lookups.** Every existing pc→metadata API
  is exact-match and panic-on-miss (`PcDesc::find_in`, `DeoptState::at`,
  `oopmap_at`), and a crash pc is generically *between* safepoints. The
  dossier does its own `partition_point` searches over the `pub` fields
  (`deopt_pcdescs`, `pcdescs`, `oopmaps`). (Beware: there are TWO `PcDesc`
  types — `codecache::nmethod`'s S12 oopmap descs and `compiler::scopes`'
  S13 deopt descs — and two `FrameView` types, the `runtime::frames` enum
  and the `runtime::deopt` struct.)
- **`contains_pc` accessors on PicTable/MegaTable/AdapterTable.** Their
  `CodeHandle`s live in private HashMaps with no iteration or containment
  API; branch veneers (`patch_branch26`) are recorded nowhere, so the
  verdict line needs an honest "in-cache, unnamed (possibly a veneer)"
  class.
- **A `sigaltstack`.** `SA_ONSTACK` is set on the SIGTRAP sigaction but no
  alternate stack is ever installed anywhere in src/ — a SIGSEGV from
  native-stack exhaustion cannot deliver to a handler without one.
- **An in-handler register capture buffer.** The SIGTRAP handler copies
  nothing today; the dossier's register view requires the whole
  `__ss` GPR file plus `__es.__far` (the fault address — currently read by
  nothing in the tree) copied to a static buffer before sigreturn.
- **A deopt/compile event ring buffer** (§4.2 step 9). The dossier shows the
  frozen instant; BUG-D-class triage needs the recent *history* — the deopt
  trace tail immediately before the failure is what localized nm 128. A
  small fixed ring fed from the existing `note_deopt`/compile/invalidate
  points, dumped as a dossier step.

---

## 2. Shared core: `vm.debug`

One new `VmState` field, `debug: DebugState`, owned by a new
`src/runtime/debug.rs`:

```rust
pub struct DebugState {
    /// Master switch — the ONLY thing the interpreter fast path reads.
    /// false ⇒ every per-bytecode debug check is one predictable branch.
    pub active: bool,
    /// (method identity-hash, bci) → Breakpoint. Identity hashes are stable
    /// across compaction (A16/S8 gate), so this map survives full GC with no
    /// remap pass — the same GC-safety trick methodSources uses.
    pub breakpoints: HashMap<(i64, u16), Breakpoint>,
    pub step: Option<StepPlan>,          // §3.4
    pub session_depth: u32,              // §3.6 reentrancy guard
    pub probe: ProbeConfig,              // §4
}
```

**Breakpoint identity is (method, bci), not a patched bytecode.** MACVM
already chose side tables over self-modifying bytecode for ICs (SPEC §4.3);
breakpoints make the same choice for the same reasons: methods stay pure,
golden disassembly stays byte-identical, and no W^X/flush question exists at
tier 0. Two rules the map must obey (v2): **`DebugState` holds no oops** — it
is not a GC root, which is exactly why the key is the identity hash and the
`Breakpoint` value carries no `MethodOop` (if that ever changes, `DebugState`
joins `roots.rs`'s one-list, not a private remap pass). And identity hashes
can collide: on a hash-lookup hit, confirm via the stored (holder-name,
selector-name) string pair before halting — a false halt on an astronomically
unlikely collision would be a confounding debugging experience precisely for
someone already debugging. Setting a breakpoint also validates `bci` against
the method's decode boundaries (a mid-instruction bci is a caller bug,
rejected at set time, not a silent never-hit).

The per-bytecode cost is gated behind `debug.active` plus a new method-flags
bit `METHOD_FLAGS_HAS_BP` (set on any method carrying at least one
breakpoint), so the hash lookup runs only inside methods that actually have
breakpoints. All checks — including the step check — fold under the ONE
master branch, so the claim below stays literally true:

```rust
// dispatch_from, alongside the existing trace checks (interpreter/mod.rs:378)
if vm.debug.active {
    if method.has_bp() {
        if let Some(bp) = vm.debug.hit(method, bci) { debug::on_breakpoint(vm, bp); }
    }
    if vm.debug.step.is_some() { debug::step_check(vm, method, bci); }
}
```

`debug.active` is false in every non-debugging run, so the entire feature
costs one untaken branch per bytecode — the same budget the `trace` checks
already spend. No opcode is consumed (the 0x44+ space stays free), and
`UPDATE_GOLDEN` never notices.

### 2.1 Setting a breakpoint = pinning the method to tier-0

`debug::set_breakpoint(vm, method, bci)`:

1. Insert into `debug.breakpoints`; set `METHOD_FLAGS_HAS_BP`.
2. `method.set_compile_disabled()` (`method.rs:154`) — the driver already
   honors this bit (`driver.rs:462`), so the method never compiles again
   while the breakpoint lives. Record whether the bit was previously clear so
   `clear_breakpoint` can restore it.
3. For every **existing** nmethod whose root or *inlined* scopes include this
   method: reuse the S13 REDEFINITION invalidation path (the
   `DependencyIndex` query + `make_not_entrant_lazy`, `flush.rs:77`) rather
   than a hand-rolled scope walk — it already finds inliners AND handles the
   super-send edge cases (S13 10d). Setting a breakpoint is, invalidation-
   wise, indistinguishable from redefining the method. Entry points get the
   deopt-stub patch; live activations convert through the **existing**
   return-path redirect / back-edge poll. No new deopt mode, no mid-stack
   frame surgery.
4. Reset the method's invocation counters so the tier-up bookkeeping doesn't
   fire pointlessly.

Clearing the last breakpoint reverses all of it (restore `compile_disabled`
only if we set it; recompilation then happens naturally by counters).

This is the whole "high-level breakpoint ⇒ debugging happens on bytecode"
mechanism: by the time execution *reaches* a breakpointed bci, it is
guaranteed to be running in `dispatch_from`, because every compiled body that
could have contained it was invalidated through the standard S13 machinery
and re-entry was blocked by `compile_disabled`.

### 2.2 The bci↔source map (new, compiler-retained)

APPS.md §5.7 already commits to this direction ("MACVM can do better: bci →
source range"). Concretely: the frontend's codegen (`frontend/codegen.rs`)
emits, per CompiledMethod, a packed `Vec<(bci: u16, start: u32, end: u32)>`
sorted by bci, stored in the same A16 world-side registry pattern —
`Smalltalk methodSourceMaps`, an IdentityDictionary parallel to
`methodSources`, gated by the same `MACVM_KEEP_SOURCE`. Lookup = binary
search for the greatest entry ≤ bci. This serves HALT's "highlight the
current range" and costs nothing when disabled.

---

## 3. HALT — the Smalltalk-level debugger

### 3.1 Entry points

All of them funnel into one function, `debug::halt(vm, reason)`:

- **Breakpoint hit** (§2's dispatch check).
- **`halt` / `halt:`** — new primitive on Object (one line in the primitive
  table): unconditionally calls `debug::halt`. This is the programmer's
  `debugger;` statement.
- **`error:` and the DNU fallback** — today these print a trace and kill the
  process (SPEC §6.3). When `debug.active`, they call `debug::halt` instead,
  turning every would-be-fatal guest error into an inspectable stop. (When
  exceptions land in the world later, unhandled-signal joins this list; the
  hook is already in the right place because DNU/`error:` are where SPEC
  routes all user-level failure.)
- **Step completion** (§3.4).

### 3.2 What a halt is

The VM is single-process today (Process is Phase E, SPEC §11), so "suspend
the debuggee" cannot mean suspending one green thread among many. Instead:
**a halt is a nested command loop at a bytecode boundary.** `debug::halt`
stops between bytecodes (dispatch-loop boundaries are GC-safe and
frame-consistent by construction — same reason the GuiHost seam runs doits
"between interpreter turns", SPEC §16.1) and services debugger commands until
told to resume. The Smalltalk program is exactly as paused as it is during a
GC pause, and for the same structural reason.

The command loop has two frontends, one protocol:

- **CLI v1**: a `(halt) ` REPL on stdin/stderr inside `debug::halt` —
  `bt`, `frame N`, `temps`, `step`, `next`, `finish`, `continue`, `disasm`,
  `print <expr>`. Ships first; needs nothing from Phase W/E. `print` runs a
  doit via the frontend's EXISTING doit-compilation path (the same one the
  REPL uses today) against the selected frame's receiver — R3 is the future
  name for that surface, not a prerequisite.
- **GUI/W-debugger**: when a `GuiHost` is attached, `debug::halt` instead
  emits a `DebugEvent::Halted` over the existing VM→GUI channel and services
  the existing GUI→VM request channel — mirror queries against the halted
  stack, resume/step commands back. The StackTraceInspector of APPS.md §5.7
  is this loop's client, unchanged in concept.

### 3.3 The backtrace and frame inspection (R5, made concrete)

R5's `processStack: limit:` = `walk_frames` + classification:

- `FrameView::Interpreted{fp}` → an `ActivationMirror` reading the SPEC §5.1
  slots directly: `method` (FP+0), `receiver` (FP+4), temps (FP+7…), operand
  stack, `saved_bci`, Context (FP+3). All existing `Frame` accessors —
  read *and*, for temps, write (assignment in the debugger is a slot store).
- `FrameView::Compiled{fp, ret_pc, nm}` → decode the scope chain at
  `bci_at(ret_pc)` (`nmethod.rs:292`, `scopes.rs:651`) and present **one
  ActivationMirror per virtual frame** — inlined activations appear as
  separate backtrace entries, receiver/temps read through `ValueLoc`
  (`deopt.rs:124` `read_value`, reused verbatim as a read-only inspector).
  **Read-only in v1**: mutating state inside a live compiled frame means
  either deopt-now-in-place or write-back-through-ValueLoc; v1 sidesteps by
  marking these frames read-only in the UI. The user who wants to mutate
  sets a breakpoint (⇒ the frame converts on return / next poll) and mutates
  in the interpreter frame. This avoids the one genuinely new piece of deopt
  machinery ("synchronous deopt of an arbitrary mid-stack frame") the
  current code doesn't have — deferred, not forgotten (§6).
- `Adapter`/`CallStub` frames are elided from the user-facing trace
  (APPS.md §5.7 already plans eliding block-value frames; same posture).

Per-activation R5 primitives (`activationReceiver:`, `activationTempAt:put:`,
`activationBci:`, `activationSourceRange:` via §2.2's map, …) are thin
wrappers over the above — "small, dumb VM primitives" exactly as §16.3 pins.

### 3.4 Stepping

A `StepPlan` armed by the command loop, checked in the dispatch hook:

```rust
pub struct StepPlan {
    pub kind: StepKind,          // Into | Over | Out
    pub base_fp: usize,          // frame the command was issued in
    pub base_serial: i64,        // that frame's FP+5 serial
}
```

- **Into**: halt at the next bci where `fp != base_fp` *or* bci changed —
  i.e. the very next bytecode anywhere; entering a send's callee halts at
  its bci 0. (Primitive-successful sends never enter the loop, so stepping
  *into* a primitive shows the send completing — correct: primitives are
  atomic at this level. That's a feature, not a limitation: the primitive
  *is* the implementation boundary HALT promises.)
- **Over**: halt at the next bci with `fp <= base_fp` (callee frames sit
  above; equality = back in the same frame; less-than = the frame returned
  early, e.g. NLR unwound it — the serial check distinguishes "same fp,
  different activation").
- **Out**: halt at the first bci with `fp < base_fp`, i.e. after this frame
  returns.

Frame serials (SPEC §5.1 FP+5, dead-home detection) make these checks robust
against fp reuse — the exact problem they already solve for closures. NLR and
`ensure:` need no special cases: unwinding runs through the ordinary
interpreter paths, and the fp/serial comparison naturally fires at whatever
frame execution surfaces in. If a step crosses into a *compiled* callee (the
callee wasn't breakpointed and is still hot), v1 steps **over** it (the
dispatch hook can't see inside; the C2I/I2C seams guarantee we regain
control at the return) — with `MACVM_JIT=off` or after §2.1 pinning, this
case vanishes. Honest limitation, documented in the UI ("stepped over
compiled callee #foo — set a breakpoint inside it to descend").

### 3.5 Resumption commands

- `continue` — clear `step`, return from `debug::halt`, dispatch proceeds.
- `step/next/finish` — arm `StepPlan`, continue.
- `return <expr>` — force the current frame to return a value: compile the
  doit (R3), then perform the interpreter's own return sequence (pop to
  receiver slot, overwrite with result — SPEC §5.1). The machinery is the
  ordinary `return_tos` path.
- `restart` — reset the current *interpreter* frame to bci 0: pop the
  operand stack to the frame floor, re-nil temps, resume at 0. (Only valid
  for frames whose method has no already-executed side effects the user
  can't reason about — which is the user's judgment call, as in every
  Smalltalk debugger.)
- **hot recompile-and-continue** (the classic Smalltalk fix-and-go) is *out
  of scope for v1* but the shape is visible: R3 compile + S3 install +
  `restart` of the affected frame. Deferred to the W-debugger wave (§6).

### 3.6 Meta-circularity and reentrancy

Strongtalk's debugger is Smalltalk code in the same image; MACVM's endgame
(Phase W + Phase E) is the same. Until Processes exist, the CLI loop is Rust,
but everything it does goes through the same R5 surface the Smalltalk
`ActivationMirror` library will use — the Rust loop is scaffolding around the
permanent primitives, not a parallel implementation to throw away.

Reentrancy rule v1: while `session_depth > 0`, breakpoint hits and `halt`
are **ignored** (the checks in §2 test `session_depth == 0`). Doits evaluated
*by* the debugger therefore can't recursively open debuggers on themselves.
When real Processes land, this becomes per-process, and "debug the debugger"
becomes possible the Smalltalk way.

---

## 4. PROBE — the compiled-code debugger

PROBE answers one question HALT structurally cannot: **"what did the compiler
actually emit, and what is the machine actually doing?"** Its audience is
whoever is chasing the next BUG D — where the corrupting write and the
crashing read are far apart, and the deopt/materialization machinery is
itself among the suspects.

### 4.1 Triggers

| Trigger | Path today | PROBE change |
|---|---|---|
| `brk #0xDE02` (compiled-code assert) | SIGTRAP handler → assert stub → `rt_compiled_assert_failed` panics (`deopt_trap.rs:685`) | handler additionally captures the register file; `rt_compiled_assert_failed` emits the dossier and exits 70 instead of panicking |
| SIGSEGV/SIGBUS with pc in a registered code-cache range | **no handler — raw crash** | new handlers, same escape pattern as SIGTRAP: capture regs+`__far` to a static buffer, rewrite `__pc` to a **dedicated probe trampoline**, all real work after sigreturn |
| `brk #0xDE10..0xDE1F` — **new: low-level breakpoint namespace** | (unallocated) | → enter PROBE interactively (DBG3) |
| `probe` CLI command / `MACVM_PROBE=break:<sel>[:bci]` | — | plant a low-level breakpoint (§4.3) |
| SIGSEGV/SIGBUS with pc *outside* any code cache | raw crash | one-line verdict via raw `write(2)` (async-signal-safe: no allocation, no formatting machinery, hand-rolled hex into a stack buffer), then restore `SIG_DFL` and return so re-execution dies with the original signal — a Rust bug stays rustc's/lldb's jurisdiction, but *the classification itself* is the first question of every crash triage |
| **fatal guest error** (DNU / `error:` without `debug.active`) | prints a 2-line trace, exits 1 | additionally emits the dossier's cheap steps (walkback, reg-block/tier-links state, ring buffer, heap verify) before exiting — no signal machinery involved; the interpreter state is coherent, so the vanilla walker works here. This is the mini-dossier the `deviceInAdd:` hunt needed |

The signal-side additions live INSIDE `deopt_trap.rs` (its charter: the
"single unsafe island for signal/ucontext work" — the hand-laid Darwin
ucontext mirrors are module-private and must not be duplicated). The handler
reads only the pre-registered atomic registry, never allocates or locks, and
escapes to Rust via a ucontext `__pc` rewrite. The captured register file
(all GPRs, sp, fp, lr, faulting pc, and for SEGV/BUS the fault address from
`__es.__far`) is copied to a static buffer before sigreturn — it *is* the
register view PROBE displays.

Three v2 corrections to the original design, all load-bearing:

- **A dedicated, non-destructive probe trampoline.** The existing uncommon
  trampoline is DESTRUCTIVE — it builds a walker-visible frame record and
  tears down the trapped frame on the way out. PROBE's trampoline must touch
  neither the trapped fp nor its frame: it only aligns sp defensively,
  marshals `x0 = x28` (&VmState) and the trap pc, and calls a `-> !` Rust
  entry. Registered as a fifth per-cache slot alongside the existing
  `(lo, hi, tramp, assert)` registry columns.
- **VmState recovery is via x28, and that is a *convention*, not a fact.**
  The dossier needs `CodeTable`/scopes/stubs — all VmState-resident; the
  static registry cannot carry them. x28 is trustworthy exactly when the
  faulting pc is inside a registered code range (the call stub establishes
  the invariant) — which is the same gate that decides "ours" vs "foreign."
  Foreign faults get the verdict-only path precisely because their x28 is
  arbitrary.
- **A reentrancy guard, because the dossier itself can fault.**
  `catch_unwind` does not catch SIGSEGV. If a second fault arrives while a
  dossier is in progress, the handler restores `SIG_DFL` and lets the
  process die — and the dossier is flushed per-step (unbuffered stderr
  writes), so whatever was produced before the recursive fault survives.
  Arming decision (v2): the handlers arm in `deopt_trap::install()` — i.e.
  for JIT-enabled runs. Interpreter-only runs have no code cache, so every
  fault there is foreign-by-definition and the verdict line is all PROBE
  could offer; revisit if that ever proves worth a standalone arming path.

### 4.2 The frozen-frame report (post-mortem mode)

On entry from a fatal trigger, PROBE emits a **crash dossier** before
anything else — so even a non-interactive CI run leaves the full evidence
(this is the "black box recorder"; BUG D's dossier was assembled by hand over
days, this automates hour one):

1. **Verdict line** — pc classified: which nmethod (`CodeTable::find_by_pc`,
   `nmethod.rs:409`), or PIC/stub/adapter (their emitters tag ranges), or
   foreign.
2. **Provenance chain** — pc → code offset → `PcDesc::find_in`
   (`scopes.rs:493`) → scope chain → `Holder>>selector @bci`, one line per
   inlined virtual frame. *"You are 3 instructions past the safepoint for
   `Point>>+` bci 7, inlined into `Rect>>area` bci 12, nmethod #23 v2."*
3. **Registers** — from the captured ucontext, each annotated: oop-looking?
   (tag bits) heap-plausible? (the stack-write auditor's mark-word check,
   `stack.rs`) in-heap / in-code-cache / on-stack classification per value.
4. **Compiler's claim vs reality** — decode the nearest safepoint's
   `SafepointState`: for every recorded `ValueLoc` (`scopes.rs:267`) print
   *where the compiler says* each receiver/temp/stack value lives and *what
   is actually there*, with the same plausibility annotation. The oopmap
   (`nmethod.rs:267`) gets the same treatment for GC-slot claims. One
   mismatched line here is a BUG-D-class root cause caught red-handed.
5. **Disassembly window** — ±16 instructions around pc, faulting instruction
   marked (§4.4).
6. **Walkback** — NOT the vanilla `walk_frames` (v2): at an async crash pc
   the anchor is always clear and the vanilla walker panics at step 0 ("no
   anchor is set"). PROBE walks from the captured `__fp`/`__pc` via the new
   raw-seeded native walker (§1's build list): classify each pc itself
   (nmethod / stub / PIC / mega / adapter / unnamed-in-cache), validate every
   fp against the thread-stack bounds BEFORE dereferencing (a fault here is
   not catchable — see §4.5 rule 4), translate deopt-redirected return
   addresses through `pending_deopts`, cap the step count. Each emitted frame
   line flushes before the next step, so a walk that dies mid-way still
   leaves its prefix — "walk died at step N" is a finding, not a failure.
7. **Heap spot-checks** — `verify.rs` heap verify under
   `catch_unwind(AssertUnwindSafe(..))` (the `rt_alloc_slow` test-hook
   pattern), same reported-not-fatal posture.
8. Machine-readable copy (`MACVM_PROBE_DUMP=<path>`, JSON — hand-rolled
   escaper, zero-dep house style; schema pinned with a `"schema": 1` field
   from day one) for regression harnesses; human copy to stderr. The human
   dossier opens with a distinctive marker line (`==== MACVM PROBE DOSSIER
   v1 ====`) because exit(70) is shared with heap/stack exhaustion — gate
   tests assert the marker, never the bare exit code.
9. **Recent history** — the deopt/compile ring buffer (§1 build list): the
   last N compile / deopt / invalidation events (nmethod id, bci, reason),
   newest last. The frozen instant tells you *where*; the ring tells you
   *what just happened* — on BUG D it was the deopt tail immediately before
   the failure that localized the culprit nmethod.

Then: if stdin is a tty and `MACVM_PROBE=interactive`, drop into the
`(probe) ` REPL — commands: `regs`, `dis [addr|sym]`, `x/<n>g <addr>`
(bounds-checked raw memory read), `frame`, `scopes`, `oopmap`, `walk`,
`verify`, `bt`, `resume` (only for non-fatal entries), `kill`. Otherwise
exit(70) after the dossier — CI stays non-hanging by default.

### 4.3 Low-level breakpoints (instruction patching)

The `0xDE10..0xDE1F` immediates are PROBE's planted breakpoints: overwrite
one instruction word in an nmethod with `brk #0xDE1n` (the encoder is
`deopt_trap.rs:94` `emit_brk`), remembering `(addr, original_word)` in
`ProbeConfig`. Patching uses the same W^X write path `flush.rs`'s §2b entry
patching already uses (this exact patch — one word, icache flush — is proven
machinery).

Placement is by provenance, not raw address: `probe break Point>>+ @bci 7`
resolves selector → nmethod(s) → `PcDesc` for that bci → code offset. Hitting
one enters the interactive REPL with the full §4.2 context.

**Continuing past** a planted breakpoint: restore the original word, flush,
single-instruction re-arm is *not* attempted in v1 (no hardware single-step
without ptrace/Mach thread-state gymnastics). v1 semantics: a planted
breakpoint is **one-shot** (auto-removed on hit, `rearm` command re-plants it
manually). This is honest and sufficient for the "get me to the Nth entry of
this nmethod with registers visible" use case; loop-iteration-N debugging
composes it with a conditional (`break ... if x0==<val>`, checked in the
probe handler against the captured regs — re-plant + resume until the
condition holds, all inside PROBE).

**Watchpoints are explicitly out of scope** — no debug-register access from
userspace on macOS without a Mach exception server; revisit only if a real
investigation demands it (§6). The stack-write auditor already covers the
highest-value watch (oop plausibility at the write) at the choke point.

### 4.4 Machine-code disassembler

PROBE needs `dis`. Options considered:

1. **Vendor Capstone** (C) — heavy, new build dependency; the QBEJIT
   evaluation explicitly dropped it for the same reason.
2. **Hand-rolled decoder for the emitted subset** — the compiler emits a
   *closed* instruction vocabulary (every word comes from
   `compiler/assembler.rs` / `vendor/wfasm/a64/encode.rs`'s emit functions).
   Write the inverse of exactly that set (~100 patterns), in
   `src/compiler/disasm_a64.rs`, with a `.word 0x????????` fallback for
   anything unrecognized.

**Decision: (2).** The encoder is ours, closed, and unit-tested; its inverse
is a weekend of table-driven code, keeps the zero-C-deps property, and — the
real argument — a *fallback line in our own disassembler is itself a
finding* ("the compiler emitted a word its own vocabulary can't name" ≈ the
QBEJIT trap-stub bug, found exactly this way). Round-trip tests
(`encode(dis(w)) == w` over the emitters' corpus — `wfasm`'s `corpus_replay`
infrastructure is sitting right there) make it trustworthy fast.

**Pool resolution (shipped with the RUSTTCL `disasm-native` verb):** an
`ldr xN, +off` literal load is only half a fact — the verb also reads the
8-byte pool word at the resolved target and annotates it
(`; pool[0x60]=0x…fb9 =false`), tagging the well-known oops
(`=true`/`=false`/`=nil`). Reading a JIT method's constants WITH its code is
what let the task-#88 investigation compare "what the compiler baked" against
"what the runtime dispatched" in one screen.

### 4.4b PROBE on Windows / x86-64 (WINVM Phase 3)

Everything above describes the macOS/AArch64 original. The Windows port
keeps the design intact — classify, capture, escape to ordinary code —
but four mechanisms differ, and one design *decision* was revisited.

**Signals become a Vectored Exception Handler.** There is no `sigaction`,
no `ucontext`, no sigreturn. `AddVectoredExceptionHandler(first=1)`
installs `veh_trap_handler`, which inspects the kernel's `CONTEXT`,
rewrites `Rip` in place, and answers `EXCEPTION_CONTINUE_EXECUTION`. The
escape is the same shape as the `__pc` rewrite; the plumbing is simpler.
Anything PROBE does not claim returns `EXCEPTION_CONTINUE_SEARCH`, so
debuggers and the default handler still see it — which gives the macOS
layer's `SIG_DFL`-restore dance for free.

**PROBE claims six NT status codes, not two signals:**

| code | why |
|---|---|
| `ACCESS_VIOLATION` | the SIGSEGV analogue; `ExceptionInformation[1]` is the faulting address (the `__far` analogue) |
| `IN_PAGE_ERROR` | the SIGBUS analogue |
| `ILLEGAL_INSTRUCTION` | what a `ud2` placeholder raises — and, during a port, what executing *the wrong architecture's bytes* usually decays into |
| `PRIVILEGED_INSTRUCTION` | same family; a wild jump landing in data |
| `INTEGER_DIVIDE_BY_ZERO` | `idiv` with a zero divisor, which on x64 traps rather than producing a value |
| `STACK_OVERFLOW` | reportable *because* the probe trampoline switches to its own stack before doing anything |

`BREAKPOINT` (`int3`) stays on the deopt path, not PROBE's.

**Registers are named for the host.** The dossier prints
`rax rcx rdx rbx rsp rbp rsi rdi r8..r15`, not `x0..x28`. This is not
cosmetic: in the x64 capture order slot 4 is `RSP` and slot 5 is `RBP`,
so a dossier labelled `x4`/`x5` invites a reader with AArch64 habits to
mistake the stack pointer for an argument register. `lr` is omitted
entirely — x64 has no link register, and printing a permanently-zero
field implies a value that does not exist.

**The disassembler decision is reversed for x64.** §4.4 chose a
hand-rolled decoder over vendoring Capstone, and the reasoning was sound
for AArch64: the encoder is ours, closed, fixed-width, and *a fallback
line is itself a finding*. None of that survives the move to x86-64 —
the instruction set is neither fixed-width nor small, and a hand-rolled
partial decoder would produce confident nonsense rather than an honest
`.word` fallback. `iced-x86` was already vendored as a dev-dependency
for the encoder's own difftests, so `disasm_x64` uses it, and it is now
a REGULAR dependency: a dossier that only exists in test builds is no
use, because the crashes that matter happen in the shipped binary.

**Variable-length instructions change how a window is chosen.** On a
fixed-width ISA any 4-byte boundary is an instruction boundary, so a
window around a faulting pc is just `pc-64 .. pc+64`. On x86-64 there is
no way to tell from the bytes alone where an instruction starts;
decoding from `pc - 64` will in general land mid-instruction and produce
a listing that is plausible and entirely wrong — worse than none, in a
crash dump someone is reading under pressure. `disasm_x64::window`
therefore decodes FORWARD from the nmethod base, which is an instruction
boundary by construction, and selects the window by *counting decoded
instructions*. A pc that is not itself a boundary is reported as such
(`MID-INSTRUCTION`) rather than snapped to the containing instruction —
"control reached a non-boundary address" is a far sharper finding than
"it crashed near here".

**Two x64-specific listing conventions**, both in `disasm-native` and the
dossier:

- *Pool loads resolve through RIP-relative addressing*, not `ldr`-literal.
  The annotation is identical (`; pool[0xf8]=0x…  =false`); only the
  decode differs.
- *A deopt trap is `int3` plus a raw `imm16` that is **data***. A faithful
  disassembler renders those two bytes as whatever instruction they
  encode — `cc 00 de` prints as `int3` then `add dh,bl`, which is correct
  and useless. The listing knows the emitter's own trap convention, prints
  `int3 0xde00  ; deopt trap`, and SKIPS the immediate bytes.

**`pcdescs` is two different things, and the listing must say which.**
An `Nmethod`'s `pcdescs` concatenates block-start descs (one per basic
block, carrying `OopMap::empty()`, for the trace path only) with the
genuine safepoints. `disasm-native` labelled every one of them
`; safepoint`, which claims the GC may run there and that a live oop map
applies — the opposite of true for a block start. In `SmallInteger>>max:`
five of six were block starts.

`oopmap == 0` does NOT discriminate: `oopmap::intern` dedupes by content,
so a real safepoint whose live set happens to be empty also interns to
index 0. The genuine deopt sites are the ones carrying a scope —
`deopt_pcdescs`. The listing now prints `; deopt safepoint` for those and
`; block start bci=N` for the rest, which doubles as a cross-check of the
block_pcs pipeline: the bcis should match an `MACVM_DBG_IR` block dump
one for one.

This mislabel is INHERITED — the AArch64 verb had it too — so the fix
applies to both hosts.

**Not yet ported, and known-broken rather than merely untested.** §4.6's
live compiled-send auditor fails on x64 today —
`it_debugger::step_call_stops_at_compiled_send_and_inspects_without_
changing_result` is red, and was red before the debugging work in this
phase, so it is a genuine gap and not a regression. §4.3's planted
low-level breakpoints (`0xDE10–0xDE1F`) are in the same position: the
trap decode covers the immediate range, but nothing has driven it. §4.2's
frozen-frame report and the walkback DO work — the RBP chain walk is
structurally identical to the `x29` one and has been used against real
faults.

### 4.5 What PROBE deliberately does not do

- **No deopt, ever.** `resume` from a planted breakpoint continues the
  compiled frame untouched. If the *user* wants the interpreter view of the
  same halt point, that's HALT's job (plant a high-level breakpoint and
  re-run) — the two-level split is precisely so PROBE never has to.
- **No allocation before the dossier's step 6** — amended (v2): steps 1–5
  necessarily read VmState-resident metadata (CodeTable, scopes, pools —
  reached via x28, §4.1), but they allocate at most transient formatting
  strings and NEVER touch the Smalltalk heap allocator; heap-DEPENDENT reads
  (selector names, pool oops) are tag-checked and best-effort, with the raw
  numeric verdict (nmethod id/version/offsets) printed FIRST so a corrupt
  heap still yields the essential report.
- **No Smalltalk evaluation.** A doit runs the interpreter runs the heap —
  everything PROBE must not trust.
- **No unguarded dereference of captured state, ever** (v2 — the rule
  findings 6/7's `catch_unwind` posture does NOT give you): `catch_unwind`
  catches panics, not faults. Every raw pointer derived from the crash
  context (fp chains, frame slots, oop-looking register values) is
  range-validated first — heap-layout containment for oop candidates,
  thread-stack bounds for frame pointers, code-cache bounds for pcs — and
  the reentrancy guard (§4.1) is the backstop when validation itself is
  wrong: second fault → SIG_DFL → die, keeping the partial, per-step-flushed
  dossier.
- **`resume` is never offered after a caught walk/verify panic** — the
  `AssertUnwindSafe` wrappers are sound only because every post-catch path
  terminates the process.

### 4.6 PROBE live mode — the compiled-send auditor (DBG5)

Full design: `docs/compiled_send_auditor_design.md`. PROBE §4.1–4.5 above is
*post-mortem* — it fires on a crash. DBG5 makes the same never-deopt,
freeze-the-frame discipline available **live, at every compiled send boundary**,
closing the gap §6 deferred as "step-into compiled callees" — a deferral whose
premise (pin to tier-0, debug in HALT) *fails* for the one bug class it didn't
anticipate: GC/oopmap/deopt-metadata faults that **exist only in compiled
execution and vanish when you deopt** (BUG D, task #94, S24 A3b's materialized-
Context crash). HALT can only show a green interpreter run of those; PROBE
post-mortem shows a downstream symptom (the `scavenge_oop` SIGSEGV) whose cause
is already gone.

**Mechanism — force-cold ICs.** The one runtime choke point every non-hit
compiled send already funnels through is `rt_resolve_send`. In auditor mode it
computes+returns the correct target but **short-circuits before the poly/mega
machinery and never memoizes the site** (stays `Unresolved`), so the next send
re-enters — every compiled send is observable with zero codegen change, no
`brk`, no hardware single-step. Observation-only: the `Unresolved` arm is the
already-correct general dispatch, so results are **byte-identical** (the headline
gate). The short-circuit sits *ahead* of the poly arms deliberately — they
free/build PIC stubs eagerly while computing the patch, so a later gate would
dangle a freed `bl`; a `debug_assert` tripwire flags any future compile-time IC
seeding that changes the born state.

Three read-only modes over that choke point:

- **`MACVM_TRACE=calls`** — one `[calls] nm#site #sel recv=<klass> → <tier>` line
  per compiled send. The "which send is live right now?" log. (Scope: ordinary
  dynamic sends; super-hits and DNU are not traced. `ic_misses` inflates —
  stderr-only stat — so never combine with an IC-transition-asserting repro.)
- **`MACVM_TRACE=oops`** — at every GC, scans each live compiled frame's oop-map
  slots and flags any mem-tagged word pointing into no used heap region, **before
  the collector dereferences it**. Reuses PROBE's `annotate_value` region
  classifier; a cheap `oop_is_suspect` pre-filter keeps a healthy GC silent.
  Turns A3b's `exit=139` into `⚠SUSPECT nm=31 slot=4 word=0x1 →OUTSIDE-HEAP`.
- **`MACVM_STEP_CALLS=1`** — interactive: stop at each send boundary, service
  `bt | slots [N] | step | continue | quit`, never deopting. `slots` shares the
  collector's exact slot walk + the `oops` classifier. No `print`-eval (§4.5).

### 4.7 What PROBE live mode does not do

- **No `print`-eval** — evaluation runs the interpreter + heap, the very things
  PROBE distrusts (§4.5). Slot/register reads cover the debugging need.
- **No mutation** — read-only, like all of PROBE (§3.3's deferred mid-stack
  deopt is HALT's territory, not this).
- **Not exhaustive tracing** — a super-hit never enters `rt_resolve_send`, so it
  is invisible to `calls`/`step-call`; the log is "every ordinary dynamic send",
  not "every send".

---

## 5. Environment & CLI surface (additions)

| Surface | Meaning |
|---|---|
| `MACVM_PROBE` | `report` (default when a fatal trigger fires: dossier, exit 70) / `interactive` / `break:<Class>>><sel>[:bci][,…]` |
| `MACVM_PROBE_DUMP` | path for the JSON dossier |
| `MACVM_DEBUG` | `1` — set `debug.active` at boot (halt-on-error behavior, §3.1) |
| `macvm debug <file.mst>` | run with `debug.active`, CLI HALT loop on halt |
| `MACVM_TRACE=debug` | one line per breakpoint set/hit/step — the channel system is open-ended by design; this is just a new name |
| `brk` immediates | `0xDE10–0xDE1F` reserved for PROBE planted breakpoints (extends the tracing_debugging.md table) |
| `MACVM_PIN` | `Class>>sel[,…]` — pin methods to tier-0 WITHOUT a breakpoint (compile-disable + invalidate dependents). The differential-diagnosis lever: bisect a wrong answer by pinning suspects until it goes away. Also the RUSTTCL `pin`/`unpin` verbs |
| `MACVM_DBG_IR=<selector>` | (debug builds) dump every compile of that selector at TWO levels: the IR (blocks, instructions, literal pool with resolved values) AND the emitted listing (the assembler's per-instruction lines — the only view that shows spill stores). The layers between `describe` (bytecode) and `disasm-native` (published machine code) |
| `MACVM_DBG_RESOLVE=1` | (debug builds) one stderr line per compiled-IC miss resolution: receiver klass, prior site state, whether the target is a c2i adapter. Compiled dispatch is otherwise invisible — hits never leave compiled code, so this IS the complete transition log |
| `MACVM_DBG_REEXEC=1` | (debug builds) one stderr line per value the deopt materializer pushes for a reexecute site's recorded operand stack (`nm`, `bci`, `ValueLoc`, value, receiver). The runtime half of `MACVM_DBG_IR`'s compile-time story: compare what the scope RECORDED against what the frame slot actually HELD. Two entries printing the same address was the tell that closed task #94 (stale slot aliasing a recycled eden-base allocation) |
| `MACVM_TRACE=calls` | (DBG5 §4.6, release-capable) force-cold ICs + one `[calls] nm#site #sel recv=<klass> → <tier> argc=<n>` line per compiled send. The complete live compiled-dispatch log (superset of `MACVM_DBG_RESOLVE`'s misses-only). Byte-identical results; `ic_misses` inflates (stderr stat) so don't mix with IC-transition-asserting repros |
| `MACVM_TRACE=oops` | (DBG5 §4.6, release-capable) at every GC, flag any live compiled-frame oop-map slot holding a mem-tagged word in no used heap region — BEFORE the collector derefs it. Silent on healthy code; `⚠SUSPECT nm=… slot=… word=… →OUTSIDE-HEAP` on a wild/stale spill (localized S24 A3b in one line) |
| `MACVM_JIT=threshold=1` + RUSTTCL `nmethods` / `disasm-native` | the first two commands of any x64 codegen investigation. `nmethods` says what compiled and how big; `disasm-native` shows the machine code WITH its pool constants, IC sites and trap sites annotated. Both work on Windows/x64 as of Phase 3 (§4.4b) |
| `MACVM_STEP_CALLS=1` | (DBG5 §4.6) interactive step-call auditor: stop at each compiled send boundary, service `bt | slots [N] | step|s | continue|c | quit`, never deopting. Force-colds ICs like `calls`; independent of `MACVM_DEBUG` |

---

## 6. Build plan and deferrals

Ordered by value-per-effort, honestly small steps — each lands testable.
**Status (2026-07-06): DBG0, DBG1, DBG2 DONE and gated; DBG3 = disassembler
+ dossier/TCL integration in progress (interactive planted breakpoints
deferred — they need hardware single-step, Mach-exception territory, per
§4.3). All three shipped waves are exposed to RUSTTCL (`dbg`, `bp`,
`bp-clear`, `bp-list`, `ring`), and the fatal-error/DNU paths route to
both the PROBE mini-dossier and (when armed) a HALT stop.**

**Status (2026-07-10): DBG5 — PROBE live mode (§4.6) DONE (`MACVM_TRACE=calls`,
`MACVM_TRACE=oops`, `MACVM_STEP_CALLS`). This reverses the "step-into compiled
callees" deferral below with the case it did not anticipate — compiled+GC bugs
that vanish under tier-0 pinning — using the auditor choke point (force-cold
`rt_resolve_send`) rather than the hardware single-step the deferral assumed
was required. Validated by localizing the live S24 A3b GC crash to
`nm=31 slot=4 word=0x1` at threshold=200. Design: `compiled_send_auditor_design.md`.**

- **DBG0 — PROBE dossier** (no interactivity): SIGSEGV/SIGBUS handlers +
  sigaltstack + register-capture buffer + probe trampoline + crash report
  steps 1–4 + 6–9 (incl. the ring buffer and the raw-fp walker) + the
  fatal-DNU mini-dossier + the 0xDE02 reroute. *Highest value first: this is
  the tool BUG D needed.* No disassembler yet (step 5 says "disasm not
  built"). Gate: a `--selftest-probe-*` flag family (planted `brk 0xDE02`,
  planted in-cache SIGSEGV, foreign-crash verdict; assert the dossier
  marker + schema + section presence — never the bare exit code, which is
  shared with heap/stack exhaustion) — the existing selftest pattern. First
  REAL customer: the open `MACVM_GC_STRESS=1` + `threshold=1` repro failure
  (tests/repros/README.md) — DBG0 is validated against it before it's called
  done.
- **DBG1 — HALT core**: `DebugState`, dispatch hook, `set_breakpoint`
  pinning (§2.1), CLI loop, backtrace/inspection over interpreter frames,
  stepping. Gate: scripted debugger session goldens (commands in, transcript
  out — same golden discipline as disasm).
- **DBG2 — compiled-frame inspection**: read-only scope-chain
  ActivationMirrors for `FrameView::Compiled` (§3.3), R5 primitive layer,
  bci↔source map. Gate: breakpoint in caller, inspect hot compiled callee
  frames mid-stack, values match `MACVM_JIT=off` run byte-for-byte.
- **DBG3 — PROBE interactive**: a64 disassembler + REPL + planted
  breakpoints. Gate: corpus round-trip + scripted probe session.
- **DBG4 — W-debugger** (Phase W/E alignment): Smalltalk `ActivationMirror`
  library + StackTraceInspector over the same R5 primitives; GUI halt events
  over the GuiHost channel. This is SPRINTS.md:293's existing row, now with
  its VM floor fully specified.

**Deferred, with the reason on record:**
- *Synchronous mid-stack deopt* (mutate a live compiled frame's temps) —
  needs new frame-surgery machinery; read-only + convert-on-return covers
  the debugging need (§3.3).
- *Step-into compiled callees* — ~~vanishes under breakpoint pinning; not
  worth a per-call debug check in compiled code (§3.4)~~ **REVERSED by DBG5
  (§4.6, 2026-07-10):** the premise fails for compiled+GC bugs that vanish
  under pinning; DBG5 delivers it via the force-cold `rt_resolve_send` choke
  point (no per-call codegen check — the cost lives in the runtime resolve
  path, which force-cold makes total only in debug mode).
- *Hardware watchpoints / single-step* — Mach exception server territory
  (§4.3); the auditor choke point (now realized as DBG5) covers the known cases.
- *Fix-and-continue* — shape known (R3 + install + restart), belongs to the
  W-debugger wave (§3.5).
- *Remote debug protocol* (DAP etc.) — the GuiHost seam is the natural
  transport when wanted; nothing in this design blocks it.

---

## 7. Open questions

1. **Halt inside a primitive** — primitives are atomic to HALT (§3.4); is
   that acceptable for long-running primitives (future FFI calls)? Likely
   yes for v1; revisit with FFI.
2. **Breakpoints in blocks** — a CompiledBlock is its own MethodOop
   (`is_block`, method.rs:57), so (method, bci) keys work unchanged; the UI
   question (how a user names "the block on line 3") waits for the
   bci↔source map to answer it.
3. **`debug.active` and the differential gates** — the off-vs-compiled
   byte-identical gate must also hold under `MACVM_DEBUG=1` with no
   breakpoints set (debug mode must be a pure observer until a breakpoint
   exists). Add that third leg to the gate matrix when DBG1 lands.
4. **Dossier stability** — the JSON dossier will be asserted by tests;
   pin its schema version field from day one so goldens survive additions.
