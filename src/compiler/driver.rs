//! `eligible` (D1) + `compile_method` (D4) — the S10 compile driver: the
//! only place that decides whether a method compiles, and that drives the
//! decode -> convert -> regalloc -> emit -> publish -> install pipeline.
//! Neither function pushes a frame or touches the process stack — both are
//! callable directly, independent of the interpreter trigger (S10 step 8).

use crate::bytecode::opcode::{decode_at, Instr};
use crate::codecache::nmethod::{IcSite, NmState, Nmethod, NmethodId, OopMap, PcDesc};
use crate::codecache::stubs::resolve_super_target_entry;
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::compiler::{decode, emit, ir, oopmap, regalloc};
use crate::interpreter::ic::InterpreterIc;
use crate::interpreter::ic::{ic_state, IcState};
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::runtime::lookup::lookup;
use crate::runtime::vm_state::VmState;
use crate::runtime::JitMode;

/// `codecache::nmethod::IcState` — spelled out at every use site rather
/// than imported under its own name: `interpreter::ic::IcState` (the
/// UNRELATED interpreter-IC lattice, already imported above under that
/// exact name) would collide with it.
type CompiledIcState = crate::codecache::nmethod::IcState;

/// D1 point 2: `primitives.rs`'s own pinned ids for
/// `{ +, -, *, bitAnd:, bitOr:, bitXor:, <, <=, >, >=, =, ~= }` — division
/// (`//`, `\\`, `bitShift:`: ids 4, 5, 9) excluded in v1. `pub(crate)`: also
/// read directly by `ir::Translator::is_smi_inlinable` (S11 step 7) — both
/// readers must agree on exactly the same set, since `eligibility_detail`'s
/// own `mono_smi_inline_send` below is what decides a method compiles AT
/// ALL, while `is_smi_inlinable` (ir.rs) decides, per send site within an
/// already-eligible method, fused fast path vs. a real `CallSend`.
pub(crate) const SMI_INLINE: [i64; 12] = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15];

/// Float fast-path (`docs/float_fastpath_design.md`): the `Double`
/// primitives a mono-Double send site fuses to native FP code instead of a
/// `CallSend` (`ir.rs`'s `is_double_inlinable`/`classify_double_send`):
/// `+ - * /` (100–103) and `< =` (104/105). `sqrt`/`floor`/`truncated`
/// (106+) stay ordinary sends — unary, and served fine by the prim shim.
pub(crate) const DOUBLE_INLINE: [i64; 6] = [100, 101, 102, 103, 104, 105];

/// Primitive ids that must NEVER become a compiled entry via
/// `is_shimmable_primitive` below, regardless of `can_allocate`/`can_fail` —
/// every other primitive returns only `PrimResult::Ok`/`Fail`, a shape
/// `codecache::stubs::rt_call_primitive`'s generic mechanism handles
/// uniformly (Fail falls through to the method's own compiled bytecode
/// body; an allocating primitive is already GC-safe by construction, since
/// `stub_call_primitive` uses the exact same anchor+RootSpill machinery
/// every other Rust-calling stub relies on — `memory::roots::
/// real_oop_rootspill_slots`'s `AdapterKind::CallPrimitive` arm). These
/// don't: `PrimResult::Activated` means the primitive already pushed a
/// real interpreter frame and reassigned `vm.regs` itself (`runtime::
/// primitives::PrimResult`'s own doc) — the `value`/`value:`/`value:value:`/
/// `value:value:value:`/`valueWithArguments:` family (50-54) and
/// `ensure:`/`ifCurtailed:` (60-61). A generic "call it, check Ok-vs-Fail"
/// shim has no way to represent "control transferred to a brand new
/// activation" — that's a materially different, bespoke protocol, not a
/// smaller version of this one (`docs/next_architecture.md`'s own framing).
const PRIM_ACTIVATES_FRAME: [i64; 7] = [50, 51, 52, 53, 54, 60, 61];

/// Primitive ids that ALREADY have a bespoke, more-specialized send-site
/// fusion in `compiler::ir` and therefore must NEVER become an independently
/// compiled entry via `is_shimmable_primitive` — they are "already compiled"
/// in the precise sense the user asked us to skip. Every one of `ir`'s fusion
/// detectors (`is_smi_inlinable`/`array_op_kind`/`alloc_site_klass` and their
/// `_on` inline twins, plus `classify_smi_send`) reads a mono send site's
/// target via `MethodOop::try_from(ic.target())` and *relies* on that
/// succeeding — a plain, never-independently-compiled `MethodOop` is the only
/// thing a mono IC ever pointed at, historically, because no primitive-bearing
/// method could be compiled. Relaxing eligibility for these would let e.g.
/// `SmallInteger>>#+` acquire its own nmethod; the caller's IC target then
/// stops being a plain `MethodOop`, `try_from` returns `None`, and the fused
/// fast path silently downgrades to a generic `CallSend` (or, in
/// `classify_smi_send`, hits its hard `.expect`). The already-fused set is
/// [`SMI_INLINE`] (arithmetic/comparison on `SmallInteger`) plus `basicNew`
/// (23, `ir::PRIM_BASIC_NEW`, inline-allocated by `alloc_site_klass`) and the
/// Array element ops `at:` (26) / `at:put:` (27) (`array_op_kind`). NOT
/// included: `basicNew:` (24) is never fused (only bare `basicNew` is), so it
/// stays a legitimate generic shim.
const PRIM_ALREADY_FUSED: [i64; 15] = [
    1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, // SMI_INLINE
    23, // basicNew (alloc_site_klass)
    26, // Array>>at:  (array_op_kind)
    27, // Array>>at:put: (array_op_kind)
];

/// Primitive ids whose implementation reads `vm.prim_arg(i)` — i.e.
/// `vm.stack[vm.prim_arg_base + i]` — to return the *relocated* receiver
/// re-read from its rooted operand-stack slot after the collection they
/// trigger (the S12-step-7 moving-GC fix: a young receiver may move DURING
/// the primitive, so `args[0]`, a pre-GC copy, is stale). `gcScavenge` (93)
/// and `gcFull` (94) are the two such primitives. The compiled shim path
/// NEVER sets `vm.prim_arg_base` (only `interpreter::send::try_primitive`
/// does), so a shimmed 93/94 would return whatever stale index
/// `prim_arg_base` last held — an arbitrary oop from some paused interpreter
/// frame, not the receiver. They are the ONLY shimmable-by-the-other-rules
/// primitives that read `prim_arg_base`: `valueWithArguments:`/`ensure:`/
/// `ifCurtailed:` also touch the operand stack but are already excluded by
/// `PRIM_ACTIVATES_FRAME`. Kept interpreter-only.
const PRIM_READS_ARG_BASE: [i64; 2] = [93, 94];

/// Primitive ids whose implementation SYNCHRONOUSLY re-enters the
/// interpreter for a whole nested run (`run_method_reentrant`) as part of
/// answering: `perform:withArguments:` (64) and `primEvalDoit:` (250, the
/// Cocoa `Worker uiDoit:` relay's evaluator). They must stay
/// interpreter-only entries: every OTHER compiled→interpreter crossing in
/// the system goes through a runtime stub that brackets the nested run with
/// `TierLink::IntoInterpreter` (`rt_interpret_call`, `rt_value_fallback`,
/// `rt_dnu`, `rt_must_be_boolean`) so `walk_frames` can pair the nested
/// entry frame's `ENTRY_FRAME_SENTINEL` back out to the native chain — but
/// a `PrimFn` body cannot know which door it was called through
/// (`interpreter::send::try_primitive`, where pushing a link would be a
/// double-count of an already-tracked crossing, vs `rt_call_primitive`,
/// where NOT pushing leaves the crossing untracked), so it can bracket
/// correctly for neither. Excluding them from shims removes the second door
/// entirely: a compiled caller reaches them via the c2i adapter, whose
/// `rt_interpret_call` brackets correctly. Found in the wild as the Cocoa
/// GUI abort `walk_frames: an ENTRY_FRAME_SENTINEL must pair with
/// TierLink::IntoInterpreter, found IntoCompiled instead` — repeated
/// Benchmark-Chart/Mandelbrot relays warmed `Worker dispatchInbox`'s chain
/// past the JIT threshold, prim 250 started arriving via
/// `rt_call_primitive`, and the first GC inside a nested doit walked the
/// journal mispaired (regression-pinned by `cocoa_gui`'s
/// `many_sequential_uidoit_relays_do_not_corrupt_tier_links`). The shim
/// only ever saved the method-entry overhead — both primitives then
/// parse/compile/lookup and run whole nested Smalltalk, so the win was
/// noise; the c2i path costs the same order and is correct.
const PRIM_REENTERS_INTERPRETER: [i64; 2] = [64, 250];

/// `true` iff `prim_id` can become a compiled primitive-call shim
/// (`emit::emit`'s prologue, driven by `eligibility_detail` below). Five
/// exclusions: `PRIM_ACTIVATES_FRAME`'s list, `PRIM_ALREADY_FUSED`'s list,
/// `PRIM_READS_ARG_BASE`'s list, `PRIM_REENTERS_INTERPRETER`'s list, and
/// `crate::oops::layout::PRIM_ID_FFI` — FFI's own dispatch
/// (`runtime::ffi::dispatch_ffi_primitive`) has a variable-arity argument
/// shape the fixed colon-counted-argc `PrimDesc` table was never built to
/// represent (`interpreter::send::try_primitive`'s own doc on why FFI is
/// intercepted before the generic `prim_by_id` lookup even runs) — it is
/// its own, later piece of work, not a shimmable `PrimDesc` entry at all.
fn is_shimmable_primitive(prim_id: i64) -> bool {
    prim_id != crate::oops::layout::PRIM_ID_FFI
        && !PRIM_ACTIVATES_FRAME.contains(&prim_id)
        && !PRIM_ALREADY_FUSED.contains(&prim_id)
        && !PRIM_READS_ARG_BASE.contains(&prim_id)
        && !PRIM_REENTERS_INTERPRETER.contains(&prim_id)
}

/// D1 point 3, "tunable".
const FRAME_BUDGET_SLOTS: i32 = 60;
/// D1 point 4, "tunable".
const MAX_BYTECODE_LEN: usize = 2048;

/// D1's own linear scan, at a finer grain than the `bool` its text
/// describes: an ELIGIBLE method compiles now; a `NoPermanent` one never
/// will (a structural property of its bytecode/flags, or a send site
/// that's already resolved to something D1 doesn't cover) and gets
/// `compile_disabled`; a `NoRetryLater` one has EVERY check passing
/// except that some send site's IC is still `Empty` — the site simply
/// hasn't been reached yet, not "reached and rejected". This distinction
/// exists because `activate_method`'s trigger fires from OUTSIDE the
/// method, before its own body has ever run: on `MACVM_JIT=threshold=1`
/// specifically, the very first call is ALSO the trigger, so any method
/// with an inner send is *guaranteed* cold on that first attempt — an
/// early draft of this scan treated that as permanent and disabled every
/// such method forever after one failed attempt, which silently defeated
/// `threshold=1`'s own stated purpose ("every eligible method compiles on
/// first send", tests_s10.md gate item 1) for anything but the rare
/// method with zero internal sends. `NoRetryLater` leaves the counter
/// alone instead of disabling, so the method's *next* call — after this
/// one has actually run its body interpreted and warmed its own sites —
/// gets a fresh attempt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Eligibility {
    Yes,
    NoPermanent,
    NoRetryLater,
}

impl Eligibility {
    fn worse(self, other: Eligibility) -> Eligibility {
        use Eligibility::*;
        match (self, other) {
            (NoPermanent, _) | (_, NoPermanent) => NoPermanent,
            (NoRetryLater, _) | (_, NoRetryLater) => NoRetryLater,
            (Yes, Yes) => Yes,
        }
    }
}

/// D1: a single linear scan of `method`'s bytecode — ALL of opcode
/// allowlist, per-send IC shape, method flags, and frame-budget bound must
/// hold for `compile_method` to attempt compiling `method`. Thin `bool`
/// wrapper over [`eligibility_detail`] matching D1's own documented
/// signature (and what this module's existing tests call) — `compile_method`
/// itself calls `eligibility_detail` directly, since it needs the
/// permanent-vs-retry distinction `bool` collapses away.
pub fn eligible(vm: &VmState, method: MethodOop) -> bool {
    // No customization klass: the B3 self-devirt path stays off, matching
    // the klass-free question this wrapper asks (tests only).
    eligibility_detail(vm, method, None) == Eligibility::Yes
}

/// Most real arguments a compiled SEND SITE can marshal — the argument
/// registers minus one for the receiver.
///
/// This is an ISA capacity, not a policy: AArch64 marshals into `x0..x7`
/// (8 registers, so 7 real args), Win64 into `RCX RDX R8 R9` (4, so 3).
/// Both hosts now cap at `ROOTSPILL_SLOTS - 1` real arguments, because
/// the RootSpill — not the register file — is the real limit: it is what
/// the GC scans for a stub frame. Win64's four argument registers are no
/// longer the constraint, since arguments past the fourth travel in the
/// caller's outgoing stack area (`assembler_x64::OUTGOING_ARG_BYTES`).
#[cfg(target_arch = "aarch64")]
const MAX_SEND_ARGC: u8 = 7;
#[cfg(not(target_arch = "aarch64"))]
const MAX_SEND_ARGC: u8 = 7;

/// Most arguments a compiled METHOD's own entry convention can accept,
/// same register budget seen from the callee side.
#[cfg(target_arch = "aarch64")]
const MAX_METHOD_ARGC: usize = 5;
#[cfg(not(target_arch = "aarch64"))]
const MAX_METHOD_ARGC: usize = 5;

/// D1 point 2 (mono-smi-inline gate): a `Send` site only clears eligibility
/// when its own IC is already `Mono`, guarded on `SmallInteger`, targeting a
/// method whose primitive is in [`SMI_INLINE`] — this is deliberately
/// NARROWER than what a compiled site can actually reach at runtime (a
/// generic, non-smi, or polymorphic send): S11 step 7 experimented with
/// widening this to "arbitrary send in ANY IC state" (D1's own text) and
/// found the C2I/reentrant-call machinery underneath (S11 step 4/5) isn't
/// robust enough yet for the much larger surface that unlocks —
/// specifically, deep block-activation nesting reached through a compiled
/// caller's own generic `CallSend` (see the fixes alongside this one:
/// `emit::Emitter::emit_call_send`'s spill/register hazard,
/// `interpreter::send::try_primitive`, `run_method_reentrant`'s `vm.regs`
/// snapshot, and `run_method`'s entry-sentinel spoof around an `Activated`
/// primitive). Those four fixes are real, verified bugs in their own right
/// and stay; this gate stays at its ORIGINAL, conservative shape until a
/// later step gives C2I reentrancy the depth of testing D1's fuller text
/// deserves — `docs/sprints/sprint_s11_detail.md` has a SPEC-QUESTION on
/// this gap.
fn eligibility_detail(
    vm: &VmState,
    method: MethodOop,
    rcvr_klass: Option<KlassOop>,
) -> Eligibility {
    // S14 step 7-II-b: `has_ctx` (M owns a heap Context because a nested block
    // captures its temps) is NO LONGER an outright reject — the escape pre-pass
    // below gates it: a has_ctx method always creates a capturing closure, so it
    // compiles ONLY if that closure is elidable (all_elidable), which promotes
    // ALL of M's ctx-temps to vregs and elides M's Context. A `has_ctx` method
    // whose block escapes still returns NoPermanent (via the pre-pass).
    if method.is_block()
        || method.argc() > MAX_METHOD_ARGC
        || (method.primitive() != 0 && !is_shimmable_primitive(method.primitive()))
        || method.bytecode_len() > MAX_BYTECODE_LEN
    {
        if vm.options.trace.is_enabled("jit") {
            eprintln!(
                "[jit] NoPermanent reason: block={} argc={} prim={} shimmable={} bc_len={}",
                method.is_block(),
                method.argc(),
                method.primitive(),
                method.primitive() == 0 || is_shimmable_primitive(method.primitive()),
                method.bytecode_len()
            );
        }
        return Eligibility::NoPermanent;
    }

    // S15, BUG D root cause 4's deeper half (tests/repros/README.md): a
    // compiled SEND site marshals receiver+args into x0..x7 — 8 registers
    // (`emit_call_send`), matching `ROOTSPILL_SLOTS`' 8-slot spill area in
    // every Rust-reaching stub. A send with more than 7 real args has no
    // register home for the tail; the method stays interpreted (the
    // interpreter's stack-based sends handle any arity). This caps SITES,
    // complementing the `method.argc() > 5` cap above on the method's OWN
    // entry convention. Found by Richards' 7-arg task initializer: before
    // the RootSpill widening, its c2i marshaling read args 6-7 from the
    // stub frame's saved fp/lr — silent heap corruption surfacing as a DNU
    // on a "SmallInteger" thousands of sends later.
    {
        let mut b = 0usize;
        while b < method.bytecode_len() {
            let (instr, next) = decode_at(method, b);
            if let Instr::Send { ic, .. } = instr {
                if crate::interpreter::ic::InterpreterIc::at(method, ic).argc() > MAX_SEND_ARGC {
                    if vm.options.trace.is_enabled("jit") {
                        eprintln!(
                            "[jit] NoPermanent reason: send site ic={ic} argc={} > 7 \
                             (register-marshaling cap)",
                            crate::interpreter::ic::InterpreterIc::at(method, ic).argc()
                        );
                    }
                    return Eligibility::NoPermanent;
                }
            }
            b = next;
        }
    }

    // S14 step 7-I: a method that CREATES a literal closure (`push_closure`) is
    // no longer rejected outright. Run the escape pre-pass ONCE (only when a
    // closure is present — the common closure-free method pays nothing) and let
    // it decide: every closure site must be provably elidable (immediately
    // invoked via a matching `value`-send, non-escaping, spliceable) or the
    // whole method stays interpreted (`NoPermanent`). An escaping closure a
    // compiled frame cannot represent is the exact soundness boundary (SPEC
    // §8.4) — "inline-or-gated". A `value`-send on a proven site bypasses the
    // IC-mono check below (its receiver is a statically-known block; it need not
    // be IC-mono to be splicable).
    let has_closure = {
        let mut b = 0usize;
        let mut found = false;
        while b < method.bytecode_len() {
            let (instr, next) = decode_at(method, b);
            if matches!(instr, Instr::PushClosure { .. }) {
                found = true;
                break;
            }
            b = next;
        }
        found
    };
    let escape = if has_closure {
        let e = crate::compiler::escape::analyze_customized(vm, method, rcvr_klass);
        // S24 A3 (design §2.7): an ESCAPING closure is no longer an outright
        // reject — `ir::convert` allocates a real `BlockClosure`
        // (`Ir::Alloc` + `StoreField` inits, dead-home sentinel) when every
        // escaping block is transitively NLR-free. A3a: non-`captures_ctx`
        // blocks (no Context). A3b: a `captures_ctx` block forces M's
        // (`has_ctx`) Context to be MATERIALIZED in the prologue
        // (`method_ctx_vreg`), stored into the escaping closure's `copied[1]`.
        // `all_elidable` (S14 splice-everything, incl. 7-II-b Context ELISION)
        // still compiles unchanged. The transitive-NLR-free scan is the one
        // hard soundness gate.
        let escaping_ok = e.all_escaping_nlr_free(method);
        if !e.all_elidable && !escaping_ok {
            // An NLR-bearing escaping block (its `^` would misdeliver through
            // the dead-home sentinel) — cannot compile M. S24 B3: when the
            // classification consulted the rcvr_klass devirt path, this
            // verdict is a function of the CUSTOMIZATION klass — another
            // klass's inputsDo: may splice fine. NoRetryLater (sets nothing)
            // instead of NoPermanent (which sets the METHOD-WIDE
            // compile_disabled bit and would poison every other klass).
            return if e.klass_sensitive {
                Eligibility::NoRetryLater
            } else {
                Eligibility::NoPermanent
            };
        }
        Some(e)
    } else {
        None
    };
    // S24 B3: klass-sensitivity of the analysis, for the site-verdict loop
    // below — an escape-classified site skips its IC check there, so a
    // NoPermanent from an UNclassified site may flip under another klass.
    let ks = escape.as_ref().is_some_and(|e| e.klass_sensitive);
    // S14 step 7-II-b: a `has_ctx` method's Context elides ONLY because every
    // capturing block is inlined. A has_ctx method with NO (elidable) closure has
    // nothing justifying the elision — real frontend output never produces one
    // (has_ctx ⟺ a capturing block exists), so reject the degenerate case rather
    // than silently compile a method whose Context has no promotable owner.
    if method.has_ctx() && escape.is_none() {
        return Eligibility::NoPermanent;
    }
    // S24 A3b review follow-up (found by the ccloop repro, silent wrong
    // answers): if M's FIRST bytecode is a loop header (some jump_back
    // targets bci 0), CFG block 0 — where `ir::convert` emits the has_ctx
    // prologue (the materialize Context alloc, or 7-II-b's elided ctx-vreg
    // nil-inits) — is re-entered on EVERY back edge, re-running the prologue
    // per iteration: a materialized Context is re-allocated (each escaping
    // closure captures its own per-iteration snapshot instead of the ONE
    // shared Context the interpreter gives them) and elided ctx vregs are
    // re-nil'ed (captured-temp mutations lost). Decline — fail-soft, the
    // method stays interpreted. The shape needs a has_ctx method with NO
    // captured params (their bci-0 capture stores would push the loop header
    // past bci 0) and a loop as its first statement — rare enough that a
    // synthetic non-branch-target entry block isn't yet warranted.
    if method.has_ctx() {
        let mut b = 0usize;
        while b < method.bytecode_len() {
            let (instr, next) = decode_at(method, b);
            if let Instr::JumpBack(d) = instr {
                // jump_back target = next_bci − distance (builder.rs) — a
                // target of 0 means block 0 is a loop header.
                if next == d as usize {
                    return Eligibility::NoPermanent;
                }
            }
            b = next;
        }
    }

    let mut verdict = Eligibility::Yes;
    let mut bci = 0usize;
    while bci < method.bytecode_len() {
        let (instr, next) = decode_at(method, bci);
        match instr {
            Instr::PushSelf
            | Instr::PushNil
            | Instr::PushTrue
            | Instr::PushFalse
            | Instr::PushSmi(_)
            | Instr::PushLiteral(_)
            | Instr::PushTemp(_)
            | Instr::StoreTemp(_)
            | Instr::StoreTempPop(_)
            | Instr::PushInstvar(_)
            | Instr::PushGlobal(_)
            | Instr::Pop
            | Instr::Dup
            | Instr::JumpFwd(_)
            | Instr::JumpBack(_)
            | Instr::BrTrueFwd(_)
            | Instr::BrFalseFwd(_)
            | Instr::ReturnTos
            | Instr::ReturnSelf => {}
            Instr::Send { ic, super_ } => {
                // S14 step 7-I: a `value`-send on a PROVEN elidable closure site
                // (the escape pre-pass mapped this bci to a splice) is eligible
                // regardless of its IC — its receiver is a statically-known block
                // that `ir::convert` splices inline, so the IC-mono check below
                // (meant for ordinary dynamic dispatch) does not apply.
                if escape.as_ref().is_some_and(|e| {
                    e.value_send_target(bci).is_some() || e.blockarg_send_target(bci).is_some()
                }) {
                    // Yes — never worse than the current verdict.
                    bci = next;
                    continue;
                }
                // D4.6: a super send's own target is resolved statically
                // (holder.superclass(), fixed at compile time) rather than
                // through the interpreter's own IC lattice, so
                // `mono_smi_inline_send`'s "is this site's own IC already
                // mono-smi" check is meaningless for it — skip straight to
                // `Yes` (never worse than the current verdict: `Yes` is
                // this enum's own best case).
                if !super_ {
                    verdict = verdict.worse(mono_smi_inline_send(vm, method, ic));
                    if verdict == Eligibility::NoPermanent {
                        // Short-circuit: nothing later can undo this. S24 B3:
                        // under a klass-sensitive analysis a site failing
                        // HERE may be escape-classified (and skipped) under
                        // another klass — don't poison the method-wide bit.
                        return if ks {
                            Eligibility::NoRetryLater
                        } else {
                            Eligibility::NoPermanent
                        };
                    }
                }
            }
            // S11 step 7 (D1): instvar/global stores are now allowed --
            // `ir.rs`'s own `StoreField{barrier:true}` conversion handles
            // both (mirrors `PushInstvar`/`PushGlobal`'s existing
            // read-side handling exactly).
            Instr::StoreInstvarPop(_) | Instr::StoreGlobalPop(_) => {}
            // S14 step 7-I: a `push_closure` in M's own top-level bytecode is
            // allowed ONLY when the escape pre-pass above proved EVERY closure
            // site elidable (we returned `NoPermanent` otherwise). `ir::convert`
            // emits no IR for it (it splices the block body at the value-send).
            Instr::PushClosure { .. } => {
                debug_assert!(
                    escape.is_some(),
                    "a push_closure reached the scan but has_closure was false"
                );
            }
            // S14 step 7-II-b: M's own captured temp access. Allowed at DEPTH 0
            // (M's own Context) — `ir::convert` promotes it to a vreg and elides
            // M's Context. A has_ctx M always creates a capturing closure, so the
            // escape pre-pass (`escape.is_some()` + all_elidable) already proved
            // the Context elidable. A `depth != 0` (nested-context access) is a
            // later slice → NoPermanent.
            Instr::PushCtxTemp { depth, .. } | Instr::StoreCtxTempPop { depth, .. } => {
                if depth != 0 || escape.is_none() {
                    return Eligibility::NoPermanent;
                }
            }
            // Block returns / NLR only legitimately appear INSIDE a block body
            // (reached via the splice, not this scan). At M's top level they are
            // malformed / an explicit NLR (7-III) → still excluded.
            Instr::BlockReturnTos | Instr::NlrTos => return Eligibility::NoPermanent,
        }
        bci = next;
    }
    if verdict != Eligibility::Yes {
        // S14 step 3: `mono_smi_inline_send` no longer returns `NoRetryLater`
        // (an `Empty` IC is now `Yes` — compilable as a trap), so in practice
        // `verdict` is always `Yes` here (a `NoPermanent` site short-circuits
        // above). This guard is kept as a defensive catch-all: any future
        // `NoRetryLater`/`NoPermanent` producer added to the loop still returns
        // its verdict rather than falling through to the frame-budget check.
        return verdict;
    }

    // Frame budget (D1 point 3): reuses ir.rs's own CFG-aware stack-depth
    // worklist (`compute_entry_depths`) rather than a second, separately
    // maintained scan — `decode`/`compute_entry_depths` are both safe to
    // call on any structurally-valid method regardless of D1 eligibility
    // (neither cares whether a `Send` site is mono-smi-guarded, only that
    // its IC has a recorded argc).
    let cfg = decode::decode(method);
    let (_, max_stack) = ir::compute_entry_depths(method, &cfg);
    if method.ntemps() as i32 + max_stack > FRAME_BUDGET_SLOTS {
        if vm.options.trace.is_enabled("jit") {
            eprintln!(
                "[jit] NoPermanent reason: frame budget (ntemps {} + max_stack {} > {})",
                method.ntemps(),
                max_stack,
                FRAME_BUDGET_SLOTS
            );
        }
        return Eligibility::NoPermanent;
    }

    Eligibility::Yes
}

/// S24 A1 (design §2.1, amended): eligibility for compiling a
/// CompiledBlock body as its own nmethod. Structural gates: argc <= 3 (the
/// `value`..`value:value:value:` arity range), no `has_ctx` (a block that
/// allocates its own Context is a later slice), no `PushClosure` (nested
/// closure creation — later slice; also what makes A3's transitive
/// NLR-free scan vacuous for compiled block BODIES in v1), ctx-temp access
/// at depth 0 only and only when `captures_ctx` (the home Context arrives
/// as `copied[1]`), the standard bytecode-length / send-site-argc / frame
/// budget caps. `NlrTos`/`BlockReturnTos` are ALLOWED — that is the point
/// (§2.4: in A1 every closure's home is an interpreter frame). Value
/// captures are declined at the TRIGGER (the closure's `ncopied` is a
/// creation-site property the block alone cannot reveal), not here.
fn eligibility_detail_block(vm: &VmState, method: MethodOop) -> Eligibility {
    debug_assert!(method.is_block());
    if method.argc() > 3 || method.has_ctx() || method.bytecode_len() > MAX_BYTECODE_LEN {
        if vm.options.trace.is_enabled("jit") {
            eprintln!(
                "[jit] block NoPermanent: argc={} has_ctx={} bc_len={}",
                method.argc(),
                method.has_ctx(),
                method.bytecode_len()
            );
        }
        return Eligibility::NoPermanent;
    }
    let mut verdict = Eligibility::Yes;
    let mut bci = 0usize;
    while bci < method.bytecode_len() {
        let (instr, next) = decode_at(method, bci);
        match instr {
            Instr::PushSelf
            | Instr::PushNil
            | Instr::PushTrue
            | Instr::PushFalse
            | Instr::PushSmi(_)
            | Instr::PushLiteral(_)
            | Instr::PushTemp(_)
            | Instr::StoreTemp(_)
            | Instr::StoreTempPop(_)
            | Instr::PushInstvar(_)
            | Instr::PushGlobal(_)
            | Instr::Pop
            | Instr::Dup
            | Instr::JumpFwd(_)
            | Instr::JumpBack(_)
            | Instr::BrTrueFwd(_)
            | Instr::BrFalseFwd(_)
            | Instr::ReturnTos
            | Instr::ReturnSelf
            | Instr::StoreInstvarPop(_)
            | Instr::StoreGlobalPop(_)
            // The block's own returns — local and non-local — are exactly
            // what `convert`'s is_block mode lowers (Ret / NlrReturn).
            | Instr::BlockReturnTos
            | Instr::NlrTos => {}
            Instr::Send { ic, super_ } => {
                if crate::interpreter::ic::InterpreterIc::at(method, ic).argc() > 7 {
                    return Eligibility::NoPermanent; // register-marshaling cap
                }
                if !super_ {
                    verdict = verdict.worse(mono_smi_inline_send(vm, method, ic));
                    if verdict == Eligibility::NoPermanent {
                        return Eligibility::NoPermanent;
                    }
                }
            }
            Instr::PushClosure { .. } => return Eligibility::NoPermanent,
            Instr::PushCtxTemp { depth, .. } | Instr::StoreCtxTempPop { depth, .. } => {
                if depth != 0 || !method.captures_ctx() {
                    return Eligibility::NoPermanent;
                }
            }
        }
        bci = next;
    }
    if verdict != Eligibility::Yes {
        return verdict;
    }
    let cfg = decode::decode(method);
    let (_, max_stack) = ir::compute_entry_depths(method, &cfg);
    if method.ntemps() as i32 + max_stack > FRAME_BUDGET_SLOTS {
        return Eligibility::NoPermanent;
    }
    Eligibility::Yes
}

/// S24 A1: compile CompiledBlock `blk`'s body as its own nmethod,
/// registered in `CodeTable::by_block`. `rcvr_klass` is the
/// no-customization filler (`closure_klass`): a block nmethod has no entry
/// guard and never self-devirts (`ir::Translator::devirt_self_target`
/// declines for blocks), so the customization key is never consulted.
pub fn compile_block(vm: &mut VmState, blk: MethodOop) -> Option<NmethodId> {
    compile_block_versioned(vm, blk, 0)
}

/// S24 A1: [`compile_block`] with an explicit version — the
/// `note_uncommon_trap` is_block branch passes `old.version + 1`, mirroring
/// `compile_method_versioned`'s recompile discipline (install-first; the
/// by_block successor guard relies on it).
pub fn compile_block_versioned(vm: &mut VmState, blk: MethodOop, version: u8) -> Option<NmethodId> {
    debug_assert!(blk.is_block(), "compile_block: not a CompiledBlock");
    let closure_klass = vm.universe.closure_klass;
    compile_method_full(vm, closure_klass, blk, version, None)
}

/// D1 point 2: `ic_idx`'s own site must already be `Mono`, guarded on
/// `SmallInteger`, targeting a method whose primitive is in [`SMI_INLINE`]
/// — OR (S11 D7) a mono `basicNew` site, which `ir.rs` compiles to an
/// inline `Ir::Alloc`. Anything else (`Empty`, `Poly`, `Mega`, a non-smi
/// non-basicNew guard, or a mono-smi target whose primitive isn't fusable)
/// keeps this method interpreted. See `eligibility_detail`'s own doc for why
/// this stays narrower than D1's full "arbitrary send in any IC state" text.
fn mono_smi_inline_send(_vm: &VmState, method: MethodOop, ic_idx: u16) -> Eligibility {
    match ic_state(method, ic_idx) {
        // S14 step 3: an `Empty` IC (the site never executed while interpreted)
        // no longer BLOCKS compilation. It is now COMPILABLE as an uncommon
        // TRAP (`SiteFeedback::Untaken` -> `inline::decide` -> `Ir::UncommonTrap`
        // at the generic send site in `ir.rs`): Self's lazy cold path. When the
        // trap fires it re-executes the send interpreted, warming the IC for the
        // NEXT compilation. So `Empty` is `Yes` here, not `NoRetryLater` — a
        // method whose ONLY sends are `Empty` still compiles (a method full of
        // traps, valid, just cold). This is what lets a first-call-is-the-
        // trigger method (`MACVM_JIT=threshold=1`, where every inner send is
        // guaranteed cold on the first attempt) compile immediately instead of
        // deferring to a warmer next call.
        //
        // INTERIM DEOPT-STORM (documented, NOT solved in step 3): an executed
        // trap warms the IC but the nmethod stays Alive with the trap still
        // compiled, so it re-traps every call until recompiled with the warm
        // feedback. That recompile-on-trap loop is S14 step 8 (recompile.rs) and
        // needs `trap_counts`/`UNCOMMON_TRAP_LIMIT`, which S13 never built. The
        // storm is CORRECTNESS-PRESERVING (each trap re-executes the send
        // exactly, identical output) — just slow, bounded per run by the call
        // count. No trap counting / recompilation is added here.
        IcState::Empty => return Eligibility::Yes,
        IcState::Mono => {}
        // S14: a POLY or MEGA site no longer blocks compilation — it compiles as
        // a generic compiled-IC `CallSend` (`inline::decide` → `Call`, there is
        // no single target to speculate on). The S11 IC machinery handles the
        // polymorphism at runtime: the site's `bl` starts at `stub_resolve` and
        // transitions Mono → PIC → Mega exactly as an interpreter IC does
        // (S11 step 5, PIC/mega already built + tested). Inlining the dominant
        // poly case behind a klass guard (DominantWithSlowPath) is a later
        // optimization; this just STOPS BLOCKING the method. The `it_world`
        // differential gate covers the runtime correctness.
        IcState::Poly(_) | IcState::Mega => return Eligibility::Yes,
    }
    let site = InterpreterIc::at(method, ic_idx);
    let Some(target) = MethodOop::try_from(site.target()) else {
        // S14 step 8: a mono site whose target is a compiled NMETHOD ID (a smi
        // handle — `set_mono_compiled` rewrites the IC once the callee
        // compiles). The method-shape checks below need a MethodOop, but a
        // generic mono send compiles fine WITHOUT one (`Ir::CallSend` through
        // the S11 machinery, exactly the 962be22 widening) — and
        // `feedback::read_send_site` resolves compiled ids on its own for the
        // inline decision. The old `NoPermanent` here PERMANENTLY DISABLED any
        // method recompiled after its callee had compiled — the exact shape
        // the recompile-on-trap loop produces (found by its storm test).
        return Eligibility::Yes;
    };
    // S11 D7: a mono `basicNew` site clears eligibility. `ir.rs` turns it
    // into an inline `Ir::Alloc` when the receiver is a compile-time Slots
    // class constant, else an ordinary generic send to the basicNew
    // PRIMITIVE — which allocates and returns WITHOUT re-entering the
    // interpreter's bytecode (a shallow c2i-to-primitive hop, not the deep
    // block-activation reentrancy the broader D1 relaxation reverted in
    // step 7 was tripping on), so admitting it here is safe. `argc == 0`:
    // `basicNew:` (prim 24, indexable) is a different, non-inlined thing.
    if target.primitive() == crate::compiler::ir::PRIM_BASIC_NEW && site.argc() == 0 {
        return Eligibility::Yes;
    }
    // S14 step 4b/4c: a mono send whose callee is a cheap NON-PRIMITIVE body is
    // inlinable regardless of the receiver klass — `ir::convert` splices it
    // behind a receiver-klass guard (cold path = an uncommon trap). This admits
    // a mono-non-smi accessor send (`^self val`, `val` a leaf — 4b) OR a cheap
    // single-block NON-leaf callee (`run [ ^self bar ]` — 4c), either of which
    // would otherwise be rejected as a "mono but non-smi guard" below. A
    // PRIMITIVE target is excluded (its bytecode is only the failure fallback,
    // not its real behaviour). Kept in EXACT lockstep with
    // `inline::decide_with_budget`'s own gate (leaf OR inline-eligible non-leaf,
    // non-primitive, within budget) so the eligibility check and the actual
    // inline decision never disagree. Budget-checked at level 1 (tier-1 level).
    if target.primitive() == 0
        && crate::compiler::inline::inline_cost(target)
            <= crate::compiler::inline::budget_for_level(1).per_call_cost
        && (crate::compiler::inline::is_leaf(target)
            || crate::compiler::inline::is_inline_eligible_nonleaf(target))
    {
        return Eligibility::Yes;
    }
    // S14 (the S11-deferred D1 widening): any remaining Mono send is compilable
    // as a plain compiled-IC `Call` (S11). The site dispatches to its single
    // known target — a direct compiled call if that target is compiled, else a
    // shallow c2i hop that interprets it and returns. This is neither
    // smi-inlinable (handled as a fused fast path in `ir::is_smi_inlinable`) nor
    // inline-eligible (handled above), but S11's compiled-IC send path handles a
    // generic mono send, and S14 step 4c validated exactly this shape — an
    // inlined non-leaf callee's OWN sends become generic `CallSend`s and run
    // correctly, incl. c2i re-entry. So a mono send NO LONGER blocks compilation
    // regardless of its guard klass or its target's primitive; `ir::convert`
    // lowers it to `Ir::CallSend` (`inline::decide` -> `Call`). The `it_world`
    // differential gate (interpreter vs `MACVM_JIT=threshold=1`) is the safety
    // net for the broader c2i-reentrancy surface this opens.
    Eligibility::Yes
}

/// D4: `eligible`? -> decode -> convert -> regalloc -> emit -> publish ->
/// install. `None` on ineligibility (having set `compile_disabled` so the
/// counter-overflow trigger, S10 step 8, doesn't re-attempt this method
/// every 10k sends) or on code-cache exhaustion (having disabled the JIT
/// trigger for the rest of this process — every future attempt would fail
/// the identical way, so there is nothing to gain by trying again).
///
/// Allocates NOTHING on the Smalltalk heap (D4): `decode`/`convert`/
/// `regalloc`/`emit` are pure Rust computations over already-existing
/// bytecode/IC/pool data, so no `HandleScope` is needed here and no GC can
/// strike mid-compile.
pub fn compile_method(
    vm: &mut VmState,
    rcvr_klass: KlassOop,
    method: MethodOop,
) -> Option<NmethodId> {
    compile_method_versioned(vm, rcvr_klass, method, 0)
}

/// S14 step 8: [`compile_method`] with an explicit `version` — the
/// recompile-on-trap loop passes `old.version + 1` so the thrash cap
/// (`recompile::MAX_VERSIONS`) can count replacement generations.
pub fn compile_method_versioned(
    vm: &mut VmState,
    rcvr_klass: KlassOop,
    method: MethodOop,
    version: u8,
) -> Option<NmethodId> {
    compile_method_full(vm, rcvr_klass, method, version, None)
}

/// S15 A2: compile WITH an OSR entry at `osr_bci` (a root-scope loop-header
/// bci — the backward jump target that overflowed the loop counter). The
/// result is a NORMAL nmethod for the key (it also serves future calls; it
/// replaces the CodeTable entry) that additionally carries an `OsrMap` +
/// entry block. Declines (None) when the method/header shape is outside the
/// v1 OSR envelope — the caller resets the loop counter and keeps
/// interpreting.
pub fn compile_method_osr(
    vm: &mut VmState,
    rcvr_klass: KlassOop,
    method: MethodOop,
    osr_bci: u16,
) -> Option<NmethodId> {
    // S24 L2 step 5 (H7, version-churn coupling): inherit the key's CURRENT
    // generation instead of hardwiring 0. Before this, a trap-retired OSR
    // nmethod's recompile (osr_bci=None → the replacement has no OSR entry)
    // left the still-hot loop re-arming a FRESH v0 every 10k backedges —
    // `note_uncommon_trap`'s MAX_VERSIONS cap reset each rebirth and the
    // compile churn was unbounded. Inheriting lets the cap accumulate
    // across OSR re-arms exactly like call-path recompiles.
    let version = crate::oops::wrappers::SymbolOop::try_from(method.selector())
        .and_then(|sel| vm.code_table.lookup(rcvr_klass, sel))
        .and_then(|id| vm.code_table.get(id))
        .map(|nm| nm.version)
        .unwrap_or(0);
    compile_method_full(vm, rcvr_klass, method, version, Some(osr_bci))
}

fn compile_method_full(
    vm: &mut VmState,
    rcvr_klass: KlassOop,
    method: MethodOop,
    version: u8,
    osr_bci: Option<u16>,
) -> Option<NmethodId> {
    let elig = if method.is_block() {
        eligibility_detail_block(vm, method)
    } else {
        // S24 B3: the customization klass reaches the escape pre-pass so
        // self-receiver block-arg sends devirtualize. Every analysis run of
        // this compile (here, the OSR gate below, convert's recompute) sees
        // the SAME Some(rcvr_klass) — the front-5 agreement invariant.
        eligibility_detail(vm, method, Some(rcvr_klass))
    };
    match elig {
        Eligibility::Yes => {}
        Eligibility::NoPermanent => {
            method.set_compile_disabled();
            if vm.options.trace.is_enabled("jit") {
                eprintln!(
                    "[jit] ineligible, compile_disabled: {}",
                    selector_string(method)
                );
            }
            return None;
        }
        Eligibility::NoRetryLater => {
            // Counter left as-is (D1's own dont-compile-bit rationale is
            // about a permanently ineligible method re-firing every 10k
            // sends — a cold IC isn't that; it's warm by the time this
            // same method is next called, see Eligibility's own doc).
            if vm.options.trace.is_enabled("jit") {
                eprintln!(
                    "[jit] not yet eligible (cold inner IC), retry next call: {}",
                    selector_string(method)
                );
            }
            return None;
        }
    }

    // S24 L2 step 3 OSR envelope (phase A — was S15 A2 v1's "no has_ctx, no
    // closures"): non-ctx closure-BEARING methods now OSR. The v1 comment's
    // worry ("escape-mode operand stacks can hold phantoms the transfer
    // buffer cannot represent") is real but does not apply at the points OSR
    // actually enters: loop-HEADER entry stacks are phantom-free by ir.rs's
    // own invariant (pinned below by the plan-builder tripwire, which
    // debug-asserts and release-DECLINES if a `stack_closures` fact ever
    // covers a header's entry stack). Phantom TEMPS at the header are fine:
    // a live one packs the interpreter's REAL closure oop into the filler
    // vreg's spill home (valid, GC-safe, and never read — every compiled use
    // was CFG-spliced against the elision); dead-but-homed ones use the
    // existing packing + method-end widening; homeless ones are omitted and
    // covered by the entry block's nil-fill. Pre-loop A3a escaping closures
    // in temps transfer as ordinary oops via the Slot arm (their NLR-free
    // proof — `all_escaping_nlr_free`, checked in eligibility over the same
    // bytecode the interpreter ran — means the real-HomeRef-vs-dead-sentinel
    // asymmetry is never consulted; the frame-serial check is the runtime
    // backstop). `has_ctx` (phase B, osr_closure_design.md step 4): a
    // MATERIALIZE-form has_ctx method OSRs by ADOPTING the interpreter
    // frame's existing heap Context into `method_ctx_vreg` — one more
    // transfer pair, zero codegen changes; identity is what makes it sound
    // (closures created interpretively BEFORE the OSR hold that same
    // Context at copied[1]; both tiers' ctx traffic keeps flowing through
    // the ONE object, and a later deopt hands it back unchanged). The one
    // UNSOUND case is the ELIDED form (`all_elidable`): the interpreted
    // prefix may have leaked a real closure over the frame Context through
    // a dynamically-dispatched block-arg send the static-plus-mono-IC
    // 7-IV-c proof never saw, and elided ctx-vreg writes would split from
    // it. Decline OSR only — normal-call compiles stay elided (R1
    // write-through rejected: it would pessimize every call to serve a
    // rare OSR; `osr_declined_elided_ctx` collects the evidence for ever
    // revisiting).
    if osr_bci.is_some() && method.has_ctx() {
        let e = crate::compiler::escape::analyze_customized(vm, method, Some(rcvr_klass));
        if e.all_elidable {
            vm.stats.osr_declined_elided_ctx += 1;
            return None;
        }
    }

    let cfg = decode::decode(method);
    let ir_method = ir::convert(vm, rcvr_klass, method, &cfg, osr_bci);
    // T8 (osr_closure_design.md): the pre-decode gate above and convert's
    // own materialize decision must agree — an OSR-compiled has_ctx method
    // is ALWAYS the materialize form (same `escape::analyze` inputs).
    debug_assert!(
        !(osr_bci.is_some() && method.has_ctx()) || ir_method.method_ctx_vreg.is_some(),
        "gate/form disagreement: OSR-compiling a has_ctx method that convert did not materialize"
    );
    // S24 B4/B5 observables: spliced-NLR blocks + multi-BB grafts.
    vm.stats.blocks_spliced_nlr += ir_method.spliced_nlr as u64;
    vm.stats.blocks_spliced_multibb += ir_method.spliced_multibb as u64;
    vm.stats.splice_declined_budget += ir_method.splice_declined_budget as u64;
    // Debugger (DBG3 companion): `MACVM_DBG_IR=<selector>` dumps every
    // compile of a matching selector at the IR level — blocks, instructions,
    // and the full literal pool with resolved values — the layer BETWEEN
    // bytecode (`describe`) and machine code (`disasm-native`). Debug builds
    // only, exact selector match, stderr. This is what separated "the
    // compiler decided wrong" from "the runtime dispatched wrong" in the
    // task-#88 c2i-dispatch investigation.
    #[cfg(debug_assertions)]
    if let Ok(want) = std::env::var("MACVM_DBG_IR") {
        let sel = crate::oops::wrappers::SymbolOop::try_from(method.selector())
            .map(|s| s.as_string())
            .unwrap_or_default();
        if sel == want {
            eprintln!("==== IR {sel} (v{version}) ====");
            for blk in &ir_method.blocks {
                eprintln!("  block {} @bci{}:", blk.id.0, blk.bci);
                for ir in &blk.code {
                    eprintln!("    {ir:?}");
                }
            }
            for (i, e) in ir_method.pool.iter().enumerate() {
                eprintln!("  pool[{i}] = {:#x} {:?}", e.value, e.kind);
            }
            eprintln!("==== END IR {sel} ====");
        }
    }
    let mut regalloc_result = regalloc::regalloc(&ir_method);

    let mut asm = JasmAssembler::new();
    let stub_poll_addr = vm.stubs.stub_poll_addr();
    let must_be_boolean_addr = vm.stubs.must_be_boolean_addr();
    let alloc_slow_addr = vm.stubs.alloc_slow_addr();
    let box_double_addr = vm.stubs.box_double_addr();
    let box_float64x2_addr = vm.stubs.box_float64x2_addr();
    let box_float32x4_addr = vm.stubs.box_float32x4_addr();
    let box_int32x4_addr = vm.stubs.box_int32x4_addr();
    let call_primitive_addr = vm.stubs.call_primitive_addr();
    let nlr_originate_addr = vm.stubs.nlr_originate_addr();
    // `eligibility_detail` already confirmed (via `is_shimmable_primitive`)
    // that a nonzero `method.primitive()` here is safe to compile a shim
    // for — `argc_plus_recv` is the method's OWN argc+1 (receiver), never
    // the send-site `IcSite::argc` (there is no inline cache for this —
    // it's an unconditional call baked in at compile time).
    let prim_shim: Option<(i64, u8)> = if method.primitive() != 0 {
        Some((method.primitive(), method.argc() as u8 + 1))
    } else {
        None
    };

    // WINVM (Phase 3): the x64 back end covers a subset of the AArch64
    // one. Decline what it cannot lower — the method stays interpreted,
    // which is always correct — instead of emitting approximate code.
    #[cfg(not(target_arch = "aarch64"))]
    {
        // An OSR compile is declined WITHOUT disabling the method: only
        // the on-stack-replacement entry is unsupported, and the ordinary
        // call-path compile of this same method may well succeed. This is
        // the same fail-soft shape the OSR planner below already uses for
        // a header it cannot resolve.
        if let Some(reason) = x64_decline_reason(&ir_method, prim_shim) {
            if vm.options.trace.is_enabled("jit") {
                eprintln!(
                    "[jit] x64 back end declines {}: {reason}",
                    selector_string(method)
                );
            }
            // Permanent: these gaps are properties of the back end, not
            // of this run's type feedback, so a retry would decline
            // again for the same reason.
            method.set_compile_disabled();
            return None;
        }
    }

    let guard = emit::EntryGuard {
        smi_klass_bits: vm.universe.smi_klass.oop().raw(),
        key_klass_bits: rcvr_klass.oop().raw(),
        resolve_addr: vm.stubs.resolve_addr(),
    };
    // S15 A2 steps 4-5: locate the header block (decode block index == IR
    // block id for original blocks), resolve its live-in entities against
    // its own linear start position, and hand emit the copy plan. Entities
    // whose interval is DEAD at the header resolve to Nil and are OMITTED
    // (scope descs materialize dead slots as nil on a later deopt — the
    // doc's honest-caveat rule). A header that cannot be found (bci not a
    // block start) or whose entry stack is unavailable declines the whole
    // OSR compile.
    let mut osr_plan: Option<(emit::EmitOsr, Vec<crate::compiler::scopes::OsrSlot>)> = None;
    if let Some(bci) = osr_bci {
        let header_ix = cfg
            .blocks
            .iter()
            .position(|b| b.bci_start == bci as usize)?;
        let header = crate::compiler::ir::BlockId(header_ix as u32);
        let header_pos = *regalloc_result
            .block_start_pos
            .get(&header_ix.try_into().ok()?)?;
        let entry_stack = &ir_method.blocks[header_ix].entry_stack;

        // S24 L2 step 3 tripwire T9: the whole phase-A envelope leans on the
        // invariant that a loop HEADER's entry stack never holds an
        // elided-closure phantom (phantoms have no spill home, so the
        // transfer buffer cannot represent one — the v1 envelope's original
        // worry). Cross-check it against the deopt facts: a `stack_closures`
        // index BELOW the header's entry-stack depth anywhere in the method
        // means a phantom occupies the persistent region of the operand
        // stack that exists across the header — conservative over-match
        // (pre-loop deep-stack sites can false-positive), but structured
        // loops have EMPTY header stacks so the scan is trivially clean
        // today; a violation debug-asserts and release-DECLINES rather than
        // packing an unrepresentable value.
        let header_depth = entry_stack.len();
        if header_depth > 0 {
            let phantom_below_header = ir_method.blocks.iter().any(|b| {
                b.deopt_sites.iter().any(|(_, raw)| {
                    raw.stack_closures
                        .iter()
                        .any(|&(ix, _)| (ix as usize) < header_depth)
                })
            });
            if phantom_below_header {
                debug_assert!(
                    false,
                    "OSR header entry stack (depth {header_depth}) may hold an \
                     elided-closure phantom — the transfer buffer cannot \
                     represent it (T9)"
                );
                return None;
            }
        }

        use crate::compiler::scopes::{OsrSlot, OsrSource, ValueLoc};
        let n_slots = method.argc() + method.ntemps();
        let mut sources: Vec<(OsrSource, crate::compiler::ir::VReg)> =
            vec![(OsrSource::Receiver, crate::compiler::ir::VReg(0))];
        for i in 0..n_slots {
            sources.push((
                OsrSource::Slot(i as u16),
                crate::compiler::ir::VReg((1 + i) as u32),
            ));
        }
        for (j, &v) in entry_stack.iter().enumerate() {
            sources.push((OsrSource::StackSlot(j as u16), v));
        }
        // S24 L2 phase B: ADOPT the interpreter frame's heap Context into
        // the materialized ctx vreg — the packer's `OsrSource::Context` arm
        // has existed since S15, never emitted until now. Resolution is
        // guaranteed FrameSlot: the regalloc pin keeps the vreg live to
        // method end with `crosses_safepoint` (⇒ spilled), and the header
        // is never block 0 for a compilable has_ctx method (the 9de470b
        // loop-header eligibility decline — load-bearing here: adoption
        // must bypass the prologue's Alloc, which stays on the normal-call
        // path only).
        if let Some((cv, _nctx)) = ir_method.method_ctx_vreg {
            sources.push((OsrSource::Context, cv));
        }
        let mut copies = Vec::new();
        let mut slots = Vec::new();
        for (src, v) in sources {
            match crate::compiler::scopes::resolve_frame_loc(
                v,
                header_pos,
                &regalloc_result.intervals,
                &regalloc_result.extra_oop_live,
            ) {
                ValueLoc::FrameSlot(off) => {
                    debug_assert!(off < 0 && off % 8 == 0);
                    let slot = crate::compiler::regalloc::SpillSlot((-off / 8 - 1) as u16);
                    copies.push(slot);
                    slots.push(OsrSlot {
                        src,
                        dst_frame_off: off,
                    });
                }
                // Dead AT THE HEADER — but "dead" here means the vreg's
                // interval doesn't cover the header POSITION, which is NOT
                // the same as "no scope inside the loop will ever read its
                // slot": a temp whose only in-loop reference is a cold
                // arm's UncommonTrap has no organic in-loop use (the arm IS
                // the trap), yet every such trap's deopt scope still
                // records the temp's canonical FrameSlot via
                // `extra_oop_live`'s exact-position facts. The interpreter
                // frame being converted holds the temp's TRUE value right
                // now, so pack it into that slot anyway whenever the vreg
                // has a spill home at all — a deopt from inside the loop
                // then rebuilds the exact interpreter state. (Before this,
                // the entry block left such slots as native-stack garbage —
                // BUG D root cause 4's first half; and nil-filling them,
                // while GC-sound, handed a mid-loop deopt `nil` for a temp
                // the resumed interpreter genuinely reads — caught by
                // `osr_frame_deopts_mid_loop_and_finishes_interpreted`.)
                // A dead OPERAND-stack entry has no interpreter-side value
                // to pack (the loop-header operand stack is empty for
                // structured loops), and a vreg with NO spill home cannot
                // be named by any scope (scopes resolve through the same
                // intervals) — the entry block's nil-fill covers those.
                ValueLoc::Nil => {
                    // T1 (phase B): the Context source must resolve to a
                    // FrameSlot — the pin makes anything else a compiler
                    // bug. Fail-soft in release (decline the OSR compile).
                    if matches!(src, OsrSource::Context) {
                        debug_assert!(
                            false,
                            "materialized ctx vreg dead at the OSR header despite the \
                             regalloc pin (T1)"
                        );
                        return None;
                    }
                    if !matches!(src, OsrSource::StackSlot(_)) {
                        let slot_home = regalloc_result.intervals.iter().find_map(|iv| {
                            if iv.vreg == v {
                                if let Some(crate::compiler::regalloc::Assignment::Spill(s)) =
                                    iv.assignment
                                {
                                    return Some(s);
                                }
                            }
                            None
                        });
                        if let Some(s) = slot_home {
                            copies.push(s);
                            slots.push(OsrSlot {
                                src,
                                dst_frame_off: -8 * (s.0 as i32 + 1),
                            });
                            // The packed slot must stay a GC-scanned oop
                            // slot at EVERY later safepoint, or a mid-loop
                            // collection leaves the packed pointer stale
                            // and the eventual trap materializes garbage
                            // (caught by the scavenge-exit verifier on the
                            // BUG D repro). Blanket-widening is SOUND for
                            // exactly this vreg — its slot is written on
                            // every entry path before any widened position
                            // (normal entry: the prologue's unconditional
                            // temp nil-init; OSR entry: the entry block's
                            // nil-fill + this very copy) — the same
                            // dominates-the-whole-method argument that
                            // keeps ctx_vregs in `deopt_live_widen`.
                            let widen_end = regalloc_result
                                .intervals
                                .iter()
                                .map(|iv| iv.end)
                                .max()
                                .unwrap_or(0)
                                + 1;
                            if let Some(iv) =
                                regalloc_result.intervals.iter_mut().find(|iv| iv.vreg == v)
                            {
                                iv.end = iv.end.max(widen_end);
                                iv.crosses_safepoint = true;
                            }
                        }
                    }
                }
                // S24 L2 step 3 tripwire T7: `resolve_frame_loc` never
                // returns `ElidedClosure` today (the phantom override is
                // applied later, in `build_deopt_metadata`) — but if a
                // future refactor ever surfaces one here, DECLINE the OSR
                // compile fail-soft instead of the debug-abort below: a
                // phantom has no spill home for the transfer buffer to fill.
                ValueLoc::ElidedClosure(_) => {
                    return None;
                }
                other => {
                    debug_assert!(
                        false,
                        "OSR live-in resolved to non-slot {other:?} (compiler bug)"
                    );
                    return None;
                }
            }
        }
        osr_plan = Some((
            emit::EmitOsr {
                header,
                copies,
                reload_pos: header_pos,
            },
            slots,
        ));
    }
    let osr_req: Option<emit::EmitOsr> = osr_plan.as_ref().map(|(req, _)| emit::EmitOsr {
        header: req.header,
        copies: req.copies.clone(),
        reload_pos: req.reload_pos,
    });
    let frame_slots_for_osr = regalloc_result.frame_slots;

    // WINVM (Phase 3, MIGRATION.md §4): back-end selection. The x64
    // emitter covers a strict subset of the AArch64 one, and the three
    // gaps are handled by DECLINING the compile above rather than
    // approximating here — see `x64_decline_reason`. What remains is a
    // shape adaptation: `emit_x64` returns a struct with `block_pcs`
    // indexed by block and traps kept separate from call safepoints,
    // where `emit::emit` returns a 6-tuple with both already merged.
    #[cfg(not(target_arch = "aarch64"))]
    let (blob, block_pcs, verified_entry_off, emitted_ic_sites, safepoint_pcs, osr_off) = {
        // Unused on this path, and deliberately so: `emit_x64` takes only
        // the three runtime addresses it can actually emit calls to. The
        // six below all belong to ops it declines above (`FBox` and
        // `VecArith` for the boxers, `prim_shim` for `call_primitive`,
        // `NlrReturn` for `nlr_originate`), so plumbing them through
        // `RuntimeAddrs` would add fields no emitted instruction reads.
        // Each grows the struct when its op's lowering lands.
        let _ = (
            &mut asm,
            box_float64x2_addr,
            box_float32x4_addr,
            box_int32x4_addr,
            &osr_req,
        );
        let em = crate::compiler::emit_x64::emit_x64(
            &ir_method,
            &regalloc_result,
            crate::compiler::emit_x64::RuntimeAddrs {
                stub_poll: stub_poll_addr,
                must_be_boolean: must_be_boolean_addr,
                alloc_slow: alloc_slow_addr,
                box_double: box_double_addr,
                call_primitive: call_primitive_addr,
                nlr_originate: nlr_originate_addr,
                old_start: vm.reg_block.old_start,
            },
            if method.is_block() { None } else { Some(&guard) },
            prim_shim,
            osr_req.as_ref(),
        );
        let block_pcs: Vec<emit::BlockPc> = em
            .block_pcs
            .iter()
            .enumerate()
            .map(|(i, &pc_off)| emit::BlockPc {
                pc_off,
                bci: ir_method.blocks[i].bci,
            })
            .collect();
        // `build_deopt_metadata` keys EVERY deopt site off this list —
        // uncommon traps included, which the AArch64 emitter reports in
        // the same vector. Keeping them separate here would make a
        // trapping method panic in `build_deopt_metadata` ("no emitted
        // safepoint") rather than mis-compile, but the union is what the
        // contract actually asks for.
        let safepoint_pcs: Vec<emit::SafepointPc> = em
            .safepoints
            .iter()
            .chain(em.trap_sites.iter())
            .map(|t| emit::SafepointPc {
                pc_off: t.pc_off,
                bci: t.bci,
                position: t.position,
            })
            .collect();
        (
            em.blob,
            block_pcs,
            em.verified_entry_off,
            em.ic_sites,
            safepoint_pcs,
            em.osr_off,
        )
    };

    #[cfg(target_arch = "aarch64")]
    let (blob, block_pcs, verified_entry_off, emitted_ic_sites, safepoint_pcs, osr_off) =
        emit::emit(
            &mut asm,
            &ir_method,
            &regalloc_result,
            stub_poll_addr,
            must_be_boolean_addr,
            alloc_slow_addr,
            box_double_addr,
            box_float64x2_addr,
            box_float32x4_addr,
            box_int32x4_addr,
            call_primitive_addr,
            nlr_originate_addr,
            prim_shim,
            // S24 A1: a block nmethod has NO receiver-klass customization
            // (design §2.1 — every closure is a closure_klass instance;
            // instvar soundness comes from prefix-stable layouts, not an
            // entry guard), so verified_entry == entry.
            if method.is_block() { None } else { Some(guard) },
            osr_req.as_ref(),
        );

    // Debugger (MACVM_DBG_IR's second half): the same selector match also
    // dumps the EMITTED LISTING — the assembler's own per-instruction lines,
    // the layer below the IR dump above and the one that shows what the
    // machine actually does with each vreg (spill stores included, which the
    // IR alone can't show). Debug builds only, stderr.
    #[cfg(debug_assertions)]
    if let Ok(want) = std::env::var("MACVM_DBG_IR") {
        let sel = crate::oops::wrappers::SymbolOop::try_from(method.selector())
            .map(|s| s.as_string())
            .unwrap_or_default();
        if sel == want {
            eprintln!("==== LISTING {sel} (v{version}) ====");
            for line in &blob.listing {
                eprintln!("  {line}");
            }
            eprintln!("==== END LISTING {sel} ====");
        }
    }

    // D4.6: pre-resolve every `send_super` site's own target BEFORE
    // publishing anything -- resolving here (not at runtime, via
    // `stub_resolve`, like every other site) is the whole point of a
    // super send's own static klass. A lookup failure (the superclass
    // genuinely doesn't implement this selector) fails the WHOLE method's
    // own compile, same as any other compile failure here -- falls back
    // to the interpreter, which already handles a super-send DNU
    // correctly via its own existing mechanism; no new runtime-DNU-from-
    // compile-time path is needed. Allocates no Smalltalk heap memory
    // (`lookup`/`resolve_target_entry`'s own c2i-adapter fallback only
    // ever touches the CODE cache), so this doesn't disturb this
    // function's own "no HandleScope needed, no GC can strike mid-
    // compile" invariant (this function's own doc, above).
    //
    // `use_verified=true` here is load-bearing, not a micro-optimization:
    // a super site's own receiver is whatever `self` actually is at the
    // call site (`Leaf`, or any further subclass) -- essentially NEVER
    // the resolved target's own `key_klass` (`super_klass`, the STATIC
    // holder's superclass), since that's the whole point of `super`. If
    // the target were reached via its own `entry` (guarded, `use_verified
    // =false` -- Mono's own convention, since a Mono site's `bl` really
    // could later see a different RECEIVER klass through the same site),
    // that guard would almost always MISMATCH and misroute back through
    // `stub_resolve`, which re-resolves from `klass_of(receiver)` -- the
    // receiver's own actual klass, NOT `super_klass` -- silently
    // collapsing back into an ordinary dynamic send and reaching an
    // override `super` was specifically supposed to skip. Caught by this
    // step's own integration test (a 3-klass override chain where an
    // ordinary send and a super send provably reach different methods);
    // `entry` was the first thing tried and failed exactly this way.
    let mut super_resolutions: Vec<Option<(KlassOop, u64)>> =
        Vec::with_capacity(emitted_ic_sites.len());
    for s in &emitted_ic_sites {
        let resolved = match ir_method.call_sites[s.site as usize].static_klass {
            Some(super_klass) => {
                let target_method = lookup(vm, super_klass, s.selector)?;
                let target = resolve_super_target_entry(vm, super_klass, s.selector, target_method);
                Some((super_klass, target))
            }
            None => None,
        };
        super_resolutions.push(resolved);
    }

    let Some(h) = vm.code_cache.alloc(blob.code.len()) else {
        vm.options.jit = JitMode::Off;
        if vm.options.trace.is_enabled("jit") {
            eprintln!(
                "[jit] code cache exhausted ({} bytes wanted), disabling JIT for the rest of \
                 this run",
                blob.code.len()
            );
        }
        return None;
    };
    vm.code_cache.publish(h, &blob);

    // S12 D2: `oopmaps[0]` is always the reserved, always-empty map —
    // block-start descs (trace path only, never a real safepoint's own
    // return address, `OopMap::empty`'s own doc) point at it; every REAL
    // safepoint below gets its own liveness-intersected map, deduplicated
    // by content (`oopmap_dedup`: two safepoints with identical live sets
    // share one entry).
    let mut oopmaps: Vec<OopMap> = vec![OopMap::empty()];
    let mut safepoint_pcdescs: Vec<PcDesc> = Vec::with_capacity(safepoint_pcs.len());
    for sp in &safepoint_pcs {
        let map = oopmap::build_for_position(
            &regalloc_result.intervals,
            // F3c S1: the TOTAL slot universe — spills + the poll-save area.
            regalloc_result.frame_slots + regalloc_result.save_area_slots,
            sp.position,
            &regalloc_result.extra_oop_live,
            &regalloc_result.poll_saves,
        );
        let idx = oopmap::intern(&mut oopmaps, map);
        safepoint_pcdescs.push(PcDesc {
            pc_off: sp.pc_off,
            bci: sp.bci,
            oopmap: idx,
        });
    }
    let mut pcdescs: Vec<PcDesc> = block_pcs
        .iter()
        .map(|bp| PcDesc {
            pc_off: bp.pc_off,
            bci: bp.bci,
            oopmap: 0,
        })
        .chain(safepoint_pcdescs)
        .collect();
    // `bci_at`'s nearest-below lookup binary-searches this, ascending.
    pcdescs.sort_by_key(|d| d.pc_off);
    let key_selector = SymbolOop::try_from(method.selector())
        .expect("compile_method: method selector is not a Symbol");

    // Block-bci granularity (matching pcdescs' own precision): the block
    // containing this method's Ir::Poll, if it has one. A mixed-tier
    // stack-trace walker has no exact native pc to work from at a poll
    // callback (S11's last_compiled_pc anchor isn't wired up yet), so this
    // is the IR-derived approximation `Nmethod::poll_bci` documents.
    let poll_bci = ir_method
        .blocks
        .iter()
        .find(|b| b.code.iter().any(|instr| matches!(instr, ir::Ir::Poll)))
        .map(|b| b.bci);

    // S11 D3: every fresh site starts Unresolved -- its `bl` still
    // self-targets (`bl_patchable`'s own placeholder) until the patch pass
    // just below aims it at `stub_resolve` (D3: "no per-site
    // pre-resolution", exactly like an empty interpreter IC) -- EXCEPT a
    // `send_super` site (D4.6), already resolved above: it starts
    // `Mono{klass, target}` directly, its `bl` already pointing at the
    // real target, never touching `stub_resolve` at all on a normal run.
    let ic_sites: Vec<IcSite> = emitted_ic_sites
        .iter()
        .zip(&super_resolutions)
        .map(|(s, resolved)| IcSite {
            off: s.off,
            selector: s.selector,
            argc: s.argc,
            state: match resolved {
                Some((klass, target)) => CompiledIcState::Mono {
                    klass: *klass,
                    target: *target,
                },
                None => CompiledIcState::Unresolved,
            },
            // S13 step 10d: `super_resolutions` is `Some` iff this is a
            // `send_super` site (D4.6) — its `klass` IS the static
            // holder-superclass. Record it so a later `not_entrant_stub`
            // re-dispatch stays super-aware even if the state is reset.
            super_klass: (*resolved).map(|(klass, _)| klass),
        })
        .collect();

    // S13 step 3b: build this method's deopt scope blob + PcDescs from the
    // safepoints the converter flagged (`ir::IrBlock::deopt_sites`). Every
    // value read after a resume bci is spilled (S12 spill-all), so
    // `scopes::resolve_frame_loc` alone resolves receiver/slots/stack; a
    // value not live at the safepoint is dead -> Nil.
    let (deopt_scopes, deopt_pcdescs) =
        build_deopt_metadata(&ir_method, &regalloc_result, &safepoint_pcs);

    let nm = Nmethod {
        id: NmethodId(0), // overwritten by CodeTable::install
        osr_cold_sends: if osr_bci.is_some() {
            ir_method.osr_cold_sends
        } else {
            0
        },
        key_klass: rcvr_klass,
        key_selector,
        code: h,
        entry_off: 0,
        verified_entry_off,
        state: NmState::Alive,
        level: 1,
        version,
        trap_count: 0,
        // S14 step 8 (A5): the feedback profile this compile SAW — the
        // effectiveness check re-snapshots at trap time and declines the
        // recompile when nothing changed.
        // Customized (WINVM deltablue-storm fix): must be computed the
        // same way `note_uncommon_trap`'s effectiveness check recomputes
        // it, or the comparison is between different quantities.
        profile_hash: crate::compiler::feedback::snapshot_profile_customized(
            vm, method, rcvr_klass,
        ),
        literal_off: blob.literal_off,
        relocs: blob.relocs,
        // F3c S1: publish the TOTAL frame (spills + poll-save area). Save
        // slots have NO static oop-ness (the same register holds different
        // vregs at different polls), so they publish conservatively `true`
        // — `oopmap::verify`'s slot_is_oop cross-check stays meaningful for
        // the spill region, and the save region's per-poll truth is
        // enforced at build time by `regalloc::verify_poll_saves`.
        frame_slots: regalloc_result.frame_slots + regalloc_result.save_area_slots,
        slot_is_oop: {
            let mut v = regalloc_result.slot_is_oop.clone();
            v.extend(std::iter::repeat(true).take(regalloc_result.save_area_slots as usize));
            v
        },
        pcdescs,
        oopmaps,
        ic_sites,
        poll_bci,
        prim_call_argc_plus_recv: prim_shim.map(|(_, argc_plus_recv)| argc_plus_recv),
        // S24 A1: a block nmethod registers under its CompiledBlock in
        // `by_block` (install() routes on this); key_klass/key_selector
        // above are fillers for blocks (closure_klass + #aBlock), never
        // consulted for lookup.
        block_method: if method.is_block() {
            Some(method)
        } else {
            None
        },
        deopt_scopes,
        deopt_pcdescs,
        // S14 step 4b: the inline dependencies the converter recorded (one
        // `(receiver_klass, selector)` per spliced leaf). `deps::
        // affected_by_install` reads these to invalidate this nmethod when an
        // inlined callee is redefined.
        inline_deps: ir_method.inline_deps.clone(),
        self_devirt: ir_method.self_devirt,
        osr_map: osr_plan.as_ref().map(|(_, slots)| {
            let frame_bytes = ((8 * frame_slots_for_osr as i64) + 15) & !15;
            crate::compiler::scopes::OsrMap {
                osr_bci: osr_bci.expect("osr_plan implies osr_bci"),
                entry_off: osr_off.expect("emit built the entry block when asked"),
                frame_words: (frame_bytes / 8) as u32,
                slots: slots.clone(),
            }
        }),
    };
    // S12 D1 enforcement point 2: "debug + stress" — reuses the existing
    // heap-verifier's own gate (`MACVM_GC_VERIFY=1` opts a release build
    // in too) rather than a second, parallel env var for the same concept.
    if crate::memory::verify::verify_enabled() {
        oopmap::verify(&nm);
    }
    let resolve_addr = vm.stubs.resolve_addr();
    for (site, resolved) in emitted_ic_sites.iter().zip(&super_resolutions) {
        let patch_target = resolved.map_or(resolve_addr, |(_, target)| target);
        vm.code_cache.patch_call_site_at(h, site.off, patch_target);
    }
    vm.stats.compilations += 1; // S15 A8 tier-balance counter
    let has_osr = nm.osr_map.is_some();
    let id = vm.code_table.install(nm);
    vm.probe_ring
        .push(crate::runtime::vm_state::ProbeEvent::Compile { nm: id.0, version });
    if has_osr {
        let sel = crate::oops::wrappers::SymbolOop::try_from(method.selector())
            .expect("a method's selector is always a Symbol");
        vm.code_table.install_osr(
            rcvr_klass,
            sel,
            osr_bci.expect("has_osr implies osr_bci"),
            id,
        );
    }
    // S24 L2 step 2 — TRIGGER UNIFICATION (user-decided policy: "the loop
    // counters have detected in a different way that the method containing
    // the loop is hot; the method is now hot"). A by_key install means SOME
    // profile counter earned a compile; saturate the invocation counter to
    // the threshold so `activate_method`'s existing gate routes every future
    // CALL into it too. Before this, an OSR-earned nmethod served loop
    // entries while every call of the same method interpreted end-to-end
    // until call #threshold — deltablue's driver methods (~103 calls,
    // threshold 200) ran 56% of the remaining interpreted tail that way.
    // No-op for threshold-triggered installs (counter already ≥ n); never
    // lowers; adds NO new compilation (install already happened — the
    // no-unguided-compilation guardrail holds by construction).
    if let crate::runtime::JitMode::Threshold(n) = vm.options.jit {
        if method.saturate_invocation(n as i64) {
            vm.stats.trigger_unifications += 1;
        }
    }
    if vm.options.trace.is_enabled("jit") {
        eprintln!(
            "[jit] compiled {} -> nmethod {} ({} bytes, {} frame slots)",
            selector_string(method),
            id.0,
            h.len,
            regalloc_result.frame_slots
        );
    }
    Some(id)
}

fn selector_string(method: MethodOop) -> String {
    SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "<?>".to_string())
}

/// S13 step 3b: build a freshly compiled method's deopt scope blob + sorted
/// `scopes::PcDesc`s from the converter-flagged safepoints
/// (`ir::IrBlock::deopt_sites`).
///
/// Re-walks `block_order` in the SAME linear-position numbering `regalloc`
/// (`compute_intervals`) and `emit` share, so each deopt site's op position
/// is re-derived here and matched — via that position — to the emitted
/// [`emit::SafepointPc`] carrying its return-address `pc_off` (the deopt
/// `PcDesc.code_off` key, same value the S12 oopmap `PcDesc` uses for a
/// call). A deopt site is always a safepoint-emitting op, so its position
/// is always present in the map; a divergence between this walk and emit's
/// would surface as a hard panic (not a silent wrong-frame deopt), and the
/// golden decodes the blob to catch a subtler slip.
///
/// Every `VReg` resolves via [`scopes::resolve_frame_loc`]: S12's spill-all
/// invariant puts every value live across a safepoint in its canonical
/// frame slot, and a value NOT live there is dead → `Nil` (safe — never
/// read before being overwritten). One scope record PER SITE (undeduped):
/// linear-scan assigns a spill slot per interval, so a multi-def temp has
/// no single canonical slot; resolving at each site's own position is
/// unambiguous. `method_pool_ix` is a `0` placeholder until materialization
/// (step 6) interns the compile-time method oop — leaving the pool
/// untouched keeps existing listing goldens byte-stable.
/// WINVM (Phase 3): why the x86-64 back end cannot compile this method,
/// or `None` if it can.
///
/// Checked against [`emit_x64::SUPPORTED_OPS`] rather than against a
/// hand-maintained list of "risky" method shapes, because that is the
/// list `emit_x64` itself branches on: an op it does not lower panics
/// with an "unsupported op" message. Consulting the same set here turns
/// that panic — a compiler crash on a legal Smalltalk program — into a
/// method that simply stays interpreted.
///
/// `prim_shim` is checked separately because it is not an IR op at all
/// but an emitter *parameter*: `emit::emit` splices a primitive call into
/// the prologue when asked, and `emit_x64` has no such parameter. A scan
/// of the IR would never see it.
#[cfg(not(target_arch = "aarch64"))]
fn x64_decline_reason(
    ir_method: &ir::IrMethod,
    prim_shim: Option<(i64, u8)>,
) -> Option<String> {
    use crate::compiler::emit_x64::{ir_op_name, SUPPORTED_OPS};

    for block in &ir_method.blocks {
        for op in &block.code {
            let name = ir_op_name(op);
            if !SUPPORTED_OPS.contains(&name) {
                return Some(format!("IR op `{name}` is not lowered on x64"));
            }
        }
    }
    None
}

fn build_deopt_metadata(
    ir_method: &ir::IrMethod,
    regalloc_result: &regalloc::RegallocResult,
    safepoint_pcs: &[emit::SafepointPc],
) -> (Vec<u8>, Vec<crate::compiler::scopes::PcDesc>) {
    use crate::compiler::scopes::{
        resolve_frame_loc, CtxLoc, SafepointState, ScopeDescData, ScopeDescRecorder, SenderLink,
    };

    let sp_by_pos: std::collections::HashMap<u32, &emit::SafepointPc> =
        safepoint_pcs.iter().map(|sp| (sp.position, sp)).collect();

    let intervals = &regalloc_result.intervals;
    let extra_oop_live = &regalloc_result.extra_oop_live;
    let n_slots = ir_method.argc as usize + ir_method.ntemps as usize;
    // F3c S1: every scope resolution in THIS function must see the poll-save
    // records (a register-resident vreg at a LoopPoll resolves to its save
    // slot). The local closure shadows the imported 4-arg resolver so all
    // sites below switch atomically — a new site added later cannot forget.
    let poll_saves = &regalloc_result.poll_saves;
    let resolve_frame_loc = |vreg: ir::VReg,
                             position: u32,
                             intervals: &[regalloc::LiveInterval],
                             extra: &[(ir::VReg, u32)]| {
        crate::compiler::scopes::resolve_frame_loc_in(
            vreg, position, intervals, extra, poll_saves,
        )
    };
    let mut rec = ScopeDescRecorder::new();
    let mut pos = 0u32;
    for &bid in &regalloc_result.block_order {
        let block = &ir_method.blocks[bid.0 as usize];
        for (idx, _ir) in block.code.iter().enumerate() {
            if let Some((_, raw)) = block.deopt_sites.iter().find(|(ci, _)| *ci == idx as u32) {
                let sp = sp_by_pos.get(&pos).unwrap_or_else(|| {
                    panic!(
                        "build_deopt_metadata: deopt site at position {pos} has no emitted \
                         safepoint -- emit/regalloc position numbering diverged"
                    )
                });
                let position = sp.position; // == pos (map is keyed by it)
                                            // The ROOT (caller) scope — the depth-1 shape S13 always built:
                                            // this method's own receiver (VReg 0) + unified slots
                                            // (VReg 1..=n_slots), `sender: None`, root method_pool_ix.
                                            // S24 A1: a BLOCK compilation's root scope records the
                                            // CLOSURE as its receiver (design §2.5 — the materialized
                                            // interpreter block frame's receiver-ARG slot holds the
                                            // closure, activate_block_interp's own shape; FP+4 is
                                            // derived as copied[0] by the materializer's root-block
                                            // arm). A method root records self (VReg 0) as always.
                let root_receiver = resolve_frame_loc(
                    ir_method.block_closure_vreg.unwrap_or(ir::VReg(0)),
                    position,
                    intervals,
                    extra_oop_live,
                );
                let root_slots = (0..n_slots)
                    .map(|i| {
                        resolve_frame_loc(
                            ir::VReg(i as u32 + 1),
                            position,
                            intervals,
                            extra_oop_live,
                        )
                    })
                    .collect();
                let root_method_ix = ir_method
                    .method_pool_ix
                    .expect("a method with a deopt site interned its own method oop");

                // S14 step 7-II-b: if M owns a heap Context that was ELIDED (its
                // captured temps promoted to `ctx_vregs`), the root scope records
                // `CtxLoc::Elided` over those vregs' frame slots so a deopt allocs
                // a fresh Context and fills it (materializer M6). No ctx-vregs →
                // M never owned a Context → `None` (unchanged S13 behaviour).
                let root_ctx = if let Some((ctx_vreg, nctx)) = ir_method.method_ctx_vreg {
                    // S24 A3b: M MATERIALIZED its heap Context (an escaping
                    // capturing closure needs a real one). A deopt REUSES that
                    // same object by identity (materializer M6's
                    // `Materialized` arm) — the escaped closures reference it
                    // too (Risk 2). Keyed on the ctx vreg's ACTUAL LIVENESS,
                    // not a site fingerprint: the fb01b7a review found the
                    // original `bci == 0 && stack.is_empty()` key collided
                    // with real deopt-capable safepoints (a root loop-poll
                    // when M's first statement is a loop; an inlined callee's
                    // loop-poll recording CALLEE-relative bci 0; a bci-0
                    // AllocClosure) — stamping `CtxLoc::None` on their root
                    // scope left a deopted has_ctx frame with a NIL context
                    // while the escaped closures still held the real one
                    // (ctx_temp_walk panic / nil copied[1]). The vreg is dead
                    // only BEFORE its prologue def (the ctx-alloc safepoint
                    // itself); everywhere else the pin (regalloc) keeps it
                    // live, so `resolve_frame_loc` distinguishes the two
                    // cases exactly.
                    match resolve_frame_loc(ctx_vreg, position, intervals, extra_oop_live) {
                        // Pre-def window (the prologue alloc's own safepoint):
                        // hand the interpreter a FRESH nil-filled Context via
                        // the Elided path — NOT `CtxLoc::None`. The deopt
                        // materializer never re-runs `activate_method` (the
                        // original comment's claim that it re-allocates at
                        // bci 0 was wrong), so `None` would leave a has_ctx
                        // frame with no context; re-execution from bci 0
                        // re-runs the frontend's param→ctx-slot copies into
                        // the fresh one, which is exactly activation's own
                        // sequence. Unreachable today (Alloc-kind sites don't
                        // deopt via trap/return/poll) but correct if that
                        // ever changes.
                        crate::compiler::scopes::ValueLoc::Nil => CtxLoc::Elided {
                            temps: vec![crate::compiler::scopes::ValueLoc::Nil; nctx],
                        },
                        live => CtxLoc::Materialized(live),
                    }
                } else if ir_method.ctx_vregs.is_empty() {
                    CtxLoc::None
                } else {
                    CtxLoc::Elided {
                        temps: ir_method
                            .ctx_vregs
                            .iter()
                            .map(|&v| resolve_frame_loc(v, position, intervals, extra_oop_live))
                            .collect(),
                    }
                };

                // S14 step 4c: a safepoint INSIDE an inlined body records a
                // NESTED scope — the inlined callee's own receiver/slots/method
                // + a `SenderLink` to the (freshly begun) caller scope. A
                // root-method safepoint (`raw.inline == None`) records the
                // depth-1 caller scope directly, exactly as S13 did (no
                // regression — `sender: None`).
                let scope = match &raw.inline {
                    None => rec.begin_scope(ScopeDescData {
                        method_pool_ix: root_method_ix,
                        // S24 A1: true for a standalone block compilation —
                        // the materializer's root-block arm keys on
                        // "is_block AND outermost".
                        is_block: ir_method.block_closure_vreg.is_some(),
                        sender: None,
                        receiver: root_receiver,
                        slots: root_slots,
                        ctx: root_ctx,
                    }),
                    Some(site) => {
                        // S14 step 7-IV-b: the inline levels form a CHAIN
                        // (`site.parent` — a block spliced inside an inlined
                        // callee is depth 3: block ← callee ← root). Begin the
                        // ROOT scope first (`begin_scope`'s invariant: a
                        // SenderLink's target must exist before its child), then
                        // each level OUTERMOST-first; every pre-7-IV site has
                        // `parent: None` (a 2-scope chain, byte-identical to the
                        // old shape).
                        let mut chain: Vec<&ir::InlineSite> = vec![site];
                        while let Some(p) = &chain.last().unwrap().parent {
                            chain.push(p);
                        }
                        let mut prev_scope = rec.begin_scope(ScopeDescData {
                            method_pool_ix: root_method_ix,
                            is_block: ir_method.block_closure_vreg.is_some(),
                            sender: None,
                            receiver: root_receiver,
                            slots: root_slots,
                            ctx: root_ctx,
                        });
                        for level in chain.iter().rev() {
                            // The SenderLink: where this level's CALLER resumes
                            // (the inlined send's bci, advanced past it by the
                            // materializer) and the caller's frozen operand
                            // stack below the send's operands.
                            let pending_stack = level
                                .caller_pending_stack
                                .iter()
                                .map(|&v| resolve_frame_loc(v, position, intervals, extra_oop_live))
                                .collect();
                            let inl_receiver = resolve_frame_loc(
                                level.receiver,
                                position,
                                intervals,
                                extra_oop_live,
                            );
                            let mut inl_slots: Vec<_> = level
                                .slots
                                .iter()
                                .map(|&v| resolve_frame_loc(v, position, intervals, extra_oop_live))
                                .collect();
                            // S14 step 7-IV-c: a slot holding an ELIDED-CLOSURE
                            // phantom overrides its (filler) vreg location — the
                            // materializer allocates the real closure.
                            for &(slot_ix, pool_ix) in &level.slot_closures {
                                inl_slots[slot_ix as usize] =
                                    crate::compiler::scopes::ValueLoc::ElidedClosure(pool_ix);
                            }
                            prev_scope = rec.begin_scope(ScopeDescData {
                                method_pool_ix: level.method_pool_ix,
                                // S14 step 7-II: an inlined spliced BLOCK records
                                // an `is_block` scope (the materializer rebuilds
                                // a block activation frame).
                                is_block: level.is_block,
                                sender: Some(SenderLink {
                                    sender: prev_scope,
                                    sender_bci: level.sender_bci,
                                    pending_stack,
                                }),
                                receiver: inl_receiver,
                                slots: inl_slots,
                                ctx: CtxLoc::None,
                            });
                        }
                        prev_scope
                    }
                };
                let mut stack: Vec<_> = raw
                    .stack
                    .iter()
                    .map(|&v| resolve_frame_loc(v, position, intervals, extra_oop_live))
                    .collect();
                // S14 step 7-IV-c: phantom stack entries override their filler
                // vregs (a block-arg send's guard-cold reexecute stack; in-callee
                // sites with the phantom below a send's operands).
                for &(ix, pool_ix) in &raw.stack_closures {
                    stack[ix as usize] = crate::compiler::scopes::ValueLoc::ElidedClosure(pool_ix);
                }
                rec.record_site(
                    sp.pc_off,
                    SafepointState {
                        scope,
                        bci: raw.bci as u16,
                        kind: raw.kind,
                        reexecute: raw.reexecute,
                        stack,
                    },
                );
            }
            pos += 1;
        }
    }
    rec.pack()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::builder::BytecodeBuilder;
    use crate::oops::smi::SmallInt;
    use crate::runtime::vm_state::VmOptions;

    fn test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: JitMode::Off,
        })
    }

    /// A throwaway method standing in for a real SmallInteger primitive —
    /// `eligible` only ever reads its `primitive()` field, never executes
    /// its bytecode (same rationale as `ir::tests::primitive_stub`, not
    /// reused across the module boundary since it's a two-line helper).
    fn primitive_stub(
        vm: &mut VmState,
        sel: crate::oops::wrappers::SymbolOop,
        prim_id: i64,
    ) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let m = b.finish(vm, sel, 1, 0);
        m.set_primitive(prim_id);
        m
    }

    /// `self + arg`, argc=1: `push_self; push_temp(0); send(#+); ret_tos`.
    /// The IC is left at whatever state the caller sets up afterward —
    /// `ic_idx` 0 is always this send site's own IC (the method's only
    /// send).
    fn plus_method(vm: &mut VmState) -> MethodOop {
        let plus_sel = vm.universe.intern(b"+");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(vm, plus_sel, 1);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"plusArg:");
        b.finish(vm, m_sel, 1, 0)
    }

    fn set_mono_smi_plus_ic(vm: &mut VmState, method: MethodOop) {
        let plus_sel = vm.universe.intern(b"+");
        let plus_target = primitive_stub(vm, plus_sel, 1);
        let smi_klass = vm.universe.smi_klass;
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(vm, smi_klass, plus_target, epoch);
    }

    #[test]
    fn eligible_accepts_mono_smi_plus() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        set_mono_smi_plus_ic(&mut vm, method);
        assert!(eligible(&vm, method));
    }

    /// S14 step 3: an `Empty` IC (the site never executed while interpreted)
    /// is now ELIGIBLE — the generic send lowers to an uncommon trap
    /// (`SiteFeedback::Untaken` -> `inline::decide` -> `Ir::UncommonTrap`), so a
    /// cold site no longer blocks compilation. (Before this step it returned
    /// `NoRetryLater` -> `eligible == false`.)
    #[test]
    fn eligible_accepts_empty_ic_as_trap() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        // No `set_mono` call: the IC is left at its fresh, empty state — which
        // is now compilable (as a trap), not a compile blocker.
        assert!(eligible(&vm, method));
    }

    /// A NON-LEAF target used by the eligibility-rejection tests below: its
    /// bytecode contains a send, so `inline::is_leaf` is false and the S14
    /// step-4b leaf-inline eligibility relaxation doesn't admit it — the older
    /// mono-SMI/non-inline-primitive gates apply unchanged. (`primitive_stub`'s
    /// `^self` body IS a leaf and would now inline.)
    fn non_leaf_stub(vm: &mut VmState, sel: SymbolOop, prim_id: i64) -> MethodOop {
        let inner = vm.universe.intern(b"innerStubSel");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.send(vm, inner, 0); // a send → non-leaf
        b.ret_tos();
        let m = b.finish(vm, sel, 1, 0);
        m.set_primitive(prim_id);
        m
    }

    /// S14 (S11-deferred D1 widening): a Mono IC guarded on a NON-smi klass and
    /// targeting a non-inlinable (too-big) non-leaf body is now ELIGIBLE — it
    /// compiles as a plain compiled-IC `Call` (was `NoPermanent`). A leaf callee
    /// would inline behind a klass guard (4b); this one is too big to inline, so
    /// it stays a generic dynamic send, which S11's IC machinery handles.
    #[test]
    fn eligible_accepts_generic_mono_non_smi_send() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        let plus_sel = vm.universe.intern(b"+");
        let plus_target = non_leaf_stub(&mut vm, plus_sel, 1);
        let object_klass = vm.universe.object_klass;
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, object_klass, plus_target, epoch);
        assert!(
            eligible(&vm, method),
            "a generic mono send compiles as a Call"
        );
    }

    /// Same widening for a Mono smi-guarded send to a non-inlinable-primitive
    /// (non-leaf) target: now compiles as a generic `Call` (was `NoPermanent`).
    #[test]
    fn eligible_accepts_generic_mono_primitive_target() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        let plus_sel = vm.universe.intern(b"+");
        let not_inline = non_leaf_stub(&mut vm, plus_sel, 23);
        let smi_klass = vm.universe.smi_klass;
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, smi_klass, not_inline, epoch);
        assert!(
            eligible(&vm, method),
            "a generic mono send compiles as a Call"
        );
    }

    /// D4.6 (S11 step 6): a `super` send's own target is resolved
    /// statically at compile time, not through the interpreter's own IC
    /// lattice — so, unlike an ordinary send, its own site's IC state is
    /// irrelevant to eligibility. `set_mono_smi_plus_ic` is deliberately
    /// NOT called here (a super send would ignore it anyway): this method
    /// must be eligible on the strength of the super send alone.
    #[test]
    fn eligible_allows_super_send() {
        let mut vm = test_vm();
        let plus_sel = vm.universe.intern(b"+");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send_super(&mut vm, plus_sel, 1);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, m_sel, 1, 0);
        assert!(eligible(&vm, method));
    }

    /// S13 step 10d: compiling a `send_super` site stamps its STATIC
    /// holder-superclass onto the runtime `IcSite` (`super_klass`) — the marker
    /// that keeps a later `not_entrant_stub` re-dispatch super-aware instead of
    /// collapsing into a receiver-klass dynamic send. A method on SmallInteger
    /// doing `super +` resolves against Integer (smi's superclass), so its
    /// compiled site must carry `super_klass == Some(Integer)`.
    #[test]
    fn compiled_super_send_records_static_super_klass() {
        let mut vm = test_vm();
        let plus_sel = vm.universe.intern(b"+");
        let smi_klass = vm.universe.smi_klass;
        let integer_klass = vm.universe.integer_klass; // smi_klass.superclass()

        // The super target must resolve: install `+` on Integer (smi's super).
        let int_plus = primitive_stub(&mut vm, plus_sel, 1);
        crate::runtime::lookup::install_method(&mut vm, integer_klass, plus_sel, int_plus);

        // `superPlus: arg [ ^super + arg ]`, holder = SmallInteger.
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send_super(&mut vm, plus_sel, 1);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"superPlus:");
        let method = b.finish(&mut vm, m_sel, 1, 0);
        crate::runtime::lookup::install_method(&mut vm, smi_klass, m_sel, method);

        let id = compile_method(&mut vm, smi_klass, method).expect("super-send method compiles");
        let nm = vm.code_table.get(id).unwrap();
        assert_eq!(
            nm.ic_sites.len(),
            1,
            "the one super send is the method's only IC site"
        );
        assert_eq!(
            nm.ic_sites[0].super_klass.map(|k| k.oop().raw()),
            Some(integer_klass.oop().raw()),
            "the super site records Integer (smi's superclass) as its static super_klass"
        );
    }

    #[test]
    fn eligible_rejects_argc_over_five() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(b"sixArgs:x:x:x:x:x:");
        let method = b.finish(&mut vm, sel, 6, 0);
        assert!(!eligible(&vm, method));
    }

    /// S14 step 7-I (was `eligible_rejects_closure_op`): a `PushClosure` no
    /// longer excludes a method outright. The escape pre-pass classifies each
    /// closure site; here `[self]` is immediately `pop`ped — a DEAD closure
    /// (SPEC A7 step 2(b) "no use") is elidable, so the method is eligible and
    /// `ir::convert` elides the closure entirely (compiles to `^self`). An
    /// ESCAPING closure (stored/returned/passed) still keeps the method
    /// interpreted — covered by `eligible_rejects_escaping_closure` below and
    /// the `escape.rs` unit tests / `it_tier1` differential tests.
    #[test]
    fn eligible_allows_dead_closure() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        let lit = b.build_block(&mut vm, 0, 0, false, 0, false, |blk, _vm| {
            blk.push_self();
            blk.ret_tos();
        });
        b.push_closure(lit, 0);
        b.pop();
        b.ret_self();
        let sel = vm.universe.intern(b"withBlock");
        let method = b.finish(&mut vm, sel, 0, 0);
        assert!(
            eligible(&vm, method),
            "a dead (immediately popped) closure is elidable → eligible (7-I)"
        );
    }

    /// S24 A3a (was S14 step 7-I's `eligible_rejects_escaping_closure`): an
    /// ESCAPING closure stored into an instvar is no longer a reject — the
    /// block here (`[self]`) is non-capturing, transitively NLR-free, and
    /// non-`captures_ctx`, so A3a compiles the method with an `Ir::AllocClosure`
    /// (real closure + dead-home sentinel) instead of keeping it interpreted.
    /// The reject now applies only to escaping blocks that are NLR-bearing or
    /// `captures_ctx` (the latter deferred to A3b) — covered by
    /// `escape.rs`'s own classification tests.
    #[test]
    fn eligible_compiles_escaping_nlr_free_non_ctx_closure() {
        let mut vm = test_vm();
        let holder = vm.universe.new_klass(
            vm.universe.object_klass,
            "S24A3aEscapeUnit",
            crate::oops::Format::Slots,
            false,
            crate::oops::layout::HEADER_WORDS + 1,
        );
        let mut b = BytecodeBuilder::new();
        let lit = b.build_block(&mut vm, 0, 0, false, 0, false, |blk, _vm| {
            blk.push_self();
            blk.ret_tos();
        });
        b.push_closure(lit, 0);
        b.store_instvar_pop(0);
        b.ret_self();
        let sel = vm.universe.intern(b"stash");
        let method = b.finish(&mut vm, sel, 0, 0);
        crate::runtime::lookup::install_method(&mut vm, holder, sel, method);
        assert!(
            eligible(&vm, method),
            "S24 A3a: a stored NLR-free non-ctx closure allocates + compiles"
        );
    }

    /// S14 step 7-II-b: `has_ctx` is no longer an outright reject (a real
    /// has_ctx method's Context ELIDES when its capturing blocks inline). But
    /// this DEGENERATE method sets `has_ctx` with NO closure at all — nothing
    /// justifies the elision, so it stays interpreted (the `has_ctx && no-closure`
    /// guard). A real has_ctx-with-elidable-block method IS eligible — covered by
    /// the `it_tier1` captured-temp differential tests.
    #[test]
    fn eligible_rejects_has_ctx() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 0);
        method.set_flags(0, 0, true, false, false, false, 1); // has_ctx = true
        assert!(!eligible(&vm, method));
    }

    /// S11 step 7 (D1): `StoreInstvarPop` is now allowed — the write
    /// barrier (`ir::Ir::StoreField{barrier:true}`) and real slow-path
    /// sends this needed are exactly this step's own deliverables.
    #[test]
    fn eligible_allows_instvar_store() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.store_instvar_pop(0);
        b.ret_self();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 0);
        assert!(eligible(&vm, method));
    }

    /// S14: a polymorphic IC (more than one klass seen at a send site) is now
    /// ELIGIBLE — it compiles as a generic compiled-IC `CallSend` and the S11
    /// PIC machinery handles the polymorphism at runtime (was `NoPermanent`).
    #[test]
    fn eligible_accepts_poly_ic() {
        let mut vm = test_vm();
        let plus_sel = vm.universe.intern(b"+");
        let plus_target = primitive_stub(&mut vm, plus_sel, 1);

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 1, 0);

        let smi_klass = vm.universe.smi_klass;
        let other_klass = vm.universe.object_klass;
        let array_klass = vm.universe.array_klass;
        let pairs = crate::memory::alloc::alloc_indexable_oops(
            &mut vm,
            array_klass,
            crate::oops::layout::IC_POLY_ARRAY_LEN,
        );
        pairs.at_put(0, smi_klass.oop());
        pairs.at_put(1, plus_target.oop());
        pairs.at_put(2, other_klass.oop());
        pairs.at_put(3, plus_target.oop());
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_poly(&mut vm, pairs, epoch);

        assert!(
            eligible(&vm, method),
            "a poly send compiles as a generic Call (PIC handles it at runtime)"
        );
    }

    /// Primitive 21 (`class`) is an ordinary Ok/Fail primitive with no
    /// bespoke send-site fusion — shimmable (`is_shimmable_primitive`), so a
    /// method carrying it is now eligible. Corrected replacement for the old
    /// `eligible_rejects_primitive_method`, which asserted D1's original
    /// blanket rejection ("a method with a primitive attached stays
    /// interpreted... S10's bailout-by-restart soundness argument doesn't
    /// extend to primitives") — true when it was written (before
    /// `codecache::stubs::rt_call_primitive` existed to make a primitive-call
    /// site itself GC-safe), superseded since. NB: a *fused* primitive such
    /// as 1 (`+`) would NOT be eligible — see
    /// `eligible_rejects_already_fused_primitive_method`.
    #[test]
    fn eligible_accepts_shimmable_primitive_method() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.ret_tos();
        let sel = vm.universe.intern(b"class");
        let method = b.finish(&mut vm, sel, 0, 0);
        method.set_primitive(21);
        assert!(eligible(&vm, method));
    }

    /// S24 A1 smoke: a send-free block body compiles and RUNS via its raw
    /// entry (blocks have no entry guard: verified_entry == entry). `[:x |
    /// x]` — the identity block — proves Param wiring (closure in x0, arg
    /// in x1 -> unified temp 0) end to end.
    #[cfg(target_arch = "aarch64")] // WINVM: executes A64 stub/compiled code; x64 in Phase 3
    #[test]
    fn compiled_block_identity_runs() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"aBlock");
        let mut b = BytecodeBuilder::new();
        b.push_temp(0);
        b.block_return_tos();
        let blk = b.finish(&mut vm, sel, 1, 0);
        // set_flags(argc, ntemps, has_ctx, is_block, prim_fails, captures_ctx, nctx)
        blk.set_flags(1, 0, false, true, false, false, 0);
        assert!(blk.is_block());

        let id = compile_block(&mut vm, blk).expect("identity block must compile");
        let nm = vm.code_table.get(id).expect("installed");
        assert!(
            nm.block_method.is_some(),
            "block nmethod carries its CompiledBlock"
        );
        assert_eq!(
            vm.code_table.lookup_block(blk),
            Some(id),
            "by_block finds the Alive block nmethod"
        );

        // A minimal closure: ncopied=1 (copied[0] = home receiver).
        let closure = crate::memory::alloc::alloc_closure(&mut vm, 1);
        closure.set_method(blk);
        closure.set_copied(0, SmallInt::new(77).oop());
        closure.set_home(SmallInt::new(0)); // never read: no NlrTos in the body

        let entry = {
            let nm = vm.code_table.get(id).unwrap();
            assert_eq!(nm.verified_entry_off, 0, "no entry guard for blocks");
            nm.code.base as u64 + nm.entry_off as u64
        };
        let stubs = vm.stubs;
        let arg = SmallInt::new(41).oop();
        let out = stubs.invoke(entry, &mut vm, &[closure.oop().raw(), arg.raw()]);
        assert_eq!(out, arg.raw(), "identity block must return its argument");
    }

    /// S24 A1 smoke: `[ self ]` — push_self inside a block reads the HOME
    /// receiver, i.e. the prologue's `LoadField closure.copied[0]`, not the
    /// closure itself (the design §2.2 environment synthesis).
    #[cfg(target_arch = "aarch64")] // WINVM: executes A64 stub/compiled code; x64 in Phase 3
    #[test]
    fn compiled_block_self_is_home_receiver() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"aBlock");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.block_return_tos();
        let blk = b.finish(&mut vm, sel, 0, 0);
        blk.set_flags(0, 0, false, true, false, false, 0);

        let id = compile_block(&mut vm, blk).expect("[self] block must compile");
        let closure = crate::memory::alloc::alloc_closure(&mut vm, 1);
        closure.set_method(blk);
        let home_recv = SmallInt::new(123456).oop();
        closure.set_copied(0, home_recv);
        closure.set_home(SmallInt::new(0));

        let entry = {
            let nm = vm.code_table.get(id).unwrap();
            nm.code.base as u64 + nm.entry_off as u64
        };
        let stubs = vm.stubs;
        let out = stubs.invoke(entry, &mut vm, &[closure.oop().raw()]);
        assert_eq!(
            out,
            home_recv.raw(),
            "self inside a compiled block is copied[0], never the closure"
        );
    }

    /// S24 A1: the block eligibility gate's rejects.
    #[test]
    fn block_eligibility_rejects_v1_exclusions() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"aBlock");
        // argc 4 (> value:value:value:'s 3) -> NoPermanent.
        let mut b = BytecodeBuilder::new();
        b.push_temp(0);
        b.block_return_tos();
        let blk = b.finish(&mut vm, sel, 4, 0);
        blk.set_flags(4, 0, false, true, false, false, 0);
        assert!(compile_block(&mut vm, blk).is_none(), "argc>3 must decline");
        // ctx-temp access without captures_ctx -> NoPermanent.
        let mut b2 = BytecodeBuilder::new();
        b2.push_ctx_temp(0, 0);
        b2.block_return_tos();
        let blk2 = b2.finish(&mut vm, sel, 0, 0);
        blk2.set_flags(0, 0, false, true, false, false, 0);
        assert!(
            compile_block(&mut vm, blk2).is_none(),
            "ctx access without captures_ctx must decline"
        );
    }

    /// The already-fused dozen-plus (`PRIM_ALREADY_FUSED`) must NOT become
    /// independently compiled: doing so would leave a caller's mono IC target
    /// no longer a plain `MethodOop`, silently defeating `ir`'s more-
    /// specialized `is_smi_inlinable`/`array_op_kind`/`alloc_site_klass`
    /// fusion (and tripping `classify_smi_send`'s hard `.expect`). Primitive 1
    /// (`+`, SMI_INLINE) is the representative — a fused site compiles it
    /// inline, so the method itself is deliberately kept interpreted.
    #[test]
    fn eligible_rejects_already_fused_primitive_method() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.ret_tos();
        let sel = vm.universe.intern(b"+");
        let method = b.finish(&mut vm, sel, 1, 0);
        method.set_primitive(1);
        assert!(!eligible(&vm, method));
    }

    /// A primitive that reads `vm.prim_arg_base` (here 93 = `gcScavenge`)
    /// must stay interpreter-only: the compiled shim never sets
    /// `prim_arg_base`, so a shimmed 93/94 would return a stale operand-stack
    /// oop instead of the (relocated) receiver (`PRIM_READS_ARG_BASE`'s own
    /// doc). Even though `gcScavenge` neither fails nor is fused — it would
    /// otherwise sail through `is_shimmable_primitive` — it must be rejected.
    #[test]
    fn eligible_rejects_arg_base_reading_primitive_method() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.ret_tos();
        let sel = vm.universe.intern(b"gcScavenge");
        let method = b.finish(&mut vm, sel, 0, 0);
        method.set_primitive(93);
        assert!(!eligible(&vm, method));
    }

    /// `PRIM_ALREADY_FUSED` must stay a strict superset of [`SMI_INLINE`] —
    /// the two lists are maintained by hand and a drift (adding an id to
    /// `SMI_INLINE` without mirroring it here) would re-introduce exactly the
    /// fusion-defeating hazard this exclusion exists to prevent. Also pins the
    /// three non-SMI fused ids (`basicNew` 23, Array `at:` 26 / `at:put:` 27).
    #[test]
    fn already_fused_covers_smi_inline_and_alloc_and_array_ops() {
        for p in SMI_INLINE {
            assert!(
                PRIM_ALREADY_FUSED.contains(&p),
                "SMI_INLINE primitive {p} must also be in PRIM_ALREADY_FUSED"
            );
            assert!(
                !is_shimmable_primitive(p),
                "fused primitive {p} must not shim"
            );
        }
        for p in [23, 26, 27] {
            assert!(
                !is_shimmable_primitive(p),
                "fused primitive {p} must not shim"
            );
        }
        // The prim_arg_base readers (gcScavenge/gcFull) are excluded too.
        assert!(
            !is_shimmable_primitive(93),
            "gcScavenge reads prim_arg_base -- must not shim"
        );
        assert!(
            !is_shimmable_primitive(94),
            "gcFull reads prim_arg_base -- must not shim"
        );
        // The interpreter-reentering evaluators are excluded too: shimmed,
        // their nested run arrives from compiled code with no
        // TierLink::IntoInterpreter bracket and the first GC inside it walks
        // the tier journal mispaired (the Cocoa uiDoit-relay abort; see
        // PRIM_REENTERS_INTERPRETER's own doc).
        assert!(
            !is_shimmable_primitive(64),
            "perform:withArguments: re-enters the interpreter -- must not shim"
        );
        assert!(
            !is_shimmable_primitive(250),
            "primEvalDoit: re-enters the interpreter -- must not shim"
        );
        // A non-fused Ok/Fail primitive still shims.
        assert!(is_shimmable_primitive(21)); // class
        assert!(is_shimmable_primitive(24)); // basicNew: (NOT fused)
    }

    /// The half of the OLD blanket rejection that's still correct: a
    /// primitive capable of `PrimResult::Activated` (here, 50 = `value`,
    /// the block-activation family) can never be shimmed — the generic
    /// "call it, branch on Ok-vs-Fail" mechanism has no way to represent
    /// "control transferred to a brand new activation" (see
    /// `PRIM_ACTIVATES_FRAME`'s own doc).
    #[test]
    fn eligible_rejects_frame_activating_primitive_method() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.ret_tos();
        let sel = vm.universe.intern(b"value");
        let method = b.finish(&mut vm, sel, 0, 0);
        method.set_primitive(50);
        assert!(!eligible(&vm, method));
    }

    /// D1's `NoPermanent` path: a STRUCTURALLY ineligible method (here,
    /// argc > 5 — never becomes eligible no matter how many times it's
    /// called) gets `compile_disabled` set (and its invocation count reset
    /// to 0 as a side effect of sharing the smi) rather than being
    /// silently skipped, so the interpreter's counter-overflow trigger
    /// (S10 step 8) doesn't re-attempt it forever.
    #[test]
    fn compile_method_disables_permanently_ineligible_method() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(b"sixArgs:x:x:x:x:x:");
        let method = b.finish(&mut vm, sel, 6, 0);
        assert!(!method.compile_disabled());
        let smi_klass = vm.universe.smi_klass;
        let id = compile_method(&mut vm, smi_klass, method);
        assert!(id.is_none());
        assert!(method.compile_disabled());
        assert_eq!(
            method.counters() & crate::oops::layout::COUNTERS_INVOCATION_MASK,
            0
        );
    }

    /// S14 step 3: a method whose only inner send is a still-cold (`Empty`) IC
    /// now COMPILES — the send lowers to an uncommon trap
    /// (`SiteFeedback::Untaken` -> `Ir::UncommonTrap`), Self's lazy cold path.
    /// Before this step the cold IC returned `NoRetryLater` and `compile_method`
    /// declined with `None` (leaving the counter alone for a warmer retry); now
    /// there is nothing to defer — the trap re-executes the send interpreted
    /// when hit, warming the IC for a *later* recompile (S14 step 8). The method
    /// is neither compile-disabled nor left uncompiled.
    #[test]
    fn compile_method_compiles_cold_ic_method_as_trap() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        // Left with an empty inner IC: now compilable as a trap.
        assert!(!method.compile_disabled());
        method.set_counters(41);
        let smi_klass = vm.universe.smi_klass;
        let id = compile_method(&mut vm, smi_klass, method);
        assert!(
            id.is_some(),
            "a cold inner IC now compiles as an uncommon trap, not a compile blocker"
        );
        assert!(
            !method.compile_disabled(),
            "a successful compilation never sets the don't-compile bit"
        );
        // The nmethod carries a deopt PcDesc — the trap site itself — proving
        // the send lowered to a trap, not a real `CallSend`.
        assert!(
            !vm.code_table
                .get(id.unwrap())
                .expect("installed")
                .deopt_pcdescs
                .is_empty(),
            "the trapped send must carry at least one deopt PcDesc (the trap site)"
        );
    }

    /// `compile_method` on an eligible method installs a real, gettable
    /// `Nmethod` keyed on the receiver klass passed in. The full
    /// pipeline's actual machine code is exercised end to end (through a
    /// real `call_stub` invocation, not just "did this return `Some`") by
    /// `compiled_plus_arg_executes_correctly` in `tests/it_tier1.rs` — that
    /// needs raw FFI `unsafe`, which this crate's `#![deny(unsafe_code)]`
    /// forbids outside a handful of codegen modules, so the executing half
    /// of this same scenario lives in the separate `tests/` crate instead.
    #[test]
    fn compile_method_installs_eligible_method() {
        let mut vm = test_vm();
        let method = plus_method(&mut vm);
        set_mono_smi_plus_ic(&mut vm, method);

        let smi_klass = vm.universe.smi_klass;
        let id = compile_method(&mut vm, smi_klass, method).expect("eligible method must compile");

        let nm = vm
            .code_table
            .get(id)
            .expect("installed nmethod must be gettable");
        assert_eq!(nm.key_klass.oop().raw(), smi_klass.oop().raw());
        assert_eq!(
            vm.code_table
                .lookup(smi_klass, SymbolOop::try_from(method.selector()).unwrap()),
            Some(id)
        );
        assert!(!method.compile_disabled());
    }

    /// S13 step 3b golden: the full deopt-metadata pipeline — capture
    /// (`ir.rs::translate_instr`) → correlate + resolve + pack
    /// (`build_deopt_metadata`) → decode — on a method with a NON-EMPTY
    /// operand stack live across a safepoint, so all three `ValueLoc`
    /// dimensions (receiver, arg/temp slots, operand stack) are exercised
    /// against REAL regalloc output, not hand-faked intervals.
    ///
    /// `foo: a [ ^self box: (self bar) with: a ]`: at the inner `self bar`
    /// send, the OUTER `self` (the box:with: receiver) is still on the
    /// operand stack, and both `self` (used again as the box:with: receiver)
    /// and `a` (its second arg) are live across it — S12 spill-all forces
    /// all three to canonical frame slots, and the decoded scope must name
    /// exactly those. Both sends are generic `CallSend`s (Call,
    /// reexecute=false). Driven through decode/convert/regalloc/emit
    /// directly (not `compile_method`, whose eligibility gate rejects
    /// generic non-smi sends) so the recorder runs on this exact shape.
    #[test]
    fn deopt_scope_blob_records_real_valuelocs() {
        use crate::compiler::scopes::{decode_scope, decode_site, SafepointKind, ValueLoc};
        use ir::VReg;

        let mut vm = test_vm();
        let bar_sel = vm.universe.intern(b"bar");
        let box_sel = vm.universe.intern(b"box:with:");
        let foo_sel = vm.universe.intern(b"foo:");

        let mut b = BytecodeBuilder::new();
        b.push_self(); // box:with: receiver
        b.push_self(); // bar receiver
        b.send(&mut vm, bar_sel, 0); // (self bar)                    <- SITE 0
        b.push_temp(0); // the arg `a`
        b.send(&mut vm, box_sel, 2); // self box: (self bar) with: a  <- SITE 1
        b.ret_tos();
        let method = b.finish(&mut vm, foo_sel, 1, 0);

        // S14 step 3: warm both send sites to Mono on a NON-smi klass so they
        // stay generic `CallSend`s (this test's premise). An Empty IC would now
        // lower to an uncommon TRAP (`SiteFeedback::Untaken`), which is a
        // different scope shape exercised by the it_tier1 trap test — here we
        // want the two-CallSend safepoint layout. S14 step 4c: the targets are
        // deliberately NON-INLINABLE (each contains a SUPER send, which the
        // inliner gates out), so the inliner leaves both sites as real
        // `CallSend`s rather than splicing a body inline (a leaf accessor OR a
        // plain non-leaf now inlines — either would collapse a safepoint and
        // break the two-CallSend premise).
        let obj_klass = vm.universe.object_klass;
        let inner_sel = vm.universe.intern(b"inner");
        let bar_target = {
            let mut tb = BytecodeBuilder::new();
            tb.push_self();
            tb.send_super(&mut vm, inner_sel, 0); // a SUPER send → not inlinable
            tb.ret_tos();
            tb.finish(&mut vm, bar_sel, 0, 0)
        };
        let box_target = {
            let mut tb = BytecodeBuilder::new();
            tb.push_self();
            tb.send_super(&mut vm, inner_sel, 0); // a SUPER send → not inlinable
            tb.ret_tos();
            tb.finish(&mut vm, box_sel, 2, 0)
        };
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, obj_klass, bar_target, epoch);
        InterpreterIc::at(method, 1).set_mono(&mut vm, obj_klass, box_target, epoch);

        let cfg = decode::decode(method);
        let ir_method = ir::convert(&vm, vm.universe.smi_klass, method, &cfg, None);
        let ra = regalloc::regalloc(&ir_method);

        let mut asm = JasmAssembler::new();
        let (_blob, _pcs, _ve, _ic, safepoint_pcs, _osr_off) =
            emit::emit(&mut asm, &ir_method, &ra, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None);
        assert_eq!(
            safepoint_pcs.len(),
            2,
            "two generic sends -> two safepoints"
        );

        let (blob, pcdescs) = build_deopt_metadata(&ir_method, &ra, &safepoint_pcs);
        assert_eq!(
            pcdescs.len(),
            2,
            "two deopt sites recorded (both main CallSends)"
        );

        // Expected FrameSlot for a vreg at a position, straight from
        // regalloc's own assignment — proves `resolve_frame_loc` read the
        // real slot, not a coincidence (same technique as the resolver
        // golden in it_tier1.rs).
        let expect_slot = |vreg: VReg, pos: u32| -> ValueLoc {
            let iv = ra
                .intervals
                .iter()
                .find(|iv| iv.vreg == vreg && iv.start <= pos && iv.end > pos)
                .unwrap_or_else(|| panic!("{vreg:?} must be live across position {pos}"));
            match iv.assignment {
                Some(regalloc::Assignment::Spill(slot)) => {
                    ValueLoc::FrameSlot(-8 * (slot.0 as i32 + 1))
                }
                other => panic!(
                    "S12 spill-all: {vreg:?} must be SPILLED across a safepoint, got {other:?}"
                ),
            }
        };

        // The `self bar` safepoint is emitted first -> lower return-address
        // offset -> pcdescs[0] (pack sorts by code_off); its position is
        // `safepoint_positions[0]` in the same numbering.
        let p0 = ra.safepoint_positions[0];
        let site0 = decode_site(&blob, pcdescs[0].site_off);
        assert_eq!(site0.kind, SafepointKind::Call);
        assert!(!site0.reexecute, "a call-return site is never reexecute");
        assert_eq!(
            site0.stack,
            vec![expect_slot(VReg(0), p0)],
            "the box:with: receiver `self` is live on the operand stack across `self bar`"
        );

        let scope0 = decode_scope(&blob, site0.scope_off);
        assert_eq!(
            scope0.receiver,
            expect_slot(VReg(0), p0),
            "self live-across"
        );
        assert_eq!(
            scope0.slots,
            vec![expect_slot(VReg(1), p0)],
            "the arg `a` (VReg 1), used again at the box:with: send, is a frame slot"
        );
        // S13: a method with deopt sites interns its own compiled MethodOop
        // into the pool (`ir::convert`), and every scope record names that
        // real pool index — no longer the 3b `0` placeholder.
        let expected_ix = ir_method
            .method_pool_ix
            .expect("a method with deopt sites interned its method oop");
        assert_eq!(
            scope0.method_pool_ix, expected_ix,
            "scope names the real interned method-oop pool index"
        );
        assert_eq!(
            ir_method.pool[expected_ix as usize].value,
            method.oop().raw(),
            "that pool word holds this method's own oop"
        );
        assert_eq!(scope0.sender, None, "S13 emits depth-1 chains only");

        // The outer `self box:with:` safepoint: receiver + both args popped,
        // so nothing is left on the operand stack below it.
        let site1 = decode_site(&blob, pcdescs[1].site_off);
        assert_eq!(site1.kind, SafepointKind::Call);
        assert!(!site1.reexecute);
        assert!(
            site1.stack.is_empty(),
            "box:with: pops receiver + 2 args -> empty stack below"
        );
    }

    /// S13 step 10b: a method with a backward-jump loop produces an nmethod
    /// whose deopt metadata carries a `LoopPoll` site (`reexecute == true`) at
    /// the poll's return offset, and its recorded scope resolves the
    /// loop-carried operand-stack vregs to real FrameSlot `ValueLoc`s (via
    /// spill-all plus `deopt_live` widening), never `Nil`. Driven through the
    /// decode/convert/regalloc/emit pipeline directly, exactly like
    /// `deopt_scope_blob_records_real_valuelocs`, on a call-free loop shape.
    ///
    /// `countTo: n [ | i | i := 0. [i < n] whileTrue: [i := i + 1]. ^i ]`,
    /// hand-built as a header-test loop: a `LOOP:` block tests `i < n`
    /// (`br_false_fwd END`), the body increments `i`, and a `jump_back LOOP`
    /// closes it — that back-edge is the `Ir::Poll` this test targets.
    #[test]
    fn loop_poll_records_loop_poll_deopt_scope() {
        use crate::compiler::scopes::{decode_scope, decode_site, SafepointKind, ValueLoc};

        let mut vm = test_vm();
        let lt_sel = vm.universe.intern(b"<");
        let plus_sel = vm.universe.intern(b"+");
        let m_sel = vm.universe.intern(b"countTo:");

        // temps: t0 = n (the arg), t1 = i (the counter).
        let mut b = BytecodeBuilder::new();
        b.push_smi_i8(0);
        b.store_temp_pop(1); // i := 0
        let loop_hdr = b.new_label();
        b.bind(loop_hdr);
        b.push_temp(1); // i
        b.push_temp(0); // n
        b.send(&mut vm, lt_sel, 1); // i < n
        let end = b.new_label();
        b.br_false_fwd(end); // exit when !(i < n)
        b.push_temp(1); // i
        b.push_smi_i8(1);
        b.send(&mut vm, plus_sel, 1); // i + 1
        b.store_temp_pop(1); // i := i + 1
        b.jump_back(loop_hdr); // <- BACKWARD JUMP -> Ir::Poll
        b.bind(end);
        b.push_temp(1); // ^i
        b.ret_tos();
        let method = b.finish(&mut vm, m_sel, 1, 1);

        // S14 step 3: warm both in-loop send sites to Mono on a NON-smi klass so
        // they stay generic `CallSend`s inside the loop (this test targets the
        // back-edge `Poll` and its loop-carried scope, which needs the loop body
        // intact). An Empty IC would now lower each send to an uncommon TRAP,
        // dismantling the loop before the poll is even reached.
        let obj_klass = vm.universe.object_klass;
        let lt_target = {
            let mut tb = BytecodeBuilder::new();
            tb.ret_self();
            tb.finish(&mut vm, lt_sel, 1, 0)
        };
        let plus_target = {
            let mut tb = BytecodeBuilder::new();
            tb.ret_self();
            tb.finish(&mut vm, plus_sel, 1, 0)
        };
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, obj_klass, lt_target, epoch);
        InterpreterIc::at(method, 1).set_mono(&mut vm, obj_klass, plus_target, epoch);

        let cfg = decode::decode(method);
        let ir_method = ir::convert(&vm, vm.universe.smi_klass, method, &cfg, None);
        let ra = regalloc::regalloc(&ir_method);

        let mut asm = JasmAssembler::new();
        let (_blob, _pcs, _ve, _ic, safepoint_pcs, _osr_off) =
            emit::emit(&mut asm, &ir_method, &ra, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None);

        let (blob, pcdescs) = build_deopt_metadata(&ir_method, &ra, &safepoint_pcs);

        // Exactly one LoopPoll site among the decoded deopt sites.
        let poll_sites: Vec<_> = pcdescs
            .iter()
            .map(|d| decode_site(&blob, d.site_off))
            .filter(|s| matches!(s.kind, SafepointKind::LoopPoll))
            .collect();
        assert_eq!(
            poll_sites.len(),
            1,
            "the single loop back-edge produces exactly one LoopPoll deopt site"
        );
        let poll = &poll_sites[0];
        assert!(
            poll.reexecute,
            "a loop-poll deopt re-executes the loop condition at the header bci"
        );
        // The loop-header bci is the resume point (re-execute the condition).
        let loop_hdr_bci = cfg
            .blocks
            .iter()
            .find(|blk| blk.is_loop_header)
            .expect("the loop has a header block")
            .bci_start;
        assert_eq!(
            poll.bci as usize, loop_hdr_bci,
            "the LoopPoll resumes at the loop-header bci (re-executes the condition)"
        );

        // Every recorded stack ValueLoc (the loop-carried operand stack, empty
        // here — the back-edge block leaves nothing on the stack — but the
        // SCOPE's own slots must resolve) must be a concrete FrameSlot, never
        // Nil: spill-all + deopt_live pin the live loop-carried vregs.
        let scope = decode_scope(&blob, poll.scope_off);
        for (i, loc) in scope.slots.iter().enumerate() {
            assert!(
                matches!(loc, ValueLoc::FrameSlot(_)),
                "loop-carried slot {i} must resolve to a FrameSlot across the poll, got {loc:?}"
            );
        }
        assert!(
            matches!(scope.receiver, ValueLoc::FrameSlot(_)),
            "the receiver must resolve to a FrameSlot across the poll, got {:?}",
            scope.receiver
        );
        for loc in &poll.stack {
            assert!(
                matches!(loc, ValueLoc::FrameSlot(_)),
                "every loop-carried operand-stack value must be a FrameSlot, got {loc:?}"
            );
        }
    }
}
