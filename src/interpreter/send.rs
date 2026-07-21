//! `send`/`send_super` end-to-end (SPEC §5.3): the IC fast path, the
//! transition-table slow path (`interpreter::ic`), `activate_method`
//! (invocation counter + primitive step + frame push), and the
//! `doesNotUnderstand:` path. The only module that mutates `InterpRegs` or
//! pushes frames on a send (`sprint_s03_detail.md` §Layer boundaries).

use crate::codecache::nmethod::{NmState, NmethodId};
use crate::oops::layout::{COUNTERS_INVOCATION_MASK, COUNTERS_INVOCATION_MAX};
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::primitives::PrimResult;
use crate::runtime::vm_state::VmState;
use crate::runtime::JitMode;

use super::compiled_call::{enter_compiled, EnterResult};
use super::ic::{ic_transition, InterpreterIc};
use super::{pop, push};

/// `klass_of` (SPEC §5.3 step 3), re-exported here for the dispatch loop's
/// convenience; the canonical implementation lives in `runtime::lookup`
/// (also used by `runtime::primitives`).
pub use crate::runtime::lookup::klass_of;

/// `pub(crate)`: also bumped by `interpreter::blocks::activate_block` —
/// blocks have counters too (SPEC §5.4, S14 feedback). Returns the
/// post-bump invocation count, so `activate_method`'s S10 D4 compile
/// trigger can check for the exact threshold crossing without a second,
/// separately maintained read of the same field.
pub fn bump_invocation(m: MethodOop) -> i64 {
    let c = m.counters();
    let inv = c & COUNTERS_INVOCATION_MASK;
    let bumped = (inv + 1).min(COUNTERS_INVOCATION_MAX);
    m.set_counters((c & !COUNTERS_INVOCATION_MASK) | bumped);
    bumped
}

/// [`try_primitive`]'s own result: `Fallthrough` when there's no primitive
/// or it Failed (the bytecode body is the fallback, receiver/args left in
/// place on `vm.stack` exactly as before), `Result` when it computed a real
/// value (already popped back to `base` — nothing else needs to touch
/// `vm.stack`), `Activated` when the primitive itself already pushed a real
/// frame and set `vm.regs` (the `value`/`ensure:`/`ifCurtailed:` family) —
/// resume dispatch, don't push another frame.
pub(crate) enum PrimitiveOutcome {
    Fallthrough,
    Result(Oop),
    Activated,
    /// S24 A1: relay of a compiled block's already-run NLR unwind
    /// (`PrimResult::Nlr`) — consumers map it exactly like
    /// `SendOutcome::Nlr`, touching neither `sp` nor `vm.regs`.
    Nlr(crate::interpreter::unwind::UnwindStep),
}

/// The primitive half of SPEC §5.3 step 5 — factored out of
/// `activate_method` so [`crate::interpreter::run_method`] (a fresh
/// top-level entry, not a send-site activation: `main.rs`'s doIt, the REPL,
/// and S11 D6.1's own `rt_interpret_call`) can try a target's primitive
/// too, instead of unconditionally pushing a frame and running straight
/// into its bytecode body. `run_method`'s own `activate_entry` had no
/// primitive step of its own — harmless for its ORIGINAL callers (a
/// synthetic doIt/REPL wrapper never carries a primitive), but S11 step 7's
/// own eligibility relaxation ("arbitrary send... in ANY IC state") made
/// `rt_interpret_call` the first caller to ever reach a genuinely
/// primitive-bearing target this way (a compiled method's own generic
/// `CallSend` to a permanently-uncompilable primitive method, C2I) — every
/// such call silently skipped the primitive and ran its bytecode fallback
/// body instead (typically just the implicit "return self" a primitive-only
/// method gets when it has no other statements), always returning `self`
/// unchanged regardless of what the primitive would have computed. Found by
/// tracing a `doesNotUnderstand:` recursion that never should have started:
/// `Message>>selector`'s `self instVarAt: 1` kept returning the `Message`
/// itself instead of the real selector `Symbol`.
pub(crate) fn try_primitive(vm: &mut VmState, m: MethodOop, argc: u8) -> PrimitiveOutcome {
    let prim_id = m.primitive();
    if prim_id == 0 {
        return PrimitiveOutcome::Fallthrough;
    }
    // S20 step 4: `PRIM_ID_FFI` (-1) is a distinct sentinel, not a real
    // `PRIMITIVES` table entry — casting it `as u16` below would wrap to
    // 65535 and either miss the table entirely (a spurious panic) or, worse,
    // alias whatever real entry happens to sit at index 65535. Intercept it
    // here, before the generic `prim_by_id` lookup, and hand the whole
    // dispatch to `runtime::ffi` — the FFI descriptor (`m.literals()`, S20
    // step 3) carries its own variable-arity argument shape that the
    // table's fixed colon-counted-argc model was never built to represent.
    if prim_id == crate::oops::layout::PRIM_ID_FFI {
        return crate::runtime::ffi::dispatch_ffi_primitive(vm, m, argc);
    }

    // `m` is held across this allocation-capable step (a `can_allocate`
    // primitive) purely for the `Fail` arm's own debug_assert below —
    // S7-10 GC_STRESS audit.
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let m_h = scope.handle(vm, m);

    let desc = crate::runtime::primitives::prim_by_id(prim_id as u16)
        .unwrap_or_else(|| panic!("try_primitive: unknown primitive id {prim_id}"));
    let argc_usize = argc as usize;
    let sp = vm.stack.sp;
    let base = sp - argc_usize - 1;
    let mut buf = [vm.universe.nil_obj; 6];
    for (i, slot) in buf.iter_mut().enumerate().take(argc_usize + 1) {
        *slot = vm.stack.get(base + i);
    }
    vm.prim_arg_base = base;
    match (desc.f)(vm, &buf[..=argc_usize]) {
        PrimResult::Ok(v) => {
            vm.stack.sp = base;
            PrimitiveOutcome::Result(v)
        }
        PrimResult::Fail => {
            debug_assert!(
                m_h.get(vm).prim_fails(),
                "try_primitive: primitive {} Failed but the method's prim_fails flag is unset",
                desc.name
            );
            // Falls through: the method's bytecode body is the fallback.
            PrimitiveOutcome::Fallthrough
        }
        PrimResult::Activated => {
            // The primitive already pushed a real frame and set `vm.regs`
            // itself (the `value` family, `ensure:`, `ifCurtailed:`) —
            // resume dispatch, push no result.
            PrimitiveOutcome::Activated
        }
        // S24 A1: a compiled block's non-local return already unwound via
        // `continue_unwind` — relay the step and touch NOTHING (in
        // particular not `sp`: after an NLR the stack belongs to a
        // different activation; `base` may be past sp or inside a live
        // frame — the amended design's §2.1 discipline).
        PrimResult::Nlr(step) => PrimitiveOutcome::Nlr(step),
    }
}

/// S11 D6.3: what a send actually did, from its dispatch-loop caller's
/// point of view. `Normal` is the entire pre-step-9 contract — a result was
/// pushed (primitive short-circuit / completed compiled call) or a frame
/// was activated, and the dispatch loop discriminates the two exactly as it
/// always has (its own `fp_before` comparison). `Nlr` is new: the send's
/// COMPILED target was unwound by a non-local return and
/// `enter_compiled`'s own resumption of that unwind produced the carried
/// [`UnwindStep`] — neither a result nor an activation; the caller maps it
/// onto the dispatch loop's continuation exactly like `OP_NLR_TOS` maps its
/// own `continue_unwind` outcome (`Escaped` → return the `NLR_SENTINEL`
/// from `dispatch` so the escape keeps propagating outward;
/// `ReturnedFromHome(Some(v))` → this dispatch's entry frame returned;
/// everything else → `vm.regs` is already stamped, reload and continue).
#[derive(Clone, Copy, Debug)]
pub enum SendOutcome {
    Normal,
    Nlr(crate::interpreter::unwind::UnwindStep),
}

/// SPEC §5.3 step 5. Bumps the invocation counter (every arrival, including
/// primitive short-circuits), tries the primitive if one is attached, and
/// otherwise pushes a real frame for the bytecode body. `vm.regs` is
/// updated on the frame-push path only — a primitive success leaves it
/// exactly as the dispatch loop set it before the send (the resume point in
/// the *same*, unpushed, calling method).
///
/// `ic_site`, S10 D4: `Some(ic_idx)` when this activation came through a
/// genuine IC send site (`send_generic`'s fast or slow path) — the one
/// piece of information the compile trigger below needs in order to
/// rewrite that site to dispatch through the fresh nmethod on every later
/// call, not just this one. `None` from every other caller (`dnu`,
/// `must_be_boolean_send`, `cannot_return`): each of those reaches a
/// method via a full lookup, never an IC, so there is nothing to rewrite —
/// the compile trigger still fires for them (S10 D4 doesn't carve out an
/// exception), it just can't speed up their own dispatch next time.
///
/// Deliberately carries only `ic_idx`, never the caller `MethodOop` itself
/// (S15, found by the same klass-skew shape as BUG A): `send_generic`'s
/// slow path resolves the target through `ic_transition`, whose rows 5/6
/// (klass-skew poly transition) allocate the pairs array and can scavenge.
/// A `MethodOop` captured before that call and threaded in here would be a
/// stale from-space address by the time the compile trigger below fires —
/// `ic_transition` re-derives its own internal copy after the allocation
/// (`ic.rs`'s `caller = vm.regs.method.expect(...)`) but has no way to hand
/// that fresher value back through its `Option<MethodOop>` return. Rather
/// than plumb a second return value through, the caller method is
/// re-resolved from `vm.regs.method` at the point of use below — a scanned
/// root that still names this same caller (no frame push has happened yet
/// for this activation).
pub fn activate_method(
    vm: &mut VmState,
    m: MethodOop,
    argc: u8,
    ic_site: Option<u16>,
) -> SendOutcome {
    let bumped = bump_invocation(m);

    if let JitMode::Threshold(n) = vm.options.jit {
        // `>=`, not `==`: a compile attempt that comes back
        // `Eligibility::NoRetryLater` (driver.rs) leaves the counter
        // untouched rather than disabling the method, specifically so the
        // *next* call — now that this one has actually run the method's
        // body and warmed whatever inner IC was still cold — gets another
        // attempt. An `==` check would only ever fire once at the exact
        // crossing and then never again, silently defeating that retry.
        if bumped >= n as i64 && !m.compile_disabled() && crate::runtime::jit_compile_enabled() {
            let rcvr = vm.stack.get(vm.stack.sp - argc as usize - 1);
            let k = klass_of(vm, rcvr);
            // S14 perf recovery (THE compile-storm fix): REUSE an Alive
            // nmethod for this (klass, selector) instead of compiling a fresh
            // one on every trigger. Without this check, every stale-IC heal
            // after a recompile-on-trap retirement RE-COMPILED the method
            // (dozens of duplicate nmethods per run), eventually exhausting
            // the code cache — which silently disabled the JIT for the rest
            // of the process and pinned every benchmark at interpreter speed.
            let sel = crate::oops::wrappers::SymbolOop::try_from(m.selector())
                .expect("a method's selector is always a Symbol");
            let existing = vm.code_table.lookup(k, sel).filter(|&id| {
                vm.code_table
                    .get(id)
                    .is_some_and(|nm| matches!(nm.state, NmState::Alive))
            });
            let compiled = existing.or_else(|| crate::compiler::driver::compile_method(vm, k, m));
            if let Some(id) = compiled {
                if let Some(ic_idx) = ic_site {
                    // Re-derived fresh, not threaded through `ic_site` — see
                    // this function's doc comment for why a caller
                    // `MethodOop` captured back in `send_generic` can no
                    // longer be trusted here.
                    let caller = vm.regs.method.expect("activate_method: no active method");
                    let epoch = vm.ic_epoch;
                    // THE RICHARDS STORM FIX (2026-07-09): seed the caller's
                    // IC only when doing so LOSES no receiver-klass history —
                    // Empty (first observation) or Mono on the SAME klass (a
                    // pure interpreted→compiled target upgrade, incl. the
                    // post-recompile re-seed the S14 compile-storm fix above
                    // relies on). An UNCONDITIONAL `set_mono_compiled` here
                    // stomped a genuinely polymorphic site back to
                    // Mono(current receiver) on every over-threshold
                    // dispatch: `ic_transition` would upgrade Mono(A)→
                    // Poly[A,B], and this line immediately rewrote it to
                    // Mono(B) — so the IC ping-ponged between Mono states
                    // forever, `feedback::snapshot_profile`'s tag-only hash
                    // never changed, `recompile::note_uncommon_trap` declined
                    // every recompile as "profile unchanged", and each
                    // customized compile of the CALLER baked whichever klass
                    // had been stomped in last as a mono-inline KlassGuard —
                    // whose fail-edge UncommonTrap then fired on roughly
                    // every other call, ~160k deopts per warm Richards run
                    // (`addInput:checkPriority:`'s `oldTask priority` site,
                    // four Task subclasses; docs/PERF.md "Dual-arm branch
                    // storm" CORRECTION entry has the full diagnosis).
                    // Leaving a Poly/Mega/other-klass-Mono IC untouched costs
                    // nothing: `enter_compiled` below runs either way (the
                    // IC write is a dispatch cache, never load-bearing), and
                    // the preserved Poly tag is exactly what lets the
                    // existing recompile machinery re-lower the caller's
                    // send to DominantWithSlowPath/Call — real dispatch, no
                    // trap — which kills the storm through the front door.
                    let seed_ok = match crate::interpreter::ic::ic_state(caller, ic_idx) {
                        crate::interpreter::ic::IcState::Empty => true,
                        crate::interpreter::ic::IcState::Mono => {
                            InterpreterIc::at(caller, ic_idx).guard().raw() == k.oop().raw()
                        }
                        crate::interpreter::ic::IcState::Poly(_)
                        | crate::interpreter::ic::IcState::Mega => false,
                    };
                    if seed_ok {
                        InterpreterIc::at(caller, ic_idx).set_mono_compiled(vm, k, id, epoch);
                    }
                }
                match enter_compiled(vm, id, argc) {
                    EnterResult::Completed => return SendOutcome::Normal,
                    EnterResult::Nlr(step) => return SendOutcome::Nlr(step),
                    EnterResult::Bailout => {
                        // D1: sound to just fall through to the normal
                        // interpreted body below, for this one call — no
                        // observable effect could have preceded a bailout.
                    }
                }
            }
        }
    }

    match try_primitive(vm, m, argc) {
        PrimitiveOutcome::Result(v) => {
            push(vm, v);
            return SendOutcome::Normal;
        }
        PrimitiveOutcome::Activated => return SendOutcome::Normal,
        // S24 A1: identical relay to EnterResult::Nlr above — the unwind
        // already ran; the caller's own SendOutcome::Nlr handling (S11
        // D6.3) takes it from here.
        PrimitiveOutcome::Nlr(step) => return SendOutcome::Nlr(step),
        PrimitiveOutcome::Fallthrough => {}
    }

    // `m` is held across an allocation-capable step below (`maybe_alloc_
    // context`, after which `m` is stamped into `vm.regs.method`) — S7-10
    // GC_STRESS audit.
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let m_h = scope.handle(vm, m);

    let saved_fp = vm.stack.fp as i64;
    let saved_bci = vm.regs.bci;
    let m = m_h.get(vm); // fresh: the failed primitive above may have allocated
    super::push_frame(vm, m, argc as usize, saved_fp, saved_bci);
    // A plain method's own Context has no enclosing chain (SPEC §5.4);
    // block activation (`activate_block`) passes its inherited context
    // instead.
    let nil = vm.universe.nil_obj;
    super::blocks::maybe_alloc_context(vm, m, vm.stack.fp, nil);
    vm.regs.method = Some(m_h.get(vm)); // fresh again: the context alloc may have moved it
    vm.regs.bci = 0;
    SendOutcome::Normal
}

/// Steps 1–4 of SPEC §5.3's `send` algorithm plus the transition-table
/// dispatch (step 5–7). `argc` must equal the site's own IC-recorded argc
/// (the dispatch loop reads both from the same place, so this is always
/// true — the assert documents the invariant rather than guarding against a
/// real divergence).
pub fn send_generic(vm: &mut VmState, argc: u8, ic_idx: u16, is_super: bool) -> SendOutcome {
    let caller = vm.regs.method.expect("send_generic: no active method");
    let ic = InterpreterIc::at(caller, ic_idx);
    debug_assert_eq!(argc, ic.argc(), "send_generic: argc/IC mismatch");

    let rcvr = vm.stack.get(vm.stack.sp - argc as usize - 1);
    let k = klass_of(vm, rcvr);

    // Fast path: mono guard, current epoch.
    if ic.guard().raw() == k.oop().raw() && ic.epoch() == vm.ic_epoch {
        // S10 D4: a smi target means mono-compiled (`set_mono_compiled`),
        // not the ordinary mono-interpreted `MethodOop` target below.
        if let Some(target_smi) = SmallInt::try_from(ic.target()) {
            let nm_id = NmethodId(target_smi.value() as u32);
            // Stale-id self-heal (D4): a freed/reused id can never
            // dispatch wrongly, since S12 flushing is checked against here
            // by identity (klass/selector), not just presence in the
            // table. S10 never frees a slot, so this is always true today
            // — implemented now so S12 doesn't have to touch this site.
            let valid = vm.code_table.get(nm_id).is_some_and(|nm| {
                matches!(nm.state, NmState::Alive)
                    && nm.key_klass.oop().raw() == ic.guard().raw()
                    && nm.key_selector.oop().raw() == ic.selector().oop().raw()
            });
            if valid {
                match enter_compiled(vm, nm_id, argc) {
                    EnterResult::Completed => return SendOutcome::Normal,
                    EnterResult::Nlr(step) => return SendOutcome::Nlr(step),
                    EnterResult::Bailout => {
                        // D1: this one call falls back to the interpreter;
                        // the compiled entry is still valid for the next
                        // one, so the IC itself is left untouched (no
                        // `ic_site` — nothing to heal or rewrite).
                        let sel = ic.selector();
                        let m = crate::runtime::lookup::lookup(vm, k, sel).expect(
                            "send_generic: bailout method must still resolve \
                             (it did when its compiled entry was installed)",
                        );
                        return activate_method(vm, m, argc, None);
                    }
                }
            }
            // Stale id: fall through to ic_transition below to self-heal,
            // same as an ordinary guard/epoch miss.
        } else {
            let m = MethodOop::try_from(ic.target())
                .expect("send_generic: mono target is not a CompiledMethod");
            return activate_method(vm, m, argc, Some(ic_idx));
        }
    }

    match ic_transition(vm, caller, ic_idx, k, is_super) {
        // Only `ic_idx` crosses into `activate_method` — NOT this `caller`:
        // `ic_transition`'s klass-skew rows (5/6) can have allocated the
        // poly-pairs array, so this outer binding (captured above, before
        // that call) may already be a stale from-space address. See
        // `activate_method`'s doc comment for the full story (S15, same
        // shape as BUG A).
        Some(m) => activate_method(vm, m, argc, Some(ic_idx)),
        None => {
            // Re-derive everything: `ic_transition` may have allocated (the
            // poly-pairs array), so the cached `ic` view (its `ics` array
            // address), `k`, and even `caller` may all be stale. `vm.regs.
            // method` is a scanned GC root and still names this caller;
            // the receiver is re-read from its (scanned) stack slot.
            let caller = vm.regs.method.expect("send_generic: no active method");
            let ic = InterpreterIc::at(caller, ic_idx);
            let rcvr = vm.stack.get(vm.stack.sp - argc as usize - 1);
            let k = klass_of(vm, rcvr);
            dnu(vm, k, ic.selector(), argc)
        }
    }
}

pub fn send_super_generic(vm: &mut VmState, argc: u8, ic_idx: u16) -> SendOutcome {
    send_generic(vm, argc, ic_idx, true)
}

/// SPEC §5.3 step 6: `#doesNotUnderstand:` resolution. Builds the `Message`
/// (array parked on the operand stack across the allocation — the S7
/// choke-point pattern), rewrites the send site's args down to the single
/// `Message` argument, and re-sends through a full lookup (never through
/// the original site's IC — that IC's state describes the *original*
/// selector, not `#doesNotUnderstand:`).
fn dnu(vm: &mut VmState, rcvr_klass: KlassOop, selector: SymbolOop, argc: u8) -> SendOutcome {
    crate::runtime::error::trace_dnu(vm, "interpreted", rcvr_klass, selector);
    let argc_usize = argc as usize;
    let sp = vm.stack.sp;
    let args_start = sp - argc_usize;

    // The args array is parked on the operand stack across the Message
    // allocation (the S3-era choke-point pattern), but `selector` and
    // `rcvr_klass` are bare parameters held across BOTH allocations below
    // and used after — they need handles (S7-10 GC_STRESS audit).
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let selector_h = scope.handle(vm, selector);
    let rcvr_klass_h = scope.handle(vm, rcvr_klass);

    let array_klass = vm.universe.array_klass;
    let args_array = crate::memory::alloc::alloc_indexable_oops(vm, array_klass, argc_usize);
    for i in 0..argc_usize {
        args_array.at_put(i, vm.stack.get(args_start + i));
    }

    // Drop the raw args (receiver stays), park the array as a GC root
    // before allocating the Message.
    vm.stack.sp = args_start;
    push(vm, args_array.oop());

    let message_klass = vm.universe.message_klass;
    let msg = crate::memory::alloc::alloc_slots(vm, message_klass);
    let parked_array = pop(vm);
    msg.set_body_oop(0, selector_h.get(vm).oop());
    msg.set_body_oop(1, parked_array);
    push(vm, msg.oop()); // stack: ..., receiver, message

    let sel_dnu = vm.universe.sel_does_not_understand;
    match crate::runtime::lookup::lookup(vm, rcvr_klass_h.get(vm), sel_dnu) {
        Some(dnu_method) => activate_method(vm, dnu_method, 1, None),
        None => crate::runtime::error::dnu_fallback(vm, selector_h.get(vm), rcvr_klass_h.get(vm)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeBuilder;
    use crate::oops::layout::{COUNTERS_INVOCATION_MASK, COUNTERS_INVOCATION_MAX, HEADER_WORDS};
    use crate::oops::smi::SmallInt;
    use crate::oops::Oop;
    use crate::runtime::lookup::install_method;
    use crate::runtime::vm_state::{VmOptions, VmState};

    fn test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        })
    }

    fn new_klass(vm: &mut VmState, name: &str) -> KlassOop {
        let object_klass = vm.universe.object_klass;
        vm.universe.new_klass(
            object_klass,
            name,
            crate::oops::Format::Slots,
            false,
            HEADER_WORDS,
        )
    }

    /// A caller with one send site: `push_temp(0..=argc)`, `send(sel,
    /// argc)`, `ret_tos` — see `interpreter::ic`'s tests for the same shape.
    fn build_caller(vm: &mut VmState, sel: SymbolOop, argc: u8) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        for t in 0..=argc {
            b.push_temp(t);
        }
        b.send(vm, sel, argc);
        b.ret_tos();
        let name = vm.universe.intern(b"caller");
        b.finish(vm, name, (argc + 1) as usize, 0)
    }

    fn send_once(vm: &mut VmState, caller: MethodOop, args: &[Oop]) -> Oop {
        let dummy_self = vm.universe.nil_obj;
        crate::interpreter::run_method(vm, caller, dummy_self, args)
    }

    #[test]
    fn activate_counter_bump() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let name = vm.universe.intern(b"foo");
        let m = b.finish(&mut vm, name, 0, 0);
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();

        for i in 1..=5i64 {
            let _ = send_once(&mut vm, caller, &[recv]);
            assert_eq!(m.counters() & COUNTERS_INVOCATION_MASK, i);
        }
    }

    #[test]
    fn counter_saturates() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let name = vm.universe.intern(b"foo");
        let m = b.finish(&mut vm, name, 0, 0);
        m.set_counters(COUNTERS_INVOCATION_MAX);
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();

        let _ = send_once(&mut vm, caller, &[recv]);
        assert_eq!(
            m.counters() & COUNTERS_INVOCATION_MASK,
            COUNTERS_INVOCATION_MAX
        );
    }

    /// A minimal but functionally real `SmallInteger>>+`: primitive 1
    /// (`prim_add`), `prim_fails=true` (required — `activate_method`'s own
    /// `debug_assert!` on a `Fail`ed primitive whose method doesn't declare
    /// it can fail), fallback body returns a fixed sentinel rather than a
    /// real LargeInteger promotion — a bare, `.mst`-free `test_vm()` has no
    /// real `SmallInteger>>+` installed at all (that lives in
    /// `world/06_smallinteger.mst`, loaded only by the real VM/world
    /// tests), and real bignum promotion isn't needed to prove this
    /// module's own bailout-fallback plumbing works.
    const OVERFLOW_SENTINEL: i64 = -1;
    fn install_smi_plus(vm: &mut VmState) -> SymbolOop {
        let smi_klass = vm.universe.smi_klass;
        let plus_sel = vm.universe.intern(b"+");
        let mut b = BytecodeBuilder::new();
        b.push_smi_i8(OVERFLOW_SENTINEL as i8);
        b.ret_tos();
        let m = b.finish(vm, plus_sel, 1, 0);
        m.set_primitive(1);
        m.set_flags(1, 0, false, false, true, false, 0); // prim_fails = true
        install_method(vm, smi_klass, plus_sel, m);
        plus_sel
    }

    /// S10 D4 end to end, through the real interpreter dispatch path (not
    /// a direct `driver::compile_method` call): a smi-eligible method
    /// (`self + arg`, defined on `SmallInteger`) warms up its own internal
    /// `#+` IC across a couple of ordinary interpreted calls, its
    /// invocation counter crosses `JitMode::Threshold`, `activate_method`'s
    /// trigger fires, and the CALLER's send-site IC gets rewritten to a
    /// compiled (smi) target — proven by reading it back, not just by the
    /// result staying correct (which an untouched interpreted IC would
    /// also produce).
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn compile_trigger_fires_and_rewrites_ic_to_compiled() {
        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(3),
        });
        let smi_klass = vm.universe.smi_klass;
        let plus_sel = install_smi_plus(&mut vm);
        let plus_arg_sel = vm.universe.intern(b"plusArg:");

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let plus_arg_method = b.finish(&mut vm, plus_arg_sel, 1, 0);
        install_method(&mut vm, smi_klass, plus_arg_sel, plus_arg_method);

        let caller = build_caller(&mut vm, plus_arg_sel, 1);

        for i in 1..=2i64 {
            let result = send_once(
                &mut vm,
                caller,
                &[SmallInt::new(5).oop(), SmallInt::new(i).oop()],
            );
            assert_eq!(result, SmallInt::new(5 + i).oop());
            assert_eq!(plus_arg_method.counters() & COUNTERS_INVOCATION_MASK, i);
            let ic = InterpreterIc::at(caller, 0);
            assert!(
                MethodOop::try_from(ic.target()).is_some(),
                "below threshold: caller's IC must still target the interpreted method"
            );
        }

        // 3rd call crosses the threshold.
        let result = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(5).oop(), SmallInt::new(3).oop()],
        );
        assert_eq!(result, SmallInt::new(8).oop());
        assert_eq!(plus_arg_method.counters() & COUNTERS_INVOCATION_MASK, 3);

        let ic = InterpreterIc::at(caller, 0);
        let nm_id = SmallInt::try_from(ic.target())
            .expect("at threshold: caller's IC must now target a compiled nmethod id");
        let nm = vm
            .code_table
            .get(NmethodId(nm_id.value() as u32))
            .expect("the id the IC was rewritten to must be installed");
        assert_eq!(nm.key_klass.oop().raw(), smi_klass.oop().raw());

        // 4th call: dispatches through the compiled fast path.
        let result4 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(100).oop(), SmallInt::new(23).oop()],
        );
        assert_eq!(result4, SmallInt::new(123).oop());
    }

    /// As above, but exercises overflow/bailout from both directions: the
    /// very call that triggers compilation itself overflows (`activate_method`'s
    /// own fallback-to-interpreter path), and — once the IC is rewritten —
    /// a *later* call also overflows (`send_generic`'s own fast-path
    /// bailout fallback). Both must reach `install_smi_plus`'s interpreted
    /// fallback (the [`OVERFLOW_SENTINEL`]) without disturbing the
    /// still-valid compiled IC entry.
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn compile_trigger_bailout_falls_back_correctly() {
        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(2),
        });
        let smi_klass = vm.universe.smi_klass;
        let plus_sel = install_smi_plus(&mut vm);
        let plus_arg_sel = vm.universe.intern(b"plusArg:");

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let plus_arg_method = b.finish(&mut vm, plus_arg_sel, 1, 0);
        install_method(&mut vm, smi_klass, plus_arg_sel, plus_arg_method);
        let caller = build_caller(&mut vm, plus_arg_sel, 1);

        // Call 1 (bumped=1, below threshold=2): plain interpreted run,
        // whose only real job is warming plusArg:'s own internal `#+` IC
        // to mono/smi so call 2's `eligible()` check has something to see
        // — a method's inner sends are cold until its body has actually
        // run at least once (this is what the earlier, threshold=1 draft
        // of this test got wrong: the triggering call and the warming
        // call cannot be the same call for any method with an inner send).
        let r1 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(1).oop(), SmallInt::new(1).oop()],
        );
        assert_eq!(r1, SmallInt::new(2).oop());

        // Call 2 (bumped=2 == threshold): triggers compile_method (now
        // eligible — the inner IC warmed by call 1), rewrites the IC, and
        // immediately runs THIS call's own (overflowing) args through the
        // freshly compiled entry — bailing out to `activate_method`'s own
        // interpreted fallback.
        let big = SmallInt::new(SmallInt::MAX);
        let r2 = send_once(&mut vm, caller, &[big.oop(), big.oop()]);
        assert_eq!(
            r2,
            SmallInt::new(OVERFLOW_SENTINEL).oop(),
            "an overflowing add at the exact trigger call must still reach the interpreted \
             fallback body via activate_method's own bailout handling"
        );

        let ic = InterpreterIc::at(caller, 0);
        let nm_id = SmallInt::try_from(ic.target())
            .expect("call 2 must have rewritten the IC to a compiled nmethod id");

        // Call 3, small operands: exercises send_generic's fast smi-IC
        // dispatch (EnterResult::Completed) — confirms the compiled entry
        // survived call 2's bailout and works correctly on its own.
        let r3 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(10).oop(), SmallInt::new(32).oop()],
        );
        assert_eq!(r3, SmallInt::new(42).oop());

        // Call 4, overflowing again: exercises send_generic's OWN
        // bailout fallback (the already-rewritten-smi-IC path, distinct
        // from call 2's activate_method-triggered one).
        let r4 = send_once(&mut vm, caller, &[big.oop(), big.oop()]);
        assert_eq!(r4, SmallInt::new(OVERFLOW_SENTINEL).oop());

        // Neither bailout may have healed/demoted the site: the IC must
        // still name the exact same compiled entry throughout.
        let ic_after = InterpreterIc::at(caller, 0);
        let nm_id_after = SmallInt::try_from(ic_after.target())
            .expect("bailouts must not disturb the IC -- the compiled entry is still valid");
        assert_eq!(nm_id_after.value(), nm_id.value());
        assert_eq!(
            vm.code_table
                .get(NmethodId(nm_id.value() as u32))
                .unwrap()
                .key_klass
                .oop()
                .raw(),
            smi_klass.oop().raw()
        );
    }

    /// tests_s10.md's `bailout_falls_back_correctly`, the OTHER half not
    /// covered by `compile_trigger_bailout_falls_back_correctly` above: a
    /// receiver of a klass the compiled entry was never installed for.
    /// `send_generic`'s own guard check (`ic.guard().raw() == k.oop().raw()`,
    /// this file's fast path) must reject the mismatch BEFORE ever calling
    /// `enter_compiled` — a Double's bits are a heap pointer, not a tagged
    /// smi, so blindly running the smi entry's own tag-checked add on it
    /// would be a memory-safety bug, not just a wrong answer.
    ///
    /// This deliberately does NOT then assert a following smi call still
    /// hits the compiled entry: SPEC's inline-cache lattice demotes a
    /// second-klass sighting to poly, and `ic_transition`'s own poly path
    /// (`ic.rs`) always re-derives an *interpreted* target — "the poly
    /// array only ever stores interpreted MethodOops" is that function's
    /// own documented design, not a bug this test should fight. The
    /// still-uses-the-nmethod guarantee tests_s10.md describes belongs to
    /// the overflow/bailout scenario above, where the IC never leaves mono.
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn compile_trigger_double_receiver_is_ic_miss_not_compiled_entry() {
        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(2),
        });
        let smi_klass = vm.universe.smi_klass;
        let double_klass = vm.universe.double_klass;
        let plus_sel = install_smi_plus(&mut vm);
        let plus_arg_sel = vm.universe.intern(b"plusArg:");

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let plus_arg_method = b.finish(&mut vm, plus_arg_sel, 1, 0);
        install_method(&mut vm, smi_klass, plus_arg_sel, plus_arg_method);

        // A second, completely distinct `plusArg:` on Double — its return
        // value could never be confused with the smi version's, or with a
        // misinterpreted compiled-entry crash/garbage result.
        const DOUBLE_SENTINEL: i64 = -77;
        let mut db = BytecodeBuilder::new();
        db.push_smi_i8(DOUBLE_SENTINEL as i8);
        db.ret_tos();
        let double_plus_arg_method = db.finish(&mut vm, plus_arg_sel, 1, 0);
        install_method(&mut vm, double_klass, plus_arg_sel, double_plus_arg_method);

        let caller = build_caller(&mut vm, plus_arg_sel, 1);

        // Calls 1-2: warm plusArg:'s own inner `#+` IC, then cross the
        // threshold — same shape as compile_trigger_fires_and_rewrites_ic_to_compiled.
        let r1 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(1).oop(), SmallInt::new(1).oop()],
        );
        assert_eq!(r1, SmallInt::new(2).oop());
        let r2 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(5).oop(), SmallInt::new(3).oop()],
        );
        assert_eq!(r2, SmallInt::new(8).oop());

        let ic = InterpreterIc::at(caller, 0);
        SmallInt::try_from(ic.target())
            .expect("call 2 must have rewritten the IC to a compiled nmethod id");
        assert_eq!(
            ic.guard().raw(),
            smi_klass.oop().raw(),
            "mono-compiled still stores a plain klass guard, same as mono-interpreted"
        );

        // Call 3: a Double receiver. Guard mismatch (double_klass !=
        // smi_klass) must route through ic_transition's normal lookup +
        // interpreted activate_method, landing on double_plus_arg_method —
        // never anywhere near enter_compiled (which would read the Double's
        // heap pointer bits as if they were a tagged smi operand).
        let d = crate::memory::alloc::alloc_double(&mut vm, 3.5);
        let r3 = send_once(&mut vm, caller, &[d.oop(), SmallInt::new(999).oop()]);
        assert_eq!(
            r3,
            SmallInt::new(DOUBLE_SENTINEL).oop(),
            "a Double receiver must dispatch to Double's own plusArg: via the IC-miss path"
        );

        // Call 4: back to a smi receiver. Whatever the IC now looks like
        // (mono-compiled still, or poly-demoted), the call must still
        // produce the right VALUE — resolving correctly is the property
        // that matters, not which of the two valid mechanisms got there.
        let r4 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(10).oop(), SmallInt::new(32).oop()],
        );
        assert_eq!(r4, SmallInt::new(42).oop());
    }

    /// tests_s10.md's `stale_id_self_heals`: a compiled id an IC still
    /// names can stop being valid (S12 flushing will do this for real —
    /// S10 itself never frees a slot organically, hence the test-only
    /// `CodeTable::test_clear_slot` hook standing in for it here).
    /// `send_generic`'s own re-validate-before-`enter_compiled` check
    /// (`code_table.get(nm_id).is_some_and(...)`) must catch the gap and
    /// fall through to `ic_transition`'s ordinary self-heal — re-resolving
    /// and re-installing a plain interpreted target — rather than calling
    /// `enter_compiled` on a dangling id.
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn stale_id_self_heals() {
        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(2),
        });
        let smi_klass = vm.universe.smi_klass;
        let plus_sel = install_smi_plus(&mut vm);
        let plus_arg_sel = vm.universe.intern(b"plusArg:");

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let plus_arg_method = b.finish(&mut vm, plus_arg_sel, 1, 0);
        install_method(&mut vm, smi_klass, plus_arg_sel, plus_arg_method);
        let caller = build_caller(&mut vm, plus_arg_sel, 1);

        // Calls 1-2: warm + cross the threshold, exactly like
        // compile_trigger_fires_and_rewrites_ic_to_compiled.
        let r1 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(1).oop(), SmallInt::new(1).oop()],
        );
        assert_eq!(r1, SmallInt::new(2).oop());
        let r2 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(5).oop(), SmallInt::new(3).oop()],
        );
        assert_eq!(r2, SmallInt::new(8).oop());

        let ic = InterpreterIc::at(caller, 0);
        let nm_id = NmethodId(
            SmallInt::try_from(ic.target())
                .expect("call 2 must have rewritten the IC to a compiled nmethod id")
                .value() as u32,
        );
        assert!(vm.code_table.get(nm_id).is_some());

        // Call 3: dispatch once through the still-valid compiled entry.
        let r3 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(10).oop(), SmallInt::new(1).oop()],
        );
        assert_eq!(r3, SmallInt::new(11).oop());

        // Forcibly clear the slot -- simulating S12 flushing reclaiming
        // it, with the IC still (stalely) naming this exact id.
        vm.code_table.test_clear_slot(nm_id);
        assert!(vm.code_table.get(nm_id).is_none());

        // Call 4: the IC's guard/epoch still match (still smi_klass,
        // still the same epoch), so send_generic takes its fast-path
        // branch and reaches the smi-target arm -- but the code_table
        // re-check now fails, so it falls through to ic_transition's
        // self-heal: resolve() re-finds plusArg:'s interpreted method and
        // ic.set_mono installs it as a plain interpreted target, and
        // send_generic then runs it through activate_method like any
        // other mono-interpreted dispatch. THAT is where this particular
        // setup gets interesting: plusArg:'s own invocation counter is
        // bumped a 3rd time right there (calls 1-2 bumped it to 2; call 3
        // went straight through enter_compiled and never touched
        // activate_method at all), crosses JitMode::Threshold(2) again,
        // and activate_method's own trigger recompiles and reinstalls a
        // FRESH compiled target on the spot -- so by the time call 4
        // returns the site is mono-compiled again, not stuck
        // interpreted. That chain reaction (stale id -> self-heal ->
        // immediate re-trigger -> fresh compiled reinstall, all inside
        // one send) is exactly what tests_s10.md's "reinstalls a
        // CompiledMethod target" describes, and is the real thing worth
        // checking here: that it resolves cleanly rather than reusing the
        // freed slot incorrectly or looping.
        let r4 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(20).oop(), SmallInt::new(4).oop()],
        );
        assert_eq!(
            r4,
            SmallInt::new(24).oop(),
            "a stale compiled id must self-heal to a correct dispatch, not misbehave"
        );

        let ic_after = InterpreterIc::at(caller, 0);
        let new_id = NmethodId(
            SmallInt::try_from(ic_after.target())
                .expect("self-heal's own re-trigger must have reinstalled a fresh compiled target")
                .value() as u32,
        );
        let new_nm = vm
            .code_table
            .get(new_id)
            .expect("the reinstalled id must be installed and alive");
        assert_eq!(new_nm.key_klass.oop().raw(), smi_klass.oop().raw());
        assert_eq!(new_nm.key_selector.oop().raw(), plus_arg_sel.oop().raw());

        // Call 5: dispatches through the freshly reinstalled compiled
        // entry, confirming it works normally afterward.
        let r5 = send_once(
            &mut vm,
            caller,
            &[SmallInt::new(1).oop(), SmallInt::new(2).oop()],
        );
        assert_eq!(r5, SmallInt::new(3).oop());
    }

    #[test]
    fn counter_bumped_on_prim_success() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"identityHash");
        let a = new_klass(&mut vm, "A");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.ret_tos();
        let name = vm.universe.intern(b"identityHash");
        let m = b.finish(&mut vm, name, 0, 0);
        m.set_primitive(20); // oops-group `identityHash`, never Fails
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();

        let _ = send_once(&mut vm, caller, &[recv]);
        assert_eq!(
            m.counters() & COUNTERS_INVOCATION_MASK,
            1,
            "counter must bump even on a primitive short-circuit"
        );
    }

    #[test]
    fn prim_fallback_runs_bytecode_body() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let mut b = BytecodeBuilder::new();
        b.push_smi_i8(7); // the fallback body's own return value
        b.ret_tos();
        let name = vm.universe.intern(b"foo");
        let m = b.finish(&mut vm, name, 1, 0); // argc=1: smi `+`'s shape
        m.set_primitive(1); // smi group `+`
        m.set_flags(1, 0, false, false, true, false, 0); // prim_fails = true
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 1);
        // The receiver is not a smi, so primitive 1 (`+`) always Fails.
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let arg = SmallInt::new(1).oop();

        let result = send_once(&mut vm, caller, &[recv, arg]);
        assert_eq!(
            result,
            SmallInt::new(7).oop(),
            "a Failed primitive must fall through to the bytecode body"
        );
    }

    #[test]
    fn stack_receiver_index() {
        let mut vm = test_vm();
        let recv = SmallInt::new(1).oop();
        let arg0 = SmallInt::new(2).oop();
        let arg1 = SmallInt::new(3).oop();
        vm.stack.push(recv);
        vm.stack.push(arg0);
        vm.stack.push(arg1);

        // The exact arithmetic `send_generic` uses to locate the receiver
        // (SPEC §5.3's pinned `sp - argc - 1` convention).
        let argc = 2usize;
        let receiver_index = vm.stack.sp - argc - 1;
        assert_eq!(vm.stack.get(receiver_index), recv);
    }

    /// The primitive call site's 3rd arm (`PrimResult::Activated`, S4) must
    /// push NO result — an off-by-one there would corrupt the new (block)
    /// frame's first temp slot with a stray value instead of leaving it
    /// `nil`.
    #[test]
    fn activated_no_result_push() {
        let mut vm = test_vm();
        let closure_klass = vm.universe.closure_klass;
        let value_sel = vm.universe.intern(b"value");
        let mut value_body = BytecodeBuilder::new();
        value_body.push_self();
        value_body.ret_self(); // unreachable: the primitive never Fails here
        let value_method = value_body.finish(&mut vm, value_sel, 0, 0);
        value_method.set_primitive(50);
        install_method(&mut vm, closure_klass, value_sel, value_method);

        let mut home = BytecodeBuilder::new();
        // The block itself has 1 local temp, never explicitly stored to —
        // if `Activated` handling pushed a stray result, it would land
        // exactly here.
        let lit = home.build_block(&mut vm, 0, 1, false, 0, false, |b, _vm| {
            b.push_temp(0);
            b.ret_tos();
        });
        home.push_closure(lit, 0);
        home.send(&mut vm, value_sel, 0);
        home.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = home.finish(&mut vm, sel, 0, 0);

        let recv = vm.universe.nil_obj;
        let result = crate::interpreter::run_method(&mut vm, method, recv, &[]);
        assert_eq!(
            result, recv,
            "the block's untouched temp[0] must be nil, not a stray pushed result"
        );
    }
}
