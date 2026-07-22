//! S15 — on-stack replacement: the interpreter→compiled transition for a
//! RUNNING loop (`sprint_s15_detail.md` A3/A4). The interpreter's
//! `jump_back` slow path calls [`rt_osr_request`] on loop-counter overflow;
//! on success the interpreter frame has been REPLACED, in place, by a
//! compiled activation that ran the rest of the loop (and the rest of the
//! method) to completion — the caller performs a standard method-return
//! with the linkage captured here. Every decline path is safe: the counter
//! is reset (A4 — in EVERY outcome, or a capped loop re-triggers a failed
//! compile on every backedge) and interpretation simply continues.

use crate::codecache::nmethod::{NmState, NmethodId};
use crate::interpreter::stack::Frame;
use crate::oops::layout::{
    COUNTERS_LOOP_MASK, COUNTERS_LOOP_SHIFT, ENTRY_FRAME_SENTINEL, FRAME_TEMPS_BASE,
    LOOP_COUNTER_LIMIT, NLR_SENTINEL,
};
use crate::oops::wrappers::{MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::lookup::{klass_of, lookup};
use crate::runtime::vm_state::{JitMode, TierLink, VmState};

/// What `jump_back`'s slow path does with the request's result.
pub enum OsrOutcome {
    /// Compiled execution ran the WHOLE remaining activation; the standard
    /// method-return has ALREADY been performed with the replaced frame's
    /// saved linkage: the result sits in the caller's receiver slot,
    /// `vm.regs` names the resume point. `entry_frame` is true when the
    /// replaced frame WAS the dispatch loop's entry frame (sentinel saved
    /// fp) — the dispatch loop returns `result` outright, exactly like
    /// `OP_RETURN_TOS`'s `Some(result)` arm.
    Completed { result: Oop, entry_frame: bool },
    /// An NLR escaped through the compiled replacement — resumed home-ward
    /// exactly like `enter_compiled`; the caller maps the step like
    /// `OP_SEND` does.
    Nlr(crate::interpreter::unwind::UnwindStep),
    /// No transition (not compilable / outside the v1 envelope / JIT off):
    /// loop counter reset; keep interpreting at the header.
    Declined,
}

/// Bump `method`'s loop counter (bits 16-31 of `counters`); true on
/// crossing [`LOOP_COUNTER_LIMIT`]. Saturating like the invocation half.
pub fn bump_loop_counter(m: MethodOop) -> bool {
    let c = m.counters();
    let cur = (c & COUNTERS_LOOP_MASK) >> COUNTERS_LOOP_SHIFT;
    let bumped = (cur + 1).min(0xFFFF);
    m.set_counters((c & !COUNTERS_LOOP_MASK) | (bumped << COUNTERS_LOOP_SHIFT));
    bumped >= LOOP_COUNTER_LIMIT
}

pub fn reset_loop_counter(m: MethodOop) {
    let c = m.counters();
    m.set_counters(c & !COUNTERS_LOOP_MASK);
}

/// A3: offer the RUNNING interpreter frame `fp`, about to re-enter the loop
/// header at `target_bci`, for on-stack replacement.
pub fn rt_osr_request(vm: &mut VmState, fp: usize, target_bci: u16) -> OsrOutcome {
    let frame = Frame { fp };
    let method = frame.method(&vm.stack);

    // A4: reset FIRST — every outcome, including the panics-averted-by-
    // decline ones below, must space re-triggers by a full LoopCounterLimit.
    reset_loop_counter(method);

    if !matches!(vm.options.jit, JitMode::Threshold(_)) {
        return OsrOutcome::Declined;
    }
    // The Debug-menu Compiler (JIT) kill-switch gates OSR too: "off" promises
    // that anything not already compiled runs interpreted, and an OSR compile
    // is exactly the not-already-compiled case. Declining here is the safe
    // outcome by construction — the counter was reset above, so re-triggers
    // stay spaced, and the loop simply continues interpreting.
    if !crate::runtime::jit_compile_enabled() {
        return OsrOutcome::Declined;
    }
    // v1 envelope guards beyond the compile-side ones (has_ctx/closures are
    // compile_method_osr's): a MARKED frame (ensure/ifCurtailed handler) or
    // a caller mid-resume-sentinel needs `do_return`'s full machinery,
    // which operates on a live frame this transition removes — decline.
    let nil = vm.universe.nil_obj;
    if frame.marker(&vm.stack).raw() != nil.raw() {
        vm.stats.osr_declined += 1;
        return OsrOutcome::Declined;
    }
    let saved_bci = frame.saved_bci(&vm.stack);
    if saved_bci >= crate::oops::layout::BCI_RESUME_ENSURE_RET {
        vm.stats.osr_declined += 1;
        return OsrOutcome::Declined;
    }

    let receiver = frame.receiver(&vm.stack);
    let k = klass_of(vm, receiver);
    let Some(sel) = SymbolOop::try_from(method.selector()) else {
        vm.stats.osr_declined += 1;
        return OsrOutcome::Declined;
    };
    // Dispatch-truth guard (the S14 c2i-hatch rule): the nmethod is keyed
    // (k, sel), so it must only exist when THIS method is what dynamic
    // lookup finds — never install an OSR compile of a shadowed/replaced
    // method under a key ordinary sends will hit.
    if lookup(vm, k, sel).map(|m2| m2.oop().raw()) != Some(method.oop().raw()) {
        vm.stats.osr_declined += 1;
        return OsrOutcome::Declined;
    }

    // OSR-heal: while the existing OSR nmethod is HALF-WARM
    // (`Nmethod::is_half_warm_osr` — cold-site generic sends baked in), do
    // NOT re-enter it and do NOT compile another: DECLINE, so this
    // interpreted run continues PAST the trip point and warms the sites the
    // first OSR compile never saw. Without this, every healing run would
    // re-trip at the same backedge and take over into the same half-warm
    // code, and the cold sites' interpreter ICs could never warm — the
    // sieve's scan-increment site stayed Empty forever exactly this way.
    if vm
        .code_table
        .lookup_osr(k, sel, target_bci)
        .and_then(|i| vm.code_table.get(i))
        .is_some_and(|nm| matches!(nm.state, NmState::Alive) && nm.is_half_warm_osr())
    {
        vm.stats.osr_declined += 1;
        return OsrOutcome::Declined;
    }

    let id: NmethodId = match vm.code_table.lookup_osr(k, sel, target_bci).filter(|&i| {
        vm.code_table
            .get(i)
            .is_some_and(|nm| matches!(nm.state, NmState::Alive))
    }) {
        Some(i) => i,
        None => match crate::compiler::driver::compile_method_osr(vm, k, method, target_bci) {
            Some(i) => i,
            None => {
                vm.stats.osr_declined += 1;
                return OsrOutcome::Declined;
            }
        },
    };

    // ── Pack the transfer buffer (A3 step 3). ALLOC-FREE from here to the
    // enter: the buffer holds RAW oops a collection would invalidate.
    let (entry, buf) = {
        let nm = vm.code_table.get(id).expect("just validated/installed");
        let map = nm
            .osr_map
            .as_ref()
            .expect("an osr_table hit carries an OsrMap");
        debug_assert_eq!(map.osr_bci, target_bci);
        // T4 (S24 L2 phase B): a Context transfer pair ADOPTS the frame's
        // heap Context by identity — it must BE one. `activate_method`
        // allocates it eagerly for every has_ctx activation, so nil here is
        // a torn frame or a flags/activation mismatch: decline fail-soft,
        // BEFORE any frame mutation (the pack is read-only, but packing nil
        // would hand compiled ctx-temp ops a nil base).
        if map
            .slots
            .iter()
            .any(|s| matches!(s.src, crate::compiler::scopes::OsrSource::Context))
        {
            let ctx = frame.context(&vm.stack);
            if ctx.raw() == vm.universe.nil_obj.raw() {
                debug_assert!(
                    false,
                    "OSR Context adoption: has_ctx frame with a nil context (T4)"
                );
                vm.stats.osr_declined += 1;
                return OsrOutcome::Declined;
            }
            debug_assert!(
                crate::oops::wrappers::ContextOop::try_from(ctx).is_some(),
                "OSR Context adoption: frame context is not a Context (T4)"
            );
            vm.stats.osr_ctx_adopted += 1;
        }
        let entry = nm.code.base as u64 + map.entry_off as u64;
        let ntemps_base = fp + FRAME_TEMPS_BASE;
        let operand_base = ntemps_base + method.ntemps();
        let mut buf: Vec<u64> = Vec::with_capacity(map.slots.len());
        for s in &map.slots {
            use crate::compiler::scopes::OsrSource;
            let v = match s.src {
                OsrSource::Receiver => frame.receiver(&vm.stack),
                OsrSource::Slot(i) => frame.temp(&vm.stack, i as usize),
                OsrSource::StackSlot(i) => vm.stack.get(operand_base + i as usize),
                OsrSource::Context => frame.context(&vm.stack),
            };
            buf.push(v.raw());
        }
        (entry, buf)
    };

    // ── Linkage first, then POP (A3 step 4): the frame is gone BEFORE
    // compiled code runs — no stale duplicate of the loop state exists for
    // a mid-loop GC to scan. `sp` drops to the receiver slot, the exact cut
    // a normal return makes; `vm.stack.fp` takes the caller (or the entry
    // sentinel, `run_method`'s own spoof convention, when this WAS the
    // entry frame — the walker's call_stub arm handles that pairing).
    let saved_fp = frame.saved_fp(&vm.stack);
    let receiver_slot = frame.receiver_slot(&vm.stack);
    let entry_frame = saved_fp == ENTRY_FRAME_SENTINEL;
    vm.stack.sp = receiver_slot;
    // Restore the CALLER as the live interpreter state BEFORE entering —
    // exactly what a completed return leaves (minus the result), mirroring
    // `pop_and_deliver`: the entry case deactivates (no frames — the
    // walker's OSR-sentinel call_stub arm pairs the link below), the inner
    // case re-activates the caller and stamps `regs` (the GC-scanned truth
    // any collection during the compiled window updates and the resume
    // reads back).
    if entry_frame {
        vm.stack.deactivate();
    } else {
        vm.stack.activate_frame(saved_fp as usize);
        vm.regs.method = Some(
            Frame {
                fp: saved_fp as usize,
            }
            .method(&vm.stack),
        );
        vm.regs.bci = saved_bci;
    }

    // ── Enter (A3 step 5): standard S11 register contract via the call
    // stub — argv[0]→x0 (receiver, unused by the entry block but keeps the
    // frame's incoming-x0 convention), argv[1]→x1 (the buffer).
    vm.tier_links.push(TierLink::IntoCompiled {
        interp_frame: if entry_frame {
            ENTRY_FRAME_SENTINEL as usize
        } else {
            saved_fp as usize
        },
        entry_sp: vm.stack.sp as u64,
        nm_id: id,
    });
    vm.compiled_depth += 1;
    let stubs = vm.stubs;
    let result_bits = stubs.invoke(entry, vm, &[buf[0], buf.as_ptr() as u64]);
    vm.tier_links.pop();
    vm.compiled_depth -= 1;
    vm.stats.osr_entries += 1;

    if result_bits == NLR_SENTINEL {
        let st = vm
            .nlr_state
            .take()
            .expect("rt_osr_request: NLR sentinel returned but no NlrState is parked");
        return OsrOutcome::Nlr(crate::interpreter::unwind::continue_unwind(
            vm, st.home, st.value, st.closure,
        ));
    }

    // ── Return (A3 step 6): the standard method-return with the saved
    // linkage, mirroring `pop_and_deliver`'s two shapes exactly: the ENTRY
    // case hands the result back IN-BAND with sp cut to the receiver slot
    // (nothing pushed — `interpret_active`'s base_sp watermark and
    // `run_method`'s epilogue both count on that); the inner case pushes it
    // onto the already-reactivated caller's operand stack (`regs` was
    // stamped before the enter). The replaced activation ran partly
    // interpreted, partly compiled, with no method re-entry.
    let result = Oop::from_raw(result_bits);
    debug_assert_eq!(vm.stack.sp, receiver_slot);
    if !entry_frame {
        vm.stack.push(result);
    }
    OsrOutcome::Completed {
        result,
        entry_frame,
    }
}
