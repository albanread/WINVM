//! The last-resort `doesNotUnderstand:` fallback (SPEC §6.3, S3): reached
//! only when `#doesNotUnderstand:` itself cannot be found (no library
//! loaded yet, or a broken image) — the ordinary DNU path
//! (`interpreter::send`) instead constructs a `Message` and sends it.
//! **Must never recurse**: this prints and terminates the process, and may
//! not send anything.

use std::io::Write;

use crate::interpreter::stack::Frame;
use crate::oops::layout::ENTRY_FRAME_SENTINEL;
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::runtime::vm_state::{TierLink, VmState};

pub(crate) fn name_of(o: crate::oops::Oop) -> String {
    SymbolOop::try_from(o)
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".to_string())
}

/// Same line, for a frame whose bci is not recoverable — see the tier
/// bridge in [`print_stack_trace`].
fn print_frame_line_unknown_bci(vm: &mut VmState, method: MethodOop) {
    let sel = name_of(method.selector());
    let holder = match KlassOop::try_from(method.holder()) {
        Some(k) => name_of(k.name()),
        None => "?".to_string(),
    };
    let _ = writeln!(vm.out, "  {holder}>>{sel} @?");
}

fn print_frame_line(vm: &mut VmState, method: MethodOop, bci: usize) {
    let sel = name_of(method.selector());
    let holder = match KlassOop::try_from(method.holder()) {
        Some(k) => name_of(k.name()),
        None => "?".to_string(),
    };
    let _ = writeln!(vm.out, "  {holder}>>{sel} @{bci}");
}

/// The one compiled frame a mixed-tier trace can ever show in S10 (D1's
/// send-free compiled methods can't nest — `vm.tier_links` never holds
/// more than one entry, see its own doc). Prints nothing if the id is
/// somehow no longer installed (defensive; not reachable from a live
/// `enter_compiled` call, which only ever names a slot it just installed
/// or already validated).
fn print_compiled_frame_line(vm: &mut VmState, nm_id: crate::codecache::nmethod::NmethodId) {
    let Some(nm) = vm.code_table.get(nm_id) else {
        return;
    };
    let sel = nm.key_selector.as_string();
    let holder = name_of(nm.key_klass.name());
    let bci = nm.poll_bci;
    let _ = match bci {
        Some(b) => writeln!(vm.out, "  {holder}>>{sel} @{b} (compiled)"),
        None => writeln!(vm.out, "  {holder}>>{sel} @? (compiled)"),
    };
}

/// One line per frame, top (currently executing) frame first — pinned
/// format for golden tests (`sprint_s03_detail.md` §Algorithms-6):
/// `  <Holder>>><selector> @<bci>`. The top frame's bci comes from
/// `vm.regs` (not yet saved into any frame slot); every ancestor's comes
/// from its child frame's `saved_bci` (where that ancestor resumes).
///
/// S10 D4 addition: if a compiled activation is currently live
/// (`vm.tier_links` non-empty), its line prints FIRST, above
/// `vm.regs.method`'s own — compiled code never pushes a `vm.stack` frame
/// or touches `vm.regs` (D1), so without consulting `tier_links` this walk
/// would show only the interpreted caller and miss the compiled callee
/// running "underneath" it entirely (tests_s10.md's `mixed_trace_golden`,
/// gate item 4).
pub fn print_stack_trace(vm: &mut VmState) {
    if let Some(TierLink::IntoCompiled { nm_id, .. }) = vm.tier_links.last().copied() {
        print_compiled_frame_line(vm, nm_id);
    }

    let top_method = vm.regs.method.expect("print_stack_trace: no active method");
    let top_bci = vm.regs.bci;
    print_frame_line(vm, top_method, top_bci);

    let mut fp = vm.stack.fp;
    // A cursor into `tier_links`, consumed from the top down as the walk
    // crosses tier boundaries (see the bridging step below).
    let mut link_cursor = vm.tier_links.len();

    // Bounded so a corrupt/cyclic chain can't spin forever, and every frame
    // read is the non-panicking variant: aborting the VM from inside its own
    // error reporter is not an option (the `Frame::saved_fp: not a smi` crash
    // this replaces).
    for _ in 0..MAX_TRACE_FRAMES {
        let frame = Frame { fp };
        let saved_fp = match frame.saved_fp_opt(&vm.stack) {
            Some(v) if v != ENTRY_FRAME_SENTINEL => v,
            _ => {
                // A tier boundary, not the bottom of the stack. Compiled
                // code pushes no `vm.stack` frame, so the interpreted
                // chain simply ends here — and this walk used to stop,
                // silently dropping every frame BELOW the compiled one.
                //
                // `TierLink::IntoCompiled` records exactly what is needed
                // to continue: which nmethod is running, and
                // `interp_frame`, the fp of the interpreted caller on the
                // far side of it. Bridge across and keep walking.
                //
                // Stopping was defensible when a compiled activation was
                // necessarily the OUTERMOST thing on the stack (D1's
                // send-free methods could not nest, so nothing was below
                // it to lose). Once compiled code can call back into the
                // interpreter that stopped being true, and the truncation
                // became a real loss: a guest error under the JIT showed a
                // stack cut off at the tier boundary.
                let Some(i) = vm.tier_links[..link_cursor]
                    .iter()
                    .rposition(|l| matches!(l, TierLink::IntoCompiled { .. }))
                else {
                    break;
                };
                let TierLink::IntoCompiled {
                    interp_frame,
                    nm_id,
                    ..
                } = vm.tier_links[i]
                else {
                    break;
                };
                link_cursor = i;
                let _ = nm_id;
                // Deliberately NOT printing a compiled frame line here.
                // The nmethod named by this link is frequently the same
                // activation the interpreted walk has already reported —
                // a compiled method that deoptimized mid-flight leaves a
                // materialized interpreted frame AND its now-dead tier
                // link, so printing both duplicates it. The link's job in
                // this walk is purely to say where the chain resumes.
                fp = interp_frame;
                let Some(m) = (Frame { fp }).method_opt(&vm.stack) else {
                    break;
                };
                // The bci is the one thing the bridge genuinely cannot
                // recover: an interpreted caller's resume point is read
                // from its CALLEE's frame, and a compiled callee has no
                // frame to read it from. Print `?` — a frame named with
                // an honest unknown beats both a missing frame and a
                // confident `@0` that reads like a real bytecode index.
                print_frame_line_unknown_bci(vm, m);
                continue;
            }
        };
        let caller_bci = frame.saved_bci_opt(&vm.stack).unwrap_or(0);
        fp = saved_fp as usize;
        let caller_method = match (Frame { fp }).method_opt(&vm.stack) {
            Some(m) => m,
            None => break,
        };
        print_frame_line(vm, caller_method, caller_bci);
    }
}

/// A hard ceiling on trace depth — defends the walker against a cyclic or
/// corrupt `saved_fp` chain (a deep-but-legitimate stack still prints in full;
/// real ones never approach this).
const MAX_TRACE_FRAMES: usize = 4096;

/// `MACVM_TRACE=dnu` (RUSTTCL's `trace dnu on`, same channel): one line per
/// `doesNotUnderstand:` dispatch, printed BEFORE the `Message` send/lookup
/// below runs — so a `#doesNotUnderstand:` override that itself traps or
/// aborts still leaves the diagnostic visible. `site` names where the send
/// originated (`"interpreted"` from `send::dnu`, `"compiled nm=<id>"` from
/// `stubs::rt_dnu`) — the detail that tells a wrong-receiver-in-compiled-
/// code bug apart from an ordinary interpreted DNU at a glance.
pub fn trace_dnu(vm: &VmState, site: &str, receiver_klass: KlassOop, selector: SymbolOop) {
    if !vm.options.trace.is_enabled("dnu") {
        return;
    }
    let klass_name = name_of(receiver_klass.name());
    let sel_str = selector.as_string();
    eprintln!("[dnu] {site}: {klass_name}>>#{sel_str} not understood");
}

/// `selector` was not understood by `receiver_klass`, AND
/// `#doesNotUnderstand:` itself could not be found from `receiver_klass`
/// (only reachable before the library installs a root `doesNotUnderstand:`
/// on `Object`, or with a broken image). Prints the pinned fallback format
/// and terminates with exit code 1 — never returns, never sends.
pub fn dnu_fallback(vm: &mut VmState, selector: SymbolOop, receiver_klass: KlassOop) -> ! {
    let sel_str = selector.as_string();
    let klass_name = name_of(receiver_klass.name());
    let message = format!("DNU #{sel_str} (receiver class {klass_name})");
    let _ = writeln!(vm.out, "{message}");
    print_stack_trace(vm);
    let _ = vm.out.flush();
    // DBG4 (mirrors prim_error's DBG1 hook): an armed debugger turns the
    // would-be-fatal DNU into a pre-mortem inspectable stop at the erring
    // activation; the DNU stays terminal on any resume.
    if crate::runtime::debug::wants_error_halt(vm) {
        if let Some(m) = vm.regs.method {
            let bci = vm.regs.bci;
            crate::runtime::debug::halt(
                vm,
                m,
                bci,
                crate::runtime::debug::HaltReason::GuestError(message.clone()),
            );
        }
    }
    // DBG0: the fatal-DNU mini-dossier (docs/DEBUGGER.md §4.1) — the
    // deviceInAdd:-hunt tool. `MACVM_PROBE=off` opts out. Skipped when a
    // recovery is actually about to happen (embedded `VmHandle::eval`,
    // `deopt_trap::raise_guest_fatal` below) — a full register/frame/heap
    // dossier is right for a condition that's about to end the process, and
    // pure noise for an everyday Workspace typo that's about to be reported
    // and recovered from like any other doIt error.
    if !crate::codecache::deopt_trap::has_registered_jmp_slot()
        && crate::runtime::probe::guest_report_enabled()
    {
        crate::runtime::probe::fatal_guest_report(vm, &message);
    }
    // See `prim_error`'s identical call: an unhandled DNU curtails everything
    // up to the entry frame, so the armed ensure:/ifCurtailed: blocks run here
    // — the jump below skips every frame without touching them.
    crate::interpreter::unwind::run_curtailment_blocks_on_error(vm);
    crate::codecache::deopt_trap::raise_guest_fatal(message);
}

/// A fatal GUEST-level error raised from Rust runtime code — the same
/// recipe as [`dnu_fallback`] (report + stack trace, debugger error-halt
/// hook, probe dossier when the condition is about to end the process,
/// curtailment blocks, `raise_guest_fatal`), packaged for call sites
/// outside the send path. An embedded `VmHandle` recovers it as an
/// ordinary `Err` and the VM keeps serving; plain CLI use stays fatal
/// (exit 1) with the message and walkback already printed.
///
/// Use this for conditions the GUEST program caused — a typo'd symbol in a
/// hand-authored FFI pragma, an unsupported declared shape token — where a
/// silent `Fallthrough` would masquerade as success (an FFI method's body
/// is empty) but a Rust `panic!` would take down the whole embedding host
/// for a Workspace-level mistake. VM invariant violations stay `panic!`.
pub fn guest_fatal(vm: &mut VmState, message: String) -> ! {
    let _ = writeln!(vm.out, "{message}");
    print_stack_trace(vm);
    let _ = vm.out.flush();
    if crate::runtime::debug::wants_error_halt(vm) {
        if let Some(m) = vm.regs.method {
            let bci = vm.regs.bci;
            crate::runtime::debug::halt(
                vm,
                m,
                bci,
                crate::runtime::debug::HaltReason::GuestError(message.clone()),
            );
        }
    }
    if !crate::codecache::deopt_trap::has_registered_jmp_slot()
        && crate::runtime::probe::guest_report_enabled()
    {
        crate::runtime::probe::fatal_guest_report(vm, &message);
    }
    crate::interpreter::unwind::run_curtailment_blocks_on_error(vm);
    crate::codecache::deopt_trap::raise_guest_fatal(message);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::vm_state::{OutputBuffer, VmOptions, VmState};

    fn sym_of(method: MethodOop) -> SymbolOop {
        SymbolOop::try_from(method.selector()).expect("method selector is not a Symbol")
    }

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

    fn trivial_method(vm: &mut VmState, name: &[u8]) -> MethodOop {
        let mut b = crate::bytecode::BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(name);
        b.finish(vm, sel, 0, 0)
    }

    #[test]
    fn message_shape() {
        // Exercises the Message klass layout independent of a real DNU
        // send: allocate one by hand and confirm the pinned 2-slot shape.
        let mut vm = test_vm();
        let message_klass = vm.universe.message_klass;
        let array_klass = vm.universe.array_klass;
        let msg = crate::memory::alloc::alloc_slots(&mut vm, message_klass);
        let sel = vm.universe.intern(b"foo:");
        let args = crate::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, 1);
        msg.set_body_oop(0, sel.oop());
        msg.set_body_oop(1, args.oop());
        assert_eq!(msg.body_oop(0), sel.oop());
        let got_args = crate::oops::wrappers::ArrayOop::try_from(msg.body_oop(1))
            .expect("arguments is an Array");
        assert_eq!(got_args.len(), 1);
    }

    #[test]
    fn stack_trace_top_frame_only() {
        let mut vm = test_vm();
        let buf = OutputBuffer::new();
        vm.out = Box::new(buf.clone());

        let object_klass = vm.universe.object_klass;
        let method = trivial_method(&mut vm, b"top");
        let sel = sym_of(method);
        crate::runtime::lookup::install_method(&mut vm, object_klass, sel, method);
        let recv = vm.universe.nil_obj;
        vm.stack.push(recv);
        crate::interpreter::activate_entry(&mut vm, method, 0);
        vm.regs.method = Some(method);
        vm.regs.bci = 3;

        print_stack_trace(&mut vm);
        let out = buf.as_string();
        assert!(out.contains("Object>>top @3"), "unexpected trace: {out}");
    }

    #[test]
    fn stack_trace_multi_frame() {
        let mut vm = test_vm();
        let buf = OutputBuffer::new();
        vm.out = Box::new(buf.clone());

        let object_klass = vm.universe.object_klass;
        let caller = trivial_method(&mut vm, b"caller");
        let callee = trivial_method(&mut vm, b"callee");
        let caller_sel = sym_of(caller);
        let callee_sel = sym_of(callee);
        crate::runtime::lookup::install_method(&mut vm, object_klass, caller_sel, caller);
        crate::runtime::lookup::install_method(&mut vm, object_klass, callee_sel, callee);

        let recv = vm.universe.nil_obj;
        vm.stack.push(recv);
        crate::interpreter::activate_entry(&mut vm, caller, 0);
        // Simulate a send from `caller` at bci 5 into `callee`.
        vm.regs.method = Some(caller);
        vm.regs.bci = 5;
        let saved_fp = vm.stack.fp as i64;
        let saved_bci = vm.regs.bci;
        vm.stack.push(recv); // callee's receiver
        crate::interpreter::push_frame(&mut vm, callee, 0, saved_fp, saved_bci);
        vm.regs.method = Some(callee);
        vm.regs.bci = 1;

        print_stack_trace(&mut vm);
        let out = buf.as_string();
        assert!(
            out.contains("Object>>callee @1"),
            "missing top frame: {out}"
        );
        assert!(
            out.contains("Object>>caller @5"),
            "missing caller frame: {out}"
        );
        let callee_line = out.find("callee").unwrap();
        let caller_line = out.find("caller").unwrap();
        assert!(callee_line < caller_line, "top frame must print first");
    }
}
