//! DBG1 — HALT, the Smalltalk-level debugger core (docs/DEBUGGER.md §2/§3).
//!
//! A halt is a nested command loop at a bytecode boundary: `halt` stops
//! between bytecodes (dispatch-loop boundaries are GC-safe and
//! frame-consistent by construction) and services commands on
//! stdin/stderr until told to resume. Breakpoints are a side table keyed
//! by (method identity-hash, bci) — never a patched bytecode (the same
//! side-table-over-self-modification choice ICs made, SPEC §4.3): methods
//! stay pure, golden disassembly stays byte-identical.
//!
//! Two rules the state obeys (v2 design):
//! - **`DebugState` holds no oops.** It is not a GC root — which is
//!   exactly why the key is the identity hash (stable across compaction,
//!   the A16 `methodSources` trick) and a `Breakpoint` carries only
//!   STRINGS for its confirm check.
//! - **Identity hashes can collide**, so a hash hit is confirmed against
//!   the stored (holder-name, selector-name) pair before halting — a
//!   false halt would be a confounding experience precisely for someone
//!   already debugging.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

/// One breakpoint. No oops (module doc) — `restore_compile` remembers
/// whether `set_breakpoint` was the one to set `compile_disabled`, so
/// clearing the last breakpoint restores tier-up eligibility only if the
/// method was eligible before.
#[derive(Clone, Debug)]
pub struct Breakpoint {
    pub holder: String,
    pub selector: String,
    pub bci: u16,
    pub restore_compile: bool,
}

/// Into | Over | Out (docs/DEBUGGER.md §3.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepKind {
    Into,
    Over,
    Out,
}

/// Armed by the halt loop's step commands; checked in the dispatch hook.
/// `base_fp`/`base_serial` identify the frame the command was issued in —
/// frame serials (SPEC §5.1 FP+5, dead-home detection) make the fp
/// comparisons robust against fp reuse.
#[derive(Clone, Copy, Debug)]
pub struct StepPlan {
    pub kind: StepKind,
    pub base_fp: usize,
    pub base_serial: u32,
    /// The bci the plan was armed at — `Into` halts on ANY change of
    /// (fp, bci), so re-halting on the very same bytecode is suppressed.
    pub base_bci: usize,
}

/// DBG4 (docs/gui_debugger_design.md): the GUI face of the HALT loop. With a
/// frontend installed, `halt()` swaps stdin/stderr for publish/next_command —
/// the same command engine, a different terminal. Installed per-VM (the Cocoa
/// supervisor installs it on the PRIMARY only; a UI-worker halt would park the
/// main thread).
pub trait DebugFrontend: Send + Sync {
    /// The full debugger state (stack + selected frame + last output) — the
    /// whole report on every change, never a delta.
    fn publish(&self, report: &str);
    /// Block until the UI sends the next command line.
    fn next_command(&self) -> String;
}

/// §2's shared core: one new `VmState` field. `active` is the master
/// switch — the ONLY thing the dispatch fast path reads when no debugging
/// is happening (one untaken branch per bytecode).
#[derive(Default)]
pub struct DebugState {
    pub active: bool,
    /// DBG4: the GUI frontend, if any (see [`DebugFrontend`]).
    pub frontend: Option<std::sync::Arc<dyn DebugFrontend>>,
    /// DBG4: route `error:`/DNU to a pre-mortem halt (inspect, then the
    /// fatal path proceeds on ANY resume — there is no catch-and-continue).
    pub halt_on_error: bool,
    breakpoints: HashMap<(u32, u16), Breakpoint>,
    pub step: Option<StepPlan>,
    /// §3.6 reentrancy: while > 0 (a halt loop is live), breakpoint hits,
    /// `halt`, and step checks are ignored — doits evaluated BY the
    /// debugger can't recursively open debuggers on themselves.
    pub session_depth: u32,
    /// Name-based breakpoint specs whose class/method doesn't exist YET —
    /// `MACVM_DEBUG=break:` naturally references classes defined in the
    /// very file about to run. `install_method` (the single point every
    /// method enters a dictionary through) retries these on each install,
    /// so the breakpoint lands the moment the method exists, before any
    /// doit can call it.
    pub pending: Vec<(String, String, u16)>,
    /// Methods pinned to tier-0 via [`pin_method`] — identity-hash → "did
    /// we set compile_disabled" (so `unpin` restores only what it disabled).
    /// The differential-diagnosis lever, no oops.
    pinned: HashMap<u32, bool>,
    /// `MACVM_PIN=Class>>sel[,…]` specs not yet installed — landed at
    /// `install_method` like `pending`, so a pin can name a class defined
    /// in the file about to run.
    pub pending_pins: Vec<(String, String)>,
    /// DBG5 (compiled_send_auditor_design.md §D3): `MACVM_STEP_CALLS=1` (or the
    /// RUSTTCL `step-call` verb) arms the interactive step-call auditor — a
    /// nested command loop at every compiled send boundary (`rt_resolve_send`,
    /// which force-cold makes total). Like `MACVM_TRACE=calls` it forces ICs
    /// cold; unlike it, it stops and services commands instead of just logging.
    pub step_calls: bool,
    /// Test/diagnostic knob: compile COLD (Untaken-feedback) sends inside
    /// INLINED bodies as uncommon traps — the pre-4c575c8 lowering. The
    /// production default (false) lowers them to plain CallSends (the
    /// markInputs: deopt-storm fix); the it_tier1 deopt-materializer pins
    /// set this per-VM to keep exercising the nested-frame rebuild through
    /// in-body traps (the materializer is lowering-agnostic, so the
    /// coverage transfers). Also useful to reproduce pre-fix behavior.
    pub cold_splice_traps: bool,
    /// DBG4 §2.2: the bci→source-line map per method, keyed
    /// `"Holder>>selector"` (` class` suffix for the class side). Lines are
    /// 1-based and RELATIVE to the method's own header (so they align with the
    /// method source the browser/debugger shows, not the file it loaded from).
    /// Populated by codegen; read by `line_for` to highlight the halted
    /// statement. Latest compile wins (redefinition-friendly).
    pub method_lines: HashMap<String, Vec<(u16, u32)>>,
}

impl DebugState {
    pub fn new() -> DebugState {
        DebugState::default()
    }

    pub fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }

    pub fn list(&self) -> Vec<String> {
        let mut rows: Vec<String> = self
            .breakpoints
            .values()
            .map(|b| format!("{}>>{} @{}", b.holder, b.selector, b.bci))
            .collect();
        rows.sort();
        rows
    }
}

/// Set a breakpoint at `(method, bci)` — §2.1's pin-to-tier-0 sequence.
/// `bci` must be an instruction boundary (validated against the decode
/// walk; a mid-instruction bci is a caller bug rejected here, not a
/// silent never-hit). Returns a human-readable confirmation.
pub fn set_breakpoint(vm: &mut VmState, method: MethodOop, bci: u16) -> Result<String, String> {
    // 1. Validate the bci against instruction boundaries.
    let mut b = 0usize;
    let mut ok = false;
    while b < method.bytecode_len() {
        if b == bci as usize {
            ok = true;
            break;
        }
        let (_, next) = crate::bytecode::decode_at(method, b);
        b = next;
    }
    if !ok {
        return Err(format!(
            "bci {bci} is not an instruction boundary in this method (len {})",
            method.bytecode_len()
        ));
    }

    let holder = KlassOop::try_from(method.holder());
    let holder_name = holder
        .map(|k| crate::runtime::error::name_of(k.name()))
        .unwrap_or_else(|| "?".into());
    let selector_name = SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".into());

    // 2. Insert + flag the method.
    let key = (vm.universe.identity_hash(method.oop()), bci);
    let restore_compile = !method.compile_disabled();
    vm.debug.breakpoints.insert(
        key,
        Breakpoint {
            holder: holder_name.clone(),
            selector: selector_name.clone(),
            bci,
            restore_compile,
        },
    );
    method.set_has_bp();
    vm.debug.active = true;

    // 3. Pin to tier-0: block future compiles, invalidate existing
    //    nmethods whose root or inlined scopes include this method — the
    //    S13 REDEFINITION path, verbatim (setting a breakpoint is,
    //    invalidation-wise, indistinguishable from redefining the method;
    //    DependencyIndex already finds inliners and handles the 10d
    //    super-send edge cases). Live activations convert through the
    //    existing return-redirect / back-edge poll.
    method.set_compile_disabled();
    method.set_has_bp(); // set_compile_disabled resets the counters word
    if let (Some(k), Some(s)) = (holder, SymbolOop::try_from(method.selector())) {
        crate::runtime::deps::invalidate_dependents(vm, k, s);
    }

    Ok(format!(
        "breakpoint set: {holder_name}>>{selector_name} @{bci}"
    ))
}

/// Clear one breakpoint; restores `compile_disabled` (only if we set it)
/// once the method's LAST breakpoint is gone. Recompilation then happens
/// naturally by counters.
pub fn clear_breakpoint(vm: &mut VmState, method: MethodOop, bci: u16) -> Result<String, String> {
    let key = (vm.universe.identity_hash(method.oop()), bci);
    let bp = vm
        .debug
        .breakpoints
        .remove(&key)
        .ok_or_else(|| format!("no breakpoint at bci {bci}"))?;
    let hash = key.0;
    let any_left = vm.debug.breakpoints.keys().any(|(h, _)| *h == hash);
    if !any_left {
        method.clear_has_bp();
        if bp.restore_compile {
            method.clear_compile_disabled();
        }
    }
    Ok(format!(
        "cleared {}>>{} @{}",
        bp.holder, bp.selector, bp.bci
    ))
}

/// The dispatch hook's breakpoint test — called only when
/// `debug.active && method.has_bp()`. Confirms the identity-hash hit
/// against the stored selector (module doc's collision rule). `&mut` only
/// because `identity_hash` assigns lazily on first ask.
pub fn hit(vm: &mut VmState, method: MethodOop, bci: usize) -> bool {
    if vm.debug.session_depth > 0 || bci > u16::MAX as usize {
        return false;
    }
    let key = (vm.universe.identity_hash(method.oop()), bci as u16);
    let Some(stored_sel) = vm.debug.breakpoints.get(&key).map(|b| b.selector.clone()) else {
        return false;
    };
    let sel = SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_default();
    sel == stored_sel
}

/// §3.4: does an armed step plan fire at `(method, bci)`? Consumes the
/// plan when it does.
pub fn step_check(vm: &mut VmState, bci: usize) -> bool {
    if vm.debug.session_depth > 0 {
        return false;
    }
    let Some(plan) = vm.debug.step else {
        return false;
    };
    let fp = vm.stack.fp;
    let fired = match plan.kind {
        // Into: the very next bytecode anywhere — any (fp, bci) change.
        StepKind::Into => fp != plan.base_fp || bci != plan.base_bci,
        // Over: at-or-below the base frame at a different point — callee
        // frames sit ABOVE (higher fp) and are skipped; equal fp is "back
        // in the same frame" whether or not the serial survived (a fresh
        // serial at the same fp means the base frame returned and
        // something reused its slot — the step is over either way, which
        // is also why `base_serial` needs no comparison here; it is kept
        // on the plan for the finer distinctions later waves may want).
        StepKind::Over => fp < plan.base_fp || (fp == plan.base_fp && bci != plan.base_bci),
        // Out: strictly below the base frame.
        StepKind::Out => fp < plan.base_fp,
    };
    if fired {
        vm.debug.step = None;
    }
    fired
}

fn frame_serial(vm: &VmState, fp: usize) -> u32 {
    crate::interpreter::stack::Frame { fp }.serial(&vm.stack)
}

/// Why the halt loop was entered — shown as the prompt's first line.
pub enum HaltReason {
    Breakpoint,
    Step,
    GuestError(String),
}

/// DBG4: the map key for a method — `Holder>>selector`, with a ` class`
/// suffix on the class side (matching `resolve_method_by_name`'s grammar and
/// the browser's own `selectedBreakpointClass`).
pub fn method_key(holder_name: &str, class_side: bool, selector: &str) -> String {
    if class_side {
        format!("{holder_name} class>>{selector}")
    } else {
        format!("{holder_name}>>{selector}")
    }
}

/// DBG4: record a method's bci→line map (codegen, at compile time). Lines are
/// already 1-based and method-relative. Empty maps (primitive-only methods)
/// are skipped.
pub fn record_line_map(vm: &mut VmState, key: String, map: Vec<(u16, u32)>) {
    if map.is_empty() {
        return;
    }
    vm.debug.method_lines.insert(key, map);
}

/// DBG4: the 1-based source line for `bci` in the method keyed `key` — the
/// largest recorded bci ≤ `bci` (the statement in progress). `None` if the
/// method has no map (a doit, a primitive, or a method compiled before this
/// feature). Read at halt to highlight the current statement.
pub fn line_for(vm: &VmState, key: &str, bci: usize) -> Option<u32> {
    let map = vm.debug.method_lines.get(key)?;
    let mut line = None;
    for &(b, l) in map {
        if (b as usize) <= bci {
            line = Some(l);
        } else {
            break;
        }
    }
    line
}

/// DBG1/DBG4: should a would-be-fatal guest error (`error:` / DNU) open the
/// halt loop before dying? CLI (`macvm debug`): whenever active. GUI (a
/// frontend is installed): additionally honors the Debug ▸ Halt on Error
/// toggle. Either way the error stays terminal on resume — the halt is for
/// looking, not healing.
pub fn wants_error_halt(vm: &VmState) -> bool {
    vm.debug.active
        && vm.debug.session_depth == 0
        && (vm.debug.frontend.is_none() || vm.debug.halt_on_error)
}

/// §3.2: the nested command loop. Runs at a bytecode boundary inside
/// `dispatch_from`; returns when the user resumes (continue/step — step
/// arms `vm.debug.step` before returning). Two faces, one engine: the CLI
/// reads stdin / writes stderr; with a [`DebugFrontend`] installed (DBG4)
/// the same commands arrive from `next_command` and the WHOLE state is
/// re-published after every one (blast, don't patch).
pub fn halt(vm: &mut VmState, method: MethodOop, bci: usize, reason: HaltReason) {
    vm.debug.session_depth += 1;
    let sel = SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".into());
    let holder = KlassOop::try_from(method.holder())
        .map(|k| crate::runtime::error::name_of(k.name()))
        .unwrap_or_else(|| "?".into());
    if vm.debug.frontend.is_some() {
        let reason_line = match &reason {
            HaltReason::Breakpoint => format!("breakpoint {holder}>>{sel} @{bci}"),
            HaltReason::Step => format!("step {holder}>>{sel} @{bci}"),
            HaltReason::GuestError(msg) => format!("{msg} (in {holder}>>{sel} @{bci})"),
        };
        gui_halt(vm, bci, &reason_line);
        vm.debug.session_depth -= 1;
        return;
    }
    match &reason {
        HaltReason::Breakpoint => eprintln!("halt: breakpoint {holder}>>{sel} @{bci}"),
        HaltReason::Step => eprintln!("halt: step {holder}>>{sel} @{bci}"),
        HaltReason::GuestError(msg) => eprintln!("halt: {msg} (in {holder}>>{sel} @{bci})"),
    }

    // The selectable frame list: interpreter frames, innermost first.
    let mut selected: usize = 0;
    let frames = interp_frames(vm);

    let stdin = std::io::stdin();
    loop {
        eprint!("(halt) ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF: resume — a scripted session that ends its input
                // means "continue"; hanging would deadlock gates.
                eprintln!("(eof — continuing)");
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [] => {}
            ["continue"] | ["c"] => break,
            ["quit"] => {
                eprintln!("(halt) quit — exiting process");
                crate::runtime::vm_state::fatal_exit(0);
            }
            ["bt"] => bt(vm),
            ["frame", n] => match n.parse::<usize>() {
                Ok(n) if n < frames.len() => {
                    selected = n;
                    show_frame(vm, frames[selected], selected);
                }
                _ => eprintln!("frame: expected 0..{}", frames.len().saturating_sub(1)),
            },
            ["temps"] => {
                if let Some(&fp) = frames.get(selected) {
                    show_frame(vm, fp, selected);
                }
            }
            ["step"] | ["s"] => {
                arm_step(vm, StepKind::Into, bci);
                break;
            }
            ["next"] | ["n"] => {
                arm_step(vm, StepKind::Over, bci);
                break;
            }
            ["finish"] => {
                arm_step(vm, StepKind::Out, bci);
                break;
            }
            ["print", rest @ ..] if !rest.is_empty() => {
                let expr = rest.join(" ");
                let fp = frames.get(selected).copied().unwrap_or(vm.stack.fp);
                print_expr(vm, fp, &expr);
            }
            ["help"] => {
                eprintln!(
                    "bt | frame N | temps | print <expr> | step|s | next|n | finish | \
                     continue|c | quit"
                );
            }
            other => eprintln!("unknown: {other:?} (try help)"),
        }
    }
    vm.debug.session_depth -= 1;
}

/// DBG4: the GUI face of the halt loop — the CLI's command engine with
/// publish/next_command in place of stderr/stdin. After EVERY command the
/// full report is re-published; on resume the frontend gets `RUNNING`.
fn gui_halt(vm: &mut VmState, bci: usize, reason_line: &str) {
    let fe = vm
        .debug
        .frontend
        .as_ref()
        .expect("gui_halt: frontend checked by caller")
        .clone();
    let frames = interp_frames(vm);
    let mut selected: usize = 0;
    let mut out_text = String::new();
    let status = format!("HALTED {reason_line}");
    fe.publish(&build_report(vm, &status, &frames, selected, bci, &out_text));
    loop {
        let line = fe.next_command();
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [] => {}
            ["continue"] | ["c"] => break,
            ["abort"] => {
                // The guest-fatal path: on the GUI primary that is
                // ExitThread → the supervisor respawns a fresh world
                // ("recover clean or die" — no catch-and-continue).
                fe.publish("RUNNING\n");
                crate::runtime::vm_state::fatal_exit(1);
            }
            ["frame", n] => match n.parse::<usize>() {
                Ok(n) if n < frames.len() => {
                    selected = n;
                    out_text.clear();
                }
                _ => {
                    out_text =
                        format!("frame: expected 0..{}", frames.len().saturating_sub(1));
                }
            },
            ["step"] | ["s"] => {
                arm_step(vm, StepKind::Into, bci);
                break;
            }
            ["next"] | ["n"] => {
                arm_step(vm, StepKind::Over, bci);
                break;
            }
            ["finish"] => {
                arm_step(vm, StepKind::Out, bci);
                break;
            }
            ["print", rest @ ..] if !rest.is_empty() => {
                let expr = rest.join(" ");
                let fp = frames.get(selected).copied().unwrap_or(vm.stack.fp);
                out_text = print_expr_text(vm, fp, &expr);
            }
            other => out_text = format!("unknown: {other:?}"),
        }
        fe.publish(&build_report(vm, &status, &frames, selected, bci, &out_text));
    }
    fe.publish("RUNNING\n");
}

/// The DBG4 report — the whole debugger state, sections by marker line,
/// stack fields `␟`(0x1f)-separated. A frame's CURRENT bci is the halt bci
/// for the top frame and the CALLEE's saved (resume) bci for every outer
/// frame — that is where the frame stands.
fn build_report(
    vm: &VmState,
    status: &str,
    frames: &[usize],
    selected: usize,
    halt_bci: usize,
    out_text: &str,
) -> String {
    const SEP: char = '\u{1f}';
    let mut r = String::new();
    r.push_str(status);
    r.push('\n');
    r.push_str("==STACK==\n");
    for (i, &fp) in frames.iter().enumerate() {
        let f = crate::interpreter::stack::Frame { fp };
        let m = f.method(&vm.stack);
        let sel = SymbolOop::try_from(m.selector())
            .map(|s| s.as_string())
            .unwrap_or_else(|| "?".into());
        let holder = KlassOop::try_from(m.holder())
            .map(|k| crate::runtime::error::name_of(k.name()))
            .unwrap_or_else(|| "?".into());
        let bci_val = if i == 0 {
            halt_bci
        } else {
            crate::interpreter::stack::Frame { fp: frames[i - 1] }
                .saved_bci_opt(&vm.stack)
                .unwrap_or(0)
        };
        // DBG4 §2.2: the 1-based method-relative source line for this frame's
        // bci (empty when the method has no map — a doit/primitive). A 5th
        // field, so old readers that split on 4 stay compatible.
        let key = format!("{holder}>>{sel}");
        let line = line_for(vm, &key, bci_val)
            .map(|l| l.to_string())
            .unwrap_or_default();
        r.push_str(&format!("{i}{SEP}{holder}{SEP}{sel}{SEP}{bci_val}{SEP}{line}\n"));
    }
    r.push_str("==FRAME==\n");
    if let Some(&fp) = frames.get(selected) {
        r.push_str(&frame_text(vm, fp, selected));
    }
    r.push_str("==OUT==\n");
    r.push_str(out_text);
    if !out_text.is_empty() && !out_text.ends_with('\n') {
        r.push('\n');
    }
    r
}

/// DBG5 §D3 (compiled_send_auditor_design.md): the interactive step-call
/// auditor — a nested command loop at a COMPILED send boundary, entered from
/// `rt_resolve_send`'s force-cold short-circuit when `vm.debug.step_calls` is
/// armed. This is PROBE's discipline live (never deopt, freeze the frame,
/// read-only): it shows the send about to happen and services inspection
/// commands until the user resumes. Reuses HALT's loop shape and `bt`, but
/// NOTHING here mutates a frame or evaluates Smalltalk (§4.5 — `print`-eval is
/// HALT's job). Commands read stdin, output stderr (guest stdout untouched).
///
/// `step`/`s` resumes and stops at the NEXT send (force-cold guarantees one);
/// `continue`/`c` disarms and lets the run finish (still force-cold if
/// `MACVM_TRACE=calls` is also on, just no more prompts).
pub fn step_call_prompt(
    vm: &mut VmState,
    caller: crate::codecache::nmethod::NmethodId,
    site_idx: usize,
    selector: SymbolOop,
    recv_klass: KlassOop,
    argc: u8,
) {
    // Reentrancy: a `bt`/`slots` command must not itself re-open a prompt via
    // the send hook — same session_depth guard HALT uses (§3.6).
    if vm.debug.session_depth > 0 {
        return;
    }
    vm.debug.session_depth += 1;
    let sel = selector.as_string();
    let kname = crate::runtime::error::name_of(recv_klass.name());
    eprintln!(
        "▸ send #{sel} to {kname}  (from nm={}#{site_idx}, argc={argc})",
        caller.0
    );

    // Snapshot the live compiled frames (fp, ret_pc, nm) for the `slots` verb —
    // walk_frames only lends `&VmState`, so collect before the command loop's
    // own `&mut` uses (`step_calls` write, session_depth).
    let mut compiled: Vec<(u64, u64, crate::codecache::nmethod::NmethodId)> = Vec::new();
    crate::runtime::frames::walk_frames(vm, |fv| {
        if let crate::runtime::frames::FrameView::Compiled { fp, ret_pc, nm } = fv {
            compiled.push((fp, ret_pc, nm));
        }
    });

    let stdin = std::io::stdin();
    loop {
        eprint!("(auditor) ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF: a scripted session that ends its input means "run to
                // completion" — disarm so we don't prompt forever, then resume.
                eprintln!("(eof — continuing)");
                vm.debug.step_calls = false;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
        match line.split_whitespace().collect::<Vec<&str>>().as_slice() {
            [] => {}
            ["step"] | ["s"] => break, // stay armed → stop at the next send
            ["continue"] | ["c"] => {
                vm.debug.step_calls = false;
                break;
            }
            ["quit"] => {
                eprintln!("(auditor) quit — exiting process");
                crate::runtime::vm_state::fatal_exit(0);
            }
            ["bt"] => bt(vm),
            ["slots"] => match compiled.first() {
                Some(&(fp, ret_pc, nm)) => {
                    crate::memory::roots::dump_frame_oops(vm, fp, nm, ret_pc)
                }
                None => eprintln!("(auditor) no compiled frame on the stack"),
            },
            ["slots", n] => match n.parse::<usize>() {
                Ok(n) if n < compiled.len() => {
                    let (fp, ret_pc, nm) = compiled[n];
                    crate::memory::roots::dump_frame_oops(vm, fp, nm, ret_pc);
                }
                _ => eprintln!(
                    "(auditor) slots: frame 0..{}",
                    compiled.len().saturating_sub(1)
                ),
            },
            ["help"] => {
                eprintln!("bt | slots [N] | step|s (→ next send) | continue|c | quit")
            }
            other => eprintln!("(auditor) unknown: {other:?} (try help)"),
        }
    }
    vm.debug.session_depth -= 1;
}

fn arm_step(vm: &mut VmState, kind: StepKind, bci: usize) {
    let fp = vm.stack.fp;
    vm.debug.step = Some(StepPlan {
        kind,
        base_fp: fp,
        base_serial: frame_serial(vm, fp),
        base_bci: bci,
    });
}

/// The interpreter-frame chain, innermost first (the halt loop's
/// selectable frames — compiled frames appear in `bt` read-only, §3.3).
fn interp_frames(vm: &VmState) -> Vec<usize> {
    let mut out = Vec::new();
    if !vm.stack.has_frame() {
        return out;
    }
    let mut fp = vm.stack.fp as i64;
    while fp != crate::oops::layout::ENTRY_FRAME_SENTINEL && out.len() < 256 {
        out.push(fp as usize);
        fp = crate::interpreter::stack::Frame { fp: fp as usize }.saved_fp(&vm.stack);
    }
    out
}

/// §3.3 / DBG2: the mixed-tier backtrace — `walk_frames` verbatim, with
/// every `Compiled` frame EXPANDED into one line per inlined virtual frame
/// via its scope chain at the frame's return safepoint (read-only;
/// adapter/call-stub frames elided per the doc). Guarded: a torn walk is
/// reported, not fatal.
pub fn bt(vm: &VmState) {
    let printed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut i = 0usize;
        crate::runtime::frames::walk_frames(vm, |fv| {
            match fv {
                crate::runtime::frames::FrameView::Interpreted { fp } => {
                    let f = crate::interpreter::stack::Frame { fp };
                    let m = f.method(&vm.stack);
                    let sel = SymbolOop::try_from(m.selector())
                        .map(|s| s.as_string())
                        .unwrap_or_else(|| "?".into());
                    let holder = KlassOop::try_from(m.holder())
                        .map(|k| crate::runtime::error::name_of(k.name()))
                        .unwrap_or_else(|| "?".into());
                    eprintln!("#{i} [interp]   {holder}>>{sel} (fp {fp})");
                    i += 1;
                }
                crate::runtime::frames::FrameView::Compiled { fp, ret_pc, nm } => {
                    for line in compiled_virtual_frames(vm, nm, ret_pc) {
                        eprintln!("#{i} [compiled] {line} (fp {fp:#x}, read-only)");
                        i += 1;
                    }
                }
                // Elided from the user-facing trace (§3.3).
                crate::runtime::frames::FrameView::Adapter { .. }
                | crate::runtime::frames::FrameView::CallStub { .. } => {}
            }
        });
    }));
    if let Err(e) = printed {
        let msg = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic".into());
        eprintln!("(bt died: {msg} — a torn walk is itself a finding)");
    }
}

/// DBG2: one `Holder>>selector @bci` line per virtual frame of a compiled
/// activation, innermost first, decoded from the scope chain at `ret_pc`'s
/// nearest-below deopt safepoint. Read-only by design (§3.3): mutating a
/// live compiled frame is the deferred mid-stack-deopt machinery.
fn compiled_virtual_frames(
    vm: &VmState,
    nm_id: crate::codecache::nmethod::NmethodId,
    ret_pc: u64,
) -> Vec<String> {
    let Some(nm) = vm.code_table.get(nm_id) else {
        return vec![format!("nmethod #{} (vanished)", nm_id.0)];
    };
    let code_off = (ret_pc - nm.code.base as u64) as u32;
    let i = nm.deopt_pcdescs.partition_point(|d| d.code_off <= code_off);
    if i == 0 {
        return vec![format!(
            "nmethod #{} v{} +{code_off:#x} (no scope recorded)",
            nm_id.0, nm.version
        )];
    }
    let desc = nm.deopt_pcdescs[i - 1];
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let site = crate::compiler::scopes::decode_site(&nm.deopt_scopes, desc.site_off);
        let mut lines = Vec::new();
        let mut scope_off = Some(site.scope_off);
        let mut bci = site.bci;
        while let Some(off) = scope_off {
            let scope = crate::compiler::scopes::decode_scope(&nm.deopt_scopes, off);
            let name = pool_method_name(vm, nm, scope.method_pool_ix);
            lines.push(format!(
                "{name} @{bci}{} [nm #{} v{}]",
                if scope.is_block { " [block]" } else { "" },
                nm_id.0,
                nm.version
            ));
            match scope.sender {
                Some((s_off, s_bci, _)) => {
                    scope_off = Some(s_off);
                    bci = s_bci;
                }
                None => scope_off = None,
            }
        }
        lines
    }));
    out.unwrap_or_else(|_| vec![format!("nmethod #{} (scope decode died)", nm_id.0)])
}

fn pool_method_name(vm: &VmState, nm: &crate::codecache::nmethod::Nmethod, pool_ix: u32) -> String {
    let _ = vm;
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let oop = crate::codecache::deopt_trap::read_pool_oop(nm, pool_ix);
        match MethodOop::try_from(oop) {
            Some(m) => {
                let sel = SymbolOop::try_from(m.selector())
                    .map(|s| s.as_string())
                    .unwrap_or_else(|| "?".into());
                let holder = KlassOop::try_from(m.holder())
                    .map(|k| crate::runtime::error::name_of(k.name()))
                    .unwrap_or_else(|| "?".into());
                format!("{holder}>>{sel}")
            }
            None => "<non-method pool entry>".into(),
        }
    }));
    out.unwrap_or_else(|_| "<unreadable>".into())
}

fn show_frame(vm: &VmState, fp: usize, index: usize) {
    eprint!("{}", frame_text(vm, fp, index));
}

/// The frame-inspection text (DBG4 string core — the CLI prints it, the GUI
/// report embeds it).
fn frame_text(vm: &VmState, fp: usize, index: usize) -> String {
    let f = crate::interpreter::stack::Frame { fp };
    let m = f.method(&vm.stack);
    let sel = SymbolOop::try_from(m.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".into());
    let recv = f.receiver(&vm.stack);
    let mut out = format!(
        "frame #{index}: {sel} (argc {}, ntemps {}) receiver={}\n",
        m.argc(),
        m.ntemps(),
        crate::memory::print_oop(&vm.universe, recv)
    );
    for t in 0..(m.argc() + m.ntemps()) {
        let v = f.temp(&vm.stack, t);
        let kind = if t < m.argc() { "arg " } else { "temp" };
        out.push_str(&format!(
            "  {kind}[{t}] = {}\n",
            crate::memory::print_oop(&vm.universe, v)
        ));
    }
    out
}

/// §3.2 `print <expr>` — compiled via the frontend's EXISTING doit path
/// with the SELECTED frame's receiver's class as holder (so ivar names
/// resolve), run via the reentrant door (`run_method_reentrant` saves and
/// restores the paused activation). `session_depth` is already > 0, so
/// nothing the doit does can recursively halt (§3.6).
fn print_expr(vm: &mut VmState, fp: usize, expr: &str) {
    eprintln!("{}", print_expr_text(vm, fp, expr));
}

/// The `print <expr>` evaluation (DBG4 string core — CLI prints, GUI report
/// embeds). Same doit path + reentrant door + `session_depth` guard as ever.
fn print_expr_text(vm: &mut VmState, fp: usize, expr: &str) -> String {
    let receiver = crate::interpreter::stack::Frame { fp }.receiver(&vm.stack);
    let holder = crate::runtime::lookup::klass_of(vm, receiver);
    // The top-item grammar wants a statement terminator; supply it so
    // `print count` works as typed.
    let expr = format!("{}.", expr.trim().trim_end_matches('.'));
    let stmt = match crate::frontend::parser::parse_one_top_item(&expr) {
        Ok(Some(crate::frontend::ast::TopItem::DoIt(stmt))) => stmt,
        Ok(Some(_)) => return "print: expected an expression, not a class definition".into(),
        Ok(None) => return String::new(),
        Err(e) => return format!("print: parse error: {e}"),
    };
    match crate::frontend::codegen::compile_doit(vm, holder, stmt) {
        Ok(m) => {
            let result = crate::interpreter::run_method_reentrant(vm, m, receiver, &[]);
            print_result(vm, result)
        }
        Err(e) => format!("print: {e}"),
    }
}

fn print_result(vm: &mut VmState, result: Oop) -> String {
    let klass = crate::runtime::lookup::klass_of(vm, result);
    let sel = vm.universe.intern(b"printString");
    if let Some(m) = crate::runtime::lookup::lookup(vm, klass, sel) {
        let s = crate::interpreter::run_method_reentrant(vm, m, result, &[]);
        if let Some(b) = crate::oops::wrappers::ByteArrayOop::try_from(s) {
            let mut buf = Vec::new();
            b.copy_bytes_out(&mut buf);
            return String::from_utf8_lossy(&buf).into_owned();
        }
    }
    crate::memory::print_oop(&vm.universe, result)
}

/// Pin `method` to tier-0 WITHOUT a breakpoint — the differential
/// diagnosis lever (docs/DEBUGGER.md §3.5, "selective de-optimization").
/// Identical to `set_breakpoint`'s step 3 (block future compiles +
/// invalidate existing nmethods via the redefinition path), minus the
/// halt: the method simply runs interpreted from now on. Answers "does
/// forcing THIS method interpreted change the result?" — the fastest way
/// to localize a wrong-value-from-compiled-code bug to one method. Kept in
/// `debug.pinned` so `unpin` can restore eligibility.
pub fn pin_method(vm: &mut VmState, method: MethodOop) -> Result<String, String> {
    let holder = KlassOop::try_from(method.holder());
    let holder_name = holder
        .map(|k| crate::runtime::error::name_of(k.name()))
        .unwrap_or_else(|| "?".into());
    let selector_name = SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".into());
    let hash = vm.universe.identity_hash(method.oop());
    let restore_compile = !method.compile_disabled();
    vm.debug.pinned.insert(hash, restore_compile);
    method.set_compile_disabled();
    if let (Some(k), Some(s)) = (holder, SymbolOop::try_from(method.selector())) {
        crate::runtime::deps::invalidate_dependents(vm, k, s);
    }
    Ok(format!("pinned to tier-0: {holder_name}>>{selector_name}"))
}

/// Reverse [`pin_method`]: restore tier-up eligibility (only if we were the
/// one that disabled it). Recompilation then resumes naturally by counters.
pub fn unpin_method(vm: &mut VmState, method: MethodOop) -> Result<String, String> {
    let hash = vm.universe.identity_hash(method.oop());
    let selector_name = SymbolOop::try_from(method.selector())
        .map(|s| s.as_string())
        .unwrap_or_else(|| "?".into());
    match vm.debug.pinned.remove(&hash) {
        Some(true) => {
            method.clear_compile_disabled();
            Ok(format!("unpinned: {selector_name}"))
        }
        Some(false) => Ok(format!(
            "unpinned: {selector_name} (was already ineligible)"
        )),
        None => Err(format!("{selector_name} was not pinned")),
    }
}

/// [`pin_method`] by class/selector name.
pub fn pin_by_name(vm: &mut VmState, class: &str, selector: &str) -> Result<String, String> {
    let method = resolve_method_by_name(vm, class, selector)?;
    pin_method(vm, method)
}

/// [`unpin_method`] by class/selector name.
pub fn unpin_by_name(vm: &mut VmState, class: &str, selector: &str) -> Result<String, String> {
    let method = resolve_method_by_name(vm, class, selector)?;
    unpin_method(vm, method)
}

/// Name-based breakpoint setting — the shared door `main.rs`
/// (`MACVM_DEBUG=break:`) and the RUSTTCL `bp` verb both use. Resolution
/// mirrors RUSTTCL's own: the class via the globals table, the method via
/// the REAL lookup walk (an inherited method resolves exactly as a send
/// would). `Class class` suffix selects the metaclass side.
pub fn set_breakpoint_by_name(
    vm: &mut VmState,
    class: &str,
    selector: &str,
    bci: u16,
) -> Result<String, String> {
    let method = resolve_method_by_name(vm, class, selector)?;
    set_breakpoint(vm, method, bci)
}

/// The clearing twin of [`set_breakpoint_by_name`].
pub fn clear_breakpoint_by_name(
    vm: &mut VmState,
    class: &str,
    selector: &str,
    bci: u16,
) -> Result<String, String> {
    let method = resolve_method_by_name(vm, class, selector)?;
    clear_breakpoint(vm, method, bci)
}

fn resolve_method_by_name(
    vm: &mut VmState,
    class: &str,
    selector: &str,
) -> Result<MethodOop, String> {
    let (name, class_side) = match class.strip_suffix(" class") {
        Some(base) => (base, true),
        None => (class, false),
    };
    let sym = vm.universe.intern(name.as_bytes());
    let assoc = crate::runtime::globals::global_lookup(vm, sym)
        .ok_or_else(|| format!("no such global: {name}"))?;
    let value = crate::oops::wrappers::MemOop::try_from(assoc)
        .expect("global association is a mem oop")
        .body_oop(1);
    let mut klass = KlassOop::try_from(value).ok_or_else(|| format!("{name} is not a class"))?;
    if class_side {
        klass = crate::runtime::lookup::klass_of(vm, klass.oop());
    }
    let sel = vm.universe.intern(selector.as_bytes());
    crate::runtime::lookup::lookup(vm, klass, sel)
        .ok_or_else(|| format!("{selector} not understood by {class}"))
}

/// Called by `lookup::install_method` on every method install while
/// `debug.active`: if the freshly installed `(holder, selector)` matches a
/// pending name-based spec, set the breakpoint now. Consumes matched
/// specs.
pub fn on_method_installed(
    vm: &mut VmState,
    holder: KlassOop,
    selector: SymbolOop,
    method: MethodOop,
) {
    if vm.debug.pending.is_empty() && vm.debug.pending_pins.is_empty() {
        return;
    }
    let holder_name = crate::runtime::error::name_of(holder.name());
    let sel_name = selector.as_string();
    // `Class class` names the metaclass side; install_method's holder for a
    // class-side method IS the metaclass, whose name prints as the base
    // class's — match on the base name either way.
    let matches = |spec: &str| spec.strip_suffix(" class").unwrap_or(spec) == holder_name;
    let mut i = 0;
    while i < vm.debug.pending.len() {
        let (c, s, bci) = vm.debug.pending[i].clone();
        if matches(&c) && s == sel_name {
            vm.debug.pending.remove(i);
            match set_breakpoint(vm, method, bci) {
                Ok(msg) => eprintln!("{msg} (deferred)"),
                Err(e) => eprintln!("MACVM_DEBUG: {e}"),
            }
        } else {
            i += 1;
        }
    }
    let mut i = 0;
    while i < vm.debug.pending_pins.len() {
        let (c, s) = vm.debug.pending_pins[i].clone();
        if matches(&c) && s == sel_name {
            vm.debug.pending_pins.remove(i);
            match pin_method(vm, method) {
                Ok(msg) => eprintln!("{msg} (deferred)"),
                Err(e) => eprintln!("MACVM_PIN: {e}"),
            }
        } else {
            i += 1;
        }
    }
}

/// `MACVM_PIN=Class>>sel[,Class>>sel...]` grammar (no bci — a whole method).
pub fn parse_pin_spec(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .filter_map(|part| {
            let (c, s) = part.split_once(">>")?;
            let c = c.trim();
            let s = s.trim();
            if c.is_empty() || s.is_empty() {
                None
            } else {
                Some((c.to_string(), s.to_string()))
            }
        })
        .collect()
}

/// `MACVM_DEBUG` grammar: `1` (just arm `debug.active`) or
/// `break:Class>>sel@bci[,Class>>sel@bci...]` (arm + set breakpoints once
/// the world is loaded — the scripted-gate entry). Parsed by `main.rs`
/// after world load; pure function for testability.
pub fn parse_debug_spec(raw: &str) -> Vec<(String, String, u16)> {
    let mut out = Vec::new();
    let Some(rest) = raw.strip_prefix("break:") else {
        return out;
    };
    for part in rest.split(',') {
        let Some((class_sel, bci)) = part.rsplit_once('@') else {
            continue;
        };
        let Some((class, sel)) = class_sel.split_once(">>") else {
            continue;
        };
        if let Ok(bci) = bci.parse::<u16>() {
            out.push((class.trim().to_string(), sel.trim().to_string(), bci));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_spec_grammar() {
        assert_eq!(parse_debug_spec("1"), vec![]);
        assert_eq!(
            parse_debug_spec("break:Foo>>bar@3,Baz>>quux:@0"),
            vec![
                ("Foo".to_string(), "bar".to_string(), 3),
                ("Baz".to_string(), "quux:".to_string(), 0)
            ]
        );
    }
}
