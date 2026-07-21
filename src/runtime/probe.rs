//! DBG0 — PROBE's crash dossier (docs/DEBUGGER.md §4.2). The report a
//! fatal trigger (SIGSEGV/SIGBUS in the code cache, `brk #0xDE02`, or a
//! fatal guest error) emits before the process dies: pc verdict, scope-
//! chain provenance, annotated registers, compiler-claim-vs-reality,
//! raw-fp walkback, heap spot-checks, and the recent-history ring.
//!
//! Trust model (§4.5): the VM is the SUSPECT. Every raw pointer derived
//! from the crash context is range-validated before it is dereferenced
//! (`catch_unwind` does not catch faults); heap-dependent pretty-printing
//! (selector names) is best-effort and printed AFTER the raw numeric
//! facts; every section flushes before the next begins, so a recursive
//! fault (backstopped by `deopt_trap`'s reentrancy guard) still leaves the
//! prefix. Nothing here ever touches the Smalltalk allocator, deopts a
//! frame, or resumes execution — every exit path terminates the process.

// This module only (not the rest of `runtime`) — the same exception
// `frames.rs` declares, for the same reason: reading raw crash-scene
// memory (register values as candidate pointers, frame slots on a frozen
// native stack, pthread stack bounds) has no safe-Rust equivalent. Every
// unsafe read here is preceded by an explicit range check — §4.5's
// no-unguarded-deref rule — because on this path `catch_unwind` cannot
// save us from being wrong.
#![allow(unsafe_code)]

use std::io::Write as _;

use crate::codecache::nmethod::NmState;
use crate::oops::layout::{MARK_TAG, MEM_TAG, RESERVED_TAG, TAG_MASK};
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::runtime::vm_state::{ProbeEvent, VmState};

/// The register file at the moment of the trigger, copied out of the
/// signal ucontext (or synthesized for synchronous triggers). Plain data —
/// no signal-specific types leak out of `deopt_trap`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CapturedRegs {
    pub x: [u64; 29],
    pub fp: u64,
    pub lr: u64,
    pub sp: u64,
    pub pc: u64,
    pub cpsr: u32,
    /// SIGSEGV/SIGBUS fault address (`__es.__far`); 0 for brk triggers.
    pub far: u64,
    /// Signal number, or 0 for a synchronous (brk / guest-fatal) trigger.
    pub sig: i32,
}

/// The human dossier's opening marker — gate tests assert THIS (plus the
/// JSON `schema` field), never the bare exit code: exit(70) is shared with
/// heap exhaustion and stack overflow (`alloc.rs`, `stack.rs`).
pub const DOSSIER_MARKER: &str = "==== MACVM PROBE DOSSIER v1 ====";

/// JSON schema version — pinned from day one (docs/DEBUGGER.md §7.4) so
/// dossier-asserting goldens survive additions.
pub const DOSSIER_SCHEMA: u32 = 1;

fn flush() {
    let _ = std::io::stderr().flush();
}

/// The full crash dossier, then `exit(70)`. `trigger` is the one-line
/// cause ("SIGSEGV", "brk #0xDE02", ...).
///
/// # Safety
/// `vm` must point at the live VmState (recovered from x28 after a
/// registered-range fault, or passed directly on synchronous paths).
pub unsafe fn crash_dossier(vm: &mut VmState, regs: &CapturedRegs, trigger: &str) -> ! {
    eprintln!("{DOSSIER_MARKER}");
    eprintln!(
        "trigger: {trigger}   pc: {:#x}   far: {:#x}",
        regs.pc, regs.far
    );
    flush();

    let mut json = JsonDossier::new(trigger, regs);

    // ── 1. Verdict: what code owns pc. Raw numeric facts first (§4.5) —
    // the heap-derived key names come after, guarded. ────────────────────
    let verdict = classify_pc(vm, regs.pc);
    eprintln!("[1] verdict: {verdict}");
    json.field("verdict", &verdict);
    flush();

    // ── 2. Provenance: pc → nearest deopt safepoint → scope chain. ──────
    if let Some(nm_id) = vm.code_table.find_by_pc(regs.pc) {
        for line in provenance_lines(vm, nm_id, regs.pc) {
            eprintln!("[2] {line}");
            json.push("provenance", &line);
        }
    } else {
        eprintln!("[2] provenance: (pc not in an nmethod — none)");
    }
    flush();

    // ── 3. Registers, annotated. ─────────────────────────────────────────
    for i in 0..29 {
        let a = annotate_value(vm, regs.x[i]);
        eprintln!("[3] x{i:<2} = {:#018x}  {a}", regs.x[i]);
        json.push("registers", &format!("x{i}={:#x} {a}", regs.x[i]));
    }
    eprintln!(
        "[3] fp  = {:#018x}  {}",
        regs.fp,
        annotate_value(vm, regs.fp)
    );
    eprintln!(
        "[3] lr  = {:#018x}  {}",
        regs.lr,
        annotate_value(vm, regs.lr)
    );
    eprintln!("[3] sp  = {:#018x}", regs.sp);
    eprintln!("[3] cpsr= {:#010x}", regs.cpsr);
    flush();

    // ── 4. Compiler's claim vs reality at the nearest safepoint. ─────────
    if let Some(nm_id) = vm.code_table.find_by_pc(regs.pc) {
        for line in claims_vs_reality(vm, nm_id, regs) {
            eprintln!("[4] {line}");
            json.push("claims", &line);
        }
    }
    flush();

    // ── 5. Disassembly window: ±16 instructions around the crash pc. ─────
    for line in disasm_window(vm, regs.pc) {
        eprintln!("[5] {line}");
    }
    flush();

    // ── 6. Walkback: raw-fp native walk + interpreter chain. ─────────────
    walkback(vm, regs, &mut |line| {
        eprintln!("[6] {line}");
    });
    flush();

    // ── 7. Heap spot-check under catch_unwind (panics are findings). ─────
    let verify_line = heap_verify_line(vm);
    eprintln!("[7] {verify_line}");
    json.field("verify", &verify_line);
    flush();

    // ── 8. JSON copy. ─────────────────────────────────────────────────────
    if let Ok(path) = std::env::var("MACVM_PROBE_DUMP") {
        match std::fs::write(&path, json.finish()) {
            Ok(()) => eprintln!("[8] json dossier: {path}"),
            Err(e) => eprintln!("[8] json dossier FAILED ({path}): {e}"),
        }
    }

    // ── 9. Recent history (the ring). ────────────────────────────────────
    ring_dump(vm, &mut |line| eprintln!("[9] {line}"));
    eprintln!("==== END DOSSIER (exit 70) ====");
    flush();
    crate::runtime::vm_state::fatal_exit(70);
}

/// The fatal-guest-error mini-dossier (docs/DEBUGGER.md §4.1's last row):
/// walkback + reg-block/tier-link state + ring + heap verify, on the
/// ordinary Rust stack with a coherent interpreter — no signal machinery.
/// The caller (`error.rs`) still owns the exit itself.
pub fn fatal_guest_report(vm: &VmState, headline: &str) {
    eprintln!("{DOSSIER_MARKER}");
    eprintln!("trigger: fatal guest error — {headline}");
    eprintln!(
        "[state] regs.bci={} regs.method={} stack.fp={:#x} stack.sp={:#x} tier_links={} \
         pending_deopts={} anchor fp={:#x} pc={:#x} kind={}",
        vm.regs.bci,
        vm.regs
            .method
            .map(|m| format!("{:#x}", m.oop().raw()))
            .unwrap_or_else(|| "None".into()),
        vm.stack.fp,
        vm.stack.sp,
        vm.tier_links.len(),
        vm.pending_deopts.len(),
        vm.reg_block.last_compiled_fp,
        vm.reg_block.last_compiled_pc,
        vm.reg_block.last_compiled_kind,
    );
    // The interpreter state is coherent here, so the VANILLA walker is the
    // right tool — under catch_unwind, its panic being a finding.
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut n = 0usize;
        crate::runtime::frames::walk_frames(vm, |fv| {
            eprintln!("[walk] {fv:?}");
            n += 1;
        });
        n
    }));
    match walked {
        Ok(n) => eprintln!("[walk] {n} frames"),
        Err(e) => eprintln!(
            "[walk] DIED: {} (a torn walk is itself a finding)",
            panic_msg(&e)
        ),
    }
    eprintln!("[7] {}", heap_verify_line(vm));
    ring_dump(vm, &mut |line| eprintln!("[9] {line}"));
    eprintln!("==== END DOSSIER (fatal guest error) ====");
    flush();
}

/// Is the PROBE report enabled for fatal guest errors? Default ON (the
/// dossier is the point); `MACVM_PROBE=off` silences it for tests that
/// pin the exact legacy stderr shape.
pub fn guest_report_enabled() -> bool {
    std::env::var("MACVM_PROBE").as_deref() != Ok("off")
}

// ───────────────────────────── internals ─────────────────────────────────

fn panic_msg(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| e.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

/// Dossier step 1: name the code that owns `pc`. Every branch is a plain
/// range compare over VmState-resident tables; the only heap reads are the
/// best-effort key names, tag-guarded.
pub fn classify_pc(vm: &VmState, pc: u64) -> String {
    if let Some(id) = vm.code_table.find_by_pc(pc) {
        let nm = vm.code_table.get(id).expect("find_by_pc implies get");
        let off = pc - nm.code.base as u64;
        let state = match nm.state {
            NmState::Alive => "Alive",
            NmState::NotEntrant => "NotEntrant",
            NmState::Zombie => "Zombie",
        };
        let name = guarded_key_name(nm.key_klass, nm.key_selector);
        return format!(
            "nmethod #{} v{} {state} +{off:#x} (len {:#x}) {name}",
            id.0, nm.version, nm.code.len
        );
    }
    let s = &vm.stubs;
    for (h, name) in [
        (s.call_stub, "call_stub"),
        (s.stub_poll, "stub_poll"),
        (s.resolve, "stub_resolve"),
        (s.c2i_shared, "c2i_shared"),
        (s.mega_shared, "mega_shared"),
        (s.dnu, "stub_dnu"),
        (s.must_be_boolean, "stub_must_be_boolean"),
        (s.alloc_slow, "stub_alloc_slow"),
        (s.not_entrant, "not_entrant_stub"),
        (s.deopt_return, "deopt_return_trampoline"),
    ] {
        let base = h.base as u64;
        if pc >= base && pc < base + h.len as u64 {
            return format!("{name} +{:#x}", pc - base);
        }
    }
    if let Some(t) = &vm.deopt_trampolines {
        for (h, name) in [
            (t.uncommon, "deopt_uncommon_trampoline"),
            (t.assert, "deopt_assert_stub"),
        ] {
            let base = h.base as u64;
            if pc >= base && pc < base + h.len as u64 {
                return format!("{name} +{:#x}", pc - base);
            }
        }
    }
    if let Some(entries) = vm.pic_table.contains_pc(pc) {
        return format!("PIC stub ({entries} entries)");
    }
    if vm.mega_table.contains_pc(pc) {
        return "mega trampoline".into();
    }
    if vm.adapters.contains_pc(pc) {
        return "c2i adapter trampoline".into();
    }
    if vm.code_cache.contains(pc) {
        // patch_branch26 veneers are recorded nowhere (DEBUGGER.md §1) —
        // this honest class is the best available today.
        return "in code cache, unnamed (possibly a branch veneer)".into();
    }
    "FOREIGN (not in any registered code range — a Rust-side pc)".into()
}

/// Best-effort `Klass>>selector` from an nmethod's key pair — tag-guarded:
/// on a corrupt heap these reads must degrade, not fault.
fn guarded_key_name(klass: KlassOop, sel: SymbolOop) -> String {
    let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let k = crate::runtime::error::name_of(klass.name());
        let s = sel.as_string();
        format!("{k}>>{s}")
    }));
    guarded.unwrap_or_else(|_| "<key names unreadable>".into())
}

/// Nearest-below search over the S13 deopt PcDescs — the non-panicking
/// lookup the existing exact-match APIs don't provide (DEBUGGER.md §1).
fn nearest_deopt_desc(
    nm: &crate::codecache::nmethod::Nmethod,
    code_off: u32,
) -> Option<crate::compiler::scopes::PcDesc> {
    let i = nm.deopt_pcdescs.partition_point(|d| d.code_off <= code_off);
    if i == 0 {
        None
    } else {
        Some(nm.deopt_pcdescs[i - 1])
    }
}

/// Dossier step 2: "`Holder>>sel @bci`, one line per inlined virtual
/// frame", at the nearest at-or-below deopt safepoint. Whole step guarded.
fn provenance_lines(
    vm: &VmState,
    id: crate::codecache::nmethod::NmethodId,
    pc: u64,
) -> Vec<String> {
    let nm = match vm.code_table.get(id) {
        Some(nm) => nm,
        None => return vec!["provenance: nmethod vanished".into()],
    };
    let code_off = (pc - nm.code.base as u64) as u32;
    let desc = match nearest_deopt_desc(nm, code_off) {
        Some(d) => d,
        None => {
            return vec![format!(
                "provenance: no deopt safepoint at-or-below +{code_off:#x}"
            )]
        }
    };
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let site = crate::compiler::scopes::decode_site(&nm.deopt_scopes, desc.site_off);
        let mut lines = vec![format!(
            "{} instruction bytes past the {:?} safepoint at +{:#x} (bci {}, reexecute {})",
            code_off - desc.code_off,
            site.kind,
            desc.code_off,
            site.bci,
            site.reexecute
        )];
        let mut scope_off = Some(site.scope_off);
        let mut bci = site.bci;
        while let Some(off) = scope_off {
            let scope = crate::compiler::scopes::decode_scope(&nm.deopt_scopes, off);
            lines.push(format!(
                "  {} @bci {}{}",
                guarded_pool_method_name(vm, nm, scope.method_pool_ix),
                bci,
                if scope.is_block { " [block]" } else { "" }
            ));
            match scope.sender {
                Some((sender_off, sender_bci, _)) => {
                    scope_off = Some(sender_off);
                    bci = sender_bci;
                }
                None => scope_off = None,
            }
        }
        lines
    }));
    out.unwrap_or_else(|e| vec![format!("provenance DIED: {}", panic_msg(&e))])
}

fn guarded_pool_method_name(
    vm: &VmState,
    nm: &crate::codecache::nmethod::Nmethod,
    pool_ix: u32,
) -> String {
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let oop = crate::codecache::deopt_trap::read_pool_oop(nm, pool_ix);
        match MethodOop::try_from(oop) {
            Some(m) => {
                let sel = SymbolOop::try_from(m.selector())
                    .map(|s| s.as_string())
                    .unwrap_or_else(|| "<non-symbol selector>".into());
                let holder = KlassOop::try_from(m.holder())
                    .map(|k| crate::runtime::error::name_of(k.name()))
                    .unwrap_or_else(|| "?".into());
                format!("{holder}>>{sel}")
            }
            None => format!("<pool[{pool_ix}] not a CompiledMethod: {:#x}>", oop.raw()),
        }
    }));
    let _ = vm; // vm reserved for future name caches; silence unused warnings
    out.unwrap_or_else(|_| format!("<pool[{pool_ix}] unreadable>"))
}

/// Dossier step 3's per-value annotation — tag class, region, and (for mem
/// oops whose address lies inside a heap space) the stack auditor's
/// mark-plausibility verdict. The region check comes BEFORE the mark-word
/// deref: §4.5's no-unguarded-deref rule.
pub fn annotate_value(vm: &VmState, v: u64) -> String {
    let mut notes: Vec<String> = Vec::new();
    let a = v as usize;
    let eden = &vm.universe.eden;
    let from = &vm.universe.from;
    let to = &vm.universe.to;
    let old = &vm.universe.old;
    let region = if a >= eden.start && a < eden.end {
        Some(if a < eden.top { "eden" } else { "eden(unused)" })
    } else if a >= from.start && a < from.end {
        Some("from-space")
    } else if a >= to.start && a < to.end {
        Some("to-space")
    } else if old.bounds.contains(a) {
        Some("old-gen")
    } else {
        None
    };
    match v & TAG_MASK {
        t if t == MEM_TAG => {
            notes.push("mem-oop".into());
            let addr = v - MEM_TAG;
            let ra = addr as usize;
            let target_region = if ra >= eden.start && ra < eden.top {
                Some("eden")
            } else if ra >= from.start && ra < from.top {
                Some("from-space")
            } else if ra >= to.start && ra < to.top {
                Some("to-space")
            } else if old.bounds.contains(ra) && ra < old.top {
                Some("old-gen")
            } else {
                None
            };
            match target_region {
                Some(r) => {
                    notes.push(format!("→{r}"));
                    // In-range: the mark deref is licensed.
                    let w = unsafe { *(ra as *const u64) };
                    let plausible = w & TAG_MASK == MARK_TAG && w & 0b100 != 0;
                    notes.push(if plausible {
                        "mark:ok".into()
                    } else {
                        format!("mark:IMPLAUSIBLE({w:#x})")
                    });
                }
                None => notes.push("→OUTSIDE-HEAP (wild pointer!)".into()),
            }
        }
        t if t == MARK_TAG => notes.push("mark-tagged (not a value!)".into()),
        t if t == RESERVED_TAG => notes.push("reserved-tag (sentinel?)".into()),
        _ => {
            notes.push(format!("smi({})", (v as i64) >> 2));
            // The dead-frame heuristic that cracked BUG D root cause 4b.
            if (0x1_0000_0000u64..0x1_7000_0000u64).contains(&v) {
                notes.push("IN CODE/STACK ADDRESS BAND (dead-frame residue?)".into());
            }
        }
    }
    if vm.code_cache.contains(v) {
        notes.push("=code-cache-addr".into());
    }
    if let Some(r) = region {
        notes.push(format!("(raw addr in {r})"));
    }
    notes.join(" ")
}

/// DBG5 (compiled_send_auditor_design.md §D2): the cheap, allocation-free
/// companion to [`annotate_value`]. Is `v` a mem-tagged word that does NOT
/// point into any USED heap region (eden/from/to/old, each up to its own
/// `top`)? That is the exact signature of a corrupt oop slot — a wild/freed
/// pointer, or the `0x…01` (`MEM_TAG` on a null base) an uninitialized or
/// stale spill slot leaves (the A3b materialized-Context crash). smi/nil/mark
/// words and every in-range live oop return `false`. Used to pre-filter the
/// `MACVM_TRACE=oops` at-GC scan so `annotate_value`'s formatting/allocation
/// only runs for genuinely suspect slots, not the thousands of healthy ones.
/// Mirrors `annotate_value`'s own `target_region` range check exactly; the
/// §4.5 rule (range-check before any deref) is preserved — this NEVER
/// dereferences `v`, it only classifies its address arithmetic.
pub fn oop_is_suspect(vm: &VmState, v: u64) -> bool {
    if v & TAG_MASK != MEM_TAG {
        return false;
    }
    let ra = (v - MEM_TAG) as usize;
    let eden = &vm.universe.eden;
    let from = &vm.universe.from;
    let to = &vm.universe.to;
    let old = &vm.universe.old;
    let in_used_region = (ra >= eden.start && ra < eden.top)
        || (ra >= from.start && ra < from.top)
        || (ra >= to.start && ra < to.top)
        || (old.bounds.contains(ra) && ra < old.top);
    !in_used_region
}

/// Dossier step 4: for the nearest safepoint, print where the compiler
/// SAYS each value lives and what is actually there.
fn claims_vs_reality(
    vm: &VmState,
    id: crate::codecache::nmethod::NmethodId,
    regs: &CapturedRegs,
) -> Vec<String> {
    let nm = match vm.code_table.get(id) {
        Some(nm) => nm,
        None => return vec![],
    };
    let code_off = (regs.pc - nm.code.base as u64) as u32;
    let desc = match nearest_deopt_desc(nm, code_off) {
        Some(d) => d,
        None => return vec!["claims: no deopt safepoint at-or-below pc".into()],
    };
    let (lo, hi) = thread_stack_bounds();
    let fp_ok = regs.fp >= lo && regs.fp < hi && regs.fp & 7 == 0;
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use crate::compiler::scopes::ValueLoc;
        let site = crate::compiler::scopes::decode_site(&nm.deopt_scopes, desc.site_off);
        let scope = crate::compiler::scopes::decode_scope(&nm.deopt_scopes, site.scope_off);
        let mut lines = Vec::new();
        let mut show = |what: String, loc: &ValueLoc| {
            let claim = format!("{loc:?}");
            let reality = match loc {
                ValueLoc::FrameSlot(off) => {
                    if fp_ok {
                        let addr = (regs.fp as i64 + *off as i64) as u64;
                        if addr >= lo && addr + 8 <= hi {
                            // SAFETY: bounds-checked against the thread stack.
                            let v = unsafe { *(addr as *const u64) };
                            format!("{v:#x} {}", annotate_value(vm, v))
                        } else {
                            "slot outside stack bounds".into()
                        }
                    } else {
                        "fp invalid — unreadable".into()
                    }
                }
                _ => "(constant — nothing to diff)".into(),
            };
            lines.push(format!("{what}: claim {claim} → {reality}"));
        };
        show("receiver".into(), &scope.receiver);
        for (i, loc) in scope.slots.iter().enumerate() {
            show(format!("slot[{i}]"), loc);
        }
        for (i, loc) in site.stack.iter().enumerate() {
            show(format!("stack[{i}]"), loc);
        }
        // The GC's parallel claim: which frame slots the oopmap scans here.
        let i = nm.pcdescs.partition_point(|d| d.pc_off <= code_off);
        if i > 0 {
            let pd = &nm.pcdescs[i - 1];
            let om = &nm.oopmaps[pd.oopmap as usize];
            let slots: Vec<u16> = om.iter_slots().collect();
            lines.push(format!(
                "oopmap at +{:#x} (bci {}): oop slots {slots:?}",
                pd.pc_off, pd.bci
            ));
        }
        lines
    }));
    out.unwrap_or_else(|e| vec![format!("claims DIED: {}", panic_msg(&e))])
}

/// Dossier step 6: the raw-fp native walkback + the interpreter chain.
fn walkback(vm: &VmState, regs: &CapturedRegs, emit: &mut dyn FnMut(String)) {
    let (lo, hi) = thread_stack_bounds();
    emit(format!("thread stack: [{lo:#x}, {hi:#x})"));
    let (n, stop) =
        crate::runtime::frames::probe_walk_native(vm, regs.fp, regs.pc, lo, hi, |i, fp, pc| {
            emit(format!(
                "native #{i}: fp {fp:#x} pc {pc:#x} — {}",
                classify_pc(vm, pc)
            ));
        });
    emit(format!("native walk: {n} frames; {stop}"));
    // Interpreter chain — coherent slots, but guarded anyway (the
    // accessors expect-panic on shape violations).
    if vm.stack.has_frame() {
        let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut fp = vm.stack.fp as i64;
            let mut n = 0usize;
            let mut lines = Vec::new();
            while fp != crate::oops::layout::ENTRY_FRAME_SENTINEL && n < 128 {
                let frame = crate::interpreter::stack::Frame { fp: fp as usize };
                let m = frame.method(&vm.stack);
                let sel = SymbolOop::try_from(m.selector())
                    .map(|s| s.as_string())
                    .unwrap_or_else(|| "?".into());
                lines.push(format!("interp #{n}: fp {fp} {sel}"));
                fp = frame.saved_fp(&vm.stack);
                n += 1;
            }
            lines
        }));
        match walked {
            Ok(lines) => {
                for l in lines {
                    emit(l);
                }
            }
            Err(e) => emit(format!("interp walk DIED: {}", panic_msg(&e))),
        }
    }
}

/// Dossier step 5 (DBG3): disassemble ±16 instructions around the crash
/// pc, with the faulting word marked. Only meaningful when the pc is
/// inside a known nmethod (whose bytes we can slice safely — a published
/// block never moves); a foreign or stub pc yields a one-line note.
fn disasm_window(vm: &VmState, pc: u64) -> Vec<String> {
    let Some(id) = vm.code_table.find_by_pc(pc) else {
        return vec!["disasm: pc not in an nmethod (see verdict)".into()];
    };
    let Some(nm) = vm.code_table.get(id) else {
        return vec!["disasm: nmethod vanished".into()];
    };
    let base = nm.code.base as u64;
    let code_off = (pc - base) as usize;
    let window = 16 * 4; // ±16 instructions
    let lo = code_off.saturating_sub(window);
    let hi = (code_off + window + 4).min(nm.code.len);
    let bytes = &nm.code.as_bytes()[lo..hi];
    let listing =
        crate::compiler::disasm_a64::disasm_slice(bytes, Some(code_off.saturating_sub(lo)));
    // Re-base the +offset labels onto the nmethod so they read absolutely.
    listing
        .lines()
        .map(|l| {
            // Each line starts "+0xNN  ..." relative to `lo`; annotate the
            // nmethod offset it really is.
            l.to_string()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, l)| format!("nm+{:#06x}  {l}", lo + i * 4))
        .collect()
}

fn heap_verify_line(vm: &VmState) -> String {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::memory::verify::verify_heap_at(vm, crate::memory::verify::VerifyPoint::Manual)
    }));
    match r {
        Ok(Ok(())) => "heap verify: OK".into(),
        Ok(Err(e)) => format!("heap verify: FAILED — {e:?}"),
        Err(e) => format!("heap verify DIED: {} (itself a finding)", panic_msg(&e)),
    }
}

fn ring_dump(vm: &VmState, emit: &mut dyn FnMut(String)) {
    let shown: Vec<ProbeEvent> = vm.probe_ring.iter_oldest_first().collect();
    emit(format!(
        "recent history: showing last {} of {} events (oldest first)",
        shown.len(),
        vm.probe_ring.total
    ));
    for e in shown {
        emit(match e {
            ProbeEvent::Compile { nm, version } => format!("compile   nm={nm} v{version}"),
            ProbeEvent::Deopt { nm, bci, reexecute } => {
                format!("deopt     nm={nm} bci={bci} reexecute={reexecute}")
            }
            ProbeEvent::Invalidate { nm } => format!("invalidate nm={nm}"),
        });
    }
}

/// The current thread's stack bounds (macOS): `[addr - size, addr)`.
#[cfg(target_os = "macos")]
fn thread_stack_bounds() -> (u64, u64) {
    unsafe {
        let t = libc::pthread_self();
        let hi = libc::pthread_get_stackaddr_np(t) as u64;
        let size = libc::pthread_get_stacksize_np(t) as u64;
        (hi - size, hi)
    }
}

/// WINVM: the current thread's stack bounds via
/// `GetCurrentThreadStackLimits` (Windows 8+): `[low, high)`.
#[cfg(windows)]
fn thread_stack_bounds() -> (u64, u64) {
    extern "system" {
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
    }
    let (mut lo, mut hi) = (0usize, 0usize);
    // SAFETY: writes two out-params for the calling thread; no other
    // memory access.
    unsafe { GetCurrentThreadStackLimits(&mut lo, &mut hi) };
    (lo as u64, hi as u64)
}

/// Zero-dep JSON assembly (house style: gui/vm_host.rs hand-rolls too).
struct JsonDossier {
    fields: Vec<(String, String)>,
    arrays: Vec<(String, Vec<String>)>,
}

impl JsonDossier {
    fn new(trigger: &str, regs: &CapturedRegs) -> JsonDossier {
        let mut j = JsonDossier {
            fields: Vec::new(),
            arrays: Vec::new(),
        };
        j.field("trigger", trigger);
        j.field("pc", &format!("{:#x}", regs.pc));
        j.field("far", &format!("{:#x}", regs.far));
        j
    }

    fn field(&mut self, key: &str, val: &str) {
        self.fields.push((key.into(), val.into()));
    }

    fn push(&mut self, array: &str, val: &str) {
        match self.arrays.iter_mut().find(|(k, _)| k == array) {
            Some((_, v)) => v.push(val.into()),
            None => self.arrays.push((array.into(), vec![val.into()])),
        }
    }

    fn finish(&self) -> String {
        let esc = |s: &str| -> String {
            let mut out = String::with_capacity(s.len() + 2);
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out
        };
        let mut s = format!("{{\"schema\": {DOSSIER_SCHEMA}");
        for (k, v) in &self.fields {
            s.push_str(&format!(", \"{}\": \"{}\"", esc(k), esc(v)));
        }
        for (k, vs) in &self.arrays {
            let items: Vec<String> = vs.iter().map(|v| format!("\"{}\"", esc(v))).collect();
            s.push_str(&format!(", \"{}\": [{}]", esc(k), items.join(", ")));
        }
        s.push('}');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_and_pins_schema() {
        let regs = CapturedRegs::default();
        let mut j = JsonDossier::new("test \"quoted\"", &regs);
        j.push("lines", "a\nb");
        let s = j.finish();
        assert!(s.starts_with(&format!("{{\"schema\": {DOSSIER_SCHEMA}")));
        assert!(s.contains("test \\\"quoted\\\""));
        assert!(s.contains("a\\nb"));
    }

    /// DBG5 §D2: the `oop_is_suspect` classifier that drives `MACVM_TRACE=oops`.
    /// The at-GC scan's silence gate proves it doesn't false-positive on real
    /// code; this proves the other half — that it FLAGS a corrupt word — so a
    /// silent scan means "clean", not "broken".
    #[test]
    fn oop_is_suspect_flags_only_wild_mem_pointers() {
        use crate::oops::layout::MEM_TAG;
        use crate::runtime::vm_state::{VmOptions, VmState};
        use crate::runtime::JitMode;
        let vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: JitMode::Off,
        });
        // Real heap oops are never suspect (this also proves nil/true/klass
        // live in a used region, so a healthy slot holding them stays silent).
        assert!(!oop_is_suspect(&vm, vm.universe.nil_obj.raw()));
        assert!(!oop_is_suspect(&vm, vm.universe.true_obj.raw()));
        assert!(!oop_is_suspect(&vm, vm.universe.object_klass.oop().raw()));
        // Non-mem-tagged words are legal slot contents, never suspect.
        assert!(!oop_is_suspect(&vm, 0)); // raw 0 / smi 0
        assert!(!oop_is_suspect(&vm, (5i64 << 2) as u64)); // smi 5
                                                           // The A3b signature: MEM_TAG on a null base → untagged addr 0.
        assert!(oop_is_suspect(&vm, MEM_TAG));
        // A wild high-address mem oop in no region.
        assert!(oop_is_suspect(&vm, 0x7fff_0000_0000 | MEM_TAG));
    }

    #[test]
    fn ring_overwrites_oldest_and_counts_total() {
        let mut r = crate::runtime::vm_state::ProbeRing::new();
        for i in 0..300u32 {
            r.push(ProbeEvent::Invalidate { nm: i });
        }
        assert_eq!(r.total, 300);
        let got: Vec<u32> = r
            .iter_oldest_first()
            .map(|e| match e {
                ProbeEvent::Invalidate { nm } => nm,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(got.len(), 256);
        assert_eq!(*got.first().unwrap(), 44); // 300 - 256
        assert_eq!(*got.last().unwrap(), 299);
    }
}
