//! S13 step 6 — frame materialization (`sprint_s13_detail.md` D5 M0–M7).
//!
//! Turns one trapped/returning COMPILED physical frame back into the N
//! interpreter frame(s) it stands for (N = 1 in S13; ≥1 once S14 inlining
//! records `SenderLink` chains) and pushes them onto the `ProcessStack`,
//! byte-identical to frames the interpreter's own `push_frame` builds — so
//! the resumed activation cannot tell it was ever compiled. This module
//! BUILDS the frames; it does NOT run them. The nested interpreter resume
//! (D5 M8, `interpret_active`) and the trampoline handoff (the
//! `rt_uncommon_trap` seam in `codecache::deopt_trap`) are S13 step 7.
//!
//! **Safe module.** `#![deny(unsafe_code)]`: every raw native-stack /
//! code-cache read goes through a licensed helper in
//! `codecache::deopt_trap` (`read_frame_slot` for a `FrameSlot`, `read_pool_
//! oop` for a `ConstPool`). No `unsafe` appears below by construction; if a
//! new raw read is ever needed, it is added there (documented), never here.
//!
//! **Layer boundary.** This is the ONLY `runtime` module that reads a
//! compiled frame's native slots (via those two doors) AND builds interpreter
//! frames — the deopt seam sits exactly between the two representations, so
//! it necessarily touches both. It allocates (Contexts, M6) and therefore may
//! GC; every value read out of the compiled frame is either pushed onto the
//! `ProcessStack` (a GC root) before the next allocation or held in a
//! `HandleScope` across it — no bare oop is carried live across an allocating
//! call (the `M6`/`M0` ordering below makes this concrete).

#![deny(unsafe_code)]

use crate::codecache::deopt_trap::{read_frame_slot, read_frame_slot_raw, read_pool_oop};
use crate::codecache::nmethod::{Nmethod, NmethodId};
use crate::compiler::scopes::{CtxLoc, DecodedScope, DeoptState, ValueLoc};
use crate::interpreter::stack::{Frame, FrameActivation};
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::MethodOop;
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

/// What [`deoptimize_frame`] hands the nested interpreter run (D5 M8). The
/// materializer builds the frames and repoints `vm.stack.fp`/`sp`; this
/// carries the three facts the resume (`interpreter::interpret_active`) needs
/// that are NOT recoverable from `vm.stack` after materialization:
/// - `base_sp` — the M2 watermark; the run pops back to here when the
///   outermost frame returns through its `ENTRY_FRAME_SENTINEL` link (a
///   debug cross-check on the sentinel-driven stop).
/// - `resume_bci` — the innermost frame's M5 resume bci, the bci the nested
///   `dispatch` picks up at. This is the innermost frame's own START point; it
///   equals that frame's `saved_bci` header only at depth 1 (where the frame
///   has no materialized caller). At depth > 1 the header is the CALLER's resume
///   (where the caller picks up when the inlined callee returns), distinct from
///   this start bci.
/// - `saved_activation` — the AMBIENT outer stack activation (plain ints)
///   snapshotted at M2, BEFORE the M3 pushes clobbered `vm.stack.fp`/`sp`;
///   `interpret_active` restores it after the run so a paused outer
///   activation (the I→C→deopt case) resumes intact — the same reentrancy
///   guard `run_method_reentrant` performs at the C→I seam.
///
/// Deliberately NOT carried here: the ambient `vm.regs`. An `InterpRegs`
/// copy holds `method` as a bare oop, and this struct lives ACROSS the M3–M6
/// materializer, whose allocations can scavenge — the live `vm.regs.method`
/// is kept current by the interp-regs-mirror root, but a copy in this struct
/// is invisible to the root walker and goes stale (found live: GC_STRESS
/// deltablue died on the scavenger's to-space tripwire two cycles after a
/// deopt, via a handle minted FROM the stale copy). `interpret_active`
/// snapshots `vm.regs` itself, at its own entry — after materialization,
/// when the mirror-rooted value is GC-current.
#[derive(Copy, Clone, Debug)]
pub struct DeoptResume {
    pub base_sp: usize,
    pub resume_bci: usize,
    pub saved_activation: FrameActivation,
}

/// The compiled physical frame to deoptimize (`sprint_s13_detail.md` D5).
/// Holds no oops — just the coordinates the materializer reads slots against
/// — so it cannot go stale across the allocations M6 performs.
#[derive(Copy, Clone, Debug)]
pub struct FrameView {
    /// The trapped / returning compiled frame's FP (a raw native-stack
    /// address, the base `FrameSlot` offsets are measured from).
    pub fp: usize,
    /// Trap pc (uncommon-trap path) or `orig_ret_pc` (return-path deopt) —
    /// the native pc whose `PcDesc` names this frame's deopt state.
    pub pc: usize,
    /// The nmethod owning `pc` (its oop pool is where `ConstPool` /
    /// `method_pool_ix` values live, GC-current).
    pub nm: NmethodId,
    /// `Some(result)` on a `reexecute == false` (call-return) site: the
    /// completed call's value, pushed onto the innermost materialized frame's
    /// operand stack (D5 M3.3). `None` on a `reexecute == true` site (the
    /// recorded stack already holds the re-executing op's inputs).
    pub incoming_result: Option<Oop>,
}

/// One decoded virtual frame plus the two per-scope facts M3 needs from the
/// site/chain (its own header resume bci and pending operand stack). Built in
/// M0, consumed outermost → innermost in M3.
struct VirtualFrame {
    scope: DecodedScope,
    /// This frame's HEADER `saved_bci` — where its CALLER (the frame below it)
    /// resumes when THIS frame returns. Derived from THIS frame's OWN
    /// `SenderLink` (`sender_bci + len(send in the caller's method)`): the
    /// innermost frame's own sender describes the send in its caller, so its
    /// header saved_bci is the CALLER's post-send resume — NOT this frame's own
    /// start bci (that is `resume_bci`, the M5 value handed to
    /// `interpret_active`, and applies only to where the INNERMOST frame begins
    /// interpreting, not to any frame's header). The outermost (root) frame has
    /// no sender → `0` (it links to `ENTRY_FRAME_SENTINEL` and never returns to
    /// a materialized caller, so its header saved_bci is unused for a return).
    header_saved_bci: usize,
    /// The frame's operand stack at the deopt point. A CALLER (non-innermost)
    /// frame: its CHILD's `SenderLink.pending_stack` (the child's sender link
    /// records the caller's — i.e. THIS frame's — frozen operand stack below the
    /// inlined send's receiver+args). Innermost: the site's own recorded `stack`
    /// (plus `incoming_result`, handled in M3.3).
    pending_stack: Vec<ValueLoc>,
    /// True for exactly the innermost frame (the one the site names).
    is_innermost: bool,
}

/// Read one `ValueLoc` out of the compiled frame `fv` against `nm` (D5 M3.1).
/// The four arms are the whole location vocabulary:
/// - `FrameSlot(off)` → the live native-stack word at `fp + off` (S12 spill
///   convention),
/// - `ConstPool(i)` → the live nmethod oop-pool word `i` (GC keeps it
///   current, so this reads it NOW, not a snapshot),
/// - `ConstSmi(v)` → tag `v` as a smi,
/// - `Nil` → the well-known `nil` oop.
///
/// S13's own recorder (`resolve_frame_loc`) only ever emits `FrameSlot`/
/// `Nil`; `ConstPool`/`ConstSmi` come from the IR constant layer (wired in a
/// later step) and from S14, but all four are handled here for correctness
/// and exercised by the hand-built test blob.
fn read_value(vm: &VmState, nm: &Nmethod, fv: &FrameView, loc: ValueLoc) -> Oop {
    match loc {
        ValueLoc::FrameSlot(off) => read_frame_slot(fv.fp, off),
        ValueLoc::ConstPool(ix) => read_pool_oop(nm, ix),
        ValueLoc::ConstSmi(v) => SmallInt::new(v).oop(),
        ValueLoc::Nil => vm.universe.nil_obj,
        // S14 step 7-IV-b: an elided closure ALLOCATES on materialization —
        // handled by the M3 frame loop's own closure pass (which has `&mut vm`,
        // the root frame, and GC-safe ordering), never by this plain reader.
        // Reaching it here means a phantom leaked into a location the compiler
        // must never record one in (a receiver, a pending stack, a ctx temp).
        ValueLoc::ElidedClosure(_) => unreachable!(
            "read_value: ValueLoc::ElidedClosure outside a frame-slot/site-stack \
             position (compiler bug — phantoms are only recorded in inlined-scope \
             slots and reexecute site stacks)"
        ),
        // Float fast-path: a raw-f64 slot must be BOXED on materialization,
        // which allocates — handled at each looped call site (scope slots,
        // pending stacks, reexecute site stacks), which have `&mut vm` and
        // GC-safe ordering; this plain reader cannot.
        ValueLoc::DoubleSlot(_) => unreachable!(
            "read_value: ValueLoc::DoubleSlot outside a slot/stack/receiver position \
             (compiler bug — fp vregs are only ever temps, operand-stack \
             copies, or an inlined frame's receiver copy, never ctx locations; \
             every such position boxes at its own call site, not here)"
        ),
    }
}

/// S14 step 7-IV-b: materialize an ELIDED closure — the compiled code spliced
/// the block inline and never allocated it, but the interpreter frame being
/// rebuilt needs the real thing. Mirrors `interpreter::blocks::make_closure`,
/// sourced from the already-materialized ROOT frame (the block's home is
/// always the compilation root in v1): `home_ref` = (root fp, root serial),
/// `copied[0]` = root receiver, `copied[1]` = root Context iff the block
/// `captures_ctx` (the root's own M6 ran before any inlined frame is built, so
/// its FRAME_CONTEXT is final — a fresh Context if the root's was elided).
/// GC-safe: the block MethodOop rides in a handle across the allocation, and
/// the root frame's fields are re-read from the (rooted) process stack after.
fn materialize_closure(vm: &mut VmState, id: NmethodId, root_fp: usize, pool_ix: u32) -> Oop {
    let blk_oop = read_pool_oop(nmethod_ref(vm, id), pool_ix);
    let blk = crate::oops::wrappers::MethodOop::try_from(blk_oop)
        .expect("materialize_closure: pool entry is not a CompiledBlock");
    let captures_ctx = blk.captures_ctx();
    let ncopied = 1 + captures_ctx as usize;

    let scope = crate::memory::handles::HandleScope::enter(vm);
    let blk_h = scope.handle(vm, blk);
    let closure = crate::memory::alloc::alloc_closure(vm, ncopied);
    closure.set_method(blk_h.get(vm));

    let root = Frame { fp: root_fp };
    closure.set_copied(0, root.receiver(&vm.stack));
    if captures_ctx {
        closure.set_copied(1, root.context(&vm.stack));
    }
    let home = crate::oops::home_ref::pack_home_ref(crate::oops::home_ref::HomeRef {
        proc: 0,
        serial: root.serial(&vm.stack),
        fp: root_fp,
    });
    closure.set_home(home);
    closure.oop()
}

/// Length in bytes of the bytecode at `bci` — `decode_at`'s `next - bci`
/// (D5 M5's `bytecode_len_at`). Reuses the single decode stage so the
/// materializer and the interpreter agree on instruction widths by
/// construction.
fn bytecode_len_at(method: MethodOop, bci: usize) -> usize {
    let (_instr, next) = crate::bytecode::opcode::decode_at(method, bci);
    next - bci
}

/// The compiler's bytecode abstract-interpreter model of the operand-stack
/// height at `resume_bci` (D5 M4) — a scan from bci 0 to `resume_bci` that
/// FOLLOWS each forward branch's resolved direction rather than walking
/// raw address order, so it never counts bytes no path to `resume_bci`
/// actually executes:
///
/// - `JumpFwd` is unconditional — always taken, so the walk always
///   continues at its target, never at `next`.
/// - `BrTrueFwd`/`BrFalseFwd` fork a structured if/else: the two arms
///   occupy disjoint bci ranges (`next..target` then `target..`), so
///   `resume_bci`'s own position tells you which arm it was reached
///   through — before the target, the fall-through arm; at or past it,
///   the taken arm — and the walk continues accordingly, entirely
///   skipping the OTHER arm's bytes.
/// - `JumpBack` (a loop's own back edge) is left alone: this is a single
///   forward pass toward `resume_bci`, so a back edge met along the way
///   is never taken (taking it would loop the model itself). A
///   structured loop body's net stack effect is zero by construction —
///   the same value entering and leaving each iteration — so walking
///   through one changes nothing either way; the loop HEADER's own
///   merge (entry edge + back edge) is the one case genuinely beyond a
///   linear scan, and the caller exempts it via `skip_merge_check`.
///
/// Without the branch-following above, a resume point inside the SECOND
/// arm of an if/else would walk through the first arm's bytes too (they
/// sit earlier in the linear stream) and wrongly fold in whatever it
/// left on the stack — that leftover isn't popped until the shared merge
/// point past BOTH arms, so a scan stopping mid-second-arm still carries
/// it. Found via `cold_branch_recompile_spill_corruption.mst`/BUG D's own
/// second bug: `process:`'s `ifFalse: [ data add2: 7 ]` arm, untaken
/// (and hence trapping) until a late recompile, inflated this model to 3
/// against a correctly-materialized 2 — the `ifTrue:` arm's own
/// `add1:` send left an unpopped result the model never walked past.
///
/// Used ONLY inside the debug-build `debug_assert_eq!` cross-check, never
/// to size the real frame (the recorded `stack` is the source of truth).
/// Returns `None` when the walk can't land on `resume_bci` exactly (a
/// mid-instruction bci — itself a bug the caller's assert will surface),
/// so the check is skipped rather than panicking on the model's own
/// limitation.
#[cfg(debug_assertions)]
fn interpreter_model_height(method: MethodOop, resume_bci: usize) -> Option<i32> {
    use crate::bytecode::opcode::Instr;
    let mut bci = 0usize;
    let mut height = 0i32;
    while bci < resume_bci {
        let (instr, next) = crate::bytecode::opcode::decode_at(method, bci);
        height += crate::compiler::ir::instr_stack_delta(method, &instr);
        bci = match instr {
            Instr::JumpFwd(d) => next + d as usize,
            Instr::BrTrueFwd(d) | Instr::BrFalseFwd(d) if resume_bci >= next + d as usize => {
                next + d as usize
            }
            _ => next,
        };
    }
    if bci == resume_bci {
        Some(height)
    } else {
        None
    }
}

/// Materialize interpreter frame(s) for `frame` onto `vm.stack` (D5 M0–M7)
/// and return the [`DeoptResume`] that drives the nested run (D5 M8). Does
/// NOT run the frames — the caller (`interpreter::interpret_active`, invoked
/// by `rt_uncommon_trap`) drives the nested interpreter resume afterward.
/// Allocates (Contexts, M6); holds no bare oop across an allocating call.
///
/// The returned `DeoptResume` carries the M2 `base_sp` watermark, the
/// innermost M5 `resume_bci`, and the AMBIENT outer `vm.stack` activation +
/// `vm.regs` snapshot captured at M2 (before the M3 pushes repoint
/// `vm.stack.fp`/`sp`) — everything the nested run needs that it can no longer
/// read off `vm.stack` once the materialized frames are in place.
#[must_use = "the DeoptResume drives the nested interpret_active run; dropping it \
              materializes frames that are never run"]
pub fn deoptimize_frame(vm: &mut VmState, frame: FrameView) -> DeoptResume {
    // ── M0: resolve + decode the scope chain, innermost → outermost, then
    //    reverse so the physical push order is outermost-first (a caller
    //    frame must sit BELOW its callee on the stack). ──────────────────
    //
    // The `DeoptState` borrows the nmethod's scopes blob; decode everything
    // it needs into owned `VirtualFrame`s up front, so the `&Nmethod` borrow
    // is dropped before the mutable `vm.stack` pushes (and the allocations
    // M6 performs) begin.
    let mut virtual_frames: Vec<VirtualFrame> = Vec::new();
    // `site_kind` is consulted only by the debug-build M4 height cross-check
    // below; a release build never reads it.
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    let (site_bci, site_reexecute, site_kind, site_stack) = {
        let nm = vm
            .code_table
            .get(frame.nm)
            .expect("deoptimize_frame: FrameView.nm is not a live nmethod");
        let deopt = DeoptState::at(nm, (frame.pc - nm.code.base as usize) as u32);

        // Walk innermost → outermost. Each scope's OWN pending stack and
        // caller-resume bci come from ITS `sender` link (the send in its
        // caller): pending_stack is the sender's frozen operand stack below
        // the inlined send's receiver+args; caller_resume_bci is
        // `sender_bci + len(send)`. The innermost scope has no incoming
        // sender-of-itself here — its pending stack is the site's recorded
        // `stack` (handled in M3), and its own resume bci is M5's — so it
        // carries empty placeholders that M3 overrides.
        let chain: Vec<DecodedScope> = deopt.scopes().collect();
        // Two facts per frame come from DIFFERENT sender links (the classic
        // frame-overlap subtlety), so they must NOT be conflated:
        //
        //  - a frame's HEADER `saved_bci` (where its CALLER resumes when THIS
        //    frame returns) comes from THIS frame's OWN sender link
        //    (`chain[i].sender`): the send in its caller, `sender_bci + len(send
        //    in the CALLER's method)`. The caller's method is `chain[i+1]`. The
        //    outermost (root) scope has `sender: None` → `0` (it links to the
        //    entry sentinel and never returns to a materialized caller).
        //
        //  - a frame's operand STACK comes from its CHILD's sender link
        //    (`chain[i-1].sender.pending_stack`): a sender link records the
        //    CALLER's — i.e. THIS frame's — frozen operand stack. The innermost
        //    frame (i == 0) has no child; its operand stack is the site's own
        //    recorded `stack` (M3.3), a placeholder here.
        //
        // S13 depth-1: the single scope has `sender: None` and IS the innermost
        // AND outermost, so both are trivial and the S13 site/M5 values drive
        // everything — the chain-derived values only matter at depth > 1 (S14
        // inlining), which is exactly what this must get right.
        for (i, scope) in chain.iter().enumerate() {
            let is_innermost = i == 0;
            // Header saved_bci: from THIS frame's own sender (the send in its
            // caller). Decode the send in the CALLER's own method (`chain[i+1]`).
            let header_saved_bci = match &scope.sender {
                Some((_sender_off, sender_bci, _pending)) => {
                    let caller_scope = chain.get(i + 1).unwrap_or_else(|| {
                        panic!(
                            "deopt M0: scope {i} has a SenderLink but no caller scope at {}",
                            i + 1
                        )
                    });
                    let caller_method = pool_method(vm, frame.nm, caller_scope.method_pool_ix);
                    *sender_bci as usize + bytecode_len_at(caller_method, *sender_bci as usize)
                }
                None if is_innermost => {
                    // A `sender: None` INNERMOST scope is the S13 depth-1 case:
                    // the single frame is both innermost and outermost, links to
                    // the entry sentinel, and never returns to a materialized
                    // caller — so its header saved_bci is unused for a return.
                    // Set it to the frame's OWN M5 resume bci (the historical
                    // S13 convention: "the frame's own resume point"), which
                    // keeps a materialized depth-1 frame byte-identical to the
                    // interpreter frame it stands for. `chain[0]`'s method is
                    // this scope's.
                    let m = pool_method(vm, frame.nm, scope.method_pool_ix);
                    let site_bci = deopt.site.bci as usize;
                    if deopt.site.reexecute {
                        site_bci
                    } else {
                        site_bci + bytecode_len_at(m, site_bci)
                    }
                }
                None => 0, // depth-2 outermost caller — links to the sentinel, unused
            };
            // Operand stack: from the CHILD's sender link (which names THIS
            // frame as the caller); innermost uses the site's stack in M3.
            let pending_stack = if i == 0 {
                Vec::new()
            } else {
                match &chain[i - 1].sender {
                    Some((_sender_off, _sender_bci, pending)) => pending.clone(),
                    None => unreachable!(
                        "deopt M0: non-innermost scope {i} has a child with no SenderLink"
                    ),
                }
            };
            virtual_frames.push(VirtualFrame {
                scope: scope.clone(),
                header_saved_bci,
                pending_stack,
                is_innermost,
            });
        }
        (
            deopt.site.bci as usize,
            deopt.site.reexecute,
            deopt.site.kind,
            deopt.site.stack.clone(),
        )
    };
    // Outermost first for the physical pushes.
    virtual_frames.reverse();

    // ── M1: bump the deopt counter; trace under MACVM_TRACE=deopt. ───────
    vm.stats.deopt_count += 1;
    vm.probe_ring
        .push(crate::runtime::vm_state::ProbeEvent::Deopt {
            nm: frame.nm.0,
            bci: site_bci as u32,
            reexecute: site_reexecute,
        });
    if vm.options.trace.is_enabled("deopt") {
        let ident = vm
            .code_table
            .get(frame.nm)
            .map(|nm| {
                format!(
                    "{}>>{}",
                    crate::memory::print_oop(&vm.universe, nm.key_klass.name()),
                    nm.key_selector.as_string()
                )
            })
            .unwrap_or_else(|| "?".to_string());
        eprintln!(
            "[deopt] nm={:?} {ident} pc={:#x} fp={:#x} bci={} reexecute={} frames={} result={}",
            frame.nm,
            frame.pc,
            frame.fp,
            site_bci,
            site_reexecute,
            virtual_frames.len(),
            frame.incoming_result.is_some(),
        );
    }

    // ── M2: record the ProcessStack watermark. The nested run (M8) ends
    //    when the outermost frame's ENTRY_FRAME_SENTINEL return pops the
    //    stack back to here. Snapshot the AMBIENT outer interpreter state
    //    (activation only) NOW, before the M3 pushes repoint
    //    `vm.stack.fp`/`sp` — `interpret_active` restores it after the run so
    //    a paused outer activation (I→C→deopt) survives, the same reentrancy
    //    guard `run_method_reentrant` applies at the C→I seam. The ambient
    //    `vm.regs` is deliberately NOT copied here: a copy would sit un-rooted
    //    across M3-M6 materializer allocations and go stale (DeoptResume's
    //    own doc) — interpret_active snapshots it at its entry instead. ──────
    let base_sp = vm.stack.sp;
    let saved_activation = vm.stack.save_activation();

    // ── M3: push each virtual frame, outermost → innermost. ──────────────
    // `saved_fp` links each frame to the previous materialized frame's FP;
    // the outermost links to the entry sentinel (there is no materialized
    // caller — the compiled frame's own native caller resumes via the
    // trampoline epilogue once the nested run returns, step 7).
    let mut prev_fp: i64 = crate::oops::layout::ENTRY_FRAME_SENTINEL;
    // S14 step 7-IV-b: the OUTERMOST (root) materialized frame's fp — the home
    // of every elided closure and every inlined `is_block` scope in this chain
    // (v1 splices blocks only into their home compilation). Set on the first
    // iteration; later frames' closure/context materialization reads the root
    // frame's receiver/context/serial from it.
    let mut root_fp: Option<usize> = None;
    // S24 B5 step 5b: the materializer's ALLOCATING fixups are deferred to a
    // single phase AFTER every dead-frame `read_value` has completed. An
    // allocation here can scavenge, and the DeoptBridge walker deliberately
    // bridges PAST the trapped compiled frame — its spill slots are never
    // GC-updated — so a read_value issued after any allocation would hand
    // back a stale pre-move address (proven live: deltablue's depth-3
    // spliced-block deopt storm; B4's Frame::verify caught copied0 fresh vs
    // FP+4 stale). Every value a deferred fixup needs is rooted first:
    // process-stack slots for the frames, `deopt_hs` handles for Elided-ctx
    // temps.
    enum LateFixup {
        ElidedCtx {
            fp: usize,
            temps: Vec<crate::memory::handles::Handle<Oop>>,
        },
        SplicedBlock {
            fp: usize,
            pool_ix: u32,
        },
        Phantom {
            fp: usize,
            slot_ix: usize,
            pool_ix: u32,
        },
        SiteElided {
            stack_ix: usize,
            pool_ix: u32,
        },
    }
    let deopt_hs = crate::memory::handles::HandleScope::enter(vm);
    let mut fixups: Vec<LateFixup> = Vec::new();

    // Track the innermost frame's fp so M5/M6/M7 can address it after all
    // pushes (a later `push_frame` cannot move an earlier frame — the stack
    // only grows — so this stays valid).
    let mut innermost_fp: usize = 0;
    // The innermost frame's M5 resume bci — the bci `interpret_active` enters
    // the nested `dispatch` at. This is the innermost frame's own START point,
    // DISTINCT from its header `saved_bci` (which is where its CALLER resumes on
    // return): in a depth-1 chain the innermost has no materialized caller so
    // the two coincide, but at depth > 1 they differ (start = the inlined op's
    // bci; header = the caller's post-send resume). Captured here to hand back
    // in `DeoptResume`.
    let mut resume_bci: usize = 0;
    for vf in virtual_frames.iter() {
        let method = pool_method(vm, frame.nm, vf.scope.method_pool_ix);
        let argc = method.argc();
        let ntemps = method.ntemps();

        // A frame's HEADER `saved_bci` is where its CALLER resumes when THIS
        // frame returns — derived in M0 from THIS frame's own `SenderLink` (the
        // send in its caller). The outermost frame links to the entry sentinel,
        // so its `header_saved_bci` (0) is never read for a return. This is the
        // header value for EVERY frame — the innermost frame's OWN start bci is
        // a separate value (`resume_bci` below), NOT its header.
        let saved_bci = vf.header_saved_bci;

        // Read receiver + unified slot values BEFORE any push, so a stray
        // realloc of the stack Vec during the pushes can't invalidate a
        // FrameSlot read (they're plain values now).
        //
        // Float fast-path (spliced-temp promotion): an INLINED frame's
        // receiver may itself be a promoted raw-f64 copy (a spliced float
        // callee's `self` — e.g. `s := self scale: s` promotes `s` and its
        // splice-plumbing copies, one of which is the inline scope's
        // receiver). Boxing HERE — unlike in the slot map below — is
        // GC-safe: it is the FIRST read of this frame's iteration, so no
        // unrooted raw `Oop` is in hand yet, and the slot reads below never
        // allocate (their alloc cases defer to after the push).
        let receiver = match vf.scope.receiver {
            ValueLoc::DoubleSlot(off) => {
                let bits = read_frame_slot_raw(frame.fp, off);
                crate::memory::alloc::alloc_double(vm, f64::from_bits(bits)).oop()
            }
            loc => read_value(vm, nmethod_ref(vm, frame.nm), &frame, loc),
        };
        let nm_ref = nmethod_ref(vm, frame.nm);
        // S14 step 7-IV-b: an inlined scope's slot may hold an ELIDED CLOSURE
        // (a `do:`-style callee's block-arg temp). Its allocation must not run
        // mid-collect (the other raw `Oop`s here are unrooted until pushed), so
        // read a nil placeholder now, remember the slot, and materialize AFTER
        // the frame is pushed (everything rooted, `set_temp` overwrites).
        let mut phantom_slots: Vec<(usize, u32)> = Vec::new();
        // Float fast-path (B5): a raw-f64 temp slot. Read the BITS now (bits
        // are GC-immune, unlike the other raw `Oop`s here), box AFTER the
        // frame is pushed and everything is rooted.
        let mut double_slots: Vec<(usize, u64)> = Vec::new();
        let slot_vals: Vec<Oop> = vf
            .scope
            .slots
            .iter()
            .enumerate()
            .map(|(i, &loc)| match loc {
                ValueLoc::ElidedClosure(pool_ix) => {
                    phantom_slots.push((i, pool_ix));
                    vm.universe.nil_obj
                }
                ValueLoc::DoubleSlot(off) => {
                    double_slots.push((i, read_frame_slot_raw(frame.fp, off)));
                    vm.universe.nil_obj
                }
                _ => read_value(vm, nm_ref, &frame, loc),
            })
            .collect();
        debug_assert_eq!(
            slot_vals.len(),
            argc + ntemps,
            "materialize: scope has {} unified slots but method wants argc {argc} + ntemps \
             {ntemps}",
            slot_vals.len()
        );

        // Receiver + args, exactly where the interpreter's caller leaves
        // them (unified slots 0..argc are the callee's negative-offset arg
        // area). `push_frame` reads its receiver copy from `stack[fp-argc-1]`,
        // so push the receiver, then the args, then let `push_frame` lay the
        // header — reusing the interpreter's own frame builder verbatim (the
        // surest way to be byte-identical for M7).
        vm.stack.push(receiver);
        for &v in &slot_vals[..argc] {
            vm.stack.push(v);
        }
        let fp_new = vm.stack.sp;
        crate::interpreter::push_frame(vm, method, argc, prev_fp, saved_bci);
        if root_fp.is_none() {
            root_fp = Some(fp_new);
        }

        // Overwrite the nil temps `push_frame` laid down with the scope's
        // real temp values (unified slots argc..). `Frame::set_temp` uses the
        // same unified indexing as the interpreter.
        let f = Frame { fp: fp_new };
        for (t, &v) in slot_vals[argc..].iter().enumerate() {
            f.set_temp(&mut vm.stack, argc + t, v);
        }
        // Float fast-path: box the raw-f64 temps over their nil placeholders.
        // GC-safe HERE (not in the read loop above): the frame is pushed, so
        // every already-written value is rooted on the process stack, and the
        // bits themselves cannot be invalidated by the allocation's scavenge.
        // The process stack never moves, so `fp_new` stays valid across it.
        for &(slot_ix, bits) in &double_slots {
            let d = crate::memory::alloc::alloc_double(vm, f64::from_bits(bits));
            let f = Frame { fp: fp_new };
            f.set_temp(&mut vm.stack, slot_ix, d.oop());
        }

        // A NON-innermost frame's frozen operand stack (its child's
        // `SenderLink.pending_stack`) goes ABOVE its own header+temps — the
        // frame's live operand-stack area, exactly where the interpreter had
        // it when the inlined send fired; the CHILD frame's receiver lands on
        // top of it next iteration. (This used to be pushed BEFORE the
        // frame's own receiver/header — i.e. BELOW the frame, in the caller's
        // territory — which resumed the frame with its frozen entries missing
        // and shifted: `lo + (self next ...)` answered `nil + ...` after the
        // inlined callee's in-body trap. Unnoticed until S14 step 5 made
        // non-empty pending stacks common: every earlier depth-2 test froze
        // an EMPTY stack, for which both orders are identical. Found by the
        // step-9 soak gate.)
        if !vf.is_innermost {
            for &loc in &vf.pending_stack {
                // Float fast-path: a frozen operand-stack entry may be a raw
                // f64 (an unboxed float copy live across an inlined send).
                // Alloc-then-push is GC-safe here: every previously pushed
                // value is already rooted, and the bits are GC-immune.
                let v = match loc {
                    ValueLoc::DoubleSlot(off) => {
                        let bits = read_frame_slot_raw(frame.fp, off);
                        crate::memory::alloc::alloc_double(vm, f64::from_bits(bits)).oop()
                    }
                    _ => read_value(vm, nmethod_ref(vm, frame.nm), &frame, loc),
                };
                vm.stack.push(v);
            }
        }

        // ── M6: context. ────────────────────────────────────────────────
        // S14 step 7-II: an `is_block` frame is a ctx-LESS block activation
        // whose FP+3 ALIASES its HOME method's Context (SPEC §5.4 — a ctx-less
        // block's frame aliases the enclosing Context directly). Its home is
        // the compilation ROOT (v1 splices blocks only into their home
        // compilation), which was built and M6'd first — read the ROOT frame's
        // Context and share it. (7-IV-b: at depth 3 — a block spliced inside an
        // inlined `do:` callee — the block's SENDER is the ctx-less callee, so
        // `prev_fp` would alias the WRONG (nil) context; the home is the root.)
        // Correct for both a nil-Context home (7-II) and an elided-Context home
        // (7-II-b). A method frame uses its own recorded `CtxLoc` (M6 proper).
        if vf.scope.is_block && prev_fp != crate::oops::layout::ENTRY_FRAME_SENTINEL {
            // S24 B4 (deferred to the fixup phase, B5 step 5b): FP+3 must
            // alias the ROOT frame's Context and the receiver-ARG slot must
            // hold a synthesized home-ref closure so an interpreted nlr_tos
            // works unmodified. BOTH wait: the root's Context may itself be
            // an Elided-ctx fixup (allocated later), and the synthesis
            // allocates. FP+4 keeps the home self `push_frame` copied —
            // which equals closure.copied(0) by construction (the fixup
            // reads the rooted root frame's receiver).
            root_fp.expect("a SPLICED is_block frame is never outermost");
            fixups.push(LateFixup::SplicedBlock {
                fp: fp_new,
                pool_ix: vf.scope.method_pool_ix,
            });
        } else if vf.scope.is_block {
            // S24 A1 (design §2.5): the ROOT-block arm — a STANDALONE-
            // compiled block deopted. Mirror `activate_block_interp`
            // step-for-step: the receiver-ARG slot already holds the
            // CLOSURE (the root scope's receiver ValueLoc records the
            // closure vreg), so derive FP+4 = copied[0] (the home
            // receiver) and alias FP+3 = copied[1] iff captures_ctx else
            // nil — IDENTITY, never a fresh Context (Risk 2: one Context
            // object shared with the home; a copy would split the world).
            // No allocation on this path.
            let f = Frame { fp: fp_new };
            let closure_oop = vm.stack.get(f.receiver_slot(&vm.stack));
            let closure = crate::oops::wrappers::ClosureOop::try_from(closure_oop).expect(
                "root-block scope's receiver ValueLoc must hold the closure \
                 (driver records block_closure_vreg there)",
            );
            let blk = closure.method();
            vm.stack.set(
                fp_new + crate::oops::layout::FRAME_RECEIVER,
                closure.copied(0),
            );
            let ctx = if blk.captures_ctx() {
                closure.copied(1)
            } else {
                vm.universe.nil_obj
            };
            f.set_context(&mut vm.stack, ctx);
        } else {
            match &vf.scope.ctx {
                // Allocates — defer; temps read NOW (pre-allocation) into
                // GC-rooted handles the fixup phase drains.
                CtxLoc::Elided { temps } => {
                    let handles: Vec<crate::memory::handles::Handle<Oop>> = temps
                        .iter()
                        .map(|&loc| {
                            let v = read_value(vm, nmethod_ref(vm, frame.nm), &frame, loc);
                            deopt_hs.handle(vm, v)
                        })
                        .collect();
                    fixups.push(LateFixup::ElidedCtx {
                        fp: fp_new,
                        temps: handles,
                    });
                }
                other => materialize_context(vm, frame.nm, &frame, other, fp_new),
            }
        }

        // S14 step 7-IV-b: this frame's ELIDED-CLOSURE slots — allocations,
        // deferred; `set_temp` overwrites the nil placeholders in the fixup
        // phase.
        for &(slot_ix, pool_ix) in &phantom_slots {
            fixups.push(LateFixup::Phantom {
                fp: fp_new,
                slot_ix,
                pool_ix,
            });
        }

        if vf.is_innermost {
            innermost_fp = fp_new;
            // ── M5: the innermost frame's OWN start bci — where
            //    `interpret_active` picks up. `site_bci` if re-executing the op,
            //    else past the completed send. DISTINCT from this frame's header
            //    saved_bci (the caller's resume). At depth 1 they coincide (no
            //    materialized caller); at depth > 1 they differ.
            resume_bci = if site_reexecute {
                site_bci
            } else {
                site_bci + bytecode_len_at(method, site_bci)
            };
            // ── M3.3 innermost operand stack: the site's recorded values,
            //    plus `incoming_result` iff the site is a call-return. ─────
            // S14 step 7-IV-b: a reexecute stack may carry an ELIDED CLOSURE
            // (the guard-cold path of a send that passed a spliced block as an
            // argument re-executes that send, which needs the real closure).
            // Alloc-then-push is GC-safe here: every previously pushed value is
            // already rooted on the process stack.
            for &loc in &site_stack {
                let v = match loc {
                    // Float fast-path: a reexecute stack entry that lived as a
                    // raw f64 (an unboxed float copy) — box it for the
                    // interpreter. Alloc-then-push is GC-safe here (previously
                    // pushed values rooted; the bits are GC-immune).
                    ValueLoc::DoubleSlot(off) => {
                        let bits = read_frame_slot_raw(frame.fp, off);
                        crate::memory::alloc::alloc_double(vm, f64::from_bits(bits)).oop()
                    }
                    ValueLoc::ElidedClosure(pool_ix) => {
                        // Allocates — push a nil placeholder at this exact
                        // slot and let the fixup phase overwrite it (B5 5b).
                        root_fp.expect("root frame exists by now");
                        fixups.push(LateFixup::SiteElided {
                            stack_ix: vm.stack.sp,
                            pool_ix,
                        });
                        vm.universe.nil_obj
                    }
                    _ => read_value(vm, nmethod_ref(vm, frame.nm), &frame, loc),
                };
                // Debugger (DBG3 companion): `MACVM_DBG_REEXEC=1` prints each
                // value the materializer pushes for a reexecute site's
                // recorded operand stack — the runtime's half of the story
                // whose compile-time half is `MACVM_DBG_IR`'s scope/listing
                // dump. This trace is what proved task #94's stale frame
                // slots (two entries reading the SAME address) and, earlier,
                // BUG C's inverted dispatch. Debug builds only, stderr.
                #[cfg(debug_assertions)]
                if std::env::var("MACVM_DBG_REEXEC").is_ok() {
                    eprintln!(
                        "REEXEC nm={:?} bci={site_bci} loc={loc:?} -> {:#x} (recv {:#x})",
                        frame.nm,
                        v.raw(),
                        receiver.raw(),
                    );
                }
                vm.stack.push(v);
            }
            if site_reexecute {
                debug_assert!(
                    frame.incoming_result.is_none(),
                    "deoptimize_frame: reexecute site must carry no incoming_result \
                     (the recorded stack already holds the op's inputs)"
                );
            } else {
                let result = frame.incoming_result.expect(
                    "deoptimize_frame: a call-return (reexecute == false) site must carry \
                     the completed call's incoming_result",
                );
                vm.stack.push(result);
            }

            // ── M4: operand-stack height cross-check (debug builds). ──────
            // Skipped for the innermost frame of a `LoopPoll` deopt: its resume
            // bci is the loop HEADER — a genuine CFG merge (entry edge + back
            // edge) — where `interpreter_model_height`'s straight-line scan is
            // documented to disagree with the CFG fixpoint (it double-counts
            // both arms of any conditional that feeds the header, e.g.
            // `x := c ifTrue:[..] ifFalse:[..]. [i<x] whileTrue:[..]`). The
            // recorded `stack` (CFG-derived) is the source of truth and the
            // materialization is correct — this check just can't model a merge.
            // Every OTHER reexecute site (traps) resumes at a mid-block op bci
            // where the linear scan is exact, so they keep the check.
            #[cfg(debug_assertions)]
            {
                // The innermost frame's OWN start bci (M5), NOT its header
                // saved_bci (the caller's resume) — the height model must be
                // evaluated at where this frame actually resumes interpreting.
                let resume_bci = resume_bci;
                let skip_merge_check = vf.is_innermost
                    && matches!(site_kind, crate::compiler::scopes::SafepointKind::LoopPoll);
                if let (false, Some(model)) = (
                    skip_merge_check,
                    interpreter_model_height(method, resume_bci),
                ) {
                    let materialized =
                        (vm.stack.sp - (fp_new + frame_header_and_temps(method))) as i32;
                    debug_assert_eq!(
                        materialized, model,
                        "deopt M4: materialized operand-stack height {materialized} != \
                         interpreter model height {model} at resume bci {resume_bci} \
                         (the classic deopt bug — reexecute={site_reexecute})"
                    );
                }
            }
        }

        prev_fp = fp_new as i64;
    }

    // ── Fixup phase (B5 step 5b): every dead-frame read is done; every
    //    source a fixup needs is rooted (process-stack slots, `deopt_hs`
    //    handles). Allocations here may scavenge freely — GC updates the
    //    rooted sources and the already-pushed frames alike. Frame order
    //    (fixups were recorded outermost-first) guarantees the ROOT's own
    //    Elided-ctx Context exists before a spliced block's FP+3 alias
    //    reads it. ─────────────────────────────────────────────────────
    for fx in fixups {
        match fx {
            LateFixup::ElidedCtx { fp, temps } => {
                let context = crate::memory::alloc::alloc_context(vm, temps.len());
                for (i, h) in temps.iter().enumerate() {
                    context.set_slot(i, h.get(vm));
                }
                Frame { fp }.set_context(&mut vm.stack, context.oop());
            }
            LateFixup::SplicedBlock { fp, pool_ix } => {
                let root = root_fp.expect("a SPLICED is_block frame is never outermost");
                let home_ctx = Frame { fp: root }.context(&vm.stack);
                Frame { fp }.set_context(&mut vm.stack, home_ctx);
                let closure = materialize_closure(vm, frame.nm, root, pool_ix);
                #[cfg(debug_assertions)]
                {
                    let m = crate::oops::wrappers::ClosureOop::try_from(closure)
                        .expect("materialize_closure returns a closure")
                        .method();
                    debug_assert!(
                        m.is_block(),
                        "spliced is_block scope's method_pool_ix must name a CompiledBlock"
                    );
                }
                let f = Frame { fp };
                let slot = f.receiver_slot(&vm.stack);
                vm.stack.set(slot, closure);
            }
            LateFixup::Phantom {
                fp,
                slot_ix,
                pool_ix,
            } => {
                let root = root_fp.expect("root frame exists by now");
                let closure = materialize_closure(vm, frame.nm, root, pool_ix);
                Frame { fp }.set_temp(&mut vm.stack, slot_ix, closure);
            }
            LateFixup::SiteElided { stack_ix, pool_ix } => {
                let root = root_fp.expect("root frame exists by now");
                let closure = materialize_closure(vm, frame.nm, root, pool_ix);
                vm.stack.set(stack_ix, closure);
            }
        }
    }

    // ── M7: every materialized frame must be indistinguishable from an
    //    interpreter-built one. Walk the fp chain from innermost back down
    //    (saved_fp links to the previous materialized frame) and verify. ──
    let mut fp = innermost_fp;
    loop {
        let f = Frame { fp };
        f.verify(&vm.stack);
        let saved = f.saved_fp(&vm.stack);
        if saved == crate::oops::layout::ENTRY_FRAME_SENTINEL {
            break;
        }
        fp = saved as usize;
    }

    // `vm.stack.fp` is now the innermost materialized frame (the last
    // `push_frame` activated it); the nested run picks up from there.
    DeoptResume {
        base_sp,
        resume_bci,
        saved_activation,
    }
}

/// Fixed header (`FRAME_TEMPS_BASE`) + `ntemps` — the count of slots between
/// a frame's FP and the base of its operand stack. Used by M4 to isolate the
/// operand-stack height from the header+temps below it.
#[cfg(debug_assertions)]
fn frame_header_and_temps(method: MethodOop) -> usize {
    crate::oops::layout::FRAME_TEMPS_BASE + method.ntemps()
}

/// Borrow `frame.nm`'s `&Nmethod` (a live-nmethod lookup that panics on a
/// dead id — a materialize against a flushed nmethod is a VM bug). Factored
/// out because the borrow must be re-taken between mutable `vm.stack` pushes
/// (an `&Nmethod` and `&mut vm.stack` cannot coexist).
fn nmethod_ref(vm: &VmState, id: NmethodId) -> &Nmethod {
    vm.code_table
        .get(id)
        .expect("deoptimize_frame: nmethod id is not live")
}

/// Resolve a scope's `method_pool_ix` to its `MethodOop` (D5 M3: the OLD
/// compile-time method — a mid-activation redefinition completes under old
/// code). Reads the LIVE oop-pool word, so a moving GC that relocated the
/// method oop is transparently accounted for (`oops_do` keeps the pool word
/// current).
///
/// `method_pool_ix` is a REAL interned index: `ir::convert` interns a
/// deopt-having method's own compiled `MethodOop` into its pool (as an `Oop`
/// reloc) and `driver::build_deopt_metadata` records that index in every
/// scope. Because the metadata holds the compile-time oop directly, a
/// mid-activation redefinition (a later step's `NotEntrant`) still deopts to
/// the OLD method, exactly as D5 requires — the reason the method lives in
/// the GC-visited pool rather than being re-derived from `key_klass`/
/// `key_selector` (whose current dict lookup would see the NEW method).
fn pool_method(vm: &VmState, id: NmethodId, method_pool_ix: u32) -> MethodOop {
    let nm = nmethod_ref(vm, id);
    let oop = read_pool_oop(nm, method_pool_ix);
    MethodOop::try_from(oop).expect(
        "deoptimize_frame: scope method_pool_ix does not resolve to a CompiledMethod \
         (ir::convert interns a deopt-having method's own oop; a mismatch is a compiler bug)",
    )
}

/// D5 M6: materialize the frame's Context slot.
/// - `Materialized(loc)` → store the read Context oop (compiled code already
///   allocated it at entry; deopt reuses it as-is).
/// - `Elided { temps }` → read every temp value into a `HandleScope` FIRST
///   (the allocation below may GC, and the compiled frame is still walkable
///   per D4, but reading them first and rooting them keeps the fill trivially
///   correct), then allocate a fresh Context, fill it, store it. S13 never
///   emits this (S14 Context-elision does); wired now because it's cheap.
/// - `None` → leave `nil` (`push_frame` already stored nil at FRAME_CONTEXT).
fn materialize_context(
    vm: &mut VmState,
    id: NmethodId,
    frame: &FrameView,
    ctx: &CtxLoc,
    fp: usize,
) {
    match ctx {
        CtxLoc::None => {
            // push_frame already left nil at FRAME_CONTEXT.
        }
        CtxLoc::Materialized(loc) => {
            let c = read_value(vm, nmethod_ref(vm, id), frame, *loc);
            // T6 (S24 L2 phase B): with the 9de470b liveness keying,
            // `Materialized` is only ever recorded where the ctx vreg is
            // live — the slot must hold the real (possibly OSR-adopted)
            // Context, never nil/garbage. Identity is the whole contract.
            debug_assert!(
                crate::oops::wrappers::ContextOop::try_from(c).is_some(),
                "CtxLoc::Materialized resolved to a non-Context (T6)"
            );
            Frame { fp }.set_context(&mut vm.stack, c);
        }
        CtxLoc::Elided { .. } => unreachable!(
            "CtxLoc::Elided allocates and is handled by deoptimize_frame's \
             deferred fixup phase (B5 5b) — never materialize it here, where \
             a later dead-frame read_value would see stale pre-move addresses"
        ),
    }
}

/// Test-only helpers shared with `codecache::deopt_trap`'s own end-to-end
/// `rt_uncommon_trap` test (which lives in the unsafe island where the raw
/// `find_by_pc` deref is allowed). Kept `pub(crate)` and outside the private
/// `tests` module below so both call sites reuse one nmethod-building path
/// rather than duplicating the pool-packing / scope-recording scaffolding.
/// All-safe (the `#![deny(unsafe_code)]` module invariant holds).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::codecache::nmethod::{NmState, Nmethod, NmethodId};
    use crate::compiler::scopes::{PcDesc, SafepointState, ScopeDescData, ScopeDescRecorder};

    /// Publish a code blob whose bytes ARE the oop pool ([method, extra] at
    /// `literal_off = 0`), so `read_pool_oop` reads live MAP_JIT words.
    pub(crate) fn publish_pool(
        vm: &mut VmState,
        method: MethodOop,
        extra: Oop,
    ) -> crate::codecache::CodeHandle {
        use crate::compiler::assembler::CodeBlob;
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&method.oop().raw().to_le_bytes());
        bytes.extend_from_slice(&extra.raw().to_le_bytes());
        let blob = CodeBlob {
            code: bytes,
            literal_off: 0,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        let handle = vm
            .code_cache
            .alloc(blob.code.len())
            .expect("test code cache room");
        vm.code_cache.publish(handle, &blob);
        handle
    }

    /// Build (not yet install) a full inert `Nmethod` around a published pool
    /// handle. Only its pool + scopes are ever read (deopt); the rest is
    /// placeholder shape, with a distinct real selector per `code.base` so
    /// `CodeTable::by_key` never collides across installs.
    pub(crate) fn build_nmethod_with_pool(
        vm: &mut VmState,
        method: MethodOop,
        extra: Oop,
    ) -> Nmethod {
        let code = publish_pool(vm, method, extra);
        let sel = format!("s{:x}", code.base as usize);
        let sel_sym = vm.universe.intern(sel.as_bytes());
        Nmethod {
            osr_cold_sends: 0,
            id: NmethodId(0),
            key_klass: vm.universe.object_klass,
            key_selector: sel_sym,
            code,
            entry_off: 0,
            verified_entry_off: 0,
            state: NmState::Alive,
            level: 1,
            version: 0,
            trap_count: 0,
            profile_hash: 0,
            literal_off: 0,
            relocs: Vec::new(),
            frame_slots: 0,
            slot_is_oop: Vec::new(),
            pcdescs: Vec::new(),
            oopmaps: Vec::new(),
            ic_sites: Vec::new(),
            poll_bci: None,
            prim_call_argc_plus_recv: None,
            block_method: None,
            deopt_scopes: Vec::new(),
            deopt_pcdescs: Vec::new(),
            inline_deps: Vec::new(),
            self_devirt: false,
            osr_map: None,
        }
    }

    pub(crate) fn install_nmethod(
        vm: &mut VmState,
        mut nm: Nmethod,
        blob: Vec<u8>,
        pcdescs: Vec<PcDesc>,
    ) -> NmethodId {
        nm.deopt_scopes = blob;
        nm.deopt_pcdescs = pcdescs;
        vm.code_table.install(nm)
    }

    /// One-shot: build + install an nmethod for `method` (pool word 1 =
    /// `extra`) carrying a single depth-1 scope described by `scope_data` and
    /// one recorded `site` at code offset `site_off`. Returns the installed
    /// `NmethodId` and the absolute trap pc (`code.base + site_off`) — exactly
    /// the two inputs `rt_uncommon_trap` derives a `FrameView` from.
    pub(crate) fn install_deopt_nmethod(
        vm: &mut VmState,
        method: MethodOop,
        extra: Oop,
        scope_data: ScopeDescData,
        site_off: u32,
        site: SafepointState,
    ) -> (NmethodId, usize) {
        let nm = build_nmethod_with_pool(vm, method, extra);
        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(scope_data);
        rec.record_site(site_off, SafepointState { scope, ..site });
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(vm, nm, blob, pcdescs);
        let pc = vm.code_table.get(nm_id).unwrap().code.base as usize + site_off as usize;
        (nm_id, pc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeBuilder;
    use crate::compiler::scopes::{
        CtxLoc, SafepointKind, SafepointState, ScopeDescData, ScopeDescRecorder, ValueLoc,
    };
    use crate::interpreter::stack::Frame;
    use crate::oops::layout::{
        ENTRY_FRAME_SENTINEL, FRAME_CONTEXT, FRAME_METHOD, FRAME_SAVED_BCI, FRAME_SAVED_FP,
        FRAME_TEMPS_BASE,
    };
    use crate::oops::smi::SmallInt;
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

    /// A method with argc args + ntemps temps whose body is `return self`
    /// (bci 0). We deopt at a re-execute site keyed at bci 0, so the resume
    /// bci is 0 and the model operand-stack height there is 0.
    fn trivial_method(vm: &mut VmState, argc: usize, ntemps: usize) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(b"deoptee");
        b.finish(vm, sel, argc, ntemps)
    }

    /// Hand-build the physical compiled frame as a stack-local `[u64]` whose
    /// address stands in for the trapped frame's FP, plus a hand-built
    /// nmethod carrying a recorder-produced scope blob whose `FrameSlot`
    /// offsets index into that array, and a `ConstPool` word interned into
    /// the nmethod's own pool. Then materialize and assert the resulting
    /// interpreter frame matches an interpreter-built one (Frame::verify +
    /// field-by-field).
    ///
    /// This is the whole step-6 contract in one test: a compiled frame in →
    /// an interpreter frame out, indistinguishable from a native push.
    #[test]
    fn materialize_reexecute_frame_matches_interpreter() {
        let mut vm = test_vm();

        // The deoptee: argc=1, ntemps=2. Unified slots: [arg0, temp0, temp1].
        // Its bytecode pushes THREE values then returns, so the compiler's
        // abstract-interpreter model height at bci 3 (our re-execute resume
        // point) is exactly 3 — matching the 3-entry recorded operand stack
        // below. (An arbitrary bci-0 site with a non-empty stack is physically
        // impossible and M4 rightly rejects it — a real re-execute site's
        // recorded height always equals the model height there.)
        let method = {
            let mut b = BytecodeBuilder::new();
            b.push_nil(); // bci 0 -> h1
            b.push_nil(); // bci 1 -> h2
            b.push_nil(); // bci 2 -> h3
            b.ret_tos(); // bci 3 (never reached by deopt; we resume AT bci 3)
            let sel = vm.universe.intern(b"deoptee3");
            b.finish(&mut vm, sel, 1, 2)
        };

        // Build the nmethod. Its oop pool must carry (word 0) the method oop
        // — a REAL `method_pool_ix` (see pool_method's S13-gap note; the test
        // sets a real index rather than the production 0 placeholder) — and
        // (word 1) a constant oop we reference via ConstPool(1). We hand-pack
        // a code blob whose literal area holds those two words.
        let receiver_val = SmallInt::new(0x11).oop();
        let arg0_val = SmallInt::new(0x22).oop();
        let temp0_val = SmallInt::new(0x33).oop();
        // A ConstPool-sourced constant (word 1 of the pool). Use a distinct
        // smi so we can tell it apart.
        let pool_const_val = SmallInt::new(0x44).oop();

        // A ConstSmi and Nil in the operand stack, to exercise those arms.
        let const_smi_val: i64 = 0x55;

        let nm = build_nmethod_with_pool(&mut vm, method, pool_const_val);

        // The physical frame: a stack-local array the FrameSlots address.
        // Layout (byte offsets from `fp`, all negative = below-FP spill
        // slots, S12 convention): we place four oops and point `fp` past
        // them so offsets -8/-16/-24 land on real words.
        //   slots[0] = receiver   (FrameSlot -24)
        //   slots[1] = arg0       (FrameSlot -16)
        //   slots[2] = temp0      (FrameSlot -8)
        //   slots[3] = (fp anchor; not read)
        let phys: [u64; 4] = [
            receiver_val.raw(),
            arg0_val.raw(),
            temp0_val.raw(),
            0, // fp points here
        ];
        let fp = (&phys[3]) as *const u64 as usize;

        // Recorder: one depth-1 scope. Unified slots argc..: temp0 from a
        // FrameSlot, temp1 from Nil. Receiver from a FrameSlot. Context None.
        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0, // real: word 0 of the pool is the method oop
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-24),
            slots: vec![
                ValueLoc::FrameSlot(-16), // arg0
                ValueLoc::FrameSlot(-8),  // temp0
                ValueLoc::Nil,            // temp1 = nil
            ],
            ctx: CtxLoc::None,
        });
        // A re-execute uncommon-trap site at bci 0. Operand stack: a
        // ConstPool oop, a ConstSmi, and a Nil — three values, height 3.
        rec.record_site(
            0x10,
            SafepointState {
                scope,
                bci: 3, // resume AT bci 3; model height there is 3
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![
                    ValueLoc::ConstPool(1),
                    ValueLoc::ConstSmi(const_smi_val),
                    ValueLoc::Nil,
                ],
            },
        );
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(&mut vm, nm, blob, pcdescs);

        // The pc that resolves the site: code_base + 0x10.
        let nm_ref = vm.code_table.get(nm_id).unwrap();
        let pc = nm_ref.code.base as usize + 0x10;

        let sp_before = vm.stack.sp;
        let deopt_before = vm.stats.deopt_count;

        let resume = deoptimize_frame(
            &mut vm,
            FrameView {
                fp,
                pc,
                nm: nm_id,
                incoming_result: None, // reexecute site
            },
        );

        // The DeoptResume the nested run consumes: base_sp is the pre-push
        // watermark, resume_bci is the innermost M5 bci (reexecute → the
        // recorded bci 3), and the ambient outer state is the pre-deopt
        // snapshot (no frame active in this bare test).
        assert_eq!(resume.base_sp, sp_before, "base_sp is the M2 watermark");
        assert_eq!(
            resume.resume_bci, 3,
            "reexecute resume bci is the recorded bci"
        );
        assert!(
            !resume.saved_activation.was_active(),
            "the bare test had no ambient frame active before deopt"
        );

        // M1: counter bumped exactly once.
        assert_eq!(vm.stats.deopt_count, deopt_before + 1);

        // One frame was pushed. Its fp is right after the pushed
        // receiver+args (receiver + 1 arg = 2 slots) from sp_before.
        let fp_new = sp_before + 2;
        let f = Frame { fp: fp_new };

        // M7 already ran inside deoptimize_frame; re-run for the test's own
        // assurance and then check fields.
        f.verify(&vm.stack);

        // Header.
        assert_eq!(
            f.method(&vm.stack).oop(),
            method.oop(),
            "FRAME_METHOD is the deoptee method"
        );
        assert_eq!(
            f.saved_fp(&vm.stack),
            ENTRY_FRAME_SENTINEL,
            "outermost frame links to the entry sentinel"
        );
        assert_eq!(
            f.saved_bci(&vm.stack),
            3,
            "reexecute site resumes AT bci 3 (the recorded bci, not past it)"
        );
        assert_eq!(
            f.context(&vm.stack),
            vm.universe.nil_obj,
            "CtxLoc::None leaves nil context"
        );

        // Receiver (fast copy + arg-area copy agree).
        assert_eq!(f.receiver(&vm.stack), receiver_val);
        // Unified slots: arg0 (index 0), temp0 (index 1), temp1=nil (index 2).
        assert_eq!(f.temp(&vm.stack, 0), arg0_val, "arg0");
        assert_eq!(f.temp(&vm.stack, 1), temp0_val, "temp0");
        assert_eq!(f.temp(&vm.stack, 2), vm.universe.nil_obj, "temp1 = nil");

        // Operand stack: ConstPool(1) -> pool_const_val, ConstSmi -> tagged
        // smi, Nil -> nil. sp points just past them.
        let opnd_base = fp_new + FRAME_TEMPS_BASE + method.ntemps();
        assert_eq!(vm.stack.sp, opnd_base + 3, "three operand values pushed");
        assert_eq!(
            vm.stack.get(opnd_base),
            pool_const_val,
            "ConstPool(1) resolved to the live pool word"
        );
        assert_eq!(
            vm.stack.get(opnd_base + 1),
            SmallInt::new(const_smi_val).oop(),
            "ConstSmi tagged as a smi"
        );
        assert_eq!(vm.stack.get(opnd_base + 2), vm.universe.nil_obj, "Nil");

        // Cross-check the header against a genuine interpreter-built frame for
        // the SAME method + receiver + args: `push_frame`'s own output must be
        // slot-for-slot identical (except the serial, which is monotonic and
        // deliberately never equal — that's what makes it a dead-home key).
        // Built on `vm` itself AFTER the materialized frame (the stack only
        // grows; the earlier frame stays put), since the method lives in
        // `vm`'s heap.
        let ref_fp = vm.stack.sp + 2; // after we push receiver + 1 arg
        vm.stack.push(receiver_val);
        vm.stack.push(arg0_val);
        // Same linkage as the materialized frame (entry sentinel + resume
        // bci 3) so every header slot is expected to match slot-for-slot.
        crate::interpreter::push_frame(&mut vm, method, 1, ENTRY_FRAME_SENTINEL, 3);
        let rf = Frame { fp: ref_fp };
        assert_eq!(
            vm.stack.get(ref_fp + FRAME_METHOD),
            vm.stack.get(fp_new + FRAME_METHOD),
            "method slot identical to an interpreter push"
        );
        assert_eq!(
            vm.stack.get(ref_fp + FRAME_SAVED_FP),
            vm.stack.get(fp_new + FRAME_SAVED_FP),
            "saved_fp slot identical"
        );
        assert_eq!(
            vm.stack.get(ref_fp + FRAME_SAVED_BCI),
            vm.stack.get(fp_new + FRAME_SAVED_BCI),
            "saved_bci slot identical"
        );
        assert_eq!(
            vm.stack.get(ref_fp + FRAME_CONTEXT),
            vm.stack.get(fp_new + FRAME_CONTEXT),
            "context slot identical (both nil)"
        );
        assert_eq!(
            rf.receiver(&vm.stack),
            f.receiver(&vm.stack),
            "receiver copy"
        );
    }

    /// A call-return site (reexecute == false): the recorded stack EXCLUDES
    /// the send's popped args and the materializer pushes `incoming_result`.
    /// Resume bci is PAST the send. Exercises M3.3's non-reexecute arm and
    /// the M5 `bci + len(send)` computation.
    #[test]
    fn materialize_call_return_frame_pushes_result() {
        let mut vm = test_vm();

        // A method: `push_self; send #foo; return_tos` — so there's a real
        // send at a known bci whose length M5 advances past. argc=0.
        let mut b = BytecodeBuilder::new();
        b.push_self(); // bci 0 (1 byte)
        let foo = vm.universe.intern(b"foo");
        b.send(&mut vm, foo, 0); // bci 1 (send: 2 bytes) -> resume at bci 3
        b.ret_tos(); // bci 3
        let sel = vm.universe.intern(b"caller");
        let method = b.finish(&mut vm, sel, 0, 0);

        let receiver_val = SmallInt::new(0x77).oop();
        let result_val = SmallInt::new(0x88).oop();

        let nil = vm.universe.nil_obj;
        let nm = build_nmethod_with_pool(&mut vm, method, nil);

        // Physical frame: just the receiver at FrameSlot -8.
        let phys: [u64; 2] = [receiver_val.raw(), 0];
        let fp = (&phys[1]) as *const u64 as usize;

        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0,
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-8),
            slots: vec![], // argc 0, ntemps 0
            ctx: CtxLoc::None,
        });
        // Call site keyed on its RETURN address; reexecute = false; recorded
        // stack is empty (the send's receiver was popped, result comes from
        // incoming_result). bci names the SEND (bci 1).
        rec.record_site(
            0x20,
            SafepointState {
                scope,
                bci: 1,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![],
            },
        );
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(&mut vm, nm, blob, pcdescs);
        let pc = vm.code_table.get(nm_id).unwrap().code.base as usize + 0x20;

        let sp_before = vm.stack.sp;
        let resume = deoptimize_frame(
            &mut vm,
            FrameView {
                fp,
                pc,
                nm: nm_id,
                incoming_result: Some(result_val),
            },
        );
        // Call-return resume bci is PAST the 2-byte send (bci 1 → bci 3).
        assert_eq!(
            resume.resume_bci, 3,
            "call-return resume bci is past the send"
        );
        assert_eq!(resume.base_sp, sp_before, "base_sp is the M2 watermark");

        let fp_new = sp_before + 1; // receiver only (argc 0)
        let f = Frame { fp: fp_new };
        f.verify(&vm.stack);
        assert_eq!(
            f.saved_bci(&vm.stack),
            3,
            "call-return resumes PAST the 2-byte send at bci 1 -> bci 3"
        );
        // Operand stack: just the incoming result.
        let opnd_base = fp_new + FRAME_TEMPS_BASE; // ntemps 0
        assert_eq!(vm.stack.sp, opnd_base + 1, "one value: the call result");
        assert_eq!(
            vm.stack.get(opnd_base),
            result_val,
            "incoming_result pushed on the operand stack"
        );
    }

    /// MACVM_TRACE=deopt exercises the trace arm without changing behavior.
    #[test]
    fn deopt_count_increments() {
        let mut vm = test_vm();
        let method = trivial_method(&mut vm, 0, 0);
        let nil = vm.universe.nil_obj;
        let nm = build_nmethod_with_pool(&mut vm, method, nil);
        let phys: [u64; 1] = [0];
        let fp = (&phys[0]) as *const u64 as usize;

        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0,
            is_block: false,
            sender: None,
            receiver: ValueLoc::Nil,
            slots: vec![],
            ctx: CtxLoc::None,
        });
        rec.record_site(
            0x10,
            SafepointState {
                scope,
                bci: 0,
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![],
            },
        );
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(&mut vm, nm, blob, pcdescs);
        let pc = vm.code_table.get(nm_id).unwrap().code.base as usize + 0x10;

        assert_eq!(vm.stats.deopt_count, 0);
        let _resume = deoptimize_frame(
            &mut vm,
            FrameView {
                fp,
                pc,
                nm: nm_id,
                incoming_result: None,
            },
        );
        assert_eq!(vm.stats.deopt_count, 1);
    }

    /// S13 step 7a END-TO-END (runtime path, no real brk): hand-build a
    /// compiled frame + a recorder-produced scope blob, then drive the same
    /// two calls `rt_uncommon_trap` makes — `deoptimize_frame` (materialize)
    /// then `interpret_active` (run) — and assert the interpreter runs the
    /// deoptee from the resume bci to completion and returns the known result.
    /// This is `rt_uncommon_trap`'s body minus the raw-pointer `find_by_pc`
    /// deref (which `deopt_trap.rs`'s own `rt_uncommon_trap_runs_to_completion`
    /// covers), so it exercises the whole safe runtime path here where the
    /// `#![deny(unsafe_code)]` materializer lives.
    ///
    /// Deoptee: `push_smi_i8(0x5A); return_tos` (argc 0, ntemps 0). We deopt at
    /// a re-execute site keyed at bci 0 with an EMPTY recorded operand stack —
    /// so the nested run starts at bci 0, executes both bytecodes, and delivers
    /// the smi `0x5A` as the activation's result. A known value produced purely
    /// by interpreting from the resume point.
    #[test]
    fn rt_uncommon_trap_runtime_path_runs_to_completion() {
        let mut vm = test_vm();

        // The deoptee body: push a known smi, return it.
        let known: i64 = 0x5A;
        let method = {
            let mut b = BytecodeBuilder::new();
            b.push_smi_i8(known as i8); // bci 0
            b.ret_tos(); // bci 2
            let sel = vm.universe.intern(b"deoptee_e2e");
            b.finish(&mut vm, sel, 0, 0)
        };

        let nil = vm.universe.nil_obj;
        let nm = build_nmethod_with_pool(&mut vm, method, nil);

        // Physical frame: just the receiver at FrameSlot -8 (argc 0, no temps).
        let receiver_val = SmallInt::new(0x11).oop();
        let phys: [u64; 2] = [receiver_val.raw(), 0];
        let fp = (&phys[1]) as *const u64 as usize;

        // Recorder: one depth-1 scope, no slots, receiver from a FrameSlot,
        // Context None. A re-execute uncommon-trap site at bci 0 with an empty
        // operand stack (the interpreter rebuilds it from bci 0).
        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0, // word 0 of the pool is the method oop
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-8),
            slots: vec![],
            ctx: CtxLoc::None,
        });
        rec.record_site(
            0x10,
            SafepointState {
                scope,
                bci: 0,
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![],
            },
        );
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(&mut vm, nm, blob, pcdescs);
        let pc = vm.code_table.get(nm_id).unwrap().code.base as usize + 0x10;

        let sp_before = vm.stack.sp;
        let deopt_before = vm.stats.deopt_count;

        // ── The two calls `rt_uncommon_trap` makes (safe path): materialize,
        //    then run. ──────────────────────────────────────────────────────
        let resume = deoptimize_frame(
            &mut vm,
            FrameView {
                fp,
                pc,
                nm: nm_id,
                incoming_result: None, // reexecute site
            },
        );
        assert_eq!(resume.resume_bci, 0, "reexecute at bci 0 resumes at bci 0");
        assert_eq!(
            resume.base_sp, sp_before,
            "base_sp is the pre-push watermark"
        );
        // deoptimize_frame left vm.stack.fp at the innermost materialized frame.
        assert!(vm.stack.has_frame(), "a materialized frame is active");

        let result = crate::interpreter::interpret_active(&mut vm, resume);

        // The interpreter ran `push_smi 0x5A; return_tos` from bci 0 and
        // delivered the smi.
        assert_eq!(
            result,
            SmallInt::new(known).oop(),
            "the nested run returns the deoptee's computed value"
        );
        // The deopt counter bumped once (M1).
        assert_eq!(vm.stats.deopt_count, deopt_before + 1);
        // The nested run popped back to the pre-deopt watermark (the outermost
        // frame's ENTRY_FRAME_SENTINEL stop) and restored the ambient (empty)
        // activation — vm.stack is exactly as it was before deopt.
        assert_eq!(vm.stack.sp, sp_before, "stack unwound back to base_sp");
        assert!(
            !vm.stack.has_frame(),
            "the ambient (no-frame) outer activation was restored"
        );
    }

    /// S14 step 7-IV-b: `ValueLoc::ElidedClosure` materialization. A scope
    /// whose SLOT and whose reexecute SITE STACK each carry an elided closure
    /// (pool word 1 = the CompiledBlock) must rebuild a frame whose temp and
    /// operand-stack entry are REAL `BlockClosure`s: method = the block,
    /// `copied[0]` = the (root) frame's receiver, and `home_ref` naming the
    /// materialized frame's own fp + serial (a live home for later
    /// `value`/`nlr` sends).
    #[test]
    fn materialize_elided_closure_slot_and_stack() {
        let mut vm = test_vm();

        // Deoptee: argc=1, ntemps=1; resume at bci 1 (model height 1 — one
        // push before it).
        let method = {
            let mut b = BytecodeBuilder::new();
            b.push_nil(); // bci 0 -> h1
            b.ret_tos(); // bci 1 (resume point)
            let sel = vm.universe.intern(b"deopteeEc");
            b.finish(&mut vm, sel, 1, 1)
        };
        // The CompiledBlock the phantom names (send-free `[42]`).
        let blk = crate::bytecode::builder::build_standalone_block(
            &mut vm,
            0,
            0,
            false,
            0,
            false,
            |bb, _vm| {
                bb.push_smi_i8(42);
                bb.block_return_tos();
            },
        );

        let receiver_val = SmallInt::new(0x77).oop();
        let arg0_val = SmallInt::new(0x88).oop();
        // Pool word 1 = the block MethodOop (what ElidedClosure(1) names).
        let nm = build_nmethod_with_pool(&mut vm, method, blk.oop());

        let phys: [u64; 3] = [receiver_val.raw(), arg0_val.raw(), 0];
        let fp = (&phys[2]) as *const u64 as usize;

        let mut rec = ScopeDescRecorder::new();
        let scope = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0,
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-16),
            slots: vec![
                ValueLoc::FrameSlot(-8),    // arg0
                ValueLoc::ElidedClosure(1), // temp0 = the phantom closure
            ],
            ctx: CtxLoc::None,
        });
        rec.record_site(
            0x10,
            SafepointState {
                scope,
                bci: 1,
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![ValueLoc::ElidedClosure(1)], // phantom on the stack too
            },
        );
        let (blob, pcdescs) = rec.pack();
        let nm_id = install_nmethod(&mut vm, nm, blob, pcdescs);
        let nm_ref = vm.code_table.get(nm_id).unwrap();
        let pc = nm_ref.code.base as usize + 0x10;

        let sp_before = vm.stack.sp;
        let _resume = deoptimize_frame(
            &mut vm,
            FrameView {
                fp,
                pc,
                nm: nm_id,
                incoming_result: None,
            },
        );

        let fp_new = sp_before + 2; // receiver + 1 arg pushed below the frame
        let f = Frame { fp: fp_new };
        f.verify(&vm.stack);
        assert_eq!(f.receiver(&vm.stack), receiver_val);
        assert_eq!(f.temp(&vm.stack, 0), arg0_val, "arg0 untouched");

        // temp0: a REAL closure materialized from the phantom.
        let c1 = crate::oops::wrappers::ClosureOop::try_from(f.temp(&vm.stack, 1))
            .expect("temp0 must hold a materialized BlockClosure");
        assert_eq!(
            c1.method().oop().raw(),
            blk.oop().raw(),
            "closure method = the block"
        );
        assert_eq!(
            c1.ncopied(),
            1,
            "non-capturing block: copied[] = [home receiver]"
        );
        assert_eq!(
            c1.copied(0),
            receiver_val,
            "copied[0] = the home (root) receiver"
        );
        let h1 = crate::oops::home_ref::unpack_home_ref(c1.home());
        assert_eq!(h1.fp, fp_new, "home_ref fp = the materialized ROOT frame");
        assert_eq!(
            h1.serial as i64,
            f.serial(&vm.stack) as i64,
            "home_ref serial = the materialized frame's serial (a LIVE home)"
        );

        // Operand stack: a second, DISTINCT closure with the same shape.
        let opnd_base = fp_new + FRAME_TEMPS_BASE + method.ntemps();
        assert_eq!(vm.stack.sp, opnd_base + 1, "one operand value pushed");
        let c2 = crate::oops::wrappers::ClosureOop::try_from(vm.stack.get(opnd_base))
            .expect("site stack entry must hold a materialized BlockClosure");
        assert_ne!(
            c1.oop().raw(),
            c2.oop().raw(),
            "slot and stack phantoms are independent allocations"
        );
        assert_eq!(c2.method().oop().raw(), blk.oop().raw());
        let h2 = crate::oops::home_ref::unpack_home_ref(c2.home());
        assert_eq!(h2.fp, fp_new);
    }

    // Test scaffolding (nmethod-building) is shared with
    // `codecache::deopt_trap`'s own `rt_uncommon_trap` end-to-end test — see
    // `super::test_support`.
    use super::test_support::{build_nmethod_with_pool, install_nmethod};
}
