//! Unified stack walker (`sprint_s12_detail.md` D3): [`FrameView`] +
//! [`walk_frames`], classifying every activation on the (single) process
//! while walking outward from the innermost one — interpreted frames on
//! `vm.stack`, and native frames (compiled nmethods, the runtime stubs
//! that call into Rust, the `call_stub` I→C trampoline) that have no
//! Rust-visible representation at all. Primary consumer: `memory::roots`'
//! `each_code_root` (S12 D4.1, GC root enumeration — not yet wired up,
//! S12 step 4/5's job).
//!
//! `runtime::error::print_stack_trace` is deliberately NOT re-pointed onto
//! this, despite D3's own text suggesting it: `stub_poll` never sets the
//! anchor (S10 D5.6 — it never allocates/GCs, so `walk_frames`' actual
//! consumer never needs to see a Poll-reached frame), but a
//! `MACVM_TRACE`-style trace triggered from INSIDE `rt_poll`
//! (`trace_on_poll`, `tests_s10.md`'s `mixed_trace_golden`) very much DOES
//! need to see the compiled frame that called `Poll` — `AdapterKind::Poll`
//! exists for exactly this reason, but making it reachable would mean
//! teaching `stub_poll`'s own hand-rolled prologue/epilogue (deliberately
//! NOT `emit_stub_prologue`, since it saves x0-x15, not just x0-x5) to
//! ALSO tag the anchor — a real, separate change to an already-working,
//! heavily-exercised mechanism, not something this step's own walker work
//! should fold in under its own momentum. `print_stack_trace` keeps its
//! existing, narrower `tier_links.last()`-based mechanism for now; SPEC-
//! QUESTION tracked in `sprint_s12_detail.md`'s own STEP-3 NOTES.
//!
//! # The anchor, and why classifying its own frame needs a fourth field
//!
//! A runtime stub (`codecache::stubs`'s `resolve`/`c2i_shared`/
//! `mega_shared`/`dnu`/`must_be_boolean`/`alloc_slow` — the six that call
//! `emit_stub_prologue`) writes `vm.reg_block.last_compiled_{fp,pc}`
//! before calling into Rust: `last_compiled_fp` = the stub's own x29,
//! `last_compiled_pc` = x30 at that moment. Standard AArch64 frame-pointer
//! convention means x30, at a callee's own prologue, is ALWAYS the
//! address inside its CALLER's code where control resumes — i.e.
//! `last_compiled_pc` describes the stub's CALLER (a real compiled
//! method), never the stub's own code. None of the six is ever reached
//! via a `bl`/`blr` FROM another stub or adapter (per-instance c2i/mega
//! trampolines and PICs are confirmed tail-jump-only, `b`, never touching
//! x30 — S11's own P2 invariant), so there is no pc anywhere that's
//! "inside this stub's own code" to classify it with — `last_compiled_kind`
//! (`layout::VMREG_LAST_COMPILED_KIND_OFFSET`, written by each of the six
//! stubs' own preamble, `codecache::stubs::KIND_*`) exists precisely to
//! answer "which of the six" directly, since no code-range lookup can.
//!
//! # Where a walk starts: the tier journal decides, not the anchor alone
//!
//! During a reentrant C→I call the anchor legitimately stays pointed at
//! the outer `c2i_shared` frame the whole time — even while the nested
//! interpreted dispatch is the true innermost activation. Step 3
//! originally resolved that tension by having `run_method_reentrant`
//! CLEAR the anchor around itself (and this walker start anchor-first);
//! step 7 replaced that with the journal-first start rule documented at
//! `walk_frames`' own head: `vm.tier_links.last()` — the one record that
//! cannot misorder the crossings — picks the innermost side, and the
//! anchor is only consulted when the journal says native is innermost
//! (or as a loud-failure breadcrumb when the journal is empty/torn). The
//! clearing approach turned out to HIDE the whole native chain whenever
//! the nested interpreted target was a frameless successful primitive —
//! see the start-rule comment in `walk_frames` for the full account.

// This module only (not the rest of `runtime`, whose own doc comment
// still says "no unsafe here") — mirrors `codecache::mod`'s own
// `#![allow(unsafe_code)]` boundary, for the same reason: walking raw
// native stack memory has no safe-Rust equivalent.
#![allow(unsafe_code)]

use crate::codecache::nmethod::NmethodId;
use crate::codecache::stubs::{
    KIND_ALLOC_SLOW, KIND_BOX_DOUBLE, KIND_BOX_FLOAT32X4, KIND_BOX_FLOAT64X2, KIND_BOX_INT32X4,
    KIND_C2I, KIND_CALL_PRIMITIVE, KIND_DEOPT_BRIDGE, KIND_DNU, KIND_MEGA, KIND_MUST_BE_BOOLEAN,
    KIND_NLR_ORIGINATE, KIND_RESOLVE, KIND_VALUE_DISPATCH,
};
use crate::interpreter::stack::Frame;
use crate::oops::layout::ENTRY_FRAME_SENTINEL;
use crate::runtime::vm_state::{PendingDeopt, TierLink, VmState};

/// Which of the six anchor-setting runtime stubs a `FrameView::Adapter`
/// belongs to (module doc above) — plus `Poll`, kept only for this enum's
/// own completeness: `stub_poll` never calls `emit_stub_prologue` at all
/// (S10 D5.6 — it never allocates or GCs, so nothing beneath it on the
/// native stack could ever trigger a walk), so `Poll` is provably never
/// actually produced by [`walk_frames`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdapterKind {
    Resolve,
    C2i,
    Mega,
    Dnu,
    MustBeBoolean,
    AllocSlow,
    /// A compiled method's own primitive-call shim, mid-call into a
    /// `PrimFn` (`compiler::driver`'s shimmable-primitive eligibility,
    /// `codecache::stubs::rt_call_primitive`). Owns exactly the calling
    /// method's own `receiver + real args` RootSpill slots — see
    /// `memory::roots::real_oop_rootspill_slots`, which reads the count
    /// from `Nmethod::prim_call_argc_plus_recv` via the caller_pc.
    CallPrimitive,
    /// S24 A1: a compiled BLOCK body mid-`rt_nlr_originate` (parking a
    /// non-local return — `codecache::stubs::build_stub_nlr_originate`).
    /// Owns exactly 2 RootSpill slots, fixed (x0 = the closure, x1 = the
    /// NLR value), like `MustBeBoolean`/`AllocSlow`'s fixed arms.
    NlrOriginate,
    /// S24 A2: a compiled `value`-family send site's block-dispatch stub
    /// (`codecache::stubs::build_stub_value_dispatch`), mid-`rt_value_target`
    /// or, more importantly for GC, mid-`rt_value_fallback` (whose nested
    /// interpretation can collect). Owns the caller site's own
    /// `IcSite::argc` RootSpill slots — the closure + its args — recovered
    /// via `caller_pc` exactly like `Resolve|C2i|Mega|Dnu`.
    ValueDispatch,
    /// Float fast-path: a compiled `Ir::FBox`'s eden-overflow tail
    /// mid-`rt_box_double` (`codecache::stubs::build_stub_box_double`).
    /// Owns ZERO RootSpill oop slots: its x0 carries raw f64 payload bits,
    /// not an oop — scanning it would treat float bits as a heap pointer.
    BoxDouble,
    /// SIMD NEON fast-path: a compiled `Ir::VecArith`'s eden-overflow tail
    /// mid-`rt_box_float64x2` (`codecache::stubs::build_stub_box_float64x2`).
    /// Owns ZERO RootSpill oop slots: its x0/x1 carry raw f64 lane bits, not
    /// oops — same posture as `BoxDouble`.
    BoxFloat64x2,
    /// SIMD: a compiled `Ir::VecArith`'s eden-overflow tail for a `Float32x4`
    /// (`build_stub_box_float32x4`). Zero RootSpill oop slots, as `BoxFloat64x2`.
    BoxFloat32x4,
    /// SIMD: same, for an `Int32x4` (`build_stub_box_int32x4`). Zero slots.
    BoxInt32x4,
    Poll,
    /// S14 step 9: not a real adapter — the synthetic anchor
    /// [`deopt_bridge_link`] plants so a walk during `interpret_active` can
    /// bridge an ABANDONED deopted frame onto its caller chain. Owns no
    /// root-spill slots at all (the dead frame's oops live on in the
    /// materialized interpreter frames).
    DeoptBridge,
}

impl AdapterKind {
    /// `codecache::stubs::KIND_*`'s own numbers — ownership of the mapping
    /// lives there (it's the module that actually emits the `movz`/`str`
    /// writing them); this just mirrors it for the read side. Panics on an
    /// unrecognized value: a VM-internal-consistency bug (a stub added
    /// later that forgot to tag itself, or memory corruption), not a
    /// user-triggerable condition — same posture as `oopmap::verify`.
    fn from_raw(raw: u64) -> AdapterKind {
        match raw {
            KIND_RESOLVE => AdapterKind::Resolve,
            KIND_C2I => AdapterKind::C2i,
            KIND_MEGA => AdapterKind::Mega,
            KIND_DNU => AdapterKind::Dnu,
            KIND_MUST_BE_BOOLEAN => AdapterKind::MustBeBoolean,
            KIND_ALLOC_SLOW => AdapterKind::AllocSlow,
            KIND_CALL_PRIMITIVE => AdapterKind::CallPrimitive,
            KIND_NLR_ORIGINATE => AdapterKind::NlrOriginate,
            KIND_VALUE_DISPATCH => AdapterKind::ValueDispatch,
            KIND_BOX_DOUBLE => AdapterKind::BoxDouble,
            KIND_BOX_FLOAT64X2 => AdapterKind::BoxFloat64x2,
            KIND_BOX_FLOAT32X4 => AdapterKind::BoxFloat32x4,
            KIND_BOX_INT32X4 => AdapterKind::BoxInt32x4,
            // The deopt trampolines' record (see `KIND_DEOPT_BRIDGE`'s own
            // doc): same walker semantics as the synthetic bridge link —
            // no RootSpill slots, pass through to the deoptee frame.
            KIND_DEOPT_BRIDGE => AdapterKind::DeoptBridge,
            other => panic!(
                "AdapterKind::from_raw: unrecognized last_compiled_kind {other} -- the anchor's \
                 kind tag and this mapping have drifted apart, or the anchor is corrupt"
            ),
        }
    }
}

/// One activation, classified (S12 D3). `Interpreted`/`Compiled` are the
/// two "ordinary" tiers; `Adapter` is one of the six runtime stubs,
/// currently mid-call into Rust; `CallStub` marks the I→C trampoline
/// boundary, where native walking hands back to the process stack.
///
/// `Adapter` can ONLY ever be the first frame of a native-mode segment —
/// reached either from the anchor itself, or from a `TierLink::
/// IntoInterpreter` transition (both come with fp/pc/kind already known:
/// the anchor's own fields, or `AdapterKind::C2i` unconditionally for
/// `IntoInterpreter`, since `rt_interpret_call` is the ONLY place that
/// variant is ever constructed). Stepping outward via a raw `[fp]`/`[fp+8]`
/// read can never land back inside one of the six stubs' own code — none
/// is ever the target of a `bl`/`blr` from another stub or from compiled
/// code in a way that would leave a chaseable return address pointing
/// back into it — so a stepped-to native frame is always either a real
/// nmethod or `call_stub`'s own frame, never `Adapter` again.
#[derive(Clone, Copy, Debug)]
pub enum FrameView {
    Interpreted {
        fp: usize,
    },
    Compiled {
        fp: u64,
        ret_pc: u64,
        nm: NmethodId,
    },
    Adapter {
        fp: u64,
        kind: AdapterKind,
        /// The stub's OWN caller's resume address — `last_compiled_pc`
        /// (or a `TierLink::IntoInterpreter.compiled_ret_pc`), i.e. the
        /// address inside a REAL compiled nmethod where the original send/
        /// alloc/runtime site lives. Root-scanning (S12 step 4/5, not this
        /// module) uses it to look up that site's own real argc — e.g. a
        /// `resolve`/`mega`/`dnu`/`c2i` frame's RootSpill area holds
        /// whatever was in x0..x5 BEFORE the call, which is only
        /// meaningful up to the site's own real arg count; the remaining
        /// slots may be stale, non-oop register content from the
        /// compiled method's own unrelated register allocation.
        caller_pc: u64,
    },
    CallStub {
        fp: u64,
    },
}

enum Mode {
    /// Just entered native mode — from the anchor, or from a
    /// `TierLink::IntoInterpreter` transition. `(fp, pc, kind)` are ALL
    /// already known; the corresponding `FrameView` is always `Adapter`.
    Anchor(u64, u64, AdapterKind),
    /// Stepping via a raw fp-chain read — `pc` must be classified (an
    /// nmethod's own range, or `call_stub`'s fixed range; never one of
    /// the six stubs, per this module's own invariant, documented on
    /// [`FrameView::Adapter`]).
    NativeStep(u64, u64),
    Interp(usize),
    Done,
}

/// A generous but finite bound (`tests_s12.md`'s
/// `walker_terminates_on_torn_tierlinks`): a genuinely well-formed
/// process/native stack is always far shorter than this. Hitting it means
/// `tier_links` or the native fp-chain is corrupt (torn, mismatched, or a
/// cycle) — looping forever trying to walk it would be worse than a clean
/// panic naming the problem.
const MAX_WALK_STEPS: usize = 100_000;

/// # Safety
/// `addr` must be a live native-stack address belonging to a frame this
/// walker's own classification just established (the anchor's own fp, a
/// `TierLink`'s own recorded fp, or the result of a previous, already-
/// validated step) — never called on an arbitrary or externally-derived
/// value. Sound in this single-threaded VM because nothing can pop a
/// frame out from under a walk that runs to completion without yielding.
unsafe fn read_u64(addr: u64) -> u64 {
    unsafe { *(addr as *const u64) }
}

fn call_stub_contains(vm: &VmState, pc: u64) -> bool {
    let h = vm.stubs.call_stub;
    let base = h.base as u64;
    pc >= base && pc < base + h.len as u64
}

/// S13 §2c: translate a native saved-LR read from a frame record back to the
/// pc it names in the caller nmethod. Normally the identity — a saved-LR IS the
/// caller's resume pc. But if §2c ([`redirect_returns_into_nm`]) redirected it
/// to `deopt_return_trampoline` (because the caller `fp_next` is an in-flight
/// `NotEntrant` activation awaiting a lazy return-path deopt), the raw slot no
/// longer points inside that caller's nmethod; the ORIGINAL pc lives in
/// `pending_deopts[fp_next]` (keyed by the victim/caller's own fp). Returning it
/// keeps `find_by_pc` — and every walk consumer (GC root scan, the step-10
/// zombie sweep) — classifying the victim frame at its true return safepoint.
/// A one-`u64`-compare no-op whenever no redirect is pending (the common case).
fn resolve_redirected_lr(vm: &VmState, fp_next: u64, saved_lr: u64) -> u64 {
    if saved_lr == vm.stubs.deopt_return_addr() {
        vm.pending_deopts
            .get(&(fp_next as usize))
            .unwrap_or_else(|| {
                panic!(
                    "walk_frames: saved-LR at fp {fp_next:#x} is deopt_return_trampoline but no \
                     pending_deopts entry keys it -- §2c redirected a slot without recording its \
                     origin (VM-consistency bug)"
                )
            })
            .orig_ret_pc as u64
    } else {
        saved_lr
    }
}

/// S14 step 9: the tier link that lets a walk BRIDGE an ABANDONED (deopted)
/// compiled frame while its materialized interpreter replacement runs
/// (`interpret_active`). The materialized region bottoms at an
/// `ENTRY_FRAME_SENTINEL`, and without a paired `IntoInterpreter` link any
/// walk during the nested run panicked at the sentinel↔`IntoCompiled`
/// mismatch (the MACVM_GC_STRESS × JIT abort). The link's anchor is the DEAD
/// frame's own fp with its saved-return pc (`[fp+8]`, §2c-translated), so the
/// `Anchor` step lands the walk directly on the dead frame's CALLER,
/// classified by the caller's own return safepoint — the dead frame itself is
/// SKIPPED entirely (its oops were consumed by the materializer and live on
/// in the interpreter frames the linear stack scan already covers).
pub fn deopt_bridge_link(vm: &VmState, dead_fp: u64) -> crate::runtime::vm_state::TierLink {
    // SAFETY: `dead_fp` is the deopted frame's own x29, still a live native
    // frame (the deopt stub sits below it) for the whole nested run — the
    // same reads the walker itself performs on anchored frames.
    let caller_fp = unsafe { read_u64(dead_fp) };
    let raw_ret = unsafe { read_u64(dead_fp + 8) };
    let ret_pc = resolve_redirected_lr(vm, caller_fp, raw_ret);
    crate::runtime::vm_state::TierLink::DeoptBridge {
        dead_fp,
        caller_ret_pc: ret_pc,
    }
}

/// D3: walk every activation of the (single) process, innermost first,
/// interleaving native compiled/stub frames and process-stack interpreter
/// frames via `vm.tier_links` + the anchor. `f` is called once per
/// activation, innermost first; this function itself never allocates or
/// touches `vm.stack`'s own `sp`/`fp` (a pure read-only walk).
///
/// # Panics
/// If the native fp-chain or `vm.tier_links` turn out to be inconsistent
/// with each other (an unclassifiable native pc, a boundary crossing that
/// doesn't find the tier-link kind it expects, or exceeding
/// [`MAX_WALK_STEPS`]) — always a VM-internal-consistency bug, never a
/// condition guest code can trigger.
pub fn walk_frames(vm: &VmState, mut f: impl FnMut(FrameView)) {
    let mut link_idx = vm.tier_links.len();
    // Start-of-walk rule (S12 step 7, revised twice from step 3 — both
    // predecessors were wrong in ways only the fallen bridge could
    // expose): the innermost side is decided by the most recent TIER
    // CROSSING, i.e. `tier_links.last()` — the one journal that cannot
    // lie about ordering — never by the anchor alone and never by
    // `has_frame` alone.
    //
    // - Last crossing was C→I (`IntoInterpreter`): the interpreter side
    //   is innermost. If it has live frames, walk them (its entry
    //   sentinel consumes that same link on the way back out, as
    //   always). If it has NO frames — a reentrant send whose target's
    //   PRIMITIVE succeeded collects BEFORE any frame is pushed
    //   (`try_primitive` precedes activation) — there is nothing
    //   interpreted to visit (the receiver/args live in plain `vm.stack`
    //   slots `for_each_root` already covers), so consume the link HERE
    //   and enter the native chain it records. Step 3's design instead
    //   had `run_method_reentrant` clear the anchor and keyed the start
    //   on anchor-nonzero — which left this exact frameless case with
    //   NOTHING to start from, silently skipping every compiled frame's
    //   spill slots (found by `mid_loop_forced_scavenge` the moment the
    //   bridge came down; the interim "has_frame first" revision fixed
    //   that but misrouted the opposite case, a live PAUSED interpreted
    //   caller beneath a compiled frame's own alloc-slow collection).
    // - Last crossing was I→C (`IntoCompiled`): the native side is
    //   innermost; the anchor names the stub currently in Rust (it must
    //   be set — GC only ever starts inside an anchor-setting stub's own
    //   runtime call).
    // - No crossings at all: pure single-tier state — a still-set anchor
    //   (possible only if something upstream tore the journal) is still
    //   honored so the walk fails LOUDLY at the call_stub boundary
    //   instead of silently reporting an empty world
    //   (`walker_terminates_on_torn_tierlinks` pins exactly this); else
    //   plain interpreter frames or nothing.
    // "Live interpreted frame" needs BOTH checks: `has_frame`, AND fp not
    // being `run_method`'s own temporary ENTRY_FRAME_SENTINEL spoof — for
    // exactly the duration of an entry's `try_primitive` attempt, fp is
    // deliberately set to the sentinel (see `run_method`'s own doc) while
    // `has_frame` still reflects the OUTER paused activation. A
    // `can_allocate` primitive collecting inside that window (every
    // allocation, under GC_STRESS=1) must be treated exactly like the
    // frameless case: nothing interpreted is live for THIS entry (its
    // receiver/args are plain `vm.stack` slots `for_each_root` covers),
    // and blindly walking `fp == usize::MAX` overflows on the first slot
    // read (found by the flagship gate the first time a reentrant
    // allocating primitive scavenged for real).
    let live_interp =
        vm.stack.has_frame() && vm.stack.fp != crate::oops::layout::ENTRY_FRAME_SENTINEL as usize;
    let mut mode = match vm.tier_links.last() {
        // S14 step 9: a DeoptBridge behaves exactly like IntoInterpreter at
        // walk start — the materialized interpreter frames are innermost for
        // the nested run's whole lifetime; the frameless edge case consumes
        // the link and enters the caller chain it anchors (same rule, same
        // reasons, as the C2i arm below).
        Some(TierLink::DeoptBridge {
            dead_fp,
            caller_ret_pc,
        }) => {
            if live_interp {
                Mode::Interp(vm.stack.fp)
            } else {
                link_idx -= 1;
                Mode::Anchor(*dead_fp, *caller_ret_pc, AdapterKind::DeoptBridge)
            }
        }
        Some(TierLink::IntoInterpreter {
            compiled_fp,
            compiled_ret_pc,
        }) => {
            if live_interp {
                Mode::Interp(vm.stack.fp)
            } else {
                link_idx -= 1;
                Mode::Anchor(*compiled_fp, *compiled_ret_pc, AdapterKind::C2i)
            }
        }
        Some(TierLink::IntoCompiled { .. }) => {
            assert_ne!(
                vm.reg_block.last_compiled_fp, 0,
                "walk_frames: innermost side is native (last tier crossing was IntoCompiled) \
                 but no anchor is set -- GC can only start inside an anchor-setting stub"
            );
            Mode::Anchor(
                vm.reg_block.last_compiled_fp,
                vm.reg_block.last_compiled_pc,
                AdapterKind::from_raw(vm.reg_block.last_compiled_kind),
            )
        }
        None => {
            if vm.reg_block.last_compiled_fp != 0 {
                Mode::Anchor(
                    vm.reg_block.last_compiled_fp,
                    vm.reg_block.last_compiled_pc,
                    AdapterKind::from_raw(vm.reg_block.last_compiled_kind),
                )
            } else if live_interp {
                Mode::Interp(vm.stack.fp)
            } else {
                Mode::Done
            }
        }
    };

    for _ in 0..MAX_WALK_STEPS {
        mode = match mode {
            Mode::Done => return,
            Mode::Anchor(fp, pc, kind) => {
                f(FrameView::Adapter {
                    fp,
                    kind,
                    caller_pc: pc,
                });
                // SAFETY: `fp` is `last_compiled_fp` or a prior
                // `TierLink::IntoInterpreter.compiled_fp`, both written
                // only by a live stub's own `stp x29,x30` prologue at the
                // moment control entered Rust — a live native frame for
                // as long as this walk runs (single-threaded VM).
                let fp_next = unsafe { read_u64(fp) };
                // The next frame's own classifying pc is the CALLER's
                // resume point, which is exactly `pc` here already (==
                // `[fp+8]` by construction — the stub's own prologue wrote
                // the same x30 value to both the anchor and its own
                // saved-lr slot) — no separate memory read needed.
                Mode::NativeStep(fp_next, pc)
            }
            Mode::NativeStep(fp, pc) => {
                if let Some(nm_id) = vm.code_table.find_by_pc(pc) {
                    f(FrameView::Compiled {
                        fp,
                        ret_pc: pc,
                        nm: nm_id,
                    });
                    // SAFETY: `fp` is a live compiled frame's own x29,
                    // established by `emit()`'s own
                    // `stp x29,x30,...; mov x29,sp` prologue, shared by
                    // every published nmethod.
                    let fp_next = unsafe { read_u64(fp) };
                    let pc_next = unsafe { read_u64(fp + 8) };
                    // S13 §2c: if this frame's saved-LR was redirected to
                    // `deopt_return_trampoline` (an in-flight `NotEntrant`
                    // activation whose callee hasn't yet returned), the raw slot
                    // no longer names a pc inside the caller (victim) nmethod —
                    // `find_by_pc` would miss and the walk would panic. The real
                    // return pc for classifying `fp_next` (the victim frame)
                    // lives in `pending_deopts[fp_next]`. Translate it back so
                    // the walk stays valid while a redirect is pending (a GC or
                    // a later invalidation walk over the SAME chain — and the
                    // step-10 zombie sweep, which counts a "redirected-LR" as a
                    // live reference to the victim nm, §2c/D1.3).
                    let pc_next = resolve_redirected_lr(vm, fp_next, pc_next);
                    Mode::NativeStep(fp_next, pc_next)
                } else if call_stub_contains(vm, pc) {
                    f(FrameView::CallStub { fp });
                    link_idx = link_idx.checked_sub(1).unwrap_or_else(|| {
                        panic!(
                            "walk_frames: reached call_stub with no matching \
                             TierLink::IntoCompiled left -- tier_links and the native fp-chain \
                             disagree"
                        )
                    });
                    match vm.tier_links[link_idx] {
                        // S15 (OSR): an OSR transition POPPED the interpreter
                        // frame it replaced; when that frame was the dispatch
                        // loop's ENTRY frame, the link records the sentinel
                        // (run_method's own spoof convention) — there is no
                        // interpreter frame to visit, so consume the NEXT
                        // boundary exactly like the sentinel-bottom arm does.
                        TierLink::IntoCompiled { interp_frame, .. }
                            if interp_frame == ENTRY_FRAME_SENTINEL as usize =>
                        {
                            if link_idx == 0 {
                                Mode::Done
                            } else {
                                link_idx -= 1;
                                match vm.tier_links[link_idx] {
                                    TierLink::IntoInterpreter {
                                        compiled_fp,
                                        compiled_ret_pc,
                                    } => {
                                        Mode::Anchor(compiled_fp, compiled_ret_pc, AdapterKind::C2i)
                                    }
                                    TierLink::DeoptBridge {
                                        dead_fp,
                                        caller_ret_pc,
                                    } => Mode::Anchor(
                                        dead_fp,
                                        caller_ret_pc,
                                        AdapterKind::DeoptBridge,
                                    ),
                                    TierLink::IntoCompiled { .. } => panic!(
                                        "walk_frames: an OSR'd entry frame's link must pair \
                                         with IntoInterpreter/DeoptBridge below, found \
                                         IntoCompiled"
                                    ),
                                }
                            }
                        }
                        TierLink::IntoCompiled { interp_frame, .. } => Mode::Interp(interp_frame),
                        TierLink::DeoptBridge { .. } => panic!(
                            "walk_frames: call_stub's own boundary must pair with \
                             TierLink::IntoCompiled, found DeoptBridge instead"
                        ),
                        TierLink::IntoInterpreter { .. } => panic!(
                            "walk_frames: call_stub's own boundary must pair with \
                             TierLink::IntoCompiled, found IntoInterpreter instead"
                        ),
                    }
                } else {
                    panic!(
                        "walk_frames: native pc {pc:#x} is not inside any alive nmethod or \
                         call_stub -- classification invariant broke"
                    );
                }
            }
            Mode::Interp(fp) => {
                f(FrameView::Interpreted { fp });
                let saved_fp = Frame { fp }.saved_fp(&vm.stack);
                if saved_fp == ENTRY_FRAME_SENTINEL {
                    if link_idx == 0 {
                        Mode::Done
                    } else {
                        link_idx -= 1;
                        match vm.tier_links[link_idx] {
                            TierLink::IntoInterpreter {
                                compiled_fp,
                                compiled_ret_pc,
                            } => Mode::Anchor(compiled_fp, compiled_ret_pc, AdapterKind::C2i),
                            // S14 step 9: the materialized region a deopt's
                            // nested run executes — bridge PAST the abandoned
                            // compiled frame onto its caller (the anchor pc is
                            // already the caller's own return safepoint).
                            TierLink::DeoptBridge {
                                dead_fp,
                                caller_ret_pc,
                            } => Mode::Anchor(dead_fp, caller_ret_pc, AdapterKind::DeoptBridge),
                            TierLink::IntoCompiled { .. } => panic!(
                                "walk_frames: an ENTRY_FRAME_SENTINEL must pair with \
                                 TierLink::IntoInterpreter, found IntoCompiled instead"
                            ),
                        }
                    }
                } else {
                    Mode::Interp(saved_fp as usize)
                }
            }
        };
    }
    panic!(
        "walk_frames: exceeded {MAX_WALK_STEPS} steps -- tier_links or the native fp-chain is \
         corrupt (torn, mismatched, or cyclic), not merely a very deep call stack"
    );
}

/// S13 D1 §2c — the lazy return-address redirection walk. `make_not_entrant`
/// calls this AFTER it has patched `nm`'s entries (§2b): it walks the native
/// stack (reusing [`walk_frames`], which already threads the interpreter↔
/// compiled boundaries) and, for every NATIVE frame `F` (a `Compiled` nmethod
/// frame or an `Adapter` runtime-stub frame) whose saved-LR slot `[F.fp + 8]`
/// points INTO `nm`'s published code range — i.e. `F`'s caller is an in-flight
/// activation of the now-`NotEntrant` `nm` — it:
///
/// - records `PendingDeopt { orig_ret_pc: [F.fp+8], nm }` keyed by
///   `saved_fp = [F.fp]` (the nm activation's OWN fp), and
/// - overwrites the saved-LR STACK SLOT at `F.fp + 8` with `tramp` (the
///   `deopt_return_trampoline` address) — PLAIN NATIVE-STACK DATA, so no JIT
///   write toggle and no icache flush (contrast §2b's MAP_JIT entry patching:
///   this is the process stack, not code).
///
/// The nm activation's OWN frame is never touched — it keeps running old code
/// until control next returns into the redirected slot (this path), traps, or
/// (step 10) polls. `saved_fp`/`saved_lr` follow the standard AArch64 frame
/// record: `[F.fp] = caller's fp`, `[F.fp+8] = return address into caller`.
///
/// **Idempotence / double-invalidation.** A slot already pointing at `tramp`
/// (an earlier §2c redirect for a DIFFERENT nm, or a re-run for the same one)
/// is left untouched and not re-inserted — its `PendingDeopt` was recorded when
/// the redirect first happened, keyed by that activation's own fp; overwriting
/// `orig_ret_pc` with `tramp` would lose the real return address. Interpreted
/// and `CallStub` boundary frames carry no native saved-LR into compiled code
/// and are skipped.
///
/// Only reads/writes native-stack slots via this module's own [`read_u64`] /
/// [`write_u64`], never re-deriving raw offsets elsewhere.
pub fn redirect_returns_into_nm(vm: &mut VmState, nm: NmethodId, tramp: u64) {
    // The target nm's published code range. A miss means the caller passed an
    // uninstalled id — a VM bug, since §2c runs on an nmethod §2a/§2b just
    // touched.
    let (lo, hi, pcdescs, scopes_blob) = {
        let n = vm
            .code_table
            .get(nm)
            .expect("redirect_returns_into_nm: nm must be installed");
        let base = n.code.base as u64;
        // Cloned so the closure below can read the deopt metadata without
        // holding a `&vm` borrow across `walk_frames(vm, ...)` — small, and
        // §2c is rare (invalidation only).
        (
            base,
            base + n.code.len as u64,
            n.deopt_pcdescs.clone(),
            n.deopt_scopes.clone(),
        )
    };

    // Read-only walk first (like `flush.rs`'s own frame-walk): collect the
    // slots to redirect into an owned Vec, because the walk borrows `vm`
    // immutably while the writes below need `&mut vm.pending_deopts`. Each
    // entry is `(slot_addr = F.fp+8, saved_fp = [F.fp], orig_ret_pc = [F.fp+8])`.
    struct Redirect {
        slot_addr: u64,
        saved_fp: usize,
        orig_ret_pc: usize,
    }
    let mut to_redirect: Vec<Redirect> = Vec::new();
    walk_frames(vm, |fv| {
        // Only NATIVE frames (`Compiled`/`Adapter`) have a native saved-LR slot
        // at `[fp+8]` that could be a return address into `nm`'s compiled code.
        // For both, `[fp+8]` is the standard AArch64 saved-lr (the anchor
        // stubs' `emit_stub_prologue` does `stp x29,x30`, so an `Adapter`'s
        // `[fp+8]` == its `caller_pc`). Interpreted/CallStub frames are process-
        // stack / boundary frames — skipped.
        let fp = match fv {
            FrameView::Compiled { fp, .. } => fp,
            FrameView::Adapter { fp, .. } => fp,
            FrameView::Interpreted { .. } | FrameView::CallStub { .. } => return,
        };
        // SAFETY: `fp` is a live native frame's own fp, established by
        // `walk_frames`' own classification (single-threaded VM, no frame can
        // be popped mid-walk). `[fp]`/`[fp+8]` are its saved {fp,lr} record.
        let saved_lr = unsafe { read_u64(fp + 8) };
        if saved_lr < lo || saved_lr >= hi {
            return; // caller is not the target nm
        }
        // Already redirected (double-invalidation): leave the slot and its
        // earlier `PendingDeopt` intact.
        if saved_lr == tramp {
            return;
        }
        // Only a reexecute=FALSE Call return site is a valid return-path deopt:
        // the materializer pushes the callee's result onto that site's operand
        // stack (D5 M3.3), which is meaningful only for a call return. Skip
        // anything else this frame's saved-LR might point at inside `nm`:
        // - an inline `Alloc`'s `alloc_slow` return is a reexecute=TRUE site
        //   (re-executes `basicNew`, wants NO pushed result — `deoptimize_frame`
        //   would assert), and
        // - a loop-poll return / any non-recorded address has no deopt scope.
        // A skipped frame just means the NotEntrant victim deopts at its NEXT
        // call-return boundary instead — still correct (it runs old code until
        // then, exactly as §2c's own "not touched until a boundary" rule says).
        let code_off = (saved_lr - lo) as u32;
        let is_call_return = pcdescs
            .binary_search_by_key(&code_off, |d: &crate::compiler::scopes::PcDesc| d.code_off)
            .ok()
            .is_some_and(|idx| {
                let site =
                    crate::compiler::scopes::decode_site(&scopes_blob, pcdescs[idx].site_off);
                !site.reexecute && matches!(site.kind, crate::compiler::scopes::SafepointKind::Call)
            });
        if !is_call_return {
            return;
        }
        // SAFETY: same as above — `[fp]` is this frame's saved caller fp.
        let saved_fp = unsafe { read_u64(fp) } as usize;
        to_redirect.push(Redirect {
            slot_addr: fp + 8,
            saved_fp,
            orig_ret_pc: saved_lr as usize,
        });
    });

    for r in to_redirect {
        // Record the pending deopt keyed by the nm activation's OWN fp, then
        // overwrite the saved-LR stack slot with the trampoline. Plain data —
        // no JIT toggle / icache flush.
        vm.pending_deopts.insert(
            r.saved_fp,
            PendingDeopt {
                orig_ret_pc: r.orig_ret_pc,
                nm,
            },
        );
        // SAFETY: `slot_addr` is `[F.fp+8]`, a live native-stack saved-LR slot
        // this walk just read; overwriting it with the trampoline address is
        // the whole point of §2c (single-threaded VM, slot stays live).
        unsafe { write_u64(r.slot_addr, tramp) };
    }
}

/// # Safety
/// `addr` must be a live, 8-byte-aligned native-stack slot this walker's own
/// classification just established (a native frame's `[fp+8]` saved-LR slot).
/// Sound in this single-threaded VM because nothing can pop the frame out from
/// under a redirect that runs to completion without yielding. Writes plain
/// process-stack DATA — never MAP_JIT code — so no JIT write toggle or icache
/// flush applies (contrast the §2b entry patching, which does).
unsafe fn write_u64(addr: u64, val: u64) {
    unsafe { *(addr as *mut u64) = val }
}

/// DBG0 (docs/DEBUGGER.md §4.2 step 6): PROBE's raw-seeded native walk.
///
/// [`walk_frames`] cannot serve a crash dossier: it seeds from the
/// anchor/tier-links, and the anchor is CLEARED during ordinary compiled
/// execution (`emit_stub_epilogue`), so at an async crash pc inside an
/// nmethod it panics immediately ("no anchor is set"). This walker instead
/// starts from the CAPTURED crash context's own `(fp, pc)` and follows the
/// arm64 frame-record chain (`[fp]` = caller fp, `[fp+8]` = saved lr) that
/// every published nmethod's `stp x29,x30; mov x29,sp` prologue guarantees.
///
/// Contrast with the GC walker's trust model (§4.5's rules):
/// - every fp is validated against `[stack_lo, stack_hi)` and 8-byte
///   alignment BEFORE it is dereferenced — the crash frame is the SUSPECT,
///   and a wild fp here would fault (not panic: `catch_unwind` can't help);
/// - unclassifiable pcs are REPORTED, never panicked on (the caller names
///   pcs itself — this function only emits raw `(fp, pc)` pairs);
/// - the walk stops (with a stated reason) at the call stub (bottom of the
///   native segment), a non-monotonic fp (frames must strictly ascend on
///   this downward-growing stack), an out-of-bounds fp, or the step cap.
///
/// Deopt-redirected saved-LRs are translated back through `pending_deopts`
/// exactly like the GC walk, so a mid-invalidation crash still names the
/// true caller pc. Returns `(frames_emitted, stop_reason)`.
pub fn probe_walk_native(
    vm: &VmState,
    seed_fp: u64,
    seed_pc: u64,
    stack_lo: u64,
    stack_hi: u64,
    mut f: impl FnMut(usize, u64, u64),
) -> (usize, String) {
    const PROBE_WALK_CAP: usize = 256;
    let mut fp = seed_fp;
    let mut pc = seed_pc;
    let mut steps = 0usize;
    loop {
        if steps >= PROBE_WALK_CAP {
            return (steps, format!("stopped: step cap {PROBE_WALK_CAP}"));
        }
        f(steps, fp, pc);
        steps += 1;
        if call_stub_contains(vm, pc) {
            return (
                steps,
                "stopped: reached call_stub (native segment floor)".into(),
            );
        }
        if fp < stack_lo || fp + 16 > stack_hi {
            return (steps, format!("stopped: fp {fp:#x} outside stack bounds"));
        }
        if fp & 7 != 0 {
            return (steps, format!("stopped: fp {fp:#x} misaligned"));
        }
        // SAFETY: just bounds- and alignment-checked against the caller's
        // stack range — the whole point of this walker vs the GC one.
        let fp_next = unsafe { read_u64(fp) };
        let raw_lr = unsafe { read_u64(fp + 8) };
        if fp_next == 0 {
            return (steps, "stopped: fp chain ends (saved fp = 0)".into());
        }
        if fp_next <= fp {
            return (
                steps,
                format!("stopped: non-monotonic fp chain ({fp_next:#x} <= {fp:#x})"),
            );
        }
        // A redirected saved-LR names deopt_return_trampoline; translate it
        // back — but tolerate a MISSING pending_deopts entry (report it)
        // instead of resolve_redirected_lr's panic: on the crash path the
        // bookkeeping itself is a suspect.
        pc = if raw_lr == vm.stubs.deopt_return_addr() {
            match vm.pending_deopts.get(&(fp_next as usize)) {
                Some(pd) => pd.orig_ret_pc as u64,
                None => {
                    return (
                        steps,
                        format!(
                            "stopped: saved-LR at fp {fp_next:#x} is deopt_return_trampoline \
                             with no pending_deopts entry (torn invalidation state — itself \
                             a finding)"
                        ),
                    )
                }
            }
        } else {
            raw_lr
        };
        fp = fp_next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeBuilder;
    use crate::compiler::driver;
    use crate::interpreter::compiled_call::{enter_compiled, EnterResult};
    use crate::oops::smi::SmallInt;
    use crate::oops::wrappers::{KlassOop, MethodOop};
    use crate::runtime::lookup::install_method;
    use crate::runtime::vm_state::{VmOptions, VmState};

    fn test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(1),
        })
    }

    /// A bare `test_vm()` has no world loaded — installs a real
    /// primitive-backed method by pinned id (mirrors `it_tier1.rs`'s own
    /// `install_prim` helper).
    fn install_prim(vm: &mut VmState, klass: KlassOop, name: &[u8], argc: usize, prim: i64) {
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.ret_self(); // never reached; the primitive always succeeds here
        let sel = vm.universe.intern(name);
        let m = b.finish(vm, sel, argc, 0);
        m.set_primitive(prim);
        install_method(vm, klass, sel, m);
    }

    fn compile(vm: &mut VmState, rcvr_klass: KlassOop, method: MethodOop) -> NmethodId {
        driver::compile_method(vm, rcvr_klass, method).expect("must compile")
    }

    /// Builds `AllocTarget` (a real class, via the frontend, so
    /// `basicNew`'s own `Format::Slots` klass shape is genuine) and a
    /// compiled `spawn` method (`^AllocTarget basicNew`, S11 D7's inline
    /// `Ir::Alloc`) — driven to its SLOW edge (`rt_alloc_slow`, one of the
    /// six anchor-setting stubs), so `walk_frames` has a real, in-flight
    /// native chain to observe via the `test_walk_capture` hook. Returns
    /// the compiled id and a receiver ready to push.
    fn build_and_compile_spawn(vm: &mut VmState) -> (NmethodId, crate::oops::Oop) {
        let target_sel = vm.universe.intern(b"AllocTarget");
        for item in
            crate::frontend::parser::parse_file("Object subclass: AllocTarget [ ]").expect("parse")
        {
            crate::frontend::classdef::execute_top_item(vm, item).expect("execute");
        }
        let target_assoc =
            crate::runtime::globals::global_lookup(vm, target_sel).expect("AllocTarget global");
        let target_klass = KlassOop::try_from(
            crate::oops::wrappers::MemOop::try_from(target_assoc)
                .unwrap()
                .body_oop(1),
        )
        .unwrap();
        let target_meta = crate::runtime::lookup::klass_of(vm, target_klass.oop());
        install_prim(vm, target_meta, b"basicNew", 0, 23);

        let mut b = BytecodeBuilder::new();
        b.push_global(vm, target_assoc);
        let basic_new_sel = vm.universe.intern(b"basicNew");
        b.send(vm, basic_new_sel, 0);
        b.ret_tos();
        let spawn_sel = vm.universe.intern(b"spawn");
        let method = b.finish(vm, spawn_sel, 0, 0);

        let smi_klass = vm.universe.smi_klass;
        let recv = SmallInt::new(1).oop();
        // Warm the site (mono, smi receiver, basicNew primitive) so it's
        // eligible, matching `allocation_fast_and_slow`'s own recipe.
        crate::interpreter::run_method(vm, method, recv, &[]);
        let nm_id = compile(vm, smi_klass, method);

        // Force the slow edge by filling eden HONESTLY (real, walkable
        // objects). The pre-S12-step-7 `eden.top = eden.end` lie is no
        // longer survivable here: with the D8 bridge gone, `rt_alloc_slow`'s
        // own ordinary allocation genuinely scavenges, and the scavenge's
        // entry verify walks eden up to `top` — straight into the lie's
        // uninitialized gap (the exact failure `it_gc_jit.rs`'s own
        // step-4 test already hit and documented).
        let array_klass = vm.universe.array_klass;
        let chunk_elems = 4096usize;
        let chunk_bytes = (crate::oops::layout::HEADER_WORDS + chunk_elems) * 8;
        while vm.universe.eden.end - vm.universe.eden.top >= chunk_bytes {
            crate::memory::alloc::alloc_indexable_oops(vm, array_klass, chunk_elems);
        }
        // Tail-fill until less than ONE MORE AllocTarget fits, so the
        // compiled inline bump is guaranteed to overflow.
        let need = target_klass.non_indexable_size() * crate::oops::layout::WORD_SIZE;
        while vm.universe.eden.end - vm.universe.eden.top >= need {
            crate::memory::alloc::alloc_slots(vm, target_klass);
        }
        (nm_id, recv)
    }

    /// `tests_s12.md`'s `walker_classifies_all_kinds`: a real I→C→(native
    /// safepoint) chain, executed for real (not hand-faked memory) —
    /// `walk_frames`, run from inside `rt_alloc_slow` via `test_walk_
    /// capture`, must report exactly, innermost first: the alloc_slow
    /// stub itself, `spawn`'s own compiled frame, then the `call_stub`
    /// boundary. This is a TOP-LEVEL `enter_compiled` — the warmup
    /// `run_method` completed and deactivated its entry frame, so no live
    /// interpreted frame exists below the boundary; the tier link records
    /// `ENTRY_FRAME_SENTINEL` and the walk ends there. (It used to report
    /// the DEAD entry-frame remnant as a 4th, `Interpreted` frame — an
    /// accident of the stale `vm.stack.fp` the link captured, which the B5
    /// step-3 gc-stress deopt test caught chasing overwritten remnant
    /// slots into an unbounded walk.)
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn walker_classifies_all_kinds() {
        let mut vm = test_vm();
        let (nm_id, recv) = build_and_compile_spawn(&mut vm);

        vm.stack.push(recv);
        vm.test_walk_capture = Some(Ok(Vec::new()));
        // `enter_compiled` calls the REAL call_stub -> spawn's compiled
        // code -> its Alloc's slow edge -> the REAL rt_alloc_slow, which
        // captures a walk_frames snapshot (the test hook) before
        // returning normally.
        assert_eq!(enter_compiled(&mut vm, nm_id, 0), EnterResult::Completed);

        let seen = vm
            .test_walk_capture
            .take()
            .expect("hook must have fired")
            .expect("walk_frames must not have panicked");
        assert_eq!(seen.len(), 3, "{seen:?}");
        assert!(
            matches!(
                seen[0],
                FrameView::Adapter {
                    kind: AdapterKind::AllocSlow,
                    ..
                }
            ),
            "innermost must be the alloc_slow stub itself: {seen:?}"
        );
        assert!(
            matches!(seen[1], FrameView::Compiled { nm, .. } if nm == nm_id),
            "next must be spawn's own compiled frame: {seen:?}"
        );
        assert!(
            matches!(seen[2], FrameView::CallStub { .. }),
            "outermost must be the call_stub boundary (the entry link is the \
             sentinel — no live interpreted frame exists below a top-level \
             enter_compiled): {seen:?}"
        );
    }

    /// `tests_s12.md`'s `walker_terminates_on_torn_tierlinks`: with
    /// `test_tear_tier_links_before_walk` armed, `rt_alloc_slow` pops the
    /// `TierLink::IntoCompiled` `enter_compiled` itself just pushed —
    /// so when the walker steps outward from `spawn`'s own compiled frame
    /// and reaches `call_stub`'s boundary, no matching tier link is left
    /// to resume interpreted walking from. Must panic with a clear
    /// message (caught by `rt_alloc_slow`'s own `catch_unwind`, reported
    /// back via `Err`), never loop forever or silently misclassify.
    #[cfg(target_arch = "aarch64")] // WINVM: compiles+executes A64 code; x64 tier-1 is Phase 3
    #[test]
    fn walker_terminates_on_torn_tierlinks() {
        let mut vm = test_vm();
        let (nm_id, recv) = build_and_compile_spawn(&mut vm);

        vm.stack.push(recv);
        vm.test_walk_capture = Some(Ok(Vec::new()));
        vm.test_tear_tier_links_before_walk = true;
        assert_eq!(enter_compiled(&mut vm, nm_id, 0), EnterResult::Completed);

        let err = vm
            .test_walk_capture
            .take()
            .expect("hook must have fired")
            .expect_err("walk_frames must have panicked against the torn tier_links");
        assert!(
            err.contains("no matching TierLink::IntoCompiled left"),
            "got: {err}"
        );
    }

    // ── S13 step 9: `redirect_returns_into_nm` unit tests ─────────────────
    //
    // These build a HAND-LAID native fp-chain (a stack-local `[u64]` whose
    // element addresses stand in for native frame pointers, exactly as
    // `read_frame_slot_reads_offset` / `deopt.rs`'s materializer tests do) plus
    // a matching anchor + one `IntoCompiled` tier link, so `walk_frames` walks
    // it end to end WITHOUT a real compiled call — the redirection logic (which
    // FrameView's slot gets rewritten, and to what key) is what's under test,
    // not the (separately-tested) walk classification.

    use crate::codecache::nmethod::{NmState, Nmethod};
    use crate::codecache::CodeHandle;
    use crate::compiler::assembler::CodeBlob;
    use crate::oops::layout::{FRAME_SAVED_FP, MEM_TAG};
    use crate::oops::Oop;

    fn off_test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        })
    }

    /// Publish a tiny real blob so it has a genuine `[base, base+len)` code
    /// range `find_by_pc` can classify a hand-built pc against.
    fn install_blob(vm: &mut VmState, len: usize) -> CodeHandle {
        let h = vm.code_cache.alloc(len).expect("code cache alloc");
        let blob = CodeBlob {
            code: vec![0u8; len],
            literal_off: len as u32,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        h
    }

    fn fake_nmethod(vm: &mut VmState, addr: usize, sel: &[u8], code: CodeHandle) -> NmethodId {
        // SAFETY: test-only tag-level klass shape, never dereferenced.
        let klass = unsafe {
            crate::oops::wrappers::KlassOop::from_oop_unchecked(Oop::from_raw(
                addr as u64 + MEM_TAG,
            ))
        };
        let selector = vm.universe.intern(sel);
        vm.code_table.install(Nmethod {
            osr_cold_sends: 0,
            id: NmethodId(0),
            key_klass: klass,
            key_selector: selector,
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
        })
    }

    /// `tests_s13.md`'s return-redirection walk: a hand-laid native chain
    /// adapter → callee(compiled) → victim(compiled, the NotEntrant nm) →
    /// call_stub → interpreter. Only the CALLEE frame's saved-LR points into
    /// the victim's range (its caller IS the victim), so `redirect_returns_
    /// into_nm(victim)` must (1) insert exactly one `pending_deopts` entry
    /// keyed by the victim's OWN fp (= the callee's `saved_fp`), carrying the
    /// original return pc, and (2) overwrite ONLY the callee's saved-LR STACK
    /// SLOT with the trampoline address — leaving the victim's own frame, the
    /// adapter, and every other slot untouched.
    #[test]
    fn redirect_walks_and_rewrites_only_the_callee_slot() {
        let mut vm = off_test_vm();

        // Two real published nmethods, distinct ranges: the victim (NotEntrant)
        // and the callee whose return lands in the victim.
        let victim_code = install_blob(&mut vm, 64);
        let callee_code = install_blob(&mut vm, 64);
        let victim = fake_nmethod(&mut vm, 0x1000, b"victim", victim_code);
        let _callee = fake_nmethod(&mut vm, 0x2000, b"callee", callee_code);
        // S13 §2c only redirects a return whose site is a reexecute=FALSE Call
        // deopt scope — give the victim one at code_off 0x10 (the callee's
        // return-into-victim offset below).
        {
            use crate::compiler::scopes;
            let mut rec = scopes::ScopeDescRecorder::new();
            let scope = rec.begin_scope(scopes::ScopeDescData {
                method_pool_ix: 0,
                is_block: false,
                sender: None,
                receiver: scopes::ValueLoc::Nil,
                slots: vec![],
                ctx: scopes::CtxLoc::None,
            });
            rec.record_site(
                0x10,
                scopes::SafepointState {
                    scope,
                    bci: 0,
                    kind: scopes::SafepointKind::Call,
                    reexecute: false,
                    stack: vec![],
                },
            );
            let (blob, pcdescs) = rec.pack();
            let nm = vm.code_table.get_mut(victim).unwrap();
            nm.deopt_scopes = blob;
            nm.deopt_pcdescs = pcdescs;
        }
        let pc_in_victim = victim_code.base as u64 + 0x10;
        let pc_in_callee = callee_code.base as u64 + 0x10;
        let call_stub_pc = vm.stubs.call_stub.base as u64; // inside call_stub range

        // One interpreted frame the walk lands on after the call_stub boundary;
        // its saved_fp = ENTRY_FRAME_SENTINEL ends the walk. Only slot
        // `interp_fp + FRAME_SAVED_FP` is read by the Interp arm.
        let interp_fp = vm.stack.sp;
        for _ in 0..(FRAME_SAVED_FP + 1) {
            vm.stack.push(vm.universe.nil_obj);
        }
        vm.stack.set(
            interp_fp + FRAME_SAVED_FP,
            SmallInt::new(ENTRY_FRAME_SENTINEL).oop(),
        );
        vm.tier_links.push(TierLink::IntoCompiled {
            interp_frame: interp_fp,
            entry_sp: vm.stack.sp as u64,
            nm_id: victim,
        });

        // The hand-laid native fp-chain. Each frame is a {saved_fp, saved_lr}
        // pair; a frame's fp is the address of its saved_fp slot.
        //   [0,1] adapter : saved_lr = 0 (harmless, not in victim range)
        //   [2,3] callee  : saved_fp = victim_fp, saved_lr = pc_in_victim  <-- redirected
        //   [4,5] victim  : saved_fp = &[6],      saved_lr = call_stub_pc
        //   [6,7] below   : reached only as the call_stub frame (slots unread)
        let mut arr = [0u64; 8];
        let base = arr.as_ptr() as u64;
        let adapter_fp = base;
        let callee_fp = base + 16;
        let victim_fp = base + 32;
        let below_fp = base + 48;
        arr[0] = callee_fp; // adapter.saved_fp -> callee
        arr[1] = 0; // adapter.saved_lr (skip)
        arr[2] = victim_fp; // callee.saved_fp -> victim (the pending_deopts KEY)
        arr[3] = pc_in_victim; // callee.saved_lr -> INTO victim (redirect target)
        arr[4] = below_fp; // victim.saved_fp -> below
        arr[5] = call_stub_pc; // victim.saved_lr -> call_stub (ends native walk)

        vm.reg_block.last_compiled_fp = adapter_fp;
        vm.reg_block.last_compiled_pc = pc_in_callee;
        vm.reg_block.last_compiled_kind = crate::codecache::stubs::KIND_ALLOC_SLOW;

        let tramp = 0xDEAD_BEEF_0000u64;
        redirect_returns_into_nm(&mut vm, victim, tramp);

        // Exactly one entry, keyed by the victim's own fp, carrying the ORIGINAL
        // return pc.
        assert_eq!(vm.pending_deopts.len(), 1, "one redirect");
        let pd = vm
            .pending_deopts
            .get(&(victim_fp as usize))
            .expect("keyed by the victim (nm-activation) fp = callee.saved_fp");
        assert_eq!(pd.orig_ret_pc, pc_in_victim as usize, "original return pc");
        assert_eq!(pd.nm, victim, "the NotEntrant nm");

        // ONLY the callee's saved-LR slot was rewritten to the trampoline.
        assert_eq!(arr[3], tramp, "callee saved-LR -> trampoline");
        assert_eq!(arr[1], 0, "adapter saved-LR untouched");
        assert_eq!(arr[2], victim_fp, "callee saved-FP untouched");
        assert_eq!(arr[5], call_stub_pc, "victim's OWN saved-LR untouched");
    }

    /// S13 §2c redirects ONLY reexecute=FALSE Call return sites. A callee whose
    /// saved-LR lands on a reexecute=TRUE site in the victim (an inline Alloc's
    /// `alloc_slow` return) is NOT redirected — the return-path deopt would push
    /// a bogus `incoming_result` onto a re-execute site (`deoptimize_frame`'s own
    /// assert). Same fp-chain as the test above, but the victim's site at 0x10 is
    /// a reexecute=TRUE Alloc → zero redirects.
    #[test]
    fn redirect_skips_reexecute_true_return_site() {
        let mut vm = off_test_vm();
        let victim_code = install_blob(&mut vm, 64);
        let callee_code = install_blob(&mut vm, 64);
        let victim = fake_nmethod(&mut vm, 0x1000, b"victimAlloc", victim_code);
        let _callee = fake_nmethod(&mut vm, 0x2000, b"calleeAlloc", callee_code);
        {
            use crate::compiler::scopes;
            let mut rec = scopes::ScopeDescRecorder::new();
            let scope = rec.begin_scope(scopes::ScopeDescData {
                method_pool_ix: 0,
                is_block: false,
                sender: None,
                receiver: scopes::ValueLoc::Nil,
                slots: vec![],
                ctx: scopes::CtxLoc::None,
            });
            rec.record_site(
                0x10,
                scopes::SafepointState {
                    scope,
                    bci: 0,
                    kind: scopes::SafepointKind::Alloc,
                    reexecute: true, // the Alloc re-execute case — must be skipped
                    stack: vec![],
                },
            );
            let (blob, pcdescs) = rec.pack();
            let nm = vm.code_table.get_mut(victim).unwrap();
            nm.deopt_scopes = blob;
            nm.deopt_pcdescs = pcdescs;
        }
        let pc_in_victim = victim_code.base as u64 + 0x10;
        let pc_in_callee = callee_code.base as u64 + 0x10;
        let call_stub_pc = vm.stubs.call_stub.base as u64;

        let interp_fp = vm.stack.sp;
        for _ in 0..(FRAME_SAVED_FP + 1) {
            vm.stack.push(vm.universe.nil_obj);
        }
        vm.stack.set(
            interp_fp + FRAME_SAVED_FP,
            SmallInt::new(ENTRY_FRAME_SENTINEL).oop(),
        );
        vm.tier_links.push(TierLink::IntoCompiled {
            interp_frame: interp_fp,
            entry_sp: vm.stack.sp as u64,
            nm_id: victim,
        });

        let mut arr = [0u64; 8];
        let base = arr.as_ptr() as u64;
        let adapter_fp = base;
        let callee_fp = base + 16;
        let victim_fp = base + 32;
        let below_fp = base + 48;
        arr[0] = callee_fp;
        arr[1] = 0;
        arr[2] = victim_fp;
        arr[3] = pc_in_victim; // callee.saved_lr -> a reexecute=TRUE site
        arr[4] = below_fp;
        arr[5] = call_stub_pc;

        vm.reg_block.last_compiled_fp = adapter_fp;
        vm.reg_block.last_compiled_pc = pc_in_callee;
        vm.reg_block.last_compiled_kind = crate::codecache::stubs::KIND_ALLOC_SLOW;

        redirect_returns_into_nm(&mut vm, victim, 0xDEAD_BEEF_0000u64);
        assert!(
            vm.pending_deopts.is_empty(),
            "a reexecute=true (Alloc) return site is NOT a valid return-path deopt → skipped"
        );
        assert_eq!(
            arr[3], pc_in_victim,
            "the callee's saved-LR slot is left intact (not redirected)"
        );
    }

    /// Double-invalidation: a re-run over a slot ALREADY pointing at the
    /// trampoline is a no-op — the slot stays, and no second (wrong) entry is
    /// inserted (which would clobber `orig_ret_pc` with the trampoline addr).
    #[test]
    fn redirect_is_idempotent_on_already_redirected_slot() {
        let mut vm = off_test_vm();

        let victim_code = install_blob(&mut vm, 64);
        let callee_code = install_blob(&mut vm, 64);
        let victim = fake_nmethod(&mut vm, 0x1000, b"victim2", victim_code);
        let _callee = fake_nmethod(&mut vm, 0x2000, b"callee2", callee_code);
        let pc_in_callee = callee_code.base as u64 + 0x10;
        let call_stub_pc = vm.stubs.call_stub.base as u64;
        // The REAL trampoline address: a live redirect always holds it, and the
        // walker translates exactly it (via `pending_deopts`) back to the
        // victim's real return pc as it steps past the redirected slot.
        let tramp = vm.stubs.deopt_return_addr();

        let interp_fp = vm.stack.sp;
        for _ in 0..(FRAME_SAVED_FP + 1) {
            vm.stack.push(vm.universe.nil_obj);
        }
        vm.stack.set(
            interp_fp + FRAME_SAVED_FP,
            SmallInt::new(ENTRY_FRAME_SENTINEL).oop(),
        );
        vm.tier_links.push(TierLink::IntoCompiled {
            interp_frame: interp_fp,
            entry_sp: vm.stack.sp as u64,
            nm_id: victim,
        });

        let pc_in_victim = victim_code.base as u64 + 0x10;
        let mut arr = [0u64; 8];
        let base = arr.as_ptr() as u64;
        let victim_fp = base + 32;
        arr[0] = base + 16; // adapter.saved_fp -> callee
        arr[1] = 0;
        arr[2] = victim_fp; // callee.saved_fp -> victim
        arr[3] = tramp; // callee.saved_lr ALREADY the trampoline
        arr[4] = base + 48; // victim.saved_fp
        arr[5] = call_stub_pc; // victim.saved_lr -> call_stub

        // A live redirect ALWAYS has its matching pending_deopts entry (they are
        // inserted together by §2c) — seed it, so the walker can translate the
        // trampoline slot back to the victim's real return pc.
        vm.pending_deopts.insert(
            victim_fp as usize,
            crate::runtime::vm_state::PendingDeopt {
                orig_ret_pc: pc_in_victim as usize,
                nm: victim,
            },
        );

        vm.reg_block.last_compiled_fp = base;
        vm.reg_block.last_compiled_pc = pc_in_callee;
        vm.reg_block.last_compiled_kind = crate::codecache::stubs::KIND_ALLOC_SLOW;

        redirect_returns_into_nm(&mut vm, victim, tramp);

        assert_eq!(
            vm.pending_deopts.len(),
            1,
            "the already-redirected slot must not insert a (wrong) second entry"
        );
        assert_eq!(
            vm.pending_deopts
                .get(&(victim_fp as usize))
                .unwrap()
                .orig_ret_pc,
            pc_in_victim as usize,
            "the pre-existing entry's orig_ret_pc is preserved (not clobbered to the trampoline)"
        );
        assert_eq!(arr[3], tramp, "slot still points at the trampoline");
    }
}
