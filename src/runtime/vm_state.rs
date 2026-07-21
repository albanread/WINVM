//! `VmState` — the one god-struct threaded as `&mut VmState` throughout the
//! VM (CONVENTIONS §2: no statics/globals). S1 gives it a `Universe` and the
//! options env parsing produces; later sprints add process/stack state.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::Instant;

use crate::codecache::nmethod::CodeTable;
use crate::codecache::{CodeCache, DEFAULT_CODE_CACHE_CAPACITY};
use crate::interpreter::stack::{ProcessStack, DEFAULT_STACK_CAPACITY};
use crate::memory::Universe;
use crate::oops::wrappers::MethodOop;
use crate::runtime::lookup::LookupCache;

/// `VmState`'s fixed-offset prefix (S10 D6) — see `oops::layout`'s
/// `VMREG_*_OFFSET` consts, the single source of truth compiled code's
/// direct `[x28, #N]` loads/stores bake in (`vmregblock_offsets_pinned`
/// below asserts the two never drift apart). Only `poll_flag` is live in
/// S10 (the `Poll` Ir op reads it, D5.3); the rest are laid out now so the
/// offsets S11 compiles against are already stable.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VmRegBlock {
    /// S12 step 7 (single-source-of-truth eden): the ADDRESS of
    /// `vm.universe.eden.top` — NOT a copy of its value. Compiled code's
    /// inline-alloc fast path dereferences this to read AND write the one
    /// canonical bump pointer, the same word every Rust-side allocator and
    /// both collectors already use. Replaces S11's `eden_top` VALUE copy +
    /// the publish/adopt sync protocol at `enter_compiled`'s outermost
    /// boundary, which was sound only while the (now-deleted) D8 bridge
    /// froze `eden.top` for the whole compiled window; with real GC
    /// running under compiled frames, ANY value copy goes stale the moment
    /// a nested allocation or collection moves the real pointer — sharing
    /// the one location makes staleness structurally impossible instead of
    /// something every current and future tier-crossing must remember to
    /// prevent. Stable for the whole process lifetime: `Universe::eden` is
    /// boxed (`Box<Eden>`), so this address survives `VmState`/`Universe`
    /// being moved by value. Set ONCE in `with_options`, never
    /// republished.
    pub eden_top_addr: u64,
    /// S11 inline alloc's bounds check. A VALUE copy, but of an immutable
    /// quantity: eden's ceiling is fixed at genesis and never grows or
    /// moves (only `old` grows — `Universe::grow_old`), so unlike the
    /// S11-era `eden_top` copy this can never go stale. Set ONCE in
    /// `with_options`, alongside `old_start`/`card_base_biased` (same
    /// fixed-at-genesis reasoning).
    pub eden_end: u64,
    /// S11 write barrier.
    pub old_start: u64,
    /// S11 write barrier: `cards − (old_start >> 9)`.
    pub card_base_biased: u64,
    /// Nonzero => an interrupt/trace poll is due (S10 `Poll` Ir op reads
    /// this; nothing sets it nonzero yet — mirrors `VmState::pending`'s own
    /// pre-S10 status as a wired-but-inert hook). nonzero comparison, not a
    /// specific sentinel value, so `cbnz` (not a full compare) suffices.
    pub poll_flag: u32,
    _pad: u32,
    /// S11 runtime entries: the anchor frame-pointer/return-pc pair native
    /// stack-walking code reads when re-entering Rust from compiled code.
    pub last_compiled_fp: u64,
    pub last_compiled_pc: u64,
    /// S12 D3: which of the 6 anchor-setting stubs wrote the anchor above —
    /// see `layout::VMREG_LAST_COMPILED_KIND_OFFSET`'s own doc for why
    /// `last_compiled_pc` alone can't answer this. Raw storage for
    /// `runtime::frames::AdapterKind` (that module owns the enum <-> u64
    /// mapping; kept a bare `u64` here rather than the enum itself so this
    /// struct — laid out to match compiled code's fixed-offset `[x28, #N]`
    /// stores — never needs `#[repr(C)]` reasoning about a foreign enum's
    /// own representation).
    pub last_compiled_kind: u64,
}

impl VmRegBlock {
    pub fn new() -> VmRegBlock {
        VmRegBlock {
            eden_top_addr: 0,
            eden_end: 0,
            old_start: 0,
            card_base_biased: 0,
            poll_flag: 0,
            _pad: 0,
            last_compiled_fp: 0,
            last_compiled_pc: 0,
            last_compiled_kind: 0,
        }
    }
}

impl Default for VmRegBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// `MACVM_JIT` (CONVENTIONS §3). `Off` never triggers `compile_method`
/// regardless of invocation counts; `Threshold(n)` compiles a method the
/// moment its invocation counter reaches `n` (SPEC §8.1's
/// `InvocationCounterLimit`) — `n = 1` is the differential-testing gate
/// (tests_s10.md gate item 1): every eligible method compiles on its first
/// send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitMode {
    Off,
    Threshold(u32),
}

impl JitMode {
    /// Provisional (SPRINTS.md standing rule 3: S10 has no perf gate beyond
    /// a 2× tripwire) — plausible enough to exercise real compilation in an
    /// ordinary run without recompiling on every send; not perf-tuned.
    pub const DEFAULT_THRESHOLD: u32 = 1000;
}

/// Global runtime kill-switch for JIT *compilation* — the Cocoa Debug menu's
/// "Compiler (JIT)" toggle. Default on. When off, the method/block compile
/// triggers ([`crate::interpreter::send::activate_method`] and the block
/// trigger) skip compiling NEW code, so anything not already compiled runs in
/// the interpreter; already-Alive nmethods keep running. Deliberately GLOBAL
/// (all VMs) and a plain relaxed `AtomicBool`: a debug switch flipped from the
/// UI (main) thread and read live by each VM's own thread at its next trigger,
/// with no per-VM plumbing and no synchronization beyond atomicity. Per-VM
/// `VmOptions.jit` (Off/Threshold) is unchanged and still applies when this is
/// on.
static JIT_COMPILE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Is JIT compilation currently allowed? (read at each compile trigger)
pub fn jit_compile_enabled() -> bool {
    JIT_COMPILE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
/// Turn JIT compilation on/off at runtime (the Debug-menu toggle).
pub fn set_jit_compile_enabled(on: bool) {
    JIT_COMPILE_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// S21 step 1: how a guest-fatal condition (an uncaught `error:`, DNU with
/// no library `doesNotUnderstand:` installed, process-stack overflow, heap/
/// eden exhaustion, a `#cannotReturn:` invariant violation, a debugger
/// `quit`, or PROBE's own crash dossier) actually terminates. `ExitProcess`
/// is every existing behavior, completely unchanged (`std::process::exit`,
/// as today). `ExitThread` is new, for a `VmHandle`-embedded VM running on
/// its own dedicated worker thread (`docs/SPEC.md` §16, the GUI's own
/// architecture): it terminates ONLY the calling OS thread
/// (`libc::pthread_exit`), never the process — so a bad guest program can
/// never take down whatever embedded it.
///
/// Deliberately a **thread-local**, not a [`VmOptions`] field.
/// `VmOptions { .. }` literals exist at over 100 call sites across this
/// codebase's tests and the CLI; threading a new required field through
/// all of them for something that answers "how should THIS THREAD behave
/// when things go fatally wrong," not "how is THIS VM instance
/// configured," would be both a large mechanical ripple and the wrong
/// shape — the setting is inherently thread-scoped, matching this whole
/// design's own "one VM, one dedicated thread" embedding model exactly.
/// It also uniformly covers the two fatal sites (`memory::alloc`'s eden/
/// heap exhaustion) that have no `VmState` (not even a `&VmState`) in
/// scope at all — there is no parameter to thread a `VmOptions`-derived
/// value through there without a much larger refactor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FatalMode {
    ExitProcess,
    ExitThread,
}

thread_local! {
    static FATAL_MODE: std::cell::Cell<FatalMode> =
        const { std::cell::Cell::new(FatalMode::ExitProcess) };
    /// A hook fired on THIS thread's `ExitThread` fatal exit, immediately
    /// before `pthread_exit`. Because `pthread_exit` runs no `Drop` glue, a
    /// supervisor cannot otherwise learn a `VmHandle` thread has died; this is
    /// the ONE exact death signal. Set by a supervisor (the Cocoa GUI's primary
    /// generation) to post a "died" event to its watchdog, so a GENUINE fatal —
    /// and only a genuine fatal, never a merely-busy VM — triggers a respawn.
    static FATAL_HOOK: std::cell::RefCell<Option<Box<dyn Fn()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Called once, by `VmHandle::boot` (S21 step 2), before any guest code
/// runs on that (newly spawned, dedicated) thread. Never called by the
/// CLI or by any test — so [`fatal_exit`]'s default (`ExitProcess`) is
/// exactly today's behavior everywhere except a `VmHandle`-embedded VM.
pub(crate) fn set_fatal_mode(mode: FatalMode) {
    FATAL_MODE.with(|f| f.set(mode));
}

/// Register (or replace) this thread's [`FATAL_HOOK`] — fired just before an
/// `ExitThread` `pthread_exit`. No-op on the `ExitProcess` path (the process is
/// ending anyway). Thread-scoped exactly like [`set_fatal_mode`].
pub(crate) fn set_fatal_hook(hook: Box<dyn Fn()>) {
    FATAL_HOOK.with(|h| *h.borrow_mut() = Some(hook));
}

/// The shared terminal call for every guest-fatal condition (S21 step 1) —
/// replaces a bare `std::process::exit(code)` at each of those sites.
/// `ExitThread`'s `pthread_exit` deliberately does NOT unwind the Rust call
/// stack at all (no `Drop` impls run, no unwind-table walk happens) — unlike
/// `panic!`/`catch_unwind`, it is sound regardless of whether a JIT-compiled
/// native frame (no Rust unwind info at all) sits underneath it on the
/// stack, which is exactly why this mechanism, not a panic-based one, is
/// used here. Callers on the `ExitThread` path must already have reported
/// the failure (message/stack-trace, written to `vm.out` exactly as today)
/// BEFORE calling this — it never returns.
#[allow(unsafe_code)]
pub(crate) fn fatal_exit(code: i32) -> ! {
    match FATAL_MODE.with(std::cell::Cell::get) {
        FatalMode::ExitProcess => std::process::exit(code),
        FatalMode::ExitThread => {
            // Signal the supervisor BEFORE the thread vanishes — `pthread_exit`
            // below runs no Drop glue, so this hook is its only chance to learn
            // of the death. The failure message/stack-trace was already written
            // to `vm.out` (forwarded to the UI transcript) by the caller.
            FATAL_HOOK.with(|h| {
                if let Some(f) = h.borrow().as_ref() {
                    f();
                }
            });
            // SAFETY: terminates only the calling thread; never touches
            // memory, never returns. `value` (the thread's "exit status"
            // pointer) is unused by anything that would ever join a
            // `pthread_exit`ed thread's `JoinHandle` (S21 step 1a's own
            // finding: a caller must never `.join()`/poll `.is_finished()`
            // on such a handle at all — `JoinHandle::join()`'s internal
            // bookkeeping expects the SPAWNED thread's own normal-or-
            // panicking completion path to run, which `pthread_exit`
            // skips, so `join()` panics and `is_finished()` never becomes
            // true; dropping an unjoined handle is safe, per the same
            // experiment).
            #[cfg(unix)]
            unsafe {
                libc::pthread_exit(std::ptr::null_mut())
            }
            // WINVM: ExitThread is the exact analogue — terminates only the
            // calling thread, runs no Rust Drop glue, never returns. The
            // same never-join contract on the spawner applies.
            #[cfg(windows)]
            {
                extern "system" {
                    fn ExitThread(code: u32) -> !;
                }
                // SAFETY: terminates only the calling thread; never returns.
                unsafe { ExitThread(code as u32) }
            }
        }
    }
}

/// S10 D6: a side-channel record of one tier transition, pushed/popped
/// around `interpreter::compiled_call::enter_compiled` — `vm.stack` itself
/// never grows for a compiled call (S10 compiled code pushes no frame of
/// its own, D1), so this `Vec` is the only record that a compiled
/// activation is (or was) live between two interpreted frames; a
/// mixed-tier stack-trace walker consults it to know where to print a
/// compiled frame's line.
#[derive(Clone, Copy, Debug)]
pub enum TierLink {
    /// I→C: pushed by `enter_compiled` before invoking the call stub.
    /// `interp_frame` is the interpreted caller's own `Frame.fp` (compiled
    /// code runs "underneath" it, not as a new `vm.stack` frame);
    /// `entry_sp` is `vm.stack.sp` at the moment of entry, for a stack
    /// walker to anchor against and for the debug assertion that a
    /// completed (non-bailout) compiled call left `vm.stack.sp` exactly
    /// where a primitive's direct return would have. `nm_id` is which
    /// nmethod is running "underneath" — the only way a mixed-tier stack
    /// trace can name it (`key_klass`/`key_selector`) and locate its own
    /// `poll_bci`, since nothing else records a compiled activation at all.
    /// S14 step 9: a synthetic crossing pushed around a deopt's
    /// `interpret_active` nested run — pairs with the materialized region's
    /// own `ENTRY_FRAME_SENTINEL` so a walk can bridge PAST the abandoned
    /// compiled frame (`dead_fp`, skipped entirely) onto its caller chain,
    /// classified at `caller_ret_pc` (the dead frame's §2c-translated saved
    /// return pc — a real safepoint in the caller). See
    /// `frames::deopt_bridge_link`.
    DeoptBridge { dead_fp: u64, caller_ret_pc: u64 },
    IntoCompiled {
        interp_frame: usize,
        entry_sp: u64,
        nm_id: crate::codecache::nmethod::NmethodId,
    },
    /// C→I: compiled code calling back into the interpreter (a runtime
    /// send, an allocation slow path). S10 never constructs this — D1's
    /// eligibility rules out every op that could need it — S11's own
    /// `CallSend`/`CallRuntime`/`Alloc` Ir ops are what will.
    IntoInterpreter {
        compiled_fp: u64,
        compiled_ret_pc: u64,
    },
}

/// S13 D1 §2c: one in-flight activation of a now-`NotEntrant` nmethod whose
/// caller's (i.e. this activation's own callee's) saved-LR stack slot has been
/// redirected to `deopt_return_trampoline` — so when that callee `ret`s, the
/// trampoline runs a lazy return-path deopt of THIS activation instead of
/// resuming its old compiled code (`sprint_s13_detail.md` D6).
///
/// Keyed in [`VmState::pending_deopts`] by the redirected activation's OWN FP
/// (the callee's `saved_fp = [callee_fp]`, which is the nm activation's frame
/// pointer). At trampoline entry, that FP is live in `x29` (the callee's
/// epilogue `ldp` restored fp to its caller = this activation), so
/// `rt_deopt_on_return` looks the entry up by it.
#[derive(Clone, Copy, Debug)]
pub struct PendingDeopt {
    /// The ORIGINAL return address into the nm activation the redirected
    /// saved-LR slot held before being overwritten — the pc whose `PcDesc`
    /// names the call-return deopt site (`reexecute == false` by construction).
    pub orig_ret_pc: usize,
    /// The `NotEntrant` nmethod this activation is running (the deopt scope
    /// chain + oop pool live here, GC-current).
    pub nm: crate::codecache::nmethod::NmethodId,
}

/// S11 D6.3: a non-local return in flight across compiled frames — the
/// `home` it targets and the `value` to deliver there, parked in
/// `VmState::nlr_state` while the `NLR_SENTINEL` propagates back through the
/// compiled frame(s) between the escaping block and its home. `home` is a
/// plain `HomeRef` (proc/serial/fp — an interpreter-frame coordinate, S4);
/// `value` is a bare `Oop` held across NO allocation (the escape parks it
/// and the resume delivers it, with only frame pops in between), so it needs
/// no handle.
#[derive(Clone, Copy, Debug)]
pub struct NlrState {
    pub home: crate::oops::home_ref::HomeRef,
    pub value: crate::oops::Oop,
    /// S24 A1: the ORIGINATING closure, parked only by compiled-block NLR
    /// origination (`codecache::stubs::rt_nlr_originate`) — interpreted
    /// `nlr_tos` leaves this `None` and behaves exactly as before. Carried
    /// so [`crate::interpreter::unwind::continue_unwind`]'s dead-home arm
    /// can deliver `#cannotReturn:` to the REAL closure: with compiled
    /// origination the block's native frame is gone by the time the
    /// liveness check runs, so `cannot_return_current_closure`'s
    /// read-the-current-frame's-receiver-slot strategy would find the
    /// value:-SENDER's receiver instead (adversarial-review BLOCKER §2 — a
    /// release `.expect` abort on legal guest code). Same
    /// no-allocation-in-transit contract as `value`; not a GC root.
    pub closure: Option<crate::oops::Oop>,
}

/// S13 deopt counters (`sprint_s13_detail.md` D5 M1). Its own small struct
/// (reached as `vm.stats.deopt_count`) rather than a bare `VmState` field, so
/// later deopt steps (stress counting kept separate from the
/// `UncommonTrapLimit` tally, D7) have a home without reshaping `VmState`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct VmStats {
    /// Compiled frames materialized back into interpreter frames — bumped
    /// once per `deoptimize_frame` call (D5 M1).
    pub deopt_count: u64,
    /// S13 D7: deopts split by trigger, indexed by [`DeoptReason`]
    /// (`[trap, return, poll]`). Attributed at the three runtime deopt entry
    /// points (`rt_uncommon_trap`/`rt_deopt_on_return`/`rt_poll`) via
    /// [`VmState::note_deopt`], so `deopt_count == sum(deopt_by_reason)` for a
    /// real run (a test that calls `deoptimize_frame` directly bumps only
    /// `deopt_count`).
    pub deopt_by_reason: [u64; 3],
    /// S14 step 8: recompilations performed by the recompile-on-trap loop
    /// (`runtime::recompile`) — a trapped-out nmethod replaced by a fresh
    /// compile against its now-warm feedback.
    pub recompiles: u64,
    /// S14 step 8 (A5): recompiles DECLINED because the feedback profile was
    /// unchanged since compile — recompiling would emit identical decisions
    /// (Self's `checkEffectiveness`; the anti-thrash gate for storms that
    /// recompilation cannot fix, e.g. a persistent smi-overflow trap).
    pub recompile_declined_ineffective: u64,
    /// S15 A8 (dispatch overhead): compiled-IC misses — every arrival at
    /// `rt_resolve_send` (a send whose site was Unresolved, or whose
    /// mono/pic guard rejected the receiver). Hits are NOT counted (they
    /// never leave compiled code — the doc's "slow paths only" rule).
    pub ic_misses: u64,
    /// S15 A8: PIC growths (a Pic-state site adding a klass pair).
    pub pic_extends: u64,
    /// S24 A1 hardening: `rt_resolve_send` left a send site unpatched
    /// because the code cache could not fit the PIC/mega stub the
    /// transition wanted (the graceful-degradation path — see the `patch`
    /// binding's doc in `rt_resolve_send`). A nonzero value means the
    /// cache is full: dispatch stays correct but poly sites resolve
    /// through the runtime on every miss.
    pub ic_patch_declined_cache_full: u64,
    /// S15 A8: Pic→Mega promotions (site went megamorphic).
    pub mega_transitions: u64,
    /// S24 B-phase (stale-PIC-c2i fix): a Pic-state site's pair whose target
    /// was a c2i adapter (baked before the callee compiled) was re-keyed to
    /// the freshly compiled nmethod by `rt_interpret_call`'s upgrade hook.
    /// Before this existed, such a pair dispatched its klass through the
    /// interpreter for the caller's whole lifetime — deltablue's t=200 tail
    /// (ScaleConstraint>>execute, 39.8% of interpreted bytecodes) was exactly
    /// this. A PIC guard HIT never re-enters `rt_resolve_send`, so the c2i
    /// crossing itself is the only place the staleness is observable.
    pub c2i_pic_rekeys: u64,
    /// S24 A2: a compiled `value`-family send site's block-dispatch stub
    /// tail-jumped straight to a compiled block nmethod (the fast path —
    /// `rt_value_target` found an Alive `by_block` entry, no interpreter
    /// crossing). The headline metric that A2 is firing at all: it stays 0
    /// until a compiled method contains a `value`-family send whose block
    /// is itself compiled (which, pre-A3, is rare — most `value:` sites live
    /// in still-interpreted closure creators).
    pub value_dispatch_hits: u64,
    /// S24 A2: a `value`-family dispatch stub fell back to interpreting
    /// `value:` (`rt_value_fallback`) — block not yet compiled, or a
    /// non-closure/wrong-argc receiver. The fallback's nested interpretation
    /// still enters an already-compiled block body (via A1's `activate_block`
    /// hook), so this is not a "slow" path per se, just "not the tail-jump".
    pub value_dispatch_fallbacks: u64,
    /// S15 A8 (tier balance): successful tier-1 compilations installed
    /// (all levels — `Nmethod.level` is 1 until the policy ladder lands).
    pub compilations: u64,
    /// S24 L2 step 2 (trigger unification): installs where saturating the
    /// method's invocation counter to the threshold actually RAISED it —
    /// i.e. compiles earned by the loop/backedge counter (OSR) or the
    /// version ladder, whose methods' future CALLS would previously have
    /// interpreted end-to-end until the invocation counter crossed on its
    /// own. "The loop counters have detected in a different way that the
    /// method containing the loop is hot; the method is now hot."
    pub trigger_unifications: u64,
    /// S15: OSR transitions actually taken (interpreter frame replaced by
    /// a compiled frame mid-loop).
    pub osr_entries: u64,
    /// S15: OSR requests declined (version cap / effectiveness / not
    /// compilable) — the loop kept interpreting, counter reset.
    pub osr_declined: u64,
    /// S24 L2 phase B: OSR compiles declined because the has_ctx method is
    /// the ELIDED form (`all_elidable`) — the one adoption-unsound case
    /// (the interpreted prefix may have leaked a real closure over the
    /// frame Context that elided vreg writes would split from). This is
    /// the evidence counter for ever revisiting write-through (design R1).
    pub osr_declined_elided_ctx: u64,
    /// S24 L2 phase B: OSR transfers that ADOPTED the interpreter frame's
    /// heap Context into the materialize-form nmethod's ctx vreg (identity
    /// preserved — pre-OSR closures and post-OSR compiled code share the
    /// ONE Context).
    pub osr_ctx_adopted: u64,
    /// S24 B4: spliced blocks whose body contained a `^` (nlr_tos lowered
    /// to `Ir::Ret` of the home compilation) — counted per COMPILE, not per
    /// execution. The per-gate "did the NLR block actually splice" signal;
    /// nonzero proves the B4 gate relaxation is firing.
    pub blocks_spliced_nlr: u64,
    /// S24 B5: multi-BB block bodies grafted (per compile) — nonzero proves
    /// the CFG graft is firing for conditional-NLR / branchy block shapes.
    pub blocks_spliced_multibb: u64,
    /// S24 B5 step 4: method-inline decisions demoted to plain sends by the
    /// cumulative total_bytes budget.
    pub splice_declined_budget: u64,
}

/// The `MACVM_TRACE=stats` dump (`main.rs::print_vm_stats`) and RUSTTCL's
/// `stats` verb both want the identical counter listing — one on process
/// exit (gated on the trace flag, to stderr), the other on demand from an
/// interactive/scripted shell (unconditional, returned as a `Value`) — so
/// this is the one place the line list is written, `[stats] `-prefixed to
/// match the trace channel's existing look either way.
pub fn format_vm_stats(vm: &VmState) -> String {
    let s = &vm.stats;
    let g = &vm.universe.gc_stats;
    let mut code_alive = 0usize;
    let mut code_zombie = 0usize;
    for nm in vm.code_table.iter_all() {
        match nm.state {
            crate::codecache::nmethod::NmState::Alive => code_alive += nm.code.len,
            _ => code_zombie += nm.code.len,
        }
    }
    [
        format!("[stats] ic_misses={}", s.ic_misses),
        format!("[stats] pic_extends={}", s.pic_extends),
        format!("[stats] mega_transitions={}", s.mega_transitions),
        format!("[stats] c2i_pic_rekeys={}", s.c2i_pic_rekeys),
        format!("[stats] trigger_unifications={}", s.trigger_unifications),
        format!(
            "[stats] ic_patch_declined_cache_full={}",
            s.ic_patch_declined_cache_full
        ),
        format!(
            "[stats] value_dispatch_hits={} value_dispatch_fallbacks={}",
            s.value_dispatch_hits, s.value_dispatch_fallbacks
        ),
        format!("[stats] compilations={}", s.compilations),
        format!("[stats] recompiles={}", s.recompiles),
        format!(
            "[stats] recompile_declined_ineffective={}",
            s.recompile_declined_ineffective
        ),
        format!(
            "[stats] deopt_count={} by_reason=[trap {}, return {}, poll {}]",
            s.deopt_count, s.deopt_by_reason[0], s.deopt_by_reason[1], s.deopt_by_reason[2]
        ),
        format!("[stats] osr_entries={}", s.osr_entries),
        format!("[stats] osr_declined={}", s.osr_declined),
        format!(
            "[stats] osr_declined_elided_ctx={}",
            s.osr_declined_elided_ctx
        ),
        format!("[stats] osr_ctx_adopted={}", s.osr_ctx_adopted),
        format!("[stats] blocks_spliced_nlr={}", s.blocks_spliced_nlr),
        format!(
            "[stats] blocks_spliced_multibb={}",
            s.blocks_spliced_multibb
        ),
        format!(
            "[stats] splice_declined_budget={}",
            s.splice_declined_budget
        ),
        format!("[stats] scavenge_count={}", g.scavenge_count),
        format!(
            "[stats] scavenge_us_total={} scavenge_us_max={}",
            g.total_scavenge_pause.as_micros(),
            g.scavenge_pause_max.as_micros()
        ),
        format!("[stats] full_gc_count={}", g.full_gc_count),
        format!(
            "[stats] full_gc_us_total={} full_gc_us_max={}",
            g.full_pause_total.as_micros(),
            g.full_pause_max.as_micros()
        ),
        format!("[stats] bytes_allocated={}", g.bytes_allocated),
        format!("[stats] bytes_promoted={}", g.bytes_promoted),
        format!("[stats] contexts_allocated={}", g.context_allocs),
        format!("[stats] code_bytes_alive={code_alive}"),
        format!("[stats] code_bytes_zombie={code_zombie}"),
        {
            // C2 (docs/cocoa_bridge_design.md §8): the DNU shape-cache
            // hit rate. Process-wide (the cache is shared by every VM in
            // the process), reported here so `__vmStats` shows it.
            // WINVM: no Cocoa bridge — report zeros so the stats shape
            // (and anything parsing it) stays identical across platforms.
            #[cfg(target_os = "macos")]
            let (hits, misses) = crate::runtime::objc_bridge::shape_stats();
            #[cfg(not(target_os = "macos"))]
            let (hits, misses) = (0u64, 0u64);
            format!("[stats] cocoa_shape_hits={hits} cocoa_shape_misses={misses}")
        },
    ]
    .join("\n")
}

/// S13 D7: which trigger fired a deopt — for `deopt_by_reason` attribution and
/// the `MACVM_TRACE=deopt` channel. The discriminant IS the `deopt_by_reason`
/// index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeoptReason {
    /// An uncommon trap (`brk`): smi overflow or `mustBeBoolean` (step 7).
    Trap = 0,
    /// A lazy return-address redirection into a `NotEntrant` frame (step 9).
    Return = 1,
    /// A loop back-edge poll of a `NotEntrant` frame (step 10b §2d).
    Poll = 2,
}

/// DBG0 (docs/DEBUGGER.md §4.2 step 9): one recent-history event for the
/// PROBE crash dossier's ring buffer. `Copy`, and deliberately carries NO
/// oops (nmethod ids only — the ring is not a GC root; names are resolved
/// best-effort at dump time). The frozen crash instant tells you *where*;
/// the ring tells you *what just happened* — on BUG D it was the deopt
/// trace tail immediately before the failure that localized the culprit
/// nmethod.
#[derive(Clone, Copy, Debug)]
pub enum ProbeEvent {
    /// A fresh nmethod was published (driver install).
    Compile { nm: u32, version: u8 },
    /// A frame deoptimized (materializer M1) at `bci` of nmethod `nm`.
    Deopt { nm: u32, bci: u32, reexecute: bool },
    /// An nmethod was made NotEntrant (redefinition/recompile/breakpoint).
    Invalidate { nm: u32 },
}

/// Fixed-size overwrite-oldest ring of [`ProbeEvent`]s. Feeds: nmethod
/// install (driver), `deoptimize_frame` M1, `make_not_entrant{,_lazy}`.
/// Cost when nothing ever dumps it: one slot store per (already-rare)
/// compile/deopt/invalidate event.
pub struct ProbeRing {
    buf: Vec<Option<ProbeEvent>>,
    next: usize,
    /// Total events ever pushed (so the dump can say "showing last N of M").
    pub total: u64,
}

impl ProbeRing {
    const CAP: usize = 256;

    pub fn new() -> ProbeRing {
        ProbeRing {
            buf: vec![None; Self::CAP],
            next: 0,
            total: 0,
        }
    }

    pub fn push(&mut self, e: ProbeEvent) {
        self.buf[self.next] = Some(e);
        self.next = (self.next + 1) % Self::CAP;
        self.total += 1;
    }

    /// Oldest-first iteration over whatever is retained.
    pub fn iter_oldest_first(&self) -> impl Iterator<Item = ProbeEvent> + '_ {
        let n = self.next;
        self.buf[n..]
            .iter()
            .chain(self.buf[..n].iter())
            .filter_map(|e| *e)
    }
}

impl Default for ProbeRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Which `MACVM_TRACE` channels are enabled. The channel set is open-ended
/// (CONVENTIONS §3); S1 only stores membership, nothing reads it yet.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceFlags {
    channels: HashSet<String>,
}

impl TraceFlags {
    pub fn parse(s: &str) -> TraceFlags {
        TraceFlags {
            channels: s
                .split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect(),
        }
    }

    pub fn is_enabled(&self, channel: &str) -> bool {
        self.channels.contains(channel)
    }

    /// RUSTTCL's `trace` verb: flip a channel on/off live, past the
    /// `MACVM_TRACE` env-var parse at startup (`is_enabled` is read on
    /// every dispatch/deopt/etc., so this takes effect on the very next
    /// check — no restart needed).
    pub fn enable(&mut self, channel: &str) {
        self.channels.insert(channel.to_string());
    }

    pub fn disable(&mut self, channel: &str) {
        self.channels.remove(channel);
    }

    /// Sorted for stable, diffable `trace` verb output.
    pub fn list(&self) -> Vec<String> {
        let mut v: Vec<String> = self.channels.iter().cloned().collect();
        v.sort();
        v
    }
}

/// A `Write` sink backed by a shared `Vec<u8>` — what tests substitute for
/// `VmState::out` (stdout by default) so transcript assertions work without
/// a subprocess, before S5's golden-transcript runner exists. `Clone`s share
/// the same buffer (an `Arc<Mutex<_>>`), so a test can install one half via
/// `vm.out = Box::new(buf.clone())` and read the other after running.
#[derive(Clone, Default)]
pub struct OutputBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl OutputBuffer {
    pub fn new() -> OutputBuffer {
        OutputBuffer::default()
    }

    pub fn contents(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }

    pub fn as_string(&self) -> String {
        String::from_utf8_lossy(&self.contents()).into_owned()
    }
}

impl Write for OutputBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct VmOptions {
    /// Address-space reservation size, in MiB. Default 8192 (SPEC §7.1's
    /// tunable default), overridden by `MACVM_HEAP`.
    pub heap_mib: usize,
    pub trace: TraceFlags,
    /// `MACVM_GC_STRESS=1` (S7-10, SPEC §7.2 A2 step 1): scavenge before
    /// every allocation through the public choke point (`memory::alloc::
    /// alloc_words`), not just on eden exhaustion — the harness that flushes
    /// out invisible-root bugs (S7-9's `Handle` discipline) deterministically
    /// instead of waiting for an unlucky eden-boundary allocation.
    pub gc_stress: bool,
    /// `MACVM_GC_STRESS=full[:N]` (S8 step 8, SPEC §12.3 item 4): run a full
    /// mark-slide-compact collection every `N` allocations through the same
    /// choke point (default `N` = 100 — `Some(100)` when the env var is
    /// exactly `"full"` with no suffix). Mutually exclusive with `gc_stress`
    /// (`=1` scavenges every allocation; `=full` full-GCs every Nth) — the
    /// harness that flushes out the reference-rewrite class of bug
    /// `for_each_root`'s S8-8 fix closed (found by running real source, not
    /// by the scavenger-only `=1` mode, which never calls `full_gc` at all).
    pub gc_stress_full_period: Option<u64>,
    /// `MACVM_EDEN=<KiB>` (S7 A1 step 3's "tests" override): eden size in
    /// KiB, overriding `layout::DEFAULT_EDEN_SIZE`. `None` keeps the
    /// default — used to shrink eden in tests so a scavenge is reachable
    /// without allocating megabytes of filler first.
    pub eden_kb: Option<usize>,
    /// `MACVM_JIT=off|threshold=N` (S10, CONVENTIONS §3). Unset => `Off` —
    /// deliberately NOT the SPEC-intent "on by default" reading: hundreds
    /// of pre-S10 tests were written and verified against pure-interpreter
    /// behavior, and `tests/common/mod.rs`'s `test_vm()` starts from
    /// `VmOptions::from_env()` precisely so a gate script's env var (same
    /// pattern as `MACVM_GC_STRESS`) reaches every in-process test VM —
    /// defaulting this on would silently let ambient shell state change
    /// unrelated test behavior. `just gate-s10` opts in explicitly.
    pub jit: JitMode,
}

impl VmOptions {
    pub const DEFAULT_HEAP_MIB: usize = 8192;
    /// `MACVM_GC_STRESS=full` with no `:N` suffix — SPEC §12.3's own
    /// tunable default period.
    pub const DEFAULT_FULL_STRESS_PERIOD: u64 = 100;

    /// Pure parse of `MACVM_GC_STRESS`'s raw value into the mutually-
    /// exclusive `(gc_stress, gc_stress_full_period)` pair — split out from
    /// `from_env` so it's unit-testable without touching the real
    /// environment (`tests/common/mod.rs`'s own doc comment: env mutation
    /// races across the multi-threaded test runner, so tests only ever
    /// READ it). `None` (the var unset) yields `(false, None)`: stress off.
    /// `pub(crate)` also for RUSTTCL's `flag` verb — same grammar, live.
    pub(crate) fn parse_gc_stress(raw: Option<&str>) -> (bool, Option<u64>) {
        let Some(s) = raw.map(str::trim) else {
            return (false, None);
        };
        if s == "1" {
            return (true, None);
        }
        let period = s.strip_prefix("full").and_then(|rest| match rest {
            "" => Some(Self::DEFAULT_FULL_STRESS_PERIOD),
            _ => rest.strip_prefix(':').and_then(|n| n.parse::<u64>().ok()),
        });
        (false, period)
    }

    /// Pure parse of `MACVM_JIT`'s raw value (same testability rationale as
    /// `parse_gc_stress`). Unset => `Off` (see `jit`'s own doc comment for
    /// why). An unrecognized value warns and falls back to `Off` too — a
    /// typo'd flag must never silently turn compilation on.
    /// `pub(crate)` also for RUSTTCL's `flag` verb — same grammar, live.
    pub(crate) fn parse_jit(raw: Option<&str>) -> JitMode {
        let Some(s) = raw.map(str::trim) else {
            return JitMode::Off;
        };
        if s == "off" {
            return JitMode::Off;
        }
        if let Some(n) = s
            .strip_prefix("threshold=")
            .and_then(|n| n.parse::<u32>().ok())
        {
            return JitMode::Threshold(n);
        }
        eprintln!("MACVM_JIT: unrecognized value {s:?}, defaulting to off");
        JitMode::Off
    }

    pub fn from_env() -> VmOptions {
        let heap_mib = std::env::var("MACVM_HEAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(Self::DEFAULT_HEAP_MIB);
        let trace = std::env::var("MACVM_TRACE")
            .ok()
            .map(|s| TraceFlags::parse(&s))
            .unwrap_or_default();
        let raw_gc_stress = std::env::var("MACVM_GC_STRESS").ok();
        let (gc_stress, gc_stress_full_period) = Self::parse_gc_stress(raw_gc_stress.as_deref());
        let eden_kb = std::env::var("MACVM_EDEN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let jit = Self::parse_jit(std::env::var("MACVM_JIT").ok().as_deref());
        VmOptions {
            heap_mib,
            trace,
            gc_stress,
            gc_stress_full_period,
            eden_kb,
            jit,
        }
    }
}

impl Default for VmOptions {
    fn default() -> Self {
        VmOptions {
            heap_mib: Self::DEFAULT_HEAP_MIB,
            trace: TraceFlags::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: JitMode::Off,
        }
    }
}

/// The currently executing activation's bytecode position and method,
/// mirrored into `VmState` (S3, SPEC §5.3 design) so `interpreter::send`,
/// `runtime::lookup`, and `runtime::error` can read/write it without the
/// dispatch loop threading `method`/`bci` through every call. `fp`/`sp`
/// deliberately stay OUT of this struct and live only on `VmState::stack`
/// (`ProcessStack::fp`/`sp`) — mirroring them here would be a second source
/// of truth for the same state. `method` is `None` only before the first
/// activation of a freshly booted `VmState`; every send/DNU helper that
/// reads it is called while dispatch is active, so `.expect(...)` at the
/// read site documents that invariant.
#[derive(Copy, Clone, Default, Debug)]
pub struct InterpRegs {
    pub bci: usize,
    pub method: Option<MethodOop>,
}

/// `#[repr(C)]` (S10 D6): `reg_block` MUST be the first field, at
/// `VmState`'s own byte offset 0, so x28 (established `&VmState` by the
/// call stub) reaches its fields directly — see `VmRegBlock`'s own doc.
/// Every other field keeps ordinary (unspecified) layout; only the prefix
/// is pinned.
#[repr(C)]
pub struct VmState {
    pub reg_block: VmRegBlock,
    pub universe: Universe,
    pub options: VmOptions,
    pub stack: ProcessStack,
    /// The active send/DNU's view of "where execution currently is" (SPEC
    /// §5.3, S3). Written by the dispatch loop immediately before every
    /// `send`/`send_super`; read by `interpreter::send`'s super-lookup
    /// (`regs.method.holder()`) and by `runtime::error::print_stack_trace`
    /// for the top frame's `@<bci>`.
    pub regs: InterpRegs,
    /// The global IC dependency-versioning counter (SPEC §6.2, coarsened —
    /// see `sprint_s03_detail.md`'s SPEC-QUESTION). Bumped by every
    /// `install_method`; an IC entry is current iff its stamped epoch
    /// equals this. `debug_assert!(ic_epoch < 1 << 24)` at the bump site —
    /// wraparound is an accepted, unreachable-in-v1 ABA risk.
    pub ic_epoch: u32,
    pub lookup_cache: LookupCache,
    /// `sp - argc - 1` at the moment a primitive was entered — lets
    /// `VmState::prim_arg` re-read a live stack slot instead of the
    /// `[Oop; 6]` copy handed to the `PrimFn`, which primitives with
    /// `can_allocate = true` must do after their first allocating call
    /// (SPEC §10, the S7 choke-point pattern).
    pub prim_arg_base: usize,
    /// Where guest-visible output goes (`printOnStdout:`, S3's dev-hook
    /// primitive group). Stdout by default; tests substitute a `Vec<u8>` so
    /// transcript assertions don't need a subprocess before S5's golden
    /// runner exists.
    pub out: Box<dyn Write>,
    /// Where game-primitive commands go (`docs/gamepane_design.md` M3): the
    /// game analogue of `out`. `None` by default (headless/CLI); the GUI
    /// installs a sink that hands `GameCommand`s across the worker-to-main
    /// channel to the native Metal pane. The core VM only defines the
    /// vocabulary, never touches Metal.
    pub game_sink: Option<Box<dyn crate::embed::GameSink>>,
    /// Multi-Smalltalk-worker role + channels (docs/multi-smalltalk-worker.md
    /// §3, workers M1): `Primary` on a VM that registered a worker boot fn,
    /// `Worker` inside each spawned VM, `None` everywhere else — every worker
    /// primitive fails cleanly when this is `None`, so the world class is
    /// harmless in any embedding (the game-sink posture).
    pub workers: Option<Box<crate::runtime::workers::WorkerState>>,
    /// Cocoa C4 (`poolDo:`): the in-heap mint-list stack — each entry is a
    /// live Array oop (`[count(smi), wrapper...]`) the bridge appends every
    /// freshly minted ObjcRef to while a pool scope is open. IN-HEAP and
    /// rooted (memory/roots.rs) by design: a Rust-side Vec of wrapper oops
    /// would be the BUG-A stale-oop-across-allocation class all over again.
    pub cocoa_mint_stack: Vec<crate::oops::Oop>,
    /// Set by the `quit`/`quit:` primitive; the dispatch loop exits (after
    /// the current activation returns) once this is true.
    pub exit_requested: bool,
    /// The process exit code `quit:` (S6) requested, if any — `None` means
    /// `quit` (0-arg) or no `quit` at all, both of which exit 0. Read by
    /// `main.rs`'s `cmd_run` after `world::load_file` returns.
    pub exit_code: Option<i32>,
    /// VM boot time, for the `millisecondClock` primitive.
    pub start_instant: Instant,
    /// Total bytecodes dispatched, counted only while `MACVM_TRACE=count` is
    /// enabled (S6 PERF procedure: cost is one `add` per dispatch,
    /// unconditionally compiled in — CONVENTIONS §3's channel list). Printed
    /// to stderr at process exit by `main.rs`.
    pub bytecode_count: u64,
    /// S24 A1 (closure-compilation design §4 gate 2, adopted per review
    /// §6): per-METHOD share of `bytecode_count`, so the interpreted TAIL
    /// of a warm run is attributable by arithmetic — which methods/blocks
    /// still interpret, and how much each contributes — rather than by
    /// qualitative census. Keyed by the resolved `Holder>>selector` LABEL
    /// (not the method oop's bits): labels are resolved while the method
    /// is provably live (at run start, holding the very oop the dispatch
    /// loop is executing), so a GC move mid-run merely re-resolves the
    /// same label and the counts merge — raw-bits keying would silently
    /// mis-attribute after an address recycle (task #93's lesson).
    /// Populated only under `MACVM_TRACE=count`; dumped sorted by
    /// `main.rs` at exit.
    pub count_by_method: std::collections::HashMap<String, u64>,
    /// The attribution run-length cache: `(method bits, label, run)` for
    /// the method the dispatch loop is CURRENTLY inside. The hot-path cost
    /// of attribution is one u64 compare + increment per bytecode; the
    /// HashMap and label resolution are only touched when the executing
    /// method CHANGES (≈ per send/return), in `count_switch_method`.
    pub count_cur: Option<(u64, String, u64)>,
    /// S13 deopt counters (`sprint_s13_detail.md` D5 M1). Bumped by
    /// `runtime::deopt::deoptimize_frame` every time a compiled frame is
    /// materialized back into interpreter frame(s); read by tests and (later)
    /// the `MACVM_DEOPT_STRESS` differential harness.
    pub stats: VmStats,
    /// S13 D7 (`MACVM_DEOPT_STRESS`) periodic-invalidation state. `deopt_stress`
    /// gates behavior 2 entirely; `stress_period` is the number of compiled
    /// entries between forced invalidations; `stress_countdown` the live counter
    /// (`enter_compiled` decrements it); `stress_rr_cursor` round-robins the
    /// victim over alive nmethods. Parsed from `MACVM_DEOPT_STRESS` in
    /// `with_options` (see [`VmState::parse_deopt_stress`]); off by default.
    pub deopt_stress: bool,
    pub stress_period: u64,
    pub stress_countdown: u64,
    pub stress_rr_cursor: usize,
    /// The next frame serial to hand out (SPEC §5.4, S4) — monotonic,
    /// post-incremented at every `push_frame`. Never reused, which is what
    /// makes a `HomeRef`'s `(fp, serial)` pair a reliable dead-home check:
    /// a frame popped and later replaced by a different activation at the
    /// same `fp` always has a different serial. u32 wrap after 4G pushes is
    /// an accepted, `debug_assert`-guarded risk.
    next_frame_serial: u32,
    /// `MACVM_DBG_OOP=<hex-addr>` (S7, lesson 11): the single-address trace
    /// hook, updated to the object's new address every time it moves.
    /// `None` unless set by [`VmState::new`] (which reads the env var);
    /// `with_options` always starts `None` — tests set it directly if they
    /// need it, avoiding the env-var-race concern `MACVM_HEAP`/
    /// `MACVM_TRACE` are parsed once for in `VmOptions::from_env`, not here.
    pub dbg_oop: Option<usize>,
    /// DBG0: the PROBE dossier's recent-history ring (docs/DEBUGGER.md §4.2
    /// step 9). Holds no oops — see [`ProbeEvent`].
    pub probe_ring: ProbeRing,
    /// DBG1: HALT's shared core (docs/DEBUGGER.md §2). Holds no oops.
    pub debug: crate::runtime::debug::DebugState,
    /// Scoped GC roots for oops held across an allocating call (S7-9, SPEC
    /// §7.6) — see `memory::handles`. Boxed so its address is stable across
    /// this struct moving; `HandleScope` points directly at the box's
    /// pointee.
    pub handle_arena: Box<crate::memory::handles::HandleArena>,
    /// The one `MAP_JIT` region tier-1 compiled methods and their stubs
    /// live in (S10 D6). Reserved unconditionally in [`VmState::with_options`]
    /// regardless of `options.jit` — one `mmap`, nothing written until the
    /// first real `publish`.
    pub code_cache: CodeCache,
    /// Installed nmethods, keyed by `(receiver klass, selector)` (S10 D6,
    /// D8). Empty for the lifetime of a `JitMode::Off` run.
    pub code_table: CodeTable,
    /// c2i adapters (S11 D6.1), one per interpreted `MethodOop` a compiled
    /// call site has resolved to. Empty for the lifetime of a
    /// `JitMode::Off` run, same as `code_table`.
    pub adapters: crate::codecache::adapters::AdapterTable,
    /// Polymorphic inline caches (S11 D4.3), one per compiled call site
    /// that outgrew `Mono`. Empty for the lifetime of a `JitMode::Off`
    /// run, same as `code_table`.
    pub pic_table: crate::codecache::pics::PicTable,
    /// Megamorphic trampolines (S11 D4.4), one per selector shared across
    /// every call site that outgrew its own PIC. Empty for the lifetime
    /// of a `JitMode::Off` run, same as `code_table`.
    pub mega_table: crate::codecache::mega::MegaTable,
    /// `call_stub`/`stub_poll` (S10 D4/D5.6) — published into `code_cache`
    /// once, here, at boot. `compile_method` needs `stub_poll_addr()` to
    /// embed as a pool constant in any method that emits `Ir::Poll`;
    /// `enter_compiled` (S10 step 8) needs `call_stub_entry()`.
    pub stubs: crate::codecache::stubs::Stubs,
    /// S20 step 2/4: the 3 shape-keyed FFI native-call trampolines
    /// (`ret_g`/`ret_f`/`ret_v` — `codecache::ffi_stubs::FfiStubs`),
    /// published into `code_cache` once, here, at boot — the exact same
    /// unconditional-install shape as `stubs` just above (FFI methods run
    /// interpreted only, per `compiler::driver::eligible`'s `primitive() !=
    /// 0` rejection, but the trampolines themselves have nothing to do with
    /// the JIT: `Stubs`/`DeoptTrampolines` are compiled-code plumbing,
    /// `FfiStubs` is the interpreter's own gateway to `dlsym`-resolved
    /// native code, reached from `runtime::ffi::dispatch_ffi_primitive`).
    /// `derive(Clone, Copy)` (see `FfiStubs`'s own doc) — cheap to store
    /// and to hand a copy to any caller that needs it, same as `stubs`.
    pub ffi_stubs: crate::codecache::ffi_stubs::FfiStubs,
    /// S13 step 5: the generated deopt trampolines (the uncommon trampoline
    /// plus the `0xDE02` assert stub), published into `code_cache` alongside
    /// `stubs`, and the process-global SIGTRAP handler that redirects into
    /// them. `Some` only when the JIT is enabled (not `JitMode::Off`) — a pure
    /// interpreter run neither compiles nor traps, so it installs no handler,
    /// matching the README's `MACVM_JIT=off` debugger caveat (D3). Later steps
    /// read the two handles to redirect returns (step 9) and to re-emit stress
    /// traps (step 11); step 5 only publishes and arms them.
    pub deopt_trampolines: Option<crate::codecache::deopt_trap::DeoptTrampolines>,
    /// S10 D6/step 8: one entry per currently-live compiled activation,
    /// pushed/popped around `enter_compiled`. Always empty under
    /// `JitMode::Off`.
    pub tier_links: Vec<TierLink>,
    /// How many compiled activations are currently nested (== `tier_links
    /// .len()` whenever only `IntoCompiled` links exist, which is every
    /// case in S10 — S11's `IntoInterpreter` links are what would make
    /// this something other than a redundant count).
    pub compiled_depth: u32,
    /// This VM's live-signal block (per-VM, no global): a monitor samples it
    /// off-thread. `compiled_depth` above is mirrored into it at the two
    /// interp->compiled boundary sites, so the GUI can derive an
    /// interpreter/compiler ratio without a request round-trip.
    pub live_stats: std::sync::Arc<crate::embed::VmLiveStats>,
    /// S13 D1 §2c: in-flight activations of `NotEntrant` nmethods whose
    /// callee return-address slots have been redirected to
    /// `deopt_return_trampoline`, keyed by the redirected activation's OWN FP
    /// (see [`PendingDeopt`]). `make_not_entrant`'s §2c walk inserts one entry
    /// per redirected slot; `rt_deopt_on_return` `remove`s it when the callee
    /// finally returns and the return-path deopt fires. Always empty under
    /// `JitMode::Off` (nothing compiled ⇒ nothing to invalidate). A
    /// redirected-but-never-returned entry (process exit, or an NLR unwinding
    /// past the victim frame) is harmless dead data. Consulted by key both on
    /// an actual trampoline return (`rt_deopt_on_return`) AND by
    /// `frames::walk_frames`' `resolve_redirected_lr` — which, whenever a live
    /// walk (a GC root scan, S12) crosses a redirected saved-LR slot, reads
    /// this map to recover the victim's ORIGINAL return pc so its frame is
    /// classified/oopmap-scanned at its real safepoint rather than the
    /// trampoline address.
    pub pending_deopts: HashMap<usize, PendingDeopt>,
    /// S11 D6.3: the parked target+value of a non-local return currently
    /// ESCAPING through one or more compiled frames. `continue_unwind` sets
    /// it (`Some`) when a block's home is on the far side of a c2i boundary
    /// and returns the `NLR_SENTINEL` instead of unwinding across the
    /// native gap; `enter_compiled`'s own sentinel arm, once the sentinel
    /// has propagated back up through the compiled frame's normal epilogue,
    /// reads it to RESUME the unwind on the home side. `None` at every
    /// `enter_compiled` ENTRY (asserted — a leaked state from an aborted
    /// unwind must be caught, P9-style, not silently consumed by the next
    /// unrelated compiled call); cleared only when the unwind finally
    /// returns from home.
    pub nlr_state: Option<NlrState>,
    /// S24 A1: >0 while a deopt's nested interpreted resume
    /// (`interpret_active`) is running. `enter_compiled`'s opportunistic
    /// zombie sweep must not walk frames in this window: the deopt
    /// machinery has already zeroed `compiled_depth` (the abandoned frame
    /// no longer counts) while the NATIVE chain above still holds the
    /// mid-deopt call_stub crossings whose tier links were consumed —
    /// `walk_frames` would hit "call_stub with no matching link". Found the
    /// moment activate_block's enter-compiled hook made a `value:` send
    /// inside a deopt resume reach enter_compiled.
    pub deopt_resume_depth: u32,
    /// Re-entrancy guard for `unwind::run_curtailment_blocks_on_error`: true
    /// while an unhandled error is running the `ensure:`/`ifCurtailed:` blocks
    /// between the erroring frame and the entry boundary. A cleanup block is
    /// ordinary Smalltalk and may itself error, which would land back in
    /// `prim_error` and re-enter the same walk over the same frames forever;
    /// one curtailment pass per error, and a cleanup that errors simply
    /// forfeits its own remaining cleanups.
    pub curtailing_on_error: bool,
    /// Test-only one-shot hook (tests_s10.md's `mixed_trace_golden`, gate
    /// item 4): compiled code is send-free (D1), so nothing inside it can
    /// call a `printStackTrace` primitive directly — a test sets this
    /// before entering a call whose compiled callee has a loop, and
    /// `codecache::stubs::rt_poll` (reached when that loop's back-edge
    /// `Poll` actually fires) checks it, prints via
    /// `runtime::error::print_stack_trace`, and clears it. Always `false`
    /// outside tests — a normal run never sets it, so this is a dormant
    /// field rather than a `#[cfg(test)]` one, matching `poll_flag`'s own
    /// "wired but usually inert" status.
    pub trace_on_poll: bool,
    /// S13 step 10b: `true` while at least one compiled activation's own
    /// nmethod is `NotEntrant` and could still be running a call-free loop that
    /// only the loop poll can deopt. `flush::make_not_entrant` sets it (and arms
    /// `reg_block.poll_flag`) as a GLOBAL "some frame needs a loop-poll deopt"
    /// signal; every compiled loop then polls, but `rt_poll` only DEOPTS a frame
    /// whose OWN nmethod is `NotEntrant` — a poll in any other (still-`Alive`)
    /// frame checks this flag, sees no self-match, and returns "continue" fast.
    /// `rt_poll` clears it (and `poll_flag`) once no `NotEntrant` compiled frame
    /// remains, so polling is bounded to the drain window (§2d). Default `false`.
    pub pending_deopt_flag: bool,
    /// S12 D3 test-only hook, same "dormant, not `#[cfg(test)]`" status as
    /// `trace_on_poll`: `codecache::stubs::rt_alloc_slow` (one of the six
    /// anchor-setting stubs, so a real place `runtime::frames::walk_frames`
    /// has real native frames to walk) checks this, and if armed
    /// (`Some(Ok(Vec::new()))`), runs a real walk and overwrites it with
    /// either the captured `FrameView` sequence or (`Err`) the message of a
    /// panic the walk itself raised — caught with `catch_unwind` INSIDE the
    /// stub's own Rust function, never allowed to unwind across the
    /// `extern "C"` boundary back into hand-assembled native code (UB).
    /// Exists because faking a native fp/pc pair by hand (rather than
    /// letting a real compiled call establish one) would have
    /// `walk_frames` dereference bogus stack memory — there is no way to
    /// test this module honestly without a real in-flight native chain.
    pub test_walk_capture: Option<Result<Vec<crate::runtime::frames::FrameView>, String>>,
    /// Companion to `test_walk_capture`: if set, `rt_alloc_slow` pops
    /// `vm.tier_links` immediately before running the armed walk —
    /// `tests_s12.md`'s `walker_terminates_on_torn_tierlinks`, proving the
    /// walker fails loudly (not silently or by looping forever) when its
    /// own cross-check invariant (a native boundary always has a matching
    /// tier link) breaks.
    pub test_tear_tier_links_before_walk: bool,
    /// S12 step 4 test hook (its step-4-era companion flag,
    /// `test_allow_moving_gc_under_compiled`, died with the D8 bridge in
    /// step 7 — a scavenge under a live compiled frame no longer needs any
    /// bypass, it's simply legal): if set, `rt_alloc_slow` runs a REAL
    /// `memory::scavenge::scavenge` before doing its
    /// own normal allocation, and overwrites this with `Ok((before, after))`
    /// — the raw bits of the SOLE live oop slot `runtime::frames::
    /// walk_frames` finds in the compiled frame beneath it, read
    /// immediately before and after the forced scavenge — or (`Err`) the
    /// message of a panic either the walk or the scavenge itself raised.
    /// Same `catch_unwind`-inside-the-`extern "C"`-boundary reasoning as
    /// `test_walk_capture`: a panic here must never unwind into the
    /// hand-assembled native frames below. One-shot (cleared immediately),
    /// matching `trace_on_poll`'s convention.
    pub test_force_scavenge_in_alloc_slow: bool,
    pub test_scavenge_probe: Option<Result<(u64, u64), String>>,
}

impl VmState {
    // S12 step 7: the ENTIRE S11 D8 bridge apparatus that lived here is
    // DELETED, not relocated — `publish_eden_to_regblock`/
    // `adopt_eden_from_regblock` (the eden-value-copy sync protocol;
    // superseded by `reg_block.eden_top_addr` sharing the one canonical
    // `universe.eden.top` word with compiled code directly) and
    // `GcPendingKind`/`gc_pending`/`request_pending_gc`/
    // `run_pending_gc_if_due` (the deferred-collection mechanism; a
    // collection under a live compiled frame is now simply an ordinary,
    // fully rooted collection, so there is nothing left to defer).

    /// `MACVM_DEOPT_STRESS`'s default invalidation period (D7's own tunable).
    pub const DEFAULT_STRESS_PERIOD: u64 = 1_000;

    /// Pure parse of `MACVM_DEOPT_STRESS` (D7): `1` → on at the default period;
    /// a bare `N` → on with period `N`; unset/other → off. Read in [`Self::new`]
    /// (NOT `with_options`, which is deliberately env-free for tests) — but
    /// unlike `MACVM_JIT` this is safe to source from the ambient environment
    /// because stress is OUTPUT-EQUIVALENT (a differential harness): it only
    /// forces extra deopts, never changes a program's result.
    /// `pub(crate)` also for RUSTTCL's `flag` verb — same grammar, live.
    pub(crate) fn parse_deopt_stress(raw: Option<&str>) -> (bool, u64) {
        match raw.map(str::trim) {
            Some("1") => (true, Self::DEFAULT_STRESS_PERIOD),
            Some(s) => match s.parse::<u64>() {
                Ok(n) if n > 0 => (true, n),
                _ => (false, Self::DEFAULT_STRESS_PERIOD),
            },
            None => (false, Self::DEFAULT_STRESS_PERIOD),
        }
    }

    /// D7: attribute a deopt to its trigger (`deopt_by_reason`) and, under
    /// `MACVM_TRACE=deopt`, print it. Called at each runtime deopt entry point
    /// AFTER `deoptimize_frame` has bumped `deopt_count`.
    pub fn note_deopt(&mut self, reason: DeoptReason) {
        self.stats.deopt_by_reason[reason as usize] += 1;
        if self.options.trace.is_enabled("deopt") {
            eprintln!(
                "[deopt] {reason:?} (#{}, by_reason=[trap {}, return {}, poll {}])",
                self.stats.deopt_count,
                self.stats.deopt_by_reason[0],
                self.stats.deopt_by_reason[1],
                self.stats.deopt_by_reason[2],
            );
        }
    }

    /// Parses options from the environment once and boots a fresh universe.
    pub fn new() -> VmState {
        let mut vm = Self::with_options(VmOptions::from_env());
        vm.dbg_oop = std::env::var("MACVM_DBG_OOP").ok().and_then(|s| {
            let s = s.trim();
            let s = s.strip_prefix("0x").unwrap_or(s);
            usize::from_str_radix(s, 16).ok()
        });
        // S13 D7: stress is a VmState field (not a VmOptions one — 78 literal
        // constructors), sourced from the ambient env only on this real entry
        // point, never in `with_options`.
        let (ds, sp) =
            Self::parse_deopt_stress(std::env::var("MACVM_DEOPT_STRESS").ok().as_deref());
        vm.deopt_stress = ds;
        vm.stress_period = sp;
        vm.stress_countdown = sp;
        vm
    }

    /// Bypasses env parsing — required by tests (parallel test runners
    /// cannot race on process-wide env vars) and by any later embedder.
    pub fn with_options(options: VmOptions) -> VmState {
        let universe = Universe::genesis(&options);
        // Every reg-block field is fixed for the whole process lifetime and
        // set ONCE here (S11 D3/P7, extended by S12 step 7): old gen's own
        // `bounds.start` never moves (`OldGen::grow` only advances
        // `committed_end`), the card table is sized for old's FULL reserved
        // range at genesis, eden's ceiling (`eden.end`) never grows or
        // moves after genesis (only `old` ever grows), and
        // `eden_top_addr` is the ADDRESS of the one live bump pointer —
        // stable because `Universe::eden` is boxed (see the field's own
        // doc), so it survives `universe` being moved into the `VmState`
        // literal below and every later move of `VmState` itself. Taken
        // via `&raw const` (never a `&`/`&mut` intermediate) so no
        // reference's provenance is ever narrowed to the read that created
        // it — compiled code will WRITE through this address for the
        // process's whole lifetime.
        let mut reg_block = VmRegBlock::new();
        reg_block.old_start = universe.layout.old_start as u64;
        reg_block.card_base_biased = universe.cards.base_biased();
        reg_block.eden_top_addr = (&raw const universe.eden.top) as u64;
        reg_block.eden_end = universe.eden.end as u64;
        // Stubs are installed unconditionally (regardless of `options.jit`)
        // so `compile_method` (S10 D4) never has to lazily bootstrap them
        // mid-compile — one small, fixed, one-time cost per VM, matching
        // `Universe::genesis` itself always building well-knowns a given
        // script may never touch.
        // A genesis-time mmap failure (like `Reservation::reserve`'s own) —
        // `fatal_exit`, not `.expect()`'s panic, so an embedded `VmHandle`
        // (S21) loses only its own thread, not the whole process, on the
        // same environment-limit condition that already terminates via
        // `fatal_exit` everywhere else in the VM.
        // `MACVM_CODE_CACHE=<MiB>` (S24 A1 hardening): pin the code-cache
        // reservation, chiefly to make EXHAUSTION cheap to reach on
        // purpose — the graceful-degradation ladder (`rt_resolve_send`'s
        // unpatched-site fallback, `driver`'s compile decline) is
        // otherwise only reachable by filling the full default, which no
        // reasonable workload does. `MACVM_CODE_CACHE=1` + DEOPT_STRESS
        // is the standing soak for that path. Parsed here, not in
        // `VmOptions` (102 construction sites; the cache is per-VM
        // plumbing, not test-relevant configuration).
        let cache_capacity = std::env::var("MACVM_CODE_CACHE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mib| mib.clamp(1, 1024) << 20)
            .unwrap_or(DEFAULT_CODE_CACHE_CAPACITY);
        let mut code_cache = match CodeCache::new(cache_capacity) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("macvm: failed to reserve JIT code cache: {e}");
                fatal_exit(71);
            }
        };
        let stubs = crate::codecache::stubs::install(&mut code_cache);
        // S20 step 4: publish the 3 FFI trampolines alongside `stubs`, same
        // unconditional-at-boot install — an FFI method's own eligibility
        // is unrelated to `options.jit` (it always runs interpreted, per
        // `compiler::driver::eligible`'s `primitive() != 0` rejection), so
        // there is no `JitMode::Off` carve-out here the way there is for
        // `deopt_trampolines` below (those exist purely to service compiled
        // code; these service `runtime::ffi::dispatch_ffi_primitive`, which
        // can run in ANY `JitMode`).
        let ffi_stubs = crate::codecache::ffi_stubs::install(&mut code_cache);
        // S13 step 5: publish the deopt trampolines and arm the SIGTRAP
        // handler — but ONLY with the JIT on. Under `JitMode::Off` nothing is
        // ever compiled, so no deopt `brk` can fire; installing a
        // process-global handler would only add a debugger caveat (D3) for no
        // benefit. Doing it here (before any compiled code runs, alongside the
        // SPEC §9 stubs) matches the sprint's "generated at startup" order.
        let deopt_trampolines = if matches!(options.jit, JitMode::Off) {
            None
        } else {
            Some(crate::codecache::deopt_trap::install(&mut code_cache))
        };
        let mut vm = VmState {
            reg_block,
            universe,
            options,
            stack: ProcessStack::with_capacity(DEFAULT_STACK_CAPACITY),
            regs: InterpRegs::default(),
            ic_epoch: 0,
            lookup_cache: LookupCache::new(),
            prim_arg_base: 0,
            out: Box::new(std::io::stdout()),
            game_sink: None,
            workers: None,
            cocoa_mint_stack: Vec::new(),
            exit_requested: false,
            exit_code: None,
            start_instant: Instant::now(),
            bytecode_count: 0,
            count_by_method: std::collections::HashMap::new(),
            count_cur: None,
            stats: VmStats::default(),
            // S13 D7: OFF here (env-free constructor); `new()` turns it on from
            // `MACVM_DEOPT_STRESS`, tests set the fields directly.
            deopt_stress: false,
            stress_period: Self::DEFAULT_STRESS_PERIOD,
            stress_countdown: Self::DEFAULT_STRESS_PERIOD,
            stress_rr_cursor: 0,
            next_frame_serial: 0,
            dbg_oop: None,
            probe_ring: ProbeRing::new(),
            debug: crate::runtime::debug::DebugState::new(),
            handle_arena: Box::new(crate::memory::handles::HandleArena::new()),
            code_cache,
            code_table: CodeTable::new(),
            adapters: crate::codecache::adapters::AdapterTable::new(),
            pic_table: crate::codecache::pics::PicTable::new(),
            mega_table: crate::codecache::mega::MegaTable::new(),
            stubs,
            ffi_stubs,
            deopt_trampolines,
            tier_links: Vec::new(),
            compiled_depth: 0,
            live_stats: std::sync::Arc::new(crate::embed::VmLiveStats::default()),
            pending_deopts: HashMap::new(),
            nlr_state: None,
            deopt_resume_depth: 0,
            curtailing_on_error: false,
            trace_on_poll: false,
            pending_deopt_flag: false,
            test_walk_capture: None,
            test_tear_tier_links_before_walk: false,
            test_force_scavenge_in_alloc_slow: false,
            test_scavenge_probe: None,
        };
        crate::runtime::globals::bootstrap_well_known(&mut vm);
        // S20 step 5 (docs/FFI.md §4): installs Alien's own instance/class-
        // side accessor methods onto the genesis-shaped `alien_klass` —
        // must run AFTER `bootstrap_well_known` above, since it relies on
        // "Alien" already being a resolvable global (`install_class_def`'s
        // REOPEN branch, not its create branch — see
        // `runtime::alien::bootstrap_alien_methods`'s own doc comment).
        crate::runtime::alien::bootstrap_alien_methods(&mut vm);
        vm
    }

    /// Re-reads live stack slot `prim_arg_base + i` — the choke-point a
    /// `can_allocate` primitive must use instead of its `args` copy after
    /// any allocating call (SPEC §10 Pitfalls).
    pub fn prim_arg(&self, i: usize) -> crate::oops::Oop {
        self.stack.get(self.prim_arg_base + i)
    }

    /// Hands out the next frame serial, post-incrementing the counter.
    pub fn alloc_frame_serial(&mut self) -> u32 {
        let s = self.next_frame_serial;
        let (next, wrapped) = self.next_frame_serial.overflowing_add(1);
        debug_assert!(
            !wrapped,
            "alloc_frame_serial: wrapped past u32::MAX (accepted risk, 4G pushes)"
        );
        self.next_frame_serial = next;
        s
    }

    /// The serial `alloc_frame_serial` will hand out *next*, without
    /// consuming it — test-only, for predicting a to-be-pushed frame's
    /// serial before triggering the push.
    #[cfg(test)]
    pub(crate) fn peek_next_frame_serial(&self) -> u32 {
        self.next_frame_serial
    }
}

impl Default for VmState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_flags_parse() {
        let t = TraceFlags::parse("gc, jit,, bytecode");
        assert!(t.is_enabled("gc"));
        assert!(t.is_enabled("jit"));
        assert!(t.is_enabled("bytecode"));
        assert!(!t.is_enabled("ic"));
    }

    #[test]
    fn vm_options_default_heap() {
        let o = VmOptions::default();
        assert_eq!(o.heap_mib, VmOptions::DEFAULT_HEAP_MIB);
    }

    #[test]
    fn gc_stress_unset_means_off() {
        assert_eq!(VmOptions::parse_gc_stress(None), (false, None));
    }

    #[test]
    fn gc_stress_1_scavenges_every_alloc() {
        assert_eq!(VmOptions::parse_gc_stress(Some("1")), (true, None));
    }

    #[test]
    fn gc_stress_full_uses_default_period() {
        assert_eq!(
            VmOptions::parse_gc_stress(Some("full")),
            (false, Some(VmOptions::DEFAULT_FULL_STRESS_PERIOD))
        );
    }

    #[test]
    fn gc_stress_full_with_explicit_period() {
        assert_eq!(
            VmOptions::parse_gc_stress(Some("full:50")),
            (false, Some(50))
        );
    }

    #[test]
    fn gc_stress_modes_are_mutually_exclusive() {
        // "1" never also sets a full-GC period, and "full[:N]" never also
        // sets the scavenge-every-alloc bool -- every valid value picks
        // exactly one mode, matching alloc_words' cascade (which checks
        // both but only one is ever Some/true for a given env value).
        let (scavenge, full) = VmOptions::parse_gc_stress(Some("full:7"));
        assert!(!scavenge);
        assert_eq!(full, Some(7));
    }

    #[test]
    fn gc_stress_garbage_value_is_off() {
        // Neither "1" nor "full"-prefixed, and "full:" with an unparseable
        // suffix -- both must fail closed (stress off), never panic or
        // silently pick a default the user didn't ask for.
        assert_eq!(VmOptions::parse_gc_stress(Some("nonsense")), (false, None));
        assert_eq!(
            VmOptions::parse_gc_stress(Some("full:notanumber")),
            (false, None)
        );
        assert_eq!(VmOptions::parse_gc_stress(Some("")), (false, None));
    }

    #[test]
    fn gc_stress_trims_whitespace() {
        assert_eq!(
            VmOptions::parse_gc_stress(Some(" full:12 ")),
            (false, Some(12))
        );
        assert_eq!(VmOptions::parse_gc_stress(Some(" 1 ")), (true, None));
    }

    /// tests_s10.md's `jit_flag_parsing`: `off`, `threshold=1`,
    /// `threshold=10000`, absent, and garbage all resolve to the correct
    /// `JitMode` (garbage falls back to `Off`, same as `parse_gc_stress`'s
    /// own precedent — never silently pick a mode the user didn't ask for).
    #[test]
    fn jit_flag_parsing() {
        assert_eq!(VmOptions::parse_jit(None), JitMode::Off);
        assert_eq!(VmOptions::parse_jit(Some("off")), JitMode::Off);
        assert_eq!(
            VmOptions::parse_jit(Some("threshold=1")),
            JitMode::Threshold(1)
        );
        assert_eq!(
            VmOptions::parse_jit(Some("threshold=10000")),
            JitMode::Threshold(10000)
        );
        assert_eq!(VmOptions::parse_jit(Some("nonsense")), JitMode::Off);
        assert_eq!(VmOptions::parse_jit(Some("threshold=")), JitMode::Off);
        assert_eq!(VmOptions::parse_jit(Some("threshold=abc")), JitMode::Off);
    }

    /// S10 D6: every `VmRegBlock` field's real, compiler-computed offset
    /// matches the `oops::layout::VMREG_*_OFFSET` constant compiled code's
    /// direct `[x28, #N]` accesses bake in — the two are hand-kept in sync
    /// (there's no `#[derive(offset_consts)]`), so this is the tripwire
    /// that catches them drifting apart.
    #[test]
    fn vmregblock_offsets_pinned() {
        use crate::oops::layout::{
            VMREG_BLOCK_SIZE, VMREG_CARD_BASE_BIASED_OFFSET, VMREG_EDEN_END_OFFSET,
            VMREG_EDEN_TOP_ADDR_OFFSET, VMREG_LAST_COMPILED_FP_OFFSET,
            VMREG_LAST_COMPILED_KIND_OFFSET, VMREG_LAST_COMPILED_PC_OFFSET, VMREG_OLD_START_OFFSET,
            VMREG_POLL_FLAG_OFFSET,
        };
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, eden_top_addr),
            VMREG_EDEN_TOP_ADDR_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, eden_end),
            VMREG_EDEN_END_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, old_start),
            VMREG_OLD_START_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, card_base_biased),
            VMREG_CARD_BASE_BIASED_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, poll_flag),
            VMREG_POLL_FLAG_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, last_compiled_fp),
            VMREG_LAST_COMPILED_FP_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, last_compiled_pc),
            VMREG_LAST_COMPILED_PC_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(VmRegBlock, last_compiled_kind),
            VMREG_LAST_COMPILED_KIND_OFFSET
        );
        assert_eq!(std::mem::size_of::<VmRegBlock>(), VMREG_BLOCK_SIZE);
        // D6's other half: reg_block must sit at VmState's own offset 0, or
        // x28 == &VmState would not equal x28 == &VmState.reg_block.
        assert_eq!(std::mem::offset_of!(VmState, reg_block), 0);
    }

    /// S11 D3/P7 + S12 step 7: `with_options` must populate EVERY reg-block
    /// GC field with real values (not `VmRegBlock::new()`'s own all-zero
    /// default) — a compiled write barrier reading a zero `old_start` would
    /// treat every object as old-gen and dirty cards at nonsense addresses;
    /// a compiled inline-alloc dereferencing a zero `eden_top_addr` would
    /// fault outright.
    #[test]
    fn reg_block_gc_fields_populated_from_real_universe() {
        let vm = VmState::with_options(VmOptions::default());
        assert_eq!(
            vm.reg_block.old_start, vm.universe.layout.old_start as u64,
            "old_start must mirror the real, fixed old-gen boundary"
        );
        assert_ne!(
            vm.reg_block.old_start, 0,
            "a zero old_start would be a genesis-ordering bug"
        );
        assert_eq!(
            vm.reg_block.card_base_biased,
            vm.universe.cards.base_biased(),
            "card_base_biased must mirror CardTable's own bias formula exactly"
        );
        assert_eq!(
            vm.reg_block.eden_end, vm.universe.eden.end as u64,
            "eden_end must mirror the real, genesis-fixed eden ceiling"
        );
        assert_eq!(
            vm.reg_block.eden_top_addr,
            (&raw const vm.universe.eden.top) as u64,
            "eden_top_addr must be the address of the ONE live bump pointer"
        );
    }

    /// S12 step 7's single-source-of-truth invariant across a MOVE of the
    /// whole `VmState` (exactly what `with_options` returning by value does
    /// to every real VM, and what every test's `let mut vm = test_vm()`
    /// does again): the once-stored `eden_top_addr` must still be the
    /// address of `universe.eden.top` afterward — which is the whole
    /// property, since equal addresses ARE the same memory word; a write
    /// through one is a write to the other by definition (the real
    /// through-the-pointer traffic is exercised end-to-end by every
    /// compiled inline-alloc test, e.g. `allocation_fast_and_slow`). Holds
    /// because the `Eden` struct itself is boxed; the `VmState`/`Universe`
    /// shells around it can move freely without moving it.
    #[test]
    fn eden_top_addr_survives_vmstate_moves() {
        let vm = VmState::with_options(VmOptions::default());
        let boxed = Box::new(vm); // deliberate move #1
        let vm = *boxed; // deliberate move #2

        assert_ne!(vm.reg_block.eden_top_addr, 0);
        assert_eq!(
            vm.reg_block.eden_top_addr,
            (&raw const vm.universe.eden.top) as u64,
            "the stored address must still name eden.top after VmState moves"
        );
    }

    // S12 step 7: the four `gc_pending`/`request_pending_gc`/
    // `run_pending_gc_if_due` tests died with the mechanism itself — the
    // property they guarded ("a collection asked for under a compiled
    // frame eventually runs") is superseded by the stronger one the
    // it_gc_jit.rs flagship tests prove: it runs IMMEDIATELY, correctly,
    // with the compiled frame's own oops as live roots.
}
