//! S13 step 5 — the `brk`-based uncommon-trap mechanism: `brk` emission, the
//! macOS SIGTRAP handler, and the startup-generated trampolines that escape
//! from signal context into ordinary Rust (`sprint_s13_detail.md` D3/D4/D6).
//!
//! This is `codecache`'s single unsafe island for signal/ucontext work
//! (layer-boundary table: `deopt_trap` "may touch sigaction/ucontext,
//! code-cache bounds, stub generation, JIT toggle" and "must not touch heap
//! allocation, Universe, interpreter"). The handler itself (D3) is
//! async-signal-safe by doing almost nothing: it inspects the fault pc,
//! namespace-checks the `brk` imm, and — for one of *our* traps — rewrites
//! the ucontext to resume in a generated trampoline. Every observable action
//! (allocation, GC, `VmState` mutation, printing) happens *after* sigreturn,
//! in the Rust reached through that trampoline. The handler never allocates,
//! locks, formats a string, unwinds, or touches `VmState`.
//!
//! PAC note (D3, `arm64.md` §5 baseline): PAC is off for VM-internal control
//! flow, so rewriting `__pc`/`__lr` and the trampoline's later frame surgery
//! need no `pacia`/`autia`. That is the reason this whole design is legal as
//! written — stated here so it is not silently assumed.
//!
//! **Step 5 boundary.** This module resolves a trapping pc to its owning
//! nmethod + `DeoptState` and hands off *toward* materialization; the
//! materializer / interpreter-frame reconstruction (D5 M0–M8) is S13 step 6,
//! living in `runtime/deopt.rs`. The handoff seam here is
//! [`rt_uncommon_trap`], which resolves the `DeoptState` and then
//! deliberately aborts with a "step 6" marker (see its body).

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::compiler::assembler::xr;
use crate::compiler::assembler::{imm, mem, mem_post, mem_pre, sp, x, Assembler, RelocKind};
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::oops::layout::{
    VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET, VMREG_LAST_COMPILED_PC_OFFSET,
};
use crate::oops::Oop;

use super::stubs::KIND_DEOPT_BRIDGE;

use super::{CodeCache, CodeHandle};

// ── The `brk #imm` uncommon-trap namespace (D3 / sprint doc §"Uncommon trap
//    sites") ─────────────────────────────────────────────────────────────

/// MACVM claims the `brk #0xDExx` immediate namespace. The trap **site**
/// (faulting pc → nmethod → `PcDesc::find`) identifies the deopt state; the
/// imm only namespaces our `brk`s against foreign ones.
///
/// `0xDE00` — uncommon trap (guard failure / cold path): deopt + count.
pub const TRAP_UNCOMMON: u16 = 0xDE00;
/// `0xDE01` — forced trap from `MACVM_DEOPT_STRESS`: deopt, counted
/// separately, does NOT count toward `UncommonTrapLimit` (S13 step 11 wires
/// the distinction; step 5 only needs the imm reserved and recognised).
pub const TRAP_STRESS: u16 = 0xDE01;
/// `0xDE02` — compiled-code assertion ("should not reach"): a VM bug; panic
/// with the pc (handled off-signal via a tiny assert stub, D3 step 4).
pub const TRAP_ASSERT: u16 = 0xDE02;

/// AArch64 `brk #imm16` base opcode: `0xD420_0000 | (imm16 << 5)`.
const BRK_BASE: u32 = 0xD420_0000;
/// Mask isolating the opcode bits of a `brk` (everything except the imm16
/// field, which sits in bits 20:5).
const BRK_OPCODE_MASK: u32 = 0xFFE0_001F;
const BRK_IMM_SHIFT: u32 = 5;

/// Encode `brk #imm16`.
pub const fn brk_word(imm16: u16) -> u32 {
    BRK_BASE | ((imm16 as u32) << BRK_IMM_SHIFT)
}

/// Decode a 32-bit instruction word as one of *our* deopt `brk`s. Returns the
/// imm16 only when the word is a `brk` **and** the imm is in our reserved
/// `0xDE00..=0xDE02` range; every other `brk` (notably Rust's `abort()`, which
/// lowers to `brk #1` on arm64, and `unreachable`/`0xDE03..` we never emit) is
/// **not ours** — `None` here drives the handler's SIG_DFL re-raise so those
/// stay fatal (Pitfalls: "Foreign `brk`s").
pub fn decode_deopt_brk(word: u32) -> Option<u16> {
    if word & BRK_OPCODE_MASK != BRK_BASE {
        return None;
    }
    let imm = ((word >> BRK_IMM_SHIFT) & 0xFFFF) as u16;
    match imm {
        TRAP_UNCOMMON | TRAP_STRESS | TRAP_ASSERT => Some(imm),
        _ => None,
    }
}

/// Emit an uncommon-trap `brk #imm` through the S9 `Assembler`. The vendored
/// encoder does not support `brk` (assembler.rs `emit_u32` doc names S13's
/// `brk #imm` traps as exactly the raw-word case), so this goes out as a raw
/// word. Recording the site's `SafepointState` (`kind = UncommonTrap`,
/// `reexecute = true`) is the *emit stage's* job (S13 step 3/7) — this helper
/// only lays down the instruction, so it is equally usable from a real emit
/// path and from this module's own hand-built test stubs.
pub fn emit_brk(a: &mut dyn Assembler, imm16: u16) {
    a.emit_u32(brk_word(imm16));
}

/// WINVM: the x86-64 deopt trap site — `int3` (`0xCC`) followed by the
/// same imm16 the AArch64 `brk` carries, little-endian. Three bytes, of
/// which only the first ever executes: the VEH ([`veh_trap_handler`])
/// reads the imm at `Rip + 1` and redirects, so control never reaches the
/// two immediate bytes. [`decode_deopt_int3`] is the reader.
///
/// The site's own offset IS the trapping pc (Windows reports a breakpoint
/// with `Rip` pointing AT the `0xCC`), so an emitter records its safepoint
/// **before** calling this — the same rule the AArch64 `brk` path follows.
pub const fn deopt_int3_bytes(imm16: u16) -> [u8; 3] {
    [0xCC, imm16 as u8, (imm16 >> 8) as u8]
}

// ── The code-cache registry (D3 step 2, multi-VmState-safe) ───────────────
//
// The handler cannot safely reach a `&CodeCache` from signal context, so each
// registered code cache's `[lo, hi)` range and its OWN trampolines are cached
// here. A trapping pc is looked up against this registry: the entry whose
// range contains it names the cache the brk came from, and that cache's own
// (guaranteed-live — the brk fired from inside it) uncommon trampoline / assert
// stub. This replaces an earlier single-range/single-trampoline pair whose
// last-writer-wins semantics silently misdirected a brk once a SECOND JIT
// `VmState` (each with its own `CodeCache` at a different address) was created
// — a real hazard in the test suite and any multi-VM host, not just theory.
//
// Signal-safety: the handler only READS these atomics (a fixed-size array
// scan, no allocation/lock), which is async-signal-safe. Mutation
// (`register`/`deregister`, at `install`/`CodeCache::drop`, never in signal
// context) is serialized by `REGISTRY_LOCK`; the handler never takes it. A
// slot's `lo` is the match key and is written LAST (Release) / read FIRST
// (Acquire), so the handler sees either a fully-populated entry or none.
//
// `lo == 0` marks an empty/retired slot (a real code-cache base is never 0).

const REGISTRY_CAP: usize = 128;
static REG_LO: [AtomicU64; REGISTRY_CAP] = [const { AtomicU64::new(0) }; REGISTRY_CAP];
static REG_HI: [AtomicU64; REGISTRY_CAP] = [const { AtomicU64::new(0) }; REGISTRY_CAP];
static REG_TRAMP: [AtomicU64; REGISTRY_CAP] = [const { AtomicU64::new(0) }; REGISTRY_CAP];
static REG_ASSERT: [AtomicU64; REGISTRY_CAP] = [const { AtomicU64::new(0) }; REGISTRY_CAP];
/// DBG0: the per-cache PROBE trampoline (docs/DEBUGGER.md §4.1) — the fifth
/// registry column, same publish discipline as the other four.
static REG_PROBE: [AtomicU64; REGISTRY_CAP] = [const { AtomicU64::new(0) }; REGISTRY_CAP];
/// High-water mark of slots ever used (the handler scans `0..REG_LEN`; retired
/// slots within it have `lo == 0` and are skipped / reused).
static REG_LEN: AtomicUsize = AtomicUsize::new(0);
static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

// ── DBG0: PROBE's in-handler capture + reentrancy state ──────────────────
//
// The register file at the trigger, copied in-handler (plain relaxed atomic
// stores — async-signal-safe) and read post-sigreturn by `rt_probe_crash`/
// `rt_compiled_assert_failed`. Layout: [0..29] x0-x28, [29] fp, [30] lr,
// [31] sp, [32] pc, [33] cpsr, [34] far, [35] signal number.
const CAP_FP: usize = 29;
const CAP_LR: usize = 30;
const CAP_SP: usize = 31;
const CAP_PC: usize = 32;
const CAP_CPSR: usize = 33;
const CAP_FAR: usize = 34;
const CAP_SIG: usize = 35;
static CAPTURED: [AtomicU64; 36] = [const { AtomicU64::new(0) }; 36];

// ── S21: per-thread foreign-fault recovery registry ──────────────────────
//
// PROBE's own fault handler (below) only ever redirects a fault whose pc
// falls inside a REGISTERED code cache (the `lookup_pc_full` scan above) —
// a fault in ordinary Rust code (e.g. `oops::wrappers`' own raw-pointer
// dereference for an indirect `Alien`, S20 step 5, never JIT-compiled at
// all) is classified "foreign" and left fatal (`write_foreign_verdict` +
// `restore_default_for`). That is exactly right for a bare CLI/test run
// (a foreign fault there really is an unrecoverable bug), but an embedded
// `VmHandle` (S21, `docs/SPEC.md` §16) running on its own dedicated worker
// thread needs a foreign fault to end ONLY that thread, not the process —
// same goal as `runtime::vm_state::fatal_exit`'s `FatalMode::ExitThread`,
// just for a genuine hardware fault instead of an ordinary guest `error:`/
// DNU/stack-overflow condition.
//
// Mechanism, empirically validated (three real, deliberately-induced
// `SIGSEGV`s in an isolated scratchpad binary before this code was
// written): `sigsetjmp`/`siglongjmp`, called DIRECTLY from inside the
// signal handler — unlike `panic!`/`catch_unwind` (unsound across a
// JIT-compiled frame with no Rust unwind tables) or PROBE's own
// PC-rewrite-to-trampoline trick (needed only for the heavier, allocating/
// frame-walking dossier work the in-code-cache path does), `siglongjmp` is
// itself on the async-signal-safe function list and needs neither. Proven:
// one real SIGSEGV recovers cleanly; a SECOND real SIGSEGV, after the
// first recovery, ALSO recovers cleanly (the test that actually matters —
// it proves `sigsetjmp(env, 1)`'s "save the signal mask" semantics
// correctly un-blocks SIGSEGV after `siglongjmp`; plain `setjmp`/`longjmp`
// do NOT restore the mask, which would leave SIGSEGV blocked and a second
// fault's behavior undefined); and the same mechanism works when the fault
// occurs on a spawned thread rather than the one that installed the
// handler (the real deployment shape).
//
// `libc` 0.2 does not expose `sigsetjmp`/`siglongjmp`/`sigjmp_buf` at all
// (verified against the vendored crate, same gap as `ucontext_t` above) —
// hand-declared here, with `sigjmp_buf`'s exact layout confirmed from this
// system's own `/usr/include/.../usr/include/setjmp.h`: on `arm64` macOS,
// `_JBLEN = (14 + 8 + 2) * 2 = 48`, `sigjmp_buf` is `int[_JBLEN + 1]` =
// `[c_int; 49]`.
const SIGJMP_BUF_LEN: usize = 49;

/// The identity key the jmp/recovery registries use for "this thread".
/// `pthread_self()` on unix; `GetCurrentThreadId()` on Windows (nonzero for
/// any real thread, satisfying the registries' `0 == empty slot` convention).
#[cfg(unix)]
fn current_thread_id() -> u64 {
    unsafe { libc::pthread_self() as u64 }
}

#[cfg(windows)]
fn current_thread_id() -> u64 {
    extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    // SAFETY: no arguments, no memory access — returns the caller's TID.
    unsafe { GetCurrentThreadId() as u64 }
}

// WINVM: no signal layer on Windows yet (Phase 2 replaces this whole
// mechanism with a Vectored Exception Handler — MIGRATION.md §2.2). Until
// then: `sigsetjmp` reports "no recovery point established" (returns 0 and
// nothing ever jumps back), and `siglongjmp` — reachable only through
// `raise_guest_fatal` on a thread that claimed a slot, i.e. the embedded
// `VmHandle` case that doesn't exist on Windows yet — aborts loudly rather
// than unwinding through JIT frames.
#[cfg(windows)]
pub(crate) unsafe fn sigsetjmp(
    _env: *mut core::ffi::c_int,
    _savemask: core::ffi::c_int,
) -> core::ffi::c_int {
    0
}

#[cfg(windows)]
pub(crate) unsafe fn siglongjmp(_env: *mut core::ffi::c_int, _val: core::ffi::c_int) -> ! {
    eprintln!("macvm: guest-fault recovery (siglongjmp) is not implemented on Windows yet");
    std::process::abort()
}

#[cfg(unix)]
extern "C" {
    /// Callers MUST invoke this DIRECTLY, inline, at their own call site —
    /// never through an intervening Rust wrapper function. `sigsetjmp`
    /// captures whichever stack frame is active at the moment it runs; if
    /// that frame belongs to a helper that then returns normally (as an
    /// earlier version of this module's own `register_and_setjmp` wrongly
    /// did), the frame is gone by the time a LATER fault tries to
    /// `siglongjmp` back into it — undefined behavior (observed here as
    /// execution silently resuming somewhere other than the intended
    /// point, not a clean crash, making it easy to misdiagnose). This is
    /// the exact same "cannot longjmp into a function that has already
    /// returned" rule C itself has always had — Rust's `extern "C"` FFI
    /// gives no compiler-level protection against getting it wrong, since
    /// `sigsetjmp` is just an ordinary function call from Rust's own point
    /// of view, not compiler-recognized the way it can be in C. The
    /// function whose frame calls this must not return until the
    /// recovery window it establishes is over (in practice: a loop that
    /// calls this fresh each iteration, e.g. once per guest eval).
    pub(crate) fn sigsetjmp(
        env: *mut core::ffi::c_int,
        savemask: core::ffi::c_int,
    ) -> core::ffi::c_int;
    /// Safe to call from anywhere, including from inside a signal handler
    /// (unlike `sigsetjmp`, `siglongjmp` itself has no "which frame is
    /// live" concern — it only ever restores a PREVIOUSLY, correctly
    /// established `sigsetjmp` point).
    pub(crate) fn siglongjmp(env: *mut core::ffi::c_int, val: core::ffi::c_int) -> !;
}

/// Same signal-safety discipline as the code-cache registry above: a fixed
/// array of plain atomics (never a `Mutex`, never a `thread_local!` — the
/// latter's lazy-init path on first touch is not obviously async-signal-safe,
/// and this registry's whole POINT is being read from inside a handler).
/// `0` marks an empty/retired slot (`libc::pthread_self()` is never 0 for a
/// real thread). Each slot's own `sigjmp_buf` is stored as
/// `[AtomicU32; SIGJMP_BUF_LEN]` (`AtomicU32` is layout-identical to the
/// `c_int` `sigjmp_buf` itself uses, and `AtomicU32::as_ptr` gives the raw
/// `*mut i32`/`*mut c_int` `sigsetjmp`/`siglongjmp` need) rather than a
/// `static mut` array, matching `CAPTURED`'s own established convention in
/// this file of using atomics even for what is conceptually a raw buffer.
const JMP_REGISTRY_CAP: usize = 64;
static JMP_OWNER: [AtomicU64; JMP_REGISTRY_CAP] = [const { AtomicU64::new(0) }; JMP_REGISTRY_CAP];
static JMP_BUFS: [[AtomicU32; SIGJMP_BUF_LEN]; JMP_REGISTRY_CAP] =
    [const { [const { AtomicU32::new(0) }; SIGJMP_BUF_LEN] }; JMP_REGISTRY_CAP];
/// The triggering fault's `(signal, pc, far)`, published into THIS thread's
/// own slot by the handler immediately before `siglongjmp` — read back by
/// whatever resumes at the `sigsetjmp` point (same thread, so no ordering
/// concern beyond plain same-thread program order) to build a real report.
static JMP_LAST_SIG: [AtomicU64; JMP_REGISTRY_CAP] =
    [const { AtomicU64::new(0) }; JMP_REGISTRY_CAP];
static JMP_LAST_PC: [AtomicU64; JMP_REGISTRY_CAP] = [const { AtomicU64::new(0) }; JMP_REGISTRY_CAP];
static JMP_LAST_FAR: [AtomicU64; JMP_REGISTRY_CAP] =
    [const { AtomicU64::new(0) }; JMP_REGISTRY_CAP];
static JMP_REGISTRY_LOCK: Mutex<()> = Mutex::new(());

/// Claims a registry slot for the CURRENT thread (reusing a stale slot this
/// same thread previously registered, or any retired slot, before growing —
/// same dedup-then-append shape as `register_with_probe`) and returns its
/// index. Pure bookkeeping — does NOT itself call `sigsetjmp` (deliberately:
/// see [`jmp_buf_ptr`]'s own doc for why that must happen inline, at the
/// caller's own call site, never inside a helper that would then return and
/// invalidate the very frame a later fault needs to jump back into — an
/// earlier version of this function called `sigsetjmp` itself and was wrong
/// in exactly this way, caught by this module's own real-`SIGSEGV` tests).
///
/// Only ever called from ordinary (non-signal) code — `VmHandle::boot`
/// (S21 step 2), once, before any guest code runs on that thread.
pub(crate) fn claim_jmp_slot() -> usize {
    let me = current_thread_id();
    let _g = JMP_REGISTRY_LOCK.lock().unwrap();
    let slot = (0..JMP_REGISTRY_CAP)
        .find(|&i| JMP_OWNER[i].load(Ordering::Acquire) == me)
        .or_else(|| (0..JMP_REGISTRY_CAP).find(|&i| JMP_OWNER[i].load(Ordering::Acquire) == 0))
        .expect("deopt_trap: jmp registry overflow (JMP_REGISTRY_CAP concurrent embedded VMs)");
    JMP_OWNER[slot].store(me, Ordering::Release);
    slot
}

/// The raw `*mut c_int` for slot `i`'s own `sigjmp_buf` storage — pass this
/// DIRECTLY to [`sigsetjmp`] (never copy it, never route it through another
/// function first) at the exact call site whose frame must remain live for
/// the whole recovery window (in practice: a loop, called fresh each
/// iteration — e.g. `VmHandle`'s own per-eval loop, S21 step 2).
///
/// # Safety
/// `i` must be a slot this thread itself claimed via [`claim_jmp_slot`] and
/// has not yet released via [`deregister_setjmp`].
#[allow(unsafe_code)]
pub(crate) unsafe fn jmp_buf_ptr(i: usize) -> *mut core::ffi::c_int {
    // SAFETY (of the cast, not the caller contract above): `JMP_BUFS[i]` is
    // a `[AtomicU32; 49]`, layout-identical to the real `sigjmp_buf`
    // (`[c_int; 49]`); this thread exclusively owns slot `i` once claimed
    // (no other thread ever touches an index it doesn't own by
    // `pthread_self()`), so treating its start as one contiguous
    // `*mut c_int` is sound.
    JMP_BUFS[i][0].as_ptr() as *mut core::ffi::c_int
}

/// Clears this thread's own registered slot, if any — called once a worker
/// thread is about to end for good (a clean shutdown, or right before the
/// `fatal_exit` that follows a caught crash), so a much-later, unrelated
/// thread that happens to reuse the same recycled `pthread_t` value never
/// finds a stale entry.
///
/// Wired from [`crate::embed::VmHandle`]'s `Drop`: the embedded worker owns
/// its `VmHandle` and drops it on a clean exit, so this runs on the very
/// thread that `claim_jmp_slot`'d and frees its slot. Without it, every
/// restart-on-death respawn would strand a slot owned by a now-dead
/// `pthread_t`, overflowing the fixed registry after `JMP_REGISTRY_CAP`
/// restarts. Keyed by `pthread_self()`, so calling it from a thread that
/// never claimed a slot is a harmless no-op.
pub(crate) fn deregister_setjmp() {
    let me = current_thread_id();
    let _g = JMP_REGISTRY_LOCK.lock().unwrap();
    for owner in JMP_OWNER.iter() {
        if owner.load(Ordering::Acquire) == me {
            owner.store(0, Ordering::Release);
        }
    }
}

/// How many `sigsetjmp` recovery slots are currently owned by a live thread —
/// the count of nonzero `JMP_OWNER` entries. Ordinary (non-signal) code only.
/// CG9's UI-worker restart leak gate uses this: a `VmHandle` drop must
/// `deregister_setjmp` its slot, so booting-then-dropping a UI worker N times
/// must leave this count unchanged (never climbing toward `JMP_REGISTRY_CAP`).
pub fn active_jmp_slots() -> usize {
    let _g = JMP_REGISTRY_LOCK.lock().unwrap();
    JMP_OWNER
        .iter()
        .filter(|o| o.load(Ordering::Acquire) != 0)
        .count()
}

/// How many code-cache PROBE registry slots are live (`REG_PROBE` nonzero) —
/// the sibling leak gate for the deopt/probe registry. A dropped `VmHandle`
/// whose nmethods were flushed must not strand PROBE entries across restarts.
pub fn active_probe_slots() -> usize {
    (0..REGISTRY_CAP)
        .filter(|&i| REG_PROBE[i].load(Ordering::Acquire) != 0)
        .count()
}

/// How many `sigsetjmp` recovery slots THIS thread owns (0 or 1 in practice —
/// `claim_jmp_slot` reuses the caller thread's slot). The parallel-safe leak
/// gate: a `VmHandle` boot/drop cycle on one thread must return this to 0,
/// immune to other threads' concurrent tests (unlike the process-global
/// [`active_jmp_slots`]).
pub fn current_thread_jmp_slots() -> usize {
    let me = current_thread_id();
    let _g = JMP_REGISTRY_LOCK.lock().unwrap();
    JMP_OWNER
        .iter()
        .filter(|o| o.load(Ordering::Acquire) == me)
        .count()
}

/// Async-signal-safe: does THIS (faulting) thread have a registered
/// recovery slot? A fixed-array linear scan matched by `pthread_self()`,
/// no lock, no allocation — the same shape as `lookup_pc_full` above, just
/// keyed by owning thread instead of by pc range.
fn lookup_jmp_slot_for_current_thread() -> Option<usize> {
    let me = current_thread_id();
    (0..JMP_REGISTRY_CAP).find(|&i| JMP_OWNER[i].load(Ordering::Acquire) == me)
}

/// Reads back (and clears) the `(signal, pc, far)` [`sig_fault_handler`]'s
/// foreign-fault branch published for the CURRENT thread just before its
/// `siglongjmp` — `None` if this thread never had a foreign fault recorded
/// (e.g. a fresh boot, `sigsetjmp`'s own `0` return). Ordinary
/// (non-signal) code only; same-thread program order makes the plain
/// `Relaxed` loads here see whatever the handler (necessarily on this same
/// thread, for a synchronous fault) already published.
pub(crate) fn take_last_crash_info() -> Option<(i32, u64, u64)> {
    let i = lookup_jmp_slot_for_current_thread()?;
    let sig = JMP_LAST_SIG[i].swap(0, Ordering::Relaxed);
    if sig == 0 {
        return None;
    }
    let pc = JMP_LAST_PC[i].swap(0, Ordering::Relaxed);
    let far = JMP_LAST_FAR[i].swap(0, Ordering::Relaxed);
    Some((sig as i32, pc, far))
}

/// Per-slot storage for [`raise_guest_fatal`]'s message — a `String`, unlike
/// `JMP_LAST_*` above, because every caller here runs in ORDINARY (non-signal-
/// handler) code: `runtime::error::dnu_fallback` and `primitives::prim_error`
/// execute during normal interpreted dispatch, never from inside
/// `sig_fault_handler`. A `Mutex` is fine for exactly that reason — this file's
/// own "must not touch heap allocation... " constraint (module doc) binds the
/// async-signal-safe handler path only, not this one.
static GUEST_FATAL_MSG: Mutex<[Option<String>; JMP_REGISTRY_CAP]> =
    Mutex::new([const { None }; JMP_REGISTRY_CAP]);

/// The `siglongjmp` resume value [`raise_guest_fatal`] uses — distinct from
/// [`sig_fault_handler`]'s own `1`, so `VmHandle::eval`'s `sigsetjmp` call
/// site can tell "a genuine native fault was recovered" (1) apart from "the
/// guest hit an unhandled DNU/`error:` and was recovered" (2) without
/// consulting anything but the return value itself.
pub(crate) const GUEST_FATAL_JMP_VAL: core::ffi::c_int = 2;

/// Ordinary (non-signal) code only — is a `sigsetjmp` recovery point
/// registered for the CURRENT thread? `runtime::error::dnu_fallback` and
/// `primitives::prim_error` call this BEFORE building a PROBE dossier: a
/// dossier (register/frame/heap dump) is the right response to a condition
/// that's about to genuinely end the process, and pure noise for a routine,
/// interactively-recovered DNU or `error:` in an embedded `VmHandle` —
/// skip it precisely when a recovery is actually about to happen.
pub(crate) fn has_registered_jmp_slot() -> bool {
    lookup_jmp_slot_for_current_thread().is_some()
}

/// The recoverable half of what used to be an unconditional
/// `runtime::vm_state::fatal_exit` call at the end of `dnu_fallback`/
/// `prim_error`: an unhandled DNU or explicit `self error:` is a genuinely
/// terminal condition for the CURRENT computation in Smalltalk's own terms
/// (`error:` "has no proceed semantics in v1", `prim_error`'s own doc) —
/// but it is NOT a sign anything about the VM itself is broken, unlike the
/// conditions `FatalMode`/`fatal_exit` exist for (heap exhaustion, stack
/// overflow, a genuine native fault). Tearing down the whole worker thread
/// (`FatalMode::ExitThread`) for an everyday Workspace typo — the single
/// most common "mistake" in interactive use — is the wrong tool: it works,
/// but every DNU pays a full VM respawn instead of an ordinary recoverable
/// error, exactly the "any mistake kills the Workspace" experience real
/// Smalltalk's own recoverable `doesNotUnderstand:` was designed to avoid.
///
/// If this thread has a registered recovery slot (i.e. it is inside
/// `VmHandle::eval`'s own `sigsetjmp`), publish `message` and jump straight
/// back there with [`GUEST_FATAL_JMP_VAL`] — same `siglongjmp` mechanism
/// `sig_fault_handler` already uses for a genuine foreign fault, chosen for
/// the identical reason: it is a raw register/SP restore, not a Rust
/// unwind, so it is sound to cross however many interpreted AND compiled
/// frames sit between here and `eval`'s call site (the standing rule this
/// project enforces elsewhere: never `catch_unwind` through JIT frames).
/// No slot registered (plain CLI/batch use, `VmHandle` never booted) falls
/// back to today's `fatal_exit` unchanged — this only changes behavior for
/// the embedded case that never had a sound recovery path before.
#[allow(unsafe_code)]
pub(crate) fn raise_guest_fatal(message: String) -> ! {
    if let Some(i) = lookup_jmp_slot_for_current_thread() {
        GUEST_FATAL_MSG.lock().unwrap()[i] = Some(message);
        unsafe {
            siglongjmp(
                JMP_BUFS[i][0].as_ptr() as *mut core::ffi::c_int,
                GUEST_FATAL_JMP_VAL,
            );
        }
    }
    crate::runtime::vm_state::fatal_exit(1);
}

/// Reads back (and clears) the message [`raise_guest_fatal`] published for
/// the CURRENT thread just before its `siglongjmp` — ordinary code only,
/// called from `VmHandle::eval` right after `sigsetjmp` returns
/// [`GUEST_FATAL_JMP_VAL`]. Same same-thread program-order reasoning as
/// [`take_last_crash_info`].
pub(crate) fn take_last_guest_fatal_message() -> Option<String> {
    let i = lookup_jmp_slot_for_current_thread()?;
    GUEST_FATAL_MSG.lock().unwrap()[i].take()
}

/// §4.5's backstop: `true` from the moment a fault is claimed for PROBE
/// until the process dies. A SECOND fault while the dossier runs (the
/// dossier itself dereferencing something the validation missed) restores
/// `SIG_DFL` and dies with the original signal — the per-step-flushed
/// dossier prefix survives on stderr.
#[cfg_attr(windows, allow(dead_code))] // WINVM: read only by the macOS handlers until the Phase-2 VEH
static PROBE_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// The dedicated stack `rt_probe_crash` runs on. A SEGV frame's own sp may
/// be garbage or exhausted; the probe trampoline unconditionally switches
/// here, which also makes stack-overflow crashes reportable. Never freed —
/// the dossier ends in exit(70).
const PROBE_STACK_BYTES: usize = 512 * 1024;
struct ProbeStack(core::cell::UnsafeCell<[u8; PROBE_STACK_BYTES]>);
// SAFETY: written only as the machine stack of the single post-crash
// dossier path (guarded by PROBE_IN_PROGRESS); never concurrently accessed.
unsafe impl Sync for ProbeStack {}
static PROBE_STACK: ProbeStack = ProbeStack(core::cell::UnsafeCell::new([0; PROBE_STACK_BYTES]));

/// The alternate SIGNAL stack (sigaltstack) — without it `SA_ONSTACK` is
/// inert (nothing in the tree ever installed one before DBG0) and a SIGSEGV
/// from native-stack exhaustion could not deliver at all.
///
/// **Per-thread, not shared (CG0).** `sigaltstack` is a PER-THREAD kernel
/// resource: when `SA_ONSTACK` delivers a signal the kernel runs THIS thread's
/// handler on whatever buffer THIS thread last registered. A single shared
/// static (the pre-CG0 shape) meant every thread's `sigaltstack` aliased the
/// same memory, so two threads faulting at once ran their two handlers on the
/// same buffer and corrupted each other's frames — tolerable while only one VM
/// ran per process, but the worker fleet and the Cocoa GUI (a primary VM on a
/// background thread + a UI callback on main) make concurrent faults *likely*.
/// Each thread now owns its buffer: a boxed array in a `thread_local`,
/// allocated lazily the first time this thread arms the handler, at a stable
/// heap address the kernel holds for the thread's life. 256 KiB is far above
/// macOS arm64's `MINSIGSTKSZ` (32 KiB), leaving ample room for the handler's
/// own frame.
///
/// Lazy `thread_local` init can allocate, which is NOT async-signal-safe — but
/// [`arm_this_threads_altstack`] runs only from [`arm_foreign_fault_handler`]/
/// [`install`], i.e. in ordinary context at boot/arm time, strictly before any
/// fault; the signal handler itself never touches this `thread_local`.
///
/// Teardown, stated honestly (CG0 review): the buffer lives for the thread's
/// active life, but the two thread-exit paths dispose of it differently —
/// **neither is a safety hazard**, because by the time either runs no VM/JIT
/// code executes on the thread, so no synchronous fault can originate there.
///   - `pthread_exit` (the `FatalMode::ExitThread` fail-fast path): TLS
///     destructors are skipped, so the 256 KiB box is **leaked, not freed** —
///     a real but bounded cost (one leak per die→respawn of a throwaway
///     worker, `feedback_recover_clean_or_die`; the Cocoa UI worker's restart
///     path (CG7) drops its `VmHandle` cleanly and does not take this route).
///   - a clean thread exit: the TLS destructor DOES free the box, and the
///     kernel's per-thread `sigaltstack` registration transiently still points
///     at freed-but-still-mapped memory until the thread is fully torn down —
///     a benign window (no signal can arrive there; freed heap is not
///     unmapped), not a live-use hazard.
const ALT_STACK_BYTES: usize = 256 * 1024;
thread_local! {
    static ALT_STACK: core::cell::RefCell<Option<Box<[u8; ALT_STACK_BYTES]>>> =
        const { core::cell::RefCell::new(None) };
}

/// Registers THIS thread's own sigaltstack, allocating its buffer on the first
/// call on this thread (see [`ALT_STACK`]). Ordinary-context only — never call
/// from inside a signal handler (the lazy `thread_local` init may allocate).
/// Idempotent: re-arming a thread just points `sigaltstack` at the same buffer.
fn arm_this_threads_altstack() {
    let sp = ALT_STACK.with(|cell| {
        cell.borrow_mut()
            .get_or_insert_with(|| Box::new([0u8; ALT_STACK_BYTES]))
            .as_mut_ptr()
    });
    // SAFETY: `sp` points at this thread's own live, thread-local-owned buffer
    // of exactly `ALT_STACK_BYTES`, valid for as long as this thread runs any
    // code that could fault (see [`ALT_STACK`] for the exit-teardown analysis);
    // the `stack_t` is fully zeroed then fully initialized.
    #[cfg(unix)]
    unsafe {
        let mut alt: libc::stack_t = core::mem::zeroed();
        alt.ss_sp = sp as *mut core::ffi::c_void;
        alt.ss_size = ALT_STACK_BYTES;
        alt.ss_flags = 0;
        let rc = libc::sigaltstack(&alt, core::ptr::null_mut());
        assert_eq!(rc, 0, "deopt_trap: sigaltstack failed");
    }
    // WINVM: exception dispatch on Windows runs on the faulting thread's own
    // stack (a VEH needs no alternate stack); the buffer is kept so the call
    // stays idempotent-shaped, but nothing registers it.
    #[cfg(windows)]
    let _ = sp;
}

/// In-handler capture of the whole integer register file (+ fault address
/// and signal number). Relaxed stores: the reader runs strictly after
/// sigreturn on the same thread.
///
/// # Safety
/// `ss`/`far` come from the kernel-provided ucontext of the CURRENT
/// delivery.
#[cfg(target_os = "macos")]
unsafe fn capture_regs(ss: &ArmThreadState64, far: u64, sig: i32) {
    for (slot, &v) in CAPTURED.iter().zip(ss.__x.iter()) {
        slot.store(v, Ordering::Relaxed);
    }
    CAPTURED[CAP_FP].store(ss.__fp, Ordering::Relaxed);
    CAPTURED[CAP_LR].store(ss.__lr, Ordering::Relaxed);
    CAPTURED[CAP_SP].store(ss.__sp, Ordering::Relaxed);
    CAPTURED[CAP_PC].store(ss.__pc, Ordering::Relaxed);
    CAPTURED[CAP_CPSR].store(ss.__cpsr as u64, Ordering::Relaxed);
    CAPTURED[CAP_FAR].store(far, Ordering::Relaxed);
    CAPTURED[CAP_SIG].store(sig as u64, Ordering::Relaxed);
}

/// The post-sigreturn read of [`capture_regs`]'s snapshot.
fn read_captured() -> crate::runtime::probe::CapturedRegs {
    let mut r = crate::runtime::probe::CapturedRegs::default();
    for (dst, slot) in r.x.iter_mut().zip(CAPTURED.iter()) {
        *dst = slot.load(Ordering::Relaxed);
    }
    r.fp = CAPTURED[CAP_FP].load(Ordering::Relaxed);
    r.lr = CAPTURED[CAP_LR].load(Ordering::Relaxed);
    r.sp = CAPTURED[CAP_SP].load(Ordering::Relaxed);
    r.pc = CAPTURED[CAP_PC].load(Ordering::Relaxed);
    r.cpsr = CAPTURED[CAP_CPSR].load(Ordering::Relaxed) as u32;
    r.far = CAPTURED[CAP_FAR].load(Ordering::Relaxed);
    r.sig = CAPTURED[CAP_SIG].load(Ordering::Relaxed) as i32;
    r
}

/// `true` once a SIGTRAP handler has been armed (by the first `install`).
/// Arming is process-global and idempotent — later caches only add registry
/// entries, they do not re-`sigaction`.
static HANDLER_ARMED: AtomicBool = AtomicBool::new(false);

/// Register one code cache's `[lo, hi)` range + its own trampolines. Reuses a
/// retired slot or dedups an existing same-`lo` entry (a re-install of a live
/// cache) before appending; panics only if `REGISTRY_CAP` distinct LIVE caches
/// coexist (absurd — deregistration on drop keeps this bounded by peak
/// concurrency, not total caches ever created).
fn register_with_probe(lo: u64, hi: u64, tramp: u64, assert: u64, probe: u64) {
    let _g = REGISTRY_LOCK.lock().unwrap();
    let n = REG_LEN.load(Ordering::Acquire);
    // Prefer an existing same-lo entry (re-install), then any retired slot.
    let slot = (0..n)
        .find(|&i| REG_LO[i].load(Ordering::Acquire) == lo)
        .or_else(|| (0..n).find(|&i| REG_LO[i].load(Ordering::Acquire) == 0));
    let i = match slot {
        Some(i) => i,
        None => {
            assert!(
                n < REGISTRY_CAP,
                "deopt_trap: code-cache registry overflow ({REGISTRY_CAP} live caches)"
            );
            REG_LEN.store(n + 1, Ordering::Release);
            n
        }
    };
    // Fields first, then `lo` (the match key) last with Release.
    REG_HI[i].store(hi, Ordering::Relaxed);
    REG_TRAMP[i].store(tramp, Ordering::Relaxed);
    REG_ASSERT[i].store(assert, Ordering::Relaxed);
    REG_PROBE[i].store(probe, Ordering::Relaxed);
    REG_LO[i].store(lo, Ordering::Release);
}

/// Retire the entry whose base is `lo` (called from `CodeCache::drop`, so a
/// freed/unmapped cache's stale range never misdirects a later trap). A no-op
/// for a cache that was never registered (a non-JIT `VmState`, or a bare test
/// cache).
pub(crate) fn deregister(lo: u64) {
    let _g = REGISTRY_LOCK.lock().unwrap();
    let n = REG_LEN.load(Ordering::Acquire);
    for slot in REG_LO.iter().take(n) {
        if slot.load(Ordering::Acquire) == lo {
            slot.store(0, Ordering::Release);
        }
    }
}

/// Look up the registry entry owning `pc` (async-signal-safe: a fixed-array
/// atomic scan, no lock/alloc). Returns `(tramp, assert)` of the owning cache,
/// or `None` if `pc` is in no registered cache. Reads `lo` with Acquire (the
/// publish key) before the rest of the entry.
#[cfg_attr(windows, allow(dead_code))] // WINVM: consumer is the macOS SIGTRAP handler
fn lookup_pc(pc: u64) -> Option<(u64, u64)> {
    lookup_pc_full(pc).map(|(t, a, _)| (t, a))
}

/// Like [`lookup_pc`] but also returns the owning cache's PROBE trampoline
/// (0 when none was registered — a pre-DBG0 test arm).
#[cfg_attr(windows, allow(dead_code))] // WINVM: consumers are the macOS fault handlers
fn lookup_pc_full(pc: u64) -> Option<(u64, u64, u64)> {
    let n = REG_LEN.load(Ordering::Acquire);
    for i in 0..n {
        let lo = REG_LO[i].load(Ordering::Acquire);
        if lo == 0 {
            continue;
        }
        let hi = REG_HI[i].load(Ordering::Relaxed);
        if pc >= lo && pc < hi {
            return Some((
                REG_TRAMP[i].load(Ordering::Relaxed),
                REG_ASSERT[i].load(Ordering::Relaxed),
                REG_PROBE[i].load(Ordering::Relaxed),
            ));
        }
    }
    None
}

// ── The macOS arm64 ucontext, laid out by hand (D3 step 1) ────────────────
//
// `libc` 0.2 does NOT expose `ucontext_t` / `__darwin_mcontext64` /
// `__darwin_arm_thread_state64` on aarch64-apple-darwin (verified against the
// vendored crate). These `#[repr(C)]` mirrors track Apple's
// `<sys/_types/_ucontext.h>` + `<mach/arm/_structs.h>` (the `_STRUCT_MCONTEXT64`
// / `_STRUCT_ARM_THREAD_STATE64` definitions). We only ever read/write fields
// up through the thread-state block, so the NEON/exception sub-structs are
// left as an opaque tail sized to match the real struct — the handler never
// touches them, and getting their *size* right keeps `mcontext64`'s own size
// honest should anything ever take `size_of` of it (nothing in step 5 does).

/// `_STRUCT_ARM_THREAD_STATE64` — the integer register file at a fault. `__x`
/// is x0..x28; `__fp`/`__lr`/`__sp`/`__pc` are the named specials; `__cpsr`
/// + `__pad` complete the 34-word block.
#[cfg(target_os = "macos")]
#[repr(C)]
struct ArmThreadState64 {
    __x: [u64; 29],
    __fp: u64,
    __lr: u64,
    __sp: u64,
    __pc: u64,
    __cpsr: u32,
    __pad: u32,
}

/// `_STRUCT_ARM_EXCEPTION_STATE64` — 3 words (far/esr/exception). Read never;
/// present only so `__ss` lands at the right offset within `mcontext64`.
#[cfg(target_os = "macos")]
#[repr(C)]
struct ArmExceptionState64 {
    __far: u64,
    __esr: u32,
    __exception: u32,
}

/// `_STRUCT_ARM_NEON_STATE64` — 32×128-bit V registers + fpsr/fpcr. Opaque
/// tail; present only for correct total size. `__v` is `[u128; 32]`; the two
/// trailing u32s are fpsr/fpcr.
#[cfg(target_os = "macos")]
#[repr(C)]
struct ArmNeonState64 {
    __v: [u128; 32],
    __fpsr: u32,
    __fpcr: u32,
}

/// `_STRUCT_MCONTEXT64` — exception state, then thread state (`__ss`, the one
/// we touch), then NEON state.
#[cfg(target_os = "macos")]
#[repr(C)]
struct Mcontext64 {
    __es: ArmExceptionState64,
    __ss: ArmThreadState64,
    __ns: ArmNeonState64,
}

/// `ucontext_t` — only `uc_mcontext` matters here; the leading fields are
/// laid out exactly so that pointer lands correctly. `uc_mcontext` is a
/// *pointer* to the `mcontext64` (Apple stores it out-of-line and points
/// `uc_mcontext` at `__mcontext_data`), which is why the handler double-
/// dereferences: `(*(*ctx).uc_mcontext).__ss.__pc`.
#[cfg(target_os = "macos")]
#[repr(C)]
struct Ucontext {
    uc_onstack: i32,
    uc_sigmask: u32,
    uc_stack: StackT,
    uc_link: *mut Ucontext,
    uc_mcsize: usize,
    uc_mcontext: *mut Mcontext64,
    // Apple appends the inline `__mcontext_data` after this; we never read it
    // (we go through `uc_mcontext`, which points at it), so it is omitted.
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct StackT {
    ss_sp: *mut core::ffi::c_void,
    ss_size: usize,
    ss_flags: i32,
}

// ── The SIGTRAP handler (D3) ──────────────────────────────────────────────

/// SA_SIGINFO handler for SIGTRAP. Async-signal-safe: no allocation, no lock,
/// no formatting, no unwinding across the signal boundary. Its whole job:
///
/// 1. Read the fault pc from `(*(*ctx).uc_mcontext).__ss.__pc`.
/// 2. Bounds-check the pc against the cached code-cache range (two `u64`
///    compares), and decode `*(pc as *const u32)` as one of our deopt `brk`s.
///    The code cache is always readable, so the load is safe.
/// 3. Not ours → restore `SIG_DFL` and return; re-execution kills the process
///    with the default disposition (this keeps Rust `abort()` fatal).
/// 4. `0xDE02` (compiled assert) → resume in the assert stub (which panics
///    off-signal). Still no work in-handler.
/// 5. Otherwise (`0xDE00`/`0xDE01`) → **redirect**: stash the trap pc in x16
///    (IP0 — scratch, never live at a safepoint, `arm64.md` §3) and set
///    `__pc` to the uncommon trampoline. All real work happens after
///    sigreturn, in Rust reached via that trampoline.
///
/// # Safety
/// Installed only via [`install`] as SIGTRAP's `sa_sigaction`; the kernel
/// guarantees `info`/`ctx` are valid for a SIGTRAP delivery. Never called
/// from Rust.
#[cfg(target_os = "macos")]
extern "C" fn sigtrap_handler(_sig: i32, _info: *mut libc::siginfo_t, ctx: *mut core::ffi::c_void) {
    // SAFETY: `ctx` is the kernel-provided `ucontext_t*` for this SIGTRAP; the
    // field layout mirrors Apple's headers (see the struct docs above). We
    // only read/write `__ss` scalars, all valid for the delivery's lifetime.
    unsafe {
        let uc = ctx as *mut Ucontext;
        if uc.is_null() {
            return; // defensive; the kernel never delivers a null context.
        }
        let mc = (*uc).uc_mcontext;
        if mc.is_null() {
            return;
        }
        let ss = &mut (*mc).__ss;
        let pc = ss.__pc;

        // (2) Which registered code cache owns this pc? The owning entry names
        // that cache's OWN (guaranteed-live — the brk fired from inside it)
        // trampoline + assert stub. A pc in NO registered cache is a foreign
        // brk (a Rust abort in ordinary text, etc.) → (3) make it fatal.
        let (tramp, assert) = match lookup_pc(pc) {
            Some(pair) => pair,
            None => {
                restore_default_and_return();
                return;
            }
        };
        // The code cache is always readable — this load cannot fault.
        let word = core::ptr::read(pc as *const u32);
        let imm = match decode_deopt_brk(word) {
            Some(imm) => imm,
            None => {
                // A brk inside a cache we don't recognise (should not happen,
                // but treat as foreign — (3), stay honest/fatal).
                restore_default_and_return();
                return;
            }
        };

        if imm == TRAP_ASSERT {
            // (4) Redirect to the assert stub, which emits the PROBE
            // dossier off-signal (DBG0 — previously it panicked). Capture
            // the register file for the dossier's step 3 and claim the
            // reentrancy guard: from here on the process is dying.
            if assert != 0 {
                capture_regs(ss, 0, 0);
                PROBE_IN_PROGRESS.store(true, Ordering::Release);
                ss.__x[16] = pc;
                ss.__pc = assert;
            } else {
                restore_default_and_return();
            }
            return;
        }

        // (5) 0xDE00 / 0xDE01 — redirect into the owning cache's uncommon
        // trampoline with the trap pc stashed in x16 (IP0). PAC is off, so no
        // signing needed.
        if tramp == 0 {
            restore_default_and_return();
            return;
        }
        ss.__x[16] = pc;
        ss.__pc = tramp;
    }
}

/// D3 step 3: reset SIGTRAP to its default disposition, so that returning from
/// the handler re-executes the faulting instruction and the process dies with
/// the default SIGTRAP behavior. Async-signal-safe (`sigaction` is on the
/// signal-safe list). Kept as a named helper so every "not ours" arm above
/// reads as one intention.
///
/// # Safety
/// Only reached from within [`sigtrap_handler`].
#[cfg(target_os = "macos")]
unsafe fn restore_default_and_return() {
    unsafe { restore_default_for(libc::SIGTRAP) }
}

/// The per-signal generalization DBG0's fault handler needs (it serves both
/// SIGSEGV and SIGBUS).
///
/// # Safety
/// Signal-handler context only.
#[cfg(target_os = "macos")]
unsafe fn restore_default_for(sig: i32) {
    let mut sa: libc::sigaction = unsafe { core::mem::zeroed() };
    sa.sa_sigaction = libc::SIG_DFL;
    // Best-effort: a failure here cannot be reported from signal context, and
    // the re-executed fault will still arrive (just possibly through this
    // handler again — which loops at most until sigaction succeeds; in
    // practice it never fails for SIG_DFL).
    let _ = unsafe { libc::sigaction(sig, &sa, core::ptr::null_mut()) };
}

// ── DBG0: the SIGSEGV/SIGBUS handler (docs/DEBUGGER.md §4.1) ─────────────

/// Async-signal-safe verdict line for a FOREIGN fault (pc outside every
/// registered code cache): raw `write(2)` of a fixed message + hand-rolled
/// hex — no allocation, no formatting machinery, no locks. "That
/// classification itself is the first question of every crash triage."
///
/// `recovering`: `false` reproduces today's exact wording (this thread —
/// and, absent S21, the whole process — is about to die). `true` is S21's
/// new case: an embedded `VmHandle`'s own worker thread has a registered
/// recovery slot (`claim_jmp_slot`) and is about to `siglongjmp` back
/// to it instead — the message says so, so a stderr log never claims
/// "dying" immediately before evidence (a subsequent report over the
/// `TranscriptSink` channel, S21 step 2) that it didn't.
#[cfg(target_os = "macos")]
unsafe fn write_foreign_verdict(sig: i32, pc: u64, far: u64, recovering: bool) {
    fn put(buf: &mut [u8; 128], n: &mut usize, s: &[u8]) {
        for &b in s {
            if *n < buf.len() {
                buf[*n] = b;
                *n += 1;
            }
        }
    }
    fn put_hex(buf: &mut [u8; 128], n: &mut usize, v: u64) {
        put(buf, n, b"0x");
        let digits = b"0123456789abcdef";
        let mut started = false;
        for i in (0..16).rev() {
            let d = ((v >> (i * 4)) & 0xf) as usize;
            if d != 0 || started || i == 0 {
                started = true;
                put(buf, n, &[digits[d]]);
            }
        }
    }
    let mut buf = [0u8; 128];
    let mut n = 0usize;
    put(
        &mut buf,
        &mut n,
        if sig == libc::SIGBUS {
            b"MACVM PROBE: SIGBUS pc "
        } else {
            b"MACVM PROBE: SIGSEGV pc "
        },
    );
    put_hex(&mut buf, &mut n, pc);
    put(&mut buf, &mut n, b" far ");
    put_hex(&mut buf, &mut n, far);
    put(
        &mut buf,
        &mut n,
        if recovering {
            b" FOREIGN (not in any code cache); embedded VM thread recovering\n"
        } else {
            b" FOREIGN (not in any code cache); dying\n"
        },
    );
    let _ = unsafe { libc::write(2, buf.as_ptr() as *const core::ffi::c_void, n) };
}

/// SA_SIGINFO handler for SIGSEGV/SIGBUS (DBG0). Same discipline as
/// [`sigtrap_handler`]: classify, capture, rewrite `__pc`, and do ALL real
/// work after sigreturn. Differences: (a) S21's own foreign-fault recovery
/// check runs FIRST, entirely independent of everything below (see its own
/// comment inline — it must never interact with `PROBE_IN_PROGRESS`, a
/// different, process-wide, never-reset-until-death concern); (b) PROBE's
/// own reentrancy check — a fault while a dossier is already in progress
/// means the dossier itself dereferenced something bad; restore `SIG_DFL`
/// and die, keeping the per-step-flushed prefix; (c) faults are classified
/// by their pc being inside a REGISTERED cache — only there is the
/// x28-is-&VmState convention trustworthy (docs/DEBUGGER.md §4.1);
/// everything else gets the raw-write verdict line and the default fatal
/// disposition.
#[cfg(target_os = "macos")]
extern "C" fn sig_fault_handler(
    sig: i32,
    _info: *mut libc::siginfo_t,
    ctx: *mut core::ffi::c_void,
) {
    // SAFETY: kernel-provided pointers for this delivery; same layout
    // contract as `sigtrap_handler`.
    unsafe {
        // S21 (`docs/SPEC.md` §16): an embedded `VmHandle`'s own worker
        // thread may have registered a recovery slot (`register_and_
        // setjmp`). Checked FIRST and entirely independently of
        // `PROBE_IN_PROGRESS` below — that flag is process-wide and never
        // reset until the process actually dies, which is exactly wrong
        // for this path: a successful recovery must leave a LATER,
        // unrelated fault (on this or another thread, foreign or
        // in-code-cache) completely unaffected, and empirically a second
        // real fault on the SAME thread after one recovery must ALSO
        // recover cleanly (validated in an isolated scratchpad binary
        // before this code was written). Only fires for a GENUINELY
        // foreign pc (`lookup_pc_full` returns no owning cache) — a fault
        // inside a registered code cache, even on an embedded thread,
        // still gets PROBE's real dossier treatment below unchanged (which
        // already correctly terminates via `runtime::vm_state::fatal_exit`
        // rather than a bare `process::exit`, per that function's own
        // `FatalMode` — S21 step 1b).
        if let Some(i) = lookup_jmp_slot_for_current_thread() {
            let uc = ctx as *mut Ucontext;
            if !uc.is_null() {
                let mc = (*uc).uc_mcontext;
                if !mc.is_null() {
                    let far = (*mc).__es.__far;
                    let pc = (*mc).__ss.__pc;
                    let in_code_cache = lookup_pc_full(pc).map(|(_, _, p)| p).unwrap_or(0) != 0;
                    if !in_code_cache {
                        write_foreign_verdict(sig, pc, far, true);
                        JMP_LAST_SIG[i].store(sig as u64, Ordering::Relaxed);
                        JMP_LAST_PC[i].store(pc, Ordering::Relaxed);
                        JMP_LAST_FAR[i].store(far, Ordering::Relaxed);
                        siglongjmp(JMP_BUFS[i][0].as_ptr() as *mut core::ffi::c_int, 1);
                    }
                }
            }
        }

        if PROBE_IN_PROGRESS.swap(true, Ordering::AcqRel) {
            restore_default_for(sig);
            return;
        }
        let uc = ctx as *mut Ucontext;
        if uc.is_null() {
            restore_default_for(sig);
            return;
        }
        let mc = (*uc).uc_mcontext;
        if mc.is_null() {
            restore_default_for(sig);
            return;
        }
        let far = (*mc).__es.__far;
        let ss = &mut (*mc).__ss;
        let pc = ss.__pc;

        let probe = match lookup_pc_full(pc) {
            Some((_, _, probe)) if probe != 0 => probe,
            _ => {
                write_foreign_verdict(sig, pc, far, false);
                restore_default_for(sig);
                return;
            }
        };
        capture_regs(ss, far, sig);
        ss.__x[16] = pc;
        ss.__pc = probe;
    }
}

// ── WINVM: the x86-64 Windows trap layer (Phase 2, MIGRATION.md §2.2) ──────
//
// The macOS layer above rewrites a signal's ucontext; here a process-wide
// Vectored Exception Handler rewrites the exception CONTEXT — same
// classify → capture → redirect discipline, same registry.
//
// **x64 trap-site encoding** (the Phase-3 emitter must honor this): `int3`
// (one byte, 0xCC) followed by the same imm16 the A64 `brk` carries, as two
// little-endian bytes — a 3-byte site whose trailing bytes never execute
// (the handler redirects; the trampoline never returns into the site).
// Windows reports a breakpoint with `CONTEXT.Rip` pointing AT the 0xCC (the
// kernel rewinds the hardware trap), so the imm16 lives at `Rip + 1`.
// x16's redirect role (the stashed trap pc) maps to R10 (MIGRATION.md §2.1
// — volatile, never an argument register, the canonical Win64 scratch).
//
// CONTEXT layout notes (and the missing-Dr4/Dr5 trap) follow JASM's proven
// `rust/src/seh.rs` — including its compile-time `offset_of!(Rip) == 0xF8`
// assertion, which once caught a 16-byte layout drift that landed Rip
// writes in the XMM save area.

/// Decode the bytes at `pc` as one of our x64 deopt trap sites: `0xCC`
/// followed by an imm16 in `TRAP_UNCOMMON..=TRAP_ASSERT`. The x64 sibling
/// of [`decode_deopt_brk`].
///
/// # Safety
/// `pc` must point into a live, readable code region (the handler only
/// calls this after the registry bounds-check).
#[cfg(windows)]
unsafe fn decode_deopt_int3(pc: u64) -> Option<u16> {
    if unsafe { core::ptr::read(pc as *const u8) } != 0xCC {
        return None;
    }
    let imm = unsafe { core::ptr::read_unaligned((pc + 1) as *const u16) };
    if (TRAP_UNCOMMON..=TRAP_ASSERT).contains(&imm) {
        Some(imm)
    } else {
        None
    }
}

/// x64 `CONTEXT` from `winnt.h`, fields through `Rip` (Windows writes the
/// rest; we never construct one — the kernel hands us a `*mut`). AMD64 has
/// no Dr4/Dr5: the debug-register block is six qwords, not eight.
#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct WinContext {
    _p_home: [u64; 6],
    ContextFlags: u32,
    MxCsr: u32,
    _seg: [u16; 6],
    EFlags: u32,
    _dr: [u64; 6],
    Rax: u64,
    Rcx: u64,
    Rdx: u64,
    Rbx: u64,
    Rsp: u64,
    Rbp: u64,
    Rsi: u64,
    Rdi: u64,
    R8: u64,
    R9: u64,
    R10: u64,
    R11: u64,
    R12: u64,
    R13: u64,
    R14: u64,
    R15: u64,
    Rip: u64,
    // (tail omitted — XMM save area, vector state)
}

// The load-bearing layout check (see the section comment above).
#[cfg(windows)]
const _: () = assert!(core::mem::offset_of!(WinContext, Rip) == 0xF8);

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct WinExceptionRecord {
    ExceptionCode: u32,
    ExceptionFlags: u32,
    ExceptionRecord: *mut WinExceptionRecord,
    ExceptionAddress: *mut core::ffi::c_void,
    NumberParameters: u32,
    _pad: u32,
    ExceptionInformation: [usize; 15],
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct WinExceptionPointers {
    ExceptionRecord: *mut WinExceptionRecord,
    ContextRecord: *mut WinContext,
}

#[cfg(windows)]
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
#[cfg(windows)]
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
#[cfg(windows)]
const STATUS_BREAKPOINT: u32 = 0x8000_0003;

// The fault codes PROBE claims, the Windows counterpart of the macOS
// layer's `SIGSEGV`/`SIGBUS` pair. A fault whose pc is inside a
// registered code cache is a *compiled-code* fault, and the one place
// the `R15 == &VmState` convention is trustworthy enough to build a
// dossier from (docs/DEBUGGER.md §4.1).
//
// `STATUS_ILLEGAL_INSTRUCTION` earns its place here on x64 specifically:
// it is what a `ud2` placeholder raises, and — more usefully during a
// port — what executing the *wrong architecture's* bytes usually decays
// into. That is not a hypothetical: it is how the JIT failed on Windows
// before the two `install` splits, and it produced no dossier at all.
#[cfg(windows)]
const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
#[cfg(windows)]
const STATUS_IN_PAGE_ERROR: u32 = 0xC000_0006;
#[cfg(windows)]
const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
#[cfg(windows)]
const STATUS_PRIVILEGED_INSTRUCTION: u32 = 0xC000_0096;
#[cfg(windows)]
const STATUS_INTEGER_DIVIDE_BY_ZERO: u32 = 0xC000_0094;
#[cfg(windows)]
const STATUS_STACK_OVERFLOW: u32 = 0xC000_00FD;

#[cfg(windows)]
fn is_probe_fault(code: u32) -> bool {
    matches!(
        code,
        STATUS_ACCESS_VIOLATION
            | STATUS_IN_PAGE_ERROR
            | STATUS_ILLEGAL_INSTRUCTION
            | STATUS_PRIVILEGED_INSTRUCTION
            | STATUS_INTEGER_DIVIDE_BY_ZERO
            | STATUS_STACK_OVERFLOW
    )
}

/// A short, human-readable name for a fault code — so the verdict line
/// says `ACCESS_VIOLATION` rather than `0xc0000005`, which matters most
/// at exactly the moment someone is reading it in a hurry.
#[cfg(windows)]
fn fault_name(code: u32) -> &'static [u8] {
    match code {
        STATUS_ACCESS_VIOLATION => b"ACCESS_VIOLATION",
        STATUS_IN_PAGE_ERROR => b"IN_PAGE_ERROR",
        STATUS_ILLEGAL_INSTRUCTION => b"ILLEGAL_INSTRUCTION",
        STATUS_PRIVILEGED_INSTRUCTION => b"PRIVILEGED_INSTRUCTION",
        STATUS_INTEGER_DIVIDE_BY_ZERO => b"INTEGER_DIVIDE_BY_ZERO",
        STATUS_STACK_OVERFLOW => b"STACK_OVERFLOW",
        STATUS_BREAKPOINT => b"BREAKPOINT",
        _ => b"FAULT",
    }
}

/// Raw `WriteFile` to stderr — the Windows counterpart of the macOS
/// layer's `libc::write`, and used for the same reason: this runs inside
/// an exception handler, where `println!` may allocate, take a lock the
/// faulting thread already holds, or reenter the very code that faulted.
#[cfg(windows)]
fn raw_stderr(bytes: &[u8]) {
    extern "system" {
        fn GetStdHandle(n: u32) -> *mut core::ffi::c_void;
        fn WriteFile(
            h: *mut core::ffi::c_void,
            buf: *const u8,
            len: u32,
            written: *mut u32,
            overlapped: *mut core::ffi::c_void,
        ) -> i32;
    }
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    let mut written = 0u32;
    // SAFETY: a plain write of a borrowed buffer to an inherited handle.
    unsafe {
        let h = GetStdHandle(STD_ERROR_HANDLE);
        WriteFile(h, bytes.as_ptr(), bytes.len() as u32, &mut written, core::ptr::null_mut());
    }
}

/// The one-line verdict for a fault PROBE will not build a dossier for,
/// written with no allocation. Mirrors the macOS `write_foreign_verdict`.
///
/// This line is the whole diagnostic when a fault lands outside every
/// registered cache, so it carries the three facts that decide what to do
/// next: what faulted, where, and what address it touched.
#[cfg(windows)]
fn write_foreign_verdict_win(code: u32, pc: u64, far: u64, in_cache: bool) {
    fn put(buf: &mut [u8; 160], n: &mut usize, s: &[u8]) {
        for &b in s {
            if *n < buf.len() {
                buf[*n] = b;
                *n += 1;
            }
        }
    }
    fn put_hex(buf: &mut [u8; 160], n: &mut usize, v: u64) {
        put(buf, n, b"0x");
        let digits = b"0123456789abcdef";
        let mut started = false;
        for i in (0..16).rev() {
            let d = ((v >> (i * 4)) & 0xf) as usize;
            if d != 0 || started || i == 0 {
                started = true;
                put(buf, n, &[digits[d]]);
            }
        }
    }
    let mut buf = [0u8; 160];
    let mut n = 0usize;
    put(&mut buf, &mut n, b"MACVM PROBE: ");
    put(&mut buf, &mut n, fault_name(code));
    put(&mut buf, &mut n, b" pc ");
    put_hex(&mut buf, &mut n, pc);
    put(&mut buf, &mut n, b" addr ");
    put_hex(&mut buf, &mut n, far);
    put(
        &mut buf,
        &mut n,
        if in_cache {
            // In-cache but no dossier: either a second fault while one was
            // already being built (the dossier itself is broken), or a
            // cache registered without a probe trampoline.
            b" IN CODE CACHE but no dossier available; dying\n"
        } else {
            b" FOREIGN (not in any code cache); dying\n"
        },
    );
    raw_stderr(&buf[..n]);
}

/// Claim a compiled-code fault for PROBE: capture the register file and
/// redirect `Rip` into this cache's probe trampoline, which switches to a
/// dedicated stack and builds the dossier in ordinary code.
///
/// Returns `EXCEPTION_CONTINUE_SEARCH` for anything it cannot own, which
/// leaves debuggers and the default handler seeing the fault exactly as
/// they would have.
///
/// # Safety
/// `ctx` is the kernel's context record for this delivery.
#[cfg(windows)]
unsafe fn handle_win_fault(ctx: &mut WinContext, rec: &WinExceptionRecord) -> i32 {
    let code = rec.ExceptionCode;
    let pc = ctx.Rip;
    // For an access violation, parameter[1] is the address touched. Other
    // codes carry nothing meaningful there, so report 0 rather than a
    // number that would read as a real address.
    let far = if matches!(code, STATUS_ACCESS_VIOLATION | STATUS_IN_PAGE_ERROR)
        && rec.NumberParameters >= 2
    {
        rec.ExceptionInformation[1] as u64
    } else {
        0
    };

    // Reentrancy: a fault while a dossier is already being built means the
    // dossier ITSELF dereferenced something bad. Do not recurse — let the
    // process die, keeping whatever prefix was already flushed. (The
    // per-step flushing in `rt_probe_crash` exists precisely so that
    // prefix is worth something.)
    if PROBE_IN_PROGRESS.load(Ordering::Acquire) {
        write_foreign_verdict_win(code, pc, far, true);
        return EXCEPTION_CONTINUE_SEARCH;
    }

    let probe = match lookup_pc_full(pc) {
        Some((_, _, p)) if p != 0 => p,
        // Outside every registered cache — the `R15 == &VmState`
        // convention does not hold, so there is nothing trustworthy to
        // build a dossier from. One honest line, then the default
        // disposition.
        other => {
            write_foreign_verdict_win(code, pc, far, other.is_some());
            return EXCEPTION_CONTINUE_SEARCH;
        }
    };

    // SAFETY: kernel-provided context for this delivery.
    unsafe { capture_regs_win(ctx, far, code) };
    PROBE_IN_PROGRESS.store(true, Ordering::Release);
    ctx.R10 = pc;
    ctx.Rip = probe;
    EXCEPTION_CONTINUE_EXECUTION
}

/// In-handler capture of the x64 integer register file into [`CAPTURED`] —
/// the Windows sibling of [`capture_regs`]. Slot mapping: `x[0..16]` hold
/// the GPRs in canonical x64 numbering (RAX, RCX, RDX, RBX, RSP, RBP, RSI,
/// RDI, R8..R15); the named specials mirror their roles (`fp`=RBP,
/// `sp`=RSP, `pc`=RIP, `cpsr`=EFLAGS; `lr`=0, x64 has no link register).
/// `sig` carries the NT status code's low bits — no POSIX signo exists here.
#[cfg(windows)]
unsafe fn capture_regs_win(ctx: &WinContext, far: u64, code: u32) {
    let gprs = [
        ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx, ctx.Rsp, ctx.Rbp, ctx.Rsi, ctx.Rdi, ctx.R8, ctx.R9,
        ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15,
    ];
    for (slot, &v) in CAPTURED.iter().zip(gprs.iter()) {
        slot.store(v, Ordering::Relaxed);
    }
    CAPTURED[CAP_FP].store(ctx.Rbp, Ordering::Relaxed);
    CAPTURED[CAP_LR].store(0, Ordering::Relaxed);
    CAPTURED[CAP_SP].store(ctx.Rsp, Ordering::Relaxed);
    CAPTURED[CAP_PC].store(ctx.Rip, Ordering::Relaxed);
    CAPTURED[CAP_CPSR].store(ctx.EFlags as u64, Ordering::Relaxed);
    CAPTURED[CAP_FAR].store(far, Ordering::Relaxed);
    CAPTURED[CAP_SIG].store(code as u64, Ordering::Relaxed);
}

/// The VEH — the Windows sibling of [`sigtrap_handler`], same steps:
/// classify the pc against the registry, decode the trap site, and
/// redirect `Rip` into the owning cache's trampoline with the trap pc
/// stashed in R10. Everything else returns `CONTINUE_SEARCH` so debuggers
/// and the default handler still see foreign exceptions — including a
/// foreign `int3` (a debugger's own breakpoint, a Rust abort), which the
/// macOS layer had to actively restore `SIG_DFL` for; VEH pass-through
/// gives that behavior for free.
///
/// # Safety
/// Installed only via `AddVectoredExceptionHandler`; the kernel guarantees
/// `info` and both records are valid for this delivery.
#[cfg(windows)]
unsafe extern "system" fn veh_trap_handler(info: *mut WinExceptionPointers) -> i32 {
    unsafe {
        if info.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let rec = (*info).ExceptionRecord;
        let ctx = (*info).ContextRecord;
        if rec.is_null() || ctx.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let code = (*rec).ExceptionCode;
        if is_probe_fault(code) {
            // A compiled-code fault: PROBE's territory, not the deopt
            // path's. Handled entirely separately below.
            return unsafe { handle_win_fault(&mut *ctx, &*rec) };
        }
        if code != STATUS_BREAKPOINT {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let pc = (*ctx).Rip;
        let Some((tramp, assert)) = lookup_pc(pc) else {
            return EXCEPTION_CONTINUE_SEARCH; // foreign int3 — not ours
        };
        let Some(imm) = decode_deopt_int3(pc) else {
            return EXCEPTION_CONTINUE_SEARCH; // in-cache but not a deopt site
        };

        if imm == TRAP_ASSERT {
            if assert != 0 {
                capture_regs_win(&*ctx, 0, STATUS_BREAKPOINT);
                PROBE_IN_PROGRESS.store(true, Ordering::Release);
                (*ctx).R10 = pc;
                (*ctx).Rip = assert;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }

        if tramp == 0 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        (*ctx).R10 = pc;
        (*ctx).Rip = tramp;
        EXCEPTION_CONTINUE_EXECUTION
    }
}

/// WINVM test hook: register `[lo, hi)` → `tramp` and arm the VEH, so a
/// test can drive a real emitted trap site through the real handler
/// without a full `CodeCache`/`VmState`. The x64 counterpart of
/// [`test_arm_handler`], and `pub(crate)` rather than module-private
/// because the compiler-side test that matters (`emit_x64`'s
/// `emitted_uncommon_trap_round_trips_through_the_veh`) lives in another
/// module — that seam between emitter and handler is exactly what needs
/// covering. Retire the entry with [`deregister`].
#[cfg(all(test, windows))]
pub(crate) fn test_register_range(lo: u64, hi: u64, tramp: u64) {
    register_with_probe(lo, hi, tramp, 0, 0);
    if !HANDLER_ARMED.swap(true, Ordering::AcqRel) {
        arm_veh();
    }
}

/// Arm the process-wide VEH (first in the handler chain). Idempotence is
/// the caller's job — [`install`] guards with [`HANDLER_ARMED`], exactly
/// like the macOS `sigaction` arm.
#[cfg(windows)]
fn arm_veh() {
    extern "system" {
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: unsafe extern "system" fn(*mut WinExceptionPointers) -> i32,
        ) -> *mut core::ffi::c_void;
    }
    // SAFETY: registers a well-formed handler; the returned cookie is only
    // needed for removal, which never happens (armed for process life).
    let h = unsafe { AddVectoredExceptionHandler(1, veh_trap_handler) };
    assert!(!h.is_null(), "deopt_trap: AddVectoredExceptionHandler failed");
}

// ── WINVM: the x86-64 deopt trampolines (Phase 3, MIGRATION.md §8) ────────
//
// The three landing pads the VEH redirects into. They complete the loop
// Phase 2 opened: the handler classifies a trap and rewrites `Rip`, and
// these run the actual work off-handler, in ordinary code.
//
// **The trap pc arrives in `R10`**, stashed by `veh_trap_handler` — the
// x16 role on AArch64.
//
// **There is no link register, and that changes the frame shape.** The
// AArch64 trampoline does `mov x30, x16` and then `stp x29, x30` so the
// frame's return-address slot records the TRAP PC (which is what makes a
// bridged frame walkable). On x64 the same layout is built by pushing the
// trap pc first and the frame pointer second: `[rbp] = trapped frame's
// RBP`, `[rbp + 8] = trap pc`, identical to what the walker expects.
//
// **The uncommon trampoline REPLACES the frame it traps in.** After
// `rt_uncommon_trap` has materialized and run the interpreter frames, the
// compiled activation is gone, so the trampoline unwinds to the trapped
// frame's own `RBP`, pops it, and returns the result to *that* frame's
// caller — not to the trap site.

/// `deopt_uncommon_trampoline` — the landing for `0xDE00`/`0xDE01`.
#[cfg(windows)]
fn build_uncommon_trampoline_x64() -> crate::compiler::assembler::CodeBlob {
    use crate::compiler::assembler_x64::{imm, mem, r64, X64Assembler, ARG_REGS, R10, R8, RAX, RBP, RSP, VM_STATE};
    let mut a = X64Assembler::new();

    // Build the bridged frame: [rbp] = trapped RBP, [rbp+8] = trap pc.
    a.emit("push", &[r64(R10)]); // trap pc, in the return-address slot
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);

    // Publish it so a GC during materialization can walk across the gap.
    a.emit("mov", &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(RBP)]);
    a.emit("mov", &[r64(RAX), mem(RBP, 8)]);
    a.emit("mov", &[mem(VM_STATE, VMREG_LAST_COMPILED_PC_OFFSET as i64), r64(RAX)]);
    a.emit("mov", &[r64(RAX), imm(KIND_DEOPT_BRIDGE as i64)]);
    a.emit("mov", &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(RAX)]);

    // rt_uncommon_trap(vm, trap_pc, trapped_fp).
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[1]), mem(RBP, 8)]);
    a.emit("mov", &[r64(ARG_REGS[2]), mem(RBP, 0)]);
    let lit = a.literal_u64(
        rt_uncommon_trap as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.emit("sub", &[r64(RSP), imm(32)]); // shadow space
    a.call_far(lit);
    a.emit("add", &[r64(RSP), imm(32)]);

    // Clear the walker record; the bridge is over.
    a.emit("mov", &[r64(R8), imm(0)]);
    a.emit("mov", &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(R8)]);
    a.emit("mov", &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(R8)]);

    // The trapped activation no longer exists: unwind to ITS frame and
    // return the deoptee's result (already in RAX) to its caller.
    a.emit("mov", &[r64(R8), mem(RBP, 0)]);
    a.emit("mov", &[r64(RSP), r64(R8)]);
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);
    a.finish()
}

/// `deopt_assert_stub` — the landing for `0xDE02`, a compiled-code
/// "should not reach". `rt_compiled_assert_failed` never returns; the
/// trailing trap is a backstop in case it somehow does.
#[cfg(windows)]
fn build_assert_stub_x64() -> crate::compiler::assembler::CodeBlob {
    use crate::compiler::assembler_x64::{imm, r64, X64Assembler, ARG_REGS, R10, RBP, RSP, VM_STATE};
    let mut a = X64Assembler::new();
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(R10)]); // trap pc, before call_far clobbers R10
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    let lit = a.literal_u64(
        rt_compiled_assert_failed as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.emit("sub", &[r64(RSP), imm(32)]);
    a.call_far(lit);
    a.emit_bytes(&deopt_int3_bytes(TRAP_ASSERT));
    a.finish()
}

/// `deopt_probe_trampoline` — the landing for a fault whose pc was inside
/// a registered code cache. Switches to a dedicated stack before calling,
/// because the faulting frame's own `RSP` may be exactly what is broken.
#[cfg(windows)]
fn build_probe_trampoline_x64() -> crate::compiler::assembler::CodeBlob {
    use crate::compiler::assembler_x64::{imm, r64, X64Assembler, ARG_REGS, R10, RSP, SCRATCH1, VM_STATE};
    let mut a = X64Assembler::new();
    a.emit("mov", &[r64(ARG_REGS[1]), r64(R10)]); // trap pc first — R10 is about to go
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    let top = {
        let base = PROBE_STACK.0.get() as u64;
        (base + PROBE_STACK_BYTES as u64) & !15
    };
    let stack_lit = a.literal_u64(top, Some(RelocKind::RuntimeAddr));
    a.load_literal(SCRATCH1, stack_lit);
    a.emit("mov", &[r64(RSP), r64(SCRATCH1)]);
    let entry_lit = a.literal_u64(
        rt_probe_crash as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.emit("sub", &[r64(RSP), imm(32)]);
    a.call_far(entry_lit);
    a.emit_bytes(&deopt_int3_bytes(TRAP_ASSERT));
    a.finish()
}

// ── Trampolines (D4 / D6) generated at startup ────────────────────────────

/// Handles to the generated deopt trampolines, published into the same
/// `CodeCache` real nmethods and the SPEC §9 stubs live in. Returned by
/// [`install`] and (in later steps) stashed alongside `Stubs`.
#[derive(Clone, Copy)]
pub struct DeoptTrampolines {
    /// D4: the stub the handler resumes into after a `0xDE00`/`0xDE01` trap.
    pub uncommon: CodeHandle,
    /// D3 step 4: the stub `0xDE02` (compiled assert) redirects to.
    pub assert: CodeHandle,
}

impl DeoptTrampolines {
    pub fn uncommon_addr(&self) -> u64 {
        self.uncommon.base as u64
    }
    pub fn assert_addr(&self) -> u64 {
        self.assert.base as u64
    }
}

/// D4: `deopt_uncommon_trampoline`. Runs right after sigreturn, with the
/// trapped compiled frame intact (FP = its frame) and the trap pc in x16.
///
/// The frame-record trick is load-bearing (D4): before calling into Rust it
/// builds a *walker-visible* frame record whose saved-lr = trap_pc, so while
/// `rt_uncommon_trap` runs (and may GC, S13 step 6+) the S12 stack walker sees
/// the trapped nmethod frame at pc = trap_pc and its oop maps cover it.
///
/// ```text
///   mov  lr, x16                 // saved-lr := trap_pc (lr is dead here)
///   stp  fp, lr, [sp, #-16]!     // push a normal frame record for the walker
///   mov  fp, sp                  //   -> [fp] = trapped_frame_fp, [fp+8] = trap_pc
///   str  fp, [x28, #LAST_FP]     // PUBLISH the record as the walker's anchor
///   str  lr, [x28, #LAST_PC]     //   (task #92 — the record alone is invisible
///   movz x9, #KIND_DEOPT_BRIDGE  //    to walk_frames' anchor-driven start rule)
///   str  x9, [x28, #LAST_KIND]
///   ldr  x2, [fp]                // x2 := fp-of-trapped-frame (= the saved fp)
///   mov  x0, x28                 // &mut VmState (VM-state register, §3)
///   mov  x1, x16                 // trap pc
///   ldr  x16, <rt_uncommon_trap>; blr x16   // Rust; result oop -> x0
///   str  xzr, [x28, #LAST_FP]    // clear the anchor (P9)
///   str  xzr, [x28, #LAST_KIND]
///   ldr  x2, [fp]                // reload trapped_frame_fp (call clobbers x2)
///   mov  sp, x2                  // tear down trampoline record AND trapped frame
///   ldp  fp, lr, [sp], #16       // pop the COMPILED frame's own saved {fp,lr}
///   ret                          // return the deopt result to the native caller
/// ```
///
/// "normative in effect, not exact instruction choice" (D4). The teardown
/// discards BOTH the trampoline record and the whole trapped compiled frame:
/// `sp := trapped_frame_fp` lands on the compiled frame's own saved `{fp,lr}`
/// pair (the native caller's fp + the real return address), which the `ldp`
/// pops so the final `ret` returns the deopt result to the compiled method's
/// original caller (the activation is gone after a deopt). `trapped_frame_fp`
/// is reloaded from the stable `[fp]` record *after* the call (x2 is caller-
/// saved and `rt_uncommon_trap` may clobber it), not carried in a register
/// across it.
///
/// (Historical: under step 5 `rt_uncommon_trap` was an `unimplemented!()`
/// seam and this teardown never executed; step 6 filled it in — it now
/// materializes the frame, runs `interpret_active`, and returns the deopt
/// result, so this epilogue is live on every uncommon trap.)
fn build_uncommon_trampoline() -> crate::compiler::assembler::CodeBlob {
    let mut a = JasmAssembler::new();

    // saved-lr := trap_pc (in x16), so the walker sees pc = trap_pc.
    a.emit("mov", &[x(30), x(16)]);
    // Push { fp, lr(=trap_pc) } and re-root fp — a normal frame record.
    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);
    // PUBLISH the record as the walker's anchor (task #92): the record alone
    // is invisible — `walk_frames`' start rule reads `last_compiled_fp/pc/
    // kind` when the innermost tier crossing is `IntoCompiled`, exactly the
    // state a GC inside `deoptimize_frame`'s materializer allocations (e.g.
    // `alloc_closure` under a block-carrying trap scope) walks from. Without
    // this the walk asserts (frames.rs "no anchor is set") AND, were the
    // assert removed, would skip the trapped frame's oops entirely. Same
    // three writes as `emit_stub_prologue`+`emit_stub_kind_tag`, same x9
    // scratch precedent (registers are dead at a deopt site — S12 spill-all).
    a.emit(
        "str",
        &[x(29), mem(28, VMREG_LAST_COMPILED_FP_OFFSET as i64)],
    );
    a.emit(
        "str",
        &[x(30), mem(28, VMREG_LAST_COMPILED_PC_OFFSET as i64)],
    );
    a.emit("movz", &[x(9), imm(KIND_DEOPT_BRIDGE as i64)]);
    a.emit(
        "str",
        &[x(9), mem(28, VMREG_LAST_COMPILED_KIND_OFFSET as i64)],
    );
    // x2 := the trapped frame's fp (= saved fp we just pushed, = [fp]). This is
    // the FP `rt_uncommon_trap`/materialization reads slots against.
    a.emit("ldr", &[x(2), mem(29, 0)]);
    // Marshal (vm, trap_pc, fp) into x0/x1/x2.
    a.emit("mov", &[x(0), x(28)]); // &mut VmState
    a.emit("mov", &[x(1), x(16)]); // trap pc
                                   // (x2 already holds fp.)
    let lit = a.literal_u64(
        rt_uncommon_trap as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    // CLEAR the anchor (P9 — a stale anchor outliving this frame would let a
    // walker step into freed stack), exactly as `emit_stub_epilogue` does:
    // xzr into fp + kind. Register 31 in a store's data position is the zero
    // register (see `emit_stub_epilogue`'s own note). x0 (the result oop) is
    // untouched.
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_FP_OFFSET as i64)],
    );
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_KIND_OFFSET as i64)],
    );
    // Teardown (D4): reload the trapped frame's fp from the stable record,
    // then set sp to it so the `ldp` pops the COMPILED frame's own {fp,lr} —
    // discarding both the trampoline record and the trapped compiled frame.
    a.emit("ldr", &[x(2), mem(29, 0)]);
    a.emit("mov", &[sp(), x(2)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]);
    a.emit("ret", &[]);

    a.finish()
}

/// D3 step 4: the assert stub `0xDE02` traps redirect to. Trap pc is in x16.
/// Sets up a minimal frame and calls [`rt_compiled_assert_failed`], which
/// panics — so this stub never actually returns. No RootSpill/anchor: the
/// panic tears the process down, so there is nothing for a walker to see.
///
/// ```text
///   stp  fp, lr, [sp, #-16]!
///   mov  fp, sp
///   mov  x0, x28          // &mut VmState
///   mov  x1, x16          // trap pc
///   ldr  x16, <rt_compiled_assert_failed>; blr x16   // -> !
///   brk  #0xDE02          // unreachable belt-and-braces (never executes)
/// ```
fn build_assert_stub() -> crate::compiler::assembler::CodeBlob {
    let mut a = JasmAssembler::new();

    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(16)]); // trap pc
    let lit = a.literal_u64(
        rt_compiled_assert_failed as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    // rt_compiled_assert_failed is `-> !`; this trap is never reached.
    emit_brk(&mut a, TRAP_ASSERT);

    a.finish()
}

/// DBG0 (docs/DEBUGGER.md §4.1): the PROBE trampoline SIGSEGV/SIGBUS
/// redirects resume into. Deliberately NON-destructive — unlike the
/// uncommon trampoline it touches neither the trapped frame's fp nor its
/// sp (the crash scene is evidence, and a SEGV sp may itself be garbage or
/// exhausted): it switches to PROBE's own dedicated static stack, marshals
/// `x0 = x28` (&VmState — trustworthy because the handler only redirects
/// registered-range faults) and the trap pc, and calls the `-> !` dossier
/// entry. The full register file was already captured in-handler.
///
/// ```text
///   mov  x0, x28              // &mut VmState
///   mov  x1, x16              // trap pc
///   ldr  x17, <probe_stack_top>
///   mov  sp, x17              // dedicated, known-good, 16-aligned stack
///   ldr  x16, <rt_probe_crash>; blr x16   // -> ! (dossier + exit 70)
///   brk  #0xDE02              // unreachable belt-and-braces
/// ```
fn build_probe_trampoline() -> crate::compiler::assembler::CodeBlob {
    let mut a = JasmAssembler::new();

    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(16)]); // trap pc
    let top = {
        let base = PROBE_STACK.0.get() as u64;
        (base + PROBE_STACK_BYTES as u64) & !15
    };
    let stack_lit = a.literal_u64(top, Some(RelocKind::RuntimeAddr));
    a.ldr_literal(xr(17), stack_lit);
    a.emit("mov", &[sp(), x(17)]);
    let entry_lit = a.literal_u64(
        rt_probe_crash as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(entry_lit);
    emit_brk(&mut a, TRAP_ASSERT);

    a.finish()
}

// ── Runtime entry points reached from the trampolines ─────────────────────

/// D4/D5: the `extern "C"` entry `deopt_uncommon_trampoline` calls after
/// sigreturn. A THIN FORWARDER across the `codecache → runtime → interpreter`
/// deopt seam (the designed crossing D4/D5 place `rt_uncommon_trap`'s
/// materialization in `runtime`, so it may reach into `runtime`/`interpreter`
/// here even though this module otherwise "must not touch the interpreter"):
///
/// 1. map `trap_pc → owning nmethod` ([`super::CodeTable::find_by_pc`]),
/// 2. build the [`FrameView`] naming the trapped physical frame,
/// 3. hand it to [`crate::runtime::deopt::deoptimize_frame`] — which owns ALL
///    materialization logic (re-resolving the `DeoptState` itself, D5 M0), so
///    no `DeoptState` resolve happens here (avoids resolving twice),
/// 4. run the materialized frame(s) to completion via
///    [`crate::interpreter::interpret_active`] (D5 M8),
/// 5. return the result oop's raw bits — the trampoline epilogue hands them to
///    the trapped method's native caller as if it had returned normally.
///
/// This is an uncommon-trap site (`0xDE00`/`0xDE01`), so `reexecute == true`
/// and `incoming_result` is `None` (the recorded stack holds the re-executing
/// op's inputs — see [`FrameView::incoming_result`]).
///
/// # Safety
/// Only reached via `blr` from `deopt_uncommon_trampoline`; `vm` is `x28`
/// (`&mut VmState`), `trap_pc` the faulting pc, `fp` the trapped compiled
/// frame's FP. Never called directly from Rust.
pub unsafe extern "C" fn rt_uncommon_trap(
    vm: *mut crate::runtime::vm_state::VmState,
    trap_pc: u64,
    fp: u64,
) -> u64 {
    // SAFETY: contract above — `vm` is the `x28` &mut VmState the trampoline
    // forwarded from the compiled call.
    let vm = unsafe { &mut *vm };

    // trap_pc → owning nmethod. A miss is a VM bug (a deopt brk must live
    // inside a published nmethod), so this is a panic, not a graceful path.
    let nm_id = vm.code_table.find_by_pc(trap_pc).unwrap_or_else(|| {
        panic!(
            "rt_uncommon_trap: trap pc {trap_pc:#x} is not inside any published nmethod \
             (a deopt brk must belong to a compiled method)"
        )
    });

    // Materialize the interpreter frame(s) for this trap, then run them to
    // completion. `deoptimize_frame` re-resolves the DeoptState (D5 M0) and
    // repoints `vm.stack.fp` at the innermost frame; `interpret_active`
    // (M8) runs from the resume bci and returns the activation's result.
    let resume = crate::runtime::deopt::deoptimize_frame(
        vm,
        crate::runtime::deopt::FrameView {
            fp: fp as usize,
            pc: trap_pc as usize,
            nm: nm_id,
            incoming_result: None, // uncommon-trap: reexecute site (no call result)
        },
    );
    vm.note_deopt(crate::runtime::vm_state::DeoptReason::Trap); // D7 stats + trace
                                                                // S14 step 9: bridge link so a GC during the nested run can walk PAST the
                                                                // abandoned compiled frame onto its caller chain (see `deopt_bridge_link`).
    let bridge = crate::runtime::frames::deopt_bridge_link(vm, fp);
    vm.tier_links.push(bridge);
    let result = crate::interpreter::interpret_active(vm, resume).raw();
    vm.tier_links.pop();
    // S14 step 8: the trap's re-execution just WARMED whatever IC was cold —
    // count it and, past the limit, recompile this nmethod against the new
    // feedback (the storm-closer). After the nested run so the profile
    // snapshot sees the warmed state.
    crate::runtime::recompile::note_uncommon_trap(vm, nm_id);
    result
}

/// D6 helper: the victim activation's ORIGINAL return pc, looked up (peek, no
/// remove) in `pending_deopts` by the victim's fp. `deopt_return_trampoline`
/// calls this FIRST, before building its walker-visible frame record, so that
/// `[record_fp + 8]` can hold `orig_ret_pc` — a pc inside the victim nm — and a
/// GC during the later `rt_deopt_on_return` call classifies the victim compiled
/// frame at its true return safepoint (its oop map covers exactly that pc, D4).
/// Pure lookup: no allocation, no GC, so it is safe to call before the record
/// exists. A miss is the same VM bug [`rt_deopt_on_return`] panics on (only
/// runs for a §2c-redirected slot); panicking here surfaces it one call
/// earlier. `rt_deopt_on_return` does the actual `remove`.
///
/// # Safety
/// Only reached via `blr` from `deopt_return_trampoline`; `vm` is `x28`, `fp`
/// the victim's FP. Never called directly from Rust (tests aside).
pub unsafe extern "C" fn rt_deopt_return_pc(
    vm: *mut crate::runtime::vm_state::VmState,
    fp: u64,
) -> u64 {
    // SAFETY: contract above.
    let vm = unsafe { &mut *vm };
    vm.pending_deopts
        .get(&(fp as usize))
        .unwrap_or_else(|| {
            panic!(
                "rt_deopt_return_pc: no pending deopt for victim fp {fp:#x} -- the return \
                 trampoline fired for a slot §2c never redirected (VM-consistency bug)"
            )
        })
        .orig_ret_pc as u64
}

/// D6: the `extern "C"` entry `deopt_return_trampoline` (`codecache::stubs`)
/// calls after a callee returns into a redirected saved-LR slot of a
/// `NotEntrant` nmethod activation. The sibling of [`rt_uncommon_trap`] on the
/// RETURN path — the two differ in exactly two inputs to the SAME
/// materialize+run machinery ([`crate::runtime::deopt::deoptimize_frame`] +
/// [`crate::interpreter::interpret_active`], reused UNCHANGED):
///
/// - `pc = orig_ret_pc` (the original return address into the victim, whose
///   `PcDesc` names a `SafepointKind::Call` deopt site — `reexecute == false`
///   by construction, since it's a return address), vs the trap pc; and
/// - `incoming_result = Some(result)` (the completed call's value, which the
///   materializer pushes onto the innermost operand stack, D5 M3.3), vs `None`.
///
/// `fp` is the victim (nm) activation's OWN fp — the key
/// [`crate::runtime::vm_state::VmState::pending_deopts`] was populated under by
/// §2c ([`crate::runtime::frames::redirect_returns_into_nm`]). Consuming that
/// entry yields `PendingDeopt { orig_ret_pc, nm }`; a MISSING entry is a VM bug
/// (the trampoline only runs for a slot §2c redirected, which always inserts
/// the entry first) → panic. Returns the deoptee's result oop raw bits, which
/// the trampoline `ret`s to the victim's OWN caller (the activation is gone).
///
/// # Safety
/// Only reached via `blr` from `deopt_return_trampoline`; `vm` is `x28`
/// (`&mut VmState`), `fp` the victim activation's FP (restored into `x29` by
/// the returning callee's epilogue), `result` the callee's own result oop.
/// Never called directly from Rust (tests aside, which honor the same
/// contract).
pub unsafe extern "C" fn rt_deopt_on_return(
    vm: *mut crate::runtime::vm_state::VmState,
    fp: u64,
    result: u64,
) -> u64 {
    // SAFETY: contract above — `vm` is the `x28` &mut VmState the trampoline
    // forwarded.
    let vm = unsafe { &mut *vm };

    // NLR escaping through a redirected frame: the callee is propagating a
    // non-local-return SENTINEL in x0, NOT a call result — §2c hijacked the
    // exact native return edge (`[callee_fp+8]`) the NLR sentinel rides on (S11
    // step 9). Do NOT deopt: drop the pending entry and hand the sentinel
    // straight back, so the trampoline's teardown `ret`s it to the victim's
    // caller — exactly what the victim's own `emit_nlr_check` at `orig_ret_pc`
    // would have done had the return not been redirected. MUST precede the
    // `Oop::from_raw` below: the sentinel (`0b0110`) is a reserved-tag word that
    // trips `from_raw`'s debug tag check (→ abort across the extern "C" edge).
    if result == crate::oops::layout::NLR_SENTINEL {
        vm.pending_deopts.remove(&(fp as usize));
        return crate::oops::layout::NLR_SENTINEL;
    }

    // Consume the pending deopt for THIS activation (keyed by its own fp). A
    // miss is a VM bug: §2c redirects a slot only after inserting its entry, so
    // the trampoline can never run for an fp with no entry.
    let pending = vm.pending_deopts.remove(&(fp as usize)).unwrap_or_else(|| {
        panic!(
            "rt_deopt_on_return: no pending deopt for victim fp {fp:#x} -- the return \
             trampoline fired for a slot §2c never redirected (VM-consistency bug)"
        )
    });

    // Reuse the step-6/7 materialize+run path EXACTLY (same two calls
    // `rt_uncommon_trap` makes): the ONLY differences are `pc = orig_ret_pc`
    // and `incoming_result = Some(result)` — a call-return (reexecute == false)
    // site, so `deoptimize_frame` pushes the result onto the innermost operand
    // stack (D5 M3.3).
    let resume = crate::runtime::deopt::deoptimize_frame(
        vm,
        crate::runtime::deopt::FrameView {
            fp: fp as usize,
            pc: pending.orig_ret_pc,
            nm: pending.nm,
            incoming_result: Some(Oop::from_raw(result)),
        },
    );
    vm.note_deopt(crate::runtime::vm_state::DeoptReason::Return); // D7 stats + trace
                                                                  // S14 step 9: same walk bridge as `rt_uncommon_trap` (see there).
    let bridge = crate::runtime::frames::deopt_bridge_link(vm, fp);
    vm.tier_links.push(bridge);
    let result = crate::interpreter::interpret_active(vm, resume).raw();
    vm.tier_links.pop();
    result
}

/// D3 step 4 (rerouted by DBG0): the off-signal landing for a `0xDE02`
/// compiled-assertion trap — a compiled-code "should not reach", always a
/// VM bug. Previously this panicked; now it emits the full PROBE dossier
/// (docs/DEBUGGER.md §4.2 — the handler captured the register file on the
/// `TRAP_ASSERT` arm) and exits 70. `-> !`: it never returns to the assert
/// stub.
///
/// # Safety
/// Only reached via `blr` from `build_assert_stub`'s listing; `vm` is `x28`.
/// Never called directly from Rust.
pub unsafe extern "C" fn rt_compiled_assert_failed(
    vm: *mut crate::runtime::vm_state::VmState,
    trap_pc: u64,
) -> ! {
    // SAFETY: contract above — x28 is &mut VmState inside compiled code.
    let vm = unsafe { &mut *vm };
    let mut regs = read_captured();
    regs.pc = trap_pc; // belt-and-braces: the capture's pc IS the trap pc
    eprintln!(
        "compiled-code assertion (brk #0xDE02) at pc {trap_pc:#x} — a 'should not reach' \
         guard fired; this is a VM/compiler bug"
    );
    // SAFETY: vm is the live VmState per this function's own contract.
    unsafe { crate::runtime::probe::crash_dossier(vm, &regs, "brk #0xDE02 (compiled assert)") }
}

/// DBG0: the off-signal landing for a SIGSEGV/SIGBUS whose pc was inside a
/// registered code cache — reached via `build_probe_trampoline` on PROBE's
/// own dedicated stack, with the register file already captured in-handler.
/// Emits the dossier and exits 70.
///
/// # Safety
/// Only reached via `blr` from `build_probe_trampoline`'s listing; `vm` is
/// `x28` as forwarded by the trampoline (trustworthy: the handler only
/// redirects registered-range faults, where the call stub's x28 invariant
/// holds). Never called directly from Rust.
pub unsafe extern "C" fn rt_probe_crash(
    vm: *mut crate::runtime::vm_state::VmState,
    _trap_pc: u64,
) -> ! {
    // SAFETY: contract above.
    let vm = unsafe { &mut *vm };
    let regs = read_captured();
    #[cfg(unix)]
    let trigger = if regs.sig == libc::SIGBUS {
        "SIGBUS (pc in code cache)"
    } else {
        "SIGSEGV (pc in code cache)"
    };
    // WINVM: unreachable until the Phase-2 VEH exists (nothing captures regs
    // or redirects here on Windows), but keep the landing honest.
    #[cfg(windows)]
    let trigger = "native fault (pc in code cache)";
    // SAFETY: vm is the live VmState per this function's own contract.
    unsafe { crate::runtime::probe::crash_dossier(vm, &regs, trigger) }
}

// ── A checked native-stack slot read (layer-boundary helper) ──────────────

/// Read the oop at byte offset `off` from a compiled frame's FP. The single
/// **safe-to-call** door `runtime/deopt.rs` (step 6, a `#![deny(unsafe_code)]`
/// module) uses to touch a raw native-stack slot: the safety table pins that
/// `runtime/deopt` may reach native-stack pointers *only* through this
/// helper. Kept here (in `codecache`, the unsafe island) so the raw deref is
/// licensed where unsafe already lives — the deref is wrapped internally, so
/// the caller (which cannot write `unsafe`) needs no `unsafe` block, exactly
/// what "the single door the deny-unsafe materializer uses" requires.
///
/// A safe `fn` whose CONTRACT is a plain precondition, not an `unsafe`
/// obligation: `fp` must be a live compiled frame's FP and `fp + off` a slot
/// within that frame (guaranteed by a `DeoptState`'s `FrameSlot` offsets,
/// which the compiler emitted for exactly this frame). A caller that violates
/// it corrupts memory the same way an out-of-bounds index would — the door is
/// "safe" in Rust's sense (no `unsafe` at the call site) while still trusting
/// its input, the standard shape for a licensed low-level primitive.
pub fn read_frame_slot(fp: usize, off: i32) -> Oop {
    let addr = (fp as isize + off as isize) as *const u64;
    // SAFETY: contract above — `fp + off` is an 8-byte-aligned live slot.
    Oop::from_raw(unsafe { core::ptr::read(addr) })
}

/// [`read_frame_slot`] for a slot holding a RAW `f64` bit pattern
/// (`ValueLoc::DoubleSlot`, the float fast-path's unboxed temps) — same
/// contract, but returns the bits WITHOUT constructing an `Oop`:
/// `Oop::from_raw`'s debug validation rejects any word whose low bits look
/// like a mark tag, which ~a quarter of genuine doubles do.
pub fn read_frame_slot_raw(fp: usize, off: i32) -> u64 {
    let addr = (fp as isize + off as isize) as *const u64;
    // SAFETY: contract above — `fp + off` is an 8-byte-aligned live slot.
    unsafe { core::ptr::read(addr) }
}

/// Read the live oop at oop-pool index `pool_ix` of `nm` — the second door
/// (alongside [`read_frame_slot`]) the `#![deny(unsafe_code)]` materializer
/// (`runtime/deopt.rs`) uses to reach a raw code-cache word. A `ValueLoc::
/// ConstPool(i)` / scope `method_pool_ix` names an entry in the nmethod's
/// oop pool, whose physical home is `code.base + literal_off + 8*i` — the
/// SAME word the S9 assembler laid down (`compiler::emit::intern_pool` maps
/// IR pool entry `i` 1:1 to assembler `LiteralId(i)`, and the pool is the
/// FIRST thing interned, so IR index == `LiteralId` == this slot) and the
/// SAME word S12's GC keeps current (`CodeTable::oops_do` relocates it in
/// place), so a moving collection never invalidates the index — only the
/// bits behind it, which is exactly why the read must be LIVE (here), never
/// a compile-time snapshot. Kept in `codecache` (the unsafe island) so the
/// raw deref is licensed where unsafe already lives, mirroring
/// [`read_frame_slot`]'s own layer-boundary rationale. Same "safe `fn` with a
/// trusted-input precondition" shape as [`read_frame_slot`].
///
/// Precondition (a `debug_assert` guards it): `literal_off + 8*(pool_ix+1) <=
/// code.len` (the slot lies fully inside the nmethod's live MAP_JIT code
/// region) — guaranteed by a `ValueLoc` the compiler emitted against THIS
/// nmethod's own pool.
pub fn read_pool_oop(nm: &crate::codecache::nmethod::Nmethod, pool_ix: u32) -> Oop {
    let off = nm.literal_off as usize + 8 * pool_ix as usize;
    debug_assert!(
        off + 8 <= nm.code.len,
        "read_pool_oop: pool_ix {pool_ix} (offset {off}) past nmethod {:?}'s code length {}",
        nm.id,
        nm.code.len
    );
    let addr = (nm.code.base as usize + off) as *const u64;
    // SAFETY: contract above — `off` is an 8-byte-aligned live pool slot
    // inside `[code.base, code.base + code.len)`.
    Oop::from_raw(unsafe { core::ptr::read(addr) })
}

// ── install (D3): sigaction once at startup + trampoline generation ───────

/// Arms the SIGSEGV/SIGBUS foreign-fault handler (+ this thread's own
/// sigaltstack) directly, without going through [`install`]. `VmState::
/// with_options` only calls `install` when the JIT is enabled — a pure
/// interpreter never emits a deopt trap, so arming SIGTRAP and publishing
/// trampolines would be pure overhead — but an embedded `VmHandle`
/// (`embed::VmHandle::boot`, S21) needs `sig_fault_handler`'s foreign-fault
/// recovery regardless of JIT mode: the safety directive behind it ("if the
/// language thread dies, the GUI must not die") draws no JIT-mode
/// exception, and `docs/SPEC.md` §16.5 itself requires the GUI's Browser
/// accept path to run with `MACVM_JIT=off` until compiled-tier redefinition
/// lands. Without this, an embedded `JitMode::Off` VM would have NO signal
/// handler armed at all (`install` never runs), so any native fault (e.g.
/// `Alien`'s raw pointer accessors, S20) would kill the whole process.
///
/// Deliberately unconditional — no "arm once" guard, unlike `install`'s
/// `HANDLER_ARMED`: `sigaltstack` is a PER-THREAD kernel resource, so if two
/// embedded `VmHandle`s ever ran on two different worker threads in the
/// same process, each needs its OWN call to register its OWN alt-stack — an
/// "arm once, process-wide" guard would silently leave the second thread
/// with none. Each thread's alt-stack is now its own per-thread buffer
/// ([`ALT_STACK`], CG0), so two threads faulting concurrently no longer race
/// on shared memory — the case the worker fleet and the Cocoa GUI (a primary
/// VM on a background thread + a UI callback on main) make likely, proven by
/// `concurrent_foreign_faults_on_two_threads_each_recover_on_their_own_altstack`.
/// Redundant calls (a JIT-enabled boot, where `install` already armed the
/// same two signals on this same thread; or calling `boot` twice on one
/// thread) are harmless — `arm_this_threads_altstack`/`sigaction` are
/// themselves idempotent, and both paths install the exact same
/// `sig_fault_handler` function pointer.
pub(crate) fn arm_foreign_fault_handler() {
    // Register THIS thread's own sigaltstack (per-thread — CG0). Ordinary
    // context here, so the lazy thread-local buffer alloc is safe.
    arm_this_threads_altstack();
    // WINVM: no fault handler to arm yet — foreign faults keep the OS
    // default disposition until the Phase-2 VEH lands.
    // SAFETY: a fully zeroed, fully initialized `sigaction`, matching the
    // SA_SIGINFO 3-arg ABI `sig_fault_handler` expects — the same shape
    // `install`'s own arm block uses below.
    #[cfg(target_os = "macos")]
    unsafe {
        let mut sf: libc::sigaction = core::mem::zeroed();
        sf.sa_sigaction = sig_fault_handler as *const () as usize;
        sf.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&mut sf.sa_mask);
        let rc = libc::sigaction(libc::SIGSEGV, &sf, core::ptr::null_mut());
        assert_eq!(
            rc, 0,
            "deopt_trap::arm_foreign_fault_handler: sigaction(SIGSEGV) failed"
        );
        let rc = libc::sigaction(libc::SIGBUS, &sf, core::ptr::null_mut());
        assert_eq!(
            rc, 0,
            "deopt_trap::arm_foreign_fault_handler: sigaction(SIGBUS) failed"
        );
    }
}

/// Install the SIGTRAP handler and publish the deopt trampolines into `cache`.
/// Call once at startup, after the code cache exists but before running any
/// compiled code (the same window `stubs::install` runs in).
///
/// Order matters: the trampolines are published and this cache's registry
/// entry is stored *before* `sigaction` arms the handler, so the very first
/// trap can never see a half-installed state. Called once per JIT `VmState`;
/// the handler is armed on the FIRST call only (`HANDLER_ARMED`), every call
/// registers its own cache's range + trampolines (multi-VmState-safe).
///
/// Debugger caveat (D3, documented loudly): under lldb, Mach `EXC_BREAKPOINT`
/// is claimed *before* POSIX SIGTRAP conversion, so every deopt trap stops the
/// debugger. Mach exception ports are the v2 fix; v1 accepts the caveat
/// (README: `MACVM_JIT=off` for lldb sessions).
pub fn install(cache: &mut CodeCache) -> DeoptTrampolines {
    // 1. Generate + publish this cache's own trampolines.
    //
    // WINVM (Phase 3w): arch-selected. Until this split these three were
    // built by the AArch64 emitters on every host, so on Windows the VEH
    // classified a trap correctly and then rewrote `Rip` to point at A64
    // encodings — the handler was right and its landing pad was garbage.
    // Same bug class as `stubs::install` had, and just as invisible: the
    // addresses are real and inside the cache.
    #[cfg(windows)]
    let (uncommon_blob, assert_blob, probe_blob) = (
        build_uncommon_trampoline_x64(),
        build_assert_stub_x64(),
        build_probe_trampoline_x64(),
    );
    #[cfg(not(windows))]
    let (uncommon_blob, assert_blob, probe_blob) = (
        build_uncommon_trampoline(),
        build_assert_stub(),
        build_probe_trampoline(),
    );
    let hu = cache
        .alloc(uncommon_blob.code.len())
        .expect("deopt_trap::install: code cache too small for uncommon trampoline");
    cache.publish(hu, &uncommon_blob);

    let ha = cache
        .alloc(assert_blob.code.len())
        .expect("deopt_trap::install: code cache too small for assert stub");
    cache.publish(ha, &assert_blob);

    let hp = cache
        .alloc(probe_blob.code.len())
        .expect("deopt_trap::install: code cache too small for probe trampoline");
    cache.publish(hp, &probe_blob);

    // 2. Register this cache's range + trampolines so the handler can map a
    //    trapping pc back to THIS cache (retired by `CodeCache::drop`).
    let (lo, hi) = cache.bounds();
    register_with_probe(lo, hi, hu.base as u64, ha.base as u64, hp.base as u64);

    // 3. Register THIS thread's own sigaltstack (per-thread — CG0), on EVERY
    //    call, before the process-global sigaction arming below. Unlike the
    //    handler disposition (armed once, `HANDLER_ARMED`), the alt-stack is a
    //    per-thread kernel resource: a second JIT `VmState` booting on another
    //    thread needs its own, or its SA_ONSTACK handler has no stack to
    //    deliver on (the exact "arm once, process-wide leaves the 2nd thread
    //    with none" trap `arm_foreign_fault_handler`'s doc calls out). SA_ONSTACK
    //    now actually means something — DBG0 installs the sigaltstack (previously
    //    none existed anywhere, making the flag inert).
    arm_this_threads_altstack();

    // 4. Arm the handlers on the first install only (process-global,
    //    idempotent). SA_SIGINFO for the 3-arg form.
    if !HANDLER_ARMED.swap(true, Ordering::AcqRel) {
        // WINVM: the Windows arm is the VEH (see the x64 trap-layer section
        // above) — one process-wide registration, first in the chain.
        #[cfg(windows)]
        arm_veh();
        // SAFETY: both handlers match the SA_SIGINFO 3-arg ABI; armed exactly
        // once (the swap above) with zeroed, fully-initialized structures.
        #[cfg(target_os = "macos")]
        unsafe {
            let mut sa: libc::sigaction = core::mem::zeroed();
            sa.sa_sigaction = sigtrap_handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigemptyset(&mut sa.sa_mask);
            let rc = libc::sigaction(libc::SIGTRAP, &sa, core::ptr::null_mut());
            assert_eq!(rc, 0, "deopt_trap::install: sigaction(SIGTRAP) failed");

            // DBG0: the PROBE fault handlers (docs/DEBUGGER.md §4.1). Armed
            // with the JIT (interpreter-only runs have no code cache, so
            // every fault there is foreign-by-definition — the v2 arming
            // decision recorded in the design doc).
            let mut sf: libc::sigaction = core::mem::zeroed();
            sf.sa_sigaction = sig_fault_handler as *const () as usize;
            sf.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigemptyset(&mut sf.sa_mask);
            let rc = libc::sigaction(libc::SIGSEGV, &sf, core::ptr::null_mut());
            assert_eq!(rc, 0, "deopt_trap::install: sigaction(SIGSEGV) failed");
            let rc = libc::sigaction(libc::SIGBUS, &sf, core::ptr::null_mut());
            assert_eq!(rc, 0, "deopt_trap::install: sigaction(SIGBUS) failed");
        }
    }

    DeoptTrampolines {
        uncommon: hu,
        assert: ha,
    }
}

// ── Test-only hooks: point the handler's redirect at a capture stub ───────
//
// `handler_redirect_smoke` (tests_s13.md) needs to exercise the
// signal→ucontext-rewrite→trampoline path in ISOLATION, WITHOUT reaching the
// real (fully implemented) `rt_uncommon_trap` — a hand-built test frame has
// no genuine deopt record to materialize from. These hooks let
// a test arm the handler + point `UNCOMMON_TRAMPOLINE` at its OWN benign
// capture stub that records (pc, x16) and returns, then restore state.

/// # Safety
/// Test-only. Registers a single `[lo, hi)` range → `uncommon_tramp` and arms
/// the handler, so a test can drive the whole signal path against a hand-built
/// stub. Serialize test use (`#[serial]`-style single-threaded runner) — the
/// registry + SIGTRAP disposition are process-global, shared with the real
/// handler. `test_disarm_handler` restores `SIG_DFL` + retires the entry.
#[cfg(all(test, target_os = "macos"))]
unsafe fn test_arm_handler(lo: u64, hi: u64, uncommon_tramp: u64) {
    register_with_probe(lo, hi, uncommon_tramp, 0, 0);
    // SAFETY: same contract as `install`'s sigaction arm.
    unsafe {
        let mut sa: libc::sigaction = core::mem::zeroed();
        sa.sa_sigaction = sigtrap_handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        let rc = libc::sigaction(libc::SIGTRAP, &sa, core::ptr::null_mut());
        assert_eq!(rc, 0, "test_arm_handler: sigaction failed");
    }
    HANDLER_ARMED.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WINVM M2 gate (MIGRATION.md §4 Phase 2): the x64 twin of
    /// `handler_redirect_smoke`, live end-to-end — a hand-built blob whose
    /// first bytes are the x64 trap site (`int3` + imm16 `0xDE00`), a
    /// capture trampoline (`mov rax, r10 ; ret`) that returns the stashed
    /// trap pc, one registry entry binding them, and the real VEH armed.
    /// Calling the blob must trap, redirect through the VEH, and return the
    /// trap pc — proving classify → decode → R10 stash → Rip rewrite →
    /// resume in one pass.
    ///
    /// Safe under parallel tests: the VEH passes every exception it does
    /// not own straight through (`CONTINUE_SEARCH`), and the registry entry
    /// is retired on exit.
    #[cfg(windows)]
    #[test]
    fn veh_redirect_smoke() {
        use crate::vendor::wfasm::native_windows::WinJit;

        let jit = WinJit::with_capacity(4096).expect("RWX region");
        let (base, _cap) = jit.region_raw();
        let entry = base as u64;
        let tramp = entry + 16;
        // entry: int3 ; .word 0xDE00 ; ret   (the ret is unreachable)
        // tramp: mov rax, r10 ; ret
        // SAFETY: writes land inside the freshly allocated RWX region.
        unsafe {
            core::ptr::copy_nonoverlapping([0xCC, 0x00, 0xDE, 0xC3].as_ptr(), base, 4);
            core::ptr::copy_nonoverlapping([0x4C, 0x89, 0xD0, 0xC3].as_ptr(), base.add(16), 4);
        }
        register_with_probe(entry, entry + 8, tramp, 0, 0);
        if !HANDLER_ARMED.swap(true, Ordering::AcqRel) {
            arm_veh();
        }

        // SAFETY: the blob traps immediately; the VEH redirects to the
        // trampoline, which returns normally with rax = r10 = trap pc.
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(entry) };
        let got = f();
        deregister(entry);
        assert_eq!(
            got, entry,
            "VEH must stash the trap pc in R10 and resume in the trampoline"
        );
    }

    /// **The deopt loop, closed end to end.** A compiled-style frame
    /// executes a trap site; the real VEH decodes it and redirects into
    /// the real uncommon trampoline; the trampoline builds its bridged
    /// frame, publishes it, calls a stand-in `rt_uncommon_trap`, and then
    /// **unwinds the trapped frame entirely** — returning the deoptee's
    /// result to that frame's OWN caller, not to the trap site.
    ///
    /// That last step is what makes this worth a test rather than a
    /// reading: the trampoline replaces a frame it did not create, so a
    /// wrong unwind returns to the wrong place with a plausible-looking
    /// value, and nothing faults.
    ///
    /// The trampoline is built here with a patched-in probe address
    /// instead of the real `rt_uncommon_trap`, so the test observes the
    /// hand-off without needing a live `VmState` and a real nmethod.
    #[cfg(windows)]
    #[test]
    fn uncommon_trampoline_unwinds_the_trapped_frame() {
        use crate::compiler::assembler_x64::{imm, mem, r64, X64Assembler, RBP, RSP, VM_STATE};
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEEN_PC: AtomicU64 = AtomicU64::new(0);
        static SEEN_FP: AtomicU64 = AtomicU64::new(0);
        static SEEN_FP_AT_CALL: AtomicU64 = AtomicU64::new(0);

        // Stands in for rt_uncommon_trap(vm, trap_pc, fp).
        extern "C" fn trap_probe(vm: u64, trap_pc: u64, fp: u64) -> u64 {
            SEEN_PC.store(trap_pc, Ordering::Relaxed);
            SEEN_FP.store(fp, Ordering::Relaxed);
            // The walker record must be live DURING the call.
            let vmreg = vm as *const u64;
            SEEN_FP_AT_CALL.store(
                unsafe { *vmreg.add(VMREG_LAST_COMPILED_FP_OFFSET / 8) },
                Ordering::Relaxed,
            );
            0xD0DE // the "deoptee's result"
        }

        // Build the trampoline, then rewrite its runtime-address pool word
        // to the probe. The pool is at `literal_off`; the RuntimeAddr
        // reloc for rt_uncommon_trap names the exact word.
        let mut tramp = build_uncommon_trampoline_x64();
        let target = tramp
            .relocs
            .iter()
            .find(|r| r.kind == RelocKind::RuntimeAddr)
            .map(|r| r.offset as usize)
            .expect("the trampoline pools rt_uncommon_trap as a RuntimeAddr");
        tramp.code[target..target + 8]
            .copy_from_slice(&(trap_probe as usize as u64).to_le_bytes());

        // A "compiled method" that establishes a frame and then traps —
        // the same prologue emit_x64 generates.
        let mut m = X64Assembler::new();
        m.emit("push", &[r64(RBP)]);
        m.emit("mov", &[r64(RBP), r64(RSP)]);
        m.emit("sub", &[r64(RSP), imm(32)]);
        let trap_at = m.offset();
        m.emit_bytes(&deopt_int3_bytes(TRAP_UNCOMMON));
        // Never reached: if the trampoline returned HERE instead of
        // unwinding, the test would see this value instead of the probe's.
        m.emit("mov", &[r64(crate::compiler::assembler_x64::RAX), imm(0xBAD)]);
        m.emit("mov", &[r64(RSP), r64(RBP)]);
        m.emit("pop", &[r64(RBP)]);
        m.emit("ret", &[]);
        let method = m.finish();

        // Caller: sets R15 (vm) and calls the method.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(VM_STATE)]);
        h.emit("sub", &[r64(RSP), imm(40)]);
        h.emit("mov", &[r64(crate::compiler::assembler_x64::R10), r64(crate::compiler::assembler_x64::RCX)]);
        h.emit("mov", &[r64(VM_STATE), r64(crate::compiler::assembler_x64::RDX)]);
        h.emit("call", &[r64(crate::compiler::assembler_x64::R10)]);
        h.emit("add", &[r64(RSP), imm(40)]);
        h.emit("pop", &[r64(VM_STATE)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();

        let total = tramp.code.len() + method.code.len() + harness.code.len() + 4096;
        let jit = WinJit::with_capacity(total).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let m_off = (tramp.code.len() + 15) & !15;
        let h_off = (m_off + method.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(tramp.code.as_ptr(), base, tramp.code.len());
            core::ptr::copy_nonoverlapping(method.code.as_ptr(), base.add(m_off), method.code.len());
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(h_off),
                harness.code.len(),
            );
        }
        let m_entry = base as u64 + m_off as u64;
        let trap_pc = m_entry + trap_at as u64;

        // Register just the METHOD's range, pointing at the trampoline.
        register_with_probe(
            m_entry,
            m_entry + method.code.len() as u64,
            base as u64,
            0,
            0,
        );
        if !HANDLER_ARMED.swap(true, Ordering::AcqRel) {
            arm_veh();
        }

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        let hf: extern "C" fn(u64, u64) -> u64 =
            unsafe { std::mem::transmute(base.add(h_off)) };
        let got = hf(m_entry, vm);
        deregister(m_entry);

        assert_eq!(
            got, 0xD0DE,
            "the deoptee's result must reach the trapped frame's caller — \
             0xBAD would mean the trampoline returned to the trap site instead"
        );
        assert_eq!(
            SEEN_PC.load(Ordering::Relaxed),
            trap_pc,
            "rt_uncommon_trap receives the trapping pc"
        );
        assert_ne!(SEEN_FP.load(Ordering::Relaxed), 0, "and the trapped frame's fp");
        assert_ne!(
            SEEN_FP_AT_CALL.load(Ordering::Relaxed),
            0,
            "last_compiled_fp must be published DURING the call, so a GC \
             inside materialization can walk across the bridge"
        );
        assert_eq!(
            vmreg[VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0,
            "...and cleared once the bridge is over"
        );
    }

    /// WINVM: the x64 trap-site decoder accepts exactly our three imm16s
    /// behind an `int3` and rejects everything else.
    #[cfg(windows)]
    #[test]
    fn decode_deopt_int3_recognizes_trap_sites() {
        let mk = |bytes: [u8; 3]| {
            let b = Box::leak(Box::new(bytes));
            b.as_ptr() as u64
        };
        // SAFETY: reads within the leaked 3-byte buffers.
        unsafe {
            assert_eq!(decode_deopt_int3(mk([0xCC, 0x00, 0xDE])), Some(TRAP_UNCOMMON));
            assert_eq!(decode_deopt_int3(mk([0xCC, 0x01, 0xDE])), Some(TRAP_STRESS));
            assert_eq!(decode_deopt_int3(mk([0xCC, 0x02, 0xDE])), Some(TRAP_ASSERT));
            assert_eq!(decode_deopt_int3(mk([0xCC, 0x03, 0xDE])), None, "imm past ASSERT");
            assert_eq!(decode_deopt_int3(mk([0x90, 0x00, 0xDE])), None, "not an int3");
        }
    }

    /// S13 step 7a END-TO-END through `rt_uncommon_trap` itself (the codecache
    /// entry the trampoline `blr`s), NOT via the safe runtime wrapper: install
    /// a real nmethod carrying a recorder scope blob + PcDesc, hand-build the
    /// trapped physical frame, and call `rt_uncommon_trap(vm, trap_pc, fp)`
    /// directly — simulating exactly what `deopt_uncommon_trampoline` does
    /// after sigreturn (minus the actual signal, which 7b's signal-driven
    /// differential test adds). Asserts the whole forwarder chain
    /// (`find_by_pc` → `deoptimize_frame` → `interpret_active`) runs the
    /// deoptee to completion and returns its result oop's raw bits.
    ///
    /// Deoptee: `push_smi_i8(0x2A); return_tos`; a re-execute uncommon-trap
    /// site at bci 0 with an empty operand stack, so the nested run starts at
    /// bci 0 and delivers the smi 42.
    #[test]
    fn rt_uncommon_trap_runs_to_completion() {
        use crate::bytecode::BytecodeBuilder;
        use crate::compiler::scopes::{
            CtxLoc, SafepointKind, SafepointState, ScopeDescData, ValueLoc,
        };
        use crate::oops::smi::SmallInt;
        use crate::runtime::deopt::test_support::install_deopt_nmethod;
        use crate::runtime::vm_state::{VmOptions, VmState};

        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        });

        let known: i64 = 0x2A;
        let method = {
            let mut b = BytecodeBuilder::new();
            b.push_smi_i8(known as i8); // bci 0
            b.ret_tos(); // bci 2
            let sel = vm.universe.intern(b"deoptee_trap_e2e");
            b.finish(&mut vm, sel, 0, 0)
        };

        let nil = vm.universe.nil_obj;
        let (_nm_id, pc) = install_deopt_nmethod(
            &mut vm,
            method,
            nil, // pool word 1 (unreferenced here)
            ScopeDescData {
                method_pool_ix: 0, // pool word 0 = the method oop
                is_block: false,
                sender: None,
                receiver: ValueLoc::FrameSlot(-8),
                slots: vec![],
                ctx: CtxLoc::None,
            },
            // Site code offset. `find_by_pc` requires `pc < code.base +
            // code.len`, and the hand-built nmethod's code blob is exactly the
            // 16-byte pool, so keep the trap offset inside it (0x8 < 16); the
            // PcDesc metadata is independent of the pool bytes at that offset.
            0x8,
            SafepointState {
                scope: 0, // placeholder; overridden by install_deopt_nmethod
                bci: 0,
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![],
            },
        );

        // The trapped physical frame: receiver at FrameSlot -8.
        let receiver_val = SmallInt::new(0x11).oop();
        let phys: [u64; 2] = [receiver_val.raw(), 0];
        let fp = (&phys[1]) as *const u64 as usize;

        let sp_before = vm.stack.sp;
        let deopt_before = vm.stats.deopt_count;

        // Drive the codecache entry exactly as the trampoline would.
        // SAFETY: `vm` is a live &mut VmState; `pc` is inside the published
        // nmethod; `fp` names the hand-built physical frame. This is the
        // trampoline's own call, made directly from the test.
        let raw = unsafe { rt_uncommon_trap(&mut vm as *mut VmState, pc as u64, fp as u64) };

        assert_eq!(
            raw,
            SmallInt::new(known).oop().raw(),
            "rt_uncommon_trap returns the deoptee's computed result as raw bits"
        );
        assert_eq!(vm.stats.deopt_count, deopt_before + 1, "one deopt counted");
        assert_eq!(
            vm.stack.sp, sp_before,
            "the nested run unwound back to the pre-trap watermark"
        );
        assert!(
            !vm.stack.has_frame(),
            "the ambient (no-frame) outer activation was restored"
        );
    }

    /// S13 step 9 — `rt_deopt_on_return` END-TO-END, driven directly like
    /// `rt_uncommon_trap_runs_to_completion` (the trampoline's own call, minus
    /// the native trampoline itself). Seeds `pending_deopts[fp]` with a Call
    /// (reexecute == false) deopt site at `orig_ret_pc`, hand-builds the victim
    /// physical frame, and calls `rt_deopt_on_return(vm, fp, result)`. Asserts
    /// it consumes the pending entry, materializes with `incoming_result` pushed
    /// on the operand stack (the deoptee resumes PAST the send), runs to
    /// completion, and returns that result.
    ///
    /// Deoptee: `push_self; send #foo; return_tos`. The site is the RETURN of
    /// `send #foo` (reexecute == false, resume bci PAST the 2-byte send at bci
    /// 1 → bci 3), so the nested run does NOT re-run the send — it resumes at
    /// `return_tos` with `incoming_result` already on the stack and delivers it.
    #[test]
    fn rt_deopt_on_return_runs_to_completion() {
        use crate::bytecode::BytecodeBuilder;
        use crate::compiler::scopes::{
            CtxLoc, SafepointKind, SafepointState, ScopeDescData, ValueLoc,
        };
        use crate::oops::smi::SmallInt;
        use crate::runtime::deopt::test_support::install_deopt_nmethod;
        use crate::runtime::vm_state::{PendingDeopt, VmOptions, VmState};

        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        });

        // `push_self; send #foo; return_tos`.
        let method = {
            let mut b = BytecodeBuilder::new();
            b.push_self(); // bci 0
            let foo = vm.universe.intern(b"foo");
            b.send(&mut vm, foo, 0); // bci 1 (2 bytes) -> resume PAST at bci 3
            b.ret_tos(); // bci 3
            let sel = vm.universe.intern(b"deoptee_return_e2e");
            b.finish(&mut vm, sel, 0, 0)
        };

        let nil = vm.universe.nil_obj;
        let (nm_id, orig_ret_pc) = install_deopt_nmethod(
            &mut vm,
            method,
            nil,
            ScopeDescData {
                method_pool_ix: 0,
                is_block: false,
                sender: None,
                receiver: ValueLoc::FrameSlot(-8),
                slots: vec![],
                ctx: CtxLoc::None,
            },
            // The call site keys on its RETURN address offset (inside the code
            // blob, which is the 16-byte pool). bci names the SEND (bci 1);
            // reexecute == false; empty recorded stack (the result comes from
            // incoming_result, not the recorded stack).
            0x8,
            SafepointState {
                scope: 0, // overridden by install_deopt_nmethod
                bci: 1,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![],
            },
        );

        // The victim (nm-activation) physical frame: receiver at FrameSlot -8.
        let receiver_val = SmallInt::new(0x11).oop();
        let phys: [u64; 2] = [receiver_val.raw(), 0];
        let fp = (&phys[1]) as *const u64 as usize;

        // The completed callee's result — what §2c's return path delivers as
        // incoming_result and the deoptee's `return_tos` then returns.
        let result_val = SmallInt::new(0x5A).oop();

        // Seed the pending deopt exactly as §2c's redirection walk would have.
        vm.pending_deopts.insert(
            fp,
            PendingDeopt {
                orig_ret_pc,
                nm: nm_id,
            },
        );

        let sp_before = vm.stack.sp;
        let deopt_before = vm.stats.deopt_count;

        // Drive the codecache entry exactly as `deopt_return_trampoline` would.
        // SAFETY: `vm` is a live &mut VmState; `fp` names the hand-built victim
        // frame; `result_val` is the callee's result. The trampoline's own call.
        let raw =
            unsafe { rt_deopt_on_return(&mut vm as *mut VmState, fp as u64, result_val.raw()) };

        assert_eq!(
            raw,
            result_val.raw(),
            "rt_deopt_on_return delivers incoming_result (resumed past the send at return_tos)"
        );
        assert_eq!(vm.stats.deopt_count, deopt_before + 1, "one deopt counted");
        assert_eq!(
            vm.stack.sp, sp_before,
            "the nested run unwound back to the pre-deopt watermark"
        );
        assert!(
            !vm.stack.has_frame(),
            "the ambient (no-frame) outer activation was restored"
        );
        assert!(
            vm.pending_deopts.is_empty(),
            "the pending_deopts entry was consumed (removed) by the return-path deopt"
        );
    }

    /// S13 step 9 (NLR fix): an NLR sentinel escaping through a redirected
    /// frame is NOT a call result — `rt_deopt_on_return` hands it straight back
    /// (the trampoline propagates it) instead of materializing a frame from the
    /// reserved-tag word. Without this, `Oop::from_raw(sentinel)` aborts (debug)
    /// or the sentinel is pushed as a bogus operand + the NLR is swallowed
    /// (release).
    #[test]
    fn rt_deopt_on_return_propagates_nlr_sentinel() {
        use crate::codecache::nmethod::NmethodId;
        use crate::runtime::vm_state::{PendingDeopt, VmOptions, VmState};

        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        });
        let fp = 0xF00usize;
        vm.pending_deopts.insert(
            fp,
            PendingDeopt {
                orig_ret_pc: 0x1000,
                nm: NmethodId(0),
            },
        );
        let before = vm.stats.deopt_count;

        // SAFETY: the NLR-sentinel arm returns before dereferencing nm/fp — the
        // dummy PendingDeopt is never read, so no real nmethod/frame is needed.
        let raw = unsafe {
            rt_deopt_on_return(
                &mut vm as *mut VmState,
                fp as u64,
                crate::oops::layout::NLR_SENTINEL,
            )
        };
        assert_eq!(
            raw,
            crate::oops::layout::NLR_SENTINEL,
            "the NLR sentinel is propagated verbatim, never deopted"
        );
        assert_eq!(
            vm.stats.deopt_count, before,
            "no deopt happens for an escaping NLR"
        );
        assert!(
            vm.pending_deopts.is_empty(),
            "the pending entry is dropped on the NLR path too"
        );
    }

    // NB: the "missing `pending_deopts[fp]` panics" guard in `rt_deopt_on_
    // return` / `rt_deopt_return_pc` is deliberately NOT unit-tested — a panic
    // out of an `extern "C"` fn ABORTS (it cannot unwind across the C ABI), so
    // `#[should_panic]` can't observe it. The guard is a documented VM-bug
    // assertion (the trampoline only ever fires for a §2c-redirected slot, which
    // always records the entry first); the safe walker's own analogue
    // (`resolve_redirected_lr`, a plain Rust fn) carries the same invariant.

    /// `brk_imm_decode` (tests_s13.md): the encoder for `brk #0xDE00..02`
    /// round-trips through the handler's decode mask, and foreign brks —
    /// notably Rust's `abort()` (`brk #1`) — are rejected (so SIG_DFL keeps
    /// them fatal).
    #[test]
    fn brk_imm_decode() {
        for imm in [TRAP_UNCOMMON, TRAP_STRESS, TRAP_ASSERT] {
            let word = brk_word(imm);
            // Encoding matches the ISA form `0xD420_0000 | imm16 << 5`.
            assert_eq!(
                word,
                0xD420_0000 | ((imm as u32) << 5),
                "brk #{imm:#x} encoding"
            );
            assert_eq!(decode_deopt_brk(word), Some(imm), "our imm must decode");
        }
        // Rust `abort()` lowers to `brk #1`; must be rejected as foreign.
        assert_eq!(
            decode_deopt_brk(brk_word(1)),
            None,
            "brk #1 (Rust abort) is foreign"
        );
        assert_eq!(decode_deopt_brk(brk_word(0)), None, "brk #0 is foreign");
        // A brk imm adjacent to our range but outside it is foreign.
        assert_eq!(
            decode_deopt_brk(brk_word(0xDDFF)),
            None,
            "0xDDFF just below range"
        );
        assert_eq!(
            decode_deopt_brk(brk_word(0xDE03)),
            None,
            "0xDE03 just above range"
        );
        // A non-brk instruction (a nop) is not a brk at all.
        assert_eq!(decode_deopt_brk(0xD503_201F), None, "nop is not a brk");
        // A `bl` with bits that happen to fall in the imm slot is still not a
        // brk (opcode mask rejects it).
        assert_eq!(decode_deopt_brk(0x9400_0000), None, "bl is not a brk");
    }

    /// The emit helper lays down exactly the raw `brk` word.
    #[test]
    fn emit_brk_lays_raw_word() {
        let mut a = JasmAssembler::new();
        emit_brk(&mut a, TRAP_UNCOMMON);
        let blob = a.finish();
        assert_eq!(&blob.code[..4], &brk_word(TRAP_UNCOMMON).to_le_bytes());
    }

    /// `handler_redirect_smoke` (tests_s13.md): publish a stub that
    /// `brk #0xDE01`s; a test-mode trampoline records (pc, x16) and returns;
    /// assert the trap pc was stashed in x16 and the handler redirected into
    /// the trampoline — i.e. the full signal → ucontext-rewrite → trampoline
    /// path fires in isolation, without reaching the step-6 seam.
    ///
    /// Runs on the CURRENT thread and installs a process-global SIGTRAP
    /// handler for the duration; `#[ignore]` by default so the ordinary
    /// `cargo test` run (which may execute tests in parallel and must not have
    /// a live deopt handler swallowing unrelated traps) stays stable. Run
    /// explicitly with `cargo test -- --ignored handler_redirect_smoke`.
    #[cfg(target_os = "macos")] // WINVM: exercises the macOS SIGTRAP handler
    #[test]
    #[ignore = "installs a process-global SIGTRAP handler; run single-threaded with --ignored"]
    fn handler_redirect_smoke() {
        use std::sync::atomic::AtomicU64;

        // A static the capture trampoline writes the observed x16 (trap pc)
        // into, and a flag proving it ran.
        static CAPTURED_X16: AtomicU64 = AtomicU64::new(0);
        static RAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

        // The capture trampoline the handler will redirect to. It receives the
        // trap pc in x16 (the handler's stash) and, per the smoke test's
        // contract, "records (pc, x16) and returns". It records x16, sets a
        // flag, then unwinds cleanly back to the caller of the brk-stub.
        //
        // The brk-stub below is called as `extern "C" fn() -> ()` and its brk
        // is its very first instruction. When the handler redirects here, the
        // machine state is: lr = the return address into `run_stub`'s caller
        // frame, fp = that caller's fp, x16 = trap pc. So a plain `ret` (after
        // recording x16) returns straight out of the stub as if it had done
        // nothing — no frame was ever built by the brk-stub.
        extern "C" fn capture_and_return() {
            // These statics are plain atomics — the only "work" done, and safe
            // to touch here because control has already left signal context
            // (we are running normal code the handler resumed into).
            // We read x16 via inline read of the stashed value. Since we can't
            // name x16 from Rust portably, the stub was entered with x16 live;
            // we recover it below through a tiny asm shim instead.
            unsafe {
                let x16: u64;
                core::arch::asm!("mov {}, x16", out(reg) x16, options(nomem, nostack, preserves_flags));
                CAPTURED_X16.store(x16, Ordering::SeqCst);
            }
            RAN.store(true, Ordering::SeqCst);
        }

        // Build a stub whose first instruction is `brk #0xDE01`, followed by a
        // `ret` (only reached if the handler somehow returns without
        // redirecting — then the test's asserts fail cleanly rather than
        // executing garbage).
        let mut cache = CodeCache::new(1 << 16).unwrap();

        // Publish the brk-stub.
        let mut sa = JasmAssembler::new();
        emit_brk(&mut sa, TRAP_STRESS); // brk #0xDE01
        sa.emit("ret", &[]);
        let stub_blob = sa.finish();
        let hs = cache.alloc(stub_blob.code.len()).unwrap();
        cache.publish(hs, &stub_blob);

        // The capture trampoline: `mov x_scratch, x16` is done inside
        // `capture_and_return` itself; here we just need a tiny native shim
        // that calls it and then returns. But simplest: point the handler
        // directly at a stub that `b`s into `capture_and_return`. We build a
        // stub: `ldr x17,<addr>; br x17`. It preserves x16 (the trap pc) into
        // `capture_and_return`, which reads it.
        let mut ta = JasmAssembler::new();
        let lit = ta.literal_u64(
            capture_and_return as *const () as u64,
            Some(RelocKind::RuntimeAddr),
        );
        ta.ldr_literal(xr(17), lit);
        ta.emit("br", &[x(17)]);
        let tramp_blob = ta.finish();
        let ht = cache.alloc(tramp_blob.code.len()).unwrap();
        cache.publish(ht, &tramp_blob);

        // Arm the handler pointing at OUR capture trampoline, with THIS
        // cache's bounds.
        let (lo, hi) = cache.bounds();
        // SAFETY: test-only global override; this test process is
        // single-threaded for the duration of the trap.
        unsafe { test_arm_handler(lo, hi, ht.base as u64) };

        // Execute the brk-stub. Control: brk #0xDE01 -> SIGTRAP -> handler
        // stashes pc in x16, sets __pc = capture trampoline -> br into
        // capture_and_return -> records x16, returns -> ret unwinds out of the
        // stub call. Result: we return here normally.
        let stub: extern "C" fn() = unsafe { core::mem::transmute(hs.base) };
        stub();

        assert!(
            RAN.load(Ordering::SeqCst),
            "capture trampoline must have run"
        );
        let captured = CAPTURED_X16.load(Ordering::SeqCst);
        assert_eq!(
            captured, hs.base as u64,
            "x16 must hold the trap pc = the brk-stub's first (and only trapping) instruction"
        );

        // Restore SIG_DFL + retire this test's registry entry so no later
        // test sees our handler or its stale range.
        unsafe {
            let mut sa: libc::sigaction = core::mem::zeroed();
            sa.sa_sigaction = libc::SIG_DFL;
            libc::sigaction(libc::SIGTRAP, &sa, core::ptr::null_mut());
        }
        deregister(lo);
        HANDLER_ARMED.store(false, Ordering::Release);
    }

    /// S21: the foreign-fault recovery path end to end, against a REAL,
    /// deliberately-induced `SIGSEGV` (a bad-pointer read) — not simulated.
    /// Proves `claim_jmp_slot`/`jmp_buf_ptr`/`sig_fault_handler`'s new
    /// branch/`take_last_crash_info` compose correctly: `sigsetjmp`,
    /// called DIRECTLY at this test closure's own call site (never through
    /// a wrapper function — see `sigsetjmp`'s own doc for why that matters;
    /// an earlier version of this code got exactly that wrong and this
    /// same test caught it), "returns twice" (once normally, once via
    /// `siglongjmp` after the real fault), and the crash info published
    /// just before the jump is readable afterward. Run on a spawned thread
    /// so a genuine miss in this mechanism (the fault reaching `SIG_DFL`
    /// instead of being recovered) is exactly the failure this whole
    /// design exists to prevent, made concrete rather than hidden.
    #[cfg(target_os = "macos")] // WINVM: exercises the macOS signal-based foreign-fault recovery
    #[test]
    fn foreign_fault_recovers_via_registered_jmp_slot_on_a_real_segv() {
        // Arms SIGTRAP/SIGSEGV/SIGBUS (idempotent — may already be armed by
        // another test in this process; `install`'s own `HANDLER_ARMED`
        // guard makes re-calling it safe, matching this file's established
        // per-test convention above).
        let mut cache = CodeCache::new(1 << 16).unwrap();
        let _trampolines = install(&mut cache);

        let handle = std::thread::spawn(|| {
            let slot = claim_jmp_slot();
            // SAFETY: `slot` was just claimed by this exact thread; the
            // `sigsetjmp` call is inline, right here, at this closure's own
            // call site — its frame stays live for the rest of the closure
            // (the recovery window), never returning in between.
            let rc = unsafe { sigsetjmp(jmp_buf_ptr(slot), 1) };
            if rc == 0 {
                // First pass: cause a REAL SIGSEGV (address 8 — the null
                // page, always unmapped).
                let bad = 8usize as *const u8;
                unsafe { std::ptr::read_volatile(bad) };
                panic!("unreachable — the segv should have recovered via siglongjmp");
            }
            // Resumed here via siglongjmp, not a normal return.
            let info = take_last_crash_info();
            deregister_setjmp();
            info
        });
        let info = handle.join().expect("worker thread must complete normally");
        let (sig, _pc, far) =
            info.expect("expected Some((sig, pc, far)) after a recovered foreign fault");
        assert_eq!(sig, libc::SIGSEGV);
        assert_eq!(
            far, 8,
            "far (fault address) must be the real address that was read"
        );
    }

    /// The test that actually matters for correctness, not just "does it
    /// work once": `sigsetjmp(env, 1)`'s own "save the signal mask"
    /// semantics must correctly un-block `SIGSEGV` after `siglongjmp`, or a
    /// SECOND real fault's behavior is undefined (plain `setjmp`/`longjmp`
    /// do NOT restore the mask — this is exactly the distinction that made
    /// `sigsetjmp`/`siglongjmp` the right choice over the plain variants).
    /// Two real, separately-induced SIGSEGVs, both recovered — `sigsetjmp`
    /// is called FRESH each loop iteration, matching the real deployment
    /// shape (once per guest eval), each call still inline at this same
    /// closure's own call site.
    #[cfg(target_os = "macos")] // WINVM: exercises the macOS signal-based foreign-fault recovery
    #[test]
    fn foreign_fault_recovers_from_a_second_real_segv_too() {
        let mut cache = CodeCache::new(1 << 16).unwrap();
        let _trampolines = install(&mut cache);

        let handle = std::thread::spawn(|| {
            let slot = claim_jmp_slot();
            let mut recoveries = 0u32;
            for addr in [8usize, 16usize] {
                let rc = unsafe { sigsetjmp(jmp_buf_ptr(slot), 1) };
                if rc == 0 {
                    let bad = addr as *const u8;
                    unsafe { std::ptr::read_volatile(bad) };
                    panic!("unreachable");
                }
                recoveries += 1;
                let _ = take_last_crash_info();
            }
            deregister_setjmp();
            recoveries
        });
        assert_eq!(
            handle.join().expect("worker thread must complete normally"),
            2,
            "both real SIGSEGVs must have been recovered"
        );
    }

    /// `take_last_crash_info` must not falsely report a crash for a thread
    /// that registered a slot but never actually faulted — `sigsetjmp`'s
    /// own `0` return (no recovery happened yet).
    #[test]
    fn take_last_crash_info_is_none_before_any_fault() {
        let mut cache = CodeCache::new(1 << 16).unwrap();
        let _trampolines = install(&mut cache);

        let handle = std::thread::spawn(|| {
            let slot = claim_jmp_slot();
            let rc = unsafe { sigsetjmp(jmp_buf_ptr(slot), 1) };
            assert_eq!(rc, 0, "first call must be the ordinary, non-resumed return");
            let info = take_last_crash_info();
            deregister_setjmp();
            info
        });
        assert_eq!(handle.join().unwrap(), None);
    }

    /// `arm_foreign_fault_handler` (S21, `embed::VmHandle::boot`'s
    /// `JitMode::Off` case) must recover a real SIGSEGV with NO `CodeCache`/
    /// `install` involved at all — proving the foreign-fault path does not
    /// secretly depend on the JIT-only `install` having run first. Confirms
    /// the actual gap this function closes: SPEC §16.5 requires the GUI's
    /// Browser accept path to run with `MACVM_JIT=off`, and `with_options`
    /// never calls `install` in that mode.
    #[cfg(target_os = "macos")] // WINVM: exercises the macOS signal-based foreign-fault recovery
    #[test]
    fn arm_foreign_fault_handler_recovers_without_install_or_code_cache() {
        let handle = std::thread::spawn(|| {
            arm_foreign_fault_handler();
            let slot = claim_jmp_slot();
            let rc = unsafe { sigsetjmp(jmp_buf_ptr(slot), 1) };
            if rc == 0 {
                let bad = 8usize as *const u8;
                unsafe { std::ptr::read_volatile(bad) };
                panic!("unreachable — the segv should have recovered via siglongjmp");
            }
            let info = take_last_crash_info();
            deregister_setjmp();
            info
        });
        let info = handle.join().expect("worker thread must complete normally");
        let (sig, _pc, far) =
            info.expect("expected Some((sig, pc, far)) after a recovered foreign fault");
        assert_eq!(sig, libc::SIGSEGV);
        assert_eq!(far, 8);
    }

    /// CG0 — the test the per-thread alt-stacks exist for. Two threads each
    /// arm the foreign-fault handler (each registering its OWN sigaltstack) and
    /// then fault IN LOCKSTEP, many times, synchronized on a `Barrier` so their
    /// two signal handlers run as close to simultaneously as the scheduler
    /// allows. With the pre-CG0 single shared alt-stack, the two handlers ran
    /// on the same buffer and stomped each other's frames — a corrupted
    /// `siglongjmp` state, a wrong recorded fault address, or a hard crash.
    /// With per-thread alt-stacks both recoveries are independent: each thread
    /// must see ITS OWN fault address across every iteration, and the whole
    /// test process must survive. Each thread faults at a DISTINCT address so a
    /// cross-thread stomp would surface as a mismatched `far`.
    #[cfg(target_os = "macos")] // WINVM: exercises the macOS signal-based foreign-fault recovery
    #[test]
    fn concurrent_foreign_faults_on_two_threads_each_recover_on_their_own_altstack() {
        use std::sync::{Arc, Barrier};

        const ITERS: usize = 300;
        let barrier = Arc::new(Barrier::new(2));

        // Each thread arms its own handler+alt-stack, claims its own jmp slot,
        // and loops: rendezvous, `sigsetjmp`, fault at its own address, recover,
        // assert it saw its own address. Returns the recovery count.
        fn faulter(addr: usize, barrier: Arc<Barrier>) -> std::thread::JoinHandle<usize> {
            std::thread::spawn(move || {
                arm_foreign_fault_handler();
                let slot = claim_jmp_slot();
                let mut recoveries = 0usize;
                for _ in 0..ITERS {
                    // Rendezvous so both threads fault together — the race the
                    // per-thread alt-stacks fix.
                    barrier.wait();
                    // SAFETY: `slot` was claimed by this exact thread; the
                    // `sigsetjmp` is inline at this call site, whose frame stays
                    // live for the whole recovery window (this loop body).
                    let rc = unsafe { sigsetjmp(jmp_buf_ptr(slot), 1) };
                    if rc == 0 {
                        let bad = addr as *const u8;
                        // SAFETY: a deliberate read of an always-unmapped low
                        // address — a genuine SIGSEGV, recovered via siglongjmp.
                        unsafe { std::ptr::read_volatile(bad) };
                        panic!("unreachable — the segv must recover via siglongjmp");
                    }
                    let (sig, _pc, far) = take_last_crash_info()
                        .expect("a recovered foreign fault must have crash info");
                    assert_eq!(sig, libc::SIGSEGV);
                    assert_eq!(
                        far, addr as u64,
                        "each thread must see ITS OWN fault address — a mismatch \
                         means the two handlers corrupted each other's state"
                    );
                    recoveries += 1;
                }
                deregister_setjmp();
                recoveries
            })
        }

        // Distinct always-unmapped low addresses (both in the null page).
        let t1 = faulter(0x8, Arc::clone(&barrier));
        let t2 = faulter(0x10, Arc::clone(&barrier));
        assert_eq!(
            t1.join().expect("thread 1 must complete normally"),
            ITERS,
            "thread 1 must recover every fault"
        );
        assert_eq!(
            t2.join().expect("thread 2 must complete normally"),
            ITERS,
            "thread 2 must recover every fault"
        );
    }

    /// `read_frame_slot` reads the 8-byte oop at `fp + off` (the one licensed
    /// native-stack door for step 6). Exercised against a stack-local array so
    /// the offsets are real.
    #[test]
    fn read_frame_slot_reads_offset() {
        // Low 2 bits = 0b00 (smi tag) so `Oop::from_raw`'s debug tag check is
        // satisfied — real spill slots hold real oops, never raw junk words.
        let slots: [u64; 4] = [0x1110, 0x2220, 0x3330, 0x4440];
        let fp = slots.as_ptr() as usize;
        // off 0, +8, +16 in bytes.
        assert_eq!(read_frame_slot(fp, 0).raw(), 0x1110);
        assert_eq!(read_frame_slot(fp, 8).raw(), 0x2220);
        assert_eq!(read_frame_slot(fp, 16).raw(), 0x3330);
        // Negative offsets (below-FP spill slots, the S12 convention) work too
        // when fp points into the middle of the array.
        let mid = (&slots[2] as *const u64) as usize;
        assert_eq!(read_frame_slot(mid, -8).raw(), 0x2220);
        assert_eq!(read_frame_slot(mid, -16).raw(), 0x1110);
    }
}
