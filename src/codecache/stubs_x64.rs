//! The x86-64 call stub (WINVM Phase 3, MIGRATION.md §4) — the sibling of
//! `stubs::build_call_stub`, and the door the interpreter goes through to
//! enter compiled code.
//!
//! ## What a call stub is for
//!
//! Compiled code runs with pinned registers the Rust side knows nothing
//! about (`R15 = &VmState` and friends — `compiler::assembler_x64`), and
//! it expects its arguments in the Win64 argument registers rather than in
//! the `argv` array the interpreter has on hand. The stub is the one place
//! that translation happens: establish the pinned registers, marshal
//! `argv` into argument registers, call, and put everything back exactly
//! as the Rust caller left it.
//!
//! ## Win64 specifics this stub has to get right
//!
//! - **Callee-saved set.** Win64 preserves `RBX RBP RSI RDI R12–R15` (and
//!   `XMM6–15`). Compiled code freely uses `RBX/RSI/RDI` (they are in the
//!   allocatable pool) and `R12`–`R14` (also allocatable since the pin
//!   census), leaving only `R15` pinned — so all seven are saved here.
//!   `XMM6–15` are NOT saved: the x64 register file deliberately leaves
//!   them out of the FP pool (see `regalloc`'s `FP_ALLOCATABLE_REGS`), so
//!   compiled code cannot touch them. When Phase 5 claims them for float
//!   residency, this stub must start saving them — that coupling is
//!   asserted by `fp_pool_is_empty_or_this_stub_must_save_xmm` below.
//! - **Shadow space.** Every Win64 call must reserve 32 bytes for the
//!   callee to spill its register arguments into, and `RSP` must be
//!   16-byte aligned *at the call instruction*. After the return address,
//!   `push rbp`, and seven register pushes, `RSP % 16 == 8`, so the stub
//!   subtracts **40** (32 shadow + 8 realignment), not 32. Getting this
//!   wrong doesn't fault immediately — it corrupts alignment-sensitive
//!   callees much later — so the arithmetic is spelled out here.
//! - **Argument-register clobber order.** The stub's own four parameters
//!   arrive in `RCX RDX R8 R9`, which are exactly the registers the
//!   compiled callee's arguments must end up in. Everything is therefore
//!   moved to scratch (`R10`/`R11`) *before* the first argument is loaded.

use crate::codecache::stubs::KIND_DEOPT_BRIDGE;
use crate::compiler::assembler::{CodeBlob, RelocKind};
use crate::compiler::assembler_x64::{
    imm, incoming_stack_slot, mem, r64, xmm, Cond, X64Assembler, ARG_REGS, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX,
    RDI, RDX, RSI, RSP, R8, R9, VM_STATE,
};

/// The ABI of the generated stub:
/// `call_stub(entry, vm, argv, argc) -> result`.
///
/// # Safety
/// `entry` must be a live compiled-method entry point in the code cache,
/// `vm` the `&mut VmState` compiled code will reach through the pinned
/// register, and `argv` an array of at least `argc` oop words.
pub type CallStubFn = unsafe extern "C" fn(u64, u64, *const u64, u64) -> u64;

/// Registers this stub saves and restores, in push order. Pop order is the
/// reverse. `RBP` is handled separately by the frame prologue/epilogue.
const SAVED: [u8; 7] = [RBX, RSI, RDI, R12, R13, R14, R15];

/// The callee-saved XMM bank this stub preserves: `xmm6`-`xmm15`, the ten
/// Win64 requires a callee to restore.
///
/// Saving them is what lets `regalloc` put them in the FP allocatable
/// pool, taking it from 3 registers to 13. The cost is paid once per
/// interpreter-to-compiled transition, NOT per send: compiled-to-compiled
/// calls never route through this stub.
///
/// Full 128 bits each, not just the low double. Win64 requires the whole
/// register preserved, and a future SIMD pool would use the upper half —
/// saving 8 bytes would work perfectly until the day it silently didn't.
const SAVED_XMM: std::ops::Range<u8> = 6..16;
/// 16 bytes per register. A multiple of 16, so reserving it leaves `RSP`'s
/// alignment phase exactly as the push sequence left it.
const XMM_SAVE_BYTES: i64 = 16 * 10;

/// Shadow space (32, mandatory on Win64) plus 8 bytes of realignment —
/// see the module header's alignment arithmetic.
const CALL_AREA: i64 = 40 + 8 * (crate::oops::layout::ROOTSPILL_SLOTS as i64 - 4);

/// Build the x86-64 call stub.
pub fn build_call_stub_x64() -> CodeBlob {
    let mut a = X64Assembler::new();

    // ── Prologue: frame + callee-saved bank ─────────────────────────────
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    for r in SAVED {
        a.emit("push", &[r64(r)]);
    }
    // The callee-saved XMM bank. `movups`, not `movaps`: after the return
    // address, `push rbp` and seven pushes, `RSP % 16 == 8`, so this area
    // is deliberately NOT 16-byte aligned and an aligned move would fault.
    a.emit("sub", &[r64(RSP), imm(XMM_SAVE_BYTES)]);
    for (i, r) in SAVED_XMM.enumerate() {
        a.emit("movups", &[mem(RSP, 16 * i as i64), xmm(r)]);
    }

    // ── Stash the parameters before their registers are reused ──────────
    // entry(RCX) and argv(R8) are needed AFTER the argument registers get
    // overwritten, so they move to the scratch pair first. argc(R9) is
    // consumed by the compare chain below, and vm(RDX) goes straight to
    // its pinned home.
    a.emit("mov", &[r64(R10), r64(RCX)]); // entry
    a.emit("mov", &[r64(VM_STATE), r64(RDX)]); // &VmState -> R15 (pinned)
    a.emit("mov", &[r64(R11), r64(R8)]); // argv
    a.emit("mov", &[r64(RAX), r64(R9)]); // argc (RAX is dead until the result)

    // ── Marshal argv[0..argc) into the argument registers ───────────────
    // A compare-and-skip chain rather than a loop: argc is tiny and known
    // to be small, and this keeps the stub straight-line and branch-
    // predictable. Same shape as the AArch64 stub's own unrolled chain.
    let args_done = a.new_label();
    for (i, dst) in ARG_REGS.iter().enumerate() {
        a.emit("cmp", &[r64(RAX), imm(i as i64 + 1)]);
        a.jcc(Cond::L, args_done);
        a.emit("mov", &[r64(*dst), mem(R11, 8 * i as i64)]);
    }
    a.bind(args_done);

    // ── The call ────────────────────────────────────────────────────────
    a.emit("sub", &[r64(RSP), imm(CALL_AREA)]);

    // Arguments 4.. go on the stack (Win64 passes only four in
    // registers), written AFTER the outgoing area is reserved because
    // they are RSP-relative. Same compare-and-skip shape as the register
    // chain above; `argv` is still in R11 and `argc` in RAX.
    //
    // `CALL_AREA` already includes the full `ROOTSPILL_SLOTS` worth of
    // stack-argument space, so every slot written here is in bounds — and
    // so is the write-back a stub's epilogue performs into it.
    let stack_done = a.new_label();
    for i in ARG_REGS.len()..crate::oops::layout::ROOTSPILL_SLOTS {
        a.emit("cmp", &[r64(RAX), imm(i as i64 + 1)]);
        a.jcc(Cond::L, stack_done);
        // RBX, not R10: R10 still holds `entry`, and by this point RCX
        // holds the receiver, so there is nothing left to reload it from.
        // RBX is in this stub's own saved bank, so it is free scratch.
        a.emit("mov", &[r64(RBX), mem(R11, 8 * i as i64)]);
        a.emit(
            "mov",
            &[
                mem(RSP, crate::compiler::assembler_x64::outgoing_stack_slot(i)),
                r64(RBX),
            ],
        );
    }
    a.bind(stack_done);
    a.emit("call", &[r64(R10)]);
    a.emit("add", &[r64(RSP), imm(CALL_AREA)]);
    // The compiled method's result is already in RAX, which is also this
    // stub's return register — nothing to move.

    // ── Epilogue ────────────────────────────────────────────────────────
    for (i, r) in SAVED_XMM.enumerate() {
        a.emit("movups", &[xmm(r), mem(RSP, 16 * i as i64)]);
    }
    a.emit("add", &[r64(RSP), imm(XMM_SAVE_BYTES)]);
    for r in SAVED.iter().rev() {
        a.emit("pop", &[r64(*r)]);
    }
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);

    a.finish()
}

// ── Runtime stubs (Phase 3, MIGRATION.md §8) ────────────────────────────────
//
// The three the x64 emitter already calls. Each is the seam between
// compiled code (which knows only the pinned registers) and a Rust `rt_*`
// entry point (which wants an ordinary Win64 call), so each does the same
// three jobs: preserve what compiled code still needs, prepend `&VmState`
// to the argument list, and restore.
//
// **The ABI divergence that matters most.** `rt_poll` returns
// `PollOutcome { result, deopted }` — a 16-byte struct. AAPCS64 returns
// that in `x0:x1`, which is why the AArch64 stub simply reads two
// registers. **Win64 returns any struct larger than 8 bytes through a
// hidden pointer**: the caller passes a buffer address as an implicit
// FIRST argument, every real argument shifts one register right, and the
// callee returns the buffer address in `RAX`. Translating the AArch64
// stub instruction-for-instruction would therefore have read `RAX`/`RDX`
// as if they held the two fields, silently getting a pointer and garbage.
// `poll_outcome_is_returned_via_hidden_pointer` pins this empirically
// rather than on my reading of the ABI.

/// Volatile GPRs a stub preserves across its runtime call. Compiled code
/// may hold live values in the allocatable volatiles (`RCX RDX R8 R9`),
/// and the AArch64 poll stub saves the whole `x0`–`x15` bank for exactly
/// that reason, so this saves every Win64 volatile rather than reasoning
/// case-by-case about which are live.
const VOLATILES: [u8; 7] = [RAX, RCX, RDX, R8, R9, R10, R11];

/// The POLL stub's own frame (it has no RootSpill — it passes no oops to
/// the runtime — but does need a return buffer and a full volatile save):
/// `[0,32)` shadow space, `[32,48)` the `PollOutcome` return buffer,
/// `[48,104)` the saved volatiles. 112 keeps `RSP` 16-aligned.
const POLL_FRAME: i64 = 112;
const SHADOW_OFF: i64 = 0;
const RETBUF_OFF: i64 = 32;
const SAVE_OFF: i64 = 48;

/// `stub_poll` — the safepoint check's slow half. Calls
/// `rt_poll(vm, loop_fp, ret_pc)`; on an ordinary return it restores
/// everything and resumes the loop, but when `deopted` comes back set,
/// the compiled frame it was polling has been replaced and this stub
/// must return the deoptee's result to that frame's OWN caller — so it
/// drops both its own frame and the loop's in one go.
pub fn build_stub_poll_x64(rt_poll_addr: u64) -> CodeBlob {
    let mut a = X64Assembler::new();
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    a.emit("sub", &[r64(RSP), imm(POLL_FRAME)]);
    for (i, r) in VOLATILES.iter().enumerate() {
        a.emit("mov", &[mem(RSP, SAVE_OFF + 8 * i as i64), r64(*r)]);
    }

    // rt_poll(&mut retbuf, vm, loop_fp, ret_pc) — the hidden return
    // pointer occupies the first argument register, shifting the rest.
    a.emit("lea", &[r64(ARG_REGS[0]), mem(RSP, RETBUF_OFF)]);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(VM_STATE)]);
    // `[rbp]` is the caller's saved RBP — i.e. the polling compiled
    // frame's own frame pointer; `[rbp+8]` is the return address into it.
    a.emit("mov", &[r64(ARG_REGS[2]), mem(RBP, 0)]);
    a.emit("mov", &[r64(ARG_REGS[3]), mem(RBP, 8)]);
    let lit = a.literal_u64(rt_poll_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    let _ = SHADOW_OFF; // the shadow area is the callee's to use

    // deopted?
    a.emit("mov", &[r64(R10), mem(RSP, RETBUF_OFF + 8)]);
    a.emit("test", &[r64(R10), r64(R10)]);
    let resume = a.new_label();
    a.jcc(Cond::E, resume);

    // Deopted: hand the deoptee's result back to the LOOP's caller.
    // `loop_fp` is the compiled frame's RBP, so restoring RSP to it and
    // popping unwinds this stub's frame and the loop's together.
    a.emit("mov", &[r64(RAX), mem(RSP, RETBUF_OFF)]);
    a.emit("mov", &[r64(RSP), mem(RBP, 0)]);
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);

    // Ordinary return: restore and resume the loop.
    a.bind(resume);
    for (i, r) in VOLATILES.iter().enumerate() {
        a.emit("mov", &[r64(*r), mem(RSP, SAVE_OFF + 8 * i as i64)]);
    }
    a.emit("add", &[r64(RSP), imm(POLL_FRAME)]);
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);
    a.finish()
}

/// A stub for a runtime entry of the shape `rt_x(vm, a, b) -> u64`: the
/// emitter has already placed the real arguments in the first argument
/// registers, so this shifts them right by one and prepends `&VmState`.
fn build_shift_and_call_stub(rt_addr: u64, argc: usize, kind: u64) -> CodeBlob {
    assert!(argc <= ARG_REGS.len() - 1, "no room to prepend &VmState");
    build_stub(rt_addr, kind, argc, None, StubTail::Return)
}

/// `stub_must_be_boolean` — `rt_must_be_boolean(vm, val) -> u64`.
pub fn build_stub_must_be_boolean_x64(rt_addr: u64) -> CodeBlob {
    build_shift_and_call_stub(rt_addr, 1, crate::codecache::stubs::KIND_MUST_BE_BOOLEAN)
}

/// `stub_alloc_slow` — `rt_alloc_slow(vm, klass_bits, size_bytes) -> u64`.
pub fn build_stub_alloc_slow_x64(rt_addr: u64) -> CodeBlob {
    build_shift_and_call_stub(rt_addr, 2, crate::codecache::stubs::KIND_ALLOC_SLOW)
}


// ── Shared stub frame (the RootSpill contract) ──────────────────────────────
//
// Every stub that can reach Rust — and therefore a GC — must leave the
// argument oops somewhere the collector can find and UPDATE them. That
// place is the RootSpill: `memory::roots` scans slot `i` at
// `[fp - ROOTSPILL_BYTES + 8*i]` for a stub frame, taking the live count
// from the call site's own arity. So the offsets below are not an
// arbitrary frame layout — they are an interface with the collector, and
// arguments are RELOADED from those slots after the call because a moving
// GC may have rewritten them in place.
//
// (An earlier version of `must_be_boolean`/`alloc_slow` here skipped this
// entirely and just shuffled registers. That is a latent GC bug:
// `rt_alloc_slow` can scavenge, and a klass oop living only in a register
// would neither be found as a root nor updated when the object moved.)

pub(crate) const ROOTSPILL: i64 = crate::oops::layout::ROOTSPILL_BYTES as i64;
/// RootSpill + 32 bytes of outgoing shadow space. At stub entry `RSP % 16
/// == 8` (the return address); `push rbp` makes it 0 and this keeps it 0.
const STUB_FRAME: i64 = ROOTSPILL + 32;

pub(crate) fn emit_stub_prologue_x64(a: &mut X64Assembler, kind: u64) {
    use crate::oops::layout::{
        VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET,
        VMREG_LAST_COMPILED_PC_OFFSET,
    };
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    a.emit("sub", &[r64(RSP), imm(STUB_FRAME)]);
    for (i, r) in ARG_REGS.iter().enumerate() {
        a.emit("mov", &[mem(RBP, -ROOTSPILL + 8 * i as i64), r64(*r)]);
    }
    // Arguments 4.. arrived on the STACK, not in registers — Win64 passes
    // only four. They still have to reach the RootSpill, because that is
    // both what the collector scans for a stub frame and the `argv` every
    // `rt_*` consumer reads. Spilling registers alone would leave a
    // high-arity send's tail arguments invisible to the GC and garbage to
    // the runtime.
    //
    // All remaining slots are copied unconditionally rather than by
    // arity: a stub is built once, not once per call site, so it cannot
    // know the arity. Copying is safe — `incoming_stack_slot` addresses
    // the caller's own frame, which is mapped whether or not it reserved
    // that much outgoing space — and slots past the real arity are never
    // read, because the collector takes its live count from the call
    // site, not from this area's size.
    for i in ARG_REGS.len()..crate::oops::layout::ROOTSPILL_SLOTS {
        a.emit("mov", &[r64(R10), mem(RBP, incoming_stack_slot(i))]);
        a.emit("mov", &[mem(RBP, -ROOTSPILL + 8 * i as i64), r64(R10)]);
    }
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(RBP)],
    );
    // The send site's return address. On AArch64 this is `x30`, which a
    // `bl` sets and which a guard's or PIC's tail-`b` leaves untouched —
    // so the original site's address survives every door into a stub.
    // x64 has no link register: the same value is the return address the
    // original `call` pushed, at `[rbp+8]` once this frame is set up, and
    // a tail-`jmp` into a stub pushes nothing, so it is still the
    // original site's. Same invariant, different storage.
    //
    // This store was MISSING, and the consequence was not a fault but a
    // stale read: `last_compiled_pc` kept whatever the previous writer
    // (an uncommon trampoline) had left, and `rt_interpret_call` used
    // that address as its caller's return site — landing in the middle of
    // an unrelated method and panicking with "no IcSite at offset ...".
    // R10 is safe to use here: the kind store below already clobbers it,
    // which is exactly why `mega_shared` moves its payload out of R10
    // before calling this.
    a.emit("mov", &[r64(R10), mem(RBP, 8)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_PC_OFFSET as i64), r64(R10)],
    );
    a.emit("mov", &[r64(R10), imm(kind as i64)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(R10)],
    );
}

/// Clears the walker record and RELOADS the arguments from the RootSpill
/// — deliberately not from registers, since a GC during the call may have
/// relocated the oops those slots hold.
pub(crate) fn emit_stub_epilogue_x64(a: &mut X64Assembler) {
    use crate::oops::layout::{VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET};
    a.emit("mov", &[r64(R10), imm(0)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(R10)],
    );
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(R10)],
    );
    for (i, r) in ARG_REGS.iter().enumerate() {
        a.emit("mov", &[r64(*r), mem(RBP, -ROOTSPILL + 8 * i as i64)]);
    }
    // The stack arguments have to go BACK, for the same reason the
    // register ones are reloaded: a moving GC rewrote the RootSpill, and
    // a tail-jumping stub's target reads its arguments from the caller's
    // outgoing area rather than from the RootSpill. Reloading registers
    // only would hand the target stale pointers for arguments 4 and up.
    //
    // Unconditional, and in-bounds because every caller reserves the full
    // `OUTGOING_ARG_BYTES` (see its doc — this write-back is exactly why
    // that reservation is a constant instead of being sized by arity).
    for i in ARG_REGS.len()..crate::oops::layout::ROOTSPILL_SLOTS {
        a.emit("mov", &[r64(R10), mem(RBP, -ROOTSPILL + 8 * i as i64)]);
        a.emit("mov", &[mem(RBP, incoming_stack_slot(i)), r64(R10)]);
    }
    a.emit("mov", &[r64(RSP), r64(RBP)]);
    a.emit("pop", &[r64(RBP)]);
}

/// The two stub shapes, sharing the frame above. Both park the runtime's
/// answer in `R11`, which the epilogue never touches.
///
/// - [`StubTail::Return`] — hand the answer back to the caller in `RAX`.
/// - [`StubTail::Jump`] — TAIL-jump to it. The frame is fully unwound
///   first, so `RSP` is back on the original return address and the
///   target's own `ret` reaches the original caller.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StubTail {
    Return,
    Jump,
}

/// Build a stub: prologue, shift the emitter's arguments right to make
/// room for `&VmState`, call, then the chosen tail.
///
/// `argc` counts the emitter-supplied arguments. `extra` optionally names
/// a register whose value becomes the LAST argument (the `argv` pointer
/// for the lookup stubs, or a selector carried in a scratch register).
fn build_stub(
    rt_addr: u64,
    kind: u64,
    argc: usize,
    argv_arg: Option<usize>,
    tail: StubTail,
) -> CodeBlob {
    let mut a = X64Assembler::new();
    emit_stub_prologue_x64(&mut a, kind);

    // Shift right-to-left so nothing is overwritten before it moves.
    for i in (0..argc).rev() {
        a.emit("mov", &[r64(ARG_REGS[i + 1]), r64(ARG_REGS[i])]);
    }
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    // A stub that hands the runtime an `argv` points it at the RootSpill,
    // which is exactly where the arguments now live.
    if let Some(slot) = argv_arg {
        a.emit("lea", &[r64(ARG_REGS[slot]), mem(RBP, -ROOTSPILL)]);
    }
    let lit = a.literal_u64(rt_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    a.emit("mov", &[r64(R11), r64(RAX)]);

    emit_stub_epilogue_x64(&mut a);
    match tail {
        StubTail::Return => {
            a.emit("mov", &[r64(RAX), r64(R11)]);
            a.emit("ret", &[]);
        }
        StubTail::Jump => a.emit("jmp", &[r64(R11)]),
    }
    a.finish()
}

// ── Send stubs: resolve and DNU ─────────────────────────────────────────────
//
// Both land on an inline-cache site that could not be satisfied — an
// unlinked one (`stub_resolve`) or one whose lookup found nothing
// (`stub_dnu`) — and both have the same shape:
//
//   spill the argument registers → call `rt_*(vm, ret_addr, argv)` →
//   restore the arguments → TAIL-JUMP to whatever the runtime resolved.
//
// Three x86-64 specifics, each a place a literal translation goes wrong:
//
// * **There is no link register.** AArch64 reads the return address out of
//   `x30`; on x64 the `call` pushed it, so after the prologue it lives at
//   `[rbp + 8]`. That address is what identifies WHICH site missed, so
//   getting it wrong sends the runtime to patch someone else's cache.
// * **The spilled arguments are the `argv` the runtime reads** — it needs
//   the receiver and arguments to do the lookup — and they double as GC
//   roots, which is why they go to a known frame offset rather than being
//   left in registers.
// * **The tail-jump must leave the stack exactly as the stub found it.**
//   Unwinding to `[rbp]` and popping leaves `RSP` pointing at the original
//   return address, so `jmp` (never `call`) hands the resolved target a
//   frame indistinguishable from the one the send site set up: its `ret`
//   goes straight back to the original caller, with this stub gone.

/// Argument registers spilled by a send stub, in slot order. These ARE the
/// `argv` array the runtime reads, so the order is the calling
/// convention's, not an arbitrary one.
const SEND_SPILL: [u8; 4] = ARG_REGS;

/// Send-stub frame: `[0,32)` outgoing shadow space, `[32,64)` the spilled
/// argument registers (the `argv` the runtime is handed). 64 keeps `RSP`
/// 16-aligned after `push rbp`.
const SEND_FRAME: i64 = 64;
const SEND_ARGV_OFF: i64 = 32;

/// Shared body of `stub_resolve` and `stub_dnu`; they differ only in the
/// runtime function called and the kind tag recorded for the stack walker.
fn build_send_stub(rt_addr: u64, kind: u64) -> CodeBlob {
    // rt_x(vm, ret_addr, argv). The return address is the SECOND argument
    // and comes off the stack, not a link register — so it is placed by
    // hand rather than shifted, and `argv` (slot 2) points at the
    // RootSpill the prologue just filled.
    let mut a = X64Assembler::new();
    emit_stub_prologue_x64(&mut a, kind);
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[1]), mem(RBP, 8)]);
    a.emit("lea", &[r64(ARG_REGS[2]), mem(RBP, -ROOTSPILL)]);
    let lit = a.literal_u64(rt_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("jmp", &[r64(R11)]);
    a.finish()
}

/// `stub_resolve` — an inline-cache site that has never been linked, and
/// the target the entry guard tail-jumps to on a customization miss.
pub fn build_stub_resolve_x64(rt_resolve_send_addr: u64) -> CodeBlob {
    build_send_stub(rt_resolve_send_addr, crate::codecache::stubs::KIND_RESOLVE)
}

/// `stub_dnu` — lookup found no method; the runtime builds and dispatches
/// `doesNotUnderstand:`.
pub fn build_stub_dnu_x64(rt_dnu_addr: u64, kind: u64) -> CodeBlob {
    build_send_stub(rt_dnu_addr, kind)
}

/// `not_entrant_stub` — the entry an invalidated nmethod is repointed at,
/// so any later call re-resolves instead of running stale code.
///
/// Its body is **identical** to [`build_stub_resolve_x64`] (the AArch64
/// pair are byte-for-byte the same too): both hand the site to
/// `rt_resolve_send` and tail-jump wherever it says. They stay separate
/// stubs because they are separate *addresses* — an nmethod's entry is
/// repointed here, while unlinked inline caches point at `stub_resolve` —
/// and the runtime distinguishes the two cases by which address it finds,
/// not by the code at it.
pub fn build_not_entrant_stub_x64(rt_resolve_send_addr: u64) -> CodeBlob {
    build_send_stub(rt_resolve_send_addr, crate::codecache::stubs::KIND_RESOLVE)
}

/// `deopt_return_trampoline` — the other half of the deopt story from the
/// uncommon trap. When a method is invalidated while its activation is
/// live, that frame's saved return address is redirected here, so a
/// callee's ordinary `ret` lands in this trampoline instead of back in
/// stale compiled code.
///
/// It is therefore entered **by a `ret`**, not a call, with `RAX` holding
/// the callee's result and `RBP` already restored to the victim frame's
/// own frame pointer by that callee's epilogue.
///
/// Two runtime calls, in order:
/// 1. `rt_deopt_return_pc(vm, victim_fp)` — the ORIGINAL return address,
///    a pc inside the victim nmethod. It becomes this trampoline's own
///    frame's return-address slot, so a GC during the second call
///    classifies the victim frame at its true safepoint rather than at
///    some arbitrary pc.
/// 2. `rt_deopt_on_return(vm, victim_fp, result)` — materializes and runs
///    the interpreter frames, answering the deoptee's result.
///
/// Then, exactly like the uncommon trampoline, the victim activation is
/// gone: unwind to `victim_fp` and return to the victim's OWN caller.
pub fn build_deopt_return_trampoline_x64(
    rt_deopt_return_pc_addr: u64,
    rt_deopt_on_return_addr: u64,
) -> CodeBlob {
    use crate::oops::layout::{
        VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET,
        VMREG_LAST_COMPILED_PC_OFFSET,
    };
    let mut a = X64Assembler::new();

    // Park the two inputs in callee-saved registers so the runtime calls
    // below cannot destroy them. (Safe to clobber RBX/RSI here: nothing
    // live is in them, by the spill-all-at-safepoints invariant, and the
    // call stub restores them at the Rust boundary.)
    a.emit("mov", &[r64(RBX), r64(RAX)]); // the callee's result
    a.emit("mov", &[r64(RSI), r64(RBP)]); // victim_fp

    // (1) orig_ret_pc = rt_deopt_return_pc(vm, victim_fp)
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(RSI)]);
    let pc_lit = a.literal_u64(rt_deopt_return_pc_addr, Some(RelocKind::RuntimeAddr));
    a.emit("sub", &[r64(RSP), imm(32)]);
    a.call_far(pc_lit);
    a.emit("add", &[r64(RSP), imm(32)]);

    // Build the bridged frame with orig_ret_pc in the return-address slot
    // — the walkability trick, same as the uncommon trampoline's.
    a.emit("push", &[r64(RAX)]); // orig_ret_pc
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(RBP)],
    );
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_PC_OFFSET as i64), r64(RAX)],
    );
    a.emit("mov", &[r64(R10), imm(KIND_DEOPT_BRIDGE as i64)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(R10)],
    );

    // (2) result = rt_deopt_on_return(vm, victim_fp, callee_result)
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(RSI)]);
    a.emit("mov", &[r64(ARG_REGS[2]), r64(RBX)]);
    let rt_lit = a.literal_u64(rt_deopt_on_return_addr, Some(RelocKind::RuntimeAddr));
    a.emit("sub", &[r64(RSP), imm(32)]);
    a.call_far(rt_lit);
    a.emit("add", &[r64(RSP), imm(32)]);

    a.emit("mov", &[r64(R10), imm(0)]);
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_FP_OFFSET as i64), r64(R10)],
    );
    a.emit(
        "mov",
        &[mem(VM_STATE, VMREG_LAST_COMPILED_KIND_OFFSET as i64), r64(R10)],
    );

    // The victim activation is gone: unwind to its frame and return the
    // deoptee's result (in RAX) to the victim's own caller.
    a.emit("mov", &[r64(RSP), r64(RSI)]);
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);
    a.finish()
}


/// `stub_mega_shared` — the megamorphic lookup tail. Reached from a
/// per-selector `mega_<sel>` thunk that carries the selector in a scratch
/// register (the `x16` role, so `R10` here), and tail-jumps to whatever
/// the lookup resolves.
pub fn build_stub_mega_shared_x64(rt_mega_lookup_addr: u64) -> CodeBlob {
    let mut a = X64Assembler::new();
    // The selector must be read BEFORE the prologue, whose kind-tag store
    // uses R10 as its scratch.
    a.emit("mov", &[r64(R11), r64(R10)]);
    emit_stub_prologue_x64(&mut a, crate::codecache::stubs::KIND_MEGA);
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(R11)]); // selector_bits
    a.emit("lea", &[r64(ARG_REGS[2]), mem(RBP, -ROOTSPILL)]); // argv
    let lit = a.literal_u64(rt_mega_lookup_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("jmp", &[r64(R11)]);
    a.finish()
}

/// `stub_box_double` — `rt_box_double(vm, bits) -> u64`. Allocates, so it
/// needs the full RootSpill frame like any other GC-reaching stub.
pub fn build_stub_box_double_x64(rt_box_double_addr: u64) -> CodeBlob {
    build_shift_and_call_stub(
        rt_box_double_addr,
        1,
        crate::codecache::stubs::KIND_BOX_DOUBLE,
    )
}

/// `stub_call_primitive` — `rt_call_primitive(vm, prim_id, argc_plus_recv)`.
pub fn build_stub_call_primitive_x64(rt_call_primitive_addr: u64, kind: u64) -> CodeBlob {
    let mut a = X64Assembler::new();

    // The payload arrives in R10/R11, NOT in the argument registers —
    // and that is the whole point. `RCX/RDX/R8/R9` here hold the
    // compiled method's own receiver and arguments, which the prologue
    // below archives into the RootSpill as the primitive's `&[Oop]`
    // argument slice. Passing `prim_id` in RCX would overwrite the
    // receiver with an integer and hand the primitive its own id as
    // `self`.
    //
    // This is exactly what the AArch64 stub does with x10/x11, and it is
    // why this stub cannot be `build_shift_and_call_stub` — an earlier
    // version was, reading RCX/RDX, and nothing caught it because
    // `prim_shim` was declined on x64 so the stub was never called.
    //
    // Both payloads arrive PACKED in R11: `(prim_id << 8) | argc_plus_recv`.
    //
    // One register, not two, because only one is actually available.
    // R10 is out on both counts — the prologue clobbers it for the
    // return-address and kind stores, and `call_far` loads its target
    // into it, so a `prim_id` placed there is overwritten by the
    // stub's own address before the call even happens. (That was the
    // first version, and it failed with "unknown primitive id
    // 140694972008352" — a code address, which named the bug exactly.)
    //
    // Packing is safe: `argc_plus_recv` is a method's arity plus one, far
    // below 256, and primitive ids are small. The alternative was
    // clobbering a callee-saved register, which would have depended on
    // spill-all-at-safepoints in a way that is true but too subtle to
    // rest an ABI on.
    a.emit("mov", &[r64(RAX), r64(R11)]); // packed payload
    emit_stub_prologue_x64(&mut a, kind);
    a.emit("mov", &[r64(ARG_REGS[2]), r64(RAX)]);
    a.emit("and", &[r64(ARG_REGS[2]), imm(0xFF)]); // argc_plus_recv
    a.emit("mov", &[r64(ARG_REGS[1]), r64(RAX)]);
    a.emit("shr", &[r64(ARG_REGS[1]), imm(8)]); // prim_id
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    let lit = a.literal_u64(rt_call_primitive_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    // Park the result across the epilogue, which reloads the argument
    // registers from the RootSpill — that reload is what lets the
    // shim's FAIL path fall straight through into the method body with
    // the receiver and arguments exactly as they arrived.
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("mov", &[r64(RAX), r64(R11)]);
    a.emit("ret", &[]);
    a.finish()
}

/// `stub_nlr_originate` — `rt_nlr_originate(vm, closure, value)`, which
/// returns nothing; the stub simply returns to its caller, whose emitted
/// NLR check then propagates the sentinel.
pub fn build_stub_nlr_originate_x64(rt_nlr_originate_addr: u64, kind: u64) -> CodeBlob {
    build_shift_and_call_stub(rt_nlr_originate_addr, 2, kind)
}

/// `stub_value_dispatch` — `value`/`value:`… on a closure receiver. The
/// only stub with TWO runtime calls and two different tails, which is why
/// it does not fit either shape above:
///
/// 1. `rt_value_target(vm, closure, argc)` asks for a compiled entry to
///    jump straight to. A non-zero answer is the fast path: **tail-jump**,
///    so the closure's own `ret` reaches this stub's caller.
/// 2. Zero means "not a closure, or not compiled" — fall back to
///    `rt_value_fallback(vm, argv, argc)`, which interprets the send and
///    **returns** a value like an ordinary stub.
///
/// `argc` is baked in at build time (there is one of these stubs per
/// arity, as on AArch64), and the fallback's `argv` points at the
/// RootSpill the prologue filled — so the arguments it interprets are the
/// GC-visible, GC-updated copies, not stale registers.
pub fn build_stub_value_dispatch_x64(
    rt_value_target_addr: u64,
    rt_value_fallback_addr: u64,
    argc: u64,
    kind: u64,
) -> CodeBlob {
    let mut a = X64Assembler::new();
    emit_stub_prologue_x64(&mut a, kind);

    // (1) rt_value_target(vm, closure_bits, argc)
    a.emit("mov", &[r64(ARG_REGS[1]), r64(ARG_REGS[0])]); // closure, before vm overwrites it
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("mov", &[r64(ARG_REGS[2]), imm(argc as i64)]);
    let target_lit = a.literal_u64(rt_value_target_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(target_lit);

    let fallback = a.new_label();
    a.emit("test", &[r64(RAX), r64(RAX)]);
    a.jcc(Cond::E, fallback);

    // Fast path: tail-jump to the compiled closure entry.
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("jmp", &[r64(R11)]);

    // (2) Fallback: rt_value_fallback(vm, argv, argc), returning a value.
    a.bind(fallback);
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("lea", &[r64(ARG_REGS[1]), mem(RBP, -ROOTSPILL)]);
    a.emit("mov", &[r64(ARG_REGS[2]), imm(argc as i64)]);
    let fallback_lit = a.literal_u64(rt_value_fallback_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(fallback_lit);
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("mov", &[r64(RAX), r64(R11)]);
    a.emit("ret", &[]);
    a.finish()
}

/// A stub for a slot the x64 back end cannot reach yet: two `ud2`s.
///
/// The `Stubs` table has one entry per stub *address*, and every entry
/// must hold something — but three of them (the SIMD boxers) exist only
/// for IR ops `emit_x64` does not lower at all: `FBox` and `VecArith` are
/// absent from its `SUPPORTED_OPS`, so a method containing one panics at
/// compile time and never reaches an emitted call. Publishing an empty
/// blob, or leaving the AArch64 bytes in place, would make an unreachable
/// path *look* live; `ud2` makes it unmistakable if it somehow isn't.
///
/// `ud2` and not `int3`: `int3` is the deopt trap encoding, and the VEH
/// handler would try to decode this as a deopt site and mis-report it.
/// `ud2` raises ILLEGAL_INSTRUCTION, which nothing in the VM claims.
pub fn build_unreachable_stub_x64() -> CodeBlob {
    let mut a = X64Assembler::new();
    a.emit_bytes(&[0x0F, 0x0B]); // ud2
    a.emit_bytes(&[0x0F, 0x0B]); // ud2 — so a one-byte skid still faults
    a.finish()
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::compiler::emit_x64::{emit_x64, RuntimeAddrs};
    use crate::compiler::ir::{
        BailoutReason, BlockId, Ir, IrBlock, IrMethod, PoolLit, SmiOp, VReg, VRegInfo,
    };
    use crate::compiler::regalloc::regalloc;

    fn smi(v: i64) -> u64 {
        (v as u64) << 2
    }

    /// Base of the distinct per-register sentinels the preservation test
    /// plants. Small and non-canonical so a leaked value is obvious.
    const SENTINEL_BASE: i64 = 0x1100;

    fn hand_method(blocks: Vec<IrBlock>, nvregs: usize, argc: u8) -> IrMethod {
        IrMethod {
            blocks,
            vregs: (0..nvregs)
                .map(|_| VRegInfo {
                    is_oop: true,
                    is_fp: false,
                })
                .collect(),
            pool: Vec::new(),
            argc,
            ntemps: 0,
            ctx_vregs: Vec::new(),
            block_closure_vreg: None,
            method_ctx_vreg: None,
            spliced_nlr: 0,
            spliced_multibb: 0,
            splice_declined_budget: 0,
            osr_cold_sends: 0,
            safepoints: Vec::new(),
            true_lit: PoolLit(0),
            false_lit: PoolLit(0),
            nil_lit: PoolLit(0),
            mark_slots_lit: PoolLit(0),
            mark_double_lit: PoolLit(0),
            double_klass_lit: PoolLit(0),
            float64x2_klass_lit: PoolLit(0),
            float32x4_klass_lit: PoolLit(0),
            int32x4_klass_lit: PoolLit(0),
            call_sites: Vec::new(),
            site_feedback: Vec::new(),
            inline_deps: Vec::new(),
            self_devirt: false,
            method_pool_ix: None,
        }
    }

    /// `^ a + b` as a compiled method — the callee for the stub tests.
    fn add_method() -> IrMethod {
        hand_method(
            vec![
                IrBlock {
                    id: BlockId(0),
                    bci: 0,
                    code: vec![
                        Ir::Param {
                            dst: VReg(0),
                            index: 0,
                        },
                        Ir::Param {
                            dst: VReg(1),
                            index: 1,
                        },
                        Ir::SmiArith {
                            op: SmiOp::Add,
                            dst: VReg(2),
                            a: VReg(0),
                            b: VReg(1),
                            fail: BlockId(1),
                        },
                        Ir::Ret { val: VReg(2) },
                    ],
                    entry_stack: Vec::new(),
                    deopt_sites: Vec::new(),
                },
                IrBlock {
                    id: BlockId(1),
                    bci: 0,
                    code: vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                    entry_stack: Vec::new(),
                    deopt_sites: Vec::new(),
                },
            ],
            3,
            2,
        )
    }

    /// End-to-end through the real stub: place the stub and a compiled
    /// method in one region, then enter the method the way the interpreter
    /// will — `call_stub(entry, vm, argv, argc)`.
    #[cfg(windows)]
    #[test]
    fn call_stub_enters_compiled_code_and_returns_its_result() {
        use crate::vendor::wfasm::native_windows::WinJit;

        let stub = build_call_stub_x64();
        let method = emit_x64(&add_method(), &regalloc(&add_method()), RuntimeAddrs::default(), None, None, None).blob;

        let jit = WinJit::with_capacity(stub.code.len() + method.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let method_off = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                method.code.as_ptr(),
                base.add(method_off),
                method.code.len(),
            );
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + method_off as u64;

        let argv = [smi(20), smi(22)];
        let got = unsafe { stub_fn(entry, 0, argv.as_ptr(), 2) };
        assert_eq!(got, smi(42), "20 + 22 through the real call stub");

        // A second call through the same stub must behave identically —
        // proving the stub restored everything it touched.
        let argv2 = [smi(-5), smi(8)];
        assert_eq!(unsafe { stub_fn(entry, 0, argv2.as_ptr(), 2) }, smi(3));
    }

    /// The stub must leave every Win64 callee-saved register exactly as it
    /// found it. This is the failure mode that does NOT show up as a wrong
    /// answer in the test above — it corrupts the *Rust caller* later, at
    /// a distance, which is exactly the kind of bug worth pinning.
    ///
    /// The callee here is a compiled method, so it really does write the
    /// allocatable callee-saved registers (RBX/RSI/RDI) and the pinned
    /// ones (R12–R15) on its way through.
    #[cfg(windows)]
    #[test]
    fn call_stub_preserves_callee_saved_registers() {
        use crate::vendor::wfasm::native_windows::WinJit;

        let stub = build_call_stub_x64();
        let method = emit_x64(&add_method(), &regalloc(&add_method()), RuntimeAddrs::default(), None, None, None).blob;

        // A harness, in machine code, that loads a distinct sentinel into
        // every callee-saved register, calls the stub, then XORs each
        // register against its sentinel and ORs the differences together.
        // It returns 0 iff every single one came back intact.
        //
        // Its own ABI is `harness(stub, entry, argv, argc)` in
        // RCX/RDX/R8/R9 — one register-shuffle away from the stub's
        // `(entry, vm, argv, argc)`, with `vm` forced to null (this callee
        // never reads it). R8/R9 already sit where the stub wants them.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        for r in SAVED {
            h.emit("push", &[r64(r)]);
        }
        h.emit("mov", &[r64(R10), r64(RCX)]); // stub address
        h.emit("mov", &[r64(RCX), r64(RDX)]); // entry
        h.emit("mov", &[r64(RDX), imm(0)]); // vm = null
        for (i, r) in SAVED.iter().enumerate() {
            h.emit("mov", &[r64(*r), imm(SENTINEL_BASE + i as i64)]);
        }
        h.emit("sub", &[r64(RSP), imm(CALL_AREA)]);
        h.emit("call", &[r64(R10)]);
        h.emit("add", &[r64(RSP), imm(CALL_AREA)]);
        h.emit("xor", &[r64(RAX), r64(RAX)]);
        for (i, r) in SAVED.iter().enumerate() {
            h.emit("mov", &[r64(R11), r64(*r)]);
            h.emit("xor", &[r64(R11), imm(SENTINEL_BASE + i as i64)]);
            h.emit("or", &[r64(RAX), r64(R11)]);
        }
        for r in SAVED.iter().rev() {
            h.emit("pop", &[r64(*r)]);
        }
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();

        let total = stub.code.len() + method.code.len() + harness.code.len() + 4096;
        let jit = WinJit::with_capacity(total).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let method_off = (stub.code.len() + 15) & !15;
        let harness_off = (method_off + method.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                method.code.as_ptr(),
                base.add(method_off),
                method.code.len(),
            );
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(harness_off),
                harness.code.len(),
            );
        }

        let harness_fn: unsafe extern "C" fn(u64, u64, *const u64, u64) -> u64 =
            unsafe { std::mem::transmute(base.add(harness_off)) };
        let stub_addr = base as u64;
        let entry = base as u64 + method_off as u64;
        let argv = [smi(3), smi(4)];

        let diff = unsafe { harness_fn(stub_addr, entry, argv.as_ptr(), 2) };
        assert_eq!(
            diff, 0,
            "a callee-saved register came back changed (XOR of all differences); \
             compiled code writes RBX/RSI/RDI and R12-R14 and pins R15, so every one of \
             {SAVED:?} must be saved and restored by the stub"
        );
    }

    /// The stub establishes `R15 = &VmState` before entering compiled
    /// code. Verified by compiling a method that simply reads a field off
    /// the pinned register's target: if R15 weren't set, this would
    /// dereference garbage rather than return the planted word.
    #[cfg(windows)]
    #[test]
    fn call_stub_pins_vm_state_in_r15() {
        use crate::vendor::wfasm::native_windows::WinJit;

        // A stand-in "VmState": the compiled method loads word 3 of it.
        let fake_vm = [0u64, 0, 0, 0xFEED_FACE_1234_5678u64];

        // A hand-built callee (not emit_x64 — reading R15 directly is not
        // in the IR's vocabulary): `mov rax, [r15 + 24]; ret`.
        let mut c = X64Assembler::new();
        c.emit("mov", &[r64(RAX), mem(R15, 24)]);
        c.emit("ret", &[]);
        let callee = c.finish();

        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + callee.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let callee_off = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                callee.code.as_ptr(),
                base.add(callee_off),
                callee.code.len(),
            );
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + callee_off as u64;
        let got = unsafe { stub_fn(entry, fake_vm.as_ptr() as u64, std::ptr::null(), 0) };
        assert_eq!(
            got, 0xFEED_FACE_1234_5678,
            "compiled code must find &VmState in R15"
        );
    }

    /// Argument marshalling honours `argc`: fewer arguments than the four
    /// register slots must not read past the end of `argv`.
    #[cfg(windows)]
    #[test]
    fn call_stub_marshals_only_argc_arguments() {
        use crate::vendor::wfasm::native_windows::WinJit;

        // A callee that returns its first argument untouched.
        let mut c = X64Assembler::new();
        c.emit("mov", &[r64(RAX), r64(RCX)]);
        c.emit("ret", &[]);
        let callee = c.finish();

        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + callee.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let callee_off = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                callee.code.as_ptr(),
                base.add(callee_off),
                callee.code.len(),
            );
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + callee_off as u64;

        // Exactly one argument available; the stub must load only that one.
        let argv = [smi(99)];
        assert_eq!(unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) }, smi(99));
    }

    /// **Pins the ABI fact the poll stub is built on**, empirically
    /// rather than from my reading of the Win64 spec: a 16-byte struct
    /// returned by an `extern "C"` function comes back through a HIDDEN
    /// POINTER passed as the implicit first argument — not in `RAX:RDX`
    /// the way AAPCS64 returns it in `x0:x1`.
    ///
    /// The check calls a real Rust `extern "C"` function returning a
    /// `PollOutcome`-shaped struct from generated machine code that
    /// follows the hidden-pointer convention, and requires both fields to
    /// arrive intact. If Rust ever lowered this differently, the poll
    /// stub would be silently wrong and this test is what fails.
    #[cfg(windows)]
    #[test]
    fn poll_outcome_is_returned_via_hidden_pointer() {
        use crate::compiler::assembler_x64::mem as xmem;
        use crate::vendor::wfasm::native_windows::WinJit;

        #[repr(C)]
        struct TwoWords {
            a: u64,
            b: u64,
        }
        extern "C" fn returns_two_words(x: u64) -> TwoWords {
            TwoWords {
                a: x + 1,
                b: x + 2,
            }
        }
        // The size is what selects the convention; state it.
        assert_eq!(
            std::mem::size_of::<TwoWords>(),
            16,
            "over 8 bytes, so Win64 uses the hidden-pointer return"
        );

        // Generated caller: rcx = &buf, rdx = the real argument.
        let mut a = X64Assembler::new();
        a.emit("push", &[r64(RBP)]);
        a.emit("mov", &[r64(RBP), r64(RSP)]);
        a.emit("sub", &[r64(RSP), imm(64)]);
        a.emit("mov", &[r64(R10), r64(RCX)]); // callee address
        a.emit("mov", &[r64(R11), r64(RDX)]); // the argument
        a.emit("lea", &[r64(RCX), xmem(RSP, 32)]); // hidden return buffer
        a.emit("mov", &[r64(RDX), r64(R11)]);
        a.emit("call", &[r64(R10)]);
        // Return a+b so BOTH fields must have landed correctly.
        a.emit("mov", &[r64(RAX), xmem(RSP, 32)]);
        a.emit("add", &[r64(RAX), xmem(RSP, 40)]);
        a.emit("add", &[r64(RSP), imm(64)]);
        a.emit("pop", &[r64(RBP)]);
        a.emit("ret", &[]);
        let blob = a.finish();

        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(base) };
        let got = f(returns_two_words as usize as u64, 10);
        assert_eq!(got, 23, "(10+1) + (10+2) — both struct fields returned");
    }

    /// `must_be_boolean` and `alloc_slow` stubs shift the emitter's
    /// arguments right and prepend `&VmState`. The probes record each
    /// argument by position, so an off-by-one shift or a missing
    /// `&VmState` shows up as a specific wrong slot rather than a value
    /// that happens to still work.
    ///
    /// The `vm` here is a REAL block, not a sentinel: since these stubs
    /// gained the RootSpill frame they publish `last_compiled_fp` through
    /// `R15`, so a fake pointer is an access violation. (That is exactly
    /// how this test caught the change.)
    #[cfg(windows)]
    #[test]
    fn shift_and_call_stubs_prepend_vm_state() {
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::{AtomicU64, Ordering};

        static A0: AtomicU64 = AtomicU64::new(0);
        static A1: AtomicU64 = AtomicU64::new(0);
        static A2: AtomicU64 = AtomicU64::new(0);

        extern "C" fn mbb(vm: u64, val: u64) -> u64 {
            A0.store(vm, Ordering::Relaxed);
            A1.store(val, Ordering::Relaxed);
            0xB001
        }
        extern "C" fn alloc(vm: u64, klass: u64, size: u64) -> u64 {
            A0.store(vm, Ordering::Relaxed);
            A1.store(klass, Ordering::Relaxed);
            A2.store(size, Ordering::Relaxed);
            0xA110C
        }

        let run = |blob: CodeBlob, vm: u64, a0: u64, a1: u64| -> u64 {
            let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
            let (base, _cap) = jit.region_raw();
            unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
            let mut h = X64Assembler::new();
            h.emit("push", &[r64(RBP)]);
            h.emit("mov", &[r64(RBP), r64(RSP)]);
            h.emit("push", &[r64(R15)]);
            h.emit("sub", &[r64(RSP), imm(40)]);
            h.emit("mov", &[r64(R10), r64(RCX)]); // stub
            h.emit("mov", &[r64(R15), r64(RDX)]); // vm
            h.emit("mov", &[r64(RCX), r64(R8)]); // emitter arg0
            h.emit("mov", &[r64(RDX), r64(R9)]); // emitter arg1
            h.emit("call", &[r64(R10)]);
            h.emit("add", &[r64(RSP), imm(40)]);
            h.emit("pop", &[r64(R15)]);
            h.emit("pop", &[r64(RBP)]);
            h.emit("ret", &[]);
            let hb = h.finish();
            let jit2 = WinJit::with_capacity(hb.code.len() + 4096).expect("RWX");
            let (hbase, _) = jit2.region_raw();
            unsafe { core::ptr::copy_nonoverlapping(hb.code.as_ptr(), hbase, hb.code.len()) };
            let hf: extern "C" fn(u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(hbase) };
            hf(base as u64, vm, a0, a1)
        };

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;

        // must_be_boolean: emitter passes (val); stub calls (vm, val).
        let got = run(build_stub_must_be_boolean_x64(mbb as usize as u64), vm, 42, 0);
        assert_eq!(got, 0xB001, "the runtime's result is returned");
        assert_eq!(A0.load(Ordering::Relaxed), vm, "&VmState prepended");
        assert_eq!(A1.load(Ordering::Relaxed), 42, "val shifted to slot 1");

        // alloc_slow: emitter passes (klass, size); stub calls (vm, klass, size).
        let got = run(build_stub_alloc_slow_x64(alloc as usize as u64), vm, 5, 9);
        assert_eq!(got, 0xA110C);
        assert_eq!(A0.load(Ordering::Relaxed), vm, "&VmState prepended");
        assert_eq!(A1.load(Ordering::Relaxed), 5, "klass in slot 1");
        assert_eq!(A2.load(Ordering::Relaxed), 9, "size in slot 2");

        // The walker record is cleared before returning to compiled code.
        assert_eq!(
            vmreg[crate::oops::layout::VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0
        );
    }

    /// A send stub end to end: a call site invokes the stub, the stub
    /// hands the runtime `(vm, ret_addr, argv)`, and then TAIL-jumps to
    /// whatever the runtime resolved — which must run with the original
    /// arguments and return straight to the original caller, with the
    /// stub's own frame gone.
    ///
    /// Every one of those is a place a literal AArch64 translation breaks:
    /// - `ret_addr` comes from the STACK, not a link register. The test
    ///   requires it to be the address immediately after the call site,
    ///   because that is what identifies which IC missed — a wrong value
    ///   would send the runtime to patch someone else's cache.
    /// - `argv` must expose the receiver and arguments in slot order.
    /// - the tail-jump must be `jmp`, not `call`: the resolved target's
    ///   `ret` has to return to the ORIGINAL caller. If the stub left its
    ///   frame on the stack, control would come back into the stub and
    ///   run off the end.
    #[cfg(windows)]
    #[test]
    fn send_stub_hands_off_and_tail_jumps_to_the_resolved_target() {
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEEN_VM: AtomicU64 = AtomicU64::new(0);
        static SEEN_RET: AtomicU64 = AtomicU64::new(0);
        static SEEN_ARG0: AtomicU64 = AtomicU64::new(0);
        static SEEN_ARG1: AtomicU64 = AtomicU64::new(0);
        static TARGET: AtomicU64 = AtomicU64::new(0);

        // Stands in for rt_resolve_send: records what it was handed and
        // answers the address of the "resolved method".
        extern "C" fn resolve_probe(vm: u64, ret_addr: u64, argv: *mut u64) -> u64 {
            SEEN_VM.store(vm, Ordering::Relaxed);
            SEEN_RET.store(ret_addr, Ordering::Relaxed);
            unsafe {
                SEEN_ARG0.store(*argv, Ordering::Relaxed);
                SEEN_ARG1.store(*argv.add(1), Ordering::Relaxed);
            }
            TARGET.load(Ordering::Relaxed)
        }
        // The "resolved method": proves it received the original
        // arguments, and its `ret` must reach the original caller.
        extern "C" fn resolved(a: u64, b: u64) -> u64 {
            a.wrapping_mul(100).wrapping_add(b)
        }

        let stub = build_stub_resolve_x64(resolve_probe as usize as u64);

        // Caller: set R15 (vm), call the stub with two arguments, return
        // whatever comes back. If the tail-jump were a `call`, or the
        // frame were left behind, this would not return cleanly at all.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(R15)]);
        h.emit("sub", &[r64(RSP), imm(40)]);
        h.emit("mov", &[r64(R10), r64(RCX)]); // stub
        h.emit("mov", &[r64(R15), r64(RDX)]); // vm
        h.emit("mov", &[r64(RCX), r64(R8)]); // arg0
        h.emit("mov", &[r64(RDX), r64(R9)]); // arg1
        h.emit("call", &[r64(R10)]);
        h.emit("add", &[r64(RSP), imm(40)]);
        h.emit("pop", &[r64(R15)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();

        let total = stub.code.len() + harness.code.len() + 4096;
        let jit = WinJit::with_capacity(total).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let hoff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(hoff),
                harness.code.len(),
            );
        }
        TARGET.store(resolved as usize as u64, Ordering::Relaxed);

        // A real, writable VM register block — NOT a sentinel value. The
        // stub publishes its frame pointer through R15 so a GC running
        // inside the lookup can walk it, so R15 must be a genuine
        // pointer. (Passing a fake one here is an instant access
        // violation, which is how this test first failed.)
        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        let hf: extern "C" fn(u64, u64, u64, u64) -> u64 =
            unsafe { std::mem::transmute(base.add(hoff)) };
        let got = hf(base as u64, vm, 7, 9);

        // The resolved target ran, with the ORIGINAL arguments, and its
        // return reached the original caller through the harness.
        assert_eq!(got, 7 * 100 + 9, "tail-jumped target's result reached the caller");
        // The runtime saw the pinned VM register...
        assert_eq!(SEEN_VM.load(Ordering::Relaxed), vm, "vm from R15");
        // ...the arguments, in slot order, through argv...
        assert_eq!(SEEN_ARG0.load(Ordering::Relaxed), 7, "argv[0]");
        assert_eq!(SEEN_ARG1.load(Ordering::Relaxed), 9, "argv[1]");
        // ...and a return address that points INTO the harness, just past
        // its call instruction — which is what identifies the IC site.
        // The walker record is cleared once the runtime call is done, so
        // a later GC does not walk a frame that no longer exists.
        use crate::oops::layout::VMREG_LAST_COMPILED_FP_OFFSET;
        assert_eq!(
            vmreg[VMREG_LAST_COMPILED_FP_OFFSET / 8], 0,
            "last_compiled_fp must be cleared before the tail-jump"
        );

        let ret = SEEN_RET.load(Ordering::Relaxed);
        let h_lo = base as u64 + hoff as u64;
        let h_hi = h_lo + harness.code.len() as u64;
        assert!(
            ret > h_lo && ret < h_hi,
            "ret_addr {ret:#x} must point inside the calling code              ({h_lo:#x}..{h_hi:#x}), not into a link register's stale value"
        );
    }

    /// The deopt RETURN trampoline, driven the way it is really reached:
    /// a victim frame calls a callee whose saved return address has been
    /// redirected here, so the callee's ordinary `ret` lands in the
    /// trampoline with its result in `RAX` and `RBP` already restored to
    /// the victim's frame pointer.
    ///
    /// What the test pins, none of which is visible from the return value
    /// alone:
    /// - both runtime calls receive `victim_fp`, and the second also the
    ///   callee's result;
    /// - the bridged frame's return-address slot holds `orig_ret_pc` (a pc
    ///   inside the victim), which is what makes a GC classify the victim
    ///   at its true safepoint;
    /// - the trampoline unwinds the VICTIM frame, so the deoptee's result
    ///   reaches the victim's caller — a `0xBAD` from the victim's own
    ///   tail would mean it returned there instead.
    #[cfg(windows)]
    #[test]
    fn deopt_return_trampoline_unwinds_the_victim_frame() {
        use crate::oops::layout::{VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_PC_OFFSET};
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEEN_FP_1: AtomicU64 = AtomicU64::new(0);
        static SEEN_FP_2: AtomicU64 = AtomicU64::new(0);
        static SEEN_RESULT: AtomicU64 = AtomicU64::new(0);
        static SEEN_BRIDGE_PC: AtomicU64 = AtomicU64::new(0);
        const ORIG_RET_PC: u64 = 0xC0DE_1234;

        extern "C" fn return_pc_probe(_vm: u64, victim_fp: u64) -> u64 {
            SEEN_FP_1.store(victim_fp, Ordering::Relaxed);
            ORIG_RET_PC
        }
        extern "C" fn on_return_probe(vm: u64, victim_fp: u64, result: u64) -> u64 {
            SEEN_FP_2.store(victim_fp, Ordering::Relaxed);
            SEEN_RESULT.store(result, Ordering::Relaxed);
            // The bridged frame must be published with orig_ret_pc.
            let vmreg = vm as *const u64;
            SEEN_BRIDGE_PC.store(
                unsafe { *vmreg.add(VMREG_LAST_COMPILED_PC_OFFSET / 8) },
                Ordering::Relaxed,
            );
            0xDEE0 // the deoptee's result
        }

        let tramp = build_deopt_return_trampoline_x64(
            return_pc_probe as usize as u64,
            on_return_probe as usize as u64,
        );

        // The "callee": returns a result. Its return address will be
        // redirected to the trampoline by the victim below.
        let mut c = X64Assembler::new();
        c.emit("mov", &[r64(RAX), imm(0x77)]);
        c.emit("ret", &[]);
        let callee = c.finish();

        // The "victim": a normal compiled frame that calls the callee,
        // but overwrites its own pushed return address so the callee
        // returns into the trampoline instead of back here.
        let mut v = X64Assembler::new();
        v.emit("push", &[r64(RBP)]);
        v.emit("mov", &[r64(RBP), r64(RSP)]);
        v.emit("sub", &[r64(RSP), imm(32)]);
        // rcx = callee, rdx = trampoline
        v.emit("mov", &[r64(R10), r64(RCX)]);
        v.emit("mov", &[r64(R11), r64(RDX)]);
        v.emit("call", &[r64(R10)]);
        // Only reached if the trampoline wrongly returned to the victim.
        v.emit("mov", &[r64(RAX), imm(0xBAD)]);
        v.emit("mov", &[r64(RSP), r64(RBP)]);
        v.emit("pop", &[r64(RBP)]);
        v.emit("ret", &[]);
        let victim = v.finish();

        let total = tramp.code.len() + callee.code.len() + victim.code.len() + 4096;
        let jit = WinJit::with_capacity(total).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let c_off = (tramp.code.len() + 15) & !15;
        let v_off = (c_off + callee.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(tramp.code.as_ptr(), base, tramp.code.len());
            core::ptr::copy_nonoverlapping(callee.code.as_ptr(), base.add(c_off), callee.code.len());
            core::ptr::copy_nonoverlapping(victim.code.as_ptr(), base.add(v_off), victim.code.len());
        }

        // A harness that sets R15 and calls the victim; it also performs
        // the return-address redirection the runtime would do, by
        // patching the callee to `jmp` the trampoline instead of `ret`.
        // (Simpler and equivalent: the callee's `ret` target IS the
        // redirected slot.) Build a callee that jumps to the trampoline
        // with its result already in RAX — exactly the state a redirected
        // `ret` produces.
        let mut c2 = X64Assembler::new();
        c2.emit("mov", &[r64(RAX), imm(0x77)]);
        c2.emit("mov", &[r64(RSP), r64(RBP)]); // pop this callee's own frame-less state
        c2.emit("jmp", &[r64(R11)]); // R11 = trampoline, as the victim set it
        let callee2 = c2.finish();
        assert!(callee2.code.len() <= callee.code.len() + 32);
        unsafe {
            core::ptr::copy_nonoverlapping(
                callee2.code.as_ptr(),
                base.add(c_off),
                callee2.code.len(),
            );
        }

        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(VM_STATE)]);
        h.emit("sub", &[r64(RSP), imm(40)]);
        h.emit("mov", &[r64(RAX), r64(RCX)]); // victim
        h.emit("mov", &[r64(VM_STATE), r64(RDX)]); // vm
        h.emit("mov", &[r64(RCX), r64(R8)]); // callee
        h.emit("mov", &[r64(RDX), r64(R9)]); // trampoline
        h.emit("call", &[r64(RAX)]);
        h.emit("add", &[r64(RSP), imm(40)]);
        h.emit("pop", &[r64(VM_STATE)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();
        let h_off = (v_off + victim.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(h_off),
                harness.code.len(),
            );
        }

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        let hf: extern "C" fn(u64, u64, u64, u64) -> u64 =
            unsafe { std::mem::transmute(base.add(h_off)) };
        let got = hf(
            base as u64 + v_off as u64,
            vm,
            base as u64 + c_off as u64,
            base as u64,
        );

        assert_eq!(
            got, 0xDEE0,
            "the deoptee's result must reach the victim's caller — 0xBAD would \
             mean the trampoline returned into the victim instead of unwinding it"
        );
        assert_eq!(SEEN_RESULT.load(Ordering::Relaxed), 0x77, "callee's result");
        assert_ne!(SEEN_FP_1.load(Ordering::Relaxed), 0, "victim_fp to call 1");
        assert_eq!(
            SEEN_FP_1.load(Ordering::Relaxed),
            SEEN_FP_2.load(Ordering::Relaxed),
            "both runtime calls must key on the SAME victim frame"
        );
        assert_eq!(
            SEEN_BRIDGE_PC.load(Ordering::Relaxed),
            ORIG_RET_PC,
            "the bridged frame must publish orig_ret_pc, so a GC classifies \
             the victim at its true safepoint"
        );
        assert_eq!(
            vmreg[VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0,
            "walker record cleared once the bridge is over"
        );
    }


    /// Every stub prologue must publish ALL THREE walker fields — fp,
    /// pc, and kind — before it can reach Rust.
    ///
    /// The pc store was missing on x64 for the whole of Phase 3, and
    /// nothing caught it, because a missing store is not a wrong value:
    /// `last_compiled_pc` simply kept whatever the previous writer left
    /// there. Every stub test asserted `fp`, several asserted the kind,
    /// none asserted the pc — so the field was read by
    /// `rt_interpret_call` as its caller's return address, pointing into
    /// a stale, unrelated method.
    ///
    /// This runs a real stub through the real call stub and reads all
    /// three back, so a future prologue that drops any one of them fails
    /// here rather than in a Smalltalk program days later.
    #[cfg(windows)]
    #[test]
    fn every_stub_prologue_publishes_fp_pc_and_kind() {
        use crate::oops::layout::{
            VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET,
            VMREG_LAST_COMPILED_PC_OFFSET,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEEN_FP: AtomicU64 = AtomicU64::new(0);
        static SEEN_PC: AtomicU64 = AtomicU64::new(0);
        static SEEN_KIND: AtomicU64 = AtomicU64::new(0);

        // Reads the walker record mid-stub, exactly where a GC or a
        // runtime function like `rt_interpret_call` would.
        extern "C" fn peek(vm: *const u64, _a: u64) -> u64 {
            // SAFETY: `vm` is the test's own register block.
            unsafe {
                SEEN_FP.store(*vm.add(VMREG_LAST_COMPILED_FP_OFFSET / 8), Ordering::Relaxed);
                SEEN_PC.store(*vm.add(VMREG_LAST_COMPILED_PC_OFFSET / 8), Ordering::Relaxed);
                SEEN_KIND.store(
                    *vm.add(VMREG_LAST_COMPILED_KIND_OFFSET / 8),
                    Ordering::Relaxed,
                );
            }
            0
        }

        let stub = build_stub_must_be_boolean_x64(peek as usize as u64);

        // A caller that CALLs the stub, so there is a genuine return
        // address to publish, and returns its own call-site address so
        // the test can compare against what the stub reported.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(VM_STATE)]);
        h.emit("sub", &[r64(RSP), imm(40)]);
        h.emit("mov", &[r64(VM_STATE), r64(RDX)]); // vm
        h.emit("mov", &[r64(R10), r64(RCX)]); // stub
        h.emit("call", &[r64(R10)]);
        let after_call = h.offset();
        h.emit("add", &[r64(RSP), imm(40)]);
        h.emit("pop", &[r64(VM_STATE)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();

        // Both blobs in one region, stub first.
        use crate::vendor::wfasm::native_windows::WinJit;
        let jit = WinJit::with_capacity(stub.code.len() + harness.code.len() + 4096)
            .expect("RWX");
        let (base, _cap) = jit.region_raw();
        let h_off = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(h_off),
                harness.code.len(),
            );
        }
        let stub_addr = base as u64;
        let harness_addr = base as u64 + h_off as u64;

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        let f: extern "C" fn(u64, u64) -> u64 =
            unsafe { std::mem::transmute(harness_addr as *const u8) };
        f(stub_addr, vm);

        assert_ne!(SEEN_FP.load(Ordering::Relaxed), 0, "fp published");
        assert_eq!(
            SEEN_KIND.load(Ordering::Relaxed),
            crate::codecache::stubs::KIND_MUST_BE_BOOLEAN,
            "kind published"
        );
        assert_eq!(
            SEEN_PC.load(Ordering::Relaxed),
            harness_addr + after_call as u64,
            "pc published, and it is the RETURN ADDRESS of the call that \
             entered the stub — the value a caller-site lookup depends on"
        );
        assert_eq!(
            vmreg[VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0,
            "walker record cleared on the way out"
        );
    }

    /// `value_dispatch` takes two different tails, and only running both
    /// distinguishes them: the fast path TAIL-JUMPS to a compiled closure
    /// (so the closure's own `ret` reaches this stub's caller, with the
    /// stub gone), while the fallback RETURNS an interpreted result
    /// normally. A stub that used one tail for both would still produce a
    /// plausible value on one of the two paths.
    #[cfg(windows)]
    #[test]
    fn value_dispatch_tail_jumps_on_hit_and_returns_on_fallback() {
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::{AtomicU64, Ordering};

        static TARGET: AtomicU64 = AtomicU64::new(0);
        static FB_ARGV0: AtomicU64 = AtomicU64::new(0);
        static FB_ARGC: AtomicU64 = AtomicU64::new(0);
        static SEEN_CLOSURE: AtomicU64 = AtomicU64::new(0);
        static SEEN_ARGC: AtomicU64 = AtomicU64::new(0);

        extern "C" fn value_target(_vm: u64, closure: u64, argc: u64) -> u64 {
            SEEN_CLOSURE.store(closure, Ordering::Relaxed);
            SEEN_ARGC.store(argc, Ordering::Relaxed);
            TARGET.load(Ordering::Relaxed) // 0 selects the fallback
        }
        extern "C" fn value_fallback(_vm: u64, argv: *const u64, argc: u64) -> u64 {
            FB_ARGV0.store(unsafe { *argv }, Ordering::Relaxed);
            FB_ARGC.store(argc, Ordering::Relaxed);
            0xFA11
        }
        // The "compiled closure" the fast path jumps to.
        extern "C" fn compiled_closure(_a: u64) -> u64 {
            0xC105
        }

        let stub = build_stub_value_dispatch_x64(
            value_target as usize as u64,
            value_fallback as usize as u64,
            1,
            crate::codecache::stubs::KIND_VALUE_DISPATCH,
        );

        let jit = WinJit::with_capacity(stub.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len()) };

        // Harness: plant R15, call the stub with the closure in arg0.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(R15)]);
        h.emit("sub", &[r64(RSP), imm(40)]);
        h.emit("mov", &[r64(R10), r64(RCX)]);
        h.emit("mov", &[r64(R15), r64(RDX)]);
        h.emit("mov", &[r64(RCX), r64(R8)]);
        h.emit("call", &[r64(R10)]);
        h.emit("add", &[r64(RSP), imm(40)]);
        h.emit("pop", &[r64(R15)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let hb = h.finish();
        let jit2 = WinJit::with_capacity(hb.code.len() + 4096).expect("RWX");
        let (hbase, _) = jit2.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(hb.code.as_ptr(), hbase, hb.code.len()) };
        let hf: extern "C" fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(hbase) };

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        const CLOSURE: u64 = 0xC0C0_0001;

        // Fast path: a compiled entry exists -> tail-jump to it.
        TARGET.store(compiled_closure as usize as u64, Ordering::Relaxed);
        assert_eq!(
            hf(base as u64, vm, CLOSURE),
            0xC105,
            "the tail-jumped closure's result must reach the stub's caller"
        );
        assert_eq!(SEEN_CLOSURE.load(Ordering::Relaxed), CLOSURE);
        assert_eq!(SEEN_ARGC.load(Ordering::Relaxed), 1, "argc is baked in");

        // Fallback: no compiled entry -> interpret and RETURN.
        TARGET.store(0, Ordering::Relaxed);
        assert_eq!(hf(base as u64, vm, CLOSURE), 0xFA11, "fallback returns");
        assert_eq!(
            FB_ARGV0.load(Ordering::Relaxed),
            CLOSURE,
            "the fallback's argv points at the RootSpill, so it interprets the              GC-visible copies rather than stale registers"
        );
        assert_eq!(FB_ARGC.load(Ordering::Relaxed), 1);
        assert_eq!(
            vmreg[crate::oops::layout::VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0,
            "walker record cleared on both paths"
        );
    }

    /// The stub saves the Win64 callee-saved GPRs but deliberately not
    /// `XMM6–15`, because the x64 FP register file currently excludes
    /// them. Those two facts must move together: the day Phase 5 puts a
    /// callee-saved XMM into the pool, this stub starts corrupting the
    /// Rust caller's floats, silently. Fail here instead.
    /// The XMM bank must actually SURVIVE a compiled call.
    ///
    /// `fp_pool_is_empty_or_this_stub_must_save_xmm` only checks that the
    /// pool and `SAVED_XMM` agree with each other — both could be
    /// consistent and the save/restore still be wrong (wrong offset,
    /// wrong width, `movaps` on an unaligned area). This runs it: plant a
    /// distinct sentinel in every callee-saved XMM, call through the real
    /// stub into code that deliberately clobbers all of them, and require
    /// every sentinel back.
    ///
    /// Only the low 64 bits are checked, which is all a `movsd` sentinel
    /// can carry — but the save is 128-bit `movups`, so a width mistake
    /// would still show up as a fault or a wrong low half.
    #[cfg(windows)]
    #[test]
    #[allow(unsafe_code)]
    fn call_stub_preserves_the_callee_saved_xmm_bank() {
        use crate::compiler::assembler_x64::xmm;
        use crate::vendor::wfasm::native_windows::WinJit;

        // The "compiled method": trash every callee-saved XMM, return 0.
        let mut m = X64Assembler::new();
        m.emit("push", &[r64(RBP)]);
        m.emit("mov", &[r64(RBP), r64(RSP)]);
        m.emit("mov", &[r64(RAX), imm(-1)]);
        for r in SAVED_XMM {
            m.emit("movq", &[xmm(r), r64(RAX)]);
        }
        m.emit("xor", &[r64(RAX), r64(RAX)]);
        m.emit("mov", &[r64(RSP), r64(RBP)]);
        m.emit("pop", &[r64(RBP)]);
        m.emit("ret", &[]);
        let method = m.finish();

        let stub = build_call_stub_x64();

        // Harness: load sentinels into xmm6-15 from a buffer, call the
        // stub, then store them all back out for inspection.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(RSP)]);
        h.emit("push", &[r64(RBX)]);
        h.emit("push", &[r64(RSI)]);
        h.emit("sub", &[r64(RSP), imm(56)]);
        h.emit("mov", &[r64(RBX), r64(RCX)]); // stub
        h.emit("mov", &[r64(RSI), r64(R9)]); // sentinel buffer
        for (i, r) in SAVED_XMM.enumerate() {
            h.emit("movsd", &[xmm(r), mem(RSI, 8 * i as i64)]);
        }
        // call_stub(entry=RDX, vm=R8, argv=null, argc=0)
        h.emit("mov", &[r64(RCX), r64(RDX)]);
        h.emit("mov", &[r64(RDX), r64(R8)]);
        h.emit("xor", &[r64(R8), r64(R8)]);
        h.emit("xor", &[r64(R9), r64(R9)]);
        h.emit("call", &[r64(RBX)]);
        for (i, r) in SAVED_XMM.enumerate() {
            h.emit("movsd", &[mem(RSI, 8 * i as i64), xmm(r)]);
        }
        h.emit("add", &[r64(RSP), imm(56)]);
        h.emit("pop", &[r64(RSI)]);
        h.emit("pop", &[r64(RBX)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();

        let total = stub.code.len() + method.code.len() + harness.code.len() + 4096;
        let jit = WinJit::with_capacity(total).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        let hoff = (moff + method.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(method.code.as_ptr(), base.add(moff), method.code.len());
            core::ptr::copy_nonoverlapping(
                harness.code.as_ptr(),
                base.add(hoff),
                harness.code.len(),
            );
        }

        let mut vmreg = [0u64; 16];
        // Distinct, non-canonical sentinels so a swap is as visible as a loss.
        let mut sentinels: Vec<u64> = (0..10).map(|i| 0xF00D_0000_u64 + i as u64).collect();
        let expected = sentinels.clone();

        let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
            unsafe { std::mem::transmute(base.add(hoff)) };
        f(
            base as u64,
            base as u64 + moff as u64,
            vmreg.as_mut_ptr() as u64,
            sentinels.as_mut_ptr() as u64,
        );

        for (i, (got, want)) in sentinels.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got,
                want,
                "xmm{} was not restored by the call stub — the FP pool includes it, \
                 so compiled code clobbering it corrupts the Rust caller's floats",
                SAVED_XMM.start as usize + i
            );
        }
    }

    #[test]
    fn fp_pool_is_empty_or_this_stub_must_save_xmm() {
        // The pool and the save bank must stay consistent. Every FP
        // register the allocator may hand out is either volatile under
        // Win64 (xmm0-5, free to clobber) or inside SAVED_XMM (preserved
        // by this stub). A register in neither set silently corrupts the
        // Rust caller's floats.
        const FP_MAX_VOLATILE: u8 = 5;
        for r in crate::compiler::regalloc::fp_allocatable_regs() {
            assert!(
                *r <= FP_MAX_VOLATILE || SAVED_XMM.contains(r),
                "xmm{r} is in the FP allocatable pool but is neither volatile nor saved \\
                 by build_call_stub_x64 — add it to SAVED_XMM first"
            );
        }
        // And the converse: the save bank must not have drifted past the
        // 16 registers that exist, nor claim a volatile one it needn't.
        assert_eq!(SAVED_XMM.end, 16, "x86-64 has xmm0-15");
        assert!(SAVED_XMM.start > FP_MAX_VOLATILE, "saving a volatile register wastes work");
        assert_eq!(
            XMM_SAVE_BYTES,
            16 * (SAVED_XMM.end - SAVED_XMM.start) as i64,
            "the reserved area must match the number of registers saved"
        );
    }
}
