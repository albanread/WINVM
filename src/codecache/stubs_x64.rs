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
//!   allocatable pool) and pins `R12–R15`, so all seven are saved here.
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

use crate::compiler::assembler::CodeBlob;
use crate::compiler::assembler_x64::{
    imm, mem, r64, Cond, X64Assembler, ARG_REGS, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX,
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

/// Shadow space (32, mandatory on Win64) plus 8 bytes of realignment —
/// see the module header's alignment arithmetic.
const CALL_AREA: i64 = 40;

/// Build the x86-64 call stub.
pub fn build_call_stub_x64() -> CodeBlob {
    let mut a = X64Assembler::new();

    // ── Prologue: frame + callee-saved bank ─────────────────────────────
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    for r in SAVED {
        a.emit("push", &[r64(r)]);
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
    a.emit("call", &[r64(R10)]);
    a.emit("add", &[r64(RSP), imm(CALL_AREA)]);
    // The compiled method's result is already in RAX, which is also this
    // stub's return register — nothing to move.

    // ── Epilogue ────────────────────────────────────────────────────────
    for r in SAVED.iter().rev() {
        a.emit("pop", &[r64(*r)]);
    }
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);

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
        let method = emit_x64(&add_method(), &regalloc(&add_method()), RuntimeAddrs::default(), None).blob;

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
        let method = emit_x64(&add_method(), &regalloc(&add_method()), RuntimeAddrs::default(), None).blob;

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
             compiled code writes RBX/RSI/RDI and pins R12-R15, so every one of \
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

    /// The stub saves the Win64 callee-saved GPRs but deliberately not
    /// `XMM6–15`, because the x64 FP register file currently excludes
    /// them. Those two facts must move together: the day Phase 5 puts a
    /// callee-saved XMM into the pool, this stub starts corrupting the
    /// Rust caller's floats, silently. Fail here instead.
    #[test]
    fn fp_pool_is_empty_or_this_stub_must_save_xmm() {
        // Mirrors regalloc's FP_ALLOCATABLE_REGS; kept as a literal so the
        // assertion is about the VALUE, not about importing the constant.
        const FP_POOL_MAX_VOLATILE: u8 = 5; // xmm0..xmm5 are volatile on Win64
        for r in crate::compiler::regalloc::fp_allocatable_regs() {
            assert!(
                *r <= FP_POOL_MAX_VOLATILE,
                "xmm{r} is callee-saved on Win64 but is in the FP allocatable pool, and \
                 build_call_stub_x64 does not save it — add the save/restore here first"
            );
        }
    }
}
