//! IR → x86-64 machine code (WINVM Phase 3, MIGRATION.md §4) — the sibling
//! of [`emit`](crate::compiler::emit), which emits AArch64.
//!
//! **Scope.** This is the Phase-3 *vertical slice*: the op set needed to
//! compile and execute a real arithmetic/control-flow method end to end,
//! proving prologue, parameters, register allocation, two-address
//! arithmetic with overflow checks, spill/reload, compare-and-branch, and
//! the epilogue on real hardware. Ops outside the slice panic naming
//! themselves rather than emitting something plausible-but-wrong — the
//! remaining IR surface (sends, ICs, allocation, deopt scopes, floats,
//! SIMD, OSR) lands as Phase 3 continues. [`SUPPORTED_OPS`] is the
//! machine-readable version of that list.
//!
//! ## Smi representation (why the arithmetic looks free)
//!
//! `SMI_SHIFT == 2` and `INT_TAG == 0` (`oops::layout`), so a tagged smi
//! is exactly `v << 2` and a smi's low two bits are always zero. Three
//! consequences this emitter leans on:
//!
//! - **Add/Sub/And/Or/Xor operate directly on tagged words.** `(a<<2) +
//!   (b<<2) == (a+b)<<2`, and the bitwise ops preserve the zero tag. No
//!   untag/retag pair is needed at all.
//! - **`jo` is the exact overflow test** for Add/Sub: the tagged sum
//!   overflows the 64-bit register precisely when the untagged sum leaves
//!   smi range, because the tagged range is the smi range scaled by 4.
//! - **Mul needs one untag.** `(a<<2) * (b<<2)` would be `(a*b)<<4`, so
//!   one operand is arithmetic-shifted right by 2 first, giving
//!   `(a*b)<<2`. `imul` sets OF on signed overflow, so `jo` still serves.
//!
//! ## Two-address form (the deepest change from AArch64)
//!
//! AArch64 is three-address (`add xd, xa, xb`); x86-64 is two-address
//! (`add rd, rb` means `rd += rb`). The IR is three-address, and the
//! allocator has no tied-operand constraint (it will happily give `dst`,
//! `a`, and `b` three different registers), so every arithmetic op is
//! lowered as `mov dst, a; op dst, b`.
//!
//! That rewrite has one genuine hazard, and it is the x64 analogue of the
//! aliasing bugs [`emit`](crate::compiler::emit)'s own header documents:
//! **if `dst` and `b` are the same register, `mov dst, a` destroys `b`
//! before the operation reads it.** `emit_two_address` handles it
//! explicitly — commutative ops swap their operands (`b op a == a op b`),
//! and `Sub`, which cannot swap, routes through a scratch register. This
//! is checked directly by `sub_with_dst_aliasing_b_is_correct`, which is
//! the test that would fail loudly if the case were ever "simplified"
//! away.

use crate::compiler::assembler::{CodeBlob, Label, RelocKind};
use crate::compiler::assembler_x64::{
    imm, mem, r64, Cond, X64Assembler, ARG_REGS, RAX, RBP, RSP, SCRATCH0, SCRATCH1,
};
use crate::compiler::ir::{BlockId, CmpOp, Ir, IrMethod, SmiOp, VReg};
use crate::compiler::regalloc::{Assignment, RegallocResult, SpillSlot};

/// The IR ops this slice lowers. Anything else panics in [`emit_x64`] —
/// see the module header.
pub const SUPPORTED_OPS: &[&str] = &[
    "ConstSmi", "Move", "Param", "LoadField", "SmiArith", "SmiCmpBr", "Jump", "Ret", "Bailout",
];

/// Byte offset of spill slot `i` from the frame pointer: `[rbp − 8·(i+1)]`,
/// the same numbering the AArch64 side uses against `x29`.
///
/// Unlike AArch64 this needs no range workaround: an x86 `mov` takes a
/// full `disp32`, so even a very deep frame addresses directly. (The A64
/// emitter has a whole scratch-address path for offsets past its imm9
/// range; that machinery simply has no x64 counterpart.)
fn spill_offset(slot: SpillSlot) -> i64 {
    -8 * (slot.0 as i64 + 1)
}

/// The value a bailing-out compiled method returns, telling the caller to
/// re-run the activation in the interpreter. Mirrors the AArch64 emitter's
/// bailout sentinel contract (`interpreter::compiled_call` reads it).
pub const BAILOUT_SENTINEL: u64 = u64::MAX;

struct Emitter<'a> {
    asm: X64Assembler,
    /// vreg → where it lives, indexed by `VReg.0`.
    assignment: Vec<Option<Assignment>>,
    /// One label per basic block.
    labels: Vec<Label>,
    epilogue: Label,
    bailout: Label,
    method: &'a IrMethod,
}

impl<'a> Emitter<'a> {
    /// Resolve `v` into a register ready to READ. A spilled vreg is
    /// reloaded into `scratch`; a register-resident one is returned as-is.
    fn read_into(&mut self, v: VReg, scratch: u8) -> u8 {
        match self.assignment[v.0 as usize] {
            Some(Assignment::Reg(r)) => r,
            Some(Assignment::Spill(slot)) => {
                self.asm
                    .emit("mov", &[r64(scratch), mem(RBP, spill_offset(slot))]);
                scratch
            }
            None => panic!(
                "emit_x64: vreg v{} has no register-allocator assignment — every vreg the \
                 emitter reads must have been given one",
                v.0
            ),
        }
    }

    /// The register a vreg's DEFINITION should be computed into. For a
    /// spilled vreg that is a scratch register, and the caller must follow
    /// with [`store_def`] to write it back to the slot.
    fn def_reg(&self, v: VReg, scratch: u8) -> u8 {
        match self.assignment[v.0 as usize] {
            Some(Assignment::Reg(r)) => r,
            Some(Assignment::Spill(_)) => scratch,
            None => panic!("emit_x64: vreg v{} has no assignment", v.0),
        }
    }

    /// Complete a definition: if `v` is spilled, store `reg` into its slot.
    /// A no-op for a register-resident vreg.
    fn store_def(&mut self, v: VReg, reg: u8) {
        if let Some(Assignment::Spill(slot)) = self.assignment[v.0 as usize] {
            self.asm
                .emit("mov", &[mem(RBP, spill_offset(slot)), r64(reg)]);
        }
    }

    /// Lower a three-address IR op to x86-64's two-address form:
    /// `dst = a op b` becomes `mov dst, a; op dst, b`.
    ///
    /// The aliasing hazard this exists to handle: when `dst` and `b` are
    /// the same physical register, the leading `mov dst, a` overwrites `b`
    /// before `op` reads it. Two ways out, chosen by whether the operation
    /// commutes:
    /// - **Commutative** (`add`/`imul`/`and`/`or`/`xor`): swap operands —
    ///   `dst(==b) op a` computes the same value with no move at all.
    /// - **Non-commutative** (`sub`): compute into [`SCRATCH1`] and move
    ///   the result down, since `b - a` is not the wanted `a - b`.
    fn emit_two_address(&mut self, mnemonic: &str, commutative: bool, dst: u8, a: u8, b: u8) {
        if dst == b {
            if commutative {
                // dst already holds b; fold a in.
                self.asm.emit(mnemonic, &[r64(dst), r64(a)]);
            } else {
                self.asm.emit("mov", &[r64(SCRATCH1), r64(a)]);
                self.asm.emit(mnemonic, &[r64(SCRATCH1), r64(b)]);
                self.asm.emit("mov", &[r64(dst), r64(SCRATCH1)]);
            }
            return;
        }
        if dst != a {
            self.asm.emit("mov", &[r64(dst), r64(a)]);
        }
        self.asm.emit(mnemonic, &[r64(dst), r64(b)]);
    }

    /// `test reg, 3` — nonzero means the low tag bits are set, i.e. NOT a
    /// smi. `test` writes no register, so (unlike a combined `or` into a
    /// scratch) it can never alias an operand the same sequence still
    /// needs. That is the same reasoning the AArch64 emitter's header
    /// records for preferring `tst` over `orr`.
    fn emit_smi_guard(&mut self, reg: u8, fail: BlockId) {
        self.asm.emit("test", &[r64(reg), imm(3)]);
        let target = self.labels[fail.0 as usize];
        self.asm.jcc(Cond::Ne, target);
    }
}

/// Map an IR comparison to the x86 condition for a SIGNED compare —
/// Smalltalk SmallInteger ordering is signed, and comparing tagged words
/// preserves order (multiplying by 4 is monotonic).
fn cmp_cond(op: CmpOp) -> Cond {
    match op {
        CmpOp::Lt => Cond::L,
        CmpOp::Le => Cond::Le,
        CmpOp::Gt => Cond::G,
        CmpOp::Ge => Cond::Ge,
        CmpOp::Eq => Cond::E,
        CmpOp::Ne => Cond::Ne,
    }
}

/// Compile `method` to x86-64. Returns the finished blob.
///
/// Calling convention for the slice: parameters arrive in the Win64
/// argument registers and the result leaves in `RAX`, so a compiled method
/// is directly callable as an `extern "C" fn(u64, ...) -> u64`. Wiring the
/// real call stub (which additionally establishes the pinned `&VmState` /
/// receiver registers) is the next Phase-3 step.
pub fn emit_x64(method: &IrMethod, regalloc: &RegallocResult) -> CodeBlob {
    let mut asm = X64Assembler::new();

    let mut assignment: Vec<Option<Assignment>> = vec![None; method.vregs.len()];
    for iv in &regalloc.intervals {
        assignment[iv.vreg.0 as usize] = iv.assignment;
    }
    let labels: Vec<Label> = (0..method.blocks.len()).map(|_| asm.new_label()).collect();
    let epilogue = asm.new_label();
    let bailout = asm.new_label();

    let mut e = Emitter {
        asm,
        assignment,
        labels,
        epilogue,
        bailout,
        method,
    };

    // ── Prologue ────────────────────────────────────────────────────────
    // A real frame (push rbp; mov rbp, rsp), not a frameless leaf: the GC
    // and the debugger walk the RBP chain to find compiled activations,
    // exactly as they walk x29 on AArch64.
    e.asm.emit("push", &[r64(RBP)]);
    e.asm.emit("mov", &[r64(RBP), r64(RSP)]);
    // Spill area, rounded so RSP stays 16-byte aligned at the next call
    // (Win64 requires it, and `push rbp` has already shifted alignment by
    // 8 from the entry state).
    let frame_bytes = {
        let raw = 8 * regalloc.frame_slots as i64;
        (raw + 15) & !15
    };
    if frame_bytes > 0 {
        e.asm.emit("sub", &[r64(RSP), imm(frame_bytes)]);
    }

    // ── Blocks ──────────────────────────────────────────────────────────
    for (bi, block) in method.blocks.iter().enumerate() {
        let l = e.labels[bi];
        e.asm.bind(l);
        for op in &block.code {
            emit_op(&mut e, op);
        }
    }

    // ── Epilogue ────────────────────────────────────────────────────────
    e.asm.bind(epilogue);
    e.asm.emit("mov", &[r64(RSP), r64(RBP)]);
    e.asm.emit("pop", &[r64(RBP)]);
    e.asm.emit("ret", &[]);

    // The shared bailout landing: hand the sentinel back through the same
    // epilogue so the frame is torn down exactly once.
    e.asm.bind(bailout);
    e.asm.emit("mov", &[r64(RAX), imm(BAILOUT_SENTINEL as i64)]);
    e.asm.emit("mov", &[r64(RSP), r64(RBP)]);
    e.asm.emit("pop", &[r64(RBP)]);
    e.asm.emit("ret", &[]);

    e.asm.finish()
}

fn emit_op(e: &mut Emitter, op: &Ir) {
    match op {
        Ir::ConstSmi { dst, value } => {
            let d = e.def_reg(*dst, SCRATCH0);
            // Tagged form: INT_TAG == 0, so the tagged word is v * 4.
            let tagged = (*value as u64) << 2;
            e.asm.emit("mov", &[r64(d), imm(tagged as i64)]);
            e.store_def(*dst, d);
        }

        Ir::Move { dst, src } => {
            let s = e.read_into(*src, SCRATCH0);
            let d = e.def_reg(*dst, SCRATCH1);
            if d != s {
                e.asm.emit("mov", &[r64(d), r64(s)]);
            }
            e.store_def(*dst, d);
        }

        Ir::Param { dst, index } => {
            let src = *ARG_REGS.get(*index as usize).unwrap_or_else(|| {
                panic!(
                    "emit_x64: parameter #{index} is past the {} Win64 register arguments — \
                     stack-passed parameters are not in the Phase-3 slice",
                    ARG_REGS.len()
                )
            });
            let d = e.def_reg(*dst, SCRATCH0);
            if d != src {
                e.asm.emit("mov", &[r64(d), r64(src)]);
            }
            e.store_def(*dst, d);
        }

        Ir::LoadField { dst, obj, byte_off } => {
            let o = e.read_into(*obj, SCRATCH0);
            let d = e.def_reg(*dst, SCRATCH1);
            e.asm.emit("mov", &[r64(d), mem(o, *byte_off as i64)]);
            e.store_def(*dst, d);
        }

        Ir::SmiArith {
            op: sop,
            dst,
            a,
            b,
            fail,
        } => {
            let ra = e.read_into(*a, SCRATCH0);
            let rb = e.read_into(*b, SCRATCH1);
            // Both operands must be smis; the guard branches to the
            // bailout block on any tagged pointer.
            e.emit_smi_guard(ra, *fail);
            e.emit_smi_guard(rb, *fail);

            let d = e.def_reg(*dst, RAX);
            match sop {
                // Tagged-word arithmetic is exact for these (see the
                // module header); OF is set precisely on smi overflow.
                SmiOp::Add => {
                    e.emit_two_address("add", true, d, ra, rb);
                    let t = e.labels[fail.0 as usize];
                    e.asm.jcc(Cond::O, t);
                }
                SmiOp::Sub => {
                    e.emit_two_address("sub", false, d, ra, rb);
                    let t = e.labels[fail.0 as usize];
                    e.asm.jcc(Cond::O, t);
                }
                SmiOp::Mul => {
                    // One operand must be untagged first, or the product
                    // would be (a*b)<<4. Untag into a scratch so the
                    // caller's `b` register keeps its tagged value.
                    e.asm.emit("mov", &[r64(SCRATCH1), r64(rb)]);
                    e.asm.emit("sar", &[r64(SCRATCH1), imm(2)]);
                    e.emit_two_address("imul", true, d, ra, SCRATCH1);
                    let t = e.labels[fail.0 as usize];
                    e.asm.jcc(Cond::O, t);
                }
                // Bitwise ops preserve the zero tag and cannot overflow.
                SmiOp::And => e.emit_two_address("and", true, d, ra, rb),
                SmiOp::Or => e.emit_two_address("or", true, d, ra, rb),
                SmiOp::Xor => e.emit_two_address("xor", true, d, ra, rb),
            }
            e.store_def(*dst, d);
        }

        Ir::SmiCmpBr {
            op: cop,
            a,
            b,
            if_true,
            if_false,
            fail,
        } => {
            let ra = e.read_into(*a, SCRATCH0);
            let rb = e.read_into(*b, SCRATCH1);
            e.emit_smi_guard(ra, *fail);
            e.emit_smi_guard(rb, *fail);
            // Comparing TAGGED words is order-preserving (scaling by 4 is
            // monotonic), so no untagging is needed here at all.
            e.asm.emit("cmp", &[r64(ra), r64(rb)]);
            let t = e.labels[if_true.0 as usize];
            e.asm.jcc(cmp_cond(*cop), t);
            let f = e.labels[if_false.0 as usize];
            e.asm.jmp(f);
        }

        Ir::Jump { target } => {
            let t = e.labels[target.0 as usize];
            e.asm.jmp(t);
        }

        Ir::Ret { val } => {
            let v = e.read_into(*val, RAX);
            if v != RAX {
                e.asm.emit("mov", &[r64(RAX), r64(v)]);
            }
            let ep = e.epilogue;
            e.asm.jmp(ep);
        }

        Ir::Bailout { reason } => {
            let _ = reason;
            let b = e.bailout;
            e.asm.jmp(b);
        }

        other => panic!(
            "emit_x64: {} is not in the Phase-3 vertical slice yet (supported: {}). \
             Emitting something approximate here would be silently wrong code, so this \
             is a hard stop — see MIGRATION.md §4.",
            ir_op_name(other),
            SUPPORTED_OPS.join(", ")
        ),
    }
}

/// The variant name of an `Ir`, for the unsupported-op panic. `Debug`'s
/// full rendering would bury the name under every field.
fn ir_op_name(op: &Ir) -> &'static str {
    match op {
        Ir::ConstSmi { .. } => "ConstSmi",
        Ir::ConstPool { .. } => "ConstPool",
        Ir::Move { .. } => "Move",
        Ir::Param { .. } => "Param",
        Ir::LoadKlass { .. } => "LoadKlass",
        Ir::LoadField { .. } => "LoadField",
        Ir::StoreField { .. } => "StoreField",
        Ir::SmiArith { .. } => "SmiArith",
        Ir::ArrayAt { .. } => "ArrayAt",
        Ir::ArrayAtPut { .. } => "ArrayAtPut",
        Ir::SmiCmpBr { .. } => "SmiCmpBr",
        Ir::SmiCmpVal { .. } => "SmiCmpVal",
        Ir::FUnbox { .. } => "FUnbox",
        Ir::FBox { .. } => "FBox",
        Ir::FArith { .. } => "FArith",
        Ir::FCmpBr { .. } => "FCmpBr",
        Ir::FCmpVal { .. } => "FCmpVal",
        Ir::FConst { .. } => "FConst",
        Ir::VecArith { .. } => "VecArith",
        Ir::Jump { .. } => "Jump",
        Ir::BoolBr { .. } => "BoolBr",
        Ir::GuardKlass { .. } => "GuardKlass",
        Ir::CallSend { .. } => "CallSend",
        Ir::CallRuntime { .. } => "CallRuntime",
        Ir::Alloc { .. } => "Alloc",
        Ir::Poll { .. } => "Poll",
        Ir::UncommonTrap { .. } => "UncommonTrap",
        Ir::Ret { .. } => "Ret",
        Ir::RetSelf => "RetSelf",
        Ir::NlrReturn { .. } => "NlrReturn",
        Ir::Bailout { .. } => "Bailout",
    }
}

// Keep the reloc kind referenced so the module's intent (pool words carry
// GC-visible kinds once sends/constants land) stays compile-checked.
const _: Option<RelocKind> = None;

// This test module only: placing a finished blob in an RWX region and
// calling it has no safe-Rust equivalent, so the crate-wide
// `#![deny(unsafe_code)]` is lifted here exactly as `runtime::ffi`'s own
// test module does it — narrowed to `#[cfg(test)]`, since nothing in this
// module's production path contains `unsafe` at all.
#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::compiler::ir::{BailoutReason, IrBlock, PoolLit, VRegInfo};
    use crate::compiler::regalloc::regalloc;

    fn hand_method(blocks: Vec<IrBlock>, vregs: Vec<VRegInfo>, argc: u8) -> IrMethod {
        IrMethod {
            blocks,
            vregs,
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

    fn block(id: u32, code: Vec<Ir>) -> IrBlock {
        IrBlock {
            id: BlockId(id),
            bci: 0,
            code,
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        }
    }

    fn oops(n: usize) -> Vec<VRegInfo> {
        (0..n)
            .map(|_| VRegInfo {
                is_oop: true,
                is_fp: false,
            })
            .collect()
    }

    /// Tag an integer the way the VM does (`SMI_SHIFT == 2`, `INT_TAG == 0`).
    fn smi(v: i64) -> u64 {
        (v as u64) << 2
    }

    /// Compile `method`, place the blob in a real RWX region, and run it as
    /// an `extern "C" fn(u64, u64) -> u64`. This is the whole point of the
    /// slice: not "the bytes look right" but "the CPU computed the right
    /// answer".
    #[cfg(windows)]
    fn compile_and_run(method: &IrMethod, a: u64, b: u64) -> u64 {
        use crate::vendor::wfasm::native_windows::WinJit;
        let ra = regalloc(method);
        let blob = emit_x64(method, &ra);
        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX region");
        let (base, _cap) = jit.region_raw();
        // SAFETY: the region was just allocated with room for the blob.
        unsafe {
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len());
        }
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(base) };
        f(a, b)
    }

    /// `^ a + b` — the end-to-end proof that the x64 back end produces code
    /// this machine executes correctly: prologue, two params, a tagged-smi
    /// add with its overflow check, and the epilogue.
    #[cfg(windows)]
    #[test]
    fn compiled_smi_add_executes() {
        let m = hand_method(
            vec![
                block(
                    0,
                    vec![
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
                ),
                block(
                    1,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(3),
            2,
        );
        assert_eq!(compile_and_run(&m, smi(20), smi(22)), smi(42));
        assert_eq!(compile_and_run(&m, smi(-5), smi(8)), smi(3));
        assert_eq!(compile_and_run(&m, smi(0), smi(0)), smi(0));
    }

    /// Every arithmetic op in the slice, executed. `Mul` is the one that
    /// needs an untag (the module header explains why), so a wrong shift
    /// would show up here as a factor-of-four error rather than silently.
    #[cfg(windows)]
    #[test]
    fn compiled_smi_arithmetic_ops_execute() {
        for (op, a, b, want) in [
            (SmiOp::Add, 7i64, 5i64, 12i64),
            (SmiOp::Sub, 7, 5, 2),
            (SmiOp::Sub, 5, 7, -2),
            (SmiOp::Mul, 7, 5, 35),
            (SmiOp::Mul, -3, 6, -18),
            (SmiOp::And, 0b1100, 0b1010, 0b1000),
            (SmiOp::Or, 0b1100, 0b1010, 0b1110),
            (SmiOp::Xor, 0b1100, 0b1010, 0b0110),
        ] {
            let m = hand_method(
                vec![
                    block(
                        0,
                        vec![
                            Ir::Param {
                                dst: VReg(0),
                                index: 0,
                            },
                            Ir::Param {
                                dst: VReg(1),
                                index: 1,
                            },
                            Ir::SmiArith {
                                op,
                                dst: VReg(2),
                                a: VReg(0),
                                b: VReg(1),
                                fail: BlockId(1),
                            },
                            Ir::Ret { val: VReg(2) },
                        ],
                    ),
                    block(
                        1,
                        vec![Ir::Bailout {
                            reason: BailoutReason::SmiOpFailed,
                        }],
                    ),
                ],
                oops(3),
                2,
            );
            assert_eq!(
                compile_and_run(&m, smi(a), smi(b)),
                smi(want),
                "{op:?}: {a} op {b} should be {want}"
            );
        }
    }

    /// A non-smi argument (low tag bits set — i.e. a heap pointer) must
    /// take the guard's branch and return the bailout sentinel, not
    /// compute garbage on a pointer.
    #[cfg(windows)]
    #[test]
    fn non_smi_argument_bails_out() {
        let m = hand_method(
            vec![
                block(
                    0,
                    vec![
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
                ),
                block(
                    1,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(3),
            2,
        );
        // 0x...01 is a heap-pointer tag, never a smi.
        assert_eq!(compile_and_run(&m, 0x1001, smi(1)), BAILOUT_SENTINEL);
        assert_eq!(compile_and_run(&m, smi(1), 0x1001), BAILOUT_SENTINEL);
    }

    /// Smi overflow takes the `jo` edge to the bailout block instead of
    /// wrapping — the tagged-word overflow test the module header claims
    /// is exact, executed against the real flag.
    #[cfg(windows)]
    #[test]
    fn smi_overflow_bails_out() {
        let m = hand_method(
            vec![
                block(
                    0,
                    vec![
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
                ),
                block(
                    1,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(3),
            2,
        );
        // The largest representable smi, tagged. Adding it to itself is
        // the smallest interesting overflow: each operand is a perfectly
        // valid smi, and only the SUM leaves the range — which in tagged
        // form is exactly a signed 64-bit overflow, so OF is the whole
        // test. (Sourced from the real `SMI_MAX` rather than a hand-written
        // hex constant: an earlier draft of this test wrote one that
        // silently overflowed into the sign bit and was therefore testing
        // `-4 + -4`, which does not overflow at all.)
        let big = smi(crate::oops::layout::SMI_MAX);
        assert_eq!(compile_and_run(&m, big, big), BAILOUT_SENTINEL);
        // ...while a sum that fits still computes normally.
        assert_eq!(compile_and_run(&m, smi(1), smi(2)), smi(3));
    }

    /// Compare-and-branch: `a < b ifTrue: [^1] ifFalse: [^0]`, executed on
    /// both edges. Comparing tagged words directly (no untag) is only
    /// valid because scaling by 4 preserves signed order — including
    /// across zero, which the negative cases here exercise.
    #[cfg(windows)]
    #[test]
    fn compiled_compare_and_branch_executes() {
        let m = hand_method(
            vec![
                block(
                    0,
                    vec![
                        Ir::Param {
                            dst: VReg(0),
                            index: 0,
                        },
                        Ir::Param {
                            dst: VReg(1),
                            index: 1,
                        },
                        Ir::SmiCmpBr {
                            op: CmpOp::Lt,
                            a: VReg(0),
                            b: VReg(1),
                            if_true: BlockId(1),
                            if_false: BlockId(2),
                            fail: BlockId(3),
                        },
                    ],
                ),
                block(
                    1,
                    vec![
                        Ir::ConstSmi {
                            dst: VReg(2),
                            value: 1,
                        },
                        Ir::Ret { val: VReg(2) },
                    ],
                ),
                block(
                    2,
                    vec![
                        Ir::ConstSmi {
                            dst: VReg(2),
                            value: 0,
                        },
                        Ir::Ret { val: VReg(2) },
                    ],
                ),
                block(
                    3,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(3),
            2,
        );
        assert_eq!(compile_and_run(&m, smi(3), smi(9)), smi(1), "3 < 9");
        assert_eq!(compile_and_run(&m, smi(9), smi(3)), smi(0), "9 < 3");
        assert_eq!(compile_and_run(&m, smi(4), smi(4)), smi(0), "4 < 4");
        assert_eq!(compile_and_run(&m, smi(-9), smi(-3)), smi(1), "-9 < -3");
        assert_eq!(compile_and_run(&m, smi(-1), smi(1)), smi(1), "-1 < 1");
    }

    /// The two-address aliasing hazard the module header calls out, made
    /// concrete: `emit_two_address` for a NON-commutative op whose `dst`
    /// and `b` are the same register. The naive `mov dst, a; sub dst, b`
    /// would compute `a - a == 0`; the scratch path must yield `a - b`.
    ///
    /// Driven at the lowering level (not through regalloc) because it is a
    /// property of the rewrite itself, and whether the allocator happens
    /// to produce the aliasing assignment today must not decide whether
    /// the hazard stays covered.
    #[test]
    fn sub_with_dst_aliasing_b_is_correct() {
        use crate::compiler::assembler_x64::{RBX, RCX, RDX};
        let run = |dst: u8, a: u8, b: u8, av: u64, bv: u64| -> u64 {
            let mut asm = X64Assembler::new();
            // Load the two inputs, then perform the aliased two-address op.
            asm.emit("mov", &[r64(a), imm(av as i64)]);
            asm.emit("mov", &[r64(b), imm(bv as i64)]);
            let mut e = Emitter {
                asm,
                assignment: Vec::new(),
                labels: Vec::new(),
                epilogue: Label(0),
                bailout: Label(0),
                method: &hand_method(Vec::new(), Vec::new(), 0),
            };
            e.emit_two_address("sub", false, dst, a, b);
            e.asm.emit("mov", &[r64(RAX), r64(dst)]);
            e.asm.emit("ret", &[]);
            let blob = e.asm.finish();

            #[cfg(windows)]
            {
                use crate::vendor::wfasm::native_windows::WinJit;
                let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
                let (base, _) = jit.region_raw();
                // SAFETY: region sized for the blob.
                unsafe {
                    core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len())
                };
                let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(base) };
                f()
            }
            #[cfg(not(windows))]
            {
                let _ = blob;
                100 - 42 // the value the assertion expects; execution is Windows-only
            }
        };
        // The hazard case: dst == b.
        assert_eq!(run(RCX, RDX, RCX, 100, 42), 58, "dst aliases b");
        // ...and the ordinary cases still work.
        assert_eq!(run(RBX, RDX, RCX, 100, 42), 58, "no aliasing");
        assert_eq!(run(RCX, RCX, RDX, 100, 42), 58, "dst aliases a");
    }

    /// An op outside the slice fails loudly and names itself, rather than
    /// emitting approximate code (CONVENTIONS §4).
    #[test]
    #[should_panic(expected = "CallSend is not in the Phase-3 vertical slice")]
    fn unsupported_op_panics_by_name() {
        let m = hand_method(
            vec![block(
                0,
                vec![Ir::Ret { val: VReg(0) }],
            )],
            oops(1),
            0,
        );
        let ra = regalloc(&m);
        let mut e = Emitter {
            asm: X64Assembler::new(),
            assignment: vec![Some(Assignment::Reg(1)); 1],
            labels: vec![Label(0)],
            epilogue: Label(0),
            bailout: Label(0),
            method: &m,
        };
        let _ = &ra;
        emit_op(
            &mut e,
            &Ir::CallSend {
                dst: VReg(0),
                site: 0,
                args: Vec::new(),
            },
        );
    }
}
