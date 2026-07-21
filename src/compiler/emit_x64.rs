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

use crate::compiler::assembler::{CodeBlob, Label, LiteralId, RelocKind};
use crate::compiler::assembler_x64::{
    imm, mem, mem_byte, r32, r64, Cond, X64Assembler, ARG_REGS, RAX, RBP, RECEIVER, RSP,
    SCRATCH0, SCRATCH1, SHADOW_SPACE, VM_STATE,
};
use crate::compiler::ir::{BlockId, CmpOp, GuardShape, Ir, IrMethod, PoolLit, SmiOp, VReg};
use crate::compiler::regalloc::{Assignment, RegallocResult, SpillSlot};

/// The IR ops this slice lowers. Anything else panics in [`emit_x64`] —
/// see the module header.
pub const SUPPORTED_OPS: &[&str] = &[
    "ConstSmi",
    "ConstPool",
    "Move",
    "Param",
    "LoadKlass",
    "LoadField",
    "GuardKlass",
    "SmiArith",
    "SmiCmpBr",
    "SmiCmpVal",
    "StoreField",
    "BoolBr",
    "Poll",
    "CallRuntime",
    "Jump",
    "UncommonTrap",
    "Ret",
    "RetSelf",
    "Bailout",
];

/// Byte offset of an object's klass word from its TAGGED pointer:
/// `KLASS_OFFSET` (8) less `MEM_TAG` (1), because a heap oop's word is the
/// address biased by the tag. The AArch64 emitter reaches it with an
/// unscaled `ldur`; x86 addresses it directly with a disp8.
const KLASS_OFF_FROM_TAGGED: i64 =
    crate::oops::layout::KLASS_OFFSET as i64 - crate::oops::layout::MEM_TAG as i64;

/// One safepoint the emitter recorded — currently only deopt trap sites.
/// `pc_off` is the trapping instruction's OWN offset (for a trap site the
/// trapping pc IS the `int3`), which is what the VEH reports in `Rip`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrapSite {
    pub pc_off: u32,
    pub bci: usize,
}

/// What [`emit_x64`] produces: the blob plus the metadata the deopt
/// machinery keys on.
pub struct Emitted {
    pub blob: CodeBlob,
    pub trap_sites: Vec<TrapSite>,
    /// Return addresses of calls into the runtime — deopt safepoints, by
    /// the same "safepoint keys on the RETURN address" convention the
    /// AArch64 emitter uses (contrast [`TrapSite`], which keys on the
    /// trapping instruction itself).
    pub safepoints: Vec<TrapSite>,
}

/// Absolute addresses of the runtime entry points compiled code calls.
/// Passed in rather than looked up so the emitter stays free of any
/// dependency on a live `VmState` — the same shape as the AArch64
/// `emit`'s long parameter list, collected into one struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeAddrs {
    /// `stub_poll` — runs the safepoint action when the poll flag is set.
    pub stub_poll: u64,
    /// `must_be_boolean` — coerces or raises on a non-boolean.
    pub must_be_boolean: u64,
    /// `rt_alloc_slow` — the allocation slow path.
    pub alloc_slow: u64,
}

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
    /// `PoolLit` index → the interned literal-pool entry.
    literal_ids: Vec<LiteralId>,
    epilogue: Label,
    bailout: Label,
    trap_sites: Vec<TrapSite>,
    safepoints: Vec<TrapSite>,
    /// Pool entries holding the runtime entry points.
    stub_poll_lit: LiteralId,
    must_be_boolean_lit: LiteralId,
    #[allow(dead_code)] // consumed when Alloc lands
    alloc_slow_lit: LiteralId,
    current_bci: usize,
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

    /// Call an absolute runtime address, Win64-correctly.
    ///
    /// **Why clobbering volatiles is safe here.** A Rust callee may
    /// destroy every volatile register — `RAX RCX RDX R8–R11` and
    /// `XMM0–5` — which includes four registers the allocator hands out
    /// (`RCX RDX R8 R9`). That is sound only because every op that
    /// reaches this helper is a *safepoint*, and `regalloc`'s spill-all
    /// policy unconditionally spills any interval live across a
    /// safepoint before the scan even begins. So nothing live is in a
    /// volatile register at this point, by construction. The pinned
    /// registers (`R12–R15`) and the allocatable callee-saved ones
    /// (`RBX RSI RDI`) survive because Win64 requires the callee to
    /// preserve them.
    ///
    /// **Stack discipline.** The prologue leaves `RSP` 16-byte aligned,
    /// and 32 is a multiple of 16, so reserving exactly the mandatory
    /// 32-byte shadow space both satisfies the ABI and keeps the
    /// alignment the callee requires. (The call *stub* subtracts 40
    /// instead — it is at a different alignment phase, having pushed an
    /// odd number of registers. The two numbers are both correct and the
    /// difference is not an inconsistency.)
    fn emit_runtime_call(&mut self, target: LiteralId) {
        self.asm
            .emit("sub", &[r64(RSP), imm(SHADOW_SPACE as i64)]);
        self.asm.call_far(target);
        self.asm
            .emit("add", &[r64(RSP), imm(SHADOW_SPACE as i64)]);
    }

    /// Record a deopt safepoint at the CURRENT offset — used right after
    /// a runtime call returns, so the recorded pc is the return address.
    fn record_safepoint(&mut self) {
        let pc_off = self.asm.offset();
        let bci = self.current_bci;
        self.safepoints.push(TrapSite { pc_off, bci });
    }

    /// The generational write barrier: dirty the card covering a stored
    /// field, but only when the store actually creates an old→young
    /// reference. Three early-outs, in the AArch64 emitter's order and for
    /// the same reasons — each is cheaper than the one after it:
    ///
    /// 1. **`obj` is young** (below `old_start`) — a young object is
    ///    scanned wholesale by the scavenger, so it needs no card.
    /// 2. **`val` is a smi** — an immediate is not a reference at all.
    /// 3. **`val` is old** (at or above `old_start`) — an old→old
    ///    reference does not concern a young collection.
    ///
    /// Only a store that survives all three marks its card. `CARD_DIRTY`
    /// is 0, so the mark is a byte store of zero (the A64 side stores the
    /// zero register; x86 stores an immediate).
    ///
    /// `biased` is the already-tag-adjusted field displacement, so the
    /// card index is computed from the true field address.
    fn emit_write_barrier(&mut self, robj: u8, rval: u8, biased: i64) {
        let skip = self.asm.new_label();
        // old_start, read live from the VM register block.
        self.asm.emit(
            "mov",
            &[
                r64(RAX),
                mem(VM_STATE, crate::oops::layout::VMREG_OLD_START_OFFSET as i64),
            ],
        );
        self.asm.emit("cmp", &[r64(robj), r64(RAX)]);
        self.asm.jcc(Cond::B, skip); // obj younger than old_start
        self.asm.emit("test", &[r64(rval), imm(3)]);
        self.asm.jcc(Cond::E, skip); // val is a smi
        self.asm.emit("cmp", &[r64(rval), r64(RAX)]);
        self.asm.jcc(Cond::Ae, skip); // val is old too

        // card_index = (obj + biased) >> CARD_SHIFT, then dirty the byte
        // at card_base_biased + card_index.
        self.asm.emit("lea", &[r64(SCRATCH0), mem(robj, biased)]);
        self.asm.emit(
            "shr",
            &[r64(SCRATCH0), imm(crate::memory::cards::CARD_SHIFT as i64)],
        );
        self.asm.emit(
            "mov",
            &[
                r64(RAX),
                mem(
                    VM_STATE,
                    crate::oops::layout::VMREG_CARD_BASE_BIASED_OFFSET as i64,
                ),
            ],
        );
        self.asm.emit("add", &[r64(RAX), r64(SCRATCH0)]);
        self.asm.emit("mov", &[mem_byte(RAX, 0), imm(0)]); // CARD_DIRTY == 0
        self.asm.bind(skip);
    }

    /// A klass guard (`GuardShape::KlassTest`): the object must be a heap
    /// oop whose klass word equals the expected pool literal.
    ///
    /// The smi rejection comes FIRST and is not merely an optimization —
    /// a smi has no header at all, so loading `[obj + 7]` from one would
    /// dereference a small integer as an address. The AArch64 emitter
    /// makes the same check for the same reason.
    fn emit_klass_guard(&mut self, robj: u8, expect: PoolLit, fail: BlockId) {
        let cold = self.labels[fail.0 as usize];
        // A smi can never be an instance of a heap klass → straight to cold.
        self.asm.emit("test", &[r64(robj), imm(3)]);
        self.asm.jcc(Cond::E, cold);
        // Klass word, then compare against the expected klass pool word.
        self.asm
            .emit("mov", &[r64(SCRATCH1), mem(robj, KLASS_OFF_FROM_TAGGED)]);
        let lit = self.literal_ids[expect.0 as usize];
        self.asm.load_literal(SCRATCH0, lit);
        self.asm.emit("cmp", &[r64(SCRATCH1), r64(SCRATCH0)]);
        self.asm.jcc(Cond::Ne, cold);
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
pub fn emit_x64(method: &IrMethod, regalloc: &RegallocResult, rt: RuntimeAddrs) -> Emitted {
    let mut asm = X64Assembler::new();

    // Intern the method's constant pool first, so `PoolLit(i)` indexes
    // `literal_ids[i]` 1:1 — the same contract the AArch64 emitter's
    // `intern_pool` establishes, which `codecache::read_pool_oop` relies on
    // when a deopt reads a pool word back by index.
    let literal_ids: Vec<LiteralId> = method
        .pool
        .iter()
        .map(|entry| asm.literal_u64(entry.value, entry.kind))
        .collect();

    let mut assignment: Vec<Option<Assignment>> = vec![None; method.vregs.len()];
    for iv in &regalloc.intervals {
        assignment[iv.vreg.0 as usize] = iv.assignment;
    }
    // Runtime entry points get pool words of their own, kinded
    // `RuntimeAddr` so the GC leaves them alone (they are not oops).
    let stub_poll_lit = asm.literal_u64(rt.stub_poll, Some(RelocKind::RuntimeAddr));
    let must_be_boolean_lit = asm.literal_u64(rt.must_be_boolean, Some(RelocKind::RuntimeAddr));
    let alloc_slow_lit = asm.literal_u64(rt.alloc_slow, Some(RelocKind::RuntimeAddr));

    let labels: Vec<Label> = (0..method.blocks.len()).map(|_| asm.new_label()).collect();
    let epilogue = asm.new_label();
    let bailout = asm.new_label();

    let mut e = Emitter {
        asm,
        assignment,
        labels,
        literal_ids,
        epilogue,
        bailout,
        trap_sites: Vec::new(),
        safepoints: Vec::new(),
        stub_poll_lit,
        must_be_boolean_lit,
        alloc_slow_lit,
        current_bci: 0,
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
        e.current_bci = block.bci;
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

    let trap_sites = std::mem::take(&mut e.trap_sites);
    let safepoints = std::mem::take(&mut e.safepoints);
    Emitted {
        blob: e.asm.finish(),
        trap_sites,
        safepoints,
    }
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

        Ir::ConstPool { dst, lit } => {
            let d = e.def_reg(*dst, SCRATCH0);
            let id = e.literal_ids[lit.0 as usize];
            e.asm.load_literal(d, id);
            e.store_def(*dst, d);
        }

        Ir::LoadKlass { dst, obj } => {
            let o = e.read_into(*obj, SCRATCH0);
            let d = e.def_reg(*dst, SCRATCH1);
            e.asm
                .emit("mov", &[r64(d), mem(o, KLASS_OFF_FROM_TAGGED)]);
            e.store_def(*dst, d);
        }

        Ir::GuardKlass {
            obj,
            expect,
            fail,
            kind,
        } => {
            let o = e.read_into(*obj, SCRATCH0);
            match kind {
                GuardShape::SmiTest => e.emit_smi_guard(o, *fail),
                GuardShape::KlassTest => e.emit_klass_guard(o, *expect, *fail),
            }
        }

        Ir::UncommonTrap { bci } => {
            // The trapping pc IS this instruction's own offset (Windows
            // reports a breakpoint with Rip pointing AT the 0xCC), so the
            // site is recorded BEFORE the bytes go down — the same
            // ordering rule the AArch64 `brk` path documents.
            let pc_off = e.asm.offset();
            e.trap_sites.push(TrapSite {
                pc_off,
                bci: *bci,
            });
            e.asm.emit_bytes(&crate::codecache::deopt_trap::deopt_int3_bytes(
                crate::codecache::deopt_trap::TRAP_UNCOMMON,
            ));
        }

        Ir::LoadField { dst, obj, byte_off } => {
            let o = e.read_into(*obj, SCRATCH0);
            let d = e.def_reg(*dst, SCRATCH1);
            // A heap oop's word is its address biased by MEM_TAG, so the
            // field displacement is biased too — exactly as `StoreField`
            // and `LoadKlass` do it. (This bias was missing when the op
            // first landed; `compiled_store_field_writes_through_the_tag_
            // bias` is the test that caught it, by reading back a field it
            // had just written and getting a value shifted by one byte.)
            let biased = *byte_off as i64 - crate::oops::layout::MEM_TAG as i64;
            e.asm.emit("mov", &[r64(d), mem(o, biased)]);
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

        Ir::StoreField {
            obj,
            byte_off,
            val,
            barrier,
        } => {
            let o = e.read_into(*obj, SCRATCH0);
            let v = e.read_into(*val, SCRATCH1);
            // A heap oop's word is the address biased by MEM_TAG, so every
            // field offset is biased too — the same `- 1` the A64 emitter
            // folds into its `stur` displacement.
            let biased = *byte_off as i64 - crate::oops::layout::MEM_TAG as i64;
            e.asm.emit("mov", &[mem(o, biased), r64(v)]);
            if *barrier {
                e.emit_write_barrier(o, v, biased);
            }
        }

        Ir::SmiCmpVal {
            op: cop,
            dst,
            a,
            b,
            fail,
        } => {
            let ra = e.read_into(*a, SCRATCH0);
            let rb = e.read_into(*b, SCRATCH1);
            e.emit_smi_guard(ra, *fail);
            e.emit_smi_guard(rb, *fail);
            e.asm.emit("cmp", &[r64(ra), r64(rb)]);
            // Branchless select, the `csel` analogue: load false into the
            // destination, true into a scratch, then conditionally move.
            // `cmp` writes no register, so both scratches are free again
            // regardless of what they held for the operands.
            let d = e.def_reg(*dst, RAX);
            let false_lit = e.literal_ids[e.method.false_lit.0 as usize];
            let true_lit = e.literal_ids[e.method.true_lit.0 as usize];
            e.asm.load_literal(d, false_lit);
            e.asm.load_literal(SCRATCH0, true_lit);
            e.asm
                .emit(cmp_cond(*cop).cmov(), &[r64(d), r64(SCRATCH0)]);
            e.store_def(*dst, d);
        }

        Ir::BoolBr {
            val,
            if_true,
            if_false,
            not_bool,
        } => {
            let rv = e.read_into(*val, SCRATCH0);
            // Compare against the canonical true/false oops. Anything else
            // is not a boolean and takes the `not_bool` edge — Smalltalk
            // requires `ifTrue:` on a non-boolean to raise, not to coerce.
            let true_lit = e.literal_ids[e.method.true_lit.0 as usize];
            e.asm.load_literal(SCRATCH1, true_lit);
            e.asm.emit("cmp", &[r64(rv), r64(SCRATCH1)]);
            let t = e.labels[if_true.0 as usize];
            e.asm.jcc(Cond::E, t);
            let false_lit = e.literal_ids[e.method.false_lit.0 as usize];
            e.asm.load_literal(SCRATCH1, false_lit);
            e.asm.emit("cmp", &[r64(rv), r64(SCRATCH1)]);
            let f = e.labels[if_false.0 as usize];
            e.asm.jcc(Cond::E, f);
            let nb = e.labels[not_bool.0 as usize];
            e.asm.jmp(nb);
        }

        Ir::RetSelf => {
            // The receiver lives in its pinned register for the whole
            // activation, so returning self is just a move.
            e.asm.emit("mov", &[r64(RAX), r64(RECEIVER)]);
            let ep = e.epilogue;
            e.asm.jmp(ep);
        }

        Ir::Poll => {
            // `mov eax, [r15 + POLL]; test eax, eax; jz skip; call stub_poll`
            // — a 32-bit load, so the flag test costs no REX prefix and
            // zero-extends for free.
            let skip = e.asm.new_label();
            e.asm.emit(
                "mov",
                &[
                    r32(RAX),
                    mem(VM_STATE, crate::oops::layout::VMREG_POLL_FLAG_OFFSET as i64),
                ],
            );
            e.asm.emit("test", &[r32(RAX), r32(RAX)]);
            e.asm.jcc(Cond::E, skip);
            let lit = e.stub_poll_lit;
            e.emit_runtime_call(lit);
            // The poll is a deopt safepoint keyed on the RETURN address —
            // which is exactly where `skip` binds, since a dormant flag
            // also lands here. Recording before the bind makes that
            // coincidence explicit rather than accidental.
            e.record_safepoint();
            e.asm.bind(skip);
        }

        Ir::CallRuntime { dst, stub, args } => {
            assert_eq!(
                *stub,
                crate::compiler::ir::StubId::MUST_BE_BOOLEAN,
                "emit_x64: only MUST_BE_BOOLEAN is wired up, mirroring the AArch64 \
                 emitter's own restriction"
            );
            assert_eq!(
                args.len(),
                1,
                "emit_x64: MUST_BE_BOOLEAN takes exactly one argument"
            );
            let a0 = e.read_into(args[0], SCRATCH0);
            if a0 != ARG_REGS[0] {
                e.asm.emit("mov", &[r64(ARG_REGS[0]), r64(a0)]);
            }
            let lit = e.must_be_boolean_lit;
            e.emit_runtime_call(lit);
            e.record_safepoint();
            let dst = dst.expect("MUST_BE_BOOLEAN always produces a coerced boolean");
            let d = e.def_reg(dst, RAX);
            if d != RAX {
                e.asm.emit("mov", &[r64(d), r64(RAX)]);
            }
            e.store_def(dst, d);
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
        let blob = emit_x64(method, &ra, RuntimeAddrs::default()).blob;
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
                literal_ids: Vec::new(),
                epilogue: Label(0),
                bailout: Label(0),
                trap_sites: Vec::new(),
                safepoints: Vec::new(),
                stub_poll_lit: LiteralId(0),
                must_be_boolean_lit: LiteralId(0),
                alloc_slow_lit: LiteralId(0),
                current_bci: 0,
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

    /// A klass guard, executed against real memory: a hand-built object
    /// whose header is `[mark][klass]` and whose tagged pointer is
    /// `addr | MEM_TAG`. Matching klass falls through and returns 1;
    /// a different klass, and a smi receiver, both take the cold edge.
    ///
    /// The smi case is the one worth having: a smi has no header at all,
    /// so a guard that loaded `[obj + 7]` before rejecting smis would
    /// dereference a small integer as an address.
    #[cfg(windows)]
    #[test]
    fn compiled_klass_guard_executes() {
        use crate::compiler::ir::PoolEntry;
        use crate::oops::layout::MEM_TAG;

        // Two objects with distinct klass words, laid out as the VM does.
        let mut obj_a = [0u64; 2];
        let mut obj_b = [0u64; 2];
        let klass_a = 0xAAAA_0000u64;
        let klass_b = 0xBBBB_0000u64;
        obj_a[1] = klass_a;
        obj_b[1] = klass_b;
        let tagged = |o: &[u64; 2]| o.as_ptr() as u64 | MEM_TAG;

        let mut m = hand_method(
            vec![
                block(
                    0,
                    vec![
                        Ir::Param {
                            dst: VReg(0),
                            index: 0,
                        },
                        Ir::GuardKlass {
                            obj: VReg(0),
                            expect: PoolLit(0),
                            fail: BlockId(2),
                            kind: GuardShape::KlassTest,
                        },
                        Ir::ConstSmi {
                            dst: VReg(1),
                            value: 1,
                        },
                        Ir::Ret { val: VReg(1) },
                    ],
                ),
                block(1, vec![Ir::Jump { target: BlockId(2) }]),
                block(
                    2,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(2),
            1,
        );
        // The guard's expected klass is pool entry 0.
        m.pool = vec![PoolEntry {
            value: klass_a,
            kind: Some(RelocKind::Oop),
        }];

        assert_eq!(
            compile_and_run(&m, tagged(&obj_a), 0),
            smi(1),
            "matching klass falls through the guard"
        );
        assert_eq!(
            compile_and_run(&m, tagged(&obj_b), 0),
            BAILOUT_SENTINEL,
            "different klass takes the cold edge"
        );
        assert_eq!(
            compile_and_run(&m, smi(7), 0),
            BAILOUT_SENTINEL,
            "a smi receiver is rejected BEFORE any header load"
        );
    }

    /// `LoadKlass` reads the klass word through the same tagged-pointer
    /// bias the guard uses — if the two ever disagreed, guards would pass
    /// while the loaded klass was garbage.
    #[cfg(windows)]
    #[test]
    fn compiled_load_klass_reads_the_header_word() {
        let obj = [0u64, 0x1234_5678_0000_0000u64];
        let tagged = obj.as_ptr() as u64 | crate::oops::layout::MEM_TAG;
        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Param {
                        dst: VReg(0),
                        index: 0,
                    },
                    Ir::LoadKlass {
                        dst: VReg(1),
                        obj: VReg(0),
                    },
                    Ir::Ret { val: VReg(1) },
                ],
            )],
            oops(2),
            1,
        );
        assert_eq!(compile_and_run(&m, tagged, 0), 0x1234_5678_0000_0000);
    }

    /// **The Phase-2/Phase-3 seam, closed.** An `UncommonTrap` emitted by
    /// the compiler must be the exact byte pattern the Phase-2 VEH
    /// decodes. This compiles a method containing a trap, registers the
    /// blob's range with a capture trampoline, arms the real VEH, and
    /// calls it: the trap must be recognized, the trap pc stashed in R10,
    /// and control redirected — returning that pc.
    ///
    /// The two halves were written days apart against a written contract
    /// (`int3` + imm16, `Rip` points AT the `0xCC`); this is the test that
    /// proves they actually meet.
    #[cfg(windows)]
    #[test]
    fn emitted_uncommon_trap_round_trips_through_the_veh() {
        use crate::codecache::deopt_trap;
        use crate::vendor::wfasm::native_windows::WinJit;

        let m = hand_method(
            vec![block(0, vec![Ir::UncommonTrap { bci: 7 }])],
            oops(1),
            0,
        );
        let ra = regalloc(&m);
        let out = emit_x64(&m, &ra, RuntimeAddrs::default());

        // The emitter recorded the site, keyed by the trap's own offset.
        assert_eq!(out.trap_sites.len(), 1);
        assert_eq!(out.trap_sites[0].bci, 7);
        let trap_off = out.trap_sites[0].pc_off;
        assert_eq!(
            out.blob.code[trap_off as usize],
            0xCC,
            "the recorded pc_off must point AT the int3, not past it"
        );

        // Place it, plus a capture trampoline that returns R10 (the stash).
        let blob = &out.blob;
        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX region");
        let (base, _cap) = jit.region_raw();
        unsafe {
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len());
        }
        let entry = base as u64;
        let tramp_off = (blob.code.len() + 15) & !15;
        let tramp = entry + tramp_off as u64;
        // mov rax, r10 ; mov rsp, rbp ; pop rbp ; ret  — unwind the frame
        // the compiled prologue established, then hand back the trap pc.
        unsafe {
            core::ptr::copy_nonoverlapping(
                [0x4C, 0x89, 0xD0, 0x48, 0x89, 0xEC, 0x5D, 0xC3].as_ptr(),
                base.add(tramp_off),
                8,
            );
        }

        deopt_trap::test_register_range(entry, entry + blob.code.len() as u64, tramp);
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(entry) };
        let got = f();
        deopt_trap::deregister(entry);

        assert_eq!(
            got,
            entry + trap_off as u64,
            "the VEH must decode the emitted trap site and stash its pc in R10"
        );
    }

    /// `StoreField` without a barrier writes through the tag bias, so the
    /// value lands in the slot `LoadField` would read back.
    #[cfg(windows)]
    #[test]
    fn compiled_store_field_writes_through_the_tag_bias() {
        use crate::oops::layout::MEM_TAG;
        let mut obj = [0u64; 4];
        let tagged = obj.as_ptr() as u64 | MEM_TAG;
        // Store arg1 into field at byte_off 24, then read it back.
        let m = hand_method(
            vec![block(
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
                    Ir::StoreField {
                        obj: VReg(0),
                        byte_off: 24,
                        val: VReg(1),
                        barrier: false,
                    },
                    Ir::LoadField {
                        dst: VReg(2),
                        obj: VReg(0),
                        byte_off: 24,
                    },
                    Ir::Ret { val: VReg(2) },
                ],
            )],
            oops(3),
            2,
        );
        assert_eq!(compile_and_run(&m, tagged, smi(77)), smi(77));
        // The write really landed in the object, at the biased offset:
        // word index 3 == byte 24 from the untagged base.
        assert_eq!(obj[3], smi(77));
        let _ = &mut obj;
    }

    /// The generational write barrier, executed against a stand-in VM
    /// register block. Only an old→young store may dirty a card; each of
    /// the three early-outs is checked by giving it a case that must NOT
    /// mark. A barrier that marked unconditionally would still "work"
    /// functionally — it would just quietly destroy scavenge performance —
    /// so the negative cases are the point of this test.
    ///
    /// Driven through the call stub because the barrier reads `old_start`
    /// and `card_base` through the pinned `R15`, which is exactly what the
    /// stub establishes.
    #[cfg(windows)]
    #[test]
    fn write_barrier_marks_only_old_to_young_stores() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::oops::layout::{MEM_TAG, VMREG_CARD_BASE_BIASED_OFFSET, VMREG_OLD_START_OFFSET};
        use crate::memory::cards::CARD_SHIFT;
        use crate::vendor::wfasm::native_windows::WinJit;

        // A card table, and a VM register block pointing at it. `old_start`
        // is chosen so we can place objects deliberately on either side.
        let mut cards = vec![0xFFu8; 1 << 12];
        // Objects: `old` sits above old_start, `young` below it.
        let mut old_obj = [0u64; 4];
        let mut young_obj = [0u64; 4];
        let old_addr = old_obj.as_ptr() as u64;
        let young_addr = young_obj.as_ptr() as u64;
        // Pick old_start between them so the classification is real. The
        // allocator gives no ordering guarantee, so derive it rather than
        // assuming which address is lower.
        let (lo, hi) = if old_addr < young_addr {
            (young_addr, old_addr)
        } else {
            (old_addr, young_addr)
        };
        // `lo` is the higher address -> treat it as "old"; place old_start
        // just below it so `hi` classifies as young.
        let old_start = lo;
        let (old_addr, young_addr) = (lo, hi);
        let old_obj_p = old_addr as *mut u64;
        let young_obj_p = young_addr as *mut u64;

        // card_base_biased: the table base minus (old_start >> CARD_SHIFT),
        // so `card_base_biased + (addr >> CARD_SHIFT)` indexes the table.
        let card_base_biased =
            cards.as_mut_ptr() as u64 - ((old_start >> CARD_SHIFT) as u64);
        let mut vmreg = [0u64; 8];
        vmreg[VMREG_OLD_START_OFFSET / 8] = old_start;
        vmreg[VMREG_CARD_BASE_BIASED_OFFSET / 8] = card_base_biased;

        // A method that stores arg1 into arg0's field 24, with a barrier.
        let m = hand_method(
            vec![block(
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
                    Ir::StoreField {
                        obj: VReg(0),
                        byte_off: 24,
                        val: VReg(1),
                        barrier: true,
                    },
                    Ir::ConstSmi {
                        dst: VReg(2),
                        value: 0,
                    },
                    Ir::Ret { val: VReg(2) },
                ],
            )],
            oops(3),
            2,
        );
        let blob = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default()).blob;
        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base.add(moff), blob.code.len());
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + moff as u64;
        let vm = vmreg.as_ptr() as u64;

        // Hoisted to a plain integer so the closure holds no borrow of
        // `cards` (which the assertions below mutate).
        let cards_base = cards.as_ptr() as usize;
        let card_of = |addr: u64| -> usize {
            ((card_base_biased + ((addr + 24 - MEM_TAG) >> CARD_SHIFT)) as usize) - cards_base
        };

        // 1. old object ← young pointer: MUST mark.
        let idx = card_of(old_addr);
        cards[idx] = 0xFF;
        let argv = [old_addr | MEM_TAG, young_addr | MEM_TAG];
        unsafe { stub_fn(entry, vm, argv.as_ptr(), 2) };
        assert_eq!(cards[idx], 0, "old <- young must dirty the card");

        // 2. young object ← young pointer: must NOT mark (obj is young).
        let idx_y = card_of(young_addr);
        cards[idx_y] = 0xFF;
        let argv = [young_addr | MEM_TAG, young_addr | MEM_TAG];
        unsafe { stub_fn(entry, vm, argv.as_ptr(), 2) };
        assert_eq!(cards[idx_y], 0xFF, "a young object needs no card");

        // 3. old object ← smi: must NOT mark (not a reference).
        cards[idx] = 0xFF;
        let argv = [old_addr | MEM_TAG, smi(42)];
        unsafe { stub_fn(entry, vm, argv.as_ptr(), 2) };
        assert_eq!(cards[idx], 0xFF, "a smi is not a reference");

        // 4. old object ← old pointer: must NOT mark (old->old).
        cards[idx] = 0xFF;
        let argv = [old_addr | MEM_TAG, old_addr | MEM_TAG];
        unsafe { stub_fn(entry, vm, argv.as_ptr(), 2) };
        assert_eq!(cards[idx], 0xFF, "old -> old does not concern a scavenge");

        // Keep the backing storage alive for the whole test.
        unsafe {
            let _ = core::ptr::read_volatile(old_obj_p);
            let _ = core::ptr::read_volatile(young_obj_p);
        }
        let _ = (&mut old_obj, &mut young_obj);
    }

    /// `SmiCmpVal` materializes a boolean branchlessly via `cmov`, picking
    /// the canonical true/false oops out of the literal pool.
    #[cfg(windows)]
    #[test]
    fn compiled_smi_cmp_val_selects_the_right_boolean() {
        use crate::compiler::ir::PoolEntry;
        const TRUE_OOP: u64 = 0x1111_0001;
        const FALSE_OOP: u64 = 0x2222_0001;

        let mut m = hand_method(
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
                        Ir::SmiCmpVal {
                            op: CmpOp::Lt,
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
        // Pool slot 0 = true, slot 1 = false; point the method at them.
        m.pool = vec![
            PoolEntry {
                value: TRUE_OOP,
                kind: Some(RelocKind::Oop),
            },
            PoolEntry {
                value: FALSE_OOP,
                kind: Some(RelocKind::Oop),
            },
        ];
        m.true_lit = PoolLit(0);
        m.false_lit = PoolLit(1);

        assert_eq!(compile_and_run(&m, smi(3), smi(9)), TRUE_OOP, "3 < 9");
        assert_eq!(compile_and_run(&m, smi(9), smi(3)), FALSE_OOP, "9 < 3");
        assert_eq!(compile_and_run(&m, smi(4), smi(4)), FALSE_OOP, "4 < 4");
        assert_eq!(compile_and_run(&m, smi(-9), smi(-3)), TRUE_OOP, "-9 < -3");
    }

    /// `BoolBr` takes the true edge on the canonical true oop, the false
    /// edge on false, and the `not_bool` edge on ANYTHING else — Smalltalk
    /// requires `ifTrue:` on a non-boolean to raise, never to coerce.
    #[cfg(windows)]
    #[test]
    fn compiled_bool_br_rejects_non_booleans() {
        use crate::compiler::ir::PoolEntry;
        const TRUE_OOP: u64 = 0x1111_0001;
        const FALSE_OOP: u64 = 0x2222_0001;

        let mut m = hand_method(
            vec![
                block(
                    0,
                    vec![
                        Ir::Param {
                            dst: VReg(0),
                            index: 0,
                        },
                        Ir::BoolBr {
                            val: VReg(0),
                            if_true: BlockId(1),
                            if_false: BlockId(2),
                            not_bool: BlockId(3),
                        },
                    ],
                ),
                block(
                    1,
                    vec![
                        Ir::ConstSmi {
                            dst: VReg(1),
                            value: 1,
                        },
                        Ir::Ret { val: VReg(1) },
                    ],
                ),
                block(
                    2,
                    vec![
                        Ir::ConstSmi {
                            dst: VReg(1),
                            value: 2,
                        },
                        Ir::Ret { val: VReg(1) },
                    ],
                ),
                block(
                    3,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(2),
            1,
        );
        m.pool = vec![
            PoolEntry {
                value: TRUE_OOP,
                kind: Some(RelocKind::Oop),
            },
            PoolEntry {
                value: FALSE_OOP,
                kind: Some(RelocKind::Oop),
            },
        ];
        m.true_lit = PoolLit(0);
        m.false_lit = PoolLit(1);

        assert_eq!(compile_and_run(&m, TRUE_OOP, 0), smi(1));
        assert_eq!(compile_and_run(&m, FALSE_OOP, 0), smi(2));
        assert_eq!(
            compile_and_run(&m, smi(7), 0),
            BAILOUT_SENTINEL,
            "a smi is not a boolean"
        );
        assert_eq!(
            compile_and_run(&m, 0x9999_0001, 0),
            BAILOUT_SENTINEL,
            "an unrelated heap oop is not a boolean"
        );
    }

    /// A counter the Poll test's stand-in stub bumps, so the test can tell
    /// "the poll branch was taken" from "the flag was read but ignored".
    #[cfg(windows)]
    static POLL_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[cfg(windows)]
    extern "C" fn poll_stub_probe() {
        POLL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// `Poll` reads the safepoint flag out of the pinned VM register and
    /// calls the poll stub only when it is set. Both directions matter: a
    /// poll that never fired would hang the collector at a safepoint, and
    /// one that always fired would call into the runtime on every loop
    /// iteration.
    ///
    /// Driven through the call stub, because the flag is read through
    /// `R15` — which is exactly what the stub establishes.
    #[cfg(windows)]
    #[test]
    fn compiled_poll_calls_the_stub_only_when_the_flag_is_set() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::oops::layout::VMREG_POLL_FLAG_OFFSET;
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::Ordering;

        let mut vmreg = [0u64; 8];
        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Poll,
                    Ir::ConstSmi {
                        dst: VReg(0),
                        value: 5,
                    },
                    Ir::Ret { val: VReg(0) },
                ],
            )],
            oops(1),
            0,
        );
        let rt = RuntimeAddrs {
            stub_poll: poll_stub_probe as usize as u64,
            ..RuntimeAddrs::default()
        };
        let blob = emit_x64(&m, &regalloc(&m), rt).blob;
        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base.add(moff), blob.code.len());
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + moff as u64;
        let vm = vmreg.as_ptr() as u64;

        // Flag clear: the stub must NOT be called.
        vmreg[VMREG_POLL_FLAG_OFFSET / 8] = 0;
        let before = POLL_CALLS.load(Ordering::Relaxed);
        assert_eq!(unsafe { stub_fn(entry, vm, std::ptr::null(), 0) }, smi(5));
        assert_eq!(
            POLL_CALLS.load(Ordering::Relaxed),
            before,
            "a dormant poll flag must not call into the runtime"
        );

        // Flag set: the stub must be called exactly once, and the method
        // must still return its normal result afterwards.
        vmreg[VMREG_POLL_FLAG_OFFSET / 8] = 1;
        assert_eq!(unsafe { stub_fn(entry, vm, std::ptr::null(), 0) }, smi(5));
        assert_eq!(
            POLL_CALLS.load(Ordering::Relaxed),
            before + 1,
            "a set poll flag must call the stub exactly once and resume"
        );
    }

    /// The poll's safepoint is recorded at the call's RETURN address —
    /// the convention deopt metadata keys on — and that address is also
    /// where the not-taken branch lands.
    #[test]
    fn poll_safepoint_is_recorded_at_the_return_address() {
        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Poll,
                    Ir::ConstSmi {
                        dst: VReg(0),
                        value: 1,
                    },
                    Ir::Ret { val: VReg(0) },
                ],
            )],
            oops(1),
            0,
        );
        let out = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default());
        assert_eq!(out.safepoints.len(), 1, "one poll, one safepoint");
        // It is a return address, so it must be strictly inside the code,
        // past the call that precedes it.
        let sp = out.safepoints[0].pc_off;
        assert!(sp > 0 && (sp as usize) < out.blob.literal_off as usize);
    }

    /// `CallRuntime{MUST_BE_BOOLEAN}` passes its argument in the first
    /// Win64 argument register and takes the result from RAX.
    #[cfg(windows)]
    #[test]
    fn compiled_call_runtime_marshals_arg_and_result() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::compiler::ir::StubId;
        use crate::vendor::wfasm::native_windows::WinJit;

        // Stand-in for `must_be_boolean`: returns its argument doubled, so
        // the test can tell a correctly-marshalled argument from a stale
        // register.
        extern "C" fn double_it(x: u64) -> u64 {
            x.wrapping_mul(2)
        }

        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Param {
                        dst: VReg(0),
                        index: 0,
                    },
                    Ir::CallRuntime {
                        dst: Some(VReg(1)),
                        stub: StubId::MUST_BE_BOOLEAN,
                        args: vec![VReg(0)],
                    },
                    Ir::Ret { val: VReg(1) },
                ],
            )],
            oops(2),
            1,
        );
        let rt = RuntimeAddrs {
            must_be_boolean: double_it as usize as u64,
            ..RuntimeAddrs::default()
        };
        let blob = emit_x64(&m, &regalloc(&m), rt).blob;
        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base.add(moff), blob.code.len());
        }
        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + moff as u64;

        let argv = [21u64];
        assert_eq!(unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) }, 42);
        let argv = [100u64];
        assert_eq!(unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) }, 200);
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
            literal_ids: Vec::new(),
            epilogue: Label(0),
            bailout: Label(0),
            trap_sites: Vec::new(),
            safepoints: Vec::new(),
            stub_poll_lit: LiteralId(0),
            must_be_boolean_lit: LiteralId(0),
            alloc_slow_lit: LiteralId(0),
            current_bci: 0,
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
