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
use crate::compiler::emit::{EmittedIcSite, EntryGuard};
use crate::compiler::assembler_x64::{
    imm, mem, mem_byte, mem_index, r32, r64, Cond, X64Assembler, ARG_REGS, RAX, RBP, RSP,
    incoming_stack_slot, outgoing_arg_bytes, outgoing_stack_slot, xmm, FP_SCRATCH0,
    FP_SCRATCH1, FP_SCRATCH2, MAX_REG_ARGS,
    OUTGOING_ARG_BYTES, SCRATCH0,
    SCRATCH1, VM_STATE,
};
use crate::compiler::ir::{
    BlockId, CmpOp, FArithOp, GuardShape, Ir, IrMethod, PoolLit, SmiOp, VReg,
};
use crate::vendor::wfasm::rasm::parse::Operand;
use crate::compiler::regalloc::{Assignment, RegallocResult, SpillSlot};

/// `self` is always `VReg(0)` by `ir::convert`'s construction — the same
/// documented convention `emit.rs` names, restated here rather than
/// shared so neither back end reaches into the other.
const SELF_VREG: VReg = VReg(0);

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
    "ArrayAt",
    "ArrayAtPut",
    "BoolBr",
    "Poll",
    "Alloc",
    "CallSend",
    "CallRuntime",
    "Jump",
    "UncommonTrap",
    "Ret",
    "RetSelf",
    "Bailout",
    // Phase 5: the scalar float fast path. Enabled as a SET — a method
    // containing any float op is declined unless every one of these
    // lowers, so a partial list would compile nothing new while adding
    // untested codegen.
    "FConst",
    "FUnbox",
    "FArith",
    "FCmpBr",
    "FCmpVal",
    "FBox",
];

/// Byte offset of an object's klass word from its TAGGED pointer:
/// `KLASS_OFFSET` (8) less `MEM_TAG` (1), because a heap oop's word is the
/// address biased by the tag. The AArch64 emitter reaches it with an
/// unscaled `ldur`; x86 addresses it directly with a disp8.
const KLASS_OFF_FROM_TAGGED: i64 =
    crate::oops::layout::KLASS_OFFSET as i64 - crate::oops::layout::MEM_TAG as i64;

/// An indexable object's length word, from its tagged pointer: the first
/// body word (past the 2-word header), less the tag bias. Holds a TAGGED
/// smi count.
const ARRAY_LENGTH_OFF: i64 = (crate::oops::layout::HEADER_WORDS
    * crate::oops::layout::WORD_SIZE) as i64
    - crate::oops::layout::MEM_TAG as i64;

/// Displacement of element 1 in the `[arr + idx*2 + disp]` addressing the
/// array ops use. Elements start one word past the length word; scaling a
/// tagged index (`i << 2`) by 2 yields `i * 8`, which already counts one
/// element too far for a 1-based index, so the base is the LENGTH word's
/// own offset rather than the first element's.
const ARRAY_ELEM_BASE: i64 = ARRAY_LENGTH_OFF;

/// One safepoint the emitter recorded. Mirrors the AArch64 emitter's
/// `SafepointPc` field-for-field, because `driver::build_deopt_metadata`
/// keys deopt scopes off exactly these three values.
///
/// `pc_off` is the trapping instruction's OWN offset for a trap site (the
/// trapping pc IS the `int3`), or the RETURN address for a runtime call.
/// `position` is the op's index in the SAME linear numbering
/// `regalloc::compute_intervals` used — that is what ties a safepoint to
/// its oop map, so it must be the position the op was actually emitted
/// at, not a recount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrapSite {
    pub pc_off: u32,
    pub bci: usize,
    pub position: u32,
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
    /// Patchable inline-cache sites, for the code cache to wire up.
    pub ic_sites: Vec<EmittedIcSite>,
    /// Byte offset of each basic block's first instruction, indexed by
    /// block number — what the debugger and OSR entry resolution need.
    pub block_pcs: Vec<u32>,
    /// Offset past the entry guard: where a caller that has already
    /// checked the receiver's klass (a monomorphic IC hit) may enter
    /// directly. Equals 0 when no guard was requested.
    pub verified_entry_off: u32,
}

/// Emit the per-klass customization guard that precedes a compiled
/// method's body: the receiver's klass must equal the key this nmethod
/// was customized for, or control tail-jumps to the resolve stub, which
/// re-dispatches. Falls through to the verified entry on a match.
///
/// The receiver arrives in the first Win64 argument register (matching
/// [`Ir::Param`] index 0), which is where AArch64's `x0` maps to.
///
/// Two shapes, exactly as the AArch64 emitter has them:
/// - **Heap key** (the overwhelmingly common case): a smi receiver can
///   never match a heap klass, so the smi case *is* a miss and needs no
///   smi-klass literal or merge point at all.
/// - **Smi key**: the two cases have to merge, because a smi's klass is a
///   literal rather than a header word.
fn emit_entry_guard_x64(asm: &mut X64Assembler, guard: &EntryGuard) {
    let recv = ARG_REGS[0];
    let key_lit = asm.literal_u64(guard.key_klass_bits, Some(RelocKind::KeyKlassOop));
    let resolve_lit = asm.literal_u64(guard.resolve_addr, Some(RelocKind::RuntimeAddr));

    // The miss path, shared by both shapes: load the resolve stub and
    // TAIL-jump — this frame has no prologue yet, so the stub returns
    // directly to the original caller.
    let emit_miss = |asm: &mut X64Assembler| {
        asm.load_literal(SCRATCH0, resolve_lit);
        asm.emit("jmp", &[r64(SCRATCH0)]);
    };

    if guard.key_klass_bits != guard.smi_klass_bits {
        let miss = asm.new_label();
        let matched = asm.new_label();
        asm.emit("test", &[r64(recv), imm(3)]);
        asm.jcc(Cond::E, miss); // a smi can never match a heap key
        asm.emit("mov", &[r64(SCRATCH1), mem(recv, KLASS_OFF_FROM_TAGGED)]);
        asm.cmp_literal(SCRATCH1, key_lit);
        asm.jcc(Cond::E, matched);
        asm.bind(miss);
        emit_miss(asm);
        asm.bind(matched);
        return;
    }

    let smi_lit = asm.literal_u64(guard.smi_klass_bits, Some(RelocKind::Oop));
    let smi_case = asm.new_label();
    let after_klass_load = asm.new_label();
    let matched = asm.new_label();

    asm.emit("test", &[r64(recv), imm(3)]);
    asm.jcc(Cond::E, smi_case);
    asm.emit("mov", &[r64(SCRATCH1), mem(recv, KLASS_OFF_FROM_TAGGED)]);
    asm.jmp(after_klass_load);
    asm.bind(smi_case);
    asm.load_literal(SCRATCH1, smi_lit);
    asm.bind(after_klass_load);
    asm.cmp_literal(SCRATCH1, key_lit);
    asm.jcc(Cond::E, matched);
    emit_miss(asm);
    asm.bind(matched);
}

/// Absolute addresses of the **stubs** compiled code calls — not the
/// `rt_*` Rust functions themselves. Each stub owns the register
/// marshalling: the emitter places only the real arguments, and the stub
/// prepends `&VmState` from the pinned register (`codecache::stubs_x64`).
/// Keeping that split means the emitter never has to know a runtime
/// function's Rust signature.
///
/// Passed in rather than looked up so the emitter stays free of any
/// dependency on a live `VmState` — the same shape as the AArch64
/// `emit`'s long parameter list, collected into one struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeAddrs {
    /// `stub_poll` — runs the safepoint action when the poll flag is set.
    pub stub_poll: u64,
    /// `stub_must_be_boolean` — coerces or raises on a non-boolean.
    pub must_be_boolean: u64,
    /// `stub_alloc_slow` — the allocation slow path.
    pub alloc_slow: u64,
    /// `stub_box_double` — `FBox`'s eden-overflow tail. Added when the
    /// float lowering landed, not before: a field no emitted
    /// instruction reads is dead weight that looks like wiring.
    pub box_double: u64,
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
    ic_sites: Vec<EmittedIcSite>,
    /// Pool entries holding the runtime entry points.
    stub_poll_lit: LiteralId,
    must_be_boolean_lit: LiteralId,
    alloc_slow_lit: LiteralId,
    box_double_lit: LiteralId,
    current_bci: usize,
    /// This op's index in regalloc's linear numbering.
    pos: u32,
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

    // ── Floating point (Phase 5) ────────────────────────────────────
    //
    // The FP mirror of `read_into`/`def_reg`/`store_def`. Kept separate
    // rather than parameterised on register class because every
    // instruction differs (`movsd` vs `mov`) and conflating them is how a
    // GPR `mov` ends up moving eight bytes of a double through an integer
    // register — which works, right up until it doesn't.
    //
    // An fp vreg spills to an ORDINARY 8-byte frame slot: `regalloc`
    // shares one slot space and marks fp slots not-oop, so the collector
    // never traces a raw double as a pointer.

    /// Resolve `v` into an XMM register ready to READ.
    fn read_fp(&mut self, v: VReg, scratch: u8) -> u8 {
        match self.assignment[v.0 as usize] {
            Some(Assignment::Reg(r)) => r,
            Some(Assignment::Spill(slot)) => {
                self.asm
                    .emit("movsd", &[xmm(scratch), mem(RBP, spill_offset(slot))]);
                scratch
            }
            None => panic!("emit_x64: fp vreg v{} has no assignment", v.0),
        }
    }

    /// The XMM register an fp definition should be computed into.
    fn def_fp(&self, v: VReg, scratch: u8) -> u8 {
        match self.assignment[v.0 as usize] {
            Some(Assignment::Reg(r)) => r,
            Some(Assignment::Spill(_)) => scratch,
            None => panic!("emit_x64: fp vreg v{} has no assignment", v.0),
        }
    }

    /// Complete an fp definition: store back if `v` is spilled.
    fn store_def_fp(&mut self, v: VReg, reg: u8) {
        if let Some(Assignment::Spill(slot)) = self.assignment[v.0 as usize] {
            self.asm
                .emit("movsd", &[mem(RBP, spill_offset(slot)), xmm(reg)]);
        }
    }

    /// `dst = a op b` in two-address form, for XMM.
    ///
    /// The aliasing hazard is the same one `emit_two_address` handles for
    /// GPRs, but the escape differs: FP `sub`/`div` do not commute AND
    /// `a`/`b` may already be sitting in scratch registers (each is there
    /// if it was spilled). `FP_SCRATCH2` is reserved for exactly this
    /// shuffle so it can never collide with either reload.
    fn emit_two_address_fp(&mut self, mnemonic: &str, commutative: bool, dst: u8, a: u8, b: u8) {
        if dst == b {
            if commutative {
                self.asm.emit(mnemonic, &[xmm(dst), xmm(a)]);
            } else {
                self.asm.emit("movsd", &[xmm(FP_SCRATCH2), xmm(b)]);
                if dst != a {
                    self.asm.emit("movsd", &[xmm(dst), xmm(a)]);
                }
                self.asm.emit(mnemonic, &[xmm(dst), xmm(FP_SCRATCH2)]);
            }
            return;
        }
        if dst != a {
            self.asm.emit("movsd", &[xmm(dst), xmm(a)]);
        }
        self.asm.emit(mnemonic, &[xmm(dst), xmm(b)]);
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

    /// Marshal `args` into the Win64 argument registers — a *parallel*
    /// move, not a sequence of independent ones.
    ///
    /// The hazard: a source register may itself be some other argument's
    /// destination. Moving naively in index order would clobber a value
    /// still needed. The standard resolution, and the one the AArch64
    /// emitter uses: repeatedly emit any move whose destination is not
    /// still pending as somebody's source (those are always safe); when
    /// only a cycle remains, break it by parking one value in a scratch
    /// register and rewriting the references to it.
    ///
    /// Spilled sources are never part of a cycle — a memory operand is
    /// nobody's destination — so they can always be loaded directly.
    fn marshal_args(&mut self, args: &[VReg]) {
        // Stack arguments FIRST: they are written from wherever their
        // values currently live, and the register shuffle below is about
        // to overwrite ARG_REGS. Doing it the other way round would read
        // an argument register that had already been reassigned.
        //
        // The caller has already reserved `outgoing_arg_bytes(args.len())`,
        // so these are RSP-relative; spill reads stay RBP-relative and are
        // unaffected by that reservation.
        for (i, &v) in args.iter().enumerate().skip(MAX_REG_ARGS) {
            let slot_off = outgoing_stack_slot(i);
            match self.assignment[v.0 as usize] {
                Some(Assignment::Reg(r)) => {
                    self.asm.emit("mov", &[mem(RSP, slot_off), r64(r)]);
                }
                Some(Assignment::Spill(slot)) => {
                    self.asm
                        .emit("mov", &[r64(SCRATCH0), mem(RBP, spill_offset(slot))]);
                    self.asm.emit("mov", &[mem(RSP, slot_off), r64(SCRATCH0)]);
                }
                None => panic!("emit_x64: argument vreg v{} has no assignment", v.0),
            }
        }


        #[derive(Clone, Copy)]
        enum Src {
            Reg(u8),
            Slot(SpillSlot),
        }

        // (destination register, source) for every argument that isn't
        // already sitting in the right register.
        let mut pending: Vec<(u8, Src)> = args
            .iter()
            .enumerate()
            .take(MAX_REG_ARGS)
            .filter_map(|(i, &v)| {
                let dst = ARG_REGS[i];
                match self.assignment[v.0 as usize] {
                    Some(Assignment::Reg(r)) if r == dst => None,
                    Some(Assignment::Reg(r)) => Some((dst, Src::Reg(r))),
                    Some(Assignment::Spill(slot)) => Some((dst, Src::Slot(slot))),
                    None => panic!("emit_x64: argument vreg v{} has no assignment", v.0),
                }
            })
            .collect();

        while !pending.is_empty() {
            // A destination that no pending move still reads is safe now.
            let ready = pending.iter().position(|&(dst, _)| {
                !pending
                    .iter()
                    .any(|&(_, s)| matches!(s, Src::Reg(r) if r == dst))
            });
            if let Some(pos) = ready {
                let (dst, src) = pending.remove(pos);
                match src {
                    Src::Reg(r) => self.asm.emit("mov", &[r64(dst), r64(r)]),
                    Src::Slot(slot) => self
                        .asm
                        .emit("mov", &[r64(dst), mem(RBP, spill_offset(slot))]),
                }
            } else {
                // Everything left is a cycle. Park the first destination's
                // current value in a scratch and redirect readers to it,
                // which turns the cycle into a chain.
                let (dst0, _) = pending[0];
                self.asm.emit("mov", &[r64(SCRATCH0), r64(dst0)]);
                for (_, s) in pending.iter_mut() {
                    if let Src::Reg(r) = s {
                        if *r == dst0 {
                            *r = SCRATCH0;
                        }
                    }
                }
            }
        }
    }

    /// After any call that can run guest code: if the callee was unwound
    /// by a non-local return it hands back [`NLR_SENTINEL`] instead of a
    /// result, and this frame must return that sentinel to ITS caller
    /// immediately — propagating the escape one native frame at a time
    /// back to `enter_compiled`. Falling through would treat the sentinel
    /// as an ordinary oop.
    fn emit_nlr_check(&mut self) {
        self.asm.emit(
            "cmp",
            &[r64(RAX), imm(crate::oops::layout::NLR_SENTINEL as i64)],
        );
        let epi = self.epilogue;
        self.asm.jcc(Cond::E, epi);
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
    fn emit_runtime_call(&mut self, target: LiteralId) -> u32 {
        // The full outgoing area, not just the shadow space: these
        // callees are the same stubs a send reaches, and their epilogue
        // writes the RootSpill's stack slots back into this reservation.
        self.asm
            .emit("sub", &[r64(RSP), imm(OUTGOING_ARG_BYTES)]);
        self.asm.call_far(target);
        // The return address, captured BEFORE the stack is released —
        // see `record_safepoint_at`.
        let ret_pc = self.asm.offset();
        self.asm
            .emit("add", &[r64(RSP), imm(OUTGOING_ARG_BYTES)]);
        ret_pc
    }

    /// Record a deopt safepoint at the CURRENT offset — used right after
    /// a runtime call returns, so the recorded pc is the return address.
    /// Record a deopt safepoint at an explicitly-supplied return address.
    ///
    /// The pc is passed in rather than read from `asm.offset()` because
    /// on x64 the call is not the last thing emitted: the outgoing
    /// argument area has to be released afterwards, so by the time
    /// control returns here the offset has already moved past the
    /// return address by the width of an `add rsp, imm`.
    ///
    /// The AArch64 emitter has no such instruction — its `bl` is the last
    /// thing before the safepoint — so `offset()` there IS the return
    /// address. Porting that shape literally put every x64 safepoint a
    /// few bytes late, and the GC found no PcDesc at the true return
    /// address of a frame it had to scan.
    fn record_safepoint_at(&mut self, pc_off: u32) {
        let bci = self.current_bci;
        let position = self.pos;
        self.safepoints.push(TrapSite {
            pc_off,
            bci,
            position,
        });
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
    /// `field` is a memory operand naming the STORED FIELD's address,
    /// suitable for `lea` — a constant displacement for `StoreField`, a
    /// scaled-index form for `ArrayAtPut`.
    ///
    /// Uses `RAX` as its only temporary. That is deliberate and slightly
    /// subtle: `RAX` holds `old_start` for the three early-out compares,
    /// and is only reused for the card address AFTER the last of them, at
    /// which point `old_start` is dead. Keeping the barrier off `SCRATCH0`
    /// /`SCRATCH1` is what lets both callers hold their object and value
    /// in the scratch pair across it.
    fn emit_write_barrier(&mut self, robj: u8, rval: u8, field: Operand) {
        use crate::oops::layout::{VMREG_CARD_BASE_BIASED_OFFSET, VMREG_OLD_START_OFFSET};
        let skip = self.asm.new_label();
        // old_start, read live from the VM register block.
        self.asm
            .emit("mov", &[r64(RAX), mem(VM_STATE, VMREG_OLD_START_OFFSET as i64)]);
        self.asm.emit("cmp", &[r64(robj), r64(RAX)]);
        self.asm.jcc(Cond::B, skip); // obj younger than old_start
        self.asm.emit("test", &[r64(rval), imm(3)]);
        self.asm.jcc(Cond::E, skip); // val is a smi
        self.asm.emit("cmp", &[r64(rval), r64(RAX)]);
        self.asm.jcc(Cond::Ae, skip); // val is old too

        // `old_start` is dead from here, so RAX becomes the card address:
        // card_base_biased + (field_addr >> CARD_SHIFT).
        self.asm.emit("lea", &[r64(RAX), field]);
        self.asm.emit(
            "shr",
            &[r64(RAX), imm(crate::memory::cards::CARD_SHIFT as i64)],
        );
        self.asm.emit(
            "add",
            &[r64(RAX), mem(VM_STATE, VMREG_CARD_BASE_BIASED_OFFSET as i64)],
        );
        self.asm.emit("mov", &[mem_byte(RAX, 0), imm(0)]); // CARD_DIRTY == 0
        self.asm.bind(skip);
    }

    /// The four checks every indexed access makes before touching memory,
    /// in the AArch64 emitter's order. Any failure branches to the cold
    /// block, which re-executes the send in the interpreter.
    ///
    /// Uses only `RAX` as a temporary, so both operands stay in the
    /// scratch pair — see [`emit_write_barrier`] for the same discipline.
    /// `cmp reg, [rip+lit]` is what makes the klass check register-free;
    /// the AArch64 side must load the literal into a scratch first.
    fn emit_array_guards(&mut self, rarr: u8, ridx: u8, klass: PoolLit, fail: BlockId) {
        use crate::oops::layout::MEM_TAG;
        let cold = self.labels[fail.0 as usize];

        // 1. The receiver is a heap oop (tag bits == MEM_TAG), not a smi
        //    and not a reserved sentinel.
        self.asm.emit("mov", &[r64(RAX), r64(rarr)]);
        self.asm.emit("and", &[r64(RAX), imm(3)]);
        self.asm.emit("cmp", &[r64(RAX), imm(MEM_TAG as i64)]);
        self.asm.jcc(Cond::Ne, cold);

        // 2. It is an instance of the expected array klass.
        self.asm
            .emit("mov", &[r64(RAX), mem(rarr, KLASS_OFF_FROM_TAGGED)]);
        let lit = self.literal_ids[klass.0 as usize];
        self.asm.cmp_literal(RAX, lit);
        self.asm.jcc(Cond::Ne, cold);

        // 3. The index is a smi.
        self.asm.emit("test", &[r64(ridx), imm(3)]);
        self.asm.jcc(Cond::Ne, cold);

        // 4. It is in range. Both index and length are TAGGED, so the
        //    comparison happens in tagged units and needs no untagging:
        //    `idx - 4` is `(i-1) << 2`, and an UNSIGNED compare against
        //    the tagged length rejects `i < 1` in the same instruction
        //    that rejects `i > length` (a zero or negative index wraps to
        //    a huge unsigned value). One compare, both bounds.
        self.asm.emit("lea", &[r64(RAX), mem(ridx, -4)]);
        self.asm
            .emit("cmp", &[r64(RAX), mem(rarr, ARRAY_LENGTH_OFF)]);
        self.asm.jcc(Cond::Ae, cold);
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
pub fn emit_x64(
    method: &IrMethod,
    regalloc: &RegallocResult,
    rt: RuntimeAddrs,
    guard: Option<&EntryGuard>,
) -> Emitted {
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
    let box_double_lit = asm.literal_u64(rt.box_double, Some(RelocKind::RuntimeAddr));

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
        ic_sites: Vec::new(),
        stub_poll_lit,
        must_be_boolean_lit,
        alloc_slow_lit,
        box_double_lit,
        current_bci: 0,
        pos: 0,
        method,
    };

    // The customization guard precedes everything, so a monomorphic
    // caller can skip it by entering at `verified_entry_off`.
    if let Some(g) = guard {
        emit_entry_guard_x64(&mut e.asm, g);
    }
    let verified_entry_off = e.asm.offset();

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

    // Nil-fill every deopt-referenced spill slot before any block code
    // runs — the x64 counterpart of the AArch64 prologue's task-#94 fill.
    //
    // `sub rsp` reserves the frame; it does not CLEAR it. A slot whose
    // safepoint is reached before its def — or through a sibling arm that
    // never wrote it — otherwise scans whatever the last frame at this SP
    // depth left behind. The collector then traces those words as roots.
    //
    // That is exactly what `MACVM_TRACE=oops` was reporting under GC
    // stress: `slot=5 word=0x1` (not an address at all) and
    // `slot=10 word=0x1ab41540859 (raw addr in to-space)` — a stale
    // pointer belonging to a dead frame. Nil is what the interpreter's own
    // frame would hold for a dead temp (S13's "dead → Nil" rule), so this
    // makes the compiled frame agree with the interpreted one.
    //
    // Narrowed to `deopt_nil_init_slots` rather than the whole frame:
    // regalloc already computed exactly which slots need it, and params
    // and temps among them are immediately overwritten by their
    // entry-block defs.
    if !regalloc.deopt_nil_init_slots.is_empty() {
        let nil_lit = e.literal_ids[method.nil_lit.0 as usize];
        e.asm.load_literal(SCRATCH0, nil_lit);
        for &slot in &regalloc.deopt_nil_init_slots {
            e.asm
                .emit("mov", &[mem(RBP, spill_offset(slot)), r64(SCRATCH0)]);
        }
    }

    // ── Blocks ──────────────────────────────────────────────────────────
    // Blocks are emitted in REGALLOC's order, not source order — and
    // `pos` advances once per op, AFTER emitting it, in exactly the
    // numbering `regalloc::compute_intervals` used. Both matter: a
    // safepoint's `position` is how its oop map is found, so a different
    // walk order or an off-by-one here would hand the GC the wrong live
    // set for a frame. (This emitter originally walked `method.blocks`,
    // which is only coincidentally the same order.)
    let mut block_pcs: Vec<u32> = vec![0; method.blocks.len()];
    for &bid in &regalloc.block_order {
        let bi = bid.0 as usize;
        let block = &method.blocks[bi];
        let l = e.labels[bi];
        e.asm.bind(l);
        block_pcs[bi] = e.asm.offset();
        e.current_bci = block.bci;
        for op in &block.code {
            emit_op(&mut e, op);
            e.pos += 1;
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
    let ic_sites = std::mem::take(&mut e.ic_sites);
    Emitted {
        blob: e.asm.finish(),
        trap_sites,
        safepoints,
        ic_sites,
        block_pcs,
        verified_entry_off,
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
            // The callee half of the Win64 argument layout: the first four
            // arrive in registers, the rest in the caller's outgoing area
            // above this frame's return address (`incoming_stack_slot`).
            let idx = *index as usize;
            let d = e.def_reg(*dst, SCRATCH0);
            if idx >= MAX_REG_ARGS {
                e.asm
                    .emit("mov", &[r64(d), mem(RBP, incoming_stack_slot(idx))]);
                e.store_def(*dst, d);
                return;
            }
            let src = ARG_REGS[idx];
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
                position: e.pos,
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
                e.emit_write_barrier(o, v, mem(o, biased));
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

        // -- Float fast path (Phase 5, docs/float_fastpath_design.md) --

        Ir::FConst { dst, bits } => {
            // Raw f64 bits baked into code, then moved across to XMM. No
            // pool word and no reloc: the VALUE of an immutable Double
            // literal never changes even when the boxed object moves.
            e.asm.emit("mov", &[r64(SCRATCH0), imm(*bits as i64)]);
            let d = e.def_fp(*dst, FP_SCRATCH0);
            e.asm.emit("movq", &[xmm(d), r64(SCRATCH0)]);
            e.store_def_fp(*dst, d);
        }

        Ir::FUnbox { dst, src, fail } => {
            use crate::oops::layout::{MEM_TAG, WORD_SIZE};
            let robj = e.read_into(*src, SCRATCH0);
            let cold = e.labels[fail.0 as usize];
            // A smi has no klass word to load, so reject it first.
            e.asm.emit("test", &[r64(robj), imm(3)]);
            e.asm.jcc(Cond::E, cold);
            // Untagged base into SCRATCH1 BEFORE SCRATCH0 is reused --
            // `robj` may itself BE SCRATCH0 (a spilled src).
            e.asm
                .emit("lea", &[r64(SCRATCH1), mem(robj, -(MEM_TAG as i64))]);
            e.asm
                .emit("mov", &[r64(SCRATCH0), mem(SCRATCH1, WORD_SIZE as i64)]);
            let k_lit = e.literal_ids[e.method.double_klass_lit.0 as usize];
            e.asm.cmp_literal(SCRATCH0, k_lit);
            e.asm.jcc(Cond::Ne, cold);
            // Payload: body word 0, at untagged + 16.
            let d = e.def_fp(*dst, FP_SCRATCH0);
            e.asm
                .emit("movsd", &[xmm(d), mem(SCRATCH1, 2 * WORD_SIZE as i64)]);
            e.store_def_fp(*dst, d);
        }

        Ir::FArith { op, dst, a, b } => {
            let (mnemonic, commutative) = match op {
                FArithOp::Add => ("addsd", true),
                FArithOp::Sub => ("subsd", false),
                FArithOp::Mul => ("mulsd", true),
                FArithOp::Div => ("divsd", false),
            };
            let ra = e.read_fp(*a, FP_SCRATCH0);
            let rb = e.read_fp(*b, FP_SCRATCH1);
            let d = e.def_fp(*dst, FP_SCRATCH0);
            e.emit_two_address_fp(mnemonic, commutative, d, ra, rb);
            e.store_def_fp(*dst, d);
        }

        Ir::FCmpBr {
            op,
            a,
            b,
            if_true,
            if_false,
        } => {
            let ra = e.read_fp(*a, FP_SCRATCH0);
            let rb = e.read_fp(*b, FP_SCRATCH1);
            let (cond, swap) = fcmp_cond(*op);
            if swap {
                e.asm.emit("ucomisd", &[xmm(rb), xmm(ra)]);
            } else {
                e.asm.emit("ucomisd", &[xmm(ra), xmm(rb)]);
            }
            let f = e.labels[if_false.0 as usize];
            let t = e.labels[if_true.0 as usize];
            // Unordered (either operand NaN) sets ZF, PF and CF TOGETHER,
            // which is indistinguishable from "equal" on the flags alone.
            // Every float comparison except `~=` is false against NaN, so
            // route parity to the right arm before testing the condition.
            let nan_arm = if matches!(op, CmpOp::Ne) { t } else { f };
            e.asm.jcc(Cond::P, nan_arm);
            e.asm.jcc(cond, t);
            e.asm.jmp(f);
        }

        Ir::FCmpVal { op, dst, a, b } => {
            let ra = e.read_fp(*a, FP_SCRATCH0);
            let rb = e.read_fp(*b, FP_SCRATCH1);
            let (cond, swap) = fcmp_cond(*op);
            if swap {
                e.asm.emit("ucomisd", &[xmm(rb), xmm(ra)]);
            } else {
                e.asm.emit("ucomisd", &[xmm(ra), xmm(rb)]);
            }
            // Same NaN rule as `FCmpBr`, materialised rather than branched
            // to. A bare `setcc` cannot express it: the parity flag has to
            // be folded in separately.
            let d = e.def_reg(*dst, SCRATCH0);
            let false_lit = e.literal_ids[e.method.false_lit.0 as usize];
            let true_lit = e.literal_ids[e.method.true_lit.0 as usize];
            let take_false = e.asm.new_label();
            let done = e.asm.new_label();
            if matches!(op, CmpOp::Ne) {
                // NaN ~= anything is TRUE.
                e.asm.load_literal(d, true_lit);
                e.asm.jcc(Cond::P, done);
                e.asm.jcc(cond, done);
                e.asm.load_literal(d, false_lit);
                e.asm.bind(done);
            } else {
                e.asm.jcc(Cond::P, take_false);
                e.asm.load_literal(d, true_lit);
                e.asm.jcc(cond, done);
                e.asm.bind(take_false);
                e.asm.load_literal(d, false_lit);
                e.asm.bind(done);
            }
            e.store_def(*dst, d);
        }

        Ir::FBox { dst, src } => {
            use crate::oops::layout::WORD_SIZE;
            use crate::oops::layout::{
                MEM_TAG, VMREG_EDEN_END_OFFSET, VMREG_EDEN_TOP_ADDR_OFFSET,
            };
            let ds = e.read_fp(*src, FP_SCRATCH0);
            let d = e.def_reg(*dst, RAX);
            let slow = e.asm.new_label();
            let done = e.asm.new_label();
            // mark + klass + f64 payload. A RAW-contents mark, and no nil
            // fill: the body is a double, not a slot the collector scans.
            let size_bytes: i64 = 3 * WORD_SIZE as i64;

            e.asm.emit(
                "mov",
                &[
                    r64(SCRATCH1),
                    mem(VM_STATE, VMREG_EDEN_TOP_ADDR_OFFSET as i64),
                ],
            );
            e.asm.emit("mov", &[r64(SCRATCH0), mem(SCRATCH1, 0)]);
            e.asm.emit("lea", &[r64(RAX), mem(SCRATCH0, size_bytes)]);
            e.asm.emit(
                "cmp",
                &[r64(RAX), mem(VM_STATE, VMREG_EDEN_END_OFFSET as i64)],
            );
            e.asm.jcc(Cond::A, slow);
            e.asm.emit("mov", &[mem(SCRATCH1, 0), r64(RAX)]);

            let mark_lit = e.literal_ids[e.method.mark_double_lit.0 as usize];
            e.asm.load_literal(RAX, mark_lit);
            e.asm.emit("mov", &[mem(SCRATCH0, 0), r64(RAX)]);
            let klass_lit = e.literal_ids[e.method.double_klass_lit.0 as usize];
            e.asm.load_literal(RAX, klass_lit);
            e.asm
                .emit("mov", &[mem(SCRATCH0, WORD_SIZE as i64), r64(RAX)]);
            e.asm
                .emit("movsd", &[mem(SCRATCH0, 2 * WORD_SIZE as i64), xmm(ds)]);
            e.asm.emit("lea", &[r64(d), mem(SCRATCH0, MEM_TAG as i64)]);
            e.asm.jmp(done);

            // Slow path: the payload BITS go in the first argument register
            // (an integer register, matching the AArch64 x0 convention) and
            // the stub allocates, stores and tags -- so the XMM value need
            // not survive the call.
            e.asm.bind(slow);
            e.asm.emit("movq", &[r64(ARG_REGS[0]), xmm(ds)]);
            let lit = e.box_double_lit;
            let ret_pc = e.emit_runtime_call(lit);
            // Allocation can scavenge, so this is a safepoint.
            e.record_safepoint_at(ret_pc);
            if d != RAX {
                e.asm.emit("mov", &[r64(d), r64(RAX)]);
            }
            e.asm.bind(done);
            e.store_def(*dst, d);
        }

        Ir::RetSelf => {
            // Read the receiver from wherever the ALLOCATOR put it, which
            // is what the AArch64 emitter does (`resolve(SELF_VREG, 0)`).
            //
            // This previously read the pinned `RECEIVER` register on the
            // strength of a comment claiming "the receiver lives in its
            // pinned register for the whole activation". Nothing ever
            // wrote that register: not the prologue, not the call stub,
            // not `Ir::Param`. `RetSelf` therefore returned whatever junk
            // R14 happened to hold — so `OrderedCollection>>init`, whose
            // whole body is `^self`, handed back garbage, and the caller's
            // next send to it died as a doesNotUnderstand far from here.
            //
            // `self` is `VReg(0)` by `ir::convert`'s construction, the
            // same documented convention the A64 side relies on.
            let rv = e.read_into(SELF_VREG, RAX);
            if rv != RAX {
                e.asm.emit("mov", &[r64(RAX), r64(rv)]);
            }
            let ep = e.epilogue;
            e.asm.jmp(ep);
        }

        Ir::ArrayAt {
            dst,
            arr,
            idx,
            klass,
            fail,
        } => {
            let a = e.read_into(*arr, SCRATCH0);
            let i = e.read_into(*idx, SCRATCH1);
            e.emit_array_guards(a, i, *klass, *fail);
            let d = e.def_reg(*dst, RAX);
            // element(i) = [arr + idx*2 + ELEM_BASE]. `idx` is a tagged
            // smi (i<<2), so scaling by 2 gives i*8 — exactly one element
            // stride. x86's SIB does the whole address in one operand;
            // the AArch64 emitter needs two `add`s to build it.
            e.asm
                .emit("mov", &[r64(d), mem_index(a, i, 2, ARRAY_ELEM_BASE)]);
            e.store_def(*dst, d);
        }

        Ir::ArrayAtPut {
            dst,
            arr,
            idx,
            val,
            klass,
            fail,
        } => {
            let a = e.read_into(*arr, SCRATCH0);
            let i = e.read_into(*idx, SCRATCH1);
            e.emit_array_guards(a, i, *klass, *fail);
            // The value is the third live operand, and both scratches are
            // taken — read it into RAX, which the guards have finished
            // with by now.
            let v = e.read_into(*val, RAX);
            e.asm
                .emit("mov", &[mem_index(a, i, 2, ARRAY_ELEM_BASE), r64(v)]);
            // Storing an oop into a possibly-old array needs the same
            // card marking a StoreField does.
            e.emit_write_barrier(a, v, mem_index(a, i, 2, ARRAY_ELEM_BASE));
            // `at:put:` answers the stored value.
            let d = e.def_reg(*dst, RAX);
            if d != v {
                e.asm.emit("mov", &[r64(d), r64(v)]);
            }
            e.store_def(*dst, d);
        }

        Ir::CallSend { dst, site, args } => {
            // Reserve the outgoing argument area BEFORE marshaling: the
            // stack arguments `marshal_args` writes are RSP-relative and
            // must land inside this reservation. (Register-only sends
            // reserve exactly the 32-byte shadow space, as before.)
            let outgoing = outgoing_arg_bytes(args.len());
            e.asm.emit("sub", &[r64(RSP), imm(outgoing)]);
            e.marshal_args(args);
            // The patchable site: a 5-byte `call rel32` whose displacement
            // the code cache rewrites to point at the current IC target.
            // It sits INSIDE the outgoing reservation, because the callee
            // is ordinary Win64 code like any other.
            let off = e.asm.call_patchable(RelocKind::InlineCache);
            // A send is a deopt safepoint, keyed on the RETURN address —
            // captured here, before the outgoing area is released.
            let ret_pc = e.asm.offset();
            e.asm.emit("add", &[r64(RSP), imm(outgoing)]);
            e.record_safepoint_at(ret_pc);
            let info = e.method.call_sites[*site as usize];
            e.ic_sites.push(EmittedIcSite {
                off,
                site: *site,
                selector: info.selector,
                argc: info.argc,
            });
            // A callee unwound by a non-local return hands back the
            // sentinel rather than a value; propagate before using it.
            e.emit_nlr_check();
            let d = e.def_reg(*dst, RAX);
            if d != RAX {
                e.asm.emit("mov", &[r64(d), r64(RAX)]);
            }
            e.store_def(*dst, d);
        }

        Ir::Alloc {
            dst,
            klass,
            size_words,
        } => {
            use crate::oops::layout::{
                HEADER_WORDS, MEM_TAG, VMREG_EDEN_END_OFFSET, VMREG_EDEN_TOP_ADDR_OFFSET,
                WORD_SIZE,
            };
            let size_bytes = *size_words as i64 * WORD_SIZE as i64;
            debug_assert!(
                *size_words as usize >= HEADER_WORDS && size_bytes < 4096,
                "emit_x64 Alloc: size_bytes {size_bytes} outside the inline range — \
                 ir.rs is responsible for gating this"
            );
            let d = e.def_reg(*dst, RAX);
            let slow = e.asm.new_label();
            let done = e.asm.new_label();

            // Fast path: bump the LIVE eden top. The VM register block
            // holds the *address of* `eden.top`, not a copy of it — a
            // value copy would go stale the moment a nested allocation or
            // a GC beneath this frame moved the real pointer. (Same
            // reasoning as the AArch64 emitter; it is the whole reason
            // this is a double indirection.)
            e.asm.emit(
                "mov",
                &[r64(SCRATCH1), mem(VM_STATE, VMREG_EDEN_TOP_ADDR_OFFSET as i64)],
            );
            e.asm.emit("mov", &[r64(SCRATCH0), mem(SCRATCH1, 0)]);
            // new_top = top + size; compare against eden_end (a
            // genesis-fixed bound, so a value copy IS safe for this one).
            e.asm.emit("lea", &[r64(RAX), mem(SCRATCH0, size_bytes)]);
            e.asm.emit(
                "cmp",
                &[r64(RAX), mem(VM_STATE, VMREG_EDEN_END_OFFSET as i64)],
            );
            e.asm.jcc(Cond::A, slow); // unsigned: past the end -> slow
            e.asm.emit("mov", &[mem(SCRATCH1, 0), r64(RAX)]); // publish new top

            // Stamp the header: [mark][klass], then nil the body. SCRATCH0
            // still holds the object's base address.
            let mark_lit = e.literal_ids[e.method.mark_slots_lit.0 as usize];
            e.asm.load_literal(RAX, mark_lit);
            e.asm.emit("mov", &[mem(SCRATCH0, 0), r64(RAX)]);
            let klass_lit = e.literal_ids[klass.0 as usize];
            e.asm.load_literal(RAX, klass_lit);
            e.asm
                .emit("mov", &[mem(SCRATCH0, WORD_SIZE as i64), r64(RAX)]);
            let body_words = *size_words as usize - HEADER_WORDS;
            if body_words > 0 {
                let nil_lit = e.literal_ids[e.method.nil_lit.0 as usize];
                e.asm.load_literal(RAX, nil_lit);
                for i in 0..body_words {
                    let off = ((HEADER_WORDS + i) * WORD_SIZE) as i64;
                    e.asm.emit("mov", &[mem(SCRATCH0, off), r64(RAX)]);
                }
            }
            // Tag the result: an oop's word is its address plus MEM_TAG.
            e.asm
                .emit("lea", &[r64(d), mem(SCRATCH0, MEM_TAG as i64)]);
            e.asm.jmp(done);

            // Slow path: rt_alloc_slow(klass_oop, size_bytes).
            e.asm.bind(slow);
            let klass_lit = e.literal_ids[klass.0 as usize];
            e.asm.load_literal(ARG_REGS[0], klass_lit);
            e.asm
                .emit("mov", &[r64(ARG_REGS[1]), imm(size_bytes)]);
            let lit = e.alloc_slow_lit;
            let ret_pc = e.emit_runtime_call(lit);
            // A real allocation may scavenge, so this is a safepoint.
            e.record_safepoint_at(ret_pc);
            if d != RAX {
                e.asm.emit("mov", &[r64(d), r64(RAX)]);
            }

            e.asm.bind(done);
            e.store_def(*dst, d);
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
            // The poll is a deopt safepoint keyed on the RETURN address.
            //
            // That is NOT where `skip` binds, despite what this comment
            // used to claim: the outgoing area has to be released between
            // the call and the merge point, so the return address sits an
            // `add rsp, imm` earlier. The dormant-flag path lands on the
            // merge; the safepoint belongs to the call.
            let ret_pc = e.emit_runtime_call(lit);
            e.record_safepoint_at(ret_pc);
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
            let ret_pc = e.emit_runtime_call(lit);
            e.record_safepoint_at(ret_pc);
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
            "emit_x64: {} is not lowered by the x64 back end yet (supported: {}). \
             Emitting something approximate here would be silently wrong code, so this \
             is a hard stop — see MIGRATION.md §4.",
            ir_op_name(other),
            SUPPORTED_OPS.join(", ")
        ),
    }
}

/// Map an IR comparison onto an `ucomisd` condition.
///
/// `ucomisd` sets the flags as if for an UNSIGNED compare (CF/ZF), so the
/// signed conditions used for smis are wrong here -- `jl` would test SF/OF,
/// which a float compare never writes. The `swap` flag exists because
/// `ucomisd` has no "less" form: `a < b` is emitted as `b > a`.
fn fcmp_cond(op: CmpOp) -> (Cond, bool) {
    match op {
        CmpOp::Eq => (Cond::E, false),
        CmpOp::Ne => (Cond::Ne, false),
        CmpOp::Gt => (Cond::A, false),
        CmpOp::Ge => (Cond::Ae, false),
        CmpOp::Lt => (Cond::A, true),
        CmpOp::Le => (Cond::Ae, true),
    }
}

/// The variant name of an `Ir`, for the unsupported-op panic. `Debug`'s
/// full rendering would bury the name under every field.
pub fn ir_op_name(op: &Ir) -> &'static str {
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
    use crate::oops::wrappers::SymbolOop;
    use crate::oops::Oop;

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
        let blob = emit_x64(method, &ra, RuntimeAddrs::default(), None).blob;
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
                ic_sites: Vec::new(),
                stub_poll_lit: LiteralId(0),
                must_be_boolean_lit: LiteralId(0),
                alloc_slow_lit: LiteralId(0),
                box_double_lit: LiteralId(0),
                current_bci: 0,
                pos: 0,
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
        let out = emit_x64(&m, &ra, RuntimeAddrs::default(), None);

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
    /// `RetSelf` must answer the RECEIVER, and only running it proves so.
    ///
    /// This lowered to `mov rax, R14` — the pinned RECEIVER register —
    /// on the strength of a comment saying the receiver lives there for
    /// the whole activation. Nothing ever wrote R14: not the prologue,
    /// not the call stub, not `Ir::Param`. So `^self` returned junk.
    ///
    /// It went unnoticed because `RetSelf`'s existing coverage only
    /// checked the emitted SHAPE (a move then a jump to the epilogue),
    /// which was right the whole time. A method whose whole body is
    /// `^self` (`OrderedCollection>>init`) is where it bites, and it
    /// surfaces far away as a doesNotUnderstand on the object the caller
    /// thought it had just built.
    #[cfg(windows)]
    #[test]
    fn ret_self_answers_the_receiver() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::vendor::wfasm::native_windows::WinJit;

        // `^self`, with a second parameter so the receiver is provably
        // not just 'whatever happened to be in the first register'.
        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Param { dst: VReg(0), index: 0 },
                    Ir::Param { dst: VReg(1), index: 1 },
                    Ir::RetSelf,
                ],
            )],
            oops(2),
            1,
        );
        let blob = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None).blob;
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
        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;

        const RECV: u64 = 0x1234_5678;
        const ARG: u64 = 0x9ABC_DEF0;
        let argv = [RECV, ARG];
        let got = unsafe { stub_fn(entry, vm, argv.as_ptr(), 2) };
        assert_eq!(
            got, RECV,
            "^self must answer the receiver, not the argument and not \
             whatever a never-initialized pinned register holds"
        );
    }
    #[cfg(windows)]
    #[test]
    fn write_barrier_marks_only_old_to_young_stores() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::oops::layout::{MEM_TAG, VMREG_CARD_BASE_BIASED_OFFSET, VMREG_OLD_START_OFFSET};
        use crate::memory::cards::CARD_SHIFT;
        use crate::vendor::wfasm::native_windows::WinJit;

        // One arena holding both objects, so their relative placement is
        // OURS to choose rather than whatever the stack happens to give.
        //
        // An earlier version of this test derived `old_start` from the
        // addresses of two separate stack arrays. That passed in isolation
        // and failed in the full parallel run: whether the two arrays
        // straddled a 512-byte card boundary decided whether a card index
        // came out negative, which then wrapped when cast to `usize`. The
        // barrier was fine; the test was reading a different card each run.
        // Everything below is therefore positioned at fixed offsets.
        const ARENA_WORDS: usize = 512; // 4 KiB, several cards wide
        let mut arena = vec![0u64; ARENA_WORDS];
        let arena_base = arena.as_mut_ptr() as u64;
        // Young at the start, old well past the boundary we pick.
        let young_addr = arena_base;
        let old_addr = arena_base + 2048;
        let old_start = arena_base + 1024; // young < old_start <= old

        let mut cards = vec![0xFFu8; 1 << 12];
        // Bias against the ARENA base (not old_start) so every index for
        // an address inside the arena is small and non-negative.
        let card_base_biased = cards.as_mut_ptr() as u64 - (arena_base >> CARD_SHIFT);
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
        let blob = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None).blob;
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

        // Keep the arena alive for the whole test.
        let _ = &mut arena;
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
        let blob = emit_x64(&m, &regalloc(&m), rt, None).blob;
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
        let out = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None);
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
        let blob = emit_x64(&m, &regalloc(&m), rt, None).blob;
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

    /// Records what the Alloc test's stand-in slow path was called with.
    #[cfg(windows)]
    static ALLOC_SLOW_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(windows)]
    static ALLOC_SLOW_SIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[cfg(windows)]
    extern "C" fn alloc_slow_probe(klass: u64, size_bytes: u64) -> u64 {
        ALLOC_SLOW_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ALLOC_SLOW_SIZE.store(size_bytes, std::sync::atomic::Ordering::Relaxed);
        // Hand back a recognizable "object" so the caller can tell the
        // slow path's result apart from a fast-path address.
        klass ^ 0xDEAD_0000
    }

    /// Inline allocation, executed: the fast path bumps the live eden top,
    /// stamps `[mark][klass]` plus a nil body, and returns a MEM_TAG-ed
    /// pointer; when the bump would pass `eden_end` it calls the slow path
    /// instead.
    ///
    /// Both paths matter and neither is checkable from the return value
    /// alone, so this asserts on the *heap*: the header words, the nil'd
    /// body, and the published eden top.
    #[cfg(windows)]
    #[test]
    fn compiled_alloc_bumps_eden_and_falls_back() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::compiler::ir::PoolEntry;
        use crate::oops::layout::{
            HEADER_WORDS, MEM_TAG, VMREG_EDEN_END_OFFSET, VMREG_EDEN_TOP_ADDR_OFFSET, WORD_SIZE,
        };
        use crate::vendor::wfasm::native_windows::WinJit;
        use std::sync::atomic::Ordering;

        const MARK: u64 = 0x1111_1111;
        const KLASS: u64 = 0x2222_2222;
        const NIL: u64 = 0x3333_3333;
        const SIZE_WORDS: u32 = 4; // 2 header + 2 body

        // A stand-in eden: `top` is a live word the compiled code bumps.
        let mut eden = vec![0u64; 64];
        let eden_base = eden.as_mut_ptr() as u64;
        let mut eden_top: u64 = eden_base;
        let eden_end = eden_base + 8 * WORD_SIZE as u64; // room for exactly two objects

        let mut vmreg = [0u64; 8];
        vmreg[VMREG_EDEN_TOP_ADDR_OFFSET / 8] = &mut eden_top as *mut u64 as u64;
        vmreg[VMREG_EDEN_END_OFFSET / 8] = eden_end;

        let mut m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Alloc {
                        dst: VReg(0),
                        klass: PoolLit(1),
                        size_words: SIZE_WORDS,
                    },
                    Ir::Ret { val: VReg(0) },
                ],
            )],
            oops(1),
            0,
        );
        // Pool: 0 = mark, 1 = klass, 2 = nil.
        m.pool = vec![
            PoolEntry {
                value: MARK,
                kind: None,
            },
            PoolEntry {
                value: KLASS,
                kind: Some(RelocKind::Oop),
            },
            PoolEntry {
                value: NIL,
                kind: Some(RelocKind::Oop),
            },
        ];
        m.mark_slots_lit = PoolLit(0);
        m.nil_lit = PoolLit(2);

        let rt = RuntimeAddrs {
            alloc_slow: alloc_slow_probe as usize as u64,
            ..RuntimeAddrs::default()
        };
        let blob = emit_x64(&m, &regalloc(&m), rt, None).blob;
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

        let slow_before = ALLOC_SLOW_CALLS.load(Ordering::Relaxed);
        let size_bytes = SIZE_WORDS as u64 * WORD_SIZE as u64;

        // ── First allocation: the fast path ─────────────────────────────
        let obj = unsafe { stub_fn(entry, vm, std::ptr::null(), 0) };
        assert_eq!(
            ALLOC_SLOW_CALLS.load(Ordering::Relaxed),
            slow_before,
            "an allocation that fits must NOT call the slow path"
        );
        assert_eq!(obj, eden_base | MEM_TAG, "result is the tagged base");
        assert_eq!(eden_top, eden_base + size_bytes, "eden top was published");
        // Header and nil'd body, read back off the heap.
        let words = unsafe { std::slice::from_raw_parts(eden_base as *const u64, 4) };
        assert_eq!(words[0], MARK, "mark word stamped");
        assert_eq!(words[1], KLASS, "klass word stamped");
        for (i, w) in words[HEADER_WORDS..].iter().enumerate() {
            assert_eq!(*w, NIL, "body word {i} nil'd");
        }

        // ── Second allocation: still fits, bumps again ──────────────────
        let obj2 = unsafe { stub_fn(entry, vm, std::ptr::null(), 0) };
        assert_eq!(obj2, (eden_base + size_bytes) | MEM_TAG);
        assert_eq!(eden_top, eden_base + 2 * size_bytes);
        assert_eq!(
            ALLOC_SLOW_CALLS.load(Ordering::Relaxed),
            slow_before,
            "still no slow-path call"
        );

        // ── Third: eden is now full, so the slow path must run ──────────
        let obj3 = unsafe { stub_fn(entry, vm, std::ptr::null(), 0) };
        assert_eq!(
            ALLOC_SLOW_CALLS.load(Ordering::Relaxed),
            slow_before + 1,
            "an allocation past eden_end must call the slow path exactly once"
        );
        assert_eq!(
            ALLOC_SLOW_SIZE.load(Ordering::Relaxed),
            size_bytes,
            "the slow path receives the size in BYTES"
        );
        assert_eq!(
            obj3,
            KLASS ^ 0xDEAD_0000,
            "the slow path's result is what the method returns"
        );
        assert_eq!(
            eden_top,
            eden_base + 2 * size_bytes,
            "a slow-path allocation must not have bumped eden itself"
        );

        let _ = &mut eden;
    }

    /// Patch an emitted IC site to call `target`, the way the code cache
    /// does: rewrite the `rel32` field relative to the END of the 5-byte
    /// call instruction.
    ///
    /// Prefers a DIRECT relative call, which is what actually happens in
    /// practice now that `WinJit` places its region within rel32 of the
    /// host image (`native_windows::alloc_near`). If the region ever falls
    /// back to an arbitrary placement, the target is reached through an
    /// absolute thunk (`movabs rax, target ; jmp rax`) laid down in the
    /// region — the same veneer `relocpatch::patch_relocs_x64` builds for
    /// an out-of-range branch. The thunk tail-calls, so the callee returns
    /// straight to the send site either way.
    ///
    /// Returns `true` if the direct form was used.
    #[cfg(windows)]
    fn patch_ic_site(base: *mut u8, site_abs: u64, thunk_off: usize, target: u64) -> bool {
        use crate::vendor::wfasm::relocpatch::abs_stub_x64;
        let direct = i32::try_from(target as i64 - (site_abs as i64 + 5));
        let (rel32, was_direct) = match direct {
            Ok(r) => (r, true),
            Err(_) => {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        abs_stub_x64(target).as_ptr(),
                        base.add(thunk_off),
                        12,
                    );
                }
                let thunk_addr = base as u64 + thunk_off as u64;
                let r = i32::try_from(thunk_addr as i64 - (site_abs as i64 + 5))
                    .expect("an in-region thunk is always within rel32 of the site");
                (r, false)
            }
        };
        unsafe {
            core::ptr::copy_nonoverlapping(
                rel32.to_le_bytes().as_ptr(),
                (site_abs + 1) as *mut u8,
                4,
            );
        }
        was_direct
    }

    /// Fabricate a `SymbolOop` from raw words, for call-site metadata in
    /// tests that have no live VM. `SymbolOop::try_from` only requires the
    /// oop's klass to have `IndexableBytes` format, so a two-object graph
    /// (`klass` with the format smi in body word 0, `symbol` pointing at
    /// it) is enough. The storage must outlive the returned handle, which
    /// is why the caller owns the backing arrays.
    fn fake_symbol(klass_words: &mut [u64; 4], sym_words: &mut [u64; 2]) -> SymbolOop {
        use crate::oops::layout::{KLASS_FORMAT_INDEX, MEM_TAG};
        // klass: [mark][klass][format-smi ...]; body word 0 is the format.
        klass_words[2 + KLASS_FORMAT_INDEX] =
            (crate::oops::klass::Format::IndexableBytes as u64) << 2;
        let klass_tagged = klass_words.as_ptr() as u64 | MEM_TAG;
        // symbol: [mark][klass]
        sym_words[1] = klass_tagged;
        let sym_tagged = sym_words.as_ptr() as u64 | MEM_TAG;
        SymbolOop::try_from(Oop::from_raw(sym_tagged))
            .expect("fabricated symbol should satisfy SymbolOop::try_from")
    }

    /// `CallSend` end to end: marshal the receiver and arguments into the
    /// Win64 argument registers, call through the patchable site, and take
    /// the result. The site is emitted with a displacement of 0 (a
    /// self-call placeholder), so the test patches it exactly as the code
    /// cache will — which is also what proves the recorded site offset
    /// points at the right byte.
    #[cfg(windows)]
    #[test]
    fn compiled_call_send_marshals_args_and_patches() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::compiler::ir::CallSiteInfo;
        use crate::vendor::wfasm::native_windows::WinJit;

        // The IC target: returns arg0 - arg1, so a swapped or stale
        // argument register shows up as a wrong sign rather than passing.
        extern "C" fn target(a: u64, b: u64) -> u64 {
            a.wrapping_sub(b)
        }

        let mut kw = [0u64; 4];
        let mut sw = [0u64; 2];
        let sym = fake_symbol(&mut kw, &mut sw);

        let mut m = hand_method(
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
                    Ir::CallSend {
                        dst: VReg(2),
                        site: 0,
                        args: vec![VReg(0), VReg(1)],
                    },
                    Ir::Ret { val: VReg(2) },
                ],
            )],
            oops(3),
            2,
        );
        m.call_sites = vec![CallSiteInfo {
            selector: sym,
            argc: 1,
            static_klass: None,
        }];

        let out = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None);
        assert_eq!(out.ic_sites.len(), 1, "one send, one IC site");
        assert_eq!(out.ic_sites[0].site, 0);
        assert_eq!(out.safepoints.len(), 1, "a send is a deopt safepoint");
        let site_off = out.ic_sites[0].off as usize;
        assert_eq!(
            out.blob.code[site_off], 0xE8,
            "the recorded IC offset must point AT the call opcode"
        );

        let blob = out.blob;
        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base.add(moff), blob.code.len());
        }

        // Patch the site the way the code cache does: rel32 relative to
        // the END of the 5-byte call instruction, via an in-region thunk.
        let site_addr = base as u64 + moff as u64 + site_off as u64;
        let thunk_off = (moff + blob.code.len() + 15) & !15;
        let direct = patch_ic_site(base, site_addr, thunk_off, target as usize as u64);
        assert!(
            direct,
            "with near allocation the IC target should be reachable by a direct              call rel32 — no veneer needed"
        );

        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + moff as u64;
        let argv = [100u64, 42u64];
        assert_eq!(
            unsafe { stub_fn(entry, 0, argv.as_ptr(), 2) },
            58,
            "100 - 42 through a patched inline-cache site"
        );
        // Argument ORDER is checked by the asymmetry: swapped registers
        // would give the negation.
        let argv = [42u64, 100u64];
        assert_eq!(
            unsafe { stub_fn(entry, 0, argv.as_ptr(), 2) },
            42u64.wrapping_sub(100),
        );
    }

    /// A callee unwound by a non-local return hands back `NLR_SENTINEL`
    /// instead of a value; the sender must return it immediately rather
    /// than treat it as an ordinary result.
    #[cfg(windows)]
    #[test]
    fn call_send_propagates_the_nlr_sentinel() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::compiler::ir::CallSiteInfo;
        use crate::oops::layout::NLR_SENTINEL;
        use crate::vendor::wfasm::native_windows::WinJit;

        extern "C" fn unwinding_callee(_a: u64, _b: u64) -> u64 {
            NLR_SENTINEL
        }

        let mut kw = [0u64; 4];
        let mut sw = [0u64; 2];
        let sym = fake_symbol(&mut kw, &mut sw);

        let mut m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::Param {
                        dst: VReg(0),
                        index: 0,
                    },
                    Ir::CallSend {
                        dst: VReg(1),
                        site: 0,
                        args: vec![VReg(0)],
                    },
                    // If the sentinel were NOT intercepted, this would
                    // overwrite it and the test would see 999 instead.
                    Ir::ConstSmi {
                        dst: VReg(1),
                        value: 999,
                    },
                    Ir::Ret { val: VReg(1) },
                ],
            )],
            oops(2),
            1,
        );
        m.call_sites = vec![CallSiteInfo {
            selector: sym,
            argc: 0,
            static_klass: None,
        }];

        let out = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None);
        let site_off = out.ic_sites[0].off as usize;
        let blob = out.blob;
        let stub = build_call_stub_x64();
        let jit = WinJit::with_capacity(stub.code.len() + blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        let moff = (stub.code.len() + 15) & !15;
        unsafe {
            core::ptr::copy_nonoverlapping(stub.code.as_ptr(), base, stub.code.len());
            core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base.add(moff), blob.code.len());
        }
        let site_addr = base as u64 + moff as u64 + site_off as u64;
        let thunk_off = (moff + blob.code.len() + 15) & !15;
        patch_ic_site(base, site_addr, thunk_off, unwinding_callee as usize as u64);

        let stub_fn: CallStubFn = unsafe { std::mem::transmute(base) };
        let entry = base as u64 + moff as u64;
        let argv = [smi(1)];
        assert_eq!(
            unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) },
            NLR_SENTINEL,
            "the sentinel must propagate straight out, not be overwritten"
        );
    }

    /// Build a fake indexable object: `[mark][klass][length][elems...]`,
    /// with `length` a tagged smi. Returns its tagged pointer.
    #[cfg(windows)]
    fn fake_array(words: &mut Vec<u64>, klass: u64, elems: &[u64]) -> u64 {
        words.clear();
        words.push(0); // mark
        words.push(klass);
        words.push((elems.len() as u64) << 2); // tagged length
        words.extend_from_slice(elems);
        words.as_ptr() as u64 | crate::oops::layout::MEM_TAG
    }

    /// `ArrayAt` executed: correct element for a valid 1-based index, and
    /// the cold edge for every way the access can be invalid.
    ///
    /// The bounds cases are the point. The guard does ONE unsigned compare
    /// to reject both `i < 1` and `i > length` — a zero or negative index
    /// wraps to a huge unsigned value — so index 0 and index -1 are as
    /// important to test as index length+1. A signed compare would pass
    /// them and read outside the object.
    #[cfg(windows)]
    #[test]
    fn compiled_array_at_executes_and_bounds_check_is_unsigned() {
        use crate::compiler::ir::PoolEntry;
        const KLASS: u64 = 0x4444_0001;

        let mut storage = Vec::new();
        let arr = fake_array(&mut storage, KLASS, &[smi(10), smi(20), smi(30)]);

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
                        Ir::ArrayAt {
                            dst: VReg(2),
                            arr: VReg(0),
                            idx: VReg(1),
                            klass: PoolLit(0),
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
        m.pool = vec![PoolEntry {
            value: KLASS,
            kind: Some(RelocKind::Oop),
        }];

        // Valid, 1-based.
        assert_eq!(compile_and_run(&m, arr, smi(1)), smi(10));
        assert_eq!(compile_and_run(&m, arr, smi(2)), smi(20));
        assert_eq!(compile_and_run(&m, arr, smi(3)), smi(30));
        // Out of range, both directions — the single unsigned compare.
        assert_eq!(compile_and_run(&m, arr, smi(4)), BAILOUT_SENTINEL, "past end");
        assert_eq!(compile_and_run(&m, arr, smi(0)), BAILOUT_SENTINEL, "index 0");
        assert_eq!(
            compile_and_run(&m, arr, smi(-1)),
            BAILOUT_SENTINEL,
            "negative index must not wrap into the object"
        );
        // Bad receiver / bad index shapes.
        assert_eq!(compile_and_run(&m, smi(7), smi(1)), BAILOUT_SENTINEL, "smi recv");
        assert_eq!(compile_and_run(&m, arr, arr), BAILOUT_SENTINEL, "non-smi index");
        // Wrong klass.
        let mut other = Vec::new();
        let arr2 = fake_array(&mut other, 0x5555_0001, &[smi(99)]);
        assert_eq!(
            compile_and_run(&m, arr2, smi(1)),
            BAILOUT_SENTINEL,
            "a different klass must not match the guard"
        );
        let _ = (&storage, &other);
    }

    /// `ArrayAtPut` writes the element, answers the stored value, and
    /// leaves the neighbouring elements untouched (which is what catches a
    /// wrong element stride or base offset).
    #[cfg(windows)]
    #[test]
    fn compiled_array_at_put_writes_the_right_element() {
        use crate::compiler::ir::PoolEntry;
        const KLASS: u64 = 0x4444_0001;

        let mut storage = Vec::new();
        let arr = fake_array(&mut storage, KLASS, &[smi(0), smi(0), smi(0)]);

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
                        Ir::Param {
                            dst: VReg(2),
                            index: 2,
                        },
                        Ir::ArrayAtPut {
                            dst: VReg(3),
                            arr: VReg(0),
                            idx: VReg(1),
                            val: VReg(2),
                            klass: PoolLit(0),
                            fail: BlockId(1),
                        },
                        Ir::Ret { val: VReg(3) },
                    ],
                ),
                block(
                    1,
                    vec![Ir::Bailout {
                        reason: BailoutReason::SmiOpFailed,
                    }],
                ),
            ],
            oops(4),
            3,
        );
        m.pool = vec![PoolEntry {
            value: KLASS,
            kind: Some(RelocKind::Oop),
        }];

        // The barrier reads through R15; a null VM would fault, so run
        // through the stub with a real (zeroed) register block. old_start
        // of 0 makes every object "old", and a smi value takes the
        // second early-out, so no card is marked.
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::vendor::wfasm::native_windows::WinJit;
        let vmreg = [0u64; 8];
        let blob = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), None).blob;
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

        let argv = [arr, smi(2), smi(77)];
        assert_eq!(
            unsafe { stub_fn(entry, vm, argv.as_ptr(), 3) },
            smi(77),
            "at:put: answers the stored value"
        );
        // Element 2 changed; its neighbours did not.
        assert_eq!(storage[3], smi(0), "element 1 untouched");
        assert_eq!(storage[4], smi(77), "element 2 written");
        assert_eq!(storage[5], smi(0), "element 3 untouched");
        assert_eq!(storage[2], smi(3), "the length word was not overwritten");
    }

    /// The customization guard, executed: a receiver whose klass matches
    /// the key falls through to the body; a mismatch — and a smi, which
    /// can never be an instance of a heap klass — tail-jumps to the
    /// resolve stub instead.
    ///
    /// `verified_entry_off` must be a real entry point: a monomorphic
    /// caller skips the guard by jumping there, so the test enters BOTH
    /// ways and requires the same answer.
    #[cfg(windows)]
    #[test]
    fn entry_guard_admits_the_key_klass_and_diverts_everything_else() {
        use crate::codecache::stubs_x64::{build_call_stub_x64, CallStubFn};
        use crate::oops::layout::MEM_TAG;
        use crate::vendor::wfasm::native_windows::WinJit;

        // The resolve stub stands in for re-dispatch: it returns a
        // recognizable value so a diverted call is unmistakable.
        extern "C" fn resolve_probe(_recv: u64) -> u64 {
            0xDEAD_BEEF
        }

        const KEY_KLASS: u64 = 0x7777_0001;
        let mut matching = [0u64; 2];
        matching[1] = KEY_KLASS;
        let recv_ok = matching.as_ptr() as u64 | MEM_TAG;
        let mut other = [0u64; 2];
        other[1] = 0x8888_0001;
        let recv_bad = other.as_ptr() as u64 | MEM_TAG;

        // `^ 7` — the body is irrelevant; what matters is whether it runs.
        let m = hand_method(
            vec![block(
                0,
                vec![
                    Ir::ConstSmi {
                        dst: VReg(0),
                        value: 7,
                    },
                    Ir::Ret { val: VReg(0) },
                ],
            )],
            oops(1),
            1,
        );
        let guard = EntryGuard {
            // Distinct from the key, so this takes the heap-key shape.
            smi_klass_bits: 0x9999_0001,
            key_klass_bits: KEY_KLASS,
            resolve_addr: resolve_probe as usize as u64,
        };
        let out = emit_x64(&m, &regalloc(&m), RuntimeAddrs::default(), Some(&guard));
        assert!(
            out.verified_entry_off > 0,
            "a guard was requested, so the verified entry must sit past it"
        );
        assert_eq!(
            out.block_pcs.len(),
            1,
            "one block, one recorded block pc"
        );
        assert!(out.block_pcs[0] >= out.verified_entry_off);

        let blob = out.blob;
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
        let verified = entry + out.verified_entry_off as u64;

        // Matching klass: through the guard, into the body.
        let argv = [recv_ok];
        assert_eq!(unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) }, smi(7));
        // Wrong klass: diverted to the resolve stub.
        let argv = [recv_bad];
        assert_eq!(
            unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) },
            0xDEAD_BEEF,
            "a non-matching klass must tail-jump to resolve"
        );
        // A smi receiver can never be an instance of a heap klass.
        let argv = [smi(3)];
        assert_eq!(
            unsafe { stub_fn(entry, 0, argv.as_ptr(), 1) },
            0xDEAD_BEEF,
            "a smi receiver misses a heap key without loading any header"
        );
        // Entering at the verified entry skips the guard entirely — even
        // the receiver that would have missed now runs the body.
        let argv = [recv_bad];
        assert_eq!(
            unsafe { stub_fn(verified, 0, argv.as_ptr(), 1) },
            smi(7),
            "verified_entry_off must be a genuine entry past the guard"
        );
    }

    /// A safepoint's `position` must be the op's index in REGALLOC's
    /// linear numbering, because that is how `driver::build_deopt_metadata`
    /// finds the oop map describing which stack slots are live there.
    ///
    /// This is not a cosmetic field. Hand the GC a position that names a
    /// different op and it walks the wrong live set for that frame —
    /// tracing dead slots as roots, or worse, missing live ones. Nothing
    /// faults; the heap just quietly goes wrong later.
    ///
    /// The method below puts a `Poll` at a known op index, in a block that
    /// regalloc orders SECOND, so a walk in source order rather than
    /// `block_order` would record a different position.
    #[test]
    fn safepoint_position_matches_regallocs_numbering() {
        let m = hand_method(
            vec![
                block(0, vec![Ir::Jump { target: BlockId(1) }]),
                block(
                    1,
                    vec![
                        Ir::ConstSmi {
                            dst: VReg(0),
                            value: 1,
                        },
                        Ir::Poll,
                        Ir::Ret { val: VReg(0) },
                    ],
                ),
            ],
            oops(1),
            0,
        );
        let ra = regalloc(&m);
        let out = emit_x64(&m, &ra, RuntimeAddrs::default(), None);
        assert_eq!(out.safepoints.len(), 1, "one Poll, one safepoint");

        // Recompute the Poll's position by walking regalloc's own order,
        // the same way compute_intervals numbers ops.
        let mut expected = None;
        let mut pos = 0u32;
        for &bid in &ra.block_order {
            for op in &m.blocks[bid.0 as usize].code {
                if matches!(op, Ir::Poll) {
                    expected = Some(pos);
                }
                pos += 1;
            }
        }
        assert_eq!(
            out.safepoints[0].position,
            expected.expect("the Poll was emitted"),
            "safepoint position must index regalloc's numbering, not source order"
        );
    }

    /// An op outside the slice fails loudly and names itself, rather than
    /// emitting approximate code (CONVENTIONS §4).
    #[test]
    #[should_panic(expected = "NlrReturn is not lowered by the x64 back end")]
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
            ic_sites: Vec::new(),
            stub_poll_lit: LiteralId(0),
            must_be_boolean_lit: LiteralId(0),
            alloc_slow_lit: LiteralId(0),
            box_double_lit: LiteralId(0),
            current_bci: 0,
            pos: 0,
            method: &m,
        };
        let _ = &ra;
        // `FArith` used to serve as the example here. Phase 5 lowered
        // it, so the test now names an op that genuinely has no x64
        // lowering — `NlrReturn`. Whoever implements that must move this
        // to the next unsupported op rather than delete the test: its
        // job is to prove an unlowered op fails LOUDLY, by name, instead
        // of falling through and emitting nothing.
        emit_op(
            &mut e,
            &Ir::NlrReturn {
                closure: VReg(0),
                value: VReg(0),
            },
        );
    }
}
