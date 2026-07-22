//! SSA-lite IR + bytecode-to-IR conversion (`sprint_s10_detail.md` D2, D3.2,
//! D3.3). Consumes a [`crate::compiler::decode::Cfg`] and a method's real
//! bytecode/literals/ICs and produces an [`IrMethod`] already shaped for
//! `regalloc.rs`'s linear scan and `emit.rs`'s direct per-op AArch64
//! sequences — D3.3's own call: "the SSA-lite IR is machine-shaped and
//! emit walks it directly," no separate lowering stage in v1.
//!
//! "SSA-lite": every value-producing bytecode effect gets its own fresh
//! [`VReg`], but the per-method "unified temp/arg slot" vregs
//! (`self_vreg`/`temp_vregs`) allow multiple defs (a real `store_temp`
//! reassigns the SAME vreg) — not textbook SSA, cheap enough for a tier-1
//! compiler that never runs SSA-only optimizations.

use std::collections::HashMap;

use crate::bytecode::opcode::{decode_at, Instr};
use crate::compiler::assembler::RelocKind;
use crate::compiler::decode::{BlockIndex, Cfg, Terminator};
use crate::compiler::scopes::SafepointKind;
use crate::interpreter::ic::InterpreterIc;
use crate::oops::layout::BODY_OFFSET;
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::runtime::vm_state::VmState;

/// `primitives.rs`'s pinned id for `basicNew` (`Object>>basicNew`,
/// `<primitive: 23>`) — the target an inline-allocatable `X basicNew` site
/// must resolve to (`Translator::alloc_site_klass`, S11 D7). `pub(crate)`:
/// `driver::mono_smi_inline_send` reads it too, to let a mono basicNew site
/// clear eligibility (both readers must agree on the same id).
pub(crate) const PRIM_BASIC_NEW: i64 = 23;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VReg(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockId(pub u32);
/// Id into an [`IrMethod`]'s own `pool` — a compiler-stage literal table,
/// distinct from (and translated into, at emit time) the vendored
/// assembler's own `LiteralId`/literal pool (S9).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolLit(pub u32);

/// Identifies a runtime stub `Ir::CallRuntime` calls into — provisional
/// here; owned for real by `codecache::stubs` once S11 actually emits
/// `CallRuntime`. S10 never constructs one (`stub_poll`, the one S10 stub,
/// is called directly from emitted `Poll` sequences, not through this Ir).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StubId(pub u32);

impl StubId {
    /// D1/D5: `codecache::stubs::Stubs::must_be_boolean` (S11 step 6) —
    /// the only `CallRuntime` target step 7 wires up; `emit.rs`'s own
    /// `Ir::CallRuntime` dispatch asserts this is the only value it ever
    /// sees, since `rt_alloc_slow`/`stub_nlr_unwind` (steps 8/9) don't
    /// exist as `CallRuntime` targets yet either.
    pub const MUST_BE_BOOLEAN: StubId = StubId(0);
}

/// A position `regalloc.rs`'s `linearize` (S11+) records as needing an
/// oop map. Always empty in S10 (D1: compiled code never allocates or
/// calls Rust, so it has no safepoints) — the field exists now so S11 only
/// adds population, not plumbing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SafepointId(pub u32);

pub struct VRegInfo {
    pub is_oop: bool,
    /// Float fast-path (`docs/float_fastpath_design.md`): this vreg holds an
    /// UNBOXED `f64` and must be allocated to the FP register file (`d0`–`d31`),
    /// independently of the x0–x15 GPR linear-scan. An fp vreg is never an oop
    /// (`is_oop == false`), so the GC root walker/oop-map ignore it.
    pub is_fp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmiOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

/// Float fast-path (`docs/float_fastpath_design.md`) arithmetic op — the
/// `Double` primitives 100–103, lowered to native `fadd`/`fsub`/`fmul`/`fdiv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// SIMD (`docs/SIMD.md`): which vector class an `Ir::VecArith` operates on —
/// selects the guard klass, the NEON arrangement (`.2d` vs `.4s`), and the box
/// stub. Both are 16-byte raw bodies (`Format::Double`), so ONLY these three
/// choices differ; the guard/unbox/box skeleton is identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VecKind {
    /// `Float64x2` — two f64 lanes, float `fadd/… v.2d`.
    F64x2,
    /// `Float32x4` — four f32 lanes, float `fadd/… v.4s`.
    F32x4,
    /// `Int32x4` — four i32 lanes, INTEGER `add/sub/mul v.4s` (no vector
    /// divide). The arrangement token is `.4s` like `F32x4`; the OPCODE
    /// (`add` vs `fadd`) is what distinguishes integer from float.
    I32x4,
}

impl VecKind {
    /// True for the integer lane types — selects integer NEON opcodes
    /// (`add`/`sub`/`mul`) over the floating (`fadd`/`fsub`/`fmul`/`fdiv`).
    pub fn is_int(self) -> bool {
        matches!(self, VecKind::I32x4)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BailoutReason {
    SmiOpFailed,
}

/// S14 step 4b: which receiver-klass test an [`Ir::GuardKlass`] lowers to
/// (mirrors `inline::GuardKind`'s `SmiTest`/`KlassTest`, minus its `None` — a
/// `None` guard emits no `GuardKlass` op at all). Kept as its own tiny enum so
/// `emit.rs` needn't depend on `compiler::inline`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardShape {
    /// `tst rcvr, #3; b.ne cold` — the receiver must be a smi (2-bit tag 00).
    SmiTest,
    /// reject-smi, load the klass word, compare against the expected klass
    /// literal, `b.ne cold` — the receiver must be a heap oop of that klass.
    KlassTest,
}

/// One compiler-stage literal-pool entry (D2) — `emit.rs` walks these at
/// `finish` time and calls `Assembler::literal_u64(value, kind)` for each,
/// getting back the vendored assembler's own `LiteralId`.
#[derive(Clone, Copy, Debug)]
pub struct PoolEntry {
    pub value: u64,
    pub kind: Option<RelocKind>,
}

/// One IR instruction. 19 variants total; `CallSend`/`CallRuntime`/`Alloc`/
/// `GuardKlass`/`StoreField` (and `LoadKlass`, needed by none of D3.2's
/// translation rows or D5.3's emit table either) are defined here for the
/// C+D-phase enum shape but never *constructed* by S10's `convert` —
/// they're S11's to emit.
#[derive(Debug)]
pub enum Ir {
    ConstSmi {
        dst: VReg,
        value: i64,
    },
    ConstPool {
        dst: VReg,
        lit: PoolLit,
    },
    Move {
        dst: VReg,
        src: VReg,
    },
    /// `index` 0 = receiver, 1.. = args (SPEC §5.1 unified numbering).
    /// Entry block only.
    Param {
        dst: VReg,
        index: u8,
    },
    LoadKlass {
        dst: VReg,
        obj: VReg,
    },
    /// `byte_off` is untagged-address-relative; `emit.rs` applies the −1
    /// tagged-pointer bias itself (D5.3 P4).
    LoadField {
        dst: VReg,
        obj: VReg,
        byte_off: i32,
    },
    StoreField {
        obj: VReg,
        byte_off: i32,
        val: VReg,
        barrier: bool,
    },
    /// Tag check + op + overflow check fused into one emit sequence
    /// (D5.3); `fail` is always the method's one shared bailout block.
    SmiArith {
        op: SmiOp,
        dst: VReg,
        a: VReg,
        b: VReg,
        fail: BlockId,
    },
    /// S15 step 7 (the sieve fix): Array element READ intrinsified — the
    /// same treatment smi arithmetic got in S14. Inline guards (receiver
    /// mem-tagged AND klass == Array, index smi, 1-based bounds vs the
    /// tagged size slot) then a single load; `fail` is a reexecute-true
    /// trap edge that re-runs the send interpreted (prim-fail → the
    /// Smalltalk error path, byte-identical semantics).
    ArrayAt {
        dst: VReg,
        arr: VReg,
        idx: VReg,
        /// Pool literal holding the Array klass oop (GC keeps it current).
        klass: PoolLit,
        fail: BlockId,
    },
    /// S15: Array element WRITE intrinsified — guards as `ArrayAt`, then a
    /// single store + the same young-into-old card barrier `StoreField`
    /// emits. Produces the stored value (the send's own result).
    ArrayAtPut {
        dst: VReg,
        arr: VReg,
        idx: VReg,
        val: VReg,
        /// Pool literal holding the Array klass oop.
        klass: PoolLit,
        fail: BlockId,
    },
    /// The fusion peephole (D3.2): a comparison send whose ONLY consumer
    /// is an immediately-following `br_true_fwd`/`br_false_fwd`.
    SmiCmpBr {
        op: CmpOp,
        a: VReg,
        b: VReg,
        if_true: BlockId,
        if_false: BlockId,
        fail: BlockId,
    },
    /// A comparison send NOT fused with a branch — materializes
    /// `true`/`false`.
    SmiCmpVal {
        op: CmpOp,
        dst: VReg,
        a: VReg,
        b: VReg,
        fail: BlockId,
    },
    /// Identity comparison — `==` (`neq` for `~~`). The frontend PINS
    /// identity as non-redefinable (codegen.rs's nil-guard family already
    /// compiles `x == nil` relying on exactly that), so this lowers to a
    /// raw pointer compare selecting `true`/`false`: NO guard, NO fail
    /// edge, sound for every operand pair. The send census that motivated
    /// it: richards spends ~130k sends per run dispatching `==` to the
    /// identity primitive.
    RefCmpVal {
        dst: VReg,
        a: VReg,
        b: VReg,
        /// `~~`: select the other way around.
        neq: bool,
    },
    /// Speculated boolean negation — `not` on evidence-boolean sites.
    /// GUARDED: `src` must be the `true` or `false` object, anything else
    /// branches to `fail` (a reexecute trap re-sending #not generically —
    /// correct lookup for any receiver). Lowered only when the LIVE
    /// `True>>not`/`False>>not` are the canonical single-op flips, pinned
    /// by inline deps so a redefinition invalidates the nmethod.
    BoolNot {
        dst: VReg,
        src: VReg,
        fail: BlockId,
    },
    /// ── Float fast-path (`docs/float_fastpath_design.md`) ──────────────────
    /// Unbox a boxed `Double` oop into an FP vreg. GUARDED: `src` must be a
    /// Double, else branch to `fail` (a cold `UncommonTrap` re-executing the
    /// send at tier-0, exactly like `SmiArith`'s fail). `dst` is an fp vreg.
    FUnbox {
        dst: VReg,
        src: VReg,
        fail: BlockId,
    },
    /// Box an FP vreg into a freshly-allocated `Double` oop. The only heap
    /// traffic on the float path; a regalloc SAFEPOINT (`Alloc`-shaped). `src`
    /// is an fp vreg, `dst` an ordinary oop vreg.
    FBox {
        dst: VReg,
        src: VReg,
    },
    /// Native FP arithmetic on two fp vregs → an fp vreg. No guard, no alloc,
    /// no send: `fadd`/`fsub`/`fmul`/`fdiv`.
    FArith {
        op: FArithOp,
        dst: VReg,
        a: VReg,
        b: VReg,
    },
    /// Float comparison fused with a branch (the `FCmpBr` analog of
    /// `SmiCmpBr`): `fcmp` + `b.cond`. `a`/`b` are fp vregs.
    FCmpBr {
        op: CmpOp,
        a: VReg,
        b: VReg,
        if_true: BlockId,
        if_false: BlockId,
    },
    /// Float comparison NOT fused with a branch — materializes `true`/`false`
    /// into an oop vreg `dst`. `a`/`b` are fp vregs.
    FCmpVal {
        op: CmpOp,
        dst: VReg,
        a: VReg,
        b: VReg,
    },
    /// An unboxed `f64` constant (the float-temp promotion's rewrite of a
    /// `Move` from a boxed Double literal — `zr := 0.0`). `bits` is the raw
    /// bit pattern, baked into code (`movz/movk` + `fmov`): the VALUE of an
    /// immutable Double literal never changes even when the boxed object
    /// moves, so no pool/reloc is involved at all.
    FConst {
        dst: VReg,
        bits: u64,
    },
    /// ── SIMD NEON fast-path (`docs/SIMD.md`) ───────────────────────────────
    /// A fused mono-vector elementwise arithmetic send (`+`/`-`/`*`/`/`) — the
    /// vector analog of `FUnbox`+`FArith`+`FBox` collapsed into ONE op. `kind`
    /// selects the vector class (`Float64x2` → `.2d`, `Float32x4` → `.4s`);
    /// `a`/`b`/`dst` are ordinary OOP vregs (the boxed vector). emit.rs lowers
    /// the whole guard→unbox (`ldr q`)→`f… v.<arr>`→box internally with fixed
    /// scratch `q`-registers — NO vector vregs, so the allocator/reducer are
    /// untouched (v1 box-per-op; the reducer/q-pool generalization is later).
    /// GUARDED like `FUnbox`: an operand of the wrong klass branches to `fail`
    /// (a cold `UncommonTrap` re-executing the send at tier-0 — the primitive's
    /// own Smalltalk fallback, byte-identical). Allocates the result box, so it
    /// is a regalloc SAFEPOINT (`is_safepoint_op`).
    VecArith {
        kind: VecKind,
        op: FArithOp,
        dst: VReg,
        a: VReg,
        b: VReg,
        fail: BlockId,
    },
    Jump {
        target: BlockId,
    },
    BoolBr {
        val: VReg,
        if_true: BlockId,
        if_false: BlockId,
        not_bool: BlockId,
    },
    /// S14 step 4b: a receiver-klass guard in front of an inlined leaf body.
    /// On a MATCH, control falls through to the inlined body; on a MISMATCH it
    /// branches to `fail` — a cold block that is a single `Ir::UncommonTrap`
    /// (reexecute=true at the inlined send's own bci), so a wrong receiver
    /// deopts and re-executes the send generically (S14 step-3 trap shape).
    /// `kind` selects the test emit.rs lowers: `SmiTest` (`tst rcvr,#3; b.ne`)
    /// when the speculated klass is the smi klass, `KlassTest`
    /// (reject-smi-then-compare-klass-word against `expect`) otherwise.
    /// `expect` is the expected klassOop, interned as a `RelocKind::Oop` pool
    /// literal (a moving GC keeps it current) — read only by `KlassTest`;
    /// `SmiTest` ignores it. This op is NOT itself a regalloc safepoint (it is
    /// a pure test); the safepoint that spills the reexecute stack is the
    /// `UncommonTrap` in the `fail` block.
    GuardKlass {
        obj: VReg,
        expect: PoolLit,
        fail: BlockId,
        kind: GuardShape,
    },
    CallSend {
        dst: VReg,
        site: u16,
        args: Vec<VReg>,
    },
    CallRuntime {
        dst: Option<VReg>,
        stub: StubId,
        args: Vec<VReg>,
    },
    /// S11 D7 inline allocation of a fixed-size `Format::Slots` object
    /// (`X new` where `X` is a compile-time-known class constant). `emit.rs`
    /// lowers this to a SELF-CONTAINED fast-path-plus-slow-call sequence
    /// (bump `reg_block.eden_top`, bounds-check `eden_end`, init header +
    /// nil body, mem-tag; on overflow `bl stub_alloc_slow`) with its own
    /// internal labels — no separate slow CFG block, mirroring how
    /// `StoreField`'s barrier is self-contained. Still a regalloc SAFEPOINT
    /// (`is_safepoint`), so every vreg live across it is already spilled
    /// before the internal slow call's `bl` can clobber a caller-saved
    /// register. `klass` is the class oop (`RelocKind::Oop` pool entry, read
    /// from the `push_global` class constant at the send site); `size_words`
    /// is `klass.non_indexable_size()`, fixed at compile time (guarded by
    /// S13 deps — until then a klass-format redefinition is a documented
    /// stale-code hole flushed by S12's sweep, same class of hole as
    /// `stale_mono_documented_hole`).
    Alloc {
        dst: VReg,
        klass: PoolLit,
        size_words: u32,
    },
    /// Loop back-edge flag check (D5.3); reads `VmRegBlock::poll_flag`.
    Poll,
    /// S13 step 7b (D3, first "organic trap client"): an uncommon-trap
    /// terminator — control leaves the compiled method entirely via a
    /// `brk #0xDE00` (`emit.rs` lowers it), so it has NO `dst` and NO fall-
    /// through. `emit.rs`'s lowering records ITS OWN offset as a safepoint pc
    /// (a trap keys on the brk instruction itself, not a return address); the
    /// SIGTRAP handler → `rt_uncommon_trap` deopts the frame and resumes the
    /// re-executing send in the interpreter. `bci` is the resume point (the
    /// failing send's own bci) — carried here so `driver::build_deopt_metadata`
    /// and the emitted `SafepointPc` agree with the recorded `DeoptRaw.bci`.
    /// It is a regalloc SAFEPOINT (see `regalloc::is_safepoint`), which is what
    /// spills every oop live across it (the re-executing send's `a`/`b`/`self`)
    /// into the frame so the deopt materializer can read them.
    UncommonTrap {
        bci: usize,
    },
    Ret {
        val: VReg,
    },
    RetSelf,
    /// S24 A1 (closure compilation, design §2.4): a compiled BLOCK body's
    /// `nlr_tos` — park the non-local return via
    /// `codecache::stubs::rt_nlr_originate` (x0 = `closure`, x1 = `value`),
    /// then return `NLR_SENTINEL` through the block's own epilogue; every
    /// compiled frame above relays via its existing per-site NLR check
    /// (S11 step 9), and the first interpreter boundary resumes
    /// `continue_unwind`. A terminator, like `Ret` — only ever produced by
    /// a block compilation (`convert`'s `is_block` mode; a METHOD's `^` is
    /// `return_tos`, never `nlr_tos`).
    NlrReturn {
        closure: VReg,
        value: VReg,
    },
    Bailout {
        reason: BailoutReason,
    },
}

impl Ir {
    /// Every vreg this op reads. Regalloc liveness and the (future) S12
    /// verifier both consume this — a new variant that forgets an operand
    /// here is caught in one place (D2).
    pub fn uses(&self, mut f: impl FnMut(VReg)) {
        match self {
            Ir::Move { src, .. } => f(*src),
            Ir::LoadKlass { obj, .. } => f(*obj),
            Ir::LoadField { obj, .. } => f(*obj),
            Ir::StoreField { obj, val, .. } => {
                f(*obj);
                f(*val);
            }
            Ir::SmiArith { a, b, .. }
            | Ir::SmiCmpBr { a, b, .. }
            | Ir::SmiCmpVal { a, b, .. }
            | Ir::RefCmpVal { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Ir::BoolNot { src, .. } => f(*src),
            Ir::FArith { a, b, .. } | Ir::FCmpBr { a, b, .. } | Ir::FCmpVal { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Ir::FUnbox { src, .. } | Ir::FBox { src, .. } => f(*src),
            Ir::VecArith { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Ir::ArrayAt { arr, idx, .. } => {
                f(*arr);
                f(*idx);
            }
            Ir::ArrayAtPut { arr, idx, val, .. } => {
                f(*arr);
                f(*idx);
                f(*val);
            }
            Ir::BoolBr { val, .. } => f(*val),
            Ir::GuardKlass { obj, .. } => f(*obj),
            Ir::CallSend { args, .. } | Ir::CallRuntime { args, .. } => {
                for &v in args {
                    f(v);
                }
            }
            Ir::Ret { val } => f(*val),
            Ir::NlrReturn { closure, value } => {
                f(*closure);
                f(*value);
            }
            // `self` is always `VReg(0)` by this module's own construction
            // (`convert()`'s `let self_vreg = VReg(0);` below) -- RetSelf
            // reads it exactly like `Ret{val}` reads `val`. Omitting this
            // use let regalloc's `compute_intervals` treat self as
            // dead-on-arrival (a trivial `[def,def]` interval) whenever a
            // later argc param's own `Param` vreg outlived it and got
            // handed the same register on retire -- emit.rs's sequential
            // (not parallel) ABI-register shuffle would then clobber that
            // register before `RetSelf`'s own "nothing to do" handler ever
            // ran, so a compiled `^self` could return the wrong value
            // whenever an explicit argument went unused (found via
            // `compiled_entry_stack_discipline_across_argc`).
            Ir::RetSelf => f(VReg(0)),
            Ir::ConstSmi { .. }
            | Ir::ConstPool { .. }
            | Ir::FConst { .. }
            | Ir::Param { .. }
            | Ir::Jump { .. }
            | Ir::Alloc { .. }
            | Ir::Poll
            // S13 step 7b: an UncommonTrap reads NO vreg directly — the
            // re-executing send's inputs (`a`/`b`/`self`) are kept live
            // across it by the fail block's own `DeoptRaw.stack`, not by an
            // operand of this op, so its liveness comes from `deopt_sites`,
            // not from `uses` (mirrors how a `CallSend`'s recorded stack, not
            // its `args`, is what a deopt reads).
            | Ir::UncommonTrap { .. }
            | Ir::Bailout { .. } => {}
        }
    }

    pub fn defs(&self, mut f: impl FnMut(VReg)) {
        match self {
            Ir::ConstSmi { dst, .. }
            | Ir::ConstPool { dst, .. }
            | Ir::Move { dst, .. }
            | Ir::Param { dst, .. }
            | Ir::LoadKlass { dst, .. }
            | Ir::LoadField { dst, .. }
            | Ir::SmiArith { dst, .. }
            | Ir::ArrayAt { dst, .. }
            | Ir::ArrayAtPut { dst, .. }
            | Ir::SmiCmpVal { dst, .. }
            | Ir::RefCmpVal { dst, .. }
            | Ir::BoolNot { dst, .. }
            | Ir::CallSend { dst, .. }
            | Ir::FUnbox { dst, .. }
            | Ir::FBox { dst, .. }
            | Ir::FArith { dst, .. }
            | Ir::FCmpVal { dst, .. }
            | Ir::FConst { dst, .. }
            | Ir::VecArith { dst, .. }
            | Ir::Alloc { dst, .. } => f(*dst),
            Ir::CallRuntime { dst: Some(d), .. } => f(*d),
            Ir::StoreField { .. }
            | Ir::SmiCmpBr { .. }
            | Ir::FCmpBr { .. }
            | Ir::Jump { .. }
            | Ir::BoolBr { .. }
            | Ir::GuardKlass { .. }
            | Ir::CallRuntime { dst: None, .. }
            | Ir::Poll
            | Ir::UncommonTrap { .. }
            | Ir::Ret { .. }
            | Ir::RetSelf
            | Ir::NlrReturn { .. }
            | Ir::Bailout { .. } => {}
        }
    }
}

pub struct IrBlock {
    pub id: BlockId,
    /// First bytecode index this block covers — a `PcDesc` seed. The one
    /// synthetic block (the shared bailout target) uses 0: a bailout
    /// restarts interpretation from bci 0 (D1), so "back to the start" is
    /// the closest thing it has to a real bci.
    pub bci: usize,
    pub code: Vec<Ir>,
    /// Merge vregs (D3.3) this block's predecessor(s) must feed on entry.
    pub entry_stack: Vec<VReg>,
    /// S13 step 3b: the deopt safepoints WITHIN this block, keyed by the
    /// `code` index of the op that emits the safepoint (a `CallSend`/`Alloc`)
    /// — the converter records one here for each of the 3 clean safepoint
    /// sites (main + super `CallSend`, `Alloc`), carrying the abstract
    /// operand stack captured at that point plus its bci/kind/reexecute.
    /// `driver.rs` re-walks `block_order` to correlate each entry's op
    /// position with the emitted [`crate::compiler::emit::SafepointPc`]
    /// (hence its return-address `pc_off`) and resolves every `VReg` to a
    /// [`crate::compiler::scopes::ValueLoc`]. The synthesized smi-fallback /
    /// not-boolean blocks record NONE (their safepoints can't deopt until
    /// step 7), so this is empty for them — every safepoint-emitting op
    /// without an entry here is simply skipped by the driver.
    pub deopt_sites: Vec<(u32, DeoptRaw)>,
}

/// S13 step 3b: one deopt-recordable safepoint, as the converter sees it —
/// the abstract operand stack (`stack`, a list of live `VReg`s below the
/// safepoint), the resume `bci`, its [`SafepointKind`], and `reexecute`
/// (THE source of truth for stack height: `false` = a call-return site
/// whose recorded stack excludes the popped receiver+args and the
/// materializer pushes the result; `true` = re-execute at `bci` with all
/// inputs still on the recorded stack). `driver.rs` maps each `VReg` to a
/// [`crate::compiler::scopes::ValueLoc`] via `resolve_frame_loc` once the
/// regalloc position is known.
#[derive(Clone, Debug)]
pub struct DeoptRaw {
    pub stack: Vec<VReg>,
    pub bci: usize,
    pub kind: SafepointKind,
    pub reexecute: bool,
    /// S14 step 7-IV-c: sparse overrides — `(index into stack, block MethodOop
    /// pool ix)` for stack entries holding an ELIDED-CLOSURE phantom (its vreg
    /// is an unread filler; the materializer allocates the real closure via
    /// `ValueLoc::ElidedClosure`). Empty except at sites recorded while a
    /// phantom rides the operand stack (a block-arg send's guard-cold
    /// reexecute stack; in-callee sites with the phantom below a send's
    /// operands).
    pub stack_closures: Vec<(u16, u32)>,
    /// S14 step 4c: `Some` iff this safepoint lives INSIDE an inlined callee's
    /// body (a `CallSend`/trap the non-leaf splicer emitted). The driver reads
    /// it to record a NESTED deopt scope — the inlined callee's own receiver/
    /// slots/method + a `SenderLink` back to the caller — instead of the root
    /// method's depth-1 scope. `None` for every root-method safepoint (S13's
    /// exact behaviour, `sender: None`, so no regression).
    pub inline: Option<InlineSite>,
}

/// S14 step 4c: the per-safepoint description of an inlined callee's virtual
/// frame — everything `driver::build_deopt_metadata` needs to record a nested
/// deopt scope for a safepoint that lives inside a spliced non-leaf body. Every
/// `VReg` here resolves to a canonical [`crate::compiler::scopes::ValueLoc::
/// FrameSlot`] at the site's position (S12 spill-all), exactly like the root
/// scope's own vregs — so the depth-2 chain reuses the depth-1 location format
/// with zero new work.
#[derive(Clone, Debug)]
pub struct InlineSite {
    /// The INLINED callee's own compiled `MethodOop` pool index (distinct from
    /// the root `IrMethod.method_pool_ix`) — the reconstructed inlined frame's
    /// `FRAME_METHOD`.
    pub method_pool_ix: u32,
    /// The callee's `self` vreg (the inlined send's receiver operand).
    pub receiver: VReg,
    /// The callee's unified arg/temp slot vregs (`0..argc+ntemps`) — args alias
    /// the send's operand vregs, temps are the fresh nil vregs the splice mints.
    pub slots: Vec<VReg>,
    /// bci of the inlined send in the CALLER (the root method) — the
    /// `SenderLink.sender_bci`; the caller frame resumes at `sender_bci +
    /// len(send)`.
    pub sender_bci: u16,
    /// The CALLER's operand stack BELOW the inlined send's receiver+args, frozen
    /// for the whole inlined extent (the `SenderLink.pending_stack`).
    pub caller_pending_stack: Vec<VReg>,
    /// S14 step 7-I: `true` iff the inlined callee is a spliced literal BLOCK
    /// (its reconstructed scope is an `is_block` scope — the deopt materializer
    /// pushes a `CompiledBlock` activation frame, not a method frame). `false`
    /// for a step-4c method inline. `driver::build_deopt_metadata` stamps the
    /// inlined scope's `ScopeDescData.is_block` from this.
    pub is_block: bool,
    /// S14 step 7-IV-b: the ENCLOSING inline level, when this callee was itself
    /// spliced inside another inlined extent (a block spliced inside an inlined
    /// `do:`-style callee — depth 3: block ← callee ← root). `None` = this
    /// callee's caller IS the root method (depth 2, every pre-7-IV shape). For
    /// a `parent` level, `sender_bci`/`caller_pending_stack` describe the send
    /// in THAT level's method, not the root. `build_deopt_metadata` walks the
    /// chain outermost-first when beginning scopes.
    pub parent: Option<Box<InlineSite>>,
    /// S14 step 7-IV-c: sparse overrides — `(unified slot index, block
    /// MethodOop pool ix)` for slots holding an ELIDED-CLOSURE phantom (a
    /// `do:`-style callee's block-arg temp). The driver emits
    /// `ValueLoc::ElidedClosure` for these instead of resolving the (filler)
    /// vreg. Shared by every safepoint of the extent via the proto.
    pub slot_closures: Vec<(u16, u32)>,
}

/// S13 step 7b: the deopt info a fused comparison (`SmiCmpBr`) carries from
/// its own smi-inline site (in `translate_instr`) forward to the block's
/// Branch terminator (in `convert`), where the fail block is actually built.
/// The fail block's reexecute deopt must resume at the SEND's own `bci` with
/// `stack` (= `[below..., a, b]`, captured before the send's operand pops)
/// still present — neither of which is recoverable at the terminator (`bci`
/// there is the block start, and `a`/`b` were already popped), so both ride
/// along here.
#[derive(Clone, Debug)]
struct PendingCmpDeopt {
    bci: usize,
    stack: Vec<VReg>,
}

/// A comparison send deferred to its block's `Branch` terminator, where the
/// fused compare-and-branch is built (`convert`). The variant records which
/// inline path the comparison took, because the two fused ops differ in what
/// rides along:
///
///   - `Smi` is fallible (a non-smi operand, per the IC's speculation), so the
///     terminator must build its fail block from the deopt info captured back
///     at the send — see [`PendingCmpDeopt`].
///   - `Float` is not. Its operands were already guard-unboxed, and `FUnbox`'s
///     own `fail` edge owns the wrong-klass trap, so `fcmp` itself cannot fail
///     — hence `Ir::FCmpBr` carries no `fail`, and nothing needs to ride along.
///
/// Both variants share the payoff: a fused compare never materializes its
/// result, so there is no Boolean object to round-trip through, and no
/// `not_bool` edge either (a branch on the hardware flags cannot find a
/// non-Boolean where it expected one).
#[derive(Clone, Debug)]
enum PendingCmp {
    Smi {
        op: CmpOp,
        a: VReg,
        b: VReg,
        deopt: PendingCmpDeopt,
    },
    Float {
        op: CmpOp,
        a: VReg,
        b: VReg,
    },
}

/// S14 step 4c: the result of [`Translator::try_inline_nonleaf`] — either the
/// body ran to its return (`Value`), or an in-body cold send became a
/// terminating uncommon trap (`Trapped`), so the caller must finish the current
/// block here (nothing after the inlined send is reachable).
enum NonLeafOutcome {
    Value(VReg),
    Trapped,
}

pub struct IrMethod {
    /// OSR-heal: see `Nmethod::osr_cold_sends` (copied there by the driver).
    pub osr_cold_sends: u16,
    /// True for an OSR compilation. F7 consumer: the prologue nil-fill
    /// shrink (regalloc) must NOT trust entry-block defs when the OSR
    /// entry jumps straight to the loop header, bypassing them.
    pub is_osr: bool,
    pub blocks: Vec<IrBlock>,
    pub vregs: Vec<VRegInfo>,
    pub pool: Vec<PoolEntry>,
    pub argc: u8,
    pub ntemps: u8,
    /// S24 A1: `Some(closure vreg)` iff this is a BLOCK compilation — the
    /// driver's deopt-scope recording uses it as the root scope's
    /// `receiver` ValueLoc (the materialized interpreter block frame's
    /// receiver-ARG slot holds the CLOSURE, `activate_block_interp`'s own
    /// shape; the FP+4 receiver copy is derived as `copied[0]`).
    pub block_closure_vreg: Option<VReg>,
    /// S14 step 7-II-b: the vregs holding M's promoted captured temps (one per
    /// ctx-slot) when M's heap Context is elided; empty otherwise.
    /// `build_deopt_metadata` records `CtxLoc::Elided` over their frame slots so
    /// a deopt rebuilds a real Context.
    pub ctx_vregs: Vec<VReg>,
    /// S24 A3b: Some iff M's heap Context is MATERIALIZED (the prologue
    /// allocated it into this vreg; ctx-temps and escaping closures' copied[1]
    /// read it). `build_deopt_metadata` records `CtxLoc::Materialized` over
    /// its frame slot so a deopt REUSES the SAME Context (identity — Risk 2).
    /// The second element is M's `nctx()` — the driver needs it to size the
    /// fresh `CtxLoc::Elided` fallback for the one window where the ctx vreg
    /// is not yet live (the prologue alloc itself; see `root_ctx`'s match).
    /// Mutually exclusive with a non-empty `ctx_vregs`.
    pub method_ctx_vreg: Option<(VReg, usize)>,
    /// S24 B4: count of spliced blocks containing a `^` in this compile —
    /// the driver bumps `vm.stats.blocks_spliced_nlr` from it.
    pub spliced_nlr: u32,
    /// S24 B5: count of multi-BB block bodies grafted in this compile.
    pub spliced_multibb: u32,
    /// S24 B5 step 4: method-inline sites demoted to a plain `CallSend`
    /// because the cumulative `total_bytes` budget would have been exceeded.
    pub splice_declined_budget: u32,
    pub safepoints: Vec<SafepointId>,
    /// Always present (`convert` interns them unconditionally) — `emit.rs`
    /// has no `VmState` of its own to intern `true_obj`/`false_obj` on
    /// demand for `SmiCmpVal`/`BoolBr`'s sequences (D5.3), so it reads
    /// these instead of relying on a fixed pool-index convention.
    pub true_lit: PoolLit,
    pub false_lit: PoolLit,
    /// S11 D7 `Alloc`: `nil_lit` is the `nil` oop the inline fast path
    /// stores into every body slot (nil-init is MANDATORY — a GC at the
    /// next safepoint would otherwise scan a garbage body); `mark_slots_lit`
    /// is the pristine `Format::Slots` mark word
    /// (`Mark::pristine().with_tagged_contents(true)`), stored into the
    /// object header. Both are `emit.rs`-facing constants with no `VmState`
    /// to intern on demand, exactly like `true_lit`/`false_lit`.
    pub nil_lit: PoolLit,
    pub mark_slots_lit: PoolLit,
    /// Float fast-path (`docs/float_fastpath_design.md`): the pristine mark
    /// word an inline `FBox` stamps into a fresh `Double` — RAW contents
    /// (`with_tagged_contents(false)`: the body is an f64 bit pattern the GC
    /// must never scan), unlike `mark_slots_lit`'s tagged form.
    pub mark_double_lit: PoolLit,
    /// The Double klass oop (`RelocKind::Oop` — a moving GC keeps it
    /// current), for `FBox`'s header stamp and `FUnbox`'s klass guard.
    pub double_klass_lit: PoolLit,
    /// SIMD (`docs/SIMD.md`): the `Float64x2` klass oop (`RelocKind::Oop`), for
    /// `VecArith`'s operand guards and the result box's header stamp. Its box
    /// reuses `mark_double_lit` — both share `Format::Double` (a raw 16-byte
    /// body the GC skips), so the mark word is identical; only the klass differs.
    pub float64x2_klass_lit: PoolLit,
    /// SIMD: the `Float32x4` klass oop — same role as `float64x2_klass_lit` for
    /// a `.4s` `VecArith` (its box likewise reuses `mark_double_lit`; only the
    /// klass and the NEON arrangement differ).
    pub float32x4_klass_lit: PoolLit,
    /// SIMD: the `Int32x4` klass oop — same role for an integer `.4s` `VecArith`.
    pub int32x4_klass_lit: PoolLit,
    /// S11 D3: one entry per `Ir::CallSend` site, indexed by its own
    /// `site: u16` — mirrors `pool`'s own "small side table, indexed by a
    /// compact id embedded in the `Ir`" shape. `emit.rs` has no `VmState`/
    /// `MethodOop` of its own to pull a selector from at emit time (same
    /// reasoning as `true_lit`/`false_lit` above), so `convert()` resolves
    /// it once, here.
    pub call_sites: Vec<CallSiteInfo>,
    /// S14 step 2 (A1 feedback annotation): the `SiteFeedback` observed for each
    /// `call_sites` entry, SAME index — the receiver types the interpreter saw
    /// at that send, read from its IC by `feedback::read_send_site` during
    /// `convert`. Pure annotation in this step: the inliner (later steps) reads
    /// it to decide speculate/inline/trap; emit ignores it (no behaviour change,
    /// so listing goldens stay byte-stable). Kept parallel to `call_sites`
    /// rather than a field on `CallSiteInfo` because `SiteFeedback` is not `Copy`
    /// (its `Poly` arm owns a `Vec`) and `CallSiteInfo` is.
    pub site_feedback: Vec<crate::compiler::feedback::SiteFeedback>,
    /// S14 step 4b: one `(receiver_klass, selector)` pair per INLINED leaf
    /// body — the guard's assumption (`lookup(receiver_klass, selector)`
    /// resolves to the callee actually spliced). `driver::compile_method`
    /// copies this verbatim into `Nmethod.inline_deps`, which `deps::
    /// affected_by_install` consults so redefining an inlined callee makes the
    /// caller nmethod `NotEntrant`. Empty for a method that inlined nothing
    /// (keeps such methods' behaviour and goldens unchanged).
    pub inline_deps: Vec<(KlassOop, SymbolOop)>,
    /// S14 step 5: true when a guard-free SELF-send inline was spliced, i.e.
    /// this code is only correct for receivers of exactly the customization
    /// klass. Super-send linking (`driver`'s pre-seed and `rt_resolve_send`'s
    /// super arm) checks the copied `Nmethod.self_devirt` and links the c2i
    /// adapter instead of `verified_entry` when set — a super site enters
    /// with SUBCLASS receivers the entry guard never sees.
    pub self_devirt: bool,
    /// S13: the physical oop-pool index holding THIS method's own compiled
    /// `MethodOop`, interned (as `RelocKind::Oop`, so a moving GC keeps it
    /// current) at the end of `convert` — but ONLY when the method has at
    /// least one deopt site (a real `CallSend`/`Alloc` safepoint). A deopt
    /// materializer (`runtime::deopt`) reads it via `read_pool_oop` to fill a
    /// reconstructed frame's `FRAME_METHOD` slot (D5 M3); every S13 scope is
    /// this same method (depth-1), so `driver::build_deopt_metadata` stamps
    /// every scope record with this one index. `None` for a method with no
    /// deopt sites (all-smi-inline / no sends) — it never deopts, so it needs
    /// no method-oop pool word, which keeps such methods' listing goldens
    /// byte-stable (the reason step 3b left `method_pool_ix` a placeholder).
    pub method_pool_ix: Option<u32>,
}

/// One `Ir::CallSend` site's selector/argc — see `IrMethod::call_sites`.
#[derive(Clone, Copy, Debug)]
pub struct CallSiteInfo {
    pub selector: SymbolOop,
    pub argc: u8,
    /// D4.6: `Some(holder.superclass())` for a `send_super` site — the
    /// STATIC starting klass for lookup, fixed at compile time (method
    /// holders don't move between klasses in v1) and independent of the
    /// runtime receiver's own klass. `None` for an ordinary dynamic send,
    /// whose starting klass is the receiver's own, only known at runtime.
    /// `driver::compile_method` reads this to pre-resolve a super site's
    /// own target and seed `IcState::Mono` directly, instead of patching
    /// to `stub_resolve` and starting `Unresolved` like every other site.
    pub static_klass: Option<KlassOop>,
}

// ── conversion ───────────────────────────────────────────────────────────

/// Net operand-stack effect of one decoded instruction — the raw material
/// for `compute_entry_depths`' worklist. `Send`'s effect depends on the
/// site's own IC (`argc` isn't in the bytecode operand, only the IC index
/// is), so this needs `method` to look it up.
pub(crate) fn instr_stack_delta(method: MethodOop, instr: &Instr) -> i32 {
    match instr {
        Instr::PushSelf
        | Instr::PushNil
        | Instr::PushTrue
        | Instr::PushFalse
        | Instr::PushSmi(_)
        | Instr::PushLiteral(_)
        | Instr::PushTemp(_)
        | Instr::PushInstvar(_)
        | Instr::PushGlobal(_)
        | Instr::Dup
        | Instr::PushCtxTemp { .. }
        | Instr::PushClosure { .. } => 1,
        Instr::StoreTemp(_) => 0,
        Instr::StoreTempPop(_)
        | Instr::StoreInstvarPop(_)
        | Instr::StoreGlobalPop(_)
        | Instr::StoreCtxTempPop { .. }
        | Instr::Pop => -1,
        Instr::Send { ic, .. } => -(InterpreterIc::at(method, *ic).argc() as i32),
        Instr::JumpFwd(_) | Instr::JumpBack(_) => 0,
        Instr::BrTrueFwd(_) | Instr::BrFalseFwd(_) => -1,
        // S24 A1: `block_return_tos` and `nlr_tos` both POP their value
        // (`interpreter::mod`'s own handlers: `let r = pop(vm)` /
        // `let value = pop(vm)`) — while blocks were compile-ineligible
        // these two rows were never consulted and sat at a harmless 0;
        // block compilation makes them load-bearing for
        // `compute_entry_depths`' accounting.
        Instr::ReturnTos | Instr::BlockReturnTos | Instr::NlrTos => -1,
        Instr::ReturnSelf => 0,
    }
}

/// D3.2's worklist pass: `entry_depth[0] = 0`; every successor either
/// receives the propagated depth (first visit) or must already agree with
/// it (frontend-emitted bytecode guarantees this — an `assert!`, not a
/// `debug_assert!`, since a mismatch means malformed/untrusted bytecode
/// reached the compiler, not merely an internal invariant). Unreached
/// blocks (`unreachable_after_return`'s dead tail) are left at depth 0 —
/// harmless, since nothing ever reads an unreached block's entry_stack.
///
/// Also returns the highest operand-stack depth reached anywhere (block
/// entries and every point strictly inside a block's body) — `driver.rs`'s
/// `eligible` reuses this same traversal for D1's frame-budget check
/// (`ntemps + max_stack <= 60`) rather than re-deriving it with a second,
/// separately-maintained scan; `convert` itself has no use for the value,
/// it just threads it back out.
pub(crate) fn compute_entry_depths(method: MethodOop, cfg: &Cfg) -> (Vec<i32>, i32) {
    let mut entry_depth: Vec<Option<i32>> = vec![None; cfg.blocks.len()];
    entry_depth[0] = Some(0);
    let mut worklist = vec![0usize];
    let mut max_depth = 0i32;

    fn record(
        entry_depth: &mut [Option<i32>],
        worklist: &mut Vec<BlockIndex>,
        b: BlockIndex,
        depth: i32,
    ) {
        match entry_depth[b] {
            Some(d) => assert_eq!(
                d, depth,
                "compute_entry_depths: block {b} reached at two different simulated stack \
                 depths ({d} vs {depth}) -- malformed bytecode (frontend is trusted to keep \
                 depth consistent across every merge)"
            ),
            None => {
                entry_depth[b] = Some(depth);
                worklist.push(b);
            }
        }
    }

    while let Some(b) = worklist.pop() {
        let block = &cfg.blocks[b];
        let mut depth = entry_depth[b].expect("worklist entries always have a known depth");
        max_depth = max_depth.max(depth);
        let mut bci = block.bci_start;
        while bci < block.bci_end {
            let (instr, next) = decode_at(method, bci);
            depth += instr_stack_delta(method, &instr);
            max_depth = max_depth.max(depth);
            bci = next;
        }
        debug_assert!(
            depth >= 0,
            "compute_entry_depths: block {b} simulated to a negative depth ({depth})"
        );
        match block.terminator {
            Terminator::Fallthrough(t) | Terminator::Jump { target: t, .. } => {
                record(&mut entry_depth, &mut worklist, t, depth);
            }
            Terminator::Branch { if_true, if_false } => {
                record(&mut entry_depth, &mut worklist, if_true, depth);
                record(&mut entry_depth, &mut worklist, if_false, depth);
            }
            Terminator::Return => {}
        }
    }
    (
        entry_depth.into_iter().map(|d| d.unwrap_or(0)).collect(),
        max_depth,
    )
}

/// D3.2's merge rule, evaluated once every block's `entry_depth` is known.
#[derive(Clone, Copy)]
enum EntryStackSource {
    /// `entry_depth == 0`: `entry_stack = []`, unconditionally.
    Empty,
    /// Exactly one predecessor, reached by a non-backward edge: inherits
    /// that predecessor's exit stack verbatim (same vregs, no `Move`s).
    Inherit(BlockIndex),
    /// A join, or a loop header with nonzero entry depth: owns fresh
    /// merge vregs; every predecessor `Move`s into them.
    Merge,
}

fn entry_stack_sources(cfg: &Cfg, entry_depth: &[i32]) -> Vec<EntryStackSource> {
    let mut predecessors: Vec<Vec<(BlockIndex, bool)>> = vec![Vec::new(); cfg.blocks.len()];
    for (i, block) in cfg.blocks.iter().enumerate() {
        match block.terminator {
            Terminator::Fallthrough(t) => predecessors[t].push((i, false)),
            Terminator::Jump {
                target,
                is_backward,
            } => predecessors[target].push((i, is_backward)),
            Terminator::Branch { if_true, if_false } => {
                predecessors[if_true].push((i, false));
                predecessors[if_false].push((i, false));
            }
            Terminator::Return => {}
        }
    }
    (0..cfg.blocks.len())
        .map(|b| {
            if entry_depth[b] == 0 {
                EntryStackSource::Empty
            } else if predecessors[b].len() == 1 && !predecessors[b][0].1 {
                EntryStackSource::Inherit(predecessors[b][0].0)
            } else {
                EntryStackSource::Merge
            }
        })
        .collect()
}

/// Which smi-inlineable shape a mono smi-guarded IC's target primitive
/// maps to (D1 item 2's `SMI_INLINE` set; ids from `runtime::primitives`:
/// 1=+ 2=- 3=* 6=bitAnd: 7=bitOr: 8=bitXor: 10=< 11=<= 12=> 13=>= 14== 15=~=).
enum SmiSendKind {
    Arith(SmiOp),
    Cmp(CmpOp),
}

/// The SmallInteger method a smi-special site dispatches to — accepting
/// BOTH a warm Mono-smi IC and a stone-cold EMPTY one.
///
/// The Empty case is the Cog-parity speculation (2026-07, the sieve
/// investigation): an OSR compile mid-first-call reads loop-TAIL sites
/// that have never executed — sieve's `count := count + 1` and its outer
/// increment sat exactly after the point the backedge counter tripped, so
/// they compiled as full CallSends inside the hottest loop (~18k sends
/// per sieveOnce, a 30x loss against Cog) and, being plain sends, never
/// trapped, so no recompile ever healed them. Cog never consults
/// feedback for arithmetic at all.
///
/// Speculating smi on an Empty `+`/`<`/... site costs, when wrong, one
/// reexecute-trap plus one recompile — and the customized profile hash
/// now sees the IC that trap warms, so the recompile actually lowers the
/// truth. When right (overwhelmingly, for these selectors), the site is
/// tight inline code with no warmup and no deopt, ever.
///
/// Resolution is against the LIVE world (`resolve_method_ro`), so a guest
/// that redefines `SmallInteger>>+` to a non-primitive method disables
/// the speculation rather than being miscompiled.
fn smi_special_target(vm: &VmState, ic: &InterpreterIc) -> Option<MethodOop> {
    let guard = ic.guard();
    if guard.raw() == vm.universe.smi_klass.oop().raw() {
        return mono_target_method(vm, ic, vm.universe.smi_klass);
    }
    if guard.raw() == vm.universe.nil_obj.raw() {
        return crate::compiler::feedback::resolve_method_ro(
            vm,
            vm.universe.smi_klass,
            ic.selector(),
        );
    }
    None
}

/// The method behind a Mono IC's target, whichever of its TWO
/// representations the slot holds. `set_mono` stores a MethodOop;
/// `set_mono_compiled` stores a SmallInt-encoded NmethodId — and a bare
/// `MethodOop::try_from(ic.target())` silently fails on the latter.
///
/// That silent failure was a compounding performance bug: the moment a
/// primitive method itself got COMPILED (prim shims made `SmallInteger>>+`
/// and `basicNew` compilable), every IC dispatching to it was repatched to
/// the NmethodId form, and every gate below — smi-inline, array-op,
/// alloc-site — began declining sites it had accepted the day before.
/// Loop increments compiled as full CallSends (sieve: ~18k sends per run,
/// a 30x loss against Cog); `basicNew` sites lost the `Ir::Alloc` inline
/// bump (alloc: 4-8x against Cog). The feedback layer already had the
/// dual decoder (`resolve_target`); the gates just never used it.
fn mono_target_method(
    vm: &VmState,
    ic: &InterpreterIc,
    guard_klass: crate::oops::wrappers::KlassOop,
) -> Option<MethodOop> {
    crate::compiler::feedback::resolve_target(vm, ic.target(), guard_klass, ic.selector())
}

/// S14 opt: is `method`'s send site `ic_idx` a mono-smi-guarded `SMI_INLINE`
/// primitive — the fuse-to-`SmiArith`/`SmiCmpVal` class? The free-function
/// twin of `Translator::is_smi_inlinable`, usable on an INLINED callee's own
/// IC table (the splicers fuse their bodies' smi ops instead of emitting
/// generic `CallSend`s — the difference between an inlined loop being tight
/// machine code and being a chain of stub calls).
fn is_smi_inlinable_on(vm: &VmState, method: MethodOop, ic_idx: u16) -> bool {
    let ic = InterpreterIc::at(method, ic_idx);
    let Some(target) = smi_special_target(vm, &ic) else {
        return false;
    };
    crate::compiler::driver::SMI_INLINE.contains(&target.primitive())
}

/// S15: free-function twin of `Translator::array_op_kind` for an INLINED
/// callee's own IC table (the splicers intrinsify their bodies' array ops
/// exactly like their smi ops). `Some(is_put)`.
fn array_op_kind_on(vm: &VmState, method: MethodOop, ic_idx: u16) -> Option<bool> {
    let ic = InterpreterIc::at(method, ic_idx);
    if ic.guard().raw() != vm.universe.array_klass.oop().raw() {
        return None;
    }
    let target = mono_target_method(vm, &ic, vm.universe.array_klass)?;
    match target.primitive() {
        26 => Some(false),
        27 => Some(true),
        _ => None,
    }
}

/// Float fast-path: free-function twin of `Translator::is_double_inlinable`
/// for an INLINED callee's own IC table (the splicers fuse their bodies'
/// Double ops to `FUnbox`/`FArith`/`FBox`/`FCmpVal` exactly like their smi
/// and array ops). Without this, every Double arithmetic/compare op in an
/// inlined body decayed to a generic `CallSend` that BOXES its result — a
/// zero-alloc float loop became box-per-op the moment it was spliced into a
/// caller (measured: the level-4 inlining probe turned Mandelbrot's 8 MB
/// into 1 GB of allocation, 12x slower, purely from this decay). Resolves
/// via (guard klass, selector) rather than the raw IC target, for the same
/// staleness reason as `classify_double_send`.
fn is_double_inlinable_on(vm: &VmState, method: MethodOop, ic_idx: u16) -> bool {
    let ic = InterpreterIc::at(method, ic_idx);
    if ic.guard().raw() != vm.universe.double_klass.oop().raw() {
        return false;
    }
    let Some(target) =
        crate::compiler::feedback::resolve_method_ro(vm, vm.universe.double_klass, ic.selector())
    else {
        return false;
    };
    crate::compiler::driver::DOUBLE_INLINE.contains(&target.primitive())
}

/// True iff the operand on top of `stack` (a send's LAST-pushed arg) was
/// produced by a `ConstPool` load — i.e. it is a compile-time NON-smi literal
/// (a Double, LargeInteger, …; a smi literal is an immediate `ConstSmi`, never
/// pooled). Such an arg can never satisfy a `SmiArith`/`SmiCmp` operand guard,
/// so fusing the send would emit a fail edge that traps on EVERY call — the
/// `SmallInteger>>asFloat` (`^self * 1.0`) storm, the smi mirror of the
/// `aDouble < <smi>` FUnbox storm. Declining the fuse lets the send fall through
/// to a plain `CallSend`, whose shim runs `SmallInteger>>*`'s own coercing
/// bytecode fallback (`self asFloat * arg`) with no deopt at all. The hot path
/// (`x + 1`, `x < n` — smi-const or smi-var args) is untouched: a `ConstSmi`
/// arg is not a `ConstPool`, and a variable arg has some other producer.
fn smi_fuse_arg_is_pooled(stack: &[VReg], code: &[Ir]) -> bool {
    matches!(
        (stack.last(), code.last()),
        (Some(&top), Some(Ir::ConstPool { dst, .. })) if *dst == top
    )
}

fn classify_smi_send(vm: &VmState, method: MethodOop, ic_idx: u16) -> SmiSendKind {
    let ic = InterpreterIc::at(method, ic_idx);
    let target = smi_special_target(vm, &ic).expect(
        "classify_smi_send: site is neither mono-smi nor an Empty smi-special -- \
         is_smi_inlinable should have rejected it (compiler bug if this fires)",
    );
    match target.primitive() {
        1 => SmiSendKind::Arith(SmiOp::Add),
        2 => SmiSendKind::Arith(SmiOp::Sub),
        3 => SmiSendKind::Arith(SmiOp::Mul),
        6 => SmiSendKind::Arith(SmiOp::And),
        7 => SmiSendKind::Arith(SmiOp::Or),
        8 => SmiSendKind::Arith(SmiOp::Xor),
        10 => SmiSendKind::Cmp(CmpOp::Lt),
        11 => SmiSendKind::Cmp(CmpOp::Le),
        12 => SmiSendKind::Cmp(CmpOp::Gt),
        13 => SmiSendKind::Cmp(CmpOp::Ge),
        14 => SmiSendKind::Cmp(CmpOp::Eq),
        15 => SmiSendKind::Cmp(CmpOp::Ne),
        other => panic!(
            "classify_smi_send: primitive {other} is not in SMI_INLINE -- driver::eligible \
             should have rejected this method (compiler bug if this fires)"
        ),
    }
}

/// SIMD (`docs/SIMD.md`): map a vector arithmetic primitive number to the NEON
/// op it fuses to. The four elementwise ops are numbered contiguously from a
/// per-kind base — Float64x2 `+ - * /` = 129-132, Float32x4 = 136-139 (world/
/// 38_simd.mst) — so ONE offset table serves both. `None` for any non-arith
/// primitive (`x:y:…`/`splat:`/`at:`), which stays an ordinary send.
fn vec_arith_op(kind: VecKind, prim: i64) -> Option<FArithOp> {
    let base = match kind {
        VecKind::F64x2 => 129,
        VecKind::F32x4 => 136,
        VecKind::I32x4 => 150,
    };
    match prim - base {
        0 => Some(FArithOp::Add),
        1 => Some(FArithOp::Sub),
        2 => Some(FArithOp::Mul),
        // NEON has no vector integer divide, so `Int32x4` stops at `*` — its
        // `at:` (prim 153, offset 3) must NOT be mistaken for a divide.
        3 if !kind.is_int() => Some(FArithOp::Div),
        _ => None,
    }
}

/// SIMD: the `classify_double_send` analog for a mono-vector-guarded site —
/// resolves the NEON arithmetic op the site's `Ir::VecArith` computes. Vectors
/// have no fused compares, so it returns `FArithOp` directly. Only ever called
/// from the fuse arm after `vec_send_kind` has already vetted the site (and
/// yielded `kind`), so both failure paths are compiler-bug guards.
fn classify_vec_send(vm: &VmState, method: MethodOop, ic_idx: u16, kind: VecKind) -> FArithOp {
    let ic = InterpreterIc::at(method, ic_idx);
    let klass = match kind {
        VecKind::F64x2 => vm.universe.float64x2_klass,
        VecKind::F32x4 => vm.universe.float32x4_klass,
        VecKind::I32x4 => vm.universe.int32x4_klass,
    };
    debug_assert_eq!(
        ic.guard().raw(),
        klass.oop().raw(),
        "classify_vec_send: IC {ic_idx} guard does not match kind {kind:?} -- \
         vec_send_kind should have rejected this site"
    );
    let target = crate::compiler::feedback::resolve_method_ro(vm, klass, ic.selector())
        .expect("classify_vec_send: vector klass must understand a vec_send_kind selector");
    vec_arith_op(kind, target.primitive()).unwrap_or_else(|| {
        panic!(
            "classify_vec_send: primitive {} is not an arith op for {kind:?} -- \
             vec_send_kind should have rejected this site (compiler bug)",
            target.primitive()
        )
    })
}

/// Float fast-path (`docs/float_fastpath_design.md` B2 rule 1): the
/// `classify_smi_send` analog for a mono-Double-guarded send site. The IC is
/// the type oracle: every observed receiver was a Double, and the resolved
/// target is one of the `Double` arithmetic/compare primitives — so the send
/// collapses to unbox/native-op/box with an uncommon-trap fail edge.
enum DoubleSendKind {
    Arith(FArithOp),
    Cmp(CmpOp),
}

fn classify_double_send(vm: &VmState, method: MethodOop, ic_idx: u16) -> DoubleSendKind {
    let ic = InterpreterIc::at(method, ic_idx);
    assert_eq!(
        ic.guard().raw(),
        vm.universe.double_klass.oop().raw(),
        "classify_double_send: IC {ic_idx} is not mono-Double-guarded -- is_double_inlinable \
         should have rejected this site (compiler bug if this fires)"
    );
    // Resolve via (guard klass, selector), NOT the raw target slot: Double's
    // arithmetic methods are shimmable and tier up early, so a warm site's
    // mono target is usually a smi NmethodId, not a MethodOop (the exact
    // staleness `feedback::resolve_target` guards against — and unlike the
    // smi prims, which are compile_disabled forever, Double methods DO
    // compile, so the smi fuse never needed this).
    let target = crate::compiler::feedback::resolve_method_ro(
        vm,
        vm.universe.double_klass,
        ic.selector(),
    )
    .expect("classify_double_send: Double must understand an is_double_inlinable selector");
    match target.primitive() {
        100 => DoubleSendKind::Arith(FArithOp::Add),
        101 => DoubleSendKind::Arith(FArithOp::Sub),
        102 => DoubleSendKind::Arith(FArithOp::Mul),
        103 => DoubleSendKind::Arith(FArithOp::Div),
        104 => DoubleSendKind::Cmp(CmpOp::Lt),
        105 => DoubleSendKind::Cmp(CmpOp::Eq),
        other => panic!(
            "classify_double_send: primitive {other} is not in DOUBLE_INLINE -- \
             is_double_inlinable should have rejected this site (compiler bug if this fires)"
        ),
    }
}

/// Compiler-stage literal pool, deduplicated by `(value, kind)` — same
/// rationale as `JasmAssembler`'s own pool dedup (P10): a value that is
/// both a plain constant and separately relocation-bearing must not share
/// a slot.
struct PoolBuilder {
    entries: Vec<PoolEntry>,
    dedup: HashMap<(u64, Option<RelocKind>), PoolLit>,
}

impl PoolBuilder {
    fn new() -> PoolBuilder {
        PoolBuilder {
            entries: Vec::new(),
            dedup: HashMap::new(),
        }
    }

    fn intern(&mut self, value: u64, kind: Option<RelocKind>) -> PoolLit {
        let key = (value, kind);
        if let Some(&id) = self.dedup.get(&key) {
            return id;
        }
        let id = PoolLit(self.entries.len() as u32);
        self.entries.push(PoolEntry { value, kind });
        self.dedup.insert(key, id);
        id
    }
}

/// Owns the state accumulated across every block's translation (`vregs`,
/// `pool`) plus the read-only per-method context every block needs
/// (`self_vreg`/`temp_vregs`) — a struct instead of a long parameter list
/// S24 B5: what kind of body [`Translator::try_inline_cfg`] is grafting.
///
/// `Method` — a (possibly devirtualized) METHOD callee: klass-guarded unless
/// statically resolved, `InlineSite { is_block: false, parent: None }`,
/// eligibility via `is_inline_eligible_cfg`, and a redefinition dependency
/// recorded. `Block` — S24 B5, a literal BLOCK's body grafted into its home
/// compilation: no guard (the receiver is statically the literal block —
/// there is nothing to guard), `is_block: true` with the given parent chain
/// (depth-3: block ← inlined callee ← root), eligibility proved by
/// `escape::block_is_spliceable_cfg` up front, `NlrTos` legal (a return from
/// the HOME compilation, `Ir::Ret` — B4's deopt closure synthesis covers any
/// in-graft deopt), depth-0 ctx-temp access routed through the home Context,
/// and NO inline dependency (a literal block is not a dispatched callee — no
/// redefinition can retarget it).
/// S24 B5: what [`Translator::splice_block`] did with the value-send.
enum SpliceOutcome {
    /// Single-BB linear splice completed in the CURRENT block; result pushed.
    Done,
    /// The (single-BB) body ended in a terminating trap/NLR — the caller
    /// must stop translating its current block.
    Trapped,
    /// S24 B5: a multi-BB body was GRAFTED — the caller must end its current
    /// block (pending_jump_target is armed at the graft entry) and continue
    /// translating in the returned continuation block.
    Continuation(BlockId),
}

enum GraftMode {
    Method,
    // Constructed by B5 step 2 (block_is_spliceable_cfg + splice routing);
    // inert in step 1's byte-identical refactor.
    #[allow(dead_code)]
    Block {
        parent: Option<Box<InlineSite>>,
    },
}

/// threaded through every helper.
struct Translator<'a> {
    vm: &'a VmState,
    method: MethodOop,
    /// S24 B4: spliced blocks whose body contained a `^` — transferred to
    /// `IrMethod.spliced_nlr`, bumped into `vm.stats` by the driver.
    spliced_nlr: u32,
    /// S24 B5: multi-BB block bodies grafted this compile.
    spliced_multibb: u32,
    /// S24 B5 step 4: cumulative bytecode bytes committed to inlined/spliced
    /// bodies this compile (`inline_cost` of every grafted callee and block).
    /// Enforced against `InlineBudget::total_bytes` ONLY at declinable
    /// method-inline decisions (fallback = plain `CallSend`); COMMITTED block
    /// splices (value sites, blockarg sites — no legal convert-time fallback)
    /// count toward it but are never declined by it (design amendment A1).
    budget_bytes_used: u32,
    /// S24 B5 step 4: declinable inline decisions demoted over-budget.
    splice_declined_budget: u32,
    /// S14 step 5 (customization, A3): the receiver klass K this compilation
    /// is FOR. The nmethod's entry guard proves every receiver has exactly
    /// this klass, so a send whose receiver is provably `self` dispatches
    /// against K statically — resolved at compile time via
    /// `resolve_method_ro(vm, rcvr_klass, selector)`, with no IC warmth
    /// required and (within budget) a guard-free inline.
    rcvr_klass: KlassOop,
    /// S14 step 5: set when any guard-free SELF-send inline was spliced —
    /// i.e. the emitted code's correctness depends on the receiver really
    /// having klass `rcvr_klass`. A super-send site elsewhere links straight
    /// to `verified_entry` (skipping the entry guard) with a SUBCLASS
    /// receiver, so `driver`/`rt_resolve_send` must refuse the compiled link
    /// (fall back to the c2i adapter) when this is set on the target.
    self_devirt: bool,
    /// This is an OSR-entry compile (triggered by a loop back-edge counter,
    /// mid-activation) rather than a normal entry-point compile (triggered
    /// by the invocation counter). It gates [`Translator::cold_send_traps`]:
    /// the three hotness signals (invocation / loop / OSR) OR together to
    /// TRIGGER a compile (trigger unification, L2 step 2), but each certifies
    /// a DIFFERENT region as profiled — the invocation signal covers the
    /// whole body, the loop/OSR signal only the loop's executed part. An
    /// `Untaken` (cold-IC) send is genuinely "cold" only under whole-body
    /// coverage; under OSR it may just be not-reached-YET, so trapping it
    /// would deopt the instant the loop exits (the sieve deopt-thrash).
    osr: bool,
    vregs: Vec<VRegInfo>,
    pool: PoolBuilder,
    self_vreg: VReg,
    temp_vregs: Vec<VReg>,
    /// S14 step 7-II-b: one vreg per M ctx-slot (`method.nctx()`) — the promoted
    /// homes of M's captured temps when M's heap Context is ELIDED (every block
    /// capturing them is inlined). `push_ctx_temp{depth:0,idx}` /
    /// `store_ctx_temp_pop{depth:0,idx}` in M's own body AND in a spliced
    /// (ctx-less, `captures_ctx`) block both address `ctx_vregs[idx]` — a
    /// ctx-less block's frame aliases M's Context, so it reaches M's ctx-temps at
    /// depth 0. Empty when M is not `has_ctx`. Nil-initialised at method entry
    /// (matching `alloc_context`'s nil-fill). The deopt root scope records
    /// `CtxLoc::Elided` over these so the materializer rebuilds a real Context.
    ctx_vregs: Vec<VReg>,
    /// S24 A1: Some iff compiling a CompiledBlock body (`convert`'s
    /// is_block mode) — the closure param vreg (x0 at entry), pinned live
    /// for the whole body (deopt reads it; NlrReturn reads it).
    block_closure_vreg: Option<VReg>,
    /// S24 A1: Some iff is_block AND captures_ctx — the HOME Context oop
    /// (`closure.copied[1]`), through which depth-0 ctx-temp ops do REAL
    /// barriered Context-slot loads/stores (never the elided-ctx vreg
    /// Moves a method compilation uses).
    block_ctx_vreg: Option<VReg>,
    /// S24 A3b: Some iff M is a `has_ctx` METHOD whose Context is
    /// MATERIALIZED (the prologue allocates it — some capturing closure
    /// escapes, so the Context cannot be elided). Depth-0 ctx-temp ops go
    /// through it exactly like `block_ctx_vreg`; the AllocClosure lowering
    /// stores it into an escaping capturing block's `copied[1]`. Mutually
    /// exclusive with `block_ctx_vreg` (a method is not a block) and with
    /// the elided `ctx_vregs` promotion (7-II-b).
    method_ctx_vreg: Option<VReg>,
    /// S11 step 7 (D1): a smi-inlined fail edge, or a `not_bool` edge, no
    /// longer bails the whole method out to the interpreter — it computes
    /// a real fallback value (`CallSend`/`CallRuntime`) and REJOINS normal
    /// control flow, so each one needs its OWN fresh synthetic `IrBlock`
    /// (a fail block can't share the single "restart from scratch" target
    /// `Ir::Bailout` used to, since it has to rejoin at a SPECIFIC point
    /// with a SPECIFIC value). `next_extra_block` hands out ids starting
    /// right after every real CFG block (mirrors the old single
    /// `bailout_id`'s own "`cfg.blocks.len()`" starting point).
    ///
    /// `blocks_by_id[k]` holds `BlockId(k)`'s own finished `IrBlock` once
    /// known, `None` until then — indexed by id, NOT creation order. This
    /// matters: a fail block and its own continuation are allocated
    /// out-of-order relative to when each one's CODE is actually finished
    /// (`fail_and_continue` mints the CONTINUATION's id before the FAIL
    /// block's own, but the fail block's own code is complete immediately
    /// while the continuation's is only finished later, possibly after
    /// ANOTHER split narrows in first) — `emit.rs`'s own `method.blocks
    /// [bid.0 as usize]` (and its parallel per-index `labels` vec) both
    /// assume `IrMethod.blocks[i].id == BlockId(i)` for every `i`, a
    /// (previously implicit, only just discovered to be load-bearing)
    /// invariant a plain "push whatever finishes, in whatever order it
    /// finishes" `Vec` cannot uphold once more than one block can still be
    /// "in flight" at once. Resized to fit whenever `fresh_block_id` mints
    /// an id beyond the current length.
    next_extra_block: u32,
    blocks_by_id: Vec<Option<IrBlock>>,
    /// D4.6: accumulates one entry per `send_super`/generic/smi-fallback
    /// send translated so far, in encounter order — `Ir::CallSend.site`
    /// indexes into this, same shape as `pool`'s own accumulate-then-hand-
    /// to-`IrMethod` pattern.
    call_sites: Vec<CallSiteInfo>,
    /// S14 step 2: `SiteFeedback` per `call_sites` entry (same index), read from
    /// each send's IC as it is lowered. See `IrMethod::site_feedback`.
    site_feedback: Vec<crate::compiler::feedback::SiteFeedback>,
    /// S14 step 4b: one `(receiver_klass, selector)` per INLINED leaf body — the
    /// dependency the guard assumes (`lookup(receiver_klass, selector)` resolves
    /// to the callee actually spliced). Copied into `IrMethod.inline_deps`; the
    /// driver copies THAT into `Nmethod.inline_deps`, and `deps::
    /// affected_by_install` invalidates this nmethod if the selector is
    /// redefined anywhere on the receiver klass's superchain.
    inline_deps: Vec<(KlassOop, SymbolOop)>,
    /// S14 step 4b: recompilation level driving the inline budget
    /// (`inline::budget_for_level`). Tier-1 compiles are always level 1 today
    /// OSR-heal: count of Untaken-feedback sites lowered to generic
    /// `CallSend`s in THIS compile (only meaningful under OSR — see
    /// `Nmethod::osr_cold_sends`).
    osr_cold_sends: u16,
    /// (`driver::compile_method` sets `level: 1`); threaded here so the budget
    /// tracks it when higher levels arrive.
    level: u8,
    /// S11 D7: vregs known at compile time to hold a specific class
    /// constant, i.e. produced by a `push_global` whose Association value
    /// is a `KlassOop`. Keyed by `VReg.0`. Used to recognize `X basicNew`
    /// at a send site (the receiver traces back to a class constant), the
    /// one shape the inline-allocation fast path (`Ir::Alloc`) fires for.
    /// The class oop is read at compile time and baked into the pool — a
    /// deliberate S13-deferred staleness hole (a later `Global := OtherClass`
    /// isn't observed; a klass-format redefinition flushes via S12's sweep,
    /// same class of hole as `stale_mono_documented_hole`).
    const_class: HashMap<u32, KlassOop>,
    /// S14 step 7-I: the escape pre-pass result for M, `Some` iff M creates a
    /// literal closure (`push_closure`) that was proven elidable
    /// (`driver::eligibility_detail` guarantees `all_elidable` before compiling,
    /// so this is only ever `Some` for a method whose EVERY closure site is a
    /// good value-send use). `translate_instr` consults `value_send_target(bci)`
    /// to decide whether a `Send` splices a block body inline.
    escape: Option<crate::compiler::escape::ClosureEscape>,
    /// S14 step 7-I: parallel to `temp_vregs` — `temp_sites[t] == Some(bci)`
    /// means the unified arg/temp slot `t` currently holds the block created at
    /// `push_closure` site `bci` (a phantom value — no real vreg backs it). A
    /// `store_temp_pop` of a block-site stack slot sets this; a `push_temp` of a
    /// block-site temp slot reads it (pushing a phantom onto the operand
    /// `stack_sites` shadow), so a `b := [..]. ^b value` splices without ever
    /// materialising the closure. Tracked linearly as `convert` walks blocks in
    /// bci order; the escape pre-pass already guaranteed no block-site ever
    /// merges lossily (that would have made M `NoPermanent`), so linear tracking
    /// suffices — a site never reaches a value-send by a path convert can't see.
    temp_sites: Vec<Option<usize>>,
    /// S14 step 7-IV-a: when a `Send` arm spliced a MULTI-BLOCK callee
    /// (`try_inline_cfg`), the caller's current block must end with a `Jump`
    /// into the callee's ENTRY block — not the continuation the arm returns as
    /// `split`. The arm sets this; `convert`'s split handling `take()`s it as
    /// the Jump target (falling back to the continuation when unset — the
    /// ordinary smi-split case).
    pending_jump_target: Option<BlockId>,
}

impl<'a> Translator<'a> {
    fn fresh(&mut self, is_oop: bool) -> VReg {
        let id = self.vregs.len() as u32;
        self.vregs.push(VRegInfo {
            is_oop,
            is_fp: false,
        });
        VReg(id)
    }

    /// A fresh UNBOXED-`f64` vreg for the float fast-path
    /// (`docs/float_fastpath_design.md`) — routed to the FP register file by
    /// regalloc, never an oop.
    #[allow(dead_code)] // wired by classify_double_send (next step)
    fn fresh_fp(&mut self) -> VReg {
        let id = self.vregs.len() as u32;
        self.vregs.push(VRegInfo {
            is_oop: false,
            is_fp: true,
        });
        VReg(id)
    }

    fn fresh_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_extra_block);
        self.next_extra_block += 1;
        if self.blocks_by_id.len() <= id.0 as usize {
            self.blocks_by_id.resize_with(id.0 as usize + 1, || None);
        }
        id
    }

    /// The ONLY way a finished `IrBlock` ever enters `blocks_by_id` —
    /// indexed by `block.id`, never appended, so a block whose id was
    /// minted early but whose own code is finished late (or vice versa)
    /// always lands in its own correct slot regardless of finishing order.
    fn finish_block(&mut self, block: IrBlock) {
        let idx = block.id.0 as usize;
        if self.blocks_by_id.len() <= idx {
            self.blocks_by_id.resize_with(idx + 1, || None);
        }
        self.blocks_by_id[idx] = Some(block);
    }

    /// A smi-inlined op's fail edge (`SmiArith`/`SmiCmpVal`, both mid-block,
    /// VALUE-PRODUCING ops with no terminator of their own). S13 step 7b (D3,
    /// the first "organic trap client"): the fail block is now a single
    /// `Ir::UncommonTrap{ bci }` — NO `CallSend`, NO jump-to-continuation.
    /// The deopt transfers control to the interpreter, which re-executes the
    /// whole send (its LargeInteger/Double fallback) itself, so there is
    /// nothing to compute back in compiled code and nothing to rejoin.
    ///
    /// The FAST-path CONTINUATION block is STILL minted (the `SmiArith`/
    /// `SmiCmpVal` success falls through to it, exactly as before) — only the
    /// FAIL block changed. Returns `(fail_block_id, continuation_block_id)`.
    ///
    /// The fail block records ONE deopt site (`DeoptRaw`) at its trap op:
    /// `reexecute = true` at the SEND's bci, with the recorded operand stack
    /// = `reexec_stack` — the snapshot the caller captured BEFORE popping the
    /// send's own operands, so `a`/`b` are still on it (see `translate_instr`'s
    /// smi-inline site). Those live-across vregs are what regalloc spill-all
    /// pins into the frame for the materializer to read.
    fn fail_and_continue(
        &mut self,
        fail_bci: usize,
        reexec_stack: Vec<VReg>,
    ) -> (BlockId, BlockId) {
        let continuation_id = self.fresh_block_id();
        let fail_id = self.fresh_block_id();
        self.finish_block(IrBlock {
            id: fail_id,
            bci: fail_bci,
            code: vec![Ir::UncommonTrap { bci: fail_bci }],
            entry_stack: Vec::new(),
            deopt_sites: vec![(
                0,
                DeoptRaw {
                    stack: reexec_stack,
                    bci: fail_bci,
                    kind: SafepointKind::UncommonTrap,
                    reexecute: true,
                    stack_closures: Vec::new(),
                    inline: None,
                },
            )],
        });
        (fail_id, continuation_id)
    }

    /// S13 step 7b-ii: a `BoolBr.not_bool` edge (the value branched on wasn't
    /// `true`/`false`) DEOPTS — a single `Ir::UncommonTrap{ bci: branch_bci }`,
    /// reexecute=true at the branch opcode. The interpreter re-executes the
    /// branch, sees the non-boolean, and runs its own `must_be_boolean_send`
    /// (SPEC §5.4 Alg 11: push the value back, roll bci to the branch, send
    /// `#mustBeBoolean`, re-test) — so the compiled path stops carrying a
    /// `CallRuntime{MUST_BE_BOOLEAN}` self-loop and defers the whole
    /// (rare, cold) protocol to the interpreter. `reexec_stack` is the
    /// operand stack BEFORE the branch (the value being tested still on top,
    /// since the deferred branch never popped it) — exactly what reexecute
    /// needs. `regalloc::deopt_live` forces every vreg this site reads
    /// (receiver + slots + `reexec_stack`) live-across the trap so it spills.
    fn fresh_not_bool_block(&mut self, branch_bci: usize, reexec_stack: Vec<VReg>) -> BlockId {
        let not_bool_id = self.fresh_block_id();
        self.finish_block(IrBlock {
            id: not_bool_id,
            bci: branch_bci,
            code: vec![Ir::UncommonTrap { bci: branch_bci }],
            entry_stack: Vec::new(),
            deopt_sites: vec![(
                0,
                DeoptRaw {
                    stack: reexec_stack,
                    bci: branch_bci,
                    kind: SafepointKind::UncommonTrap,
                    reexecute: true,
                    stack_closures: Vec::new(),
                    inline: None,
                },
            )],
        });
        not_bool_id
    }

    /// `SmiCmpBr`'s own fail edge (the FUSED compare+branch case). S13 step
    /// 7b: like `fail_and_continue`, the whole fail block collapses to a
    /// single `Ir::UncommonTrap{ bci }` — the CallSend, the BoolBr, AND its
    /// synthesized `not_bool` retry block are all gone (the deopt re-executes
    /// the comparison send in the interpreter, which runs both the real
    /// selector AND its own `mustBeBoolean` handling if the override returns a
    /// non-boolean). No continuation split is needed here (this fires only at
    /// a block's own Branch terminator, whose `if_true`/`if_false` are already
    /// real CFG targets) — but for a smi-cmp fail those targets are simply
    /// never reached from this fail edge; control leaves via the trap.
    ///
    /// Records ONE deopt site: `reexecute = true` at the comparison send's
    /// bci, recorded stack = `reexec_stack` (captured before the send's
    /// operand pops, so `a`/`b` are present) — same convention as
    /// `fail_and_continue`.
    fn fail_and_branch(&mut self, fail_bci: usize, reexec_stack: Vec<VReg>) -> BlockId {
        let fail_id = self.fresh_block_id();
        self.finish_block(IrBlock {
            id: fail_id,
            bci: fail_bci,
            code: vec![Ir::UncommonTrap { bci: fail_bci }],
            entry_stack: Vec::new(),
            deopt_sites: vec![(
                0,
                DeoptRaw {
                    stack: reexec_stack,
                    bci: fail_bci,
                    kind: SafepointKind::UncommonTrap,
                    reexecute: true,
                    stack_closures: Vec::new(),
                    inline: None,
                },
            )],
        });
        fail_id
    }

    /// D1's "arbitrary send in ANY IC state" relaxation vs. D3.2's own
    /// mono-smi-inline fast path: TRUE iff `ic_idx`'s site is monomorphic,
    /// smi-guarded, and its cached target's primitive is smi-inlinable —
    /// the EXACT condition `driver::eligibility_detail`'s own
    /// `mono_smi_inline_send` allows through as `Eligibility::Yes` via the
    /// fast-path arm (kept in sync by hand, cross-referenced doc comments
    /// on both sides, same as this module's own established tolerance for
    /// controlled duplication over a cross-module coupling for a
    /// three-line predicate — `project_s11_step4_design`'s own "two
    /// conventions, kept intentionally separate" precedent). Every OTHER
    /// IC state falls to the generic `Instr::Send` arm: S14 step 3 an `Empty`
    /// site now CAN reach a real compile (its guard isn't the smi klass, so
    /// this returns `false`) and lowers there to an uncommon TRAP via
    /// `inline::decide` (`SiteFeedback::Untaken`); Poly, Mega, mono-but-non-smi,
    /// and mono-smi-but-not-inlinable lower to an ordinary generic
    /// `Ir::CallSend`.
    /// S14 step 5: the statically-resolved target of a SELF-send against the
    /// known receiver klass `k` — `None` when devirtualization must NOT
    /// apply: unresolvable selectors (the DNU path keeps its feedback
    /// lowering), and SMI-FUSABLE primitives on a smi customization — the
    /// feedback path's trap→warm→recompile cycle re-lowers those to fused
    /// `SmiArith`/`SmiCmpVal`, which a devirtualized plain call would forfeit
    /// for the nmethod's whole lifetime (review finding, this step).
    fn devirt_self_target(
        &self,
        k: KlassOop,
        selector: crate::oops::wrappers::SymbolOop,
    ) -> Option<MethodOop> {
        // S24 A1: NEVER devirtualize self-sends in a block compilation —
        // `self` inside a block is the HOME receiver (any klass in the home
        // holder's subtree), while `k` here would be the no-customization
        // closure_klass filler; resolving against it would splice the WRONG
        // method (design §2.1: self-sends stay generic sends in block
        // bodies, v1).
        if self.method.is_block() {
            return None;
        }
        let target = crate::compiler::feedback::resolve_method_ro(self.vm, k, selector)?;
        let smi_fusable = k.oop().raw() == self.vm.universe.smi_klass.oop().raw()
            && crate::compiler::driver::SMI_INLINE.contains(&target.primitive());
        if smi_fusable {
            None
        } else {
            Some(target)
        }
    }

    /// Cog-parity special selector `==`/`~~` (identity): `Some(neq)` when
    /// send site `ic_idx` of `holder` is the one-argument identity test.
    /// No feedback consulted and no target check — the frontend pins
    /// identity as non-redefinable (codegen.rs compiles the nil-guard
    /// family on that exact invariant), so a raw pointer compare is the
    /// send's semantics for EVERY receiver. x86-64 only until the AArch64
    /// emitter grows its `RefCmpVal` arm.
    fn ref_eq_kind(&self, holder: MethodOop, ic_idx: u16) -> Option<bool> {
        if !cfg!(target_arch = "x86_64") {
            return None;
        }
        let ic = InterpreterIc::at(holder, ic_idx);
        if ic.argc() != 1 {
            return None;
        }
        match ic.selector().as_string().as_str() {
            "==" => Some(false),
            "~~" => Some(true),
            _ => None,
        }
    }

    /// Boolean-`not` speculation gate: the site's evidence is boolean-only
    /// (Empty, Mono on True/False, or Poly over exactly those two) AND the
    /// live `True>>not` / `False>>not` are the canonical single-op flips
    /// (`^false` / `^true`). The caller records the two inline deps. The
    /// evidence gate is what makes the speculation storm-proof: a wrong
    /// guess traps once, the trap warms the IC with the non-boolean klass,
    /// and the recompile (customized hash sees it) declines here.
    fn bool_not_speculatable(&self, holder: MethodOop, ic_idx: u16) -> bool {
        if !cfg!(target_arch = "x86_64") {
            return false;
        }
        let ic = InterpreterIc::at(holder, ic_idx);
        if ic.argc() != 0 || ic.selector().as_string() != "not" {
            return false;
        }
        let u = &self.vm.universe;
        let (tk, fk) = (u.true_klass.oop().raw(), u.false_klass.oop().raw());
        let guard = ic.guard();
        let boolean_only = if guard.raw() == u.nil_obj.raw()
            || guard.raw() == tk
            || guard.raw() == fk
        {
            true
        } else if matches!(
            crate::interpreter::ic::ic_state(holder, ic_idx),
            crate::interpreter::ic::IcState::Poly(_)
        ) {
            match crate::oops::wrappers::ArrayOop::try_from(ic.target()) {
                Some(pairs) => (0..pairs.len() / 2).all(|i| {
                    let k = pairs.at(2 * i).raw();
                    k == tk || k == fk || k == u.nil_obj.raw()
                }),
                None => false,
            }
        } else {
            false
        };
        if !boolean_only {
            return false;
        }
        // Both live implementations must BE the canonical flips — a
        // redefined True>>not must never be constant-folded to the flip.
        let canonical = |k: KlassOop, want_true_push: bool| -> bool {
            let Some(m) = crate::compiler::feedback::resolve_method_ro(self.vm, k, ic.selector())
            else {
                return false;
            };
            if m.primitive() != 0 {
                return false;
            }
            use crate::bytecode::opcode::decode_at;
            let len = m.bytecode_len();
            let (i0, n0) = decode_at(m, 0);
            let push_ok = match i0 {
                Instr::PushTrue => want_true_push,
                Instr::PushFalse => !want_true_push,
                _ => false,
            };
            if !push_ok || n0 >= len {
                return false;
            }
            let (i1, n1) = decode_at(m, n0);
            matches!(i1, Instr::ReturnTos) && n1 >= len
        };
        canonical(self.vm.universe.true_klass, false) && canonical(self.vm.universe.false_klass, true)
    }

    /// S15 (the sieve fix): is send site `ic_idx` a mono-Array-guarded
    /// `at:` / `at:put:` primitive — the intrinsify-to-`ArrayAt`/
    /// `ArrayAtPut` class? Exactly `is_smi_inlinable`'s shape, one klass
    /// over: the IC observed only Array receivers and the target is the
    /// indexable-oops primitive, so the element access compiles to inline
    /// guards + one memory op instead of a per-element c2i round trip.
    fn array_op_kind(&self, method: MethodOop, ic_idx: u16) -> Option<bool /* is_put */> {
        let ic = InterpreterIc::at(method, ic_idx);
        if ic.guard().raw() != self.vm.universe.array_klass.oop().raw() {
            return None;
        }
        let target = mono_target_method(self.vm, &ic, self.vm.universe.array_klass)?;
        match target.primitive() {
            26 => Some(false), // at:
            27 => Some(true),  // at:put:
            _ => None,
        }
    }

    fn is_smi_inlinable(&self, ic_idx: u16) -> bool {
        let ic = InterpreterIc::at(self.method, ic_idx);
        // Accepts warm Mono-smi AND cold Empty (the Cog-parity
        // speculation) — see `smi_special_target`.
        let Some(target) = smi_special_target(self.vm, &ic) else {
            return false;
        };
        crate::compiler::driver::SMI_INLINE.contains(&target.primitive())
    }

    /// Float fast-path: `is_smi_inlinable`'s Double twin — a mono-Double-
    /// guarded site whose resolved method is a `DOUBLE_INLINE` arithmetic/
    /// compare primitive fuses to `FUnbox`/`FArith`/`FBox` instead of a
    /// `CallSend`. Resolves via (guard klass, selector) rather than the raw
    /// IC target: a warm Double site's mono target is usually a tiered-up
    /// smi NmethodId, not a MethodOop (see `classify_double_send`).
    fn is_double_inlinable(&self, ic_idx: u16) -> bool {
        let ic = InterpreterIc::at(self.method, ic_idx);
        if ic.guard().raw() != self.vm.universe.double_klass.oop().raw() {
            return false;
        }
        let Some(target) = crate::compiler::feedback::resolve_method_ro(
            self.vm,
            self.vm.universe.double_klass,
            ic.selector(),
        ) else {
            return false;
        };
        crate::compiler::driver::DOUBLE_INLINE.contains(&target.primitive())
    }

    /// SIMD NEON fast-path (`docs/SIMD.md`): the vector analog of
    /// `is_double_inlinable`. Returns `Some(kind)` when this send site is
    /// mono-guarded on a vector klass (`Float64x2`/`Float32x4`) AND resolves to
    /// one of that kind's four elementwise arithmetic primitives — the sends
    /// that fuse to a single guarded `Ir::VecArith` (NEON `f… v.<arr>` + box).
    /// Resolves via (guard klass, selector) rather than the raw IC target, for
    /// the same reason as `is_double_inlinable`: a warm site's mono target is
    /// usually a tiered-up smi NmethodId, not a MethodOop.
    fn vec_send_kind(&self, ic_idx: u16) -> Option<VecKind> {
        let ic = InterpreterIc::at(self.method, ic_idx);
        let guard = ic.guard().raw();
        let (kind, klass) = if guard == self.vm.universe.float64x2_klass.oop().raw() {
            (VecKind::F64x2, self.vm.universe.float64x2_klass)
        } else if guard == self.vm.universe.float32x4_klass.oop().raw() {
            (VecKind::F32x4, self.vm.universe.float32x4_klass)
        } else if guard == self.vm.universe.int32x4_klass.oop().raw() {
            (VecKind::I32x4, self.vm.universe.int32x4_klass)
        } else {
            return None;
        };
        let target = crate::compiler::feedback::resolve_method_ro(self.vm, klass, ic.selector())?;
        vec_arith_op(kind, target.primitive()).map(|_| kind)
    }

    /// S11 D7: is this send site an inline-allocatable `X basicNew`? Returns
    /// `Some((klass, size_words))` when: the `receiver` vreg is a known
    /// compile-time class constant `X` (from a `push_global`, tracked in
    /// `const_class`); the send takes no arguments; the site's mono target
    /// really is the `basicNew` primitive (guards against a class that
    /// overrides `basicNew` — a Poly/Mega/non-method target fails the
    /// `MethodOop::try_from` exactly as `is_smi_inlinable` relies on); and
    /// `X` is a fixed-size `Format::Slots` class small enough for the inline
    /// fast path's 12-bit `add`/`str` immediates. Anything else stays an
    /// ordinary generic `basicNew` send.
    ///
    /// gc_alloc_gap cost 1 (customized `self basicNew`): the second receiver
    /// shape is `self` under a CLASS-SIDE customization — `rcvr_klass` is a
    /// metaclass, whose sole instance (the class object every `self basicNew`
    /// here instantiates) is a compile-time constant recovered from the
    /// global namespace. This path is statically sound exactly like
    /// self-devirt (the nmethod's entry guard proved self's klass), so the
    /// target comes from `resolve_method_ro` with NO IC warmth required —
    /// the caller records a `(rcvr_klass, selector)` inline dep so a later
    /// `basicNew` override on the class side invalidates the nmethod. This
    /// is the allocation-heavy path Cog wins on: `Point x: y:`-style
    /// class-side constructors stop paying the CallSend → prim-shim →
    /// `rt_call_primitive` Rust crossing per object and take the same
    /// inline eden bump the `X basicNew` constant form always took.
    fn alloc_site_klass(&mut self, ic_idx: u16, receiver: VReg) -> Option<(KlassOop, u32)> {
        let self_klass = if self.method.is_block() {
            // A block's `self` is the HOME receiver (any klass), not the
            // customization key — same refusal as `devirt_self_target`.
            None
        } else {
            Some(self.rcvr_klass)
        };
        self.alloc_site_klass_on(self.method, ic_idx, receiver, self.self_vreg, self_klass)
    }

    /// The `_on` twin (same pattern as `is_smi_inlinable_on`): the alloc
    /// gate against an arbitrary method's own IC table, usable on an
    /// INLINED callee's sites — `holder_self`/`holder_self_klass` are that
    /// body's own `self` vreg and its statically-proven klass (the splice
    /// guard klass; the root passes its customization klass). Records the
    /// `(metaclass, selector)` inline dep itself when the statically-
    /// resolved self form fires, so every caller stays invalidation-sound.
    fn alloc_site_klass_on(
        &mut self,
        holder: MethodOop,
        ic_idx: u16,
        receiver: VReg,
        holder_self: VReg,
        holder_self_klass: Option<KlassOop>,
    ) -> Option<(KlassOop, u32)> {
        let ic = InterpreterIc::at(holder, ic_idx);
        if ic.argc() != 0 {
            return None;
        }
        let klass = if let Some(&k) = self.const_class.get(&receiver.0) {
            // Constant-receiver form: feedback-gated (warm mono IC whose
            // target is the basicNew primitive), exactly as before.
            let guard = crate::oops::wrappers::KlassOop::try_from(ic.guard())?;
            let target = mono_target_method(self.vm, &ic, guard)?;
            if target.primitive() != PRIM_BASIC_NEW {
                return None;
            }
            k
        } else {
            // Statically-known-self form: `self basicNew` where this body's
            // receiver klass is a metaclass proven at compile time (entry
            // guard or splice guard) — the sole instance is the class every
            // such send instantiates. Resolution is against the LIVE world
            // (like self-devirt), no IC warmth required.
            if receiver != holder_self {
                return None;
            }
            let meta = holder_self_klass?;
            let sole = crate::runtime::globals::metaclass_sole_instance(self.vm, meta)?;
            let target =
                crate::compiler::feedback::resolve_method_ro(self.vm, meta, ic.selector())?;
            if target.primitive() != PRIM_BASIC_NEW {
                return None;
            }
            self.record_inline_dep(meta, ic.selector());
            sole
        };
        if !matches!(klass.format(), crate::oops::klass::Format::Slots) {
            return None;
        }
        let size_words = klass.non_indexable_size();
        // Header (2 words) .. the 12-bit immediate ceiling emit_alloc's own
        // debug_assert enforces; a giant class stays a generic send.
        if size_words < crate::oops::layout::HEADER_WORDS
            || size_words * crate::oops::layout::WORD_SIZE >= 4096
        {
            return None;
        }
        Some((klass, size_words as u32))
    }

    /// S14 step 4b: splice a **single-block straight-line leaf** callee inline.
    ///
    /// `receiver`/`args` are the send's OPERAND vregs (already popped by the
    /// caller); they alias the callee's `self`/argument slots directly, no
    /// stores (A2 scope-splicing rule: "callee args map to the vregs holding the
    /// send's operands"). `guard_klass` is the observed receiver klass whose
    /// `expect`-literal the guard compares against, `guard_shape` the test to
    /// emit. `selector` is the send's own selector (for the inline dependency).
    ///
    /// Returns `Some(result_vreg)` — the vreg holding the inlined body's return
    /// value, which the caller pushes onto its operand stack — after appending
    /// the guard + spliced body to `code`, minting the cold trap block (and its
    /// reexecute deopt site), and recording the inline dependency. Returns
    /// `None` (splice DECLINED — the caller falls back to a plain `Call`) when
    /// the callee is not a single straight-line block ending in a return, or
    /// contains an opcode this narrow step doesn't splice.
    ///
    /// No safepoint is recorded for the body: a leaf has no send / no smi-inline
    /// overflow trap / no alloc, so no deopt can occur inside it (the ONLY deopt
    /// is the guard's own cold trap, which lives in the `fail` block). The
    /// caller therefore records NO `CallSiteInfo`/`site_feedback`/`IcSite` for
    /// this site.
    #[allow(clippy::too_many_arguments)]
    fn try_inline_leaf(
        &mut self,
        callee: MethodOop,
        guard: Option<(KlassOop, GuardShape)>,
        dep_klass: KlassOop,
        selector: SymbolOop,
        receiver: VReg,
        args: &[VReg],
        send_bci: usize,
        pre_pop_stack: &[VReg],
        code: &mut Vec<Ir>,
    ) -> Option<VReg> {
        // MINIMUM viable scope: the callee must be a SINGLE straight-line block
        // ending in a return, and its arity must match what we popped. A
        // multi-block leaf (a branch/loop with no sends) or an arity mismatch is
        // out of this narrow step's scope — decline, and the caller does a plain
        // Call. Validate against the CFG BEFORE emitting anything.
        let callee_cfg = crate::compiler::decode::decode(callee);
        if callee_cfg.blocks.len() != 1
            || !matches!(
                callee_cfg.blocks[0].terminator,
                crate::compiler::decode::Terminator::Return
            )
            || callee.argc() != args.len()
        {
            return None;
        }
        // DRY-RUN: confirm every callee opcode is one this narrow step splices
        // BEFORE emitting the guard/cold block/body — a mid-splice decline would
        // leave a guard dominating a half-built body while the caller falls back
        // to a `Call`, corrupting the operand-stack discipline. If any opcode is
        // unsupported, decline cleanly here having emitted NOTHING.
        if !leaf_body_is_spliceable(callee) {
            return None;
        }

        // The callee's unified arg/temp slots: args 0..argc alias the send's
        // operand vregs (NO stores — A2's scope-splicing rule); the callee's own
        // temps argc..argc+ntemps become fresh nil vregs. `callee_self` is the
        // send's receiver vreg.
        let callee_self = receiver;
        let mut callee_slots: Vec<VReg> = Vec::with_capacity(callee.argc() + callee.ntemps());
        callee_slots.extend_from_slice(args);

        // Emit the guard FIRST so it dominates the whole spliced body. On a
        // klass MISMATCH it branches to `cold_id` (a single-op UncommonTrap,
        // reexecute=true at the send's own bci, recorded stack = the operand
        // stack BEFORE the send's pops, so receiver + args are still present) —
        // identical to the S14 step-3 trap: a wrong receiver deopts and
        // re-executes the send generically. On a MATCH it falls through into the
        // body appended below (same block, no branch — the leaf is straight
        // line). `expect` is the observed klass, a moving-GC-tracked Oop lit
        // (read only by KlassTest; SmiTest ignores it).
        if let Some((guard_klass, guard_shape)) = guard {
            let expect = self
                .pool
                .intern(guard_klass.oop().raw(), Some(RelocKind::Oop));
            let cold_id = self.fresh_block_id();
            self.finish_block(IrBlock {
                id: cold_id,
                bci: send_bci,
                code: vec![Ir::UncommonTrap { bci: send_bci }],
                entry_stack: Vec::new(),
                deopt_sites: vec![(
                    0,
                    DeoptRaw {
                        stack: pre_pop_stack.to_vec(),
                        bci: send_bci,
                        kind: SafepointKind::UncommonTrap,
                        reexecute: true,
                        stack_closures: Vec::new(),
                        inline: None,
                    },
                )],
            });
            code.push(Ir::GuardKlass {
                obj: callee_self,
                expect,
                fail: cold_id,
                kind: guard_shape,
            });
        }

        // Fresh nil temps for the callee's own locals (after the guard, so a
        // wrong-klass receiver traps before any body op runs).
        let nil_lit = self
            .pool
            .intern(self.vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
        for _ in 0..callee.ntemps() {
            let t = self.fresh(true);
            code.push(Ir::ConstPool {
                dst: t,
                lit: nil_lit,
            });
            callee_slots.push(t);
        }

        // Translate the callee's straight-line body onto a fresh operand stack.
        // A leaf has no send/smi-inline/alloc, so only value-shuffling and
        // field/return opcodes can appear. `return_tos` → the send's result is
        // the callee's TOS; `return_self` → the receiver vreg. The `expect`
        // (single-block Return terminator) means exactly one return terminates.
        let mut cstack: Vec<VReg> = Vec::new();
        let mut result: Option<VReg> = None;
        let len = callee.bytecode_len();
        let mut bci = 0;
        while bci < len {
            let (instr, next) = decode_at(callee, bci);
            match instr {
                Instr::PushSelf => cstack.push(callee_self),
                Instr::PushNil => {
                    cstack.push(self.push_well_known(self.vm.universe.nil_obj.raw(), code))
                }
                Instr::PushTrue => {
                    cstack.push(self.push_well_known(self.vm.universe.true_obj.raw(), code))
                }
                Instr::PushFalse => {
                    cstack.push(self.push_well_known(self.vm.universe.false_obj.raw(), code))
                }
                Instr::PushSmi(v) => {
                    let dst = self.fresh(true);
                    code.push(Ir::ConstSmi {
                        dst,
                        value: v as i64,
                    });
                    cstack.push(dst);
                }
                Instr::PushLiteral(idx) => {
                    let lit_oop = callee.literals().at(idx as usize);
                    cstack.push(self.push_well_known(lit_oop.raw(), code));
                }
                Instr::PushGlobal(idx) => {
                    let assoc = callee.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                    });
                    cstack.push(dst);
                }
                // S15 (DeltaBlue port finding): a classVar/global STORE in an
                // inlined non-leaf body — `is_inline_eligible_nonleaf` always
                // admitted the opcode but this walker had no arm, so any
                // callee like `reset [ Current := self new ]` aborted on the
                // unreachable! below the moment it inlined. Mirrors the root
                // arm exactly (association word 1 = value, barrier on).
                Instr::StoreGlobalPop(idx) => {
                    let assoc = callee.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let val = cstack
                        .pop()
                        .expect("inline splice: store_global_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::PushTemp(n) => {
                    // Fresh copy (same v1 rule as the main path): the slot may be
                    // reassigned later on this path.
                    let dst = self.fresh(true);
                    code.push(Ir::Move {
                        dst,
                        src: callee_slots[n as usize],
                    });
                    cstack.push(dst);
                }
                Instr::StoreTemp(n) => {
                    let tos = *cstack
                        .last()
                        .expect("inline splice: store_temp on empty stack");
                    code.push(Ir::Move {
                        dst: callee_slots[n as usize],
                        src: tos,
                    });
                }
                Instr::StoreTempPop(n) => {
                    let v = cstack
                        .pop()
                        .expect("inline splice: store_temp_pop on empty stack");
                    code.push(Ir::Move {
                        dst: callee_slots[n as usize],
                        src: v,
                    });
                }
                Instr::PushInstvar(n) => {
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: callee_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                    });
                    cstack.push(dst);
                }
                Instr::StoreInstvarPop(n) => {
                    let val = cstack
                        .pop()
                        .expect("inline splice: store_instvar_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: callee_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::Pop => {
                    cstack.pop().expect("inline splice: pop on empty stack");
                }
                Instr::Dup => {
                    let tos = *cstack.last().expect("inline splice: dup on empty stack");
                    cstack.push(tos);
                }
                Instr::ReturnTos => {
                    result = Some(
                        cstack
                            .pop()
                            .expect("inline splice: return_tos on empty stack"),
                    );
                    break;
                }
                Instr::ReturnSelf => {
                    result = Some(callee_self);
                    break;
                }
                // Pre-validated by `leaf_body_is_spliceable` above — no opcode
                // outside the supported set (or the multi-block case) reaches
                // here, so a decline past the guard's emission is impossible.
                other => unreachable!(
                    "inline splice: {other:?} passed leaf_body_is_spliceable but has no arm"
                ),
            }
            bci = next;
        }
        let result = result.expect("inline splice: single-return leaf must produce a result");

        self.budget_commit(callee);
        self.record_inline_dep(dep_klass, selector);
        Some(result)
    }

    /// S14 step 4c: splice a **single-block straight-line NON-leaf** callee
    /// inline — a callee that HAS its own sends (so it has in-body safepoints
    /// and can deopt), extending [`try_inline_leaf`] with a `Send` arm. The
    /// crux of step 4c: each in-body safepoint records a NESTED deopt scope
    /// (the inlined callee's own receiver/slots/method + a `SenderLink` back to
    /// the caller), so a deopt at one of them rebuilds BOTH the inlined frame
    /// AND the caller frame from the ONE physical compiled frame — the first
    /// time the depth-N materializer runs at depth > 1.
    ///
    /// Same operand/scope-splicing rules as the leaf splice: `receiver`/`args`
    /// alias the send's operand vregs (no stores), temps become fresh nil vregs
    /// with canonical spill homes (S12 spill-all → every entity has a frame slot
    /// the deopt materializer reads via `ValueLoc::FrameSlot`). The KEY new
    /// invariant this builds: the `InlineSite` on each in-body `DeoptRaw`
    /// carries `sender_bci` (the inlined send's bci in the CALLER) and
    /// `caller_pending_stack` (the caller's operand stack BELOW the send's
    /// receiver+args) — exactly what M6 uses to rebuild the caller frame's
    /// resume point + operand stack.
    ///
    /// Appends the guard + spliced body to `code` and records in-body deopt
    /// safepoints into `deopt`. Returns:
    ///   - `Some(NonLeafOutcome::Value(vreg))` — the body ran to its return; the
    ///     caller pushes `vreg` and continues,
    ///   - `Some(NonLeafOutcome::Trapped)` — an in-body send was cold (Untaken)
    ///     and became a terminating `UncommonTrap`; the caller must finish the
    ///     current block here (set `trapped`), nothing after is reachable,
    ///   - `None` — the callee is not the single straight-line shape this
    ///     splices (the caller falls back to a plain `Call`).
    #[allow(clippy::too_many_arguments)]
    fn try_inline_nonleaf(
        &mut self,
        callee: MethodOop,
        guard: Option<(KlassOop, GuardShape)>,
        dep_klass: KlassOop,
        selector: SymbolOop,
        receiver: VReg,
        args: &[VReg],
        send_bci: usize,
        pre_pop_stack: &[VReg],
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
    ) -> Option<NonLeafOutcome> {
        // EXACT arm-set validation, not a caller courtesy: the body walk below
        // cannot decline once the guard is emitted (its own `unreachable!`
        // comment), and callers' budget gates are NOT shape-exact — e.g. a
        // sendless `^[aBlock]` body is `is_leaf` (no sends) yet the leaf
        // splicer declines it (PushClosure), which used to land it HERE and
        // abort on the closure opcode (review finding — reachable through the
        // warm-mono feedback path too, not just step-5 devirt).
        if !crate::compiler::inline::is_inline_eligible_nonleaf(callee) {
            return None;
        }
        // Same MINIMUM viable shape as the leaf splice: a single straight-line
        // block ending in a return, arity matching. Validate against the CFG
        // BEFORE emitting anything. `is_inline_eligible_nonleaf` already proved
        // the single-block/return/no-super shape at decision time, but the
        // arity check is site-specific, so re-confirm here.
        let callee_cfg = crate::compiler::decode::decode(callee);
        if callee_cfg.blocks.len() != 1
            || !matches!(
                callee_cfg.blocks[0].terminator,
                crate::compiler::decode::Terminator::Return
            )
            || callee.argc() != args.len()
        {
            return None;
        }

        // The inlined callee's own compiled MethodOop, interned (idempotent,
        // deduped) into the SAME pool as the root method's — the reconstructed
        // inlined frame's FRAME_METHOD names THIS method, not the caller's.
        let callee_pool_ix = self.pool.intern(callee.oop().raw(), Some(RelocKind::Oop)).0;

        // The caller's operand stack BELOW the inlined send's receiver+args:
        // frozen for the whole inlined extent as the `SenderLink.pending_stack`.
        // `pre_pop_stack` is `[...below, receiver, arg0, ..]`; strip the
        // receiver (1) + args (argc) off the top.
        let n_operands = args.len() + 1;
        debug_assert!(
            pre_pop_stack.len() >= n_operands,
            "inline splice: pre-pop stack shorter than the send's operands"
        );
        let caller_pending_stack = pre_pop_stack[..pre_pop_stack.len() - n_operands].to_vec();

        // Callee slots: args alias the send operands (no stores); temps become
        // fresh nil vregs. `receiver` is the callee's `self`.
        let callee_self = receiver;

        // S14 step 5: the statically-proven klass of `callee_self` — the
        // guard klass when this splice is receiver-guarded, or the
        // customization klass when the receiver is the root method's own
        // `self` (guard-free super/self splices, whose receiver the entry
        // guard already checked). `None` cannot occur today (every
        // guard-free splice passes a self receiver) but stays sound if it
        // ever does: inner self-sends just keep their feedback lowering.
        let callee_self_klass: Option<KlassOop> = match guard {
            Some((gk, _)) => Some(gk),
            None if receiver == self.self_vreg => Some(self.rcvr_klass),
            None => None,
        };
        let mut callee_slots: Vec<VReg> = Vec::with_capacity(callee.argc() + callee.ntemps());
        callee_slots.extend_from_slice(args);

        // The guard fronts the whole thing, exactly as the leaf splice: on a
        // klass MISMATCH branch to a single-op reexecute trap (re-executes the
        // send generically); on a MATCH fall through into the body. `None` for
        // a STATICALLY-resolved target (a super send) — no guard needed.
        if let Some((guard_klass, guard_shape)) = guard {
            let expect = self
                .pool
                .intern(guard_klass.oop().raw(), Some(RelocKind::Oop));
            let cold_id = self.fresh_block_id();
            self.finish_block(IrBlock {
                id: cold_id,
                bci: send_bci,
                code: vec![Ir::UncommonTrap { bci: send_bci }],
                entry_stack: Vec::new(),
                deopt_sites: vec![(
                    0,
                    DeoptRaw {
                        stack: pre_pop_stack.to_vec(),
                        bci: send_bci,
                        kind: SafepointKind::UncommonTrap,
                        reexecute: true,
                        stack_closures: Vec::new(),
                        // The guard's cold trap re-executes the WHOLE send in
                        // the CALLER, so it deopts to the CALLER scope (depth-1,
                        // `inline: None`) — NOT the inlined scope. Only
                        // safepoints INSIDE the body carry an `InlineSite`.
                        inline: None,
                    },
                )],
            });
            code.push(Ir::GuardKlass {
                obj: callee_self,
                expect,
                fail: cold_id,
                kind: guard_shape,
            });
        }

        // Fresh nil temps for the callee's own locals (after the guard).
        let nil_lit = self
            .pool
            .intern(self.vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
        for _ in 0..callee.ntemps() {
            let t = self.fresh(true);
            code.push(Ir::ConstPool {
                dst: t,
                lit: nil_lit,
            });
            callee_slots.push(t);
        }

        // Build the InlineSite prototype (shared shape for every in-body
        // safepoint — the slots/receiver/method/sender are constant across the
        // extent; only the operand `stack` varies per site). `sender_bci` is
        // the CALLER's bci of the inlined send.
        let inline_proto = InlineSite {
            method_pool_ix: callee_pool_ix,
            receiver: callee_self,
            slots: callee_slots.clone(),
            sender_bci: send_bci as u16,
            caller_pending_stack: caller_pending_stack.clone(),
            // A step-4c METHOD inline, not a spliced block.
            is_block: false,
            parent: None,
            slot_closures: Vec::new(),
        };

        // Translate the callee's straight-line body onto a fresh operand stack.
        // Value-shuffle / field / return opcodes mirror the leaf splice; the
        // NEW arm is `Send` — a compiled `CallSend` (or a step-3 trap for a cold
        // IC) recording an in-body deopt scope. Do NOT recurse into the inlinee
        // (depth-1 this step): a nested send stays a plain `Call`.
        let mut cstack: Vec<VReg> = Vec::new();
        let mut result: Option<VReg> = None;
        let mut trapped = false;
        let len = callee.bytecode_len();
        let mut bci = 0;
        while bci < len {
            let (instr, next) = decode_at(callee, bci);
            match instr {
                Instr::PushSelf => cstack.push(callee_self),
                Instr::PushNil => {
                    cstack.push(self.push_well_known(self.vm.universe.nil_obj.raw(), code))
                }
                Instr::PushTrue => {
                    cstack.push(self.push_well_known(self.vm.universe.true_obj.raw(), code))
                }
                Instr::PushFalse => {
                    cstack.push(self.push_well_known(self.vm.universe.false_obj.raw(), code))
                }
                Instr::PushSmi(v) => {
                    let dst = self.fresh(true);
                    code.push(Ir::ConstSmi {
                        dst,
                        value: v as i64,
                    });
                    cstack.push(dst);
                }
                Instr::PushLiteral(idx) => {
                    let lit_oop = callee.literals().at(idx as usize);
                    cstack.push(self.push_well_known(lit_oop.raw(), code));
                }
                Instr::PushGlobal(idx) => {
                    let assoc = callee.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                    });
                    // Mirror the root arm: remember class-valued globals so an
                    // in-body `X basicNew` can fuse to `Ir::Alloc` below —
                    // without this the SPLICED copy of a constructor demotes
                    // its allocation to a generic send.
                    let value = crate::oops::wrappers::MemOop::try_from(assoc)
                        .map(|a| a.body_oop(1))
                        .unwrap_or(assoc);
                    if let Some(klass) = KlassOop::try_from(value) {
                        self.const_class.insert(dst.0, klass);
                    }
                    cstack.push(dst);
                }
                // S15 (DeltaBlue port finding): classVar/global STORE — the
                // eligibility gate always admitted the opcode; this walker
                // lacked the arm (its leaf twin above got it first by
                // mistake, which is harmless there — leaf bodies with stores
                // are legal too if `leaf_body_is_spliceable` ever admits
                // them). Mirrors the root arm: association word 1 = value,
                // barrier on.
                Instr::StoreGlobalPop(idx) => {
                    let assoc = callee.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let val = cstack
                        .pop()
                        .expect("inline splice: store_global_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::PushTemp(n) => {
                    let dst = self.fresh(true);
                    code.push(Ir::Move {
                        dst,
                        src: callee_slots[n as usize],
                    });
                    cstack.push(dst);
                }
                Instr::StoreTemp(n) => {
                    let tos = *cstack
                        .last()
                        .expect("inline splice: store_temp on empty stack");
                    code.push(Ir::Move {
                        dst: callee_slots[n as usize],
                        src: tos,
                    });
                }
                Instr::StoreTempPop(n) => {
                    let v = cstack
                        .pop()
                        .expect("inline splice: store_temp_pop on empty stack");
                    code.push(Ir::Move {
                        dst: callee_slots[n as usize],
                        src: v,
                    });
                }
                Instr::PushInstvar(n) => {
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: callee_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                    });
                    cstack.push(dst);
                }
                Instr::StoreInstvarPop(n) => {
                    let val = cstack
                        .pop()
                        .expect("inline splice: store_instvar_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: callee_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::Pop => {
                    cstack.pop().expect("inline splice: pop on empty stack");
                }
                Instr::Dup => {
                    let tos = *cstack.last().expect("inline splice: dup on empty stack");
                    cstack.push(tos);
                }
                Instr::Send { ic, super_ } => {
                    // `is_inline_eligible_nonleaf` rejected super sends, so this
                    // is always an ordinary dynamic send.
                    debug_assert!(!super_, "non-leaf splice: super send should be gated out");
                    let inner_ic = InterpreterIc::at(callee, ic);
                    let inner_argc = inner_ic.argc();
                    let inner_sel = inner_ic.selector();
                    // The bci of THIS in-body send within the callee — the
                    // reconstructed inlined frame's own resume bci on a deopt.
                    let inner_bci = bci;
                    // S14 opt (post-step-8): FUSE the inlined body's own
                    // mono-smi ops instead of emitting generic `CallSend`s —
                    // the difference between an inlined loop being tight
                    // machine code and a chain of stub calls. Same lowering as
                    // the root's smi-inline path minus branch fusion; the fail
                    // edge is an InlineSite-chained reexecute trap (`a`/`b`
                    // still on the recorded stack).
                    // Cog-parity special selectors in the SPLICED body —
                    // same lowerings as the root arm (identity: unguarded;
                    // not: boolean-guarded flip with an inlined-frame trap).
                    if let Some(neq) = self.ref_eq_kind(callee, ic) {
                        let b_op = cstack.pop().expect("ref-eq fuse: missing rhs");
                        let a_op = cstack.pop().expect("ref-eq fuse: missing lhs");
                        let dst = self.fresh(true);
                        code.push(Ir::RefCmpVal {
                            dst,
                            a: a_op,
                            b: b_op,
                            neq,
                        });
                        cstack.push(dst);
                        bci = next;
                        continue;
                    }
                    if self.bool_not_speculatable(callee, ic) {
                        self.record_inline_dep(self.vm.universe.true_klass, inner_sel);
                        self.record_inline_dep(self.vm.universe.false_klass, inner_sel);
                        let src = cstack.pop().expect("bool-not fuse: missing receiver");
                        let mut reexec = cstack.clone();
                        reexec.push(src);
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let dst = self.fresh(true);
                        code.push(Ir::BoolNot { dst, src, fail });
                        cstack.push(dst);
                        bci = next;
                        continue;
                    }
                    if let Some(is_put) = array_op_kind_on(self.vm, callee, ic) {
                        // S15: in-body array intrinsic (see the root arm).
                        let val = if is_put {
                            Some(cstack.pop().expect("array fuse: missing value"))
                        } else {
                            None
                        };
                        let idx_op = cstack.pop().expect("array fuse: missing index");
                        let arr_op = cstack.pop().expect("array fuse: missing receiver");
                        let mut reexec = cstack.clone();
                        reexec.push(arr_op);
                        reexec.push(idx_op);
                        if let Some(v) = val {
                            reexec.push(v);
                        }
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let klass = self.pool.intern(
                            self.vm.universe.array_klass.oop().raw(),
                            Some(RelocKind::Oop),
                        );
                        let dst = self.fresh(true);
                        match val {
                            Some(v) => code.push(Ir::ArrayAtPut {
                                dst,
                                arr: arr_op,
                                idx: idx_op,
                                val: v,
                                klass,
                                fail,
                            }),
                            None => code.push(Ir::ArrayAt {
                                dst,
                                arr: arr_op,
                                idx: idx_op,
                                klass,
                                fail,
                            }),
                        }
                        cstack.push(dst);
                        bci = next;
                        continue;
                    }
                    if is_smi_inlinable_on(self.vm, callee, ic)
                        && !smi_fuse_arg_is_pooled(cstack.as_slice(), code.as_slice())
                    {
                        debug_assert_eq!(inner_argc, 1, "SMI_INLINE ops are all binary");
                        let b_op = cstack.pop().expect("smi fuse: missing rhs");
                        let a_op = cstack.pop().expect("smi fuse: missing lhs");
                        let mut reexec = cstack.clone();
                        reexec.push(a_op);
                        reexec.push(b_op);
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let dst = self.fresh(true);
                        match classify_smi_send(self.vm, callee, ic) {
                            SmiSendKind::Arith(op) => code.push(Ir::SmiArith {
                                op,
                                dst,
                                a: a_op,
                                b: b_op,
                                fail,
                            }),
                            SmiSendKind::Cmp(op) => code.push(Ir::SmiCmpVal {
                                op,
                                dst,
                                a: a_op,
                                b: b_op,
                                fail,
                            }),
                        }
                        cstack.push(dst);
                        bci = next;
                        continue;
                    }
                    // Float fast-path in the splice: `is_double_inlinable`'s
                    // `_on` twin, same pattern as the smi/array twins above.
                    if is_double_inlinable_on(self.vm, callee, ic) {
                        debug_assert_eq!(inner_argc, 1, "DOUBLE_INLINE ops are all binary");
                        // b4d55a8's storm guard, ported to the splice: a
                        // compile-time smi arg (`x < 0` in an inlined body) can
                        // never FUnbox — emit the coerced FConst instead,
                        // byte-identical to the interpreter fallback's asDouble.
                        let arg_const_smi: Option<i64> = match (cstack.last(), code.last()) {
                            (Some(&top), Some(Ir::ConstSmi { dst, value })) if *dst == top => {
                                Some(*value)
                            }
                            _ => None,
                        };
                        let b_op = cstack.pop().expect("double fuse: missing rhs");
                        let a_op = cstack.pop().expect("double fuse: missing lhs");
                        let mut reexec = cstack.clone();
                        reexec.push(a_op);
                        reexec.push(b_op);
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let ua = self.fresh_fp();
                        let ub = self.fresh_fp();
                        code.push(Ir::FUnbox {
                            dst: ua,
                            src: a_op,
                            fail,
                        });
                        match arg_const_smi {
                            Some(v) => code.push(Ir::FConst {
                                dst: ub,
                                bits: (v as f64).to_bits(),
                            }),
                            None => code.push(Ir::FUnbox {
                                dst: ub,
                                src: b_op,
                                fail,
                            }),
                        }
                        let dst = self.fresh(true);
                        match classify_double_send(self.vm, callee, ic) {
                            DoubleSendKind::Arith(op) => {
                                let fd = self.fresh_fp();
                                code.push(Ir::FArith {
                                    op,
                                    dst: fd,
                                    a: ua,
                                    b: ub,
                                });
                                code.push(Ir::FBox { dst, src: fd });
                            }
                            DoubleSendKind::Cmp(op) => {
                                code.push(Ir::FCmpVal {
                                    op,
                                    dst,
                                    a: ua,
                                    b: ub,
                                });
                            }
                        }
                        cstack.push(dst);
                        bci = next;
                        continue;
                    }

                    // gc_alloc_gap cost 1, splice half: an in-body `X basicNew`
                    // / `self basicNew` fuses to the same inline `Ir::Alloc`
                    // the root arm emits. Without this, splicing a tiny
                    // constructor DEMOTES its allocation from inline eden bump
                    // to a generic send — and on hosts where `basicNew` has no
                    // compiled form (prim 23 is not shimmable on x64), to a
                    // per-object c2i interpreter round trip: the Windows
                    // alloc-bench pathology (24M interpreted `basicNew`s).
                    if inner_argc == 0 {
                        if let Some(&recv) = cstack.last() {
                            if let Some((aklass, size_words)) = self.alloc_site_klass_on(
                                callee,
                                ic,
                                recv,
                                callee_self,
                                callee_self_klass,
                            ) {
                                cstack.pop();
                                // Reexecute deopt: the interpreter re-runs the
                                // `basicNew` send, so the recorded stack still
                                // carries the receiver (root-arm rule), and the
                                // inlined-frame proto reconstructs the callee
                                // activation around it.
                                let mut reexec = cstack.clone();
                                reexec.push(recv);
                                deopt.push((
                                    code.len() as u32,
                                    DeoptRaw {
                                        stack: reexec,
                                        bci: inner_bci,
                                        kind: SafepointKind::Alloc,
                                        reexecute: true,
                                        stack_closures: Vec::new(),
                                        inline: Some(inline_proto.clone()),
                                    },
                                ));
                                let klass_lit = self
                                    .pool
                                    .intern(aklass.oop().raw(), Some(RelocKind::Oop));
                                let dst = self.fresh(true);
                                code.push(Ir::Alloc {
                                    dst,
                                    klass: klass_lit,
                                    size_words,
                                });
                                cstack.push(dst);
                                bci = next;
                                continue;
                            }
                        }
                    }

                    // Consult the callee's own feedback (its IC). A cold
                    // (Untaken) inner IC lowers to a step-3 uncommon trap INSIDE
                    // the inlined body — the cleanest way to force an in-body
                    // deopt; a warm mono/poly/mega stays a plain compiled
                    // `CallSend`. We do NOT recursively inline (depth-1).
                    let inner_fb =
                        crate::compiler::feedback::read_send_site(self.vm, callee, ic, None);

                    if self.osr
                        && matches!(
                            crate::compiler::inline::decide(&inner_fb),
                            crate::compiler::inline::InlineDecision::Trap
                        )
                    {
                        self.osr_cold_sends += 1; // OSR-heal census (see root arm)
                    }
                    // Pop the inner send's operands off the callee's own stack.
                    let mut inner_args: Vec<VReg> = (0..inner_argc)
                        .map(|_| {
                            cstack
                                .pop()
                                .expect("inline splice: inner send missing arg operand")
                        })
                        .collect();
                    inner_args.reverse();
                    let inner_recv = cstack
                        .pop()
                        .expect("inline splice: inner send missing receiver operand");

                    // S14 step 5: an inner SELF-send (receiver == the spliced
                    // callee's own self) with a statically-known receiver klass
                    // never traps cold — its target is fixed at compile time,
                    // so an Empty IC just compiles as the plain lazy CallSend
                    // below (no interpreted warm-up, no in-body deopt).
                    let inner_static_self = inner_recv == callee_self
                        && callee_self_klass
                            .is_some_and(|k| self.devirt_self_target(k, inner_sel).is_some());
                    if !inner_static_self && self.inlined_cold_send_traps() {
                        if let crate::compiler::inline::InlineDecision::Trap =
                            crate::compiler::inline::decide(&inner_fb)
                        {
                            // Cold inner send → terminating uncommon trap INSIDE the
                            // inlined body, re-executing the inner send interpreted.
                            // Its recorded operand stack is the inlined body's stack
                            // WITH the inner send's receiver+args still present
                            // (reexecute=true), plus the receiver+args just popped.
                            let mut reexec_stack = cstack.clone();
                            reexec_stack.push(inner_recv);
                            reexec_stack.extend_from_slice(&inner_args);
                            deopt.push((
                                code.len() as u32,
                                DeoptRaw {
                                    stack: reexec_stack,
                                    // Resume bci is the INNER send's bci in the
                                    // INLINED callee (the inlined frame's own resume
                                    // point); the caller frame's resume comes from
                                    // the SenderLink.
                                    bci: inner_bci,
                                    kind: SafepointKind::UncommonTrap,
                                    reexecute: true,
                                    stack_closures: Vec::new(),
                                    inline: Some(inline_proto.clone()),
                                },
                            ));
                            code.push(Ir::UncommonTrap { bci: inner_bci });
                            trapped = true;
                            break;
                        }
                    }

                    // A real compiled send inside the inlined body. Its site is
                    // the callee's own (klass, selector) speculation-independent
                    // dispatch — a plain S11 compiled IC.
                    let mut send_args = vec![inner_recv];
                    send_args.extend_from_slice(&inner_args);
                    let site = self.call_sites.len() as u16;
                    self.call_sites.push(CallSiteInfo {
                        selector: inner_sel,
                        argc: inner_argc + 1,
                        static_klass: None,
                    });
                    self.site_feedback.push(inner_fb);
                    let dst = self.fresh(true);
                    // Call-return safepoint (reexecute=false): recorded stack is
                    // the inlined body's operand stack BELOW the send (receiver+
                    // args already popped); the materializer pushes the result.
                    deopt.push((
                        code.len() as u32,
                        DeoptRaw {
                            stack: cstack.clone(),
                            bci: inner_bci,
                            kind: SafepointKind::Call,
                            reexecute: false,
                            stack_closures: Vec::new(),
                            inline: Some(inline_proto.clone()),
                        },
                    ));
                    code.push(Ir::CallSend {
                        dst,
                        site,
                        args: send_args,
                    });
                    cstack.push(dst);
                }
                Instr::ReturnTos => {
                    result = Some(
                        cstack
                            .pop()
                            .expect("inline splice: return_tos on empty stack"),
                    );
                    break;
                }
                Instr::ReturnSelf => {
                    result = Some(callee_self);
                    break;
                }
                // `is_inline_eligible_nonleaf` rejected super/ctx/closure/NLR;
                // every other opcode is handled above.
                other => {
                    // A shape slipped past the eligibility gate — decline
                    // cleanly. But we have already emitted the guard + cold
                    // block; that is harmless (the guard dominates a body we now
                    // abandon), yet the caller's operand-stack discipline breaks
                    // if we return None mid-splice. So this must be unreachable
                    // by construction (the gate is exact); assert it.
                    unreachable!(
                        "non-leaf inline splice: {other:?} passed is_inline_eligible_nonleaf \
                         but has no arm"
                    )
                }
            }
            bci = next;
        }

        self.budget_commit(callee);
        self.record_inline_dep(dep_klass, selector);
        if trapped {
            Some(NonLeafOutcome::Trapped)
        } else {
            let result =
                result.expect("inline splice: single-return non-leaf must produce a result");
            Some(NonLeafOutcome::Value(result))
        }
    }

    /// S14 step 7-IV-a: an uncommon-trap block for a deopt site INSIDE an
    /// inlined extent — like [`Translator::fresh_not_bool_block`] /
    /// [`Translator::fail_and_branch`] but carrying the extent's `InlineSite`,
    /// so the deopt rebuilds BOTH the inlined frame and its caller.
    fn fresh_inlined_trap_block(
        &mut self,
        trap_bci: usize,
        reexec_stack: Vec<VReg>,
        inline: InlineSite,
    ) -> BlockId {
        let trap_id = self.fresh_block_id();
        self.finish_block(IrBlock {
            id: trap_id,
            bci: trap_bci,
            code: vec![Ir::UncommonTrap { bci: trap_bci }],
            entry_stack: Vec::new(),
            deopt_sites: vec![(
                0,
                DeoptRaw {
                    stack: reexec_stack,
                    bci: trap_bci,
                    kind: SafepointKind::UncommonTrap,
                    reexecute: true,
                    stack_closures: Vec::new(),
                    inline: Some(inline),
                },
            )],
        });
        trap_id
    }

    /// S14 step 7-IV-a: splice a MULTI-BLOCK callee (branches, loops — the
    /// general CFG form `inline::is_inline_eligible_cfg` admits) inline behind
    /// a receiver-klass guard. Where the leaf/nonleaf splicers paste a single
    /// straight line into the caller's current block, this maps the callee's
    /// WHOLE decoded CFG onto fresh caller `IrBlock`s:
    ///
    ///   - every callee CFG block gets a fresh caller block id; its
    ///     `Jump`/`Branch` targets are remapped through that table;
    ///   - the callee's operand-stack merge discipline is rebuilt with the SAME
    ///     machinery `convert` uses on the root method (`compute_entry_depths` +
    ///     `entry_stack_sources` + `emit_merges` + fresh merge vregs);
    ///   - every `return` becomes `Move {result_vreg, v}; Jump {continuation}` —
    ///     the caller resumes in `continuation` with `result_vreg` as the send's
    ///     value;
    ///   - a backward jump becomes an `Ir::Poll` with an in-body LOOP-POLL deopt
    ///     scope (`InlineSite`-chained, reexecute at the callee's loop-header
    ///     bci) — the first time a loop-poll deopt reconstructs depth 2;
    ///   - a `Branch` becomes a (non-fused) `Ir::BoolBr` whose `not_bool` edge is
    ///     an `InlineSite`-chained trap. The condition is POPPED from the sim
    ///     stack after capturing the trap's reexecute snapshot — the
    ///     strictly-correct accounting (`instr_stack_delta(br) == -1`), keeping
    ///     Inherit/Merge successors depth-exact at ANY stack depth (convert's
    ///     root loop gets away without the pop only because its branch
    ///     successors sit at depth 0 and `Empty` discards the leftover);
    ///   - the callee's own sends lower exactly as in the nonleaf splice: a cold
    ///     (Untaken) IC → a TERMINATING in-body trap; a warm IC → a plain
    ///     compiled `CallSend` with a call-return deopt scope. No smi fusion, no
    ///     branch fusion, no recursive inlining inside the inlinee (correctness
    ///     first; those are optimizations on top).
    ///
    /// On success returns `(result_vreg, callee_entry_id, continuation_id)`:
    /// the caller pushes `result_vreg`, sets `pending_jump_target =
    /// callee_entry_id`, and returns `continuation_id` as its `split`. Returns
    /// `None` (having emitted NOTHING) only on an up-front shape mismatch.
    #[allow(clippy::too_many_arguments)]
    fn try_inline_cfg(
        &mut self,
        callee: MethodOop,
        guard: Option<(KlassOop, GuardShape)>,
        dep_klass: KlassOop,
        selector: SymbolOop,
        receiver: VReg,
        args: &[VReg],
        send_bci: usize,
        pre_pop_stack: &[VReg],
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
        blockarg: Option<(usize, MethodOop, u32)>,
        mode: GraftMode,
    ) -> Option<(VReg, BlockId, BlockId)> {
        // S24 B5 step 1: destructured once — the walk consults plain locals.
        let (is_block_graft, graft_parent): (bool, Option<Box<InlineSite>>) = match mode {
            GraftMode::Method => (false, None),
            GraftMode::Block { parent } => (true, parent),
        };
        if is_block_graft {
            // Both graft consumers (direct value sites, in-callee blockarg
            // value sites) route here — the one place the stat can't miss.
            self.spliced_multibb += 1;
        }
        let _ = deopt; // caller-block deopt vec unused: guard cold path is its own block
                       // Same exact-arm-set validation as `try_inline_nonleaf` (see the
                       // comment there): the CFG walk aborts on shapes outside its arm list,
                       // so this splicer proves its own eligibility instead of trusting the
                       // caller's budget gate.
        if !is_block_graft && !crate::compiler::inline::is_inline_eligible_cfg(callee) {
            return None;
        }
        let ccfg = crate::compiler::decode::decode(callee);
        if ccfg.blocks.is_empty() || callee.argc() != args.len() {
            return None;
        }

        let callee_pool_ix = self.pool.intern(callee.oop().raw(), Some(RelocKind::Oop)).0;
        let n_operands = args.len() + 1;
        debug_assert!(pre_pop_stack.len() >= n_operands);
        let caller_pending_stack = pre_pop_stack[..pre_pop_stack.len() - n_operands].to_vec();
        let callee_self = receiver;

        // S14 step 5: the statically-proven klass of `callee_self` — the
        // guard klass when this splice is receiver-guarded, or the
        // customization klass when the receiver is the root method's own
        // `self` (guard-free super/self splices, whose receiver the entry
        // guard already checked). `None` cannot occur today (every
        // guard-free splice passes a self receiver) but stays sound if it
        // ever does: inner self-sends just keep their feedback lowering.
        let callee_self_klass: Option<KlassOop> = match guard {
            Some((gk, _)) => Some(gk),
            None if receiver == self.self_vreg => Some(self.rcvr_klass),
            None => None,
        };

        // Guard fronting the whole extent (cold = reexecute the send in the
        // CALLER scope — `inline: None`, exactly the leaf/nonleaf convention).
        // `None` for a STATICALLY-resolved target (a super send): the entry
        // check already proved everything there is to prove — no guard at all.
        if let Some((guard_klass, guard_shape)) = guard {
            let expect = self
                .pool
                .intern(guard_klass.oop().raw(), Some(RelocKind::Oop));
            let cold_id = self.fresh_block_id();
            // S14 step 7-IV-c: a block-arg send's reexecute stack carries the
            // PHANTOM closure at the arg position — the materializer must
            // allocate the real one before the interpreter re-executes it.
            let guard_stack_closures: Vec<(u16, u32)> = match blockarg {
                Some((arg_ix, _, blk_pool_ix)) => {
                    let pos = pre_pop_stack.len() - args.len() + arg_ix;
                    vec![(pos as u16, blk_pool_ix)]
                }
                None => Vec::new(),
            };
            self.finish_block(IrBlock {
                id: cold_id,
                bci: send_bci,
                code: vec![Ir::UncommonTrap { bci: send_bci }],
                entry_stack: Vec::new(),
                deopt_sites: vec![(
                    0,
                    DeoptRaw {
                        stack: pre_pop_stack.to_vec(),
                        bci: send_bci,
                        kind: SafepointKind::UncommonTrap,
                        reexecute: true,
                        stack_closures: guard_stack_closures,
                        inline: None,
                    },
                )],
            });
            code.push(Ir::GuardKlass {
                obj: callee_self,
                expect,
                fail: cold_id,
                kind: guard_shape,
            });
        }

        // Callee slots: args alias the send operands; temps = fresh nil vregs
        // (initialised in the CALLER's block, dominating the whole extent).
        let mut callee_slots: Vec<VReg> = Vec::with_capacity(callee.argc() + callee.ntemps());
        callee_slots.extend_from_slice(args);
        let nil_lit = self
            .pool
            .intern(self.vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
        for _ in 0..callee.ntemps() {
            let t = self.fresh(true);
            code.push(Ir::ConstPool {
                dst: t,
                lit: nil_lit,
            });
            callee_slots.push(t);
        }

        let inline_proto = InlineSite {
            method_pool_ix: callee_pool_ix,
            receiver: callee_self,
            slots: callee_slots.clone(),
            sender_bci: send_bci as u16,
            caller_pending_stack,
            is_block: is_block_graft,
            parent: graft_parent,
            // S14 step 7-IV-c: the block-arg temp holds a PHANTOM (its vreg is
            // an unread filler) — every in-callee deopt scope materializes the
            // real closure for that slot.
            slot_closures: match blockarg {
                Some((arg_ix, _, blk_pool_ix)) => vec![(arg_ix as u16, blk_pool_ix)],
                None => Vec::new(),
            },
        };

        // ── CFG scaffolding: the same machinery convert runs on the root. ──
        let (entry_depth, _max) = compute_entry_depths(callee, &ccfg);
        let sources = entry_stack_sources(&ccfg, &entry_depth);
        let n = ccfg.blocks.len();
        let ir_ids: Vec<BlockId> = (0..n).map(|_| self.fresh_block_id()).collect();
        let continuation_id = self.fresh_block_id();
        let result_vreg = self.fresh(true);

        let mut entry_stacks: Vec<Option<Vec<VReg>>> = vec![None; n];
        for (b, source) in sources.iter().enumerate() {
            match source {
                EntryStackSource::Empty => entry_stacks[b] = Some(Vec::new()),
                EntryStackSource::Merge => {
                    let depth = entry_depth[b] as usize;
                    entry_stacks[b] = Some((0..depth).map(|_| self.fresh(true)).collect());
                }
                EntryStackSource::Inherit(_) => {}
            }
        }
        let mut exit_stacks: Vec<Option<Vec<VReg>>> = vec![None; n];
        let mut reachable = vec![false; n];
        reachable[0] = true;

        for b in 0..n {
            if !reachable[b] {
                // Dead (post-trap / post-return) callee block: an inert,
                // structurally-valid placeholder, never jumped to (mirrors
                // convert's own dead-block handling).
                self.finish_block(IrBlock {
                    id: ir_ids[b],
                    bci: send_bci,
                    code: vec![Ir::RetSelf],
                    entry_stack: Vec::new(),
                    deopt_sites: Vec::new(),
                });
                continue;
            }
            let entry_stack = match sources[b] {
                EntryStackSource::Empty | EntryStackSource::Merge => {
                    entry_stacks[b].clone().expect("pre-allocated above")
                }
                EntryStackSource::Inherit(pred) => exit_stacks[pred]
                    .clone()
                    .expect("single non-backward predecessor translated first"),
            };
            let cfg_block = &ccfg.blocks[b];
            let mut cstack = entry_stack.clone();
            // S24 B5 step 5 (H6 cur_id threading): the IrBlock id the CURRENT
            // accumulating segment of callee block `b` will be finished as.
            // A multi-BB block graft at an in-callee value site SPLITS the
            // segment (mirroring the root loop's split protocol): the pre-
            // split segment ends with a Jump to the graft entry, and the
            // graft's continuation becomes the new segment. Predecessor
            // edges into `b` still target `ir_ids[b]` (the FIRST segment);
            // the terminator, `exit_stacks[b]`, and merge-moves land in the
            // FINAL segment via this variable.
            let mut cur_id = ir_ids[b];
            // S14 step 7-IV-c: parallel phantom shadow — `true` where the
            // callee's sim stack holds the block-arg PHANTOM (pushed by
            // `push_temp arg_ix`). Never survives a block boundary
            // (transparency's boundary rule), so entry is all-false.
            let mut cstack_ph: Vec<bool> = vec![false; cstack.len()];
            let mut bcode: Vec<Ir> = Vec::new();
            let mut bdeopt: Vec<(u32, DeoptRaw)> = Vec::new();
            let mut trapped = false;
            let mut returned = false;
            let mut last_instr_bci = cfg_block.bci_start;
            let mut bci = cfg_block.bci_start;
            while bci < cfg_block.bci_end {
                last_instr_bci = bci;
                let (instr, next) = decode_at(callee, bci);
                match instr {
                    Instr::PushSelf => cstack.push(callee_self),
                    Instr::PushNil => cstack
                        .push(self.push_well_known(self.vm.universe.nil_obj.raw(), &mut bcode)),
                    Instr::PushTrue => cstack
                        .push(self.push_well_known(self.vm.universe.true_obj.raw(), &mut bcode)),
                    Instr::PushFalse => cstack
                        .push(self.push_well_known(self.vm.universe.false_obj.raw(), &mut bcode)),
                    Instr::PushSmi(v) => {
                        let dst = self.fresh(true);
                        bcode.push(Ir::ConstSmi {
                            dst,
                            value: v as i64,
                        });
                        cstack.push(dst);
                    }
                    Instr::PushLiteral(idx) => {
                        let lit_oop = callee.literals().at(idx as usize);
                        cstack.push(self.push_well_known(lit_oop.raw(), &mut bcode));
                    }
                    Instr::PushGlobal(idx) => {
                        let assoc = callee.literals().at(idx as usize);
                        let assoc_vreg = self.push_well_known(assoc.raw(), &mut bcode);
                        let dst = self.fresh(true);
                        bcode.push(Ir::LoadField {
                            dst,
                            obj: assoc_vreg,
                            byte_off: (BODY_OFFSET + 8) as i32,
                        });
                        // Mirror the root arm (and the nonleaf splicer): track
                        // class-valued globals for the in-body alloc fuse.
                        let value = crate::oops::wrappers::MemOop::try_from(assoc)
                            .map(|a| a.body_oop(1))
                            .unwrap_or(assoc);
                        if let Some(klass) = KlassOop::try_from(value) {
                            self.const_class.insert(dst.0, klass);
                        }
                        cstack.push(dst);
                    }
                    Instr::PushTemp(t) => {
                        // The block-arg temp pushes the PHANTOM (no Move — its
                        // filler vreg is never read; the shadow carries it).
                        if let Some((arg_ix, _, _)) = blockarg {
                            if t as usize == arg_ix {
                                cstack.push(callee_self); // harmless filler
                                cstack_ph.resize(cstack.len() - 1, false);
                                cstack_ph.push(true);
                                bci = next;
                                continue;
                            }
                        }
                        let dst = self.fresh(true);
                        bcode.push(Ir::Move {
                            dst,
                            src: callee_slots[t as usize],
                        });
                        cstack.push(dst);
                    }
                    Instr::StoreTemp(t) => {
                        let tos = *cstack.last().expect("cfg splice: store_temp empty stack");
                        bcode.push(Ir::Move {
                            dst: callee_slots[t as usize],
                            src: tos,
                        });
                    }
                    Instr::StoreTempPop(t) => {
                        let v = cstack
                            .pop()
                            .expect("cfg splice: store_temp_pop empty stack");
                        bcode.push(Ir::Move {
                            dst: callee_slots[t as usize],
                            src: v,
                        });
                    }
                    Instr::PushInstvar(iv) => {
                        let dst = self.fresh(true);
                        bcode.push(Ir::LoadField {
                            dst,
                            obj: callee_self,
                            byte_off: (BODY_OFFSET + 8 * iv as usize) as i32,
                        });
                        cstack.push(dst);
                    }
                    Instr::StoreInstvarPop(iv) => {
                        let val = cstack
                            .pop()
                            .expect("cfg splice: store_instvar_pop empty stack");
                        bcode.push(Ir::StoreField {
                            obj: callee_self,
                            byte_off: (BODY_OFFSET + 8 * iv as usize) as i32,
                            val,
                            barrier: true,
                        });
                    }
                    Instr::StoreGlobalPop(idx) => {
                        let assoc = callee.literals().at(idx as usize);
                        let assoc_vreg = self.push_well_known(assoc.raw(), &mut bcode);
                        let val = cstack
                            .pop()
                            .expect("cfg splice: store_global_pop empty stack");
                        bcode.push(Ir::StoreField {
                            obj: assoc_vreg,
                            byte_off: (BODY_OFFSET + 8) as i32,
                            val,
                            barrier: true,
                        });
                    }
                    Instr::Pop => {
                        cstack.pop().expect("cfg splice: pop empty stack");
                    }
                    Instr::Dup => {
                        let tos = *cstack.last().expect("cfg splice: dup empty stack");
                        cstack.push(tos);
                    }
                    Instr::Send { ic, super_ } => {
                        debug_assert!(!super_, "cfg splice: super gated by is_inline_eligible_cfg");
                        let inner_ic = InterpreterIc::at(callee, ic);
                        let inner_argc = inner_ic.argc();
                        let inner_sel = inner_ic.selector();
                        let inner_bci = bci;
                        // S14 step 7-IV-c: a value-family send whose RECEIVER is
                        // the phantom splices the BLOCK's body right here, with
                        // an InlineSite CHAINED to this callee's own (depth 3:
                        // block ← callee ← root). Transparency proved the shape
                        // (matching argc, no other phantom on the stack).
                        cstack_ph.resize(cstack.len(), false);
                        let recv_is_phantom = cstack_ph
                            .get(cstack.len() - 1 - inner_argc as usize)
                            .copied()
                            .unwrap_or(false);
                        if recv_is_phantom {
                            let (_, blk, blk_pool_ix) =
                                blockarg.expect("phantom shadow set without a blockarg");
                            // Snapshot BEFORE popping: the recursive graft
                            // derives its caller_pending_stack (this callee's
                            // stack below phantom+args) from the pre-pop view.
                            let pre_pop_cstack = cstack.clone();
                            let mut blk_args: Vec<VReg> = (0..inner_argc)
                                .map(|_| {
                                    cstack_ph.pop();
                                    cstack.pop().expect("cfg splice: value arg")
                                })
                                .collect();
                            blk_args.reverse();
                            cstack_ph.pop();
                            cstack.pop().expect("cfg splice: phantom receiver");
                            debug_assert!(
                                !cstack_ph.iter().any(|&ph| ph),
                                "cfg splice: phantom below a value site (transparency bug)"
                            );
                            // S24 B5 step 5: a multi-BB block grafts via the
                            // CFG engine (depth 3: block <- callee <- root),
                            // splitting this callee block's segment. The
                            // recursion allocates its own fresh IrBlock ids
                            // (no collision with ir_ids) and finishes its own
                            // blocks; its continuation becomes our new
                            // segment. mem::take per segment keeps bdeopt's
                            // per-block code indices correct (they restart at
                            // 0 in the new segment's bcode).
                            if crate::compiler::decode::decode(blk).blocks.len() > 1 {
                                let (res, entry_id, cont_id) = self
                                    .try_inline_cfg(
                                        blk,
                                        None,
                                        dep_klass,
                                        inner_sel,
                                        self.self_vreg,
                                        &blk_args,
                                        inner_bci,
                                        &pre_pop_cstack,
                                        &mut bcode,
                                        &mut bdeopt,
                                        None,
                                        GraftMode::Block {
                                            parent: Some(Box::new(inline_proto.clone())),
                                        },
                                    )
                                    .expect(
                                        "cfg splice: escape proved the block-arg's multi-BB \
                                         body graftable but the graft declined (compiler bug)",
                                    );
                                let mut seg_code = std::mem::take(&mut bcode);
                                let seg_deopt = std::mem::take(&mut bdeopt);
                                seg_code.push(Ir::Jump { target: entry_id });
                                self.finish_block(IrBlock {
                                    id: cur_id,
                                    bci: send_bci,
                                    code: seg_code,
                                    entry_stack: if cur_id == ir_ids[b] {
                                        entry_stack.clone()
                                    } else {
                                        // A split-off continuation's own
                                        // entry_stack is never consulted
                                        // (root split protocol's rule).
                                        Vec::new()
                                    },
                                    deopt_sites: seg_deopt,
                                });
                                cur_id = cont_id;
                                cstack.push(res);
                                cstack_ph.push(false);
                                bci = next;
                                continue;
                            }
                            // The block's slots: value args alias + fresh nils.
                            let mut bslots: Vec<VReg> =
                                Vec::with_capacity(blk.argc() + blk.ntemps());
                            bslots.extend_from_slice(&blk_args);
                            let nil_lit2 = self
                                .pool
                                .intern(self.vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
                            for _ in 0..blk.ntemps() {
                                let t = self.fresh(true);
                                bcode.push(Ir::ConstPool {
                                    dst: t,
                                    lit: nil_lit2,
                                });
                                bslots.push(t);
                            }
                            let blk_proto = InlineSite {
                                method_pool_ix: blk_pool_ix,
                                receiver: self.self_vreg, // home self = ROOT's
                                slots: bslots.clone(),
                                sender_bci: inner_bci as u16,
                                caller_pending_stack: cstack.clone(),
                                is_block: true,
                                parent: Some(Box::new(inline_proto.clone())),
                                slot_closures: Vec::new(),
                            };
                            match self.translate_spliced_block_body(
                                blk,
                                &bslots,
                                self.self_vreg,
                                &blk_proto,
                                &mut bcode,
                                &mut bdeopt,
                            ) {
                                Ok(res) => {
                                    cstack.push(res);
                                    cstack_ph.push(false);
                                    bci = next;
                                    continue;
                                }
                                Err(()) => {
                                    // Terminating trap/NLR inside the block:
                                    // this callee block ends here.
                                    trapped = true;
                                    break;
                                }
                            }
                        }
                        // S14 opt: fuse the callee body's own mono-smi ops
                        // (the loop callee's `i <= n` / `i + 1` become real
                        // cmp/adds instead of per-iteration stub calls). After
                        // the phantom check: a phantom receiver is never a
                        // fusable smi site. The shadow pops in lockstep
                        // (operands are never phantoms — transparency).
                        if let Some(is_put) = array_op_kind_on(self.vm, callee, ic) {
                            // S15: in-body array intrinsic (see the root arm).
                            let popped = if is_put { 3 } else { 2 };
                            cstack_ph.truncate(cstack.len().saturating_sub(popped));
                            let val = if is_put {
                                Some(cstack.pop().expect("array fuse: missing value"))
                            } else {
                                None
                            };
                            let idx_op = cstack.pop().expect("array fuse: missing index");
                            let arr_op = cstack.pop().expect("array fuse: missing receiver");
                            let mut reexec = cstack.clone();
                            reexec.push(arr_op);
                            reexec.push(idx_op);
                            if let Some(v) = val {
                                reexec.push(v);
                            }
                            let fail = self.fresh_inlined_trap_block(
                                inner_bci,
                                reexec,
                                inline_proto.clone(),
                            );
                            let klass = self.pool.intern(
                                self.vm.universe.array_klass.oop().raw(),
                                Some(RelocKind::Oop),
                            );
                            let dst = self.fresh(true);
                            match val {
                                Some(v) => bcode.push(Ir::ArrayAtPut {
                                    dst,
                                    arr: arr_op,
                                    idx: idx_op,
                                    val: v,
                                    klass,
                                    fail,
                                }),
                                None => bcode.push(Ir::ArrayAt {
                                    dst,
                                    arr: arr_op,
                                    idx: idx_op,
                                    klass,
                                    fail,
                                }),
                            }
                            cstack.push(dst);
                            cstack_ph.push(false);
                            bci = next;
                            continue;
                        }
                        // Cog-parity special selectors in the CFG-spliced
                        // body. The UNGUARDED identity compare must never
                        // see a phantom operand (an elided-closure filler
                        // holds `callee_self`, not the closure) — screened
                        // on the top two shadow entries; `not`'s boolean
                        // guard traps correctly on a filler, but is
                        // screened too so the reexec stack stays exact.
                        if self
                            .ref_eq_kind(callee, ic)
                            .filter(|_| {
                                !cstack_ph.iter().rev().take(2).any(|&ph| ph)
                            })
                            .is_some()
                        {
                            let neq = self
                                .ref_eq_kind(callee, ic)
                                .expect("guard just matched");
                            cstack_ph.truncate(cstack.len().saturating_sub(2));
                            let b_op = cstack.pop().expect("ref-eq fuse: missing rhs");
                            let a_op = cstack.pop().expect("ref-eq fuse: missing lhs");
                            let dst = self.fresh(true);
                            bcode.push(Ir::RefCmpVal {
                                dst,
                                a: a_op,
                                b: b_op,
                                neq,
                            });
                            cstack.push(dst);
                            cstack_ph.push(false);
                            bci = next;
                            continue;
                        }
                        if self.bool_not_speculatable(callee, ic)
                            && !cstack_ph.last().copied().unwrap_or(false)
                        {
                            self.record_inline_dep(self.vm.universe.true_klass, inner_sel);
                            self.record_inline_dep(self.vm.universe.false_klass, inner_sel);
                            cstack_ph.truncate(cstack.len().saturating_sub(1));
                            let src = cstack.pop().expect("bool-not fuse: missing receiver");
                            let mut reexec = cstack.clone();
                            reexec.push(src);
                            let fail = self.fresh_inlined_trap_block(
                                inner_bci,
                                reexec,
                                inline_proto.clone(),
                            );
                            let dst = self.fresh(true);
                            bcode.push(Ir::BoolNot { dst, src, fail });
                            cstack.push(dst);
                            cstack_ph.push(false);
                            bci = next;
                            continue;
                        }
                        if is_smi_inlinable_on(self.vm, callee, ic)
                            && !smi_fuse_arg_is_pooled(cstack.as_slice(), bcode.as_slice())
                        {
                            debug_assert_eq!(inner_argc, 1, "SMI_INLINE ops are all binary");
                            cstack_ph.truncate(cstack.len().saturating_sub(2));
                            let b_op = cstack.pop().expect("smi fuse: missing rhs");
                            let a_op = cstack.pop().expect("smi fuse: missing lhs");
                            let mut reexec = cstack.clone();
                            reexec.push(a_op);
                            reexec.push(b_op);
                            let fail = self.fresh_inlined_trap_block(
                                inner_bci,
                                reexec,
                                inline_proto.clone(),
                            );
                            let dst = self.fresh(true);
                            match classify_smi_send(self.vm, callee, ic) {
                                SmiSendKind::Arith(op) => bcode.push(Ir::SmiArith {
                                    op,
                                    dst,
                                    a: a_op,
                                    b: b_op,
                                    fail,
                                }),
                                SmiSendKind::Cmp(op) => bcode.push(Ir::SmiCmpVal {
                                    op,
                                    dst,
                                    a: a_op,
                                    b: b_op,
                                    fail,
                                }),
                            }
                            cstack.push(dst);
                            cstack_ph.push(false);
                            bci = next;
                            continue;
                        }
                        // Float fast-path in the splice: `is_double_inlinable`'s
                        // `_on` twin, same pattern as the smi twin above.
                        if is_double_inlinable_on(self.vm, callee, ic) {
                            debug_assert_eq!(inner_argc, 1, "DOUBLE_INLINE ops are all binary");
                            // b4d55a8's storm guard, ported: a compile-time smi
                            // arg can never FUnbox — emit the coerced FConst.
                            let arg_const_smi: Option<i64> = match (cstack.last(), bcode.last()) {
                                (Some(&top), Some(Ir::ConstSmi { dst, value })) if *dst == top => {
                                    Some(*value)
                                }
                                _ => None,
                            };
                            cstack_ph.truncate(cstack.len().saturating_sub(2));
                            let b_op = cstack.pop().expect("double fuse: missing rhs");
                            let a_op = cstack.pop().expect("double fuse: missing lhs");
                            let mut reexec = cstack.clone();
                            reexec.push(a_op);
                            reexec.push(b_op);
                            let fail = self.fresh_inlined_trap_block(
                                inner_bci,
                                reexec,
                                inline_proto.clone(),
                            );
                            let ua = self.fresh_fp();
                            let ub = self.fresh_fp();
                            bcode.push(Ir::FUnbox {
                                dst: ua,
                                src: a_op,
                                fail,
                            });
                            match arg_const_smi {
                                Some(v) => bcode.push(Ir::FConst {
                                    dst: ub,
                                    bits: (v as f64).to_bits(),
                                }),
                                None => bcode.push(Ir::FUnbox {
                                    dst: ub,
                                    src: b_op,
                                    fail,
                                }),
                            }
                            let dst = self.fresh(true);
                            match classify_double_send(self.vm, callee, ic) {
                                DoubleSendKind::Arith(op) => {
                                    let fd = self.fresh_fp();
                                    bcode.push(Ir::FArith {
                                        op,
                                        dst: fd,
                                        a: ua,
                                        b: ub,
                                    });
                                    bcode.push(Ir::FBox { dst, src: fd });
                                }
                                DoubleSendKind::Cmp(op) => {
                                    bcode.push(Ir::FCmpVal {
                                        op,
                                        dst,
                                        a: ua,
                                        b: ub,
                                    });
                                }
                            }
                            cstack.push(dst);
                            cstack_ph.push(false);
                            bci = next;
                            continue;
                        }
                        // gc_alloc_gap cost 1, cfg-splice half: same in-body
                        // `basicNew` → `Ir::Alloc` fuse as the nonleaf splicer
                        // (see that arm for the full story). Phantom-free
                        // stacks only: with no phantom anywhere, the deopt's
                        // `stack_closures` is empty by construction — the
                        // conservative gate that keeps this arm simple.
                        if inner_argc == 0 && !cstack_ph.iter().any(|&ph| ph) {
                            if let Some(&recv) = cstack.last() {
                                if let Some((aklass, size_words)) = self.alloc_site_klass_on(
                                    callee,
                                    ic,
                                    recv,
                                    callee_self,
                                    callee_self_klass,
                                ) {
                                    cstack.pop();
                                    cstack_ph.truncate(cstack.len());
                                    let mut reexec = cstack.clone();
                                    reexec.push(recv);
                                    bdeopt.push((
                                        bcode.len() as u32,
                                        DeoptRaw {
                                            stack: reexec,
                                            bci: inner_bci,
                                            kind: SafepointKind::Alloc,
                                            reexecute: true,
                                            stack_closures: Vec::new(),
                                            inline: Some(inline_proto.clone()),
                                        },
                                    ));
                                    let klass_lit = self
                                        .pool
                                        .intern(aklass.oop().raw(), Some(RelocKind::Oop));
                                    let dst = self.fresh(true);
                                    bcode.push(Ir::Alloc {
                                        dst,
                                        klass: klass_lit,
                                        size_words,
                                    });
                                    cstack.push(dst);
                                    cstack_ph.push(false);
                                    bci = next;
                                    continue;
                                }
                            }
                        }
                        let inner_fb =
                            crate::compiler::feedback::read_send_site(self.vm, callee, ic, None);
                        if self.osr
                            && matches!(
                                crate::compiler::inline::decide(&inner_fb),
                                crate::compiler::inline::InlineDecision::Trap
                            )
                        {
                            self.osr_cold_sends += 1; // OSR-heal census (see root arm)
                        }

                        let mut inner_args: Vec<VReg> = (0..inner_argc)
                            .map(|_| cstack.pop().expect("cfg splice: send missing arg"))
                            .collect();
                        inner_args.reverse();
                        let inner_recv = cstack.pop().expect("cfg splice: send missing receiver");
                        // Phantom entries below this send's operands must be
                        // recorded as ElidedClosure at its deopt sites.
                        cstack_ph.truncate(cstack.len());
                        let ph_closures: Vec<(u16, u32)> = match blockarg {
                            Some((_, _, blk_pool_ix)) => cstack_ph
                                .iter()
                                .enumerate()
                                .filter(|(_, &ph)| ph)
                                .map(|(i, _)| (i as u16, blk_pool_ix))
                                .collect(),
                            None => Vec::new(),
                        };

                        // S14 step 5: an inner SELF-send (receiver == the spliced
                        // callee's own self) with a statically-known receiver klass
                        // never traps cold — its target is fixed at compile time,
                        // so an Empty IC just compiles as the plain lazy CallSend
                        // below (no interpreted warm-up, no in-body deopt).
                        let inner_static_self = inner_recv == callee_self
                            && callee_self_klass
                                .is_some_and(|k| self.devirt_self_target(k, inner_sel).is_some());
                        if !inner_static_self && self.inlined_cold_send_traps() {
                            if let crate::compiler::inline::InlineDecision::Trap =
                                crate::compiler::inline::decide(&inner_fb)
                            {
                                let mut reexec_stack = cstack.clone();
                                reexec_stack.push(inner_recv);
                                reexec_stack.extend_from_slice(&inner_args);
                                bdeopt.push((
                                    bcode.len() as u32,
                                    DeoptRaw {
                                        stack: reexec_stack,
                                        bci: inner_bci,
                                        kind: SafepointKind::UncommonTrap,
                                        reexecute: true,
                                        stack_closures: ph_closures.clone(),
                                        inline: Some(inline_proto.clone()),
                                    },
                                ));
                                bcode.push(Ir::UncommonTrap { bci: inner_bci });
                                trapped = true;
                                break;
                            }
                        }

                        let mut send_args = vec![inner_recv];
                        send_args.extend_from_slice(&inner_args);
                        let site = self.call_sites.len() as u16;
                        self.call_sites.push(CallSiteInfo {
                            selector: inner_sel,
                            argc: inner_argc + 1,
                            static_klass: None,
                        });
                        self.site_feedback.push(inner_fb);
                        let dst = self.fresh(true);
                        bdeopt.push((
                            bcode.len() as u32,
                            DeoptRaw {
                                stack: cstack.clone(),
                                bci: inner_bci,
                                kind: SafepointKind::Call,
                                reexecute: false,
                                stack_closures: ph_closures,
                                inline: Some(inline_proto.clone()),
                            },
                        ));
                        bcode.push(Ir::CallSend {
                            dst,
                            site,
                            args: send_args,
                        });
                        cstack.push(dst);
                    }
                    Instr::ReturnTos => {
                        let v = cstack.pop().expect("cfg splice: return_tos empty stack");
                        bcode.push(Ir::Move {
                            dst: result_vreg,
                            src: v,
                        });
                        bcode.push(Ir::Jump {
                            target: continuation_id,
                        });
                        returned = true;
                        break;
                    }
                    Instr::ReturnSelf => {
                        bcode.push(Ir::Move {
                            dst: result_vreg,
                            src: callee_self,
                        });
                        bcode.push(Ir::Jump {
                            target: continuation_id,
                        });
                        returned = true;
                        break;
                    }
                    // S24 B5 (Block graft only): the block's own `^value` at
                    // the value-send — joins the continuation exactly like a
                    // method callee's ReturnTos.
                    Instr::BlockReturnTos if is_block_graft => {
                        let v = cstack
                            .pop()
                            .expect("cfg graft: block_return_tos empty stack");
                        bcode.push(Ir::Move {
                            dst: result_vreg,
                            src: v,
                        });
                        bcode.push(Ir::Jump {
                            target: continuation_id,
                        });
                        returned = true;
                        break;
                    }
                    // S24 B5 (Block graft only): `^expr` — a return from the
                    // HOME compilation (the block's home is the root; B4's
                    // materializer closure synthesis makes any in-graft deopt's
                    // interpreted nlr_tos sound). No join edge: control leaves M.
                    Instr::NlrTos if is_block_graft => {
                        let val = cstack.pop().expect("cfg graft: nlr_tos empty stack");
                        self.spliced_nlr += 1;
                        bcode.push(Ir::Ret { val });
                        returned = true;
                        break;
                    }
                    // S24 B5 (Block graft only): depth-0 ctx-temp access routes
                    // through the HOME Context — identical to the linear splice
                    // walk's arms (materialize form via method_ctx_vreg, elided
                    // form via ctx_vregs).
                    Instr::PushCtxTemp { depth, idx } if is_block_graft => {
                        debug_assert_eq!(depth, 0, "graft ctx-temp access must be depth 0");
                        let dst = self.fresh(true);
                        if let Some(ctx) = self.method_ctx_vreg {
                            bcode.push(Ir::LoadField {
                                dst,
                                obj: ctx,
                                byte_off: (crate::oops::layout::BODY_OFFSET
                                    + 8 * (2 + idx as usize))
                                    as i32,
                            });
                        } else {
                            bcode.push(Ir::Move {
                                dst,
                                src: self.ctx_vregs[idx as usize],
                            });
                        }
                        cstack.push(dst);
                    }
                    Instr::StoreCtxTempPop { depth, idx } if is_block_graft => {
                        debug_assert_eq!(depth, 0, "graft ctx-temp access must be depth 0");
                        let v = cstack
                            .pop()
                            .expect("cfg graft: store_ctx_temp_pop empty stack");
                        if let Some(ctx) = self.method_ctx_vreg {
                            bcode.push(Ir::StoreField {
                                obj: ctx,
                                byte_off: (crate::oops::layout::BODY_OFFSET
                                    + 8 * (2 + idx as usize))
                                    as i32,
                                val: v,
                                barrier: true,
                            });
                        } else {
                            bcode.push(Ir::Move {
                                dst: self.ctx_vregs[idx as usize],
                                src: v,
                            });
                        }
                    }
                    // Branch/jump opcodes are the block's own terminator —
                    // handled below from the decoded `Terminator` (they need the
                    // remapped target ids).
                    Instr::JumpFwd(_)
                    | Instr::JumpBack(_)
                    | Instr::BrTrueFwd(_)
                    | Instr::BrFalseFwd(_) => {}
                    other => unreachable!(
                        "cfg splice: {other:?} passed the graft eligibility gate but has no arm"
                    ),
                }
                bci = next;
            }

            if !trapped && !returned {
                match cfg_block.terminator {
                    crate::compiler::decode::Terminator::Return => {
                        unreachable!(
                            "cfg splice: a Return-terminated callee block must end in a \
                             return instruction"
                        )
                    }
                    crate::compiler::decode::Terminator::Fallthrough(t) => {
                        reachable[t] = true;
                        emit_merges(&sources, &entry_stacks, t, &cstack, &mut bcode);
                        bcode.push(Ir::Jump { target: ir_ids[t] });
                    }
                    crate::compiler::decode::Terminator::Jump {
                        target,
                        is_backward,
                    } => {
                        reachable[target] = true;
                        if is_backward {
                            // In-body loop poll: reexecute at the CALLEE's
                            // loop-header bci, InlineSite-chained (a mid-loop
                            // NotEntrant deopt rebuilds callee + caller frames).
                            let poll_ci = bcode.len() as u32;
                            bcode.push(Ir::Poll);
                            bdeopt.push((
                                poll_ci,
                                DeoptRaw {
                                    stack: cstack.clone(),
                                    bci: ccfg.blocks[target].bci_start,
                                    kind: SafepointKind::LoopPoll,
                                    reexecute: true,
                                    stack_closures: Vec::new(),
                                    inline: Some(inline_proto.clone()),
                                },
                            ));
                        }
                        emit_merges(&sources, &entry_stacks, target, &cstack, &mut bcode);
                        bcode.push(Ir::Jump {
                            target: ir_ids[target],
                        });
                    }
                    crate::compiler::decode::Terminator::Branch { if_true, if_false } => {
                        reachable[if_true] = true;
                        reachable[if_false] = true;
                        let val = *cstack
                            .last()
                            .expect("cfg splice: branch with empty simulated stack");
                        // not_bool reexecute snapshot INCLUDES the condition
                        // (the interpreter re-tests it); the sim stack pops it
                        // for the successors (strictly-correct accounting).
                        let not_bool_id = self.fresh_inlined_trap_block(
                            last_instr_bci,
                            cstack.clone(),
                            inline_proto.clone(),
                        );
                        cstack.pop();
                        emit_merges(&sources, &entry_stacks, if_true, &cstack, &mut bcode);
                        emit_merges(&sources, &entry_stacks, if_false, &cstack, &mut bcode);
                        bcode.push(Ir::BoolBr {
                            val,
                            if_true: ir_ids[if_true],
                            if_false: ir_ids[if_false],
                            not_bool: not_bool_id,
                        });
                    }
                }
            }

            exit_stacks[b] = Some(cstack);
            self.finish_block(IrBlock {
                id: cur_id,
                bci: send_bci,
                code: bcode,
                entry_stack: if cur_id == ir_ids[b] {
                    entry_stack
                } else {
                    Vec::new() // split-off tail segment; entry never consulted
                },
                deopt_sites: bdeopt,
            });
        }

        self.budget_commit(callee);
        if !is_block_graft {
            self.record_inline_dep(dep_klass, selector);
        }
        Some((result_vreg, ir_ids[0], continuation_id))
    }

    /// S24 B5 step 4: commit a grafted body's bytes to the cumulative
    /// budget. Called at every splice success exit (leaf, non-leaf, CFG in
    /// both graft modes, and the linear block-body walk) — committed sites
    /// included, so the counter reflects real emitted volume.
    fn budget_commit(&mut self, callee: MethodOop) {
        self.budget_bytes_used = self
            .budget_bytes_used
            .saturating_add(crate::compiler::inline::inline_cost(callee));
    }

    /// Whether grafting `callee` would push the compile past
    /// `InlineBudget::total_bytes`. Consulted ONLY at declinable decisions.
    fn budget_would_exceed(&self, callee: MethodOop) -> bool {
        let budget = crate::compiler::inline::budget_for_level(self.level);
        self.budget_bytes_used
            .saturating_add(crate::compiler::inline::inline_cost(callee))
            > budget.total_bytes
    }

    /// Whether an `Untaken` (cold-IC) send may be speculatively lowered to a
    /// terminating uncommon trap (S14 step 3) rather than a real `CallSend`.
    /// True only for a whole-body-profiled compile — i.e. NOT an OSR entry.
    /// See the `osr` field's doc: under OSR the not-yet-dispatched site is
    /// "not reached yet", not "cold", so a trap there deopts on loop exit
    /// (the sieve thrash — a compiled method that could never STAY compiled).
    /// Reliability over speculation: emit the plain dispatch instead. Applies
    /// uniformly to the four cold-send decision sites (root send, inlined
    /// leaf/nonleaf body, block-arg in-callee graft, spliced block body).
    fn cold_send_traps(&self) -> bool {
        !self.osr
    }

    /// Whether a COLD (`Untaken`) send spliced from an INLINED callee body may
    /// be speculatively lowered to a terminating uncommon trap. Always `false`
    /// — the inlined counterpart to `cold_send_traps`, split off because the
    /// two heal DIFFERENTLY. A ROOT send's trap re-executes in THIS method and
    /// warms THIS method's own IC, so `recompile::note_uncommon_trap`'s profile
    /// snapshot sees the change and recompiles it away. An INLINED send's trap
    /// re-executes in the CALLEE and warms the CALLEE's IC — which the caller's
    /// snapshot cannot reliably observe, so the recompile DECLINES ("profile
    /// unchanged") and the trap fires FOREVER. Found in the wild as
    /// `ScaleConstraint>>markInputs:` (grafts its OWN `inputsDo:` override,
    /// whose `direction == #forward` site is cold when the customized
    /// `markInputs:` compiles): the branch is taken every call, so v0 trapped
    /// ~102×/run and every resnapshot declined to recompile — a permanent deopt
    /// storm. `EqualityConstraint>>markInputs:` escaped it only by luck (it
    /// inlines the SHARED `BinaryConstraint>>inputsDo:`, warm from every other
    /// binary constraint). Reliability over speculation — the same principle
    /// `cold_send_traps` already applies to OSR: emit the plain `CallSend`,
    /// which dispatches correctly on the first call and never storms. A
    /// genuinely-cold inlined path just compiles as an unreached dispatch — a
    /// few bytes, zero runtime cost — instead of a trap.
    fn inlined_cold_send_traps(&self) -> bool {
        // Test/diagnostic override (`DebugState::cold_splice_traps`): the
        // deopt-materializer pins opt back into the pre-4c575c8 trap
        // lowering per-VM.
        if self.vm.debug.cold_splice_traps {
            return self.cold_send_traps();
        }
        false
    }

    fn record_inline_dep(&mut self, klass: KlassOop, selector: SymbolOop) {
        // Dedup: an inlined method reached via two identical sites need only one
        // dependency (the invalidation is idempotent, but keeping the vec tight
        // keeps the driver's copy small).
        let pair = (klass.oop().raw(), selector.oop().raw());
        if !self
            .inline_deps
            .iter()
            .any(|(k, s)| (k.oop().raw(), s.oop().raw()) == pair)
        {
            self.inline_deps.push((klass, selector));
        }
    }

    /// Translates every instruction except a block's own terminator
    /// (`Jump`/`Branch`-family and `Return`-family are structurally
    /// distinct: `Return` needs nothing beyond what's already on hand and
    /// is handled directly here, but `Jump`/`Branch` need the *already-
    /// resolved* CFG target — decode.rs's job — plus merge-vreg info that
    /// spans the whole method, so `convert` handles those itself once this
    /// loop finishes). `fusable` tells a comparison send whether it's
    /// eligible for the branch-fusion peephole (D3.2): true only when this
    /// instruction is immediately followed by nothing but the block's own
    /// terminator AND that terminator is specifically a `Branch` (a
    /// comparison can just as easily be immediately followed by a `Jump`
    /// or `Return` instead — `^ x < y` returns the plain boolean, no
    /// branch at all — so both conditions matter, not just position).
    /// Returns `Some(continuation_id)` exactly when this instruction just
    /// split the CURRENT block (a smi-inlined `SmiArith`/`SmiCmpVal` fail
    /// edge, S11 step 7's own `fail_and_continue`) — the caller (`convert`'s
    /// own per-CFG-block loop) must then finish its accumulating `code` vec
    /// with a `Jump { target: continuation_id }` and start a fresh one for
    /// the returned id. `None` for every other instruction, including
    /// `SmiCmpBr`'s own fused fail (handled entirely at the BRANCH
    /// terminator, which already has its own real targets — no split
    /// needed there) and the fast, non-failing smi-inline path.
    ///
    /// S14 step 3: `trapped` is set `true` when a generic send site's feedback
    /// is `Untaken` and [`crate::compiler::inline::decide`] lowers it to an
    /// uncommon trap (`Ir::UncommonTrap`) INSTEAD of an `Ir::CallSend`. A trap
    /// is a terminator — control leaves the compiled method entirely — so the
    /// caller (`convert`'s per-CFG-block loop) must finish the current `IrBlock`
    /// at the trap and skip both the rest of this CFG block's instructions and
    /// its own terminator (all dead after the trap). Distinct from `split`,
    /// which ends a block WITH a continuation to fall into; a trapped block has
    /// no continuation.
    #[allow(clippy::too_many_arguments)]
    fn translate_instr(
        &mut self,
        instr: &Instr,
        fusable: bool,
        bci: usize,
        stack: &mut Vec<VReg>,
        stack_sites: &mut Vec<Option<usize>>,
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
        pending_cmp: &mut Option<PendingCmp>,
        trapped: &mut bool,
    ) -> Option<BlockId> {
        debug_assert_eq!(
            stack.len(),
            stack_sites.len(),
            "translate_instr: operand-stack / block-site shadow desync"
        );
        // S14 step 7-I: handle every opcode that can create, move, or consume a
        // phantom block-site FIRST, keeping `stack`/`stack_sites` in lockstep,
        // and return early. A `push_closure` proven elidable emits NO IR (the
        // block body is spliced at its value-send); the temp/dup/pop/value-send
        // arms propagate or consume the site. Reached only for a method the
        // escape pre-pass proved `all_elidable` (`self.escape.is_some()`), so a
        // site here is always a good, spliceable use. Every OTHER opcode falls
        // through to the main match below, after which `stack_sites` is resynced
        // to `stack`'s new length with `None` (those opcodes never touch a site).
        if self.escape.is_some() {
            if let Some(r) =
                self.translate_site_instr(instr, bci, stack, stack_sites, code, deopt, trapped)
            {
                return r;
            }
        }
        let mut split: Option<BlockId> = None;
        match *instr {
            Instr::PushSelf => stack.push(self.self_vreg),
            Instr::PushNil => {
                stack.push(self.push_well_known(self.vm.universe.nil_obj.raw(), code))
            }
            Instr::PushTrue => {
                stack.push(self.push_well_known(self.vm.universe.true_obj.raw(), code))
            }
            Instr::PushFalse => {
                stack.push(self.push_well_known(self.vm.universe.false_obj.raw(), code))
            }
            Instr::PushSmi(v) => {
                let dst = self.fresh(true);
                code.push(Ir::ConstSmi {
                    dst,
                    value: v as i64,
                });
                stack.push(dst);
            }
            Instr::PushLiteral(idx) => {
                let lit_oop = self.method.literals().at(idx as usize);
                stack.push(self.push_well_known(lit_oop.raw(), code));
            }
            Instr::PushTemp(n) => {
                // Always a fresh copy (v1 rule, D3.2): temp_vregs[n] may be
                // reassigned later on this path; regalloc coalescing away
                // the redundant Move is a later nicety, not S10's job.
                let dst = self.fresh(true);
                code.push(Ir::Move {
                    dst,
                    src: self.temp_vregs[n as usize],
                });
                stack.push(dst);
            }
            Instr::StoreTemp(n) => {
                let tos = *stack.last().expect("store_temp: empty simulated stack");
                code.push(Ir::Move {
                    dst: self.temp_vregs[n as usize],
                    src: tos,
                });
            }
            Instr::StoreTempPop(n) => {
                let v = stack.pop().expect("store_temp_pop: empty simulated stack");
                code.push(Ir::Move {
                    dst: self.temp_vregs[n as usize],
                    src: v,
                });
            }
            Instr::PushInstvar(n) => {
                let dst = self.fresh(true);
                code.push(Ir::LoadField {
                    dst,
                    obj: self.self_vreg,
                    byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                });
                stack.push(dst);
            }
            Instr::PushGlobal(idx) => {
                let assoc = self.method.literals().at(idx as usize);
                let assoc_vreg = self.push_well_known(assoc.raw(), code);
                let dst = self.fresh(true);
                // Association body word 1 = value (word 0 = key), matching
                // the interpreter's own `assoc.body_oop(1)`.
                code.push(Ir::LoadField {
                    dst,
                    obj: assoc_vreg,
                    byte_off: (BODY_OFFSET + 8) as i32,
                });
                // S11 D7: if this global's CURRENT value is a class, remember
                // `dst` holds that class constant, so an immediately
                // following `X basicNew` send can compile to an inline
                // `Ir::Alloc` (`alloc_site_klass`). Reading the value here at
                // compile time is the deliberate S13-deferred staleness hole
                // (see `const_class`' own doc). The `LoadField` above still
                // emits — it's dead if the send turns into an `Alloc` (which
                // bakes the klass from the pool), a negligible cost a future
                // peephole could elide.
                let value = crate::oops::wrappers::MemOop::try_from(assoc)
                    .map(|a| a.body_oop(1))
                    .unwrap_or(assoc);
                if let Some(klass) = KlassOop::try_from(value) {
                    self.const_class.insert(dst.0, klass);
                }
                stack.push(dst);
            }
            Instr::Pop => {
                stack.pop().expect("pop: empty simulated stack");
            }
            Instr::Dup => {
                let tos = *stack.last().expect("dup: empty simulated stack");
                stack.push(tos);
            }
            Instr::Send { ic, super_ } if super_ => {
                // D4.6: the target METHOD is resolved at compile time
                // (`driver::compile_method`, after this whole IrMethod
                // comes back) via `holder.superclass()` — fixed here only
                // as far as WHICH klass to start that lookup from; method
                // holders don't move between klasses in v1, so this
                // klass itself never goes stale. Ordinary `Ir::CallSend`,
                // same as any other send site — `CallSiteInfo::
                // static_klass` is what tells `compile_method` to
                // pre-resolve and seed `Mono` instead of patching to
                // `stub_resolve` and starting `Unresolved`.
                let ic_view = InterpreterIc::at(self.method, ic);
                let selector = ic_view.selector();
                let real_argc = ic_view.argc();
                let mut real_args: Vec<VReg> = (0..real_argc)
                    .map(|_| stack.pop().expect("send_super: missing arg operand"))
                    .collect();
                real_args.reverse();
                let receiver = stack.pop().expect("send_super: missing receiver operand");
                let mut args = vec![receiver];
                args.append(&mut real_args);

                let holder = KlassOop::try_from(self.method.holder())
                    .expect("send_super: compiled method with no installed holder");
                let super_klass = KlassOop::try_from(holder.superclass())
                    .expect("send_super: holder's own superclass field is not a klass");

                // S14 step 5 (static devirtualization): a super send's target
                // is STATIC — `lookup(holder.superclass, selector)` is fixed at
                // compile time, and the receiver is `self`, whose klass the
                // nmethod's own entry check already proved. So a cheap-enough
                // super target inlines with NO GUARD AT ALL (`guard: None`) —
                // the whole per-call dispatch disappears. The inline dep is
                // recorded against the RESOLVED super klass, so redefining the
                // target still invalidates this nmethod.
                let budget = crate::compiler::inline::budget_for_level(self.level);
                if let Some(target) =
                    crate::compiler::feedback::resolve_method_ro(self.vm, super_klass, selector)
                {
                    // Same shape gates as `decide_with_budget` (and the step-5
                    // self-send devirt): a closure/ctx/super-bearing callee has
                    // no splicer arm — without these the walk aborts on
                    // `unreachable!` (review finding; pre-existing here, fixed
                    // alongside step 5's identical gate).
                    if target.primitive() == 0
                        && crate::compiler::inline::inline_cost(target) <= budget.per_call_cost
                        && (crate::compiler::inline::is_leaf(target)
                            || crate::compiler::inline::is_inline_eligible_nonleaf(target)
                            || crate::compiler::inline::is_inline_eligible_cfg(target))
                        // S24 B5 step 4: cumulative total_bytes — declinable
                        // here (falls to the plain send tail), evaluated LAST
                        // so a shape-ineligible site never counts as a
                        // budget decline.
                        && {
                            let over = self.budget_would_exceed(target);
                            if over {
                                self.splice_declined_budget += 1;
                            }
                            !over
                        }
                    {
                        let pre_pop_stack = {
                            let mut s = stack.clone();
                            s.push(receiver);
                            s.extend_from_slice(&args[1..]);
                            s
                        };
                        if let Some(result) = self.try_inline_leaf(
                            target,
                            None,
                            super_klass,
                            selector,
                            receiver,
                            &args[1..],
                            bci,
                            &pre_pop_stack,
                            code,
                        ) {
                            stack.push(result);
                            return split;
                        }
                        match self.try_inline_nonleaf(
                            target,
                            None,
                            super_klass,
                            selector,
                            receiver,
                            &args[1..],
                            bci,
                            &pre_pop_stack,
                            code,
                            deopt,
                        ) {
                            Some(NonLeafOutcome::Value(result)) => {
                                stack.push(result);
                                return split;
                            }
                            Some(NonLeafOutcome::Trapped) => {
                                *trapped = true;
                                return split;
                            }
                            None => {}
                        }
                        if let Some((result, entry_id, continuation_id)) = self.try_inline_cfg(
                            target,
                            None,
                            super_klass,
                            selector,
                            receiver,
                            &args[1..],
                            bci,
                            &pre_pop_stack,
                            code,
                            deopt,
                            None,
                            GraftMode::Method,
                        ) {
                            stack.push(result);
                            debug_assert!(self.pending_jump_target.is_none());
                            self.pending_jump_target = Some(entry_id);
                            return Some(continuation_id);
                        }
                    }
                }

                let site = self.call_sites.len() as u16;
                self.call_sites.push(CallSiteInfo {
                    selector,
                    argc: real_argc + 1,
                    static_klass: Some(super_klass),
                });
                // S14 step 2: annotate the site with its observed feedback (the
                // inliner prefers the STATIC super_klass above, but the field is
                // kept parallel to `call_sites` for every site uniformly).
                self.site_feedback
                    .push(crate::compiler::feedback::read_send_site(
                        self.vm,
                        self.method,
                        ic,
                        None,
                    ));

                let dst = self.fresh(true);
                // S13 step 3b: a call-return safepoint (reexecute=false). The
                // receiver+args are already popped and `dst` is not yet
                // pushed, so `stack` is exactly the operand stack the deopt
                // materializer restores BELOW the send; it pushes the
                // incoming result itself. `code.len()` is the index the
                // CallSend is about to occupy (driver correlates it to the
                // emitted return-address safepoint).
                deopt.push((
                    code.len() as u32,
                    DeoptRaw {
                        stack: stack.clone(),
                        bci,
                        kind: SafepointKind::Call,
                        reexecute: false,
                        stack_closures: Vec::new(),
                        inline: None,
                    },
                ));
                code.push(Ir::CallSend { dst, site, args });
                stack.push(dst);
            }
            Instr::Send { ic, .. }
                if self.is_smi_inlinable(ic) && !smi_fuse_arg_is_pooled(stack, code) =>
            {
                // S13 step 7b: a smi-overflow deopt is `reexecute=true` at the
                // SEND's bci — the interpreter re-executes the WHOLE send, so
                // the recorded operand stack must be the state BEFORE this
                // op's own pops, with ITS INPUTS (`a`, `b`) still present.
                // Snapshot `stack` HERE, before the two pops below, so the
                // snapshot is exactly `[below..., a, b]` = the reexecute stack.
                let reexec_stack = stack.clone();
                let b = stack.pop().expect("send: missing arg operand");
                let a = stack.pop().expect("send: missing receiver operand");
                match classify_smi_send(self.vm, self.method, ic) {
                    SmiSendKind::Cmp(op) if fusable => {
                        // `fusable` already confirms the terminator is a
                        // Branch -- `convert` consumes `pending_cmp` there.
                        // Carry the reexecute snapshot AND this send's own bci
                        // to the branch site, which builds the fail block
                        // (after a/b are popped, and where `bci` is the block
                        // start, not this send).
                        *pending_cmp = Some(PendingCmp::Smi {
                            op,
                            a,
                            b,
                            deopt: PendingCmpDeopt {
                                bci,
                                stack: reexec_stack,
                            },
                        });
                    }
                    SmiSendKind::Cmp(op) => {
                        let dst = self.fresh(true);
                        let (fail_id, continuation_id) = self.fail_and_continue(bci, reexec_stack);
                        code.push(Ir::SmiCmpVal {
                            op,
                            dst,
                            a,
                            b,
                            fail: fail_id,
                        });
                        stack.push(dst);
                        split = Some(continuation_id);
                    }
                    SmiSendKind::Arith(op) => {
                        let dst = self.fresh(true);
                        let (fail_id, continuation_id) = self.fail_and_continue(bci, reexec_stack);
                        code.push(Ir::SmiArith {
                            op,
                            dst,
                            a,
                            b,
                            fail: fail_id,
                        });
                        stack.push(dst);
                        split = Some(continuation_id);
                    }
                }
            }
            // S15 (the sieve fix): mono-Array `at:`/`at:put:` intrinsified —
            // the same fusion smi arithmetic got in S14. The fail edge is a
            // reexecute-true trap that re-runs the send interpreted, so
            // bounds errors, non-Array receivers, and non-smi indices keep
            // byte-identical Smalltalk semantics (the primitive's own
            // fallback path).
            Instr::Send { ic, .. } if self.array_op_kind(self.method, ic).is_some() => {
                let is_put = self
                    .array_op_kind(self.method, ic)
                    .expect("guard just matched");
                let reexec_stack = stack.clone();
                let klass = self.pool.intern(
                    self.vm.universe.array_klass.oop().raw(),
                    Some(RelocKind::Oop),
                );
                // Fail-only trap block, NO continuation split: the intrinsic
                // is an ordinary mid-block op (exactly like the splicers'
                // in-body fused ops) — splitting here collided with branch
                // TERMINATORS when the send feeds `ifTrue:` directly (the
                // split's continuation left the result outside the block the
                // terminator's depth accounting expected — the sieve warm-
                // compile merge-depth panic).
                let fail_id = self.fresh_block_id();
                self.finish_block(IrBlock {
                    id: fail_id,
                    bci,
                    code: vec![Ir::UncommonTrap { bci }],
                    entry_stack: Vec::new(),
                    deopt_sites: vec![(
                        0,
                        DeoptRaw {
                            stack: reexec_stack,
                            bci,
                            kind: SafepointKind::UncommonTrap,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    )],
                });
                let dst = self.fresh(true);
                if is_put {
                    let val = stack.pop().expect("at:put:: missing value operand");
                    let idx = stack.pop().expect("at:put:: missing index operand");
                    let arr = stack.pop().expect("at:put:: missing receiver operand");
                    code.push(Ir::ArrayAtPut {
                        dst,
                        arr,
                        idx,
                        val,
                        klass,
                        fail: fail_id,
                    });
                } else {
                    let idx = stack.pop().expect("at:: missing index operand");
                    let arr = stack.pop().expect("at:: missing receiver operand");
                    code.push(Ir::ArrayAt {
                        dst,
                        arr,
                        idx,
                        klass,
                        fail: fail_id,
                    });
                }
                stack.push(dst);
            }
            // Float fast-path (docs/float_fastpath_design.md B2 rule 1): a
            // mono-Double arithmetic/compare send collapses to guarded
            // unboxes + one native FP op (+ a box for arithmetic). Same
            // fail-only trap-block shape as the array intrinsics above: the
            // fail edge re-executes the WHOLE send interpreted (a non-Double
            // operand — the IC only ever guards the receiver — takes the
            // primitive's own Smalltalk fallback, byte-identical semantics).
            Instr::Send { ic, .. } if self.is_double_inlinable(ic) => {
                let reexec_stack = stack.clone();
                let fail_id = self.fresh_block_id();
                self.finish_block(IrBlock {
                    id: fail_id,
                    bci,
                    code: vec![Ir::UncommonTrap { bci }],
                    entry_stack: Vec::new(),
                    deopt_sites: vec![(
                        0,
                        DeoptRaw {
                            stack: reexec_stack,
                            bci,
                            kind: SafepointKind::UncommonTrap,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    )],
                });
                let b = stack.pop().expect("double fuse: missing arg operand");
                let a = stack.pop().expect("double fuse: missing receiver operand");
                // A compile-time SmallInteger arg (`aFloat < 0`, `aFloat * 2`)
                // can NEVER be FUnbox'd — a tagged smi is not a boxed Double, so
                // the unbox fails on EVERY call and the whole fused send deopts
                // forever: a STEADY-STATE trap storm, because the literal is
                // invariant so the recompile snapshot never changes and keeps
                // declining ("profile unchanged"). Left the whole world's
                // `Double>>truncated` (`^self < 0 ifTrue:…`) at ~1200 deopts per
                // test-suite run. The interpreter fallback `Double>>< aNumber`
                // takes is `^self < aNumber asDouble` — a plain smi→f64
                // coercion — so do exactly that at COMPILE time via `FConst`,
                // byte-identical (`n asDouble` is the same `n as f64`, same
                // precision loss for a >2^53 smi). The hot Double-vs-Double path
                // is UNTOUCHED: its arg is not a `ConstSmi`, so it falls through
                // to the unbox below. The arg's `ConstSmi` is deliberately left
                // in `code` — the fail edge's `reexec_stack` still names `b`, so
                // a receiver-guard miss re-executes correctly.
                let arg_const_smi: Option<i64> = match code.last() {
                    Some(Ir::ConstSmi { dst, value }) if *dst == b => Some(*value),
                    _ => None,
                };
                let ua = self.fresh_fp();
                let ub = self.fresh_fp();
                code.push(Ir::FUnbox {
                    dst: ua,
                    src: a,
                    fail: fail_id,
                });
                match arg_const_smi {
                    Some(v) => code.push(Ir::FConst {
                        dst: ub,
                        bits: (v as f64).to_bits(),
                    }),
                    None => code.push(Ir::FUnbox {
                        dst: ub,
                        src: b,
                        fail: fail_id,
                    }),
                }
                match classify_double_send(self.vm, self.method, ic) {
                    DoubleSendKind::Arith(op) => {
                        let fd = self.fresh_fp();
                        code.push(Ir::FArith {
                            op,
                            dst: fd,
                            a: ua,
                            b: ub,
                        });
                        let dst = self.fresh(true);
                        code.push(Ir::FBox { dst, src: fd });
                        stack.push(dst);
                    }
                    // The float analog of the smi fusion peephole (D3.2): when
                    // the comparison's only consumer is the block's own
                    // immediately-following branch, defer it to the terminator
                    // (`convert`) and emit ONE `FCmpBr` — `fcmp` + `b.cond` —
                    // instead of materializing a `true`/`false` OBJECT and then
                    // testing that object back against the true oop to branch on
                    // a condition `fcmp` already put in the flags. `fusable`
                    // itself has always been true here; only the smi arm ever
                    // read it, which left `Ir::FCmpBr` built-but-unreachable.
                    DoubleSendKind::Cmp(op) if fusable => {
                        *pending_cmp = Some(PendingCmp::Float { op, a: ua, b: ub });
                    }
                    DoubleSendKind::Cmp(op) => {
                        let dst = self.fresh(true);
                        code.push(Ir::FCmpVal {
                            op,
                            dst,
                            a: ua,
                            b: ub,
                        });
                        stack.push(dst);
                    }
                }
            }
            // SIMD NEON fast-path (docs/SIMD.md): a mono-vector arithmetic send
            // (+/-/*/ /) on a Float64x2 or Float32x4 collapses to ONE guarded
            // `Ir::VecArith` — emit lowers guard(both operands the right klass)
            // → `ldr q` unbox → `f… v.<arr>` → box internally. Same fail-only
            // trap-block shape as the double fuse above: a wrong-klass operand
            // (the IC only guards the receiver) re-executes the WHOLE send
            // interpreted, taking the primitive's own Smalltalk fallback —
            // byte-identical semantics.
            // Cog-parity special selectors (send census: richards spends
            // ~130k sends/run on identity `==` and ~90k on boolean `not`).
            // Identity lowers unguarded (pinned non-redefinable); `not`
            // lowers to a boolean-guarded flip with a reexecute-trap fail
            // edge. Phantom operands (escape-mode elided-closure fillers)
            // must not reach the UNGUARDED compare, hence the stack_sites
            // screens.
            Instr::Send { ic, .. }
                if self.ref_eq_kind(self.method, ic).is_some()
                    && !(self.escape.is_some()
                        && stack.len() >= 2
                        && (stack.len() - 2..stack.len()).any(|ix| {
                            stack_sites.get(ix).is_some_and(|s| s.is_some())
                        })) =>
            {
                let neq = self
                    .ref_eq_kind(self.method, ic)
                    .expect("guard just matched");
                let b = stack.pop().expect("==: missing arg operand");
                let a = stack.pop().expect("==: missing receiver operand");
                let dst = self.fresh(true);
                code.push(Ir::RefCmpVal { dst, a, b, neq });
                stack.push(dst);
            }
            Instr::Send { ic, .. }
                if self.bool_not_speculatable(self.method, ic)
                    && !(self.escape.is_some()
                        && !stack.is_empty()
                        && stack_sites
                            .get(stack.len() - 1)
                            .is_some_and(|s| s.is_some())) =>
            {
                let selector = InterpreterIc::at(self.method, ic).selector();
                self.record_inline_dep(self.vm.universe.true_klass, selector);
                self.record_inline_dep(self.vm.universe.false_klass, selector);
                // Fail-only trap block, no continuation split — the same
                // mid-block shape as the array intrinsics above.
                let reexec_stack = stack.clone();
                let fail_id = self.fresh_block_id();
                self.finish_block(IrBlock {
                    id: fail_id,
                    bci,
                    code: vec![Ir::UncommonTrap { bci }],
                    entry_stack: Vec::new(),
                    deopt_sites: vec![(
                        0,
                        DeoptRaw {
                            stack: reexec_stack,
                            bci,
                            kind: SafepointKind::UncommonTrap,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    )],
                });
                let src = stack.pop().expect("not: missing receiver operand");
                let dst = self.fresh(true);
                code.push(Ir::BoolNot {
                    dst,
                    src,
                    fail: fail_id,
                });
                stack.push(dst);
            }
            Instr::Send { ic, .. } if self.vec_send_kind(ic).is_some() => {
                let kind = self.vec_send_kind(ic).expect("guard just confirmed Some");
                let reexec_stack = stack.clone();
                let fail_id = self.fresh_block_id();
                self.finish_block(IrBlock {
                    id: fail_id,
                    bci,
                    code: vec![Ir::UncommonTrap { bci }],
                    entry_stack: Vec::new(),
                    deopt_sites: vec![(
                        0,
                        DeoptRaw {
                            stack: reexec_stack,
                            bci,
                            kind: SafepointKind::UncommonTrap,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    )],
                });
                let b = stack.pop().expect("vector fuse: missing arg operand");
                let a = stack.pop().expect("vector fuse: missing receiver operand");
                let op = classify_vec_send(self.vm, self.method, ic, kind);
                let dst = self.fresh(true);
                code.push(Ir::VecArith {
                    kind,
                    op,
                    dst,
                    a,
                    b,
                    fail: fail_id,
                });
                stack.push(dst);
            }
            // D1's "arbitrary send/send_w in ANY IC state": every send this
            // step's own smi-inline guard above doesn't handle (Poly, Mega,
            // mono-but-non-smi, mono-smi-but-not-inlinable) still compiles
            // now, as an ordinary generic `Ir::CallSend` -- no fast path
            // attempted, `stub_resolve` sorts out Unresolved/Mono/Pic/Mega
            // from here exactly like any other real send site.
            Instr::Send { ic, .. } => {
                // S11 D7: a known `X basicNew` (receiver is a compile-time
                // class constant, target is the basicNew primitive, X a
                // fixed-size Slots class) compiles to an inline `Ir::Alloc`
                // instead of a generic send. `stack.last()` is the receiver
                // exactly when the send takes no arguments -- which
                // `alloc_site_klass` requires (`ic.argc() == 0`) before it
                // ever consults `const_class`, so a non-zero-argc send whose
                // top-of-stack is an arg simply returns `None` here.
                // Phantom guard for the customized-self form: an elided
                // closure pushes `self_vreg` as a placeholder (shadowed by a
                // `Some` in `stack_sites`) — that operand is NOT self, so it
                // must not fuse. Mirrors the self-devirt check below; the
                // const-class form is unaffected (a phantom slot only ever
                // holds `self_vreg`, never a `const_class` vreg).
                let phantom_self = self.escape.is_some()
                    && stack
                        .len()
                        .checked_sub(1)
                        .and_then(|ix| stack_sites.get(ix))
                        .is_some_and(|s| s.is_some());
                if let Some(receiver) = stack.last().copied().filter(|_| !phantom_self) {
                    if let Some((klass, size_words)) = self.alloc_site_klass(ic, receiver) {
                        // S13 step 3b: an `Alloc` deopts by RE-EXECUTING the
                        // `basicNew` send in the interpreter (reexecute=true),
                        // so its recorded stack must still carry the receiver
                        // (the class const) that the send consumes — capture
                        // BEFORE the pop below.
                        let deopt_stack = stack.clone();
                        stack.pop(); // consume the receiver
                        let klass_lit = self.pool.intern(klass.oop().raw(), Some(RelocKind::Oop));
                        let dst = self.fresh(true);
                        deopt.push((
                            code.len() as u32,
                            DeoptRaw {
                                stack: deopt_stack,
                                bci,
                                kind: SafepointKind::Alloc,
                                reexecute: true,
                                stack_closures: Vec::new(),
                                inline: None,
                            },
                        ));
                        code.push(Ir::Alloc {
                            dst,
                            klass: klass_lit,
                            size_words,
                        });
                        stack.push(dst);
                        return split;
                    }
                }

                // S14 step 5 (customization, A3): SELF-send devirtualization.
                // If the receiver is provably `self` — its stack slot holds
                // `self_vreg` and is not an escape-mode PHANTOM filler (elided
                // closures push `self_vreg` as a placeholder, shadowed by a
                // `Some` entry in `stack_sites`) — the target is
                // `lookup(rcvr_klass, selector)`, fixed at compile time: the
                // nmethod's entry guard already proved the receiver klass.
                // Static knowledge beats feedback (A1): a cheap target inlines
                // GUARD-FREE exactly like a super send (the dep still recorded
                // against K so redefinition invalidates); anything else stays a
                // plain lazy `CallSend`, but the cold-IC TRAP below is skipped —
                // a self-send needs no interpreted warm-up to know its target.
                let self_send_target: Option<MethodOop> = {
                    let ic_view = InterpreterIc::at(self.method, ic);
                    let real_argc = ic_view.argc() as usize;
                    stack
                        .len()
                        .checked_sub(real_argc + 1)
                        .filter(|&rcvr_ix| {
                            stack[rcvr_ix] == self.self_vreg
                                && !(self.escape.is_some()
                                    && stack_sites.get(rcvr_ix).is_some_and(|s| s.is_some()))
                        })
                        .and_then(|_| self.devirt_self_target(self.rcvr_klass, ic_view.selector()))
                };
                if let Some(target) = self_send_target {
                    let ic_view = InterpreterIc::at(self.method, ic);
                    let selector = ic_view.selector();
                    let real_argc = ic_view.argc();
                    let budget = crate::compiler::inline::budget_for_level(self.level);
                    // Shape gates mirror `decide_with_budget` exactly: the
                    // splicers' body walks have no arms for closures /
                    // ctx-temps / super sends and would abort (or, for a
                    // release-mode super send, silently miscompile) on a
                    // callee the eligibility fns reject (review finding).
                    if target.primitive() == 0
                        && crate::compiler::inline::inline_cost(target) <= budget.per_call_cost
                        && (crate::compiler::inline::is_leaf(target)
                            || crate::compiler::inline::is_inline_eligible_nonleaf(target)
                            || crate::compiler::inline::is_inline_eligible_cfg(target))
                        // S24 B5 step 4: cumulative total_bytes — declinable
                        // here (falls to the plain send tail), evaluated LAST
                        // so a shape-ineligible site never counts as a
                        // budget decline.
                        && {
                            let over = self.budget_would_exceed(target);
                            if over {
                                self.splice_declined_budget += 1;
                            }
                            !over
                        }
                    {
                        let pre_pop_stack = stack.clone();
                        let mut real_args: Vec<VReg> = (0..real_argc)
                            .map(|_| stack.pop().expect("self send: missing arg operand"))
                            .collect();
                        real_args.reverse();
                        let receiver = stack.pop().expect("self send: missing receiver operand");
                        debug_assert_eq!(receiver, self.self_vreg);
                        if let Some(result) = self.try_inline_leaf(
                            target,
                            None,
                            self.rcvr_klass,
                            selector,
                            receiver,
                            &real_args,
                            bci,
                            &pre_pop_stack,
                            code,
                        ) {
                            self.self_devirt = true;
                            stack.push(result);
                            return split;
                        }
                        match self.try_inline_nonleaf(
                            target,
                            None,
                            self.rcvr_klass,
                            selector,
                            receiver,
                            &real_args,
                            bci,
                            &pre_pop_stack,
                            code,
                            deopt,
                        ) {
                            Some(NonLeafOutcome::Value(result)) => {
                                self.self_devirt = true;
                                stack.push(result);
                                return split;
                            }
                            Some(NonLeafOutcome::Trapped) => {
                                self.self_devirt = true;
                                *trapped = true;
                                return split;
                            }
                            None => {}
                        }
                        if let Some((result, entry_id, continuation_id)) = self.try_inline_cfg(
                            target,
                            None,
                            self.rcvr_klass,
                            selector,
                            receiver,
                            &real_args,
                            bci,
                            &pre_pop_stack,
                            code,
                            deopt,
                            None,
                            GraftMode::Method,
                        ) {
                            self.self_devirt = true;
                            stack.push(result);
                            debug_assert!(self.pending_jump_target.is_none());
                            self.pending_jump_target = Some(entry_id);
                            return Some(continuation_id);
                        }
                        // No splicer took it: restore the operands and fall
                        // through to the plain-CallSend tail below (which pops
                        // them again itself).
                        stack.push(receiver);
                        stack.extend_from_slice(&real_args);
                    }
                }

                // S14 step 3: consult the site's type feedback (read here,
                // parallel to S14 step 2's own `read_send_site` call) BEFORE
                // touching the operand stack. An `Untaken` site (IC still
                // `Empty` at compile time — now COMPILABLE as of this step's
                // eligibility relaxation, see `driver::mono_smi_inline_send`)
                // lowers to an uncommon TRAP instead of a real send: it never
                // executed interpreted, so there is no target to speculate on.
                // Everything else (`Mono`/`Poly`/`Mega`) stays a plain generic
                // `Ir::CallSend` (method/poly INLINING is S14 steps 4/6).
                // S14 step 5: a devirtualized self-send never traps — its
                // target is static, so a cold IC just means the site compiles
                // as the lazy `CallSend` below and resolves on first call.
                let feedback =
                    crate::compiler::feedback::read_send_site(self.vm, self.method, ic, None);
                if self_send_target.is_none() && self.cold_send_traps() {
                    if let crate::compiler::inline::InlineDecision::Trap =
                        crate::compiler::inline::decide(&feedback)
                    {
                        // The trap re-executes the WHOLE send in the interpreter
                        // (reexecute=true at THIS send's own bci), which populates
                        // the IC and returns the identical result — so the recorded
                        // operand stack must be the state BEFORE the send's pops,
                        // with the receiver + all args STILL PRESENT (snapshot
                        // `stack` HERE, before any pop below). Same
                        // reexecute-stack convention as the smi-overflow /
                        // basicNew-alloc / loop-poll sites; `regalloc::deopt_live`
                        // forces every recorded vreg live-across the trap so
                        // spill-all pins it to a frame slot the deopt materializer
                        // reads.
                        //
                        // We push NEITHER a `CallSiteInfo` NOR a `site_feedback`
                        // entry for a trapped site: it never dispatches, so it owns
                        // no `Ir::CallSend.site` index (the `call_sites`/
                        // `site_feedback` vecs stay parallel, indexed by real send
                        // sites only). And we push no `dst`: control leaves the
                        // method via the trap, so nothing after it in this block is
                        // reachable — `trapped` tells `convert` to finish the block
                        // here and skip the rest.
                        //
                        // INTERIM DEOPT-STORM (documented, NOT solved here): an
                        // EXECUTED trap re-executes the send interpreted and warms
                        // the IC, but this nmethod stays Alive with the trap still
                        // compiled, so the NEXT call re-traps — a storm until the
                        // method is recompiled with the now-warm feedback. That
                        // recompile-on-trap loop is S14 step 8 (recompile.rs) and
                        // needs `trap_counts` / `UNCOMMON_TRAP_LIMIT`, which S13
                        // never built. Step 3 accepts the storm: it is
                        // CORRECTNESS-PRESERVING (every trap re-executes the send
                        // exactly, identical output) — just slow, and bounded per
                        // run by the call count. No trap counting / recompilation
                        // is implemented here.
                        let reexec_stack = stack.clone();
                        deopt.push((
                            code.len() as u32,
                            DeoptRaw {
                                stack: reexec_stack,
                                bci,
                                kind: SafepointKind::UncommonTrap,
                                reexecute: true,
                                stack_closures: Vec::new(),
                                inline: None,
                            },
                        ));
                        code.push(Ir::UncommonTrap { bci });
                        *trapped = true;
                        return split; // (always None on the trap path)
                    }
                }

                // OSR-heal census: an `Untaken` site reaching the generic
                // CallSend below (the S15 de-speculation kept it from
                // trapping) marks this OSR compile HALF-WARM — see
                // `Nmethod::osr_cold_sends`.
                if self.osr
                    && matches!(
                        crate::compiler::inline::decide(&feedback),
                        crate::compiler::inline::InlineDecision::Trap
                    )
                {
                    self.osr_cold_sends += 1;
                }

                let ic_view = InterpreterIc::at(self.method, ic);
                let real_argc = ic_view.argc();

                // S14 step 4b: leaf-method inlining. Consult the budget-aware
                // decision on the SAME `feedback` already read for the trap
                // check. A `Mono` site whose callee is a cheap leaf splices
                // inline behind a receiver-klass guard (cold path = the SAME
                // step-3 uncommon trap, so a wrong receiver deopts + re-executes
                // the send generically); everything else stays a plain
                // `CallSend` below. The pre-pop operand `stack` (receiver + args
                // still present) is the reexecute stack the guard's cold trap
                // records — snapshot it BEFORE popping.
                let budget = crate::compiler::inline::budget_for_level(self.level);
                let smi_bits = self.vm.universe.smi_klass.oop().raw();
                // S14 step 5: devirtualized self-sends skip the feedback-driven
                // inline too — the devirt block above already tried the same
                // splicers guard-free, and a shared method's IC may have
                // observed a DIFFERENT customization's receiver klass, so a
                // guarded inline here would fail its guard on every call.
                let feedback_inline = if self_send_target.is_none() {
                    crate::compiler::inline::decide_with_budget(&feedback, &budget, smi_bits)
                } else {
                    crate::compiler::inline::InlineDecision::Call
                };
                // S24 B5 step 4: the cumulative `total_bytes` budget is
                // enforced HERE — a declinable decision whose fallback is the
                // plain `CallSend` tail below. (Committed block splices are
                // counted but never declined; amendment A1.)
                let feedback_inline = match feedback_inline {
                    crate::compiler::inline::InlineDecision::Inline { callee, .. }
                        if self.budget_would_exceed(callee) =>
                    {
                        self.splice_declined_budget += 1;
                        crate::compiler::inline::InlineDecision::Call
                    }
                    crate::compiler::inline::InlineDecision::DominantWithSlowPath {
                        case_method,
                        ..
                    } if self.budget_would_exceed(case_method) => {
                        self.splice_declined_budget += 1;
                        crate::compiler::inline::InlineDecision::Call
                    }
                    other => other,
                };
                // S14 step 6: a POLY site with a dominant case — inline the
                // dominant LEAF body behind a klass guard whose FAIL edge is a
                // REAL compiled send that REJOINS at a continuation (never a
                // trap: the minority receivers are known-taken; trapping would
                // deopt-storm, SPEC §8.4). Both paths write the same `dst`
                // vreg, so the continuation resumes with one canonical result
                // regardless of which side ran. The inline dep is recorded
                // against the DOMINANT klass (the guard's assumption); the
                // slow path is an ordinary lazy compiled-IC send needing none.
                if let crate::compiler::inline::InlineDecision::DominantWithSlowPath {
                    case_klass,
                    case_method,
                    guard,
                } = &feedback_inline
                {
                    let case_klass = *case_klass;
                    let case_method = *case_method;
                    // Same dry-run rule as `try_inline_leaf`: prove the WHOLE
                    // splice succeeds before emitting anything (the slow block
                    // is minted first, and an orphaned block would be dead
                    // weight; a half-spliced fast path would be corruption).
                    let callee_cfg = crate::compiler::decode::decode(case_method);
                    let spliceable = callee_cfg.blocks.len() == 1
                        && matches!(
                            callee_cfg.blocks[0].terminator,
                            crate::compiler::decode::Terminator::Return
                        )
                        && case_method.argc() == real_argc as usize
                        && leaf_body_is_spliceable(case_method);
                    if spliceable {
                        let selector = ic_view.selector();
                        let pre_pop_stack = stack.clone();
                        let mut inline_args: Vec<VReg> = (0..real_argc)
                            .map(|_| stack.pop().expect("dominant send: missing arg operand"))
                            .collect();
                        inline_args.reverse();
                        let receiver = stack
                            .pop()
                            .expect("dominant send: missing receiver operand");

                        // The rejoining SLOW block: a full generic send whose
                        // result is moved into the shared `dst`, then a jump to
                        // the continuation. Its call-return safepoint records
                        // the caller stack BELOW the send (reexecute=false),
                        // exactly like a root-level CallSend.
                        let continuation_id = self.fresh_block_id();
                        let slow_id = self.fresh_block_id();
                        let dst = self.fresh(true);
                        let dst_slow = self.fresh(true);
                        let mut send_args = vec![receiver];
                        send_args.extend_from_slice(&inline_args);
                        let site = self.call_sites.len() as u16;
                        self.call_sites.push(CallSiteInfo {
                            selector,
                            argc: real_argc + 1,
                            static_klass: None,
                        });
                        self.site_feedback.push(feedback.clone());
                        self.finish_block(IrBlock {
                            id: slow_id,
                            bci,
                            code: vec![
                                Ir::CallSend {
                                    dst: dst_slow,
                                    site,
                                    args: send_args,
                                },
                                Ir::Move { dst, src: dst_slow },
                                Ir::Jump {
                                    target: continuation_id,
                                },
                            ],
                            entry_stack: Vec::new(),
                            deopt_sites: vec![(
                                0,
                                DeoptRaw {
                                    stack: stack.clone(),
                                    bci,
                                    kind: SafepointKind::Call,
                                    reexecute: false,
                                    stack_closures: Vec::new(),
                                    inline: None,
                                },
                            )],
                        });

                        // FAST path, in the current block: the guard (fail →
                        // the rejoining slow block), the spliced dominant leaf
                        // body (guard=None — the guard above already dominates
                        // it; the dep vs case_klass is recorded inside), and
                        // the move into the shared result vreg.
                        let expect = self
                            .pool
                            .intern(case_klass.oop().raw(), Some(RelocKind::Oop));
                        let guard_shape = match guard {
                            crate::compiler::inline::GuardKind::SmiTest => GuardShape::SmiTest,
                            _ => GuardShape::KlassTest,
                        };
                        code.push(Ir::GuardKlass {
                            obj: receiver,
                            expect,
                            fail: slow_id,
                            kind: guard_shape,
                        });
                        let result = self
                            .try_inline_leaf(
                                case_method,
                                None,
                                case_klass,
                                selector,
                                receiver,
                                &inline_args,
                                bci,
                                &pre_pop_stack,
                                code,
                            )
                            .expect("dominant leaf was pre-validated spliceable");
                        code.push(Ir::Move { dst, src: result });
                        stack.push(dst);
                        return Some(continuation_id);
                    }
                    // Unspliceable dominant shape: fall through to the plain
                    // generic CallSend tail below (nothing was emitted).
                }

                if let crate::compiler::inline::InlineDecision::Inline { callee, guard } =
                    feedback_inline
                {
                    let guard_shape = match guard {
                        crate::compiler::inline::GuardKind::SmiTest => GuardShape::SmiTest,
                        // `KlassTest` and (defensively) `None` — a `None`-guard
                        // elision is a later step; today `decide_with_budget`
                        // only ever returns SmiTest/KlassTest for a real inline,
                        // so this maps the rest to a full klass test (never
                        // wrong, at worst one redundant compare).
                        _ => GuardShape::KlassTest,
                    };
                    // The observed receiver klass IS the guard klass; the callee
                    // is the resolved leaf method. The guard depends on
                    // `lookup(guard_klass, selector)` == callee.
                    let guard_klass = match &feedback {
                        crate::compiler::feedback::SiteFeedback::Mono { klass, .. } => *klass,
                        // Unreachable: `decide_with_budget` only returns `Inline`
                        // for a `Mono` site.
                        _ => unreachable!("Inline decision from a non-Mono feedback"),
                    };
                    let selector = ic_view.selector();
                    let pre_pop_stack = stack.clone();
                    // Pop the send's operands (they alias the callee's
                    // self/args — no stores).
                    let mut inline_args: Vec<VReg> = (0..real_argc)
                        .map(|_| stack.pop().expect("send: missing arg operand"))
                        .collect();
                    inline_args.reverse();
                    let receiver = stack.pop().expect("send: missing receiver operand");

                    if let Some(result) = self.try_inline_leaf(
                        callee,
                        Some((guard_klass, guard_shape)),
                        guard_klass,
                        selector,
                        receiver,
                        &inline_args,
                        bci,
                        &pre_pop_stack,
                        code,
                    ) {
                        // Inlined: push the body's result. NO CallSiteInfo /
                        // site_feedback / IcSite / call-return deopt for this
                        // site — it never dispatches (A2).
                        stack.push(result);
                        return split; // (None here — no smi split on this path)
                    }
                    // S14 step 4c: a NON-leaf callee (the leaf splice declined
                    // because the body has sends). Splice it inline with nested
                    // deopt scopes chained to the caller via a `SenderLink`. Its
                    // own sends become plain `CallSend`s (or step-3 traps) in the
                    // inlined body.
                    match self.try_inline_nonleaf(
                        callee,
                        Some((guard_klass, guard_shape)),
                        guard_klass,
                        selector,
                        receiver,
                        &inline_args,
                        bci,
                        &pre_pop_stack,
                        code,
                        deopt,
                    ) {
                        Some(NonLeafOutcome::Value(result)) => {
                            // Inlined: push the body's result. The inlined send
                            // itself dispatches nothing — only the body's OWN
                            // inner sends did (they pushed their CallSiteInfo /
                            // IcSite above).
                            stack.push(result);
                            return split;
                        }
                        Some(NonLeafOutcome::Trapped) => {
                            // An in-body cold send became a terminating uncommon
                            // trap: control leaves the method here, nothing after
                            // the inlined send in THIS block is reachable. Tell
                            // `convert` to finish the block at the trap (same as
                            // a root-level trapped send).
                            *trapped = true;
                            return split; // (always None on the trap path)
                        }
                        None => {}
                    }
                    // S14 step 7-IV-a: a MULTI-BLOCK callee (branches/loops —
                    // the single-straight-line splicers above declined). Map its
                    // whole CFG into fresh caller blocks; the send's value
                    // arrives in `result` via the callee's return edges, and the
                    // caller resumes in the returned continuation block.
                    if let Some((result, entry_id, continuation_id)) = self.try_inline_cfg(
                        callee,
                        Some((guard_klass, guard_shape)),
                        guard_klass,
                        selector,
                        receiver,
                        &inline_args,
                        bci,
                        &pre_pop_stack,
                        code,
                        deopt,
                        None,
                        GraftMode::Method,
                    ) {
                        stack.push(result);
                        debug_assert!(
                            self.pending_jump_target.is_none(),
                            "cfg splice: a pending jump target is already armed"
                        );
                        self.pending_jump_target = Some(entry_id);
                        return Some(continuation_id);
                    }
                    // Declined mid-way (an unspliceable shape slipped past the
                    // budget/leaf gate): fall through to a plain CallSend, but
                    // the operands are already popped — rebuild `args` from them.
                    let mut args = vec![receiver];
                    args.extend_from_slice(&inline_args);
                    let site = self.call_sites.len() as u16;
                    self.call_sites.push(CallSiteInfo {
                        selector,
                        argc: real_argc + 1,
                        static_klass: None,
                    });
                    self.site_feedback.push(feedback);
                    let dst = self.fresh(true);
                    deopt.push((
                        code.len() as u32,
                        DeoptRaw {
                            stack: stack.clone(),
                            bci,
                            kind: SafepointKind::Call,
                            reexecute: false,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    ));
                    code.push(Ir::CallSend { dst, site, args });
                    stack.push(dst);
                    return split;
                }

                let mut real_args: Vec<VReg> = (0..real_argc)
                    .map(|_| stack.pop().expect("send: missing arg operand"))
                    .collect();
                real_args.reverse();
                let receiver = stack.pop().expect("send: missing receiver operand");
                let mut args = vec![receiver];
                args.append(&mut real_args);

                let site = self.call_sites.len() as u16;
                self.call_sites.push(CallSiteInfo {
                    selector: ic_view.selector(),
                    argc: real_argc + 1,
                    static_klass: None,
                });
                // S14 step 2: annotate this send with its interpreter feedback
                // (already read above for the trap decision — kept parallel to
                // `call_sites` for every REAL send site).
                self.site_feedback.push(feedback);

                let dst = self.fresh(true);
                // S13 step 3b: call-return safepoint (reexecute=false), same
                // convention as the `send_super` site above — recorded stack
                // is the frame BELOW the send, result pushed by the deopt
                // materializer.
                deopt.push((
                    code.len() as u32,
                    DeoptRaw {
                        stack: stack.clone(),
                        bci,
                        kind: SafepointKind::Call,
                        reexecute: false,
                        stack_closures: Vec::new(),
                        inline: None,
                    },
                ));
                code.push(Ir::CallSend { dst, site, args });
                stack.push(dst);
            }
            Instr::JumpFwd(_) | Instr::JumpBack(_) | Instr::BrTrueFwd(_) | Instr::BrFalseFwd(_) => {
                // Deferred to `convert` (see this fn's own doc).
            }
            Instr::ReturnTos => {
                let val = stack.pop().expect("return_tos: empty simulated stack");
                code.push(Ir::Ret { val });
            }
            Instr::ReturnSelf => code.push(Ir::RetSelf),
            Instr::StoreInstvarPop(n) => {
                // Mirrors `PushInstvar`'s own `byte_off` exactly (D1's
                // store-relaxation is the write-side of that same field
                // access) -- `barrier: true` unconditionally: `val` could
                // be any heap oop at runtime regardless of what this
                // specific store site happens to see in practice, exactly
                // like `memory::store::store`'s own barrier, which never
                // skips the check based on the STATIC shape of a store.
                let val = stack
                    .pop()
                    .expect("store_instvar_pop: empty simulated stack");
                code.push(Ir::StoreField {
                    obj: self.self_vreg,
                    byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                    val,
                    barrier: true,
                });
            }
            Instr::StoreGlobalPop(idx) => {
                // Mirrors `PushGlobal`'s own association-value convention
                // exactly (word 1 = value, word 0 = key).
                let assoc = self.method.literals().at(idx as usize);
                let assoc_vreg = self.push_well_known(assoc.raw(), code);
                let val = stack
                    .pop()
                    .expect("store_global_pop: empty simulated stack");
                code.push(Ir::StoreField {
                    obj: assoc_vreg,
                    byte_off: (BODY_OFFSET + 8) as i32,
                    val,
                    barrier: true,
                });
            }
            // S14 step 7-II-b: M's own captured temp, promoted to `ctx_vregs`
            // (M's heap Context is elided). Only depth 0 (M's own Context) is
            // eligible — a `depth != 0` was rejected by `driver::eligible`.
            Instr::PushCtxTemp { depth, idx } => {
                debug_assert_eq!(depth, 0, "compiled ctx-temp access must be depth 0");
                if let Some(ctx) = self.block_ctx_vreg.or(self.method_ctx_vreg) {
                    // S24 A1 (design §2.2): inside a compiled BLOCK, depth-0
                    // ctx temps live in the REAL home Context object
                    // (`closure.copied[1]`, loaded into `ctx` by the entry
                    // prologue) — the compiled mirror of `activate_block`'s
                    // FP+3 aliasing. S24 A3b: in a MATERIALIZING method they
                    // live in the prologue-allocated Context (`method_ctx_vreg`)
                    // — the SAME object escaping closures capture (Risk 2).
                    // Context body layout (`oops::context`): [0]=home_hint,
                    // [1]=size slot, [2+i]=slot i.
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: ctx,
                        byte_off: (crate::oops::layout::BODY_OFFSET + 8 * (2 + idx as usize))
                            as i32,
                    });
                    stack.push(dst);
                } else {
                    let dst = self.fresh(true);
                    code.push(Ir::Move {
                        dst,
                        src: self.ctx_vregs[idx as usize],
                    });
                    stack.push(dst);
                }
            }
            Instr::StoreCtxTempPop { depth, idx } => {
                debug_assert_eq!(depth, 0, "compiled ctx-temp access must be depth 0");
                let v = stack
                    .pop()
                    .expect("store_ctx_temp_pop: empty simulated stack");
                if let Some(ctx) = self.block_ctx_vreg.or(self.method_ctx_vreg) {
                    // Real, barriered Context-slot store — the home
                    // (interpreted or otherwise) reads the SAME object, so
                    // identity and the card table both matter (Risk 2's
                    // one-Context invariant).
                    code.push(Ir::StoreField {
                        obj: ctx,
                        byte_off: (crate::oops::layout::BODY_OFFSET + 8 * (2 + idx as usize))
                            as i32,
                        val: v,
                        barrier: true,
                    });
                } else {
                    code.push(Ir::Move {
                        dst: self.ctx_vregs[idx as usize],
                        src: v,
                    });
                }
            }
            Instr::BlockReturnTos => {
                // S24 A1: a block's LOCAL return (fall-off-the-end) — the
                // exact `Ret` shape `return_tos` lowers to; delivered to the
                // value:-sender like any compiled return.
                debug_assert!(
                    self.method.is_block(),
                    "block_return_tos outside a block compilation"
                );
                let val = stack
                    .pop()
                    .expect("block_return_tos: empty simulated stack");
                code.push(Ir::Ret { val });
            }
            Instr::NlrTos => {
                // S24 A1 (design §2.4): non-local return origination.
                debug_assert!(
                    self.method.is_block(),
                    "nlr_tos outside a block compilation"
                );
                let value = stack.pop().expect("nlr_tos: empty simulated stack");
                let closure = self
                    .block_closure_vreg
                    .expect("nlr_tos: block compilation must carry the closure vreg");
                code.push(Ir::NlrReturn { closure, value });
            }
            Instr::PushClosure { lit, ncopied } => {
                // S24 A3a (design §2.7): allocate a REAL BlockClosure for an
                // ESCAPING site (the escape pass proved it transitively
                // NLR-free + non-`captures_ctx`; the driver gate accepted M).
                // Elidable sites never reach here — `translate_site_instr`
                // handles them as phantoms and returns `Some`. Pure IR
                // composition: `Ir::Alloc` (which nil-inits the whole body,
                // emit.rs) + straight-line `StoreField` initializers, no new
                // emit or stub. Layout (`oops::layout`, `make_closure` is the
                // interpreter twin): method@16, home@24, ncopied-size@32,
                // copied[j]@40+8j (copied[0]=self, value-captures @48+).
                debug_assert!(
                    self.escape
                        .as_ref()
                        .is_some_and(|e| e.is_escaping_site(bci)),
                    "PushClosure reached translate_instr but is not an escaping A3a site"
                );
                let block = MethodOop::try_from(self.method.literals().at(lit as usize))
                    .expect("push_closure literal is not a CompiledBlock");
                // S24 A3a: copied[0]=self only. A3b: a captures_ctx block also
                // takes copied[1] = M's MATERIALIZED Context (the SAME object,
                // Risk 2), so `value_base` (where the stack captures land)
                // shifts to 2. Mirrors `make_closure`'s `value_base = 1 +
                // captures_ctx`.
                let captures_ctx = block.captures_ctx();
                let value_base = 1 + captures_ctx as usize;
                debug_assert!(
                    !captures_ctx || self.method_ctx_vreg.is_some(),
                    "A3b: a captures_ctx escaping block requires M to have materialized its Context"
                );
                let n_value_captures = ncopied as usize;
                debug_assert!(
                    stack.len() >= n_value_captures,
                    "push_closure: {n_value_captures} value-captures but stack has {}",
                    stack.len()
                );
                let base = stack.len() - n_value_captures;
                // The captures are ordinary values, never elidable phantoms (a
                // closure captured into another closure ESCAPES, so the escape
                // pass never leaves it a live Site here).
                debug_assert!(
                    stack_sites[base..].iter().all(|s| s.is_none()),
                    "push_closure: a value-capture is a phantom block-site"
                );

                // A `push_closure` deopts by RE-EXECUTING (the interpreter
                // rebuilds the closure via `make_closure`), so its recorded
                // stack must still carry the value-captures the bytecode pops —
                // snapshot BEFORE the truncate below.
                let deopt_stack = stack.clone();
                deopt.push((
                    code.len() as u32,
                    DeoptRaw {
                        stack: deopt_stack,
                        bci,
                        kind: SafepointKind::Alloc,
                        reexecute: true,
                        stack_closures: Vec::new(),
                        inline: None,
                    },
                ));

                // total copied slots = value_base (self [+ ctx]) + value
                // captures. Body = [method, home, ncopied-size] (3) + copied
                // (`total`); +HEADER_WORDS(2). Mirrors `alloc_closure`'s own
                // `words = nis(4) + 1 + total`.
                let total = value_base + n_value_captures;
                let size_words = (crate::oops::layout::HEADER_WORDS + 3 + total) as u32;
                let closure_klass = self.vm.universe.closure_klass;
                let klass_lit = self
                    .pool
                    .intern(closure_klass.oop().raw(), Some(RelocKind::Oop));
                let dst = self.fresh(true);
                code.push(Ir::Alloc {
                    dst,
                    klass: klass_lit,
                    size_words,
                });
                // method@16
                let blk_lit = self.pool.intern(block.oop().raw(), Some(RelocKind::Oop));
                let blk_v = self.fresh(true);
                code.push(Ir::ConstPool {
                    dst: blk_v,
                    lit: blk_lit,
                });
                code.push(Ir::StoreField {
                    obj: dst,
                    byte_off: 16,
                    val: blk_v,
                    barrier: false,
                });
                // home@24 = the structurally-dead sentinel (a smi; §2.7)
                let home_v = self.fresh(false);
                code.push(Ir::ConstSmi {
                    dst: home_v,
                    value: crate::oops::home_ref::home_dead_sentinel().value(),
                });
                code.push(Ir::StoreField {
                    obj: dst,
                    byte_off: 24,
                    val: home_v,
                    barrier: false,
                });
                // ncopied-size word@32 = total (a smi; alloc_closure's own
                // `set_raw_body_word(size_idx, SmallInt::new(ncopied))`)
                let size_v = self.fresh(false);
                code.push(Ir::ConstSmi {
                    dst: size_v,
                    value: total as i64,
                });
                code.push(Ir::StoreField {
                    obj: dst,
                    byte_off: 32,
                    val: size_v,
                    barrier: false,
                });
                // copied[0]@40 = self (the home receiver — M is the home)
                code.push(Ir::StoreField {
                    obj: dst,
                    byte_off: 40,
                    val: self.self_vreg,
                    barrier: false,
                });
                // A3b: copied[1]@48 = M's materialized Context (the SAME object
                // M's own ctx-temp ops address — one Context by identity).
                if captures_ctx {
                    let ctx = self
                        .method_ctx_vreg
                        .expect("captures_ctx escaping block: M must have a materialized Context");
                    code.push(Ir::StoreField {
                        obj: dst,
                        byte_off: 48,
                        val: ctx,
                        barrier: false,
                    });
                }
                // copied[value_base+i]@40+8*(value_base+i) = value captures, in
                // stack order (deepest ⇒ copied[value_base]), matching
                // `make_closure`'s `set_copied(value_base+i, stack[base+i])`.
                // Initializing stores into a fresh young object ⇒ no barrier.
                for i in 0..n_value_captures {
                    let cap = stack[base + i];
                    code.push(Ir::StoreField {
                        obj: dst,
                        byte_off: (40 + 8 * (value_base + i)) as i32,
                        val: cap,
                        barrier: false,
                    });
                }
                stack.truncate(base);
                stack.push(dst);
            }
        }
        // S14 step 7-I: `stack_sites` is resynced to `stack`'s new length by the
        // CALLER (`convert`'s per-instruction loop), NOT here — `translate_instr`
        // has many early `return split` paths (Alloc, trap, inline, generic
        // send) that would otherwise skip a resync and leave the shadow stale.
        split
    }

    /// S14 step 7-I: intercept the opcodes that create, move, or consume a
    /// phantom block-site (a `push_closure` the escape pre-pass proved elidable)
    /// BEFORE `translate_instr`'s main match, keeping `stack` and its
    /// `stack_sites` shadow in lockstep. Returns:
    ///   - `None` — this opcode does not touch a live block-site here; the caller
    ///     falls through to the main match (which handles the real value) and
    ///     resyncs `stack_sites` afterwards;
    ///   - `Some(split)` — handled here; the caller returns `split` directly.
    ///     A `push_closure` (emits no IR) and the temp/dup/pop bookkeeping return
    ///     `Some(None)`; a value-send splices the block body inline and also
    ///     returns `Some(None)` (a send-free leaf block adds no continuation).
    ///
    /// Only reached for a method the escape pre-pass proved `all_elidable`
    /// (`self.escape.is_some()`), so a live block-site here is always an
    /// immediately-invoked, non-escaping literal block. 7-II: a spliced block
    /// with a cold in-body send lowers to a terminating trap — `splice_block`
    /// returns `true`, which sets `*trapped` so `convert` finishes the block.
    #[allow(clippy::too_many_arguments)]
    fn translate_site_instr(
        &mut self,
        instr: &Instr,
        bci: usize,
        stack: &mut Vec<VReg>,
        stack_sites: &mut Vec<Option<usize>>,
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
        trapped: &mut bool,
    ) -> Option<Option<BlockId>> {
        match *instr {
            // A proven-elidable closure: emit NO IR. Push a phantom onto the
            // operand stack (`self_vreg` as a harmless filler — a leaked phantom
            // reads `self`, and `convert`'s block-boundary assert catches any
            // leak in debug) with its site id in the shadow. The real carrier is
            // `stack_sites`; the vreg is never read (its only consumer, the
            // value-send, reads `stack_sites`, not the vreg).
            Instr::PushClosure { .. } => {
                // S24 A3a: an ESCAPING site allocates a real closure — fall
                // through (`None`) to `translate_instr`'s main match, which
                // emits the `Ir::Alloc` + `StoreField` lowering. Only a
                // proven-ELIDABLE site becomes a phantom here.
                if self
                    .escape
                    .as_ref()
                    .is_some_and(|e| e.is_escaping_site(bci))
                {
                    return None;
                }
                stack.push(self.self_vreg);
                stack_sites.push(Some(bci));
                Some(None)
            }
            // A temp currently holding a block-site pushes the phantom; else fall
            // through so the main match emits the real `Move`.
            Instr::PushTemp(t) => {
                if let Some(site) = self.temp_sites[t as usize] {
                    stack.push(self.self_vreg);
                    stack_sites.push(Some(site));
                    Some(None)
                } else {
                    None
                }
            }
            Instr::StoreTempPop(t) => match stack_sites.last().copied().flatten() {
                Some(site) => {
                    // Top is a block-site: record it in the temp, drop the
                    // phantom, emit no `Move`.
                    stack.pop();
                    stack_sites.pop();
                    self.temp_sites[t as usize] = Some(site);
                    Some(None)
                }
                None => {
                    // Storing a real value: clear any stale site the temp held,
                    // then fall through to emit the real `Move`.
                    self.temp_sites[t as usize] = None;
                    None
                }
            },
            Instr::StoreTemp(t) => match stack_sites.last().copied().flatten() {
                // `store_temp` peeks (no pop) — mirror the shadow, leave the site
                // on the stack.
                Some(site) => {
                    self.temp_sites[t as usize] = Some(site);
                    Some(None)
                }
                None => {
                    self.temp_sites[t as usize] = None;
                    None
                }
            },
            Instr::Dup => match stack_sites.last().copied().flatten() {
                Some(site) => {
                    stack.push(self.self_vreg);
                    stack_sites.push(Some(site));
                    Some(None)
                }
                None => None,
            },
            Instr::Pop => match stack_sites.last().copied().flatten() {
                // Discard a dead closure — SPEC A7 "dead" is elidable, no IR.
                Some(_) => {
                    stack.pop();
                    stack_sites.pop();
                    Some(None)
                }
                None => None,
            },
            Instr::Send { ic, .. } => {
                // A `value`-family send the escape pre-pass mapped to a splice.
                // (`value_send_target` returns a `Copy` `usize`, so the immutable
                // borrow of `self.escape` ends before the `&mut self` splice.)
                // S14 step 7-IV-c: a BLOCK-ARG send (the block rides as an
                // ARGUMENT into a transparent, CFG-inlinable mono callee — the
                // `do:` pattern). MUST splice: the phantom has no runtime value,
                // so there is no Call fallback.
                if let Some((site, arg_ix, devirt)) = self
                    .escape
                    .as_ref()
                    .and_then(|e| e.blockarg_send_target(bci))
                {
                    let cont = self.splice_blockarg_send(
                        site,
                        arg_ix,
                        devirt,
                        ic,
                        bci,
                        stack,
                        stack_sites,
                        code,
                        deopt,
                    );
                    return Some(Some(cont));
                }
                let target = self.escape.as_ref().and_then(|e| e.value_send_target(bci));
                if let Some(site) = target {
                    match self.splice_block(site, ic, bci, stack, stack_sites, code, deopt) {
                        SpliceOutcome::Trapped => {
                            *trapped = true;
                            Some(None)
                        }
                        SpliceOutcome::Done => Some(None),
                        // S24 B5: multi-BB graft — continue in its continuation.
                        SpliceOutcome::Continuation(cont) => Some(Some(cont)),
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// S14 step 7-I/7-II: splice a proven-elidable literal block's body inline at
    /// its `value`-family send. Like [`Translator::try_inline_nonleaf`] MINUS the
    /// receiver guard — the receiver is statically the block created at
    /// `site_bci`, so there is no klass to test and no dependency to record. The
    /// block's home `self` is M's receiver (`self_vreg` = the closure's captured
    /// `copied[0]`, SPEC §5.4). Pops the value-send's args + the phantom receiver
    /// (keeping `stack`/`stack_sites` in lockstep).
    ///
    /// **7-II**: the block body MAY contain its own (non-super) sends — each
    /// becomes a plain `Ir::CallSend` (or, for a cold IC, a step-3
    /// `Ir::UncommonTrap`) inside the spliced extent, recording an in-body deopt
    /// scope with an `is_block: true` `InlineSite` chained to M. A deopt there
    /// rebuilds the block's OWN activation frame: it is a method-shaped frame, and
    /// the block frame's only structural difference (the closure in its
    /// receiver-arg slot rather than the home self) is read ONLY by `nlr_tos` /
    /// nested `push_closure`, both still gated out — so the deopt materializer's
    /// method-frame path reconstructs it soundly (`runtime/deopt.rs`).
    ///
    /// Returns `true` iff an in-body send was cold and became a TERMINATING trap
    /// (control leaves the method — the caller must finish the current block,
    /// nothing after is reachable); on a normal return it pushes the block's
    /// result onto `stack` and returns `false`.
    #[allow(clippy::too_many_arguments)]
    fn splice_block(
        &mut self,
        site_bci: usize,
        ic: u16,
        send_bci: usize,
        stack: &mut Vec<VReg>,
        stack_sites: &mut Vec<Option<usize>>,
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
    ) -> SpliceOutcome {
        // Resolve the CompiledBlock from the `push_closure` literal at `site_bci`.
        let (site_instr, _) = decode_at(self.method, site_bci);
        let lit = match site_instr {
            Instr::PushClosure { lit, .. } => lit,
            _ => unreachable!("splice_block: site bci is not a push_closure"),
        };
        let block = MethodOop::try_from(self.method.literals().at(lit as usize))
            .expect("splice_block: push_closure literal is not a CompiledBlock");

        let argc = InterpreterIc::at(self.method, ic).argc() as usize;
        debug_assert_eq!(
            argc,
            block.argc(),
            "splice_block: value-send argc must match block argc (escape guaranteed)"
        );

        // S24 B5 step 2: probe the body's CFG BEFORE touching the sim stack.
        // A multi-BB body (frontend-inlined ifTrue:/and:/or: inside the
        // block — hence CONDITIONAL `^`) grafts through the SAME engine
        // method inlining uses (GraftMode::Block), which mints its own
        // IrBlocks + continuation; the linear walk below keeps the proven
        // single-BB path. This is a COMMITTED site (the phantom receiver has
        // no runtime value), so the graft must succeed — mirrored from
        // splice_blockarg_send's own expect.
        let block_cfg = crate::compiler::decode::decode(block);
        let single_bb = block_cfg.blocks.len() == 1
            && matches!(
                block_cfg.blocks[0].terminator,
                crate::compiler::decode::Terminator::Return
            );
        if !single_bb {
            let args: Vec<VReg> = stack[stack.len() - argc..].to_vec();
            let pre_pop_stack = stack.clone();
            let selector = InterpreterIc::at(self.method, ic).selector();
            let (result, entry_id, continuation_id) = self
                .try_inline_cfg(
                    block,
                    None, // no guard: the receiver is statically the literal block
                    self.rcvr_klass, // unused in Block mode (no inline dep)
                    selector,
                    self.self_vreg, // the block's home self
                    &args,
                    send_bci,
                    &pre_pop_stack,
                    code,
                    deopt,
                    None,
                    GraftMode::Block { parent: None },
                )
                .expect(
                    "splice_block: escape proved the block CFG-graftable but the graft                      declined (compiler bug)",
                );
            for _ in 0..(argc + 1) {
                stack_sites.pop();
                stack.pop().expect("value-send: missing operand");
            }
            stack.push(result);
            stack_sites.push(None);
            debug_assert!(
                self.pending_jump_target.is_none(),
                "block graft: a pending jump target is already armed"
            );
            self.pending_jump_target = Some(entry_id);
            return SpliceOutcome::Continuation(continuation_id);
        }

        // Pop the value-send's args (real vregs) + the phantom receiver, keeping
        // `stack`/`stack_sites` in lockstep. An arg is never a block-site (an
        // arg-site would have escaped), so its shadow entry is `None`.
        let mut blk_args: Vec<VReg> = (0..argc)
            .map(|_| {
                stack_sites.pop();
                stack.pop().expect("value-send: missing arg operand")
            })
            .collect();
        blk_args.reverse();
        stack_sites.pop(); // the phantom receiver's site marker
        stack
            .pop()
            .expect("value-send: missing receiver (phantom closure)");

        // The block's home self = M's receiver (only read by push_self / instvar
        // access inside the block body).
        let block_self = self.self_vreg;

        // Arity re-check (the shape was routed above; the arity is
        // site-specific). Operands are already popped — no clean recovery.
        assert!(
            block.argc() == blk_args.len(),
            "splice_block: escape proved spliceable but arity re-check failed"
        );

        // Callee slots: args alias the value-send operands (no stores); temps
        // become fresh nil vregs with canonical spill homes.
        let mut slots: Vec<VReg> = Vec::with_capacity(block.argc() + block.ntemps());
        slots.extend_from_slice(&blk_args);
        let nil_lit = self
            .pool
            .intern(self.vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
        for _ in 0..block.ntemps() {
            let t = self.fresh(true);
            code.push(Ir::ConstPool {
                dst: t,
                lit: nil_lit,
            });
            slots.push(t);
        }

        // 7-II: the InlineSite prototype for every in-body deopt safepoint. The
        // reconstructed inlined scope is an `is_block` scope whose sender is M
        // (this method): `sender_bci` is the value-send's bci and
        // `caller_pending_stack` is M's operand stack BELOW the value-send's
        // receiver+args (which we just popped, so `stack` IS that now). The
        // scope's receiver is the block's home self (M's receiver) and its slots
        // are the block's args+temps.
        let block_pool_ix = self.pool.intern(block.oop().raw(), Some(RelocKind::Oop)).0;
        let inline_proto = InlineSite {
            method_pool_ix: block_pool_ix,
            receiver: block_self,
            slots: slots.clone(),
            sender_bci: send_bci as u16,
            caller_pending_stack: stack.clone(),
            is_block: true,
            parent: None,
            slot_closures: Vec::new(),
        };

        match self.translate_spliced_block_body(
            block,
            &slots,
            block_self,
            &inline_proto,
            code,
            deopt,
        ) {
            Err(()) => SpliceOutcome::Trapped,
            Ok(result) => {
                stack.push(result);
                stack_sites.push(None);
                SpliceOutcome::Done
            }
        }
    }

    /// S14 step 7-IV-c: the SHARED spliced-block body translator — used by
    /// [`Translator::splice_block`] (a block value'd directly in M) AND by
    /// [`Translator::try_inline_cfg`]'s block-arg path (a block spliced at a
    /// `value` site INSIDE an inlined callee, with a chained `InlineSite`).
    /// Translates the single straight-line body onto a fresh operand stack:
    /// value-shuffle/field arms, ctx-temp arms (→ the home's promoted
    /// `ctx_vregs`), the block's own sends (cold → terminating trap, warm →
    /// `CallSend`, both recording `inline_proto`), `nlr_tos` → `Ir::Ret` (a
    /// return from the whole compilation — the block's home is the root).
    /// Returns `Ok(result_vreg)` on a normal block return, `Err(())` when the
    /// body ended in a TERMINATING op (trap or NLR) — the caller must stop
    /// translating its current block.
    fn translate_spliced_block_body(
        &mut self,
        block: MethodOop,
        slots: &[VReg],
        block_self: VReg,
        inline_proto: &InlineSite,
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
    ) -> Result<VReg, ()> {
        // S24 B5 step 4: a committed splice — count it toward the cumulative
        // budget up front (never declined here; amendment A1).
        self.budget_commit(block);
        // S14 step 5: a spliced block's home self is the ROOT method's own
        // `self` whenever `block_self == self_vreg` (the only shape today),
        // so its klass is statically the customization klass.
        let block_self_klass: Option<KlassOop> = if block_self == self.self_vreg {
            Some(self.rcvr_klass)
        } else {
            None
        };
        let mut bstack: Vec<VReg> = Vec::new();
        let mut result: Option<VReg> = None;
        let mut trapped = false;
        let len = block.bytecode_len();
        let mut bci = 0;
        while bci < len {
            let (instr, next) = decode_at(block, bci);
            match instr {
                Instr::PushSelf => bstack.push(block_self),
                Instr::PushNil => {
                    bstack.push(self.push_well_known(self.vm.universe.nil_obj.raw(), code))
                }
                Instr::PushTrue => {
                    bstack.push(self.push_well_known(self.vm.universe.true_obj.raw(), code))
                }
                Instr::PushFalse => {
                    bstack.push(self.push_well_known(self.vm.universe.false_obj.raw(), code))
                }
                Instr::PushSmi(v) => {
                    let dst = self.fresh(true);
                    code.push(Ir::ConstSmi {
                        dst,
                        value: v as i64,
                    });
                    bstack.push(dst);
                }
                Instr::PushLiteral(idx) => {
                    let lit_oop = block.literals().at(idx as usize);
                    bstack.push(self.push_well_known(lit_oop.raw(), code));
                }
                Instr::PushGlobal(idx) => {
                    let assoc = block.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                    });
                    bstack.push(dst);
                }
                Instr::PushTemp(n) => {
                    let dst = self.fresh(true);
                    code.push(Ir::Move {
                        dst,
                        src: slots[n as usize],
                    });
                    bstack.push(dst);
                }
                Instr::StoreTemp(n) => {
                    let tos = *bstack
                        .last()
                        .expect("block splice: store_temp on empty stack");
                    code.push(Ir::Move {
                        dst: slots[n as usize],
                        src: tos,
                    });
                }
                Instr::StoreTempPop(n) => {
                    let v = bstack
                        .pop()
                        .expect("block splice: store_temp_pop on empty stack");
                    code.push(Ir::Move {
                        dst: slots[n as usize],
                        src: v,
                    });
                }
                Instr::PushInstvar(n) => {
                    let dst = self.fresh(true);
                    code.push(Ir::LoadField {
                        dst,
                        obj: block_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                    });
                    bstack.push(dst);
                }
                Instr::StoreInstvarPop(n) => {
                    let val = bstack
                        .pop()
                        .expect("block splice: store_instvar_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: block_self,
                        byte_off: (BODY_OFFSET + 8 * n as usize) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::StoreGlobalPop(idx) => {
                    let assoc = block.literals().at(idx as usize);
                    let assoc_vreg = self.push_well_known(assoc.raw(), code);
                    let val = bstack
                        .pop()
                        .expect("block splice: store_global_pop on empty stack");
                    code.push(Ir::StoreField {
                        obj: assoc_vreg,
                        byte_off: (BODY_OFFSET + 8) as i32,
                        val,
                        barrier: true,
                    });
                }
                Instr::Pop => {
                    bstack.pop().expect("block splice: pop on empty stack");
                }
                Instr::Dup => {
                    let tos = *bstack.last().expect("block splice: dup on empty stack");
                    bstack.push(tos);
                }
                Instr::ReturnTos | Instr::BlockReturnTos => {
                    result = Some(bstack.pop().expect("block splice: return on empty stack"));
                    break;
                }
                Instr::ReturnSelf => {
                    result = Some(block_self);
                    break;
                }
                // 7-III / S24 B4: `^expr` (non-local return). The block's home
                // is M (the block is created + value'd in M), so the NLR is
                // just a RETURN FROM M — `Ir::Ret` of the value. Control
                // leaves the method, so (like a trap) the caller must finish
                // the current block. Since B4 the block may ALSO contain
                // sends: an in-block deopt at one rebuilds an is_block frame
                // whose receiver-arg slot the materializer fills with a
                // SYNTHESIZED home-ref closure (deopt.rs M6), so the
                // interpreter's `nlr_tos` works unmodified. Markers can't
                // intervene: ensure:/ifCurtailed: plant markers only on the
                // protected block's own fresh interpreter activation, are
                // primitive methods excluded from inlining AND block-arg
                // splicing, and a protected block lexically containing an
                // NLR block fails block_transitively_nlr_free at eligibility.
                Instr::NlrTos => {
                    let val = bstack.pop().expect("block splice: nlr_tos on empty stack");
                    self.spliced_nlr += 1;
                    code.push(Ir::Ret { val });
                    trapped = true;
                    break;
                }
                // 7-II: the block's OWN (non-super) sends. A cold IC (Untaken)
                // lowers to a TERMINATING uncommon trap inside the inlined block
                // (re-executing the send interpreted); a warm IC stays a plain
                // compiled `CallSend`. Both record an `is_block` in-body deopt
                // scope. Mirrors `try_inline_nonleaf`'s Send arm exactly, minus
                // recursion (depth-1 into the block: a nested send stays a Call).
                Instr::Send {
                    ic: inner_ic_idx,
                    super_,
                } => {
                    debug_assert!(
                        !super_,
                        "block splice: super send gated out by block_is_spliceable"
                    );
                    let inner_ic = InterpreterIc::at(block, inner_ic_idx);
                    let inner_argc = inner_ic.argc();
                    let inner_sel = inner_ic.selector();
                    let inner_bci = bci;
                    // S14 opt: fuse the block body's own mono-smi ops (see
                    // the nonleaf splicer's identical arm) — THE flagship
                    // accumulate block `[:e| sum := sum + e]` becomes a real
                    // `adds` instead of a per-iteration stub call.
                    if let Some(is_put) = array_op_kind_on(self.vm, block, inner_ic_idx) {
                        // S15: in-body array intrinsic (see the root arm).
                        let val = if is_put {
                            Some(bstack.pop().expect("array fuse: missing value"))
                        } else {
                            None
                        };
                        let idx_op = bstack.pop().expect("array fuse: missing index");
                        let arr_op = bstack.pop().expect("array fuse: missing receiver");
                        let mut reexec = bstack.clone();
                        reexec.push(arr_op);
                        reexec.push(idx_op);
                        if let Some(v) = val {
                            reexec.push(v);
                        }
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let klass = self.pool.intern(
                            self.vm.universe.array_klass.oop().raw(),
                            Some(RelocKind::Oop),
                        );
                        let dst = self.fresh(true);
                        match val {
                            Some(v) => code.push(Ir::ArrayAtPut {
                                dst,
                                arr: arr_op,
                                idx: idx_op,
                                val: v,
                                klass,
                                fail,
                            }),
                            None => code.push(Ir::ArrayAt {
                                dst,
                                arr: arr_op,
                                idx: idx_op,
                                klass,
                                fail,
                            }),
                        }
                        bstack.push(dst);
                        bci = next;
                        continue;
                    }
                    if is_smi_inlinable_on(self.vm, block, inner_ic_idx)
                        && !smi_fuse_arg_is_pooled(bstack.as_slice(), code.as_slice())
                    {
                        debug_assert_eq!(inner_argc, 1, "SMI_INLINE ops are all binary");
                        let b_op = bstack.pop().expect("smi fuse: missing rhs");
                        let a_op = bstack.pop().expect("smi fuse: missing lhs");
                        let mut reexec = bstack.clone();
                        reexec.push(a_op);
                        reexec.push(b_op);
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let dst = self.fresh(true);
                        match classify_smi_send(self.vm, block, inner_ic_idx) {
                            SmiSendKind::Arith(op) => code.push(Ir::SmiArith {
                                op,
                                dst,
                                a: a_op,
                                b: b_op,
                                fail,
                            }),
                            SmiSendKind::Cmp(op) => code.push(Ir::SmiCmpVal {
                                op,
                                dst,
                                a: a_op,
                                b: b_op,
                                fail,
                            }),
                        }
                        bstack.push(dst);
                        bci = next;
                        continue;
                    }
                    // Float fast-path in the splice: `is_double_inlinable`'s
                    // `_on` twin, same pattern as the smi twin above.
                    if is_double_inlinable_on(self.vm, block, inner_ic_idx) {
                        debug_assert_eq!(inner_argc, 1, "DOUBLE_INLINE ops are all binary");
                        // b4d55a8's storm guard, ported: a compile-time smi
                        // arg can never FUnbox — emit the coerced FConst.
                        let arg_const_smi: Option<i64> = match (bstack.last(), code.last()) {
                            (Some(&top), Some(Ir::ConstSmi { dst, value })) if *dst == top => {
                                Some(*value)
                            }
                            _ => None,
                        };
                        let b_op = bstack.pop().expect("double fuse: missing rhs");
                        let a_op = bstack.pop().expect("double fuse: missing lhs");
                        let mut reexec = bstack.clone();
                        reexec.push(a_op);
                        reexec.push(b_op);
                        let fail =
                            self.fresh_inlined_trap_block(inner_bci, reexec, inline_proto.clone());
                        let ua = self.fresh_fp();
                        let ub = self.fresh_fp();
                        code.push(Ir::FUnbox {
                            dst: ua,
                            src: a_op,
                            fail,
                        });
                        match arg_const_smi {
                            Some(v) => code.push(Ir::FConst {
                                dst: ub,
                                bits: (v as f64).to_bits(),
                            }),
                            None => code.push(Ir::FUnbox {
                                dst: ub,
                                src: b_op,
                                fail,
                            }),
                        }
                        let dst = self.fresh(true);
                        match classify_double_send(self.vm, block, inner_ic_idx) {
                            DoubleSendKind::Arith(op) => {
                                let fd = self.fresh_fp();
                                code.push(Ir::FArith {
                                    op,
                                    dst: fd,
                                    a: ua,
                                    b: ub,
                                });
                                code.push(Ir::FBox { dst, src: fd });
                            }
                            DoubleSendKind::Cmp(op) => {
                                code.push(Ir::FCmpVal {
                                    op,
                                    dst,
                                    a: ua,
                                    b: ub,
                                });
                            }
                        }
                        bstack.push(dst);
                        bci = next;
                        continue;
                    }
                    let inner_fb = crate::compiler::feedback::read_send_site(
                        self.vm,
                        block,
                        inner_ic_idx,
                        None,
                    );
                    if self.osr
                        && matches!(
                            crate::compiler::inline::decide(&inner_fb),
                            crate::compiler::inline::InlineDecision::Trap
                        )
                    {
                        self.osr_cold_sends += 1; // OSR-heal census (see root arm)
                    }

                    let mut inner_args: Vec<VReg> = (0..inner_argc)
                        .map(|_| {
                            bstack
                                .pop()
                                .expect("block splice: inner send missing arg operand")
                        })
                        .collect();
                    inner_args.reverse();
                    let inner_recv = bstack
                        .pop()
                        .expect("block splice: inner send missing receiver operand");

                    // S14 step 5: an inner SELF-send (receiver == the spliced
                    // callee's own self) with a statically-known receiver klass
                    // never traps cold — its target is fixed at compile time,
                    // so an Empty IC just compiles as the plain lazy CallSend
                    // below (no interpreted warm-up, no in-body deopt).
                    let inner_static_self = inner_recv == block_self
                        && block_self_klass
                            .is_some_and(|k| self.devirt_self_target(k, inner_sel).is_some());
                    if !inner_static_self && self.inlined_cold_send_traps() {
                        if let crate::compiler::inline::InlineDecision::Trap =
                            crate::compiler::inline::decide(&inner_fb)
                        {
                            // Cold inner send → terminating trap re-executing it
                            // interpreted. Recorded stack has the receiver+args still
                            // present (reexecute=true), in the BLOCK's own scope.
                            let mut reexec_stack = bstack.clone();
                            reexec_stack.push(inner_recv);
                            reexec_stack.extend_from_slice(&inner_args);
                            deopt.push((
                                code.len() as u32,
                                DeoptRaw {
                                    stack: reexec_stack,
                                    bci: inner_bci,
                                    kind: SafepointKind::UncommonTrap,
                                    reexecute: true,
                                    stack_closures: Vec::new(),
                                    inline: Some(inline_proto.clone()),
                                },
                            ));
                            code.push(Ir::UncommonTrap { bci: inner_bci });
                            trapped = true;
                            break;
                        }
                    }

                    let mut send_args = vec![inner_recv];
                    send_args.extend_from_slice(&inner_args);
                    let site = self.call_sites.len() as u16;
                    self.call_sites.push(CallSiteInfo {
                        selector: inner_sel,
                        argc: inner_argc + 1,
                        static_klass: None,
                    });
                    self.site_feedback.push(inner_fb);
                    let dst = self.fresh(true);
                    // Call-return safepoint (reexecute=false): recorded stack is
                    // the block's operand stack BELOW the send; the materializer
                    // pushes the result.
                    deopt.push((
                        code.len() as u32,
                        DeoptRaw {
                            stack: bstack.clone(),
                            bci: inner_bci,
                            kind: SafepointKind::Call,
                            reexecute: false,
                            stack_closures: Vec::new(),
                            inline: Some(inline_proto.clone()),
                        },
                    ));
                    code.push(Ir::CallSend {
                        dst,
                        site,
                        args: send_args,
                    });
                    bstack.push(dst);
                }
                // S14 step 7-II-b: the block reads/writes its HOME method's
                // captured temps. A ctx-less (`captures_ctx`) block's frame
                // aliases M's Context, so it addresses M's ctx-temps at depth 0 —
                // the SAME promoted `ctx_vregs` M itself uses. `depth != 0` was
                // rejected by `block_is_spliceable`. S24 A3b: if M MATERIALIZES
                // its Context (an escaping capturing block forced a real
                // object), the spliced block must access that SAME object, not
                // the elided vregs — else the inlined and escaped views split
                // (Risk 2).
                Instr::PushCtxTemp { depth, idx } => {
                    debug_assert_eq!(depth, 0, "block ctx-temp access must be depth 0");
                    let dst = self.fresh(true);
                    if let Some(ctx) = self.method_ctx_vreg {
                        code.push(Ir::LoadField {
                            dst,
                            obj: ctx,
                            byte_off: (crate::oops::layout::BODY_OFFSET + 8 * (2 + idx as usize))
                                as i32,
                        });
                    } else {
                        code.push(Ir::Move {
                            dst,
                            src: self.ctx_vregs[idx as usize],
                        });
                    }
                    bstack.push(dst);
                }
                Instr::StoreCtxTempPop { depth, idx } => {
                    debug_assert_eq!(depth, 0, "block ctx-temp access must be depth 0");
                    let v = bstack
                        .pop()
                        .expect("block splice: store_ctx_temp_pop on empty stack");
                    if let Some(ctx) = self.method_ctx_vreg {
                        code.push(Ir::StoreField {
                            obj: ctx,
                            byte_off: (crate::oops::layout::BODY_OFFSET + 8 * (2 + idx as usize))
                                as i32,
                            val: v,
                            barrier: true,
                        });
                    } else {
                        code.push(Ir::Move {
                            dst: self.ctx_vregs[idx as usize],
                            src: v,
                        });
                    }
                }
                // `block_is_spliceable` still rejects super/closure/NLR and
                // depth>0 ctx access, and the single-block shape excludes
                // jumps/branches; every other opcode is handled above.
                other => unreachable!(
                    "block splice: {other:?} passed block_is_spliceable but has no arm"
                ),
            }
            bci = next;
        }

        if trapped {
            return Err(());
        }
        Ok(result.expect("block splice: a returning block must produce a result"))
    }

    /// S14 step 7-IV-c: lower a BLOCK-ARG send — `recv D-selector: ... B ...`
    /// where `B` is a proven-elidable literal block and the mono callee D is
    /// block-arg transparent. Splices D via [`Translator::try_inline_cfg`]
    /// with the phantom threaded through (D's `value` sites splice B's body,
    /// depth-3 InlineSite chains, `ElidedClosure` recorded for the phantom's
    /// slot/stack positions). There is NO fallback: the escape pre-pass
    /// committed to this shape, and the phantom has no runtime value to pass
    /// to a real send. Pops the send's operands, pushes the result, arms
    /// `pending_jump_target` with D's entry block, and returns the
    /// continuation the caller must adopt as its `split`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn splice_blockarg_send(
        &mut self,
        site_bci: usize,
        arg_ix: usize,
        devirt: bool,
        ic: u16,
        send_bci: usize,
        stack: &mut Vec<VReg>,
        stack_sites: &mut Vec<Option<usize>>,
        code: &mut Vec<Ir>,
        deopt: &mut Vec<(u32, DeoptRaw)>,
    ) -> BlockId {
        // Resolve B from its push_closure literal; D per the RESOLUTION MODE
        // the escape pre-pass recorded (S24 B3) — one decision, made there,
        // obeyed here, so the two passes can never resolve different callees.
        let (site_instr, _) = decode_at(self.method, site_bci);
        let lit = match site_instr {
            Instr::PushClosure { lit, .. } => lit,
            _ => unreachable!("splice_blockarg_send: site bci is not a push_closure"),
        };
        let blk = MethodOop::try_from(self.method.literals().at(lit as usize))
            .expect("splice_blockarg_send: push_closure literal is not a CompiledBlock");
        let blk_pool_ix = self.pool.intern(blk.oop().raw(), Some(RelocKind::Oop)).0;

        let selector = InterpreterIc::at(self.method, ic).selector();
        let argc = InterpreterIc::at(self.method, ic).argc() as usize;

        // S24 B3: a DEVIRTUALIZED site (self receiver, callee resolved per
        // the customization klass) splices GUARD-FREE — the nmethod's entry
        // guard already proved every receiver has exactly rcvr_klass, the
        // same convention as the S14 step-5 self-send devirt. dep_klass =
        // rcvr_klass keeps the lookup-shaped inline dep sound (redefining or
        // shadowing rcvr_klass>>selector invalidates this nmethod). A
        // guarded site resolves D + guard klass from the send's own (mono)
        // feedback — the SAME IC state the escape pre-pass read moments ago
        // in this no-GC compile window.
        let (guard, dep_klass, callee) = if devirt {
            let callee =
                crate::compiler::feedback::resolve_method_ro(self.vm, self.rcvr_klass, selector)
                    .expect(
                        "splice_blockarg_send: escape devirt-resolved this site against the \
                         same rcvr_klass moments ago in this no-GC compile window (compiler \
                         bug)",
                    );
            (None, self.rcvr_klass, callee)
        } else {
            let feedback =
                crate::compiler::feedback::read_send_site(self.vm, self.method, ic, None);
            let (guard_klass, callee) = match &feedback {
                crate::compiler::feedback::SiteFeedback::Mono { klass, method } => {
                    (*klass, *method)
                }
                other => unreachable!(
                    "splice_blockarg_send: escape proved a mono block-arg site but feedback \
                     is {other:?} (compiler bug — IC state cannot change mid-compile)"
                ),
            };
            let guard_shape = if guard_klass.oop().raw() == self.vm.universe.smi_klass.oop().raw() {
                GuardShape::SmiTest
            } else {
                GuardShape::KlassTest
            };
            (Some((guard_klass, guard_shape)), guard_klass, callee)
        };

        // Pop operands (keeping the phantom shadow in lockstep). The phantom
        // arg's vreg is a filler; its POSITION is what matters (arg_ix).
        let pre_pop_stack = stack.clone();
        let mut args: Vec<VReg> = (0..argc)
            .map(|_| {
                stack_sites.pop();
                stack.pop().expect("blockarg send: missing arg operand")
            })
            .collect();
        args.reverse();
        stack_sites.pop();
        let receiver = stack
            .pop()
            .expect("blockarg send: missing receiver operand");

        // Tripwire for the one residual divergence class: a devirt site's
        // receiver must BE the self vreg (escape's narrow AV::Slf tracks
        // exactly direct push_self/dup flow, which convert aliases as
        // self_vreg by construction).
        debug_assert!(
            !devirt || receiver == self.self_vreg,
            "devirt blockarg site's receiver vreg is not self_vreg (escape/convert \
             self-provenance divergence)"
        );
        let (result, entry_id, continuation_id) = self
            .try_inline_cfg(
                callee,
                guard,
                dep_klass,
                selector,
                receiver,
                &args,
                send_bci,
                &pre_pop_stack,
                code,
                deopt,
                Some((arg_ix, blk, blk_pool_ix)),
                GraftMode::Method,
            )
            .expect(
                "splice_blockarg_send: escape proved the callee CFG-inlinable but the \
                 splice declined (compiler bug)",
            );
        stack.push(result);
        stack_sites.push(None);
        debug_assert!(
            self.pending_jump_target.is_none(),
            "blockarg splice: a pending jump target is already armed"
        );
        self.pending_jump_target = Some(entry_id);
        continuation_id
    }

    fn push_well_known(&mut self, raw: u64, code: &mut Vec<Ir>) -> VReg {
        let lit = self.pool.intern(raw, Some(RelocKind::Oop));
        let dst = self.fresh(true);
        code.push(Ir::ConstPool { dst, lit });
        dst
    }
}

/// S14 step 4b: does `callee`'s body consist solely of opcodes
/// [`Translator::try_inline_leaf`] can splice? A dry-run over the bytecode used
/// to decide inlining BEFORE emitting the guard, so a mid-splice bailout (which
/// would leave a guard dominating a half-built body) is impossible. The set
/// mirrors `try_inline_leaf`'s own match arms exactly; a `Send` never appears
/// (the caller only reaches here for a leaf, `inline::is_leaf`).
fn leaf_body_is_spliceable(callee: MethodOop) -> bool {
    let len = callee.bytecode_len();
    let mut bci = 0;
    while bci < len {
        let (instr, next) = decode_at(callee, bci);
        let ok = matches!(
            instr,
            Instr::PushSelf
                | Instr::PushNil
                | Instr::PushTrue
                | Instr::PushFalse
                | Instr::PushSmi(_)
                | Instr::PushLiteral(_)
                | Instr::PushGlobal(_)
                | Instr::PushTemp(_)
                | Instr::StoreTemp(_)
                | Instr::StoreTempPop(_)
                | Instr::PushInstvar(_)
                | Instr::StoreInstvarPop(_)
                | Instr::Pop
                | Instr::Dup
                | Instr::ReturnTos
                | Instr::ReturnSelf
        );
        if !ok {
            return false;
        }
        bci = next;
    }
    true
}

fn emit_merges(
    sources: &[EntryStackSource],
    entry_stacks: &[Option<Vec<VReg>>],
    target: BlockIndex,
    local_exit: &[VReg],
    code: &mut Vec<Ir>,
) {
    if let EntryStackSource::Merge = sources[target] {
        let merge_vregs = entry_stacks[target]
            .as_ref()
            .expect("emit_merges: merge vregs are pre-allocated before any block is translated");
        assert_eq!(
            merge_vregs.len(),
            local_exit.len(),
            "emit_merges: depth mismatch feeding block {target} (compiler bug)"
        );
        for (&dst, &src) in merge_vregs.iter().zip(local_exit.iter()) {
            code.push(Ir::Move { dst, src });
        }
    }
}

/// D3.2/D3.3: convert `method`'s bytecode (already decoded into `cfg`) into
/// an [`IrMethod`]. `vm` supplies well-known oops (`nil`/`true`/`false`)
/// and the smi klass every inlined send's IC is checked against.
/// S14 opt: rewrite `op`'s vreg USES through `f` (the mutable twin of
/// [`Ir::uses`] — kept in exact lockstep with it).
fn map_uses(op: &mut Ir, mut f: impl FnMut(VReg) -> VReg) {
    match op {
        Ir::Move { src, .. } => *src = f(*src),
        Ir::LoadKlass { obj, .. } => *obj = f(*obj),
        Ir::LoadField { obj, .. } => *obj = f(*obj),
        Ir::StoreField { obj, val, .. } => {
            *obj = f(*obj);
            *val = f(*val);
        }
        Ir::SmiArith { a, b, .. }
        | Ir::SmiCmpBr { a, b, .. }
        | Ir::SmiCmpVal { a, b, .. }
        | Ir::RefCmpVal { a, b, .. } => {
            *a = f(*a);
            *b = f(*b);
        }
        Ir::BoolNot { src, .. } => *src = f(*src),
        Ir::FArith { a, b, .. } | Ir::FCmpBr { a, b, .. } | Ir::FCmpVal { a, b, .. } => {
            *a = f(*a);
            *b = f(*b);
        }
        Ir::FUnbox { src, .. } | Ir::FBox { src, .. } => *src = f(*src),
        Ir::VecArith { a, b, .. } => {
            *a = f(*a);
            *b = f(*b);
        }
        Ir::ArrayAt { arr, idx, .. } => {
            *arr = f(*arr);
            *idx = f(*idx);
        }
        Ir::ArrayAtPut { arr, idx, val, .. } => {
            *arr = f(*arr);
            *idx = f(*idx);
            *val = f(*val);
        }
        Ir::BoolBr { val, .. } => *val = f(*val),
        Ir::GuardKlass { obj, .. } => *obj = f(*obj),
        Ir::CallSend { args, .. } | Ir::CallRuntime { args, .. } => {
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Ir::Ret { val } => *val = f(*val),
        Ir::NlrReturn { closure, value } => {
            *closure = f(*closure);
            *value = f(*value);
        }
        // RetSelf's implicit VReg(0) use is never a copy-prop target (VReg(0)
        // is the multi-use self param, never a single-def Move dst).
        Ir::RetSelf
        | Ir::ConstSmi { .. }
        | Ir::ConstPool { .. }
        | Ir::FConst { .. }
        | Ir::Param { .. }
        | Ir::Jump { .. }
        | Ir::Alloc { .. }
        | Ir::Poll
        | Ir::UncommonTrap { .. }
        | Ir::Bailout { .. } => {}
    }
}

/// S14 opt: PER-BLOCK COPY PROPAGATION — the pass that makes compiled loops
/// worth having. `convert`'s D3.2 "always a fresh copy" rule manufactures a
/// `Move {fresh, src}` for every temp push (soundness against later temp
/// redefinition), and under S12 spill-all every one of those single-use
/// copies becomes an `ldr`+`str` round-trip through a fresh frame slot — the
/// measured reason a fully-compiled, fully-fused `sumTo:` loop ran at
/// INTERPRETER speed.
///
/// The pass, per block, scans forward keeping `alias: dst → src` for each
/// `Move` whose `dst` has exactly ONE def in the whole method (fresh copies;
/// never merge vregs or temp slots — both multi-def). Subsequent uses (in ops
/// AND in deopt-site metadata) rewrite through the alias; an alias dies the
/// moment its `src` (or `dst`) is redefined — exactly the redefinition hazard
/// the fresh-copy rule guards against, now checked instead of paid for. A
/// `Move` whose every use got rewritten (global use count == in-window
/// rewrites) is deleted, with deopt-site indices re-keyed.
///
/// Sound with spill-all/deopt: a rewritten reference reads `src`'s canonical
/// slot; `src` is not redefined between the (deleted) copy and the reference
/// (the kill rule), so the slot holds the identical value at every
/// safepoint in between. Cross-block uses of an alias are left in place (the
/// window is per-block), which simply keeps that `Move` alive — conservative,
/// never wrong.
pub(crate) fn copy_propagate(m: &mut IrMethod) {
    let n = m.vregs.len();
    let mut def_count = vec![0u32; n];
    let mut use_count = vec![0u32; n];
    for b in &m.blocks {
        for op in &b.code {
            op.defs(|v| def_count[v.0 as usize] += 1);
            op.uses(|v| use_count[v.0 as usize] += 1);
        }
        for (_, raw) in &b.deopt_sites {
            for v in &raw.stack {
                use_count[v.0 as usize] += 1;
            }
            let mut lvl = raw.inline.as_ref();
            while let Some(site) = lvl {
                use_count[site.receiver.0 as usize] += 1;
                for v in &site.slots {
                    use_count[v.0 as usize] += 1;
                }
                for v in &site.caller_pending_stack {
                    use_count[v.0 as usize] += 1;
                }
                lvl = site.parent.as_deref();
            }
        }
        // A merge target's entry stack names its vregs implicitly — pin them.
        for v in &b.entry_stack {
            use_count[v.0 as usize] += 1;
        }
    }

    fn resolve(alias: &HashMap<u32, VReg>, v: VReg) -> VReg {
        let mut cur = v;
        let mut hops = 0;
        while let Some(&next) = alias.get(&cur.0) {
            cur = next;
            hops += 1;
            debug_assert!(hops < 64, "copy_propagate: alias cycle");
        }
        cur
    }
    fn rewrite_raw(raw: &mut DeoptRaw, alias: &HashMap<u32, VReg>, rewrites: &mut [u32]) {
        for v in raw.stack.iter_mut() {
            let r = resolve(alias, *v);
            if r != *v {
                rewrites[v.0 as usize] += 1;
                *v = r;
            }
        }
        let mut lvl = raw.inline.as_mut();
        while let Some(site) = lvl {
            let r = resolve(alias, site.receiver);
            if r != site.receiver {
                rewrites[site.receiver.0 as usize] += 1;
                site.receiver = r;
            }
            for v in site.slots.iter_mut() {
                let r = resolve(alias, *v);
                if r != *v {
                    rewrites[v.0 as usize] += 1;
                    *v = r;
                }
            }
            for v in site.caller_pending_stack.iter_mut() {
                let r = resolve(alias, *v);
                if r != *v {
                    rewrites[v.0 as usize] += 1;
                    *v = r;
                }
            }
            lvl = site.parent.as_mut().map(|p| p.as_mut());
        }
    }

    // ── Phase A: per-block forward windows. Rewrite each block's own ops +
    //    deopt metadata; RECORD the alias state at every fail-edge op so the
    //    fail BLOCK's reexecute metadata (captured at exactly that point, but
    //    living in its own IrBlock) can be rewritten identically in phase B —
    //    without this, every fused op's operands stay pinned by their own
    //    fail-block references and nothing ever deletes. ──────────────────
    let mut rewrites = vec![0u32; n];
    let mut fail_rewrites: Vec<(BlockId, HashMap<u32, VReg>)> = Vec::new();
    for b in &mut m.blocks {
        let mut alias: HashMap<u32, VReg> = HashMap::new();
        let mut deopt_iter = b.deopt_sites.iter_mut().peekable();
        for (i, op) in b.code.iter_mut().enumerate() {
            while let Some((ci, _)) = deopt_iter.peek() {
                if *ci != i as u32 {
                    break;
                }
                let (_, raw) = deopt_iter.next().unwrap();
                rewrite_raw(raw, &alias, &mut rewrites);
            }
            if !alias.is_empty() {
                match op {
                    Ir::ArrayAt { fail, .. }
                    | Ir::ArrayAtPut { fail, .. }
                    | Ir::SmiArith { fail, .. }
                    | Ir::SmiCmpVal { fail, .. }
                    | Ir::SmiCmpBr { fail, .. }
                    | Ir::BoolNot { fail, .. }
                    | Ir::GuardKlass { fail, .. } => {
                        fail_rewrites.push((*fail, alias.clone()));
                    }
                    Ir::BoolBr { not_bool, .. } => {
                        fail_rewrites.push((*not_bool, alias.clone()));
                    }
                    _ => {}
                }
            }
            map_uses(op, |v| {
                let r = resolve(&alias, v);
                if r != v {
                    rewrites[v.0 as usize] += 1;
                }
                r
            });
            let mut defined: Option<VReg> = None;
            op.defs(|v| defined = Some(v));
            if let Some(d) = defined {
                alias.retain(|&k, &mut s| k != d.0 && s != d);
            }
            if let Ir::Move { dst, src } = *op {
                if def_count[dst.0 as usize] == 1 {
                    let r = resolve(&alias, src);
                    alias.insert(dst.0, r);
                }
            }
        }
    }

    // ── Phase B: fail-edge blocks' metadata, with the recorded alias state
    //    of the op that owns the edge (their reexecute stacks were captured
    //    at exactly that point). ────────────────────────────────────────────
    for (fail_id, alias) in fail_rewrites {
        if let Some(fb) = m.blocks.iter_mut().find(|b| b.id == fail_id) {
            for (_, raw) in fb.deopt_sites.iter_mut() {
                rewrite_raw(raw, &alias, &mut rewrites);
            }
        }
    }

    // ── Phase C: delete fully-propagated copies, re-keying deopt indices. ──
    for b in &mut m.blocks {
        let mut removed_before = vec![0u32; b.code.len() + 1];
        let mut removed = 0u32;
        let mut keep: Vec<bool> = Vec::with_capacity(b.code.len());
        for (i, op) in b.code.iter().enumerate() {
            removed_before[i] = removed;
            let deletable = match op {
                Ir::Move { dst, .. } => {
                    def_count[dst.0 as usize] == 1
                        && use_count[dst.0 as usize] > 0
                        && rewrites[dst.0 as usize] == use_count[dst.0 as usize]
                }
                _ => false,
            };
            keep.push(!deletable);
            if deletable {
                removed += 1;
            }
        }
        removed_before[b.code.len()] = removed;
        if removed == 0 {
            continue;
        }
        let mut it = keep.iter();
        b.code.retain(|_| *it.next().unwrap());
        for (ci, _) in b.deopt_sites.iter_mut() {
            *ci -= removed_before[*ci as usize];
        }
    }
}

/// `MACVM_SPLICE_PROMOTE=1` selects [`promote_float_temps_spliced`] (the
/// dormant generalized promotion) over the default root-temp-only pass.
/// Read once per process — a compile-path check, but cached anyway per the
/// standing env-var-cost lesson.
fn splice_promote_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MACVM_SPLICE_PROMOTE").map(|v| v == "1").unwrap_or(false)
    })
}

/// Float fast-path rule 4 (`docs/float_fastpath_design.md` B5): FLOAT-TEMP
/// PROMOTION — a method temp that provably always holds a Double becomes an
/// unboxed fp vreg, living raw across the whole loop (including safepoints:
/// spill-all parks it in a non-oop frame slot; `resolve_frame_loc` records
/// `ValueLoc::DoubleSlot`, and the deopt materializer boxes it back).
///
/// A temp `t` qualifies iff ALL hold:
/// - every real def is `Move { t, s }` with `s` an `FBox` result or a boxed
///   Double literal (`ConstPool` of a Double oop — rewritten to `FConst`);
/// - its mandatory nil-init (`ConstPool { t, nil }`) is provably DEAD: a
///   qualifying def occurs in the ENTRY block before any use of `t` and
///   before any safepoint (so neither the program nor a deopt can ever
///   observe the nil — Smalltalk's read-before-write would otherwise DNU
///   where promoted code would silently read a stale f64);
/// - every use is an `FUnbox { _, src: t }` (rewritten to a direct fp read,
///   guard deleted — the value is a Double by construction) or a
///   `Move { c, t }` whose `c` is DEOPT-ONLY (an operand-stack copy pinned
///   by reexecute metadata: `c` is promoted alongside `t`, and its DeoptRaw
///   references become `DoubleSlot` automatically);
/// - `t` appears in no merge entry stack.
///
/// GATED OFF for OSR compiles: the OSR entry copies INTERPRETER slot values
/// (boxed oops) into compiled frame slots — an fp slot would need an
/// unbox-with-guard at OSR entry, deferred.
fn promote_float_temps(m: &mut IrMethod) {
    use std::collections::HashMap;
    let n = m.vregs.len();

    // ── Whole-method censuses. ──────────────────────────────────────────
    let mut fbox_src: HashMap<u32, VReg> = HashMap::new(); // FBox dst → fp src
    let mut const_double: HashMap<u32, u64> = HashMap::new(); // ConstPool dst → f64 bits
    let mut op_uses: Vec<u32> = vec![0; n];
    let mut entry_used = vec![false; n];
    let mut funbox_uses: Vec<u32> = vec![0; n];
    let mut move_copy_of: Vec<Vec<u32>> = vec![Vec::new(); n]; // t → its copies c
    let mut bad_def = vec![false; n];
    let mut move_defs: Vec<u32> = vec![0; n];
    let mut nil_init_def: Vec<u32> = vec![0; n];
    let mut deopt_used = vec![false; n];

    for b in &m.blocks {
        for v in &b.entry_stack {
            entry_used[v.0 as usize] = true;
        }
        for (_, raw) in &b.deopt_sites {
            for v in &raw.stack {
                deopt_used[v.0 as usize] = true;
            }
            let mut lvl = raw.inline.as_ref();
            while let Some(site) = lvl {
                deopt_used[site.receiver.0 as usize] = true;
                for v in &site.slots {
                    deopt_used[v.0 as usize] = true;
                }
                for v in &site.caller_pending_stack {
                    deopt_used[v.0 as usize] = true;
                }
                lvl = site.parent.as_deref();
            }
        }
        for op in &b.code {
            op.uses(|v| op_uses[v.0 as usize] += 1);
            match op {
                Ir::FBox { dst, src } => {
                    fbox_src.insert(dst.0, *src);
                }
                Ir::ConstPool { dst, lit } => {
                    let e = &m.pool[lit.0 as usize];
                    if e.kind == Some(RelocKind::Oop) {
                        if let Some(d) = crate::oops::wrappers::DoubleOop::try_from(
                            crate::oops::Oop::from_raw(e.value),
                        ) {
                            const_double.insert(dst.0, d.value().to_bits());
                        }
                    }
                    // Def classification: EXACTLY the nil literal is the
                    // mandatory temp nil-init; any other ConstPool def (a
                    // Symbol, a non-Double literal…) disqualifies outright.
                    if *lit == m.nil_lit {
                        nil_init_def[dst.0 as usize] += 1;
                    } else {
                        bad_def[dst.0 as usize] = true;
                    }
                }
                Ir::FUnbox { src, .. } => funbox_uses[src.0 as usize] += 1,
                Ir::Move { dst, src } => {
                    move_copy_of[src.0 as usize].push(dst.0);
                    move_defs[dst.0 as usize] += 1;
                }
                _ => {}
            }
            // Non-Move/non-ConstPool defs disqualify their targets.
            if !matches!(op, Ir::Move { .. } | Ir::ConstPool { .. }) {
                op.defs(|v| bad_def[v.0 as usize] = true);
            }
        }
    }

    // ── Qualification. ──────────────────────────────────────────────────
    let mut qualified: Vec<u32> = Vec::new();
    'cand: for t in 1..n as u32 {
        let ti = t as usize;
        if m.vregs[ti].is_fp
            || move_defs[ti] == 0
            || bad_def[ti]
            || entry_used[ti]
            || nil_init_def[ti] > 1
        {
            continue;
        }
        // Uses: only FUnbox srcs + Move copies (each verified deopt-only).
        let copies = &move_copy_of[ti];
        if op_uses[ti] != funbox_uses[ti] + copies.len() as u32 {
            continue;
        }
        for &c in copies {
            let ci = c as usize;
            if op_uses[ci] != 0
                || entry_used[ci]
                || move_defs[ci] != 1
                || nil_init_def[ci] != 0
                || bad_def[ci]
            {
                continue 'cand;
            }
        }
        // Every Move def's source is an FBox result or a Double constant.
        for b in &m.blocks {
            for op in &b.code {
                if let Ir::Move { dst, src } = op {
                    if dst.0 == t
                        && !fbox_src.contains_key(&src.0)
                        && !const_double.contains_key(&src.0)
                    {
                        continue 'cand;
                    }
                }
            }
        }
        // Every FUnbox of t must be cleanly cancellable: all uses of its dst
        // are IN THE SAME BLOCK, after the unbox, before any redefinition of
        // t (the redefinition hazard), and account for the dst's every use.
        for b in &m.blocks {
            for (i, op) in b.code.iter().enumerate() {
                let u = match op {
                    Ir::FUnbox { dst, src, .. } if src.0 == t => dst.0,
                    _ => continue,
                };
                // Deleting the FUnbox deletes u's only def — u must not be
                // named by any deopt metadata.
                if deopt_used[u as usize] {
                    continue 'cand;
                }
                let mut in_window = 0u32;
                let mut redefined = false;
                for later in &b.code[i + 1..] {
                    let mut uses_u = false;
                    later.uses(|v| {
                        if v.0 == u {
                            uses_u = true;
                        }
                    });
                    if uses_u {
                        if redefined {
                            continue 'cand; // stranded past a redefinition
                        }
                        in_window += 1;
                    }
                    let mut defs_t = false;
                    later.defs(|v| {
                        if v.0 == t {
                            defs_t = true;
                        }
                    });
                    if defs_t {
                        redefined = true;
                    }
                }
                if in_window != op_uses[u as usize] {
                    continue 'cand; // cross-block use of the unboxed value
                }
            }
        }
        // Dead nil-init: a qualifying Move def in the ENTRY block before any
        // use of t and before any safepoint — so neither the program nor a
        // deopt can ever observe the nil this promotion erases.
        let entry = &m.blocks[0];
        let mut ok = false;
        for op in &entry.code {
            if is_safepoint_op(op) {
                break;
            }
            let mut used_here = false;
            op.uses(|v| {
                if v.0 == t {
                    used_here = true;
                }
            });
            if used_here {
                break;
            }
            if let Ir::Move { dst, .. } = op {
                if dst.0 == t {
                    ok = true;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        qualified.push(t);
    }
    if qualified.is_empty() {
        return;
    }

    // ── Rewrite (shapes pre-verified above — no bail-outs from here). ───
    let mut is_promoted = vec![false; n];
    for &t in &qualified {
        is_promoted[t as usize] = true;
        m.vregs[t as usize].is_fp = true;
        m.vregs[t as usize].is_oop = false;
        for &c in &move_copy_of[t as usize] {
            m.vregs[c as usize].is_fp = true;
            m.vregs[c as usize].is_oop = false;
        }
    }
    for b in &mut m.blocks {
        let len = b.code.len();
        let mut delete = vec![false; len];
        for i in 0..len {
            let replace: Option<Ir> = match &b.code[i] {
                Ir::ConstPool { dst, .. } if is_promoted[dst.0 as usize] => {
                    // The provably-dead nil-init — value irrelevant; an
                    // FConst keeps the def so interval shapes are unchanged.
                    Some(Ir::FConst { dst: *dst, bits: 0 })
                }
                Ir::Move { dst, src } if is_promoted[dst.0 as usize] => {
                    if let Some(&fp) = fbox_src.get(&src.0) {
                        Some(Ir::Move { dst: *dst, src: fp })
                    } else if let Some(&bits) = const_double.get(&src.0) {
                        Some(Ir::FConst { dst: *dst, bits })
                    } else {
                        unreachable!("promotion qualified a def it can't rewrite")
                    }
                }
                _ => None,
            };
            if let Some(new_op) = replace {
                b.code[i] = new_op;
                continue;
            }
            let (u, t) = match &b.code[i] {
                Ir::FUnbox { dst, src, .. } if is_promoted[src.0 as usize] => (dst.0, *src),
                _ => continue,
            };
            for later in b.code[i + 1..].iter_mut() {
                rewrite_uses(later, u, t);
            }
            delete[i] = true;
        }
        if delete.iter().any(|&d| d) {
            let mut removed = 0u32;
            let mut removed_before: Vec<u32> = vec![0; len + 1];
            for (i, item) in delete.iter().enumerate() {
                removed_before[i] = removed;
                if *item {
                    removed += 1;
                }
            }
            removed_before[len] = removed;
            let mut idx = 0usize;
            b.code.retain(|_| {
                let keep = !delete[idx];
                idx += 1;
                keep
            });
            for (ci, _) in b.deopt_sites.iter_mut() {
                *ci -= removed_before[*ci as usize];
            }
        }
    }
}

/// SPLICED-TEMP PROMOTION — `promote_float_temps` generalized to copy
/// trees. **GATED OFF by default; selected by `MACVM_SPLICE_PROMOTE=1`**
/// (`splice_promote_enabled`). Built 2026-07-22, kept dormant: it is green
/// under the full battery (world suite, GC/deopt stress, byte-identical
/// census) but no REAL workload demonstrated a win — the targeted-inlining
/// hypothesis it was built to enable was falsified (inlining the Mandelbrot
/// kernel regressed 9.4→14.6 ms/render even with this promotion working;
/// the call passed already-boxed temps, so call-boundary boxing was never
/// the gap). Full story: docs/float_fastpath_design.md addendum. Flip the
/// flag on (and re-run the battery) if a real workload shows box-per-
/// backedge splice shapes — small float helpers inlined at level 1.
///
/// Generalized (spliced-temp promotion) from single root temps to a COPY
/// TREE: the candidate `t` plus every vreg reachable from it through
/// single-def `Move`s — exactly the plumbing the splicers thread a value
/// through (the receiver copy into an inlined body; the boxed result
/// hopping back out). Without the tree, a loop-carried temp fed through an
/// inlined float callee re-boxed once per backedge forever
/// (`s := self scale: s`: the FBox flowed result → splice-return hop → t,
/// and t → splice receiver hop → FUnbox — neither side matched the old
/// direct-def / deopt-only-copy tests).
///
/// A candidate `t` (with its tree) qualifies iff ALL hold:
/// - every real def of `t` is `Move { t, s }` with `s` a tree member
///   (an fp-to-fp move after promotion) or resolvable to an `FBox` result /
///   Double constant by CHASING single-def `Move` hops (splice-return
///   plumbing; the emptied hops are swept, orphaning their FBox for rule 3);
/// - every tree node's every op use is an `FUnbox { _, src: node }`
///   (rewritten to a direct fp read, guard deleted) or the defining `Move`
///   of another tree node / of `t` itself. Deopt references are fine on any
///   node — the node is promoted and its references become `DoubleSlot`
///   (the materializer re-boxes, including as an inlined frame's receiver);
/// - DEFINED-BEFORE-OBSERVED, by forward dataflow over the real CFG
///   (`regalloc::successors`): no path from entry reaches an observation
///   point while `t` is still undefined — else promotion would show a
///   garbage f64 where the interpreter shows nil. Observation points are
///   every op use of a tree node, every deopt site naming one, and — for a
///   ROOT TEMP (detected by its mandatory nil-init) — every safepoint,
///   since a root temp is implicitly visible in EVERY safepoint's scope. A
///   non-temp vreg (splice plumbing, spliced callee temps) is scope-visible
///   only where deopt metadata names it explicitly. This subsumes the old
///   "real def in the entry block before any safepoint" gate;
/// - no tree node appears in a merge entry stack; the nil-init (root temps
///   only) is unique.
///
/// GATED OFF for OSR compiles: the OSR entry copies INTERPRETER slot values
/// (boxed oops) into compiled frame slots — an fp slot would need an
/// unbox-with-guard at OSR entry, deferred.
fn promote_float_temps_spliced(m: &mut IrMethod, osr_bci: Option<u16>) {
    use std::collections::{HashMap, HashSet};
    let n = m.vregs.len();

    // ── Whole-method censuses. ──────────────────────────────────────────
    let mut fbox_src: HashMap<u32, VReg> = HashMap::new(); // FBox dst → fp src
    let mut const_double: HashMap<u32, u64> = HashMap::new(); // ConstPool dst → f64 bits
    let mut op_uses: Vec<u32> = vec![0; n];
    let mut entry_used = vec![false; n];
    let mut funbox_uses: Vec<u32> = vec![0; n];
    let mut move_copy_of: Vec<Vec<u32>> = vec![Vec::new(); n]; // t → its copies c
    let mut bad_def = vec![false; n];
    let mut move_defs: Vec<u32> = vec![0; n];
    let mut move_single_src: Vec<u32> = vec![u32::MAX; n]; // valid iff move_defs==1
    let mut nil_init_def: Vec<u32> = vec![0; n];
    // ROOT-temp discriminator: a root temp's mandatory nil-init sits in
    // BLOCK 0 (the translation prologue); a SPLICED callee temp's nil-init
    // (the splicers mirror the interpreter's activation nil-init) sits
    // mid-method, inside the spliced region. Only a root temp is
    // implicitly visible at every safepoint's scope — a spliced temp is
    // visible only where inline deopt metadata names it explicitly.
    let mut nil_init_in_entry = vec![false; n];
    let mut deopt_used = vec![false; n];
    let mut ret_uses: Vec<u32> = vec![0; n];
    // Per BLOCK INDEX: each deopt site's (code index, referenced vregs) —
    // the dataflow's explicit observation points.
    let mut deopt_refs: Vec<Vec<(u32, Vec<u32>)>> = vec![Vec::new(); m.blocks.len()];

    for (bi, b) in m.blocks.iter().enumerate() {
        for v in &b.entry_stack {
            entry_used[v.0 as usize] = true;
        }
        for (ci, raw) in &b.deopt_sites {
            let mut refs: Vec<u32> = Vec::new();
            for v in &raw.stack {
                deopt_used[v.0 as usize] = true;
                refs.push(v.0);
            }
            let mut lvl = raw.inline.as_ref();
            while let Some(site) = lvl {
                deopt_used[site.receiver.0 as usize] = true;
                refs.push(site.receiver.0);
                for v in &site.slots {
                    deopt_used[v.0 as usize] = true;
                    refs.push(v.0);
                }
                for v in &site.caller_pending_stack {
                    deopt_used[v.0 as usize] = true;
                    refs.push(v.0);
                }
                lvl = site.parent.as_deref();
            }
            deopt_refs[bi].push((*ci, refs));
        }
        for op in &b.code {
            op.uses(|v| op_uses[v.0 as usize] += 1);
            match op {
                Ir::FBox { dst, src } => {
                    fbox_src.insert(dst.0, *src);
                }
                Ir::ConstPool { dst, lit } => {
                    let e = &m.pool[lit.0 as usize];
                    if e.kind == Some(RelocKind::Oop) {
                        if let Some(d) = crate::oops::wrappers::DoubleOop::try_from(
                            crate::oops::Oop::from_raw(e.value),
                        ) {
                            const_double.insert(dst.0, d.value().to_bits());
                        }
                    }
                    // Def classification: EXACTLY the nil literal is the
                    // mandatory temp nil-init; any other ConstPool def (a
                    // Symbol, a non-Double literal…) disqualifies outright.
                    if *lit == m.nil_lit {
                        nil_init_def[dst.0 as usize] += 1;
                        if bi == 0 {
                            nil_init_in_entry[dst.0 as usize] = true;
                        }
                    } else {
                        bad_def[dst.0 as usize] = true;
                    }
                }
                Ir::FUnbox { src, .. } => funbox_uses[src.0 as usize] += 1,
                Ir::Move { dst, src } => {
                    move_copy_of[src.0 as usize].push(dst.0);
                    move_defs[dst.0 as usize] += 1;
                    move_single_src[dst.0 as usize] = src.0;
                }
                Ir::Ret { val } => ret_uses[val.0 as usize] += 1,
                _ => {}
            }
            // Non-Move/non-ConstPool defs disqualify their targets.
            if !matches!(op, Ir::Move { .. } | Ir::ConstPool { .. }) {
                op.defs(|v| bad_def[v.0 as usize] = true);
            }
        }
    }

    // CFG predecessors (by block INDEX) for the defined-before-observed
    // dataflow — same edge set the allocator walks.
    let idx_of: HashMap<u32, usize> = m
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.0, i))
        .collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); m.blocks.len()];
    for (bi, b) in m.blocks.iter().enumerate() {
        for s in crate::compiler::regalloc::successors(b) {
            if let Some(&si) = idx_of.get(&s.0) {
                preds[si].push(bi);
            }
        }
    }

    // Chase single-def Move plumbing (the splice-return hops) back to an
    // FBox result or a Double constant. Hop vregs are recorded for the
    // dead-plumbing sweep once their consumer is rewritten away.
    enum Root {
        Fp(VReg),
        Bits(u64),
    }
    let chase = |start: u32, hops: &mut Vec<u32>| -> Option<Root> {
        let mut v = start;
        for _ in 0..16 {
            if let Some(&fp) = fbox_src.get(&v) {
                return Some(Root::Fp(fp));
            }
            if let Some(&bits) = const_double.get(&v) {
                return Some(Root::Bits(bits));
            }
            let vi = v as usize;
            if move_defs[vi] == 1 && nil_init_def[vi] == 0 && !bad_def[vi] {
                hops.push(v);
                v = move_single_src[vi];
            } else {
                return None;
            }
        }
        None
    };

    // ── Qualification. ──────────────────────────────────────────────────
    // A chased def's rewrite, planned at its exact (block index, op index).
    enum DefRw {
        Fp(VReg, VReg),   // Move { dst, fp_src }
        Bits(VReg, u64),  // FConst { dst, bits }
    }
    struct Promo {
        tree: Vec<u32>,
        def_plan: HashMap<(usize, usize), DefRw>,
        hops: Vec<u32>,
    }
    let mut promos: Vec<Promo> = Vec::new();
    let mut claimed = vec![false; n]; // each vreg joins at most one tree
    // Debugger (MACVM_DBG_IR's promotion companion): MACVM_DBG_PROMOTE=<N>
    // prints which gate rejects vreg N in every compile that considers it.
    // Debug builds only, stderr.
    #[cfg(debug_assertions)]
    let dbg_promote: Option<u32> = std::env::var("MACVM_DBG_PROMOTE")
        .ok()
        .and_then(|s| s.parse().ok());
    #[cfg(not(debug_assertions))]
    let dbg_promote: Option<u32> = None;
    macro_rules! dbg_reject {
        ($t:expr, $($msg:tt)*) => {
            if dbg_promote == Some($t) {
                eprintln!("[promote] v{} REJECTED: {}", $t, format!($($msg)*));
            }
        };
    }
    // The OSR header's IR block index (first match = the original block —
    // spliced blocks share the send's bci but are appended after all
    // originals). It is a SECOND dataflow entry below: OSR execution begins
    // there, with every candidate value undefined.
    let osr_header: Option<usize> =
        osr_bci.and_then(|bci| m.blocks.iter().position(|b| b.bci == bci as usize));
    'cand: for t in 1..n as u32 {
        let ti = t as usize;
        // OSR compile: a ROOT temp is copied into the frame as a BOXED
        // interpreter slot by the OSR entry — promoting it would type-pun
        // that copy. Spliced temps/plumbing are dead at the header and
        // stay eligible.
        if osr_bci.is_some() && nil_init_in_entry[ti] {
            dbg_reject!(t, "OSR compile: root temp (OSR entry copies boxed slots)");
            continue;
        }
        if m.vregs[ti].is_fp
            || claimed[ti]
            || move_defs[ti] == 0
            || bad_def[ti]
            || entry_used[ti]
            || nil_init_def[ti] > 1
        {
            dbg_reject!(
                t,
                "basic filter: is_fp={} claimed={} move_defs={} bad_def={} entry_used={} \
                 nil_inits={}",
                m.vregs[ti].is_fp,
                claimed[ti],
                move_defs[ti],
                bad_def[ti],
                entry_used[ti],
                nil_init_def[ti]
            );
            continue;
        }

        // The copy tree: t plus every vreg reachable via single-def Moves.
        let mut tree: Vec<u32> = vec![t];
        let mut in_tree: HashSet<u32> = HashSet::new();
        in_tree.insert(t);
        let mut wl: Vec<u32> = vec![t];
        while let Some(node) = wl.pop() {
            for &c in &move_copy_of[node as usize] {
                if c == t {
                    continue; // Move { t, node } — a def of t, gated below
                }
                if in_tree.contains(&c) {
                    dbg_reject!(t, "second Move def into tree node v{c}");
                    continue 'cand; // a second Move def into a tree node
                }
                let ci = c as usize;
                if move_defs[ci] == 1
                    && nil_init_def[ci] == 0
                    && !bad_def[ci]
                    && !entry_used[ci]
                    && !claimed[ci]
                    && !m.vregs[ci].is_fp
                {
                    in_tree.insert(c);
                    tree.push(c);
                    wl.push(c);
                } else {
                    dbg_reject!(
                        t,
                        "invalid copy v{c} of v{node}: move_defs={} nil_inits={} bad_def={} \
                         entry_used={} claimed={} is_fp={}",
                        move_defs[ci],
                        nil_init_def[ci],
                        bad_def[ci],
                        entry_used[ci],
                        claimed[ci],
                        m.vregs[ci].is_fp
                    );
                    continue 'cand;
                }
            }
        }
        // Every node's op uses are exactly its FUnboxes + its outgoing
        // Moves (each verified above to land in the tree or redefine t) +
        // its Rets (re-boxed once at the return — per call, not per
        // iteration; the rewrite's final pass).
        for &node in &tree {
            let ni = node as usize;
            if op_uses[ni]
                != funbox_uses[ni] + move_copy_of[ni].len() as u32 + ret_uses[ni]
            {
                dbg_reject!(
                    t,
                    "node v{node} use mismatch: op_uses={} funbox={} moves={} rets={}",
                    op_uses[ni],
                    funbox_uses[ni],
                    move_copy_of[ni].len(),
                    ret_uses[ni]
                );
                continue 'cand;
            }
        }

        // Every FUnbox of a tree node must be cleanly cancellable: all uses
        // of its dst in the same block, after the unbox, before any
        // redefinition of that node (only t can be redefined — copies are
        // single-def), accounting for the dst's every use.
        for b in &m.blocks {
            for (i, op) in b.code.iter().enumerate() {
                let (u, node) = match op {
                    Ir::FUnbox { dst, src, .. } if in_tree.contains(&src.0) => (dst.0, src.0),
                    _ => continue,
                };
                // Deleting the FUnbox deletes u's only def — u must not be
                // named by any deopt metadata.
                if deopt_used[u as usize] {
                    dbg_reject!(t, "FUnbox dst v{u} of node v{node} is deopt-referenced");
                    continue 'cand;
                }
                let mut in_window = 0u32;
                let mut redefined = false;
                for later in &b.code[i + 1..] {
                    let mut uses_u = false;
                    later.uses(|v| {
                        if v.0 == u {
                            uses_u = true;
                        }
                    });
                    if uses_u {
                        if redefined {
                            dbg_reject!(t, "FUnbox dst v{u}: use stranded past a redefinition of v{node}");
                            continue 'cand; // stranded past a redefinition
                        }
                        in_window += 1;
                    }
                    let mut defs_node = false;
                    later.defs(|v| {
                        if v.0 == node {
                            defs_node = true;
                        }
                    });
                    if defs_node {
                        redefined = true;
                    }
                }
                if in_window != op_uses[u as usize] {
                    dbg_reject!(t, "FUnbox dst v{u}: cross-block use ({in_window} of {})", op_uses[u as usize]);
                    continue 'cand; // cross-block use of the unboxed value
                }
            }
        }

        // Every real def of t: a tree member (fp-to-fp move after
        // promotion) or a chase to an FBox / Double-const root.
        let mut def_plan: HashMap<(usize, usize), DefRw> = HashMap::new();
        let mut hops: Vec<u32> = Vec::new();
        for (bi, b) in m.blocks.iter().enumerate() {
            for (i, op) in b.code.iter().enumerate() {
                if let Ir::Move { dst, src } = op {
                    if dst.0 != t || in_tree.contains(&src.0) {
                        continue;
                    }
                    match chase(src.0, &mut hops) {
                        Some(Root::Fp(fp)) => {
                            def_plan.insert((bi, i), DefRw::Fp(*dst, fp));
                        }
                        Some(Root::Bits(bits)) => {
                            def_plan.insert((bi, i), DefRw::Bits(*dst, bits));
                        }
                        None => {
                            dbg_reject!(t, "def Move src v{} unresolvable (chase failed)", src.0);
                            continue 'cand;
                        }
                    }
                }
            }
        }

        // Defined-before-observed dataflow: no path from entry may reach an
        // observation of the tree while t is still undefined (promotion
        // would show garbage f64 where the interpreter shows nil). A ROOT
        // temp (nil-init in the entry block) is implicitly visible in
        // EVERY safepoint's scope; a spliced callee temp (nil-init inside
        // the spliced region) and plain plumbing are visible only where
        // deopt metadata names them explicitly — which is what lets a
        // mid-method spliced init qualify despite the caller's earlier
        // safepoints.
        let is_temp = nil_init_def[ti] == 1 && nil_init_in_entry[ti];
        let nb = m.blocks.len();
        let mut first_obs: Vec<Option<usize>> = vec![None; nb];
        let mut first_def: Vec<Option<usize>> = vec![None; nb];
        for (bi, b) in m.blocks.iter().enumerate() {
            for (i, op) in b.code.iter().enumerate() {
                let mut observes = false;
                op.uses(|v| {
                    if in_tree.contains(&v.0) {
                        observes = true;
                    }
                });
                if is_temp && is_safepoint_op(op) {
                    observes = true;
                }
                if observes && first_obs[bi].is_none() {
                    first_obs[bi] = Some(i);
                }
                if first_def[bi].is_none() {
                    if let Ir::Move { dst, .. } = op {
                        if dst.0 == t {
                            first_def[bi] = Some(i);
                        }
                    }
                }
            }
            for (ci, refs) in &deopt_refs[bi] {
                if refs.iter().any(|r| in_tree.contains(r)) {
                    let at = *ci as usize;
                    if first_obs[bi].is_none_or(|o| at < o) {
                        first_obs[bi] = Some(at);
                    }
                }
            }
        }
        let mut undef_in = vec![false; nb];
        let mut undef_out = vec![false; nb];
        loop {
            let mut changed = false;
            for bi in 0..nb {
                let inn = bi == 0
                    || Some(bi) == osr_header
                    || preds[bi].iter().any(|&p| undef_out[p]);
                let out = if first_def[bi].is_some() { false } else { inn };
                if inn != undef_in[bi] || out != undef_out[bi] {
                    undef_in[bi] = inn;
                    undef_out[bi] = out;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for bi in 0..nb {
            if !undef_in[bi] {
                continue;
            }
            if let Some(o) = first_obs[bi] {
                // Observed while possibly undefined — unless a real def in
                // this block strictly precedes the observation.
                if first_def[bi].is_none_or(|d| o <= d) {
                    dbg_reject!(t, "observed-while-undef in block index {bi} (obs at {o}, def {:?})", first_def[bi]);
                    continue 'cand;
                }
            }
        }

        for &v in &tree {
            claimed[v as usize] = true;
        }
        promos.push(Promo {
            tree,
            def_plan,
            hops,
        });
    }
    if promos.is_empty() {
        return;
    }

    // ── Rewrite (shapes pre-verified above — no bail-outs from here). ───
    let mut is_promoted = vec![false; n];
    let mut all_hops: HashSet<u32> = HashSet::new();
    let mut def_plan_all: HashMap<(usize, usize), DefRw> = HashMap::new();
    for p in promos {
        for &v in &p.tree {
            is_promoted[v as usize] = true;
            m.vregs[v as usize].is_fp = true;
            m.vregs[v as usize].is_oop = false;
        }
        all_hops.extend(p.hops);
        def_plan_all.extend(p.def_plan);
    }
    for (bi, b) in m.blocks.iter_mut().enumerate() {
        let len = b.code.len();
        let mut delete = vec![false; len];
        for i in 0..len {
            // A chased def of t: retarget the Move straight at the FBox's
            // fp source (or bake the constant), bypassing the plumbing.
            if let Some(rw) = def_plan_all.get(&(bi, i)) {
                b.code[i] = match rw {
                    DefRw::Fp(dst, fp) => Ir::Move {
                        dst: *dst,
                        src: *fp,
                    },
                    DefRw::Bits(dst, bits) => Ir::FConst {
                        dst: *dst,
                        bits: *bits,
                    },
                };
                continue;
            }
            let replace: Option<Ir> = match &b.code[i] {
                Ir::ConstPool { dst, .. } if is_promoted[dst.0 as usize] => {
                    // The provably-dead nil-init — value irrelevant; an
                    // FConst keeps the def so interval shapes are unchanged.
                    Some(Ir::FConst { dst: *dst, bits: 0 })
                }
                Ir::Move { dst, src } if is_promoted[dst.0 as usize] => {
                    if is_promoted[src.0 as usize] {
                        None // intra-tree fp-to-fp move — correct as marked
                    } else {
                        unreachable!(
                            "promotion qualified a def it can't rewrite \
                             (non-tree defs are all in def_plan)"
                        )
                    }
                }
                _ => None,
            };
            if let Some(new_op) = replace {
                b.code[i] = new_op;
                continue;
            }
            let (u, t) = match &b.code[i] {
                Ir::FUnbox { dst, src, .. } if is_promoted[src.0 as usize] => (dst.0, *src),
                _ => continue,
            };
            for later in b.code[i + 1..].iter_mut() {
                rewrite_uses(later, u, t);
            }
            delete[i] = true;
        }
        if delete.iter().any(|&d| d) {
            let mut removed = 0u32;
            let mut removed_before: Vec<u32> = vec![0; len + 1];
            for (i, item) in delete.iter().enumerate() {
                removed_before[i] = removed;
                if *item {
                    removed += 1;
                }
            }
            removed_before[len] = removed;
            let mut idx = 0usize;
            b.code.retain(|_| {
                let keep = !delete[idx];
                idx += 1;
                keep
            });
            for (ci, _) in b.deopt_sites.iter_mut() {
                *ci -= removed_before[*ci as usize];
            }
        }
    }

    // ── Dead-plumbing sweep. The chased hops' consumers were retargeted
    // above; a hop whose value is now completely unobserved (no op use, no
    // deopt/entry reference) deletes with its defining Move — iterated,
    // since each deletion frees the next hop up the chain. The FBoxes this
    // orphans die in rule 3, which runs after promotion. A hop that IS
    // still referenced (e.g. pinned in a reexecute stack) simply keeps its
    // Move and its boxed value — correct, just unoptimized. ──────────────
    while !all_hops.is_empty() {
        let mut uses: Vec<u32> = vec![0; n];
        for b in &m.blocks {
            for op in &b.code {
                op.uses(|v| uses[v.0 as usize] += 1);
            }
        }
        let dead: HashSet<u32> = all_hops
            .iter()
            .copied()
            .filter(|&h| {
                uses[h as usize] == 0 && !deopt_used[h as usize] && !entry_used[h as usize]
            })
            .collect();
        if dead.is_empty() {
            break;
        }
        for h in &dead {
            all_hops.remove(h);
        }
        for b in &mut m.blocks {
            let len = b.code.len();
            let mut removed = 0u32;
            let mut removed_before: Vec<u32> = vec![0; len + 1];
            let mut keep: Vec<bool> = Vec::with_capacity(len);
            for (i, op) in b.code.iter().enumerate() {
                removed_before[i] = removed;
                let del = matches!(op, Ir::Move { dst, .. } if dead.contains(&dst.0));
                keep.push(!del);
                if del {
                    removed += 1;
                }
            }
            removed_before[len] = removed;
            if removed == 0 {
                continue;
            }
            let mut it = keep.iter();
            b.code.retain(|_| *it.next().unwrap());
            for (ci, _) in b.deopt_sites.iter_mut() {
                *ci -= removed_before[*ci as usize];
            }
        }
    }

    // ── Ret re-boxing. A promoted value that IS the method's answer boxes
    // exactly once, at the return — per call, never per iteration. `Ret` is
    // always its block's final op, so the insertion shifts no deopt-site
    // index (sites never point at or past a terminator). ─────────────────
    let mut ret_fixes: Vec<(usize, VReg)> = Vec::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        if let Some(Ir::Ret { val }) = b.code.last() {
            if is_promoted[val.0 as usize] {
                ret_fixes.push((bi, *val));
            }
        }
    }
    for (bi, val) in ret_fixes {
        let boxed = VReg(m.vregs.len() as u32);
        m.vregs.push(VRegInfo {
            is_oop: true,
            is_fp: false,
        });
        let b = &mut m.blocks[bi];
        let ret_ix = b.code.len() - 1;
        b.code[ret_ix] = Ir::FBox { dst: boxed, src: val };
        b.code.push(Ir::Ret { val: boxed });
    }
}

/// `op` reads `from` → make it read `to` (fp positions only — the promotion
/// only ever rewrites fp-valued reads: FArith/FCmp*/FBox sources and fp
/// Moves).
fn rewrite_uses(op: &mut Ir, from: u32, to: VReg) {
    match op {
        Ir::FArith { a, b, .. } | Ir::FCmpVal { a, b, .. } | Ir::FCmpBr { a, b, .. } => {
            if a.0 == from {
                *a = to;
            }
            if b.0 == from {
                *b = to;
            }
        }
        Ir::FBox { src, .. } | Ir::FUnbox { src, .. } | Ir::Move { src, .. } => {
            if src.0 == from {
                *src = to;
            }
        }
        _ => {}
    }
}

/// Op-level safepoint test for the promotion's dead-nil-init gate — mirrors
/// `regalloc::is_safepoint` (kept in sync by the shared-variant match).
fn is_safepoint_op(ir: &Ir) -> bool {
    matches!(
        ir,
        Ir::CallSend { .. }
            | Ir::CallRuntime { .. }
            | Ir::Alloc { .. }
            | Ir::FBox { .. }
            | Ir::VecArith { .. }
            | Ir::UncommonTrap { .. }
            | Ir::Poll
    )
}

/// Float fast-path (`docs/float_fastpath_design.md` B2 rules 2–3): the
/// box/unbox cancellation reducer. Per block, forward:
///
/// - **Rule 2 — `FUnbox(FBox x) → x`.** An `FUnbox` whose source is an
///   in-block `FBox` result is DELETED (with its klass guard: the value is
///   provably a Double — the box just made it) and downstream fp reads
///   (`FArith`/`FCmp*`/`FBox` sources) rewrite to the original unboxed value.
///   Sound across safepoints: if the rewritten fp value's interval now
///   crosses one, regalloc's fp spill-all parks it in a non-oop slot.
/// - **Rule 3 — dead-box elimination.** An `FBox` whose boxed result has no
///   remaining use anywhere — ops, deopt metadata (a later fused op's
///   reexecute stack legitimately pins an intermediate box: the interpreter
///   needs the REAL boxed Double on re-execution), or a merge entry stack —
///   is deleted. This is where the allocation disappears. (Sinking a
///   deopt-pinned box into its fail block — deferred; see the design doc.)
///
/// Runs after [`copy_propagate`] (which collapses the Move chains between an
/// `FBox` and its consumer `FUnbox`, making rule 2's def visible in-block).
pub(crate) fn reduce_float_boxes(m: &mut IrMethod, osr_bci: Option<u16>) {
    // ── Rule 2: per-block cancel. ────────────────────────────────────────
    for b in &mut m.blocks {
        let mut boxed_src: HashMap<u32, VReg> = HashMap::new(); // FBox dst → fp src
        let mut fp_alias: HashMap<u32, VReg> = HashMap::new(); // FUnbox dst → fp value
        let mut keep: Vec<bool> = Vec::with_capacity(b.code.len());
        let mut removed = 0u32;
        let mut removed_before: Vec<u32> = vec![0; b.code.len() + 1];
        for (i, op) in b.code.iter_mut().enumerate() {
            removed_before[i] = removed;
            if !fp_alias.is_empty() {
                match op {
                    Ir::FArith { a, b: rb, .. }
                    | Ir::FCmpVal { a, b: rb, .. }
                    | Ir::FCmpBr { a, b: rb, .. } => {
                        if let Some(&r) = fp_alias.get(&a.0) {
                            *a = r;
                        }
                        if let Some(&r) = fp_alias.get(&rb.0) {
                            *rb = r;
                        }
                    }
                    Ir::FBox { src, .. } => {
                        if let Some(&r) = fp_alias.get(&src.0) {
                            *src = r;
                        }
                    }
                    _ => {}
                }
            }
            let deletable = if let Ir::FUnbox { dst, src, .. } = op {
                match boxed_src.get(&src.0) {
                    Some(&fp) => {
                        fp_alias.insert(dst.0, fp);
                        true
                    }
                    None => false,
                }
            } else {
                false
            };
            if let Ir::FBox { dst, src } = op {
                boxed_src.insert(dst.0, *src);
            }
            keep.push(!deletable);
            if deletable {
                removed += 1;
            }
        }
        removed_before[b.code.len()] = removed;
        if removed == 0 {
            continue;
        }
        let mut it = keep.iter();
        b.code.retain(|_| *it.next().unwrap());
        for (ci, _) in b.deopt_sites.iter_mut() {
            *ci -= removed_before[*ci as usize];
        }
    }

    // ── Rule 2.5: FUnbox of a Double LITERAL folds to FConst. The guard is
    // provably true (the pool word IS a Double) and an immutable literal's
    // VALUE is a compile-time constant even though the boxed object moves —
    // so the guard, the untag, and the payload load all vanish into a
    // register constant. The ConstPool def stays (reexecute stacks may pin
    // the boxed literal); if dead it just materializes an unused pool word.
    {
        use std::collections::HashMap;
        let mut const_double: HashMap<u32, u64> = HashMap::new();
        for b in &m.blocks {
            for op in &b.code {
                if let Ir::ConstPool { dst, lit } = op {
                    let e = &m.pool[lit.0 as usize];
                    if e.kind == Some(RelocKind::Oop) {
                        if let Some(d) = crate::oops::wrappers::DoubleOop::try_from(
                            crate::oops::Oop::from_raw(e.value),
                        ) {
                            const_double.insert(dst.0, d.value().to_bits());
                        }
                    }
                }
            }
        }
        if !const_double.is_empty() {
            for b in &mut m.blocks {
                for op in b.code.iter_mut() {
                    let (dst, bits) = match op {
                        Ir::FUnbox { dst, src, .. } => match const_double.get(&src.0) {
                            Some(&bits) => (*dst, bits),
                            None => continue,
                        },
                        _ => continue,
                    };
                    *op = Ir::FConst { dst, bits };
                }
            }
        }
    }

    // ── Rule 4 (docs B5): float-temp promotion — before the sink/census
    // so the temp-store FBoxes it orphans die in rule 3 below. The DEFAULT
    // is the original root-temp-only pass (OSR compiles gated out — the
    // OSR entry copies boxed interpreter slots). MACVM_SPLICE_PROMOTE=1
    // selects the generalized spliced-temp pass instead — dormant
    // capability, see `promote_float_temps_spliced`'s own doc.
    if splice_promote_enabled() {
        promote_float_temps_spliced(m, osr_bci);
    } else if osr_bci.is_none() {
        promote_float_temps(m);
    }

    // ── Rule 3a: deopt-sunk boxing. An intermediate FBox whose boxed
    // result is referenced ONLY by trap-only fail blocks' reexecute
    // metadata (a LATER fused op's DeoptRaw.stack — the interpreter needs
    // the real boxed Double if that op deopts) moves INTO those fail
    // blocks: the box executes only when a deopt actually happens, and the
    // hot path allocates nothing for it. No new deopt machinery — the
    // DeoptRaw still names an ordinary boxed oop vreg; its def just lives
    // in the cold block now, and the existing deopt-live spill discipline
    // covers it at the trap. Conservative gates: zero op uses, zero
    // entry-stack uses (a cross-block operand-stack value stays boxed), and
    // every referencing block is exactly `[UncommonTrap]`. ────────────────
    {
        let nv = m.vregs.len();
        let mut op_used = vec![false; nv];
        let mut entry_used = vec![false; nv];
        // vreg → referencing block indices (deopt metadata only).
        let mut deopt_refs: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (bi, b) in m.blocks.iter().enumerate() {
            for op in &b.code {
                op.uses(|v| op_used[v.0 as usize] = true);
            }
            for v in &b.entry_stack {
                entry_used[v.0 as usize] = true;
            }
            for (_, raw) in &b.deopt_sites {
                for v in &raw.stack {
                    deopt_refs[v.0 as usize].push(bi);
                }
                let mut lvl = raw.inline.as_ref();
                while let Some(site) = lvl {
                    deopt_refs[site.receiver.0 as usize].push(bi);
                    for v in &site.slots {
                        deopt_refs[v.0 as usize].push(bi);
                    }
                    for v in &site.caller_pending_stack {
                        deopt_refs[v.0 as usize].push(bi);
                    }
                    lvl = site.parent.as_deref();
                }
            }
        }
        let trap_only: Vec<bool> = m
            .blocks
            .iter()
            .map(|b| b.code.len() == 1 && matches!(b.code[0], Ir::UncommonTrap { .. }))
            .collect();

        // Collect (home block, op index, target blocks, op clone) first;
        // mutate after — insertions into fail blocks must not disturb the
        // scan, and a home block's own indices re-key on deletion.
        let mut deletions: Vec<Vec<usize>> = vec![Vec::new(); m.blocks.len()];
        let mut insertions: Vec<Vec<Ir>> = (0..m.blocks.len()).map(|_| Vec::new()).collect();
        for (bi, b) in m.blocks.iter().enumerate() {
            for (i, op) in b.code.iter().enumerate() {
                let Ir::FBox { dst, src } = op else { continue };
                let d = dst.0 as usize;
                if op_used[d] || entry_used[d] || deopt_refs[d].is_empty() {
                    continue;
                }
                if !deopt_refs[d].iter().all(|&t| trap_only[t] && t != bi) {
                    continue;
                }
                deletions[bi].push(i);
                let mut targets = deopt_refs[d].clone();
                targets.sort_unstable();
                targets.dedup();
                for t in targets {
                    insertions[t].push(Ir::FBox { dst: *dst, src: *src });
                }
            }
        }
        for (bi, b) in m.blocks.iter_mut().enumerate() {
            if !deletions[bi].is_empty() {
                let dels = &deletions[bi];
                let mut removed_before: Vec<u32> = vec![0; b.code.len() + 1];
                let mut removed = 0u32;
                for i in 0..b.code.len() {
                    removed_before[i] = removed;
                    if dels.contains(&i) {
                        removed += 1;
                    }
                }
                removed_before[b.code.len()] = removed;
                let mut idx = 0usize;
                b.code.retain(|_| {
                    let keep = !dels.contains(&idx);
                    idx += 1;
                    keep
                });
                for (ci, _) in b.deopt_sites.iter_mut() {
                    *ci -= removed_before[*ci as usize];
                }
            }
            let ins = std::mem::take(&mut insertions[bi]);
            if !ins.is_empty() {
                let k = ins.len() as u32;
                for (ci, _) in b.deopt_sites.iter_mut() {
                    *ci += k;
                }
                for (j, op) in ins.into_iter().enumerate() {
                    b.code.insert(j, op);
                }
            }
        }
    }

    // ── Rule 3: whole-method use census, then delete unused FBoxes. ─────
    let n = m.vregs.len();
    let mut used = vec![false; n];
    for b in &m.blocks {
        for op in &b.code {
            op.uses(|v| used[v.0 as usize] = true);
        }
        for (_, raw) in &b.deopt_sites {
            for v in &raw.stack {
                used[v.0 as usize] = true;
            }
            let mut lvl = raw.inline.as_ref();
            while let Some(site) = lvl {
                used[site.receiver.0 as usize] = true;
                for v in &site.slots {
                    used[v.0 as usize] = true;
                }
                for v in &site.caller_pending_stack {
                    used[v.0 as usize] = true;
                }
                lvl = site.parent.as_deref();
            }
        }
        for v in &b.entry_stack {
            used[v.0 as usize] = true;
        }
    }
    for b in &mut m.blocks {
        let mut keep: Vec<bool> = Vec::with_capacity(b.code.len());
        let mut removed = 0u32;
        let mut removed_before: Vec<u32> = vec![0; b.code.len() + 1];
        for (i, op) in b.code.iter().enumerate() {
            removed_before[i] = removed;
            let dead = matches!(op, Ir::FBox { dst, .. } if !used[dst.0 as usize]);
            keep.push(!dead);
            if dead {
                removed += 1;
            }
        }
        removed_before[b.code.len()] = removed;
        if removed == 0 {
            continue;
        }
        let mut it = keep.iter();
        b.code.retain(|_| *it.next().unwrap());
        for (ci, _) in b.deopt_sites.iter_mut() {
            *ci -= removed_before[*ci as usize];
        }
    }
}

pub fn convert(
    vm: &VmState,
    rcvr_klass: KlassOop,
    method: MethodOop,
    cfg: &Cfg,
    osr_bci: Option<u16>,
) -> IrMethod {
    let osr = osr_bci.is_some();
    let (entry_depth, _max_stack_depth) = compute_entry_depths(method, cfg);
    let sources = entry_stack_sources(cfg, &entry_depth);

    let block_id = |i: BlockIndex| BlockId(i as u32);

    // S14 step 7-I / S24 A3: the escape pre-pass, iff M creates a literal
    // closure. Hoisted before the vreg allocation because A3b's
    // materialize-vs-elide decision below depends on it.
    let escape: Option<crate::compiler::escape::ClosureEscape> = {
        let mut has_closure = false;
        let mut b = 0usize;
        let bc_len = method.bytecode_len();
        while b < bc_len {
            let (instr, next) = decode_at(method, b);
            if matches!(instr, Instr::PushClosure { .. }) {
                has_closure = true;
                break;
            }
            b = next;
        }
        if has_closure {
            // S24 B3: same Some(rcvr_klass) as the driver's two analysis
            // runs — all three must agree (front-5). Blocks never reach
            // here (no PushClosure in a compiled block's own bytecode),
            // but guard anyway: a block compile's rcvr_klass is the
            // closure_klass filler, meaningless for devirt.
            let k = if method.is_block() {
                None
            } else {
                Some(rcvr_klass)
            };
            Some(crate::compiler::escape::analyze_customized(vm, method, k))
        } else {
            None
        }
    };
    // S24 A3b (design §2.7): M owns a heap Context AND compiles via the
    // ESCAPING path (some closure escapes — not `all_elidable`), so the
    // Context CANNOT be elided (an escaping capturing closure needs a REAL
    // Context in `copied[1]`). The prologue MATERIALIZES it and every
    // depth-0 ctx-temp op goes through it, exactly like A1's block
    // `copied[1]`. When `all_elidable` (every capturing block inlines), the
    // Context still elides (7-II-b) and this stays false.
    let materialize_ctx =
        method.has_ctx() && !method.is_block() && escape.as_ref().is_some_and(|e| !e.all_elidable);

    let self_vreg = VReg(0);
    let mut vregs = vec![VRegInfo { is_oop: true, is_fp: false }];
    // `temp_vregs[i]` is the unified arg/temp slot `Frame::temp_index` also
    // uses (SPEC §5.1): `0..argc` are args, `argc..argc+ntemps` are the
    // method's own local temps — `method.ntemps()` alone is only the
    // LATTER count (`interpreter::stack::Frame::temp_index`'s own
    // `t < argc + ntemps` bound confirms this), so the vreg vector needs
    // `argc + ntemps` total entries, not `ntemps`.
    let temp_vregs: Vec<VReg> = (0..method.argc() + method.ntemps())
        .map(|i| {
            vregs.push(VRegInfo { is_oop: true, is_fp: false });
            VReg((i + 1) as u32)
        })
        .collect();
    // S14 step 7-II-b: a vreg per M ctx-slot (only when M is `has_ctx` — its
    // heap Context is elided and its captured temps live in these vregs).
    // S24 A3b: a MATERIALIZE method routes every ctx-temp through its real heap
    // Context (`method_ctx_vreg`), so these elided 7-II-b promotion vregs are
    // VESTIGIAL — and mutually exclusive with `method_ctx_vreg` by design (its
    // doc). Allocating them anyway made each an `is_oop` vreg the deopt scope
    // keeps live method-wide but which nothing ever writes OR nil-fills (the
    // materialize prologue's Context-alloc branch skips the elided nil-init at
    // §5516), leaving a live oopmap slot holding stack garbage at a GC
    // safepoint — the A3b crash (`printOn:` slot 4 = raw 0x1, localized by
    // `MACVM_TRACE=oops`). Enforce the exclusivity here: no elided vregs when
    // materializing.
    let ctx_vregs: Vec<VReg> = if materialize_ctx {
        Vec::new()
    } else {
        (0..method.nctx())
            .map(|_| {
                let n = vregs.len() as u32;
                vregs.push(VRegInfo { is_oop: true, is_fp: false });
                VReg(n)
            })
            .collect()
    };
    // S24 A1: a block compilation's environment vregs (design §2.2). The
    // closure arrives as Param 0 (the value:-send's receiver slot, exactly
    // what `enter_compiled`/the A2 stub already marshal into x0); `self`
    // and the home Context are prologue-synthesized LoadFields off it.
    let block_closure_vreg: Option<VReg> = if method.is_block() {
        let n = vregs.len() as u32;
        vregs.push(VRegInfo { is_oop: true, is_fp: false });
        Some(VReg(n))
    } else {
        None
    };
    let block_ctx_vreg: Option<VReg> = if method.is_block() && method.captures_ctx() {
        let n = vregs.len() as u32;
        vregs.push(VRegInfo { is_oop: true, is_fp: false });
        Some(VReg(n))
    } else {
        None
    };
    // S24 A3b: the vreg holding M's own MATERIALIZED heap Context (the
    // prologue allocates it; ctx-temp ops and the AllocClosure `copied[1]`
    // store both read it). Distinct from `ctx_vregs` (7-II-b's ELIDED
    // per-slot promotion) — mutually exclusive per method.
    let method_ctx_vreg: Option<VReg> = if materialize_ctx {
        let n = vregs.len() as u32;
        vregs.push(VRegInfo { is_oop: true, is_fp: false });
        Some(VReg(n))
    } else {
        None
    };
    let mut t = Translator {
        vm,
        method,
        spliced_nlr: 0,
        spliced_multibb: 0,
        budget_bytes_used: 0,
        splice_declined_budget: 0,
        rcvr_klass,
        self_devirt: false,
        osr,
        vregs,
        pool: PoolBuilder::new(),
        self_vreg,
        temp_vregs: temp_vregs.clone(),
        ctx_vregs: ctx_vregs.clone(),
        block_closure_vreg,
        block_ctx_vreg,
        method_ctx_vreg,
        next_extra_block: cfg.blocks.len() as u32,
        blocks_by_id: (0..cfg.blocks.len()).map(|_| None).collect(),
        call_sites: Vec::new(),
        site_feedback: Vec::new(),
        inline_deps: Vec::new(),
        // Tier-1 compiles are always level 1 today (`driver::compile_method`
        // sets `level: 1`); the inline budget scales with this when higher
        // levels arrive (S14 step 8).
        level: 1,
        osr_cold_sends: 0,
        const_class: HashMap::new(),
        // S14 step 7-I: run the escape pre-pass iff M creates a literal closure.
        // Hoisted above (A3b needs it for the materialize decision); a
        // closure-free method's `escape` is `None` and pays nothing.
        escape,
        temp_sites: vec![None; method.argc() + method.ntemps()],
        pending_jump_target: None,
    };

    // Pre-allocate merge vregs for every block that needs them (D3.2) —
    // predecessors need these ids available when THEY emit their own
    // terminator, which for a forward edge happens before the merge
    // block itself is ever translated.
    let mut entry_stacks: Vec<Option<Vec<VReg>>> = vec![None; cfg.blocks.len()];
    for (b, source) in sources.iter().enumerate() {
        match source {
            EntryStackSource::Empty => entry_stacks[b] = Some(Vec::new()),
            EntryStackSource::Merge => {
                let depth = entry_depth[b] as usize;
                let regs = (0..depth).map(|_| t.fresh(true)).collect();
                entry_stacks[b] = Some(regs);
            }
            EntryStackSource::Inherit(_) => {}
        }
    }

    let nil_lit = t
        .pool
        .intern(vm.universe.nil_obj.raw(), Some(RelocKind::Oop));
    // `SmiCmpVal`/`BoolBr`'s own emit sequences (D5.3) need true_obj/
    // false_obj as pool constants regardless of whether this method's
    // bytecode happens to contain an explicit push_true/push_false —
    // eagerly interning them here (not gated on the fused-comparison rule
    // that produces those Ir ops needing them) guarantees `emit.rs`, which
    // has no VmState access of its own, always finds them via
    // `IrMethod::true_lit`/`false_lit` rather than a fragile fixed-index
    // convention.
    let true_lit = t
        .pool
        .intern(vm.universe.true_obj.raw(), Some(RelocKind::Oop));
    let false_lit = t
        .pool
        .intern(vm.universe.false_obj.raw(), Some(RelocKind::Oop));
    // S11 D7: the pristine mark word an inline `Alloc` stamps into a fresh
    // `Format::Slots` object's header — a raw immediate (NOT a
    // `RelocKind::Oop`: it's a mark word, not a heap reference the GC
    // updates), matching `memory::alloc::init_object_at`'s own
    // `Mark::pristine().with_tagged_contents(true)`.
    let mark_slots_lit = t.pool.intern(
        crate::oops::mark::Mark::pristine()
            .with_tagged_contents(true)
            .word(),
        None,
    );
    // Float fast-path: a fresh Double's mark (raw, untagged contents — the
    // f64 payload must never be scanned as an oop) + the Double klass, for
    // FBox/FUnbox. Interned unconditionally like mark_slots_lit (the pool
    // dedups; a method with no Double sends just carries two idle words).
    let mark_double_lit = t.pool.intern(
        crate::oops::mark::Mark::pristine()
            .with_tagged_contents(false)
            .word(),
        None,
    );
    let double_klass_lit = t
        .pool
        .intern(vm.universe.double_klass.oop().raw(), Some(RelocKind::Oop));
    // SIMD: the Float64x2 / Float32x4 klasses, for VecArith's guards + box
    // header (docs/SIMD.md).
    let float64x2_klass_lit = t
        .pool
        .intern(vm.universe.float64x2_klass.oop().raw(), Some(RelocKind::Oop));
    let float32x4_klass_lit = t
        .pool
        .intern(vm.universe.float32x4_klass.oop().raw(), Some(RelocKind::Oop));
    let int32x4_klass_lit = t
        .pool
        .intern(vm.universe.int32x4_klass.oop().raw(), Some(RelocKind::Oop));
    // S24 A3b: the Context klass, interned once for the prologue Context
    // allocation of a MATERIALIZING has_ctx method.
    let ctx_klass_lit = if materialize_ctx {
        Some(
            t.pool
                .intern(vm.universe.context_klass.oop().raw(), Some(RelocKind::Oop)),
        )
    } else {
        None
    };

    let mut exit_stacks: Vec<Option<Vec<VReg>>> = vec![None; cfg.blocks.len()];

    // S14 step 3: forward reachability over LIVE (non-trap) control flow. A
    // block that lowers a send to an uncommon TRAP terminates the method there,
    // so its CFG successors are unreachable UNLESS a live edge from another
    // block also reaches them. `compute_entry_depths`/`entry_stack_sources` are
    // computed on the no-trap CFG (they can't see a per-site trap decision), so
    // once a block traps, a downstream merge that was to receive a value from it
    // would otherwise be fed a mismatched (or empty) operand stack — corrupting
    // the merge (an `emit_merges` depth-mismatch panic, or worse a wrong Move).
    // Tracking reachability lets `convert` skip DEAD blocks entirely: they are
    // finished as trivial `Ir::Bailout` blocks (never reached at runtime, and
    // regalloc's own reachability DFS never orders them into the live path — the
    // exact treatment decode's post-return dead blocks already get) and, crucially,
    // never feed a live merge. Blocks are processed in increasing bci order, so
    // every forward predecessor of a merge runs (and marks it reachable) before
    // the merge itself; a backward-edge (loop) target was already reached
    // forward, so its liveness is already settled.
    let mut reachable = vec![false; cfg.blocks.len()];
    reachable[0] = true;

    for (b, cfg_block) in cfg.blocks.iter().enumerate() {
        // A dead block (unreachable once traps are accounted for): finish a
        // trivial placeholder so `blocks_by_id` is complete, but translate
        // nothing and feed no merges. Its `entry_stack`/`exit_stacks` are never
        // consulted (no live successor inherits from it — any `Inherit(dead)`
        // block is itself dead and skipped here too).
        if !reachable[b] {
            t.finish_block(IrBlock {
                id: block_id(b),
                bci: cfg_block.bci_start,
                // `RetSelf`: a valid, operand-free terminator (reads only
                // `VReg(0)`, always present). This block is never reached at
                // runtime and never ordered into the live path by regalloc's DFS
                // from block 0; the terminator exists only so the block is
                // structurally well-formed if the DFS's dead-code sweep appends
                // it.
                code: vec![Ir::RetSelf],
                entry_stack: Vec::new(),
                deopt_sites: Vec::new(),
            });
            continue;
        }

        let entry_stack = match sources[b] {
            EntryStackSource::Empty | EntryStackSource::Merge => {
                entry_stacks[b].clone().expect("pre-allocated above")
            }
            EntryStackSource::Inherit(pred) => exit_stacks[pred].clone().expect(
                "entry_stack_sources guarantees a non-backward single predecessor is \
                 translated first (blocks are walked in increasing bci order)",
            ),
        };

        let mut code = Vec::new();
        // S13 step 3b: deopt sites accumulate alongside `code`, keyed by the
        // op's index in THIS code vec — `std::mem::take`n into the finished
        // IrBlock in lockstep with `code` at every split/finish so the
        // indices stay valid against the vec they name (a split resets both
        // to empty, and the continuation's own ops renumber from 0).
        let mut deopt: Vec<(u32, DeoptRaw)> = Vec::new();
        if b == 0 {
            if let Some(closure_vreg) = block_closure_vreg {
                // S24 A1 (design §2.2): a BLOCK compilation's entry. Param 0
                // is the CLOSURE (the value:-send's receiver — exactly what
                // `enter_compiled` marshals into x0); `self` is the home
                // receiver, `copied[0]`; the home Context (iff captures_ctx)
                // is `copied[1]`. All plain LoadFields — no allocation, no
                // safepoint. Closure body layout (`oops::closure`):
                // [0]=method, [1]=home, [2]=ncopied(size slot), [3+i]=copied[i].
                use crate::oops::layout::{BODY_OFFSET, CLOSURE_NAMED_WORDS};
                let copied_base = CLOSURE_NAMED_WORDS + 1; // past the size slot
                code.push(Ir::Param {
                    dst: closure_vreg,
                    index: 0,
                });
                for (i, &tv) in temp_vregs.iter().enumerate().take(method.argc()) {
                    code.push(Ir::Param {
                        dst: tv,
                        index: (i + 1) as u8,
                    });
                }
                code.push(Ir::LoadField {
                    dst: self_vreg,
                    obj: closure_vreg,
                    byte_off: (BODY_OFFSET + 8 * copied_base) as i32,
                });
                if let Some(ctx) = block_ctx_vreg {
                    code.push(Ir::LoadField {
                        dst: ctx,
                        obj: closure_vreg,
                        byte_off: (BODY_OFFSET + 8 * (copied_base + 1)) as i32,
                    });
                }
                // Local temps nil-init. Value captures would be pre-loads
                // here instead — v1 declines capture-bearing closures at the
                // compile trigger (`ncopied > 1 + captures_ctx`), so every
                // non-arg temp is a plain local (activate_block's own
                // convention with zero captures).
                for &tv in temp_vregs.iter().skip(method.argc()) {
                    code.push(Ir::ConstPool {
                        dst: tv,
                        lit: nil_lit,
                    });
                }
                debug_assert!(
                    ctx_vregs.is_empty(),
                    "convert(is_block): has_ctx blocks are compile-ineligible in A1, \
                     so a block never owns elided-ctx vregs"
                );
            } else {
                code.push(Ir::Param {
                    dst: self_vreg,
                    index: 0,
                });
                for (i, &tv) in temp_vregs.iter().enumerate().take(method.argc()) {
                    code.push(Ir::Param {
                        dst: tv,
                        index: (i + 1) as u8,
                    });
                }
                for &tv in temp_vregs.iter().skip(method.argc()) {
                    code.push(Ir::ConstPool {
                        dst: tv,
                        lit: nil_lit,
                    });
                }
                if let Some(ctx) = method_ctx_vreg {
                    // S24 A3b (design §2.7): MATERIALIZE M's own heap Context
                    // in the prologue — an escaping capturing closure will
                    // store this SAME object into its `copied[1]` (Risk 2:
                    // one Context by identity). `Ir::Alloc` nil-inits the whole
                    // body (`emit.rs`), so home_hint@16 (nil — M is a method,
                    // no enclosing chain) and the nctx slots@32+ are already
                    // nil; only the size word@24 needs a store. Layout mirrors
                    // `alloc_context`: [0]=home_hint, [1]=size, [2+i]=slot.
                    let nctx = method.nctx();
                    let size_words = (crate::oops::layout::HEADER_WORDS + 2 + nctx) as u32;
                    // The prologue alloc is a GC safepoint (never deopts in
                    // practice — no trap/poll/return-redirect here); its
                    // reexecute point is method entry (bci 0), where the
                    // interpreter's own `activate_method` re-allocates the
                    // Context. The Context is NOT yet live at this safepoint,
                    // so `build_deopt_metadata` records `CtxLoc::None` here
                    // (keyed on this being the ctx-alloc site).
                    deopt.push((
                        code.len() as u32,
                        DeoptRaw {
                            stack: Vec::new(),
                            bci: 0,
                            kind: SafepointKind::Alloc,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    ));
                    code.push(Ir::Alloc {
                        dst: ctx,
                        klass: ctx_klass_lit.expect("materialize_ctx => ctx_klass_lit interned"),
                        size_words,
                    });
                    let size_v = t.fresh(false);
                    code.push(Ir::ConstSmi {
                        dst: size_v,
                        value: nctx as i64,
                    });
                    code.push(Ir::StoreField {
                        obj: ctx,
                        byte_off: (crate::oops::layout::BODY_OFFSET + 8) as i32,
                        val: size_v,
                        barrier: false,
                    });
                } else {
                    // S14 step 7-II-b: nil-init the promoted ctx-temp vregs (M's
                    // elided Context's slots start nil, matching
                    // `alloc_context`'s nil-fill — so a ctx-temp read before it
                    // is written yields nil).
                    for &cv in ctx_vregs.iter() {
                        code.push(Ir::ConstPool {
                            dst: cv,
                            lit: nil_lit,
                        });
                    }
                }
            }
        }

        let is_branch_terminator = matches!(cfg_block.terminator, Terminator::Branch { .. });
        let mut stack = entry_stack.clone();
        // S14 step 7-I: parallel to `stack` — `stack_sites[i] == Some(bci)` iff
        // operand-stack slot `i` currently holds the (phantom) block created at
        // `push_closure` site `bci`. A block-site never survives a block boundary
        // in the 7-I straight-line shapes (push_closure and its value-send are in
        // the same block, adjacent or via a temp), so entry is all-`None`;
        // `translate_instr` maintains it in lockstep with `stack`.
        let mut stack_sites: Vec<Option<usize>> = vec![None; stack.len()];
        let mut pending_cmp: Option<PendingCmp> = None;
        // S11 step 7: a fallible smi-inline op mid-block (SmiArith/
        // SmiCmpVal, never a terminator) now needs a REAL fallback that
        // REJOINS this same block's own continuation, not a bailout-and-
        // restart -- so a single `cfg_block` can now emit SEVERAL `IrBlock`s
        // (this one, then one per split), not just one. `cur_id`/`cur_bci`
        // track whichever one is CURRENTLY accumulating into `code`;
        // finished ones are pushed to `ir_blocks` immediately, same as the
        // final one still is after this loop.
        let mut cur_id = block_id(b);
        let mut cur_bci = cfg_block.bci_start;
        let mut bci = cfg_block.bci_start;
        // S13 step 7b-ii: the bci of the block's LAST instruction. For a
        // `Branch` terminator that is the `br_true`/`br_false` opcode itself —
        // the reexecute resume point for a non-boolean `mustBeBoolean` deopt
        // (the interpreter rolls its own bci back to exactly this opcode).
        let mut last_instr_bci = cfg_block.bci_start;
        // S14 step 3: set true when a generic send in this block lowered to an
        // uncommon TRAP (an `Untaken` site). A trap is a terminator (control
        // leaves the method), so the rest of this CFG block's instructions AND
        // its terminator are dead — finish the block at the trap and move on.
        let mut trapped = false;
        while bci < cfg_block.bci_end {
            last_instr_bci = bci;
            let (instr, next) = decode_at(method, bci);
            // Fusable iff nothing but the block's own terminator follows
            // this instruction, AND that terminator is a Branch (D3.2 —
            // see `translate_instr`'s own doc for why position alone,
            // `next == cfg_block.bci_end`, isn't sufficient: it only
            // proves THIS instruction ends the block, not that a Branch
            // immediately follows it -- a block can hold several plain
            // instructions before its one terminator, e.g. `push_self;
            // push_temp; send; br_false_fwd` all in one block).
            let fusable = is_branch_terminator
                && next < cfg_block.bci_end
                && decode_at(method, next).1 == cfg_block.bci_end;
            let split = t.translate_instr(
                &instr,
                fusable,
                bci,
                &mut stack,
                &mut stack_sites,
                &mut code,
                &mut deopt,
                &mut pending_cmp,
                &mut trapped,
            );
            // S14 step 7-I: resync the block-site shadow to `stack`'s new length
            // after EVERY `translate_instr` (its many early `return split` paths
            // would skip an in-function resync). A site-carrying opcode kept the
            // two in lockstep itself (this is a no-op then); a plain opcode only
            // ever pushes/pops non-site values, so padding new slots with `None`
            // and truncating popped ones is exact — a real send's receiver/args
            // are never sites (a site receiver is a splice, handled earlier).
            stack_sites.resize(stack.len(), None);
            if trapped {
                // The `Ir::UncommonTrap` is already the last op in `code` (and
                // its deopt site is already in `deopt`). Stop translating this
                // block: everything after the trap is unreachable. `code`/
                // `deopt` are finished into `cur_id` by the trapped-block path
                // below (which skips the terminator entirely).
                break;
            }
            if let Some(continuation_id) = split {
                // S14 step 7-IV-a: a CFG-inlined send set `pending_jump_target`
                // to the callee's ENTRY block — the current block jumps THERE,
                // and control reaches `continuation_id` only via the inlined
                // callee's return edges. Unset (the ordinary smi-split case),
                // the jump goes straight to the continuation.
                let jump_target = t.pending_jump_target.take().unwrap_or(continuation_id);
                code.push(Ir::Jump {
                    target: jump_target,
                });
                t.finish_block(IrBlock {
                    id: cur_id,
                    bci: cur_bci,
                    code: std::mem::take(&mut code),
                    entry_stack: if cur_id == block_id(b) {
                        entry_stack.clone()
                    } else {
                        Vec::new() // a split-off continuation's own entry_stack is never consulted (only real merge points are)
                    },
                    deopt_sites: std::mem::take(&mut deopt),
                });
                cur_id = continuation_id;
                cur_bci = next;
            }
            bci = next;
        }
        // S14 step 7-I: a phantom block-site must never survive to a block
        // boundary — the escape pre-pass proved every closure is consumed by a
        // value-send in the straight-line shape 7-I splices, so `stack_sites` is
        // all-`None` here. If one leaked, `emit_merges`/terminator handling would
        // emit an `Ir::Move` of a never-defined phantom vreg. Asserted so a
        // future out-of-scope closure shape is caught in testing, not silently
        // miscompiled.
        debug_assert!(
            stack_sites.iter().all(|s| s.is_none()),
            "S14 7-I: a block-site survived to a block boundary (out-of-scope closure shape)"
        );
        let local_exit = stack;

        // S14 step 3: a trapped block ends at its `Ir::UncommonTrap` (already
        // the last op in `code`, its deopt site already in `deopt`) — control
        // leaves the method, so there is NO terminator to emit and NO merge to
        // propagate. Finish the current accumulating block here and move on.
        // `exit_stacks[b]` is still recorded (with the pre-trap operand stack)
        // so any successor whose `entry_stack_sources` said `Inherit(b)` finds
        // a well-formed stack; that successor is now unreachable from `b` (the
        // trap doesn't branch to it) and, absent another predecessor, becomes
        // dead code that `regalloc`'s reachability DFS simply never orders into
        // the live path (its own `unreachable`-block handling, already exercised
        // by decode's post-return dead blocks).
        if trapped {
            exit_stacks[b] = Some(local_exit);
            t.finish_block(IrBlock {
                id: cur_id,
                bci: cur_bci,
                code,
                entry_stack: if cur_id == block_id(b) {
                    entry_stack
                } else {
                    Vec::new()
                },
                deopt_sites: deopt,
            });
            continue;
        }

        // S14 step 3: this block is live (reachable) and did NOT trap (a trapped
        // block `continue`d above), so its terminator's successors are live too
        // — mark them reachable BEFORE the terminator match feeds their merges.
        // A `Return` has no successor.
        let mut had_unfused_branch = false;
        match cfg_block.terminator {
            Terminator::Return => {}
            Terminator::Fallthrough(t) | Terminator::Jump { target: t, .. } => reachable[t] = true,
            Terminator::Branch { if_true, if_false } => {
                reachable[if_true] = true;
                reachable[if_false] = true;
            }
        }

        match cfg_block.terminator {
            Terminator::Return => {}
            Terminator::Fallthrough(target) => {
                emit_merges(&sources, &entry_stacks, target, &local_exit, &mut code);
                code.push(Ir::Jump {
                    target: block_id(target),
                });
            }
            Terminator::Jump {
                target,
                is_backward,
            } => {
                if is_backward {
                    // S13 step 10b: a backward-jump `Poll` is a deopt SAFEPOINT
                    // (mirroring the UncommonTrap fail blocks) — if this loop's
                    // own nmethod is made `NotEntrant` mid-loop, only the poll
                    // can deopt it (a call-free loop never returns into a
                    // redirected slot). Record ONE deopt site at the poll:
                    //   - `kind = LoopPoll`, `reexecute = true`;
                    //   - `stack` = `local_exit`, the pre-merge exit operand
                    //     stack live AT the poll (merges are just vreg renames,
                    //     so pre-merge vregs resolve to the same VALUES);
                    //   - `bci` = the LOOP-HEADER bci = the backward-jump
                    //     `target` block's `bci_start`: the interpreter resumes
                    //     there and RE-EXECUTES the loop condition.
                    // The poll's `SafepointPc` is emitted at the `bl stub_poll`
                    // RETURN address (`emit::emit_poll`), so this site keys on
                    // that return offset via the shared position numbering (like
                    // a CallSend), and `driver::build_deopt_metadata` correlates
                    // it exactly like any other `deopt_sites` entry — no
                    // kind-specific handling there. `regalloc::deopt_live` forces
                    // receiver + slots + `local_exit` live-across the poll so
                    // spill-all pins them to frame slots (never Nil).
                    let poll_ci = code.len() as u32;
                    code.push(Ir::Poll);
                    deopt.push((
                        poll_ci,
                        DeoptRaw {
                            stack: local_exit.clone(),
                            bci: cfg.blocks[target].bci_start,
                            kind: SafepointKind::LoopPoll,
                            reexecute: true,
                            stack_closures: Vec::new(),
                            inline: None,
                        },
                    ));
                }
                emit_merges(&sources, &entry_stacks, target, &local_exit, &mut code);
                code.push(Ir::Jump {
                    target: block_id(target),
                });
            }
            Terminator::Branch { if_true, if_false } => {
                // The stack the SUCCESSORS see: a FUSED comparison never
                // pushed its condition (pending_cmp — local_exit is already
                // successor-shaped), but a NON-fused BoolBr consumes the
                // condition it finds on top, so the merges (and the Inherit
                // record below) must see local_exit WITH THE CONDITION
                // POPPED. Feeding the full stack left every successor one
                // deep — latent since 7b-ii shipped BoolBr, hidden because a
                // cold condition send cut the block at a trap and a warm one
                // was, until S15's deeper dissolved-loop nesting, always
                // smi-fusable (found by the sieve warm compile's merge-depth
                // assert).
                let succ_exit: Vec<VReg> = if pending_cmp.is_some() {
                    local_exit.clone()
                } else {
                    let mut v = local_exit.clone();
                    v.pop()
                        .expect("branch terminator with an empty simulated stack (compiler bug)");
                    v
                };
                emit_merges(&sources, &entry_stacks, if_true, &succ_exit, &mut code);
                emit_merges(&sources, &entry_stacks, if_false, &succ_exit, &mut code);
                if let Some(pc) = pending_cmp.take() {
                    match pc {
                        // S13 step 7b: `deopt` was captured at the fused
                        // comparison's own smi-inline site (before its operand
                        // pops) — `deopt.stack` carries `a`/`b`, and
                        // `deopt.bci` is the comparison SEND's own bci (NOT
                        // `cfg_block.bci_start`, which is the block's first
                        // instruction) — the reexecute resume point.
                        PendingCmp::Smi { op, a, b: b_, deopt } => {
                            let fail_id = t.fail_and_branch(deopt.bci, deopt.stack);
                            code.push(Ir::SmiCmpBr {
                                op,
                                a,
                                b: b_,
                                if_true: block_id(if_true),
                                if_false: block_id(if_false),
                                fail: fail_id,
                            });
                        }
                        // No fail edge and nothing to carry: the operands were
                        // already guard-unboxed at the send (`FUnbox`'s own
                        // `fail` owns the wrong-klass trap), so `fcmp` is
                        // infallible — see `PendingCmp`.
                        PendingCmp::Float { op, a, b: b_ } => {
                            code.push(Ir::FCmpBr {
                                op,
                                a,
                                b: b_,
                                if_true: block_id(if_true),
                                if_false: block_id(if_false),
                            });
                        }
                    }
                } else {
                    had_unfused_branch = true;
                    let val = *local_exit
                        .last()
                        .expect("branch terminator with an empty simulated stack (compiler bug)");
                    // S13 step 7b-ii: the not_bool edge deopts at the branch
                    // opcode. `local_exit` is the operand stack BEFORE the
                    // branch (the deferred `br_true`/`br_false` never popped
                    // `val`), so it IS the reexecute stack.
                    let not_bool_id = t.fresh_not_bool_block(last_instr_bci, local_exit.to_vec());
                    code.push(Ir::BoolBr {
                        val,
                        if_true: block_id(if_true),
                        if_false: block_id(if_false),
                        not_bool: not_bool_id,
                    });
                }
            }
        }

        // Inherit-successors must ALSO see the branch condition popped —
        // record the successor-shaped stack for Branch terminators (identity
        // for every other terminator).
        exit_stacks[b] = Some(match cfg_block.terminator {
            Terminator::Branch { .. } => {
                let mut v = local_exit.clone();
                if had_unfused_branch {
                    v.pop();
                }
                v
            }
            _ => local_exit,
        });
        t.finish_block(IrBlock {
            id: cur_id,
            bci: cur_bci,
            code,
            entry_stack: if cur_id == block_id(b) {
                entry_stack
            } else {
                Vec::new()
            },
            deopt_sites: deopt,
        });
    }

    // Indexed by id (`finish_block`'s own doc), never append order --
    // required so `emit.rs`'s own `method.blocks[bid.0 as usize]` (and its
    // parallel per-index `labels` vec) actually find the right block.
    let ir_blocks: Vec<IrBlock> = t
        .blocks_by_id
        .into_iter()
        .enumerate()
        .map(|(i, slot)| {
            slot.unwrap_or_else(|| {
                panic!("block {i} was allocated an id but never finished (compiler bug)")
            })
        })
        .collect();

    // S13: if this method has any deopt site, intern its own compiled
    // `MethodOop` into the pool (as an `Oop` reloc, so `oops_do`/GC keeps the
    // pool word current) so the deopt materializer can name the method for a
    // reconstructed frame's `FRAME_METHOD` slot. Interned LAST (highest pool
    // index) — existing `PoolLit` references keep their indices — and only
    // when needed, so an all-smi-inline method's pool (and its listing golden)
    // is untouched.
    let method_pool_ix = if ir_blocks.iter().any(|b| !b.deopt_sites.is_empty()) {
        Some(t.pool.intern(method.oop().raw(), Some(RelocKind::Oop)).0)
    } else {
        None
    };

    let mut irm = IrMethod {
        osr_cold_sends: t.osr_cold_sends,
        is_osr: osr,
        blocks: ir_blocks,
        vregs: t.vregs,
        pool: t.pool.entries,
        argc: method.argc() as u8,
        ntemps: method.ntemps() as u8,
        ctx_vregs,
        method_ctx_vreg: method_ctx_vreg.map(|v| (v, method.nctx())),
        spliced_nlr: t.spliced_nlr,
        spliced_multibb: t.spliced_multibb,
        splice_declined_budget: t.splice_declined_budget,
        block_closure_vreg,
        safepoints: Vec::new(),
        true_lit,
        false_lit,
        nil_lit,
        mark_slots_lit,
        mark_double_lit,
        double_klass_lit,
        float64x2_klass_lit,
        float32x4_klass_lit,
        int32x4_klass_lit,
        call_sites: t.call_sites,
        site_feedback: t.site_feedback,
        inline_deps: t.inline_deps,
        self_devirt: t.self_devirt,
        method_pool_ix,
    };
    copy_propagate(&mut irm);
    reduce_float_boxes(&mut irm, osr_bci);
    irm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeBuilder;
    use crate::compiler::decode;
    use crate::interpreter::ic::InterpreterIc;
    use crate::runtime::{JitMode, VmOptions, VmState};

    fn test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: JitMode::Off,
        })
    }

    /// A throwaway method standing in for a real SmallInteger primitive —
    /// `classify_smi_send` only ever reads its `primitive()` field, never
    /// executes its bytecode.
    fn primitive_stub(
        vm: &mut VmState,
        sel: crate::oops::wrappers::SymbolOop,
        prim_id: i64,
    ) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let m = b.finish(vm, sel, 1, 0);
        m.set_primitive(prim_id);
        m
    }

    /// S14 step 2/3: `convert` annotates each real (non-inlined, non-trapped)
    /// `call_sites` entry with its interpreter feedback, same index. A mono send
    /// to a NON-smi klass stays a real `CallSend` and carries `Mono` feedback;
    /// a fresh (empty) IC is `Untaken` and — S14 step 3 — lowers to an uncommon
    /// TRAP instead of a `CallSend`, so it produces NO `call_sites`/
    /// `site_feedback` entry but DOES record an `UncommonTrap` deopt site.
    #[test]
    fn convert_annotates_site_feedback() {
        use crate::compiler::feedback::SiteFeedback;
        let mut vm = test_vm();
        let foo_sel = vm.universe.intern(b"foo");
        // S14 step 4c: a target that is NOT inline-eligible (its body has a
        // SUPER send, which the inliner gates out) so a warm mono site to it
        // stays a real `CallSend` instead of being spliced inline. A leaf
        // accessor inlines (4b), a plain non-leaf now inlines too (4c); a
        // super-send callee is the shape that still dispatches.
        let inner_sel = vm.universe.intern(b"inner");
        let target = {
            let mut b = BytecodeBuilder::new();
            b.push_self();
            b.send_super(&mut vm, inner_sel, 0);
            b.ret_tos();
            b.finish(&mut vm, foo_sel, 0, 0)
        };

        // `m [ ^self foo ]` — one real send site.
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.send(&mut vm, foo_sel, 0);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 0);

        // Fresh IC (never dispatched) → Untaken → an uncommon trap: no
        // call_site, no site_feedback entry, but ONE UncommonTrap deopt site.
        {
            let cfg = decode::decode(method);
            let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);
            assert_eq!(
                ir.site_feedback.len(),
                ir.call_sites.len(),
                "one site_feedback per real call site (both empty for a trapped-only method)"
            );
            assert_eq!(
                ir.call_sites.len(),
                0,
                "an Untaken send lowers to a trap, not a CallSend -> no call_site"
            );
            let trap_sites: usize = ir
                .blocks
                .iter()
                .flat_map(|blk| blk.deopt_sites.iter())
                .filter(|(_, raw)| matches!(raw.kind, SafepointKind::UncommonTrap))
                .count();
            assert_eq!(
                trap_sites, 1,
                "the one Untaken `foo` send became exactly one uncommon-trap deopt site"
            );
        }

        // Warm the IC to mono on a NON-smi klass → Mono feedback → real CallSend.
        let klass = vm.universe.object_klass;
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, klass, target, epoch);
        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);
        assert_eq!(
            ir.call_sites.len(),
            1,
            "a warm mono send is a real CallSend"
        );
        match &ir.site_feedback[0] {
            SiteFeedback::Mono {
                klass: k,
                method: m,
            } => {
                assert_eq!(k.oop().raw(), klass.oop().raw());
                assert_eq!(m.oop().raw(), target.oop().raw());
            }
            other => panic!("expected Mono, got {other:?}"),
        }
    }

    /// S14 step 4b: a warm mono send to a cheap LEAF accessor (`^instvar0`)
    /// splices inline — the IR gains an `Ir::GuardKlass` + a `LoadField` off the
    /// receiver, NO `Ir::CallSend` for the site, and the method records one
    /// `inline_deps` pair `(receiver_klass, selector)`. The cold trap block
    /// carries a reexecute `UncommonTrap` at the send's bci.
    #[test]
    fn convert_inlines_leaf_accessor() {
        let mut vm = test_vm();
        let recv_klass = vm.universe.new_klass(
            vm.universe.object_klass,
            "S14InlineRecv",
            crate::oops::Format::Slots,
            false,
            crate::oops::layout::HEADER_WORDS + 1,
        );
        // `val [ ^instvar0 ]` — a leaf accessor.
        let val_sel = vm.universe.intern(b"val");
        let val = {
            let mut vb = BytecodeBuilder::new();
            vb.push_instvar(0);
            vb.ret_tos();
            vb.finish(&mut vm, val_sel, 0, 0)
        };

        // `m [ ^self val ]`.
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.send(&mut vm, val_sel, 0);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, m_sel, 0, 0);

        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, recv_klass, val, epoch);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        // No CallSend for the inlined site.
        assert_eq!(ir.call_sites.len(), 0, "inlined leaf → no CallSend");
        assert_eq!(ir.site_feedback.len(), 0);

        // Exactly one GuardKlass, one instvar LoadField off the receiver, and no
        // CallSend anywhere.
        let (mut guards, mut loads, mut calls) = (0, 0, 0);
        for blk in &ir.blocks {
            for op in &blk.code {
                match op {
                    Ir::GuardKlass { obj, kind, .. } => {
                        guards += 1;
                        assert_eq!(*obj, VReg(0), "guard tests the receiver (self)");
                        assert_eq!(*kind, GuardShape::KlassTest, "non-smi klass → KlassTest");
                    }
                    Ir::LoadField { obj, byte_off, .. } => {
                        // The accessor's instvar0 read off the receiver.
                        if *obj == VReg(0) && *byte_off == BODY_OFFSET as i32 {
                            loads += 1;
                        }
                    }
                    Ir::CallSend { .. } => calls += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(guards, 1, "one receiver-klass guard");
        assert_eq!(loads, 1, "one inlined instvar0 load off the receiver");
        assert_eq!(calls, 0, "no CallSend — the send was inlined");

        // One inline dependency: (recv_klass, val).
        assert_eq!(ir.inline_deps.len(), 1);
        assert_eq!(ir.inline_deps[0].0.oop().raw(), recv_klass.oop().raw());
        assert_eq!(ir.inline_deps[0].1.oop().raw(), val_sel.oop().raw());

        // A cold reexecute trap at the send's bci.
        let trap_sites: usize = ir
            .blocks
            .iter()
            .flat_map(|blk| blk.deopt_sites.iter())
            .filter(|(_, raw)| matches!(raw.kind, SafepointKind::UncommonTrap) && raw.reexecute)
            .count();
        assert_eq!(trap_sites, 1, "the guard's cold path is one reexecute trap");
    }

    /// S14 step 4b: `^self` (a bare return-self leaf) inlines — the result is
    /// the receiver vreg itself (no field load), behind a klass guard, and a
    /// mono SMI receiver picks the `SmiTest` guard shape.
    #[test]
    fn convert_inlines_self_and_arg_smi_guard() {
        let mut vm = test_vm();
        // `id [ ^self ]` on SmallInteger (smi klass → SmiTest guard).
        let id_sel = vm.universe.intern(b"idSelf");
        let id_method = {
            let mut vb = BytecodeBuilder::new();
            vb.ret_self();
            vb.finish(&mut vm, id_sel, 0, 0)
        };
        // `m [ ^self idSelf ]`.
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.send(&mut vm, id_sel, 0);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"mSelf");
        let method = b.finish(&mut vm, m_sel, 0, 0);
        let smi_klass = vm.universe.smi_klass;
        let epoch = vm.ic_epoch;
        InterpreterIc::at(method, 0).set_mono(&mut vm, smi_klass, id_method, epoch);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);
        assert_eq!(ir.call_sites.len(), 0, "^self inlined, no CallSend");
        let smi_guards = ir
            .blocks
            .iter()
            .flat_map(|b| b.code.iter())
            .filter(|op| {
                matches!(
                    op,
                    Ir::GuardKlass {
                        kind: GuardShape::SmiTest,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(smi_guards, 1, "a smi receiver klass → one SmiTest guard");
        assert_eq!(ir.inline_deps.len(), 1);
        assert_eq!(ir.inline_deps[0].1.oop().raw(), id_sel.oop().raw());
    }

    /// `send site with mono smi IC + prim '+'` (D1 item 2): a monomorphic,
    /// smi-guarded send whose target's primitive is `+` (id 1) translates
    /// to `SmiArith{Add}`. S13 step 7b: `fail` is now a single-op
    /// `Ir::UncommonTrap` deopt block (a `brk` re-executes the send in the
    /// interpreter — its LargeInteger/Double fallback), carrying ONE
    /// reexecute deopt site whose recorded stack holds the send's own inputs
    /// (`a`, `b`). No `CallSend`, no jump, no `Ir::Bailout`.
    #[test]
    fn smi_send_inlined_mono() {
        let mut vm = test_vm();
        let plus_sel = vm.universe.intern(b"+");
        let plus_method = primitive_stub(&mut vm, plus_sel, 1);

        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, plus_sel, 1);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 1, 0);

        let ic = InterpreterIc::at(method, 0);
        let smi_klass = vm.universe.smi_klass;
        let epoch = vm.ic_epoch;
        ic.set_mono(&mut vm, smi_klass, plus_method, epoch);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        let (fail, a, b_) = ir.blocks[0]
            .code
            .iter()
            .find_map(|op| match op {
                Ir::SmiArith {
                    op: SmiOp::Add,
                    fail,
                    a,
                    b,
                    ..
                } => Some((*fail, *a, *b)),
                _ => None,
            })
            .expect("send site must translate to SmiArith{Add}");
        let fail_block = ir
            .blocks
            .iter()
            .find(|b| b.id == fail)
            .expect("fail must name a real synthesized block");
        // S13 step 7b: the fail block is a single-op deopt trap.
        assert!(
            matches!(fail_block.code[..], [Ir::UncommonTrap { .. }]),
            "the fail block must be a single Ir::UncommonTrap, got {:?}",
            fail_block.code
        );
        // ONE reexecute deopt site, keyed at the trap op (code index 0), whose
        // recorded stack holds the send's own inputs `a`,`b` (the reexecute
        // stack `[below.., a, b]`; here `below` is empty).
        assert_eq!(fail_block.deopt_sites.len(), 1);
        let (ci, raw) = &fail_block.deopt_sites[0];
        assert_eq!(*ci, 0);
        assert!(raw.reexecute);
        assert_eq!(raw.kind, SafepointKind::UncommonTrap);
        // Post copy-propagation the op's own operands may be rewritten to
        // their copy SOURCES while the fail block's recorded stack (a separate
        // IrBlock — outside the per-block window) keeps the original copy
        // vregs; the copies survive (cross-block use blocks deletion), so both
        // name the same VALUES. Assert the shape, not vreg identity.
        assert_eq!(
            raw.stack.len(),
            2,
            "reexecute stack carries the send's two inputs"
        );
        let _ = (a, b_);
        // No CallSend/Bailout anywhere in the smi fail path anymore.
        assert!(
            !ir.blocks
                .iter()
                .any(|b| b.code.iter().any(|op| matches!(op, Ir::Bailout { .. }))),
            "Ir::Bailout must never be constructed by convert()"
        );
        assert!(
            !fail_block
                .code
                .iter()
                .any(|op| matches!(op, Ir::CallSend { .. })),
            "S13 step 7b: the smi fail block no longer emits a CallSend fallback"
        );
    }

    /// The fusion peephole (D3.2): a comparison send immediately consumed
    /// by `br_false_fwd` becomes a single `SmiCmpBr` — no intermediate
    /// `SmiCmpVal` materialization, no separate `BoolBr`.
    #[test]
    fn smi_cmp_fused_with_branch() {
        let mut vm = test_vm();
        let lt_sel = vm.universe.intern(b"<");
        let lt_method = primitive_stub(&mut vm, lt_sel, 10);

        let mut b = BytecodeBuilder::new();
        let l1 = b.new_label();
        let l2 = b.new_label();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, lt_sel, 1);
        b.br_false_fwd(l1);
        b.push_smi_i8(1);
        b.jump_fwd(l2);
        b.bind(l1);
        b.push_smi_i8(0);
        b.bind(l2);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 1, 0);

        let ic = InterpreterIc::at(method, 0);
        let smi_klass = vm.universe.smi_klass;
        let epoch = vm.ic_epoch;
        ic.set_mono(&mut vm, smi_klass, lt_method, epoch);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        let block0 = &ir.blocks[0];
        assert!(
            block0
                .code
                .iter()
                .any(|op| matches!(op, Ir::SmiCmpBr { op: CmpOp::Lt, .. })),
            "must fuse into a single SmiCmpBr"
        );
        assert!(
            !block0
                .code
                .iter()
                .any(|op| matches!(op, Ir::SmiCmpVal { .. } | Ir::BoolBr { .. })),
            "fusion must skip both the materialize step and the separate branch"
        );
    }

    /// The float twin of `smi_cmp_fused_with_branch`: a mono-Double `<`
    /// immediately consumed by `br_false_fwd` becomes a single `FCmpBr`.
    ///
    /// This is the test whose absence WAS the bug. `Ir::FCmpBr` and
    /// `emit::emit_fcmp_br` both shipped complete with the float fast path, but
    /// nothing ever CONSTRUCTED an `FCmpBr` — only the smi arm read `fusable`,
    /// so the op sat unreachable and every Double compare paid a ~9-instruction
    /// Boolean round-trip: materialize `true`/`false` as an OBJECT
    /// (`ldr =true`/`ldr =false`/`csel`), then compare that object back against
    /// the true oop (`ldr =true`/`cmp`/`b.eq`) to branch on a condition `fcmp`
    /// had already put in the flags. Found by disassembling Mandelbrot's inner
    /// loop (`docs/mandelbrot_walkthrough.md` §5).
    #[test]
    fn double_cmp_fused_with_branch() {
        let mut vm = test_vm();
        let lt_sel = vm.universe.intern(b"<");
        // `is_double_inlinable`/`classify_double_send` resolve through
        // (guard klass, selector) rather than the IC's target slot, so unlike
        // the smi twin this needs Double to REALLY understand `<` — as
        // primitive 104, DOUBLE_INLINE's `Cmp(Lt)`.
        let lt_method = primitive_stub(&mut vm, lt_sel, 104);
        let double_klass = vm.universe.double_klass;
        crate::runtime::lookup::install_method(&mut vm, double_klass, lt_sel, lt_method);

        let mut b = BytecodeBuilder::new();
        let l1 = b.new_label();
        let l2 = b.new_label();
        b.push_self();
        b.push_temp(0);
        b.send(&mut vm, lt_sel, 1);
        b.br_false_fwd(l1);
        b.push_smi_i8(1);
        b.jump_fwd(l2);
        b.bind(l1);
        b.push_smi_i8(0);
        b.bind(l2);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 1, 0);

        let ic = InterpreterIc::at(method, 0);
        let epoch = vm.ic_epoch;
        ic.set_mono(&mut vm, double_klass, lt_method, epoch);

        let cfg = decode::decode(method);
        let ir = convert(&vm, double_klass, method, &cfg, None);

        // Scan every block, not just `blocks[0]`: the double fuse finishes its
        // guard-fail trap block DURING the send's own translation, so the
        // straight-line block is not necessarily first.
        assert!(
            ir.blocks
                .iter()
                .any(|blk| blk
                    .code
                    .iter()
                    .any(|op| matches!(op, Ir::FCmpBr { op: CmpOp::Lt, .. }))),
            "must fuse into a single FCmpBr"
        );
        assert!(
            !ir.blocks.iter().any(|blk| blk
                .code
                .iter()
                .any(|op| matches!(op, Ir::FCmpVal { .. } | Ir::BoolBr { .. }))),
            "fusion must skip both the materialize step and the separate branch"
        );
    }

    /// D3.2's merge rule: the `ifTrue:ifFalse:` value pattern's join block
    /// has entry_depth 1, owns one fresh merge vreg, and both predecessors
    /// end their code with a `Move` into it.
    #[test]
    fn stack_sim_depths_agree() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        let l1 = b.new_label();
        let l2 = b.new_label();
        b.push_true();
        b.br_false_fwd(l1);
        b.push_smi_i8(1);
        b.jump_fwd(l2);
        b.bind(l1);
        b.push_smi_i8(2);
        b.bind(l2);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 0);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        // Same block layout as decode::tests::leaders_if_else: 0=condition,
        // 1=true arm, 2=false arm, 3=merge.
        let merge = &ir.blocks[3];
        assert_eq!(
            merge.entry_stack.len(),
            1,
            "entry_depth 1 -> one merge vreg"
        );
        let merge_vreg = merge.entry_stack[0];

        for pred_idx in [1usize, 2] {
            let pred = &ir.blocks[pred_idx];
            let last_move = pred.code.iter().rev().find_map(|op| match op {
                Ir::Move { dst, src } => Some((*dst, *src)),
                _ => None,
            });
            assert_eq!(
                last_move.map(|(dst, _)| dst),
                Some(merge_vreg),
                "block {pred_idx} must Move its exit value into the merge vreg"
            );
        }
    }

    /// No gratuitous copies: a block with exactly one non-backward
    /// predecessor inherits its exit stack's vregs verbatim.
    #[test]
    fn single_pred_inherits_stack() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        let l = b.new_label();
        b.push_smi_i8(1);
        b.jump_fwd(l);
        b.bind(l);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 0);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        // 2 real blocks (push+jump, then the target) -- S11 step 7: no
        // extra synthetic block anymore unless one is actually NEEDED (a
        // fallible smi-inline op or a Boolean branch), and this method's
        // bytecode has neither.
        assert_eq!(ir.blocks.len(), 2);
        let const_dst = ir.blocks[0]
            .code
            .iter()
            .find_map(|op| match op {
                Ir::ConstSmi { dst, value: 1 } => Some(*dst),
                _ => None,
            })
            .expect("block 0 has a ConstSmi(1)");
        assert_eq!(ir.blocks[1].entry_stack, vec![const_dst]);
        assert!(
            !ir.blocks[0]
                .code
                .iter()
                .any(|op| matches!(op, Ir::Move { .. })),
            "a single non-backward predecessor must not insert a copy"
        );
    }

    /// SSA-lite's temp rule: `store_temp_pop` and a later block's
    /// `push_temp` both reference the SAME persistent per-method vreg for
    /// that slot (`push_temp` still makes its own defensive copy of it —
    /// D3.2's v1 rule — but the copy's *source* is the one shared vreg).
    #[test]
    fn temp_slots_single_vreg() {
        let mut vm = test_vm();
        let mut b = BytecodeBuilder::new();
        let l1 = b.new_label();
        let l2 = b.new_label();
        b.push_smi_i8(5);
        b.store_temp_pop(0);
        b.push_true();
        b.br_false_fwd(l1);
        b.push_temp(0);
        b.jump_fwd(l2);
        b.bind(l1);
        b.push_smi_i8(0);
        b.bind(l2);
        b.ret_tos();
        let sel = vm.universe.intern(b"m");
        let method = b.finish(&mut vm, sel, 0, 1);

        let cfg = decode::decode(method);
        let ir = convert(&vm, vm.universe.smi_klass, method, &cfg, None);

        let stored_vreg = ir.blocks[0]
            .code
            .iter()
            .find_map(|op| match op {
                Ir::Move { dst, .. } => Some(*dst),
                _ => None,
            })
            .expect("block 0 stores into temp 0 via a Move");
        let push_temp_src = ir.blocks[1]
            .code
            .iter()
            .find_map(|op| match op {
                Ir::Move { src, .. } => Some(*src),
                _ => None,
            })
            .expect("block 1 reads temp 0 via a Move");
        assert_eq!(push_temp_src, stored_vreg);
    }
}
