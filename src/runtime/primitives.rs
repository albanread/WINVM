//! The primitive mechanism (SPEC Â§10): pinned ids/semantics for the smi,
//! oops, bytes, and system (dev-hook) groups. Every `PrimFn` validates its
//! own receiver/arg tags and formats â€” a violation is always `PrimResult::Fail`
//! (bytecode-body fallback), never a Rust panic; `args[0]` is always the
//! receiver.
//!
//! Layer boundary (`sprint_s03_detail.md` Â§Layer boundaries): this module
//! reads/writes the operand stack only through `VmState::prim_arg`/the
//! `args` slice handed in by the interpreter â€” it never pushes a frame.

use crate::memory::alloc;
use crate::oops::klass::Format;
use crate::oops::layout::{HEADER_WORDS, SMI_MAX, SMI_MIN};
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{ArrayOop, ByteArrayOop, KlassOop, MemOop};
// WINVM: DoubleOop is consumed only by the macOS-gated cocoa_double_array.
#[cfg(target_os = "macos")]
use crate::oops::wrappers::DoubleOop;
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

use std::io::Write;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PrimResult {
    Ok(Oop),
    Fail,
    /// The primitive already replaced the sender's continuation â€” pushed a
    /// real frame and set `vm.regs` itself (SPEC Â§10, S4: the `value`
    /// family, `ensure:`, `ifCurtailed:`). The interpreter's primitive-call
    /// site must push NO result and just return to dispatch; pushing one
    /// would corrupt the new frame's temp area by one slot.
    Activated,
    /// S24 A1: a COMPILED block body invoked by `activate_block`'s
    /// enter-compiled fast path performed a non-local return, and
    /// `continue_unwind` has ALREADY run â€” this is its outcome, to be
    /// relayed exactly like `EnterResult::Nlr`/`SendOutcome::Nlr` (S11
    /// D6.3). Produced ONLY by `interpreter::blocks::activate_block`; the
    /// consumer must NOT touch the operand stack (`sp = base` would be
    /// wrong â€” after an NLR the stack belongs to a different activation).
    Nlr(crate::interpreter::unwind::UnwindStep),
}

pub type PrimFn = fn(vm: &mut VmState, args: &[Oop]) -> PrimResult;

pub struct PrimDesc {
    pub id: u16,
    pub name: &'static str,
    pub f: PrimFn,
    /// Argument count EXCLUDING the receiver.
    pub argc: u8,
    pub can_allocate: bool,
    pub can_fail: bool,
}

/// Binary-searched by [`prim_by_id`] â€” MUST stay sorted by `id` (the
/// `prim_table_sorted_unique` test enforces this).
pub static PRIMITIVES: &[PrimDesc] = &[
    PrimDesc {
        id: 1,
        name: "+",
        f: prim_add,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 2,
        name: "-",
        f: prim_sub,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 3,
        name: "*",
        f: prim_mul,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 4,
        name: "//",
        f: prim_div,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 5,
        name: "\\\\",
        f: prim_mod,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 6,
        name: "bitAnd:",
        f: prim_bit_and,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 7,
        name: "bitOr:",
        f: prim_bit_or,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 8,
        name: "bitXor:",
        f: prim_bit_xor,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 9,
        name: "bitShift:",
        f: prim_bit_shift,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 10,
        name: "<",
        f: prim_lt,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 11,
        name: "<=",
        f: prim_le,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 12,
        name: ">",
        f: prim_gt,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 13,
        name: ">=",
        f: prim_ge,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 14,
        name: "=",
        f: prim_eq,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 15,
        name: "~=",
        f: prim_ne,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 20,
        name: "identityHash",
        f: prim_identity_hash,
        argc: 0,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 21,
        name: "class",
        f: prim_class,
        argc: 0,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 22,
        name: "==",
        f: prim_identical,
        argc: 1,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 23,
        name: "basicNew",
        f: prim_basic_new,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 24,
        name: "basicNew:",
        f: prim_basic_new_colon,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 25,
        name: "instVarAt:",
        f: prim_inst_var_at,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 26,
        name: "at:",
        f: prim_at,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 27,
        name: "at:put:",
        f: prim_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 28,
        name: "size",
        f: prim_size,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 40,
        name: "byteAt:",
        f: prim_byte_at,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 41,
        name: "byteAt:put:",
        f: prim_byte_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 42,
        name: "byteSize",
        f: prim_byte_size,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 43,
        name: "replaceFrom:to:with:",
        f: prim_replace_from_to_with,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 44,
        name: "hashBytes",
        f: prim_hash_bytes,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 45,
        name: "compare:",
        f: prim_compare,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 50,
        name: "value",
        f: prim_value0,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 51,
        name: "value:",
        f: prim_value1,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 52,
        name: "value:value:",
        f: prim_value2,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 53,
        name: "value:value:value:",
        f: prim_value3,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 54,
        name: "valueWithArguments:",
        f: prim_value_with_arguments,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 60,
        name: "ensure:",
        f: prim_ensure,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 61,
        name: "ifCurtailed:",
        f: prim_if_curtailed,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    // R2 reflection (docs/APPS.md Â§3). Kept in id order â€” `prim_by_id`
    // binary-searches, so the table must stay sorted.
    PrimDesc {
        id: 62,
        name: "primitiveOf:selector:",
        f: prim_primitive_of,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 63,
        name: "methodSends:selector:target:",
        f: prim_method_sends,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 64,
        name: "perform:withArguments:",
        f: prim_perform_with_arguments,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 90,
        name: "quit",
        f: prim_quit,
        argc: 0,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 91,
        name: "printOnStdout:",
        f: prim_print_on_stdout,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 92,
        name: "millisecondClock",
        f: prim_millisecond_clock,
        argc: 0,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 93,
        name: "gcScavenge",
        f: prim_gc_scavenge,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
    PrimDesc {
        id: 94,
        name: "gcFull",
        f: prim_gc_full,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
    PrimDesc {
        id: 95,
        name: "error:",
        f: prim_error,
        argc: 1,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 96,
        name: "quit:",
        f: prim_quit_colon,
        argc: 1,
        can_allocate: false,
        can_fail: false,
    },
    PrimDesc {
        id: 97,
        name: "gcStats",
        f: prim_gc_stats,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
    PrimDesc {
        id: 98,
        name: "allClasses",
        f: prim_all_classes,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
    PrimDesc {
        id: 99,
        name: "selectorsOf:",
        f: prim_selectors_of,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // --- Double group (S6, SPEC Â§1.3) ------------------------------------
    PrimDesc {
        id: 100,
        name: "+",
        f: prim_double_add,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 101,
        name: "-",
        f: prim_double_sub,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 102,
        name: "*",
        f: prim_double_mul,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 103,
        name: "/",
        f: prim_double_div,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 104,
        name: "<",
        f: prim_double_lt,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 105,
        name: "=",
        f: prim_double_eq,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 106,
        name: "sqrt",
        f: prim_double_sqrt,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 107,
        name: "floor",
        f: prim_double_floor,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 108,
        name: "asDouble",
        f: prim_smi_as_double,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 109,
        name: "printDigits",
        f: prim_double_print_digits,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    // --- Symbol group (S6) -------------------------------------------------
    PrimDesc {
        id: 110,
        name: "asSymbol",
        f: prim_string_as_symbol,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    // --- S15 A8 dev hook ----------------------------------------------------
    PrimDesc {
        id: 111,
        name: "__vmStats",
        f: prim_vm_stats,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
    // --- Alien group (S20 step 5) --------------------------------------------
    // docs/FFI.md Â§4 "Representation": `Alien`'s own typed byte-level
    // accessors + constructors (`runtime::alien`'s own module doc has the
    // full design rationale â€” reused `Format::IndexableBytes` shape, one
    // named external-address field, direct-vs-indirect split). Every one of
    // these validates its own receiver/arg tags exactly like every other
    // group in this table; `can_allocate` is true only for `doubleAt:` (the
    // one allocating accessor â€” see `runtime::alien::prim_alien_double_at`'s
    // own doc comment) and the two constructors (`new:`/`forAddress:size:`,
    // which allocate the fresh Alien itself).
    PrimDesc {
        id: 112,
        name: "byteAt:",
        f: crate::runtime::alien::prim_alien_byte_at,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 113,
        name: "byteAt:put:",
        f: crate::runtime::alien::prim_alien_byte_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 114,
        name: "signedLongAt:",
        f: crate::runtime::alien::prim_alien_signed_long_at,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 115,
        name: "signedLongAt:put:",
        f: crate::runtime::alien::prim_alien_signed_long_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 116,
        name: "doubleAt:",
        f: crate::runtime::alien::prim_alien_double_at,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 117,
        name: "doubleAt:put:",
        f: crate::runtime::alien::prim_alien_double_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 118,
        name: "size",
        f: crate::runtime::alien::prim_alien_size,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 119,
        name: "new:",
        f: crate::runtime::alien::prim_alien_new,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 120,
        name: "forAddress:size:",
        f: crate::runtime::alien::prim_alien_for_address_size,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    // â”€â”€ libm transcendentals (float fast-path companions): unary Double
    //    receivers, exactly `sqrt`(106)'s shape â€” the shim makes them
    //    compiled-callable, and the surrounding arithmetic fuses to native
    //    FP, so a plotted curve is one libm call per point plus register
    //    maths. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    PrimDesc {
        id: 121,
        name: "sin",
        f: prim_double_sin,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 122,
        name: "cos",
        f: prim_double_cos,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 123,
        name: "tan",
        f: prim_double_tan,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 124,
        name: "exp",
        f: prim_double_exp,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 125,
        name: "ln",
        f: prim_double_ln,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 126,
        name: "atan",
        f: prim_double_atan,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    // â”€â”€ SIMD Float64x2 (docs/SIMD.md): the interpreter baseline for the
    //    2-lane f64 vector value class â€” the analog of Double's own
    //    primitives, which the JIT vector fast-path will later fuse to NEON.
    //    Lane math is done here in scalar f64 (bit-identical to per-lane
    //    Double arithmetic, which the fast-path must also match). â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    PrimDesc {
        id: 127,
        name: "x:y:",
        f: prim_x2_xy,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 128,
        name: "splat:",
        f: prim_x2_splat,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 129,
        name: "+",
        f: prim_x2_add,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 130,
        name: "-",
        f: prim_x2_sub,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 131,
        name: "*",
        f: prim_x2_mul,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 132,
        name: "/",
        f: prim_x2_div,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 133,
        name: "at:",
        f: prim_x2_at,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // SIMD Float32x4 (docs/SIMD.md) â€” the 4-lane f32 companion to Float64x2.
    // `x:y:z:w:` is the only 4-arg constructor; the rest mirror Float64x2's
    // elementwise ops, guarded on the Float32x4 klass (the fast-path fuse's
    // is_float32x4_inlinable maps 136-139 â†’ `.4s` NEON arithmetic).
    PrimDesc {
        id: 134,
        name: "x:y:z:w:",
        f: prim_x4_xyzw,
        argc: 4,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 135,
        name: "splat:",
        f: prim_x4_splat,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 136,
        name: "+",
        f: prim_x4_add,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 137,
        name: "-",
        f: prim_x4_sub,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 138,
        name: "*",
        f: prim_x4_mul,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 139,
        name: "/",
        f: prim_x4_div,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 140,
        name: "at:",
        f: prim_x4_at,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // SIMD level 2: FloatArray (docs/SIMD.md Part E) â€” element access + the
    // explicit-NEON bulk kernels (+@ elementwise, sum/dot: fast reductions).
    PrimDesc {
        id: 141,
        name: "new:",
        f: prim_farray_new,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 142,
        name: "at:",
        f: prim_farray_at,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 143,
        name: "at:put:",
        f: prim_farray_at_put,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 144,
        name: "size",
        f: prim_farray_size,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 145,
        name: "+@",
        f: prim_farray_add,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 146,
        name: "sum",
        f: prim_farray_sum,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 147,
        name: "dot:",
        f: prim_farray_dot,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // SIMD Int32x4 (docs/SIMD.md) â€” 4-lane i32. `+ - *` fuse to NEON integer
    // `add/sub/mul v.4s` (is_int32x4 via the fuse's vec_arith_op, base 150);
    // there is NO vector integer divide, so no `/`.
    PrimDesc {
        id: 148,
        name: "x:y:z:w:",
        f: prim_i4_xyzw,
        argc: 4,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 149,
        name: "splat:",
        f: prim_i4_splat,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 150,
        name: "+",
        f: prim_i4_add,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 151,
        name: "-",
        f: prim_i4_sub,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 152,
        name: "*",
        f: prim_i4_mul,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 153,
        name: "at:",
        f: prim_i4_at,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // SIMD level 2: more FloatArray NEON kernels â€” scale: (elementwise Ã—scalar)
    // + max/min reductions (docs/SIMD.md Part E; max/min are order-independent
    // so bit-exact, unlike the FP sum).
    PrimDesc {
        id: 154,
        name: "scale:",
        f: prim_farray_scale,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 155,
        name: "max",
        f: prim_farray_max,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 156,
        name: "min",
        f: prim_farray_min,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    // --- variable reflection (docs/APPS.md Â§3, the variable half of R2) ---
    PrimDesc {
        id: 157,
        name: "instanceVariablesOf:",
        f: prim_instance_variables_of,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 158,
        name: "classVariablesOf:",
        f: prim_class_variables_of,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // --- game group (docs/gamepane_design.md): id block 200+ ---
    PrimDesc {
        id: 200,
        name: "GamePane>>clearR:g:b:",
        f: prim_game_clear,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 201,
        name: "GamePane>>paletteAt:r:g:b:",
        f: prim_game_palette,
        argc: 4,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 202,
        name: "GamePane>>cls:",
        f: prim_game_cls,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 203,
        name: "GamePane>>point:y:color:",
        f: prim_game_pset,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 204,
        name: "GamePane>>line:y:to:y:color:",
        f: prim_game_line,
        argc: 5,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 205,
        name: "GamePane>>fill:y:width:height:color:",
        f: prim_game_fillrect,
        argc: 5,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 206,
        name: "GamePane>>disc:y:radius:color:",
        f: prim_game_disc,
        argc: 4,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 207,
        name: "GamePane>>present",
        f: prim_game_present,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 208,
        name: "GamePane>>run",
        f: prim_game_run,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 209,
        name: "GamePane>>stop",
        f: prim_game_stop,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 210,
        name: "GamePane>>primDefineSprite:rows:",
        f: prim_game_define_sprite,
        argc: 2,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 211,
        name: "GamePane>>primSpriteColor:index:r:g:b:",
        f: prim_game_sprite_color,
        argc: 5,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 212,
        name: "GamePane>>primMoveSprite:x:y:",
        f: prim_game_move_sprite,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 213,
        name: "Sound>>primPlay:",
        f: prim_game_play_sound,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 214,
        name: "Tune>>primPlayTune:",
        f: prim_game_play_tune,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 215,
        name: "GamePane>>blit:",
        f: prim_game_blit,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    // Worker group (docs/multi-smalltalk-worker.md Â§5) â€” spawn/send/poll (M1)
    // + the MOP pickle pair (M0, provable solo).
    PrimDesc {
        id: 220,
        name: "Worker class>>primSpawn:",
        f: prim_worker_spawn,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 221,
        name: "Worker class>>primSend:corr:bytes:",
        f: prim_worker_send,
        argc: 3,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 222,
        name: "Worker class>>primPoll",
        f: prim_worker_poll,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 223,
        name: "Worker class>>primAwaitInbox:",
        f: prim_worker_await,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 224,
        name: "Worker class>>primTerminate:",
        f: prim_worker_terminate,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 225,
        name: "Worker class>>primAlive:",
        f: prim_worker_alive,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 226,
        name: "Worker class>>primSelfId",
        f: prim_worker_self_id,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 227,
        name: "Worker class>>pickle:",
        f: prim_mop_pickle,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 228,
        name: "Worker class>>unpickle:",
        f: prim_mop_unpickle,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // cocoa bridge C0 (docs/cocoa_bridge_design.md Â§8, prims 230-239).
    PrimDesc {
        id: 230,
        name: "Cocoa class>>primClassNamed:",
        f: prim_cocoa_class_named,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 231,
        name: "ObjcRef>>primSend:",
        f: prim_cocoa_send0,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 232,
        name: "ObjcRef>>primSend:with:",
        f: prim_cocoa_send1,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 233,
        name: "ObjcRef>>primSend:with:with:",
        f: prim_cocoa_send2,
        argc: 3,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 234,
        name: "ObjcRef>>primSendI64:",
        f: prim_cocoa_send_i64,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 235,
        name: "ObjcRef>>primSendString:",
        f: prim_cocoa_send_string,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 236,
        name: "ObjcRef>>primRelease",
        f: prim_cocoa_release,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 237,
        name: "Cocoa class>>primNSString:",
        f: prim_cocoa_nsstring,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 238,
        name: "Cocoa class>>primDrainPool",
        f: prim_cocoa_drain_pool,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 239,
        name: "ObjcRef>>primIsValid",
        f: prim_cocoa_is_valid,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 240,
        name: "ObjcRef>>primSend:args:ret:",
        f: prim_cocoa_send_general,
        argc: 3,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 241,
        name: "ObjcRef>>primSendAuto:args:",
        f: prim_cocoa_send_auto,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 242,
        name: "ObjcRef>>primSendMainAuto:args:",
        f: prim_cocoa_send_main_auto,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 243,
        name: "Cocoa class>>primNewAction:",
        f: prim_cocoa_new_action,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 244,
        name: "Cocoa class>>primPoolPush",
        f: prim_cocoa_pool_push,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 245,
        name: "Cocoa class>>primPoolPop",
        f: prim_cocoa_pool_pop,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    // --- reflection group ----------------------------------------------
    PrimDesc {
        id: 246,
        name: "respondsTo:",
        f: prim_responds_to,
        argc: 1,
        can_allocate: false,
        can_fail: true,
    },
    PrimDesc {
        id: 247,
        name: "shallowCopy",
        f: prim_shallow_copy,
        argc: 0,
        can_allocate: true,
        can_fail: true,
    },
    PrimDesc {
        id: 248,
        name: "numArgs",
        f: prim_block_num_args,
        argc: 0,
        can_allocate: false,
        can_fail: true,
    },
    // --- Cocoa bridge C6: reverse dispatch (delegates), CG3 --------------
    PrimDesc {
        id: 249,
        name: "MacvmDelegate class>>primNewDelegate:ticket:",
        f: prim_cocoa_new_delegate,
        argc: 2,
        can_allocate: true,
        can_fail: true,
    },
    // --- Cocoa GUI CG4: primary-side doit evaluation ---------------------
    PrimDesc {
        id: 250,
        name: "Worker class>>primEvalDoit:",
        f: prim_eval_doit,
        argc: 1,
        can_allocate: true,
        can_fail: true,
    },
    // --- DBG4: the programmer's `debugger;` â€” open the debugger here ------
    PrimDesc {
        id: 251,
        name: "halt",
        f: prim_halt,
        argc: 0,
        can_allocate: true,
        can_fail: false,
    },
];

pub fn prim_by_id(id: u16) -> Option<&'static PrimDesc> {
    PRIMITIVES
        .binary_search_by_key(&id, |d| d.id)
        .ok()
        .map(|i| &PRIMITIVES[i])
}

// --- game group (docs/gamepane_design.md, id block 200+) --------------------
//
// Each primitive validates its Smalltalk SmallInt arguments, then emits a
// `GameCommand` over `vm.game_sink` and returns the receiver (`self`). A bad
// argument fails the primitive so the Smalltalk fallback (`^self`) runs â€” no
// value ever reaches an `assert!`-panicking engine setter. A headless VM (no
// sink installed) silently drops the command.

use crate::embed::GameCommand;

/// A `SmallInt` argument as a `0..=255` colour/palette byte, or `None` (fail).
fn smi_byte(oop: Oop) -> Option<u8> {
    u8::try_from(SmallInt::try_from(oop)?.value()).ok()
}

/// A `SmallInt` argument as an `i64` coordinate, or `None` (fail).
fn smi_i64(oop: Oop) -> Option<i64> {
    Some(SmallInt::try_from(oop)?.value())
}

/// Emit `cmd` to the game pane if a sink is installed (else drop it).
fn game_emit(vm: &mut VmState, cmd: GameCommand) {
    if let Some(sink) = vm.game_sink.as_mut() {
        sink.emit(cmd);
    }
}

/// `clearR:g:b:` (200): clear to an opaque RGB colour and present.
fn prim_game_clear(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(r), Some(g), Some(b)) = (smi_byte(args[1]), smi_byte(args[2]), smi_byte(args[3]))
    else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::ClearTo { r, g, b });
    PrimResult::Ok(args[0])
}

/// `paletteAt:r:g:b:` (201): set palette entry `index` (16..=255) to RGB.
fn prim_game_palette(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(index), Some(r), Some(g), Some(b)) = (
        smi_byte(args[1]),
        smi_byte(args[2]),
        smi_byte(args[3]),
        smi_byte(args[4]),
    ) else {
        return PrimResult::Fail;
    };
    if index < 16 {
        return PrimResult::Fail; // set_rgb asserts index >= 16
    }
    game_emit(vm, GameCommand::PaletteAt { index, r, g, b });
    PrimResult::Ok(args[0])
}

/// `cls:` (202): clear the pane to palette `index` (0..=255).
fn prim_game_cls(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(index) = smi_byte(args[1]) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::Cls { index });
    PrimResult::Ok(args[0])
}

/// `point:y:color:` (203): plot a pixel.
fn prim_game_pset(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(x), Some(y), Some(index)) = (smi_i64(args[1]), smi_i64(args[2]), smi_byte(args[3]))
    else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::Pset { x, y, index });
    PrimResult::Ok(args[0])
}

/// `line:y:to:y:color:` (204): draw a line.
fn prim_game_line(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(x0), Some(y0), Some(x1), Some(y1), Some(index)) = (
        smi_i64(args[1]),
        smi_i64(args[2]),
        smi_i64(args[3]),
        smi_i64(args[4]),
        smi_byte(args[5]),
    ) else {
        return PrimResult::Fail;
    };
    game_emit(
        vm,
        GameCommand::Line {
            x0,
            y0,
            x1,
            y1,
            index,
        },
    );
    PrimResult::Ok(args[0])
}

/// `fill:y:width:height:color:` (205): fill a rectangle.
fn prim_game_fillrect(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(x), Some(y), Some(w), Some(h), Some(index)) = (
        smi_i64(args[1]),
        smi_i64(args[2]),
        smi_i64(args[3]),
        smi_i64(args[4]),
        smi_byte(args[5]),
    ) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::FillRect { x, y, w, h, index });
    PrimResult::Ok(args[0])
}

/// `disc:y:radius:color:` (206): fill a disc.
fn prim_game_disc(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(cx), Some(cy), Some(r), Some(index)) = (
        smi_i64(args[1]),
        smi_i64(args[2]),
        smi_i64(args[3]),
        smi_byte(args[4]),
    ) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::Disc { cx, cy, r, index });
    PrimResult::Ok(args[0])
}

/// `present` (207): upload the CPU buffer and show the frame.
fn prim_game_present(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    game_emit(vm, GameCommand::Present);
    PrimResult::Ok(args[0])
}

/// `run` (208): start the frame loop (the GUI timer begins pulling GameSteps).
fn prim_game_run(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    game_emit(vm, GameCommand::StartLoop);
    PrimResult::Ok(args[0])
}

/// `stop` (209): stop the frame loop.
fn prim_game_stop(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    game_emit(vm, GameCommand::StopLoop);
    PrimResult::Ok(args[0])
}

/// A String/ByteArray argument as a Rust `String`, or `None` (fail).
fn smi_str(oop: Oop) -> Option<String> {
    let b = ByteArrayOop::try_from(oop)?;
    let mut buf = Vec::new();
    b.copy_bytes_out(&mut buf);
    String::from_utf8(buf).ok()
}

/// `primDefineSprite:rows:` (210): define a sprite from hex-row art and place
/// it, keyed by the VM-minted `id`.
fn prim_game_define_sprite(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(id), Some(rows)) = (smi_i64(args[1]), smi_str(args[2])) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::DefineSprite { id, rows });
    PrimResult::Ok(args[0])
}

/// `primSpriteColor:index:r:g:b:` (211): set sprite `id`'s palette entry
/// `index` (0..=15) to an RGB colour.
fn prim_game_sprite_color(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(id), Some(index), Some(r), Some(g), Some(b)) = (
        smi_i64(args[1]),
        smi_byte(args[2]),
        smi_byte(args[3]),
        smi_byte(args[4]),
        smi_byte(args[5]),
    ) else {
        return PrimResult::Fail;
    };
    if index > 15 {
        return PrimResult::Fail; // sprite palettes hold 16 entries
    }
    game_emit(vm, GameCommand::SpriteColor { id, index, r, g, b });
    PrimResult::Ok(args[0])
}

/// `primMoveSprite:x:y:` (212): move sprite `id`'s instance to `(x, y)`.
fn prim_game_move_sprite(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (Some(id), Some(x), Some(y)) = (smi_i64(args[1]), smi_i64(args[2]), smi_i64(args[3]))
    else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::MoveSprite { id, x, y });
    PrimResult::Ok(args[0])
}

/// `primPlay:` (213): play SFX preset `n` (0..=9) on the shared engine.
fn prim_game_play_sound(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(preset) = smi_byte(args[1]) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::PlaySound { preset });
    PrimResult::Ok(args[0])
}

/// `primPlayTune:` (214): play an ABC-notation tune once in the background.
fn prim_game_play_tune(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(abc) = smi_str(args[1]) else {
        return PrimResult::Fail;
    };
    game_emit(vm, GameCommand::PlayTune { abc });
    PrimResult::Ok(args[0])
}

/// `blit:` (215): overwrite the active pane buffer from a `ByteArray` of
/// palette indices (row-major) in one command â€” the bulk path for a
/// CPU-generated frame. A non-`ByteArray` argument fails.
fn prim_game_blit(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(bytes) = ByteArrayOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let mut data = Vec::new();
    bytes.copy_bytes_out(&mut data);
    game_emit(vm, GameCommand::Blit { data });
    PrimResult::Ok(args[0])
}

// --- worker group: MOP pickle (docs/multi-smalltalk-worker.md Â§5, M0) --------
//
// The copy-passing boundary's serializer, exposed as its own primitives so
// the whole format is testable in ONE VM with zero threads. Ids 220â€“226 (the
// registry/spawn/send half) land with M1; only 227/228 exist yet.

/// `pickle:` (227): serialize an object graph to a MOP ByteArray. Fails
/// (never panics) on unpicklable kinds â€” blocks, contexts, methods, classes,
/// aliens â€” and on the size/depth guards; the world method's fallback body
/// turns that into a clean Smalltalk error.
fn prim_mop_pickle(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    // Pickling is read-only by construction (`mop::pickle` takes `&VmState`,
    // so its identity map's addresses stay stable); the result ByteArray is
    // allocated only afterwards.
    let bytes = match crate::runtime::mop::pickle(vm, args[1]) {
        Ok(b) => b,
        Err(_) => return PrimResult::Fail,
    };
    let ba = alloc::alloc_indexable_bytes(vm, vm.universe.bytearray_klass, bytes.len());
    for (i, b) in bytes.iter().enumerate() {
        ba.byte_at_put(i, *b);
    }
    PrimResult::Ok(ba.oop())
}

/// `unpickle:` (228): rebuild an object graph from a MOP ByteArray. The
/// bytes are copied out FIRST (unpickling allocates, so the source may move
/// mid-build); any malformed/truncated/unknown-class input fails cleanly.
fn prim_mop_unpickle(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(src) = ByteArrayOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    // Exactly a ByteArray â€” not any bytes-format object (a String or Alien
    // holding pickle bytes would be a type confusion worth failing loudly).
    let src_m = MemOop::try_from(args[1]).expect("ByteArrayOop implies a mem oop");
    if src_m.klass().oop().raw() != vm.universe.bytearray_klass.oop().raw() {
        return PrimResult::Fail;
    }
    let mut buf = Vec::new();
    src.copy_bytes_out(&mut buf);
    match crate::runtime::mop::unpickle(vm, &buf) {
        Ok(o) => PrimResult::Ok(o),
        Err(_) => PrimResult::Fail,
    }
}

/// A smi argument as a worker id (â‰¥ 0; 0 means the primary on the reply path).
fn smi_worker_id(oop: Oop) -> Option<u32> {
    let v = SmallInt::try_from(oop)?.value();
    u32::try_from(v).ok()
}

/// Build a fresh Smalltalk `String` oop from Rust bytes (the inverse of
/// [`string_arg`]) â€” used where a primitive answers a Rust-constructed message
/// (CG4's `primEvalDoit:` compile-error text).
fn string_oop(vm: &mut VmState, s: &str) -> Oop {
    let klass = vm.universe.string_klass;
    let b = alloc::alloc_indexable_bytes(vm, klass, s.len());
    for (i, byte) in s.bytes().enumerate() {
        b.byte_at_put(i, byte);
    }
    b.oop()
}

/// `primEvalDoit:` (250, Cocoa GUI CG4, `cocoa_gui_design.md` Â§7.3): evaluate a
/// doit source `String` on THIS (primary) VM through the existing
/// `execute_do_it` path and answer the RESULT oop. A Workspace `âŒ˜P`/`âŒ˜D` ships
/// its selection here as a `{#uiReq. corr. #doit. source}` request; the primary
/// evaluates it (the doit runs where the persistent objects live) and replies
/// the result.
///
/// Runs **inline under the caller's top-level recovery guard** â€” the dispatching
/// `exec("Worker dispatchInbox.")`. The doit is compiled to an anonymous `#doIt`
/// method (the `execute_do_it` shape) and run through
/// [`crate::interpreter::run_method_reentrant`] â€” the SAME nested-interpreter
/// entry `perform:withArguments:` (prim 64) uses. Plain `run_method`
/// (`execute_do_it`'s own path) is a *top-level* entry that does not
/// save/restore the caller's activation, so calling it from inside a running
/// method corrupts the caller's continuation â€” the reason this primitive cannot
/// simply reuse `execute_do_it`. No nested `sigsetjmp` is armed (which would
/// clobber the one-per-thread jmp slot). A compile/parse error answers a
/// `String` describing it, so the `#uiReply` carries the message instead of
/// raising; a runtime raise unwinds to the caller's guard as any `perform` does
/// (v1: the dispatch aborts and no reply is sent â€” the RPC path's documented
/// property, `47_worker.mst`).
///
/// NON-shimmable (`compiler::driver::PRIM_REENTERS_INTERPRETER`, with
/// `perform:withArguments:`): a compiled caller must reach this through the
/// c2i adapter â€” whose `rt_interpret_call` brackets the crossing with
/// `TierLink::IntoInterpreter` â€” never through `rt_call_primitive`, which
/// brackets nothing. Shimmed, the first GC inside a nested doit walked the
/// tier journal mispaired and aborted the process (the Cocoa `Worker uiDoit:`
/// relay crash after ~10 warmed dispatches); see the driver list's own doc
/// for why the bracket cannot live here in the `PrimFn` body instead.
fn prim_eval_doit(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(src) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    let item = match crate::frontend::parser::parse_one_top_item(&src) {
        Ok(Some(item)) => item,
        Ok(None) => return PrimResult::Ok(vm.universe.nil_obj),
        Err(e) => return PrimResult::Ok(string_oop(vm, &format!("parse error: {e}"))),
    };
    match item {
        crate::frontend::ast::TopItem::DoIt(stmt) => {
            let holder = vm.universe.undefined_object_klass;
            let m = match crate::frontend::codegen::compile_doit(vm, holder, stmt) {
                Ok(m) => m,
                Err(e) => return PrimResult::Ok(string_oop(vm, &format!("compile error: {e}"))),
            };
            // Nested interpreter entry (activation saved/restored), receiver nil
            // â€” the `#doIt` convention. No allocation between compile and run,
            // so `m` needs no extra rooting.
            let recv = vm.universe.nil_obj;
            let result = crate::interpreter::run_method_reentrant(vm, m, recv, &[]);
            PrimResult::Ok(result)
        }
        // A class definition from the Workspace: install it (no nested doit run
        // â€” `install_class_def` only compiles methods), answer nil.
        other => match crate::frontend::classdef::execute_top_item(vm, other) {
            Ok(_) => PrimResult::Ok(vm.universe.nil_obj),
            Err(e) => PrimResult::Ok(string_oop(vm, &format!("compile error: {e}"))),
        },
    }
}

/// Read a String argument's UTF-8 content; `None` for anything else.
fn string_arg(vm: &VmState, oop: Oop) -> Option<String> {
    let m = MemOop::try_from(oop)?;
    if m.klass().oop().raw() != vm.universe.string_klass.oop().raw() {
        return None;
    }
    let b = ByteArrayOop::try_from(oop)?;
    let mut buf = Vec::new();
    b.copy_bytes_out(&mut buf);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Build the guest-facing envelope `{fromId. corr. bytes}` â€” a 3-slot Array.
/// Allocation order + HandleScope: the ByteArray first (rooted), then the
/// Array (whose alloc may move the bytes; refetch through the handle).
fn envelope_to_oop(vm: &mut VmState, env: crate::runtime::workers::Envelope) -> Oop {
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let k_bytes = vm.universe.bytearray_klass;
    let ba = alloc::alloc_indexable_bytes(vm, k_bytes, env.bytes.len());
    for (i, b) in env.bytes.iter().enumerate() {
        ba.byte_at_put(i, *b);
    }
    let ba_h = scope.handle(vm, ba.oop());
    let k_arr = vm.universe.array_klass;
    let arr = alloc::alloc_indexable_oops(vm, k_arr, 3);
    arr.at_put(0, SmallInt::new(i64::from(env.from)).oop());
    arr.at_put(1, SmallInt::new(env.corr as i64).oop());
    arr.at_put(2, ba_h.get(vm));
    arr.oop()
}

/// `primSpawn:` (220): boot a worker VM via the registered boot closure and
/// answer its id; the argument is an init doit source String (run once in
/// the fresh worker â€” how its `Worker onMessage:` handler gets installed) or
/// nil for none. Fails with no boot fn registered, from a worker (star
/// topology), or at the cap.
fn prim_worker_spawn(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let init = if args[1].raw() == vm.universe.nil_obj.raw() {
        None
    } else {
        match string_arg(vm, args[1]) {
            Some(s) => Some(s),
            None => return PrimResult::Fail,
        }
    };
    match crate::runtime::workers::spawn(vm, init) {
        Some(id) => PrimResult::Ok(SmallInt::new(i64::from(id)).oop()),
        None => PrimResult::Fail,
    }
}

/// `primSend:corr:bytes:` (221): enqueue a MOP ByteArray on worker `id`'s
/// channel (or, from a worker, `id` 0 = the primary â€” echoing `corr` routes
/// the reply to its continuation). The send fires the coalesced inbox wake
/// (Â§3.1) on the receiving side's router.
fn prim_worker_send(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(id) = smi_worker_id(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(corr) = SmallInt::try_from(args[2]).and_then(|s| u64::try_from(s.value()).ok()) else {
        return PrimResult::Fail;
    };
    let Some(bytes) = ByteArrayOop::try_from(args[3]) else {
        return PrimResult::Fail;
    };
    let mut buf = Vec::new();
    bytes.copy_bytes_out(&mut buf);
    if crate::runtime::workers::send(vm, id, corr, buf) {
        PrimResult::Ok(args[0])
    } else {
        PrimResult::Fail
    }
}

/// `primPoll` (222): the next envelope for THIS vm as `{fromId. corr. bytes}`,
/// or nil â€” non-blocking, called from inside a wake-triggered dispatch (never
/// a poll loop). Primary: the shared inbox. Worker: the staged pending
/// message its host loop parked.
fn prim_worker_poll(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let _ = args;
    let Some(ws) = vm.workers.as_mut() else {
        return PrimResult::Ok(vm.universe.nil_obj);
    };
    match ws.poll() {
        Some(env) => PrimResult::Ok(envelope_to_oop(vm, env)),
        None => PrimResult::Ok(vm.universe.nil_obj),
    }
}

/// `primAwaitInbox:` (223): the headless run loop's sleep â€” block in the
/// inbox up to `timeoutMs`, answering the envelope or nil. This block IS the
/// primary's idle state (the channel send is the wake; zero spin). Primary
/// only.
fn prim_worker_await(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(ms) = SmallInt::try_from(args[1]).and_then(|s| u64::try_from(s.value()).ok()) else {
        return PrimResult::Fail;
    };
    let Some(ws) = vm.workers.as_mut() else {
        return PrimResult::Fail;
    };
    match ws.await_inbox(ms) {
        Some(env) => PrimResult::Ok(envelope_to_oop(vm, env)),
        None => PrimResult::Ok(vm.universe.nil_obj),
    }
}

/// `primTerminate:` (224): drop worker `id`'s channel (its thread exits on
/// its next recv) and mark it dead. Idempotent.
fn prim_worker_terminate(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(id) = smi_worker_id(args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::workers::terminate(vm, id) {
        PrimResult::Ok(args[0])
    } else {
        PrimResult::Fail
    }
}

/// `primAlive:` (225): is worker `id` believed alive? False once death is
/// DETECTED (failed send / terminate), not instantly at crash.
fn prim_worker_alive(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(id) = smi_worker_id(args[1]) else {
        return PrimResult::Fail;
    };
    let b = crate::runtime::workers::alive(vm, id);
    PrimResult::Ok(if b {
        vm.universe.true_obj
    } else {
        vm.universe.false_obj
    })
}

/// `primSelfId` (226): 0 in the primary, i â‰¥ 1 in worker i â€” lets shared
/// world code know which side of the boundary it is on.
fn prim_worker_self_id(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let _ = args;
    let id = vm.workers.as_ref().map_or(0, |ws| ws.self_id());
    PrimResult::Ok(SmallInt::new(i64::from(id)).oop())
}

// --- cocoa group: the bridge C0 (docs/cocoa_bridge_design.md Â§8) ------------
//
// ObjcRef sends + ownership + the class/NSString entry points. Every send
// goes through the @try shim (objc_bridge::try_send) â€” an NSException comes
// back as a description, is written to the transcript, and the prim FAILS
// (the world method's fallback raises a Smalltalk error; the VM never sees
// an ObjC unwind). Marshalling in C0: ObjcRef â†’ its id, nil â†’ NULL,
// SmallInteger â†’ the value as a GPR word. Everything else fails.

/// Report a caught NSException to the transcript, then fail the prim.
#[cfg(target_os = "macos")]
fn cocoa_exception_fail(vm: &mut VmState, selector: &str, desc: &str) -> PrimResult {
    let _ = writeln!(vm.out, "Cocoa exception in {selector}: {desc}");
    PrimResult::Fail
}

/// A String OR Symbol argument's text (both are byte-format; only the
/// klass differs) â€” the general send's selector/ret-token arguments read
/// naturally as either `'id'` or `#id`.
#[cfg(target_os = "macos")]
fn text_arg(vm: &VmState, oop: Oop) -> Option<String> {
    let m = MemOop::try_from(oop)?;
    let k = m.klass().oop().raw();
    if k != vm.universe.string_klass.oop().raw() && k != vm.universe.symbol_klass.oop().raw() {
        return None;
    }
    let b = ByteArrayOop::try_from(oop)?;
    let mut buf = Vec::new();
    b.copy_bytes_out(&mut buf);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// C0 argument marshalling (design Â§2 clause 3): copies and ids only.
#[cfg(target_os = "macos")]
fn cocoa_marshal_arg(vm: &VmState, o: Oop) -> Option<*mut std::os::raw::c_void> {
    if o.raw() == vm.universe.nil_obj.raw() {
        return Some(std::ptr::null_mut());
    }
    if let Some(smi) = SmallInt::try_from(o) {
        return Some(smi.value() as *mut std::os::raw::c_void);
    }
    crate::runtime::objc_bridge::read_id(vm, o)
}

/// `Cocoa class >> primClassNamed:` (230): an ObjcRef on the named
/// Objective-C class, or fail if no such class is registered.
#[cfg(target_os = "macos")]
fn prim_cocoa_class_named(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(name) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    // CG2 (cocoa_gui_design.md Â§8): fail LOUDLY if a background VM tries to
    // reach an AppKit UI class â€” only the main-thread UI worker may. The
    // transcript line names the refusal; the primitive fails so the guest send
    // raises rather than corrupting AppKit's main-thread-only state.
    if let Err(msg) = crate::runtime::objc_bridge::check_appkit_main_thread(&name) {
        let _ = writeln!(vm.out, "[cocoa-guard] {msg}");
        return PrimResult::Fail;
    }
    match crate::runtime::objc_bridge::class_named(&name) {
        Some(cls) => PrimResult::Ok(crate::runtime::objc_bridge::wrap(vm, cls)),
        None => PrimResult::Fail,
    }
}

/// The shared send core for 231/232/233: n GPR args, id result, wrapped â€”
/// through the +1-family classifier (C1): an `alloc`/`new`/`copy`/
/// `mutableCopy`/`init` result is already caller-owned so the wrap skips
/// its retain, and an init-family send consumes the RECEIVER's ownership
/// (poison-without-release, BEFORE the allocating wrap â€” `args[0]` would
/// be stale after a scavenge).
#[cfg(target_os = "macos")]
fn cocoa_send_n(vm: &mut VmState, args: &[Oop], argn: usize) -> PrimResult {
    let Some(target) = crate::runtime::objc_bridge::read_id(vm, args[0]) else {
        return PrimResult::Fail; // not an ObjcRef, or poisoned
    };
    let Some(sel) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::objc_bridge::is_manual_memory_selector(&sel) {
        return PrimResult::Fail; // raw refcount/dealloc: bridge-owned, never guest-sent
    }
    let mut gpr = [std::ptr::null_mut(); 2];
    for i in 0..argn {
        let Some(m) = cocoa_marshal_arg(vm, args[2 + i]) else {
            return PrimResult::Fail;
        };
        gpr[i] = m;
    }
    let fam = crate::runtime::objc_bridge::selector_family(&sel);
    match crate::runtime::objc_bridge::try_send(target, &sel, gpr[0], gpr[1]) {
        Ok(r) => {
            if fam == crate::runtime::objc_bridge::Family::Init {
                crate::runtime::objc_bridge::consume_receiver(vm, args[0]);
            }
            PrimResult::Ok(crate::runtime::objc_bridge::wrap_result(vm, r, &sel))
        }
        Err(desc) => {
            // ns_consumes_self: init consumed the receiver even on a
            // throw â€” a live wrapper here would over-release later.
            if fam == crate::runtime::objc_bridge::Family::Init {
                crate::runtime::objc_bridge::consume_receiver(vm, args[0]);
            }
            cocoa_exception_fail(vm, &sel, &desc)
        }
    }
}

/// `ObjcRef >> primSend:args:ret:` (240) â€” the C1 general send: any mix of
/// up to 6 GPR-class + 8 FPR-class arguments, spilling to 4 shared stack
/// words in declaration order â€” AAPCS64's exact stage-C rule for SCALAR
/// 8-byte arguments. Two honest limits of the flat model (adversarial-
/// review findings, deliberately deferred to C2's `cocoa_data`-driven
/// shapes): a struct-by-value argument must fit ENTIRELY in its class's
/// remaining registers (a composite that would straddle the register/
/// stack boundary is placed differently by the real ABI â€” whole-composite
/// to stack, registers left idle â€” which this per-scalar model can't
/// express), and a spilled argument narrower than 8 bytes (BOOL/int in
/// stack position) packs to natural size under Darwin, not a full word.
///
/// Argument marshalling (flat register model â€” design Â§2 clause 3):
///   nilâ†’NULL, true/falseâ†’1/0, SmallIntegerâ†’GPR word, ObjcRefâ†’its id,
///   Doubleâ†’FPR double, Stringâ†’a temp NSString (+0 under the bottom pool).
/// A register-resident struct-by-value argument IS its fields in
/// consecutive slots under this model: an NSRange argument is two
/// SmallIntegers, a CGPoint two Doubles, a CGRect four Doubles.
///
/// Ret tokens (the register-classifiable subset of `cocoa_data`'s ABI
/// vocabulary): #id #i64 #i32 #f64 #bool #str #void #range #point #size
/// #rect. #i32 exists because a callee returning C `int` writes only w0
/// (zero-extended) â€” reading it as #i64 makes a negative int a huge
/// positive smi, silently; #i32 sign-extends from bit 31 instead.
/// More args than slots, an unknown token, or an unmarshalable argument
/// all FAIL cleanly (the world fallback raises) â€” the FFI arc's
/// argv-overflow lesson, applied at this entry point from day one.
#[cfg(target_os = "macos")]
fn prim_cocoa_send_general(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    use crate::runtime::objc_bridge::{
        self, RetKind, SEND_FPR_SLOTS, SEND_GPR_SLOTS, SEND_STACK_SLOTS,
    };

    let Some(target) = objc_bridge::read_id(vm, args[0]) else {
        return PrimResult::Fail;
    };
    let Some(sel) = text_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::objc_bridge::is_manual_memory_selector(&sel) {
        return PrimResult::Fail; // raw refcount/dealloc: bridge-owned, never guest-sent
    }
    let Some(arr) = ArrayOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let Some(ret_tok) = text_arg(vm, args[3]) else {
        return PrimResult::Fail;
    };
    let ret = match ret_tok.as_str() {
        "id" => RetKind::Gpr,
        "i64" | "i32" => RetKind::Gpr,
        "bool" => RetKind::Gpr,
        "str" => RetKind::Gpr,
        "f64" => RetKind::Fpr,
        "point" | "size" => RetKind::Hfa2,
        "rect" => RetKind::Hfa4,
        "range" => RetKind::IntPair,
        "void" => RetKind::Void,
        _ => return PrimResult::Fail,
    };
    // Ownership gate (C1 adversarial-review findings): a +1-family
    // selector (alloc/new/copy/mutableCopy/init) answers an object the
    // caller must OWN â€” only #id can take that ownership. Any other ret
    // token would leak the +1 invisibly (no wrap, no counter), so the
    // send is refused BEFORE it runs (no side effects).
    let fam = objc_bridge::selector_family(&sel);
    if fam != objc_bridge::Family::Plus0 && ret_tok != "id" {
        return PrimResult::Fail;
    }

    // Marshal every argument to plain native words BEFORE the call (and
    // before any Smalltalk allocation) â€” GPR class fills x2..x7 then the
    // shared stack words; FPR class fills d0..d7 then the same stack words,
    // in declaration order, per AAPCS64.
    let mut gpr = [0u64; SEND_GPR_SLOTS];
    let mut fpr = [0.0f64; SEND_FPR_SLOTS];
    let mut stack = [0u64; SEND_STACK_SLOTS];
    let (mut ng, mut nf, mut ns) = (0usize, 0usize, 0usize);
    for i in 0..arr.len() {
        let a = arr.at(i);
        // FPR class: a Double argument.
        if let Some(d) = DoubleOop::try_from(a) {
            if nf < SEND_FPR_SLOTS {
                fpr[nf] = d.value();
                nf += 1;
            } else if ns < SEND_STACK_SLOTS {
                stack[ns] = d.value().to_bits();
                ns += 1;
            } else {
                return PrimResult::Fail; // oversized call shape
            }
            continue;
        }
        // Everything else is GPR class.
        let word = if a.raw() == vm.universe.nil_obj.raw() {
            0u64
        } else if a.raw() == vm.universe.true_obj.raw() {
            1u64
        } else if a.raw() == vm.universe.false_obj.raw() {
            0u64
        } else if let Some(smi) = SmallInt::try_from(a) {
            smi.value() as u64
        } else if let Some(id) = objc_bridge::read_id(vm, a) {
            id as u64
        } else if let Some(s) = string_arg(vm, a) {
            // A temp NSString: +0 autoreleased under the bottom pool â€”
            // alive for the call, never wrapped, never retained.
            match objc_bridge::nsstring_from(s.as_bytes()) {
                Ok(ns_id) => ns_id as u64,
                Err(desc) => return cocoa_exception_fail(vm, &sel, &desc),
            }
        } else {
            return PrimResult::Fail; // unmarshalable argument type
        };
        if ng < SEND_GPR_SLOTS {
            gpr[ng] = word;
            ng += 1;
        } else if ns < SEND_STACK_SLOTS {
            stack[ns] = word;
            ns += 1;
        } else {
            return PrimResult::Fail; // oversized call shape
        }
    }

    let out = match objc_bridge::try_send_full(target, &sel, &gpr, &fpr, &stack, ret) {
        Ok(o) => o,
        Err(desc) => {
            // ns_consumes_self: an init-family send consumed the receiver
            // even when it THREW â€” leaving the wrapper live would let a
            // later release over-release (the one direction the design
            // forbids). Consume on both outcomes.
            if fam == objc_bridge::Family::Init {
                objc_bridge::consume_receiver(vm, args[0]);
            }
            return cocoa_exception_fail(vm, &sel, &desc);
        }
    };

    // An init-family send consumed the receiver's ownership â€” poison it
    // FIRST (allocation-free; `args[0]` is still valid because nothing
    // has allocated since entry).
    if fam == objc_bridge::Family::Init {
        objc_bridge::consume_receiver(vm, args[0]);
    }

    // Result construction â€” the only allocating region of this primitive.
    match ret_tok.as_str() {
        "id" => PrimResult::Ok(objc_bridge::wrap_result(
            vm,
            out.gpr[0] as *mut std::os::raw::c_void,
            &sel,
        )),
        "i64" => {
            let v = out.gpr[0] as i64;
            if (crate::oops::layout::SMI_MIN..=crate::oops::layout::SMI_MAX).contains(&v) {
                PrimResult::Ok(SmallInt::new(v).oop())
            } else {
                PrimResult::Fail
            }
        }
        "i32" => {
            // A C `int` return lives in w0 (upper bits zero) â€” sign-extend
            // from bit 31 so `intValue`'s -5 answers -5, not 2^32-5.
            let v = out.gpr[0] as u32 as i32 as i64;
            PrimResult::Ok(SmallInt::new(v).oop())
        }
        "bool" => {
            // BOOL comes back in w0's low byte; AAPCS64 leaves the upper
            // bits unspecified â€” mask before judging.
            PrimResult::Ok(if (out.gpr[0] & 0xFF) != 0 {
                vm.universe.true_obj
            } else {
                vm.universe.false_obj
            })
        }
        "f64" => PrimResult::Ok(alloc::alloc_double(vm, out.fpr[0]).oop()),
        "str" => {
            let ns = out.gpr[0] as *mut std::os::raw::c_void;
            if ns.is_null() {
                return PrimResult::Fail;
            }
            match objc_bridge::nsstring_utf8_bytes(ns) {
                Ok(bytes) => {
                    let k = vm.universe.string_klass;
                    let s = alloc::alloc_indexable_bytes(vm, k, bytes.len());
                    for (i, b) in bytes.iter().enumerate() {
                        s.byte_at_put(i, *b);
                    }
                    PrimResult::Ok(s.oop())
                }
                Err(desc) => cocoa_exception_fail(vm, &sel, &desc),
            }
        }
        "void" => PrimResult::Ok(vm.universe.nil_obj),
        "range" => {
            let (loc, len) = (out.gpr[0] as i64, out.gpr[1] as i64);
            let smi_ok =
                |v: i64| (crate::oops::layout::SMI_MIN..=crate::oops::layout::SMI_MAX).contains(&v);
            if !smi_ok(loc) || !smi_ok(len) {
                return PrimResult::Fail; // NSNotFound territory â€” report, don't wrap wrong
            }
            let a = alloc::alloc_indexable_oops(vm, vm.universe.array_klass, 2);
            a.at_put(0, SmallInt::new(loc).oop());
            a.at_put(1, SmallInt::new(len).oop());
            PrimResult::Ok(a.oop())
        }
        "point" | "size" | "rect" => {
            let n = if ret_tok == "rect" { 4 } else { 2 };
            PrimResult::Ok(cocoa_double_array(vm, &out.fpr[..n]))
        }
        _ => PrimResult::Fail, // unreachable: token validated above
    }
}

/// A fresh Array of fresh Doubles â€” the HFA-result materializer shared by
/// prims 240/241. The array must survive each `alloc_double` (which may
/// move it) â€” handle-rooted, re-derived per store. The store goes through
/// the write-barrier door (`store_tail_oop`), NOT bare `at_put`: a
/// mid-loop scavenge can PROMOTE the array (and re-clean its card once no
/// slot points young), after which a bare store of the next fresh Double
/// is an oldâ†’new reference no future scavenge would ever see â€” the C1
/// adversarial review's dangling-slot scenario under tenuring-threshold 0.
#[cfg(target_os = "macos")]
fn cocoa_double_array(vm: &mut VmState, vals: &[f64]) -> Oop {
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let arr_out = alloc::alloc_indexable_oops(vm, vm.universe.array_klass, vals.len());
    let arr_h = scope.handle(vm, arr_out.oop());
    for (i, v) in vals.iter().enumerate() {
        let d = alloc::alloc_double(vm, *v);
        let arr_now =
            MemOop::try_from(arr_h.get(vm)).expect("handle-rooted array survived the move");
        crate::memory::store::store_tail_oop(vm, arr_now, i, d.oop());
    }
    arr_h.get(vm)
}

/// `ObjcRef >> primSendAuto:args:` (241) â€” the C2 DNU engine: resolve the
/// selector's ABI shape from the LIVE runtime (`method_getTypeEncoding`
/// via [`objc_bridge::resolve_shape`], cached per classÃ—selector), then
/// marshal ENCODING-DRIVEN â€” the callee's signature decides each
/// argument's register class, so `numberWithDouble: 3` coerces the
/// SmallInteger to d0 instead of misplacing it in a GPR. Fails cleanly
/// (the world `doesNotUnderstand:` fallback raises) when the class has no
/// such method, the shape is outside C2's vocabulary, an argument can't
/// coerce, or a struct argument would straddle the register file.
#[cfg(target_os = "macos")]
fn prim_cocoa_send_auto(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    cocoa_send_auto_impl(vm, args, false)
}

/// `ObjcRef >> primSendMainAuto:args:` (242) â€” the C3 variant: identical
/// resolution + marshalling, dispatched synchronously ON THE MAIN THREAD
/// (design Â§4 path 2 â€” AppKit is main-thread-only). Inline when already
/// on main; a clean failure when no GUI host has enabled the hop.
#[cfg(target_os = "macos")]
fn prim_cocoa_send_main_auto(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    cocoa_send_auto_impl(vm, args, true)
}

/// The shared engine behind prims 241/242 â€” `on_main` picks the dispatch
/// leg ([`objc_bridge::try_send_full`] vs [`try_send_full_on_main`]);
/// everything before and after the call is identical, including the
/// ownership rules (the marshalled registers are plain words either way,
/// so the main thread never sees an oop).
#[cfg(target_os = "macos")]
fn cocoa_send_auto_impl(vm: &mut VmState, args: &[Oop], on_main: bool) -> PrimResult {
    use crate::runtime::objc_bridge::{
        self, ObjcArg, ObjcRet, RetKind, SEND_FPR_SLOTS, SEND_GPR_SLOTS, SEND_STACK_SLOTS,
    };

    let Some(target) = objc_bridge::read_id(vm, args[0]) else {
        return PrimResult::Fail;
    };
    let Some(sel) = text_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::objc_bridge::is_manual_memory_selector(&sel) {
        return PrimResult::Fail; // raw refcount/dealloc: bridge-owned, never guest-sent
    }
    let Some(arr) = ArrayOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    // `MACVM_COCOA_DIAG=1` traces each auto-send's selector, resolved ABI
    // shape, and the reason for any failure â€” the channel that localized the
    // nil-SEL menu-init bug. Cached: cocoa sends are rare, but one atomic load
    // beats an env lookup per send.
    let diag = {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("MACVM_COCOA_DIAG").is_some())
    };
    let Some(shape) = objc_bridge::resolve_shape(target, &sel) else {
        if diag {
            eprintln!("[cocoa-diag] '{sel}' on_main={on_main}: resolve_shape FAILED");
        }
        return PrimResult::Fail; // no such method / unmarshalable shape
    };
    if diag {
        eprintln!(
            "[cocoa-diag] '{sel}' on_main={on_main} shape ret={:?} args={:?} argv_len={}",
            shape.ret,
            shape.args,
            arr.len()
        );
    }
    if shape.args.len() != arr.len() {
        if diag {
            eprintln!(
                "[cocoa-diag] '{sel}': ARITY mismatch {} != {}",
                shape.args.len(),
                arr.len()
            );
        }
        return PrimResult::Fail; // arity mismatch (a keyword-count bug)
    }
    // The ownership gate, same rule as prim 240: a +1-family result must
    // be taken as an id. Real family methods always return `@`; a family
    // NAME on a non-object return is the annotated-corner smell â€” refuse.
    let fam = objc_bridge::selector_family(&sel);
    if fam != objc_bridge::Family::Plus0 && shape.ret != ObjcRet::Id {
        if diag {
            eprintln!(
                "[cocoa-diag] '{sel}': FAMILY gate fam={fam:?} ret={:?}",
                shape.ret
            );
        }
        return PrimResult::Fail;
    }

    let mut gpr = [0u64; SEND_GPR_SLOTS];
    let mut fpr = [0.0f64; SEND_FPR_SLOTS];
    let mut stack = [0u64; SEND_STACK_SLOTS];
    let (mut ng, mut nf, mut ns) = (0usize, 0usize, 0usize);

    // Tiny local pushers: scalars may spill to the shared stack words in
    // declaration order (AAPCS64 stage C for 8-byte scalars); STRUCTS must
    // fit entirely in their register class (C1's honest no-straddle limit).
    macro_rules! push_g {
        ($w:expr) => {
            if ng < SEND_GPR_SLOTS {
                gpr[ng] = $w;
                ng += 1;
            } else if ns < SEND_STACK_SLOTS {
                stack[ns] = $w;
                ns += 1;
            } else {
                return PrimResult::Fail;
            }
        };
    }
    macro_rules! push_f {
        ($v:expr) => {
            if nf < SEND_FPR_SLOTS {
                fpr[nf] = $v;
                nf += 1;
            } else if ns < SEND_STACK_SLOTS {
                stack[ns] = $v.to_bits();
                ns += 1;
            } else {
                return PrimResult::Fail;
            }
        };
    }

    // Numeric readers, shared by the scalar and struct arms.
    let as_f64 = |o: Oop| -> Option<f64> {
        if let Some(d) = DoubleOop::try_from(o) {
            Some(d.value())
        } else {
            SmallInt::try_from(o).map(|s| s.value() as f64)
        }
    };
    let as_i64 = |o: Oop| SmallInt::try_from(o).map(|s| s.value());

    for (i, cls) in shape.args.iter().enumerate() {
        let a = arr.at(i);
        match cls {
            ObjcArg::Id => {
                let word = if a.raw() == vm.universe.nil_obj.raw() {
                    0u64
                } else if let Some(id) = objc_bridge::read_id(vm, a) {
                    id as u64
                } else if let Some(s) = text_arg(vm, a) {
                    match objc_bridge::nsstring_from(s.as_bytes()) {
                        Ok(ns_id) => ns_id as u64,
                        Err(desc) => return cocoa_exception_fail(vm, &sel, &desc),
                    }
                } else {
                    return PrimResult::Fail;
                };
                push_g!(word);
            }
            ObjcArg::Bool => {
                let word = if a.raw() == vm.universe.true_obj.raw() {
                    1u64
                } else if a.raw() == vm.universe.false_obj.raw() {
                    0u64
                } else if let Some(v) = as_i64(a) {
                    if v == 0 || v == 1 {
                        v as u64
                    } else {
                        return PrimResult::Fail;
                    }
                } else {
                    return PrimResult::Fail;
                };
                // BOOL is 1 byte â€” register-only: a spilled BOOL packs to
                // natural size on Darwin, so an 8-byte spill word would
                // shift every later stack offset (C2 review).
                if ng >= SEND_GPR_SLOTS {
                    return PrimResult::Fail;
                }
                push_g!(word);
            }
            ObjcArg::Int { signed, bits } => {
                let Some(v) = as_i64(a) else {
                    return PrimResult::Fail;
                };
                // Range-check against the DECLARED width â€” the callee
                // reads exactly that many bits; silently truncating a
                // too-big SmallInteger would be a wrong answer.
                let fits = if *signed {
                    match bits {
                        8 => (-128..=127).contains(&v),
                        16 => (i16::MIN as i64..=i16::MAX as i64).contains(&v),
                        32 => (i32::MIN as i64..=i32::MAX as i64).contains(&v),
                        _ => true,
                    }
                } else {
                    v >= 0
                        && match bits {
                            8 => v <= u8::MAX as i64,
                            16 => v <= u16::MAX as i64,
                            32 => v <= u32::MAX as i64,
                            _ => true,
                        }
                };
                if !fits {
                    return PrimResult::Fail;
                }
                // Narrow integers refuse to SPILL (Darwin packs stack
                // args to natural size â€” the C2 review's mis-marshal
                // finding); registers are width-agnostic and always fine.
                if *bits < 64 && ng >= SEND_GPR_SLOTS {
                    return PrimResult::Fail;
                }
                push_g!(v as u64);
            }
            ObjcArg::F64 => {
                let Some(v) = as_f64(a) else {
                    return PrimResult::Fail;
                };
                push_f!(v);
            }
            ObjcArg::F32 => {
                let Some(v) = as_f64(a) else {
                    return PrimResult::Fail;
                };
                // Register-only (a spilled float packs to 4 bytes).
                if nf >= SEND_FPR_SLOTS {
                    return PrimResult::Fail;
                }
                // The f32 bits ride the d-register's LOW half â€” exactly
                // the callee's `s`-register view of the same register.
                push_f!(f64::from_bits((v as f32).to_bits() as u64));
            }
            ObjcArg::Sel => {
                // `nil` marshals to a NULL SEL â€” a legitimate AppKit idiom
                // (e.g. a submenu-holding `NSMenuItem`'s `action: nil`, or any
                // `target:action:` cleared to no-action). Only a NON-nil value
                // must be a selector name.
                if a.raw() == vm.universe.nil_obj.raw() {
                    push_g!(0u64);
                } else {
                    let Some(name) = text_arg(vm, a) else {
                        return PrimResult::Fail;
                    };
                    let Some(p) = objc_bridge::register_selector(&name) else {
                        return PrimResult::Fail;
                    };
                    push_g!(p as u64);
                }
            }
            ObjcArg::Point | ObjcArg::Rect => {
                let n = if *cls == ObjcArg::Rect { 4 } else { 2 };
                let Some(fields) = ArrayOop::try_from(a) else {
                    return PrimResult::Fail;
                };
                if fields.len() != n || nf + n > SEND_FPR_SLOTS {
                    return PrimResult::Fail; // wrong arity, or would straddle
                }
                for j in 0..n {
                    let Some(v) = as_f64(fields.at(j)) else {
                        return PrimResult::Fail;
                    };
                    fpr[nf] = v;
                    nf += 1;
                }
            }
            ObjcArg::Range => {
                let Some(fields) = ArrayOop::try_from(a) else {
                    return PrimResult::Fail;
                };
                if fields.len() != 2 || ng + 2 > SEND_GPR_SLOTS {
                    return PrimResult::Fail;
                }
                for j in 0..2 {
                    let Some(v) = as_i64(fields.at(j)) else {
                        return PrimResult::Fail;
                    };
                    if v < 0 {
                        return PrimResult::Fail;
                    }
                    gpr[ng] = v as u64;
                    ng += 1;
                }
            }
        }
    }

    let ret_kind = match shape.ret {
        ObjcRet::Id | ObjcRet::Bool | ObjcRet::Int { .. } | ObjcRet::CharStar => RetKind::Gpr,
        ObjcRet::F64 => RetKind::Fpr,
        ObjcRet::F32 => RetKind::F32,
        ObjcRet::Point => RetKind::Hfa2,
        ObjcRet::Rect => RetKind::Hfa4,
        ObjcRet::Range => RetKind::IntPair,
        ObjcRet::Void => RetKind::Void,
    };

    // The hop takes result OWNERSHIP on the main thread (C3 review): a +0
    // object result is retained INSIDE the hop (before main's pools can
    // pop it), a char* result is copied to owned bytes there â€” so the VM
    // thread always receives a +1 id (`hop_owned_id`, wrapped with
    // wrap_owned) or owned Rust memory (`hop_bytes`), never a pointer
    // whose lifetime main still controls.
    let mut hop_bytes: Option<Vec<u8>> = None;
    let hop_owned_id = on_main && shape.ret == ObjcRet::Id;
    let send_result = if on_main {
        let retain_result = hop_owned_id && fam == objc_bridge::Family::Plus0;
        let copy_cstr = shape.ret == ObjcRet::CharStar;
        objc_bridge::try_send_full_on_main_owned(
            target,
            &sel,
            &gpr,
            &fpr,
            &stack,
            ret_kind,
            retain_result,
            copy_cstr,
        )
        .map(|(out, bytes)| {
            hop_bytes = bytes;
            out
        })
    } else {
        objc_bridge::try_send_full(target, &sel, &gpr, &fpr, &stack, ret_kind)
    };
    let out = match send_result {
        Ok(o) => o,
        Err(desc) => {
            // ns_consumes_self holds on the throw path (C1 review). An
            // un-enabled hop fails BEFORE the send ran â€” but the hop error
            // and a thrown NSException are indistinguishable here, and
            // consuming on both keeps the bias leak-side either way.
            if fam == objc_bridge::Family::Init {
                objc_bridge::consume_receiver(vm, args[0]);
            }
            return cocoa_exception_fail(vm, &sel, &desc);
        }
    };
    if fam == objc_bridge::Family::Init {
        objc_bridge::consume_receiver(vm, args[0]);
    }

    let smi_range = crate::oops::layout::SMI_MIN..=crate::oops::layout::SMI_MAX;
    match shape.ret {
        ObjcRet::Id => {
            let id = out.gpr[0] as *mut std::os::raw::c_void;
            PrimResult::Ok(if hop_owned_id {
                // The hop already secured the +1 ON MAIN (retain for
                // Plus0, the family's own +1 otherwise) â€” a retaining
                // wrap here would double-own.
                objc_bridge::wrap_owned(vm, id)
            } else {
                objc_bridge::wrap_result(vm, id, &sel)
            })
        }
        ObjcRet::Bool => PrimResult::Ok(if (out.gpr[0] & 0xFF) != 0 {
            vm.universe.true_obj
        } else {
            vm.universe.false_obj
        }),
        ObjcRet::Int { signed, bits } => {
            // Truncate to the DECLARED width, then sign-/zero-extend from
            // there (the #i32 lesson at every width â€” a callee returning
            // `c`/`s`/`i` writes only that many meaningful bits, upper
            // bits unspecified). A char return is a SmallInteger, never a
            // Boolean: on arm64 BOOL encodes `B`.
            let raw = out.gpr[0];
            let v: i64 = if signed {
                match bits {
                    8 => raw as u8 as i8 as i64,
                    16 => raw as u16 as i16 as i64,
                    32 => raw as u32 as i32 as i64,
                    _ => raw as i64,
                }
            } else {
                match bits {
                    8 => (raw & 0xFF) as i64,
                    16 => (raw & 0xFFFF) as i64,
                    32 => (raw & 0xFFFF_FFFF) as i64,
                    _ => {
                        if raw > crate::oops::layout::SMI_MAX as u64 {
                            // NSNotFound territory â€” report, don't wrap wrong.
                            return PrimResult::Fail;
                        }
                        raw as i64
                    }
                }
            };
            if smi_range.contains(&v) {
                PrimResult::Ok(SmallInt::new(v).oop())
            } else {
                PrimResult::Fail
            }
        }
        ObjcRet::F64 | ObjcRet::F32 => PrimResult::Ok(alloc::alloc_double(vm, out.fpr[0]).oop()),
        ObjcRet::Void => PrimResult::Ok(args[0]),
        ObjcRet::Point => PrimResult::Ok(cocoa_double_array(vm, &out.fpr[..2])),
        ObjcRet::Rect => PrimResult::Ok(cocoa_double_array(vm, &out.fpr[..4])),
        ObjcRet::Range => {
            let (loc, len) = (out.gpr[0] as i64, out.gpr[1] as i64);
            if !smi_range.contains(&loc) || !smi_range.contains(&len) {
                return PrimResult::Fail;
            }
            let a = alloc::alloc_indexable_oops(vm, vm.universe.array_klass, 2);
            a.at_put(0, SmallInt::new(loc).oop());
            a.at_put(1, SmallInt::new(len).oop());
            PrimResult::Ok(a.oop())
        }
        ObjcRet::CharStar => {
            // Copied to owned bytes BEFORE any pool can drain them â€” on
            // the MAIN thread for a hopped send (the copy came back in
            // `hop_bytes`), on this thread otherwise.
            let bytes = if on_main {
                match hop_bytes {
                    Some(b) => b,
                    None => return PrimResult::Fail,
                }
            } else {
                match objc_bridge::cstr_bytes(out.gpr[0] as *const std::os::raw::c_char) {
                    Some(b) => b,
                    None => return PrimResult::Fail,
                }
            };
            let k = vm.universe.string_klass;
            let s = alloc::alloc_indexable_bytes(vm, k, bytes.len());
            for (i, b) in bytes.iter().enumerate() {
                s.byte_at_put(i, *b);
            }
            PrimResult::Ok(s.oop())
        }
    }
}

/// `ObjcRef >> primSend:` (231) / `primSend:with:` (232) /
/// `primSend:with:with:` (233).
#[cfg(target_os = "macos")]
fn prim_cocoa_send0(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    cocoa_send_n(vm, args, 0)
}
#[cfg(target_os = "macos")]
fn prim_cocoa_send1(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    cocoa_send_n(vm, args, 1)
}
#[cfg(target_os = "macos")]
fn prim_cocoa_send2(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    cocoa_send_n(vm, args, 2)
}

/// `ObjcRef >> primSendI64:` (234): a send whose GPR result is an
/// NSInteger/NSUInteger, answered as a SmallInteger. Fails if the value
/// can't be a smi (a >2^61 NSUInteger â€” report, don't wrap wrong), and
/// REFUSES +1-family selectors before sending (the result would be an
/// owned object this integer path can never own â€” C1 review finding).
#[cfg(target_os = "macos")]
fn prim_cocoa_send_i64(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(target) = crate::runtime::objc_bridge::read_id(vm, args[0]) else {
        return PrimResult::Fail;
    };
    let Some(sel) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::objc_bridge::is_manual_memory_selector(&sel) {
        return PrimResult::Fail; // raw refcount/dealloc: bridge-owned, never guest-sent
    }
    if crate::runtime::objc_bridge::selector_family(&sel)
        != crate::runtime::objc_bridge::Family::Plus0
    {
        return PrimResult::Fail;
    }
    match crate::runtime::objc_bridge::try_send(
        target,
        &sel,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ) {
        Ok(r) => {
            let v = r as i64;
            if (crate::oops::layout::SMI_MIN..=crate::oops::layout::SMI_MAX).contains(&v) {
                PrimResult::Ok(SmallInt::new(v).oop())
            } else {
                PrimResult::Fail
            }
        }
        Err(desc) => cocoa_exception_fail(vm, &sel, &desc),
    }
}

/// `ObjcRef >> primSendString:` (235): a send answering an NSString, copied
/// out as a fresh Smalltalk String (design Â§2 clause 3 â€” data crosses by
/// copy; the intermediate NSString stays +0 under the bottom pool).
/// +1-family selectors are refused before sending, same as 234: the +1
/// NSString would leak invisibly behind the copy.
#[cfg(target_os = "macos")]
fn prim_cocoa_send_string(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(target) = crate::runtime::objc_bridge::read_id(vm, args[0]) else {
        return PrimResult::Fail;
    };
    let Some(sel) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    if crate::runtime::objc_bridge::is_manual_memory_selector(&sel) {
        return PrimResult::Fail; // raw refcount/dealloc: bridge-owned, never guest-sent
    }
    if crate::runtime::objc_bridge::selector_family(&sel)
        != crate::runtime::objc_bridge::Family::Plus0
    {
        return PrimResult::Fail;
    }
    match crate::runtime::objc_bridge::try_send_string(target, &sel) {
        Ok(bytes) => {
            let k = vm.universe.string_klass;
            let s = alloc::alloc_indexable_bytes(vm, k, bytes.len());
            for (i, b) in bytes.iter().enumerate() {
                s.byte_at_put(i, *b);
            }
            PrimResult::Ok(s.oop())
        }
        Err(desc) => cocoa_exception_fail(vm, &sel, &desc),
    }
}

/// `ObjcRef >> primRelease` (236): release-with-poison. A double release
/// fails cleanly (the leak-side bias, design Â§3.3).
#[cfg(target_os = "macos")]
fn prim_cocoa_release(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if crate::runtime::objc_bridge::release(vm, args[0]) {
        PrimResult::Ok(args[0])
    } else {
        PrimResult::Fail
    }
}

/// `Cocoa class >> primNSString:` (237): a Smalltalk String copied into a
/// fresh NSString, wrapped (+0 return â†’ wrap retains).
#[cfg(target_os = "macos")]
fn prim_cocoa_nsstring(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(s) = string_arg(vm, args[1]) else {
        return PrimResult::Fail;
    };
    match crate::runtime::objc_bridge::nsstring_from(s.as_bytes()) {
        Ok(ns) => PrimResult::Ok(crate::runtime::objc_bridge::wrap(vm, ns)),
        Err(desc) => cocoa_exception_fail(vm, "stringWithUTF8String:", &desc),
    }
}

/// `Cocoa class >> primDrainPool` (238): drain + renew this thread's bottom
/// pool (the `poolDo:` mechanism, v1-lite).
#[cfg(target_os = "macos")]
fn prim_cocoa_drain_pool(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let _ = vm;
    crate::runtime::objc_bridge::drain_pool_at_doit_boundary();
    PrimResult::Ok(args[0])
}

/// `ObjcRef >> primIsValid` (239): false once poisoned.
#[cfg(target_os = "macos")]
fn prim_cocoa_is_valid(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let b = crate::runtime::objc_bridge::read_id(vm, args[0]).is_some();
    PrimResult::Ok(if b {
        vm.universe.true_obj
    } else {
        vm.universe.false_obj
    })
}

/// `Cocoa class >> primNewAction:` (243) â€” C4: mint a `MacvmAction`
/// target/action trampoline bound to `ticket`. The `{#cocoaEvent. ticket}`
/// fire payload is pickled HERE, on the VM thread, and stored with the
/// registry entry â€” the ObjC-side fire IMP never sees a VmState. Routes to
/// THIS VM's own inbox (`self_inbox_sender`): a Primary posts to itself
/// (unchanged C4/CocoaPad behaviour); a Worker â€” the Cocoa GUI's UI worker â€”
/// posts along its `to_primary` link, lifting the primary-only refusal
/// (`cocoa_gui_design.md` Â§4.3, review item 5). Fails cleanly only when the VM
/// has no worker role at all (no inbox to post to).
#[cfg(target_os = "macos")]
fn prim_cocoa_new_action(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if SmallInt::try_from(args[1]).is_none() {
        return PrimResult::Fail;
    }
    let Some(sender) = crate::runtime::workers::self_inbox_sender(vm) else {
        return PrimResult::Fail;
    };
    // Build {#cocoaEvent. ticket} in-heap, pickle it, drop the oops. The
    // interned Symbol rides a handle across the Array allocation; the
    // ticket is a smi (an immediate â€” no move can stale it).
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let sym = vm.universe.intern(b"cocoaEvent");
    let sym_h = scope.handle(vm, sym.oop());
    let arr = alloc::alloc_indexable_oops(vm, vm.universe.array_klass, 2);
    arr.at_put(0, sym_h.get(vm));
    arr.at_put(1, args[1]);
    let Ok(bytes) = crate::runtime::mop::pickle(vm, arr.oop()) else {
        return PrimResult::Fail;
    };
    match crate::runtime::objc_bridge::new_action(sender, bytes) {
        Ok(inst) => PrimResult::Ok(crate::runtime::objc_bridge::wrap_owned(vm, inst)),
        Err(desc) => cocoa_exception_fail(vm, "macvmAction", &desc),
    }
}

/// `MacvmDelegate class >> primNewDelegate:ticket:` (249) â€” C6 reverse dispatch
/// (`cocoa_gui_design.md` Â§4, CG3): mint a per-role ObjC delegate/data-source
/// instance (`args[1]` a role Symbol â€” `#window`/`#text`/`#table`/`#outline`)
/// bound Rust-side to `(current UI-VM generation, args[2] ticket)`. The world-
/// side `MacvmDelegate` class owns the ticketâ†’receiver map; the ObjC selectors
/// this instance answers dispatch synchronously as top-level entries into THIS
/// VM (`objc_delegate`). Works from ANY VM role, including the Worker UI worker:
/// unlike a C4 action it posts no envelope, so it needs no inbox sender (design
/// Â§4.3). Answers a +1 id (alloc/init â€” wrapped with `wrap_owned`).
#[cfg(target_os = "macos")]
fn prim_cocoa_new_delegate(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(role) = crate::oops::wrappers::SymbolOop::try_from(args[1]).map(|s| s.as_string())
    else {
        return PrimResult::Fail;
    };
    let Some(ticket) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    // The generation live NOW (design Â§4.3): a callback later refuses to dispatch
    // if the UI worker has since been restarted past this generation.
    let gen = crate::embed::current_ui_vm_generation();
    match crate::runtime::objc_delegate::new_delegate(&role, gen, ticket.value()) {
        Ok(inst) => PrimResult::Ok(crate::runtime::objc_bridge::wrap_owned(vm, inst)),
        Err(desc) => cocoa_exception_fail(vm, "macvmDelegate", &desc),
    }
}

/// `Cocoa class >> primPoolPush` (244) â€” open a `poolDo:` mint-list scope:
/// a fresh in-heap list `[count(smi)=0, 8 slots]` pushed on the rooted
/// stack. Every wrapper minted while the scope is open is appended
/// (objc_bridge::mint_note), surviving any number of collections because
/// the list IS a heap object.
#[cfg(target_os = "macos")]
fn prim_cocoa_pool_push(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let arr = alloc::alloc_indexable_oops(vm, vm.universe.array_klass, 9);
    arr.at_put(0, SmallInt::new(0).oop());
    vm.cocoa_mint_stack.push(arr.oop());
    PrimResult::Ok(args[0])
}

/// `Cocoa class >> primPoolPop` (245) â€” close the innermost mint-list
/// scope and answer its list (`[count, w1..wN, nilsâ€¦]`) for the world's
/// release sweep. Fails if no scope is open (an unbalanced pop).
#[cfg(target_os = "macos")]
fn prim_cocoa_pool_pop(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let _ = args;
    match vm.cocoa_mint_stack.pop() {
        Some(o) => PrimResult::Ok(o),
        None => PrimResult::Fail,
    }
}

// --- smi group (SPEC Â§10 appendix) ------------------------------------------


// WINVM: the Cocoa bridge is macOS-only; on other platforms every cocoa
// primitive exists (the prim table above is platform-neutral) but fails
// cleanly, writing one honest line to the transcript. The world method''s
// fallback then raises an ordinary Smalltalk error, same as any prim Fail.
#[cfg(not(target_os = "macos"))]
macro_rules! cocoa_unavailable_prims {
    ($($name:ident),+ $(,)?) => {
        $(fn $name(vm: &mut VmState, _args: &[Oop]) -> PrimResult {
            let _ = writeln!(vm.out, "Cocoa bridge is macOS-only; primitive unavailable on this platform");
            PrimResult::Fail
        })+
    };
}

#[cfg(not(target_os = "macos"))]
cocoa_unavailable_prims!(
    prim_cocoa_class_named,
    prim_cocoa_send0,
    prim_cocoa_send1,
    prim_cocoa_send2,
    prim_cocoa_send_i64,
    prim_cocoa_send_string,
    prim_cocoa_release,
    prim_cocoa_nsstring,
    prim_cocoa_drain_pool,
    prim_cocoa_is_valid,
    prim_cocoa_send_general,
    prim_cocoa_send_auto,
    prim_cocoa_send_main_auto,
    prim_cocoa_new_action,
    prim_cocoa_pool_push,
    prim_cocoa_pool_pop,
    prim_cocoa_new_delegate,
);

fn smi2(args: &[Oop]) -> Option<(SmallInt, SmallInt)> {
    Some((SmallInt::try_from(args[0])?, SmallInt::try_from(args[1])?))
}

fn prim_add(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args).and_then(|(a, b)| a.checked_add(b)) {
        Some(r) => PrimResult::Ok(r.oop()),
        None => PrimResult::Fail,
    }
}

fn prim_sub(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args).and_then(|(a, b)| a.checked_sub(b)) {
        Some(r) => PrimResult::Ok(r.oop()),
        None => PrimResult::Fail,
    }
}

fn prim_mul(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args).and_then(|(a, b)| a.checked_mul(b)) {
        Some(r) => PrimResult::Ok(r.oop()),
        None => PrimResult::Fail,
    }
}

fn prim_div(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args).and_then(|(a, b)| a.checked_div(b)) {
        Some(r) => PrimResult::Ok(r.oop()),
        None => PrimResult::Fail,
    }
}

fn prim_mod(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args).and_then(|(a, b)| a.checked_rem(b)) {
        Some(r) => PrimResult::Ok(r.oop()),
        None => PrimResult::Fail,
    }
}

fn prim_bit_and(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args) {
        Some((a, b)) => PrimResult::Ok(SmallInt::new(a.value() & b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_bit_or(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args) {
        Some((a, b)) => PrimResult::Ok(SmallInt::new(a.value() | b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_bit_xor(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match smi2(args) {
        Some((a, b)) => PrimResult::Ok(SmallInt::new(a.value() ^ b.value()).oop()),
        None => PrimResult::Fail,
    }
}

/// `count > 0`: left shift (Fail on overflow past the smi range). `count <
/// 0`: arithmetic right shift (never overflows). `|count| >= 62`: Fail
/// regardless of the receiver's value (pinned appendix rule, independent of
/// `SmallInt::checked_shl`'s own range check).
fn prim_bit_shift(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((r, c)) = smi2(args) else {
        return PrimResult::Fail;
    };
    let count = c.value();
    if !(-61..=61).contains(&count) {
        return PrimResult::Fail;
    }
    if count >= 0 {
        match r.checked_shl(count as u32) {
            Some(v) => PrimResult::Ok(v.oop()),
            None => PrimResult::Fail,
        }
    } else {
        PrimResult::Ok(SmallInt::new(r.value() >> (-count)).oop())
    }
}

fn prim_lt(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a < b)
}
fn prim_le(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a <= b)
}
fn prim_gt(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a > b)
}
fn prim_ge(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a >= b)
}
fn prim_eq(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a == b)
}
fn prim_ne(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    smi_cmp(_vm, args, |a, b| a != b)
}

fn smi_cmp(vm: &mut VmState, args: &[Oop], f: impl Fn(i64, i64) -> bool) -> PrimResult {
    match smi2(args) {
        Some((a, b)) => PrimResult::Ok(bool_oop(vm, f(a.value(), b.value()))),
        None => PrimResult::Fail,
    }
}

fn bool_oop(vm: &VmState, b: bool) -> Oop {
    if b {
        vm.universe.true_obj
    } else {
        vm.universe.false_obj
    }
}

// --- oops group --------------------------------------------------------------

pub fn klass_of(vm: &VmState, o: Oop) -> KlassOop {
    crate::runtime::lookup::klass_of(vm, o)
}

fn prim_identity_hash(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let recv = args[0];
    if recv.is_smi() {
        PrimResult::Ok(recv)
    } else {
        let h = vm.universe.identity_hash(recv);
        PrimResult::Ok(SmallInt::new(h as i64).oop())
    }
}

fn prim_class(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    PrimResult::Ok(klass_of(vm, args[0]).oop())
}

fn prim_identical(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    PrimResult::Ok(bool_oop(_vm, args[0].raw() == args[1].raw()))
}

fn prim_basic_new(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(klass) = KlassOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    match klass.format() {
        Format::Slots => PrimResult::Ok(alloc::alloc_slots(vm, klass).oop()),
        Format::IndexableOops => PrimResult::Ok(alloc::alloc_indexable_oops(vm, klass, 0).oop()),
        Format::IndexableBytes => PrimResult::Ok(alloc::alloc_indexable_bytes(vm, klass, 0).oop()),
        _ => PrimResult::Fail,
    }
}

fn prim_basic_new_colon(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(klass) = KlassOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(n) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    if n.value() < 0 {
        return PrimResult::Fail;
    }
    let n = n.value() as usize;
    match klass.format() {
        Format::IndexableOops => PrimResult::Ok(alloc::alloc_indexable_oops(vm, klass, n).oop()),
        Format::IndexableBytes => PrimResult::Ok(alloc::alloc_indexable_bytes(vm, klass, n).oop()),
        _ => PrimResult::Fail,
    }
}

fn prim_inst_var_at(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(m) = MemOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(idx) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let i = idx.value();
    let named = (m.klass().non_indexable_size() - HEADER_WORDS) as i64;
    if i < 1 || i > named {
        return PrimResult::Fail;
    }
    PrimResult::Ok(m.body_oop((i - 1) as usize))
}

fn prim_at(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(a) = ArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(idx) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let i = idx.value();
    if i < 1 || i as usize > a.len() {
        return PrimResult::Fail;
    }
    PrimResult::Ok(a.at((i - 1) as usize))
}

fn prim_at_put(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(a) = ArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(idx) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let i = idx.value();
    if i < 1 || i as usize > a.len() {
        return PrimResult::Fail;
    }
    crate::memory::store::store_tail_oop(vm, a.as_mem(), (i - 1) as usize, args[2]);
    PrimResult::Ok(args[2])
}

fn prim_size(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(m) = MemOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    match m.klass().format() {
        Format::IndexableOops | Format::IndexableBytes => {
            PrimResult::Ok(SmallInt::new(m.indexable_len() as i64).oop())
        }
        _ => PrimResult::Fail,
    }
}

// --- bytes group (receiver format must be IndexableBytes) --------------------

fn is_symbol(vm: &VmState, b: ByteArrayOop) -> bool {
    b.as_mem().klass() == vm.universe.symbol_klass
}

fn prim_byte_at(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(idx) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let i = idx.value();
    if i < 1 || i as usize > b.len() {
        return PrimResult::Fail;
    }
    PrimResult::Ok(SmallInt::new(b.byte_at((i - 1) as usize) as i64).oop())
}

fn prim_byte_at_put(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    if is_symbol(vm, b) {
        return PrimResult::Fail;
    }
    let Some(idx) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(val) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let i = idx.value();
    let v = val.value();
    if i < 1 || i as usize > b.len() {
        return PrimResult::Fail;
    }
    if !(0..=255).contains(&v) {
        return PrimResult::Fail;
    }
    b.byte_at_put((i - 1) as usize, v as u8);
    PrimResult::Ok(args[2])
}

fn prim_byte_size(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    PrimResult::Ok(SmallInt::new(b.len() as i64).oop())
}

/// Copies `arg3[1 .. to-from+1]` (source always addressed from 1,
/// independent of `from`) into `receiver[from..to]`. `to < from` is a
/// no-op. Same-object overlap is handled via an intermediate buffer â€” the
/// heap accessor discipline (`oops::heap`) never exposes a raw `&mut [u8]`
/// into the heap, so there is no slice to `copy_within` on directly.
fn prim_replace_from_to_with(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(recv) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    if is_symbol(vm, recv) {
        return PrimResult::Fail;
    }
    let Some(from) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(to) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let Some(src) = ByteArrayOop::try_from(args[3]) else {
        return PrimResult::Fail;
    };
    let (from, to) = (from.value(), to.value());
    if to < from {
        return PrimResult::Ok(args[0]);
    }
    if from < 1 {
        return PrimResult::Fail;
    }
    let count = (to - from + 1) as usize;
    if to as usize > recv.len() || count > src.len() {
        return PrimResult::Fail;
    }

    let mut buf = vec![0u8; count];
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = src.byte_at(i);
    }
    let base = (from - 1) as usize;
    for (i, &b) in buf.iter().enumerate() {
        recv.byte_at_put(base + i, b);
    }
    PrimResult::Ok(args[0])
}

fn prim_hash_bytes(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64 offset basis
    for i in 0..b.len() {
        h ^= b.byte_at(i) as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3); // FNV prime
    }
    PrimResult::Ok(SmallInt::new((h & 0x3FFF_FFFF) as i64).oop())
}

fn prim_compare(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(a) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(b) = ByteArrayOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let (la, lb) = (a.len(), b.len());
    for i in 0..la.min(lb) {
        let (ba, bb) = (a.byte_at(i), b.byte_at(i));
        if ba != bb {
            return PrimResult::Ok(SmallInt::new(if ba < bb { -1 } else { 1 }).oop());
        }
    }
    let r = match la.cmp(&lb) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    };
    PrimResult::Ok(SmallInt::new(r).oop())
}

// --- block-value group (SPEC Â§10, S4) ----------------------------------------
// Layer boundary (sprint_s04_detail.md Â§Layer boundaries): the arrow from
// runtime/ into interpreter/ is allowed ONLY for this Activated family â€”
// these primitives never touch InterpRegs directly, they just delegate to
// `interpreter::blocks::activate_block`.

fn prim_value0(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(cl) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    crate::interpreter::blocks::activate_block(vm, cl, 0)
}

fn prim_value1(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(cl) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    crate::interpreter::blocks::activate_block(vm, cl, 1)
}

fn prim_value2(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(cl) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    crate::interpreter::blocks::activate_block(vm, cl, 2)
}

fn prim_value3(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(cl) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    crate::interpreter::blocks::activate_block(vm, cl, 3)
}

/// Spreads `args[1]` (an `Array` whose size must equal the block's argc) in
/// place on the operand stack â€” pure stack surgery, no allocation. Checks
/// the size *before* popping the array (Pitfalls: never allocate/mutate on
/// a still-unvalidated argument).
fn prim_value_with_arguments(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(cl) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(arr) = ArrayOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let n = arr.len();
    if n != cl.method().argc() {
        return PrimResult::Fail;
    }
    let array_slot = vm.prim_arg_base + 1;
    vm.stack.sp = array_slot; // drop the Array argument itself
    for i in 0..n {
        vm.stack.push(arr.at(i));
    }
    crate::interpreter::blocks::activate_block_interp(vm, cl, n)
}

/// SPEC Â§5.4 Algorithm 6: `ensure:`/`ifCurtailed:` both activate the
/// receiver (the protected block, argc 0), then arm the handler as that
/// *new* activation's marker â€” the marker lives on the protected block's
/// own frame, never on a frame of its own for `ensure:` itself.
fn prim_ensure_like(
    vm: &mut VmState,
    args: &[Oop],
    kind: crate::interpreter::stack::MarkerKind,
) -> PrimResult {
    let Some(protected) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let Some(handler) = crate::oops::wrappers::ClosureOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    if protected.method().argc() != 0 || handler.method().argc() != 0 {
        return PrimResult::Fail;
    }
    // `activate_block` computes the new frame's receiver-arg slot as
    // `sp - argc - 1` on the CALLER's stack â€” but the caller's stack here
    // still holds the handler as `ensure:`'s own keyword argument, one
    // slot above `protected`. Drop it (already captured in `handler`
    // above) so `sp - 1` lands on `protected`, matching a plain `value`
    // send's stack shape exactly.
    vm.stack.sp -= 1;
    // `activate_block_interp`, NOT the trigger-bearing `activate_block`
    // (S24 A1): the marker set below lives ON the protected block's
    // interpreter frame â€” it is what `do_return` intercepts for the
    // normal-completion handler run and what `continue_unwind`'s scan
    // finds for the NLR case. The compiled path completes the whole block
    // INSIDE the primitive (no frame, `Completed`/`Nlr`, never
    // `Activated`), so a compiled protected block would skip its handler
    // on BOTH paths.
    match crate::interpreter::blocks::activate_block_interp(vm, protected, 0) {
        PrimResult::Activated => {
            let frame = crate::interpreter::stack::Frame { fp: vm.stack.fp };
            frame.set_marker(&mut vm.stack, handler.oop(), kind);
            PrimResult::Activated
        }
        other => other, // unreachable given the argc pre-check above; propagate defensively
    }
}

// --- reflection group ---------------------------------------------------------

/// `anObject respondsTo: #foo` â€” does a lookup from the receiver's klass find
/// the selector? Answers the question directly through the SAME chain walk
/// (and lookup cache) real dispatch uses, rather than the Smalltalk-level
/// alternative of materializing every class's `selectorsOf:` Array up the
/// superclass chain just to test membership.
fn prim_responds_to(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(sel) = crate::oops::wrappers::SymbolOop::try_from(args[1]) else {
        return PrimResult::Fail; // not a Symbol
    };
    let klass = klass_of(vm, args[0]);
    let found = crate::runtime::lookup::lookup(vm, klass, sel).is_some();
    PrimResult::Ok(bool_oop(vm, found))
}

/// `anObject shallowCopy` â€” a fresh object of the receiver's exact klass and
/// size, with every named instance variable and every indexed element copied
/// verbatim (one level deep; the elements themselves are shared).
///
/// FAILS (so the Smalltalk fallback answers `self`) for anything that is not a
/// copyable heap shape: immediates (smis/chars), Doubles, and the internal
/// Klass/Method/Closure/Context/Process formats â€” all either value-like or
/// not meaningfully duplicable. `Symbol` is IndexableBytes and so *would* copy
/// here; 32_object_ext.mst overrides it back to `^self`, because a Symbol's
/// whole contract is that identity IS equality.
fn prim_shallow_copy(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(src) = MemOop::try_from(args[0]) else {
        return PrimResult::Fail; // immediate â€” the fallback answers self
    };
    let klass = src.klass();
    let format = klass.format();
    let named = klass.non_indexable_size() - HEADER_WORDS;
    let len = match format {
        Format::Slots => 0,
        Format::IndexableOops | Format::IndexableBytes => src.indexable_len(),
        _ => return PrimResult::Fail,
    };

    // The allocation below can scavenge, which MOVES the receiver â€” every read
    // of `src` after it must go through the handle, never the stale local.
    // (`alloc_words` roots the klass it is handed itself, so `klass` needs no
    // handle of its own.)
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let src_h = scope.handle(vm, args[0]);
    let copy = match format {
        Format::Slots => alloc::alloc_slots(vm, klass),
        Format::IndexableOops => alloc::alloc_indexable_oops(vm, klass, len).as_mem(),
        Format::IndexableBytes => alloc::alloc_indexable_bytes(vm, klass, len).as_mem(),
        _ => unreachable!("format was filtered above"),
    };
    let src = MemOop::try_from(src_h.get(vm)).expect("receiver is still a mem oop after the copy");

    for i in 0..named {
        copy.set_body_oop(i, src.body_oop(i));
    }
    match format {
        // Body layout is [named ivars][size slot][elements] â€” element i sits
        // at body index `named + 1 + i` (`heap::instance_size_words`).
        Format::IndexableOops => {
            for i in 0..len {
                copy.set_body_oop(named + 1 + i, src.body_oop(named + 1 + i));
            }
        }
        // Bytes go through the tail byte accessors, NOT body_oop: raw byte
        // words are not oops, and Oop::from_raw's debug validation rejects
        // arbitrary bit patterns.
        Format::IndexableBytes => {
            for i in 0..len {
                copy.set_tail_byte_at(i, src.tail_byte_at(i));
            }
        }
        _ => {}
    }
    PrimResult::Ok(copy.oop())
}

/// `aBlock numArgs` â€” the block's declared argument count, straight off its
/// CompiledBlock.
fn prim_block_num_args(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(closure) = crate::oops::wrappers::ClosureOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    PrimResult::Ok(SmallInt::new(closure.method().argc() as i64).oop())
}

fn prim_ensure(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    prim_ensure_like(vm, args, crate::interpreter::stack::MarkerKind::Ensure)
}

fn prim_if_curtailed(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    prim_ensure_like(vm, args, crate::interpreter::stack::MarkerKind::IfCurtailed)
}

// --- system group (dev hooks) --------------------------------------------------

fn prim_quit(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    vm.exit_requested = true;
    PrimResult::Ok(args[0])
}

fn prim_print_on_stdout(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let mut buf = Vec::new();
    b.copy_bytes_out(&mut buf);
    let _ = vm.out.write_all(&buf);
    PrimResult::Ok(args[0])
}

fn prim_millisecond_clock(vm: &mut VmState, _args: &[Oop]) -> PrimResult {
    let millis = vm.start_instant.elapsed().as_millis() as i64;
    PrimResult::Ok(SmallInt::new(millis).oop())
}

/// `Smalltalk gcScavenge` (SPEC Â§10 system group): runs one young-gen
/// collection, answers the receiver. A stall here is handled exactly like
/// the allocation cascade's own terminal stall (`alloc::stall_exit`) â€” an
/// explicit `gcScavenge` call finding no way forward is just as fatal as
/// one the allocator triggered itself, not a different situation.
fn prim_gc_scavenge(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    // S12 step 7: S11 D8's defer arm (`if compiled_depth > 0 {
    // request_pending_gc(...); return }`) lived here and is DELETED â€” a
    // scavenge under a live compiled frame is now an ordinary, fully
    // rooted collection (`roots::each_code_root`), so this primitive
    // always collects immediately, at any depth.
    let _ = args;
    if let Err(err) = crate::memory::scavenge::scavenge(vm) {
        alloc::stall_exit(err);
    }
    // `prim_arg(0)`, NOT `args[0]`: the scavenge above may have MOVED the
    // receiver, and `args` is a pre-call copy of raw bits (SPEC Â§10
    // Pitfalls â€” re-read every arg through the live stack slot after any
    // allocating/collecting call). Latent since S8, unreachable until now:
    // every real sender was `Smalltalk gcScavenge`, and the `smalltalk`
    // singleton is tenured by the time user code runs, so the stale copy
    // always happened to equal the live slot. A YOUNG receiver â€” exactly
    // what `mid_loop_forced_scavenge`'s own `p scav` sends â€” moves, and
    // returning the stale copy would resurrect its vacated address as a
    // value.
    PrimResult::Ok(vm.prim_arg(0))
}

/// `Smalltalk gcFull` (SPEC Â§10 system group): runs the full mark-slide-
/// compact collection, answers the receiver. `full_gc` never actually
/// returns `Err` (pure compaction needs no new memory â€” see its own doc);
/// the `expect` documents that rather than silently discarding a `Result`.
fn prim_gc_full(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    // S12 step 7: defer arm deleted, receiver re-read live â€” both for
    // exactly `prim_gc_scavenge`'s reasons (see its comments; a full GC's
    // compaction slides young AND old objects, so the stale-`args[0]`
    // hazard is even broader here).
    let _ = args;
    crate::memory::fullgc::full_gc(vm)
        .expect("full_gc: pure compaction never returns Err (see its own doc)");
    PrimResult::Ok(vm.prim_arg(0))
}

/// `Smalltalk gcStats` (SPEC Â§10 dev hook, S8): answers an 8-element Array
/// of smis in the SPEC-pinned order `(scavengeCount fullGcCount edenUsed
/// oldUsed oldCommitted bytesPromoted markedBytesLast contextAllocs)` â€”
/// used by the soak harness and S14's Context-elision gate, which both
/// index into it by POSITION, so this order is load-bearing, not
/// cosmetic. `can_allocate: true` (the result Array itself allocates); no
/// arg oops are read after the allocation, so there is nothing here for
/// that to invalidate.
/// R1 reflection (`docs/APPS.md` Â§3): every class object in the system, as an
/// `Array`. Walks the global namespace (`vm.universe.smalltalk`: slot 0 is the
/// tally, then `tally` Associations) and collects each association value that
/// is a klass â€” the image-side `ClassMirror` filters these by `superclass` to
/// compute subclasses (the VM keeps no subclass index, exactly as Strongtalk's
/// `ClassVMMirror` walks `Smalltalk classesDo:`). Receiver/args are ignored.
///
/// GC-safe two-pass: pass 1 counts (no allocation); `alloc_indexable_oops` may
/// then scavenge, so pass 2 re-reads `vm.universe.smalltalk` (a scanned root,
/// updated across the collection) rather than caching any oop across the alloc.
/// `Object >> perform:withArguments:` (64) â€” a dynamic send: look the
/// `selector` (a Symbol) up on the receiver's class and invoke the found
/// method with the elements of `argsArray` spread as its arguments,
/// answering the result. This is Smalltalk's reflective "call a method by
/// name" and the mechanism behind the worker-VM RPC layer
/// (docs/multi-smalltalk-worker.md): a Symbol and an argument Array both
/// pickle across a VM boundary, so a primary can name a method for a
/// worker to run.
///
/// Fails cleanly (the world fallback raises) on: a non-Symbol selector, a
/// non-Array args, an argc mismatch between the array and the resolved
/// method, or a selector the receiver's class does not implement
/// (`doesNotUnderstand:` territory â€” reported by the world method rather
/// than silently). The heavy lifting is [`run_method_reentrant`], the same
/// synchronous-nested-invoke `rt_dnu` uses: it handles a primitive,
/// compiled, or interpreted target uniformly and returns the result oop.
/// NON-shimmable for the same reason as `primEvalDoit:` (250) â€” see
/// `compiler::driver::PRIM_REENTERS_INTERPRETER`: the nested run needs the
/// c2i bracket when the caller is compiled, and a `PrimFn` body cannot
/// supply it correctly for both of its entry doors.
fn prim_perform_with_arguments(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let receiver = args[0];
    let Some(sel) = crate::oops::wrappers::SymbolOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(arr) = ArrayOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let n = arr.len();
    let k = klass_of(vm, receiver);
    let Some(method) = crate::runtime::lookup::lookup(vm, k, sel) else {
        return PrimResult::Fail; // not understood â€” world fallback raises
    };
    if method.argc() != n {
        return PrimResult::Fail; // wrong number of arguments for this method
    }
    // Root the receiver, the resolved method, and every argument across
    // the nested run (which allocates freely), then materialize the call
    // slice fresh immediately before invoking â€” nothing allocates between
    // `get` and `run_method_reentrant`'s own receiver/args push.
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let recv_h = scope.handle(vm, receiver);
    let method_h = scope.handle(vm, method.oop());
    let arg_hs: Vec<_> = (0..n).map(|i| scope.handle(vm, arr.at(i))).collect();
    let recv = recv_h.get(vm);
    let m = crate::oops::wrappers::MethodOop::try_from(method_h.get(vm))
        .expect("rooted method survived the scope");
    let argv: Vec<Oop> = arg_hs.iter().map(|h| h.get(vm)).collect();
    let result = crate::interpreter::run_method_reentrant(vm, m, recv, &argv);
    PrimResult::Ok(result)
}

fn prim_all_classes(vm: &mut VmState, _args: &[Oop]) -> PrimResult {
    fn tally_of(vm: &VmState) -> (ArrayOop, usize) {
        let arr = ArrayOop::try_from(vm.universe.smalltalk)
            .expect("allClasses: the global namespace is not an Array");
        let tally = SmallInt::try_from(arr.at(0))
            .expect("allClasses: globals tally is not a smi")
            .value() as usize;
        (arr, tally)
    }
    let is_class = |assoc: Oop| -> Option<Oop> {
        let value = MemOop::try_from(assoc)
            .expect("allClasses: globals slot is not an Association")
            .body_oop(1);
        KlassOop::try_from(value).map(|_| value)
    };

    // Pass 1: count (allocation-free).
    let (arr, tally) = tally_of(vm);
    let mut count = 0usize;
    for i in 0..tally {
        if is_class(arr.at(1 + i)).is_some() {
            count += 1;
        }
    }

    // Allocate the result (may move the heap).
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, count);

    // Pass 2: re-read the (possibly moved) namespace root and fill.
    let (arr, tally) = tally_of(vm);
    let mut j = 0usize;
    for i in 0..tally {
        if let Some(value) = is_class(arr.at(1 + i)) {
            if j < count {
                result.at_put(j, value);
                j += 1;
            }
        }
    }
    PrimResult::Ok(result.oop())
}

/// R2 reflection (`docs/APPS.md` Â§3): the selectors a behavior defines *in
/// its own* method dictionary (not inherited), as an `Array` of `Symbol`s in
/// unspecified order (the caller sorts). `args[1]` is the behavior â€” pass a
/// class for its instance selectors, `aClass class` (the metaclass) for its
/// class-side selectors. A behavior with no method dictionary yet answers an
/// empty `Array`.
///
/// GC-safe: the behavior is handle-rooted across `alloc_indexable_oops` (which
/// may scavenge and move it), and its dictionary is re-derived after the
/// allocation rather than cached across it.
fn prim_selectors_of(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if KlassOop::try_from(args[1]).is_none() {
        return PrimResult::Fail; // not a behavior
    }
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let behavior_h = scope.handle(vm, args[1]);

    // Pass 1: count (allocation-free).
    let mut count = 0usize;
    if let Some(dict) = selector_dict(vm, behavior_h.get(vm)) {
        dict.each_pair(vm, |_sel, _m| count += 1);
    }

    // Allocate the result (may move the heap).
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, count);

    // Pass 2: re-derive the (possibly moved) dictionary and fill.
    if let Some(dict) = selector_dict(vm, behavior_h.get(vm)) {
        let mut j = 0usize;
        dict.each_pair(vm, |sel, _m| {
            if j < count {
                result.at_put(j, sel.oop());
                j += 1;
            }
        });
    }
    PrimResult::Ok(result.oop())
}

/// R2 reflection (`docs/APPS.md` Â§3): a class's OWN instance-variable names (not
/// inherited), as an `Array` of `Symbol`s in declaration order â€” the variable
/// half of source-level reflection, the analogue of `selectorsOf:`. `args[1]` is
/// the class. A FRESH array is answered (never the Klass's own `inst_var_names`
/// array), so a caller can't mutate the class's shape. GC-safe: the behavior is
/// handle-rooted across the allocation and its names array is re-derived after.
fn prim_instance_variables_of(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if KlassOop::try_from(args[1]).is_none() {
        return PrimResult::Fail; // not a behavior
    }
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let behavior_h = scope.handle(vm, args[1]);

    let count = KlassOop::try_from(behavior_h.get(vm))
        .and_then(|k| ArrayOop::try_from(k.inst_var_names()))
        .map(|a| a.len())
        .unwrap_or(0);
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, count);

    if let Some(names) =
        KlassOop::try_from(behavior_h.get(vm)).and_then(|k| ArrayOop::try_from(k.inst_var_names()))
    {
        for i in 0..count.min(names.len()) {
            result.at_put(i, names.at(i));
        }
    }
    PrimResult::Ok(result.oop())
}

/// R2 reflection: a class's OWN class-variable names, as an `Array` of `Symbol`s.
/// The Klass stores class vars as `Association`s (name -> value binding); this
/// answers just the keys (`body_oop(0)`). Same freshness + GC contract as
/// `instanceVariablesOf:`. No allocation happens inside the fill loop, so the
/// re-derived associations stay valid throughout.
fn prim_class_variables_of(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if KlassOop::try_from(args[1]).is_none() {
        return PrimResult::Fail; // not a behavior
    }
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let behavior_h = scope.handle(vm, args[1]);

    let count = KlassOop::try_from(behavior_h.get(vm))
        .and_then(|k| ArrayOop::try_from(k.class_vars()))
        .map(|a| a.len())
        .unwrap_or(0);
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, count);

    let nil = vm.universe.nil_obj;
    if let Some(assocs) =
        KlassOop::try_from(behavior_h.get(vm)).and_then(|k| ArrayOop::try_from(k.class_vars()))
    {
        for i in 0..count.min(assocs.len()) {
            let key = MemOop::try_from(assocs.at(i))
                .map(|m| m.body_oop(0))
                .unwrap_or(nil);
            result.at_put(i, key);
        }
    }
    PrimResult::Ok(result.oop())
}

/// R2 reflection: the primitive number of `behavior`'s method for `selector`
/// (`args[1]` = behavior, `args[2]` = selector Symbol). `0` = not a primitive
/// (or the method isn't defined here); a positive id is a table primitive;
/// `-1` (`PRIM_ID_FFI`) is an FFI primitive. Lets a tool show primitive
/// methods read-only rather than as an editable source box.
fn prim_primitive_of(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(behavior) = KlassOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(sel) = crate::oops::wrappers::SymbolOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let n = match crate::oops::method_dict::MethodDictOop::try_from(behavior.methods()) {
        Some(dict) => dict.probe(vm, sel).map(|m| m.primitive()).unwrap_or(0),
        None => 0,
    };
    PrimResult::Ok(SmallInt::new(n).oop())
}

/// R2 reflection (senders): does `behavior`'s method for `msel` (`args[2]`)
/// SEND `target` (`args[3]`)? Scans the method's inline-cache side table
/// (`ics`, one stride-`IC_STRIDE` slot per send site, the selector at
/// `IC_SEL_OFFSET`) â€” selectors are interned, so an identity compare suffices.
/// `false` if the method is undefined here. (Sends inlined by the compiler â€”
/// `ifTrue:` etc. â€” have no IC and so aren't counted, same limit Strongtalk's
/// own senders had.)
fn prim_method_sends(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    use crate::oops::layout::{IC_SEL_OFFSET, IC_STRIDE};
    let Some(behavior) = KlassOop::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(msel) = crate::oops::wrappers::SymbolOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let Some(target) = crate::oops::wrappers::SymbolOop::try_from(args[3]) else {
        return PrimResult::Fail;
    };
    let m = match crate::oops::method_dict::MethodDictOop::try_from(behavior.methods()) {
        Some(dict) => dict.probe(vm, msel),
        None => None,
    };
    let Some(m) = m else {
        return PrimResult::Ok(vm.universe.false_obj);
    };
    let ics = m.ics();
    let target_bits = target.oop().raw();
    let sites = ics.len() / IC_STRIDE;
    for i in 0..sites {
        if ics.at(i * IC_STRIDE + IC_SEL_OFFSET).raw() == target_bits {
            return PrimResult::Ok(vm.universe.true_obj);
        }
    }
    PrimResult::Ok(vm.universe.false_obj)
}

/// A behavior's own `MethodDictionary`, or `None` if it isn't a behavior or
/// has no dictionary yet.
fn selector_dict(_vm: &VmState, behavior: Oop) -> Option<crate::oops::method_dict::MethodDictOop> {
    let k = KlassOop::try_from(behavior)?;
    crate::oops::method_dict::MethodDictOop::try_from(k.methods())
}

fn prim_gc_stats(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let smi = |v: u64| SmallInt::new(v as i64).oop();
    let u = &vm.universe;
    let fields = [
        smi(u.gc_stats.scavenge_count),
        smi(u.gc_stats.full_gc_count),
        smi((u.eden.top - u.eden.start) as u64),
        smi((u.old.top - u.old.bounds.start) as u64),
        smi((u.old.committed_end - u.old.bounds.start) as u64),
        smi(u.gc_stats.bytes_promoted),
        smi(u.gc_stats.marked_bytes_last),
        smi(u.gc_stats.context_allocs),
    ];
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, fields.len());
    for (i, &v) in fields.iter().enumerate() {
        result.at_put(i, v);
    }
    let _ = args;
    PrimResult::Ok(result.oop())
}

/// `__vmStats` (S15 A8 dev hook): answers a 16-element Array of smis in
/// PINNED order â€” `(icMisses picExtends megaTransitions compilations
/// recompiles recompileDeclined deoptCount deoptTrap deoptReturn deoptPoll
/// osrEntries osrDeclined scavengeCount fullGcCount contextsAllocated
/// bytesPromoted)` â€” the tier/dispatch/speculation twin of `gcStats`
/// (same positional-array convention, same reason: harnesses index by
/// position, so the order is load-bearing). Values are captured BEFORE
/// the result Array allocates, so the allocation cannot skew them beyond
/// its own GC side effects (which land AFTER the snapshot â€” acceptable
/// for a diagnostics hook).
fn prim_vm_stats(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let smi = |v: u64| SmallInt::new(v as i64).oop();
    let st = vm.stats;
    let (g_scav, g_full, g_ctx, g_promoted) = (
        vm.universe.gc_stats.scavenge_count,
        vm.universe.gc_stats.full_gc_count,
        vm.universe.gc_stats.context_allocs,
        vm.universe.gc_stats.bytes_promoted,
    );
    let fields = [
        smi(st.ic_misses),
        smi(st.pic_extends),
        smi(st.mega_transitions),
        smi(st.compilations),
        smi(st.recompiles),
        smi(st.recompile_declined_ineffective),
        smi(st.deopt_count),
        smi(st.deopt_by_reason[0]),
        smi(st.deopt_by_reason[1]),
        smi(st.deopt_by_reason[2]),
        smi(st.osr_entries),
        smi(st.osr_declined),
        smi(g_scav),
        smi(g_full),
        smi(g_ctx),
        smi(g_promoted),
    ];
    let array_klass = vm.universe.array_klass;
    let result = alloc::alloc_indexable_oops(vm, array_klass, fields.len());
    for (i, &v) in fields.iter().enumerate() {
        result.at_put(i, v);
    }
    let _ = args;
    PrimResult::Ok(result.oop())
}

/// SPEC Â§6.3: prints the message + a VM stack trace, terminates. Never
/// returns (its Rust return type is only `PrimResult` so it fits the
/// `PrimFn` signature â€” `std::process::exit`'s `!` unifies with it).
/// DBG4 `halt` (Object) â€” the programmer's `debugger;` statement: open the
/// debugger HERE when one is armed, then continue normally (unlike `error:`,
/// `halt` is NOT terminal â€” a resume falls through and returns self). A no-op
/// when no debugger is active. `session_depth` already guards the debugger's
/// own print-eval doits from recursively halting.
fn prim_halt(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    if vm.debug.active && vm.debug.session_depth == 0 {
        if let Some(m) = vm.regs.method {
            let bci = vm.regs.bci;
            crate::runtime::debug::halt(vm, m, bci, crate::runtime::debug::HaltReason::Breakpoint);
        }
    }
    PrimResult::Ok(args[0]) // answer self
}

fn prim_error(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let text = match ByteArrayOop::try_from(args[1]) {
        Some(b) => {
            let mut buf = Vec::new();
            b.copy_bytes_out(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        }
        None => "(non-string error message)".to_string(),
    };
    let _ = writeln!(vm.out, "Error: {text}");
    crate::runtime::error::print_stack_trace(vm);
    let _ = vm.out.flush();
    // DBG1 (docs/DEBUGGER.md Â§3.1): when the debugger is active, a fatal
    // guest error becomes an inspectable stop â€” the halt loop opens on the
    // erring activation. The error stays terminal on resume (`error:` has
    // no proceed semantics in v1); the halt is for looking, not healing.
    if crate::runtime::debug::wants_error_halt(vm) {
        if let Some(m) = vm.regs.method {
            let bci = vm.regs.bci;
            crate::runtime::debug::halt(
                vm,
                m,
                bci,
                crate::runtime::debug::HaltReason::GuestError(format!("Error: {text}")),
            );
        }
    }
    // DBG0 (docs/DEBUGGER.md Â§4.1): a fatal guest error gets the PROBE
    // mini-dossier â€” walkback + tier-link/anchor state + recent-history
    // ring + heap verify. No signal machinery: the interpreter is coherent
    // here. `MACVM_PROBE=off` silences it for exact-stderr golden tests.
    // Skipped when a recovery is actually about to happen (embedded
    // `VmHandle::eval`, `deopt_trap::raise_guest_fatal` below) â€” see
    // `dnu_fallback`'s identical reasoning.
    if !crate::codecache::deopt_trap::has_registered_jmp_slot()
        && crate::runtime::probe::guest_report_enabled()
    {
        crate::runtime::probe::fatal_guest_report(vm, &format!("error: {text}"));
    }
    // An error curtails everything between here and the entry frame, so the
    // armed ensure:/ifCurtailed: blocks must run â€” the jump below is a raw
    // register restore that would otherwise skip straight past them. After
    // the report above, so the error's own diagnosis stays intact.
    crate::interpreter::unwind::run_curtailment_blocks_on_error(vm);
    crate::codecache::deopt_trap::raise_guest_fatal(format!("Error: {text}"))
}

fn prim_quit_colon(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let code = SmallInt::try_from(args[1])
        .map(|s| s.value() as i32)
        .unwrap_or(0);
    vm.exit_requested = true;
    vm.exit_code = Some(code);
    PrimResult::Ok(args[0])
}

// --- Double group (S6, SPEC Â§1.3) --------------------------------------------

fn double2(
    args: &[Oop],
) -> Option<(
    crate::oops::wrappers::DoubleOop,
    crate::oops::wrappers::DoubleOop,
)> {
    Some((
        crate::oops::wrappers::DoubleOop::try_from(args[0])?,
        crate::oops::wrappers::DoubleOop::try_from(args[1])?,
    ))
}

fn prim_double_add(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        Some((a, b)) => PrimResult::Ok(alloc::alloc_double(vm, a.value() + b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_double_sub(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        Some((a, b)) => PrimResult::Ok(alloc::alloc_double(vm, a.value() - b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_double_mul(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        Some((a, b)) => PrimResult::Ok(alloc::alloc_double(vm, a.value() * b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_double_div(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        // IEEE division by zero yields inf/nan, never fails (pinned,
        // sprint_s06_detail.md Â§08 Double) â€” only a non-Double arg fails.
        Some((a, b)) => PrimResult::Ok(alloc::alloc_double(vm, a.value() / b.value()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_double_lt(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        Some((a, b)) => PrimResult::Ok(bool_oop(vm, a.value() < b.value())),
        None => PrimResult::Fail,
    }
}

fn prim_double_eq(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match double2(args) {
        Some((a, b)) => PrimResult::Ok(bool_oop(vm, a.value() == b.value())),
        None => PrimResult::Fail,
    }
}

/// libm transcendentals â€” one shape, six functions: unbox the Double
/// receiver, call the (state-preserving, AAPCS64) libm routine via Rust's
/// f64 methods, box the result. Fails only on a non-Double receiver.
macro_rules! prim_double_unary {
    ($name:ident, $method:ident) => {
        fn $name(vm: &mut VmState, args: &[Oop]) -> PrimResult {
            match crate::oops::wrappers::DoubleOop::try_from(args[0]) {
                Some(a) => PrimResult::Ok(alloc::alloc_double(vm, a.value().$method()).oop()),
                None => PrimResult::Fail,
            }
        }
    };
}
prim_double_unary!(prim_double_sin, sin);
prim_double_unary!(prim_double_cos, cos);
prim_double_unary!(prim_double_tan, tan);
prim_double_unary!(prim_double_exp, exp);
prim_double_unary!(prim_double_ln, ln);
prim_double_unary!(prim_double_atan, atan);

// â”€â”€ SIMD Float64x2 helpers + primitives (docs/SIMD.md) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Read the two f64 lanes of a Float64x2 receiver â€” checks the KLASS, not the
// format: Float64x2 shares `Format::Double` with a scalar Double (both raw
// bodies), so a format-based wrapper would silently read a Double's single
// lane as a vector. body_word_raw{0,1} are the two 16-byte-body lanes.
fn as_float64x2(vm: &VmState, o: Oop) -> Option<(f64, f64)> {
    let m = MemOop::try_from(o)?;
    if m.klass().oop().raw() != vm.universe.float64x2_klass.oop().raw() {
        return None;
    }
    Some((
        f64::from_bits(m.body_word_raw(0)),
        f64::from_bits(m.body_word_raw(1)),
    ))
}

/// A scalar lane argument (a constructor operand): a Double, or a
/// SmallInteger coerced â€” so `Float64x2 x: 1 y: 2` is as friendly as
/// `x: 1.0 y: 2.0`.
fn as_scalar_f64(vm: &VmState, o: Oop) -> Option<f64> {
    if let Some(n) = SmallInt::try_from(o) {
        return Some(n.value() as f64);
    }
    let m = MemOop::try_from(o)?;
    if m.klass().oop().raw() != vm.universe.double_klass.oop().raw() {
        return None;
    }
    Some(f64::from_bits(m.body_word_raw(0)))
}

/// `Float64x2 x: a y: b` (class-side; args = [class, a, b]).
fn prim_x2_xy(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match (as_scalar_f64(vm, args[1]), as_scalar_f64(vm, args[2])) {
        (Some(a), Some(b)) => PrimResult::Ok(alloc::alloc_float64x2(vm, a, b)),
        _ => PrimResult::Fail,
    }
}

/// `Float64x2 splat: v` â€” broadcast one scalar to both lanes.
fn prim_x2_splat(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match as_scalar_f64(vm, args[1]) {
        Some(v) => PrimResult::Ok(alloc::alloc_float64x2(vm, v, v)),
        None => PrimResult::Fail,
    }
}

/// Elementwise binary op on two Float64x2 â€” each lane a single IEEE f64 op,
/// bit-identical to the per-lane scalar Double op (the invariant the JIT
/// vector fast-path must also honour).
macro_rules! prim_x2_binop {
    ($name:ident, $op:tt) => {
        fn $name(vm: &mut VmState, args: &[Oop]) -> PrimResult {
            match (as_float64x2(vm, args[0]), as_float64x2(vm, args[1])) {
                (Some((a0, a1)), Some((b0, b1))) => {
                    PrimResult::Ok(alloc::alloc_float64x2(vm, a0 $op b0, a1 $op b1))
                }
                _ => PrimResult::Fail,
            }
        }
    };
}
prim_x2_binop!(prim_x2_add, +);
prim_x2_binop!(prim_x2_sub, -);
prim_x2_binop!(prim_x2_mul, *);
prim_x2_binop!(prim_x2_div, /);

/// `aFloat64x2 at: i` â€” lane 1 or 2 as a Double (1-based, Smalltalk).
fn prim_x2_at(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (a, b) = match as_float64x2(vm, args[0]) {
        Some(v) => v,
        None => return PrimResult::Fail,
    };
    let i = match SmallInt::try_from(args[1]) {
        Some(n) => n.value(),
        None => return PrimResult::Fail,
    };
    let lane = match i {
        1 => a,
        2 => b,
        _ => return PrimResult::Fail,
    };
    PrimResult::Ok(alloc::alloc_double(vm, lane).oop())
}

// â”€â”€ SIMD Float32x4 helpers + primitives (docs/SIMD.md) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Read the four f32 lanes of a Float32x4 receiver (KLASS-checked, like
// Float64x2 â€” it too shares Format::Double). Lanes unpack in NEON `.4s`
// element order: lane 0/1 = low/high 32 bits of body word 0, lane 2/3 = word
// 1. This MUST match `alloc::alloc_float32x4`'s packing and the `ldr q`/`.4s`
// the JIT fuse uses, so the interpreter and compiled tiers agree bit-for-bit.
fn as_float32x4(vm: &VmState, o: Oop) -> Option<[f32; 4]> {
    let m = MemOop::try_from(o)?;
    if m.klass().oop().raw() != vm.universe.float32x4_klass.oop().raw() {
        return None;
    }
    let w0 = m.body_word_raw(0);
    let w1 = m.body_word_raw(1);
    Some([
        f32::from_bits(w0 as u32),
        f32::from_bits((w0 >> 32) as u32),
        f32::from_bits(w1 as u32),
        f32::from_bits((w1 >> 32) as u32),
    ])
}

/// A scalar lane argument narrowed to f32 (a constructor operand): a Double or
/// a SmallInteger, rounded to f32 â€” the SAME single-precision rounding the
/// `.4s` lanes carry.
fn as_scalar_f32(vm: &VmState, o: Oop) -> Option<f32> {
    as_scalar_f64(vm, o).map(|v| v as f32)
}

/// `Float32x4 x: a y: b z: c w: d` (class-side; args = [class, a, b, c, d]).
fn prim_x4_xyzw(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match (
        as_scalar_f32(vm, args[1]),
        as_scalar_f32(vm, args[2]),
        as_scalar_f32(vm, args[3]),
        as_scalar_f32(vm, args[4]),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => {
            PrimResult::Ok(alloc::alloc_float32x4(vm, [a, b, c, d]))
        }
        _ => PrimResult::Fail,
    }
}

/// `Float32x4 splat: v` â€” broadcast one scalar to all four lanes.
fn prim_x4_splat(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match as_scalar_f32(vm, args[1]) {
        Some(v) => PrimResult::Ok(alloc::alloc_float32x4(vm, [v, v, v, v])),
        None => PrimResult::Fail,
    }
}

/// Elementwise binary op on two Float32x4 â€” each lane a single IEEE f32 op,
/// bit-identical to the `.4s` NEON lane (the invariant the JIT fuse honours).
macro_rules! prim_x4_binop {
    ($name:ident, $op:tt) => {
        fn $name(vm: &mut VmState, args: &[Oop]) -> PrimResult {
            match (as_float32x4(vm, args[0]), as_float32x4(vm, args[1])) {
                (Some(a), Some(b)) => PrimResult::Ok(alloc::alloc_float32x4(
                    vm,
                    [a[0] $op b[0], a[1] $op b[1], a[2] $op b[2], a[3] $op b[3]],
                )),
                _ => PrimResult::Fail,
            }
        }
    };
}
prim_x4_binop!(prim_x4_add, +);
prim_x4_binop!(prim_x4_sub, -);
prim_x4_binop!(prim_x4_mul, *);
prim_x4_binop!(prim_x4_div, /);

/// `aFloat32x4 at: i` â€” lane 1..4 as a Double (exact f32â†’f64 widening, no
/// rounding). 1-based (Smalltalk convention).
fn prim_x4_at(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let lanes = match as_float32x4(vm, args[0]) {
        Some(v) => v,
        None => return PrimResult::Fail,
    };
    let i = match SmallInt::try_from(args[1]) {
        Some(n) => n.value(),
        None => return PrimResult::Fail,
    };
    if !(1..=4).contains(&i) {
        return PrimResult::Fail;
    }
    PrimResult::Ok(alloc::alloc_double(vm, lanes[(i - 1) as usize] as f64).oop())
}

// â”€â”€ SIMD Int32x4 helpers + primitives (docs/SIMD.md) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Read the four i32 lanes of an Int32x4 (KLASS-checked). Lanes unpack in NEON
// `.4s` element order, identical to Float32x4's layout â€” only the type
// (32-bit two's-complement integer) differs. Arithmetic WRAPS on 32-bit
// overflow, matching NEON `add/sub/mul v.4s` (a fixed-width lane is not a
// promote-to-BigInt Smalltalk integer â€” you asked for 32-bit lanes).
fn as_int32x4(vm: &VmState, o: Oop) -> Option<[i32; 4]> {
    let m = MemOop::try_from(o)?;
    if m.klass().oop().raw() != vm.universe.int32x4_klass.oop().raw() {
        return None;
    }
    let w0 = m.body_word_raw(0);
    let w1 = m.body_word_raw(1);
    Some([
        w0 as u32 as i32,
        (w0 >> 32) as u32 as i32,
        w1 as u32 as i32,
        (w1 >> 32) as u32 as i32,
    ])
}

/// A scalar lane argument truncated to a 32-bit lane: a SmallInteger's low 32
/// bits (C-style narrowing â€” consistent with the wrapping arithmetic).
fn as_scalar_i32(o: Oop) -> Option<i32> {
    SmallInt::try_from(o).map(|n| n.value() as i32)
}

/// `Int32x4 x: a y: b z: c w: d` (class-side; args = [class, a, b, c, d]).
fn prim_i4_xyzw(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match (
        as_scalar_i32(args[1]),
        as_scalar_i32(args[2]),
        as_scalar_i32(args[3]),
        as_scalar_i32(args[4]),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => {
            PrimResult::Ok(alloc::alloc_int32x4(vm, [a, b, c, d]))
        }
        _ => PrimResult::Fail,
    }
}

/// `Int32x4 splat: v` â€” broadcast one integer to all four lanes.
fn prim_i4_splat(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match as_scalar_i32(args[1]) {
        Some(v) => PrimResult::Ok(alloc::alloc_int32x4(vm, [v, v, v, v])),
        None => PrimResult::Fail,
    }
}

/// Elementwise WRAPPING integer op on two Int32x4 â€” each lane a single 32-bit
/// two's-complement op, bit-identical to the `.4s` NEON lane (the invariant the
/// JIT fuse honours). NO divide: NEON has no vector integer divide.
macro_rules! prim_i4_binop {
    ($name:ident, $op:ident) => {
        fn $name(vm: &mut VmState, args: &[Oop]) -> PrimResult {
            match (as_int32x4(vm, args[0]), as_int32x4(vm, args[1])) {
                (Some(a), Some(b)) => PrimResult::Ok(alloc::alloc_int32x4(
                    vm,
                    [
                        a[0].$op(b[0]),
                        a[1].$op(b[1]),
                        a[2].$op(b[2]),
                        a[3].$op(b[3]),
                    ],
                )),
                _ => PrimResult::Fail,
            }
        }
    };
}
prim_i4_binop!(prim_i4_add, wrapping_add);
prim_i4_binop!(prim_i4_sub, wrapping_sub);
prim_i4_binop!(prim_i4_mul, wrapping_mul);

/// `anInt32x4 at: i` â€” lane 1..4 as a SmallInteger (an i32 always fits).
fn prim_i4_at(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let lanes = match as_int32x4(vm, args[0]) {
        Some(v) => v,
        None => return PrimResult::Fail,
    };
    let i = match SmallInt::try_from(args[1]) {
        Some(n) => n.value(),
        None => return PrimResult::Fail,
    };
    if !(1..=4).contains(&i) {
        return PrimResult::Fail;
    }
    PrimResult::Ok(SmallInt::new(lanes[(i - 1) as usize] as i64).oop())
}

// â”€â”€ SIMD level 2: FloatArray + NEON bulk kernels (docs/SIMD.md Part E) â”€â”€â”€â”€â”€
//
// A FloatArray is a Format::IndexableBytes buffer of N f64 lanes (N*8 bytes,
// GC-skipped body). Lane j (0-based) lives at body word (1 + j) â€” word 0 is
// the byte-count size slot (alloc_indexable_bytes). Klass-checked, like the
// vector value classes.
fn as_float_array(vm: &VmState, o: Oop) -> Option<MemOop> {
    let m = MemOop::try_from(o)?;
    (m.klass().oop().raw() == vm.universe.float_array_klass.oop().raw()).then_some(m)
}

/// f64 lane count = byte length / 8.
fn float_array_len(m: MemOop) -> usize {
    m.indexable_len() / 8
}

/// Copy the lanes out into an owned `Vec<f64>` â€” done BEFORE any result
/// allocation so a scavenge can't leave a dangling body pointer (the vector
/// value classes' GC lesson). The kernels then run on the slice, which LLVM
/// auto-vectorizes to NEON `fadd v.2d` etc.
fn float_array_lanes(m: MemOop) -> Vec<f64> {
    let n = float_array_len(m);
    (0..n)
        .map(|j| f64::from_bits(m.body_word_raw(1 + j)))
        .collect()
}

// The FloatArray bulk kernels are EXPLICIT hand-written NEON â€” see
// `crate::runtime::simd_kernels` (the one module allowed `unsafe` for hardware
// intrinsics). NOT a scalar loop left to rustc/LLVM to maybe vectorize: a
// `<primitive:>` bulk op deliberately uses the hardware (docs/SIMD.md Part E).
use crate::runtime::simd_kernels::{
    neon_add, neon_max, neon_min, neon_scale, pairwise_dot, pairwise_sum,
};

/// `FloatArray new: n` (class-side; args = [class, n]) â€” n zeroed f64 lanes.
fn prim_farray_new(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let n = match SmallInt::try_from(args[1]) {
        Some(k) if k.value() >= 0 => k.value() as usize,
        _ => return PrimResult::Fail,
    };
    let nbytes = match n.checked_mul(8) {
        Some(b) => b,
        None => return PrimResult::Fail,
    };
    let klass = vm.universe.float_array_klass;
    PrimResult::Ok(alloc::alloc_indexable_bytes(vm, klass, nbytes).oop())
}

/// `FloatArray >> size` â†’ lane count.
fn prim_farray_size(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match as_float_array(vm, args[0]) {
        Some(m) => PrimResult::Ok(SmallInt::new(float_array_len(m) as i64).oop()),
        None => PrimResult::Fail,
    }
}

/// `FloatArray >> at: i` â†’ the i-th lane as a Double (1-based).
fn prim_farray_at(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let m = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    let i = match SmallInt::try_from(args[1]) {
        Some(k) => k.value(),
        None => return PrimResult::Fail,
    };
    if i < 1 || i as usize > float_array_len(m) {
        return PrimResult::Fail;
    }
    let lane = f64::from_bits(m.body_word_raw(1 + (i as usize - 1)));
    PrimResult::Ok(alloc::alloc_double(vm, lane).oop())
}

/// `FloatArray >> at: i put: aDouble` â†’ aDouble (1-based; SmallInteger coerced).
fn prim_farray_at_put(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let m = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    let i = match SmallInt::try_from(args[1]) {
        Some(k) => k.value(),
        None => return PrimResult::Fail,
    };
    if i < 1 || i as usize > float_array_len(m) {
        return PrimResult::Fail;
    }
    let v = match as_scalar_f64(vm, args[2]) {
        Some(v) => v,
        None => return PrimResult::Fail,
    };
    m.set_body_word_raw(1 + (i as usize - 1), v.to_bits());
    PrimResult::Ok(args[2])
}

/// `FloatArray >> +@ other` â†’ a NEW FloatArray of the elementwise sums.
/// Per-lane exact (bit-identical to scalar Double add â€” the elementwise
/// discipline, docs/SIMD.md Â§B4). Fails on a length mismatch.
fn prim_farray_add(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (ma, mb) = match (as_float_array(vm, args[0]), as_float_array(vm, args[1])) {
        (Some(a), Some(b)) => (a, b),
        _ => return PrimResult::Fail,
    };
    let n = float_array_len(ma);
    if n != float_array_len(mb) {
        return PrimResult::Fail;
    }
    // Copy both operands out BEFORE allocating the result (the alloc may GC).
    let a = float_array_lanes(ma);
    let b = float_array_lanes(mb);
    let mut c = vec![0.0f64; n];
    neon_add(&a, &b, &mut c); // explicit `fadd v.2d` stream + scalar tail
    let klass = vm.universe.float_array_klass;
    let out = alloc::alloc_indexable_bytes(vm, klass, n * 8);
    // No allocation between here and the last write â€” `out` cannot move.
    for (i, &ci) in c.iter().enumerate() {
        out.as_mem().set_body_word_raw(1 + i, ci.to_bits());
    }
    PrimResult::Ok(out.oop())
}

/// `FloatArray >> sum` â†’ Double (fast pairwise NEON reduction, docs/SIMD.md D).
fn prim_farray_sum(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let m = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    let a = float_array_lanes(m);
    PrimResult::Ok(alloc::alloc_double(vm, pairwise_sum(&a)).oop())
}

/// `FloatArray >> dot: other` â†’ Double (fast pairwise NEON reduction). Fails
/// on a length mismatch.
fn prim_farray_dot(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let (ma, mb) = match (as_float_array(vm, args[0]), as_float_array(vm, args[1])) {
        (Some(a), Some(b)) => (a, b),
        _ => return PrimResult::Fail,
    };
    let n = float_array_len(ma);
    if n != float_array_len(mb) {
        return PrimResult::Fail;
    }
    let a = float_array_lanes(ma);
    let b = float_array_lanes(mb);
    PrimResult::Ok(alloc::alloc_double(vm, pairwise_dot(&a, &b)).oop())
}

/// `FloatArray >> scale: aNumber` â†’ a NEW FloatArray of the lanes times the
/// scalar (`explicit `fmul v.2d`; per-lane bit-identical to scalar multiply).
fn prim_farray_scale(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let ma = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    let k = match as_scalar_f64(vm, args[1]) {
        Some(k) => k,
        None => return PrimResult::Fail,
    };
    let n = float_array_len(ma);
    // Copy out BEFORE allocating the result (the alloc may GC).
    let a = float_array_lanes(ma);
    let mut c = vec![0.0f64; n];
    neon_scale(&a, k, &mut c);
    let klass = vm.universe.float_array_klass;
    let out = alloc::alloc_indexable_bytes(vm, klass, n * 8);
    for (i, &ci) in c.iter().enumerate() {
        out.as_mem().set_body_word_raw(1 + i, ci.to_bits());
    }
    PrimResult::Ok(out.oop())
}

/// `FloatArray >> max` â†’ Double, the largest lane (explicit `fmax v.2d`
/// reduction; order-independent, so bit-exact). Fails on an EMPTY array (no
/// maximum) â€” the world method turns that into a sensible error.
fn prim_farray_max(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let m = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    if float_array_len(m) == 0 {
        return PrimResult::Fail;
    }
    let a = float_array_lanes(m);
    PrimResult::Ok(alloc::alloc_double(vm, neon_max(&a)).oop())
}

/// `FloatArray >> min` â†’ Double, the smallest lane (`fmin v.2d`). Fails on empty.
fn prim_farray_min(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let m = match as_float_array(vm, args[0]) {
        Some(m) => m,
        None => return PrimResult::Fail,
    };
    if float_array_len(m) == 0 {
        return PrimResult::Fail;
    }
    let a = float_array_lanes(m);
    PrimResult::Ok(alloc::alloc_double(vm, neon_min(&a)).oop())
}

fn prim_double_sqrt(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match crate::oops::wrappers::DoubleOop::try_from(args[0]) {
        Some(a) => PrimResult::Ok(alloc::alloc_double(vm, a.value().sqrt()).oop()),
        None => PrimResult::Fail,
    }
}

fn prim_double_floor(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match crate::oops::wrappers::DoubleOop::try_from(args[0]) {
        Some(a) => {
            let f = a.value().floor();
            if !f.is_finite() || f < SMI_MIN as f64 || f > SMI_MAX as f64 {
                return PrimResult::Fail;
            }
            PrimResult::Ok(SmallInt::new(f as i64).oop())
        }
        None => PrimResult::Fail,
    }
}

fn prim_smi_as_double(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    match SmallInt::try_from(args[0]) {
        Some(s) => PrimResult::Ok(alloc::alloc_double(vm, s.value() as f64).oop()),
        None => PrimResult::Fail,
    }
}

/// Shortest round-trip decimal text for a Double (SPEC Â§1.3's `printOn:`
/// support) â€” mirrors `memory::print::print_f64` but without that
/// function's debug-printer framing (no `nan`/`inf` word wrapping beyond
/// what Rust's own `Display` gives; String result, not a Rust `String`
/// consumed by a printer).
fn prim_double_print_digits(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(d) = crate::oops::wrappers::DoubleOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let v = d.value();
    let text = if v.is_nan() {
        "nan".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    };
    let klass = vm.universe.string_klass;
    let b = alloc::alloc_indexable_bytes(vm, klass, text.len());
    for (i, byte) in text.bytes().enumerate() {
        b.byte_at_put(i, byte);
    }
    PrimResult::Ok(b.oop())
}

// --- Symbol group (S6) -----------------------------------------------------

fn prim_string_as_symbol(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(b) = ByteArrayOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    let mut buf = Vec::new();
    b.copy_bytes_out(&mut buf);
    PrimResult::Ok(vm.universe.intern(&buf).oop())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oops::layout::SMI_MAX;
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

    fn smi(v: i64) -> Oop {
        SmallInt::new(v).oop()
    }

    fn call(id: u16, vm: &mut VmState, args: &[Oop]) -> PrimResult {
        (prim_by_id(id).unwrap().f)(vm, args)
    }

    /// SIMD Float64x2 (`docs/SIMD.md`): elementwise ops are bit-identical to
    /// per-lane scalar Double arithmetic, and the raw 16-byte vector bodies
    /// survive GC. The GC point is a REGRESSION LOCK: `float64x2_klass` must be
    /// a GC root (`memory::roots`) â€” it was missed on first wiring, and a
    /// moved-but-not-updated klass pointer read poison mid-alloc. The
    /// alloc-churn loop under `gc_stress` forces the exact scavenge that
    /// caught it.
    fn x2_lane(vm: &mut VmState, v: Oop, i: i64) -> f64 {
        match call(133, vm, &[v, smi(i)]) {
            PrimResult::Ok(o) => f64::from_bits(MemOop::try_from(o).unwrap().body_word_raw(0)),
            other => panic!("x2 at: failed: {other:?}"),
        }
    }

    #[test]
    fn float64x2_arithmetic_and_gc_survival() {
        // Part 1 â€” bit-identity, gc_stress OFF: a handful of allocations
        // trigger no GC, so the raw local oops stay put; each lane must equal
        // the per-lane scalar Double op.
        let mut vm = test_vm();
        let va = alloc::alloc_float64x2(&mut vm, 1.5, 2.5);
        let vb = alloc::alloc_float64x2(&mut vm, 2.0, 4.0);
        let prod = match call(131, &mut vm, &[va, vb]) {
            PrimResult::Ok(o) => o,
            other => panic!("x2 * failed: {other:?}"),
        };
        let half = alloc::alloc_float64x2(&mut vm, 0.5, 0.5);
        let sum = match call(129, &mut vm, &[prod, half]) {
            PrimResult::Ok(o) => o,
            other => panic!("x2 + failed: {other:?}"),
        };
        assert_eq!(x2_lane(&mut vm, sum, 1), 1.5 * 2.0 + 0.5);
        assert_eq!(x2_lane(&mut vm, sum, 2), 2.5 * 4.0 + 0.5);

        // Part 2 â€” GC-root survival (the regression lock). Root a vector on
        // the process stack (a real GC root, as the interpreter does), force
        // scavenges that MOVE the klass and the vector, and allocate a NEW
        // vector across each â€” that alloc reads `float64x2_klass`, which must
        // stay a live root or it poison-reads a moved klass. The rooted
        // vector's lanes must survive intact.
        let slot = vm.stack.sp;
        let keep = alloc::alloc_float64x2(&mut vm, 42.0, 99.0);
        vm.stack.push(keep);
        let nil = vm.universe.nil_obj;
        for _ in 0..8 {
            call_rooted(93 /* gcScavenge */, &mut vm, nil);
            let _fresh = alloc::alloc_float64x2(&mut vm, 1.0, 2.0); // uses the (moved) klass
        }
        let moved = vm.stack.get(slot); // the GC updated this root in place
        assert_eq!(x2_lane(&mut vm, moved, 1), 42.0);
        assert_eq!(x2_lane(&mut vm, moved, 2), 99.0);
    }

    /// SIMD Float32x4 (`docs/SIMD.md`): the 4-lane f32 companion. Each lane is
    /// a single-precision op â€” bit-identical to the `.4s` NEON lane the JIT
    /// fuse emits (all test values are exact in f32, so `at:`'s f32â†’f64
    /// widening is lossless and the asserts are sharp). Same GC-root
    /// regression lock as Float64x2: `float32x4_klass` must be in
    /// `memory::roots`, or an alloc after a scavenge poison-reads a moved klass.
    fn x4_lane(vm: &mut VmState, v: Oop, i: i64) -> f64 {
        match call(140, vm, &[v, smi(i)]) {
            PrimResult::Ok(o) => f64::from_bits(MemOop::try_from(o).unwrap().body_word_raw(0)),
            other => panic!("x4 at: failed: {other:?}"),
        }
    }

    #[test]
    fn float32x4_arithmetic_and_gc_survival() {
        // Part 1 â€” bit-identity, gc_stress OFF.
        let mut vm = test_vm();
        let va = alloc::alloc_float32x4(&mut vm, [1.5, 2.5, 3.5, 4.5]);
        let vb = alloc::alloc_float32x4(&mut vm, [2.0, 4.0, 8.0, 16.0]);
        let prod = match call(138 /* x4 * */, &mut vm, &[va, vb]) {
            PrimResult::Ok(o) => o,
            other => panic!("x4 * failed: {other:?}"),
        };
        let half = alloc::alloc_float32x4(&mut vm, [0.5, 0.5, 0.5, 0.5]);
        let sum = match call(136 /* x4 + */, &mut vm, &[prod, half]) {
            PrimResult::Ok(o) => o,
            other => panic!("x4 + failed: {other:?}"),
        };
        assert_eq!(x4_lane(&mut vm, sum, 1), ((1.5f32 * 2.0) + 0.5) as f64);
        assert_eq!(x4_lane(&mut vm, sum, 2), ((2.5f32 * 4.0) + 0.5) as f64);
        assert_eq!(x4_lane(&mut vm, sum, 3), ((3.5f32 * 8.0) + 0.5) as f64);
        assert_eq!(x4_lane(&mut vm, sum, 4), ((4.5f32 * 16.0) + 0.5) as f64);

        // Part 2 â€” GC-root survival (the regression lock, as for Float64x2).
        let slot = vm.stack.sp;
        let keep = alloc::alloc_float32x4(&mut vm, [42.0, 99.0, -7.0, 3.25]);
        vm.stack.push(keep);
        let nil = vm.universe.nil_obj;
        for _ in 0..8 {
            call_rooted(93 /* gcScavenge */, &mut vm, nil);
            let _fresh = alloc::alloc_float32x4(&mut vm, [1.0, 2.0, 3.0, 4.0]);
        }
        let moved = vm.stack.get(slot);
        assert_eq!(x4_lane(&mut vm, moved, 1), 42.0);
        assert_eq!(x4_lane(&mut vm, moved, 2), 99.0);
        assert_eq!(x4_lane(&mut vm, moved, 3), -7.0);
        assert_eq!(x4_lane(&mut vm, moved, 4), 3.25);
    }

    /// SIMD Int32x4 (`docs/SIMD.md`): 4-lane integer arithmetic WRAPS on 32-bit
    /// overflow (matching NEON `add/mul v.4s`), lane access answers a
    /// SmallInteger, and the raw 16-byte bodies survive GC (same `int32x4_klass`
    /// GC-root regression lock as the other value classes).
    fn i4_lane(vm: &mut VmState, v: Oop, i: i64) -> i64 {
        match call(153, vm, &[v, smi(i)]) {
            PrimResult::Ok(o) => SmallInt::try_from(o).unwrap().value(),
            other => panic!("i4 at: failed: {other:?}"),
        }
    }

    #[test]
    fn int32x4_wrapping_arithmetic_and_gc_survival() {
        // Part 1 â€” arithmetic incl. 32-bit wrap, gc_stress OFF.
        let mut vm = test_vm();
        let a = alloc::alloc_int32x4(&mut vm, [1, 2, 3, i32::MAX]);
        let b = alloc::alloc_int32x4(&mut vm, [10, 20, 30, 1]);
        let sum = match call(150 /* + */, &mut vm, &[a, b]) {
            PrimResult::Ok(o) => o,
            other => panic!("i4 + failed: {other:?}"),
        };
        assert_eq!(i4_lane(&mut vm, sum, 1), 11);
        assert_eq!(i4_lane(&mut vm, sum, 2), 22);
        assert_eq!(i4_lane(&mut vm, sum, 3), 33);
        // i32::MAX + 1 wraps to i32::MIN.
        assert_eq!(i4_lane(&mut vm, sum, 4), i32::MIN as i64);
        let prod = match call(152 /* * */, &mut vm, &[a, b]) {
            PrimResult::Ok(o) => o,
            other => panic!("i4 * failed: {other:?}"),
        };
        assert_eq!(i4_lane(&mut vm, prod, 1), 10);
        assert_eq!(i4_lane(&mut vm, prod, 4), i32::MAX as i64); // MAX*1

        // Part 2 â€” GC-root survival (int32x4_klass must be a GC root).
        let slot = vm.stack.sp;
        let keep = alloc::alloc_int32x4(&mut vm, [42, -99, 7, -1]);
        vm.stack.push(keep);
        let nil = vm.universe.nil_obj;
        for _ in 0..8 {
            call_rooted(93 /* gcScavenge */, &mut vm, nil);
            let _fresh = alloc::alloc_int32x4(&mut vm, [1, 2, 3, 4]);
        }
        let moved = vm.stack.get(slot);
        assert_eq!(i4_lane(&mut vm, moved, 1), 42);
        assert_eq!(i4_lane(&mut vm, moved, 2), -99);
        assert_eq!(i4_lane(&mut vm, moved, 3), 7);
        assert_eq!(i4_lane(&mut vm, moved, 4), -1);
    }

    /// SIMD level 2 (`docs/SIMD.md` Part E): the FloatArray NEON bulk-kernel
    /// primitives (`+@`/`sum`/`dot:`) compute correctly, and `float_array_klass`
    /// is a GC root (same regression lock as the value classes â€” a moved-but-
    /// unrooted klass poison-reads mid-alloc). `sum`/`dot:` verify against the
    /// DEFINED pairwise order, NOT a scalar fold (docs/SIMD.md Part D).
    fn make_farray(vm: &mut VmState, xs: &[f64]) -> Oop {
        let k = vm.universe.float_array_klass;
        let arr = alloc::alloc_indexable_bytes(vm, k, xs.len() * 8);
        for (i, &x) in xs.iter().enumerate() {
            arr.as_mem().set_body_word_raw(1 + i, x.to_bits());
        }
        arr.oop()
    }

    #[test]
    fn float_array_kernels_and_gc_survival() {
        // Part 1 â€” kernel correctness (gc_stress OFF; raw locals stay put).
        let mut vm = test_vm();
        let a = make_farray(&mut vm, &[1.5, 2.5, 3.5, 4.5, 5.5]);
        let b = make_farray(&mut vm, &[0.5, 0.5, 0.5, 0.5, 0.5]);
        // +@ â†’ elementwise (per-lane exact).
        let c = match call(145, &mut vm, &[a, b]) {
            PrimResult::Ok(o) => o,
            other => panic!("+@ failed: {other:?}"),
        };
        for (i, expect) in [2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
            let lane = f64::from_bits(MemOop::try_from(c).unwrap().body_word_raw(1 + i));
            assert_eq!(lane, *expect);
        }
        // sum â†’ the DEFINED pairwise order: (a0+a2+a4) + (a1+a3).
        let sum = match call(146, &mut vm, &[a]) {
            PrimResult::Ok(o) => crate::oops::wrappers::DoubleOop::try_from(o)
                .unwrap()
                .value(),
            other => panic!("sum failed: {other:?}"),
        };
        assert_eq!(sum, (1.5 + 3.5 + 5.5) + (2.5 + 4.5));
        // dot â†’ same pairwise order over the products.
        let dot = match call(147, &mut vm, &[a, b]) {
            PrimResult::Ok(o) => crate::oops::wrappers::DoubleOop::try_from(o)
                .unwrap()
                .value(),
            other => panic!("dot failed: {other:?}"),
        };
        assert_eq!(
            dot,
            (1.5 * 0.5 + 3.5 * 0.5 + 5.5 * 0.5) + (2.5 * 0.5 + 4.5 * 0.5)
        );

        // Part 2 â€” GC-root survival (float_array_klass must be a GC root). Root
        // a FloatArray on the stack, force scavenges that MOVE the klass and the
        // array, and allocate a NEW array across each (that alloc reads the
        // klass). The rooted array's lanes must survive intact.
        let slot = vm.stack.sp;
        let keep = make_farray(&mut vm, &[42.0, 99.0, -7.0]);
        vm.stack.push(keep);
        let nil = vm.universe.nil_obj;
        for _ in 0..8 {
            call_rooted(93 /* gcScavenge */, &mut vm, nil);
            let _fresh = make_farray(&mut vm, &[1.0, 2.0, 3.0, 4.0]);
        }
        let moved = vm.stack.get(slot);
        let m = MemOop::try_from(moved).unwrap();
        assert_eq!(f64::from_bits(m.body_word_raw(1)), 42.0);
        assert_eq!(f64::from_bits(m.body_word_raw(2)), 99.0);
        assert_eq!(f64::from_bits(m.body_word_raw(3)), -7.0);
    }

    /// `tests_s06.md`'s `prim_ids_frozen`: a regression lock on the idâ†’name
    /// map every `.mst` `<primitive: N>` binds against. This registry
    /// deliberately diverges from `sprint_s06_detail.md`'s suggested
    /// numbering (S3/S4 pinned ids 1-61/90-96 first; the doc's own text
    /// permits this) â€” the table below is MY pinned numbering, not the
    /// doc's. Adding/renumbering a primitive must update this table
    /// deliberately, not silently.
    #[test]
    fn prim_ids_frozen() {
        let expected: &[(u16, &str)] = &[
            (1, "+"),
            (2, "-"),
            (3, "*"),
            (4, "//"),
            (5, "\\\\"),
            (6, "bitAnd:"),
            (7, "bitOr:"),
            (8, "bitXor:"),
            (9, "bitShift:"),
            (10, "<"),
            (11, "<="),
            (12, ">"),
            (13, ">="),
            (14, "="),
            (15, "~="),
            (20, "identityHash"),
            (21, "class"),
            (22, "=="),
            (23, "basicNew"),
            (24, "basicNew:"),
            (25, "instVarAt:"),
            (26, "at:"),
            (27, "at:put:"),
            (28, "size"),
            (40, "byteAt:"),
            (41, "byteAt:put:"),
            (42, "byteSize"),
            (43, "replaceFrom:to:with:"),
            (44, "hashBytes"),
            (45, "compare:"),
            (50, "value"),
            (51, "value:"),
            (52, "value:value:"),
            (53, "value:value:value:"),
            (54, "valueWithArguments:"),
            (60, "ensure:"),
            (61, "ifCurtailed:"),
            (62, "primitiveOf:selector:"),
            (63, "methodSends:selector:target:"),
            (64, "perform:withArguments:"),
            (90, "quit"),
            (91, "printOnStdout:"),
            (92, "millisecondClock"),
            (93, "gcScavenge"),
            (94, "gcFull"),
            (95, "error:"),
            (96, "quit:"),
            (97, "gcStats"),
            (98, "allClasses"),
            (99, "selectorsOf:"),
            (100, "+"),
            (101, "-"),
            (102, "*"),
            (103, "/"),
            (104, "<"),
            (105, "="),
            (106, "sqrt"),
            (107, "floor"),
            (108, "asDouble"),
            (109, "printDigits"),
            (110, "asSymbol"),
            (111, "__vmStats"),
            (112, "byteAt:"),
            (113, "byteAt:put:"),
            (114, "signedLongAt:"),
            (115, "signedLongAt:put:"),
            (116, "doubleAt:"),
            (117, "doubleAt:put:"),
            (118, "size"),
            (119, "new:"),
            (120, "forAddress:size:"),
            (121, "sin"),
            (122, "cos"),
            (123, "tan"),
            (124, "exp"),
            (125, "ln"),
            (126, "atan"),
            (127, "x:y:"),
            (128, "splat:"),
            (129, "+"),
            (130, "-"),
            (131, "*"),
            (132, "/"),
            (133, "at:"),
            (134, "x:y:z:w:"),
            (135, "splat:"),
            (136, "+"),
            (137, "-"),
            (138, "*"),
            (139, "/"),
            (140, "at:"),
            (141, "new:"),
            (142, "at:"),
            (143, "at:put:"),
            (144, "size"),
            (145, "+@"),
            (146, "sum"),
            (147, "dot:"),
            (148, "x:y:z:w:"),
            (149, "splat:"),
            (150, "+"),
            (151, "-"),
            (152, "*"),
            (153, "at:"),
            (154, "scale:"),
            (155, "max"),
            (156, "min"),
            (157, "instanceVariablesOf:"),
            (158, "classVariablesOf:"),
            // game group (docs/gamepane_design.md)
            (200, "GamePane>>clearR:g:b:"),
            (201, "GamePane>>paletteAt:r:g:b:"),
            (202, "GamePane>>cls:"),
            (203, "GamePane>>point:y:color:"),
            (204, "GamePane>>line:y:to:y:color:"),
            (205, "GamePane>>fill:y:width:height:color:"),
            (206, "GamePane>>disc:y:radius:color:"),
            (207, "GamePane>>present"),
            (208, "GamePane>>run"),
            (209, "GamePane>>stop"),
            (210, "GamePane>>primDefineSprite:rows:"),
            (211, "GamePane>>primSpriteColor:index:r:g:b:"),
            (212, "GamePane>>primMoveSprite:x:y:"),
            (213, "Sound>>primPlay:"),
            (214, "Tune>>primPlayTune:"),
            (215, "GamePane>>blit:"),
            (220, "Worker class>>primSpawn:"),
            (221, "Worker class>>primSend:corr:bytes:"),
            (222, "Worker class>>primPoll"),
            (223, "Worker class>>primAwaitInbox:"),
            (224, "Worker class>>primTerminate:"),
            (225, "Worker class>>primAlive:"),
            (226, "Worker class>>primSelfId"),
            (227, "Worker class>>pickle:"),
            (228, "Worker class>>unpickle:"),
            // cocoa bridge C0 (docs/cocoa_bridge_design.md)
            (230, "Cocoa class>>primClassNamed:"),
            (231, "ObjcRef>>primSend:"),
            (232, "ObjcRef>>primSend:with:"),
            (233, "ObjcRef>>primSend:with:with:"),
            (234, "ObjcRef>>primSendI64:"),
            (235, "ObjcRef>>primSendString:"),
            (236, "ObjcRef>>primRelease"),
            (237, "Cocoa class>>primNSString:"),
            (238, "Cocoa class>>primDrainPool"),
            (239, "ObjcRef>>primIsValid"),
            (240, "ObjcRef>>primSend:args:ret:"),
            (241, "ObjcRef>>primSendAuto:args:"),
            (242, "ObjcRef>>primSendMainAuto:args:"),
            (243, "Cocoa class>>primNewAction:"),
            (244, "Cocoa class>>primPoolPush"),
            (245, "Cocoa class>>primPoolPop"),
            (246, "respondsTo:"),
            (247, "shallowCopy"),
            (248, "numArgs"),
            (249, "MacvmDelegate class>>primNewDelegate:ticket:"),
            (250, "Worker class>>primEvalDoit:"),
            (251, "halt"),
        ];
        assert_eq!(
            PRIMITIVES.len(),
            expected.len(),
            "PRIMITIVES table grew/shrank without updating prim_ids_frozen"
        );
        for (d, (id, name)) in PRIMITIVES.iter().zip(expected.iter()) {
            assert_eq!(d.id, *id, "id mismatch at {}", d.name);
            assert_eq!(d.name, *name, "name mismatch at id {}", d.id);
        }
    }

    #[test]
    fn vm_stats_prim_shape_and_positions() {
        let mut vm = test_vm();
        vm.stats.ic_misses = 7;
        vm.stats.compilations = 3;
        vm.universe.gc_stats.context_allocs = 9;
        let nil = vm.universe.nil_obj;
        let r = match prim_vm_stats(&mut vm, &[nil]) {
            PrimResult::Ok(o) => crate::oops::wrappers::ArrayOop::try_from(o).expect("an Array"),
            other => panic!("__vmStats must succeed, got {other:?}"),
        };
        assert_eq!(r.len(), 16, "pinned 16-element shape");
        let at = |i: usize| {
            crate::oops::smi::SmallInt::try_from(r.at(i))
                .expect("all smis")
                .value()
        };
        assert_eq!(at(0), 7, "icMisses is position 0 (pinned)");
        assert_eq!(at(3), 3, "compilations is position 3 (pinned)");
        assert_eq!(at(14), 9, "contextsAllocated is position 14 (pinned)");
    }

    #[test]
    fn prim_table_sorted_unique() {
        let mut last: Option<u16> = None;
        for d in PRIMITIVES {
            if let Some(l) = last {
                assert!(d.id > l, "PRIMITIVES not strictly sorted at id {}", d.id);
            }
            last = Some(d.id);
            let colons = d.name.matches(':').count();
            let expected_argc = if colons == 0 { 0 } else { colons };
            // Binary selectors (+, -, ==, etc.) have argc 1 despite 0 colons.
            // `_`-prefixed dev hooks (`__vmStats`) are ordinary unary names,
            // not operators.
            let first = d.name.chars().next().unwrap();
            let is_binary = !first.is_alphabetic() && first != '_';
            if is_binary {
                assert_eq!(d.argc, 1, "{} expected argc 1", d.name);
            } else {
                assert_eq!(
                    d.argc as usize, expected_argc,
                    "{} argc/colon mismatch",
                    d.name
                );
            }
        }
    }

    #[test]
    fn prim_smi_matrix() {
        let mut vm = test_vm();
        assert_eq!(call(1, &mut vm, &[smi(2), smi(3)]), PrimResult::Ok(smi(5)));
        assert_eq!(call(1, &mut vm, &[smi(SMI_MAX), smi(1)]), PrimResult::Fail);
        let nil = vm.universe.nil_obj;
        assert_eq!(call(1, &mut vm, &[nil, smi(1)]), PrimResult::Fail);

        assert_eq!(call(2, &mut vm, &[smi(5), smi(3)]), PrimResult::Ok(smi(2)));
        assert_eq!(call(3, &mut vm, &[smi(5), smi(3)]), PrimResult::Ok(smi(15)));
        assert_eq!(
            call(6, &mut vm, &[smi(0b110), smi(0b011)]),
            PrimResult::Ok(smi(0b010))
        );
        assert_eq!(
            call(7, &mut vm, &[smi(0b110), smi(0b011)]),
            PrimResult::Ok(smi(0b111))
        );
        assert_eq!(
            call(8, &mut vm, &[smi(0b110), smi(0b011)]),
            PrimResult::Ok(smi(0b101))
        );

        assert_eq!(
            call(10, &mut vm, &[smi(1), smi(2)]),
            PrimResult::Ok(vm.universe.true_obj)
        );
        assert_eq!(
            call(10, &mut vm, &[smi(2), smi(1)]),
            PrimResult::Ok(vm.universe.false_obj)
        );
        assert_eq!(
            call(14, &mut vm, &[smi(2), smi(2)]),
            PrimResult::Ok(vm.universe.true_obj)
        );
        assert_eq!(
            call(15, &mut vm, &[smi(2), smi(2)]),
            PrimResult::Ok(vm.universe.false_obj)
        );
    }

    #[test]
    fn prim_floored_division() {
        let mut vm = test_vm();
        assert_eq!(
            call(4, &mut vm, &[smi(-7), smi(2)]),
            PrimResult::Ok(smi(-4))
        );
        assert_eq!(
            call(4, &mut vm, &[smi(7), smi(-2)]),
            PrimResult::Ok(smi(-4))
        );
        assert_eq!(call(5, &mut vm, &[smi(-7), smi(2)]), PrimResult::Ok(smi(1)));
        assert_eq!(
            call(5, &mut vm, &[smi(7), smi(-2)]),
            PrimResult::Ok(smi(-1))
        );
        assert_eq!(call(4, &mut vm, &[smi(7), smi(0)]), PrimResult::Fail);
    }

    #[test]
    fn prim_bitshift_edges() {
        let mut vm = test_vm();
        assert_eq!(call(9, &mut vm, &[smi(1), smi(61)]), PrimResult::Fail);
        assert_eq!(
            call(9, &mut vm, &[smi(1), smi(60)]),
            PrimResult::Ok(smi(1i64 << 60))
        );
        assert_eq!(call(9, &mut vm, &[smi(1), smi(62)]), PrimResult::Fail);
        assert_eq!(call(9, &mut vm, &[smi(1), smi(-62)]), PrimResult::Fail);
        assert_eq!(
            call(9, &mut vm, &[smi(-8), smi(-1)]),
            PrimResult::Ok(smi(-4))
        );
    }

    #[test]
    fn prim_smi_overflow_fails() {
        let mut vm = test_vm();
        assert_eq!(call(1, &mut vm, &[smi(SMI_MAX), smi(1)]), PrimResult::Fail);
    }

    #[test]
    fn prim_identity_hash_stable() {
        let mut vm = test_vm();
        let object_klass = vm.universe.object_klass;
        let a = alloc::alloc_slots(&mut vm, object_klass).oop();
        let b = alloc::alloc_slots(&mut vm, object_klass).oop();
        let h1 = call(20, &mut vm, &[a]);
        let h1_again = call(20, &mut vm, &[a]);
        assert_eq!(h1, h1_again);
        let h2 = call(20, &mut vm, &[b]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn prim_oops_matrix() {
        let mut vm = test_vm();
        let array_klass = vm.universe.array_klass;
        let double_klass = vm.universe.double_klass;

        // basicNew on an indexable klass succeeds with a size-0 instance;
        // basicNew on a Double-format klass fails (excluded format).
        match call(23, &mut vm, &[array_klass.oop()]) {
            PrimResult::Ok(o) => {
                let a = ArrayOop::try_from(o).expect("basicNew on Array must yield an ArrayOop");
                assert_eq!(a.len(), 0);
            }
            PrimResult::Fail => panic!("basicNew on Array must not fail"),
            PrimResult::Activated => unreachable!("basicNew never activates"),
            PrimResult::Nlr(_) => unreachable!("basicNew never NLRs"),
        }
        assert_eq!(call(23, &mut vm, &[double_klass.oop()]), PrimResult::Fail);

        let arr = alloc::alloc_indexable_oops(&mut vm, array_klass, 3).oop();
        assert_eq!(call(28, &mut vm, &[arr]), PrimResult::Ok(smi(3)));
        assert_eq!(
            call(26, &mut vm, &[arr, smi(1)]),
            PrimResult::Ok(vm.universe.nil_obj)
        );
        assert_eq!(
            call(27, &mut vm, &[arr, smi(1), smi(9)]),
            PrimResult::Ok(smi(9))
        );
        assert_eq!(call(26, &mut vm, &[arr, smi(1)]), PrimResult::Ok(smi(9)));
        assert_eq!(call(26, &mut vm, &[arr, smi(4)]), PrimResult::Fail);

        let bytearray_klass = vm.universe.bytearray_klass;
        let bytes = alloc::alloc_indexable_bytes(&mut vm, bytearray_klass, 2).oop();
        assert_eq!(call(26, &mut vm, &[bytes, smi(1)]), PrimResult::Fail);
    }

    #[test]
    fn prim_bytes_matrix() {
        let mut vm = test_vm();
        let bytearray_klass = vm.universe.bytearray_klass;
        let b = alloc::alloc_indexable_bytes(&mut vm, bytearray_klass, 3).oop();
        assert_eq!(
            call(41, &mut vm, &[b, smi(1), smi(65)]),
            PrimResult::Ok(smi(65))
        );
        assert_eq!(call(40, &mut vm, &[b, smi(1)]), PrimResult::Ok(smi(65)));
        assert_eq!(call(41, &mut vm, &[b, smi(1), smi(300)]), PrimResult::Fail);

        let sym = vm.universe.intern(b"foo").oop();
        assert_eq!(call(41, &mut vm, &[sym, smi(1), smi(1)]), PrimResult::Fail);
    }

    /// The primitive's source is always addressed from 1, so the only
    /// same-object overlap hazard this signature can produce is `from > 1`
    /// (dest shifted right of source â€” a naive left-to-right in-place copy
    /// would read already-overwritten bytes). `from == 1` is an exact
    /// self-overlap (dest == source) and must be a safe identity copy.
    /// Both are covered here since the buffer-based implementation must get
    /// both right.
    #[test]
    fn prim_replace_overlap_forward_back() {
        let mut vm = test_vm();
        let bytearray_klass = vm.universe.bytearray_klass;
        let b = alloc::alloc_indexable_bytes(&mut vm, bytearray_klass, 5).oop();
        for (i, v) in [1u8, 2, 3, 4, 5].into_iter().enumerate() {
            let _ = call(41, &mut vm, &[b, smi(i as i64 + 1), smi(v as i64)]);
        }
        // Shift +1: replaceFrom:2 to:5 with: self (same object) â€” copies
        // self[1..4] into self[2..5].
        let _ = call(43, &mut vm, &[b, smi(2), smi(5), b]);
        for (i, expect) in [1u8, 1, 2, 3, 4].into_iter().enumerate() {
            assert_eq!(
                call(40, &mut vm, &[b, smi(i as i64 + 1)]),
                PrimResult::Ok(smi(expect as i64))
            );
        }

        // Exact self-overlap: replaceFrom:1 to:5 with: self is an identity
        // copy (source and destination ranges coincide).
        let _ = call(43, &mut vm, &[b, smi(1), smi(5), b]);
        for (i, expect) in [1u8, 1, 2, 3, 4].into_iter().enumerate() {
            assert_eq!(
                call(40, &mut vm, &[b, smi(i as i64 + 1)]),
                PrimResult::Ok(smi(expect as i64))
            );
        }
    }

    #[test]
    fn prim_compare_prefix() {
        let mut vm = test_vm();
        let ab = vm.universe.intern(b"ab").oop();
        let abc = vm.universe.intern(b"abc").oop();
        assert_eq!(call(45, &mut vm, &[ab, abc]), PrimResult::Ok(smi(-1)));
        assert_eq!(call(45, &mut vm, &[abc, ab]), PrimResult::Ok(smi(1)));
        assert_eq!(call(45, &mut vm, &[ab, ab]), PrimResult::Ok(smi(0)));
    }

    /// A trivial `CompiledBlock` closure with `argc` args and no captures â€”
    /// enough shape to drive `activate_block`'s argc check and frame push
    /// without needing any real computation inside the block body.
    fn make_block_closure(vm: &mut VmState, argc: usize) -> Oop {
        let mut home = crate::bytecode::BytecodeBuilder::new();
        let lit = home.build_block(vm, argc, argc, false, 0, false, |b, _vm| {
            b.ret_self();
        });
        home.push_closure(lit, 0);
        home.ret_tos();
        let sel = vm.universe.intern(format!("mk{argc}").as_bytes());
        let method = home.finish(vm, sel, 0, 0);
        let recv = vm.universe.nil_obj;
        crate::interpreter::run_method(vm, method, recv, &[])
    }

    #[test]
    fn value_family_matrix() {
        for argc in 0usize..=3 {
            let mut vm = test_vm();
            let closure_oop = make_block_closure(&mut vm, argc);
            vm.stack.push(closure_oop);
            for _ in 0..argc {
                vm.stack.push(vm.universe.nil_obj);
            }
            vm.prim_arg_base = vm.stack.sp - argc - 1;
            let mut buf = [vm.universe.nil_obj; 6];
            for (i, slot) in buf.iter_mut().enumerate().take(argc + 1) {
                *slot = vm.stack.get(vm.prim_arg_base + i);
            }
            let prim_id = 50 + argc as u16;
            let result = call(prim_id, &mut vm, &buf[..=argc]);
            assert_eq!(result, PrimResult::Activated, "argc={argc}");
        }

        // Non-closure receiver -> Fail, for every arity in the family.
        for argc in 0usize..=3 {
            let mut vm = test_vm();
            let prim_id = 50 + argc as u16;
            let mut buf = [smi(1); 4];
            buf[0] = vm.universe.nil_obj; // not a Closure
            let result = call(prim_id, &mut vm, &buf[..=argc]);
            assert_eq!(result, PrimResult::Fail, "argc={argc}");
        }
    }

    #[test]
    fn value_with_arguments() {
        let mut vm = test_vm();
        let closure_oop = make_block_closure(&mut vm, 2);
        let array_klass = vm.universe.array_klass;
        let arr = crate::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, 2);
        arr.at_put(0, smi(10));
        arr.at_put(1, smi(20));

        vm.stack.push(closure_oop);
        vm.stack.push(arr.oop());
        vm.prim_arg_base = vm.stack.sp - 2;
        let buf = [closure_oop, arr.oop()];
        let result = call(54, &mut vm, &buf);
        assert_eq!(result, PrimResult::Activated);
        // The array's 2 elements must have been spread onto the stack in
        // place of the array itself.
        let fp = vm.stack.fp;
        let frame = crate::interpreter::stack::Frame { fp };
        assert_eq!(frame.temp(&vm.stack, 0), smi(10));
        assert_eq!(frame.temp(&vm.stack, 1), smi(20));

        // Size mismatch -> Fail, no stack surgery.
        let mut vm2 = test_vm();
        let closure_oop2 = make_block_closure(&mut vm2, 2);
        let array_klass2 = vm2.universe.array_klass;
        let arr2 = crate::memory::alloc::alloc_indexable_oops(&mut vm2, array_klass2, 3);
        vm2.stack.push(closure_oop2);
        vm2.stack.push(arr2.oop());
        vm2.prim_arg_base = vm2.stack.sp - 2;
        let sp_before = vm2.stack.sp;
        let buf2 = [closure_oop2, arr2.oop()];
        assert_eq!(call(54, &mut vm2, &buf2), PrimResult::Fail);
        assert_eq!(vm2.stack.sp, sp_before, "Fail must not touch the stack");

        // Non-Array argument -> Fail.
        let mut vm3 = test_vm();
        let closure_oop3 = make_block_closure(&mut vm3, 1);
        let buf3 = [closure_oop3, smi(5)];
        assert_eq!(call(54, &mut vm3, &buf3), PrimResult::Fail);
    }

    /// S6: `instVarAt:` (p25) on a `Klass`-format receiver reads its 8
    /// named fields 1-based (SPEC Â§2.4 order) â€” Behavior's accessors
    /// (`name`, `superclass`, `instVarNames`, â€¦) depend on this.
    #[test]
    fn prim_instvarat_klass() {
        let mut vm = test_vm();
        let object_klass = vm.universe.object_klass;
        let sym = object_klass.name();
        assert_eq!(
            call(25, &mut vm, &[object_klass.oop(), smi(5)]),
            PrimResult::Ok(sym)
        );
        let sup = object_klass.superclass();
        assert_eq!(
            call(25, &mut vm, &[object_klass.oop(), smi(3)]),
            PrimResult::Ok(sup)
        );
    }

    #[test]
    fn prim_double_matrix() {
        let mut vm = test_vm();
        let d1 = alloc::alloc_double(&mut vm, 1.5).oop();
        let d2 = alloc::alloc_double(&mut vm, 2.5).oop();
        let true_obj = vm.universe.true_obj;
        match call(100, &mut vm, &[d1, d2]) {
            PrimResult::Ok(o) => {
                assert_eq!(
                    crate::oops::wrappers::DoubleOop::try_from(o)
                        .unwrap()
                        .value(),
                    4.0
                )
            }
            other => panic!("expected Ok, got {other:?}"),
        }
        assert_eq!(call(104, &mut vm, &[d1, d2]), PrimResult::Ok(true_obj));
        assert_eq!(call(105, &mut vm, &[d1, d1]), PrimResult::Ok(true_obj));
        // Non-Double arg fails (Smalltalk fallback coerces, SPEC Â§1.3).
        assert_eq!(call(100, &mut vm, &[d1, smi(1)]), PrimResult::Fail);

        let big = alloc::alloc_double(&mut vm, 1e10).oop();
        let dgt = call(109, &mut vm, &[big]);
        match dgt {
            PrimResult::Ok(o) => {
                let b = ByteArrayOop::try_from(o).unwrap();
                let mut buf = Vec::new();
                b.copy_bytes_out(&mut buf);
                assert_eq!(String::from_utf8(buf).unwrap(), "10000000000.0");
            }
            other => panic!("expected Ok, got {other:?}"),
        }

        match call(108, &mut vm, &[smi(3)]) {
            PrimResult::Ok(o) => {
                assert_eq!(
                    crate::oops::wrappers::DoubleOop::try_from(o)
                        .unwrap()
                        .value(),
                    3.0
                )
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    /// `tests_s06.md`'s `prim_double_print`: p109 round-trips the shortest
    /// decimal form for a handful of adversarial values (`prim_error`'s own
    /// trace/exit-1 path can't be unit-tested in-process â€” it calls
    /// `std::process::exit`, so that requirement is covered at the CLI
    /// integration layer instead, e.g. `tests/it_world.rs`).
    #[test]
    fn prim_double_print() {
        let mut vm = test_vm();
        let cases: &[(f64, &str)] = &[
            (0.1, "0.1"),
            (1e10, "10000000000.0"),
            (1.5e-3, "0.0015"),
            (-0.0, "-0.0"),
        ];
        for (v, expected) in cases {
            let d = alloc::alloc_double(&mut vm, *v).oop();
            match call(109, &mut vm, &[d]) {
                PrimResult::Ok(o) => {
                    let b = ByteArrayOop::try_from(o).unwrap();
                    let mut buf = Vec::new();
                    b.copy_bytes_out(&mut buf);
                    assert_eq!(String::from_utf8(buf).unwrap(), *expected, "for {v}");
                }
                other => panic!("expected Ok for {v}, got {other:?}"),
            }
        }
    }

    #[test]
    fn prim_quit_colon_sets_exit_code() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        assert_eq!(call(96, &mut vm, &[recv, smi(7)]), PrimResult::Ok(recv));
        assert!(vm.exit_requested);
        assert_eq!(vm.exit_code, Some(7));
    }

    #[test]
    fn prim_string_as_symbol_interns() {
        let mut vm = test_vm();
        let bytearray_klass = vm.universe.bytearray_klass;
        let s = alloc::alloc_indexable_bytes(&mut vm, bytearray_klass, 3);
        s.byte_at_put(0, b'f');
        s.byte_at_put(1, b'o');
        s.byte_at_put(2, b'o');
        let result = call(110, &mut vm, &[s.oop()]);
        let expected = vm.universe.intern(b"foo").oop();
        assert_eq!(result, PrimResult::Ok(expected));
    }

    /// S12 step 7: the GC prims re-read their receiver through
    /// `vm.prim_arg(0)` after collecting (the `args` copy is stale bits
    /// once the receiver itself moves â€” latent since S8, only reachable
    /// now that a collection can actually run mid-prim with young
    /// receivers). The bare `call` helper above invokes the prim fn
    /// directly with an unrooted slice, so these tests must mimic
    /// `try_primitive`'s own protocol: push the receiver onto the live
    /// stack and point `prim_arg_base` at it, exactly like a real send.
    fn call_rooted(id: u16, vm: &mut VmState, recv: Oop) -> PrimResult {
        let base = vm.stack.sp;
        vm.stack.push(recv);
        vm.prim_arg_base = base;
        let r = (prim_by_id(id).unwrap().f)(vm, &[recv]);
        vm.stack.sp = base;
        r
    }

    /// `Smalltalk gcScavenge` must run a REAL scavenge (id 93), not the old
    /// no-op stub â€” proven by the counter it bumps, not just "didn't crash".
    /// The answered receiver must be the LIVE (post-scavenge) nil â€” a young
    /// receiver moves, and the pre-call `recv` bits are the vacated address.
    #[test]
    fn prim_gc_scavenge_runs_a_real_scavenge() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        let before = vm.universe.gc_stats.scavenge_count;
        let result = call_rooted(93, &mut vm, recv);
        assert_eq!(vm.universe.gc_stats.scavenge_count, before + 1);
        assert_eq!(
            result,
            PrimResult::Ok(vm.universe.nil_obj),
            "must answer the RELOCATED receiver (nil moved -- it was young)"
        );
        assert_ne!(
            vm.universe.nil_obj.raw(),
            recv.raw(),
            "precondition: genesis nil is young, so the scavenge must actually move it \
             (otherwise this test proves nothing about the stale-args hazard)"
        );
    }

    /// `Smalltalk gcFull` must run a REAL full GC (id 94) â€” same shape as
    /// the scavenge test above, this time against `fullGcCount`.
    #[test]
    fn prim_gc_full_runs_a_real_full_gc() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        let before = vm.universe.gc_stats.full_gc_count;
        let result = call_rooted(94, &mut vm, recv);
        assert_eq!(vm.universe.gc_stats.full_gc_count, before + 1);
        assert_eq!(result, PrimResult::Ok(vm.universe.nil_obj));
    }

    /// S12 step 7 (inverts S11's `prim_gc_scavenge_defers_under_compiled_
    /// frame`): the D8 defer arm is gone, so `gcScavenge` runs a REAL
    /// scavenge immediately even with `compiled_depth > 0` (a faked depth
    /// with an empty native stack is honest here: `walk_frames` starts
    /// from the anchor, which is 0, so the code-root walk correctly sees
    /// no native frames). `gc_under_compiled` must count the hard case.
    #[test]
    fn prim_gc_scavenge_runs_immediately_under_compiled_depth() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        let before = vm.universe.gc_stats.scavenge_count;
        let under_before = vm.universe.gc_stats.gc_under_compiled;
        vm.compiled_depth = 1;
        let result = call_rooted(93, &mut vm, recv);
        vm.compiled_depth = 0;
        assert_eq!(result, PrimResult::Ok(vm.universe.nil_obj));
        assert_eq!(
            vm.universe.gc_stats.scavenge_count,
            before + 1,
            "the defer arm is gone -- gcScavenge collects immediately at any depth"
        );
        assert_eq!(vm.universe.gc_stats.gc_under_compiled, under_before + 1);
    }

    /// Sibling of the scavenge test above, for `gcFull` (id 94).
    #[test]
    fn prim_gc_full_runs_immediately_under_compiled_depth() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        let before = vm.universe.gc_stats.full_gc_count;
        vm.compiled_depth = 1;
        let result = call_rooted(94, &mut vm, recv);
        vm.compiled_depth = 0;
        assert_eq!(result, PrimResult::Ok(vm.universe.nil_obj));
        assert_eq!(vm.universe.gc_stats.full_gc_count, before + 1);
    }

    /// `Smalltalk gcStats` (id 97) answers an 8-smi Array in the SPEC-pinned
    /// order â€” checked both structurally (an Array of exactly 8 smis) and
    /// against the actual counters after a real scavenge + full GC, so a
    /// field silently swapped to the wrong position would fail this, not
    /// just "returns something array-shaped".
    #[test]
    fn prim_gc_stats_reports_pinned_fields_in_order() {
        let mut vm = test_vm();
        let recv = vm.universe.nil_obj;
        call_rooted(93, &mut vm, recv); // one scavenge

        // Re-read the LIVE nil for the second call: the scavenge above
        // moved it, and pushing the stale pre-scavenge `recv` bits as a
        // root is exactly the dangling-root shape the full-gc entry
        // verifier (correctly) rejects â€” this test tripped it for real
        // the first time this line reused `recv`.
        let recv2 = vm.universe.nil_obj;
        call_rooted(94, &mut vm, recv2); // one full GC

        // Snapshot expectations BEFORE calling gcStats: the primitive
        // itself allocates its own result Array (in eden), so reading
        // eden's usage back out AFTER the call would include that very
        // allocation â€” gcStats correctly reports the state as of the
        // moment it was called, not after its own side effect.
        let expected = (
            vm.universe.gc_stats.scavenge_count as i64,
            vm.universe.gc_stats.full_gc_count as i64,
            (vm.universe.eden.top - vm.universe.eden.start) as i64,
            (vm.universe.old.top - vm.universe.old.bounds.start) as i64,
            (vm.universe.old.committed_end - vm.universe.old.bounds.start) as i64,
            vm.universe.gc_stats.bytes_promoted as i64,
            vm.universe.gc_stats.marked_bytes_last as i64,
            vm.universe.gc_stats.context_allocs as i64,
        );

        let result = call(97, &mut vm, &[recv]);
        let PrimResult::Ok(oop) = result else {
            panic!("gcStats must succeed, got {result:?}")
        };
        let arr = ArrayOop::try_from(oop).expect("gcStats must answer an Array");
        assert_eq!(arr.len(), 8);
        let as_i64 = |i: usize| {
            SmallInt::try_from(arr.at(i))
                .expect("every field is a smi")
                .value()
        };

        assert_eq!(as_i64(0), expected.0, "scavengeCount");
        assert_eq!(as_i64(1), expected.1, "fullGcCount");
        assert_eq!(as_i64(2), expected.2, "edenUsed");
        assert_eq!(as_i64(3), expected.3, "oldUsed");
        assert_eq!(as_i64(4), expected.4, "oldCommitted");
        assert_eq!(as_i64(5), expected.5, "bytesPromoted");
        assert_eq!(as_i64(6), expected.6, "markedBytesLast");
        assert_eq!(as_i64(7), expected.7, "contextAllocs");
    }
}
