//! `call_stub`/`stub_poll` (S10 D5.6) and, from S11, the runtime-stub
//! table's growing collection of anchor-and-call-into-Rust trampolines
//! (D5) — hand-assembled once at startup, not derived from any `Ir`. All
//! published into the same [`CodeCache`] real compiled methods live in.

use crate::compiler::assembler::{
    d,    imm, mem, mem_post, mem_pre, sp, x, xr, Assembler, CodeBlob, Cond, RelocKind,
};
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::oops::layout::{
    ROOTSPILL_BYTES, VMREG_LAST_COMPILED_FP_OFFSET, VMREG_LAST_COMPILED_KIND_OFFSET,
    VMREG_LAST_COMPILED_PC_OFFSET,
};
use crate::oops::wrappers::{KlassOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::lookup::{klass_of, lookup};
use crate::runtime::vm_state::{TierLink, VmState};

use super::nmethod::{IcState, NmethodId};
use super::pics::PIC_MAX_ENTRIES;
use super::{CodeCache, CodeHandle};

/// S12 D3: `runtime::frames::AdapterKind`'s raw wire values, written by
/// [`emit_stub_kind_tag`] and read back by the frame walker via
/// `VmRegBlock::last_compiled_kind`. Owned HERE (not in `runtime::frames`)
/// because this is the module that actually emits the `movz`/`str` writing
/// them — `frames.rs`'s `AdapterKind::from_raw`/`as_raw` just mirror these
/// same six numbers. `Poll` has no entry: `stub_poll` never calls
/// `emit_stub_prologue` at all (S10 D5.6: it never allocates/GCs, so it
/// never needs the anchor), so this kind is never actually written --
/// `frames.rs` keeps a `Poll` variant only for `AdapterKind`'s own
/// completeness, provably dead on this specific path.
/// Byte width of the patchable call an IC site occupies — how far back
/// from a return address the call instruction starts. AArch64 `bl` is one
/// 4-byte word; x86-64 `E8 rel32` is 5 bytes.
#[cfg(target_arch = "aarch64")]
pub const CALL_INSN_LEN: u64 = 4;
#[cfg(not(target_arch = "aarch64"))]
pub const CALL_INSN_LEN: u64 = 5;

pub const KIND_RESOLVE: u64 = 0;
pub const KIND_C2I: u64 = 1;
pub const KIND_MEGA: u64 = 2;
pub const KIND_DNU: u64 = 3;
pub const KIND_MUST_BE_BOOLEAN: u64 = 4;
pub const KIND_ALLOC_SLOW: u64 = 5;
/// The two deopt trampolines' anchor tag (`build_uncommon_trampoline` /
/// `build_deopt_return_trampoline`): their frame record — `[fp]` = the
/// deoptee's fp, `[fp+8]` = the pc classifying it (trap pc / orig return
/// pc) — anchors a GC that starts inside `deoptimize_frame`'s materializer
/// allocations. Maps to `AdapterKind::DeoptBridge` deliberately: like the
/// synthetic post-materialization bridge link, the record itself owns NO
/// RootSpill slots (`real_oop_rootspill_slots` → 0) — the deoptee frame's
/// oops are covered by its OWN oop map at the recorded pc, via the
/// `FrameView::Compiled` the walker emits right after the adapter.
pub const KIND_DEOPT_BRIDGE: u64 = 6;
/// A compiled method's own primitive-call shim (`compiler::driver`'s
/// shimmable-primitive eligibility). See [`build_stub_call_primitive`].
pub const KIND_CALL_PRIMITIVE: u64 = 7;
/// S24 A1 (closure compilation): a compiled BLOCK body's `nlr_tos` parking
/// its non-local return (`docs/closure_compilation_design.md` §2.4 —
/// origination only; all propagation is the existing S11 sentinel relay).
/// RootSpill live slots = 2 (x0 = closure, x1 = value). See
/// [`build_stub_nlr_originate`].
pub const KIND_NLR_ORIGINATE: u64 = 8;
/// S24 A2 (closure compilation): a compiled `value`-family send site's
/// direct block dispatch (`docs/closure_compilation_design.md` §2.3). The
/// stub probes `by_block` for the receiver closure and either tail-jumps
/// to the block nmethod or falls back to interpreting `value:`. RootSpill
/// live slots = the caller site's own `IcSite::argc` (`1 + block args`,
/// exactly as `Resolve|C2i|Mega|Dnu` recover it — the fallback's nested
/// interpretation can GC while closure+args sit in RootSpill). See
/// [`build_stub_value_dispatch`].
pub const KIND_VALUE_DISPATCH: u64 = 9;
/// Float fast-path (`docs/float_fastpath_design.md`): `Ir::FBox`'s inline
/// eden-bump overflow tail — box an unboxed `f64` into a fresh `Double`.
/// x0 carries the RAW PAYLOAD BITS, not an oop, so this kind owns **zero**
/// live RootSpill oop slots (`memory::roots::real_oop_rootspill_slots`) —
/// scanning x0 as an oop would treat float bits as a heap pointer. See
/// [`build_stub_box_double`].
pub const KIND_BOX_DOUBLE: u64 = 10;
/// SIMD NEON fast-path (`docs/SIMD.md`, Increment 2): `Ir::VecArith`'s
/// eden-overflow tail — box a NEON result into a fresh `Float64x2`. x0/x1
/// carry the RAW LANE BITS (two f64 patterns), not oops, so this kind owns
/// **zero** live RootSpill oop slots (exactly like `KIND_BOX_DOUBLE`). See
/// [`build_stub_box_float64x2`].
pub const KIND_BOX_FLOAT64X2: u64 = 11;
/// SIMD: `Ir::VecArith`'s eden-overflow tail for a `Float32x4` — same shape as
/// `KIND_BOX_FLOAT64X2` (x0/x1 are two raw 64-bit halves of the 128-bit lane
/// body, never oops → zero RootSpill oop slots). See [`build_stub_box_float32x4`].
pub const KIND_BOX_FLOAT32X4: u64 = 12;
/// SIMD: `Ir::VecArith`'s eden-overflow tail for an `Int32x4` — same shape as
/// the float box kinds (x0/x1 are two raw 64-bit halves of the 128-bit lane
/// body, never oops → zero RootSpill oop slots). See [`build_stub_box_int32x4`].
pub const KIND_BOX_INT32X4: u64 = 13;

/// D5's shared stub skeleton, part 1: anchor + AAPCS frame + RootSpill
/// (x0..x5). Every stub that calls into Rust starts with this; follow with
/// the stub's own `call_far`, then [`emit_stub_epilogue`]. `x28` must
/// already be `&VmState` (established once by `call_stub`, D5.1) — every
/// site this is used from is reached with that invariant already holding.
/// A free function taking `&mut dyn Assembler`, not a method, so it stays
/// reusable across every stub builder in this module without any of them
/// needing to share a common base type beyond the trait itself.
///
/// Deliberately does NOT also write `last_compiled_kind` (S12): an earlier
/// draft tried folding that in here too, using x16/x17 as scratch for the
/// immediate — but `build_c2i_shared`/`build_mega_shared` read x17/x16
/// (the method/selector oop carried through from the per-instance
/// adapter/mega trampoline's own tail-jump) as their OWN first instruction
/// after this prologue returns, so writing through either register here
/// would clobber it on every c2i/mega call in the system. Each of the 6
/// callers below calls [`emit_stub_kind_tag`] itself, immediately after
/// this, using a register IT has independently confirmed is free (x9 —
/// none of the 6 touch it this early; matches `build_call_stub`'s own
/// "caller-saved, x9-x11" scratch convention).
pub fn emit_stub_prologue(a: &mut dyn Assembler) {
    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);
    a.emit("sub", &[sp(), sp(), imm(ROOTSPILL_BYTES as i64)]);
    a.emit("stp", &[x(0), x(1), mem(31, 0)]);
    a.emit("stp", &[x(2), x(3), mem(31, 16)]);
    a.emit("stp", &[x(4), x(5), mem(31, 32)]);
    // x6/x7 too (S15, BUG D root cause 4's deeper half — see
    // `layout::ROOTSPILL_SLOTS`'s own doc): a compiled send site marshals
    // up to receiver+7 args into x0..x7, and the Rust call this stub is
    // about to make treats x6/x7 as caller-saved scratch — without this
    // pair a ≥6-arg send's tail args are clobbered on every resolve/mega/
    // dnu, and c2i's argv read walked off the old 6-slot area into this
    // frame's own saved fp/lr.
    a.emit("stp", &[x(6), x(7), mem(31, 48)]);
    // Anchor (D4.1): last_compiled_fp/pc so a stack walker (S12) can find
    // this stub's own frame, and everything above it, while control is in
    // Rust.
    a.emit(
        "str",
        &[x(29), mem(28, VMREG_LAST_COMPILED_FP_OFFSET as i64)],
    );
    a.emit(
        "str",
        &[x(30), mem(28, VMREG_LAST_COMPILED_PC_OFFSET as i64)],
    );
}

/// S12 D3: writes `last_compiled_kind` — called by each of the 6
/// anchor-setting stubs' own preamble, immediately after
/// `emit_stub_prologue` returns, using x9 (`build_call_stub`'s own
/// "caller-saved, x9-x11" scratch precedent; independently confirmed free
/// at this exact point in all 6 callers — none of them reads x9 before
/// their own first `mov`/`ldr` off x0-x5/x16/x17/x28/x30). `kind` is one of
/// this module's own `KIND_*` constants, never `AdapterKind::Poll`'s
/// value (unwritten by construction — see `emit_stub_prologue`'s own doc).
fn emit_stub_kind_tag(a: &mut dyn Assembler, kind: u64) {
    a.emit("movz", &[x(9), imm(kind as i64)]);
    a.emit(
        "str",
        &[x(9), mem(28, VMREG_LAST_COMPILED_KIND_OFFSET as i64)],
    );
}

/// Part 2: clears the anchor (P9 — a stale anchor makes S12's walker walk
/// freed frames), reloads x0..x7 from RootSpill, tears down the frame.
/// Does NOT emit the final `ret`/`br`/`b` — callers differ on that (plain
/// `ret` for a callee-shaped stub like `dnu`, `br x16` tail-jump for
/// `resolve`/`mega`, P4) — and does not touch whatever scratch register
/// the caller is about to jump through (typically x16), matching the same
/// per-caller variation.
pub fn emit_stub_epilogue(a: &mut dyn Assembler) {
    // x(31) here means xzr, not sp: register 31 in a store's *data* (Rt)
    // position is unconditionally the zero register on this ISA — `is_sp`
    // only disambiguates ALU/move *operand* positions (`assembler.rs`'s
    // own `sp()` doc comment), which this is not.
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_FP_OFFSET as i64)],
    );
    // S12: clear the kind tag alongside the fp (defense in depth, not load-
    // bearing — nothing ever reads it while `last_compiled_fp == 0` — but a
    // cleared-and-therefore-obviously-wrong value is strictly better than a
    // stale-but-plausible one if some future bug ever left `last_compiled_fp`
    // stale too).
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_KIND_OFFSET as i64)],
    );
    a.emit("ldp", &[x(0), x(1), mem(31, 0)]);
    a.emit("ldp", &[x(2), x(3), mem(31, 16)]);
    a.emit("ldp", &[x(4), x(5), mem(31, 32)]);
    a.emit("ldp", &[x(6), x(7), mem(31, 48)]);
    a.emit("add", &[sp(), sp(), imm(ROOTSPILL_BYTES as i64)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]);
}

/// `extern "C" fn(entry, vm, argv, argc) -> u64` (D4) — the shape every
/// call through `call_stub` must be transmuted to.
pub type CallStubFn =
    unsafe extern "C" fn(entry: u64, vm: *mut VmState, argv: *const u64, argc: u64) -> u64;

/// `Copy`: both fields are just a pointer+len (`CodeHandle` itself is
/// `Copy`) — copying a `Stubs` duplicates no real resource, and lets
/// `enter_compiled` read it out of `vm.stubs` before also needing `&mut
/// VmState` for the call itself (`vm.stubs.invoke(entry, vm, ...)` would
/// otherwise borrow `vm.stubs` and all of `vm` at once).
#[derive(Clone, Copy)]
pub struct Stubs {
    pub call_stub: CodeHandle,
    pub stub_poll: CodeHandle,
    /// D4.1: the shared `stub_resolve`/`stub_ic_miss` door — `emit.rs`
    /// embeds its address as every `IcSite`'s initial `bl` target (S11
    /// step 2).
    pub resolve: CodeHandle,
    /// D6.1: the shared c2i adapter tail — every per-method `c2i_<method>`
    /// trampoline (`codecache::adapters::build_c2i_adapter`) `br`s here
    /// with the target method's own oop already in x17 (S11 step 4).
    pub c2i_shared: CodeHandle,
    /// D4.4: the shared megamorphic-lookup tail — every per-selector
    /// `mega_<sel>` trampoline (`codecache::mega::build_mega_trampoline`)
    /// `br`s here with the selector's own oop already in x16 (S11 step 5).
    pub mega_shared: CodeHandle,
    /// D5.4: `rt_resolve_send`/`rt_mega_lookup`'s own lookup-failure
    /// return value — never patched into a site, just tail-jumped to for
    /// THIS one dispatch (S11 step 6).
    pub dnu: CodeHandle,
    /// D1/D5: the `BoolBr.not_bool` runtime helper — built and callable
    /// now (S11 step 6), but not yet reachable from any REAL compiled
    /// method: emitting a call to it from a `not_bool` block is
    /// `Ir::CallRuntime`'s own job, S11 step 7's eligibility relaxation.
    pub must_be_boolean: CodeHandle,
    /// D7: the inline-allocation slow-path helper — an `Ir::Alloc` fast
    /// path whose eden bump overflows `bl`s here with the class oop in x0
    /// and the size in bytes in x1 (S11 step 8). Callee-shaped (plain `ret`,
    /// result oop in x0), same contract as `must_be_boolean`.
    pub alloc_slow: CodeHandle,
    /// A shimmable primitive-bearing method's own shim prologue `bl`s here
    /// with `prim_id` in x10, `receiver + real args` count in x11, and
    /// x0..x5 already the real receiver+args (`compiler::driver`'s
    /// shimmable-primitive eligibility). See [`build_stub_call_primitive`].
    pub call_primitive: CodeHandle,
    /// Float fast-path: `Ir::FBox`'s eden-overflow tail — x0 = the raw f64
    /// payload BITS (not an oop; zero RootSpill oop slots). Allocates and
    /// initializes a `Double`, answering its tagged oop in x0. See
    /// [`build_stub_box_double`].
    pub box_double: CodeHandle,
    /// SIMD NEON fast-path: `Ir::VecArith`'s eden-overflow tail — x0/x1 = the
    /// two raw f64 lane BITS (not oops; zero RootSpill oop slots). Allocates
    /// and initializes a `Float64x2`, answering its tagged oop in x0. See
    /// [`build_stub_box_float64x2`].
    pub box_float64x2: CodeHandle,
    /// SIMD: `Ir::VecArith`'s eden-overflow tail for a `Float32x4` — same shape
    /// as `box_float64x2` (x0/x1 = the two 64-bit halves of the `.4s` result).
    /// See [`build_stub_box_float32x4`].
    pub box_float32x4: CodeHandle,
    /// SIMD: `Ir::VecArith`'s eden-overflow tail for an `Int32x4` — same shape.
    /// See [`build_stub_box_int32x4`].
    pub box_int32x4: CodeHandle,
    /// S24 A1: a compiled BLOCK body's `nlr_tos` lowering `bl`s here with
    /// x0 = the closure, x1 = the NLR value; `rt_nlr_originate` parks
    /// `vm.nlr_state` and the emitted continuation returns `NLR_SENTINEL`
    /// through the block's own epilogue (`closure_compilation_design.md`
    /// §2.4). See [`build_stub_nlr_originate`].
    pub nlr_originate: CodeHandle,
    /// S24 A2: the four shared `value`-family block-dispatch stubs, indexed
    /// by block argc 0..=3 (`value`/`value:`/`value:value:`/
    /// `value:value:value:`). `resolve_target_entry` patches a compiled
    /// `value`-family send site's `bl` at one of these instead of the
    /// generic c2i adapter; the stub probes `by_block` and tail-jumps to
    /// the block nmethod, or falls back to interpreting `value:`
    /// (`closure_compilation_design.md` §2.3). See
    /// [`build_stub_value_dispatch`].
    pub value_dispatch: [CodeHandle; 4],
    /// S13 D1 §2b: the shared `not_entrant_stub` — a now-`NotEntrant`
    /// nmethod's `entry`/`verified_entry` are patched (by
    /// `nmethod::make_not_entrant`) to `b not_entrant_stub`, so a future call
    /// through a compiled caller's `bl` (which still targets the old entry)
    /// lands here with receiver+args untouched in x0..x5 and x30 = the
    /// caller's own send-site return address. It re-dispatches EXACTLY like an
    /// IC miss (`stub_resolve`/[`rt_resolve_send`]): re-resolve
    /// (receiver klass, selector) for that site, patch the site's `bl` to the
    /// CURRENT method's target, and tail-jump there. See
    /// [`build_not_entrant_stub`] for why this shares the resolve tail
    /// verbatim rather than being its own runtime function.
    pub not_entrant: CodeHandle,
    /// S13 D6: the `deopt_return_trampoline` — the address §2c
    /// ([`crate::runtime::frames::redirect_returns_into_nm`]) writes into an
    /// in-flight `NotEntrant` activation's callee saved-LR slots, so a callee's
    /// `ret` deopts the caller lazily instead of resuming old compiled code.
    /// See [`build_deopt_return_trampoline`].
    pub deopt_return: CodeHandle,
}

impl Stubs {
    pub fn call_stub_entry(&self) -> *const u8 {
        self.call_stub.base
    }
    pub fn stub_poll_addr(&self) -> u64 {
        self.stub_poll.base as u64
    }
    pub fn resolve_addr(&self) -> u64 {
        self.resolve.base as u64
    }
    pub fn c2i_shared_addr(&self) -> u64 {
        self.c2i_shared.base as u64
    }
    pub fn mega_shared_addr(&self) -> u64 {
        self.mega_shared.base as u64
    }
    pub fn dnu_addr(&self) -> u64 {
        self.dnu.base as u64
    }
    pub fn must_be_boolean_addr(&self) -> u64 {
        self.must_be_boolean.base as u64
    }
    pub fn alloc_slow_addr(&self) -> u64 {
        self.alloc_slow.base as u64
    }
    pub fn call_primitive_addr(&self) -> u64 {
        self.call_primitive.base as u64
    }
    /// Float fast-path: `emit`'s `Ir::FBox` slow edge `bl`s here (see
    /// [`build_stub_box_double`]).
    pub fn box_double_addr(&self) -> u64 {
        self.box_double.base as u64
    }
    /// SIMD: `emit`'s `Ir::VecArith` slow edge `bl`s here (see
    /// [`build_stub_box_float64x2`]).
    pub fn box_float64x2_addr(&self) -> u64 {
        self.box_float64x2.base as u64
    }
    /// SIMD: `emit`'s `Ir::VecArith` (`Float32x4`) slow edge `bl`s here.
    pub fn box_float32x4_addr(&self) -> u64 {
        self.box_float32x4.base as u64
    }
    /// SIMD: `emit`'s `Ir::VecArith` (`Int32x4`) slow edge `bl`s here.
    pub fn box_int32x4_addr(&self) -> u64 {
        self.box_int32x4.base as u64
    }
    /// S24 A1: `emit`'s `Ir::NlrReturn` lowering `bl`s here (see
    /// [`build_stub_nlr_originate`]).
    pub fn nlr_originate_addr(&self) -> u64 {
        self.nlr_originate.base as u64
    }
    /// S24 A2: the `value`-family dispatch stub for a block of arity `argc`
    /// (0..=3) — `resolve_target_entry`'s own patch target for a
    /// `value`-family site (see [`build_stub_value_dispatch`]).
    pub fn value_dispatch_addr(&self, argc: usize) -> u64 {
        self.value_dispatch[argc].base as u64
    }
    /// S13 §2b: the address `make_not_entrant` patches an invalidated
    /// nmethod's `entry`/`verified_entry` to branch to.
    pub fn not_entrant_addr(&self) -> u64 {
        self.not_entrant.base as u64
    }
    /// S13 D6: the address §2c redirects an in-flight `NotEntrant` activation's
    /// callee saved-LR slots to (see [`build_deopt_return_trampoline`]).
    pub fn deopt_return_addr(&self) -> u64 {
        self.deopt_return.base as u64
    }

    /// Invokes `entry` (a compiled method's own entry point — an `Nmethod`'s
    /// `code.base + entry_off`) through the real, published `call_stub`,
    /// exactly as if it were an ordinary `extern "C"` function taking
    /// `(vm, argv, argc)`. `argv[0]` is the receiver, `argv[1..]` the real
    /// Smalltalk arguments (`ir::convert`'s own `Param{index}` convention).
    ///
    /// The one place `interpreter::compiled_call::enter_compiled` needs
    /// unsafe FFI, tucked in here instead of allowed at that call site —
    /// this module (`codecache`) is `crate`'s one designated owner of every
    /// raw pointer call into `MAP_JIT` memory (this file's own module doc),
    /// so `interpreter` stays covered by the crate-root `#![deny(unsafe_code)]`.
    /// Takes `&mut VmState`, not a raw pointer, so the public signature
    /// itself stays safe (clippy's `not_unsafe_ptr_arg_deref`) — the
    /// reference-to-pointer conversion `call_stub`'s own FFI shape needs is
    /// done internally instead.
    pub fn invoke(&self, entry: u64, vm: &mut VmState, argv: &[u64]) -> u64 {
        let call: CallStubFn = unsafe { std::mem::transmute(self.call_stub_entry()) };
        let vm_ptr: *mut VmState = vm;
        unsafe { call(entry, vm_ptr, argv.as_ptr(), argv.len() as u64) }
    }
}

/// Build and publish both stubs into `cache`. Call once, before compiling
/// or running any real method — `emit.rs` needs `stub_poll`'s address
/// (`Stubs::stub_poll_addr`) to embed as a pool constant in any method
/// that emits `Ir::Poll`.
///
/// WINVM (Phase 3, MIGRATION.md §4): arch-selected. Until this split, the
/// AArch64 builders below ran on **every** host, so an x86-64 build
/// published A64 encodings into the code cache and handed their addresses
/// to the compiler as `stub_poll_addr()` and friends. Nothing detected
/// that — the addresses are real, the cache bounds-checks pass, and the
/// bytes only reveal themselves as garbage when something jumps to them.
pub fn install(cache: &mut CodeCache) -> Stubs {
    #[cfg(target_arch = "aarch64")]
    {
        install_a64(cache)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        install_x64(cache)
    }
}

/// Publish one blob and return its handle, naming the stub in the panic so
/// an undersized cache says *which* stub did not fit.
fn place(cache: &mut CodeCache, name: &str, blob: &CodeBlob) -> CodeHandle {
    let h = cache
        .alloc(blob.code.len())
        .unwrap_or_else(|| panic!("stubs::install: code cache too small for {name}"));
    cache.publish(h, blob);
    h
}

/// The x86-64 stub table (WINVM Phase 3).
///
/// Two shape differences from the AArch64 side, both deliberate:
///
/// * **The builders take their runtime addresses as arguments** rather
///   than referencing `rt_*` directly, which is what lets them live in
///   `stubs_x64`/`thunks_x64` and be unit-tested against stand-in
///   functions. This function is where the real `rt_*` addresses are
///   supplied, so it is the single place the two halves meet.
/// * **The three SIMD boxers are `ud2`.** `emit_x64` does not lower
///   `FBox` or `VecArith` at all (they are absent from its
///   `SUPPORTED_OPS`, so such a method panics during compilation and
///   never reaches an emitted call). See [`build_unreachable_stub_x64`].
#[cfg(not(target_arch = "aarch64"))]
fn install_x64(cache: &mut CodeCache) -> Stubs {
    use super::stubs_x64::{
        build_call_stub_x64, build_deopt_return_trampoline_x64, build_not_entrant_stub_x64,
        build_stub_alloc_slow_x64, build_stub_box_double_x64, build_stub_call_primitive_x64,
        build_stub_dnu_x64, build_stub_mega_shared_x64, build_stub_must_be_boolean_x64,
        build_stub_nlr_originate_x64, build_stub_poll_x64, build_stub_resolve_x64,
        build_stub_value_dispatch_x64, build_unreachable_stub_x64,
    };
    use super::deopt_trap::{rt_deopt_on_return, rt_deopt_return_pc};
    use super::thunks_x64::build_c2i_shared_x64;

    let a = |f: *const ()| f as u64;

    let call_stub = place(cache, "call_stub", &build_call_stub_x64());
    let stub_poll = place(
        cache,
        "stub_poll",
        &build_stub_poll_x64(a(rt_poll as *const ())),
    );
    let resolve = place(
        cache,
        "stub_resolve",
        &build_stub_resolve_x64(a(rt_resolve_send as *const ())),
    );
    let c2i_shared = place(
        cache,
        "c2i_shared",
        &build_c2i_shared_x64(a(rt_interpret_call as *const ()), KIND_C2I),
    );
    let mega_shared = place(
        cache,
        "mega_shared",
        &build_stub_mega_shared_x64(a(rt_mega_lookup as *const ())),
    );
    let dnu = place(
        cache,
        "dnu",
        &build_stub_dnu_x64(a(rt_dnu as *const ()), KIND_DNU),
    );
    let must_be_boolean = place(
        cache,
        "must_be_boolean",
        &build_stub_must_be_boolean_x64(a(rt_must_be_boolean as *const ())),
    );
    let alloc_slow = place(
        cache,
        "alloc_slow",
        &build_stub_alloc_slow_x64(a(rt_alloc_slow as *const ())),
    );
    let not_entrant = place(
        cache,
        "not_entrant",
        &build_not_entrant_stub_x64(a(rt_resolve_send as *const ())),
    );
    let deopt_return = place(
        cache,
        "deopt_return",
        &build_deopt_return_trampoline_x64(
            a(rt_deopt_return_pc as *const ()),
            a(rt_deopt_on_return as *const ()),
        ),
    );
    let call_primitive = place(
        cache,
        "call_primitive",
        &build_stub_call_primitive_x64(a(rt_call_primitive as *const ()), KIND_CALL_PRIMITIVE),
    );
    let nlr_originate = place(
        cache,
        "nlr_originate",
        &build_stub_nlr_originate_x64(a(rt_nlr_originate as *const ()), KIND_NLR_ORIGINATE),
    );
    let value_dispatch = std::array::from_fn(|argc| {
        let blob = build_stub_value_dispatch_x64(
            a(rt_value_target as *const ()),
            a(rt_value_fallback as *const ()),
            argc as u64,
            KIND_VALUE_DISPATCH,
        );
        place(cache, "value_dispatch", &blob)
    });
    let box_double = place(
        cache,
        "box_double",
        &build_stub_box_double_x64(a(rt_box_double as *const ())),
    );
    let box_float64x2 = place(cache, "box_float64x2", &build_unreachable_stub_x64());
    let box_float32x4 = place(cache, "box_float32x4", &build_unreachable_stub_x64());
    let box_int32x4 = place(cache, "box_int32x4", &build_unreachable_stub_x64());

    Stubs {
        call_stub,
        stub_poll,
        resolve,
        c2i_shared,
        mega_shared,
        dnu,
        must_be_boolean,
        alloc_slow,
        not_entrant,
        deopt_return,
        call_primitive,
        nlr_originate,
        value_dispatch,
        box_double,
        box_float64x2,
        box_float32x4,
        box_int32x4,
    }
}

/// The AArch64 stub table — this file's original `install`, unchanged.
///
/// Compiled on every host (the builders only write bytes, and their
/// encoding tests run everywhere), but only *called* on AArch64.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn install_a64(cache: &mut CodeCache) -> Stubs {
    let call_stub_blob = build_call_stub();
    let h1 = cache
        .alloc(call_stub_blob.code.len())
        .expect("stubs::install: code cache too small for call_stub");
    cache.publish(h1, &call_stub_blob);

    let stub_poll_blob = build_stub_poll();
    let h2 = cache
        .alloc(stub_poll_blob.code.len())
        .expect("stubs::install: code cache too small for stub_poll");
    cache.publish(h2, &stub_poll_blob);

    let stub_resolve_blob = build_stub_resolve();
    let h3 = cache
        .alloc(stub_resolve_blob.code.len())
        .expect("stubs::install: code cache too small for stub_resolve");
    cache.publish(h3, &stub_resolve_blob);

    let c2i_shared_blob = build_c2i_shared();
    let h4 = cache
        .alloc(c2i_shared_blob.code.len())
        .expect("stubs::install: code cache too small for c2i_shared");
    cache.publish(h4, &c2i_shared_blob);

    let mega_shared_blob = build_mega_shared();
    let h5 = cache
        .alloc(mega_shared_blob.code.len())
        .expect("stubs::install: code cache too small for mega_shared");
    cache.publish(h5, &mega_shared_blob);

    let dnu_blob = build_stub_dnu();
    let h6 = cache
        .alloc(dnu_blob.code.len())
        .expect("stubs::install: code cache too small for dnu");
    cache.publish(h6, &dnu_blob);

    let must_be_boolean_blob = build_stub_must_be_boolean();
    let h7 = cache
        .alloc(must_be_boolean_blob.code.len())
        .expect("stubs::install: code cache too small for must_be_boolean");
    cache.publish(h7, &must_be_boolean_blob);

    let alloc_slow_blob = build_stub_alloc_slow();
    let h8 = cache
        .alloc(alloc_slow_blob.code.len())
        .expect("stubs::install: code cache too small for alloc_slow");
    cache.publish(h8, &alloc_slow_blob);

    let not_entrant_blob = build_not_entrant_stub();
    let h9 = cache
        .alloc(not_entrant_blob.code.len())
        .expect("stubs::install: code cache too small for not_entrant");
    cache.publish(h9, &not_entrant_blob);

    let deopt_return_blob = build_deopt_return_trampoline();
    let h10 = cache
        .alloc(deopt_return_blob.code.len())
        .expect("stubs::install: code cache too small for deopt_return");
    cache.publish(h10, &deopt_return_blob);

    let call_primitive_blob = build_stub_call_primitive();
    let h11 = cache
        .alloc(call_primitive_blob.code.len())
        .expect("stubs::install: code cache too small for call_primitive");
    cache.publish(h11, &call_primitive_blob);

    let nlr_originate_blob = build_stub_nlr_originate();
    let h12 = cache
        .alloc(nlr_originate_blob.code.len())
        .expect("stubs::install: code cache too small for nlr_originate");
    cache.publish(h12, &nlr_originate_blob);

    // S24 A2: one `value`-family dispatch stub per argc 0..=3.
    let value_dispatch = std::array::from_fn(|argc| {
        let blob = build_stub_value_dispatch(argc as u64);
        let h = cache.alloc(blob.code.len()).unwrap_or_else(|| {
            panic!("stubs::install: code cache too small for value_dispatch[{argc}]")
        });
        cache.publish(h, &blob);
        h
    });

    let box_double_blob = build_stub_box_double();
    let h13 = cache
        .alloc(box_double_blob.code.len())
        .expect("stubs::install: code cache too small for box_double");
    cache.publish(h13, &box_double_blob);

    let box_f64x2_blob = build_stub_box_float64x2();
    let h14 = cache
        .alloc(box_f64x2_blob.code.len())
        .expect("stubs::install: code cache too small for box_float64x2");
    cache.publish(h14, &box_f64x2_blob);

    let box_f32x4_blob = build_stub_box_float32x4();
    let h15 = cache
        .alloc(box_f32x4_blob.code.len())
        .expect("stubs::install: code cache too small for box_float32x4");
    cache.publish(h15, &box_f32x4_blob);

    let box_i32x4_blob = build_stub_box_int32x4();
    let h16 = cache
        .alloc(box_i32x4_blob.code.len())
        .expect("stubs::install: code cache too small for box_int32x4");
    cache.publish(h16, &box_i32x4_blob);

    Stubs {
        call_stub: h1,
        stub_poll: h2,
        resolve: h3,
        c2i_shared: h4,
        mega_shared: h5,
        dnu: h6,
        must_be_boolean: h7,
        alloc_slow: h8,
        not_entrant: h9,
        deopt_return: h10,
        call_primitive: h11,
        nlr_originate: h12,
        value_dispatch,
        box_double: h13,
        box_float64x2: h14,
        box_float32x4: h15,
        box_int32x4: h16,
    }
}

/// D4: saves x19..x28 + fp/lr (callee-saved, AAPCS64), moves the incoming
/// `entry`/`vm`/`argv`/`argc` (x0..x3) out of the way of the compiled
/// call's own x0..x5 argument registers *before* touching any of them
/// (x0..x3 alias two of the very registers this stub is about to
/// populate — `argv`/`argc` specifically would clobber themselves
/// mid-copy otherwise), then conditionally loads up to 6 words from
/// `argv` into x0..x5 depending on the runtime `argc` value, `blr`s into
/// the compiled entry, and restores. Its own `x29` is the anchor a stack
/// walker stops at (D4's own words) — a completely ordinary frame-pointer
/// chain, nothing extra required for that beyond the standard prologue.
fn build_call_stub() -> CodeBlob {
    let mut a = JasmAssembler::new();

    a.emit("stp", &[x(29), x(30), mem_pre(31, -160)]);
    a.emit("stp", &[x(19), x(20), mem(31, 16)]);
    a.emit("stp", &[x(21), x(22), mem(31, 32)]);
    a.emit("stp", &[x(23), x(24), mem(31, 48)]);
    a.emit("stp", &[x(25), x(26), mem(31, 64)]);
    a.emit("stp", &[x(27), x(28), mem(31, 80)]);
    // Float fast-path residency: compiled code may hold unboxed float temps
    // in d8–d15 (AAPCS64 callee-saved) — preserve them for the Rust caller,
    // exactly like x19–x28 above.
    a.emit("stp", &[d(8), d(9), mem(31, 96)]);
    a.emit("stp", &[d(10), d(11), mem(31, 112)]);
    a.emit("stp", &[d(12), d(13), mem(31, 128)]);
    a.emit("stp", &[d(14), d(15), mem(31, 144)]);
    a.emit("mov", &[x(29), sp()]);

    // Move incoming params to scratch (caller-saved, x9-x11) registers
    // before any of x0..x5 get overwritten below.
    a.emit("mov", &[x(9), x(0)]); // entry
    a.emit("mov", &[x(28), x(1)]); // vm -> x28 (D5.1's own convention)
    a.emit("mov", &[x(10), x(2)]); // argv
    a.emit("mov", &[x(11), x(3)]); // argc

    let args_done = a.new_label();
    for (i, off) in [(1, 0), (2, 8), (3, 16), (4, 24), (5, 32), (6, 40)] {
        a.emit("cmp", &[x(11), imm(i)]);
        a.b_cond(Cond::Lt, args_done);
        a.emit("ldr", &[x((i - 1) as u8), mem(10, off)]);
    }
    a.bind(args_done);

    a.emit("blr", &[x(9)]);

    a.emit("ldp", &[d(14), d(15), mem(31, 144)]);
    a.emit("ldp", &[d(12), d(13), mem(31, 128)]);
    a.emit("ldp", &[d(10), d(11), mem(31, 112)]);
    a.emit("ldp", &[d(8), d(9), mem(31, 96)]);
    a.emit("ldp", &[x(27), x(28), mem(31, 80)]);
    a.emit("ldp", &[x(25), x(26), mem(31, 64)]);
    a.emit("ldp", &[x(23), x(24), mem(31, 48)]);
    a.emit("ldp", &[x(21), x(22), mem(31, 32)]);
    a.emit("ldp", &[x(19), x(20), mem(31, 16)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 160)]);
    a.emit("ret", &[]);

    a.finish()
}

/// D5.6: called via `bl` from compiled code with x28 already `&VmState`
/// (established once by `call_stub`, still live at any `Poll` site) —
/// saves all of x0..x15 (compiled code's own live values, which the
/// `Poll` site's caller expects untouched, S10 P6), calls
/// [`rt_poll`], restores, returns. x16/x17/flags are NOT saved: they're
/// documented assembler/veneer scratch (D3.5), never expected to survive
/// an Ir op boundary in the first place, so a Poll site never relies on
/// them either.
fn build_stub_poll() -> CodeBlob {
    let mut a = JasmAssembler::new();

    // Frame: [x29] = saved caller x29 = LOOP's fp, [x29+8] = saved x30 = the
    // return address INSIDE the loop right after `bl stub_poll` (== the poll's
    // ret_pc). `rt_poll` is a C call and preserves x29, so x29 still names THIS
    // frame after it returns — we reload loop_fp/ret_pc from [x29] not registers.
    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);
    a.emit("sub", &[sp(), sp(), imm(128)]);
    a.emit("stp", &[x(0), x(1), mem(31, 0)]);
    a.emit("stp", &[x(2), x(3), mem(31, 16)]);
    a.emit("stp", &[x(4), x(5), mem(31, 32)]);
    a.emit("stp", &[x(6), x(7), mem(31, 48)]);
    a.emit("stp", &[x(8), x(9), mem(31, 64)]);
    a.emit("stp", &[x(10), x(11), mem(31, 80)]);
    a.emit("stp", &[x(12), x(13), mem(31, 96)]);
    a.emit("stp", &[x(14), x(15), mem(31, 112)]);

    // rt_poll(vm, loop_fp, ret_pc) -> PollOutcome{result:x0, deopted:x1}.
    a.emit("mov", &[x(0), x(28)]); // x0 = &mut VmState
    a.emit("ldr", &[x(1), mem(29, 0)]); // x1 = loop_fp = saved x29
    a.emit("ldr", &[x(2), mem(29, 8)]); // x2 = ret_pc  = saved x30
    let rt_poll_lit = a.literal_u64(rt_poll as *const () as u64, Some(RelocKind::RuntimeAddr));
    a.call_far(rt_poll_lit);

    // deopted == 0 -> normal loop resume (restore x0-x15, ret to the loop).
    // deopted != 0 -> deopt teardown: x0 already holds the deoptee's result;
    // discard BOTH stub_poll's frame AND the loop frame and `ret` the result to
    // the loop's own native caller (the loop activation is gone after a deopt).
    let cont = a.new_label();
    a.cbz(xr(1), cont);

    // --- deopt teardown (mirrors deopt_trap::build_uncommon_trampoline) ---
    // x0 (result) is NOT restored — it must survive to the loop's caller.
    a.emit("ldr", &[x(2), mem(29, 0)]); // x2 = loop_fp (= saved x29)
    a.emit("mov", &[sp(), x(2)]); // sp := loop_fp: drop stub_poll frame + loop locals
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]); // pop loop frame's own {caller_fp, caller_lr}
    a.emit("ret", &[]); // return result (x0) to the loop's caller

    // --- continue: restore the loop's live regs and resume it ---
    a.bind(cont);
    a.emit("ldp", &[x(0), x(1), mem(31, 0)]);
    a.emit("ldp", &[x(2), x(3), mem(31, 16)]);
    a.emit("ldp", &[x(4), x(5), mem(31, 32)]);
    a.emit("ldp", &[x(6), x(7), mem(31, 48)]);
    a.emit("ldp", &[x(8), x(9), mem(31, 64)]);
    a.emit("ldp", &[x(10), x(11), mem(31, 80)]);
    a.emit("ldp", &[x(12), x(13), mem(31, 96)]);
    a.emit("ldp", &[x(14), x(15), mem(31, 112)]);
    a.emit("add", &[sp(), sp(), imm(128)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]);
    a.emit("ret", &[]);

    a.finish()
}

/// D4.1: `stub_resolve`/`stub_ic_miss` — one shared stub, two doors, same
/// tail (S11 step 2 wires the `bl`-initial door as every fresh `IcSite`'s
/// target; the guard-miss `b` door lands S11 step 5 alongside PICs). x30 on
/// entry is the ORIGINAL send site's own return address either way (a `bl`
/// naturally sets it; a guard's `b` doesn't touch it) — captured into x1
/// (`rt_resolve_send`'s `ret_addr` argument) before `call_far`'s own `blr`
/// would otherwise clobber it.
fn build_stub_resolve() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_RESOLVE);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(30)]); // ret_addr (captured before call_far clobbers x30)
    a.emit("sub", &[x(2), x(29), imm(ROOTSPILL_BYTES as i64)]); // argv = &RootSpill
    let rt_resolve_lit = a.literal_u64(
        rt_resolve_send as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(rt_resolve_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16 (P4: survives the epilogue's own x0 reload)
    emit_stub_epilogue(&mut a);
    a.emit("br", &[x(16)]);

    a.finish()
}

/// D4.1/D4.3/D4.4's shared target-resolution step, used by `rt_resolve_send`
/// (both the mono path and every Pic/Mega transition), `rt_mega_lookup`,
/// and — `pub(crate)` specifically for this — `compiler::driver::
/// compile_method`'s own compile-time resolution of a `send_super` site
/// (D4.6): the same "real nmethod or c2i adapter" choice applies whether
/// the resolving call happens at runtime or at compile time.
///
/// `use_verified` picks between a real nmethod's `entry` and its
/// `verified_entry`: a MONO site's `bl` is a static patch that could later
/// see a DIFFERENT receiver klass through the SAME site, so its target
/// must re-verify (`entry`, `use_verified=false`) — but a PIC's own guard
/// chain, or `rt_mega_lookup`'s own fresh `klass_of`+`lookup`, has ALREADY
/// verified this exact (klass, method) pair by the time it's about to
/// jump, so re-checking would be pure redundant overhead (`verified_entry`,
/// `use_verified=true`, D4.3's own "the pair's klass was just compared").
/// A c2i adapter has no guard of its own either way (D6.1), so the
/// distinction is moot there.
pub(crate) fn resolve_target_entry(
    vm: &mut VmState,
    k: KlassOop,
    selector: SymbolOop,
    method: MethodOop,
    use_verified: bool,
) -> u64 {
    // S24 A2 (design §2.3, review MINOR (a)): a `value`-family send routes to
    // the shared per-argc block-dispatch stub, NOT a c2i adapter. Placed
    // FIRST, before the `code_table.lookup` below, so EVERY lattice state
    // inherits it — a fresh Unresolved site, a Mono re-patch, and each pair
    // of a PIC promoted after a non-closure receiver all resolve their
    // `(BlockClosure, value:)` target here identically. (The value
    // primitives are never compiled as methods — `driver::eligible` rejects
    // `primitive() != 0` unless shimmable, and 50..=53 are not shimmable —
    // so the `Some(target_id)` arm below can never fire for them anyway;
    // this check just makes the routing explicit and lattice-uniform.) A
    // `super value:` send never reaches here — it goes through
    // `resolve_super_target_entry`, left untouched, so its interpreted
    // re-dispatch keeps correct super semantics.
    let prim = method.primitive();
    if (50..=53).contains(&prim) {
        let argc = (prim - 50) as usize; // 50->value(0) .. 53->value:value:value:(3)
        debug_assert_eq!(
            selector.as_string().matches(':').count(),
            argc,
            "resolve_target_entry: value primitive {prim} vs selector {} arity disagree",
            selector.as_string()
        );
        return vm.stubs.value_dispatch_addr(argc);
    }
    match vm.code_table.lookup(k, selector) {
        Some(target_id) => {
            let target_nm = vm
                .code_table
                .get(target_id)
                .expect("code_table.lookup just returned this id");
            let off = if use_verified {
                target_nm.verified_entry_off
            } else {
                target_nm.entry_off
            };
            target_nm.code.base as u64 + off as u64
        }
        // D6.1: no compiled nmethod yet -- a c2i adapter, callable exactly
        // like a real nmethod's own `entry` (`bl`, x0=result on return),
        // stands in until (if ever) `method` gets compiled for real.
        None => {
            let c2i_shared_addr = vm.stubs.c2i_shared_addr();
            vm.adapters
                .get_or_make(&mut vm.code_cache, c2i_shared_addr, method)
        }
    }
}

/// S14 step 5: `resolve_target_entry` for a SUPER-send site. A super site
/// links `verified_entry` directly (skipping the entry guard) and its
/// receivers are SUBCLASS instances — any klass below the holder — so it must
/// never enter an nmethod whose code was customized to its key klass
/// (`self_devirt`: guard-free self-send inlines baked against exactly that
/// klass). Such a target dispatches through the method's c2i adapter instead:
/// the interpreter re-dispatches its self-sends against the receiver's REAL
/// klass. Non-devirtualized targets keep the fast compiled link (their code is
/// receiver-klass-independent, the pre-step-5 status quo).
pub(crate) fn resolve_super_target_entry(
    vm: &mut VmState,
    super_klass: KlassOop,
    selector: SymbolOop,
    method: MethodOop,
) -> u64 {
    let devirt = vm
        .code_table
        .lookup(super_klass, selector)
        .and_then(|id| vm.code_table.get(id))
        .is_some_and(|nm| nm.self_devirt);
    if devirt {
        let c2i_shared_addr = vm.stubs.c2i_shared_addr();
        vm.adapters
            .get_or_make(&mut vm.code_cache, c2i_shared_addr, method)
    } else {
        resolve_target_entry(vm, super_klass, selector, method, true)
    }
}

/// Finds the caller nmethod and its own `IcSite` from a send site's return
/// address. `ret_addr - CALL_INSN_LEN` is always the call/guard-miss site
/// itself —
/// D4.1's own invariant, true through every door a compiled send can be
/// re-entered from (a fresh `bl`, a guard's own `b`, a PIC's own indirect
/// tail-call): none of them ever touch x30 except the original `bl`
/// itself. Shared by [`rt_resolve_send`] and [`rt_dnu`] — the latter is
/// reached from EITHER `rt_resolve_send`'s own `stub_resolve` or
/// `rt_mega_lookup`'s own `stub_mega_shared`, both of which restore x30 to
/// the original send's own continuation before tail-jumping onward, so
/// this same derivation is valid starting from either one.
fn find_caller_site(
    vm: &VmState,
    ret_addr: u64,
) -> (NmethodId, CodeHandle, u32, usize, SymbolOop, u8) {
    let caller_id = vm
        .code_table
        .find_by_pc(ret_addr)
        .expect("ret_addr must fall inside a published nmethod (D4.1)");
    let caller_nm = vm
        .code_table
        .get(caller_id)
        .expect("find_by_pc just returned this id");
    let caller_code = caller_nm.code;
    // Back up from the return address to the call instruction's own
    // offset — the width of the call, which is ISA-specific: a 4-byte
    // AArch64 `bl` against a 5-byte x86-64 `E8 rel32`. Getting this wrong
    // does not read a wrong site, it fails to find one at all (the
    // `position` lookup below panics), so it is at least loud.
    let site_off = (ret_addr - CALL_INSN_LEN - caller_code.base as u64) as u32;
    let site_idx = caller_nm
        .ic_sites
        .iter()
        .position(|s| s.off == site_off)
        .unwrap_or_else(|| {
            // Say enough to diagnose without a second run: which method,
            // where we thought the site was, and what sites actually
            // exist. The recorded offsets are the key fact — a site list
            // that BRACKETS `site_off` means the back-up width is wrong
            // for this call shape, while one that misses entirely means
            // the return address came from a door that is not an IC site
            // at all.
            let sites: Vec<String> = caller_nm
                .ic_sites
                .iter()
                .map(|s| format!("{:#x}({})", s.off, s.selector.as_string()))
                .collect();
            let sel = crate::oops::wrappers::SymbolOop::try_from(caller_nm.key_selector.oop())
                .map(|s| s.as_string())
                .unwrap_or_else(|| "?".into());
            // What instruction actually precedes `ret_addr` is the whole
            // question, so decode it rather than describe the offset.
            #[cfg(not(target_arch = "aarch64"))]
            let context = {
                // SAFETY: published, immovable code bytes of this nmethod.
                let bytes = unsafe {
                    std::slice::from_raw_parts(caller_code.base, caller_code.len)
                };
                crate::compiler::disasm_x64::window(
                    bytes,
                    caller_code.base as u64,
                    site_off as usize,
                    3,
                    2,
                    None,
                )
                .join("\n    ")
            };
            #[cfg(target_arch = "aarch64")]
            let context = String::from("(disassembly: aarch64 path)");
            panic!(
                "no IcSite at offset {site_off:#x} in nmethod {caller_id:?} ({sel}) \
                 v{}: ret_addr={ret_addr:#x} base={:#x} len={:#x} \
                 call_len={CALL_INSN_LEN}; recorded sites: [{}]\n    {context}",
                caller_nm.version,
                caller_code.base as u64,
                caller_code.len,
                sites.join(", ")
            )
        });
    let site = &caller_nm.ic_sites[site_idx];
    (
        caller_id,
        caller_code,
        site_off,
        site_idx,
        site.selector,
        site.argc,
    )
}

/// # Safety
/// Only ever reached via `bl`/`b` from `stub_resolve`'s own hand-assembled
/// listing above, never called directly from Rust — same contract as
/// [`rt_poll`]. `argv` points at `stub_resolve`'s own RootSpill area (6
/// live `u64`s, receiver + up to 5 args); valid for the duration of this
/// call only (the stub reloads and may clobber that memory immediately
/// after, on its way to the tail-jump).
///
/// D4.1's full state machine: finds the caller nmethod and its `IcSite`
/// from `ret_addr` (`ret_addr - CALL_INSN_LEN` is always the call/guard-miss site
/// itself — D4.1's own invariant, true through EITHER door), resolves the
/// real target the SAME two-step way an interpreted send does (`klass_of`
/// + `runtime::lookup::lookup`), then transitions:
/// - `Unresolved` / `Mono` (same klass): patch directly, stay/become
///   `Mono{klass, target}` (`resolve_target_entry`'s own `entry`, not
///   `verified_entry` — see that function's doc).
/// - `Mono` (different klass): promote to a fresh 2-entry PIC (D4.3).
///   The OLD pair is re-resolved via `verified_entry` rather than reusing
///   its stored MONO-era target — consistent with the "same klass" arm
///   just above, which ALSO always re-resolves fresh rather than trusting
///   a stale stored value (a redefinition could have moved it since).
/// - `Pic`, room for one more: rebuild with `n+1` pairs (D3.4: PICs are
///   immutable once published, no in-place growth), free the old stub.
/// - `Pic`, already at `PIC_MAX_ENTRIES`: free the PIC, promote to the
///   selector's own (possibly newly-built, possibly shared with other
///   sites) megamorphic trampoline (D4.4) — which never patches anything
///   itself again, so `Mega` can never reach this function (D4.1: "unreachable").
///
/// An interpreted-only target (`code_table.lookup` finds no nmethod) falls
/// back to a c2i adapter (D6.1) at every one of the above — from
/// `IcState`'s own perspective this is no different from a real nmethod's
/// entry, just a different source for a target address. DNU (`lookup`
/// finds nothing) returns `stubs.dnu`'s own address WITHOUT patching the
/// site at all or touching its recorded state (D5.4) — a LATER call
/// through the same site, with a different receiver klass, might still
/// resolve successfully, so nothing here may assume this klass is
/// permanent the way a real resolution does.
pub unsafe extern "C" fn rt_resolve_send(vm: *mut VmState, ret_addr: u64, argv: *mut u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_resolve`.
    let vm = unsafe { &mut *vm };
    // P9, checked from the inside: any stub reaching Rust must have set
    // the anchor before the call (a bare `VmState::reg_block` starts at
    // 0, and `enter_compiled` debug-asserts it's clear on ENTRY — this is
    // the complementary check, that it's genuinely non-zero while a
    // runtime call is actually in flight, not just cleared again on the
    // way out).
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_resolve_send: anchor must be set by stub_resolve's prologue before this call"
    );

    // S15 A8: every arrival here IS a compiled-IC miss (Unresolved first
    // touch, or a mono/pic guard rejecting the receiver) — hits never leave
    // compiled code, so misses are the whole observable.
    vm.stats.ic_misses += 1;
    let (caller_id, caller_code, site_off, site_idx, selector, argc) =
        find_caller_site(vm, ret_addr);
    let site = &vm.code_table.get(caller_id).unwrap().ic_sites[site_idx];
    let prev_state = site.state;
    let super_klass = site.super_klass;

    // DBG5 (compiled_send_auditor_design.md §1): FORCE-COLD inline caches.
    // When the compiled-send auditor is armed (`MACVM_TRACE=calls`, and later
    // step-call), compute+return the correct target but NEVER memoize it —
    // skip `patch_branch26_at` and the `IcState` write below, so the site
    // stays `Unresolved` and the NEXT send re-enters here, making every
    // compiled send observable at this one choke point with zero codegen
    // change. Observation-only: the `Unresolved` arm is the already-correct
    // general dispatch, so results are byte-identical — only speed and
    // visibility change (the sprint's headline gate). Either the `calls`
    // tracer (§D1) or the interactive `step-call` auditor (§D3) forces cold.
    let trace_calls = vm.options.trace.is_enabled("calls");
    let step_calls = vm.debug.step_calls;
    let force_cold = trace_calls || step_calls;

    // S13 step 10d: a `send_super` site re-dispatches from its STATIC
    // holder-superclass (D4.6), NEVER the receiver's klass — skipping subclass
    // overrides is the whole point of `super`. Its `bl` reached here only
    // because its compile-time-resolved target was invalidated (patched to
    // `not_entrant_stub`) or flushed to `Unresolved`; re-resolve the SAME static
    // super binding and re-point the `bl`, staying `Mono{super_klass, target}`.
    // This bypasses the receiver-klass dynamic lookup + poly/mega machinery
    // below entirely — a super site is monomorphic by construction. Without
    // this, the site would collapse into an ordinary dynamic send reaching the
    // very override `super` was meant to skip (the step-8-deferred blocker).
    if let Some(sk) = super_klass {
        let Some(method) = lookup(vm, sk, selector) else {
            return vm.stubs.dnu_addr();
        };
        let target = resolve_super_target_entry(vm, sk, selector, method);
        if !force_cold {
            vm.code_cache
                .patch_call_site_at(caller_code, site_off, target);
            vm.code_table.get_mut(caller_id).unwrap().ic_sites[site_idx].state =
                IcState::Mono { klass: sk, target };
        }
        return target;
    }

    // SAFETY: this function's own contract above -- `argv[0]` is always
    // the receiver, D4.1's register protocol.
    let receiver = Oop::from_raw(unsafe { *argv });
    let k = klass_of(vm, receiver);

    let Some(method) = lookup(vm, k, selector) else {
        return vm.stubs.dnu_addr();
    };

    // DBG5 (compiled_send_auditor_design.md §1): under the auditor, dispatch
    // EXACTLY as a first-touch `Unresolved` site would and return — WITHOUT
    // entering the `match prev_state` below. That match's poly arms
    // (`Mono→Pic`, `Pic→Pic`) build and FREE PIC stubs eagerly *while
    // computing* `patch` (before any gate), so merely skipping the final
    // `patch_branch26_at` would leave a site whose freed stub is still named
    // by its `bl` — a dangling jump the byte-identical gate can't see. Ordinary
    // sites are born `Unresolved` (driver.rs: `super_resolutions` is `Some`
    // only for super sites) and force-cold keeps them there, so those arms are
    // unreachable today; short-circuiting here makes that robust by
    // construction against future compile-time IC seeding, not just by that
    // emergent invariant. `resolve_target_entry(..false)` is the identical
    // call the `Unresolved` arm makes.
    if force_cold {
        debug_assert!(
            matches!(prev_state, IcState::Unresolved),
            "force-cold reached an ordinary site in state {prev_state:?}, not Unresolved — \
             compile-time IC seeding changed the born state; the short-circuit stays correct \
             (fresh resolve, no memoization) but this tripwire flags that the poly-arm \
             free/build machinery is now reachable and its gating must be re-audited"
        );
        let t = resolve_target_entry(vm, k, selector, method, false);
        // §D1 tracer: `MACVM_TRACE=calls` logs one line per (now every) send.
        if trace_calls {
            let sel = selector.as_string();
            let kname = crate::memory::print_oop(&vm.universe, k.name());
            let tier = if method.primitive() != 0 {
                format!("prim({})", method.primitive())
            } else if vm.code_table.lookup(k, selector).is_some() {
                "C".to_string()
            } else {
                "I".to_string()
            };
            eprintln!(
                "[calls] nm={}#{site_idx} #{sel} recv={kname} → {tier} argc={argc}",
                caller_id.0
            );
        }
        // §D3 interactive auditor: stop at this send boundary and service
        // read-only inspection commands (reentrancy-guarded inside).
        if step_calls {
            crate::runtime::debug::step_call_prompt(vm, caller_id, site_idx, selector, k, argc);
        }
        return t;
    }

    // `patch_target`: what the SITE's own `bl` gets repointed to (governs
    // every FUTURE call through it). `ret_target`: the address THIS
    // dispatch tail-jumps to right now -- the two differ only on
    // promotion to Mega (the site is patched to the shared trampoline,
    // but this call already has (k, method) in hand, so it skips straight
    // to the real target instead of bouncing through the trampoline's own
    // fresh lookup).
    // `patch` is `None` ONLY on code-cache exhaustion in a poly arm (S24
    // A1 hardening): the site is left EXACTLY as it was — old patch
    // target, old state, old PIC stub (if any) still installed — and this
    // one dispatch proceeds to `ret_target` directly. Every further miss
    // re-enters here and re-attempts; if space frees up (a full-GC sweep
    // of drained NotEntrant nmethods), the site heals itself. Slow,
    // correct, and it can never abort the VM the way the old
    // `expect`-on-alloc did (deltablue t=1 under DEOPT_STRESS filled the
    // cache with everything working as designed).
    let (patch, ret_target) = match prev_state {
        IcState::Unresolved => {
            let t = resolve_target_entry(vm, k, selector, method, false);
            (
                Some((
                    t,
                    IcState::Mono {
                        klass: k,
                        target: t,
                    },
                )),
                t,
            )
        }
        IcState::Mono { klass, .. } if klass == k => {
            let t = resolve_target_entry(vm, k, selector, method, false);
            (
                Some((
                    t,
                    IcState::Mono {
                        klass: k,
                        target: t,
                    },
                )),
                t,
            )
        }
        IcState::Mono { klass: old_k, .. } => {
            let old_method = lookup(vm, old_k, selector)
                .expect("a klass that was already Mono-resolved must still resolve to a method");
            let old_t = resolve_target_entry(vm, old_k, selector, old_method, true);
            let new_t = resolve_target_entry(vm, k, selector, method, true);
            let resolve_addr = vm.stubs.resolve_addr();
            let smi_klass_bits = vm.universe.smi_klass.oop().raw();
            let patch = vm
                .pic_table
                .build(
                    &mut vm.code_cache,
                    smi_klass_bits,
                    resolve_addr,
                    vec![(old_k, old_t), (k, new_t)],
                )
                .map(|stub| (stub.base as u64, IcState::Pic { stub }));
            (patch, new_t)
        }
        IcState::Pic { stub } => {
            let mut pairs = vm.pic_table.pairs_of(stub).to_vec();
            let new_t = resolve_target_entry(vm, k, selector, method, true);
            // A klass ALREADY in the PIC re-resolving means its baked
            // target went stale (its nmethod was invalidated/recompiled —
            // DEOPT_STRESS does this constantly; organic recompiles do it
            // occasionally): RE-KEY that pair in place. Appending instead
            // would (a) be unreachable dispatch — the guard chain takes
            // the FIRST match — and (b) collapse to a DUPLICATE klass
            // pool word via `literal_u64`'s by-value interning, so
            // `klass_pool_offs` lists one slot twice and `each_code_root`
            // visits it twice per collection — the second visit hands the
            // just-forwarded TO-SPACE address back to `scavenge_oop`,
            // tripping its double-copy guard (deltablue at threshold=1
            // under DEOPT_STRESS; latent since S13's lazy invalidation,
            // unmasked by A1's block-compile volume).
            if let Some(existing) = pairs.iter_mut().find(|p| p.0 == k) {
                existing.1 = new_t;
                let resolve_addr = vm.stubs.resolve_addr();
                let smi_klass_bits = vm.universe.smi_klass.oop().raw();
                let patch = vm
                    .pic_table
                    .build(&mut vm.code_cache, smi_klass_bits, resolve_addr, pairs)
                    .map(|new_stub| {
                        vm.pic_table.free(&mut vm.code_cache, stub);
                        (new_stub.base as u64, IcState::Pic { stub: new_stub })
                    });
                (patch, new_t)
            } else if pairs.len() < PIC_MAX_ENTRIES {
                pairs.push((k, new_t));
                let resolve_addr = vm.stubs.resolve_addr();
                let smi_klass_bits = vm.universe.smi_klass.oop().raw();
                // Build FIRST, free the old stub only on success — on
                // exhaustion the old (still-installed) PIC keeps serving
                // its existing klasses; only `k` re-resolves per miss.
                let patch = vm
                    .pic_table
                    .build(&mut vm.code_cache, smi_klass_bits, resolve_addr, pairs)
                    .map(|new_stub| {
                        vm.stats.pic_extends += 1;
                        vm.pic_table.free(&mut vm.code_cache, stub);
                        (new_stub.base as u64, IcState::Pic { stub: new_stub })
                    });
                (patch, new_t)
            } else {
                let mega_shared_addr = vm.stubs.mega_shared_addr();
                // Same order: mint the mega trampoline BEFORE freeing the
                // full PIC it replaces.
                let patch = vm
                    .mega_table
                    .get_or_make(&mut vm.code_cache, mega_shared_addr, selector)
                    .map(|mega_stub| {
                        vm.stats.mega_transitions += 1;
                        vm.pic_table.free(&mut vm.code_cache, stub);
                        (mega_stub.base as u64, IcState::Mega { stub: mega_stub })
                    });
                (patch, new_t)
            }
        }
        IcState::Mega { .. } => unreachable!(
            "rt_resolve_send: a Mega site's own rt_mega_lookup never patches, so a site can \
             never return to stub_resolve once megamorphic (D4.4)"
        ),
    };

    // force_cold never reaches here — it returns at the short-circuit above,
    // deliberately BEFORE this poly machinery (see §1 rationale there).
    if let Some((patch_target, new_state)) = patch {
        vm.code_cache
            .patch_call_site_at(caller_code, site_off, patch_target);
        vm.code_table
            .get_mut(caller_id)
            .expect("caller nmethod is still installed -- this call is running ON it")
            .ic_sites[site_idx]
            .state = new_state;
    } else {
        vm.stats.ic_patch_declined_cache_full += 1;
    }

    // Debugger (DBG3 companion): `MACVM_DBG_RESOLVE=1` traces every
    // compiled-IC miss resolution — receiver klass, prior site state, and
    // whether the resolved target is a c2i adapter (`c2i=true` means the
    // callee runs interpreted behind this site). Debug builds only, stderr.
    // One line per rt_resolve_send arrival; hits never leave compiled code,
    // so this IS the complete compiled-dispatch transition log.
    #[cfg(debug_assertions)]
    if std::env::var("MACVM_DBG_RESOLVE").is_ok() {
        let sel = selector.as_string();
        let is_c2i = vm.code_table.lookup(k, selector).is_none();
        let kname = crate::memory::print_oop(&vm.universe, k.name());
        let patch_desc = match &patch {
            Some((t, _)) => format!("{t:#x}"),
            None => "declined(cache-full)".to_string(),
        };
        eprintln!(
            "RESOLVE sel=#{sel} recv={:#x} klass={kname} prev={prev_state:?} \
             c2i={is_c2i} ret_target={ret_target:#x} patch_target={patch_desc}",
            receiver.raw(),
        );
    }

    ret_target
}

/// S13 D1 §2b: `not_entrant_stub`. A now-`NotEntrant` nmethod's `entry` and
/// `verified_entry` are patched (`nmethod::make_not_entrant`) to `b
/// not_entrant_stub`. A future call reaches it through a compiled caller's
/// still-live `bl` (whose target is the old entry): at that instant x0..x5
/// hold the receiver+args exactly as the caller set them (the patched `b` is
/// the FIRST instruction at the entry — nothing has run yet), and x30 is the
/// caller's own send-site return address (a `b` doesn't touch x30, so it is
/// still whatever the caller's `bl` set — the SAME invariant `stub_resolve`'s
/// `bl`-door relies on).
///
/// The whole re-dispatch job — find the caller's `IcSite` from x30, re-resolve
/// (receiver klass, selector) via `code_table.lookup`, patch the site's `bl`
/// to the CURRENT method's target, tail-jump there — is EXACTLY
/// [`rt_resolve_send`]'s existing D4.1 machine, with nothing about it specific
/// to a "miss" versus an "invalidated entry": the MONO re-resolve arm
/// (`Mono {klass,..} if klass == k`) already re-runs a fresh `lookup`, so a
/// site whose `bl` had been patched straight to the OLD (now-NotEntrant) entry
/// gets repointed to whatever `code_table.lookup(k, selector)` returns TODAY —
/// the redefined method's fresh nmethod, or a c2i adapter if it's no longer
/// compiled. So this stub is byte-for-byte `stub_resolve`'s own listing rather
/// than a new runtime function: same prologue, same `rt_resolve_send`, same
/// tail-jump. A distinct handle (not just an alias of `resolve`) is kept only
/// so the invalidation path has its OWN stable branch target, decoupled from
/// any future divergence of the fresh-`bl` door.
///
/// **KNOWN LIMITATION — `send_super` re-dispatch (step-10 blocker).** This
/// reuse is correct for ORDINARY sends (Mono/PIC/Mega/Unresolved/c2i, all
/// adversarially verified S13 step 8). It is WRONG for a compiled `send_super`
/// site: `rt_resolve_send` re-resolves via `lookup(klass_of(self), selector)`,
/// but a super site must resolve from its STATIC holder-superclass — and the
/// runtime `IcSite` carries no super marker (`static_klass` lives only in the
/// discarded compile-time `CallSiteInfo`). So if a super-send's TARGET nmethod
/// is made `NotEntrant`, the super site silently collapses into a dynamic send
/// reaching a subclass override `super` was meant to skip. LATENT until a
/// super-target is invalidated (only the step-10 redefinition trigger does
/// that; no step-8 test hits it). The fix belongs with step 10: give the
/// runtime `IcSite` a super/static-klass marker so re-resolution stays
/// super-aware, exactly as `driver::compile_method`'s own super pre-resolution
/// already is at compile time.
fn build_not_entrant_stub() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_RESOLVE);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(30)]); // ret_addr (captured before call_far clobbers x30)
    a.emit("sub", &[x(2), x(29), imm(ROOTSPILL_BYTES as i64)]); // argv = &RootSpill
    let rt_resolve_lit = a.literal_u64(
        rt_resolve_send as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(rt_resolve_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16 (P4: survives the epilogue's own x0 reload)
    emit_stub_epilogue(&mut a);
    a.emit("br", &[x(16)]);

    a.finish()
}

/// S13 D6: `deopt_return_trampoline` — the RETURN-path sibling of
/// `codecache::deopt_trap::build_uncommon_trampoline`. A callee's `ret` lands
/// here when its saved-LR slot was redirected by §2c
/// ([`crate::runtime::frames::redirect_returns_into_nm`]) because its caller is
/// an in-flight activation of a now-`NotEntrant` nmethod.
///
/// **Entry state** (established by the returning callee's own epilogue
/// `ldp x29,x30,[sp],#16; ret`): `x0` = the callee's result oop; `x29` = the
/// VICTIM (nm-activation) FP — the `ldp` restored fp to the callee's caller,
/// which IS the victim; `[x29+8]` = the victim's OWN caller's return address
/// (untouched — §2c redirected the CALLEE's slot, never the victim's own).
/// `x28` = `&VmState` (preserved across the whole compiled activation, AAPCS64
/// callee-saved). This is exactly why `fp == victim_fp` holds here.
///
/// Shape vs [`crate::codecache::deopt_trap::build_uncommon_trampoline`] (D4):
/// both build a walker-visible frame record (`[fp]=victim_fp`,
/// `[fp+8]=<pc in victim nm>`) so a GC during the Rust call sees the victim
/// compiled frame at its safepoint, and both tear the record + victim frame
/// down and `ret` the deopt result to the victim's own caller. The RETURN path
/// differs in two ways: (1) the classifying pc for the record is `orig_ret_pc`
/// from `pending_deopts` (fetched via
/// [`crate::codecache::deopt_trap::rt_deopt_return_pc`], a pure lookup), where
/// D4 had the trap pc in `x16` from the signal handler; and (2) the callee's
/// result flows THROUGH — saved in a callee-saved reg across the helper call,
/// then handed to [`crate::codecache::deopt_trap::rt_deopt_on_return`] as
/// `incoming_result`.
///
/// ```text
///   mov  x19, x0                 // save callee result (x19 callee-saved)
///   mov  x20, x29                // save victim_fp     (x20 callee-saved)
///   mov  x0, x28                 // vm
///   mov  x1, x20                 // victim_fp
///   ldr  x16,<rt_deopt_return_pc>; blr x16   // x0 := orig_ret_pc (no alloc)
///   mov  x30, x0                 // saved-lr := orig_ret_pc
///   stp  x29, x30, [sp,#-16]!    // record: [fp]=victim_fp, [fp+8]=orig_ret_pc
///   mov  x29, sp                 //   re-root fp onto the record
///   str  x29, [x28, #LAST_FP]    // publish the record as the walker's anchor
///   str  x30, [x28, #LAST_PC]    //   (task #92, same as the uncommon trampoline)
///   movz x9, #KIND_DEOPT_BRIDGE
///   str  x9,  [x28, #LAST_KIND]
///   mov  x0, x28                 // vm
///   mov  x1, x20                 // victim_fp
///   mov  x2, x19                 // result
///   ldr  x16,<rt_deopt_on_return>; blr x16   // x0 := deopt result oop bits
///   str  xzr, [x28, #LAST_FP]    // clear the anchor (P9)
///   str  xzr, [x28, #LAST_KIND]
///   mov  sp, x20                 // tear down record + victim frame
///   ldp  x29, x30, [sp], #16     // pop the VICTIM's own saved {fp,lr}
///   ret                          // return the deopt result to the victim's caller
/// ```
///
/// The teardown (`sp := victim_fp`) discards BOTH the trampoline record and the
/// whole victim compiled frame, landing `sp` on the victim's own saved
/// `{fp,lr}` pair (its caller's fp + real return address), which the `ldp` pops
/// so the final `ret` returns the deopt result to the victim's ORIGINAL caller
/// — the victim activation is gone after a return-path deopt, exactly as D4's
/// trapped activation is gone after an uncommon-trap deopt. PAC is off
/// (`arm64.md` §5), so rewriting `x30`/frame surgery needs no signing.
fn build_deopt_return_trampoline() -> CodeBlob {
    let mut a = JasmAssembler::new();

    // Save the two live inputs into callee-saved regs so they survive both
    // calls: x19 = callee result, x20 = victim_fp.
    a.emit("mov", &[x(19), x(0)]);
    a.emit("mov", &[x(20), x(29)]);

    // Fetch orig_ret_pc (the victim's original return address into the nm) via
    // a pure map lookup — no allocation, so safe to call before the walker-
    // visible record exists.
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(20)]); // victim_fp
    let pc_lit = a.literal_u64(
        crate::codecache::deopt_trap::rt_deopt_return_pc as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(pc_lit);

    // Build the D4-shape walker-visible frame record: [fp]=victim_fp,
    // [fp+8]=orig_ret_pc, then re-root fp onto it so a GC inside
    // rt_deopt_on_return classifies the victim frame at its return safepoint.
    a.emit("mov", &[x(30), x(0)]); // saved-lr := orig_ret_pc
    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);
    // PUBLISH the record as the walker's anchor (task #92, the same hole as
    // `build_uncommon_trampoline`'s — see the KIND_DEOPT_BRIDGE doc): the
    // record alone is invisible to `walk_frames`' anchor-driven start rule,
    // so a GC inside `rt_deopt_on_return`'s `deoptimize_frame` materializer
    // allocations would assert ("no anchor is set") — or, without the
    // assert, skip the victim frame's oops. x9 scratch per
    // `emit_stub_kind_tag`'s precedent (dead here: the callee has fully
    // returned, the victim is being abandoned).
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

    // Marshal (vm, victim_fp, result) and run the return-path deopt.
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(20)]); // victim_fp (the pending_deopts key)
    a.emit("mov", &[x(2), x(19)]); // result (incoming_result)
    let rt_lit = a.literal_u64(
        crate::codecache::deopt_trap::rt_deopt_on_return as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(rt_lit);

    // CLEAR the anchor (P9), exactly as `emit_stub_epilogue`: xzr into fp +
    // kind. x0 (the deopt result / NLR sentinel) is untouched.
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_FP_OFFSET as i64)],
    );
    a.emit(
        "str",
        &[x(31), mem(28, VMREG_LAST_COMPILED_KIND_OFFSET as i64)],
    );

    // Teardown (D6): sp := victim_fp discards the trampoline record AND the
    // whole victim frame; the ldp pops the victim's OWN saved {fp,lr}; ret
    // returns the deopt result (still in x0) to the victim's original caller.
    a.emit("mov", &[sp(), x(20)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]);
    a.emit("ret", &[]);

    a.finish()
}

/// D6.1: the shared c2i adapter tail. Reused unchanged from
/// [`emit_stub_prologue`]/[`emit_stub_epilogue`] rather than a hand-rolled
/// frame — "every stub that can reach Rust follows the same skeleton"
/// ([`Stubs`]'s own doc) holds for this one too, and RootSpill (x0..x5,
/// `sp`-relative, contiguous ascending once the prologue's `sub sp` runs)
/// is exactly the `argv` shape [`rt_interpret_call`] needs, for free. The
/// ONE thing that differs from `stub_resolve`'s shape: this is a
/// callee-shaped stub (a plain `ret` with x0=result, matching how the
/// ORIGINAL compiled call site's own `bl` expects an ordinary nmethod
/// `entry` to behave), not a tail-jumper — so the real result is parked in
/// x16 (spared by the epilogue, same reasoning as `stub_resolve`'s own use
/// of it, P4) across the epilogue's own x0..x5 reload, which would
/// otherwise clobber it right back to the pre-call receiver.
///
/// `c2i_<method>` (`codecache::adapters::build_c2i_adapter`) is what
/// carries the target method's own oop here, already loaded into x17 by
/// the time its own `br` lands — moved straight into x1
/// (`rt_interpret_call`'s `method_bits` argument) rather than spilled to a
/// stack slot: it only needs to survive the few instructions between
/// `c2i_<method>`'s own `ldr x17,<pool>` and this stub's `mov x1,x17`,
/// which nothing in between touches.
fn build_c2i_shared() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_C2I);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(17)]); // method_bits, carried through from c2i_<method>
    a.emit("mov", &[x(2), sp()]); // argv = &RootSpill (x0..x7, contiguous)
    let rt_interpret_call_lit = a.literal_u64(
        rt_interpret_call as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(rt_interpret_call_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr`/`bl` from `c2i_shared`'s own hand-assembled
/// listing above (through `call_far`), never called directly from Rust —
/// same contract as [`rt_poll`]/[`rt_resolve_send`]. `argv` points at
/// `c2i_shared`'s own RootSpill area (8 live `u64`s, receiver + up to 7
/// args, contiguous ascending — matching `emit_call_send`'s x0..x7
/// marshaling and eligibility's receiver+7 send-site cap); valid for the
/// duration of this call only.
///
/// D6.1's C→I path: resolves `method_bits` to a real `MethodOop` (trusted
/// — it came from the adapter's own `RelocKind::Oop` pool word, which
/// `AdapterTable::get_or_make` only ever populates from a genuine
/// `MethodOop`), reads exactly `method.argc()` real args off `argv`
/// (deliberately NOT a separately-passed argc parameter — D5's own
/// pseudocode signature suggests one, but a method's own arity is a fixed
/// property of the method oop, always the SAME value this call's own
/// adapter was generated for, so there is nothing independent for a
/// second value to cross-check), pushes `TierLink::IntoInterpreter` (the
/// anchor D6.3's later NLR unwinder will need to find this boundary),
/// runs the method via `interpreter::run_method_reentrant` (NOT plain
/// `run_method` — this call may itself be nested inside an OUTER,
/// currently-paused interpreter activation reached via an earlier I→C
/// transition; `run_method_reentrant`'s own doc explains why that
/// distinction matters), pops the `TierLink`, and returns the raw result.
pub unsafe extern "C" fn rt_interpret_call(
    vm: *mut VmState,
    method_bits: u64,
    argv: *const u64,
) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `c2i_shared`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_interpret_call: anchor must be set by c2i_shared's prologue before this call"
    );

    let baked_method = MethodOop::try_from(Oop::from_raw(method_bits)).expect(
        "rt_interpret_call: method_bits must be a genuine MethodOop -- AdapterTable::get_or_make's own RelocKind::Oop pool word",
    );
    // The selector is fixed by the adapter's baked method; the true arity is a
    // property of the selector, so it is the SAME whichever method actually
    // runs below.
    let real_argc = baked_method.argc();
    // SAFETY: this function's own contract above -- argv[0] is the
    // receiver, argv[1..=real_argc] the real Smalltalk args, D4.1's
    // register protocol (shared with `rt_resolve_send`'s own argv).
    let receiver = Oop::from_raw(unsafe { *argv });
    let args: Vec<Oop> = (0..real_argc)
        .map(|i| Oop::from_raw(unsafe { *argv.add(1 + i) }))
        .collect();

    // C2I DISPATCH TRUTH. A c2i adapter carries NO receiver-klass guard, unlike
    // a real nmethod `entry` (`emit_entry_guard`). A compiled Mono/Pic IC site
    // therefore can dispatch a receiver of the WRONG klass straight into this
    // adapter: the site was patched to `Mono{K1 -> c2i_M1}` (D6.1 -- an
    // interpreted callee resolves to its adapter, not a guarded entry), then
    // later called with a K2 receiver. A real entry would have guarded-and-
    // missed (re-resolve -> PIC); the adapter, having no guard, would otherwise
    // run `M1` unconditionally on the K2 receiver -- a SILENT wrong answer
    // (e.g. `Mono{True -> c2i_(True>>not)}` executing `True>>not` = `^false` for
    // a `false` receiver, where `False>>not` = `^true` was due). So the baked
    // method is only a HINT: for an ORDINARY send the truth is dynamic lookup on
    // the receiver's ACTUAL klass (this also transparently handles an override
    // arriving at an inherited method's shared adapter, and a redefinition the
    // nested run performed). ONLY a `super` send legitimately runs the baked
    // ANCESTOR method that `lookup(receiver_klass)` would skip -- detected by
    // the calling site's `super_klass` marker. The common healthy case
    // (`lookup == baked`) needs no site inspection at all.
    let sel = SymbolOop::try_from(baked_method.selector())
        .expect("a method's selector is always a Symbol");
    let k = klass_of(vm, receiver);
    let method = match lookup(vm, k, sel) {
        Some(m) if m.oop().raw() == baked_method.oop().raw() => baked_method,
        looked_up => {
            // Divergence: either a `super` send (dispatch from the site's
            // STATIC super klass) or a klass-mismatched ordinary dispatch /
            // override / redefinition (run what lookup found). Disambiguate
            // via the calling site.
            let ret_addr = vm.reg_block.last_compiled_pc;
            let super_k = vm.code_table.find_by_pc(ret_addr).and_then(|caller_id| {
                let (_, _, _, site_idx, _, _) = find_caller_site(vm, ret_addr);
                vm.code_table.get(caller_id).unwrap().ic_sites[site_idx].super_klass
            });
            if let Some(sk) = super_k {
                // Mono-SUPER c2i staleness fix (2026-07 review): the baked
                // method is only the COMPILE-TIME resolution of
                // `lookup(super_klass, sel)` — an ancestor redefinition
                // installs a fresh MethodOop that key-selector invalidation
                // never propagates here (the caller's own key is its own
                // selector; the adapter lives outside `code_table`; the
                // escape hatch below skips super sites). Re-run the same
                // lookup the compile ran (`driver.rs`'s super resolution,
                // and what `rt_resolve_send` already does for the nmethod
                // super case) so a redefined ancestor body takes effect on
                // the next call instead of never. `None` (the ancestor
                // method was removed outright) falls back to the baked oop,
                // the same pathological-edge convention as the ordinary arm.
                lookup(vm, sk, sel).unwrap_or(baked_method)
            } else {
                // `None` (receiver genuinely does not understand the selector)
                // is a pathological polymorphic-site edge; fall back to the
                // baked method, exactly as the pre-fix code always did.
                looked_up.unwrap_or(baked_method)
            }
        }
    };

    // Captured BEFORE the nested run: the interpreted body may itself enter
    // and leave compiled code, clobbering the anchor the repatch below needs.
    let ret_addr = vm.reg_block.last_compiled_pc;
    // The nested run below can trigger a MOVING GC (scavenge or full), so the
    // raw `method`/`receiver` oops the escape hatch needs AFTERWARDS are held
    // through handles for the duration — the same discipline
    // `run_method_reentrant` itself applies to the saved outer method
    // (review finding: post-run reads through the pre-run raw copies would
    // dangle after a collection).
    let (result, method, receiver) = {
        let scope = crate::memory::handles::HandleScope::enter(vm);
        let method_h = scope.handle(vm, method.oop());
        let receiver_h = scope.handle(vm, receiver);
        vm.tier_links.push(TierLink::IntoInterpreter {
            compiled_fp: vm.reg_block.last_compiled_fp,
            compiled_ret_pc: vm.reg_block.last_compiled_pc,
        });
        let result = crate::interpreter::run_method_reentrant(vm, method, receiver, &args);
        vm.tier_links.pop();
        let method = MethodOop::try_from(method_h.get(vm))
            .expect("rt_interpret_call: the c2i adapter's method survived the nested run");
        (result, method, receiver_h.get(vm))
    };
    // S14 step 5 (the c2i escape hatch): a method reached ONLY through
    // compiled c2i calls never passes the interpreter's own compile trigger
    // (`activate_method`) — before step 5, some caller always trapped into
    // the interpreter eventually and warmed things up, but a fully
    // devirtualized caller never deopts, so without this the callee would
    // interpret FOREVER behind a frozen c2i site (the bench-smoke tripwire
    // caught exactly that: arith collapsing back to interpreter speed).
    // Mirror the trigger here: bump the invocation counter, compile (or
    // reuse) the (receiver-klass, selector) nmethod past the threshold, and
    // re-point the calling MONO site at the fresh compiled entry so every
    // FUTURE call dispatches compiled. This runs AFTER the interpreted run
    // below on purpose: that run just WARMED the method's own inner ICs, so
    // the compile sees real feedback (compiling before it would bake a cold
    // trap-laden v0 whose second trap re-executes an arbitrarily long call
    // interpreted — the arith bench's own 5M-iteration worst case).
    if let crate::runtime::vm_state::JitMode::Threshold(n) = vm.options.jit {
        let bumped = crate::interpreter::send::bump_invocation(method);
        if bumped >= n as i64 && !method.compile_disabled() {
            let k = klass_of(vm, receiver);
            let sel = SymbolOop::try_from(method.selector())
                .expect("a method's selector is always a Symbol");
            // DISPATCH-TRUTH guard (review finding): compile+install ONLY when
            // this method is what dynamic lookup for (k, sel) actually finds.
            // A SUPER-dispatched c2i callee is an ANCESTOR method — installing
            // it under the SUBCLASS key would make every future normal send of
            // sel to a k receiver execute the ancestor instead of the
            // override; likewise a method the nested run just REDEFINED must
            // not be resurrected under the fresh key.
            let is_dispatch_truth =
                lookup(vm, k, sel).is_some_and(|m2| m2.oop().raw() == method.oop().raw());
            if !is_dispatch_truth {
                return result.raw();
            }
            let existing = vm.code_table.lookup(k, sel).filter(|&id| {
                vm.code_table
                    .get(id)
                    .is_some_and(|nm| matches!(nm.state, crate::codecache::nmethod::NmState::Alive))
            });
            let compiled =
                existing.or_else(|| crate::compiler::driver::compile_method(vm, k, method));
            if let Some(id) = compiled {
                // Re-point the calling site's `bl` (its return address is the
                // anchored last_compiled_pc) — but ONLY a Mono site whose
                // recorded klass is this receiver's: a Pic/Mega site's shared
                // stub serves other klasses too and is left to its own
                // machinery. The patched target is the FULL `entry` (klass
                // guard included), same as rt_resolve_send's mono link.
                if vm.code_table.find_by_pc(ret_addr).is_some() {
                    let (caller_id, caller_code, site_off, site_idx, site_sel, _argc) =
                        find_caller_site(vm, ret_addr);
                    let site = &vm.code_table.get(caller_id).unwrap().ic_sites[site_idx];
                    let is_mono_k = matches!(site.state, IcState::Mono { klass, .. } if klass == k);
                    // Task #93: `ret_addr` was captured BEFORE the nested run,
                    // and the nested run can invalidate, FLUSH, and REUSE the
                    // caller's code-cache region for a brand-new nmethod (under
                    // MACVM_DEOPT_STRESS, routinely). `find_by_pc` then names
                    // the NEW occupant, and if the same offset happens to be
                    // one of ITS IcSites, patching it links a site for a
                    // DIFFERENT selector straight to this callee's entry —
                    // whose klass guard happily passes for any same-klass
                    // receiver: the wrong METHOD runs with a valid receiver
                    // (instvar indices out of range, phantom DNUs, garbage
                    // smis into arithmetic — task #93's whole zoo). The
                    // selector equality check makes the patch safe: a reused
                    // region's coincidental site can only match if it sends
                    // the SAME selector, in which case the link is correct
                    // for it too.
                    let site_state = site.state;
                    let site_super = site.super_klass;
                    if is_mono_k && site.super_klass.is_none() && site_sel == sel {
                        let nm = vm.code_table.get(id).expect("just compiled/looked up");
                        let target = nm.code.base as u64 + nm.entry_off as u64;
                        vm.code_cache
                            .patch_call_site_at(caller_code, site_off, target);
                        vm.code_table.get_mut(caller_id).unwrap().ic_sites[site_idx].state =
                            IcState::Mono { klass: k, target };
                    } else if let (IcState::Pic { stub }, None) = (site_state, site_super) {
                        // S24 B-phase (stale-PIC-c2i fix; mechanism confirmed
                        // by the fb01b7a review's understand pass): this
                        // site's PIC has a pair (k -> c2i adapter) baked
                        // BEFORE the callee compiled. The PIC guard HITS for
                        // k, so `rt_resolve_send`'s re-key arm is unreachable
                        // for it forever — adapters have no not-entrant
                        // lifecycle, PIC growth copies pairs verbatim, and
                        // install does no caller-IC work (adapters.rs's own
                        // doc records the eager repatch as deferred since S11
                        // step 10). Result: k dispatched its FULL body
                        // interpreted for the caller's whole lifetime —
                        // deltablue's dominant t=200 tail. Heal it HERE, the
                        // one place the staleness is observable: re-key the
                        // k pair in place to the freshly compiled target and
                        // rebuild the stub (immutable-PIC discipline D3.4,
                        // build-before-free, degrade-on-cache-full — all
                        // exactly `rt_resolve_send`'s own Pic re-key arm).
                        // `site_sel == sel` guards the task-#93 reused-region
                        // hazard, same as the Mono arm above.
                        if site_sel == sel {
                            let mut pairs = vm.pic_table.pairs_of(stub).to_vec();
                            if let Some(pair) = pairs.iter_mut().find(|p| p.0 == k) {
                                // The same call rt_resolve_send's PIC arms
                                // make: verified entry (the guard chain
                                // already proved the klass).
                                let new_t = resolve_target_entry(vm, k, sel, method, true);
                                if pair.1 != new_t {
                                    pair.1 = new_t;
                                    let resolve_addr = vm.stubs.resolve_addr();
                                    let smi_klass_bits = vm.universe.smi_klass.oop().raw();
                                    if let Some(new_stub) = vm.pic_table.build(
                                        &mut vm.code_cache,
                                        smi_klass_bits,
                                        resolve_addr,
                                        pairs,
                                    ) {
                                        vm.stats.c2i_pic_rekeys += 1;
                                        vm.pic_table.free(&mut vm.code_cache, stub);
                                        vm.code_cache.patch_call_site_at(
                                            caller_code,
                                            site_off,
                                            new_stub.base as u64,
                                        );
                                        vm.code_table.get_mut(caller_id).unwrap().ic_sites
                                            [site_idx]
                                            .state = IcState::Pic { stub: new_stub };
                                    } else {
                                        // Cache full: leave the old PIC
                                        // installed — correct, just slow;
                                        // the established degrade pattern.
                                        vm.stats.ic_patch_declined_cache_full += 1;
                                    }
                                }
                            }
                        }
                    } else if is_mono_k && site.super_klass.is_none() {
                        #[cfg(debug_assertions)]
                        if std::env::var("MACVM_DBG_RESOLVE").is_ok() {
                            eprintln!(
                                "C2I-REPATCH BLOCKED: site sel {:?} != callee sel {:?} \
                                 (caller {caller_id:?} off {site_off:#x} — stale ret_addr \
                                 into a reused code region)",
                                site_sel.as_string(),
                                sel.as_string(),
                            );
                        }
                    }
                }
            }
        }
    }

    result.raw()
}

/// D4.4: the shared megamorphic-lookup tail. Tail-jump-shaped, like
/// `stub_resolve` (NOT callee-shaped like `c2i_shared`/`rt_interpret_call`)
/// — `rt_mega_lookup` never patches anything, so there is no "this call's
/// own result" to distinguish from "what a FUTURE call through the same
/// site should reach"; it's always the same address, tail-jumped to
/// directly with the ORIGINAL send site's own x30 still intact.
fn build_mega_shared() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_MEGA);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(16)]); // selector_bits, carried through from mega_<sel>
    a.emit("mov", &[x(2), sp()]); // argv = &RootSpill (x0..x7, contiguous)
    let rt_mega_lookup_lit = a.literal_u64(
        rt_mega_lookup as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(rt_mega_lookup_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("br", &[x(16)]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr`/`bl` from `stub_mega_shared`'s own
/// hand-assembled listing above (through `call_far`), never called
/// directly from Rust — same contract as [`rt_resolve_send`]. `argv`
/// points at `stub_mega_shared`'s own RootSpill area (6 live `u64`s,
/// receiver + up to 5 args, contiguous ascending); valid for the duration
/// of this call only.
///
/// D4.4's own words: "probes lookup cache (SPEC §6.1), full lookup on
/// miss, NEVER patches the site" -- `runtime::lookup::lookup` already IS
/// that probe-then-full-walk (`LookupCache` is its own first stop), so
/// this is a thin wrapper around the SAME `resolve_target_entry` helper
/// `rt_resolve_send` itself uses, with `use_verified=true` (this
/// function's own fresh `klass_of`+`lookup` IS the verification a PIC's
/// guard chain would otherwise have done, `resolve_target_entry`'s own
/// doc) -- and, critically, no patch call anywhere: once a site is
/// `Mega`, it stays `Mega` forever (D4.1's own state table has no exit
/// from it).
pub unsafe extern "C" fn rt_mega_lookup(
    vm: *mut VmState,
    selector_bits: u64,
    argv: *mut u64,
) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_mega_shared`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_mega_lookup: anchor must be set by stub_mega_shared's prologue before this call"
    );

    let selector = SymbolOop::try_from(Oop::from_raw(selector_bits)).expect(
        "rt_mega_lookup: selector_bits must be a genuine SymbolOop -- MegaTable::get_or_make's \
         own RelocKind::Oop pool word",
    );
    // SAFETY: this function's own contract above -- argv[0] is the
    // receiver, D4.1's register protocol (shared with `rt_resolve_send`'s
    // own argv).
    let receiver = Oop::from_raw(unsafe { *argv });
    let k = klass_of(vm, receiver);

    let Some(method) = lookup(vm, k, selector) else {
        return vm.stubs.dnu_addr();
    };
    resolve_target_entry(vm, k, selector, method, true)
}

/// D5.4: the shared, callee-shaped DNU tail. Tail-jumped to (never
/// patched into any site — `rt_resolve_send`/`rt_mega_lookup`'s own doc)
/// from EITHER `stub_resolve`'s or `stub_mega_shared`'s own generic
/// epilogue, both of which restore x30 to the ORIGINAL send's own
/// continuation before getting here — same "capture ret_addr before
/// call_far clobbers it, callee-shaped return" pattern as `c2i_shared`.
fn build_stub_dnu() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_DNU);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("mov", &[x(1), x(30)]); // ret_addr, captured before call_far clobbers it
    a.emit("mov", &[x(2), sp()]); // argv = &RootSpill (x0..x7, contiguous)
    let rt_dnu_lit = a.literal_u64(rt_dnu as *const () as u64, Some(RelocKind::RuntimeAddr));
    a.call_far(rt_dnu_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `br` from `stub_resolve`'s or `stub_mega_shared`'s
/// own epilogue (both tail-jumping to whatever `rt_resolve_send`/
/// `rt_mega_lookup` returned on a lookup miss), never called directly from
/// Rust. `argv` points at the ORIGINAL send site's own RootSpill area (6
/// live `u64`s, receiver + up to 5 args, contiguous ascending) — neither
/// intermediate stub's own epilogue touches it beyond the standard
/// reload, so it's exactly what the original send set up.
///
/// D5.4: builds a `Message` (selector + args array) mirroring
/// `interpreter::send::dnu`'s own recipe (that function reads its args off
/// `vm.stack`; this one off the RootSpill argv, matching every other S11
/// runtime stub) — `find_caller_site` (shared with `rt_resolve_send`)
/// re-derives the selector/argc this SITE was built with from `ret_addr`,
/// exactly the same way regardless of which door (`rt_resolve_send` or
/// `rt_mega_lookup`) led here, since an `IcSite`'s own `selector`/`argc`
/// never change, only its `state` does. Then sends `#doesNotUnderstand:`
/// via the same lookup+`run_method_reentrant` pattern `rt_interpret_call`
/// already uses (this call may be nested inside a paused interpreter
/// activation just like any other C→I path, for the exact same reason).
/// Falls back to `runtime::error::dnu_fallback` (prints + exits, `-> !`)
/// only if EVEN `#doesNotUnderstand:` itself isn't found on the
/// receiver's klass — unreachable in a real Smalltalk image (`Object`
/// always implements it), matching how the interpreter's own `dnu()`
/// treats the identical case.
pub unsafe extern "C" fn rt_dnu(vm: *mut VmState, ret_addr: u64, argv: *mut u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_dnu`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_dnu: anchor must be set by stub_dnu's prologue before this call"
    );

    let (caller_id, _caller_code, _site_off, _site_idx, selector, argc_total) =
        find_caller_site(vm, ret_addr);
    let real_argc = argc_total as usize - 1;

    // SAFETY: this function's own contract above -- argv[0] is the
    // receiver, argv[1..=real_argc] the real Smalltalk args, D4.1's
    // register protocol.
    let receiver = Oop::from_raw(unsafe { *argv });
    let k = klass_of(vm, receiver);
    crate::runtime::error::trace_dnu(vm, &format!("compiled nm={}", caller_id.0), k, selector);

    // `selector`/`k`/`receiver` are bare parameters held across BOTH
    // allocations below (the interpreter's own `send::dnu` holds the
    // same values across the same two allocations, for the same reason —
    // S7-10 GC_STRESS audit).
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let selector_h = scope.handle(vm, selector);
    let k_h = scope.handle(vm, k);
    let receiver_h = scope.handle(vm, receiver);

    let array_klass = vm.universe.array_klass;
    let args_array = crate::memory::alloc::alloc_indexable_oops(vm, array_klass, real_argc);
    for i in 0..real_argc {
        // SAFETY: this function's own contract above.
        args_array.at_put(i, Oop::from_raw(unsafe { *argv.add(1 + i) }));
    }
    let array_h = scope.handle(vm, args_array.oop());

    let message_klass = vm.universe.message_klass;
    let msg = crate::memory::alloc::alloc_slots(vm, message_klass);
    msg.set_body_oop(0, selector_h.get(vm).oop());
    msg.set_body_oop(1, array_h.get(vm));

    let sel_dnu = vm.universe.sel_does_not_understand;
    // `TierLink::IntoInterpreter` around the nested `#doesNotUnderstand:` run —
    // the SAME bracket `rt_interpret_call`/`rt_value_fallback` push around
    // theirs, and for the same reason: this stub is always entered from
    // compiled code (anchor asserted above), so the nested run's entry frame
    // needs a C→I journal record for `walk_frames` to consume when a GC lands
    // inside the handler; without it the walk pairs the sentinel against
    // whatever OUTER crossing is on top and panics (the `found IntoCompiled
    // instead` class). `dnu_fallback` diverges (`-> !`), so the unbalanced
    // push on that arm never meets another walk.
    vm.tier_links.push(TierLink::IntoInterpreter {
        compiled_fp: vm.reg_block.last_compiled_fp,
        compiled_ret_pc: vm.reg_block.last_compiled_pc,
    });
    let result = match lookup(vm, k_h.get(vm), sel_dnu) {
        Some(dnu_method) => {
            let result = crate::interpreter::run_method_reentrant(
                vm,
                dnu_method,
                receiver_h.get(vm),
                &[msg.oop()],
            );
            result.raw()
        }
        None => crate::runtime::error::dnu_fallback(vm, selector_h.get(vm), k_h.get(vm)),
    };
    vm.tier_links.pop();
    result
}

/// D1/D5: the shared `BoolBr.not_bool` runtime helper — callee-shaped
/// (plain `ret`, x0=result), same pattern as `c2i_shared`/`stub_dnu`. Only
/// one argument (`val`, arriving in x0), so it's copied to x1 (`rt_must_be
/// _boolean`'s own 2nd parameter) BEFORE x0 gets overwritten with `vm` —
/// `emit_stub_prologue`'s own `stp x0,x1,...` already spilled val's
/// original value to RootSpill by this point, so the register itself is
/// free to reuse.
///
/// Built and published now (S11 step 6, completing the `RuntimeStubs`
/// table's own roster), but not yet reachable from any REAL compiled
/// method — `Ir::CallRuntime{stub: MustBeBoolean}`'s own emission (a
/// `not_bool` block calling this instead of the deleted `Ir::Bailout`,
/// D1's own words) is S11 step 7's job, tied to that step's broader
/// eligibility relaxation. Testable in isolation now regardless, via a
/// direct `Stubs::invoke` the same way any nmethod entry would be.
fn build_stub_must_be_boolean() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_MUST_BE_BOOLEAN);
    a.emit("mov", &[x(1), x(0)]); // val -> x1, before x0 is overwritten below
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_must_be_boolean as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_must_be_boolean`'s own
/// hand-assembled listing above (through `call_far`), never called
/// directly from Rust.
///
/// Mirrors the interpreter's own `must_be_boolean_send` exactly (same
/// klass_of + lookup + activate/fallback shape; that function additionally
/// rolls `vm.regs.bci` back to the branch opcode itself first, which has
/// no compiled-code equivalent — compiled code has no bci at all): sends
/// `#mustBeBoolean` (unary) via lookup + `run_method_reentrant`, expected
/// to return a real boolean the caller's own re-tested branch will
/// consume (once S11 step 7 wires the caller side). If NOT found, goes
/// STRAIGHT to `runtime::error::dnu_fallback` (prints + exits, `-> !`) —
/// deliberately not a full `#doesNotUnderstand:` send-and-fallback the way
/// [`rt_dnu`] is: the interpreter's own `must_be_boolean_send` makes the
/// identical choice, since this isn't a real user-level message send in
/// the first place, just a boolean-coercion protocol step.
pub unsafe extern "C" fn rt_must_be_boolean(vm: *mut VmState, val: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_must_be_boolean`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_must_be_boolean: anchor must be set by stub_must_be_boolean's prologue before this call"
    );

    let t = Oop::from_raw(val);
    let k = klass_of(vm, t);
    let sel = vm.universe.sel_must_be_boolean;
    // Same C→I bracket as `rt_dnu` above (this stub too is always entered
    // from compiled code), so a GC inside the nested `#mustBeBoolean` run
    // finds the crossing journaled. `dnu_fallback` diverges; the unbalanced
    // push on that arm never meets another walk.
    vm.tier_links.push(TierLink::IntoInterpreter {
        compiled_fp: vm.reg_block.last_compiled_fp,
        compiled_ret_pc: vm.reg_block.last_compiled_pc,
    });
    let result = match lookup(vm, k, sel) {
        Some(m) => crate::interpreter::run_method_reentrant(vm, m, t, &[]).raw(),
        None => crate::runtime::error::dnu_fallback(vm, sel, k),
    };
    vm.tier_links.pop();
    result
}

/// Float fast-path (`docs/float_fastpath_design.md`): `stub_box_double` —
/// `Ir::FBox`'s eden-overflow tail. The compiled fast path bump-allocates
/// and stores the payload inline; on overflow it `bl`s here with the RAW
/// f64 payload bits in x0 (NOT an oop — `KIND_BOX_DOUBLE` reports zero
/// RootSpill oop slots, so a GC never scans it). Same callee-shaped
/// skeleton as `stub_alloc_slow`. Marshals into `rt_box_double(vm, bits)`;
/// answers the fresh `Double`'s tagged oop in x0.
fn build_stub_box_double() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_BOX_DOUBLE);
    a.emit("mov", &[x(1), x(0)]); // payload bits -> x1, before x0 is overwritten
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_box_double as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_box_double`'s own hand-assembled
/// listing above (through `call_far`), never called directly from Rust.
///
/// `bits` is a raw `f64` bit pattern (never an oop). Allocates a fresh
/// boxed `Double` — this can scavenge, which is exactly why the compiled
/// fast path's overflow lands here rather than growing eden inline — and
/// answers its tagged oop. The payload is stored HERE (not by the compiled
/// continuation) so the object is fully formed before any further
/// safepoint can observe it.
pub unsafe extern "C" fn rt_box_double(vm: *mut VmState, bits: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_box_double`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_box_double: anchor must be set by stub_box_double's prologue before this call"
    );
    crate::memory::alloc::alloc_double(vm, f64::from_bits(bits))
        .oop()
        .raw()
}

/// SIMD NEON fast-path (`docs/SIMD.md`): `stub_box_float64x2` —
/// `Ir::VecArith`'s eden-overflow tail. The compiled fast path bump-
/// allocates the 32-byte box and stores the NEON result (`str q`) inline; on
/// overflow it `bl`s here with the two RAW lane bit patterns in x0 (lane 0)
/// and x1 (lane 1) — NEITHER is an oop (`KIND_BOX_FLOAT64X2` reports zero
/// RootSpill oop slots, so a GC never scans them). Same callee-shaped
/// skeleton as `stub_box_double`, but marshals two payload words:
/// `rt_box_float64x2(vm, lane0_bits, lane1_bits)`; answers the fresh
/// `Float64x2`'s tagged oop in x0.
fn build_stub_box_float64x2() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_BOX_FLOAT64X2);
    // Shift the two lane words up (read x1 before it's clobbered, then x0):
    // rt_box_float64x2(x0=vm, x1=lane0, x2=lane1).
    a.emit("mov", &[x(2), x(1)]); // lane1 -> x2
    a.emit("mov", &[x(1), x(0)]); // lane0 -> x1
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_box_float64x2 as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_box_float64x2`'s own hand-assembled
/// listing above (through `call_far`), never called directly from Rust.
///
/// `lane0`/`lane1` are raw `f64` bit patterns (never oops). Allocates a fresh
/// boxed `Float64x2` — this can scavenge, which is why the compiled fast
/// path's overflow lands here rather than growing eden inline — and answers
/// its tagged oop. Both lanes are stored HERE (not by the compiled
/// continuation) so the object is fully formed before any further safepoint
/// can observe it.
pub unsafe extern "C" fn rt_box_float64x2(vm: *mut VmState, lane0: u64, lane1: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by stub_box_float64x2.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_box_float64x2: anchor must be set by stub_box_float64x2's prologue before this call"
    );
    crate::memory::alloc::alloc_float64x2(vm, f64::from_bits(lane0), f64::from_bits(lane1)).raw()
}

/// SIMD: `stub_box_float32x4` — `Ir::VecArith`'s eden-overflow tail for a
/// `Float32x4`. Identical to `stub_box_float64x2` (the box layout is
/// klass-agnostic — a 16-byte raw body): x0/x1 hold the two 64-bit halves of
/// the `.4s` result register (each packs two f32 lanes), neither an oop.
/// Marshals `rt_box_float32x4(vm, half0, half1)`; answers the tagged oop.
fn build_stub_box_float32x4() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_BOX_FLOAT32X4);
    a.emit("mov", &[x(2), x(1)]); // half1 -> x2
    a.emit("mov", &[x(1), x(0)]); // half0 -> x1
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_box_float32x4 as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]);
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]);
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_box_float32x4`'s own hand-assembled
/// listing above, never called directly from Rust.
///
/// `half0`/`half1` are the two raw 64-bit halves of the `.4s` result (each
/// packs two f32 lanes in NEON element order); NEITHER is an oop. Unpacks them
/// into the four f32 lanes and allocates a fresh `Float32x4` — this can
/// scavenge, which is why the compiled fast path's overflow lands here. The
/// unpack→`alloc_float32x4` repack is bit-exact (it is the identity on the raw
/// words), so the boxed object is byte-for-byte the inline fast path's.
pub unsafe extern "C" fn rt_box_float32x4(vm: *mut VmState, half0: u64, half1: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by stub_box_float32x4.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_box_float32x4: anchor must be set by stub_box_float32x4's prologue before this call"
    );
    let lanes = [
        f32::from_bits(half0 as u32),
        f32::from_bits((half0 >> 32) as u32),
        f32::from_bits(half1 as u32),
        f32::from_bits((half1 >> 32) as u32),
    ];
    crate::memory::alloc::alloc_float32x4(vm, lanes).raw()
}

/// SIMD: `stub_box_int32x4` — `Ir::VecArith`'s eden-overflow tail for an
/// `Int32x4`. Identical shape to the float box stubs (the body is a 16-byte
/// raw lane blob): x0/x1 hold the two 64-bit halves of the `.4s` result (each
/// packs two i32 lanes), neither an oop. Marshals `rt_box_int32x4(vm, half0,
/// half1)`; answers the tagged oop.
fn build_stub_box_int32x4() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_BOX_INT32X4);
    a.emit("mov", &[x(2), x(1)]); // half1 -> x2
    a.emit("mov", &[x(1), x(0)]); // half0 -> x1
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_box_int32x4 as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]);
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]);
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_box_int32x4`'s own hand-assembled
/// listing above, never called directly from Rust.
///
/// `half0`/`half1` are the two raw 64-bit halves of the `.4s` result (each
/// packs two i32 lanes in NEON element order); NEITHER is an oop. Unpacks them
/// into the four i32 lanes and allocates a fresh `Int32x4` (the repack is the
/// identity on the raw words, so the boxed object is byte-for-byte the inline
/// fast path's).
pub unsafe extern "C" fn rt_box_int32x4(vm: *mut VmState, half0: u64, half1: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by stub_box_int32x4.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_box_int32x4: anchor must be set by stub_box_int32x4's prologue before this call"
    );
    let lanes = [
        half0 as u32 as i32,
        (half0 >> 32) as u32 as i32,
        half1 as u32 as i32,
        (half1 >> 32) as u32 as i32,
    ];
    crate::memory::alloc::alloc_int32x4(vm, lanes).raw()
}

/// D7: `stub_alloc_slow` — the inline-allocation fast path's overflow tail.
/// The compiled `Ir::Alloc` sequence puts the class oop in x0 and the object
/// size in bytes in x1, then `bl`s here. Same callee-shaped skeleton as
/// `stub_must_be_boolean` (anchor → frame → RootSpill → `call_far` →
/// restore → clear anchor → plain `ret`, result oop in x0). Marshals into
/// `rt_alloc_slow(vm, klass_bits, size_bytes)`: `vm` from x28 into x0, and
/// x0/x1 shifted up to x1/x2 (size first, so the klass in x0 isn't clobbered
/// before it's read).
fn build_stub_alloc_slow() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_ALLOC_SLOW);
    a.emit("mov", &[x(2), x(1)]); // size_bytes -> x2 (read x1 before it's overwritten)
    a.emit("mov", &[x(1), x(0)]); // klass_bits -> x1 (read x0 before it's overwritten)
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_alloc_slow as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// A shimmable primitive-bearing method's own compiled entry (`compiler::
/// driver`'s eligibility, `compiler::emit`'s shim prologue) puts
/// `prim_id` in x10 and `receiver + real args` count in x11 — deliberately
/// NOT x0/x1 (unlike `stub_alloc_slow`'s klass_bits/size_bytes): x0..x5
/// here are the METHOD's own receiver+args, and `emit_stub_prologue`'s
/// RootSpill save below is exactly what makes them `rt_call_primitive`'s
/// own `&[Oop]` args slice — colliding them with this stub's own scalar
/// parameters would corrupt the very data being marshaled. x10/x11 survive
/// both `emit_stub_prologue` (only saves x0..x7) and `emit_stub_kind_tag`
/// (scratches x9, not x10/x11).
///
/// Unlike `stub_alloc_slow`, this does NOT override x0 with the raw result
/// after the epilogue — x0..x7 must come back to the CALLER exactly as it
/// had them: on `PrimResult::Fail`, the calling method's own shim prologue
/// falls through to its normal compiled bytecode body using x0 as `self`,
/// same as any ordinary compiled entry (SPEC's own pinned calling
/// convention). The tagged result travels in x16 instead, which
/// `emit_stub_epilogue`'s x0..x7 reload never touches.
fn build_stub_call_primitive() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_CALL_PRIMITIVE);
    a.emit("mov", &[x(2), x(11)]); // argc_plus_recv -> x2
    a.emit("mov", &[x(1), x(10)]); // prim_id -> x1
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_call_primitive as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    a.emit("mov", &[x(16), x(0)]); // tagged result -> x16, survives the epilogue's x0..x7 reload
    emit_stub_epilogue(&mut a);
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_call_primitive`'s own
/// hand-assembled listing above, never called directly from Rust — `vm`
/// must be `x28` (D4's own invariant, established by `call_stub`).
///
/// Reads the calling method's own receiver+args back out of RootSpill
/// (`[fp − ROOTSPILL_BYTES, fp)`, `emit_stub_prologue`'s own layout — same
/// formula `memory::roots`'s own RootSpill walk uses) rather than being
/// handed a pointer, since that memory is exactly where `emit_stub_
/// prologue` just archived x0..x7 — no separate marshaling needed. Calls
/// `prim_id`'s own `PrimFn` exactly as `interpreter::send::try_primitive`
/// does, and returns a tagged `u64`: the real oop's raw bits on `Ok`, or
/// [`crate::oops::layout::PRIM_FAIL_SENTINEL`] on `Fail`. `Activated`
/// (the `value`/`ensure:`/`ifCurtailed:` family) can never reach here —
/// `compiler::driver`'s shimmable-primitive eligibility excludes every
/// primitive capable of it, enforced here by the `debug_assert` below, not
/// silently assumed.
pub unsafe extern "C" fn rt_call_primitive(
    vm: *mut VmState,
    prim_id: u64,
    argc_plus_recv: u64,
) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_call_primitive`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_call_primitive: anchor must be set by stub_call_primitive's prologue before this call"
    );
    let desc = crate::runtime::primitives::prim_by_id(prim_id as u16).unwrap_or_else(|| {
        panic!("rt_call_primitive: unknown primitive id {prim_id} -- driver.rs baked a bad literal")
    });
    let fp = vm.reg_block.last_compiled_fp;
    let rootspill_base = fp - crate::oops::layout::ROOTSPILL_BYTES as u64;
    let n = argc_plus_recv as usize;
    debug_assert!(
        n <= crate::oops::layout::ROOTSPILL_SLOTS,
        "rt_call_primitive: {n} args exceeds ROOTSPILL_SLOTS -- driver.rs's own argc<=5 \
         eligibility cap (receiver+5=6) should make this impossible"
    );
    let mut buf = [crate::oops::Oop::from_raw(0); crate::oops::layout::ROOTSPILL_SLOTS];
    for (i, slot) in buf.iter_mut().enumerate().take(n) {
        // SAFETY: `rootspill_base + 8*i` is exactly the address
        // `emit_stub_prologue`'s own `stp x_i, x_{i+1}, [sp, #8*i]` wrote,
        // for `i` in `0..n <= ROOTSPILL_SLOTS` (asserted above).
        let addr = (rootspill_base + 8 * i as u64) as *const u64;
        *slot = crate::oops::Oop::from_raw(unsafe { *addr });
    }
    match (desc.f)(vm, &buf[..n]) {
        crate::runtime::primitives::PrimResult::Ok(v) => v.raw(),
        crate::runtime::primitives::PrimResult::Fail => {
            debug_assert!(
                desc.can_fail,
                "rt_call_primitive: primitive {} Failed but its own PrimDesc.can_fail is false",
                desc.name
            );
            crate::oops::layout::PRIM_FAIL_SENTINEL
        }
        crate::runtime::primitives::PrimResult::Activated => unreachable!(
            "rt_call_primitive: primitive {} returned Activated -- compiler::driver's \
             shimmable-primitive eligibility must exclude every Activated-capable primitive \
             (the value family, ensure:, ifCurtailed:) -- this is a compiler bug, not a guest one",
            desc.name
        ),
        crate::runtime::primitives::PrimResult::Nlr(_) => unreachable!(
            "rt_call_primitive: primitive {} returned Nlr -- only activate_block's \
             enter-compiled hook produces it, and prims 50-54/60-61 are \
             PRIM_ACTIVATES_FRAME-excluded from shims (review §7's ripple note)",
            desc.name
        ),
    }
}

/// S24 A1: a compiled block body's `nlr_tos` origination
/// (`closure_compilation_design.md` §2.4). The emitted lowering `bl`s here
/// with x0 = the closure, x1 = the NLR value; this stub only parks the
/// state — the emitted continuation materializes `NLR_SENTINEL` in x0 and
/// branches to the block's own epilogue, from where every compiled frame
/// above relays via its existing per-site `emit_nlr_check` (S11 step 9's
/// machinery, unchanged). `emit_stub_prologue`'s RootSpill save is what
/// makes x0/x1 (closure, value) GC-visible while `rt_nlr_originate` runs —
/// `memory::roots::real_oop_rootspill_slots`'s `NlrOriginate` arm reads
/// exactly 2 live slots (fixed, like `MustBeBoolean`/`AllocSlow`), though
/// the runtime fn itself never allocates.
fn build_stub_nlr_originate() -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_NLR_ORIGINATE);
    a.emit("mov", &[x(2), x(1)]); // value -> x2
    a.emit("mov", &[x(1), x(0)]); // closure -> x1
    a.emit("mov", &[x(0), x(28)]); // vm
    let lit = a.literal_u64(
        rt_nlr_originate as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(lit);
    emit_stub_epilogue(&mut a);
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `stub_nlr_originate`'s own
/// hand-assembled listing above, never called directly from Rust — `vm`
/// must be `x28` (D4's own invariant, established by `call_stub`).
///
/// Parks the in-flight non-local return: `home` unpacked from the closure's
/// own `home` field, `value` as given, and — the adversarial-review
/// BLOCKER-§2 fix — the ORIGINATING closure itself, so
/// `continue_unwind`'s dead-home arm can deliver `#cannotReturn:` to the
/// real closure (the block's native frame is gone by the time the liveness
/// check runs; the legacy read-the-current-frame's-receiver-slot strategy
/// would find the value:-sender's receiver instead). Liveness is
/// deliberately NOT checked here — parking without checking preserves the
/// interpreter's exact evaluation point (the check happens once, in
/// `continue_unwind`, same as interpreted `nlr_tos`). No allocation, no
/// lookup, no GC window.
pub unsafe extern "C" fn rt_nlr_originate(vm: *mut VmState, closure_bits: u64, value_bits: u64) {
    // SAFETY: this function's own contract, guaranteed by `stub_nlr_originate`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_nlr_originate: anchor must be set by stub_nlr_originate's prologue before this call"
    );
    debug_assert!(
        vm.nlr_state.is_none(),
        "rt_nlr_originate: an NLR is originating while one is already parked \
         (enter_compiled's own invariant, compiled_call.rs)"
    );
    let closure_oop = crate::oops::Oop::from_raw(closure_bits);
    let closure = crate::oops::wrappers::ClosureOop::try_from(closure_oop).expect(
        "rt_nlr_originate: x0 must be the block's own closure (the emitted lowering \
         passes the pinned closure vreg) -- compiler bug, not a guest one",
    );
    let home = crate::oops::home_ref::unpack_home_ref(closure.home());
    vm.nlr_state = Some(crate::runtime::vm_state::NlrState {
        home,
        value: crate::oops::Oop::from_raw(value_bits),
        closure: Some(closure_oop),
    });
}

/// S24 A2: a `value`-family send site's block-dispatch stub, one per block
/// arity `argc` (0..=3) — `closure_compilation_design.md` §2.3. On entry x0
/// is the receiver closure and x1..x_argc its args (the block calling
/// convention a normal send marshals). The stub:
///
/// 1. `emit_stub_prologue` spills x0..x7 to RootSpill and sets the anchor
///    (`KIND_VALUE_DISPATCH`) — so the fallback's nested interpretation is
///    GC-walkable, closure+args covered (`real_oop_rootspill_slots`'s
///    `ValueDispatch` arm reads the site's own `IcSite::argc`).
/// 2. Probes [`rt_value_target`] `(vm, closure, argc)` → the block
///    nmethod's entry, or 0.
/// 3. Nonzero: restore the registers (`emit_stub_epilogue`) and TAIL-jump
///    to the block entry — the block returns DIRECTLY to the original send
///    site (its own `emit_nlr_check` relays an `NLR_SENTINEL` with zero new
///    plumbing, exactly as for any compiled callee).
/// 4. Zero: [`rt_value_fallback`] interprets `value:` (byte-identical to a
///    cold `value:` send — including the block-not-compiled, non-closure,
///    and wrong-argc cases), returning the result the way `c2i_shared`
///    does (parked in x16 across the epilogue's own x0 reload).
///
/// The argc is baked (per-stub), so neither runtime fn needs a call-site
/// lookup — the fast path is stub-prologue + one probe + tail-jump.
fn build_stub_value_dispatch(argc: u64) -> CodeBlob {
    let mut a = JasmAssembler::new();

    emit_stub_prologue(&mut a);
    emit_stub_kind_tag(&mut a, KIND_VALUE_DISPATCH);
    // rt_value_target(vm, closure_bits, argc) -> entry addr or 0.
    a.emit("mov", &[x(1), x(0)]); // closure_bits -> x1 (x0 was the receiver closure)
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("movz", &[x(2), imm(argc as i64)]); // argc (baked)
    let target_lit = a.literal_u64(
        rt_value_target as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(target_lit);

    let fallback = a.new_label();
    a.cbz(xr(0), fallback);
    // Fast path: entry in x0. Park it in x16 (spared by the epilogue's own
    // x0..x7 reload), restore the closure+args, tail-jump.
    a.emit("mov", &[x(16), x(0)]);
    emit_stub_epilogue(&mut a);
    a.emit("br", &[x(16)]);

    // Fallback: interpret `value:` — NO baked method oop (a shared per-argc
    // stub cannot carry one without a #93-family staleness bug); the helper
    // re-resolves `(klass_of(closure), value-selector)` itself.
    a.bind(fallback);
    a.emit("mov", &[x(0), x(28)]); // vm
    a.emit("sub", &[x(1), x(29), imm(ROOTSPILL_BYTES as i64)]); // argv = &RootSpill
    a.emit("movz", &[x(2), imm(argc as i64)]); // argc (baked)
    let fallback_lit = a.literal_u64(
        rt_value_fallback as *const () as u64,
        Some(RelocKind::RuntimeAddr),
    );
    a.call_far(fallback_lit);
    a.emit("mov", &[x(16), x(0)]); // result -> x16, survives the epilogue's own x0 reload
    emit_stub_epilogue(&mut a);
    a.emit("mov", &[x(0), x(16)]); // restore the real result
    a.emit("ret", &[]);

    a.finish()
}

/// # Safety
/// Only ever reached via `blr` from `build_stub_value_dispatch`'s own
/// hand-assembled listing, never called directly from Rust — `vm` must be
/// `x28` (D4's invariant, established by `call_stub`). Does NOT allocate and
/// does NOT run any guest code, so it opens no GC window of its own; the
/// anchor its caller set is belt-and-braces (the fallback, not this probe,
/// is what can collect).
///
/// The Alive-only `by_block` probe (design §2.3, review §4 BLOCKER): returns
/// the block nmethod's `entry` ONLY for a genuine closure whose arity
/// matches the site and whose body is compiled AND Alive. Every other
/// case — non-closure receiver, wrong argc, uncompiled/NotEntrant/zombie
/// block — returns 0, routing to [`rt_value_fallback`], which reproduces the
/// interpreter's exact `value:` semantics (including the resulting error for
/// the non-closure/wrong-argc cases).
pub unsafe extern "C" fn rt_value_target(vm: *mut VmState, closure_bits: u64, argc: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by the stub.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_value_target: anchor must be set by the value-dispatch stub's prologue"
    );
    let Some(closure) = crate::oops::wrappers::ClosureOop::try_from(Oop::from_raw(closure_bits))
    else {
        return 0; // non-closure -> fallback interprets value: (byte-identical error)
    };
    let blk = closure.method();
    if blk.argc() as u64 != argc {
        return 0; // wrongArgc -> fallback interprets (byte-identical error)
    }
    let Some(id) = vm.code_table.lookup_block(blk) else {
        return 0; // not compiled, or NotEntrant/zombie (lookup_block is Alive-only)
    };
    let nm = vm
        .code_table
        .get(id)
        .expect("rt_value_target: lookup_block just returned this id");
    // A block nmethod has no entry guard (verified_entry == entry, A1), so
    // either offset is safe to `br` to with x0 = closure.
    let entry = nm.code.base as u64 + nm.entry_off as u64;
    vm.stats.value_dispatch_hits += 1;
    entry
}

/// # Safety
/// Only ever reached via `blr` from `build_stub_value_dispatch`'s fallback
/// branch, never called directly from Rust — `vm` must be `x28`. `argv`
/// points at the stub's own RootSpill (`1 + argc` live `u64`s: the closure
/// at `[0]`, its args at `[1..=argc]`), valid for this call only.
///
/// The block-not-compiled (and non-closure / wrong-argc) fallback: interpret
/// `value:` EXACTLY as a cold interpreted send would, by re-resolving the
/// `value`-family selector for `argc` on the receiver's ACTUAL klass and
/// running it reentrantly. This funnels a genuine closure straight back into
/// A1's `activate_block` (which bumps the block's counter, so warmup happens
/// naturally through the fallback, and enters the compiled body once it
/// exists); a non-closure or a receiver with no such method DNUs, matching
/// the interpreter byte-for-byte. Modeled on [`rt_interpret_call`]/[`rt_dnu`]:
/// the `TierLink::IntoInterpreter` anchor is what an escaping non-local
/// return (the block does `^`) uses to find the compiled boundary and relay
/// its `NLR_SENTINEL` back through the original send site.
pub unsafe extern "C" fn rt_value_fallback(vm: *mut VmState, argv: *const u64, argc: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by the stub.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_value_fallback: anchor must be set by the value-dispatch stub's prologue"
    );
    vm.stats.value_dispatch_fallbacks += 1;

    let n = argc as usize;
    // SAFETY: `argv[0]` is the receiver closure, `argv[1..=n]` its args —
    // the stub's RootSpill layout (`emit_stub_prologue`'s x0..x7 spill),
    // read exactly like `rt_dnu`/`rt_call_primitive` read theirs.
    let receiver = Oop::from_raw(unsafe { *argv });
    let args: Vec<Oop> = (0..n)
        .map(|i| Oop::from_raw(unsafe { *argv.add(1 + i) }))
        .collect();

    // The `value`-family selector is fixed by the baked argc.
    let sel_name: &[u8] = match argc {
        0 => b"value",
        1 => b"value:",
        2 => b"value:value:",
        3 => b"value:value:value:",
        _ => unreachable!("value-dispatch stubs exist only for argc 0..=3, got {argc}"),
    };
    let sel = vm.universe.intern(sel_name);
    let k = klass_of(vm, receiver);

    // `TierLink::IntoInterpreter` around the nested run: the same anchor
    // `rt_interpret_call` pushes so an escaping NLR finds this C→I boundary
    // (`compiled_ret_pc` is the original `value:` site's return address, set
    // as the stub's own `last_compiled_pc`). The reentrant run returns the
    // ordinary result, or `NLR_SENTINEL` if the block's `^` escaped past
    // here — which the original send site's `emit_nlr_check` then relays.
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let receiver_h = scope.handle(vm, receiver);
    let sel_h = scope.handle(vm, sel.oop());
    let k_h = scope.handle(vm, k);
    vm.tier_links.push(TierLink::IntoInterpreter {
        compiled_fp: vm.reg_block.last_compiled_fp,
        compiled_ret_pc: vm.reg_block.last_compiled_pc,
    });
    let result = match lookup(vm, k, sel) {
        Some(method) => crate::interpreter::run_method_reentrant(vm, method, receiver, &args),
        None => {
            // DNU — mirror `rt_dnu`'s recipe (Message on a handle-parked
            // array, `#doesNotUnderstand:` via a full lookup). Reachable
            // only for a non-closure receiver whose klass has no `value:`
            // (a mono value-site whose block var was reassigned) — the
            // interpreter DNUs identically.
            let array_klass = vm.universe.array_klass;
            let args_array = crate::memory::alloc::alloc_indexable_oops(vm, array_klass, n);
            for (i, a) in args.iter().enumerate() {
                args_array.at_put(i, *a);
            }
            let array_h = scope.handle(vm, args_array.oop());
            let message_klass = vm.universe.message_klass;
            let msg = crate::memory::alloc::alloc_slots(vm, message_klass);
            msg.set_body_oop(0, sel_h.get(vm));
            msg.set_body_oop(1, array_h.get(vm));
            let sel_dnu = vm.universe.sel_does_not_understand;
            match lookup(vm, k_h.get(vm), sel_dnu) {
                Some(dnu_method) => crate::interpreter::run_method_reentrant(
                    vm,
                    dnu_method,
                    receiver_h.get(vm),
                    &[msg.oop()],
                ),
                None => {
                    let sel_sym = crate::oops::wrappers::SymbolOop::try_from(sel_h.get(vm))
                        .expect("rt_value_fallback: the value selector is always a Symbol");
                    crate::runtime::error::dnu_fallback(vm, sel_sym, k_h.get(vm))
                }
            }
        }
    };
    vm.tier_links.pop();
    result.raw()
}

/// # Safety
/// Only ever reached via `blr` from `stub_alloc_slow`'s own hand-assembled
/// listing above (through `call_far`), never called directly from Rust —
/// `vm` must be `x28` (D4's own invariant, established by `call_stub`).
///
/// D7's slow edge: an inline eden bump overflowed, so allocate the object
/// the ordinary way. Because this is reached FROM compiled code,
/// `compiled_depth > 0`, so `alloc_slots` → `alloc_words` takes the D8
/// bridge's old-direct path (no moving GC — the outer compiled frame's
/// spill-slot oops are still invisible to the collector until S12). Returns
/// the freshly allocated object's tagged oop bits in x0. `klass_bits` came
/// from the `Alloc` site's own `RelocKind::Oop` pool word (a genuine
/// `KlassOop`, GC-tracked); `size_bytes` is redundant with the klass's own
/// `non_indexable_size` and cross-checked in debug.
pub unsafe extern "C" fn rt_alloc_slow(vm: *mut VmState, klass_bits: u64, size_bytes: u64) -> u64 {
    // SAFETY: this function's own contract, guaranteed by `stub_alloc_slow`.
    let vm = unsafe { &mut *vm };
    debug_assert_ne!(
        vm.reg_block.last_compiled_fp, 0,
        "rt_alloc_slow: anchor must be set by stub_alloc_slow's prologue before this call"
    );
    debug_assert!(
        vm.compiled_depth > 0,
        "rt_alloc_slow: only ever reached from compiled code (compiled_depth must be > 0)"
    );
    let klass = KlassOop::try_from(Oop::from_raw(klass_bits)).expect(
        "rt_alloc_slow: klass_bits must be a genuine KlassOop -- the Alloc site's own RelocKind::Oop pool word",
    );
    // S24 A3 widening of D7's contract: `Ir::Alloc` is no longer only inlined
    // `basicNew` of Format::Slots — `Ir::AllocClosure` (A3a) and the
    // materialize-Context prologue (A3b) emit it for Format::Closure and
    // Format::Context, whose site-baked `size_bytes` covers the VARIABLE tail
    // (`nis + 1 + n` words), NOT `nis`. The original slow path ignored
    // `size_bytes` and always ran `alloc_slots` (nis words) — correct for
    // Slots, but for a Closure/Context it returned an object `n+1` words too
    // SHORT, and the compiled continuation's size/copied/slot stores then
    // corrupted the neighboring heap object. Debug builds aborted on the old
    // Format::Slots assert below; release builds corrupted silently — found
    // as deltablue's doesNotUnderstand #value: at threshold=200 (the closure
    // volume A3b added made the slow path reachable under real allocation
    // pressure; every small repro stayed on the inline fast path). Dispatch
    // by format to the interpreter's OWN allocators, which produce exactly
    // the fast path's layout (nil-filled body; their size-slot store is
    // idempotent with the compiled continuation's own).
    let words = size_bytes as usize / crate::oops::layout::WORD_SIZE;
    let nis = klass.non_indexable_size();
    match klass.format() {
        crate::oops::klass::Format::Slots => {
            debug_assert_eq!(
                nis * crate::oops::layout::WORD_SIZE,
                size_bytes as usize,
                "rt_alloc_slow: a Slots Alloc site's baked size must match the klass's own \
                 non_indexable_size"
            );
        }
        crate::oops::klass::Format::Closure | crate::oops::klass::Format::Context => {
            debug_assert!(
                words > nis,
                "rt_alloc_slow: a Closure/Context Alloc site's baked size must cover the \
                 size slot and tail (words {words} <= nis {nis})"
            );
        }
        other => unreachable!(
            "rt_alloc_slow: Ir::Alloc is only ever emitted for Slots (inlined basicNew), \
             Closure (AllocClosure), or Context (materialize prologue) — got {other:?}"
        ),
    }
    // S12 D3 test hook (`VmState::test_walk_capture`'s own doc): this is
    // one of the six anchor-setting stubs, so it's a real place a test can
    // observe `runtime::frames::walk_frames` against a genuine in-flight
    // native chain rather than hand-faked memory. `catch_unwind` here (not
    // at the test level) is load-bearing, not defensive style: a panic
    // must never be allowed to unwind across this function's own
    // `extern "C"` boundary into the hand-assembled native frames below it
    // (UB) -- the walker's own panics (a real possibility this hook
    // exists specifically to exercise, `walker_terminates_on_torn_
    // tierlinks`) are caught and reported back through the Result instead.
    if vm.test_walk_capture.is_some() {
        let torn = if vm.test_tear_tier_links_before_walk {
            vm.test_tear_tier_links_before_walk = false;
            vm.tier_links.pop()
        } else {
            None
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut seen = Vec::new();
            crate::runtime::frames::walk_frames(vm, |fv| seen.push(fv));
            seen
        }))
        .map_err(|e| {
            e.downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "walk_frames panicked with a non-string payload".to_string())
        });
        // Restore what the tear took: the test's claim is only that the
        // WALKER fails loudly against torn links — the tear must not
        // outlive the observed walk, because (S12 step 7, no more D8
        // bridge) the ordinary `alloc_slots` below can genuinely
        // scavenge, and THAT walk needs the links intact (a panic there
        // would unwind across this fn's own `extern "C"` boundary — UB —
        // exactly what this hook's own `catch_unwind` exists to prevent
        // for the walk it actually wants to observe).
        vm.tier_links.extend(torn);
        vm.test_walk_capture = Some(result);
    }
    // S12 step 4 test hook (`VmState::test_scavenge_probe`'s own doc):
    // same rationale as `test_walk_capture` just above — this is the one
    // place a test can force a REAL `scavenge` to run against a REAL live
    // compiled frame's own spill slot, which is the only honest way to
    // prove `memory::roots::each_code_root`'s new frame-scanning code
    // actually relocates it correctly. `catch_unwind` here for the same
    // not-safe-to-unwind-across-`extern "C"` reason as `test_walk_capture`.
    if vm.test_force_scavenge_in_alloc_slow {
        vm.test_force_scavenge_in_alloc_slow = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut frames = Vec::new();
            crate::runtime::frames::walk_frames(vm, |fv| frames.push(fv));
            let (fp, ret_pc, nm) = frames
                .iter()
                .find_map(|fv| match fv {
                    crate::runtime::frames::FrameView::Compiled { fp, ret_pc, nm } => {
                        Some((*fp, *ret_pc, *nm))
                    }
                    _ => None,
                })
                .expect(
                    "test_force_scavenge_in_alloc_slow: caller must have a live compiled frame \
                     beneath rt_alloc_slow",
                );
            let slot = {
                let nmethod = vm.code_table.get(nm).expect("just walked through it");
                let mut slots = nmethod.oopmap_at(ret_pc).iter_slots();
                let slot = slots.next().expect(
                    "test_force_scavenge_in_alloc_slow: the caller's compiled frame must have \
                     exactly one live oop slot (its own receiver, kept alive across the Alloc \
                     safepoint) -- zero found",
                );
                assert!(
                    slots.next().is_none(),
                    "test_force_scavenge_in_alloc_slow: expected exactly one live oop slot, \
                     found more than one"
                );
                slot
            };
            let addr = (fp - 8 * (slot as u64 + 1)) as *mut u64;
            // SAFETY: `fp`/`slot` are exactly `memory::roots::
            // each_code_root`'s own compiled-frame formula, just derived
            // independently here so the test can read the slot BEFORE and
            // AFTER without going through that function itself.
            let before = unsafe { *addr };
            crate::memory::scavenge::scavenge(vm)
                .expect("test_force_scavenge_in_alloc_slow: scavenge must not stall");
            let after = unsafe { *addr };
            (before, after)
        }))
        .map_err(|e| {
            e.downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| {
                    "test_force_scavenge_in_alloc_slow panicked with a non-string payload"
                        .to_string()
                })
        });
        vm.test_scavenge_probe = Some(result);
    }
    match klass.format() {
        crate::oops::klass::Format::Closure => {
            debug_assert_eq!(
                klass.oop().raw(),
                vm.universe.closure_klass.oop().raw(),
                "rt_alloc_slow: a Format::Closure Alloc site's klass must be THE closure klass"
            );
            crate::memory::alloc::alloc_closure(vm, words - nis - 1)
                .oop()
                .raw()
        }
        crate::oops::klass::Format::Context => {
            debug_assert_eq!(
                klass.oop().raw(),
                vm.universe.context_klass.oop().raw(),
                "rt_alloc_slow: a Format::Context Alloc site's klass must be THE context klass"
            );
            crate::memory::alloc::alloc_context(vm, words - nis - 1)
                .oop()
                .raw()
        }
        _ => crate::memory::alloc::alloc_slots(vm, klass).oop().raw(),
    }
}

/// S13 step 10b: `rt_poll`'s return, split across two registers by the
/// AArch64 C ABI so `stub_poll` can branch on it. A struct of two ≤8-byte
/// all-integer fields is ≤16 bytes and "fully integer", so the AAPCS64
/// "return in registers" rule (§B.2/§5.5) returns it in `x0:x1` with field 0
/// (`result`) in `x0` and field 1 (`deopted`) in `x1` — NOT via a hidden
/// x8 sret pointer (that path is only for aggregates > 16 bytes or containing
/// floats). `stub_poll`'s teardown depends on exactly this: after `bl
/// rt_poll`, `x0 = result` and `x1 = deopted`. The `#[repr(C)]` pins field
/// order = declaration order so `offset_of!(result) == 0` (see the ABI unit
/// test). `deopted` is 0 (continue the loop) or 1 (this frame was deopted;
/// `result` is the deoptee's own result, to hand to the loop's native caller).
#[repr(C)]
pub struct PollOutcome {
    pub result: u64,
    pub deopted: u64,
}

/// S13 step 10b: the loop back-edge poll's runtime side. Reached via `bl
/// rt_poll` from `stub_poll` whenever a compiled loop crosses a back-edge with
/// `reg_block.poll_flag` armed (which `flush::make_not_entrant` sets, §2d).
///
/// Two jobs:
///  1. the pre-existing one-shot `trace_on_poll` hook (fires regardless of any
///     deopt — `mixed_trace_golden`'s gate);
///  2. the loop-poll DEOPT: if `pending_deopt_flag` is set AND the polling
///     frame's OWN nmethod (the one `ret_pc` lands in) is `NotEntrant`, deopt
///     this frame exactly like `rt_uncommon_trap` does — materialize the
///     interpreter frame(s) at the LoopPoll scope and run them to completion,
///     returning the deoptee's result with `deopted = 1` so `stub_poll` tears
///     down to the loop's native caller. A poll in any OTHER (still-`Alive`)
///     frame, or with the flag unset, returns `{0, 0}` (continue the loop).
///
/// After a deopt, if no `NotEntrant` compiled frame remains on the stack, both
/// `pending_deopt_flag` and `poll_flag` are cleared (bounding polling to the
/// drain window). An escaping `NLR_SENTINEL` is propagated verbatim (never
/// treated as a normal result), exactly as `rt_deopt_on_return` does.
///
/// # Safety
/// Only ever reached via `bl rt_poll` from `stub_poll`'s own hand-assembled
/// listing above, never called directly from Rust (tests aside, which honor
/// the same contract) — `vm` must be `x28` (`&mut VmState`), `loop_fp` the
/// polling compiled frame's own FP (the saved x29 `stub_poll` reloads), and
/// `ret_pc` the return address of the `bl stub_poll` (inside the polling
/// nmethod, the LoopPoll scope's `PcDesc.code_off` key).
pub unsafe extern "C" fn rt_poll(vm: *mut VmState, loop_fp: u64, ret_pc: u64) -> PollOutcome {
    // SAFETY: this function's own contract, guaranteed by every caller
    // (`stub_poll`'s assembly, the only one there is).
    let vm = unsafe { &mut *vm };

    // (1) The one-shot trace hook — independent of any deopt.
    if vm.trace_on_poll {
        vm.trace_on_poll = false;
        crate::runtime::error::print_stack_trace(vm);
    }

    // (2) Fast continue: nothing is NotEntrant, so no loop needs deopting.
    if !vm.pending_deopt_flag {
        return PollOutcome {
            result: 0,
            deopted: 0,
        };
    }

    // Which nmethod owns this poll? A poll pc is always inside a published
    // nmethod (the poll instruction lives in compiled code); a miss should not
    // happen, so continue defensively (debug-assert to surface a VM bug).
    let nm_id = match vm.code_table.find_by_pc(ret_pc) {
        Some(id) => id,
        None => {
            debug_assert!(
                false,
                "rt_poll: ret_pc {ret_pc:#x} is not inside any published nmethod"
            );
            return PollOutcome {
                result: 0,
                deopted: 0,
            };
        }
    };

    // Is THIS frame's own nmethod the NotEntrant one? If not, the flag is for
    // some OTHER frame (or is stale) — this frame is fine, continue.
    let is_not_entrant = matches!(
        vm.code_table.get(nm_id).map(|nm| nm.state),
        Some(crate::codecache::nmethod::NmState::NotEntrant)
    );
    if !is_not_entrant {
        return PollOutcome {
            result: 0,
            deopted: 0,
        };
    }

    // ANCHOR (task #94): `stub_poll` deliberately skips `emit_stub_prologue`
    // (S10 D5.6 — polling itself never allocates), so unlike the six real
    // anchor-setting stubs, NOTHING publishes `last_compiled_fp/pc/kind` on
    // entry here. But `deoptimize_frame` below CAN allocate (materialize_
    // context/materialize_closure) and `interpret_active` runs arbitrary
    // interpreted code that certainly can — and `walk_frames`' start rule
    // reads exactly this anchor whenever the innermost tier crossing is
    // `IntoCompiled` (which it is: `enter_compiled`'s own crossing, still on
    // `vm.tier_links`, is untouched until `deopt_bridge_link` pushes below).
    // Without it, a GC mid-materialization has no way to find — and
    // therefore correctly relocate the oops in — the still-live POLLING
    // compiled frame at `(loop_fp, ret_pc)`: exactly the same hole
    // `build_uncommon_trampoline`/`build_deopt_return_trampoline` had before
    // their own anchor fix, just reached from the loop-poll deopt door
    // instead of the trap/return doors. Same remedy, set from Rust instead
    // of assembly since this stub's hot (non-deopting) path must stay
    // anchor-free — only the rare deopt-taking path pays this cost.
    vm.reg_block.last_compiled_fp = loop_fp;
    vm.reg_block.last_compiled_pc = ret_pc;
    vm.reg_block.last_compiled_kind = KIND_DEOPT_BRIDGE;

    // Deopt this frame exactly like `rt_uncommon_trap`: a reexecute (LoopPoll)
    // site — `incoming_result: None`; the recorded loop-carried stack is read
    // from the frame, and the interpreter resumes at the loop-header bci and
    // re-executes the loop condition.
    let resume = crate::runtime::deopt::deoptimize_frame(
        vm,
        crate::runtime::deopt::FrameView {
            fp: loop_fp as usize,
            pc: ret_pc as usize,
            nm: nm_id,
            incoming_result: None,
        },
    );
    vm.note_deopt(crate::runtime::vm_state::DeoptReason::Poll); // D7 stats + trace
                                                                // S14 step 9: same walk bridge as `rt_uncommon_trap` (see
                                                                // `frames::deopt_bridge_link`).
    let bridge = crate::runtime::frames::deopt_bridge_link(vm, loop_fp);
    vm.tier_links.push(bridge);
    let result = crate::interpreter::interpret_active(vm, resume).raw();
    vm.tier_links.pop();

    // CLEAR the anchor (P9) now that both the materializer and the nested
    // interpreter run are done — before EITHER return path below, so
    // `rt_poll` never hands control back to `stub_poll` with a stale
    // nonzero anchor (which `walk_frames` would otherwise trust for an
    // unrelated LATER poll or trap).
    vm.reg_block.last_compiled_fp = 0;
    vm.reg_block.last_compiled_kind = 0;

    // An NLR escaping through the deoptee: the nested interpreter run returned
    // the reserved-tag sentinel, NOT an oop. Hand it straight back (deopted=1
    // so `stub_poll` tears down and `ret`s it to the loop's caller, which then
    // propagates it via its own `emit_nlr_check`), never as a normal result.
    // (A call-free loop cannot itself originate an NLR — no sends — so this arm
    // is defensive symmetry with `rt_deopt_on_return`; MUST still precede any
    // `Oop::from_raw(result)`, though nothing here rebuilds an oop.)
    if result == crate::oops::layout::NLR_SENTINEL {
        return PollOutcome {
            result: crate::oops::layout::NLR_SENTINEL,
            deopted: 1,
        };
    }

    // The flags stay ARMED here. Disarming needs a native-stack walk to prove
    // "no NotEntrant frame is still running a loop", but `rt_poll` is NOT a
    // legal place to call `walk_frames`: it runs (via `bl stub_poll`) with the
    // enter_compiled `TierLink::IntoCompiled` still on `vm.tier_links` and NO
    // anchor set (`stub_poll` is not one of the six anchor-setting stubs), so
    // `walk_frames`' start rule would assert (`last_compiled_fp == 0` under an
    // IntoCompiled innermost) and — since this is `extern "C"` — that panic
    // aborts the VM. The just-deopted nmethod is still `NotEntrant` anyway (it
    // is freed only by the step-10c zombie sweep), so re-arming is idempotent
    // and correct; disarming is deferred to that sweep, which runs at a GC
    // safepoint where the walk IS legal. Until then every loop keeps polling
    // and `rt_poll` returns `{0,0}` fast for any still-`Alive` frame — a bounded
    // perf cost, never a correctness issue.
    PollOutcome { result, deopted: 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache() -> CodeCache {
        CodeCache::new(1 << 20).unwrap()
    }

    /// S10 P6: the stub's own listing must save AND restore exactly the
    /// allocatable set (x0-x15) — no more, no less.
    #[test]
    fn poll_stub_preserves_x0_x15() {
        let blob = build_stub_poll();
        // Reg's derived Debug is `Reg { class: X, num: N, is_sp: false }` —
        // no listing line ever literally spells "x29", so the fp/lr save
        // (the one `stp`/`ldp` this function DIDN'T reuse for x0-x15) is
        // excluded by its `num: 29` field instead. The trailing comma on
        // `want` matters: `num: 1` is a substring of `num: 10`/`num: 11`.
        let is_x0_15_pair = |l: &&String, mnem: &str| {
            l.contains(mnem) && !l.contains("num: 29,") && !l.contains("num: 30,")
        };
        let saved: Vec<String> = blob
            .listing
            .iter()
            .filter(|l| is_x0_15_pair(l, "stp "))
            .cloned()
            .collect();
        let restored: Vec<String> = blob
            .listing
            .iter()
            .filter(|l| is_x0_15_pair(l, "ldp "))
            .cloned()
            .collect();
        assert_eq!(saved.len(), 8, "8 stp pairs cover x0..x15");
        assert_eq!(restored.len(), 8, "8 ldp pairs restore x0..x15");
        for i in 0..16u8 {
            let want = format!("num: {i},");
            assert!(
                saved.iter().any(|l| l.contains(&want)),
                "x{i} must appear in a saving stp"
            );
            assert!(
                restored.iter().any(|l| l.contains(&want)),
                "x{i} must appear in a restoring ldp"
            );
        }
    }

    /// S13 step 10b: the poll stub's CONTINUE path still restores x0-x15 (the
    /// loop's live regs), AND it now has a deopt-teardown branch. Structural
    /// check on the listing: a `cbz` on the `deopted` flag (x1), and — reached
    /// only when `deopted != 0` — the teardown that discards the loop frame
    /// (`mov sp, x2` after loading loop_fp, then `ldp x29,x30` popping the loop
    /// frame's own record). The x0-x15 save/restore invariant is covered by
    /// `poll_stub_preserves_x0_x15`; this asserts the NEW branch exists.
    #[test]
    fn poll_stub_has_deopt_teardown_branch() {
        let blob = build_stub_poll();
        // A `cbz` (conditional-branch-on-zero) must appear — the `deopted == 0`
        // continue test. `cbz` lowers to a `cbz`/`b`-shaped listing entry.
        assert!(
            blob.listing.iter().any(|l| l.contains("cbz")),
            "the poll stub must branch on rt_poll's `deopted` flag (cbz), got:\n{}",
            blob.listing.join("\n")
        );
        // The teardown's `mov sp, x2` — the SP-register move that discards the
        // stub_poll frame + loop locals down to loop_fp. `sp` is x31/is_sp, so
        // the listing spells the destination with `is_sp: true`.
        assert!(
            blob.listing
                .iter()
                .any(|l| l.contains("mov ") && l.contains("is_sp: true")),
            "the deopt teardown must `mov sp, <reg>` to discard down to loop_fp, got:\n{}",
            blob.listing.join("\n")
        );
        // TWO `ldp x29,x30` epilogues now (one per branch: teardown + continue),
        // vs. exactly one before this step.
        let fp_lr_pops = blob
            .listing
            .iter()
            .filter(|l| l.contains("ldp ") && l.contains("num: 29,"))
            .count();
        assert_eq!(
            fp_lr_pops, 2,
            "two `ldp x29,x30` frame-record pops (deopt-teardown branch + continue branch)"
        );
    }

    /// S13 step 10b: the AArch64 C ABI returns `PollOutcome{result, deopted}`
    /// (two ≤8-byte integer fields, ≤16 bytes total, all-integer) in `x0:x1`
    /// with field 0 in x0, field 1 in x1 — `stub_poll`'s teardown depends on
    /// exactly that. `#[repr(C)]` pins declaration order, so `result` is at
    /// offset 0 and `deopted` at offset 8; the struct is 16 bytes.
    #[test]
    fn poll_outcome_abi_layout() {
        use std::mem::{offset_of, size_of};
        assert_eq!(offset_of!(PollOutcome, result), 0, "result -> x0 (field 0)");
        assert_eq!(
            offset_of!(PollOutcome, deopted),
            8,
            "deopted -> x1 (field 1)"
        );
        assert_eq!(
            size_of::<PollOutcome>(),
            16,
            "a 16-byte all-integer struct returns in x0:x1, not via an x8 sret pointer"
        );
    }

    /// S13 step 10b: `rt_poll`'s decision logic, exercised directly (no native
    /// frame needed for the early-return arms). With `pending_deopt_flag` unset
    /// it returns `{0, 0}` (continue) regardless of any nmethod state — the
    /// fast path every non-deopting poll takes. This is the arm that keeps a
    /// normal run's polls cheap.
    #[test]
    fn rt_poll_continues_when_flag_unset() {
        let mut vm = test_vm();
        vm.pending_deopt_flag = false;
        vm.trace_on_poll = false;
        // ret_pc/loop_fp are irrelevant on this arm (returns before touching
        // code_table). SAFETY: honors rt_poll's contract — `vm` is a live &mut
        // VmState; the flag-unset arm returns before dereferencing loop_fp.
        let out = unsafe { rt_poll(&mut vm as *mut VmState, 0, 0) };
        assert_eq!(out.deopted, 0, "flag unset -> continue (no deopt)");
        assert_eq!(out.result, 0, "continue outcome carries no result");
    }

    #[test]
    fn stubs_install_and_publish() {
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        assert!(cache.contains(stubs.call_stub_entry() as u64));
        assert!(cache.contains(stubs.stub_poll_addr()));
        assert!(cache.contains(stubs.resolve_addr()));
        assert!(cache.contains(stubs.c2i_shared_addr()));
        assert!(cache.contains(stubs.mega_shared_addr()));
        assert!(cache.contains(stubs.dnu_addr()));
        assert!(cache.contains(stubs.must_be_boolean_addr()));
        assert!(cache.contains(stubs.alloc_slow_addr()));
        assert!(cache.contains(stubs.not_entrant_addr()));
    }

    /// WINVM: `install` must publish the **x86-64** builders' output on an
    /// x86-64 host.
    ///
    /// The test above passes either way — it only checks that the
    /// addresses land inside the cache, which they did while `install`
    /// was still publishing AArch64 encodings on every host. So this one
    /// compares the published bytes against the x64 builders directly,
    /// which is the only thing that distinguishes "a stub is installed"
    /// from "the right architecture's stub is installed".
    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    #[allow(unsafe_code)]
    fn install_publishes_x64_stub_encodings() {
        use super::super::stubs_x64::{build_call_stub_x64, build_unreachable_stub_x64};

        let mut cache = test_cache();
        let stubs = install(&mut cache);

        // SAFETY: each handle names live, published code-cache bytes of
        // at least the corresponding blob's length.
        let published = |base: *const u8, len: usize| -> Vec<u8> {
            unsafe { std::slice::from_raw_parts(base, len) }.to_vec()
        };

        let want = build_call_stub_x64();
        assert_eq!(
            published(stubs.call_stub.base, want.code.len()),
            want.code,
            "call_stub must be the x64 encoding — AArch64 bytes here execute as garbage"
        );

        // The three SIMD boxers are deliberately unreachable on x64: no
        // `FBox`/`VecArith` lowering exists, so nothing can call them.
        let ud2 = build_unreachable_stub_x64();
        for (name, h) in [
            ("box_float64x2", stubs.box_float64x2),
            ("box_float32x4", stubs.box_float32x4),
            ("box_int32x4", stubs.box_int32x4),
        ] {
            assert_eq!(
                published(h.base, ud2.code.len()),
                ud2.code,
                "{name} has no x64 lowering yet and must fault loudly, not run stale bytes"
            );
        }
    }

    fn test_vm() -> VmState {
        VmState::with_options(crate::runtime::vm_state::VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        })
    }

    /// S11 step 6's own explicit scope for `must_be_boolean`: the STUB
    /// itself, testable in isolation via a direct `Stubs::invoke` (same
    /// shape any nmethod entry uses) -- NOT its wiring into
    /// `Ir::CallRuntime`, which is S11 step 7's job and has no emission
    /// path to test through yet.
    #[cfg(target_arch = "aarch64")] // WINVM: executes A64 stub/compiled code; x64 in Phase 3
    #[test]
    fn must_be_boolean_sends_and_returns_result() {
        let mut vm = test_vm();
        let smi_klass = vm.universe.smi_klass;
        let sel = vm.universe.sel_must_be_boolean;
        let mut b = crate::bytecode::builder::BytecodeBuilder::new();
        b.push_true();
        b.ret_tos();
        let handler = b.finish(&mut vm, sel, 0, 0);
        crate::runtime::lookup::install_method(&mut vm, smi_klass, sel, handler);

        let stubs = vm.stubs;
        let val = crate::oops::smi::SmallInt::new(42).oop();
        let true_obj = vm.universe.true_obj;
        let result = stubs.invoke(stubs.must_be_boolean_addr(), &mut vm, &[val.raw()]);
        assert_eq!(
            result,
            true_obj.raw(),
            "a non-boolean receiver with a real #mustBeBoolean handler must reach it"
        );
    }

    // S11 step 1 had a placeholder `rt_resolve_send` (echoed `ret_addr`
    // straight back) and two tests scoped to exactly that: a round trip
    // through real assembly, and a direct Rust-level call. S11 step 3
    // replaces the placeholder with the real D4.1 lookup+patch logic,
    // which requires `ret_addr` to fall inside a genuinely installed
    // `Nmethod` with a matching `IcSite` — neither placeholder test's
    // fabricated `ret_addr`/bare `VmState` satisfies that anymore, and
    // both are superseded by `tests/it_tier1.rs`'s
    // `mono_resolve_patches_call_site_and_dispatches`, a strictly stronger
    // claim (real dispatch reaches the right target and returns the right
    // value, not just "some bytes came back unchanged").
}
