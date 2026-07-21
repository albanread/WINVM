//! Runtime support: the `VmState` god-struct, environment options, method
//! lookup, primitives, and (later) activation stacks and deoptimization
//! (SPEC §3, §6, §10). No `unsafe` here — everything above `oops`/`memory`
//! stays safe Rust (CONVENTIONS §1) — with ONE deliberate exception:
//! `frames` (S12 D3), which walks the NATIVE call stack (compiled/stub
//! frames have no Rust-visible representation at all) and so needs raw
//! pointer reads no safe abstraction can provide — the same category of
//! exception `codecache`'s own "sole owner of raw MAP_JIT pointer calls"
//! boundary already establishes, just for stack memory instead of code
//! memory.

pub mod alien;
pub mod debug;
pub mod deopt;
pub mod deps;
pub mod error;
pub mod ffi;
// WINVM (docs/FFI.md): the Windows API knowledge-base resolver — the
// substitution of cocoa_data as the FFI's source of truth.
pub mod winkb;
pub mod frames;
pub mod globals;
pub mod lookup;
pub mod mop; // MOP object pickle — multi-Smalltalk workers M0 (docs/multi-smalltalk-worker.md §4)
// WINVM: the Cocoa bridge is macOS-only by definition (objc_msgSend, the
// objc_shim.m @try/@catch shim, AppKit delegates). Its Win32 counterpart
// arrives with the Phase-6 native shell (MIGRATION.md §6).
#[cfg(target_os = "macos")]
pub mod objc_bridge; // Cocoa bridge C0 — ObjcRef + ownership + bottom pool (docs/cocoa_bridge_design.md)
#[cfg(target_os = "macos")]
pub mod objc_delegate; // Cocoa bridge C6 — reverse dispatch: delegates as top-level entries (cocoa_gui_design.md §4, CG3)
pub mod osr;
pub mod primitives;
pub mod probe;
pub mod recompile;
pub mod simd_kernels; // SIMD level 2: explicit hand-written NEON bulk kernels (docs/SIMD.md Part E)
pub mod vm_state;
pub mod workers; // primary/worker registry + channels — workers M1 (docs/multi-smalltalk-worker.md §3)

pub use vm_state::{
    jit_compile_enabled, set_jit_compile_enabled, InterpRegs, JitMode, TierLink, TraceFlags,
    VmOptions, VmRegBlock, VmState,
};

/// A guest-execution failure (SPEC §3.2 bootstrap, `sprint_s05_detail.md`
/// §Interfaces for later sprints). No Rust-level failure path exists yet
/// for a running method (DNU/errors print-and-exit, SPEC §6.3) — this type
/// exists so the frontend/world loader's call shape is stable once real
/// exception handling lands.
#[derive(Debug)]
pub struct VmError {
    pub msg: String,
}

/// `frontend/`'s single entry point into `interpreter/` (layer boundary,
/// `sprint_s05_detail.md` §Layer boundaries) — runs a compiled `#doIt`
/// (nil receiver, no args) to completion.
pub fn execute_doit(
    vm: &mut VmState,
    m: crate::oops::wrappers::MethodOop,
) -> Result<crate::oops::Oop, VmError> {
    let nil = vm.universe.nil_obj;
    Ok(crate::interpreter::run_method(vm, m, nil, &[]))
}

/// Sends `selector` (unary) to `receiver` iff it is understood; a miss is a
/// silent no-op (used only for the world-load bootstrap's `Smalltalk
/// startUp` send, SPEC §3.2 step 3 — a real send whose absence in an
/// early/incomplete world must never abort the load).
pub fn send_unary_if_understood(
    vm: &mut VmState,
    receiver: crate::oops::Oop,
    selector: crate::oops::wrappers::SymbolOop,
) -> Option<crate::oops::Oop> {
    let klass = lookup::klass_of(vm, receiver);
    let m = lookup::lookup(vm, klass, selector)?;
    Some(crate::interpreter::run_method(vm, m, receiver, &[]))
}
