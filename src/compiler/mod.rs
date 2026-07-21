//! Adaptive optimizing compiler.
//!
//! Recompiles hot methods using type feedback gathered by the inline caches in
//! [`crate::interpreter::ic`], and emits native code through the backend-neutral
//! [`assembler::Assembler`] trait, over JASM's vendored AArch64 encoder
//! (`docs/DESIGN.md` §5 D6; `docs/sprints/sprint_s09_detail.md`).
//! [`jasm_assembler::JasmAssembler`] is the only implementor.
//!
//! Tier 1 pipeline (S10, `docs/sprints/sprint_s10_detail.md`): bytecode ->
//! [`decode::Cfg`] -> SSA-lite `Ir` -> linear-scan regalloc -> emit.

pub mod assembler;
// WINVM (Phase 3, MIGRATION.md §4): the x86-64 emitter, sibling of
// `jasm_assembler`. Compiles on every host (it only writes bytes), so its
// encoding tests run everywhere; only the tier-1 wiring is target-gated.
pub mod assembler_x64;
pub mod decode;
// WINVM (Phase 3): the x86-64 IR lowering, sibling of `emit`. Currently a
// documented vertical slice (see its `SUPPORTED_OPS`).
pub mod emit_x64;
pub mod disasm_a64;
// WINVM (Phase 3x): x86-64 disassembly for the PROBE dossier. Compiles
// everywhere (iced-x86 decodes x64 on any host); only the dossier's
// choice of decoder is target-gated.
pub mod disasm_x64;
pub mod driver;
pub mod emit;
pub mod escape;
pub mod feedback;
pub mod inline;
pub mod ir;
pub mod jasm_assembler;
pub mod oopmap;
pub mod regalloc;
pub mod scopes;
