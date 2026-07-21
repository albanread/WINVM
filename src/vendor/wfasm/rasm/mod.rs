// Vendored from JASM (wfasm), https://github.com/albanread/JASM  commit f2177391538cbede0c8cfcaa29bd3303ae421d0c
// Original path: rust/src/rasm/mod.rs.  License: MIT (see LICENSE-JASM in this
// directory; Copyright (c) 2026 alban read).
// Local modifications are marked with `// WINVM:` comments — keep the diff
// against upstream minimal so re-vendoring stays mechanical.
// WINVM: `crate::backend`/`crate::rasm` -> `crate::vendor::wfasm::*` (modules
// moved under WINVM's own tree).

//! Rasm — the native x86-64 encoder that replaces LLVM-MC.
//!
//! Input: the assembled (post-macro-expansion) MC-flavour Intel-syntax text the
//! `asm/` front-end already produces. Output: an [`EncodedModule`](crate::vendor::wfasm::backend::EncodedModule)
//! the native [`NativeJit`](crate::vendor::wfasm::native_windows::WinJit) loads. Tables/logic are
//! derived from LLVM-MC for byte-identity (see WF65 docs/design/rasm-replace-llvm.md).
//!
//! Layering: [`parse`] (text → [`Line`]) → [`encode`] (one instruction → bytes)
//! → this module's two-pass driver (assign offsets, resolve internal labels +
//! branch relaxation, emit relocs) → `EncodedModule`.

pub mod assemble;
pub mod encode;
pub mod parse;

pub use assemble::assemble;
pub use encode::{encode, Encoded, Fixup, FixupKind};
pub use parse::{Directive, Line, Mem, MemSize, Operand, Reg, RegClass};

/// The native from-scratch x86-64 [`Encoder`](crate::vendor::wfasm::backend::Encoder) — the
/// owned replacement for LLVM-MC.
#[derive(Debug, Default, Clone, Copy)]
pub struct RasmEncoder;

impl crate::vendor::wfasm::backend::Encoder for RasmEncoder {
    fn encode(&self, asm_text: &str) -> anyhow::Result<crate::vendor::wfasm::backend::EncodedModule> {
        assemble(asm_text)
    }
}
