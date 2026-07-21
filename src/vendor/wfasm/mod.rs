//! Vendored slice of JASM's `wfasm` AArch64 encoder + macOS JIT loader +
//! relocation patcher, plus a frozen-corpus replay test carrying its
//! LLVM-MC-verified trust gate into MACVM without an LLVM dependency. See
//! `VENDOR.md` for the exact file list, source commit, and per-file edits;
//! `docs/sprints/sprint_s09_detail.md` D1 for the vendoring procedure this
//! tree follows.
//!
//! `src/compiler/jasm_assembler.rs` is the only consumer outside this tree
//! (sprint_s09_detail.md D4: nothing else may `use crate::vendor::…`).

#![allow(dead_code)]
// MACVM uses a subset of the vendored surface
// Vendored/derived code (this whole tree) keeps upstream's original style
// per VENDOR.md's "minimal diff against upstream" rule — these are the
// specific clippy lints that style trips (encode.rs's bit-math OR-chains
// and wide instruction-field argument lists; a `% 2` parity check and a
// `vec![x; n]` kept from difftest.rs's own `unhex`/test code), not a
// blanket suppression for anything MACVM writes fresh under this tree.
#![allow(
    clippy::identity_op,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::useless_vec
)]
pub mod a64;
pub mod backend;
// WINVM (Phase 2, MIGRATION.md §4): the native x86-64 encoder — JASM's
// `rasm`, the sibling of `a64`. Vendored whole; compiles on every platform
// (pure Rust emitting bytes), used by `native_windows::WinJit`.
pub mod rasm;
#[cfg(test)]
mod corpus_replay;
#[cfg(target_os = "macos")]
pub mod native_macos;
// WINVM: the Windows x86-64 sibling loader (VirtualAlloc RWX region,
// FlushInstructionCache, LoadLibrary/GetProcAddress) — MIGRATION.md §2.2.
#[cfg(windows)]
pub mod native_windows;
pub mod relocpatch;

// WINVM: the per-OS loader under one portable name, so consumers
// (`codecache`, `runtime::ffi`) need no per-OS import forests. The Cocoa
// bridge keeps importing `native_macos` directly — it is macOS-only anyway.
#[cfg(target_os = "macos")]
pub use native_macos as native;
#[cfg(windows)]
pub use native_windows as native;
