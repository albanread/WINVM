//! `X64Assembler` — the x86-64 code emitter (WINVM Phase 3, MIGRATION.md
//! §4), the sibling of [`JasmAssembler`](crate::compiler::jasm_assembler)
//! over the vendored native encoder (`crate::vendor::wfasm::rasm::encode`).
//!
//! It produces the SAME [`CodeBlob`] the AArch64 side does — code first,
//! then an 8-byte-aligned literal pool at `literal_off`, plus `relocs` and
//! a `listing` — so everything downstream (`CodeCache::publish`,
//! `nmethod`, the GC's pool-word walk) is unchanged.
//!
//! ## Three deliberate differences from the A64 assembler
//!
//! 1. **Variable-length instructions.** `offset()` is not 4-aligned and an
//!    instruction is 1..15 bytes. Every intra-blob branch is therefore
//!    emitted in its **rel32 form unconditionally** (`jmp rel32` = 5 bytes,
//!    `jcc rel32` = 6) rather than letting the encoder pick a short form:
//!    the width must be known before the displacement is, or a relaxation
//!    pass would be required. Wasting 3 bytes on a short jump is the right
//!    trade against a whole extra pass.
//!
//! 2. **Symbol operands ARE used** — the exact opposite of the A64 side's
//!    P6 rule. That rule existed because the vendored *A64* encoder's `Sym`
//!    path belonged to its text front end. The x64 encoder instead returns
//!    structured [`Fixup`](crate::vendor::wfasm::rasm::Fixup)s
//!    (`Rel32`/`RipRel32`) naming the field offset within the instruction,
//!    which is precisely the hook a structured emitter wants: we hand it
//!    `Operand::Sym("L3")`, take back the fixup, and resolve it ourselves
//!    at [`finish`]. No text is parsed; the operands stay structured.
//!
//! 3. **RIP-relative literals instead of `ldr`-literal.** A pool word is
//!    read with `mov reg, qword ptr [rip + litN]`, whose `RipRel32` fixup
//!    resolves the same way a branch does. Displacements are relative to
//!    the *end of the instruction* (x86's RIP semantics), which is why
//!    every fixup records `insn_end` and not just the field offset.
//!
//! ## Register roles (MIGRATION.md §2.1)
//!
//! The pinned VM registers are callee-saved on Win64, so a compiled frame
//! that establishes them keeps them across calls into the runtime:
//!
//! | role | x64 | A64 original |
//! |---|---|---|
//! | `&mut VmState` | `R15` | `x28` |
//! | receiver cache | `R14` | `x27` |
//! | bytecode pointer | `R13` | `x26` |
//! | method/frame info | `R12` | `x25` |
//! | frame pointer | `RBP` | `x29` |
//! | scratch | `R10`/`R11` | `x16`/`x17` |
//!
//! `R10` doubles as the deopt trap-pc stash the VEH writes
//! (`codecache::deopt_trap`), mirroring x16's role on the macOS side.

use std::collections::HashMap;

use crate::compiler::assembler::{CodeBlob, Label, LiteralId, Reloc, RelocKind};
use crate::vendor::wfasm::rasm::encode;
use crate::vendor::wfasm::rasm::parse::{Mem, MemSize, Operand, Reg, RegClass};

// ── Register numbering and pinned roles ─────────────────────────────────

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

/// `&mut VmState` — the x28 analogue. Callee-saved, so it survives the
/// runtime calls compiled code makes.
pub const VM_STATE: u8 = R15;
/// Cached receiver (x27 analogue).
pub const RECEIVER: u8 = R14;
/// Bytecode pointer for tier-0 interop (x26 analogue).
pub const BCP: u8 = R13;
/// Method / frame info (x25 analogue).
pub const METHOD: u8 = R12;
/// First scratch (x16/IP0 analogue) — also the deopt trap-pc stash.
pub const SCRATCH0: u8 = R10;
/// Second scratch (x17/IP1 analogue).
pub const SCRATCH1: u8 = R11;

/// Win64 integer argument registers, in order.
pub const ARG_REGS: [u8; 4] = [RCX, RDX, R8, R9];
/// Win64's mandatory 32-byte shadow space every call must reserve.
pub const SHADOW_SPACE: i32 = 32;

// ── Condition codes ─────────────────────────────────────────────────────

/// x86-64 condition codes. Signed comparisons use `L`/`Le`/`G`/`Ge`
/// (Smalltalk SmallInteger compares are signed); the unsigned `B`/`A`
/// family is for tag and bounds checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cond {
    E,
    Ne,
    L,
    Le,
    G,
    Ge,
    B,
    Be,
    A,
    Ae,
    O,
    No,
    S,
    Ns,
}

impl Cond {
    /// The `jcc` mnemonic for this condition.
    pub fn jcc(self) -> &'static str {
        match self {
            Cond::E => "je",
            Cond::Ne => "jne",
            Cond::L => "jl",
            Cond::Le => "jle",
            Cond::G => "jg",
            Cond::Ge => "jge",
            Cond::B => "jb",
            Cond::Be => "jbe",
            Cond::A => "ja",
            Cond::Ae => "jae",
            Cond::O => "jo",
            Cond::No => "jno",
            Cond::S => "js",
            Cond::Ns => "jns",
        }
    }

    /// The condition that is true exactly when this one is false — for
    /// inverting a branch when the emitter wants fallthrough on the other
    /// side.
    pub fn inverse(self) -> Cond {
        match self {
            Cond::E => Cond::Ne,
            Cond::Ne => Cond::E,
            Cond::L => Cond::Ge,
            Cond::Ge => Cond::L,
            Cond::Le => Cond::G,
            Cond::G => Cond::Le,
            Cond::B => Cond::Ae,
            Cond::Ae => Cond::B,
            Cond::Be => Cond::A,
            Cond::A => Cond::Be,
            Cond::O => Cond::No,
            Cond::No => Cond::O,
            Cond::S => Cond::Ns,
            Cond::Ns => Cond::S,
        }
    }
}

// ── Operand helpers ─────────────────────────────────────────────────────

/// A 64-bit GPR operand (`rax`..`r15`).
pub fn r64(n: u8) -> Operand {
    Operand::Reg(Reg {
        class: RegClass::R64,
        num: n,
    })
}
/// A 32-bit GPR operand (`eax`..`r15d`) — for 32-bit ops whose zero-extend
/// to 64 bits is free.
pub fn r32(n: u8) -> Operand {
    Operand::Reg(Reg {
        class: RegClass::R32,
        num: n,
    })
}
/// An SSE register operand (`xmm0`..`xmm15`) — unboxed doubles.
pub fn xmm(n: u8) -> Operand {
    Operand::Reg(Reg {
        class: RegClass::Xmm,
        num: n,
    })
}
pub fn imm(v: i64) -> Operand {
    Operand::Imm(v)
}
/// `qword ptr [base + disp]`.
pub fn mem(base: u8, disp: i64) -> Operand {
    Operand::Mem(Mem {
        size: Some(MemSize::Qword),
        base: Some(Reg {
            class: RegClass::R64,
            num: base,
        }),
        index: None,
        scale: 1,
        disp,
        rip_sym: None,
    })
}
/// `qword ptr [base + index*scale + disp]` — indexed object/array access.
pub fn mem_index(base: u8, index: u8, scale: u8, disp: i64) -> Operand {
    debug_assert!(
        matches!(scale, 1 | 2 | 4 | 8),
        "x86 SIB scale must be 1, 2, 4, or 8 (got {scale})"
    );
    Operand::Mem(Mem {
        size: Some(MemSize::Qword),
        base: Some(Reg {
            class: RegClass::R64,
            num: base,
        }),
        index: Some(Reg {
            class: RegClass::R64,
            num: index,
        }),
        scale,
        disp,
        rip_sym: None,
    })
}
/// A bare symbol operand — a branch target or a RIP-relative literal.
fn sym(name: String) -> Operand {
    Operand::Sym(name)
}
/// `qword ptr [rip + name]`.
fn mem_rip(name: String) -> Operand {
    Operand::Mem(Mem {
        size: Some(MemSize::Qword),
        base: None,
        index: None,
        scale: 1,
        disp: 0,
        rip_sym: Some(name),
    })
}

// ── The assembler ───────────────────────────────────────────────────────

/// A fixup whose target is not known when the instruction is emitted.
struct Pending {
    /// Absolute offset of the displacement field within the buffer.
    field: u32,
    /// Absolute offset of the first byte AFTER the instruction — x86
    /// displacements are relative to this (RIP semantics), never to the
    /// field itself.
    insn_end: u32,
    target: Target,
}

enum Target {
    Label(Label),
    Literal(LiteralId),
}

/// The label naming scheme handed to the encoder as `Operand::Sym`.
/// Internal to this module — the names never escape into a `CodeBlob`.
fn label_sym(l: Label) -> String {
    format!("L{}", l.0)
}
fn literal_sym(id: LiteralId) -> String {
    format!("lit{}", id.0)
}

pub struct X64Assembler {
    buf: Vec<u8>,
    labels: Vec<Option<u32>>,
    pending: Vec<Pending>,
    literals: Vec<(u64, Option<RelocKind>)>,
    lit_dedup: HashMap<(u64, Option<RelocKind>), LiteralId>,
    relocs: Vec<Reloc>,
    listing: Vec<String>,
    finished: bool,
}

impl X64Assembler {
    pub fn new() -> Self {
        X64Assembler {
            buf: Vec::new(),
            labels: Vec::new(),
            pending: Vec::new(),
            literals: Vec::new(),
            lit_dedup: HashMap::new(),
            relocs: Vec::new(),
            listing: Vec::new(),
            finished: false,
        }
    }

    /// Current write position (bytes from blob start). Unlike the A64
    /// assembler this has NO alignment guarantee.
    pub fn offset(&self) -> u32 {
        self.buf.len() as u32
    }

    fn push_listing(&mut self, offset: u32, bytes: &[u8], text: &str) {
        if cfg!(debug_assertions) || cfg!(test) {
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            self.listing.push(format!("{offset:06x}  {hex:<20}  {text}"));
        }
    }

    /// Encode one instruction. Any fixup the encoder returns must name a
    /// label or literal symbol this assembler minted (`emit` with a raw
    /// `Sym` the caller invented is a compiler bug — the resolver would
    /// have nothing to point it at).
    fn emit_inner(&mut self, mnemonic: &str, ops: &[Operand], target: Option<Target>) {
        let start = self.offset();
        let enc = encode::encode(mnemonic, ops)
            .unwrap_or_else(|e| panic!("X64Assembler::emit(\"{mnemonic}\", {ops:?}): {e:#}"));
        let insn_end = start + enc.bytes.len() as u32;
        match (enc.fixups.len(), target) {
            (0, None) => {}
            (1, Some(t)) => {
                let f = &enc.fixups[0];
                self.pending.push(Pending {
                    field: start + f.at as u32,
                    insn_end,
                    target: t,
                });
            }
            (n, t) => panic!(
                "X64Assembler::emit(\"{mnemonic}\", {ops:?}): encoder produced {n} fixup(s) but \
                 {} target was supplied — every symbol operand must come from this assembler's \
                 own label/literal minting",
                if t.is_some() { "a" } else { "no" }
            ),
        }
        self.buf.extend_from_slice(&enc.bytes);
        self.push_listing(start, &enc.bytes, &format!("{mnemonic} {ops:?}"));
    }

    /// Encode one plain instruction (no symbol operands).
    pub fn emit(&mut self, mnemonic: &str, ops: &[Operand]) {
        self.emit_inner(mnemonic, ops, None);
    }

    /// Emit raw pre-encoded bytes — the escape hatch for forms the encoder
    /// doesn't cover (the deopt trap site's `int3` + imm16, S13's x64
    /// counterpart).
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        let start = self.offset();
        self.buf.extend_from_slice(bytes);
        self.push_listing(start, bytes, "<raw>");
    }

    // ── labels ──────────────────────────────────────────────────────────

    pub fn new_label(&mut self) -> Label {
        let id = self.labels.len() as u32;
        self.labels.push(None);
        Label(id)
    }

    /// Fix `l` at the current offset.
    pub fn bind(&mut self, l: Label) {
        let off = self.offset();
        debug_assert!(
            self.labels[l.0 as usize].is_none(),
            "X64Assembler::bind: label L{} bound twice",
            l.0
        );
        self.labels[l.0 as usize] = Some(off);
    }

    /// `jmp rel32` (always the near form — see the module doc).
    pub fn jmp(&mut self, l: Label) {
        self.emit_inner("jmp", &[sym(label_sym(l))], Some(Target::Label(l)));
    }

    /// `jcc rel32` (always the near form).
    pub fn jcc(&mut self, c: Cond, l: Label) {
        self.emit_inner(c.jcc(), &[sym(label_sym(l))], Some(Target::Label(l)));
    }

    // ── literal pool ────────────────────────────────────────────────────

    /// Intern an 8-byte pool constant, deduplicated by `(value, kind)` —
    /// never by value alone (P10, same rule as the A64 pool: an address
    /// that is both a `RuntimeAddr` and an `Oop` must not share a word, or
    /// GC would rewrite the runtime address).
    pub fn literal_u64(&mut self, v: u64, kind: Option<RelocKind>) -> LiteralId {
        let key = (v, kind);
        if let Some(&id) = self.lit_dedup.get(&key) {
            return id;
        }
        let id = LiteralId(self.literals.len() as u32);
        self.literals.push((v, kind));
        self.lit_dedup.insert(key, id);
        id
    }

    /// `mov dst, qword ptr [rip + litN]` — load an interned pool word.
    pub fn load_literal(&mut self, dst: u8, lit: LiteralId) {
        self.emit_inner(
            "mov",
            &[r64(dst), mem_rip(literal_sym(lit))],
            Some(Target::Literal(lit)),
        );
    }

    // ── calls ───────────────────────────────────────────────────────────

    /// A patchable `call rel32` placeholder (self-call, displacement 0)
    /// plus its reloc. Returns the CALL instruction's own byte offset; the
    /// rel32 field is at `offset + 1`. The 5-byte shape is fixed so the
    /// code cache can rewrite the target with a single aligned-enough
    /// 4-byte store (single-threaded execution — MIGRATION.md §2.2).
    pub fn call_patchable(&mut self, kind: RelocKind) -> u32 {
        let offset = self.offset();
        // E8 rel32, displacement 0 == "call the next instruction", the
        // placeholder the patcher overwrites.
        self.emit_bytes(&[0xE8, 0x00, 0x00, 0x00, 0x00]);
        self.relocs.push(Reloc { offset, kind });
        offset
    }

    /// `mov r10, [rip + pool]; call r10` — call an absolute address. Rust
    /// runtime functions routinely sit outside a blob's ±2 GB `rel32`
    /// reach, and routing through the pool keeps the target word
    /// relocatable (the A64 side's `call_far` reasoning, unchanged).
    pub fn call_far(&mut self, target: LiteralId) {
        self.load_literal(SCRATCH0, target);
        self.emit("call", &[r64(SCRATCH0)]);
    }

    /// Resolve every pending fixup, append the 8-byte-aligned literal pool,
    /// and hand over the finished blob. Panics on an unbound label.
    pub fn finish(&mut self) -> CodeBlob {
        assert!(!self.finished, "X64Assembler::finish called twice");
        self.finished = true;

        // The pool follows the code, 8-aligned (GC rewrites pool words in
        // place and never touches instruction bytes).
        while !self.buf.len().is_multiple_of(8) {
            self.buf.push(0x90); // nop padding, so the gap stays disassemblable
        }
        let literal_off = self.buf.len() as u32;
        let mut relocs = std::mem::take(&mut self.relocs);
        for (i, &(v, kind)) in self.literals.iter().enumerate() {
            self.buf.extend_from_slice(&v.to_le_bytes());
            if let Some(kind) = kind {
                relocs.push(Reloc {
                    offset: literal_off + 8 * i as u32,
                    kind,
                });
            }
        }

        // Now every target offset is known. x86 displacements are relative
        // to the END of the instruction.
        for p in &self.pending {
            let target = match p.target {
                Target::Label(l) => self.labels[l.0 as usize].unwrap_or_else(|| {
                    panic!("X64Assembler::finish: unbound label L{}", l.0)
                }),
                Target::Literal(lit) => literal_off + 8 * lit.0,
            };
            let disp = target as i64 - p.insn_end as i64;
            let disp32 = i32::try_from(disp).unwrap_or_else(|_| {
                panic!("X64Assembler::finish: displacement {disp} exceeds rel32 range")
            });
            let at = p.field as usize;
            self.buf[at..at + 4].copy_from_slice(&disp32.to_le_bytes());
        }

        CodeBlob {
            code: std::mem::take(&mut self.buf),
            literal_off,
            relocs,
            listing: std::mem::take(&mut self.listing),
        }
    }
}

impl Default for X64Assembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Disassemble `code[..len]` with iced-x86 and return one Intel-syntax
    /// line per instruction. The encoder's own byte-level correctness is
    /// JASM's difftest business; what these tests verify is that THIS
    /// layer wires operands, labels, and displacements up correctly — and
    /// reading that back through an independent decoder is far stronger
    /// evidence than asserting hand-copied byte strings.
    fn disasm(code: &[u8], base: u64) -> Vec<String> {
        use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};
        let mut d = Decoder::with_ip(64, code, base, DecoderOptions::NONE);
        let mut f = IntelFormatter::new();
        let mut out = Vec::new();
        let mut insn = Instruction::default();
        while d.can_decode() {
            d.decode_out(&mut insn);
            let mut s = String::new();
            f.format(&insn, &mut s);
            out.push(s);
        }
        out
    }

    #[test]
    fn plain_instructions_encode() {
        let mut a = X64Assembler::new();
        a.emit("mov", &[r64(RAX), imm(42)]);
        a.emit("add", &[r64(RAX), r64(RCX)]);
        a.emit("mov", &[r64(RBX), mem(R15, 16)]);
        a.emit("ret", &[]);
        let blob = a.finish();
        let text = disasm(&blob.code[..blob.literal_off as usize], 0x1000);
        assert_eq!(text[0], "mov rax,2Ah");
        assert_eq!(text[1], "add rax,rcx");
        assert_eq!(text[2], "mov rbx,[r15+10h]");
        assert_eq!(text[3], "ret");
    }

    /// The pinned VM registers encode as the registers MIGRATION.md §2.1
    /// names — a mix-up here would be silent and catastrophic.
    #[test]
    fn pinned_register_roles_are_the_documented_ones() {
        let mut a = X64Assembler::new();
        a.emit("mov", &[r64(VM_STATE), r64(RECEIVER)]);
        a.emit("mov", &[r64(BCP), r64(METHOD)]);
        a.emit("mov", &[r64(SCRATCH0), r64(SCRATCH1)]);
        let blob = a.finish();
        let text = disasm(&blob.code[..blob.literal_off as usize], 0);
        assert_eq!(text[0], "mov r15,r14");
        assert_eq!(text[1], "mov r13,r12");
        assert_eq!(text[2], "mov r10,r11");
    }

    /// A forward branch: the displacement is resolved at `finish` against
    /// a label bound later, and is relative to the END of the jump.
    #[test]
    fn forward_branch_resolves() {
        let mut a = X64Assembler::new();
        let l = a.new_label();
        a.jmp(l);
        a.emit("nop", &[]);
        a.emit("nop", &[]);
        a.bind(l);
        a.emit("ret", &[]);
        let blob = a.finish();
        // jmp rel32 is 5 bytes; two nops follow; target is offset 7.
        let disp = i32::from_le_bytes(blob.code[1..5].try_into().unwrap());
        assert_eq!(disp, 2, "target 7 - insn_end 5");
        let text = disasm(&blob.code[..blob.literal_off as usize], 0);
        assert_eq!(text[0], "jmp 7", "decodes as a jump to offset 7");
    }

    /// A backward branch produces a negative displacement.
    #[test]
    fn backward_branch_resolves() {
        let mut a = X64Assembler::new();
        let l = a.new_label();
        a.bind(l);
        a.emit("nop", &[]);
        a.emit("nop", &[]);
        a.jcc(Cond::E, l);
        let blob = a.finish();
        // Two nops (2 bytes), then jcc rel32 (6 bytes) ending at 8.
        let disp = i32::from_le_bytes(blob.code[4..8].try_into().unwrap());
        assert_eq!(disp, -8, "target 0 - insn_end 8");
        let text = disasm(&blob.code[..blob.literal_off as usize], 0);
        assert_eq!(text[2], "je 0", "decodes as a jump back to offset 0");
    }

    /// Every condition round-trips to the jcc the decoder agrees with, and
    /// `inverse` is a true involution over the whole set.
    #[test]
    fn conditions_map_to_the_right_jcc() {
        let all = [
            Cond::E,
            Cond::Ne,
            Cond::L,
            Cond::Le,
            Cond::G,
            Cond::Ge,
            Cond::B,
            Cond::Be,
            Cond::A,
            Cond::Ae,
            Cond::O,
            Cond::No,
            Cond::S,
            Cond::Ns,
        ];
        for c in all {
            assert_eq!(c.inverse().inverse(), c, "{c:?} inverse is an involution");
            assert_ne!(c.inverse(), c);
            let mut a = X64Assembler::new();
            let l = a.new_label();
            a.jcc(c, l);
            a.bind(l);
            let blob = a.finish();
            let text = disasm(&blob.code[..blob.literal_off as usize], 0);
            // iced canonicalizes synonym mnemonics (jc/jnae both print as
            // jb), so assert the shape: a conditional jump landing at 6,
            // the fallthrough label right after the 6-byte jcc.
            assert!(
                text[0].starts_with('j') && text[0].ends_with(" 6"),
                "{c:?} -> {}",
                text[0]
            );
        }
    }

    /// A RIP-relative pool load resolves to the pool word's address, and
    /// the pool itself is 8-aligned after the code.
    #[test]
    fn literal_load_is_rip_relative_to_the_pool_word() {
        let mut a = X64Assembler::new();
        let lit = a.literal_u64(0xDEAD_BEEF_CAFE_1234, Some(RelocKind::Oop));
        a.load_literal(RAX, lit);
        a.emit("ret", &[]);
        let blob = a.finish();
        assert_eq!(blob.literal_off % 8, 0, "pool is 8-aligned");

        // The pool word holds the value...
        let off = blob.literal_off as usize;
        assert_eq!(
            u64::from_le_bytes(blob.code[off..off + 8].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_1234
        );
        // ...the reloc points at it...
        assert_eq!(blob.relocs.len(), 1);
        assert_eq!(blob.relocs[0].kind, RelocKind::Oop);
        assert_eq!(blob.relocs[0].offset, blob.literal_off);
        // ...and the instruction's RIP-relative target IS that word.
        let base = 0x4000u64;
        let text = disasm(&blob.code[..blob.literal_off as usize], base);
        let want = format!("mov rax,[{:X}h]", base + blob.literal_off as u64);
        assert_eq!(text[0], want, "RIP-relative target IS the pool word");
    }

    /// Pool entries dedup by `(value, kind)` — never by value alone (P10).
    #[test]
    fn pool_dedup_is_by_value_and_kind() {
        let mut a = X64Assembler::new();
        let l1 = a.literal_u64(100, Some(RelocKind::Oop));
        let l2 = a.literal_u64(100, Some(RelocKind::Oop));
        let l3 = a.literal_u64(100, None);
        let l4 = a.literal_u64(200, None);
        assert_eq!(l1, l2, "same value AND kind dedups");
        assert_ne!(l1, l3, "same value, different kind must not dedup");
        assert_ne!(l3, l4);
        let blob = a.finish();
        assert_eq!(
            blob.code.len() - blob.literal_off as usize,
            3 * 8,
            "three distinct pool words"
        );
    }

    /// `call_patchable` lays the exact 5-byte `E8 rel32` shape the code
    /// cache patches, and records the reloc at the instruction's start.
    #[test]
    fn call_patchable_shape_and_reloc() {
        let mut a = X64Assembler::new();
        a.emit("nop", &[]);
        let site = a.call_patchable(RelocKind::InlineCache);
        let blob = a.finish();
        assert_eq!(site, 1, "site offset is the CALL's own first byte");
        assert_eq!(blob.code[1], 0xE8, "E8 = call rel32");
        assert_eq!(
            &blob.code[2..6],
            &[0, 0, 0, 0],
            "displacement 0 placeholder"
        );
        assert_eq!(blob.relocs.len(), 1);
        assert_eq!(blob.relocs[0].kind, RelocKind::InlineCache);
        assert_eq!(blob.relocs[0].offset, site);
    }

    /// `call_far` loads the pool word into the scratch register and calls
    /// through it — the pool word stays relocatable.
    #[test]
    fn call_far_goes_through_the_pool() {
        let mut a = X64Assembler::new();
        let lit = a.literal_u64(0x7FFF_1234_5678, Some(RelocKind::RuntimeAddr));
        a.call_far(lit);
        let blob = a.finish();
        let text = disasm(&blob.code[..blob.literal_off as usize], 0x2000);
        assert!(text[0].starts_with("mov r10,["), "pool load: {}", text[0]);
        assert_eq!(text[1], "call r10");
        assert_eq!(blob.relocs[0].kind, RelocKind::RuntimeAddr);
    }

    /// Indexed addressing (array element access) encodes with the SIB byte.
    #[test]
    fn indexed_memory_operand() {
        let mut a = X64Assembler::new();
        a.emit("mov", &[r64(RAX), mem_index(RBX, RCX, 8, 16)]);
        let blob = a.finish();
        let text = disasm(&blob.code[..blob.literal_off as usize], 0);
        assert_eq!(text[0], "mov rax,[rbx+rcx*8+10h]");
    }

    #[test]
    #[should_panic(expected = "unbound label")]
    fn unbound_label_panics() {
        let mut a = X64Assembler::new();
        let l = a.new_label();
        a.jmp(l);
        a.finish();
    }

    #[test]
    #[should_panic(expected = "finish called twice")]
    fn finish_twice_panics() {
        let mut a = X64Assembler::new();
        a.finish();
        a.finish();
    }

    /// Compiler-bug loudness: an unknown mnemonic is never a guest-visible
    /// condition (CONVENTIONS §4).
    #[test]
    #[should_panic(expected = "frobnicate")]
    fn bad_mnemonic_panics() {
        let mut a = X64Assembler::new();
        a.emit("frobnicate", &[]);
    }
}
