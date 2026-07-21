// Vendored from JASM (wfasm), https://github.com/albanread/JASM  commit f2177391538cbede0c8cfcaa29bd3303ae421d0c
// Original path: rust/src/rasm/assemble.rs.  License: MIT (see LICENSE-JASM in this
// directory; Copyright (c) 2026 alban read).
// Local modifications are marked with `// WINVM:` comments — keep the diff
// against upstream minimal so re-vendoring stays mechanical.
// WINVM: `crate::backend`/`crate::rasm` -> `crate::vendor::wfasm::*` (modules
// moved under WINVM's own tree).

//! Two-pass assembler driver: assembled text â†’ [`EncodedModule`].
//!
//! 1. Parse every line into an [`Item`] (code bytes / data / label / globl /
//!    align / relaxable branch).
//! 2. Branch relaxation: internal `jmp`/`jcc` start short (rel8) and grow to
//!    rel32 only when the displacement overflows i8 â€” iterated to a fixpoint
//!    (branches only grow, so it converges). This mirrors LLVM-MC's
//!    start-short/relax-on-overflow policy, for byte-identity. `call` is always
//!    rel32; branches to externs are always rel32.
//! 3. Emit: lay out final bytes, patch internal branch + RIP-rel displacements,
//!    and emit a [`Reloc`] for every reference to a symbol not defined here
//!    (host externs).

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use crate::vendor::wfasm::backend::{EncodedModule, Reloc, RelocKind};

use super::encode::{encode, FixupKind};
use super::parse::{parse_line, Directive, Line, Operand};

/// A relaxable branch/call.
struct Branch {
    /// Short-form opcode bytes (1-byte rel8 follows). `None` for `call`.
    short: Option<Vec<u8>>,
    /// Long-form opcode bytes (4-byte rel32 follows).
    long: Vec<u8>,
    target: String,
    /// Decided during relaxation.
    is_long: bool,
}

impl Branch {
    fn size(&self) -> usize {
        if self.is_long {
            self.long.len() + 4
        } else {
            self.short.as_ref().unwrap().len() + 1
        }
    }
}

enum Item {
    /// Fixed code/data bytes, with RIP-rel fixups (offset-within-bytes, target).
    Code { bytes: Vec<u8>, riprel: Vec<(usize, String)> },
    Label(String),
    Globl(String),
    /// `.align`/`.p2align` â€” pad to a 2^n boundary (n already normalized).
    AlignP2(u32),
    Branch(Branch),
}

impl Item {
    fn size_at(&self, off: usize) -> usize {
        match self {
            Item::Code { bytes, .. } => bytes.len(),
            Item::Label(_) | Item::Globl(_) => 0,
            Item::Branch(b) => b.size(),
            Item::AlignP2(n) => {
                let align = 1usize << *n;
                (align - (off % align)) % align
            }
        }
    }
}

/// Emit `count` bytes of alignment padding using the same canonical multi-byte
/// NOP encodings LLVM-MC's `X86AsmBackend::writeNopData` uses in a code section
/// â€” required for byte-identity (a run of `0x90` would diverge). Lengths 1..=10
/// come straight from the table; 11..=15 prepend `count-10` `0x66` operand-size
/// prefixes to the 10-byte form. Pads longer than the max single NOP (15) are
/// split into successive NOPs, largest first.
fn write_nop_padding(code: &mut Vec<u8>, count: usize) {
    // Canonical NOPs by length (index = len-1).
    const NOPS: [&[u8]; 10] = [
        &[0x90],
        &[0x66, 0x90],
        &[0x0F, 0x1F, 0x00],
        &[0x0F, 0x1F, 0x40, 0x00],
        &[0x0F, 0x1F, 0x44, 0x00, 0x00],
        &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
        &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
        &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
    ];
    const MAX_NOP: usize = 15; // x86-64 generic: NOPs up to 15 bytes (66-prefixed).
    let mut remaining = count;
    while remaining != 0 {
        let this = remaining.min(MAX_NOP);
        let prefixes = this.saturating_sub(10);
        for _ in 0..prefixes {
            code.push(0x66);
        }
        let rest = this - prefixes;
        if rest != 0 {
            code.extend_from_slice(NOPS[rest - 1]);
        }
        remaining -= this;
    }
}

fn branch_for(mnemonic: &str, target: &str) -> Option<Branch> {
    if mnemonic == "call" {
        return Some(Branch { short: None, long: vec![0xE8], target: target.to_string(), is_long: true });
    }
    if mnemonic == "jmp" {
        return Some(Branch { short: Some(vec![0xEB]), long: vec![0xE9], target: target.to_string(), is_long: false });
    }
    if let Some(cc) = super::encode::jcc_nibble(mnemonic) {
        return Some(Branch {
            short: Some(vec![0x70 | cc]),
            long: vec![0x0F, 0x80 | cc],
            target: target.to_string(),
            is_long: false,
        });
    }
    None
}

/// Assemble a whole module's worth of text into an [`EncodedModule`].
pub fn assemble(text: &str) -> Result<EncodedModule> {
    // â”€â”€ Pass 1: parse into items â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut items: Vec<Item> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        // MC allows `label: insn` / `label: .quad ...` on one line; peel any
        // leading label into its own Item, then parse the remainder.
        let clean = super::parse::strip_comment(raw);
        let (label, rest) = super::parse::split_leading_label(clean);
        if let Some(name) = label {
            items.push(Item::Label(name.to_string()));
            if rest.is_empty() {
                continue;
            }
        }
        let body = if label.is_some() { rest } else { clean };
        let line = parse_line(body).with_context(|| format!("line {}: `{raw}`", lineno + 1))?;
        match line {
            Line::Empty => {}
            Line::Label(name) => items.push(Item::Label(name)),
            Line::Directive(d) => push_directive(&mut items, d)?,
            Line::Insn { mnemonic, ops } => {
                // Relaxable branch/call to a symbol?
                if let [Operand::Sym(target)] = ops.as_slice() {
                    if let Some(br) = branch_for(&mnemonic, target) {
                        items.push(Item::Branch(br));
                        continue;
                    }
                }
                let enc = encode(&mnemonic, &ops)
                    .with_context(|| format!("line {}: encode `{raw}`", lineno + 1))?;
                let mut riprel = Vec::new();
                for f in &enc.fixups {
                    match f.kind {
                        FixupKind::RipRel32 => riprel.push((f.at, f.target.clone())),
                        FixupKind::Rel32 => {
                            // A non-branch rel32 fixup shouldn't occur here.
                            bail!("line {}: unexpected branch fixup in `{raw}`", lineno + 1);
                        }
                    }
                }
                items.push(Item::Code { bytes: enc.bytes, riprel });
            }
        }
    }

    // Which symbols are defined in this module (labels)?
    let defined: std::collections::HashSet<String> = items
        .iter()
        .filter_map(|it| match it {
            Item::Label(n) => Some(n.clone()),
            _ => None,
        })
        .collect();

    // â”€â”€ Pass 2: branch relaxation to a fixpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    loop {
        let (offsets, labels) = layout(&items);
        let mut changed = false;
        let mut off_iter = offsets.iter();
        for it in &mut items {
            let off = *off_iter.next().unwrap();
            if let Item::Branch(b) = it {
                if b.is_long {
                    continue;
                }
                // Extern target â†’ must be long.
                let must_long = match labels.get(&b.target) {
                    None => true,
                    Some(&tgt) => {
                        let after = off + b.size();
                        let disp = tgt as i64 - after as i64;
                        !(-128..=127).contains(&disp)
                    }
                };
                if must_long {
                    b.is_long = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // â”€â”€ Pass 3: emit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let (offsets, labels) = layout(&items);
    let mut code: Vec<u8> = Vec::new();
    let mut symbols: BTreeMap<String, usize> = BTreeMap::new();
    let mut relocs: Vec<Reloc> = Vec::new();
    let mut externs: Vec<String> = Vec::new();
    let mut globls: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut idx = 0usize;
    for it in &items {
        let off = offsets[idx];
        idx += 1;
        debug_assert_eq!(off, code.len());
        match it {
            Item::Label(n) => {
                symbols.insert(n.clone(), code.len());
            }
            Item::Globl(n) => {
                globls.insert(n.clone());
            }
            Item::AlignP2(n) => {
                let align = 1usize << *n;
                let pad = (align - (code.len() % align)) % align;
                write_nop_padding(&mut code, pad);
            }
            Item::Code { bytes, riprel } => {
                let base = code.len();
                code.extend_from_slice(bytes);
                for (at, target) in riprel {
                    let field = base + at;
                    if let Some(&tgt) = labels.get(target) {
                        let disp = tgt as i64 - (field as i64 + 4);
                        let d = i32::try_from(disp).context("RIP-rel disp32 overflow")?;
                        code[field..field + 4].copy_from_slice(&d.to_le_bytes());
                    } else {
                        relocs.push(Reloc { at: field, size: 4, kind: RelocKind::RipRel32, target: target.clone(), addend: 0 });
                        externs.push(target.clone());
                    }
                }
            }
            Item::Branch(b) => {
                let base = code.len();
                if b.is_long {
                    code.extend_from_slice(&b.long);
                    let field = code.len();
                    code.extend_from_slice(&[0, 0, 0, 0]);
                    if let Some(&tgt) = labels.get(&b.target) {
                        let disp = tgt as i64 - (field as i64 + 4);
                        let d = i32::try_from(disp).context("branch rel32 overflow")?;
                        code[field..field + 4].copy_from_slice(&d.to_le_bytes());
                    } else {
                        relocs.push(Reloc { at: field, size: 4, kind: RelocKind::BranchRel32, target: b.target.clone(), addend: 0 });
                        externs.push(b.target.clone());
                    }
                } else {
                    code.extend_from_slice(b.short.as_ref().unwrap());
                    let field = code.len();
                    code.push(0);
                    let tgt = *labels.get(&b.target).expect("short branch to extern impossible");
                    let disp = tgt as i64 - (field as i64 + 1);
                    code[field] = i8::try_from(disp).context("branch rel8 overflow")? as u8;
                }
                let _ = base;
            }
        }
    }

    // Only export symbols that were .globl'd (others are module-local labels).
    symbols.retain(|name, _| globls.contains(name));
    externs.sort();
    externs.dedup();

    Ok(EncodedModule { code, symbols, relocs, externs })
}

/// Compute the byte offset of each item and the offset of every label.
fn layout(items: &[Item]) -> (Vec<usize>, BTreeMap<String, usize>) {
    let mut offsets = Vec::with_capacity(items.len());
    let mut labels = BTreeMap::new();
    let mut off = 0usize;
    for it in items {
        offsets.push(off);
        if let Item::Label(n) = it {
            labels.insert(n.clone(), off);
        }
        off += it.size_at(off);
    }
    (offsets, labels)
}

fn push_directive(items: &mut Vec<Item>, d: Directive) -> Result<()> {
    match d {
        Directive::IntelSyntax | Directive::Text | Directive::Other(_) => {}
        Directive::Globl(n) => items.push(Item::Globl(n)),
        Directive::Quad(vs) => items.push(Item::Code {
            bytes: vs.iter().flat_map(|v| v.to_le_bytes()).collect(),
            riprel: vec![],
        }),
        Directive::Byte(b) => items.push(Item::Code { bytes: vec![b], riprel: vec![] }),
        Directive::Zero(n) => items.push(Item::Code { bytes: vec![0u8; n], riprel: vec![] }),
        Directive::Ascii(bytes, nul) => {
            let mut v = bytes;
            if nul {
                v.push(0);
            }
            items.push(Item::Code { bytes: v, riprel: vec![] });
        }
        Directive::Align(bytes) => {
            // byte alignment -> log2 (must be power of two)
            if !bytes.is_power_of_two() {
                bail!(".align {bytes} is not a power of two");
            }
            items.push(Item::AlignP2(bytes.trailing_zeros()));
        }
        Directive::P2align(n) => items.push(Item::AlignP2(n)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_call_and_short_branch_resolve() {
        // Two procs; an internal call (resolved, no reloc) and a short jcc.
        let src = "\
.text
.globl helper
helper:
mov rax, rcx
ret
.globl entry
entry:
test rax, rax
jz entry$$skip
call helper
entry$$skip:
ret
";
        let m = assemble(src).unwrap();
        // Both globls exported.
        assert!(m.symbols.contains_key("helper") && m.symbols.contains_key("entry"));
        // entry$$skip is local â€” not exported.
        assert!(!m.symbols.contains_key("entry$$skip"));
        // helper is internal â†’ the `call helper` is resolved, NOT a reloc.
        assert!(m.relocs.is_empty(), "internal targets must not produce relocs: {:?}", m.relocs);
        assert!(m.externs.is_empty());
        // jz short form (74) present.
        assert!(m.code.windows(1).any(|w| w == [0x74]), "expected short jz (74) in {:02x?}", m.code);
    }

    #[test]
    fn extern_call_becomes_reloc() {
        let src = "\
.text
.globl w
w:
call rt_emit
ret
";
        let m = assemble(src).unwrap();
        assert_eq!(m.externs, vec!["rt_emit".to_string()]);
        assert_eq!(m.relocs.len(), 1);
        assert_eq!(m.relocs[0].kind, RelocKind::BranchRel32);
        assert_eq!(m.relocs[0].target, "rt_emit");
        // call is E8 rel32 (5 bytes) + ret.
        assert_eq!(m.code[0], 0xE8);
    }

    #[test]
    fn far_branch_relaxes_to_rel32() {
        // A forward jz over >127 bytes of filler must relax to the rel32 form.
        let mut src = String::from(".text\n.globl w\nw:\njz w$$end\n");
        for _ in 0..45 {
            src.push_str("mov rax, rcx\n"); // 3 bytes each = 135 bytes, well past rel8
        }
        src.push_str("w$$end:\nret\n");
        let m = assemble(&src).unwrap();
        // Long jcc form: 0F 8x.
        assert_eq!(&m.code[0..2], &[0x0F, 0x84], "expected near jz (0F 84): {:02x?}", &m.code[0..4]);
    }

    #[test]
    fn rip_relative_internal_and_extern() {
        let src = "\
.text
.globl w
w:
lea r8, [rip + helper]
lea r9, [rip + rt_emit]
ret
.globl helper
helper:
ret
";
        let m = assemble(src).unwrap();
        // helper is internal â†’ patched; rt_emit â†’ reloc.
        assert_eq!(m.relocs.len(), 1);
        assert_eq!(m.relocs[0].kind, RelocKind::RipRel32);
        assert_eq!(m.relocs[0].target, "rt_emit");
    }
}
