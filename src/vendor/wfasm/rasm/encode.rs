// Vendored from JASM (wfasm), https://github.com/albanread/JASM  commit f2177391538cbede0c8cfcaa29bd3303ae421d0c
// Original path: rust/src/rasm/encode.rs.  License: MIT (see LICENSE-JASM in this
// directory; Copyright (c) 2026 alban read).
// Local modifications are marked with `// WINVM:` comments — keep the diff
// against upstream minimal so re-vendoring stays mechanical.
// WINVM: `crate::backend`/`crate::rasm` -> `crate::vendor::wfasm::*` (modules
// moved under WINVM's own tree).

//! Encode one parsed [`Insn`](super::Line) into machine-code bytes + fixups.
//!
//! The encoding machinery (REX / ModRM / SIB / displacement / immediate) is the
//! reusable core; per-mnemonic logic sits on top. Branch/call/RIP-rel targets
//! become [`Fixup`]s the two-pass driver later resolves (internal label →
//! patch, extern → `Reloc`).
//!
//! Form choices (e.g. ALU `r/m,r` vs `r,r/m`, disp8 vs disp32) are chosen to
//! match LLVM-MC so output is byte-identical; the golden differential gates that.

use anyhow::{bail, Result};

use super::parse::{Mem, MemSize, Operand, Reg, RegClass};

/// A symbolic reference left in the encoded bytes for the driver to resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixup {
    /// Offset of the field within this instruction's bytes.
    pub at: usize,
    pub kind: FixupKind,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// 4-byte branch displacement (call/jmp/jcc rel32).
    Rel32,
    /// 4-byte RIP-relative displacement (`lea`/SSE `[rip+sym]`).
    RipRel32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<Fixup>,
}

impl Encoded {
    fn b(&mut self, byte: u8) {
        self.bytes.push(byte);
    }
    fn ext(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

fn rex_byte(w: bool, r: bool, x: bool, b: bool) -> Option<u8> {
    if w || r || x || b {
        Some(0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8))
    } else {
        None
    }
}

fn is64(r: Reg) -> bool {
    r.class == RegClass::R64
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpSize {
    B8,
    B16,
    B32,
    B64,
}

fn reg_size(r: Reg) -> OpSize {
    match r.class {
        RegClass::R8 => OpSize::B8,
        RegClass::R16 => OpSize::B16,
        RegClass::R32 => OpSize::B32,
        RegClass::R64 | RegClass::Xmm | RegClass::Ymm | RegClass::Zmm => OpSize::B64,
    }
}

fn mem_opsize(m: &Mem) -> Option<OpSize> {
    m.size.map(|s| match s {
        MemSize::Byte => OpSize::B8,
        MemSize::Word => OpSize::B16,
        MemSize::Dword => OpSize::B32,
        MemSize::Qword => OpSize::B64,
        // xmmword is only ever an SSE operand; SSE encoders ignore mem_opsize.
        MemSize::Xmmword => OpSize::B64,
    })
}

/// Operand size of a two-operand integer instruction: a register operand wins;
/// else the sized memory operand; else default 64-bit.
fn two_op_size(a: &Operand, b: &Operand) -> OpSize {
    for op in [a, b] {
        if let Operand::Reg(r) = op {
            return reg_size(*r);
        }
    }
    for op in [a, b] {
        if let Operand::Mem(m) = op {
            if let Some(s) = mem_opsize(m) {
                return s;
            }
        }
    }
    OpSize::B64
}

/// 66 operand-size prefix for 16-bit (emitted before REX).
fn size_mandatory(s: OpSize) -> &'static [u8] {
    if s == OpSize::B16 {
        &[0x66]
    } else {
        &[]
    }
}

fn size_rexw(s: OpSize) -> bool {
    s == OpSize::B64
}

/// 8-bit forms use opcode-1 for the standard two-operand /r encodings.
fn op8(op: u8, s: OpSize) -> u8 {
    if s == OpSize::B8 {
        op - 1
    } else {
        op
    }
}

/// Append a sign/size-appropriate immediate for an operand of size `s`.
fn push_imm_sized(e: &mut Encoded, v: i64, s: OpSize) {
    match s {
        OpSize::B8 => e.b(v as u8),
        OpSize::B16 => e.ext(&(v as i16).to_le_bytes()),
        _ => e.ext(&(v as i32).to_le_bytes()),
    }
}

/// Emit `mandatory` prefixes, REX (if needed), `opcode`, then a ModRM for a
/// register r/m (`mod = 11`). `reg_field` and `rm` are full 0..15 numbers.
fn emit_reg_rm(e: &mut Encoded, rex_w: bool, mandatory: &[u8], opcode: &[u8], reg_field: u8, rm: u8) {
    e.ext(mandatory);
    if let Some(r) = rex_byte(rex_w, reg_field >= 8, false, rm >= 8) {
        e.b(r);
    }
    e.ext(opcode);
    e.b(0xC0 | ((reg_field & 7) << 3) | (rm & 7));
}

/// High bits (bit 3) of a memory operand's index and base — for REX.X/REX.B or
/// the (inverted) VEX.X̄/VEX.B̄. RIP-relative and absent index/base yield false.
fn mem_xb(mem: &Mem) -> (bool, bool) {
    if mem.rip_sym.is_some() {
        return (false, false);
    }
    (
        mem.index.map(|r| r.num >= 8).unwrap_or(false),
        mem.base.map(|r| r.num >= 8).unwrap_or(false),
    )
}

/// Emit ModRM for a register r/m (`mod=11`). No prefix/REX/opcode — for VEX and
/// legacy paths that emit their own prefix bytes.
fn emit_modrm_reg(e: &mut Encoded, reg_field: u8, rm: u8) {
    e.b(0xC0 | ((reg_field & 7) << 3) | (rm & 7));
}

/// Emit ModRM (+SIB +disp) for a memory r/m with `reg_field` in the reg slot.
/// No prefix/REX/opcode. Records a RIP-rel fixup when `mem.rip_sym` is set.
/// Matches MC's disp sizing.
fn emit_modrm_mem(e: &mut Encoded, reg_field: u8, mem: &Mem) -> Result<()> {
    // RIP-relative: mod=00, rm=101, disp32 (fixup).
    if let Some(sym) = &mem.rip_sym {
        e.b(0x00 | ((reg_field & 7) << 3) | 0b101);
        let at = e.len();
        e.ext(&[0, 0, 0, 0]);
        e.fixups.push(Fixup { at, kind: FixupKind::RipRel32, target: sym.clone() });
        return Ok(());
    }

    let base = mem.base;
    let index = mem.index;
    let reg3 = (reg_field & 7) << 3;

    // Decide mod + whether a SIB is needed.
    let needs_sib = index.is_some() || matches!(base.map(|b| b.num & 7), Some(0b100)); // rsp/r12 base

    let base_low3 = base.map(|b| b.num & 7);
    // [rbp]/[r13] (low3==5) base with no disp can't use mod=00 (that encodes
    // rip/disp32, or no-base in a SIB) — force disp8=0. This applies whether or
    // not a SIB is present, e.g. `[rbp + rax*8]` -> mod=01.
    let force_disp8 = matches!(base_low3, Some(0b101)) && mem.disp == 0;

    let md: u8 = if base.is_none() {
        // [disp32] / [index*scale + disp32] — mod=00 with SIB.base=101 form.
        0b00
    } else if force_disp8 {
        0b01
    } else if mem.disp == 0 {
        0b00
    } else if (-128..=127).contains(&mem.disp) {
        0b01
    } else {
        0b10
    };

    if needs_sib {
        let rm = 0b100u8; // SIB follows
        e.b((md << 6) | reg3 | rm);
        let scale_bits = match mem.scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            other => bail!("bad scale {other}"),
        };
        let index_bits = match index {
            Some(r) => r.num & 7,
            None => 0b100, // no index
        };
        let base_bits = match base_low3 {
            Some(b) => b,
            None => 0b101, // no base (disp32)
        };
        e.b((scale_bits << 6) | (index_bits << 3) | base_bits);
    } else {
        let rm = base_low3.unwrap_or(0b101);
        e.b((md << 6) | reg3 | rm);
    }

    // Displacement.
    match md {
        0b01 => e.b(mem.disp as i8 as u8),
        0b10 => e.ext(&(mem.disp as i32).to_le_bytes()),
        0b00 if base.is_none() => e.ext(&(mem.disp as i32).to_le_bytes()),
        _ => {}
    }
    Ok(())
}

/// Emit `mandatory`, REX, `opcode`, then ModRM+SIB+disp for a memory r/m.
fn emit_mem_rm(
    e: &mut Encoded,
    rex_w: bool,
    mandatory: &[u8],
    opcode: &[u8],
    reg_field: u8,
    mem: &Mem,
) -> Result<()> {
    e.ext(mandatory);
    let (rex_x, rex_b) = mem_xb(mem);
    if let Some(r) = rex_byte(rex_w, reg_field >= 8, rex_x, rex_b) {
        e.b(r);
    }
    e.ext(opcode);
    emit_modrm_mem(e, reg_field, mem)
}

/// Emit a register or memory r/m with the given reg field.
fn emit_rm(
    e: &mut Encoded,
    rex_w: bool,
    mandatory: &[u8],
    opcode: &[u8],
    reg_field: u8,
    rm: &Operand,
) -> Result<()> {
    match rm {
        Operand::Reg(r) => {
            emit_reg_rm(e, rex_w, mandatory, opcode, reg_field, r.num);
            Ok(())
        }
        Operand::Mem(m) => emit_mem_rm(e, rex_w, mandatory, opcode, reg_field, m),
        other => bail!("expected reg/mem r/m, got {other:?}"),
    }
}

// ── group-1 ALU (add/or/adc/sbb/and/sub/xor/cmp) ────────────────────────────
// reg/reg + mem,reg use the `r/m, r` form (opcode base 0x01); reg,mem uses the
// `r, r/m` form (base 0x03); r/m,imm uses the 0x81/0x83 group with /digit.

struct Alu {
    /// /digit for the 0x81/0x83 imm group.
    ext: u8,
    /// base opcode for the `r/m, r` form (0x01 family).
    rm_r: u8,
    /// accumulator-immediate short opcode (AL form, e.g. `cmp al`=0x3C); the
    /// AX/EAX/RAX form is `acc8 + 1`.
    acc8: u8,
}

fn alu(mnem: &str) -> Option<Alu> {
    Some(match mnem {
        "add" => Alu { ext: 0, rm_r: 0x01, acc8: 0x04 },
        "or" => Alu { ext: 1, rm_r: 0x09, acc8: 0x0C },
        "adc" => Alu { ext: 2, rm_r: 0x11, acc8: 0x14 },
        "sbb" => Alu { ext: 3, rm_r: 0x19, acc8: 0x1C },
        "and" => Alu { ext: 4, rm_r: 0x21, acc8: 0x24 },
        "sub" => Alu { ext: 5, rm_r: 0x29, acc8: 0x2C },
        "xor" => Alu { ext: 6, rm_r: 0x31, acc8: 0x34 },
        "cmp" => Alu { ext: 7, rm_r: 0x39, acc8: 0x3C },
        _ => return None,
    })
}

/// Encode a single instruction. `ops` are the parsed operands.
/// Operand width of a single r/m operand (register width, or a memory operand's
/// explicit size; defaults to 64-bit).
fn rm_size(op: &Operand) -> OpSize {
    match op {
        Operand::Reg(r) => reg_size(*r),
        Operand::Mem(m) => mem_opsize(m).unwrap_or(OpSize::B64),
        _ => OpSize::B64,
    }
}

/// Miscellaneous integer instructions outside the ALU/shift/unary groups:
/// bit tests, `bswap`, `cmpxchg`, `movbe`, `endbr64`, bare (non-`rep`) string
/// ops, `test r/m,imm`, and `push r/m`/`push imm`.
fn try_int_misc(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            ("endbr64", []) => e.ext(&[0xF3, 0x0F, 0x1E, 0xFA]),
            // bare string ops (no `rep`): reuse the rep table's opcode/REX.W.
            (m, []) if string_op(m).is_some() => {
                let (op, w) = string_op(m).unwrap();
                if w {
                    e.b(0x48);
                }
                e.b(op);
            }
            // bswap r32/r64 : [REX] 0F C8+rd
            ("bswap", [Operand::Reg(rg)])
                if rg.class == RegClass::R32 || rg.class == RegClass::R64 =>
            {
                if let Some(rex) = rex_byte(rg.class == RegClass::R64, false, false, rg.num >= 8) {
                    e.b(rex);
                }
                e.b(0x0F);
                e.b(0xC8 + (rg.num & 7));
            }
            // bt/bts/btr/btc r/m, r : 0F A3/AB/B3/BB /r (reg field = bit index)
            ("bt" | "bts" | "btr" | "btc", [rm, Operand::Reg(s)]) => {
                let op = match mnemonic {
                    "bt" => 0xA3u8,
                    "bts" => 0xAB,
                    "btr" => 0xB3,
                    "btc" => 0xBB,
                    _ => unreachable!(),
                };
                let size = two_op_size(rm, &Operand::Reg(*s));
                emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[0x0F, op], s.num, rm)?;
            }
            // bt/bts/btr/btc r/m, imm8 : 0F BA /ext ib
            ("bt" | "bts" | "btr" | "btc", [rm, Operand::Imm(v)]) => {
                let ext = match mnemonic {
                    "bt" => 4u8,
                    "bts" => 5,
                    "btr" => 6,
                    "btc" => 7,
                    _ => unreachable!(),
                };
                let size = rm_size(rm);
                emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[0x0F, 0xBA], ext, rm)?;
                e.b(*v as u8);
            }
            // cmpxchg r/m, r : 0F B0 (8-bit) / 0F B1 /r
            ("cmpxchg", [rm, Operand::Reg(s)]) => {
                let size = two_op_size(rm, &Operand::Reg(*s));
                let op = if size == OpSize::B8 { 0xB0 } else { 0xB1 };
                emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[0x0F, op], s.num, rm)?;
            }
            // movbe r, m : 0F 38 F0 ; movbe m, r : 0F 38 F1
            ("movbe", [Operand::Reg(d), Operand::Mem(m)]) => {
                let size = reg_size(*d);
                emit_mem_rm(&mut e, size_rexw(size), size_mandatory(size), &[0x0F, 0x38, 0xF0], d.num, m)?;
            }
            ("movbe", [Operand::Mem(m), Operand::Reg(s)]) => {
                let size = reg_size(*s);
                emit_mem_rm(&mut e, size_rexw(size), size_mandatory(size), &[0x0F, 0x38, 0xF1], s.num, m)?;
            }
            // push r/m64 : FF /6 (memory; `push reg64` is handled by the main match)
            ("push", [Operand::Mem(m)]) => {
                emit_mem_rm(&mut e, false, &[], &[0xFF], 6, m)?;
            }
            // push imm : 6A ib (imm8) / 68 id
            ("push", [Operand::Imm(v)]) => {
                if (-128..=127).contains(v) {
                    e.b(0x6A);
                    e.b(*v as i8 as u8);
                } else {
                    e.b(0x68);
                    e.ext(&(*v as i32).to_le_bytes());
                }
            }
            // test r/m, imm : accumulator short A8/A9, else F6/F7 /0
            ("test", [rm, Operand::Imm(v)]) => {
                let size = rm_size(rm);
                if matches!(rm, Operand::Reg(r) if r.num == 0) {
                    e.ext(size_mandatory(size));
                    if size_rexw(size) {
                        e.b(0x48);
                    }
                    e.b(if size == OpSize::B8 { 0xA8 } else { 0xA9 });
                } else {
                    let op = if size == OpSize::B8 { 0xF6 } else { 0xF7 };
                    emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[op], 0, rm)?;
                }
                push_imm_sized(&mut e, *v, size);
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

pub fn encode(mnemonic: &str, ops: &[Operand]) -> Result<Encoded> {
    if let Some(r) = try_vex(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_sse(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_unary(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_shift(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_cc(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_misc(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_int_misc(mnemonic, ops) {
        return r;
    }
    let mut e = Encoded::default();
    match (mnemonic, ops) {
        ("ret", []) => e.b(0xC3),
        ("nop", []) => e.b(0x90),
        ("cqo", []) => {
            e.b(0x48);
            e.b(0x99);
        }
        ("leave", []) => e.b(0xC9),
        ("std", []) => e.b(0xFD),
        ("cld", []) => e.b(0xFC),
        ("clc", []) => e.b(0xF8),
        ("stc", []) => e.b(0xF9),
        ("cmc", []) => e.b(0xF5),
        ("sahf", []) => e.b(0x9E),
        ("lahf", []) => e.b(0x9F),
        ("int3", []) => e.b(0xCC),
        ("int", [Operand::Imm(3)]) => e.b(0xCC),
        ("int", [Operand::Imm(n)]) => {
            e.b(0xCD);
            e.b(*n as u8);
        }
        ("syscall", []) => e.ext(&[0x0F, 0x05]),
        ("cpuid", []) => e.ext(&[0x0F, 0xA2]),
        ("rdtsc", []) => e.ext(&[0x0F, 0x31]),
        ("cdqe", []) => e.ext(&[0x48, 0x98]),
        ("cwde", []) => e.b(0x98),
        ("cdq", []) => e.b(0x99),
        ("pause", []) => e.ext(&[0xF3, 0x90]),

        // mov
        ("mov", [dst, src]) => encode_mov(&mut e, dst, src)?,
        // movabs r64, imm64 — always the REX.W B8+r imm64 form, even for a
        // small immediate (that is the whole point of `movabs`; plain `mov`
        // would shrink it to a sign-extended imm32). LET bakes libm addresses
        // (>2^32) this way.
        ("movabs", [Operand::Reg(d), Operand::Imm(v)]) if d.class == RegClass::R64 => {
            if let Some(r) = rex_byte(true, false, false, d.num >= 8) {
                e.b(r);
            }
            e.b(0xB8 + (d.num & 7));
            e.ext(&(*v as u64).to_le_bytes());
        }
        // lea reg, mem
        ("lea", [Operand::Reg(d), Operand::Mem(m)]) => {
            emit_mem_rm(&mut e, is64(*d), &[], &[0x8D], d.num, m)?;
        }
        // push/pop reg64
        ("push", [Operand::Reg(r)]) if r.class == RegClass::R64 => push_pop(&mut e, 0x50, r.num),
        ("pop", [Operand::Reg(r)]) if r.class == RegClass::R64 => push_pop(&mut e, 0x58, r.num),

        // call/jmp/jcc target
        ("call", [Operand::Sym(s)]) => {
            e.b(0xE8);
            rel32_fixup(&mut e, s);
        }
        ("jmp", [Operand::Sym(s)]) => {
            e.b(0xE9);
            rel32_fixup(&mut e, s);
        }
        // Indirect jmp/call through r/m64 : FF /4 (jmp), FF /2 (call).
        ("jmp", [rm @ (Operand::Reg(_) | Operand::Mem(_))]) => {
            emit_rm(&mut e, false, &[], &[0xFF], 4, rm)?;
        }
        ("call", [rm @ (Operand::Reg(_) | Operand::Mem(_))]) => {
            emit_rm(&mut e, false, &[], &[0xFF], 2, rm)?;
        }
        (m, [Operand::Sym(s)]) if jcc_code(m).is_some() => {
            e.b(0x0F);
            e.b(0x80 | jcc_code(m).unwrap());
            rel32_fixup(&mut e, s);
        }

        // group-1 ALU
        (m, [dst, src]) if alu(m).is_some() => encode_alu(&mut e, alu(m).unwrap(), dst, src)?,

        // test r/m, r : 85 /r (84 for 8-bit)
        ("test", [rm, Operand::Reg(r)]) => {
            let size = two_op_size(rm, &Operand::Reg(*r));
            emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[op8(0x85, size)], r.num, rm)?;
        }

        _ => bail!("rasm: unsupported instruction `{mnemonic}` with {} operand(s)", ops.len()),
    }
    Ok(e)
}

fn push_pop(e: &mut Encoded, base: u8, num: u8) {
    if num >= 8 {
        e.b(0x41); // REX.B
    }
    e.b(base + (num & 7));
}

fn rel32_fixup(e: &mut Encoded, sym: &str) {
    let at = e.len();
    e.ext(&[0, 0, 0, 0]);
    e.fixups.push(Fixup { at, kind: FixupKind::Rel32, target: sym.to_string() });
}

/// Condition-code nibble for a bare condition suffix (shared by jcc/setcc/cmovcc).
fn cc_code(cc: &str) -> Option<u8> {
    Some(match cc {
        "o" => 0x0,
        "no" => 0x1,
        "b" | "c" | "nae" => 0x2,
        "ae" | "nb" | "nc" => 0x3,
        "e" | "z" => 0x4,
        "ne" | "nz" => 0x5,
        "be" | "na" => 0x6,
        "a" | "nbe" => 0x7,
        "s" => 0x8,
        "ns" => 0x9,
        "p" | "pe" => 0xA,
        "np" | "po" => 0xB,
        "l" | "nge" => 0xC,
        "ge" | "nl" => 0xD,
        "le" | "ng" => 0xE,
        "g" | "nle" => 0xF,
        _ => return None,
    })
}

fn jcc_code(m: &str) -> Option<u8> {
    m.strip_prefix('j').and_then(cc_code)
}

/// The condition nibble for a `jcc` mnemonic (`jz`→4, …) — used by the
/// two-pass driver to build the short/long branch forms. `None` for `jmp`.
pub(crate) fn jcc_nibble(m: &str) -> Option<u8> {
    jcc_code(m)
}

fn src_size_word(src: &Operand) -> Option<bool> {
    match src {
        Operand::Mem(m) => match m.size {
            Some(MemSize::Byte) => Some(false),
            Some(MemSize::Word) => Some(true),
            _ => None,
        },
        Operand::Reg(r) => match r.class {
            RegClass::R8 => Some(false),
            RegClass::R16 => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// `(opcode, needs REX.W)` for a `rep`-prefixed string op.
fn string_op(s: &str) -> Option<(u8, bool)> {
    Some(match s {
        "movsb" => (0xA4, false),
        "movsq" => (0xA5, true),
        "stosb" => (0xAA, false),
        "stosq" => (0xAB, true),
        "cmpsb" => (0xA6, false),
        "cmpsq" => (0xA7, true),
        "scasb" => (0xAE, false),
        "scasq" => (0xAF, true),
        "lodsb" => (0xAC, false),
        "lodsq" => (0xAD, true),
        _ => return None,
    })
}

/// setcc r/m8 (0F 90+cc /0) and cmovcc r, r/m (0F 40+cc /r).
fn try_cc(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    if let Some(cc) = mnemonic.strip_prefix("set").and_then(cc_code) {
        let [rm] = ops else {
            return Some(Err(anyhow::anyhow!("setcc needs 1 operand")));
        };
        return Some(emit_rm(&mut e, false, &[], &[0x0F, 0x90 | cc], 0, rm).map(|()| e));
    }
    if let Some(cc) = mnemonic.strip_prefix("cmov").and_then(cc_code) {
        if let [Operand::Reg(d), src] = ops {
            return Some(emit_rm(&mut e, is64(*d), &[], &[0x0F, 0x40 | cc], d.num, src).map(|()| e));
        }
    }
    None
}

/// movzx/movsx/movsxd, 2-operand imul, xchg, xadd, and rep-string ops.
fn try_misc(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            ("movzx" | "movsx", [Operand::Reg(d), src]) => {
                let word = src_size_word(src)
                    .ok_or_else(|| anyhow::anyhow!("movzx/movsx needs a sized source: {src:?}"))?;
                let op = match (mnemonic, word) {
                    ("movzx", false) => 0xB6,
                    ("movzx", true) => 0xB7,
                    ("movsx", false) => 0xBE,
                    ("movsx", true) => 0xBF,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, is64(*d), &[], &[0x0F, op], d.num, src)?;
            }
            ("movsxd", [Operand::Reg(d), src]) => {
                // movsxd r64, r/m32 : REX.W 63 /r
                emit_rm(&mut e, true, &[], &[0x63], d.num, src)?;
            }
            ("imul", [Operand::Reg(d), src]) => {
                // 2-operand imul r, r/m : 0F AF /r
                emit_rm(&mut e, is64(*d), &[], &[0x0F, 0xAF], d.num, src)?;
            }
            ("imul", [Operand::Reg(d), src, Operand::Imm(v)]) => {
                // 3-operand imul r, r/m, imm : 6B /r ib (imm8) or 69 /r id (imm32)
                if (-128..=127).contains(v) {
                    emit_rm(&mut e, is64(*d), &[], &[0x6B], d.num, src)?;
                    e.b(*v as i8 as u8);
                } else {
                    emit_rm(&mut e, is64(*d), &[], &[0x69], d.num, src)?;
                    e.ext(&(*v as i32).to_le_bytes());
                }
            }
            ("popcnt", [Operand::Reg(d), src]) => {
                emit_rm(&mut e, is64(*d), &[0xF3], &[0x0F, 0xB8], d.num, src)?;
            }
            ("lzcnt", [Operand::Reg(d), src]) => {
                emit_rm(&mut e, is64(*d), &[0xF3], &[0x0F, 0xBD], d.num, src)?;
            }
            ("tzcnt", [Operand::Reg(d), src]) => {
                emit_rm(&mut e, is64(*d), &[0xF3], &[0x0F, 0xBC], d.num, src)?;
            }
            ("bsr", [Operand::Reg(d), src]) => {
                emit_rm(&mut e, is64(*d), &[], &[0x0F, 0xBD], d.num, src)?;
            }
            ("bsf", [Operand::Reg(d), src]) => {
                emit_rm(&mut e, is64(*d), &[], &[0x0F, 0xBC], d.num, src)?;
            }
            // xchg rAX, r / r, rAX (two regs, width 16/32/64): MC uses the
            // accumulator short form `90+rd`, not `87 /r`. `xchg <acc>,<acc>`
            // collapses to the width nop with NO REX.W (matches MC: `xchg rax,rax`
            // -> `90`, not `48 90`).
            ("xchg", [Operand::Reg(d), Operand::Reg(s)]) => {
                let size = reg_size(*d);
                let acc_d = d.num == 0 && size != OpSize::B8;
                let acc_s = s.num == 0 && size != OpSize::B8;
                if acc_d || acc_s {
                    e.ext(size_mandatory(size)); // 66 for 16-bit
                    if acc_d && acc_s {
                        e.b(0x90);
                    } else {
                        let other = if acc_d { *s } else { *d };
                        if let Some(rex) = rex_byte(size_rexw(size), false, false, other.num >= 8) {
                            e.b(rex);
                        }
                        e.b(0x90 + (other.num & 7));
                    }
                } else {
                    emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[op8(0x87, size)], d.num, &Operand::Reg(*s))?;
                }
            }
            // xchg r, m : 87 /r (the FIRST operand is the reg field; a memory
            // first operand is handled by the `Mem, Reg` arm below).
            ("xchg", [Operand::Reg(d), src]) => {
                let size = two_op_size(&Operand::Reg(*d), src);
                emit_rm(&mut e, size_rexw(size), size_mandatory(size), &[op8(0x87, size)], d.num, src)?;
            }
            ("xchg", [Operand::Mem(mm), Operand::Reg(s)]) => {
                let size = reg_size(*s);
                emit_mem_rm(&mut e, size_rexw(size), size_mandatory(size), &[op8(0x87, size)], s.num, mm)?;
            }
            ("xadd", [dst, Operand::Reg(s)]) => {
                // xadd r/m, r : 0F C1 /r
                emit_rm(&mut e, is64(*s), &[], &[0x0F, 0xC1], s.num, dst)?;
            }
            ("rep" | "repe" | "repz" | "repne" | "repnz", [Operand::Sym(strop)]) => {
                let pfx = if mnemonic.starts_with("repn") { 0xF2u8 } else { 0xF3 };
                let (opc, w) =
                    string_op(strop).ok_or_else(|| anyhow::anyhow!("unknown string op `{strop}`"))?;
                e.b(pfx);
                if w {
                    e.b(0x48); // REX.W (after the F3/F2 prefix, before the opcode)
                }
                e.b(opc);
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

fn encode_mov(e: &mut Encoded, dst: &Operand, src: &Operand) -> Result<()> {
    let size = two_op_size(dst, src);
    let mand = size_mandatory(size);
    let w = size_rexw(size);
    match (dst, src) {
        // mov r/m, r : 89 /r (88 for 8-bit)
        (rm, Operand::Reg(r)) => emit_rm(e, w, mand, &[op8(0x89, size)], r.num, rm),
        // mov r, r/m (mem) : 8B /r (8A for 8-bit)
        (Operand::Reg(d), Operand::Mem(m)) => emit_mem_rm(e, w, mand, &[op8(0x8B, size)], d.num, m),
        // mov reg, imm — size-specific
        (Operand::Reg(d), Operand::Imm(v)) => {
            match size {
                OpSize::B64 => {
                    if i32::try_from(*v).is_ok() {
                        emit_reg_rm(e, true, &[], &[0xC7], 0, d.num); // C7 /0 id
                        e.ext(&(*v as i32).to_le_bytes());
                    } else {
                        if let Some(r) = rex_byte(true, false, false, d.num >= 8) {
                            e.b(r);
                        }
                        e.b(0xB8 + (d.num & 7)); // movabs r64, imm64
                        e.ext(&(*v as u64).to_le_bytes());
                    }
                }
                OpSize::B32 => {
                    if d.num >= 8 {
                        e.b(0x41);
                    }
                    e.b(0xB8 + (d.num & 7)); // B8+r id
                    e.ext(&(*v as i32 as u32).to_le_bytes());
                }
                OpSize::B16 => {
                    e.b(0x66);
                    if d.num >= 8 {
                        e.b(0x41);
                    }
                    e.b(0xB8 + (d.num & 7)); // 66 B8+r iw
                    e.ext(&(*v as i16).to_le_bytes());
                }
                OpSize::B8 => {
                    if d.num >= 8 {
                        e.b(0x41);
                    }
                    e.b(0xB0 + (d.num & 7)); // B0+r ib
                    e.b(*v as u8);
                }
            }
            Ok(())
        }
        // mov m, imm : C7 /0 id (C6 /0 ib for 8-bit)
        (Operand::Mem(m), Operand::Imm(v)) => {
            emit_mem_rm(e, w, mand, &[op8(0xC7, size)], 0, m)?;
            push_imm_sized(e, *v, size);
            Ok(())
        }
        _ => bail!("rasm: unsupported mov form {dst:?} <- {src:?}"),
    }
}

fn mem_is_qword(m: &Mem) -> bool {
    matches!(m.size, Some(MemSize::Qword) | None)
}

fn encode_alu(e: &mut Encoded, a: Alu, dst: &Operand, src: &Operand) -> Result<()> {
    let size = two_op_size(dst, src);
    let mand = size_mandatory(size);
    let w = size_rexw(size);
    match (dst, src) {
        // r/m, r : (rm_r) /r  (rm_r-1 for 8-bit)
        (rm, Operand::Reg(r)) => emit_rm(e, w, mand, &[op8(a.rm_r, size)], r.num, rm),
        // r, r/m (mem) : (rm_r + 2) /r  (rm_r+1 for 8-bit)
        (Operand::Reg(d), Operand::Mem(m)) => {
            let opc = if size == OpSize::B8 { a.rm_r + 1 } else { a.rm_r + 2 };
            emit_mem_rm(e, w, mand, &[opc], d.num, m)
        }
        // r/m, imm. MC's choices, for byte-identity:
        //   al + imm8        -> accumulator short form `acc8 ib` (no ModRM)
        //   r/m8 + imm8      -> 80 /ext ib
        //   imm fits i8      -> 83 /ext ib (shortest; even for rax)
        //   acc + wide imm   -> `acc8+1 iz` (no ModRM, 1 byte shorter than 81)
        //   else             -> 81 /ext iz
        (rm, Operand::Imm(v)) => {
            let is_acc = matches!(rm, Operand::Reg(r) if r.num == 0);
            if size == OpSize::B8 {
                if is_acc {
                    e.b(a.acc8);
                } else {
                    emit_rm(e, w, mand, &[0x80], a.ext, rm)?;
                }
                e.b(*v as u8);
            } else if (-128..=127).contains(v) {
                emit_rm(e, w, mand, &[0x83], a.ext, rm)?;
                e.b(*v as i8 as u8);
            } else if is_acc {
                e.ext(mand);
                if w {
                    e.b(0x48); // REX.W (rax form)
                }
                e.b(a.acc8 + 1);
                push_imm_sized(e, *v, size);
            } else {
                emit_rm(e, w, mand, &[0x81], a.ext, rm)?;
                push_imm_sized(e, *v, size);
            }
            Ok(())
        }
        _ => bail!("rasm: unsupported alu form {dst:?}, {src:?}"),
    }
}

fn operand_w(op: &Operand) -> bool {
    match op {
        Operand::Reg(r) => is64(*r),
        Operand::Mem(m) => mem_is_qword(m),
        _ => true,
    }
}

/// Regular two-operand SSE: `xmm, xmm/m` — mandatory prefix + fixed opcode, no
/// REX.W, no immediate. Covers scalar/packed arithmetic & logicals, unpack,
/// ordered compares, packed-integer ops, and xmm→xmm conversions. The
/// scalar-double arithmetic the kernel already used keeps its inline arms in
/// [`try_sse`]; this table fills in the rest of the SSE/SSE2 surface.
fn sse_rrm(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (pfx, op): (&[u8], &[u8]) = match mnemonic {
        // scalar single — F3 0F
        "addss" => (&[0xF3], &[0x0F, 0x58]),
        "subss" => (&[0xF3], &[0x0F, 0x5C]),
        "mulss" => (&[0xF3], &[0x0F, 0x59]),
        "divss" => (&[0xF3], &[0x0F, 0x5E]),
        "minss" => (&[0xF3], &[0x0F, 0x5D]),
        "maxss" => (&[0xF3], &[0x0F, 0x5F]),
        "sqrtss" => (&[0xF3], &[0x0F, 0x51]),
        // ordered compares (set EFLAGS) — packed-form prefixes
        "comiss" => (&[], &[0x0F, 0x2F]),
        "ucomiss" => (&[], &[0x0F, 0x2E]),
        "comisd" => (&[0x66], &[0x0F, 0x2F]),
        // packed single — no prefix
        "addps" => (&[], &[0x0F, 0x58]),
        "subps" => (&[], &[0x0F, 0x5C]),
        "mulps" => (&[], &[0x0F, 0x59]),
        "divps" => (&[], &[0x0F, 0x5E]),
        "minps" => (&[], &[0x0F, 0x5D]),
        "maxps" => (&[], &[0x0F, 0x5F]),
        "sqrtps" => (&[], &[0x0F, 0x51]),
        "andps" => (&[], &[0x0F, 0x54]),
        "andnps" => (&[], &[0x0F, 0x55]),
        "orps" => (&[], &[0x0F, 0x56]),
        "xorps" => (&[], &[0x0F, 0x57]),
        "unpcklps" => (&[], &[0x0F, 0x14]),
        "unpckhps" => (&[], &[0x0F, 0x15]),
        // packed double — 66 0F
        "addpd" => (&[0x66], &[0x0F, 0x58]),
        "subpd" => (&[0x66], &[0x0F, 0x5C]),
        "mulpd" => (&[0x66], &[0x0F, 0x59]),
        "divpd" => (&[0x66], &[0x0F, 0x5E]),
        "minpd" => (&[0x66], &[0x0F, 0x5D]),
        "maxpd" => (&[0x66], &[0x0F, 0x5F]),
        "sqrtpd" => (&[0x66], &[0x0F, 0x51]),
        // packed integer — 66 0F
        "paddb" => (&[0x66], &[0x0F, 0xFC]),
        "paddw" => (&[0x66], &[0x0F, 0xFD]),
        "paddd" => (&[0x66], &[0x0F, 0xFE]),
        "paddq" => (&[0x66], &[0x0F, 0xD4]),
        "psubb" => (&[0x66], &[0x0F, 0xF8]),
        "psubw" => (&[0x66], &[0x0F, 0xF9]),
        "psubd" => (&[0x66], &[0x0F, 0xFA]),
        "psubq" => (&[0x66], &[0x0F, 0xFB]),
        "pmullw" => (&[0x66], &[0x0F, 0xD5]),
        "pmulld" => (&[0x66], &[0x0F, 0x38, 0x40]), // SSE4.1, 3-byte opcode
        "pand" => (&[0x66], &[0x0F, 0xDB]),
        "pandn" => (&[0x66], &[0x0F, 0xDF]),
        "por" => (&[0x66], &[0x0F, 0xEB]),
        "pxor" => (&[0x66], &[0x0F, 0xEF]),
        "pcmpeqb" => (&[0x66], &[0x0F, 0x74]),
        "pcmpeqw" => (&[0x66], &[0x0F, 0x75]),
        "pcmpeqd" => (&[0x66], &[0x0F, 0x76]),
        // xmm→xmm conversions (no GPR, no REX.W)
        "cvtsd2ss" => (&[0xF2], &[0x0F, 0x5A]),
        "cvtss2sd" => (&[0xF3], &[0x0F, 0x5A]),
        "cvtdq2pd" => (&[0xF3], &[0x0F, 0xE6]),
        "cvtdq2ps" => (&[], &[0x0F, 0x5B]),
        "cvtpd2ps" => (&[0x66], &[0x0F, 0x5A]),
        "cvtps2pd" => (&[], &[0x0F, 0x5A]),
        _ => return None,
    };
    let [Operand::Reg(d), src] = ops else { return None };
    if d.class != RegClass::Xmm {
        return None;
    }
    let mut e = Encoded::default();
    match emit_rm(&mut e, false, pfx, op, d.num, src) {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// Three-operand SSE with an imm8: `shufps`/`shufpd`/`pshufd xmm, xmm/m, imm8`.
fn sse_shuffle(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (pfx, op): (&[u8], u8) = match mnemonic {
        "shufps" => (&[], 0xC6),
        "shufpd" => (&[0x66], 0xC6),
        "pshufd" => (&[0x66], 0x70),
        _ => return None,
    };
    let [Operand::Reg(d), src, Operand::Imm(imm)] = ops else { return None };
    if d.class != RegClass::Xmm {
        return None;
    }
    let mut e = Encoded::default();
    if let Err(err) = emit_rm(&mut e, false, pfx, &[0x0F, op], d.num, src) {
        return Some(Err(err));
    }
    e.b(*imm as u8);
    Some(Ok(e))
}

/// Whether an operand is a general-purpose register (not a vector register).
fn is_gpr_reg(op: &Operand) -> bool {
    matches!(op, Operand::Reg(r) if r.class != RegClass::Xmm && r.class != RegClass::Ymm)
}

fn is_vec_class(c: RegClass) -> bool {
    matches!(c, RegClass::Xmm | RegClass::Ymm | RegClass::Zmm)
}

// ── AVX / VEX encoding ───────────────────────────────────────────────────────
//
// VEX replaces the legacy prefix+REX+escape bytes with a compact 2- or 3-byte
// prefix carrying the inverted REX bits, an NDS register (`vvvv`), the vector
// length (`L`: 0=xmm/128, 1=ymm/256), the implied legacy prefix (`pp`: 0=none,
// 1=66, 2=F3, 3=F2), and the opcode map (`mmmmm`: 1=0F, 2=0F38, 3=0F3A). LLVM
// picks the 2-byte form (C5) whenever map==0F, W==0, and X==B==0; else 3-byte
// (C4). Matching that choice is required for byte-identity.

/// Emit a VEX prefix. `r`/`x`/`b` are the *true* (un-inverted) high bits of
/// ModRM.reg / SIB.index / ModRM.rm-base; `vvvv` is the NDS register (0..15, or
/// 0 when unused — it encodes as 1111).
fn vex(e: &mut Encoded, r: bool, x: bool, b: bool, map: u8, w: bool, vvvv: u8, l: bool, pp: u8) {
    let vvvv_inv = (!vvvv) & 0x0F;
    if map == 1 && !w && !x && !b {
        e.b(0xC5);
        e.b(((!r as u8) << 7) | (vvvv_inv << 3) | ((l as u8) << 2) | (pp & 3));
    } else {
        e.b(0xC4);
        e.b(((!r as u8) << 7) | ((!x as u8) << 6) | ((!b as u8) << 5) | (map & 0x1F));
        e.b(((w as u8) << 7) | (vvvv_inv << 3) | ((l as u8) << 2) | (pp & 3));
    }
}

/// Encode a VEX instruction: prefix + single-byte opcode + ModRM for `rm`.
/// `reg` is the ModRM.reg register; `vvvv` the NDS register (None = unused).
fn emit_vex_rm(
    e: &mut Encoded,
    map: u8,
    w: bool,
    pp: u8,
    l: bool,
    opcode: u8,
    reg: u8,
    vvvv: Option<u8>,
    rm: &Operand,
) -> Result<()> {
    let (x, b) = match rm {
        Operand::Reg(r) => (false, r.num >= 8),
        Operand::Mem(m) => mem_xb(m),
        other => bail!("expected reg/mem r/m, got {other:?}"),
    };
    vex(e, reg >= 8, x, b, map, w, vvvv.unwrap_or(0), l, pp);
    e.b(opcode);
    match rm {
        Operand::Reg(r) => emit_modrm_reg(e, reg, r.num),
        Operand::Mem(m) => emit_modrm_mem(e, reg, m)?,
        _ => unreachable!(),
    }
    Ok(())
}

/// Vector length code: 0=xmm/128, 1=ymm/256, 2=zmm/512.
fn vec_len(c: RegClass) -> u8 {
    match c {
        RegClass::Ymm => 1,
        RegClass::Zmm => 2,
        _ => 0,
    }
}

/// EVEX is required for a 512-bit op or any extended vector register (16..=31).
fn use_evex(ll: u8, reg: u8, vvvv: Option<u8>, rm: &Operand) -> bool {
    ll == 2
        || reg >= 16
        || vvvv.map_or(false, |v| v >= 16)
        || matches!(rm, Operand::Reg(r) if r.num >= 16)
}

/// Emit a 4-byte EVEX-encoded instruction (AVX-512): `62 P0 P1 P2 opcode ModRM`.
/// `ll`: 0=128, 1=256, 2=512. `mask`/`z`/`bcast` are 0/false for the unmasked,
/// no-broadcast forms (masking lands in a later increment).
#[allow(clippy::too_many_arguments)]
fn emit_evex_rm(
    e: &mut Encoded,
    map: u8,
    w: bool,
    pp: u8,
    ll: u8,
    opcode: u8,
    reg: u8,
    vvvv: Option<u8>,
    rm: &Operand,
) -> Result<()> {
    let vv = vvvv.unwrap_or(0);
    // reg (ModRM.reg) extends to 5 bits: R = bit3, R' = bit4.
    let r = (reg >> 3) & 1;
    let r2 = (reg >> 4) & 1;
    // rm extension: register-direct rm extends via B (bit3) and X (bit4); a
    // memory rm takes X/B from index/base like REX.
    let (x, b) = match rm {
        Operand::Reg(rr) => ((rr.num >> 4) & 1, (rr.num >> 3) & 1),
        Operand::Mem(m) => {
            let (xx, bb) = mem_xb(m);
            (xx as u8, bb as u8)
        }
        other => bail!("expected reg/mem r/m, got {other:?}"),
    };
    let vlo = vv & 0x0F;
    let vhi = (vv >> 4) & 1; // V'
    let (mask, z, bcast) = (0u8, 0u8, 0u8);

    e.b(0x62);
    // P0: R̄ X̄ B̄ R̄' 0 0 m m  (R/X/B/R' inverted)
    e.b(((r ^ 1) << 7) | ((x ^ 1) << 6) | ((b ^ 1) << 5) | ((r2 ^ 1) << 4) | (map & 0x03));
    // P1: W v̄v̄v̄v̄ 1 pp
    e.b(((w as u8) << 7) | (((!vlo) & 0x0F) << 3) | (1 << 2) | (pp & 3));
    // P2: z L'L b V̄' aaa
    e.b((z << 7) | ((ll & 3) << 5) | (bcast << 4) | (((vhi ^ 1) & 1) << 3) | (mask & 7));
    e.b(opcode);
    match rm {
        Operand::Reg(rr) => emit_modrm_reg(e, reg, rr.num),
        Operand::Mem(m) => emit_modrm_mem(e, reg, m)?,
        _ => unreachable!(),
    }
    Ok(())
}

/// Encode a vector reg/vvvv/rm instruction, choosing VEX or EVEX automatically.
#[allow(clippy::too_many_arguments)]
fn emit_vec_rm(
    e: &mut Encoded,
    map: u8,
    w: bool,
    pp: u8,
    ll: u8,
    opcode: u8,
    reg: u8,
    vvvv: Option<u8>,
    rm: &Operand,
) -> Result<()> {
    if use_evex(ll, reg, vvvv, rm) {
        // EVEX W is semantic (W1 for double/qword elements).
        emit_evex_rm(e, map, w, pp, ll, opcode, reg, vvvv, rm)
    } else {
        // All VEX forms here are WIG; LLVM normalizes the byte to W=0.
        emit_vex_rm(e, map, false, pp, ll == 1, opcode, reg, vvvv, rm)
    }
}

/// Packed VEX ops whose two sources are interchangeable, so LLVM may swap them
/// to keep a high register out of the `rm` field (shorter 2-byte VEX). Scalar
/// ops are excluded: their upper lanes come from `vvvv`, so the sources are not
/// interchangeable even though the arithmetic is commutative.
fn vex_commutative(m: &str) -> bool {
    matches!(
        m,
        "vaddps" | "vaddpd" | "vmulps" | "vmulpd" | "vandps" | "vandpd" | "vorps" | "vorpd"
            | "vxorps" | "vxorpd" | "vpaddb" | "vpaddw" | "vpaddd" | "vpaddq" | "vpand" | "vpor"
            | "vpxor" | "vpcmpeqb" | "vpcmpeqd" | "vpmullw" | "vpmulld"
    )
}

/// 3-operand VEX `dest, vvvv(src1), rm(src2)` — packed/scalar arithmetic,
/// logicals, packed integer. `(pp, map, w, opcode)`.
fn vex_rvm(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (pp, map, w, op): (u8, u8, bool, u8) = match mnemonic {
        // packed single — none.0F
        "vaddps" => (0, 1, false, 0x58),
        "vsubps" => (0, 1, false, 0x5C),
        "vmulps" => (0, 1, false, 0x59),
        "vdivps" => (0, 1, false, 0x5E),
        "vminps" => (0, 1, false, 0x5D),
        "vmaxps" => (0, 1, false, 0x5F),
        "vandps" => (0, 1, false, 0x54),
        "vandnps" => (0, 1, false, 0x55),
        "vorps" => (0, 1, false, 0x56),
        "vxorps" => (0, 1, false, 0x57),
        "vunpcklps" => (0, 1, false, 0x14),
        "vunpckhps" => (0, 1, false, 0x15),
        // packed double — 66.0F.W1 (W is EVEX-semantic; VEX forces it to 0)
        "vaddpd" => (1, 1, true, 0x58),
        "vsubpd" => (1, 1, true, 0x5C),
        "vmulpd" => (1, 1, true, 0x59),
        "vdivpd" => (1, 1, true, 0x5E),
        "vminpd" => (1, 1, true, 0x5D),
        "vmaxpd" => (1, 1, true, 0x5F),
        "vandpd" => (1, 1, true, 0x54),
        "vandnpd" => (1, 1, true, 0x55),
        "vorpd" => (1, 1, true, 0x56),
        "vxorpd" => (1, 1, true, 0x57),
        // scalar single — F3.0F
        "vaddss" => (2, 1, false, 0x58),
        "vsubss" => (2, 1, false, 0x5C),
        "vmulss" => (2, 1, false, 0x59),
        "vdivss" => (2, 1, false, 0x5E),
        "vminss" => (2, 1, false, 0x5D),
        "vmaxss" => (2, 1, false, 0x5F),
        "vsqrtss" => (2, 1, false, 0x51),
        // scalar double — F2.0F.W1
        "vaddsd" => (3, 1, true, 0x58),
        "vsubsd" => (3, 1, true, 0x5C),
        "vmulsd" => (3, 1, true, 0x59),
        "vdivsd" => (3, 1, true, 0x5E),
        "vminsd" => (3, 1, true, 0x5D),
        "vmaxsd" => (3, 1, true, 0x5F),
        "vsqrtsd" => (3, 1, true, 0x51),
        // packed integer — 66.0F (vpmulld is 66.0F38)
        "vpaddb" => (1, 1, false, 0xFC),
        "vpaddw" => (1, 1, false, 0xFD),
        "vpaddd" => (1, 1, false, 0xFE),
        "vpaddq" => (1, 1, true, 0xD4),
        "vpsubb" => (1, 1, false, 0xF8),
        "vpsubw" => (1, 1, false, 0xF9),
        "vpsubd" => (1, 1, false, 0xFA),
        "vpsubq" => (1, 1, true, 0xFB),
        "vpand" => (1, 1, false, 0xDB),
        "vpandn" => (1, 1, false, 0xDF),
        "vpor" => (1, 1, false, 0xEB),
        "vpxor" => (1, 1, false, 0xEF),
        "vpcmpeqb" => (1, 1, false, 0x74),
        "vpcmpeqd" => (1, 1, false, 0x76),
        "vpmullw" => (1, 1, false, 0xD5),
        "vpmulld" => (1, 2, false, 0x40),
        _ => return None,
    };
    let [Operand::Reg(d), Operand::Reg(v), rm] = ops else { return None };
    if !is_vec_class(d.class) {
        return None;
    }
    let ll = vec_len(d.class);
    let evex = use_evex(ll, d.num, Some(v.num), rm);
    // Commutative-source swap (VEX only): if src2 is a high reg and src1 is low,
    // swap so the high reg lands in vvvv (not rm), enabling the 2-byte form. Only
    // worth it when 2-byte VEX is reachable (map==0F, no EVEX); LLVM doesn't swap
    // a 0F38 op like `vpmulld` (always 3-byte) or any EVEX op (no 2-byte form).
    let swap = !evex
        && vex_commutative(mnemonic)
        && map == 1
        && matches!(rm, Operand::Reg(r2) if r2.num >= 8)
        && v.num < 8;
    let (vvvv, rm_ref): (u8, &Operand) = if swap {
        let r2 = match rm {
            Operand::Reg(r) => r.num,
            _ => unreachable!(),
        };
        (r2, &ops[1])
    } else {
        (v.num, rm)
    };
    let mut e = Encoded::default();
    match emit_vec_rm(&mut e, map, w, pp, ll, op, d.num, Some(vvvv), rm_ref) {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// 2-operand VEX `dest, rm` (vvvv unused) — packed sqrt and reciprocals, packed
/// conversions.
fn vex_rm(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (pp, map, w, op): (u8, u8, bool, u8) = match mnemonic {
        "vsqrtps" => (0, 1, false, 0x51),
        "vsqrtpd" => (1, 1, true, 0x51),
        "vrcpps" => (0, 1, false, 0x53),
        "vrsqrtps" => (0, 1, false, 0x52),
        "vcvtdq2ps" => (0, 1, false, 0x5B),
        "vcvtps2dq" => (1, 1, false, 0x5B),
        "vcvttps2dq" => (2, 1, false, 0x5B),
        _ => return None,
    };
    let [Operand::Reg(d), rm] = ops else { return None };
    if !is_vec_class(d.class) {
        return None;
    }
    let mut e = Encoded::default();
    match emit_vec_rm(&mut e, map, w, pp, vec_len(d.class), op, d.num, None, rm) {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// VEX moves with load (`vec ← vec/m`) and store (`m ← vec`) directions.
fn vex_mov(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    // (pp, load opcode, store opcode, EVEX-semantic W)
    let (pp, load, store, w): (u8, u8, u8, bool) = match mnemonic {
        "vmovaps" => (0, 0x28, 0x29, false),
        "vmovups" => (0, 0x10, 0x11, false),
        "vmovapd" => (1, 0x28, 0x29, true),
        "vmovupd" => (1, 0x10, 0x11, true),
        "vmovdqa" => (1, 0x6F, 0x7F, false),
        "vmovdqu" => (2, 0x6F, 0x7F, false),
        _ => return None,
    };
    let mut e = Encoded::default();
    let r = match ops {
        // reg-reg: LLVM flips to the store opcode when src is a high reg and dest
        // is low — that puts the high reg in the `reg`/R field (fine for 2-byte
        // VEX) instead of `rm`/B (which would force 3-byte). VEX only; EVEX has
        // no 2-byte form and encodes any register either way.
        [Operand::Reg(d), Operand::Reg(s)] if is_vec_class(d.class) && is_vec_class(s.class) => {
            let ll = vec_len(d.class);
            let evex = use_evex(ll, d.num, None, &Operand::Reg(*s));
            if !evex && s.num >= 8 && d.num < 8 {
                emit_vec_rm(&mut e, 1, w, pp, ll, store, s.num, None, &Operand::Reg(*d))
            } else {
                emit_vec_rm(&mut e, 1, w, pp, ll, load, d.num, None, &Operand::Reg(*s))
            }
        }
        [Operand::Reg(d), src] if is_vec_class(d.class) => {
            emit_vec_rm(&mut e, 1, w, pp, vec_len(d.class), load, d.num, None, src)
        }
        [dst @ Operand::Mem(_), Operand::Reg(s)] if is_vec_class(s.class) => {
            emit_vec_rm(&mut e, 1, w, pp, vec_len(s.class), store, s.num, None, dst)
        }
        _ => return None,
    };
    match r {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// VEX shuffles with an imm8: `vshufps`/`vshufpd` (3-op RVMI) and `vpshufd`
/// (2-op RMI, vvvv unused).
fn vex_shuffle(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    // RVMI: dest, vvvv, rm, imm8.
    if let Some((pp, op, w)) = match mnemonic {
        "vshufps" => Some((0u8, 0xC6u8, false)),
        "vshufpd" => Some((1, 0xC6, true)),
        _ => None,
    } {
        let [Operand::Reg(d), Operand::Reg(v), rm, Operand::Imm(imm)] = ops else { return None };
        if !is_vec_class(d.class) {
            return None;
        }
        let mut e = Encoded::default();
        if let Err(err) = emit_vec_rm(&mut e, 1, w, pp, vec_len(d.class), op, d.num, Some(v.num), rm) {
            return Some(Err(err));
        }
        e.b(*imm as u8);
        return Some(Ok(e));
    }
    // RMI: vpshufd dest, rm, imm8.
    if mnemonic == "vpshufd" {
        let [Operand::Reg(d), rm, Operand::Imm(imm)] = ops else { return None };
        if !is_vec_class(d.class) {
            return None;
        }
        let mut e = Encoded::default();
        if let Err(err) = emit_vec_rm(&mut e, 1, false, 1, vec_len(d.class), 0x70, d.num, None, rm) {
            return Some(Err(err));
        }
        e.b(*imm as u8);
        return Some(Ok(e));
    }
    None
}

/// AVX/AVX2 (VEX-encoded) instructions.
fn try_vex(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    vex_rvm(mnemonic, ops)
        .or_else(|| vex_rm(mnemonic, ops))
        .or_else(|| vex_mov(mnemonic, ops))
        .or_else(|| vex_shuffle(mnemonic, ops))
}

/// SSE moves with load (`xmm ← xmm/m`) and store (`m ← xmm`) directions. The
/// kernel's `movsd`/`movups` keep their inline arms; this covers the rest.
fn sse_mov(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    // (prefix, load opcode, store opcode) — opcodes follow 0F.
    let (pfx, load, store): (&[u8], u8, u8) = match mnemonic {
        "movss" => (&[0xF3], 0x10, 0x11),
        "movaps" => (&[], 0x28, 0x29),
        "movapd" => (&[0x66], 0x28, 0x29),
        "movupd" => (&[0x66], 0x10, 0x11),
        "movdqa" => (&[0x66], 0x6F, 0x7F),
        "movdqu" => (&[0xF3], 0x6F, 0x7F),
        _ => return None,
    };
    let mut e = Encoded::default();
    let r = match ops {
        [Operand::Reg(d), src] if d.class == RegClass::Xmm => {
            emit_rm(&mut e, false, pfx, &[0x0F, load], d.num, src)
        }
        [Operand::Mem(m), Operand::Reg(s)] if s.class == RegClass::Xmm => {
            emit_mem_rm(&mut e, false, pfx, &[0x0F, store], s.num, m)
        }
        _ => return None,
    };
    match r {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// `movd`/`movq` between xmm and GPR/memory. The `movq` GPR register↔register
/// forms keep their inline arms (`66 REX.W 0F 6E/7E`); this adds `movd`, the
/// memory forms, and the xmm↔xmm `movq` (`F3 0F 7E` load / `66 0F D6` store).
fn sse_movd_q(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            // movd xmm, r/m32 : 66 0F 6E (no REX.W)
            ("movd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x6E], d.num, src)?;
            }
            // movd r/m32, xmm : 66 0F 7E
            ("movd", [dst, Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x7E], s.num, dst)?;
            }
            // movq xmm, xmm/m64 : F3 0F 7E (load). A GPR source falls through to
            // the inline `66 REX.W 0F 6E` arm.
            ("movq", [Operand::Reg(d), src]) if d.class == RegClass::Xmm && !is_gpr_reg(src) => {
                emit_rm(&mut e, false, &[0xF3], &[0x0F, 0x7E], d.num, src)?;
            }
            // movq m64, xmm : 66 0F D6 (store). A GPR dest falls through to inline.
            ("movq", [Operand::Mem(m), Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_mem_rm(&mut e, false, &[0x66], &[0x0F, 0xD6], s.num, m)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

/// Conversions that touch a GPR: `cvtsi2ss` (GPR/m → xmm, REX.W per source) and
/// `cvt(t)sd2si`/`cvt(t)ss2si` (xmm/m → GPR, REX.W per dest). `cvtsi2sd` keeps
/// its inline arm.
fn sse_cvt_gpr(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            ("cvtsi2ss", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, operand_w(src), &[0xF3], &[0x0F, 0x2A], d.num, src)?;
            }
            ("cvtsd2si" | "cvttsd2si" | "cvtss2si" | "cvttss2si", [Operand::Reg(d), src])
                if d.class == RegClass::R64 || d.class == RegClass::R32 =>
            {
                let (pfx, op): (u8, u8) = match mnemonic {
                    "cvtsd2si" => (0xF2, 0x2D),
                    "cvttsd2si" => (0xF2, 0x2C),
                    "cvtss2si" => (0xF3, 0x2D),
                    "cvttss2si" => (0xF3, 0x2C),
                    _ => unreachable!(),
                };
                emit_rm(&mut e, is64(*d), &[pfx], &[0x0F, op], d.num, src)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

/// SSE2 scalar-double + the xmm move/convert family (the FTOS/REX.R island).
fn try_sse(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    if let Some(r) = sse_rrm(mnemonic, ops) {
        return Some(r);
    }
    if let Some(r) = sse_shuffle(mnemonic, ops) {
        return Some(r);
    }
    if let Some(r) = sse_mov(mnemonic, ops) {
        return Some(r);
    }
    if let Some(r) = sse_movd_q(mnemonic, ops) {
        return Some(r);
    }
    if let Some(r) = sse_cvt_gpr(mnemonic, ops) {
        return Some(r);
    }
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            // movsd: load (xmm <- r/m) F2 0F 10 ; store (m <- xmm) F2 0F 11
            ("movsd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0xF2], &[0x0F, 0x10], d.num, src)?;
            }
            ("movsd", [Operand::Mem(m), Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_mem_rm(&mut e, false, &[0xF2], &[0x0F, 0x11], s.num, m)?;
            }
            // movups: load 0F 10 ; store 0F 11 (no mandatory prefix)
            ("movups", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[], &[0x0F, 0x10], d.num, src)?;
            }
            ("movups", [Operand::Mem(m), Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_mem_rm(&mut e, false, &[], &[0x0F, 0x11], s.num, m)?;
            }
            // arithmetic: F2 0F <op> /r, dst xmm = reg field
            ("addsd" | "subsd" | "mulsd" | "divsd", [Operand::Reg(d), src])
                if d.class == RegClass::Xmm =>
            {
                let op = match mnemonic {
                    "addsd" => 0x58,
                    "subsd" => 0x5C,
                    "mulsd" => 0x59,
                    "divsd" => 0x5E,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, false, &[0xF2], &[0x0F, op], d.num, src)?;
            }
            ("ucomisd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x2E], d.num, src)?;
            }
            ("xorpd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x57], d.num, src)?;
            }
            // packed-double logical ops: 66 0F <op> /r — LET abs/select blends.
            ("andpd" | "andnpd" | "orpd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                let op = match mnemonic {
                    "andpd" => 0x54,
                    "andnpd" => 0x55,
                    "orpd" => 0x56,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, false, &[0x66], &[0x0F, op], d.num, src)?;
            }
            // sqrtsd / minsd / maxsd : F2 0F <op> /r.
            ("sqrtsd" | "minsd" | "maxsd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                let op = match mnemonic {
                    "sqrtsd" => 0x51,
                    "minsd" => 0x5D,
                    "maxsd" => 0x5F,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, false, &[0xF2], &[0x0F, op], d.num, src)?;
            }
            // cmpsd pseudo-ops: F2 0F C2 /r ib — predicate encoded in the
            // mnemonic (eq=0, lt=1, le=2, neq=4). LET comparisons.
            ("cmpeqsd" | "cmpltsd" | "cmplesd" | "cmpneqsd", [Operand::Reg(d), src])
                if d.class == RegClass::Xmm =>
            {
                let pred: u8 = match mnemonic {
                    "cmpeqsd" => 0,
                    "cmpltsd" => 1,
                    "cmplesd" => 2,
                    "cmpneqsd" => 4,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, false, &[0xF2], &[0x0F, 0xC2], d.num, src)?;
                e.b(pred);
            }
            // roundsd xmm, xmm/m, imm8 : 66 0F 3A 0B /r ib (SSE4.1) — LET
            // floor/ceil/round/trunc intrinsics.
            ("roundsd", [Operand::Reg(d), src, Operand::Imm(mode)]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x3A, 0x0B], d.num, src)?;
                e.b(*mode as u8);
            }
            // movq xmm, r64 : 66 REX.W 0F 6E /r ; movq r64, xmm : 66 REX.W 0F 7E /r
            ("movq", [Operand::Reg(d), Operand::Reg(s)])
                if d.class == RegClass::Xmm && s.class == RegClass::R64 =>
            {
                emit_reg_rm(&mut e, true, &[0x66], &[0x0F, 0x6E], d.num, s.num);
            }
            ("movq", [Operand::Reg(d), Operand::Reg(s)])
                if d.class == RegClass::R64 && s.class == RegClass::Xmm =>
            {
                emit_reg_rm(&mut e, true, &[0x66], &[0x0F, 0x7E], s.num, d.num);
            }
            // cvtsi2sd xmm, r/m32|64 : F2 0F 2A /r, REX.W set ONLY for a 64-bit
            // source (r64/qword) — a 32-bit source must NOT carry REX.W, else the
            // CPU reads 8 bytes instead of 4.
            ("cvtsi2sd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, operand_w(src), &[0xF2], &[0x0F, 0x2A], d.num, src)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

/// One-operand F7/FF group: neg/not/mul/imul/div/idiv (F7 /ext) and inc/dec
/// (FF /ext).
fn try_unary(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (opcode, ext) = match mnemonic {
        "not" => (0xF7u8, 2u8),
        "neg" => (0xF7, 3),
        "mul" => (0xF7, 4),
        "imul" if ops.len() == 1 => (0xF7, 5),
        "div" => (0xF7, 6),
        "idiv" => (0xF7, 7),
        "inc" => (0xFF, 0),
        "dec" => (0xFF, 1),
        _ => return None,
    };
    let [rm] = ops else { return None };
    let mut e = Encoded::default();
    match emit_rm(&mut e, operand_w(rm), &[], &[opcode], ext, rm) {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// Shift group: shl/sal/shr/sar/rol/ror by 1 (D1), imm8 (C1), or cl (D3).
fn try_shift(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let ext = match mnemonic {
        "rol" => 0u8,
        "ror" => 1,
        "rcl" => 2,
        "rcr" => 3,
        "shl" | "sal" => 4,
        "shr" => 5,
        "sar" => 7,
        _ => return None,
    };
    let [rm, count] = ops else { return None };
    let mut e = Encoded::default();
    let w = operand_w(rm);
    let r = match count {
        Operand::Imm(1) => emit_rm(&mut e, w, &[], &[0xD1], ext, rm),
        Operand::Imm(n) => emit_rm(&mut e, w, &[], &[0xC1], ext, rm).map(|()| e.b(*n as u8)),
        Operand::Reg(r) if r.class == RegClass::R8 && r.num == 1 => {
            emit_rm(&mut e, w, &[], &[0xD3], ext, rm) // shift by cl
        }
        other => Err(anyhow::anyhow!("bad shift count {other:?}")),
    };
    match r {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vendor::wfasm::rasm::parse::parse_line;
    use crate::vendor::wfasm::rasm::Line;
    use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};

    fn enc(line: &str) -> Encoded {
        match parse_line(line).unwrap() {
            Line::Insn { mnemonic, ops } => encode(&mnemonic, &ops).unwrap(),
            other => panic!("not an insn: {other:?}"),
        }
    }

    fn bytes(line: &str) -> Vec<u8> {
        enc(line).bytes
    }

    /// Decode our bytes with iced and format back to Intel syntax — the
    /// round-trip oracle (catches "valid but wrong instruction").
    fn roundtrip(line: &str) -> String {
        let b = bytes(line);
        let mut dec = Decoder::with_ip(64, &b, 0x1000, DecoderOptions::NONE);
        let insn: Instruction = dec.decode();
        assert!(!insn.is_invalid(), "iced could not decode `{line}` -> {b:02x?}");
        assert_eq!(dec.position(), b.len(), "trailing bytes after `{line}`: {b:02x?}");
        let mut f = IntelFormatter::new();
        let mut s = String::new();
        f.format(&insn, &mut s);
        s
    }

    #[test]
    fn exact_bytes_match_golden_leaves() {
        // dup_ body: mov [rbp-8], rax ; sub rbp, 8 ; ret  (golden 48 89 45 f8 / 48 83 ed 08 / c3)
        assert_eq!(bytes("mov [rbp - 8], rax"), vec![0x48, 0x89, 0x45, 0xF8]);
        assert_eq!(bytes("sub rbp, 8"), vec![0x48, 0x83, 0xED, 0x08]);
        assert_eq!(bytes("ret"), vec![0xC3]);
        // plus body: add rax, [rbp] ; add rbp, 8 ; ret  (golden 48 03 45 00 / 48 83 c5 08 / c3)
        assert_eq!(bytes("add rax, [rbp]"), vec![0x48, 0x03, 0x45, 0x00]);
        assert_eq!(bytes("add rbp, 8"), vec![0x48, 0x83, 0xC5, 0x08]);
    }

    #[test]
    fn mov_forms() {
        assert_eq!(roundtrip("mov rax, rcx"), "mov rax,rcx");
        assert_eq!(roundtrip("mov rcx, [rbx + 4632]"), "mov rcx,[rbx+1218h]");
        assert_eq!(roundtrip("mov [rbx + 4632], rcx"), "mov [rbx+1218h],rcx");
        assert_eq!(roundtrip("mov rax, 0xDEADBEEF"), "mov rax,0DEADBEEFh");
        assert_eq!(roundtrip("mov r8, [rsp]"), "mov r8,[rsp]");
        assert_eq!(roundtrip("mov rax, 0x4010000000000000"), "mov rax,4010000000000000h");
    }

    #[test]
    fn alu_forms() {
        assert_eq!(roundtrip("add rax, rcx"), "add rax,rcx");
        assert_eq!(roundtrip("sub rbp, 8"), "sub rbp,8");
        assert_eq!(roundtrip("cmp rax, [rbp]"), "cmp rax,[rbp]");
        assert_eq!(roundtrip("and rax, 0x1FF"), "and rax,1FFh");
        assert_eq!(roundtrip("xor r10, r11"), "xor r10,r11");
        assert_eq!(roundtrip("add qword ptr [rsp], 8"), "add qword ptr [rsp],8");
    }

    #[test]
    fn push_pop_lea_test() {
        assert_eq!(bytes("push rbx"), vec![0x53]);
        assert_eq!(bytes("push r15"), vec![0x41, 0x57]);
        assert_eq!(bytes("pop rbp"), vec![0x5D]);
        assert_eq!(roundtrip("lea r8, [rax + rax*1]"), "lea r8,[rax+rax]");
        assert_eq!(roundtrip("test rax, rax"), "test rax,rax");
    }

    #[test]
    fn sse_ftos_xmm15_island() {
        // f_plus golden: addsd xmm15,[rcx] = f2 44 0f 58 39 (REX.R for xmm15)
        assert_eq!(bytes("addsd xmm15, qword ptr [rcx]"), vec![0xF2, 0x44, 0x0F, 0x58, 0x39]);
        // f_fetch golden fragment: movsd qword ptr [rdx-8], xmm15 (store, REX.R)
        assert_eq!(bytes("movsd qword ptr [rcx - 8], xmm15"), vec![0xF2, 0x44, 0x0F, 0x11, 0x79, 0xF8]);
        // movsd xmm15, [rcx] (load, REX.R)
        assert_eq!(bytes("movsd xmm15, qword ptr [rcx]"), vec![0xF2, 0x44, 0x0F, 0x10, 0x39]);
        // round-trips for the rest of the island
        assert_eq!(roundtrip("subsd xmm15, xmm14"), "subsd xmm15,xmm14");
        assert_eq!(roundtrip("mulsd xmm0, xmm1"), "mulsd xmm0,xmm1");
        assert_eq!(roundtrip("divsd xmm6, qword ptr [rbx]"), "divsd xmm6,[rbx]");
        assert_eq!(roundtrip("ucomisd xmm15, xmm0"), "ucomisd xmm15,xmm0");
        assert_eq!(roundtrip("xorpd xmm0, xmm0"), "xorpd xmm0,xmm0");
        assert_eq!(roundtrip("movups xmm8, [rsp]"), "movups xmm8,[rsp]");
        assert_eq!(roundtrip("movups [rsp + 32], xmm8"), "movups [rsp+20h],xmm8");
        assert_eq!(roundtrip("movq xmm15, rdx"), "movq xmm15,rdx");
        assert_eq!(roundtrip("movq r10, xmm15"), "movq r10,xmm15");
        assert_eq!(roundtrip("cvtsi2sd xmm0, rcx"), "cvtsi2sd xmm0,rcx");
        assert_eq!(roundtrip("cvttsd2si rcx, xmm15"), "cvttsd2si rcx,xmm15");
    }

    #[test]
    fn sse_let_island() {
        // F2 0F <op> /r single-double ops.
        assert_eq!(bytes("sqrtsd xmm6, xmm6"), vec![0xF2, 0x0F, 0x51, 0xF6]);
        assert_eq!(bytes("minsd xmm0, xmm1"), vec![0xF2, 0x0F, 0x5D, 0xC1]);
        assert_eq!(bytes("maxsd xmm0, xmm1"), vec![0xF2, 0x0F, 0x5F, 0xC1]);
        // 66 0F <op> /r packed-double logicals.
        assert_eq!(bytes("andpd xmm6, xmm7"), vec![0x66, 0x0F, 0x54, 0xF7]);
        assert_eq!(bytes("andnpd xmm0, xmm1"), vec![0x66, 0x0F, 0x55, 0xC1]);
        assert_eq!(bytes("orpd xmm0, xmm1"), vec![0x66, 0x0F, 0x56, 0xC1]);
        // xmmword ptr memory operand: opcode + ModRM for [rcx] (mod=00, rm=001).
        assert_eq!(bytes("andpd xmm6, xmmword ptr [rcx]"), vec![0x66, 0x0F, 0x54, 0x31]);
        // cmpsd pseudo-ops: F2 0F C2 /r ib, predicate from the mnemonic.
        assert_eq!(bytes("cmpeqsd xmm0, xmm1"), vec![0xF2, 0x0F, 0xC2, 0xC1, 0x00]);
        assert_eq!(bytes("cmpltsd xmm0, xmm1"), vec![0xF2, 0x0F, 0xC2, 0xC1, 0x01]);
        assert_eq!(bytes("cmplesd xmm0, xmm1"), vec![0xF2, 0x0F, 0xC2, 0xC1, 0x02]);
        assert_eq!(bytes("cmpneqsd xmm0, xmm1"), vec![0xF2, 0x0F, 0xC2, 0xC1, 0x04]);
        // roundsd xmm, xmm, imm8 : 66 0F 3A 0B /r ib (SSE4.1).
        assert_eq!(bytes("roundsd xmm0, xmm0, 1"), vec![0x66, 0x0F, 0x3A, 0x0B, 0xC0, 0x01]);
        assert_eq!(bytes("roundsd xmm2, xmm3, 0"), vec![0x66, 0x0F, 0x3A, 0x0B, 0xD3, 0x00]);
    }

    #[test]
    fn multi_value_quad() {
        // .quad a, b emits two little-endian 8-byte words back to back.
        let m = crate::vendor::wfasm::rasm::assemble(".quad 0x8000000000000000, 0\n").unwrap();
        assert_eq!(
            m.code,
            vec![0, 0, 0, 0, 0, 0, 0, 0x80, /* */ 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn unary_and_shift() {
        assert_eq!(roundtrip("neg rcx"), "neg rcx");
        assert_eq!(roundtrip("not rax"), "not rax");
        assert_eq!(roundtrip("idiv rcx"), "idiv rcx");
        assert_eq!(roundtrip("inc qword ptr [rbx]"), "inc qword ptr [rbx]");
        assert_eq!(roundtrip("dec rax"), "dec rax");
        // shifts: by-1 → D1 (not C1 imm=1), to match MC
        assert_eq!(bytes("shl rax, 1"), vec![0x48, 0xD1, 0xE0]);
        assert_eq!(roundtrip("shl rax, 3"), "shl rax,3");
        assert_eq!(roundtrip("sar rdx, 63"), "sar rdx,3Fh");
        assert_eq!(roundtrip("shr r9, cl"), "shr r9,cl");
    }

    #[test]
    fn extend_setcc_cmov_imul_xchg_string() {
        // movzx/movsx with sized source (kernel forms)
        assert_eq!(roundtrip("movzx rax, byte ptr [rax]"), "movzx rax,byte ptr [rax]");
        assert_eq!(roundtrip("movsx rax, word ptr [rax]"), "movsx rax,word ptr [rax]");
        assert_eq!(roundtrip("movsxd rdx, edx"), "movsxd rdx,edx");
        // setcc / cmovcc
        assert_eq!(bytes("sete cl"), vec![0x0F, 0x94, 0xC1]);
        assert_eq!(bytes("setb cl"), vec![0x0F, 0x92, 0xC1]);
        assert_eq!(roundtrip("cmovl rax, rcx"), "cmovl rax,rcx");
        assert_eq!(roundtrip("cmovg rax, rcx"), "cmovg rax,rcx");
        // imul: 1-operand (F7 /5) vs 2-operand (0F AF)
        assert_eq!(roundtrip("imul rax, [rbp]"), "imul rax,[rbp]");
        assert_eq!(roundtrip("imul qword ptr [rbp]"), "imul qword ptr [rbp]");
        // xchg / xadd — assert bytes (golden-verified); iced formats xchg with
        // rm first, so the round-trip string is "xchg rsp,rbp".
        assert_eq!(bytes("xchg rbp, rsp"), vec![0x48, 0x87, 0xEC]);
        assert_eq!(roundtrip("xadd [rcx], rax"), "xadd [rcx],rax");
        // rep string ops
        assert_eq!(bytes("rep movsq"), vec![0xF3, 0x48, 0xA5]);
        assert_eq!(bytes("rep stosb"), vec![0xF3, 0xAA]);
    }

    #[test]
    fn rbp_r13_rsp_r12_modrm_traps() {
        // [rbp] forces disp8=0 (mod=01).
        assert_eq!(bytes("mov rax, [rbp]"), vec![0x48, 0x8B, 0x45, 0x00]);
        // [rsp] forces a SIB byte.
        assert_eq!(bytes("mov rax, [rsp]"), vec![0x48, 0x8B, 0x04, 0x24]);
        // [r13] forces disp8=0 + REX.B.
        assert_eq!(bytes("mov rax, [r13]"), vec![0x49, 0x8B, 0x45, 0x00]);
        // [r12] forces SIB + REX.B.
        assert_eq!(bytes("mov rax, [r12]"), vec![0x49, 0x8B, 0x04, 0x24]);
    }
}
