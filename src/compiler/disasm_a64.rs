//! DBG3 — machine-code disassembler for the *closed* AArch64 vocabulary the
//! MACVM compiler emits (docs/DEBUGGER.md §4.4, decision 2).
//!
//! This is deliberately NOT a general A64 disassembler. Every word in a
//! compiled blob comes from one of three emitters:
//! - the vendored structured encoder (`vendor/wfasm/a64/encode.rs`) via
//!   [`Assembler::emit`](crate::compiler::assembler::Assembler::emit),
//! - `JasmAssembler`'s hand-encoded intra-blob branches
//!   (`b`/`b.cond`/`cbz`/`cbnz`/`tbz`/`tbnz`) and `ldr`-literal / `bl .`
//!   placeholder words,
//! - `deopt_trap::brk_word` raw `brk #imm` words.
//!
//! This module is the inverse of exactly that set and nothing more. Any word
//! outside it prints as `.word 0x????????` — per §4.4 that fallback is
//! *itself a finding* ("the compiler emitted a word its own vocabulary can't
//! name"), so the decoder must never guess.
//!
//! Alias policy: every printed line must re-encode byte-identically through
//! the vendored parser + encoder (the corpus round-trip test below enforces
//! this). That rules out some conventional alias prints: e.g. `orr Rd, xzr,
//! #bitmask` is NOT printed as `mov Rd, #imm`, because the encoder's
//! `mov #imm` lowering prefers `movz`/`movn` and would pick a different word
//! for movz-encodable masks (`0xffff` is both a valid bitmask and a movz
//! group). Aliases that always re-encode to the same word (`cmp`/`cmn`/`tst`,
//! `mov` reg-reg, `lsl`/`lsr`/`asr`/`ubfx`/`sxtw`/…, `cset`/`csetm`) are
//! printed in their conventional form.
//!
//! Register 31 is positional (encode.rs / parse.rs `is_sp`): it means SP as a
//! load/store base, as Rd/Rn of add/sub-immediate (when Rd is not the
//! `cmp`/`cmn` alias), and as Rd/Rn of the logical-immediate forms' Rd; it
//! means XZR/WZR everywhere else (shifted-register forms, move-wide, Rt,
//! multiplies, csel, cbz/tbz). Each decoder below picks the naming per field.

// ── formatting helpers ──────────────────────────────────────────────────────

/// Condition-code names, indexed by the 4-bit `cond` field. Matches
/// `encode::cond_code` (which also accepts the `cs`/`cc` synonyms we never
/// print).
const COND: [&str; 16] = [
    "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

/// Shift-kind names, indexed by the 2-bit `shift` field.
const SHIFT: [&str; 4] = ["lsl", "lsr", "asr", "ror"];

/// Sign-extend the low `bits` of `v`.
fn sext(v: u32, bits: u32) -> i64 {
    ((v as i64) << (64 - bits)) >> (64 - bits)
}

/// A pc-relative byte displacement, printed the way the callers of `dis`
/// consume it: `+0x14` / `-0x8` (the caller adds absolute context).
fn rel(disp: i64) -> String {
    if disp < 0 {
        format!("-0x{:x}", -disp)
    } else {
        format!("+0x{disp:x}")
    }
}

/// Name a general-purpose register field. `r31_is_sp` encodes the positional
/// rule from the module doc: register 31 prints as `sp`/`wsp` where the ISA
/// reads it as the stack pointer, `xzr`/`wzr` where it is the zero register.
fn gp(sf: u32, num: u32, r31_is_sp: bool) -> String {
    match (sf, num) {
        (1, 31) if r31_is_sp => "sp".to_string(),
        (0, 31) if r31_is_sp => "wsp".to_string(),
        (1, 31) => "xzr".to_string(),
        (0, 31) => "wzr".to_string(),
        (1, n) => format!("x{n}"),
        (_, n) => format!("w{n}"),
    }
}

// ── public API ──────────────────────────────────────────────────────────────

/// Disassemble one instruction word. Unrecognized words yield
/// `.word 0x????????` — never a guess (docs/DEBUGGER.md §4.4).
pub fn disasm_word(word: u32) -> String {
    decode(word).unwrap_or_else(|| format!(".word 0x{word:08x}"))
}

/// Disassemble a code slice, one line per word:
/// `+0xOFF  0xWORD  mnemonic operands`, with `  <== HERE` appended on the
/// line containing `mark_off` (byte offset into `code`). Trailing bytes that
/// do not fill a word are reported, not dropped — a partial word in a blob is
/// itself a finding.
pub fn disasm_slice(code: &[u8], mark_off: Option<usize>) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut off = 0usize;
    for chunk in code.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)"));
        let _ = write!(out, "+0x{off:04x}  0x{word:08x}  {}", disasm_word(word));
        if mark_off.is_some_and(|m| (off..off + 4).contains(&m)) {
            out.push_str("  <== HERE");
        }
        out.push('\n');
        off += 4;
    }
    let rem = code.len() % 4;
    if rem != 0 {
        let _ = writeln!(out, "+0x{off:04x}  .incomplete ({rem} trailing byte(s))");
    }
    out
}

// ── the decoder ─────────────────────────────────────────────────────────────

/// Family dispatch. Field masks are cross-checked against the base words in
/// `vendor/wfasm/a64/encode.rs` (and `jasm_assembler.rs` for the branch and
/// `ldr`-literal forms it hand-encodes); families are mutually exclusive by
/// mask, so order only groups related forms.
fn decode(word: u32) -> Option<String> {
    decode_fixed(word)
        .or_else(|| decode_branch(word))
        .or_else(|| decode_move_wide(word))
        .or_else(|| decode_addsub_imm(word))
        .or_else(|| decode_addsub_shifted(word))
        .or_else(|| decode_logical_shifted(word))
        .or_else(|| decode_logical_imm(word))
        .or_else(|| decode_bitfield(word))
        .or_else(|| decode_csel(word))
        .or_else(|| decode_data3(word))
        .or_else(|| decode_ldst_pair(word))
        .or_else(|| decode_ldst(word))
        .or_else(|| decode_fp_scalar(word))
        .or_else(|| decode_fp_vector(word))
}

/// SIMD NEON fast-path (`docs/SIMD.md`): the 128-bit vector forms `emit`'s
/// `Ir::VecArith` lowering produces — the `.2d` elementwise arithmetic
/// (`fadd`/`fsub`/`fmul`/`fdiv`), the `q`-register `ldr`/`str` (unboxing the
/// two lanes / storing the boxed result), and the `umov Xd, Vn.d[i]` that
/// marshals a lane into a GPR for the box slow-path stub. Only these forms
/// decode; any other NEON encoding keeps the honest `.word` fallback.
fn decode_fp_vector(word: u32) -> Option<String> {
    let rd = word & 0x1F;
    let rn = (word >> 5) & 0x1F;
    let rm = (word >> 16) & 0x1F;
    // 3-same FP arithmetic, arrangement `.2d` (Q=1 bit30, sz=1 bit22) or `.4s`
    // (Q=1, sz=0). The mask keeps the op/U/sz/Q signature, clearing Rm/Rn/Rd.
    // Bases from `vendor::wfasm::a64::encode` (`0x…D400`/`…FC00` | Q | sz): the
    // `.4s` forms drop the sz bit (0x0040_0000) that the `.2d` forms carry.
    let arith = match word & 0xFFE0_FC00 {
        0x4E60_D400 => Some(("fadd", "2d")),
        0x4EE0_D400 => Some(("fsub", "2d")),
        0x6E60_DC00 => Some(("fmul", "2d")),
        0x6E60_FC00 => Some(("fdiv", "2d")),
        0x4E20_D400 => Some(("fadd", "4s")),
        0x4EA0_D400 => Some(("fsub", "4s")),
        0x6E20_DC00 => Some(("fmul", "4s")),
        0x6E20_FC00 => Some(("fdiv", "4s")),
        // SIMD Int32x4: integer 3-same `.4s` (Q=1, size=10) — `add/sub/mul`
        // (base `0x…8400`/`0x…9C00` | Q | size); no vector integer divide.
        0x4EA0_8400 => Some(("add", "4s")),
        0x6EA0_8400 => Some(("sub", "4s")),
        0x4EA0_9C00 => Some(("mul", "4s")),
        _ => None,
    };
    if let Some((mnem, arr)) = arith {
        return Some(format!("{mnem} v{rd}.{arr}, v{rn}.{arr}, v{rm}.{arr}"));
    }
    // `ldr`/`str q`, 128-bit unsigned scaled offset (scale 16): 0x3DC0/0x3D80.
    if word & 0xFFC0_0000 == 0x3DC0_0000 {
        let imm = ((word >> 10) & 0xFFF) * 16;
        return Some(format!("ldr q{rd}, [x{rn}, #{imm}]"));
    }
    if word & 0xFFC0_0000 == 0x3D80_0000 {
        let imm = ((word >> 10) & 0xFFF) * 16;
        return Some(format!("str q{rd}, [x{rn}, #{imm}]"));
    }
    // umov Xd, Vn.d[i]: base 0x4E00_3C00; imm5 (bits[20:16]) = 0b01000 for a
    // D lane, with the index in bit4 (imm5 = 8 + 16*index). `rm` above IS the
    // imm5 field here (same bit position).
    if word & 0xFFE0_FC00 == 0x4E00_3C00 && rm & 0xF == 0x8 {
        let index = (rm >> 4) & 1;
        return Some(format!("umov x{rd}, v{rn}.d[{index}]"));
    }
    None
}

/// Float fast-path (`docs/float_fastpath_design.md`): the scalar
/// double-precision FP forms `emit`'s FUnbox/FBox/FArith/FCmp lowerings
/// produce — 2-source arithmetic, `fcmp`, `fmov` gpr↔fp, `scvtf`, and the
/// unsigned-scaled-offset FP `ldr`/`str`. Only ftype=01 (double) decodes;
/// anything else keeps the honest `.word` fallback.
fn decode_fp_scalar(word: u32) -> Option<String> {
    let rd = word & 0x1F;
    let rn = (word >> 5) & 0x1F;
    let rm = (word >> 16) & 0x1F;
    // 2-source: 0x1E60_0800 | opc<<12 (double ftype already folded in).
    if word & 0xFFE0_8C00 == 0x1E60_0800 {
        let mnem = match (word >> 12) & 0xF {
            0 => "fmul",
            1 => "fdiv",
            2 => "fadd",
            3 => "fsub",
            4 => "fmax",
            5 => "fmin",
            6 => "fmaxnm",
            7 => "fminnm",
            8 => "fnmul",
            _ => return None,
        };
        return Some(format!("{mnem} d{rd}, d{rn}, d{rm}"));
    }
    // fcmp/fcmpe dN, dM: 0x1E60_2000 (+0x10 for fcmpe; +8 for #0.0 form).
    if word & 0xFFE0_FC07 == 0x1E60_2000 {
        let e = if word & 0x10 != 0 { "fcmpe" } else { "fcmp" };
        if word & 0x8 != 0 {
            return Some(format!("{e} d{rn}, #0.0"));
        }
        return Some(format!("{e} d{rn}, d{rm}"));
    }
    // fmov d<-d (register): 0x1E60_4000. The single most common instruction
    // in float-heavy compiled code — every unboxed temp shuffle between the
    // d8-d15 residency registers and emit's d16/d17 scratch is one of these —
    // and it read as a bare `.word` until this arm existed, which is exactly
    // the code a float disassembly is wanted for (found reading the
    // Mandelbrot inner loop, docs/mandelbrot_walkthrough.md).
    if word & 0xFFFF_FC00 == 0x1E60_4000 {
        return Some(format!("fmov d{rd}, d{rn}"));
    }
    // fmov x<->d: 0x9E66_0000 (fmov Xd, Dn) / 0x9E67_0000 (fmov Dd, Xn).
    if word & 0xFFFF_FC00 == 0x9E66_0000 {
        return Some(format!("fmov x{rd}, d{rn}"));
    }
    if word & 0xFFFF_FC00 == 0x9E67_0000 {
        return Some(format!("fmov d{rd}, x{rn}"));
    }
    // scvtf Dd, Xn: 0x9E62_0000.
    if word & 0xFFFF_FC00 == 0x9E62_0000 {
        return Some(format!("scvtf d{rd}, x{rn}"));
    }
    // FP scalar ldr/str, 64-bit unsigned scaled offset: 0xFD40/0xFD00.
    if word & 0xFFC0_0000 == 0xFD40_0000 {
        let imm = ((word >> 10) & 0xFFF) * 8;
        return Some(format!("ldr d{rd}, [x{rn}, #{imm}]"));
    }
    if word & 0xFFC0_0000 == 0xFD00_0000 {
        let imm = ((word >> 10) & 0xFFF) * 8;
        return Some(format!("str d{rd}, [x{rn}, #{imm}]"));
    }
    // FP scalar stur/ldur, 64-bit UNSCALED imm9: 0xFC00/0xFC40, idx(11:10)=00.
    // Every float spill/reload is one of these — a frame slot is at a negative
    // offset from x29, which only the unscaled form can express. `decode_ldst`
    // deliberately rejects V=1 and handles the GP family only, so without these
    // two arms an entire float method's spill traffic reads as bare `.word`s.
    // (Mask covers bits 31:21 + the 11:10 index field, so the pre/post-index
    // forms — which emit never generates for FP — correctly fall through.)
    if word & 0xFFE0_0C00 == 0xFC40_0000 {
        let off = sext((word >> 12) & 0x1FF, 9);
        return Some(format!("ldur d{rd}, [x{rn}, #{off}]"));
    }
    if word & 0xFFE0_0C00 == 0xFC00_0000 {
        let off = sext((word >> 12) & 0x1FF, 9);
        return Some(format!("stur d{rd}, [x{rn}, #{off}]"));
    }
    None
}

/// `nop`, `ret`/`br`/`blr`, `brk #imm`.
fn decode_fixed(word: u32) -> Option<String> {
    if word == 0xD503_201F {
        return Some("nop".to_string());
    }
    // brk: 0xD4200000 | imm16<<5 (encode.rs and deopt_trap::brk_word).
    if word & 0xFFE0_001F == 0xD420_0000 {
        return Some(format!("brk #0x{:x}", (word >> 5) & 0xFFFF));
    }
    // br/blr/ret: 0xD61F/0xD63F/0xD65F_0000 | Rn<<5.
    let rn = (word >> 5) & 0x1F;
    match word & 0xFFFF_FC1F {
        0xD61F_0000 => Some(format!("br {}", gp(1, rn, false))),
        0xD63F_0000 => Some(format!("blr {}", gp(1, rn, false))),
        0xD65F_0000 if rn == 30 => Some("ret".to_string()),
        0xD65F_0000 => Some(format!("ret {}", gp(1, rn, false))),
        _ => None,
    }
}

/// PC-relative control flow + the `ldr`-literal pool load. Branch targets
/// print as relative byte offsets (`+0x14`) — the §4.2 dossier adds context.
fn decode_branch(word: u32) -> Option<String> {
    // b / bl: imm26.
    if word & 0x7C00_0000 == 0x1400_0000 {
        let mnem = if word >> 31 == 1 { "bl" } else { "b" };
        return Some(format!("{mnem} {}", rel(sext(word & 0x03FF_FFFF, 26) * 4)));
    }
    // b.cond: 0x54000000 | imm19<<5 | cond (bit4 must be clear).
    if word & 0xFF00_0010 == 0x5400_0000 {
        let cond = COND[(word & 0xF) as usize];
        return Some(format!(
            "b.{cond} {}",
            rel(sext((word >> 5) & 0x7FFFF, 19) * 4)
        ));
    }
    // cbz/cbnz: sf | 011010 | op | imm19<<5 | Rt.
    if word & 0x7E00_0000 == 0x3400_0000 {
        let mnem = if (word >> 24) & 1 == 1 { "cbnz" } else { "cbz" };
        return Some(format!(
            "{mnem} {}, {}",
            gp(word >> 31, word & 0x1F, false),
            rel(sext((word >> 5) & 0x7FFFF, 19) * 4)
        ));
    }
    // tbz/tbnz: b5 | 011011 | op | b40<<19 | imm14<<5 | Rt (jasm_assembler's
    // hand encoding — not in the vendored encoder's scope at all). The
    // register prints as W when the tested bit is below 32, per convention.
    if word & 0x7E00_0000 == 0x3600_0000 {
        let mnem = if (word >> 24) & 1 == 1 { "tbnz" } else { "tbz" };
        let b5 = word >> 31;
        let bit = (b5 << 5) | ((word >> 19) & 0x1F);
        return Some(format!(
            "{mnem} {}, #{bit}, {}",
            gp(b5, word & 0x1F, false),
            rel(sext((word >> 5) & 0x3FFF, 14) * 4)
        ));
    }
    // ldr Xt, <pc-rel literal>: 0x58000000 | imm19<<5 | Rt (jasm_assembler's
    // hand encoding, D3.4 — pool words are always 8 bytes, so only the
    // 64-bit form exists in the vocabulary).
    if word & 0xFF00_0000 == 0x5800_0000 {
        return Some(format!(
            "ldr {}, {}",
            gp(1, word & 0x1F, false),
            rel(sext((word >> 5) & 0x7FFFF, 19) * 4)
        ));
    }
    None
}

/// `movz`/`movn`/`movk`. Printed literally (not as `mov #imm`) so the word
/// always re-encodes identically — see the module doc's alias policy.
fn decode_move_wide(word: u32) -> Option<String> {
    if (word >> 23) & 0x3F != 0b100101 {
        return None;
    }
    let mnem = match (word >> 29) & 3 {
        0b00 => "movn",
        0b10 => "movz",
        0b11 => "movk",
        _ => return None, // opc=01 unallocated
    };
    let sf = word >> 31;
    let hw = (word >> 21) & 3;
    if sf == 0 && hw > 1 {
        return None;
    }
    let imm16 = (word >> 5) & 0xFFFF;
    let shift = if hw > 0 {
        format!(", lsl #{}", 16 * hw)
    } else {
        String::new()
    };
    Some(format!(
        "{mnem} {}, #0x{imm16:x}{shift}",
        gp(sf, word & 0x1F, false)
    ))
}

/// Add/subtract immediate, with the `cmp`/`cmn` (Rd=31, S=1) and
/// `mov to/from sp` (add #0 with SP involved) aliases.
fn decode_addsub_imm(word: u32) -> Option<String> {
    if (word >> 23) & 0x3F != 0b100010 {
        return None;
    }
    let (sf, op, s) = (word >> 31, (word >> 30) & 1, (word >> 29) & 1);
    let sh = (word >> 22) & 1;
    let imm12 = (word >> 10) & 0xFFF;
    let (rn, rd) = ((word >> 5) & 0x1F, word & 0x1F);
    let shift = if sh == 1 { ", lsl #12" } else { "" };
    if s == 1 && rd == 31 {
        // cmp/cmn Rn, #imm — Rn is SP-position here (subs Xzr, sp, #n is
        // `cmp sp, #n`), Rd=31 is the discarded XZR result.
        let mnem = if op == 1 { "cmp" } else { "cmn" };
        return Some(format!("{mnem} {}, #{imm12}{shift}", gp(sf, rn, true)));
    }
    if s == 0 && op == 0 && sh == 0 && imm12 == 0 && (rd == 31 || rn == 31) {
        // The encoder's `mov` involving SP is `add Rd, Rn, #0` (mov_reg in
        // encode.rs); ARM's preferred disassembly is `mov` exactly when SP
        // is one of the operands.
        return Some(format!("mov {}, {}", gp(sf, rd, true), gp(sf, rn, true)));
    }
    let mnem = match (op, s) {
        (0, 0) => "add",
        (0, _) => "adds",
        (_, 0) => "sub",
        _ => "subs",
    };
    // Rd is SP for the non-flag-setting forms, XZR when S=1; Rn is always SP.
    Some(format!(
        "{mnem} {}, {}, #{imm12}{shift}",
        gp(sf, rd, s == 0),
        gp(sf, rn, true),
        imm12 = imm12
    ))
}

/// Shifted-register operand tail: empty for `lsl #0`, else `, <shift> #n`.
/// A non-lsl shift with amount 0 still prints (the shift bits differ in the
/// word, so dropping it would break the re-encode guarantee).
fn shift_tail(shift: u32, amt: u32) -> String {
    if shift == 0 && amt == 0 {
        String::new()
    } else {
        format!(", {} #{amt}", SHIFT[shift as usize])
    }
}

/// Add/subtract shifted register, with `cmp`/`cmn` aliases. Register 31 is
/// XZR in every position of this form (SP forces the extended-register form,
/// which the compiler never emits — that whole form is outside the
/// vocabulary and falls back).
fn decode_addsub_shifted(word: u32) -> Option<String> {
    if word & 0x1F20_0000 != 0x0B00_0000 {
        return None;
    }
    let (sf, op, s) = (word >> 31, (word >> 30) & 1, (word >> 29) & 1);
    let shift = (word >> 22) & 3;
    if shift == 3 {
        return None; // ror is reserved for add/sub
    }
    let amt = (word >> 10) & 0x3F;
    if sf == 0 && amt > 31 {
        return None;
    }
    let (rm, rn, rd) = ((word >> 16) & 0x1F, (word >> 5) & 0x1F, word & 0x1F);
    let tail = shift_tail(shift, amt);
    if s == 1 && rd == 31 {
        let mnem = if op == 1 { "cmp" } else { "cmn" };
        return Some(format!(
            "{mnem} {}, {}{tail}",
            gp(sf, rn, false),
            gp(sf, rm, false)
        ));
    }
    let mnem = match (op, s) {
        (0, 0) => "add",
        (0, _) => "adds",
        (_, 0) => "sub",
        _ => "subs",
    };
    Some(format!(
        "{mnem} {}, {}, {}{tail}",
        gp(sf, rd, false),
        gp(sf, rn, false),
        gp(sf, rm, false)
    ))
}

/// Logical shifted register (`and`/`orr`/`eor`/`ands`), with the `mov` (orr
/// from XZR) and `tst` (ands to XZR) aliases. The N=1 inverted forms
/// (`bic`/`orn`/`eon`/`bics`/`mvn`) are never emitted and fall back.
fn decode_logical_shifted(word: u32) -> Option<String> {
    if word & 0x1F00_0000 != 0x0A00_0000 {
        return None;
    }
    if (word >> 21) & 1 == 1 {
        return None; // N=1: not in the emitted vocabulary
    }
    let sf = word >> 31;
    let opc = (word >> 29) & 3;
    let shift = (word >> 22) & 3;
    let amt = (word >> 10) & 0x3F;
    if sf == 0 && amt > 31 {
        return None;
    }
    let (rm, rn, rd) = ((word >> 16) & 0x1F, (word >> 5) & 0x1F, word & 0x1F);
    if opc == 1 && rn == 31 && shift == 0 && amt == 0 {
        // orr Rd, xzr, Rm — the encoder's register-register `mov`.
        return Some(format!("mov {}, {}", gp(sf, rd, false), gp(sf, rm, false)));
    }
    let tail = shift_tail(shift, amt);
    if opc == 3 && rd == 31 {
        return Some(format!(
            "tst {}, {}{tail}",
            gp(sf, rn, false),
            gp(sf, rm, false)
        ));
    }
    let mnem = ["and", "orr", "eor", "ands"][opc as usize];
    Some(format!(
        "{mnem} {}, {}, {}{tail}",
        gp(sf, rd, false),
        gp(sf, rn, false),
        gp(sf, rm, false)
    ))
}

/// Decode a logical (bitmask) immediate — the inverse of encode.rs's
/// `encode_bitmask`. Returns `None` for field combinations the encoder can
/// never produce (non-canonical rotations, the all-ones element).
fn decode_bitmask(sf: u32, n: u32, immr: u32, imms: u32) -> Option<u64> {
    let reg_size: u32 = if sf == 1 { 64 } else { 32 };
    if sf == 0 && n == 1 {
        return None;
    }
    // Element size: highest set bit of N:NOT(imms).
    let combined = (n << 6) | ((!imms) & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    let esize = 1u32 << len;
    if esize > reg_size {
        return None;
    }
    let s = imms & (esize - 1);
    let r = immr;
    if s == esize - 1 {
        return None; // all-ones element is not a legal bitmask
    }
    if r >= esize {
        return None; // canonical encodings keep immr < esize
    }
    // ones(s+1) rotated right by r within the element, replicated to reg_size.
    let welem: u64 = (1u64 << (s + 1)) - 1;
    let elem = if r == 0 {
        welem
    } else {
        let emask = if esize == 64 {
            u64::MAX
        } else {
            (1u64 << esize) - 1
        };
        ((welem >> r) | (welem << (esize - r))) & emask
    };
    let mut val = elem;
    let mut sz = esize;
    while sz < reg_size {
        val |= val << sz;
        sz *= 2;
    }
    Some(val)
}

/// Logical immediate (`and`/`orr`/`eor`/`ands` `#bitmask`), with the `tst`
/// alias. `orr Rd, xzr, #imm` stays literal — see the module doc's alias
/// policy. Rd is SP-position for the non-flag-setting forms (logical
/// immediates may write SP), XZR for `ands`; Rn is always XZR-position.
fn decode_logical_imm(word: u32) -> Option<String> {
    if (word >> 23) & 0x3F != 0b100100 {
        return None;
    }
    let sf = word >> 31;
    let opc = (word >> 29) & 3;
    let n = (word >> 22) & 1;
    let (immr, imms) = ((word >> 16) & 0x3F, (word >> 10) & 0x3F);
    let val = decode_bitmask(sf, n, immr, imms)?;
    let (rn, rd) = ((word >> 5) & 0x1F, word & 0x1F);
    if opc == 3 && rd == 31 {
        return Some(format!("tst {}, #0x{val:x}", gp(sf, rn, false)));
    }
    let mnem = ["and", "orr", "eor", "ands"][opc as usize];
    Some(format!(
        "{mnem} {}, {}, #0x{val:x}",
        gp(sf, rd, opc != 3),
        gp(sf, rn, false)
    ))
}

/// UBFM/SBFM with the standard alias selection (`lsl`/`lsr`/`asr` immediate,
/// `ubfx`/`sbfx`, `ubfiz`/`sbfiz`, `sxtb`/`sxth`/`sxtw`, `uxtb`/`uxth`) —
/// exactly the alias set encode.rs lowers *from*, so every print re-encodes
/// to the identical word. BFM itself (`bfi`/`bfxil`, opc=01) is never
/// emitted and falls back.
fn decode_bitfield(word: u32) -> Option<String> {
    if (word >> 23) & 0x3F != 0b100110 {
        return None;
    }
    let sf = word >> 31;
    let n = (word >> 22) & 1;
    if n != sf {
        return None;
    }
    let bits = if sf == 1 { 64u32 } else { 32 };
    let (immr, imms) = ((word >> 16) & 0x3F, (word >> 10) & 0x3F);
    if immr >= bits || imms >= bits {
        return None;
    }
    let rd = gp(sf, word & 0x1F, false);
    let rn = gp(sf, (word >> 5) & 0x1F, false);
    // The sign/zero-extend aliases print their source as a W register.
    let wn = gp(0, (word >> 5) & 0x1F, false);
    match (word >> 29) & 3 {
        // SBFM.
        0b00 => Some(if imms == bits - 1 {
            format!("asr {rd}, {rn}, #{immr}")
        } else if immr == 0 && imms == 7 {
            format!("sxtb {rd}, {wn}")
        } else if immr == 0 && imms == 15 {
            format!("sxth {rd}, {wn}")
        } else if immr == 0 && imms == 31 && sf == 1 {
            format!("sxtw {rd}, {wn}")
        } else if imms >= immr {
            format!("sbfx {rd}, {rn}, #{immr}, #{}", imms - immr + 1)
        } else {
            format!("sbfiz {rd}, {rn}, #{}, #{}", bits - immr, imms + 1)
        }),
        // UBFM.
        0b10 => Some(if imms != bits - 1 && imms + 1 == immr {
            format!("lsl {rd}, {rn}, #{}", bits - 1 - imms)
        } else if imms == bits - 1 {
            format!("lsr {rd}, {rn}, #{immr}")
        } else if immr == 0 && imms == 7 && sf == 0 {
            format!("uxtb {rd}, {wn}")
        } else if immr == 0 && imms == 15 && sf == 0 {
            format!("uxth {rd}, {wn}")
        } else if imms >= immr {
            format!("ubfx {rd}, {rn}, #{immr}, #{}", imms - immr + 1)
        } else {
            format!("ubfiz {rd}, {rn}, #{}, #{}", bits - immr, imms + 1)
        }),
        _ => None, // BFM (bfi/bfxil): not in the emitted vocabulary
    }
}

/// Conditional select (`csel`/`csinc`/`csinv`/`csneg`) with the
/// `cset`/`csetm` aliases (Rn=Rm=XZR, inverted condition).
fn decode_csel(word: u32) -> Option<String> {
    if (word >> 21) & 0xFF != 0b1101_0100 || (word >> 29) & 1 == 1 {
        return None;
    }
    let sf = word >> 31;
    let op = (word >> 30) & 1;
    let op2 = (word >> 10) & 3;
    if op2 > 1 {
        return None;
    }
    let cond = (word >> 12) & 0xF;
    let (rm, rn, rd) = ((word >> 16) & 0x1F, (word >> 5) & 0x1F, word & 0x1F);
    if rn == 31 && rm == 31 && cond < 14 {
        // cset/csetm store the *inverted* condition (encode.rs `cc(c)? ^ 1`).
        let alias = match (op, op2) {
            (0, 1) => Some("cset"),
            (1, 0) => Some("csetm"),
            _ => None,
        };
        if let Some(alias) = alias {
            return Some(format!(
                "{alias} {}, {}",
                gp(sf, rd, false),
                COND[(cond ^ 1) as usize]
            ));
        }
    }
    let mnem = match (op, op2) {
        (0, 0) => "csel",
        (0, _) => "csinc",
        (_, 0) => "csinv",
        _ => "csneg",
    };
    Some(format!(
        "{mnem} {}, {}, {}, {}",
        gp(sf, rd, false),
        gp(sf, rn, false),
        gp(sf, rm, false),
        COND[cond as usize]
    ))
}

/// Data-processing 3-source: `madd`/`msub` (with the `mul`/`mneg` aliases
/// when Ra=XZR) and the fixed-form `smulh`/`umulh`. The widening multiplies
/// (`smull`/`umull`) are never emitted and fall back.
fn decode_data3(word: u32) -> Option<String> {
    if word & 0x1F00_0000 != 0x1B00_0000 {
        return None;
    }
    // smulh/umulh: sf=1, fixed o0=0 and Ra=11111.
    let (rm, rn, rd) = ((word >> 16) & 0x1F, (word >> 5) & 0x1F, word & 0x1F);
    match word & 0xFFE0_FC00 {
        0x9B40_7C00 => {
            return Some(format!(
                "smulh {}, {}, {}",
                gp(1, rd, false),
                gp(1, rn, false),
                gp(1, rm, false)
            ))
        }
        0x9BC0_7C00 => {
            return Some(format!(
                "umulh {}, {}, {}",
                gp(1, rd, false),
                gp(1, rn, false),
                gp(1, rm, false)
            ))
        }
        _ => {}
    }
    if (word >> 21) & 7 != 0 {
        return None; // widening multiplies etc.: not in the vocabulary
    }
    let sf = word >> 31;
    let ra = (word >> 10) & 0x1F;
    let sub = (word >> 15) & 1 == 1;
    if ra == 31 {
        let mnem = if sub { "mneg" } else { "mul" };
        return Some(format!(
            "{mnem} {}, {}, {}",
            gp(sf, rd, false),
            gp(sf, rn, false),
            gp(sf, rm, false)
        ));
    }
    let mnem = if sub { "msub" } else { "madd" };
    Some(format!(
        "{mnem} {}, {}, {}, {}",
        gp(sf, rd, false),
        gp(sf, rn, false),
        gp(sf, rm, false),
        gp(sf, ra, false)
    ))
}

/// `ldp`/`stp`, GP registers only (the compiler never emits FP pairs):
/// signed imm7 offset, pre-index, post-index.
fn decode_ldst_pair(word: u32) -> Option<String> {
    if (word >> 27) & 7 != 0b101 || (word >> 26) & 1 != 0 {
        return None;
    }
    let (sf, scale) = match word >> 30 {
        0b00 => (0u32, 4i64),
        0b10 => (1, 8),
        _ => return None, // ldpsw / FP opc values: not in the vocabulary
    };
    let mnem = if (word >> 22) & 1 == 1 { "ldp" } else { "stp" };
    let off = sext((word >> 15) & 0x7F, 7) * scale;
    let (rt2, rn, rt) = ((word >> 10) & 0x1F, (word >> 5) & 0x1F, word & 0x1F);
    let regs = format!("{mnem} {}, {}", gp(sf, rt, false), gp(sf, rt2, false));
    let base = gp(1, rn, true);
    match (word >> 23) & 7 {
        0b001 => Some(format!("{regs}, [{base}], #{off}")),
        0b010 if off == 0 => Some(format!("{regs}, [{base}]")),
        0b010 => Some(format!("{regs}, [{base}, #{off}]")),
        0b011 => Some(format!("{regs}, [{base}, #{off}]!")),
        _ => None, // ldnp/stnp and reserved index modes
    }
}

/// `(mnemonic, Rt is 64-bit, access size)` for one GP load/store row.
type LdstEntry = Option<(&'static str, bool, u32)>;

/// GP load/store table indexed by `(size, opc)` — the inverse of encode.rs
/// `ldst_kind` (`None` marks prefetch/unallocated rows). FP/SIMD forms
/// (V=1) fall back at the call sites.
const LDST_GP: [[LdstEntry; 4]; 4] = [
    [
        Some(("strb", false, 1)),
        Some(("ldrb", false, 1)),
        Some(("ldrsb", true, 1)),
        Some(("ldrsb", false, 1)),
    ],
    [
        Some(("strh", false, 2)),
        Some(("ldrh", false, 2)),
        Some(("ldrsh", true, 2)),
        Some(("ldrsh", false, 2)),
    ],
    [
        Some(("str", false, 4)),
        Some(("ldr", false, 4)),
        Some(("ldrsw", true, 4)),
        None,
    ],
    [Some(("str", true, 8)), Some(("ldr", true, 8)), None, None],
];

/// Scalar GP loads/stores: unsigned scaled offset, unscaled (`ldur`/`stur`),
/// pre/post-index, and the lsl-only register-offset form.
fn decode_ldst(word: u32) -> Option<String> {
    if (word >> 27) & 7 != 0b111 || (word >> 26) & 1 != 0 {
        return None; // not a load/store, or V=1 (FP/SIMD: never emitted)
    }
    let size = (word >> 30) & 3;
    let opc = (word >> 22) & 3;
    let (mnem, rt_is_x, scale) = LDST_GP[size as usize][opc as usize]?;
    let rt = gp(rt_is_x as u32, word & 0x1F, false);
    let base = gp(1, (word >> 5) & 0x1F, true);

    // Unsigned scaled offset: bits[25:24] = 01.
    if (word >> 24) & 3 == 1 {
        let off = ((word >> 10) & 0xFFF) as i64 * scale as i64;
        return Some(if off == 0 {
            format!("{mnem} {rt}, [{base}]")
        } else {
            format!("{mnem} {rt}, [{base}, #{off}]")
        });
    }
    if (word >> 24) & 3 != 0 {
        return None;
    }
    // Register offset: bit21=1, bits[11:10] = 10. Only the `lsl` option
    // (UXTX) is in the vocabulary; the extend options fall back.
    if (word >> 21) & 1 == 1 {
        if (word >> 10) & 3 != 0b10 || (word >> 13) & 7 != 0b011 {
            return None;
        }
        let rm = gp(1, (word >> 16) & 0x1F, false);
        return Some(if (word >> 12) & 1 == 1 {
            format!("{mnem} {rt}, [{base}, {rm}, lsl #{}]", scale.ilog2())
        } else {
            format!("{mnem} {rt}, [{base}, {rm}]")
        });
    }
    // Unscaled imm9 family: idx 00 = ldur/stur, 01 = post-index, 11 = pre.
    let off = sext((word >> 12) & 0x1FF, 9);
    match (word >> 10) & 3 {
        0b00 => {
            // `ldr`→`ldur`, `strb`→`sturb`, … — the inverse of ldst_ur's
            // `mnem.replacen("ur", "r", 1)`.
            let unscaled = mnem.replacen('r', "ur", 1);
            Some(format!("{unscaled} {rt}, [{base}, #{off}]"))
        }
        0b01 => Some(format!("{mnem} {rt}, [{base}], #{off}")),
        0b11 => Some(format!("{mnem} {rt}, [{base}, #{off}]!")),
        _ => None, // unprivileged (ldtr/sttr): not in the vocabulary
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::assembler::{xr, Assembler, Cond};
    use crate::compiler::jasm_assembler::JasmAssembler;
    use crate::vendor::wfasm::a64::encode::encode;
    use crate::vendor::wfasm::a64::parse::{parse_line, Line};

    /// Assemble one instruction through the vendored text parser + encoder,
    /// returning the word. Panics on anything that isn't a single fixup-free
    /// word — the corpus and goldens only contain those.
    fn asm_word(text: &str) -> u32 {
        let Line::Insn { mnemonic, ops } = parse_line(text).expect("parse") else {
            panic!("not an instruction: {text}");
        };
        let enc = encode(&mnemonic, &ops).unwrap_or_else(|e| panic!("encode `{text}`: {e:#}"));
        assert!(enc.fixups.is_empty(), "`{text}` produced fixups");
        assert_eq!(enc.bytes.len(), 4, "`{text}` is not one word");
        u32::from_le_bytes(enc.bytes[..4].try_into().unwrap())
    }

    /// Corpus lines that are *outside* the emitted vocabulary and therefore
    /// expected to hit the `.word` fallback (docs/DEBUGGER.md §4.4: the
    /// fallback is a feature, not a gap — nothing is silently skipped, every
    /// excluded form is named here). Two exclusion rules:
    /// 1. Any form with SIMD/FP operands (`v*`/`b*`/`h*`/`s*`/`d*`/`q*`
    ///    registers) — the tier-1 compiler is integer-only.
    /// 2. The GP mnemonics below, which the MACVM compiler never emits.
    const EXCLUDED_GP_MNEMONICS: &[&str] = &[
        // add/sub with carry — no multi-word arithmetic in tier-1 code.
        "adc", "adcs", "sbc", "sbcs",
        // variable (register-amount) shifts — only immediate shifts emitted.
        "lslv", "lsrv", "asrv", "rorv",
        // BFM insert aliases (opc=01) — never emitted.
        "bfi", "bfxil", // N=1 inverted logicals — never emitted.
        "bic", "bics", "orn", "eon",  // conditional compare — never emitted.
        "ccmp", // data-processing 1-source — never emitted.
        "clz", "cls", "rbit", "rev", "rev16", "rev32", // CRC — never emitted.
        "crc32b", "crc32h", "crc32w", "crc32x", "crc32cb",
        // system / barriers / hints — never emitted (brk/nop ARE emitted and
        // are decoded, so they are absent from this list). `msr`/`mrs`
        // touch system registers (nzcv, etc.); tier-1 code never does.
        "dmb", "dsb", "isb", "mrs", "msr", "svc", "hlt", "hvc", "smc", "brk", "sev", "sevl", "wfe",
        "wfi", "yield", "nop", "esb", "psb", "clrex",
        // atomics and exclusives — the VM is single-threaded on the mutator.
        "ldadd", "ldadda", "ldaddal", "ldaddl", "ldclr", "ldeor", "ldset", "ldsmax", "ldumin",
        "ldar", "ldaxr", "ldxr", "stlxr", "stxr", "swp", "swpal", "cas", "casal",
        // integer division — Smalltalk division goes through runtime calls.
        "sdiv", "udiv",
    ];

    /// Rule 1 above: does the corpus form use SIMD/FP operands?
    fn is_simd_or_fp_form(asm: &str) -> bool {
        let rest = asm.split_once(char::is_whitespace).map_or("", |(_, r)| r);
        rest.split(|c: char| c.is_whitespace() || matches!(c, ',' | '[' | ']' | '{' | '}' | '-'))
            .filter(|t| !t.is_empty() && !t.starts_with('#'))
            .any(|t| {
                let mut ch = t.chars();
                matches!(ch.next(), Some('v' | 'b' | 'h' | 's' | 'd' | 'q'))
                    && ch.next().is_some_and(|c| c.is_ascii_digit())
            })
    }

    fn unhex_word(hex: &str) -> u32 {
        assert_eq!(hex.len(), 8, "not a single 4-byte word: {hex}");
        let bytes: Vec<u8> = (0..8)
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect();
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    /// (a) Round-trip over the frozen encoder corpus: every in-vocabulary
    /// form must decode (no fallback) AND re-encode byte-identically through
    /// the vendored parser + encoder; every fallback must be on the
    /// documented exclusion list. The corpus has no relocation-carrying
    /// (branch/adrp) lines at all, so no masking is needed; the branch forms
    /// are covered by the jasm goldens below instead.
    #[test]
    fn disasm_corpus_round_trip() {
        let corpus = include_str!("../vendor/wfasm/corpus/aarch64.tsv");
        let mut decoded = 0usize;
        let mut excluded = 0usize;
        for (i, line) in corpus.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut fields = line.splitn(3, '\t');
            let asm = fields.next().unwrap();
            let hex = fields
                .next()
                .unwrap_or_else(|| panic!("corpus line {}: no code field", i + 1));
            let reloc = fields.next().unwrap_or("");
            assert!(
                reloc.is_empty(),
                "corpus line {} unexpectedly carries a reloc: {asm}",
                i + 1
            );
            let word = unhex_word(hex);
            let text = disasm_word(word);
            let mnemonic = asm.split_whitespace().next().unwrap();
            if text.starts_with(".word") {
                assert!(
                    is_simd_or_fp_form(asm) || EXCLUDED_GP_MNEMONICS.contains(&mnemonic),
                    "corpus form `{asm}` (0x{word:08x}) not decoded and not on \
                     the documented exclusion list"
                );
                excluded += 1;
                continue;
            }
            let word2 = asm_word(&text);
            assert_eq!(
                word2, word,
                "round-trip failed on corpus form `{asm}`: printed `{text}`, \
                 which re-encodes to 0x{word2:08x} instead of 0x{word:08x}"
            );
            decoded += 1;
        }
        // Keep the gate meaningful: the GP subset of the corpus is ~400 forms.
        assert!(
            decoded >= 350,
            "suspiciously few corpus forms decoded ({decoded}; {excluded} excluded)"
        );
    }

    /// (b) Direct goldens: representative words hand-assembled through the
    /// vendored encoder, asserting the exact disassembly strings (including
    /// the alias prints and the register-31 naming rules).
    #[test]
    fn disasm_goldens() {
        for (asm, expect) in [
            // add/sub immediate + the sp/cmp aliases.
            ("add x17, x19, #6", "add x17, x19, #6"),
            ("sub sp, sp, #128", "sub sp, sp, #128"),
            ("add x0, x1, #1, lsl #12", "add x0, x1, #1, lsl #12"),
            ("mov sp, x29", "mov sp, x29"),
            ("mov x2, sp", "mov x2, sp"),
            ("cmp x11, #7", "cmp x11, #7"),
            // register forms + shifted-register cmp (emitted by emit.rs's
            // smi-overflow checks).
            ("adds x5, x6, x7", "adds x5, x6, x7"),
            ("cmp x16, x17, asr #63", "cmp x16, x17, asr #63"),
            ("mov x0, x28", "mov x0, x28"),
            // logicals: bitmask immediates + tst aliases.
            ("and x19, x2, #3", "and x19, x2, #0x3"),
            ("tst x0, #3", "tst x0, #0x3"),
            ("orr x1, xzr, #0xff", "orr x1, xzr, #0xff"),
            ("eor x3, x4, x5", "eor x3, x4, x5"),
            // move-wide.
            ("movz x9, #7", "movz x9, #0x7"),
            ("movz x16, #0", "movz x16, #0x0"),
            ("movk x0, #0xabcd, lsl #32", "movk x0, #0xabcd, lsl #32"),
            ("movn x0, #0", "movn x0, #0x0"),
            // shifts / multiplies.
            ("asr x16, x2, #2", "asr x16, x2, #2"),
            ("lsl x1, x2, #3", "lsl x1, x2, #3"),
            ("mul x17, x16, x3", "mul x17, x16, x3"),
            ("smulh x16, x16, x3", "smulh x16, x16, x3"),
            // conditional select.
            ("csel x0, x16, x17, eq", "csel x0, x16, x17, eq"),
            ("cset x0, ne", "cset x0, ne"),
            // loads/stores: scaled, unscaled, pair pre/post, xzr store.
            ("ldr x0, [x1, #8]", "ldr x0, [x1, #8]"),
            ("ldr x19, [x20]", "ldr x19, [x20]"),
            ("str w3, [x2, #4]", "str w3, [x2, #4]"),
            ("ldur x17, [x0, #7]", "ldur x17, [x0, #7]"),
            ("stur x2, [x19, #15]", "stur x2, [x19, #15]"),
            ("strb wzr, [x20]", "strb wzr, [x20]"),
            ("stp x29, x30, [sp, #-16]!", "stp x29, x30, [sp, #-16]!"),
            ("ldp x29, x30, [sp], #96", "ldp x29, x30, [sp], #96"),
            ("stp x19, x20, [sp, #16]", "stp x19, x20, [sp, #16]"),
            // indirect branches + brk/nop.
            ("br x17", "br x17"),
            ("blr x16", "blr x16"),
            ("ret", "ret"),
            ("brk #0xde00", "brk #0xde00"),
            ("nop", "nop"),
            // SIMD NEON fast-path (docs/SIMD.md): the exact forms
            // emit::emit_vec2arith produces — .2d arithmetic, q ldr/str, and
            // the lane umov. Round-trips through the vendored encoder.
            ("fadd v16.2d, v16.2d, v17.2d", "fadd v16.2d, v16.2d, v17.2d"),
            ("fsub v16.2d, v16.2d, v17.2d", "fsub v16.2d, v16.2d, v17.2d"),
            ("fmul v16.2d, v16.2d, v17.2d", "fmul v16.2d, v16.2d, v17.2d"),
            ("fdiv v16.2d, v16.2d, v17.2d", "fdiv v16.2d, v16.2d, v17.2d"),
            ("fadd v16.4s, v16.4s, v17.4s", "fadd v16.4s, v16.4s, v17.4s"),
            ("fsub v16.4s, v16.4s, v17.4s", "fsub v16.4s, v16.4s, v17.4s"),
            ("fmul v16.4s, v16.4s, v17.4s", "fmul v16.4s, v16.4s, v17.4s"),
            ("fdiv v16.4s, v16.4s, v17.4s", "fdiv v16.4s, v16.4s, v17.4s"),
            // SIMD Int32x4 integer .4s (add/sub/mul; no vector divide).
            ("add v16.4s, v16.4s, v17.4s", "add v16.4s, v16.4s, v17.4s"),
            ("sub v16.4s, v16.4s, v17.4s", "sub v16.4s, v16.4s, v17.4s"),
            ("mul v16.4s, v16.4s, v17.4s", "mul v16.4s, v16.4s, v17.4s"),
            ("ldr q16, [x17, #16]", "ldr q16, [x17, #16]"),
            ("str q16, [x19, #16]", "str q16, [x19, #16]"),
            ("umov x0, v16.d[0]", "umov x0, v16.d[0]"),
            ("umov x1, v16.d[1]", "umov x1, v16.d[1]"),
            // FP scalar spill traffic (emit::fp_spill_access). A frame slot is
            // NEGATIVE from x29, so the encoder picks the unscaled STUR/LDUR
            // form — note the asymmetry these pairs pin down: the vendored
            // encoder only accepts `ldr`/`str` as the FP mnemonic and chooses
            // scaled-vs-unscaled from the offset itself, so the input says
            // `str` and the correct disassembly says `stur`.
            ("str d16, [x29, #-80]", "stur d16, [x29, #-80]"),
            ("ldr d8, [x29, #-32]", "ldur d8, [x29, #-32]"),
            ("str d15, [x29, #-256]", "stur d15, [x29, #-256]"),
            ("ldr d0, [x16, #0]", "ldr d0, [x16, #0]"),
        ] {
            assert_eq!(disasm_word(asm_word(asm)), expect, "golden for `{asm}`");
        }
    }

    /// (b, continued) The pc-relative forms the vendored encoder cannot
    /// produce standalone: label branches and `ldr`-literal, built through
    /// `JasmAssembler` exactly the way the compiler builds them.
    #[test]
    fn disasm_branch_goldens() {
        const NOP: u32 = 0xD503_201F;
        let word_at = |code: &[u8], off: usize| -> u32 {
            u32::from_le_bytes(code[off..off + 4].try_into().unwrap())
        };

        // Forward: b.eq +0x14, cbz x17 +0x18, tbz x3 (bit 7) +0x10.
        let mut a = JasmAssembler::new();
        let l = a.new_label();
        a.b_cond(Cond::Eq, l); // +0x00: 5 words to the bind point
        a.cbz(xr(17), l); // +0x04
        a.tbz(xr(3), 7, l); // +0x08
        a.b(l); // +0x0c
        a.emit_u32(NOP); // +0x10
        a.bind(l); // +0x14
        let blob = a.finish();
        assert_eq!(disasm_word(word_at(&blob.code, 0)), "b.eq +0x14");
        assert_eq!(disasm_word(word_at(&blob.code, 4)), "cbz x17, +0x10");
        // tbz on a low bit prints the conventional W register name.
        assert_eq!(disasm_word(word_at(&blob.code, 8)), "tbz w3, #7, +0xc");
        assert_eq!(disasm_word(word_at(&blob.code, 12)), "b +0x8");

        // Backward: tbnz on a high bit (x-form), cbnz, b.
        let mut a = JasmAssembler::new();
        let l = a.new_label();
        a.bind(l);
        a.emit_u32(NOP);
        a.tbnz(xr(3), 40, l); // at +0x04, target -0x4
        a.cbnz(xr(0), l); // at +0x08, target -0x8
        a.b(l); // at +0x0c, target -0xc
        let blob = a.finish();
        assert_eq!(disasm_word(word_at(&blob.code, 4)), "tbnz x3, #40, -0x4");
        assert_eq!(disasm_word(word_at(&blob.code, 8)), "cbnz x0, -0x8");
        assert_eq!(disasm_word(word_at(&blob.code, 12)), "b -0xc");

        // The `bl .` self-branch placeholder (bl_patchable) and ldr-literal.
        let mut a = JasmAssembler::new();
        let lit = a.literal_u64(0xDEAD_BEEF, None);
        a.ldr_literal(xr(16), lit); // +0x00; pool lands at +0x10
        a.emit_u32(NOP);
        a.emit_u32(NOP);
        a.emit_u32(NOP);
        let blob = a.finish();
        assert_eq!(blob.literal_off, 16);
        assert_eq!(disasm_word(word_at(&blob.code, 0)), "ldr x16, +0x10");
        assert_eq!(disasm_word(0x9400_0000), "bl +0x0");
    }

    /// (c) Unrecognized words fall back to `.word` — never a guess
    /// (docs/DEBUGGER.md §4.4: the fallback line is itself a finding).
    #[test]
    fn disasm_fallback() {
        // Not valid A64 at all.
        assert_eq!(disasm_word(0x0000_0000), ".word 0x00000000");
        assert_eq!(disasm_word(0xFFFF_FFFF), ".word 0xffffffff");
        // Valid A64, but outside the emitted vocabulary.
        assert_eq!(disasm_word(asm_word("sdiv x0, x1, x2")), ".word 0x9ac20c20");
        assert_eq!(disasm_word(asm_word("dmb ish")), ".word 0xd5033bbf");
    }

    /// `disasm_slice` line format: offset, word, text, `<== HERE` marker on
    /// the marked word, honest reporting of a truncated tail.
    #[test]
    fn disasm_slice_format() {
        let mut code = Vec::new();
        code.extend_from_slice(&asm_word("stp x29, x30, [sp, #-16]!").to_le_bytes());
        code.extend_from_slice(&asm_word("mov x29, sp").to_le_bytes());
        code.extend_from_slice(&asm_word("ret").to_le_bytes());
        let out = disasm_slice(&code, Some(4));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines,
            [
                "+0x0000  0xa9bf7bfd  stp x29, x30, [sp, #-16]!",
                "+0x0004  0x910003fd  mov x29, sp  <== HERE",
                "+0x0008  0xd65f03c0  ret",
            ]
        );

        // No marker → no HERE; trailing bytes are reported, not dropped.
        code.push(0xAB);
        let out = disasm_slice(&code, None);
        assert!(!out.contains("<== HERE"));
        assert!(out.ends_with("+0x000c  .incomplete (1 trailing byte(s))\n"));
    }

    /// (e) The most direct proof: disassemble a REAL compiled nmethod's
    /// own code region and assert no `.word` fallback appears — a fallback
    /// here would name an instruction the emitter genuinely produced that
    /// this decoder can't, exactly the gap the corpus test approximates
    /// but this exercises against live codegen output.
    // WINVM: this one disassembles whatever the DRIVER emitted, so once
    // the x64 back end was wired in (Phase 3v) it started decoding x86-64
    // bytes as AArch64 — which is how it caught the seam going live. The
    // other tests in this module feed the decoder fixed A64 words and
    // stay host-independent; only this one is target-coupled.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn round_trips_a_real_compiled_method() {
        use crate::runtime::vm_state::{VmOptions, VmState};
        let mut vm = VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Threshold(1),
        });
        let mut b = crate::bytecode::BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(b"foo");
        let m = b.finish(&mut vm, sel, 0, 0);
        let klass = vm.universe.object_klass;
        crate::runtime::lookup::install_method(&mut vm, klass, sel, m);
        let id = crate::compiler::driver::compile_method(&mut vm, klass, m)
            .expect("trivial method compiles");
        let nm = vm.code_table.get(id).unwrap();
        // Code region only (up to the literal pool). The final word may be
        // a zero pad to the 8-byte pool boundary — allow exactly that one.
        let code = &nm.code.as_bytes()[..nm.literal_off as usize];
        let listing = disasm_slice(code, None);
        for line in listing.lines() {
            if line.contains(".word") {
                assert!(
                    line.contains("0x00000000"),
                    "decoder gap in real emitted code: {line}\n(full listing:\n{listing})"
                );
            }
        }
    }
}
