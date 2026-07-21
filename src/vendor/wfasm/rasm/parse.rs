// Vendored from JASM (wfasm), https://github.com/albanread/JASM  commit f2177391538cbede0c8cfcaa29bd3303ae421d0c
// Original path: rust/src/rasm/parse.rs.  License: MIT (see LICENSE-JASM in this
// directory; Copyright (c) 2026 alban read).
// Local modifications are marked with `// WINVM:` comments — keep the diff
// against upstream minimal so re-vendoring stays mechanical.
// WINVM: `crate::backend`/`crate::rasm` -> `crate::vendor::wfasm::*` (modules
// moved under WINVM's own tree).

//! Parse one line of assembled MC-flavour Intel-syntax assembly into a [`Line`].
//!
//! The input is what `asm::emit` produces: one statement per line, comments
//! already stripped, macros resolved, spacing normalized (`mov rax, rcx`,
//! `[rbp - 8]`, `qword ptr [rcx]`, `lea r8, [rip + label]`, `.globl name`,
//! `name:`, `name$$local:`). Numbers are decimal, `0x..` hex, optionally signed,
//! with `_` separators.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    R8,
    R16,
    R32,
    R64,
    Xmm,
    Ymm,
    Zmm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub class: RegClass,
    /// Architectural register number 0..=15 (rax=0, rcx=1, â€¦ r15=15; xmm0..15).
    pub num: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSize {
    Byte,
    Word,
    Dword,
    Qword,
    /// `xmmword ptr` â€” 16-byte SSE operand. A size hint only; SSE opcodes carry
    /// their own operand size, so this never affects encoding (used by LET's
    /// `andpd/orpd/xorpd xmm, xmmword ptr [rip + mask]`).
    Xmmword,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem {
    pub size: Option<MemSize>,
    pub base: Option<Reg>,
    pub index: Option<Reg>,
    pub scale: u8, // 1, 2, 4, 8
    pub disp: i64,
    /// `[rip + sym]` â€” RIP-relative to a symbol (mutually exclusive with base/index).
    pub rip_sym: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    Mem(Mem),
    Imm(i64),
    /// A bare symbol operand â€” a branch/call target or `lea`-rip target.
    Sym(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Directive {
    IntelSyntax,
    Text,
    Globl(String),
    /// `.quad a, b, ...` â€” one or more 8-byte little-endian values. LET's
    /// SSE masks emit two (`.quad 0x8000..., 0x0000...`).
    Quad(Vec<i64>),
    Byte(u8),
    Zero(usize),
    /// `.align`/`.balign N` (byte alignment).
    Align(u32),
    /// `.p2align N` (power-of-two alignment).
    P2align(u32),
    Ascii(Vec<u8>, bool), // (bytes, nul-terminated)
    /// Any directive we don't special-case (kept verbatim for diagnostics).
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    Empty,
    Directive(Directive),
    Label(String),
    Insn { mnemonic: String, ops: Vec<Operand> },
}

/// Strip a trailing `#`/`;` end-of-line comment (LET codegen annotates lines
/// like `movabs rax, 0x.. # &sin`; the kernel front-end pre-strips its `;`
/// comments, so this is a harmless no-op there). The `#`/`;` is only honored at
/// top level â€” not inside `[]` (no comment chars occur in operands anyway).
pub fn strip_comment(s: &str) -> &str {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            '#' | ';' if depth == 0 => return &s[..i],
            _ => {}
        }
    }
    s
}

/// If the (comment-stripped) line begins with `symbol:`, return
/// `(Some(symbol), rest)` where `rest` is everything after the colon. Lets the
/// assembler accept MC's combined `label: insn` / `label: .quad ...` lines â€”
/// LET codegen emits its constant-pool data labels inline.
pub fn split_leading_label(line: &str) -> (Option<&str>, &str) {
    let line = line.trim_start();
    let mut depth = 0i32;
    for (i, c) in line.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ':' if depth == 0 => {
                let name = line[..i].trim();
                return if is_symbol(name) {
                    (Some(name), line[i + 1..].trim_start())
                } else {
                    (None, line)
                };
            }
            _ => {}
        }
    }
    (None, line)
}

/// Parse a single line. Trailing/leading whitespace is ignored.
pub fn parse_line(raw: &str) -> Result<Line> {
    let line = strip_comment(raw).trim();
    if line.is_empty() {
        return Ok(Line::Empty);
    }
    if let Some(d) = line.strip_prefix('.') {
        return parse_directive(d);
    }
    // Label: `name:` (and only that on the line).
    if let Some(name) = line.strip_suffix(':') {
        let name = name.trim();
        if is_symbol(name) {
            return Ok(Line::Label(name.to_string()));
        }
    }
    // Instruction: mnemonic then comma-separated operands.
    let (mnemonic, rest) = split_mnemonic(line);
    let mut ops = Vec::new();
    for part in split_operands(rest) {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        ops.push(parse_operand(p)?);
    }
    Ok(Line::Insn { mnemonic: mnemonic.to_ascii_lowercase(), ops })
}

fn parse_directive(d: &str) -> Result<Line> {
    let mut it = d.split_whitespace();
    let head = it.next().unwrap_or("");
    let arg = d[head.len()..].trim();
    let dir = match head {
        "intel_syntax" => Directive::IntelSyntax,
        "text" => Directive::Text,
        "globl" | "global" => Directive::Globl(arg.to_string()),
        "quad" => Directive::Quad(
            arg.split(',').map(|v| parse_int(v.trim())).collect::<Result<Vec<_>>>()?,
        ),
        "byte" => Directive::Byte(parse_int(arg)? as u8),
        "zero" | "skip" | "space" => Directive::Zero(parse_int(arg)? as usize),
        "align" | "balign" => Directive::Align(parse_int(arg)? as u32),
        "p2align" => Directive::P2align(parse_int(arg)? as u32),
        "ascii" => Directive::Ascii(parse_string(arg)?, false),
        "asciz" | "string" => Directive::Ascii(parse_string(arg)?, true),
        _ => Directive::Other(d.to_string()),
    };
    Ok(Line::Directive(dir))
}

fn split_mnemonic(line: &str) -> (&str, &str) {
    match line.find(|c: char| c.is_whitespace()) {
        Some(i) => (&line[..i], line[i..].trim_start()),
        None => (line, ""),
    }
}

/// Split the operand list on top-level commas (commas inside `[]` stay).
fn split_operands(rest: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&rest[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start <= rest.len() {
        out.push(&rest[start..]);
    }
    out
}

fn parse_operand(p: &str) -> Result<Operand> {
    // Size-prefixed memory: "qword ptr [..]" etc.
    let (size, rest) = strip_size_prefix(p);
    let rest = rest.trim();
    if rest.starts_with('[') {
        return Ok(Operand::Mem(parse_mem(rest, size)?));
    }
    if size.is_some() {
        bail!("size prefix on non-memory operand: `{p}`");
    }
    if rest.starts_with('\'') {
        return Ok(Operand::Imm(parse_char(rest)?));
    }
    if let Some(reg) = parse_reg(rest) {
        return Ok(Operand::Reg(reg));
    }
    if looks_like_number(rest) {
        return Ok(Operand::Imm(parse_int(rest)?));
    }
    if let Some(v) = eval_const(rest) {
        return Ok(Operand::Imm(v));
    }
    if is_symbol(rest) {
        return Ok(Operand::Sym(rest.to_string()));
    }
    bail!("cannot parse operand: `{p}`")
}

fn strip_size_prefix(p: &str) -> (Option<MemSize>, &str) {
    let lower = p.trim_start();
    for (kw, sz) in [
        ("byte ptr", MemSize::Byte),
        ("word ptr", MemSize::Word),
        ("dword ptr", MemSize::Dword),
        ("qword ptr", MemSize::Qword),
        ("xmmword ptr", MemSize::Xmmword),
    ] {
        if let Some(rest) = strip_ci_prefix(lower, kw) {
            return (Some(sz), rest);
        }
    }
    (None, p)
}

fn strip_ci_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Parse `[ base (+ index*scale)? (+/- disp)? ]` or `[rip + sym]`.
fn parse_mem(s: &str, size: Option<MemSize>) -> Result<Mem> {
    let inner = s
        .trim()
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .ok_or_else(|| anyhow::anyhow!("malformed memory operand: `{s}`"))?
        .trim();

    let mut mem = Mem { size, base: None, index: None, scale: 1, disp: 0, rip_sym: None };

    // Tokenize into + / - separated terms (each term: reg, reg*scale, number, or sym).
    let mut sign = 1i64;
    let mut term = String::new();
    let mut terms: Vec<(i64, String)> = Vec::new();
    for c in inner.chars() {
        match c {
            '+' => {
                terms.push((sign, term.trim().to_string()));
                term.clear();
                sign = 1;
            }
            '-' => {
                terms.push((sign, term.trim().to_string()));
                term.clear();
                sign = -1;
            }
            _ => term.push(c),
        }
    }
    terms.push((sign, term.trim().to_string()));

    for (sgn, t) in terms {
        if t.is_empty() {
            continue;
        }
        if t.eq_ignore_ascii_case("rip") {
            // [rip + sym] â€” the sym is another term.
            mem.rip_sym = Some(String::new()); // marker; filled by the sym term
            continue;
        }
        if let Some((lhs, rhs)) = t.split_once('*') {
            let (lhs, rhs) = (lhs.trim(), rhs.trim());
            if let Some(reg) = parse_reg(lhs) {
                mem.index = Some(reg);
                mem.scale = parse_int(rhs)? as u8;
            } else if let Some(reg) = parse_reg(rhs) {
                // `scale*reg` form.
                mem.index = Some(reg);
                mem.scale = parse_int(lhs)? as u8;
            } else {
                // Constant product, e.g. `2*8` (a displacement, not index*scale).
                mem.disp += sgn * parse_int(lhs)? * parse_int(rhs)?;
            }
            continue;
        }
        if let Some(reg) = parse_reg(&t) {
            if mem.base.is_none() {
                mem.base = Some(reg);
            } else {
                mem.index = Some(reg);
            }
            continue;
        }
        if looks_like_number(&t) {
            mem.disp += sgn * parse_int(&t)?;
            continue;
        }
        // A bare symbol inside [] is the rip target.
        if is_symbol(&t) {
            mem.rip_sym = Some(t.clone());
            continue;
        }
        bail!("unparseable memory term `{t}` in `{s}`");
    }
    // If we saw `rip` but no symbol term, that's malformed.
    if matches!(mem.rip_sym.as_deref(), Some("")) {
        bail!("[rip + â€¦] with no symbol in `{s}`");
    }
    Ok(mem)
}

// â”€â”€ registers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn parse_reg(s: &str) -> Option<Reg> {
    let s = s.trim().to_ascii_lowercase();
    reg_table(&s)
}

fn reg_table(s: &str) -> Option<Reg> {
    // 64-bit
    const R64: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    const R32: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d",
        "r12d", "r13d", "r14d", "r15d",
    ];
    const R16: [&str; 16] = [
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w",
        "r13w", "r14w", "r15w",
    ];
    const R8: [&str; 16] = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
        "r13b", "r14b", "r15b",
    ];
    for (i, n) in R64.iter().enumerate() {
        if s == *n {
            return Some(Reg { class: RegClass::R64, num: i as u8 });
        }
    }
    for (i, n) in R32.iter().enumerate() {
        if s == *n {
            return Some(Reg { class: RegClass::R32, num: i as u8 });
        }
    }
    for (i, n) in R16.iter().enumerate() {
        if s == *n {
            return Some(Reg { class: RegClass::R16, num: i as u8 });
        }
    }
    for (i, n) in R8.iter().enumerate() {
        if s == *n {
            return Some(Reg { class: RegClass::R8, num: i as u8 });
        }
    }
    // Vector registers â€” 0..=31 in AVX-512 (xmm/ymm16-31 and zmm need EVEX).
    for (prefix, class) in [
        ("xmm", RegClass::Xmm),
        ("ymm", RegClass::Ymm),
        ("zmm", RegClass::Zmm),
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            if let Ok(n) = rest.parse::<u8>() {
                if n < 32 {
                    return Some(Reg { class, num: n });
                }
            }
        }
    }
    None
}

// â”€â”€ literals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn looks_like_number(s: &str) -> bool {
    let t = s.strip_prefix(['-', '+']).unwrap_or(s);
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    !t.is_empty() && t.chars().all(|c| c.is_ascii_hexdigit() || c == '_')
        && (s.starts_with("0x") || s.starts_with("0X") || s.starts_with(['-', '+'])
            || t.chars().all(|c| c.is_ascii_digit() || c == '_'))
}

/// Evaluate a constant integer expression of numbers with `*`, `+`, `-`
/// (no parens; `*` binds tighter than `+`/`-`). Returns `None` if any token
/// isn't a number (e.g. a bare symbol). Matches what MC folds in operands like
/// `2*8` or `2*8 + 1`.
fn eval_const(s: &str) -> Option<i64> {
    enum T {
        N(i64),
        Plus,
        Minus,
        Times,
    }
    let chars: Vec<char> = s.trim().chars().collect();
    if chars.is_empty() {
        return None;
    }
    let mut toks: Vec<T> = Vec::new();
    let mut i = 0;
    let mut prev_value = false;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if prev_value && (c == '+' || c == '-' || c == '*') {
            toks.push(match c {
                '+' => T::Plus,
                '-' => T::Minus,
                _ => T::Times,
            });
            prev_value = false;
            i += 1;
            continue;
        }
        // A number (optional leading unary sign).
        let start = i;
        if c == '+' || c == '-' {
            i += 1;
        }
        while i < chars.len()
            && (chars[i].is_ascii_hexdigit() || chars[i] == 'x' || chars[i] == 'X' || chars[i] == '_')
        {
            i += 1;
        }
        let numstr: String = chars[start..i].iter().collect();
        toks.push(T::N(parse_int(&numstr).ok()?));
        prev_value = true;
    }
    if !prev_value {
        return None; // trailing operator
    }
    // Collapse products first.
    let mut terms: Vec<i64> = Vec::new();
    let mut sign = 1i64;
    let mut acc: Option<i64> = None;
    let mut pending_times = false;
    for t in toks {
        match t {
            T::N(v) => {
                if pending_times {
                    acc = Some(acc.unwrap_or(1) * v);
                    pending_times = false;
                } else {
                    if let Some(a) = acc.take() {
                        terms.push(a);
                    }
                    acc = Some(v);
                }
            }
            T::Times => pending_times = true,
            T::Plus | T::Minus => {
                if let Some(a) = acc.take() {
                    terms.push(sign * a);
                }
                sign = if matches!(t, T::Minus) { -1 } else { 1 };
            }
        }
    }
    if let Some(a) = acc {
        terms.push(sign * a);
    }
    Some(terms.iter().sum())
}

/// Parse a character-literal immediate: `'a'`, `'\n'`, `'\''`.
fn parse_char(s: &str) -> Result<i64> {
    let inner = s
        .strip_prefix('\'')
        .and_then(|x| x.strip_suffix('\''))
        .ok_or_else(|| anyhow::anyhow!("malformed char literal: `{s}`"))?;
    let mut chars = inner.chars();
    let v = match chars.next() {
        Some('\\') => match chars.next() {
            Some('n') => b'\n' as i64,
            Some('t') => b'\t' as i64,
            Some('r') => b'\r' as i64,
            Some('0') => 0,
            Some('\\') => b'\\' as i64,
            Some('\'') => b'\'' as i64,
            Some(o) => o as i64,
            None => bail!("dangling escape in char literal `{s}`"),
        },
        Some(c) => c as i64,
        None => bail!("empty char literal `{s}`"),
    };
    Ok(v)
}

/// Parse a signed integer: decimal or `0x` hex, `_` separators allowed.
pub fn parse_int(s: &str) -> Result<i64> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let body: String = body.chars().filter(|&c| c != '_').collect();
    let v: i64 = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        // Parse as u64 then reinterpret, so 0xFFFF.. wide constants survive.
        u64::from_str_radix(hex, 16).map(|u| u as i64).map_err(|e| anyhow::anyhow!("bad hex `{s}`: {e}"))?
    } else {
        body.parse::<i64>().map_err(|e| anyhow::anyhow!("bad int `{s}`: {e}"))?
    };
    Ok(if neg { -v } else { v })
}

fn parse_string(arg: &str) -> Result<Vec<u8>> {
    let arg = arg.trim();
    let inner = arg
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .ok_or_else(|| anyhow::anyhow!("malformed string literal: `{arg}`"))?;
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push(b'\n'),
                Some('t') => out.push(b'\t'),
                Some('r') => out.push(b'\r'),
                Some('0') => out.push(0),
                Some('\\') => out.push(b'\\'),
                Some('"') => out.push(b'"'),
                Some(other) => out.push(other as u8),
                None => bail!("dangling escape in `{arg}`"),
            }
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    Ok(out)
}

/// A legal symbol/label identifier (allows the `$$` proc-local mangling and `.`).
fn is_symbol(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_' || c == '.').unwrap_or(false)
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '.')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insn(s: &str) -> (String, Vec<Operand>) {
        match parse_line(s).unwrap() {
            Line::Insn { mnemonic, ops } => (mnemonic, ops),
            other => panic!("expected insn, got {other:?} for `{s}`"),
        }
    }

    #[test]
    fn directives_and_labels() {
        assert_eq!(parse_line(".intel_syntax noprefix").unwrap(), Line::Directive(Directive::IntelSyntax));
        assert_eq!(parse_line(".text").unwrap(), Line::Directive(Directive::Text));
        assert_eq!(parse_line(".globl dup_").unwrap(), Line::Directive(Directive::Globl("dup_".into())));
        assert_eq!(parse_line(".quad 0").unwrap(), Line::Directive(Directive::Quad(vec![0])));
        assert_eq!(
            parse_line(".quad 0x8000000000000000, 0").unwrap(),
            Line::Directive(Directive::Quad(vec![0x8000000000000000u64 as i64, 0])),
        );
        assert_eq!(parse_line("dup_:").unwrap(), Line::Label("dup_".into()));
        assert_eq!(parse_line("qdup$$nodup:").unwrap(), Line::Label("qdup$$nodup".into()));
        assert_eq!(parse_line("").unwrap(), Line::Empty);
    }

    #[test]
    fn reg_reg_and_imm() {
        let (m, ops) = insn("mov rax, rcx");
        assert_eq!(m, "mov");
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::R64, num: 0 }));
        assert_eq!(ops[1], Operand::Reg(Reg { class: RegClass::R64, num: 1 }));
        assert_eq!(insn("sub rbp, 8").1[1], Operand::Imm(8));
        assert_eq!(insn("mov rax, 0xDEAD_BEEF").1[1], Operand::Imm(0xDEADBEEF));
        assert_eq!(insn("add rcx, -8").1[1], Operand::Imm(-8));
    }

    #[test]
    fn memory_forms() {
        // [base]
        assert_eq!(
            insn("add rax, [rbp]").1[1],
            Operand::Mem(Mem { size: None, base: Some(Reg { class: RegClass::R64, num: 5 }), index: None, scale: 1, disp: 0, rip_sym: None })
        );
        // [base + disp]
        assert_eq!(
            insn("mov rcx, [rbx + 4632]").1[1],
            Operand::Mem(Mem { size: None, base: Some(Reg { class: RegClass::R64, num: 3 }), index: None, scale: 1, disp: 4632, rip_sym: None })
        );
        // [base - disp] as dest with size prefix
        assert_eq!(
            insn("movsd qword ptr [rcx - 8], xmm15").1[0],
            Operand::Mem(Mem { size: Some(MemSize::Qword), base: Some(Reg { class: RegClass::R64, num: 1 }), index: None, scale: 1, disp: -8, rip_sym: None })
        );
        // [base + index*scale]
        assert_eq!(
            insn("lea rax, [rax + rax*1]").1[1],
            Operand::Mem(Mem { size: None, base: Some(Reg{class:RegClass::R64,num:0}), index: Some(Reg{class:RegClass::R64,num:0}), scale: 1, disp: 0, rip_sym: None })
        );
    }

    #[test]
    fn rip_relative() {
        assert_eq!(
            insn("lea r8, [rip + compile_word]").1[1],
            Operand::Mem(Mem { size: None, base: None, index: None, scale: 1, disp: 0, rip_sym: Some("compile_word".into()) })
        );
    }

    #[test]
    fn branch_and_call_targets() {
        assert_eq!(insn("call throw_word").1[0], Operand::Sym("throw_word".into()));
        assert_eq!(insn("jz qdup$$nodup").1[0], Operand::Sym("qdup$$nodup".into()));
        assert_eq!(insn("ret").1.len(), 0);
    }

    #[test]
    fn xmm_and_sse() {
        let (m, ops) = insn("addsd xmm15, qword ptr [rcx]");
        assert_eq!(m, "addsd");
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::Xmm, num: 15 }));
        assert_eq!(ops[1], Operand::Mem(Mem { size: Some(MemSize::Qword), base: Some(Reg { class: RegClass::R64, num: 1 }), index: None, scale: 1, disp: 0, rip_sym: None }));
    }
}
