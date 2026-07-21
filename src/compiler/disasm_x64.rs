//! x86-64 disassembly for the PROBE crash dossier and `MACVM_DBG_IR`
//! listings (WINVM Phase 3x) — the sibling of [`crate::compiler::disasm_a64`].
//!
//! ## Why this is not a straight port of the AArch64 one
//!
//! `disasm_a64::disasm_slice` takes a byte slice and decodes it in
//! 4-byte chunks. Every AArch64 instruction is 4 bytes and every 4-byte
//! boundary is an instruction boundary, so *any* slice of a code blob can
//! be decoded correctly, and a window around a faulting pc is just
//! `pc-64 .. pc+64`.
//!
//! None of that holds on x86-64. Instructions are 1–15 bytes, and there
//! is no way to tell from the bytes alone where an instruction *starts* —
//! decoding from `pc - 64` will in general land mid-instruction and
//! produce a plausible-looking listing that is entirely wrong. That
//! failure mode is worse than no listing: it is confident and false, in a
//! crash dump someone is reading under pressure.
//!
//! So this decodes **forward from a known boundary**. The caller supplies
//! the nmethod's own base, which is an instruction boundary by
//! construction, and the window is chosen by *counting decoded
//! instructions* rather than by byte arithmetic. That is exact, and it is
//! the reason this module wants the whole code slice rather than a
//! pre-cut window.

use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};

/// One decoded instruction: its offset from the slice start, its bytes,
/// and its text.
pub struct DecodedInsn {
    pub off: usize,
    pub len: usize,
    pub text: String,
    /// Absolute address this instruction's RIP-relative memory operand
    /// resolves to, if it has one.
    ///
    /// This is the x64 analogue of an AArch64 `ldr`-literal target, and
    /// it is what lets a listing show the *constant* a pool load fetches
    /// rather than just `[rip+0x2f]`. Comparing what the compiler baked
    /// against what the runtime dispatched, in one screen, is the whole
    /// point of the `disasm-native` verb.
    pub ip_rel_target: Option<u64>,
}

/// Decode `code` (starting at virtual address `rip`) from its first byte,
/// which the caller warrants is an instruction boundary.
///
/// Stops early at `limit_off` if given — used to avoid decoding a blob's
/// trailing literal pool as if it were code, which produces impressive
/// nonsense.
pub fn decode_all(code: &[u8], rip: u64, limit_off: Option<usize>) -> Vec<DecodedInsn> {
    let end = limit_off.unwrap_or(code.len()).min(code.len());
    let mut decoder = Decoder::with_ip(64, &code[..end], rip, DecoderOptions::NONE);
    let mut fmt = IntelFormatter::new();
    let mut out = Vec::new();
    let mut insn = Instruction::default();
    while decoder.can_decode() {
        let off = decoder.position();
        decoder.decode_out(&mut insn);
        let mut text = String::new();
        fmt.format(&insn, &mut text);
        let ip_rel_target = if insn.is_ip_rel_memory_operand() {
            Some(insn.ip_rel_memory_address())
        } else {
            None
        };
        out.push(DecodedInsn {
            off,
            len: insn.len(),
            text,
            ip_rel_target,
        });
    }
    out
}

/// The dossier's step-5 window: `before` instructions leading up to the
/// one containing `mark_off`, that instruction (flagged), and `after`
/// following it.
///
/// `mark_off` is a byte offset into `code`. If it falls strictly inside
/// an instruction rather than on its first byte — which is itself a
/// finding, since it means control reached a mid-instruction address —
/// the containing instruction is marked and the discrepancy is called
/// out rather than silently rounded away.
pub fn window(
    code: &[u8],
    rip: u64,
    mark_off: usize,
    before: usize,
    after: usize,
    limit_off: Option<usize>,
) -> Vec<String> {
    let insns = decode_all(code, rip, limit_off);
    if insns.is_empty() {
        return vec!["disasm: nothing decodable at this address".into()];
    }
    let hit = insns
        .iter()
        .position(|i| mark_off >= i.off && mark_off < i.off + i.len);
    let Some(hit) = hit else {
        return vec![format!(
            "disasm: pc offset {mark_off:#x} is past the decoded range \
             (last insn ends at {:#x})",
            insns.last().map(|i| i.off + i.len).unwrap_or(0)
        )];
    };

    let lo = hit.saturating_sub(before);
    let hi = (hit + after + 1).min(insns.len());
    let mut out = Vec::with_capacity(hi - lo + 1);
    for (idx, i) in insns[lo..hi].iter().enumerate() {
        let idx = lo + idx;
        let raw: String = code[i.off..i.off + i.len]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("");
        let mut line = format!("+{:#06x}  {raw:<20}  {}", i.off, i.text);
        if idx == hit {
            line.push_str("  <== HERE");
            if mark_off != i.off {
                line.push_str(" (MID-INSTRUCTION — control reached a non-boundary address)");
            }
        }
        out.push(line);
    }
    out
}

/// If `code[off..]` begins a deopt trap site, its 16-bit immediate.
///
/// A trap is `int3` followed by a raw `imm16` that is *data*, not an
/// operand — so a disassembler faithfully renders the two bytes after
/// `0xCC` as whatever instruction they happen to encode. In the panic
/// output that closed the `last_compiled_pc` bug, `cc 00 de` printed as
/// `int3` then `add dh,bl`, which is correct and useless. A listing that
/// knows the emitter's own trap convention can say `0xDE00` instead.
pub fn trap_imm_at(code: &[u8], off: usize) -> Option<u16> {
    if off + 3 > code.len() || code[off] != 0xCC {
        return None;
    }
    Some(u16::from_le_bytes([code[off + 1], code[off + 2]]))
}

/// Plain linear listing of a whole blob — the `MACVM_DBG_IR` companion,
/// and what a test asserts against.
pub fn disasm_slice(code: &[u8], rip: u64) -> String {
    let mut s = String::new();
    for i in decode_all(code, rip, None) {
        s.push_str(&format!("+{:#06x}  {}\n", i.off, i.text));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window is chosen by counting instructions from a known
    /// boundary, so a variable-length prefix cannot shift it.
    ///
    /// The byte sequence below is deliberately mixed-length (1, 3, 7, 2 …):
    /// slicing `pc-N .. pc+N` and decoding from the cut — the AArch64
    /// module's approach — would start mid-instruction and mis-decode
    /// everything up to the pc.
    #[test]
    fn window_counts_instructions_not_bytes() {
        // push rbp            55            (1)
        // mov rbp, rsp        48 89 e5      (3)
        // mov rax, 0x1234     48 c7 c0 34 12 00 00  (7)
        // xor ecx, ecx        31 c9         (2)
        // ret                 c3            (1)
        let code = [
            0x55, 0x48, 0x89, 0xE5, 0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00, 0x31, 0xC9, 0xC3,
        ];
        // Mark the `xor`, at byte offset 11.
        let lines = window(&code, 0x1000, 11, 2, 1, None);
        assert_eq!(lines.len(), 4, "2 before + the hit + 1 after");
        assert!(lines[0].contains("mov"), "first line is the mov rbp,rsp: {lines:?}");
        assert!(
            lines[2].contains("xor") && lines[2].contains("<== HERE"),
            "the marked line is the xor: {lines:?}"
        );
        assert!(lines[3].contains("ret"), "trailing context: {lines:?}");
    }

    /// A pc that is not an instruction boundary is REPORTED as such
    /// rather than quietly snapped to the containing instruction — it
    /// means control jumped into the middle of an instruction, which is
    /// a far more specific finding than "it crashed near here".
    #[test]
    fn mid_instruction_pc_is_called_out() {
        let code = [0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00, 0xC3];
        let lines = window(&code, 0x2000, 3, 0, 0, None);
        assert!(
            lines[0].contains("MID-INSTRUCTION"),
            "a non-boundary pc must say so: {lines:?}"
        );
    }

    /// The literal pool is data, not code. Decoding past `limit_off`
    /// would render pool words as instructions — confident nonsense in
    /// exactly the place a reader is least able to check it.
    #[test]
    fn decoding_stops_at_the_literal_pool() {
        let code = [0x55, 0xC3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let all = decode_all(&code, 0x3000, Some(2));
        assert_eq!(all.len(), 2, "only the two real instructions: {all:?}",);
        assert!(all[0].text.contains("push"));
        assert!(all[1].text.contains("ret"));
    }
}

impl std::fmt::Debug for DecodedInsn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "+{:#x} {}", self.off, self.text)
    }
}
