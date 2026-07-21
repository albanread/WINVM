//! Polymorphic inline caches, x86-64 (WINVM Phase 3) — the sibling of
//! `pics::build_pic_stub`.
//!
//! A PIC body is a linear chain: load the receiver's klass once, compare
//! it against each cached klass in turn, and **tail-jump** to that
//! klass's target on a hit. Falling off the end means the receiver's
//! klass is not cached yet, so the chain ends in a jump to the resolve
//! stub, which will rebuild a longer PIC.
//!
//! Tail-jumping (never calling) is what makes a PIC free: the target
//! method returns straight to whoever performed the send, with no PIC
//! frame in between. Nothing here establishes a frame at all.
//!
//! ## Two things x86-64 does better than the AArch64 original
//!
//! * **The klass comparison needs no register.** `cmp reg, [rip + lit]`
//!   compares directly against the pool word, where AArch64 must first
//!   `ldr` the literal into a scratch. Over a 4-entry PIC that removes
//!   four instructions from the hot path.
//! * **A smi receiver needs no branch to reach its klass**, only a
//!   literal load — same as AArch64, but the merge is cheaper because
//!   the compare that follows reads memory rather than a register.
//!
//! ## The GC contract
//!
//! Each cached klass lives in a literal-pool word, and the builder
//! returns those words' byte offsets. A moving collection rewrites them
//! in place, which is why the klass is compared out of the pool on every
//! dispatch rather than baked into an immediate — an immediate could not
//! be updated.

use crate::compiler::assembler::{CodeBlob, RelocKind};
use crate::compiler::assembler_x64::{
    imm, mem, r64, Cond, X64Assembler, ARG_REGS, SCRATCH0, SCRATCH1,
};
use crate::oops::wrappers::KlassOop;

/// Byte offset of the klass word from a tagged heap oop.
const KLASS_OFF_FROM_TAGGED: i64 =
    crate::oops::layout::KLASS_OFFSET as i64 - crate::oops::layout::MEM_TAG as i64;

/// Build a PIC body for `pairs` (cached klass → target entry), falling
/// through to `resolve_addr`.
///
/// Returns the blob and the byte offsets of the klass pool words, for the
/// GC to relocate.
pub fn build_pic_stub_x64(
    pairs: &[(KlassOop, u64)],
    smi_klass_bits: u64,
    resolve_addr: u64,
) -> (CodeBlob, Vec<u32>) {
    // A PIC compares klass words for IDENTITY and never dereferences
    // them, so the body works on raw bits. Splitting it here keeps this
    // signature identical to the AArch64 builder (drop-in for the
    // caller) while letting the tests below exercise the body without
    // fabricating a structurally-valid `KlassOop`.
    let raw: Vec<(u64, u64)> = pairs.iter().map(|&(k, t)| (k.oop().raw(), t)).collect();
    build_pic_stub_raw(&raw, smi_klass_bits, resolve_addr)
}

fn build_pic_stub_raw(
    pairs: &[(u64, u64)],
    smi_klass_bits: u64,
    resolve_addr: u64,
) -> (CodeBlob, Vec<u32>) {
    let mut a = X64Assembler::new();
    let recv = ARG_REGS[0];

    // Load the receiver's klass into SCRATCH1 once, for the whole chain.
    let smi_lit = a.literal_u64(smi_klass_bits, Some(RelocKind::Oop));
    let smi_case = a.new_label();
    let after_klass_load = a.new_label();
    a.emit("test", &[r64(recv), imm(3)]);
    a.jcc(Cond::E, smi_case);
    a.emit("mov", &[r64(SCRATCH1), mem(recv, KLASS_OFF_FROM_TAGGED)]);
    a.jmp(after_klass_load);
    a.bind(smi_case);
    a.load_literal(SCRATCH1, smi_lit);
    a.bind(after_klass_load);

    let mut klass_lits = Vec::with_capacity(pairs.len());
    for &(k_bits, t) in pairs {
        let next = a.new_label();
        let k_lit = a.literal_u64(k_bits, Some(RelocKind::Oop));
        klass_lits.push(k_lit);
        let t_lit = a.literal_u64(t, Some(RelocKind::RuntimeAddr));
        // Compare straight against the pool word — no scratch needed.
        a.cmp_literal(SCRATCH1, k_lit);
        a.jcc(Cond::Ne, next);
        a.load_literal(SCRATCH0, t_lit);
        a.emit("jmp", &[r64(SCRATCH0)]);
        a.bind(next);
    }

    // Miss: hand it to resolve, which rebuilds a longer PIC.
    let resolve_lit = a.literal_u64(resolve_addr, Some(RelocKind::RuntimeAddr));
    a.load_literal(SCRATCH0, resolve_lit);
    a.emit("jmp", &[r64(SCRATCH0)]);

    let blob = a.finish();
    let klass_pool_offs = klass_lits
        .iter()
        .map(|l| blob.literal_off + 8 * l.0)
        .collect();
    (blob, klass_pool_offs)
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::oops::layout::MEM_TAG;

    /// A stand-in klass: any tagged heap word serves, since a PIC only
    /// ever compares klass words for identity.
    fn fake_klass(storage: &mut [u64; 2]) -> u64 {
        storage.as_ptr() as u64 | MEM_TAG
    }

    /// A PIC dispatches to the entry cached for the receiver's klass, and
    /// to resolve when the klass is not in the chain.
    ///
    /// Every arm is exercised — first, last, and miss — because a PIC is
    /// a linear chain and an off-by-one in the compare/branch pairing
    /// would still dispatch *some* entry correctly.
    #[cfg(windows)]
    #[test]
    fn pic_dispatches_by_klass_and_falls_through_to_resolve() {
        use crate::vendor::wfasm::native_windows::WinJit;

        extern "C" fn target_a(_r: u64) -> u64 {
            0xAAA
        }
        extern "C" fn target_b(_r: u64) -> u64 {
            0xBBB
        }
        extern "C" fn resolve(_r: u64) -> u64 {
            0x9999
        }

        let mut ka = [0u64; 2];
        let mut kb = [0u64; 2];
        let mut kc = [0u64; 2];
        let ka_bits = fake_klass(&mut ka);
        let kb_bits = fake_klass(&mut kb);
        let kc_bits = fake_klass(&mut kc);

        // Receivers: heap objects whose klass words name the above.
        let mut obj_a = [0u64, ka_bits];
        let mut obj_b = [0u64, kb_bits];
        let mut obj_c = [0u64, kc_bits];
        let recv = |o: &[u64; 2]| o.as_ptr() as u64 | MEM_TAG;

        const SMI_KLASS: u64 = 0x5151_0001;
        let (blob, klass_offs) = build_pic_stub_raw(
            &[
                (ka_bits, target_a as usize as u64),
                (kb_bits, target_b as usize as u64),
            ],
            SMI_KLASS,
            resolve as usize as u64,
        );

        // The GC contract: one pool word per cached klass, holding it.
        assert_eq!(klass_offs.len(), 2, "one relocatable word per cached klass");
        for (off, want) in klass_offs.iter().zip([ka_bits, kb_bits]) {
            let w = u64::from_le_bytes(
                blob.code[*off as usize..*off as usize + 8].try_into().unwrap(),
            );
            assert_eq!(w, want, "pool word holds the klass, so a GC can relocate it");
        }

        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        let pic: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(base) };

        assert_eq!(pic(recv(&obj_a)), 0xAAA, "first chain entry");
        assert_eq!(pic(recv(&obj_b)), 0xBBB, "last chain entry");
        assert_eq!(pic(recv(&obj_c)), 0x9999, "uncached klass falls to resolve");
        let _ = (&mut obj_a, &mut obj_b, &mut obj_c);
    }

    /// A smi receiver has no header to load, so the PIC substitutes the
    /// smi klass literal. A PIC that dereferenced a smi would fault; one
    /// that skipped the substitution would always miss to resolve.
    #[cfg(windows)]
    #[test]
    fn pic_handles_a_smi_receiver_without_dereferencing_it() {
        use crate::vendor::wfasm::native_windows::WinJit;

        extern "C" fn smi_target(_r: u64) -> u64 {
            0x5A1
        }
        extern "C" fn resolve(_r: u64) -> u64 {
            0x9999
        }

        // Cache the SMI klass itself, so a smi receiver must HIT.
        let mut ks = [0u64; 2];
        let ks_bits = fake_klass(&mut ks);
        let (blob, _) = build_pic_stub_raw(
            &[(ks_bits, smi_target as usize as u64)],
            ks_bits, // smi_klass_bits == the cached klass
            resolve as usize as u64,
        );

        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        let pic: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(base) };

        // A tagged smi (low bits clear) must take the smi path and hit.
        assert_eq!(pic(42 << 2), 0x5A1, "smi receiver uses the smi klass");
        assert_eq!(pic(0), 0x5A1, "zero is a smi too");
        let _ = &mut ks;
    }

    /// An empty PIC is all miss — the degenerate case a freshly-created
    /// cache starts from.
    #[cfg(windows)]
    #[test]
    fn empty_pic_goes_straight_to_resolve() {
        use crate::vendor::wfasm::native_windows::WinJit;
        extern "C" fn resolve(_r: u64) -> u64 {
            0x9999
        }
        let (blob, offs) = build_pic_stub_raw(&[], 0x5151_0001, resolve as usize as u64);
        assert!(offs.is_empty());
        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        let pic: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(base) };
        assert_eq!(pic(42 << 2), 0x9999);
    }
}
