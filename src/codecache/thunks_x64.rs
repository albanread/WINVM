//! Per-payload thunks, x86-64 (WINVM Phase 3) — the siblings of
//! `mega::build_mega_trampoline` and `adapters::build_c2i_adapter`.
//!
//! Both are the same three-instruction shape: park a payload oop in a
//! scratch register, load the shared stub's address, and **tail-jump**.
//! There is one thunk per selector (megamorphic) or per method (c2i), and
//! one shared stub they all funnel into — which is the whole point: the
//! per-payload part stays tiny, and the expensive body exists once.
//!
//! ## The register carrying the payload is a contract
//!
//! Each thunk hands its shared stub a value in a specific register, and
//! that stub reads it from there. The AArch64 originals use `x16` for a
//! selector and `x17` for a method — a distinction that looks arbitrary
//! but is load-bearing, because `mega_shared` and `c2i_shared` each read
//! only their own. The x64 mapping preserves it:
//!
//! | payload | AArch64 | x86-64 | read by |
//! |---|---|---|---|
//! | selector | `x16` | `R10` | `stubs_x64::build_stub_mega_shared_x64` |
//! | method | `x17` | `R11` | [`build_c2i_shared_x64`] |
//!
//! Getting these crossed would not fault — each stub would simply act on
//! whatever oop happened to be in the register it reads, dispatching the
//! wrong selector or interpreting the wrong method.

use crate::codecache::stubs_x64::{emit_stub_epilogue_x64, emit_stub_prologue_x64, ROOTSPILL};
use crate::compiler::assembler::{CodeBlob, RelocKind};
use crate::compiler::assembler_x64::{
    imm, mem, r64, X64Assembler, ARG_REGS, R10, R11, RAX, RBP, VM_STATE,
};

/// One megamorphic thunk: carry `selector_bits` in `R10` and jump to the
/// shared megamorphic lookup.
pub fn build_mega_trampoline_x64(selector_bits: u64, stub_mega_shared_addr: u64) -> CodeBlob {
    let mut a = X64Assembler::new();
    let sel_lit = a.literal_u64(selector_bits, Some(RelocKind::Oop));
    let shared_lit = a.literal_u64(stub_mega_shared_addr, Some(RelocKind::RuntimeAddr));
    a.load_literal(R10, sel_lit); // the selector — mega_shared reads R10
    a.load_literal(R11, shared_lit);
    a.emit("jmp", &[r64(R11)]);
    a.finish()
}

/// One compiled-to-interpreted adapter: carry `method_bits` in `R11` and
/// jump to the shared c2i entry. Used when a send site resolves to a
/// method with no compiled form — the adapter lets compiled code call it
/// exactly like any other target.
pub fn build_c2i_adapter_x64(method_bits: u64, c2i_shared_addr: u64) -> CodeBlob {
    let mut a = X64Assembler::new();
    let method_lit = a.literal_u64(method_bits, Some(RelocKind::Oop));
    let shared_lit = a.literal_u64(c2i_shared_addr, Some(RelocKind::RuntimeAddr));
    a.load_literal(R11, method_lit); // the method — c2i_shared reads R11
    a.load_literal(R10, shared_lit);
    a.emit("jmp", &[r64(R10)]);
    a.finish()
}

/// `c2i_shared` — the one body every c2i adapter funnels into. Hands the
/// method and the spilled arguments to `rt_interpret_call`, which runs
/// the method in the interpreter, and returns its result like an ordinary
/// stub.
///
/// The `argv` it passes is the RootSpill, so the interpreter works on the
/// GC-visible copies of the arguments (see `stubs_x64`'s frame contract).
pub fn build_c2i_shared_x64(rt_interpret_call_addr: u64, kind: u64) -> CodeBlob {
    let mut a = X64Assembler::new();
    // The method arrives in R11 and must be read before the prologue,
    // whose kind-tag store uses R10 — and before the argument shuffle.
    a.emit("mov", &[r64(RAX), r64(R11)]);
    emit_stub_prologue_x64(&mut a, kind);
    a.emit("mov", &[r64(ARG_REGS[1]), r64(RAX)]); // method_bits
    a.emit("mov", &[r64(ARG_REGS[0]), r64(VM_STATE)]);
    a.emit("lea", &[r64(ARG_REGS[2]), mem(RBP, -ROOTSPILL)]); // argv
    let lit = a.literal_u64(rt_interpret_call_addr, Some(RelocKind::RuntimeAddr));
    a.call_far(lit);
    a.emit("mov", &[r64(R11), r64(RAX)]);
    emit_stub_epilogue_x64(&mut a);
    a.emit("mov", &[r64(RAX), r64(R11)]);
    a.emit("ret", &[]);
    let _ = imm(0);
    a.finish()
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::vendor::wfasm::native_windows::WinJit;

    /// Place a blob and return its entry address, keeping the region
    /// alive for the caller.
    fn place(blob: &CodeBlob) -> (WinJit, u64) {
        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        let addr = base as u64;
        (jit, addr)
    }

    /// A megamorphic thunk delivers its selector in `R10` and jumps to
    /// the shared stub. The probe reads `R10` back, so a thunk that used
    /// the wrong register — or that `call`ed instead of jumping — fails
    /// rather than silently dispatching some other selector.
    #[cfg(windows)]
    #[test]
    fn mega_thunk_carries_the_selector_in_r10_and_tail_jumps() {
        const SELECTOR: u64 = 0x5E1E_C704;

        // Stand-in for mega_shared: returns whatever it finds in R10.
        let mut s = X64Assembler::new();
        s.emit("mov", &[r64(RAX), r64(R10)]);
        s.emit("ret", &[]);
        let shared = s.finish();
        let (_jit_s, shared_addr) = place(&shared);

        let thunk = build_mega_trampoline_x64(SELECTOR, shared_addr);
        let (_jit_t, thunk_addr) = place(&thunk);

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(thunk_addr as *const u8) };
        assert_eq!(
            f(),
            SELECTOR,
            "mega_shared must find the selector in R10 — a crossed register would \
             dispatch a different selector without faulting"
        );
    }

    /// A c2i adapter delivers its method in `R11`. Deliberately the OTHER
    /// register from the mega thunk: the two shared stubs read different
    /// ones, and crossing them is a silent wrong-method bug.
    #[cfg(windows)]
    #[test]
    fn c2i_adapter_carries_the_method_in_r11_and_tail_jumps() {
        const METHOD: u64 = 0x3E70_D001;

        let mut s = X64Assembler::new();
        s.emit("mov", &[r64(RAX), r64(R11)]);
        s.emit("ret", &[]);
        let shared = s.finish();
        let (_jit_s, shared_addr) = place(&shared);

        let adapter = build_c2i_adapter_x64(METHOD, shared_addr);
        let (_jit_a, adapter_addr) = place(&adapter);

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(adapter_addr as *const u8) };
        assert_eq!(f(), METHOD, "c2i_shared must find the method in R11");
    }

    /// The two conventions must stay distinct. This is the test that
    /// fails if someone "tidies up" both thunks to use the same scratch.
    #[cfg(windows)]
    #[test]
    fn mega_and_c2i_use_different_payload_registers() {
        // A probe returning R10 sees the mega thunk's selector but NOT
        // the c2i adapter's method, and vice versa.
        let mut r10p = X64Assembler::new();
        r10p.emit("mov", &[r64(RAX), r64(R10)]);
        r10p.emit("ret", &[]);
        let r10_probe = r10p.finish();
        let (_j1, r10_addr) = place(&r10_probe);

        let mut r11p = X64Assembler::new();
        r11p.emit("mov", &[r64(RAX), r64(R11)]);
        r11p.emit("ret", &[]);
        let r11_probe = r11p.finish();
        let (_j2, r11_addr) = place(&r11_probe);

        const PAYLOAD: u64 = 0xABCD_0001;
        let mega = build_mega_trampoline_x64(PAYLOAD, r10_addr);
        let (_j3, mega_addr) = place(&mega);
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mega_addr as *const u8) };
        assert_eq!(f(), PAYLOAD, "mega payload is in R10");

        let c2i = build_c2i_adapter_x64(PAYLOAD, r11_addr);
        let (_j4, c2i_addr) = place(&c2i);
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(c2i_addr as *const u8) };
        assert_eq!(f(), PAYLOAD, "c2i payload is in R11");
    }

    /// `c2i_shared` hands `rt_interpret_call` the method it was given and
    /// an `argv` pointing at the RootSpill, then returns its result.
    #[cfg(windows)]
    #[test]
    fn c2i_shared_forwards_method_and_rootspill_argv() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEEN_METHOD: AtomicU64 = AtomicU64::new(0);
        static SEEN_ARG0: AtomicU64 = AtomicU64::new(0);

        extern "C" fn interpret_call(_vm: u64, method: u64, argv: *const u64) -> u64 {
            SEEN_METHOD.store(method, Ordering::Relaxed);
            SEEN_ARG0.store(unsafe { *argv }, Ordering::Relaxed);
            0x1_7E_12
        }

        let shared = build_c2i_shared_x64(
            interpret_call as usize as u64,
            crate::codecache::stubs::KIND_C2I,
        );
        let (_jit, shared_addr) = place(&shared);

        // Harness: plant R15 (vm), R11 (method), and an argument in RCX,
        // then jump to c2i_shared the way an adapter would.
        let mut h = X64Assembler::new();
        h.emit("push", &[r64(RBP)]);
        h.emit("mov", &[r64(RBP), r64(crate::compiler::assembler_x64::RSP)]);
        h.emit("push", &[r64(VM_STATE)]);
        h.emit("sub", &[r64(crate::compiler::assembler_x64::RSP), imm(40)]);
        h.emit("mov", &[r64(R10), r64(crate::compiler::assembler_x64::RCX)]); // shared
        h.emit("mov", &[r64(VM_STATE), r64(crate::compiler::assembler_x64::RDX)]); // vm
        h.emit("mov", &[r64(R11), r64(crate::compiler::assembler_x64::R8)]); // method
        h.emit("mov", &[r64(crate::compiler::assembler_x64::RCX), r64(crate::compiler::assembler_x64::R9)]); // arg0
        h.emit("call", &[r64(R10)]);
        h.emit("add", &[r64(crate::compiler::assembler_x64::RSP), imm(40)]);
        h.emit("pop", &[r64(VM_STATE)]);
        h.emit("pop", &[r64(RBP)]);
        h.emit("ret", &[]);
        let harness = h.finish();
        let (_jh, harness_addr) = place(&harness);

        let mut vmreg = [0u64; 16];
        let vm = vmreg.as_mut_ptr() as u64;
        const METHOD: u64 = 0x3E70_D001;
        const ARG0: u64 = 0x4242;
        let hf: extern "C" fn(u64, u64, u64, u64) -> u64 =
            unsafe { std::mem::transmute(harness_addr as *const u8) };
        let got = hf(shared_addr, vm, METHOD, ARG0);

        assert_eq!(got, 0x1_7E_12, "the interpreted result is returned");
        assert_eq!(SEEN_METHOD.load(Ordering::Relaxed), METHOD, "method from R11");
        assert_eq!(
            SEEN_ARG0.load(Ordering::Relaxed),
            ARG0,
            "argv points at the RootSpill, so the interpreter sees the GC-visible \
             argument copies"
        );
        assert_eq!(
            vmreg[crate::oops::layout::VMREG_LAST_COMPILED_FP_OFFSET / 8],
            0,
            "walker record cleared"
        );
    }
}
