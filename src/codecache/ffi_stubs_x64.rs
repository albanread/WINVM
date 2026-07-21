//! WINVM FFI trampolines, x86-64 (`docs/FFI.md` §5) — the Win64 sibling of
//! [`ffi_stubs`](crate::codecache::ffi_stubs).
//!
//! ## Why this is not a transliteration of the AArch64 trampoline
//!
//! The AArch64 trampoline takes **two** argument buffers, `argv_g` and
//! `argv_f`, because AAPCS64 has two *independent* register sequences: the
//! nth integer argument goes in `x{n}` and the nth float in `d{n}`,
//! regardless of how they interleave in the signature. Splitting the
//! arguments by class is therefore lossless there.
//!
//! **Win64 has a single argument-slot sequence.** Slot `i` is
//! `RCX/RDX/R8/R9` *or* `XMM0..XMM3` — the parameter's type picks which
//! register file, and its POSITION picks the slot. Mixed signatures do not
//! share slots:
//!
//! ```text
//!     f(int a, double b, int c)   ->   a: RCX     b: XMM1     c: R8
//!                                          slot 0     slot 1      slot 2
//! ```
//!
//! Two class-partitioned buffers cannot express that: they record that
//! there was one integer and one float before `c`, but not that the float
//! *consumed slot 1*, so a faithful marshaller cannot tell whether `c`
//! belongs in `RDX` or `R8`. (JASM's own Win32 generator states the same
//! rule: "Position determines the slot — slot 0 is rcx-OR-xmm0".)
//!
//! So this trampoline takes ONE buffer in **position order** plus a
//! `class_mask` naming which positions are floating point. That shape is
//! strictly more expressive than the split pair — an AAPCS64 marshaller
//! can partition by the mask and recover its own two sequences — so the
//! Windows requirement did not force a worse interface, only a different
//! one.
//!
//! ## Uniform shape, three return classes
//!
//! Like the AArch64 side there is exactly one trampoline per *return*
//! class (`g`/`f`/`v`) and no per-signature family: argument marshalling
//! is driven entirely by runtime data. The handful of extra instructions
//! that costs a small call is noise beside the native call itself.

use crate::compiler::assembler::CodeBlob;
use crate::compiler::assembler_x64::{
    imm, mem, r64, xmm, X64Assembler, ARG_REGS, MAX_REG_ARGS, RAX, RBP, RBX, RCX, RDX, RSI, RSP,
};

/// Every x64 FFI trampoline's Rust-side signature.
///
/// * `target` — the resolved native address.
/// * `argv` — exactly [`ARGV_WORDS`] words, in **signature position
///   order**. Integer/pointer arguments are raw bits; floating-point ones
///   are `f64::to_bits()` (an `f32` is widened to `f64` by the caller,
///   matching how Win64 passes a float in the low half of an XMM).
/// * `class_mask` — bit `i` set means position `i` is floating point.
/// * `argc` — how many of `argv`'s words are real arguments.
///
/// The returned `u64` is the raw result: `ret_g` callers use it directly,
/// `ret_f` callers apply `f64::from_bits`, `ret_v` callers ignore it.
pub type FfiCallFnX64 =
    unsafe extern "C" fn(target: u64, argv: *const u64, class_mask: u32, argc: u32) -> u64;

/// Words in the argument buffer: 4 register slots plus 12 stack slots.
/// Bounded comfortably above `METHOD_ARGC_MAX` (15), the real ceiling any
/// pragma can declare.
pub const ARGV_WORDS: usize = 16;

/// Which register class an FFI call's own RETURN value uses — the only
/// per-call dimension that cannot be handled by runtime data, because it
/// decides which register the result is read out of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FfiRetClassX64 {
    /// Integer/pointer, and `void` (which shares the shape and is ignored).
    G,
    /// `double`/`float` — the result comes back in `XMM0`.
    F,
}

/// Build one trampoline.
///
/// The generated code is an ordinary Win64 function. It marshals `argv`
/// into the outgoing argument area, calls `target`, and returns the raw
/// result.
///
/// ## Marshalling, slot by slot
///
/// A loop would be smaller, but the register *number* is part of the
/// instruction encoding — there is no "move to register RCX+i" — so the
/// four register slots are unrolled. The stack slots genuinely are a
/// loop, and are emitted as one.
pub fn build_ffi_trampoline_x64(ret: FfiRetClassX64) -> CodeBlob {
    let mut a = X64Assembler::new();

    // Callee-saved scratch this trampoline needs across the native call.
    // RBX = target, RSI = argv, RDI unused, R12/R13 = mask/argc.
    a.emit("push", &[r64(RBP)]);
    a.emit("mov", &[r64(RBP), r64(RSP)]);
    a.emit("push", &[r64(RBX)]);
    a.emit("push", &[r64(RSI)]);

    // Park the four incoming parameters before their registers are reused
    // as the callee's own argument registers.
    a.emit("mov", &[r64(RBX), r64(ARG_REGS[0])]); // target
    a.emit("mov", &[r64(RSI), r64(ARG_REGS[1])]); // argv
    // `class_mask` and `argc` are parked but not yet consulted: every
    // register slot is loaded into BOTH files below, which is correct
    // for any mask AND is what a variadic callee requires. They stay in
    // the signature because a future narrowing marshaller needs them.
    a.emit("mov", &[r64(RAX), r64(ARG_REGS[2])]); // class_mask
    a.emit("mov", &[r64(RDX), r64(ARG_REGS[3])]); // argc

    // Outgoing area: shadow space plus a slot for every stack argument.
    // Sized for the maximum so it is a constant, and 16-aligned so `RSP`
    // is correct at the call. Two pushes after `push rbp` leave RSP at
    // its original phase, so a multiple of 16 preserves it.
    const OUT_BYTES: i64 = 32 + 8 * (ARGV_WORDS as i64 - MAX_REG_ARGS as i64);
    const _: () = assert!(OUT_BYTES % 16 == 0);
    a.emit("sub", &[r64(RSP), imm(OUT_BYTES)]);

    // ── Stack slots 4.. ────────────────────────────────────────────────
    // Emitted before the register slots, because the register moves below
    // clobber RCX/RDX/R8/R9 and RDX still holds `argc` here.
    //
    // Every stack slot is copied unconditionally rather than up to
    // `argc`: the destination is inside this frame's own reservation, so
    // copying a word the callee will never read is harmless, and it
    // avoids a runtime-variable loop in hand-written assembly. Floats and
    // integers share the stack representation, so the mask is irrelevant
    // past slot 3.
    for i in MAX_REG_ARGS..ARGV_WORDS {
        a.emit("mov", &[r64(RCX), mem(RSI, 8 * i as i64)]);
        a.emit(
            "mov",
            &[mem(RSP, 32 + 8 * (i - MAX_REG_ARGS) as i64), r64(RCX)],
        );
    }

    // ── Register slots 0..4 ────────────────────────────────────────────
    // Each slot is loaded into BOTH its integer and its float register,
    // then the wrong one is simply never read by the callee.
    //
    // That is not laziness: it is also exactly what Win64 requires for a
    // VARIADIC callee, which must find a floating-point argument in the
    // integer register as well. Doing it unconditionally makes variadic
    // and non-variadic calls identical and removes a whole class of "the
    // mask was wrong" bug — at the cost of four `movq`s.
    for (i, gpr) in ARG_REGS.iter().enumerate().take(MAX_REG_ARGS) {
        a.emit("mov", &[r64(*gpr), mem(RSI, 8 * i as i64)]);
        a.emit("movq", &[xmm(i as u8), r64(*gpr)]);
    }

    a.emit("call", &[r64(RBX)]);
    a.emit("add", &[r64(RSP), imm(OUT_BYTES)]);

    // A float result arrives in XMM0; hand it back as raw bits so every
    // trampoline shares one `-> u64` Rust signature.
    if ret == FfiRetClassX64::F {
        a.emit("movq", &[r64(RAX), xmm(0)]);
    }

    a.emit("pop", &[r64(RSI)]);
    a.emit("pop", &[r64(RBX)]);
    a.emit("pop", &[r64(RBP)]);
    a.emit("ret", &[]);
    a.finish()
}

/// Resolve a COM method's address: `(*(*this))[index]`.
///
/// A COM interface pointer's first field is `lpVtbl`, a pointer to an
/// array of function pointers, so a method is two loads away and needs no
/// trampoline of its own — the ordinary [`FfiCallFnX64`] call works once
/// the target is known, with `this` in argument slot 0.
///
/// Lives here rather than in `runtime::ffi` because `codecache` is the
/// crate's designated owner of raw pointer work; everything else is under
/// `deny(unsafe_code)`.
///
/// `None` means the vtable pointer was null — i.e. `this` was not a COM
/// interface pointer. A non-null but WRONG pointer cannot be detected
/// here and will fault on use, which is the same contract every FFI call
/// operates under.
///
/// Safe by the same convention as [`FfiStubs::invoke`](crate::codecache::ffi_stubs::FfiStubs::invoke):
/// the whole FFI surface trusts guest-supplied addresses, so wrapping the
/// dereference in an `unsafe fn` would push that judgement outward without
/// making anything safer. The unsafety is contained and documented here.
pub fn com_vtable_slot(this: u64, index: usize) -> Option<u64> {
    if this == 0 {
        return None;
    }
    // SAFETY: the caller's contract above.
    unsafe {
        let vtbl = *(this as *const *const u64);
        if vtbl.is_null() {
            return None;
        }
        Some(*vtbl.add(index))
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::vendor::wfasm::native_windows::WinJit;

    fn place(blob: &CodeBlob) -> (WinJit, u64) {
        let jit = WinJit::with_capacity(blob.code.len() + 4096).expect("RWX");
        let (base, _cap) = jit.region_raw();
        unsafe { core::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, blob.code.len()) };
        (jit, base as u64)
    }

    /// The case the two-buffer AArch64 shape cannot express: an argument
    /// list where a float sits BETWEEN two integers.
    ///
    /// Under Win64 `b` must claim slot 1 (XMM1), which pushes `c` to slot
    /// 2 (R8). A marshaller that partitioned by class would put `c` in
    /// RDX — slot 1 — and the callee would read the float's bit pattern
    /// as an integer. It would not fault; it would just be wrong.
    #[cfg(windows)]
    #[test]
    fn interleaved_int_float_args_land_in_the_right_win64_slots() {
        extern "C" fn probe(a: u64, b: f64, c: u64, d: f64) -> u64 {
            assert_eq!(a, 0x1111, "slot 0 -> RCX");
            assert_eq!(b, 2.5, "slot 1 -> XMM1, NOT RDX");
            assert_eq!(c, 0x3333, "slot 2 -> R8, pushed along by the float");
            assert_eq!(d, 4.5, "slot 3 -> XMM3");
            0xABCD
        }

        let blob = build_ffi_trampoline_x64(FfiRetClassX64::G);
        let (_j, addr) = place(&blob);
        let call: FfiCallFnX64 = unsafe { std::mem::transmute(addr as *const u8) };

        let mut argv = [0u64; ARGV_WORDS];
        argv[0] = 0x1111;
        argv[1] = 2.5f64.to_bits();
        argv[2] = 0x3333;
        argv[3] = 4.5f64.to_bits();
        // bits 1 and 3 are floating point.
        let mask = (1u32 << 1) | (1u32 << 3);

        let got = unsafe { call(probe as usize as u64, argv.as_ptr(), mask, 4) };
        assert_eq!(got, 0xABCD);
    }

    /// Arguments past the fourth go on the stack, above the shadow space.
    #[cfg(windows)]
    #[test]
    fn stack_arguments_reach_the_callee() {
        extern "C" fn probe(
            a0: u64,
            a1: u64,
            a2: u64,
            a3: u64,
            a4: u64,
            a5: u64,
            a6: u64,
        ) -> u64 {
            assert_eq!((a0, a1, a2, a3), (10, 11, 12, 13), "register slots");
            assert_eq!((a4, a5, a6), (14, 15, 16), "stack slots");
            0x5EED
        }

        let blob = build_ffi_trampoline_x64(FfiRetClassX64::G);
        let (_j, addr) = place(&blob);
        let call: FfiCallFnX64 = unsafe { std::mem::transmute(addr as *const u8) };

        let mut argv = [0u64; ARGV_WORDS];
        for (i, w) in argv.iter_mut().enumerate().take(7) {
            *w = 10 + i as u64;
        }
        let got = unsafe { call(probe as usize as u64, argv.as_ptr(), 0, 7) };
        assert_eq!(got, 0x5EED);
    }

    /// A `double` result comes back in XMM0 and must be handed over as
    /// raw bits, since every trampoline shares one `-> u64` signature.
    #[cfg(windows)]
    #[test]
    fn float_return_comes_back_through_xmm0() {
        extern "C" fn probe(a: f64, b: f64) -> f64 {
            a * b
        }

        let blob = build_ffi_trampoline_x64(FfiRetClassX64::F);
        let (_j, addr) = place(&blob);
        let call: FfiCallFnX64 = unsafe { std::mem::transmute(addr as *const u8) };

        let mut argv = [0u64; ARGV_WORDS];
        argv[0] = 1.5f64.to_bits();
        argv[1] = 3.0f64.to_bits();
        let raw = unsafe { call(probe as usize as u64, argv.as_ptr(), 0b11, 2) };
        assert_eq!(f64::from_bits(raw), 4.5);
    }

    /// A real Win32 call, resolved the way the FFI actually resolves one.
    /// `GetTickCount` takes no arguments and returns a DWORD, so it
    /// exercises the whole path end to end with nothing to marshal.
    #[cfg(windows)]
    #[test]
    fn calls_a_real_win32_export() {
        let target = crate::vendor::wfasm::native_windows::dlsym_resolve(None, "GetTickCount")
            .expect("GetTickCount must resolve from kernel32");

        let blob = build_ffi_trampoline_x64(FfiRetClassX64::G);
        let (_j, addr) = place(&blob);
        let call: FfiCallFnX64 = unsafe { std::mem::transmute(addr as *const u8) };

        let argv = [0u64; ARGV_WORDS];
        let a = unsafe { call(target, argv.as_ptr(), 0, 0) } as u32;
        std::thread::sleep(std::time::Duration::from_millis(30));
        let b = unsafe { call(target, argv.as_ptr(), 0, 0) } as u32;
        assert!(a != 0, "GetTickCount returned 0");
        assert!(
            b.wrapping_sub(a) >= 10,
            "tick count did not advance across a 30ms sleep: {a} -> {b}"
        );
    }
}
