//! S20 FFI (docs/FFI.md §5) — the shape-keyed native-call trampolines.
//!
//! One hand-assembled, Rust-callable trampoline PER RETURN CLASS (`g`
//! integer/pointer, `f` float/double, `v` void — docs/FFI.md §1's token
//! vocabulary), not one per call site and not one per exact argument-class
//! sequence: each trampoline unconditionally loads a FIXED 8 GPR argument
//! slots (`x0..x7`) and 8 FPR argument slots (`d0..d7`), and SPILLS a
//! further fixed 8 integer-class slots to the stack (`[sp, #0..56]`,
//! AAPCS64's home for integer args 9+ — the A3 stack-spill tier,
//! docs/accelerate_design.md U2, which is what makes `vDSP_mmulD` and
//! `cblas_dgemm` callable), all from two marshaled buffers before the
//! call. This is sound under AAPCS64 regardless of how many of those slots
//! the real callee's own C signature actually declares — a function reads
//! only the registers and stack words its own prototype names, so
//! supplying extra (unread) argument words in unused slots is always
//! harmless, the same reasoning any general-purpose FFI (libffi, `ctypes`)
//! relies on internally. This collapses what would otherwise be a
//! combinatorial "one trampoline per (g-count, f-count, interleaving)"
//! problem down to exactly 3 fixed blobs, covering every real POSIX
//! function (≤6 `g` args), every plain-numeric Cocoa method, and BLAS/
//! LAPACK drivers up to `METHOD_ARGC_MAX` total args (HFA/struct-by-value
//! shapes — `h2` `h3` `h4` `i1` `i2` `b` `s` — remain Tier 2's problem,
//! deferred: S20 step 7 / docs/FFI.md §3).
//!
//! Deliberately NOT anchored the way `codecache::stubs`'s runtime-stub
//! table is (`VMREG_LAST_COMPILED_FP_OFFSET` etc.): those trampolines are
//! reached FROM compiled Smalltalk code via a `bl`, exposing a live
//! compiled frame a GC must be able to walk mid-call. This trampoline runs
//! the OPPOSITE direction — Rust calls it directly (like `stubs::call_stub`,
//! its closest existing precedent) — with every Smalltalk oop already
//! converted to a plain native word by the caller BEFORE this trampoline
//! ever runs, so no MACVM heap object is reachable only through a register
//! this code touches. No `VmState` involved at all.

use crate::compiler::assembler::{d, imm, mem, mem_post, mem_pre, sp, x, Assembler, CodeBlob};
use crate::compiler::jasm_assembler::JasmAssembler;

use super::{CodeCache, CodeHandle};

/// Every FFI trampoline's own Rust-side signature: `target` is the resolved
/// native function address (S20 step 1's `dlsym_resolve`); `argv_g` points
/// at exactly [`ARGV_G_WORDS`] (16) `u64` words and `argv_f` at exactly 8 —
/// `argv_g[i]` is the `i`th integer-class argument's raw bits (an integer,
/// a pointer, or a `bool`/`char` widened to 64 bits — all `g`-class per
/// docs/FFI.md §1), `argv_f[i]` is the `i`th FPR argument's
/// `f64::to_bits()` (an `f32` is widened to `f64` bits by the caller —
/// AAPCS64 passes it in the LOW 32 bits of the SAME `d`-register a double
/// would use, so this one shape covers both). g-args 0..8 load into
/// `x0..x7`; g-args 8..16 SPILL TO THE STACK per AAPCS64 (`[sp, #0..56]`
/// at call time) — the A3 unlock (docs/accelerate_design.md U2) that makes
/// `vDSP_mmulD` (9 g) and `cblas_dgemm` (12 g + 2 f) callable. The
/// returned `u64` is the raw result: `ret_g` callers use it directly (or
/// narrow/widen per the real return type), `ret_f` callers apply
/// `f64::from_bits`, `ret_v` callers ignore it.
pub type FfiCallFn =
    unsafe extern "C" fn(target: u64, argv_g: *const u64, argv_f: *const u64) -> u64;

/// The g-argument buffer's fixed word count: 8 register slots plus 8
/// stack-spilled slots. One uniform shape for every call (the ~16 extra
/// instructions the spill costs a small call are noise next to the native
/// call itself), so there is no second trampoline family and no per-call
/// shape decision. Bounded comfortably above `METHOD_ARGC_MAX` (15), the
/// real arity ceiling any pragma can declare.
pub const ARGV_G_WORDS: usize = 16;

/// The 3 published trampolines' own addresses, installed once at VM
/// startup (mirrors `codecache::stubs::Stubs`'s own shape).
#[derive(Clone, Copy)]
pub struct FfiStubs {
    ret_g: CodeHandle,
    ret_f: CodeHandle,
    ret_v: CodeHandle,
}

/// Which register class an FFI call's OWN return value uses — the only
/// per-call dimension left once argument marshaling is uniform (both
/// argument buffers are always exactly 8 words, real arity or not).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FfiRetClass {
    /// `g` — integer/pointer/id (docs/FFI.md §1); also covers `v` (void):
    /// the caller simply discards the result, so `ret_g`'s trampoline
    /// (which always leaves SOMETHING valid in `x0`) serves both without a
    /// separate `ret_v` call path being load-bearing. `ret_v` still exists
    /// as its own trampoline (below) for symmetry with the ABI token
    /// vocabulary and so a future caller never has to reason about whether
    /// discarding `G`'s result is safe (it always is, but a dedicated `V`
    /// variant makes that a non-question).
    G,
    /// `f` — float/double; the raw `u64` result is `d0`'s bits, moved into
    /// `x0` by the trampoline itself (`fmov x0, d0`) so every trampoline
    /// shares ONE Rust-side return type regardless of class.
    F,
    /// `v` — void; the callee's own C return type. This trampoline never
    /// reads `x0`/`d0` after the call at all — no register-shuffling to
    /// mis-order, since there is nothing to mis-order.
    V,
}

impl FfiStubs {
    /// Resolved trampoline entry address for `ret_class` — pair with
    /// [`FfiCallFn`]'s `transmute`, exactly [`crate::codecache::stubs::
    /// Stubs::invoke`]'s own calling pattern for `call_stub`.
    pub fn addr_for(&self, ret_class: FfiRetClass) -> u64 {
        match ret_class {
            FfiRetClass::G => self.ret_g.base as u64,
            FfiRetClass::F => self.ret_f.base as u64,
            FfiRetClass::V => self.ret_v.base as u64,
        }
    }

    /// Convenience wrapper mirroring `Stubs::invoke` — resolves the right
    /// trampoline for `ret_class` and calls through it. `argv_g` is always
    /// exactly [`ARGV_G_WORDS`] words and `argv_f` exactly 8 (unused
    /// trailing slots may hold any value at all, per this module's own doc
    /// — never read by a callee whose own C signature doesn't declare that
    /// many arguments).
    pub fn invoke(
        &self,
        ret_class: FfiRetClass,
        target: u64,
        argv_g: &[u64; ARGV_G_WORDS],
        argv_f: &[u64; 8],
    ) -> u64 {
        let entry = self.addr_for(ret_class);
        let call: FfiCallFn = unsafe { std::mem::transmute(entry) };
        unsafe { call(target, argv_g.as_ptr(), argv_f.as_ptr()) }
    }
}

/// Shared prologue every trampoline below starts with: stash the 3
/// incoming Rust-side args into scratch registers (x9-x11, all
/// caller-saved — safe to clobber, and clobbered again immediately by the
/// argument loads that follow) BEFORE x0-x2 get overwritten by the real
/// marshaled arguments, then load all 8 GPR + 8 FPR argument slots. x12 is
/// a second scratch used only to round-trip an `f64`'s raw bits through a
/// GPR before `fmov`ing them into their real FPR home (there is no direct
/// `ldr d0, [mem]` form exercised anywhere in this codebase's own
/// corpus — `fmov` GPR<->FPR bit moves are, so the FP argument path reuses
/// the SAME plain `ldr x_, [...]` this whole codebase already trusts, just
/// followed by one bit-preserving move into the real FP register).
fn emit_ffi_prologue(a: &mut JasmAssembler) {
    a.emit("stp", &[x(29), x(30), mem_pre(31, -16)]);
    a.emit("mov", &[x(29), sp()]);

    a.emit("mov", &[x(9), x(0)]); // target
    a.emit("mov", &[x(10), x(1)]); // argv_g
    a.emit("mov", &[x(11), x(2)]); // argv_f

    // Stack-spilled g args (A3/U2): AAPCS64 places integer args 9+ at
    // `[sp, #0], [sp, #8], …` at call time — carve 64 bytes (16-byte
    // aligned already) and copy argv_g[8..16] down. Unconditional for
    // every call, same reasoning as the fixed 8-register loads below: a
    // callee only reads the stack words its own prototype names, so
    // supplying extras is harmless, and ONE shape beats a per-arity
    // trampoline family.
    a.emit("sub", &[sp(), sp(), imm(64)]);
    for i in 8u8..16 {
        a.emit("ldr", &[x(12), mem(10, 8 * i as i64)]);
        a.emit("str", &[x(12), mem(31, 8 * (i as i64 - 8))]);
    }

    // GPR args: x0..x7 <- argv_g[0..8].
    for i in 0u8..8 {
        a.emit("ldr", &[x(i), mem(10, 8 * i as i64)]);
    }
    // FPR args: d0..d7 <- fmov from argv_f[0..8]'s raw bits (via x12).
    for i in 0u8..8 {
        a.emit("ldr", &[x(12), mem(11, 8 * i as i64)]);
        a.emit("fmov", &[d(i), x(12)]);
    }

    a.emit("blr", &[x(9)]);
}

fn emit_ffi_epilogue(a: &mut JasmAssembler) {
    // Release the stack-spill area BEFORE restoring fp/lr (the prologue
    // carved it after the stp, so tear down in reverse).
    a.emit("add", &[sp(), sp(), imm(64)]);
    a.emit("ldp", &[x(29), x(30), mem_post(31, 16)]);
    a.emit("ret", &[]);
}

/// `ret_class = g` (and `v`, which shares this shape — see [`FfiRetClass::G`]'s
/// own doc): the callee's result is already exactly where the Rust caller
/// wants it, `x0`, the instant `blr` returns. Nothing to do after the call.
fn build_ffi_call_ret_g() -> CodeBlob {
    let mut a = JasmAssembler::new();
    emit_ffi_prologue(&mut a);
    emit_ffi_epilogue(&mut a);
    a.finish()
}

/// `ret_class = f`: the callee's real return value comes back in `d0`
/// (AAPCS64), but every trampoline shares ONE Rust-side `-> u64` shape
/// (this module's own doc) — move `d0`'s raw bits into `x0` before
/// returning, so the Rust caller's `f64::from_bits(result)` recovers the
/// exact value with no precision loss (a bit move, not a numeric convert).
fn build_ffi_call_ret_f() -> CodeBlob {
    let mut a = JasmAssembler::new();
    emit_ffi_prologue(&mut a);
    a.emit("fmov", &[x(0), d(0)]);
    emit_ffi_epilogue(&mut a);
    a.finish()
}

/// `ret_class = v`: the callee's C return type is void — `x0`/`d0` are
/// whatever the callee left them as (uninitialized from THIS call's own
/// point of view), so this trampoline deliberately never reads either
/// after `blr`, unlike `ret_g`/`ret_f` which read exactly one of them.
fn build_ffi_call_ret_v() -> CodeBlob {
    let mut a = JasmAssembler::new();
    emit_ffi_prologue(&mut a);
    emit_ffi_epilogue(&mut a);
    a.finish()
}

/// Build and publish all 3 trampolines into `cache`. Call once, alongside
/// `codecache::stubs::install`, before any FFI primitive can run.
pub fn install(cache: &mut CodeCache) -> FfiStubs {
    let ret_g_blob = build_ffi_call_ret_g();
    let ret_g = cache
        .alloc(ret_g_blob.code.len())
        .expect("ffi_stubs::install: code cache too small for ffi_call_ret_g");
    cache.publish(ret_g, &ret_g_blob);

    let ret_f_blob = build_ffi_call_ret_f();
    let ret_f = cache
        .alloc(ret_f_blob.code.len())
        .expect("ffi_stubs::install: code cache too small for ffi_call_ret_f");
    cache.publish(ret_f, &ret_f_blob);

    let ret_v_blob = build_ffi_call_ret_v();
    let ret_v = cache
        .alloc(ret_v_blob.code.len())
        .expect("ffi_stubs::install: code cache too small for ffi_call_ret_v");
    cache.publish(ret_v, &ret_v_blob);

    FfiStubs {
        ret_g,
        ret_f,
        ret_v,
    }
}

// WINVM: these tests EXECUTE the A64 trampolines — macOS-only until the
// Phase-3 x64 stubs exist.
#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn test_cache() -> CodeCache {
        CodeCache::new(1 << 16).unwrap()
    }

    /// Simplest possible round-trip: a real, no-arg libc call through the
    /// `ret_g` trampoline. Proves the mechanism end-to-end against a REAL
    /// native function (not a test double) before any argument marshaling
    /// is exercised at all.
    #[test]
    fn ret_g_zero_args_calls_real_getpid() {
        extern "C" {
            fn getpid() -> i32;
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let want = unsafe { getpid() };
        let got = stubs.invoke(FfiRetClass::G, getpid as *const () as u64, &[0; ARGV_G_WORDS], &[0; 8]);
        assert_eq!(got as i32, want);
    }

    /// All 8 GPR argument slots, in order, arrive in the exact registers
    /// AAPCS64 promises — a test function that returns which SLOT holds a
    /// sentinel value proves the trampoline's own `ldr x_, [argv_g, #8*i]`
    /// loop is neither off-by-one nor reversed.
    #[test]
    fn ret_g_marshals_all_eight_gpr_args_in_order() {
        extern "C" fn sum8(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64) -> u64 {
            a + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000 + g * 1000000 + h * 10000000
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let mut argv_g = [0u64; ARGV_G_WORDS];
        argv_g[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let got = stubs.invoke(FfiRetClass::G, sum8 as *const () as u64, &argv_g, &[0; 8]);
        assert_eq!(
            got,
            1 + 20 + 300 + 4000 + 50000 + 600000 + 7000000 + 80000000
        );
    }

    /// The A3 stack-spill tier (docs/accelerate_design.md U2): args 9+
    /// pass on the STACK per AAPCS64. A real 12-integer-arg callee proves
    /// both halves land — distinct positional weights make any swapped,
    /// dropped, or mis-offset slot (register OR stack) change the sum. A
    /// mixed g/f callee below it proves the spill leaves the FPR path and
    /// the 16-byte stack alignment intact (a misaligned sp would fault or
    /// corrupt the varargs-free callee's own frame).
    #[test]
    fn ret_g_spills_stack_args_nine_and_beyond() {
        extern "C" fn sum12(
            a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64,
            i: u64, j: u64, k: u64, l: u64,
        ) -> u64 {
            a + b * 2 + c * 4 + d * 8 + e * 16 + f * 32 + g * 64 + h * 128
                + i * 256 + j * 512 + k * 1024 + l * 2048
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let mut argv_g = [0u64; ARGV_G_WORDS];
        for (n, slot) in argv_g.iter_mut().take(12).enumerate() {
            *slot = (n + 1) as u64;
        }
        let got = stubs.invoke(FfiRetClass::G, sum12 as *const () as u64, &argv_g, &[0; 8]);
        let want: u64 = (0..12u64).map(|n| (n + 1) << n).sum();
        assert_eq!(got, want, "a register or stack slot landed wrong");

        extern "C" fn mixed(
            a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64,
            i: u64, j: u64, x: f64, y: f64,
        ) -> u64 {
            a + b + c + d + e + f + g + h + i * 100 + j * 1000 + ((x * y) as u64)
        }
        let mut argv_g = [0u64; ARGV_G_WORDS];
        for (n, slot) in argv_g.iter_mut().take(10).enumerate() {
            *slot = (n + 1) as u64;
        }
        let mut argv_f = [0u64; 8];
        argv_f[0] = 2.5f64.to_bits();
        argv_f[1] = 4.0f64.to_bits();
        let got = stubs.invoke(FfiRetClass::G, mixed as *const () as u64, &argv_g, &argv_f);
        assert_eq!(got, (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) + 900 + 10000 + 10);
    }

    /// FPR args: the `fmov`-via-GPR path (this module's own doc rationale)
    /// must deliver bit-exact doubles into `d0..d7`, mixed with a couple of
    /// GPR args to prove the two register files are independently indexed
    /// (AAPCS64's own rule — a `g` arg never consumes an FPR slot or vice
    /// versa), matching `NSColor colorWithRed:green:blue:alpha:`'s real
    /// shape (docs/FFI.md §1) at a plain-C proxy scale.
    #[test]
    fn ret_f_marshals_fpr_args_and_returns_a_double() {
        extern "C" fn combine(a: u64, x: f64, y: f64, b: u64, z: f64) -> f64 {
            (a as f64) + x + y * 2.0 + (b as f64) * 100.0 + z * 3.0
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let mut argv_g = [0u64; ARGV_G_WORDS];
        argv_g[0] = 7; // a
        argv_g[1] = 2; // b
        let mut argv_f = [0u64; 8];
        argv_f[0] = 1.5f64.to_bits(); // x
        argv_f[1] = 2.0f64.to_bits(); // y
        argv_f[2] = 0.5f64.to_bits(); // z
        let got = f64::from_bits(stubs.invoke(
            FfiRetClass::F,
            combine as *const () as u64,
            &argv_g,
            &argv_f,
        ));
        let want = 7.0 + 1.5 + 2.0 * 2.0 + 2.0 * 100.0 + 0.5 * 3.0;
        assert_eq!(got, want);
    }

    /// `ret_v`: a real side-effecting void call (writes through a pointer
    /// passed as a `g` arg) proves the callee actually ran with the right
    /// arguments even though this trampoline never inspects its return
    /// registers — the OBSERVABLE proof is the side effect, not `x0`.
    #[test]
    fn ret_v_calls_a_real_void_function_with_side_effects() {
        extern "C" fn set_via_ptr(ptr: *mut u64, value: u64) {
            unsafe { *ptr = value };
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let mut cell: u64 = 0;
        let mut argv_g = [0u64; ARGV_G_WORDS];
        argv_g[0] = &mut cell as *mut u64 as u64;
        argv_g[1] = 0xDEAD_BEEF;
        stubs.invoke(
            FfiRetClass::V,
            set_via_ptr as *const () as u64,
            &argv_g,
            &[0; 8],
        );
        assert_eq!(cell, 0xDEAD_BEEF);
    }

    /// A function that only reads its first two args must ignore whatever
    /// garbage sits in slots 2-7 — the "always load all 8, harmless if
    /// unread" invariant this whole design leans on (this module's own doc)
    /// — proven with a deliberately noisy remainder, not zeros, so a latent
    /// bug reading past arity couldn't hide behind an all-zero buffer.
    #[test]
    fn unused_trailing_slots_are_never_read_by_a_narrower_callee() {
        extern "C" fn add2(a: u64, b: u64) -> u64 {
            a + b
        }
        let mut cache = test_cache();
        let stubs = install(&mut cache);
        let mut argv_g = [0xBADu64; ARGV_G_WORDS];
        argv_g[0] = 10;
        argv_g[1] = 20;
        let argv_f: [u64; 8] = [0xBAD; 8];
        let got = stubs.invoke(FfiRetClass::G, add2 as *const () as u64, &argv_g, &argv_f);
        assert_eq!(got, 30);
    }
}
