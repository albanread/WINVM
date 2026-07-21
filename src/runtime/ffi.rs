//! S20 step 4 (docs/FFI.md Â§5): the runtime primitive behind a compiled
//! `<primitive: FFI â€¦>` pragma (S20 step 3, `frontend::codegen::
//! build_ffi_descriptor`) â€” resolves the native symbol, marshals the real
//! Smalltalk call arguments into AAPCS64 register words, calls through the
//! shape-keyed trampolines (S20 step 2, `codecache::ffi_stubs`), and
//! unmarshals the result back into an `Oop`.
//!
//! Reached from `interpreter::send::try_primitive`, which intercepts
//! `MethodOop::primitive() == PRIM_ID_FFI` BEFORE its generic
//! `prim_by_id` lookup â€” that lookup casts a numbered primitive id `as
//! u16` to index `primitives::PRIMITIVES`, and `PRIM_ID_FFI` (`-1i64`)
//! would wrap to `65535` under that cast, silently aliasing whatever real
//! entry (if any) happens to sit at that index instead of ever reaching
//! this module. `try_primitive`'s own doc comment and this module's entry
//! point, [`dispatch_ffi_primitive`], are the two halves of that
//! interception.
//!
//! Compiled (tier-1 JIT) code can never reach an FFI method in the first
//! place: `compiler::driver::eligibility_detail` rejects any method whose
//! `primitive() != 0` (`driver.rs`'s `NoPermanent` arm), and `PRIM_ID_FFI`
//! satisfies that inequality exactly like any real numbered primitive
//! would â€” FFI methods are permanently interpreter-only, so this module
//! never needs to think about a compiled call site, an oop-map, or a GC
//! safepoint mid-call.
//!
//! Error-handling policy, spelled out once here rather than re-litigated
//! at each call site below: this function draws a hard line between two
//! completely different kinds of "this didn't work" â€”
//!   - **Bad Smalltalk-level data that a DIFFERENT call could get right**
//!     (a wrong argument type) follows every other primitive's convention
//!     (`runtime::primitives`'s own module doc): `PrimitiveOutcome::
//!     Fallthrough`, never a Rust panic. Note the line moved in 2026-07:
//!     an args-token/arity MISMATCH used to sit in this bucket too, but it
//!     is baked into the method â€” no call can ever succeed â€” and on an
//!     empty pragma body the Fallthrough masqueraded as success, so it now
//!     fails loud with the second bucket.
//!   - **Missing runtime/feature support or a bad binding** (an ABI shape
//!     token with no trampoline yet, Tier 2 Cocoa dispatch, a symbol that
//!     fails to resolve, a return value no oop can represent) fails LOUD,
//!     naming the missing piece â€” never a silent `Fallthrough` (which,
//!     for an FFI pragma whose generated method body is otherwise EMPTY,
//!     would return the receiver and look exactly like quiet success).
//!     But loud at the GUEST level (`error::guest_fatal`: message + stack
//!     trace + debugger/probe hooks, then a guest-fatal raise an embedded
//!     `VmHandle` recovers as an ordinary `Err`), NOT a Rust `panic!`:
//!     every one of these conditions is reachable from a hand-authored
//!     pragma in ordinary Smalltalk source (all of world/61's Posix
//!     surface is hand-authored; a Workspace typo in a `function:` name
//!     lands exactly on the dlsym arm), and a Workspace-level mistake
//!     must cost that doit, not the whole embedding host. Genuine VM
//!     invariants (the compiler-built descriptor's own shape) stay
//!     `expect`/`panic!`.

use crate::interpreter::send::PrimitiveOutcome;
use crate::memory::alloc;
use crate::oops::layout::{SMI_MAX, SMI_MIN};
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{ArrayOop, DoubleOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

/// The 6 fixed descriptor slot indices (`build_ffi_descriptor`'s own doc,
/// `frontend/codegen.rs`) â€” named here so this module never has a bare
/// `desc.at(4)` whose meaning depends on remembering the layout by heart.
const DESC_KIND: usize = 0;
const DESC_NAME: usize = 1;
// `DESC_CLASS` (2) and `DESC_CLASS_SIDE` (3) are Tier 2-only (ObjC class
// name + classSide flag) â€” Tier 1 dispatch never reads either, per this
// step's own brief.
const DESC_RET: usize = 4;
const DESC_ARGS: usize = 5;
/// The resolved native address, cached by the first call (nil until then â€”
/// `build_ffi_descriptor` allocates the slot nil-filled). A SmallInt of
/// raw address bits: an IMMEDIATE, so the runtime `at_put` needs no write
/// barrier and the cache is GC-inert. Correct to cache forever: the FFI
/// resolves against RTLD_DEFAULT (plus RTLD_GLOBAL dlopens), and a symbol's
/// address in an already-loaded image never changes for the process life.
/// Added 2026-07 (docs/accelerate_design.md U1) â€” dlsym-per-call measured
/// ~14 Âµs, the dominant cost of every small-N Accelerate call.
const DESC_ADDR_CACHE: usize = 6;

/// S20 step 4's entry point, called directly from `interpreter::send::
/// try_primitive` once it has recognized `m.primitive() == PRIM_ID_FFI`.
/// `argc` is the method's real declared arity (`MethodOop::argc()`'s own
/// value, threaded down from the send site exactly like every other
/// primitive receives it) â€” used both to size the read off `vm.stack` and
/// to cross-check the descriptor's own `args` array length below.
///
/// GC-safety note, load-bearing enough to spell out explicitly (contrast:
/// `build_ffi_descriptor` in `codegen.rs` needed real `HandleScope`
/// protection because it built and returned fresh oops WHILE holding
/// other newly-made oops live across further allocation). This function
/// never does that: every oop it ever touches â€” the descriptor's own
/// Symbols (`kind`/`name`/`ret`/each `args` element), the receiver and
/// argument oops read off `vm.stack` â€” is converted to an owned, plain
/// Rust value (`String`, `i64`, `f64`, `u64`) and then DROPPED, well
/// before the one and only allocating step in this whole function (`ret
/// == "f"`'s `alloc::alloc_double` call, right at the very end). By the
/// time that allocation can run, nothing oop-typed from earlier in this
/// function is still alive in a local for a scavenge to invalidate â€”
/// there is nothing here for a `HandleScope` to protect.
pub(crate) fn dispatch_ffi_primitive(vm: &mut VmState, m: MethodOop, argc: u8) -> PrimitiveOutcome {
    // WINVM: the FFI invoke trampolines (`codecache::ffi_stubs`) are still
    // AArch64 machine code â€” executing one on x64 is an instant fault. Fail
    // the doit cleanly until the Phase-3 x64 backend re-emits the stubs
    // (MIGRATION.md Â§4). Guest-fatal, not a prim Fail: same rationale as
    // the Tier-2 arm below â€” the generated method body is empty besides the
    // pragma, so a silent fallthrough would look like a successful send.
    if cfg!(not(target_arch = "aarch64")) {
        crate::runtime::error::guest_fatal(
            vm,
            format!(
                "FFI: native calls aren't available on this platform yet (the invoke \
                 trampolines are AArch64; the x64 backend is Phase 3 â€” MIGRATION.md) â€” \
                 function {name:?}",
                name = sym_text(m.literals().at(DESC_NAME)),
            ),
        );
    }
    let desc = m.literals();

    let kind = sym_text(desc.at(DESC_KIND));
    if kind != "function" {
        // Tier 2 (`kind == "selector"`, ObjC message dispatch) has no
        // runtime support yet (S20 step 7) â€” and unlike a genuinely bad
        // argument, a Tier-2 pragma's generated method body is EMPTY
        // besides the pragma itself, so a silent `Fallthrough` here would
        // return the receiver and look exactly like the send succeeded
        // while doing nothing whatsoever. Loud failure, naming why â€” but a
        // GUEST-fatal one, not a Rust panic: a Tier-2 pragma compiles from
        // ordinary Smalltalk source, so reaching here is a guest program's
        // doing, and it must cost that guest's doit, not the whole
        // embedding host (`error::guest_fatal`'s contract).
        crate::runtime::error::guest_fatal(
            vm,
            format!(
                "FFI: Tier 2 dispatch (kind {kind:?}, selector {name:?}) isn't implemented yet \
                 â€” S20 step 7",
                name = sym_text(desc.at(DESC_NAME)),
            ),
        );
    }
    let name = sym_text(desc.at(DESC_NAME));

    let ret_tok = sym_text(desc.at(DESC_RET));
    let ret_class = match ret_tok.as_str() {
        "g" => crate::codecache::ffi_stubs::FfiRetClass::G,
        "f" => crate::codecache::ffi_stubs::FfiRetClass::F,
        "v" => crate::codecache::ffi_stubs::FfiRetClass::V,
        // A declared shape with no trampoline. The token comes straight
        // from guest source (`ret: #h2` parses fine), so this is a guest
        // mistake/unsupported-feature report, not a VM invariant â€” fatal
        // to the DOIT (recoverable when embedded), never a host panic.
        other => crate::runtime::error::guest_fatal(
            vm,
            format!(
                "FFI: unsupported return-shape token {other:?} for function {name:?} â€” only \
                 \"g\"/\"f\"/\"v\" have a trampoline; struct/HFA return shapes (h2/h3/h4/i1/i2/\
                 b/s) are Tier 2/deferred territory (docs/FFI.md Â§3)"
            ),
        ),
    };

    let args_desc = ArrayOop::try_from(desc.at(DESC_ARGS))
        .expect("runtime::ffi::dispatch_ffi_primitive: descriptor's args slot must be an Array");
    let argc_usize = argc as usize;
    if args_desc.len() != argc_usize {
        // A hand-authored pragma whose declared arg-token list doesn't
        // match the method's own real arity. This used to Fallthrough â€”
        // but the pragma body is empty, so that answered the receiver and
        // masqueraded as success, and unlike a wrong ARGUMENT (which a
        // different call might get right) an arity mismatch is baked into
        // the method: it can never succeed on any call. Found the hard way
        // building world/61a's Accel bindings, where a 4-keyword selector
        // over a 7-token list silently no-opped every vDSP kernel.
        // Guest-fatal, naming both counts.
        crate::runtime::error::guest_fatal(
            vm,
            format!(
                "FFI: function {name:?}'s pragma declares {} arg token(s) but the method \
                 takes {argc_usize} argument(s) â€” the token list must match the selector's \
                 arity exactly",
                args_desc.len(),
            ),
        );
    }

    // Read the real call arguments directly off `vm.stack` â€” deliberately
    // NOT `try_primitive`'s own shared 6-element `buf` (too small for an
    // FFI call's own arity: docs/FFI.md Â§6.3's `mmap` example alone is
    // argc=6, needing 7 slots including the receiver, and this brief's own
    // scope cut keeps that shared hot-path buffer untouched). Index 0 is
    // the receiver â€” for a Tier 1 `#function` call there is no meaningful
    // receiver to marshal (the example's `FFIPosix class` receiver is
    // never touched by the native call), so it's simply skipped; indices
    // `1..=argc` are the real arguments, in declared order.
    let base = vm.stack.sp - argc_usize - 1;

    let mut argv_g = [0u64; crate::codecache::ffi_stubs::ARGV_G_WORDS];
    let mut argv_f = [0u64; 8];
    let mut next_g = 0usize;
    let mut next_f = 0usize;
    for i in 0..argc_usize {
        let arg_oop = vm.stack.get(base + 1 + i);
        let tok = sym_text(args_desc.at(i));
        match tok.as_str() {
            "g" => {
                let Some(word) = marshal_g(arg_oop) else {
                    // Wrong Smalltalk-level argument type (not a
                    // SmallInt) â€” a genuine calling error a Smalltalk
                    // caller could trigger, same convention as every
                    // other primitive's own argument-tag validation.
                    return PrimitiveOutcome::Fallthrough;
                };
                if next_g >= argv_g.len() {
                    // More than ARGV_G_WORDS (16) "g"-class arguments.
                    // Since the A3 stack-spill tier (args 9..16 pass on
                    // the stack â€” docs/accelerate_design.md U2) this is
                    // unreachable from real source: METHOD_ARGC_MAX (15)
                    // caps a pragma's total arity below the buffer.
                    // Defensive and loud all the same.
                    crate::runtime::error::guest_fatal(
                        vm,
                        format!(
                            "FFI: function {name:?} declares more than 16 integer/pointer \
                             (\"g\") args â€” beyond even the stack-spill tier's buffer"
                        ),
                    );
                }
                argv_g[next_g] = word;
                next_g += 1;
            }
            "f" => {
                let Some(word) = marshal_f(arg_oop) else {
                    return PrimitiveOutcome::Fallthrough;
                };
                if next_f >= argv_f.len() {
                    // Same reasoning as the "g" arm above, for the FPR
                    // register file.
                    crate::runtime::error::guest_fatal(
                        vm,
                        format!(
                            "FFI: function {name:?} declares more than 8 float (\"f\") args â€” \
                             args 9+ pass on the stack, which the trampoline does not support \
                             yet (docs/accelerate_design.md U2)"
                        ),
                    );
                }
                argv_f[next_f] = word;
                next_f += 1;
            }
            // Same class as the return-shape case above: a guest-declared
            // token with no marshaling path â€” guest-fatal, not a panic.
            other => crate::runtime::error::guest_fatal(
                vm,
                format!(
                    "FFI: unsupported argument-shape token {other:?} (arg #{i} of function \
                     {name:?}) â€” only \"g\"/\"f\" have a marshaling path today; struct/HFA \
                     argument shapes are Tier 2/deferred territory (docs/FFI.md Â§3)"
                ),
            ),
        }
    }

    // Every argument is now marshaled into `argv_g`/`argv_f`, and no
    // `Fallthrough` (which needs the args left on the stack for the method's
    // bytecode fallback) can happen past this point â€” so restore the operand
    // stack to the receiver slot, the exact convention `try_primitive` applies
    // to a table primitive's own `Ok` (`vm.stack.sp = base`). The FFI path
    // returns straight out of `try_primitive` and so bypassed that truncation;
    // leaving the receiver+args on the stack was masked in the interpreter
    // (the calling method's return truncates them) but diverged a COMPILED
    // caller's static stack model, tripping `enter_compiled`'s sp assert
    // (`compiled_call.rs`) â€” e.g. `Time millisecondClockValue` twice under the
    // JIT. The caller pushes the result at `base`, leaving exactly `[result]`.
    vm.stack.sp = base;

    // Resolve once, cache in the descriptor (slot 6, a SmallInt of raw
    // address bits â€” immediate, so no write barrier). The old
    // resolve-on-every-call scope cut cost ~14 Âµs/call and dominated every
    // small-N Accelerate kernel (docs/accelerate_design.md U1).
    if let Some(cached) = SmallInt::try_from(desc.at(DESC_ADDR_CACHE)) {
        let target = cached.value() as u64;
        let result = vm.ffi_stubs.invoke(ret_class, target, &argv_g, &argv_f);
        return unmarshal_ret(vm, ret_class, result, &name);
    }
    let Some(target) = crate::vendor::wfasm::native::dlsym_resolve(None, &name) else {
        // A `ffi_gen`-generated binding names only functions verified to
        // exist in the real ABI database (docs/FFI.md) â€” but bindings are
        // also HAND-authored every day (all of world/61's Posix surface, a
        // Workspace experiment), and a typo'd symbol name lands exactly
        // here on first call. That is a guest-program mistake: loud, named,
        // fatal to the doit â€” and recoverable when embedded, instead of a
        // Rust panic taking down the whole GUI for a misspelled binding.
        crate::runtime::error::guest_fatal(
            vm,
            format!(
                "FFI: dlsym found no symbol named {name:?} in the process-global namespace \
                 (RTLD_DEFAULT) â€” check the function: name in the pragma"
            ),
        );
    };

    // Cache the resolution (an immediate SmallInt â€” no write barrier
    // needed) so every later call takes the fast path above.
    desc.at_put(DESC_ADDR_CACHE, SmallInt::new(target as i64).oop());

    let result = vm.ffi_stubs.invoke(ret_class, target, &argv_g, &argv_f);
    unmarshal_ret(vm, ret_class, result, &name)
}

/// The return-value unmarshal, shared by the cached-address fast path and
/// the first-call resolve path (factored out when the U1 address cache
/// split dispatch into those two exits).
fn unmarshal_ret(
    vm: &mut VmState,
    ret_class: crate::codecache::ffi_stubs::FfiRetClass,
    result: u64,
    name: &str,
) -> PrimitiveOutcome {
    match ret_class {
        crate::codecache::ffi_stubs::FfiRetClass::V => {
            // `ret_v` callers ignore the trampoline's raw `u64` entirely
            // (`ffi_stubs.rs`'s own doc) â€” the callee's C return type is
            // void, there is no value to unmarshal.
            PrimitiveOutcome::Result(vm.universe.nil_obj)
        }
        crate::codecache::ffi_stubs::FfiRetClass::G => {
            let signed = result as i64;
            if !(SMI_MIN..=SMI_MAX).contains(&signed) {
                // On real macOS/arm64 (48-bit-or-smaller user virtual
                // address space) every real POSIX return value â€” pointers,
                // fds, error sentinels like -1 â€” always fits an SMI's
                // 61-bit magnitude. But a HAND-authored binding can name a
                // function whose full-width u64 return genuinely overflows
                // (strtoull of user data, a hash), so this is
                // guest-reachable, not "can't happen". Silently truncating
                // would corrupt the value far worse than failing loud â€” but
                // the loud failure belongs to the GUEST's doit (recoverable
                // when embedded), not to the host process.
                crate::runtime::error::guest_fatal(
                    vm,
                    format!(
                        "FFI: function {name:?}'s \"g\" return value {signed} overflows \
                         SmallInt's range ({SMI_MIN}..={SMI_MAX}) â€” no BigInt/LargeInteger oop \
                         exists yet to fall back to"
                    ),
                );
            }
            PrimitiveOutcome::Result(SmallInt::new(signed).oop())
        }
        crate::codecache::ffi_stubs::FfiRetClass::F => {
            let v = f64::from_bits(result);
            let d = alloc::alloc_double(vm, v);
            PrimitiveOutcome::Result(d.oop())
        }
    }
}

/// Marshal one `"g"`-class (integer/pointer) Smalltalk argument to its raw
/// register word. `None` means `arg` wasn't a SmallInt â€” a genuine
/// Smalltalk-level calling-convention violation, handled by the caller as
/// `PrimitiveOutcome::Fallthrough`, never a panic (this codebase's
/// established convention: every primitive validates its own argument
/// tags, per `runtime::primitives`).
fn marshal_g(arg: Oop) -> Option<u64> {
    SmallInt::try_from(arg).map(|smi| smi.value() as u64)
}

/// Marshal one `"f"`-class (double) Smalltalk argument to its raw FPR
/// register word â€” a bit-reinterpret via `f64::to_bits`, matching
/// `ffi_stubs.rs`'s own doc that `argv_f[i]` holds `f64::to_bits()`, never
/// a numeric cast. `None` means `arg` wasn't a Double, same Fallthrough
/// convention as [`marshal_g`].
fn marshal_f(arg: Oop) -> Option<u64> {
    DoubleOop::try_from(arg).map(|d| d.value().to_bits())
}

/// Small shared helper: a descriptor slot (or an `args` array element) is
/// always a Symbol (`build_ffi_descriptor`'s own fixed shape) â€” extract
/// its text once, in one place, rather than repeating the
/// `SymbolOop::try_from(...).expect(...).as_string()` idiom at every call
/// site (the same idiom `codegen.rs`'s own `sym_str` test helper uses).
fn sym_text(o: Oop) -> String {
    SymbolOop::try_from(o)
        .expect("runtime::ffi: expected a Symbol oop in the FFI descriptor")
        .as_string()
}

// WINVM: these tests call real libc symbols through the A64 invoke
// trampolines — macOS-only until the Phase-3 x64 stubs exist.
#[cfg(all(test, target_os = "macos"))]
// This test module only (not `dispatch_ffi_primitive` or its helpers
// above, which contain no `unsafe` at all): the `getpid` cross-check test
// below needs a raw `extern "C"` call to compare against, exactly the
// same one-off need `codecache::ffi_stubs`'s own `getpid` test and
// `vendor::wfasm::native_macos`'s own `dlsym_resolve` test already have â€”
// mirrors `runtime::frames`'s own module-scoped `#![allow(unsafe_code)]`
// boundary rationale (a real native call/read has no safe-Rust
// equivalent), just narrowed to `#[cfg(test)]` since production code here
// never needs it.
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::frontend::ast::TopItem;
    use crate::frontend::codegen::compile_method;
    use crate::frontend::parser::parse_file;
    use crate::interpreter::run_method;
    use crate::oops::wrappers::KlassOop;
    use crate::runtime::vm_state::VmOptions;

    fn test_vm() -> VmState {
        VmState::with_options(VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        })
    }

    /// Exactly `codegen.rs`'s own test-module pattern (`test_klass`) â€” a
    /// fresh, empty `Object` subclass to hang a single method off of.
    fn test_klass(vm: &mut VmState, name: &str) -> KlassOop {
        let object_klass = vm.universe.object_klass;
        vm.universe.new_klass(
            object_klass,
            name,
            crate::oops::Format::Slots,
            false,
            crate::oops::layout::HEADER_WORDS,
        )
    }

    /// Exactly `codegen.rs`'s own test-module pattern (`first_method_of`) â€”
    /// parse a one-method class body and pull out its `MethodNode`.
    fn first_method_of(src: &str) -> crate::frontend::ast::MethodNode {
        let items = parse_file(src).expect("parse");
        let TopItem::ClassDef(c) = items.into_iter().next().unwrap() else {
            panic!("expected a class def")
        };
        c.methods.into_iter().next().expect("expected a method")
    }

    /// Compile `src`'s first (and only) method on a fresh test klass named
    /// `klass_name`, then actually RUN it through the real interpreter
    /// send/primitive path (`interpreter::run_method`, the same "compile
    /// then execute and read back the result" helper `codegen.rs`'s own
    /// `run_top` uses for a bare doIt) with `recv`/`args` as the real
    /// call â€” end to end from source text through `try_primitive`'s new
    /// `PRIM_ID_FFI` interception into this module.
    fn compile_and_run(
        vm: &mut VmState,
        klass_name: &str,
        src: &str,
        recv: Oop,
        args: &[Oop],
    ) -> Oop {
        let klass = test_klass(vm, klass_name);
        let mut method = first_method_of(src);
        let m = compile_method(vm, klass, false, &mut method).expect("compile");
        run_method(vm, m, recv, args)
    }

    /// Zero-arg, `ret: #g`, a real libc function â€” the simplest possible
    /// end-to-end round trip through `dispatch_ffi_primitive`, proving
    /// symbol resolution + the `ret_g` trampoline + SMI unmarshaling all
    /// work together against a REAL system call (this sprint's own
    /// established convention: no mocks â€” see `ffi_stubs.rs`'s own
    /// `getpid` test).
    #[test]
    fn ffi_getpid_zero_args_ret_g_matches_real_getpid() {
        extern "C" {
            fn getpid() -> i32;
        }
        let mut vm = test_vm();
        let nil = vm.universe.nil_obj;
        let result = compile_and_run(
            &mut vm,
            "FFIGetpid",
            "Object subclass: FFIGetpid [ \
                getpid [ <primitive: FFI function: #getpid ret: #g args: #()> ] \
            ]",
            nil,
            &[],
        );
        let want = unsafe { getpid() } as i64;
        let got = SmallInt::try_from(result)
            .expect("expected a SmallInt result")
            .value();
        assert_eq!(got, want);
    }

    /// One `g`-class argument, exercising the real GPR marshal path
    /// (`marshal_g` -> `argv_g[0]`) â€” `llabs(-5) == 5`, a real libc call,
    /// not a test double.
    #[test]
    fn ffi_llabs_one_g_arg_marshals_gpr_correctly() {
        let mut vm = test_vm();
        let nil = vm.universe.nil_obj;
        let arg = SmallInt::new(-5).oop();
        let result = compile_and_run(
            &mut vm,
            "FFILlabs",
            "Object subclass: FFILlabs [ \
                llabsOf: n [ <primitive: FFI function: #llabs ret: #g args: #(g)> ] \
            ]",
            nil,
            &[arg],
        );
        let got = SmallInt::try_from(result)
            .expect("expected a SmallInt result")
            .value();
        assert_eq!(got, 5);
    }

    /// One `f`-class argument AND `f`-class return in the SAME call â€”
    /// exercises the FPR marshal path (`marshal_f` -> `argv_f[0]`) end to
    /// end, including the allocating `ret == "f"` unmarshal step
    /// (`alloc::alloc_double`). `fabs(-3.5) == 3.5`, a real libc call.
    #[test]
    fn ffi_fabs_one_f_arg_ret_f_marshals_fpr_correctly() {
        let mut vm = test_vm();
        let nil = vm.universe.nil_obj;
        let arg = alloc::alloc_double(&mut vm, -3.5).oop();
        let result = compile_and_run(
            &mut vm,
            "FFIFabs",
            "Object subclass: FFIFabs [ \
                fabsOf: x [ <primitive: FFI function: #fabs ret: #f args: #(f)> ] \
            ]",
            nil,
            &[arg],
        );
        let got = DoubleOop::try_from(result)
            .expect("expected a Double result")
            .value();
        assert_eq!(got, 3.5);
    }

    // The args-arity-mismatch Fallthrough test that lived here migrated to
    // the embed gate (case 6 of ffi_guest_mistakes_recover_as_errors_not_
    // host_panics) when the mismatch became a GUEST fatal: on an
    // empty-bodied pragma method, Fallthrough answered the receiver and
    // masqueraded as success â€” found live when world/61a's first-draft
    // Accel bindings (4-keyword selectors over 7-token lists) silently
    // no-opped every vDSP kernel. Note the mismatch IS reachable from the
    // real compiler: the pragma's token list and the selector's arity are
    // authored independently.

    // The >8-g-args story, in three acts (docs/accelerate_design.md):
    // pre-A0 a 9-g-arg pragma FELL THROUGH here â€” an empty-bodied pragma
    // method answering the receiver, masquerading as success (found live:
    // vDSP_mmulD silently no-opped). A0 made it a loud guest fatal.
    // A3's stack-spill trampoline tier then made 9..15 g args genuinely
    // WORK (args 9+ pass on the stack), so today only >16 g (unreachable â€”
    // METHOD_ARGC_MAX is 15) and >8 f raise. The live gates: embed case 5
    // asserts a 9-g-arg binding SUCCEEDS, and ffi_stubs's
    // ret_g_spills_stack_args_nine_and_beyond pins the spill itself.

    // The unsupported-shape / Tier-2 / typo'd-symbol paths were once
    // `#[should_panic]` tests here; they now raise a GUEST fatal
    // (`error::guest_fatal` â€” recoverable when embedded, `fatal_exit` in
    // plain CLI use), which a bare `test_vm()` cannot observe without
    // killing the test process. Their gates live in `embed::tests`
    // (`ffi_guest_mistakes_recover_as_errors_not_host_panics`), where the
    // recovery contract they now follow is the very thing under test.

    // A `ret: #v` end-to-end `.mst`-level test is deliberately deferred:
    // there is no side-effect-observable, pointer-free void libc function
    // to call yet that this test module could verify actually ran (a real
    // void POSIX function worth calling â€” e.g. writing through a pointer
    // argument â€” needs a byte-array/pointer argument representation, which
    // is S20 step 5's Alien work, not built yet). `FfiRetClass::V`'s own
    // unmarshal arm above (`PrimitiveOutcome::Result(vm.universe.nil_obj)`)
    // is exercised by `ffi_stubs.rs`'s own lower-level `ret_v` trampoline
    // test in the meantime. Revisit once step 5 lands.
}
