//! S20 step 5 (docs/FFI.md §4 "Representation"): `Alien`, MACVM's built-in
//! FFI byte-level memory-access type, and its 9 typed accessor/constructor
//! primitives (ids 112-120, `runtime::primitives::PRIMITIVES`).
//!
//! An `Alien` wraps a fixed-size block of bytes that is EITHER:
//!   - **direct**: the object's own indexable byte tail (allocated ON the
//!     MACVM heap, scanned/moved by GC exactly like any `ByteArray` body —
//!     `Universe::alien_klass`'s own doc comment covers the klass shape);
//!   - **indirect**: a raw external (non-heap, non-GC-owned) address —
//!     typically a `mmap`'d page or a `malloc`'d buffer obtained via S20
//!     step 4's `<primitive: FFI ...>` call mechanism — that this Alien's
//!     accessors read/write through directly via raw pointer arithmetic.
//!
//! `AlienOop::external_addr()` (`oops::wrappers`) is the single bit that
//! distinguishes the two: `0` means direct, nonzero means indirect and IS
//! the real address. Every accessor below computes its base address from
//! that one check and is otherwise indifferent to which mode it's in.
//!
//! ## Why this design, not a literal port of real Strongtalk's `Alien`
//!
//! Real Strongtalk (`Alien.dlt`, if that checkout exists at
//! `~/claudeprojects/strongtalk-repo/StrongtalkSource/Alien.dlt`) encodes
//! direct-vs-indirect via a SIGN-FLIPPED size field shared with the
//! object's own ordinary indexable size slot (negative = indirect, an
//! `addressField` holds the real pointer, `size.abs()` is the real byte
//! count). Porting that literally would mean teaching `oops::heap::
//! MemOop::raw_size_slot`/`indexable_len` — functions EVERY `ByteArrayOop`/
//! `String`/`Symbol` in the entire VM depends on — to interpret magnitude
//! via `.abs()`, a small but genuinely shared-blast-radius change for a
//! feature only Alien needs.
//!
//! Instead, this design reuses a mechanism MACVM already has and already
//! trusts for exactly this shape of problem: `CompiledMethod`'s own klass
//! (`Format::Method`) already mixes several named header fields with a
//! trailing indexable byte tail in ONE object, via the generic `nis_words`
//! parameter every klass takes at creation. `Alien` reuses that SAME
//! mechanism (`Format::IndexableBytes`, not a new `Format` variant) with
//! exactly ONE extra named field holding a raw external address — ZERO
//! changes needed to `raw_size_slot`/`indexable_len`/`tail_byte_at`/
//! `tail_start_word` (`oops::heap`), all of which are already klass-nis-
//! aware and already correctly handle `Method`'s own 7-named-field-plus-
//! tail shape. See `Universe::alien_klass`'s own doc comment
//! (`memory::universe`) for the exact klass-shape derivation, and
//! `oops::layout::ALIEN_EXTERNAL_ADDR_INDEX`/`ALIEN_NAMED_WORDS` for the
//! named-field/nis_words constants this module and `oops::wrappers::
//! AlienOop` both depend on.
//!
//! ## Uniform bounds/size, even for indirect Aliens
//!
//! Bounds/size for BOTH direct and indirect Aliens is uniformly
//! `indexable_len()` (the format's own regular, already-existing size
//! slot). For a direct Alien this is simply the real number of tail bytes
//! physically allocated. For an INDIRECT Alien, [`prim_alien_for_address_size`]
//! allocates a REAL tail of that same declared size too, even though it is
//! never read or written (the real bytes live at the external address
//! instead) — this deliberately WASTES real heap space equal to the
//! wrapped region's size for indirect Aliens (e.g. wrapping a 4KB `mmap`'d
//! page wastes 4KB of real VM heap too). This is a documented, deliberate
//! tradeoff for this step, not an oversight: it avoids a second named field
//! and keeps bounds-checking, allocation, and GC scanning IDENTICAL between
//! direct and indirect modes with zero special-casing. A future
//! optimization could give indirect Aliens a near-zero real tail if the
//! wasted memory ever matters — that would need a second named field back
//! at that point, not now.
//!
//! ## Error-handling policy
//!
//! Unlike `runtime::ffi`'s own two-tier panic-vs-`Fallthrough` policy
//! (which exists because FFI has a genuine "feature doesn't exist yet"
//! dimension — unsupported ABI tokens, Tier 2), these 9 primitives have NO
//! such dimension: every possible failure here (wrong argument type,
//! out-of-range index) is ordinary bad Smalltalk-level data. Every one of
//! them follows `runtime::primitives`'s own established, uniform
//! convention exactly (its own module doc: "every `PrimFn` validates its
//! own receiver/arg tags and formats — a violation is always
//! `PrimResult::Fail`, never a Rust panic") — matching `prim_byte_at`'s own
//! existing style: 1-based index convention, bounds-checked against
//! `indexable_len()`.
//!
//! ## Address computation and the one allocating accessor
//!
//! For the DIRECT case, every accessor composes its read/write from the
//! EXISTING `MemOop::tail_byte_at`/`set_tail_byte_at` (`oops::heap`) calls —
//! these already correctly handle Alien's own `ALIEN_NAMED_WORDS`-shifted
//! tail start, so a 4-byte read is 4 individual `tail_byte_at` calls
//! assembled little-endian, an 8-byte read is 8. No new raw-pointer-
//! returning method is added to `oops::heap` for this: that module's own
//! doc says it is "the ONLY module in the crate that dereferences heap
//! addresses", and composing from its already-published byte-level API
//! keeps that invariant intact with zero new surface there.
//!
//! For the INDIRECT case, this genuinely IS foreign (non-heap, non-GC-
//! owned) memory — the small, clearly-justified `unsafe` blocks below
//! (confined to this module, matching how `codecache::ffi_stubs` and
//! `vendor::wfasm::native_macos` already confine their own unsafe native-
//! memory work) do a raw pointer read/write for however many bytes the
//! accessor needs. A bad/garbage `external_addr` (or one from an munmap'd/
//! freed region) can genuinely crash the whole process here — this is an
//! accepted, inherent risk of wrapping arbitrary native memory (the same
//! risk class S20 step 4's own FFI call primitive already accepted for
//! calling an arbitrary resolved native function), not something this step
//! attempts to further guard against.
//!
//! [`prim_double_at`] is the ONE allocating accessor (it must wrap its
//! result in a real `DoubleOop` via `memory::alloc::alloc_double`) — see
//! its own doc comment for the GC-safety ordering this requires. Every
//! other accessor (`byteAt:`, `signedLongAt:`, `size`, and both `...:put:`
//! writers) never allocates at all.

use crate::memory::alloc;
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::AlienOop;
use crate::oops::Oop;
use crate::runtime::primitives::PrimResult;
use crate::runtime::vm_state::VmState;

/// The base address this Alien's accessors should read/write through:
/// `external_addr()` if nonzero (indirect — a real foreign address), else
/// the object's own tail start address (direct — an ordinary heap read/
/// write through `tail_byte_at`/`set_tail_byte_at`).
///
/// Returns `None` for the direct case (there is no meaningful "address" to
/// hand back — heap objects can move under GC, so callers needing the
/// direct path must go through `tail_byte_at`/`set_tail_byte_at`, never a
/// cached raw pointer) paired with `Some(addr)` for the indirect case.
/// Every accessor below branches on this exactly once, right before doing
/// its real work.
fn indirect_base(a: AlienOop) -> Option<u64> {
    let addr = a.external_addr();
    if addr == 0 {
        None
    } else {
        Some(addr)
    }
}

/// The Alien's declared byte size — the single bounds limit every accessor
/// checks against. Direct: the real tail length (`indexable_len`).
/// Indirect: the `ALIEN_INDIRECT_SIZE_INDEX` named field, because since
/// 2026-07 an indirect Alien's real tail is ZERO bytes (wrapping an 8 MB
/// mmap region used to waste 8 MB of real heap — the deliberate v1
/// tradeoff this module's doc describes, retired for the Accelerate
/// design's multi-megabyte NativeFloatArray regions).
fn effective_len(a: AlienOop) -> usize {
    match indirect_base(a) {
        None => a.as_mem().indexable_len(),
        Some(_) => a.indirect_size(),
    }
}

/// Common receiver/index validation every accessor needs: `args[0]` must be
/// an `Alien`, `args[1]` must be a SmallInt whose 1-based value `i` fits
/// within `[1, len - width + 1]` for a `width`-byte access starting at
/// index `i` (1-based, matching `prim_byte_at`'s own convention exactly).
/// Returns the validated `(alien, zero_based_byte_offset)` pair, or `None`
/// on ANY validation failure (wrong type, non-positive index, access that
/// would run off the end of the declared `indexable_len()`) — the caller
/// turns `None` into `PrimResult::Fail` uniformly.
fn validate_access(args: &[Oop], width: usize) -> Option<(AlienOop, usize)> {
    let a = AlienOop::try_from(args[0])?;
    let idx = SmallInt::try_from(args[1])?;
    let i = idx.value();
    if i < 1 {
        return None;
    }
    let zero_based = (i - 1) as usize;
    let len = effective_len(a);
    // `zero_based + width > len` (rather than `zero_based >= len`) catches
    // a multi-byte access that starts in-bounds but would read/write past
    // the declared end — e.g. `signedLongAt: len` on an 8-byte Alien must
    // fail, not silently read 7 real bytes plus 1 past the end.
    if zero_based.checked_add(width)? > len {
        return None;
    }
    Some((a, zero_based))
}

/// Reads `width` bytes starting at zero-based byte offset `offset`,
/// assembled little-endian into a `u64` (callers needing fewer than 8
/// bytes, e.g. `byteAt:`'s `width == 1`, just mask/cast the low bits back
/// down — every call site below already only reads the width it declared).
///
/// Direct case: `width` individual `MemOop::tail_byte_at` calls — see this
/// module's own top-of-file doc for why that's the right way to read a
/// direct Alien's bytes (never a new raw-pointer method on `oops::heap`).
///
/// Indirect case: a raw pointer read of `width` bytes from `base + offset`.
/// # Safety justification (indirect case)
/// `base` is a real process address (an already-resolved `mmap`/`malloc`
/// result, or whatever else `Alien class >> forAddress:size:`'s caller
/// supplied) that this module has no way to independently validate — the
/// same trust boundary S20 step 4's own FFI call primitive already
/// accepts for calling an arbitrary resolved native function address. The
/// read is bounds-checked against Alien's own DECLARED size
/// (`validate_access`, above) but NOT against the real mapped region's
/// actual size/permissions/lifetime — a stale, freed, or undersized
/// mapping genuinely crashes the process here, by design (this module's
/// own top-of-file doc elaborates).
#[allow(unsafe_code)]
fn read_bytes(a: AlienOop, offset: usize, width: usize) -> u64 {
    match indirect_base(a) {
        None => {
            let mut acc = 0u64;
            for i in 0..width {
                let b = a.as_mem().tail_byte_at(offset + i);
                acc |= (b as u64) << (i * 8);
            }
            acc
        }
        Some(base) => {
            let mut acc = 0u64;
            for i in 0..width {
                // SAFETY: see this function's own doc comment — `base` is
                // an opaque, caller-supplied native address; MACVM cannot
                // verify it points at `width` real readable bytes. This is
                // an accepted, inherent risk of wrapping arbitrary native
                // memory, not a bug in this read.
                let b = unsafe { *((base as usize + offset + i) as *const u8) };
                acc |= (b as u64) << (i * 8);
            }
            acc
        }
    }
}

/// Writes `width` bytes of `value` (little-endian, low byte first) starting
/// at zero-based byte offset `offset` — the write-side twin of
/// [`read_bytes`]; see that function's own doc comment for the direct-vs-
/// indirect split and the indirect case's safety justification (identical
/// reasoning, mirrored for a write instead of a read).
#[allow(unsafe_code)]
fn write_bytes(a: AlienOop, offset: usize, width: usize, value: u64) {
    match indirect_base(a) {
        None => {
            for i in 0..width {
                let b = ((value >> (i * 8)) & 0xFF) as u8;
                a.as_mem().set_tail_byte_at(offset + i, b);
            }
        }
        Some(base) => {
            for i in 0..width {
                let b = ((value >> (i * 8)) & 0xFF) as u8;
                // SAFETY: see `read_bytes`'s own doc comment — same opaque-
                // address trust boundary, mirrored for a write.
                unsafe { *((base as usize + offset + i) as *mut u8) = b };
            }
        }
    }
}

// --- id 112/113: byteAt: / byteAt:put: -------------------------------------

/// `Alien>>byteAt:` (id 112, argc 1) — an unsigned byte read, 1-based index
/// (matching `prim_byte_at`'s own convention). Never allocates.
pub(crate) fn prim_alien_byte_at(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((a, offset)) = validate_access(args, 1) else {
        return PrimResult::Fail;
    };
    let b = read_bytes(a, offset, 1) as u8;
    PrimResult::Ok(SmallInt::new(b as i64).oop())
}

/// `Alien>>byteAt:put:` (id 113, argc 2). `args[2]` must be a SmallInt in
/// `0..=255` — an out-of-range value (negative, or >= 256) is exactly the
/// kind of bad Smalltalk-level data this module's uniform policy treats as
/// `PrimResult::Fail`, not a silent truncation (silently masking to 8 bits
/// would let `byteAt:put: 300` succeed and store `44`, which is far more
/// surprising than a clean failure). Never allocates.
pub(crate) fn prim_alien_byte_at_put(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((a, offset)) = validate_access(args, 1) else {
        return PrimResult::Fail;
    };
    let Some(v) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let raw = v.value();
    if !(0..=255).contains(&raw) {
        return PrimResult::Fail;
    }
    write_bytes(a, offset, 1, raw as u64);
    PrimResult::Ok(args[0])
}

// --- id 114/115: signedLongAt: / signedLongAt:put: --------------------------

/// `Alien>>signedLongAt:` (id 114, argc 1) — an 8-byte, little-endian,
/// SIGNED read (a real machine `long`/pointer-width value — `docs/FFI.md`
/// §4's own `signedLongAt:` naming). The raw `u64` composed by
/// [`read_bytes`] is reinterpreted as `i64` bit-for-bit (`as i64`, not a
/// numeric cast — matching `runtime::ffi`'s own `f64::from_bits` idiom for
/// its FPR unmarshal, just for the integer case), so the sign bit round-
/// trips correctly. That `i64` must ALSO fit `SmallInt`'s own narrower
/// 61-bit range (`SmallInt::new` panics outside `[SMI_MIN, SMI_MAX]` — see
/// `oops::smi`) — a real 64-bit value from arbitrary foreign memory could
/// exceed it, so this validates the range explicitly and fails cleanly
/// rather than letting `SmallInt::new` panic on ordinary (if unusual)
/// foreign-memory data. Never allocates.
pub(crate) fn prim_alien_signed_long_at(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    use crate::oops::layout::{SMI_MAX, SMI_MIN};
    let Some((a, offset)) = validate_access(args, 8) else {
        return PrimResult::Fail;
    };
    let raw = read_bytes(a, offset, 8) as i64;
    if !(SMI_MIN..=SMI_MAX).contains(&raw) {
        return PrimResult::Fail;
    }
    PrimResult::Ok(SmallInt::new(raw).oop())
}

/// `Alien>>signedLongAt:put:` (id 115, argc 2) — the write-side twin.
/// `args[2]` is a SmallInt (any in-range value is representable as a real
/// 8-byte signed long, since `SmallInt`'s own 61-bit range is strictly
/// narrower than `i64`), written little-endian via `write_bytes`. Never
/// allocates.
pub(crate) fn prim_alien_signed_long_at_put(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((a, offset)) = validate_access(args, 8) else {
        return PrimResult::Fail;
    };
    let Some(v) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    write_bytes(a, offset, 8, v.value() as u64);
    PrimResult::Ok(args[0])
}

// --- id 116/117: doubleAt: / doubleAt:put: ----------------------------------

/// `Alien>>doubleAt:` (id 116, argc 1) — THE one allocating accessor in
/// this whole module: its result must be wrapped in a real `DoubleOop` via
/// `memory::alloc::alloc_double`, exactly like `runtime::ffi`'s own
/// `ret == "f"` unmarshal arm.
///
/// GC-safety ordering, load-bearing enough to spell out explicitly here
/// (the exact BUG-A-class hazard S20 steps 3 and 4 each found and fixed
/// once already this sprint): the raw `f64` value is extracted (via
/// [`read_bytes`], direct or indirect) into a plain Rust local FIRST, and
/// only THEN does `alloc_double` run. Nothing here holds the Alien oop, or
/// any address/offset computed from it, in a local that gets read AGAIN
/// after `alloc_double` — a scavenge inside `alloc_double` can relocate a
/// DIRECT Alien, since it's an ordinary GC-heap object like any other, and
/// `read_bytes` has already finished touching `a` by the time `alloc_double`
/// is called below. (An INDIRECT Alien's `external_addr` is a plain `u64`
/// copied out of `a` by `read_bytes`'s own `indirect_base` call, so there is
/// no lingering heap reference in that case either — but the ordering
/// discipline is identical either way, and enforced the same way: read
/// everything out of the oop first, allocate second, never the reverse.)
///
/// Every OTHER accessor in this module (`byteAt:`, `signedLongAt:`, `size`,
/// and both `...:put:` writers) never allocates at all, so this ordering
/// concern is specific to this one function alone.
pub(crate) fn prim_alien_double_at(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((a, offset)) = validate_access(args, 8) else {
        return PrimResult::Fail;
    };
    // Extract the raw bits into a plain Rust local BEFORE any allocation —
    // see this function's own doc comment above.
    let raw = read_bytes(a, offset, 8);
    let v = f64::from_bits(raw);
    let d = alloc::alloc_double(vm, v);
    PrimResult::Ok(d.oop())
}

/// `Alien>>doubleAt:put:` (id 117, argc 2) — never allocates (unlike its
/// `doubleAt:` sibling): the incoming `Double` argument is already a real
/// oop the caller (`try_primitive`) is keeping alive on the stack, and this
/// function only ever reads its bits out (`DoubleOop::value`) then writes
/// raw bytes — no allocation, so none of `doubleAt:`'s ordering concern
/// applies here.
pub(crate) fn prim_alien_double_at_put(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some((a, offset)) = validate_access(args, 8) else {
        return PrimResult::Fail;
    };
    let Some(d) = crate::oops::wrappers::DoubleOop::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    write_bytes(a, offset, 8, d.value().to_bits());
    PrimResult::Ok(args[0])
}

// --- id 118: size ------------------------------------------------------------

/// `Alien>>size` (id 118, argc 0) — the format's own regular size slot
/// (`indexable_len()`), uniformly meaningful for both direct and indirect
/// Aliens (this module's own top-of-file doc explains why an indirect
/// Alien's declared size is real, physically-backed heap space too, not
/// just a remembered number). Never allocates.
pub(crate) fn prim_alien_size(_vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(a) = AlienOop::try_from(args[0]) else {
        return PrimResult::Fail;
    };
    PrimResult::Ok(SmallInt::new(effective_len(a) as i64).oop())
}

// --- id 119/120: Alien class >> new: / forAddress:size: ---------------------

/// `Alien class >> new: n` (id 119, argc 1) — a fresh DIRECT Alien with `n`
/// real tail bytes (nil/zero-filled by `alloc::alloc_indexable_bytes`,
/// matching every other indexable allocation's own zero-fill convention).
/// `ALIEN_EXTERNAL_ADDR_INDEX` is left at its zero-filled default (`0` —
/// SmallInt zero is an all-zero-bits raw oop, SPEC §2.1, so no explicit
/// write is even needed), marking it direct. `n` must be a non-negative
/// SmallInt — anything else is `PrimResult::Fail`, the same uniform policy
/// as every other primitive in this module. Allocates (the fresh Alien
/// itself) but touches no OTHER oop first, so there is no BUG-A-class
/// ordering hazard here (contrast `prim_alien_double_at`, above).
pub(crate) fn prim_alien_new(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(n) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let nbytes = n.value();
    if nbytes < 0 {
        return PrimResult::Fail;
    }
    let klass = vm.universe.alien_klass;
    let obj = alloc::alloc_indexable_bytes(vm, klass, nbytes as usize);
    // `alloc_indexable_bytes`'s own doc: the object's REAL klass field is
    // whatever klass is passed in (here, `alien_klass`, `Format::
    // IndexableBytes` — exactly what `AlienOop::try_from` checks), so this
    // re-wrap is a Rust-side static-type fix-up only, never a real klass
    // mismatch.
    let a = AlienOop::try_from(obj.oop())
        .expect("prim_alien_new: freshly allocated with alien_klass must be a valid AlienOop");
    PrimResult::Ok(a.oop())
}

/// `Alien class >> forAddress: addr size: n` (id 120, argc 2) — a fresh
/// INDIRECT Alien wrapping the real external address `addr`. Its real tail
/// is ZERO bytes: the declared size lives in the `ALIEN_INDIRECT_SIZE_INDEX`
/// named field, which `effective_len` (and so every bounds check plus
/// `size`) reads instead of `indexable_len` for indirect Aliens. (Until
/// 2026-07 this allocated a real n-byte tail that was never touched — a
/// deliberate v1 tradeoff whose cost became untenable when the Accelerate
/// design started wrapping multi-megabyte mmap regions.)
/// `addr` and `n` are both SmallInts — `addr`'s own range is trusted the
/// same way S20 step 4's own FFI call primitive already trusts a real
/// native address returned as a plain SmallInt for `ret: #g` (confirmed
/// real-address-range-safe there; no extra range validation is needed here
/// beyond "is it a SmallInt"). `n` must be non-negative, same policy as
/// `new:`. Allocates the fresh Alien; `addr`'s own value is a plain `i64`
/// local by the time `set_external_addr` runs, not a live oop reference, so
/// there is no BUG-A-class ordering hazard here either.
pub(crate) fn prim_alien_for_address_size(vm: &mut VmState, args: &[Oop]) -> PrimResult {
    let Some(addr) = SmallInt::try_from(args[1]) else {
        return PrimResult::Fail;
    };
    let Some(n) = SmallInt::try_from(args[2]) else {
        return PrimResult::Fail;
    };
    let nbytes = n.value();
    if nbytes < 0 {
        return PrimResult::Fail;
    }
    let addr_value = addr.value();
    let klass = vm.universe.alien_klass;
    let obj = alloc::alloc_indexable_bytes(vm, klass, 0);
    let a = AlienOop::try_from(obj.oop()).expect(
        "prim_alien_for_address_size: freshly allocated with alien_klass must be a valid AlienOop",
    );
    a.set_external_addr(addr_value as u64);
    a.set_indirect_size(nbytes as usize);
    PrimResult::Ok(a.oop())
}

// --- Bootstrap: Alien's own methods, compiled from an embedded source ------
//
// `world/*.mst` (top-level files) is off-limits for this step — every OTHER
// built-in's real methods normally live there (e.g. `SmallInteger>>+` is in
// `world/06_smallinteger.mst`), but Alien's methods must NOT create or
// touch anything under `world/` this step. Instead, they are compiled from
// an EMBEDDED RUST STRING CONSTANT at VM boot, using the exact same
// production compilation pipeline real `world/*.mst` files go through —
// just sourced from a `&str` literal in this file instead of a loaded file.
// This was an explicit, deliberate decision (confirmed before writing this
// module), not a shortcut to route around silently.
//
// Why a two-part genesis-klass-plus-reopen-for-methods split is necessary
// at all (rather than one ordinary `Object subclass: Alien [ ... ]` with an
// `| ivars |` pragma, the way most of `world/*.mst` declares its classes):
// Alien's real shape (`Format::IndexableBytes`, `ALIEN_NAMED_WORDS`) can
// only be expressed via `Universe::genesis`'s own `remaining_klass!`
// machinery (`memory::universe`) — the ordinary `subclass:` parser has no
// syntax for choosing a `Format` or a raw `nis_words` value, only for
// declaring named instance variables under `Format::Slots`. So genesis
// fixes Alien's SHAPE first (`Universe::alien_klass`), and this function
// only adds its METHODS afterward, once "Alien" already resolves as a
// global (`bootstrap_well_known`, `runtime::globals`, which this function's
// own caller — `VmState::with_options` — always runs first).
//
// Every method below carries an explicit `^self` after its pragma — NOT
// decorative, and NOT the same "empty body, forward-declared" minimalism
// `ffi_gen`'s own generated `<primitive: FFI ...>` bindings use. Those FFI
// pragma methods dispatch through `runtime::ffi::dispatch_ffi_primitive`, a
// SEPARATE path `interpreter::send::try_primitive` reaches BEFORE its
// generic `prim_by_id` match arms — an empty body there never touches the
// generic path's own `debug_assert!(m.prim_fails(), ...)` at all. Alien's
// primitives (112-120) are ORDINARY numbered primitives going through that
// SAME generic path every other primitive in this codebase uses, and
// `frontend::codegen`'s own `prim_fails = primitive.is_some() &&
// !method.body.is_empty()` computes `false` for a body with zero explicit
// statements — so a genuinely EMPTY body here (as this file originally,
// wrongly, had) means the very first real out-of-range `byteAt:`/bad-typed
// `doubleAt:put:`/etc. call trips that debug_assert and panics, instead of
// falling through to a real fallback body. Found by actually running the
// new `world/tests/30_ffi_alien_tests.mst` smoke test (S20 step 6) through
// the real world-loaded test harness (`tests/it_world.rs`'s `suite_green`)
// — none of this module's own unit tests caught it, since every one of
// them calls the primitive Rust function directly, bypassing full
// dispatch entirely. `^self` matches this codebase's own universal
// convention for every other primitive-backed method in `world/*.mst`
// (e.g. `SmallInteger>>+`'s `<primitive: 1> ^self`) — on success the
// primitive already returns before reaching this bytecode; `^self` only
// ever runs on a genuine `PrimResult::Fail`, giving these accessors the
// same "answers the receiver on bad data" fallback every other Fail-able
// primitive already has.
const ALIEN_BOOTSTRAP_SRC: &str = "Object subclass: Alien [ \
    Alien class >> new: n [ <primitive: 119> ^self ] \
    Alien class >> forAddress: addr size: n [ <primitive: 120> ^self ] \
    byteAt: i [ <primitive: 112> ^self ] \
    byteAt: i put: v [ <primitive: 113> ^self ] \
    signedLongAt: i [ <primitive: 114> ^self ] \
    signedLongAt: i put: v [ <primitive: 115> ^self ] \
    doubleAt: i [ <primitive: 116> ^self ] \
    doubleAt: i put: v [ <primitive: 117> ^self ] \
    size [ <primitive: 118> ^self ] \
]";

/// Called once from `VmState::with_options`, immediately after
/// `runtime::globals::bootstrap_well_known` — parses
/// [`ALIEN_BOOTSTRAP_SRC`] via `frontend::parser::parse_file`, pulls out
/// its single `TopItem::ClassDef` node (mirroring the `let TopItem::
/// ClassDef(c) = items.into_iter().next().unwrap() else { panic!(...) }`
/// idiom already used by `frontend::codegen`'s and `runtime::ffi`'s own
/// test helpers — `first_method_of`), and passes it to
/// `frontend::classdef::install_class_def`.
///
/// Because "Alien" is ALREADY a declared global by the time this runs,
/// `install_class_def` takes its REOPEN branch (not its create branch) —
/// it simply compiles and installs each listed method onto the already-
/// correctly-shaped genesis `alien_klass` (instance methods) or its
/// metaclass (class-side methods, via each method's own `class_side`
/// flag) — `install_class_def`'s own existing reopen logic already handles
/// that instance-vs-class-side split; this function does not reimplement
/// it.
///
/// `.expect(...)`s the result: a bootstrap-source compile failure is a
/// build-breaking bug in this file's own embedded string (a typo in a
/// primitive id, a syntax error), never a runtime data problem a Smalltalk
/// program could trigger — the same "loud failure" posture
/// `runtime::ffi`'s own module doc draws between "bad Smalltalk data" and
/// "missing feature/environment support", just applied to a VM-internal
/// bootstrap step instead of a running program.
pub(crate) fn bootstrap_alien_methods(vm: &mut VmState) {
    let items = crate::frontend::parser::parse_file(ALIEN_BOOTSTRAP_SRC)
        .expect("runtime::alien::bootstrap_alien_methods: ALIEN_BOOTSTRAP_SRC failed to parse");
    let crate::frontend::ast::TopItem::ClassDef(mut node) = items.into_iter().next().unwrap()
    else {
        panic!(
            "runtime::alien::bootstrap_alien_methods: ALIEN_BOOTSTRAP_SRC must be a single class def"
        )
    };
    crate::frontend::classdef::install_class_def(vm, &mut node).expect(
        "runtime::alien::bootstrap_alien_methods: ALIEN_BOOTSTRAP_SRC failed to install \
         (Alien's methods) — a build-breaking bug in this embedded source, not a runtime \
         data problem",
    );
}

#[cfg(test)]
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

    /// Exactly `runtime::ffi`'s own test-module pattern (`test_klass`) — a
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

    /// Exactly `runtime::ffi`'s own test-module pattern (`first_method_of`)
    /// — parse a one-method class body and pull out its `MethodNode`.
    fn first_method_of(src: &str) -> crate::frontend::ast::MethodNode {
        let items = parse_file(src).expect("parse");
        let TopItem::ClassDef(c) = items.into_iter().next().unwrap() else {
            panic!("expected a class def")
        };
        c.methods.into_iter().next().expect("expected a method")
    }

    /// Compile `src`'s first (and only) method on a fresh test klass named
    /// `klass_name`, then actually RUN it through the real interpreter
    /// send/primitive path (`interpreter::run_method`), receiver `nil`, no
    /// args — exactly `runtime::ffi`'s own `compile_and_run` helper.
    ///
    /// A REAL method body (not a bare top-level do-it) is used for every
    /// multi-step test below, deliberately: a top-level do-it is exactly
    /// ONE `Expr` with `temps: vec![]` (`frontend::codegen::compile_doit`),
    /// so `| a | a := ... . ^...`-style temp-variable snippets are NOT
    /// valid there — only inside a real method body, which DOES support
    /// `| temps |`. This is still real end-to-end execution through the
    /// full interpreter/primitive dispatch path, just driven via a tiny
    /// helper class instead of the top-level do-it grammar.
    fn compile_and_run(vm: &mut VmState, klass_name: &str, src: &str) -> Oop {
        let klass = test_klass(vm, klass_name);
        let mut method = first_method_of(src);
        let m = compile_method(vm, klass, false, &mut method).expect("compile");
        let nil = vm.universe.nil_obj;
        run_method(vm, m, nil, &[])
    }

    /// For a class def with MORE than one method (`first_method_of`/
    /// `compile_and_run` above only ever handle one): install the WHOLE
    /// class def for real via `frontend::classdef::install_class_def` (the
    /// same real production path `bootstrap_alien_methods` itself uses —
    /// class-side AND instance-side methods both installed, exactly as a
    /// real `.mst` file's class def would be), then send `run` (unary, no
    /// args) to a fresh instance and return the result. Used by the mmap
    /// capstone test, whose helper class needs a class-side `mmapAddr:...`
    /// FFI method AND an instance-side `run` method that calls it.
    fn install_and_run(vm: &mut VmState, src: &str) -> Oop {
        let items = parse_file(src).expect("parse");
        let TopItem::ClassDef(mut c) = items.into_iter().next().unwrap() else {
            panic!("expected a class def")
        };
        crate::frontend::classdef::install_class_def(vm, &mut c).expect("install");
        let name_sym = vm.universe.intern(c.name.as_bytes());
        let assoc = crate::runtime::globals::global_lookup(vm, name_sym)
            .expect("class def must be globally declared after install_class_def");
        let klass = crate::oops::wrappers::KlassOop::try_from(
            crate::oops::wrappers::MemOop::try_from(assoc)
                .expect("global association is a mem oop")
                .body_oop(1),
        )
        .expect("global value must be the installed klass");
        let run_sym = vm.universe.intern(b"run");
        let m = crate::runtime::lookup::lookup(vm, klass, run_sym)
            .expect("installed class must understand #run");
        let receiver =
            alloc::alloc_words(vm, crate::oops::layout::HEADER_WORDS, klass.oop(), false);
        run_method(vm, m, receiver.oop(), &[])
    }

    /// Empirical check for `Universe::alien_klass`'s own `nis_words`
    /// derivation (`oops::layout::ALIEN_NAMED_WORDS`), per this brief's own
    /// explicit instruction: allocate an Alien with N tail bytes, confirm
    /// `indexable_len()` reads back exactly N, and confirm writing tail
    /// byte 0 does NOT clobber the named external-address field (i.e. the
    /// address field and the tail don't alias). If either check fails, the
    /// `nis_words` arithmetic is wrong and needs fixing before anything
    /// else in this module makes sense — this test is deliberately the
    /// FIRST real thing this module's whole design stands on.
    #[test]
    fn alien_nis_words_derivation_is_correct() {
        let mut vm = test_vm();
        let klass = vm.universe.alien_klass;
        let obj = alloc::alloc_indexable_bytes(&mut vm, klass, 8);
        let a = AlienOop::try_from(obj.oop()).expect("fresh Alien is a valid AlienOop");

        // Size slot reads back exactly N.
        assert_eq!(a.as_mem().indexable_len(), 8);

        // Freshly allocated (zero-filled): direct by default.
        assert_eq!(a.external_addr(), 0);

        // Set the external-address field to a recognizable nonzero value,
        // then write a DIFFERENT recognizable value into tail byte 0 —
        // if the two fields aliased, one write would clobber the other.
        a.set_external_addr(0xDEAD_BEEF_u64);
        a.as_mem().set_tail_byte_at(0, 0x42);
        assert_eq!(
            a.external_addr(),
            0xDEAD_BEEF_u64,
            "tail write clobbered the address field"
        );
        assert_eq!(
            a.as_mem().tail_byte_at(0),
            0x42,
            "address-field write clobbered the tail"
        );
    }

    /// Direct round-trip: `Alien new: 8`, `doubleAt: 1 put: 3.5`, `doubleAt:
    /// 1` returns 3.5 — exercises the one allocating accessor
    /// (`prim_alien_double_at`) end to end through full dispatch.
    #[test]
    fn direct_double_round_trip() {
        let mut vm = test_vm();
        let result = compile_and_run(
            &mut vm,
            "AlienDoubleRoundTrip",
            "Object subclass: AlienDoubleRoundTrip [ \
                run [ \
                    | a | \
                    a := Alien new: 8. \
                    a doubleAt: 1 put: 3.5. \
                    ^a doubleAt: 1 \
                ] \
            ]",
        );
        let got = crate::oops::wrappers::DoubleOop::try_from(result)
            .expect("expected a Double result")
            .value();
        assert_eq!(got, 3.5);
    }

    /// Direct byte-level, unsigned: `byteAt:put: 200` then `byteAt:` must
    /// return 200 exactly — not -56 or any other sign-confused value,
    /// proving `byteAt:`'s read path never accidentally sign-extends.
    #[test]
    fn direct_byte_at_is_unsigned() {
        let mut vm = test_vm();
        let result = compile_and_run(
            &mut vm,
            "AlienByteAtUnsigned",
            "Object subclass: AlienByteAtUnsigned [ \
                run [ \
                    | a | \
                    a := Alien new: 4. \
                    a byteAt: 1 put: 200. \
                    ^a byteAt: 1 \
                ] \
            ]",
        );
        let got = SmallInt::try_from(result)
            .expect("expected a SmallInt result")
            .value();
        assert_eq!(got, 200);
    }

    /// Direct `signedLongAt:` round trip with a NEGATIVE value, proving
    /// sign-correctness end to end (not just "doesn't crash").
    #[test]
    fn direct_signed_long_at_round_trip_negative() {
        let mut vm = test_vm();
        let result = compile_and_run(
            &mut vm,
            "AlienSignedLongNegative",
            "Object subclass: AlienSignedLongNegative [ \
                run [ \
                    | a | \
                    a := Alien new: 8. \
                    a signedLongAt: 1 put: -12345. \
                    ^a signedLongAt: 1 \
                ] \
            ]",
        );
        let got = SmallInt::try_from(result)
            .expect("expected a SmallInt result")
            .value();
        assert_eq!(got, -12345);
    }

    /// Bounds-check test: an out-of-range index must fail cleanly. Calls
    /// the primitive function directly (not through full dispatch) — same
    /// reasoning as `runtime::ffi`'s own
    /// `ffi_args_arity_mismatch_falls_through_not_panics` test for why a
    /// direct call is the right shape here: this is testing the
    /// primitive's OWN validation, not the surrounding dispatch machinery.
    #[test]
    fn byte_at_out_of_range_index_fails() {
        let mut vm = test_vm();
        let klass = vm.universe.alien_klass;
        let obj = alloc::alloc_indexable_bytes(&mut vm, klass, 4);
        let a = AlienOop::try_from(obj.oop()).expect("fresh Alien is a valid AlienOop");
        let args = [a.oop(), SmallInt::new(99).oop()];
        let outcome = prim_alien_byte_at(&mut vm, &args);
        assert_eq!(outcome, PrimResult::Fail);
    }

    /// A negative or zero index must also fail (1-based indexing, same
    /// convention as `prim_byte_at`).
    #[test]
    fn byte_at_zero_index_fails() {
        let mut vm = test_vm();
        let klass = vm.universe.alien_klass;
        let obj = alloc::alloc_indexable_bytes(&mut vm, klass, 4);
        let a = AlienOop::try_from(obj.oop()).expect("fresh Alien is a valid AlienOop");
        let args = [a.oop(), SmallInt::new(0).oop()];
        let outcome = prim_alien_byte_at(&mut vm, &args);
        assert_eq!(outcome, PrimResult::Fail);
    }

    /// A multi-byte access starting in-bounds but running off the end must
    /// also fail — `signedLongAt:` needs 8 bytes, so index 2 on a 8-byte
    /// Alien (bytes 2..=9) must fail even though index 2 alone is in range.
    #[test]
    fn signed_long_at_overrun_fails() {
        let mut vm = test_vm();
        let klass = vm.universe.alien_klass;
        let obj = alloc::alloc_indexable_bytes(&mut vm, klass, 8);
        let a = AlienOop::try_from(obj.oop()).expect("fresh Alien is a valid AlienOop");
        let args = [a.oop(), SmallInt::new(2).oop()];
        let outcome = prim_alien_signed_long_at(&mut vm, &args);
        assert_eq!(outcome, PrimResult::Fail);
    }

    /// Negative test used only to confirm `first_method_of` stays exercised
    /// (parity with `runtime::ffi`'s own helper set) — a real one-argument
    /// Alien-adjacent method compiles and its declared arity matches. Not
    /// itself a very interesting assertion; the real value of this helper
    /// is available to future tests in this module that need it.
    #[test]
    fn first_method_of_smoke() {
        let mut method = first_method_of(
            "Object subclass: AlienSmokeTest [ \
                probe: n [ <primitive: 112> ] \
            ]",
        );
        let mut vm = test_vm();
        let object_klass = vm.universe.object_klass;
        let klass = vm.universe.new_klass(
            object_klass,
            "AlienSmokeTest",
            crate::oops::Format::Slots,
            false,
            crate::oops::layout::HEADER_WORDS,
        );
        let m = compile_method(&mut vm, klass, false, &mut method).expect("compile");
        assert_eq!(m.argc(), 1);
        let _ = run_method; // silence unused-import warning if this is ever the only user
    }

    /// **The capstone integration test, spanning S20 steps 1 through 5 in
    /// one real end-to-end call.** Uses a REAL `mmap` call through S20 step
    /// 4's already-working FFI primitive (matching docs/FFI.md §6.3's own
    /// real signature) with `MAP_PRIVATE|MAP_ANON` flags and fd `-1`, to
    /// get a fresh anonymous page without needing a real file descriptor.
    /// The real macOS/arm64 flag values (verified against
    /// `/usr/include/sys/mman.h` and a standalone C program while writing
    /// this test, not guessed): `PROT_READ|PROT_WRITE == 0x03`,
    /// `MAP_PRIVATE|MAP_ANON == 0x1002`.
    ///
    /// This is the single highest-value test in this whole step: it proves
    /// the address returned by a REAL native `mmap` syscall, wrapped as an
    /// indirect Alien via `forAddress:size:`, genuinely reads and writes
    /// REAL mapped memory through this module's raw-pointer path — not a
    /// mock, not a direct-mode-only round trip.
    #[cfg(target_os = "macos")] // WINVM: calls real mmap via FFI; needs the Phase-3 x64 stubs
    #[test]
    fn mmap_capstone_indirect_alien_reads_writes_real_mapped_memory() {
        let mut vm = test_vm();
        // Real macOS/arm64 flag values (verified against `/usr/include/
        // sys/mman.h` and a standalone C program while writing this test,
        // not guessed): `PROT_READ|PROT_WRITE == 0x03`,
        // `MAP_PRIVATE|MAP_ANON == 0x1002`. `fd = -1`, `offset = 0` — the
        // standard "give me a fresh anonymous page" incantation, needing
        // no real file descriptor.
        let result = install_and_run(
            &mut vm,
            "Object subclass: FFIMmapCapstone [ \
                FFIMmapCapstone class >> mmapAddr: a1 length: a2 prot: a3 flags: a4 fd: a5 offset: a6 [ \
                    <primitive: FFI function: #mmap ret: #g args: #(g g g g g g)> \
                ] \
                run [ \
                    | addr a | \
                    addr := FFIMmapCapstone \
                        mmapAddr: 0 length: 4096 prot: 3 flags: 4098 fd: -1 offset: 0. \
                    a := Alien forAddress: addr size: 4096. \
                    a byteAt: 1 put: 42. \
                    ^a byteAt: 1 \
                ] \
            ]",
        );
        let got = SmallInt::try_from(result)
            .expect("expected a SmallInt result")
            .value();
        assert_eq!(
            got, 42,
            "indirect Alien must read back what it wrote to real mapped memory"
        );
    }
}
