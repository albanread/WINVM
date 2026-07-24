//! Bit/offset constants for the tagged-oop and mark-word representation,
//! plus (S10) the analogous fixed-offset contract for [`VmRegBlock`] — not
//! an oop layout, but the same "single source of truth for an offset
//! compiled/hand-written code bakes in" role, so it lives here rather than
//! starting a second such file.
//!
//! Single source of truth (CONVENTIONS §2): nothing outside `oops::layout`
//! may name a tag value, shift, or mask directly. SPEC §2.1 (tags), §2.2
//! (mark word).

// --- oop tag scheme (SPEC §2.1) --------------------------------------------

pub const TAG_BITS: u32 = 2;
pub const TAG_MASK: u64 = 0b11;
pub const INT_TAG: u64 = 0b00; // smi
pub const MEM_TAG: u64 = 0b01; // heap oop; address = word - MEM_TAG
pub const RESERVED_TAG: u64 = 0b10; // future immediate Character; illegal in v1
pub const MARK_TAG: u64 = 0b11; // only as header word 0

/// S11 D6.3: the two RESERVED_TAG sentinel values a compiled call can return
/// in x0 that are NOT real oops (both have `& TAG_MASK == RESERVED_TAG`, so
/// `Oop::from_raw` rejects them and no real oop can collide). `BAILOUT`
/// (`interpreter::compiled_call`, S10 D1) means "fall back to interpreted";
/// `NLR_SENTINEL` (S11 step 9) means "a non-local return is escaping through
/// this compiled frame — target/value are parked in `VmState::nlr_state`".
/// Distinct full values, so a single exact `cmp x0, #6` never confuses them.
pub const NLR_SENTINEL: u64 = 0b0110; // = 6

/// A third RESERVED_TAG sentinel, same family as `NLR_SENTINEL` (an extra
/// bit beyond the bare tag, so `BAILOUT_SENTINEL`'s plain `0b10` — a
/// different, already-established meaning: give up on compiled execution
/// entirely, go interpret — can never collide with this): a compiled
/// primitive-call shim's own signal for `PrimResult::Fail` (`compiler::
/// driver`'s shimmable-primitive eligibility, `codecache::stubs::
/// rt_call_primitive`). Distinct from BAILOUT specifically because Fail
/// does NOT mean "leave compiled code" — it means "continue in this SAME
/// compiled frame, running the method's own bytecode fallback body",
/// exactly mirroring the interpreter's own `PrimitiveOutcome::Fallthrough`.
pub const PRIM_FAIL_SENTINEL: u64 = 0b1010; // = 10

/// SPEC §7.6 (amended per S7 review, `sprint_s07_detail.md` "SPEC-QUESTION"):
/// Rust can't rewrite live stale locals/handles the way a moving GC does in
/// a language with a rewritable-roots runtime, so the enforceable
/// equivalent is poisoning the memory a stale reference would land on. This
/// pattern is MARK_TAG-shaped (low 2 bits `0b11`) so it parses as a header
/// tag, but bit 2 (`MARK_SENTINEL_MASK`) is 0 — a mark word with sentinel=0
/// is otherwise impossible (`Mark::pristine()` always sets it), so
/// `Mark::from_word`'s existing `debug_assert!` on that bit trips
/// immediately and deterministically on any stale dereference, with no new
/// checking mechanism needed.
pub const POISON: u64 = 0xF0F0_F0F0_F0F0_F0F3;

pub const SMI_SHIFT: u32 = 2;
pub const SMI_BITS: u32 = 62;
pub const SMI_MAX: i64 = (1i64 << 61) - 1; //  2_305_843_009_213_693_951
pub const SMI_MIN: i64 = -(1i64 << 61); // -2_305_843_009_213_693_952

// --- heap layout (SPEC §2.2) ------------------------------------------------

pub const WORD_SIZE: usize = 8;
pub const ALLOC_ALIGN: usize = 8; // SPEC §2.1: objects 8-byte aligned (NOT 16)
pub const MARK_OFFSET: usize = 0; // header word 0
pub const KLASS_OFFSET: usize = 8; // header word 1
pub const BODY_OFFSET: usize = 16; // first body word
pub const HEADER_WORDS: usize = 2;

// --- mark word fields (SPEC §2.2) — bit positions, low to high -------------

pub const MARK_SENTINEL_SHIFT: u32 = 2; // always 1 in a real mark
pub const MARK_NEAR_DEATH_SHIFT: u32 = 3;
pub const MARK_TAGGED_CONTENTS_SHIFT: u32 = 4;
pub const MARK_AGE_SHIFT: u32 = 5; // bits 11:5
pub const MARK_AGE_BITS: u32 = 7;
pub const MARK_AGE_MAX: u8 = 127;
pub const MARK_HASH_SHIFT: u32 = 12; // bits 43:12
pub const MARK_HASH_BITS: u32 = 32;
// bits 63:44 reserved, must stay zero in v1.

pub const MARK_SENTINEL_MASK: u64 = 1 << MARK_SENTINEL_SHIFT;
pub const MARK_NEAR_DEATH_MASK: u64 = 1 << MARK_NEAR_DEATH_SHIFT;
pub const MARK_TAGGED_CONTENTS_MASK: u64 = 1 << MARK_TAGGED_CONTENTS_SHIFT;
pub const MARK_AGE_MASK: u64 = ((1u64 << MARK_AGE_BITS) - 1) << MARK_AGE_SHIFT;
pub const MARK_HASH_MASK: u64 = ((1u64 << MARK_HASH_BITS) - 1) << MARK_HASH_SHIFT;

/// Bit 44: the full-GC mark bit (SPEC §2.2, amendment A: added at S8 review).
/// Always clear outside a collection.
pub const MARK_GC_MARK_SHIFT: u32 = 44;
pub const MARK_GC_MARK_MASK: u64 = 1 << MARK_GC_MARK_SHIFT;

/// The mark word of a freshly allocated object: tag=MARK_TAG, sentinel=1,
/// every other field zero. `0b111` = 7.
pub const MARK_PRISTINE: u64 = MARK_TAG | MARK_SENTINEL_MASK;

// --- klass body layout (SPEC §2.4), in declaration order -------------------
// `*_INDEX` constants are body-word indexes (word 0 = byte offset
// BODY_OFFSET); byte offset = BODY_OFFSET + 8*index.

pub const KLASS_FORMAT_INDEX: usize = 0; // smi: format enum + flags
pub const KLASS_NON_INDEXABLE_SIZE_INDEX: usize = 1; // smi: words incl. header
pub const KLASS_SUPERCLASS_INDEX: usize = 2; // klassOop | nil
pub const KLASS_METHODS_INDEX: usize = 3; // MethodDictionary | nil (nil until S3)
pub const KLASS_NAME_INDEX: usize = 4; // Symbol | nil (during genesis)
pub const KLASS_INST_VAR_NAMES_INDEX: usize = 5; // Array | nil
pub const KLASS_CLASS_VARS_INDEX: usize = 6; // Array | nil
pub const KLASS_MIXIN_INDEX: usize = 7; // always nil in v1 (SPEC §2.4 Δ)
pub const KLASS_BODY_WORDS: usize = 8;
pub const KLASS_SIZE_WORDS: usize = HEADER_WORDS + KLASS_BODY_WORDS; // 10

// Format smi encoding: bits 7:0 = Format discriminant,
// bit 8 = has_untagged_contents (SPEC §2.4 "format enum + flags").
pub const FORMAT_KIND_MASK: i64 = 0xFF;
pub const FORMAT_UNTAGGED_BIT: i64 = 1 << 8;

// --- CompiledMethod body layout (SPEC §4.4) --------------------------------

pub const METHOD_SELECTOR_INDEX: usize = 0; // Symbol
pub const METHOD_HOLDER_INDEX: usize = 1; // klassOop | nil (nil until S3 install)
pub const METHOD_FLAGS_INDEX: usize = 2; // smi, packing below
pub const METHOD_PRIMITIVE_INDEX: usize = 3; // smi, 0 = none
/// S20 FFI (docs/FFI.md §5/§6.3): `primitive()`'s sentinel for `<primitive:
/// FFI …>` — a method whose REAL dispatch info is a small descriptor Array
/// at `literals()[0]` (`frontend::codegen::build_ffi_descriptor`), not a
/// numbered primitive at all. Negative, so it can never collide with a real
/// numbered primitive (`1..=65535`, `parser::parse_method_pragma`'s own
/// validated range) or the "no primitive" default (`0`).
pub const PRIM_ID_FFI: i64 = -1;
pub const METHOD_COUNTERS_INDEX: usize = 4; // smi (invocation:16 | reserved)
pub const METHOD_LITERALS_INDEX: usize = 5; // Array
pub const METHOD_ICS_INDEX: usize = 6; // Array (stride 4, SPEC §4.3)
pub const METHOD_NAMED_WORDS: usize = 7; // => klass nis = 9
pub const METHOD_SIZE_INDEX: usize = 7; // smi: bytecode byte count
/// Absolute byte offset of the first bytecode byte.
pub const METHOD_BYTECODE_BYTE_OFFSET: usize = BODY_OFFSET + 8 * (METHOD_SIZE_INDEX + 1); // 80

// --- Alien body layout (S20 step 5, docs/FFI.md §4 "Representation") ------
//
// `Alien` reuses `Format::IndexableBytes` (the same format `ByteArrayOop`
// already uses) rather than inventing a new `Format` variant, following the
// SAME "named fields precede an indexable byte tail" idiom `CompiledMethod`
// (`Format::Method`, above) already established for a mixed shape. This is
// a deliberate, considered adaptation of real Strongtalk's `Alien.dlt`
// design (`~/claudeprojects/strongtalk-repo/StrongtalkSource/Alien.dlt`,
// §2): Strongtalk encodes direct-vs-indirect via a SIGN-FLIPPED size field
// shared with the object's own ordinary size slot (negative size = indirect,
// with a companion `addressField` holding the real pointer). Porting that
// literally would mean teaching `MemOop::raw_size_slot`/`indexable_len`
// (`oops/heap.rs`) to interpret magnitude via `.abs()` — a small change,
// but one EVERY `ByteArrayOop`/`String`/`Symbol` in the entire VM shares,
// for a feature only `Alien` needs. Instead, Alien gets exactly ONE named
// field (below) ahead of its own implicit size slot, exactly like
// `CompiledMethod`'s 7 named fields sit ahead of `METHOD_SIZE_INDEX` — zero
// changes needed anywhere in `oops/heap.rs`'s already-klass-nis-aware
// `raw_size_slot`/`indexable_len`/`tail_start_word`/`tail_byte_at` family.
/// The one named field: a SmallInt raw external address. `0` means
/// "direct" — the object's own indexable byte tail (found via the same
/// `tail_start_word`/`tail_byte_at` machinery `ByteArrayOop`/`Symbol` already
/// use) IS the real storage. A nonzero value means "indirect" — that value
/// IS the real external (non-heap, non-GC-owned) address to read/write
/// through instead; the object's own tail bytes still physically exist
/// (allocated to the same declared size, `Universe::alien_klass`'s creation
/// call + `runtime::alien::prim_alien_for_address_size`) but are never
/// touched — a deliberate, documented waste of real heap space equal to the
/// wrapped region's size, traded for identical bounds-checking/allocation/
/// GC-scanning code paths between direct and indirect Aliens with zero
/// special-casing (`runtime::alien`'s own module doc elaborates).
pub const ALIEN_EXTERNAL_ADDR_INDEX: usize = 0;
/// Body-word-1 named field: an INDIRECT Alien's declared byte size (a
/// SmallInt, like the address field). This is the "second named field"
/// the original design deferred (`runtime::alien`'s module doc): with it,
/// `forAddress:size:` allocates a ZERO-byte real tail — wrapping an 8 MB
/// mmap region no longer wastes 8 MB of real VM heap — and bounds checks
/// read this field instead of `indexable_len()` when the Alien is
/// indirect. Always 0 for a direct Alien (whose `indexable_len()` remains
/// the single source of truth). Added 2026-07 for the Accelerate design
/// (`docs/accelerate_design.md`), whose NativeFloatArray wraps
/// multi-megabyte regions.
pub const ALIEN_INDIRECT_SIZE_INDEX: usize = 1;
/// `Universe::alien_klass`'s `nis_words`: two named fields
/// (`ALIEN_EXTERNAL_ADDR_INDEX`, `ALIEN_INDIRECT_SIZE_INDEX`) ahead of
/// Alien's own implicit size slot, exactly the same pattern
/// `METHOD_NAMED_WORDS` (7) establishes for `Format::Method`'s klass
/// (`nis = HEADER_WORDS + METHOD_NAMED_WORDS`). Empirically verified
/// (`runtime::alien`'s own `alien_nis_words_derivation_is_correct` test):
/// allocating an Alien with N tail bytes yields `indexable_len() == N`
/// exactly, and writing tail byte 0 never clobbers either named field —
/// the size slot lands at body-word 2, the tail starts at body-word 3, so
/// the fields and the tail never alias.
pub const ALIEN_NAMED_WORDS: usize = 2;

// flags packing — SPEC §4.4 (S4-extended) "argc:4 | ntemps:8 | has_ctx:1 |
// is_block:1 | prim_fails:1 | captures_ctx:1 | nctx:8"
pub const METHOD_FLAGS_ARGC_SHIFT: u32 = 0;
pub const METHOD_FLAGS_ARGC_BITS: u32 = 4;
pub const METHOD_FLAGS_NTEMPS_SHIFT: u32 = 4;
pub const METHOD_FLAGS_NTEMPS_BITS: u32 = 8;
pub const METHOD_FLAGS_HAS_CTX_SHIFT: u32 = 12;
pub const METHOD_FLAGS_IS_BLOCK_SHIFT: u32 = 13;
pub const METHOD_FLAGS_PRIM_FAILS_SHIFT: u32 = 14;
pub const METHOD_FLAGS_CAPTURES_CTX_SHIFT: u32 = 15;
pub const METHOD_FLAGS_NCTX_SHIFT: u32 = 16;
pub const METHOD_FLAGS_NCTX_BITS: u32 = 8;

pub const METHOD_FLAGS_ARGC_MASK: i64 =
    ((1i64 << METHOD_FLAGS_ARGC_BITS) - 1) << METHOD_FLAGS_ARGC_SHIFT;
pub const METHOD_FLAGS_NTEMPS_MASK: i64 =
    ((1i64 << METHOD_FLAGS_NTEMPS_BITS) - 1) << METHOD_FLAGS_NTEMPS_SHIFT;
pub const METHOD_FLAGS_HAS_CTX_MASK: i64 = 1 << METHOD_FLAGS_HAS_CTX_SHIFT;
pub const METHOD_FLAGS_IS_BLOCK_MASK: i64 = 1 << METHOD_FLAGS_IS_BLOCK_SHIFT;
pub const METHOD_FLAGS_PRIM_FAILS_MASK: i64 = 1 << METHOD_FLAGS_PRIM_FAILS_SHIFT;
pub const METHOD_FLAGS_CAPTURES_CTX_MASK: i64 = 1 << METHOD_FLAGS_CAPTURES_CTX_SHIFT;
pub const METHOD_FLAGS_NCTX_MASK: i64 =
    ((1i64 << METHOD_FLAGS_NCTX_BITS) - 1) << METHOD_FLAGS_NCTX_SHIFT;

pub const METHOD_ARGC_MAX: usize = (1 << METHOD_FLAGS_ARGC_BITS) - 1; // 15
pub const METHOD_NTEMPS_MAX: usize = (1 << METHOD_FLAGS_NTEMPS_BITS) - 1; // 255
pub const METHOD_NCTX_MAX: usize = (1 << METHOD_FLAGS_NCTX_BITS) - 1; // 255

// --- frame layout (SPEC §5.1, S2/S3, extended S4) ---------------------------
// S4 inserts two slots (serial, marker); FRAME_TEMPS_BASE moved from 5 to 7 —
// never hard-code either value outside this constant.

pub const FRAME_METHOD: usize = 0;
pub const FRAME_SAVED_FP: usize = 1;
pub const FRAME_SAVED_BCI: usize = 2;
pub const FRAME_CONTEXT: usize = 3;
pub const FRAME_RECEIVER: usize = 4;
/// Smi: bits 31:0 = the per-push frame serial (dead-home detection, SPEC
/// §5.4); bit 32 = marker kind (0 = Ensure, 1 = IfCurtailed), meaningful
/// only while `FRAME_MARKER` holds an armed handler closure. Read-modify-
/// write only — `set_marker` must preserve bits 31:0.
pub const FRAME_SERIAL: usize = 5;
/// `nil` (no marker) | `ClosureOop` (an armed `ensure:`/`ifCurtailed:`
/// handler) | `ArrayOop` (an `UnwindToken`, while a suspended NLR's handler
/// runs) — SPEC §5.4.
pub const FRAME_MARKER: usize = 6;
pub const FRAME_TEMPS_BASE: usize = 7;
pub const ENTRY_FRAME_SENTINEL: i64 = -1;

pub const FRAME_SERIAL_BITS: u32 = 32;
pub const FRAME_SERIAL_MASK: i64 = (1i64 << FRAME_SERIAL_BITS) - 1;
pub const FRAME_MARKER_KIND_SHIFT: u32 = FRAME_SERIAL_BITS; // bit 32
pub const FRAME_MARKER_KIND_MASK: i64 = 1i64 << FRAME_MARKER_KIND_SHIFT;

// --- home reference packing (SPEC §5.4, S4) ---------------------------------
// `proc:8 | serial:32 | fp:22` packed into one smi's 62-bit two's-complement
// value space (SMI_BITS=62, i.e. SMI_MIN=-(2^61)..SMI_MAX=2^61-1 — the SAME
// range a 62-bit signed integer covers). Field values with the top bit set
// (e.g. proc=255) legitimately produce a *negative* smi; pack/unpack must
// round-trip via sign-extension, never via a plain unsigned OR-then-hope-it-
// fits (that silently exceeds SMI_MAX and panics `SmallInt::new` for proc
// >= ~73).
pub const HOME_REF_FP_BITS: u32 = 22;
pub const HOME_REF_SERIAL_BITS: u32 = 32;
pub const HOME_REF_PROC_BITS: u32 = 8;
pub const HOME_REF_FP_SHIFT: u32 = 0;
pub const HOME_REF_SERIAL_SHIFT: u32 = HOME_REF_FP_SHIFT + HOME_REF_FP_BITS; // 22
pub const HOME_REF_PROC_SHIFT: u32 = HOME_REF_SERIAL_SHIFT + HOME_REF_SERIAL_BITS; // 54
pub const HOME_REF_FP_MAX: usize = (1 << HOME_REF_FP_BITS) - 1;
pub const HOME_REF_FP_MASK: u64 = ((1u64 << HOME_REF_FP_BITS) - 1) << HOME_REF_FP_SHIFT;
pub const HOME_REF_SERIAL_MASK: u64 = ((1u64 << HOME_REF_SERIAL_BITS) - 1) << HOME_REF_SERIAL_SHIFT;
pub const HOME_REF_PROC_MASK: u64 = ((1u64 << HOME_REF_PROC_BITS) - 1) << HOME_REF_PROC_SHIFT;
/// All 62 packed bits — used to strip the sign-extension bits back off on
/// unpack.
pub const HOME_REF_ALL_MASK: u64 = HOME_REF_FP_MASK | HOME_REF_SERIAL_MASK | HOME_REF_PROC_MASK;

// --- resume sentinels (SPEC §5.4, S4) ----------------------------------------
// A `saved_bci` above any real bci — `debug_assert!(bytecode_len <
// BCI_SENTINEL_BASE)` at method creation guarantees these can never collide
// with a real bytecode index.
pub const BCI_SENTINEL_BASE: usize = 0x7FFF_0000;
pub const BCI_RESUME_ENSURE_RET: usize = 0x7FFF_0001;
pub const BCI_RESUME_UNWIND: usize = 0x7FFF_0002;
/// Not itself in SPEC's pseudocode, but required to detect "a
/// `#cannotReturn:` handler returned normally" (SPEC §5.4 Algorithm 12: "if
/// a user handler *returns*, the failed NLR cannot be resumed... print
/// trace, terminate process") — without this sentinel, that return would
/// silently resume wherever the escaping block happened to be.
pub const BCI_RESUME_CANNOT_RETURN: usize = 0x7FFF_0003;

// --- IC side table (SPEC §4.3) ----------------------------------------------
// Stride 4 per site: [sel: Symbol][meta: smi argc:8|epoch:24][guard][target].

pub const IC_STRIDE: usize = 4;
pub const IC_SEL_OFFSET: usize = 0;
pub const IC_META_OFFSET: usize = 1;
pub const IC_GUARD_OFFSET: usize = 2;
pub const IC_TARGET_OFFSET: usize = 3;

pub const IC_META_ARGC_SHIFT: u32 = 0;
pub const IC_META_ARGC_BITS: u32 = 8;
pub const IC_META_EPOCH_SHIFT: u32 = 8;
pub const IC_META_EPOCH_BITS: u32 = 24;
pub const IC_META_ARGC_MASK: i64 = ((1i64 << IC_META_ARGC_BITS) - 1) << IC_META_ARGC_SHIFT;
pub const IC_META_EPOCH_MASK: i64 = ((1i64 << IC_META_EPOCH_BITS) - 1) << IC_META_EPOCH_SHIFT;
pub const IC_EPOCH_MAX: u32 = (1 << IC_META_EPOCH_BITS) - 1;

/// Guard-slot sentinels distinguishing poly/mega from a mono `KlassOop`
/// guard (a real klass oop is always mem-tagged, tag 01; these are smis,
/// tag 00, so the two families can never collide).
pub const IC_GUARD_POLY: i64 = 1;
pub const IC_GUARD_MEGA: i64 = 2;

pub const IC_POLY_MAX_PAIRS: usize = 4;
/// `[k0,m0, k1,m1, k2,m2, k3,m3, c0,c1,c2,c3]` — the stride-2 pairs region
/// (SPEC §4.3 lattice payload, layout unchanged) followed by a COUNT TAIL
/// (dart124_compiler_lessons items 2+3 substrate): one smi per arm, `nil`
/// reads as 0, bumped ONLY by the interpreter's row-7 poly hit — the
/// unoptimized tier is the profiler, compiled code never counts.
pub const IC_POLY_ARRAY_LEN: usize = IC_POLY_MAX_PAIRS * 3;

// --- method counters (SPEC §4.4) --------------------------------------------

pub const COUNTERS_INVOCATION_MASK: i64 = 0xFFFF;
pub const COUNTERS_INVOCATION_MAX: i64 = 0xFFFF;
/// S15 (SPEC §5.5 / the §4.4 SPEC-QUESTION pin): the LOOP counter lives in
/// bits 16-31 of `MethodOop.counters` — invocation:16 | loop:16 | rest.
pub const COUNTERS_LOOP_SHIFT: u32 = 16;
pub const COUNTERS_LOOP_MASK: i64 = 0xFFFF << COUNTERS_LOOP_SHIFT;
/// Backedges through ONE method before the interpreter offers the running
/// frame for OSR (SPEC §5.5). Well under the field's 65,535 ceiling.
pub const LOOP_COUNTER_LIMIT: i64 = 10_000;
/// S10 D1: set by `compiler::driver::compile_method` when `eligible`
/// rejects a method, so the interpreter's counter-overflow trigger doesn't
/// re-attempt (and re-reject) the same never-compilable method every 10k
/// sends. Bit 33 (S24 L2 step 1): the ORIGINAL bit 16 predated S15, whose
/// loop counter then claimed bits 16-31 — `bump_loop_counter`/
/// `reset_loop_counter` do masked RMWs over that whole field, so ONE
/// backedge through a loopy compile-disabled method silently clobbered the
/// disable bit and the method re-attempted (and re-rejected) compilation
/// every trigger, forever — exactly the unguided re-compilation D1 exists
/// to prevent. Bit 32 is `COUNTERS_HAS_BP_BIT`; 33 is the next bit clear of
/// the invocation (0-15) and loop (16-31) fields, preserved by both fields'
/// masked read-modify-write updates.
pub const COUNTERS_COMPILE_DISABLED_BIT: i64 = 1 << 33;

/// DBG1 (docs/DEBUGGER.md §2): set on any method carrying at least one
/// breakpoint, so the dispatch hook's hash lookup runs only inside methods
/// that actually have breakpoints. Bit 32 — clear of the invocation (0-15)
/// and loop (16-31) fields, and preserved by their masked
/// read-modify-write updates.
pub const COUNTERS_HAS_BP_BIT: i64 = 1 << 32;

// --- MethodDictionary (SPEC §2.4, S3) ---------------------------------------
// One named field (tally) then an indexable [k0,v0,k1,v1,...] tail.

pub const METHODDICT_TALLY_INDEX: usize = 0;

// --- BlockClosure (SPEC §2.3, S4) --------------------------------------------
// 2 named fields (method, home) then an indexable [ncopied: smi][copied...]
// tail (the generic Format::Closure size-slot shape from oops::heap).

pub const CLOSURE_METHOD_INDEX: usize = 0;
pub const CLOSURE_HOME_INDEX: usize = 1;
pub const CLOSURE_NAMED_WORDS: usize = 2;

// --- Context (SPEC §2.3, S4) -------------------------------------------------
// 1 named field (home_hint) then an indexable [size: smi][slot...] tail (the
// generic Format::Context size-slot shape from oops::heap).

pub const CONTEXT_HOME_HINT_INDEX: usize = 0;
pub const CONTEXT_NAMED_WORDS: usize = 1;

// --- VmState register block (S10 D6) ----------------------------------------
// A #[repr(C)] prefix struct embedded as VmState's own first field (see
// `runtime::vm_state::VmRegBlock`), so x28 (== &VmState, established by the
// call stub — SPEC §8.1/S10 D5.1) reaches these fields via fixed-offset
// loads/stores with no Rust-level indirection. `old_start`/
// `card_base_biased` (S11 barrier) and `last_compiled_{fp,pc,kind}`
// (S11/S12 runtime entries) are value fields; `poll_flag` is live from S10
// (the `Poll` Ir op reads it, D5.3).
//
// S12 step 7: slot +0 is `eden_top_addr` — the ADDRESS of the one live
// `universe.eden.top` word (stable: the Eden struct is boxed), which the
// inline-alloc fast path DEREFERENCES to read/bump the same bump pointer
// every Rust-side allocator and both collectors use. It replaced S11's
// `eden_top` VALUE copy + the publish/adopt sync protocol, which was only
// sound while the (deleted) D8 bridge froze eden for the whole compiled
// window. `eden_end` (+8) stays a value copy of a genesis-fixed, immutable
// bound — set once, can never go stale.

pub const VMREG_EDEN_TOP_ADDR_OFFSET: usize = 0;
pub const VMREG_EDEN_END_OFFSET: usize = 8;
pub const VMREG_OLD_START_OFFSET: usize = 16;
pub const VMREG_CARD_BASE_BIASED_OFFSET: usize = 24;
/// `POLL_OFF` (D5.3): `ldr w16, [x28, #VMREG_POLL_FLAG_OFFSET]; cbnz w16, poll_slow`.
pub const VMREG_POLL_FLAG_OFFSET: usize = 32;
pub const VMREG_LAST_COMPILED_FP_OFFSET: usize = 40;
pub const VMREG_LAST_COMPILED_PC_OFFSET: usize = 48;
/// S12 D3: which of the 6 anchor-setting runtime stubs (resolve, c2i_shared,
/// mega_shared, dnu, must_be_boolean, alloc_slow) wrote the anchor —
/// `last_compiled_pc` alone can't answer this (it's x30 at anchor-write
/// time, which every one of these stubs sets to the same kind of value:
/// the address INSIDE ITS OWN CALLER, never an address inside the stub's
/// own code — none of them are ever reached via a `bl`/`blr` whose return
/// address could instead be read back out of a deeper frame). Written by
/// each of the 6 stubs' own preamble (never inside the shared
/// `emit_stub_prologue` itself — an earlier draft tried that and it
/// clobbers x16/x17, which `mega_shared`/`c2i_shared` carry a live
/// selector/method oop through at exactly that point), cleared by
/// `emit_stub_epilogue` alongside `last_compiled_fp` (defense in depth:
/// a stale kind paired with a stale-but-nonzero fp should never be able to
/// happen, but if it somehow did, clearing both make it fail loudly rather
/// than plausibly).
pub const VMREG_LAST_COMPILED_KIND_OFFSET: usize = 56;
pub const VMREG_BLOCK_SIZE: usize = 64;

/// S11 D4.1/D5/P8: the RootSpill area every runtime-reaching stub uses to
/// park x0..x7 (receiver + up to 7 args) as GC roots while control is in
/// Rust — "RootSpill offsets are ABI" (P8): the pre-S12 bridge (D8) and
/// S12's own root enumeration both read these slots by fixed offset from
/// the stub's own `x29`, so a change here needs a matching `FrameView`
/// decoder update, not just a recompile. 8 slots, 8 bytes each — already
/// 16-byte aligned (AArch64's hard SP-alignment invariant), so no padding
/// is needed for this area specifically (per-stub frames that add their
/// own fields on top, e.g. the c2i adapter's extra method-oop slot, pad
/// themselves).
///
/// WHY 8, not the original 6 (S15, BUG D root cause 4's deeper half —
/// tests/repros/README.md): a compiled send site marshals receiver+args
/// into x0..x7 (`emit_call_send`), i.e. up to EIGHT live values, and
/// `driver::eligibility_detail` caps compiled SEND sites at that same
/// receiver+7 so the register convention is never exceeded. The original 6
/// covered a compiled METHOD's own argc cap (5) but not its SEND sites'
/// — Richards' 7-arg task initializer sent through a c2i adapter read args
/// 6 and 7 from PAST the 6-slot area: the stub frame's own saved fp/lr,
/// whose (stack-address, code-address) values parse as smis, flowed
/// silently into the created object's last two ivars, and surfaced only
/// thousands of sends later as a `doesNotUnderstand:` on a "SmallInteger".
/// x6/x7 are also caller-saved in the C ABI, so EVERY ≥6-arg send passing
/// through any Rust-reaching stub (resolve/mega/dnu, not just c2i) needed
/// them spilled and reloaded regardless of the c2i read.
pub const ROOTSPILL_SLOTS: usize = 8;
pub const ROOTSPILL_BYTES: usize = ROOTSPILL_SLOTS * 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_relations() {
        assert_eq!(MARK_HASH_SHIFT, MARK_AGE_SHIFT + MARK_AGE_BITS);
        assert_eq!(MARK_AGE_SHIFT, MARK_TAGGED_CONTENTS_SHIFT + 1);
        assert_eq!(SMI_MAX, -(SMI_MIN + 1));
        assert_eq!(SMI_BITS, 64 - TAG_BITS);
    }

    #[test]
    fn layout_values_pinned() {
        assert_eq!(INT_TAG, 0);
        assert_eq!(MEM_TAG, 1);
        assert_eq!(RESERVED_TAG, 2);
        assert_eq!(MARK_TAG, 3);
        assert_eq!(SMI_MAX, 2_305_843_009_213_693_951);
        assert_eq!(ALLOC_ALIGN, 8);
        assert_eq!(BODY_OFFSET, 16);
    }

    #[test]
    fn mask_disjointness() {
        assert_eq!(MARK_AGE_MASK & MARK_HASH_MASK, 0);
        let flag_bits = MARK_SENTINEL_MASK | MARK_NEAR_DEATH_MASK | MARK_TAGGED_CONTENTS_MASK;
        assert_eq!(MARK_AGE_MASK & flag_bits, 0);
        assert_eq!(MARK_HASH_MASK & flag_bits, 0);
        assert_eq!(MARK_AGE_MASK & TAG_MASK, 0);
        assert_eq!(MARK_HASH_MASK & TAG_MASK, 0);
        assert_eq!(flag_bits & TAG_MASK, 0);
        // fields partition bits 43:0 with no overlap and leave 63:44 reserved
        // (the gc_mark bit at 44 is the one exception, checked separately).
        let used = MARK_AGE_MASK | MARK_HASH_MASK | TAG_MASK | flag_bits;
        assert_eq!(used >> 44, 0);
        assert_eq!(MARK_GC_MARK_MASK & used, 0);
    }
}
