//! `Universe`: the heap, well-known oops, and `genesis()` — the metaclass
//! knot (SPEC §3.1–§3.2 step 1). After genesis,
//! `nil.klass().name() == #UndefinedObject` holds in a running process.
//!
//! Genesis is built almost entirely from **local variables**, not by
//! mutating a partially-constructed `Universe`: the circularities (every
//! object needs a klass, but the first klasses don't exist yet; fields want
//! `nil`, but `nil` needs `UndefinedObject`; klass names are Symbols, but
//! Symbols need `symbol_klass` and the table) are resolved by allocating
//! with placeholder fields and patching, exactly as
//! `sprint_s01_detail.md` §Design prescribes — but the `Universe` struct
//! literal itself is only ever constructed once, at the very end (step 12),
//! so there is never a "partially valid" `Universe` for other code to
//! observe. `PLACEHOLDER = SmallInt::new(0).oop()` throughout: a valid oop
//! that can never be mistaken for a heap reference (tag 00, not 01), so any
//! accidental read through it fails fast in a typed wrapper's `try_from`
//! rather than performing a wild read.

use crate::oops::klass::Format;
use crate::oops::layout::{ALIEN_NAMED_WORDS, HEADER_WORDS, KLASS_SIZE_WORDS};
use crate::oops::mark::Mark;
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{KlassOop, MemOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmOptions;

use super::alloc;
use super::cards::CardTable;
use super::layout::HeapLayout;
use super::offsets::OffsetTable;
use super::reservation::Reservation;
use super::spaces::{Eden, OldGen, SurvivorSpace};
use super::stats::GcStats;
use super::symbols::SymbolTable;

pub struct Universe {
    #[allow(dead_code)] // kept alive for its Drop impl; no other field reads it in S1
    reservation: Reservation,
    /// Boxed (S12 step 7) so `&eden.top` is a STABLE heap address for the
    /// whole process lifetime, immune to `Universe`/`VmState` moves —
    /// `VmRegBlock::eden_top_addr` (set once, `VmState::with_options`)
    /// hands that address to compiled code, whose inline-alloc fast path
    /// reads and writes the bump pointer THROUGH it. One canonical
    /// location shared by both worlds, replacing S11's value-copy +
    /// publish/adopt sync protocol (see `eden_top_addr`'s own doc for the
    /// full why). Everything Rust-side is unaffected: `Box<Eden>` derefs
    /// transparently at every existing `vm.universe.eden.*` use.
    pub eden: Box<Eden>,
    /// Fixed geometry `eden`/`from`/`to`/`old` bounds were carved out of
    /// (SPEC §7.1) — `is_old`/`is_new` are defined here.
    pub layout: HeapLayout,
    /// Scavenge copy destination this cycle (S7-7 swaps `from`/`to` after
    /// each scavenge); both start empty and are not yet allocated into.
    pub from: SurvivorSpace,
    pub to: SurvivorSpace,
    /// Committed prefix of the reserved old-gen range; not yet allocated
    /// into by the interpreter (S7-7's promotion path is the first real
    /// caller of `OldGen::allocate`).
    pub old: OldGen,
    /// The write barrier's remembered set (SPEC §7.4) — `memory::store`
    /// dirties a card whenever an old-gen slot is written a new-gen oop.
    pub cards: CardTable,
    /// Object-start table over old gen, maintained in lockstep with
    /// `old`'s allocations — lets a dirty-card scan find an object's
    /// header without walking from `old.bounds.start`.
    pub offsets: OffsetTable,
    /// GC counters (SPEC §12.4-ish observability) — read by `GcStallError`
    /// and, later, `Smalltalk`-level introspection.
    pub gc_stats: GcStats,
    /// Per-age surviving-byte histogram for the scavenge in progress
    /// (SPEC §7.3 step 4); cleared at the start of every scavenge.
    pub age_table: super::scavenge::AgeTable,
    /// Ages `>= tenuring_threshold` promote directly to old gen instead of
    /// copying to the `to` survivor space. Starts at 127 (never tenure by
    /// age) until a scavenge recomputes it; forced to 127 during S7-7's
    /// own bring-up (promotion enabled once copy+scan are solid).
    pub tenuring_threshold: u8,
    /// `false` only during genesis itself — the half-built metaobject knot
    /// must never be scanned (SPEC §7.3 A1). Set `true` once genesis
    /// returns; `scavenge` asserts it.
    pub gc_enabled: bool,
    /// Set by `scavenge_oop` when a copy/promotion allocation fails
    /// mid-scavenge; `scavenge()` checks it between phases and turns it
    /// into a returned `Err` rather than silently leaving a root pointed
    /// at an un-copied from-space object. `None` the rest of the time.
    pub pending_stall: Option<super::stall::GcStallError>,
    /// S8 step 7: whether `alloc::ensure_promotion_guarantee` believes old
    /// gen has enough committed room to promote the entire young generation
    /// in the worst case, as of the last time it was checked (right before
    /// the scavenge about to run). `true` by construction whenever the
    /// guarantee holds; `false` only when the reservation is genuinely
    /// exhausted even after a full GC and growth — a real, bounded-memory
    /// terminal condition (`eden_exhaustion_aborts` deliberately drives a
    /// 16 MiB heap into exactly this), not a cascade bug. `record_stall`
    /// reads this to tell the two apart: a `ScavengePromote` stall is only
    /// a guarantee VIOLATION (debug-assert-worthy) when this was `true`.
    pub promotion_guarantee_met: bool,
    /// S8 step 8: allocations through `alloc_words` since the last
    /// `MACVM_GC_STRESS=full[:N]` trigger — compared against
    /// `VmOptions::gc_stress_full_period` to decide when to run one. Plain
    /// count, not bytes (matches the doc's "every N allocations" wording).
    pub full_stress_alloc_count: u64,

    pub nil_obj: Oop,
    pub true_obj: Oop,
    pub false_obj: Oop,

    pub metaclass_klass: KlassOop,
    pub class_klass: KlassOop,
    pub object_klass: KlassOop,
    pub undefined_object_klass: KlassOop,
    pub boolean_klass: KlassOop,
    pub true_klass: KlassOop,
    pub false_klass: KlassOop,
    pub smi_klass: KlassOop,
    pub character_klass: KlassOop,
    pub double_klass: KlassOop,
    /// SIMD (`docs/SIMD.md`): a 2-lane `f64` vector — a fixed 16-byte RAW body
    /// (`Format::Double`, GC-skips the body; size is klass-driven, so a 2-word
    /// body Just Works). Boxed at rest; the JIT vector fast-path loads it into
    /// a `q`-register with `ldr q,[obj+16]`. The first of the `FloatNxM`
    /// value-class family.
    pub float64x2_klass: KlassOop,
    pub float32x4_klass: KlassOop,
    pub int32x4_klass: KlassOop,
    /// SIMD level 2 (`docs/SIMD.md` Part E): `FloatArray` — a contiguous
    /// `Format::IndexableBytes` buffer of N f64 lanes (`n*8` bytes, GC-skipped
    /// body) that the NEON bulk kernels (`+@`/`sum`/`dot:`) stream over.
    pub float_array_klass: KlassOop,
    pub string_klass: KlassOop,
    pub symbol_klass: KlassOop,
    pub array_klass: KlassOop,
    pub bytearray_klass: KlassOop,
    /// S20 step 5 (docs/FFI.md §4): the FFI byte-level memory-access type.
    /// `Format::IndexableBytes` like `bytearray_klass` (same body shape,
    /// same `oops::heap` machinery), but superclass `object_klass` — NOT
    /// `bytearray_klass`/`arrayed_collection_klass` — because Alien
    /// deliberately does not inherit ByteArray's own `size`/`at:`/
    /// `byteAt:`-family primitives, which assume zero named fields
    /// (`oops::layout::ALIEN_NAMED_WORDS` shifts Alien's tail by one word)
    /// and would silently misread/miswrite against Alien's shape, plus
    /// have no notion of "indirect" (external-pointer-backed) mode at all.
    /// Alien gets its own, complete accessor primitive set instead
    /// (`runtime::alien`). Methods are installed post-genesis by
    /// `runtime::alien::bootstrap_alien_methods`, not here — genesis only
    /// fixes the klass's SHAPE (`nis_words` can't be expressed by the
    /// `subclass:` parser, only by this same `remaining_klass!` machinery).
    pub alien_klass: KlassOop,
    /// `ObjcRef` (Cocoa bridge C0, docs/cocoa_bridge_design.md §2): a
    /// bytes-format wrapper whose 8-byte tail holds a raw Objective-C `id`.
    /// ZERO named words on purpose — the id must live in the byte tail (the
    /// one storage the collector never walks), NOT in Alien's named smi
    /// slot: arm64 tagged-pointer ids set bit 63 (out of smi range) and a
    /// raw word in a scanned slot would corrupt the heap.
    pub objcref_klass: KlassOop,
    pub association_klass: KlassOop,
    pub methoddict_klass: KlassOop,
    pub method_klass: KlassOop,
    pub closure_klass: KlassOop,
    pub context_klass: KlassOop,
    pub process_klass: KlassOop,
    /// 2 named slots: `selector`, `arguments` — built by the DNU path
    /// (`interpreter::send`, S3) and handed to `#doesNotUnderstand:`.
    pub message_klass: KlassOop,
    /// `IndexableBytes`, base-256 little-endian magnitude (S5's BigInt
    /// literals, SPEC-QUESTION in `sprint_s05_detail.md` §Literal frame).
    /// No arithmetic primitives target these yet — S6 adds Large fallback
    /// code; this sprint only needs the klasses to exist so literal building
    /// has somewhere to allocate.
    pub large_pos_int_klass: KlassOop,
    pub large_neg_int_klass: KlassOop,

    // --- S6 genesis skeleton (SPEC §3.1 amendment, A5's pinned tree) -------
    /// Superclass of `class_klass`/`metaclass_klass` (Smalltalk-80's
    /// Behavior/ClassDescription layer, collapsed to one klass in v1).
    pub behavior_klass: KlassOop,
    pub magnitude_klass: KlassOop,
    pub number_klass: KlassOop,
    pub integer_klass: KlassOop,
    /// Abstract; `large_pos_int_klass`/`large_neg_int_klass` are its only
    /// concrete subclasses, same `IndexableBytes` shape.
    pub large_integer_klass: KlassOop,
    pub collection_klass: KlassOop,
    pub sequenceable_collection_klass: KlassOop,
    pub arrayed_collection_klass: KlassOop,
    /// The klass of the `smalltalk` global-namespace object itself (format
    /// `IndexableOops`, same `[tally][assoc-or-nil…]` layout
    /// `runtime::globals` already used under `array_klass` — S6 only
    /// reclassifies it, no representation change, SPEC-QUESTION A5).
    pub system_dictionary_klass: KlassOop,

    /// The global namespace (SPEC §3.1: "a Dictionary of Association", A5:
    /// "a SystemDictionary, VM-managed layout"). `nil` until the first
    /// `runtime::globals::global_declare` call; from then on a
    /// `system_dictionary_klass` `ArrayOop`-shaped object laid out
    /// `[tally][assoc-or-nil…]` (dense, append-only).
    pub smalltalk: Oop,
    /// Set once `frontend::world::load_world` finishes (SPEC §3.2 step 2).
    pub world_loaded: bool,

    pub symbols: SymbolTable,
    hash_counter: u32,

    // --- well-known selectors (S3, SPEC §6.3/§4.2) --------------------------
    pub sel_does_not_understand: SymbolOop,
    pub sel_must_be_boolean: SymbolOop,
    pub sel_cannot_return: SymbolOop,
}

// Default eden sizing moved to `layout::default_eden_for` (scales with the
// reservation; 32 MiB cap). `options.eden_kb` (`MACVM_EDEN` env override,
// S7-10) still overrides it — the S7 GC-stress test's actual consumer: a
// small eden makes a scavenge reachable without first allocating megabytes
// of filler.
/// Initial symbol table capacity *(tunable)* — SPEC §3.1.
const SYMBOL_TABLE_CAPACITY: usize = 1024;

impl Universe {
    pub fn genesis(options: &VmOptions) -> Universe {
        // --- step 1: heap up (sprint_s07_detail.md A1) -----------------------
        let heap_bytes = options.heap_mib << 20;
        let eden_size = options
            .eden_kb
            .map(|kb| kb << 10)
            .unwrap_or_else(|| super::layout::default_eden_for(heap_bytes));
        let reservation = Reservation::reserve(heap_bytes);
        // Small-reservation clamp: a heap smaller than the default eden +
        // both survivors (possible since the 16 MiB default eden, 40fc343)
        // must still boot — shrink eden so old keeps at least an eden's
        // worth of room, floored at 1 MiB. Explicit MACVM_EDEN values are
        // clamped the same way (a tiny heap is the stronger constraint).
        let survivors = 2 * crate::memory::layout::SURVIVOR_SIZE;
        let max_eden = heap_bytes
            .saturating_sub(survivors)
            .saturating_div(2)
            .max(1 << 20);
        let eden_size = eden_size.min(max_eden);
        let layout = HeapLayout::new(reservation.base(), heap_bytes, eden_size);
        reservation.commit(layout.eden.start - reservation.base(), layout.eden.len());
        reservation.commit(layout.from.start - reservation.base(), layout.from.len());
        reservation.commit(layout.to.start - reservation.base(), layout.to.len());
        let old_committed_len = super::layout::OLD_INITIAL_SEGMENT.min(layout.old.len());
        reservation.commit(layout.old.start - reservation.base(), old_committed_len);

        let mut eden = Eden::new(layout.eden.start, layout.eden.len());
        let from = SurvivorSpace::new(layout.from);
        let to = SurvivorSpace::new(layout.to);
        let old = OldGen::new(layout.old, layout.old.start + old_committed_len);
        let cards = CardTable::new(layout.old);
        let offsets = OffsetTable::new(layout.old);
        let gc_stats = GcStats::new();
        let age_table = super::scavenge::AgeTable::new();
        let tenuring_threshold = 127; // never tenure by age until a scavenge recomputes it

        let placeholder = SmallInt::new(0).oop();

        // --- step 2: nil shell (Slots, 0 named fields) ----------------------
        let nil_shell =
            alloc::alloc_words_raw(&mut eden, placeholder, HEADER_WORDS, placeholder, true);
        // From here on, tagged allocations nil-fill their bodies with the
        // REAL nil oop (still klass-less until step 6's patch).
        let nil = nil_shell.oop();

        // --- step 3: metaclass knot -----------------------------------------
        let metaclass_klass = alloc::alloc_klass_raw(&mut eden, nil, placeholder);
        metaclass_klass.set_format(Format::Klass, false);
        metaclass_klass.set_non_indexable_size(KLASS_SIZE_WORDS);
        metaclass_klass.set_superclass(placeholder); // patched at step 5c

        let metaclass_meta = alloc::alloc_klass_raw(&mut eden, nil, metaclass_klass.oop());
        metaclass_meta.set_format(Format::Klass, false);
        metaclass_meta.set_non_indexable_size(KLASS_SIZE_WORDS);
        metaclass_meta.set_superclass(placeholder); // patched at step 5d

        // Patch: close the knot. Metaclass class class == Metaclass.
        metaclass_klass.set_klass(metaclass_meta);

        // --- step 4: Object and Class ----------------------------------------
        let object_meta = alloc::alloc_klass_raw(&mut eden, nil, metaclass_klass.oop());
        object_meta.set_format(Format::Klass, false);
        object_meta.set_non_indexable_size(KLASS_SIZE_WORDS);
        object_meta.set_superclass(placeholder); // patched at step 4d

        let object_klass = alloc::alloc_klass_raw(&mut eden, nil, object_meta.oop());
        object_klass.set_format(Format::Slots, false);
        object_klass.set_non_indexable_size(HEADER_WORDS);
        object_klass.set_superclass(nil);

        let class_meta = alloc::alloc_klass_raw(&mut eden, nil, metaclass_klass.oop());
        class_meta.set_format(Format::Klass, false);
        class_meta.set_non_indexable_size(KLASS_SIZE_WORDS);

        // Δ (pinned): v1 has no Behavior/ClassDescription layer —
        // `Class superclass == Object`.
        let class_klass = alloc::alloc_klass_raw(&mut eden, nil, class_meta.oop());
        class_klass.set_format(Format::Klass, false);
        class_klass.set_non_indexable_size(KLASS_SIZE_WORDS);
        class_klass.set_superclass(object_klass.oop());

        // "Class class" inherits from "Object class".
        class_meta.set_superclass(object_meta.oop());

        // Patch: Object class superclass == Class (the Smalltalk-80 rule).
        object_meta.set_superclass(class_klass.oop());

        // --- step 5: close Metaclass's superclasses ---------------------------
        // Δ (pinned): Smalltalk-80 has `Metaclass superclass ==
        // ClassDescription`; v1 pins `Class`.
        metaclass_klass.set_superclass(class_klass.oop());
        metaclass_meta.set_superclass(class_meta.oop());

        // --- step 6: UndefinedObject + nil patch -------------------------------
        let undefined_object_klass = new_klass_core(
            &mut eden,
            nil,
            metaclass_klass,
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS,
        );
        nil_shell.set_klass(undefined_object_klass);

        // --- step 7: String, Symbol klasses -------------------------------------
        let string_klass = new_klass_core(
            &mut eden,
            nil,
            metaclass_klass,
            object_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS,
        );
        let symbol_klass = new_klass_core(
            &mut eden,
            nil,
            metaclass_klass,
            string_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS,
        );

        // --- step 8: symbol table + name patch pass -------------------------------
        let mut symbols = SymbolTable::with_capacity(SYMBOL_TABLE_CAPACITY);

        let name_of = |eden: &mut Eden, symbols: &mut SymbolTable, s: &str| {
            intern_core(eden, nil, symbols, symbol_klass, s.as_bytes())
        };

        let set_name_and_meta_name =
            |eden: &mut Eden, symbols: &mut SymbolTable, k: KlassOop, plain: &str| {
                let plain_sym = name_of(eden, symbols, plain);
                k.set_name(plain_sym.oop());
                let meta_name = format!("{plain} class");
                let meta_sym = name_of(eden, symbols, &meta_name);
                k.klass().set_name(meta_sym.oop());
            };

        set_name_and_meta_name(&mut eden, &mut symbols, object_klass, "Object");
        set_name_and_meta_name(&mut eden, &mut symbols, class_klass, "Class");
        set_name_and_meta_name(&mut eden, &mut symbols, metaclass_klass, "Metaclass");
        set_name_and_meta_name(
            &mut eden,
            &mut symbols,
            undefined_object_klass,
            "UndefinedObject",
        );
        set_name_and_meta_name(&mut eden, &mut symbols, string_klass, "String");
        set_name_and_meta_name(&mut eden, &mut symbols, symbol_klass, "Symbol");

        // --- step 9: remaining klasses -------------------------------------------
        macro_rules! remaining_klass {
            ($name:literal, $superclass:expr, $format:expr, $untagged:expr, $nis:expr) => {{
                let k = new_klass_core(
                    &mut eden,
                    nil,
                    metaclass_klass,
                    $superclass,
                    $format,
                    $untagged,
                    $nis,
                );
                set_name_and_meta_name(&mut eden, &mut symbols, k, $name);
                k
            }};
        }

        // --- step 9a: S6 genesis skeleton scaffold (pure hierarchy, no
        // instances of their own — superclass targets for the well-known
        // oops below and for world/*.mst reopens) --------------------------
        let behavior_klass = remaining_klass!(
            "Behavior",
            object_klass.oop(),
            Format::Klass,
            false,
            KLASS_SIZE_WORDS
        );
        let magnitude_klass = remaining_klass!(
            "Magnitude",
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let number_klass = remaining_klass!(
            "Number",
            magnitude_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let integer_klass = remaining_klass!(
            "Integer",
            number_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let large_integer_klass = remaining_klass!(
            "LargeInteger",
            integer_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );
        let collection_klass = remaining_klass!(
            "Collection",
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let sequenceable_collection_klass = remaining_klass!(
            "SequenceableCollection",
            collection_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let arrayed_collection_klass = remaining_klass!(
            "ArrayedCollection",
            sequenceable_collection_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let system_dictionary_klass = remaining_klass!(
            "SystemDictionary",
            object_klass.oop(),
            Format::IndexableOops,
            false,
            HEADER_WORDS
        );

        // Reparent the well-known klasses genesis already built (S1-S5)
        // whose FINAL superclass wasn't known until the scaffold above
        // existed — never re-created (they're well-known oops baked into
        // Rust code), just their `superclass`/metaclass-`superclass`
        // fields patched in place.
        macro_rules! reparent {
            ($k:expr, $new_super:expr) => {{
                $k.set_superclass($new_super.oop());
                $k.klass().set_superclass($new_super.klass().oop());
            }};
        }
        reparent!(class_klass, behavior_klass);
        reparent!(metaclass_klass, behavior_klass);
        reparent!(string_klass, arrayed_collection_klass);

        let boolean_klass = remaining_klass!(
            "Boolean",
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let true_klass = remaining_klass!(
            "True",
            boolean_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let false_klass = remaining_klass!(
            "False",
            boolean_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let smi_klass = remaining_klass!(
            "SmallInteger",
            integer_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS
        );
        let character_klass = remaining_klass!(
            "Character",
            magnitude_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS + 1
        );
        let double_klass = remaining_klass!(
            "Double",
            number_klass.oop(),
            Format::Double,
            true,
            HEADER_WORDS + 1
        );
        // SIMD (docs/SIMD.md): Float64x2 — same raw-body Format::Double as a
        // scalar Double, but a 2-word (16-byte) body for two f64 lanes. The GC
        // skips a Format::Double body regardless of its klass-driven size, so
        // this is GC-safe by construction.
        let float64x2_klass = remaining_klass!(
            "Float64x2",
            object_klass.oop(),
            Format::Double,
            true,
            HEADER_WORDS + 2
        );
        // SIMD: Float32x4 — the 4-lane f32 companion. SAME 16-byte raw body as
        // Float64x2 (four f32 = two words), so an identical HEADER_WORDS + 2
        // klass shape; only the lane interpretation (in the primitives / the
        // `.4s` NEON arrangement) differs.
        let float32x4_klass = remaining_klass!(
            "Float32x4",
            object_klass.oop(),
            Format::Double,
            true,
            HEADER_WORDS + 2
        );
        // SIMD: Int32x4 — four 32-bit INTEGER lanes (the other half of "2/4/8
        // 32-bit lanes"). Same 16-byte raw body as Float32x4; the primitives
        // read it as i32 and the NEON fuse uses integer `add/sub/mul v.4s`.
        // Integer add is associative → its reductions are bit-exact.
        let int32x4_klass = remaining_klass!(
            "Int32x4",
            object_klass.oop(),
            Format::Double,
            true,
            HEADER_WORDS + 2
        );
        let array_klass = remaining_klass!(
            "Array",
            arrayed_collection_klass.oop(),
            Format::IndexableOops,
            false,
            HEADER_WORDS
        );
        let bytearray_klass = remaining_klass!(
            "ByteArray",
            arrayed_collection_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );
        // SIMD level 2 (docs/SIMD.md Part E): FloatArray — same IndexableBytes
        // shape as ByteArray (raw GC-skipped body, no named ivars), but its
        // primitives read/write the body as f64 lanes and the NEON bulk kernels
        // (+@/sum/dot:) stream over it. A genesis klass because IndexableBytes
        // can't be requested from a plain `.mst` subclass.
        let float_array_klass = remaining_klass!(
            "FloatArray",
            arrayed_collection_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );
        // S20 step 5 (docs/FFI.md §4): `Alien`, right after `ByteArray`
        // since they share `Format::IndexableBytes` — but superclass
        // `object_klass`, NOT `bytearray_klass`/`arrayed_collection_klass`
        // (see `Universe::alien_klass`'s own doc comment for why: Alien's
        // tail is shifted by `ALIEN_NAMED_WORDS` and ByteArray's own
        // `at:`/`byteAt:`-family primitives assume no shift at all).
        // `untagged = true`, matching `bytearray_klass`'s own 4th argument
        // (a raw byte tail, never oop-scanned). `nis_words = HEADER_WORDS +
        // ALIEN_NAMED_WORDS` places the two named fields
        // (`ALIEN_EXTERNAL_ADDR_INDEX` at body-word 0,
        // `ALIEN_INDIRECT_SIZE_INDEX` at body-word 1 — the zero-tail
        // indirect-Alien size field, 2026-07) ahead of Alien's own
        // implicit size slot at body-word 2, with the byte tail starting
        // at body-word 3 — the exact same "named fields, then size slot,
        // then tail" pattern `method_klass` below uses for
        // `Format::Method`'s 7 named fields, just with 2 instead of 7.
        // This derivation is
        // empirically checked by `runtime::alien`'s
        // `alien_nis_words_derivation_is_correct` test (allocate, confirm
        // `indexable_len()` and no aliasing between the address field and
        // tail byte 0) — see that test before trusting this arithmetic
        // further.
        let alien_klass = remaining_klass!(
            "Alien",
            object_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS + ALIEN_NAMED_WORDS
        );
        // ObjcRef: bytes format, untagged tail, NO named words (its own
        // implicit size slot at body-word 0, tail from body-word 1) — see
        // the `objcref_klass` field doc for why the tail, not a named slot.
        let objcref_klass = remaining_klass!(
            "ObjcRef",
            object_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );
        let association_klass = remaining_klass!(
            "Association",
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS + 2
        );
        // One named field (tally: smi) before the indexable [k,v,k,v,...]
        // tail (S3's MethodDictOop).
        let methoddict_klass = remaining_klass!(
            "MethodDictionary",
            object_klass.oop(),
            Format::IndexableOops,
            false,
            HEADER_WORDS + 1
        );
        let method_klass = remaining_klass!(
            "CompiledMethod",
            object_klass.oop(),
            Format::Method,
            true,
            HEADER_WORDS + 7
        );
        let closure_klass = remaining_klass!(
            "BlockClosure",
            object_klass.oop(),
            Format::Closure,
            false,
            HEADER_WORDS + 2
        );
        let context_klass = remaining_klass!(
            "Context",
            object_klass.oop(),
            Format::Context,
            false,
            HEADER_WORDS + 1
        );
        let process_klass = remaining_klass!(
            "Process",
            object_klass.oop(),
            Format::Process,
            true,
            HEADER_WORDS
        );
        let message_klass = remaining_klass!(
            "Message",
            object_klass.oop(),
            Format::Slots,
            false,
            HEADER_WORDS + 2
        );
        let large_pos_int_klass = remaining_klass!(
            "LargePositiveInteger",
            large_integer_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );
        let large_neg_int_klass = remaining_klass!(
            "LargeNegativeInteger",
            large_integer_klass.oop(),
            Format::IndexableBytes,
            true,
            HEADER_WORDS
        );

        // Association's named instance variables: #key #value.
        {
            let key_sym = name_of(&mut eden, &mut symbols, "key");
            let value_sym = name_of(&mut eden, &mut symbols, "value");
            let ivn = alloc::alloc_words_raw(
                &mut eden,
                nil,
                HEADER_WORDS + 1 + 2,
                array_klass.oop(),
                true,
            );
            ivn.set_raw_body_word(0, SmallInt::new(2).oop().raw());
            ivn.set_raw_body_word(1, key_sym.oop().raw());
            ivn.set_raw_body_word(2, value_sym.oop().raw());
            association_klass.set_inst_var_names(ivn.oop());
        }
        // Character's named instance variable: #value.
        {
            let value_sym = name_of(&mut eden, &mut symbols, "value");
            let ivn = alloc::alloc_words_raw(
                &mut eden,
                nil,
                HEADER_WORDS + 1 + 1,
                array_klass.oop(),
                true,
            );
            ivn.set_raw_body_word(0, SmallInt::new(1).oop().raw());
            ivn.set_raw_body_word(1, value_sym.oop().raw());
            character_klass.set_inst_var_names(ivn.oop());
        }
        // Message's named instance variables: #selector #arguments.
        {
            let selector_sym = name_of(&mut eden, &mut symbols, "selector");
            let arguments_sym = name_of(&mut eden, &mut symbols, "arguments");
            let ivn = alloc::alloc_words_raw(
                &mut eden,
                nil,
                HEADER_WORDS + 1 + 2,
                array_klass.oop(),
                true,
            );
            ivn.set_raw_body_word(0, SmallInt::new(2).oop().raw());
            ivn.set_raw_body_word(1, selector_sym.oop().raw());
            ivn.set_raw_body_word(2, arguments_sym.oop().raw());
            message_klass.set_inst_var_names(ivn.oop());
        }

        // --- step 9b: well-known selectors (S3, SPEC §6.3/§4.2) -------------------
        let sel_does_not_understand = name_of(&mut eden, &mut symbols, "doesNotUnderstand:");
        let sel_must_be_boolean = name_of(&mut eden, &mut symbols, "mustBeBoolean");
        let sel_cannot_return = name_of(&mut eden, &mut symbols, "cannotReturn:");

        // --- step 10: true/false --------------------------------------------------
        let true_obj = alloc::alloc_words_raw(&mut eden, nil, HEADER_WORDS, true_klass.oop(), true);
        true_obj.set_mark(Mark::pristine().with_tagged_contents(true));
        let false_obj =
            alloc::alloc_words_raw(&mut eden, nil, HEADER_WORDS, false_klass.oop(), true);
        false_obj.set_mark(Mark::pristine().with_tagged_contents(true));

        // --- step 11: smalltalk + hash counter --------------------------------------
        let smalltalk = nil;
        let hash_counter = 1u32;

        let universe = Universe {
            reservation,
            // Boxed HERE, at the end of genesis, so the whole bootstrap
            // above keeps borrowing the plain local `eden` — only the
            // final, settled struct needs the stable address (`pub eden`'s
            // own doc).
            eden: Box::new(eden),
            layout,
            from,
            to,
            old,
            cards,
            offsets,
            gc_stats,
            age_table,
            tenuring_threshold,
            gc_enabled: false, // flipped true just before genesis() returns (SPEC §7.3 A1)
            pending_stall: None,
            promotion_guarantee_met: true,
            full_stress_alloc_count: 0,
            nil_obj: nil,
            true_obj: true_obj.oop(),
            false_obj: false_obj.oop(),
            metaclass_klass,
            class_klass,
            object_klass,
            undefined_object_klass,
            boolean_klass,
            true_klass,
            false_klass,
            smi_klass,
            character_klass,
            double_klass,
            float64x2_klass,
            float32x4_klass,
            int32x4_klass,
            float_array_klass,
            string_klass,
            symbol_klass,
            array_klass,
            bytearray_klass,
            alien_klass,
            objcref_klass,
            association_klass,
            methoddict_klass,
            method_klass,
            closure_klass,
            context_klass,
            process_klass,
            message_klass,
            large_pos_int_klass,
            large_neg_int_klass,
            behavior_klass,
            magnitude_klass,
            number_klass,
            integer_klass,
            large_integer_klass,
            collection_klass,
            sequenceable_collection_klass,
            arrayed_collection_klass,
            system_dictionary_klass,
            smalltalk,
            world_loaded: false,
            symbols,
            hash_counter,
            sel_does_not_understand,
            sel_must_be_boolean,
            sel_cannot_return,
        };

        // --- step 12: verify -------------------------------------------------------
        super::verify::verify_heap(&universe).expect("genesis produced an invalid heap");

        let mut universe = universe;
        universe.gc_enabled = true; // the half-built metaobject knot is safe to scan now
        universe
    }

    /// Create a new klass with a fresh metaclass, wired into the given
    /// superclass's metaclass chain, and intern+patch its name. The helper
    /// genesis itself uses (`new_klass_core`) taking explicit pieces rather
    /// than `&mut Universe` — genesis has no live `Universe` to borrow yet.
    /// This is the post-genesis entry point S5's `subclass:` drives; `pub`
    /// (not `pub(crate)`) so S3's integration tests can build fresh test
    /// hierarchies too, without waiting on the real `subclass:` bytecode.
    pub fn new_klass(
        &mut self,
        superclass: KlassOop,
        name: &str,
        format: Format,
        untagged: bool,
        nis_words: usize,
    ) -> KlassOop {
        let metaclass_klass = self.metaclass_klass;
        let nil = self.nil_obj;
        let k = new_klass_core(
            &mut self.eden,
            nil,
            metaclass_klass,
            superclass.oop(),
            format,
            untagged,
            nis_words,
        );
        let plain_sym = self.intern(name.as_bytes());
        k.set_name(plain_sym.oop());
        let meta_name = format!("{name} class");
        let meta_sym = self.intern(meta_name.as_bytes());
        k.klass().set_name(meta_sym.oop());
        k
    }

    /// Commit one more old-gen segment (S8 step 2), returning bytes newly
    /// committed (0 iff the reservation is exhausted). Wraps `OldGen::grow`
    /// so the private `reservation` stays encapsulated; `self.old` and
    /// `self.reservation` are disjoint fields, so this borrows cleanly.
    pub fn grow_old(&mut self, segment: usize) -> usize {
        let grown = self.old.grow(&self.reservation, segment);
        if grown > 0 {
            self.gc_stats.old_grow_count += 1;
        }
        grown
    }

    /// Interning entry point (SPEC §3.1). Probes by content; on miss
    /// allocates a Symbol and inserts. Rehashes ×2 at 75% load.
    pub fn intern(&mut self, bytes: &[u8]) -> SymbolOop {
        let symbol_klass = self.symbol_klass;
        let nil = self.nil_obj;
        intern_core(&mut self.eden, nil, &mut self.symbols, symbol_klass, bytes)
    }

    /// SPEC §2.2's lazy identity hash. `0` always means "unassigned"; the
    /// counter never hands out `0` on wraparound. Smis hash from their
    /// value directly — they have no mark word.
    pub fn identity_hash(&mut self, o: Oop) -> u32 {
        if let Some(smi) = SmallInt::try_from(o) {
            let v = smi.value();
            return (v as u32) ^ ((v >> 32) as u32);
        }
        let m = MemOop::try_from(o).expect("identity_hash: not a smi or mem oop");
        let mark = m.mark();
        let existing = mark.hash();
        if existing != 0 {
            return existing;
        }
        let h = self.next_hash();
        m.set_mark(mark.with_hash(h));
        h
    }

    fn next_hash(&mut self) -> u32 {
        self.hash_counter = self.hash_counter.wrapping_add(1);
        if self.hash_counter == 0 {
            self.hash_counter = 1;
        }
        self.hash_counter
    }

    #[cfg(test)]
    pub(crate) fn set_hash_counter_for_test(&mut self, v: u32) {
        self.hash_counter = v;
    }
}

/// The genesis-and-`new_klass`-shared core: allocate a fresh metaclass
/// (klass-shaped, `klass = metaclass_klass`), then the class itself
/// (`klass = ` the new meta), and wire `meta.superclass` to the given
/// superclass's own metaclass. `superclass` must already be a real
/// (non-placeholder) klass oop — true for every call site from genesis
/// step 6 onward.
fn new_klass_core(
    eden: &mut Eden,
    nil_fill: Oop,
    metaclass_klass: KlassOop,
    superclass: Oop,
    format: Format,
    untagged: bool,
    nis_words: usize,
) -> KlassOop {
    let meta = alloc::alloc_klass_raw(eden, nil_fill, metaclass_klass.oop());
    meta.set_format(Format::Klass, false);
    meta.set_non_indexable_size(KLASS_SIZE_WORDS);

    let klass = alloc::alloc_klass_raw(eden, nil_fill, meta.oop());
    klass.set_format(format, untagged);
    klass.set_non_indexable_size(nis_words);
    klass.set_superclass(superclass);

    let super_klass = KlassOop::try_from(superclass).expect("new_klass: superclass is not a klass");
    meta.set_superclass(super_klass.klass().oop());

    klass
}

fn symbol_bytes_eq(sym: SymbolOop, bytes: &[u8]) -> bool {
    if sym.len() != bytes.len() {
        return false;
    }
    (0..bytes.len()).all(|i| sym.as_mem().tail_byte_at(i) == bytes[i])
}

fn intern_core(
    eden: &mut Eden,
    nil_fill: Oop,
    symbols: &mut SymbolTable,
    symbol_klass: KlassOop,
    bytes: &[u8],
) -> SymbolOop {
    let hash = SymbolTable::content_hash(bytes);
    if let Some(found) = probe(symbols, symbol_klass, hash, bytes) {
        return found;
    }

    let nis = symbol_klass.non_indexable_size();
    let sym =
        alloc::alloc_indexable_bytes_raw(eden, nil_fill, symbol_klass.oop(), nis, bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        sym.byte_at_put(i, b);
    }
    // SAFETY: freshly allocated with klass = symbol_klass.
    let sym = unsafe { SymbolOop::from_oop_unchecked(sym.oop()) };

    // Eager identity hash (S3's MethodDictionary/LookupCache probing must
    // never mutate a mark word mid-probe): derived from the content hash
    // already computed above, `| 1` guarantees nonzero (0 = unassigned,
    // SPEC §2.2) unconditionally regardless of the input.
    let mem = sym.as_mem();
    mem.set_mark(mem.mark().with_hash((hash as u32) | 1));

    insert(symbols, hash, sym.oop());

    if symbols.count.saturating_mul(4) >= symbols.buckets.len().saturating_mul(3) {
        rehash(symbols, symbol_klass);
    }

    sym
}

/// `None` = not found, but ALSO tells the caller nothing about which slot
/// is free (a fresh linear probe is cheap and avoids returning a second,
/// easy-to-misuse output).
fn probe(
    symbols: &SymbolTable,
    symbol_klass: KlassOop,
    hash: u64,
    bytes: &[u8],
) -> Option<SymbolOop> {
    let mask = symbols.buckets.len() - 1;
    let mut idx = (hash as usize) & mask;
    loop {
        match symbols.buckets[idx] {
            Some(existing) => {
                // SAFETY: every occupied slot holds a valid Symbol oop.
                let sym = unsafe { SymbolOop::from_oop_unchecked(existing) };
                debug_assert_eq!(sym.as_mem().klass().oop(), symbol_klass.oop());
                if symbol_bytes_eq(sym, bytes) {
                    return Some(sym);
                }
            }
            None => return None,
        }
        idx = (idx + 1) & mask;
    }
}

fn insert(symbols: &mut SymbolTable, hash: u64, sym: Oop) {
    let mask = symbols.buckets.len() - 1;
    let mut idx = (hash as usize) & mask;
    while symbols.buckets[idx].is_some() {
        idx = (idx + 1) & mask;
    }
    symbols.buckets[idx] = Some(sym);
    symbols.count += 1;
}

fn rehash(symbols: &mut SymbolTable, symbol_klass: KlassOop) {
    let new_cap = symbols.buckets.len() * 2;
    let old = std::mem::replace(symbols, SymbolTable::with_capacity(new_cap));
    for slot in old.buckets.into_iter().flatten() {
        // SAFETY: every occupied slot held a valid Symbol oop.
        let sym = unsafe { SymbolOop::from_oop_unchecked(slot) };
        let mut bytes = Vec::new();
        let len = sym.len();
        for i in 0..len {
            bytes.push(sym.as_mem().tail_byte_at(i));
        }
        let hash = SymbolTable::content_hash(&bytes);
        insert(symbols, hash, slot);
    }
    let _ = symbol_klass; // reserved: format re-checks could go here later
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boot() -> Universe {
        Universe::genesis(&VmOptions {
            heap_mib: 64,
            trace: Default::default(),
            gc_stress: false,
            gc_stress_full_period: None,
            eden_kb: None,
            jit: crate::runtime::JitMode::Off,
        })
    }

    #[test]
    fn genesis_knot() {
        let u = boot();
        assert_eq!(u.object_klass.klass().klass(), u.metaclass_klass);
        assert_eq!(u.metaclass_klass.klass().klass(), u.metaclass_klass);
        assert_eq!(u.class_klass.klass().klass(), u.metaclass_klass);
    }

    #[test]
    fn genesis_superclasses() {
        let u = boot();
        assert_eq!(u.object_klass.superclass(), u.nil_obj);
        assert_eq!(
            u.object_klass.klass().superclass().raw(),
            u.class_klass.oop().raw()
        );
        // S6: Class/Metaclass both sit under Behavior (Smalltalk-80's
        // Behavior/ClassDescription layer, collapsed to one klass in v1),
        // not directly under Object.
        assert_eq!(
            u.class_klass.superclass().raw(),
            u.behavior_klass.oop().raw()
        );
        assert_eq!(
            u.behavior_klass.superclass().raw(),
            u.object_klass.oop().raw()
        );
        assert_eq!(
            u.metaclass_klass.superclass().raw(),
            u.behavior_klass.oop().raw()
        );
        assert_eq!(u.true_klass.superclass().raw(), u.boolean_klass.oop().raw());
        assert_eq!(
            u.true_klass.klass().superclass().raw(),
            u.boolean_klass.klass().oop().raw()
        );
    }

    /// S6's pinned genesis skeleton (sprint_s06_detail.md §Design tree):
    /// every listed superclass link, exact.
    #[test]
    fn genesis_skeleton() {
        let u = boot();
        let links: &[(KlassOop, KlassOop)] = &[
            (u.magnitude_klass, u.object_klass),
            (u.character_klass, u.magnitude_klass),
            (u.number_klass, u.magnitude_klass),
            (u.double_klass, u.number_klass),
            (u.integer_klass, u.number_klass),
            (u.smi_klass, u.integer_klass),
            (u.large_integer_klass, u.integer_klass),
            (u.large_pos_int_klass, u.large_integer_klass),
            (u.large_neg_int_klass, u.large_integer_klass),
            (u.collection_klass, u.object_klass),
            (u.sequenceable_collection_klass, u.collection_klass),
            (u.arrayed_collection_klass, u.sequenceable_collection_klass),
            (u.array_klass, u.arrayed_collection_klass),
            (u.bytearray_klass, u.arrayed_collection_klass),
            // Alien: NOT under ArrayedCollection (see `Universe::alien_klass`'s
            // own doc comment) — direct `Object` subclass, S20 step 5.
            (u.alien_klass, u.object_klass),
            // ObjcRef: Cocoa bridge C0 — same direct-Object posture as Alien.
            (u.objcref_klass, u.object_klass),
            (u.string_klass, u.arrayed_collection_klass),
            (u.symbol_klass, u.string_klass),
            (u.behavior_klass, u.object_klass),
            (u.class_klass, u.behavior_klass),
            (u.metaclass_klass, u.behavior_klass),
            (u.association_klass, u.object_klass),
            (u.message_klass, u.object_klass),
            (u.closure_klass, u.object_klass),
            (u.method_klass, u.object_klass),
            (u.system_dictionary_klass, u.object_klass),
        ];
        for (k, expected_super) in links {
            assert_eq!(
                k.superclass().raw(),
                expected_super.oop().raw(),
                "{}'s superclass must be {}",
                crate::memory::print_oop(&u, k.name()),
                crate::memory::print_oop(&u, expected_super.name())
            );
        }
    }

    /// `Object class superclass == Class`, `Class superclass == Behavior` —
    /// the class-side lookup path S6's Behavior accessors depend on.
    #[test]
    fn genesis_metaclass_chain() {
        let u = boot();
        assert_eq!(
            u.object_klass.klass().superclass().raw(),
            u.class_klass.oop().raw()
        );
        assert_eq!(
            u.class_klass.superclass().raw(),
            u.behavior_klass.oop().raw()
        );
        assert_eq!(
            u.class_klass.klass().superclass().raw(),
            u.behavior_klass.klass().oop().raw()
        );
    }

    /// `tests_s06.md`'s `instvar_layouts` entry: Character/Message's
    /// genesis-declared named instvars match the doc exactly (Association's
    /// own case is covered separately by `association_ivar_names`). The
    /// other classes in that inventory row (OrderedCollection/Dictionary/
    /// Interval/WriteStream/TestResult/TestCase) are declared at the
    /// parser level, not genesis, so they have no Rust-side counterpart.
    #[test]
    fn instvar_layouts() {
        let mut u = boot();
        let char_ivn =
            crate::oops::wrappers::ArrayOop::try_from(u.character_klass.inst_var_names())
                .expect("Character inst_var_names is an Array");
        assert_eq!(char_ivn.len(), 1);
        assert_eq!(char_ivn.at(0), u.intern(b"value").oop());

        let msg_ivn = crate::oops::wrappers::ArrayOop::try_from(u.message_klass.inst_var_names())
            .expect("Message inst_var_names is an Array");
        assert_eq!(msg_ivn.len(), 2);
        assert_eq!(msg_ivn.at(0), u.intern(b"selector").oop());
        assert_eq!(msg_ivn.at(1), u.intern(b"arguments").oop());
    }

    #[test]
    fn genesis_nil_true_false() {
        let mut u = boot();
        let nil = MemOop::try_from(u.nil_obj).unwrap();
        assert_eq!(nil.klass(), u.undefined_object_klass);
        assert_eq!(
            u.undefined_object_klass.name(),
            u.intern(b"UndefinedObject").oop()
        );
        assert_eq!(u.true_klass.name(), u.intern(b"True").oop());
        assert_eq!(u.false_klass.name(), u.intern(b"False").oop());
        assert_ne!(u.true_obj.raw(), u.false_obj.raw());
    }

    #[test]
    fn genesis_no_placeholders() {
        let u = boot();
        let klasses = [
            u.metaclass_klass,
            u.class_klass,
            u.object_klass,
            u.undefined_object_klass,
            u.boolean_klass,
            u.true_klass,
            u.false_klass,
            u.smi_klass,
            u.character_klass,
            u.double_klass,
            u.string_klass,
            u.symbol_klass,
            u.array_klass,
            u.bytearray_klass,
            u.alien_klass,
            u.association_klass,
            u.methoddict_klass,
            u.method_klass,
            u.closure_klass,
            u.context_klass,
            u.process_klass,
            u.message_klass,
            u.large_pos_int_klass,
            u.large_neg_int_klass,
            u.behavior_klass,
            u.magnitude_klass,
            u.number_klass,
            u.integer_klass,
            u.large_integer_klass,
            u.collection_klass,
            u.sequenceable_collection_klass,
            u.arrayed_collection_klass,
            u.system_dictionary_klass,
        ];
        for k in klasses {
            assert!(k.oop().is_mem());
            let sc = k.superclass();
            assert!(sc.raw() == u.nil_obj.raw() || KlassOop::try_from(sc).is_some());
            assert!(SymbolOop::try_from(k.name()).is_some());
            assert_eq!(k.mixin(), u.nil_obj);
        }
    }

    #[test]
    fn genesis_formats_and_sizes() {
        let u = boot();
        assert_eq!(u.method_klass.non_indexable_size(), HEADER_WORDS + 7);
        assert_eq!(u.association_klass.non_indexable_size(), HEADER_WORDS + 2);
        assert_eq!(u.double_klass.non_indexable_size(), HEADER_WORDS + 1);
        assert_eq!(u.double_klass.format(), Format::Double);
        assert_eq!(u.array_klass.format(), Format::IndexableOops);
        assert_eq!(u.bytearray_klass.format(), Format::IndexableBytes);
        assert!(u.bytearray_klass.has_untagged_contents());
        assert!(!u.array_klass.has_untagged_contents());
        // S20 step 5 (docs/FFI.md §4): Alien is `Format::IndexableBytes`
        // like ByteArray, but shifted by `ALIEN_NAMED_WORDS` (1) for its own
        // external-address field — `runtime::alien`'s own test module
        // empirically checks the ACTUAL body-word arithmetic this implies
        // (`alien_nis_words_derivation_is_correct`); this assertion only
        // pins the klass-level shape inputs that derivation depends on.
        assert_eq!(
            u.alien_klass.non_indexable_size(),
            HEADER_WORDS + crate::oops::layout::ALIEN_NAMED_WORDS
        );
        assert_eq!(u.alien_klass.format(), Format::IndexableBytes);
        assert!(u.alien_klass.has_untagged_contents());
    }

    #[test]
    fn metaclass_names() {
        let mut u = boot();
        assert_eq!(
            u.object_klass.klass().name(),
            u.intern(b"Object class").oop()
        );
        assert_eq!(u.array_klass.klass().name(), u.intern(b"Array class").oop());
        assert_eq!(
            u.metaclass_klass.klass().name(),
            u.intern(b"Metaclass class").oop()
        );
    }

    #[test]
    fn metaclass_shape() {
        let u = boot();
        let klasses = [
            u.object_klass,
            u.class_klass,
            u.metaclass_klass,
            u.array_klass,
            u.process_klass,
        ];
        for k in klasses {
            let meta = k.klass();
            assert_eq!(meta.format(), Format::Klass);
            assert_eq!(meta.non_indexable_size(), KLASS_SIZE_WORDS);
            assert_eq!(meta.klass(), u.metaclass_klass);
        }
    }

    #[test]
    fn new_klass_regular_path() {
        let mut u = boot();
        let zork = u.new_klass(u.object_klass, "Zork", Format::Slots, false, HEADER_WORDS);
        assert_eq!(zork.klass().klass(), u.metaclass_klass);
        assert_eq!(
            zork.klass().superclass().raw(),
            u.object_klass.klass().oop().raw()
        );
        assert_eq!(zork.name(), u.intern(b"Zork").oop());
    }

    #[test]
    fn symbol_intern_identity() {
        let mut u = boot();
        let a = u.intern(b"foo");
        let b = u.intern(b"foo");
        assert_eq!(a.oop().raw(), b.oop().raw());
        let c = u.intern(b"bar");
        assert_ne!(a.oop().raw(), c.oop().raw());
        let empty1 = u.intern(b"");
        let empty2 = u.intern(b"");
        assert_eq!(empty1.oop().raw(), empty2.oop().raw());
    }

    #[test]
    fn symbol_intern_content() {
        let mut u = boot();
        let s = u.intern(b"at:put:");
        assert_eq!(s.len(), 7);
        assert_eq!(s.as_string(), "at:put:");
        assert_eq!(s.as_mem().klass(), u.symbol_klass);
    }

    #[test]
    fn symbol_rehash() {
        let mut u = boot();
        // Genesis itself already interns every klass/metaclass name (and
        // #key/#value) before this test runs — the table does not start
        // empty. Assert the RELATIVE growth across several rehash cycles,
        // not an absolute count.
        let before = u.symbols.len();
        let names: Vec<String> = (0..2000).map(|i| format!("sym{i}")).collect();
        let first: Vec<Oop> = names.iter().map(|n| u.intern(n.as_bytes()).oop()).collect();
        for (n, expected) in names.iter().zip(first.iter()) {
            assert_eq!(u.intern(n.as_bytes()).oop().raw(), expected.raw());
        }
        assert_eq!(u.symbols.len(), before + 2000);
    }

    fn alloc_plain_object(u: &mut Universe) -> MemOop {
        let object_klass = u.object_klass;
        let nil = u.nil_obj;
        alloc::alloc_slots_raw(&mut u.eden, nil, object_klass)
    }

    #[test]
    fn identity_hash_lazy() {
        let mut u = boot();
        let a = alloc_plain_object(&mut u);
        let b = alloc_plain_object(&mut u);
        assert_eq!(a.mark().hash(), 0);

        let h1 = u.identity_hash(a.oop());
        assert_ne!(h1, 0);
        let h1_again = u.identity_hash(a.oop());
        assert_eq!(h1, h1_again);

        let h2 = u.identity_hash(b.oop());
        assert_ne!(h1, h2);
    }

    #[test]
    fn identity_hash_smi() {
        let mut u = boot();
        let eden_top_before = u.eden.top;
        let five = SmallInt::new(5).oop();
        assert_eq!(u.identity_hash(five), u.identity_hash(five));
        assert_eq!(u.eden.top, eden_top_before, "smi hashing must not allocate");
    }

    #[test]
    fn identity_hash_counter_skips_zero() {
        let mut u = boot();
        u.set_hash_counter_for_test(u32::MAX);
        let a = alloc_plain_object(&mut u);
        let b = alloc_plain_object(&mut u);
        let h1 = u.identity_hash(a.oop());
        let h2 = u.identity_hash(b.oop());
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
    }

    #[test]
    fn association_ivar_names() {
        let mut u = boot();
        let ivn = crate::oops::wrappers::ArrayOop::try_from(u.association_klass.inst_var_names())
            .expect("Association inst_var_names is an Array");
        assert_eq!(ivn.len(), 2);
        assert_eq!(ivn.at(0), u.intern(b"key").oop());
        assert_eq!(ivn.at(1), u.intern(b"value").oop());
    }

    #[test]
    fn eden_initial_state() {
        let u = boot();
        assert_eq!(u.eden.start % 8, 0);
        assert!(u.eden.top > u.eden.start);
        assert!(u.eden.top - u.eden.start < 256 * 1024);
        // `boot()` reserves 64 MiB, so `default_eden_for` sizes eden at a
        // quarter of that (the 32 MiB cap doesn't bind).
        assert_eq!(
            u.eden.end - u.eden.start,
            super::super::layout::default_eden_for(64 << 20)
        );
    }

    #[test]
    fn verify_heap_post_genesis() {
        let u = boot();
        super::super::verify::verify_heap(&u).expect("post-genesis heap must verify");
    }
}
