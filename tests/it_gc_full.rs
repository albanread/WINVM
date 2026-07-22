//! Sprint S8 integration/golden tests (`tests_s08.md` §Integration/golden
//! tests, gate items 2-3). `memory::fullgc`'s own unit tests cover the
//! phases in isolation on hand-built micro-heaps; these exercise the whole
//! `full_gc` pipeline at a scale a unit test wouldn't bother with, entirely
//! through the crate's public API — same spirit as `it_memory.rs`'s S1/S7
//! coverage.

mod common;

use macvm::memory::alloc;
use macvm::memory::fullgc::full_gc;
use macvm::memory::handles::HandleScope;
use macvm::memory::scavenge::scavenge;
use macvm::memory::store;
use macvm::memory::verify;
use macvm::oops::layout::HEADER_WORDS;
use macvm::oops::{ArrayOop, ByteArrayOop, DoubleOop, Format, KlassOop, MemOop, Oop, SmallInt};
use macvm::runtime::VmState;

/// A structural, address-independent fold over a live object graph: klass
/// NAME is deliberately never read (class objects fold their format +
/// shape instead, the `Format::Klass` arm) and no arm ever touches an
/// address or an identity hash — exactly the properties compaction is and
/// isn't allowed to disturb (`tests_s08.md`'s `fragmentation_checksum`).
fn checksum(vm: &VmState, oop: Oop) -> u64 {
    fn mix(h: u64, v: u64) -> u64 {
        (h ^ v).wrapping_mul(0x0100_0000_01b3)
    }
    if let Some(smi) = SmallInt::try_from(oop) {
        return mix(0x5A11, smi.value() as u64);
    }
    let u = &vm.universe;
    if oop == u.nil_obj {
        return 0x1111;
    }
    if oop == u.true_obj {
        return 0x2222;
    }
    if oop == u.false_obj {
        return 0x3333;
    }
    let obj = MemOop::try_from(oop).expect("checksum: oop is neither smi, nil/true/false, nor mem");
    let klass = obj.klass();
    match klass.format() {
        Format::IndexableOops => {
            let arr = ArrayOop::try_from(oop).unwrap();
            let mut h = mix(0x10, arr.len() as u64);
            for i in 0..arr.len() {
                h = mix(h, checksum(vm, arr.at(i)));
            }
            h
        }
        Format::IndexableBytes => {
            let ba = ByteArrayOop::try_from(oop).unwrap();
            let mut h = mix(0x20, ba.len() as u64);
            for i in 0..ba.len() {
                h = mix(h, ba.byte_at(i) as u64);
            }
            h
        }
        Format::Double => {
            let d = DoubleOop::try_from(oop).unwrap();
            mix(0x30, d.value().to_bits())
        }
        Format::Slots => {
            let n = klass.non_indexable_size() - HEADER_WORDS;
            let mut h = mix(0x40, n as u64);
            for i in 0..n {
                h = mix(h, checksum(vm, obj.body_oop(i)));
            }
            h
        }
        Format::Klass => {
            let k = KlassOop::try_from(oop).unwrap();
            mix(mix(0x50, k.format() as u64), k.non_indexable_size() as u64)
        }
        other => panic!("checksum: unhandled format {other:?} in this test's mix"),
    }
}

/// One of 5 formats, cycling on `i` — Slots/IndexableOops/IndexableBytes
/// (odd, non-word-aligned lengths)/Double/a class object, per
/// `tests_s08.md`'s "mixed formats" requirement. The class-object case
/// references one of the VM's own permanent klasses rather than
/// synthesizing fresh ones — a hand-built klass has enough of its own
/// invariants (format, superclass chain, ivar-names array, metaclass) that
/// getting one right from a test harness risks testing the harness, not
/// full GC.
/// Every klass is fetched fresh from `vm.universe` on each call, never
/// cached by the caller across a loop: a full GC can relocate even a
/// permanent well-known klass (it is compacted like any other old-gen
/// object, just always re-rooted), so a `KlassOop` captured before an
/// earlier iteration's GC event is exactly the stale-reference bug this
/// whole sprint has been about — this time in test code, not production.
fn make_mixed(vm: &mut VmState, i: usize) -> Oop {
    let array_klass = vm.universe.array_klass;
    let bytearray_klass = vm.universe.bytearray_klass;
    let association_klass = vm.universe.association_klass;
    let class_refs = [
        array_klass,
        bytearray_klass,
        vm.universe.double_klass,
        association_klass,
    ];
    match i % 5 {
        0 => {
            let a = alloc::alloc_indexable_oops(vm, array_klass, 2);
            a.at_put(0, SmallInt::new(i as i64).oop());
            a.at_put(1, SmallInt::new((i as i64) * 3).oop());
            a.oop()
        }
        1 => {
            let len = (i % 7) + 1; // odd, deliberately not word-aligned
            let ba = alloc::alloc_indexable_bytes(vm, bytearray_klass, len);
            for j in 0..len {
                ba.byte_at_put(j, ((i * 31 + j * 7) & 0xFF) as u8);
            }
            ba.oop()
        }
        2 => alloc::alloc_double(vm, i as f64 * 1.5 + 0.25).oop(),
        3 => {
            let s = alloc::alloc_slots(vm, association_klass);
            s.set_body_oop(0, SmallInt::new(i as i64).oop());
            s.set_body_oop(1, SmallInt::new(-(i as i64)).oop());
            s.oop()
        }
        _ => class_refs[i % class_refs.len()].oop(),
    }
}

/// Gate item 2: fragment old gen (mixed formats, half abandoned), full-GC,
/// and prove the live graph's CONTENT survived exactly (not just that
/// nothing crashed) while space was measurably reclaimed.
#[test]
fn fragmentation_checksum() {
    let mut vm = common::test_vm();
    vm.universe.tenuring_threshold = 0;

    let n = 2_000usize;
    let k = vm.universe.array_klass;
    let full = alloc::alloc_indexable_oops(&mut vm, k, n);
    vm.stack.push(full.oop());
    for i in 0..n {
        let o = make_mixed(&mut vm, i);
        // Re-fetch `full`'s current address every iteration: `make_mixed`
        // allocates, which under MACVM_GC_STRESS=full:1 can compact old gen
        // (and `full` gets promoted into old gen well before this loop
        // ends) on literally every call.
        let full_now = ArrayOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
        // Barriered store, not a raw `at_put`: under MACVM_GC_STRESS=1,
        // every allocation scavenges first (with tenuring_threshold=0), so
        // `full` can be promoted to old gen within the first few
        // iterations — long before this loop's own explicit `scavenge`
        // call — making every subsequent store an old object gaining a
        // fresh young-gen reference.
        let full_mem = MemOop::try_from(full_now.oop()).unwrap();
        store::store_tail_oop(&mut vm, full_mem, i, o);
    }
    scavenge(&mut vm).expect("promote the whole population into old gen");

    let half = n / 2;
    // `array_klass` fetched fresh here, not reused from before the
    // `scavenge` above — same stale-klass hazard `make_mixed` guards
    // against.
    let k = vm.universe.array_klass;
    let survivors = alloc::alloc_indexable_oops(&mut vm, k, half);
    let full_now = ArrayOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
    for i in 0..half {
        survivors.at_put(i, full_now.at(2 * i)); // keep evens, abandon odds
    }
    vm.stack.pop(); // drop `full` — every odd-indexed object is now garbage
    vm.stack.push(survivors.oop());

    let old_used_before = vm.universe.old.top - vm.universe.old.bounds.start;
    let s = ArrayOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
    let checksum_before: Vec<u64> = (0..half).map(|i| checksum(&vm, s.at(i))).collect();

    full_gc(&mut vm).expect("full_gc must succeed");

    let old_used_after = vm.universe.old.top - vm.universe.old.bounds.start;
    let s = ArrayOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
    let checksum_after: Vec<u64> = (0..half).map(|i| checksum(&vm, s.at(i))).collect();

    assert_eq!(
        checksum_before, checksum_after,
        "the live graph's structural content must be identical before and after compaction"
    );
    assert!(
        old_used_after <= old_used_before * 6 / 10,
        "compacting away half the population must reclaim substantial space: \
         before {old_used_before}, after {old_used_after}"
    );
}

/// Gate item 3: identity hash and identity-keyed lookups survive
/// compaction. Uses `Handle<Oop>` (SPEC §7.6.1's GC-safe indirection) as
/// the Rust-side "identity-keyed Dictionary" analogue — `handle.get(vm)`
/// IS an identity-preserving lookup by construction, so recomputing each
/// hash from the handle's current (possibly relocated) oop after two full
/// GCs is exactly the doc's "Vec<(Handle, u64)>" check.
/// `world/tests/gc_full_test.mst`'s `testIdentityDictionaryAfterGcFull`
/// covers the world-level Dictionary half of this same gate item.
#[test]
fn identity_hash_stability() {
    let mut vm = common::test_vm();
    let scope = HandleScope::enter(&mut vm);

    let mut handles = Vec::new();
    for i in 0..1000 {
        // Re-fetched every iteration: under MACVM_GC_STRESS=1, EVERY
        // allocation scavenges first, so a klass captured once before this
        // loop is stale by iteration 2 — same hazard `make_mixed` guards
        // against, just reachable here too since a 1000-iteration loop
        // gives stress mode 1000 chances to move it.
        let klass = vm.universe.array_klass;
        let o = alloc::alloc_indexable_oops(&mut vm, klass, 1);
        o.at_put(0, SmallInt::new(i as i64).oop());
        handles.push(scope.handle(&mut vm, o.oop()));
    }

    let hashes_before: Vec<u32> = handles
        .iter()
        .map(|h| {
            let o = h.get(&vm);
            vm.universe.identity_hash(o)
        })
        .collect();

    full_gc(&mut vm).expect("first gcFull");
    full_gc(&mut vm).expect("second gcFull");

    for (h, &expected) in handles.iter().zip(hashes_before.iter()) {
        let current = h.get(&vm);
        assert_eq!(
            vm.universe.identity_hash(current),
            expected,
            "identity hash must survive two full GCs"
        );
    }
    drop(scope);
}

/// Old gen must actually climb its growth ladder under sustained live
/// retention (not just tolerate one `grow` call), then give every reclaimed
/// byte back once that retained data is dropped and compacted away.
#[test]
fn old_growth_ladder() {
    let mut vm = common::test_vm();
    let old_baseline = vm.universe.old.top - vm.universe.old.bounds.start;

    // ByteArrays, not Arrays: `scavenge::scan_slots_in_window`'s
    // IndexableOops branch loops over an object's ENTIRE oop tail on every
    // dirty card it's asked to scan (window membership is filtered inside
    // the loop body, not by bounding the loop itself), so a large old-gen
    // Array costs O(cards * length) to rescan, not O(length) — a real
    // production perf bug (raw bytes have no such loop at all — nothing to
    // scan — so IndexableBytes doesn't trigger it; flagged separately,
    // worth its own fix outside this sprint's scope). Each ~10 MiB chunk is
    // bigger than eden (4 MiB), so `alloc_indexable_bytes` routes it
    // straight to a direct old-gen allocation (alloc.rs's cascade step 4).
    // 5 * ~10 MiB = ~50 MiB, comfortably past 2 growth increments
    // (OLD_INITIAL_SEGMENT + OLD_GROWTH_SEGMENT, 16 MiB each) while leaving
    // slack under a 64 MiB test heap's ~59 MiB old-gen reservation.
    let ak = vm.universe.array_klass;
    let keep = alloc::alloc_indexable_oops(&mut vm, ak, 5);
    vm.stack.push(keep.oop());

    for i in 0..5 {
        // Re-fetched every iteration: an earlier chunk's own direct
        // allocation/`grow_old` can relocate even a permanent klass (same
        // hazard `make_mixed` guards against).
        let bk = vm.universe.bytearray_klass;
        let chunk = alloc::alloc_indexable_bytes(&mut vm, bk, 10_000_000);
        // No write barrier needed: `keep` (5 slots) never leaves eden, so
        // this is a young object gaining an old-gen reference — the
        // barrier only cares about the other direction (store()'s own doc).
        let keep_now = ArrayOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
        keep_now.at_put(i, chunk.oop());
    }
    assert!(
        vm.universe.gc_stats.old_grow_count >= 2,
        "old gen must have grown past its initial segment at least twice, got {}",
        vm.universe.gc_stats.old_grow_count
    );

    verify::verify_heap(&vm.universe).expect("heap must verify after sustained growth");

    vm.stack.pop(); // drop everything this test retained
    full_gc(&mut vm).expect("final gcFull after dropping everything");

    let old_used_after = vm.universe.old.top - vm.universe.old.bounds.start;
    assert!(
        old_used_after <= old_baseline.max(4096) * 3,
        "old gen must return to near its pre-test baseline once everything is dropped: \
         baseline {old_baseline}, after {old_used_after}"
    );
}

/// Gate item 1's flip side: when the cascade truly cannot help, it must
/// fail in a controlled, honest way — a terminal `GcStallError` whose own
/// progress numbers show a full GC ran and found nothing worth reclaiming
/// (`--selftest-alloc-loop` keeps every object reachable forever, so a
/// pre-failure full GC is 100% live — the truthful reading is "reclaimed
/// approximately nothing", not a crash and not a silent hang). Reuses the
/// self-test binary mode `it_memory.rs`'s S7 `eden_exhaustion_aborts`
/// already established; this test's own contribution is the S8-specific
/// assertion that `full_gc_count > 0` by the time the cascade gives up.
#[test]
fn stall_after_full() {
    let exe = env!("CARGO_BIN_EXE_macvm");
    let output = std::process::Command::new(exe)
        .arg("--selftest-alloc-loop")
        .env("MACVM_HEAP", "16")
        // These tests pin exhaustion behavior at a specific tiny geometry;
        // the 16 MiB default eden (40fc343) cannot even be carved from a
        // 16 MiB reservation, so pin the old 4 MiB eden explicitly.
        .env("MACVM_EDEN", "4096")
        .env_remove("MACVM_GC_STRESS")
        .output()
        .expect("failed to spawn macvm binary");

    assert_eq!(
        output.status.code(),
        Some(70),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "must be a controlled exit, not a panic: {stderr}"
    );
    assert!(
        stderr.contains("heap exhausted"),
        "stderr did not mention exhaustion: {stderr}"
    );

    // "  gc:        N scavenges, M full GCs" — the count comes BEFORE the
    // marker token.
    let find_token_before = |marker: &str| -> u64 {
        stderr
            .lines()
            .find_map(|l| {
                let tokens: Vec<&str> = l.split_whitespace().collect();
                let idx = tokens.iter().position(|&t| t == marker)?;
                tokens.get(idx.checked_sub(1)?)?.parse().ok()
            })
            .unwrap_or_else(|| panic!("stderr has no parseable '{marker}' line: {stderr}"))
    };
    // "  progress:  ... / reclaimed N bytes" — the count comes AFTER.
    let find_token_after = |marker: &str| -> u64 {
        stderr
            .lines()
            .find_map(|l| {
                let tokens: Vec<&str> = l.split_whitespace().collect();
                let idx = tokens.iter().position(|&t| t == marker)?;
                tokens.get(idx + 1)?.parse().ok()
            })
            .unwrap_or_else(|| panic!("stderr has no parseable '{marker}' line: {stderr}"))
    };

    let full_gc_count = find_token_before("full");
    assert!(
        full_gc_count > 0,
        "the cascade must have attempted at least one full GC before giving up: {stderr}"
    );
    let reclaimed = find_token_after("reclaimed");
    assert!(
        reclaimed < 4096,
        "an all-live workload's terminal full GC must reclaim ~nothing \
         (progress numbers telling the truth, not padding): reclaimed {reclaimed}: {stderr}"
    );
}

/// `MACVM_DBG_OOP`'s traced address must follow one object through BOTH a
/// scavenge (promotion into old gen) and a full GC (compaction within old
/// gen), always resolving to the live object — never a stale or forwarded
/// address — with a real, restored mark at the end.
#[test]
fn dbg_oop_through_compaction() {
    let mut vm = common::test_vm();

    // `tenuring_threshold` is reset to 0 immediately before EVERY call
    // that can trigger a scavenge — not just the explicit `scavenge()`
    // calls below, but every allocation too: under MACVM_GC_STRESS=1, an
    // allocation scavenges BEFORE it runs, and `scavenge` always ends by
    // recomputing the threshold from the adaptive age-table policy
    // (`age_table.compute_threshold`), overwriting whatever it was set to.
    // A single `= 0` at the top of this test is not durable across any of
    // that.
    //
    // `dead` is promoted FIRST, so it claims the lower old-gen address —
    // abandoning it later is what gives `obj` (promoted second, so it
    // lands right after `dead`) somewhere to shift down INTO during
    // full_gc. Allocated and promoted before `obj` even exists, so there
    // is nothing yet to root or trace.
    vm.universe.tenuring_threshold = 0;
    let klass = vm.universe.array_klass;
    let dead = alloc::alloc_indexable_oops(&mut vm, klass, 0);
    vm.stack.push(dead.oop());
    vm.universe.tenuring_threshold = 0;
    scavenge(&mut vm).expect("scavenge must promote the dead sibling");
    vm.stack.pop(); // abandon it — unreachable going into full_gc, but sits in old gen until then

    // Re-fetch the klass: the scavenge above can have relocated even this
    // permanent klass (same hazard as fragmentation_checksum /
    // old_growth_ladder).
    vm.universe.tenuring_threshold = 0;
    let klass = vm.universe.array_klass;
    let obj = alloc::alloc_indexable_oops(&mut vm, klass, 0);
    vm.stack.push(obj.oop());
    vm.dbg_oop = Some(obj.oop().mem_addr());

    vm.universe.tenuring_threshold = 0;
    scavenge(&mut vm).expect("scavenge must promote");
    let after_scavenge = vm.dbg_oop.expect("dbg_oop must survive a scavenge");
    assert!(
        vm.universe.layout.is_old(after_scavenge),
        "dbg_oop must follow the object into old gen"
    );
    assert_eq!(
        Some(vm.stack.get(vm.stack.sp - 1).mem_addr()),
        vm.dbg_oop,
        "dbg_oop must match the promoted object's actual address"
    );

    full_gc(&mut vm).expect("full_gc must succeed");

    let after_full_gc = vm.dbg_oop.expect("dbg_oop must survive a full GC");
    assert_ne!(
        after_full_gc, after_scavenge,
        "the object must have actually moved during compaction"
    );
    assert_eq!(
        Some(vm.stack.get(vm.stack.sp - 1).mem_addr()),
        vm.dbg_oop,
        "dbg_oop must match the compacted object's actual address"
    );

    let moved = MemOop::try_from(vm.stack.get(vm.stack.sp - 1)).unwrap();
    assert!(
        !moved.is_forwarded(),
        "the traced object's final mark must be restored, not a leftover forwarding tag"
    );
}
