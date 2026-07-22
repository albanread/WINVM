//! Sprint S1 integration tests (`tests_s01.md` §Integration/golden tests).
//! Exercises the crate's public API surface only — this is what an
//! embedder (and S2's interpreter) builds on.

use macvm::memory::verify;
use macvm::oops::wrappers::{ArrayOop, ByteArrayOop, KlassOop, MemOop};
use macvm::runtime::{VmOptions, VmState};

/// A small, fixed-seed xorshift PRNG — no external crate needed, and a
/// fixed seed makes any `alloc_torture_fill` failure reproducible.
struct Xorshift(u64);
impl Xorshift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn boot_and_verify() {
    let vm = VmState::new();
    let u = &vm.universe;

    let nil = MemOop::try_from(u.nil_obj).expect("nil is a mem oop");
    assert_eq!(nil.klass(), u.undefined_object_klass);

    assert_eq!(u.object_klass.klass().klass(), u.metaclass_klass);
    assert_eq!(u.metaclass_klass.klass().klass(), u.metaclass_klass);

    assert_eq!(u.object_klass.superclass(), u.nil_obj);
    assert_eq!(
        u.object_klass.klass().superclass().raw(),
        u.class_klass.oop().raw()
    );

    verify::verify_heap(u).expect("post-genesis heap must verify");
}

#[test]
fn small_heap_boot() {
    let vm = VmState::with_options(VmOptions {
        heap_mib: 16,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Off,
    });
    verify::verify_heap(&vm.universe).expect("genesis must succeed in a 16 MiB reservation");
}

#[test]
fn eden_exhaustion_aborts() {
    let exe = env!("CARGO_BIN_EXE_macvm");
    let output = std::process::Command::new(exe)
        .arg("--selftest-alloc-loop")
        .env("MACVM_HEAP", "16")
        // These tests pin exhaustion behavior at a specific tiny geometry;
        // the 16 MiB default eden (40fc343) cannot even be carved from a
        // 16 MiB reservation, so pin the old 4 MiB eden explicitly.
        .env("MACVM_EDEN", "4096")
        // This self-test loop doesn't root its allocations via Handles, so
        // an inherited `MACVM_GC_STRESS=1` would let a mid-loop scavenge
        // reclaim eden back to empty every time — exhaustion would never
        // actually happen. Force it off regardless of the parent env.
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
        stderr.contains("heap exhausted"),
        "stderr did not mention exhaustion: {stderr}"
    );
}

#[test]
fn alloc_torture_fill() {
    // Deliberately keeps allocated objects reachable only via a raw
    // `Vec<Oop>` (not `Handle`s), so it must force `gc_stress: false` here
    // rather than inheriting the ambient `MACVM_GC_STRESS` env var: under
    // stress-every-alloc, a scavenge mid-loop would find these objects
    // unrooted and reclaim eden right back to empty, so `target` is never
    // reached and the loop never terminates.
    let mut vm = VmState::with_options(VmOptions {
        heap_mib: VmOptions::DEFAULT_HEAP_MIB,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Off,
    });
    let target = vm.universe.eden.end - (1 << 20); // stop ~1 MiB below capacity
    let mut rng = Xorshift(0x5EED_1234_ABCD_EF01);

    let mut objects: Vec<macvm::oops::Oop> = Vec::new();
    while vm.universe.eden.top < target {
        let kind = rng.range(3);
        let n = rng.range(201);
        match kind {
            0 => {
                let klass = vm.universe.object_klass;
                let obj = macvm::memory::alloc::alloc_slots(&mut vm, klass);
                objects.push(obj.oop());
            }
            1 => {
                let klass = vm.universe.array_klass;
                let obj = macvm::memory::alloc::alloc_indexable_oops(&mut vm, klass, n);
                objects.push(obj.oop());
            }
            _ => {
                let klass = vm.universe.bytearray_klass;
                let obj = macvm::memory::alloc::alloc_indexable_bytes(&mut vm, klass, n);
                objects.push(obj.oop());
            }
        }
    }

    verify::verify_heap(&vm.universe).expect("heap must verify after torture fill");

    // Spot-check 100 deterministically-chosen objects (same rng sequence,
    // so any failure is reproducible by line number / iteration index).
    let mut check_rng = Xorshift(0xC0FF_EE00_1122_3344);
    for _ in 0..100.min(objects.len()) {
        let idx = check_rng.range(objects.len());
        let o = objects[idx];
        let m = MemOop::try_from(o).expect("torture object is mem-tagged");
        let k = m.klass();
        if k == vm.universe.array_klass {
            let a = ArrayOop::try_from(o).unwrap();
            for i in 0..a.len() {
                assert_eq!(a.at(i), vm.universe.nil_obj, "Array element not nil-filled");
            }
        } else if k == vm.universe.bytearray_klass {
            let b = ByteArrayOop::try_from(o).unwrap();
            let mut bytes = Vec::new();
            b.copy_bytes_out(&mut bytes);
            // Padding-to-word-boundary bytes are zero forever; every byte
            // here is padding since nothing wrote real content.
            assert!(bytes.iter().all(|&b| b == 0), "ByteArray not zero-filled");
        } else {
            assert_eq!(k, vm.universe.object_klass);
        }
    }
}

#[test]
fn genesis_is_deterministic() {
    let vm1 = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Off,
    });
    let vm2 = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Off,
    });

    let klasses1: Vec<KlassOop> = well_known(&vm1);
    let klasses2: Vec<KlassOop> = well_known(&vm2);

    for (k1, k2) in klasses1.iter().zip(klasses2.iter()) {
        let off1 = k1.addr() - vm1.universe.eden.start;
        let off2 = k2.addr() - vm2.universe.eden.start;
        assert_eq!(off1, off2, "genesis layout is not deterministic");
    }
}

fn well_known(vm: &VmState) -> Vec<KlassOop> {
    let u = &vm.universe;
    vec![
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
        u.association_klass,
        u.methoddict_klass,
        u.method_klass,
        u.closure_klass,
        u.context_klass,
        u.process_klass,
    ]
}

// --- oops::wrappers behavior (needs real allocated objects, hence here
// rather than in-module: `oops` must not import `memory` even for tests,
// keeping the production layer boundary meaningful) ------------------------

#[test]
fn array_at_put() {
    let mut vm = VmState::new();
    let klass = vm.universe.array_klass;
    let a = macvm::memory::alloc::alloc_indexable_oops(&mut vm, klass, 3);
    let seven = macvm::oops::smi::SmallInt::new(7).oop();
    a.at_put(0, seven);
    assert_eq!(a.at(0), seven);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "out of bounds")]
fn array_at_out_of_bounds_panics() {
    let mut vm = VmState::new();
    let klass = vm.universe.array_klass;
    let a = macvm::memory::alloc::alloc_indexable_oops(&mut vm, klass, 3);
    let _ = a.at(3);
}

#[test]
fn bytearray_at_put() {
    let mut vm = VmState::new();
    let klass = vm.universe.bytearray_klass;
    let b = macvm::memory::alloc::alloc_indexable_bytes(&mut vm, klass, 9);
    for i in 0..9u8 {
        b.byte_at_put(i as usize, i);
    }
    for i in 0..9u8 {
        assert_eq!(b.byte_at(i as usize), i);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "out of bounds")]
fn bytearray_byte_at_out_of_bounds_panics() {
    let mut vm = VmState::new();
    let klass = vm.universe.bytearray_klass;
    let b = macvm::memory::alloc::alloc_indexable_bytes(&mut vm, klass, 9);
    let _ = b.byte_at(9);
}

#[test]
fn wrapper_format_checks() {
    let mut vm = VmState::new();
    // Each object is allocated and immediately checked with NO allocation in
    // between, and each klass is fetched from the universe right before its
    // use: under MACVM_GC_STRESS=1 every alloc scavenges, so caching klasses
    // up front or holding an earlier result oop across a later alloc would
    // dangle it (SPEC §7.6.1).
    let array_klass = vm.universe.array_klass;
    let array_oop = macvm::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, 1).oop();
    assert!(ArrayOop::try_from(array_oop).is_some());
    assert!(KlassOop::try_from(vm.universe.array_klass.oop()).is_some());

    let bytearray_klass = vm.universe.bytearray_klass;
    let bytearray_oop =
        macvm::memory::alloc::alloc_indexable_bytes(&mut vm, bytearray_klass, 1).oop();
    assert!(ArrayOop::try_from(bytearray_oop).is_none());

    // A String shares IndexableBytes format with Symbol but must not pass
    // the exact-klass Symbol check.
    let string_klass = vm.universe.string_klass;
    let string_oop = macvm::memory::alloc::alloc_indexable_bytes(&mut vm, string_klass, 3).oop();
    let symbol_klass = vm.universe.symbol_klass;
    assert!(macvm::oops::wrappers::SymbolOop::try_from_exact(string_oop, symbol_klass).is_none());
}

#[test]
fn klass_reopened_as_wrong_format() {
    let mut vm = VmState::new();
    let array_klass = vm.universe.array_klass;
    let a = macvm::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, 0).oop();
    assert!(ArrayOop::try_from(a).is_some());

    // Hand-flip the klass's format (a legitimate, non-test-only API — no
    // special backdoor needed): wrappers must re-check at every
    // construction, never cache. Fetch `array_klass` fresh here rather than
    // reusing a copy taken before the allocation above: under
    // MACVM_GC_STRESS that alloc scavenges, and array_klass is a new-gen
    // object that can move — a pre-GC copy would flip the format on a
    // vacated from-space address while the live klass stays untouched.
    let klass = vm.universe.array_klass;
    klass.set_format(macvm::oops::Format::IndexableBytes, true);
    assert!(ArrayOop::try_from(a).is_none());
    assert!(ByteArrayOop::try_from(a).is_some());

    // Restore, so any later assertions in the same process aren't confused
    // (defensive; each #[test] gets a fresh VmState in this crate anyway).
    vm.universe
        .array_klass
        .set_format(macvm::oops::Format::IndexableOops, false);
}

#[test]
fn double_adversarial_bit_patterns() {
    let mut vm = VmState::new();
    // A Double's raw body word whose bits happen to look like a mark word
    // (tag 11) or a heap oop (tag 01) must not confuse the heap walker,
    // which dispatches on FORMAT before ever reading body words as oops.
    let eden_start = vm.universe.eden.start;
    let d1 = macvm::memory::alloc::alloc_double(&mut vm, f64::from_bits(0b111));
    let d2 = macvm::memory::alloc::alloc_double(&mut vm, f64::from_bits(eden_start as u64 | 1));
    verify::verify_heap(&vm.universe).expect("adversarial Double bits must not confuse the walker");
    let _ = (d1, d2);
}

#[test]
fn interning_collision_pressure() {
    let mut u = VmState::new();
    let names: Vec<String> = (0..10_000).map(|i| format!("pfx{}", i % 16)).collect();
    // Names collide heavily on a 4-char prefix but are NOT all identical
    // (the `i` suffix varies) — wait, "pfx{i%16}" repeats every 16, so this
    // exercises real duplicate interning (same content -> same oop) under
    // hash-bucket pressure, which is the actual risk this row targets.
    let first: Vec<macvm::oops::Oop> = names
        .iter()
        .map(|n| u.universe.intern(n.as_bytes()).oop())
        .collect();
    for (n, expected) in names.iter().zip(first.iter()) {
        assert_eq!(u.universe.intern(n.as_bytes()).oop().raw(), expected.raw());
    }
}
