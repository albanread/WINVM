//! The `smalltalk` global namespace (SPEC §3.1: "a Dictionary of
//! Association"). S5's minimal representation: `Universe::smalltalk` is
//! `nil` until the first declaration, then an `ArrayOop` laid out
//! `[tally][assoc-or-nil…]` — dense, append-only, doubling growth. A real
//! `SystemDictionary` (S6, A5) replaces this; every caller goes through
//! `global_lookup`/`global_declare` so that swap is confined to this module.

use crate::memory::alloc;
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{ArrayOop, MemOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

const INITIAL_CAPACITY: usize = 16;

fn assoc_key(assoc: Oop) -> Oop {
    MemOop::try_from(assoc)
        .expect("globals: slot is not an Association")
        .body_oop(0)
}

/// Binds every genesis well-known klass under its own name (SPEC §3.2:
/// `Object subclass: Foo` — and everything else source can name — needs
/// `Object` etc. resolvable as an ordinary global). Called once, right
/// after genesis, by `VmState::with_options`; idempotent (re-running would
/// just no-op via `global_declare`'s existing-lookup fast path), so tests
/// that build a `Universe` directly (bypassing `VmState`) are unaffected.
pub(crate) fn bootstrap_well_known(vm: &mut VmState) {
    // Each klass is re-read fresh from `vm.universe.$f` (never captured
    // into a Rust array up front, the way this loop originally worked) and
    // handle-protected across `global_declare`'s own allocation — S7-9/
    // S7-10 (found via `MACVM_GC_STRESS=1`): a plain `[KlassOop; 32]` built
    // before the loop starts is exactly the "Rust container holding raw
    // oops" MacNCL lesson 13 warns about — every entry after the first
    // goes stale the moment the first `global_declare` call scavenges.
    let scope = crate::memory::handles::HandleScope::enter(vm);
    macro_rules! bootstrap_one {
        ($($f:ident),* $(,)?) => {
            $(
                let k = vm.universe.$f;
                let k_h = scope.handle(vm, k.oop());
                let name_sym = SymbolOop::try_from(k.name())
                    .expect("genesis klass name is always a Symbol");
                let name_sym_h = scope.handle(vm, name_sym);
                let assoc = global_declare(vm, name_sym_h.get(vm));
                MemOop::try_from(assoc)
                    .expect("global association is a mem oop")
                    .set_body_oop(1, k_h.get(vm));
            )*
        };
    }
    bootstrap_one!(
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
    );

    // `Smalltalk` itself (SPEC §3.2 step 3's `Smalltalk startUp` send target)
    // — bound last, once the namespace's own backing array definitely
    // exists (any `global_declare` call above already forced it).
    let smalltalk_sym = vm.universe.intern(b"Smalltalk");
    let assoc = global_declare(vm, smalltalk_sym);
    let smalltalk_obj = vm.universe.smalltalk;
    MemOop::try_from(assoc)
        .expect("global association is a mem oop")
        .set_body_oop(1, smalltalk_obj);

    // WINVM port: `Platform` — a Symbol naming the host OS, bound by the
    // VM binary itself so shared world source can select a platform FFI
    // surface at load time (world/30_date_time.mst's clock bindings are
    // the first user: clock_gettime does not exist on Windows, and an FFI
    // resolve miss is guest-FATAL, so the choice cannot be probed at
    // runtime). Symbols are interned, so `Platform == #windows` is a
    // plain identity test guest-side.
    let plat_sym = vm.universe.intern(b"Platform");
    let plat_assoc = global_declare(vm, plat_sym);
    let plat_assoc_h = scope.handle(vm, plat_assoc);
    let value = vm.universe.intern(if cfg!(target_os = "windows") {
        &b"windows"[..]
    } else if cfg!(target_os = "macos") {
        &b"macos"[..]
    } else {
        &b"unix"[..]
    });
    MemOop::try_from(plat_assoc_h.get(vm))
        .expect("global association is a mem oop")
        .set_body_oop(1, value.oop());
}

/// The bound `Association` for `name`, if declared.
pub fn global_lookup(vm: &VmState, name: SymbolOop) -> Option<Oop> {
    let arr = ArrayOop::try_from(vm.universe.smalltalk)?;
    let tally = SmallInt::try_from(arr.at(0))
        .expect("globals: tally is not a smi")
        .value() as usize;
    for i in 0..tally {
        let assoc = arr.at(1 + i);
        if assoc_key(assoc).raw() == name.oop().raw() {
            return Some(assoc);
        }
    }
    if std::env::var("MACVM_DBG_GLOBALS").is_ok() {
        eprintln!(
            "GDBG lookup MISS for {:?} (sym {:#x}); table tally={tally}:",
            name.as_string(),
            name.oop().raw()
        );
        for i in 0..tally {
            let assoc = arr.at(1 + i);
            let k = assoc_key(assoc);
            let ks = crate::oops::wrappers::SymbolOop::try_from(k)
                .map(|s| s.as_string())
                .unwrap_or_else(|| format!("<non-symbol {:#x}>", k.raw()));
            eprintln!(
                "GDBG   [{i}] assoc={:#x} key={:#x} {ks:?}",
                assoc.raw(),
                k.raw()
            );
        }
    }
    None
}

/// The sole instance of metaclass `meta` — the class object whose `klass()`
/// is `meta` — found by scanning the global namespace (every class the
/// compiler can meet is bound there; `Object subclass:` declares the name
/// before anything can send to the class). Compile-time use only: it is a
/// linear scan, priced for the JIT's once-per-compile `self basicNew`
/// fusion, not for a runtime path. Returns `None` when `meta` is not a
/// metaclass, or its class object isn't a declared global (an anonymous
/// class stays on the generic send path — correct, just not fused).
pub fn metaclass_sole_instance(
    vm: &VmState,
    meta: crate::oops::wrappers::KlassOop,
) -> Option<crate::oops::wrappers::KlassOop> {
    use crate::oops::wrappers::KlassOop;
    if meta.klass().oop().raw() != vm.universe.metaclass_klass.oop().raw() {
        return None;
    }
    let arr = ArrayOop::try_from(vm.universe.smalltalk)?;
    let tally = SmallInt::try_from(arr.at(0))
        .expect("globals: tally is not a smi")
        .value() as usize;
    for i in 0..tally {
        let assoc = arr.at(1 + i);
        let value = MemOop::try_from(assoc)
            .expect("globals: slot is not an Association")
            .body_oop(1);
        if let Some(k) = KlassOop::try_from(value) {
            if k.klass().oop().raw() == meta.oop().raw() {
                return Some(k);
            }
        }
    }
    None
}

/// The `Association` for `name`, creating it (value `nil`) if absent.
pub fn global_declare(vm: &mut VmState, name: SymbolOop) -> Oop {
    if let Some(assoc) = global_lookup(vm, name) {
        return assoc;
    }
    // `name` (reachable via the symbol table root, but a separate unrooted
    // copy in this parameter) is held across `ensure_capacity`'s own
    // possible allocation below — needs protecting (S7-9/S7-10, found via
    // `MACVM_GC_STRESS=1`).
    let scope = crate::memory::handles::HandleScope::enter(vm);
    let name_h = scope.handle(vm, name);

    ensure_capacity(vm);

    let association_klass = vm.universe.association_klass;
    let assoc = alloc::alloc_slots(vm, association_klass);
    let nil = vm.universe.nil_obj;
    assoc.set_body_oop(0, name_h.get(vm).oop());
    assoc.set_body_oop(1, nil);

    let arr = ArrayOop::try_from(vm.universe.smalltalk)
        .expect("globals: smalltalk is not an Array after ensure_capacity");
    let tally = SmallInt::try_from(arr.at(0)).unwrap().value() as usize;
    arr.at_put(1 + tally, assoc.oop());
    arr.at_put(0, SmallInt::new((tally + 1) as i64).oop());
    // The globals array is long-lived (promoted early); appending a fresh
    // (young) Association through the raw `at_put` creates an unbarriered
    // old→new edge (S7-10 — this exact write was the first clean-card
    // violation the A9 verifier caught during a GC_STRESS world load).
    crate::memory::store::post_write_barrier(vm, arr.as_mem());
    assoc.oop()
}

/// Grows (or allocates for the first time) so at least one more slot is
/// free past the current tally.
fn ensure_capacity(vm: &mut VmState) {
    // Same `[tally][assoc-or-nil…]` `IndexableOops` layout `array_klass`
    // itself uses — `system_dictionary_klass` just reclassifies the object
    // (SPEC-QUESTION A5: `smalltalk` is a SystemDictionary, not a bare
    // Array), no representation change.
    let sysdict_klass = vm.universe.system_dictionary_klass;
    match ArrayOop::try_from(vm.universe.smalltalk) {
        None => {
            let arr = alloc::alloc_indexable_oops(vm, sysdict_klass, 1 + INITIAL_CAPACITY);
            arr.at_put(0, SmallInt::new(0).oop());
            vm.universe.smalltalk = arr.oop();
        }
        Some(arr) => {
            let tally = SmallInt::try_from(arr.at(0)).unwrap().value() as usize;
            let cap = arr.len() - 1;
            if tally < cap {
                return;
            }
            let new_cap = cap * 2;
            let grown = alloc::alloc_indexable_oops(vm, sysdict_klass, 1 + new_cap);
            let arr = ArrayOop::try_from(vm.universe.smalltalk).unwrap();
            grown.at_put(0, SmallInt::new(tally as i64).oop());
            for i in 0..tally {
                grown.at_put(1 + i, arr.at(1 + i));
            }
            vm.universe.smalltalk = grown.oop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn declare_then_lookup() {
        let mut vm = test_vm();
        let name = vm.universe.intern(b"Foo");
        assert!(global_lookup(&vm, name).is_none());
        let assoc = global_declare(&mut vm, name);
        assert_eq!(global_lookup(&vm, name), Some(assoc));
        assert_eq!(assoc_key(assoc), name.oop());
        assert_eq!(
            MemOop::try_from(assoc).unwrap().body_oop(1),
            vm.universe.nil_obj
        );
    }

    #[test]
    fn redeclare_is_idempotent() {
        let mut vm = test_vm();
        let name = vm.universe.intern(b"Foo");
        let a1 = global_declare(&mut vm, name);
        let a2 = global_declare(&mut vm, name);
        assert_eq!(a1, a2);
    }

    #[test]
    fn grows_past_initial_capacity() {
        let mut vm = test_vm();
        let mut assocs = Vec::new();
        for i in 0..(INITIAL_CAPACITY * 3) {
            let name = vm.universe.intern(format!("G{i}").as_bytes());
            assocs.push((name, global_declare(&mut vm, name)));
        }
        for (name, assoc) in assocs {
            assert_eq!(global_lookup(&vm, name), Some(assoc));
        }
    }

    #[test]
    fn value_write_persists() {
        let mut vm = test_vm();
        let name = vm.universe.intern(b"Foo");
        let assoc = global_declare(&mut vm, name);
        let m = MemOop::try_from(assoc).unwrap();
        let v = SmallInt::new(42).oop();
        m.set_body_oop(1, v);
        let assoc2 = global_lookup(&vm, name).unwrap();
        assert_eq!(MemOop::try_from(assoc2).unwrap().body_oop(1), v);
    }
}
