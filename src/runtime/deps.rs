//! S13 D1 §2 + D2: method-redefinition dependency invalidation — the hook
//! that turns steps 8/9's *mechanism* (`flush::make_not_entrant`) into live
//! *behaviour*. [`crate::runtime::lookup::install_method`] calls
//! [`invalidate_dependents`] after publishing a `(holder, selector)` binding;
//! every compiled method whose lookup result would change is made
//! `NotEntrant` (§2a/§2b) and its in-flight frames redirected (§2c) so they
//! deopt when control next leaves them.
//!
//! # Why S13 has no `DependencyIndex` cache
//!
//! `sprint_s13_detail.md`'s "Dependency index" specifies a stateful
//! `HashMap<(Oop, Oop), SmallVec<NmethodId>>` rebuilt from each nmethod's
//! `deps[]` array on every GC. That machinery earns its keep only once an
//! nmethod can depend on a method OTHER than its own — i.e. once **S14's
//! inliner** records `DependencyIndex::record(callee_klass, callee_sel, nm)`
//! per inlined body (the doc says exactly that: "S14's inliner records one
//! dependency per inline"). A **tier-1, non-inlining** S13 nmethod makes
//! exactly ONE assumption: that `lookup(key_klass, key_selector)` resolves to
//! the very method it compiled. Its entire dependency set is therefore the
//! `(key_klass, key_selector)` pair it already stores — there is nothing to
//! cache beyond [`CodeTable::iter_alive`](crate::codecache::nmethod::CodeTable::iter_alive),
//! and [`affected_by_install`] simply scans it. Scanning, not caching,
//! deliberately sidesteps the exact oop-keyed-staleness hazard the doc itself
//! flags as the reason its cache must be rebuilt after every collection
//! (young symbols/klasses move under scavenge) — a hazard with zero upside
//! while `deps == key`. The `DependencyIndex` struct + `record` /
//! `invalidate_cache` land with the inliner in S14.
//!
//! Cost: every `install_method` now pays an `O(alive nmethods)` scan. Installs
//! are rare relative to sends (bootstrap installs each selector once, before
//! the JIT has compiled anything; steady-state installs are redefinitions),
//! and the alive-nmethod count is small under tier-1, so this stays off any
//! hot path — consistent with the sprint's "no perf work before S15" stance.

use crate::codecache::nmethod::NmethodId;
use crate::memory::Universe;
use crate::oops::wrappers::{KlassOop, SymbolOop};
use crate::runtime::vm_state::VmState;

/// D2. Every **alive** nmethod whose sole dependency `(key_klass,
/// key_selector)` is invalidated by installing `selector` in `holder`.
///
/// The conservative v1 rule (doc §D2): the pair is affected iff
/// `nm.key_selector == selector` **and** `holder` lies on `nm.key_klass`'s
/// superclass chain — `key_klass` itself included — because `lookup(key_klass,
/// selector)` walks exactly that chain, so a new binding for `selector`
/// anywhere on it can change the method found. Superclass *changes* are
/// unsupported in v1 (world files only add methods, SPEC §1.2), so this
/// direction-of-walk is complete.
///
/// Identity is compared on raw oops: selectors are interned (one canonical
/// oop each) and klasses are genesis singletons, so raw equality *is* object
/// identity here. Valid because the whole scan runs to completion within one
/// GC-free window — installation allocates nothing past this point and runs
/// no guest code, so no collection moves an oop mid-scan.
pub fn affected_by_install(vm: &VmState, holder: KlassOop, selector: SymbolOop) -> Vec<NmethodId> {
    let sel_raw = selector.oop().raw();
    let mut victims = Vec::new();
    for nm in vm.code_table.iter_alive() {
        // Own key: the nmethod compiled `lookup(key_klass, key_selector)` and
        // assumed that result (S13's sole dependency).
        let own_key_affected = nm.key_selector.oop().raw() == sel_raw
            && superchain_contains(&vm.universe, nm.key_klass, holder);
        // S14 step 4b: any INLINED leaf's dependency. The guard assumed
        // `lookup(dep_klass, dep_sel)` resolved to the spliced callee, so
        // installing `dep_sel` anywhere on `dep_klass`'s superchain — `holder`
        // ON that chain — could change that result and must invalidate. Same
        // selector-match + walk-direction rule as the own key (D2), applied per
        // inline dependency.
        let inline_dep_affected = nm.inline_deps.iter().any(|(dk, ds)| {
            ds.oop().raw() == sel_raw && superchain_contains(&vm.universe, *dk, holder)
        });
        if own_key_affected || inline_dep_affected {
            victims.push(nm.id);
        }
    }
    victims
}

/// Whether `holder` appears on `k`'s superclass chain, `k` itself first, up to
/// the `nil` terminator. Mirrors [`crate::runtime::lookup`]'s own chain walk
/// exactly (same `nil_obj` sentinel, same "the superclass field is always a
/// klass" invariant), because it answers the same question the lookup does:
/// "does a method installed at `holder` sit on the path `lookup(k, ·)` takes?"
fn superchain_contains(universe: &Universe, k: KlassOop, holder: KlassOop) -> bool {
    let nil = universe.nil_obj.raw();
    let target = holder.oop().raw();
    let mut cur = k;
    loop {
        if cur.oop().raw() == target {
            return true;
        }
        let sc = cur.superclass();
        if sc.raw() == nil {
            return false;
        }
        cur = KlassOop::try_from(sc).expect("deps: superclass field is not a klass");
    }
}

/// D1. Method-redefinition / dependency invalidation — the install-time hook.
/// Collects the affected nmethods (a read-only scan whose borrow of `vm` ends
/// before any mutation), then makes each `NotEntrant` and redirects its
/// in-flight frames via [`crate::codecache::flush::make_not_entrant`].
///
/// Allocates nothing and runs no Smalltalk code (doc §D1): every victim is
/// handled by pure stack/code surgery, so this is legal at the tail of
/// `install_method`, after the dictionary insert's own allocation and handle
/// scope are done. Must NOT run during GC; installation is a primitive and GC
/// never installs methods, so that holds by construction.
///
/// A **first** definition of `(holder, selector)` finds no prior nmethod for
/// that key and no-ops; only a genuine *re*definition — or a definition on a
/// superclass of an already-compiled subclass method (the D2 subclass rule) —
/// yields victims. Each victim is distinct (one entry per nmethod from
/// `iter_alive`) and `Alive` at scan time, so no `make_not_entrant`
/// double-invalidation can occur inside the loop.
pub fn invalidate_dependents(vm: &mut VmState, holder: KlassOop, selector: SymbolOop) {
    let victims = affected_by_install(vm, holder, selector);
    for id in victims {
        crate::codecache::flush::make_not_entrant(vm, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecache::nmethod::{NmState, Nmethod};
    use crate::codecache::CodeHandle;
    use crate::compiler::assembler::CodeBlob;
    use crate::oops::wrappers::{KlassOop, SymbolOop};
    use crate::runtime::vm_state::{VmOptions, VmState};

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

    /// A tiny published blob so a fake nmethod owns a real (never-executed)
    /// code range — `affected_by_install` only reads `key_klass`/
    /// `key_selector`/`state`, but `Nmethod` still needs a valid `CodeHandle`.
    fn install_blob(vm: &mut VmState, len: usize) -> CodeHandle {
        let h = vm.code_cache.alloc(len).expect("code cache alloc");
        let blob = CodeBlob {
            code: vec![0u8; len],
            literal_off: len as u32,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        h
    }

    fn fake_nmethod(key_klass: KlassOop, key_selector: SymbolOop, code: CodeHandle) -> Nmethod {
        Nmethod {
            osr_cold_sends: 0,
            id: NmethodId(0),
            key_klass,
            key_selector,
            code,
            entry_off: 0,
            verified_entry_off: 0,
            state: NmState::Alive,
            level: 1,
            version: 0,
            trap_count: 0,
            profile_hash: 0,
            literal_off: 0,
            relocs: Vec::new(),
            frame_slots: 0,
            slot_is_oop: Vec::new(),
            pcdescs: Vec::new(),
            oopmaps: Vec::new(),
            ic_sites: Vec::new(),
            poll_bci: None,
            prim_call_argc_plus_recv: None,
            block_method: None,
            deopt_scopes: Vec::new(),
            deopt_pcdescs: Vec::new(),
            inline_deps: Vec::new(),
            self_devirt: false,
            osr_map: None,
        }
    }

    /// D2's chain walk over the real genesis hierarchy
    /// `SmallInt → Integer → Number → Magnitude → Object → nil`.
    #[test]
    fn superchain_walk_matches_lookup_direction() {
        let vm = test_vm();
        let u = &vm.universe;
        // self, parent, and every ancestor are ON the chain...
        assert!(superchain_contains(u, u.smi_klass, u.smi_klass));
        assert!(superchain_contains(u, u.smi_klass, u.integer_klass));
        assert!(superchain_contains(u, u.smi_klass, u.number_klass));
        assert!(superchain_contains(u, u.smi_klass, u.object_klass));
        // ...but a *descendant* is NOT (Object's chain never reaches SmallInt),
        // and an unrelated branch (Boolean) never appears.
        assert!(!superchain_contains(u, u.object_klass, u.smi_klass));
        assert!(!superchain_contains(u, u.smi_klass, u.boolean_klass));
        assert!(!superchain_contains(u, u.smi_klass, u.double_klass));
    }

    /// `affected_by_install` selects exactly the nmethods whose own key is
    /// invalidated: selector must match AND the install klass must be `key_klass`
    /// or one of its ancestors (the subclass rule — redefining `Integer>>#foo`
    /// invalidates a compiled `SmallInt>>#foo`).
    #[test]
    fn affected_by_install_applies_selector_and_subclass_rules() {
        let mut vm = test_vm();
        let foo = vm.universe.intern(b"foo");
        let bar = vm.universe.intern(b"bar");
        let smi = vm.universe.smi_klass;

        let code = install_blob(&mut vm, 16);
        let id = vm.code_table.install(fake_nmethod(smi, foo, code));

        // Exact key, ancestor keys → hit; each returns exactly the one nmethod.
        for holder in [
            vm.universe.smi_klass,
            vm.universe.integer_klass,
            vm.universe.object_klass,
        ] {
            assert_eq!(
                affected_by_install(&vm, holder, foo),
                vec![id],
                "installing #foo at {holder:?} must invalidate SmallInt>>#foo",
            );
        }
        // Wrong selector → miss even at the exact klass.
        assert!(affected_by_install(&vm, smi, bar).is_empty());
        // Right selector but an unrelated / descendant klass → miss.
        assert!(affected_by_install(&vm, vm.universe.boolean_klass, foo).is_empty());
        assert!(affected_by_install(&vm, vm.universe.double_klass, foo).is_empty());
    }
}
