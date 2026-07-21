//! `MegaTable` (S11 D4.4) — one shared megamorphic trampoline per
//! selector, `mega_<sel>`. Unlike `pics::PicTable` (1:1 with a specific
//! call site), this is shared across every site that goes megamorphic for
//! the SAME selector — mirrors `adapters::AdapterTable`'s own "keyed for
//! reuse" shape, keyed by selector instead of method.

use std::collections::HashMap;

use crate::compiler::assembler::{x, xr, Assembler, CodeBlob, RelocKind};
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::oops::wrappers::SymbolOop;

use super::guard::JitWriteGuard;
use super::{CodeCache, CodeHandle};

struct CachedMega {
    handle: CodeHandle,
    selector_pool_off: u32,
}

#[derive(Default)]
pub struct MegaTable {
    by_selector: HashMap<u64, CachedMega>,
}

impl MegaTable {
    pub fn new() -> MegaTable {
        MegaTable::default()
    }

    /// Returns `selector`'s own `mega_<sel>` trampoline handle, building
    /// and caching one on first request. Returns the full `CodeHandle`
    /// (not just its address) since `IcState::Mega{stub}` needs one.
    /// `None` when the code cache cannot fit the trampoline (same
    /// degrade-don't-abort contract as `PicTable::build` — the caller
    /// leaves the send site unpatched and every miss keeps resolving
    /// through `rt_resolve_send`: slow, correct, and self-healing if
    /// cache space ever frees up).
    pub fn get_or_make(
        &mut self,
        cache: &mut CodeCache,
        stub_mega_shared_addr: u64,
        selector: SymbolOop,
    ) -> Option<CodeHandle> {
        let key = selector.oop().raw();
        if let Some(m) = self.by_selector.get(&key) {
            return Some(m.handle);
        }
        #[cfg(target_arch = "aarch64")]
        let blob = build_mega_trampoline(selector, stub_mega_shared_addr);
        #[cfg(not(target_arch = "aarch64"))]
        let blob = crate::codecache::thunks_x64::build_mega_trampoline_x64(
            selector.oop().raw(),
            stub_mega_shared_addr,
        );
        let selector_pool_off = blob.literal_off; // selector is the first literal interned
        let h = cache.alloc(blob.code.len())?;
        cache.publish(h, &blob);
        self.by_selector.insert(
            key,
            CachedMega {
                handle: h,
                selector_pool_off,
            },
        );
        Some(h)
    }

    /// D8-adjacent (pre-S12 bridge): visits every cached trampoline's own
    /// embedded selector pool word.
    pub fn oops_do(&mut self, f: &mut dyn FnMut(&mut u64)) {
        let mut guard = JitWriteGuard::new();
        for m in self.by_selector.values() {
            guard.note(m.handle.base, m.handle.len);
            debug_assert!(
                (m.selector_pool_off as usize) + 8 <= m.handle.len,
                "oops_do: selector_pool_off {} + 8 exceeds this trampoline's own length {}",
                m.selector_pool_off,
                m.handle.len
            );
            // SAFETY: every `CachedMega` came from `cache.alloc` +
            // `cache.publish` in `get_or_make` above, with
            // `selector_pool_off` read from that SAME blob's own
            // `literal_off` — live MAP_JIT memory, 8-byte aligned (D3.3),
            // guarded (this function's own `guard`, noted for this exact
            // range).
            let addr = unsafe { m.handle.base.add(m.selector_pool_off as usize) } as *mut u64;
            unsafe { f(&mut *addr) };
        }
    }

    /// DBG0 (docs/DEBUGGER.md §4.2 step 1): does `pc` fall inside any
    /// cached mega trampoline's code range? Crash-dossier verdict only —
    /// linear scan, never hot.
    pub fn contains_pc(&self, pc: u64) -> bool {
        self.by_selector.values().any(|m| {
            let base = m.handle.base as u64;
            pc >= base && pc < base + m.handle.len as u64
        })
    }

    /// D4.4's own explicit ask ("rehashed after full GC like CodeTable")
    /// — unlike `AdapterTable`'s deferred key staleness, this one just
    /// reads each entry's OWN pool word back (already kept current by
    /// `oops_do`, called earlier in the same full-GC pass) rather than
    /// needing a separate `forward_chase` pass of its own.
    pub fn rehash(&mut self) {
        let entries: Vec<(u64, CachedMega)> = self.by_selector.drain().collect();
        for (_, m) in entries {
            // SAFETY: same as `oops_do` above — `selector_pool_off` is
            // this entry's own, already-relocated pool word.
            let addr = unsafe { m.handle.base.add(m.selector_pool_off as usize) } as *const u64;
            let new_key = unsafe { *addr };
            self.by_selector.insert(new_key, m);
        }
    }
}

/// D4.4: `mega_<sel>` — loads `selector`'s own oop into x16 (the register
/// `stub_mega_shared` expects it in on arrival, D4.4's own convention),
/// `c2i_shared`'s address into x17 (an indirect `br` rather than D4.4's
/// own literal `b stub_mega_shared` — same Branch26-sidestepping tradeoff
/// as `adapters::build_c2i_adapter`'s own jump to `c2i_shared`, and for
/// the identical reason: x16 is already spoken for here, carrying the
/// selector THROUGH the jump rather than the jump target itself, so the
/// two adapters' scratch-register roles are simply swapped).
fn build_mega_trampoline(selector: SymbolOop, stub_mega_shared_addr: u64) -> CodeBlob {
    let mut a = JasmAssembler::new();
    let sel_lit = a.literal_u64(selector.oop().raw(), Some(RelocKind::Oop));
    let shared_lit = a.literal_u64(stub_mega_shared_addr, Some(RelocKind::RuntimeAddr));
    a.ldr_literal(xr(16), sel_lit);
    a.ldr_literal(xr(17), shared_lit);
    a.emit("br", &[x(17)]);
    a.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn trampoline_listing_shape() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo:");
        let blob = build_mega_trampoline(sel, 0xDEAD_0000);
        let mnemonics: Vec<&str> = blob
            .listing
            .iter()
            .map(|l| l.split_whitespace().nth(2).unwrap_or(""))
            .collect();
        assert_eq!(
            mnemonics,
            vec!["ldr", "ldr", "br"],
            "mega_<sel> must be exactly load selector, load stub_mega_shared, br -- got:\n{}",
            blob.listing.join("\n")
        );
        assert_eq!(blob.relocs.len(), 2);
        assert!(matches!(blob.relocs[0].kind, RelocKind::Oop));
        assert!(matches!(blob.relocs[1].kind, RelocKind::RuntimeAddr));
    }

    #[test]
    fn get_or_make_caches_by_selector() {
        let mut vm = test_vm();
        let mut cache = CodeCache::new(1 << 20).unwrap();
        let mut mega = MegaTable::new();
        let s1 = vm.universe.intern(b"foo:");
        let s2 = vm.universe.intern(b"bar:");

        let a1 = mega.get_or_make(&mut cache, 0xC0FFEE, s1);
        let a1_again = mega.get_or_make(&mut cache, 0xC0FFEE, s1);
        let a2 = mega.get_or_make(&mut cache, 0xC0FFEE, s2);

        assert_eq!(
            a1, a1_again,
            "same selector must return the SAME trampoline"
        );
        assert_ne!(a1, a2, "different selectors must get distinct trampolines");
    }

    /// `oops_do` is wired into BOTH `memory::scavenge` and `memory::
    /// fullgc` (mirroring `AdapterTable`'s own pool-word treatment,
    /// load-bearing on either path since a fresh selector -- like a fresh
    /// method -- can be young). Real evidence, not a hand-fed closure:
    /// exercises the actual `memory::scavenge::scavenge` wiring.
    #[test]
    fn oops_do_relocates_selector_word_across_real_scavenge() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo:"); // fresh -- young, in Eden
        let old_bits = sel.oop().raw();
        let handle = vm
            .mega_table
            .get_or_make(&mut vm.code_cache, 0xC0FFEE, sel)
            .expect("test cache has room");

        crate::memory::scavenge::scavenge(&mut vm).expect("scavenge must succeed");

        let off = vm
            .mega_table
            .by_selector
            .values()
            .next()
            .unwrap()
            .selector_pool_off;
        let new_bits = unsafe { *(handle.base.add(off as usize) as *const u64) };
        assert_ne!(
            new_bits, old_bits,
            "a fresh young selector must actually move across a real scavenge"
        );
        let moved = SymbolOop::try_from(crate::oops::Oop::from_raw(new_bits))
            .expect("the relocated pool word must still read back as a genuine SymbolOop");
        assert_eq!(moved.oop().raw(), new_bits);
    }

    /// `rehash` -- D4.4's own explicit ask -- is wired directly into
    /// `memory::fullgc`'s own phase C (unlike `AdapterTable`, which defers
    /// its own key-staleness gap): `full_gc` alone, with no separate
    /// manual `rehash()` call, must already leave `by_selector` keyed by
    /// the POST-gc selector bits. Mirrors `CodeTable`'s own precedent
    /// (`nmethod.rs`'s `full_gc_updates_pool_word_and_key_klass`), which
    /// checks the same way: call `full_gc`, then check the result
    /// directly, no manual follow-up step.
    #[test]
    fn rehash_re_keys_by_selector_across_real_full_gc() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"bar:");
        let old_bits = sel.oop().raw();
        vm.mega_table
            .get_or_make(&mut vm.code_cache, 0xC0FFEE, sel)
            .expect("test cache has room");

        crate::memory::fullgc::full_gc(&mut vm).expect("full gc must succeed");

        assert!(
            !vm.mega_table.by_selector.contains_key(&old_bits),
            "the old key must miss -- full_gc's own phase C already rehashed"
        );
        // Looked up with a FRESH re-intern (`nmethod.rs`'s own established
        // pattern), not the pre-GC `sel` local -- that local is a plain
        // Rust Copy value the GC has no way to reach and update in place.
        let new_bits = vm.universe.intern(b"bar:").oop().raw();
        assert!(
            vm.mega_table.by_selector.contains_key(&new_bits),
            "the new (post-full-gc) key must hit"
        );
    }
}
