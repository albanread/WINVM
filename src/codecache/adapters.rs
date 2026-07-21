//! `AdapterTable` (S11 D6.1) — per-`MethodOop` c2i adapter cache, mirroring
//! `nmethod::CodeTable`'s own "Rust-side table of published blobs, keyed
//! by an oop's raw bits" shape. [`AdapterTable::oops_do`] keeps each
//! adapter's own embedded method-oop pool word current across a moving GC
//! (a freshly compiled method can be young — `runtime::lookup
//! ::install_method`'s own doc) — load-bearing, not deferred, unlike
//! `by_method`'s own key staleness (a cache MISS on a stale key just
//! rebuilds a redundant but correct adapter; a stale pool word would read
//! garbage as an oop). Rehashing `by_method` after a moving GC, and
//! eagerly repatching call sites already bound to an adapter once its own
//! target is later compiled for real, are both deferred to S11 step 10 —
//! D6.1's own text groups adapter GC handling with step 5's `PicTable`,
//! which doesn't exist yet either.

use std::collections::HashMap;

use crate::compiler::assembler::{x, xr, Assembler, CodeBlob, RelocKind};
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::oops::wrappers::MethodOop;

use super::guard::JitWriteGuard;
use super::{CodeCache, CodeHandle};

/// One cached adapter: its published code plus its own method-oop pool
/// word's byte offset. NOT a fixed constant — `JasmAssembler::finish` pads
/// the code to the next 8-byte boundary before laying out the pool (D3.3),
/// so the offset depends on the code length, which isn't itself a magic
/// number worth hand-deriving and risking silent drift if
/// `build_c2i_adapter`'s own instruction count ever changes.
struct CachedAdapter {
    handle: CodeHandle,
    method_pool_off: u32,
}

#[derive(Default)]
pub struct AdapterTable {
    by_method: HashMap<u64, CachedAdapter>,
}

impl AdapterTable {
    pub fn new() -> AdapterTable {
        AdapterTable::default()
    }

    /// Returns `method`'s c2i adapter's own entry address, building and
    /// caching one on first request. Callable exactly like a real
    /// nmethod's `entry` (`bl`, x0 = result on return) — `rt_resolve_send`
    /// (D4.1) doesn't need to know the difference when recording
    /// `IcState::Mono{klass, target}`.
    ///
    /// Task #93: the module doc's own "a cache MISS on a stale key just
    /// rebuilds a redundant but correct adapter" assumed a stale key could
    /// only ever produce a MISS — true only if addresses are never reused.
    /// They routinely are: `method` is an ordinary heap oop (block methods
    /// especially — created and collected constantly), so once its OLD
    /// address is freed by a GC, a LATER, wholly unrelated method can be
    /// allocated at that exact address (eden-base recycling, task #94's
    /// same mechanism) and collide with the stale key still sitting in
    /// `by_method`. That is a false HIT, not a miss: `method`'s OWN c2i
    /// request returns a DIFFERENT method's adapter — the mono site gets
    /// patched to run the wrong callee forever, with a real (valid, live)
    /// receiver each time, so no tag/plausibility check anywhere ever
    /// catches it (task #93's whole DNU/wrong-arithmetic zoo: `nm=247`'s
    /// `#not` site silently running an unrelated `#==`/`#bitAnd:`/etc.
    /// method's c2i adapter). Guarded by re-deriving the CACHED entry's
    /// identity from its own embedded pool word — kept current across
    /// every GC by `oops_do`, unlike the key — and rejecting the hit on
    /// mismatch (falls through and rebuilds, exactly the miss path; the
    /// orphaned stale entry is left in place, a harmless one-adapter leak
    /// under the same key it will be overwritten at below).
    pub fn get_or_make(
        &mut self,
        cache: &mut CodeCache,
        c2i_shared_addr: u64,
        method: MethodOop,
    ) -> u64 {
        let key = method.oop().raw();
        if let Some(a) = self.by_method.get(&key) {
            // SAFETY: `method_pool_off` was recorded as this SAME blob's own
            // `literal_off` at build time and is kept 8-byte-aligned,
            // in-bounds, and current (oops_do) for as long as the entry
            // lives — identical access pattern to `oops_do` above.
            let cached_method_bits = unsafe {
                std::ptr::read(a.handle.base.add(a.method_pool_off as usize) as *const u64)
            };
            if cached_method_bits == key {
                return a.handle.base as u64;
            }
        }
        // WINVM (Phase 3y): arch-selected, like `stubs::install` and
        // `deopt_trap::install` before it. This is the THIRD place the
        // same bug lived — the emitters were ported but only the two
        // static installs were rewired, leaving the three *dynamic* code
        // producers (adapters, mega trampolines, PICs) still building
        // AArch64 blobs at runtime on Windows.
        #[cfg(target_arch = "aarch64")]
        let blob = build_c2i_adapter(method, c2i_shared_addr);
        #[cfg(not(target_arch = "aarch64"))]
        let blob = crate::codecache::thunks_x64::build_c2i_adapter_x64(
            method.oop().raw(),
            c2i_shared_addr,
        );
        let method_pool_off = blob.literal_off; // method oop is the FIRST literal interned
        let h = cache
            .alloc(blob.code.len())
            .expect("AdapterTable::get_or_make: code cache too small for a c2i adapter");
        cache.publish(h, &blob);
        self.by_method.insert(
            key,
            CachedAdapter {
                handle: h,
                method_pool_off,
            },
        );
        h.base as u64
    }

    /// D8-adjacent (pre-S12 bridge): visits every cached adapter's own
    /// embedded method-oop pool word. Iterates `by_method`'s VALUES only —
    /// unlike `CodeTable::update_keys`/`rehash`, staleness in the KEYS
    /// (this table's own index) is the deferred, lower-stakes gap (this
    /// module's own doc); the handles (addresses) walked here are
    /// themselves unaffected by a moving GC regardless (the code cache
    /// never moves a published block).
    pub fn oops_do(&mut self, f: &mut dyn FnMut(&mut u64)) {
        let mut guard = JitWriteGuard::new();
        for a in self.by_method.values() {
            guard.note(a.handle.base, a.handle.len);
            debug_assert!(
                (a.method_pool_off as usize) + 8 <= a.handle.len,
                "oops_do: method_pool_off {} + 8 exceeds this adapter's own length {}",
                a.method_pool_off,
                a.handle.len
            );
            // SAFETY: every `CachedAdapter` in `by_method` came from
            // `cache.alloc` + `cache.publish` in `get_or_make` above, with
            // `method_pool_off` read directly from that SAME blob's own
            // `literal_off` — live MAP_JIT memory, 8-byte aligned (D3.3),
            // guarded (this function's own `guard`, noted for this exact
            // range) for the duration of the write.
            let addr = unsafe { a.handle.base.add(a.method_pool_off as usize) } as *mut u64;
            unsafe { f(&mut *addr) };
        }
    }

    /// DBG0 (docs/DEBUGGER.md §4.2 step 1): does `pc` fall inside any
    /// cached c2i adapter's code range? Crash-dossier verdict only —
    /// linear scan, never hot.
    pub fn contains_pc(&self, pc: u64) -> bool {
        self.by_method.values().any(|a| {
            let base = a.handle.base as u64;
            pc >= base && pc < base + a.handle.len as u64
        })
    }
}

/// D6.1: `c2i_<method>` — loads `method`'s own oop (`RelocKind::Oop`,
/// GC-tracked via [`AdapterTable::oops_do`]) into x17 and `c2i_shared`'s
/// address (`RelocKind::RuntimeAddr`, fixed for the VM's lifetime) into
/// x16, then reaches it via an indirect `br` — the same technique
/// `emit_entry_guard`'s own miss path and `Poll`'s far call already use,
/// rather than a direct Branch26 `b` (sidesteps range reasoning entirely)
/// — critically never touching x30, so `c2i_shared`'s own `stp x29,x30,…`
/// prologue captures the ORIGINAL compiled call site's own return address,
/// not this trampoline's.
///
/// No per-call-site argc is passed here at all — deliberately simpler
/// than D5's own `rt_interpret_call(vm, method_bits, argv, argc)`
/// signature suggests: a method's own arity is a fixed property of the
/// method oop (`MethodOop::argc()`), always the SAME value this adapter
/// was generated for (it's built once per `method`, D6.1's own "cached
/// per CompiledMethod"), so there is nothing for a separately-passed argc
/// to cross-check that isn't already tautological — unlike `IcSite.argc`
/// at a compiled call site, which genuinely is independent, authored
/// data that could in principle drift.
fn build_c2i_adapter(method: MethodOop, c2i_shared_addr: u64) -> CodeBlob {
    let mut a = JasmAssembler::new();
    let method_lit = a.literal_u64(method.oop().raw(), Some(RelocKind::Oop));
    let shared_lit = a.literal_u64(c2i_shared_addr, Some(RelocKind::RuntimeAddr));
    a.ldr_literal(xr(17), method_lit);
    a.ldr_literal(xr(16), shared_lit);
    a.emit("br", &[x(16)]);
    a.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::builder::BytecodeBuilder;
    use crate::oops::Oop;
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

    fn test_cache() -> CodeCache {
        CodeCache::new(1 << 20).unwrap()
    }

    fn trivial_method(vm: &mut VmState, name: &[u8], argc: usize) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let sel = vm.universe.intern(name);
        b.finish(vm, sel, argc, 0)
    }

    #[test]
    fn get_or_make_caches_by_method() {
        let mut vm = test_vm();
        let mut cache = test_cache();
        let mut adapters = AdapterTable::new();
        let m1 = trivial_method(&mut vm, b"foo", 0);
        let m2 = trivial_method(&mut vm, b"bar", 1);

        let a1 = adapters.get_or_make(&mut cache, 0xC0FFEE, m1);
        let a1_again = adapters.get_or_make(&mut cache, 0xC0FFEE, m1);
        let a2 = adapters.get_or_make(&mut cache, 0xC0FFEE, m2);

        assert_eq!(
            a1, a1_again,
            "same method must return the SAME cached adapter"
        );
        assert_ne!(a1, a2, "different methods must get distinct adapters");
        assert!(cache.contains(a1));
        assert!(cache.contains(a2));
    }

    /// The host's own c2i adapter blob, so a test recomputing
    /// `method_pool_off` independently matches what `get_or_make`
    /// actually built. (Reaching for the AArch64 builder here passed on
    /// macOS and compared against the wrong blob on Windows.)
    fn host_c2i_adapter(m: MethodOop, shared: u64) -> crate::compiler::assembler::CodeBlob {
        #[cfg(target_arch = "aarch64")]
        {
            build_c2i_adapter(m, shared)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            crate::codecache::thunks_x64::build_c2i_adapter_x64(m.oop().raw(), shared)
        }
    }

    #[test]
    fn adapter_listing_shape() {
        let mut vm = test_vm();
        let m = trivial_method(&mut vm, b"foo", 0);
        let blob = build_c2i_adapter(m, 0xDEAD_0000);
        let mnemonics: Vec<&str> = blob
            .listing
            .iter()
            .map(|l| l.split_whitespace().nth(2).unwrap_or(""))
            .collect();
        assert_eq!(
            mnemonics,
            vec!["ldr", "ldr", "br"],
            "c2i_<method> must be exactly load method, load c2i_shared, br -- got:\n{}",
            blob.listing.join("\n")
        );
        assert_eq!(
            blob.relocs.len(),
            2,
            "exactly two pool words: method oop + c2i_shared addr"
        );
        assert!(matches!(blob.relocs[0].kind, RelocKind::Oop));
        assert!(matches!(blob.relocs[1].kind, RelocKind::RuntimeAddr));
    }

    #[test]
    fn oops_do_relocates_adapter_method_word() {
        let mut vm = test_vm();
        let mut adapters = AdapterTable::new();
        let m = trivial_method(&mut vm, b"foo", 0);
        let old_bits = m.oop().raw();
        let addr = adapters.get_or_make(&mut vm.code_cache, 0xC0FFEE, m);

        let new_bits = old_bits ^ 0x1000; // a fake "moved" address
        adapters.oops_do(&mut |w| {
            if *w == old_bits {
                *w = new_bits;
            }
        });

        // Independently recomputed (not reaching into `AdapterTable`'s own
        // private `method_pool_off`) -- `build_c2i_adapter` is
        // deterministic, so a fresh build's own `literal_off` is exactly
        // where the cached adapter's method-oop pool word lives too.
        let method_pool_off = host_c2i_adapter(m, 0xC0FFEE).literal_off;
        let pool_addr = unsafe { (addr as *const u8).add(method_pool_off as usize) as *const u64 };
        assert_eq!(unsafe { *pool_addr }, new_bits);
    }

    /// The test above drives `oops_do` directly with a fake closure --
    /// real evidence (this project's own standing rule: verify GC claims
    /// against a REAL collection, not just a hand-fed transform) means
    /// actually calling `memory::scavenge::scavenge`, which is what wires
    /// `AdapterTable::oops_do` into a real run in the first place. `m` is
    /// deliberately never `install_method`-ed anywhere: the adapter's own
    /// pool word is the ONLY reference to it, so this also proves
    /// `oops_do` genuinely acts as a root by itself, not merely riding
    /// along after some OTHER root already copied the method.
    #[test]
    fn oops_do_relocates_adapter_method_word_across_real_scavenge() {
        let mut vm = test_vm();
        let m = trivial_method(&mut vm, b"foo", 0); // fresh -- young, in Eden
        let old_bits = m.oop().raw();
        let addr = vm.adapters.get_or_make(&mut vm.code_cache, 0xC0FFEE, m);

        crate::memory::scavenge::scavenge(&mut vm).expect("scavenge must succeed");

        let method_pool_off = host_c2i_adapter(m, 0xC0FFEE).literal_off;
        let pool_addr = unsafe { (addr as *const u8).add(method_pool_off as usize) as *const u64 };
        let new_bits = unsafe { *pool_addr };
        assert_ne!(
            new_bits, old_bits,
            "a fresh young method must actually move across a real scavenge"
        );
        let moved = MethodOop::try_from(Oop::from_raw(new_bits))
            .expect("the relocated pool word must still read back as a genuine MethodOop");
        assert_eq!(
            moved.argc(),
            0,
            "the moved method's own contents must survive intact"
        );
    }

    /// As above, but a real full GC (`memory::fullgc::full_gc`) -- the
    /// phase-C `forward_chase` pass wired into `codecache::adapters`'
    /// second `fullgc.rs` call site, exercised for real rather than only
    /// reasoned about.
    #[test]
    fn oops_do_relocates_adapter_method_word_across_real_full_gc() {
        let mut vm = test_vm();
        let m = trivial_method(&mut vm, b"bar", 1);
        let old_bits = m.oop().raw();
        let addr = vm.adapters.get_or_make(&mut vm.code_cache, 0xC0FFEE, m);

        crate::memory::fullgc::full_gc(&mut vm).expect("full gc must succeed");

        let method_pool_off = host_c2i_adapter(m, 0xC0FFEE).literal_off;
        let pool_addr = unsafe { (addr as *const u8).add(method_pool_off as usize) as *const u64 };
        let new_bits = unsafe { *pool_addr };
        assert_ne!(
            new_bits, old_bits,
            "full GC must relocate this method (young or old, it always slides)"
        );
        let moved = MethodOop::try_from(Oop::from_raw(new_bits))
            .expect("the relocated pool word must still read back as a genuine MethodOop");
        assert_eq!(
            moved.argc(),
            1,
            "the moved method's own contents must survive intact"
        );
    }
}
