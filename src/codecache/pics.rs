//! `PicTable` (S11 D4.3) — per-call-site polymorphic inline caches. Unlike
//! `adapters::AdapterTable`/`mega::MegaTable` (keyed for reuse across many
//! call sites), a PIC is 1:1 with the specific call site that outgrew
//! mono — keyed by its own stub handle so `stubs::rt_resolve_send`'s
//! rebuild-on-grow path can find "the pairs I was built with" without
//! re-parsing published machine code.

use std::collections::HashMap;

use crate::compiler::assembler::{imm, mem, x, xr, Assembler, CodeBlob, Cond, RelocKind};
use crate::compiler::jasm_assembler::JasmAssembler;
use crate::oops::wrappers::KlassOop;
use crate::oops::Oop;

use super::guard::JitWriteGuard;
use super::{CodeCache, CodeHandle};

/// SPEC §4.3's tunable, shared with the interpreter's own PICs — the 5th
/// distinct klass at a site promotes straight to megamorphic rather than
/// growing a 5th time (D4.1's own state table).
pub const PIC_MAX_ENTRIES: usize = 4;

struct PicDesc {
    handle: CodeHandle,
    pairs: Vec<(KlassOop, u64)>,
    /// Byte offsets of each pair's own klass pool word, parallel to
    /// `pairs` — what `oops_do` walks.
    klass_pool_offs: Vec<u32>,
}

#[derive(Default)]
pub struct PicTable {
    by_handle: HashMap<u64, PicDesc>,
}

impl PicTable {
    pub fn new() -> PicTable {
        PicTable::default()
    }

    /// Builds and publishes a fresh PIC stub for `pairs`, returns its own
    /// entry address (a PIC has no separate verified-entry — its whole
    /// body IS the guard). Not registered under its OWNING call site: S12
    /// D6.2's own flush sweep turned out not to need that (it iterates
    /// every alive nmethod's own `ic_sites` directly, already getting the
    /// `(nmethod, offset)` pair for free from the site itself, then
    /// consults [`Self::pairs_of`] only for sites already known to be
    /// `IcState::Pic`) — an earlier S11 draft of this struct carried a
    /// `site: (NmethodId, u32)` field for exactly that future pass, never
    /// read by anything until now; removed once this pass confirmed it
    /// really doesn't need it, rather than leaving it as permanent
    /// speculative dead code.
    /// `None` when the code cache cannot fit the stub (S24 A1 hardening:
    /// deltablue at threshold=1 under DEOPT_STRESS legitimately filled the
    /// old 1 MiB cache and ABORTED here — a PIC is an optimization, so
    /// exhaustion must degrade, never panic; `rt_resolve_send` falls back
    /// down its dispatch ladder, ultimately leaving the site unpatched).
    pub fn build(
        &mut self,
        cache: &mut CodeCache,
        smi_klass_bits: u64,
        resolve_addr: u64,
        pairs: Vec<(KlassOop, u64)>,
    ) -> Option<CodeHandle> {
        debug_assert!(
            !pairs.is_empty() && pairs.len() <= PIC_MAX_ENTRIES,
            "PicTable::build: {} pairs (must be 1..={PIC_MAX_ENTRIES})",
            pairs.len()
        );
        // No duplicate klasses: besides being unreachable dispatch (the
        // guard chain takes the first match), a duplicate collapses to ONE
        // interned pool word listed TWICE in `klass_pool_offs`, and the GC
        // root walk visiting the same slot twice per cycle trips
        // `scavenge_oop`'s to-space double-copy guard. `rt_resolve_send`'s
        // re-key arm maintains this; this assert catches any future
        // duplicate source at BUILD time instead of one scavenge later.
        debug_assert!(
            {
                let mut ks: Vec<u64> = pairs.iter().map(|p| p.0.oop().raw()).collect();
                ks.sort_unstable();
                ks.windows(2).all(|w| w[0] != w[1])
            },
            "PicTable::build: duplicate klass in PIC pairs"
        );
        #[cfg(target_arch = "aarch64")]
        let (blob, klass_pool_offs) = build_pic_stub(&pairs, smi_klass_bits, resolve_addr);
        #[cfg(not(target_arch = "aarch64"))]
        let (blob, klass_pool_offs) = crate::codecache::pics_x64::build_pic_stub_x64(
            &pairs,
            smi_klass_bits,
            resolve_addr,
        );
        let h = cache.alloc(blob.code.len())?;
        cache.publish(h, &blob);
        self.by_handle.insert(
            h.base as u64,
            PicDesc {
                handle: h,
                pairs,
                klass_pool_offs,
            },
        );
        Some(h)
    }

    /// The pairs a currently-live PIC (found by its own stub handle) was
    /// built with — `rt_resolve_send`'s own `IcState::Pic` arm reads this
    /// to seed the next rebuild with one more pair.
    pub fn pairs_of(&self, stub: CodeHandle) -> &[(KlassOop, u64)] {
        &self
            .by_handle
            .get(&(stub.base as u64))
            .expect(
                "PicTable::pairs_of: stub not registered -- IcState::Pic named a handle this \
                 table never built",
            )
            .pairs
    }

    /// Frees `stub`'s own code-cache space and drops its bookkeeping —
    /// called on both a grow (replaced by a bigger PIC) and a promotion
    /// to megamorphic (replaced by a mega stub).
    pub fn free(&mut self, cache: &mut CodeCache, stub: CodeHandle) {
        cache.free(stub);
        self.by_handle.remove(&(stub.base as u64));
    }

    /// DBG0 (docs/DEBUGGER.md §4.2 step 1): does `pc` fall inside any live
    /// PIC stub's code range? Returns the stub's own entry count when so —
    /// enough for a crash dossier's verdict line to say "PIC stub, N
    /// entries" instead of "in-cache, unnamed". Linear over live PICs:
    /// crash-path-only, never hot.
    pub fn contains_pc(&self, pc: u64) -> Option<usize> {
        self.by_handle.values().find_map(|d| {
            let base = d.handle.base as u64;
            (pc >= base && pc < base + d.handle.len as u64).then_some(d.pairs.len())
        })
    }

    /// D8-adjacent (pre-S12 bridge): visits every live PIC's own embedded
    /// klass pool words — load-bearing, same reasoning as
    /// `adapters::AdapterTable::oops_do` (a receiver klass compared
    /// against can be young).
    pub fn oops_do(&mut self, f: &mut dyn FnMut(&mut u64)) {
        // DBGPIC overlap check: two live PICs must never own overlapping
        // code-cache ranges (a double-free / freelist-corruption symptom
        // would surface as the same pool word scanned twice -> the
        // scavenge to-space guard).
        #[cfg(debug_assertions)]
        {
            let mut ranges: Vec<(usize, usize)> = self
                .by_handle
                .values()
                .map(|d| (d.handle.base as usize, d.handle.len))
                .collect();
            ranges.sort_unstable();
            for w in ranges.windows(2) {
                assert!(
                    w[0].0 + w[0].1 <= w[1].0,
                    "PicTable: OVERLAPPING live PIC stubs [{:#x}+{}] and [{:#x}+{}]",
                    w[0].0,
                    w[0].1,
                    w[1].0,
                    w[1].1
                );
            }
        }
        let mut guard = JitWriteGuard::new();
        for d in self.by_handle.values() {
            guard.note(d.handle.base, d.handle.len);
            for &off in &d.klass_pool_offs {
                debug_assert!(
                    (off as usize) + 8 <= d.handle.len,
                    "oops_do: klass_pool_off {off} + 8 exceeds this PIC's own length {}",
                    d.handle.len
                );
                // SAFETY: `off` came from this SAME blob's own
                // `build_pic_stub` return value — live MAP_JIT memory,
                // 8-byte aligned (D3.3), guarded (this function's own
                // `guard`, noted for this exact range).
                let addr = unsafe { d.handle.base.add(off as usize) } as *mut u64;
                unsafe { f(&mut *addr) };
            }
        }
    }

    /// DBGPIC: pre-scan every registered PIC's klass pool words against
    /// the given to-space range BEFORE the collector visits them, and
    /// report any word that is ALREADY a to-space address — the
    /// double-copy panic's input, but with the owning PIC identified.
    /// Debug-diagnosis only.
    pub fn scan_report(&self, to_start: usize, to_top: usize) {
        for d in self.by_handle.values() {
            for (i, &off) in d.klass_pool_offs.iter().enumerate() {
                let addr = unsafe { d.handle.base.add(off as usize) } as *const u64;
                let word = unsafe { std::ptr::read(addr) };
                let a = word as usize & !0x7;
                if a >= to_start && a < to_top {
                    eprintln!(
                        "DBGPIC PRE-SCAN ANOMALY pic_base={:#x} len={} slot={i} off={off} word={word:#x} pairs_klass={:#x} npairs={}",
                        d.handle.base as usize,
                        d.handle.len,
                        d.pairs.get(i).map(|p| p.0.oop().raw()).unwrap_or(0),
                        d.pairs.len()
                    );
                }
            }
        }
    }

    /// Full-GC-only: `pairs`' own `KlassOop` values are a Rust-side
    /// mirror of what `oops_do` already keeps the MACHINE CODE pool words
    /// current for — same "not MAP_JIT memory, no guard needed" treatment
    /// as `nmethod::CodeTable::update_keys`'s own `key_klass`.
    pub fn update_keys(&mut self, f: &mut dyn FnMut(Oop) -> Oop) {
        for d in self.by_handle.values_mut() {
            for pair in &mut d.pairs {
                let nk = f(pair.0.oop());
                // SAFETY: a collector transform never changes an oop's
                // shape, only (at most) its address — same reasoning as
                // `CodeTable::update_keys`'s own `key_klass`/
                // `key_selector`.
                pair.0 = unsafe { KlassOop::from_oop_unchecked(nk) };
            }
        }
    }
}

/// D4.3: the guard-chain sequence. The klass-load prefix mirrors
/// `compiler::emit::emit_entry_guard`'s own 5-word sequence (can't call it
/// directly — private, and shaped for exactly one klass with a
/// fall-through match rather than N klasses each with their own target),
/// then one guarded indirect tail-call per pair.
///
/// Deviates from D4.3's own literal pseudocode the SAME way S11 steps 2/4
/// already did for similar cases: each pair's own match reaches its
/// target via an indirect `ldr x16,<pool:t_i>; br x16` rather than a
/// direct Branch26 `b t_i` — avoids needing a NEW "emit placeholder now,
/// patch immediately after publish" mechanism just for this one case (the
/// nmethod entry guard's own miss path and the c2i adapter's own jump to
/// `c2i_shared` both already chose the same tradeoff). Costs one extra
/// pool word and one extra code word per entry versus the doc's own
/// "4 words + 1 pool word" budget — immaterial at n≤4.
///
/// Returns the built blob plus each pair's own klass pool word's byte
/// offset (parallel to `pairs`), for `PicTable::build` to hand to
/// `oops_do`.
fn build_pic_stub(
    pairs: &[(KlassOop, u64)],
    smi_klass_bits: u64,
    resolve_addr: u64,
) -> (CodeBlob, Vec<u32>) {
    let mut a = JasmAssembler::new();
    let smi_lit = a.literal_u64(smi_klass_bits, Some(RelocKind::Oop));

    let smi_case = a.new_label();
    let after_klass_load = a.new_label();

    a.emit("tst", &[x(0), imm(3)]);
    a.b_cond(Cond::Eq, smi_case);
    // Heap case: klass word at untagged-address + KLASS_OFFSET(8),
    // MEM_TAG(1)-biased — same sequence as `emit_entry_guard`.
    a.emit("ldur", &[x(17), mem(0, 7)]);
    a.b(after_klass_load);
    a.bind(smi_case);
    a.ldr_literal(xr(17), smi_lit);
    a.bind(after_klass_load);

    let mut klass_lits = Vec::with_capacity(pairs.len());
    for &(k, t) in pairs {
        let next = a.new_label();
        let k_lit = a.literal_u64(k.oop().raw(), Some(RelocKind::Oop));
        klass_lits.push(k_lit);
        let t_lit = a.literal_u64(t, Some(RelocKind::RuntimeAddr));
        a.ldr_literal(xr(16), k_lit);
        a.emit("cmp", &[x(17), x(16)]);
        a.b_cond(Cond::Ne, next);
        a.ldr_literal(xr(16), t_lit);
        a.emit("br", &[x(16)]);
        a.bind(next);
    }
    // All miss: the SAME `stub_resolve`/`stub_ic_miss` door the nmethod
    // entry guard's own miss path reaches (D4.1: "one shared stub, two
    // doors") — x30 is still the ORIGINAL send site's own return address,
    // untouched by anything above.
    let resolve_lit = a.literal_u64(resolve_addr, Some(RelocKind::RuntimeAddr));
    a.ldr_literal(xr(16), resolve_lit);
    a.emit("br", &[x(16)]);

    let blob = a.finish();
    let klass_pool_offs = klass_lits
        .iter()
        .map(|l| blob.literal_off + 8 * l.0)
        .collect();
    (blob, klass_pool_offs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oops::layout::MEM_TAG;
    use crate::oops::Format;
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

    fn new_klass(vm: &mut VmState, name: &str) -> KlassOop {
        let object_klass = vm.universe.object_klass;
        vm.universe.new_klass(
            object_klass,
            name,
            Format::Slots,
            false,
            crate::oops::layout::HEADER_WORDS,
        )
    }

    #[test]
    fn build_two_entry_pic_listing_shape() {
        let mut vm = test_vm();
        let k1 = new_klass(&mut vm, "PicA");
        let k2 = new_klass(&mut vm, "PicB");
        let (blob, offs) = build_pic_stub(&[(k1, 0x1000), (k2, 0x2000)], 0x9000, 0xDEAD_0000);
        assert_eq!(offs.len(), 2, "one klass pool offset per pair");

        let mnemonics: Vec<&str> = blob
            .listing
            .iter()
            .map(|l| l.split_whitespace().nth(2).unwrap_or(""))
            .collect();
        assert_eq!(mnemonics.first(), Some(&"tst"), "opens with the tag check");
        assert_eq!(
            mnemonics.iter().filter(|&&m| m == "br").count(),
            3,
            "one br per pair's own match (2) plus the final all-miss door -- got:\n{}",
            blob.listing.join("\n")
        );
        assert!(
            !mnemonics.contains(&"bl") && !mnemonics.contains(&"blr"),
            "a PIC must never touch x30 -- got:\n{}",
            blob.listing.join("\n")
        );
    }

    #[test]
    fn pic_build_degrades_to_none_on_cache_exhaustion() {
        // S24 A1 hardening regression: a full code cache must yield
        // `None` (rt_resolve_send leaves the site unpatched), never
        // panic — the old `expect` here ABORTED deltablue at threshold=1
        // under DEOPT_STRESS once blocks joined the compile population.
        let mut vm = test_vm();
        // Smallest legal cache: page-granular, and exhausted by design —
        // fill it with one dummy alloc spanning (almost) everything.
        // (`CodeCache::new` rounds up to page granularity — 16 KiB on
        // Apple Silicon — so "fill it" must mean: alloc until refusal.)
        let mut cache = CodeCache::new(1 << 12).unwrap();
        while cache.alloc(32).is_some() {}
        let mut pics = PicTable::new();
        let k1 = new_klass(&mut vm, "PicX");
        let k2 = new_klass(&mut vm, "PicY");
        let got = pics.build(
            &mut cache,
            0x9000,
            0xDEAD_0000,
            vec![(k1, 0x1000u64), (k2, 0x2000u64)],
        );
        assert!(
            got.is_none(),
            "exhausted cache must decline the PIC, not panic"
        );
    }

    #[test]
    fn pic_table_build_pairs_of_free_round_trip() {
        let mut vm = test_vm();
        let mut cache = CodeCache::new(1 << 20).unwrap();
        let mut pics = PicTable::new();
        let k1 = new_klass(&mut vm, "PicC");
        let k2 = new_klass(&mut vm, "PicD");
        let pairs = vec![(k1, 0x1000u64), (k2, 0x2000u64)];

        let h = pics
            .build(&mut cache, 0x9000, 0xDEAD_0000, pairs.clone())
            .expect("test cache has room");
        assert_eq!(pics.pairs_of(h), pairs.as_slice());
        assert!(cache.contains(h.base as u64));

        pics.free(&mut cache, h);
        // `CodeCache::contains` only checks the mmap'd RANGE (never
        // shrinks), so it stays true regardless of free/publish state --
        // what `PicTable::free` is actually responsible for is dropping
        // ITS OWN bookkeeping, and returning the space to `cache`'s own
        // freelist for reuse (checked here by immediately re-allocating
        // exactly `h`'s own size and confirming it lands at `h`'s own
        // base -- `CodeCache::free`'s own doc: "always reclaims exactly
        // what was carved out").
        let h2 = cache.alloc(h.len).expect("freed space must be reusable");
        assert_eq!(
            h2.base, h.base,
            "the freed space must be exactly what's reused"
        );
    }

    #[test]
    fn oops_do_relocates_pic_klass_words() {
        let mut vm = test_vm();
        let mut cache = CodeCache::new(1 << 20).unwrap();
        let mut pics = PicTable::new();
        let k1 = new_klass(&mut vm, "PicE");
        let old_bits = k1.oop().raw();

        let h = pics
            .build(&mut cache, 0x9000, 0xDEAD_0000, vec![(k1, 0x1000)])
            .expect("test cache has room");

        let new_bits = old_bits ^ 0x1000;
        pics.oops_do(&mut |w| {
            if *w == old_bits {
                *w = new_bits;
            }
        });

        let pool_addr = pics
            .by_handle
            .get(&(h.base as u64))
            .unwrap()
            .klass_pool_offs[0];
        let read_back = unsafe { *(h.base.add(pool_addr as usize) as *const u64) };
        assert_eq!(read_back, new_bits);
    }

    #[test]
    fn fake_klass_tag_sanity() {
        // Guards against accidentally testing against an address that
        // isn't even mem-tagged -- would make every assertion above
        // vacuous.
        assert_eq!(MEM_TAG, 1);
    }
}
