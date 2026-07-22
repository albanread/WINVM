//! `Nmethod`/`CodeTable` — the tier-1 compiled-method store
//! (`sprint_s10_detail.md` D2, D8). `CodeTable::oops_do`/`update_keys` are
//! this module's half of D8's GC integration; the other half (calling
//! them at the right point in each collector, under the right transform)
//! lives in `memory::scavenge`/`memory::fullgc` themselves, matching how
//! `runtime::lookup::gc_epilogue` (the closest existing precedent for a
//! Rust-side address-keyed cache that dangles across a moving GC) is
//! called FROM the collectors rather than the collectors being taught
//! about `LookupCache` internals.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::codecache::guard::JitWriteGuard;
use crate::codecache::CodeHandle;
use crate::compiler::assembler::{Reloc, RelocKind};
use crate::oops::wrappers::{KlassOop, MemOop, MethodOop, SymbolOop};
use crate::oops::Oop;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NmethodId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NmState {
    Alive,
    /// S12/S13 use these — no S10 nmethod is ever constructed in either.
    NotEntrant,
    Zombie,
}

/// Which spill slots hold a LIVE oop at one specific safepoint (S12 D2) —
/// a packed bitmap over spill-slot indices, one bit per slot (`bit i set ⇔
/// frame slot i, at `[fp − 8·(i+1)]`, holds a live oop at the safepoint(s)
/// this map is attached to via `PcDesc.oopmap`). `SmallVec<[u64; 1]>`
/// (matching the project's existing `smallvec` use, `codecache::guard`'s
/// own precedent): `frame_slots <= 60` (S10/S11 eligibility's own frame
/// budget) means `ceil(60/64) == 1` word in every real case, so the inline
/// capacity avoids a heap allocation on the common path. Built by
/// `compiler::oopmap::build_for_position` (liveness ∩ is_oop at one exact
/// program position — NOT "every slot this method ever uses for an oop
/// anywhere", which is `Nmethod::slot_is_oop`'s job instead); multiple
/// safepoints with identical live sets share one entry in `Nmethod::
/// oopmaps` (deduplicated by content, D2).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OopMap {
    bits: SmallVec<[u64; 1]>,
}

impl OopMap {
    /// An empty map (every bit clear) — `oopmaps[0]`'s reserved slot for
    /// PcDescs that don't correspond to a real safepoint (S10's block-start
    /// descs, kept for `bci_at`'s trace-path lookup) and never consulted by
    /// the GC path (`oopmap_at` only ever reaches a REAL safepoint's own
    /// entry, D1: every compiled frame at GC time is suspended at one, by
    /// construction).
    pub fn empty() -> OopMap {
        OopMap {
            bits: SmallVec::new(),
        }
    }

    pub fn set(&mut self, slot: u16) {
        let word = slot as usize / 64;
        if word >= self.bits.len() {
            self.bits.resize(word + 1, 0);
        }
        self.bits[word] |= 1u64 << (slot as usize % 64);
    }

    pub fn is_oop(&self, slot: u16) -> bool {
        let word = slot as usize / 64;
        match self.bits.get(word) {
            Some(w) => (w >> (slot as usize % 64)) & 1 != 0,
            None => false,
        }
    }

    /// Every set bit's slot index, ascending — the GC root-scan's own
    /// iteration order (`each_code_root`, S12 D4.1); order doesn't matter
    /// for correctness, only for determinism (tests pin exact sequences).
    pub fn iter_slots(&self) -> impl Iterator<Item = u16> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, &word)| {
            (0..64u16)
                .filter(move |&b| (word >> b) & 1 != 0)
                .map(move |b| (w * 64) as u16 + b)
        })
    }
}

/// One block-start (S10, trace-path only) or call-return-address (S11+,
/// GC safepoint) position, mapped back to a bytecode index for mixed-tier
/// stack traces (D3.6) and — for the latter kind only — an `oopmap` index
/// (`Nmethod::oopmaps`) the GC root-scan reads via `Nmethod::oopmap_at`'s
/// EXACT match. A block-start desc's own `oopmap` is always `0`
/// (`OopMap::empty()`, reserved — see that constructor's own doc): it
/// exists only so `bci_at`'s nearest-below trace lookup has full pc
/// coverage, never a real safepoint's own return address (S11's send/
/// alloc-slow sites always emit at least one more instruction — the NLR
/// check, or the fast-path merge — before any new block could start, so
/// the two kinds of entry never collide in practice; even if they did,
/// `oopmap_at`'s caller only ever passes a genuine suspended-frame return
/// address, D1).
#[derive(Clone, Copy, Debug)]
pub struct PcDesc {
    pub pc_off: u32,
    pub bci: usize,
    pub oopmap: u16,
}

/// A compiled call site's own IC lattice state (S11 D3/D4) — mirrors the
/// interpreter IC's mono→poly→mega shape (SPEC §5.3) but lives entirely on
/// the Rust side (compiled code itself only ever sees "call whatever the
/// patched `bl` currently targets"). `Mono` records the klass/target pair
/// it was resolved for — D4.1's own state table needs the OLD pair on a
/// later "different klass" resolve, to seed a fresh PIC alongside the new
/// one. `Pic`/`Mega` carry only the generated stub's own handle — NOT a
/// pair count or the pairs themselves: `codecache::pics::PicTable`/
/// `codecache::mega::MegaTable` are the single source of truth for both
/// (keyed by the stub's own handle/selector respectively), so there is
/// nothing here that could drift out of sync with them. A later
/// transition (grow, or promote to mega) reads the OLD stub's own pairs
/// from there, then frees it there too (P1/P2: rebuild-and-swing, never
/// an in-place edit).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IcState {
    Unresolved,
    Mono { klass: KlassOop, target: u64 },
    Pic { stub: CodeHandle },
    Mega { stub: CodeHandle },
}

/// A compiled inline-cache call site (S11 D3): one per `Ir::CallSend` in a
/// compiled method's body. `off` is the `bl` instruction's own byte offset
/// within `Nmethod::code` (matches `Reloc::offset`'s convention for
/// `InlineCache` relocs — `bl_patchable`'s own doc). `selector`/`argc`
/// identify the send; `state` starts `Unresolved` at publish (every fresh
/// site's `bl` targets `stub_resolve`, D3) and evolves as real receivers
/// arrive (D4.1's state machine).
///
/// `super_klass` (S13 step 10d) marks a `send_super` site with its STATIC
/// holder-superclass — the klass a super send resolves against (D4.6), fixed at
/// compile time and *independent* of the actual receiver. `None` for an
/// ordinary dynamic send. It lives on the `IcSite` itself, NOT inside
/// `IcState::Mono`, precisely so it survives a state reset: a flush (D6.2) can
/// knock a super site back to `Unresolved`, and `rt_resolve_send` must STILL
/// re-resolve it super-aware rather than collapsing into a receiver-klass
/// dynamic send that reaches the subclass override `super` was meant to skip
/// (the step-8-deferred blocker). GC keeps it current in [`Self::oops_do`].
#[derive(Clone, Copy, Debug)]
pub struct IcSite {
    pub off: u32,
    pub selector: SymbolOop,
    pub argc: u8,
    pub state: IcState,
    pub super_klass: Option<KlassOop>,
}

pub struct Nmethod {
    pub id: NmethodId,
    /// Customization key — the receiver klass this nmethod was compiled
    /// for. Strong until S12 (D8's own SPEC-QUESTION: weak treatment is
    /// S12's job; S10 just needs the mechanism in place).
    pub key_klass: KlassOop,
    pub key_selector: SymbolOop,
    pub code: CodeHandle,
    /// `== verified_entry_off` until S11 gives a customized nmethod a
    /// separate unverified entry.
    pub entry_off: u32,
    pub verified_entry_off: u32,
    pub state: NmState,
    pub level: u8,
    pub version: u8,
    /// S14 step 8: uncommon traps taken through this nmethod's code — bumped by
    /// `rt_uncommon_trap` after each completed trap deopt. Crossing
    /// `recompile::UNCOMMON_TRAP_LIMIT` triggers the recompile-on-trap check.
    pub trap_count: u32,
    /// S14 step 8 (A5): `feedback::snapshot_profile` of the method at compile
    /// time — the effectiveness check compares the live profile against this.
    pub profile_hash: u64,
    pub literal_off: u32,
    /// Blob-relative, exactly as `CodeBlob` produced them — add
    /// `code.base` for an absolute address (matches S9's own convention).
    pub relocs: Vec<Reloc>,
    pub frame_slots: u16,
    /// S12 D2's independent ground truth, one bool per spill slot: could
    /// THIS slot EVER hold an oop, anywhere in the method (copied verbatim
    /// from `regalloc::RegallocResult::slot_is_oop` at compile time — slots
    /// are allocated monotonically, one per interval, never reused across
    /// intervals of different `is_oop`-ness, D3.5 point 3, so this is a
    /// per-slot, not per-safepoint, fact). `compiler::oopmap::verify`'s own
    /// cross-check against every `oopmaps[i]`'s bits — independent of
    /// `oopmaps` itself, so a builder bug that sets a bit for a slot that's
    /// never actually oop-typed is still caught, not just self-consistent
    /// by construction.
    pub slot_is_oop: Vec<bool>,
    pub pcdescs: Vec<PcDesc>,
    pub oopmaps: Vec<OopMap>,
    pub ic_sites: Vec<IcSite>,
    /// The bci of the block containing this method's `Ir::Poll`, if any
    /// (D5.3: at most one loop back-edge site matters for S10 — a method
    /// with none has no loop at all). Block-bci granularity, matching
    /// `pcdescs`' own precision — a mixed-tier stack-trace walker
    /// (`runtime::error::print_stack_trace`) has no exact native pc to
    /// work from at a poll callback (S11's `last_compiled_pc` anchor isn't
    /// wired up yet), so this is computed once at compile time from the IR
    /// directly instead, purely as Rust-side bookkeeping alongside the
    /// unchanged emitted code.
    pub poll_bci: Option<usize>,
    /// This method's own `receiver + real args` count, if it opens with a
    /// compiled primitive-call shim (`compiler::driver`'s shimmable-
    /// primitive eligibility) — `None` for a method with no primitive, or
    /// one whose primitive isn't shimmable (still `NoPermanent`, so no
    /// `Nmethod` exists to hold this field at all in that case). At most
    /// one per method, same "computed once from the method/IR, Rust-side
    /// bookkeeping only" shape as `poll_bci` right above. Read by
    /// `memory::roots::real_oop_rootspill_slots` (via the calling method's
    /// own `caller_pc` when a GC lands mid-primitive-call) to know how many
    /// of RootSpill's slots are live oops for `AdapterKind::CallPrimitive` —
    /// exactly the receiver+args the shim spilled, no more.
    pub prim_call_argc_plus_recv: Option<u8>,
    /// S24 A1 (closure compilation): `Some(B)` iff this nmethod compiles
    /// CompiledBlock B's body. Registered in [`CodeTable::by_block`] keyed
    /// on B's oop bits instead of `by_key` (its `key_klass`/`key_selector`
    /// are fillers — closure_klass and the `#aBlock` placeholder — never
    /// consulted for lookup). A Rust-side oop field with the same GC
    /// discipline as `key_selector`: ALWAYS visited by
    /// [`CodeTable::update_keys`] under both collectors, and
    /// [`CodeTable::rehash`] rebuilds `by_block` from it — which is what
    /// makes the raw-bits map safe across moving GC (#93's lesson by
    /// construction). `note_uncommon_trap` branches on this BEFORE its key
    /// lookup (adversarial-review §5); the S12 D5 weak-key sweep skips
    /// block nmethods.
    pub block_method: Option<MethodOop>,
    /// S13: the packed deopt scope-descriptor blob (`compiler::scopes`
    /// format — LEB128 scope records + safepoint records). Empty until S13
    /// step 3 records real sites from emit. The GC never scans this: every
    /// oop it references lives only as a `ConstPool` index into this
    /// nmethod's own oop pool, which `oops_do` already keeps current.
    pub deopt_scopes: Vec<u8>,
    /// S13: `code_off`-sorted deopt PcDescs indexing into `deopt_scopes`
    /// (DISTINCT from S12's oopmap `pcdescs` above — different key
    /// convention and payload). Empty until step 3.
    pub deopt_pcdescs: Vec<crate::compiler::scopes::PcDesc>,
    /// S14 step 4b: one `(receiver_klass, selector)` pair per INLINED leaf
    /// body — the assumption the guard makes (`lookup(receiver_klass,
    /// selector)` resolves to the callee actually spliced). `deps::
    /// affected_by_install` consults these so redefining an inlined callee
    /// (`install_method` of `selector` anywhere on `receiver_klass`'s
    /// superchain) makes THIS nmethod `NotEntrant`. Both oops are kept
    /// GC-current by [`CodeTable::oops_do_inline_deps`] (same treatment as the
    /// `IcSite` guard klass / `super_klass`): the pair names live heap objects
    /// a moving collector relocates. Empty for a non-inlining nmethod.
    pub inline_deps: Vec<(KlassOop, SymbolOop)>,
    /// S14 step 5: this nmethod contains guard-free SELF-send inlines baked
    /// against `key_klass` — its code is only correct for receivers of
    /// exactly that klass. Super-send linking (`resolve_super_target_entry`)
    /// must use the c2i adapter instead of `verified_entry` for such a
    /// target, because super sites enter with subclass receivers the entry
    /// guard never checks.
    pub self_devirt: bool,
    /// S15: the interpreter→compiled frame-conversion map, present iff this
    /// nmethod was compiled WITH an OSR entry (at most one in v1). The
    /// nmethod is otherwise normal — same key, normal entries, scope descs
    /// everywhere; `rt_osr_request` reads this to pack the transfer buffer.
    pub osr_map: Option<crate::compiler::scopes::OsrMap>,
    /// OSR-heal (the Cog-comparison sieve finding): how many send sites this
    /// compile lowered to a generic `CallSend` BECAUSE their feedback was
    /// `Untaken` — under an OSR compile, sites later in iteration order than
    /// the loop-counter trip point (LOOP_COUNTER_LIMIT backedges into the
    /// FIRST call) have never executed, and de-speculating them is correct
    /// (S15) but bakes in-loop calls that (a) dispatch c2i→interpreter every
    /// iteration for never-compilable smi primitives and (b) set
    /// `crosses_call` on every loop-spanning interval, killing residency.
    /// Nonzero + `osr_map` = a HALF-WARM OSR nmethod: ordinary sends decline
    /// it (c2i instead), one fully-interpreted run warms every IC, and the
    /// c2i escape hatch / activate_method replace it with a normal compile.
    /// Zero keeps the S24-L2 trigger-unification behavior untouched.
    pub osr_cold_sends: u16,
}

impl Nmethod {
    /// See [`Nmethod::osr_cold_sends`]: an OSR-entried compile that baked
    /// cold-site generic sends. Serves ONLY its OSR (loop-takeover) entry;
    /// every ordinary-dispatch path treats the method as not-compiled so a
    /// warm whole-body replacement can happen.
    pub fn is_half_warm_osr(&self) -> bool {
        self.osr_map.is_some() && self.osr_cold_sends > 0
    }

    /// Every `Reloc` this nmethod's literal pool actually needs the GC to
    /// visit (`Oop`/`KeyKlassOop`; `RuntimeAddr`/`InternalWord`/
    /// `InlineCache` are never oops).
    fn oop_relocs(&self) -> impl Iterator<Item = &Reloc> {
        self.relocs
            .iter()
            .filter(|r| matches!(r.kind, RelocKind::Oop | RelocKind::KeyKlassOop))
    }

    /// S12 D2/P1: the GC root-scan's own lookup — `ret_pc` is a
    /// COMPILED FRAME's saved return address (absolute; converted to a
    /// blob-relative offset here), i.e. exactly where a safepoint's `bl`
    /// resumes. EXACT match only, never nearest-below: D1's own invariant
    /// is that every compiled frame at GC time is suspended at a real
    /// safepoint's return address, by construction (loop polls never
    /// allocate/GC — S10 D5.6 — so nothing else can be live on the stack
    /// when a collection starts). A miss means that invariant broke
    /// somewhere upstream (a safepoint got emitted without its own
    /// `PcDesc`, or the walker mis-stepped) — panicking here, rather than
    /// silently falling back to the nearest map, is P1's whole point: a
    /// near-miss would trace the WRONG slots as oops, corrupting the heap
    /// three tests later instead of failing loudly at the source.
    pub fn oopmap_at(&self, ret_pc: u64) -> &OopMap {
        let off = (ret_pc - self.code.base as u64) as u32;
        let idx = self
            .pcdescs
            .iter()
            .position(|d| d.pc_off == off)
            .unwrap_or_else(|| {
                panic!(
                    "oopmap_at: nmethod {:?} has no safepoint PcDesc at return-address offset \
                     {off:#x} (ret_pc {ret_pc:#x}) — D1's invariant broke: every compiled frame \
                     live at GC time must be suspended at a recorded safepoint",
                    self.id
                )
            });
        &self.oopmaps[self.pcdescs[idx].oopmap as usize]
    }

    /// Trace path (`runtime::error::print_stack_trace`, mixed-tier stack
    /// traces): nearest-below lookup over ALL descs (block-start AND real
    /// safepoint alike) — any pc inside the method resolves to whichever
    /// desc most recently preceded it, never panics (a trace is best-effort
    /// diagnostics, not a GC-correctness path — P1's exact-match
    /// requirement is `oopmap_at`'s alone). `pcdescs` is sorted by
    /// `pc_off` ascending (both `pcdescs_sorted_by_pc_off` and driver.rs's
    /// own construction maintain this).
    pub fn bci_at(&self, pc: u64) -> usize {
        let off = (pc - self.code.base as u64) as u32;
        let idx = self.pcdescs.partition_point(|d| d.pc_off <= off);
        if idx == 0 {
            self.pcdescs.first().map(|d| d.bci).unwrap_or(0)
        } else {
            self.pcdescs[idx - 1].bci
        }
    }

    /// Test-only, minimal fake — shared by this module's own tests and
    /// `compiler::oopmap`'s (neither cares about `key_klass`/`key_selector`
    /// /`code` beyond their tag-level shape; both DO care about
    /// `frame_slots`/`slot_is_oop`/`oopmaps`, which this takes explicitly).
    #[cfg(test)]
    pub(crate) fn test_fake(
        frame_slots: u16,
        slot_is_oop: Vec<bool>,
        oopmaps: Vec<OopMap>,
    ) -> Nmethod {
        use crate::oops::layout::MEM_TAG;
        use crate::oops::Oop;
        Nmethod {
            osr_cold_sends: 0,
            id: NmethodId(0),
            // SAFETY: test-only, tag-level shape is never dereferenced by
            // anything `test_fake`'s own callers exercise (same reasoning
            // as this module's own private `fake_klass`/`fake_selector`).
            key_klass: unsafe { KlassOop::from_oop_unchecked(Oop::from_raw(0x1000 + MEM_TAG)) },
            key_selector: unsafe { SymbolOop::from_oop_unchecked(Oop::from_raw(0x2000 + MEM_TAG)) },
            code: CodeHandle {
                base: 0x1000 as *const u8,
                len: 0x1000,
            },
            entry_off: 0,
            verified_entry_off: 0,
            state: NmState::Alive,
            level: 1,
            version: 0,
            trap_count: 0,
            profile_hash: 0,
            literal_off: 0,
            relocs: Vec::new(),
            frame_slots,
            slot_is_oop,
            pcdescs: Vec::new(),
            oopmaps,
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
}

#[derive(Default)]
pub struct CodeTable {
    slots: Vec<Option<Nmethod>>,
    by_key: HashMap<(u64, u64), NmethodId>,
    /// S24 A1: CompiledBlock-oop bits → that block's nmethod (design §2.1,
    /// amended). Lifecycle mirrors `by_key` EXACTLY: unhooked in
    /// `set_not_entrant`/`mark_zombie` under the same successor guard, and
    /// rehashed under BOTH collectors by `update_keys`/`rehash` (the keys
    /// are always live — each block nmethod's oop pool pins its
    /// CompiledBlock, so every key has a forwarding pointer at GC time; the
    /// #93/AdapterTable raw-key lesson applied by construction).
    by_block: HashMap<u64, NmethodId>,
    /// S15: `(klass, selector, osr_bci)` → OSR-entry nmethod. Same weak /
    /// rebuild-after-GC / conditional-retirement discipline as `by_key`.
    osr_table: HashMap<(u64, u64, u16), NmethodId>,
    /// `(code base, id)`, sorted by base — `find_by_pc` binary-searches
    /// this; the code cache never moves a published block (S9/S12's own
    /// invariant), so this table is never invalidated by a GC the way
    /// `by_key` is.
    by_addr: Vec<(u64, NmethodId)>,
}

impl CodeTable {
    pub fn new() -> CodeTable {
        CodeTable::default()
    }

    /// Installs `nm`, reusing a freed slot if one exists (S12 flushing
    /// will create them; S10 never does, so this always appends for now —
    /// still correct either way, and means S12 doesn't have to touch this
    /// method at all).
    /// S24 A1: the `by_block` key for a block nmethod (`None` for methods).
    fn slots_block_key(&self, nm: &Nmethod) -> Option<u64> {
        nm.block_method.map(|b| b.oop().raw())
    }

    /// S24 A1: the Alive nmethod compiled for CompiledBlock `b`, if any.
    /// Callers require `NmState::Alive` semantics (adversarial-review
    /// BLOCKER §4): a NotEntrant block nmethod was already unhooked in
    /// `set_not_entrant`, so a hit here is Alive by construction — the
    /// state check below is belt-and-braces, same posture as
    /// `send_generic`'s own validate-before-enter.
    pub fn lookup_block(&self, b: MethodOop) -> Option<NmethodId> {
        let id = self.by_block.get(&b.oop().raw()).copied()?;
        let nm = self.slots.get(id.0 as usize)?.as_ref()?;
        if matches!(nm.state, NmState::Alive) {
            Some(id)
        } else {
            None
        }
    }

    pub fn install(&mut self, mut nm: Nmethod) -> NmethodId {
        let key = (nm.key_klass.oop().raw(), nm.key_selector.oop().raw());
        let base = nm.code.base as u64;

        let id = match self.slots.iter().position(|s| s.is_none()) {
            Some(idx) => NmethodId(idx as u32),
            None => {
                let idx = self.slots.len();
                self.slots.push(None);
                NmethodId(idx as u32)
            }
        };
        nm.id = id;
        let block_key = self.slots_block_key(&nm);
        self.slots[id.0 as usize] = Some(nm);
        match block_key {
            // S24 A1: a block nmethod is keyed by its CompiledBlock's oop
            // bits in `by_block`; its key_klass/key_selector are fillers
            // that must NEVER enter `by_key` (they would collide across
            // every block: (closure_klass, #aBlock)).
            Some(bk) => {
                self.by_block.insert(bk, id);
            }
            None => {
                self.by_key.insert(key, id);
            }
        }

        let pos = self.by_addr.partition_point(|&(b, _)| b < base);
        self.by_addr.insert(pos, (base, id));

        id
    }

    pub fn get(&self, id: NmethodId) -> Option<&Nmethod> {
        self.slots.get(id.0 as usize)?.as_ref()
    }

    /// D4.1: `rt_resolve_send` needs this to patch the CALLER's own
    /// `ic_sites[i].state` after `patch_branch26_at` repoints its `bl` —
    /// `get`'s `&self` can't do that, and the two are never live at once
    /// (every caller reads what it needs from `get`'s borrow into owned
    /// locals before reaching for this one).
    pub fn get_mut(&mut self, id: NmethodId) -> Option<&mut Nmethod> {
        self.slots.get_mut(id.0 as usize)?.as_mut()
    }

    /// Binary-searches `by_addr` for the nmethod whose published range
    /// `[base, base+len)` contains `pc`.
    pub fn find_by_pc(&self, pc: u64) -> Option<NmethodId> {
        let idx = self.by_addr.partition_point(|&(base, _)| base <= pc);
        if idx == 0 {
            return None;
        }
        let (base, id) = self.by_addr[idx - 1];
        let nm = self.get(id)?;
        if pc >= base && pc < base + nm.code.len as u64 {
            Some(id)
        } else {
            None
        }
    }

    pub fn lookup(&self, k: KlassOop, sel: SymbolOop) -> Option<NmethodId> {
        self.by_key.get(&(k.oop().raw(), sel.oop().raw())).copied()
    }

    /// S15: the OSR side table — `(klass, selector, loop-header bci)` → the
    /// nmethod carrying that OSR entry.
    pub fn lookup_osr(&self, k: KlassOop, sel: SymbolOop, bci: u16) -> Option<NmethodId> {
        self.osr_table
            .get(&(k.oop().raw(), sel.oop().raw(), bci))
            .copied()
    }

    pub fn install_osr(&mut self, k: KlassOop, sel: SymbolOop, bci: u16, id: NmethodId) {
        self.osr_table
            .insert((k.oop().raw(), sel.oop().raw(), bci), id);
    }

    /// D8: visit every embedded-oop pool word in every nmethod (S10 has
    /// only `Alive` ones), wrapped in one `JitWriteGuard` covering every
    /// nmethod's own range — the guard's flush is harmless even when `f`
    /// leaves a word unchanged (pool words are data, not code, but the
    /// guard doesn't know or care which).
    ///
    /// D8's own text writes this as `code_table.oops_do(cache, &mut f)`,
    /// taking the owning `CodeCache` as a second parameter. That signature
    /// cannot be satisfied at the scavenge.rs call site: `f` there closes
    /// over `&mut VmState` (`scavenge_oop` needs the whole state, not just
    /// a field), and a `cache: &vm.code_cache` argument evaluated in the
    /// same call would borrow `vm` immutably while the closure holds it
    /// mutably — `CodeCache` has no cheap `Default` to `std::mem::take` it
    /// out of the way the way [`CodeTable`] itself is taken out (a real
    /// `Default` would mean a second `mmap` per scavenge, not a placeholder
    /// swap). Dropping the parameter sidesteps the conflict entirely, and
    /// the containment check it would have enabled is available more
    /// precisely from data this method already has: `reloc.offset` against
    /// *this nmethod's own* `code.len`, tighter than "is this address
    /// anywhere in the whole cache" (which would miss a reloc that drifted
    /// into a *neighboring* nmethod's bytes).
    /// `include_key_klass` (S12 D5): `false` ONLY during a full GC's own
    /// MARK pass — the customization key stays unmarked so a truly dead
    /// receiver klass can be detected (`weak_sweep`) instead of being kept
    /// artificially alive forever by the nmethod that happens to still
    /// name it. Every other caller (scavenge, which never flushes and
    /// treats all code roots strongly; full GC's own later update/rewrite
    /// pass, which must keep a SURVIVING key's address current — "weak ≠
    /// unmaintained") passes `true`. Everything else in a pool (method
    /// literals, selectors, PIC guard klasses, adapter method oops) has no
    /// such distinction and is unconditionally strong (D5: "only the key
    /// is weak").
    pub fn oops_do(&mut self, include_key_klass: bool, f: &mut dyn FnMut(&mut u64)) {
        let mut guard = JitWriteGuard::new();
        for nm in self.slots.iter().flatten() {
            // S13 §2b/§3: `Alive` OR `NotEntrant`, skipping ONLY `Zombie`. A
            // `NotEntrant` nmethod is invalidated for FUTURE entry but may
            // still have in-flight activations (step 9's lazy return-address
            // redirection is exactly what lets its frames keep running old
            // code); those frames `ldr` their oops — `ConstPool` literals, the
            // key_klass guard word — straight out of THIS nmethod's own pool,
            // so a moving GC that skipped it would leave those words stale and
            // the running old code would read a dangling oop. The doc's §3
            // ("existing activations complete under the OLD method") REQUIRES
            // an invalidated nmethod's oops stay GC-current until it is
            // `Zombie`d (step 10, only after no frame references it). A
            // `Zombie` has, by that step's own precondition, no live
            // activation, so its pool need not be maintained.
            if matches!(nm.state, NmState::Zombie) {
                continue;
            }
            guard.note(nm.code.base, nm.code.len);
            let base = nm.code.base as usize;
            for reloc in nm.oop_relocs() {
                // S12 D5: an ALIVE nmethod's `key_klass` is WEAK — the full-GC
                // mark pass (`include_key_klass == false`) skips it so a klass
                // death can flush the nmethod (`weak_sweep`). S13 §2b: a
                // `NotEntrant` nmethod's key is STRONG instead — it has
                // in-flight activations whose code `ldr`s this very key_klass
                // guard word, and the state gate above keeps that word
                // GC-current; if the mark pass also skipped it, the klass could
                // die unmarked and the update pass would chase a never-installed
                // forwarding word into a dangling read (`weak_sweep`/`iter_alive`
                // exclude NotEntrant, so nothing else keeps it alive). Only
                // Zombie can never reach here (skipped wholesale above). Keeping
                // a redefined method's key alive until its frames drain (Zombie,
                // step 10) is the correct conservative choice.
                if !include_key_klass
                    && matches!(reloc.kind, RelocKind::KeyKlassOop)
                    && matches!(nm.state, NmState::Alive)
                {
                    continue;
                }
                debug_assert!(
                    (reloc.offset as usize) + 8 <= nm.code.len,
                    "oops_do: reloc offset {} + 8 exceeds nmethod {:?}'s own code length {}",
                    reloc.offset,
                    nm.id,
                    nm.code.len
                );
                let addr = (base + reloc.offset as usize) as *mut u64;
                // SAFETY: `addr` is `code.base + reloc.offset`, and
                // `CodeBlob`'s own contract (S9) guarantees every `Reloc`
                // offset lands inside `[0, code.len)` of the blob it came
                // from, which `publish` copied byte-for-byte into this
                // handle — so `addr` is inside `[nm.code.base,
                // nm.code.base + nm.code.len)`, live MAP_JIT memory, and
                // 8-byte aligned (D3.3: the pool starts 8-aligned and
                // every entry is one `u64`). The write is guarded (this
                // function's own `guard`, noted for this exact range).
                unsafe { f(&mut *addr) };
            }
        }
    }

    /// S12 D5 point 2: nmethods whose OWN `key_klass` did NOT get marked by
    /// the full GC's own just-finished mark pass (`oops_do`/`update_keys`
    /// called with `include_key_klass: false`) — the receiver klass this
    /// nmethod was customized for is genuinely dead (nothing else in the
    /// live graph references it), so this nmethod must be flushed
    /// (`CodeTable::flush`, S12 step 6). Must run strictly between phase A
    /// (mark) and phase B (`forwarding_compute`): a klass's plain mark bit
    /// is only meaningfully readable in that window — once `forwarding_
    /// compute` reaches its address, a MARKED klass's header becomes a
    /// forwarding pointer (no longer a plain mark word at all).
    pub fn weak_sweep(&self) -> Vec<NmethodId> {
        self.iter_alive()
            .filter(|nm| {
                let obj = MemOop::try_from(nm.key_klass.oop())
                    .expect("weak_sweep: key_klass must always be mem-shaped");
                !obj.mark().gc_mark()
            })
            .map(|nm| nm.id)
            .collect()
    }

    /// Every currently-`Alive` nmethod — the iteration base for `weak_sweep`
    /// above and `codecache::flush`'s own compiled-site invalidation sweep
    /// (S12 D6.2). Deliberately EXCLUDES `NotEntrant` (S13 §2b's
    /// `make_not_entrant` now produces it): a NotEntrant nmethod must NOT be
    /// weak-swept — it may have in-flight activations, so it can only be
    /// reclaimed by step 10's frame-reference check (→ `Zombie`), never by a
    /// dead key. Its key is kept STRONG in `oops_do` above precisely so it is
    /// never seen dead here anyway; the two decisions are the paired halves of
    /// "a NotEntrant nmethod stays fully live until its frames drain".
    /// S15 A8: every installed nmethod regardless of state — the stats
    /// dump's code-cache byte accounting (`Alive` vs not) reads this.
    pub fn iter_all(&self) -> impl Iterator<Item = &Nmethod> {
        self.slots.iter().flatten()
    }

    pub fn iter_alive(&self) -> impl Iterator<Item = &Nmethod> {
        self.slots
            .iter()
            .flatten()
            .filter(|nm| matches!(nm.state, NmState::Alive))
    }

    /// S13 §3 (step 10c): the ids of every currently-`NotEntrant` nmethod — the
    /// candidate set the full-GC zombie sweep (`codecache::flush::
    /// sweep_not_entrant_zombies`) tests against live frame references. The
    /// counterpart to [`Self::iter_alive`]'s deliberate NotEntrant exclusion: a
    /// NotEntrant nmethod is reclaimed ONLY here (→ `Zombie`), never by the
    /// dead-key weak sweep.
    pub fn not_entrant_ids(&self) -> Vec<NmethodId> {
        self.slots
            .iter()
            .flatten()
            .filter(|nm| matches!(nm.state, NmState::NotEntrant))
            .map(|nm| nm.id)
            .collect()
    }

    /// S12 D6.1 step 1: marks `id` `Zombie` and removes it from `by_key` —
    /// `by_addr`/the slot itself deliberately stay put until [`Self::
    /// remove`] (step 4), so a concurrent walk this cycle still classifies
    /// any stale return address as belonging to SOME nmethod rather than
    /// finding a hole (`flush` is only ever called once step 2's
    /// no-live-activation invariant already holds, but the ordering costs
    /// nothing and matches D6's own text exactly).
    pub fn mark_zombie(&mut self, id: NmethodId) {
        let nm = self.slots[id.0 as usize]
            .as_mut()
            .expect("mark_zombie: id must be alive");
        nm.state = NmState::Zombie;
        let key = (nm.key_klass.oop().raw(), nm.key_selector.oop().raw());
        if let Some(m) = &nm.osr_map {
            let okey = (key.0, key.1, m.osr_bci);
            if self.osr_table.get(&okey) == Some(&id) {
                self.osr_table.remove(&okey);
            }
        }
        // Only unhook the key if it still names THIS nmethod: a replaced
        // (recompiled) nmethod's key already points at its successor, and
        // removing it here would orphan the LIVE replacement — every later
        // `lookup` would miss, so compiled sites would resolve to c2i
        // adapters (interpreting forever) and the interpreter trigger would
        // compile duplicate after duplicate. (The S14 perf-recovery bug.)
        if self.by_key.get(&key) == Some(&id) {
            self.by_key.remove(&key);
        }
        // S24 A1: the by_block twin, same successor guard — recompilation
        // installs the replacement FIRST (compile_block_versioned mirrors
        // recompile.rs's install-first rule), so only remove the entry if
        // it still names THIS id.
        if let Some(bk) = self
            .slots
            .get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .and_then(|nm| nm.block_method)
            .map(|b| b.oop().raw())
        {
            if self.by_block.get(&bk) == Some(&id) {
                self.by_block.remove(&bk);
            }
        }
    }

    /// S13 D1 §2a: mark `id` `NotEntrant` and unhook it from `by_key`, so a
    /// new send for `(key_klass, key_selector)` MISSES the lookup map and
    /// re-resolves against the current MethodDictionary (the redefined
    /// method's fresh nmethod, or the interpreter). Interpreter ICs holding a
    /// stale smi id of this nmethod self-heal on their own next use: the
    /// dispatch path re-validates `code_table.get(id).is_some_and(state ==
    /// Alive)` before entering (`interpreter::send::send_generic`), and a
    /// `NotEntrant` state fails that check.
    ///
    /// Crucially — UNLIKE [`Self::mark_zombie`] — the `Nmethod` RECORD and its
    /// `by_addr` entry stay put: an in-flight activation is still running this
    /// nmethod's code, and a compiled caller's `bl` still targets its
    /// (now-patched) entry, so `get(id)`/`find_by_pc(pc)` must keep resolving
    /// to it for the GC frame-root scan (`oopmap_at`) and the step-9 lazy
    /// return-address walk. Freeing the code block and dropping the record is
    /// step 10's zombie sweep, only after no frame references it. The caller
    /// (`flush::make_not_entrant`) does the entry patching separately (it
    /// needs `&mut CodeCache`, which this table doesn't own).
    pub fn set_not_entrant(&mut self, id: NmethodId) {
        let nm = self.slots[id.0 as usize]
            .as_mut()
            .expect("set_not_entrant: id must still be installed");
        debug_assert!(
            matches!(nm.state, NmState::Alive),
            "set_not_entrant: {id:?} must be Alive (double-invalidation, or a Zombie \
             being resurrected, is a VM-consistency bug)"
        );
        nm.state = NmState::NotEntrant;
        let key = (nm.key_klass.oop().raw(), nm.key_selector.oop().raw());
        if let Some(m) = &nm.osr_map {
            let okey = (key.0, key.1, m.osr_bci);
            if self.osr_table.get(&okey) == Some(&id) {
                self.osr_table.remove(&okey);
            }
        }
        // Same successor guard as `mark_zombie`: recompilation installs the
        // replacement FIRST (recompile.rs relies on the key never pointing
        // at nothing), so by the time the old version is retired the key
        // already names the new one — removing it unconditionally would
        // orphan the replacement from every future `lookup`.
        if self.by_key.get(&key) == Some(&id) {
            self.by_key.remove(&key);
        }
        // S24 A1: the by_block twin, same successor guard — recompilation
        // installs the replacement FIRST (compile_block_versioned mirrors
        // recompile.rs's install-first rule), so only remove the entry if
        // it still names THIS id.
        if let Some(bk) = self
            .slots
            .get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .and_then(|nm| nm.block_method)
            .map(|b| b.oop().raw())
        {
            if self.by_block.get(&bk) == Some(&id) {
                self.by_block.remove(&bk);
            }
        }
    }

    /// S12 D6.1 step 4: removes `id` from `by_addr`, drops its `Nmethod`
    /// from the slot (`None` — reusable by a future `install`, safe
    /// because of S10 D4's own id-validation dispatch), and returns it so
    /// the caller can free its `CodeCache` space — this table doesn't own
    /// the cache, so it can't do that part itself.
    pub fn remove(&mut self, id: NmethodId) -> Nmethod {
        let nm = self.slots[id.0 as usize]
            .take()
            .expect("remove: id must still be installed");
        let pos = self
            .by_addr
            .iter()
            .position(|&(_, i)| i == id)
            .expect("remove: id must have a by_addr entry");
        self.by_addr.remove(pos);
        nm
    }

    /// `key_klass`/`key_selector` are plain Rust struct fields, not
    /// `MAP_JIT` memory — no guard needed, just `f`'s transform applied in
    /// place. Call [`Self::rehash`] afterward. Called by BOTH collectors:
    /// the full GC passes `forward_chase`, and the SCAVENGE passes a
    /// `|oop| scavenge_oop(vm, oop)` closure (so `f` may capture `&mut
    /// VmState` — that's fine, `code_table` is taken out via
    /// `std::mem::take` for the duration so the borrow doesn't conflict).
    /// An earlier draft treated these as full-GC-only "strong-until-S12"
    /// roots and had the scavenge skip them entirely — a real
    /// use-after-free: a scavenge that relocates a YOUNG `key_selector`
    /// symbol (or `key_klass`) updated the interning symbol/class table's
    /// own reference but left this SEPARATE nmethod copy dangling at the
    /// vacated address, which a later full GC would then chase into poisoned
    /// memory. Found via `MACVM_GC_STRESS=1 + MACVM_JIT=threshold=1`
    /// together (see `Justfile`'s `gate-s11`).
    ///
    /// `include_key_klass` (S12 D5, same rule as [`Self::oops_do`]'s own
    /// parameter — pass the SAME value at both call sites, one call each
    /// per collector phase): `false` only during a full GC's own mark
    /// pass, so `key_klass` is left exactly as `weak_sweep` needs to find
    /// it (a plain, not-yet-forwarded mark word); `key_selector` and every
    /// `IcSite` field are ALWAYS visited regardless — D5's own text: only
    /// the key is weak, everything else stays strong (`key_selector`
    /// specifically must survive even a dying key_klass, since D6's own
    /// flush sweep still needs to read it).
    pub fn update_keys(&mut self, include_key_klass: bool, f: &mut dyn FnMut(Oop) -> Oop) {
        for nm in self.slots.iter_mut().flatten() {
            if include_key_klass {
                let nk = f(nm.key_klass.oop());
                // SAFETY: a collector transform never changes an oop's
                // shape, only (at most) its address — this was
                // klass-shaped before, it's klass-shaped after (same
                // reasoning as `roots.rs`'s own `root_klass!`/`root_sel!`
                // macros, which this mirrors).
                nm.key_klass = unsafe { KlassOop::from_oop_unchecked(nk) };
            }
            let ns = f(nm.key_selector.oop());
            nm.key_selector = unsafe { SymbolOop::from_oop_unchecked(ns) };
            // S24 A1: the block nmethod's own CompiledBlock — ALWAYS
            // visited (never weak): `rehash` rebuilds `by_block` from this
            // field, and `note_uncommon_trap`'s is_block branch derefs it,
            // so a moving GC must keep it current exactly like
            // `key_selector` above.
            if let Some(b) = nm.block_method {
                let nb = f(b.oop());
                nm.block_method = Some(unsafe { MethodOop::from_oop_unchecked(nb) });
            }
            // S11 D3: each IcSite's own selector is the SAME kind of
            // Rust-side (not MAP_JIT) oop field as key_selector above —
            // the machine-code pool word carrying the same selector
            // (RelocKind::Oop, emitted alongside the InlineCache reloc)
            // is handled by `oops_do` already; this is purely the
            // Rust-struct-field half.
            for site in &mut nm.ic_sites {
                let ss = f(site.selector.oop());
                site.selector = unsafe { SymbolOop::from_oop_unchecked(ss) };
                // A resolved-mono site also mirrors its guard `klass` as a
                // Rust-side oop (the machine-code guard's own klass pool word
                // is `oops_do`'s job; this is the Rust-struct half). Same
                // relocation hazard as `key_selector`: `rt_resolve_send`'s
                // poly-transition path derefs this `klass`
                // (`lookup(vm, old_k, ..)`), so a stale one is a
                // use-after-free once a moving GC relocates that klass.
                if let IcState::Mono { klass, target } = site.state {
                    let nk = f(klass.oop());
                    site.state = IcState::Mono {
                        klass: unsafe { KlassOop::from_oop_unchecked(nk) },
                        target,
                    };
                }
                // S13 step 10d: a super site's static holder-superclass is
                // derefed by `rt_resolve_send` (`lookup(super_klass, ..)`) on a
                // re-dispatch, so a moving GC must keep it current too — same
                // relocation hazard as the `Mono` guard klass above.
                if let Some(sk) = site.super_klass {
                    let nsk = f(sk.oop());
                    site.super_klass = Some(unsafe { KlassOop::from_oop_unchecked(nsk) });
                }
            }
            // S14 step 4b: each inline dependency `(klass, selector)` is a
            // Rust-side oop pair `deps::affected_by_install` derefs
            // (`superchain_contains(vm, klass, ..)`, selector raw-compare) on
            // every `install_method`, so a moving GC must keep both current —
            // same relocation hazard as `key_selector`/`super_klass`. ALWAYS
            // visited (never weak): unlike `key_klass`, an inline-dep klass is a
            // callee's receiver klass the guard genuinely assumes stays alive,
            // and the invalidation walk must find it at its live address.
            for (dk, ds) in &mut nm.inline_deps {
                let ndk = f(dk.oop());
                *dk = unsafe { KlassOop::from_oop_unchecked(ndk) };
                let nds = f(ds.oop());
                *ds = unsafe { SymbolOop::from_oop_unchecked(nds) };
            }
        }
    }

    /// Rebuilds `by_key` from every nmethod's current `key_klass`/
    /// `key_selector` — mandatory after [`Self::update_keys`] moves any of
    /// them (P7): the map is keyed on raw oop bits, which a moving full GC
    /// changes out from under it exactly like `LookupCache`'s own entries.
    pub fn rehash(&mut self) {
        self.by_key.clear();
        self.by_block.clear();
        self.osr_table.clear();
        for nm in self.slots.iter().flatten() {
            // Only ALIVE nmethods re-enter either map: retirement REMOVED a
            // NotEntrant/Zombie entry's key on purpose (S14's successor
            // guard), and blindly re-inserting here would resurrect it —
            // slot-order iteration could then shadow a live replacement
            // sitting in a LOWER (reused) slot. (Latent since S12's rehash;
            // made observable by S15's osr_table sharing the discipline.)
            if !matches!(nm.state, NmState::Alive) {
                continue;
            }
            if let Some(b) = nm.block_method {
                // S24 A1: a block nmethod re-enters by_block (its filler
                // key must never enter by_key — it would collide across
                // every block).
                self.by_block.insert(b.oop().raw(), nm.id);
                continue;
            }
            let key = (nm.key_klass.oop().raw(), nm.key_selector.oop().raw());
            self.by_key.insert(key, nm.id);
            if let Some(m) = &nm.osr_map {
                self.osr_table.insert((key.0, key.1, m.osr_bci), nm.id);
            }
        }
    }

    /// Test-only hook standing in for S12 flushing (the real way a slot an
    /// IC still names can go away — S10 never frees one organically):
    /// simulates a stale id by clearing `id`'s slot out from under any IC
    /// that still holds it, so `send_generic`'s own re-validate-before-
    /// `enter_compiled` check (`code_table.get(nm_id).is_some_and(...)`)
    /// has a real gap to catch (tests_s10.md's `stale_id_self_heals`).
    #[cfg(test)]
    pub(crate) fn test_clear_slot(&mut self, id: NmethodId) {
        self.slots[id.0 as usize] = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::assembler::CodeBlob;
    use crate::memory::alloc;
    use crate::memory::{fullgc, scavenge};
    use crate::oops::layout::MEM_TAG;
    use crate::oops::smi::SmallInt;
    use crate::oops::wrappers::ArrayOop;
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

    /// A syntactically valid but never-allocated `KlassOop`/`SymbolOop`,
    /// for tests that only exercise `CodeTable`'s own bookkeeping (raw
    /// key bits, id/address arithmetic) — none of `install`/`get`/
    /// `lookup`/`find_by_pc` ever dereference `key_klass`/`key_selector`
    /// or `code.base`, only `MemOop::from_oop_unchecked`'s tag-level cast
    /// (`oops::wrappers` doc: "tag-level check only") and pointer/`u64`
    /// arithmetic, so a fake but correctly mem-tagged address is sound
    /// here — unlike [`CodeTable::oops_do`], which really does write
    /// through `code.base` and therefore needs a real `CodeCache`
    /// allocation (see the scavenge/full-GC tests below).
    fn fake_klass(addr: usize) -> KlassOop {
        unsafe { KlassOop::from_oop_unchecked(Oop::from_raw(addr as u64 + MEM_TAG)) }
    }
    fn fake_selector(addr: usize) -> SymbolOop {
        unsafe { SymbolOop::from_oop_unchecked(Oop::from_raw(addr as u64 + MEM_TAG)) }
    }
    fn fake_nmethod(
        key_klass: KlassOop,
        key_selector: SymbolOop,
        base: usize,
        len: usize,
    ) -> Nmethod {
        Nmethod {
            osr_cold_sends: 0,
            id: NmethodId(0),
            key_klass,
            key_selector,
            code: CodeHandle {
                base: base as *const u8,
                len,
            },
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

    /// `tests_s12.md`'s `oopmap_roundtrip`: slots {0, 5, 63, 64} on a
    /// 70-slot map — 63 and 64 straddle the word boundary (`SmallVec<[u64;
    /// 1]>` needs a second word once any bit >= 64 is set), the exact case
    /// a plain per-bit `Vec<bool>` would never exercise.
    #[test]
    fn oopmap_roundtrip() {
        let mut map = OopMap::empty();
        for &s in &[0u16, 5, 63, 64] {
            map.set(s);
        }
        let got: Vec<u16> = map.iter_slots().collect();
        assert_eq!(got, vec![0, 5, 63, 64]);
        for s in 0..70u16 {
            assert_eq!(map.is_oop(s), [0u16, 5, 63, 64].contains(&s), "slot {s}");
        }
    }

    /// `iter_slots`/`is_oop` on a genuinely empty map (no `set` calls at
    /// all) must never panic and never report anything live — the
    /// `oopmaps[0]` reserved-empty-map convention's own correctness
    /// depends on this.
    #[test]
    fn oopmap_empty_reports_nothing() {
        let map = OopMap::empty();
        assert_eq!(map.iter_slots().count(), 0);
        for s in 0..128u16 {
            assert!(!map.is_oop(s));
        }
    }

    /// `tests_s12.md`'s `pcdesc_exact_match_required` (P1): `oopmap_at`
    /// must hit EXACTLY, never nearest-below — `ret_pc ± 4` (one
    /// instruction either side of the real safepoint) must panic, not
    /// silently return a neighboring, WRONG map.
    #[test]
    fn pcdesc_exact_match_required() {
        let base = 0x10000usize;
        let mut map0 = OopMap::empty();
        map0.set(2);
        let nm = Nmethod {
            frame_slots: 4,
            slot_is_oop: vec![false, false, true, false],
            pcdescs: vec![PcDesc {
                pc_off: 0x20,
                bci: 3,
                oopmap: 0,
            }],
            oopmaps: vec![map0],
            ..fake_nmethod(fake_klass(0x1000), fake_selector(0x2000), base, 0x100)
        };

        let hit = nm.oopmap_at((base + 0x20) as u64);
        assert!(hit.is_oop(2));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            nm.oopmap_at((base + 0x20 - 4) as u64)
        }));
        assert!(
            result.is_err(),
            "4 bytes before the real safepoint must miss"
        );

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            nm.oopmap_at((base + 0x20 + 4) as u64)
        }));
        assert!(
            result.is_err(),
            "4 bytes after the real safepoint must miss too"
        );
    }

    /// tests_s10.md: install -> get by id, lookup by key, find_by_pc
    /// inside the code range and at its last byte.
    #[test]
    fn codetable_install_get_find() {
        let mut table = CodeTable::new();
        let k = fake_klass(0x1000);
        let s = fake_selector(0x2000);
        let nm = fake_nmethod(k, s, 0x4000, 64);
        let id = table.install(nm);

        assert_eq!(table.get(id).unwrap().code.base as usize, 0x4000);
        assert_eq!(table.lookup(k, s), Some(id));
        assert_eq!(
            table.lookup(fake_klass(0x1000), fake_selector(0x9990)),
            None,
            "different selector must miss"
        );

        // Inside the range: [0x4000, 0x4040).
        assert_eq!(table.find_by_pc(0x4000), Some(id), "first byte");
        assert_eq!(table.find_by_pc(0x4030), Some(id), "mid-range byte");
        assert_eq!(table.find_by_pc(0x403f), Some(id), "last byte");
        assert_eq!(table.find_by_pc(0x4040), None, "one past the end");
        assert_eq!(table.find_by_pc(0x3fff), None, "one before the start");
    }

    /// tests_s10.md (P7): mutate key oop bits (simulated move), rehash,
    /// lookup by the new key works, the old key misses.
    #[test]
    fn codetable_rehash_after_move() {
        let mut table = CodeTable::new();
        let old_k = fake_klass(0x1000);
        let s = fake_selector(0x2000);
        let nm = fake_nmethod(old_k, s, 0x4000, 16);
        let id = table.install(nm);
        assert_eq!(table.lookup(old_k, s), Some(id));

        // Simulate a full-GC move: the klass's address changes, exactly
        // what `update_keys` would apply via a real `forward_chase`. `true`
        // -- this test is about `by_key` staying in sync after a move, not
        // S12's weak-key mark filtering (a separate, later concern).
        let new_k = fake_klass(0x9000);
        table.update_keys(true, &mut |oop| {
            if oop.raw() == old_k.oop().raw() {
                new_k.oop()
            } else {
                oop
            }
        });

        assert_eq!(
            table.lookup(old_k, s),
            Some(id),
            "by_key is stale until rehash runs"
        );
        table.rehash();
        assert_eq!(table.lookup(old_k, s), None, "old key must now miss");
        assert_eq!(table.lookup(new_k, s), Some(id), "new key must hit");
    }

    /// Publishes a tiny real blob into a real `CodeCache`, patches one
    /// pool word to a live array's oop, installs an `Nmethod` referencing
    /// it via a `Reloc::Oop`, then runs a real scavenge and confirms
    /// `CodeTable::oops_do` (wired into `scavenge::scavenge`) updated the
    /// pool word to the array's new address with its contents intact —
    /// the scavenge half of D8, tested without waiting for `compile_method`
    /// (S10 step 7) to exist.
    #[test]
    fn oops_do_relocates_pool_word_across_scavenge() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;
        let arr = alloc::alloc_indexable_oops(&mut vm, klass, 1);
        arr.at_put(0, SmallInt::new(42).oop());
        let old_oop = arr.oop();

        let h = vm
            .code_cache
            .alloc(16)
            .expect("code cache alloc for test blob");
        let blob = CodeBlob {
            code: vec![0u8; 16],
            literal_off: 8,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        let pool_addr = unsafe { h.base.add(8) } as *mut u64;
        vm.code_cache.patch_pool_word(pool_addr, old_oop.raw());

        let sel = vm.universe.intern(b"testSelector");
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let nm = Nmethod {
            relocs: vec![crate::compiler::assembler::Reloc {
                offset: 8,
                kind: RelocKind::Oop,
            }],
            ..nm
        };
        let id = vm.code_table.install(nm);

        scavenge::scavenge(&mut vm).expect("scavenge must succeed");

        let nm2 = vm.code_table.get(id).unwrap();
        let new_bits = unsafe { *(nm2.code.base.add(8) as *const u64) };
        let new_oop = Oop::from_raw(new_bits);
        assert_ne!(
            new_oop.raw(),
            old_oop.raw(),
            "the pool word must have been updated to the array's new address"
        );
        let new_arr = ArrayOop::try_from(new_oop).expect("pool word must still be array-shaped");
        assert_eq!(new_arr.len(), 1);
        assert_eq!(new_arr.at(0), SmallInt::new(42).oop());
    }

    /// S12 D5 (`tests_s12.md`'s `weak_key_skipped_in_mark`): `include_key_
    /// klass: false` must skip a `RelocKind::KeyKlassOop` pool word
    /// entirely (a full GC's own mark-phase visitor must never receive
    /// it) while still visiting an ordinary `RelocKind::Oop` word right
    /// next to it; `true` must visit both — proving the filter is real,
    /// not accidentally always-on or always-off.
    #[test]
    fn oops_do_skips_key_klass_reloc_when_excluded() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;

        let h = vm
            .code_cache
            .alloc(32)
            .expect("code cache alloc for test blob");
        let blob = CodeBlob {
            code: vec![0u8; 32],
            literal_off: 8,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        let ordinary_addr = unsafe { h.base.add(8) } as *mut u64;
        let key_addr = unsafe { h.base.add(16) } as *mut u64;
        vm.code_cache.patch_pool_word(ordinary_addr, 0x1111);
        vm.code_cache.patch_pool_word(key_addr, 0x2222);

        let sel = vm.universe.intern(b"weakKeyTestSelector");
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let nm = Nmethod {
            relocs: vec![
                Reloc {
                    offset: 8,
                    kind: RelocKind::Oop,
                },
                Reloc {
                    offset: 16,
                    kind: RelocKind::KeyKlassOop,
                },
            ],
            ..nm
        };
        vm.code_table.install(nm);

        let mut seen: Vec<u64> = Vec::new();
        vm.code_table.oops_do(false, &mut |word| seen.push(*word));
        assert_eq!(
            seen,
            vec![0x1111],
            "include_key_klass=false must skip the KeyKlassOop word"
        );

        seen.clear();
        vm.code_table.oops_do(true, &mut |word| seen.push(*word));
        seen.sort();
        assert_eq!(
            seen,
            vec![0x1111, 0x2222],
            "include_key_klass=true must visit both words"
        );
    }

    /// S13 §2b (the paired half of the `oops_do` state-gate fix): once an
    /// nmethod is `NotEntrant`, its `key_klass` becomes STRONG — the full-GC
    /// mark pass (`include_key_klass=false`) must NOW visit it, because the
    /// nmethod may have in-flight activations whose code `ldr`s that guard word
    /// (which `oops_do` keeps current). An `Alive` nmethod's key stays weak
    /// (the sibling test above); a NotEntrant one does not, or its key could
    /// die unmarked and the update pass would chase a dangling forwarding.
    #[test]
    fn oops_do_visits_notentrant_key_klass_as_strong() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;
        let h = vm.code_cache.alloc(32).expect("alloc");
        vm.code_cache.publish(
            h,
            &CodeBlob {
                code: vec![0u8; 32],
                literal_off: 8,
                relocs: Vec::new(),
                listing: Vec::new(),
            },
        );
        vm.code_cache
            .patch_pool_word(unsafe { h.base.add(8) } as *mut u64, 0x1111);
        vm.code_cache
            .patch_pool_word(unsafe { h.base.add(16) } as *mut u64, 0x2222);
        let sel = vm.universe.intern(b"notEntrantKeyStrong");
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let nm = Nmethod {
            relocs: vec![
                Reloc {
                    offset: 8,
                    kind: RelocKind::Oop,
                },
                Reloc {
                    offset: 16,
                    kind: RelocKind::KeyKlassOop,
                },
            ],
            ..nm
        };
        let id = vm.code_table.install(nm);

        // While Alive: key is weak — the mark pass skips it.
        let mut seen: Vec<u64> = Vec::new();
        vm.code_table.oops_do(false, &mut |w| seen.push(*w));
        assert_eq!(seen, vec![0x1111], "Alive: key_klass weak (mark skips it)");

        // After make-not-entrant: key is STRONG — the mark pass visits it.
        vm.code_table.set_not_entrant(id);
        seen.clear();
        vm.code_table.oops_do(false, &mut |w| seen.push(*w));
        seen.sort();
        assert_eq!(
            seen,
            vec![0x1111, 0x2222],
            "NotEntrant: key_klass strong (mark pass MUST visit it)"
        );
    }

    /// S13 §2a: `set_not_entrant` flips state + unhooks from `by_key` (so new
    /// sends miss + re-resolve), but RETAINS the record (`get`) and the
    /// address map (`find_by_pc`) so an in-flight frame that traps or is walked
    /// still resolves its nmethod.
    #[test]
    fn set_not_entrant_unhooks_by_key_retains_record() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;
        let sel = vm.universe.intern(b"notEntrantInvariants");
        let h = vm.code_cache.alloc(32).expect("alloc");
        vm.code_cache.publish(
            h,
            &CodeBlob {
                code: vec![0u8; 32],
                literal_off: 8,
                relocs: Vec::new(),
                listing: Vec::new(),
            },
        );
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let id = vm.code_table.install(nm);
        let pc = h.base as u64 + 4; // an address inside the code

        assert_eq!(
            vm.code_table.lookup(klass, sel),
            Some(id),
            "installed → lookup"
        );
        vm.code_table.set_not_entrant(id);
        assert!(
            matches!(vm.code_table.get(id).unwrap().state, NmState::NotEntrant),
            "state → NotEntrant"
        );
        assert_eq!(
            vm.code_table.lookup(klass, sel),
            None,
            "unhooked from by_key → a fresh send misses + re-resolves"
        );
        assert!(
            vm.code_table.get(id).is_some(),
            "record retained (get by id still works)"
        );
        assert_eq!(
            vm.code_table.find_by_pc(pc),
            Some(id),
            "by_addr retained (find_by_pc) → an in-flight/trapping frame still resolves"
        );
    }

    /// S12 D5's Rust-side half of the same filter: `key_selector` (and, by
    /// extension, everything else `update_keys` touches) is ALWAYS visited
    /// — only `key_klass` is conditional on `include_key_klass`.
    #[test]
    fn update_keys_skips_key_klass_when_excluded() {
        let mut table = CodeTable::new();
        let k = fake_klass(0x1000);
        let s = fake_selector(0x2000);
        let nm = fake_nmethod(k, s, 0x4000, 16);
        let id = table.install(nm);

        let new_k = fake_klass(0x9000);
        let new_s = fake_selector(0xA000);
        table.update_keys(false, &mut |oop| {
            if oop.raw() == k.oop().raw() {
                new_k.oop()
            } else if oop.raw() == s.oop().raw() {
                new_s.oop()
            } else {
                oop
            }
        });
        let nm2 = table.get(id).unwrap();
        assert_eq!(
            nm2.key_klass.oop().raw(),
            k.oop().raw(),
            "include_key_klass=false must leave key_klass untouched"
        );
        assert_eq!(
            nm2.key_selector.oop().raw(),
            new_s.oop().raw(),
            "include_key_klass=false must still update key_selector (D5: only the key is weak)"
        );

        table.update_keys(true, &mut |oop| {
            if oop.raw() == k.oop().raw() {
                new_k.oop()
            } else {
                oop
            }
        });
        let nm3 = table.get(id).unwrap();
        assert_eq!(
            nm3.key_klass.oop().raw(),
            new_k.oop().raw(),
            "include_key_klass=true must update key_klass"
        );
    }

    /// S12 D5 point 2: a fresh key_klass, never marked this cycle, must
    /// appear in `weak_sweep`'s own result (outside any full GC, F1's own
    /// invariant guarantees no mark word has `MARK_GC_BIT` set); marking it
    /// (mirroring what phase A itself does for a klass ALSO reachable some
    /// other way) must exclude it.
    #[test]
    fn weak_sweep_finds_unmarked_key_klass() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;
        let sel = vm.universe.intern(b"weakSweepTestSelector");
        let nm = fake_nmethod(klass, sel, 0x4000, 16);
        let id = vm.code_table.install(nm);

        assert_eq!(vm.code_table.weak_sweep(), vec![id]);

        let obj = MemOop::try_from(klass.oop()).unwrap();
        let m = obj.mark();
        obj.set_mark(m.with_gc_mark(true));
        assert_eq!(vm.code_table.weak_sweep(), Vec::<NmethodId>::new());

        // Leave no visible trace on this shared well-known klass's mark.
        obj.set_mark(m);
    }

    /// As above, but drives a real full GC — exercising `oops_do`'s
    /// full-GC call site AND `update_keys`/`rehash` (`key_klass` is the
    /// live `array_klass`, which a full GC's slide can itself relocate)
    /// in the same pass.
    #[test]
    fn full_gc_updates_pool_word_and_key_klass() {
        let mut vm = test_vm();
        let klass = vm.universe.array_klass;
        let arr = alloc::alloc_indexable_oops(&mut vm, klass, 1);
        arr.at_put(0, SmallInt::new(7).oop());
        let old_oop = arr.oop();

        let h = vm
            .code_cache
            .alloc(16)
            .expect("code cache alloc for test blob");
        let blob = CodeBlob {
            code: vec![0u8; 16],
            literal_off: 8,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        let pool_addr = unsafe { h.base.add(8) } as *mut u64;
        vm.code_cache.patch_pool_word(pool_addr, old_oop.raw());

        let sel = vm.universe.intern(b"anotherTestSelector");
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let nm = Nmethod {
            relocs: vec![crate::compiler::assembler::Reloc {
                offset: 8,
                kind: RelocKind::Oop,
            }],
            ..nm
        };
        let id = vm.code_table.install(nm);

        fullgc::full_gc(&mut vm).expect("full gc must succeed");

        let nm2 = vm.code_table.get(id).unwrap();
        let new_bits = unsafe { *(nm2.code.base.add(8) as *const u64) };
        let new_oop = Oop::from_raw(new_bits);
        let new_arr = ArrayOop::try_from(new_oop).expect("pool word must still be array-shaped");
        assert_eq!(new_arr.len(), 1);
        assert_eq!(new_arr.at(0), SmallInt::new(7).oop());

        // The installed nmethod's own key_klass must track wherever
        // array_klass ended up, and by_key must have been rehashed to
        // match — otherwise a post-GC `lookup` would silently miss an
        // nmethod that is, in fact, still installed and alive. Looked up
        // with a FRESH re-intern of the same selector bytes, not the
        // pre-GC `sel` local — that local is a plain Rust `Copy` value
        // the GC has no way to reach and update in place, so reusing it
        // post-GC would just be asserting a stale key correctly misses
        // (a different, less interesting claim than this test is for).
        let new_klass_bits = vm.universe.array_klass.oop().raw();
        assert_eq!(nm2.key_klass.oop().raw(), new_klass_bits);
        let sel2 = vm.universe.intern(b"anotherTestSelector");
        assert_eq!(
            vm.code_table.lookup(vm.universe.array_klass, sel2),
            Some(id),
            "by_key must be rehashed to the post-GC key"
        );
    }

    /// The SCAVENGE counterpart: a young `key_selector` symbol that a
    /// scavenge relocates must have the nmethod's own Rust-side copy updated
    /// too — not just the interning symbol table's reference. Regression for
    /// the use-after-free `gate-s11`'s combined gc-stress + JIT run exposed
    /// (the scavenge updated code-pool oops via `oops_do` but skipped
    /// `update_keys`, leaving this copy dangling at the vacated address).
    #[test]
    fn scavenge_updates_nmethod_key_selector() {
        let mut vm = test_vm();
        // The selector is freshly interned -> young -> relocated by the
        // scavenge. (`array_klass` itself is also young in this tiny genesis
        // heap and moves too -- so the lookup below re-reads
        // `vm.universe.array_klass` post-scavenge, never a stale local.)
        let klass = vm.universe.array_klass;
        let sel = vm.universe.intern(b"aFreshYoungKeySelectorForScavengeTest");
        let old_sel_bits = sel.oop().raw();

        let h = vm
            .code_cache
            .alloc(16)
            .expect("code cache alloc for test blob");
        let blob = CodeBlob {
            code: vec![0u8; 16],
            literal_off: 8,
            relocs: Vec::new(),
            listing: Vec::new(),
        };
        vm.code_cache.publish(h, &blob);
        let nm = fake_nmethod(klass, sel, h.base as usize, h.len);
        let id = vm.code_table.install(nm);

        scavenge::scavenge(&mut vm).expect("scavenge must succeed");

        let nm2 = vm.code_table.get(id).unwrap();
        assert_ne!(
            nm2.key_selector.oop().raw(),
            old_sel_bits,
            "scavenge must relocate the nmethod's Rust-side key_selector (the young symbol moved)"
        );
        // ...and it must still name the SAME interned symbol (an idempotent
        // re-intern returns it at its post-scavenge address).
        let sel2 = vm.universe.intern(b"aFreshYoungKeySelectorForScavengeTest");
        assert_eq!(
            nm2.key_selector.oop().raw(),
            sel2.oop().raw(),
            "the updated key_selector must resolve to the same symbol"
        );
        assert_eq!(
            vm.code_table.lookup(vm.universe.array_klass, sel2),
            Some(id),
            "by_key must be rehashed after the scavenge relocated key_selector (and key_klass)"
        );
    }
}
