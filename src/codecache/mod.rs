//! `CodeCache` — the single `MAP_JIT` region nmethods and stubs live in
//! (`docs/sprints/sprint_s09_detail.md` D2.3). One region, bump-allocated
//! with first-fit freelist reuse (D3.1); every write to already-published
//! bytes goes through [`guard::JitWriteGuard`], never a bare pointer store.
//!
//! `src/compiler/` may not call `mmap`/`pthread_jit_write_protect_np`
//! directly — this module (plus [`crate::vendor::wfasm::native_macos`],
//! which owns the actual FFI) is the only place those calls happen (D4).
//! `CodeCache` itself knows nothing about IR, bytecode, or the
//! interpreter — it only ever sees byte lengths, addresses, and
//! [`crate::compiler::assembler::CodeBlob`]. [`stubs`] is the one
//! exception (S10 D4/D5.6): `call_stub`/`stub_poll` are hand-assembled
//! Rust↔compiled trampolines, so they necessarily know `VmState`'s shape
//! and use `JasmAssembler` directly — everything else in this module
//! stays oblivious to both.

#![allow(unsafe_code)]

pub mod adapters;
pub mod deopt_trap;
pub mod ffi_stubs;
pub mod flush;
pub mod guard;
pub mod mega;
pub mod nmethod;
pub mod pics;
pub mod stubs;
// WINVM (Phase 3): the x86-64 call stub — the interpreter's door into
// compiled code, sibling of `stubs::build_call_stub`.
pub mod pics_x64;
pub mod stubs_x64;
pub mod thunks_x64;

use std::collections::BTreeMap;

use anyhow::Result;

use crate::compiler::assembler::CodeBlob;
use crate::vendor::wfasm::native::{jit_write_protect, MacJit};
use crate::vendor::wfasm::relocpatch::{abs_veneer, VENEER_LEN};
use guard::JitWriteGuard;

fn round_up16(n: usize) -> usize {
    (n + 15) & !15
}

/// `VmState::with_options`'s unconditional reservation (S10 D6) — paid even
/// under `JitMode::Off`, since it's one `mmap` reservation, not a working
/// set: nothing is written until the first real `publish`.
///
/// 16 MiB (S24 A1): the original 1 MiB was sized for S10's handful of
/// arithmetic methods; whole-world `threshold=1` now compiles every
/// method AND every eligible block, and recompile-on-trap keeps up to
/// `MAX_VERSIONS` generations while NotEntrant predecessors await their
/// full-GC sweep — deltablue at t=1 under DEOPT_STRESS legitimately
/// exceeded 1 MiB with everything working as designed (caught by the A1
/// benchmark stress matrix). Still one lazy reservation: untouched pages
/// cost address space only.
pub const DEFAULT_CODE_CACHE_CAPACITY: usize = 1 << 24;

/// A reservation within a [`CodeCache`]'s region. `len` is how much space
/// this handle actually owns — which may exceed what was requested from
/// [`CodeCache::alloc`] (a sub-32-byte remainder gets absorbed into the
/// grant rather than tracked as its own freelist sliver, D3.1) — so that
/// [`CodeCache::free`] always reclaims exactly what was carved out and
/// nothing is ever silently lost to fragmentation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CodeHandle {
    pub base: *const u8,
    pub len: usize,
}

impl CodeHandle {
    /// DBG3 (docs/DEBUGGER.md §4.4): a read-only view of this handle's
    /// published bytes, for the machine-code disassembler. Sound because a
    /// published block never moves or resizes for its whole lifetime (the
    /// S12 invariant this module's own doc pins) and the code cache is
    /// always readable; `codecache` is the crate's designated owner of
    /// raw MAP_JIT pointer access, so the one `unsafe` stays here rather
    /// than at every disassembly call site.
    ///
    /// # Safety of the returned slice
    /// Valid only while the owning `CodeCache` is alive and the block has
    /// not been `free`d — both guaranteed for any nmethod the caller found
    /// live in the code table.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: see the doc above — published, immovable, readable.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }
}

/// One `MAP_JIT` region plus its allocation bookkeeping. Never moves or
/// resizes blocks once handed out (S12 depends on this: a published
/// nmethod's address is stable for its whole lifetime).
pub struct CodeCache {
    _jit: MacJit,
    base: *mut u8,
    cap: usize,
    /// Bump pointer for never-yet-used space, always 16-aligned.
    top: usize,
    /// offset -> len, address-ordered (a `BTreeMap` iterates by key, so
    /// this is "for free" rather than a separate invariant to maintain).
    free: BTreeMap<usize, usize>,
}

impl CodeCache {
    /// Reserve `cap` bytes (rounded up to a page by the underlying
    /// `MacJit`) of `MAP_JIT` space. Leaves the region in executable mode —
    /// `MacJit::with_capacity` itself leaves the calling thread *writable*
    /// (S9 P2), which would otherwise make the very first call into any
    /// published code fault; flipping here means every state a `CodeCache`
    /// is observed in from the outside is "protected, until a
    /// `JitWriteGuard` says otherwise", matching what `JitWriteGuard::drop`
    /// itself leaves behind.
    pub fn new(cap: usize) -> Result<CodeCache> {
        let jit = MacJit::with_capacity(cap)?;
        let (base, cap) = jit.region_raw();
        jit_write_protect(true);
        Ok(CodeCache {
            _jit: jit,
            base,
            cap,
            top: 0,
            free: BTreeMap::new(),
        })
    }

    /// Reserve `len` bytes (rounded up to 16), first-fit from the freelist
    /// before bump-allocating fresh space. `None` means the cache is full —
    /// compaction/growth is out of scope for S9 (the caller's problem).
    pub fn alloc(&mut self, len: usize) -> Option<CodeHandle> {
        let len = round_up16(len);
        let hit = self
            .free
            .iter()
            .find(|&(_, &hole_len)| hole_len >= len)
            .map(|(&off, &hole_len)| (off, hole_len));
        if let Some((off, hole_len)) = hit {
            self.free.remove(&off);
            let remainder = hole_len - len;
            let granted = if remainder >= 32 {
                self.free.insert(off + len, remainder);
                len
            } else {
                // Too small to ever satisfy a future alloc on its own —
                // absorbed into this grant instead of tracked separately.
                hole_len
            };
            return Some(CodeHandle {
                base: unsafe { self.base.add(off) },
                len: granted,
            });
        }
        let off = self.top;
        let new_top = off.checked_add(len)?;
        if new_top > self.cap {
            return None;
        }
        self.top = new_top;
        Some(CodeHandle {
            base: unsafe { self.base.add(off) },
            len,
        })
    }

    /// Return `h`'s space to the freelist, coalescing with any adjacent
    /// free neighbor. A block that (after coalescing) abuts `top` retracts
    /// `top` instead of being listed — keeps the freelist from accumulating
    /// entries a simple bump-deallocation would have avoided entirely.
    pub fn free(&mut self, h: CodeHandle) {
        let mut start = h.base as usize - self.base as usize;
        let mut len = h.len;

        if let Some((&prev_off, &prev_len)) = self.free.range(..start).next_back() {
            if prev_off + prev_len == start {
                self.free.remove(&prev_off);
                start = prev_off;
                len += prev_len;
            }
        }
        if let Some((&next_off, &next_len)) = self.free.range(start + len..).next() {
            if next_off == start + len {
                self.free.remove(&next_off);
                len += next_len;
            }
        }
        if start + len == self.top {
            self.top = start;
        } else {
            self.free.insert(start, len);
        }
    }

    /// `true` if `addr` falls within this cache's region (any offset, not
    /// necessarily a live allocation's start).
    pub fn contains(&self, addr: u64) -> bool {
        let base = self.base as u64;
        addr >= base && addr < base + self.cap as u64
    }

    /// `[lo, hi)` — the region's absolute half-open address range. S13's
    /// SIGTRAP handler (`deopt_trap`) caches this pair in a write-once static
    /// at startup and does the *whole* in-signal-handler "is this pc ours?"
    /// check with two `u64` compares against it (D3 step 2) — the one global
    /// the honest-handler design permits, because a `&CodeCache` is not
    /// async-signal-safe to reach from a signal frame. Same bound `contains`
    /// tests, exposed as the raw pair the handler needs.
    pub fn bounds(&self) -> (u64, u64) {
        let base = self.base as u64;
        (base, base + self.cap as u64)
    }

    /// Bytes reserved so far (bump-pointer position) — diagnostics/stats
    /// only, never consulted by an allocation decision.
    pub fn used_bytes(&self) -> usize {
        self.top
    }

    /// Copy `blob`'s code into `h` and make it executable (D3.5): flip
    /// writable, copy, flip back to executable, flush the icache over the
    /// whole handle — in that order, all inside one [`JitWriteGuard`]. Only
    /// after this returns may `h.base` be called into or referenced from
    /// another published blob's relocations.
    pub fn publish(&mut self, h: CodeHandle, blob: &CodeBlob) -> *const u8 {
        assert!(
            blob.code.len() <= h.len,
            "CodeCache::publish: blob ({} bytes) exceeds its handle ({} bytes)",
            blob.code.len(),
            h.len
        );
        let mut g = JitWriteGuard::new();
        g.note(h.base, h.len);
        unsafe {
            std::ptr::copy_nonoverlapping(blob.code.as_ptr(), h.base as *mut u8, blob.code.len());
        }
        drop(g);
        h.base
    }

    /// Patch a `bl`/`b` site's imm26 to branch to `target`. In range
    /// (±128 MB), this is a direct field patch; out of range, `target` is
    /// routed through a freshly-allocated absolute veneer (`movz/movk
    /// x16…; br x16`) and the site is patched to branch to *that* instead —
    /// the veneer is the exception, not the common case (S11 sizes inline
    /// caches assuming most targets are near).
    pub fn patch_branch26(&mut self, site: *const u8, target: u64) {
        let site_addr = site as i64;
        let disp = target as i64 - site_addr;
        if in_branch26_range(disp) {
            let mut g = JitWriteGuard::new();
            g.note(site, 4);
            patch_branch26_field(site, disp);
            return; // g drops here: exec mode, then icache flush
        }
        let h = self
            .alloc(VENEER_LEN)
            .expect("CodeCache::patch_branch26: veneer space exhausted");
        let veneer_disp = h.base as i64 - site_addr;
        debug_assert!(
            in_branch26_range(veneer_disp),
            "a freshly bump-allocated veneer must itself be in Branch26 range of the site"
        );
        let words = abs_veneer(target);
        let mut g = JitWriteGuard::new();
        g.note(h.base, h.len);
        g.note(site, 4);
        unsafe {
            let veneer_bytes = std::slice::from_raw_parts_mut(h.base as *mut u8, VENEER_LEN);
            for (i, w) in words.iter().enumerate() {
                veneer_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
        // Still inside the guard: both the veneer body and the branch site
        // must be written before the one shared drop flips back to exec and
        // flushes both noted ranges (two_blobs_one_guard's multi-range
        // pattern, S12 P5 — patching the site before the veneer exists would
        // leave a window where it points at not-yet-written bytes).
        patch_branch26_field(site, veneer_disp);
    }

    /// As [`Self::patch_branch26`], but addressed as `(nmethod's own
    /// CodeHandle, byte offset within it)` rather than a raw pointer —
    /// `codecache` is `crate`'s one designated owner of raw pointer/`MAP_JIT`
    /// arithmetic (this module's own doc), so callers outside it (`driver.rs`,
    /// under the crate-root `#![deny(unsafe_code)]`) that only ever have an
    /// `Nmethod`'s `(code, ic_site.off)` pair — never a raw address of their
    /// own — go through this instead of computing the pointer themselves.
    pub fn patch_branch26_at(&mut self, handle: CodeHandle, off: u32, target: u64) {
        debug_assert!(
            (off as usize) + 4 <= handle.len,
            "patch_branch26_at: offset {off} + 4 exceeds the handle's own length {}",
            handle.len
        );
        let site = unsafe { handle.base.add(off as usize) };
        self.patch_branch26(site, target);
    }

    /// S13 §2b: overwrite the WHOLE 4-byte instruction word at `(handle, off)`
    /// with an unconditional `b target` (AArch64 Branch26: `0x1400_0000 |
    /// imm26`), replacing whatever instruction was there — unlike
    /// [`Self::patch_branch26_at`], which patches only the imm26 FIELD of an
    /// existing `bl`/`b`. `make_not_entrant` needs this because the site it
    /// stamps (a compiled method's `entry`/`verified_entry`) currently holds
    /// the klass-guard/prologue's first instruction (a `tst`/`stp`), NOT a
    /// branch — so there are no opcode bits worth preserving; the entire word
    /// must become a fresh `b`.
    ///
    /// `target` must be within Branch26 range (±128 MB). Both the site and the
    /// stub live in this same `CodeCache` (one `mmap`, ≤ tens of MB), so this
    /// always holds — asserted rather than veneered (the veneer machinery in
    /// [`Self::patch_branch26`] exists for far *cross-cache* targets, which do
    /// not arise here). The write is W^X-correct by construction: it happens
    /// inside one [`JitWriteGuard`] whose `Drop` flips back to exec mode and
    /// THEN flushes the icache over the noted word (guard.rs's own P9 order).
    pub fn write_branch26_at(&mut self, handle: CodeHandle, off: u32, target: u64) {
        debug_assert!(
            (off as usize) + 4 <= handle.len,
            "write_branch26_at: offset {off} + 4 exceeds the handle's own length {}",
            handle.len
        );
        let site = unsafe { handle.base.add(off as usize) };
        let disp = target as i64 - site as i64;
        assert!(
            in_branch26_range(disp),
            "write_branch26_at: target {target:#x} is out of Branch26 range of site {:#x} \
             (disp {disp:#x}) — the not_entrant_stub must live in the same code cache",
            site as u64
        );
        let word = 0x1400_0000u32 | (((disp >> 2) as u32) & 0x03FF_FFFF);
        let mut g = JitWriteGuard::new();
        g.note(site, 4);
        unsafe {
            let s = std::slice::from_raw_parts_mut(site as *mut u8, 4);
            s.copy_from_slice(&word.to_le_bytes());
        }
        // g drops here: exec mode first, then icache-flush the noted word.
    }

    /// Overwrite an 8-byte literal-pool word in place (GC relocation, or an
    /// inline cache's cached klass — S12). No re-publish, no separate flush
    /// call needed beyond the guard's own.
    pub fn patch_pool_word(&mut self, addr: *mut u64, value: u64) {
        let mut g = JitWriteGuard::new();
        let addr_u8 = addr as *const u8;
        g.note(addr_u8, 8);
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(addr as *mut u8, 8);
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    }
}

impl Drop for CodeCache {
    /// S13: retire this cache's `deopt_trap` registry entry (if any) BEFORE the
    /// `_jit: MacJit` field's own `Drop` unmaps the region (fields drop after
    /// the struct's own `Drop::drop`). Without this, the SIGTRAP handler could
    /// keep a live registry range pointing at freed memory and redirect a trap
    /// into a dangling trampoline. A no-op for a cache that was never
    /// registered (a non-JIT `VmState`, or a bare test cache).
    fn drop(&mut self) {
        deopt_trap::deregister(self.base as u64);
    }
}

fn in_branch26_range(disp: i64) -> bool {
    (-(1i64 << 27)..(1i64 << 27)).contains(&disp)
}

/// Patch the imm26 field of the branch word at `site`, preserving its top 6
/// opcode bits — this alone is not W^X-safe; every call site wraps it in a
/// live [`JitWriteGuard`] that also notes `site` for the icache flush.
fn patch_branch26_field(site: *const u8, disp: i64) {
    let bits = ((disp >> 2) as u32) & 0x03FF_FFFF;
    unsafe {
        let s = std::slice::from_raw_parts_mut(site as *mut u8, 4);
        let word = (u32::from_le_bytes(s.try_into().unwrap()) & !0x03FF_FFFF) | bits;
        s.copy_from_slice(&word.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::assembler::Reloc as AsmReloc;

    /// D3.1: alloc 64/128/64, free the middle, alloc 96 reuses that hole
    /// with a 32-byte remainder retained (the split threshold is `>= 32`).
    #[test]
    fn alloc_first_fit_and_split() {
        let mut cc = CodeCache::new(1 << 16).unwrap();
        let a = cc.alloc(64).unwrap();
        let b = cc.alloc(128).unwrap();
        let c = cc.alloc(64).unwrap();
        cc.free(b);

        let d = cc.alloc(96).unwrap();
        assert_eq!(d.base, b.base, "reuses b's hole");
        assert_eq!(d.len, 96);
        assert_eq!(cc.free.len(), 1, "32-byte remainder retained");
        let (&off, &len) = cc.free.iter().next().unwrap();
        assert_eq!(off, (b.base as usize - cc.base as usize) + 96);
        assert_eq!(len, 32);

        assert_eq!(a.len, 64);
        assert_eq!(c.len, 64);
    }

    /// D3.1: two adjacent frees coalesce into one entry; a coalesced block
    /// that ends up abutting `top` retracts `top` instead of being listed.
    #[test]
    fn free_coalesces_neighbors() {
        let mut cc = CodeCache::new(1 << 16).unwrap();
        let a = cc.alloc(64).unwrap();
        let b = cc.alloc(64).unwrap();
        let c = cc.alloc(64).unwrap();

        cc.free(a);
        cc.free(b);
        assert_eq!(cc.free.len(), 1, "a+b coalesce into one entry");
        let (&off, &len) = cc.free.iter().next().unwrap();
        assert_eq!(off, 0);
        assert_eq!(len, 128);

        let top_before = cc.top;
        cc.free(c);
        assert!(
            cc.free.is_empty(),
            "a+b+c all abut top -> retracted, not listed"
        );
        assert_eq!(cc.top, 0);
        assert!(cc.top < top_before);
    }

    /// AAPCS/pool alignment: every handle base and length is 16-aligned.
    #[test]
    fn alloc_alignment() {
        let mut cc = CodeCache::new(1 << 16).unwrap();
        for len in [1usize, 5, 15, 16, 17, 31, 100] {
            let h = cc.alloc(len).unwrap();
            assert_eq!((h.base as usize) % 16, 0, "len={len}");
            assert_eq!(h.len % 16, 0, "len={len}");
        }
    }

    /// SPEC §9: a full cache returns `None`, never panics — growth/eviction
    /// policy is the caller's problem, not this module's.
    #[test]
    fn alloc_exhaustion_returns_none() {
        let mut cc = CodeCache::new(4096).unwrap();
        assert!(cc.alloc(1 << 20).is_none());
        assert!(cc.alloc(64).is_some(), "exhaustion must not be sticky");
    }

    /// A near branch target patches the imm26 field directly and allocates
    /// no veneer — the veneer path must be the exception, not the rule.
    #[test]
    fn patch_branch26_in_range_no_veneer() {
        let mut cc = CodeCache::new(1 << 16).unwrap();
        let h1 = cc.alloc(16).unwrap();
        let h2 = cc.alloc(16).unwrap();
        let blob = CodeBlob {
            code: 0x9400_0000u32.to_le_bytes().to_vec(),
            literal_off: 4,
            relocs: Vec::<AsmReloc>::new(),
            listing: Vec::new(),
        };
        cc.publish(h1, &blob);

        let top_before = cc.top;
        let free_before = cc.free.len();
        cc.patch_branch26(h1.base, h2.base as u64);
        assert_eq!(cc.top, top_before, "near target: no veneer bump-allocated");
        assert_eq!(
            cc.free.len(),
            free_before,
            "near target: no veneer freelist entry"
        );

        let word = {
            let s = unsafe { std::slice::from_raw_parts(h1.base, 4) };
            u32::from_le_bytes(s.try_into().unwrap())
        };
        let disp = (h2.base as i64 - h1.base as i64) >> 2;
        let expected = 0x9400_0000u32 | ((disp as u32) & 0x03FF_FFFF);
        assert_eq!(word, expected);
    }

    /// A handle inside the cache's region reports contained; an unrelated
    /// stack address does not.
    #[test]
    fn cache_contains_its_own_allocations() {
        let mut cc = CodeCache::new(1 << 16).unwrap();
        let h = cc.alloc(64).unwrap();
        assert!(cc.contains(h.base as u64));
        let local = 0u8;
        assert!(!cc.contains(&local as *const u8 as u64));
    }

    /// Fuzz-lite (seeded, deterministic — no external `rand` crate for one
    /// test): 10k random alloc/free ops never overlap live allocations,
    /// never push `top` past capacity, and freelist reuse is exercised at
    /// least once.
    #[test]
    fn alloc_free_random_churn() {
        struct Rng(u32);
        impl Rng {
            fn next_u32(&mut self) -> u32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                self.0 = x;
                x
            }
            fn below(&mut self, n: usize) -> usize {
                (self.next_u32() as usize) % n
            }
        }

        let cap = 1 << 20;
        let mut cc = CodeCache::new(cap).unwrap();
        let mut rng = Rng(0xC0FF_EE01);
        let mut live: Vec<CodeHandle> = Vec::new();
        let mut ever_used_bases: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut reused_from_freelist = false;

        for _ in 0..10_000 {
            assert!(cc.top <= cc.cap, "top must never exceed capacity");
            let should_alloc = live.is_empty() || rng.below(2) == 0;
            if should_alloc {
                let len = 1 + rng.below(512);
                if let Some(h) = cc.alloc(len) {
                    let (a0, a1) = (h.base as usize, h.base as usize + h.len);
                    for other in &live {
                        let (b0, b1) = (other.base as usize, other.base as usize + other.len);
                        assert!(a1 <= b0 || b1 <= a0, "overlapping live allocations");
                    }
                    if ever_used_bases.contains(&(h.base as usize)) {
                        reused_from_freelist = true;
                    }
                    ever_used_bases.insert(h.base as usize);
                    live.push(h);
                }
            } else {
                let idx = rng.below(live.len());
                cc.free(live.remove(idx));
            }
        }

        assert!(
            reused_from_freelist,
            "1MiB cache, 10k small ops must reuse freed space"
        );
    }
}
