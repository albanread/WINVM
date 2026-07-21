//! `JitWriteGuard` — the sole path through which `codecache` (or its callers,
//! via [`crate::codecache::CodeCache`]'s own methods) may write to `MAP_JIT`
//! memory (`docs/sprints/sprint_s09_detail.md` D2.4).
//!
//! Every write to already-published code — a fresh `publish`, a
//! `patch_branch26`, a `patch_pool_word`, a batch of several under one GC
//! pass (S12 P5) — must happen between a guard's construction and its drop:
//! construction flips this thread's `MAP_JIT` pages to writable
//! (`arm64.md` §2's per-thread `pthread_jit_write_protect_np` toggle, not
//! `mprotect`); drop flips back to executable *first*, then invalidates the
//! icache over every noted range (P9 — that ordering matters: invalidating
//! before the exec-mode flip races a concurrent instruction fetch on
//! another core against a write this thread hasn't finished committing to
//! the executable view yet).

use std::cell::Cell;

use smallvec::SmallVec;

use crate::vendor::wfasm::native::{icache_invalidate, jit_write_protect};

thread_local! {
    /// P3: the write-protect toggle is per-thread, and nesting a second
    /// guard on the same thread would have the inner `Drop` flip back to
    /// exec mode while the outer guard's writes are still in flight —
    /// forbidden in v1, asserted here rather than left as a silent footgun.
    static GUARD_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII write window over one or more `MAP_JIT` byte ranges. See the module
/// doc for the protocol; [`note`](JitWriteGuard::note) before writing, not
/// after — a range written but never noted is a write that will run without
/// its icache invalidation.
pub struct JitWriteGuard {
    ranges: SmallVec<[(usize, usize); 2]>,
}

impl JitWriteGuard {
    /// Flip this thread's `MAP_JIT` pages to writable. Panics (debug builds)
    /// if a guard is already open on this thread (P3 — no nesting in v1).
    pub fn new() -> JitWriteGuard {
        GUARD_DEPTH.with(|d| {
            let depth = d.get();
            debug_assert_eq!(
                depth, 0,
                "JitWriteGuard: nested guard on the same thread (depth {depth}) — \
                 forbidden in v1, D2.4"
            );
            d.set(depth + 1);
        });
        jit_write_protect(false);
        JitWriteGuard {
            ranges: SmallVec::new(),
        }
    }

    /// Record a byte range this guard's writes touched, so `Drop` flushes
    /// its icache. `start` need not be inside any particular allocation —
    /// `CodeCache` calls this with code-cache-relative addresses.
    pub fn note(&mut self, start: *const u8, len: usize) {
        self.ranges.push((start as usize, len));
    }
}

impl Default for JitWriteGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for JitWriteGuard {
    fn drop(&mut self) {
        // Exec mode first (P9), so a concurrent fetch on another core never
        // observes writable-but-stale-icache pages as executable.
        jit_write_protect(true);
        for &(start, len) in &self.ranges {
            icache_invalidate(start as *const u8, len);
        }
        GUARD_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P3: a second guard opened while the first is still alive panics
    /// (debug builds — `cargo test`'s default profile).
    #[test]
    #[should_panic(expected = "nested guard")]
    fn guard_depth_asserted() {
        let _outer = JitWriteGuard::new();
        let _inner = JitWriteGuard::new();
    }

    /// RAII is the point: even when the code between `new` and the implicit
    /// drop panics, unwinding still runs `Drop` and restores exec mode —
    /// proven here by a *second*, fresh guard opening cleanly afterward
    /// (which would itself panic on P3's nesting check if the first guard's
    /// depth counter had leaked from a `Drop` that never ran).
    #[test]
    fn guard_restores_exec_on_panic() {
        let result = std::panic::catch_unwind(|| {
            let _g = JitWriteGuard::new();
            panic!("simulated failure mid-guard");
        });
        assert!(result.is_err());
        // If `Drop` had not run (or GUARD_DEPTH were left non-zero), this
        // would panic on the P3 nesting check instead of constructing
        // cleanly.
        let _g = JitWriteGuard::new();
    }
}
