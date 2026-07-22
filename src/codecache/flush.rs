//! S12 D6: flushing an nmethod (also the S13 zombie path's own substrate,
//! per that section's own text). A free function taking `vm: &mut
//! VmState` — NOT a `CodeTable` method — because it genuinely needs THREE
//! of `VmState`'s own tables at once (`code_table`, `pic_table`,
//! `code_cache`); the same reasoning `memory::roots::each_code_root`'s own
//! doc gives for its own signature (a method receiver borrows `self` for
//! the call, so `code_table.flush(vm, id)` would try to borrow `vm.
//! code_table` and `vm` simultaneously — rejected regardless of what the
//! body does).
//!
//! Called by `memory::fullgc::full_gc`, right after its own weak sweep
//! (`CodeTable::weak_sweep`, S12 D5 point 2) — one call per member of
//! `flush_set`, strictly BEFORE the update/rewrite phase (D5 point 3: "no
//! updater touches freed memory"). The caller's own precondition (D5
//! point 2's invariant: a dead key klass implies no live activation of
//! this nmethod exists) is checked THERE (`fullgc::debug_assert_weak_
//! sweep_invariant`), not re-checked here.

use crate::runtime::vm_state::VmState;

use super::nmethod::{IcState, NmethodId};
use super::CodeHandle;

/// S13 D1 §2a + §2b: the invalidation MECHANISM — make `id` stop being
/// ENTERED (future calls re-resolve to the current method), while any
/// in-flight activation keeps running its old code. Called directly by the
/// redefinition/dependency path (step 10) — and, for now, by tests. This is
/// NOT the redirection of in-flight frames (§2c = step 9); it deliberately
/// leaves those frames untouched.
///
/// Two effects, matching D1 §2a/§2b exactly:
///
/// - **§2a** — `code_table.set_not_entrant(id)`: state → `NotEntrant`, unhook
///   from the `(klass, selector)` lookup map so new sends miss and re-resolve,
///   and interpreter ICs holding its stale id self-heal (their re-validate
///   check fails on the non-`Alive` state). The `Nmethod` record and code
///   block are NOT freed — an in-flight frame still runs this code, and a
///   patched compiled caller's `bl` still points at its entry (step 10's
///   zombie sweep frees it once nothing references it).
///
/// - **§2b** — entry patching: overwrite the FIRST instruction word at BOTH
///   `entry` (offset 0) and `verified_entry` (`verified_entry_off`) with `b
///   not_entrant_stub` (a full-word Branch26, [`super::CodeCache::
///   write_branch26_at`]), so a future call through a compiled caller's still-
///   live `bl` (which targets the old entry) lands in the shared stub and
///   re-dispatches like an IC miss. S11 guarantees a patchable instruction at
///   offset 0 of each entry. When `entry_off == verified_entry_off` (an
///   un-customized nmethod with no separate klass-guard entry — S10's own
///   convention), both offsets name the SAME word; patching it twice is
///   idempotent (the second write lays down the byte-identical `b`), so no
///   special-casing is needed.
///
/// W^X: each `write_branch26_at` opens its own [`super::guard::JitWriteGuard`],
/// writes the one word, then on `Drop` flips back to exec mode and flushes the
/// icache over exactly that word (guard.rs's own P9 order). Two separate words
/// ⇒ two guard cycles; nesting a single guard over both is FORBIDDEN in v1
/// (the guard's own depth assert), so per-word guards are the correct idiom
/// here, not a batched one. Invalidation is rare, so the extra flush is free.
///
/// # Panics
/// If `id` is not installed, or is not `Alive` (double-invalidation) — a
/// VM-consistency bug, never guest-triggerable.
/// S14 step 8: [`make_not_entrant`] MINUS the §2c in-flight return-redirect
/// walk — for retiring a nmethod from the RECOMPILE-ON-TRAP context
/// (`rt_uncommon_trap`, after a completed deopt), where the native chain is
/// not walkable (the trapped frame is gone; the tier-link state says
/// IntoCompiled with no anchor — the same constraint that moved `rt_poll`'s
/// disarm to the 10c sweep).
///
/// Skipping §2c is SOUND **for recompilation only**: the replacement compiles
/// the SAME method (identical semantics, better feedback-driven codegen), so
/// an in-flight activation of the old code deeper in the stack may simply run
/// to completion on it. Future entries are blocked by the §2b entry patch,
/// and a call-free loop still deopts via the §2d poll arm. SEMANTIC
/// invalidation (redefinition — deps.rs) must keep using [`make_not_entrant`]:
/// there the old code is WRONG and in-flight activations must deopt eagerly.
pub fn make_not_entrant_lazy(vm: &mut VmState, id: NmethodId) {
    let (code, entry_off, verified_entry_off) = {
        let nm = vm
            .code_table
            .get(id)
            .expect("make_not_entrant_lazy: id must be installed");
        (nm.code, nm.entry_off, nm.verified_entry_off)
    };
    vm.probe_ring
        .push(crate::runtime::vm_state::ProbeEvent::Invalidate { nm: id.0 });
    vm.code_table.set_not_entrant(id); // §2a
    let not_entrant_addr = vm.stubs.not_entrant_addr(); // §2b
    vm.code_cache
        .write_jump_at(code, entry_off, not_entrant_addr);
    if verified_entry_off != entry_off {
        vm.code_cache
            .write_jump_at(code, verified_entry_off, not_entrant_addr);
    }
    // §2d: the poll arm (the 10c sweep disarms once drained). NO §2c walk.
    vm.pending_deopt_flag = true;
    vm.reg_block.poll_flag = 1;
}

pub fn make_not_entrant(vm: &mut VmState, id: NmethodId) {
    // Read the two entry offsets + code handle BEFORE flipping state (the
    // borrow of `code_table` here ends before the mutable `code_cache` writes
    // below — disjoint fields, but sequenced cleanly regardless).
    let (code, entry_off, verified_entry_off) = {
        let nm = vm
            .code_table
            .get(id)
            .expect("make_not_entrant: id must be installed");
        (nm.code, nm.entry_off, nm.verified_entry_off)
    };

    // §2a: state → NotEntrant + unhook from by_key (record + by_addr retained).
    vm.probe_ring
        .push(crate::runtime::vm_state::ProbeEvent::Invalidate { nm: id.0 });
    vm.code_table.set_not_entrant(id);

    // §2b: redirect BOTH entries to the shared not_entrant_stub. Each write is
    // its own guard/flush cycle (no guard nesting in v1).
    let not_entrant_addr = vm.stubs.not_entrant_addr();
    vm.code_cache
        .write_jump_at(code, entry_off, not_entrant_addr);
    if verified_entry_off != entry_off {
        vm.code_cache
            .write_jump_at(code, verified_entry_off, not_entrant_addr);
    }

    // §2c: redirect the saved-LR slot of every in-flight callee whose caller is
    // an activation of `id` to `deopt_return_trampoline`, so those already-
    // running activations deopt lazily when their callee returns (§2b only
    // stops FUTURE entries; this is what handles the IN-FLIGHT ones). Plain
    // native-stack writes — no JIT toggle / icache flush (unlike §2b's entry
    // patching, which touches MAP_JIT code). The trampoline only exists with
    // the JIT on; under `JitMode::Off` there is nothing compiled to invalidate,
    // so `pending_deopts` stays empty and this is a walk over an empty native
    // chain regardless.
    let tramp = vm.stubs.deopt_return_addr();
    crate::runtime::frames::redirect_returns_into_nm(vm, id, tramp);

    // §2d (S13 step 10b): arm the loop poll. §2b stops FUTURE entries and §2c
    // redirects in-flight callee RETURNS — but a CALL-FREE loop (pure inlined
    // arithmetic, no sends → no returns to redirect) can run arbitrarily long
    // in `id`'s now-NotEntrant code, and ONLY the loop poll can deopt it.
    // Setting `poll_flag` makes every compiled loop back-edge actually call
    // `stub_poll`; `pending_deopt_flag` is the runtime-side gate. This is a
    // GLOBAL arm — all loops in all frames now poll — but `rt_poll` only DEOPTS
    // a frame whose OWN nmethod is NotEntrant; a poll in any other frame sees no
    // self-match and returns "continue" fast. The disarm is NOT done by
    // `rt_poll` (it runs with an IntoCompiled tier link + no anchor — a stack
    // walk there would abort); the step-10c zombie sweep
    // ([`sweep_not_entrant_zombies`]) clears both flags at a GC-safe walk point
    // once no NotEntrant nmethod remains. Idempotent: a second
    // `make_not_entrant` before the first drains just re-sets an already-set
    // flag.
    vm.pending_deopt_flag = true;
    vm.reg_block.poll_flag = 1;
}

/// D6.1/D6.2: flushes `id` — marks it `Zombie` and unhooks it from
/// `by_key` (step 1), sweeps every OTHER alive nmethod's own `IcSite`s and
/// resets any whose current target lands inside `id`'s own code range
/// (step 2, D6.2), then removes it from `by_addr` and returns its
/// `CodeCache` space (step 4). Step 3 (D6.3, interpreter ICs) is
/// deliberately a no-op here — S10 D4's own id-validation dispatch already
/// makes a stale smi id self-heal on next use; no eager sweep is needed or
/// wanted for it.
///
/// # Panics
/// If `id` is not currently installed (a double-flush, or a caller that
/// didn't check `weak_sweep`'s own output) — a VM-internal-consistency
/// bug, never a condition guest code can trigger.
pub fn flush_nmethod(vm: &mut VmState, id: NmethodId) {
    let flushed_range = {
        let nm = vm
            .code_table
            .get(id)
            .expect("flush_nmethod: id must be alive");
        let base = nm.code.base as u64;
        base..(base + nm.code.len as u64)
    };

    vm.code_table.mark_zombie(id);

    // --- D6.2: compiled-site invalidation sweep --------------------------
    // Collected into an owned `Vec` first, exactly like `memory::roots::
    // each_code_root`'s own frame-walk does and for the identical reason:
    // this loop only needs SHARED access to `code_table`/`pic_table` (both
    // fine simultaneously — disjoint fields of `vm`, and neither is a
    // capturing closure that would alias `vm` itself), but applying a
    // patch needs MUTABLE access to `code_cache`/`pic_table`/`code_table`
    // in turn — cleanly sequenced only if the "what needs patching" pass
    // has already finished and released its own borrows.
    struct PatchSite {
        caller_id: NmethodId,
        caller_code: CodeHandle,
        site_off: u32,
        site_idx: usize,
        free_pic: Option<CodeHandle>,
    }
    let mut to_patch: Vec<PatchSite> = Vec::new();
    for nm in vm.code_table.iter_alive() {
        for (site_idx, site) in nm.ic_sites.iter().enumerate() {
            match site.state {
                // `Unresolved` already points at `stub_resolve`;
                // `IcState::Mega` never encodes a specific nmethod's
                // entry at all (`rt_mega_lookup` re-derives its target
                // fresh on every call, D4.4 — `mega.rs`'s own doc), so
                // neither can ever reference a dying nmethod's range.
                IcState::Mono { target, .. } if flushed_range.contains(&target) => {
                    to_patch.push(PatchSite {
                        caller_id: nm.id,
                        caller_code: nm.code,
                        site_off: site.off,
                        site_idx,
                        free_pic: None,
                    });
                }
                IcState::Pic { stub } => {
                    let hits_flushed = vm
                        .pic_table
                        .pairs_of(stub)
                        .iter()
                        .any(|&(_, t)| flushed_range.contains(&t));
                    if hits_flushed {
                        to_patch.push(PatchSite {
                            caller_id: nm.id,
                            caller_code: nm.code,
                            site_off: site.off,
                            site_idx,
                            free_pic: Some(stub),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    let resolve_addr = vm.stubs.resolve_addr();
    for p in to_patch {
        if let Some(stub) = p.free_pic {
            vm.pic_table.free(&mut vm.code_cache, stub);
        }
        // `patch_branch26_at` guards its own write (a fresh `JitWriteGuard`
        // per call, `codecache::mod`'s own doc) — flushes are rare (klass
        // death, redefinition), so one guard per patched site costs
        // nothing worth batching against (this project's own "perf work
        // is out of scope before S15" stance, `memory::fullgc`'s doc).
        vm.code_cache
            .patch_call_site_at(p.caller_code, p.site_off, resolve_addr);
        vm.code_table
            .get_mut(p.caller_id)
            .expect("still installed -- we're mid-flush of a DIFFERENT nmethod entirely")
            .ic_sites[p.site_idx]
            .state = IcState::Unresolved;
    }

    // --- D6.1 step 4 ------------------------------------------------------
    let nm = vm.code_table.remove(id);
    vm.code_cache.free(nm.code);
}

/// S13 D1 §3 (step 10c): the ZOMBIE SWEEP. A `NotEntrant` nmethod that no live
/// activation references has no old code left to complete, so it can finally be
/// reclaimed — its record dropped and its code block returned to the free list
/// ([`flush_nmethod`], which also resets any compiled caller's `bl` that still
/// targeted it). Called by `memory::fullgc::full_gc` at its single
/// between-mark-and-rewrite point (the one place a full stack walk is legal
/// during GC — the same window the S12 weak sweep and its own
/// `debug_assert_weak_sweep_invariant` use).
///
/// "Referenced" = any live compiled frame's nmethod (`walk_frames` already
/// translates a step-9 redirected saved-LR back to the victim's real pc via
/// `pending_deopts`, so a redirected in-flight frame is seen here) PLUS every
/// `pending_deopts` target (belt-and-suspenders: a redirect not yet consumed
/// keeps its nmethod's code alive for `deopt_return_trampoline`).
///
/// Once the last `NotEntrant` nmethod is gone, the §2d loop-poll arming is
/// cleared — its whole job (deopting call-free NotEntrant loops that no return
/// or trap can reach) is discharged, so every loop can stop paying the poll.
/// This is the SAFE home for the disarm that [`make_not_entrant`] arms:
/// `rt_poll` itself cannot walk the stack to decide (it runs with an
/// `IntoCompiled` tier link and no anchor — it would abort), but a full GC can.
pub fn sweep_not_entrant_zombies(vm: &mut VmState) {
    let candidates = vm.code_table.not_entrant_ids();
    if candidates.is_empty() {
        return;
    }

    let mut referenced: std::collections::HashSet<NmethodId> = std::collections::HashSet::new();
    crate::runtime::frames::walk_frames(vm, |fv| {
        if let crate::runtime::frames::FrameView::Compiled { nm, .. } = fv {
            referenced.insert(nm);
        }
    });
    for pd in vm.pending_deopts.values() {
        referenced.insert(pd.nm);
    }

    for id in candidates {
        if !referenced.contains(&id) {
            flush_nmethod(vm, id);
        }
    }

    // §2d disarm: nothing NotEntrant left → no call-free loop can still need a
    // poll deopt, so drop the global arm (and its per-loop poll cost).
    if vm.code_table.not_entrant_ids().is_empty() {
        vm.pending_deopt_flag = false;
        vm.reg_block.poll_flag = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecache::nmethod::{IcSite, NmState, Nmethod};
    use crate::compiler::assembler::CodeBlob;
    use crate::oops::layout::MEM_TAG;
    use crate::oops::wrappers::{KlassOop, SymbolOop};
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

    fn fake_klass(vm: &mut VmState, addr: usize) -> KlassOop {
        let _ = vm;
        // SAFETY: test-only, tag-level shape never dereferenced (this
        // module's own tests never read a klass's body, only compare raw
        // bits) -- same reasoning as `nmethod.rs`'s own private
        // `fake_klass` helper.
        unsafe { KlassOop::from_oop_unchecked(Oop::from_raw(addr as u64 + MEM_TAG)) }
    }

    /// Publishes a tiny real blob, installs it as `nm`, returns its id
    /// alongside the handle (tests build a SEPARATE "caller" nmethod whose
    /// own `ic_sites` reference `nm`'s range, then flush `nm` and check the
    /// caller's site was reset).
    fn install_blob(vm: &mut VmState, len: usize) -> CodeHandle {
        let h = vm
            .code_cache
            .alloc(len)
            .expect("code cache alloc for test blob");
        // These fixtures stand in for compiled methods whose IcSites get
        // repointed, so the bytes at a site must look like the host's
        // patchable call — `patch_call_rel32_at` refuses to patch
        // anything that is not an `E8`, precisely so it can never corrupt
        // a real instruction. Filling with `E8` makes every offset a
        // valid site, which is what these tests want and what an
        // all-zero buffer silently was not.
        #[cfg(target_arch = "aarch64")]
        let fill = 0u8;
        #[cfg(not(target_arch = "aarch64"))]
        let fill = 0xE8u8;
        let blob = CodeBlob {
            code: vec![fill; len],
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

    /// `tests_s12.md`'s `flush_resets_dependent_sites`: nmethod A is
    /// called by compiled B (a Mono site whose own target is INSIDE A's
    /// own code range) and by a PIC in C (one of the PIC's own pairs
    /// targets A). Flushing A must: reset B's site to `Unresolved` (its
    /// own `bl` repatched to `stub_resolve`), free C's PIC AND reset its
    /// owning site the same way, and return A's own code-cache range to
    /// the free list.
    #[test]
    fn flush_resets_dependent_sites() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"target:");
        let other_sel = vm.universe.intern(b"other:");
        let klass_a = fake_klass(&mut vm, 0x1000);
        let klass_b = fake_klass(&mut vm, 0x2000);
        let klass_c = fake_klass(&mut vm, 0x3000);
        let klass_x = fake_klass(&mut vm, 0x4000);
        let klass_y = fake_klass(&mut vm, 0x5000);

        // A: the nmethod about to be flushed. Its own entry is what B's
        // and C's PIC's sites will "call".
        let a_code = install_blob(&mut vm, 32);
        let a_entry = a_code.base as u64 + 4; // an arbitrary in-range "entry"
        let a_id = vm.code_table.install(fake_nmethod(klass_a, sel, a_code));

        // B: a caller with ONE Mono IcSite whose target is A's own entry.
        let b_code = install_blob(&mut vm, 16);
        let mut b_nm = fake_nmethod(klass_b, other_sel, b_code);
        b_nm.ic_sites.push(IcSite {
            off: 0,
            selector: sel,
            argc: 0,
            state: IcState::Mono {
                klass: klass_x,
                target: a_entry,
            },
            super_klass: None,
        });
        let b_id = vm.code_table.install(b_nm);

        // C: a caller whose ONE IcSite is a Pic naming a stub built with a
        // pair that targets A's own entry (alongside an unrelated pair).
        let c_code = install_blob(&mut vm, 16);
        let smi_klass_bits = vm.universe.smi_klass.oop().raw();
        let resolve_addr = vm.stubs.resolve_addr();
        let pic_stub = vm
            .pic_table
            .build(
                &mut vm.code_cache,
                smi_klass_bits,
                resolve_addr,
                vec![(klass_x, 0xDEAD_0000), (klass_y, a_entry)],
            )
            .expect("test cache has room");
        let mut c_nm = fake_nmethod(klass_c, other_sel, c_code);
        c_nm.ic_sites.push(IcSite {
            off: 0,
            selector: sel,
            argc: 0,
            state: IcState::Pic { stub: pic_stub },
            super_klass: None,
        });
        let c_id = vm.code_table.install(c_nm);

        flush_nmethod(&mut vm, a_id);

        assert!(
            vm.code_table.get(a_id).is_none(),
            "the flushed nmethod's own slot must be gone"
        );
        assert!(
            matches!(
                vm.code_table.get(b_id).unwrap().ic_sites[0].state,
                IcState::Unresolved
            ),
            "B's own site must be reset to Unresolved"
        );
        assert!(
            matches!(
                vm.code_table.get(c_id).unwrap().ic_sites[0].state,
                IcState::Unresolved
            ),
            "C's own site must ALSO be reset to Unresolved (its PIC referenced A)"
        );

        // The freed range must be reusable (`CodeCache::free`'s own
        // contract, exercised the same way `pics.rs`'s own
        // `pic_table_build_pairs_of_free_round_trip` checks it).
        let reused = vm
            .code_cache
            .alloc(a_code.len)
            .expect("A's own freed space must be reusable");
        assert_eq!(reused.base, a_code.base);
    }

    /// `tests_s12.md`'s `flush_id_reuse_safe`: after A is flushed, its own
    /// `NmethodId` slot is reusable by a completely unrelated nmethod
    /// WITHOUT resurrecting any stale reference to the flushed one — proof
    /// that step 4's "slot -> None, id reusable" is actually safe, not
    /// just documented as such.
    #[test]
    fn flush_id_reuse_safe() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"target:");
        let klass_a = fake_klass(&mut vm, 0x1000);
        let klass_z = fake_klass(&mut vm, 0x6000);

        let a_code = install_blob(&mut vm, 16);
        let a_id = vm.code_table.install(fake_nmethod(klass_a, sel, a_code));
        flush_nmethod(&mut vm, a_id);

        let z_code = install_blob(&mut vm, 16);
        let z_sel = vm.universe.intern(b"unrelated:");
        let z_id = vm.code_table.install(fake_nmethod(klass_z, z_sel, z_code));

        assert_eq!(
            z_id, a_id,
            "the freed slot must be the one reused (CodeTable::install's own \
             'reuse a freed slot if one exists' doc)"
        );
        let z_nm = vm.code_table.get(z_id).unwrap();
        assert_eq!(z_nm.key_klass.oop().raw(), klass_z.oop().raw());
        assert_eq!(z_nm.code.base, z_code.base);
        assert_eq!(
            vm.code_table.lookup(klass_a, sel),
            None,
            "the flushed (klass_a, sel) key must stay gone -- install re-keys by \
             (klass_z, z_sel), never resurrects the old key"
        );
        assert_eq!(vm.code_table.lookup(klass_z, z_sel), Some(z_id));
    }

    /// S13 §3 (step 10c): the zombie sweep frees a `NotEntrant` nmethod that no
    /// live frame references, and — being the last one — disarms the §2d loop
    /// poll (`pending_deopt_flag` + `poll_flag`).
    #[test]
    fn zombie_sweep_frees_unreferenced_not_entrant_and_disarms() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"gone:");
        let klass = fake_klass(&mut vm, 0x1000);
        let code = install_blob(&mut vm, 32);
        let id = vm.code_table.install(fake_nmethod(klass, sel, code));

        vm.code_table.set_not_entrant(id);
        vm.pending_deopt_flag = true; // §2d arming, as make_not_entrant does
        vm.reg_block.poll_flag = 1;

        // A bare test VM has no native compiled frames, so nothing references it.
        sweep_not_entrant_zombies(&mut vm);

        assert!(
            vm.code_table.get(id).is_none(),
            "an unreferenced NotEntrant nmethod is zombied + removed"
        );
        assert!(
            !vm.pending_deopt_flag,
            "last NotEntrant gone -> pending_deopt_flag disarmed"
        );
        assert_eq!(vm.reg_block.poll_flag, 0, "poll_flag disarmed");
    }

    /// A `NotEntrant` nmethod with a still-outstanding redirected return (a live
    /// `pending_deopts` entry naming it) is NOT reclaimed —
    /// `deopt_return_trampoline` still needs its code — and the poll stays armed.
    #[test]
    fn zombie_sweep_keeps_pending_deopt_referenced() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"pending:");
        let klass = fake_klass(&mut vm, 0x2000);
        let code = install_blob(&mut vm, 32);
        let id = vm.code_table.install(fake_nmethod(klass, sel, code));

        vm.code_table.set_not_entrant(id);
        vm.pending_deopt_flag = true;
        vm.reg_block.poll_flag = 1;
        vm.pending_deopts.insert(
            0xdead_beef,
            crate::runtime::vm_state::PendingDeopt {
                orig_ret_pc: 0x1234,
                nm: id,
            },
        );

        sweep_not_entrant_zombies(&mut vm);

        assert!(
            matches!(
                vm.code_table.get(id).map(|n| n.state),
                Some(NmState::NotEntrant)
            ),
            "a pending-redirect NotEntrant nmethod survives the sweep"
        );
        assert!(
            vm.pending_deopt_flag,
            "a NotEntrant frame is still pending -> stay armed"
        );
        assert_eq!(vm.reg_block.poll_flag, 1, "poll_flag stays armed");
    }
}
