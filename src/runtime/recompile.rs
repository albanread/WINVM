//! S14 step 8: the recompile-on-trap loop — the storm-closer.
//!
//! A cold-compiled method (its ICs Empty at compile time) is full of uncommon
//! traps: every trapped send deopts, re-executes interpreted (WARMING the IC),
//! and returns — but the nmethod stays Alive with the trap still compiled, so
//! the next call traps again. That storm made cold compiled code ~6x SLOWER
//! than interpreting (the S14 benchmark). This module closes the loop: after
//! [`UNCOMMON_TRAP_LIMIT`] traps through one nmethod, re-snapshot its feedback
//! profile — if it CHANGED since compile (the storm case: `Untaken → Mono`, or
//! a guard storm's `Mono → Poly`), recompile against the warm feedback and
//! retire the old code; if it is UNCHANGED (a storm recompilation cannot fix,
//! e.g. persistent smi-overflow deopts), DECLINE (Self's `checkEffectiveness`
//! — A5, "not optional") and re-arm the counter. [`MAX_VERSIONS`] caps
//! replacement generations — the second layer of the anti-thrash defense.
//!
//! The doc's full recompilation POLICY (nmethod prologue invocation counters,
//! levels 2–4 escalation, the caller-preference stack walk) is deliberately a
//! LATER slice: none of it is needed to close the deopt storm, which is the
//! item the S14 perf finding demands. `Nmethod.level` stays 1 until then.

use crate::codecache::nmethod::{NmState, NmethodId};
use crate::runtime::vm_state::VmState;

/// Traps through one nmethod before the recompile check runs. 2, not 1: the
/// FIRST trap warms the IC (its re-execution is what populates it); a site
/// that only ever executes once never traps again, so recompiling after one
/// trap would waste a compile on code that already stopped trapping. Tunable.
pub const UNCOMMON_TRAP_LIMIT: u32 = 2;

/// Replacement generations per (klass, selector) before recompilation gives up
/// (SPEC §8.1). The old code stays Alive and keeps deopting — correct, just
/// slow — rather than churning compiles it has already proven unhelpful.
pub const MAX_VERSIONS: u8 = 3;

/// Bump `nm`'s trap counter and, past [`UNCOMMON_TRAP_LIMIT`], run the
/// recompile-on-trap check. Called by `rt_uncommon_trap` AFTER the trap's
/// nested interpreted run completed — the perfect moment: whatever the trap
/// re-executed has ALREADY warmed its IC, so a recompile sees the new truth.
///
/// The nested run may have done arbitrary work (GC, invalidation, even a
/// competing compile), so everything is re-checked against the live tables.
pub fn note_uncommon_trap(vm: &mut VmState, id: NmethodId) {
    // The Debug-menu Compiler (JIT) kill-switch: with compilation off, don't
    // version-recompile a trap-storming nmethod — the old code stays Alive and
    // keeps trap-deopting to the interpreter, the same accepted behavior as
    // the MAX_VERSIONS thrash cap below ("live with the deopts"). Counting
    // resumes when the switch turns back on.
    if !crate::runtime::jit_compile_enabled() {
        return;
    }
    let Some(nm) = vm.code_table.get(id) else {
        return; // flushed during the nested run
    };
    if !matches!(nm.state, NmState::Alive) {
        return; // already invalidated/replaced during the nested run
    }
    let version = nm.version;
    let key_klass = nm.key_klass;
    let method_sel = nm.key_selector;
    let old_hash = nm.profile_hash;
    // S24 A1 (adversarial review §5): a BLOCK nmethod's key pair is a
    // filler — `lookup(key_klass, key_selector)` below would either
    // early-return (leaving a trap-dense cold block re-trapping FOREVER)
    // or resolve an UNRELATED method under (closure_klass, #aBlock) and
    // version-churn it. Branch BEFORE the key lookup: the CompiledBlock is
    // carried on the nmethod itself with key_selector's GC discipline.
    let block_method = nm.block_method;

    let count = {
        let nm = vm.code_table.get_mut(id).expect("checked Alive just above");
        nm.trap_count += 1;
        nm.trap_count
    };
    if count < UNCOMMON_TRAP_LIMIT {
        return;
    }
    if version >= MAX_VERSIONS {
        return; // thrash cap: live with the deopts
    }

    // S24 A1: the block branch — same LIMIT/version/profile discipline as
    // methods, but resolved via `block_method` and replaced via
    // `compile_block_versioned` + the by_block successor guard.
    if let Some(blk) = block_method {
        let now_hash = crate::compiler::feedback::snapshot_profile(vm, blk);
        if now_hash == old_hash {
            vm.stats.recompile_declined_ineffective += 1;
            let nm = vm.code_table.get_mut(id).expect("still present");
            nm.trap_count = 0;
            return;
        }
        match crate::compiler::driver::compile_block_versioned(vm, blk, version + 1) {
            Some(_) => {
                vm.stats.recompiles += 1;
                if vm.options.trace.is_enabled("deopt") {
                    eprintln!(
                        "[deopt] recompiled BLOCK nm={} v{} (trap storm)",
                        id.0, version
                    );
                }
                crate::codecache::flush::make_not_entrant_lazy(vm, id);
            }
            None => {
                let nm = vm.code_table.get_mut(id).expect("still present");
                nm.version = MAX_VERSIONS;
            }
        }
        return;
    }

    // The compiled method: resolve back from the key (the nmethod's key names
    // the customized receiver klass + selector; the METHOD is what the lookup
    // finds — same resolution the interpreter dispatch performs).
    let Some(method) = crate::runtime::lookup::lookup(vm, key_klass, method_sel) else {
        return; // method removed during the nested run
    };

    // A5 effectiveness: identical profile → identical decisions → decline.
    // Customized: includes the ICs of every send site's key_klass
    // resolution, so a trap that warms a GRAFTED CALLEE's ICs (not the
    // root's) still flips the hash — see `snapshot_profile_customized`.
    let now_hash =
        crate::compiler::feedback::snapshot_profile_customized(vm, method, key_klass);
    if now_hash == old_hash {
        vm.stats.recompile_declined_ineffective += 1;
        if vm.options.trace.is_enabled("deopt") {
            eprintln!(
                "[deopt] recompile declined (profile unchanged) nm={} v{}",
                id.0, version
            );
        }
        let nm = vm.code_table.get_mut(id).expect("still present");
        nm.trap_count = 0; // re-check after another LIMIT traps
        return;
    }

    // Recompile against the warm feedback, then retire the old code. Order
    // matters (S13 D1): install the replacement FIRST so the key never points
    // at nothing; only then make the old nmethod NotEntrant.
    match crate::compiler::driver::compile_method_versioned(vm, key_klass, method, version + 1) {
        Some(new_id) => {
            vm.stats.recompiles += 1;
            if vm.options.trace.is_enabled("deopt") {
                eprintln!(
                    "[deopt] recompiled nm={} v{} -> nm={} v{} (trap storm)",
                    id.0,
                    version,
                    new_id.0,
                    version + 1
                );
            }
            // LAZY retirement (no §2c walk): this runs in the post-deopt
            // rt context where the native chain is not walkable, and the
            // replacement is semantically identical — see make_not_entrant_lazy.
            crate::codecache::flush::make_not_entrant_lazy(vm, id);
        }
        None => {
            // Ineligible NOW (feedback shifted under an eligibility rule, e.g.
            // a block-arg site going poly). The old code still runs correctly
            // via its deopts; stop re-trying (identical failure every time).
            let nm = vm.code_table.get_mut(id).expect("still present");
            nm.version = MAX_VERSIONS;
        }
    }
}
