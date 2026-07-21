//! S14 step 1: the type-feedback reader (SPEC §8.4). Reads one send site's
//! observed receiver types out of the interpreter IC side table (S3/S5,
//! SPEC §4.3) into a [`SiteFeedback`] the inliner (later steps) consumes to
//! decide whether to speculate/inline. This step reads the INTERPRETER side
//! only; the richer compiled-PIC source (per-entry counts) arrives once PICs
//! carry counter words (a later step), so the `prev` parameter is accepted but
//! not yet consulted.
//!
//! **Raw oops, not `Handle`s.** The sprint doc's `SiteFeedback` uses
//! `Handle<KlassOop>`; the as-built compiler runs `compile_method` in a NO-GC
//! window (`driver`'s own invariant: "no HandleScope needed, no GC can strike
//! mid-compile"), so a klass/method read here stays valid for the whole
//! compile — plain `KlassOop`/`MethodOop` are correct and simpler. If a later
//! step ever compiles across a collection, this becomes `Handle`s then.
//!
//! **Read-only** (layer table): takes `&VmState`, never patches or allocates.
//! An nmethod-id target is resolved with a local read-only method walk (the
//! `lookup` walk minus its `&mut` cache insert), not `runtime::lookup::lookup`.

use crate::bytecode::opcode::{decode_at, Instr};
use crate::codecache::nmethod::NmethodId;
use crate::interpreter::ic::InterpreterIc;
use crate::oops::layout::{IC_GUARD_MEGA, IC_GUARD_POLY, IC_POLY_MAX_PAIRS};
use crate::oops::method_dict::MethodDictOop;
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{ArrayOop, KlassOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

/// One send site's observed receivers (SPEC §8.4). The inliner maps this onto a
/// codegen decision: `Untaken` → uncommon trap (Self's lazy cold path);
/// `Mono` → speculate on the single klass; `Poly` → inline the dominant case
/// with a slow-path fallback; `Mega` → a plain dynamic send.
#[derive(Clone, Debug)]
pub enum SiteFeedback {
    /// IC still empty — the site never executed while interpreted.
    Untaken,
    Mono {
        klass: KlassOop,
        method: MethodOop,
    },
    /// Cases ordered by count when counts exist (compiled PIC, a later step),
    /// else by first-seen order — the interpreter POLY array is count-free
    /// (stride pinned by SPEC §4.3), so every `count` here is `None` for now.
    Poly {
        cases: Vec<FeedbackCase>,
    },
    Mega,
}

/// One (receiver klass → resolved method) observation, with an optional
/// execution count (`None` for the count-free interpreter POLY array).
#[derive(Clone, Debug)]
pub struct FeedbackCase {
    pub klass: KlassOop,
    pub method: MethodOop,
    pub count: Option<u32>,
}

/// Read the feedback for send site `ic_index` of `method` (SPEC §8.4). Source:
/// the interpreter IC side table. `prev` (the nmethod being replaced, whose
/// compiled PIC carries richer counts) is accepted for the eventual
/// source-priority rule but not yet consulted — PIC counter words are a later
/// S14 step.
pub fn read_send_site(
    vm: &VmState,
    method: MethodOop,
    ic_index: u16,
    prev: Option<NmethodId>,
) -> SiteFeedback {
    let _ = prev; // compiled-PIC source: later step
    let ic = InterpreterIc::at(method, ic_index);
    let guard = ic.guard();

    // Mega / Poly are smi-tagged guards (SPEC §4.3); Mono is a klassOop guard;
    // Empty is `nil`.
    if let Some(smi) = SmallInt::try_from(guard) {
        return match smi.value() {
            v if v == IC_GUARD_MEGA => SiteFeedback::Mega,
            v if v == IC_GUARD_POLY => read_poly(vm, ic),
            other => panic!("read_send_site: unrecognized IC guard smi {other}"),
        };
    }
    match KlassOop::try_from(guard) {
        Some(klass) => match resolve_target(vm, ic.target(), klass, ic.selector()) {
            Some(method) => SiteFeedback::Mono { klass, method },
            // A stale compiled-id whose (klass, selector) no longer resolves:
            // treat the site as never-taken — the trap re-dispatches against
            // the runtime truth. (Never speculate on unverifiable feedback.)
            None => SiteFeedback::Untaken,
        },
        None => SiteFeedback::Untaken, // guard == nil
    }
}

/// Walk the `[k1, m1, k2, m2, …]` pairs array (empty slots hold `nil` in the
/// key position — `KlassOop::try_from` rejects them, `ic::poly_arity`'s own
/// convention). First-seen order, counts `None`.
fn read_poly(vm: &VmState, ic: InterpreterIc) -> SiteFeedback {
    let pairs = ArrayOop::try_from(ic.target()).expect("poly IC target must be an Array");
    let mut cases = Vec::new();
    for i in 0..IC_POLY_MAX_PAIRS {
        let Some(klass) = KlassOop::try_from(pairs.at(2 * i)) else {
            break; // first empty slot: the rest are empty too
        };
        // Poly pairs only ever store interpreted MethodOops (`reverify_poly`
        // re-derives, never preserves a compiled id — ic.rs's own rule), but
        // resolve defensively anyway; a pair that no longer resolves is
        // dropped rather than speculated on.
        let Some(method) = resolve_target(vm, pairs.at(2 * i + 1), klass, ic.selector()) else {
            continue;
        };
        cases.push(FeedbackCase {
            klass,
            method,
            count: None,
        });
    }
    SiteFeedback::Poly { cases }
}

/// A mono/poly target is either a plain `MethodOop` (interpreter-resolved) or a
/// smi `NmethodId` (the site tiered up — `ic::set_mono_compiled`). Resolve both
/// to the underlying `MethodOop` via the SITE's own (guard klass, selector) —
/// never through the nmethod the stale id happens to name today:
///
/// An interpreter IC holding a compiled id heals only on its next DISPATCH
/// (send_generic's validity check); the COMPILER can read it any time after
/// the nmethod was invalidated, swept, or — the dangerous case — its table
/// slot REUSED for a different (klass, selector) entirely. Trusting the slot
/// blindly panicked on a freed id (the MACVM_DEOPT_STRESS sieve crash, S14
/// step 9) and, worse, would hand back the REUSED entry's method: wrong-method
/// feedback behind a passing klass guard is a silent miscompile. Resolving
/// through (klass, selector) is immune to both; `None` means the site has no
/// verifiable target and the caller must not speculate.
fn resolve_target(
    vm: &VmState,
    target: Oop,
    klass: KlassOop,
    selector: crate::oops::wrappers::SymbolOop,
) -> Option<MethodOop> {
    if let Some(m) = MethodOop::try_from(target) {
        return Some(m);
    }
    debug_assert!(
        SmallInt::try_from(target).is_some(),
        "mono IC target must be a MethodOop or an nmethod id"
    );
    resolve_method_ro(vm, klass, selector)
}

/// Read-only method lookup — the `runtime::lookup::lookup` walk minus its
/// `&mut` lookup-cache insert (this reader is `&VmState` by contract). Probes
/// each klass's `MethodDictOop` up the superclass chain to `nil`.
pub(crate) fn resolve_method_ro(
    vm: &VmState,
    klass: KlassOop,
    selector: SymbolOop,
) -> Option<MethodOop> {
    let nil = vm.universe.nil_obj;
    let mut k = klass;
    loop {
        if let Some(dict) = MethodDictOop::try_from(k.methods()) {
            if let Some(m) = dict.probe(vm, selector) {
                return Some(m);
            }
        }
        let sc = k.superclass();
        if sc.raw() == nil.raw() {
            return None;
        }
        k = KlassOop::try_from(sc).expect("resolve_method_ro: superclass field is not a klass");
    }
}

/// S14 step 8 (A5): a canonical digest of "what the feedback said" — FNV-1a
/// over every send site's IC lattice STATE TAG (Empty=0, Mono=1, Poly=2,
/// Mega=3), in bytecode order. Stored in the nmethod (`profile_hash`) at
/// compile time; the recompile-on-trap loop re-snapshots at trap time and
/// DECLINES the recompile when equal — the compiler would see the same states
/// and make the same decisions (Self's `checkEffectiveness`).
///
/// Deviation from the sprint doc (documented): the doc digests klass identity
/// SETS; state TAGS suffice for the storm-closer (the storm transition IS
/// `Untaken → Mono`, and a guard storm is `Mono → Poly` — both tag-visible).
/// A klass-set-preserving change (same-tag re-targeting) is invisible here,
/// but redefinition already invalidates through the dependency index.
pub fn snapshot_profile(_vm: &VmState, method: MethodOop) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    snapshot_into(method, &mut h, 0);
    h
}

/// The CUSTOMIZED profile hash: [`snapshot_profile`] plus, for every send
/// site in the root method, the ICs of the method that site resolves to
/// against `key_klass` — the same resolution B3 self-devirt grafting uses.
///
/// Why the plain hash was not enough, found on deltablue: `markInputs:`
/// (one method on AbstractConstraint) crossed the compile threshold on
/// EqualityConstraint receivers, so when a ScaleConstraint FIRST received
/// it, the customized compile happened immediately — grafting
/// `ScaleConstraint>>inputsDo:`, a method that had never executed. Every
/// send in that graft lowered as an Empty-IC trap, by design ("first call
/// is the trap"). The traps then warmed `inputsDo:`'s OWN ICs — but the
/// effectiveness check hashed only the root method and its literal
/// blocks, saw no change, and declined the recompile forever: a permanent
/// deopt storm (141k traps in a 100-solve run at threshold=20, the whole
/// gap between t=20's 103ms and t=1000's 55ms). S24 B5 5b fixed exactly
/// this for grafted BLOCKS, on this same benchmark; this is the method
/// flavor.
///
/// Deliberately OVER-inclusive rather than a faithful replay of the
/// inliner's decisions: every send site is resolved against `key_klass`,
/// whether or not it is a self-send and whether or not the inliner took
/// it. Both producers of the hash (the compile, and the decline check)
/// compute it identically, so determinism holds; the cost of including a
/// callee the graft ignored is at worst one extra recompile — and this
/// hash is only ever CONSULTED while the method is already storming,
/// where an extra recompile attempt is precisely the desired behaviour
/// (`MAX_VERSIONS` still caps thrash). Duplicating the inliner's real
/// policy here would be exact today and silently wrong after its next
/// change.
pub fn snapshot_profile_customized(
    vm: &mut VmState,
    method: MethodOop,
    key_klass: crate::oops::wrappers::KlassOop,
) -> u64 {
    let mut h = snapshot_profile(vm, method);
    let fnv = |h: &mut u64, byte: u8| {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    let len = method.bytecode_len();
    let mut bci = 0;
    while bci < len {
        let (instr, next) = decode_at(method, bci);
        if let Instr::Send { ic, super_: false } = instr {
            let ic_view = crate::interpreter::ic::InterpreterIc::at(method, ic);
            if let Some(callee) = crate::runtime::lookup::lookup(vm, key_klass, ic_view.selector())
            {
                // Never the root itself (a self-recursive send would loop).
                if callee.oop().raw() != method.oop().raw() {
                    fnv(&mut h, 0xC7);
                    snapshot_into(callee, &mut h, 1);
                }
            }
        }
        bci = next;
    }
    h
}

/// One method's (or block's) send-IC states folded into `h`, recursing into
/// literal blocks (S24 B5 5b): a compile GRAFTS block bodies inline and reads
/// THEIR ICs for lowering decisions, so a trap inside a grafted block that
/// warms only the block's own IC must still flip the profile hash — before
/// this, `note_uncommon_trap` hashed the root method alone and declined the
/// recompile forever ("profile unchanged"), a permanent deopt storm on
/// deltablue's constraintsConsuming:do: once multi-BB blockarg splicing
/// landed. `depth` seeds each nesting level so identical IC layouts at
/// different levels don't cancel.
fn snapshot_into(method: MethodOop, h: &mut u64, depth: u8) {
    use crate::interpreter::ic::{ic_state, IcState};
    let fnv = |h: &mut u64, byte: u8| {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    fnv(h, 0xB5);
    fnv(h, depth);
    let len = method.bytecode_len();
    let mut bci = 0;
    while bci < len {
        let (instr, next) = decode_at(method, bci);
        match instr {
            Instr::Send { ic, super_: false } => {
                let tag: u8 = match ic_state(method, ic) {
                    IcState::Empty => 0,
                    IcState::Mono => 1,
                    IcState::Poly(_) => 2,
                    IcState::Mega => 3,
                };
                fnv(h, ic as u8);
                fnv(h, (ic >> 8) as u8);
                fnv(h, tag);
            }
            Instr::PushClosure { lit, .. } => {
                if let Some(blk) = MethodOop::try_from(method.literals().at(lit as usize)) {
                    snapshot_into(blk, h, depth.saturating_add(1));
                }
            }
            _ => {}
        }
        bci = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::builder::BytecodeBuilder;
    use crate::oops::layout::IC_POLY_ARRAY_LEN;
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

    /// A host method with exactly one send site (IC index 0) whose feedback the
    /// tests set then read back.
    fn host_with_send(vm: &mut VmState) -> MethodOop {
        let sel = vm.universe.intern(b"foo:");
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.push_self();
        b.send(vm, sel, 1);
        b.ret_tos();
        let m_sel = vm.universe.intern(b"host:");
        b.finish(vm, m_sel, 1, 0)
    }

    /// A trivial `MethodOop` to use as an IC target (its body is never run).
    fn a_method(vm: &mut VmState, name: &[u8]) -> MethodOop {
        let sel = vm.universe.intern(name);
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        b.finish(vm, sel, 0, 0)
    }

    #[test]
    fn reads_untaken_from_empty_ic() {
        let mut vm = test_vm();
        let host = host_with_send(&mut vm);
        assert!(matches!(
            read_send_site(&vm, host, 0, None),
            SiteFeedback::Untaken
        ));
    }

    #[test]
    fn reads_mono() {
        let mut vm = test_vm();
        let host = host_with_send(&mut vm);
        let klass = vm.universe.smi_klass;
        let target = a_method(&mut vm, b"target");
        let epoch = vm.ic_epoch;
        InterpreterIc::at(host, 0).set_mono(&mut vm, klass, target, epoch);
        match read_send_site(&vm, host, 0, None) {
            SiteFeedback::Mono {
                klass: k,
                method: m,
            } => {
                assert_eq!(k.oop().raw(), klass.oop().raw());
                assert_eq!(m.oop().raw(), target.oop().raw());
            }
            other => panic!("expected Mono, got {other:?}"),
        }
    }

    #[test]
    fn reads_mega() {
        let mut vm = test_vm();
        let host = host_with_send(&mut vm);
        let nil = vm.universe.nil_obj;
        InterpreterIc::at(host, 0).set_mega(&mut vm, nil);
        assert!(matches!(
            read_send_site(&vm, host, 0, None),
            SiteFeedback::Mega
        ));
    }

    /// The interpreter POLY array is count-free and first-seen ordered.
    #[test]
    fn reads_poly_first_seen_order_count_free() {
        let mut vm = test_vm();
        let host = host_with_send(&mut vm);
        let k1 = vm.universe.smi_klass;
        let k2 = vm.universe.boolean_klass;
        let m1 = a_method(&mut vm, b"m1");
        let m2 = a_method(&mut vm, b"m2");
        let array_klass = vm.universe.array_klass;
        // Fill a fresh pairs array [k1, m1, k2, m2, nil, …] AFTER the last
        // allocation (m2), so nothing moves it before the raw fills below.
        let pairs =
            crate::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, IC_POLY_ARRAY_LEN);
        pairs.at_put(0, k1.oop());
        pairs.at_put(1, m1.oop());
        pairs.at_put(2, k2.oop());
        pairs.at_put(3, m2.oop());
        let epoch = vm.ic_epoch;
        InterpreterIc::at(host, 0).set_poly(&mut vm, pairs, epoch);

        match read_send_site(&vm, host, 0, None) {
            SiteFeedback::Poly { cases } => {
                assert_eq!(cases.len(), 2, "two occupied pairs, rest empty");
                assert_eq!(
                    cases[0].klass.oop().raw(),
                    k1.oop().raw(),
                    "first-seen order"
                );
                assert_eq!(cases[0].method.oop().raw(), m1.oop().raw());
                assert!(
                    cases[0].count.is_none(),
                    "interpreter POLY carries no counts"
                );
                assert_eq!(cases[1].klass.oop().raw(), k2.oop().raw());
                assert_eq!(cases[1].method.oop().raw(), m2.oop().raw());
            }
            other => panic!("expected Poly, got {other:?}"),
        }
    }
}
