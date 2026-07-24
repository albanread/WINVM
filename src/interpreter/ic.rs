//! The per-send-site inline cache: the 4-slot `InterpreterIc` view onto a
//! `CompiledMethod`'s `ics` array, and the state-lattice transition table
//! (SPEC §4.3, §6.2, `sprint_s03_detail.md` §Algorithms-2 — the 13-row
//! table this module implements one arm per row). Bytecode is immutable
//! (SPEC §4, Δ): all dispatch state lives here, in ordinary heap `Array`s
//! GC can scan for free — never in the bytecode stream.

use crate::codecache::nmethod::NmethodId;
use crate::oops::layout::{
    IC_GUARD_MEGA, IC_GUARD_OFFSET, IC_GUARD_POLY, IC_META_ARGC_MASK, IC_META_ARGC_SHIFT,
    IC_META_EPOCH_MASK, IC_META_EPOCH_SHIFT, IC_META_OFFSET, IC_POLY_ARRAY_LEN, IC_POLY_MAX_PAIRS,
    IC_SEL_OFFSET, IC_STRIDE, IC_TARGET_OFFSET,
};
use crate::oops::smi::SmallInt;
use crate::oops::wrappers::{ArrayOop, KlassOop, MethodOop, SymbolOop};
use crate::oops::Oop;
use crate::runtime::vm_state::VmState;

/// A `Copy` view onto one 4-word IC site. NOTE (S7): the cached `ics`
/// field is an `ArrayOop` — an address — so this view DOES go stale
/// across any allocation, exactly like a cached field value (the original
/// S3-era comment claimed otherwise; that predates a moving GC).
/// Re-derive via [`InterpreterIc::at`] after anything that can scavenge.
#[derive(Copy, Clone)]
pub struct InterpreterIc {
    pub ics: ArrayOop,
    pub base: usize,
}

impl InterpreterIc {
    pub fn at(method: MethodOop, ic_idx: u16) -> InterpreterIc {
        InterpreterIc {
            ics: method.ics(),
            base: ic_idx as usize * IC_STRIDE,
        }
    }

    pub fn selector(&self) -> SymbolOop {
        SymbolOop::try_from(self.ics.at(self.base + IC_SEL_OFFSET))
            .expect("InterpreterIc::selector: sel slot is not a Symbol")
    }

    fn meta(&self) -> i64 {
        SmallInt::try_from(self.ics.at(self.base + IC_META_OFFSET))
            .expect("InterpreterIc::meta: meta slot is not a smi")
            .value()
    }

    pub fn argc(&self) -> u8 {
        ((self.meta() & IC_META_ARGC_MASK) >> IC_META_ARGC_SHIFT) as u8
    }

    pub fn epoch(&self) -> u32 {
        ((self.meta() & IC_META_EPOCH_MASK) >> IC_META_EPOCH_SHIFT) as u32
    }

    fn set_meta(&self, epoch: u32) {
        let v =
            ((self.argc() as i64) << IC_META_ARGC_SHIFT) | ((epoch as i64) << IC_META_EPOCH_SHIFT);
        self.ics
            .at_put(self.base + IC_META_OFFSET, SmallInt::new(v).oop());
    }

    pub fn guard(&self) -> Oop {
        self.ics.at(self.base + IC_GUARD_OFFSET)
    }

    pub fn target(&self) -> Oop {
        self.ics.at(self.base + IC_TARGET_OFFSET)
    }

    // Every setter below takes `vm` and ends with a post-write barrier:
    // a method's ics array is as long-lived as the method itself, so by
    // the time an IC transitions it is routinely OLD while the klass/
    // method/pairs being written are young — raw `at_put`s here were one
    // of the unbarriered old→new writer families the A9 verifier caught
    // under `MACVM_GC_STRESS=1` (S7-10).

    pub fn set_mono(&self, vm: &mut VmState, k: KlassOop, m: MethodOop, epoch: u32) {
        self.ics.at_put(self.base + IC_GUARD_OFFSET, k.oop());
        self.ics.at_put(self.base + IC_TARGET_OFFSET, m.oop());
        self.set_meta(epoch);
        crate::memory::store::post_write_barrier(vm, self.ics.as_mem());
    }

    /// S10 D4: still mono state (SPEC §4.3's lattice gains no new top-level
    /// state for this) — same guard write as [`Self::set_mono`], but the
    /// target is a smi `NmethodId` instead of a `MethodOop`'s oop. The send
    /// fast path's `target.is_smi()` check is what tells the two apart at
    /// dispatch time.
    pub fn set_mono_compiled(&self, vm: &mut VmState, k: KlassOop, id: NmethodId, epoch: u32) {
        self.ics.at_put(self.base + IC_GUARD_OFFSET, k.oop());
        self.ics.at_put(
            self.base + IC_TARGET_OFFSET,
            SmallInt::new(id.0 as i64).oop(),
        );
        self.set_meta(epoch);
        crate::memory::store::post_write_barrier(vm, self.ics.as_mem());
    }

    pub fn set_poly(&self, vm: &mut VmState, pairs: ArrayOop, epoch: u32) {
        self.ics.at_put(
            self.base + IC_GUARD_OFFSET,
            SmallInt::new(IC_GUARD_POLY).oop(),
        );
        self.ics.at_put(self.base + IC_TARGET_OFFSET, pairs.oop());
        self.set_meta(epoch);
        crate::memory::store::post_write_barrier(vm, self.ics.as_mem());
    }

    /// Mega is a sink: no transition back exists in v1, so its epoch is
    /// left as-is (never read once mega — rows 12/13 never check it).
    /// `nil` must be passed explicitly (SPEC's pseudocode elides it):
    /// `InterpreterIc` carries no `Universe` handle, so the caller (which
    /// has a `&VmState`) supplies it.
    pub fn set_mega(&self, vm: &mut VmState, nil: Oop) {
        self.ics.at_put(
            self.base + IC_GUARD_OFFSET,
            SmallInt::new(IC_GUARD_MEGA).oop(),
        );
        self.ics.at_put(self.base + IC_TARGET_OFFSET, nil);
        crate::memory::store::post_write_barrier(vm, self.ics.as_mem());
    }

    pub fn set_empty(&self, vm: &mut VmState, nil: Oop) {
        self.ics.at_put(self.base + IC_GUARD_OFFSET, nil);
        self.ics.at_put(self.base + IC_TARGET_OFFSET, nil);
        crate::memory::store::post_write_barrier(vm, self.ics.as_mem());
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum IcState {
    Empty,
    Mono,
    Poly(u8),
    Mega,
}

/// Index of the first unoccupied pair (empty pairs hold `nil` in the key
/// slot; `KlassOop::try_from` rejects `nil` since its klass's format is
/// `Slots`, never `Klass` — no explicit `nil` comparison needed).
fn poly_arity(pairs: ArrayOop) -> u8 {
    let mut n = 0u8;
    for i in 0..IC_POLY_MAX_PAIRS {
        if KlassOop::try_from(pairs.at(2 * i)).is_none() {
            break;
        }
        n += 1;
    }
    n
}

/// Arm `i`'s hit count from the pairs array's count tail (see
/// `IC_POLY_ARRAY_LEN`): a smi, with `nil` (fresh slot, or one cleared by
/// `reverify_poly`) reading as 0.
pub fn poly_count_at(pairs: ArrayOop, i: usize) -> i64 {
    SmallInt::try_from(pairs.at(2 * IC_POLY_MAX_PAIRS + i)).map_or(0, |s| s.value())
}

/// Row-7 profiling: saturating bump of arm `i`'s count. A smi store into an
/// existing slot — no barrier, no allocation, nothing for GC to see.
fn poly_count_bump(pairs: ArrayOop, i: usize) {
    let c = poly_count_at(pairs, i);
    if c < i32::MAX as i64 {
        pairs.at_put(2 * IC_POLY_MAX_PAIRS + i, SmallInt::new(c + 1).oop());
    }
}

/// Test + S14 feedback readout (SPEC §8.4): reconstructs the lattice state
/// from the raw guard/target slots.
pub fn ic_state(method: MethodOop, ic_idx: u16) -> IcState {
    let ic = InterpreterIc::at(method, ic_idx);
    let guard = ic.guard();
    if let Some(smi) = SmallInt::try_from(guard) {
        return match smi.value() {
            v if v == IC_GUARD_MEGA => IcState::Mega,
            v if v == IC_GUARD_POLY => {
                let pairs =
                    ArrayOop::try_from(ic.target()).expect("ic_state: poly target is not an Array");
                IcState::Poly(poly_arity(pairs))
            }
            other => panic!("ic_state: unrecognized guard smi {other}"),
        };
    }
    if KlassOop::try_from(guard).is_some() {
        return IcState::Mono;
    }
    IcState::Empty // guard == nil
}

/// `L(k)` (SPEC §5.3 step 7): for a super site, the lookup start is
/// **static per site** — `method.holder().superclass()` — regardless of
/// which receiver klass triggered the miss; for a normal site it is the
/// receiver klass itself. `None` for a super site whose holder has no
/// superclass (only `Object`, which installs no super sites in practice).
fn resolve(
    vm: &mut VmState,
    caller: MethodOop,
    selector: SymbolOop,
    is_super: bool,
    rcvr_klass: KlassOop,
) -> Option<MethodOop> {
    if is_super {
        let holder = KlassOop::try_from(caller.holder())
            .expect("resolve: super send from a method with no installed holder");
        let super_klass = KlassOop::try_from(holder.superclass())?;
        crate::runtime::lookup::lookup(vm, super_klass, selector)
    } else {
        crate::runtime::lookup::lookup(vm, rcvr_klass, selector)
    }
}

fn alloc_poly_pairs(vm: &mut VmState) -> ArrayOop {
    let array_klass = vm.universe.array_klass;
    crate::memory::alloc::alloc_indexable_oops(vm, array_klass, IC_POLY_ARRAY_LEN)
}

/// Row 8: each `(k_i, m_i)` pair is re-resolved via `L(k_i)`; a pair whose
/// lookup now fails is dropped and the survivors compacted left. Mutates
/// `pairs` in place (an already-allocated, already-reachable heap object —
/// no allocation here, so no ordering hazard).
fn reverify_poly(
    vm: &mut VmState,
    caller: MethodOop,
    selector: SymbolOop,
    is_super: bool,
    pairs: ArrayOop,
) -> ArrayOop {
    let arity = poly_arity(pairs);
    let mut kept: Vec<(Oop, Oop, i64)> = Vec::with_capacity(arity as usize);
    for i in 0..arity as usize {
        let k = pairs.at(2 * i);
        let kk = KlassOop::try_from(k).expect("reverify_poly: pair key is not a KlassOop");
        if let Some(m) = resolve(vm, caller, selector, is_super, kk) {
            // The count travels with its arm through compaction — dropping
            // it would let one redefinition erase a hot arm's dominance.
            kept.push((k, m.oop(), poly_count_at(pairs, i)));
        }
    }
    let nil = vm.universe.nil_obj;
    for (i, &(k, m, c)) in kept.iter().enumerate() {
        pairs.at_put(2 * i, k);
        pairs.at_put(2 * i + 1, m);
        pairs.at_put(2 * IC_POLY_MAX_PAIRS + i, SmallInt::new(c).oop());
    }
    for i in kept.len()..IC_POLY_MAX_PAIRS {
        pairs.at_put(2 * i, nil);
        pairs.at_put(2 * i + 1, nil);
        pairs.at_put(2 * IC_POLY_MAX_PAIRS + i, nil);
    }
    // `pairs` is as long-lived as its IC — routinely old by reverify time,
    // while the re-resolved methods may be young (S7-10).
    crate::memory::store::post_write_barrier(vm, pairs.as_mem());
    pairs
}

/// Handles every non-fast-path case of the 13-row transition table. The
/// row-3 fast hit (mono, same klass, epoch current) is handled by the
/// caller (`interpreter::send::send_generic`) before this is ever called —
/// it never writes the IC, so routing it here too would be harmless but
/// wasteful on the hottest path. Returns the method to activate, or `None`
/// ⇒ the caller runs the DNU path. Never activates anything itself.
pub fn ic_transition(
    vm: &mut VmState,
    caller: MethodOop,
    ic_idx: u16,
    rcvr_klass: KlassOop,
    is_super: bool,
) -> Option<MethodOop> {
    let ic = InterpreterIc::at(caller, ic_idx);
    let nil = vm.universe.nil_obj;
    let epoch = vm.ic_epoch;
    let selector = ic.selector();
    let guard = ic.guard();

    // Rows 1, 2: empty.
    if guard.raw() == nil.raw() {
        return match resolve(vm, caller, selector, is_super, rcvr_klass) {
            Some(m) => {
                ic.set_mono(vm, rcvr_klass, m, epoch);
                Some(m)
            }
            None => None,
        };
    }

    // Mono states: rows 4, 5, 6.
    if let Some(k0) = KlassOop::try_from(guard) {
        if k0.oop().raw() == rcvr_klass.oop().raw() {
            // Row 4: same klass, stale epoch (the fresh-epoch case is the
            // row-3 fast path the caller already handled).
            return match resolve(vm, caller, selector, is_super, k0) {
                Some(m) => {
                    ic.set_mono(vm, k0, m, epoch);
                    Some(m)
                }
                None => {
                    ic.set_empty(vm, nil);
                    None
                }
            };
        }
        // Rows 5, 6: different klass.
        return match resolve(vm, caller, selector, is_super, rcvr_klass) {
            Some(m) => {
                // The pairs allocation below can scavenge, which moves
                // BOTH the values this arm still needs (`rcvr_klass`, `m`
                // — handle-protected) AND the ics array the cached `ic`
                // view points into (`InterpreterIc` caches an `ArrayOop`,
                // which is an address like any other — the S3-era claim
                // that the view "never goes stale" predates a moving GC).
                // `ic` is re-derived afterwards through `vm.regs.method`,
                // a scanned root that names this same caller; k0/m0 are
                // then read through the FRESH view — reading them through
                // the stale one would hand back from-space images whose
                // body slots hold pre-scavenge addresses.
                let scope = crate::memory::handles::HandleScope::enter(vm);
                let rcvr_klass_h = scope.handle(vm, rcvr_klass);
                let m_h = scope.handle(vm, m);
                // S15 (BUG A, found by the Richards port): `selector` must be
                // handle-protected across the allocation too — it is read
                // AFTER `alloc_poly_pairs` by the compiled-id re-resolve
                // below, and a scavenge moves a YOUNG symbol (any selector
                // interned by freshly loaded code). Probing the method
                // dictionary with the stale pre-move address misses every
                // entry, so the expect below fired as "klass k0 must still
                // resolve selector" — an abort, since the whole transition
                // can run inside rt_uncommon_trap's no-unwind extern "C"
                // frame. World selectors survive genesis GCs into old space,
                // which is why only new benchmark code ever hit this.
                let selector_h = scope.handle(vm, selector.oop());

                let pairs = alloc_poly_pairs(vm);

                let selector = SymbolOop::try_from(selector_h.get(vm))
                    .expect("ic_transition: selector survived the pairs allocation");
                let caller = vm.regs.method.expect("ic_transition: no active method");
                let ic = InterpreterIc::at(caller, ic_idx);
                let k0 =
                    KlassOop::try_from(ic.guard()).expect("ic_transition: guard changed under us");
                // S10 D4: the mono entry being demoted to poly may itself
                // be compiled (`set_mono_compiled`'s smi target, not a
                // MethodOop) — the poly array only ever stores interpreted
                // MethodOops (`reverify_poly` always re-derives via a
                // fresh `resolve`, never preserves a compiled id), so
                // re-derive k0's interpreted method the same way rather
                // than reading the smi as if it were one.
                let m0 = match MethodOop::try_from(ic.target()) {
                    Some(m0) => m0,
                    None => resolve(vm, caller, selector, is_super, k0).expect(
                        "ic_transition: klass k0 must still resolve selector \
                         (it did when its compiled entry was installed)",
                    ),
                };
                let m = m_h.get(vm);
                pairs.at_put(0, k0.oop());
                pairs.at_put(1, m0.oop());
                pairs.at_put(2, rcvr_klass_h.get(vm).oop());
                pairs.at_put(3, m.oop());
                ic.set_poly(vm, pairs, epoch);
                Some(m)
            }
            None => None,
        };
    }

    // Poly/mega states: guard is a smi.
    let guard_smi = SmallInt::try_from(guard)
        .expect("ic_transition: guard is neither nil, a KlassOop, nor a smi");
    match guard_smi.value() {
        v if v == IC_GUARD_POLY => {
            let mut pairs = ArrayOop::try_from(ic.target())
                .expect("ic_transition: poly target is not an Array");
            if ic.epoch() != epoch {
                // Row 8: reverify ALL pairs, then restamp — never restamp
                // after healing only the hit pair (stale-dispatch bug).
                pairs = reverify_poly(vm, caller, selector, is_super, pairs);
                if poly_arity(pairs) == 0 {
                    ic.set_empty(vm, nil);
                    return match resolve(vm, caller, selector, is_super, rcvr_klass) {
                        Some(m) => {
                            ic.set_mono(vm, rcvr_klass, m, epoch);
                            Some(m)
                        }
                        None => None,
                    };
                }
                ic.set_poly(vm, pairs, epoch);
            }
            let arity = poly_arity(pairs);
            for i in 0..arity as usize {
                if pairs.at(2 * i).raw() == rcvr_klass.oop().raw() {
                    // Row 7 — and the profiling tier's one job (dart124
                    // items 2+3): count the hit so `feedback::read_poly`
                    // can order cases by measured dominance.
                    poly_count_bump(pairs, i);
                    return MethodOop::try_from(pairs.at(2 * i + 1));
                }
            }
            match resolve(vm, caller, selector, is_super, rcvr_klass) {
                Some(m) => {
                    if (arity as usize) < IC_POLY_MAX_PAIRS {
                        pairs.at_put(2 * arity as usize, rcvr_klass.oop());
                        pairs.at_put(2 * arity as usize + 1, m.oop());
                        // `pairs` itself may be old, the new pair young.
                        crate::memory::store::post_write_barrier(vm, pairs.as_mem());
                        ic.set_poly(vm, pairs, epoch); // row 9
                    } else {
                        ic.set_mega(vm, nil); // row 10
                    }
                    Some(m)
                }
                None => None, // row 11
            }
        }
        v if v == IC_GUARD_MEGA => {
            // Rows 12, 13: mega is a sink, never writes.
            resolve(vm, caller, selector, is_super, rcvr_klass)
        }
        other => panic!("ic_transition: unrecognized guard smi {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeBuilder;
    use crate::oops::layout::HEADER_WORDS;
    use crate::oops::smi::SmallInt;
    use crate::runtime::lookup::install_method;
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

    /// A klass directly under `Object`, with its own fresh metaclass.
    fn new_klass(vm: &mut VmState, name: &str) -> KlassOop {
        let object_klass = vm.universe.object_klass;
        vm.universe.new_klass(
            object_klass,
            name,
            crate::oops::Format::Slots,
            false,
            HEADER_WORDS,
        )
    }

    /// A trivial unary (argc=0) method returning the smi `tag` — used so
    /// distinct method identities are also distinguishable by their
    /// *behavior* when sent. Every `build_caller` in this module sends with
    /// argc=0 (unary), and `return_from_frame` reads the *callee's own*
    /// declared argc to locate the receiver slot to overwrite — it must
    /// match the argc actually used to push the frame, not just happen to
    /// have a body that never reads temps.
    fn method_returning(vm: &mut VmState, name: &str, tag: i64) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        b.push_smi_i8(tag as i8);
        b.ret_tos();
        let sel = vm.universe.intern(name.as_bytes());
        b.finish(vm, sel, 0, 0)
    }

    /// Installs a harmless `#doesNotUnderstand:` stub on `Object` so DNU
    /// sends resolve to it instead of `error::dnu_fallback`'s
    /// `std::process::exit(1)` — every row-2/6/11 test needs this.
    fn install_stub_dnu(vm: &mut VmState) {
        let object_klass = vm.universe.object_klass;
        let sel = vm.universe.sel_does_not_understand;
        let mut b = BytecodeBuilder::new();
        b.push_smi_i8(-1);
        b.ret_tos();
        let handler_name = vm.universe.intern(b"dnuStub");
        let m = b.finish(vm, handler_name, 1, 0);
        install_method(vm, object_klass, sel, m);
    }

    /// A caller method with exactly one send site (`ic_idx == 0`):
    /// `push_temp(0..=argc)` (receiver + args, all unified arg/temp slots of
    /// the caller itself), `send(sel, argc)`, `ret_tos`.
    fn build_caller(vm: &mut VmState, sel: SymbolOop, argc: u8) -> MethodOop {
        let mut b = BytecodeBuilder::new();
        for t in 0..=argc {
            b.push_temp(t);
        }
        b.send(vm, sel, argc);
        b.ret_tos();
        let name = vm.universe.intern(b"caller");
        b.finish(vm, name, (argc + 1) as usize, 0)
    }

    fn send_once(vm: &mut VmState, caller: MethodOop, args: &[Oop]) -> Oop {
        let dummy_self = vm.universe.nil_obj;
        crate::interpreter::run_method(vm, caller, dummy_self, args)
    }

    /// The 4 raw IC-site words, for exact "must not write" assertions —
    /// reads straight off the heap `Array`, not through any accessor that
    /// might itself normalize a value.
    fn ic_words(caller: MethodOop, ic_idx: u16) -> [Oop; 4] {
        let ic = InterpreterIc::at(caller, ic_idx);
        [
            ic.ics.at(ic.base),
            ic.ics.at(ic.base + 1),
            ic.ics.at(ic.base + 2),
            ic.ics.at(ic.base + 3),
        ]
    }

    #[test]
    fn ic_empty_to_mono() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let m = method_returning(&mut vm, "foo", 1);
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 0);

        assert_eq!(ic_state(caller, 0), IcState::Empty);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let result = send_once(&mut vm, caller, &[recv]);
        assert_eq!(result, SmallInt::new(1).oop());
        assert_eq!(ic_state(caller, 0), IcState::Mono);
    }

    #[test]
    fn ic_empty_dnu_untouched() {
        let mut vm = test_vm();
        install_stub_dnu(&mut vm);
        let sel = vm.universe.intern(b"nope");
        let a = new_klass(&mut vm, "A");
        let caller = build_caller(&mut vm, sel, 0);

        let before = ic_words(caller, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let result = send_once(&mut vm, caller, &[recv]);
        assert_eq!(
            result,
            SmallInt::new(-1).oop(),
            "must have run the DNU stub"
        );
        assert_eq!(ic_state(caller, 0), IcState::Empty);
        assert_eq!(ic_words(caller, 0), before, "row 2: IC must be untouched");
    }

    #[test]
    fn ic_mono_hit_no_write() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let m = method_returning(&mut vm, "foo", 1);
        install_method(&mut vm, a, sel, m);
        let caller = build_caller(&mut vm, sel, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();

        let _ = send_once(&mut vm, caller, &[recv]);
        assert_eq!(ic_state(caller, 0), IcState::Mono);
        let before = ic_words(caller, 0);
        let result = send_once(&mut vm, caller, &[recv]);
        assert_eq!(result, SmallInt::new(1).oop());
        assert_eq!(
            ic_words(caller, 0),
            before,
            "row 3: fast path must not write"
        );
    }

    #[test]
    fn ic_mono_selfheal() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let m1 = method_returning(&mut vm, "foo1", 1);
        install_method(&mut vm, a, sel, m1);
        let caller = build_caller(&mut vm, sel, 0);
        let recv = crate::memory::alloc::alloc_slots(&mut vm, a).oop();

        let r1 = send_once(&mut vm, caller, &[recv]);
        assert_eq!(r1, SmallInt::new(1).oop());
        assert_eq!(ic_state(caller, 0), IcState::Mono);

        // Redefine `foo` on the SAME klass — bumps the global epoch,
        // stamping this site's cached epoch stale.
        let m2 = method_returning(&mut vm, "foo2", 2);
        install_method(&mut vm, a, sel, m2);

        let r2 = send_once(&mut vm, caller, &[recv]);
        assert_eq!(
            r2,
            SmallInt::new(2).oop(),
            "must dispatch the redefined method"
        );
        assert_eq!(ic_state(caller, 0), IcState::Mono);
        let ic = InterpreterIc::at(caller, 0);
        assert_eq!(
            ic.epoch(),
            vm.ic_epoch,
            "must have restamped the current epoch"
        );
    }

    #[test]
    fn ic_mono_selfheal_to_empty() {
        // Row 4b, driven directly through `ic_transition` (as the sprint
        // doc's own design intends: "provide the transition logic as one
        // function so tests can drive it directly") — a dictionary can
        // only grow in v1 (no method removal), so there is no *realistic*
        // sequence of real sends that makes a previously-found selector
        // become unfindable; this hand-crafts the precondition instead.
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"neverInstalled");
        let a = new_klass(&mut vm, "A");
        let stale_method = method_returning(&mut vm, "stale", 9);
        let caller = build_caller(&mut vm, sel, 0);

        let ic = InterpreterIc::at(caller, 0);
        let stale_epoch = vm.ic_epoch;
        ic.set_mono(&mut vm, a, stale_method, stale_epoch);
        vm.ic_epoch += 1; // now stale relative to the IC's stamped epoch

        let result = ic_transition(&mut vm, caller, 0, a, false);
        assert_eq!(result, None, "L(a) is None: sel was never installed on a");
        assert_eq!(ic_state(caller, 0), IcState::Empty);
    }

    #[test]
    fn ic_mono_to_poly() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let b = new_klass(&mut vm, "B");
        let m_a = method_returning(&mut vm, "foo_a", 1);
        let m_b = method_returning(&mut vm, "foo_b", 2);
        install_method(&mut vm, a, sel, m_a);
        install_method(&mut vm, b, sel, m_b);
        let caller = build_caller(&mut vm, sel, 0);
        let recv_a = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let recv_b = crate::memory::alloc::alloc_slots(&mut vm, b).oop();

        assert_eq!(
            send_once(&mut vm, caller, &[recv_a]),
            SmallInt::new(1).oop()
        );
        assert_eq!(ic_state(caller, 0), IcState::Mono);
        assert_eq!(
            send_once(&mut vm, caller, &[recv_b]),
            SmallInt::new(2).oop()
        );
        assert_eq!(ic_state(caller, 0), IcState::Poly(2));

        let ic = InterpreterIc::at(caller, 0);
        let pairs = ArrayOop::try_from(ic.target()).unwrap();
        assert_eq!(pairs.at(0), a.oop());
        assert_eq!(pairs.at(1), m_a.oop());
        assert_eq!(pairs.at(2), b.oop());
        assert_eq!(pairs.at(3), m_b.oop());
    }

    #[test]
    fn ic_poly_append() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let klasses: Vec<KlassOop> = ["A", "B", "C", "D"]
            .iter()
            .map(|n| new_klass(&mut vm, n))
            .collect();
        for (i, &k) in klasses.iter().enumerate() {
            let m = method_returning(&mut vm, &format!("foo{i}"), i as i64);
            install_method(&mut vm, k, sel, m);
        }
        let caller = build_caller(&mut vm, sel, 0);

        for (i, &k) in klasses.iter().enumerate() {
            let recv = crate::memory::alloc::alloc_slots(&mut vm, k).oop();
            let result = send_once(&mut vm, caller, &[recv]);
            assert_eq!(result, SmallInt::new(i as i64).oop());
        }
        assert_eq!(ic_state(caller, 0), IcState::Poly(4));
    }

    #[test]
    fn ic_poly_to_mega() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let klasses: Vec<KlassOop> = ["A", "B", "C", "D", "E"]
            .iter()
            .map(|n| new_klass(&mut vm, n))
            .collect();
        for (i, &k) in klasses.iter().enumerate() {
            let m = method_returning(&mut vm, &format!("foo{i}"), i as i64);
            install_method(&mut vm, k, sel, m);
        }
        let caller = build_caller(&mut vm, sel, 0);

        for &k in &klasses {
            let recv = crate::memory::alloc::alloc_slots(&mut vm, k).oop();
            let _ = send_once(&mut vm, caller, &[recv]);
        }
        assert_eq!(ic_state(caller, 0), IcState::Mega);
        let ic = InterpreterIc::at(caller, 0);
        assert_eq!(ic.target(), vm.universe.nil_obj);
    }

    #[test]
    fn ic_mega_no_writes() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let klasses: Vec<KlassOop> = ["A", "B", "C", "D", "E", "F"]
            .iter()
            .map(|n| new_klass(&mut vm, n))
            .collect();
        for (i, &k) in klasses.iter().enumerate() {
            let m = method_returning(&mut vm, &format!("foo{i}"), i as i64);
            install_method(&mut vm, k, sel, m);
        }
        let caller = build_caller(&mut vm, sel, 0);
        for &k in &klasses[..5] {
            let recv = crate::memory::alloc::alloc_slots(&mut vm, k).oop();
            let _ = send_once(&mut vm, caller, &[recv]);
        }
        assert_eq!(ic_state(caller, 0), IcState::Mega);
        let before = ic_words(caller, 0);
        for &k in &klasses[..6] {
            let recv = crate::memory::alloc::alloc_slots(&mut vm, k).oop();
            let _ = send_once(&mut vm, caller, &[recv]);
        }
        assert_eq!(ic_words(caller, 0), before, "mega must never write the IC");
    }

    #[test]
    fn ic_dnu_rows_untouched() {
        let mut vm = test_vm();
        install_stub_dnu(&mut vm);
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let b = new_klass(&mut vm, "B");
        let m_a = method_returning(&mut vm, "foo_a", 1);
        install_method(&mut vm, a, sel, m_a);
        let caller = build_caller(&mut vm, sel, 0);
        let recv_a = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let recv_b = crate::memory::alloc::alloc_slots(&mut vm, b).oop(); // no `foo` on B

        let _ = send_once(&mut vm, caller, &[recv_a]);
        assert_eq!(ic_state(caller, 0), IcState::Mono);
        let before = ic_words(caller, 0);
        let result = send_once(&mut vm, caller, &[recv_b]); // row 6: DNU, IC untouched
        assert_eq!(result, SmallInt::new(-1).oop());
        assert_eq!(ic_state(caller, 0), IcState::Mono);
        assert_eq!(ic_words(caller, 0), before);
    }

    #[test]
    fn ic_poly_reverify_all() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let b = new_klass(&mut vm, "B");
        let c = new_klass(&mut vm, "C");
        let m_a = method_returning(&mut vm, "foo_a", 1);
        let m_b1 = method_returning(&mut vm, "foo_b1", 2);
        let m_c = method_returning(&mut vm, "foo_c", 3);
        install_method(&mut vm, a, sel, m_a);
        install_method(&mut vm, b, sel, m_b1);
        install_method(&mut vm, c, sel, m_c);
        let caller = build_caller(&mut vm, sel, 0);
        let recv_a = crate::memory::alloc::alloc_slots(&mut vm, a).oop();
        let recv_b = crate::memory::alloc::alloc_slots(&mut vm, b).oop();
        let recv_c = crate::memory::alloc::alloc_slots(&mut vm, c).oop();

        let _ = send_once(&mut vm, caller, &[recv_a]);
        let _ = send_once(&mut vm, caller, &[recv_b]);
        let _ = send_once(&mut vm, caller, &[recv_c]);
        assert_eq!(ic_state(caller, 0), IcState::Poly(3));

        // Redefine ONLY b's method — bumps the global epoch, staling every
        // pair's stamped epoch, not just b's.
        let m_b2 = method_returning(&mut vm, "foo_b2", 20);
        install_method(&mut vm, b, sel, m_b2);
        let epoch_after_redefine = vm.ic_epoch;

        // Send to `a` — a DIFFERENT pair than the redefined one — must
        // trigger row 8's all-pairs reverify before re-probing.
        let r = send_once(&mut vm, caller, &[recv_a]);
        assert_eq!(r, SmallInt::new(1).oop());
        assert_eq!(ic_state(caller, 0), IcState::Poly(3));
        let ic = InterpreterIc::at(caller, 0);
        assert_eq!(ic.epoch(), epoch_after_redefine, "must have restamped");

        // b's pair, never directly sent to since the redefinition, must
        // already reflect the new method.
        let pairs = ArrayOop::try_from(ic.target()).unwrap();
        let b_slot = (0..3)
            .find(|&i| pairs.at(2 * i) == b.oop())
            .expect("b's pair must survive reverification");
        assert_eq!(pairs.at(2 * b_slot + 1), m_b2.oop());
    }

    /// dart124 items 2+3 substrate: row-7 hits bump the count tail, and
    /// reverification carries each surviving arm's count through compaction
    /// while clearing the dropped tail — one redefinition must not erase a
    /// hot arm's measured dominance, and a dead arm's count must not leak
    /// into whichever arm compacts into its slot.
    #[test]
    fn ic_poly_counts_bump_and_survive_reverify() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let dead = new_klass(&mut vm, "Dead");
        let m_a = method_returning(&mut vm, "foo_a", 1);
        install_method(&mut vm, a, sel, m_a);
        let caller = build_caller(&mut vm, sel, 0);

        let stale_method = method_returning(&mut vm, "stale", 99);
        let array_klass = vm.universe.array_klass;
        let pairs =
            crate::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, IC_POLY_ARRAY_LEN);
        // dead first, a second: compaction must MOVE a's count left.
        pairs.at_put(0, dead.oop());
        pairs.at_put(1, stale_method.oop());
        pairs.at_put(2, a.oop());
        pairs.at_put(3, m_a.oop());
        for _ in 0..3 {
            poly_count_bump(pairs, 1); // a's arm
        }
        poly_count_bump(pairs, 0); // dead's arm
        assert_eq!(poly_count_at(pairs, 0), 1);
        assert_eq!(poly_count_at(pairs, 1), 3);

        let pairs = reverify_poly(&mut vm, caller, sel, false, pairs);
        assert_eq!(poly_arity(pairs), 1, "dead pair dropped");
        assert_eq!(pairs.at(0).raw(), a.oop().raw());
        assert_eq!(
            poly_count_at(pairs, 0),
            3,
            "a's count traveled with its arm through compaction"
        );
        assert_eq!(poly_count_at(pairs, 1), 0, "dropped arm's count cleared");
    }

    #[test]
    fn ic_poly_reverify_drops_dead() {
        // A pair's klass is only reachable through a live send in v1 (no
        // klass/method removal exists), so "the lookup now fails" is
        // exercised the same way row 4b is: hand-craft the precondition
        // and drive `ic_transition` directly.
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"foo");
        let a = new_klass(&mut vm, "A");
        let dead = new_klass(&mut vm, "Dead");
        let m_a = method_returning(&mut vm, "foo_a", 1);
        install_method(&mut vm, a, sel, m_a);
        let caller = build_caller(&mut vm, sel, 0);

        // Hand-build a stale Poly(2) with one live pair (a) and one dead
        // pair (dead, which has no `foo`).
        let stale_method = method_returning(&mut vm, "stale", 99);
        let array_klass = vm.universe.array_klass;
        let pairs =
            crate::memory::alloc::alloc_indexable_oops(&mut vm, array_klass, IC_POLY_ARRAY_LEN);
        pairs.at_put(0, a.oop());
        pairs.at_put(1, m_a.oop());
        pairs.at_put(2, dead.oop());
        pairs.at_put(3, stale_method.oop());
        let ic = InterpreterIc::at(caller, 0);
        let stale_epoch = vm.ic_epoch;
        ic.set_poly(&mut vm, pairs, stale_epoch);
        vm.ic_epoch += 1;

        let result = ic_transition(&mut vm, caller, 0, a, false);
        assert_eq!(result, Some(m_a));
        assert_eq!(
            ic_state(caller, 0),
            IcState::Poly(1),
            "dead pair must be dropped and survivors compacted"
        );
        let pairs2 = ArrayOop::try_from(ic.target()).unwrap();
        assert_eq!(pairs2.at(0), a.oop());
        assert_eq!(pairs2.at(1), m_a.oop());
    }

    #[test]
    fn ic_super_static_target() {
        let mut vm = test_vm();
        let sel = vm.universe.intern(b"greet");
        let root = new_klass(&mut vm, "Root");
        let mid =
            vm.universe
                .new_klass(root, "Mid", crate::oops::Format::Slots, false, HEADER_WORDS);
        let leaf =
            vm.universe
                .new_klass(mid, "Leaf", crate::oops::Format::Slots, false, HEADER_WORDS);
        let root_impl = method_returning(&mut vm, "greetRoot", 42);
        install_method(&mut vm, root, sel, root_impl);

        // `midMethod`, installed on Mid, does `super greet` — the static
        // lookup start is `midMethod.holder().superclass()` == Root,
        // regardless of the receiver's actual (more-derived) klass. The
        // super send's receiver is `self` (`push_self`), not an argument —
        // `midMethod` itself is unary (argc=0), sent via `build_caller`'s
        // own unary-send shape.
        let mut b = BytecodeBuilder::new();
        b.push_self();
        b.send_super(&mut vm, sel, 0);
        b.ret_tos();
        let mid_sel = vm.universe.intern(b"midMethod");
        let mid_method = b.finish(&mut vm, mid_sel, 0, 0);
        install_method(&mut vm, mid, mid_sel, mid_method);

        let caller = build_caller(&mut vm, mid_sel, 0);
        let recv_mid = crate::memory::alloc::alloc_slots(&mut vm, mid).oop();
        let recv_leaf = crate::memory::alloc::alloc_slots(&mut vm, leaf).oop();

        let r1 = send_once(&mut vm, caller, &[recv_mid]);
        assert_eq!(r1, SmallInt::new(42).oop());
        assert_eq!(ic_state(mid_method, 0), IcState::Mono);

        let r2 = send_once(&mut vm, caller, &[recv_leaf]);
        assert_eq!(
            r2,
            SmallInt::new(42).oop(),
            "super target is static per site"
        );
        assert_eq!(
            ic_state(mid_method, 0),
            IcState::Poly(2),
            "guard still promotes on receiver klass, even though the target is shared"
        );
        let ic = InterpreterIc::at(mid_method, 0);
        let pairs = ArrayOop::try_from(ic.target()).unwrap();
        assert_eq!(pairs.at(1), root_impl.oop());
        assert_eq!(
            pairs.at(3),
            root_impl.oop(),
            "both pairs share the same super target"
        );
    }

    #[test]
    fn ic_meta_packing() {
        let mut vm = test_vm();
        let a = new_klass(&mut vm, "A");
        let m = method_returning(&mut vm, "foo", 1);

        // One method, one send site per argc value — `add_send` alone (no
        // matching bytecode instruction needed) is enough to materialize
        // the IC entry via `finish()`.
        let mut b = BytecodeBuilder::new();
        b.ret_self();
        let argcs = [0u8, 1, 15];
        let mut sites = Vec::new();
        for &argc in &argcs {
            let sel = vm.universe.intern(format!("sel{argc}").as_bytes());
            sites.push((b.add_send(&mut vm, sel, argc), argc));
        }
        let name = vm.universe.intern(b"holder");
        let holder_method = b.finish(&mut vm, name, 0, 0);

        for (ic_idx, argc) in sites {
            let ic = InterpreterIc::at(holder_method, ic_idx);
            assert_eq!(ic.argc(), argc, "argc round-trip for argc={argc}");
            for epoch in [0u32, 1, (1 << crate::oops::layout::IC_META_EPOCH_BITS) - 1] {
                ic.set_mono(&mut vm, a, m, epoch);
                assert_eq!(ic.epoch(), epoch, "epoch round-trip for argc={argc}");
                assert_eq!(ic.argc(), argc, "argc must survive an epoch/guard rewrite");
            }
        }
    }

    #[test]
    fn epoch_wrap_debug_assert() {
        let mut vm = test_vm();
        let limit = 1u32 << crate::oops::layout::IC_META_EPOCH_BITS;
        vm.ic_epoch = limit - 2;
        let a = new_klass(&mut vm, "A");
        let sel = vm.universe.intern(b"foo");
        let m = method_returning(&mut vm, "foo", 1);
        install_method(&mut vm, a, sel, m); // bumps to limit-1: still valid
        assert_eq!(vm.ic_epoch, limit - 1);
        if cfg!(debug_assertions) {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                install_method(&mut vm, a, sel, m); // bumps to limit: over the line
            }));
            assert!(
                result.is_err(),
                "must debug_assert past the 24-bit epoch limit"
            );
        }
    }
}
