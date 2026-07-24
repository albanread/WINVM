//! `OopMap` builder + verifier (`sprint_s12_detail.md` D2). The format
//! itself (`OopMap`, `PcDesc`) lives in `codecache::nmethod` — this module
//! is purely the compiler-side logic that PRODUCES a map from regalloc's
//! own interval data, and the independent check that a published
//! `Nmethod`'s maps are internally consistent. `compiler` already depends
//! on `codecache::nmethod` directly (`driver.rs`'s own imports), so this
//! is not a new dependency direction, just a new file.

use crate::codecache::nmethod::{Nmethod, OopMap};
use crate::compiler::ir::VReg;
use crate::compiler::regalloc::{Assignment, LiveInterval};

/// D2: the map of which spill slots hold a LIVE oop at exactly `position`
/// (regalloc's own linear numbering, `RegallocResult::safepoint_positions`'
/// own scheme) — `slot_is_oop ∧ slot_live_at(position)`, computed directly
/// rather than via a separately-maintained `slot_is_oop` array intersected
/// after the fact: an interval only ever occupies ONE physical slot for
/// its ENTIRE lifetime (regalloc's monotonic spill-slot allocation, never
/// reused across intervals — D3.5 point 3), so "is slot i live-and-oop at
/// `position`" reduces to "does the interval CURRENTLY assigned to slot i
/// span `position`", checked directly against `intervals`.
///
/// Liveness is STRICTLY-INSIDE on both ends (`start < position && end >
/// position`) — deliberately TIGHTER than `LiveInterval::
/// crosses_safepoint`'s own `start <= position` spill POLICY comparison,
/// because the two answer different questions. An interval whose only
/// reference IS this safepoint's argument list ends AT the call (`end ==
/// position`) and is already consumed — excluded, as before. But an
/// interval that STARTS at this safepoint is the call's own DESTINATION:
/// its value is produced by the RETURN, so for the whole duration of the
/// call — exactly when a GC can walk this frame — its spill slot still
/// holds uninitialized native-stack garbage that has never been written.
/// The original `start <= position` included it, and the flagship gate's
/// `GC_STRESS=1` run caught that as a real SIGSEGV the first time an
/// alloc slow edge scavenged in a frame whose dst slot held nonzero
/// stack junk (`allocation_fast_and_slow`; the mid-loop flagships dodged
/// it only because THEIR dst slots happened to hold zeros, which read as
/// a harmless smi). For the spill POLICY the `<=` form stays correct —
/// the dst does need a slot assigned across the call — it just must not
/// be TRACED at this particular safepoint; the next safepoint (start <
/// position by then) covers it normally.
/// `extra_oop_live` is `RegallocResult::extra_oop_live` — exact `(vreg,
/// position)` facts a deopt site's own recorded stack/slots need, checked
/// here as an ADDITIONAL, exact-position fact alongside each interval's
/// plain `[start,end]`. Kept separate rather than folded into the interval
/// itself: a vreg needed only by one far-away trap (traps linearize into
/// the cold tail — `emit_uncommon_trap`'s own doc) would otherwise need its
/// interval widened to cover everything numerically in between, which is
/// unsound wherever that span crosses an if/else merge also reachable from
/// a sibling arm that never wrote it (`compute_intervals`'s own doc has
/// the full story — this is what `extra_oop_live` exists to avoid).
pub fn build_for_position(
    intervals: &[LiveInterval],
    frame_slots: u16,
    position: u32,
    extra_oop_live: &[(VReg, u32)],
    poll_saves: &[(u32, Vec<crate::compiler::regalloc::PollSave>)],
) -> OopMap {
    let mut map = OopMap::empty();
    // F3c S1: at a POLL position, the save slots of oop-holding saved
    // registers are live roots — the slow path stored them before anything
    // GC-capable runs, and the restore reads back whatever the root walk
    // rewrote. `frame_slots` here is the TOTAL universe (spills + save
    // area); every other position leaves the save region untraced.
    if let Some((_, saves)) = poll_saves.iter().find(|(p, _)| *p == position) {
        for s in saves.iter().filter(|s| s.is_oop) {
            debug_assert!(
                s.save_slot.0 < frame_slots,
                "build_for_position: save slot {} out of range (total={frame_slots})",
                s.save_slot.0
            );
            map.set(s.save_slot.0);
        }
    }
    // One pass over the facts per POSITION (not per interval × position):
    // the head-2 fix legitimately multiplied the fact count (loop-carried
    // slots get one fact per in-loop safepoint), and deltablue's steady
    // trap→recompile churn turned a per-interval linear `.any()` into a
    // measured +103% wall regression — of pure COMPILE time. The mutator
    // never touches this; keep the builder linear in the facts.
    let exact_here: std::collections::HashSet<u32> = extra_oop_live
        .iter()
        .filter(|&&(_, p)| p == position)
        .map(|&(v, _)| v.0)
        .collect();
    for iv in intervals {
        if !iv.is_oop {
            continue;
        }
        let Some(Assignment::Spill(slot)) = iv.assignment else {
            continue;
        };
        let live =
            (iv.start < position && iv.end > position) || exact_here.contains(&iv.vreg.0);
        if live {
            debug_assert!(
                slot.0 < frame_slots,
                "build_for_position: slot {} out of range (frame_slots={frame_slots})",
                slot.0
            );
            map.set(slot.0);
        }
    }
    map
}

/// D2's dedup rule ("two safepoints with identical liveness share one
/// oopmaps index"): finds `map` in `oopmaps` by CONTENT (not identity —
/// `OopMap`'s own `PartialEq`) and returns its index, appending only if no
/// equal map already exists. `driver.rs`'s `compile_method` is the only
/// real caller (called once per real safepoint, `oopmaps` starting
/// pre-seeded with the reserved `OopMap::empty()` at index 0 — a safepoint
/// with no live oops interns right back onto that same reserved slot, not
/// a fresh duplicate).
pub fn intern(oopmaps: &mut Vec<OopMap>, map: OopMap) -> u16 {
    match oopmaps.iter().position(|m| *m == map) {
        Some(idx) => idx as u16,
        None => {
            oopmaps.push(map);
            (oopmaps.len() - 1) as u16
        }
    }
}

/// D1 enforcement point 2 ("`oopmap::verify(nm)` (debug + stress): decodes
/// every map and checks `bit ⇒ slot < frame_slots` and `bit ⇒
/// slot_is_oop[slot]`") — an INDEPENDENT cross-check against `Nmethod::
/// slot_is_oop` (regalloc's own per-slot ground truth), not a tautology:
/// `build_for_position` above already only ever sets a bit for an
/// `is_oop` interval, so a passing verify here doesn't just confirm that
/// function ran correctly once — it catches ANY later corruption of
/// `oopmaps`/`slot_is_oop` (a bad patch, a future refactor that
/// reconstructs a map differently) that would otherwise trace a raw,
/// non-oop bit pattern as a pointer. Panics on the first violation found
/// (a VM-internal-consistency bug, not a user-triggerable condition —
/// same posture as `regalloc::verify_spill_all`).
pub fn verify(nm: &Nmethod) {
    for (map_idx, map) in nm.oopmaps.iter().enumerate() {
        for slot in map.iter_slots() {
            assert!(
                slot < nm.frame_slots,
                "oopmap::verify: nmethod {:?} oopmaps[{map_idx}] sets slot {slot} >= \
                 frame_slots ({})",
                nm.id,
                nm.frame_slots
            );
            assert!(
                nm.slot_is_oop[slot as usize],
                "oopmap::verify: nmethod {:?} oopmaps[{map_idx}] marks slot {slot} live but \
                 regalloc never assigned it an oop-typed interval (slot_is_oop[{slot}] = false)",
                nm.id
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::VReg;
    use crate::compiler::regalloc::SpillSlot;

    /// `tests_s12.md`'s `oopmap_dedup`: two safepoints with IDENTICAL
    /// liveness intern to the SAME index; a genuinely different one gets
    /// its own; the pre-seeded reserved empty map (index 0) is reused for
    /// an empty safepoint rather than duplicated.
    #[test]
    fn oopmap_dedup() {
        let mut oopmaps = vec![OopMap::empty()];

        let mut a = OopMap::empty();
        a.set(1);
        let idx_a = intern(&mut oopmaps, a.clone());
        assert_eq!(oopmaps.len(), 2, "a genuinely new map appends");

        let idx_a_again = intern(&mut oopmaps, a);
        assert_eq!(
            idx_a_again, idx_a,
            "identical content reuses the same index"
        );
        assert_eq!(oopmaps.len(), 2, "must NOT append a duplicate");

        let mut b = OopMap::empty();
        b.set(1);
        b.set(3);
        let idx_b = intern(&mut oopmaps, b);
        assert_ne!(idx_b, idx_a, "a different live set gets its own index");
        assert_eq!(oopmaps.len(), 3);

        let idx_empty = intern(&mut oopmaps, OopMap::empty());
        assert_eq!(
            idx_empty, 0,
            "an empty safepoint reuses the reserved slot 0"
        );
        assert_eq!(
            oopmaps.len(),
            3,
            "must not append for the empty case either"
        );
    }

    fn spilled(vreg: u32, slot: u16, is_oop: bool, start: u32, end: u32) -> LiveInterval {
        LiveInterval {
            vreg: VReg(vreg),
            start,
            end,
            is_oop,
            is_fp: false,
            crosses_safepoint: true,
            crosses_call: false,
            resident_reg: None,
            assignment: Some(Assignment::Spill(SpillSlot(slot))),
        }
    }

    /// D2/P2 (`tests_s12.md`'s `oopmap_liveness_intersection`): a slot
    /// whose oop interval ENDS strictly before the safepoint position must
    /// be excluded, even though `is_oop` is true and the slot itself would
    /// otherwise qualify — liveness is per-POSITION, not per-slot.
    #[test]
    fn oopmap_liveness_intersection() {
        let intervals = vec![
            spilled(0, 0, true, 0, 5),  // ends at 5, safepoint is at 10: dead
            spilled(1, 1, true, 0, 20), // spans 10: live
        ];
        let map = build_for_position(&intervals, 2, 10, &[], &[]);
        assert!(
            !map.is_oop(0),
            "interval ending before the safepoint must be excluded"
        );
        assert!(
            map.is_oop(1),
            "interval spanning the safepoint must be included"
        );
    }

    /// BUG D (regression, `cold_branch_recompile_spill_corruption.mst`): a
    /// vreg's organic interval ends at its own last real use, but a FAR-AWAY
    /// deopt site (traps linearize into the cold tail) also needs it —
    /// `extra_oop_live` records that ONE exact position, and must not leak
    /// liveness into anything in between, e.g. a real safepoint on a
    /// SIBLING if/else arm that never wrote this vreg's slot at all.
    #[test]
    fn oopmap_extra_oop_live_is_exact_not_a_widened_range() {
        // Organic interval [0,5): dead well before either query position.
        let intervals = vec![spilled(0, 0, true, 0, 5)];
        let extra = [(VReg(0), 80)];
        assert!(
            !build_for_position(&intervals, 1, 68, &extra, &[]).is_oop(0),
            "an unrelated safepoint numerically between the organic end and \
             the forced trap position must NOT see the vreg as live"
        );
        assert!(
            build_for_position(&intervals, 1, 80, &extra, &[]).is_oop(0),
            "the EXACT forced position must see the vreg as live"
        );
        assert!(
            !build_for_position(&intervals, 1, 81, &extra, &[]).is_oop(0),
            "one past the forced position must not — this is a point fact, \
             not a range"
        );
    }

    /// An interval ending EXACTLY at the safepoint position is consumed BY
    /// it (the safepoint's own argument), not live ACROSS it — matches
    /// `crosses_safepoint`'s own `end > sp` (not `>=`) convention exactly.
    #[test]
    fn oopmap_excludes_interval_ending_at_position() {
        let intervals = vec![spilled(0, 0, true, 0, 10)];
        let map = build_for_position(&intervals, 1, 10, &[], &[]);
        assert!(!map.is_oop(0));
    }

    /// An interval STARTING exactly at the safepoint position is the
    /// call's own DESTINATION — its value is produced by the RETURN, so
    /// its spill slot holds uninitialized native-stack garbage for the
    /// whole duration of the call, exactly when a GC can walk the frame.
    /// Regression for a real SIGSEGV the flagship gate's `GC_STRESS=1`
    /// run caught (`allocation_fast_and_slow`: the alloc slow edge
    /// scavenged, the map's original `start <= position` traced the
    /// unwritten dst slot's stack junk as an oop, and `mark_word_raw`
    /// dereferenced a 0-page address) — see `build_for_position`'s own
    /// doc for the full liveness-vs-spill-policy distinction.
    #[test]
    fn oopmap_excludes_interval_starting_at_position() {
        let intervals = vec![spilled(0, 0, true, 10, 20)];
        let map = build_for_position(&intervals, 1, 10, &[], &[]);
        assert!(
            !map.is_oop(0),
            "a call's own dst (def AT the safepoint) must not be traced during the call"
        );
        // ...and the very next safepoint, once the value genuinely exists,
        // covers it normally.
        let map_later = build_for_position(&intervals, 1, 15, &[], &[]);
        assert!(map_later.is_oop(0));
    }

    /// A non-oop interval (e.g. a raw untagged scratch value) sharing the
    /// safepoint's own live range must never set its bit, regardless of
    /// timing.
    #[test]
    fn oopmap_excludes_non_oop_interval() {
        let intervals = vec![spilled(0, 0, false, 0, 20)];
        let map = build_for_position(&intervals, 1, 10, &[], &[]);
        assert!(!map.is_oop(0));
    }

    /// `tests_s12.md`'s `oopmap_verifier_catches_nonoop_bit`: hand-corrupt
    /// a map to set a bit `slot_is_oop` disagrees with — `verify` must
    /// panic, not silently accept it.
    #[test]
    #[should_panic(expected = "regalloc never assigned it an oop-typed interval")]
    fn oopmap_verifier_catches_nonoop_bit() {
        let mut bad_map = OopMap::empty();
        bad_map.set(0); // slot 0 is NOT oop per slot_is_oop below
        let nm = Nmethod::test_fake(2, vec![false, true], vec![bad_map]);
        verify(&nm);
    }

    /// The verifier's own happy path: a correctly-built map over a
    /// genuinely oop-typed slot passes cleanly.
    #[test]
    fn oopmap_verifier_accepts_consistent_map() {
        let mut good_map = OopMap::empty();
        good_map.set(1);
        let nm = Nmethod::test_fake(2, vec![false, true], vec![good_map]);
        verify(&nm); // must not panic
    }

    /// The verifier's OTHER check (D1 point 2's two halves are independent):
    /// a bit set past `frame_slots` must panic too — P3's own reasoning
    /// (an off-by-one here would read the saved-fp/lr pair as an "oop").
    /// Constructed by hand (not via `build_for_position`, which has its own
    /// separate debug_assert against this) so `verify`'s own check is what
    /// actually fires.
    #[test]
    #[should_panic(expected = "slot 5 >= frame_slots")]
    fn oopmap_verifier_catches_out_of_range_slot() {
        let mut bad_map = OopMap::empty();
        bad_map.set(5); // frame_slots is only 2 below
        let nm = Nmethod::test_fake(2, vec![false, true], vec![bad_map]);
        verify(&nm);
    }
}
