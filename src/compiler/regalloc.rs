//! Linearization + live intervals + linear-scan register allocation
//! (`sprint_s10_detail.md` D3.4/D3.5). Operates purely on an [`IrMethod`] —
//! no `VmState`/bytecode/`MethodOop` involved, so every test here builds
//! its `IrMethod` (or, for `allocate` alone, its `LiveInterval`s) by hand.
//!
//! Two independently useful stages, matching tests_s10.md's own split:
//! [`compute_intervals`] (linearize + conservative `[min def, max use]`
//! intervals per vreg) and [`allocate`] (the spill-all-at-safepoints +
//! classic linear-scan policy, D3.5), plus [`regalloc`] gluing both for
//! `driver.rs`'s pipeline.

use std::collections::HashMap;

use crate::compiler::ir::{BlockId, Ir, IrBlock, IrMethod, VReg};
use crate::compiler::scopes::SafepointKind;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assignment {
    Reg(u8),
    Spill(SpillSlot),
}

/// Slot `i` lives at `[x29 − 8·(i+1)]` (D3.4) — an opaque index here;
/// `emit.rs` computes the real frame offset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SpillSlot(pub u16);

/// One vreg's conservative live range: `[start, end]` (both inclusive,
/// instruction positions) covering every def and every use, holes ignored
/// (classic Poletto/Sarkar linear scan — D3.4's own call: this is
/// deliberately simpler than precise per-branch liveness, and correct for
/// SSA-lite's multiple-defs-per-temp-vreg shape, not merely convenient for
/// it: `interval_multi_def_union` is the intended behavior, not a
/// tolerated approximation).
#[derive(Debug)]
pub struct LiveInterval {
    pub vreg: VReg,
    pub start: u32,
    pub end: u32,
    pub is_oop: bool,
    /// Float fast-path (`docs/float_fastpath_design.md`): this vreg is an
    /// UNBOXED `f64` — allocated from the FP `d0`–`d7` pool (disjoint from
    /// the x0–x15 GPR scan; never resident, never an oop-map entry). A
    /// crossing-safepoint fp interval spills exactly like a GPR one; its
    /// slot is non-oop (`is_oop == false`), so the GC skips it.
    pub is_fp: bool,
    /// True iff some `CallSend`/`CallRuntime`/`Alloc` position `p` satisfies
    /// `start <= p && end > p` — defined by, not merely used at, that
    /// safepoint (an interval whose only reference IS the safepoint's own
    /// argument list ends exactly at `p` and does not need to survive it).
    pub crosses_safepoint: bool,
    /// S14 perf recovery: true iff a REAL CALL (`CallSend`/`CallRuntime`)
    /// position sits strictly inside this interval. A call clobbers every
    /// caller-saved register AND its callee may itself use the resident
    /// registers, so only call-free intervals qualify for residency.
    pub crosses_call: bool,
    pub assignment: Option<Assignment>,
    /// S14 perf recovery (the 135x-regression fix): a callee-saved register
    /// (x21–x23) this SPILLED interval's value ALSO lives in between
    /// GC-continuing safepoints. The frame slot stays canonical — every def
    /// writes BOTH (write-through), deopt/oop-maps read slots unchanged — but
    /// reads prefer the register, and the only re-syncs are the Poll/Alloc
    /// SLOW paths (their fast paths neither call nor GC; trap fail-edges are
    /// terminating, so a stale register is never read after one). `None` for
    /// register-assigned or call-crossing intervals.
    pub resident_reg: Option<u8>,
}

fn is_safepoint(ir: &Ir) -> bool {
    matches!(
        ir,
        // S13 step 7b: `UncommonTrap` is a safepoint too — every oop live
        // across it (the re-executing send's `a`/`b`/`self`, kept live by the
        // fail block's `DeoptRaw.stack`) must be spilled (spill-all) and get
        // an OopMap, exactly like a call, so the deopt materializer can read
        // them from the frame. Its position keys BOTH the S12 OopMap and the
        // S13 deopt scope at the brk offset.
        //
        // S13 step 10b: a loop back-edge `Poll` is now a safepoint too — its
        // `bl stub_poll` may deopt the frame (if the loop's own nmethod became
        // `NotEntrant`), so the loop-carried operand stack + receiver + slots
        // (its `DeoptRaw.stack`, forced live-across by `deopt_live` below) must
        // be spilled to frame slots the materializer reads. Its position keys
        // the OopMap (over the `bl` call) AND the LoopPoll deopt scope, at the
        // poll's return offset.
        Ir::CallSend { .. }
            | Ir::CallRuntime { .. }
            | Ir::Alloc { .. }
            // Float fast-path: FBox allocates (inline bump + overflow `bl
            // stub_box_double`) — a safepoint exactly like `Alloc`.
            | Ir::FBox { .. }
            | Ir::UncommonTrap { .. }
            | Ir::Poll
    )
}

/// Every block a given block's terminator can transfer control to —
/// includes `fail`/`not_bool`/`slow` edges (the bailout block, or an S11
/// deopt/slow-path block), not just the "normal" successors.
fn successors(block: &IrBlock) -> Vec<BlockId> {
    let mut succs = Vec::new();
    for ir in &block.code {
        match ir {
            Ir::Jump { target } => succs.push(*target),
            Ir::BoolBr {
                if_true,
                if_false,
                not_bool,
                ..
            } => {
                succs.push(*if_true);
                succs.push(*if_false);
                succs.push(*not_bool);
            }
            Ir::SmiCmpBr {
                if_true,
                if_false,
                fail,
                ..
            } => {
                succs.push(*if_true);
                succs.push(*if_false);
                succs.push(*fail);
            }
            Ir::FCmpBr {
                if_true, if_false, ..
            } => {
                succs.push(*if_true);
                succs.push(*if_false);
            }
            Ir::SmiArith { fail, .. }
            | Ir::SmiCmpVal { fail, .. }
            | Ir::ArrayAt { fail, .. }
            | Ir::ArrayAtPut { fail, .. }
            | Ir::FUnbox { fail, .. }
            | Ir::VecArith { fail, .. } => succs.push(*fail),
            Ir::GuardKlass { fail, .. } => succs.push(*fail),
            // S11 D7: `Alloc` is self-contained (fast path + internal slow
            // call, `emit::emit_alloc`) — no slow CFG successor. It stays a
            // safepoint via `is_safepoint` so live-across vregs spill before
            // the internal `bl`; it just doesn't branch to another block.
            _ => {}
        }
    }
    succs
}

/// D3.4: entry first, loop bodies contiguous. A plain postorder DFS,
/// reversed — any block unreachable from the entry (dead code, e.g.
/// decode's own `unreachable_after_return` case surviving into the IR)
/// still needs a position, so any DFS root left unvisited afterward is
/// walked too, appended in index order (dead code never affects real
/// blocks' relative order, only its own).
/// D3.4/D5's own hard requirement: block 0 (the method's real entry) MUST
/// come first in the returned order, unconditionally — `emit.rs`'s prologue
/// falls straight through into whichever block is emitted first, with no
/// guard, so anything else there runs before the method's own body ever
/// does. Standard reverse-postorder-of-a-single-DFS-tree guarantees the
/// root is last in postorder (hence first after reversing THAT tree's own
/// segment) — but block 0 frequently has NO graph successors at all (any
/// straight-line method with no inlined branch/smi-arith fail edge, e.g. a
/// bare accessor like `^value` or `^false` — LoadField/Ret/RetSelf/Bailout
/// aren't matched in `successors` at all), making it its own singleton DFS
/// component. A version of this function that looped over every root
/// in order but reversed the WHOLE accumulated postorder only ONCE, at the
/// end, inverted the relative order BETWEEN components too, not just
/// within each one — block 0's tiny (or singleton) component, visited and
/// pushed first, ended up LAST after a single global reversal. Reversing
/// each root's own segment separately, then concatenating the segments in
/// root order, is what actually preserves "block 0's component comes
/// first" for a forest, not just for one connected tree.
fn reverse_postorder(method: &IrMethod) -> Vec<BlockId> {
    fn dfs(b: usize, blocks: &[IrBlock], visited: &mut [bool], postorder: &mut Vec<BlockId>) {
        if visited[b] {
            return;
        }
        visited[b] = true;
        for succ in successors(&blocks[b]) {
            dfs(succ.0 as usize, blocks, visited, postorder);
        }
        postorder.push(BlockId(b as u32));
    }

    let n = method.blocks.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for b in 0..n {
        if visited[b] {
            continue;
        }
        let mut postorder = Vec::new();
        dfs(b, &method.blocks, &mut visited, &mut postorder);
        postorder.reverse();
        order.extend(postorder);
    }
    order
}

/// D3.4: number every instruction sequentially (walking blocks in
/// `reverse_postorder`), then fold each vreg's defs/uses into one
/// conservative `[min, max]` interval — a single linear pass, no separate
/// per-block live-in/live-out fixpoint (every def and use is already
/// explicit in the Ir stream — `ir.rs`'s own Move-based merge handling
/// means nothing needs inferring across a block boundary).
///
/// That last claim is true for values that flow through the explicit
/// merge-vreg mechanism at a join — but a temp vreg (`ir.rs`'s "SSA-lite
/// temp rule": one persistent vreg per source temp, reused directly, never
/// re-merged) that's both defined AND used inside the SAME loop body block
/// is a real gap in it: the body block appears exactly once in this linear
/// position space even though it runs many times at runtime, so a def near
/// the block's own end feeding a use near its own start (the next
/// iteration, via the back-edge) has its def position AFTER its use
/// position in linear terms — invisible to a plain `[min_def, max_use]`
/// fold, which would let some OTHER vreg's interval start immediately
/// after that "last" use and steal the same register out from under a
/// value the loop is still very much using. `reverse_postorder` itself
/// only promises block 0 first (S10 step 9's own bug) and every
/// predecessor-except-back-edges before its successors — it does not, and
/// for an if/else-vs-loop-body sibling pair generally cannot, promise a
/// loop body's blocks all precede whatever follows the loop. The fix below
/// doesn't try to fix the linearization further; it widens intervals after
/// the fact: any back edge B->A (A's block starting at or before B's own
/// start, by position) defines a loop range `[start of A, end of B]`, and
/// every interval touching that range at all gets conservatively widened
/// to cover the whole thing — sound (if pessimistic) for nested loops too,
/// via a fixpoint over every back edge found.
///
/// The third return value, `safepoint_positions`, is S12's own addition:
/// the exact linear position of every `CallSend`/`CallRuntime`/`Alloc` op,
/// in this SAME numbering — `compiler::oopmap::build_for_position` (and
/// `emit.rs`'s own position counter, which walks `block_order` identically)
/// depend on this being the exact same sequence `crosses_safepoint` above
/// was computed against, not a re-derivation that could drift out of sync.
///
/// The fifth, `extra_oop_live` (a bug-fix era addition — see this
/// function's own `deopt_live_exact` doc further down), is
/// `RegallocResult::extra_oop_live` — exact `(vreg, position)` facts kept
/// SEPARATE from the plain `[start,end]` intervals in the second return
/// value, for the same reason: folding them in would widen an interval
/// across everything numerically in between, unsound wherever that spans
/// an if/else merge reachable from a sibling arm that never wrote the vreg.
#[allow(clippy::type_complexity)]
pub fn compute_intervals(
    method: &IrMethod,
) -> (
    Vec<BlockId>,
    Vec<LiveInterval>,
    Vec<u32>,
    std::collections::HashMap<u32, u32>,
    Vec<(VReg, u32)>,
) {
    let block_order = reverse_postorder(method);

    let mut pos: u32 = 0;
    let mut safepoint_positions: Vec<u32> = Vec::new();
    let mut call_positions: Vec<u32> = Vec::new();
    let mut min_def: HashMap<u32, u32> = HashMap::new();
    let mut max_use: HashMap<u32, u32> = HashMap::new();
    let mut block_start_pos: HashMap<u32, u32> = HashMap::new();
    let mut block_end_pos: HashMap<u32, u32> = HashMap::new();
    // S13 step 7b: every vreg an UNCOMMON-TRAP deopt site reads must be LIVE
    // ACROSS its safepoint (spilled to a frame slot the materializer can
    // read), not merely live UP TO it. `driver::build_deopt_metadata` resolves,
    // for each site: the receiver (VReg 0), every arg/temp slot
    // (VReg 1..=argc+ntemps), and the recorded operand `stack` — so all three
    // must cross. This matters for a reexecute UncommonTrap because its fail
    // block has NO fall-through and is linearized LAST (a DFS dead end), so
    // NOTHING is naturally live across it: a value "used after the send" is
    // used in the CONTINUATION block, which linearizes BEFORE the trap, so its
    // interval ends before the trap position, it keeps a register, and it
    // would resolve to Nil — a silently-wrong deopt. Collected here (with the
    // safepoint's own position) and forced to `end > pos` below, so
    // `crosses_safepoint` fires and spill-all pins them.
    //
    // Scoped to `UncommonTrap` and `LoopPoll` — NOT `Call`/`Alloc`: those (S13
    // step 3b) sit inline in a block whose successors run AFTER them, so their
    // recorded vregs are already naturally live-across (used later) and already
    // spilled; widening THOSE would spill genuinely-dead values (a call-return
    // site's popped receiver/args, an Alloc's class const) into their OopMaps,
    // needlessly enlarging them and disturbing S12's GC-root tests — and is
    // unnecessary, since natural liveness already covers exactly what those
    // sites read.
    //
    // S13 step 10b: a `LoopPoll` site (an `Ir::Poll` at a loop back-edge) needs
    // the SAME widening. Its recorded `stack` is the loop-carried operand stack
    // — genuinely live (re-read on the next loop iteration), NOT dead like a
    // call-return's popped operands. Loop-range widening (below) already extends
    // loop-carried intervals to `loop_end`, but the poll can sit AT `loop_end`,
    // so those intervals may `end == poll_pos` rather than STRICTLY across it
    // (`crosses_safepoint` needs `end > pos`). Forcing `end > pos` here pins
    // receiver + slots + the recorded stack to canonical frame slots the deopt
    // materializer reads, exactly as for an UncommonTrap.
    // Two SEPARATE lists, folded differently below (see the fold sites'
    // own docs): `deopt_live_exact`'s vregs (a trap's own receiver/slots/
    // recorded stack) have NO dominance guarantee over other safepoints
    // that merely sit at a nearby LINEAR position, so widening their
    // interval to reach a far-away trap is unsound; `deopt_live_widen`'s
    // vregs (ctx-temps) DO — every declared Smalltalk temp is nil-initialized
    // unconditionally at method entry (Smalltalk's own semantics), before
    // any branch, so their write genuinely dominates every later safepoint
    // in the method, and widening them is sound.
    let mut deopt_live_exact: Vec<(u32, u32)> = Vec::new(); // (vreg, safepoint pos)
    let mut deopt_live_widen: Vec<(u32, u32)> = Vec::new(); // (vreg, safepoint pos)
    let n_slots = method.argc as u32 + method.ntemps as u32;

    for &bid in &block_order {
        let block = &method.blocks[bid.0 as usize];
        block_start_pos.insert(bid.0, pos);
        for (idx, ir) in block.code.iter().enumerate() {
            if is_safepoint(ir) {
                safepoint_positions.push(pos);
            }
            if matches!(ir, Ir::CallSend { .. } | Ir::CallRuntime { .. }) {
                call_positions.push(pos);
            }
            if let Some((_, raw)) = block.deopt_sites.iter().find(|(ci, raw)| {
                *ci == idx as u32
                    && matches!(
                        raw.kind,
                        SafepointKind::UncommonTrap | SafepointKind::LoopPoll
                    )
            }) {
                // Receiver (0) + every unified arg/temp slot + the recorded
                // operand stack are exactly the vregs the driver resolves for
                // this site.
                //
                // Task #94 (the second `extra_oop_live` gap, sibling of BUG D
                // root cause 3): recording each vreg at the TRAP position
                // alone makes the oop map correct AT THE TRAP — but a GC can
                // strike at any EARLIER safepoint on the way there (a
                // `CallSend`'s callee allocating — under GC_STRESS, every
                // one), and the oop map at THAT safepoint knew nothing of
                // these slots, so the collector left them stale while the
                // objects moved. The trap's materializer then read
                // relocated-away addresses — under scavenge-per-allocation
                // the old eden-base address aliases the NEXT fresh object,
                // producing a wrong-but-VALID oop (the repro's `s setOn: s`:
                // `WriteStream class>>on:`'s spilled String argument, dead
                // organically after its spill, aliased the `basicNew` result
                // allocated at the recycled eden base). So each fact is
                // recorded at EVERY safepoint up to and including the trap.
                // Sound because `emit` nil-fills exactly these slots in the
                // prologue (`deopt_nil_init_slots` below): a safepoint
                // reached before the vreg's def — or via a sibling arm that
                // never wrote it — scans nil (or a conservatively-kept older
                // value), never uninitialized native stack. The same
                // path-insensitivity that made interval-widening UNSOUND
                // (root cause 1/3's lesson) is made harmless by the fill,
                // NOT by pretending liveness is linear.
                let mut record = |v: u32| {
                    deopt_live_exact.push((v, pos));
                    for &sp in &safepoint_positions {
                        if sp < pos {
                            deopt_live_exact.push((v, sp));
                        }
                    }
                };
                record(0);
                for s in 1..=n_slots {
                    record(s);
                }
                for &v in &raw.stack {
                    record(v.0);
                }
            }
            // S14 step 4c: an INLINED-body safepoint (of ANY kind, INCLUDING a
            // `Call` — a real `CallSend` inside a spliced non-leaf body). Its
            // deopt rebuilds TWO interpreter frames from this one physical
            // frame, so the driver resolves, for THIS site, not just the root
            // scope's receiver/slots but ALSO the INLINED scope's receiver +
            // slots and the caller's `pending_stack` — none of which are
            // guaranteed naturally live-across (the inlined body may just return
            // after the send, leaving the caller's frozen operand stack and the
            // root slots dead in the compiled code, hence Nil'd — a silently
            // wrong depth-2 deopt). Widen them all to `end > pos` so spill-all
            // pins every entity of BOTH frames to a canonical slot the
            // materializer reads. The `Call` exclusion above is deliberately not
            // relaxed for ROOT `Call`s (natural liveness covers them); this is
            // the narrow inlined-site case only.
            if let Some((_, raw)) = block
                .deopt_sites
                .iter()
                .find(|(ci, raw)| *ci == idx as u32 && raw.inline.is_some())
            {
                // Same task-#94 earlier-safepoint coverage as the plain-trap
                // arm above — an inlined site's rebuilt frames read the very
                // same slots, exposed to the very same mid-callee GC.
                let mut record = |v: u32| {
                    deopt_live_exact.push((v, pos));
                    for &sp in &safepoint_positions {
                        if sp < pos {
                            deopt_live_exact.push((v, sp));
                        }
                    }
                };
                // Caller (root) frame: receiver + every unified slot.
                record(0);
                for s in 1..=n_slots {
                    record(s);
                }
                // S14 step 7-IV-b: EVERY inline level of the chain (a block
                // spliced inside an inlined callee is depth 3) — each level's
                // receiver + slots + frozen pending stack must be live-across so
                // spill-all pins every entity of every rebuilt frame.
                let mut level = raw.inline.as_ref();
                while let Some(site) = level {
                    record(site.receiver.0);
                    for slot in &site.slots {
                        record(slot.0);
                    }
                    for v in &site.caller_pending_stack {
                        record(v.0);
                    }
                    level = site.parent.as_deref();
                }
                // The innermost recorded operand stack.
                for &v in &raw.stack {
                    record(v.0);
                }
            }
            // S14 step 7-II-b: M's promoted ctx-temps back the ELIDED Context the
            // ROOT scope materializes at EVERY deopt (any kind — trap, root
            // `Call`, inlined-body site). They are frequently DEAD in the compiled
            // code after a terminating trap (their post-trap ctx-temp reads never
            // emit), so natural liveness does NOT cover them — force each live
            // across so spill-all pins it to a frame slot the `CtxLoc::Elided`
            // materializer reads. Only for a `has_ctx` M (else `ctx_vregs` empty).
            if !method.ctx_vregs.is_empty()
                && block.deopt_sites.iter().any(|(ci, _)| *ci == idx as u32)
            {
                for &cv in &method.ctx_vregs {
                    deopt_live_widen.push((cv.0, pos));
                }
            }
            ir.uses(|v| {
                max_use
                    .entry(v.0)
                    .and_modify(|e| *e = (*e).max(pos))
                    .or_insert(pos);
                min_def.entry(v.0).or_insert(pos);
            });
            ir.defs(|v| {
                min_def
                    .entry(v.0)
                    .and_modify(|e| *e = (*e).min(pos))
                    .or_insert(pos);
                max_use.entry(v.0).or_insert(pos);
            });
            pos += 1;
        }
        // `pos - 1`: the position of this block's own last instruction (a
        // block always has at least one instruction — every terminator is
        // itself an Ir op); every block gets a position above, so `pos`
        // has advanced past `block_start_pos[bid]` by the time we get here.
        block_end_pos.insert(bid.0, pos.saturating_sub(1));
    }

    // Back-edge loop-range widening (see this function's own doc above).
    let mut loop_ranges: Vec<(u32, u32)> = Vec::new();
    for &bid in &block_order {
        let b = bid.0 as usize;
        let b_start = block_start_pos[&bid.0];
        for succ in successors(&method.blocks[b]) {
            if let Some(&a_start) = block_start_pos.get(&succ.0) {
                if a_start <= b_start {
                    let loop_end = block_end_pos[&bid.0];
                    loop_ranges.push((a_start, loop_end));
                }
            }
        }
    }
    // GC_STRESS head-2 fix (deltablue `projectionTest:`): an exact deopt
    // fact `(v, P)` lying OUTSIDE v's `[min_def, max_use]`, with a loop
    // range connecting P to the interval, is PROOF the value is
    // loop-carried — the deopt metadata at P (a loop-head send's recorded
    // operand stack) reads the PREVIOUS iteration's value, which flows to P
    // around the back edge. The exact fact alone makes only P's own oop map
    // claim the slot; the safepoints in the loop TAIL (between the def and
    // the back edge) claim nothing, so under stress the slot sat unscanned
    // for 13 scavenges while its object moved, and P's map then handed the
    // rotted address to the collector (the to-space tripwire, nm slot 41,
    // rel_pc 0xc44 vs 0xd50–0xeb8).
    //
    // The fix is MAP-ONLY, deliberately: v is already spill-assigned (every
    // deopt-referenced vreg is force-spilled below) and every def writes its
    // canonical slot, so between defs the slot always holds v's last value —
    // the GC just needs to be TOLD about it at the loop's other safepoints.
    // So this adds exact `(v, safepoint)` facts for every safepoint inside
    // the connecting loop range, leaving the interval — and therefore
    // register allocation, spill decisions, and every emitted instruction —
    // byte-identical. (The first cut widened the INTERVAL instead: sound,
    // but the widening cascaded through the fixpoint below and measured
    // +104% on deltablue. Never tax the mutator for a GC-visibility fact.)
    // BUG D's precision rule below is untouched: wholly-inside temps have no
    // out-of-interval exact facts. A straight-line exact fact (task #94's
    // case: a trap position past the last use, no loop) matches no loop
    // range containing both and stays exact.
    {
        let mut loop_map_facts: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();
        for &(v, p) in &deopt_live_exact {
            let s = *min_def.get(&v).unwrap_or(&u32::MAX);
            let e = *max_use.get(&v).unwrap_or(&0);
            if s == u32::MAX {
                continue; // never defined in compiled code — the [0,0] path below
            }
            if p >= s && p <= e {
                continue; // fact already inside the interval
            }
            for &(ls, le) in &loop_ranges {
                let contains_p = ls <= p && p <= le;
                let touches_interval = s <= le && e >= ls;
                if contains_p && touches_interval {
                    for &sp in &safepoint_positions {
                        // Only the GAP safepoints: the strictly-inside window
                        // of the interval is already map-live via the normal
                        // interval test; duplicating those would only bloat
                        // the fact list the oopmap builder filters per call.
                        if sp >= ls && sp <= le && !(sp > s && sp < e) {
                            loop_map_facts.insert((v, sp));
                        }
                    }
                }
            }
        }
        deopt_live_exact.extend(loop_map_facts);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &(loop_start, loop_end) in &loop_ranges {
            // A vreg is widened only if it has at least one endpoint
            // STRICTLY OUTSIDE the loop range — i.e. it's genuinely
            // connected to the loop from outside (a pre-loop init reaching
            // the header, or a post-loop use of a value the body last
            // wrote), not merely a vreg whose entire def/use span happens
            // to fall inside [loop_start, loop_end] by coincidence.
            //
            // `reverse_postorder` is free to lay out a SIBLING branch off
            // the loop header (e.g. the `if_false` arm of the loop's own
            // condition, leading somewhere else entirely) positionally
            // BETWEEN the header and the body/latch — the range check
            // alone can't tell that apart from a real loop-carried value.
            // Requiring containment to be non-total closes exactly that
            // gap: a vreg whose whole lifetime is inside the range (both
            // endpoints inside) never needs the loop to keep its slot
            // "live" for a next iteration that never reads it, so leaving
            // it alone is strictly more precise, never wrong (found via
            // `cold_branch_recompile_spill_corruption.mst`/BUG D: a
            // second, still-cold `to:do:` loop's own init+trap blocks,
            // laid out between a first loop's header and latch, had their
            // OWN short-lived temps smeared across the first loop's ENTIRE
            // body — falsely marking their spill slots live at real call
            // sites inside it that never write them, so the GC read
            // uninitialized frame memory as an oop there).
            let touched: Vec<u32> = min_def
                .keys()
                .chain(max_use.keys())
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .filter(|&v| {
                    let s = *min_def.get(&v).unwrap_or(&u32::MAX);
                    let e = *max_use.get(&v).unwrap_or(&0);
                    s <= loop_end && e >= loop_start && (s < loop_start || e > loop_end)
                })
                .collect();
            for v in touched {
                let s = min_def.entry(v).or_insert(loop_start);
                if *s > loop_start {
                    *s = loop_start;
                    changed = true;
                }
                let e = max_use.entry(v).or_insert(loop_end);
                if *e < loop_end {
                    *e = loop_end;
                    changed = true;
                }
            }
        }
    }

    // S14 step 7-II-b's ctx-temps (`deopt_live_widen`): their write genuinely
    // DOMINATES every later safepoint (Smalltalk nil-initializes every
    // declared temp unconditionally at method entry, before any branch), so
    // widening `[start,end]` out to `sp_pos + 1` — the ORIGINAL S13 step 7b
    // mechanism — is sound for them specifically. Left unchanged.
    for &(v, sp_pos) in &deopt_live_widen {
        min_def.entry(v).or_insert(0);
        let e = max_use.entry(v).or_insert(sp_pos + 1);
        if *e <= sp_pos {
            *e = sp_pos + 1;
        }
    }

    // S13 step 7b: every deopt-referenced vreg must be SPILLED (a stable
    // frame slot, not a register that a later call/branch could clobber or
    // that regalloc could hand to a different interval once it thinks this
    // one is dead) so the deopt materializer / GC root scan can find it at
    // its own recorded safepoint. Originally this ALSO widened the vreg's
    // plain `[start,end]` interval out to `sp_pos + 1` — which works for the
    // spill decision (a boolean: does ANY safepoint fall in range) but is
    // unsound for `oopmap::build_for_position`'s PER-SAFEPOINT liveness
    // check, since a single interval can't express "live at my own organic
    // uses, and ALSO at this one far-away trap, but not at everything
    // numerically in between." A trap is typically linearized far down in
    // the cold tail (`emit_uncommon_trap`'s own doc), so "everything in
    // between" routinely includes OTHER blocks entirely — e.g. an if/else's
    // shared post-merge continuation, reachable from a SIBLING arm that
    // never wrote this vreg's slot at all. That continuation's own,
    // unrelated safepoints would then wrongly see the slot as a live oop.
    // (Found via `cold_branch_recompile_spill_corruption.mst`, a second
    // instance of BUG D: `process:`'s inlined `add1:` arm's `payload` temp —
    // needed only by ITS OWN smi-overflow trap in the cold tail — bled
    // "live" into the shared continuation also reachable from the sibling
    // `add2:` arm, which never touches that slot; a debug-build "mark tag"
    // GC panic caught it reading raw, never-written stack memory as an oop.
    // The SAME shape as the earlier loop-widening bug in this same
    // function, one layer up: a position-interval standing in for real
    // per-branch liveness, unsound wherever it's asked to span a merge.
    //
    // UNLIKE `deopt_live_widen`'s ctx-temps above, a plain trap's own
    // receiver/slots/recorded-stack vregs have NO such dominance guarantee
    // — they're ordinary values, not unconditionally-initialized declared
    // temps, so a sibling arm can easily reach a later safepoint without
    // ever having written them.
    //
    // Fixed by keeping `min_def`/`max_use` at their ORGANIC values (a vreg
    // referenced only here, never by any op — a slot for an argument the
    // body never touches — still gets a bare `[0,0]` so it's assignable at
    // all) and recording each exact `(vreg, trap position)` pair separately
    // in `extra_oop_live` instead of folding it into the interval.
    // `crosses_safepoint` ORs in plain membership (deopt_referenced) so the
    // spill decision is unaffected; `oopmap::build_for_position` checks
    // `extra_oop_live` as an ADDITIONAL, exact-position fact alongside the
    // (now unwidened) interval.
    let deopt_referenced: std::collections::HashSet<u32> = deopt_live_exact
        .iter()
        .map(|&(v, _)| v)
        .chain(deopt_live_widen.iter().map(|&(v, _)| v))
        .collect();
    let extra_oop_live: Vec<(VReg, u32)> = deopt_live_exact
        .iter()
        .map(|&(v, pos)| (VReg(v), pos))
        .collect();
    for &v in &deopt_referenced {
        min_def.entry(v).or_insert(0);
        max_use.entry(v).or_insert(0);
    }

    let intervals = (0..method.vregs.len() as u32)
        .filter_map(|vid| {
            let start = *min_def.get(&vid)?;
            let end = *max_use.get(&vid).unwrap_or(&start);
            let crosses_safepoint = deopt_referenced.contains(&vid)
                || safepoint_positions
                    .iter()
                    .any(|&sp| start <= sp && end > sp);
            let crosses_call = call_positions.iter().any(|&cp| start <= cp && end > cp);
            Some(LiveInterval {
                vreg: VReg(vid),
                start,
                end,
                is_oop: method.vregs[vid as usize].is_oop,
                is_fp: method.vregs[vid as usize].is_fp,
                crosses_safepoint,
                crosses_call,
                assignment: None,
                resident_reg: None,
            })
        })
        .collect();

    (
        block_order,
        intervals,
        safepoint_positions,
        block_start_pos,
        extra_oop_live,
    )
}

// ── The register file (the ONLY architecture-specific part of this module;
//    the scan, spill policy, oop-map bookkeeping, and every public type are
//    arch-neutral and shared verbatim between targets — WINVM MIGRATION.md
//    §2.1). Pools are explicit lists of ARCHITECTURAL register numbers, not
//    a dense `0..N` count, because x86-64's allocatable set has holes (RSP
//    and RBP sit in the middle of the numbering).

/// x0–x15 (`arm64.md` §3); x16/x17 scratch, x18 platform, x19/x20 alloc
/// scratch, x21–x27 the S14 residency pool (below), x28 = &VmState,
/// x29/x30/sp — none of those are linear-scan allocatable.
#[cfg(target_arch = "aarch64")]
const ALLOCATABLE_REGS: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// WINVM x86-64: RCX, RDX, RBX, RSI, RDI, R8, R9 — seven, against
/// AArch64's sixteen. Excluded and why:
/// - **RAX (0)** — emit's third scratch: the two-address fixup temp and
///   the fixed operand of `imul`/`idiv`, plus the ABI return register.
/// - **RSP (4) / RBP (5)** — stack and frame pointer; spill slots are
///   `[rbp − 8·(i+1)]` and the GC/debugger walk the RBP chain.
/// - **R10 (10) / R11 (11)** — emit's spill-reload scratches, the x16/x17
///   analogues. R10 is additionally the VEH's deopt trap-pc stash.
/// - **R12–R15** — the pinned VM registers (method, bcp, receiver,
///   &VmState).
///
/// The pressure this creates is real and expected (MIGRATION.md §2.1
/// flags it): more spills than AArch64. It is survivable because the
/// spill-all-at-safepoints policy below already keeps the register file
/// idle across calls, and the residency tier reclaims what the scan
/// leaves unused.
#[cfg(not(target_arch = "aarch64"))]
const ALLOCATABLE_REGS: &[u8] = &[1, 2, 3, 6, 7, 8, 9];

/// Float fast-path FP pool: `d0`–`d7`, caller-saved scratch — zero
/// prologue/epilogue cost, clobbered by any call, which is safe because a
/// crossing-safepoint fp interval is spilled (spill-all) exactly like a GPR
/// one. `d8`–`d15` (callee-saved; the write-through residency tier) and
/// `d16`/`d17` (emit's fp spill scratch, mirroring x16/x17) stay out of the
/// pool.
#[cfg(target_arch = "aarch64")]
const FP_ALLOCATABLE_REGS: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7];

/// WINVM x86-64: `xmm0`–`xmm2`, all volatile under Win64 — the same
/// "zero prologue cost, clobbered by any call, safe because crossing
/// intervals are spilled" rationale as the AArch64 pool.
///
/// `xmm3`–`xmm5` are reserved as emit's FP scratch (the `d16`/`d17`
/// analogue, and volatile so they need no save). THREE, not one, and the
/// reason is x86-64's two-address form: `dst = a - b` where `dst` and `b`
/// are the same register needs somewhere to park `b` before `dst` is
/// overwritten, and both `a` and `b` may ALREADY be occupying scratch
/// registers if each was spilled. One reload scratch per operand plus one
/// for the shuffle is the worst case, and paying it in register budget is
/// better than a subtly wrong `subsd`.
///
/// Three allocatable FP registers is tight but adequate: float kernels
/// keep few values simultaneously live, and anything crossing a safepoint
/// is spilled regardless. `xmm6`–`xmm15` are callee-saved on Win64 and
/// would each cost a prologue `movsd` — deliberately unused until the
/// residency tier claims them with an explicit save/restore, which the
/// call stub asserts against (`fp_pool_is_empty_or_this_stub_must_save_xmm`).
#[cfg(not(target_arch = "aarch64"))]
const FP_ALLOCATABLE_REGS: &[u8] = &[0, 1, 2];

/// The FP allocatable pool, for cross-module invariant checks — the x64
/// call stub asserts against it that it is not required to save any
/// callee-saved XMM (`codecache::stubs_x64`). An accessor rather than a
/// `pub` const so the pool stays this module's business to define.
pub fn fp_allocatable_regs() -> &'static [u8] {
    FP_ALLOCATABLE_REGS
}

/// Allocatable registers the residency tier may claim when the main scan
/// leaves them globally unused. Excludes the ABI argument/result registers
/// (written mid-body by call marshalling and the allocation slow path):
/// x0–x5 on AArch64; RAX/RCX/RDX/R8/R9 on x86-64, which leaves exactly the
/// callee-saved allocatable three (RBX, RSI, RDI).
#[cfg(target_arch = "aarch64")]
const RESIDENCY_CANDIDATES: &[u8] = &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
#[cfg(not(target_arch = "aarch64"))]
const RESIDENCY_CANDIDATES: &[u8] = &[3, 6, 7];

/// One past the highest architectural register number either file uses —
/// the size of the `reg_used` marking array. Register NUMBERS index it, so
/// it is emphatically not `ALLOCATABLE_REGS.len()` (on x86-64 the pool has
/// 7 entries but numbers run up to 9).
const MAX_REG_NUM: usize = 16;

/// D3.5's policy, in order: (1) every `crosses_safepoint` interval spills
/// unconditionally, whole-lifetime, before the main scan even starts — the
/// invariant S12's oop maps stand on (registers are never live across a
/// safepoint; maps cover stack slots only), enforced here via a
/// `debug_assert!` rather than merely relied upon. (2) Remaining intervals:
/// classic linear scan — sorted by start, an active list expired as
/// intervals end, and when all registers are busy, the active interval
/// with the furthest end is spilled to make room (Poletto/Sarkar). (3)
/// Spill slots are handed out monotonically; each records its interval's
/// `is_oop` — the raw material for S12's `OopMap`s.
/// S14 perf recovery: give call-free SPILLED intervals a RESIDENT register
/// (x21–x23, callee-saved; disjoint from the x0–x15 allocatable pool and from
/// emit's x16/x17/x19/x20 scratches). Longest intervals first — loop-carried
/// variables win the registers. The slot stays canonical (write-through); see
/// [`LiveInterval::resident_reg`].
pub fn assign_residents(intervals: &mut [LiveInterval]) {
    // Base pool: x21–x27 (callee-saved, never touched by emit's scratches
    // x16/x17/x19/x20 or the ABI paths; x24–x27 had no role of MACVM's own —
    // VMregisters.md — and build_call_stub already saves the whole x19–x28
    // bank at the boundary, so claiming them here costs nothing anywhere).
    // EXTENDED by every x6–x15 register the main allocator left GLOBALLY
    // unused in this method — in the spill-heavy hot methods residency exists
    // for, nearly all of them (the whole point is that spill-all left the
    // register file idle). x0–x5 stay out (ABI argument/result/alloc-slow
    // paths write them mid-body).
    // WINVM: on AArch64 this base pool is DISJOINT from the allocatable
    // file — x21–x27 are callee-saved registers with no other role, free
    // for the taking. x86-64 has no such surplus: every callee-saved
    // register is either pinned (R12–R15, RBP) or already allocatable
    // (RBX, RSI, RDI). So the x64 base pool is EMPTY and residency draws
    // purely from the extension below — the registers the main scan left
    // globally unused. That is sound for exactly the reason the AArch64
    // extension is: residency only ever claims `!crosses_call` intervals,
    // so a volatile register is as good as a callee-saved one here.
    #[cfg(target_arch = "aarch64")]
    let mut pool: Vec<u8> = vec![21, 22, 23, 24, 25, 26, 27];
    #[cfg(not(target_arch = "aarch64"))]
    let mut pool: Vec<u8> = Vec::new();
    let mut reg_used = [false; MAX_REG_NUM];
    for iv in intervals.iter() {
        // An fp interval's Reg(n) names dN, not xN — a different file
        // entirely; it neither occupies nor frees a GPR here.
        if iv.is_fp {
            continue;
        }
        if let Some(Assignment::Reg(r)) = iv.assignment {
            reg_used[r as usize] = true;
        }
    }
    for &r in RESIDENCY_CANDIDATES {
        if !reg_used[r as usize] {
            pool.push(r);
        }
    }

    let mut taken: Vec<Vec<(u32, u32)>> = vec![Vec::new(); pool.len()];
    // Float fast-path residency: spilled fp intervals get a callee-saved
    // d8–d15 register the same way (write-through: the canonical non-oop
    // slot stays authoritative for deopt; reads prefer the register). Same
    // !crosses_call gate — a resident interval never spans a CallSend, so
    // compiled callees can't clobber it; the Poll/Alloc/FBox SLOW paths
    // (`bl` into Rust, which uses d8–d15 freely) already re-load residents.
    // Floats need no GC resync at all beyond that — a raw f64 never moves.
    // WINVM x86-64: empty. Win64's callee-saved XMMs (xmm6–xmm15) each
    // cost a prologue/epilogue save, which the AArch64 tier never had to
    // pay (d8–d15 are saved by the call stub's own bank). Wiring that is
    // Phase 5 work, alongside the float regions themselves; until then no
    // fp interval gets a resident and the canonical slot is simply read
    // back. Correctness is unaffected — residency is a pure optimization.
    #[cfg(target_arch = "aarch64")]
    let fp_pool: Vec<u8> = (8..16).collect();
    #[cfg(not(target_arch = "aarch64"))]
    let fp_pool: Vec<u8> = Vec::new();
    let mut fp_taken: Vec<Vec<(u32, u32)>> = vec![Vec::new(); fp_pool.len()];
    let mut order: Vec<usize> = (0..intervals.len())
        .filter(|&i| {
            matches!(intervals[i].assignment, Some(Assignment::Spill(_)))
                && !intervals[i].crosses_call
                && intervals[i].end > intervals[i].start
        })
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(intervals[i].end - intervals[i].start));
    for i in order {
        let (s, e) = (intervals[i].start, intervals[i].end);
        let (p, t) = if intervals[i].is_fp {
            (&fp_pool, &mut fp_taken)
        } else {
            (&pool, &mut taken)
        };
        for (ri, reg) in p.iter().enumerate() {
            if t[ri].iter().all(|&(ts, te)| e <= ts || te <= s) {
                t[ri].push((s, e));
                intervals[i].resident_reg = Some(*reg);
                break;
            }
        }
    }
}

pub fn allocate(intervals: &mut [LiveInterval]) -> (u16, Vec<bool>) {
    let mut slot_is_oop: Vec<bool> = Vec::new();
    let spill = |iv: &mut LiveInterval, slot_is_oop: &mut Vec<bool>| {
        let slot = SpillSlot(slot_is_oop.len() as u16);
        slot_is_oop.push(iv.is_oop);
        iv.assignment = Some(Assignment::Spill(slot));
    };

    for iv in intervals.iter_mut() {
        if iv.crosses_safepoint {
            spill(iv, &mut slot_is_oop);
        }
    }

    let mut order: Vec<usize> = (0..intervals.len())
        .filter(|&i| intervals[i].assignment.is_none())
        .collect();
    order.sort_by_key(|&i| intervals[i].start);

    // Two independent register files (docs/float_fastpath_design.md B4):
    // GPR x0..x15 for ordinary vregs, FP d0..d7 (caller-saved scratch — no
    // prologue cost) for unboxed-f64 vregs. Same linear scan, two disjoint
    // free/active pools selected by `is_fp`; eviction only ever considers
    // the same class (a d-reg can't satisfy a GPR interval or vice versa).
    let mut active: Vec<usize> = Vec::new();
    let mut active_fp: Vec<usize> = Vec::new();
    let mut free_regs: Vec<u8> = ALLOCATABLE_REGS.iter().rev().copied().collect();
    let mut free_fp_regs: Vec<u8> = FP_ALLOCATABLE_REGS.iter().rev().copied().collect();

    for i in order {
        let start = intervals[i].start;
        let is_fp = intervals[i].is_fp;
        {
            let (act, free) = if is_fp {
                (&mut active_fp, &mut free_fp_regs)
            } else {
                (&mut active, &mut free_regs)
            };
            act.retain(|&j| {
                if intervals[j].end < start {
                    if let Some(Assignment::Reg(r)) = intervals[j].assignment {
                        free.push(r);
                    }
                    false
                } else {
                    true
                }
            });
        }
        // Also expire the OTHER class's dead intervals so its free list is
        // current when its next interval starts (harmless bookkeeping).
        let (act, free) = if is_fp {
            (&mut active_fp, &mut free_fp_regs)
        } else {
            (&mut active, &mut free_regs)
        };

        if let Some(r) = free.pop() {
            intervals[i].assignment = Some(Assignment::Reg(r));
            act.push(i);
        } else {
            let (pos_in_active, &furthest) = act
                .iter()
                .enumerate()
                .max_by_key(|&(_, &j)| intervals[j].end)
                .expect(
                    "allocate: no free register and no active interval to spill -- \
                     NUM_ALLOCATABLE_REGS must be wrong if this fires",
                );
            if intervals[furthest].end > intervals[i].end {
                let r = match intervals[furthest].assignment {
                    Some(Assignment::Reg(r)) => r,
                    _ => unreachable!("active intervals always hold a register"),
                };
                spill(&mut intervals[furthest], &mut slot_is_oop);
                act.remove(pos_in_active);
                intervals[i].assignment = Some(Assignment::Reg(r));
                act.push(i);
            } else {
                spill(&mut intervals[i], &mut slot_is_oop);
            }
        }
    }

    verify_spill_all(intervals);

    (slot_is_oop.len() as u16, slot_is_oop)
}

/// S12 D1's spill-all invariant, enforced HERE — not merely assumed, and
/// not merely `debug_assert!`ed: this is the exact guarantee S12's oop maps
/// stand on (registers are never live across a safepoint; maps cover stack
/// slots only), so it runs ALWAYS, release builds included, per the sprint
/// doc's own text ("a release-mode-cheap pass... trivial", O(intervals)).
/// A future regalloc change that lets a `crosses_safepoint` interval keep a
/// register would otherwise corrupt the heap silently instead of panicking
/// at the source — exactly the class of bug a debug-only check would only
/// catch in SOME builds.
pub fn verify_spill_all(intervals: &[LiveInterval]) {
    for iv in intervals {
        assert!(
            !(iv.crosses_safepoint && matches!(iv.assignment, Some(Assignment::Reg(_)))),
            "regalloc: {:?} crosses a safepoint but holds a register (S12's oop-map \
             invariant: registers are all spilled at safepoints)",
            iv.vreg
        );
    }
}

pub struct RegallocResult {
    pub block_order: Vec<BlockId>,
    /// Final intervals, `assignment` populated — indexed arbitrarily (by
    /// `compute_intervals`' own vreg-ascending order), not by vreg id;
    /// look up by `.vreg` if you need a specific one.
    pub intervals: Vec<LiveInterval>,
    pub frame_slots: u16,
    pub slot_is_oop: Vec<bool>,
    /// S12: every safepoint's exact linear position, in the SAME numbering
    /// `intervals`' own `start`/`end`/`crosses_safepoint` were computed
    /// against — `emit.rs` walks `block_order` identically (its own
    /// position counter) to correlate each REAL emitted safepoint with one
    /// of these, and `compiler::oopmap::build_for_position` intersects
    /// `intervals` against it to build that safepoint's own `OopMap`.
    pub safepoint_positions: Vec<u32>,
    /// S15 (OSR): each block id's first linear position, in the SAME
    /// numbering as `intervals`/`safepoint_positions` — the driver resolves
    /// the OSR header block's live-in entities against exactly this
    /// position, and emit reloads residents live there.
    pub block_start_pos: std::collections::HashMap<u32, u32>,
    /// S13 step 7b (bug-fix revision): exact `(vreg, safepoint position)`
    /// facts a deopt site's own recorded stack/slots need, kept SEPARATE
    /// from `intervals`' own `[start,end]` — folding these into the
    /// interval would widen it to cover EVERY position in between, which is
    /// unsound whenever that span crosses an if/else merge reachable from a
    /// sibling arm that never wrote this vreg at all (`compute_intervals`'
    /// own doc has the full story). `oopmap::build_for_position` checks
    /// this as an ADDITIONAL, exact-position fact alongside the interval.
    ///
    /// Task #94 extension: each deopt-referenced vreg's facts cover not
    /// just its trap's own position but EVERY earlier safepoint too — a GC
    /// striking mid-`CallSend` on the way to the trap must keep these
    /// slots current or the trap's materializer reads relocated-away
    /// addresses (see `compute_intervals`' task-#94 comment for the full
    /// mechanism and why `deopt_nil_init_slots` makes this sound).
    pub extra_oop_live: Vec<(VReg, u32)>,
    /// Task #94: spill slots `emit` must nil-fill in the prologue — the
    /// final slot of every deopt-referenced vreg. A safepoint reached
    /// before the vreg's def (or via a sibling arm that never wrote it)
    /// then scans nil instead of uninitialized native stack, which is what
    /// makes `extra_oop_live`'s earlier-safepoint facts sound without
    /// path-sensitive liveness. Sorted, deduplicated.
    pub deopt_nil_init_slots: Vec<SpillSlot>,
}

pub fn regalloc(method: &IrMethod) -> RegallocResult {
    let (block_order, mut intervals, safepoint_positions, block_start_pos, extra_oop_live) =
        compute_intervals(method);
    // S24 A1 (design Risk 1): PIN the block compilation's closure vreg live
    // for the whole method — the root deopt scope's receiver ValueLoc names
    // its spill slot, and `Ir::NlrReturn` reads it, at ANY safepoint. A
    // liveness-derived interval would end at the closure's last textual use
    // (often the entry prologue), leaving the recorded slot dead/garbage at
    // a later deopt (found immediately by depth3_deopt: the standalone-
    // compiled block's receiver slot held junk). Widening the interval end
    // to the method's last position makes spill-all keep the slot canonical
    // and every liveness-intersected oopmap include it — one slot, and the
    // whole analysis dimension disappears.
    if let Some(cv) = method.block_closure_vreg {
        // +2: resolve_frame_loc/build_for_position use STRICT upper bounds
        // (`iv.end > pos`), so ending exactly AT the last position would
        // resolve Nil at the final safepoint -- the very deopt this exists
        // for (observed: scope_recv=Nil on depth3's block deopt).
        let max_pos = intervals.iter().map(|iv| iv.end).max().unwrap_or(0) + 2;
        if let Some(iv) = intervals.iter_mut().find(|iv| iv.vreg == cv) {
            iv.end = max_pos;
            iv.crosses_safepoint = true;
        }
    }
    // S24 A3b (design Risk 1/2): PIN M's MATERIALIZED Context vreg live for
    // the whole method — EVERY safepoint's root deopt scope names its slot
    // (`CtxLoc::Materialized`), and it is the SAME object escaped closures
    // reference. Its compiled uses end at the last ctx-temp op, but an
    // uncommon-trap that TERMINATES a block before that op would leave the
    // slot dead — `resolve_frame_loc` then returns `Nil`, the materializer
    // writes nil into the rebuilt frame's Context, and a ctx-temp read after
    // the deopt corrupts (observed: `printOn:` with a captured stream, ctxloc
    // -> Nil at the trap). Same one-slot pin as `block_closure_vreg` above.
    if let Some((cv, _nctx)) = method.method_ctx_vreg {
        let max_pos = intervals.iter().map(|iv| iv.end).max().unwrap_or(0) + 2;
        if let Some(iv) = intervals.iter_mut().find(|iv| iv.vreg == cv) {
            iv.end = max_pos;
            iv.crosses_safepoint = true;
        }
    }
    let (frame_slots, slot_is_oop) = allocate(&mut intervals);
    // S14 perf recovery: call-free spilled intervals also get a resident
    // register (slots stay canonical; see LiveInterval::resident_reg).
    assign_residents(&mut intervals);
    // Task #94: the final spill slot of every deopt-referenced vreg (all are
    // spill-assigned — `crosses_safepoint` is forced for them). `emit`
    // nil-fills these in the prologue, which is what makes `extra_oop_live`'s
    // earlier-safepoint facts sound on paths that never wrote the slot (see
    // `compute_intervals`' own task-#94 comment).
    let mut deopt_nil_init_slots: Vec<SpillSlot> = {
        let referenced: std::collections::HashSet<u32> =
            extra_oop_live.iter().map(|&(v, _)| v.0).collect();
        intervals
            .iter()
            .filter(|iv| referenced.contains(&iv.vreg.0))
            .filter_map(|iv| match iv.assignment {
                Some(Assignment::Spill(slot)) => Some(slot),
                _ => None,
            })
            .collect()
    };
    deopt_nil_init_slots.sort_by_key(|s| s.0);
    deopt_nil_init_slots.dedup();
    RegallocResult {
        block_order,
        intervals,
        frame_slots,
        slot_is_oop,
        safepoint_positions,
        block_start_pos,
        extra_oop_live,
        deopt_nil_init_slots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{BailoutReason, CmpOp, PoolLit, SmiOp, StubId, VRegInfo};

    fn hand_method(blocks: Vec<IrBlock>, vregs: Vec<VRegInfo>) -> IrMethod {
        IrMethod {
            blocks,
            vregs,
            pool: Vec::new(),
            argc: 0,
            ntemps: 0,
            ctx_vregs: Vec::new(),
            block_closure_vreg: None,
            method_ctx_vreg: None,
            spliced_nlr: 0,
            spliced_multibb: 0,
            splice_declined_budget: 0,
            safepoints: Vec::new(),
            true_lit: PoolLit(0),
            false_lit: PoolLit(0),
            nil_lit: PoolLit(0),
            mark_slots_lit: PoolLit(0),
            mark_double_lit: PoolLit(0),
            double_klass_lit: PoolLit(0),
            float64x2_klass_lit: PoolLit(0),
            float32x4_klass_lit: PoolLit(0),
            int32x4_klass_lit: PoolLit(0),
            call_sites: Vec::new(),
            site_feedback: Vec::new(),
            inline_deps: Vec::new(),
            self_devirt: false,
            method_pool_ix: None,
        }
    }

    /// "hand IR: def at 2, uses at 5 and 9" -> interval `[2, 9]`.
    #[test]
    fn intervals_basic() {
        let v0 = VReg(0);
        let filler = VReg(1);
        let block = IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::Poll,
                Ir::Poll,
                Ir::ConstSmi { dst: v0, value: 1 }, // pos 2: def
                Ir::Poll,
                Ir::Poll,
                Ir::Move {
                    dst: filler,
                    src: v0,
                }, // pos 5: use
                Ir::Poll,
                Ir::Poll,
                Ir::Poll,
                Ir::Ret { val: v0 }, // pos 9: use
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let method = hand_method(
            vec![block],
            vec![VRegInfo { is_oop: true, is_fp: false }, VRegInfo { is_oop: true, is_fp: false }],
        );

        let (_order, intervals, _safepoints, _bsp, _extra) = compute_intervals(&method);
        let iv = intervals
            .iter()
            .find(|iv| iv.vreg == v0)
            .expect("v0 has an interval");
        assert_eq!(iv.start, 2);
        assert_eq!(iv.end, 9);
    }

    /// A temp vreg defined on two different blocks (SSA-lite's multiple-
    /// defs shape) gets ONE interval covering every def and the eventual
    /// use, not two separate intervals.
    #[test]
    fn interval_multi_def_union() {
        let v0 = VReg(0);
        let block0 = IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::ConstSmi { dst: v0, value: 1 },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let block1 = IrBlock {
            id: BlockId(1),
            bci: 10,
            code: vec![Ir::ConstSmi { dst: v0, value: 2 }, Ir::Ret { val: v0 }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let method = hand_method(vec![block0, block1], vec![VRegInfo { is_oop: true, is_fp: false }]);

        let (order, intervals, _safepoints, _bsp, _extra) = compute_intervals(&method);
        assert_eq!(
            order,
            vec![BlockId(0), BlockId(1)],
            "block0 must be linearized first"
        );
        assert_eq!(intervals.len(), 1, "one vreg -> one interval, never two");
        assert_eq!(intervals[0].start, 0);
        assert_eq!(intervals[0].end, 3);
    }

    /// THE S12 invariant, enforced early: every oop interval live across a
    /// `CallRuntime` gets `Spill`, never `Reg`.
    #[test]
    fn spill_all_crossing_safepoint() {
        let v0 = VReg(0);
        let block = IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::ConstPool {
                    dst: v0,
                    lit: PoolLit(0),
                },
                Ir::CallRuntime {
                    dst: None,
                    stub: StubId(0),
                    args: vec![],
                },
                Ir::Ret { val: v0 },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let method = hand_method(vec![block], vec![VRegInfo { is_oop: true, is_fp: false }]);

        let (_order, mut intervals, _safepoints, _bsp, _extra) = compute_intervals(&method);
        assert!(
            intervals[0].crosses_safepoint,
            "v0 is defined before and used after the call"
        );

        allocate(&mut intervals);
        assert!(matches!(
            intervals[0].assignment,
            Some(Assignment::Spill(_))
        ));
    }

    /// Oop-map raw material: `slot_is_oop` records each spill slot's
    /// interval's own `is_oop`, correctly per-slot.
    #[test]
    fn spill_slot_oopness_recorded() {
        let mut intervals = vec![
            LiveInterval {
                vreg: VReg(0),
                start: 0,
                end: 5,
                is_oop: true,
                is_fp: false,
                crosses_safepoint: true,
                crosses_call: false,
                resident_reg: None,
                assignment: None,
            },
            LiveInterval {
                vreg: VReg(1),
                start: 0,
                end: 5,
                is_oop: false,
                is_fp: false,
                crosses_safepoint: true,
                crosses_call: false,
                resident_reg: None,
                assignment: None,
            },
        ];

        let (frame_slots, slot_is_oop) = allocate(&mut intervals);
        assert_eq!(frame_slots, 2);
        assert_eq!(slot_is_oop.len(), 2);

        let slot_of = |iv: &LiveInterval| match iv.assignment {
            Some(Assignment::Spill(s)) => s.0 as usize,
            _ => panic!("expected a spill assignment"),
        };
        assert!(slot_is_oop[slot_of(&intervals[0])]);
        assert!(!slot_is_oop[slot_of(&intervals[1])]);
    }

    /// `tests_s12.md`'s `verify_spill_all_catches_reg` (D1 enforcement
    /// point 1): a hand-built interval claiming BOTH `crosses_safepoint`
    /// AND a `Reg` assignment is exactly the invariant violation S12's oop
    /// maps depend on never happening — `verify_spill_all` must panic on
    /// it directly, independent of whether `allocate` itself could ever
    /// actually produce such a state (this test constructs the bad state
    /// by hand, bypassing `allocate` entirely, to test the CHECK, not the
    /// policy that normally prevents it).
    #[test]
    #[should_panic(expected = "crosses a safepoint but holds a register")]
    fn verify_spill_all_catches_reg() {
        let intervals = vec![LiveInterval {
            vreg: VReg(0),
            start: 0,
            end: 10,
            is_oop: true,
            is_fp: false,
            crosses_safepoint: true,
            crosses_call: false,
            resident_reg: None,
            assignment: Some(Assignment::Reg(3)),
        }];
        verify_spill_all(&intervals);
    }

    /// The same shape, but `Spill`-assigned (the correct outcome) — must
    /// NOT panic, so the test above is exercising the actual invariant,
    /// not merely "any crosses_safepoint interval panics".
    #[test]
    fn verify_spill_all_accepts_spilled_crossing_interval() {
        let intervals = vec![LiveInterval {
            vreg: VReg(0),
            start: 0,
            end: 10,
            is_oop: true,
            is_fp: false,
            crosses_safepoint: true,
            crosses_call: false,
            resident_reg: None,
            assignment: Some(Assignment::Spill(SpillSlot(0))),
        }];
        verify_spill_all(&intervals); // must not panic
    }

    /// WINVM: the x86-64 register file must never offer a register that
    /// has another job. Handing out RSP or RBP would corrupt the stack or
    /// the frame chain the GC and debugger walk; handing out R12–R15 would
    /// silently clobber `&VmState`/receiver/bcp/method; handing out
    /// R10/R11 would collide with emit's spill-reload scratches (and R10
    /// with the VEH's trap-pc stash); handing out RAX would collide with
    /// emit's two-address fixup temp and `imul`/`idiv`. Each of those is a
    /// silent-corruption bug, not a crash, so the pools are asserted
    /// directly rather than left to be discovered downstream.
    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn x64_register_file_excludes_every_reserved_register() {
        use crate::compiler::assembler_x64 as x64;
        for (num, name) in [
            (x64::RSP, "RSP (stack pointer)"),
            (x64::RBP, "RBP (frame pointer / spill base)"),
            (x64::R12, "R12 (method)"),
            (x64::R13, "R13 (bytecode pointer)"),
            (x64::R14, "R14 (receiver)"),
            (x64::R15, "R15 (&VmState)"),
            (x64::R10, "R10 (emit scratch / VEH trap-pc stash)"),
            (x64::R11, "R11 (emit scratch)"),
            (x64::RAX, "RAX (emit two-address temp / imul / idiv / ABI return)"),
        ] {
            assert!(
                !ALLOCATABLE_REGS.contains(&num),
                "{name} must never be linear-scan allocatable"
            );
            assert!(
                !RESIDENCY_CANDIDATES.contains(&num),
                "{name} must never be a residency candidate"
            );
        }
        // And the pool is exactly the seven MIGRATION.md §2.1 names.
        assert_eq!(
            ALLOCATABLE_REGS,
            &[x64::RCX, x64::RDX, x64::RBX, x64::RSI, x64::RDI, x64::R8, x64::R9]
        );
        // Residency candidates must themselves be allocatable (the tier
        // reclaims scan leftovers) and callee-saved (never an ABI arg).
        for r in RESIDENCY_CANDIDATES {
            assert!(ALLOCATABLE_REGS.contains(r));
            assert!(
                !x64::ARG_REGS.contains(r),
                "an ABI argument register is written mid-body; it cannot host a resident"
            );
        }
    }

    /// Linear-scan core: with N allocatable registers, N+1 mutually
    /// overlapping (call-free) intervals force exactly one spill — the
    /// furthest-ending one, whether it's encountered first or last.
    ///
    /// WINVM: expressed in terms of [`ALLOCATABLE_REGS`] rather than the
    /// old hardcoded 16/17, so it states the *invariant* (one more live
    /// value than registers costs exactly one spill, and it's the furthest-
    /// ending one) on every target instead of AArch64's register count.
    #[test]
    fn furthest_end_spilled_under_pressure() {
        let n = ALLOCATABLE_REGS.len() as u32;
        let mut intervals = vec![LiveInterval {
            vreg: VReg(0),
            start: 0,
            end: 999,
            is_oop: false,
            is_fp: false,
            crosses_safepoint: false,
            crosses_call: false,
            resident_reg: None,
            assignment: None,
        }];
        for i in 1..=n {
            intervals.push(LiveInterval {
                vreg: VReg(i),
                start: 0,
                end: 10,
                is_oop: false,
                is_fp: false,
                crosses_safepoint: false,
                crosses_call: false,
                resident_reg: None,
                assignment: None,
            });
        }
        assert_eq!(intervals.len(), n as usize + 1);

        let (frame_slots, _slot_is_oop) = allocate(&mut intervals);
        assert_eq!(frame_slots, 1, "exactly one spill");

        let spilled: Vec<&LiveInterval> = intervals
            .iter()
            .filter(|iv| matches!(iv.assignment, Some(Assignment::Spill(_))))
            .collect();
        assert_eq!(spilled.len(), 1);
        assert_eq!(
            spilled[0].vreg,
            VReg(0),
            "the furthest-ending interval (end=999) is spilled"
        );

        let regs: std::collections::HashSet<u8> = intervals[1..]
            .iter()
            .map(|iv| match iv.assignment {
                Some(Assignment::Reg(r)) => r,
                _ => panic!("expected every other interval to hold a register"),
            })
            .collect();
        assert_eq!(
            regs.len(),
            ALLOCATABLE_REGS.len(),
            "every allocatable register used, none double-booked"
        );
        assert!(
            regs.iter().all(|r| ALLOCATABLE_REGS.contains(r)),
            "the allocator only ever hands out registers from the pool — \
             a number outside it would be a wrong (or reserved) register: {regs:?}"
        );
    }

    /// Sanity check that `reverse_postorder`/`compute_intervals` don't
    /// panic or hang on a block unreachable from the entry (decode's own
    /// `unreachable_after_return` shape, surviving into the IR).
    #[test]
    fn unreachable_block_gets_a_position_not_a_panic() {
        let block0 = IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![Ir::RetSelf],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let dead = IrBlock {
            id: BlockId(1),
            bci: 5,
            code: vec![Ir::Bailout {
                reason: BailoutReason::SmiOpFailed,
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let method = hand_method(vec![block0, dead], Vec::new());
        let (order, _intervals, _safepoints, _bsp, _extra) = compute_intervals(&method);
        assert_eq!(
            order.len(),
            2,
            "both blocks get a position, reachable or not"
        );
        // The real bug this hand-built shape once caught (S10 step 9): with
        // NO graph edge from block 0 to `dead` (RetSelf/Bailout are both
        // absent from `successors`' match), block 0 is its own singleton
        // DFS component — a version of `reverse_postorder` that reversed
        // the whole accumulated postorder only once, at the end, put block
        // 0 SECOND, meaning emit.rs's prologue fell straight through into
        // the OTHER block first. `emit.rs` has no guard against this: it
        // just emits `block_order` in order, right after the prologue.
        assert_eq!(
            order[0],
            BlockId(0),
            "block 0 (the method's real entry) must always be emitted first"
        );
    }

    /// The second, deeper S10 step 9 bug this hand-built shape catches: an
    /// accumulator vreg (`s`, matching `sumTo:`'s own `s := s + i` loop)
    /// that is both defined AND used inside the loop body block, and ALSO
    /// read once more after the loop, at the exit block. `reverse_postorder`
    /// only promises block 0 first and predecessors-before-successors
    /// except across back edges — for a loop header with two successors
    /// (body, exit), it does not promise the body block comes before the
    /// exit block in the LINEAR position space (and for this exact shape,
    /// it doesn't: the exit block, a DFS dead end, finishes and gets
    /// pushed to postorder before the body block's own back-edge-laden
    /// subtree does). A `[min_def, max_use]` fold that never widens for
    /// back edges puts `s`'s LAST use at the exit block's read — entirely
    /// missing that the body block, which the linearization places AFTER
    /// the exit block, both reads AND redefines `s` on every iteration.
    /// Without the loop-range widening this test checks for, `s`'s
    /// register could be (and, before this fix, was) handed to some other
    /// vreg live only in the "later" body block, silently corrupting the
    /// accumulator — `sumTo: 10` returned 11 (the loop counter's own final
    /// value) instead of 55 the first time this ran through the real
    /// compiler, in `world/tests/tier1.mst`.
    #[test]
    fn loop_carried_vreg_interval_spans_whole_loop() {
        let s = VReg(0); // accumulator, live across the back edge
        let i = VReg(1); // loop counter
        let bound = VReg(2);
        let tmp = VReg(3);
        let one = VReg(4);
        let result = VReg(5);

        let entry = IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::ConstSmi { dst: s, value: 0 },
                Ir::ConstSmi { dst: i, value: 1 },
                Ir::ConstSmi {
                    dst: bound,
                    value: 10,
                },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let header = IrBlock {
            id: BlockId(1),
            bci: 10,
            code: vec![Ir::SmiCmpBr {
                op: CmpOp::Le,
                a: i,
                b: bound,
                if_true: BlockId(2),
                if_false: BlockId(3),
                fail: BlockId(4),
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let body = IrBlock {
            id: BlockId(2),
            bci: 20,
            code: vec![
                Ir::SmiArith {
                    op: SmiOp::Add,
                    dst: tmp,
                    a: s,
                    b: i,
                    fail: BlockId(4),
                },
                Ir::Move { dst: s, src: tmp }, // redefines s, deep in the body
                Ir::ConstSmi { dst: one, value: 1 },
                Ir::SmiArith {
                    op: SmiOp::Add,
                    dst: tmp,
                    a: i,
                    b: one,
                    fail: BlockId(4),
                },
                Ir::Move { dst: i, src: tmp },
                Ir::Jump { target: BlockId(1) }, // the back edge
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let exit = IrBlock {
            id: BlockId(3),
            bci: 30,
            code: vec![
                Ir::Move {
                    dst: result,
                    src: s,
                }, // reads s once, "before" the body in linear position
                Ir::Ret { val: result },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };
        let bailout = IrBlock {
            id: BlockId(4),
            bci: 40,
            code: vec![Ir::Bailout {
                reason: BailoutReason::SmiOpFailed,
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        };

        let method = hand_method(
            vec![entry, header, body, exit, bailout],
            (0..6).map(|_| VRegInfo { is_oop: true, is_fp: false }).collect(),
        );
        let (order, intervals, _safepoints, _bsp, _extra) = compute_intervals(&method);

        // Confirms this hand-built shape actually reproduces the bug's own
        // precondition: the exit block linearized before the body block.
        let pos_of = |bid: BlockId| order.iter().position(|&b| b == bid).unwrap();
        assert!(
            pos_of(BlockId(3)) < pos_of(BlockId(2)),
            "this test's whole point is a linearization where the loop exit \
             precedes the loop body — order was {order:?}"
        );

        let s_iv = intervals
            .iter()
            .find(|iv| iv.vreg == s)
            .expect("s has an interval");
        let body_last_pos = {
            // Sum instruction counts for every block up to and including
            // the body block (BlockId(2)), in `order`'s own sequence, then
            // back off one for its own LAST instruction's position (not
            // one-past-the-end).
            let mut pos = 0u32;
            for &bid in &order {
                let blk = &method.blocks[bid.0 as usize];
                pos += blk.code.len() as u32;
                if bid == BlockId(2) {
                    break;
                }
            }
            pos - 1
        };
        assert!(
            s_iv.end >= body_last_pos,
            "s's interval (end={}) must extend at least through the loop \
             body's own last position ({body_last_pos}) -- otherwise its \
             register is free to be handed to something else mid-loop",
            s_iv.end
        );
    }
}
