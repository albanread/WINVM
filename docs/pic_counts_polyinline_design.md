# PIC arm counts + polymorphic arm inlining — design (dart124 lessons items 2+3)

Status: DESIGN — implementation not started. Prerequisite reading:
`docs/dart124_compiler_lessons.md` items 2 (budgeted inliner) and 3
(poly-arm inlining); Dart reference `flow_graph_inliner.cc` (`CallSites`
ranking at 219-315, `PolymorphicInliner` at 1417-1902,
`BuildDecisionGraph` at 1652).

## Why counts come first

Two consumers, one substrate:

- `inline.rs` `decide_with_budget` trusts poly dominance ONLY at exactly
  two cases because "the interpreter's POLY array carries no counts, so
  dominance is a first-seen guess" (its own comment — the pinned
  restriction until counts arrive). Multi-arm inlining without counts
  would speculate on arm ORDER, which first-seen cannot justify.
- The Dart-style worklist inliner ranks call sites by
  `CallCount()/max_count >= 10%`. Without per-site counts there is no
  ranking signal at all.

## Where counts live and who pays

Mirror Dart's cost model exactly: **only the unoptimized tier counts.**

- Interpreter IC (`src/interpreter/ic.rs`): the poly `pairs` ArrayOop
  (today `(klass, method)` × `IC_POLY_MAX_PAIRS`) widens to triples
  `(klass, method, count-smi)` — counts as smis are GC-transparent and
  need no barrier (smi stores). Touch points: `poly_arity` (stride 3),
  `set_poly`/`alloc_poly_pairs` (`IC_POLY_ARRAY_LEN`), `reverify_poly`
  (carry counts through compaction), the interpreter poly-hit dispatch
  (saturating smi bump), and the S14 feedback readout. Mono sites count
  in the IC's spare meta word (site hotness for the worklist inliner).
- Compiled code: **no counting**. t=200 yields ~200 interpreted samples
  per site before compilation — the same signal budget Dart gets from
  its unoptimized tier at threshold 30k (scaled), and the compiled hot
  path stays untouched.
- Compiled-PIC misses that re-enter the runtime (`rt_interpret_call`'s
  upgrade hook, the L1 re-key arm) already cross into Rust: bump the
  matching interpreter-IC arm there too, so post-compile phase changes
  (a klass that only becomes hot AFTER tier-up) still register. This is
  the "compiled-PIC counts arrive on recompile" hook the inline.rs
  comment anticipated, at zero compiled-fast-path cost.

## Feedback snapshot

`feedback.rs` `SiteFeedback::Poly { cases }` carries counts, cases
sorted descending. **Profile-hash discipline** (the IC-stomp lesson):
the recompile-trigger hash must cover the SET of (klass, target) pairs
and the CHOSEN-ARM list — never raw counts — or count drift would flap
the hash and re-trigger compiles forever.

## Decision (inline.rs)

Extend `InlineDecision` with:

```rust
PolyInline {
    cases: Vec<(KlassOop, MethodOop)>, // 2..=4, count-descending
    // fallback is always a real compiled send — minorities are
    // known-taken; trapping would storm (SPEC §8.4 unchanged)
}
```

`decide_with_budget`, Poly arm, in order:
1. cases where every candidate arm is `primitive() == 0`, within
   `per_call_cost`, and leaf/nonleaf/CFG-eligible (same ladder as Mono),
   arm count ≥ 10% of site total (Dart `FLAG_inlining_hotness`), max 4
   arms (Richards' task set): → `PolyInline`.
2. else count-proven dominant (cases[0] by count, any cases.len()) and
   dominant is eligible: → `DominantWithSlowPath` (the len==2 pin
   DELETES — counts replace first-seen).
3. else → `Call`.

Duplicate targets (two klasses, one method — Dart's
`CheckInlinedDuplicate`): splice the body once, point both guards at it.
First slice may simply not dedup (correct, just larger); dedup is a
follow-up inside the splicer.

## Lowering (driver.rs)

`DominantWithSlowPath` lowering generalizes from one guarded arm to a
chain: `GuardKlass k1 → body1; GuardKlass k2 → body2; … ; CallSend`
(fallback), all arms rejoining at the site's merge block with the
existing per-arm deopt-scope chaining (`SenderLink` machinery unchanged
— each arm is exactly a DominantWithSlowPath fast path). The klass-id
RANGE collapse (Dart's cid_start/cid_end) is NOT in scope until world
genesis orders sibling klass ids adjacently — noted as a later lever.

## Gates

- Unit: decide() picks/refuses arms per count distributions; hash
  stability under count drift.
- Integration: richards `processWork:`/`addInput:checkPriority:` compile
  with 4 arms; `MACVM_TRACE=deopt` storm counter ~0 at t=200; dispatch
  bench unchanged (its 2-case site must still fold to the same code).
- World differential off vs t=200 + all three stress modes,
  byte-identical; benches correct under stress.
- Perf gate (tracking): richards warm at t=20; expect the residual
  send-fallback share of the 27ms to compress; record honestly either
  way. fib is NOT expected to move (self-send recursion is task #3's
  recursion-depth knob, not this).
