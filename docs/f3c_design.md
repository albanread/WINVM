# F3c — registers live across safepoints via slow-path save (design)

Status: DESIGN, slice 1 specified to implementation depth. Reference:
`docs/dart124_compiler_lessons.md` item 1 (Dart 1.24.3
`flow_graph_compiler.cc:724-812` — `RecordSafepoint` appends live
registers to the frame's stack bitmap; `SlowPathCode` saves/restores them
around the runtime call; the GC only ever walks described stack memory).

## Motivation, measured

- arith 33-35 warm vs dart 1.24.3's 9.6 on identical semantics — the
  loop-carried accumulator and induction variable are slot-pinned by the
  back-edge `Poll` and re-stored every iteration (PERF.md 2026-07-22:
  "spill-all across the back-edge Poll" is the named loop-quality
  residual).
- richards 21 after the dart124 dispatch work; its remaining
  decomposition (PERF.md slice-3 entry) is F3c.
- Dart's 3-cycle arith iteration is register residency + this exact
  slow-path-save discipline.

## The invariant today (why spill-all)

- `LiveInterval::crosses_safepoint` (regalloc.rs:51) — an interval
  spanning ANY safepoint position is pinned to a canonical frame slot;
  oop maps scan slots, never registers (emit_x64.rs `resident` doc).
- `Poll`/`UncommonTrap` additionally FORCE their deopt-recorded vregs
  live-across (`deopt_live_exact`, regalloc.rs:280-312: "forced to
  `end > pos` so `crosses_safepoint` fires and spill-all pins them") —
  the materializer reads interpreter state from those slots.
- S14 "residency" mitigates READS (callee-saved cache, slot stays
  authoritative, never across a call) but every def still write-throughs.

## The design (Dart's model, adapted to slot-only walkers)

**Registers never become a GC or materializer concept.** Instead:

1. **Poll-save slots.** The frame grows a small dedicated save area —
   one slot per physical GP register the allocator can assign (≤14) —
   after the ordinary spill area (`spill_offset` numbering continues;
   `graph_entry spill_slot_count + save area` is the new frame size).
2. **The pin relaxes.** An interval whose safepoint crossings are ONLY
   `Poll`s keeps its register (slice 1: non-OSR compiles). Crossing a
   `CallSend`/`CallRuntime`/`Alloc`/`FBox`/`UncommonTrap` still pins to a
   slot exactly as today.
3. **Per-poll live-register record.** Regalloc emits, per `Poll`
   position: `[(phys_reg, save_slot, is_oop, vreg)]` for intervals live
   across that poll in registers.
4. **The slow path saves.** `Ir::Poll` emission (emit_x64.rs:1912 —
   already `test flag; jz skip; call stub_poll`) wraps the call:
   `mov [rbp - save_i], reg` for each live reg, `call stub_poll`,
   `mov reg, [rbp - save_i]` after. The flag-clear fast path is
   UNTOUCHED — zero cost per quiet iteration; that is the whole win.
5. **OopMap union.** The poll's map = today's slot bitmap ∪ the save
   slots of oop-holding saved registers (oopmap.rs's builder gains the
   per-poll record as an additional exact-position source, exactly the
   `extra_oop_live` precedent at regalloc.rs:647).
6. **Deopt reads save slots.** The driver's vreg→slot resolution for the
   `LoopPoll` deopt scope consults the per-poll record: a
   register-resident vreg resolves to its SAVE slot. The materializer is
   UNCHANGED — it reads frame slots, as always; the values were stored
   there by step 4 before `stub_poll` could deopt anything.

Soundness argument, in one paragraph: the save stores dominate the only
GC/deopt-capable instruction on the path (`call stub_poll`); the save
slots are ordinary described frame memory, dead (untraced) at every
other safepoint because only THIS poll's map sets their bits; the
restore dominates the rejoin, so the fast path observes register state
identical to today's slot-reloaded state; and any interval touching a
call-shaped safepoint is slot-pinned exactly as before, so nothing else
in the pipeline (resident reloads, c2i, PIC misses, trap materializers)
sees a new shape.

## Slices, each landing gated

- **S1 — Poll, non-OSR, GP only** (specified above). FP intervals stay
  pinned (not oops; deopt plumbing for FP save slots is S4 polish).
  OSR compiles keep today's rule (`is_osr` gate) — OSR entry seeds
  SLOTS (F6 comment's own invariant) and must not meet slot-less
  loop-carried vregs yet. Expected movement: arith (the accumulator/
  induction pair stops round-tripping), sieve residual, fib tails;
  richards partial (its loops contain sends → those values stay pinned).
  DEOPT_STRESS=64 is the sharp gate: it forces poll deopts, exercising
  save-slot materialization on every loop.
- **S2 — Alloc/FBox slow paths.** Same wrap at the inline-bump overflow
  edge; the fast bump path keeps registers. Frees allocation-heavy loops
  (deltablue constructors, alloc bench).
- **S3 — call-crossing residency.** Callee-saved registers + prologue
  save area described across the whole method; values live across
  `CallSend` ride callee-saved and the oop map covers their prologue
  slots. This is the richards lever (its hot loops contain real sends).
  Interacts with S14 residency (which becomes the degenerate cached-read
  form of the general mechanism) — expect to RETIRE resident_reloads.
- **S4 — OSR + FP.** OSR entry learns to seed registers (or the entry
  block reloads from seeded slots once, then goes register-resident);
  FP save slots for float loops.

## Touch points (slice 1)

| Where | What |
|---|---|
| regalloc.rs:51 `crosses_safepoint` | split into `crosses_call_shaped` (pins) vs `crosses_poll_only` (records for save) |
| regalloc.rs:280-312, 355 | `deopt_live_exact` at `LoopPoll` no longer forces `end > pos` slot-pinning for register-resident vregs — the save-slot record satisfies the materializer instead |
| regalloc.rs:647 `extra_oop_live` | the per-poll save record rides the same exact-position channel into oopmap |
| oopmap.rs builder | union in save-slot bits at poll positions |
| emit_x64.rs:1912 `Ir::Poll` | save/restore wrap around `emit_runtime_call(stub_poll)` |
| emit_x64.rs frame setup | reserve the save area; `spill_offset` numbering extended |
| driver.rs deopt resolution | `LoopPoll` scope vreg→location consults the per-poll record |
| A64 emit.rs | same wrap (mirrors; listing-level tests both sides) |

## Gates (every slice)

World differential off vs t=200 byte-identical, plain + GC_STRESS=1 +
GC_STRESS=full:64 + **DEOPT_STRESS=64** (the flagship for this work), on
the RELEASE binary. Lib + it_tier1 poly/dominant/OSR suites. Pinned
same-session bench pair with expected movement named in advance; record
honestly either way (winvm-gating-cadence: heavy gates once per landing).
A new tripwire test per slice: S1's is "loop-carried vreg live across a
poll deopts under DEOPT_STRESS and materializes the interpreter frame
from SAVE slots with the exact pre-deopt values" — the direct heir of
the OSR uninit-slot and materializer-ordering bugs this machinery's
history is made of.
