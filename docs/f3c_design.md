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

## Recon deltas (2026-07-24, second pass — the diff-level facts)

- **The deopt side is one arm, not a subsystem.** Deopt metadata encodes
  per-vreg `ValueLoc::FrameSlot(byte_off)` (driver's
  `build_deopt_metadata`/`resolve_frame_loc`; golden test at
  driver.rs:~2578 shows the exact `-8*(slot+1)` mapping). A
  register-resident vreg at a `LoopPoll` site resolves to
  `FrameSlot(-8*(frame_slots + reg_index + 1))` — same encoding, so the
  MATERIALIZER IS UNCHANGED, exactly as designed.
- **`verify_spill_all` (regalloc.rs:1123) anticipated this change** — a
  RELEASE-mode assert whose doc names silent heap corruption as the
  failure mode. S1 evolves it: register across a CALL-SHAPED safepoint
  still panics; register across a poll panics UNLESS that poll's save
  record covers the (vreg, poll) pair.
- **The organic-span pin rule (subtle, load-bearing).** `LoopPoll`
  deopt-referenced vregs pin today via MEMBERSHIP (`deopt_referenced`,
  regalloc.rs:647 — intervals stay organic per task #94). A poll-deopt
  vreg whose organic interval does NOT span the poll (interpreter-
  visible, dead in compiled code) must KEEP membership pinning — its
  register could be legally reused before the poll, so there is nothing
  to save. The exemption applies ONLY to intervals organically spanning
  the poll (`start <= p && end > p` from real uses) — which is precisely
  the loop-carried hot set (arith's `s`/`i`), so the win is untouched.
- **`extra_oop_live` entries may now name register-assigned vregs**
  (recorded conservatively at every earlier safepoint, task #94);
  `build_for_position` must skip non-`Spill` assignments — their poll
  bits come from the save record instead.
- **Pin computation site**: regalloc.rs:669-673 (`crosses_safepoint` =
  membership ∪ spans-any-safepoint). S1: spans-any-PINNING-position ∪
  trap-membership ∪ non-spanning-poll-membership; `is_fp` and
  `method.is_osr` keep the old full pin.
- **Frame size**: emit_x64.rs:1026 (`raw = 8 * frame_slots`) grows by
  the save area; the nil-fill loop at :1173 does NOT cover save slots
  (their bits are set only at poll positions the stores dominate).
- **OopMap** is a plain bitmap (`nmethod.rs:46`); `build_for_position(
  intervals, frame_slots, position, extra_oop_live)` (oopmap.rs:52,
  called from driver.rs:1354) gains the per-poll save records and the
  widened slot universe.

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

## S1 landing findings (2026-07-24, post-implementation)

S1 landed fully gated — 744 lib tests (two new tripwires: register-kept-
with-covering-save-record + OSR negative; the organic-span rule), world
differential byte-identical plain + GC_STRESS=1 + full:64 + **DEOPT_
STRESS=64** (the flagship: forced poll-deopts materializing from save
slots across 5860 tests) — and the benches are FLAT, with the cause
diagnosed by the new `MACVM_TRACE=pollsave` channel on arith:

```
[pollsave] osr=true polls=1 pinning_positions=3 pin_exact=27 poll_deopt=7 widen=0
[pollsave] poll@23: saved_regs=0 spanning_spilled=5
```

Two blockers, both structural and both now named:

1. **Hot loop methods only ever compile as OSR.** Trigger unification
   (S24 L2) saturates the invocation counter on by_key install, so every
   later CALL enters the OSR-earned nmethod — a second, non-OSR compile
   never happens. S1's `!is_osr` gate therefore excludes exactly the only
   compile that exists for the loops it targets. The fix is S4 (OSR
   envelope): the OSR entry already has the resident-reload pattern
   (`emit_resident_reloads_at(header)`) — register-exempt vregs need the
   same "seed slot, then load register at entry" treatment.
2. **Smi-overflow fail edges pin the loop slots anyway** (`pin_exact=27`
   on a 5-instruction loop): every `SmiArith` trap site records
   receiver+slots+stack, and trap references are membership-pinned
   regardless of organic span. S1b: extend the save-record mechanism to
   UncommonTrap sites — the trap's cold block saves spanning registers to
   the same save slots before the `brk`, and the trap materializer reads
   them exactly as the LoopPoll one now does. Same soundness argument,
   same slots, one more record producer.

Sequencing update: S1b before S2 (alloc slow paths) — the trap pins
dominate every smi loop. S4 remains the unlock for the whole family.
