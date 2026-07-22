# The allocation gap vs Cog — root cause and ranked fixes

Status: investigated 2026-07-22, no fixes built. The matched-source Cog
comparison (Pharo 13.1/Cog, identical checksums; see the benchmark story in
`docs/deopt_fixes.md`'s sibling investigations) left ONE bench to Cog:

    alloc x5 (200k Association chain): Cog 7 ms, MACVM 46 ms  (6.6x)

The headline of the investigation: **it is mostly not the GC.** The gap
decomposes into three independent costs, separated by a 2x2x2 experiment
matrix — inline-fused vs send-path allocation, surviving vs dying objects,
4 MB vs 32 MB eden:

| variant (ms per 5x200k) | eden 4 MB | eden 32 MB |
|---|---|---|
| chain — send-alloc, survives (THE bench) | 44 | 32 |
| chainFused — inline-alloc, survives | 16 | **2** |
| dead — send-alloc, dies immediately | 31 | 31 |
| deadFused — inline-alloc, dies | 3 | 2 |

**Bound: chainFused @ 32 MB eden = 2 ms vs Cog's 7 — with both pathologies
removed, MACVM is 3.5x FASTER than Cog on this bench.** The collector's
core is competitive; the losses are a compiler gap and a configuration gap
stacked together.

## Cost 1 (~65%): the allocation SEND path, not collection

`Association key: i value: last` allocates via `self basicNew` inside the
class-side constructor, which compiles as a generic
`CallSend -> shim -> rt_call_primitive -> prim_basic_new` — ~28 ns/object.
The `dead` row (31 ms) is invariant to eden size: zero GC involvement.
The fused literal `X basicNew` (the `Ir::Alloc` inline eden bump) costs
2-3 ns. `ir::alloc_site_klass` only fires for a literal class-constant
receiver, so every real constructor in the world — `key:value:`, every
`X new` -> `self basicNew` chain — misses the fusion.

**Fix:** extend the alloc fusion to customized `self basicNew`: under
customization the receiver klass IS the metaclass, whose sole instance
(the class) and fixed instance size are compile-time derivable. Wide
payoff — accelerates essentially all object construction. Medium effort.

## Cost 2 (~30%): nursery geometry

`DEFAULT_EDEN_SIZE` = 4 MB; `SURVIVOR_SIZE` = 512 KB (layout.rs). The
bench's live chain is 4.8 MB — larger than eden — so every mid-build
scavenge overflows the survivor space and the designed cascade promotes
nearly everything: 65% of ALL allocated bytes were promoted; the promoted
chain then dies IN OLD SPACE, old fills with garbage, and full GCs fire
(7 per baseline run, 4.8 ms max pause). At `MACVM_EDEN=32768` (the env
var takes KiB) the whole cost class vanishes.

**Fix:** raise the default eden (16 MB ≈ +24 MB/VM covers this class),
and/or adaptive nursery growth on a promote-storm signal (promoted-bytes
ratio per scavenge is already in gc_stats). Trivial-to-small effort.

## Cost 3 (~2x residual): scavenge mechanics at matched geometry

~1.9 GB/s effective. The copy loop itself is sound (`copy_nonoverlapping`
bulk copy + forwarding install, scavenge.rs), but each PROMOTED object
pays: an individual `old.allocate` call, a per-object
`cards.record_multistores` range, two validated `MemOop::try_from`s, and
age-table bookkeeping. Batching (one old-space carve per scavenge,
coalesced card ranges) closes most of it. Fix AFTER 1+2 — it only shows
once they are gone.

## Recommended order

1. Eden default bump (one line + the standard battery).
2. Customized-`self basicNew` alloc fusion (the big general win).
3. Scavenge batching (secondary).

---

## Outcome (same day): both fixes landed, gap closed to a TIE

- `40fc343` — default eden 4 -> 16 MiB (cost 2).
- `ad2846f` — `alloc_site_klass_on`: spliced constructors' `self basicNew`
  fuses to the inline eden bump in both in-body splice walks (cost 1; the
  constructor pattern routes through the CFG splicer — its `| a |` temp
  makes the nonleaf splicer decline, which is where a first attempt
  silently missed; found by probing the check, not by re-reading it).

Measured end state (matched-source harness): **alloc x5 = 7 ms on BOTH
VMs — a dead tie**, from 46 ms (6.6x behind). The full comparison now
reads: MACVM wins or ties every bench. Cost 3 (scavenge batching) was
never needed for parity and remains unimplemented.
