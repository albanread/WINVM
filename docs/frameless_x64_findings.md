# Frameless leaves on x64 — implemented, measured, parked (2026-07-24)

The MACVM team's frameless-leaf work (F0–F3, `frameless_leaf_methods.md`) was
ported to x64 and measured. **Verdict: parked, not landed.** Two findings, one
about frameless itself and one — more important — that it surfaced.

The work lives on branch **`frameless-x64-wip`** (`b4ce302`): the full x64
emission (prologue/epilogue elision, blob-finish metadata asserts, the x64
stack-arg `Param` eligibility gate), plus the three upstream cherry-picks it
sits on. It builds, its unit tests pass, and it executes correctly.

## Finding 1 — the x64 frameless win is ~3 %, at the noise floor

A/B, pinned P-core, same session, `MACVM_FRAMELESS` on vs off:

| bench | frameless ON | OFF |
|---|---|---|
| richards (warm, ×10) | 27–28 ms | 27–29 ms |
| fib (10× fib 30) | 76–77 ms | 78–80 ms |

The direction is consistent (ON ≤ OFF), so there is a *small real* win — but it
is inside measurement noise and nowhere near arm64's "~2–4 % richards."

**Why x64 gets so little, and it was predictable.** `frameless_leaf_methods.md`
§1 says it outright: AArch64 makes the frameless leaf *literally* zero-overhead
— `bl` leaves the return address in `x30`, so a leaf returns with a bare `ret`
and the return address never touches memory. On x64 the return address is
pushed to `[rsp]` by `call` no matter what, and the framed prologue it elides —
`push rbp; mov rbp,rsp; sub rsp,imm` — is nearly free on modern x64 (the stack
engine handles `push`/`pop` off the critical path). So eliding it saves the
handful of nil-fill stores and three cheap instructions, not arm64's genuine
memory-traffic + RSB win.

Consistent with fib's own shape: `fib:` makes recursive *sends*, so it is
framed regardless; only its leaf helpers qualify, which is why fib barely
moves. The frameless win is a richards-accessor-swarm effect, and on x64 even
that swarm is cheap to frame.

## Finding 2 (the important one) — F0's scan exposes a latent GC-stress crash

Landing F0 turned the world suite's `MACVM_GC_STRESS=1` run from green to a
hard crash:

```
MACVM PROBE: ACCESS_VIOLATION pc 0x… addr 0x8 FOREIGN (not in any code cache); dying
```

Bisected on the windows-port line: **3/3 pass at `165ccea` (pre-frameless),
reproducible crash once F0 lands.** Then isolated precisely:

- **Not the emission.** `MACVM_FRAMELESS=0` (scan runs, emission off) still
  crashes.
- **Not the `Nmethod` struct field.** Short-circuiting the eligibility scan
  (`return false` before the loop) while keeping the added `frameless_eligible:
  bool` field makes GC-stress pass 5860/0.
- **Not stack depth.** Raising the main-thread reserve to 8 MiB and then 64 MiB
  (`/STACK`) does not help — so it is not a tight-margin overflow.

That leaves one mechanism: **the eligibility scan is pure read-only** (it
iterates `IrBlock.code` and `RegallocResult.intervals`, allocates nothing,
dereferences no oop), so it cannot corrupt memory directly. It can only
**perturb scavenge timing** — the extra instructions shift *where* in the
compile path a GC-stress scavenge fires — and that exposes a **pre-existing
latent GC-root gap in the compile path**: an oop live across one of the
compile's own allocations that is not registered as a root, so a scavenge in
that window moves it and leaves a stale pointer that a later deref faults on.
The prime suspect is the literal-pool oops held across the `Nmethod`/code-cache
allocations during `compile_method_full`.

This is a **real correctness bug, not a test-mode artifact** — GC-stress exists
precisely to catch missing roots by making every allocation a collection point.
A latent one that any timing change can trigger is a production hazard too. It
was masked by luck before; the scan unmasked it.

## Recommendation

1. **Do not land frameless-x64 as-is** — a ~3 % noise-floor win is not worth
   regressing a green GC gate, and the arm64 rationale (zero-overhead leaf)
   does not transfer to x64.
2. **Root-cause finding 2 with a debugger.** No `cdb`/`windbg`/`llvm-symbolizer`
   was available this session, which is why the faulting `pc` could not be
   symbolized to a function. With a debugger: run the world suite under
   `MACVM_GC_STRESS=1` on the `frameless-x64-wip` branch, break on the first
   access violation, and walk back to the compile-path frame that holds the
   unrooted oop. Fixing it hardens the VM independent of frameless.
3. **The real fib/richards lever is F3c** (register-resident oops across
   safepoints, `x64_codegen_perf.md`) — the structural project that removes the
   spill/reload traffic around sends, which is where the deep-call residual
   actually lives. Frameless was the cheap arm64 win; on x64 it is not the
   lever.

## If frameless is revived later

The `frameless-x64-wip` branch is complete and correct — reviving it is:
(a) fix finding 2 first, (b) gate the F0 census scan behind
`frameless_emission_on()` so a default build that leaves emission off does not
even run the scan, and (c) re-measure to confirm the ~3 % is worth the
complexity. The x64-specific piece worth keeping regardless is the eligibility
gate for stack-argument `Param`s (index ≥ `MAX_REG_ARGS`): those read the
caller's outgoing area RBP-relative, which only exists relative to our own
`push rbp`, so such a method must stay framed — an x64 constraint arm64 (all
args in registers) never has.
