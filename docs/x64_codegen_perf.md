# Why x86-64 compiled code lags AArch64 — a codegen review, with fixes

Status: investigated 2026-07-22 (Windows x86-64, `f9451ab`). Evidence-backed.
UPDATE (same day): F1, F2-lite (store-imm + single-def remat; deopt slots kept
canonical), F3a, F3b (loop-weighted crossing residents + post-call reloads,
in lieu of interval splitting), F4, and F6 are BUILT and verified (5860-test
differential suite byte-identical under interp/JIT/GC-stress). Sieve `run`:
332→284 movs pre-residency. Wall clock moved modestly (arith 36→34, dict
14→13; fib is machine-noise-bound 177-218); Richards is FLAT — its residual
is per-activation fixed cost + spill-all around sends, i.e. F7/F3c, which
remain open (F3c is the structural project). F5 (within-block reload cache)
also remains open. The bench-harness Packet/Strength collision is fixed
(world/bench classes renamed Bm*; the collision source is world/41a, not 61c). This is a review of the **quality of the code the x64 JIT
emits**, not the GC/allocation gap — that is a separate, arch-independent issue
already root-caused in [`gc_alloc_gap.md`](gc_alloc_gap.md) and is explicitly
noted there to affect the Mac too.

## The question

Same overall design, same IR, same middle end. On AArch64 the JIT meets or
exceeds Cog on the macro benchmarks; on x86-64 it lags — Richards ~2.5× behind
Cog, tight-loop residuals ~4× behind. Why?

## The one-sentence answer

**x64 compiled code is dominated by spill/reload memory traffic that AArch64
avoids.** In the sieve `run` method the emitted code is **332 `mov` out of 516
instructions (64 %), and 193 of those movs touch a spill slot.** AArch64's
16-register file plus a residency tier that actually captures the hot values
keeps those same values in registers; x64's 7–10-register file, an unconditional
*spill-all-at-safepoints* policy, and a residency tier that is measurably "not a
win" push almost everything through the stack. On top of that structural cost sit
several concrete, cheap-to-fix codegen misses (identity moves, spilled constants,
redundant register hops, unnecessary write barriers) that AArch64 never paid
because on AArch64 they were free.

The gap is **codegen quality, not the middle end**. Every finding below is either
a peephole the emitter skips or an allocation-policy choice, none of it in the
shared IR/inliner/type-feedback layers.

---

## How this was measured

- Benchmarks run on this machine (i7-12700) with the release build:
  `MACVM_JIT=off` vs `MACVM_JIT=threshold=20 ./target/release/macvm run
  world/bench/<b>.mst --world world`.
  - **sieve**: interp 59 ms → JIT **4 ms** (14.75× speedup) — but still ~4×
    Cog's <1 ms. Residual is loop code quality, per PERF.md.
  - Richards/DeltaBlue could not be re-timed here: the current world defines
    `Packet`/`Strength` (in `world/61c_sockets.mst`), which collide with the
    bench classes — a harness/world regression worth a separate fix. Their
    documented figures stand: Richards x64 81 ms vs Cog 33 ms (2.5×); AArch64
    Richards ~6 ms (34× over its own interpreter).
- Generated code inspected with `MACVM_DBG_IR=<selector>` (IR + emitted x64
  listing, debug build; the emitted bytes are identical to release) and
  `MACVM_DBG_RESIDENTS=1` (residency assignment trace).
- Cross-checked against the AArch64 emitter (`src/compiler/emit.rs`) and the
  shared allocator (`src/compiler/regalloc.rs`).

### The primary exhibit: the sieve innermost loop

`[k <= size] whileTrue: [ flags at: k put: false. k := k + prime ]`, as x64
emits it (cleaned; `[rbp-N]` are spill slots, `[r15+N]` is the VM register
block, `[litN]` a literal-pool load):

```
  ; --- flags at: k put: false ---
  mov   r10, [lit1]        ; false
  mov   [rbp-216], r10     ;   ← SPILL a compile-time constant to a slot
  mov   r12, r10
  mov   r10, [rbp-16]      ;   ← reload `flags`
  mov   r11, [rbp-88]      ;   ← reload `k`
  mov   rax, r10
  and   rax, 3             ; ) klass guard (smi check + header compare)
  cmp   rax, 1             ; )
  jne   L23
  mov   rax, [r10+7]
  cmp   rax, [lit13]
  jne   L23
  test  r11, 3
  jne   L23
  lea   rax, [r11-4]       ; ) bounds check (one unsigned compare)
  cmp   rax, [r10+15]      ; )
  jae   L23
  mov   [r10+r11*2+15], r12 ; THE STORE
  mov   rax, [r15+16]      ; ) write barrier — 11 instructions to store a
  cmp   r10, rax           ; )   value (`false`) that can NEVER need one
  jb    L53
  test  r12, 3
  je    L53
  cmp   r12, rax
  jae   L53
  lea   rax, [r10+r11*2+15]
  shr   rax, 9
  add   rax, [r15+24]
  mov   [rax], 0
  ; --- k := k + prime ---
  mov   rcx, r12           ;   ← dead move (rcx never read)
  mov   r10, 4             ; the increment constant (tagged 1)
  mov   [rbp-224], r10     ;   ← SPILL a compile-time constant to a slot
  mov   r12, r10
  mov   r10, [rbp-88]      ;   ← reload `k` AGAIN (same slot, same block)
  test  r10, 3             ; ) smi guards for the add (the const operand is
  jne   L25                ; )   provably a smi — its guard is dead)
  test  r12, 3
  jne   L25
  mov   rcx, r10           ; two-address fixup
  add   rcx, r12
  jo    L25
  jmp   L24
L24:
  mov   r11, rcx
  mov   [rbp-88], r11      ;   ← store `k` back
  ; --- back-edge Poll ---
  mov   rax, [r15+32]
  test  rax, rax
  je    L54                ; flag clear (common) — no call
  mov   r10, [lit15]
  call  r10               ; poll stub (rare)
L54:
  mov   r10, [rbp-128]     ; ) Move v15,v15 — an IDENTITY move, emitted as
  mov   r11, r10           ; )   load+move+store: 3 instructions doing nothing
  mov   [rbp-128], r11     ; )
  mov   r10, [rbp-136]     ; ) Move v16,v16 — same, another 3 no-op instructions
  mov   r11, r10           ; )
  mov   [rbp-136], r11     ; )
  jmp   L3
```

Of ~46 instructions in this loop body, **~14 are pure waste**: 6 for two
identity moves, 4 to spill two loop-invariant constants, 1 dead `mov rcx`, 1
redundant reload of `k`, plus a full write barrier that the store can never
trigger. AArch64 pays none of these — the identity moves resolve to the same
register and vanish, the constants stay in registers, and the barrier still
runs but is a smaller fraction of a much tighter loop.

---

## Findings, ranked by (impact ÷ effort)

### F1 — Identity moves (`Move vN, vN`) are not elided  · **high impact, trivial**

`src/compiler/emit_x64.rs`, `Ir::Move`:

```rust
Ir::Move { dst, src } => {
    let s = e.read_into(*src, SCRATCH0);   // dst==src spilled ⇒ mov S0,[slot]
    let d = e.def_reg(*dst, SCRATCH1);
    if d != s { e.asm.emit("mov", &[r64(d), r64(s)]); }  // mov S1,S0
    e.store_def(*dst, d);                  // mov [slot],S1
}
```

There is no `dst == src` guard. When the value is spilled (the common case for
anything loop-carried), `Move vN, vN` becomes `mov scratch,[slot]; mov
scratch2,scratch; mov [slot],scratch2` — a 3-instruction round trip that leaves
the slot exactly as it was. The sieve `run` method has **4 such identity moves,
all in hot-loop back-edge blocks** (confirmed: `Move v15,v15`, `v16,v16`,
`v19,v19`, `v20,v20`), so this fires every iteration of the two innermost loops.

Why AArch64 never noticed: there `Move v15,v15` resolves `src` and `dst` to the
**same register** (v15 got one of the 16 GPRs), the emitter's `if dr.num !=
rs.num` guard sees they match, and it emits nothing. The cost only appears when
the value is spilled — i.e. only on x64.

**Fix:** at the top of `Ir::Move`, `if dst == src { return; }`. Also short-circuit
when `src` and `dst` resolve to the same physical home (same slot or same
register). Safe unconditionally — a self-move is a no-op wherever the value
lives. ~3 lines.

These identity moves are phi-copies for loop-carried values that are *unchanged*
on the back-edge. A cleaner variant of the fix is to drop them during
phi-resolution (IR level) so no backend pays them; the emitter guard is the
belt-and-braces version and should exist regardless.

### F2 — Constants are spilled to memory instead of rematerialised  · **high, medium**

`ConstSmi`/`ConstPool` define a vreg like any other, so when the allocator
spills that vreg the constant gets a frame slot. In a loop the emitted pattern
per iteration is: **materialise the constant into a register, store it to the
slot, then reload it for use** — three memory-touching instructions to stand in
for a value that costs one `mov r, imm` to recreate. Visible above: `false`
(`mov [rbp-216]`) and the increment constant `4` (`mov [rbp-224]`) are both
spilled inside the innermost loop.

Spilling a constant is *strictly* worse than rematerialising it: a spill is a
store + a reload (2 memory ops) versus one immediate move, and the store is pure
loss. Real allocators special-case this ("rematerialisation").

**Fix, cheapest first:**
1. In `allocate`, never hand a spill slot to an interval whose def is a
   `ConstSmi`/`ConstPool`/`FConst`. Mark such intervals *rematerialisable*; at
   each use, re-emit the constant into a scratch instead of reloading a slot,
   and skip the store at the def. Removes both the store and the reload.
2. Even before that: hoist loop-invariant `Const*` out of loop bodies (LICM at
   the IR level). Bigger, and F2.1 subsumes most of the benefit.

### F3 — The residency tier captures the wrong intervals; it is "not a win" for a reason  · **high, medium-large**

Residency (`resident_reg`) is meant to be x64's answer to its small register
file: give a *spilled* interval a callee-saved register so reads hit the
register instead of the slot. `MACVM_DBG_RESIDENTS=1` on sieve shows what it
actually does:

```
v30 len=4 -> r12      v53 len=1 -> r12      v73 len=1 -> r12
v31 len=3 -> r13      v57 len=1 -> r13      v77 len=1 -> r12
v56 len=3 -> r12      v66 len=1 -> r12      v80 len=1 -> r12
v41 len=1 -> r12      v44 len=1 -> r12
```

**All 11 residents have `len ≤ 4`; eight are `len = 1`.** A `len=1` resident is
useless — the value is read exactly once, immediately after its def, so it was
already in a register; residency just adds the write-through `mov` at the def and
saves nothing. Meanwhile the values that dominate the loop — `k`, `count`,
`size`, `prime` — appear **nowhere** in the list. They are excluded because the
eligibility filter is `!crosses_call`, and their live ranges span the method's
`Array new:` send, so a single send anywhere in a value's lifetime bars it from
residency for its *whole* lifetime.

Two design faults compound:

1. **Write-through defeats the point.** `store_def` writes a resident value to
   **both** the slot and the register on every def (the canonical slot must stay
   authoritative because oop maps scan slots, never registers). So residency
   *adds* a `mov` per def to *save* a `mov` per read — a net win only when reads
   greatly outnumber defs. For `len=1` intervals it is a strict loss; for the
   loop-carried values it can't apply at all.

2. **`crosses_call` is whole-lifetime and coarse.** A value used across one send,
   but hot in a loop that contains no send, is disqualified everywhere. The Poll
   is *not* a call-position (only `CallSend`/`CallRuntime` are — confirmed in
   `regalloc.rs`), so a genuinely send-free loop's induction variable *should*
   qualify — but any reuse of that vreg outside the loop that does cross a send
   taints the single interval.

Contrast AArch64: residency there draws from **seven dedicated, disjoint
callee-saved registers (x21–x27)**; on x64 the pool is **three** (R12–R14) drawn
from scan leftovers, and — crucially — the wrong intervals win them.

**Fixes, in increasing order of payoff and effort:**
- **F3a (cheap):** never assign a resident to a `len==1` (or `len < 2`) interval
  — it can only lose. One line in the residency `order` filter. Removes the
  write-through overhead the trace shows.
- **F3b (medium):** split intervals at safepoint/send boundaries so the
  send-free hot portion of a loop-carried value becomes its own `!crosses_call`
  sub-interval that residency *can* claim. This is what lets `k`/`count` live in
  R12–R14 across the loop body, reloaded once after each Poll — the shape
  AArch64 gets for free from its wider file. This is the single change most
  likely to move Richards.
- **F3c (largest, highest ceiling):** relax *spill-all-at-safepoints* so oop
  values may live in callee-saved registers across a safepoint, with the oop map
  covering those registers (the standard JIT contract; `oopmap.rs` currently
  scans slots only — MIGRATION.md §3j notes this was a deliberate simplification
  that "removed work," and it is now the work to add back). This removes the
  spill/reload around every send wholesale and is where the send-heavy gap
  ultimately lives. It touches the GC/deopt oop-map contract, so it is a project,
  not a peephole — but it is the true ceiling.

### F4 — Spill→spill moves use three instructions where two suffice  · **medium, trivial**

`read_into` loads the source into `SCRATCH0`, `def_reg` returns `SCRATCH1`, then
a `mov SCRATCH1,SCRATCH0`, then `store_def` writes `SCRATCH1` to the slot. A
spill-to-spill copy is thus `mov S0,[src]; mov S1,S0; mov [dst],S1` — the middle
hop is dead. It should be `mov S0,[src]; mov [dst],S0`. Same reload-scratch for
read and store when the def is spilled. Pervasive (every `Move`/`Param`/… whose
dst is spilled). ~5 lines in the `Move` path (and the same shape reused by
`Param`, `LoadField`, etc.).

### F5 — Values are reloaded on every use within a block  · **medium, medium**

`read_into` unconditionally reloads a spilled vreg from its slot. In the loop
above, `k` (`[rbp-88]`) is reloaded twice in one iteration (once for the store
index, once for the increment) though nothing wrote it in between. A minimal
*within-block reload cache* — remember "slot S is currently live in scratch R"
and reuse R until R is clobbered or the block ends — removes the repeat loads
with no change to the allocator or oop maps. This is a local, conservative
optimisation (invalidate the cache at every def, call, and block boundary).

### F6 — The write barrier is emitted for stores that can never need one  · **medium, small**

`flags at: k put: false` emits the full 11-instruction generational barrier,
though `false` is a compile-time-constant permanent oop that can never create an
old→young reference — the barrier's three runtime early-outs will always take
the skip, so all eleven instructions are dead. The emitter already has the
stored value's IR; when it is a **smi** or a **constant non-young oop**
(`nil`/`true`/`false`/an interned literal-pool oop that lives in perm/old space)
the barrier can be omitted entirely at compile time. `StoreField` and
`ArrayAtPut` both call `emit_write_barrier` unconditionally; gate that call on
"the stored value's IR is not a provably-no-barrier constant." Common in
initialisation loops (`at:put:` with a literal) and field stores of `nil`.

### F7 — Per-method fixed cost: the nil-fill prologue and frame size  · **context, measure first**

The sieve frame is **352 bytes (44 slots)** and the prologue nil-fills every
deopt-referenced slot (`deopt_nil_init_slots`) — dozens of `mov [rbp-N], r10` at
entry. For a large loop method that is amortised; for **send-heavy code with many
small, frequently-entered activations (Richards)** it is per-call overhead that
scales with the send count. Two levers: (a) shrink the deopt-nil set (many slots
are params/temps overwritten immediately by their entry-block def — those need
no nil), and (b) reduce the slot count itself, which F2/F3 do directly. Worth
measuring on Richards specifically once F1–F4 land, because Richards is where the
fixed per-activation cost is paid most often.

---

## Recommended sequencing

1. **F1, F4, F6, F3a** — a day's work between them, all local to the emitter or a
   one-line allocator filter, all strictly safe, each verified by the existing
   execution tests plus a before/after instruction-count on the sieve/arith
   listings. These remove the visible waste and should shave the tight-loop
   residual measurably.
2. **F2.1 (constant rematerialisation)** — the highest-value medium change;
   removes constant spill traffic from every loop.
3. **F5 (within-block reload cache)** — removes repeat reloads; composes with F2.
4. **F3b (interval splitting for residency)** — the change most likely to close
   the **Richards** send-heavy gap by keeping loop-carried values in R12–R14
   across send-free loop bodies.
5. **F3c (register-resident oops across safepoints)** — the structural ceiling;
   a scoped project touching the oop-map contract, to be taken only after 1–4
   quantify how much gap remains.

## How to verify each fix

- **Correctness:** the whole differential suite (interpreter-vs-JIT oracle) plus
  `cargo test --lib`; every op already has execution tests.
- **Impact:** `MACVM_DBG_IR=<selector>` before/after and diff the emitted
  listing (instruction count, `mov` count, memory-touching `mov` count — the
  64 % figure above is the headline metric to drive down); then the bench wall
  times. Fix the `world/61c_sockets.mst` `Packet`/`Strength` name collision first
  so Richards/DeltaBlue can be re-timed on this machine — without them the
  send-heavy case (the biggest gap) can't be measured locally.

## What is explicitly *not* the problem

- **The middle end** (IR, inlining, type feedback, escape analysis) — arch-neutral
  and shared; both targets run it identically.
- **The allocation-send path** (`gc_alloc_gap.md`) — real, but arch-independent
  and already documented as helping the Mac too. Out of scope here.
- **The send *site* sequence itself** — `CallSend` marshals arguments with a
  proper parallel move, uses a direct 5-byte `call rel32` (the near-host
  placement win is in place), and does a 2-instruction NLR check. The send is
  tight; its cost is the spill-all traffic *around* it (F3), not the call.

---

## F3c — the register-residence project, fully scoped (2026-07-22, reconnaissance)

Deep reconnaissance (GC frame scan, oop-map format, deopt materializer, call
stubs) turned the vague "structural ceiling" into a concrete, staged, de-risked
plan. The headline findings and the plan:

### What the code actually guarantees today
- **GC scans stack slots only** (`memory/roots.rs::each_code_root` reads
  `[fp − 8·(slot+1)]` per the per-safepoint `OopMap` bitmap; `oopmap.rs`). No
  register concept anywhere in the map.
- **Deopt reads stack slots only** (`scopes.rs::ValueLoc` has `FrameSlot`/
  `DoubleSlot`/`ConstPool`/`ConstSmi`/`Nil`/`ElidedClosure` — no `Register`;
  `resolve_frame_loc` only matches `Assignment::Spill`). The trapped register
  file is captured on the PROBE/crash path (`capture_regs_win`) but NOT on the
  `0xDE00/0xDE01` deopt path.
- **nmethod prologues save NO callee-saved registers** (`push rbp; mov rbp,rsp;
  sub rsp,N` — nothing else). A callee freely clobbers RBX/RSI/RDI/R12–R14. The
  only save of the Win64 callee bank is once at the interpreter→compiled
  boundary in `call_stub` (`stubs_x64.rs` `SAVED`). This is *why* residency
  (F3b) reloads after every call: the register does not survive a compiled call.
- **Multiple stubs rely on "spill-all ⇒ callee-saved regs are dead" and clobber
  them as scratch.** The deopt-return trampoline is explicit
  (`stubs_x64.rs:556`: "Safe to clobber RBX/RSI here: nothing live is in them,
  by the spill-all-at-safepoints invariant"). PIC/mega/ffi stubs use RBX/RSI but
  push/pop them (already compliant).

### The split that de-risks everything
There are TWO F3c's, and they have very different risk:

**F3c-oops (the RegisterMap project — DEFER).** To let an *oop* live in a
callee-saved register across a GC safepoint, the collector must find and
relocate it while the mutator is suspended deep in a call. Its physical home is
whatever inner frame last saved that register — so the frame walker needs a
HotSpot-style RegisterMap that tracks callee-saved save locations as it unwinds.
Plus register bits in the `OopMap`, plus `ValueLoc::Register` resolved through
the same RegisterMap at deopt. This touches the crown-jewel GC/deopt contract;
its failure mode is heap corruption. Large, highest-risk. Not worth it for the
current gap.

**F3c-nonoops (the safe, high-value subset — the recommended next step).**
A *non-oop* value (smi, raw) is NEVER in an oop map and is NEVER relocated by
GC. So keeping it in a callee-saved register across a call needs **no GC change
and no deopt-data change**: the canonical slot stays authoritative (write-through
at def, exactly as residency today), deopt reads the slot unchanged, GC ignores
it. The ONLY thing gained over F3b residency: **drop the post-call reload** for
non-oop residents — because the value can't have moved and (once prologues are
Win64-compliant) the register survives the call. This directly attacks fib
(smi-dominated: `n` reloaded 3×/activation across its send) and part of richards.

### F3c-nonoops implementation stages (each ends green + 4-way stress)
1. **Win64-compliant prologues.** Every nmethod saves/restores the callee-saved
   GPRs it assigns, in a save bank BELOW the spill slots so `spill_offset =
   −8·(slot+1)` and every GC/deopt slot read stays byte-identical. Frame layout:
   `[rbp]`, slots `[rbp−8..]`, save bank, outgoing area at `[rsp]`; all summands
   16-rounded.
2. **Two-pool regalloc.** Main scan PREFERS volatile regs (RCX RDX R8 R9) so
   leaf/small methods clobber no callee-saved reg and pay zero save cost;
   callee-saved regs (RBX RSI RDI R12–R14) are drawn only under pressure or for
   cross-call residents. Save-set = callee-saved regs actually assigned.
3. **Convert every stub that clobbers a callee-saved register** to preserve it
   (push/pop or use volatile scratch). Known site: the deopt-return trampoline
   (`stubs_x64.rs:556`, currently `mov rbx,rax; mov rsi,rbp`). Audit: PIC/mega/
   ffi already push/pop; `call_stub` saves the whole bank; Rust callees preserve
   per Win64. This is the exhaustive-audit gate — a miss corrupts a caller's
   resident.
4. **Skip the post-call reload for non-oop residents** (`emit_resident_reloads`
   gains `is_oop` filtering; keep the reload for oop residents, which still
   spill and may be GC-relocated). Loosen the F3b admit gate for non-oops now
   that they cost only a write-through, no reload.
5. **Verify:** fib/richards pinned vs Cog; lib + it_world; the 5860-test
   differential byte-identical under interp / JIT / GC-stress / DEOPT-stress
   (the last exercises the converted trampoline).

This is a calling-convention change, not a peephole — correct-by-construction
for GC/deopt (slots stay canonical) but with a corruption failure mode in the
stub conversion, so it wants a focused session with full stress-validation
budget, not a rushed tail-end implementation.

## 2026-07-22 — two 9cb272e findings from the arm64 (MACVM) port

MACVM ported the special selectors the same day (its `20b37b0`, writing the
A64 sequences this repo cfg-gated off) and its verification pass surfaced two
defects that are present in THIS tree too. Confirmed against this checkout,
not just the Mac's.

### 1. `Ir::BoolNot` has NEVER fired — the canonical-flip check can't pass

`bool_not_speculatable`'s canonical check (ir.rs:1793) ends with

    matches!(i1, Instr::ReturnTos) && n1 >= len

but the frontend **always appends a dead trailing `ret_self`** after a method
body (frontend/codegen.rs ~1046: "`ret_self` is always appended; … this
trailing `ret_self` is simply unreachable dead code"). So the live
`True>>not` / `False>>not` (`^false` / `^true`, world/02_nil_boolean.mst)
decode as `PushFalse; ReturnTos; ReturnSelf` with `len = 3, n1 = 2` —
`n1 >= len` is false for EVERY method this frontend compiles, `canonical`
returns false, and every `not` site silently stays a generic `CallSend`.
9cb272e's richards 85 → 41-44 was therefore RefCmpVal + F7 alone; the
~90k-activations/run `not` half of the census was never captured, and its
"the first cut silently DECLINED every method containing them" lesson has a
quieter sibling: this decline is invisible because generic sends are
correct, just slow.

Fix (as landed on the Mac): drop the `n1 >= len` conjunct — anything after
an unconditional `ReturnTos` is unreachable, so the leading
`PushTrue/PushFalse; ReturnTos` pair fully determines the method's
behavior:

    let (i1, _n1) = decode_at(m, n0);
    matches!(i1, Instr::ReturnTos)

Verify it actually fires this time (the Mac's probe recipe): debug build,
`MACVM_DBG_IR=<sel>` on a method whose `not` site has boolean evidence —
expect `BoolNot { … }` in the dump, not `CallSend`; then send `not` to a
non-boolean through the warmed compiled site — expect exactly ONE deopt
line and the correct DNU-free result. On the Mac this half was worth
richards 21.0 → 20.1 ms on top of RefCmpVal; given this port's higher
per-activation costs it plausibly closes a similar or larger slice of the
remaining ~1.15x richards gap vs Cog.

### 2. `successors()` omits `BoolNot`'s trap edge — latent, benign today

regalloc.rs's `successors()` fail-edge group (SmiArith / SmiCmpVal /
ArrayAt / ArrayAtPut / FUnbox / VecArith / GuardKlass) never learned
`Ir::BoolNot { fail, .. }`. Nothing breaks TODAY only because
`reverse_postorder` seeds a DFS from every unvisited block, so the
CFG-unreachable trap block still gets laid out and its label binds — it
just lands at the tail of the block order instead of near its guard. But
the function is the compiler's single source of CFG truth (block layout,
instruction numbering, and — once fixed — any future dominance/liveness
consumer), and an edge it lies about is a trap for the next analysis that
trusts it. One-line fix: add `| Ir::BoolNot { fail, .. }` to the
`succs.push(*fail)` group.
