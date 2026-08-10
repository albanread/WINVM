# G0 — Re-entrant embed entries (V1/S1): implementation spec

**Branch:** `dolphin-mvp` (forked from `windows-port` at the GC fix `6c899f1`).
**Sprint:** the keystone of `dolphin_ui_sprints.md` — a pure VM-soundness change,
no GUI/Dolphin code, testable with Rust doubles. Everything downstream builds on
it, so it lands first and is proven to depth before any Dolphin class arrives.

**Scope decision (2026-07-24): Dolphin is Windows-only for now.** The entire
re-entrant nesting path serves the Win32 wndproc, so the whole G0 stack machinery
— the LIFO recovery stack, the top-of-stack fault lookup, the fail-closed removal,
the `&mut` re-entry token — is **`#[cfg(windows)]`**. macOS/Cocoa keeps its
existing single-slot, top-level-only model (`claim_jmp_slot` + single idle
baseline + the fail-closed `dispatch_callback` guard) **byte-identical to today**;
Cocoa's own "always top-level" doctrine means it never needs nesting. Consequence:
**no thread-local is ever read from a POSIX signal handler** — the signal-safety
hazard below is resolved by construction, not by careful POSIX handling.

This spec is grounded in a read of the current substrate; file:line are on this
branch's tree.

## The problem (why nesting is currently refused)

Win32 is synchronous and re-entrant: a message handler calls an FFI function
(`CreateWindowExW`, `SendMessageW`, `DestroyWindow`) which drives the wndproc,
which re-enters the image — to arbitrary depth. WINVM's embed entry
(`VmHandle::dispatch_callback`, [`src/embed.rs:764`](../../src/embed.rs)) is
sound only *top-level*: it holds **one** recovery slot and **one** idle baseline
per thread, and a thread-local `IN_CALLBACK: bool` that **fails closed** on
nesting ([`embed.rs:780`](../../src/embed.rs) returns the shape default).

Three single-valued things become LIFO stacks:

| Thing | Now | File |
|---|---|---|
| recovery `sigjmp_buf` slot | one per thread, reused each entry (`claim_jmp_slot`) | `deopt_trap.rs:414` |
| idle baseline (stack/arena watermark) | one field `self.idle_baseline` | `embed.rs:522`, struct field |
| nesting guard | `IN_CALLBACK: Cell<bool>`, fail-closed | `embed.rs:141,780` |

## Target model

A per-thread **stack of `(jmp-slot, baseline)` entry frames**; a fault
`siglongjmp`s to the **top** frame only (LIFO — nested entries return normally,
so recovery of the innermost is all that's ever needed). Depth is a counter.

### Change 1 — slot stack (`deopt_trap.rs`)

- Add two thread-locals: `ENTRY_SLOT_STACK: RefCell<Vec<usize>>` (the LIFO of
  claimed registry slots, touched only in ordinary code) and
  `TOP_ENTRY_SLOT: Cell<usize>` (the current top, `usize::MAX` = none).
- `push_entry_frame() -> usize`: claim a **fresh free** registry slot (owner==0;
  never *reuse* this thread's existing slot — that is what would clobber the
  outer frame's `sigjmp_buf`), push it, set `TOP_ENTRY_SLOT`, return it. The
  caller runs `sigsetjmp` inline on it (the `jmp_buf_ptr` rule at
  [`deopt_trap.rs:425`](../../src/codecache/deopt_trap.rs) — never through a
  wrapper that returns).
- `pop_entry_frame()`: release the top slot (`JMP_OWNER[slot]=0`), restore
  `TOP_ENTRY_SLOT` to the previous. `JMP_REGISTRY_CAP=64` is ample.
- `lookup_jmp_slot_for_current_thread()` ([`:523`](../../src/codecache/deopt_trap.rs))
  returns the **top**.

  ✅ **Signal-safety — resolved by the Windows-only scope.** The top-of-stack
  read happens only in the Windows VEH (`handle_win_fault`), where a thread-local
  `Cell::get` is safe (same thread, TLS intact), and it is **`#[cfg(windows)]`**.
  The POSIX `sig_fault_handler` keeps the existing single-slot `JMP_OWNER` search
  and never reads a thread-local (a TLS read there could call `__tls_get_addr`,
  which is not async-signal-safe). The hazard cannot arise.

### Change 2 — baseline stack + depth (`embed.rs`)

- `self.idle_baseline: IdleBaseline` → `self.idle_baselines: Vec<IdleBaseline>`.
  `snapshot_idle_baseline` ([`:522`](../../src/embed.rs)) **pushes**;
  `restore_after_guest_fatal` ([`:595`](../../src/embed.rs)) restores from the
  **top**. Popping is done by the entry that owns the frame (see wiring).
- `IN_CALLBACK: Cell<bool>` → `CALLBACK_DEPTH: Cell<u32>`; `callback_active()`
  keeps its meaning (`depth > 0`) for existing callers; inc on entry, dec on
  every exit arm (normal, guest-fatal, native-fault) — the arms already run
  before the `Drop`-skipping `siglongjmp` landing.
- **Wiring** — `eval` ([`:656`](../../src/embed.rs)), `exec` ([`:709`](../../src/embed.rs)),
  and `dispatch_callback` ([`:764`](../../src/embed.rs)) all switch from
  `claim_jmp_slot` + single `snapshot_idle_baseline` to `push_entry_frame` +
  baseline push, and pop both on exit. **Remove the fail-closed guard**
  (`embed.rs:780`). eval/exec must migrate too, because Change 1 makes the fault
  lookup read the stack top — a non-migrated caller would leave the top unset.

### Change 3 — the `&mut VmState` re-entry token (`runtime/ffi.rs`)

The outer entry holds `&mut VmState` across the FFI stub while the native call
runs; the nested entry (driven from the wndproc *inside* that native call) needs
it back. Mechanism (mirrors the Cocoa trampolines' thread-local `*mut VmHandle`):
before the FFI stub transfers to native code, publish the `VmState` pointer in a
thread-local **re-entry token**; the wndproc trampoline re-materializes `&mut`
from it. **Soundness contract** (documented at both sites, mirroring the FFI
layer's existing "no allocation before the call" rule): *the FFI stub touches no
VM state between publishing the token and the native call's return*, so the outer
borrow is quiescent for the whole nested window. Unsafe-but-disciplined, of the
kind the codebase already maintains (`src/codecache/ffi_stubs_x64.rs`,
`src/runtime/ffi.rs`).

`UI_VM_GENERATION` ([`embed.rs:108`](../../src/embed.rs)) checks stay outermost:
nested entries inherit the outer entry's generation (same VM, same thread).

## Out of scope (V1)

Re-entry from *other* threads (workers never call back); re-entry into a
*different* VM; JIT-frame unwind (S12 — GUI dispatch stays interpreter-tier,
which FFI methods already are).

## The G0 gate (5 tests)

1. `nested_entry_depth_5` — a Rust native-fn double re-enters `dispatch_callback`
   5 deep; each level allocates (GC-eligible) and returns a distinct value;
   assert all five land.
2. `dnu_at_depth_3_unwinds_one_level` — guest DNU at depth 3 → the depth-3 entry
   answers its default, depths 2/1 continue and return their own values; exactly
   one walkback line on the transcript.
3. `native_fault_at_depth_2` — recovered AV in nested marshalling → same
   containment.
4. `lifo_baseline_restore` — after any recovery arm, the *outer* entry's later
   fault still lands in the outer slot (regression for the returned-frame
   `siglongjmp` hazard at `embed.rs:151-156`).
5. Existing lib suite green; the Cocoa-path fail-closed unit test is **replaced**
   (its premise — nesting is refused — is now deliberately obsolete).

## Implementation sequence (each step ends green)

1. ✅ **Slot-stack kernel** (`deopt_trap.rs`): `push`/`pop`/top + the platform-safe
   lookup, proven in isolation by a nested-LIFO `sigsetjmp`/`siglongjmp` unit
   test in the module's own test suite (it already has real-fault recovery
   tests to model on). De-risks the riskiest kernel before any embed change.
2. ✅ **Baseline stack + depth + wiring** (`embed.rs`): migrate eval/exec/
   dispatch_callback; remove fail-closed. Gate: existing embed tests green +
   `nested_entry_depth_5` (no re-entry token yet — the test double calls the
   handle directly).
3. **Re-entry token** (`ffi.rs`): the `&mut` seam; gate with a test that nests
   *through* a real FFI stub double.
4. Tests 2–4 + replace the Cocoa fail-closed test; full suite green = G0 done.

### Step 2 as landed (2026-08-10)

Three decisions the spec left open, resolved by the implementation:

- **`push_entry_frame`/`pop_entry_frame` are platform-neutral, the machinery
  behind them is not.** On Windows they are the LIFO stack; on POSIX `push` *is*
  `claim_jmp_slot` and `pop` is a no-op (the single slot is released by
  `deregister_setjmp` at `VmHandle` Drop, exactly as before). So macOS keeps
  byte-identical behavior *and* the seven `sigsetjmp` entry points in `embed.rs`
  carry no `cfg` of their own. Only the fail-closed guard is written as a `cfg`
  at its site, because that is a genuine behavior fork.
- **All seven entry points migrated**, not just the three the spec named:
  `render_fragment`, `fire_widget_action`, `eval_to_string` and `eval_to_bytes`
  are entries too, and any of them can be invoked from the dev control channel
  *inside* a live handler. Each body now sits in a labeled block so every exit —
  including the early `Ok(None)`/parse-error arms and the `?` that used to sit in
  `eval_to_bytes` — funnels through one `pop_entry`; a `return` past it would
  strand the frame as the thread's LIFO recovery target.
- **The lookup's fallback is load-bearing.** `lookup_jmp_slot_for_current_thread`
  consults `entry_slot_top()` first and falls back to the owner scan, so
  single-slot `claim_jmp_slot` users (this module's own recovery tests) keep
  working. Mutation-checked: deleting the top-of-stack shortcut makes
  `guest_fatal_at_depth_3_unwinds_one_level` fail exactly as predicted — the
  scan finds the OUTERMOST slot, so a raise at depth 3 unwinds all three levels
  instead of one.

Gate tests live in `src/embed.rs::g0_nested_entries` (`#[cfg(all(test,
windows))]`): `nested_entry_depth_5` (five nested `dispatch_callback` entries,
each running a real doit through `eval` so both entry kinds interleave on the
LIFO, each asserting from *inside* the nest that it owns a distinct recovery
slot and its own idle baseline) and `guest_fatal_at_depth_3_unwinds_one_level`
(spec gate test 2, pulled forward — it is the direct proof of LIFO recovery).

⚠️ **Coverage gap found while landing this.** `embed.rs`'s main test module is
`#[cfg(all(test, target_os = "macos"))]` — a Phase-2 WINVM decision whose stated
reason ("relies on `sigsetjmp`-based guest-fatal recovery, stubbed on Windows")
is now obsolete: the VEH + hand-written setjmp/longjmp landed in `eb8bdd7`. So
~90 embed tests, including the recovery-to-a-clean-baseline ones that guard
exactly the machinery this sprint reshapes, do not run on Windows at all. The
G0 module above is the only Windows coverage of `VmHandle`'s entry paths.
Un-gating that suite (the FFI/worker/Cocoa demos will need their own narrower
gates) is worth its own slice before G1 leans harder on this substrate.
