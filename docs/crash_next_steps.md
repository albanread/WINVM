# GC-stress compile-path crash — debugging next steps

A latent access-violation surfaced while porting frameless leaves to x64
(`frameless_x64_findings.md`, finding 2). It is a **real GC-root bug in the
compile path**, not a frameless bug — frameless's read-only eligibility scan
merely shifted scavenge timing enough to unmask it. This is the playbook for
root-causing it. Written because no debugger was available in the session that
found it; the first job below is to fix that.

## TL;DR

- **Symptom:** `ACCESS_VIOLATION … addr 0x8` (a near-null field deref) during a
  `MACVM_GC_STRESS=1` world-suite run, followed by the crash reporter
  re-faulting (repeated identical `pc` in a system DLL).
- **Almost certainly:** an oop is live across one of the compile path's own
  heap allocations (`Nmethod` / code-cache / pool) without being a GC root, so
  a stress scavenge in that window moves it and leaves a stale pointer.
- **Prime suspect:** literal-pool oops held across `compile_method_full`'s
  allocations.
- **It is timing-masked, not absent** on `main` today — any future change that
  shifts compile-path timing can retrigger it. Worth fixing regardless of
  frameless.

## Reproduce

On branch **`frameless-x64-wip`** (`b4ce302` — has F0's scan, which exposes it):

```bash
git checkout frameless-x64-wip
cargo build --release
# Build the world test concatenation (posix-only files excluded on Windows):
grep -v '^#' world/tests/tests.list | grep -v '^$' | grep -v '^posix-only:' \
  | sed 's|^|world/tests/|' | xargs cat > /tmp/world_tests.mst
MACVM_JIT=threshold=20 MACVM_GC_STRESS=1 \
  ./target/release/macvm run /tmp/world_tests.mst --world world
```

Observed: dies right after the `ScaledDecimal` block (`testPrintString`,
`world/tests/46_scaleddecimal_tests.mst`). The death point is a *timing*
artifact, not necessarily the guilty method — treat it as a starting anchor,
not the answer.

**Control (proves it is the scan's timing, not emission/struct/stack):**
- `MACVM_FRAMELESS=0` (scan runs, emission off) → still crashes.
- Short-circuit the scan (re-add the temporary probe below) → **green, 5860/0.**
- `/STACK:8388608` and `/STACK:67108864` → no change (not a stack overflow).

Temporary bisection probe (was used, then reverted) — re-add to
`driver.rs::frameless_eligible`'s first line:
```rust
if std::env::var_os("MACVM_NOSCAN").is_some() { let _ = (m, ra); return false; }
```
Then `MACVM_NOSCAN=1 …` should pass. This confirms you're looking at a
timing-exposed latent bug, not new code.

## Step 0 — get a debugger AND make the crash address stable

This is the single biggest unblock. Two independent moves:

1. **Install a debugger.** WinDbg (Microsoft Store: "WinDbg") or `cdb.exe` from
   the Windows SDK "Debugging Tools for Windows" feature. Then:
   ```
   cdb -g -G target\release\macvm.exe run C:\tmp\world_tests.mst --world world
   ```
   (set `MACVM_JIT`/`MACVM_GC_STRESS` in the env first). On the first-chance AV,
   `k` gives the stack, `r` the registers, `!analyze -v` the summary. The Rust
   frames symbolize from `target/release/macvm.pdb` (already emitted).

2. **Defeat ASLR so the pc is stable across runs** — lets you symbolize without
   a live debugger and compare runs. Add to `.cargo/config.toml`:
   ```toml
   [target.x86_64-pc-windows-msvc]
   rustflags = ["-C", "link-arg=/DYNAMICBASE:NO", "-C", "link-arg=/MAP:macvm.map"]
   ```
   `/MAP` emits a symbol map; with a fixed base, `RVA = fault_pc - image_base`
   resolves straight out of `macvm.map` — no debugger needed for the first cut.
   (Remove both flags before shipping; `/DYNAMICBASE:NO` weakens the binary.)

Without either, you're where this session was: a bare hex `pc` you can't name.

## Step 1 — make the crash reporter survive (cheap, high-leverage)

The PROBE dossier currently gives up on this fault: the pc is in Rust runtime
code, not the code cache, so `deopt_trap`'s handler prints
`FOREIGN (not in any code cache); dying` and builds no dossier — and then the
reporter itself **re-faults** (the repeated identical system-DLL `pc`). Two
small fixes turn the crash into a self-report:

- In the Windows VEH foreign-fault path (`codecache/deopt_trap.rs`,
  `write_foreign_verdict_win`), also print **module base + RVA** of the faulting
  pc (via `GetModuleHandleExW`/`GetModuleInformation`), so the one honest line
  is symbolizable straight away.
- Find why the reporter re-faults (it should print one line and exit). A guard
  against re-entrancy in the foreign-fault path likely already exists for the
  recovery case — the reporting-only case may be missing it. Fixing this alone
  would have made this session's diagnosis trivial.

## Step 2 — shrink the repro

The full suite is a blunt instrument. Narrow to the smallest crashing input:

- **Bisect the test list.** Halve `/tmp/world_tests.mst` (keep the harness
  prefix that defines `TestCase`/`TestRunner`) until a minimal set still
  crashes. The anchor says start around `46_scaleddecimal` / the LargeInteger
  and Fraction neighbours.
- **Find the last method compiled before the fault.** Add a one-line
  `eprintln!` of `holder>>selector` at the top of `compile_method_full` (debug
  build), run under GC-stress, and read the last line before the AV. That names
  the method whose compile (or whose freshly-compiled execution) hits the fatal
  scavenge window.
- **Tighten GC further** to make it fire more deterministically: `MACVM_EDEN`
  set very small forces a scavenge at almost every allocation, which should make
  the crash immediate and position-stable.

## Step 3 — confirm the hypothesis (compile-path root gap)

The theory: `compile_method_full` (and callees) hold an oop — most likely
literal-pool entries, which ARE oops (`pool[i] = 0x… Some(Oop)`) — across a heap
allocation (`Nmethod`, code-cache growth, pool interning) that can scavenge
under stress, without that oop being on the root stack the scavenger scans.

Confirm/refute without guessing at the fix:

- With the debugger stopped at the AV, look at the faulting register/operand:
  `addr 0x8` means "base pointer was null/near-null, read field +8." Whose
  field? Walk `k` to the compile-path or nmethod-execution frame and identify
  the object.
- **Root-set audit:** grep the compile path for oops held in locals across an
  allocating call. Compare against what `memory::roots` / the handle scope
  actually scans. The MACVM side does this correctly (their GC-stress is green),
  so a **three-way diff of `compile_method_full` and its oop handling between
  this tree, `frameless-x64-wip`, and `upstream/main`** may show the gap
  directly — especially anything the windows-port added that keeps an oop in a
  raw local rather than a rooted handle.
- **Instrument the scavenger:** under `MACVM_GC_STRESS`, log every object it
  moves whose old address matches a pool word the current compile is holding.
  A hit is the smoking gun.

## Step 4 — the likely fix shapes

Depending on what step 3 finds:

- **Missing handle scope in the compile path:** wrap the oops the compiler holds
  (receiver klass, method oop, pool oops) in a `HandleScope` / push them as
  roots for the duration of `compile_method_full`, so a mid-compile scavenge
  updates them. This is the most probable fix.
- **Pool interned after an allocation that can move its inputs:** reorder so the
  pool oop is rooted before the allocation that can scavenge, or intern before
  allocating the `Nmethod`.
- **A raw `*const`/`*mut` oop cached across a `vm.alloc`:** re-read it from its
  rooted home after the allocation instead of caching.

Whatever it is, the acceptance gate is the same: **`frameless-x64-wip`'s
GC-stress world run goes green (5860/0), repeatedly**, and then the same run on
`main` stays green under a deliberately perturbed compile path (e.g. the scan
re-enabled) — proving the root gap is closed, not just re-masked.

## Why this matters beyond frameless

`MACVM_GC_STRESS=1` is the tool that certifies GC-root coverage by making every
allocation a collection point. A crash it can produce is a **real missing root**
— today masked by allocation timing, but a hazard for any future compiler or IR
change that shifts that timing (and, in principle, for an unlucky production GC).
Fixing it is worth a session on its own; frameless was just the messenger.

## Pointers

- Branch with the repro: `frameless-x64-wip` (`b4ce302`).
- Ruled-out matrix and the perf verdict: `docs/frameless_x64_findings.md`.
- Windows fault handling: `src/codecache/deopt_trap.rs`
  (`veh_trap_handler`, `write_foreign_verdict_win`, the recovery registry).
- Compile entry: `src/compiler/driver.rs` `compile_method_full`.
- GC roots: `src/memory/roots.rs`; scavenger: `src/memory/`.
- Symbols: `target/release/macvm.pdb` is emitted for release builds.
