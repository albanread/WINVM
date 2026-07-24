# Upstream sync plan — MACVM main → windows-port, 2026-07-24

Status: PLAN (reviewed, not yet executed). Upstream fetched at `562f48d`;
merge base `ad716f9`; **40 new commits** to bring over. Last sync was
`fc2b813` (OSR-heal / de-speculation era).

## What landed upstream, grouped

| Group | Commits | What it is |
|---|---|---|
| **GC alloc-gap fixes** | `40fc343` `ad2846f` `5d79c27` `8704792` `9330b4b` `86aec53` | The two fixes `gc_alloc_gap.md` prescribed: default eden 4→16→32 MiB (scaled via `default_eden_for`), spliced-constructor `self basicNew` fused to the inline eden bump, plus a precise pooled-arg smi guard and a small-heap eden clamp from the stability battery. Outcome on Mac: **alloc bench now ties Cog 7 ms / 7 ms.** |
| **Special selectors** | `83c3c19` `20b37b0` | `==`/`~~`/`not` lower inline (`RefCmpVal`, `BoolNot` IR ops + arm64 emission). |
| **Frameless leaf methods (Mac F0–F3, F7)** | `7f11acd` `4ead13e` `d9587cc` `cec97ec` `145c881` `3072c77` | Cog-style `needsFrame` analysis: eligibility scan in the driver (F0), arm64 frameless emission (F1), **ON by default** with `MACVM_FRAMELESS=0` kill switch (F3), and the prologue nil-fill narrowed to skip slots unconditionally defined at entry (their F7 — the same F7 our `x64_codegen_perf.md` called out). ~2–4 % richards on Mac; scoreboard now **MACVM wins all seven vs Cog**. |
| **Optional static types T0′–T4** | `59b6b13` `6e5e686` `b358be9` `afa5a1e` `aef12da` `a6eb8bd` + 5 annotation slices | Strongtalk-style type annotations: parser captures them, a **new `src/types/` crate-module** (TypeExpr parser, interface builder, subtype rules, send rule), a `macvm typecheck` subcommand, and **54 world files annotated** (number tower, collections, strings, kernel, streams). |
| **OTP-style supervision O0–O3** | `9899307` `c67d767` `79adb24` `7b42a9e` `c5e17ad` | `WorkerSupervisor` (#oneForOne/#oneForAll/#restForOne), supervisor trees + escalation, live crash→respawn, `ServiceWorker` call/cast with deadline sweep; `world/74_supervisor.mst` + tests; IoWorker adopted onto it. |
| **Workspace multi-statement Do It** | `02e91c2` | Tagged "cocoa-gui" but the change is in **shared code** (`frontend/parser.rs` + `runtime/primitives.rs`) — our web-GUI Workspace inherits it from the merge for free. |
| **Bench/docs** | `7c495f1` `3a8121d` `ac0b750` `2b9e1af` `1160033` + README/help | Mac-side mirror of OUR cog-bench harness; Cog scoreboard docs; a Workers/OTP help page added to `gui/reference` (ships in our GUI too). |

## Portability read

Almost everything is portable-by-construction; the two arm64-specific items
are already handled or cleanly deferrable:

- **Special selectors: already converged.** The windows-port side
  independently cherry-picked the IR half — `RefCmpVal`/`BoolNot` exist in
  our `ir.rs` **byte-identical** to upstream (same fields, same doc
  comments), and `emit_x64` already lowers both (lines ~1673/1691). The
  merge should collapse to near-no-op here; any conflict resolves to
  either side.
- **Frameless emission: arm64-only for now, and safely so.** Upstream's
  driver computes `frameless_eligible` (arch-neutral) and passes
  `emit_frameless` into the **arm64** emit call. Our driver's emit call
  site arch-dispatches to `emit_x64`, which takes no such flag — so the
  merge resolution at that call site simply doesn't thread the flag on
  x64, and frameless stays inert on Windows (eligibility census still
  runs, which is useful data). An x64 frameless port is a real follow-up
  design item (RBP-chain walking must learn to tolerate frame-less leaves,
  as upstream taught the x29 walk), **not** part of this sync.
- **Their F7 nil-fill narrowing is in `regalloc.rs`/`ir.rs` (shared)** —
  x64 gets it free. That closes the F7 line item our
  `x64_codegen_perf.md` left open; update that doc's status after the
  merge.
- **Eden 32 MiB + basicNew fusion are `memory/` + `ir.rs` (shared).** This
  is the big one for Windows: our worst bench vs Cog was **alloc at
  138–141 ms vs Cog's 17–34** and `gc_alloc_gap.md` predicted exactly
  these two fixes. Expect the gap to collapse; re-measure.
- **Types, OTP, multi-statement Do It, help pages: fully portable** (new
  module, world text, shared frontend/runtime, HTML).

## Merge mechanics

House pattern: `git merge upstream/main` into `windows-port` (as
`d80fbe1`, `fc2b813`, `666da4b` before). Both-sides-touched files and the
intended resolution:

| File | Ours since base | Theirs | Resolution |
|---|---|---|---|
| `src/compiler/ir.rs` | RefCmpVal/BoolNot (cherry-picked), F2-lite remat, fp-Move ctx | same RefCmpVal/BoolNot + basicNew fusion + F7 | Converged text unions cleanly; take both fusions. Verify `SUPPORTED_OPS`/decline list still matches emit_x64's real coverage. |
| `src/compiler/driver.rs` | x64 emit dispatch, F-fixes | F0 eligibility + F3 default-on | Take theirs; do **not** thread `emit_frameless` into `emit_x64` (stays arm64-only). |
| `src/compiler/regalloc.rs` | loop-weighted residency (F3b), pool changes | F7 nil-fill narrowing + special-selector bits | Union — different functions. |
| `src/compiler/emit.rs` | untouched by us (arm64 file) | frameless emission +148 | Take theirs verbatim. |
| `src/memory/{layout,universe}.rs` | small Windows seams | eden geometry + clamp | Take theirs; re-check the `VirtualAlloc` reservation path against `default_eden_for`'s scaling. |
| `src/embed.rs` | Waker/recovery/idle-baseline (Windows) | eden default plumbing | Union — different regions. |
| `src/runtime/primitives.rs` | Windows prim gating | multi-statement doit | Union. |
| `src/main.rs` | Windows CLI bits | `typecheck` subcommand | Union. |
| `world/30_date_time.mst` | **Windows clock FFI** (VirtualAlloc + GetSystemTimePreciseAsFileTime, platform-selected) | type annotations | Take both — different hunks; annotations are header/ivar syntax. |
| `world/tests/{tests.list,99_run_all}` | Windows gating | supervisor tests | Union. |
| `README.md` | **rewritten for the Windows port** | Cog scoreboard + further-reading sections | **Keep ours**; port the scoreboard/further-reading ideas manually if wanted (ours already documents the Windows Cog table). |
| `scripts/cog-bench.{sh,st}`, `mst2st.py` | we authored them | Mac mirrored + extended (µs clock) | Take theirs where extension, keep our Windows invocation paths — these started on OUR side, so review hunk-by-hunk. |
| `image_store/src/lib.rs` | batched backfill txn | (their side change) | Union; keep the one-transaction backfill. |

## Execution order

1. **Preflight.** Clean tree (coordinate with the parallel session — it is
   actively editing `compiler/`); record baseline: 733 lib tests, world
   5891/0, bench numbers (sieve 4 ms, alloc ~139 ms, arith/dict/fib).
2. **`git merge upstream/main`**, resolve per the table above.
3. **Build gates.** `cargo build` (zero new warnings), `cargo build -p
   winvm-gui`, then `cargo test --lib` (expect ≥733 + the new types/OTP
   tests), world suite, `tests/it_typecheck.rs`, integration suites
   re-armed by `86aec53`.
4. **Windows-specific re-verification.**
   - Differential suite (interp vs JIT) — the special-selector and
     basicNew-fusion paths now execute through `emit_x64`.
   - OTP supervisor world tests on Windows (worker threads + the Windows
     waker path; watch the crash→respawn tests against our
     `FatalMode::ExitThread` semantics).
   - GUI smoke: boot, doit, browser, canvas Mandelbrot (fp Move fix must
     survive the merge), Workspace **multi-statement** Do It (new).
   - `macvm typecheck` subcommand runs against the annotated world.
5. **Re-measure benches** (the point of half these commits):
   - alloc: expect 138 ms → near Cog parity (Mac went to 7 ms tie).
   - richards/deltablue/sieve/fib/arith/dict: fresh scoreboard, and note
     the bench-class rename (`Bm*`) already fixed the Packet/Strength
     collision on our side.
6. **Docs.** Update `x64_codegen_perf.md` (F7 now closed by upstream's
   shared fix; frameless = new x64 follow-up item), MIGRATION.md status
   log entry for the sync.
7. **Commit** the merge with the reconciliation notes in the message
   (house style: name what conflicted and which way each went).

## Follow-ups spawned by this sync (not blocking it)

- **x64 frameless leaves** — port F1's emission + RBP-walk tolerance;
  upstream measured ~2–4 % on richards, and our richards residual is
  per-activation fixed cost, so it should help x64 *more* than arm64.
- **Known pre-existing GUI-suite abort** (`workspace_print_it_works_with_
  the_jit_enabled` → `deopt.rs:665` receiver-ValueLoc assert →
  `panic_cannot_unwind` in `rt_uncommon_trap`). Upstream's deopt changes
  may move this; re-test right after the merge and re-triage either way.
- **Annotate Windows-only world files** (sockets, ioworker extensions,
  supervisor adoptions) to keep `typecheck` clean across the whole world.

## Risks

- The parallel session is mid-flight in `compiler/` — merging under it
  invites the same live-conflict churn seen at `d80fbe1`. Do the merge in
  a quiet window or in a worktree, then land atomically.
- Eden 32 MiB default changes memory pressure on small machines; the
  clamp (`8704792`) exists, but verify the Windows `VirtualAlloc`
  reservation math honors it.
- Type annotations touching 54 world files is a wide textual surface
  against our world edits; conflicts should be mechanical (header
  syntax) but each needs eyes — a silent mis-merge here breaks the boot
  oracle, which is the compatibility guarantee between the two VMs.
