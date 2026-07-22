# Dolphin MVP GUI — VM change specs, gap register, and the Phase-G sprint plan

**Status:** plan of record for executing [dolphin_ui_porting.md](dolphin_ui_porting.md).
**Date:** 2026-07-22. House conventions per [docs/SPRINTS.md](docs/SPRINTS.md): every
sprint ends green (all prior tests pass + the sprint's own gate); sizing S/M/L.
Sprint letters: **G** (GUI-native track), plus **W5** (the world exceptions phase
already designed in [docs/WORLD.md](docs/WORLD.md) §7 — scheduled here, not redesigned).

Architecture alignment: this plan adapts the port to the **MULTIVM** design
([docs/multi-smalltalk-worker.md](docs/multi-smalltalk-worker.md)) — share-nothing
worker VMs, copy-passing, continuations, **no green threads** — and to the
no-`become:` philosophy. The adaptation decisions themselves live in the design doc
(§5.8 MULTIVM amendment, §5.9 become-audit); this doc is *how we build it*.

---

## Part 1 — VM/substrate change specifications

Numbered V-specs; each is owned by exactly one sprint. File references are to today's
`windows-port` branch.

### V1 — Re-entrant embed entries (design §5.3 S1) — owner: G0

**Problem.** `VmHandle::dispatch_callback` (`src/embed.rs`) is sound only for
*top-level* entries: one shared `sigsetjmp` slot + one idle-baseline watermark per
thread (`deopt_trap::claim_jmp_slot`, `snapshot_idle_baseline`), guarded by a
thread-local `IN_CALLBACK` boolean that **fails closed** on nesting
(`embed.rs:780` returns the shape default). Win32 forces nesting: handler → FFI
(`CreateWindowExW`, `SendMessageW`, `DestroyWindow`) → wndproc → nested entry, to
arbitrary depth.

**Change.**
1. **Slot stack.** In `codecache/deopt_trap.rs`: replace the single recovery slot +
   idle-baseline watermark with a per-thread **stack of (jmp-slot, baseline)** frames.
   `claim_jmp_slot` → `push_entry_frame` / `pop_entry_frame`; `siglongjmp` recovery
   targets the **top** frame only (LIFO — the discipline Dolphin's callback cookies
   enforce with `Retry`; we get it for free because nested entries return normally).
   `restore_after_guest_fatal` restores to the top frame's baseline, then pops.
2. **Depth counter.** `IN_CALLBACK: Cell<bool>` → `CALLBACK_DEPTH: Cell<u32>`.
   `callback_active()` keeps its name/meaning (`depth > 0`) for existing callers.
   All exit arms — normal return, guest-fatal arm, native-fault arm — decrement;
   the arms already run before `Drop`-skipping `siglongjmp` landings, same as today.
3. **Aliasing discipline (the `&mut` seam).** The outer entry holds `&mut VmState`
   through `dispatch_ffi_primitive` while the native call runs. The nested entry
   needs it back. Mechanism: before the FFI stub transfers to native code, the
   dispatch layer publishes the `VmState` pointer in a thread-local **re-entry
   token** (the same pattern as the Cocoa trampolines' thread-local `*mut VmHandle`);
   the wndproc trampoline re-materializes `&mut` from the token. Soundness contract
   (documented at both sites, mirroring the FFI layer's existing "no allocation
   before the call" rule): **the FFI stub touches no VM state between publishing the
   token and the native call's return**, so the outer borrow is quiescent for the
   whole nested window. This is an unsafe-but-disciplined seam of the kind the
   codebase already maintains deliberately.
4. **Generation checks stay outermost.** `UI_VM_GENERATION` (`embed.rs:134`) is read
   at the trampoline door as today; nested entries inherit the generation of their
   outer entry (same VM, same thread).

**Explicitly out of scope:** re-entry from *other threads* (workers never call back);
re-entry into a *different* VM; JIT-frame unwind (V8/S12 — GUI dispatch stays
interpreter-tier, which FFI methods already are).

**Tests (the G0 gate):**
- `nested_entry_depth_5`: synthetic native fn (Rust test double standing in for
  `CreateWindowExW`) that re-enters `dispatch_callback` 5 deep; each level allocates
  (forces GC eligibility) and returns a distinct value; assert all five land.
- `dnu_at_depth_3_unwinds_one_level`: guest DNU at depth 3 → depth-3 entry answers
  its default, depths 2/1 continue normally and return their own values; transcript
  carries exactly one walkback line.
- `native_fault_at_depth_2`: recovered AV in nested marshalling → same containment.
- `lifo_baseline_restore`: after any recovery arm, the *outer* entry's later fault
  still lands in the outer slot (regression for the "returned-frame longjmp" hazard
  described at `embed.rs:151-156`).
- Existing 730+ lib tests green; the Cocoa-path fail-closed unit test is **replaced**
  (its premise — nesting is refused — is deliberately obsolete).

### V2 — W5 exceptions (design §5.3 S7) — owner: W5 sprint

Per [docs/WORLD.md](docs/WORLD.md) §7, already committed design: Strongtalk's ANSI
exception layer is 100 % in-image (blocks + NLR + `ensure:` + a handler chain),
**zero new VM features**. Scheduling and MVP-specific scope only:

- Port the ~10 classes (Exception, Error, Warning, Notification, ZeroDivide, Halt,
  MessageNotUnderstood, ExceptionSet, LinkedExceptionHandler, BlockExceptionHandler)
  via the existing `dlt2mst` converter, with the upstream tests
  (ExceptionTest/ZeroDivideTest).
- Single-process v1: `handlerChain` in one global (`Smalltalk handlerChain`) — the
  W5 design's own call; the MVP GUI is single-process, so this suffices for the
  whole GUI track. (Nested `dispatch_callback` entries interleave handler frames on
  the one chain in strict LIFO — nesting is just deeper Smalltalk recursion from the
  chain's point of view; add one test proving a handler installed at callback depth
  2 doesn't leak into depth 1 after the entry returns.)
- Wire-in: DNU → `MessageNotUnderstood` **raised through the chain first**; only an
  unhandled exception falls through to today's guest-fatal recovery at the embed
  boundary (behavior for unhandled errors is *unchanged* — the pump answers
  DefWindowProc and continues). `error:` → `Error signal:`.
- MVP-facing surface needed by G5 (verify these exist in the Strongtalk port or add
  thin methods): `signal`/`signal:`, `on:do:`, `pass` (the `basicPaint:` re-raise),
  `return:`, `retry`, resumable `Warning`/`Notification` (converter beeps are
  non-resuming; nothing in the v1 scope needs `resume:` beyond Notification's
  default). Dolphin's `Signal description:` idiom and `Win32Error`/`InvalidFormat`/
  `OperationAborted` are **dolphin_compat subclasses** (G3), not W5 work.
- ⚠️ WORLD.md's own caveat stands: signal-through-*optimized*-frames is an S13
  stress item. Until exercised, exception-heavy GUI paths stay interpreter-tier
  (FFI methods already are; nothing pins hot pure-Smalltalk methods yet — accept
  and note, since the x64 JIT currently declines NLR-bearing methods anyway).

**Gate:** ported exception test suite green interpreted; DNU→MNU catchable
(`[nil foo] on: MessageNotUnderstood do: [:e | …]`); unhandled path byte-identical
to today (existing embed tests unchanged); the depth-2 handler-chain test above.

### V3 — `win_gui` native host (design §5.3 S2) — owner: G1 (skeleton), grows through G7

New crate `gui_native/` (bin working name `winvm-mvp`), following the
`macvm-cocoa`-as-dedicated-bin precedent; shares world/boot/embed plumbing with
`gui/` where practical.

Surface (Rust):
- `register_window_class("WinvmWindow", wndproc)` + the **shared wndproc**: reads
  hwnd→generation + hwnd→oldWndProc maps; enters the image via
  `dispatch_callback` → `UiSession wndProc: hwnd message: m wParam: w lParam: l`;
  Integer answer ⇒ LRESULT; nil/recovery ⇒ `DefWindowProcW` or
  `CallWindowProcW(oldWndProc)`; **WM_PAINT backstop**: if the entry came back via
  the recovery path, `ValidateRect(hwnd, NULL)` before answering (kills the
  pre-W5 paint-error storm — risk table item).
- **WH_CBT hook** (thread-scoped): `HCBT_CREATEWND` → image `UiSession
  windowCreated: hwnd` *only when* a view-under-construction is pending (the
  `newWindow`-slot analogue lives in UiSession); everything else passes through.
  Fallback path (if the hook misbehaves with native dialogs): `lpParam`+WM_NCCREATE
  association for our class, post-create `SetWindowLongPtrW` subclassing for
  controls — keep both behind a flag until G4 settles it.
- `subclass_control(hwnd)` / oldWndProc bookkeeping; `unsubclass` on WM_NCDESTROY.
- **The pump**: `GetMessageW` loop with native fast paths — registered HACCEL →
  `TranslateAcceleratorW`, registered dialog-shell hwnd → `IsDialogMessageW` — then
  Translate/Dispatch. **Pump-empty hook**: when `PeekMessageW` reports an empty
  queue after activity, one `UiSession onIdle` entry (idle-time command
  revalidation, tooltip housekeeping).
- `run_modal_loop(owner_hwnd)` / `end_modal(cookie)`: nested pump for `showModal`
  and mouse-capture loops (design §5.8 table).
- **Message-only window** for posted actions: host API `post_action()` +
  `WM_APP_DRAIN` → `UiSession evaluateDeferredAction`; this same window is the
  **MULTIVM inbox wake target** (`inbox_wake = PostMessageW(msg_window, …)`), so
  worker replies fire continuations between messages.
- Lifecycle: boot world + `UiSession startUp`; `WM_QUIT` → orderly image shutdown →
  process exit. **Supervisor arm** (post-G4): on hard-fatal thread death —
  generation bump, `DestroyWindow` survivors, reboot VM, re-run `UiSession startUp`
  ("reconnecting" model from the Cocoa doc, §5.8).

### V4 — UTF-16 support (design §5.3 S3) — owner: G1 (minimal), G2 (full)

- Two primitives (or one prim pair on the Alien layer): `utf8→utf16`
  (String → freshly-malloc'd NUL-terminated Alien, answers {alien. codeUnitCount})
  and `utf16→utf8` (address+len → String). Pure-Smalltalk fallback over
  `Alien byteAt:put:` exists conceptually but W-API frequency says primitive.
- World side (dolphin_compat, G3): `Utf16String` value-ish class wrapping
  {alien. codeUnits}, `String>>asUtf16String`, `withUtf16Do:` (scoped, `ensure:`d
  free), `Utf16String>>asString`. **Code-unit lengths live on Utf16String** —
  `EM_GETSEL`-style index math never touches byte-indexed world Strings (gap G-c).

### V5 — FFI ergonomics + last-error (design §5.3 S4/S6) — owner: G1

- Audit + (where missing) add: `g`-arg coercions for `nil`→0 and Alien→address;
  **sign-extension round-trip** for negative LPARAM/LRESULT (mouse coordinates are
  packed signed shorts — add explicit tests at the 61-bit smi boundary, both
  directions).
- **Last-error capture**: the x64 FFI stub (`src/codecache/ffi_stubs_x64.rs` /
  `src/runtime/ffi.rs`) stores `GetLastError` into a per-VM slot immediately after
  the call; expose as a primitive → `UiSession lastError`. Replaces Dolphin's
  read-it-later contract (design §3.5).
- **winkb resilience**: widen the dlsym fallback probe list with the user32 / gdi32 /
  comctl32 / shell32 staples the GUI needs, so a missing
  `E:\windows_api\windows_api.db` degrades with a diagnostic instead of dying
  (risk-table item).

### V6..V8 — deferred (sketches only, scheduled in G8+)

- **V6 (S9) weak refs + finalization**: single finalize queue drained by a
  `UiSession onIdle` step; weak-value map for the events registry + Dolphin's
  `CommandDescriptionRegistry`/`SharedBitmaps`. Mourning-array semantics only if we
  later choose full events-registry fidelity.
- **V7 (S10) generic callback thunks**: Rust thunk allocator (cookie → block via
  `dispatch_callback`) for `EnumWindows`/`EnumFontFamiliesEx`; v1 scope avoids
  these APIs entirely.
- **V8 (S11/S12)**: worker-thread FFI for non-freezing blocking calls;
  `RtlAddFunctionTable` unwind registration if GUI code is ever allowed to fault
  inside JIT frames. Both parked with rationale in the design doc.

---

## Part 2 — Gap register (this pass)

| # | Gap found | Resolution | Owner |
|---|---|---|---|
| G-a | Cocoa doctrine says "no nested-entry machinery"; Win32 requires nesting | Promoted to designed VM feature **V1** with LIFO slot stack; doctrine amended in design §5.8 | G0 |
| G-b | Pre-W5 paint-handler error → `EndPaint` skipped → infinite WM_PAINT storm | Rust `ValidateRect` backstop in the wndproc recovery arm | G1 (V3) |
| G-c | UTF-16 code-unit vs UTF-8 byte index math (selections, `BCM_GETNOTE`, DrawText lengths) | `Utf16String` owns code-unit counts; conversions only at control boundaries; translator flags String index arithmetic (reuses WORLD.md §6 review rule) | G2/G3 (V4) |
| G-d | Dolphin dialogs dismissable out-of-order (forked mains); nested pumps are LIFO | Accepted divergence (standard Win32 modality); documented §5.8/§7 | — |
| G-e | Cocoa "dumb terminal" snapshot doctrine can't host MVP (authoritative triads, synchronous Win32 answers) | MVP UI VM declared authoritative; crash story = per-message recovery + generation-checked respawn | design §5.8 |
| G-f | Long doits freeze the GUI (no green threads) | MULTIVM workers + `send:…onReply:`; `inbox_wake` = PostMessage to the host message-only window; >50 ms command doctrine; ProgressDialog rewritten on envelopes | G5/G7 |
| G-g | `ProgressDialog`/splash/autoscroll fork green processes | Site-by-site replacements (SetTimer / workers / cut) — exhaustive table in design §5.8 | G5–G7 |
| G-h | Idle-time command revalidation had no home without the Smalltalk pump | Pump-empty hook → `UiSession onIdle` | G1 (V3) |
| G-i | GUI-VM hard-fatal respawn vs live HWNDs | Generation-stamped wndproc dispatch (stale → DefWindowProc), supervisor teardown + `UiSession startUp` re-run | G4+ (V3) |
| G-j | CBT hook observes foreign windows (native dialog internals) | Associate only with a pending view-under-construction; flagged fallback path (lpParam/NCCREATE) kept alive until G4 | G1 (V3) |
| G-k | `perform:` arity — `dispatchMessage:` needs 3-arg perform | `perform:withArguments:` (prim 64) already takes an array; translator emits the array form where Dolphin used `perform:with:with:with:` if a fixed-arity form is missing | G2 |
| G-l | Nested entries × exception handler chain (single global) | LIFO interleaving is naturally correct; pinned by a W5 test (handler at depth 2 must not leak to depth 1) | W5 |
| G-m | Dolphin's non-freezing `MessageBoxIndirectW` used `<overlap>` | v1 blocks (as Dolphin's own common dialogs do); revisit at V8 | — |
| G-n | `become:` exposure | Fully audited closed — design §5.9 (one site, eliminated by resource transpilation; filer unported; recreate is destroy+recreate, no identity swap) | — |

---

## Part 3 — Phase G sprint plan

Dependency spine: `G0 → G1 → {G2, G3, W5 in parallel} → G4 → G5 → G6 → G7 → G8`.
G2 and G3 are independent of each other; W5 needs nothing from G-track and gates G5.
Sizing: S = a focused day or two, M = up to a week, L = 1–2 weeks (house scale).

### G0 — Re-entrant embed entries `M` — *the keystone*
Implements **V1** (slot stack, depth counter, re-entry token, LIFO restore).
- Touches `src/codecache/deopt_trap.rs`, `src/embed.rs`, the FFI dispatch seam in
  `src/runtime/ffi.rs`.
- No GUI code, no Dolphin code — a pure VM-soundness sprint, testable with Rust
  test doubles (no real windows needed).
- **Gate:** the five V1 tests green + full suite green. This sprint is deliberately
  first: if the re-entry model has a flaw, it must surface before anything is built
  on it.

### G1 — `win_gui` skeleton + hello window `M`
Implements **V3** (skeleton: class registration, wndproc, CBT hook, pump, posted
actions, WM_PAINT backstop), **V4-minimal** (`asUtf16String` alien path), **V5**
(coercions, sign tests, last-error, probe list). A ~200-line hand-written `.mst`
(`world/mvp/00_uisession_spike.mst`) — *not* Dolphin code — drives it.
- **Gate (= design milestone UI-0):** from the image: register class, create a
  top-level window + native BUTTON child, `TextOutW` "hello" from a paint handler,
  click → Transcript round trip, clean close. Plus the two acceptance tortures:
  window created from inside a `WM_CREATE` handler **5 levels deep**, and a DNU in
  a handler that the pump survives (next message dispatches normally).

### W5 — Exceptions (parallel track, after G0 lands) `M`
Implements **V2** as specified above (the WORLD.md §7 plan, unmodified, plus the
callback-depth handler test).
- **Gate:** as V2. Gates G5; nothing in G2–G4 may depend on `on:do:`.

### G2 — `dolphin2mst` translator core `L`
The ingestion tool (design §5.4): chunk parser (BOM/CRLF/`!!`), `.pax` manifest +
loose-methods + inline-pool reader, D8 name resolution + flattening + rename table,
pool/`##()`/`classConstants:` folding, `??`/annotation rewrites, `<stdcall:>` →
WINVM FFI rewriter, struct-accessor generation (**V4-full** lands here for
`Utf16String` conversions in generated wrappers), collision report, patch overlay.
First corpus: **Dolphin Basic Geometry** + the `OS.UserLibrary`/`OS.GDILibrary`
signature subset + core structs (RECT/POINT/MSG/WNDCLASS/PAINTSTRUCT/NMHDR/
SCROLLINFO).
- **Gate:** translated `Graphics.Point`/`Rectangle` tests green in the world (per
  the collision decision, §8 Q1); a translated `UserLibrary getClientRect:` call
  fills an Alien-backed RECT correctly from live Win32; translator golden-file
  tests (input chunk → expected `.mst`) for every rewrite class; re-run over dsfork
  is byte-stable.

### G3 — Compat kernel `M`
Hand-written `world/dolphin_compat/`: the **event system ported verbatim** from
`Core.Object`/`Kernel.Events*` semantics (strong storage v1) + a dedicated
semantics test file (trigger return value, argument merge from the left,
add-during-trigger exclusion, idempotent registration, `removeEventsTriggeredFor:`,
`noEventsDo:`); `Model`, `SearchPolicy`, `DeafObject`/`DeadObject`, `LookupTable`
alias, `GUID newUnique`, `expandMacros`/`<<`, `propertyAt:`, `Cursor showWhile:`
shim, exception-compat subclasses (`Win32Error`, `InvalidFormat`,
`OperationAborted` — depends on W5 only for their superclass; stubs raise `error:`
if W5 hasn't landed, swapped after); **`UiSession` proper** (window registry +
`lastWindow` cache, `windowCreated:`, `wndProc:` entry routing on the ported
`dispatchMessage:` shape, deferred actions, `onIdle`, startup/shutdown).
- **Gate:** event-semantics suite green (written from design §3.2 before porting
  consumers); UiSession drives the G1 spike windows unchanged (spike `.mst`
  deleted, replaced by compat classes).

### G4 — View core vertical slice `L` (= milestone UI-2)
First translated MVP corpus: `UI.View` (trimmed per §5.6 stubs: dpi ^96, no
drag-drop, no theme), `ContainerView`, `ShellView`, `CreateWindow*`,
`BorderLayout` + `LayoutContext`, `PushButton`/`CheckBox`/`StaticText`/`GroupBox`,
`GdiCanvas` + Pen/Brush/Font/Color basics, MessageMap dispatch, `WindowsEvent`/
`PaintEvent`/`KeyEvent`/`MouseEvent`. Supervisor respawn arm of V3 lands here
(generation teardown demo).
- **Gate:** code-built shell — caption, BorderLayout with label + two buttons, live
  resize relayout, WM_PAINT via GdiCanvas, focus/tab between buttons, clean
  destroy with registry hygiene (`purgeDeadWindows`-equivalent asserts empty);
  kill -9 the VM thread mid-run → supervisor rebuilds the shell.

### G5 — MVP triad `L` (= milestone UI-3; **needs W5**)
Presenter/Shell/ValueModels/ListModel/TypeConverters/Command framework +
Menu/MenuBar/accelerators + `TextEdit` (per §8 Q2 decision) + `ListBox`. Worker
doctrine lands: a demo command ships work to a MULTIVM worker, continuation updates
a ValueModel (G-f).
- **Gate:** a real MVP app: model object edited through two `TextPresenter`s with a
  `NumberToText` converter (bad input → `InvalidFormat` beep-and-revert path, i.e.
  exceptions in anger), menu commands with `queryCommand:` enablement, an
  accelerator, and a long-running command that visibly does **not** freeze the pump
  (worker + continuation).

### G6 — Common controls `L` (= milestone UI-4)
`ListView` (report/icon, static update mode), `TreeView` (#dynamic), `TabView`/
`CardContainer`, `StatusBar`, `ScrollingDecorator`, `Splitter` +
`ProportionalLayout`, ImageList/.ico icons, `ListPresenter`/`TreePresenter`.
- **Gate:** the two-pane browser shell (tree of world classes left, method list
  right, text pane bottom) over live reflection data — the classic Dolphin idiom
  as the acceptance app.

### G7 — Dialogs & polish `M` (= milestone UI-5)
`DialogView` modality on `run_modal_loop`, `AspectBuffer` ok/cancel semantics,
MessageBox (blocking `MessageBoxW`/`TaskDialogIndirect`), `GetOpenFileNameW`
family, Prompter-family via transpiled resources (first use of the `--resources`
subcommand), clipboard text, keyboard-navigation audit (IsDialogMessage edge cases
from design §3.3).
- **Gate:** modal prompter editing a value with buffered OK/Cancel opened from the
  G6 browser; two stacked modals (LIFO); file-open dialog round trip; paste into
  TextEdit.

### G8 — Fidelity backlog (ordered, pull-based)
V6 weak/finalization → events-registry + GDI leak closure; Toolbar + idle
revalidation completion; ListView virtual mode/custom draw; runtime STL reader;
system-DPI then per-monitor DPI (unstub the three funnels); themes; RichEdit; V7
thunks (EnumFontFamilies → Font dialog); V8 worker-thread FFI (non-freezing
MessageBox, sockets parity); drag-drop.

### Corpus checkpoints
G4 ≈ 30–40 translated classes · G5 ≈ +60 · G6 ≈ +40 · G7 ≈ +30 — matching the
design doc's 150–200-class v1 band. The translator (G2) and compat kernel (G3) are
the long poles before visible progress; G0/G1 are small but gate everything.

---

## Part 4 — Decision log (this pass)

1. **Nested VM entries are in** (V1/G0) — Win32's synchronous sends make the Cocoa
   "always top-level" doctrine unavailable here; the LIFO slot stack is the honest
   version of what Dolphin's callback cookies do.
2. **The MVP UI VM is authoritative, not a snapshot terminal** — required by
   synchronous message answers and by MVP's state model; crash safety comes from
   per-message recovery + generation-checked respawn instead (design §5.8).
3. **W5 exceptions run as their own parallel sprint after G0**, gating G5. Nothing
   before G5 may use `on:do:`.
4. **Dedicated `winvm-mvp` bin** (the `macvm-cocoa` precedent), sharing world/embed
   plumbing.
5. **Green-thread sites are replaced site-by-site** (design §5.8 table) — no
   green-thread shim is built, ever, for this track; if S17-style processes land
   later they slot in without changing this plan.
6. **`become:` stays at zero** — design §5.9 audit closed all exposure.
