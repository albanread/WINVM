# Dolphin MVP → WINVM: the Windows-native GUI port

**Status:** design proposal (no porting started). **Date:** 2026-07-22.
**Scope:** port Dolphin Smalltalk 8's MVP GUI framework — the Smalltalk classes only —
from `e:\dsfork` onto WINVM as an **optional, Windows-centric native GUI**, i.e. the
"second wave" already anticipated by [MIGRATION.md](MIGRATION.md) §Phase 6:
*"Win32-native shell, UI written in Smalltalk driving real Win32 controls."*

This document is grounded in a file-level review of both trees (seven parallel review
passes over `e:\dsfork` and `E:\WINVM`). Every class/method/mechanism named below was
read in the fork's sources, not recalled from general knowledge.

**Companion:** [dolphin_ui_sprints.md](dolphin_ui_sprints.md) — VM-change
specifications (re-entrant callbacks, W5 exceptions, win_gui host, UTF-16, …), the
gap-review register, and the Phase-G sprint plan. §5.8/§5.9 below are the
MULTIVM / no-green-threads / no-`become:` adaptation amendments from that pass.

---

## 0. Executive summary

- **The source is clean and portable in licence and shape.** `e:\dsfork` is a fork of
  Object Arts Dolphin Smalltalk 8 (MIT, © 2015 Object Arts). Its Smalltalk side is
  **stock upstream D8** — the fork's 105 commits diverge almost entirely in the C++ VM
  (its own x64 modernization); only 4 Smalltalk files changed, none in MVP. We port
  from unmodified D8 sources.
- **The minimal "Shell + buttons/lists/text" closure is ~700 classes / ~165k lines**
  (391-class base kernel + 16 add-on packages). We do **not** port the Dolphin kernel —
  WINVM keeps its own; we port the MVP/Graphics layers and bridge their kernel
  dependencies through a compatibility layer. The realistic v1 translated corpus is
  **~150–200 classes / 40–60k lines**, mechanically translated, plus ~5k hand-written
  compat/substrate lines.
- **The language gap is small and translatable away at ingestion time.** In the whole
  MVP tree: **one** functional `become:` (the view-resource proxy — we design around
  it), **zero** `thisContext`, **zero** runtime compilation. Namespaces flatten; pool
  constants and `##(…)` fold to literals; `??` rewrites to `ifNil:`.
- **The runtime gap is where the effort lives**, in this order:
  1. **C→Smalltalk callbacks (the wndproc)** — WINVM has none on Windows, but the macOS
     Cocoa GUI already proved the exact shape (`VmHandle::dispatch_callback`); we build
     the Win32 twin, with *re-entrancy* (nested entries) as a hard requirement.
  2. **Exceptions** (`on:do:`/`signal`) — designed (Phase W5) but unbuilt; a soft
     prerequisite from milestone UI-3 onward.
  3. **UTF-16 marshalling** — every `…W` API call and every text draw needs it.
  4. **Weak refs + finalization** — substitutable v1 (strong events + explicit `free`),
     wanted eventually for fidelity and leak-safety.
- **WINVM's x64 FFI already works** (`<primitive: FFI function: #MulDiv ret: #g args: #(g g g)>`
  through the `winkb` API database, plus COM vtable calls), and its external-memory
  `Alien` model **sidesteps the moving-GC problem by construction** — which is exactly
  the adaptation Dolphin needs, because Dolphin's FFI silently assumes a non-moving
  heap.
- **Recommended architecture:** the Cocoa two-tier pattern, Win32 edition — a **UI VM
  on its own OS thread that owns the HWNDs**, a **Rust-owned message pump** on that
  thread, one shared Rust wndproc reflecting every message into the image as a
  synchronous `dispatch_callback` entry, and Dolphin's `GuiInputState` green-process
  machinery **replaced**, not ported (WINVM has no green processes; nothing needs them
  once the pump is native).
- **Phasing:** six milestones (UI-0 substrate spike → UI-5 dialogs/polish), each ending
  in a runnable demo, with the translator and the compat kernel as the long poles.

---

## 1. Goal, context, and relationship to the dsfork VM effort

**Goal.** Give WINVM an optional, Windows-native GUI *written in Smalltalk*: real Win32
windows, controls, menus and GDI drawing driven from the image, structured as Dolphin's
Model–View–Presenter framework. It complements, not replaces, the existing WebView2
environment (`gui/`): the web environment remains the development UI; the MVP port is
the application-GUI story (and eventually an alternative shell).

**Why Dolphin's MVP specifically.**
- It is the best Windows-native Smalltalk GUI framework ever shipped: a thin, honest
  skin over `user32`/`gdi32`/`comctl32` with 25 years of edge-case hardening (DPI,
  themes, IsDialogMessage quirks, control subclassing) already encoded in Smalltalk.
- It is MIT-licensed and the sources sit in this workshop, in a fork whose author-side
  docs (`ffi.md`, `ffi-callbacks.md`, `ffi-async.md`, `gc.md` in `e:\dsfork`) already
  dissect the exact VM seams a port must reproduce.
- The minimal closure needs **zero COM/ActiveX and zero non-standard DLLs** — only
  `kernel32`/`user32`/`gdi32`/`comctl32`.

**Relationship to dsfork's own x64 work.** The fork is mid-flight porting *Dolphin's
C++ VM* to x64 (boot progresses through image load; FFI/callbacks still fragile). That
effort and this one are complementary consumers of the same stock D8 image sources.
This port takes none of the fork's VM code, but leans heavily on its analysis docs, and
its hard-won lessons transfer directly (e.g. *"null handles by explicit class marker,
not byte-shape heuristics"* — from the SaveImage64 LargeInteger-zeroing incident).

**Non-goals (v1).** The Dolphin IDE/tools (View Composer, browsers), Scintilla,
GDI+/GdiPlus, ActiveX/COM presentation, OLE drag-drop, RichEdit, per-monitor DPI,
themes beyond defaults, printing, and image snapshot/restore of live windows (WINVM is
rebuild-from-source; window state must be reconstructible by re-running startup code).

---

## 2. The source: what `e:\dsfork` actually is

| Fact | Finding |
|---|---|
| Identity | Dolphin Smalltalk 8 ("source-only corporate fork" of `dolphinsmalltalk/Dolphin`), branch `master`, 105 commits over baseline `c7f4b77` |
| Licence | MIT, © 2015 Object Arts (`LICENSE`) — no obstacle to porting or redistribution |
| Smalltalk-side divergence | **4 files**: `Boot.st` (+x64 cross-build chunk) and 3 × `Tools.Dolphin*Product.cls` (exclude ActiveX packages from product builds). **MVP is untouched upstream D8.** |
| Dialect | Namespaced D8: class files named `UI.View.cls`, namespaces are classes (`Kernel.Namespace` subclasses `Core`, `Kernel`, `External`, `OS`, `Graphics`, `UI`), resolution via class-def `imports:` |
| Source format | UTF-8 **with BOM**, CRLF, bang-chunk format (`!!` escaping); one class per `.cls`; `.pax` = package manifest + shared-pool class defs + **loose methods** |
| Scale | 3,164 `.cls` tree-wide; 972 under `MVP\` (~13k methods); base kernel `Base\` = 391 `.cls` / 94.4k lines |
| Fork docs worth rereading during the port | `ffi.md`, `ffi-dll-calls.md`, `ffi-callbacks.md`, `ffi-async.md`, `gc.md`, `modernize*.md`, `RESUME.md` |

**A trap for tooling, discovered early:** the `.pax` files are load-bearing source.
`Dolphin MVP Base.pax` carries the GUI constant pools (`OS.ButtonConstants`,
`OS.CommCtrlConstants`, …) *and* ~200 loose `OS.UserLibrary` FFI methods (BeginPaint,
SendMessage, …). An ingester that reads only `.cls` files silently loses the entire
User32 binding.

---

## 3. Architecture review: how Dolphin's GUI actually works

This section is the detailed review of the Smalltalk side. Paths are relative to
`e:\dsfork\Core\Object Arts\Dolphin\`.

### 3.1 Package topology — the minimal closure

Starting from `MVP\Base\Dolphin MVP Base.pax` + `MVP\Views\Common Controls\Dolphin
Common Controls.pax` and following `setPrerequisites:` transitively, the closure is 17
nodes ending at the base kernel (which has **no `.pax` at all** — "package Dolphin" is
the boot image; its source form is the 391 loose `Base\*.cls`):

| Package | ≈classes | Role |
|---|---|---|
| Dolphin MVP Base | 117 | View/ShellView/Shell/Presenter, layouts, Menu/Command, events plumbing, DPI, clipboard |
| Dolphin Common Controls | 46 | ListView, TreeView, TabView + their Win32 structs |
| Dolphin Basic Geometry | 9 | `Graphics.Point`, `Graphics.Rectangle` + POINT/RECT structs (geometry lives **here**, not in base) |
| Dolphin ControlViews Base | 4 | ControlView / ValueConvertingControlView abstract layer |
| Dolphin GDI Graphics | 82 | Canvas, Color, Font, Pen, Brush, Bitmap, Icon, Region, `OS.GDILibrary` |
| Dolphin Type Converters | 6 | model↔display converters |
| Dolphin Value Models | 11 | ValueHolder/adaptors/buffers |
| Dolphin List Models / Tree Models | 1 / 5 | observable list & tree |
| Dolphin List Presenter | 7 | ListPresenter + **ListBox/ComboBox views** |
| STx Filer Core + Literal Filer | 14 | STL/STB object serialization (view resources) |
| + 5 small support packages | ~15 | sort algorithms, conformant arrays, command-line parser, ComCtl library |
| **Base kernel (the image)** | ~391 | Object/collections/streams/exceptions/processes/FFI type system/SessionManager |

Totals: **~700 classes ≈ 165k lines**. Optional but adjacent: buttons and static text
are cheap leaves (2–4 classes each); **a plain `TextEdit` is not** — it lives in
`Dolphin Text Presenter`, dragging Common Dialogs + Find/Replace + MessageBox
(~33 classes across 4 packages). Scintilla (external DLL, 12.4k-line wrapper), GdiPlus
(141 files), and all COM/ActiveX are **outside** the closure.

### 3.2 The triad and the event system — semantics that must be exact

**Events (SASE).** `Core.Object>>when:send:to:` registers an `EventMessageSend`
(receiver held **weakly**) in a per-subject `EventsCollection`; storage is the global
`Object._EventsRegister` — a `WeakIdentityDictionary` keyed by subject — with a plain
`events` instance-variable override on `Core.Model`, `UI.Presenter`, `UI.ListModel`
(and views). `trigger:` answers the **last respondent's result** (events are used as
request/reply — e.g. `topShell trigger: #requestCommandPolicy:` passes a ValueHolder);
trigger-time arguments merge with registration-time arguments **winning from the left**
(`EventMessageSend>>forwardTo:withArguments:`); handlers added during a trigger don't
run in that trigger; observers die silently via `MourningWeakArray` + GC corpse
notification (`DeadObject.Current` + `elementsExpired:` pathologist callbacks). D8 has
**no `when:do:`** (block handlers leak; the API steers to selector sends). These exact
semantics — return value, merge order, reentrancy, idempotent registration — are relied
on by application code and must be ported verbatim.

**The triad in practice** (from `UI.Presenter`, 152 methods):
- `Presenter class>>on:` → `createView:` → `view:` does the handshake:
  `attachSubPresenterViews:` (match sub-presenters to same-**named** sub-views) →
  `onViewAvailable` → `connectView` (`view presenter: self`, `view model: self viewModel`).
- **The view observes the model directly** (`ValueConvertingControlView>>connectModel`
  subscribes `#valueChanged`; list views subscribe all five `ListModel` events) and
  refreshes itself without presenter involvement.
- **The view never triggers events off itself** — user gestures become
  `self presenter trigger: #selectionChanged` etc.; application code always observes
  the presenter.
- Round-trip guards are scattered and load-bearing: `settingValue` (ValueModel),
  comparison policies, `isStateRestoring`, `noEventsDo:`. Copy them, don't redesign.
- Unattached presenters hold `DeafObject.Current` (message-absorbing null object) as
  their view.

**Models.** `ValueModel` (`#value`/`#value:`/`#valueChanged`, pluggable
`comparisonPolicy`, default `SearchPolicy equality`), `ValueHolder`,
`ValueAspectAdaptor` (get = `subject perform: aspect`, put = `aspect,':'`),
`ValueBuffer`/`AspectBuffer` (dialog write-buffering via subject *copy* + replay-on-apply),
`ListModel` (a collection with events: `#listChanged`, `#item:addedAtIndex:`,
`#items:addedAtIndex:`, `#item:removedAtIndex:`, `#item:updatedAtIndex:`),
`TreeModel`/`VirtualTreeModel` (`#treeChanged:`, `#item:addedInParent:`,
`#item:removedFromParent:`, `#removingItem:`, …).

**Commands.** `CommandPolicy>>route:` builds a fresh route per query: source view's
presenter → its view → its model (if it `conformsToProtocol: #commandTarget`) → parent
presenter → … up to the shell. Enablement: `<commandQuery: #sel>` method annotations,
legacy registered handlers, then the fallback — *a target that merely `respondsTo:` the
command selector is enabled*. Toolbars re-validate at **idle time**
(`invalidateUserInterface`), menus on `WM_INITMENUPOPUP`. `CommandQuery` stores state
as Win32 `MFS_*` flags — Windows-isms reach even the "abstract" layer, which is fine
for a Windows-centric port.

### 3.3 The view layer and the VM↔window handshake

`UI.View` (6,050 lines, ~583 methods, direct subclass of `Core.Object`) wraps one HWND.

- **Registry:** HWND→View lives in `Kernel.InputState`'s `windows` — a **strong**
  `LookupTable` keyed by integer handle with a 1-slot `lastWindow` cache; entries are
  removed explicitly on `WM_NCDESTROY` (`winFinalize`), *not* by GC. Guarding is
  `Processor enableAsyncEvents: false/true`, not locks.
- **Window classes:** one Smalltalk-registered WNDCLASS `'DolphinWindow'` whose
  `lpfnWndProc` is **the VM's exported C `WndProc`** (`VM getWndProc`); native controls
  (BUTTON, EDIT, `SysListView32`, …) are created as their own class and **subclassed to
  the same VM WndProc** via `SetWindowLongPtr(GWLP_WNDPROC)` (old proc kept in
  `oldWndProc`, used by `ControlView>>defaultWindowProcessing:` → `CallWindowProc`).
- **Creation handshake:** `basicCreateWindow:` parks the view in
  `Processor activeProcess newWindow` (a real `Core.Process` ivar) → `CreateWindowExW`
  → the **VM's CBT hook** fires inside creation and calls
  `GuiInputState>>windowCreated:param:` → `subclassWindow:` → `attachHandle:` registers
  the map entry — so the View receives *every* message from `WM_NCCREATE` onward.
- **Dispatch:** the VM WndProc reflects into
  `InputState>>wndProc:message:wParam:lParam:cookie:` (re-entrant callback cookie), →
  `View>>dispatchMessage:wParam:lParam:` → a 1024-slot class-side `MessageMap`
  (message id + 1 → selector like `#wmPaint:wParam:lParam:`, ~68 messages mapped,
  extended per class via `registerMessageMappings:`). Handler answers an Integer ⇒
  that's the LRESULT; anything else ⇒ default processing (DefWindowProc /
  CallWindowProc). `WM_COMMAND`/`WM_NOTIFY`/`WM_CTLCOLOR*`/`WM_DRAWITEM`/`WM_*SCROLL`
  arrive at the parent and are **reflected to the child view** by handle lookup
  (`command:id:`, `nmNotify:` + per-class notification maps built as `##(…)` literals).
- **Painting:** `wmPaint:` → `ensureLayoutValid` (layout is validated lazily, on paint)
  → BeginPaint/EndPaint sandwich around `onPaintRequired: aPaintEvent` with an explicit
  EndPaint-on-error dance; `DoubleBufferedView` renders to a back `Bitmap` and blits.
- **Layout:** managers (`BorderLayout`, `FramingLayout` + `FramingConstraints`,
  `GridLayout`, `FlowLayout`, `ProportionalLayout` + `Splitter`, `CardLayout`) hold
  per-child arrangements keyed by view; invalidation bubbles up
  (`invalidateLayout`→`childLayoutInvalidated`), validation batches geometry with
  `Begin/Defer/EndDeferWindowPos` via a `LayoutContext`. **`subViews` is not stored** —
  child order *is* HWND z-order (walked via `GetWindow`), and layouts depend on it.
- **Interaction:** `interactor` delegate (default: the view itself; presenters are
  interposed), mouse capture with a **nested Smalltalk modal loop**
  (`CapturingInteractor>>captureMouse` runs `inputState loopWhile:`), `MouseTracker`
  drag protocol, native tab/focus via `WS_TABSTOP`/`WS_EX_CONTROLPARENT` +
  `IsDialogMessage`, accelerators as real HACCELs applied in per-message
  `preTranslateMessage:` walks by the pump.
- **Destruction is Windows-driven, not GC-driven:** `DestroyWindow` → `onDestroyed` →
  `WM_NCDESTROY` → `winFinalize` (deregister, unsubclass, release events) → dead views'
  `presenter`/`parentView`/`interactor` become `DeafObject.Current`.
- **DPI:** fully per-monitor-v2 (dpi-tagged creation rectangles, `FontSeries` per-DPI
  clones, `WM_DPICHANGED` rescale + menu-bar recreate) — but all scaling funnels
  through `View>>dpi`, `Font>>atDpi:`, `SystemMetrics forDpi:`, so a 96-DPI-only first
  cut is coherent.

**Control wrapper sizes** (for effort calibration): `ListView` 2,760 lines (virtual
mode, custom draw, column objects — a small application), `TextEdit` 1,698,
`Toolbar` 1,615, `TreeView` 1,440 (+ shared `IconicListAbstract` 1,069), `RichTextEdit`
1,121, `Menu` 943, `ListBox` 804, `TabView` 727, `ComboBox` 464, `StatusBar` 458,
`Slider` 512, `ScrollingDecorator` 409; buttons/statics are trivial (~100–300 each).

### 3.4 Message loop and session (what we will replace)

Dolphin's pump is pure Smalltalk on green processes (`Kernel.InputState`):
- A **Main** process runs `loopWhile:` — PeekMessage guard → GetMessage →
  `preTranslateMessage:` ancestor walk (TranslateAccelerator, IsDialogMessage) →
  Translate/Dispatch → drain deferred-action queues → sleep on `inputSemaphore`.
- An **Idler** process parks the single OS thread in
  `MsgWaitForMultipleObjectsEx(wakeupEvent, 15s, QS-mask, ALERTABLE)`.
- The VM samples the input queue (~60 ms, primitive 94) while CPU-bound green processes
  run, signalling the `InputSemaphore` so the UI preempts background work; a VM-created
  **message-only window** carries posted actions (`postToMessageQueue`) into foreign
  modal loops; a 100 ms `WM_ENTERIDLE` timer keeps green threads alive during OS menus
  and common dialogs.
- **Dialog modality is a process trick:** `DialogView>>runModalLoop` forks a
  *replacement Main* pump and blocks the opener on an `endModal` Semaphore (an
  in-process nested-loop variant `runModalInProcessLoop` also exists).
- `SessionManager` startup: VM interrupt #8 → `primaryStartup` (re-null stale handles,
  new input state, thunk reallocation) → `startUI` (re-register WNDCLASS, fresh window
  registry) → fork Main → app's `main` runs as the first deferred action
  (`DefaultShellSessionManager>>main` = `mainShellClass show`). The Idler auto-restarts
  a dead Main — a crashed handler never kills the pump.

WINVM has **no green processes**, so none of this machinery ports; §5.2 replaces it
with a native pump and keeps only the *policy* surface (window registry, deferred
actions, pre-translation, session lifecycle).

### 3.5 Dolphin's FFI and callbacks (what we will adapt, not port)

- Declarations are method annotations — `<stdcall: intptr SendMessageW handle uint32
  uintptr uintptr>` with body `^self invalidCall: _failureCode` running only on
  primitive failure; ~30-name type vocabulary; struct args by pointer
  (`WNDCLASS*`); `<virtual …>` for COM; `<overlap …>` for calls run on per-process OS
  worker threads (sockets, Sleep, MessageBoxIndirect — **not** common dialogs, which
  block by design). The MVP closure contains **~1,020 signatures** and ~60 struct
  classes with a `defineFields` offset DSL.
- **The unstated axiom: Dolphin's GC never moves object bodies.** The whole FFI passes
  interior body pointers and holds them across calls, callbacks, and worker threads;
  `newFixed:` (pinned heap) exists but is rarely used. This is the single biggest
  impedance mismatch with WINVM's moving GC — and WINVM's existing answer (external
  non-moving `Alien` buffers, never heap addresses) is the correct adaptation.
- **Callbacks:** the *image writes 16-byte x86 thunks* into RWX memory
  (`External.Callback>>allocateThunks`) targeting the VM export `GenericCallback`;
  re-entry uses setjmp cookies with LIFO exit enforcement. This is 32-bit-only by
  construction (the fork lists callbacks as broken on its x64 too). A port must make
  thunks a VM/host service. The **wndproc is not a per-window callback** — one VM
  WndProc for all windows, reflecting into a registered image-side Dispatcher.
- `GetLastError` is read *by image code after the failing call* — an implicit contract
  that the VM makes no intervening Win32 calls. WINVM should capture last-error in the
  FFI stub instead (safer; see S6).

### 3.6 Graphics layer

`Graphics.Canvas` (126 methods) is a thin HDC wrapper: acquisition via a two-message
`#dcSource` protocol (view GetDC / BeginPaint / display DC / bitmap memory DC),
selected-tool caching (keeps `pen`/`brush`/`font` wrappers alive against GC while
selected — reproduce this or get use-after-free of HGDIOBJs), full
line/shape/text/blit/region/transform protocol, per-DC `dpi`.
`Graphics.GraphicsTool` objects (Pen/Brush/Font/Bitmap/Icon/Region/ImageList) are
**virtual**: they hold a logical recipe (LOGPEN/LOGFONT/initializer) and lazily
`createHandle` on first use; mutation frees and re-realizes; startup `clearCached`
sweeps stale handles — a model that happens to fit WINVM's rebuild-from-source
philosophy perfectly. Cleanup is `Object>>finalize` → `free` via Dolphin's finalizer
queue (WINVM substitute: explicit free discipline v1, VM finalization later).
`Color` is abstract with `RGB`/`ARGB`/`IndexedColor` plus **dynamic** `SystemColor`
(`rgbCode` = `GetSysColor` on every read — theme changes propagate free);
`SysColorBrush` handles are shared and must never be deleted. `Font` carries an
explicit `dpi` + `FontSeries` per-DPI clone cache. Images delegate creation to
swappable `ImageInitializer`s (file/resource/bytes/blank); `ImageManager` maps images →
Win32 ImageList indices for ListView/TreeView. Text measurement has two non-mixable
regimes (GetTextExtentPoint32W ↔ TextOutW; DT_CALCRECT ↔ DrawTextExW), and **every text
API call converts via `asUtf16String`**.

### 3.7 View resources (STL) — and the decision not to need them

Every `resource_*_view` class-side method returns an **STL 6 literal array** —
`#(#'!STL' 6 … #{UI.STBViewProxy} #{UI.TextEdit} … #{Core.MessageSequence} …
#createWindow: …)` — a textual serialization of the view tree: 214 such resources in
MVP. Materialization runs through `Kernel.STLInFiler` → `UI.STBViewProxy`
(snapshot of *raw instance variables* + a replayable `MessageSequence` whose first send
is `createWindow:`) → **`become:`** swaps proxy→view → recursive child restore with
DPI rescaling. This is the **only functional `become:` in the entire MVP tree**, and it
sits together with `instVarAt:put:` reflection and the STB version-migration ladders
(`stbConvertFrom:` × 406).

**Crucially, resources are optional.** `Presenter>>view:` is public;
`ShellView new create` + `addSubView:name:` + layout managers build identical trees in
code (the framework itself and its test suite do exactly this). Only the convenience
constructors (`Presenter class>>show` naming a resource) and `ReferenceView` require
resources. **Decision: v1 bypasses STL entirely** — a translator subcommand transpiles
any needed `resource_*_view` into a builder method offline (killing `become:`,
`instVarAt:`, and the filer in one stroke); a runtime STL reader (~300–500 lines) is a
later fidelity option.

### 3.8 Dialect & kernel dependency inventory (with MVP-wide counts)

| Feature | Count (uses/files) | Port strategy |
|---|---|---|
| Namespaces + `imports:` | every class | flatten at translation; rename map for collisions |
| SharedPool constants (24 GUI pools + `OS.Win32Constants`) | pervasive, compile-time bound | **constant-fold at translation** (values live in the fork's sources) |
| `##(…)` compile-time eval | 816 / 307 | evaluate during translation |
| `#{X}` binding refs in code | 250 / 174 (mostly in STL arrays & imports) | resolve to flattened globals / `valueOrNil` → lookup thunk |
| `become:` | **1 functional site** (`STBViewProxy>>restoreView`) | designed around (§3.7) |
| `thisContext`, runtime `Compiler evaluate:` | **0** in core MVP | — |
| `??` nil-coalescing binary op | 104 | translator rewrite → `ifNil:` |
| `<stdcall:…>` + `_failureCode` | ~1,020 sigs | regenerate onto WINVM FFI (§5.4) |
| `<commandQuery: #sel>` annotations | 87 | translator lowers to generated registration methods |
| `expandMacros` (`<1p>` style) + `<<` | ~50 | small String shim |
| `trigger:` / `when:send:to:` | ~340 | compat-kernel event system (§5.5), exact semantics |
| `perform:` (MessageMap dispatch), `respondsTo:` (commands), `instVarAt:` (resources), `conformsToProtocol:` | structural | `perform:`/`respondsTo:` exist in WINVM; protocols degrade to respondsTo:/markers |
| `Semaphore`/`Delay`/`fork`/`Mutex` | ~15 sites (dialog modality, splash, drag-drop autoscroll, tooltip timing) | re-architect per site (§5.2); no green threads v1 |
| Weak collections | 2 sites in MVP proper (CommandDescriptionRegistry, SharedBitmaps) + the events registry | strong + explicit purge v1; VM weak refs later |
| Finalization | GraphicsTool/Canvas/External.Memory | explicit `free` + debug leak registry v1 |
| `Utf16String` traffic | 363 / 69 | real UTF-8↔UTF-16 converter (S3) |
| Class-instance vars | 6 non-trivial (e.g. `View class` `theme`) | per-class side table or metaclass ivars (small) |
| Exceptions: `ensure:` 56, `on:do:` modest (`InvalidFormat`, `Win32Error`, `OperationAborted`) | load-bearing | `ensure:`/`ifCurtailed:` exist today; `on:do:` needs Phase W5 |
| `LookupTable`/`IdentityDictionary`/`OrderedCollection`/streams | pervasive | WINVM has Dictionary/IdentityDictionary/OrderedCollection/streams; add `LookupTable` alias |
| `Fraction` (DPI ratios), `Duration` (`65 milliseconds`), `Locale` | modest | Fraction exists; Duration/Locale shims |

### 3.9 Scale summary

MVP Base 35k lines / 132 files; Views+Presenters+Models 72.6k / 271; GDI Graphics 82
classes / ~1,900 methods; base kernel 94.4k / 391 (we bridge, not port). The v1 subset
(§5.6) translates to roughly **40–60k lines across 150–200 classes**.

---

## 4. The target: WINVM today

From the WINVM-side survey (branch `windows-port`, all verified in code, not from the
MIGRATION status log):

| Capability | State today | Evidence |
|---|---|---|
| Call Win64 functions from the image | **Works.** `<primitive: FFI function: #Name ret: #g args: #(g g…)>`; resolution via the `winkb` SQLite API DB (≈18k functions, 46k COM methods, 97k constants, struct offsets); address cached in the descriptor | `src/runtime/ffi.rs`, `src/runtime/winkb.rs`, `src/codecache/ffi_stubs_x64.rs`, `world/tests/49_win32_ffi_tests.mst` |
| COM vtable calls | Works (Tier-2 by vtable index) | `ffi.rs:129-308` |
| Arg/return types | **`g` (int/pointer) and `f` (double) only**; struct-by-value refused; `g` return overflowing 61-bit smi is guest-fatal | `ffi.rs:425-434, 657-665` |
| Buffers under the moving GC | **Safe by construction**: indirect `Alien`s wrap malloc/VirtualAlloc memory with stable addresses; heap objects are never passed; FFI methods are interpreter-only with no mid-call safepoint | `src/runtime/alien.rs`, `ffi.rs:18-24` |
| UTF-16 | **Missing** — strings are UTF-8 byte strings; `…W` APIs need hand-built buffers today | `docs/WORLD.md:119-133` |
| C→Smalltalk callbacks | **Missing on Windows.** Proven shape on macOS: ObjC delegate IMPs → `VmHandle::dispatch_callback` → `perform:withArguments:` (prim 64), synchronous, per-entry fault recovery — but the `callback_active` guard **fails closed on re-entrancy** | `src/runtime/objc_delegate.rs`, `src/embed.rs:144-159, 764` |
| Synchronous Rust→Smalltalk entry | Works: `VmHandle::eval/exec/dispatch_callback`, each with an inline recovery slot (guest error → Rust `Err`, VM survives); must run on the VM-owning thread | `src/embed.rs:655-895` |
| Runtime class/method installation | Works at scale (whole 1,269-method world loads this way; browser live-compiles via `vm.exec`); ivar-shape changes need restart | `src/frontend/classdef.rs`, `gui/src/vm_host.rs:1091` |
| Exceptions | **Not built** — `ensure:`/`ifCurtailed:` + non-local return + terminal `error:` only; ANSI layer designed as Phase W5 ("zero new VM features") | `docs/WORLD.md:135-159` |
| Green processes / Delay / Semaphore | **None** (`fork` is a stub); concurrency = separate OS-thread worker VMs with message copying | `world/47_worker.mst` |
| Weak refs / finalization / `become:` | None / none / design-doc only | grep `src/` |
| Namespaces / pools / class-inst vars | None (flat globals, `<classVars:…>` pragma) | `src/frontend/classdef.rs` |
| GUI hosting precedent | Web GUI: Rust main thread owns HWND+WebView2, VM on worker thread, `PostMessageW` wakeups. Cocoa design: **UI VM pinned to the pump thread**, callbacks as top-level entries, persistent VM on background thread | `gui/src/shell/win.rs`, `docs/cocoa_gui_design.md` |
| Image model | Rebuild-from-source (`.mst` + SQLite source DB); no heap snapshots | `docs/WORLD.md:242-257` |

Environment note: `winkb` needs `E:\windows_api\windows_api.db` (овerridable via
`WINKB_DB`); without it, resolution degrades to a small dlsym probe list. The port
makes this DB (or a widened probe list) a hard dependency of the GUI feature.

---

## 5. The design

### 5.1 Strategy: three layers, translate-and-adapt

```
┌────────────────────────────────────────────────────────────┐
│ L3  Translated Dolphin MVP        (~150–200 classes, .mst) │  mechanical, via translator
│     View/Presenter/Model/Graphics/controls                 │
├────────────────────────────────────────────────────────────┤
│ L2  Compat kernel  "dolphin_compat" (~25–40 classes, .mst) │  hand-written once
│     events(SASE) · geometry merge · Utf16String · structs  │
│     LookupTable/DeafObject/Signal shims · UiSession        │
├────────────────────────────────────────────────────────────┤
│ L1  WINVM substrate additions      (Rust, ~2–4k lines)     │  VM/host work
│     win_gui host: pump + wndproc trampoline + CBT hook     │
│     re-entrant dispatch_callback · UTF-16 prims ·          │
│     last-error capture · (later: weak/finalize, thunks)    │
└────────────────────────────────────────────────────────────┘
```

Principles:
- **Translate, don't emulate.** Every dialect feature that can be resolved at
  translation time (namespaces, pools, `##()`, `??`, annotations) is resolved then —
  the running system stays plain WINVM Smalltalk with zero namespace/pool machinery.
- **Adapt at the OS boundary, port above it.** Dolphin's `External.*` layer assumed a
  non-moving heap and image-written thunks; we substitute WINVM's Alien/winkb FFI and a
  Rust callback service. Everything above (`View` up) ports with its logic intact.
- **Replace the process machinery, keep the policy.** No green processes: the pump,
  idle, modality, and "overlapped" concerns move into the Rust host / nested native
  pumps; `GuiInputState` shrinks to a registry + deferred-action policy object.
- **Fidelity where application code can see it** (event semantics, triad wiring,
  command routing, message-map behavior); **liberty where only the framework can see
  it** (how the pump is driven, how buffers are allocated).

### 5.2 Hosting model: UI VM + Rust pump (the Cocoa pattern, Win32 edition)

A new launcher mode / binary (working name `winvm-mvp`, or a `--native-gui` mode of
`winvm-gui`):

- **One dedicated OS thread runs both the UI VM and the message pump.** The VM boots
  the world + the dolphin_compat + MVP packages, runs the app's `UiSession startUp`
  (which creates shells via FFI), then **returns control to Rust**, which enters the
  pump: `GetMessageW → (native pre-translate) → TranslateMessage → DispatchMessageW`.
- **One shared Rust wndproc** (the `VM getWndProc` analogue) for the registered
  `"WinvmWindow"` class **and** for subclassed native controls. Every message becomes a
  synchronous `dispatch_callback` into the image:
  `UiSession wndProc: hwnd message: msg wParam: w lParam: l` → View lookup →
  `dispatchMessage:wParam:lParam:` (the ported Dolphin path). Integer answer ⇒ LRESULT;
  nil ⇒ Rust calls `DefWindowProcW`/`CallWindowProcW(oldWndProc)` itself (the Rust side
  keeps its own hwnd→oldWndProc map, mirroring what Dolphin's VM does).
- **Creation handshake, faithfully:** Rust installs a thread-local **WH_CBT hook**;
  `HCBT_CREATEWND` calls `UiSession windowCreated: hwnd` so the image can bind the
  "view under construction" (a `UiSession` slot replacing `Process>>newWindow`) before
  `WM_NCCREATE` arrives. (Fallback if the hook proves troublesome: `lpParam`-based
  association at `WM_NCCREATE` for our own class + post-creation `SetWindowLongPtr`
  subclassing for controls — Dolphin's own pre-CBT-era design.)
- **Native fast paths in the pump:** the image registers the active accelerator table
  (HACCEL) and the current "dialog-message" shell HWND via two host calls; Rust runs
  `TranslateAcceleratorW` + `IsDialogMessageW` natively per message — the semantics of
  Dolphin's `preTranslateMessage:` walk for its two dominant uses, without a VM entry
  per message. (A per-message image hook remains possible later for exotic views.)
- **Modality without green threads:** `DialogView>>showModal` is re-expressed on a host
  call `RunModalLoop(ownerHwnd, donePredicate)` — a nested native pump (Dolphin's own
  `runModalInProcessLoop` variant): disable owner, pump until the dialog's end-flag,
  re-enable in `ensure:`. OS-modal loops (menus, `MessageBoxW`, common dialogs) already
  nest their own pumps on our thread — wndproc callbacks keep arriving, so Dolphin
  windows keep painting, exactly as in Dolphin's plain-`stdcall` common-dialog path.
- **The web environment coexists.** The existing WebView2 shell keeps its own thread
  and windows; per-thread message queues make the two pumps independent. (Hazard noted:
  cross-thread `SendMessage` between the two window sets can deadlock if both block —
  the design rule is that the two GUIs communicate only via the existing channel
  mechanism, never via cross-thread SendMessage.)
- **Background work** uses the already-shipping worker-VM mechanism (message-copy
  channels), standing in for Dolphin's background green processes and overlapped calls.
  A blocking FFI call on the UI VM freezes the GUI — same as Dolphin's common dialogs;
  acceptable v1, revisit with a worker-thread call facility later.
- **Fault containment per message:** every `dispatch_callback` entry already owns a
  recovery slot — a DNU inside a handler abandons *that message* (Rust answers
  `DefWindowProc`) and the pump continues; this reproduces Dolphin's
  "crashed handler never kills the pump" property without the Idler/forkMain dance.
  A transcript/log hook reports the walkback text.

**Mapping of Dolphin's process-dependent sites (all ~15):** dialog modality → nested
native pump (above); splash timeout / tooltip delays / validation-debounce →
`SetTimer` + `#timerTick:` (already the Dolphin idiom for UI timers); drag-drop
autoscroll fork → timer; `Cursor wait showWhile:` → plain `ensure:`;
`Processor enableAsyncEvents:` → no-op shim (single-threaded, no async preemption);
`postToInputQueue`/`postToMessageQueue` → `PostMessageW` to a Rust message-only window
that re-enters `UiSession evaluateDeferredAction`.

### 5.3 WINVM substrate work items (L1 — the gating VM/host work)

| # | Item | Notes |
|---|---|---|
| **S1** | **Re-entrant `dispatch_callback`** | Replace the fail-closed `callback_active` flag with an entry **stack** (recovery slot per depth). Must support: callback → Smalltalk → FFI call (e.g. `CreateWindowExW`) → nested callback → … to arbitrary depth. This is *the* gating feature: `SendMessage` re-entry is normal Win32. Test: create-window-inside-WM_CREATE, 5 levels deep. |
| **S2** | **`win_gui` host module** | Window-class registration, shared wndproc, WH_CBT hook, pump, hwnd→oldWndProc map, message-only window for posted actions, `RunModalLoop`, accelerator/dialog-message registration calls, `PostQuitMessage`→clean shutdown. Rust, in `gui/` or a new crate; reuses `shell/win.rs` idioms. |
| **S3** | **UTF-16 support** | `String>>asUtf16Alien` / `Utf16String` compat class (external-memory backed), UTF-8↔UTF-16 conversion primitives (or pure-Smalltalk over Alien byte ops — measure; primitive preferred), and read-back (`utf16FromAlien:length:`). Needed by every `…W` call and every `TextOutW`. |
| **S4** | **FFI ergonomics for Win32** | (a) allow `g` args to accept nil→NULL and Alien→address (if not already); (b) LRESULT/LPARAM sign handling across the 61-bit smi boundary (sign-extended negatives round-trip; wider values → boxed or guest-fatal with clear message — audit `GetWindowLongPtr` uses); (c) winkb probe-list fallback widened for user32/gdi32/comctl32 staples so the GUI degrades gracefully without the DB. |
| **S5** | **Struct accessor generation** | Translator emits Alien-offset accessors from Dolphin's `defineFields` (offsets computed at translation time with MSVC packing rules; cross-checked against winkb where present). No runtime field machinery. |
| **S6** | **Last-error capture** | FFI stub captures `GetLastError` immediately after the call into a per-VM slot readable from the image (`UiSession lastError`), replacing Dolphin's fragile read-it-later contract. |
| **S7** | **Exceptions (Phase W5)** | The designed ANSI exception layer (`on:do:`, `signal`, Error/Warning/Notification). Soft prerequisite: required for `InvalidFormat` converter handling, `Win32Error signal:`, command abort — i.e. from UI-3 on. Independently valuable; per its design doc, zero new VM features. |
| **S8** | **`SetTimer` plumbing** | Nothing VM-side: WM_TIMER flows through S2's wndproc. Listed to record that `Delay` is *not* needed. |
| **S9** | *(later)* Weak refs + finalization queue | For events-registry fidelity, `CommandDescriptionRegistry`, GDI leak-safety. Two-queue split (finalize / bereavement) like Dolphin only if we adopt mourning semantics; a single finalize queue drained on the UI thread suffices for GDI. |
| **S10** | *(later)* Generic callback thunks | For `EnumWindows`/`EnumFontFamilies`-style APIs: a Rust thunk allocator binding cookie→block via `dispatch_callback`. v1 avoids these APIs (enumeration alternates exist: `GetWindow` walks; font enumeration deferred). |
| **S11** | *(later)* Worker-thread FFI ("overlapped") | For non-freezing long calls. v1: don't block the UI VM; use worker VMs for computation. |
| **S12** | *(later)* `RtlAddFunctionTable` for JIT frames | Open question from MIGRATION.md §5: a wndproc callback that faults *inside JIT-compiled Smalltalk* is the "foreign entry through JIT frames" case. v1 mitigation: run GUI dispatch interpreter-only (FFI methods already are; optionally pin UI-facing methods to tier-0) until unwind registration lands. |

### 5.4 Source ingestion: the `dolphin2mst` translator

A standalone tool (Rust, `tools/dolphin2mst/`), run offline; its output `.mst` files are
committed. Nothing of Dolphin's format survives into the runtime.

**Inputs:** package list (closure order from `.pax` `setPrerequisites:`, mirroring
`Boot.st`'s install order); the `.cls` files; the `.pax` inline pools + loose methods
(§2's trap); a hand-maintained **rename/patch table**.

**Pipeline per method:**
1. Chunk parsing (UTF-8 BOM, CRLF, `!!` unescaping, `''` strings) → class defs, GUIDs
   (dropped), comments, `methodsFor!` bodies, `categoriesForMethods` (kept as metadata
   comment or dropped).
2. Name resolution exactly as D8 does (class scope → `imports:` pools → owning
   namespace → its imports → global), then **flattening**: `UI.View` → `View`,
   `OS.Win32Constants.WM_PAINT` → literal `16rF`, `View.UIValidMask`-style qualified
   class-var refs → generated class-side accessors.
3. Folding: pool constants → numeric literals; `##(expr)` → evaluated literal (the
   translator embeds a tiny evaluator for the constant-expression subset: integers,
   `|`/`bitOr:`/`+`/`*`, pool refs, `_OffsetOf_*`); `classConstants:` → class-side
   methods or `<classVars:>` initialization.
4. Rewrites: `??` → `ifNil:`; `<commandQuery: #s>` → emitted registration in a
   generated `initializeCommandAnnotations` method; `<stdcall: ret Name types>` →
   WINVM `<primitive: FFI function: #Name ret: #X args: #(…)>` plus a thin wrapper for
   type adaptation (bool tests, handle boxing, Utf16 conversion, struct→alien address);
   `Utf8String`→`String`; `newFixed:`→Alien allocation.
5. Class-shape emission: `.mst` class definitions (`Super subclass: Name [ … ]` with
   `| ivars |`, `<classVars:…>`), **collision handling** via the rename table
   (known collisions with the existing world: `Point` and `Canvas` at minimum —
   policy: *merge* `Graphics.Point`/`Graphics.Rectangle` protocol into world classes as
   extensions where semantics agree; *rename* on conflict, e.g. `Graphics.Canvas` →
   `GdiCanvas`, since the world's `Canvas` is the web-GUI pixmap canvas). The
   translator emits a collision report; the table is reviewed by hand.
6. Resource transpilation (`--resources`): decode `resource_*_view` STL arrays
   structurally (prefix-code reader over the literal array) and emit equivalent
   builder methods (`createWindow:` replays become explicit `create` + property sends;
   named sub-view map preserved via `addSubView:name:`).
7. Per-method **patch overlay**: a directory of hand-edited method replacements keyed
   by `Class>>selector`, applied last — so re-running the translator against dsfork
   never clobbers manual fixes, and the manual-fix set stays visible and reviewable
   (the same discipline as the vendored-JASM `// MACVM:` markers).

**Explicitly not translated:** `MourningWeakArray` semantics (compat layer decides),
`Processor`/process code (rewritten per §5.2 site table), STB/STL filer classes,
SessionManager/InputState (replaced by hand-written `UiSession`), anything in the
excluded packages.

### 5.5 The compat kernel (`world/dolphin_compat/*.mst`, hand-written)

| Dolphin dependency | Provision on WINVM |
|---|---|
| Events: `when:send:to:(with:)`, `trigger:(with:with:)`, `removeEventsTriggeredFor:`, `EventsCollection`, `EventMessageSend/Sequence`, `noEventsDo:` | Ported near-verbatim from `Core.Object`/`Kernel.Events*` sources, **strong** storage v1: global `IdentityDictionary` registry + `events` ivar overrides on Model/Presenter/View/ListModel (same override triple as D8). Exact trigger-return, argument-merge, add-during-trigger, idempotency semantics. Leak containment: the framework's own `release`/`winFinalize` discipline already unsubscribes on destruction; weak upgrade rides S9. |
| `Model`, `SearchPolicy` singletons, `DeafObject`, `DeadObject` | Direct small ports. |
| Geometry `Point`/`Rectangle` | Extend the world's `Point`; add `Rectangle` (new); `asParameter` marshalling helpers target Alien-backed structs. |
| `LookupTable`, `IdentityDictionary` gaps, `SharedIdentityDictionary` | Alias/extend world Dictionary classes (`SharedIdentityDictionary` needs no mutex — single-threaded). |
| `Utf16String`, `External.Handle`-alikes | Utf16 = S3; handles travel as smis (61-bit covers user-mode pointers); an `ExternalHandle` wrapper only where nil/NULL distinction matters. |
| Structs (RECT, POINT, MSG, WNDCLASS, PAINTSTRUCT, NMHDR, SCROLLINFO, LVITEMW, …) | Generated Alien-offset classes (S5), allocated from external memory; `newBuffer` = malloc'd Alien + explicit `free` (paired in `ensure:` where transient). |
| `Signal`/`Win32Error`/`InvalidFormat`/`OperationAborted` | On W5 exceptions (S7); pre-W5 stubs raise terminal `error:` (acceptable for UI-0..2). |
| `GUID newUnique` | Trivial (counter or `CoCreateGuid` via FFI). |
| `expandMacros`/`<<`, `displayString` conventions | Small String/Stream extensions. |
| `propertyAt:`, `conformsToProtocol:`, `Cursor showWhile:` | Property table (strong, keyed IdentityDictionary); protocol test → `respondsTo:`-based shim; cursor scope via `ensure:`. |
| `UiSession` (replaces SessionManager/InputState) | Hand-written: window registry (strong `IdentityDictionary`, `lastWindow` cache, `windowCreated:`/`removeWindowAt:`), `wndProc:…` entry, deferred-action queue, accelerator/dialog registration with the host, startup/shutdown (`main` hook opens the initial shell), per-entry error reporting. Modeled on the D8 startup sequence in §3.4. |

### 5.6 Scope: the v1 cut

**In (UI-0..5):** View/ContainerView/ShellView/ControlView; BorderLayout,
ProportionalLayout+Splitter, FramingLayout, GridLayout, FlowLayout; Shell/Presenter/
Model/ValueModels/ListModel/TreeModel/TypeConverters; Command framework + Menu/MenuBar
+ accelerators; GdiCanvas + Pen/Brush/Font/Color/Bitmap/Icon(.ico files)/ImageList
subset; PushButton/CheckBox/RadioButton, StaticText/StaticRectangle, GroupBox,
TextEdit/MultilineTextEdit (accepting the dialogs tail it drags, minus Find/Replace),
ListBox, ComboBox, ListView (report + icon modes, static update mode first),
TreeView (#dynamic mode), TabView/CardContainer, ScrollBar/ScrollingDecorator,
StatusBar, ProgressBar, Slider, Tooltips (basic); DialogView + MessageBox (native
`MessageBoxW`/`TaskDialogIndirect` blocking) + `GetOpenFileNameW`-family common file
dialogs (plain blocking calls, as Dolphin does).

**Out (v1):** Scintilla, GdiPlus (and therefore `InternalIcon` PNG icons — v1 uses
`.ico` files via `LoadImage`), RichTextEdit, Toolbar (v1.5 — owner-draw heavy),
ListView virtual mode + custom draw (v1.5), MoenTree, drag-drop (OLE and internal),
clipboard beyond plain text, per-monitor DPI (fixed 96: stub `View>>dpi` ^96,
`Font>>atDpi:` ^self, `SystemMetrics forDpi:` ^default — the funnel points identified
in §3.6/§3.3), themes/UxTheme (`OS.ThemeLibrary` → `MockThemeLibrary` no-op twin, which
the tree already contains), COM anywhere, printing, Deprecated/Old Names packages,
image save of live windows.

### 5.7 What we deliberately do differently from Dolphin (and why)

| Dolphin | This port | Why |
|---|---|---|
| Smalltalk-side pump on green processes + VM input sampling | Rust-owned pump; VM entered per message | No green processes in WINVM; embed API entries are the proven, fault-recovered door |
| Image-written x86 callback thunks | Rust wndproc + (later) Rust thunk service | 64-bit, W^X, and the image should not write machine code |
| Non-moving heap bodies passed to APIs | External-memory Aliens only | WINVM GC moves objects; Alien discipline already exists and is tested |
| GC finalization frees GDI handles | Explicit `free` + debug leak registry; finalization later | No finalization in WINVM yet; explicit discipline is verifiable |
| Weak events registry + mourning arrays | Strong registry + framework teardown discipline | No weak refs yet; framework destruction paths already unsubscribe |
| `GetLastError` read later by image code | Captured in the FFI stub (S6) | Removes an implicit timing contract |
| STL resources + `become:` restore | Offline-transpiled builder methods | Kills the only `become:`, `instVarAt:` writes, and the filer dependency |
| Overlapped calls for MessageBox/sockets | Blocking calls v1; worker VMs for background work | Matches Dolphin's own common-dialog behavior; S11 later |

### 5.8 MULTIVM adaptation (amendment, 2026-07-22)

This section reconciles the port with the house multi-VM architecture
([docs/multi-smalltalk-worker.md](docs/multi-smalltalk-worker.md): share-nothing
primary/worker VMs, deep-copy messages, `send:…onReply:` continuations, star
topology) and with the Cocoa GUI's three-tier doctrine
([docs/cocoa_gui_design.md](docs/cocoa_gui_design.md)).

**Tier assignment.** The MVP GUI VM is a *fourth tier variant* — deliberately **not**
the Cocoa design's "dumb terminal" UI worker:

| Tier | Thread | State | Crash profile |
|---|---|---|---|
| **MVP UI VM** (this port) | its own pump thread, owns all HWNDs | **authoritative**: the triads (models/presenters/views) live here | per-message recovery for guest errors; supervisor respawn for hard fatals |
| Compute workers | background | none (share-nothing) | die + respawn (MULTIVM as designed) |
| Web-GUI primary | its existing worker thread | the dev environment | unchanged, coexists |

Why the MVP UI VM must be authoritative where the Cocoa UI worker is a snapshot
terminal: (a) **Win32 is synchronous** — `WM_NCCALCSIZE`, `WM_CTLCOLOR`, `WM_NOTIFY`
demand answers *now*, computed from real view/model state; a cross-thread snapshot
protocol cannot answer them and MVP's whole design (views observing live models) is
that state. (b) The Cocoa doctrine's two objections are both answered differently on
Windows: *crash safety* — guest errors (DNU/`error:`) no longer kill anything (the
per-entry recovery slot returns `Err`, the pump answers `DefWindowProc` and continues
— proven by the web GUI's DNU recovery); *hard* fatals (heap exhaustion, stack
overflow) tear down the GUI VM, and rebuild-from-source makes respawn legitimate: the
supervisor destroys surviving HWNDs (generation-checked, cf. `UI_VM_GENERATION` in
`src/embed.rs`), reboots the VM, and re-runs `UiSession startUp` — windows are
reconstructed the same way the world is. *Responsiveness* — a long doit freezes the
GUI exactly as it does in real Dolphin; the fix is the next paragraph, not a snapshot
protocol.

**Long work ships to workers, replies land as posted actions.** The MULTIVM
continuation model maps 1:1 onto the pump: the GUI VM's `inbox_wake` hook is
`PostMessageW` to the host's message-only window; a drain runs between messages as a
top-level entry and fires the `onReply:` continuation. So the MVP doctrine is:
command handlers that may exceed ~50 ms send `{selector. args-copy}` to a worker and
update the model in the continuation — Dolphin's `forkAt: userBackgroundPriority`
idiom, re-expressed as share-nothing RPC. `ProgressDialog` becomes: worker +
progress envelopes → posted actions updating a `ValueModel` (its Dolphin
implementation forks; it is rewritten, not translated).

**Nested VM entries are a designed requirement, not an accident.** The Cocoa design
proudly avoided nested-recovery machinery because AppKit lets the run loop own every
callback with the VM quiescent. **Win32 does not offer that option**: `CreateWindowExW`
delivers `WM_NCCREATE`/`WM_CREATE` synchronously *during* the FFI call;
`SetWindowText`, `DestroyWindow`, `SendMessage` all re-enter before returning. So the
callback path here is: `dispatch_callback` (message N) → Smalltalk handler → FFI call
→ wndproc → **nested** `dispatch_callback` — to arbitrary depth. S1 therefore
upgrades the embed layer from "fail closed on re-entry" (`callback_active`,
`src/embed.rs:780`) to a **per-entry recovery-slot stack** — detailed as sprint G0 in
[dolphin_ui_sprints.md](dolphin_ui_sprints.md). A guest fault at depth N unwinds to
entry N's slot only (LIFO — the same discipline Dolphin's callback cookies enforce),
answers that message's default, and the outer FFI call continues.

**No-green-threads site map (exhaustive).** Every `Process`/`Semaphore`/`Delay`/
`Mutex` dependency in the port scope and its replacement:

| Dolphin site | Mechanism there | Replacement here |
|---|---|---|
| `InputState` Main/Idler processes, `inputSemaphore`, `WakeupEvent`, prim 94 sampling | green scheduler + VM | **gone** — native pump (S2); no background green work to preempt |
| `DialogView>>runModalLoop` (`forkMainIfMain` + `endModal` Semaphore) | replacement main pump | host `RunModalLoop` nested native pump (LIFO modality — see divergence note in §7) |
| `CapturingInteractor>>captureMouse` (`loopWhile:` nested loop) | nested Smalltalk pump | same host `RunModalLoop` primitive with a capture predicate; tracking state machine unchanged |
| `postToInputQueue` / `postToMessageQueue` | SharedQueue + posted WM_USER | `PostMessageW` to host message-only window → `UiSession evaluateDeferredAction` |
| Splash timeout, tooltip dwell, validation debounce, MessageBubble | `Delay`/forked waiters | `SetTimer` + `#timerTick:` (already the Dolphin idiom for UI timers) |
| Drag-drop autoscroll `forkAt:` | background process | `SetTimer` tick |
| `ProgressDialog` forked computation | green process | worker VM + progress envelopes → posted actions (rewritten) |
| Idle-time `invalidateUserInterface` revalidation | queue-empty detection in Smalltalk pump | pump-empty hook: `PeekMessage` miss → `UiSession onIdle` (S2) |
| `Processor enableAsyncEvents:` critical sections | interrupt masking | no-op shim (single-threaded VM, no async preemption) |
| `SharedIdentityDictionary` / `Mutex` (RichText converter) | thread safety | plain dictionary / no-op (RichText out of scope v1) |
| `Cursor wait showWhile:` | dynamic scope | `ensure:` |
| Overlapped `<overlap stdcall:>` calls | per-process OS worker threads | none v1 (blocking, as Dolphin's own common dialogs); S11 later |

### 5.9 `become:`-free audit (amendment)

The no-`become:` philosophy holds with **zero** runtime `become:` in the ported
system:

| Dolphin `become:` exposure | Status in the port |
|---|---|
| `STBViewProxy>>restoreView` (`self become: newView`) — the only functional site in MVP | **eliminated** — STL resources transpiled offline to builder methods (§3.7/§5.4.6) |
| `STxProxy>>stbFixup:` (`become: self value`) in the filer | filer **not ported** |
| Dev-time class reshaping (Dolphin mutates live instances on redefinition) | n/a — translation is offline; at runtime WINVM's existing rule applies (method/classVar changes live, ivar-shape changes need restart — same as the web GUI today) |
| `View>>recreate` (style changes) | no `become:` involved — Dolphin destroys and recreates the HWND on the *same* view object; ports as-is |
| Identity across VM boundaries (workers) | never arises — MULTIVM copies by design |

---

## 6. Phases

Each milestone ends with a runnable demo on this machine. Sizing is relative to
completed WINVM phases (calendar time in this project has compressed absurdly; the
honest unit is "comparable effort to phase X").

> **Execution plan:** the milestones below are decomposed into concrete sprints
> (Phase **G**, house SPRINTS.md format, with the VM-change specs S1/W5/etc. written
> against the real code) in **[dolphin_ui_sprints.md](dolphin_ui_sprints.md)**.
> Milestone↔sprint mapping: UI-0 ≈ G0–G1, UI-1 ≈ G2–G3, UI-2 ≈ G4, UI-3 ≈ G5 (+W5),
> UI-4 ≈ G6, UI-5 ≈ G7.

- **UI-0 — Substrate spike** *(comparable to Phase 2/M2)*. S1 (re-entrant
  dispatch_callback) + S2 skeleton + S3 minimal (asUtf16Alien). No Dolphin code. Demo:
  from the image, register class, `CreateWindowExW` a top-level window with a native
  BUTTON child, paint "hello" via `TextOutW` in a `wmPaint:` handler, button click
  round-trips to the Transcript, close cleanly. **Acceptance gate: a window created
  from inside a `WM_CREATE` handler, 5 callback levels deep, plus a DNU inside a
  handler that the pump survives.**
- **UI-1 — Translator + compat kernel** *(the long pole; comparable to Phase 3's
  scope)*. `dolphin2mst` end-to-end on Basic Geometry + the struct/FFI surface; compat
  events with a ported subset of Dolphin's own event unit tests; `UiSession`. Demo:
  translated `Graphics.Point`/`Rectangle` tests green; events semantics tests green
  (trigger return value, argument merge, add-during-trigger).
- **UI-2 — View core vertical slice**. Translated View/ContainerView/ShellView +
  BorderLayout + PushButton/StaticText + GdiCanvas basics. Demo: a `ShellView` built in
  code — caption, menu-less, a bordered layout with label + two buttons, live resize,
  clean destroy; `purgeDeadWindows`-style hygiene verified.
- **UI-3 — MVP proper** *(wants S7/W5 exceptions)*. Presenter/Shell/Model/ValueModels/
  TypeConverters/Command routing + Menu/MenuBar + accelerators + TextEdit + ListBox.
  Demo: a real MVP app — a two-field dialog-style shell over a ValueModel-backed model
  object with menu commands, enablement via `queryCommand:`, and a TypeConverter
  rejecting bad input.
- **UI-4 — Common controls**. ListView (report), TreeView (dynamic), TabView,
  StatusBar, ScrollingDecorator, Splitter+ProportionalLayout, ImageList/.ico icons.
  Demo: a two-pane class-browser-shaped shell (tree left, list right, text bottom) —
  the classic Dolphin idiom, over live world reflection data.
- **UI-5 — Dialogs & polish**. DialogView modality on `RunModalLoop`, MessageBox,
  common file dialogs, Prompter-family (transpiled resources or builders), clipboard
  text, keyboard navigation audit (IsDialogMessage edge cases from §3.3).
  Demo: modal prompter editing a value with OK/Cancel buffering (`AspectBuffer`
  semantics), opened from the UI-4 browser.
- **UI-6+ — Fidelity backlog** (ordered by value): Toolbar + idle-time command
  revalidation; ListView virtual mode/custom draw; weak events + finalization (S9);
  runtime STL reader; per-monitor DPI (unstub the three funnels); themes; RichEdit;
  generic callback thunks (S10); worker-thread FFI (S11); drag-drop.

Corpus checkpoints: UI-2 ≈ 30–40 translated classes; UI-3 ≈ +60; UI-4 ≈ +40;
UI-5 ≈ +30 — total in the 150–200 class band predicted in §0.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Nested callback re-entry destabilizes the VM** (interpreter state, recovery slots, GC at unexpected depth) | High — it's the foundation | UI-0 acceptance gate is exactly this, tested to depth before any Dolphin code lands; FFI-methods-stay-interpreted invariant already removes safepoint-mid-call; JIT-frame unwind risk parked via S12 (interpreter-only dispatch v1) |
| **Callback faults inside JIT frames** (wndproc → compiled method → AV) | Medium | v1: UI entry points effectively tier-0 (FFI methods never compile; audit hot handlers); revisit with `RtlAddFunctionTable` (MIGRATION §5 already anticipates this) |
| **Translator semantic drift** (a folded constant wrong, a rewrite subtly off) | High-frequency, low-individual-cost | Port Dolphin's own MVP/Graphics unit tests alongside (the tree ships them); golden-file translator tests; the patch-overlay keeps hand fixes reviewable |
| **Event-system infidelity** breaks app code invisibly | High | Port `Kernel.Events*` logic verbatim + a dedicated semantics test file (return values, merges, reentrancy) written from §3.2's findings before any consumer lands |
| **Strong events + no finalization leaks** (observers, GDI handles) | Medium, slow burn | Framework teardown paths already unsubscribe/free; add a debug leak registry (counts by class) surfaced in the transcript; S9 closes it properly |
| **smi width vs LPARAM/LRESULT extremes** | Low-medium | 61-bit smis cover user-mode pointers; audit sign-extension at the stub (S4b); tests with negative LPARAMs (mouse coords are packed signed shorts!) |
| **Pump coexistence with the WebView2 shell** (cross-thread SendMessage deadlock) | Low | Design rule: channels only between the two GUIs; no cross-thread window ownership |
| **winkb DB absent on a target machine** | Low | S4c widened probe list for the core three DLLs; clear startup diagnostic |
| **Scope creep toward the IDE** | Medium | §1 non-goals; the web environment remains the dev UI |
| **TextEdit/ListView complexity underestimated** (1.7k/2.8k lines each) | Medium | They are UI-3/UI-4 tail items with static modes first; virtual/custom-draw explicitly deferred |
| **WM_PAINT error storm pre-W5**: a failing paint handler unwinds past `EndPaint`, the update region never validates, Windows re-sends WM_PAINT forever | High until W5 | Rust wndproc backstop: if the image entry for WM_PAINT returns via the recovery path, host calls `ValidateRect` before answering (G1); post-W5 the ported `basicPaint:` `on: Error` dance restores fidelity |
| **UTF-16 code-unit index math**: `EM_GETSEL`/selection ranges/`BCM_GETNOTE` count UTF-16 units; world strings are UTF-8 byte-indexed | Medium, subtle | `Utf16String` compat class carries code-unit length; conversions only at control boundaries; translator flags String index arithmetic for review (the WORLD.md §6 rule, reused) |
| **Modality divergence**: Dolphin's forked-main pumps allow dismissing stacked dialogs in any order; nested native pumps are LIFO-only | Low (UX nuance) | Accepted divergence — LIFO modality is standard Win32 behavior; documented in §5.8 |
| **CBT hook sees every window on the thread** (incl. internals of native dialogs/menus) | Low | Hook associates only when `UiSession` has a pending view-under-construction; all other creations pass through untouched (Dolphin's hook behaves the same) |
| **GUI-VM respawn vs. live HWNDs** after a hard fatal | Medium | Generation-stamped dispatch: wndproc checks the VM generation, answers `DefWindowProc` for stale windows; supervisor destroys survivors then re-runs `UiSession startUp` (§5.8) |

---

## 8. Open questions (updated after the MULTIVM/sprint pass)

Decided in [dolphin_ui_sprints.md](dolphin_ui_sprints.md):
- ~~W5 exceptions timing~~ → **decided**: W5 runs as its own sprint, parallel after
  G1, gating G5 (UI-3). Pre-W5 milestones use no exception-dependent paths.
- ~~Launcher shape~~ → **decided**: a dedicated bin (working name `winvm-mvp`),
  following the `macvm-cocoa` precedent — one crate arm, shared world/bridge code.

Still wanted before G2 (first translation sprint):
1. **Naming/collision policy sign-off** (§5.4.5): merge `Point`/`Rectangle` into world
   classes vs. keep a parallel geometry; `GdiCanvas` rename OK?
2. **TextEdit tail**: accept the ~33-class dialogs tail in UI-3/G5, or split a
   "TextEdit-lite" (no find/replace, no document presenter) via the patch overlay?
3. **Where the code lives**: translated output committed under `world/mvp/` in this
   repo (recommended — versioned like the rest of the world), or a sibling repo?
4. **DPI ambition**: is 96-DPI v1 acceptable on this machine's monitors, or is
   system-DPI scaling (one global factor, still not per-monitor) worth pulling into
   UI-2/G4?

---

## Appendix A — Dolphin package dispositions

| Package | Disposition |
|---|---|
| Dolphin Basic Geometry | Translate (merge/extend world Point; new Rectangle) — UI-1 |
| Dolphin GDI Graphics | Translate subset (Canvas→GdiCanvas, tools, Color, Font, Bitmap/Icon/ImageList; skip palette/print/metafile edges) — UI-2 |
| Dolphin MVP Base | Translate core (View machinery, layouts, Menu, Command, Shell/Presenter; skip drag-drop, DPI internals→stubs, splash) — UI-2/3 |
| Dolphin ControlViews Base / Type Converters / Value Models / List+Tree Models | Translate whole — UI-3 |
| Dolphin List Presenter / Text Presenter (trimmed) / Boolean / Choice / Number presenters | Translate — UI-3 |
| Dolphin Common Controls (ListView/TreeView/TabView) | Translate, static modes first — UI-4 |
| Scrollbars / Splitter / Cards / Static* / Buttons / Slider / SpinButton / Progress / Tooltips | Translate — UI-2..4 as scheduled |
| Base Dialogs / Message Box / Common Dialogs (file, minus print) | Translate + rework modality — UI-5 |
| STx/Literal Filer | **Do not port**; translator-side STL decoding only |
| Control Bars (Toolbar/StatusBar) | StatusBar UI-4; Toolbar UI-6 |
| Scintilla, GdiPlus, Metafiles, ActiveX/COM, DolphinSure, Lagoon, IDE/Tools, Deprecated/Old Names | **Out** |
| Base kernel (`Base\*.cls`) | **Not ported** — bridged by dolphin_compat (§5.5); the ~200 UserLibrary loose methods + OS.GDILibrary regenerate through the translator's FFI rewriter |

## Appendix B — Dolphin VM-service contract → WINVM provision

| Dolphin VM service | WINVM provision |
|---|---|
| Exported C `WndProc` shared by all windows | Rust wndproc in `win_gui` (S2) |
| CBT hook → `windowCreated:param:` + `Process>>newWindow` slot | Rust WH_CBT → `UiSession windowCreated:` + session slot (S2) |
| Callback cookies, setjmp/LIFO exits, prims 104/107/114/117 | Not replicated: each entry is a normal nested `dispatch_callback` that returns; recovery-slot stack (S1) |
| Image-written x86 thunks + `GenericCallback` | Deferred Rust thunk service (S10); v1 needs none |
| Input sampling (prim 94), InputSemaphore, WakeupEvent, MsgWait idler | Not needed: native pump; no green processes to preempt |
| Message-only window for posted actions | Rust message-only window → `UiSession evaluateDeferredAction` (S2) |
| One-shot timer prim 100 / TimingSemaphore (`Delay`) | Not needed v1: UI timing via `SetTimer`; no `Delay` |
| FFI prims 96/80/48 + descriptor marshalling | Existing `dispatch_ffi_primitive` + winkb + trampolines; translator regenerates signatures |
| `GetLastError` read-later contract | Stub-side capture (S6) |
| Finalize/bereavement queues + semaphores | Deferred (S9); v1 explicit free + strong registries |
| `newFixed:` pinned heap | External-memory Aliens (already shipping) |
| Interrupt vector (onStartup, idlePanic, userBreak, …) | Host-driven lifecycle calls into `UiSession`; guest-fatal recovery already covers fault paths |
| Image save handle-nulling | N/A — rebuild-from-source; handles never persist |

## Appendix C — effort calibration numbers (measured, not guessed)

- Dolphin MVP tree: 972 `.cls`, ~13.0k methods; minimal closure ~700 classes /
  ~165k lines; v1 translated target 150–200 classes / 40–60k lines.
- Biggest single items: `UI.View` 6,050 lines / ~583 methods; `ListView` 2,760;
  `TextEdit` 1,698; `Toolbar` 1,615 (deferred); `TreeView` 1,440; GDI Graphics
  ~1,900 methods across 82 classes (subset ported).
- FFI surface to regenerate: ~1,020 `<stdcall:>` signatures (≈200 of them loose methods
  in `.pax` files), ~60 struct classes, 24 GUI constant pools + `OS.Win32Constants`.
- Event system to reimplement: 4 kernel classes + 15 `Object` methods (§3.2), ~340
  usage sites that depend on its exact semantics.
- WINVM side today: 730+ lib tests green; FFI/COM tests passing against live Win32;
  `dispatch_callback`/embed recovery proven on macOS Cocoa and in the Windows web GUI.
