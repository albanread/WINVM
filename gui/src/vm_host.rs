//! The GUI↔VM threading scaffold from `../../docs/SPEC.md` §16.1 (decision
//! A18): "AppKit owns the main thread; the VM runs on a dedicated worker
//! thread... GUI→VM requests (doits, mirror queries, accepts) are queued
//! over a channel and executed between interpreter turns; VM→GUI output
//! (transcript, rendered fragments) returns via a channel drained on the
//! main thread into `evaluateJavaScript`."
//!
//! There is no real VM to embed yet (`VmHandle`, SPEC §16.2, needs core
//! eval — G2 per `../PLAN.md` §6) — so `worker_loop` below stands in for
//! `VmHandle::eval`, doing exactly what `main.rs`'s inline G1 stub used to
//! do (echo the doit's source text). What's real here is the *threading*:
//! this is a genuine `std::thread::spawn` OS thread, talking to the main
//! thread only through the two channels below, with no shared mutable
//! state and no direct cross-thread AppKit/WebKit calls — the actual shape
//! G2 will need, just with a stub instead of `VmHandle` on the worker end.
//! Only `doit` is routed here: toolbar/navigate are pure UI/file-loading
//! concerns (`NavState`), not VM work, so per SPEC §16.1 they stay on the
//! main thread untouched.
//!
//! Three things this scaffold is already shaped for, even though nothing
//! exercises them yet (no real VM, no smappl-built widgets):
//!
//! **Bulk payloads (RGBA pixel blocks, large text)** — `VmResponse`'s
//! variants own their data outright (`String`, and future `Vec<u8>`/
//! `Arc<[u8]>`), so a large buffer crossing `response_tx.send(..)` is a
//! move, not a copy. The open question is *delivery into the DOM*, not the
//! channel: `evaluateJavaScript` takes a JS source string, so a naive
//! approach (base64-inline a pixel buffer into that string) works but
//! doesn't scale to big/frequent images. The planned mechanism instead: a
//! `WKURLSchemeHandler` registered on the webview's configuration serving
//! `macvm-pixels://<id>` by reading straight from a Rust-side buffer — the
//! page just does `<img src="macvm-pixels://42">` and re-fetches on
//! update, no JS-string-serialization involved. (A temp-file-per-frame,
//! matching the existing `.rendered/` pattern, is the fallback if a custom
//! scheme handler proves awkward to wire through this bridge — strictly
//! worse for high-frequency updates but zero new ObjC surface.)
//!
//! **Slow UI thread vs. a fast producer** — not every response should
//! queue. Discrete, one-off events (a transcript line, a single doit
//! result) belong on the ordered `mpsc` channel below: `drain_responses`
//! already applies a whole backlog in one wake, so a burst just gets
//! batched, never dropped or reordered. But a *continuous* stream (a
//! redrawing pixel buffer, a live status string during a long computation)
//! must not queue — if the UI thread falls behind, stale frames piling up
//! and rendering in sequence is wasted work at best. That case wants
//! "latest value wins," which is exactly what [`LatestSlot`] below
//! provides: publishing a new value drops whatever was there, unread or
//! not. Route continuous updates through a `LatestSlot` keyed by widget id
//! (a `Mutex<HashMap<Id, LatestSlot<T>>>`, once there's a real per-widget
//! id space to key on), and discrete ones through the `mpsc` channel — two
//! disciplines, chosen per response kind, not one queue trying to serve
//! both.
//!
//! **Routing a click to the right closure, and locking** — once smappl
//! (`../smappl.md`) is real, each button/widget captures its own action
//! closure at build time (mirroring Strongtalk's own `Button
//! withImage:action:`). The natural home for that closure table is the
//! worker thread itself, *not* a `Mutex` shared with the UI thread: the UI
//! thread never touches closures directly — a click just carries the
//! widget's opaque id (`data-node-id`, same idea `docs/APPS.md` §7's
//! `ToolRegistry` already uses) through `smtk.js` → `on_script_message` →
//! a new `VmRequest` variant (e.g. `InvokeHandler { id }`) across the
//! *existing* request channel. The VM thread looks the id up in its own,
//! entirely private `HashMap<Id, Handler>` and calls it — no lock needed
//! for that table at all, because only one thread ever touches it. This is
//! "share memory by communicating," not "share memory, protect it with a
//! mutex" — the request channel is already the synchronization.

#![allow(dead_code)] // LatestSlot has no producer yet — no bulk/continuous
                     // response exists until a real VM or smappl widget
                     // needs one; see this module's doc comment.

use crate::browser_render::{self, BrowserSelection, SourceEditTarget};
use crate::shell::Waker;
use macvm::embed::{GameCommand, GameSink, TranscriptSink, VmHandle, VmLiveStats, VmMetrics};
use macvm_mock_vm::MockWorld;
pub use macvm_mock_vm::Side;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The two top-level doIts the `.mst` world runs at load (`Transcript` bind +
/// `Character initTable`). S22-B moves these into the image; until then they
/// are replayed explicitly by [`crate::world_boot::load_world_from_image`].
pub(crate) const WORLD_DOITS: &[&str] = &[
    "Transcript := TranscriptStream new.",
    "Character initTable.",
];

/// The package list(s) this GUI's VMs boot: just the base world (M4,
/// `docs/package_aware_editing_design.md` §4.5) — the WKWebView GUI has no
/// Cocoa-GUI-specific role, so it never needs `cocoaui`'s classes live.
pub(crate) const WORLD_LISTS: &[&str] = &["world"];

/// GUI → VM. `Doit` is the only one with a real (stub) handler; the
/// `BrowserSelect*` variants drive the class browser view
/// (`browser_render.rs`) against `macvm-mock-vm`'s invented class world —
/// see that crate's doc comment for why this is scaffolding, not the real
/// mirror layer. Real mirror queries/accepts join `Doit` once G2 gives them
/// something real to ask the VM thread for.
pub enum VmRequest {
    Doit {
        code: String,
    },
    /// One frame tick of a running game loop (`docs/gamepane_design.md` M4):
    /// run `GamePane stepWithKeys: keys`. The GUI's frame timer submits this
    /// once per tick to an IDLE worker (single-outstanding via
    /// [`VmHost::is_idle`]), so the step block runs as a fresh top-level
    /// request, never nested inside a still-running eval.
    GameStep {
        keys: i64,
    },
    /// Worker envelopes are waiting in this (primary) VM's inbox
    /// (docs/multi-smalltalk-worker.md §3.1, M3): run `Worker dispatchInbox.`
    /// so continuations/handlers fire. Queued — coalesced — by the inbox wake
    /// hook FROM a worker thread the moment its send lands; the primary VM
    /// thread's ordinary `requests.recv()` sleep is what it interrupts, so
    /// delivery is event-driven with zero polling. Deliberately silent on
    /// success (a per-wake `> …` echo would flood the transcript).
    WorkerInbox,
    /// Text editor (`docs/editor_design.md` M3): open a session on `text` (a
    /// class's source), replacing any prior one; answers a full-document
    /// [`VmResponse::EditorView`] for the initial render.
    EditorOpen {
        text: String,
    },
    /// One EDITING key for the current session — a printable string, `"Enter"`,
    /// `"Backspace"`, or `"Delete"` — together with `at`, the textarea's real
    /// 1-based caret offset at the moment of the key. The VM moves its cursor
    /// there before applying the key, so the edit lands where the user is even
    /// though navigation (mouse, arrows, selection) never round-trips through
    /// the VM. Like `GameStep`, a fresh top-level request per key. Answers an
    /// [`VmResponse::EditorView`].
    EditorKey {
        key: String,
        at: i64,
    },
    /// An editor command by name (`"undo"`, `"redo"`; accept/close are M4/M6).
    /// Answers a full-document [`VmResponse::EditorView`].
    EditorCommand {
        name: String,
    },
    /// Save (accept) the current editor session: syntax-check the buffer by
    /// compiling it into the running VM, then — only if that succeeds —
    /// persist the class to the image, writing just the methods that actually
    /// changed. Reports the outcome on the transcript. (docs/editor_design.md
    /// M4, a first cut: whole-class buffer, method-granular commit.)
    EditorAccept,
    /// A `<smappl visual="CODE">` on the current page (`../smappl.md`): render
    /// the `visual=` expression to an HTML fragment (`VmHandle::render_fragment`,
    /// D-G5). `id` is the placeholder span's `data-widget-id`; the worker
    /// answers [`VmResponse::SmapplFragment`] with the same `id` so `main.rs`
    /// can swap the right span. A shape that can't be built yet (needs the
    /// Phase-W mirror stack) renders nothing — the placeholder box stays.
    SmapplRender {
        id: String,
        code: String,
    },
    /// A rendered smappl widget was clicked (`smtk.js`): run its action
    /// closure. `action_id` is the opaque id the widget carries in
    /// `data-widget-action`, stored image-side in `SmapplRegistry`; the
    /// worker runs `SmapplRegistry fire: action_id`, whose transcript/effects
    /// flow back over the normal channels (`../PLAN.md` G1).
    SmapplAction {
        action_id: String,
    },
    /// Cmd+S in a ClassOutliner source editor (`smtk.js`): accept an edited
    /// method. The worker **versions** it back into image_store
    /// (`set_method_source` → a new `method_versions` row, never an overwrite)
    /// AND live-compiles it into the running VM — the same proven path the
    /// class browser's `BrowserSaveSource` uses.
    SmapplAccept {
        cls: String,
        side: String,
        sel: String,
        text: String,
        /// The `data-widget-id` of the outliner being edited — left out of the
        /// post-accept live refresh so its editor isn't yanked out mid-edit.
        widget_id: String,
    },
    /// Outliner "＋ new method": create a brand-new method on `cls`/`side` from
    /// the typed template source — the selector is parsed from its message
    /// pattern (`image_store::mst::parse_method_selector`). Versions it into the
    /// image (`create_or_reopen_method`, a fresh `method_versions` row) AND
    /// live-compiles it, then refreshes every open outliner so it appears. The
    /// create-side analog of [`SmapplAccept`] (which only edits existing ones).
    SmapplNewMethod {
        cls: String,
        side: String,
        text: String,
    },
    /// Outliner "＋ new class": create a brand-new class (plus any methods in the
    /// definition) from the typed class-definition source, version it into the
    /// image (`create_or_reopen_class`), live-compile it, and refresh every open
    /// outliner so it appears.
    SmapplNewClass {
        text: String,
    },
    /// Outliner "＋ add instance/class variable": union a new variable NAME into
    /// the class's stored shell (image `reimport_class_shell`). A class variable
    /// also goes live immediately (reopening a class to append a `<classVars: …>`
    /// entry is not a shape change), so the outliner shows it at once; an
    /// instance variable is a shape change the running VM's fixed object layout
    /// can't apply, so it persists and takes effect on the next VM restart.
    SmapplAddVar {
        cls: String,
        is_class_var: bool,
        name: String,
    },
    /// Drill down: a class name in a hierarchy outliner was clicked. Replace
    /// that widget (`widget_id`) with the class's method browser (a
    /// `ClassOutliner`, with editable source), carrying a back link to `root`'s
    /// hierarchy.
    SmapplOpenClass {
        cls: String,
        root: String,
        widget_id: String,
    },
    /// The back link in a drilled-down ClassOutliner: re-open `root`'s
    /// hierarchy in that widget.
    SmapplOpenHierarchy {
        root: String,
        widget_id: String,
    },
    /// A find-tool query (docs/APPS.md §5.4). Definition + Implementors are
    /// answered directly from image_store (SQL, no VM round trip); Senders still
    /// needs the VM's IC-table scan. Result HTML goes back for the find page's
    /// results div.
    Find {
        tool: String,
        query: String,
    },
    /// The option list for a find view's combobox (`<datalist>`) — class names
    /// for Find-Definition, selectors for Implementors/Senders. Requested once
    /// when a find page loads; answered from image_store.
    FindOptions {
        tool: String,
    },
    /// File ▸ "Save World to Files" — write the image back over `world/*.mst`
    /// (`image_store::export`, surgical) so interactive edits can be checked into
    /// source control. Handled in `worker_loop` (it holds `world_dir`).
    ExportWorld,
    /// File ▸ "Load World from Files" — re-import `world/*.mst` into the image
    /// (the `seed` direction). Takes effect in the running VM after a restart.
    ImportWorld,
    /// Reset the worker's `BrowserSelection` to match the fresh default
    /// state the initial page render always shows (see
    /// `main.rs::open_class_browser`) — sent once whenever the browser
    /// view (re)opens, so the worker's retained selection can't drift out
    /// of sync with what's actually on screen.
    BrowserOpen,
    BrowserSelectPackage {
        name: String,
    },
    BrowserSelectClass {
        name: String,
    },
    BrowserSelectSide {
        side: Side,
    },
    BrowserSelectCategory {
        name: String,
    },
    BrowserSelectMethod {
        selector: String,
    },
    /// Cmd+S in the source pane's editor (`../smtk.js`) — "accept" the
    /// edited text against whatever the worker's `selection.edit_target`
    /// currently points at (a method, the class comment, the class
    /// definition, or — for the two `New*` targets — parse an identity
    /// (selector, or class name+header) out of the text itself and
    /// create/reopen/redefine accordingly). Stand-in for `mirror
    /// addMethod:category:ifFail:` (`../../docs/APPS.md` §5.2) — real
    /// `.mst`-grammar parsing (`image_store::mst`) now, for the parts of
    /// that grammar that exist, but still no compiler.
    ///
    /// The six `saved_*` fields are a snapshot of the `BrowserSelection`
    /// the source pane was rendered against (`browser_render.rs`'s
    /// `data-save-*` attributes, round-tripped through `smtk.js`), not just
    /// the edited text — without them, a selection-change request queued
    /// between the render and this arriving would apply the edit against
    /// whatever's *now* selected instead of what was actually on screen
    /// when the user hit Cmd+S. `handle`'s arm rejects (no apply, just an
    /// explanatory transcript line) if they no longer match.
    BrowserSaveSource {
        text: String,
        saved_package: String,
        saved_class: String,
        saved_side: String,
        saved_category: String,
        saved_method: String,
        saved_target: String,
    },
    /// "New Method" — switches the source pane into an editable blank
    /// template for `selection.class`/`side`/`category` (all three must
    /// already be selected; a no-op otherwise). Cmd+S on the result is what
    /// actually creates it — see `BrowserSaveSource`.
    BrowserNewMethod,
    /// "New Class" — switches the source pane into an editable blank
    /// class-definition template, prefilling the superclass with whatever
    /// class (if any) is currently selected. Requires `selection.package`
    /// (there's no "file" to derive a package from here, unlike a real
    /// `.mst` file — see `image_store::mst`'s doc comment) — a no-op if
    /// none is selected.
    BrowserNewClass,
    /// Switches the source pane to the selected class's "definition" tab
    /// (superclass + instance variables, as real `.mst` header syntax) —
    /// alongside the existing default "comment" tab. A no-op if no class
    /// is selected.
    BrowserEditDefinition,
    /// Switches back to the "comment" tab — symmetric with
    /// `BrowserEditDefinition`, for the tab pair above the source pane.
    BrowserEditComment,
    /// Remove the selected class. Unconditional — `docs/IMAGE.md`'s
    /// soft-delete means this is undo-able the same way any other edit is
    /// (no dedicated "unremove"), and the *decision* to allow removing a
    /// class that still has subclasses (rather than block, or
    /// auto-reparent them) was made deliberately: the browser is expected
    /// to warn about affected subclasses in the confirmation UI
    /// (`browser_render.rs`) *before* sending this, not have the request
    /// itself carry that logic.
    BrowserRemoveClass,
    /// Remove the selected method. See `BrowserRemoveClass`'s doc comment.
    BrowserRemoveMethod,
    /// The Workspace tool's (`workspace_render.rs`, `docs/APPS.md` §5.5)
    /// "Print it" — evaluate `code` (whatever's selected, or the whole
    /// buffer if nothing is) and answer its printString to insert inline,
    /// rather than just echoing to the transcript the way plain `Doit`
    /// does ("Do it" in the workspace *is* plain `Doit` — no new request
    /// needed for that half). Real `eval` (SPEC §16.2) already answers a
    /// printString directly (`Result<String, GuestError>`), so this is
    /// shaped to become a near-direct caller of it, not a stand-in that'll
    /// need reshaping later.
    WorkspacePrintIt {
        code: String,
    },
    /// "Smalltalk allocates a widget of a specific size" (`docs/CANVAS.md`
    /// §5.1) — stands in for a real `Canvas extent: w @ h` send (no VM-side
    /// primitive exists yet, same "prove the wire mechanics, don't
    /// simulate evaluation" posture as `WorkspacePrintIt`). Always
    /// (re)creates the single `id: 0` canvas — multiple simultaneous
    /// canvases are deferred (`docs/CANVAS.md` §7).
    CanvasCreate {
        width: u32,
        height: u32,
    },
    /// The Canvas view's (`canvas_render.rs`, `../../docs/CANVAS.md`)
    /// "Run Demo" button — same stand-in posture as `CanvasCreate`.
    CanvasRunDemo,
    /// Generic "let Smalltalk drive the canvas": evaluate `code` (any
    /// expression answering a command-batch String, `docs/CANVAS.md` §5.2) and
    /// draw the result. This is the general-purpose path — the GUI carries an
    /// opaque code string and gets back an opaque command string, exactly like
    /// `Doit`; it holds NO knowledge of what is being drawn. The Mandelbrot
    /// demo (`world/35_mandelbrot.mst`) is just one caller; any drawing or
    /// animation frame (`anim frameAt: k`) uses the same message unchanged.
    CanvasEval {
        code: String,
    },
    /// Generic "let Smalltalk paint a pixel buffer" (`docs/CANVAS.md` pixel
    /// path): evaluate `code`, which answers a `width*height*4` RGBA
    /// `ByteArray` (a `Pixmap`'s bytes, `world/36_pixmap.mst`), and blit it to
    /// the canvas in one `putImageData`. The right primitive for a per-pixel
    /// image (the Mandelbrot) vs `CanvasEval`'s vector command batch. Generic —
    /// the worker holds no knowledge of what is drawn.
    CanvasEvalPixels {
        code: String,
        width: u32,
        height: u32,
    },
    /// Clears the canvas — a single `clearRect` batch, not a separate
    /// code path (`docs/CANVAS.md` §5.3: "full redraw" is a command-batch
    /// convention, not a different channel).
    CanvasClear,
    /// Poll for a metrics snapshot (the GUI's periodic dashboard sampler,
    /// `main.rs`). The worker answers with `VmResponse::Metrics`. Cheap — a
    /// field read of live counters, no eval.
    GetMetrics,
}

/// VM → GUI. `Transcript` is what the G1 stub host already produces.
/// `BrowserPanes` carries all five rendered panes at once on every browser
/// selection — simple and cheap at this scale (see `handle`'s doc comment)
/// rather than tracking which panes actually changed. Real rendered
/// fragments (D-G5, once outliners are real) would likely want the finer-
/// grained per-node form `docs/APPS.md` §5.1 describes; this is deliberately
/// the simplest thing that lets the browser *view* and the *messaging
/// protocol* be built and tested now.
#[derive(Debug)]
pub enum VmResponse {
    Transcript(String),
    /// Answer to `VmRequest::GetMetrics`: a snapshot of the VM's runtime
    /// counters for the toolbar metrics dashboard (`main.rs`).
    Metrics(VmMetrics),
    /// Sent once right after (each) boot: a clone of this VM's per-VM live-
    /// signal block, so the main-thread sampler can read `compiled_depth`
    /// off-thread for the interpreter/compiler ratio. Consumed by
    /// `drain_responses` (like `WorkerIdle`), never forwarded to the UI.
    LiveStats(Arc<VmLiveStats>),
    /// A game-primitive command (`docs/gamepane_design.md` M3) emitted by the
    /// worker via `ChannelGameSink`; `main.rs` applies it to the native Metal
    /// game pane (`game_pane::apply_command`) on the main thread.
    Game(GameCommand),
    /// The reply to an editor key/command (`docs/editor_design.md` M3), decoded
    /// from `EditorSession>>render`: the WHOLE buffer text (a few KB — a class)
    /// plus the 0-based caret offset. The JS terminal blasts both into the
    /// textarea; no incremental patching. `cursor` is ready for
    /// `setSelectionRange`.
    EditorView {
        cursor: i64,
        text: String,
    },
    BrowserPanes {
        packages_html: String,
        classes_html: String,
        categories_html: String,
        methods_html: String,
        source_html: String,
    },
    /// Distinct from `BrowserPanes` on purpose, not just a "was this the
    /// first one" flag on the same variant: the *handling* genuinely
    /// differs (`main.rs` builds and loads a brand-new page for this one,
    /// vs. patching an already-loaded page's existing panes for
    /// `BrowserPanes`) — see `browser_render::assemble_panes`'s doc comment
    /// for the race this distinction fixes.
    BrowserOpened {
        packages_html: String,
        classes_html: String,
        categories_html: String,
        methods_html: String,
        source_html: String,
    },
    /// Answers `WorkspacePrintIt` — `main.rs` inserts `text` into the
    /// workspace's textarea right after the selection that was evaluated,
    /// rather than routing it to the transcript (`smtk.js`'s
    /// `macvmInsertPrintResult`).
    WorkspacePrintResult {
        text: String,
    },
    /// Answers `CanvasCreate` — `main.rs` (re)sets the `<canvas>` element's
    /// `width`/`height` attributes (which, per the HTML5 canvas spec,
    /// clears its content as a side effect — matches "allocate a widget of
    /// a specific size" reading as a fresh surface, not a resize-in-place).
    CanvasCreated {
        id: u32,
        width: u32,
        height: u32,
    },
    /// Answers `CanvasRunDemo` — `commands_json` is a JSON array of
    /// `[opName, ...args]` (`docs/CANVAS.md` §5.2), executed by
    /// `smtk.js`'s `macvmCanvasDraw` against a real `CanvasRenderingContext2D`.
    CanvasDraw {
        id: u32,
        commands_json: String,
    },
    /// Answers `CanvasEvalPixels` — a `width`x`height` RGBA image, base64 of
    /// the raw `ByteArray` (`docs/CANVAS.md` pixel path). `main.rs` hands it to
    /// `smtk.js`'s `macvmCanvasPutPixels`, which decodes it into an `ImageData`
    /// and `putImageData`s it onto the canvas in one blit. Base64 because
    /// `evaluateJavaScript` carries a JS source string, not binary — the
    /// `WKURLSchemeHandler` path (module doc) is the scale-up for animation.
    CanvasPixels {
        id: u32,
        width: u32,
        height: u32,
        base64: String,
    },
    /// Answers `CanvasClear`.
    CanvasCleared {
        id: u32,
    },
    /// Answers [`VmRequest::SmapplRender`] — `main.rs` calls `smtk.js`'s
    /// `macvmRenderSmappl(id, html)` to swap the placeholder span with `id`
    /// for the rendered widget `html` (D-G5).
    SmapplFragment {
        id: String,
        html: String,
    },
    /// Answers [`VmRequest::Find`] — `main.rs` fills the find page's results
    /// container with `html`.
    FindResults {
        html: String,
    },
    /// Answers [`VmRequest::FindOptions`] — `main.rs` populates the find page's
    /// `<datalist>` (`macvmSetFindOptions`) so its combobox offers `options`.
    FindOptions {
        tool: String,
        options: Vec<String>,
    },
    /// A live widget action asked to pop a modal dialog (`Visual>>promptOk:…`,
    /// the differences2.html "Press Me!" demo) — `main.rs` calls `smtk.js`'s
    /// `macvmShowOverlay(html)` to float `html` over the current page. Only a
    /// `String`-answering action produces this; side-effect actions don't.
    SmapplOverlay {
        html: String,
    },
    /// A live widget action asked to open a native view (a tour-page toolbar
    /// button — `Launcher launchers anElement browseStartPage`, etc.). The action
    /// answers a `'macvm-nav:TARGET'` marker; `main.rs` routes `target` through
    /// the same toolbar dispatch a native toolbar click uses.
    SmapplNavigate {
        target: String,
    },
    /// Internal supervisor liveness marker (S21 step 3) — NOT a UI response.
    /// The worker sends one after fully completing each request, i.e. once
    /// it is back to idle and provably still alive. `VmHost::drain_responses`
    /// consumes it (clearing the in-flight timer) and never forwards it, so
    /// the main thread never actually receives it — the arm in
    /// `main.rs::vm_bridge_drain` exists only to satisfy match exhaustiveness.
    /// The whole point: a worker that dies mid-request via
    /// `FatalMode::ExitThread` (`libc::pthread_exit`, `embed.rs`) never
    /// reaches this send, and because `pthread_exit` also leaks the request-
    /// channel receiver (no `Drop`, so `Sender::send` keeps succeeding and
    /// can't report the death), this absent marker is the ONLY signal the
    /// worker is gone — see [`WORKER_RESPONSE_TIMEOUT`].
    WorkerIdle,
}

// The waker itself lives in `crate::shell` — waking the UI thread is the one
// piece of this module that is platform-specific (a `performSelectorOnMain
// Thread:` on macOS, a `PostMessageW` on Windows). Both are documented as
// safe to call from any thread, which is the property this module depends on;
// both are `Copy` so `ChannelTranscript` (S21) can hold its own copy
// alongside the worker loop's; and both short-circuit when they have no UI
// thread behind them, so a headless test needs no windowing system at all
// (`Waker::none`).

/// How long `VmHost::submit` waits for ANY response before assuming the
/// worker thread is gone and respawning (S21 step 3 — "if the language
/// thread dies, the GUI must not die"). Generous on purpose: a real
/// Smalltalk doit can legitimately run for a while, and a false-positive
/// respawn would abandon a still-live computation. This is the ONLY
/// detection mechanism, not a fallback for a rarer case: a worker that
/// terminates via `FatalMode::ExitThread` (`libc::pthread_exit`) runs no
/// `Drop` glue at all, so the response channel never reports disconnected
/// and `JoinHandle::join()`/`.is_finished()` panic/hang (see `embed.rs`'s
/// module doc) — a bounded "nothing came back" timeout is the only signal
/// available for that death mode. An ordinary Rust panic (not one of
/// `embed.rs`'s `fatal_exit` sites) disconnects the channel immediately
/// instead, which `submit` also handles, faster than this timeout.
const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);

/// The mutable state a respawn must replace atomically — kept together so
/// "swap in a fresh worker" can never leave the request/response halves
/// pointing at two different worker generations.
struct HostInner {
    requests: Sender<VmRequest>,
    responses: Receiver<VmResponse>,
    /// Held ONLY so it can be `drop()`-ped when superseded — NEVER
    /// `.join()`-ed or `.is_finished()`-polled (see `WORKER_RESPONSE_
    /// TIMEOUT`'s own doc for why).
    #[allow(dead_code)]
    worker: std::thread::JoinHandle<()>,
    wake: Waker,
    world_dir: std::path::PathBuf,
    timeout: Duration,
    /// `Some(t)` from the moment a request is sent until the worker signals
    /// it finished that request (a `VmResponse::WorkerIdle` marker, cleared
    /// in `drain_responses`). Crucially NOT cleared by ordinary responses:
    /// a fatal doit emits its error transcript (via `ChannelTranscript`)
    /// and only THEN `pthread_exit`s, so clearing on any output would hide
    /// exactly the death this is meant to catch. If `submit` finds this
    /// still `Some` and older than `timeout`, the worker is presumed dead.
    pending_since: Option<Instant>,
    /// This VM generation's per-VM live-signal block (delivered by the boot-time
    /// `LiveStats` response), for the metrics sampler to read `compiled_depth`
    /// off-thread. `None` until the worker announces it just after boot; a
    /// respawn's new VM announces a fresh one, replacing this.
    live_stats: Option<Arc<VmLiveStats>>,
}

/// Handle held on the main thread: submits requests, drains responses, and
/// transparently respawns the worker thread if it ever stops responding.
pub struct VmHost {
    inner: Mutex<HostInner>,
}

impl VmHost {
    /// Is the worker idle (no request in flight)? The game-loop timer gates on
    /// this so it never queues a second `GameStep` before the previous frame's
    /// step returns — single-outstanding backpressure (`docs/gamepane_design.md`).
    pub fn is_idle(&self) -> bool {
        self.inner.lock().unwrap().pending_since.is_none()
    }

    pub fn submit(&self, request: VmRequest) {
        let mut inner = self.inner.lock().unwrap();
        if inner
            .pending_since
            .is_some_and(|t| t.elapsed() > inner.timeout)
        {
            respawn(&mut inner, RESPAWN_NOTICE_TIMEOUT);
        }
        let request = match inner.requests.send(request) {
            Ok(()) => {
                inner.pending_since = Some(Instant::now());
                return;
            }
            Err(mpsc::SendError(request)) => request,
        };
        // The worker's receiver is already gone even though we hadn't yet
        // timed out waiting for it — an ordinary Rust panic disconnects
        // the channel immediately (unlike a `pthread_exit` death, which
        // disconnects nothing at all). Respawn now and deliver this
        // request to the fresh worker instead of dropping it.
        respawn(&mut inner, RESPAWN_NOTICE_TIMEOUT);
        let _ = inner.requests.send(request);
        inner.pending_since = Some(Instant::now());
    }

    /// Kill the current language-thread worker and start a fresh one (a new
    /// VM booted from the image, new channels, new OS thread) — the "Restart
    /// VM Thread" menu item (`main.rs`). Safe to invoke at any time: if the
    /// worker is wedged in a runaway loop, it can't be force-killed (a Rust
    /// thread has no safe async kill), but it is abandoned and disconnected,
    /// and the fresh worker takes over immediately. Wakes the main thread so
    /// the "restarted" transcript notice shows right away rather than waiting
    /// for the next response.
    pub fn restart(&self) {
        let wake = {
            let mut inner = self.inner.lock().unwrap();
            respawn(&mut inner, "--- VM thread restarted ---");
            inner.wake
        };
        wake.notify();
    }

    /// Drain whatever responses have arrived since the last call. Called
    /// from the main-thread selector `perform_selector_on_main_thread`
    /// triggers. `WorkerIdle` markers are consumed here (clearing the
    /// in-flight timer) and never returned — see that variant's doc.
    pub fn drain_responses(&self) -> Vec<VmResponse> {
        let mut inner = self.inner.lock().unwrap();
        let mut out = Vec::new();
        while let Ok(response) = inner.responses.try_recv() {
            match response {
                VmResponse::WorkerIdle => inner.pending_since = None,
                VmResponse::LiveStats(arc) => inner.live_stats = Some(arc),
                other => out.push(other),
            }
        }
        out
    }

    /// A clone of the current VM's per-VM live-signal block, once the worker
    /// has announced it (just after boot). The metrics sampler reads
    /// `compiled_depth` from it off-thread — no request, no lock on the VM.
    pub fn live_stats(&self) -> Option<Arc<VmLiveStats>> {
        self.inner.lock().unwrap().live_stats.clone()
    }
}

/// One worker generation's channel endpoints + thread handle, as produced
/// by `spawn_worker` — grouped so a respawn always replaces all three
/// together (never a torn mix of an old receiver with a new sender, etc).
struct WorkerHandles {
    requests: Sender<VmRequest>,
    responses: Receiver<VmResponse>,
    /// A clone of the worker's own response sender, for injecting a
    /// supervisor-level notice (e.g. "restarted") that doesn't come from
    /// the worker itself. `mpsc::Sender` is multi-producer by design —
    /// holding a second clone alongside the worker's own doesn't change
    /// anything about the channel's behavior or lifetime.
    notices: Sender<VmResponse>,
    worker: std::thread::JoinHandle<()>,
}

fn spawn_worker(wake: Waker, world_dir: std::path::PathBuf) -> WorkerHandles {
    let (request_tx, request_rx) = mpsc::channel::<VmRequest>();
    let (response_tx, response_rx) = mpsc::channel::<VmResponse>();
    let notices = response_tx.clone();
    // The worker loop gets its OWN request sender: the multi-Smalltalk-worker
    // inbox wake (M3) queues a `WorkerInbox` request from a worker VM's thread
    // the moment an envelope lands — the send is the wake, and this thread's
    // `requests.recv()` sleep is what it interrupts.
    let self_tx = request_tx.clone();
    let worker = std::thread::spawn(move || {
        worker_loop(request_rx, response_tx, self_tx, wake, &world_dir);
    });
    WorkerHandles {
        requests: request_tx,
        responses: response_rx,
        notices,
        worker,
    }
}

/// Replaces `inner`'s worker with a brand new one (fresh channels, fresh
/// `VmHandle`, on a fresh OS thread) and drops the stale `JoinHandle` (safe
/// — never joined). `notice` is queued on the FRESH channel so the user sees
/// in the transcript why their workspace just "came back." The old worker,
/// if idle, exits when its request channel is dropped here; if it's
/// mid-computation (a runaway loop) it is simply abandoned — disconnected,
/// its output ignored — and the fresh worker serves everything from now on.
fn respawn(inner: &mut HostInner, notice: &str) {
    let w = spawn_worker(inner.wake, inner.world_dir.clone());
    let _ = w.notices.send(VmResponse::Transcript(notice.to_string()));
    inner.requests = w.requests;
    inner.responses = w.responses;
    inner.worker = w.worker;
    inner.pending_since = None;
}

const RESPAWN_NOTICE_TIMEOUT: &str =
    "--- the language thread stopped responding; a fresh one has been started ---";

/// Spawn the VM worker thread and wire it to wake the UI thread via `wake`
/// whenever a response is ready. The woken handler should call
/// [`VmHost::drain_responses`] and apply each one (see
/// `main.rs::on_vm_drain`). The world directory defaults to `world` (relative
/// to the process's own launch directory, same convention as the CLI's
/// `--world`), overridable via `MACVM_WORLD_PATH` — reading (not mutating) an
/// env var here is race-free even under a parallel test runner (see
/// `spawn_with_world_and_timeout`'s own doc for why tests use an explicit
/// path instead of this env var regardless).
pub fn spawn(wake: Waker) -> VmHost {
    let world_dir = std::env::var_os("MACVM_WORLD_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("world"));
    spawn_with_world_and_timeout(wake, world_dir, WORKER_RESPONSE_TIMEOUT)
}

/// `spawn`'s fully-parameterized form — a testability seam, not a second
/// production entry point. Real callers always go through `spawn` (fixed
/// world-directory convention, the real 30s timeout); tests need an
/// explicit `world_dir` (`cargo test`'s working directory is the crate
/// root, `gui/`, not the workspace root `world/` lives in — env var
/// mutation to fix that up would race under a parallel test runner, per
/// `tests/common/mod.rs`'s own established rule in the main crate) and a
/// much shorter `timeout` (so a respawn test doesn't need to sleep 30 real
/// seconds).
fn spawn_with_world_and_timeout(
    wake: Waker,
    world_dir: std::path::PathBuf,
    timeout: Duration,
) -> VmHost {
    let w = spawn_worker(wake, world_dir.clone());
    VmHost {
        inner: Mutex::new(HostInner {
            requests: w.requests,
            responses: w.responses,
            worker: w.worker,
            wake,
            world_dir,
            timeout,
            pending_since: None,
            live_stats: None,
        }),
    }
}

fn worker_loop(
    requests: Receiver<VmRequest>,
    responses: Sender<VmResponse>,
    self_requests: Sender<VmRequest>,
    wake: Waker,
    world_dir: &Path,
) {
    // Owned entirely by this thread — no `Mutex`, no `static`. The UI
    // thread never touches `world`/`selection` directly, only ever asks
    // for a change via a `VmRequest`; see this module's doc comment on
    // "routing a click to the right closure, and locking" for why that's
    // the right shape, not just a convenient one. `world` is `mut` now
    // that `BrowserSaveSource` can edit it (still worker-thread-only).
    //
    // S22: the versioned image database is the single source of truth. The
    // VM boots FROM the image (genesis + `load_world_from_image`, not the
    // `.mst` file tree), and the browser's mock mirrors the SAME image, so
    // an edit written through to both stays consistent. The image is
    // auto-seeded from `world/*.mst` on first run (zero setup). If the DB
    // can't be established at all, fall back to the old `.mst` boot + mock
    // seed so the GUI still works.
    let image_path = resolve_image_path(world_dir);
    eprintln!("PROBE worker_loop: image_path={}", image_path.display());
    let (image, mut vm) = match open_or_seed_image(world_dir, &image_path) {
        Ok(img) => match { eprintln!("PROBE: image opened, booting vm"); let r = boot_vm_from_image(&img, responses.clone(), wake); eprintln!("PROBE: boot_vm_from_image -> {}", r.is_some()); r } {
            Some(vm) => (Some(img), vm),
            // The image opened but the world load failed — most likely a
            // STALE image written by an older importer. Fall back to the
            // .mst boot + mock so the GUI still works, rather than
            // respawn-looping forever on a bad image. (A boot failure was
            // already reported to the transcript by `boot_vm_from_image`.)
            None => match boot_real_vm(responses.clone(), wake, world_dir) {
                Some(vm) => (None, vm),
                None => return,
            },
        },
        Err(e) => {
            eprintln!("vm_host: {e} — falling back to .mst boot + mock world");
            match boot_real_vm(responses.clone(), wake, world_dir) {
                Some(vm) => (None, vm),
                None => return,
            }
        }
    };
    let mut world = match &image {
        Some(img) => mock_world_from_image(img),
        None => MockWorld::seed(),
    };
    let mut selection = BrowserSelection::default();

    // Multi-Smalltalk workers, M3 (docs/multi-smalltalk-worker.md §9): this
    // VM is a worker-spawning PRIMARY. Workers boot plain from the SAME
    // image (no GUI sinks — their transcript forwards through the inbox,
    // [wN]-tagged), and an inbound envelope wakes THIS thread by queueing a
    // WorkerInbox request from the sending worker's thread: event-driven,
    // coalesced, zero polling.
    let boot: macvm::runtime::workers::WorkerBootFn = match &image {
        Some(_) => {
            let img_path = image_path.clone();
            Arc::new(move || {
                let img = image_store::Image::open(&img_path).map_err(|e| {
                    macvm::runtime::VmError {
                        msg: format!("worker image open: {e}"),
                    }
                })?;
                let mut h = VmHandle::boot_without_world(gui_vm_options());
                crate::world_boot::load_world_from_image(&mut h, &img, WORLD_LISTS, WORLD_DOITS)
                    .map_err(|msg| macvm::runtime::VmError { msg })?;
                Ok(h)
            })
        }
        None => {
            let dir = world_dir.to_path_buf();
            Arc::new(move || VmHandle::boot(gui_vm_options(), &dir))
        }
    };
    vm.set_worker_boot(boot);
    vm.set_inbox_wake(Arc::new(move || {
        let _ = self_requests.send(VmRequest::WorkerInbox);
    }));

    // Announce this VM's per-VM live-signal block to the main thread once, so
    // the metrics sampler can read compiled_depth off-thread (a respawn boots a
    // fresh VM and re-announces its block). Consumed by drain_responses.
    let _ = responses.send(VmResponse::LiveStats(vm.live_stats()));

    for request in requests {
        // Export/import touch the world/*.mst file tree, so they're serviced
        // here where `world_dir` is in scope (`handle` has no `world_dir`).
        let batch = match request {
            VmRequest::ExportWorld => vec![export_world_response(image.as_ref(), world_dir)],
            VmRequest::ImportWorld => vec![import_world_response(image.as_ref(), world_dir)],
            VmRequest::EditorAccept => {
                vec![editor_accept_response(&mut vm, image.as_ref(), world_dir)]
            }
            other => handle(other, &mut world, &mut selection, image.as_ref(), &mut vm),
        };
        // Reaching HERE (rather than `pthread_exit`ing inside `handle` on a
        // fatal guest condition) is exactly the "still alive" fact the
        // WorkerIdle marker below reports — so it is sent unconditionally
        // AFTER the batch, as the last thing each iteration does before
        // blocking for the next request.
        let mut disconnected = false;
        for response in batch
            .into_iter()
            .chain(std::iter::once(VmResponse::WorkerIdle))
        {
            if responses.send(response).is_err() {
                disconnected = true; // main thread's receiver is gone
                break;
            }
        }
        if disconnected {
            break;
        }
        wake.notify();
    }
}

/// Routes `Transcript show:`/`printOnStdout:` output (SPEC §16.2) onto the
/// worker's own response channel as ordinary `VmResponse::Transcript`
/// messages, waking the main thread after EVERY line rather than only once
/// the whole request finishes. That immediacy matters most for the case
/// this whole sprint (S21) exists to handle: a fatal guest condition writes
/// its error message here BEFORE the worker thread terminates
/// (`FatalMode::ExitThread`/`fatal_exit`, `embed.rs`) — batching the wake
/// until `handle` returns would mean it never fires at all for that
/// request, since `handle` never returns in that case.
struct ChannelTranscript {
    responses: Sender<VmResponse>,
    wake: Waker,
}

impl TranscriptSink for ChannelTranscript {
    fn show(&mut self, text: &str) {
        if self
            .responses
            .send(VmResponse::Transcript(text.to_string()))
            .is_ok()
        {
            self.wake.notify();
        }
    }
}

/// The game analogue of [`ChannelTranscript`] (`docs/gamepane_design.md` M3):
/// a worker-thread [`GameSink`] that forwards each `GameCommand` over the same
/// response channel as `VmResponse::Game`, waking the main thread so
/// `main.rs`'s drain applies it to the native Metal pane.
struct ChannelGameSink {
    responses: Sender<VmResponse>,
    wake: Waker,
}

impl GameSink for ChannelGameSink {
    fn emit(&mut self, cmd: GameCommand) {
        if self.responses.send(VmResponse::Game(cmd)).is_ok() {
            self.wake.notify();
        }
    }
}

/// VM options for the GUI's own worker. Same as `VmOptions::from_env`
/// (`MACVM_JIT`/`MACVM_HEAP`/… all honored), except the JIT defaults to ON
/// when `MACVM_JIT` is unset — the GUI runs real programs interactively, and
/// the JIT is the whole point. `threshold=10` compiles a method after a brief
/// warmup: the *interactive* workload (tool/outliner rendering) is many
/// methods each called dozens of times, not one 2000-iteration hot loop, so a
/// high threshold (the old 2000) never compiled the UI code and left it
/// interpreter-slow; 10 gets it compiled within the first render or two while
/// still not compiling genuinely one-shot boot code. `MACVM_JIT` still
/// overrides (including `MACVM_JIT=off`); the library default
/// (`from_env` → off) is deliberately left alone so the test suite keeps its
/// pure-interpreter baseline (gate scripts opt in explicitly).
pub(crate) fn gui_vm_options() -> macvm::runtime::VmOptions {
    let mut opts = macvm::runtime::VmOptions::from_env();
    if std::env::var_os("MACVM_JIT").is_none() {
        opts.jit = macvm::runtime::JitMode::Threshold(10);
    }
    opts
}

/// Boots the real embedded VM (SPEC §16.2, `src/embed.rs`) against
/// `world_dir` using [`gui_vm_options`] (JIT on by default). Used as the
/// `.mst` fallback when the image database can't be established.
fn boot_real_vm(
    responses: Sender<VmResponse>,
    wake: Waker,
    world_dir: &Path,
) -> Option<VmHandle> {
    let opts = gui_vm_options();
    match VmHandle::boot(opts, world_dir) {
        Ok(mut vm) => {
            vm.set_transcript(Box::new(ChannelTranscript {
                responses: responses.clone(),
                wake,
            }));
            vm.set_game_sink(Box::new(ChannelGameSink { responses, wake }));
            Some(vm)
        }
        Err(e) => {
            let _ = responses.send(VmResponse::Transcript(format!("VM boot failed: {}", e.msg)));
            wake.notify();
            None
        }
    }
}

/// The image database path (S22): `MACVM_IMAGE_PATH` if set, else the
/// default `<world_dir>/image.sqlite3` — so the DB is the out-of-the-box
/// source of truth, no env var required.
pub(crate) fn resolve_image_path(world_dir: &Path) -> PathBuf {
    std::env::var_os("MACVM_IMAGE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| world_dir.join("image.sqlite3"))
}

/// Opens the image at `image_path` (creating the file if missing), seeding
/// it from `world_dir`'s `.mst` files the first time (when it has no
/// classes) so a fresh checkout Just Works with zero setup. `docs/IMAGE.md`
/// §6/§8. A thin wrapper: the real logic moved to `image_store::import::
/// open_or_seed` (M4, `docs/package_aware_editing_design.md` §4.5) so
/// `cocoa_gui` shares it rather than duplicating it — `image_store`'s own
/// doc comment on that function explains why `world.list`-only would have
/// been a real bug for the other GUI.
fn open_or_seed_image(world_dir: &Path, image_path: &Path) -> Result<image_store::Image, String> {
    image_store::import::open_or_seed(world_dir, image_path)
}

/// File ▸ "Save World to Files" — write the image back over `world/*.mst`
/// (surgically), reporting the result to the transcript.
fn export_world_response(image: Option<&image_store::Image>, world_dir: &Path) -> VmResponse {
    let msg = match image {
        None => "Save World: no image database is loaded.".to_string(),
        Some(img) => match image_store::export::export_world_dir(img, world_dir) {
            Ok(s) => format!(
                "Saved world to {} — {} file(s) changed ({} method(s) updated, {} added, {} new class(es)). Review with `git diff`.",
                world_dir.display(),
                s.files_changed,
                s.methods_updated,
                s.methods_added,
                s.classes_added
            ),
            Err(e) => format!("Save World failed: {e}"),
        },
    };
    VmResponse::Transcript(msg)
}

/// File ▸ "Load World from Files" — re-import `world/*.mst` into the image (the
/// seed direction). Live only after a VM restart.
fn import_world_response(image: Option<&image_store::Image>, world_dir: &Path) -> VmResponse {
    let msg = match image {
        None => "Load World: no image database is loaded.".to_string(),
        Some(img) => match image_store::import::import_world_dir(img, world_dir) {
            Ok(stats) => {
                let sends = img.backfill_method_sends().unwrap_or(0);
                format!(
                    "Loaded world from {} into the image ({stats:?}, {sends} send-edges). Restart the VM (VM menu) to run the changes.",
                    world_dir.display()
                )
            }
            Err(e) => format!("Load World failed: {e}"),
        },
    };
    VmResponse::Transcript(msg)
}

/// Boots the VM FROM the image (S22): genesis, then replay the whole world
/// out of the database ([`crate::world_boot::load_world_from_image`]) rather
/// than the `.mst` file tree. Installs the transcript sink first so any
/// load-time output reaches the GUI. Returns `None` (reporting to the
/// transcript) on a load failure, so the caller can end this worker
/// generation and let the supervisor respawn.
fn boot_vm_from_image(
    image: &image_store::Image,
    responses: Sender<VmResponse>,
    wake: Waker,
) -> Option<VmHandle> {
    let opts = gui_vm_options();
    let mut vm = VmHandle::boot_without_world(opts);
    vm.set_transcript(Box::new(ChannelTranscript {
        responses: responses.clone(),
        wake,
    }));
    vm.set_game_sink(Box::new(ChannelGameSink {
        responses: responses.clone(),
        wake,
    }));
    match crate::world_boot::load_world_from_image(&mut vm, image, WORLD_LISTS, WORLD_DOITS) {
        Ok(stats) => {
            eprintln!(
                "vm_host: booted VM from image ({} classes, {} methods, {} doits)",
                stats.classes, stats.methods, stats.doits
            );
            Some(vm)
        }
        Err(e) => {
            let _ = responses.send(VmResponse::Transcript(format!(
                "VM boot from image failed: {e}"
            )));
            wake.notify();
            None
        }
    }
}

/// Wraps one method's edited `.mst` `text` as a reopen of `class_name`, so
/// [`live_compile`] installs/replaces just that method in the running VM.
/// The superclass (from the mock, which was just updated) is needed for the
/// reopen — `install_class_def` checks it matches the live class. Class-side
/// methods carry their own `<Class> class >>` prefix in the stored source,
/// so `text` drops straight in either way.
fn reopen_one_method(world: &MockWorld, class_name: &str, text: &str) -> String {
    let superclass = world
        .class_named(class_name)
        .and_then(|c| c.superclass.clone())
        .unwrap_or_else(|| "nil".to_string());
    format!(
        "{superclass} subclass: {class_name} [\n{}\n]\n",
        text.trim()
    )
}

/// A conservative check that `s` is a usable variable name — a Smalltalk-style
/// identifier (a letter or `_`, then letters/digits/`_`). Guards the outliner's
/// add-variable path from persisting junk into a class's stored shell.
fn is_valid_var_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Compiles `mst_source` into the running VM (S22-E) so a browser edit is
/// LIVE — usable immediately — not merely saved to the mock + image. A
/// compile failure is returned as a short string (the edit is still saved;
/// it just isn't live until fixed), never a panic or a thread kill: a syntax
/// error is a `GuestError::Compile`, and `exec` recovers a genuine native
/// fault too (`embed.rs`).
fn live_compile(vm: &mut VmHandle, mst_source: &str) -> Result<(), String> {
    vm.exec(mst_source).map_err(|e| e.to_string())
}

/// M5 (`docs/package_aware_editing_design.md` §4.4): whether `vm` — the
/// ACTUAL VM a reopen is about to run on — already has `class_name` defined.
/// `world` (the `MockWorld` mirror) is NOT a safe stand-in for this since M4:
/// it mirrors the WHOLE database, including packages this specific VM never
/// booted (`import_all_lists`), so `world.class_named(...).is_some()` no
/// longer implies `vm` has it. A bare reference to an unknown global fails
/// to compile (`GuestError::Compile`); a known class name evaluates to
/// itself — so evaluating it directly is a cheap, exact existence check.
fn vm_has_class(vm: &mut VmHandle, class_name: &str) -> bool {
    vm.eval(class_name).is_ok()
}

/// What attempting a live-compile reopen produced.
enum ReopenOutcome {
    Applied,
    /// Skipped — never attempted, so no compile-error text would be honest
    /// (§4.4: a reopen has no ivar declaration, so running it unconditionally
    /// on a VM that never loaded `class_name` would silently DEFINE A WRONG-
    /// SHAPE SHELL under that name instead of failing — the real bug this
    /// gate exists to close, §3).
    NotLoaded,
    Failed(String),
}

/// The M5-gated counterpart to [`live_compile`] for a REOPEN specifically
/// (an edit to an existing class assumed to already have its real shape —
/// `acceptMethod`'s/`reopen_one_method`'s shape). Only ever attempts the
/// compile once [`vm_has_class`] confirms the target VM already has
/// `class_name`. Does NOT apply to a full class definition (real ivars
/// included, e.g. `SmapplNewClass`/`ClassDefinition`) — that shape either
/// resolves correctly on a VM seeing it for the first time or fails an
/// honest compile error, never the silent-wrong-shape failure this guards.
fn live_compile_reopen(vm: &mut VmHandle, class_name: &str, mst_source: &str) -> ReopenOutcome {
    if !vm_has_class(vm, class_name) {
        return ReopenOutcome::NotLoaded;
    }
    match live_compile(vm, mst_source) {
        Ok(()) => ReopenOutcome::Applied,
        Err(e) => ReopenOutcome::Failed(e),
    }
}

fn side_to_image(side: Side) -> image_store::Side {
    match side {
        Side::Instance => image_store::Side::Instance,
        Side::Class => image_store::Side::Class,
    }
}

fn side_from_image(side: image_store::Side) -> Side {
    match side {
        image_store::Side::Instance => Side::Instance,
        image_store::Side::Class => Side::Class,
    }
}

/// What a successful mirror-side save/remove reports about the *image*
/// side of the same operation — `image_error: None` means it actually
/// persisted (or there's no image configured at all, pure-mock mode);
/// `Some(reason)` means it didn't, so the transcript message can say so
/// honestly instead of unconditionally claiming "(persisted to image)"
/// the way this used to regardless of outcome.
struct SaveOk {
    what: String,
    image_error: Option<String>,
}

/// Describes what happened writing through to the image for the common
/// `Result<bool, _>` shape (`set_method_source`/`set_class_comment`/
/// `set_class_definition`/`remove_class`/`remove_method` all return this,
/// as `rusqlite::Result<bool>` — generic here rather than naming that type
/// directly since this crate doesn't depend on `rusqlite` itself, only on
/// `image_store`). `Ok(false)` — the mirror found the target but the image
/// didn't — is a mirror/image drift case, not a "successfully did
/// nothing": it's reported the same as a hard error, not silently treated
/// as success.
fn describe_image_write<E: std::fmt::Display>(result: Result<bool, E>) -> Option<String> {
    match result {
        Ok(true) => None,
        Ok(false) => Some("not found in the image (mirror/image out of sync?)".to_string()),
        Err(e) => Some(e.to_string()),
    }
}

/// One-shot mirror of every class/method `image`'s latest versions hold
/// into a fresh `MockWorld` — `browser_render.rs` renders from this exactly
/// as it would the invented seed; `handle`'s `BrowserSaveSource` arm keeps
/// both in sync afterward (mirror updated for immediate display, `image`
/// updated so the edit survives a restart).
fn mock_world_from_image(image: &image_store::Image) -> MockWorld {
    let mut world = MockWorld::empty();
    let classes = image.all_classes().unwrap_or_default();
    for c in &classes {
        world.add_class(
            &c.name,
            c.superclass.as_deref(),
            &c.category,
            &c.comment,
            &c.instance_vars,
            &c.class_vars,
        );
    }
    for c in &classes {
        for m in image.all_methods_of(&c.name).unwrap_or_default() {
            world.add_method(
                &c.name,
                side_from_image(m.side),
                &m.selector,
                &m.category,
                &m.source,
            );
        }
    }
    world
}

thread_local! {
    /// The open smappl outliners on the current page — the `ToolRegistry`
    /// (`docs/APPS.md` §7): `(widget_id, visual= code)`. Populated as each
    /// outliner renders; consulted after an accept to live-refresh the others
    /// so an edit in one view shows up in the rest without a page reload. Kept
    /// thread-local on the single-threaded worker so `handle`'s signature (and
    /// its ~15 test call sites) stay put. Bounded: `data-widget-id`s restart
    /// at `s0` per page, so re-rendering a page replaces its own entries.
    static OPEN_OUTLINERS: std::cell::RefCell<Vec<(String, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Render a `visual=` expression and fill any ClassOutliner source blocks from
/// image_store (the VM keeps no method source). `None` on a render failure.
fn render_and_inject(
    vm: &mut VmHandle,
    image: Option<&image_store::Image>,
    code: &str,
) -> Option<String> {
    let html = vm.render_fragment(code).ok()?;
    Some(match image {
        Some(img) => crate::preprocess::inject_method_sources(&html, |c, s, sel| {
            let side = if s == "class" {
                image_store::Side::Class
            } else {
                image_store::Side::Instance
            };
            img.method_source(c, side, sel).ok().flatten()
        }),
        None => html,
    })
}

use crate::preprocess::tag_widget_id;

/// The "tool unavailable" fallback (no image handle, or a failed render).
fn find_unavailable() -> String {
    "<div class=\"st-find-empty\">(this find tool isn't available)</div>".to_string()
}

fn find_results_wrap(head: &str, body: &str) -> String {
    format!("<div class=\"st-find-results\"><div class=\"st-find-head\">{head}</div>{body}</div>")
}

/// Find-Definition results — every class whose name contains `query`
/// (case-insensitive) — in the same `.st-find-*` structure the old Smalltalk
/// `DefinitionsView` produced, so smtk.js's result-click drill-in
/// (`.st-class-link[data-open-class]`) keeps working unchanged.
fn render_definition_results(img: &image_store::Image, query: &str) -> String {
    let needle = query.to_lowercase();
    let hits: Vec<String> = img
        .class_names()
        .unwrap_or_default()
        .into_iter()
        .filter(|n| n.to_lowercase().contains(&needle))
        .collect();
    let head = format!(
        "Classes matching \u{201c}{}\u{201d}",
        crate::preprocess::html_escape_text(query)
    );
    if hits.is_empty() {
        return find_results_wrap(&head, "<div class=\"st-find-empty\">(no matching classes)</div>");
    }
    let items: String = hits
        .iter()
        .map(|n| {
            let e = crate::preprocess::html_escape_text(n);
            format!("<div class=\"st-find-item\"><span class=\"st-class-link\" data-open-class=\"{e}\">{e}</span></div>")
        })
        .collect();
    find_results_wrap(&head, &items)
}

/// Implementors results — every class implementing `query` as a selector, on
/// either side, from image_store (SQL, no VM round trip).
fn render_implementors_results(img: &image_store::Image, query: &str) -> String {
    let hits = img.implementors_of(query).unwrap_or_default();
    let head = format!("Implementors of {}", crate::preprocess::html_escape_text(query));
    if hits.is_empty() {
        return find_results_wrap(&head, "<div class=\"st-find-empty\">(no implementors)</div>");
    }
    let items: String = hits
        .iter()
        .map(|(name, side)| {
            let e = crate::preprocess::html_escape_text(name);
            let s = match side {
                image_store::Side::Class => "class",
                image_store::Side::Instance => "instance",
            };
            format!("<div class=\"st-find-item\"><span class=\"st-class-link\" data-open-class=\"{e}\">{e}</span> <span class=\"st-find-side\">{s}</span></div>")
        })
        .collect();
    find_results_wrap(&head, &items)
}

/// Senders results — every method whose latest source sends `query`, from the
/// `method_sends` index (image_store::senders_of), no VM round trip. Each row
/// drills into the sending class and names the sending method.
fn render_senders_results(img: &image_store::Image, query: &str) -> String {
    let hits = img.senders_of(query).unwrap_or_default();
    let head = format!("Senders of {}", crate::preprocess::html_escape_text(query));
    if hits.is_empty() {
        return find_results_wrap(&head, "<div class=\"st-find-empty\">(no senders)</div>");
    }
    let items: String = hits
        .iter()
        .map(|(cls, msel, side)| {
            let ec = crate::preprocess::html_escape_text(cls);
            let em = crate::preprocess::html_escape_text(msel);
            let s = match side {
                image_store::Side::Class => "class",
                image_store::Side::Instance => "instance",
            };
            format!("<div class=\"st-find-item\"><span class=\"st-class-link\" data-open-class=\"{ec}\">{ec}</span> &raquo; {em} <span class=\"st-find-side\">{s}</span></div>")
        })
        .collect();
    find_results_wrap(&head, &items)
}

/// Re-render every open outliner except `skip_id` (the one being edited, whose
/// DOM must not be disrupted mid-edit) and answer fragment updates for them —
/// the live half of the `ToolRegistry` (`docs/APPS.md` §7). Called after an
/// accept so a change shows up across all open outliners.
fn refresh_open_outliners(
    vm: &mut VmHandle,
    image: Option<&image_store::Image>,
    skip_id: &str,
) -> Vec<VmResponse> {
    let entries: Vec<(String, String)> = OPEN_OUTLINERS.with(|o| o.borrow().clone());
    let mut out = Vec::new();
    for (id, code) in entries {
        if id == skip_id {
            continue;
        }
        if let Some(html) = render_and_inject(vm, image, &code) {
            out.push(VmResponse::SmapplFragment {
                html: tag_widget_id(&html, &id),
                id,
            });
        }
    }
    out
}

/// `Doit`/`WorkspacePrintIt` (S21 step 3) call the real `vm.eval` — see
/// `boot_real_vm`. Every other request is still the Browser/Canvas mock
/// (`docs/SPEC.md` §16.3's mirror-primitive bridge is a separate, later
/// piece of work). The `BrowserSelect*` arms update `selection` then
/// re-render every pane — simplest-thing-that-works at mock scale (see
/// `VmResponse::BrowserPanes`'s doc comment). Returns a `Vec` rather than a
/// single `VmResponse` because `BrowserSaveSource` needs to deliver two
/// things — updated panes *and* a transcript confirmation — not because
/// any request produces a high-frequency stream (that's `LatestSlot`'s job,
/// see this module's doc comment).
/// Wrap `s` as a Smalltalk string literal for splicing into an `eval` — quote
/// it and double every embedded quote (newlines are legal verbatim inside a
/// Smalltalk string, so a whole class's source passes through unescaped).
fn smalltalk_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push('\'');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// Decode `EditorSession>>render` (the 0-based caret offset on the first line,
/// then the whole buffer text after the first newline) into the blast the JS
/// terminal consumes.
fn decode_editor_view(enc: &str) -> VmResponse {
    let (header, text) = enc.split_once('\n').unwrap_or((enc, ""));
    VmResponse::EditorView {
        cursor: header.trim().parse().unwrap_or(0),
        text: text.to_string(),
    }
}

/// Save (accept) the current editor buffer — to the DATABASE and the world
/// `.mst` files, NOT the live VM (docs/editor_design.md M4). Redefining a class
/// in the running VM cannot change its shape when it has live instances (the
/// "cannot change shape of existing class" error), and mutating the running VM
/// under the user is the wrong thing anyway: the image is the boot source, so
/// the edit takes effect on the next boot. So this only READS the buffer from
/// the session, SYNTAX-checks it with the real compiler's parser (no install),
/// and persists — to the image (diffed) and out to `world/*.mst` (surgical).
fn editor_accept_response(
    vm: &mut VmHandle,
    image: Option<&image_store::Image>,
    world_dir: &Path,
) -> VmResponse {
    let text = match vm.eval_to_string("EditorSession current documentString") {
        Ok(t) => t,
        Err(e) => {
            return VmResponse::Transcript(format!("editor: nothing to save — no session ({e})"))
        }
    };
    // Syntax gate ONLY — the same parser that will compile this on the next
    // boot (`frontend::parser::parse_file`), so a red here means a real
    // compile error, and there is no live install to fail on a shape change.
    if let Err(e) = macvm::frontend::parser::parse_file(&text) {
        return VmResponse::Transcript(format!("editor: not saved — {e}"));
    }
    let img = match image {
        Some(i) => i,
        None => return VmResponse::Transcript("editor: not saved — no image database".to_string()),
    };
    let db = persist_editor_class(img, &text);
    // Push the image's changes back out to the world .mst tree (surgical —
    // only the changed methods are spliced), so the edit is on disk too.
    let world = match image_store::export::export_world_dir(img, world_dir) {
        Ok(s) => format!(
            "; world +{} file(s), {} method(s) updated, {} added",
            s.files_changed, s.methods_updated, s.methods_added
        ),
        Err(e) => format!("; world export failed: {e}"),
    };
    VmResponse::Transcript(format!("editor: {db}{world} (takes effect on next launch)"))
}

/// Persist the editor buffer's class to the image, writing only what changed
/// (docs/editor_design.md M4). `image_store::mst` re-parses the (already
/// syntax-checked) buffer purely to diff against what's stored: a method whose
/// source is byte-identical is left alone
/// (no new `method_versions` row — the version history and every live nmethod
/// are spared a no-op churn), changed/new methods are reopened, and methods that
/// vanished from the buffer are removed. Returns a one-line summary.
///
/// First-cut scope: instance/class methods and the class shell (superclass +
/// instance vars). The class comment and class variables are not persisted here
/// yet — a comment/classVars-aware round trip is follow-on work; a method edit,
/// the common case, is fully covered.
fn persist_editor_class(img: &image_store::Image, text: &str) -> String {
    use std::collections::{HashMap, HashSet};
    let parsed = image_store::mst::parse_mst_source(text);
    let cls = match parsed.first() {
        Some(c) => c,
        None => return "nothing to save (no class definition in the buffer)".to_string(),
    };
    // Ensure the class exists, then update its SHAPE (superclass + instance
    // vars) — `create_or_reopen_class` only reopens for methods, so a changed
    // shape needs `set_class_definition` (this is exactly the edit a live VM
    // redefinition can't do, and the reason Save goes to the database).
    let _ = img.create_or_reopen_class(
        &cls.name,
        cls.superclass.as_deref(),
        "",
        "",
        &cls.instance_vars,
    );
    let _ = img.set_class_definition(&cls.name, cls.superclass.as_deref(), &cls.instance_vars);

    // (selector, is_class_side) -> stored source, for the diff.
    let stored: HashMap<(String, bool), String> = img
        .all_methods_of(&cls.name)
        .unwrap_or_default()
        .into_iter()
        .map(|m| ((m.selector, m.side == image_store::Side::Class), m.source))
        .collect();

    let (mut changed, mut added) = (0u32, 0u32);
    for m in &cls.methods {
        let side = if m.is_class_side {
            image_store::Side::Class
        } else {
            image_store::Side::Instance
        };
        match stored.get(&(m.selector.clone(), m.is_class_side)) {
            Some(old) if *old == m.source => {}                     // unchanged: skip
            Some(_) => {
                let _ = img.create_or_reopen_method(&cls.name, side, &m.selector, "", &m.source);
                changed += 1;
            }
            None => {
                let _ = img.create_or_reopen_method(&cls.name, side, &m.selector, "", &m.source);
                added += 1;
            }
        }
    }

    // Methods that were in the image but are gone from the buffer → removed.
    let present: HashSet<(String, bool)> = cls
        .methods
        .iter()
        .map(|m| (m.selector.clone(), m.is_class_side))
        .collect();
    let mut removed = 0u32;
    for (sel, is_cls) in stored.keys() {
        if !present.contains(&(sel.clone(), *is_cls)) {
            let side = if *is_cls {
                image_store::Side::Class
            } else {
                image_store::Side::Instance
            };
            let _ = img.remove_method(&cls.name, side, sel);
            removed += 1;
        }
    }

    if changed + added + removed == 0 {
        format!("{} — no changes", cls.name)
    } else {
        format!(
            "saved {} ({changed} changed, {added} added, {removed} removed)",
            cls.name
        )
    }
}

/// Run an editor bridge expression that answers an `EditorSession>>render`
/// string and decode it. A guest error (e.g. no session open yet) surfaces on
/// the transcript rather than as a bogus view.
fn editor_eval(vm: &mut VmHandle, src: &str) -> Vec<VmResponse> {
    match vm.eval_to_string(src) {
        Ok(enc) => vec![decode_editor_view(&enc)],
        Err(e) => vec![VmResponse::Transcript(format!("editor error: {e}"))],
    }
}

fn handle(
    request: VmRequest,
    world: &mut MockWorld,
    selection: &mut BrowserSelection,
    image: Option<&image_store::Image>,
    vm: &mut VmHandle,
) -> Vec<VmResponse> {
    match request {
        VmRequest::Doit { code } => {
            // "Do it": evaluate for effect (Transcript show:, etc. arrive
            // separately via ChannelTranscript) and echo what ran — the
            // result value itself is deliberately not shown, matching
            // "Print it"'s (WorkspacePrintIt) own distinct job below.
            match vm.eval(&code) {
                Ok(_) => vec![VmResponse::Transcript(format!("> {code}"))],
                Err(e) => vec![VmResponse::Transcript(format!("> {code}\n{e}"))],
            }
        }
        VmRequest::GameStep { keys } => {
            // One frame: run the registered step block with this tick's held
            // keys. The block's drawing reaches the GUI via the game sink
            // (VmResponse::Game), NOT this return value — so answer nothing
            // (echoing every frame to the transcript would flood it). A step
            // that errors is reported once and the loop keeps ticking.
            match vm.eval(&format!("GamePane stepWithKeys: {keys}.")) {
                Ok(_) => vec![],
                Err(e) => vec![VmResponse::Transcript(format!("game step error: {e}"))],
            }
        }
        VmRequest::WorkerInbox => {
            // Drain + dispatch every queued worker envelope (continuations,
            // handlers, forwarded transcripts). Silent on success — the wake
            // hook queues one of these per burst (coalesced), and anything
            // user-visible arrives separately via ChannelTranscript/game sink.
            match vm.eval("Worker dispatchInbox.") {
                Ok(_) => vec![],
                Err(e) => vec![VmResponse::Transcript(format!("worker inbox error: {e}"))],
            }
        }
        VmRequest::EditorOpen { text } => {
            // Open a session, then ask it for a full-document damage — one
            // eval, the block's last expression is the encoded record.
            let src = format!(
                "EditorSession openOn: {}. EditorSession current render",
                smalltalk_string_literal(&text)
            );
            editor_eval(vm, &src)
        }
        VmRequest::EditorKey { key, at } => editor_eval(
            vm,
            &format!(
                "EditorSession current handleKey: {} at: {at}",
                smalltalk_string_literal(&key)
            ),
        ),
        VmRequest::EditorCommand { name } => editor_eval(
            vm,
            &format!(
                "EditorSession current handleCommand: {}",
                smalltalk_string_literal(&name)
            ),
        ),
        VmRequest::GetMetrics => vec![VmResponse::Metrics(vm.metrics())],
        VmRequest::SmapplRender { id, code } => {
            // Render the widget image-side (D-G5). A shape that can't be
            // built yet fails cleanly — answer nothing so the placeholder box
            // remains (the swallow-to-fallback contract, `../smappl.md` §2),
            // rather than surfacing the error onto the page.
            match render_and_inject(vm, image, &code) {
                Some(html) => {
                    // Register open outliners in the ToolRegistry so an accept
                    // elsewhere can live-refresh them (docs/APPS.md §7);
                    // replace any prior entry for this widget id (page reload).
                    if html.contains("st-outliner") {
                        OPEN_OUTLINERS.with(|o| {
                            let mut v = o.borrow_mut();
                            v.retain(|(wid, _)| wid != &id);
                            v.push((id.clone(), code.clone()));
                        });
                    }
                    vec![VmResponse::SmapplFragment {
                        html: tag_widget_id(&html, &id),
                        id,
                    }]
                }
                None => vec![],
            }
        }
        VmRequest::SmapplAction { action_id } => {
            // Fire the widget's stored action closure. A String answer is a
            // modal dialog to float over the page (Visual>>promptOk:…); any
            // other answer is a pure side-effect action, so nothing is shown.
            // Transcript output the action makes arrives separately via
            // ChannelTranscript; a failure is echoed to the transcript like a
            // doit rather than dropped.
            match vm.fire_widget_action(&action_id) {
                // A `'macvm-nav:TARGET'` answer is a navigation request (a tour
                // toolbar button); any other String is a modal-dialog overlay.
                Ok(Some(answer)) => match answer.strip_prefix("macvm-nav:") {
                    Some(target) => vec![VmResponse::SmapplNavigate {
                        target: target.to_string(),
                    }],
                    None => vec![VmResponse::SmapplOverlay { html: answer }],
                },
                Ok(None) => vec![],
                Err(e) => vec![VmResponse::Transcript(format!("{e}"))],
            }
        }
        VmRequest::SmapplAccept {
            cls,
            side,
            sel,
            text,
            widget_id,
        } => {
            // Reuse the class browser's proven accept path (BrowserSaveSource):
            // (1) VERSION the edit into image_store — a new method_versions row,
            //     never an overwrite (the user's explicit requirement);
            // (2) mirror it into `world` so the browser view stays consistent;
            // (3) live-compile it into the running VM so it takes effect now.
            let img_side = if side == "class" {
                image_store::Side::Class
            } else {
                image_store::Side::Instance
            };
            let mirror_side = if side == "class" {
                Side::Class
            } else {
                Side::Instance
            };
            world.set_method_source(&cls, mirror_side, &sel, text.clone());
            let versioned = image
                .map(|img| img.set_method_source(&cls, img_side, &sel, &text))
                .transpose();
            // Reopen source needs the real superclass. image_store is the
            // authoritative source (the running VM has it too, but the mock
            // `world` may not — it isn't synced to arbitrary VM classes).
            let superclass = image
                .and_then(|img| img.superclass_of(&cls).ok().flatten())
                .or_else(|| world.class_named(&cls).and_then(|c| c.superclass.clone()))
                .unwrap_or_else(|| "nil".to_string());
            let reopen = format!("{superclass} subclass: {cls} [\n{}\n]\n", text.trim());
            let versioned = match versioned {
                Ok(_) => match live_compile_reopen(vm, &cls, &reopen) {
                    ReopenOutcome::Applied => format!("Accepted {cls}>>{sel} — versioned to the image and live"),
                    ReopenOutcome::NotLoaded => format!(
                        "Accepted {cls}>>{sel} to the image — this VM does not have {cls} loaded, not applied live"
                    ),
                    ReopenOutcome::Failed(e) => format!("Accepted {cls}>>{sel} to the image, but compile failed: {e}"),
                },
                Err(e) => format!("{cls}>>{sel}: image write FAILED (not versioned): {e}"),
            };
            // Live-refresh every OTHER open outliner so the edit shows up
            // across all views on the page (ToolRegistry, docs/APPS.md §7);
            // the edited outliner (`widget_id`) is skipped so its editor stays.
            let mut responses = vec![VmResponse::Transcript(versioned)];
            responses.extend(refresh_open_outliners(vm, image, &widget_id));
            responses
        }
        VmRequest::SmapplNewMethod { cls, side, text } => {
            // Create a NEW method — the same create_or_reopen_method + live
            // compile the class browser's BrowserSaveSource NewMethod arm uses,
            // but stateless (cls/side ride in the message, like SmapplAccept).
            let mirror_side = if side == "class" {
                Side::Class
            } else {
                Side::Instance
            };
            let img_side = side_to_image(mirror_side);
            let category = "as yet unclassified";
            let msg = match image_store::mst::parse_method_selector(&text) {
                None => "Could not parse a message pattern from the method source.".to_string(),
                Some(selector) => {
                    world.create_or_reopen_method(&cls, mirror_side, &selector, category, &text);
                    let image_err = image.and_then(|img| {
                        match img.create_or_reopen_method(&cls, img_side, &selector, category, &text) {
                            Ok(Some(_)) => None,
                            Ok(None) => Some("class not found in the image".to_string()),
                            Err(e) => Some(e.to_string()),
                        }
                    });
                    let reopen = reopen_one_method(world, &cls, &text);
                    match (image_err, live_compile_reopen(vm, &cls, &reopen)) {
                        (None, ReopenOutcome::Applied) => {
                            format!("Added {cls}>>{selector} — versioned to the image and live")
                        }
                        (None, ReopenOutcome::NotLoaded) => format!(
                            "Added {cls}>>{selector} to the image — this VM does not have {cls} loaded, not applied live"
                        ),
                        (None, ReopenOutcome::Failed(e)) => {
                            format!("Added {cls}>>{selector} to the image, but compile failed: {e}")
                        }
                        (Some(ie), _) => format!("{cls}>>{selector}: image write issue: {ie}"),
                    }
                }
            };
            // skip_id "" — refresh EVERY open outliner (including the one that
            // launched this) so the new method shows up in the tree.
            let mut responses = vec![VmResponse::Transcript(msg)];
            responses.extend(refresh_open_outliners(vm, image, ""));
            responses
        }
        VmRequest::SmapplNewClass { text } => {
            // Create a NEW class (+ its methods) — the BrowserSaveSource
            // NewClass arm's logic, stateless. The outliner isn't package-scoped,
            // so the class gets an empty category (the schema's own default).
            let msg = match image_store::mst::parse_mst_source(&text).into_iter().next() {
                None => "Could not parse the class definition — check the syntax.".to_string(),
                Some(pc) if world.class_named(&pc.name).is_some() => {
                    format!("A class named {} already exists.", pc.name)
                }
                Some(pc) => {
                    let new_name = pc.name.clone();
                    world.create_or_reopen_class(
                        &pc.name,
                        pc.superclass.as_deref(),
                        "",
                        "",
                        &pc.instance_vars,
                    );
                    let mut image_err = image.and_then(|img| {
                        match img.create_or_reopen_class(
                            &pc.name,
                            pc.superclass.as_deref(),
                            "",
                            "",
                            &pc.instance_vars,
                        ) {
                            Ok(image_store::ClassCreateOutcome::AlreadyLive) => {
                                Some(format!("image already has a live class named {new_name}"))
                            }
                            Ok(_) => None,
                            Err(e) => Some(e.to_string()),
                        }
                    });
                    let mut failed = 0usize;
                    for m in &pc.methods {
                        let side = if m.is_class_side {
                            Side::Class
                        } else {
                            Side::Instance
                        };
                        world.create_or_reopen_method(
                            &new_name,
                            side,
                            &m.selector,
                            "as yet unclassified",
                            &m.source,
                        );
                        if let Some(img) = image {
                            if img
                                .create_or_reopen_method(
                                    &new_name,
                                    side_to_image(side),
                                    &m.selector,
                                    "as yet unclassified",
                                    &m.source,
                                )
                                .is_err()
                            {
                                failed += 1;
                            }
                        }
                    }
                    if failed > 0 {
                        let note = format!("{failed} method(s) failed to persist to the image");
                        image_err = Some(match image_err {
                            Some(e) => format!("{e}; {note}"),
                            None => note,
                        });
                    }
                    match (image_err, live_compile(vm, &text)) {
                        (None, Ok(())) => {
                            format!("Created {new_name} — versioned to the image and live")
                        }
                        (None, Err(e)) => {
                            format!("Created {new_name} in the image, but compile failed: {e}")
                        }
                        (Some(ie), _) => format!("Created {new_name}, but image issue: {ie}"),
                    }
                }
            };
            let mut responses = vec![VmResponse::Transcript(msg)];
            responses.extend(refresh_open_outliners(vm, image, ""));
            responses
        }
        VmRequest::SmapplAddVar {
            cls,
            is_class_var,
            name,
        } => {
            let name = name.trim().to_string();
            let msg = if name.is_empty() {
                "Enter a variable name.".to_string()
            } else if !is_valid_var_name(&name) {
                format!("'{name}' is not a valid variable name (letters, digits, _).")
            } else if world.class_named(&cls).is_none() {
                format!("No class named {cls}.")
            } else {
                // Union the new name into the class's stored shell (idempotent).
                let (iv, cv): (&str, &str) = if is_class_var {
                    ("", name.as_str())
                } else {
                    (name.as_str(), "")
                };
                let image_err = image.and_then(|img| {
                    img.reimport_class_shell(&cls, iv, cv)
                        .err()
                        .map(|e| e.to_string())
                });
                // Mirror into the mock so the Rust browser (image-based) shows
                // the variable at once — the dual-placement counterpart to the
                // outliner, which reflects the live VM (so an instance var there
                // waits for a restart, but here it appears immediately).
                world.add_var(&cls, is_class_var, name.as_str());
                if is_class_var {
                    // A class variable is a separate association, not part of the
                    // instance shape — reopening the class to append it compiles
                    // cleanly and takes effect immediately, PROVIDED the target VM
                    // already has this class (M5, §4.4): the reopen carries no
                    // ivars, so on a VM that never loaded `cls` it would silently
                    // define a wrong-shape shell instead (§3).
                    let sc = world
                        .class_named(&cls)
                        .and_then(|c| c.superclass.clone())
                        .unwrap_or_else(|| "Object".to_string());
                    let def = format!("{sc} subclass: {cls} [\n    <classVars: {name}>\n]");
                    match (image_err, live_compile_reopen(vm, &cls, &def)) {
                        (None, ReopenOutcome::Applied) => {
                            format!("Added class variable {name} to {cls} — live in the VM.")
                        }
                        (None, ReopenOutcome::NotLoaded) => format!(
                            "Added class variable {name} to {cls} in the image — this VM does not have {cls} loaded, not applied live."
                        ),
                        (None, ReopenOutcome::Failed(e)) => format!(
                            "Added class variable {name} to {cls} in the image, but not live: {e}."
                        ),
                        (Some(ie), _) => {
                            format!("Added class variable {name} to {cls}, but image issue: {ie}.")
                        }
                    }
                } else {
                    // An instance variable grows every instance's shape, which the
                    // running VM's fixed object layout can't do live (no instance
                    // migration) — persisted now, applied on the next VM restart.
                    match image_err {
                        None => format!(
                            "Added instance variable {name} to {cls} — restart the GUI to apply it \
                             to the running VM (instance shape is fixed once a class is defined)."
                        ),
                        Some(ie) => {
                            format!("Added instance variable {name} to {cls}, but image issue: {ie}.")
                        }
                    }
                }
            };
            let mut responses = vec![VmResponse::Transcript(msg)];
            responses.extend(refresh_open_outliners(vm, image, ""));
            // Refresh the browser too, if it's the open view (replace_pane is a
            // no-op when its panes aren't on the page — so this is safe from the
            // outliner as well). The mock was just updated, so the added
            // variable shows in the categories pane immediately.
            responses.push(render_browser_panes(world, selection));
            responses
        }
        VmRequest::SmapplOpenClass {
            cls,
            root,
            widget_id,
        } => {
            let code = format!("ClassOutliner for: (ClassMirror on: {cls})");
            match render_and_inject(vm, image, &code) {
                Some(inner) => {
                    // Wrap with a back link to the hierarchy; the wrapper is
                    // the fragment root the widget-id is stamped on.
                    // `data-hierarchy-root` on the wrapper lets a nested drill
                    // (a subclass link inside this browser, e.g. Boolean→True)
                    // keep the same back-to-hierarchy target.
                    let wrapped = format!(
                        "<div class=\"st-classbrowser\" data-hierarchy-root=\"{root}\">\
                         <div class=\"st-back\">\
                         <span class=\"st-class-link\" data-open-hierarchy=\"{root}\">\
                         &#8249; {root} hierarchy</span></div>{inner}</div>"
                    );
                    OPEN_OUTLINERS.with(|o| {
                        let mut v = o.borrow_mut();
                        v.retain(|(wid, _)| wid != &widget_id);
                        v.push((widget_id.clone(), code));
                    });
                    vec![VmResponse::SmapplFragment {
                        html: tag_widget_id(&wrapped, &widget_id),
                        id: widget_id,
                    }]
                }
                None => vec![VmResponse::Transcript(format!("(cannot open {cls})"))],
            }
        }
        VmRequest::SmapplOpenHierarchy { root, widget_id } => {
            let code = format!("ClassHierarchyOutliner imbeddedVisualForClass: {root}");
            match render_and_inject(vm, image, &code) {
                Some(html) => {
                    OPEN_OUTLINERS.with(|o| {
                        let mut v = o.borrow_mut();
                        v.retain(|(wid, _)| wid != &widget_id);
                        v.push((widget_id.clone(), code));
                    });
                    vec![VmResponse::SmapplFragment {
                        html: tag_widget_id(&html, &widget_id),
                        id: widget_id,
                    }]
                }
                None => vec![VmResponse::Transcript(format!("(cannot open {root} hierarchy)"))],
            }
        }
        VmRequest::Find { tool, query } => {
            let q = query.trim();
            // All three tools are now answered straight from image_store (SQL)
            // — the image is kept in sync with the running VM, so no reflection
            // round trip is needed. Senders reads the `method_sends` index
            // (referenced selectors parsed from source at save/seed time).
            let html = match tool.as_str() {
                "definition" => image
                    .map(|img| render_definition_results(img, q))
                    .unwrap_or_else(find_unavailable),
                "senders" => image
                    .map(|img| render_senders_results(img, q))
                    .unwrap_or_else(find_unavailable),
                _ => image
                    .map(|img| render_implementors_results(img, q))
                    .unwrap_or_else(find_unavailable),
            };
            vec![VmResponse::FindResults { html }]
        }
        VmRequest::FindOptions { tool } => {
            let options = image
                .map(|img| match tool.as_str() {
                    "definition" => img.class_names().unwrap_or_default(),
                    _ => img.all_selectors().unwrap_or_default(),
                })
                .unwrap_or_default();
            vec![VmResponse::FindOptions { tool, options }]
        }
        // Serviced in `worker_loop` (need `world_dir`); never reach `handle`.
        VmRequest::ExportWorld | VmRequest::ImportWorld | VmRequest::EditorAccept => {
            unreachable!("world I/O is handled in worker_loop")
        }
        VmRequest::BrowserOpen => {
            // No longer resets `selection` to default — that made sense
            // when the initial page render was a hardcoded mock (so the
            // worker's state needed to match what was already on screen),
            // but every render now comes from live worker state regardless
            // (`BrowserOpened` vs. `BrowserPanes` is only about *how* the
            // page gets shown, not what data it shows — see
            // `browser_render::assemble_panes`'s doc comment). Preserving
            // `selection` here means refresh, a theme switch, and a
            // font-scale change (all of which route through this same
            // request via `reload_current_page`) keep your package/class/
            // method position instead of silently discarding it. The
            // worker's own initial `BrowserSelection::default()`
            // (`worker_loop`) still covers first-ever open.
            let VmResponse::BrowserPanes {
                packages_html,
                classes_html,
                categories_html,
                methods_html,
                source_html,
            } = render_browser_panes(world, selection)
            else {
                unreachable!("render_browser_panes always returns BrowserPanes")
            };
            vec![VmResponse::BrowserOpened {
                packages_html,
                classes_html,
                categories_html,
                methods_html,
                source_html,
            }]
        }
        VmRequest::BrowserSelectPackage { name } => {
            selection.package = Some(name);
            selection.class = None;
            selection.category = None;
            selection.method = None;
            selection.edit_target = SourceEditTarget::ClassComment;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserSelectClass { name } => {
            selection.class = Some(name);
            selection.category = None;
            selection.method = None;
            selection.edit_target = SourceEditTarget::ClassComment;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserSelectSide { side } => {
            selection.side = side;
            selection.category = None;
            selection.method = None;
            selection.edit_target = SourceEditTarget::ClassComment;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserSelectCategory { name } => {
            // An empty name is the "(all)" pseudo-category
            // (`browser_render.rs`'s `render_categories_pane`) clearing
            // back to "show every method on this side," not a category
            // literally named "".
            selection.category = if name.is_empty() { None } else { Some(name) };
            selection.method = None;
            selection.edit_target = SourceEditTarget::ClassComment;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserSelectMethod { selector } => {
            selection.method = Some(selector);
            selection.edit_target = SourceEditTarget::Method;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserNewMethod => {
            // No longer requires a category to already be selected — a
            // fresh, still-methodless class has none to pick from at all
            // (`render_methods_pane`'s doc comment); `BrowserSaveSource`'s
            // `NewMethod` arm defaults the new method to "as yet
            // unclassified" when `selection.category` is `None`.
            if selection.class.is_some() {
                selection.method = None;
                selection.edit_target = SourceEditTarget::NewMethod;
            }
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserNewClass => {
            // Same "default to the first package" fallback
            // `render_classes_pane` already applies for display — so the
            // "+ New Class" button, shown whenever any package exists, has
            // a real package to work with even before the user's ever
            // explicitly clicked one.
            if selection.package.is_none() {
                selection.package = world.packages().first().cloned();
            }
            selection.class = None;
            selection.category = None;
            selection.method = None;
            selection.edit_target = SourceEditTarget::NewClass;
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserEditDefinition => {
            if selection.class.is_some() {
                selection.method = None;
                selection.edit_target = SourceEditTarget::ClassDefinition;
            }
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserEditComment => {
            if selection.class.is_some() {
                selection.method = None;
                selection.edit_target = SourceEditTarget::ClassComment;
            }
            vec![render_browser_panes(world, selection)]
        }
        VmRequest::BrowserRemoveClass => {
            let message = match selection.class.clone() {
                Some(class_name) => {
                    let removed = world.remove_class(&class_name);
                    let image_error =
                        image.and_then(|img| describe_image_write(img.remove_class(&class_name)));
                    if removed {
                        selection.class = None;
                        selection.category = None;
                        selection.method = None;
                        selection.edit_target = SourceEditTarget::ClassComment;
                        match image_error {
                            None => format!("Removed {class_name}."),
                            Some(reason) => format!("Removed {class_name}, but failed to persist the removal to the image: {reason}."),
                        }
                    } else {
                        "Nothing selected to remove.".to_string()
                    }
                }
                None => "Nothing selected to remove.".to_string(),
            };
            vec![
                render_browser_panes(world, selection),
                VmResponse::Transcript(message),
            ]
        }
        VmRequest::BrowserRemoveMethod => {
            let message = match (selection.class.clone(), selection.method.clone()) {
                (Some(class_name), Some(selector)) => {
                    let removed = world.remove_method(&class_name, selection.side, &selector);
                    let image_error = image.and_then(|img| {
                        describe_image_write(img.remove_method(
                            &class_name,
                            side_to_image(selection.side),
                            &selector,
                        ))
                    });
                    if removed {
                        selection.method = None;
                        selection.edit_target = SourceEditTarget::ClassComment;
                        match image_error {
                            None => format!("Removed {class_name}>>{selector}."),
                            Some(reason) => format!("Removed {class_name}>>{selector}, but failed to persist the removal to the image: {reason}."),
                        }
                    } else {
                        "Nothing selected to remove.".to_string()
                    }
                }
                _ => "Nothing selected to remove.".to_string(),
            };
            vec![
                render_browser_panes(world, selection),
                VmResponse::Transcript(message),
            ]
        }
        VmRequest::BrowserSaveSource {
            text,
            saved_package,
            saved_class,
            saved_side,
            saved_category,
            saved_method,
            saved_target,
        } => {
            // The source pane's `data-save-*` snapshot (`browser_render.rs`)
            // must still match what's actually selected right now — a
            // selection-change request queued between the render this came
            // from and this save arriving would otherwise apply the edited
            // text to whatever's selected *now* instead of what was on
            // screen when the user hit Cmd+S. Reject rather than guess:
            // the panes still get re-rendered (showing current reality),
            // but the stale edit is simply dropped, same as any other
            // unsaved-edit-discarded-by-navigation case.
            let snapshot = BrowserSelection::from_wire(
                &saved_package,
                &saved_class,
                &saved_side,
                &saved_category,
                &saved_method,
                &saved_target,
            );
            if snapshot != *selection {
                return vec![
                    render_browser_panes(world, selection),
                    VmResponse::Transcript(
                        "Selection changed since this edit was opened — not saved.".to_string(),
                    ),
                ];
            }

            // Write through to both: `world` (the in-memory mirror
            // `browser_render.rs` reads) for immediate display, and the
            // real `image` (if `MACVM_IMAGE_PATH` gave us one) so the edit
            // actually survives a restart — `docs/IMAGE.md` §8. The mirror
            // save is authoritative for *display* either way (it's what
            // gets re-rendered right below), so a mirror/image mismatch
            // can't leave the browser showing something that isn't there —
            // but it can mean the edit doesn't survive a restart, which is
            // now reported honestly (`SaveOk`/`describe_image_write`)
            // rather than unconditionally claiming "(persisted to image)".
            // S22-E: remember the edit KIND before the arms below mutate
            // `selection.edit_target` (NewMethod→Method, NewClass→ClassComment),
            // so the post-match live-compile knows how to make it live.
            let edit_kind = selection.edit_target.clone();
            let outcome: Result<SaveOk, String> = match &selection.edit_target {
                SourceEditTarget::ClassComment => match &selection.class {
                    Some(class_name) if world.set_class_comment(class_name, text.clone()) => {
                        let image_error = image.and_then(|img| describe_image_write(img.set_class_comment(class_name, &text)));
                        Ok(SaveOk { what: format!("Saved {class_name} comment"), image_error })
                    }
                    _ => Err("Nothing selected to save.".to_string()),
                },
                SourceEditTarget::Method => match (&selection.class, &selection.method) {
                    (Some(class_name), Some(selector)) if world.set_method_source(class_name, selection.side, selector, text.clone()) => {
                        let image_error =
                            image.and_then(|img| describe_image_write(img.set_method_source(class_name, side_to_image(selection.side), selector, &text)));
                        Ok(SaveOk { what: format!("Saved {class_name}>>{selector}"), image_error })
                    }
                    _ => Err("Nothing selected to save.".to_string()),
                },
                SourceEditTarget::ClassDefinition => match &selection.class {
                    Some(class_name) => match image_store::mst::parse_mst_source(&text).into_iter().next() {
                        Some(pc) if pc.name == *class_name => {
                            world.set_class_definition(class_name, pc.superclass.clone(), pc.instance_vars.clone());
                            let image_error = image
                                .and_then(|img| describe_image_write(img.set_class_definition(class_name, pc.superclass.as_deref(), &pc.instance_vars)));
                            Ok(SaveOk { what: format!("Saved {class_name} definition"), image_error })
                        }
                        Some(pc) => Err(format!("Definition names \"{}\", not \"{class_name}\" — renaming a class isn't supported here.", pc.name)),
                        None => Err("Could not parse the class definition — check the syntax.".to_string()),
                    },
                    None => Err("Nothing selected to save.".to_string()),
                },
                SourceEditTarget::NewMethod => match selection.class.clone() {
                    Some(class_name) => {
                        // No category selected means "(all)" was showing
                        // (`render_categories_pane`) — default the new
                        // method to "as yet unclassified" (the schema's own
                        // default for an uncategorized method) rather than
                        // requiring one to be picked first. Deliberately
                        // *not* overwritten into `selection.category` on
                        // success: leaving the filter exactly as it was
                        // means an "(all)" view stays showing everything
                        // (including the new method), and an explicit
                        // category stays narrowed to what it already was.
                        let category = selection.category.clone().unwrap_or_else(|| "as yet unclassified".to_string());
                        match image_store::mst::parse_method_selector(&text) {
                            Some(selector) => {
                                let created = world.create_or_reopen_method(&class_name, selection.side, &selector, &category, &text);
                                let image_error = image.and_then(|img| {
                                    match img.create_or_reopen_method(&class_name, side_to_image(selection.side), &selector, &category, &text) {
                                        Ok(Some(_)) => {
                                            // M3 (docs/package_aware_editing_design.md §4.2): only
                                            // assign a home if this exact method has none yet — an
                                            // edit of an already-imported method must keep its real
                                            // source_file, never the synthetic marker.
                                            let has_home = img
                                                .method_source_file(&class_name, side_to_image(selection.side), &selector)
                                                .ok()
                                                .flatten()
                                                .is_some();
                                            if !has_home {
                                                let _ = img.set_method_home_file(
                                                    &class_name,
                                                    side_to_image(selection.side),
                                                    &selector,
                                                    image_store::flows::INTERACTIVE_SOURCE_FILE,
                                                );
                                            }
                                            None
                                        }
                                        Ok(None) => Some("class not found in the image (mirror/image out of sync?)".to_string()),
                                        Err(e) => Some(e.to_string()),
                                    }
                                });
                                if created {
                                    selection.method = Some(selector.clone());
                                    selection.edit_target = SourceEditTarget::Method;
                                    Ok(SaveOk { what: format!("Saved {class_name}>>{selector}"), image_error })
                                } else {
                                    Err("Nothing selected to save.".to_string())
                                }
                            }
                            None => Err("Could not parse a message pattern from the method source.".to_string()),
                        }
                    }
                    None => Err("Select a class first.".to_string()),
                },
                SourceEditTarget::NewClass => match selection.package.clone() {
                    Some(package) => match image_store::mst::parse_mst_source(&text).into_iter().next() {
                        Some(pc) if world.class_named(&pc.name).is_some() => Err(format!("A class named {} already exists.", pc.name)),
                        Some(pc) => {
                            let new_name = pc.name.clone();
                            world.create_or_reopen_class(&pc.name, pc.superclass.as_deref(), &package, "", &pc.instance_vars);
                            let mut image_error = image.and_then(|img| {
                                match img.create_or_reopen_class(&pc.name, pc.superclass.as_deref(), &package, "", &pc.instance_vars) {
                                    Ok(image_store::ClassCreateOutcome::AlreadyLive) => {
                                        Some(format!("image already has a live class named {new_name} (mirror/image out of sync?)"))
                                    }
                                    Ok(_) => None,
                                    Err(e) => Some(e.to_string()),
                                }
                            });
                            let mut failed_methods = 0usize;
                            for m in &pc.methods {
                                let side = if m.is_class_side { Side::Class } else { Side::Instance };
                                world.create_or_reopen_method(&new_name, side, &m.selector, "as yet unclassified", &m.source);
                                if let Some(img) = image {
                                    if img.create_or_reopen_method(&new_name, side_to_image(side), &m.selector, "as yet unclassified", &m.source).is_err() {
                                        failed_methods += 1;
                                    } else {
                                        // M3 (docs/package_aware_editing_design.md §4.2): every
                                        // method of a BRAND NEW class is unconditionally new, so
                                        // (unlike NewMethod's edit-or-create case) no "already
                                        // homed" check is needed here.
                                        let _ = img.set_method_home_file(
                                            &new_name,
                                            side_to_image(side),
                                            &m.selector,
                                            image_store::flows::INTERACTIVE_SOURCE_FILE,
                                        );
                                    }
                                }
                            }
                            if failed_methods > 0 {
                                let note = format!("{failed_methods} method(s) failed to persist to the image");
                                image_error = Some(match image_error {
                                    Some(existing) => format!("{existing}; {note}"),
                                    None => note,
                                });
                            }
                            selection.package = Some(package);
                            selection.class = Some(new_name.clone());
                            selection.category = None;
                            selection.method = None;
                            selection.edit_target = SourceEditTarget::ClassComment;
                            Ok(SaveOk { what: format!("Created {new_name}"), image_error })
                        }
                        None => Err("Could not parse the class definition — check the syntax.".to_string()),
                    },
                    None => Err("Select a package first.".to_string()),
                },
            };
            // S22-E: make a SUCCESSFUL edit LIVE in the running VM — not just
            // saved to the mock + image. `text` is the edited source; the
            // (post-edit) `selection` names the class. A ClassComment edit
            // compiles nothing (`None`). A live-compile failure is reported
            // but does NOT undo the save — the source is persisted; the user
            // fixes and re-accepts.
            // M5 (§4.4): Method/NewMethod is a REOPEN (no ivars) — gated on the
            // VM already having the class, same as SmapplAccept/SmapplNewMethod.
            // ClassDefinition/NewClass ships a full definition (real ivars via
            // `text`) — never gated, same as `acceptClass`: it either resolves
            // correctly on a VM seeing it for the first time or fails an honest
            // compile error, never the silent-wrong-shape failure §3 describes.
            let vm_live: Option<ReopenOutcome> = match (&outcome, &edit_kind) {
                (Ok(_), SourceEditTarget::Method | SourceEditTarget::NewMethod) => selection
                    .class
                    .clone()
                    .map(|c| live_compile_reopen(vm, &c, &reopen_one_method(world, &c, &text))),
                (Ok(_), SourceEditTarget::ClassDefinition | SourceEditTarget::NewClass) => {
                    Some(match live_compile(vm, &text) {
                        Ok(()) => ReopenOutcome::Applied,
                        Err(e) => ReopenOutcome::Failed(e),
                    })
                }
                _ => None,
            };

            let message = match outcome {
                Ok(SaveOk { what, image_error }) => {
                    let base = match (image.is_some(), image_error) {
                        (true, None) => format!("{what} (persisted to image)"),
                        (_, None) => what,
                        (_, Some(reason)) => {
                            format!("{what}, but failed to persist to the image: {reason}")
                        }
                    };
                    match vm_live {
                        Some(ReopenOutcome::Applied) => format!("{base} — live in the VM."),
                        Some(ReopenOutcome::NotLoaded) => {
                            format!("{base}; this VM does not have it loaded, not applied live.")
                        }
                        Some(ReopenOutcome::Failed(e)) => {
                            format!("{base}; saved but NOT live in the VM: {e}.")
                        }
                        None => format!("{base}."),
                    }
                }
                Err(e) => e,
            };
            vec![
                render_browser_panes(world, selection),
                VmResponse::Transcript(message),
            ]
        }
        VmRequest::WorkspacePrintIt { code } => {
            // "Print it": insert the real printString inline right after
            // the selection (`main.rs`'s `WorkspacePrintResult` handler) —
            // `eval`'s `Ok(String)` already IS that printString (e.g. a
            // Smalltalk String's own printString already carries its own
            // quotes), so it's inserted as-is, not re-quoted.
            let text = match vm.eval(&code) {
                Ok(result) => format!(" {result}"),
                Err(e) => format!(" \"ERROR: {e}\""),
            };
            vec![VmResponse::WorkspacePrintResult { text }]
        }
        VmRequest::CanvasCreate { width, height } => {
            vec![VmResponse::CanvasCreated {
                id: CANVAS_ID,
                width,
                height,
            }]
        }
        VmRequest::CanvasRunDemo => {
            vec![VmResponse::CanvasDraw {
                id: CANVAS_ID,
                commands_json: canvas_demo_commands().to_json(),
            }]
        }
        VmRequest::CanvasEval { code } => {
            // Real Smalltalk builds the whole batch (docs/CANVAS.md §5.2); Rust
            // only transports the string. Generic — the worker neither knows
            // nor cares what `code` draws. A compute failure is echoed to the
            // transcript like a doit, not dropped. The wall-clock compute time
            // is reported to the transcript (matching the Strongtalk Mandelbrot
            // demo, which prints its compute time in ms) — generic, so ANY
            // canvas drawing gets a timing readout, not just the Mandelbrot.
            let started = std::time::Instant::now();
            match vm.eval_to_string(&code) {
                Ok(commands_json) => {
                    let ms = started.elapsed().as_millis();
                    vec![
                        VmResponse::CanvasDraw {
                            id: CANVAS_ID,
                            commands_json,
                        },
                        VmResponse::Transcript(format!("canvas: computed in {ms} ms")),
                    ]
                }
                Err(e) => vec![VmResponse::Transcript(format!("canvas eval failed: {e}"))],
            }
        }
        VmRequest::CanvasEvalPixels {
            code,
            width,
            height,
        } => {
            // Real Smalltalk fills an RGBA Pixmap; `code` answers its raw
            // ByteArray. Rust base64s the bytes for the evaluateJavaScript hop
            // (module doc) — no vector command batch, no per-pixel strings.
            // Wall-clock compute time reported to the transcript, same as
            // CanvasEval. A short/failed buffer degrades to a transcript note.
            let started = std::time::Instant::now();
            match vm.eval_to_bytes(&code) {
                Ok(bytes) => {
                    let ms = started.elapsed().as_millis();
                    let expected = (width as usize) * (height as usize) * 4;
                    if bytes.len() != expected {
                        vec![VmResponse::Transcript(format!(
                            "canvas pixels: expected {expected} bytes for {width}x{height}, got {}",
                            bytes.len()
                        ))]
                    } else {
                        vec![
                            VmResponse::CanvasPixels {
                                id: CANVAS_ID,
                                width,
                                height,
                                base64: base64_encode(&bytes),
                            },
                            VmResponse::Transcript(format!("canvas: computed in {ms} ms")),
                        ]
                    }
                }
                Err(e) => vec![VmResponse::Transcript(format!("canvas pixels failed: {e}"))],
            }
        }
        VmRequest::CanvasClear => {
            let mut cmds = CanvasCommands::new();
            cmds.call(
                "clearRect",
                &[0.0.into(), 0.0.into(), 10_000.0.into(), 10_000.0.into()],
            );
            vec![
                VmResponse::CanvasCleared { id: CANVAS_ID },
                VmResponse::CanvasDraw {
                    id: CANVAS_ID,
                    commands_json: cmds.to_json(),
                },
            ]
        }
    }
}

/// v1's only canvas (`docs/CANVAS.md` §7: multiple canvases deferred).
const CANVAS_ID: u32 = 0;

/// A small ergonomic builder for a canvas command batch
/// (`VmResponse::CanvasDraw`'s `commands_json`, `docs/CANVAS.md` §5.2) —
/// stands in for how the real Smalltalk-side `Canvas` would accumulate
/// calls before flushing (§3), just in Rust for the demo/test path here.
/// Hand-writes JSON directly rather than pulling in a serde dependency
/// for a handful of numbers and strings.
#[derive(Default)]
pub struct CanvasCommands {
    commands: Vec<String>,
}

/// One command argument — canvas ops take a mix of numbers (coordinates,
/// widths) and strings (colors, fonts, text), so a single `call` needs a
/// heterogeneous arg list rather than separate numeric/string methods.
pub enum CanvasArg {
    Num(f64),
    Str(String),
}

impl From<f64> for CanvasArg {
    fn from(v: f64) -> Self {
        CanvasArg::Num(v)
    }
}

impl From<&str> for CanvasArg {
    fn from(v: &str) -> Self {
        CanvasArg::Str(v.to_string())
    }
}

impl CanvasCommands {
    pub fn new() -> Self {
        Self::default()
    }

    /// `name` is either a `CanvasRenderingContext2D` method (called as
    /// `ctx[name](...args)`) or a property (assigned as `ctx[name] =
    /// args[0]`) — `smtk.js`'s `macvmCanvasDraw` decides which via its own
    /// two allowlists; this builder doesn't need to know which is which.
    pub fn call(&mut self, name: &str, args: &[CanvasArg]) -> &mut Self {
        let mut parts = vec![json_string(name)];
        for a in args {
            parts.push(match a {
                CanvasArg::Num(n) => n.to_string(),
                CanvasArg::Str(s) => json_string(s),
            });
        }
        self.commands.push(format!("[{}]", parts.join(",")));
        self
    }

    pub fn to_json(&self) -> String {
        format!("[{}]", self.commands.join(","))
    }
}

/// Standard base64 (RFC 4648, `+`/`/`, `=` padding) — hand-rolled to avoid a
/// dependency for the one place bulk binary crosses `evaluateJavaScript` (a
/// `CanvasPixels` RGBA buffer). `atob` on the JS side decodes it.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) & 63] as char);
        out.push(ALPHABET[(n >> 12) & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Minimal JSON string escaping — only what this fixed command/color/text
/// vocabulary ever actually needs (quotes, backslashes), not a general-
/// purpose JSON encoder.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// "Run Demo"'s fixed content — a filled rectangle, a stroked circle, a
/// line, and a text label, deliberately exercising one op from each
/// vocabulary group (`docs/CANVAS.md` §5.2: paint, path, state/transform,
/// text, properties) so a visual check of this one demo is a real check
/// of the whole interpreter, not just one code path through it.
fn canvas_demo_commands() -> CanvasCommands {
    let mut c = CanvasCommands::new();
    c.call(
        "clearRect",
        &[0.0.into(), 0.0.into(), 10_000.0.into(), 10_000.0.into()],
    );
    c.call("fillStyle", &["steelblue".into()]);
    c.call(
        "fillRect",
        &[20.0.into(), 20.0.into(), 120.0.into(), 80.0.into()],
    );
    c.call("strokeStyle", &["crimson".into()]);
    c.call("lineWidth", &[3.0.into()]);
    c.call("beginPath", &[]);
    c.call(
        "arc",
        &[
            220.0.into(),
            60.0.into(),
            40.0.into(),
            0.0.into(),
            (std::f64::consts::PI * 2.0).into(),
        ],
    );
    c.call("stroke", &[]);
    c.call("save", &[]);
    c.call("strokeStyle", &["seagreen".into()]);
    c.call("lineWidth", &[2.0.into()]);
    c.call("beginPath", &[]);
    c.call("moveTo", &[20.0.into(), 140.0.into()]);
    c.call("lineTo", &[300.0.into(), 140.0.into()]);
    c.call("stroke", &[]);
    c.call("restore", &[]);
    c.call("fillStyle", &["#222".into()]);
    c.call("font", &["16px sans-serif".into()]);
    c.call(
        "fillText",
        &["MACVM Canvas".into(), 20.0.into(), 175.0.into()],
    );
    c
}

fn render_browser_panes(world: &MockWorld, selection: &BrowserSelection) -> VmResponse {
    VmResponse::BrowserPanes {
        packages_html: browser_render::render_packages_pane(world, selection),
        classes_html: browser_render::render_classes_pane(world, selection),
        categories_html: browser_render::render_categories_pane(world, selection),
        methods_html: browser_render::render_methods_pane(world, selection),
        source_html: browser_render::render_source_pane(world, selection),
    }
}

/// A single-value mailbox where publishing overwrites whatever's there,
/// read or not — the "latest wins" counterpart to the ordered `mpsc`
/// channel above, for continuous/high-frequency updates (a redrawing pixel
/// buffer, a live status line) where a slow consumer should skip straight
/// to the newest value instead of working through a backlog of stale ones.
/// See this module's doc comment for which response kind wants which
/// discipline.
pub struct LatestSlot<T> {
    value: Mutex<Option<T>>,
}

impl<T> LatestSlot<T> {
    pub fn new() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }

    /// Replace whatever value was there. The old one (if any, if unread)
    /// is simply dropped — that's the point.
    pub fn publish(&self, value: T) {
        *self.value.lock().unwrap() = Some(value);
    }

    /// Take the current value, leaving the slot empty. Returns `None` if
    /// nothing's been published since the last `take`.
    pub fn take(&self) -> Option<T> {
        self.value.lock().unwrap().take()
    }
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `world/` directory, resolved via `CARGO_MANIFEST_DIR`
    /// rather than a bare relative path: `cargo test`'s working directory
    /// is this crate's own root (`gui/`), not the workspace root `world/`
    /// actually lives under.
    fn test_world_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../world"))
    }

    /// A real, booted `VmHandle` (heap kept small for a test — the GUI's
    /// own worker uses `VmOptions::from_env`'s larger default) for tests
    /// that exercise `handle()`'s arms directly.
    fn test_vm_handle(jit: macvm::runtime::JitMode) -> VmHandle {
        VmHandle::boot(
            macvm::runtime::VmOptions {
                heap_mib: 64,
                jit,
                ..Default::default()
            },
            &test_world_dir(),
        )
        .expect("boot against the real world/ directory must succeed")
    }

    #[test]
    fn browser_select_category_treats_empty_name_as_clearing_to_all() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection {
            class: Some("Object".to_string()),
            ..Default::default()
        };
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        handle(
            VmRequest::BrowserSelectCategory {
                name: "printing".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert_eq!(selection.category.as_deref(), Some("printing"));

        handle(
            VmRequest::BrowserSelectCategory {
                name: String::new(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert_eq!(selection.category, None);
    }

    #[test]
    fn browser_new_method_only_requires_a_class_not_a_category() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection {
            class: Some("Object".to_string()),
            ..Default::default()
        };
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        assert_eq!(selection.category, None); // "(all)" — no category picked
        handle(
            VmRequest::BrowserNewMethod,
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert_eq!(selection.edit_target, SourceEditTarget::NewMethod);
    }

    #[test]
    fn browser_save_source_rejects_a_stale_snapshot() {
        let mut world = MockWorld::seed();
        // Selection has already moved on to "hash" by the time this save
        // arrives, but the snapshot (as if captured when "printString" was
        // still open) still names "printString" — exactly the race a
        // selection-change request queued ahead of a Cmd+S creates.
        let mut selection = BrowserSelection {
            class: Some("Object".to_string()),
            category: Some("comparing".to_string()),
            method: Some("hash".to_string()),
            edit_target: SourceEditTarget::Method,
            ..Default::default()
        };
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::BrowserSaveSource {
                text: "printString\n\t^'HIJACKED'".to_string(),
                saved_package: String::new(),
                saved_class: "Object".to_string(),
                saved_side: "instance".to_string(),
                saved_category: "printing".to_string(),
                saved_method: "printString".to_string(),
                saved_target: "method".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        // Neither method was touched — not hash (what's *currently*
        // selected) and not printString (what the save actually said it
        // was for).
        assert_eq!(
            world
                .method_source("Object", Side::Instance, "hash")
                .unwrap(),
            "hash\n\t^self identityHash"
        );
        assert!(world
            .method_source("Object", Side::Instance, "printString")
            .unwrap()
            .starts_with("printString\n\t^String"));
        let message = responses
            .iter()
            .find_map(|r| match r {
                VmResponse::Transcript(t) => Some(t.clone()),
                _ => None,
            })
            .expect("expected a transcript message");
        assert!(message.contains("Selection changed"), "{message}");
    }

    #[test]
    fn browser_save_source_applies_when_snapshot_matches_current_selection() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection {
            class: Some("Object".to_string()),
            category: Some("comparing".to_string()),
            method: Some("hash".to_string()),
            edit_target: SourceEditTarget::Method,
            ..Default::default()
        };
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        handle(
            VmRequest::BrowserSaveSource {
                text: "hash\n\t^42".to_string(),
                saved_package: String::new(),
                saved_class: "Object".to_string(),
                saved_side: "instance".to_string(),
                saved_category: "comparing".to_string(),
                saved_method: "hash".to_string(),
                saved_target: "method".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert_eq!(
            world
                .method_source("Object", Side::Instance, "hash")
                .unwrap(),
            "hash\n\t^42"
        );
    }

    #[test]
    fn browser_save_source_reports_image_write_failure_honestly() {
        let mut world = MockWorld::empty();
        world.add_class("Ghost", None, "Test", "old comment", "", "");
        // Deliberately never added to `image` — simulates mirror/image
        // drift (handle()'s own codepaths always keep both in sync, but
        // this exercises the defensive honesty check regardless of how
        // drift could happen).
        let image = image_store::Image::open_in_memory().unwrap();
        let mut selection = BrowserSelection {
            class: Some("Ghost".to_string()),
            edit_target: SourceEditTarget::ClassComment,
            ..Default::default()
        };
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::BrowserSaveSource {
                text: "new comment".to_string(),
                saved_package: String::new(),
                saved_class: "Ghost".to_string(),
                saved_side: "instance".to_string(),
                saved_category: String::new(),
                saved_method: String::new(),
                saved_target: "comment".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        // The mirror save still succeeds — immediate display is unaffected...
        assert_eq!(world.class_named("Ghost").unwrap().comment, "new comment");
        // ...but the transcript must not falsely claim it reached the image.
        let message = responses
            .iter()
            .find_map(|r| match r {
                VmResponse::Transcript(t) => Some(t.clone()),
                _ => None,
            })
            .expect("expected a transcript message");
        assert!(!message.contains("(persisted to image)"), "{message}");
        assert!(message.contains("failed to persist"), "{message}");
    }

    #[test]
    fn canvas_commands_builds_a_json_array_of_op_arrays() {
        let mut cmds = CanvasCommands::new();
        cmds.call("beginPath", &[]);
        cmds.call("lineTo", &[20.0.into(), 140.0.into()]);
        cmds.call("fillStyle", &["steelblue".into()]);
        assert_eq!(
            cmds.to_json(),
            r#"[["beginPath"],["lineTo",20,140],["fillStyle","steelblue"]]"#
        );
    }

    #[test]
    fn json_string_escapes_quotes_and_backslashes() {
        assert_eq!(json_string("plain"), "\"plain\"");
        assert_eq!(json_string(r#"say "hi" \ ok"#), r#""say \"hi\" \\ ok""#);
    }

    #[test]
    fn canvas_create_echoes_back_the_requested_size_under_the_fixed_v1_id() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasCreate {
                width: 640,
                height: 480,
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(matches!(
            responses.as_slice(),
            [VmResponse::CanvasCreated {
                id: 0,
                width: 640,
                height: 480
            }]
        ));
    }

    #[test]
    fn canvas_run_demo_returns_one_nonempty_draw_batch_for_the_fixed_canvas() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasRunDemo,
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::CanvasDraw { id, commands_json }] = responses.as_slice() else {
            panic!("expected exactly one CanvasDraw response");
        };
        assert_eq!(*id, 0);
        assert!(
            commands_json.starts_with('[') && commands_json.ends_with(']'),
            "{commands_json}"
        );
        assert!(commands_json.contains("fillRect"), "{commands_json}");
        assert!(commands_json.contains("MACVM Canvas"), "{commands_json}");
    }

    #[test]
    fn canvas_eval_runs_real_smalltalk_and_draws_its_command_batch() {
        // The generic vector path (docs/CANVAS.md §5.2): arbitrary Smalltalk
        // answers a command-batch String, drawn as a CanvasDraw. A String
        // literal answers itself, so the GUI holds no per-drawing knowledge.
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasEval {
                code: "'[[\"fillRect\",0,0,10,10]]'".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::CanvasDraw { id, commands_json }, VmResponse::Transcript(timing)] =
            responses.as_slice()
        else {
            panic!("expected [CanvasDraw, Transcript(timing)], got {responses:?}");
        };
        assert_eq!(*id, 0);
        assert_eq!(commands_json, "[[\"fillRect\",0,0,10,10]]");
        assert!(
            timing.contains("ms"),
            "the compute time must be reported to the transcript, got {timing:?}"
        );
    }

    #[test]
    fn canvas_eval_pixels_blits_a_real_smalltalk_pixmap() {
        // The generic pixel path (docs/CANVAS.md): Smalltalk fills a Pixmap and
        // answers its RGBA ByteArray; the worker base64s it into CanvasPixels.
        // Here the Mandelbrot — the GUI just forwarded the code + canvas size.
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasEvalPixels {
                code: "Mandelbrot new pixelsForWidth: 40 height: 30".to_string(),
                width: 40,
                height: 30,
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::CanvasPixels {
            id,
            width,
            height,
            base64,
        }, VmResponse::Transcript(timing)] = responses.as_slice()
        else {
            panic!("expected [CanvasPixels, Transcript(timing)], got {responses:?}");
        };
        assert_eq!((*id, *width, *height), (0, 40, 30));
        // base64 of 40*30*4 = 4800 bytes → 6400 chars (4800 is divisible by 3,
        // so no padding).
        assert_eq!(base64.len(), 6400, "base64 length must match 40x30 RGBA");
        assert!(timing.contains("ms"), "must report compute time, got {timing:?}");
    }

    #[test]
    fn canvas_eval_pixels_reports_a_size_mismatch_instead_of_blitting() {
        // If the answered buffer isn't width*height*4, degrade to a transcript
        // note rather than putImageData a wrong-sized buffer.
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasEvalPixels {
                code: "Mandelbrot new pixelsForWidth: 40 height: 30".to_string(),
                width: 41, // deliberately wrong
                height: 30,
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(
            matches!(responses.as_slice(), [VmResponse::Transcript(t)] if t.contains("expected")),
            "a size mismatch must surface as a transcript note, got {responses:?}"
        );
    }

    #[test]
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(&[0, 255, 0, 255]), "AP8A/w==");
    }

    #[test]
    fn canvas_eval_reports_a_bad_expression_to_the_transcript_not_a_draw() {
        // A generic eval that can't produce a batch must degrade to a
        // transcript note, never a broken draw or a panic.
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasEval {
                code: "42".to_string(), // answers an Integer, not a batch String
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(
            matches!(responses.as_slice(), [VmResponse::Transcript(t)] if t.contains("canvas eval failed")),
            "a non-String answer must surface as a transcript note, got {responses:?}"
        );
    }

    #[test]
    fn canvas_clear_answers_cleared_then_a_full_canvas_clear_rect_batch() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::CanvasClear,
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::CanvasCleared { id: cleared_id }, VmResponse::CanvasDraw {
            id: draw_id,
            commands_json,
        }] = responses.as_slice()
        else {
            panic!("expected [CanvasCleared, CanvasDraw]");
        };
        assert_eq!(*cleared_id, 0);
        assert_eq!(*draw_id, 0);
        assert!(commands_json.contains("clearRect"), "{commands_json}");
    }

    #[test]
    fn latest_slot_overwrites_unread_values() {
        let slot = LatestSlot::new();
        slot.publish(1);
        slot.publish(2);
        slot.publish(3);
        // Only the most recent publish survives — 1 and 2 were never read.
        assert_eq!(slot.take(), Some(3));
        assert_eq!(slot.take(), None);
    }

    #[test]
    fn latest_slot_read_then_publish_then_read() {
        let slot = LatestSlot::new();
        slot.publish("frame-1");
        assert_eq!(slot.take(), Some("frame-1"));
        assert_eq!(slot.take(), None);
        slot.publish("frame-2");
        assert_eq!(slot.take(), Some("frame-2"));
    }

    // ── S21 step 3: real VmHandle wiring + restart-on-death ───────────────

    fn transcript_containing(rs: &[VmResponse], needle: &str) -> bool {
        rs.iter()
            .any(|r| matches!(r, VmResponse::Transcript(t) if t.contains(needle)))
    }

    /// G2 smappl slice: a `<smappl>` render request evaluates the `visual=`
    /// code and answers a live button fragment; the fragment's advertised
    /// action id fires the widget's closure; an unbuildable shape answers no
    /// fragment (the placeholder box stays). Exercises the worker seam
    /// (`handle`) end to end, the same path `main.rs` drives.
    #[test]
    fn smappl_render_and_action_round_trip_through_handle() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);

        let rendered = handle(
            VmRequest::SmapplRender {
                id: "s0".to_string(),
                code: "Button labeled: 'Hi' action: [ :b | Transcript show: 'fired' ]".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let (id, html) = match rendered.as_slice() {
            [VmResponse::SmapplFragment { id, html }] => (id.clone(), html.clone()),
            other => panic!("expected one SmapplFragment, got {other:?}"),
        };
        assert_eq!(id, "s0", "the answer must carry back the placeholder's id");
        assert!(
            html.contains("smappl-button") && html.contains("Hi"),
            "must render a live beveled button carrying its label, got {html:?}"
        );

        // The action id the fragment advertises fires the stored closure.
        let marker = "data-widget-action=\"";
        let start = html.find(marker).expect("fragment has an action id") + marker.len();
        let end = html[start..].find('"').unwrap() + start;
        let action_id = html[start..end].to_string();
        let fired = handle(
            VmRequest::SmapplAction { action_id },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        // A clean fire produces no UI response of its own (its Transcript
        // output goes to the sink, absent in this direct test) — the point is
        // it ran without surfacing an error back.
        assert!(
            fired.is_empty(),
            "a clean action fire returns no error responses, got {fired:?}"
        );

        // A tour-page toolbar button (its action answers a 'macvm-nav:…' marker
        // via the Launcher stubs) fires to a SmapplNavigate the GUI routes
        // through its toolbar dispatch.
        let nav = handle(
            VmRequest::SmapplRender {
                id: "sn".to_string(),
                code: "Button labeled: 'H' action: [ :b | Launcher launchers anElement browseStartPage ]"
                    .to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let nav_html = match nav.as_slice() {
            [VmResponse::SmapplFragment { html, .. }] => html.clone(),
            other => panic!("expected nav button fragment, got {other:?}"),
        };
        let ns = nav_html.find(marker).expect("nav fragment has an action id") + marker.len();
        let ne = nav_html[ns..].find('"').unwrap() + ns;
        let nav_fired = handle(
            VmRequest::SmapplAction {
                action_id: nav_html[ns..ne].to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(
            matches!(nav_fired.as_slice(), [VmResponse::SmapplNavigate { target }] if target == "home"),
            "a tour toolbar button fires to a navigation request, got {nav_fired:?}"
        );

        // A shape that genuinely isn't built (an unknown send) renders nothing,
        // so the GUI keeps the G0 placeholder box.
        //
        // macOS-only for now: reaching this asserts that the DNU the unknown
        // send raises is CAUGHT — that is guest-fatal recovery (`siglongjmp`
        // back to `eval`'s `sigsetjmp`), and on Windows both are still stubs,
        // so the abort would take the whole test binary down rather than fail
        // this assertion. See `worker_survives_an_unhandled_runtime_error_
        // and_serves_the_next_request` for the full note and `../MIGRATION.md`
        // §6 for the tracked follow-up. The rest of this round trip is
        // platform-neutral and does run on Windows.
        #[cfg(target_os = "macos")]
        {
            let unbuildable = handle(
                VmRequest::SmapplRender {
                    id: "s1".to_string(),
                    code: "Object doesNotExistXyz".to_string(),
                },
                &mut world,
                &mut selection,
                None,
                &mut vm,
            );
            assert!(
                unbuildable.is_empty(),
                "an unbuildable shape must yield no fragment, got {unbuildable:?}"
            );
        }

        // CodeView (gui/smappl.md §3.5) IS built now — it renders a source box,
        // with the source HTML-escaped once (a `<` in the source → `&lt;`).
        let codeview = handle(
            VmRequest::SmapplRender {
                id: "s2".to_string(),
                code: "(CodeView forString) model: 'a < b'".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(
            codeview.iter().any(|r| matches!(r, VmResponse::SmapplFragment { html, .. }
                if html.contains("st-smappl-codeview") && html.contains("a &lt; b"))),
            "CodeView renders a source box with escaped source, got {codeview:?}"
        );
    }

    /// The Implementors find tool lists every class that defines the selector,
    /// each a drill-link into that class's browser.
    #[test]
    fn find_implementors_lists_classes_that_define_the_selector() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        // Implementors is now answered from image_store (SQL), so the find tools
        // need the image the real GUI always has.
        let tmp =
            std::env::temp_dir().join(format!("macvm_find_impl_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&test_world_dir(), &tmp).expect("seed");
        let responses = handle(
            VmRequest::Find {
                tool: "implementors".to_string(),
                query: "printOn:".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        let html = match responses.as_slice() {
            [VmResponse::FindResults { html }] => html.clone(),
            other => panic!("expected FindResults, got {other:?}"),
        };
        assert!(
            html.contains("Implementors of") && html.contains("data-open-class=\"Point\""),
            "must list Point among printOn: implementors (a drill-link), got a {}-char result",
            html.len()
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// Outliner "＋ add variable": a class variable goes LIVE (reflection shows
    /// it at once); an instance variable is a shape change the running VM can't
    /// apply, so it persists for the next boot and is NOT live; junk is rejected.
    #[test]
    fn add_variable_class_var_is_live_instance_var_awaits_restart() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let tmp =
            std::env::temp_dir().join(format!("macvm_addvar_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&test_world_dir(), &tmp).expect("seed");
        // Mirror the image (so the mock's Point superclass matches the VM's — the
        // class-var reopen would otherwise be rejected for a superclass mismatch).
        let mut world = mock_world_from_image(&image);
        let mut sel = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let add = |world: &mut MockWorld, sel: &mut BrowserSelection, vm: &mut VmHandle, cv: bool, name: &str| {
            handle(
                VmRequest::SmapplAddVar {
                    cls: "Point".to_string(),
                    is_class_var: cv,
                    name: name.to_string(),
                },
                world,
                sel,
                Some(&image),
                vm,
            )
        };

        // Class variable -> live.
        let r = add(&mut world, &mut sel, &mut vm, true, "Origin");
        assert!(
            matches!(r.first(), Some(VmResponse::Transcript(m)) if m.contains("live")),
            "class var must report live, got {r:?}"
        );
        assert!(
            vm.eval("ClassMirror classVariablesOf: Point")
                .expect("eval")
                .contains("Origin"),
            "class var must be live in the VM"
        );

        // Instance variable -> persisted but NOT live (fixed shape).
        let r2 = add(&mut world, &mut sel, &mut vm, false, "z");
        assert!(
            matches!(r2.first(), Some(VmResponse::Transcript(m)) if m.contains("restart")),
            "instance var must report restart-to-apply, got {r2:?}"
        );
        assert!(
            !vm.eval("ClassMirror instanceVariablesOf: Point")
                .expect("eval")
                .contains("#z"),
            "instance var must NOT be live (shape change)"
        );

        // Junk name -> rejected.
        let r3 = add(&mut world, &mut sel, &mut vm, false, "3bad");
        assert!(
            matches!(r3.first(), Some(VmResponse::Transcript(m)) if m.contains("not a valid")),
            "invalid name must be rejected, got {r3:?}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// Find Definition does a case-insensitive substring search over class
    /// names.
    #[test]
    fn find_definition_matches_class_names() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let tmp =
            std::env::temp_dir().join(format!("macvm_find_def_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&test_world_dir(), &tmp).expect("seed");
        let responses = handle(
            VmRequest::Find {
                tool: "definition".to_string(),
                query: "collect".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        let html = match responses.as_slice() {
            [VmResponse::FindResults { html }] => html.clone(),
            other => panic!("expected FindResults, got {other:?}"),
        };
        assert!(
            html.contains("data-open-class=\"Collection\"")
                && html.contains("data-open-class=\"OrderedCollection\""),
            "'collect' must match Collection classes, got {}-char result",
            html.len()
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// The Senders find tool lists methods that SEND the selector — now from the
    /// image's `method_sends` index (parsed from source), not a VM IC-scan.
    /// Object>>printString sends printOn:.
    #[test]
    fn find_senders_lists_methods_that_send_the_selector() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        // Senders reads method_sends; open_or_seed_image seeds AND backfills it.
        let tmp =
            std::env::temp_dir().join(format!("macvm_find_send_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&test_world_dir(), &tmp).expect("seed");
        let responses = handle(
            VmRequest::Find {
                tool: "senders".to_string(),
                query: "printOn:".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        let html = match responses.as_slice() {
            [VmResponse::FindResults { html }] => html.clone(),
            other => panic!("expected FindResults, got {other:?}"),
        };
        assert!(
            html.contains("Senders of") && html.contains("data-open-class=\"Object\""),
            "Object>>printString sends printOn:, so Object must appear, got {}-char result: {html}",
            html.len()
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// Drill-down: clicking a class in a hierarchy outliner opens that class's
    /// method browser (a ClassOutliner with editors) in the same widget, with a
    /// back link to the hierarchy.
    #[test]
    fn open_class_drills_into_a_method_browser_with_a_back_link() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp = std::env::temp_dir()
            .join(format!("macvm_drill_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");
        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();

        let opened = handle(
            VmRequest::SmapplOpenClass {
                cls: "Point".to_string(),
                root: "Object".to_string(),
                widget_id: "s0".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        let html = match opened.as_slice() {
            [VmResponse::SmapplFragment { id, html }] if id == "s0" => html.clone(),
            other => panic!("expected a SmapplFragment for s0, got {other:?}"),
        };
        assert!(
            html.contains("st-classoutliner")
                && html.contains("st-smappl-src")
                && html.contains("instance side"),
            "drilling into Point must show its method browser with editors, got a {}-char fragment",
            html.len()
        );
        assert!(
            html.contains("data-open-hierarchy=\"Object\""),
            "the class browser must carry a back link to the Object hierarchy: {html}"
        );

        // Back re-opens the hierarchy in the same widget.
        let back = handle(
            VmRequest::SmapplOpenHierarchy {
                root: "Object".to_string(),
                widget_id: "s0".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        assert!(
            matches!(back.as_slice(), [VmResponse::SmapplFragment { id, html }]
                if id == "s0" && html.contains("st-outliner") && html.contains("data-open-class")),
            "back must re-open the clickable hierarchy, got {back:?}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// View sync (ToolRegistry, docs/APPS.md §7): with two outliners open, an
    /// accept in one live-refreshes the OTHER (so the edit shows everywhere)
    /// but skips the one being edited (so its editor isn't yanked mid-edit).
    #[test]
    fn accept_refreshes_other_open_outliners_but_not_the_edited_one() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp = std::env::temp_dir()
            .join(format!("macvm_sync_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");
        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();

        // Two open Point outliners: s0 (being edited) and s1 (a bystander).
        for id in ["s0", "s1"] {
            handle(
                VmRequest::SmapplRender {
                    id: id.to_string(),
                    code: "ClassOutliner for: (ClassMirror on: Point)".to_string(),
                },
                &mut world,
                &mut sel,
                Some(&image),
                &mut vm,
            );
        }

        let responses = handle(
            VmRequest::SmapplAccept {
                cls: "Point".to_string(),
                side: "instance".to_string(),
                sel: "y".to_string(),
                text: "y [ ^42 ]".to_string(),
                widget_id: "s0".to_string(), // the edited outliner
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );

        let refreshed: Vec<&str> = responses
            .iter()
            .filter_map(|r| match r {
                VmResponse::SmapplFragment { id, html } => {
                    assert!(html.contains("^42"), "refresh must carry the new source");
                    Some(id.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            refreshed,
            vec!["s1"],
            "only the bystander outliner refreshes; the edited one (s0) is skipped, got {refreshed:?}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// The outliner "＋ new method" / "＋ new class" affordances CREATE (not
    /// edit): the selector/class name is parsed from the typed template, and
    /// the result is versioned into image_store — the same create backend the
    /// class browser uses, reached statelessly.
    #[test]
    fn outliner_creates_and_versions_a_new_method_and_class() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_newmc_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");
        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");
        let mut world = MockWorld::seed();
        let mut sel = BrowserSelection::default();

        // ＋ new method on an existing class (Point) — selector parsed from src.
        let r = handle(
            VmRequest::SmapplNewMethod {
                cls: "Point".to_string(),
                side: "instance".to_string(),
                text: "tripled\n\t^self * 3".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        assert!(
            r.iter().any(|resp| matches!(resp, VmResponse::Transcript(t) if t.contains("Added Point>>tripled"))),
            "new-method transcript, got {r:?}"
        );
        assert_eq!(
            image
                .method_source("Point", image_store::Side::Instance, "tripled")
                .unwrap()
                .as_deref()
                .map(|s| s.contains("^self * 3")),
            Some(true),
            "the new method must be versioned into the image"
        );

        // ＋ new class — the class AND its inline method both get versioned.
        let r = handle(
            VmRequest::SmapplNewClass {
                text: "Object subclass: WidgetX [\n\tspin [ ^1 ]\n]".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        assert!(
            r.iter().any(|resp| matches!(resp, VmResponse::Transcript(t) if t.contains("Created WidgetX"))),
            "new-class transcript, got {r:?}"
        );
        assert!(
            image.class_exists("WidgetX").unwrap(),
            "the new class must exist in the image"
        );
        assert_eq!(
            image
                .method_source("WidgetX", image_store::Side::Instance, "spin")
                .unwrap()
                .as_deref()
                .map(|s| s.contains("^1")),
            Some(true),
            "the new class's method must be versioned too"
        );

        // Creating the same class again is rejected, not duplicated.
        let dup = handle(
            VmRequest::SmapplNewClass {
                text: "Object subclass: WidgetX [\n]".to_string(),
            },
            &mut world,
            &mut sel,
            Some(&image),
            &mut vm,
        );
        assert!(
            dup.iter().any(|resp| matches!(resp, VmResponse::Transcript(t) if t.contains("already exists"))),
            "duplicate class must be refused, got {dup:?}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// M3 (docs/package_aware_editing_design.md §4.2): unlike
    /// `SmapplNewMethod`/`SmapplNewClass` (a separate, deliberately
    /// out-of-scope code path exercised by the test above), `BrowserSaveSource`
    /// is the real class browser's own accept path — its `NewMethod`/`NewClass`
    /// arms must give every created method a non-`NULL` `source_file` too, so
    /// `class_home_file` is never left unanswerable for browser-created content.
    #[test]
    fn browser_save_source_new_method_and_new_class_get_a_home_in_the_image() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp = std::env::temp_dir()
            .join(format!("macvm_browser_home_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");
        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");
        let mut world = MockWorld::seed();

        // A new method on an existing class (Point, already used above).
        let mut selection = BrowserSelection {
            class: Some("Point".to_string()),
            edit_target: SourceEditTarget::NewMethod,
            ..Default::default()
        };
        handle(
            VmRequest::BrowserSaveSource {
                text: "quadrupled\n\t^self * 4".to_string(),
                saved_package: String::new(),
                saved_class: "Point".to_string(),
                saved_side: "instance".to_string(),
                saved_category: String::new(),
                saved_method: String::new(),
                saved_target: "new_method".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        assert_eq!(
            image
                .method_source_file("Point", image_store::Side::Instance, "quadrupled")
                .unwrap(),
            Some(image_store::flows::INTERACTIVE_SOURCE_FILE.to_string()),
            "a new method through the real browser accept path must get a home"
        );

        // A new class: every one of its methods gets a home too.
        let mut selection = BrowserSelection {
            package: Some("scratch".to_string()),
            edit_target: SourceEditTarget::NewClass,
            ..Default::default()
        };
        handle(
            VmRequest::BrowserSaveSource {
                text: "Object subclass: BrowserHomeTest [\n\tping [ ^1 ]\n]".to_string(),
                saved_package: "scratch".to_string(),
                saved_class: String::new(),
                saved_side: "instance".to_string(),
                saved_category: String::new(),
                saved_method: String::new(),
                saved_target: "new_class".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        assert_eq!(
            image.class_home_file("BrowserHomeTest").unwrap(),
            Some(image_store::flows::INTERACTIVE_SOURCE_FILE.to_string()),
            "a new class's methods through the real browser accept path must get a home"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// Accepting a ClassOutliner edit VERSIONS it into image_store (a new
    /// `method_versions` row, never an overwrite — the user's explicit
    /// requirement) AND live-compiles it into the running VM.
    #[test]
    fn smappl_accept_versions_the_edit_and_makes_it_live() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp = std::env::temp_dir()
            .join(format!("macvm_accept_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();

        let versions_before = image
            .method_version_count("Point", image_store::Side::Instance, "y")
            .unwrap();
        assert_eq!(
            vm.eval("(Point x: 7 y: 2) y.").unwrap(),
            "2",
            "sanity: Point>>y returns the y ivar"
        );

        // Redefine Point>>y to return the x ivar — proves redefinition is live
        // AND that reopening preserved the ivars.
        let responses = handle(
            VmRequest::SmapplAccept {
                cls: "Point".to_string(),
                side: "instance".to_string(),
                sel: "y".to_string(),
                text: "y [ ^x ]".to_string(),
                widget_id: String::new(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        assert!(
            transcript_containing(&responses, "versioned"),
            "accept must report it versioned the edit, got {responses:?}"
        );

        // 1. A NEW version row exists (not an overwrite).
        let versions_after = image
            .method_version_count("Point", image_store::Side::Instance, "y")
            .unwrap();
        assert_eq!(
            versions_after,
            versions_before + 1,
            "each accept must add exactly one method_versions row"
        );
        // 2. The latest version in the DB is the edited text.
        assert_eq!(
            image
                .method_source("Point", image_store::Side::Instance, "y")
                .unwrap()
                .as_deref(),
            Some("y [ ^x ]"),
            "the image's latest version must be the edit"
        );
        // 3. The running VM now runs the redefined method (ivars preserved).
        assert_eq!(
            vm.eval("(Point x: 7 y: 2) y.").unwrap(),
            "7",
            "the edit must be live in the VM, with the x ivar preserved"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// M5 (`docs/package_aware_editing_design.md` §4.4): since M4, `world`
    /// (the `MockWorld` mirror) and the image can know about a class THIS
    /// PARTICULAR VM never booted — a `cocoaui`-only class, when this VM
    /// only requested `WORLD_LISTS` — so a live-compile reopen must be gated
    /// on the VM ITSELF having the class, not on `world`/`image` knowing
    /// about it. A reopen has no ivar declaration, so running it
    /// unconditionally on a VM that never loaded the class would silently
    /// DEFINE A WRONG-SHAPE SHELL under that name instead of failing (§3) —
    /// the real bug this design exists to close. (The "gate applies when
    /// the VM DOES have the class" path is already covered by
    /// `smappl_accept_versions_the_edit_and_makes_it_live` above — `Point`
    /// is base-world content, present on that test's VM too — so this test
    /// only needs to add the NEW "gate skips" behavior.)
    #[test]
    fn smappl_new_method_skips_live_compile_when_the_vm_lacks_the_class() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_m5gate_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        // world.list AND cocoaui.list both land in the SAME image
        // (import_all_lists) — the image knows about CocoaHelp.
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");
        assert!(
            image.class_named("CocoaHelp").unwrap().is_some(),
            "sanity: the image must know about CocoaHelp (cocoaui.list)"
        );

        // This VM boots WORLD_LISTS ONLY — it never loaded cocoaui, so it
        // has no CocoaHelp — exactly M4's selective-boot split.
        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS)
            .expect("db load");
        assert!(
            vm.eval("CocoaHelp").is_err(),
            "sanity: this VM must NOT have CocoaHelp loaded"
        );

        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();

        let responses = handle(
            VmRequest::SmapplNewMethod {
                cls: "CocoaHelp".to_string(),
                side: "instance".to_string(),
                text: "m5GateProbe [ ^1 ]".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        assert!(
            transcript_containing(&responses, "not applied live"),
            "must honestly report the VM doesn't have this class loaded, got {responses:?}"
        );

        // The real proof: no wrong-shape shell was defined on the VM under
        // this name (§3's actual bug) — CocoaHelp is still simply
        // undeclared there, exactly as before the request.
        assert!(
            vm.eval("CocoaHelp").is_err(),
            "the gate must not have silently defined a wrong-shape CocoaHelp shell on the VM"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// The `SmapplAddVar` class-var branch builds its OWN reopen (a
    /// SEPARATE code path from `SmapplNewMethod`, sharing only
    /// `live_compile_reopen`) — same M5 gate, same shape of proof.
    #[test]
    fn smappl_add_class_var_skips_live_compile_when_the_vm_lacks_the_class() {
        OPEN_OUTLINERS.with(|o| o.borrow_mut().clear());
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_m5gate_cvar_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS)
            .expect("db load");
        assert!(vm.eval("CocoaHelp").is_err(), "sanity: not loaded here");

        let mut world = MockWorld::seed();
        // `SmapplAddVar` has its OWN, unrelated pre-existing sanity guard —
        // "does the browser mirror know this class at all" — which
        // `MockWorld::seed()`'s standalone hardcoded fixture doesn't; add it
        // so that guard passes and the M5 gate itself is what's exercised.
        // This accurately models the real M4 divergence: the DATABASE
        // knows about CocoaHelp (M1 imported it), but THIS VM never booted
        // it (M4 made boot selective).
        world.add_class("CocoaHelp", Some("Visual"), "GUI", "", "", "");
        let mut selection = BrowserSelection::default();

        let responses = handle(
            VmRequest::SmapplAddVar {
                cls: "CocoaHelp".to_string(),
                is_class_var: true,
                name: "M5GateProbeCVar".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        assert!(
            transcript_containing(&responses, "not applied live"),
            "must honestly report the VM doesn't have this class loaded, got {responses:?}"
        );
        assert!(
            vm.eval("CocoaHelp").is_err(),
            "the gate must not have silently defined a wrong-shape CocoaHelp shell on the VM"
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// A `ClassOutliner` rendered through `handle` gets its method-source
    /// `<pre>` blocks filled from image_store (the VM keeps no source) —
    /// exercises the full worker seam incl. `inject_method_sources`.
    #[test]
    fn class_outliner_source_is_filled_from_the_image() {
        let world_dir = test_world_dir();
        let tmp = std::env::temp_dir()
            .join(format!("macvm_co_src_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");

        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let responses = handle(
            VmRequest::SmapplRender {
                id: "s0".to_string(),
                code: "ClassOutliner for: (ClassMirror on: Point)".to_string(),
            },
            &mut world,
            &mut selection,
            Some(&image),
            &mut vm,
        );
        let html = match responses.as_slice() {
            [VmResponse::SmapplFragment { html, .. }] => html.clone(),
            other => panic!("expected one SmapplFragment, got {other:?}"),
        };
        // Point>>printOn:'s source body (from the image) must be spliced into
        // its selector node's <pre> — a snippet unique to the source, not the
        // selector list.
        assert!(
            html.contains("nextPutAll:") && html.contains("st-smappl-src"),
            "the class outliner must carry Point's method source, got a {}-char fragment",
            html.len()
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// The real GUI boots from a DB image (S22), not `world/*.mst` directly.
    /// `33_smappl.mst` must survive that import/export round trip — S22 found
    /// several image_store fidelity bugs, so a widget rendering through the
    /// `.mst` boot is no guarantee it renders through the DB boot the shell
    /// actually uses. Seed → DB-boot → render the same button.
    #[test]
    fn smappl_renders_through_a_db_booted_vm() {
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_smappl_db_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");

        let html = vm
            .render_fragment("Button labeled: 'DB' action: [ :b | b ]")
            .expect("the smappl classes must render through a DB-booted VM");
        assert!(
            html.contains("smappl-button") && html.contains("DB"),
            "the DB-booted world must render a live button, got {html:?}"
        );

        // The Phase-W outliner (allClasses primitive + ClassMirror +
        // HtmlWriter, world/34_tools.mst) must also survive the DB round trip
        // — this is the start page's own smappl.
        let tree = vm
            .render_fragment("ClassHierarchyOutliner imbeddedVisualForClass: Object")
            .expect("the hierarchy outliner must render through a DB-booted VM");
        assert!(
            tree.contains("st-outliner") && tree.contains("Object") && tree.contains("Behavior"),
            "the DB-booted world must render the class-hierarchy tree, got {tree:?}"
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn doit_evaluates_through_the_real_vm_and_echoes_the_source() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::Doit {
                code: "3 + 4.".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        // "Do it" echoes what ran (the result value itself is "Print it"'s
        // job, not shown here) — but it went through the real evaluator, so
        // a well-formed doit produces exactly the echo and nothing else.
        assert!(matches!(responses.as_slice(), [VmResponse::Transcript(t)] if t.contains("3 + 4")));
    }

    #[test]
    fn workspace_print_it_returns_the_real_printstring() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::WorkspacePrintIt {
                code: "3 + 4.".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::WorkspacePrintResult { text }] = responses.as_slice() else {
            panic!("expected exactly one WorkspacePrintResult, got {responses:?}");
        };
        // The real printString of 7 — no longer the old "(no VM yet: ...)"
        // stub. Leading space is the inline-insertion convention.
        assert_eq!(text, " 7");
    }

    /// Regression for "the workspace evaluates nothing": the default
    /// placeholder buffer must be a leading comment FOLLOWED BY a runnable
    /// statement, so Do it / Print it on the untouched buffer (nothing
    /// selected → the whole buffer is the eval target) produce a real
    /// result. The old placeholder was pure comment, which parses to no
    /// statement and answers empty — indistinguishable from "nothing
    /// happened".
    #[test]
    fn the_default_workspace_placeholder_prints_a_real_result() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::WorkspacePrintIt {
                code: crate::workspace_render::initial_text().to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::WorkspacePrintResult { text }] = responses.as_slice() else {
            panic!("expected exactly one WorkspacePrintResult, got {responses:?}");
        };
        assert_eq!(
            text, " 7",
            "the untouched workspace buffer must Print it to a real 7, not empty"
        );
    }

    /// "the JIT MUST be supported" (the S21 directive) — end-to-end through
    /// the GUI's own request path, not just the embedding layer's unit test.
    /// `Threshold(1)` compiles on the first call.
    #[test]
    fn workspace_print_it_works_with_the_jit_enabled() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Threshold(1));
        let responses = handle(
            VmRequest::WorkspacePrintIt {
                code: "6 * 7.".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        let [VmResponse::WorkspacePrintResult { text }] = responses.as_slice() else {
            panic!("expected exactly one WorkspacePrintResult, got {responses:?}");
        };
        assert_eq!(text, " 42");
    }

    /// A malformed doit is a `GuestError::Compile`, surfaced as a transcript
    /// line — NOT a thread death. The same `vm` keeps serving afterward.
    #[test]
    fn doit_compile_error_is_reported_and_the_vm_survives() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let responses = handle(
            VmRequest::Doit {
                code: "3 + + 4.".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(transcript_containing(&responses, "error"), "{responses:?}");
        // Same handle still works — the compile error didn't kill anything.
        let after = handle(
            VmRequest::WorkspacePrintIt {
                code: "1 + 1.".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );
        assert!(
            matches!(after.as_slice(), [VmResponse::WorkspacePrintResult { text }] if text == " 2")
        );
    }

    /// The editor bridge (docs/editor_design.md M3), end to end through the
    /// real `handle()` dispatch: EditorOpen/EditorKey/EditorCommand → the
    /// Smalltalk EditorSession → a `render` blast → decoded EditorView. Proves
    /// the wire round-trip, the string escaping (the doc holds a `'`), that an
    /// edit lands at the given caret, and that the WHOLE buffer + caret come back.
    #[test]
    fn editor_bridge_open_key_and_command_round_trip() {
        let mut world = MockWorld::seed();
        let mut selection = BrowserSelection::default();
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);

        let view = |rs: &[VmResponse]| -> (i64, String) {
            match rs {
                [VmResponse::EditorView { cursor, text }] => (*cursor, text.clone()),
                other => panic!("expected one EditorView, got {other:?}"),
            }
        };

        // Open on a two-line doc containing a quote — exercises the literal
        // escaper; the blast is the whole doc with the caret at 0.
        let (cur, text) = view(&handle(
            VmRequest::EditorOpen {
                text: "ab'c\ndef".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        ));
        assert_eq!(cur, 0);
        assert_eq!(text, "ab'c\ndef", "the quote survived the round trip");

        // Insert 'X' at caret offset 3 (before 'c' on line 1) — the edit lands
        // exactly there, and the whole buffer comes back with the new caret.
        let (cur, text) = view(&handle(
            VmRequest::EditorKey {
                key: "X".to_string(),
                at: 3,
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        ));
        assert_eq!(text, "abX'c\ndef", "inserted at the given caret, whole buffer back");
        assert_eq!(cur, 3, "0-based caret advanced past the inserted char");

        // Undo restores the original document.
        let (_cur, text) = view(&handle(
            VmRequest::EditorCommand {
                name: "undo".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        ));
        assert_eq!(text, "ab'c\ndef");
    }

    /// Save goes to the DATABASE, not the live VM — so a SHAPE change (adding an
    /// instance var), which a live redefinition refuses ("cannot change shape of
    /// existing class"), now saves fine; and a syntax error is rejected by the
    /// real parser without any install. This is the bug the user hit.
    #[test]
    fn editor_accept_saves_a_shape_change_and_gates_syntax() {
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        let tmp = std::env::temp_dir().join(format!("macvm_edacc_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let img = image_store::Image::open(&tmp.join("image.sqlite3")).unwrap();
        img.create_or_reopen_class("Object", None, "Kernel", "", "").unwrap();
        img.create_or_reopen_class("FooShape", Some("Object"), "App", "", "a")
            .unwrap();
        img.create_or_reopen_method("FooShape", image_store::Side::Instance, "m", "", "m [ ^a ]")
            .unwrap();

        // 1) A syntactically broken buffer (unclosed brackets) is rejected by
        //    the parser — nothing saved.
        vm.exec("EditorSession openOn: 'Object subclass: FooShape [ m [ ^a'")
            .unwrap();
        let bad = editor_accept_response(&mut vm, Some(&img), &tmp);
        assert!(
            matches!(&bad, VmResponse::Transcript(t) if t.contains("not saved")),
            "syntax error must not save, got {bad:?}"
        );

        // 2) A SHAPE change (add instance var `b`, add method `n`) parses and
        //    saves to the image — the old live-install path failed here.
        vm.exec(
            "EditorSession openOn: 'Object subclass: FooShape [ | a b | m [ ^a ] n [ ^b ] ]'",
        )
        .unwrap();
        let ok = editor_accept_response(&mut vm, Some(&img), &tmp);
        match &ok {
            VmResponse::Transcript(t) => assert!(
                t.contains("saved") && !t.contains("cannot change shape"),
                "shape change must save, got {t}"
            ),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            img.class_named("FooShape").unwrap().unwrap().instance_vars,
            "a b",
            "the new shape persisted"
        );
        assert!(img
            .method_source("FooShape", image_store::Side::Instance, "n")
            .unwrap()
            .is_some());
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The editor Save persists ONLY what changed (docs/editor_design.md M4):
    /// an unchanged method must not spawn a no-op `method_versions` row (which
    /// would churn history and invalidate the live nmethod for nothing), a
    /// changed/new one is written, and a vanished one is removed.
    #[test]
    fn persist_editor_class_writes_only_the_diff() {
        let img = image_store::Image::open_in_memory().unwrap();
        img.create_or_reopen_class("Object", None, "Kernel", "", "").unwrap();
        img.create_or_reopen_class("Foo", Some("Object"), "App", "", "").unwrap();
        img.create_or_reopen_method("Foo", image_store::Side::Instance, "bar", "", "bar [ ^1 ]")
            .unwrap();
        img.create_or_reopen_method("Foo", image_store::Side::Instance, "gone", "", "gone [ ^0 ]")
            .unwrap();
        let versions_of = |sel: &str| {
            img.method_version_count("Foo", image_store::Side::Instance, sel)
                .unwrap()
        };
        assert_eq!(versions_of("bar"), 1);

        // Edit `bar`, add `baz`, drop `gone`.
        let text = "Object subclass: Foo [\n    bar [ ^2 ]\n    baz [ ^3 ]\n]\n";
        let msg = persist_editor_class(&img, text);
        assert!(msg.contains("1 changed"), "{msg}");
        assert!(msg.contains("1 added"), "{msg}");
        assert!(msg.contains("1 removed"), "{msg}");
        assert_eq!(versions_of("bar"), 2, "bar changed -> a new version");
        assert_eq!(
            img.method_source("Foo", image_store::Side::Instance, "baz")
                .unwrap()
                .as_deref(),
            Some("baz [ ^3 ]")
        );
        assert!(img
            .method_source("Foo", image_store::Side::Instance, "gone")
            .unwrap()
            .is_none());

        // Save again with no edits — nothing is written (no version churn).
        let msg2 = persist_editor_class(&img, text);
        assert!(msg2.contains("no changes"), "{msg2}");
        assert_eq!(versions_of("bar"), 2, "an unchanged save must not re-version");
    }

    /// Polls `drain_responses` (accumulating across calls) until `pred`
    /// holds or a generous deadline passes — booting the real world twice
    /// (original + respawned worker) genuinely takes a moment.
    fn poll_until(host: &VmHost, pred: impl Fn(&[VmResponse]) -> bool) -> Vec<VmResponse> {
        let start = Instant::now();
        let mut collected = Vec::new();
        while start.elapsed() < Duration::from_secs(20) {
            collected.extend(host.drain_responses());
            if pred(&collected) {
                return collected;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        collected
    }

    /// Multi-Smalltalk workers M3, end to end through the REAL host loop: a
    /// spawned worker VM's reply reaches its handler with NO frame timer and
    /// NO manual dispatch anywhere — the worker's send fires the inbox wake,
    /// which queues a `WorkerInbox` request from the worker's own thread, and
    /// the primary VM thread's ordinary `recv()` sleep services it
    /// (docs/multi-smalltalk-worker.md §3.1/§9: event-driven, zero polling).
    #[test]
    fn worker_reply_is_dispatched_by_the_inbox_wake() {
        let tmp_dir =
            std::env::temp_dir().join(format!("macvm_wkwake_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        {
            let img = image_store::Image::open(&tmp_dir.join("image.sqlite3")).unwrap();
            image_store::import::import_world_dir(&img, &test_world_dir()).unwrap();
        }
        let host = spawn_with_world_and_timeout(
            Waker::none(),
            tmp_dir,
            Duration::from_secs(30),
        );
        host.submit(VmRequest::Doit {
            code: "Worker onReply: [:m | Transcript show: 'WKREPLY ', m payload printString]."
                .to_string(),
        });
        host.submit(VmRequest::Doit {
            code: "(Worker spawn: 'Worker onMessage: [:m | Worker reply: m payload + 1]') send: 41."
                .to_string(),
        });
        // From here on, NOTHING drives the primary except the wake: the reply
        // envelope must arrive, queue WorkerInbox, dispatch, and print.
        let seen = poll_until(&host, |rs| transcript_containing(rs, "WKREPLY 42"));
        assert!(
            transcript_containing(&seen, "WKREPLY 42"),
            "the worker's reply must be dispatched by the inbox wake alone, got {seen:?}"
        );
    }

    /// The S21 "GUI must not die" guarantee, in its STRONGEST form: an
    /// unhandled runtime error (a DNU) does not kill the worker at all — the
    /// guest-fatal recovery (`raise_guest_fatal` -> `siglongjmp` back to
    /// `eval`'s `sigsetjmp`, `docs/SPEC.md`) unwinds to the doit boundary, the
    /// worker reports the error, goes idle, and serves the very next request on
    /// the SAME VM. No thread death, no respawn, no lost state.
    ///
    /// This is the runtime-error twin of `doit_compile_error_is_reported_and_
    /// the_vm_survives` (which covers a compile-time error), and it is why a
    /// workspace typo can't restart your VM. The supervisor's respawn path —
    /// for a genuinely wedged/dead worker — is covered separately by
    /// `restart_swaps_in_a_fresh_worker_that_still_serves`. That the recovery
    /// is genuinely CLEAN (the aborted doit's frames/handles don't leak onto
    /// the surviving VM) is proven at the VM layer by
    /// `embed::tests::eval_recovers_to_a_clean_stack_and_arena_without_accumulating`.
    ///
    /// (History: this test used to assert the OPPOSITE — that a DNU
    /// `pthread_exit`s the worker and forces a respawn. That only happened
    /// because error reporting once *panicked* in the worker thread; once that
    /// panic was fixed the guest-fatal recovery does its job and the worker
    /// survives, which is the behavior we actually want and now assert.)
    ///
    /// Wake target is absent — `Waker::notify` short-circuits, so
    /// this needs no windowing system (headless-safe).
    ///
    /// **macOS-only until Windows guest-fatal recovery lands** (the known
    /// Phase-2 follow-up in `../MIGRATION.md` §6, not a GUI-port regression).
    /// The recovery this asserts *is* `siglongjmp` back to `eval`'s
    /// `sigsetjmp`, and on Windows both are stubs:
    /// `codecache::deopt_trap::siglongjmp` calls `std::process::abort()`
    /// (which is what surfaces as `STATUS_STACK_BUFFER_OVERRUN`/0xc0000409).
    /// So on Windows today a DNU reports its error to the transcript and then
    /// takes the process down — this test would not fail, it would abort the
    /// whole test binary and take every sibling test with it.
    ///
    /// Re-enable by deleting this gate once the `catch_unwind`-based recovery
    /// `MIGRATION.md` proposes exists; it is the acceptance test for it.
    #[cfg(target_os = "macos")]
    #[test]
    fn worker_survives_an_unhandled_runtime_error_and_serves_the_next_request() {
        // Isolate the image to a temp dir, pre-seeded once, so the worker boots
        // from the DB (S22) without seeding or touching the repo's own
        // world/image.sqlite3. The worker derives its image path as
        // `<world_dir>/image.sqlite3`, so passing this temp dir as the
        // world_dir points it here.
        let tmp_dir = std::env::temp_dir().join(format!("macvm_survive_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        {
            let img = image_store::Image::open(&tmp_dir.join("image.sqlite3")).unwrap();
            image_store::import::import_world_dir(&img, &test_world_dir()).unwrap();
        }

        // A NIL SEL is fine: notify() never dereferences it (NIL target).
        let host = spawn_with_world_and_timeout(
            Waker::none(),
            tmp_dir.clone(),
            Duration::from_millis(150),
        );

        // 1. A doit that raises an unhandled DNU. The worker emits its "Error:
        //    does not understand …" transcript and recovers at the doit
        //    boundary — it does NOT die.
        host.submit(VmRequest::Doit {
            code: "3 thisSelectorDoesNotExistAnywhereInTheBaseWorld.".to_string(),
        });
        let seen = poll_until(&host, |rs| transcript_containing(rs, "does not understand"));
        assert!(
            transcript_containing(&seen, "does not understand"),
            "the worker must have reported its DNU error, got {seen:?}"
        );

        // 2. Recovery, made concrete: the worker returns to idle (it sends
        //    WorkerIdle, which `drain_responses` turns into `pending_since =
        //    None`) rather than dying silently — a dead worker never sends
        //    WorkerIdle, so `is_idle` would stay false. Keep draining until it
        //    flips (the WorkerIdle may trail the error transcript by a beat).
        poll_until(&host, |_| host.is_idle());
        assert!(
            host.is_idle(),
            "the worker must return to idle after the error — it recovered, not died"
        );

        // 3. The SAME worker serves the next request. No respawn notice, because
        //    nothing was respawned — the recovery kept the original VM alive.
        host.submit(VmRequest::Doit {
            code: "6 * 7.".to_string(),
        });
        let seen = poll_until(&host, |rs| transcript_containing(rs, "6 * 7"));
        assert!(
            transcript_containing(&seen, "6 * 7"),
            "the surviving worker must have served the follow-up doit, got {seen:?}"
        );
        assert!(
            !transcript_containing(&seen, "fresh one has been started"),
            "no respawn should have happened — the worker survived the error, got {seen:?}"
        );

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    /// S22-E: editing a method through the browser's accept path makes the
    /// new behaviour LIVE in the running VM — a subsequent eval sees it, not
    /// just the mock/image. The class exists in both the mock and the VM (as
    /// it would after a DB boot); a class-side method keeps the test free of
    /// instantiation.
    #[test]
    fn browser_method_edit_is_live_in_the_running_vm() {
        let mut world = MockWorld::empty();
        world.add_class("LiveFoo", Some("Object"), "Test", "", "", "");
        world.add_method(
            "LiveFoo",
            Side::Class,
            "ping",
            "accessing",
            "LiveFoo class >> ping [ ^1 ]",
        );
        let mut vm = test_vm_handle(macvm::runtime::JitMode::Off);
        vm.exec("Object subclass: LiveFoo [ LiveFoo class >> ping [ ^1 ] ]")
            .unwrap();
        assert_eq!(
            vm.eval("LiveFoo ping.").unwrap(),
            "1",
            "sanity: starts at 1"
        );

        let mut selection = BrowserSelection {
            class: Some("LiveFoo".to_string()),
            category: Some("accessing".to_string()),
            method: Some("ping".to_string()),
            side: Side::Class,
            edit_target: SourceEditTarget::Method,
            ..Default::default()
        };
        let responses = handle(
            VmRequest::BrowserSaveSource {
                text: "LiveFoo class >> ping [ ^42 ]".to_string(),
                saved_package: String::new(),
                saved_class: "LiveFoo".to_string(),
                saved_side: "class".to_string(),
                saved_category: "accessing".to_string(),
                saved_method: "ping".to_string(),
                saved_target: "method".to_string(),
            },
            &mut world,
            &mut selection,
            None,
            &mut vm,
        );

        assert!(
            transcript_containing(&responses, "live in the VM"),
            "the save must report it went live, got {responses:?}"
        );
        // The real proof: the running VM now runs the edited method.
        assert_eq!(
            vm.eval("LiveFoo ping.").unwrap(),
            "42",
            "the browser edit must be live in the VM"
        );
    }

    /// S22-E part A: a fresh image path is auto-seeded from `world/*.mst`
    /// (zero setup); re-opening the same file does NOT re-seed.
    #[test]
    fn open_or_seed_image_seeds_then_reuses() {
        let world_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../world"));
        let tmp =
            std::env::temp_dir().join(format!("macvm_seed_test_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();

        let img = open_or_seed_image(&world_dir, &tmp).expect("seed fresh");
        let n = img.all_classes().unwrap().len();
        drop(img);
        assert!(
            n >= 40,
            "a fresh path must be seeded with the whole world, got {n}"
        );

        // Re-open the same file: it already has classes, so it is NOT
        // re-seeded (same count, no duplicate-class error).
        let img2 = open_or_seed_image(&world_dir, &tmp).expect("reopen seeded");
        assert_eq!(img2.all_classes().unwrap().len(), n);
        drop(img2);
        std::fs::remove_file(&tmp).ok();
    }

    /// The "Restart VM Thread" menu action (`VmHost::restart`) swaps in a
    /// fresh worker + VM that serves the next request, and posts a
    /// "restarted" notice. A generous timeout so the timeout-respawn path
    /// can't fire — this tests the EXPLICIT restart, not death detection.
    #[test]
    fn restart_swaps_in_a_fresh_worker_that_still_serves() {
        let tmp_dir = std::env::temp_dir().join(format!("macvm_restart_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        {
            let img = image_store::Image::open(&tmp_dir.join("image.sqlite3")).unwrap();
            image_store::import::import_world_dir(&img, &test_world_dir()).unwrap();
        }
        let host = spawn_with_world_and_timeout(
            Waker::none(),
            tmp_dir.clone(),
            Duration::from_secs(30),
        );

        // The original worker serves a request.
        host.submit(VmRequest::Doit {
            code: "3 + 4.".to_string(),
        });
        let seen = poll_until(&host, |rs| transcript_containing(rs, "3 + 4"));
        assert!(
            transcript_containing(&seen, "3 + 4"),
            "original served: {seen:?}"
        );

        // Explicit restart → fresh worker + "restarted" notice serving next.
        host.restart();
        host.submit(VmRequest::Doit {
            code: "6 * 7.".to_string(),
        });
        let seen = poll_until(&host, |rs| transcript_containing(rs, "6 * 7"));
        assert!(
            transcript_containing(&seen, "VM thread restarted"),
            "the restart notice must appear: {seen:?}"
        );
        assert!(
            transcript_containing(&seen, "6 * 7"),
            "the fresh worker must serve the next request: {seen:?}"
        );

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    /// FFI (S20) and the DB-boot world loader (S22) had never been exercised
    /// TOGETHER before — `world/tests/30_ffi_alien_tests.mst` proves FFI
    /// works, but it isn't in `world.list`, so it's never part of the seeded
    /// image; the actual GUI boots via `load_world_from_image`
    /// (`boot_without_world` + two-phase replay), not the `.mst` file tree a
    /// plain `VmHandle::boot` would use. This defines the exact same
    /// `<primitive: FFI function: #getpid ret: #g args: #()>` class the S20
    /// test uses, live, against a DB-booted VM (the real Workspace/browser
    /// path — `live_compile`), and checks a real OS pid comes back.
    #[test]
    fn ffi_works_through_a_db_booted_vm() {
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_ffi_probe_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");

        vm.exec(
            "Object subclass: FFIProbe [\n\
             FFIProbe class >> getpid [\n\
             <primitive: FFI function: #getpid ret: #g args: #()>\n\
             ]\n\
             ]",
        )
        .expect("FFI class must compile against a DB-booted VM");
        let pid: i64 = vm
            .eval("FFIProbe getpid.")
            .expect("getpid must run")
            .parse()
            .expect("must answer a plain integer");
        assert!(pid > 0, "expected a real positive pid, got {pid}");

        std::fs::remove_file(&tmp).ok();
    }

    /// `Time class>>millisecondClockValue` (`world/30_date_time.mst`, S22) —
    /// a real wall-clock FFI primitive (`clock_gettime`) combined with a
    /// `<classVars:>`-cached mmap scratch buffer and Alien struct reads, all
    /// through the actual DB-boot path the real GUI uses. Checks two calls a
    /// moment apart return distinct, increasing, real epoch-millisecond
    /// values — not just that it runs without erroring.
    ///
    /// **macOS-only: the WORLD's clock is POSIX, not the GUI's.**
    /// `world/30_date_time.mst` implements this with `mmap` + `clock_gettime`
    /// and hardcoded macOS `PROT_*`/`MAP_*` flag values; neither symbol exists
    /// on Windows, so the FFI resolver correctly reports "no exported symbol
    /// named mmap" and the resulting guest error aborts the process (see the
    /// guest-fatal note on
    /// `worker_survives_an_unhandled_runtime_error_and_serves_the_next_request`).
    ///
    /// This is the same class of gap `../MIGRATION.md` M1 already records for
    /// the four FFI-dependent world test files, and the fix belongs to the
    /// world, not this crate: a Windows `Time` built on
    /// `GetSystemTimeAsFileTime` through the winkb resolver that the Win32 FFI
    /// work already landed. Durations are unaffected — `millisecondsToRun:`
    /// deliberately uses the VM's own monotonic `Smalltalk millisecondClock`.
    #[cfg(target_os = "macos")]
    #[test]
    fn time_millisecond_clock_value_works_through_a_db_booted_vm() {
        let world_dir = test_world_dir();
        let tmp =
            std::env::temp_dir().join(format!("macvm_clock_probe_{}.sqlite3", std::process::id()));
        std::fs::remove_file(&tmp).ok();
        let image = open_or_seed_image(&world_dir, &tmp).expect("seed");

        let mut vm = VmHandle::boot_without_world(macvm::runtime::VmOptions {
            heap_mib: 64,
            jit: macvm::runtime::JitMode::Off,
            ..Default::default()
        });
        crate::world_boot::load_world_from_image(&mut vm, &image, WORLD_LISTS, WORLD_DOITS).expect("db load");

        let parse_ms = |vm: &mut VmHandle| -> i64 {
            vm.eval("Time millisecondClockValue.")
                .expect("millisecondClockValue must run")
                .parse()
                .expect("must answer a plain integer")
        };
        let first = parse_ms(&mut vm);
        let second = parse_ms(&mut vm);

        // Sane epoch range (after 2023-01-01) — catches "answered garbage/
        // uninitialized memory" without pinning an exact value.
        assert!(
            first > 1_672_531_200_000,
            "not a plausible epoch ms: {first}"
        );
        assert!(
            second >= first,
            "the clock must not go backwards between two calls: {first} then {second}"
        );

        std::fs::remove_file(&tmp).ok();
    }
}
