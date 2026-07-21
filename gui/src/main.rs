//! MACVM GUI test shell — `../PLAN.md` phase G0 (faithful static shell) plus
//! a G1-shaped stub host (doit clicks echo to the in-page transcript,
//! satisfying that phase's "full JS↔Rust round trip" gate) so this is
//! usable as a real, clickable foundation now, not just a static render.
//!
//! No VM bridge yet (`GuiHost`/`VmHandle`, `../PLAN.md` G2, needs core eval —
//! S5/S6) — every doit/toolbar action just echoes into the Launcher-style
//! transcript pane via `evaluateJavaScript`, standing in for the real
//! `Transcript show:` round trip until G2.

mod browser_render;
mod canvas_render;
mod editor_render;
/// The native Metal game pane — macOS-only and off by default
/// (`Cargo.toml`'s `gamepane` feature; `../MIGRATION.md` §1 defers the game
/// demos on Windows, where D3D11/XAudio2 would be their counterpart).
#[cfg(feature = "gamepane")]
mod game_pane;
/// The dependency-free Objective-C bridge — used only by the macOS shell.
#[cfg(target_os = "macos")]
mod objc;
mod preprocess;
/// The native shell (window, web view, menus, timers, clipboard). Everything
/// platform-specific in this crate lives behind this one module — see
/// `shell/mod.rs`.
mod shell;
mod vm_host;
mod workspace_render;
// Moved to image_store (docs/package_aware_editing_design.md M2): that
// crate has no [lib] target, so nothing else could ever depend on it —
// cocoa_gui needs this same replay logic for its own primary's DB-boot (a
// later milestone). Re-exported under its old name so every existing
// `crate::world_boot::...` call site in this crate needed no changes.
pub(crate) use image_store::world_boot;

use preprocess::Theme;
use shell::ScriptMessage;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Set when `winvm-gui mandelvm` runs (the standalone MandelVM demo): the
/// window hosts only the game pane, so when the demo's frame loop stops there
/// is no browser to return to and the whole app quits instead.
#[cfg(feature = "gamepane")]
static MANDELVM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// True while the "Mandelbrot — spawned VM" demo is running: a genuinely SECOND
/// VM instance (`DEMO_VM`) drives the game pane while the GUI's own `VM` keeps
/// serving the Workspace/browser untouched.
#[cfg(feature = "gamepane")]
static DEMO_VM_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "gamepane")]
thread_local! {
    /// The second, spawned VM instance behind the "Mandelbrot — spawned VM" demo
    /// (a full `VmHost`: its own worker thread + fresh `VmHandle`, independent of
    /// the GUI's own `VM`). `Some` only while that demo is on screen; dropped at
    /// teardown, which closes its request channel and lets its worker exit.
    /// Main-thread only (spawned, driven, and dropped all from the main thread).
    static DEMO_VM: std::cell::RefCell<Option<vm_host::VmHost>> =
        const { std::cell::RefCell::new(None) };
}
static NAV: OnceLock<Mutex<NavState>> = OnceLock::new();

/// The GUI's handle onto the VM worker thread (`vm_host.rs`, SPEC §16.1 /
/// decision A18). Submitting a request from here just does a channel
/// send — the actual work, and the wake-up back to this thread, happen off
/// of it.
static VM: OnceLock<vm_host::VmHost> = OnceLock::new();

/// Current theme (`../PLAN.md` Theme menu), read on every `navigate_to` and
/// flipped by the native Theme menu's target/action handlers below. A plain
/// atomic INDEX into `Theme::ALL` rather than the `Mutex<NavState>` pattern
/// above: `Theme` is `Copy`, and AppKit only ever calls into this process on
/// the main thread (see `MainThreadPtr`'s doc comment), so there's no real
/// concurrency to protect against — `Ordering::Relaxed` is enough. Defaults
/// to `Theme::ALL`'s Hi-Def index rather than 0 (Classic) — the Theme menu
/// (and its checkmarks, computed from `current_theme()` at menu-build time)
/// still let you switch to any theme including Classic.
static THEME: AtomicU8 = AtomicU8::new(1); // Theme::ALL[1] == Theme::HiDef

fn current_theme() -> Theme {
    let idx = THEME.load(Ordering::Relaxed) as usize;
    Theme::ALL.get(idx).copied().unwrap_or(Theme::HiDef)
}

fn set_theme(theme: Theme) {
    if let Some(idx) = Theme::ALL.iter().position(|&t| t == theme) {
        THEME.store(idx as u8, Ordering::Relaxed);
    }
}

/// Page zoom percentage (G5 `biggerText`/`smallerText`, `../PLAN.md` §4),
/// read on every `navigate_to` same as `THEME`. Stored directly as the
/// percentage (not steps) — `u32` would be the natural type, but a `Copy`
/// atomic needs a fixed-width integer type with atomic support, and `AtomicU32`
/// is fine here since the value only ever moves in small increments from a
/// single thread.
static FONT_SCALE_PERCENT: AtomicU32 = AtomicU32::new(100);
const FONT_SCALE_STEP: u32 = 10;
const FONT_SCALE_MIN: u32 = 60;
const FONT_SCALE_MAX: u32 = 200;

fn current_font_scale_percent() -> u32 {
    FONT_SCALE_PERCENT.load(Ordering::Relaxed)
}

/// The Workspace's last-known text, kept for the lifetime of the app run
/// only (not written to disk anywhere) — `None` means "untouched this
/// run, show `workspace_render::initial_text()`'s placeholder comment";
/// `Some(s)` (even `Some(String::new())`, if the user deliberately
/// cleared it) means "restore exactly this." Updated from `smtk.js`'s
/// `input` listener on every keystroke in the workspace's textarea (a
/// `postMessage` per keystroke is cheap enough at this scale — see
/// `on_script_message`'s `"workspaceTextChanged"` arm), since this bridge
/// has no way to *pull* the current DOM value back out of a WKWebView
/// (`objc.rs`'s module doc: `evaluateJavaScript:completionHandler:` is
/// always called with a nil handler, so there's no return-value path).
static WORKSPACE_TEXT: Mutex<Option<String>> = Mutex::new(None);

/// The transcript's full history for this app run (not written to disk) —
/// appended to by `append_transcript`, the single function that already
/// pushes every new line into the *live* page via `evaluateJavaScript`, so
/// the persisted copy can never drift from what a running page shows.
/// Threaded into every freshly-built page (`preprocess::render_generated_page`/
/// `load_and_preprocess`'s new `transcript` parameter) so navigating,
/// switching themes, or changing font scale — all of which throw away and
/// rebuild the page — no longer silently erases transcript history the way
/// `statusbar_html`'s old hardcoded "Ready." greeting did.
static TRANSCRIPT: Mutex<String> = Mutex::new(String::new());

/// Empty only means "nothing's been printed yet this run" — same "show a
/// friendly placeholder instead of literally nothing" idea as
/// `workspace_render::initial_text()`.
fn current_transcript() -> String {
    let buf = TRANSCRIPT.lock().unwrap();
    if buf.is_empty() {
        "MACVM Transcript\nReady.".to_string()
    } else {
        buf.clone()
    }
}

fn bump_font_scale(delta: i32) {
    let current = current_font_scale_percent() as i32;
    let next = (current + delta).clamp(FONT_SCALE_MIN as i32, FONT_SCALE_MAX as i32);
    FONT_SCALE_PERCENT.store(next as u32, Ordering::Relaxed);
}

struct NavState {
    history: Vec<PathBuf>,
    index: usize,
}

impl NavState {
    fn new(start: PathBuf) -> Self {
        Self {
            history: vec![start],
            index: 0,
        }
    }
    fn current(&self) -> PathBuf {
        self.history[self.index].clone()
    }
    fn go(&mut self, path: PathBuf) {
        self.history.truncate(self.index + 1);
        self.history.push(path);
        self.index = self.history.len() - 1;
    }
    fn back(&mut self) -> Option<PathBuf> {
        if self.index > 0 {
            self.index -= 1;
            Some(self.current())
        } else {
            None
        }
    }
    fn forward(&mut self) -> Option<PathBuf> {
        if self.index + 1 < self.history.len() {
            self.index += 1;
            Some(self.current())
        } else {
            None
        }
    }
}

/// The GUI's resource root (start page, `assets/`, `reference/`, and the
/// `.rendered/` scratch dir). Defaults to this crate's source dir
/// (`CARGO_MANIFEST_DIR`) for a dev `cargo run`, but is overridable via
/// `MACVM_GUI_ROOT` so a packaged `.app` can point it at a bundled,
/// writable copy of `gui/` (the app launcher sets this) — the baked
/// build-time path doesn't exist on another machine, or after the source
/// tree moves.
fn gui_root() -> PathBuf {
    std::env::var_os("MACVM_GUI_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// A sentinel `NavState` entry for the class browser view — not a real
/// file (the browser is Rust-generated, `browser_render.rs`), but pushing
/// it onto `NavState` like any other page lets back/forward navigate to
/// and from it uniformly. `navigate_to` special-cases this exact path.
fn browser_view_marker() -> PathBuf {
    gui_root().join(".browser-view")
}

/// Same idea as `browser_view_marker`, for the Workspace tool
/// (`workspace_render.rs`).
fn workspace_view_marker() -> PathBuf {
    gui_root().join(".workspace-view")
}

/// Same idea as `browser_view_marker`, for the Canvas view
/// (`canvas_render.rs`, `../docs/CANVAS.md`).
fn canvas_view_marker() -> PathBuf {
    gui_root().join(".canvas-view")
}

/// Same idea, for the native Metal game pane (`game_pane.rs`,
/// `../docs/gamepane_design.md`). Unlike the other markers this swaps a native
/// `NSView` in as the window content view, not HTML into the WKWebView.
fn game_view_marker() -> PathBuf {
    gui_root().join(".game-view")
}

fn start_page() -> PathBuf {
    gui_root().join("reference/pages/startPage.html")
}

/// MACVM's own authored help viewer (`reference/pages/macvm-help/`) — the
/// MACVM documentation and tour. This is what the toolbar's Documentation
/// button, the start page's "Browse Documentation" link, and the native
/// Help menu all open. The original Strongtalk documentation corpus
/// (`reference/pages/documentation/`, copied byte-identical from
/// `../../strongtalk-repo`) is kept as reference material and reached via a
/// link from this index, not as the primary documentation.
fn macvm_help_index() -> PathBuf {
    gui_root().join("reference/pages/macvm-help/index.html")
}

/// Load, D-G4-preprocess, and display `path` in the web view.
///
/// Writes the transformed HTML to a scratch file under `gui/.rendered/`
/// and loads *that* via `loadFileURL:allowingReadAccessToURL:<gui root>`,
/// rather than `loadHTMLString:baseURL:` — WKWebView does not grant local
/// `file://` subresource access outside `baseURL`'s own directory for the
/// latter (confirmed by actually running the shell: `strongtalk.css`,
/// `smtk.js`, and every toolbar icon — all outside the page's own
/// directory — silently failed to load, leaving only the browser's
/// default stylesheet, easy to mistake for "it worked" at a glance since
/// default HTML rendering is passably similar in the wrong way). Granting
/// read access to the whole `gui/` root covers the rendered file, the
/// original page's own directory (so *its* relative links/images keep
/// resolving), and `assets/`/`reference/icons-png/` in one grant.
/// `macvm-gui render <page.html> [--theme NAME] [--world DIR] [-o OUT]` — the
/// headless "eyes" command. Runs the full page pipeline (preprocess + real-VM
/// smappl resolution + icon theming) with NO Cocoa window, inlines the theme
/// CSS so the output is self-contained, and writes it (default:
/// `gui/.rendered/headless.html`). View it in a browser to see exactly what a
/// corpus page renders as — the offline substitute for driving the native
/// WKWebView (which this objc bridge can't snapshot: no block ABI for
/// `takeSnapshotWithConfiguration:`).
fn cmd_render(args: &[String]) {
    let mut page: Option<PathBuf> = None;
    let mut theme = preprocess::Theme::Classic;
    let mut world_dir = PathBuf::from("world");
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--theme" => {
                i += 1;
                theme = args
                    .get(i)
                    .and_then(|n| preprocess::Theme::from_cli_name(n))
                    .unwrap_or(preprocess::Theme::Classic);
            }
            "--world" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    world_dir = PathBuf::from(d);
                }
            }
            "-o" | "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            other if page.is_none() => page = Some(PathBuf::from(other)),
            _ => {}
        }
        i += 1;
    }
    let Some(page) = page else {
        eprintln!("usage: macvm-gui render <page.html> [--theme NAME] [--world DIR] [-o OUT]");
        std::process::exit(2);
    };

    let chromed = match preprocess::load_and_preprocess(&page, theme, 100, "Ready") {
        Ok(h) => h,
        Err(e) => {
            eprintln!("render: cannot read {}: {e}", page.display());
            std::process::exit(1);
        }
    };

    let mut vm = match macvm::embed::VmHandle::boot(macvm::runtime::VmOptions::default(), &world_dir)
    {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("render: VM boot failed: {e:?}");
            std::process::exit(1);
        }
    };
    // One-shot CLI on the main thread: a genuinely-fatal render should exit
    // the process cleanly rather than pthread_exit the main thread.
    macvm::embed::set_fatal_mode(macvm::embed::FatalMode::ExitProcess);

    let resolved =
        preprocess::resolve_smappl_spans(&chromed, theme, |code| vm.render_fragment(code).ok());

    // Fill ClassOutliner method-source blocks from an image_store DB beside the
    // world, if one exists (the source of method text — the VM keeps none).
    let image_path = world_dir.join("image.sqlite3");
    let resolved = if image_path.exists() {
        match image_store::Image::open(&image_path) {
            Ok(img) => preprocess::inject_method_sources(&resolved, |c, s, sel| {
                let side = if s == "class" {
                    image_store::Side::Class
                } else {
                    image_store::Side::Instance
                };
                img.method_source(c, side, sel).ok().flatten()
            }),
            Err(_) => resolved,
        }
    } else {
        resolved
    };

    // Inline the theme CSS and smtk.js so the file renders self-contained and
    // interactive (the chrome's own `file://` stylesheet/script links won't
    // resolve when served over http). smtk.js's `post()` no-ops without a
    // native host, so client-side behavior (outliner expand/collapse) still
    // works in a plain browser.
    let gui_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let css = std::fs::read_to_string(theme.stylesheet_path()).unwrap_or_default();
    let js = std::fs::read_to_string(gui_root.join("assets/smtk.js")).unwrap_or_default();
    let head = format!("<style>{css}</style><script>{js}</script>");
    let self_contained = if resolved.contains("</head>") {
        resolved.replacen("</head>", &format!("{head}</head>"), 1)
    } else {
        format!("{head}{resolved}")
    };

    let out = out.unwrap_or_else(|| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push(".rendered");
        std::fs::create_dir_all(&p).ok();
        p.push("headless.html");
        p
    });
    if let Err(e) = std::fs::write(&out, self_contained) {
        eprintln!("render: cannot write {}: {e}", out.display());
        std::process::exit(1);
    }
    println!("{}", out.display());
}

/// `macvm-gui seed [--world DIR]` — import `world/*.mst` into
/// `<world_dir>/image.sqlite3` without launching the GUI. Headless complement
/// to `render`: the class browser and the ClassOutliner's method-source
/// blocks read their text from this DB (the running VM keeps no source).
fn cmd_seed(args: &[String]) {
    let mut world_dir = PathBuf::from("world");
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--world" {
            i += 1;
            if let Some(d) = args.get(i) {
                world_dir = PathBuf::from(d);
            }
        }
        i += 1;
    }
    let image_path = world_dir.join("image.sqlite3");
    let image = match image_store::Image::open(&image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("seed: cannot open {}: {e}", image_path.display());
            std::process::exit(1);
        }
    };
    match image_store::import::import_all_lists(&image, &world_dir) {
        Ok(stats) => {
            let sends = image.backfill_method_sends().unwrap_or(0);
            let lists = image.package_lists().unwrap_or_default();
            println!(
                "seeded {} ({stats:?}, {sends} send-edges indexed, package lists: {lists:?})",
                image_path.display()
            );
        }
        Err(e) => {
            eprintln!("seed: import failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `macvm-gui run <file.mst> [--world DIR] [--from-image]` (M7,
/// `docs/package_aware_editing_design.md` §4.5/§6). Boots a VM and runs a
/// `.mst` program to completion, exactly as the bare `macvm run` CLI does —
/// with an OPT-IN `--from-image` that boots the world from the SQLite image
/// (via the M2-extracted `world_boot::load_world_from_image`, requesting the
/// base `WORLD_LISTS`) instead of the `.mst`-direct default.
///
/// Why this lives in `macvm-gui`, not the bare `macvm` bin (the design doc's
/// literal M7 target): `image_store` depends on `macvm`, so the root `macvm`
/// bin cannot depend on `image_store` — a dependency cycle. This crate
/// depends on both, so it is the CLI home for a DB-boot path. The `.mst`
/// default deliberately stays the bare bin's only mode: it is the DB-free
/// ground-truth boot oracle the differential tests
/// (`world_image_installs_all_classes_without_error`) compare against.
///
/// Both boot paths must produce identical output — the image is a faithful,
/// byte-identical projection of the `.mst` world
/// (`image_store::world_boot::db_booted_world_matches_mst_booted_world` proves
/// the boot equivalence at the library level; `tests/it_run_from_image.rs`
/// proves it end-to-end through THIS command).
fn cmd_run_gui(args: &[String]) {
    let mut world_dir = PathBuf::from("world");
    let mut from_image = false;
    let mut file: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--world" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    world_dir = PathBuf::from(d);
                }
            }
            "--from-image" => from_image = true,
            other if file.is_none() => file = Some(other.to_string()),
            other => {
                eprintln!("macvm-gui run: unexpected argument {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let Some(file) = file else {
        eprintln!("usage: macvm-gui run <file.mst> [--world DIR] [--from-image]");
        std::process::exit(2);
    };

    let opts = vm_host::gui_vm_options();
    let mut vm = if from_image {
        let image_path = vm_host::resolve_image_path(&world_dir);
        let image = match image_store::import::open_or_seed(&world_dir, &image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("macvm-gui run --from-image: {e}");
                std::process::exit(1);
            }
        };
        let mut vm = macvm::embed::VmHandle::boot_without_world(opts);
        if let Err(e) = world_boot::load_world_from_image(
            &mut vm,
            &image,
            vm_host::WORLD_LISTS,
            vm_host::WORLD_DOITS,
        ) {
            eprintln!("macvm-gui run --from-image: DB-boot failed: {e}");
            std::process::exit(1);
        }
        vm
    } else {
        match macvm::embed::VmHandle::boot(opts, &world_dir) {
            Ok(vm) => vm,
            Err(e) => {
                eprintln!("macvm-gui run: boot failed: {}", e.msg);
                std::process::exit(1);
            }
        }
    };

    match vm.run_file(Path::new(&file)) {
        Ok(()) => std::process::exit(vm.exit_code().unwrap_or(0)),
        Err(e) => {
            eprintln!("{}", e.msg);
            std::process::exit(1);
        }
    }
}

/// `macvm-gui export --world DIR` — write the image back over `DIR`'s
/// `world/*.mst` (in place, surgically) so interactive edits can be reviewed and
/// checked into source control. Headless complement to `seed` (the reverse
/// direction); the GUI's File ▸ "Save World to Files" runs the same code.
fn cmd_export(args: &[String]) {
    let mut world_dir = PathBuf::from("world");
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--world" {
            i += 1;
            if let Some(d) = args.get(i) {
                world_dir = PathBuf::from(d);
            }
        }
        i += 1;
    }
    let image_path = std::env::var_os("MACVM_IMAGE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| world_dir.join("image.sqlite3"));
    let image = match image_store::Image::open(&image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("export: cannot open {}: {e}", image_path.display());
            std::process::exit(1);
        }
    };
    match image_store::export::export_world_dir(&image, &world_dir) {
        Ok(stats) => println!(
            "exported {} -> {} ({stats:?})",
            image_path.display(),
            world_dir.display()
        ),
        Err(e) => {
            eprintln!("export: failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Displays `path` (a view marker or a corpus file). Returns `false` if a file
/// load failed — the caller must NOT commit a failed target to history, or a
/// broken relative link's bad path becomes `current()` and the next relative
/// click resolves against it, compounding (the documentation-tour
/// `tour/tour/tour/…` bug). Marker views always succeed.
fn navigate_to(path: &Path) -> bool {
    if path == browser_view_marker() {
        // Back/forward/refresh landing back on the browser marker: NAV
        // already points here (unlike `open_class_browser`, which pushes
        // it), so just re-request current state — same "never show a page
        // before its data arrives" rule applies on the way back in, too.
        request_browser_open();
        return true;
    }
    if path == workspace_view_marker() {
        // Unlike the browser marker, this rebuilds and displays the page
        // directly — see `open_workspace`'s doc comment for why there's no
        // VM round trip (or content persistence) involved here.
        display_workspace();
        return true;
    }
    if path == canvas_view_marker() {
        // Same shape as the workspace marker: rebuild and display directly.
        // Whatever was drawn is lost on navigating away, same accepted
        // limitation `open_workspace`'s doc comment already notes for its
        // own content — see `open_canvas`'s doc comment.
        display_canvas();
        return true;
    }
    if path == class_outliner_view_marker() {
        // Rebuild the smappl class-outliner page; its `<smappl>` resolves
        // via the worker on load, same as any corpus page.
        display_class_outliner();
        return true;
    }
    if path == editor_view_marker() {
        // Rebuild the editor picker (empty buffer). A specific class is
        // loaded via the `editorOpen` post, not the marker, so back/forward
        // to the editor always lands on the picker — fine for M0.
        display_editor(None);
        return true;
    }
    if path == game_view_marker() {
        // Swap the native Metal game pane in as the window content view and
        // render a frame — not HTML into the web view (see `game_pane.rs`).
        #[cfg(feature = "gamepane")]
        display_game_pane();
        #[cfg(not(feature = "gamepane"))]
        report_no_game_pane();
        return true;
    }
    if let Some(tool) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix(".find-"))
    {
        display_find(tool);
        return true;
    }
    let html = match preprocess::load_and_preprocess(
        path,
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    ) {
        Ok(html) => html,
        Err(e) => {
            eprintln!("macvm-gui: failed to load {}: {e}", path.display());
            return false;
        }
    };
    display_html(&html);
    true
}

/// Render the class browser's initial state and display it — shared by
/// `open_class_browser` (first open) and `navigate_to`/`reload_current_page`
/// (back/forward/refresh landing back on the browser view). Synchronous,
/// local `MockWorld`/`BrowserSelection` render, not a round trip through
/// `vm_host` — see `open_class_browser`'s doc comment for why.
/// Opens the class browser view (the "hierarchy" toolbar button — real
/// Strongtalk's own "Full hierarchy" action, `../PLAN.md` §1) — and
/// likewise what `navigate_to` calls when back/forward/refresh lands back
/// on the browser marker. Deliberately does **not** render or load
/// anything itself: it only pushes the NAV marker and asks the worker
/// thread for the current state. The actual page only ever gets built and
/// loaded once, from real data, when the matching `VmResponse::BrowserOpened`
/// arrives (`vm_bridge_drain`) — see `browser_render::assemble_panes`'s doc
/// comment for the race this avoids (an earlier version rendered a
/// hardcoded mock page here immediately, then tried to asynchronously
/// patch in real data — which could lose the race against the page still
/// loading and silently no-op).
fn open_class_browser() {
    let marker = browser_view_marker();
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(marker);
    }
    request_browser_open();
}

fn request_browser_open() {
    if let Some(vm) = VM.get() {
        vm.submit(vm_host::VmRequest::BrowserOpen);
    }
}

/// Opens the Workspace tool (`workspace_render.rs`, the toolbar's
/// long-unwired "Workspace" button — `../PLAN.md`'s nine, `blankSheet`
/// icon). Unlike `open_class_browser`, this builds and displays the page
/// synchronously, with no VM round trip at all: the browser's initial
/// content depends on worker-thread state (`MockWorld`), but the
/// Workspace's content doesn't depend on anything the worker thread
/// knows — it's just a starting placeholder comment, so there's nothing
/// to race and nothing to wait for.
///
/// Known limitation, accepted rather than solved here: whatever's typed
/// into the workspace lives only in the page's own DOM, which a fresh
/// `loadFileURL:` throws away — navigating away and back (or
/// quitting) loses it, same as any unsaved scratch buffer. Worth
/// revisiting (e.g. persisting the latest text to a small in-memory
/// static, mirroring `FONT_SCALE_PERCENT`/`THEME`) if that turns out to
/// be annoying in practice; not built now since it's not needed for the
/// hooks themselves to be real.
fn open_workspace() {
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(workspace_view_marker());
    }
    display_workspace();
}

/// The Smalltalk text editor (`docs/editor_design.md`, the toolbar's
/// `texteditor` icon). M0: pick a class, see its whole source rendered from
/// the image (`Image::class_source`) as read-only text. Later milestones make
/// it editable, syntax-checked, and backed by a per-document worker VM.
fn editor_view_marker() -> PathBuf {
    gui_root().join(".editor-view")
}

/// The image the editor reads class text from — same resolution the boot path
/// uses (`MACVM_IMAGE_PATH`, else `<world>/image.sqlite3`). The GUI keeps no
/// global world dir on the main thread, so this mirrors the boot default.
fn editor_image_path() -> PathBuf {
    std::env::var_os("MACVM_IMAGE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("world").join("image.sqlite3"))
}

fn open_editor() {
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(editor_view_marker());
    }
    display_editor(None);
}

/// Render the editor page. `class_name = None` shows the picker with an empty
/// buffer; `Some(name)` renders that class's source (read-only in M0). The
/// body itself is built by the pure, unit-tested `editor_render` module; this
/// function is just the image read plus the Objective-C display.
fn display_editor(class_name: Option<&str>) {
    let image = image_store::Image::open(&editor_image_path()).ok();
    let class_names = image
        .as_ref()
        .and_then(|img| img.class_names().ok())
        .unwrap_or_default();
    let source = class_name.and_then(|name| {
        image
            .as_ref()
            .and_then(|img| img.class_source(name).ok().flatten())
    });
    let (current, source) = editor_render::buffer_for(class_name, source);
    let body = editor_render::render_editor(&class_names, &current, &source);
    let html = preprocess::render_generated_page(
        "Text Editor",
        &body,
        &gui_root(),
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    );
    display_html(&html);
}

/// The "User hierarchy" toolbar button (`preprocess::TOOLBAR_BUTTONS`) — a
/// full-page class browser built on the smappl `ClassHierarchyOutliner`
/// (click a class name to drill into its editable method browser). Distinct
/// from the "Full hierarchy" button, which opens the older MockWorld browser.
fn class_outliner_view_marker() -> PathBuf {
    gui_root().join(".class-outliner-view")
}

fn open_class_outliner() {
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(class_outliner_view_marker());
    }
    display_class_outliner();
}

/// The find-tool pages (Implementors / Senders / Find definition) — a search
/// input over the reflection layer (docs/APPS.md §5.4). Distinct marker per
/// tool so back/forward/refresh rebuild the right one.
fn find_view_marker(tool: &str) -> PathBuf {
    gui_root().join(format!(".find-{tool}"))
}

fn open_find(tool: &str) {
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(find_view_marker(tool));
    }
    display_find(tool);
}

fn display_find(tool: &str) {
    let (title, placeholder) = match tool {
        "senders" => ("Senders", "selector, e.g. printOn:"),
        "definition" => ("Find Definition", "class name, e.g. Collection"),
        _ => ("Implementors", "selector, e.g. printOn:"),
    };
    // The results div carries a widget id + hierarchy root so clicking a
    // result (a .st-class-link) drills into that class via the same
    // smapplOpenClass path the outliner uses.
    // A dropdown+search combobox: the text input filters as you type against a
    // <datalist> of options (class names / selectors) that smtk.js requests from
    // the worker (image_store) on load — `VmRequest::FindOptions`. Pick an option
    // or type freely, then Enter runs the query.
    let body = format!(
        "<h1>{title}</h1>\
         <p>Type or pick from the list, then press Enter.</p>\
         <input class=\"st-find-input\" type=\"text\" data-find-tool=\"{tool}\" \
          list=\"st-find-options\" placeholder=\"{placeholder}\" autocomplete=\"off\" spellcheck=\"false\">\
         <datalist id=\"st-find-options\"></datalist>\
         <div id=\"find-results\" data-widget-id=\"find\" data-hierarchy-root=\"Object\"></div>"
    );
    let html = preprocess::render_generated_page(
        title,
        &body,
        &gui_root(),
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    );
    display_html(&html);
}

fn display_class_outliner() {
    // A `<smappl>` tag: `render_generated_page` rewrites it to a placeholder
    // span (like any corpus page), and smtk.js resolves it via the worker on
    // load — no smappl evaluation happens here on the main thread.
    let body = "<h1>Class Browser</h1>\
        <p>Click a class name to open its methods; the &#9656; toggles subclasses.</p>\
        <smappl visual=\"ClassHierarchyOutliner imbeddedVisualForClass: Object\">";
    let html = preprocess::render_generated_page(
        "Class Browser",
        body,
        &gui_root(),
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    );
    display_html(&html);
}

fn display_workspace() {
    let text = WORKSPACE_TEXT
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| workspace_render::initial_text().to_string());
    let body = workspace_render::render_workspace(&text);
    let html = preprocess::render_generated_page(
        "Workspace",
        &body,
        &gui_root(),
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    );
    display_html(&html);
}

/// Opens the Canvas view (`canvas_render.rs`, `../docs/CANVAS.md`, the
/// toolbar's "abstract" icon). Displays synchronously at the default
/// size, exactly like `open_workspace`, then *also* submits a real
/// `CanvasCreate` request — "allocate a widget of a specific size" is a
/// genuine round trip through `vm_host`, not just a static initial
/// render, so opening the view exercises it for real every time, not
/// only when "Run Demo" is clicked.
///
/// Same accepted limitation as Workspace: whatever's drawn lives only in
/// the page's own `<canvas>` pixel buffer, which a fresh `loadFileURL:`
/// throws away on navigating elsewhere.
fn open_canvas() {
    if let Some(nav) = NAV.get() {
        nav.lock().unwrap().go(canvas_view_marker());
    }
    display_canvas();
    if let Some(vm) = VM.get() {
        vm.submit(vm_host::VmRequest::CanvasCreate {
            width: canvas_render::DEFAULT_WIDTH,
            height: canvas_render::DEFAULT_HEIGHT,
        });
    }
}

// ── Native game pane (macOS-only, `gamepane` feature) ──────────────────────
//
// `../MIGRATION.md` §1 defers the game demos on Windows (D3D11/XAudio2 would
// be their counterpart), and the pane is Metal-backed, so this whole section
// compiles out unless the feature is on. The toolbar's `gamePane` button and
// the `.game-view` marker are handled without it — see `navigate_toolbar`.
#[cfg(feature = "gamepane")]
mod game_pane_driver {
    use super::*;

/// Opens the native Metal game pane (`game_pane.rs`,
    /// `../docs/gamepane_design.md`) — the toolbar's `gamePane` action and the
    /// `MACVM_GAMEPANE_DEMO` env trigger. Pushes the marker and swaps the native
    /// view in, mirroring `open_canvas`'s marker/display split.
    fn open_game_pane() {
        if let Some(nav) = NAV.get() {
            nav.lock().unwrap().go(game_view_marker());
        }
        display_game_pane();
    }
    
    /// Swap the native game pane's `NSView` in as the window content view (out of
    /// the WKWebView) and render one frame. Main thread only. A no-op with a note
    /// if this Mac has no Metal device.
    fn display_game_pane() {
        let Some(view) = game_pane::ensure_native_view(game_pane::PANE_W, game_pane::PANE_H) else {
            append_transcript("(game pane: no Metal device available)");
            return;
        };
        // Start each game session with a clean held-key table. HELD_KEYS is only
        // cleared by a keyUp: delivered to the game view, so the Escape that closed
        // the PREVIOUS game — its keyUp: went to the webview that replaced the pane,
        // never to us — would otherwise still read as held here, and the frame
        // loop's level-triggered Escape check would close this new game on its very
        // first tick. Clearing on open fixes that "second demo exits immediately".
        macgamepane_graphics::input::clear_all();
        if let Some(window) = WINDOW.get() {
            objc::send1_id(window.0, sel("setContentView:"), view);
            // Make the key-capable game view first responder so keyDown:/keyUp:
            // populate HELD_KEYS while the game is focused (docs/gamepane_design.md).
            objc::send1_id(window.0, sel("makeFirstResponder:"), view);
        }
        game_pane::render_native_frame();
    }
    
    // ── Game-loop frame driver (docs/gamepane_design.md M4) ─────────────────────
    
    /// The `MacvmGameTimer` target object whose `gameTick:` runs [`on_game_tick`].
    fn build_game_timer_target() -> Id {
        let cls = objc::allocate_class(objc::get_class("NSObject"), "MacvmGameTimer");
        objc::add_method(cls, sel("gameTick:"), on_game_tick as *const _, "v@:@");
        objc::register_class(cls);
        objc::alloc_init("MacvmGameTimer")
    }
    
    /// NSTimer callback (~60 Hz on the main run loop while a game loop runs): if the
    /// worker is idle, submit exactly one `GameStep` carrying this tick's held keys.
    /// Single-outstanding — a still-running step means we skip, never pile up.
    /// Rendering happens later, when the step's draw commands drain (not here).
    extern "C" fn on_game_tick(_this: Id, _cmd: Sel, _timer: Id) {
        // Escape ends the game. Normally that returns to the browser
        // (`close_game_pane`); in the standalone mandelvm demo it quits the app, and
        // in the spawned-VM demo it tears the second VM down — both via the frame
        // loop's stop path. macOS virtual key code 53 = Escape.
        if macgamepane_graphics::input::key_held(53) {
            if DEMO_VM_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
                || MANDELVM.load(std::sync::atomic::Ordering::Relaxed)
            {
                on_game_loop_stopped();
            } else {
                close_game_pane();
            }
            return;
        }
        let keys = current_game_key_mask();
        // The spawned-VM demo drives the SECOND VM instance; every other game runs in
        // the GUI's own VM. Same single-outstanding backpressure (`is_idle`) either way.
        if DEMO_VM_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            DEMO_VM.with(|d| {
                if let Some(vm) = d.borrow().as_ref() {
                    if vm.is_idle() {
                        vm.submit(vm_host::VmRequest::GameStep { keys });
                    }
                }
            });
            return;
        }
        let Some(vm) = VM.get() else { return };
        if !vm.is_idle() {
            return;
        }
        vm.submit(vm_host::VmRequest::GameStep { keys });
    }
    
    /// Close the game pane and return to the GUI (bound to Escape): stop the frame
    /// loop, restore the WKWebView (by navigating back to the pre-game page, whose
    /// load swaps the content view back — see `display_html`), make it first
    /// responder, then drop the native pane so a reopen starts fresh. Order matters:
    /// swap the game view out of the window *before* freeing its GPU resources.
    fn close_game_pane() {
        stop_game_loop_timer();
        // Reset the VM-side game state so the closed demo leaves nothing behind:
        // GamePane>>reset nils the registered step block (releasing the demo object
        // it closes over) and clears the held-key mask. teardown() below drops the
        // native pane's own state; together that is a full reset, so the next demo
        // launch starts clean. Queued to the worker (FIFO), so a later launch runs
        // after it. (NextId stays monotonic by design — that is what keeps a stale
        // sprite id from aliasing a fresh one; see gamepane_design.md.)
        if let Some(vm) = VM.get() {
            vm.submit(vm_host::VmRequest::Doit {
                code: "GamePane reset.".to_string(),
            });
        }
        let prev = NAV
            .get()
            .and_then(|n| n.lock().unwrap().back())
            .unwrap_or_else(start_page);
        navigate_to(&prev);
        if let (Some(window), Some(webview)) = (WINDOW.get(), WEBVIEW.get()) {
            objc::send1_id(window.0, sel("makeFirstResponder:"), webview.0);
        }
        game_pane::teardown();
    }
    
    /// Pack MacGamePane's process-global held-key table into the bitmask
    /// `GamePane>>keyHeld:` reads (bit 0=Left … 5=B), by macOS virtual key code.
    fn current_game_key_mask() -> i64 {
        // Left, Right, Up, Down, Space (A), Z (B) — matches GamePane class>>keyLeft…keyB.
        const CODES: [u16; 6] = [123, 124, 126, 125, 49, 6];
        let mut mask = 0i64;
        for (bit, code) in CODES.iter().enumerate() {
            if macgamepane_graphics::input::key_held(*code) {
                mask |= 1 << bit;
            }
        }
        mask
    }
    
    /// Start (or restart) the frame timer — called on `StartLoop` (`GamePane>>run`).
    pub(crate) fn start_game_loop_timer() {
        let target = GAME_TIMER_TARGET
            .get_or_init(|| MainThreadPtr(build_game_timer_target()))
            .0;
        GAME_TIMER.with(|t| {
            let existing = t.get();
            if existing != NIL {
                objc::send0(existing, sel("invalidate"));
            }
            t.set(objc::scheduled_timer(1.0 / 60.0, target, sel("gameTick:"), true));
        });
    }
    
    /// Stop the frame timer — called on `StopLoop` (`GamePane>>stop`).
    pub(crate) fn stop_game_loop_timer() {
        GAME_TIMER.with(|t| {
            let existing = t.get();
            if existing != NIL {
                objc::send0(existing, sel("invalidate"));
                t.set(NIL);
            }
        });
    }
    
    /// True while the native game-pane frame loop is running (the 60 Hz timer is
    /// scheduled). `game_pane::present_if_dirty` uses this to stay out of the way
    /// during the loop, so a frame is shown only by its own explicit `Present` and
    /// never mid-stream (the anti-flicker invariant — see `present_if_dirty`).
    pub(crate) fn game_loop_active() -> bool {
        GAME_TIMER.with(|t| t.get() != NIL)
    }
    
    /// A game's frame loop has stopped — the guest sent `StopLoop` (`pane stop`),
    /// e.g. `MandelVM` after finishing its one dive. Always stops the timer; in the
    /// standalone `mandelvm` mode there is no browser to fall back to, so the demo
    /// ending means the app (and its VM instance) exits.
    pub(crate) fn on_game_loop_stopped() {
        stop_game_loop_timer();
        if DEMO_VM_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            // The spawned-VM demo ended: drop that second instance, return to the GUI.
            teardown_spawned_demo();
        } else if MANDELVM.load(std::sync::atomic::Ordering::Relaxed) {
            // The standalone `mandelvm` window has nothing to return to: quit.
            let app = objc::send0(objc::get_class("NSApplication"), sel("sharedApplication"));
            objc::send1_id(app, sel("terminate:"), NIL);
        }
    }
}

#[cfg(feature = "gamepane")]
pub(crate) use game_pane_driver::*;

// ── VM metrics dashboard (native sampler → ring buffer → toolbar) ────────────
//
// A main-thread NSTimer samples THIS VM's live signals (`compiled_depth`, via
// the per-VM `Arc` the worker handed over at boot) at ~15 Hz, and about once a
// second polls a full counter snapshot (`VmRequest::GetMetrics`). Each snapshot
// folds that second's worth of exec-mode samples into one `MetricsSample` and
// pushes it into a bounded RING BUFFER; the ring is processed into the current
// readout plus tiny sparklines and shipped to the webview toolbar
// (`#macvm-metrics`). Everything is per-VM — the atomic lives in the VM's own
// `Arc` and the ring is per this `VmHost` — so several VMs never blend.

/// Ring capacity — one sample per snapshot (~1 Hz) ⇒ ~2 minutes of history.
const METRICS_RING_CAP: usize = 120;
/// Sampler ticks per snapshot poll: a 15 Hz timer, poll ~once a second.
const METRICS_TICKS_PER_POLL: u32 = 15;
/// Points sent to the webview per sparkline (the tail of the ring).
const METRICS_SPARK_POINTS: usize = 48;

/// One processed metrics sample (~one per second), the unit stored in the ring.
#[derive(Clone, Copy, Default)]
struct MetricsSample {
    heap_used: u64,
    heap_cap: u64,
    alloc_rate: u64, // bytes/sec since the previous sample
    nmethods: u64,
    code_used: u64,
    scavenges: u64,
    full_gcs: u64,
    compiled_frac: f32, // interp/compiler execution ratio this window (0..1); NaN if idle
}

#[derive(Default)]
struct MetricsState {
    ring: std::collections::VecDeque<MetricsSample>,
    prev_alloc: u64,
    have_prev_alloc: bool,
    tick: u32,
    compiled_ticks: u32, // exec-mode accumulators over the current window
    busy_ticks: u32,
    poll_pending: bool, // single-outstanding GetMetrics
}

thread_local! {
    static METRICS: std::cell::RefCell<MetricsState> =
        std::cell::RefCell::new(MetricsState::default());
}

/// The ~15 Hz sampler tick (UI thread), driven by the shell's timer. Records
/// whether the VM is executing compiled code right now (from its per-VM
/// live-signal `Arc`), and once a second asks the worker for a full counter
/// snapshot.
pub(crate) fn on_metrics_tick() {
    let Some(vm) = VM.get() else { return };
    // Exec-mode sample: only meaningful while the VM is busy (idle == neither
    // interpreting nor compiling). compiled_depth > 0 == in compiled code.
    if !vm.is_idle() {
        let in_compiled = vm
            .live_stats()
            .is_some_and(|ls| ls.compiled_depth.load(Ordering::Relaxed) > 0);
        METRICS.with(|m| {
            let mut m = m.borrow_mut();
            m.busy_ticks += 1;
            if in_compiled {
                m.compiled_ticks += 1;
            }
        });
    }
    // Once per ~second: request a snapshot (single-outstanding, no pile-up).
    let want_poll = METRICS.with(|m| {
        let mut m = m.borrow_mut();
        m.tick += 1;
        if m.tick >= METRICS_TICKS_PER_POLL {
            m.tick = 0;
            if !m.poll_pending {
                m.poll_pending = true;
                return true;
            }
        }
        false
    });
    if want_poll {
        vm.submit(vm_host::VmRequest::GetMetrics);
    }
}

/// A counter snapshot came back: fold the window's exec-mode samples into one
/// ring sample, then render the ring to the toolbar.
fn metrics_on_snapshot(m: macvm::embed::VmMetrics) {
    let payload = METRICS.with(|st| {
        let mut st = st.borrow_mut();
        st.poll_pending = false;

        let alloc_rate = if st.have_prev_alloc {
            m.bytes_allocated.saturating_sub(st.prev_alloc)
        } else {
            0
        };
        st.prev_alloc = m.bytes_allocated;
        st.have_prev_alloc = true;

        let compiled_frac = if st.busy_ticks > 0 {
            st.compiled_ticks as f32 / st.busy_ticks as f32
        } else {
            // Idle window: carry the last known ratio rather than snap to 0.
            st.ring.back().map_or(f32::NAN, |s| s.compiled_frac)
        };
        st.compiled_ticks = 0;
        st.busy_ticks = 0;

        st.ring.push_back(MetricsSample {
            heap_used: m.eden_used + m.old_used,
            heap_cap: m.eden_capacity + m.old_reserved,
            alloc_rate,
            nmethods: m.nmethods,
            code_used: m.code_used,
            scavenges: m.scavenges,
            full_gcs: m.full_gcs,
            compiled_frac,
        });
        while st.ring.len() > METRICS_RING_CAP {
            st.ring.pop_front();
        }
        build_metrics_payload(&st)
    });
    eval_js(&format!(
        "if(window.macvmSetMetrics)window.macvmSetMetrics({payload})"
    ));
}

/// Serialize the ring's tail into the compact JSON the toolbar renderer wants:
/// current values plus short sparkline arrays (a `null` ratio point == idle).
fn build_metrics_payload(st: &MetricsState) -> String {
    let last = st.ring.back().copied().unwrap_or_default();
    let skip = st.ring.len().saturating_sub(METRICS_SPARK_POINTS);
    let frac_str = |f: f32| {
        if f.is_nan() {
            "null".to_string()
        } else {
            format!("{f:.3}")
        }
    };
    let heap_spark: Vec<String> = st
        .ring
        .iter()
        .skip(skip)
        .map(|s| s.heap_used.to_string())
        .collect();
    let alloc_spark: Vec<String> = st
        .ring
        .iter()
        .skip(skip)
        .map(|s| s.alloc_rate.to_string())
        .collect();
    let ratio_spark: Vec<String> = st
        .ring
        .iter()
        .skip(skip)
        .map(|s| frac_str(s.compiled_frac))
        .collect();
    format!(
        "{{\"heapUsed\":{},\"heapCap\":{},\"allocRate\":{},\"nmethods\":{},\"codeUsed\":{},\
         \"scavenges\":{},\"fullGcs\":{},\"compiledFrac\":{},\
         \"heapSpark\":[{}],\"allocSpark\":[{}],\"ratioSpark\":[{}]}}",
        last.heap_used,
        last.heap_cap,
        last.alloc_rate,
        last.nmethods,
        last.code_used,
        last.scavenges,
        last.full_gcs,
        frac_str(last.compiled_frac),
        heap_spark.join(","),
        alloc_spark.join(","),
        ratio_spark.join(","),
    )
}

fn display_canvas() {
    let body =
        canvas_render::render_canvas(canvas_render::DEFAULT_WIDTH, canvas_render::DEFAULT_HEIGHT);
    let html = preprocess::render_generated_page(
        "Canvas",
        &body,
        &gui_root(),
        current_theme(),
        current_font_scale_percent(),
        &current_transcript(),
    );
    display_html(&html);
}

/// The "write to `.rendered/current.html`, then `loadFileURL:` it" half of
/// `navigate_to`, factored out so Rust-generated pages (the class browser
/// view, `open_class_browser`) can reuse the exact same load mechanism —
/// and its doc comment's reasoning — without going through a corpus file
/// on disk first.
fn display_html(html: &str) {
    let rendered_dir = gui_root().join(".rendered");
    if let Err(e) = std::fs::create_dir_all(&rendered_dir) {
        eprintln!("winvm-gui: failed to create {}: {e}", rendered_dir.display());
        return;
    }
    // A single reusable filename: navigation is synchronous from the main
    // thread and one-at-a-time, so there's no concurrent writer to race.
    let rendered_path = rendered_dir.join("current.html");
    if let Err(e) = std::fs::write(&rendered_path, html) {
        eprintln!(
            "winvm-gui: failed to write {}: {e}",
            rendered_path.display()
        );
        return;
    }
    shell::load_rendered_page(&rendered_path);
}

fn eval_js(js: &str) {
    shell::eval_js(js);
}

fn js_string_literal(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "")
    )
}

fn append_transcript(text: &str) {
    {
        let mut buf = TRANSCRIPT.lock().unwrap();
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(text);
    }
    eval_js(&format!(
        "window.macvmAppendTranscript({})",
        js_string_literal(text)
    ));
}

/// Cut/Select All from `smtk.js`'s custom context menu (`../PLAN.md`'s
/// native-context-menu takeover). Copy and Paste deliberately do NOT come
/// through here — they go straight to the real clipboard below, because
/// routing them through the host's own editing machinery never reliably
/// reached a web-view selection (the user-visible "copy does nothing" bug).
fn send_edit_action(action: &str) {
    shell::edit_action(action);
}

/// The REAL clipboard: smtk.js captures the selected text at
/// context-menu-open time and ships it with the editAction message; this
/// writes it straight onto the system clipboard.
fn clipboard_write(text: &str) {
    shell::clipboard_write(text);
}

/// The paste half: read the clipboard's string (empty when it holds no text)
/// — smtk.js inserts it at the focused field's cursor via `macvmPasteText`.
fn clipboard_read() -> String {
    shell::clipboard_read()
}

/// Run a toolbar action by name — shared by the native toolbar buttons
/// (`kind:"toolbar"`) and by a tour page's smappl button navigating to its
/// MACVM equivalent (`VmResponse::SmapplNavigate`, e.g. `Launcher launchers
/// anElement openClassHierarchyOutliner` → `hierarchy`). Action names match the
/// `TOOLBAR_BUTTONS` data-action strings.
/// The native game pane is macOS-only and off by default (`Cargo.toml`'s
/// `gamepane` feature; `../MIGRATION.md` §1 defers the game demos on
/// Windows, where D3D11/XAudio2 would be their counterpart). Say so in the
/// transcript rather than having the button do nothing at all, which reads
/// as a bug.
#[cfg(not(feature = "gamepane"))]
fn report_no_game_pane() {
    append_transcript(
        "(game pane: not built in — it is macOS-only and off by default; see MIGRATION.md §1)",
    );
}

fn navigate_toolbar(button: &str) {
    match button {
        "goBack" => {
            if let Some(path) = NAV.get().and_then(|n| n.lock().unwrap().back()) {
                navigate_to(&path);
            }
        }
        "goForward" => {
            if let Some(path) = NAV.get().and_then(|n| n.lock().unwrap().forward()) {
                navigate_to(&path);
            }
        }
        "home" => {
            let home = start_page();
            if let Some(nav) = NAV.get() {
                nav.lock().unwrap().go(home.clone());
            }
            navigate_to(&home);
        }
        "documentation" => {
            // The toolbar Documentation button opens MACVM's own help (which
            // hosts the MACVM tour). The Strongtalk reference corpus is still
            // reachable as a link from that index.
            let doc_index = macvm_help_index();
            if let Some(nav) = NAV.get() {
                nav.lock().unwrap().go(doc_index.clone());
            }
            navigate_to(&doc_index);
        }
        "userHierarchy" => open_class_outliner(),
        "find" => open_find("definition"),
        "implementors" => open_find("implementors"),
        "senders" => open_find("senders"),
        "hierarchy" => open_class_browser(),
        "workspace" => open_workspace(),
        "editor" => open_editor(),
        "canvas" => open_canvas(),
        #[cfg(feature = "gamepane")]
        "gamePane" => open_game_pane(),
        #[cfg(not(feature = "gamepane"))]
        "gamePane" => report_no_game_pane(),
        "refresh" => reload_current_page(),
        "biggerText" => {
            bump_font_scale(FONT_SCALE_STEP as i32);
            reload_current_page();
        }
        "smallerText" => {
            bump_font_scale(-(FONT_SCALE_STEP as i32));
            reload_current_page();
        }
        other => append_transcript(&format!("(toolbar: {other} — not wired yet)")),
    }
}

// ── WKScriptMessageHandler: userContentController:didReceiveScriptMessage: ─

pub(crate) fn on_script_message(body: &ScriptMessage) {
    match body.kind().as_str() {
        "doit" => {
            let code = body.get("code");
            // SPEC §16.1: doits cross the GUI→VM channel to the worker
            // thread rather than running inline here. The worker (still a
            // stub — no real VM yet, ../PLAN.md G2) answers by echoing the
            // code back over the response channel; `vm_bridge_drain` picks
            // it up on this thread and appends it to the transcript. The
            // round trip is real even though the "VM" on the other end
            // isn't yet.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::Doit { code });
            }
        }
        "toolbar" => navigate_toolbar(&body.get("button")),
        "clearTranscript" => {
            // Empty the persisted history so the clear survives navigation (the
            // next page's `statusbar_html` renders from this buffer), then clear
            // the live pane to match — same native-drives-the-DOM shape as
            // `append_transcript`.
            TRANSCRIPT.lock().unwrap().clear();
            eval_js("window.macvmClearTranscript && window.macvmClearTranscript()");
        }
        "navigate" => {
            let href = body.get("href");
            let Some(nav) = NAV.get() else { return };
            let current_dir = {
                let n = nav.lock().unwrap();
                n.current()
                    .parent()
                    .unwrap_or_else(|| Path::new("/"))
                    .to_path_buf()
            };
            let target = normalize_path(&current_dir.join(&href));
            // Commit to history only if the page actually loads. A broken link
            // otherwise makes its non-existent path `current()`, so the next
            // relative click resolves against it and the path compounds
            // (documentation-tour `tour/tour/…`). On failure the displayed page
            // and `current()` are both unchanged.
            if navigate_to(&target) {
                nav.lock().unwrap().go(target);
            } else {
                eval_js(&format!(
                    "if(window.macvmSetStatus)window.macvmSetStatus({})",
                    js_string_literal(&format!("Not found: {href}"))
                ));
            }
        }
        // Class browser view (`browser_render.rs`, `../PLAN.md`'s "hierarchy"
        // toolbar action) — each of these just submits a request to the VM
        // worker thread; the resulting pane HTML comes back asynchronously
        // via `vm_bridge_drain`/`replace_pane`, same round trip as `doit`.
        "browserSelectPackage" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSelectPackage {
                    name: body.get("name"),
                });
            }
        }
        "browserSelectClass" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSelectClass {
                    name: body.get("name"),
                });
            }
        }
        "browserSelectSide" => {
            let side = if body.get("side") == "class" {
                vm_host::Side::Class
            } else {
                vm_host::Side::Instance
            };
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSelectSide { side });
            }
        }
        "browserSelectCategory" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSelectCategory {
                    name: body.get("name"),
                });
            }
        }
        "browserSelectMethod" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSelectMethod {
                    selector: body.get("name"),
                });
            }
        }
        "browserSaveSource" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserSaveSource {
                    text: body.get("text"),
                    saved_package: body.get("savedPackage"),
                    saved_class: body.get("savedClass"),
                    saved_side: body.get("savedSide"),
                    saved_category: body.get("savedCategory"),
                    saved_method: body.get("savedMethod"),
                    saved_target: body.get("savedTarget"),
                });
            }
        }
        "browserNewMethod" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserNewMethod);
            }
        }
        "browserNewClass" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserNewClass);
            }
        }
        "browserEditDefinition" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserEditDefinition);
            }
        }
        "browserEditComment" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserEditComment);
            }
        }
        "browserRemoveClass" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserRemoveClass);
            }
        }
        "browserRemoveMethod" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::BrowserRemoveMethod);
            }
        }
        "workspacePrintIt" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::WorkspacePrintIt {
                    code: body.get("code"),
                });
            }
        }
        "workspaceTextChanged" => {
            *WORKSPACE_TEXT.lock().unwrap() = Some(body.get("text"));
        }
        "canvasRunDemo" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::CanvasRunDemo);
            }
        }
        "canvasEval" => {
            // Generic: whatever Smalltalk the clicked control carried in its
            // `data-canvas-eval` attribute is evaluated and its command-batch
            // answer drawn. The GUI holds no per-drawing knowledge.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::CanvasEval {
                    code: body.get("code"),
                });
            }
        }
        "canvasEvalPixels" => {
            // Generic pixel path: the clicked control's `data-canvas-eval`
            // answers a width*height*4 RGBA buffer, blitted via putImageData.
            // width/height arrive as strings (only dict_get_string exists).
            if let Some(vm) = VM.get() {
                let width = body.get("width").parse().unwrap_or(0);
                let height = body.get("height").parse().unwrap_or(0);
                vm.submit(vm_host::VmRequest::CanvasEvalPixels {
                    code: body.get("code"),
                    width,
                    height,
                });
            }
        }
        "canvasClear" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::CanvasClear);
            }
        }
        "editAction" => {
            let action = body.get("action");
            match action.as_str() {
                // Copy: the selected text rode in with the message
                // (captured at menu-open in smtk.js) — write it to the
                // real pasteboard. An empty payload falls back to the
                // old responder-chain path (harmless, occasionally right).
                "copy" => {
                    let text = body.get("text");
                    if text.is_empty() {
                        send_edit_action("copy");
                    } else {
                        clipboard_write(&text);
                    }
                }
                // Paste: read the pasteboard and hand the string back to
                // the page; smtk.js inserts it at the focused cursor.
                "paste" => {
                    let text = clipboard_read();
                    if !text.is_empty() {
                        eval_js(&format!("macvmPasteText({})", js_string_literal(&text)));
                    }
                }
                other => send_edit_action(other),
            }
        }
        "smappl" => {
            // A `<smappl>` placeholder on the loaded page asks to be rendered
            // (smtk.js posts one per span on load). The worker evaluates the
            // `visual=` code and answers a fragment (D-G5, ../smappl.md).
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplRender {
                    id: body.get("id"),
                    code: body.get("code"),
                });
            }
        }
        "smapplAction" => {
            // A rendered smappl widget was clicked — fire its action closure.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplAction {
                    action_id: body.get("actionId"),
                });
            }
        }
        "smapplAccept" => {
            // Cmd+S in a ClassOutliner source editor — version the edit into
            // the image and live-compile it.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplAccept {
                    cls: body.get("cls"),
                    side: body.get("side"),
                    sel: body.get("sel"),
                    text: body.get("text"),
                    widget_id: body.get("widgetId"),
                });
            }
        }
        "smapplNewMethod" => {
            // Cmd+S in an outliner "＋ new method" template — create the method.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplNewMethod {
                    cls: body.get("cls"),
                    side: body.get("side"),
                    text: body.get("text"),
                });
            }
        }
        "smapplNewClass" => {
            // Cmd+S in an outliner "＋ new class" template — create the class.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplNewClass {
                    text: body.get("text"),
                });
            }
        }
        "smapplAddVar" => {
            // ⏎/Cmd+S in an outliner "＋ add instance/class variable" field.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplAddVar {
                    cls: body.get("cls"),
                    is_class_var: body.get("varKind") == "class",
                    name: body.get("name"),
                });
            }
        }
        "smapplOpenClass" => {
            // Drill from a hierarchy outliner into a class's method browser.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplOpenClass {
                    cls: body.get("cls"),
                    root: body.get("root"),
                    widget_id: body.get("widgetId"),
                });
            }
        }
        "smapplOpenHierarchy" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::SmapplOpenHierarchy {
                    root: body.get("root"),
                    widget_id: body.get("widgetId"),
                });
            }
        }
        "find" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::Find {
                    tool: body.get("tool"),
                    query: body.get("query"),
                });
            }
        }
        "findOptions" => {
            // A find page loaded — fetch its combobox options from image_store.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::FindOptions {
                    tool: body.get("tool"),
                });
            }
        }
        "editorOpen" => {
            // The editor's class picker fired (Enter on a class name). Rebuild
            // the page with that class's source. Pure main-thread image read —
            // no VM round trip, same as the find/inject-method-sources path.
            let name = body.get("class");
            display_editor(if name.is_empty() { None } else { Some(&name) });
        }
        "editorSession" => {
            // The freshly-rendered editor page seeded its VM edit session on the
            // loaded text (docs/editor_design.md M3). The full-document
            // EditorDamage reply flows back through vm_bridge_drain.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::EditorOpen {
                    text: body.get("text"),
                });
            }
        }
        "editorKey" => {
            // One editing key for the current session, carrying the textarea's
            // real caret offset (`at`, posted as a string) so the VM edits where
            // the user is. Answers an EditorDamage the JS terminal patches in.
            if let Some(vm) = VM.get() {
                let at = body.get("at").parse::<i64>().unwrap_or(1);
                vm.submit(vm_host::VmRequest::EditorKey {
                    key: body.get("key"),
                    at,
                });
            }
        }
        "editorCommand" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::EditorCommand {
                    name: body.get("name"),
                });
            }
        }
        "editorAccept" => {
            // Save: syntax-check + live-install + persist the diffed class. The
            // outcome comes back on the transcript.
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::EditorAccept);
            }
        }
        _ => {}
    }
}

/// Resolve `..`/`.` components without touching the filesystem (the target
/// may not exist yet if `href` is wrong — better to report a clean load
/// error than to silently fail path resolution).
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Reload whatever page `NAV` currently points at — used after a theme
/// switch, so the change is visible immediately rather than on the next
/// navigation.
fn reload_current_page() {
    if let Some(nav) = NAV.get() {
        let current = nav.lock().unwrap().current();
        navigate_to(&current);
    }
}

/// A menu item was chosen. Both shells name their items with these same
/// action strings (`shell/mac.rs`'s selectors, `shell/win.rs`'s command ids),
/// so menu behaviour is defined once here rather than twice.
///
/// Themes are `theme:<index into Theme::ALL>` — one action for every theme,
/// so adding a theme is purely a `Theme::ALL` entry with no second list to
/// keep in sync. That is exactly how the old per-theme `selectXTheme:`
/// handlers silently dropped newly-added themes from the menu.
pub(crate) fn on_menu_action(action: &str) {
    match action {
        // World I/O: submit to the worker (which owns the image + world_dir),
        // result reported to the transcript.
        "saveWorld" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::ExportWorld);
            }
        }
        "loadWorld" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::ImportWorld);
            }
        }
        // Kills the current language thread and boots a fresh VM. Runs on the
        // UI thread, which is independent of the (possibly wedged) worker, so
        // this stays clickable even when a runaway doit has hung the VM — the
        // whole reason they are on separate threads.
        "restartVm" => {
            if let Some(vm) = VM.get() {
                vm.restart();
            }
        }
        "help" => {
            let help = macvm_help_index();
            if let Some(nav) = NAV.get() {
                nav.lock().unwrap().go(help.clone());
            }
            navigate_to(&help);
        }
        "cut" | "copy" | "paste" | "selectAll" => send_edit_action(action),
        #[cfg(feature = "gamepane")]
        "demoBreakout" => run_game_demo("Breakout launch."),
        #[cfg(feature = "gamepane")]
        "demoMandel" => run_game_demo("MandelZoom launch."),
        #[cfg(feature = "gamepane")]
        "demoParallelMandel" => run_game_demo("ParallelMandel launch."),
        #[cfg(feature = "gamepane")]
        "demoSpawnedMandel" => run_spawned_mandel_demo(),
        #[cfg(feature = "gamepane")]
        "demoCocoaPad" => {
            if let Some(vm) = VM.get() {
                vm.submit(vm_host::VmRequest::Doit {
                    code: "CocoaPad launch.".to_string(),
                });
            }
        }
        other => {
            if let Some(index) = other.strip_prefix("theme:") {
                if let Some(&theme) = index.parse().ok().and_then(|i: usize| Theme::ALL.get(i)) {
                    set_theme(theme);
                    update_theme_menu_checkmarks();
                    reload_current_page();
                }
            }
        }
    }
}

/// Reflect the active theme as a checkmark on the Theme menu.
fn update_theme_menu_checkmarks() {
    let current = current_theme();
    if let Some(index) = Theme::ALL.iter().position(|&t| t == current) {
        shell::set_theme_checkmark(index);
    }
}

/// Open the native game pane and launch a Smalltalk demo in the GUI's own VM
/// (`docs/gamepane_design.md`). Same path as the `MACVM_GAMEPANE_DEMO` launch
/// trigger.
#[cfg(feature = "gamepane")]
fn run_game_demo(code: &str) {
    open_game_pane();
    if let Some(vm) = VM.get() {
        vm.submit(vm_host::VmRequest::Doit {
            code: code.to_string(),
        });
    }
}

/// The "Mandelbrot — spawned VM" demo: the real multi-VM case. Spins up a
/// genuinely SECOND VM instance (its own worker thread + fresh `VmHandle`,
/// booted from the image — independent of the GUI's own `VM`), runs the
/// single-dive `MandelVM` in it in the game pane, and tears that instance back
/// down when the dive ends (`on_game_loop_stopped` -> `teardown_spawned_demo`).
/// The GUI's own VM never pauses — it keeps serving the Workspace/browser.
#[cfg(feature = "gamepane")]
fn run_spawned_mandel_demo() {
    if DEMO_VM_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        return; // one spawned demo at a time
    }
    // Its own OS thread, its own VmHandle, and its own waker so its responses
    // drain separately from the main VM's.
    let demo = vm_host::spawn(shell::demo_waker());
    DEMO_VM.with(|d| *d.borrow_mut() = Some(demo));
    DEMO_VM_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
    append_transcript("--- spawned a fresh VM instance to run MandelVM (one dive, then it exits) ---");
    open_game_pane();
    DEMO_VM.with(|d| {
        if let Some(vm) = d.borrow().as_ref() {
            vm.submit(vm_host::VmRequest::Doit {
                code: "MandelVM launch.".to_string(),
            });
        }
    });
}

/// Drain the SPAWNED demo VM's responses (a separate channel from the main
/// `VM`'s) and apply its game-draw commands to the shared pane. Collect the
/// batch first and release the `DEMO_VM` borrow BEFORE applying: a `StopLoop`
/// in the batch runs `teardown_spawned_demo`, which drops `DEMO_VM` and so
/// needs a fresh borrow. `MandelVM` emits only game (and, on error,
/// transcript) responses.
#[cfg(feature = "gamepane")]
pub(crate) fn on_demo_vm_drain() {
    let responses = DEMO_VM.with(|d| {
        d.borrow()
            .as_ref()
            .map(|vm| vm.drain_responses())
            .unwrap_or_default()
    });
    for response in responses {
        match response {
            vm_host::VmResponse::Game(cmd) => game_pane::apply_command(&cmd),
            vm_host::VmResponse::Transcript(text) => append_transcript(&text),
            _ => {} // the mandel demo produces nothing else
        }
    }
    game_pane::present_if_dirty();
}

/// End the spawned-VM demo: swap the browser back in (the main VM's content),
/// drop the native pane, then drop the spawned `DEMO_VM` — closing its request
/// channel lets its worker thread exit. The GUI's own `VM` is untouched
/// throughout, so the Workspace/browser is live again immediately. Order
/// matters: restore the webview BEFORE freeing the pane's GPU resources (as
/// `close_game_pane`), and take the `DEMO_VM` borrow only after the drain that
/// triggered us has released it.
#[cfg(feature = "gamepane")]
fn teardown_spawned_demo() {
    DEMO_VM_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
    let prev = NAV
        .get()
        .and_then(|n| n.lock().unwrap().back())
        .unwrap_or_else(start_page);
    navigate_to(&prev);
    shell::focus_webview();
    game_pane::teardown();
    DEMO_VM.with(|d| *d.borrow_mut() = None);
    append_transcript("--- MandelVM finished; the spawned VM instance was torn down (the GUI VM never paused) ---");
}

// ── VM bridge: the main-thread side of vm_host's worker thread ────────────

/// Runs on the UI thread, woken by the VM worker thread (`vm_host::spawn`)
/// whenever a response is ready. Drains every response queued since the last
/// call (not just one) — wakeups can coalesce in principle, and this is cheap
/// and correct either way.
pub(crate) fn on_vm_drain() {
    let Some(vm) = VM.get() else { return };
    for response in vm.drain_responses() {
        match response {
            vm_host::VmResponse::Transcript(text) => append_transcript(&text),
            #[cfg(feature = "gamepane")]
            vm_host::VmResponse::Game(cmd) => game_pane::apply_command(&cmd),
            // The native game pane is macOS-only (`../MIGRATION.md` §1 defers
            // the game demos); a guest that draws anyway is told so once, in
            // the transcript, rather than silently drawing nothing.
            #[cfg(not(feature = "gamepane"))]
            vm_host::VmResponse::Game(_) => {}
            vm_host::VmResponse::EditorView { cursor, text } => {
                // Blast the whole buffer + caret into the JS terminal, which
                // just sets textarea.value and the selection — the VM is the
                // source of truth, so there is nothing to reconcile
                // (docs/editor_design.md M3).
                eval_js(&format!(
                    "if(window.macvmEditorView)window.macvmEditorView({cursor},{})",
                    js_string_literal(&text)
                ));
            }
            vm_host::VmResponse::BrowserPanes {
                packages_html,
                classes_html,
                categories_html,
                methods_html,
                source_html,
            } => {
                replace_pane("macvm-browser-packages", &packages_html);
                replace_pane("macvm-browser-classes", &classes_html);
                replace_pane("macvm-browser-categories", &categories_html);
                replace_pane("macvm-browser-methods", &methods_html);
                replace_pane("macvm-browser-source", &source_html);
            }
            vm_host::VmResponse::BrowserOpened {
                packages_html,
                classes_html,
                categories_html,
                methods_html,
                source_html,
            } => {
                // The only place a fresh page gets built from real data
                // rather than patched into an existing one — see
                // `browser_render::assemble_panes`'s doc comment for why.
                let body = browser_render::assemble_panes(
                    &packages_html,
                    &classes_html,
                    &categories_html,
                    &methods_html,
                    &source_html,
                );
                let html = preprocess::render_generated_page(
                    "Class Browser",
                    &body,
                    &gui_root(),
                    current_theme(),
                    current_font_scale_percent(),
                    &current_transcript(),
                );
                display_html(&html);
            }
            vm_host::VmResponse::WorkspacePrintResult { text } => {
                eval_js(&format!(
                    "window.macvmInsertPrintResult({})",
                    js_string_literal(&text)
                ));
            }
            vm_host::VmResponse::CanvasCreated { id, width, height } => {
                // The response is the authority on size, not the page's
                // initial static guess (`canvas_render::render_canvas`) —
                // keeps the `<canvas>` pixel buffer in sync if a future
                // `Canvas extent:` ever requests a size other than the
                // default this view opens with.
                eval_js(&format!(
                    "window.macvmCanvasCreated({id}, {width}, {height})"
                ));
            }
            vm_host::VmResponse::CanvasDraw { id, commands_json } => {
                eval_js(&format!(
                    "window.macvmCanvasDraw({id}, {})",
                    js_string_literal(&commands_json)
                ));
            }
            vm_host::VmResponse::CanvasPixels {
                id,
                width,
                height,
                base64,
            } => {
                // Blit the RGBA buffer in one putImageData (docs/CANVAS.md pixel
                // path). smtk.js decodes the base64 into an ImageData.
                eval_js(&format!(
                    "window.macvmCanvasPutPixels({id}, {width}, {height}, {})",
                    js_string_literal(&base64)
                ));
            }
            vm_host::VmResponse::CanvasCleared { .. } => {
                // No DOM action of its own — the paired `CanvasDraw`
                // response (a `clearRect` batch, see `vm_host::handle`'s
                // `CanvasClear` arm) already does the actual pixel clear.
                // This variant exists for a future GUI-side content
                // cache/replay-on-reopen (`../docs/CANVAS.md` §7) to
                // invalidate against, not for anything the DOM needs today.
            }
            vm_host::VmResponse::FindResults { html } => {
                // Fill the find page's results container (keeping its
                // data-widget-id so a result click can drill via smapplOpenClass).
                eval_js(&format!(
                    "window.macvmSetFindResults({})",
                    js_string_literal(&html)
                ));
            }
            vm_host::VmResponse::FindOptions { tool: _, options } => {
                // Populate the find page's <datalist> combobox from image_store.
                let items: Vec<String> = options.iter().map(|o| js_string_literal(o)).collect();
                eval_js(&format!(
                    "window.macvmSetFindOptions([{}])",
                    items.join(",")
                ));
            }
            vm_host::VmResponse::SmapplFragment { id, html } => {
                // Swap the placeholder span with `data-widget-id == id` for
                // the rendered widget (D-G5, ../smappl.md). smtk.js locates it
                // by the widget id rather than a DOM `id`, so this can't
                // collide with the page's own element ids. Icon buttons carry
                // a `data-icon` logical name resolved to the active theme's
                // asset here (theme is a main-thread concept; the worker that
                // built the fragment doesn't know it — ../smappl.md §5).
                let html = preprocess::rewrite_smappl_icons(&html, current_theme());
                eval_js(&format!(
                    "window.macvmRenderSmappl({}, {})",
                    js_string_literal(&id),
                    js_string_literal(&html)
                ));
            }
            vm_host::VmResponse::SmapplOverlay { html } => {
                // A live widget action popped a modal dialog (Visual>>promptOk:,
                // the "Press Me!" demo). Float it over the current page; smtk.js
                // owns the overlay lifecycle (backdrop click / OK / Esc close).
                eval_js(&format!(
                    "if(window.macvmShowOverlay)window.macvmShowOverlay({})",
                    js_string_literal(&html)
                ));
            }
            vm_host::VmResponse::SmapplNavigate { target } => {
                // A tour page's toolbar button (Launcher/Workspace/…) — open the
                // MACVM view its action maps to, via the same dispatch a native
                // toolbar click uses.
                navigate_toolbar(&target);
            }
            vm_host::VmResponse::Metrics(m) => metrics_on_snapshot(m),
            vm_host::VmResponse::WorkerIdle | vm_host::VmResponse::LiveStats(_) => {
                // Never actually delivered here — `VmHost::drain_responses`
                // consumes the worker-liveness marker and the boot-time
                // live-stats handoff internally and never returns them (see
                // those variants' docs). These arms exist only to keep the
                // match exhaustive.
            }
        }
    }
    // A whole batch of game draw commands (a frame) presents exactly once,
    // here, after they've all been applied to the pane (game_pane.rs).
    #[cfg(feature = "gamepane")]
    game_pane::present_if_dirty();
}

/// Replace a browser pane's entire element (its `outerHTML`, not just its
/// contents) with freshly-rendered markup carrying the same `dom_id` —
/// `browser_render.rs`'s pane functions always return a complete,
/// self-contained `<div id="...">`, so this works uniformly for every pane
/// on every selection change without the renderer needing two different
/// shapes (wrapped vs. bare) for initial render vs. later patching.
fn replace_pane(dom_id: &str, html: &str) {
    // The trailing `macvmHighlightCodeEditors()` call re-syncs the source
    // pane's highlighted `<pre>` after a fresh `<textarea>` value lands in
    // it (see `browser_render::render_source_pane`) — harmless/no-op for
    // every other pane, which has no `.st-code-editor` to find.
    eval_js(&format!(
        "{{ const el = document.getElementById({}); if (el) el.outerHTML = {}; \
         if (window.macvmHighlightCodeEditors) window.macvmHighlightCodeEditors(); }}",
        js_string_literal(dom_id),
        js_string_literal(html),
    ));
}

fn main() {
    // Headless "eyes" commands — render a page (with all smappls resolved by
    // a real VM) to self-contained HTML, seed/export the image, or run a
    // `.mst` file. Guarded before `shell::init()` so they run anywhere, not
    // just in a windowing session.
    let cli: Vec<String> = std::env::args().skip(1).collect();
    match cli.first().map(String::as_str) {
        Some("render") => return cmd_render(&cli[1..]),
        Some("seed") => return cmd_seed(&cli[1..]),
        Some("export") => return cmd_export(&cli[1..]),
        Some("run") => return cmd_run_gui(&cli[1..]),
        _ => {}
    }

    // `winvm-gui mandelvm` — the standalone MandelVM demo window. It flows
    // through the normal windowed setup below, then opens straight into the
    // game pane running MandelVM instead of the browser, and quits itself
    // when the dive ends (see `on_game_loop_stopped`).
    #[cfg(feature = "gamepane")]
    let mandelvm_mode = cli.first().map(|s| s == "mandelvm").unwrap_or(false);

    shell::init();
    // Cocoa bridge C3: this process is a GUI host — the run loop will drain
    // the main queue, so Smalltalk's sync main-thread hop (ObjcRef>>
    // sendMain:args: / onMain, prim 242) is live. Headless hosts never call
    // this; there the hop fails cleanly instead of hanging.
    //
    // The bridge is macOS-only *in the VM crate itself* (`runtime/mod.rs`
    // gates the whole module: objc_msgSend, the objc_shim.m @try/@catch shim,
    // AppKit delegates). There is nothing to enable on Windows, and its Win32
    // counterpart arrives with the native shell (`../MIGRATION.md` §6).
    #[cfg(target_os = "macos")]
    macvm::runtime::objc_bridge::enable_main_hop();

    let start = start_page();
    if !start.exists() {
        eprintln!(
            "winvm-gui: {} not found — set MACVM_GUI_ROOT to the gui/ directory.",
            start.display()
        );
        std::process::exit(1);
    }
    NAV.set(Mutex::new(NavState::new(start.clone()))).ok();

    VM.set(vm_host::spawn(shell::waker()))
        .ok()
        .expect("VM.set called twice");

    // The window must exist before the menu bar (Windows attaches the menu to
    // an HWND) and before the first navigation (there must be a web view to
    // load into). The macOS shell is order-insensitive here, so this ordering
    // is correct for both.
    shell::create_window_and_webview();
    shell::build_menu_bar();
    update_theme_menu_checkmarks();
    navigate_to(&start);

    // Demo trigger (docs/gamepane_design.md): open the native game pane and
    // start a real Smalltalk-driven frame loop, exercising the whole M4 path
    // (timer -> GameStep -> step block -> draw -> present).
    #[cfg(feature = "gamepane")]
    if mandelvm_mode {
        // Standalone MandelVM demo: skip the browser, open the game pane, and
        // run the single-dive MandelVM. `MANDELVM` makes the frame loop's stop
        // (dive finished, or Escape) quit the app instead of returning to a page.
        MANDELVM.store(true, std::sync::atomic::Ordering::Relaxed);
        run_game_demo("MandelVM launch.");
    } else if let Some(demo) = std::env::var_os("MACVM_GAMEPANE_DEMO") {
        run_game_demo(if demo.to_string_lossy().eq_ignore_ascii_case("mandel") {
            "MandelZoom launch."
        } else {
            "Breakout launch."
        });
    }

    // Start the always-on metrics sampler that feeds the toolbar dashboard.
    shell::start_metrics_timer();
    shell::run();
}
