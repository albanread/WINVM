//! The macOS shell: `NSWindow` + `WKWebView` over the dependency-free
//! Objective-C bridge in `crate::objc`. The Windows twin is `win.rs`; the
//! seam they both implement is documented in `mod.rs`.
//!
//! This is the original MACVM shell, relocated behind that seam rather than
//! rewritten — the reasoning recorded in each function's comments was learned
//! by running the app, so it is kept verbatim where it still applies.

use std::path::Path;
use std::sync::OnceLock;

use crate::objc::{self, sel, Id, Sel, NIL};
use crate::preprocess::Theme;
use crate::shell::ScriptMessage;

/// `Id`/`Sel` are raw pointers (not `Send`/`Sync`) — sound to store in a
/// `static` here because AppKit only ever calls into this process on the
/// main thread, and nothing in this crate spawns another one.
struct MainThreadPtr(Id);
unsafe impl Send for MainThreadPtr {}
unsafe impl Sync for MainThreadPtr {}

static WEBVIEW: OnceLock<MainThreadPtr> = OnceLock::new();
/// The `NSWindow`, stored so the game pane can swap its content view (once the
/// game view is installed, the WKWebView's own `.window` goes nil, so it can't
/// be re-derived from the webview — it must be held separately).
static WINDOW: OnceLock<MainThreadPtr> = OnceLock::new();
/// One `(Theme, menu-item-pointer)` pair per `Theme::ALL` entry, so their
/// checkmarks can be updated when the active theme changes.
static THEME_MENU_ITEMS: OnceLock<Vec<(Theme, MainThreadPtr)>> = OnceLock::new();
static METRICS_TIMER_TARGET: OnceLock<MainThreadPtr> = OnceLock::new();

// ── Worker -> UI wakeup ────────────────────────────────────────────────────

/// A thread-safe handle the VM worker uses to wake the main thread
/// (`vm_host.rs`).
///
/// `Id`/`Sel` are raw pointers, not `Send` by default. Safe to send into the
/// worker thread here specifically because the only use they are put to is as
/// arguments to `performSelectorOnMainThread:withObject:waitUntilDone:`,
/// which is itself documented as safe to call from any thread — unlike
/// arbitrary AppKit/WebKit message sends, which are main-thread-only.
#[derive(Clone, Copy)]
pub struct Waker {
    target: Id,
    selector: Sel,
}

unsafe impl Send for Waker {}

impl Waker {
    /// A waker with no target — headless tests drive `VmHost::drain_responses`
    /// directly, so they need no Objective-C runtime at all (rather than
    /// relying on `objc_msgSend`-to-nil being a no-op).
    pub fn none() -> Self {
        Self {
            target: NIL,
            selector: NIL,
        }
    }

    pub fn notify(self) {
        if self.target.is_null() {
            return;
        }
        objc::perform_selector_on_main_thread(self.target, self.selector, NIL, false);
    }
}

/// The waker for this process's VM bridge object.
pub fn waker() -> Waker {
    Waker {
        target: vm_bridge(),
        selector: sel("drainVmResponses:"),
    }
}

fn vm_bridge() -> Id {
    static BRIDGE: OnceLock<MainThreadPtr> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmVmBridge");
            objc::add_method(
                cls,
                sel("drainVmResponses:"),
                vm_bridge_drain as *const _,
                "v@:@",
            );
            objc::register_class(cls);
            MainThreadPtr(objc::alloc_init("WinvmVmBridge"))
        })
        .0
}

extern "C" fn vm_bridge_drain(_this: Id, _cmd: Sel, _arg: Id) {
    crate::on_vm_drain();
}

// ── Startup ────────────────────────────────────────────────────────────────

pub fn init() {
    objc::bootstrap();
}

/// URL for an asset under the GUI resource root, resolved at RUNTIME via
/// [`crate::gui_root`] (which honours `MACVM_GUI_ROOT`), not the build-time
/// `CARGO_MANIFEST_DIR`: a baked build-time path names the dev checkout
/// (absent on another machine) AND falls outside the webview's
/// `loadFileURL:allowingReadAccessToURL:` grant, so the stylesheet/script
/// silently load as nothing.
pub fn asset_url(relative: &str) -> String {
    let full = format!("{}/{relative}", crate::gui_root().to_string_lossy());
    format!("file://{}", crate::preprocess::percent_encode_path(&full))
}

/// Base URL for a page's own directory (with the trailing slash `<base>`
/// needs to be treated as a directory, not a file).
pub fn base_url(dir: &Path) -> String {
    format!(
        "file://{}/",
        crate::preprocess::percent_encode_path(&dir.to_string_lossy())
    )
}

// ── Window + WKWebView ─────────────────────────────────────────────────────

const STYLE_TITLED_CLOSABLE_MINIATURIZABLE_RESIZABLE: u64 = 15;
const BACKING_BUFFERED: u64 = 2;
const AUTORESIZING_WIDTH_HEIGHT_SIZABLE: u64 = 18; // NSViewWidthSizable(2) | NSViewHeightSizable(16)

pub fn create_window_and_webview() {
    let window = objc::send0(objc::get_class("NSWindow"), sel("alloc"));
    let window = objc::send_window_init(
        window,
        sel("initWithContentRect:styleMask:backing:defer:"),
        0.0,
        0.0,
        980.0,
        760.0,
        STYLE_TITLED_CLOSABLE_MINIATURIZABLE_RESIZABLE,
        BACKING_BUFFERED,
        false,
    );
    objc::send1_id(window, sel("setTitle:"), objc::nsstring("WINVM"));
    objc::send0(window, sel("center"));

    let config = objc::alloc_init("WKWebViewConfiguration");
    let user_content_controller = objc::send0(config, sel("userContentController"));
    objc::send2_id(
        user_content_controller,
        sel("addScriptMessageHandler:name:"),
        build_message_handler(),
        objc::nsstring("macvm"),
    );

    let webview = objc::send0(objc::get_class("WKWebView"), sel("alloc"));
    let webview = objc::send_frame_config_init(
        webview,
        sel("initWithFrame:configuration:"),
        0.0,
        0.0,
        980.0,
        760.0,
        config,
    );
    objc::send1_i64(
        webview,
        sel("setAutoresizingMask:"),
        AUTORESIZING_WIDTH_HEIGHT_SIZABLE as i64,
    );
    objc::send1_id(window, sel("setContentView:"), webview);

    WEBVIEW
        .set(MainThreadPtr(webview))
        .ok()
        .expect("create_window_and_webview called twice");
    WINDOW.set(MainThreadPtr(window)).ok();

    objc::send1_id(window, sel("makeKeyAndOrderFront:"), NIL);
}

fn build_message_handler() -> Id {
    let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmScriptMessageHandler");
    objc::add_method(
        cls,
        sel("userContentController:didReceiveScriptMessage:"),
        on_script_message as *const _,
        "v@:@@",
    );
    objc::register_class(cls);
    objc::alloc_init("WinvmScriptMessageHandler")
}

/// Decode the `NSDictionary` body into the platform-neutral [`ScriptMessage`]
/// and hand it to the shared dispatch in `main.rs`.
///
/// The key list is read off the dictionary itself rather than a fixed
/// allow-list, so adding a field to a `smtk.js` message needs no change here.
extern "C" fn on_script_message(_this: Id, _cmd: Sel, _controller: Id, message: Id) {
    let body = objc::send0(message, sel("body"));
    let mut fields = std::collections::HashMap::new();
    if !body.is_null() {
        let keys = objc::send0(body, sel("allKeys"));
        let count = objc::send0_i64(keys, sel("count"));
        for i in 0..count {
            let key = objc::send1_i64(keys, sel("objectAtIndex:"), i);
            let name = objc::string_from_nsstring(key);
            let value = objc::send1_id(body, sel("objectForKey:"), key);
            fields.insert(name, objc::string_from_nsstring(value));
        }
    }
    crate::on_script_message(&ScriptMessage::from_fields(fields));
}

// ── Page loading and scripting ─────────────────────────────────────────────

/// Load `path` (the freshly-written `.rendered/current.html`).
///
/// `loadFileURL:allowingReadAccessToURL:<gui root>` rather than
/// `loadHTMLString:baseURL:` — WKWebView does not grant local `file://`
/// subresource access outside `baseURL`'s own directory for the latter
/// (confirmed by actually running the shell: `strongtalk.css`, `smtk.js`, and
/// every toolbar icon silently failed to load, easy to mistake for "it
/// worked" at a glance). Granting read access to the whole `gui/` root covers
/// the rendered file, the original page's own directory, and `assets/` in one
/// grant.
pub fn load_rendered_page(path: &Path) {
    // If the native game pane swapped itself in as the content view, restore
    // the WKWebView before loading HTML — a no-op on every other navigation.
    if let (Some(window), Some(webview)) = (WINDOW.get(), WEBVIEW.get()) {
        let current = objc::send0(window.0, sel("contentView"));
        if current != webview.0 {
            objc::send1_id(window.0, sel("setContentView:"), webview.0);
        }
    }

    let Some(webview) = WEBVIEW.get() else { return };
    let target_url = make_file_url(path);
    let access_root_url = make_file_url(&crate::gui_root());
    objc::send2_id(
        webview.0,
        sel("loadFileURL:allowingReadAccessToURL:"),
        target_url,
        access_root_url,
    );
}

fn make_file_url(dir: &Path) -> Id {
    let cls = objc::get_class("NSURL");
    let path_str = objc::nsstring(&dir.to_string_lossy());
    objc::send1_id(cls, sel("fileURLWithPath:"), path_str)
}

/// `[webview evaluateJavaScript:completionHandler:]` with a nil completion
/// handler — Cocoa accepts nil there when the caller doesn't need the result
/// (this shell never does), which sidesteps building an Objective-C block
/// literal entirely (see `objc.rs`'s module doc: this bridge deliberately
/// doesn't implement the block ABI).
pub fn eval_js(js: &str) {
    let Some(webview) = WEBVIEW.get() else { return };
    objc::send2_id(
        webview.0,
        sel("evaluateJavaScript:completionHandler:"),
        objc::nsstring(js),
        NIL,
    );
}

// ── Clipboard ──────────────────────────────────────────────────────────────

/// `"public.utf8-plain-text"` is `NSPasteboardTypeString`'s underlying UTI —
/// used literally so the bridge needs no AppKit constant linkage.
const UTF8_PLAIN_TEXT: &str = "public.utf8-plain-text";

pub fn clipboard_write(text: &str) {
    let pb = objc::send0(objc::get_class("NSPasteboard"), sel("generalPasteboard"));
    objc::send0(pb, sel("clearContents"));
    objc::send2_id(
        pb,
        sel("setString:forType:"),
        objc::nsstring(text),
        objc::nsstring(UTF8_PLAIN_TEXT),
    );
}

pub fn clipboard_read() -> String {
    let pb = objc::send0(objc::get_class("NSPasteboard"), sel("generalPasteboard"));
    let ns = objc::send1_id(pb, sel("stringForType:"), objc::nsstring(UTF8_PLAIN_TEXT));
    if ns.is_null() {
        String::new()
    } else {
        objc::string_from_nsstring(ns)
    }
}

/// Cut/Select All from `smtk.js`'s custom context menu, routed through
/// `NSApp sendAction:to:from:` with a nil target so AppKit's standard
/// responder-chain dispatch finds whatever is actually focused — the exact
/// mechanism a native Edit-menu item uses when clicked.
pub fn edit_action(action: &str) {
    let sel_name = match action {
        "cut" => "cut:",
        "copy" => "copy:",
        "paste" => "paste:",
        "selectAll" => "selectAll:",
        _ => return,
    };
    let app = objc::send0(objc::get_class("NSApplication"), sel("sharedApplication"));
    objc::send3_id(app, sel("sendAction:to:from:"), sel(sel_name), NIL, NIL);
}

// ── Menu bar ───────────────────────────────────────────────────────────────

const NS_CONTROL_STATE_VALUE_ON: i64 = 1;
const NS_CONTROL_STATE_VALUE_OFF: i64 = 0;

/// `NSApplicationActivationPolicyRegular` — a normal foreground app (Dock
/// icon, menu bar, window activation). Without this, a bare executable (no
/// `.app` bundle / `Info.plist`) defaults to behaving like a background
/// agent: no window ever becomes visible or key.
const NS_APPLICATION_ACTIVATION_POLICY_REGULAR: i64 = 0;

fn menu_item(title: &str, action: Option<&str>, key_equivalent: &str) -> Id {
    let item = objc::send0(objc::get_class("NSMenuItem"), sel("alloc"));
    objc::send3_id(
        item,
        sel("initWithTitle:action:keyEquivalent:"),
        objc::nsstring(title),
        action.map(objc::sel).unwrap_or(NIL),
        objc::nsstring(key_equivalent),
    )
}

fn separator_item() -> Id {
    objc::send0(objc::get_class("NSMenuItem"), sel("separatorItem"))
}

fn submenu(title: &str, items: &[Id]) -> Id {
    let menu_item = objc::send0(objc::get_class("NSMenuItem"), sel("alloc"));
    let menu_item = objc::send3_id(
        menu_item,
        sel("initWithTitle:action:keyEquivalent:"),
        objc::nsstring(title),
        NIL,
        objc::nsstring(""),
    );
    let menu = objc::alloc_init("NSMenu");
    for &item in items {
        objc::send1_id(menu, sel("addItem:"), item);
    }
    objc::send1_id(menu_item, sel("setSubmenu:"), menu);
    menu_item
}

/// Every menu action funnels through one delegate object and one selector per
/// action name, each forwarding to `main.rs::on_menu_action` with the same
/// action string the Windows shell uses — so menu dispatch is shared.
fn build_action_delegate() -> Id {
    let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmMenuDelegate");
    objc::add_method(cls, sel("saveWorld:"), act_save_world as *const _, "v@:@");
    objc::add_method(cls, sel("loadWorld:"), act_load_world as *const _, "v@:@");
    objc::add_method(cls, sel("restartVm:"), act_restart_vm as *const _, "v@:@");
    objc::add_method(cls, sel("openHelp:"), act_help as *const _, "v@:@");
    objc::add_method(cls, sel("selectTheme:"), act_theme as *const _, "v@:@");
    objc::register_class(cls);
    objc::alloc_init("WinvmMenuDelegate")
}

extern "C" fn act_save_world(_t: Id, _c: Sel, _s: Id) {
    crate::on_menu_action("saveWorld");
}
extern "C" fn act_load_world(_t: Id, _c: Sel, _s: Id) {
    crate::on_menu_action("loadWorld");
}
extern "C" fn act_restart_vm(_t: Id, _c: Sel, _s: Id) {
    crate::on_menu_action("restartVm");
}
extern "C" fn act_help(_t: Id, _c: Sel, _s: Id) {
    crate::on_menu_action("help");
}

/// The clicked item's `tag` is its index into `Theme::ALL`, so ONE handler
/// serves every theme and adding a theme is purely a `Theme::ALL` entry — no
/// new selector, no second list to keep in sync (which is exactly how the old
/// per-theme handlers silently dropped newly-added themes from the menu).
extern "C" fn act_theme(_t: Id, _c: Sel, sender: Id) {
    let idx = objc::send0_i64(sender, sel("tag")) as usize;
    crate::on_menu_action(&format!("theme:{idx}"));
}

extern "C" fn app_should_terminate_after_last_window_closed(
    _this: Id,
    _cmd: Sel,
    _sender: Id,
) -> bool {
    true
}

fn build_quit_on_last_window_delegate() -> Id {
    let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmAppDelegate");
    objc::add_method(
        cls,
        sel("applicationShouldTerminateAfterLastWindowClosed:"),
        app_should_terminate_after_last_window_closed as *const _,
        "B@:@",
    );
    objc::register_class(cls);
    objc::alloc_init("WinvmAppDelegate")
}

pub fn build_menu_bar() {
    let app = objc::send0(objc::get_class("NSApplication"), sel("sharedApplication"));
    objc::send1_i64(
        app,
        sel("setActivationPolicy:"),
        NS_APPLICATION_ACTIVATION_POLICY_REGULAR,
    );

    let delegate = build_action_delegate();
    let targeted = |title: &str, action: &str, key: &str| {
        let item = menu_item(title, Some(action), key);
        objc::send1_id(item, sel("setTarget:"), delegate);
        item
    };

    let app_menu = submenu("WINVM", &[menu_item("Quit WINVM", Some("terminate:"), "q")]);
    let file_menu = submenu(
        "File",
        &[
            menu_item("Close Window", Some("performClose:"), "w"),
            separator_item(),
            targeted("Save World to Files", "saveWorld:", ""),
            targeted("Load World from Files", "loadWorld:", ""),
        ],
    );
    let edit_menu = submenu(
        "Edit",
        &[
            menu_item("Cut", Some("cut:"), "x"),
            menu_item("Copy", Some("copy:"), "c"),
            menu_item("Paste", Some("paste:"), "v"),
            separator_item(),
            menu_item("Select All", Some("selectAll:"), "a"),
        ],
    );
    let vm_menu = submenu("VM", &[targeted("Restart VM Thread", "restartVm:", "r")]);

    // Walk Theme::ALL (the single source of truth) — one item per theme, its
    // tag its index into ALL, every item firing the one `selectTheme:` action.
    let mut checkmark_entries = Vec::with_capacity(Theme::ALL.len());
    let mut theme_items = Vec::with_capacity(Theme::ALL.len());
    for (idx, &theme) in Theme::ALL.iter().enumerate() {
        let item = targeted(theme.menu_label(), "selectTheme:", "");
        objc::send1_i64(item, sel("setTag:"), idx as i64);
        checkmark_entries.push((theme, MainThreadPtr(item)));
        theme_items.push(item);
    }
    THEME_MENU_ITEMS.set(checkmark_entries).ok();
    let theme_menu = submenu("Theme", &theme_items);

    let help_menu = submenu("Help", &[targeted("WINVM GUI Shell Help", "openHelp:", "")]);

    let main_menu = objc::alloc_init("NSMenu");
    for &item in &[
        app_menu, file_menu, edit_menu, vm_menu, theme_menu, help_menu,
    ] {
        objc::send1_id(main_menu, sel("addItem:"), item);
    }
    objc::send1_id(app, sel("setMainMenu:"), main_menu);
}

pub fn set_theme_checkmark(active_index: usize) {
    let Some(items) = THEME_MENU_ITEMS.get() else {
        return;
    };
    for (idx, (_theme, item)) in items.iter().enumerate() {
        objc::send1_i64(
            item.0,
            sel("setState:"),
            if idx == active_index {
                NS_CONTROL_STATE_VALUE_ON
            } else {
                NS_CONTROL_STATE_VALUE_OFF
            },
        );
    }
}

/// Make the web view first responder again (after the game pane had the
/// window's content view).
pub fn focus_webview() {
    if let (Some(window), Some(webview)) = (WINDOW.get(), WEBVIEW.get()) {
        objc::send1_id(window.0, sel("makeFirstResponder:"), webview.0);
    }
}

/// The window, for the game pane to swap its content view into
/// (`game_pane.rs`; macOS-only, `gamepane` feature).
#[cfg(feature = "gamepane")]
pub fn window() -> Id {
    WINDOW.get().map(|w| w.0).unwrap_or(NIL)
}

/// A second waker, for the spawned demo VM's own response channel — it must
/// drain separately from the main VM's (`main.rs::on_demo_vm_drain`).
#[cfg(feature = "gamepane")]
pub fn demo_waker() -> Waker {
    static BRIDGE: OnceLock<MainThreadPtr> = OnceLock::new();
    let target = BRIDGE
        .get_or_init(|| {
            let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmDemoVmBridge");
            objc::add_method(
                cls,
                sel("drainDemoVm:"),
                demo_bridge_drain as *const _,
                "v@:@",
            );
            objc::register_class(cls);
            MainThreadPtr(objc::alloc_init("WinvmDemoVmBridge"))
        })
        .0;
    Waker {
        target,
        selector: sel("drainDemoVm:"),
    }
}

#[cfg(feature = "gamepane")]
extern "C" fn demo_bridge_drain(_this: Id, _cmd: Sel, _arg: Id) {
    crate::on_demo_vm_drain();
}

/// The frame timer for the native game pane's ~60 Hz loop.
#[cfg(feature = "gamepane")]
pub fn start_game_loop_timer() {
    static TARGET: OnceLock<MainThreadPtr> = OnceLock::new();
    let target = TARGET
        .get_or_init(|| {
            let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmGameTimer");
            objc::add_method(cls, sel("gameTick:"), on_game_tick as *const _, "v@:@");
            objc::register_class(cls);
            MainThreadPtr(objc::alloc_init("WinvmGameTimer"))
        })
        .0;
    GAME_TIMER.with(|t| {
        let existing = t.get();
        if existing != NIL {
            objc::send0(existing, sel("invalidate"));
        }
        t.set(objc::scheduled_timer(
            1.0 / 60.0,
            target,
            sel("gameTick:"),
            true,
        ));
    });
}

#[cfg(feature = "gamepane")]
extern "C" fn on_game_tick(_this: Id, _cmd: Sel, _timer: Id) {
    crate::on_game_tick();
}

#[cfg(feature = "gamepane")]
pub fn stop_game_loop_timer() {
    GAME_TIMER.with(|t| {
        let existing = t.get();
        if existing != NIL {
            objc::send0(existing, sel("invalidate"));
            t.set(NIL);
        }
    });
}

#[cfg(feature = "gamepane")]
pub fn game_loop_active() -> bool {
    GAME_TIMER.with(|t| t.get() != NIL)
}

#[cfg(feature = "gamepane")]
thread_local! {
    /// The currently-scheduled frame `NSTimer` (NIL when the loop is stopped).
    /// Main-thread only.
    static GAME_TIMER: std::cell::Cell<Id> = const { std::cell::Cell::new(NIL) };
}

/// Quit the application (the standalone demo window has nothing to return to).
#[cfg(feature = "gamepane")]
pub fn quit() {
    let app = objc::send0(objc::get_class("NSApplication"), sel("sharedApplication"));
    objc::send1_id(app, sel("terminate:"), NIL);
}

// ── Timers and the event loop ──────────────────────────────────────────────

extern "C" fn on_metrics_tick(_this: Id, _cmd: Sel, _timer: Id) {
    crate::on_metrics_tick();
}

pub fn start_metrics_timer() {
    let target = METRICS_TIMER_TARGET
        .get_or_init(|| {
            let cls = objc::allocate_class(objc::get_class("NSObject"), "WinvmMetricsTimer");
            objc::add_method(cls, sel("metricsTick:"), on_metrics_tick as *const _, "v@:@");
            objc::register_class(cls);
            MainThreadPtr(objc::alloc_init("WinvmMetricsTimer"))
        })
        .0;
    objc::scheduled_timer(1.0 / 15.0, target, sel("metricsTick:"), true);
}

pub fn run() {
    let app = objc::send0(objc::get_class("NSApplication"), sel("sharedApplication"));
    // Quit when the (only) window closes, so a `cargo run` test session exits
    // cleanly instead of lingering window-less.
    objc::send1_id(app, sel("setDelegate:"), build_quit_on_last_window_delegate());
    objc::send1_bool(app, sel("activateIgnoringOtherApps:"), true);
    objc::send0(app, sel("run"));
}
