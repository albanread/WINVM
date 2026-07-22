//! The Windows shell: a Win32 window hosting WebView2 (`../MIGRATION.md` §6,
//! milestone M6). The Cocoa twin is `mac.rs`; the seam they both implement is
//! documented in `mod.rs`.
//!
//! Three things differ from the macOS shell in ways worth stating up front,
//! because each replaces a mechanism that simply does not exist here.
//!
//! **1. Local assets load over a virtual host, not `file://`.** The macOS
//! shell calls `loadFileURL:allowingReadAccessToURL:<gui root>` to widen
//! WKWebView's `file://` subresource sandbox to the whole GUI tree. WebView2
//! has no such grant — a `file://` page cannot pull in `assets/smtk.js` or a
//! theme stylesheet at all. The supported equivalent is
//! `SetVirtualHostNameToFolderMapping`, which publishes the GUI root at
//! [`ASSET_ORIGIN`], so every asset is an ordinary same-origin `http://`
//! fetch. That is why [`crate::preprocess`] emits origin-relative URLs on
//! Windows and `file://` URLs on macOS: same tree, same pages, different
//! grant model.
//!
//! **2. `window.webkit.messageHandlers` is synthesised.** `gui/assets/smtk.js`
//! is shared verbatim with MACVM and posts through the WKWebView API. Rather
//! than fork the asset (it would then drift), the shell injects a shim via
//! `AddScriptToExecuteOnDocumentCreated` that forwards those calls to
//! `window.chrome.webview.postMessage`. The web environment therefore needs
//! **no Windows-specific edits at all** — which is what `MIGRATION.md` §1
//! means by "the HTML/CSS/JS environment itself is portable".
//!
//! **3. Never block on an async WebView2 call from inside a callback.**
//! `webview2-com`'s `wait_for_async_operation` runs a *nested message loop*.
//! That is fine during startup, but calling it from inside an event handler
//! (e.g. `ExecuteScript` from within `WebMessageReceived`) pumps messages
//! re-entrantly inside a COM callback and deadlocks — confirmed by actually
//! hanging a probe on it, not by reading docs. So [`eval_js`] is
//! fire-and-forget with a no-op completion handler, which costs nothing: the
//! shell never wants a script's result, exactly as the macOS side passes a
//! nil completion handler for the same reason.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};
use std::sync::mpsc;

use webview2_com::Microsoft::Web::WebView2::Win32::*;
use webview2_com::*;
use windows::core::{Interface, PCWSTR, PWSTR};
use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows::Win32::UI::HiDpi::{
    SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::preprocess::Theme;

/// The origin the GUI resource root is published at (see this module's doc
/// comment). A `.local`-suffixed name that cannot collide with a real host,
/// mapped to [`crate::gui_root`] at startup.
pub const ASSET_ORIGIN: &str = "http://winvm.local";

/// The VM worker thread posts this to wake the UI thread and drain responses
/// — the `performSelectorOnMainThread:` analogue. `PostMessageW` is the one
/// documented-thread-safe way into a window's queue, which is exactly the
/// property that call was chosen for on macOS.
const WM_VM_DRAIN: u32 = WM_APP + 1;

/// The ~15 Hz metrics sampler (`SetTimer` id).
const TIMER_METRICS: usize = 1;
const METRICS_INTERVAL_MS: u32 = 1000 / 15;

// Menu command ids. Fixed actions occupy a low block; theme items are
// `MENU_THEME_BASE + index into Theme::ALL`, mirroring the macOS shell's use
// of an NSMenuItem `tag` so one handler serves every theme and adding a theme
// stays a `Theme::ALL` edit.
const MENU_CLOSE: usize = 101;
const MENU_SAVE_WORLD: usize = 102;
const MENU_LOAD_WORLD: usize = 103;
const MENU_QUIT: usize = 104;
const MENU_CUT: usize = 110;
const MENU_COPY: usize = 111;
const MENU_PASTE: usize = 112;
const MENU_SELECT_ALL: usize = 113;
const MENU_RESTART_VM: usize = 120;
const MENU_HELP: usize = 130;
const MENU_THEME_BASE: usize = 200;

const CF_UNICODETEXT: u32 = 13;

/// The window handle, as a raw `isize` so it can live in an atomic and be
/// copied into the (`Send`) [`Waker`] handed to the VM worker thread.
static HWND_MAIN: AtomicIsize = AtomicIsize::new(0);

/// Cache-buster for [`load_rendered_page`] — see that function's doc.
static PAGE_GENERATION: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The WebView2 controller and view. UI-thread only, so a `thread_local`
    /// says exactly that in the type system — no `unsafe impl Send` needed
    /// (unlike the macOS shell, whose `Id` statics had to assert it by hand).
    static WEBVIEW: RefCell<Option<ICoreWebView2>> = const { RefCell::new(None) };
    static CONTROLLER: RefCell<Option<ICoreWebView2Controller>> = const { RefCell::new(None) };
    /// The Theme submenu, kept so its radio checkmark can be updated.
    static THEME_MENU: RefCell<Option<HMENU>> = const { RefCell::new(None) };
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn hwnd() -> HWND {
    HWND(HWND_MAIN.load(Ordering::Relaxed) as *mut _)
}

// ── Asset URLs ─────────────────────────────────────────────────────────────

/// URL for an asset under the GUI resource root, as an origin-relative URL on
/// the virtual host (see this module's doc comment for why not `file://`).
pub fn asset_url(relative: &str) -> String {
    format!(
        "{ASSET_ORIGIN}/{}",
        crate::preprocess::percent_encode_path(&relative.replace('\\', "/"))
    )
}

/// Base URL for a loaded page's own directory, so the page's relative links
/// and images resolve against where it actually lives rather than against
/// `.rendered/`.
///
/// The virtual host publishes [`crate::gui_root`], so an absolute directory
/// has to be expressed relative to that root — and with `/` separators, since
/// a Windows `\` is a path separator on disk but an ordinary character in a
/// URL. A directory outside the root cannot be addressed on the virtual host
/// at all; it falls back to the root, which keeps the page's chrome working
/// and degrades only its own relative links (all corpus pages live under the
/// root, so this is a guard, not a normal path).
pub fn base_url(dir: &Path) -> String {
    let root = crate::gui_root();
    match dir.strip_prefix(&root) {
        Ok(rel) => {
            let rel = rel.to_string_lossy().replace('\\', "/");
            let rel = rel.trim_matches('/');
            if rel.is_empty() {
                format!("{ASSET_ORIGIN}/")
            } else {
                format!(
                    "{ASSET_ORIGIN}/{}/",
                    crate::preprocess::percent_encode_path(rel)
                )
            }
        }
        Err(_) => format!("{ASSET_ORIGIN}/"),
    }
}

// ── Worker -> UI wakeup ────────────────────────────────────────────────────

/// A thread-safe handle the VM worker uses to wake the UI thread
/// (`vm_host.rs`). The actual cross-thread call is `PostMessageW`, which is
/// documented as safe from any thread — the property this whole design rests
/// on.
///
/// It deliberately carries only "am I a real waker", and reads the window
/// handle from [`HWND_MAIN`] at `notify` time rather than capturing it at
/// construction. That is not defensive style, it is a bug fix: `main` spawns
/// the VM worker *before* creating the window, so it can boot the world while
/// the shell comes up. A waker that captured the handle would capture 0, and
/// then every wakeup for the life of the process would silently do nothing —
/// the VM would compute correct answers that never reached the page. The
/// macOS waker has no such hazard (it lazily builds its bridge object), so
/// this is exactly the kind of shared-convention mismatch between two
/// separately-correct components that this port keeps turning up.
///
/// Found by driving a real doit through the running app and watching nothing
/// happen, not by reading the code.
#[derive(Clone, Copy)]
pub struct Waker {
    enabled: bool,
}

impl Waker {
    /// A waker with no UI thread behind it — headless tests drive
    /// `VmHost::drain_responses` directly and must not need a window. Unused
    /// outside `cfg(test)` by design.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn none() -> Self {
        Self { enabled: false }
    }

    pub fn notify(self) {
        if !self.enabled {
            return;
        }
        let hwnd = HWND_MAIN.load(Ordering::Relaxed);
        if hwnd == 0 {
            return; // window not up yet; the next response will wake it
        }
        unsafe {
            let _ = PostMessageW(Some(HWND(hwnd as *mut _)), WM_VM_DRAIN, WPARAM(0), LPARAM(0));
        }
    }
}

/// The waker for this process's main window. Safe to call before the window
/// exists — see [`Waker`].
pub fn waker() -> Waker {
    Waker { enabled: true }
}

// ── Startup ────────────────────────────────────────────────────────────────

/// Put this thread in a single-threaded COM apartment and make the process
/// per-monitor DPI aware.
///
/// Both are prerequisites rather than niceties: every WebView2 entry point
/// fails with `CO_E_NOTINITIALIZED` before `CoInitializeEx`, and without the
/// DPI call Windows bitmap-stretches the whole web view on a scaled display,
/// which looks like a blurry-font bug in the environment rather than a
/// missing shell call.
pub fn init() {
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        // S_FALSE ("already initialised on this thread") is success here.
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
    }
}

extern "system" fn wndproc(window: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    unsafe {
        match msg {
            // The VM worker finished something; drain its response queue.
            WM_VM_DRAIN => {
                crate::on_vm_drain();
                LRESULT(0)
            }
            WM_TIMER if w.0 == TIMER_METRICS => {
                crate::on_metrics_tick();
                LRESULT(0)
            }
            WM_COMMAND => {
                dispatch_menu_command((w.0 & 0xffff) as usize);
                LRESULT(0)
            }
            // Keep the web view filling the client area.
            WM_SIZE => {
                let mut rect = RECT::default();
                if GetClientRect(window, &mut rect).is_ok() {
                    CONTROLLER.with(|c| {
                        if let Some(controller) = c.borrow().as_ref() {
                            let _ = controller.SetBounds(rect);
                        }
                    });
                }
                LRESULT(0)
            }
            // Clicking the frame should land focus back in the page, not on
            // the bare host window (where keystrokes would go nowhere).
            WM_SETFOCUS => {
                CONTROLLER.with(|c| {
                    if let Some(controller) = c.borrow().as_ref() {
                        let _ = controller
                            .MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
                    }
                });
                LRESULT(0)
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            _ => DefWindowProcW(window, msg, w, l),
        }
    }
}

fn dispatch_menu_command(id: usize) {
    match id {
        MENU_CLOSE | MENU_QUIT => unsafe {
            let _ = PostMessageW(Some(hwnd()), WM_CLOSE, WPARAM(0), LPARAM(0));
        },
        MENU_SAVE_WORLD => crate::on_menu_action("saveWorld"),
        MENU_LOAD_WORLD => crate::on_menu_action("loadWorld"),
        MENU_CUT => crate::on_menu_action("cut"),
        MENU_COPY => crate::on_menu_action("copy"),
        MENU_PASTE => crate::on_menu_action("paste"),
        MENU_SELECT_ALL => crate::on_menu_action("selectAll"),
        MENU_RESTART_VM => crate::on_menu_action("restartVm"),
        MENU_HELP => crate::on_menu_action("help"),
        other if other >= MENU_THEME_BASE && other < MENU_THEME_BASE + Theme::ALL.len() => {
            crate::on_menu_action(&format!("theme:{}", other - MENU_THEME_BASE));
        }
        _ => {}
    }
}

/// Build the menu bar — the `NSMenu` counterpart. Item *actions* are named
/// with the same strings the macOS shell's selectors stand for, so
/// `main.rs::on_menu_action` is one shared dispatch for both platforms.
pub fn build_menu_bar() {
    unsafe {
        let Ok(menubar) = CreateMenu() else { return };

        let append = |menu: HMENU, id: usize, label: &str| {
            let text = wide(label);
            let _ = AppendMenuW(menu, MF_STRING, id, PCWSTR(text.as_ptr()));
        };
        let attach = |bar: HMENU, sub: HMENU, label: &str| {
            let text = wide(label);
            let _ = AppendMenuW(bar, MF_POPUP, sub.0 as usize, PCWSTR(text.as_ptr()));
        };

        if let Ok(file) = CreatePopupMenu() {
            append(file, MENU_CLOSE, "Close Window");
            let _ = AppendMenuW(file, MF_SEPARATOR, 0, PCWSTR::null());
            append(file, MENU_SAVE_WORLD, "Save World to Files");
            append(file, MENU_LOAD_WORLD, "Load World from Files");
            let _ = AppendMenuW(file, MF_SEPARATOR, 0, PCWSTR::null());
            append(file, MENU_QUIT, "Exit");
            attach(menubar, file, "&File");
        }

        if let Ok(edit) = CreatePopupMenu() {
            append(edit, MENU_CUT, "Cut\tCtrl+X");
            append(edit, MENU_COPY, "Copy\tCtrl+C");
            append(edit, MENU_PASTE, "Paste\tCtrl+V");
            let _ = AppendMenuW(edit, MF_SEPARATOR, 0, PCWSTR::null());
            append(edit, MENU_SELECT_ALL, "Select All\tCtrl+A");
            attach(menubar, edit, "&Edit");
        }

        if let Ok(vm) = CreatePopupMenu() {
            append(vm, MENU_RESTART_VM, "Restart VM Thread");
            attach(menubar, vm, "&VM");
        }

        if let Ok(theme) = CreatePopupMenu() {
            for (idx, t) in Theme::ALL.iter().enumerate() {
                append(theme, MENU_THEME_BASE + idx, t.menu_label());
            }
            THEME_MENU.with(|m| *m.borrow_mut() = Some(theme));
            attach(menubar, theme, "&Theme");
        }

        if let Ok(help) = CreatePopupMenu() {
            append(help, MENU_HELP, "WINVM GUI Shell Help");
            attach(menubar, help, "&Help");
        }

        let _ = SetMenu(hwnd(), Some(menubar));
        let _ = DrawMenuBar(hwnd());
    }
}

/// Reflect the active theme as a radio checkmark (the `setState:` analogue).
pub fn set_theme_checkmark(active_index: usize) {
    THEME_MENU.with(|m| {
        if let Some(menu) = *m.borrow() {
            unsafe {
                let _ = CheckMenuRadioItem(
                    menu,
                    0,
                    Theme::ALL.len() as u32 - 1,
                    active_index as u32,
                    MF_BYPOSITION.0,
                );
            }
        }
    });
}

/// Create the window and the WebView2 that fills it, install the
/// `window.webkit` shim and the JS->Rust message handler, and publish the GUI
/// root at [`ASSET_ORIGIN`].
///
/// Blocking (it pumps a nested loop via `wait_for_async_operation`) so that
/// `main.rs` keeps the same straight-line startup as the macOS shell: by the
/// time this returns, `navigate_to` can load a page. That is safe *here*,
/// during startup, and only here — see this module's doc comment.
pub fn create_window_and_webview() {
    unsafe {
        let instance = GetModuleHandleW(None).expect("GetModuleHandleW");
        let class_name = wide("WinvmGuiWindow");
        let class = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: instance.into(),
            lpszClassName: PCWSTR(class_name.as_ptr()),
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            ..Default::default()
        };
        RegisterClassW(&class);

        let title = wide("WINVM");
        let window = CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            PCWSTR(class_name.as_ptr()),
            PCWSTR(title.as_ptr()),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            980,
            760,
            None,
            None,
            Some(instance.into()),
            None,
        )
        .expect("CreateWindowExW");
        HWND_MAIN.store(window.0 as isize, Ordering::Relaxed);
        let _ = ShowWindow(window, SW_SHOW);

        // The environment's user-data folder defaults to beside the exe,
        // which is read-only for an installed build; put it under LOCALAPPDATA.
        let user_data = std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("WINVM")
            .join("WebView2");
        let _ = std::fs::create_dir_all(&user_data);
        let user_data_w = wide(&user_data.to_string_lossy());

        // Turn OFF SmartScreen (Edge's URL-reputation service) for this
        // control. This is not a security downgrade for a general browser —
        // the environment only ever loads this app's OWN local files over the
        // [`ASSET_ORIGIN`] virtual host; there is no untrusted navigation to
        // vet. Leaving it on made every click cost ~2 s: each navigation
        // targets a fresh `?g=<n>` URL (the cache-buster), so the reputation
        // check can never be cached and phones home to Microsoft on every
        // navigation, blocking the commit until it times out (the file itself
        // is served locally in ~2 ms — measured). Disabling it drops
        // click-to-content from ~2000 ms to ~5 ms. Set on the ENVIRONMENT so
        // it needs no launch-time flag.
        let options: ICoreWebView2EnvironmentOptions =
            CoreWebView2EnvironmentOptions::default().into();
        let browser_args = wide("--disable-features=msSmartScreenProtection");
        let _ = options.SetAdditionalBrowserArguments(PCWSTR(browser_args.as_ptr()));

        let (tx, rx) = mpsc::channel();
        CreateCoreWebView2EnvironmentCompletedHandler::wait_for_async_operation(
            Box::new(move |handler| {
                CreateCoreWebView2EnvironmentWithOptions(
                    PCWSTR::null(),
                    PCWSTR(user_data_w.as_ptr()),
                    &options,
                    &handler,
                )
                .map_err(Into::into)
            }),
            Box::new(move |code, env| {
                code?;
                let _ = tx.send(env);
                Ok(())
            }),
        )
        .expect(
            "WebView2 environment creation failed — is the WebView2 Runtime installed? \
             (It ships with Windows 11; otherwise install the Evergreen runtime.)",
        );
        let environment = rx.recv().ok().flatten().expect("no WebView2 environment");

        let (tx, rx) = mpsc::channel();
        CreateCoreWebView2ControllerCompletedHandler::wait_for_async_operation(
            Box::new(move |handler| {
                environment
                    .CreateCoreWebView2Controller(window, &handler)
                    .map_err(Into::into)
            }),
            Box::new(move |code, controller| {
                code?;
                let _ = tx.send(controller);
                Ok(())
            }),
        )
        .expect("WebView2 controller creation failed");
        let controller = rx.recv().ok().flatten().expect("no WebView2 controller");
        let webview = controller.CoreWebView2().expect("CoreWebView2");

        let mut rect = RECT::default();
        if GetClientRect(window, &mut rect).is_ok() {
            let _ = controller.SetBounds(rect);
        }

        // Publish the GUI tree at ASSET_ORIGIN so stylesheets, smtk.js, icons
        // and corpus images all load as ordinary same-origin requests.
        if let Ok(wv3) = webview.cast::<ICoreWebView2_3>() {
            let host = wide(ASSET_ORIGIN.trim_start_matches("http://"));
            let root = wide(&crate::gui_root().to_string_lossy());
            let _ = wv3.SetVirtualHostNameToFolderMapping(
                PCWSTR(host.as_ptr()),
                PCWSTR(root.as_ptr()),
                COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW,
            );
        }

        install_webkit_shim(&webview);
        install_message_handler(&webview);

        WEBVIEW.with(|w| *w.borrow_mut() = Some(webview));
        CONTROLLER.with(|c| *c.borrow_mut() = Some(controller));
    }
}

/// Give the page the `window.webkit.messageHandlers.macvm.postMessage` API
/// `gui/assets/smtk.js` expects, forwarding to WebView2's own channel — so
/// the shared web assets need no Windows-specific branch.
fn install_webkit_shim(webview: &ICoreWebView2) {
    let shim = wide(
        "window.webkit = window.webkit || {};\
         window.webkit.messageHandlers = window.webkit.messageHandlers || {};\
         window.webkit.messageHandlers.macvm = window.webkit.messageHandlers.macvm || {\
           postMessage: function (m) { window.chrome.webview.postMessage(m); }\
         };",
    );
    let wv = webview.clone();
    let (tx, rx) = mpsc::channel();
    let _ = AddScriptToExecuteOnDocumentCreatedCompletedHandler::wait_for_async_operation(
        Box::new(move |handler| unsafe {
            wv.AddScriptToExecuteOnDocumentCreated(PCWSTR(shim.as_ptr()), &handler)
                .map_err(Into::into)
        }),
        Box::new(move |code, _id| {
            code?;
            let _ = tx.send(());
            Ok(())
        }),
    );
    let _ = rx.recv();
}

fn install_message_handler(webview: &ICoreWebView2) {
    let mut token = 0i64;
    let _ = unsafe {
        webview.add_WebMessageReceived(
            &WebMessageReceivedEventHandler::create(Box::new(move |_wv, args| {
                let Some(args) = args else { return Ok(()) };
                let mut json = PWSTR::null();
                args.WebMessageAsJson(&mut json)?;
                let text = json.to_string().unwrap_or_default();
                let message = crate::shell::ScriptMessage::from_fields(parse_flat_json(&text));
                crate::on_script_message(&message);
                Ok(())
            })),
            &mut token,
        )
    };
}

// ── Page loading and scripting ─────────────────────────────────────────────

/// Show the freshly-written `.rendered/current.html` (see
/// `main.rs::display_html`).
///
/// The URL carries a monotonically increasing `?g=` generation, because every
/// navigation targets that same path: without it WebView2 can treat the load
/// as a same-document navigation and serve the previous bytes, which would
/// show a stale page after a theme switch or a browser-pane update. The
/// page's own relative links are unaffected — they resolve against the
/// `<base href>` `preprocess` injects, not against this URL.
pub fn load_rendered_page(_path: &Path) {
    let generation = PAGE_GENERATION.fetch_add(1, Ordering::Relaxed);
    let url = wide(&format!("{ASSET_ORIGIN}/.rendered/current.html?g={generation}"));
    WEBVIEW.with(|w| {
        if let Some(webview) = w.borrow().as_ref() {
            unsafe {
                let _ = webview.Navigate(PCWSTR(url.as_ptr()));
            }
        }
    });
}

/// Run `js` in the page, discarding the result (see this module's doc
/// comment: a blocking wait here would deadlock when called from inside a
/// WebView2 event handler, which is most of the time).
pub fn eval_js(js: &str) {
    let script = wide(js);
    WEBVIEW.with(|w| {
        if let Some(webview) = w.borrow().as_ref() {
            unsafe {
                let _ = webview.ExecuteScript(
                    PCWSTR(script.as_ptr()),
                    &ExecuteScriptCompletedHandler::create(Box::new(|_, _| Ok(()))),
                );
            }
        }
    });
}

// ── Clipboard ──────────────────────────────────────────────────────────────

/// The general clipboard's text (`NSPasteboard`'s counterpart), empty when it
/// holds no text.
pub fn clipboard_read() -> String {
    unsafe {
        if OpenClipboard(Some(hwnd())).is_err() {
            return String::new();
        }
        let text = match GetClipboardData(CF_UNICODETEXT) {
            Ok(handle) if !handle.is_invalid() => {
                let global = HGLOBAL(handle.0);
                let ptr = GlobalLock(global) as *const u16;
                if ptr.is_null() {
                    String::new()
                } else {
                    let mut len = 0usize;
                    while *ptr.add(len) != 0 {
                        len += 1;
                    }
                    let s = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
                    let _ = GlobalUnlock(global);
                    s
                }
            }
            _ => String::new(),
        };
        let _ = CloseClipboard();
        text
    }
}

/// Put `text` on the general clipboard.
pub fn clipboard_write(text: &str) {
    let utf16 = wide(text);
    unsafe {
        if OpenClipboard(Some(hwnd())).is_err() {
            return;
        }
        let _ = EmptyClipboard();
        let bytes = utf16.len() * std::mem::size_of::<u16>();
        if let Ok(global) = GlobalAlloc(GMEM_MOVEABLE, bytes) {
            let ptr = GlobalLock(global) as *mut u16;
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(utf16.as_ptr(), ptr, utf16.len());
                let _ = GlobalUnlock(global);
                // Ownership transfers to the clipboard on success; on failure
                // the handle leaks one buffer, which beats a double free.
                let _ = SetClipboardData(CF_UNICODETEXT, Some(HANDLE(global.0)));
            }
        }
        let _ = CloseClipboard();
    }
}

/// Cut / Select All from `smtk.js`'s custom context menu. Copy and Paste do
/// not come through here — `main.rs` handles those against the real clipboard
/// (the text rides in with the message on copy; paste is injected back into
/// the focused field), the same split the macOS shell settled on after the
/// responder-chain route proved unreliable for a web view selection.
pub fn edit_action(action: &str) {
    match action {
        "cut" => eval_js("document.execCommand('cut')"),
        "selectAll" => eval_js("document.execCommand('selectAll')"),
        _ => {}
    }
}

/// Return keyboard focus to the page. On Windows the web view is a child of
/// the host window, so focus can sit on the bare host after a frame click.
///
/// Part of the shell seam both platforms implement; its only caller today is
/// the game pane's teardown, which is macOS-only — hence unused here rather
/// than missing.
#[allow(dead_code)]
pub fn focus_webview() {
    CONTROLLER.with(|c| {
        if let Some(controller) = c.borrow().as_ref() {
            unsafe {
                let _ = controller.MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
            }
        }
    });
}

// ── Timers and the event loop ──────────────────────────────────────────────

/// Start the always-on ~15 Hz metrics sampler (`NSTimer`'s counterpart).
pub fn start_metrics_timer() {
    unsafe {
        SetTimer(Some(hwnd()), TIMER_METRICS, METRICS_INTERVAL_MS, None);
    }
}

/// Run the message loop until the window closes.
pub fn run() {
    unsafe {
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

// ── JSON ───────────────────────────────────────────────────────────────────

/// Decode WebView2's `WebMessageAsJson` into the flat string map
/// [`crate::shell::ScriptMessage`] wraps.
///
/// Deliberately a hand-rolled scanner rather than a serde dependency: every
/// message `smtk.js` posts is a flat object of scalars (see `smtk.js`'s
/// `post()` calls — `kind`, `code`, `name`, `text`, `at`, …), numbers and
/// booleans are read back with `.parse()` at the use site, and the macOS
/// shell's `NSDictionary` path is equally untyped. Values that are not
/// scalars are skipped rather than flattened, so a future nested message
/// fails visibly at the use site instead of silently stringifying.
fn parse_flat_json(input: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes: Vec<char> = input.chars().collect();
    let mut i = 0usize;

    let skip_ws = |i: &mut usize, b: &[char]| {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    };

    skip_ws(&mut i, &bytes);
    if i >= bytes.len() || bytes[i] != '{' {
        return out;
    }
    i += 1;

    loop {
        skip_ws(&mut i, &bytes);
        if i >= bytes.len() || bytes[i] == '}' {
            break;
        }
        if bytes[i] == ',' {
            i += 1;
            continue;
        }
        let Some(key) = scan_string(&bytes, &mut i) else {
            break;
        };
        skip_ws(&mut i, &bytes);
        if i >= bytes.len() || bytes[i] != ':' {
            break;
        }
        i += 1;
        skip_ws(&mut i, &bytes);
        if i >= bytes.len() {
            break;
        }
        match bytes[i] {
            '"' => {
                if let Some(value) = scan_string(&bytes, &mut i) {
                    out.insert(key, value);
                }
            }
            '{' | '[' => skip_nested(&bytes, &mut i),
            _ => {
                let start = i;
                while i < bytes.len() && bytes[i] != ',' && bytes[i] != '}' {
                    i += 1;
                }
                let raw: String = bytes[start..i].iter().collect();
                out.insert(key, raw.trim().to_string());
            }
        }
    }
    out
}

/// Scan a JSON string literal at `*i`, resolving the escapes JSON defines.
fn scan_string(b: &[char], i: &mut usize) -> Option<String> {
    if *i >= b.len() || b[*i] != '"' {
        return None;
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        match b[*i] {
            '"' => {
                *i += 1;
                return Some(out);
            }
            '\\' => {
                *i += 1;
                if *i >= b.len() {
                    return None;
                }
                match b[*i] {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hex: String = b.get(*i + 1..*i + 5)?.iter().collect();
                        let code = u32::from_str_radix(&hex, 16).ok()?;
                        // A lone surrogate cannot be a `char`; emit the
                        // replacement rather than dropping the field.
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        *i += 4;
                    }
                    other => out.push(other),
                }
                *i += 1;
            }
            other => {
                out.push(other);
                *i += 1;
            }
        }
    }
    None
}

/// Step over a nested object/array value, respecting strings so a brace
/// inside one does not end the scan early.
fn skip_nested(b: &[char], i: &mut usize) {
    let mut depth = 0usize;
    while *i < b.len() {
        match b[*i] {
            '{' | '[' => {
                depth += 1;
                *i += 1;
            }
            '}' | ']' => {
                depth -= 1;
                *i += 1;
                if depth == 0 {
                    return;
                }
            }
            '"' => {
                scan_string(b, i);
            }
            _ => *i += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact message shapes `smtk.js` posts must survive the round trip
    /// WebView2 puts them through — this is the Windows equivalent of trusting
    /// `dict_get_string` on macOS, so it is worth pinning rather than assuming.
    #[test]
    fn parses_the_message_shapes_smtk_js_posts() {
        let m = parse_flat_json(r#"{"kind":"doit","code":"Transcript show: 'hi'."}"#);
        assert_eq!(m.get("kind").unwrap(), "doit");
        assert_eq!(m.get("code").unwrap(), "Transcript show: 'hi'.");

        // Numeric fields arrive unquoted and are `.parse()`d at the use site.
        let m = parse_flat_json(r#"{"kind":"editorKey","key":"a","at":42}"#);
        assert_eq!(m.get("at").unwrap(), "42");
        assert_eq!(m.get("at").unwrap().parse::<i64>().unwrap(), 42);
    }

    /// Source text crossing this bridge is full of newlines, quotes and
    /// backslashes; a naive scanner would truncate a method at its first
    /// escaped quote and silently save a corrupted body.
    #[test]
    fn resolves_json_escapes_in_source_text() {
        let m = parse_flat_json(
            r#"{"kind":"smapplAccept","text":"foo\n  ^'a \"quoted\" \\ tail'","sel":"foo"}"#,
        );
        assert_eq!(m.get("text").unwrap(), "foo\n  ^'a \"quoted\" \\ tail'");
        assert_eq!(m.get("sel").unwrap(), "foo");
    }

    #[test]
    fn resolves_unicode_escapes() {
        let m = parse_flat_json(r#"{"kind":"doit","code":"a → b"}"#);
        assert_eq!(m.get("code").unwrap(), "a → b");
    }

    /// A nested value must not swallow the fields after it.
    #[test]
    fn skips_nested_values_without_losing_later_fields() {
        let m = parse_flat_json(r#"{"kind":"x","nested":{"a":{"b":1}},"after":"kept"}"#);
        assert_eq!(m.get("kind").unwrap(), "x");
        assert_eq!(m.get("after").unwrap(), "kept");
        assert!(m.get("nested").is_none());
    }

    #[test]
    fn malformed_input_yields_no_fields_rather_than_panicking() {
        assert!(parse_flat_json("not json").is_empty());
        assert!(parse_flat_json("").is_empty());
        assert!(parse_flat_json(r#"{"unterminated":"#).is_empty());
    }
}
