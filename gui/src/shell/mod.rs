//! The native shell seam — everything about this GUI that is *not* portable.
//!
//! WINVM's programming environment is HTML/CSS/JS (`gui/assets`,
//! `gui/reference`) rendered by [`crate::preprocess`] and driven by a VM
//! worker thread (`crate::vm_host`). All of that is platform-neutral by
//! construction. What is *not* neutral is the surrounding host: a window, a
//! web view, a menu bar, a timer, a clipboard, and a way for the VM worker
//! thread to wake the UI thread.
//!
//! This module is that boundary, and it is deliberately narrow — a dozen
//! free functions plus [`ScriptMessage`] and [`Waker`]. Both implementations
//! provide the same set:
//!
//! | | macOS (`mac.rs`) | Windows (`win.rs`) |
//! |---|---|---|
//! | window | `NSWindow` | `CreateWindowExW` |
//! | web view | `WKWebView` | WebView2 (`ICoreWebView2`) |
//! | JS -> Rust | `WKScriptMessageHandler` | `add_WebMessageReceived` |
//! | Rust -> JS | `evaluateJavaScript:` | `ExecuteScript` |
//! | page load | `loadFileURL:allowingReadAccessToURL:` | virtual host mapping |
//! | menu bar | `NSMenu` | `HMENU` + `WM_COMMAND` |
//! | periodic tick | `NSTimer` | `SetTimer` + `WM_TIMER` |
//! | worker -> UI wakeup | `performSelectorOnMainThread:` | `PostMessageW` |
//!
//! **Direction of control.** The shell owns the event loop and calls *up*
//! into `main.rs` (`crate::on_script_message`, `crate::on_vm_drain`,
//! `crate::on_menu_action`, `crate::on_metrics_tick`); `main.rs` calls
//! *down* for effects ([`eval_js`], [`load_rendered_page`], …). Neither
//! platform module knows anything about browsers, workspaces, or themes.

#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(windows)]
mod win;
#[cfg(windows)]
pub use win::*;

use std::collections::HashMap;

/// One message posted from the page (`smtk.js`'s `post()`), already decoded
/// into flat string fields by the platform shell.
///
/// Both hosts deliver a JSON-ish object of string values; `smtk.js` never
/// posts a nested structure, and numeric fields (`width`, `at`, …) are read
/// back with `.parse()` at the use site. Keeping the decoded form a plain
/// `HashMap<String, String>` means `main.rs`'s dispatch is identical on both
/// platforms — the macOS build reads an `NSDictionary`, the Windows build
/// parses the WebView2 JSON string, and neither difference escapes here.
#[derive(Debug, Default, Clone)]
pub struct ScriptMessage {
    fields: HashMap<String, String>,
}

impl ScriptMessage {
    pub fn from_fields(fields: HashMap<String, String>) -> Self {
        Self { fields }
    }

    /// The value for `key`, or `""` when absent — the same forgiving
    /// behaviour the Cocoa `dict_get_string` had, and for the same reason:
    /// this is data that crossed the JS bridge and must not be able to bring
    /// the shell down.
    pub fn get(&self, key: &str) -> String {
        self.fields.get(key).cloned().unwrap_or_default()
    }

    /// The `kind` discriminator every `smtk.js` message carries.
    pub fn kind(&self) -> String {
        self.get("kind")
    }
}
