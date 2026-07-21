//! The D-G4 HTML translator (`../PLAN.md` §3): rewrites the two Strongtalk
//! HTML extensions in memory, at load time, so the original `.html` files
//! under `reference/` stay byte-identical as a test corpus. Also injects the
//! in-page toolbar/status-bar chrome (D-G1: those are HTML inside the web
//! view, not native controls) and the `strongtalk.css`/`smtk.js` links.
//!
//! Scope note: ordinary internal cross-page links (`<a href="progenv2.html">`)
//! are deliberately *not* rewritten here — `smtk.js` intercepts those
//! generically at click time (any relative `.html` href), so this module
//! only has to handle the two attributes that don't already mean anything
//! to a browser: `doit=` and the unclosed `<smappl>` tag.

use std::path::Path;

/// The visual themes the shell's native Theme menu can switch between
/// (`main.rs::build_theme_menu`). Every theme reuses the exact same class
/// names (`.st-toolbar`, `.st-raised`, `.st-lowered`, `.st-transcript`,
/// `.st-statusbar`, `.st-field`, `.smappl`, `.st-outliner`, …) — see
/// `assets/hidef.css`'s own doc comment — so switching is just "load a
/// different stylesheet and a different icon set," never a template change.
/// `Classic` is the one genuinely different structural design (its own
/// PNG icon set, Win95 bevels); every other non-`HiDef` theme is a color/
/// font reskin of `hidef.css`'s exact structure, reusing its SVG icon set
/// (`assets/icons-hidef/`) — for the CRT/monochrome themes, recolored via
/// a CSS `filter` in that theme's own stylesheet rather than needing
/// dedicated icon assets (see e.g. `assets/crt-amber.css`'s own comment).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theme {
    /// The period-accurate Strongtalk (1996) look — `../PLAN.md` §2.
    Classic,
    /// A modern flat theme: system fonts, vector icons, no bevels — added
    /// because the pixel-art/Win95 chrome is true to the original but reads
    /// as dated for extended use; same layout and semantics, just restyled.
    HiDef,
    /// Modern dark mode — `assets/dark.css`.
    Dark,
    /// 1970s-80s amber-phosphor terminal — `assets/crt-amber.css`.
    CrtAmber,
    /// 1970s-80s green-phosphor terminal — `assets/crt-green.css`.
    CrtGreen,
    /// Squeak's playful, colorful 1996+ Morphic look — `assets/squeak.css`.
    Squeak,
    /// Xerox Alto/Star, the monochrome bitmap-display era Smalltalk itself
    /// was born on (1970s-early 80s) — `assets/alto-mono.css`.
    AltoMono,
    /// Self — the direct ancestor (README's Self→Strongtalk lineage) and the
    /// birthplace of the outliner UI; warm-gray Sun-workstation look —
    /// `assets/self.css`.
    SelfLang,
    /// Smalltalk-80 "Blue Book" — the canonical original: 1-bit black-on-white
    /// with dithered gray and the Blue Book's blue — `assets/smalltalk80.css`.
    Smalltalk80,
    /// BYTE, August 1981 — the issue that introduced Smalltalk-80 (the balloon
    /// cover); warm cream-paper palette — `assets/byte81.css`.
    Byte81,
    /// Solarized Light — Ethan Schoonover's beloved code palette —
    /// `assets/solarized-light.css`.
    SolarizedLight,
    /// Solarized Dark — the dark half of the Solarized pair —
    /// `assets/solarized-dark.css`.
    SolarizedDark,
    /// High Contrast — an accessibility theme: maximum contrast, larger type —
    /// `assets/high-contrast.css`.
    HighContrast,
}

impl Theme {
    /// Every variant, in native Theme-menu display order — the single
    /// source of truth `main.rs::build_theme_menu` walks to build the menu
    /// and the checkmark list, so adding a theme never means updating two
    /// separate lists by hand.
    pub const ALL: [Theme; 13] = [
        Theme::Classic,
        Theme::HiDef,
        Theme::Dark,
        Theme::CrtAmber,
        Theme::CrtGreen,
        Theme::Squeak,
        Theme::AltoMono,
        Theme::SelfLang,
        Theme::Smalltalk80,
        Theme::Byte81,
        Theme::SolarizedLight,
        Theme::SolarizedDark,
        Theme::HighContrast,
    ];

    /// Parse a CLI theme name (case-insensitive, dashes optional) — used by
    /// the headless `macvm-gui render` "eyes" command.
    pub fn from_cli_name(name: &str) -> Option<Theme> {
        let key: String = name.chars().filter(|c| *c != '-' && *c != '_').flat_map(|c| c.to_lowercase()).collect();
        Theme::ALL.into_iter().find(|t| {
            let label: String = t.menu_label().chars().filter(|c| *c != '-' && *c != ' ').flat_map(|c| c.to_lowercase()).collect();
            label == key
        })
    }

    /// Absolute path to this theme's stylesheet file (for inlining it into a
    /// self-contained headless render).
    pub fn stylesheet_path(self) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(self.stylesheet_relative_path())
    }

    /// Native Theme-menu label.
    pub fn menu_label(self) -> &'static str {
        match self {
            Theme::Classic => "Classic",
            Theme::HiDef => "Hi-Def",
            Theme::Dark => "Dark",
            Theme::CrtAmber => "CRT Amber",
            Theme::CrtGreen => "CRT Green",
            Theme::Squeak => "Squeak Morphic",
            Theme::AltoMono => "Alto Mono",
            Theme::SelfLang => "Self",
            Theme::Smalltalk80 => "Smalltalk-80",
            Theme::Byte81 => "BYTE 1981",
            Theme::SolarizedLight => "Solarized Light",
            Theme::SolarizedDark => "Solarized Dark",
            Theme::HighContrast => "High Contrast",
        }
    }

    fn stylesheet_relative_path(self) -> &'static str {
        match self {
            Theme::Classic => "assets/strongtalk.css",
            Theme::HiDef => "assets/hidef.css",
            Theme::Dark => "assets/dark.css",
            Theme::CrtAmber => "assets/crt-amber.css",
            Theme::CrtGreen => "assets/crt-green.css",
            Theme::Squeak => "assets/squeak.css",
            Theme::AltoMono => "assets/alto-mono.css",
            Theme::SelfLang => "assets/self.css",
            Theme::Smalltalk80 => "assets/smalltalk80.css",
            Theme::Byte81 => "assets/byte81.css",
            Theme::SolarizedLight => "assets/solarized-light.css",
            Theme::SolarizedDark => "assets/solarized-dark.css",
            Theme::HighContrast => "assets/high-contrast.css",
        }
    }

    fn icon_url(self, icon: &str) -> String {
        match self {
            Theme::Classic => gui_file_url(&format!("reference/icons-png/{icon}.png")),
            _ => gui_file_url(&format!("assets/icons-hidef/{icon}.svg")),
        }
    }

    /// The monochrome silhouette set (`assets/icons-mono/`) the toolbar's
    /// `currentColor`-masked spans use (`toolbar_html`), embedded at compile
    /// time and served as a `data:` URI. Separate from `icon_url`'s colored
    /// hidef set on purpose: masking needs glyph-only alpha (the hidef icons
    /// carry an opaque background card, which a mask renders as a solid
    /// blob), while smappl `<img>` icons still want the colored set. A `data:`
    /// URI rather than a `file://` one because WKWebView does NOT load CSS
    /// mask-image subresources from file:// pages (a blocked mask masks
    /// everything away — the icons simply vanish in the real app, while an
    /// http-served harness shows them fine).
    fn mono_icon_data_url(icon: &str) -> Option<String> {
        let svg = match icon {
            "goBack" => include_str!("../assets/icons-mono/goBack.svg"),
            "goForward" => include_str!("../assets/icons-mono/goForward.svg"),
            "home" => include_str!("../assets/icons-mono/home.svg"),
            "open" => include_str!("../assets/icons-mono/open.svg"),
            "implementors" => include_str!("../assets/icons-mono/implementors.svg"),
            "senders" => include_str!("../assets/icons-mono/senders.svg"),
            "userHierarchy" => include_str!("../assets/icons-mono/userHierarchy.svg"),
            "hierarchy" => include_str!("../assets/icons-mono/hierarchy.svg"),
            "blankSheet" => include_str!("../assets/icons-mono/blankSheet.svg"),
            "texteditor" => include_str!("../assets/icons-mono/texteditor.svg"),
            "documentation" => include_str!("../assets/icons-mono/documentation.svg"),
            "abstract" => include_str!("../assets/icons-mono/abstract.svg"),
            "refresh" => include_str!("../assets/icons-mono/refresh.svg"),
            "smallerText" => include_str!("../assets/icons-mono/smallerText.svg"),
            "biggerText" => include_str!("../assets/icons-mono/biggerText.svg"),
            _ => return None,
        };
        // Minimal percent-encoding for a data: URI that must survive inside
        // url('…') inside a double-quoted HTML style attribute.
        let mut enc = String::with_capacity(svg.len() * 2);
        for c in svg.trim().chars() {
            match c {
                '%' => enc.push_str("%25"),
                '#' => enc.push_str("%23"),
                '"' => enc.push_str("%22"),
                '<' => enc.push_str("%3C"),
                '>' => enc.push_str("%3E"),
                '\'' => enc.push_str("%27"),
                '\n' => enc.push_str("%0A"),
                ' ' => enc.push_str("%20"),
                other => enc.push(other),
            }
        }
        Some(format!("data:image/svg+xml,{enc}"))
    }
}

/// Resolve the `data-icon="resources/NAME.bmp"` attributes a smappl `Button
/// withImage:` fragment carries (`world/33_smappl.mst`) into a themed icon:
/// inserts a `src` pointing at the active theme's asset for NAME. Keyed by
/// *logical name*, not the literal path (`gui/smappl.md` §5), so the corpus
/// text stays byte-identical (D-G4) while what actually loads follows the
/// theme. A path that doesn't reduce to a known cataloged name is left
/// as-is (no `src`) — the fall-through the doc describes.
pub fn rewrite_smappl_icons(html: &str, theme: Theme) -> String {
    const NEEDLE: &str = "data-icon=\"";
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(pos) = rest.find(NEEDLE) else {
            out.push_str(rest);
            break;
        };
        let val_start = pos + NEEDLE.len();
        let Some(q) = rest[val_start..].find('"') else {
            out.push_str(rest);
            break;
        };
        let path = &rest[val_start..val_start + q];
        let stripped = path.strip_prefix("resources/").unwrap_or(path);
        let name = stripped.strip_suffix(".bmp").unwrap_or(stripped);
        out.push_str(&rest[..pos]);
        if !name.is_empty() {
            out.push_str(&format!("src=\"{}\" ", theme.icon_url(name)));
        }
        out.push_str(&rest[pos..val_start + q + 1]); // keep the data-icon="..." attr
        rest = &rest[val_start + q + 1..];
    }
    out
}

/// Reverse of [`html_escape_attr`] — turn an escaped `data-code` attribute
/// value back into the raw Smalltalk source. `&amp;` is undone last so a
/// literal `&lt;` in the source round-trips correctly.
fn unescape_attr(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// Resolve every smappl placeholder span (from [`rewrite_smappl_placeholders`])
/// in `html` to its rendered widget fragment, the way the live GUI does
/// asynchronously — but synchronously, for the headless renderer (`macvm-gui
/// render`, the offline "eyes" on a page). `render(code)` evaluates one
/// `visual=` expression (`VmHandle::render_fragment`); `None` (an unbuildable
/// shape) leaves the placeholder box in place, exactly the live fallback.
/// Icon `data-icon` names are themed via [`rewrite_smappl_icons`].
pub fn resolve_smappl_spans(
    html: &str,
    theme: Theme,
    mut render: impl FnMut(&str) -> Option<String>,
) -> String {
    const OPEN: &str = "<span class=\"smappl\" data-widget-id=\"";
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(pos) = rest.find(OPEN) else {
            out.push_str(rest);
            break;
        };
        let Some(tag_end_rel) = rest[pos..].find('>') else {
            out.push_str(rest);
            break;
        };
        let after_open = pos + tag_end_rel + 1;
        let Some(close_rel) = rest[after_open..].find("</span>") else {
            out.push_str(rest);
            break;
        };
        let span_end = after_open + close_rel + "</span>".len();
        let opening_tag = &rest[pos..after_open];

        out.push_str(&rest[..pos]);
        let code = attr_value(opening_tag, "data-code").as_deref().map(unescape_attr);
        let widget_id = attr_value(opening_tag, "data-widget-id");
        let replaced = match code.as_deref().and_then(&mut render) {
            Some(frag) => {
                let themed = rewrite_smappl_icons(&frag, theme);
                // Tag the fragment root with the span's id so a later live
                // refresh can re-find it (matches the worker's own tagging).
                match &widget_id {
                    Some(id) => tag_widget_id(&themed, id),
                    None => themed,
                }
            }
            None => rest[pos..span_end].to_string(), // leave the placeholder box
        };
        out.push_str(&replaced);
        rest = &rest[span_end..];
    }
    out
}

/// Fill the empty `<textarea class="st-smappl-src" data-src-class data-src-side
/// data-src-sel></textarea>` source editors a `ClassOutliner` fragment emits
/// (`world/34_tools.mst`) with method source. The running VM keeps no source
/// — it lives in image_store — so the worker (and the headless renderer)
/// enrich the VM-rendered fragment here. `source_of(class, side, selector)`
/// returns the text (or `None`, leaving the box empty). Reused by both call
/// sites so they can't drift.
pub fn inject_method_sources(
    html: &str,
    mut source_of: impl FnMut(&str, &str, &str) -> Option<String>,
) -> String {
    const OPEN: &str = "<textarea class=\"st-smappl-src";
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(pos) = rest.find(OPEN) else {
            out.push_str(rest);
            break;
        };
        let Some(gt) = rest[pos..].find('>') else {
            out.push_str(rest);
            break;
        };
        let tag_end = pos + gt + 1;
        let tag = rest[pos..tag_end].to_string();
        if rest[tag_end..].find("</textarea>").is_none() {
            out.push_str(rest);
            break;
        }
        // Emit up to and including the opening tag, then the resolved source
        // as its content; the original (empty) body and the `</pre>` are left
        // for the next iteration / final push.
        out.push_str(&rest[..tag_end]);
        if let (Some(c), Some(s), Some(sel)) = (
            attr_value(&tag, "data-src-class"),
            attr_value(&tag, "data-src-side"),
            attr_value(&tag, "data-src-sel"),
        ) {
            if let Some(src) = source_of(&unescape_attr(&c), &s, &unescape_attr(&sel)) {
                out.push_str(&html_escape_text(src.trim_end()));
            }
        }
        rest = &rest[tag_end..];
    }
    out
}

/// Stamp `data-widget-id` onto a rendered fragment's root element, so a live
/// refresh can re-find the fragment after the original placeholder span it
/// replaced is gone (`smtk.js`'s `macvmRenderSmappl`). Shared by the worker
/// (`vm_host`) and the headless renderer so both tag identically.
pub fn tag_widget_id(html: &str, id: &str) -> String {
    match html.find('<') {
        Some(lt) => {
            let rel = html[lt + 1..]
                .find(|c: char| c.is_whitespace() || c == '>')
                .map(|i| lt + 1 + i)
                .unwrap_or(html.len());
            format!("{} data-widget-id=\"{}\"{}", &html[..rel], id, &html[rel..])
        }
        None => html.to_string(),
    }
}

/// Value of `name="..."` within a single tag, or `None` if absent.
fn attr_value(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let len = tag[start..].find('"')?;
    Some(tag[start..start + len].to_string())
}

/// URL for an asset under the GUI resource root — the live page's theme
/// `<link rel="stylesheet">` and smtk.js `<script src>` (see `chrome_head`).
///
/// The scheme is the shell's business, because the two hosts grant local
/// resource access in fundamentally different ways: WKWebView widens a
/// `file://` page's sandbox via `loadFileURL:allowingReadAccessToURL:`, while
/// WebView2 has no such grant and instead publishes the tree at an
/// `http://` virtual host. Both are resolved at RUNTIME from
/// [`crate::gui_root`] (which honors `MACVM_GUI_ROOT`), never the build-time
/// `CARGO_MANIFEST_DIR` — a baked build-time path names the dev checkout
/// (absent on another machine) and falls outside the grant, so the
/// stylesheet/script silently load as nothing.
fn gui_file_url(relative: &str) -> String {
    crate::shell::asset_url(relative)
}

/// Percent-encode a filesystem path for embedding in a URL: keeps the path
/// separator and the RFC 3986 unreserved set literal, byte-encodes everything
/// else (so a space becomes `%20` and any non-ASCII byte is UTF-8
/// percent-encoded). Enough for a `<link href>`/`<script src>` the web view
/// must resolve; the asset suffixes we pass are plain ASCII, so their
/// `assets/…` text survives verbatim (the theme tests assert on it).
///
/// `pub(crate)` because both shells build their URLs with it — the encoding
/// rule is the same whether the result is a `file://` or an `http://` URL.
pub(crate) fn percent_encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'.' | b'-' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Rewrite `<a doit="CODE">` → `<a class="doit" href="javascript:void(0)"
/// data-code="ESCAPED">`, preserving the link's own text content and
/// everything else about the tag (in the corpus, `doit=` is always the only
/// attribute, so there's nothing else to carry over). Handles multi-line
/// `doit="..."` values (none observed in the corpus, but the scan is the
/// same either way).
fn rewrite_doit_links(html: &str) -> String {
    let needle = "<a doit=\"";
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(tag_start) = rest.find(needle) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..tag_start]);
        let code_start = tag_start + needle.len();
        let Some(code_end_rel) = rest[code_start..].find('"') else {
            out.push_str(&rest[tag_start..]);
            break;
        };
        let code = &rest[code_start..code_start + code_end_rel];
        out.push_str(&format!(
            "<a class=\"doit\" href=\"javascript:void(0)\" data-code=\"{}\"",
            html_escape_attr(code)
        ));
        // Leave the tag's closing `>` (and everything after) for the next
        // iteration's plain copy — mirrors the closing-`>` handling in
        // rewrite_smappl_placeholders below.
        rest = &rest[code_start + code_end_rel + 1..];
    }
    out
}

/// Rewrite `<smappl visual="CODE">` (no closing tag in the corpus — see
/// this module's doc comment) → a placeholder `<span>` carrying the code and
/// a stable `data-widget-id`. The span starts as the G0 lowered-bevel box
/// naming its code; on page load `smtk.js` posts a `{kind:"smappl"}` message
/// per span, the VM worker renders the `visual=` expression to an HTML
/// fragment (`VmHandle::render_fragment`, D-G5), and `main.rs` swaps the
/// span's `outerHTML` for it — so the box is only what the user sees for the
/// instant before the live widget arrives, or permanently if the shape isn't
/// buildable yet (the render fails and the box stays).
fn rewrite_smappl_placeholders(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let mut widget_seq = 0usize;
    loop {
        let Some(tag_start) = rest.find("<smappl") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..tag_start]);
        let after_tag_name = &rest[tag_start..];
        let Some(attr_start) = after_tag_name.find("visual=\"") else {
            // Malformed/unexpected shape — leave it alone rather than guess.
            out.push_str("<smappl");
            rest = &after_tag_name[7..];
            continue;
        };
        let code_start = attr_start + "visual=\"".len();
        let Some(code_end_rel) = after_tag_name[code_start..].find('"') else {
            out.push_str(&after_tag_name[..code_start]);
            rest = &after_tag_name[code_start..];
            continue;
        };
        let code = &after_tag_name[code_start..code_start + code_end_rel];
        let after_close_quote = &after_tag_name[code_start + code_end_rel + 1..];
        let Some(tag_end_rel) = after_close_quote.find('>') else {
            // No closing `>` found — bail out on the rest untouched.
            out.push_str(&after_tag_name[..code_start + code_end_rel + 1]);
            rest = after_close_quote;
            continue;
        };

        // The visual= value is an HTML attribute, so it arrives HTML-ENCODED (a
        // `<` in the Smalltalk source is written `&lt;`). Decode it once to the
        // true source, and keep its NEWLINES verbatim in data-code — a CodeView
        // `model:` string is multi-line and collapsing it would flatten the
        // displayed code (and any other widget with a multi-line string arg).
        // data-code re-encodes for the attribute; resolve_smappl_spans / smtk.js
        // decode it again before the VM — one clean round-trip, so source with
        // `<`/`>`/`&` no longer reaches the evaluator double-encoded. The visible
        // placeholder box stays whitespace-collapsed (it's only the G0 fallback).
        let source = unescape_attr(code);
        let placeholder_text = source.split_whitespace().collect::<Vec<_>>().join(" ");
        out.push_str(&format!(
            "<span class=\"smappl\" data-widget-id=\"s{}\" data-code=\"{}\">{}</span>",
            widget_seq,
            html_escape_attr(&source),
            html_escape_text(&placeholder_text)
        ));
        widget_seq += 1;
        rest = &after_close_quote[tag_end_rel + 1..];
    }
    out
}

fn html_escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub(crate) fn html_escape_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The nine Launcher toolbar buttons (`../PLAN.md` §1) plus back/forward/home
/// navigation (§G0: "back/forward/home navigation works" — the three icons
/// exist in the set but aren't part of the documented nine, so they're
/// placed first as a judgment call, not a source-anchored fact; see
/// `../PLAN.md` §5 if this needs revisiting).
const TOOLBAR_BUTTONS: &[(&str, &str, &str)] = &[
    ("goBack", "goBack", "Back"),
    ("goForward", "goForward", "Forward"),
    ("home", "home", "Start page"),
    ("open", "find", "Find definition"),
    ("implementors", "implementors", "Implementors"),
    ("senders", "senders", "Senders"),
    ("userHierarchy", "userHierarchy", "User hierarchy"),
    ("hierarchy", "hierarchy", "Full hierarchy"),
    ("blankSheet", "workspace", "Workspace"),
    ("texteditor", "editor", "Text editor"),
    ("documentation", "documentation", "Documentation"),
    ("abstract", "canvas", "Canvas"),
    // G5 polish items (`../PLAN.md` §4 G5) — not part of the documented
    // nine either, same judgment call as goBack/goForward/home above.
    ("refresh", "refresh", "Refresh"),
    ("smallerText", "smallerText", "Smaller text"),
    ("biggerText", "biggerText", "Bigger text"),
];

fn toolbar_html(theme: Theme) -> String {
    let buttons: String = TOOLBAR_BUTTONS
        .iter()
        .map(|(icon, action, title)| {
            // Classic keeps its period PNG pixel-art (masking would flatten it);
            // every other theme renders the monochrome silhouette set
            // (`icons-mono/`) as a `currentColor` mask (`.st-ico`,
            // chrome_layout_style), so each icon takes that theme's own ink
            // colour automatically — no per-theme filter, and nothing left
            // hidef-blue or crushed to black.
            let glyph = match Theme::mono_icon_data_url(icon) {
                Some(data_url) if theme != Theme::Classic => format!(
                    "<span class=\"st-ico\" role=\"img\" aria-label=\"{title}\" \
                     style=\"-webkit-mask-image:url('{data_url}');mask-image:url('{data_url}')\"></span>"
                ),
                _ => {
                    let icon_url = theme.icon_url(icon);
                    format!("<img src=\"{icon_url}\" alt=\"{title}\">")
                }
            };
            format!("<button class=\"st-toolbtn\" data-action=\"{action}\" title=\"{title}\">{glyph}</button>")
        })
        .collect();
    // The metrics mini-dashboard lives at the right end of the toolbar (pushed
    // there by `margin-left:auto`); the native sampler fills it via
    // `window.macvmSetMetrics` (main.rs / smtk.js). Empty until the first
    // snapshot arrives.
    format!(
        "<div class=\"st-toolbar st-raised\" id=\"macvm-toolbar\">{buttons}\
         <div id=\"macvm-metrics\" class=\"st-metrics\" title=\"VM &amp; GC metrics\"></div></div>"
    )
}

/// `transcript` is the persisted history (`main.rs`'s `TRANSCRIPT`, appended
/// to by `append_transcript` on every `VmResponse::Transcript`) — every
/// freshly-built page starts the transcript pane showing real history
/// instead of always resetting to a blank "Ready." greeting, which used to
/// make any navigation (even just a theme switch) silently erase it.
fn statusbar_html(transcript: &str) -> String {
    // Newest-first: the persisted buffer is chronological (oldest → newest, the
    // order `append_transcript` writes it), so reverse the lines for display —
    // the most recent entry sits at the top of the pane, matching the live
    // prepend `macvmAppendTranscript` does. The log text lives in its own
    // `#macvm-transcript-log` child, NOT directly in `#macvm-transcript`,
    // because `macvmAppendTranscript`/`macvmClearTranscript` set that child's
    // `.textContent` — which would otherwise wipe out the sibling Clear button.
    let log: String = transcript.split('\n').rev().collect::<Vec<_>>().join("\n");
    format!(
        "<div class=\"st-transcript st-lowered\" id=\"macvm-transcript\" style=\"height: 72px\">\
         <button type=\"button\" class=\"st-transcript-clear\" data-action=\"clearTranscript\" \
         title=\"Clear the transcript\">Clear</button>\
         <div class=\"st-transcript-log\" id=\"macvm-transcript-log\">{}</div>\
         </div>\
         <div class=\"st-statusbar st-raised\"><span class=\"st-field\" id=\"macvm-status\">Ready</span></div>",
        html_escape_text(&log)
    )
}

/// A `<base href>` for `dir` (with the trailing slash `<base>` needs to be
/// treated as a directory, not a file). Required because the rendered HTML
/// is loaded from `gui/.rendered/current.html` (see `main.rs::navigate_to`
/// for why: `loadFileURL:allowingReadAccessToURL:` needs a real file to
/// load, and a single wide read-access grant is much simpler than one
/// scoped per page directory) — without this, the *page's own* relative
/// links/images would resolve against `.rendered/` instead of wherever the
/// original file actually lives.
fn base_href_tag(dir: &Path) -> String {
    // The scheme is the shell's (see `gui_file_url`); the percent-encoding is
    // needed either way, because in a packaged build this directory can live
    // under a spaced root and an unencoded base silently breaks every
    // relative link and image on the page.
    format!("<base href=\"{}\">", crate::shell::base_url(dir))
}

/// The `zoom` CSS property (non-standard, but WebKit implements it fully —
/// this shell only ever runs inside WKWebView) is the simplest way to scale
/// a whole rendered page uniformly: text, images, and layout together, like
/// a browser's page zoom. That's exactly what the G5 `biggerText`/
/// `smallerText` toolbar buttons are for (`../PLAN.md` §4 G5), so there's no
/// need to touch `strongtalk.css`/`hidef.css`'s own fixed-px sizes at all —
/// one `<style>` override here covers both themes identically.
fn font_scale_style(font_scale_percent: u32) -> String {
    format!("<style>body {{ zoom: {font_scale_percent}%; }}</style>")
}

/// Pin the app chrome: the toolbar to the top of the window and the transcript
/// + status bar to the bottom, with the page's own content (wrapped in
/// `#macvm-scroll` by [`preprocess_html`]) scrolling endlessly in the band
/// between them. One `<style>` injected after the theme stylesheet, so it wins
/// on document order without editing all seven theme files. It replaces the
/// themes' normal-flow chrome — including the old `.st-transcript { margin-top:
/// 30vh }` push-down that used to *fake* a bottom-anchored transcript on short
/// pages (it scrolled away with everything else once a page got tall).
///
/// `body` becomes a fixed-height flex column (`height: 100%` + `overflow:
/// hidden`), the toolbar and the bottom bars are non-shrinking end caps
/// (`flex: 0 0 auto`), and only `#macvm-scroll` scrolls (`flex: 1` +
/// `min-height: 0` so it can shrink below its content and actually scroll —
/// the classic flexbox-scroll gotcha). The two "fill the viewport" views
/// (`.st-workspace`, `.st-browser`, formerly sized `calc(100vh - 120px)` to
/// dodge the chrome by hand) just fill the scroll band now.
fn chrome_layout_style() -> String {
    "<style>\
     html, body { height: 100%; }\
     body { margin: 0; display: flex; flex-direction: column; overflow: hidden; }\
     #macvm-toolbar { flex: 0 0 auto; z-index: 5; display: flex; align-items: center; }\
     #macvm-metrics { margin-left: auto; display: flex; align-items: center; gap: 9px; \
       padding: 0 8px; white-space: nowrap; overflow: hidden; \
       font-family: var(--st-font-widget, ui-monospace, monospace); \
       font-size: 11px; line-height: 1.1; \
       color: var(--hd-text, var(--st-foreground, #202020)); }\
     .st-metric { display: flex; flex-direction: column; align-items: flex-start; gap: 1px; }\
     .st-metric-l { font-size: 8px; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.55; }\
     .st-metric-v { font-variant-numeric: tabular-nums; }\
     .st-metric svg { display: block; opacity: 0.85; }\
     .st-metric-bar { width: 52px; height: 5px; \
       background: var(--hd-text-muted, var(--st-gray, #9a9a9a)); }\
     .st-metric-bar > i { display: block; height: 100%; width: 0; \
       background: var(--hd-accent, var(--st-blue, #2b6cff)); }\
     #macvm-scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; }\
     #macvm-transcript { flex: 0 0 auto; margin-top: 0; position: relative; overflow: hidden; }\
     #macvm-transcript-log { height: 100%; overflow-y: auto; overflow-x: hidden; overscroll-behavior: contain; }\
     .st-transcript-clear { position: absolute; top: 3px; right: 3px; z-index: 2; \
       font-family: var(--st-font-widget, sans-serif); font-size: 11px; line-height: 1.2; \
       padding: 1px 6px; cursor: pointer; background: var(--st-background-gray, #c0c0c0); \
       color: var(--st-foreground, #000); border: 1px solid var(--st-gray, #808080); }\
     .st-transcript-clear:active { box-shadow: inset 1px 1px 0 var(--st-gray, #808080); }\
     .st-statusbar { flex: 0 0 auto; }\
     #macvm-scroll > .st-workspace, #macvm-scroll > .st-browser { height: 100%; }\
     #macvm-browser-packages { flex: var(--st-bw-pkg, 1) 1 0; min-width: 0; }\
     #macvm-browser-classes { flex: var(--st-bw-cls, 1) 1 0; min-width: 0; }\
     #macvm-browser-categories { flex: var(--st-bw-cat, 1) 1 0; min-width: 0; }\
     #macvm-browser-methods { flex: var(--st-bw-mth, 1) 1 0; min-width: 0; }\
     #macvm-browser > .st-browser-lists { flex: var(--st-bh-lists, 2) 1 0; min-height: 0; }\
     #macvm-browser > #macvm-browser-source { flex: var(--st-bh-src, 1) 1 0; min-height: 0; }\
     .st-splitter { flex: 0 0 6px; align-self: stretch; cursor: col-resize; touch-action: none; \
       background: transparent; border-radius: 3px; transition: background-color 100ms ease; }\
     .st-splitter[data-split=\"row\"] { cursor: row-resize; }\
     .st-splitter:hover, .st-splitter.st-splitting { \
       background: var(--hd-accent-soft, var(--st-background-gray, #c0c0c0)); }\
     .st-add-node > .st-header { opacity: 0.7; font-style: italic; }\
     .st-smappl-codeview { height: 15em; width: 100%; }\
     #macvm-toolbar button { color: inherit; }\
     .st-editor-class { display: block; width: 100%; max-width: 32em; margin: 0 0 8px 0; \
       box-sizing: border-box; padding: 4px 6px; \
       font-family: var(--st-font-widget, sans-serif); font-size: 13px; \
       background: var(--st-background, #fff); color: var(--st-foreground, #000); \
       border: 1px solid var(--st-gray, #808080); }\
     .st-editor-buffer { display: block; width: 100%; box-sizing: border-box; \
       min-height: 60vh; resize: vertical; padding: 8px; white-space: pre; \
       tab-size: 4; overflow: auto; \
       font-family: var(--st-font-code, ui-monospace, monospace); font-size: 13px; \
       line-height: 1.4; background: var(--st-background, #fff); \
       color: var(--st-foreground, #000); border: 1px solid var(--st-gray, #808080); }\
     .st-editor-buffer[readonly] { opacity: 0.92; }\
     .st-ico { display: inline-block; width: 24px; height: 24px; background-color: currentColor; \
       -webkit-mask-repeat: no-repeat; -webkit-mask-position: center; -webkit-mask-size: contain; \
       mask-repeat: no-repeat; mask-position: center; mask-size: contain; vertical-align: middle; }\
     </style>"
        .to_string()
}

fn chrome_head_extra(original_dir: &Path, theme: Theme, font_scale_percent: u32) -> String {
    // The `<meta charset>` matters even though the original corpus files
    // don't declare one: none of them use non-ASCII bytes, so the gap was
    // silent there, but this module's own injected/authored HTML (e.g. the
    // documentation pages) uses UTF-8 punctuation (em dashes, arrows), and
    // without an explicit charset WKWebView guessed a Latin-1-ish encoding
    // and mangled it. Declaring UTF-8 here covers every loaded page.
    format!(
        "<meta charset=\"utf-8\">\n{}\n<link rel=\"stylesheet\" href=\"{}\">\n<script src=\"{}\"></script>\n{}\n{}",
        base_href_tag(original_dir),
        gui_file_url(theme.stylesheet_relative_path()),
        gui_file_url("assets/smtk.js"),
        font_scale_style(font_scale_percent),
        chrome_layout_style(),
    )
}

/// Insert `extra` right before the first `</head>` (or, failing that,
/// right before `<body`, for a page with no explicit head section).
fn inject_before_head_close(html: &str, extra: &str) -> String {
    if let Some(pos) = html.to_ascii_lowercase().find("</head>") {
        format!("{}{}{}", &html[..pos], extra, &html[pos..])
    } else if let Some(pos) = html.to_ascii_lowercase().find("<body") {
        format!("{}{}{}", &html[..pos], extra, &html[pos..])
    } else {
        format!("{extra}{html}")
    }
}

/// Insert `extra` right after the opening `<body...>` tag's `>`.
fn inject_after_body_open(html: &str, extra: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let Some(body_start) = lower.find("<body") else {
        return format!("{extra}{html}");
    };
    let Some(tag_close_rel) = html[body_start..].find('>') else {
        return format!("{extra}{html}");
    };
    let insert_at = body_start + tag_close_rel + 1;
    format!("{}{}{}", &html[..insert_at], extra, &html[insert_at..])
}

/// Insert `extra` right before `</body>`.
fn inject_before_body_close(html: &str, extra: &str) -> String {
    if let Some(pos) = html.to_ascii_lowercase().find("</body>") {
        format!("{}{}{}", &html[..pos], extra, &html[pos..])
    } else {
        format!("{html}{extra}")
    }
}

/// Load `path`, apply the full D-G4 transform plus chrome injection, and
/// return the resulting HTML string ready to be written to a scratch file
/// (`gui/.rendered/current.html`, see `main.rs::navigate_to`) and loaded via
/// `loadFileURL:allowingReadAccessToURL:`. Since the loaded file's own
/// location is then `.rendered/`, not `path`'s directory, an injected
/// `<base href="file://{path's parent}/">` (see `base_href_tag`) is what
/// keeps the page's *own* relative links/images resolving exactly as they
/// do for the original file — the chrome this function injects uses
/// absolute `file://` URLs regardless (see `gui_file_url`), so it isn't
/// affected either way.
pub fn load_and_preprocess(
    path: &Path,
    theme: Theme,
    font_scale_percent: u32,
    transcript: &str,
) -> std::io::Result<String> {
    let raw = std::fs::read_to_string(path)?;
    let original_dir = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(preprocess_html(
        raw,
        original_dir,
        theme,
        font_scale_percent,
        transcript,
    ))
}

/// The chrome-injection half of [`load_and_preprocess`], factored out so
/// Rust-generated pages (not loaded from a corpus file — e.g. the class
/// browser view, `browser_render.rs`) get the exact same toolbar/status
/// bar/theme/font-scale treatment as any corpus page, via
/// [`render_generated_page`].
fn preprocess_html(
    raw: String,
    original_dir: &Path,
    theme: Theme,
    font_scale_percent: u32,
    transcript: &str,
) -> String {
    let mut html = raw;
    html = rewrite_doit_links(&html);
    html = rewrite_smappl_placeholders(&html);
    html = inject_before_head_close(
        &html,
        &chrome_head_extra(original_dir, theme, font_scale_percent),
    );
    // Wrap the page's own content in a single scrolling pane (`#macvm-scroll`)
    // so `chrome_layout_style` can pin the toolbar above it and the transcript/
    // status bar below it while the content scrolls endlessly in between. The
    // toolbar goes just inside `<body>` and opens the wrapper; the wrapper
    // closes just before the bottom bars, right before `</body>`.
    html = inject_after_body_open(
        &html,
        &format!(
            "{}<div class=\"st-scroll\" id=\"macvm-scroll\">",
            toolbar_html(theme)
        ),
    );
    html = inject_before_body_close(&html, &format!("</div>{}", statusbar_html(transcript)));
    html
}

/// Wrap `body_content` (already-built HTML, e.g. `browser_render`'s pane
/// markup) in a minimal page skeleton and run it through the same D-G4
/// preprocessing/chrome pipeline as a corpus page — for views the GUI
/// generates itself rather than loads from `reference/pages/`. `base_dir`
/// only matters if `body_content` has its own relative links/images (the
/// browser view doesn't, today); pass [`gui_root`]-relative paths there if
/// that ever changes. There's no original file whose relative links need
/// preserving, so `base_dir` doubles as the `<base href>` target directly
/// (contrast `load_and_preprocess`, where that's the corpus file's own
/// parent directory).
pub fn render_generated_page(
    title: &str,
    body_content: &str,
    base_dir: &Path,
    theme: Theme,
    font_scale_percent: u32,
    transcript: &str,
) -> String {
    let raw = format!(
        "<html><head><title>{}</title></head><body>{body_content}</body></html>",
        html_escape_text(title)
    );
    preprocess_html(raw, base_dir, theme, font_scale_percent, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doit_link_gets_class_and_data_code() {
        let input = r#"<a doit="VM collectGarbage">Collect Garbage</a>"#;
        let out = rewrite_doit_links(input);
        assert!(out.contains(r#"class="doit""#), "{out}");
        assert!(out.contains(r#"data-code="VM collectGarbage""#), "{out}");
        assert!(out.ends_with("Collect Garbage</a>"), "{out}");
    }

    #[test]
    fn smappl_multiline_becomes_placeholder_span() {
        let input = "<smappl visual=\" Button\n\t\t\twithImage: (Image fromFile: 'x')\n\t\t\taction: [ :b | ]\t\">";
        let out = rewrite_smappl_placeholders(input);
        assert!(out.starts_with(r#"<span class="smappl""#), "{out}");
        // data-code preserves the source (newlines kept — a multi-line CodeView
        // model: must survive); the VISIBLE placeholder box text is collapsed.
        assert!(out.contains(r#"data-code=""#), "{out}");
        let head = out.split("</span>").next().unwrap();
        let visible = head.rsplit('>').next().unwrap();
        assert!(
            visible.contains("Button withImage:") && !visible.contains('\n'),
            "visible box text is collapsed to one line: {visible:?}"
        );
    }

    #[test]
    fn resolve_smappl_spans_swaps_rendered_and_keeps_placeholder_on_none() {
        // Two placeholder spans (as rewrite_smappl_placeholders emits them).
        let html = "before \
            <span class=\"smappl\" data-widget-id=\"s0\" data-code=\"Glue xRigid: 4\">Glue xRigid: 4</span> \
            mid \
            <span class=\"smappl\" data-widget-id=\"s1\" data-code=\"CodeView forString\">CodeView forString</span> \
            after";
        let out = resolve_smappl_spans(&html, Theme::Classic, |code| {
            if code.starts_with("Glue") {
                Some("<span class=\"glue\"></span>".to_string())
            } else {
                None // unbuildable → leave the placeholder
            }
        });
        // The rendered fragment is swapped in AND tagged with the span's id.
        assert!(
            out.contains("class=\"glue\"") && out.contains("data-widget-id=\"s0\""),
            "rendered span swapped in and tagged: {out}"
        );
        assert!(
            out.contains("data-code=\"CodeView forString\""),
            "the None (unbuildable) span keeps its placeholder: {out}"
        );
        assert!(out.starts_with("before ") && out.ends_with(" after"), "surrounding text intact: {out}");
    }

    #[test]
    fn inject_method_sources_fills_empty_pre_blocks() {
        // Two selector editors; one has source, one (a `<=` selector, escaped
        // in the attr) resolves to None and stays blank.
        let html = concat!(
            r#"<textarea class="st-smappl-src" data-src-class="Point" data-src-side="instance" data-src-sel="x"></textarea>"#,
            r#"<textarea class="st-smappl-src" data-src-class="Point" data-src-side="instance" data-src-sel="&lt;="></textarea>"#,
        );
        let out = inject_method_sources(html, |cls, side, sel| {
            assert_eq!((cls, side), ("Point", "instance"));
            match sel {
                "x" => Some("x [ ^x ]".to_string()),
                "<=" => None, // the escaped attr must reach us unescaped
                other => panic!("unexpected selector {other}"),
            }
        });
        assert!(out.contains("x [ ^x ]"), "source spliced into the editor: {out}");
        // The None editor stays empty.
        assert!(out.contains(r#"data-src-sel="&lt;="></textarea>"#), "blank editor preserved: {out}");
    }

    #[test]
    fn resolve_smappl_spans_unescapes_data_code_for_the_renderer() {
        // A code value with an escaped char must reach the renderer raw.
        let html = r#"<span class="smappl" data-widget-id="s0" data-code="a &lt; b">x</span>"#;
        let mut seen = String::new();
        let _ = resolve_smappl_spans(html, Theme::Classic, |code| {
            seen = code.to_string();
            Some(String::new())
        });
        assert_eq!(seen, "a < b", "data-code must be HTML-unescaped for the VM");
    }

    #[test]
    fn theme_from_cli_name_is_case_and_dash_insensitive() {
        assert_eq!(Theme::from_cli_name("classic"), Some(Theme::Classic));
        assert_eq!(Theme::from_cli_name("Hi-Def"), Some(Theme::HiDef));
        assert_eq!(Theme::from_cli_name("crtamber"), Some(Theme::CrtAmber));
        assert_eq!(Theme::from_cli_name("nonsense"), None);
    }

    #[test]
    fn transcript_renders_newest_first_with_a_clear_button() {
        // The persisted buffer is chronological; the pane must show it reversed
        // (newest at the top) inside a `#macvm-transcript-log` child, with the
        // Clear button as a sibling that a textContent write won't clobber.
        let out = statusbar_html("first\nsecond\nthird");
        let log = &out[out
            .find("macvm-transcript-log")
            .expect("log wrapper present")..];
        let p3 = log.find("third").expect("third present");
        let p2 = log.find("second").expect("second present");
        let p1 = log.find("first").expect("first present");
        assert!(p3 < p2 && p2 < p1, "newest-first (third→second→first): {out}");
        assert!(
            out.contains(r#"data-action="clearTranscript""#),
            "clear button present: {out}"
        );
    }

    #[test]
    fn smappl_icon_resolves_to_a_themed_src() {
        let frag = r#"<img class="smappl-icon" width="20" height="20" data-icon="resources/edit.bmp">"#;
        let classic = rewrite_smappl_icons(frag, Theme::Classic);
        assert!(
            classic.contains("src=\"") && classic.contains("reference/icons-png/edit.png"),
            "classic theme resolves the logical name to a png, got {classic}"
        );
        // The original data-icon attribute is preserved (byte-identical corpus).
        assert!(classic.contains(r#"data-icon="resources/edit.bmp""#), "{classic}");
        // A different theme picks a different asset for the same name.
        let dark = rewrite_smappl_icons(frag, Theme::Dark);
        assert!(
            dark.contains("assets/icons-hidef/edit.svg"),
            "dark theme resolves to the svg, got {dark}"
        );
    }

    #[test]
    fn smappl_without_closing_tag_does_not_swallow_following_content() {
        let input = r#"<li><smappl visual="ClassHierarchyOutliner imbeddedVisualForClass: Object"></li><li>next</li>"#;
        let out = rewrite_smappl_placeholders(input);
        assert!(out.contains("</span></li><li>next</li>"), "{out}");
    }

    #[test]
    fn plain_links_are_untouched() {
        let input = r#"<a href="documentation/index.html">Browse Documentation</a>"#;
        let out = rewrite_doit_links(&rewrite_smappl_placeholders(input));
        assert_eq!(out, input);
    }

    #[test]
    fn themes_pick_distinct_stylesheets_and_icon_formats() {
        let classic_head = chrome_head_extra(Path::new("."), Theme::Classic, 100);
        assert!(
            classic_head.contains("assets/strongtalk.css"),
            "{classic_head}"
        );
        let hidef_head = chrome_head_extra(Path::new("."), Theme::HiDef, 100);
        assert!(hidef_head.contains("assets/hidef.css"), "{hidef_head}");

        let classic_toolbar = toolbar_html(Theme::Classic);
        assert!(
            classic_toolbar.contains("reference/icons-png/home.png"),
            "{classic_toolbar}"
        );
        let hidef_toolbar = toolbar_html(Theme::HiDef);
        assert!(
            hidef_toolbar.contains("data:image/svg+xml")
                && hidef_toolbar.contains("mask-image"),
            "{hidef_toolbar}"
        );
    }

    /// Every non-Classic theme: its own stylesheet, and the SAME monochrome
    /// silhouette icon set rendered as a `currentColor` mask (`.st-ico`), so
    /// the toolbar takes each theme's ink automatically — only `Classic`
    /// keeps its period PNG pixel-art as plain `<img>`s.
    #[test]
    fn non_classic_themes_share_the_mono_mask_icons() {
        for theme in Theme::ALL {
            let head = chrome_head_extra(Path::new("."), theme, 100);
            assert!(
                head.contains(theme.stylesheet_relative_path()),
                "{theme:?}: {head}"
            );
            let toolbar = toolbar_html(theme);
            if theme == Theme::Classic {
                assert!(
                    toolbar.contains("reference/icons-png/home.png"),
                    "{theme:?}: {toolbar}"
                );
            } else {
                assert!(
                    toolbar.contains("data:image/svg+xml")
                        && toolbar.contains("st-ico"),
                    "{theme:?}: {toolbar}"
                );
            }
        }
    }

    #[test]
    fn percent_encode_path_encodes_spaces_keeps_separators() {
        // A space (the packaged bundle root has one) → %20; separators and the
        // unreserved set pass through so `assets/dark.css` stays readable.
        assert_eq!(percent_encode_path("/tmp/My Bundle/gui"), "/tmp/My%20Bundle/gui");
        assert_eq!(percent_encode_path("/plain/ascii-_.~/x.css"), "/plain/ascii-_.~/x.css");
    }

    /// Regression for "no theme CSS / dead themes in the packaged `.app`": the
    /// live page's theme stylesheet and smtk.js links must resolve at RUNTIME
    /// under `gui_root()` (the `MACVM_GUI_ROOT` the launcher sets), NOT the
    /// baked `CARGO_MANIFEST_DIR` — and must survive the space in the bundle
    /// root (`~/Library/Application Support/…`). A build-time link there is
    /// absent on another machine and outside the webview's read-access grant,
    /// so it loads as nothing.
    #[cfg(target_os = "macos")]
    #[test]
    fn theme_links_use_runtime_gui_root_and_encode_spaces() {
        let prev = std::env::var_os("MACVM_GUI_ROOT");
        std::env::set_var("MACVM_GUI_ROOT", "/tmp/My Bundle/gui");
        let head = chrome_head_extra(Path::new("."), Theme::Dark, 100);
        // Restore immediately so a failed assert can't leak the override into
        // sibling tests (they assert only on env-independent relative suffixes).
        match prev {
            Some(v) => std::env::set_var("MACVM_GUI_ROOT", v),
            None => std::env::remove_var("MACVM_GUI_ROOT"),
        }
        assert!(
            head.contains("file:///tmp/My%20Bundle/gui/assets/dark.css"),
            "theme stylesheet must be rooted at the runtime gui_root, space-encoded: {head}"
        );
        assert!(
            head.contains("file:///tmp/My%20Bundle/gui/assets/smtk.js"),
            "smtk.js src must be rooted at the runtime gui_root: {head}"
        );
    }

    /// The Windows counterpart. WebView2 cannot load `file://` subresources at
    /// all, so the shell publishes the GUI tree at a virtual host instead
    /// (`shell::win`'s module doc) and asset links must be origin-relative
    /// URLs under it. A `file://` link here is the exact same user-visible
    /// failure the macOS test guards — an unstyled page with dead icons — so
    /// it is asserted against explicitly rather than only implied.
    #[cfg(windows)]
    #[test]
    fn theme_links_resolve_against_the_asset_origin() {
        let head = chrome_head_extra(Path::new("."), Theme::Dark, 100);
        let origin = crate::shell::ASSET_ORIGIN;
        assert!(
            head.contains(&format!("{origin}/assets/dark.css")),
            "theme stylesheet must resolve against the virtual host: {head}"
        );
        assert!(
            head.contains(&format!("{origin}/assets/smtk.js")),
            "smtk.js src must resolve against the virtual host: {head}"
        );
        assert!(
            !head.contains("file://"),
            "no file:// URL may survive into a WebView2 page — it silently loads as nothing: {head}"
        );
    }

    #[test]
    fn theme_all_has_no_duplicate_stylesheets_and_matches_its_own_length() {
        assert_eq!(Theme::ALL.len(), 13);
        let mut paths: Vec<&str> = Theme::ALL
            .iter()
            .map(|t| t.stylesheet_relative_path())
            .collect();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(
            paths.len(),
            Theme::ALL.len(),
            "every theme must have a distinct stylesheet"
        );
    }

    #[test]
    fn font_scale_is_injected_as_a_zoom_style() {
        let head = chrome_head_extra(Path::new("."), Theme::Classic, 130);
        assert!(head.contains("zoom: 130%"), "{head}");
    }
}
