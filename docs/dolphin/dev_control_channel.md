# Dolphin GUI dev control channel + screenshots (TCL-over-net)

**Why:** the GUI is developed by an agent that cannot see the screen. It must be
**driven programmatically and snapshotted to images** during development —
open a view, click, type, then capture the window to a PNG and read it back.

**Status:** the harness already exists and is proven on the macOS Cocoa GUI. Only
the **Windows server side** is new, and it must land **with sprint G1's first
window** so every G1+ step is drive-and-verify from the first pixel. Scope:
Windows-only, opt-in, loopback-only (a dev tool, never in a shipping build path).

## What already exists (reuse as-is)

- **rusttcl client** (`src/rusttcl/verbs.rs` `verb_gui` + `gui_request`): the
  `gui` verb family — `connect ?port?`, `eval <st>`, `doit <st>`, `view <name>`,
  **`snap <path>`**, `sleep <ms>` — over a `TcpStream` (`RusttclCtx::gui_conn`),
  framed `<len>\n<bytes>`, `OK`/`ERR` replies. I drive the GUI from
  `macvm rusttcl <script.tcl>`. **No client change needed.**
- **Protocol + server pattern** (`cocoa_gui/src/control.rs`): opt-in
  (`MACVM_COCOA_CTL=<port>`), loopback `TcpListener`; a **listener thread** reads
  frames, queues a `CtlReq`, and **wakes the UI thread** — it never touches the
  GUI or the VM. `serve()` runs on the UI thread: `eval`/`doit` via
  `VmHandle`, `snap` via `snapshot_client_area(path)`. **The blueprint to mirror.**

## The Windows server (new — in the `win_gui`/`winvm-mvp` host, V3/S2)

1. **Listener** — `MACVM_MVP_CTL=<port>` (default 7644 for parity). Loopback
   `TcpListener` on its own thread; same `<len>\n<bytes>` framing (copy
   `read_frame`/`write_frame`). One request in flight. Queue each request to the
   UI thread over an `mpsc` channel and wake it via **`PostMessageW` to the
   message-only window** (the `WM_APP_DRAIN` inbox V3 already provides) — the
   socket thread must never call into the image or Win32 GUI directly.
2. **serve()** on the UI thread (drained from the message-only window):
   - `eval <st>` → run on the Dolphin UI VM, answer `OK <printString>` / `ERR`.
   - `doit <st>` → run, `OK` / `ERR`. Both go through the re-entrant
     `dispatch_callback` (G0) so they interleave correctly with live message
     handling and recover per-entry.
   - `view <name>` → sugar for a `doit` (`UiSession switchToView: #name`).
   - `snap <path>` → `snapshot_client_area(hwnd, path)` (below), `OK` /
     `ERR no window`.
   - `sleep <ms>` → listener-side pause so scripts can wait out async worker
     replies / relayout.
3. **`snapshot_client_area(hwnd, path)`** (Win32, new): `GetClientRect` →
   create a compatible DC + DIB section → **`PrintWindow(hwnd, hdc,
   PW_CLIENTONLY|PW_RENDERFULLCONTENT)`** (falls back to `BitBlt` from the window
   DC if PrintWindow misses layered/child content) → `GetDIBits` into a BGRA
   buffer → encode PNG via **WIC** (`IWICImagingFactory`/`CLSID_WICPngEncoder` —
   Windows-native, no new crate) → write `path`. Which HWND: the UiSession's
   `lastWindow`/active shell (a `snap <path> <hwnd>` form can target a specific
   registered window later).

## G1 acceptance addition

Fold into the G1 gate (`dolphin_ui_sprints.md` G1): after "hello window +
button + paint + click round-trip", **the same window is driven and captured
over the control channel** — a rusttcl script does `gui connect`,
`gui doit 'Transcript show: …'`, `gui snap g1.png`, and the PNG is a non-empty
client-area capture of the window. That single test proves the whole
develop-by-automation loop this project depends on.

## Notes

- **Security:** loopback-only + opt-in env var, exactly like the Cocoa side; it
  executes arbitrary Smalltalk by design (it's a dev console), so it is never
  enabled in a distributed build.
- **Interpreter-tier:** control-channel doits stay interpreter-tier like all GUI
  dispatch (S12), so no JIT-frame-unwind concern crosses the socket boundary.
- **Ordering vs G0:** `eval`/`doit` from the channel are just more entries on the
  re-entrant stack; they need G0 (nested entries) only if invoked *from inside* a
  live handler. Top-level control doits work as soon as the host + a single-depth
  entry exist, so basic snapshotting is available even before deep nesting.
