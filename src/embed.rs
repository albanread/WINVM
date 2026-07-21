//! The embedding API (`docs/SPEC.md` Â§16.2, amendment A17): boot a
//! `VmState`, evaluate Smalltalk source, and route guest output through a
//! caller-supplied sink â€” the one library-consumable entry point besides
//! the CLI (`main.rs`). `gui/`'s worker thread (SPEC Â§16.1) is the first
//! real caller (S21 step 3).
//!
//! # Safety model (S21)
//!
//! A `VmHandle` MUST be driven from a dedicated thread the caller is
//! prepared to see disappear out from under it â€” `boot` arms
//! `FatalMode::ExitThread` (`runtime::vm_state`), so every guest-fatal
//! condition (`error:`, DNU, stack overflow, heap exhaustion, a genesis-time
//! mmap failure...) terminates only the calling thread
//! (`libc::pthread_exit`, which does not unwind â€” safe regardless of any
//! JIT-compiled frames on the native stack, sidestepping the
//! panic-through-hand-assembled-code hazard entirely rather than working
//! around it) instead of the whole process. `eval` additionally recovers a
//! genuine native fault outside the JIT code cache (`Alien`'s raw pointer
//! accessors, S20) as an ordinary `Err`, via `codecache::deopt_trap`'s
//! `sigsetjmp`/`siglongjmp` registry â€” see `eval`'s own doc for why that one
//! case does NOT terminate the thread.
//!
//! The caller must never call `.join()`/`.is_finished()` on that thread's
//! `JoinHandle` â€” a thread that exits via `pthread_exit` never completes
//! `JoinHandle`'s own bookkeeping (`std::thread`'s `Arc<Packet>` handshake
//! requires the spawned thread's normal-or-panicking completion path, which
//! `pthread_exit` skips), so `join`/`is_finished` panics/hangs. Detect death
//! via a crash-report message sent before termination instead; dropping an
//! unjoined handle afterward is safe.

use std::cell::Cell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::codecache::deopt_trap;
use crate::frontend::{self, CompileError};
use crate::oops::Oop;
use crate::runtime::{VmError, VmOptions, VmState};

pub use crate::runtime::vm_state::FatalMode;

/// Overrides how a guest-fatal condition (`error:`, DNU, stack overflow,
/// heap exhaustion) terminates on the CURRENT thread. [`VmHandle::boot`]/
/// [`VmHandle::boot_without_world`] set [`FatalMode::ExitThread`] (the GUI's
/// "kill the language thread, not the process" model, module doc). An
/// embedder that would rather a guest-fatal condition abort the whole
/// process â€” e.g. a test or a batch CLI runner, where a silently-dying
/// thread would just hang the harness â€” calls this with
/// [`FatalMode::ExitProcess`] AFTER booting (both settings are thread-local,
/// so this affects every `VmHandle` driven from this thread).
///
/// **A VM booted on the process's main thread MUST set `ExitProcess` (CG0).**
/// `boot` unconditionally arms `ExitThread`, whose terminal is
/// `pthread_exit` â€” sound for a background worker thread a supervisor can
/// respawn, but on the *main* thread a true fatal (heap exhaustion, stack
/// overflow) would `pthread_exit` the UI thread and leave a headless zombie:
/// the AppKit run loop's thread gone, the window frozen, the process neither
/// alive nor dead. This is the post-boot flip the Cocoa GUI's UI worker uses
/// (`cocoa_gui_design.md` Â§3 step 4, Â§5): boot (arms `ExitThread`) then
/// immediately `set_fatal_mode(FatalMode::ExitProcess)`, so a true fatal
/// exits the process (a nonzero `std::process::exit`) instead of zombifying
/// the UI thread. No new boot option is needed â€” this setter is the
/// mechanism.
pub fn set_fatal_mode(mode: FatalMode) {
    crate::runtime::vm_state::set_fatal_mode(mode);
}

/// Register a hook fired on THIS thread the instant before an `ExitThread`
/// fatal `pthread_exit` â€” the one exact signal a supervisor can use to learn
/// its `VmHandle` thread has died (`pthread_exit` runs no `Drop`, so a dropped
/// channel/`join` cannot report it). Thread-scoped like [`set_fatal_mode`];
/// call it once on the VM's own thread after boot. The Cocoa GUI's primary
/// generation uses this to post a "died" event to its watchdog, so ONLY a
/// genuine fatal (never a merely-busy VM running a long doit) triggers a
/// respawn. No-op if that thread's [`FatalMode`] is `ExitProcess`.
pub fn set_thread_fatal_hook(hook: Box<dyn Fn()>) {
    crate::runtime::vm_state::set_fatal_hook(hook);
}

// â”€â”€ The UI worker's thread-local `*mut VmHandle` + VM generation (CG3) â”€â”€â”€â”€â”€â”€â”€â”€
//
// design (`cocoa_gui_design.md` Â§3 step 4, Â§4.3): the UI worker â€” pinned to the
// process's main thread by AppKit â€” publishes a raw pointer to its own
// `VmHandle` here, so an AppKitâ†’Smalltalk callback trampoline (C6 reverse
// dispatch, `runtime::objc_delegate`) can read it and dispatch as a *top-level*
// `eval`/`perform`-style entry (through [`VmHandle::dispatch_callback`]). It is a
// raw pointer (not an `Arc`/reference) precisely because the trampolines are
// `extern "C"` IMPs AppKit invokes with no Rust lifetime to borrow against; the
// pointer stays valid because the UI worker `VmHandle` outlives the run loop (it
// is dropped only at process exit, or, in CG7, re-published across an in-place
// restart).
//
// This is the CANONICAL location (`cocoa_gui/src/boot.rs` re-exports it): a core
// callback trampoline in `runtime::objc_delegate` cannot reach a `cocoa_gui`-crate
// thread-local, and a headless core integration test must be able to publish a
// pointer and drive a delegate directly.
thread_local! {
    static UI_VM: Cell<*mut VmHandle> = const { Cell::new(std::ptr::null_mut()) };
}

/// Monotonic UI-VM generation, bumped every time a non-null `VmHandle` is
/// published (design Â§4.3): a delegate records the generation live at its mint,
/// and a callback trampoline refuses to dispatch a delegate whose recorded
/// generation is stale â€” one minted against a UI worker that has since been
/// restarted (CG7). Process-wide because delegate instances (ObjC objects) and
/// the trampolines that fire them are process-wide; the fail-*closed* stale
/// check never dispatches into a dead VM.
static UI_VM_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Publish this thread's UI worker `VmHandle` for the CG3 callback trampolines
/// (design Â§3 step 4). Call on the main thread after the boot handshake, before
/// running `CocoaUI startup` / entering `[NSApp run]`. Publishing a **non-null**
/// pointer BUMPS the UI-VM generation, so any delegate minted against a prior
/// (now-replaced) UI worker fails closed at its next callback (design Â§4.3);
/// publishing null (teardown) only clears the door and does NOT bump.
pub fn publish_ui_vm(p: *mut VmHandle) {
    UI_VM.with(|c| c.set(p));
    if !p.is_null() {
        UI_VM_GENERATION.fetch_add(1, Ordering::Release);
    }
}

/// The calling thread's published UI worker `VmHandle` pointer, or null if none
/// â€” the door a CG3 callback trampoline reads before dispatching. Null-safe.
pub fn ui_vm() -> *mut VmHandle {
    UI_VM.with(|c| c.get())
}

/// The current UI-VM generation (design Â§4.3). A delegate mint records this
/// value; a callback trampoline compares the delegate's recorded generation to
/// this and fails closed on a mismatch (a stale delegate from a restarted UI
/// worker). Zero before the first [`publish_ui_vm`], so a delegate can never be
/// minted at generation 0 and pass the check by accident.
pub fn current_ui_vm_generation() -> u64 {
    UI_VM_GENERATION.load(Ordering::Acquire)
}

thread_local! {
    /// True while a C6 delegate callback ([`VmHandle::dispatch_callback`]) is
    /// running on THIS thread. See [`callback_active`].
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Is a C6 delegate callback currently executing on this thread? (CG3 review.)
///
/// A delegate callback is a **top-level** VM entry, sound precisely because the
/// UI worker is quiescent when AppKit calls back. A *nested* callback â€” an
/// AppKit modal / menu-tracking / live-resize run loop pumped from INSIDE a
/// handler (which CG5+ introduces) â€” would re-borrow the same `&mut VmState`,
/// clobber the single per-thread `sigsetjmp` recovery slot, and overwrite the
/// one idle-baseline watermark, so a later fault would `siglongjmp` into a
/// returned frame and rewind to the wrong baseline. The delegate dispatch
/// trampoline reads this flag BEFORE it re-borrows the `VmHandle` and, if a
/// callback is already active, fails **closed** (returns the shape default â€”
/// the same safe answer a stale/unknown delegate gets) instead. No such nesting
/// path exists in CG3, but failing closed keeps the door sound in advance.
pub fn callback_active() -> bool {
    IN_CALLBACK.with(Cell::get)
}

/// Set/clear the [`callback_active`] flag. Private: only [`VmHandle::
/// dispatch_callback`] owns the flag's lifecycle, and it must clear it on EVERY
/// exit arm â€” including the `siglongjmp` recovery arms, which skip `Drop`, so an
/// RAII guard cannot be used here.
fn set_callback_active(v: bool) {
    IN_CALLBACK.with(|c| c.set(v));
}

/// Test-only hook to drive [`callback_active`] without a real nested callback,
/// so the trampoline's fail-closed guard is unit-testable off the main thread.
#[cfg(test)]
pub(crate) fn set_callback_active_for_test(v: bool) {
    set_callback_active(v);
}

/// A running, embedded VM instance â€” owns its `VmState` (and, through it,
/// the whole heap, code cache, and loaded world) outright. See the module
/// doc for the thread-lifetime contract every method here assumes.
pub struct VmHandle {
    vm: VmState,
    /// The clean idle watermark captured at the top of each `eval`/`exec`/
    /// `render_fragment`, so a guest-fatal `siglongjmp` â€” which skips every
    /// RAII `Drop` between the fault and the recovery point â€” can restore the
    /// VM to exactly its pre-doit state. Without it, the aborted doit's frames
    /// stay on `vm.stack` and its open `HandleScope`s stay in the handle arena,
    /// and both LEAK AND ACCUMULATE across errors (a workspace of typos slowly
    /// bloats the stack toward overflow and pins dead objects as GC roots) â€”
    /// the "recover into some other state, worse than useless" failure. See
    /// [`VmHandle::restore_after_guest_fatal`].
    idle_baseline: IdleBaseline,
    /// What to do when guest code raises an unhandled error â€” see
    /// [`ErrorPolicy`]. Default [`ErrorPolicy::Resume`].
    error_policy: ErrorPolicy,
}

/// A snapshot of the VM's clean, between-doits state â€” see
/// [`VmHandle::idle_baseline`].
#[derive(Clone, Copy, Default)]
struct IdleBaseline {
    stack_sp: usize,
    stack_fp: usize,
    stack_has_frame: bool,
    arena_len: usize,
}

/// Releases this thread's `sigsetjmp` recovery slot when the handle is
/// dropped. `eval`/`exec` claim one via `deopt_trap::claim_jmp_slot`
/// (idempotent per thread); in the embedded model the worker thread owns its
/// `VmHandle` and drops it on its own way out â€” a clean `worker_loop` return,
/// the common restart-on-death path where an idle worker exits as its request
/// channel is dropped â€” so `deregister_setjmp` runs on the very thread that
/// claimed the slot and frees it. Without this, every respawn would strand a
/// slot owned by a now-dead `pthread_t`, overflowing the fixed-size registry
/// after `JMP_REGISTRY_CAP` restarts. `deregister_setjmp` is keyed by
/// `pthread_self()`, so a `VmHandle` dropped on a thread that never claimed a
/// slot is simply a no-op there â€” safe on any thread. (A worker torn down via
/// `pthread_exit` â€” a genuinely fatal, unrecovered condition â€” skips `Drop`
/// by design; that path is rare now that DNU/`error:` recover in-thread, and
/// its slot is reclaimed if that `pthread_t` value is ever reused.)
impl Drop for VmHandle {
    fn drop(&mut self) {
        crate::codecache::deopt_trap::deregister_setjmp();
    }
}

/// A guest-visible evaluation failure â€” never a Rust panic (module doc's
/// safety model). `Compile` covers lex/parse/codegen errors (`eval`'s
/// source didn't compile). `RuntimeError` is an unhandled DNU or explicit
/// `self error:` â€” genuinely terminal for the CURRENT computation in
/// Smalltalk's own terms (no proceed semantics in v1), but NOT a sign the
/// VM itself is broken, so it's recovered at this same boundary rather than
/// tearing down the whole worker thread the way it did before this existed
/// (`runtime::error::dnu_fallback`/`primitives::prim_error`, via
/// `codecache::deopt_trap::raise_guest_fatal`) â€” this is what makes an
/// everyday Workspace typo an ordinary recoverable error instead of a full
/// VM respawn, matching real Smalltalk's own recoverable
/// `doesNotUnderstand:`. `NativeFault` is a genuinely recovered SIGSEGV/
/// SIGBUS in ordinary (non-JIT) native code â€” reachable today only through
/// `Alien`'s raw pointer accessors (S20) â€” turned into an ordinary `Err`
/// rather than terminating the thread, because `eval`'s own call frame is
/// the recovery point (see `eval`'s body).
#[derive(Debug)]
pub enum GuestError {
    Compile(CompileError),
    RuntimeError(String),
    NativeFault { sig: i32, pc: u64, far: u64 },
}

impl std::fmt::Display for GuestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuestError::Compile(e) => write!(f, "{e}"),
            GuestError::RuntimeError(msg) => write!(f, "{msg}"),
            GuestError::NativeFault { sig, pc, far } => write!(
                f,
                "native fault (signal {sig}) at pc=0x{pc:x} far=0x{far:x} â€” recovered, this eval aborted"
            ),
        }
    }
}

impl std::error::Error for GuestError {}

/// What a VM does when guest code raises an unhandled error â€” an unhandled DNU
/// or an explicit `self error:` (NOT a compile error, and NOT a VM-fatal
/// condition like heap exhaustion, which always terminates via `fatal_exit`).
/// Set per-VM with [`VmHandle::set_error_policy`]; the default is [`Resume`].
///
/// [`Resume`]: ErrorPolicy::Resume
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ErrorPolicy {
    /// Recover in-thread: abort the doit, rewind the VM to its clean idle
    /// baseline ([`VmHandle::restore_after_guest_fatal`]), and hand the error
    /// back as `Err(GuestError::RuntimeError)`. The VM stays alive and ready
    /// for the next doit. The right choice for anything interactive and
    /// long-lived â€” a REPL, the GUI Workspace, the editor â€” where a typo must
    /// never restart the VM. This is what a plain `doesNotUnderstand:` is in
    /// real Smalltalk: recoverable.
    #[default]
    Resume,
    /// Terminate the worker on any unhandled guest error, exactly as a VM-fatal
    /// condition does: [`crate::runtime::vm_state::fatal_exit`] â€” a
    /// `pthread_exit` under [`FatalMode::ExitThread`], so a supervisor
    /// (`gui::vm_host`) respawns a fresh VM, or a `process::exit` under
    /// `ExitProcess`. For **throwaway / pooled compute workers**, where a
    /// guaranteed-fresh VM after a failed job is safer and simpler than reusing
    /// a recovered one. The error is already on the transcript (written by
    /// `prim_error`/`dnu_fallback` before the unwind), so nothing is lost by
    /// not returning it. Only meaningful with `FatalMode::ExitThread`: pairing
    /// `Die` with `ExitProcess` exits the whole process on a guest typo, which
    /// is never what an interactive host wants.
    Die,
}

/// Where `Transcript show:`/`printOnStdout:` output goes (SPEC Â§16.2).
/// `Send` because the GUI's sink hands output across the worker-to-main-
/// thread channel.
pub trait TranscriptSink: Send {
    fn show(&mut self, text: &str);
}

/// A structured command from a game-primitive to the native game pane
/// (`docs/gamepane_design.md`). The core VM defines only this vocabulary; the
/// GUI applies each to the real Metal pane (`gui/src/game_pane.rs`). Drawing
/// commands mutate the pane's CPU buffer only; `Present` uploads and shows the
/// frame â€” so a whole frame's drawing costs one present, not one per op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameCommand {
    /// Set palette entry `index` (16..=255) to an opaque RGB colour.
    PaletteAt { index: u8, r: u8, g: u8, b: u8 },
    /// Clear the pane to palette `index` (0..=255).
    Cls { index: u8 },
    /// Clear the pane to an opaque RGB colour (convenience: uses palette 16).
    ClearTo { r: u8, g: u8, b: u8 },
    /// Plot a pixel at `(x, y)` in palette `index`.
    Pset { x: i64, y: i64, index: u8 },
    /// Draw a line from `(x0, y0)` to `(x1, y1)` in palette `index`.
    Line {
        x0: i64,
        y0: i64,
        x1: i64,
        y1: i64,
        index: u8,
    },
    /// Fill a `w`x`h` rectangle at `(x, y)` in palette `index`.
    FillRect {
        x: i64,
        y: i64,
        w: i64,
        h: i64,
        index: u8,
    },
    /// Fill a disc of radius `r` centred at `(cx, cy)` in palette `index`.
    Disc { cx: i64, cy: i64, r: i64, index: u8 },
    /// Overwrite the whole active buffer from a row-major slice of palette
    /// indices (`GamePane>>blit:`). The bulk path for CPU-generated frames â€” one
    /// command instead of one `Pset` per pixel.
    Blit { data: Vec<u8> },
    /// Upload the CPU buffer and present the frame.
    Present,
    /// Start the frame loop: the GUI's main-thread timer begins pulling one
    /// `GameStep` per tick (single-outstanding) to run the registered step
    /// block. `GamePane>>run`.
    StartLoop,
    /// Stop the frame loop. `GamePane>>stop`.
    StopLoop,
    /// Define a sprite from `/`-separated hex-row art (4 bits/pixel, palette
    /// index per cell) and place an instance of it, both keyed by the VM-chosen
    /// `id` so later `SpriteColor`/`MoveSprite` can address it. `id` is a
    /// monotonic counter minted Smalltalk-side; the GUI registry maps it to
    /// MacGamePane's own def/instance ids.
    DefineSprite { id: i64, rows: String },
    /// Set sprite `id`'s palette entry `index` (0..15) to an opaque RGB colour.
    SpriteColor {
        id: i64,
        index: u8,
        r: u8,
        g: u8,
        b: u8,
    },
    /// Move sprite `id`'s instance to `(x, y)`.
    MoveSprite { id: i64, x: i64, y: i64 },
    /// Play a named SFX preset (0=coin, 1=jump, 2=zap, 3=shoot, 4=explode,
    /// 5=powerup, 6=hurt, 7=click, 8=bang, 9=blip) on the one shared `Sfx`
    /// engine. `Sound coin play`.
    PlaySound { preset: u8 },
    /// Play an ABC-notation tune once in the background (a chiptune via the
    /// engine's ABC->MIDI path). `(Tune fromAbc: '...') playOnce`.
    PlayTune { abc: String },
}

/// Where game-primitive commands go â€” the game analogue of [`TranscriptSink`].
/// `Send` because the GUI's sink hands commands across the worker-to-main
/// thread channel, exactly like the transcript sink hands text.
pub trait GameSink: Send {
    fn emit(&mut self, cmd: GameCommand);
}

/// Per-VM, lock-free live signals a monitor (e.g. the GUI metrics dashboard)
/// samples at high frequency WITHOUT going through the VM's request queue â€” so
/// they stay live even while the VM is busy inside a long doit. One block per
/// `VmState`, shared out by `Arc`; deliberately NOT a process global, so
/// several VMs in one process each keep their own signals (a global would blend
/// them). Sampling is a plain relaxed atomic load â€” no lock, no worker round-trip.
#[derive(Debug, Default)]
pub struct VmLiveStats {
    /// Mirror of `VmState::compiled_depth` â€” the number of nested compiled
    /// activations currently on the native stack. A sampler reads `> 0` as
    /// "executing compiled code right now", which (sampled over time while the
    /// VM is busy) gives the interpreter/compiler execution ratio.
    pub compiled_depth: std::sync::atomic::AtomicU32,
}

/// A snapshot of a VM's slower runtime counters for the metrics dashboard â€”
/// read on the worker thread by [`VmHandle::metrics`] (a cheap field read, no
/// allocation, no GC) and shipped to the GUI. Bytes are raw; the GUI diffs
/// successive snapshots for rates (e.g. allocation/sec) and keeps a ring of
/// them for graphs. The interpreter/compiler ratio is NOT here â€” it is sampled
/// live from [`VmLiveStats`], because at the moment the worker services a
/// metrics request its Smalltalk stack is empty.
#[derive(Clone, Copy, Debug, Default)]
pub struct VmMetrics {
    // â”€â”€ memory (bytes) â”€â”€
    pub eden_used: u64,
    pub eden_capacity: u64,
    pub old_used: u64,
    pub old_committed: u64,
    pub old_reserved: u64,
    // â”€â”€ GC â”€â”€
    pub scavenges: u64,
    pub full_gcs: u64,
    pub bytes_allocated: u64,
    pub last_reclaimed: u64,
    // â”€â”€ compiled code â”€â”€
    pub nmethods: u64,
    pub code_used: u64,
    pub code_capacity: u64,
    // â”€â”€ JIT activity â”€â”€
    pub compilations: u64,
    pub deopts: u64,
    pub osr_entries: u64,
    pub ic_misses: u64,
}

/// Adapts a `TranscriptSink` into the plain `std::io::Write` that
/// `VmState::out` already expects â€” SPEC Â§16.2's sink trait is
/// guest-output-shaped (whole strings), `Write` is byte-shaped; this is the
/// one place that gap is bridged. Guest output is always valid UTF-8
/// (Smalltalk `String`s produced by `printString`/`displayString`), so a
/// lossy conversion here would only ever mask a pre-existing bug elsewhere â€”
/// hence `from_utf8` + `expect` rather than `from_utf8_lossy`.
struct SinkWriter(Box<dyn TranscriptSink>);

impl std::io::Write for SinkWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = std::str::from_utf8(buf).expect(
            "guest output must be valid UTF-8 (VmState::out only ever \
             receives printString/displayString bytes)",
        );
        self.0.show(text);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl VmHandle {
    /// Boots a fresh VM: arms `FatalMode::ExitThread` and this thread's
    /// foreign-fault handler (module doc), runs genesis, and loads
    /// `world_dir/world.list` â€” the same base image the CLI's `run`/`repl`
    /// subcommands load via `--world` (default `"world"`, `main.rs`'s
    /// `load_world_with_warning`). A missing `world.list` is not an error
    /// (matches `load_world_with_warning`'s own `Ok(false)` handling â€” the
    /// VM boots successfully with just genesis's built-in classes); a
    /// `world.list` that exists but fails to load (a real compile error)
    /// surfaces as `Err(VmError)`.
    ///
    /// Takes `world_dir` explicitly rather than hardcoding `"world"`
    /// internally (`docs/SPEC.md` Â§16.2's sketch shows `boot(opts)` alone) â€”
    /// deliberate: a hardcoded relative path would be untestable (exercising
    /// the `Err` path would need mutating the whole test process's cwd, a
    /// global, unsynchronized change unsafe under a parallel test runner)
    /// and would silently assume a launch-directory convention no embedder
    /// has actually agreed to. A future Browser bridge may need a different
    /// world source entirely (the `image_store` SQLite image, not a `.mst`
    /// file tree) â€” deferred, per `shiny-snacking-pine.md`'s own Deferred
    /// section; `gui/`'s Workspace-only caller (S21 step 3) just always
    /// passes `Path::new("world")`, the same effective default as today's
    /// CLI.
    ///
    /// MUST be called on the dedicated thread the caller is prepared to see
    /// terminate out from under it (module doc) â€” `set_fatal_mode` and the
    /// foreign-fault handler's sigaltstack are both thread-scoped, so
    /// calling `boot` on the wrong thread arms the wrong one.
    pub fn boot(opts: VmOptions, world_dir: &Path) -> Result<VmHandle, VmError> {
        set_fatal_mode(FatalMode::ExitThread);
        // Regardless of `opts.jit`: `VmState::with_options` only arms
        // PROBE's SIGSEGV/SIGBUS handler when the JIT is enabled (a pure
        // interpreter never emits a deopt trap), but an embedded VM needs
        // foreign-fault recovery either way â€” `docs/SPEC.md` Â§16.5 itself
        // requires the GUI's Browser accept path to run with
        // `MACVM_JIT=off`. See `arm_foreign_fault_handler`'s own doc.
        deopt_trap::arm_foreign_fault_handler();
        let mut vm = VmState::with_options(opts);
        if let Err(e) = frontend::world::load_world(&mut vm, world_dir) {
            return Err(VmError { msg: e.to_string() });
        }
        Ok(VmHandle {
            vm,
            idle_baseline: IdleBaseline::default(),
            error_policy: ErrorPolicy::default(),
        })
    }

    /// Boots a bare, genesis-only VM â€” the built-in classes exist (`Object`,
    /// `Behavior`, the immediates, â€¦) but no `world/` library is loaded. Same
    /// thread-safety arming as [`boot`] (module doc). For an embedder that
    /// supplies the library some other way than a `.mst` file tree â€” notably
    /// loading it from the versioned image database class-by-class via
    /// `eval` (the GUI's "load the world from the database" path, S22): boot
    /// genesis-only, then replay each stored class definition in load order.
    /// Never fails via `Result` (genesis itself uses `fatal_exit` for a
    /// heap-reservation failure, like every other VM entry point).
    pub fn boot_without_world(opts: VmOptions) -> VmHandle {
        set_fatal_mode(FatalMode::ExitThread);
        deopt_trap::arm_foreign_fault_handler();
        VmHandle {
            vm: VmState::with_options(opts),
            idle_baseline: IdleBaseline::default(),
            error_policy: ErrorPolicy::default(),
        }
    }

    /// Snapshot the VM's clean, between-doits watermark â€” call on the
    /// initial (`rc == 0`) pass of every `sigsetjmp`-guarded entry, before any
    /// guest code runs. Its partner [`restore_after_guest_fatal`] rewinds to it
    /// if the doit aborts. Stored in `self` (not a `sigsetjmp`-frame local), so
    /// it survives the `siglongjmp` that clobbers such locals.
    #[inline]
    fn snapshot_idle_baseline(&mut self) {
        self.idle_baseline = IdleBaseline {
            stack_sp: self.vm.stack.sp,
            stack_fp: self.vm.stack.fp,
            stack_has_frame: self.vm.stack.has_frame(),
            arena_len: self.vm.handle_arena.len(),
        };
    }

    /// Set this VM's [`ErrorPolicy`] â€” how it responds when guest code raises
    /// an unhandled error (a DNU or `self error:`). Default
    /// [`ErrorPolicy::Resume`]. Call after boot, before serving requests; a
    /// throwaway/pooled compute worker sets [`ErrorPolicy::Die`] so a failed
    /// job terminates it (and the supervisor respawns a fresh VM) instead of
    /// recovering the current one.
    pub fn set_error_policy(&mut self, policy: ErrorPolicy) {
        self.error_policy = policy;
    }

    /// This VM's current [`ErrorPolicy`].
    pub fn error_policy(&self) -> ErrorPolicy {
        self.error_policy
    }

    /// The decision at a guest-fatal recovery point, applying [`ErrorPolicy`].
    /// Under [`ErrorPolicy::Resume`] it rewinds to the clean idle baseline and
    /// yields the error to return as `Err`. Under [`ErrorPolicy::Die`] it never
    /// returns: the error is already on the transcript (written before the
    /// unwind), so it terminates the worker via `fatal_exit` â€” `pthread_exit`
    /// under `FatalMode::ExitThread`, letting the supervisor respawn a fresh
    /// VM. The `siglongjmp` that landed us here already did the hard part
    /// (unwinding safely out of deep JIT/guest frames to this Rust frame), so
    /// even the `Die` path terminates from a clean, ordinary call site.
    #[inline]
    fn handle_guest_fatal(&mut self, message: String) -> GuestError {
        match self.error_policy {
            ErrorPolicy::Resume => {
                self.restore_after_guest_fatal();
                GuestError::RuntimeError(message)
            }
            ErrorPolicy::Die => crate::runtime::vm_state::fatal_exit(1),
        }
    }

    /// The decision at a native-fault recovery point (a recovered SIGSEGV/
    /// SIGBUS â€” e.g. a bad `Alien` deref, S20), applying [`ErrorPolicy`]
    /// exactly like [`handle_guest_fatal`]: the `siglongjmp` that landed us
    /// here skipped every `Drop` just as a guest fatal's does, so Resume must
    /// rewind to the clean idle baseline before returning the error â€” leaving
    /// the aborted doit's frames/handle scopes/tier journal in place would
    /// not merely leak, it would be captured as the new "clean" state by the
    /// NEXT entry point's `snapshot_idle_baseline`, and a stale tier link or
    /// anchor makes a later GC walk panic or touch freed native stack.
    #[inline]
    fn handle_native_fault(&mut self, sig: i32, pc: u64, far: u64) -> GuestError {
        match self.error_policy {
            ErrorPolicy::Resume => {
                self.restore_after_guest_fatal();
                GuestError::NativeFault { sig, pc, far }
            }
            ErrorPolicy::Die => crate::runtime::vm_state::fatal_exit(1),
        }
    }

    /// Rewind the VM to the last [`snapshot_idle_baseline`] after a guest-fatal
    /// `siglongjmp` landed on the recovery branch. The jump skipped every RAII
    /// `Drop` between the fault and here, so three things the aborted doit left
    /// behind must be reclaimed by hand or they leak AND accumulate across
    /// errors: its abandoned frames on `vm.stack`, its still-open `HandleScope`s
    /// in the handle arena (permanent GC roots otherwise), and any `Cocoa
    /// poolDo:` mint-list scope. This is what makes a recovered VM genuinely
    /// return to its ready state rather than limp on in "some other state."
    #[inline]
    fn restore_after_guest_fatal(&mut self) {
        let b = self.idle_baseline;
        self.vm
            .stack
            .restore_baseline(b.stack_sp, b.stack_fp, b.stack_has_frame);
        self.vm.handle_arena.reset_to(b.arena_len);
        // A `poolDo:` that died left its mint-list scope open â€” from then on
        // every wrapper minted anywhere would append to a stale rooted list
        // forever (the C4 review's poisoned-machinery finding). A pool scope is
        // lexical and can never legitimately span doits.
        self.vm.cocoa_mint_stack.clear();
        // The tier journal and its companions are balanced only by the normal
        // return paths (`enter_compiled`/`rt_interpret_call`/... each pop what
        // they pushed); a raise that unwinds via `siglongjmp` from under
        // compiled frames skips every one of those pops, exactly like the
        // skipped `Drop`s above. At the idle baseline nothing compiled is
        // active by definition, so the clean state is empty/zero across the
        // board â€” a stale `IntoCompiled` left on top would make the NEXT
        // doit's first GC walk panic (`found IntoCompiled instead`) or, with
        // a stale anchor, walk freed native stack memory. Same for a parked
        // NLR whose frames are gone and `pending_deopts` entries keyed by
        // now-dead frame addresses (a recycled fp would falsely translate a
        // live frame's return pc).
        self.vm.tier_links.clear();
        self.vm.compiled_depth = 0;
        self.vm
            .live_stats
            .compiled_depth
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.vm.reg_block.last_compiled_fp = 0;
        self.vm.reg_block.last_compiled_kind = 0;
        self.vm.nlr_state = None;
        self.vm.pending_deopts.clear();
    }

    /// Compiles `source` as a single top-level item (SPEC Â§16.2: "compile as
    /// a doit, S5 REPL machinery, run, answer printString") and, for a doit,
    /// evaluates it and answers its `printString` â€” the same logic
    /// `main.rs`'s REPL uses (`print_result`). A class definition has no
    /// result value; answers `""`, same as an empty/whitespace-only
    /// `source`.
    ///
    /// Never panics or exits the process/thread on a GUEST failure (a
    /// compile error, an unhandled DNU/`error:`, or a recovered native
    /// fault) â€” all three become `Err`, per the module doc's safety model.
    /// A truly VM-fatal condition (stack overflow, heap exhaustion â€” the
    /// VM's OWN invariants/resources, not the guest program's correctness)
    /// still terminates the calling thread via `fatal_exit`: there is no
    /// Rust-level `Result` for those, and shouldn't be â€” a full worker
    /// respawn is the right response when the VM itself may be compromised,
    /// unlike an ordinary DNU (`shiny-snacking-pine.md`'s Context section:
    /// panic/`catch_unwind` was rejected as the *general* unwind mechanism
    /// here because it cannot safely cross a JIT-compiled frame â€” DNU/
    /// `error:` recovery below reuses `sigsetjmp`/`siglongjmp` instead,
    /// precisely because that mechanism doesn't do Rust-style unwinding at
    /// all and is already trusted through JIT frames for the native-fault
    /// case). `eval` simply never returns for a genuinely VM-fatal
    /// condition; the failure message was already written to the
    /// transcript sink first.
    #[allow(unsafe_code)]
    pub fn eval(&mut self, source: &str) -> Result<String, GuestError> {
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: `sigsetjmp` is called directly, inline, at this exact call
        // site â€” its frame (this `eval` invocation) stays live for the whole
        // recovery window: control does not return to the caller until
        // either the guest code below completes (normally, with a compile
        // error, or with an unhandled DNU/`error:` recovered via
        // `deopt_trap::raise_guest_fatal`) or a foreign fault `siglongjmp`s
        // straight back to here. Calling `sigsetjmp` through an intervening
        // wrapper function that itself returns before the fault happens is
        // unsound (the S21 setjmp-into-a-returned-frame bug found and fixed
        // in `codecache::deopt_trap`) â€” see `deopt_trap::sigsetjmp`'s own
        // doc.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }

        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(String::new()),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        match frontend::classdef::execute_top_item(&mut self.vm, item) {
            Ok(Some(result)) => Ok(print_result(&mut self.vm, result)),
            Ok(None) => Ok(String::new()),
            Err(e) => Err(GuestError::Compile(e)),
        }
    }

    /// Like [`eval`](Self::eval) but runs `source` purely for effect and does
    /// NOT compute the result's `printString`. Use this for loading class
    /// definitions and initialization doIts: computing a result's printString
    /// can itself invoke guest code that isn't ready yet during a boot (e.g.
    /// `Character value:` before `Character initTable` has populated its
    /// table), and a load doesn't want the printed value anyway. A `.mst`
    /// file load has the same property â€” it executes each top item and
    /// discards the value.
    #[allow(unsafe_code)]
    pub fn exec(&mut self, source: &str) -> Result<(), GuestError> {
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `eval` â€” `sigsetjmp` inline at this call site, whose
        // frame stays live for the whole recovery window.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            // Guest fatal (error:, DNU, â€¦) mid-exec â€” the same recovery arm
            // `eval` has. (Missing here until the worker M1 tests ran the
            // first-ever `error:` through `exec`: the fall-through hit the
            // native-fault expect below and panicked instead of Err-ing.)
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }
        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(()),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        frontend::classdef::execute_top_item(&mut self.vm, item).map_err(GuestError::Compile)?;
        Ok(())
    }

    /// The C6 reverse-dispatch callback door (`cocoa_gui_design.md` Â§4, Â§5
    /// Layer 1): run `body` â€” an AppKitâ†’Smalltalk delegate dispatch that marshals
    /// its native arguments, performs the handler, and marshals the native return
    /// â€” as a **top-level VM entry**, inside the very same per-entry `sigsetjmp`
    /// recovery window `eval`/`exec` install. The UI worker is quiescent whenever
    /// AppKit calls back (the run loop is Rust's, the VM at rest â€” design Â§3), so
    /// a callback is never a re-entrant `&mut VmState`; it is this fresh entry.
    ///
    /// Recovery is Layer 1: a handler that `error:`s or DNUs (a guest fatal), or
    /// a genuine native fault (`SIGSEGV`/`SIGBUS`) in our marshalling or a bad
    /// `Alien` inside the handler, unwinds via `siglongjmp` back to HERE; the VM
    /// is rewound to its clean idle baseline and `body`'s `default` (the return
    /// shape's defined default â€” `0`/`NO`/`nil`, all zero) is answered to AppKit,
    /// so the delegate return slot is never left undefined and the run loop pumps
    /// on. Unlike [`eval`](Self::eval) this **always resumes** â€” it never consults
    /// [`ErrorPolicy`] and never `Die`s: a delegate typo mid-run-loop must not
    /// kill the UI worker out from under AppKit (that is the design's whole
    /// point). A genuinely VM-fatal condition (heap exhaustion, stack overflow)
    /// still terminates via `fatal_exit` â€” on the main-thread UI worker that is
    /// `ExitProcess` (CG0), the honest outcome, not a recoverable callback error.
    #[allow(unsafe_code)]
    pub fn dispatch_callback(
        &mut self,
        default: u64,
        body: impl FnOnce(&mut VmState) -> u64,
    ) -> u64 {
        use std::io::Write as _;
        // Re-entrancy guard (CG3 review): a delegate callback is a TOP-LEVEL
        // entry, sound only because the VM is quiescent. If one is already active
        // on this thread â€” a nested AppKit callback pumped from a modal/tracking
        // run loop inside a handler (CG5+) â€” fail CLOSED with the shape default
        // rather than clobber the shared `sigsetjmp` slot + idle baseline (a
        // later fault would `siglongjmp` into a returned frame) and alias
        // `&mut VmState`. The delegate trampoline (`objc_delegate::dispatch`)
        // also checks this BEFORE re-borrowing the `VmHandle`, so the aliasing
        // `&mut` is avoided at source; this is the second line of defense for a
        // direct caller and the one that owns the flag's lifecycle.
        if callback_active() {
            return default;
        }
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `eval` â€” `sigsetjmp` inline at this exact call site, whose
        // frame (this `dispatch_callback` invocation) stays live for the whole
        // recovery window; `body` runs deeper on the stack and any fault
        // `siglongjmp`s straight back here.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            // A delegate handler raised (`error:`/DNU). The error was already
            // written to the transcript before the unwind; rewind to the clean
            // idle baseline (never `Die` â€” the run loop must keep pumping) and
            // answer the shape default. Clear the guard: the unwind skipped the
            // normal-return clear below, and the run loop must be able to
            // dispatch the NEXT callback.
            let _ = deopt_trap::take_last_guest_fatal_message();
            self.restore_after_guest_fatal();
            set_callback_active(false);
            return default;
        }
        if rc != 0 {
            // A recovered native fault (SIGSEGV/SIGBUS) inside our marshalling or
            // a bad `Alien` deref in the handler. Same recovery: restore + report
            // + shape default. (No prior transcript line exists for a raw fault,
            // unlike a guest `error:`, so name it here.)
            let info = deopt_trap::take_last_crash_info();
            self.restore_after_guest_fatal();
            set_callback_active(false);
            if let Some((sig, pc, far)) = info {
                let _ = writeln!(
                    self.vm.out,
                    "[cocoa-delegate] native fault (signal {sig}) at pc=0x{pc:x} far=0x{far:x} â€” recovered, delegate answered its default"
                );
            }
            return default;
        }
        // Capture the clean watermark before any guest code runs, so a fatal
        // abort rewinds to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        // Mark the callback active across `body` ONLY (after the `sigsetjmp`
        // landing arms, so a fault unwinds through the arms above which clear
        // it). Cleared explicitly on the normal-return path â€” an RAII guard
        // can't be used, since a `siglongjmp` skips `Drop`.
        set_callback_active(true);
        let out = body(&mut self.vm);
        set_callback_active(false);
        out
    }

    /// Evaluates a `<smappl visual="...">` expression and returns the HTML
    /// fragment the image renders for it (GUI D-G5 / `docs/APPS.md` Â§6: the
    /// Visual renders *itself* to HTML; Rust only transports the string).
    /// `code` is the raw `visual=` source; this wraps it as
    /// `(Visual coerce: ([<code>] value)) htmlFragment` â€” the
    /// `ElementSMAPPL.dlt` shape (`gui/smappl.md` Â§2) with the body run through
    /// a block â€” and hands back the resulting `String`'s raw bytes.
    ///
    /// The `[â€¦] value` wrapper is load-bearing: several corpus visuals are
    /// multi-statement with temp declarations (`progenv2.html`'s
    /// `| h v | h := (ClassHierarchyOutliner for: â€¦) filterOnâ€¦; orSubclasses.
    /// v := â€¦ . v`). Those can't be spliced straight into `(Visual coerce:
    /// (â€¦))` â€” `(| h v | â€¦)` is a parse error â€” but a block accepts temps and
    /// statements and answers its last expression, so wrapping evaluates both
    /// single-expression and multi-statement bodies uniformly.
    ///
    /// Unlike [`eval`](Self::eval) this does NOT run `printString` on the
    /// result: `htmlFragment` already answers a `String`, and printString
    /// would re-quote it. A non-`String` result (a widget shape whose
    /// `htmlFragment` isn't built yet, so the send DNUs, or `coerce:` let a
    /// non-Visual through) surfaces as `Err` and the caller shows the G0
    /// placeholder box â€” errors are swallowed to a fallback, never a broken
    /// page, matching `ElementSMAPPL`'s own `ifError:` discipline.
    #[allow(unsafe_code)]
    pub fn render_fragment(&mut self, code: &str) -> Result<String, GuestError> {
        let source = format!("(Visual coerce: ([{code}] value)) htmlFragment.");
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `eval` â€” `sigsetjmp` inline at this call site, whose frame
        // stays live for the whole recovery window.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }
        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(&source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(String::new()),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        match frontend::classdef::execute_top_item(&mut self.vm, item) {
            Ok(Some(result)) => match fragment_bytes(result) {
                Some(html) => Ok(html),
                // The fragment method answered a non-String â€” treat as a
                // render failure so the caller falls back to the placeholder.
                None => Err(GuestError::RuntimeError(
                    "smappl visual did not render to a String".to_string(),
                )),
            },
            Ok(None) => Ok(String::new()),
            Err(e) => Err(GuestError::Compile(e)),
        }
    }

    /// Fires a live widget's stored action closure (`SmapplRegistry fire:
    /// '<id>'`) and, if that closure answers a `String`, hands back its raw
    /// bytes â€” the HTML overlay a dialog action produces (`Visual>>promptOk:â€¦`,
    /// the differences2.html "Press Me!" demo). A non-`String` answer is a
    /// pure side-effect action (an icon button's `[:b | â€¦]`) and yields
    /// `Ok(None)` â€” no overlay. Any `Transcript` output the action makes still
    /// flows separately via the transcript sink.
    ///
    /// This is [`render_fragment`](Self::render_fragment)'s sibling: same
    /// signal-guarded top-item execution, but it wraps the `fire:` send instead
    /// of `htmlFragment` and treats a non-`String` result as "nothing to show"
    /// rather than a render error.
    #[allow(unsafe_code)]
    pub fn fire_widget_action(&mut self, action_id: &str) -> Result<Option<String>, GuestError> {
        // action_id is a worker-minted 'wN' id (SmapplRegistry), never user
        // text, so it needs no quoting â€” but guard the assumption cheaply.
        debug_assert!(action_id.bytes().all(|b| b.is_ascii_alphanumeric()));
        let source = format!("SmapplRegistry fire: '{action_id}'.");
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `render_fragment` â€” `sigsetjmp` inline at this call site,
        // whose frame stays live for the whole recovery window.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }
        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(&source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(None),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        match frontend::classdef::execute_top_item(&mut self.vm, item) {
            // A String answer is the dialog overlay; anything else (self, nil)
            // is a side-effect-only action, so there is nothing to inject.
            Ok(Some(result)) => Ok(fragment_bytes(result)),
            Ok(None) => Ok(None),
            Err(e) => Err(GuestError::Compile(e)),
        }
    }

    /// Evaluates `code` (wrapped `[<code>] value`, so multi-statement bodies
    /// with temps are fine â€” see [`render_fragment`](Self::render_fragment))
    /// and returns the answered `String`'s raw bytes. Used for image-side code
    /// that builds a plain string payload rather than a widget fragment â€” e.g.
    /// `Mandelbrot new commandsForWidth:height:` answering a Canvas
    /// draw-command batch (`docs/CANVAS.md` Â§5.2). A non-`String` answer is an
    /// `Err`, like `render_fragment`'s own non-`String` case.
    #[allow(unsafe_code)]
    pub fn eval_to_string(&mut self, code: &str) -> Result<String, GuestError> {
        let source = format!("([{code}] value).");
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `render_fragment` â€” `sigsetjmp` inline at this call site,
        // whose frame stays live for the whole recovery window.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }
        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(&source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(String::new()),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        match frontend::classdef::execute_top_item(&mut self.vm, item) {
            Ok(Some(result)) => fragment_bytes(result).ok_or_else(|| {
                GuestError::RuntimeError("expression did not answer a String".to_string())
            }),
            Ok(None) => Ok(String::new()),
            Err(e) => Err(GuestError::Compile(e)),
        }
    }

    /// Evaluates `code` (wrapped `[<code>] value`, like
    /// [`eval_to_string`](Self::eval_to_string)) and returns the answered
    /// `ByteArray`/`String`'s bytes RAW â€” no UTF-8 conversion, so arbitrary
    /// binary is preserved. Used for bulk pixel data: `Mandelbrot new
    /// pixelsForWidth:height:` answers a `w*h*4` RGBA `ByteArray`
    /// (`world/36_pixmap.mst`, `docs/CANVAS.md` pixel path). A non-byte-indexable
    /// answer is an `Err`.
    #[allow(unsafe_code)]
    pub fn eval_to_bytes(&mut self, code: &str) -> Result<Vec<u8>, GuestError> {
        let source = format!("([{code}] value).");
        let slot = deopt_trap::claim_jmp_slot();
        // SAFETY: as `render_fragment` â€” `sigsetjmp` inline at this call site,
        // whose frame stays live for the whole recovery window.
        let rc = unsafe { deopt_trap::sigsetjmp(deopt_trap::jmp_buf_ptr(slot), 1) };
        if rc == deopt_trap::GUEST_FATAL_JMP_VAL {
            let message = deopt_trap::take_last_guest_fatal_message().expect(
                "sigsetjmp returned GUEST_FATAL_JMP_VAL without a recorded guest-fatal message",
            );
            // Apply this VM's error policy: Resume rewinds to the clean idle
            // baseline and returns the error; Die terminates the worker (the
            // message is already on the transcript). See `handle_guest_fatal`.
            return Err(self.handle_guest_fatal(message));
        }
        if rc != 0 {
            let (sig, pc, far) = deopt_trap::take_last_crash_info()
                .expect("sigsetjmp returned nonzero without a recorded crash");
            return Err(self.handle_native_fault(sig, pc, far));
        }
        // Capture the clean watermark before any guest code runs, so a
        // guest-fatal abort can rewind to exactly here (`restore_after_guest_fatal`).
        self.snapshot_idle_baseline();
        let item = match frontend::parser::parse_one_top_item(&source) {
            Ok(Some(item)) => item,
            Ok(None) => return Ok(Vec::new()),
            Err(e) => return Err(GuestError::Compile(e)),
        };
        match frontend::classdef::execute_top_item(&mut self.vm, item) {
            Ok(Some(result)) => {
                let b = crate::oops::wrappers::ByteArrayOop::try_from(result).ok_or_else(|| {
                    GuestError::RuntimeError("expression did not answer a ByteArray".to_string())
                })?;
                let mut bytes = Vec::new();
                b.copy_bytes_out(&mut bytes);
                Ok(bytes)
            }
            Ok(None) => Ok(Vec::new()),
            Err(e) => Err(GuestError::Compile(e)),
        }
    }

    /// Installs `sink` as where guest output (`Transcript show:`,
    /// `printOnStdout:`) goes from now on. Default is stdout
    /// (`VmState::with_options`) â€” an embedder calls this once, right after
    /// `boot`, before the first `eval`.
    pub fn set_transcript(&mut self, sink: Box<dyn TranscriptSink>) {
        self.vm.out = Box::new(SinkWriter(sink));
    }

    /// Installs `sink` as where game-primitive commands go
    /// (`docs/gamepane_design.md` M3) â€” the game analogue of `set_transcript`.
    /// Default is `None` (a headless VM silently drops game commands); the GUI
    /// installs a channel-backed sink once, right after `boot`.
    pub fn set_game_sink(&mut self, sink: Box<dyn GameSink>) {
        self.vm.game_sink = Some(sink);
    }

    /// DBG4 (docs/gui_debugger_design.md): install the GUI debugger frontend
    /// on THIS vm â€” the halt loop then publishes reports to it and reads
    /// commands from it instead of the stdin `(halt)` REPL. Also arms
    /// `debug.active` (the halt primitive / breakpoints / stepping master
    /// switch) and defaults halt-on-error ON. The Cocoa supervisor installs
    /// this on the PRIMARY only â€” a UI-worker halt would park the main thread.
    pub fn set_debug_frontend(
        &mut self,
        frontend: std::sync::Arc<dyn crate::runtime::debug::DebugFrontend>,
    ) {
        self.vm.debug.frontend = Some(frontend);
        self.vm.debug.active = true;
        self.vm.debug.halt_on_error = true;
    }

    /// The Debug â–¸ Halt on Error toggle (only meaningful with a frontend).
    pub fn set_halt_on_error(&mut self, on: bool) {
        self.vm.debug.halt_on_error = on;
    }

    /// DBG4 "Break on entry": plant a breakpoint at `class>>selector` bci 0
    /// (`class` may end in " class" for the metaclass side). Pins the method
    /// to tier-0 so the dispatch hook fires. Runs on the primary's own thread
    /// (the supervisor pump services the GUI's request). `Ok(msg)`/`Err(why)`.
    pub fn set_breakpoint_by_name(&mut self, class: &str, selector: &str) -> Result<String, String> {
        self.vm.debug.active = true;
        crate::runtime::debug::set_breakpoint_by_name(&mut self.vm, class, selector, 0)
    }

    /// The clearing twin of [`set_breakpoint_by_name`].
    pub fn clear_breakpoint_by_name(&mut self, class: &str, selector: &str) -> Result<String, String> {
        crate::runtime::debug::clear_breakpoint_by_name(&mut self.vm, class, selector, 0)
    }

    /// Registers how THIS vm spawns worker VMs (docs/multi-smalltalk-worker.md
    /// Â§3, workers M1) â€” the `GameSink` pattern: the CLI/tests pass a
    /// `VmHandle::boot(opts, world_dir)` closure, the GUI its image-boot path,
    /// so a worker's world matches the primary's. Installing the boot fn is
    /// what makes this VM the PRIMARY (creates its inbox + registry); without
    /// it, `Worker spawn` fails cleanly. The closure runs ON each new worker
    /// thread.
    pub fn set_worker_boot(&mut self, f: crate::runtime::workers::WorkerBootFn) {
        self.vm.workers = Some(Box::new(crate::runtime::workers::WorkerState::new_primary(
            f,
        )));
    }

    /// Registers the router's wake hook (Â§3.1): fired â€” coalesced â€” whenever
    /// a worker envelope lands in this (primary) VM's inbox, so a sleeping
    /// host can submit a `Worker dispatchInbox.` doit. Headless embeddings
    /// skip this and sleep in `Worker runLoopWhile:` instead (the channel
    /// wakeup IS the router there). Call after `set_worker_boot`.
    pub fn set_inbox_wake(&mut self, f: crate::runtime::workers::InboxWakeFn) {
        if let Some(ws) = self.vm.workers.as_ref() {
            ws.set_wake(f);
        }
    }

    /// Register an *externally-hosted* worker (CG1, `cocoa_gui_design.md` Â§3
    /// step 3) on THIS primary VM â€” the surface the Cocoa GUI's boot handshake
    /// needs from outside the crate. Delegates to
    /// [`crate::runtime::workers::register_hosted_worker`]: mints the same
    /// registry entry `Worker spawn` does (a normal-numbered link so
    /// `send:`/`alive`/`terminate` target it with no special-casing) but hands
    /// back the receiving side â€” `(id, HostedInbox, InboxSender)` â€” so the
    /// caller can drive its own drain loop instead of a spawned thread's
    /// recv-loop. `wake` is the caller's run-loop poke, fired (coalesced) on
    /// every `send` to this worker. `None` if this VM is not a primary (call
    /// [`set_worker_boot`](Self::set_worker_boot) first) or the fleet is at its
    /// cap. See the CG1 gate `hosted_worker_registered_on_this_thread_round_trips`.
    pub fn register_hosted_worker(
        &mut self,
        wake: crate::runtime::workers::InboxWakeFn,
    ) -> Option<(
        u32,
        crate::runtime::workers::HostedInbox,
        crate::runtime::workers::InboxSender,
    )> {
        crate::runtime::workers::register_hosted_worker(&mut self.vm, wake)
    }

    /// Send `bytes` (a MOP pickle, or empty for a bare connectivity poke) to
    /// worker `id` from THIS primary VM, correlated by `corr` (0 =
    /// uncorrelated) â€” the public face of [`crate::runtime::workers::send`], so
    /// the Cocoa GUI's watchdog thread can drive the primaryâ†’UI-worker link
    /// (initial snapshot blasts, CG4). Fires the worker's (coalesced) run-loop
    /// wake. `false` if there is no such live worker.
    pub fn send_to_worker(&mut self, id: u32, corr: u64, bytes: Vec<u8>) -> bool {
        crate::runtime::workers::send(&mut self.vm, id, corr, bytes)
    }

    /// Take on the Worker role (called by the worker thread body right after
    /// boot, before any guest code): this VM is worker `self_id`, replying to
    /// the primary through `to_primary`. Also called from OUTSIDE the crate by
    /// the Cocoa GUI's boot handshake (CG2): the UI worker is booted in place
    /// on main, then takes on its Worker role so its future `reply:`/`send:`
    /// reach the primary â€” the same wiring a spawned `worker_main` does, driven
    /// by the run loop instead of a recv loop.
    pub fn install_worker_role(
        &mut self,
        self_id: u32,
        to_primary: crate::runtime::workers::InboxSender,
    ) {
        // Guard the public-API sharp edge: this overwrites `workers`, so calling
        // it on a VM that has already become the Primary would silently drop its
        // registry of spawned-worker links. The role is a boot-time, once-per-VM
        // decision (a VM is EITHER the primary OR a worker); a re-role is a
        // caller bug. Correct callers: the worker thread body, and the Cocoa
        // GUI's boot handshake for the UI worker (never the primary).
        debug_assert!(
            !matches!(
                self.vm.workers.as_deref(),
                Some(crate::runtime::workers::WorkerState::Primary { .. })
            ),
            "install_worker_role would clobber a live Primary's worker registry"
        );
        self.vm.workers = Some(Box::new(crate::runtime::workers::WorkerState::new_worker(
            self_id, to_primary,
        )));
    }

    /// Load an EXTRA world list on top of the already-booted base world (CG1,
    /// `cocoa_gui_design.md` Â§12.3) â€” the public face of
    /// [`crate::frontend::world::load_list`]. The Cocoa GUI's UI worker calls
    /// this once after [`boot`](Self::boot) to layer `world/cocoaui.list` (the
    /// `CocoaUI` view classes, files 63+) that the CLI, the WKWebView GUI, and
    /// the base test suite carry none of. Paths in the list are relative to the
    /// list file's own directory. A load error (bad path, compile error) is
    /// returned as [`VmError`]; unlike [`eval`](Self::eval)/[`exec`](Self::exec)
    /// this is not signal-guarded (it mirrors `boot`'s own unguarded
    /// `load_world`, run before the run loop is live).
    pub fn load_list(&mut self, path: &Path) -> Result<(), VmError> {
        frontend::world::load_list(&mut self.vm, path).map_err(|e| VmError { msg: e.to_string() })
    }

    /// Run a `.mst` program file to completion â€” every top-level item in
    /// order, exactly as the bare `macvm run <file>` CLI does
    /// (`frontend::world::load_file`). The file-run analog of [`load_list`]
    /// (which loads a *world*); an embedder that wants CLI-style "boot, then
    /// run this program" reaches for this. A compile / uncaught-guest error
    /// surfaces as [`VmError`]. Any exit code the program set is then read via
    /// [`exit_code`](Self::exit_code).
    pub fn run_file(&mut self, path: &Path) -> Result<(), VmError> {
        frontend::world::load_file(&mut self.vm, path).map_err(|e| VmError { msg: e.to_string() })
    }

    /// The process exit code a program requested (`Smalltalk exit:` / the
    /// SPEC exit primitive), or `None` if it never set one â€” the caller
    /// (a CLI `run`) propagates it, matching bare `macvm run`'s
    /// `std::process::exit(vm.exit_code.unwrap_or(0))`.
    pub fn exit_code(&self) -> Option<i32> {
        self.vm.exit_code
    }

    /// Flip THIS (primary) VM's transcript so its output is forwarded to the UI
    /// worker's inbox as `{#workerTranscript. 0. text}` envelopes (Cocoa GUI
    /// CG4, `cocoa_gui_design.md` Â§7.4) â€” the exact `ForwardTranscript` machinery
    /// M2 uses workerâ†’primary, direction-flipped and UNtagged (the primary is
    /// the environment's own console, not a sub-worker). The UI worker's
    /// `dispatchOne:` shows each line on its Transcript view. `ui_id` is the UI
    /// worker's id in this primary's registry (from [`register_hosted_worker`]);
    /// a no-op if there is no such live worker. Call after registering the UI
    /// worker, re-called on each respawn (Â§5.1).
    pub fn forward_transcript_to_ui(&mut self, ui_id: u32) {
        if let Some(dest) = crate::runtime::workers::worker_inbox_sender(&self.vm, ui_id) {
            self.set_transcript(Box::new(crate::runtime::workers::ForwardTranscript::to(
                0, dest,
            )));
        }
    }

    /// Drain one inbound envelope into THIS (hosted UI worker) VM and route it â€”
    /// the public face of the `stage_pending` + `exec("Worker dispatchInbox.")`
    /// pair (Cocoa GUI CG4): the main-thread run-loop drain source calls this per
    /// envelope it pulls off the [`crate::runtime::workers::HostedInbox`]. The UI
    /// worker routes via `dispatchInbox` â†’ `dispatchOne:` (NOT `dispatchPending`)
    /// so a `#uiReply` fires its pending continuation and a `{#workerTranscript.
    /// 0. text}` reaches the Transcript view. Errors surface as [`GuestError`]
    /// (the caller reports; the UI worker's `ErrorPolicy::Resume` keeps it alive).
    pub fn dispatch_hosted_envelope(
        &mut self,
        env: crate::runtime::workers::Envelope,
    ) -> Result<(), GuestError> {
        self.stage_pending(env);
        self.exec("Worker dispatchInbox.")
    }

    /// Park an inbound envelope in the Worker-role staging slot (the
    /// `GameStep` pattern): the host loop calls this, then execs
    /// `Worker dispatchPending.`, whose `primPoll` takes it. Rust bytes only
    /// â€” nothing here is visible to the GC.
    pub(crate) fn stage_pending(&mut self, env: crate::runtime::workers::Envelope) {
        if let Some(ws) = self.vm.workers.as_mut() {
            if let crate::runtime::workers::WorkerState::Worker { pending, .. } = &mut **ws {
                *pending = Some(env);
            }
        }
    }

    /// Hand a monitor a clone of THIS VM's live-signal block (a per-VM `Arc`, no
    /// global) so it can sample `compiled_depth` at high frequency off-thread,
    /// without a request round-trip â€” the basis of the interpreter/compiler
    /// ratio. Safe to call once at boot and keep.
    pub fn live_stats(&self) -> std::sync::Arc<VmLiveStats> {
        self.vm.live_stats.clone()
    }

    /// Snapshot this VM's slower runtime counters for the metrics dashboard.
    /// A cheap read of existing fields â€” no allocation, no GC. Runs on the
    /// worker thread (the VM's owner).
    pub fn metrics(&self) -> VmMetrics {
        let vm = &self.vm;
        let u = &vm.universe;
        let (code_lo, code_hi) = vm.code_cache.bounds();
        VmMetrics {
            eden_used: (u.eden.top - u.eden.start) as u64,
            eden_capacity: (u.eden.end - u.eden.start) as u64,
            old_used: (u.old.top - u.old.bounds.start) as u64,
            old_committed: (u.old.committed_end - u.old.bounds.start) as u64,
            old_reserved: (u.old.bounds.end - u.old.bounds.start) as u64,
            scavenges: u.gc_stats.scavenge_count,
            full_gcs: u.gc_stats.full_gc_count,
            bytes_allocated: u.gc_stats.bytes_allocated,
            last_reclaimed: u.gc_stats.last_reclaimed_bytes,
            nmethods: vm.code_table.iter_alive().count() as u64,
            code_used: vm.code_cache.used_bytes() as u64,
            code_capacity: code_hi.saturating_sub(code_lo),
            compilations: vm.stats.compilations,
            deopts: vm.stats.deopt_count,
            osr_entries: vm.stats.osr_entries,
            ic_misses: vm.stats.ic_misses,
        }
    }
}

/// `main.rs::print_result`'s exact logic (run `printString`, fall back to
/// the Rust formatter for a pre-S6 world), duplicated rather than shared
/// because `main.rs`'s copy is a private `fn` in a binary crate `embed.rs`
/// cannot depend on.
fn print_result(vm: &mut VmState, result: Oop) -> String {
    let klass = crate::runtime::lookup::klass_of(vm, result);
    let sel = vm.universe.intern(b"printString");
    if let Some(m) = crate::runtime::lookup::lookup(vm, klass, sel) {
        let s = crate::interpreter::run_method(vm, m, result, &[]);
        if let Some(b) = crate::oops::wrappers::ByteArrayOop::try_from(s) {
            let mut bytes = Vec::new();
            b.copy_bytes_out(&mut bytes);
            return String::from_utf8_lossy(&bytes).into_owned();
        }
    }
    crate::memory::print_oop(&vm.universe, result)
}

/// Raw bytes of a guest `String`/`ByteArray` result, or `None` if `result`
/// isn't byte-indexable. Used by [`VmHandle::render_fragment`] to return an
/// HTML fragment verbatim, without the printString requoting `print_result`
/// would apply.
fn fragment_bytes(result: Oop) -> Option<String> {
    let b = crate::oops::wrappers::ByteArrayOop::try_from(result)?;
    let mut bytes = Vec::new();
    b.copy_bytes_out(&mut bytes);
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::JitMode;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn boot_test_vm(jit: JitMode) -> VmHandle {
        VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                jit,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("boot against the real world/ directory must succeed")
    }

    #[test]
    fn eval_arithmetic_returns_printstring() {
        let mut vm = boot_test_vm(JitMode::Off);
        let result = vm.eval("3 + 4.").expect("3 + 4 must evaluate cleanly");
        assert_eq!(result, "7");
    }

    /// A bare Do-it (no terminating period) must evaluate, not fail with
    /// "expected '.' after statement". This is how every GUI doit arrives â€”
    /// the tour's `doit="Mandelbrot new launch"` and a Workspace "Do it" on a
    /// selected expression both lack a trailing period.
    #[test]
    fn eval_tolerates_a_missing_trailing_period() {
        let mut vm = boot_test_vm(JitMode::Off);
        assert_eq!(
            vm.eval("3 + 4")
                .expect("a bare expression must evaluate without a trailing period"),
            "7"
        );
        // A trailing period still works (unchanged).
        assert_eq!(vm.eval("10 * 2.").expect("with period still fine"), "20");
        // Genuine trailing garbage after a complete statement is still an error.
        assert!(
            vm.eval("3 + 4  5").is_err(),
            "two statements without a separating period must still be rejected"
        );
    }

    /// "the JIT MUST be supported" (the S21 directive this whole module
    /// exists to satisfy) â€” `boot`/`eval` place no restriction on
    /// `opts.jit` at all. `Threshold(1)` compiles on the very first call,
    /// exercising the compiled path immediately rather than needing a hot
    /// loop to cross a higher threshold.
    #[test]
    fn eval_works_with_jit_enabled_and_aggressive_threshold() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        let result = vm.eval("6 * 7.").expect("6 * 7 must evaluate cleanly");
        assert_eq!(result, "42");
    }

    /// G2 smappl slice: a `visual=` labeled-button expression renders to a
    /// live beveled `<button>` fragment (image-side, per D-G5) whose
    /// `data-widget-action` id fires the stored action closure on click.
    #[test]
    fn render_fragment_builds_a_button_and_fires_its_action() {
        struct VecSink(Arc<Mutex<Vec<String>>>);
        impl TranscriptSink for VecSink {
            fn show(&mut self, text: &str) {
                self.0.lock().unwrap().push(text.to_string());
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_transcript(Box::new(VecSink(captured.clone())));

        let html = vm
            .render_fragment(
                "Button labeled: 'Press Me!' action: [ :b | Transcript show: 'clicked' ]",
            )
            .expect("a labeled button must render to an HTML fragment");
        assert!(
            html.contains("smappl-button") && html.contains("Press Me!"),
            "fragment must be a beveled button carrying its label, got {html:?}"
        );
        // The id the fragment advertises is what a click posts back.
        let id_start = html
            .find("data-widget-action=\"")
            .expect("fragment has an action id")
            + "data-widget-action=\"".len();
        let id_end = html[id_start..].find('"').unwrap() + id_start;
        let id = &html[id_start..id_end];

        vm.exec(&format!("SmapplRegistry fire: '{id}'."))
            .expect("firing the registered widget id must run its action");
        let lines = captured.lock().unwrap();
        assert!(
            lines.iter().any(|l| l.contains("clicked")),
            "firing the button's id must run its action closure, got {lines:?}"
        );
    }

    /// The differences2.html "Press Me!" demo: a labeled button whose action
    /// is `b promptOk:title:type:action:`. Firing it must answer the modal
    /// dialog's HTML (a String) â€” `fire_widget_action` surfaces that as the
    /// overlay to float; a pure side-effect action instead answers `None`.
    #[test]
    fn firing_a_promptok_action_yields_the_dialog_overlay_html() {
        let mut vm = boot_test_vm(JitMode::Off);

        // The exact corpus shape (differences2.html), collapsed to one line.
        let html = vm
            .render_fragment(
                "Button labeled: 'Press Me!' action: [ :b | \
                 b promptOk: 'The UI can do native things easily.' \
                 title: 'It works!' type: #info action: [] ]",
            )
            .expect("the Press Me button must render");
        let id_start = html
            .find("data-widget-action=\"")
            .expect("has an action id")
            + "data-widget-action=\"".len();
        let id_end = html[id_start..].find('"').unwrap() + id_start;
        let id = html[id_start..id_end].to_string();

        let overlay = vm
            .fire_widget_action(&id)
            .expect("firing must succeed")
            .expect("a promptOk action must answer dialog HTML, not None");
        assert!(
            overlay.contains("st-modal")
                && overlay.contains("st-modal-info")
                && overlay.contains("It works!")
                && overlay.contains("The UI can do native things easily.")
                && overlay.contains("st-modal-ok"),
            "overlay must be an info modal carrying title, message and an OK button, got {overlay:?}"
        );

        // A side-effect-only action (no String answer) yields no overlay.
        let side = vm
            .render_fragment("Button labeled: 'x' action: [ :b | 1 + 1 ]")
            .expect("button renders");
        let s0 = side.find("data-widget-action=\"").unwrap() + "data-widget-action=\"".len();
        let s1 = side[s0..].find('"').unwrap() + s0;
        let sid = side[s0..s1].to_string();
        assert_eq!(
            vm.fire_widget_action(&sid).expect("firing must succeed"),
            None,
            "a non-String action result must not produce an overlay"
        );
    }

    /// The Canvas Mandelbrot demo (`world/35_mandelbrot.mst`,
    /// `docs/CANVAS.md` pixel path): `Mandelbrot new pixelsForWidth:height:`
    /// computes the set in real Smalltalk `Double` arithmetic and fills a
    /// `Pixmap`, answering its raw `w*h*4` RGBA `ByteArray`. The buffer must be
    /// exactly the right size, fully opaque, and contain both interior (black)
    /// and escaped (coloured) pixels â€” i.e. the float compute really
    /// discriminated points, not painted one flat colour.
    #[test]
    fn mandelbrot_fills_an_rgba_pixmap() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        let (w, h) = (120usize, 90usize);
        let bytes = vm
            .eval_to_bytes(&format!("Mandelbrot new pixelsForWidth: {w} height: {h}"))
            .expect("Mandelbrot must answer an RGBA ByteArray");

        assert_eq!(bytes.len(), w * h * 4, "buffer must be exactly w*h*4 RGBA");
        // Every alpha byte (every 4th) must be fully opaque.
        assert!(
            bytes.iter().skip(3).step_by(4).all(|&a| a == 255),
            "every pixel must be opaque (alpha 255)"
        );
        // A black interior pixel exists (some point reached maxIter)...
        let has_black = bytes.chunks(4).any(|p| p[0] == 0 && p[1] == 0 && p[2] == 0);
        // ...and a non-black escaped pixel exists (the bands).
        let has_colour = bytes.chunks(4).any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0);
        assert!(
            has_black && has_colour,
            "the fractal must have both interior (black) and escaped (coloured) pixels"
        );
    }

    /// Float fast-path deopt-sunk boxing (`docs/float_fastpath_design.md`):
    /// an intermediate `FBox` pinned only by a later fused op's reexecute
    /// stack moves into that op's fail block. If the later op deopts (a
    /// non-Double ARG â€” the IC only guards receivers), the interpreter
    /// re-executes the send and must see the CORRECT boxed intermediate,
    /// built by the sunk box on the cold path. Wrong bits here would be a
    /// silent wrong answer, so this pins the exact fallback values.
    #[test]
    fn float_fuse_deopt_reboxes_sunk_intermediates_correctly() {
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.exec(
            "Object subclass: FSinkT [ mix: a with: b plus: c [ ^a * b + c ] \
             chain: a with: b [ ^2.0 * a * b + 0.5 ] ]",
        )
        .expect("class definition");
        // Warm the fused sites mono-Double, past the threshold.
        vm.exec(
            "[ | f | f := FSinkT new. 1 to: 500 do: [ :i | \
             f mix: 3.5 with: 2.0 plus: 1.25. f chain: 1.5 with: 0.25 ] ] value.",
        )
        .expect("warmup");
        // The + arg-unbox fails: reexec needs the sunk box of 3.5*2.0 = 7.0.
        assert_eq!(
            vm.eval("FSinkT new mix: 3.5 with: 2.0 plus: 1")
                .expect("mixed"),
            "8.0"
        );
        // Mid-chain deopt: the second * fails; its reexec stack holds the
        // sunk box of 2.0*1.5 = 3.0 â†’ 3.0*2 = 6.0 (int fallback) + 0.5.
        assert_eq!(
            vm.eval("FSinkT new chain: 1.5 with: 2").expect("chained"),
            "6.5"
        );
        // And the pure-Double path still answers exactly.
        assert_eq!(
            vm.eval("FSinkT new mix: 3.5 with: 2.0 plus: 1.25")
                .expect("pure"),
            "8.25"
        );
    }

    /// Float fast-path step 5b (`docs/float_fastpath_design.md` B5): PROMOTED
    /// float temps live as raw f64 frame slots across safepoints; the deopt
    /// map records `ValueLoc::DoubleSlot` and the materializer boxes them.
    /// This test forces an uncommon trap IN THE MIDDLE of a loop whose float
    /// temps are promoted (a fused site warmed mono-Double, then handed an
    /// integer receiver at iteration 500): the materializer must rebuild
    /// `a`/`b` exactly from their raw slots, and the remaining ~500
    /// iterations run interpreted on the rebuilt frame. Wrong bits anywhere
    /// would change the final sum. Asserts a deopt actually fired, so the
    /// path can never silently stop being exercised.
    #[test]
    fn float_temp_promotion_materializes_raw_slots_on_a_midloop_deopt() {
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.exec(
            "Object subclass: WTestT [ run: coll [ | a b i x | \
             a := 1.5. b := 0.25. i := 0. x := 0.0. \
             [ i < 1000 ] whileTrue: [ \
                 a := a * 1.001. b := b + a. \
                 x := (coll at: i + 1) + 2.0. i := i + 1 ]. \
             ^b + x ] ]",
        )
        .expect("class definition");
        vm.exec(
            "[ | w good | w := WTestT new. good := Array new: 1000. \
             1 to: 1000 do: [ :k | good at: k put: 0.5 ]. \
             1 to: 30 do: [ :k | w run: good ] ] value.",
        )
        .expect("warmup");
        let clean = vm
            .eval("WTestT new run: ((1 to: 1000) inject: (Array new: 1000) into: [ :a :k | a at: k put: 0.5. a ])")
            .unwrap_or_default();
        let deopts_before = vm.vm.stats.deopt_count;
        // Poison element 501: iteration 500's fused `+ 2.0` receiver is an
        // integer â†’ trap mid-loop with promoted a/b live.
        let poisoned = vm
            .eval(
                "[ | bad | bad := Array new: 1000. \
                 1 to: 1000 do: [ :k | bad at: k put: 0.5 ]. \
                 bad at: 501 put: 7. \
                 WTestT new run: bad ] value.",
            )
            .expect("poisoned run");
        assert!(
            vm.vm.stats.deopt_count > deopts_before,
            "the poisoned run must actually deopt mid-loop (else this test \
             is not exercising DoubleSlot materialization)"
        );
        // Interpreter truth for the poisoned input: 0.5+2.0 everywhere except
        // element 501 (7 + 2.0 = 9.0 â€” but x is overwritten each iteration,
        // so only the LAST element's x survives; the sum differs from `clean`
        // only through the b accumulation being identical and x identical) â€”
        // compare against the same expression run fully interpreted instead
        // of hand-computing.
        let mut interp = boot_test_vm(JitMode::Off);
        interp
            .exec(
                "Object subclass: WTestT [ run: coll [ | a b i x | \
                 a := 1.5. b := 0.25. i := 0. x := 0.0. \
                 [ i < 1000 ] whileTrue: [ \
                     a := a * 1.001. b := b + a. \
                     x := (coll at: i + 1) + 2.0. i := i + 1 ]. \
                 ^b + x ] ]",
            )
            .expect("class definition (interp)");
        let interp_poisoned = interp
            .eval(
                "[ | bad | bad := Array new: 1000. \
                 1 to: 1000 do: [ :k | bad at: k put: 0.5 ]. \
                 bad at: 501 put: 7. \
                 WTestT new run: bad ] value.",
            )
            .expect("poisoned run (interp)");
        assert_eq!(
            poisoned, interp_poisoned,
            "mid-loop deopt with promoted float temps must be byte-identical \
             to the interpreter"
        );
        assert!(!clean.is_empty(), "clean run sanity");
    }

    /// A `visual=` that returns a `Glue` spacer (the side-effecting shape,
    /// gui/smappl.md Â§3 shape 6) renders to an invisible fixed-width span.
    #[test]
    fn render_fragment_glue_is_an_invisible_spacer() {
        let mut vm = boot_test_vm(JitMode::Off);
        let html = vm
            .render_fragment("Glue xRigid: 12")
            .expect("Glue must render to a fragment");
        assert!(
            html.contains("class=\"glue\"") && html.contains("width:12px"),
            "Glue must render as a 12px spacer, got {html:?}"
        );
    }

    /// Phase-W first tool: the start page's own smappl
    /// (`ClassHierarchyOutliner imbeddedVisualForClass: Object`) renders to a
    /// real class-hierarchy tree â€” the `allClasses` reflection primitive â†’
    /// `ClassMirror` subclass sweep â†’ `HtmlWriter` fragment path, end to end.
    #[test]
    fn render_fragment_class_hierarchy_outliner() {
        let mut vm = boot_test_vm(JitMode::Off);
        let html = vm
            .render_fragment("ClassHierarchyOutliner imbeddedVisualForClass: Object")
            .expect("the hierarchy outliner must render");
        assert!(
            html.contains("st-outliner") && html.contains("Object"),
            "must be an outliner tree rooted at Object, got {html:?}"
        );
        // Real subclasses computed from the allClasses sweep must appear â€”
        // Behavior and Magnitude are both direct or indirect subclasses of
        // Object in the seed world.
        assert!(
            html.contains("Behavior") && html.contains("Magnitude"),
            "the tree must include Object's subclasses, got {html:?}"
        );
        // Collapsible structure: a toggle glyph, nested children containers,
        // and descendants collapsed by default (only the root is open).
        assert!(
            html.contains("st-tw") && html.contains("st-children"),
            "nodes must be collapsible (toggle glyph + children container), got {html:?}"
        );
        assert!(
            html.contains("style=\"display:none\""),
            "descendant subtrees must start collapsed, got {html:?}"
        );
    }

    /// progenv2.html's filtered-hierarchy visual is multi-statement with temp
    /// declarations (`| h v | h := (ClassHierarchyOutliner for: â€¦) filterOnâ€¦;
    /// orSubclasses. v := (h topVisualWithHRule: false) withBorder: â€¦. v`).
    /// The `[â€¦] value` wrapper must let it render (a live, unfiltered outliner)
    /// rather than trip a parse error on the leading `| h v |`.
    #[test]
    fn render_fragment_handles_a_multi_statement_visual_with_temps() {
        let mut vm = boot_test_vm(JitMode::Off);
        let html = vm
            .render_fragment(
                "| h v | \
                 h := (ClassHierarchyOutliner for: (ClassMirror on: Object)) \
                     filterOnCommentsContaining: '%HTML'; orSubclasses. \
                 v := (h topVisualWithHRule: false) \
                     withBorder: (Border standard3DRaised: true). \
                 v",
            )
            .expect("a multi-statement visual with temps must render, not parse-fail");
        assert!(
            html.contains("st-outliner") && html.contains("data-hierarchy-root=\"Object\""),
            "the cascade+temps body must yield the live Object outliner, got {html:?}"
        );
    }

    /// Phase-W method nodes: `ClassOutliner for: (ClassMirror on: Point)`
    /// renders the class's own instance- and class-side selectors (the
    /// `selectorsOf:` R2 primitive â†’ sorted selector leaves), including the
    /// full corpus decoration chain (`topVisualWithHRule:`/`withBorder:`/
    /// `Border standard3DRaised:`, all identity for HTML).
    #[test]
    fn render_fragment_class_outliner_lists_selectors() {
        let mut vm = boot_test_vm(JitMode::Off);
        // The corpus decoration chain is two sends (note the inner parens):
        // `(x topVisualWithHRule: false) withBorder: (...)` â€” gui/smappl.md Â§3.4.
        let html = vm
            .render_fragment(
                "((ClassOutliner for: (ClassMirror on: Point)) topVisualWithHRule: false) \
                 withBorder: (Border standard3DRaised: true)",
            )
            .expect("the class outliner must render");
        assert!(
            html.contains("st-classoutliner") && html.contains("Point"),
            "must be a class outliner for Point, got {html:?}"
        );
        assert!(
            html.contains("instance side") && html.contains("class side"),
            "must show both sides, got {html:?}"
        );
        // Point's own instance selectors (x, +, printOn:) and a class-side one
        // (origin) must appear.
        assert!(
            html.contains(">x<") || html.contains("st-selector\">x"),
            "instance selectors must be listed, got {html:?}"
        );
        assert!(
            html.contains("printOn:") && html.contains("origin"),
            "got {html:?}"
        );
    }

    /// The `selectorsOf:` primitive answers a class's own selectors and fails
    /// gracefully on a non-behavior.
    #[test]
    fn selectors_of_primitive() {
        let mut vm = boot_test_vm(JitMode::Off);
        let n = vm
            .eval("(ClassMirror selectorsOf: Point) size.")
            .expect("selectorsOf: must run")
            .parse::<i64>()
            .expect("size is an integer");
        assert!(n > 5, "Point defines many instance selectors, got {n}");
        // A non-behavior fails the primitive and hits the method's fallback
        // body (an empty Array) rather than erroring.
        let empty = vm
            .eval("(ClassMirror selectorsOf: 3) size.")
            .expect("selectorsOf: on a non-behavior falls back cleanly");
        assert_eq!(empty, "0", "a non-behavior has no selectors");
    }

    /// `primitiveOf:selector:` (R2) distinguishes a primitive method (VM code,
    /// shown read-only in the browser) from an ordinary Smalltalk one â€” and
    /// the ClassOutliner renders the two differently.
    #[test]
    fn primitive_methods_render_read_only() {
        let mut vm = boot_test_vm(JitMode::Off);
        // Alien>>byteAt: is a table primitive; Point>>x is ordinary Smalltalk.
        assert_ne!(
            vm.eval("ClassMirror primitiveOf: Alien selector: #byteAt:.")
                .unwrap(),
            "0",
            "byteAt: must report a primitive number"
        );
        assert_eq!(
            vm.eval("ClassMirror primitiveOf: Point selector: #x.")
                .unwrap(),
            "0",
            "an ordinary method has no primitive"
        );
        // Alien's browser shows read-only primitive notes and no editors.
        let alien = vm
            .render_fragment("ClassOutliner for: (ClassMirror on: Alien)")
            .expect("render Alien");
        assert!(
            alien.contains("st-prim-note") && !alien.contains("st-smappl-src"),
            "primitive methods must be read-only notes, not editors"
        );
        // Point's browser is all editable.
        let point = vm
            .render_fragment("ClassOutliner for: (ClassMirror on: Point)")
            .expect("render Point");
        assert!(
            point.contains("st-smappl-src") && !point.contains("st-prim-note"),
            "ordinary methods must be editable"
        );
    }

    /// A `visual=` shape that doesn't resolve to a real widget class surfaces
    /// as `Err`, so the GUI falls back to the G0 placeholder box rather than
    /// breaking the page. (Originally probed `CodeView`, which was since built
    /// â€” the contract is graceful failure for ANY unbuildable shape, so this
    /// now names a class that will never exist, keeping the test independent of
    /// which widgets happen to be implemented.)
    #[test]
    fn render_fragment_unbuilt_shape_is_err_not_panic() {
        let mut vm = boot_test_vm(JitMode::Off);
        let err = vm
            .render_fragment("NoSuchWidgetClass forString")
            .expect_err("an unbuilt widget shape must fail, not render");
        match err {
            GuestError::Compile(_) | GuestError::RuntimeError(_) => {}
            other => panic!("expected Compile/RuntimeError, got {other:?}"),
        }
    }

    /// The `allClasses` reflection primitive answers a non-empty set that
    /// includes the well-known genesis classes.
    #[test]
    fn all_classes_primitive_enumerates_the_world() {
        let mut vm = boot_test_vm(JitMode::Off);
        let count = vm
            .eval("ClassMirror allClasses size.")
            .expect("allClasses must run")
            .parse::<i64>()
            .expect("size is an integer");
        assert!(count > 20, "the seed world has many classes, got {count}");
    }

    /// Regression: `dispatch_ffi_primitive` left the receiver+args on the
    /// operand stack instead of truncating to `base` like every table
    /// primitive. Masked in the interpreter (the calling method's return
    /// truncates them), but a COMPILED caller tracks the stack statically, so
    /// the divergence tripped `enter_compiled`'s sp assert
    /// (`compiled_call.rs`) â€” `Time millisecondClockValue` twice under the JIT
    /// aborted the process. Under `Threshold(1)` the second call runs the
    /// compiled `millisecondClockValue` (which calls an FFI primitive), so a
    /// clean return proves the stack is balanced.
    #[test]
    fn ffi_primitive_under_jit_keeps_the_operand_stack_balanced() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        let out = vm
            .eval("[ Time millisecondClockValue. Time millisecondClockValue ] value.")
            .expect("an FFI primitive called from a compiled method must not corrupt the stack");
        assert!(
            out.parse::<i64>().map(|n| n > 0).unwrap_or(false),
            "expected a positive epoch-ms value, got {out:?}"
        );
    }

    #[test]
    fn eval_compile_error_surfaces_as_err_not_panic() {
        let mut vm = boot_test_vm(JitMode::Off);
        // Same "two consecutive binary operators" shape as tests/it_cli.rs's
        // own run_compile_err ("a + + b.") â€” a proven-broken source in this
        // codebase's own conventions.
        let err = vm
            .eval("3 + + 4")
            .expect_err("malformed source must fail to compile");
        match err {
            GuestError::Compile(_) => {}
            other => panic!("expected GuestError::Compile, got {other:?}"),
        }
    }

    /// `ensure:`/`ifCurtailed:` must fire when the protected block ERRORS â€”
    /// not only on normal completion and non-local return. Before
    /// `unwind::run_curtailment_blocks_on_error`, an unhandled error did not
    /// unwind at all (it `siglongjmp`ed past every frame from inside
    /// `prim_error`), so `[stream do: ...] ensure: [stream close]` silently
    /// left the stream open.
    ///
    /// This lives here rather than in the world suite because asserting it
    /// needs something that can CATCH the escape: `eval`'s guest-fatal
    /// recovery turns the unhandled error into an `Err`, leaving the VM alive
    /// so the next `eval` can read back what the cleanup recorded.
    #[test]
    fn ensure_block_runs_when_the_protected_block_errors() {
        let mut vm = boot_test_vm(JitMode::Off);
        vm.eval(
            "Object subclass: CurtailProbe [ \
                 <classVars: Log> \
                 CurtailProbe class >> log [ ^Log ifNil: [ Log := OrderedCollection new ] ] \
                 CurtailProbe class >> reset [ Log := OrderedCollection new ] \
                 boom [ [ CurtailProbe log add: #body. self error: 'boom' ] \
                          ensure: [ CurtailProbe log add: #cleanup ] ] \
                 dnuBoom [ [ CurtailProbe log add: #body. nil zorkNoSuchSelector ] \
                             ensure: [ CurtailProbe log add: #cleanup ] ] \
                 curtailed [ [ self error: 'boom' ] \
                               ifCurtailed: [ CurtailProbe log add: #curtailed ] ] \
                 nested [ [[ self error: 'boom' ] ensure: [ CurtailProbe log add: #inner ]] \
                            ensure: [ CurtailProbe log add: #outer ] ] \
                 ]",
        )
        .expect("class definition");

        // 1. explicit `self error:` â€” the cleanup runs, and the error still
        //    surfaces as an ordinary recoverable Err (the VM stays alive).
        vm.eval("CurtailProbe reset").expect("reset");
        let err = vm.eval("CurtailProbe new boom").expect_err("must error");
        assert!(matches!(err, GuestError::RuntimeError(_)), "got {err:?}");
        let log = vm.eval("CurtailProbe log printString").expect("read log");
        assert!(
            log.contains("body") && log.contains("cleanup"),
            "ensure: block did not run on the error path: {log}"
        );

        // 2. an unhandled DNU takes the other fatal route (`dnu_fallback`).
        vm.eval("CurtailProbe reset").expect("reset");
        vm.eval("CurtailProbe new dnuBoom")
            .expect_err("DNU must error");
        let log = vm.eval("CurtailProbe log printString").expect("read log");
        assert!(
            log.contains("cleanup"),
            "ensure: block did not run on the DNU path: {log}"
        );

        // 3. an error curtails, so ifCurtailed: fires too.
        vm.eval("CurtailProbe reset").expect("reset");
        vm.eval("CurtailProbe new curtailed")
            .expect_err("must error");
        let log = vm.eval("CurtailProbe log printString").expect("read log");
        assert!(
            log.contains("curtailed"),
            "ifCurtailed: block did not run on the error path: {log}"
        );

        // 4. nested handlers run innermost-first.
        vm.eval("CurtailProbe reset").expect("reset");
        vm.eval("CurtailProbe new nested").expect_err("must error");
        let log = vm.eval("CurtailProbe log printString").expect("read log");
        let inner = log.find("inner").expect("inner cleanup ran");
        let outer = log.find("outer").expect("outer cleanup ran");
        assert!(inner < outer, "cleanups ran outermost-first: {log}");
    }

    #[test]
    fn set_game_sink_routes_game_commands_from_a_smalltalk_doit() {
        // The M3 vertical slice end to end: a Smalltalk doit -> GamePane
        // primitive (id 200) -> GameCommand -> the installed sink. Headless,
        // deterministic, no GPU/window â€” this is the real proof of the VM->GUI
        // game channel (docs/gamepane_design.md M3).
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));
        vm.eval("GamePane new clearR: 200 g: 40 b: 40.")
            .expect("GamePane>>clearR:g:b: must evaluate cleanly");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![GameCommand::ClearTo {
                r: 200,
                g: 40,
                b: 40
            }],
            "the game sink must capture exactly the ClearTo command"
        );
    }

    #[test]
    fn mandelzoom_renders_the_set_through_the_game_channel() {
        // The MandelZoom demo (world/45_mandelzoom.mst) really renders the
        // Mandelbrot set through the VM->GUI game channel: drive its per-frame
        // draw commands into a 320x240 palette-indexed buffer (exactly as the
        // native pane's CPU buffer would receive them) and assert the rendered
        // structure â€” a filled interior plus a many-banded exterior. Headless,
        // no GPU/window â€” same proof style as the sink test above.
        const W: usize = 320;
        const H: usize = 240;
        struct Raster(Arc<Mutex<Vec<u8>>>);
        impl GameSink for Raster {
            fn emit(&mut self, cmd: GameCommand) {
                let mut b = self.0.lock().unwrap();
                match cmd {
                    GameCommand::Cls { index } => b.iter_mut().for_each(|p| *p = index),
                    GameCommand::Pset { x, y, index } => {
                        if x >= 0 && y >= 0 && (x as usize) < W && (y as usize) < H {
                            b[y as usize * W + x as usize] = index;
                        }
                    }
                    GameCommand::Blit { data } => {
                        let n = data.len().min(b.len());
                        b[..n].copy_from_slice(&data[..n]);
                    }
                    // PaletteAt/Present/StartLoop don't affect the index buffer.
                    _ => {}
                }
            }
        }

        let buf = Arc::new(Mutex::new(vec![0u8; W * H]));
        // A low JIT threshold so the hot escape loop compiles quickly (it is
        // the JIT's strong suit); the rendered pixels are identical to the
        // interpreter's (same Double semantics).
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.set_game_sink(Box::new(Raster(buf.clone())));
        vm.exec("MandelZoom launch.")
            .expect("MandelZoom launch must run");
        // Each tick computes a whole frame and blits it, so one step suffices;
        // the first frame is at scale 3.5, so it shows the whole set.
        vm.exec("GamePane stepWithKeys: 0.")
            .expect("a game step must run");

        let pixels = buf.lock().unwrap();
        let inside = pixels.iter().filter(|&&p| p == 16).count();
        let mut seen = [false; 256];
        for &p in pixels.iter() {
            seen[p as usize] = true;
        }
        let exterior_colours = seen.iter().skip(17).filter(|&&s| s).count();

        // A recognizable set: a substantial filled interior (the cardioid and
        // bulbs have real area) and a many-banded exterior (the escape-time
        // gradient), not a monochrome fill.
        let total = (W * H) as f64;
        assert!(
            inside as f64 > total * 0.10 && (inside as f64) < total * 0.60,
            "interior (palette 16) should be a real fraction of the frame, got {inside}/{}",
            W * H
        );
        assert!(
            exterior_colours > 20,
            "exterior should show many escape bands, got {exterior_colours} distinct colours"
        );

        // Eyeball aid (visible under `--nocapture`): a downsampled ASCII view,
        // interior '#', exterior shaded by escape band.
        let ramp = [' ', '.', ':', '-', '=', '+', '*', '%'];
        let (cols, rows) = (80usize, 40usize);
        let mut art = String::from("\nMandelZoom frame @ scale 3.5 (whole set):\n");
        for ry in 0..rows {
            for rx in 0..cols {
                let p = pixels[(ry * H / rows) * W + (rx * W / cols)];
                art.push(if p == 16 {
                    '#'
                } else {
                    ramp[(p as usize) % ramp.len()]
                });
            }
            art.push('\n');
        }
        println!("{art}");
    }

    #[test]
    fn mandelvm_dives_once_then_stops_itself() {
        // MandelVM (world/46_mandelvm.mst) is MandelZoom that dives ONCE and then
        // ends â€” the standalone-window demo's "run, then exit" contract. Drive it
        // past one full dive and assert it (a) rendered real frames and (b) told
        // the host to stop (StopLoop, from `pane stop`), which is what makes the
        // `macvm-gui mandelvm` window quit itself. Also proves subclassing works
        // (MandelVM inherits MandelZoom's compute + overrides only diveBottomed).
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));
        vm.exec("MandelVM launch.")
            .expect("MandelVM launch must run cleanly");
        // One dive is ~106 frames (scale 3.5 * 0.9^n < 0.00005); drive well past
        // it. Once stopped, later steps keep re-stopping â€” harmless.
        let mut stopped_at = None;
        for frame in 0..140 {
            vm.exec("GamePane stepWithKeys: 0.")
                .expect("a game step must not error");
            if stopped_at.is_none()
                && captured
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|c| matches!(c, GameCommand::StopLoop))
            {
                stopped_at = Some(frame);
            }
        }

        let cmds = captured.lock().unwrap();
        assert!(
            cmds.iter().any(|c| matches!(c, GameCommand::Blit { .. })),
            "MandelVM must render real frames (blits)"
        );
        assert!(
            stopped_at.is_some(),
            "MandelVM must end its dive with a StopLoop (pane stop) within 140 frames"
        );
    }

    #[test]
    fn gamepane_reset_stops_the_running_demo() {
        // Escape's close path submits `GamePane reset.` (gui close_game_pane).
        // Prove the VM-side contract it relies on: reset nils the registered
        // step block, so a later frame tick runs nothing and draws nothing â€”
        // the demo leaves no state behind.
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        vm.exec("MandelZoom launch.").expect("launch must run");
        // A frame while running draws (MandelZoom blits a whole frame).
        vm.exec("GamePane stepWithKeys: 0.")
            .expect("a step must run");
        assert!(
            captured
                .lock()
                .unwrap()
                .iter()
                .any(|c| matches!(c, GameCommand::Blit { .. })),
            "a running demo's frame must draw"
        );

        // Reset (what Escape does), then tick again: nothing draws.
        vm.exec("GamePane reset.").expect("reset must run");
        captured.lock().unwrap().clear();
        vm.exec("GamePane stepWithKeys: 0.")
            .expect("a post-reset step must run");
        let after = captured.lock().unwrap();
        assert!(
            after.is_empty(),
            "after reset the step block is gone, so a tick draws nothing, got {after:?}"
        );
    }

    #[test]
    fn metrics_snapshot_reports_live_counters() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        let m0 = vm.metrics();
        assert!(m0.eden_capacity > 0, "eden must report a capacity");
        assert!(
            m0.code_capacity > 0,
            "the code cache must report a capacity"
        );
        // Run a hot looping method (so it compiles) that also allocates enough
        // to force a scavenge (so the GC byte counter moves).
        vm.exec(
            "Object subclass: MetricProbe [ \
               loop: n [ | s | s := 0. 1 to: n do: [:i | s := s + i]. ^s ] \
               churn: n [ 1 to: n do: [:i | Array new: 8] ] ].",
        )
        .expect("probe class must compile");
        for _ in 0..30 {
            vm.exec("MetricProbe new loop: 5000; churn: 3000.")
                .expect("workload must run");
        }
        let m1 = vm.metrics();
        assert!(
            m1.bytes_allocated > m0.bytes_allocated,
            "allocation must move the GC byte counter"
        );
        assert!(
            m1.nmethods > 0 && m1.compilations > 0,
            "the hot method must have compiled (nmethods={}, compilations={})",
            m1.nmethods,
            m1.compilations
        );
    }

    #[test]
    fn live_stats_lets_a_monitor_observe_compiled_execution_off_thread() {
        // The interpreter/compiler ratio depends on a monitor sampling a VM's
        // `compiled_depth` from ANOTHER thread while the VM runs. Prove it: warm
        // a method until it compiles, then sample its live_stats from a second
        // thread during a long compiled loop and confirm the sampler sees
        // `compiled_depth > 0`. The block is per-VM (an Arc), never a global.
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        vm.exec("Object subclass: MetricProbe [ loop: n [ | s | s := 0. 1 to: n do: [:i | s := s + i]. ^s ] ].")
            .expect("probe class");
        for _ in 0..30 {
            vm.exec("MetricProbe new loop: 3000.").expect("warmup");
        }
        assert!(
            vm.metrics().compilations > 0,
            "the probe loop must have compiled before we sample it"
        );

        let live = vm.live_stats();
        let stop = Arc::new(AtomicBool::new(false));
        let max_depth = Arc::new(AtomicU32::new(0));
        let sampler = {
            let (live, stop, max_depth) = (live.clone(), stop.clone(), max_depth.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let d = live.compiled_depth.load(Ordering::Relaxed);
                    max_depth.fetch_max(d, Ordering::Relaxed);
                }
            })
        };
        // A long compiled run â€” the sampler thread should catch compiled_depth>0.
        vm.exec("MetricProbe new loop: 40000000.")
            .expect("long compiled run");
        stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        assert!(
            max_depth.load(Ordering::Relaxed) > 0,
            "an off-thread monitor must observe compiled_depth > 0 during a compiled loop"
        );
    }

    #[test]
    fn class_mirror_reflects_instance_and_class_variable_names() {
        // The dynamic half of the dual placement: live VM reflection (the new
        // primitives 157/158, surfaced through ClassMirror) reports a class's
        // OWN variable names â€” what the Smalltalk outliner draws its variables
        // section from, the Rust browser drawing the same names from the image.
        let mut vm = boot_test_vm(JitMode::Off);
        let iv = vm
            .eval("ClassMirror instanceVariablesOf: MandelZoom")
            .expect("instanceVariablesOf: must eval");
        assert!(
            iv.contains("centerR") && iv.contains("scale"),
            "instance var names: {iv}"
        );
        let cv = vm
            .eval("ClassMirror classVariablesOf: Character")
            .expect("classVariablesOf: must eval");
        assert!(cv.contains("Table"), "class var names: {cv}");
        // A non-behavior argument fails the primitive -> the empty-array
        // fallback in ClassMirror, not a crash.
        let none = vm
            .eval("ClassMirror instanceVariablesOf: 42")
            .expect("a non-behavior must not crash");
        assert_eq!(
            none, "#()",
            "a non-behavior yields the empty-array fallback"
        );
    }

    #[test]
    fn game_primitive_fails_on_out_of_range_colour_and_emits_nothing() {
        // r=300 is out of 0..=255, so `smi_byte` fails, the primitive fails,
        // and the method falls through to `^self` â€” no command emitted. This
        // is the design's rule: validate at the primitive boundary before a
        // value can reach an assert!-panicking engine setter.
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));
        vm.eval("GamePane new clearR: 300 g: 0 b: 0.")
            .expect("an out-of-range colour must not crash â€” the primitive just fails");
        assert!(
            captured.lock().unwrap().is_empty(),
            "an out-of-range colour must emit no game command"
        );
    }

    #[test]
    fn game_drawing_commands_reach_the_sink_in_order() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));
        // A cascade (all messages to one `GamePane new`) â€” top-level `| temp |`
        // declarations aren't valid in the doit dialect.
        vm.eval(
            "GamePane new \
               paletteAt: 16 r: 10 g: 20 b: 30; \
               cls: 16; \
               point: 5 y: 7 color: 16; \
               line: 0 y: 0 to: 9 y: 9 color: 16; \
               fill: 1 y: 2 width: 3 height: 4 color: 16; \
               disc: 8 y: 8 radius: 2 color: 16; \
               present.",
        )
        .expect("the drawing doit must evaluate cleanly");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![
                GameCommand::PaletteAt {
                    index: 16,
                    r: 10,
                    g: 20,
                    b: 30
                },
                GameCommand::Cls { index: 16 },
                GameCommand::Pset {
                    x: 5,
                    y: 7,
                    index: 16
                },
                GameCommand::Line {
                    x0: 0,
                    y0: 0,
                    x1: 9,
                    y1: 9,
                    index: 16
                },
                GameCommand::FillRect {
                    x: 1,
                    y: 2,
                    w: 3,
                    h: 4,
                    index: 16
                },
                GameCommand::Disc {
                    cx: 8,
                    cy: 8,
                    r: 2,
                    index: 16
                },
                GameCommand::Present,
            ],
            "every drawing method must reach the sink as its command, in order"
        );
    }

    #[test]
    fn frame_loop_run_registers_a_step_block_the_gui_can_pull() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        // Registering a step block and calling `run` emits StartLoop and
        // returns immediately (no blocking loop).
        vm.eval("GamePane new onStep: [ GamePane new cls: 3 ]; run.")
            .expect("onStep:/run must evaluate cleanly");
        assert_eq!(*captured.lock().unwrap(), vec![GameCommand::StartLoop]);

        // A GUI frame tick (`GamePane stepWithKeys:`) runs the step block, so
        // its drawing reaches the sink â€” the pull the GUI timer performs.
        captured.lock().unwrap().clear();
        vm.eval("GamePane stepWithKeys: 0.")
            .expect("stepWithKeys: must run the step block");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![GameCommand::Cls { index: 3 }]
        );
    }

    #[test]
    fn frame_loop_keyheld_reads_the_tick_key_mask() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        // The step block draws cls:7 only when Left is held, cls:8 only when
        // Right is. A tick with mask 5 (bits 0=Left and 2=Up) must run cls:7
        // and not cls:8 â€” proving keyHeld: reads the mask stepWithKeys: set.
        vm.eval(
            "GamePane new onStep: [ \
               (GamePane keyHeld: GamePane keyLeft)  ifTrue: [ GamePane new cls: 7 ]. \
               (GamePane keyHeld: GamePane keyRight) ifTrue: [ GamePane new cls: 8 ] ].",
        )
        .expect("onStep: must evaluate cleanly");
        vm.eval("GamePane stepWithKeys: 5.")
            .expect("stepWithKeys: must run");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![GameCommand::Cls { index: 7 }],
            "Left held -> cls:7 only; Right not held -> no cls:8"
        );
    }

    #[test]
    fn sprite_commands_reach_the_sink_from_smalltalk() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        // defineSprite: mints id 1 and emits DefineSprite; the returned Sprite's
        // cascade emits SpriteColor then MoveSprite for that same id.
        vm.eval(
            "(GamePane new defineSprite: 'f0f/0f0/f0f') \
               colorAt: 15 r: 240 g: 240 b: 0; \
               moveTo: 100 y: 80.",
        )
        .expect("the sprite doit must evaluate cleanly");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![
                GameCommand::DefineSprite {
                    id: 1,
                    rows: "f0f/0f0/f0f".to_string()
                },
                GameCommand::SpriteColor {
                    id: 1,
                    index: 15,
                    r: 240,
                    g: 240,
                    b: 0
                },
                GameCommand::MoveSprite {
                    id: 1,
                    x: 100,
                    y: 80
                },
            ],
            "define/color/move must reach the sink for the minted sprite id"
        );
    }

    #[test]
    fn sound_play_reaches_the_sink_from_smalltalk() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        // `Sound <preset> play` reaches the sink as PlaySound{preset} â€” the
        // named presets map to 0..9. (Headless: the command, not actual audio.)
        vm.eval("Sound coin play.")
            .expect("Sound coin play must evaluate cleanly");
        vm.eval("Sound bang play.")
            .expect("Sound bang play must evaluate cleanly");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![
                GameCommand::PlaySound { preset: 0 }, // coin
                GameCommand::PlaySound { preset: 8 }, // bang
            ]
        );
    }

    #[test]
    fn tune_playonce_reaches_the_sink_from_smalltalk() {
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        vm.eval("(Tune fromAbc: 'C D E') playOnce.")
            .expect("Tune playOnce must evaluate cleanly");
        assert_eq!(
            *captured.lock().unwrap(),
            vec![GameCommand::PlayTune {
                abc: "C D E".to_string()
            }]
        );
    }

    #[test]
    fn breakout_demo_game_launches_and_steps_without_error() {
        // The whole engine end to end in one Smalltalk class: launch the game,
        // then drive 120 frames with no keys held. The ball starts at y=200
        // heading up at 3px/frame, so it reaches the brick wall (y<110) within
        // ~32 frames and hitBricks fires â€” knocking out a brick and playing a
        // blip. Driving well past that exercises the wall/paddle/brick physics
        // (all integer SmallInteger sends), so any missing world method (`//`,
        // `abs`, `min:`, `max:`, `and:`) surfaces as a DNU here, not a hang.
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));

        vm.eval("Breakout launch.")
            .expect("Breakout launch must run cleanly");
        for _ in 0..120 {
            vm.eval("GamePane stepWithKeys: 0.")
                .expect("a game step must not error");
        }

        let cmds = captured.lock().unwrap();
        assert!(
            cmds.iter().any(|c| matches!(c, GameCommand::StartLoop)),
            "launch starts the frame loop"
        );
        assert!(
            cmds.iter()
                .any(|c| matches!(c, GameCommand::FillRect { .. })),
            "bricks and the paddle draw as filled rects"
        );
        assert!(
            cmds.iter().any(|c| matches!(c, GameCommand::Disc { .. })),
            "the ball draws as a disc"
        );
        assert!(
            cmds.iter().any(|c| matches!(c, GameCommand::Present)),
            "each frame presents"
        );
        assert!(
            cmds.iter()
                .any(|c| matches!(c, GameCommand::PlaySound { .. })),
            "the ball clears a brick within 120 frames, playing a sound (runs hitBricks)"
        );
    }

    #[test]
    fn breakout_soaks_without_soft_lock_or_out_of_bounds() {
        // A long soak that would catch the two ways this physics could go wrong:
        // (1) the ball tunneling out of the field or an integer going haywire
        //     (assert every drawn ball centre stays in bounds), and (2) a
        //     soft-lock â€” the ball trapped in a cycle that clears no more bricks
        //     (assert brick-clear blips keep coming across the whole run, not
        //     just at the start). The paddle sweeps left/right in a triangle
        //     wave to simulate real play, so the board clears and resets repeat.
        struct VecGameSink(Arc<Mutex<Vec<GameCommand>>>);
        impl GameSink for VecGameSink {
            fn emit(&mut self, cmd: GameCommand) {
                self.0.lock().unwrap().push(cmd);
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_game_sink(Box::new(VecGameSink(captured.clone())));
        vm.eval("Breakout launch.")
            .expect("Breakout launch must run cleanly");

        // A triangle-wave key mask: 80 frames Right (2), then 80 frames Left (1).
        let mut sounds_in_first_half = 0usize;
        let mut sounds_in_second_half = 0usize;
        const FRAMES: usize = 4000;
        for f in 0..FRAMES {
            let mask = if (f / 80) % 2 == 0 { 2 } else { 1 };
            {
                let mut buf = captured.lock().unwrap();
                buf.clear();
            }
            vm.eval(&format!("GamePane stepWithKeys: {mask}."))
                .unwrap_or_else(|e| panic!("frame {f} must not error: {e}"));
            let buf = captured.lock().unwrap();
            for c in buf.iter() {
                if let GameCommand::Disc { cx, cy, .. } = c {
                    assert!(
                        (0..=320).contains(cx) && (0..=240).contains(cy),
                        "frame {f}: ball centre ({cx},{cy}) left the field â€” a physics escape"
                    );
                }
                if matches!(c, GameCommand::PlaySound { .. }) {
                    if f < FRAMES / 2 {
                        sounds_in_first_half += 1;
                    } else {
                        sounds_in_second_half += 1;
                    }
                }
            }
        }

        // Progress must continue throughout â€” a soft-lock would fall silent
        // after the ball got stuck, so the second half must also clear bricks.
        assert!(
            sounds_in_first_half > 5,
            "the ball should clear many bricks early ({sounds_in_first_half} sounds)"
        );
        assert!(
            sounds_in_second_half > 5,
            "the game must keep making progress in the second half â€” no soft-lock \
             ({sounds_in_second_half} sounds in frames {}..{FRAMES})",
            FRAMES / 2
        );
    }

    // â”€â”€ multi-Smalltalk workers, M1 (docs/multi-smalltalk-worker.md Â§10) â”€â”€

    /// The standard test primary: boots the real world and registers the
    /// worker boot closure (same world, same options) â€” the CLI shape.
    fn boot_worker_primary() -> VmHandle {
        let mut vm = boot_test_vm(JitMode::Off);
        vm.set_worker_boot(Arc::new(|| {
            VmHandle::boot(
                VmOptions {
                    heap_mib: 64,
                    jit: JitMode::Off,
                    ..Default::default()
                },
                Path::new("world"),
            )
        }));
        // A tiny in-language scoreboard for the async assertions: replies
        // bump Count (and Bad on a wrong value); the run loop's condition
        // bumps Tick so a broken loop bails instead of hanging the suite.
        vm.exec(
            "Object subclass: WkTest [
                <classVars: Count Bad Tick Died W1 W2 Rpc>
                WkTest class >> reset [ Count := 0. Bad := 0. Tick := 0. Died := 0. Rpc := nil ]
                WkTest class >> w1: w [ W1 := w ]
                WkTest class >> w1 [ ^W1 ]
                WkTest class >> w2: w [ W2 := w ]
                WkTest class >> w2 [ ^W2 ]
                WkTest class >> bump: ok [
                    Count := Count + 1.
                    ok ifFalse: [ Bad := Bad + 1 ] ]
                WkTest class >> noteDied [ Died := Died + 1 ]
                WkTest class >> count [ ^Count ]
                WkTest class >> bad [ ^Bad ]
                WkTest class >> died [ ^Died ]
                WkTest class >> rpc: r [ Rpc := r ]
                WkTest class >> rpc [ ^Rpc ]
                WkTest class >> tickCapped: n [ Tick := Tick + 1. ^Tick < n ]
            ]",
        )
        .expect("WkTest scoreboard must compile");
        vm.exec("WkTest reset.").expect("reset");
        vm
    }

    #[test]
    fn worker_echo_ping_pong_with_correlated_continuations() {
        // The M1 gate: a spawned worker VM echoes 200 correlated requests;
        // every reply routes to ITS OWN continuation (r = i * 2, checked
        // in-language); the primary never polls â€” it sleeps in runLoopWhile:
        // (primAwaitInbox: recv_timeout) and is woken by the sends.
        let mut vm = boot_worker_primary();
        // NB: `exec` runs ONE top item per call, so each statement is its own
        // doit; state persists in WkTest's class vars.
        vm.exec(
            "WkTest w1: (Worker spawn: 'Worker onMessage: [:m | Worker reply: m payload * 2]').",
        )
        .expect("spawn the echo worker");
        vm.exec(
            "1 to: 200 do: [:i | WkTest w1 send: i onReply: [:r | WkTest bump: r = (i * 2)] ].",
        )
        .expect("send 200 correlated requests");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 100) and: [ WkTest count < 200 ] ].")
            .expect("run the event loop until all replies land");
        assert_eq!(
            vm.eval("WkTest count").expect("count").trim(),
            "200",
            "every reply must arrive"
        );
        assert_eq!(
            vm.eval("WkTest bad").expect("bad").trim(),
            "0",
            "every reply must reach ITS OWN continuation with the right value"
        );
    }

    /// Async I/O end to end (docs/asyncio_design.md slice B): the primary spawns
    /// a dedicated IoWorker VM, watches a pipe's read end on it, writes bytes
    /// from the primary, and the data comes back as a message â€” the IoWorker did
    /// the kqueue poll + read on ITS thread while the primary only ever slept in
    /// its inbox. Proves the whole stack: FFI syscalls, cross-VM fd sharing (one
    /// process â†’ the pipe made in the primary is valid in the IoWorker), the
    /// kqueue readiness engine, and the message-driven pump loop.
    #[test]
    fn ioworker_multiplexes_a_pipe_read_back_to_the_primary() {
        let mut vm = boot_worker_primary();
        vm.exec(
            "Object subclass: IoProbe [
                <classVars: Buf ReadFd WriteFd Got>
                IoProbe class >> setUp: iow [
                    Buf := NativeBuffer page.
                    Posix pipeInto: Buf address.
                    ReadFd := Buf u32At: 0. WriteFd := Buf u32At: 4.
                    Got := nil.
                    iow watchRead: ReadFd onData: [:bytes :eof | Got := bytes ].
                    Buf byteAt: 512 put: 65. Buf byteAt: 513 put: 66. Buf byteAt: 514 put: 67.
                    Posix write: WriteFd from: Buf address + 512 count: 3.
                    iow startPumping: 50 ]
                IoProbe class >> gotSize [ ^Got isNil ifTrue: [ -1 ] ifFalse: [ Got size ] ]
                IoProbe class >> got [ ^Got ]
                IoProbe class >> sum [ | s | s := 0. Got do: [:b | s := s + b]. ^s ]
            ]",
        )
        .expect("define IoProbe");
        vm.exec("WkTest reset.").expect("reset");
        vm.exec("WkTest w1: IoWorker spawn.")
            .expect("spawn the IoWorker VM");
        vm.exec("IoProbe setUp: WkTest w1.")
            .expect("watch the pipe, write, start pumping");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ IoProbe got isNil ] ].")
            .expect("run the loop until the read comes back");
        assert_eq!(
            vm.eval("IoProbe gotSize").expect("gotSize").trim(),
            "3",
            "the 3 bytes written in the primary came back via the IoWorker"
        );
        assert_eq!(
            vm.eval("IoProbe sum").expect("sum").trim(),
            "198", // 65 + 66 + 67
            "the exact bytes (A B C) round-tripped through the kqueue read"
        );
    }

    /// The cadence (docs/asyncio_design.md): the IoWorker's pump sleeps in an
    /// INFINITE kevent â€” zero idle CPU, no heartbeat â€” and the primary wakes it
    /// by poking the kqueue's EVFILT_USER event after every non-pump send.
    /// This test starts the infinite pump FIRST (the worker goes to sleep with
    /// nothing watched) and only THEN registers the watch and writes: the watch
    /// request can only be serviced if the poke ends the sleep. If the wake
    /// were broken, the first pump would sleep forever, the watchRead envelope
    /// would never be dispatched, and this test would time out red at its cap â€”
    /// the poke is load-bearing, not an optimization. (The trigger LATCHES, so
    /// every send/sleep interleaving passes â€” no timing sleeps needed here.)
    #[test]
    fn ioworker_infinite_pump_is_woken_by_a_mid_sleep_watch() {
        let mut vm = boot_worker_primary();
        vm.exec(
            "Object subclass: IoProbe2 [
                <classVars: Buf ReadFd WriteFd Got>
                IoProbe2 class >> setUp: iow [
                    Buf := NativeBuffer page.
                    Posix pipeInto: Buf address.
                    ReadFd := Buf u32At: 0. WriteFd := Buf u32At: 4.
                    Got := nil.
                    iow startPumping.
                    iow watchRead: ReadFd onData: [:bytes :eof | Got := bytes ].
                    Buf byteAt: 512 put: 72. Buf byteAt: 513 put: 73.
                    Posix write: WriteFd from: Buf address + 512 count: 2 ]
                IoProbe2 class >> got [ ^Got ]
                IoProbe2 class >> sum [ | s | s := 0. Got do: [:b | s := s + b]. ^s ]
            ]",
        )
        .expect("define IoProbe2");
        vm.exec("WkTest reset.").expect("reset");
        vm.exec("WkTest w1: IoWorker spawn.")
            .expect("spawn the IoWorker VM");
        vm.exec("IoProbe2 setUp: WkTest w1.")
            .expect("infinite pump first, then watch + write mid-sleep");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ IoProbe2 got isNil ] ].")
            .expect("run the loop until the read comes back");
        assert_eq!(
            vm.eval("IoProbe2 sum").expect("sum").trim(),
            "145", // 72 + 73
            "bytes written AFTER an infinite pump went to sleep still arrive: \
             the EVFILT_USER poke woke the sleep so the watch got installed"
        );
    }

    /// The sockets capstone (docs/asyncio_design.md): a TCP echo server whose
    /// event loop is the IoWorker. One IoWorker multiplexes THREE fds at once â€”
    /// the listener (accept, bounded by the kevent backlog count), the
    /// server-side connection (read the request), and the client socket (read
    /// the echo) â€” all on infinite kevent sleeps, while the primary supplies
    /// the logic: its onConnection: continuation registers the data watch
    /// (a mid-sleep add, so the poke is load-bearing here too) and its onData:
    /// continuation writes the echo back DIRECTLY on the fd (legal because fds
    /// are process-wide). Loopback only, ephemeral port, no firewall prompt.
    #[test]
    fn ioworker_tcp_echo_server_round_trips() {
        let mut vm = boot_worker_primary();
        vm.exec(
            "Object subclass: EchoProbe [
                <classVars: Buf Listener Client Got>
                EchoProbe class >> setUp: iow [
                    | port |
                    Buf := NativeBuffer page.
                    Got := nil.
                    Listener := Posix tcpListenLoopback: 8.
                    port := Posix boundPortOf: Listener.
                    iow watchAccept: Listener onConnection: [:conn |
                        iow watchRead: conn onData: [:bytes :eof |
                            eof ifFalse: [ EchoProbe echo: bytes on: conn ] ] ].
                    iow startPumping.
                    Client := Posix tcpConnectLoopback: port.
                    iow watchRead: Client onData: [:bytes :eof |
                        eof ifFalse: [ Got := bytes ] ].
                    Buf byteAt: 700 put: 80. Buf byteAt: 701 put: 73.
                    Buf byteAt: 702 put: 78. Buf byteAt: 703 put: 71.
                    Posix write: Client from: Buf address + 700 count: 4 ]
                EchoProbe class >> echo: bytes on: fd [
                    1 to: bytes size do: [ :i |
                        Buf byteAt: 640 + i - 1 put: (bytes at: i) ].
                    Posix write: fd from: Buf address + 640 count: bytes size ]
                EchoProbe class >> got [ ^Got ]
                EchoProbe class >> gotSize [ ^Got isNil ifTrue: [ -1 ] ifFalse: [ Got size ] ]
                EchoProbe class >> sum [ | s | s := 0. Got do: [:b | s := s + b]. ^s ]
            ]",
        )
        .expect("define EchoProbe");
        vm.exec("WkTest reset.").expect("reset");
        vm.exec("WkTest w1: IoWorker spawn.")
            .expect("spawn the IoWorker VM");
        vm.exec("EchoProbe setUp: WkTest w1.")
            .expect("listen, watch accept, connect, send PING");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ EchoProbe got isNil ] ].")
            .expect("run the loop until the echo comes back");
        assert_eq!(
            vm.eval("EchoProbe gotSize").expect("gotSize").trim(),
            "4",
            "the 4-byte echo came back to the client watch"
        );
        assert_eq!(
            vm.eval("EchoProbe sum").expect("sum").trim(),
            "302", // P(80) + I(73) + N(78) + G(71)
            "the exact bytes PING round-tripped: client -> accepted conn -> \
             echo write -> client, every hop multiplexed by the one IoWorker"
        );
    }

    /// The slice-C capstone (docs/asyncio_design.md): the STREAM library end
    /// to end. A line-echo TCP server built entirely from the ergonomic
    /// surface â€” `TcpListener onConnection:` hands the accepted fd over as a
    /// ready-made `IoStream`, the server logic is one `eachLineDo:` +
    /// `writeLine:`, and the client reads replies with chained
    /// `nextLineDo:`s. The proof of FRAMING (the whole point of slice C over
    /// slice B's raw batches): the client sends `PING\nPONG\n` split
    /// MID-LINE across two separate writes, so the server's kevent batches
    /// cannot align with line boundaries â€” both sides must reassemble. Two
    /// intact `ECHO:`-prefixed lines back means: accept â†’ per-connection
    /// stream â†’ server-side line reassembly â†’ echo â†’ client-side line
    /// reassembly, all multiplexed by one IoWorker on infinite sleeps.
    ///
    /// The split must be DETERMINISTIC to test anything: two back-to-back
    /// client writes coalesce in the loopback socket buffer and arrive as
    /// one kevent batch (verified live â€” a broken buffer still passed).
    /// So the second write is sequenced behind a marker pipe: the marker
    /// is written inside `onConnection:` (accept time â€” the conn was not
    /// yet watched, so its data can't be in that batch), which puts 'PIN'
    /// (bounded read, kevent data=3) and the marker in the NEXT batch,
    /// and the marker's continuation issues the second write only then â€”
    /// strictly after 'PIN' was already drained alone.
    #[test]
    fn iostream_line_echo_server_reassembles_split_lines() {
        let mut vm = boot_worker_primary();
        vm.exec(
            "Object subclass: LineProbe [
                <classVars: L Client Replies Buf>
                LineProbe class >> setUp: iow [
                    Replies := OrderedCollection new.
                    Buf := NativeBuffer page.
                    Posix pipeInto: Buf address.
                    L := TcpListener on: iow backlog: 8.
                    L onConnection: [ :s |
                        s eachLineDo: [ :line | s writeLine: 'ECHO:' , line ].
                        Posix write: (Buf u32At: 4) from: Buf address + 512 count: 1 ].
                    iow watchRead: (Buf u32At: 0) onData: [ :bytes :eof |
                        Client write: 'G' , String lf , 'PONG' , String lf ].
                    iow startPumping.
                    Client := IoStream connectLoopback: L port on: iow.
                    Client nextLineDo: [ :l1 |
                        Replies add: l1.
                        Client nextLineDo: [ :l2 | Replies add: l2 ] ].
                    Client write: 'PIN' ]
                LineProbe class >> count [ ^Replies size ]
                LineProbe class >> reply: i [ ^Replies at: i ]
                LineProbe class >> teardown [ Client close. L close ]
            ]",
        )
        .expect("define LineProbe");
        vm.exec("WkTest reset.").expect("reset");
        vm.exec("WkTest w1: IoWorker spawn.")
            .expect("spawn the IoWorker VM");
        vm.exec("LineProbe setUp: WkTest w1.")
            .expect("listen, serve lines, connect, send split lines");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ LineProbe count < 2 ] ].")
            .expect("run the loop until both echoed lines land");
        assert_eq!(
            vm.eval("LineProbe count").expect("count").trim(),
            "2",
            "both echoed lines must arrive as separate framed lines"
        );
        assert_eq!(
            vm.eval("(LineProbe reply: 1) = 'ECHO:PING'")
                .expect("reply 1")
                .trim(),
            "true",
            "line one reassembled across the mid-line write split"
        );
        assert_eq!(
            vm.eval("(LineProbe reply: 2) = 'ECHO:PONG'")
                .expect("reply 2")
                .trim(),
            "true",
            "line two framed and echoed intact"
        );
        // Teardown gates the accept-side lifecycle: IoStream close
        // (unwatchRead tombstone) and TcpListener close (the new
        // unwatchAccept round-trip clearing the worker's Accepting mark â€”
        // without it a reused fd number would route into dead accepts).
        vm.exec("LineProbe teardown.")
            .expect("close the client stream and the listener cleanly");
        vm.exec("WkTest reset.").expect("fresh tick budget");
        vm.exec("Worker runLoopWhile: [ WkTest tickCapped: 20 ].")
            .expect("drain the unwatch round-trips");
    }

    #[test]
    fn perform_calls_a_method_by_name() {
        // `perform:withArguments:` (prim 64) + its arity sugar: a Symbol
        // names a method and the real method body runs â€” a primitive, an
        // interpreted, or (JIT on) a compiled one, uniformly.
        let mut vm = boot_test_vm(JitMode::Off);
        assert_eq!(vm.eval("3 perform: #+ with: 4").unwrap().trim(), "7");
        assert_eq!(vm.eval("'abcd' perform: #size").unwrap().trim(), "4");
        assert_eq!(
            vm.eval("#(10 20 30) perform: #at: with: 2").unwrap().trim(),
            "20"
        );
        assert_eq!(
            vm.eval("10 perform: #between:and: withArguments: (Array with: 5 with: 15)")
                .unwrap()
                .trim(),
            "true"
        );
        assert_eq!(
            vm.eval("5 perform: #+ withArguments: (Array with: 100)")
                .unwrap()
                .trim(),
            "105"
        );
        // A selector the receiver doesn't understand fails cleanly (the
        // world fallback raises); the VM lives on.
        assert!(
            vm.exec("3 perform: #totallyBogusSelectorXyzzy.").is_err(),
            "an unknown selector must raise, not silently no-op"
        );
        // An argument-count mismatch also fails cleanly.
        assert!(
            vm.exec("3 perform: #+ withArguments: (Array with: 1 with: 2).")
                .is_err(),
            "argc mismatch must raise"
        );
        assert_eq!(vm.eval("6 * 7").unwrap().trim(), "42");
    }

    #[test]
    fn worker_rpc_calls_a_method_by_name() {
        // The multi-VM RPC (Â§5): a worker booted with NO onMessage: handler
        // still serves RPCs from the shared world. The primary names a
        // class + selector + args; the worker resolves the class, performs
        // the method, and the (deep-copied) result returns to the
        // continuation. Target Array class>>with:with:with: â€” a shared-world
        // class-side method, so the worker (its own fresh VM/heap) has it.
        let mut vm = boot_worker_primary();
        vm.exec("WkTest w1: (Worker spawn: '').")
            .expect("spawn a bare RPC-serving worker (empty init)");
        vm.exec(
            "WkTest w1 call: #with:with:with: on: #Array args: #(10 20 30) \
             onReply: [:r | WkTest rpc: r ].",
        )
        .expect("issue the RPC");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ WkTest rpc isNil ] ].")
            .expect("await the reply");
        assert_eq!(
            vm.eval("WkTest rpc size").expect("result size").trim(),
            "3",
            "the result Array must return deep-copied"
        );
        assert_eq!(vm.eval("WkTest rpc at: 2").expect("elem").trim(), "20");
    }

    #[test]
    fn worker_rpc_unknown_class_reports_an_error() {
        // A named class the worker doesn't have: the worker replies an
        // error envelope (not a value), so the onError: branch fires and
        // the onReply: block does not â€” no crash, no hang.
        let mut vm = boot_worker_primary();
        vm.exec("WkTest w1: (Worker spawn: '').").expect("spawn");
        vm.exec(
            "WkTest w1 call: #foo on: #NoSuchClassXyzzy args: #() \
             onReply: [:r | WkTest rpc: r ] \
             onError: [:msg | WkTest bump: false ].",
        )
        .expect("issue the RPC to a missing class");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ WkTest count < 1 ] ].")
            .expect("await the error reply");
        assert_eq!(
            vm.eval("WkTest count").unwrap().trim(),
            "1",
            "the onError: branch must have fired exactly once"
        );
        assert_eq!(
            vm.eval("WkTest rpc isNil").unwrap().trim(),
            "true",
            "the onReply: value branch must NOT have fired"
        );
    }

    #[test]
    fn worker_crash_is_isolated_and_reported_as_a_message() {
        // Â§8: a worker whose handler errors dies ALONE â€” the primary gets a
        // {#workerDied. id} message through the ordinary inbox, the process
        // survives, and a sibling worker keeps answering afterwards.
        let mut vm = boot_worker_primary();
        vm.exec("WkTest w1: (Worker spawn: 'Worker onMessage: [:m | nil error: ''boom'']').")
            .expect("spawn the crasher");
        vm.exec("WkTest w2: (Worker spawn: 'Worker onMessage: [:m | Worker reply: m payload]').")
            .expect("spawn the echo sibling");
        vm.exec("Worker onReply: [:m | m isWorkerDied ifTrue: [ WkTest noteDied ] ].")
            .expect("install the reply handler");
        vm.exec("WkTest w1 send: 1.").expect("poke the crasher");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 100) and: [ WkTest died < 1 ] ].")
            .expect("run until the death notice lands");
        vm.exec("WkTest w2 send: 42 onReply: [:r | WkTest bump: r = 42 ].")
            .expect("ask the sibling");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 200) and: [ WkTest count < 1 ] ].")
            .expect("run until the sibling answers");
        assert_eq!(
            vm.eval("WkTest died").expect("died").trim(),
            "1",
            "the crash must arrive as one #workerDied message"
        );
        assert_eq!(
            vm.eval("WkTest count").expect("count").trim(),
            "1",
            "the sibling worker must still answer after the crash"
        );
    }

    #[test]
    fn worker_terminate_and_liveness() {
        let mut vm = boot_worker_primary();
        vm.exec("WkTest w1: (Worker spawn: 'Worker onMessage: [:m | Worker reply: m payload]').")
            .expect("spawn");
        vm.exec(
            "WkTest w1 isAlive ifFalse: [ nil error: 'freshly spawned worker must be alive' ].",
        )
        .expect("fresh worker is alive");
        vm.exec("WkTest w1 terminate.").expect("terminate");
        vm.exec("WkTest w1 isAlive ifTrue: [ nil error: 'terminated worker must not be alive' ].")
            .expect("terminated worker is not alive");
        // Sending to a terminated worker raises (primSend fails -> the world
        // method's error fallback), surfacing as a GuestError here.
        assert!(
            vm.exec("(Worker new setId: 1) send: 5.").is_err(),
            "send to a terminated worker must raise"
        );
    }

    #[test]
    fn parallel_mandel_computes_a_full_frame_across_worker_vms() {
        // The M4 capstone, headless: ParallelMandel fans one frame out to 4
        // worker VMs (a band each), the continuations assemble `buf`, and the
        // completed round blits. Drive the two doit streams a GUI would run â€”
        // frame ticks + inbox dispatches â€” until the blit lands, then assert
        // EVERY band really computed (no band left zero) and the image is a
        // recognizable set (filled interior + many-banded exterior), i.e. the
        // work genuinely happened in the workers.
        struct Raster(Arc<Mutex<Option<Vec<u8>>>>);
        impl GameSink for Raster {
            fn emit(&mut self, cmd: GameCommand) {
                if let GameCommand::Blit { data } = cmd {
                    *self.0.lock().unwrap() = Some(data);
                }
            }
        }
        // JIT ON both sides: a band is ~19k escape-time iterations â€” the
        // debug INTERPRETER needs ~8s+ per band (the MandelZoom test compiles
        // for the same reason); each worker's own tier-1 JIT makes it seconds.
        let mut vm = boot_test_vm(JitMode::Threshold(10));
        vm.set_worker_boot(Arc::new(|| {
            VmHandle::boot(
                VmOptions {
                    heap_mib: 64,
                    jit: JitMode::Threshold(10),
                    ..Default::default()
                },
                Path::new("world"),
            )
        }));
        let frame = Arc::new(Mutex::new(None));
        vm.set_game_sink(Box::new(Raster(frame.clone())));
        vm.exec("ParallelMandel launch.")
            .expect("ParallelMandel launch must run cleanly");
        // Interleave ticks and dispatches (the GUI's timer + WorkerInbox), with
        // a real wait for the workers' first boot+compute.
        for _ in 0..1200 {
            vm.exec("Worker dispatchInbox.").expect("dispatch");
            vm.exec("GamePane stepWithKeys: 0.").expect("tick");
            if frame.lock().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let got = frame.lock().unwrap().clone();
        let Some(pixels) = got else {
            panic!("no frame blitted â€” the parallel round never completed");
        };
        assert_eq!(pixels.len(), 320 * 240);
        // Every band computed: an unanswered band would still be all zeros
        // (palette 0 is never written by computeBand:).
        for band in 0..4 {
            let rows = &pixels[band * 60 * 320..(band + 1) * 60 * 320];
            let zeros = rows.iter().filter(|&&p| p == 0).count();
            assert!(
                zeros == 0,
                "band {band} has {zeros} unwritten pixels â€” its worker never answered"
            );
        }
        // And it is really the set (same shape checks as the MandelZoom test).
        let inside = pixels.iter().filter(|&&p| p == 16).count() as f64;
        let mut seen = [false; 256];
        for &p in &pixels {
            seen[p as usize] = true;
        }
        let exterior = seen.iter().skip(17).filter(|&&s| s).count();
        let total = (320 * 240) as f64;
        assert!(
            inside > total * 0.10 && inside < total * 0.60,
            "interior fraction off: {inside}/{total}"
        );
        assert!(exterior > 20, "too few escape bands: {exterior}");
    }

    #[test]
    fn worker_transcript_forwards_to_the_primary() {
        // M2: a worker's `Transcript show:` (its vm.out) arrives on the
        // PRIMARY's transcript, [w<id>]-tagged, through the ordinary inbox â€”
        // a worker never owns a console of its own.
        struct VecSink(Arc<Mutex<Vec<String>>>);
        impl TranscriptSink for VecSink {
            fn show(&mut self, text: &str) {
                self.0.lock().unwrap().push(text.to_string());
            }
        }
        let mut vm = boot_worker_primary();
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_transcript(Box::new(VecSink(captured.clone())));
        vm.exec(
            "WkTest w1: (Worker spawn: 'Worker onMessage: [:m | Transcript show: ''hello from the worker'']').",
        )
        .expect("spawn the printing worker");
        vm.exec("WkTest w1 send: 1.").expect("poke it");
        vm.exec("Worker runLoopWhile: [ WkTest tickCapped: 8 ].")
            .expect("run the loop a few beats");
        let lines = captured.lock().unwrap();
        assert!(
            lines.iter().any(|l| l.contains("[w1] hello from the worker")),
            "the worker's transcript line must arrive tagged on the primary's transcript, got {lines:?}"
        );
    }

    #[test]
    fn worker_spawn_cap_is_enforced() {
        let mut vm = boot_worker_primary();
        vm.exec("1 to: 16 do: [:i | Worker spawn ].")
            .expect("16 spawns fit under the cap");
        assert!(
            vm.exec("Worker spawn.").is_err(),
            "the 17th spawn must raise"
        );
        // Tidy up: drop every channel so the (still booting) workers exit.
        vm.exec("1 to: 16 do: [:i | (Worker new setId: i) terminate ].")
            .expect("terminate all");
    }

    #[test]
    fn worker_spawn_without_boot_fn_fails_cleanly() {
        // The GamePane posture: with no registered boot closure the world
        // class is harmless â€” spawn raises a clean error, nothing hangs.
        let mut vm = boot_test_vm(JitMode::Off);
        assert!(
            vm.exec("Worker spawn.").is_err(),
            "spawn with no boot fn must raise, not hang or panic"
        );
    }

    #[test]
    fn worker_inbox_wake_fires_and_coalesces() {
        // Â§3.1: the send itself is the wake, coalesced by the pending flag â€”
        // a burst of N replies produces at least one wake and at most N.
        let mut vm = boot_worker_primary();
        let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let w2 = wakes.clone();
        vm.set_inbox_wake(Arc::new(move || {
            w2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }));
        vm.exec("WkTest w1: (Worker spawn: 'Worker onMessage: [:m | Worker reply: m payload]').")
            .expect("spawn");
        vm.exec("1 to: 50 do: [:i | WkTest w1 send: i onReply: [:r | WkTest bump: true] ].")
            .expect("send the burst");
        vm.exec("Worker runLoopWhile: [ (WkTest tickCapped: 100) and: [ WkTest count < 50 ] ].")
            .expect("run until the burst is answered");
        assert_eq!(vm.eval("WkTest count").expect("count").trim(), "50");
        let n = wakes.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            (1..=50).contains(&n),
            "wakes must fire and coalesce: got {n} for 50 replies"
        );
    }

    // â”€â”€ Cocoa GUI CG1: externally-hosted worker + run-loop wake + load_list â”€â”€

    #[test]
    fn hosted_worker_registered_on_this_thread_round_trips() {
        // The CG1 gate (docs/cocoa_gui_design.md Â§3, sprint_cocoa_gui.md CG1):
        // register a worker on the CURRENT thread with NO thread::spawn â€” the
        // arrangement the UI worker needs (its thread is main, blocked in
        // [NSApp run], not recv()). The primary `send:`s it, the caller-supplied
        // wake fires, THIS thread drains the staged envelope + execs
        // `Worker dispatchPending.`, the handler `reply:`s, and the reply routes
        // to the primary's `send:onReply:` continuation â€” the whole no-spawn +
        // wake path end to end, one process, two logical VMs.
        let mut primary = boot_worker_primary();

        // The run-loop-poke wake stands in for performSelectorOnMainThread
        // (CG2); in CG1 it just counts fires so the gate can assert causality
        // (zero before the send, >=1 after).
        let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let wakes_hook = wakes.clone();

        // Register on THIS thread â€” no spawn. `id` shares the spawned id-space;
        // `inbox` is what this thread drains; `to_primary` lets the hosted VM
        // reply back to the primary (the `to_primary` a spawned worker_main
        // gets).
        let (id, inbox, to_primary) = crate::runtime::workers::register_hosted_worker(
            &mut primary.vm,
            Arc::new(move || {
                wakes_hook.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }),
        )
        .expect("registering a hosted worker on a primary must succeed");

        // Boot the hosted worker VM IN PLACE (no spawn), take on the Worker
        // role replying through `to_primary`, install its echo handler â€”
        // exactly what a spawned worker_main does, but driven by this thread.
        let mut hosted = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                jit: JitMode::Off,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("boot the hosted worker VM in place");
        hosted.install_worker_role(id, to_primary);
        hosted
            .exec("Worker onMessage: [:m | Worker reply: m payload * 2].")
            .expect("install the hosted worker's echo handler");

        // The primary needs a Worker handle for this externally-registered id
        // (no `spawn:` returned one) â€” build one over the id, the same
        // `self new setId:` spawn: uses.
        primary
            .exec(&format!("WkTest w1: (Worker new setId: {id})."))
            .expect("wrap the hosted worker id in a Worker handle");

        // No wake yet, no drain yet.
        assert_eq!(wakes.load(std::sync::atomic::Ordering::Relaxed), 0);

        // The primary `send:`s the hosted worker â€” this is the wake trigger.
        primary
            .exec("WkTest w1 send: 21 onReply: [:r | WkTest bump: r = 42].")
            .expect("send a correlated request to the hosted worker");

        // The send fired the wake: a parked host would now know to drain.
        assert!(
            wakes.load(std::sync::atomic::Ordering::Relaxed) >= 1,
            "the primary's send: must fire the hosted worker's run-loop-poke wake"
        );

        // Drain on THIS thread (what a parked host does once woken): stage each
        // envelope into the Worker-role VM, exec dispatchPending â€” that runs
        // the handler and its reply:.
        while let Some(env) = inbox.poll() {
            hosted.stage_pending(env);
            hosted
                .exec("Worker dispatchPending.")
                .expect("dispatch the staged envelope in the hosted worker");
        }

        // The reply now sits in the primary's own inbox; drain it so the
        // correlated continuation runs.
        primary
            .exec("Worker dispatchInbox.")
            .expect("drain the primary's inbox for the reply");

        assert_eq!(
            primary.eval("WkTest count").expect("count").trim(),
            "1",
            "the hosted worker's reply must reach the primary's continuation"
        );
        assert_eq!(
            primary.eval("WkTest bad").expect("bad").trim(),
            "0",
            "the continuation must see the right echoed value (21*2 = 42)"
        );
    }

    // â”€â”€ Cocoa GUI CG4: request protocol + (peer,corr) namespacing + restart â”€â”€

    /// A UI worker with a tiny result scoreboard, booted in place on this thread
    /// (base world â€” the request protocol lives in `47_worker.mst`, so the
    /// conditional Cocoa layer is not needed to exercise it).
    fn boot_ui_worker(id: u32, to_primary: crate::runtime::workers::InboxSender) -> VmHandle {
        let mut ui = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                jit: JitMode::Off,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("boot the UI worker VM in place");
        // The conditional Cocoa layer, exactly as `cocoa_gui`'s boot does â€”
        // class definitions only; nothing here touches AppKit until a view is
        // built, so the layer is fully loadable headless (the CG6 pure-rule
        // gates run against the real `CocoaUI`).
        crate::frontend::world::load_list(&mut ui.vm, Path::new("world/cocoaui.list"))
            .expect("layer the Cocoa UI world onto the UI worker");
        ui.install_worker_role(id, to_primary);
        ui.exec(
            "Object subclass: UiT [ <classVars: R>
                UiT class >> r: x [ R := x ]  UiT class >> r [ ^R ] ]",
        )
        .expect("UI-side scoreboard");
        ui
    }

    /// Drive one Workspace âŒ˜P round-trip end to end: the UI worker ships `src`
    /// as a `#doit`, the primary evaluates + replies, the UI worker's inbox is
    /// drained so the continuation records the result â€” assert it equals
    /// `expect` (a printString).
    fn ui_doit_round_trip(
        primary: &mut VmHandle,
        ui: &mut VmHandle,
        inbox: &crate::runtime::workers::HostedInbox,
        src: &str,
        expect: &str,
    ) {
        ui.exec("UiT r: nil.").expect("reset the scoreboard");
        ui.exec(&format!("Worker uiDoit: '{src}' onReply: [:r | UiT r: r]."))
            .expect("ship the doit request to the primary");
        // The request now sits in the primary's inbox: drain + route + evaluate
        // + reply â€” all on the primary, through execute_do_it.
        primary
            .exec("Worker dispatchInbox.")
            .expect("the primary serves the #uiReq and replies");
        // The reply now sits in the UI worker's inbox. The UI worker routes via
        // dispatchInbox â†’ dispatchOne: (NOT dispatchPending): dispatchOne: fires
        // the pending continuation keyed by (peer 0, corr).
        let mut drained = 0;
        while let Some(env) = inbox.poll() {
            ui.stage_pending(env);
            ui.exec("Worker dispatchInbox.")
                .expect("the UI worker routes the #uiReply to its continuation");
            drained += 1;
        }
        assert!(
            drained >= 1,
            "the primary's #uiReply must reach the UI worker's inbox"
        );
        assert_eq!(
            ui.eval("UiT r").expect("result").trim(),
            expect,
            "the doit ran on the primary and its result came back to the UI worker"
        );
    }

    #[test]
    fn ui_request_doit_round_trips_through_the_primary() {
        // CG4 gate (cocoa_gui_design.md Â§7.3): a UI worker â†’ primary {#uiReq.
        // corr. #doit. source} runs the doit ON the primary (where the
        // persistent objects live) through the existing execute_do_it path, and
        // its {#uiReply. corr. result} comes back to the UI worker's
        // continuation. Two logical VMs in one process, drains driven by this
        // thread â€” the hosted-worker arrangement now carrying the request
        // protocol.
        let mut primary = boot_worker_primary();
        let (id, inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker on the primary");
        let mut ui = boot_ui_worker(id, to_primary);

        ui_doit_round_trip(&mut primary, &mut ui, &inbox, "3 + 4", "'7'");
        // A second, different doit proves the corr advances and the loop keeps
        // routing to the right continuation.
        ui_doit_round_trip(&mut primary, &mut ui, &inbox, "6 * 7", "'42'");
    }

    /// The CG6 headless gate (sprint_cocoa_gui.md): the Workspace's two pure
    /// rules, tested in a real UI-worker VM (cocoaui.list loaded, no AppKit).
    /// `evalTargetFor:loc:len:` is the selection-or-everything rule;
    /// `splice:into:at:` is Print It's inline insert with the captured
    /// insertion point clamped â€” the async-race case (`pendingPrintInsertAt`)
    /// where the buffer shrank before the `#uiReply` landed.
    #[test]
    fn cocoaui_workspace_selection_and_print_splice_rules_are_pure() {
        let mut primary = boot_worker_primary();
        let (id, _inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker on the primary");
        let mut ui = boot_ui_worker(id, to_primary);

        // Selection rule: a real selection evaluates exactly the substringâ€¦
        assert_eq!(
            ui.eval("(CocoaUI evalTargetFor: '3 + 4. 6 * 7.' loc: 7 len: 5) at: 1")
                .expect("selected substring")
                .trim(),
            "'6 * 7'"
        );
        // â€¦and a collapsed (len 0) selection falls back to the whole buffer,
        // inserting at the end.
        assert_eq!(
            ui.eval("(CocoaUI evalTargetFor: '3 + 4.' loc: 3 len: 0) at: 1")
                .expect("whole buffer")
                .trim(),
            "'3 + 4.'"
        );
        assert_eq!(
            ui.eval("(CocoaUI evalTargetFor: '3 + 4.' loc: 3 len: 0) at: 2")
                .expect("insert at end")
                .trim(),
            "6"
        );

        // Print It splice: the result lands right after the captured pointâ€¦
        assert_eq!(
            ui.eval("(CocoaUI splice: '7' into: '3 + 4. rest' at: 6) at: 1")
                .expect("spliced text")
                .trim(),
            "'3 + 4. 7 rest'"
        );
        assert_eq!(
            ui.eval("(CocoaUI splice: '7' into: '3 + 4. rest' at: 6) at: 2")
                .expect("caret lands after the inserted result")
                .trim(),
            "8"
        );
        // â€¦and a stale insertion point beyond the (shrunk) buffer clamps to the
        // end instead of raising â€” the race `pendingPrintInsertAt` exists for.
        assert_eq!(
            ui.eval("(CocoaUI splice: '7' into: '3 + 4.' at: 999) at: 1")
                .expect("clamped splice")
                .trim(),
            "'3 + 4. 7'"
        );
    }

    /// The CG7 primary-side gate: `UiBrowserService browseSnapshot` projects the
    /// LIVE hierarchy into a names-only tree that (a) matches the class model's
    /// own answers row for row (the differential vs the same `ClassMirror` calls
    /// the WKWebView outliner renders from), (b) pickles clean (a class oop
    /// ANYWHERE in the tree would make `Worker pickle:` raise â€” R3's enforcement),
    /// and (c) arrives end-to-end through a real `{#uiReq. corr. #refresh.
    /// #browser}` round trip between two VMs.
    #[test]
    fn browse_snapshot_matches_the_class_model_and_round_trips() {
        let mut primary = boot_worker_primary();
        primary
            .exec(
                "Object subclass: CGDiff [ <classVars: T> \
                   CGDiff class >> t: x [ T := x ]  CGDiff class >> t [ ^T ] \
                   CGDiff class >> find: aName in: node [ \
                       | f | \
                       (node at: 1) = aName ifTrue: [ ^node ]. \
                       (node at: 6) do: [ :k | \
                           f := CGDiff find: aName in: k. \
                           f isNil ifFalse: [ ^f ] ]. \
                       ^nil ] \
                   CGDiff class >> has: aString in: anArray [ \
                       anArray do: [ :s | s = aString ifTrue: [ ^true ] ]. \
                       ^false ] ]",
            )
            .expect("define the tree walker");
        primary
            .exec("CGDiff t: UiBrowserService browseSnapshot.")
            .expect("produce the snapshot once");

        // The root is Object, and a mid-hierarchy class's row set matches the
        // model exactly (count + a known member, both sides sorted the same).
        assert_eq!(
            primary.eval("CGDiff t at: 1").expect("root").trim(),
            "'Object'"
        );
        assert_eq!(
            primary
                .eval(
                    "((CGDiff find: 'OrderedCollection' in: CGDiff t) at: 4) size \
                     = (ClassMirror selectorsOf: OrderedCollection) size"
                )
                .expect("instance-selector row count")
                .trim(),
            "true",
            "the snapshot's instance selectors must match ClassMirror's own count"
        );
        assert_eq!(
            primary
                .eval("CGDiff has: 'add:' in: ((CGDiff find: 'OrderedCollection' in: CGDiff t) at: 4)")
                .expect("known instance selector")
                .trim(),
            "true"
        );
        // Class-side selectors and ivar names project too.
        assert_eq!(
            primary
                .eval("CGDiff has: 'errno' in: ((CGDiff find: 'Posix' in: CGDiff t) at: 5)")
                .expect("known class-side selector")
                .trim(),
            "true"
        );
        assert_eq!(
            primary
                .eval("((CGDiff find: 'Kqueue' in: CGDiff t) at: 2) size")
                .expect("Kqueue's two ivars")
                .trim(),
            "2",
            "instance-variable names must ride the node (kq, buf)"
        );

        // (b) The whole tree pickles + unpickles â€” no class oop crossed.
        assert_eq!(
            primary
                .eval("(Worker unpickle: (Worker pickle: CGDiff t)) at: 1")
                .expect("pickle round trip")
                .trim(),
            "'Object'",
            "the snapshot must survive the worker pickle (names only, R3)"
        );

        // (c) End to end: the UI worker ships {#uiReq. corr. #refresh. #browser},
        // the primary's late-bound UiBrowserService serves it, and the
        // {#browserTree. tree} payload lands in the UI-side continuation.
        let (id, inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker");
        let mut ui = boot_ui_worker(id, to_primary);
        ui.exec("Worker uiRequest: #refresh args: (Array with: #browser) onReply: [:r | UiT r: r].")
            .expect("ship the #refresh request");
        primary
            .exec("Worker dispatchInbox.")
            .expect("the primary serves the #refresh");
        let mut drained = 0;
        while let Some(env) = inbox.poll() {
            ui.stage_pending(env);
            ui.exec("Worker dispatchInbox.")
                .expect("the UI worker routes the reply");
            drained += 1;
        }
        assert!(drained >= 1, "the #uiReply must reach the UI worker");
        assert_eq!(
            ui.eval("UiT r at: 1").expect("payload tag").trim(),
            "#browserTree"
        );
        assert_eq!(
            ui.eval("(UiT r at: 2) at: 1").expect("tree root").trim(),
            "'Object'",
            "the tree itself crossed the channel intact"
        );
    }

    /// The CG7 UI-side gate: `CocoaBrowser`'s path scheme â€” the pure model the
    /// NSOutlineView data-source callbacks answer from â€” resolved over an
    /// installed snapshot, headless. A class node's combined child list is
    /// [instance sels][class sels][subclasses]; paths are 0-based hops; stale
    /// or invalid paths resolve to nil and every consumer fails CLOSED (0
    /// children / empty label), never raises â€” the property that makes a
    /// callback racing a re-blast safe.
    #[test]
    fn cocoa_browser_resolves_paths_over_a_snapshot_and_fails_closed() {
        let mut primary = boot_worker_primary();
        let (id, _inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker");
        let mut ui = boot_ui_worker(id, to_primary);
        ui.exec(
            "Object subclass: CGB7 [
                CGB7 class >> mk [
                    | root sub |
                    sub := Array new: 6.
                    sub at: 1 put: 'Kid'.          sub at: 2 put: #('x').
                    sub at: 3 put: (Array new: 0). sub at: 4 put: #('kidM').
                    sub at: 5 put: (Array new: 0). sub at: 6 put: (Array new: 0).
                    root := Array new: 6.
                    root at: 1 put: 'Object'.      root at: 2 put: (Array new: 0).
                    root at: 3 put: (Array new: 0). root at: 4 put: #('foo' 'bar:').
                    root at: 5 put: #('make').     root at: 6 put: (Array with: sub).
                    ^root ] ]",
        )
        .expect("define the snapshot builder");
        ui.exec("CocoaBrowser installSnapshot: CGB7 mk.")
            .expect("install a small snapshot (no outline built â€” headless)");

        // Paths are 0-based SUBCLASS hops: '' = the root class node, '0' its
        // first subclass. (The multi-pane browser: classes in the outline,
        // selectors in the table.)
        assert_eq!(
            ui.eval("(CocoaBrowser resolvePath: '') at: 1").expect("root name").trim(),
            "'Object'"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser resolvePath: '0') at: 1")
                .expect("first subclass")
                .trim(),
            "'Kid'"
        );
        // The selector pane's data model, side-aware and pure.
        assert_eq!(
            ui.eval("(CocoaBrowser selectorsForPath: '' side: #instance) size")
                .expect("root instance selectors")
                .trim(),
            "2"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser selectorsForPath: '' side: #instance) at: 1")
                .expect("first instance selector")
                .trim(),
            "'foo'"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser selectorsForPath: '' side: #class) at: 1")
                .expect("the class-side selector")
                .trim(),
            "'make'"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser selectorsForPath: '0' side: #instance) at: 1")
                .expect("subclass instance selector")
                .trim(),
            "'kidM'"
        );
        // The class-search helper the scripting/selection drivers build on.
        assert_eq!(
            ui.eval("CocoaBrowser pathToClassNamed: 'Kid'")
                .expect("path to Kid")
                .trim(),
            "'0'"
        );
        assert_eq!(
            ui.eval("CocoaBrowser pathToClassNamed: 'Object'")
                .expect("path to the root")
                .trim(),
            "''"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser pathToClassNamed: 'NoSuch') isNil")
                .expect("missing class")
                .trim(),
            "true"
        );
        // Fail-closed: an out-of-range hop resolves nil â†’ empty selectors.
        assert_eq!(
            ui.eval("(CocoaBrowser resolvePath: '9') isNil")
                .expect("out of range resolves nil")
                .trim(),
            "true"
        );
        assert_eq!(
            ui.eval("(CocoaBrowser selectorsForPath: '9' side: #instance) size")
                .expect("stale path â†’ zero rows")
                .trim(),
            "0"
        );
    }

    /// The live-compile guarantee the browser's Accept flows rest on: a CLASS
    /// DEFINITION (not just an expression) shipped as an ordinary `#doit`
    /// compiles into the live primary â€” the same `vm.exec` semantics the web
    /// GUI's `live_compile` uses, reached over the request channel.
    #[test]
    fn ui_doit_live_compiles_a_class_definition_on_the_primary() {
        let mut primary = boot_worker_primary();
        let (id, inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker");
        let mut ui = boot_ui_worker(id, to_primary);
        ui_doit_round_trip(
            &mut primary,
            &mut ui,
            &inbox,
            "Object subclass: CGLive [ foo [ ^41 + 1 ] ]",
            "'nil'",
        );
        // The class is now live on the primary: methods compile and run.
        ui_doit_round_trip(&mut primary, &mut ui, &inbox, "CGLive new foo", "'42'");
        // And a REOPEN adds a method to the existing class (the one-method
        // accept path's exact live-compile shape).
        ui_doit_round_trip(
            &mut primary,
            &mut ui,
            &inbox,
            "Object subclass: CGLive [ bar [ ^self foo * 2 ] ]",
            "'nil'",
        );
        ui_doit_round_trip(&mut primary, &mut ui, &inbox, "CGLive new bar", "'84'");
    }

    /// The CG9 soundness gate: booting a UI-worker-style VmHandle, publishing
    /// it, and dropping it â€” the exact restart-in-place lifecycle â€” must return
    /// the fixed sigsetjmp + PROBE registries to baseline every cycle, so many
    /// rebuilds never climb toward the caps (JMP=64, PROBE=128) the design
    /// warns a leak would exhaust. Runs on THIS thread (so each boot/drop
    /// claims and releases the SAME `pthread_self()` slot â€” the tightest case:
    /// a stranded slot would be immediately visible as growth). JIT off keeps
    /// the PROBE registry empty of confounding entries; the point is the
    /// Dropâ†’deregisterâ†’release wiring, not codegen.
    #[test]
    fn ui_worker_restart_lifecycle_leaks_no_registry_slots() {
        use crate::codecache::deopt_trap::current_thread_jmp_slots;
        // A hosted primary to adopt a role against (as the real UI worker does).
        let mut primary = boot_worker_primary();
        let (id, _inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the hosted UI worker");

        // THIS thread's slot count is the parallel-safe measure: `claim_jmp_slot`
        // reuses the caller thread's own slot, so boot/eval owns exactly 1 and a
        // clean drop returns it to whatever the primary left (0 or 1). Immune to
        // other threads' concurrent tests, unlike the process-global count.
        let baseline = current_thread_jmp_slots();

        for cycle in 0..40 {
            let mut ui = boot_ui_worker(id, to_primary.clone());
            publish_ui_vm(&mut ui as *mut VmHandle);
            // Exercise it so it actually claims a slot + runs guest code.
            assert_eq!(ui.eval("3 + 4").expect("eval").trim(), "7");
            // Unpublish before drop (the trampolines must never read a dangling
            // pointer) â€” exactly `rebuild_ui`'s order.
            publish_ui_vm(std::ptr::null_mut());
            drop(ui); // Drop = Reservation munmap + deopt deregister + slot release
            // The load-bearing assertion: THIS thread's slot count never grows
            // across cycles â€” a stranded slot would climb toward the 64 cap.
            assert!(
                current_thread_jmp_slots() <= baseline,
                "cycle {cycle}: this thread's jmp slots {} exceeded baseline {baseline} â€” a restart stranded a recovery slot",
                current_thread_jmp_slots()
            );
        }
        assert!(current_thread_jmp_slots() <= baseline);
    }

    #[test]
    fn peer_corr_namespacing_prevents_cross_peer_continuation_collision() {
        // Review R4 (cocoa_gui_design.md Â§7.3): PendingReplies keyed by corr
        // ALONE lets peerA's corr=1 reply fire peerB's corr=1 continuation,
        // because each VM runs its OWN NextCorr. Construct BOTH (peerA=1, corr=1)
        // and (peerB=2, corr=1) continuations, land a distinguishable reply from
        // each (both corr=1), and prove the RIGHT two continuations fire â€” not
        // swapped, not lost (keyed by corr alone, the second registration would
        // overwrite the first at key 1 and one continuation would vanish).
        let mut primary = boot_worker_primary();
        primary
            .exec(
                "Object subclass: R4 [ <classVars: A B>
                    R4 class >> a: x [ A := x ]  R4 class >> b: x [ B := x ]
                    R4 class >> a [ ^A ]  R4 class >> b [ ^B ] ]",
            )
            .expect("R4 scoreboard");
        primary
            .exec("Worker registerReply: [:p | R4 a: p] fromPeer: 1 corr: 1.")
            .expect("register peer-1 continuation at corr 1");
        primary
            .exec("Worker registerReply: [:p | R4 b: p] fromPeer: 2 corr: 1.")
            .expect("register peer-2 continuation at corr 1");

        // Land a distinguishable reply from each peer, both corr=1, straight
        // into the primary's inbox (the same transport a real worker reply uses).
        let bytes_a = primary
            .eval_to_bytes("Worker pickle: 'fromA'")
            .expect("pickle A");
        let bytes_b = primary
            .eval_to_bytes("Worker pickle: 'fromB'")
            .expect("pickle B");
        let inbox = crate::runtime::workers::primary_inbox_sender(&primary.vm)
            .expect("the primary's inbox sender");
        inbox
            .send(crate::runtime::workers::Envelope {
                from: 1,
                corr: 1,
                bytes: bytes_a,
            })
            .expect("land peer-1 reply");
        inbox
            .send(crate::runtime::workers::Envelope {
                from: 2,
                corr: 1,
                bytes: bytes_b,
            })
            .expect("land peer-2 reply");

        primary
            .exec("Worker dispatchInbox.")
            .expect("route both replies");

        assert_eq!(
            primary.eval("R4 a").expect("a").trim(),
            "'fromA'",
            "peer 1's corr=1 reply fired peer 1's continuation"
        );
        assert_eq!(
            primary.eval("R4 b").expect("b").trim(),
            "'fromB'",
            "peer 2's corr=1 reply fired peer 2's continuation â€” NOT swapped, NOT lost"
        );
    }

    #[test]
    fn primary_respawn_from_source_re_syncs_the_ui_worker() {
        // CG4 Â§5/Â§5.1, the headless slice of the watchdog restart: the primary is
        // respawned FROM SOURCE and the UI worker re-syncs â€” a fresh primary
        // registers the UI worker anew, the UI worker's reply link is re-pointed
        // to it, and the next doit round-trips. Death DETECTION (a fatal doit
        // pthread_exiting the primary thread, caught by the watchdog) is the
        // user's on-screen gate; this proves the respawn + re-sync + next-doit
        // core the watchdog drives, without a real-time timeout or a real
        // pthread_exit.
        //
        // The UI worker holds no durable state, so it survives the primary's
        // death untouched (feedback_recover_clean_or_die): boot it ONCE, outside
        // the generations.
        let mut ui = {
            // A throwaway link just to construct the worker; re-pointed below.
            let mut g0 = boot_worker_primary();
            let (id, _inbox, to_primary) =
                crate::runtime::workers::register_hosted_worker(&mut g0.vm, Arc::new(|| {}))
                    .expect("bootstrap link");
            boot_ui_worker(id, to_primary)
        };

        // Generation 1: a from-source primary, the UI worker re-pointed onto it.
        let mut gen1 = boot_worker_primary();
        let (id1, inbox1, to_primary1) =
            crate::runtime::workers::register_hosted_worker(&mut gen1.vm, Arc::new(|| {}))
                .expect("register the UI worker on generation 1");
        ui.install_worker_role(id1, to_primary1);
        ui_doit_round_trip(&mut gen1, &mut ui, &inbox1, "6 * 7", "'42'");

        // Scripted primary death: drop the whole VM generation. Its heap unmaps,
        // its inbox receiver drops â€” the honest clean loss the design takes over
        // a fake rollback (feedback_recover_clean_or_die). Any outstanding
        // continuations to it are orphaned with the dead VM.
        drop(gen1);
        drop(inbox1);

        // Respawn FROM SOURCE (the watchdog's boot closure) + re-register the UI
        // worker, re-pointing its reply link onto the fresh primary â€” the
        // Â§5.1 re-sync.
        let mut gen2 = boot_worker_primary();
        let (id2, inbox2, to_primary2) =
            crate::runtime::workers::register_hosted_worker(&mut gen2.vm, Arc::new(|| {}))
                .expect("register the UI worker on the respawned primary");
        ui.install_worker_role(id2, to_primary2);

        // The next doit works â€” the environment recovered and the UI re-synced.
        ui_doit_round_trip(&mut gen2, &mut ui, &inbox2, "100 + 1", "'101'");
    }

    #[test]
    fn primary_transcript_forwards_to_the_ui_worker() {
        // CG4 Â§7.4: the primary's OWN transcript is forwarded to the UI worker's
        // inbox (ForwardTranscript, direction-flipped, UNtagged) and the UI
        // worker's dispatchOne: shows each line on ITS Transcript â€” the "primary
        // â†’ UI transcript sink" the on-screen Transcript view renders.
        struct VecSink(Arc<Mutex<Vec<String>>>);
        impl TranscriptSink for VecSink {
            fn show(&mut self, text: &str) {
                self.0.lock().unwrap().push(text.to_string());
            }
        }
        let mut primary = boot_worker_primary();
        let (id, inbox, to_primary) =
            crate::runtime::workers::register_hosted_worker(&mut primary.vm, Arc::new(|| {}))
                .expect("register the UI worker");
        let mut ui = boot_ui_worker(id, to_primary);

        // The UI worker's Transcript view stands in as a capturing sink.
        let captured = Arc::new(Mutex::new(Vec::new()));
        ui.set_transcript(Box::new(VecSink(captured.clone())));

        // Flip the primary's transcript to the UI worker's inbox, then write.
        primary.forward_transcript_to_ui(id);
        primary
            .exec("Transcript showCr: 'hello from the primary'.")
            .expect("primary writes to its (now forwarded) transcript");

        // Drain the UI worker's inbox â€” dispatchOne: shows the forwarded line.
        while let Some(env) = inbox.poll() {
            ui.dispatch_hosted_envelope(env)
                .expect("UI worker routes the forwarded transcript");
        }
        let lines = captured.lock().unwrap();
        assert!(
            lines.iter().any(|l| l.contains("hello from the primary")),
            "the primary's transcript line must reach the UI worker's Transcript, got {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("[w0]")),
            "the primary's own transcript must be UNtagged (no [w0]), got {lines:?}"
        );
    }

    #[test]
    fn load_list_layers_an_extra_world_on_top_of_the_base() {
        // The CG1 gate for the conditional world layer (docs/cocoa_gui_design.md
        // Â§12.3): a class in world/cocoaui.list (63_cocoaui_stub.mst) is ABSENT
        // from the base world and PRESENT â€” its method runnable â€” only after
        // load_list. Proves the extra layer loads on top of a booted base world
        // without being in world/world.list.
        let mut vm = boot_test_vm(JitMode::Off);
        // Absent from the base world: referencing it is an undeclared-variable
        // compile error (surfaced as Err, never a process exit).
        assert!(
            vm.eval("CocoaUIStub ping").is_err(),
            "CocoaUIStub must be absent from the base world \
             (world.list carries none of cocoaui.list)"
        );
        // Layer the extra list on top of the already-booted base world.
        crate::frontend::world::load_list(&mut vm.vm, Path::new("world/cocoaui.list"))
            .expect("load_list must layer world/cocoaui.list cleanly");
        // Now the fresh class resolves and its method actually runs.
        assert_eq!(
            vm.eval("CocoaUIStub ping")
                .expect("CocoaUIStub ping must run after load_list")
                .trim(),
            "#cocoaUiStubReady",
            "the class + method from the extra layer must be live after load_list"
        );
    }

    // â”€â”€ Cocoa bridge C0 gates (docs/cocoa_bridge_design.md Â§8) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// The wrap/release counters are process-wide and the test harness is
    /// parallel â€” every Cocoa test takes this lock so counter deltas (and
    /// pool traffic) can't interleave.
    fn cocoa_serial() -> std::sync::MutexGuard<'static, ()> {
        static L: Mutex<()> = Mutex::new(());
        L.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn cocoa_c0_process_name_round_trips() {
        let _serial = cocoa_serial();
        // The canonical C0 gate: a real Foundation object, a real send, a
        // real NSString copied back â€” on the VM thread, headless.
        let mut vm = boot_test_vm(JitMode::Off);
        let name = vm
            .eval("((Cocoa classNamed: 'NSProcessInfo') send: 'processInfo') sendString: 'processName'")
            .expect("the processName round-trip must run cleanly");
        assert!(
            name.contains("macvm"),
            "processName should be this test binary's name, got {name}"
        );
    }

    #[test]
    fn cocoa_c0_tagged_pointer_ids_survive_the_byte_tail() {
        let _serial = cocoa_serial();
        // The adversarial-review regression: small NSNumbers and short
        // NSStrings are TAGGED POINTERS (bit 63 set) â€” they would have
        // panicked SmallInt::new under the named-slot idiom and been
        // corrupted by an oop scan as raw words. In the byte tail they are
        // just bytes.
        let mut vm = boot_test_vm(JitMode::Off);
        let n = vm
            .eval("((Cocoa classNamed: 'NSNumber') send: 'numberWithInteger:' with: 42) sendI64: 'integerValue'")
            .expect("tagged-pointer NSNumber round-trip");
        assert_eq!(n.trim(), "42");
        let s = vm
            .eval("(Cocoa nsString: 'hi') sendString: 'uppercaseString'")
            .expect("tagged-pointer NSString round-trip");
        assert!(s.contains("HI"), "expected HI, got {s}");
    }

    #[test]
    fn cocoa_c0_release_poisons_and_double_release_fails() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaT [ <classVars: R> CocoaT class >> r: x [ R := x ] CocoaT class >> r [ ^R ] ]")
            .expect("holder");
        vm.exec("CocoaT r: (Cocoa nsString: 'poison me').")
            .expect("wrap");
        assert_eq!(vm.eval("CocoaT r isValid").unwrap().trim(), "true");
        vm.exec("CocoaT r release.").expect("first release");
        assert_eq!(vm.eval("CocoaT r isValid").unwrap().trim(), "false");
        assert!(
            vm.exec("CocoaT r sendString: 'description'.").is_err(),
            "a send through a poisoned wrapper must raise"
        );
        assert!(
            vm.exec("CocoaT r release.").is_err(),
            "a double release must raise (leak-side bias, never over-release)"
        );
    }

    #[test]
    fn cocoa_c0_nsexception_is_caught_not_fatal() {
        let _serial = cocoa_serial();
        // An unrecognized ObjC selector throws NSInvalidArgumentException;
        // the shim catches it, the prim fails, Smalltalk raises â€” and the
        // VM keeps working afterwards.
        let mut vm = boot_test_vm(JitMode::Off);
        assert!(
            vm.exec("(Cocoa nsString: 'x') send: 'thisSelectorDoesNotExistXyzzy'.")
                .is_err(),
            "the NSException must surface as a Smalltalk error"
        );
        assert_eq!(
            vm.eval("3 + 4").expect("the VM must still work").trim(),
            "7"
        );
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c0_wrap_release_counters_balance() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        let (w0, r0, _) = crate::runtime::objc_bridge::counters();
        vm.exec("(Cocoa nsString: 'one') release.").expect("1");
        vm.exec("(Cocoa nsString: 'two') release.").expect("2");
        vm.exec("(Cocoa nsString: 'three') release.").expect("3");
        let (w1, r1, _) = crate::runtime::objc_bridge::counters();
        assert_eq!(w1 - w0, 3, "three wraps");
        assert_eq!(r1 - r0, 3, "three releases â€” balanced");
    }

    #[test]
    fn cocoa_c0_wrappers_survive_gc_stress_churn() {
        let _serial = cocoa_serial();
        // The moving-GC gate: wrappers churn (and one stays live) while
        // every allocation collects. The wrapper OOPS move constantly; the
        // ids in their byte tails must not.
        let mut vm = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                gc_stress: true,
                jit: JitMode::Off,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("gc-stress boot");
        vm.exec("Object subclass: CocoaG [ <classVars: K> CocoaG class >> k: x [ K := x ] CocoaG class >> k [ ^K ] ]")
            .expect("holder");
        // A long-lived wrapper that will be moved by many collectionsâ€¦
        vm.exec("CocoaG k: (Cocoa nsString: 'survivor').")
            .expect("keep");
        // â€¦while churn wraps + releases around it.
        for _ in 0..40 {
            vm.exec("(Cocoa nsString: 'churn') release.")
                .expect("churn");
        }
        let s = vm
            .eval("CocoaG k sendString: 'uppercaseString'")
            .expect("the survivor must still answer after heavy GC");
        assert!(s.contains("SURVIVOR"), "got {s}");
        vm.exec("CocoaG k release.").expect("tidy");
    }

    // â”€â”€ Cocoa bridge C1 gates (marshalling breadth + ownership families) â”€
    //
    // Every ABI shape asserted here was cross-checked against cocoa_data's
    // register classification (docs/FFI.md Â§1 tokens) before being pinned:
    // numberWithDouble: takes `f`; rangeOfString: returns `i2` (x0/x1);
    // valueWithPoint:/pointValue are `h2` (d0/d1); valueWithRect:/rectValue
    // are `h4` (d0..d3); dateWithEra:â€¦nanosecond: is 8 `g` args â€” six ride
    // x2..x7, the last two cross on the STACK.

    #[test]
    fn cocoa_c1_double_and_bool_marshal_both_directions() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        // A Double ARGUMENT (FPR class) and a double RESULT (d0) in one
        // round trip through NSNumber.
        let d = vm
            .eval(
                "((Cocoa classNamed: 'NSNumber') send: 'numberWithDouble:' args: #(2.75) ret: #id) \
                 sendF64: 'doubleValue'",
            )
            .expect("double round-trip");
        assert_eq!(d.trim(), "2.75");
        // A BOOL result (w0's low byte, masked) â€” both polarities, with a
        // String argument auto-bridged to a temp NSString each time.
        let t = vm
            .eval("(Cocoa nsString: 'abc') sendBool: 'isEqualToString:' args: #('abc')")
            .expect("bool true");
        assert_eq!(t.trim(), "true");
        let f = vm
            .eval("(Cocoa nsString: 'abc') sendBool: 'isEqualToString:' args: #('xyz')")
            .expect("bool false");
        assert_eq!(f.trim(), "false");
        // The adversarial-review regression (#i32): a C `int` return is
        // w0-only â€” read as #i64, intValue's -5 would arrive as 2^32-5, a
        // silently wrong (in-smi-range!) answer. #i32 sign-extends.
        let n = vm
            .eval(
                "((Cocoa classNamed: 'NSNumber') send: 'numberWithInteger:' with: -5) \
                 send: 'intValue' args: #() ret: #i32",
            )
            .expect("negative int return");
        assert_eq!(n.trim(), "-5", "#i32 must sign-extend w0");
    }

    #[test]
    fn cocoa_c1_nsrange_returns_the_x0_x1_pair() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaRg [ <classVars: R> CocoaRg class >> r: x [ R := x ] CocoaRg class >> r [ ^R ] ]")
            .expect("holder");
        vm.exec("CocoaRg r: ((Cocoa nsString: 'hello world') sendRange: 'rangeOfString:' args: #('world')).")
            .expect("rangeOfString:");
        assert_eq!(vm.eval("CocoaRg r at: 1").unwrap().trim(), "6", "location");
        assert_eq!(vm.eval("CocoaRg r at: 2").unwrap().trim(), "5", "length");
        // An NSException thrown through the NEW general entry point is
        // still caught â€” the @try boundary moved with the shim.
        assert!(
            vm.exec("(Cocoa nsString: 'x') send: 'noSuchSelectorZyx' args: #() ret: #range.")
                .is_err(),
            "the exception must surface as a Smalltalk error, not kill the VM"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
    }

    #[test]
    fn cocoa_c1_hfa_point_and_rect_round_trip() {
        let _serial = cocoa_serial();
        // The flat-register model's HFA payoff: a CGPoint argument IS two
        // Doubles (d0/d1), a CGRect four â€” and the HFA RESULTS come back
        // out of d0..d3. Headless Foundation round-trip through NSValue.
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaHf [ <classVars: P R> CocoaHf class >> p: x [ P := x ] CocoaHf class >> p [ ^P ] CocoaHf class >> r: x [ R := x ] CocoaHf class >> r [ ^R ] ]")
            .expect("holder");
        vm.exec("CocoaHf p: (((Cocoa classNamed: 'NSValue') send: 'valueWithPoint:' args: #(3.5 4.5) ret: #id) sendPoint: 'pointValue').")
            .expect("point round-trip");
        assert_eq!(vm.eval("CocoaHf p at: 1").unwrap().trim(), "3.5");
        assert_eq!(vm.eval("CocoaHf p at: 2").unwrap().trim(), "4.5");
        vm.exec("CocoaHf r: (((Cocoa classNamed: 'NSValue') send: 'valueWithRect:' args: #(1.5 2.5 30.25 40.75) ret: #id) sendRect: 'rectValue').")
            .expect("rect round-trip");
        assert_eq!(vm.eval("CocoaHf r at: 1").unwrap().trim(), "1.5");
        assert_eq!(vm.eval("CocoaHf r at: 2").unwrap().trim(), "2.5");
        assert_eq!(vm.eval("CocoaHf r at: 3").unwrap().trim(), "30.25");
        assert_eq!(vm.eval("CocoaHf r at: 4").unwrap().trim(), "40.75");
    }

    #[test]
    fn cocoa_c1_eight_arg_send_spills_to_the_stack() {
        let _serial = cocoa_serial();
        // dateWithEra:year:month:day:hour:minute:second:nanosecond: is 8
        // GPR-class arguments: era..minute ride x2..x7, SECOND and
        // nanosecond cross on the stack words. Reading the second back
        // (45) proves the stack path end-to-end against a real Foundation
        // method â€” the FFI arc's argv-overflow bug, re-gated as a
        // wired-through feature instead of a crash.
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaCal [ <classVars: C D> CocoaCal class >> c: x [ C := x ] CocoaCal class >> c [ ^C ] CocoaCal class >> d: x [ D := x ] CocoaCal class >> d [ ^D ] ]")
            .expect("holder");
        vm.exec("CocoaCal c: ((Cocoa classNamed: 'NSCalendar') send: 'currentCalendar').")
            .expect("calendar");
        vm.exec("CocoaCal c send: 'setTimeZone:' args: (Array with: ((Cocoa classNamed: 'NSTimeZone') send: 'timeZoneForSecondsFromGMT:' with: 0)) ret: #void.")
            .expect("pin UTC so the read-back is deterministic");
        vm.exec("CocoaCal d: (CocoaCal c send: 'dateWithEra:year:month:day:hour:minute:second:nanosecond:' args: #(1 2026 7 14 12 30 45 0) ret: #id).")
            .expect("the 8-arg send");
        // NSCalendarUnitSecond = 128 (cocoa_data's enum table).
        let s = vm
            .eval("CocoaCal c send: 'component:fromDate:' args: (Array with: 128 with: CocoaCal d) ret: #i64")
            .expect("read the second back");
        assert_eq!(s.trim(), "45", "second=45 crossed via the stack words");
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c1_alloc_init_transfers_ownership_and_balances() {
        let _serial = cocoa_serial();
        // The +1-family classifier live (design Â§3.2): alloc's result is
        // already owned (no double retain), init CONSUMES the alloc
        // receiver (its wrapper poisons â€” class clusters may swap the
        // object) and answers a +1 result. The counters must balance as
        // wraps == releases + consumed.
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaOwn [ <classVars: K A B> CocoaOwn class >> k: x [ K := x ] CocoaOwn class >> k [ ^K ] CocoaOwn class >> a: x [ A := x ] CocoaOwn class >> a [ ^A ] CocoaOwn class >> b: x [ B := x ] CocoaOwn class >> b [ ^B ] ]")
            .expect("holder");
        let (w0, r0, c0) = crate::runtime::objc_bridge::counters();
        vm.exec("CocoaOwn k: (Cocoa classNamed: 'NSMutableString').")
            .expect("class");
        vm.exec("CocoaOwn a: (CocoaOwn k send: 'alloc').")
            .expect("alloc (+1 family)");
        vm.exec("CocoaOwn b: (CocoaOwn a send: 'init').")
            .expect("init (consumes the receiver)");
        assert_eq!(
            vm.eval("CocoaOwn a isValid").unwrap().trim(),
            "false",
            "init consumed the alloc receiver â€” its wrapper must be poisoned"
        );
        assert_eq!(vm.eval("CocoaOwn b isValid").unwrap().trim(), "true");
        // The initialized object actually works (append via the temp-
        // NSString bridge, then read back).
        vm.exec("CocoaOwn b send: 'appendString:' args: #('grown') ret: #void.")
            .expect("append");
        let s = vm
            .eval("CocoaOwn b sendString: 'description'")
            .expect("read");
        assert!(s.contains("grown"), "got {s}");
        vm.exec("CocoaOwn b release.").expect("release the result");
        vm.exec("CocoaOwn k release.")
            .expect("release the class wrapper");
        let (w1, r1, c1) = crate::runtime::objc_bridge::counters();
        assert_eq!(w1 - w0, 3, "three wraps: class, alloc, init result");
        assert_eq!(r1 - r0, 2, "two releases: result + class wrapper");
        assert_eq!(c1 - c0, 1, "one init-family consume");
        assert_eq!(
            (w1 - w0),
            (r1 - r0) + (c1 - c0),
            "ownership balance: wraps == releases + consumed"
        );
        // A +1-family selector through a path that can't take ownership is
        // REFUSED before sending (C1 review findings): the +1 result would
        // leak invisibly behind an integer/string return. The receiver is
        // pre-minted so the refused sends themselves are the only thing
        // between the two counter snapshots.
        vm.exec("CocoaOwn a: (Cocoa nsString: 'x').")
            .expect("a fresh receiver for the refusal checks");
        let (w2, r2, c2) = crate::runtime::objc_bridge::counters();
        assert!(
            vm.exec("CocoaOwn a sendI64: 'copy'.").is_err(),
            "sendI64: must refuse a +1-family selector"
        );
        assert!(
            vm.exec("CocoaOwn a send: 'copy' args: #() ret: #str.")
                .is_err(),
            "a +1-family selector with a non-#id ret token must be refused"
        );
        let (w3, r3, c3) = crate::runtime::objc_bridge::counters();
        assert_eq!(
            (w3, r3, c3),
            (w2, r2, c2),
            "refused sends must not move any counter"
        );
        vm.exec("CocoaOwn a release.").expect("tidy");
    }

    #[test]
    fn cocoa_c1_hfa_results_survive_gc_stress() {
        let _serial = cocoa_serial();
        // The write-barrier regression (C1 review finding 1): the
        // point/rect result arm allocates the array THEN each Double, so a
        // mid-loop scavenge can promote the array â€” the subsequent stores
        // must go through the barrier door or an oldâ†’new slot goes
        // invisible and dangles. Under gc_stress every allocation
        // collects, exercising every promote/store interleaving the
        // adaptive tenuring policy produces.
        let mut vm = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                gc_stress: true,
                jit: JitMode::Off,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("gc-stress boot");
        vm.exec("Object subclass: CocoaHfG [ <classVars: R> CocoaHfG class >> r: x [ R := x ] CocoaHfG class >> r [ ^R ] ]")
            .expect("holder");
        for i in 0..25 {
            vm.exec("CocoaHfG r: (((Cocoa classNamed: 'NSValue') send: 'valueWithRect:' args: #(1.5 2.5 30.25 40.75) ret: #id) sendRect: 'rectValue').")
                .expect("rect round-trip under stress");
            // Force more churn between construction and the reads.
            vm.exec("(Cocoa nsString: 'churn') release.")
                .expect("churn");
            for (ix, want) in [(1, "1.5"), (2, "2.5"), (3, "30.25"), (4, "40.75")] {
                let got = vm
                    .eval(&format!("CocoaHfG r at: {ix}"))
                    .expect("element read must not dangle");
                assert_eq!(got.trim(), want, "iteration {i}, slot {ix}");
            }
        }
    }

    // â”€â”€ Cocoa bridge C2 gates (DNU dispatch + cached shape resolution) â”€â”€

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c2_keyword_sends_drive_foundation() {
        let _serial = cocoa_serial();
        // The design's own acceptance shape: a Workspace-style doit drives
        // Foundation with ordinary Smalltalk keyword sends â€” alloc/init
        // (ownership families through DNU), a void append, an NSUInteger
        // read-back. No send:args:ret: anywhere.
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaDnu [ <classVars: A S> CocoaDnu class >> a: x [ A := x ] CocoaDnu class >> a [ ^A ] CocoaDnu class >> s: x [ S := x ] CocoaDnu class >> s [ ^S ] ]")
            .expect("holder");
        let (w0, r0, c0) = crate::runtime::objc_bridge::counters();
        vm.exec("CocoaDnu a: ((Cocoa classNamed: 'NSMutableString') alloc).")
            .expect("alloc through DNU (+1 family)");
        vm.exec("CocoaDnu s: (CocoaDnu a) init.")
            .expect("init through DNU (consumes the receiver)");
        // The C2 review's gate gap, closed: the init-consume must fire on
        // the DNU path (prim 241), not just C1's explicit send: (prim 231).
        assert_eq!(
            vm.eval("CocoaDnu a isValid").unwrap().trim(),
            "false",
            "init through DNU must consume the alloc receiver"
        );
        assert_eq!(vm.eval("CocoaDnu s isValid").unwrap().trim(), "true");
        vm.exec("CocoaDnu s appendString: 'hello'.")
            .expect("void keyword send");
        vm.exec("CocoaDnu s appendString: ' world'.")
            .expect("second append");
        assert_eq!(
            vm.eval("CocoaDnu s length").expect("NSUInteger ret").trim(),
            "11"
        );
        let s = vm.eval("CocoaDnu s asString").expect("description");
        assert!(s.contains("hello world"), "got {s}");
        vm.exec("CocoaDnu s release.").expect("tidy");
        let (w1, r1, c1) = crate::runtime::objc_bridge::counters();
        // classNamed: wrap + alloc wrap + init wrap = 3 (the inline class
        // wrapper leaks by design â€” leak-side bias, classes are immortal).
        assert_eq!(w1 - w0, 3, "class, alloc, init-result wraps");
        assert_eq!(r1 - r0, 1, "one release (the result)");
        assert_eq!(c1 - c0, 1, "one DNU init-family consume");
    }

    #[test]
    fn cocoa_c2_encoding_driven_coercion() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        // The CALLEE's signature decides the register class now: a
        // SmallInteger 3 passed to numberWithDouble: (encoding `d`)
        // coerces to d0 â€” under C1's tag-driven marshal it would have
        // ridden a GPR and the callee read garbage.
        let d = vm
            .eval("((Cocoa classNamed: 'NSNumber') numberWithDouble: 3) doubleValue")
            .expect("intâ†’double coercion");
        assert_eq!(d.trim(), "3.0");
        // #i32 via the encoding (`i`), no explicit token needed.
        let n = vm
            .eval("((Cocoa classNamed: 'NSNumber') numberWithInteger: -5) intValue")
            .expect("negative int return");
        assert_eq!(n.trim(), "-5");
        // BOOL via the encoding, both polarities.
        assert_eq!(
            vm.eval("(Cocoa nsString: 'abc') isEqualToString: 'abc'")
                .expect("bool true")
                .trim(),
            "true"
        );
        assert_eq!(
            vm.eval("(Cocoa nsString: 'abc') isEqualToString: 'xyz'")
                .expect("bool false")
                .trim(),
            "false"
        );
        // float (f32) argument AND return â€” the s-register path.
        let f = vm
            .eval("((Cocoa classNamed: 'NSNumber') numberWithFloat: 2.5) floatValue")
            .expect("f32 round-trip");
        assert_eq!(f.trim(), "2.5");
        // A `c` return is a signed CHAR, answered as a SmallInteger â€” on
        // arm64 BOOL encodes `B`, so Bool-ifying `c` returned true for
        // charValue 65 (the C2 review's silent-wrong-answer finding).
        let c = vm
            .eval("((Cocoa classNamed: 'NSNumber') numberWithInteger: 65) charValue")
            .expect("char return");
        assert_eq!(c.trim(), "65", "charValue answers the char, not true");
        let cn = vm
            .eval("((Cocoa classNamed: 'NSNumber') numberWithInteger: -5) charValue")
            .expect("negative char return");
        assert_eq!(cn.trim(), "-5", "char sign-extends from 8 bits");
        // Manual reference counting is refused at EVERY send path â€”
        // ownership belongs to the bridge, and `dealloc` through DNU
        // would be a use-after-free (C2 review).
        assert!(
            vm.exec("(Cocoa nsString: 'x') primSendAuto: 'retain' args: #().")
                .is_err(),
            "raw retain must be refused"
        );
        assert!(
            vm.exec("(Cocoa nsString: 'x') send: 'dealloc'.").is_err(),
            "dealloc must be refused on the C1 path too"
        );
    }

    #[test]
    fn cocoa_nil_selector_argument_marshals_to_null_sel() {
        // A nil SEL argument marshals to a NULL SEL, NOT a failed send â€” the
        // on-screen CocoaUI bug: a submenu-holding NSMenuItem is built with
        // `action: nil`, and the auto-marshaller rejected nil for a `:` slot,
        // so the whole menu build (and startup) died. `respondsToSelector: nil`
        // is `[obj respondsToSelector: NULL]` â†’ NO, and must not raise.
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        let r = vm
            .eval(
                "(Cocoa nsString: 'x') primSendAuto: 'respondsToSelector:' args: (Array with: nil)",
            )
            .expect("a nil SEL arg must marshal to NULL, not fail the send");
        assert_eq!(r.trim(), "false", "respondsToSelector: NULL answers NO");
        // The non-nil branch still resolves a real selector (else-arm guard).
        let t = vm
            .eval("(Cocoa nsString: 'x') primSendAuto: 'respondsToSelector:' args: (Array with: 'length')")
            .expect("a real SEL arg still resolves");
        assert_eq!(t.trim(), "true");
    }

    #[test]
    fn cocoa_c2_struct_shapes_via_dnu() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaDnS [ <classVars: R P> CocoaDnS class >> r: x [ R := x ] CocoaDnS class >> r [ ^R ] CocoaDnS class >> p: x [ P := x ] CocoaDnS class >> p [ ^P ] ]")
            .expect("holder");
        // NSRange return, resolved from the encoding â€” an Array answer.
        vm.exec("CocoaDnS r: ((Cocoa nsString: 'hello world') rangeOfString: 'world').")
            .expect("rangeOfString: via DNU");
        assert_eq!(vm.eval("CocoaDnS r at: 1").unwrap().trim(), "6");
        assert_eq!(vm.eval("CocoaDnS r at: 2").unwrap().trim(), "5");
        // A CGPoint ARGUMENT is an Array of 2 numbers under the encoding-
        // driven marshal; the HFA result comes back as an Array of Doubles.
        vm.exec("CocoaDnS p: (((Cocoa classNamed: 'NSValue') valueWithPoint: (Array with: 3.5 with: 4.5)) pointValue).")
            .expect("CGPoint round-trip via DNU");
        assert_eq!(vm.eval("CocoaDnS p at: 1").unwrap().trim(), "3.5");
        assert_eq!(vm.eval("CocoaDnS p at: 2").unwrap().trim(), "4.5");
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c2_shape_cache_hits_are_visible_in_stats() {
        let _serial = cocoa_serial();
        // The design's "PIC hit-rate visible in stats": repeated DNU sends
        // of one selector cost ONE runtime resolution; the rest are cache
        // hits, and __vmStats surfaces both counters.
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaHit [ <classVars: N> CocoaHit class >> n: x [ N := x ] CocoaHit class >> n [ ^N ] ]")
            .expect("holder");
        vm.exec("CocoaHit n: (Cocoa nsString: 'hit rate').")
            .expect("receiver");
        let (h0, m0) = crate::runtime::objc_bridge::shape_stats();
        vm.exec("1 to: 20 do: [:i | CocoaHit n length ].")
            .expect("20 DNU sends of one selector");
        let (h1, m1) = crate::runtime::objc_bridge::shape_stats();
        assert!(
            m1 - m0 <= 2,
            "one selector on one class must resolve at most twice (got {} misses)",
            m1 - m0
        );
        assert!(
            h1 - h0 >= 18,
            "the remaining sends must be cache hits (got {})",
            h1 - h0
        );
        let stats = crate::runtime::vm_state::format_vm_stats(&vm.vm);
        assert!(
            stats.contains("cocoa_shape_hits="),
            "the hit-rate must be visible in the stats surface, got:\n{stats}"
        );
        vm.exec("CocoaHit n release.").expect("tidy");
    }

    #[test]
    fn cocoa_c2_unknown_selector_and_non_cocoa_dnu_fail_cleanly() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        // An unknown ObjC selector: resolution fails, the world fallback
        // raises, the VM lives on.
        assert!(
            vm.exec("(Cocoa nsString: 'x') fooBarBazQux.").is_err(),
            "an unresolvable selector must raise cleanly"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
        // Object's own doesNotUnderstand: is untouched â€” a non-Cocoa DNU
        // still errors the classic way (regression guard).
        assert!(
            vm.exec("3 fooBarBazQux.").is_err(),
            "ordinary DNU must still raise"
        );
        // A keyword-arity mismatch against the real signature also fails
        // cleanly (length declares no arguments).
        assert!(
            vm.exec("(Cocoa nsString: 'x') primSendAuto: 'length' args: #(1 2).")
                .is_err(),
            "an arity mismatch must raise cleanly"
        );
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c3_hop_disabled_fails_cleanly() {
        let _serial = cocoa_serial();
        // Headless: nothing drains the main dispatch queue, so the sync
        // hop must FAIL CLEANLY (a Smalltalk error), never hang. Nothing
        // in the lib-test process ever calls enable_main_hop â€” the real
        // dispatch hop is proven by the harness=false integration test
        // (tests/cocoa_main_hop.rs), which owns a genuine main thread.
        assert!(
            !crate::runtime::objc_bridge::main_hop_enabled(),
            "no lib test may enable the process-wide hop"
        );
        let mut vm = boot_test_vm(JitMode::Off);
        assert!(
            vm.exec("(Cocoa classNamed: 'NSThread') sendMain: 'isMainThread' args: #().")
                .is_err(),
            "an un-enabled hop must raise, not hang"
        );
        assert!(
            vm.exec("(Cocoa classNamed: 'NSThread') onMain isMainThread.")
                .is_err(),
            "the onMain proxy must raise too"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
    }

    // â”€â”€ Cocoa bridge C4 gates (callbacks + the in-heap mint-list) â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn cocoa_c4_action_fires_and_dead_ticket_drops() {
        let _serial = cocoa_serial();
        // The full callback circle, headless: Cocoa action: registers a
        // block + mints a MacvmAction; a `macvmFire:` send (through DNU!)
        // posts the {#cocoaEvent. ticket} envelope; Worker dispatchInbox
        // runs the block BETWEEN doits on the VM thread.
        let mut vm = boot_worker_primary();
        vm.exec("Object subclass: CocoaCb [ <classVars: N A> CocoaCb class >> reset [ N := 0 ] CocoaCb class >> bump [ N := N + 1 ] CocoaCb class >> n [ ^N ] CocoaCb class >> a: x [ A := x ] CocoaCb class >> a [ ^A ] ]")
            .expect("holder");
        vm.exec("CocoaCb reset.").expect("reset");
        vm.exec("CocoaCb a: (Cocoa action: [ CocoaCb bump ]).")
            .expect("register the action");
        vm.exec("CocoaCb a macvmFire: nil.").expect("fire 1");
        vm.exec("CocoaCb a macvmFire: nil.").expect("fire 2");
        assert_eq!(
            vm.eval("CocoaCb n").unwrap().trim(),
            "0",
            "fires queue; nothing runs mid-doit (the strictly-serial rule)"
        );
        vm.exec("Worker dispatchInbox.").expect("dispatch");
        assert_eq!(vm.eval("CocoaCb n").unwrap().trim(), "2");
        // Unregister: a late fire for a dead ticket is dropped silently
        // (tickets are monotonic from 1 in a fresh VM).
        vm.exec("Cocoa unregisterAction: 1.").expect("tombstone");
        vm.exec("CocoaCb a macvmFire: nil.").expect("late fire");
        vm.exec("Worker dispatchInbox.").expect("dispatch again");
        assert_eq!(
            vm.eval("CocoaCb n").unwrap().trim(),
            "2",
            "a dead ticket's fire must be dropped, not an error"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c4_pool_releases_minted_keeps_kept() {
        let _serial = cocoa_serial();
        let mut vm = boot_test_vm(JitMode::Off);
        vm.exec("Object subclass: CocoaPl [ <classVars: K> CocoaPl class >> k: x [ K := x ] CocoaPl class >> k [ ^K ] ]")
            .expect("holder");
        let (w0, r0, _) = crate::runtime::objc_bridge::counters();
        vm.exec("CocoaPl k: (Cocoa poolDo: [:p | (Cocoa nsString: 'a'). (Cocoa nsString: 'b'). p keep: (Cocoa nsString: 'kept') ]).")
            .expect("poolDo: with 3 mints, 1 kept");
        let (w1, r1, _) = crate::runtime::objc_bridge::counters();
        assert_eq!(w1 - w0, 3, "three wrappers minted in the scope");
        assert_eq!(r1 - r0, 2, "the two non-kept were released on the way out");
        assert_eq!(vm.eval("CocoaPl k isValid").unwrap().trim(), "true");
        let s = vm.eval("CocoaPl k sendString: 'description'").unwrap();
        assert!(s.contains("kept"), "got {s}");
        vm.exec("CocoaPl k release.").expect("tidy");
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c4_pool_and_callbacks_survive_gc_stress() {
        let _serial = cocoa_serial();
        // The design's own C4 soak gate: poolDo: scopes with enough mints
        // to force in-heap mint-list GROWTH (past the initial 8 slots),
        // interleaved with callback fires + dispatch, every allocation
        // collecting. The mint-list arrays move constantly; being rooted
        // heap objects, nothing dangles and the release sweep stays exact.
        let mut vm = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                gc_stress: true,
                jit: JitMode::Off,
                ..Default::default()
            },
            Path::new("world"),
        )
        .expect("gc-stress boot");
        vm.set_worker_boot(Arc::new(|| {
            VmHandle::boot(
                VmOptions {
                    heap_mib: 64,
                    jit: JitMode::Off,
                    ..Default::default()
                },
                Path::new("world"),
            )
        }));
        vm.exec("Object subclass: CocoaSk [ <classVars: N A K> CocoaSk class >> reset [ N := 0 ] CocoaSk class >> bump [ N := N + 1 ] CocoaSk class >> n [ ^N ] CocoaSk class >> a: x [ A := x ] CocoaSk class >> a [ ^A ] CocoaSk class >> k: x [ K := x ] CocoaSk class >> k [ ^K ] ]")
            .expect("holder");
        vm.exec("CocoaSk reset.").expect("reset");
        vm.exec("CocoaSk a: (Cocoa action: [ CocoaSk bump ]).")
            .expect("action");
        let (w0, r0, _) = crate::runtime::objc_bridge::counters();
        for i in 0..10 {
            // 12 mints per scope: growth from 8 â†’ 16 slots mid-scope.
            vm.exec("CocoaSk k: (Cocoa poolDo: [:p | 1 to: 11 do: [:j | Cocoa nsString: 'churn' ]. p keep: (Cocoa nsString: 'kept') ]).")
                .expect("pool scope under stress");
            vm.exec("CocoaSk a macvmFire: nil.").expect("fire");
            vm.exec("Worker dispatchInbox.").expect("dispatch");
            let k = vm.eval("CocoaSk k sendString: 'description'").unwrap();
            assert!(k.contains("kept"), "iteration {i}: got {k}");
            vm.exec("CocoaSk k release.").expect("tidy the survivor");
        }
        assert_eq!(vm.eval("CocoaSk n").unwrap().trim(), "10");
        let (w1, r1, _) = crate::runtime::objc_bridge::counters();
        assert_eq!(w1 - w0, 120, "12 mints Ã— 10 scopes");
        assert_eq!(
            r1 - r0,
            120,
            "11 swept per scope + the kept one released after â€” balanced"
        );
    }

    #[test]
    #[cfg(target_os = "macos")] // WINVM: drives the real Cocoa bridge
    fn cocoa_c4_error_in_pool_scope_clears_the_stack() {
        let _serial = cocoa_serial();
        // The C4 review's F1: a doit that raises INSIDE poolDo: aborts with
        // the scope still pushed â€” the recovery arm must clear the stack,
        // or every future mint (anywhere) appends to a stale rooted list
        // forever.
        let mut vm = boot_test_vm(JitMode::Off);
        assert!(
            vm.exec("Cocoa poolDo: [:p | (Cocoa nsString: 'doomed'). Cocoa error: 'boom' ].")
                .is_err(),
            "the block's error must abort the doit"
        );
        vm.exec("Object subclass: CocoaEr [ <classVars: S> CocoaEr class >> s: x [ S := x ] CocoaEr class >> s [ ^S ] ]")
            .expect("holder");
        let (_, r0, _) = crate::runtime::objc_bridge::counters();
        // A mint OUTSIDE any scope after the abortâ€¦
        vm.exec("CocoaEr s: (Cocoa nsString: 'free agent').")
            .expect("mint outside any scope");
        // â€¦must survive a subsequent balanced poolDo: untouched.
        vm.exec("Cocoa poolDo: [:p | Cocoa nsString: 'swept' ].")
            .expect("a later balanced scope");
        let (_, r1, _) = crate::runtime::objc_bridge::counters();
        assert_eq!(r1 - r0, 1, "only the in-scope mint was swept");
        assert_eq!(
            vm.eval("CocoaEr s isValid").unwrap().trim(),
            "true",
            "the free agent must not have been swept into a stale scope"
        );
        vm.exec("CocoaEr s release.").expect("tidy");
    }

    #[test]
    fn cocoa_c5_cocoapad_fails_cleanly_headless() {
        let _serial = cocoa_serial();
        // The C5 demo class loads everywhere; headless (no AppKit linked,
        // no main run loop) its launch must raise cleanly â€” never hang or
        // crash. On-screen behavior is verified in the GUI (run-gui.sh).
        let mut vm = boot_test_vm(JitMode::Off);
        // The launch's own Smalltalk prerequisites must exist even where
        // AppKit doesn't â€” the on-screen run found Array's 4-element
        // constructor missing (the frame rectangles), invisible headless
        // because the NSWindow lookup fails first. Pin it directly.
        assert_eq!(
            vm.eval("(Array with: 1 with: 2 with: 3 with: 4) at: 4")
                .unwrap()
                .trim(),
            "4"
        );
        assert!(
            vm.exec("CocoaPad launch.").is_err(),
            "headless launch must raise (no NSWindow class / no hop)"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
    }

    #[test]
    fn cocoa_c2_dnu_sends_survive_the_jit() {
        let _serial = cocoa_serial();
        // DNU sends from COMPILED callers: threshold-1 compiles the loop
        // method immediately, so the ObjcRef sends flow through the
        // compiled DNU path (S11 step 6's rt_dnu â†’ Message â†’ ObjcRef>>
        // doesNotUnderstand:) rather than the interpreter's.
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        vm.exec("Object subclass: CocoaJit [ <classVars: N> CocoaJit class >> n: x [ N := x ] CocoaJit class >> n [ ^N ] CocoaJit class >> sum [ | t | t := 0. 1 to: 50 do: [:i | t := t + CocoaJit n length ]. ^t ] ]")
            .expect("holder + hot loop");
        vm.exec("CocoaJit n: (Cocoa nsString: 'jitted').")
            .expect("receiver");
        assert_eq!(
            vm.eval("CocoaJit sum").expect("hot DNU loop").trim(),
            "300",
            "50 Ã— length('jitted'=6) through compiled DNU sends"
        );
        vm.exec("CocoaJit n release.").expect("tidy");
    }

    #[test]
    fn cocoa_c1_oversized_send_fails_cleanly() {
        let _serial = cocoa_serial();
        // 11 GPR-class arguments = 6 registers + 4 stack words + 1 too
        // many: the prim must FAIL (world fallback raises) rather than
        // overflow any buffer â€” the FFI arc's argv-overflow lesson,
        // re-gated at this entry point.
        let mut vm = boot_test_vm(JitMode::Off);
        assert!(
            vm.exec(
                "(Cocoa nsString: 'x') send: 'whatever:' args: #(1 2 3 4 5 6 7 8 9 10 11) ret: #id."
            )
            .is_err(),
            "an oversized call shape must raise cleanly"
        );
        assert_eq!(vm.eval("3 + 4").unwrap().trim(), "7");
    }

    #[test]
    fn set_transcript_routes_transcript_show_to_the_sink() {
        struct VecSink(Arc<Mutex<Vec<String>>>);
        impl TranscriptSink for VecSink {
            fn show(&mut self, text: &str) {
                self.0.lock().unwrap().push(text.to_string());
            }
        }

        let mut vm = boot_test_vm(JitMode::Off);
        let captured = Arc::new(Mutex::new(Vec::new()));
        vm.set_transcript(Box::new(VecSink(captured.clone())));
        vm.eval("Transcript show: 'hello from embed'.")
            .expect("Transcript show: must evaluate cleanly");
        let lines = captured.lock().unwrap();
        assert!(
            lines.iter().any(|l| l.contains("hello from embed")),
            "sink must have captured the Transcript show: text, got {lines:?}"
        );
    }

    #[test]
    fn boot_surfaces_a_bad_world_list_entry_as_err_not_a_process_exit() {
        let dir =
            std::env::temp_dir().join(format!("macvm_embed_bad_world_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("world.list"), "nonexistent_file.mst\n").unwrap();

        let result = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                jit: JitMode::Off,
                ..Default::default()
            },
            &dir,
        );
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            result.is_err(),
            "a world.list referencing a nonexistent file must fail boot(), not panic/exit"
        );
    }

    #[test]
    fn boot_with_no_world_list_at_all_still_succeeds() {
        let dir = std::env::temp_dir().join(format!("macvm_embed_no_world_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Deliberately no world.list written â€” matches load_world's own
        // Ok(false) "no world.list found" case, not an error.
        let result = VmHandle::boot(
            VmOptions {
                heap_mib: 64,
                jit: JitMode::Off,
                ..Default::default()
            },
            &dir,
        );
        std::fs::remove_dir_all(&dir).ok();
        assert!(result.is_ok(), "a missing world.list must not fail boot()");
    }

    /// The test that actually matters for the whole S21 safety model: a
    /// guest-fatal condition (here, an unhandled DNU â€” the base world's own
    /// `Object>>doesNotUnderstand:` routes to the `error:` primitive, one of
    /// the 8 `fatal_exit`-converted sites, S21 step 1) must terminate ONLY
    /// the worker thread `boot`/`eval` ran on, never the test process
    /// itself. Per Step 1a's validated finding, `.join()`/`.is_finished()`
    /// on a `pthread_exit`-terminated thread's `JoinHandle` panics/hangs â€”
    /// so this test (like the real GUI supervisor, S21 step 3) never calls
    /// either. It proves the thread died by a channel message NEVER
    /// arriving within a generous timeout (the sending half, moved into the
    /// crashing closure, is never dropped either â€” `pthread_exit` runs no
    /// `Drop` glue at all â€” so a real disconnect would never show up as
    /// `RecvTimeoutError::Disconnected`; a plain `Timeout` is the correct,
    /// only-possible signature of "that thread is gone"), then simply keeps
    /// running: the surrounding test binary process surviving to report a
    /// normal pass IS the proof the whole mechanism works.
    #[test]
    fn eval_fatal_condition_kills_only_the_worker_thread() {
        let (tx, rx) = mpsc::channel::<&'static str>();
        let handle = std::thread::spawn(move || {
            let mut vm = boot_test_vm(JitMode::Off);
            // DNU/`error:` no longer belong here â€” `raise_guest_fatal`
            // recovers those at `eval`'s own boundary now (see
            // `eval_dnu_recovers_as_runtime_error_and_vm_stays_usable`
            // below); this test exists to prove the *actually* fatal path
            // (the VM's own invariants/resources, not the guest program's
            // correctness) still correctly kills the thread. Unbounded
            // self-recursion exhausts `ProcessStack` ->
            // `interpreter::stack`'s own "process stack overflow" fatal
            // path -> `fatal_exit`, untouched by this fix.
            vm.eval("Object subclass: MacvmInfiniteRecursionProbe [ go [ ^self go ] ].")
                .expect("defining the recursive-probe class must succeed");
            tx.send("reached-pre-crash-checkpoint").unwrap();
            // Never returns if FatalMode::ExitThread correctly pthread_exits.
            let _ = vm.eval("MacvmInfiniteRecursionProbe new go.");
            // Only reachable if the thread survived the "fatal" condition â€”
            // itself exactly the bug this test exists to catch.
            tx.send("UNREACHABLE-thread-survived-a-fatal-condition")
                .unwrap();
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)),
            Ok("reached-pre-crash-checkpoint"),
            "worker thread must have booted and reached the crash trigger"
        );
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "worker thread must NOT have returned from a fatal eval â€” \
             it should have pthread_exit'd"
        );

        // Deliberately no handle.join()/is_finished() â€” see this test's own
        // doc comment and the module doc for why that would panic/hang on a
        // thread that called pthread_exit.
        drop(handle);

        // The strongest proof of all: this test process is still alive and
        // able to run more assertions after the fact.
        assert_eq!(2 + 2, 4);
    }

    /// The actual fix: an unhandled DNU used to be indistinguishable from a
    /// genuinely fatal condition (see the previous test's own history) â€”
    /// every everyday Workspace typo paid a full worker respawn, exactly
    /// the "any mistake kills the VM" experience real Smalltalk's own
    /// recoverable `doesNotUnderstand:` exists to avoid. Proves both
    /// halves: the failure surfaces as an ordinary `Err`, AND the same
    /// `VmHandle` keeps serving requests afterward â€” the second half is
    /// the one that actually matters; a DNU that merely fails to crash but
    /// leaves the VM unusable wouldn't be a real fix.
    /// The recovery must return the VM to its CLEAN idle state â€” not merely
    /// leave it "usable enough" to compute `6 * 7`. A guest-fatal `siglongjmp`
    /// skips every RAII `Drop`, so without `restore_after_guest_fatal` the
    /// aborted doit's frames stay on `vm.stack` and its open `HandleScope`s
    /// stay in the handle arena, and both LEAK AND ACCUMULATE across errors
    /// (measured: stack 0 -> 60 -> 79 slots, arena 0 -> 1 -> 2, per error).
    /// This pins the invariant the "recover into its VMapp or die" rule
    /// demands: after any recovered error the stack and arena are byte-for-byte
    /// back at the between-doits baseline, no matter how many errors fire.
    #[test]
    fn eval_recovers_to_a_clean_stack_and_arena_without_accumulating() {
        for jit in [JitMode::Off, JitMode::Threshold(1)] {
            let mut vm = boot_test_vm(jit);
            // The clean baseline: a normal doit, then read the idle watermark.
            vm.eval("3 + 4.").unwrap();
            let base_sp = vm.vm.stack.sp;
            let base_frame = vm.vm.stack.has_frame();
            let base_arena = vm.vm.handle_arena.len();

            // Fire a mix of erroring doits, several times over, checking after
            // each that the VM snapped back to the exact idle baseline.
            let bombs = [
                "3 thisSelectorDoesNotExistAnywhereInTheBaseWorld.",
                "(1 to: 5) inject: 0 into: [:a :b | a nope: b].",
                "nil foo.",
                "self error: 'boom'.",
            ];
            for round in 0..3 {
                for bomb in &bombs {
                    let _ = vm.eval(bomb); // Err expected; the point is the state after
                    assert_eq!(
                        vm.vm.stack.sp, base_sp,
                        "jit={jit:?} round={round} bomb={bomb:?}: stack sp leaked \
                         ({} vs base {base_sp})",
                        vm.vm.stack.sp
                    );
                    assert_eq!(
                        vm.vm.stack.has_frame(),
                        base_frame,
                        "jit={jit:?} bomb={bomb:?}: a frame was left active after recovery"
                    );
                    assert_eq!(
                        vm.vm.handle_arena.len(),
                        base_arena,
                        "jit={jit:?} round={round} bomb={bomb:?}: handle arena leaked \
                         ({} vs base {base_arena})",
                        vm.vm.handle_arena.len()
                    );
                }
            }

            // And the VM is genuinely healthy afterward, not just clean-looking.
            assert_eq!(vm.eval("6 * 7.").unwrap(), "42");
        }
    }

    /// The same clean-baseline invariant as the previous test, but for a
    /// recovered NATIVE fault (SIGSEGV via a bad `Alien` deref, the S20/S21
    /// mechanism) arriving through `eval`/`exec` â€” not through
    /// `dispatch_callback`, which always had its own restore. The native-fault
    /// arm of the six ordinary entry points used to return
    /// `Err(NativeFault)` WITHOUT `restore_after_guest_fatal`, so the aborted
    /// doit's frames/handle scopes/tier journal survived the `siglongjmp` and
    /// the NEXT eval's `snapshot_idle_baseline` captured the polluted state as
    /// the new "clean" watermark â€” baking the leak in and (with a stale tier
    /// link) arming a GC-walk panic. This pins the fix: after a recovered
    /// native fault the VM is byte-for-byte back at the idle baseline, across
    /// repeated faults, under both JIT modes, through both `eval` and `exec`.
    #[test]
    fn eval_recovers_to_a_clean_baseline_after_a_native_fault() {
        for jit in [JitMode::Off, JitMode::Threshold(1)] {
            let mut vm = boot_test_vm(jit);
            vm.eval("3 + 4.").unwrap();
            let base_sp = vm.vm.stack.sp;
            let base_frame = vm.vm.stack.has_frame();
            let base_arena = vm.vm.handle_arena.len();

            // A deref of unmapped low memory: a genuine SIGSEGV outside the
            // JIT code cache, recovered by the foreign-fault handler.
            let bomb = "(Alien forAddress: 8 size: 8) byteAt: 1.";
            for round in 0..3 {
                let err = vm.eval(bomb).expect_err("the bad deref must Err");
                assert!(
                    matches!(err, GuestError::NativeFault { .. }),
                    "jit={jit:?} round={round}: expected NativeFault, got {err:?}"
                );
                assert_eq!(
                    vm.vm.stack.sp, base_sp,
                    "jit={jit:?} round={round}: stack sp leaked after a native fault"
                );
                assert_eq!(
                    vm.vm.stack.has_frame(),
                    base_frame,
                    "jit={jit:?} round={round}: a frame was left active"
                );
                assert_eq!(
                    vm.vm.handle_arena.len(),
                    base_arena,
                    "jit={jit:?} round={round}: handle arena leaked"
                );
                assert!(
                    vm.vm.tier_links.is_empty() && vm.vm.compiled_depth == 0,
                    "jit={jit:?} round={round}: stale tier journal after recovery"
                );
                // `exec` shares the same arm; alternate it in.
                let err = vm.exec(bomb).expect_err("exec's bad deref must Err");
                assert!(matches!(err, GuestError::NativeFault { .. }));
                assert_eq!(vm.vm.stack.sp, base_sp, "exec leaked stack");
                assert_eq!(vm.vm.handle_arena.len(), base_arena, "exec leaked arena");
            }

            assert_eq!(vm.eval("6 * 7.").unwrap(), "42");
        }
    }

    /// FFI hardening (2026-07 review follow-up): every guest-reachable
    /// mistake in a hand-authored `<primitive: FFI â€¦>` pragma â€” a typo'd
    /// symbol name, an unsupported declared shape token, a Tier-2 selector
    /// pragma with no runtime yet â€” used to `panic!` in
    /// `dispatch_ffi_primitive`, taking the whole embedding host down for
    /// a Workspace-level error. They now raise GUEST fatals: the eval
    /// answers `Err` with the named cause, and the same VM keeps serving.
    /// (The old `#[should_panic]` gates in `runtime/ffi.rs` moved here â€”
    /// a bare test VM has no jmp slot and cannot observe the recovery.)
    #[test]
    fn ffi_guest_mistakes_recover_as_errors_not_host_panics() {
        let mut vm = boot_test_vm(JitMode::Off);

        // (1) A typo'd function name â€” the everyday case.
        vm.exec(
            "Object subclass: FfiTypo [ \
               FfiTypo class >> go [ <primitive: FFI function: #noSuchSymbolXyzzyQ ret: #g args: #()> ] ]",
        )
        .expect("the pragma compiles fine â€” the typo only surfaces at call time");
        let err = vm
            .eval("FfiTypo go.")
            .expect_err("a typo'd symbol must Err, not kill the host");
        assert!(
            format!("{err}").contains("no symbol named \"noSuchSymbolXyzzyQ\""),
            "the error must name the missing symbol, got: {err}"
        );
        assert_eq!(vm.eval("6 * 7.").unwrap(), "42", "the VM must keep serving");

        // (2) An unsupported return-shape token (struct/HFA, no trampoline).
        vm.exec(
            "Object subclass: FfiBadRet [ \
               FfiBadRet class >> go [ <primitive: FFI function: #getpid ret: #h4 args: #()> ] ]",
        )
        .expect("compiles");
        let err = vm.eval("FfiBadRet go.").expect_err("h4 must Err");
        assert!(
            format!("{err}").contains("unsupported return-shape token \"h4\""),
            "must name the token, got: {err}"
        );

        // (3) A Tier-2 (`selector:`) pragma â€” no runtime support yet.
        vm.exec(
            "Object subclass: FfiTier2 [ \
               frame [ <primitive: FFI selector: #frame class: #NSView ret: #h4> ] ]",
        )
        .expect("compiles");
        let err = vm.eval("FfiTier2 new frame.").expect_err("Tier 2 must Err");
        assert!(
            format!("{err}").contains("Tier 2"),
            "must name the missing tier, got: {err}"
        );

        // (4) An unsupported argument-shape token.
        vm.exec(
            "Object subclass: FfiBadArg [ \
               FfiBadArg class >> go: x [ <primitive: FFI function: #abs ret: #g args: #(s)> ] ]",
        )
        .expect("compiles");
        let err = vm.eval("FfiBadArg go: 3.").expect_err("arg token s must Err");
        assert!(
            format!("{err}").contains("unsupported argument-shape token \"s\""),
            "must name the token, got: {err}"
        );

        // (5) More than 8 same-class register args â€” once a SILENT no-op
        // (pre-A0), then briefly a loud limit error (A0), now genuinely
        // SUPPORTED by the A3 stack-spill tier: args 9+ pass on the stack,
        // and a callee that reads none of them (getpid) simply works. The
        // spill's correctness proper is pinned by ffi_stubs's own
        // ret_g_spills_stack_args_nine_and_beyond and the world suite's
        // matrix gates; this case just pins that dispatch ACCEPTS the
        // arity end to end.
        vm.exec(
            "Object subclass: FfiNineArgs [ \
               FfiNineArgs class >> a: p1 b: p2 c: p3 d: p4 e: p5 f: p6 g: p7 h: p8 i: p9 [ \
                 <primitive: FFI function: #getpid ret: #g args: #(g g g g g g g g g)> ] ]",
        )
        .expect("compiles");
        let pid = vm
            .eval("FfiNineArgs a: 1 b: 2 c: 3 d: 4 e: 5 f: 6 g: 7 h: 8 i: 9.")
            .expect("9 g args must now succeed through the stack-spill tier");
        assert!(
            pid.trim().parse::<i64>().is_ok_and(|p| p > 0),
            "getpid through a 9-arg binding must answer a real pid, got: {pid}"
        );

        // (6) A token-list/arity mismatch â€” authored independently, so a
        // 2-keyword selector over a 3-token list compiles fine and then
        // used to no-op silently (the exact authoring bug that no-opped
        // every vDSP kernel in world/61a's first draft). Must Err naming
        // both counts.
        vm.exec(
            "Object subclass: FfiArity [ \
               FfiArity class >> a: p1 b: p2 [ \
                 <primitive: FFI function: #getpid ret: #g args: #(g g g)> ] ]",
        )
        .expect("compiles â€” the mismatch only surfaces at call time");
        let err = vm
            .eval("FfiArity a: 1 b: 2.")
            .expect_err("a token/arity mismatch must Err");
        assert!(
            format!("{err}").contains("3 arg token(s)")
                && format!("{err}").contains("takes 2 argument(s)"),
            "must name both counts, got: {err}"
        );

        // After all four recovered guest fatals: still byte-for-byte alive.
        assert_eq!(vm.eval("3 + 4.").unwrap(), "7");
    }

    /// Mono-SUPER c2i staleness (2026-07 review, formerly filed-unfixed): a
    /// compiled method whose `super sel` target was INTERPRETED-ONLY at
    /// compile time links that site to a c2i adapter baking the ancestor
    /// MethodOop â€” and an ancestor reached only via super stays interpreted
    /// forever (the c2i compile escape hatch skips super sites by design),
    /// so the adapter is permanent. Redefining the ancestor installs a
    /// fresh MethodOop that key-selector invalidation never routes to the
    /// caller (its own key is its own selector; the adapter lives outside
    /// `code_table`). Before the fix the site ran the PRE-redefinition body
    /// forever; now `rt_interpret_call`'s super arm re-runs the compile's
    /// own `lookup(super_klass, sel)` on every c2i super call, exactly as
    /// `rt_resolve_send` already did for the nmethod super case.
    #[test]
    fn compiled_super_send_sees_ancestor_redefinition_through_c2i() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        // Two traps this test's own drafts fell into, kept as documentation:
        // the caller's selector (`probe`) must DIFFER from the super-sent
        // one (`tag`) â€” the classic `tag [ ^super tag ]` override shape
        // shares the selector, so selector-keyed invalidation flushes the
        // caller and heals it by coincidence â€” AND the ancestor body must
        // be big enough that the inliner declines it (a `^1` leaf gets
        // spliced, its `inline_deps` edge invalidates the caller, healed
        // again). The bug lives only in the NON-inlined, interpreted-only
        // super target: the c2i-adapter link with no dependency edge.
        vm.exec(
            "Object subclass: SupRedefA [ \
               tag [ | s | s := 0. 1 to: 3 do: [ :i | s := s + i ]. ^s ] ]",
        )
        .expect("ancestor (loopy body â€” non-inlinable, stays interpreted)");
        vm.exec("SupRedefA subclass: SupRedefB [ probe [ ^super tag + 10 ] ]")
            .expect("subclass with the super send under a different selector");
        // Warm: Threshold(1) compiles SupRedefB>>probe on its first
        // activation; subsequent calls run the COMPILED super site through
        // the ancestor's c2i adapter (the ancestor never compiles â€” no
        // ordinary sends ever reach it, and the c2i compile escape hatch
        // skips super sites).
        for _ in 0..3 {
            assert_eq!(vm.eval("SupRedefB new probe.").unwrap(), "16");
        }
        // Live-redefine the ancestor's method (the browser-Accept shape) â€”
        // same shape, different bound, still non-inlinable.
        vm.exec(
            "Object subclass: SupRedefA [ \
               tag [ | s | s := 0. 1 to: 4 do: [ :i | s := s + i ]. ^s ] ]",
        )
        .expect("redefine the ancestor");
        assert_eq!(
            vm.eval("SupRedefB new probe.").unwrap(),
            "20",
            "the compiled super site must dispatch the REDEFINED ancestor \
             body, not the c2i adapter's baked pre-redefinition oop"
        );
    }

    #[test]
    fn error_policy_defaults_to_resume_and_round_trips() {
        let mut vm = boot_test_vm(JitMode::Off);
        assert_eq!(vm.error_policy(), ErrorPolicy::Resume);
        vm.set_error_policy(ErrorPolicy::Die);
        assert_eq!(vm.error_policy(), ErrorPolicy::Die);
        vm.set_error_policy(ErrorPolicy::Resume);
        assert_eq!(vm.error_policy(), ErrorPolicy::Resume);
    }

    /// `ErrorPolicy::Die`: an unhandled guest error must TERMINATE the worker
    /// (throwaway-worker semantics), not recover it. Run on a dedicated thread
    /// with `FatalMode::ExitThread` so the `fatal_exit` is a `pthread_exit`
    /// that kills only that thread â€” never `process::exit`, which would take
    /// down the whole test binary. The thread signals "booted" before the
    /// error and "survived" after; under `Die` the second signal must never
    /// arrive (the thread is gone). `pthread_exit` runs no destructors, so the
    /// `Sender` is not dropped either â€” hence the follow-up is a TIMEOUT, not a
    /// disconnect. (`Resume`'s opposite behavior â€” the worker survives and
    /// stays usable â€” is covered by the sibling recovery tests.)
    #[test]
    fn error_policy_die_terminates_the_worker_on_an_unhandled_error() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<&'static str>();
        let _thread = std::thread::spawn(move || {
            let mut vm = boot_test_vm(JitMode::Off); // VmHandle::boot arms ExitThread
            vm.set_error_policy(ErrorPolicy::Die);
            tx.send("booted").unwrap();
            // A Resume VM returns Err here and continues to the next line; a Die
            // VM pthread_exits inside this call and never returns.
            let _ = vm.eval("3 thisSelectorDoesNotExistAnywhereInTheBaseWorld.");
            let _ = tx.send("survived"); // MUST be unreachable under Die
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_secs(30)).ok(),
            Some("booted"),
            "the worker thread must boot and arm before the error"
        );
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(msg) => panic!(
                "ErrorPolicy::Die did not terminate the worker â€” it ran past the error and sent {msg:?}"
            ),
            Err(mpsc::RecvTimeoutError::Timeout)
            | Err(mpsc::RecvTimeoutError::Disconnected) => { /* worker died: correct */ }
        }
        // Do NOT join `_thread`: a pthread_exited thread can't be joined
        // (JoinHandle::join would panic â€” see fatal_exit's own doc).
    }

    /// CG0 Deliverable 2 â€” the post-boot `ExitProcess` flip that a main-thread
    /// (UI worker) VM uses. This is the CHILD body: only runs its fatal work
    /// when re-invoked as a subprocess with the env var set; a normal
    /// `cargo test` run reaches it with the var UNSET and it is a harmless
    /// no-op. It boots a VM (which arms `ExitThread`), flips to
    /// `FatalMode::ExitProcess`, then triggers a genuine fatal (unbounded
    /// recursion -> `ProcessStack` overflow -> `fatal_exit(70)`). Under
    /// `ExitProcess` that reaches `std::process::exit(70)` and the WHOLE
    /// process exits 70; it must never return.
    #[test]
    fn cg0_exitprocess_child_body_do_not_run_directly() {
        if std::env::var("MACVM_CG0_EXITPROCESS_CHILD").is_err() {
            return; // Normal test run: this is a no-op; the parent drives it.
        }
        let mut vm = boot_test_vm(JitMode::Off); // VmHandle::boot arms ExitThread
        set_fatal_mode(FatalMode::ExitProcess); // the post-boot flip under test
        vm.eval("Object subclass: MacvmCg0ExitProcessProbe [ go [ ^self go ] ].")
            .expect("defining the recursive probe class must succeed");
        // Unbounded recursion -> process stack overflow -> fatal_exit(70).
        // Under ExitProcess this is std::process::exit(70), never a return.
        let _ = vm.eval("MacvmCg0ExitProcessProbe new go.");
        // Reachable ONLY if ExitProcess failed to exit the process. Exit 0 so
        // the parent's `code == Some(70)` assertion fails loudly.
        std::process::exit(0);
    }

    /// CG0 Deliverable 2 â€” the subprocess harness proving the mechanism. A VM
    /// booted then set to `FatalMode::ExitProcess` (the pattern a main-thread
    /// UI worker uses so a true fatal exits the process rather than
    /// `pthread_exit`ing the UI thread into a zombie) must, on a genuine fatal,
    /// exit the WHOLE process with the fatal code (70), not `pthread_exit` a
    /// single thread. Re-invokes this very test binary, filtered to the child
    /// body above, with the env var set, and asserts the child exited exactly
    /// 70 â€” precisely the `std::process::exit(70)` `ExitProcess` produces, and
    /// distinct from the buggy `ExitThread`-on-a-worker-thread path (a
    /// libtest-join panic / abort with a different code).
    #[test]
    fn set_fatal_mode_exit_process_makes_a_true_fatal_exit_the_whole_process() {
        use std::process::{Command, Stdio};
        let exe = std::env::current_exe().expect("current_exe for the subprocess re-invoke");
        let status = Command::new(exe)
            // A unique substring filter (NOT `--exact`, which would need the
            // full `embed::tests::â€¦` path) -> libtest runs only this one test.
            .arg("cg0_exitprocess_child_body_do_not_run_directly")
            .arg("--test-threads=1")
            .env("MACVM_CG0_EXITPROCESS_CHILD", "1")
            // Silence the child's "process stack overflow" report / dossier.
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("spawning the child test process must succeed");
        assert!(
            !status.success(),
            "under FatalMode::ExitProcess a true fatal must exit the process NONZERO, got {status:?}"
        );
        assert_eq!(
            status.code(),
            Some(70),
            "the child must exit via std::process::exit(70) (ExitProcess on the \
             process-stack-overflow fatal path), not pthread_exit a thread or exit 0; got {status:?}"
        );
    }

    #[test]
    fn eval_dnu_recovers_as_runtime_error_and_vm_stays_usable() {
        let mut vm = boot_test_vm(JitMode::Off);
        let err = vm
            .eval("3 thisSelectorDoesNotExistAnywhereInTheBaseWorld.")
            .expect_err("an unhandled DNU must surface as Err, not run to completion");
        match &err {
            GuestError::RuntimeError(msg) => assert!(
                msg.contains("thisSelectorDoesNotExistAnywhereInTheBaseWorld"),
                "message: {msg}"
            ),
            other => panic!("expected GuestError::RuntimeError, got {other:?}"),
        }
        let result = vm
            .eval("6 * 7.")
            .expect("VM must still be usable after a recovered DNU");
        assert_eq!(result, "42");
    }

    /// `error:`'s own doc comment: "has no proceed semantics in v1" â€” this
    /// only asserts it's recoverable at `eval`'s OWN boundary (abort this
    /// one doIt, VM stays usable for the next), not that the erroring
    /// computation itself can be resumed mid-flight.
    #[test]
    fn eval_error_colon_recovers_as_runtime_error_and_vm_stays_usable() {
        let mut vm = boot_test_vm(JitMode::Off);
        let err = vm
            .eval("3 error: 'boom'.")
            .expect_err("an unhandled error: must surface as Err, not run to completion");
        match &err {
            GuestError::RuntimeError(msg) => assert!(msg.contains("boom"), "message: {msg}"),
            other => panic!("expected GuestError::RuntimeError, got {other:?}"),
        }
        let result = vm
            .eval("6 * 7.")
            .expect("VM must still be usable after a recovered error:");
        assert_eq!(result, "42");
    }

    /// The riskiest part of the fix, tested directly rather than only by
    /// analogy to the already-proven native-fault case: `raise_guest_fatal`
    /// reuses `siglongjmp` specifically because it's already trusted to
    /// cross JIT-compiled frames soundly (never `catch_unwind` through
    /// them â€” this project's standing rule). This exercises that for real:
    /// `go` compiles under threshold=1, and its OWN send is what DNUs, so
    /// `go`'s COMPILED frame is still live on the stack when
    /// `dnu_fallback` fires (S11 step 6's "DNU... from compiled code"
    /// path) â€” not just an interpreter-only DNU.
    #[test]
    fn eval_dnu_from_a_compiled_caller_recovers_cleanly() {
        let mut vm = boot_test_vm(JitMode::Threshold(1));
        vm.eval(
            "Object subclass: MacvmDnuFromCompiledProbe [ \
                go [ ^3 thisSelectorDoesNotExistAnywhereInTheBaseWorld ] \
            ].",
        )
        .expect("defining the probe class must succeed");
        let err = vm
            .eval("MacvmDnuFromCompiledProbe new go.")
            .expect_err("a DNU reached through a compiled caller must still surface as Err");
        assert!(matches!(err, GuestError::RuntimeError(_)), "{err:?}");
        let result = vm
            .eval("6 * 7.")
            .expect("VM must still be usable after recovering through a compiled frame");
        assert_eq!(result, "42");
    }
}
