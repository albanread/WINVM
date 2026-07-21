// WINVM: the Windows x86-64 sibling of `native_macos.rs`, following the shape
// of JASM's own `rust/src/native.rs` (`NativeJit`) — see MIGRATION.md §2.2.
// It provides exactly the surface MACVM consumes from the macOS side:
// `MacJit` (aliased from [`WinJit`] so `codecache` call sites stay identical),
// the free `jit_write_protect`/`icache_invalidate` W^X primitives, and
// `dlsym_resolve`.
//
// Memory model differences from macOS (all simplifications):
//
// * One region is `VirtualAlloc`'d `PAGE_EXECUTE_READWRITE` — Windows has no
//   `MAP_JIT`/per-thread write-protect toggle, so [`jit_write_protect`] is a
//   no-op and the region stays RWX for its lifetime. (A W^X-hygienic
//   RW↔RX `VirtualProtect` flip can replace this later without changing any
//   caller — the call sites already bracket writes correctly.)
// * x86 has coherent I/D caches; `FlushInstructionCache` is called anyway
//   (pro forma, and it is the documented contract for cross-modifying code).
//
// Phase 2 (MIGRATION.md §4): the builder path assembles with the vendored
// native x86-64 encoder (`rasm`) and patches with `relocpatch::
// patch_relocs_x64` — placed code EXECUTES on this machine (proven by this
// file's own tests). The compiler's tier-1 `emit.rs` still produces A64
// until Phase 3; nothing routes those blobs here for execution.

#![cfg(windows)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::ptr;

use anyhow::{bail, Context, Result};

use crate::vendor::wfasm::backend::{EncodedModule, Reloc};
use crate::vendor::wfasm::relocpatch::{self, ResolvedReloc, STUB_LEN_X64};

// ── kernel32 FFI ────────────────────────────────────────────────────────────

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;

extern "system" {
    fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn FlushInstructionCache(process: *mut c_void, base: *const c_void, size: usize) -> i32;
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// x64 Windows pages are 4 KiB (vs 16 KiB on Apple Silicon).
const PAGE: usize = 0x1000;

fn round_up(n: usize, to: usize) -> usize {
    (n + to - 1) & !(to - 1)
}

// ── Near-host allocation (MIGRATION.md §2.2) ────────────────────────────────
//
// Placing the code cache *anywhere* costs every call to a host runtime
// function an absolute veneer (`mov r10, [rip+pool]; call r10`, a load plus
// an indirect branch) instead of a plain `call rel32`. Since rel32 reaches
// ±2 GB, asking the OS for the region within a window of the host image
// removes that cost entirely — intra-cache calls AND host-extern calls both
// become direct 5-byte relative calls.
//
// `VirtualAlloc2` (Windows 10+) is what accepts an address requirement.
// It is resolved dynamically because it lives in kernelbase.dll and older
// systems lack it; when it or the near placement is unavailable, we fall
// back to placing the region anywhere and the absolute-veneer path in
// `relocpatch::patch_relocs_x64` handles the out-of-range targets. Nothing
// is *incorrect* in the fallback — it is just slower.

#[repr(C)]
struct MemAddressRequirements {
    lowest_starting_address: *mut c_void,
    highest_ending_address: *mut c_void,
    alignment: usize,
}

#[repr(C)]
struct MemExtendedParameter {
    type_and_reserved: u64,
    pointer: *mut c_void,
}

const MEM_EXTENDED_PARAMETER_ADDRESS_REQUIREMENTS: u64 = 1;

type VirtualAlloc2Fn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    usize,
    u32,
    u32,
    *mut MemExtendedParameter,
    u32,
) -> *mut c_void;

fn virtual_alloc2() -> Option<VirtualAlloc2Fn> {
    // SAFETY: string literals are NUL-terminated; every result is checked
    // for null before use, and the transmute matches the documented
    // signature of VirtualAlloc2.
    unsafe {
        let lib = {
            let l = GetModuleHandleA(b"kernelbase.dll\0".as_ptr() as *const c_char);
            if l.is_null() {
                LoadLibraryA(b"kernelbase.dll\0".as_ptr() as *const c_char)
            } else {
                l
            }
        };
        if lib.is_null() {
            return None;
        }
        let p = GetProcAddress(lib, b"VirtualAlloc2\0".as_ptr() as *const c_char);
        if p.is_null() {
            return None;
        }
        Some(std::mem::transmute::<*mut c_void, VirtualAlloc2Fn>(p))
    }
}

/// The half-width of the placement window. rel32 spans ±2 GB; 1.75 GB
/// leaves margin so a target a little beyond the anchor on the far side is
/// still reachable, rather than sitting exactly at the limit.
const NEAR_WINDOW: u64 = 0x7000_0000;

/// Try to reserve `size` RWX bytes within [`NEAR_WINDOW`] of `anchor`.
/// `None` if `VirtualAlloc2` is unavailable or the window is too crowded —
/// the caller falls back to placing the region anywhere.
fn alloc_near(anchor: u64, size: usize) -> Option<*mut u8> {
    let va2 = virtual_alloc2()?;
    const GRANULARITY: u64 = 0x10000;
    let low = (anchor.saturating_sub(NEAR_WINDOW) + GRANULARITY - 1) & !(GRANULARITY - 1);
    let low = low.max(GRANULARITY);
    let high = (anchor.saturating_add(NEAR_WINDOW) & !(GRANULARITY - 1)).saturating_sub(1);

    let mut req = MemAddressRequirements {
        lowest_starting_address: low as *mut c_void,
        highest_ending_address: high as *mut c_void,
        alignment: 0,
    };
    let mut param = MemExtendedParameter {
        type_and_reserved: MEM_EXTENDED_PARAMETER_ADDRESS_REQUIREMENTS,
        pointer: &mut req as *mut _ as *mut c_void,
    };
    // SAFETY: a fixed-shape allocation call; the extended parameter array
    // is one live, fully-initialized element. Result checked for null.
    let base = unsafe {
        va2(
            ptr::null_mut(),
            ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_EXECUTE_READWRITE,
            &mut param,
            1,
        )
    };
    if base.is_null() {
        None
    } else {
        Some(base as *mut u8)
    }
}

/// The anchor every code region is placed near: the address of a function
/// in this image. Any host symbol serves — they all live in the same
/// module — and using one of our own keeps the choice self-evident.
fn host_anchor() -> u64 {
    host_anchor as usize as u64
}

struct Placed {
    base: u64,
    relocs: Vec<Reloc>,
}

/// The Windows JIT region owner — field-for-field the same protocol as the
/// macOS `MacJit` (build/load/finalize, or `region_raw` for `CodeCache`'s
/// own segment management).
pub struct WinJit {
    region: *mut u8,
    cap: usize,
    used: usize,
    /// Defined symbol → absolute runtime address.
    symbols: HashMap<String, u64>,
    /// Host extern name → absolute address.
    externs: HashMap<String, u64>,
    placed: Vec<Placed>,
    finalized: bool,
    writable: bool,
    /// Accumulated text for the [`Loader`](crate::vendor::wfasm::backend::Loader) builder path.
    pending_text: String,
}

/// Alias so `codecache`'s `use …::native::MacJit` call sites are identical on
/// both platforms; the honest local name is [`WinJit`].
pub type MacJit = WinJit;

impl WinJit {
    /// Reserve `cap` bytes (rounded to a page) of RWX code space. "Writable"
    /// is bookkeeping only on Windows — the pages are always RWX.
    pub fn with_capacity(cap: usize) -> Result<Self> {
        let cap = round_up(cap.max(PAGE), PAGE);
        // Prefer a placement within rel32 of this image, so compiled code
        // reaches host runtime functions with a direct `call rel32` rather
        // than an absolute veneer (see the near-allocation section above).
        // Falling back to "anywhere" stays correct — `patch_relocs_x64`
        // routes any out-of-range branch through a stub — just slower.
        let region = match alloc_near(host_anchor(), cap) {
            Some(p) => p,
            None => {
                // SAFETY: fixed-shape allocation call, result checked.
                let p = unsafe {
                    VirtualAlloc(
                        ptr::null_mut(),
                        cap,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_EXECUTE_READWRITE,
                    )
                };
                p as *mut u8
            }
        };
        if region.is_null() {
            bail!("VirtualAlloc(PAGE_EXECUTE_READWRITE) failed — JIT memory unavailable");
        }
        Ok(WinJit {
            region,
            cap,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            writable: true,
            pending_text: String::new(),
        })
    }

    /// A builder-mode loader: accumulate text + externs, assemble + place on the
    /// first `lookup_addr`.
    pub fn new() -> Self {
        WinJit {
            region: ptr::null_mut(),
            cap: 0,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            writable: false,
            pending_text: String::new(),
        }
    }

    /// Bind a host extern (a Rust `extern "C"` function) by name.
    pub fn define_extern(&mut self, name: &str, addr: u64) {
        self.externs.insert(name.to_string(), addr);
    }

    /// Copy `m`'s code into the region (16-byte aligned), record its symbols at
    /// their final addresses, and queue its relocations. Returns the base.
    pub fn load_module(&mut self, m: &EncodedModule) -> Result<u64> {
        if self.finalized {
            bail!("WinJit already finalized");
        }
        debug_assert!(self.writable, "region must be writable to load");
        let start = round_up(self.used, 16);
        let end = start + m.code.len();
        if end > self.cap {
            bail!("code region exhausted: need {end} bytes, cap {}", self.cap);
        }
        let base = self.region as u64 + start as u64;
        // SAFETY: `[start, end)` is within the region (checked above);
        // source is `m.code`'s own live slice.
        unsafe { ptr::copy_nonoverlapping(m.code.as_ptr(), self.region.add(start), m.code.len()) };
        for (name, off) in &m.symbols {
            self.symbols.insert(name.clone(), base + *off as u64);
        }
        self.placed.push(Placed {
            base,
            relocs: m.relocs.clone(),
        });
        self.used = end;
        Ok(base)
    }

    fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols
            .get(name)
            .copied()
            .or_else(|| self.externs.get(name).copied())
    }

    /// Apply all relocations (building far-call veneers as needed) and flush
    /// the instruction cache. Idempotent. Same resolution logic as the macOS
    /// twin; the bit-patching lives in the shared `relocpatch`.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        let region_base = self.region as u64;
        let mut resolved: Vec<ResolvedReloc> = Vec::new();
        for p in &self.placed {
            let module_offset = (p.base - region_base) as usize;
            for r in &p.relocs {
                let target = self
                    .resolve(&r.target)
                    .with_context(|| format!("unresolved reloc target `{}`", r.target))?;
                resolved.push(ResolvedReloc {
                    field_offset: module_offset + r.at,
                    kind: r.kind,
                    target: (target as i64 + r.addend) as u64,
                });
            }
        }

        // SAFETY: the region is one live RWX allocation of exactly `cap`
        // bytes owned by `self`.
        let code = unsafe { std::slice::from_raw_parts_mut(self.region, self.cap) };
        self.used = relocpatch::patch_relocs_x64(code, region_base, self.used, &resolved)?;

        icache_invalidate(self.region as *const u8, self.cap);
        self.writable = false;
        self.finalized = true;
        Ok(())
    }

    /// Builder path: assemble accumulated text with the native x86-64
    /// encoder (`rasm`), reserve a region, place + relocate. Idempotent.
    fn build_if_needed(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        if self.region.is_null() {
            let module = crate::vendor::wfasm::rasm::assemble(&self.pending_text)
                .context("RasmEncoder: assemble text")?;
            // code + one far-branch stub per extern + slack.
            let cap = module.code.len() + module.externs.len() * STUB_LEN_X64 + PAGE;
            let externs = std::mem::take(&mut self.externs);
            *self = WinJit::with_capacity(cap)?;
            self.externs = externs;
            self.load_module(&module)?;
        }
        self.finalize()
    }

    /// Runtime address of a defined symbol (after [`finalize`]).
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    pub fn has_symbol(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }

    /// Expose the raw region so `CodeCache` can manage it directly (segment
    /// allocation/publish/patch), using this struct purely for the
    /// alloc/flush primitives — same contract as the macOS `region_raw`.
    pub fn region_raw(&self) -> (*mut u8, usize) {
        (self.region, self.cap)
    }
}

// ── W^X primitives (free fns, same names/signatures as the macOS twin) ─────

/// No-op on Windows: the region is permanently RWX (no per-thread toggle
/// exists). Kept so `JitWriteGuard`'s bracketing calls stay platform-neutral;
/// a `VirtualProtect`-based RW↔RX flip can be dropped in here later.
pub fn jit_write_protect(_exec: bool) {}

/// `FlushInstructionCache` over `[start, start+len)` — pro forma on coherent
/// x86, but it is the documented contract for cross-modifying code.
pub fn icache_invalidate(start: *const u8, len: usize) {
    // SAFETY: current-process pseudo-handle; the range is the caller's own
    // live code region. The call reads no memory through `start` itself.
    unsafe {
        FlushInstructionCache(GetCurrentProcess(), start as *const c_void, len);
    }
}

// ── dlsym_resolve (S20 FFI Tier 1) ─────────────────────────────────────────

/// The modules probed for a `lib: None` lookup — Windows has no
/// `RTLD_DEFAULT` "search everything already loaded", so Tier-1 resolution
/// (libc-ish names) probes the CRT then the core system DLLs, in order.
const DEFAULT_PROBE: &[&str] = &["ucrtbase.dll", "msvcrt.dll", "kernel32.dll"];

/// Resolve one native symbol to its absolute runtime address —
/// `LoadLibraryA` + `GetProcAddress`. `None` probes [`DEFAULT_PROBE`]
/// (the closest Windows analogue of the macOS `RTLD_DEFAULT` search).
/// `LoadLibraryA` on an already-loaded module is cheap (refcount + same
/// handle), matching the macOS twin's no-cache rationale.
pub fn dlsym_resolve(lib: Option<&str>, symbol: &str) -> Option<u64> {
    let c_sym = CString::new(symbol).ok()?;
    // The MSVC CRT exports POSIX names with a leading underscore
    // (`getpid` → `_getpid`, `open` → `_open`, …); resolve the plain name
    // first, then the underscore alias, so Tier-1 world bindings written
    // against POSIX names keep working unchanged.
    let c_sym_underscore = CString::new(format!("_{symbol}")).ok()?;
    let lookup = |handle: *mut c_void| {
        // SAFETY: `handle` is a live module handle, both names valid C strings.
        unsafe {
            let addr = GetProcAddress(handle, c_sym.as_ptr());
            let addr = if addr.is_null() {
                GetProcAddress(handle, c_sym_underscore.as_ptr())
            } else {
                addr
            };
            if addr.is_null() {
                None
            } else {
                Some(addr as u64)
            }
        }
    };
    match lib {
        Some(path) => {
            let c_path = CString::new(path).ok()?;
            // SAFETY: valid C string; result checked for null.
            let h = unsafe { LoadLibraryA(c_path.as_ptr()) };
            if h.is_null() {
                return None;
            }
            lookup(h)
        }
        None => DEFAULT_PROBE.iter().find_map(|m| {
            let c_mod = CString::new(*m).ok()?;
            // SAFETY: valid C string. GetModuleHandleA first (no refcount
            // bump for the always-loaded CRT/system DLLs), LoadLibraryA as
            // the fallback for a CRT flavor not yet mapped.
            let h = unsafe {
                let h = GetModuleHandleA(c_mod.as_ptr());
                if h.is_null() {
                    LoadLibraryA(c_mod.as_ptr())
                } else {
                    h
                }
            };
            if h.is_null() {
                return None;
            }
            lookup(h)
        }),
    }
}

impl Default for WinJit {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::vendor::wfasm::backend::Loader for WinJit {
    fn add_asm(&mut self, asm_text: &str) -> Result<()> {
        self.pending_text.push_str(asm_text);
        self.pending_text.push('\n');
        Ok(())
    }
    fn declare_fn(&mut self, _name: &str, _arg_count: usize) -> Result<()> {
        Ok(())
    }
    fn define_extern_fn(&mut self, name: &str, _arg_count: usize, addr: *mut c_void) -> Result<()> {
        self.externs.insert(name.to_string(), addr as u64);
        Ok(())
    }
    fn lookup_addr(&mut self, name: &str) -> Result<u64> {
        self.build_if_needed()?;
        self.symbols
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("windows loader: symbol `{name}` not found"))
    }
}

impl Drop for WinJit {
    fn drop(&mut self) {
        if !self.region.is_null() {
            // SAFETY: `region` is exactly the allocation from `with_capacity`;
            // nothing else aliases it (WinJit is not Clone). MEM_RELEASE
            // requires size 0.
            unsafe {
                VirtualFree(self.region as *mut c_void, 0, MEM_RELEASE);
            }
            self.region = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vendor::wfasm::backend::Loader;

    /// End-to-end: encode `42`, JIT it, run it. Proves the whole pipe —
    /// rasm → VirtualAlloc region → reloc patch → execute. The x64 twin of
    /// the macOS loader's own `leaf_executes`.
    #[test]
    fn leaf_executes() {
        let mut jit = WinJit::new();
        jit.add_asm(".globl entry\nentry:\n  mov rax, 42\n  ret\n")
            .expect("add_asm");
        let f: extern "C" fn() -> u64 =
            unsafe { jit.lookup_fn("entry").expect("lookup entry") };
        assert_eq!(f(), 42);
    }

    /// Internal `call` (BranchRel32 to a local label, patched in place):
    /// entry doubles via a leaf helper. Win64 ABI: arg in rcx, result rax.
    #[test]
    fn internal_call_executes() {
        let src = "\
.globl entry
entry:
  sub rsp, 40
  call dbl
  add rsp, 40
  ret
dbl:
  lea rax, [rcx + rcx]
  ret
";
        let mut jit = WinJit::new();
        jit.add_asm(src).expect("add_asm");
        let f: extern "C" fn(u64) -> u64 =
            unsafe { jit.lookup_fn("entry").expect("lookup entry") };
        assert_eq!(f(21), 42);
    }

    extern "C" fn host_inc(x: u64) -> u64 {
        x + 1
    }

    /// Host callback: JIT'd code `call`s a Rust `extern "C"` function bound
    /// as an extern — exercising the far-branch `movabs rax ; jmp rax` stub
    /// whenever the host lands outside ±2 GB of the region (and the direct
    /// rel32 when it doesn't; correct either way).
    #[test]
    fn host_callback_executes() {
        let src = "\
.globl entry
entry:
  sub rsp, 40
  call host_inc
  add rsp, 40
  add rax, rax
  ret
";
        let mut jit = WinJit::new();
        jit.define_extern_fn("host_inc", 1, host_inc as usize as *mut c_void)
            .expect("define extern");
        jit.add_asm(src).expect("add_asm");
        let f: extern "C" fn(u64) -> u64 =
            unsafe { jit.lookup_fn("entry").expect("lookup entry") };
        assert_eq!(f(10), 22, "(10+1)*2 via host callback");
        assert_eq!(f(0), 2);
    }

    /// The region must land within `rel32` of this image, so compiled code
    /// reaches host runtime functions with a direct `call rel32` instead of
    /// an absolute veneer (MIGRATION.md §2.2).
    ///
    /// Asserted against the FULL round trip a real call makes — from the
    /// far end of the region to the host anchor — not merely against the
    /// region base, because it is the worst-case distance that decides
    /// whether a veneer is needed.
    #[test]
    fn region_lands_within_rel32_of_the_host_image() {
        let cap = 1 << 20;
        let jit = WinJit::with_capacity(cap).expect("RWX region");
        let (base, actual_cap) = jit.region_raw();
        let anchor = host_anchor() as i64;
        let lo = base as i64;
        let hi = lo + actual_cap as i64;
        for end in [lo, hi] {
            let disp = anchor - end;
            assert!(
                i32::try_from(disp).is_ok(),
                "code region at {lo:#x}..{hi:#x} is {disp} bytes from the host anchor \
                 {anchor:#x} — outside rel32, so every host call would need a veneer"
            );
        }
    }

    /// A compiled `call rel32` straight to a host function, with no veneer
    /// — the payoff of near allocation, executed.
    #[test]
    fn direct_rel32_call_to_a_host_function_executes() {
        extern "C" fn host_triple(x: u64) -> u64 {
            x * 3
        }
        let mut jit = WinJit::new();
        jit.define_extern("host_triple", host_triple as usize as u64);
        jit.add_asm(
            ".globl entry\nentry:\n  sub rsp, 40\n  call host_triple\n  add rsp, 40\n  ret\n",
        )
        .expect("add_asm");
        let f: extern "C" fn(u64) -> u64 = unsafe { jit.lookup_fn("entry").expect("entry") };
        assert_eq!(f(14), 42);
    }

    #[test]
    fn dlsym_resolves_crt_and_kernel32() {
        // Tier-1 style: a CRT symbol with no explicit library...
        assert!(dlsym_resolve(None, "malloc").is_some());
        // ...an explicit-library symbol...
        assert!(dlsym_resolve(Some("kernel32.dll"), "GetTickCount64").is_some());
        // ...and a miss stays a clean None.
        assert!(dlsym_resolve(None, "definitely_not_a_symbol_xyzzy").is_none());
    }
}
