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
// NOTE (Phase 0/1): the encoder behind `build_if_needed` is still the
// vendored *AArch64* one — placed code is not executable on x64 and nothing
// may jump to it until the Phase-3 x64 backend replaces the encoder. The
// interpreter-only tier never does; publishing/patching are plain memory
// writes and remain valid to exercise.

#![cfg(windows)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::ptr;

use anyhow::{bail, Context, Result};

use crate::vendor::wfasm::backend::{EncodedModule, Reloc};
use crate::vendor::wfasm::relocpatch::{self, ResolvedReloc, VENEER_LEN};

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
        // SAFETY: fixed-shape allocation call, result checked before use.
        let region = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                cap,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };
        if region.is_null() {
            bail!("VirtualAlloc(PAGE_EXECUTE_READWRITE) failed — JIT memory unavailable");
        }
        Ok(WinJit {
            region: region as *mut u8,
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
        self.used = relocpatch::patch_relocs(code, region_base, self.used, &resolved)?;

        icache_invalidate(self.region as *const u8, self.cap);
        self.writable = false;
        self.finalized = true;
        Ok(())
    }

    /// Builder path: assemble accumulated text, reserve a region, place +
    /// relocate. Idempotent. (Still the vendored AArch64 encoder — see the
    /// module header; nothing may execute the result until Phase 3.)
    fn build_if_needed(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        if self.region.is_null() {
            let module = crate::vendor::wfasm::a64::assemble(&self.pending_text)
                .context("A64Encoder: assemble kernel text")?;
            // code + one veneer per extern + slack.
            let cap = module.code.len() + module.externs.len() * VENEER_LEN + PAGE;
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
    let lookup = |handle: *mut c_void| {
        // SAFETY: `handle` is a live module handle, `c_sym` a valid C string.
        let addr = unsafe { GetProcAddress(handle, c_sym.as_ptr()) };
        if addr.is_null() {
            None
        } else {
            Some(addr as u64)
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

    /// The Windows region + symbol/reloc bookkeeping round-trips: place a
    /// module, finalize, and read the placed bytes back. (No execution — the
    /// vendored encoder is still AArch64; execution tests arrive with the
    /// Phase-3 x64 backend.)
    #[test]
    fn place_and_readback() {
        use crate::vendor::wfasm::backend::Encoder;
        let m = crate::vendor::wfasm::a64::A64Encoder
            .encode(".globl entry\nentry:\nret\n")
            .expect("encode");
        let mut jit = WinJit::with_capacity(m.code.len() + PAGE).expect("VirtualAlloc");
        let base = jit.load_module(&m).expect("load");
        jit.finalize().expect("finalize");
        let sym = jit.lookup("entry").expect("symbol");
        assert_eq!(sym, base);
        // The placed bytes are exactly the encoded module's bytes.
        let placed = unsafe { std::slice::from_raw_parts(base as *const u8, m.code.len()) };
        assert_eq!(placed, &m.code[..]);
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
