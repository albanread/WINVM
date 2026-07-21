//! Address-space reservation (SPEC §7.1): one `mmap` reservation,
//! `PROT_NONE`, committed lazily with `mprotect` — the MacNCL `Backing`
//! pattern (`sprint_s07_detail.md` §Design). `commit`/`decommit` are
//! idempotent: `mprotect`/`madvise` are themselves idempotent at the OS
//! level, so no separate range-bookkeeping structure is needed for
//! correctness (re-committing an already-committed range, or decommitting
//! an already-decommitted one, is just a repeat syscall — never an error).
//! [`test_heap`](Reservation::test_heap) swaps the mmap for a plain
//! `Box<[u8]>` for deterministic, allocation-cheap unit tests: no real
//! `mmap`/`mprotect` calls, `commit`/`decommit` are no-ops (a `Box<[u8]>`
//! is already fully readable/writable and there is nothing meaningful to
//! protect).

use std::ffi::c_void;

// WINVM: kernel32 address-space primitives, declared directly (the `libc`
// crate exposes only CRT functions on Windows, not Win32). Same shape as
// JASM's `native.rs`. `MEM_DECOMMIT`+re-`MEM_COMMIT` returns zeroed pages,
// which satisfies (exceeds) the "contents unspecified" contract below.
#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_DECOMMIT: u32 = 0x4000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_NOACCESS: u32 = 0x01;
    pub const PAGE_READWRITE: u32 = 0x04;

    #[repr(C)]
    pub struct SystemInfo {
        pub processor_arch: u16,
        pub reserved: u16,
        pub page_size: u32,
        pub min_app_addr: *mut c_void,
        pub max_app_addr: *mut c_void,
        pub active_processor_mask: usize,
        pub number_of_processors: u32,
        pub processor_type: u32,
        pub allocation_granularity: u32,
        pub processor_level: u16,
        pub processor_revision: u16,
    }

    extern "system" {
        pub fn VirtualAlloc(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn GetSystemInfo(info: *mut SystemInfo);
    }
}

/// A single `mmap` address-space reservation, OR (in test mode) a plain
/// heap allocation standing in for one. Uncommitted pages are `PROT_NONE`
/// and touching them faults; [`commit`](Reservation::commit) makes a
/// sub-range read/write.
pub struct Reservation {
    base: usize,
    size: usize,
    /// `Some` in `test_heap` mode: owns the backing memory (dropped
    /// normally by Rust), and `commit`/`decommit` become no-ops.
    test_box: Option<Box<[u8]>>,
}

impl Reservation {
    /// Reserve `size` bytes of address space (`PROT_NONE`,
    /// `MAP_PRIVATE|MAP_ANON`). `size` is rounded up to the system page size.
    /// Calls `fatal_exit` (not a panic — an environment limit, not a VM bug)
    /// if the OS refuses the reservation: the whole process for the CLI/
    /// tests, or just the calling thread for an embedded `VmHandle` (S21) —
    /// see `runtime::vm_state::fatal_exit`'s own doc.
    #[cfg(unix)]
    pub fn reserve(size: usize) -> Reservation {
        let size = round_up(size, page_size());
        // SAFETY: a fixed-shape mmap call with a null hint address (the
        // kernel picks the base) and no file descriptor; the result is
        // checked for MAP_FAILED before use.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            eprintln!(
                "macvm: failed to reserve {size} bytes: {}",
                std::io::Error::last_os_error()
            );
            crate::runtime::vm_state::fatal_exit(71);
        }
        Reservation {
            base: ptr as usize,
            size,
            test_box: None,
        }
    }

    /// WINVM: `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` — reserved-but-
    /// uncommitted pages fault on touch, the exact `PROT_NONE` semantics
    /// the unix path gets from `mmap`.
    #[cfg(windows)]
    pub fn reserve(size: usize) -> Reservation {
        let size = round_up(size, page_size());
        // SAFETY: null hint (kernel picks the base), size page-rounded;
        // result checked for null before use.
        let ptr = unsafe {
            win::VirtualAlloc(std::ptr::null_mut(), size, win::MEM_RESERVE, win::PAGE_NOACCESS)
        };
        if ptr.is_null() {
            eprintln!(
                "macvm: failed to reserve {size} bytes: {}",
                std::io::Error::last_os_error()
            );
            crate::runtime::vm_state::fatal_exit(71);
        }
        Reservation {
            base: ptr as usize,
            size,
            test_box: None,
        }
    }

    /// A deterministic small heap for unit tests: a plain `Box<[u8]>`
    /// (8-byte aligned via `vec![0u8; size]`'s allocator guarantee), no
    /// `mmap` involved. `commit`/`decommit` are no-ops — the memory is
    /// already fully accessible.
    pub fn test_heap(size: usize) -> Reservation {
        let size = round_up(size, 8);
        let mut v = vec![0u8; size].into_boxed_slice();
        let base = v.as_mut_ptr() as usize;
        Reservation {
            base,
            size,
            test_box: Some(v),
        }
    }

    /// Make `[off, off+len)` (page-rounded outward) readable and writable.
    /// Calls `fatal_exit` if `mprotect` fails (see `reserve`'s own doc). No-op
    /// in `test_heap` mode.
    pub fn commit(&self, off: usize, len: usize) {
        if self.test_box.is_some() {
            return;
        }
        let page = page_size();
        let start = round_down(off, page);
        let end = round_up(off.checked_add(len).expect("commit range overflow"), page);
        debug_assert!(
            end <= self.size,
            "commit range [{start},{end}) exceeds reservation of {} bytes",
            self.size
        );
        let ptr = (self.base + start) as *mut c_void;
        // SAFETY: `ptr` and `end - start` are within this reservation
        // (checked above); mprotect only changes protection, no memory is
        // read or written by this call itself.
        #[cfg(unix)]
        let ok = unsafe { libc::mprotect(ptr, end - start, libc::PROT_READ | libc::PROT_WRITE) } == 0;
        // WINVM: committing an already-committed page with the same
        // protection is an idempotent no-op, matching the unix contract.
        #[cfg(windows)]
        let ok = !unsafe {
            win::VirtualAlloc(ptr, end - start, win::MEM_COMMIT, win::PAGE_READWRITE)
        }
        .is_null();
        if !ok {
            eprintln!(
                "macvm: failed to commit {len} bytes at offset {off}: {}",
                std::io::Error::last_os_error()
            );
            crate::runtime::vm_state::fatal_exit(71);
        }
    }

    /// `madvise(MADV_FREE)` (the effective "give pages back" advice on
    /// macOS, not `MADV_DONTNEED`) then `mprotect(PROT_NONE)` over
    /// `[off, off+len)`, page-rounded outward. Reclaimed memory is NOT
    /// guaranteed zeroed on a later re-`commit` (lesson 5 — callers must
    /// never assume it). No-op in `test_heap` mode. Calls `fatal_exit` on
    /// failure, same policy as `commit`.
    pub fn decommit(&self, off: usize, len: usize) {
        if self.test_box.is_some() {
            return;
        }
        let page = page_size();
        let start = round_down(off, page);
        let end = round_up(off.checked_add(len).expect("decommit range overflow"), page);
        debug_assert!(
            end <= self.size,
            "decommit range [{start},{end}) exceeds reservation of {} bytes",
            self.size
        );
        let ptr = (self.base + start) as *mut c_void;
        // SAFETY: as `commit` — range is within this reservation; madvise
        // is advisory (never a correctness requirement) and mprotect only
        // changes protection.
        #[cfg(unix)]
        let ok = {
            unsafe {
                libc::madvise(ptr, end - start, libc::MADV_FREE);
            }
            unsafe { libc::mprotect(ptr, end - start, libc::PROT_NONE) == 0 }
        };
        // WINVM: MEM_DECOMMIT both returns the pages AND makes the range
        // fault on touch — one call covers madvise+mprotect(PROT_NONE).
        #[cfg(windows)]
        let ok = unsafe { win::VirtualFree(ptr, end - start, win::MEM_DECOMMIT) } != 0;
        if !ok {
            eprintln!(
                "macvm: failed to decommit {len} bytes at offset {off}: {}",
                std::io::Error::last_os_error()
            );
            crate::runtime::vm_state::fatal_exit(71);
        }
    }

    #[inline]
    pub fn base(&self) -> usize {
        self.base
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.test_box.is_some() {
            return; // Box drops itself normally — no mapping to release.
        }
        // SAFETY: `self.base`/`self.size` describe exactly the mapping
        // created in `reserve`; nothing else can alias it (Reservation is
        // not Clone).
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.base as *mut c_void, self.size);
        }
        // WINVM: MEM_RELEASE frees the whole reservation; size must be 0.
        #[cfg(windows)]
        unsafe {
            win::VirtualFree(self.base as *mut c_void, 0, win::MEM_RELEASE);
        }
    }
}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: sysconf with a valid, static name; no memory access.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[cfg(windows)]
fn page_size() -> usize {
    // SAFETY: GetSystemInfo writes the whole struct; no other memory access.
    unsafe {
        let mut info = std::mem::zeroed::<win::SystemInfo>();
        win::GetSystemInfo(&mut info);
        info.page_size as usize
    }
}

fn round_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

fn round_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_commit_rw() {
        let r = Reservation::reserve(64 << 20);
        r.commit(0, 1 << 20);

        let p0 = r.base() as *mut u64;
        let p_last = (r.base() + (1 << 20) - 8) as *mut u64;
        unsafe {
            p0.write(0xDEAD_BEEF_CAFE_0000);
            assert_eq!(p0.read(), 0xDEAD_BEEF_CAFE_0000);
            p_last.write(0x1234_5678_9ABC_DEF0);
            assert_eq!(p_last.read(), 0x1234_5678_9ABC_DEF0);
        }
    }

    #[test]
    fn commit_page_rounding() {
        let r = Reservation::reserve(16 << 20);
        r.commit(0, 4097);
        let p = (r.base() + 8100) as *mut u8;
        unsafe {
            p.write(42);
            assert_eq!(p.read(), 42);
        }
    }

    #[test]
    fn round_trip_helpers() {
        assert_eq!(round_up(0, 16), 0);
        assert_eq!(round_up(1, 16), 16);
        assert_eq!(round_up(16, 16), 16);
        assert_eq!(round_down(17, 16), 16);
        assert_eq!(round_down(16, 16), 16);
    }

    /// `tests_s07.md`: double `commit` of the same range succeeds; the
    /// range is still readable/writable afterwards (mprotect is naturally
    /// idempotent, but this pins the contract, not just the mechanism).
    #[test]
    fn reserve_commit_idempotent() {
        let r = Reservation::reserve(16 << 20);
        r.commit(0, 4096);
        r.commit(0, 4096);
        let p = r.base() as *mut u64;
        unsafe {
            p.write(0x1122_3344_5566_7788);
            assert_eq!(p.read(), 0x1122_3344_5566_7788);
        }
    }

    /// Apple Silicon pages are 16 KiB — committing a single byte must make
    /// the WHOLE surrounding 16 KiB page writable, not just a 4 KiB slice.
    /// WINVM: Windows x64 pages are 4 KiB, so writing at the 16 KiB offset
    /// after a 1-byte commit is genuinely uncommitted here — the assertion
    /// this test makes is macOS-page-size-specific.
    #[cfg(target_os = "macos")]
    #[test]
    fn commit_rounds_to_host_page() {
        let r = Reservation::reserve(1 << 20);
        r.commit(0, 1);
        let page = 16 << 10;
        let last_in_page = (r.base() + page - 8) as *mut u64;
        unsafe {
            last_in_page.write(0xAA);
            assert_eq!(last_in_page.read(), 0xAA);
        }
    }

    /// decommit → recommit → the range is readable/writable again
    /// (contents unspecified — lesson 5: reclaimed memory is never assumed
    /// zeroed).
    #[test]
    fn decommit_then_recommit() {
        let r = Reservation::reserve(16 << 20);
        r.commit(0, 4096);
        let p = r.base() as *mut u64;
        unsafe { p.write(0xDEAD) };
        r.decommit(0, 4096);
        r.commit(0, 4096);
        unsafe {
            p.write(0xBEEF);
            assert_eq!(p.read(), 0xBEEF);
        }
    }

    /// `test_heap` needs no `mmap`; `commit`/`decommit` are no-ops and the
    /// backing memory is immediately readable/writable.
    #[test]
    fn test_heap_deterministic() {
        let r = Reservation::test_heap(4096);
        // No commit() call at all — test_heap memory starts accessible.
        let p = r.base() as *mut u64;
        unsafe {
            p.write(0x42);
            assert_eq!(p.read(), 0x42);
        }
        r.commit(0, 4096); // no-op, must not panic or fault
        r.decommit(0, 4096); // no-op, must not panic or fault
        unsafe {
            assert_eq!(p.read(), 0x42); // decommit didn't actually reclaim it
        }
    }
}
