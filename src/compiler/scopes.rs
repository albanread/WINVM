//! Scope descriptors + PcDescs (`sprint_s13_detail.md` §Data structures) —
//! the compiler-side metadata that lets a compiled frame be turned back
//! into the interpreter frame(s) it stands for (deoptimization). This file
//! is steps 1-2: the LEB128 codec, the location/scope/safepoint types, and
//! the packed byte format with its `ScopeDescRecorder` producer and a
//! decode side. Pure and self-contained — no emit/regalloc integration yet
//! (step 3), no signal/frame machinery (that's `codecache::deopt_trap` /
//! `runtime::deopt`).
//!
//! Every value the metadata names lives EITHER in the nmethod's oop pool
//! (a `ConstPool` index — the single GC-visible home, so a moving GC keeps
//! it current for free) or in a canonical frame slot (`FrameSlot`, the S12
//! spill-slot convention) or is a bare constant. Heap oops NEVER appear raw
//! in the blob — the GC does not scan it.

use crate::codecache::nmethod::Nmethod;
use crate::compiler::ir::VReg;
use crate::compiler::regalloc::{Assignment, LiveInterval, PollSave};

// ── vreg → ValueLoc resolution (step 3b) ──────────────────────────────────

/// Where `vreg` physically lives at safepoint position `pos`, as deopt
/// metadata (`sprint_s13_detail.md` M3). The FRAME-slot half only:
///
/// - A vreg whose live interval spans `pos` (`start <= pos && end > pos`,
///   the SAME "survives across the safepoint" test `regalloc::
///   crosses_safepoint` uses) and is `Assignment::Spill(slot)` → its
///   canonical `FrameSlot(-8·(slot+1))` (S12's `emit::spill_offset`).
///   S12's spill-all invariant GUARANTEES every value that survives a
///   safepoint is spilled there, so a surviving value is always found here
///   — never left in a register (there is no `Register` `ValueLoc`).
/// - OR `(vreg, pos)` is an exact `extra_oop_live` fact (`RegallocResult::
///   extra_oop_live` — a deopt site's own recorded receiver/slots/stack
///   vreg, forced to a canonical slot WITHOUT widening the vreg's plain
///   interval; see `compute_intervals`'s own doc for why the widening was
///   unsound) → same `FrameSlot`, read off the SAME `assignment` any other
///   entry for this vreg carries (a vreg has exactly one, per-method-wide
///   `SpillSlot` once spilled — regalloc's monotonic, never-reused
///   assignment, D3.5 point 3 — so any interval naming this vreg has it).
/// - Otherwise the value is DEAD at this safepoint (its last use was at or
///   before `pos`, and no deopt site needs it here either) → `Nil`. Safe by
///   the same invariant: anything genuinely read at or after the resume
///   bci is live-across (organically OR via `extra_oop_live`) → spilled →
///   handled above, so a not-found vreg is exactly one whose materialized
///   value is never read before being overwritten.
///
/// Recorded per-safepoint (NOT a pc-independent per-scope location): the
/// linear-scan regalloc assigns a spill slot per INTERVAL, so a multi-def
/// temp has no single canonical slot — resolving at each `pos` against
/// that pos's own live interval is unambiguous. Constant vregs
/// (`ConstSmi`/`ConstPool`/`Nil` from the IR) are a separate layer (the
/// caller knows the IR's constant map); this handles only frame-resident
/// values.
///
/// **Regression note** (`cold_branch_recompile_spill_corruption.mst`,
/// found immediately after `extra_oop_live` was introduced): this function
/// has the EXACT SAME "does the interval span `pos`" shape
/// `oopmap::build_for_position` does, and was missed the first time
/// `extra_oop_live` replaced the old blanket-widening — a narrowed
/// interval made THIS resolve to `Nil` at the very safepoint it's most
/// needed (the trap itself), so the materializer pushed `Nil` in place of
/// the real value (here, `payload`) onto the reexecuted stack — `nil + 5`
/// instead of the real addition, which cascaded into DNU handling deeply
/// enough to overflow the native stack. `extra_oop_live` must be checked
/// EVERYWHERE `build_for_position`'s own interval check is, not just there.
/// F3c S1 wrapper: contexts where a poll save can never apply — OSR-map
/// building (OSR compiles full-pin, `regalloc`'s `is_osr` gate) and the
/// spill-resolution goldens. Real deopt-scope building goes through
/// [`resolve_frame_loc_in`] with the compile's `poll_saves`, or a
/// register-resident vreg at a LoopPoll would silently resolve `Nil`.
pub fn resolve_frame_loc(
    vreg: VReg,
    pos: u32,
    intervals: &[LiveInterval],
    extra_oop_live: &[(VReg, u32)],
) -> ValueLoc {
    resolve_frame_loc_in(vreg, pos, intervals, extra_oop_live, &[])
}

pub fn resolve_frame_loc_in(
    vreg: VReg,
    pos: u32,
    intervals: &[LiveInterval],
    extra_oop_live: &[(VReg, u32)],
    poll_saves: &[(u32, Vec<PollSave>)],
) -> ValueLoc {
    // F3c S1: a register-resident vreg at a POLL resolves to its SAVE slot —
    // the poll's slow path stored it there before `stub_poll` (the only
    // deopt-capable instruction on the path) runs, and GC root updates land
    // in the same slot, so slot semantics are identical to a spill's.
    if let Some((_, saves)) = poll_saves.iter().find(|(p, _)| *p == pos) {
        if let Some(s) = saves.iter().find(|s| s.vreg == vreg) {
            return ValueLoc::FrameSlot(-8 * (s.save_slot.0 as i32 + 1));
        }
    }
    for iv in intervals {
        if iv.vreg == vreg
            && (iv.start <= pos && iv.end > pos
                || extra_oop_live.iter().any(|&(v, p)| v == vreg && p == pos))
        {
            if let Some(Assignment::Spill(slot)) = iv.assignment {
                let off = -8 * (slot.0 as i32 + 1);
                // Float fast-path: an unboxed-f64 vreg's slot holds raw bits,
                // not an oop — the materializer must box, not adopt.
                if iv.is_fp {
                    return ValueLoc::DoubleSlot(off);
                }
                return ValueLoc::FrameSlot(off);
            }
        }
    }
    ValueLoc::Nil
}

// ── OSR map (S15 A2/A3) ───────────────────────────────────────────────────

/// Where one OSR-transferred value comes FROM in the interpreter frame
/// being replaced (S15 A3 step 3 reads these; SPEC §5.1 slot layout).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OsrSource {
    Receiver,
    /// Unified arg/temp slot index (SPEC §4.1 numbering).
    Slot(u16),
    /// Interpreter operand-stack element i at the loop header (0 = bottom).
    StackSlot(u16),
    /// The frame's heap Context oop — S24 L2 phase B: the OSR entry ADOPTS
    /// the interpreter frame's existing Context into the materialize-form
    /// nmethod's `method_ctx_vreg` spill home, BY IDENTITY (pre-OSR closures
    /// hold the same object at `copied[1]`; a later deopt hands it back
    /// unchanged; re-OSR re-adopts). Emitted for `has_ctx` materialize
    /// compiles only; the elided form declines OSR
    /// (`osr_declined_elided_ctx`).
    Context,
}

/// One transferred value: its interpreter-side source and the compiled
/// frame's canonical spill home (byte offset from FP) of the vreg that is
/// live-in at the OSR entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OsrSlot {
    pub src: OsrSource,
    pub dst_frame_off: i32,
}

/// The interpreter→compiled frame-conversion map, packed into the nmethod
/// alongside the scope-desc blob family (same LEB primitives). One nmethod
/// has AT MOST ONE OSR entry in v1 (the loop that triggered).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OsrMap {
    /// The root-scope loop-header bci this entry serves.
    pub osr_bci: u16,
    /// Code offset of the OSR entry block.
    pub entry_off: u32,
    /// Compiled frame size (in 8-byte words) the OSR prologue allocates —
    /// the same value the normal prologue uses.
    pub frame_words: u32,
    /// In buffer order: the packer (A3 step 3) reads sources in exactly
    /// this order, and the entry block's copy sequence consumes the buffer
    /// in the same order.
    pub slots: Vec<OsrSlot>,
}

impl OsrMap {
    /// Tag values for `OsrSource` (pinned — the blob is a persistent
    /// format the runtime unpacks).
    const TAG_RECEIVER: u64 = 0;
    const TAG_SLOT: u64 = 1;
    const TAG_STACK: u64 = 2;
    const TAG_CONTEXT: u64 = 3;

    pub fn pack(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_uleb(&mut out, self.osr_bci as u64);
        write_uleb(&mut out, self.entry_off as u64);
        write_uleb(&mut out, self.frame_words as u64);
        write_uleb(&mut out, self.slots.len() as u64);
        for s in &self.slots {
            match s.src {
                OsrSource::Receiver => write_uleb(&mut out, Self::TAG_RECEIVER),
                OsrSource::Slot(i) => {
                    write_uleb(&mut out, Self::TAG_SLOT);
                    write_uleb(&mut out, i as u64);
                }
                OsrSource::StackSlot(i) => {
                    write_uleb(&mut out, Self::TAG_STACK);
                    write_uleb(&mut out, i as u64);
                }
                OsrSource::Context => write_uleb(&mut out, Self::TAG_CONTEXT),
            }
            // dst offsets are negative (below FP) — zigzag via the same
            // scheme resolve_frame_loc's FrameSlot uses when packed in
            // scope blobs: encode magnitude of the always-negative offset.
            debug_assert!(s.dst_frame_off < 0, "spill homes live below FP");
            write_uleb(&mut out, (-s.dst_frame_off) as u64);
        }
        out
    }

    pub fn unpack(buf: &[u8]) -> OsrMap {
        let mut pos = 0usize;
        let osr_bci = read_uleb(buf, &mut pos) as u16;
        let entry_off = read_uleb(buf, &mut pos) as u32;
        let frame_words = read_uleb(buf, &mut pos) as u32;
        let n = read_uleb(buf, &mut pos) as usize;
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            let src = match read_uleb(buf, &mut pos) {
                Self::TAG_RECEIVER => OsrSource::Receiver,
                Self::TAG_SLOT => OsrSource::Slot(read_uleb(buf, &mut pos) as u16),
                Self::TAG_STACK => OsrSource::StackSlot(read_uleb(buf, &mut pos) as u16),
                Self::TAG_CONTEXT => OsrSource::Context,
                other => panic!("OsrMap::unpack: unknown source tag {other}"),
            };
            let dst_frame_off = -(read_uleb(buf, &mut pos) as i32);
            slots.push(OsrSlot { src, dst_frame_off });
        }
        OsrMap {
            osr_bci,
            entry_off,
            frame_words,
            slots,
        }
    }
}

// ── LEB128 ────────────────────────────────────────────────────────────────

/// Unsigned LEB128 (SPEC §12.1's named round-trip test): 7 payload bits per
/// byte, high bit = "more follow". Used for indices, counts, and bcis.
pub fn write_uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Reads a ULEB128 from `buf` at `*pos`, advancing `*pos` past it.
pub fn read_uleb(buf: &[u8], pos: &mut usize) -> u64 {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return result;
        }
        shift += 7;
        debug_assert!(shift < 64, "read_uleb: overlong encoding");
    }
}

/// Signed LEB128 — for frame-slot byte offsets (negative below FP) and
/// untagged smi constants. Sign-extends the final group.
pub fn write_sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift — sign propagates
        let sign_bit_set = byte & 0x40 != 0;
        if (v == 0 && !sign_bit_set) || (v == -1 && sign_bit_set) {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Reads an SLEB128 from `buf` at `*pos`, advancing `*pos`.
pub fn read_sleb(buf: &[u8], pos: &mut usize) -> i64 {
    let mut result = 0i64;
    let mut shift = 0u32;
    loop {
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            // Sign-extend if the sign bit of this last group is set.
            if shift < 64 && byte & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return result;
        }
        debug_assert!(shift < 64, "read_sleb: overlong encoding");
    }
}

// ── Location + scope types ────────────────────────────────────────────────

/// Where an interpreter-visible value lives in a compiled frame. **No
/// `Register` variant in v1** — S12 pins that every value named by deopt
/// metadata is in its canonical frame slot at every safepoint (emit
/// flushes dirty vregs to spill homes before recording). A live-register
/// variant (and Self's register-mask machinery) is deferred to a later
/// regalloc upgrade.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueLoc {
    /// Oop in the nmethod's literal/oop pool (GC keeps pool words current).
    ConstPool(u32),
    /// A small integer constant, stored untagged; re-tag on materialization.
    ConstSmi(i64),
    /// Frame slot at byte offset `off` from the compiled frame's FP
    /// (negative = below FP, the S12 spill-slot convention).
    FrameSlot(i32),
    /// Shorthand for the `nil_obj` well-known oop (very common).
    Nil,
    /// S14 step 7-IV-b: an ELIDED literal closure — the compiled code never
    /// allocated it (its block was spliced inline), but the interpreter frame
    /// being rebuilt needs the real thing (a `do:`-style callee's block-arg
    /// temp, or the guard-cold reexecute stack of the send that passed it).
    /// The payload is the CompiledBlock `MethodOop`'s pool index. The
    /// materializer allocates a fresh `BlockClosure { method: pool[ix],
    /// home_ref: (root frame fp, serial), copied[0]: root receiver,
    /// copied[1]: root Context iff captures_ctx }` — the block's home is
    /// always the compilation ROOT in v1 (a block whose home isn't the root
    /// never elides), and the root frame is fully built (incl. its M6
    /// Context) before any inlined frame's values are read.
    ElidedClosure(u32),
    /// Float fast-path (`docs/float_fastpath_design.md` B5): the frame slot
    /// at byte offset `off` holds a RAW `f64` bit pattern (an unboxed float
    /// temp/copy — `VRegInfo::is_fp`), NOT an oop. The materializer BOXES it
    /// (`alloc_double`) into the rebuilt interpreter frame; the oop map
    /// already marks the slot non-oop, so the GC never scans it.
    DoubleSlot(i32),
}

impl ValueLoc {
    fn write(self, out: &mut Vec<u8>) {
        match self {
            ValueLoc::ConstPool(ix) => {
                out.push(0);
                write_uleb(out, ix as u64);
            }
            ValueLoc::ConstSmi(v) => {
                out.push(1);
                write_sleb(out, v);
            }
            ValueLoc::FrameSlot(off) => {
                out.push(2);
                write_sleb(out, off as i64);
            }
            ValueLoc::Nil => out.push(3),
            ValueLoc::ElidedClosure(ix) => {
                out.push(4);
                write_uleb(out, ix as u64);
            }
            ValueLoc::DoubleSlot(off) => {
                out.push(5);
                write_sleb(out, off as i64);
            }
        }
    }

    fn read(buf: &[u8], pos: &mut usize) -> ValueLoc {
        let tag = buf[*pos];
        *pos += 1;
        match tag {
            0 => ValueLoc::ConstPool(read_uleb(buf, pos) as u32),
            1 => ValueLoc::ConstSmi(read_sleb(buf, pos)),
            2 => ValueLoc::FrameSlot(read_sleb(buf, pos) as i32),
            3 => ValueLoc::Nil,
            4 => ValueLoc::ElidedClosure(read_uleb(buf, pos) as u32),
            5 => ValueLoc::DoubleSlot(read_sleb(buf, pos) as i32),
            other => panic!("ValueLoc::read: bad tag {other}"),
        }
    }
}

/// The heap Context of a scope. `Elided` is designed NOW but never emitted
/// by S13 (S14 Context-elision emits it) — the packed format supports it
/// from day one so no metadata reshaping is needed later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtxLoc {
    /// `has_ctx == 0`.
    None,
    /// A real Context exists (compiled code allocated it at entry). Deopt
    /// reuses it as-is.
    Materialized(ValueLoc),
    /// S14: no Context exists; each captured temp's value has its own
    /// location. Deopt allocates a fresh Context and populates it.
    Elided { temps: Vec<ValueLoc> },
}

impl CtxLoc {
    /// The 2-bit `ctx_kind` for the scope flags byte.
    fn kind(&self) -> u8 {
        match self {
            CtxLoc::None => 0,
            CtxLoc::Materialized(_) => 1,
            CtxLoc::Elided { .. } => 2,
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            CtxLoc::None => {}
            CtxLoc::Materialized(loc) => loc.write(out),
            CtxLoc::Elided { temps } => {
                write_uleb(out, temps.len() as u64);
                for t in temps {
                    t.write(out);
                }
            }
        }
    }

    fn read(kind: u8, buf: &[u8], pos: &mut usize) -> CtxLoc {
        match kind {
            0 => CtxLoc::None,
            1 => CtxLoc::Materialized(ValueLoc::read(buf, pos)),
            2 => {
                let n = read_uleb(buf, pos);
                let temps = (0..n).map(|_| ValueLoc::read(buf, pos)).collect();
                CtxLoc::Elided { temps }
            }
            other => panic!("CtxLoc::read: bad ctx_kind {other}"),
        }
    }
}

pub type ScopeId = u32;

/// An inlined-scope link. S13 always emits `sender: None` (depth-1 chains);
/// the packed format carries arbitrary depth from day one (S14 inlining).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SenderLink {
    pub sender: ScopeId,
    /// bci of the inlined send in the sender.
    pub sender_bci: u16,
    /// Sender's operand stack BELOW the inlined send's receiver+args, frozen
    /// for the whole inlined extent. Receiver+args are NOT here — they are
    /// reconstructed from THIS scope's receiver/slots (frame overlap, §5.1).
    pub pending_stack: Vec<ValueLoc>,
}

/// One virtual (interpreter) frame — recorder-side unpacked form. Per-scope
/// locations are pc-INDEPENDENT: each interpreter entity has one canonical
/// frame slot for the whole compiled method. Only the operand stack varies
/// per safepoint, so it lives in `SafepointState`, not here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeDescData {
    /// `MethodOop`, via the nmethod oop pool (the compile-time method — a
    /// mid-activation redefinition completes under old code).
    pub method_pool_ix: u32,
    /// A `CompiledBlock` scope (S14).
    pub is_block: bool,
    pub sender: Option<SenderLink>,
    pub receiver: ValueLoc,
    /// Unified arg/temp slots `0..argc+ntemps` (SPEC §4.1 `push_temp`
    /// numbering), argc first.
    pub slots: Vec<ValueLoc>,
    pub ctx: CtxLoc,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SafepointKind {
    Call,
    LoopPoll,
    Alloc,
    UncommonTrap,
}

impl SafepointKind {
    fn to_bits(self) -> u8 {
        match self {
            SafepointKind::Call => 0,
            SafepointKind::LoopPoll => 1,
            SafepointKind::Alloc => 2,
            SafepointKind::UncommonTrap => 3,
        }
    }
    fn from_bits(b: u8) -> SafepointKind {
        match b {
            0 => SafepointKind::Call,
            1 => SafepointKind::LoopPoll,
            2 => SafepointKind::Alloc,
            3 => SafepointKind::UncommonTrap,
            other => panic!("SafepointKind::from_bits: bad kind {other}"),
        }
    }
}

/// The per-pc deopt state. `reexecute` is THE single source of truth for
/// operand-stack height (the classic deopt bug is getting this wrong):
/// `true` → resume by RE-EXECUTING the bytecode at `bci`, recorded stack is
/// the state BEFORE it with all inputs present; `false` → a call-return
/// site, resume AFTER the send at `bci`, recorded stack EXCLUDES the send's
/// popped receiver+args and the materializer pushes the incoming result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafepointState {
    /// Innermost scope at this pc.
    pub scope: ScopeId,
    pub bci: u16,
    pub kind: SafepointKind,
    pub reexecute: bool,
    /// Innermost scope's operand stack at this pc (outer scopes' stacks come
    /// from their `SenderLink.pending_stack` chains).
    pub stack: Vec<ValueLoc>,
}

/// Fixed-width, sorted by `code_off`, binary-searchable — stored in the
/// nmethod as a plain array. `code_off` KEY convention (asserted in
/// `record_site`): calls key on the RETURN-address offset (pc after the
/// `bl`); traps/polls key on the `brk`/poll instruction offset itself.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PcDesc {
    pub code_off: u32,
    /// Byte offset of the packed `SafepointState` within the scopes blob.
    pub site_off: u32,
}

impl PcDesc {
    /// Exact-match binary search over a `code_off`-sorted slice (a miss is a
    /// VM bug — every deopt-relevant pc was recorded at compile time): panics
    /// on a miss, never falls back to nearest (P1, same posture as S12's
    /// `oopmap_at`). Step 4 adds an `&Nmethod` wrapper over this core once
    /// the nmethod carries its own `deopt_pcdescs` (step 3).
    pub fn find(descs: &[PcDesc], code_off: u32) -> &PcDesc {
        match descs.binary_search_by_key(&code_off, |d| d.code_off) {
            Ok(idx) => &descs[idx],
            Err(_) => panic!(
                "PcDesc::find: no deopt PcDesc at code_off {code_off:#x} -- every \
                 deopt-relevant pc must be recorded at compile time"
            ),
        }
    }

    /// Step 4: the `&Nmethod` wrapper the sprint doc names — looks the pc up
    /// in the nmethod's own `deopt_pcdescs` (S13 step 3a's field, distinct
    /// from the S12 oopmap `pcdescs`). Thin delegation to [`Self::find`];
    /// same exact-match / panic-on-miss posture (P1), with the nmethod id in
    /// the message for a real deopt-time miss.
    pub fn find_in(nm: &Nmethod, code_off: u32) -> &PcDesc {
        match nm
            .deopt_pcdescs
            .binary_search_by_key(&code_off, |d| d.code_off)
        {
            Ok(idx) => &nm.deopt_pcdescs[idx],
            Err(_) => panic!(
                "PcDesc::find_in: nmethod {:?} has no deopt PcDesc at code_off {code_off:#x} \
                 -- every deopt-relevant pc must be recorded at compile time",
                nm.id
            ),
        }
    }
}

// ── ScopeDescRecorder — the compiler-side producer ────────────────────────

/// Accumulates scopes and per-site states during emit, then `pack`s them
/// into the blob + sorted PcDesc array the nmethod stores.
#[derive(Default)]
pub struct ScopeDescRecorder {
    scopes: Vec<ScopeDescData>,
    sites: Vec<(u32, SafepointState)>,
}

impl ScopeDescRecorder {
    pub fn new() -> ScopeDescRecorder {
        ScopeDescRecorder::default()
    }

    /// S13: exactly one call per compilation, `sender: None`. S14: once per
    /// inlined scope, with a `SenderLink` whose `sender` was begun earlier
    /// (so its byte offset is already known when this scope is packed).
    pub fn begin_scope(&mut self, data: ScopeDescData) -> ScopeId {
        if let Some(link) = &data.sender {
            debug_assert!(
                (link.sender as usize) < self.scopes.len(),
                "begin_scope: sender {} must be begun before its child",
                link.sender
            );
        }
        let id = self.scopes.len() as ScopeId;
        self.scopes.push(data);
        id
    }

    /// Emit stage, at each safepoint/trap. `code_off` follows `PcDesc`'s own
    /// key convention (asserted here against `state.kind`).
    pub fn record_site(&mut self, code_off: u32, state: SafepointState) {
        debug_assert!(
            (state.scope as usize) < self.scopes.len(),
            "record_site: scope {} not begun",
            state.scope
        );
        // P (PcDesc key convention): a Call site keys on its RETURN address,
        // so it must NOT also be `reexecute` (a re-execute site keys on the
        // instruction itself). Catches a code_off computed against the wrong
        // convention (a 4-byte `find` miss later).
        debug_assert!(
            !(matches!(state.kind, SafepointKind::Call) && state.reexecute),
            "record_site: a Call safepoint keys on its return address and cannot be reexecute"
        );
        self.sites.push((code_off, state));
    }

    /// Dedup (one record per scope, shared by all its safepoints) + pack;
    /// returns `(scopes blob, PcDesc array sorted by code_off)`.
    pub fn pack(self) -> (Vec<u8>, Vec<PcDesc>) {
        let mut blob = Vec::new();

        // Scope records first, in ScopeId order (a sender is always begun —
        // and therefore serialized — before its child, so `sender_scope_off`
        // is already known). Record each scope's byte offset for the
        // safepoint records and sender links to reference.
        let mut scope_offs: Vec<u32> = Vec::with_capacity(self.scopes.len());
        for scope in &self.scopes {
            let off = blob.len() as u32;
            scope_offs.push(off);
            write_scope_record(&mut blob, scope, &scope_offs);
        }

        // Safepoint records next; each site's byte offset becomes its
        // PcDesc.site_off.
        let mut pcdescs: Vec<PcDesc> = Vec::with_capacity(self.sites.len());
        for (code_off, state) in &self.sites {
            let site_off = blob.len() as u32;
            write_safepoint_record(&mut blob, state, &scope_offs);
            pcdescs.push(PcDesc {
                code_off: *code_off,
                site_off,
            });
        }

        pcdescs.sort_by_key(|d| d.code_off);
        (blob, pcdescs)
    }
}

fn write_scope_record(out: &mut Vec<u8>, scope: &ScopeDescData, scope_offs: &[u32]) {
    let has_sender = scope.sender.is_some();
    let flags = (has_sender as u8) | ((scope.is_block as u8) << 1) | (scope.ctx.kind() << 2);
    out.push(flags);
    write_uleb(out, scope.method_pool_ix as u64);
    if let Some(link) = &scope.sender {
        write_uleb(out, scope_offs[link.sender as usize] as u64);
        write_uleb(out, link.sender_bci as u64);
        write_uleb(out, link.pending_stack.len() as u64);
        for v in &link.pending_stack {
            v.write(out);
        }
    }
    scope.receiver.write(out);
    write_uleb(out, scope.slots.len() as u64);
    for v in &scope.slots {
        v.write(out);
    }
    scope.ctx.write(out);
}

fn write_safepoint_record(out: &mut Vec<u8>, state: &SafepointState, scope_offs: &[u32]) {
    write_uleb(out, scope_offs[state.scope as usize] as u64);
    write_uleb(out, state.bci as u64);
    let flags = (state.reexecute as u8) | (state.kind.to_bits() << 1);
    out.push(flags);
    write_uleb(out, state.stack.len() as u64);
    for v in &state.stack {
        v.write(out);
    }
}

// ── Decode side ───────────────────────────────────────────────────────────

/// A scope record decoded from the blob at a byte offset. The `sender`
/// here carries the sender's own BYTE OFFSET (not a `ScopeId` — the packed
/// form is offset-linked); `runtime::deopt`'s chain walk follows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedScope {
    pub method_pool_ix: u32,
    pub is_block: bool,
    /// `(sender_scope_off, sender_bci, pending_stack)`.
    pub sender: Option<(u32, u16, Vec<ValueLoc>)>,
    pub receiver: ValueLoc,
    pub slots: Vec<ValueLoc>,
    pub ctx: CtxLoc,
}

/// A safepoint record decoded from the blob (`PcDesc.site_off`). `scope_off`
/// is the byte offset of the innermost scope record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedSite {
    pub scope_off: u32,
    pub bci: u16,
    pub kind: SafepointKind,
    pub reexecute: bool,
    pub stack: Vec<ValueLoc>,
}

/// Decodes the scope record at `off`.
pub fn decode_scope(blob: &[u8], off: u32) -> DecodedScope {
    let mut pos = off as usize;
    let flags = blob[pos];
    pos += 1;
    let has_sender = flags & 1 != 0;
    let is_block = flags & 2 != 0;
    let ctx_kind = (flags >> 2) & 0b11;
    let method_pool_ix = read_uleb(blob, &mut pos) as u32;
    let sender = if has_sender {
        let sender_off = read_uleb(blob, &mut pos) as u32;
        let sender_bci = read_uleb(blob, &mut pos) as u16;
        let n = read_uleb(blob, &mut pos);
        let pending = (0..n).map(|_| ValueLoc::read(blob, &mut pos)).collect();
        Some((sender_off, sender_bci, pending))
    } else {
        None
    };
    let receiver = ValueLoc::read(blob, &mut pos);
    let n_slots = read_uleb(blob, &mut pos);
    let slots = (0..n_slots)
        .map(|_| ValueLoc::read(blob, &mut pos))
        .collect();
    let ctx = CtxLoc::read(ctx_kind, blob, &mut pos);
    DecodedScope {
        method_pool_ix,
        is_block,
        sender,
        receiver,
        slots,
        ctx,
    }
}

/// Decodes the safepoint record at `site_off`.
pub fn decode_site(blob: &[u8], site_off: u32) -> DecodedSite {
    let mut pos = site_off as usize;
    let scope_off = read_uleb(blob, &mut pos) as u32;
    let bci = read_uleb(blob, &mut pos) as u16;
    let flags = blob[pos];
    pos += 1;
    let reexecute = flags & 1 != 0;
    let kind = SafepointKind::from_bits((flags >> 1) & 0b111);
    let n_stack = read_uleb(blob, &mut pos);
    let stack = (0..n_stack)
        .map(|_| ValueLoc::read(blob, &mut pos))
        .collect();
    DecodedSite {
        scope_off,
        bci,
        kind,
        reexecute,
        stack,
    }
}

// ── ScopeDesc::at — the deopt-time lookup entry point (step 4) ─────────────

/// The decoded deopt state at one native pc: the innermost safepoint record
/// (`site` — bci/kind/reexecute + the innermost operand stack) plus a
/// lazily-walked chain of scope records (innermost → outermost, following
/// `SenderLink` byte offsets). S13 chains are always depth-1; the walk
/// handles arbitrary depth from day one (S14 inlining). The materializer
/// (step 6) drives this: `site.reexecute` decides stack height, `scopes()`
/// yields one interpreter frame to rebuild per scope.
///
/// Borrows the nmethod's scopes blob for its lifetime; holds no decoded
/// scope eagerly (each `scopes()` step decodes on demand).
pub struct DeoptState<'a> {
    blob: &'a [u8],
    pub site: DecodedSite,
}

impl<'a> DeoptState<'a> {
    /// Resolve the deopt state at `code_off` (a native pc minus the code
    /// base) within `nm`. Panics if `code_off` is not a recorded deopt pc
    /// (P1 — same posture as [`PcDesc::find_in`]).
    pub fn at(nm: &'a Nmethod, code_off: u32) -> DeoptState<'a> {
        Self::at_parts(&nm.deopt_scopes, &nm.deopt_pcdescs, code_off)
    }

    /// The `&Nmethod`-free core (blob + PcDesc slice), so the decode/chain
    /// logic is unit-testable straight off `ScopeDescRecorder::pack` without
    /// standing up a whole `Nmethod`.
    pub fn at_parts(blob: &'a [u8], pcdescs: &[PcDesc], code_off: u32) -> DeoptState<'a> {
        let pc = PcDesc::find(pcdescs, code_off);
        let site = decode_site(blob, pc.site_off);
        DeoptState { blob, site }
    }

    /// The innermost scope — the one the safepoint record names directly.
    pub fn innermost(&self) -> DecodedScope {
        decode_scope(self.blob, self.site.scope_off)
    }

    /// Scope records innermost → outermost, following `SenderLink` byte-offset
    /// chains. S13 yields exactly one (depth-1); S14 inlining yields the full
    /// chain. Decodes each scope on demand.
    pub fn scopes(&self) -> ScopeChainIter<'a> {
        ScopeChainIter {
            blob: self.blob,
            next_off: Some(self.site.scope_off),
        }
    }
}

/// Innermost → outermost scope-record walk (see [`DeoptState::scopes`]).
pub struct ScopeChainIter<'a> {
    blob: &'a [u8],
    next_off: Option<u32>,
}

impl Iterator for ScopeChainIter<'_> {
    type Item = DecodedScope;
    fn next(&mut self) -> Option<DecodedScope> {
        let off = self.next_off?;
        let scope = decode_scope(self.blob, off);
        // `sender` carries the sender scope record's own byte offset (the
        // packed form is offset-linked) — follow it, or stop at the root.
        self.next_off = scope.sender.as_ref().map(|(sender_off, _, _)| *sender_off);
        Some(scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osr_map_round_trip() {
        let m = OsrMap {
            osr_bci: 7,
            entry_off: 0x1a4,
            frame_words: 12,
            slots: vec![
                OsrSlot {
                    src: OsrSource::Receiver,
                    dst_frame_off: -8,
                },
                OsrSlot {
                    src: OsrSource::Slot(3),
                    dst_frame_off: -32,
                },
                OsrSlot {
                    src: OsrSource::StackSlot(0),
                    dst_frame_off: -48,
                },
                OsrSlot {
                    src: OsrSource::Context,
                    dst_frame_off: -160,
                },
            ],
        };
        let packed = m.pack();
        assert_eq!(OsrMap::unpack(&packed), m, "pack/unpack round-trip");
    }

    #[test]
    fn osr_map_empty_slots() {
        let m = OsrMap {
            osr_bci: 0,
            entry_off: 0,
            frame_words: 2,
            slots: Vec::new(),
        };
        assert_eq!(OsrMap::unpack(&m.pack()), m);
    }

    fn uleb_roundtrip(v: u64) {
        let mut out = Vec::new();
        write_uleb(&mut out, v);
        let mut pos = 0;
        assert_eq!(read_uleb(&out, &mut pos), v, "uleb {v}");
        assert_eq!(pos, out.len(), "uleb {v} consumed exactly");
    }

    fn sleb_roundtrip(v: i64) {
        let mut out = Vec::new();
        write_sleb(&mut out, v);
        let mut pos = 0;
        assert_eq!(read_sleb(&out, &mut pos), v, "sleb {v}");
        assert_eq!(pos, out.len(), "sleb {v} consumed exactly");
    }

    /// SPEC §12.1's named LEB128 round-trip test: boundaries at each 7-bit
    /// group edge, plus extremes.
    #[test]
    fn leb128_roundtrip() {
        for &v in &[
            0u64,
            1,
            0x7f,
            0x80,
            0x3fff,
            0x4000,
            127,
            128,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ] {
            uleb_roundtrip(v);
        }
        for &v in &[
            0i64,
            1,
            -1,
            63,
            64,
            -64,
            -65,
            0x3f,
            -0x40,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ] {
            sleb_roundtrip(v);
        }
    }

    #[test]
    fn value_loc_roundtrip() {
        for loc in [
            ValueLoc::ConstPool(0),
            ValueLoc::ConstPool(300),
            ValueLoc::ConstSmi(0),
            ValueLoc::ConstSmi(-99999),
            ValueLoc::ConstSmi(123456789),
            ValueLoc::FrameSlot(-8),
            ValueLoc::FrameSlot(-480),
            ValueLoc::FrameSlot(16),
            ValueLoc::Nil,
            ValueLoc::ElidedClosure(0),
            ValueLoc::ElidedClosure(777),
        ] {
            let mut out = Vec::new();
            loc.write(&mut out);
            let mut pos = 0;
            assert_eq!(ValueLoc::read(&out, &mut pos), loc);
            assert_eq!(pos, out.len());
        }
    }

    /// A full recorder → pack → decode round-trip: one depth-1 scope (S13
    /// shape) with a materialized context, plus two safepoints (a call and
    /// a re-execute trap) that share the one scope record.
    #[test]
    fn recorder_pack_decode_roundtrip() {
        let mut rec = ScopeDescRecorder::new();
        let scope_data = ScopeDescData {
            method_pool_ix: 7,
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-8),
            slots: vec![
                ValueLoc::FrameSlot(-16), // arg 0
                ValueLoc::ConstSmi(42),   // temp holding a constant
                ValueLoc::Nil,            // temp = nil
            ],
            ctx: CtxLoc::Materialized(ValueLoc::FrameSlot(-24)),
        };
        let scope = rec.begin_scope(scope_data.clone());

        // A call-return site (reexecute=false, stack excludes popped args).
        rec.record_site(
            0x40,
            SafepointState {
                scope,
                bci: 10,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![ValueLoc::FrameSlot(-8)],
            },
        );
        // An uncommon-trap site (reexecute=true, all inputs on the stack).
        rec.record_site(
            0x20,
            SafepointState {
                scope,
                bci: 4,
                kind: SafepointKind::UncommonTrap,
                reexecute: true,
                stack: vec![ValueLoc::FrameSlot(-16), ValueLoc::ConstSmi(1)],
            },
        );

        let (blob, pcdescs) = rec.pack();

        // PcDescs sorted by code_off.
        assert_eq!(pcdescs.len(), 2);
        assert_eq!(pcdescs[0].code_off, 0x20);
        assert_eq!(pcdescs[1].code_off, 0x40);

        // Both safepoints must reference the SAME (deduplicated) scope record.
        let site_trap = decode_site(&blob, pcdescs[0].site_off);
        let site_call = decode_site(&blob, pcdescs[1].site_off);
        assert_eq!(
            site_trap.scope_off, site_call.scope_off,
            "one shared scope record"
        );

        assert_eq!(site_call.bci, 10);
        assert_eq!(site_call.kind, SafepointKind::Call);
        assert!(!site_call.reexecute);
        assert_eq!(site_call.stack, vec![ValueLoc::FrameSlot(-8)]);

        assert_eq!(site_trap.bci, 4);
        assert_eq!(site_trap.kind, SafepointKind::UncommonTrap);
        assert!(site_trap.reexecute);
        assert_eq!(
            site_trap.stack,
            vec![ValueLoc::FrameSlot(-16), ValueLoc::ConstSmi(1)]
        );

        // The shared scope record decodes back to what we recorded.
        let dec = decode_scope(&blob, site_call.scope_off);
        assert_eq!(dec.method_pool_ix, 7);
        assert!(!dec.is_block);
        assert_eq!(dec.sender, None);
        assert_eq!(dec.receiver, ValueLoc::FrameSlot(-8));
        assert_eq!(dec.slots, scope_data.slots);
        assert_eq!(dec.ctx, CtxLoc::Materialized(ValueLoc::FrameSlot(-24)));
    }

    /// The packed format carries a depth-2 inlined chain (S14 shape) with an
    /// elided context — S13 never emits this, but the format must round-trip
    /// it now so no reshaping is needed later.
    #[test]
    fn nested_scope_and_elided_ctx_roundtrip() {
        let mut rec = ScopeDescRecorder::new();
        let outer = rec.begin_scope(ScopeDescData {
            method_pool_ix: 1,
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-8),
            slots: vec![],
            ctx: CtxLoc::None,
        });
        let inner = rec.begin_scope(ScopeDescData {
            method_pool_ix: 2,
            is_block: true,
            sender: Some(SenderLink {
                sender: outer,
                sender_bci: 6,
                pending_stack: vec![ValueLoc::ConstSmi(5), ValueLoc::Nil],
            }),
            receiver: ValueLoc::ConstPool(3),
            slots: vec![ValueLoc::FrameSlot(-40)],
            ctx: CtxLoc::Elided {
                temps: vec![ValueLoc::FrameSlot(-48), ValueLoc::ConstSmi(-7)],
            },
        });
        rec.record_site(
            0x10,
            SafepointState {
                scope: inner,
                bci: 2,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![],
            },
        );
        let (blob, pcdescs) = rec.pack();

        let site = decode_site(&blob, pcdescs[0].site_off);
        let inner_dec = decode_scope(&blob, site.scope_off);
        assert!(inner_dec.is_block);
        assert_eq!(
            inner_dec.ctx,
            CtxLoc::Elided {
                temps: vec![ValueLoc::FrameSlot(-48), ValueLoc::ConstSmi(-7)]
            }
        );
        let (sender_off, sender_bci, pending) = inner_dec.sender.clone().expect("has sender");
        assert_eq!(sender_bci, 6);
        assert_eq!(pending, vec![ValueLoc::ConstSmi(5), ValueLoc::Nil]);

        // Follow the offset link to the outer scope record.
        let outer_dec = decode_scope(&blob, sender_off);
        assert_eq!(outer_dec.method_pool_ix, 1);
        assert_eq!(outer_dec.sender, None);
        assert_eq!(outer_dec.ctx, CtxLoc::None);
    }

    /// Step 4: `DeoptState::at_parts` resolves a pc to its safepoint record,
    /// and `.scopes()` walks the `SenderLink` chain innermost → outermost,
    /// terminating at the root. Uses a depth-2 chain (S14 shape) so the walk
    /// is exercised past S13's own depth-1 — a depth-1 real compile can't
    /// prove the iterator actually FOLLOWS a link and STOPS correctly.
    #[test]
    fn deopt_state_walks_scope_chain() {
        let mut rec = ScopeDescRecorder::new();
        let outer = rec.begin_scope(ScopeDescData {
            method_pool_ix: 11,
            is_block: false,
            sender: None,
            receiver: ValueLoc::FrameSlot(-8),
            slots: vec![ValueLoc::FrameSlot(-16)],
            ctx: CtxLoc::None,
        });
        let inner = rec.begin_scope(ScopeDescData {
            method_pool_ix: 22,
            is_block: true,
            sender: Some(SenderLink {
                sender: outer,
                sender_bci: 9,
                pending_stack: vec![ValueLoc::ConstSmi(3)],
            }),
            receiver: ValueLoc::FrameSlot(-24),
            slots: vec![ValueLoc::Nil],
            ctx: CtxLoc::None,
        });
        rec.record_site(
            0x80,
            SafepointState {
                scope: inner,
                bci: 7,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![ValueLoc::FrameSlot(-24)],
            },
        );
        let (blob, pcdescs) = rec.pack();

        let ds = DeoptState::at_parts(&blob, &pcdescs, 0x80);
        // The safepoint record itself.
        assert_eq!(ds.site.bci, 7);
        assert_eq!(ds.site.kind, SafepointKind::Call);
        assert!(!ds.site.reexecute);
        assert_eq!(ds.site.stack, vec![ValueLoc::FrameSlot(-24)]);
        // Innermost scope names the block activation.
        assert_eq!(ds.innermost().method_pool_ix, 22);

        // The chain: innermost (block, method 22) then its sender (method 11),
        // then STOP — exactly two frames, in that order.
        let chain: Vec<_> = ds.scopes().collect();
        assert_eq!(chain.len(), 2, "depth-2 chain yields exactly two scopes");
        assert_eq!(chain[0].method_pool_ix, 22);
        assert!(chain[0].is_block);
        assert_eq!(chain[0].receiver, ValueLoc::FrameSlot(-24));
        assert_eq!(chain[1].method_pool_ix, 11);
        assert!(!chain[1].is_block);
        assert_eq!(chain[1].sender, None, "outermost has no sender");
    }

    /// A pc that was never recorded is a VM bug, not a soft miss — P1 panic
    /// (mirrors `PcDesc::find`'s own posture, exercised via the parts path).
    #[test]
    #[should_panic(expected = "no deopt PcDesc")]
    fn deopt_state_panics_on_unrecorded_pc() {
        let mut rec = ScopeDescRecorder::new();
        let s = rec.begin_scope(ScopeDescData {
            method_pool_ix: 0,
            is_block: false,
            sender: None,
            receiver: ValueLoc::Nil,
            slots: vec![],
            ctx: CtxLoc::None,
        });
        rec.record_site(
            0x40,
            SafepointState {
                scope: s,
                bci: 0,
                kind: SafepointKind::Call,
                reexecute: false,
                stack: vec![],
            },
        );
        let (blob, pcdescs) = rec.pack();
        // 0x41 was never recorded.
        let _ = DeoptState::at_parts(&blob, &pcdescs, 0x41);
    }
}
