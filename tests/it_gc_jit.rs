//! Sprint S12 integration tests (`tests_s12.md`). This file is allowed
//! `unsafe` (it lives in `tests/`, a separate crate from `macvm` itself) --
//! not that anything here needs it directly; the walker/hook machinery
//! being exercised is `unsafe` internally, not this file's own code.

use macvm::bytecode::builder::BytecodeBuilder;
use macvm::codecache::nmethod::{IcSite, IcState, NmState, Nmethod, NmethodId, OopMap, PcDesc};
use macvm::compiler::assembler::RelocKind;
use macvm::compiler::driver;
use macvm::compiler::ir::{
    BailoutReason, BlockId, CallSiteInfo, CmpOp, Ir, IrBlock, IrMethod, PoolEntry, PoolLit, SmiOp,
    VReg, VRegInfo,
};
use macvm::compiler::jasm_assembler::JasmAssembler;
use macvm::compiler::oopmap;
use macvm::compiler::regalloc::{self, RegallocResult};
use macvm::compiler::{emit, emit::SafepointPc};
use macvm::frontend::{classdef, parser};
use macvm::interpreter::compiled_call::{enter_compiled, EnterResult};
use macvm::memory::alloc;
use macvm::memory::fullgc;
use macvm::memory::scavenge::scavenge;
use macvm::oops::layout::HEADER_WORDS;
use macvm::oops::mark::Mark;
use macvm::oops::smi::SmallInt;
use macvm::oops::wrappers::{KlassOop, MemOop};
use macvm::oops::{Format, Oop};
use macvm::runtime::lookup::{install_method, klass_of};
use macvm::runtime::{JitMode, VmOptions, VmState};

fn test_vm() -> VmState {
    VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        // A smaller-than-default eden (default is a meaningful fraction of
        // `heap_mib`) so this file's own filler loop (see
        // `compiled_frame_spill_slot_survives_forced_scavenge`) genuinely,
        // honestly tops it up with a reasonable number of allocations
        // rather than tens of thousands -- but still comfortably bigger
        // than class-creation's own genesis overhead (4 KiB turned out too
        // small: `Object subclass: AllocTarget [ ]` alone exhausted it).
        eden_kb: Some(256),
        jit: JitMode::Off,
    })
}

/// S12 D4.1 (`tests_s12.md`'s `mid_loop_forced_scavenge`, REDUCED — see
/// this test's own note below for why): proves `memory::roots::
/// each_code_root`'s new compiled-frame-spill-slot scanning finds and
/// correctly relocates a live oop held ONLY in a live compiled frame's own
/// spill slot, across a REAL scavenge. Forced directly
/// (`VmState::test_force_scavenge_in_alloc_slow`, wired into
/// `codecache::stubs::rt_alloc_slow`), bypassing only `scavenge`'s own
/// internal `debug_assert` via the narrow, default-off
/// `test_allow_moving_gc_under_compiled` flag — none of the S11 D8
/// bridge's three PRODUCTION doors (`alloc::alloc_words`'s diversion arm,
/// `prim_gc_scavenge`/`prim_gc_full`'s own `gc_pending` defer) are touched
/// at all, so ordinary Smalltalk code still cannot reach a moving
/// collector under a live compiled frame until step 7 opens all three
/// together (see `memory::scavenge::scavenge`'s own doc comment).
///
/// NOT literally `mid_loop_forced_scavenge` as `tests_s12.md` describes it
/// (a `T>>run:` loop sending the REAL `Gc scavenge` primitive 100 times,
/// asserting `gc_under_compiled >= 100`): that design needs
/// `prim_gc_scavenge`'s OWN defer-guard lifted too, which `alloc.rs`'s own
/// doc comment groups with the rest of the D8 bridge as one interlock,
/// deleted together in step 7 once full-GC integration (step 5) and
/// nmethod flushing (step 6) make opening every door safe. The full
/// loop-based golden (100 iterations, real `Gc scavenge` sends) is
/// deferred to step 7/8; this test proves the SAME underlying mechanism —
/// a real scavenge, a real live compiled frame, a real relocation — today,
/// via a direct collector call instead of a Smalltalk-level send.
#[test]
fn compiled_frame_spill_slot_survives_forced_scavenge() {
    let mut vm = test_vm();
    for item in parser::parse_file("Object subclass: AllocTarget [ ]").expect("parse") {
        classdef::execute_top_item(&mut vm, item).expect("execute");
    }

    // Tenure EVERYTHING now (the class, its metaclass, its method dict —
    // all fresh eden allocations from `execute_top_item` just above) BEFORE
    // deriving any Rust-local handle onto them: a `KlassOop` local
    // computed before this point would itself go stale the moment this
    // scavenge promotes/relocates it — the exact hazard this test's own
    // forced scavenge (below) exists to probe, just one step earlier than
    // intended. `it_tier1.rs`'s own `full_gc_updates_pool_word_and_key_
    // klass` test hits the same shape ("look up the FRESH post-GC value,
    // not a stale pre-GC local"); this sidesteps it by simply not HAVING a
    // pre-GC local yet. Old gen is never touched by a scavenge (`memory::
    // scavenge`'s own module doc), so once tenured, `target_klass` (etc.,
    // derived fresh right after) stays valid for the rest of this test —
    // nothing else scavenges until this test's own forced one, deliberately
    // isolated to `rt_alloc_slow`'s hook far below.
    vm.universe.tenuring_threshold = 0;
    scavenge(&mut vm).expect("scavenge must promote AllocTarget's klass into old gen");
    vm.universe.tenuring_threshold = 127;

    let target_sym = vm.universe.intern(b"AllocTarget");
    let target_assoc =
        macvm::runtime::globals::global_lookup(&vm, target_sym).expect("AllocTarget global");
    let target_klass =
        KlassOop::try_from(MemOop::try_from(target_assoc).unwrap().body_oop(1)).unwrap();

    // A bare `test_vm()` has no world loaded, so `basicNew` (a world
    // method) isn't installed -- install the real basicNew primitive (id
    // 23) on AllocTarget's own metaclass, mirroring `it_tier1.rs`'s own
    // `allocation_fast_and_slow`.
    let basic_new_sel = vm.universe.intern(b"basicNew");
    let target_meta = klass_of(&vm, target_klass.oop());
    let basic_new_method = {
        let mut nb = BytecodeBuilder::new();
        nb.ret_self(); // fallback body -- never reached (prim always succeeds here)
        let m = nb.finish(&mut vm, basic_new_sel, 0, 0);
        m.set_primitive(23);
        m.set_flags(0, 0, false, false, true, false, 0);
        m
    };
    install_method(&mut vm, target_meta, basic_new_sel, basic_new_method);

    // S12 D4.1's own eventual hazard (not this test's job to fix): once
    // step 7 removes the OTHER two D8 bridge doors, `rt_alloc_slow`'s
    // `klass_bits`/`size_bytes` parameters — copied out of the Alloc
    // site's own RootSpill slots into plain Rust locals BEFORE its own
    // `alloc_slots` call (`stub_alloc_slow`'s `mov x1,x0`/`mov x2,x1` run
    // AFTER `emit_stub_prologue`'s `stp` already parked the ORIGINAL
    // x0/x1 to RootSpill, so a native memory copy and a separate Rust
    // copy coexist) — would go stale if a real scavenge relocated the
    // klass INSIDE that call: `each_code_root` correctly updates the
    // NATIVE copy, but nothing re-derives `rt_alloc_slow`'s own Rust
    // locals from it afterward. Moot today (no caller reaches a real GC
    // from inside any of the six stub handlers before step 7), tracked in
    // this sprint's own STEP-4 NOTES for when it stops being moot. This
    // test sidesteps it by keeping `AllocTarget`'s klass tenured (above)
    // for its OWN different reason (a stale Rust-local `KlassOop` in the
    // TEST itself, not in `rt_alloc_slow`) — same underlying shape, worth
    // naming once rather than twice.

    // `keepSelfAlive [ AllocTarget basicNew. ^self ]` -- the alloc's own
    // result is discarded (popped); `self` (the receiver) is used again
    // AFTER it, so it must stay live across the Alloc safepoint. D1's
    // spill-all invariant then forces the register allocator to park it in
    // a spill slot -- exactly the scenario `each_code_root`'s new
    // compiled-frame path exists to scan.
    let mut b = BytecodeBuilder::new();
    b.push_global(&mut vm, target_assoc);
    b.send(&mut vm, basic_new_sel, 0);
    b.pop();
    b.push_self();
    b.ret_tos();
    let sel = vm.universe.intern(b"keepSelfAlive");
    let method = b.finish(&mut vm, sel, 0, 0);
    install_method(&mut vm, target_klass, sel, method);

    // Warm interpreted (mono the basicNew site) on a throwaway receiver --
    // never used again.
    let warm_recv = alloc::alloc_slots(&mut vm, target_klass).oop();
    let warm_result = macvm::interpreter::run_method(&mut vm, method, warm_recv, &[]);
    assert_eq!(
        klass_of(&vm, warm_result).oop().raw(),
        target_klass.oop().raw(),
        "warmup must produce a real AllocTarget"
    );

    assert!(
        driver::eligible(&vm, method),
        "a mono basicNew site is eligible"
    );
    let nm_id = driver::compile_method(&mut vm, target_klass, method).expect("must compile");

    // A FRESH, young receiver -- allocated AFTER the tenuring scavenge
    // above, so it lands in eden and genuinely needs relocating.
    let recv = alloc::alloc_slots(&mut vm, target_klass).oop();

    // Force the Alloc slow edge HONESTLY: top up eden with real, fully
    // valid throwaway instances until less than one more `AllocTarget`
    // fits, rather than lying about `eden.top` (`it_tier1.rs`'s own
    // `allocation_fast_and_slow` does exactly that lie -- `vm.universe.
    // eden.top = vm.universe.eden.end` -- but that test never scavenges,
    // so the gap of never-initialized memory between the real frontier and
    // the lied-about `top` never gets walked. THIS test's own forced
    // scavenge (below) walks the WHOLE heap up to `eden.top` as part of
    // `scavenge`'s own entry-verify (always on in debug builds,
    // `verify::verify_enabled`) -- found empirically the first time this
    // test ran, via a "malformed mark word" panic reading straight into
    // that uninitialized gap.
    let need = target_klass.non_indexable_size() * macvm::oops::layout::WORD_SIZE;
    while vm.universe.eden.end - vm.universe.eden.top >= need {
        alloc::alloc_slots(&mut vm, target_klass);
    }
    vm.test_force_scavenge_in_alloc_slow = true;
    vm.stack.push(recv);
    let gc_under_compiled_before = vm.universe.gc_stats.gc_under_compiled;
    assert_eq!(enter_compiled(&mut vm, nm_id, 0), EnterResult::Completed);

    let (before, after) = vm
        .test_scavenge_probe
        .take()
        .expect("rt_alloc_slow's hook must have fired")
        .expect("the forced scavenge must not have panicked");
    assert_eq!(
        before,
        recv.raw(),
        "the slot's bits before the scavenge must be the receiver this test pushed"
    );
    assert_ne!(
        before, after,
        "the compiled frame's own spill slot must show DIFFERENT bits after a real scavenge \
         relocated the young receiver held there"
    );

    let result = vm.stack.pop();
    assert_eq!(
        result.raw(),
        after,
        "the compiled method's OWN control flow, resuming after rt_alloc_slow returns and \
         reloading self from the (now-updated) spill slot, must observe the SAME relocated \
         address the hook already saw -- not just the hook's own isolated snapshot"
    );
    assert_eq!(
        klass_of(&vm, result).oop().raw(),
        target_klass.oop().raw(),
        "the relocated object must still be a valid, correctly-classed AllocTarget"
    );
    assert!(
        vm.universe.gc_stats.gc_under_compiled > gc_under_compiled_before,
        "the forced scavenge must have counted itself as running under a live compiled frame"
    );
}

/// S12 D5/D6 (`tests_s12.md`'s `key_klass_death_flushes`, REDUCED — see
/// this test's own note, and its companion
/// `compiled_mono_caller_guard_keeps_key_klass_alive` just below, for why):
/// a class with NO global binding and NO live instances, referenced ONLY
/// via a compiled method's own (weakly-treated) `key_klass` field, must
/// actually die on a real full GC — and that death must actually flush
/// the nmethod (gone from `code_table`, its own code-cache space
/// reusable).
///
/// NOT literally `key_klass_death_flushes` as `tests_s12.md` describes it
/// (ALSO driving a compiled caller mono to the dying method, and an
/// interpreter IC to it): tracing `CodeTable::update_keys`'s own existing,
/// correct behavior (D4.1's "everything else in pools stays STRONG" —
/// unconditional, not gated by the weak-key flag at all) shows that ANY
/// live Mono/PIC caller's own guard-klass mirror — or an interpreter IC's
/// own `guard()` oop, an ordinary strong heap field — would ALSO
/// reference the same dying klass, keeping it marked regardless of the
/// dying nmethod's own `key_klass` being correctly excluded. A klass can
/// only actually die once EVERYTHING that ever compared against it —
/// compiled guards, interpreter ICs, the customizing nmethod's own key —
/// has ALSO become unreachable, not just the one nmethod being tested.
/// This test proves the achievable core: the customization-only case.
/// `compiled_mono_caller_guard_keeps_key_klass_alive` proves the OTHER
/// half directly (a surviving caller genuinely blocks the death) — real
/// executed evidence for why the fuller scenario doesn't apply here,
/// documented in `sprint_s12_detail.md`'s own STEP-6 NOTES.
#[test]
fn key_klass_death_flushes() {
    let mut vm = test_vm();
    let object_klass = vm.universe.object_klass;
    let tmp_klass = vm
        .universe
        .new_klass(object_klass, "Tmp", Format::Slots, false, HEADER_WORDS);

    let hot_sel = vm.universe.intern(b"hot");
    let mut b = BytecodeBuilder::new();
    b.push_smi_i8(42);
    b.ret_tos();
    let hot_method = b.finish(&mut vm, hot_sel, 0, 0);
    install_method(&mut vm, tmp_klass, hot_sel, hot_method);

    // Warm interpreted (no sends in `hot`'s own body, so nothing to
    // resolve -- this just satisfies `compile_method`'s own established
    // "warm before compiling" idiom).
    let tmp_instance = alloc::alloc_slots(&mut vm, tmp_klass).oop();
    let warm = macvm::interpreter::run_method(&mut vm, hot_method, tmp_instance, &[]);
    assert_eq!(warm, SmallInt::new(42).oop());

    assert!(driver::eligible(&vm, hot_method), "hot must be eligible");
    let hot_id = driver::compile_method(&mut vm, tmp_klass, hot_method).expect("must compile");
    assert_eq!(
        vm.code_table.get(hot_id).unwrap().key_klass.oop().raw(),
        tmp_klass.oop().raw(),
        "hot's own nmethod must be customized (keyed) for Tmp"
    );

    // `tmp_instance`/`tmp_klass` are now Rust locals ONLY -- never pushed
    // onto `vm.stack`, never stored in a handle, no global binding was
    // ever created (`Universe::new_klass` doesn't install one, unlike
    // `Object subclass: ... [...]` -- confirmed directly from its own
    // source before relying on it here). Nothing but hot's own key_klass
    // field references Tmp at all by this point.
    fullgc::full_gc(&mut vm).expect("full gc must succeed");

    assert!(
        vm.code_table.get(hot_id).is_none(),
        "hot's own nmethod must have been flushed once Tmp became truly unreachable"
    );
    let reused = vm
        .code_cache
        .alloc(1)
        .expect("the code cache must still be able to allocate (no leak/corruption)");
    let _ = reused;
}

/// Companion to `key_klass_death_flushes` just above — real executed
/// proof of the finding that test's own doc explains: a LIVE compiled
/// Mono caller's own guard-klass mirror (kept unconditionally strong by
/// `CodeTable::update_keys`, D4.1's "everything else stays STRONG", NOT
/// gated by S12's weak-key flag at all) is a SEPARATE, independent root
/// for the same klass — so a class customizing a dying-looking nmethod
/// does NOT actually die while any such caller survives, regardless of
/// that nmethod's own `key_klass` being correctly excluded from marking.
///
/// `callHot`'s own send to `hot` is built as HAND-CONSTRUCTED `IrMethod`
/// (an `Ir::CallSend`), bypassing `driver::compile_method`'s own
/// eligibility gate entirely, exactly like `it_tier1.rs`'s own
/// `mono_resolve_patches_call_site_and_dispatches` — that gate's own D1
/// point 2 restricts a compiled `Send` to sites ALREADY interpreter-IC
/// Mono AND guarded on `SmallInteger` targeting an `SMI_INLINE`
/// primitive (a real, separate finding from writing this test: an
/// ordinary dynamic dispatch to a non-smi receiver, like `self hot` here,
/// can never pass `driver::eligible` at all today — S11's own
/// conservative gate, not something this test needed to work around by
/// accident).
#[test]
fn compiled_mono_caller_guard_keeps_key_klass_alive() {
    use macvm::codecache::nmethod::{IcSite, NmState, Nmethod, NmethodId};
    use macvm::codecache::stubs::CallStubFn;
    use macvm::compiler::emit;
    use macvm::compiler::ir::{
        BlockId, CallSiteInfo, Ir, IrBlock, IrMethod, PoolLit, VReg, VRegInfo,
    };
    use macvm::compiler::jasm_assembler::JasmAssembler;
    use macvm::compiler::regalloc;

    let mut vm = test_vm();
    let object_klass = vm.universe.object_klass;
    let tmp_klass = vm
        .universe
        .new_klass(object_klass, "Tmp", Format::Slots, false, HEADER_WORDS);

    let hot_sel = vm.universe.intern(b"hot");
    let mut hb = BytecodeBuilder::new();
    hb.push_smi_i8(42);
    hb.ret_tos();
    let hot_method = hb.finish(&mut vm, hot_sel, 0, 0);
    install_method(&mut vm, tmp_klass, hot_sel, hot_method);
    let tmp_instance = alloc::alloc_slots(&mut vm, tmp_klass).oop();
    let warm = macvm::interpreter::run_method(&mut vm, hot_method, tmp_instance, &[]);
    assert_eq!(warm, SmallInt::new(42).oop());
    assert!(driver::eligible(&vm, hot_method));
    let hot_id = driver::compile_method(&mut vm, tmp_klass, hot_method).expect("hot must compile");
    let hot_nm = vm.code_table.get(hot_id).unwrap();
    let hot_entry = unsafe { hot_nm.code.base.add(hot_nm.entry_off as usize) } as u64;

    // `callHot`: one param (the receiver), one send of `hot` against it --
    // self=x0 -- mirrors `mono_resolve_patches_call_site_and_dispatches`'s
    // own caller shape exactly.
    let vregs: Vec<VRegInfo> = (0..2).map(|_| VRegInfo { is_oop: true, is_fp: false }).collect();
    let block0 = IrBlock {
        id: BlockId(0),
        bci: 0,
        code: vec![
            Ir::Param {
                dst: VReg(0),
                index: 0,
            },
            Ir::CallSend {
                dst: VReg(1),
                site: 0,
                args: vec![VReg(0)],
            },
            Ir::Ret { val: VReg(1) },
        ],
        entry_stack: Vec::new(),
        deopt_sites: Vec::new(),
    };
    let call_hot_method_ir = IrMethod {
        osr_cold_sends: 0,
        blocks: vec![block0],
        vregs,
        pool: Vec::new(),
        argc: 1,
        ntemps: 0,
        ctx_vregs: Vec::new(),
        block_closure_vreg: None,
        method_ctx_vreg: None,
        spliced_nlr: 0,
        spliced_multibb: 0,
        splice_declined_budget: 0,
        safepoints: Vec::new(),
        true_lit: PoolLit(0),
        false_lit: PoolLit(0),
        nil_lit: PoolLit(0),
        mark_slots_lit: PoolLit(0),
        mark_double_lit: PoolLit(0),
        double_klass_lit: PoolLit(0),
        float64x2_klass_lit: PoolLit(0),
        float32x4_klass_lit: PoolLit(0),
        int32x4_klass_lit: PoolLit(0),
        call_sites: vec![CallSiteInfo {
            selector: hot_sel,
            argc: 0,
            static_klass: None,
        }],
        site_feedback: Vec::new(),
        inline_deps: Vec::new(),
        self_devirt: false,
        method_pool_ix: None,
    };
    let ra = regalloc::regalloc(&call_hot_method_ir);
    let mut asm = JasmAssembler::new();
    let (blob, _pcs, _verified_entry_off, emitted_ic_sites, _safepoints, _osr_off) = emit::emit(
        &mut asm,
        &call_hot_method_ir,
        &ra,
        vm.stubs.stub_poll_addr(),
        vm.stubs.must_be_boolean_addr(),
        vm.stubs.alloc_slow_addr(),
        vm.stubs.box_double_addr(),
        vm.stubs.box_float64x2_addr(),
        vm.stubs.box_float32x4_addr(),
        vm.stubs.box_int32x4_addr(),
        vm.stubs.call_primitive_addr(),
        vm.stubs.nlr_originate_addr(),
        None,
        None,
        None,
    );
    assert_eq!(emitted_ic_sites.len(), 1, "exactly one Ir::CallSend");

    let h = vm.code_cache.alloc(blob.code.len()).unwrap();
    vm.code_cache.publish(h, &blob);
    let resolve_addr = vm.stubs.resolve_addr();
    for site in &emitted_ic_sites {
        vm.code_cache.patch_branch26_at(h, site.off, resolve_addr);
    }
    let call_hot_sel = vm.universe.intern(b"callHotProbe");
    let ic_sites: Vec<IcSite> = emitted_ic_sites
        .iter()
        .map(|s| IcSite {
            off: s.off,
            selector: s.selector,
            argc: s.argc,
            state: IcState::Unresolved,
            super_klass: None,
        })
        .collect();
    let call_hot_nm = Nmethod {
        osr_cold_sends: 0,
        id: NmethodId(0),
        key_klass: tmp_klass,
        key_selector: call_hot_sel,
        code: h,
        entry_off: 0,
        verified_entry_off: 0,
        state: NmState::Alive,
        level: 1,
        version: 0,
        trap_count: 0,
        profile_hash: 0,
        literal_off: blob.literal_off,
        relocs: blob.relocs.clone(),
        frame_slots: ra.frame_slots,
        slot_is_oop: ra.slot_is_oop.clone(),
        pcdescs: Vec::new(),
        oopmaps: Vec::new(),
        ic_sites,
        poll_bci: None,
        prim_call_argc_plus_recv: None,
        block_method: None,
        deopt_scopes: Vec::new(),
        deopt_pcdescs: Vec::new(),
        inline_deps: Vec::new(),
        self_devirt: false,
        osr_map: None,
    };
    let call_hot_id = vm.code_table.install(call_hot_nm);
    let call_hot_entry = h.base as u64; // entry_off == verified_entry_off == 0 (no guard, `None`)

    // First call resolves the site to Mono{klass: Tmp, target: hot_entry}
    // (through stub_resolve, exactly `mono_resolve_patches_call_site_
    // and_dispatches`'s own first-dispatch arm).
    let call: CallStubFn = unsafe { std::mem::transmute(vm.stubs.call_stub_entry()) };
    let vm_ptr: *mut VmState = &mut vm;
    let argv = [tmp_instance.raw()];
    let result = unsafe { call(call_hot_entry, vm_ptr, argv.as_ptr(), 1) };
    assert_eq!(result, SmallInt::new(42).oop().raw());
    match vm.code_table.get(call_hot_id).unwrap().ic_sites[0].state {
        IcState::Mono { klass, target } => {
            assert_eq!(klass, tmp_klass, "must record the receiver's own klass");
            assert_eq!(target, hot_entry, "must record hot's own entry");
        }
        other => panic!("expected Mono after the first resolve, got {other:?}"),
    }

    // Same "no global, no live instance" setup as `key_klass_death_
    // flushes` -- the ONLY difference is callHot's own live Mono guard,
    // which still names Tmp.
    fullgc::full_gc(&mut vm).expect("full gc must succeed");

    assert!(
        vm.code_table.get(hot_id).is_some(),
        "hot must survive: callHot's own live Mono guard still names Tmp, keeping it \
         reachable regardless of hot's own key_klass being correctly excluded from marking"
    );
    assert!(
        vm.code_table.get(call_hot_id).is_some(),
        "callHot must survive too -- its own key_klass is ALSO Tmp, kept alive the same way"
    );
    assert!(
        matches!(
            vm.code_table.get(call_hot_id).unwrap().ic_sites[0].state,
            IcState::Mono { .. }
        ),
        "callHot's own site must be untouched -- nothing was flushed"
    );
}

// ── The flagship: mid_loop_forced_scavenge (tests_s12.md gate item 2) ─────

/// Shared scaffolding for the two flagship tests below: hand-builds a
/// compiled loop method (`driver::eligible` still rejects ordinary
/// non-smi sends outright — the same S11 eligibility gate documented in
/// step 6's STEP-6 NOTES — so `driver::compile_method` cannot be the
/// front door; `mono_resolve_patches_call_site_and_dispatches`'s own
/// hand-built-IrMethod precedent applies), attaches REAL per-safepoint
/// GC metadata exactly the way `driver::compile_method` does (the same
/// `oopmap::build_for_position` + `intern` calls over the same
/// `RegallocResult` — without this, `Nmethod::oopmap_at`'s exact-match
/// panic is the FIRST thing any scavenge under the loop hits), installs
/// it, and patches every send site to `stub_resolve`.
#[allow(clippy::too_many_arguments)]
fn install_loop_nmethod(
    vm: &mut VmState,
    key_klass: KlassOop,
    blocks: Vec<IrBlock>,
    vregs: Vec<VRegInfo>,
    pool: Vec<PoolEntry>,
    call_sites: Vec<CallSiteInfo>,
    argc: u8,
    probe_name: &[u8],
) -> NmethodId {
    let ir = IrMethod {
        osr_cold_sends: 0,
        blocks,
        vregs,
        pool,
        argc,
        ntemps: 0,
        ctx_vregs: Vec::new(),
        block_closure_vreg: None,
        method_ctx_vreg: None,
        spliced_nlr: 0,
        spliced_multibb: 0,
        splice_declined_budget: 0,
        safepoints: Vec::new(),
        true_lit: PoolLit(1),
        false_lit: PoolLit(2),
        nil_lit: PoolLit(0),
        mark_slots_lit: PoolLit(3),
        mark_double_lit: PoolLit(0),
        double_klass_lit: PoolLit(0),
        float64x2_klass_lit: PoolLit(0),
        float32x4_klass_lit: PoolLit(0),
        int32x4_klass_lit: PoolLit(0),
        call_sites,
        site_feedback: Vec::new(),
        inline_deps: Vec::new(),
        self_devirt: false,
        method_pool_ix: None,
    };
    let ra: RegallocResult = regalloc::regalloc(&ir);
    let mut asm = JasmAssembler::new();
    let (blob, _pcs, verified_entry_off, emitted_ic_sites, safepoints, _osr_off): (
        _,
        _,
        u32,
        _,
        Vec<SafepointPc>,
        Option<u32>,
    ) = emit::emit(
        &mut asm,
        &ir,
        &ra,
        vm.stubs.stub_poll_addr(),
        vm.stubs.must_be_boolean_addr(),
        vm.stubs.alloc_slow_addr(),
        vm.stubs.box_double_addr(),
        vm.stubs.box_float64x2_addr(),
        vm.stubs.box_float32x4_addr(),
        vm.stubs.box_int32x4_addr(),
        vm.stubs.call_primitive_addr(),
        vm.stubs.nlr_originate_addr(),
        None,
        None,
        None,
    );

    let h = vm.code_cache.alloc(blob.code.len()).unwrap();
    vm.code_cache.publish(h, &blob);
    let resolve_addr = vm.stubs.resolve_addr();
    for site in &emitted_ic_sites {
        vm.code_cache.patch_branch26_at(h, site.off, resolve_addr);
    }

    // REAL GC metadata — the same construction `driver::compile_method`
    // performs (oopmaps[0] reserved empty; one liveness-intersected,
    // deduplicated map per real safepoint). Block-start descs are omitted:
    // they exist only for `bci_at`'s trace path, which nothing here uses.
    let mut oopmaps: Vec<OopMap> = vec![OopMap::empty()];
    let mut pcdescs: Vec<PcDesc> = Vec::with_capacity(safepoints.len());
    for sp in &safepoints {
        let map = oopmap::build_for_position(
            &ra.intervals,
            ra.frame_slots,
            sp.position,
            &ra.extra_oop_live,
        );
        let idx = oopmap::intern(&mut oopmaps, map);
        pcdescs.push(PcDesc {
            pc_off: sp.pc_off,
            bci: sp.bci,
            oopmap: idx,
        });
    }
    pcdescs.sort_by_key(|d| d.pc_off);

    let probe_sel = vm.universe.intern(probe_name);
    let ic_sites: Vec<IcSite> = emitted_ic_sites
        .iter()
        .map(|s| IcSite {
            off: s.off,
            selector: s.selector,
            argc: s.argc,
            state: IcState::Unresolved,
            super_klass: None,
        })
        .collect();
    let nm = Nmethod {
        osr_cold_sends: 0,
        id: NmethodId(0),
        key_klass,
        key_selector: probe_sel,
        code: h,
        entry_off: 0,
        verified_entry_off,
        state: NmState::Alive,
        level: 1,
        version: 0,
        trap_count: 0,
        profile_hash: 0,
        literal_off: blob.literal_off,
        relocs: blob.relocs.clone(),
        frame_slots: ra.frame_slots,
        slot_is_oop: ra.slot_is_oop.clone(),
        pcdescs,
        oopmaps,
        ic_sites,
        poll_bci: None,
        prim_call_argc_plus_recv: None,
        block_method: None,
        deopt_scopes: Vec::new(),
        deopt_pcdescs: Vec::new(),
        inline_deps: Vec::new(),
        self_devirt: false,
        osr_map: None,
    };
    vm.code_table.install(nm)
}

/// The standard well-known-literal pool prefix `install_loop_nmethod`'s
/// own `PoolLit(0..=3)` indices assume: nil, true, false, pristine-Slots
/// mark — mirroring `ir::convert`'s own interning order for the same four.
fn base_pool(vm: &VmState) -> Vec<PoolEntry> {
    vec![
        PoolEntry {
            value: vm.universe.nil_obj.raw(),
            kind: Some(RelocKind::Oop),
        },
        PoolEntry {
            value: vm.universe.true_obj.raw(),
            kind: Some(RelocKind::Oop),
        },
        PoolEntry {
            value: vm.universe.false_obj.raw(),
            kind: Some(RelocKind::Oop),
        },
        PoolEntry {
            value: Mark::pristine().with_tagged_contents(true).word(),
            kind: None,
        },
    ]
}

/// `tests_s12.md` gate item 2, THE FLAGSHIP, full version (step 7 — the
/// bridge is gone, so the real `gcScavenge` primitive genuinely collects
/// when sent from inside a compiled loop): ONE compiled activation whose
/// spill slot holds an eden object across ~100 REAL scavenges, each
/// triggered by a real `scav` send (compiled `bl` → `stub_resolve` →
/// c2i adapter → `rt_interpret_call` → interpreter → primitive 93 → a
/// full scavenge with THIS compiled frame live on the native stack).
///
/// Adapted from the doc's own sketch exactly as steps 4/6 already
/// established (their STEP NOTES document why): the method is hand-built
/// IR, not `T>>run:` source (`driver::eligible` rejects every non-smi
/// send — the loop itself is plain `SmiCmpBr`/`SmiArith`/`Jump`, exactly
/// what source `to:do:` lowers to anyway); the arithmetic accumulator is
/// dropped (the correctness signal is the OBJECT's own identity/ivars
/// surviving ~100 relocations, which `s` never strengthened); and the
/// `vm.test_hooks.on_scavenge` instrumentation the doc imagines doesn't
/// exist — the frame slot's correctness is proven OBSERVABLY instead:
/// every iteration's `scav` send reloads the receiver from the same
/// spill slot each_code_root just rewrote, so ONE stale-slot failure
/// poisons every subsequent iteration's dispatch (in debug builds the
/// scavenge-entry verifier also walks the whole heap between every
/// single pair of iterations — 100 independent full-heap validations).
///
/// Assertion (c) of the doc is kept verbatim: `gc_under_compiled`
/// incremented ≥ 100 — P10's inversion, the proof the hard case ran.
/// Assertion (d) is implicit and stronger than the doc asks: EVERY
/// scavenge's own `oopmap_at(ret_pc)` lookup panics on a non-exact
/// PcDesc match (P1), so a single mis-recorded safepoint fails loudly.
#[test]
fn mid_loop_forced_scavenge() {
    const N: i64 = 100;
    let mut vm = test_vm();
    let object_klass = vm.universe.object_klass;
    let t_klass = vm.universe.new_klass(
        object_klass,
        "MidLoopT",
        Format::Slots,
        false,
        HEADER_WORDS + 2,
    );

    // `scav` on T: the REAL gcScavenge primitive (93), interpreted-only
    // (primitive methods are never compiled), reached via a real c2i
    // adapter from the compiled loop's own send site.
    let scav_sel = vm.universe.intern(b"scav");
    let mut sb = BytecodeBuilder::new();
    sb.ret_self(); // fallback body -- never reached (prim always succeeds)
    let scav_method = sb.finish(&mut vm, scav_sel, 0, 0);
    scav_method.set_primitive(93);
    scav_method.set_flags(0, 0, false, false, true, false, 0);
    install_method(&mut vm, t_klass, scav_sel, scav_method);

    // The loop: i := 0. [i < N] whileTrue: [self scav. i := i + 1]. ^self
    //   B0: v0=Param(self); v1=0; v2=N; Jump B1
    //   B1: SmiCmpBr Lt v1 v2 -> B2 / B3, fail B4
    //   B2: CallSend scav(v0) -> v3 (dead); v4=1; v1 = v1+v4; Jump B1
    //   B3: Ret v0
    //   B4: Bailout
    // `self` (v0) is live across B2's CallSend safepoint on every
    // iteration -> D1's spill-all forces it into the SAME spill slot for
    // the whole method, the slot each of the ~100 scavenges must find
    // and rewrite.
    let vregs: Vec<VRegInfo> = vec![
        VRegInfo { is_oop: true, is_fp: false },  // v0 self
        VRegInfo { is_oop: false, is_fp: false }, // v1 i
        VRegInfo { is_oop: false, is_fp: false }, // v2 N
        VRegInfo { is_oop: true, is_fp: false },  // v3 send result (dead)
        VRegInfo { is_oop: false, is_fp: false }, // v4 const 1
    ];
    let blocks = vec![
        IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::Param {
                    dst: VReg(0),
                    index: 0,
                },
                Ir::ConstSmi {
                    dst: VReg(1),
                    value: 0,
                },
                Ir::ConstSmi {
                    dst: VReg(2),
                    value: N,
                },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(1),
            bci: 1,
            code: vec![Ir::SmiCmpBr {
                op: CmpOp::Lt,
                a: VReg(1),
                b: VReg(2),
                if_true: BlockId(2),
                if_false: BlockId(3),
                fail: BlockId(4),
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(2),
            bci: 2,
            code: vec![
                Ir::CallSend {
                    dst: VReg(3),
                    site: 0,
                    args: vec![VReg(0)],
                },
                Ir::ConstSmi {
                    dst: VReg(4),
                    value: 1,
                },
                Ir::SmiArith {
                    op: SmiOp::Add,
                    dst: VReg(1),
                    a: VReg(1),
                    b: VReg(4),
                    fail: BlockId(4),
                },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(3),
            bci: 3,
            code: vec![Ir::Ret { val: VReg(0) }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(4),
            bci: 4,
            code: vec![Ir::Bailout {
                reason: BailoutReason::SmiOpFailed,
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
    ];
    let call_sites = vec![CallSiteInfo {
        selector: scav_sel,
        // Receiver INCLUDED (ir.rs's own `ic_view.argc() + 1` convention)
        // -- this exact count is what `roots::real_oop_rootspill_slots`
        // reads to bound the c2i RootSpill scan while the scavenge runs
        // INSIDE the adapter, so this test also covers the off-by-one its
        // first draft had (an extra garbage slot scanned as an oop).
        argc: 1,
        static_klass: None,
    }];
    let pool = base_pool(&vm);
    let nm_id = install_loop_nmethod(
        &mut vm,
        t_klass,
        blocks,
        vregs,
        pool,
        call_sites,
        1,
        b"midLoopRunProbe",
    );

    // p: a YOUNG T instance with recognizable ivars.
    let p = alloc::alloc_slots(&mut vm, t_klass);
    p.set_body_oop(0, SmallInt::new(3).oop());
    p.set_body_oop(1, SmallInt::new(4).oop());
    let p_original_bits = p.oop().raw();

    // Root the test's own klass handle UNDER the receiver (enter_compiled
    // reads [receiver] from the top of stack; deeper slots are ordinary
    // roots every scavenge rewrites) -- after ~100 collections the
    // Rust-local `t_klass` is long stale.
    // Root the klass + p first, THEN run one real interpreted call, THEN
    // push the receiver copy on top. Ordering is load-bearing:
    // `enter_compiled`'s TierLink records `vm.stack.fp`, and `walk_frames`
    // (run by every scavenge under the compiled frame) crosses the
    // CallStub boundary into THAT frame and reads its `saved_fp`,
    // expecting a genuine entry-frame layout with the
    // ENTRY_FRAME_SENTINEL — exactly what every production compiled call
    // has (compiled code is only ever entered from a real interpreted
    // send, whose caller frame is LIVE above everything else). A bare
    // test VM only gets the DEAD REMNANT of a completed warm-up run's
    // entry frame — which works, but only if nothing overwrites its
    // slots: pushing test roots AFTER the warm-up plants them exactly
    // where that dead frame's own header lives (sp rewound to 0), and
    // the walker then reads a pushed mem oop as `saved_fp` (found the
    // hard way, twice: "Frame::saved_fp: not a smi" from inside the
    // first scavenge's own walk). Pushing them FIRST moves the warm-up
    // frame — and its intact sentinel — safely above them.
    vm.stack.push(t_klass.oop());
    vm.stack.push(p.oop());
    let idle_sel = vm.universe.intern(b"idleWarm");
    let mut ib = BytecodeBuilder::new();
    ib.ret_self();
    let idle_method = ib.finish(&mut vm, idle_sel, 0, 0);
    let _ = macvm::interpreter::run_method(&mut vm, idle_method, SmallInt::new(0).oop(), &[]);
    vm.stack.push(p.oop()); // the receiver `enter_compiled` reads from the top
    let scav_before = vm.universe.gc_stats.scavenge_count;
    let under_before = vm.universe.gc_stats.gc_under_compiled;

    assert_eq!(enter_compiled(&mut vm, nm_id, 0), EnterResult::Completed);

    let result = vm.stack.pop();
    let _p_root = vm.stack.pop(); // the slot-1 root copy (live, but done with)
    let t_klass_now = KlassOop::try_from(vm.stack.pop()).expect("rooted klass survived");
    assert_eq!(
        vm.universe.gc_stats.scavenge_count,
        scav_before + N as u64,
        "every one of the {N} scav sends must have run a REAL scavenge"
    );
    assert!(
        vm.universe.gc_stats.gc_under_compiled >= under_before + N as u64,
        "gate item 2(c) / P10: all {N} scavenges ran UNDER the live compiled frame"
    );
    assert_ne!(
        result.raw(),
        p_original_bits,
        "p must have MOVED (a young object cannot sit still through {N} scavenges)"
    );
    assert_eq!(
        klass_of(&vm, result).oop().raw(),
        t_klass_now.oop().raw(),
        "the returned object must still be a T"
    );
    let result_mem = MemOop::try_from(result).expect("result must be a mem oop");
    assert_eq!(
        result_mem.body_oop(0),
        SmallInt::new(3).oop(),
        "ivar x must survive ~{N} relocations intact"
    );
    assert_eq!(
        result_mem.body_oop(1),
        SmallInt::new(4).oop(),
        "ivar y must survive ~{N} relocations intact"
    );
    assert!(
        matches!(
            vm.code_table.get(nm_id).unwrap().ic_sites[0].state,
            IcState::Mono { .. }
        ),
        "the scav site must have gone mono on iteration 1 and STAYED valid across every \
         scavenge (update_keys keeps its guard-klass mirror current)"
    );
}

/// Primitive-shim GC-safety (docs/prim_shims.md §4-5): a JIT-compiled method
/// carrying `<primitive: 24>` (`basicNew:`) allocates INSIDE its own shim
/// call (`stub_call_primitive` → `rt_call_primitive` → `alloc::*`). Under
/// `gc_stress` every such allocation forces a real scavenge while the
/// `AdapterKind::CallPrimitive` stub frame is the innermost native frame — so
/// the walker must traverse it and scan exactly `prim_call_argc_plus_recv`
/// (= method.argc()+1 = 2) RootSpill slots as roots. A wrong count would
/// scan stale non-oop registers as oops and corrupt the heap (or panic).
///
/// Driven via the REAL `enter_compiled` (never raw `stubs::invoke`, which
/// leaves no `TierLink::IntoCompiled` and would panic the walker at the
/// call_stub boundary), with the canonical base frame this file establishes
/// everywhere: push a root, run a throwaway `idleWarm` so a sentinel-
/// terminated entry-frame remnant exists for `walk_frames` to bottom out on,
/// then push the receiver+arg. Compiled customized to `klass_of(receiver)`
/// (Array's metaclass) so the entry guard matches. A co-rooted YOUNG canary,
/// pushed under the base frame, must survive every collection with its ivar
/// intact — proof the walker did not corrupt the heap while walking the
/// CallPrimitive frame.
#[test]
fn compiled_prim_shim_basicnew_under_gc_stress() {
    const N: usize = 16;
    let mut vm = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: true,
        gc_stress_full_period: None,
        eden_kb: Some(256),
        jit: JitMode::Off,
    });

    // `probeNew: n` == `<primitive: 24> ^self`: Array class + non-negative smi
    // → a fresh n-element Array (the `^self` fallback is never taken).
    let sel = vm.universe.intern(b"probeNew:");
    let mut b = BytecodeBuilder::new();
    b.ret_self();
    let method = b.finish(&mut vm, sel, 1, 0);
    method.set_primitive(24);
    method.set_flags(1, 0, false, false, true, false, 0);

    // enter_compiled enters at entry_off (the S14 guard); customize to the
    // receiver's OWN klass = klass_of(Array-the-class) = Array's metaclass.
    let recv0 = vm.universe.array_klass.oop();
    let recv_klass = klass_of(&vm, recv0);
    let id = driver::compile_method(&mut vm, recv_klass, method)
        .expect("primitive-24 (basicNew:) method must be shim-eligible and compile");

    // A YOUNG canary with a recognizable ivar, rooted on the operand stack
    // UNDER the base frame.
    let object_klass = vm.universe.object_klass;
    let canary_klass = vm.universe.new_klass(
        object_klass,
        "ShimCanary",
        Format::Slots,
        false,
        HEADER_WORDS + 1,
    );
    let canary = alloc::alloc_slots(&mut vm, canary_klass);
    canary.set_body_oop(0, SmallInt::new(1234).oop());

    // Base frame: root the canary FIRST, then run a throwaway interpreted
    // method so its dead-remnant entry frame — sentinel intact — sits above
    // the canary for the walker to terminate on.
    vm.stack.push(canary.oop());
    let idle_sel = vm.universe.intern(b"idleWarm");
    let mut ib = BytecodeBuilder::new();
    ib.ret_self();
    let idle_method = ib.finish(&mut vm, idle_sel, 0, 0);
    let _ = macvm::interpreter::run_method(&mut vm, idle_method, SmallInt::new(0).oop(), &[]);

    let scav_before = vm.universe.gc_stats.scavenge_count;
    for n in 0..N {
        let recv = vm.universe.array_klass.oop(); // permanent; re-read for clarity
        let count = SmallInt::new(n as i64).oop();
        vm.stack.push(recv);
        vm.stack.push(count);
        assert_eq!(
            enter_compiled(&mut vm, id, 1),
            EnterResult::Completed,
            "compiled basicNew: {n} must complete (guard matched, shim ran)"
        );
        let result = vm.stack.pop();
        let arr = macvm::oops::wrappers::ArrayOop::try_from(result)
            .unwrap_or_else(|| panic!("compiled basicNew: {n} shim must return an Array"));
        assert_eq!(
            arr.len(),
            n,
            "the shim must allocate an n-element Array across a stressed collection"
        );
    }

    assert!(
        vm.universe.gc_stats.scavenge_count > scav_before,
        "gc_stress must have forced real scavenges through the CallPrimitive frame"
    );

    // The co-rooted young canary survived every collection uncorrupted.
    let canary_now = MemOop::try_from(vm.stack.pop()).expect("canary survived as a mem oop");
    assert_eq!(
        canary_now.body_oop(0),
        SmallInt::new(1234).oop(),
        "the co-rooted young canary's ivar must survive the collections intact"
    );
}

/// `tests_s12.md` gate item 2's VARIANT: the scavenges come from the
/// ALLOC slow edge instead of a send — the loop body is an inline
/// `Ir::Alloc` (a real compiled eden bump through `eden_top_addr`), with
/// eden pre-filled so the very first iteration overflows into
/// `rt_alloc_slow` → the ordinary cascade → a real scavenge under the
/// live compiled frame; each subsequent fill cycle (~64 iterations of
/// 510-word objects through a 256 KiB eden) scavenges again. Covers the
/// `Alloc` safepoint's own oop map (a DIFFERENT safepoint kind than the
/// send variant's CallSend) plus the post-scavenge inline fast path
/// re-reading the live (reset) bump pointer through the shared word.
#[test]
fn mid_loop_alloc_edge_forced_scavenge() {
    const N: i64 = 200;
    const CHUNK_WORDS: u32 = 510; // 4080 bytes -- just under emit's 4096 gate
    let mut vm = test_vm();
    let object_klass = vm.universe.object_klass;
    let t_klass = vm.universe.new_klass(
        object_klass,
        "AllocEdgeT",
        Format::Slots,
        false,
        HEADER_WORDS + 2,
    );
    let chunk_klass = vm.universe.new_klass(
        object_klass,
        "AllocEdgeChunk",
        Format::Slots,
        false,
        CHUNK_WORDS as usize,
    );

    // Loop: same skeleton as the send flagship, body = inline Alloc.
    let vregs: Vec<VRegInfo> = vec![
        VRegInfo { is_oop: true, is_fp: false },  // v0 self
        VRegInfo { is_oop: false, is_fp: false }, // v1 i
        VRegInfo { is_oop: false, is_fp: false }, // v2 N
        VRegInfo { is_oop: true, is_fp: false },  // v3 alloc result (dead)
        VRegInfo { is_oop: false, is_fp: false }, // v4 const 1
    ];
    let mut pool = base_pool(&vm);
    let chunk_klass_lit = PoolLit(pool.len() as u32);
    pool.push(PoolEntry {
        value: chunk_klass.oop().raw(),
        kind: Some(RelocKind::Oop),
    });
    let blocks = vec![
        IrBlock {
            id: BlockId(0),
            bci: 0,
            code: vec![
                Ir::Param {
                    dst: VReg(0),
                    index: 0,
                },
                Ir::ConstSmi {
                    dst: VReg(1),
                    value: 0,
                },
                Ir::ConstSmi {
                    dst: VReg(2),
                    value: N,
                },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(1),
            bci: 1,
            code: vec![Ir::SmiCmpBr {
                op: CmpOp::Lt,
                a: VReg(1),
                b: VReg(2),
                if_true: BlockId(2),
                if_false: BlockId(3),
                fail: BlockId(4),
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(2),
            bci: 2,
            code: vec![
                Ir::Alloc {
                    dst: VReg(3),
                    klass: chunk_klass_lit,
                    size_words: CHUNK_WORDS,
                },
                Ir::ConstSmi {
                    dst: VReg(4),
                    value: 1,
                },
                Ir::SmiArith {
                    op: SmiOp::Add,
                    dst: VReg(1),
                    a: VReg(1),
                    b: VReg(4),
                    fail: BlockId(4),
                },
                Ir::Jump { target: BlockId(1) },
            ],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(3),
            bci: 3,
            code: vec![Ir::Ret { val: VReg(0) }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
        IrBlock {
            id: BlockId(4),
            bci: 4,
            code: vec![Ir::Bailout {
                reason: BailoutReason::SmiOpFailed,
            }],
            entry_stack: Vec::new(),
            deopt_sites: Vec::new(),
        },
    ];
    let nm_id = install_loop_nmethod(
        &mut vm,
        t_klass,
        blocks,
        vregs,
        pool,
        Vec::new(),
        1,
        b"allocEdgeRunProbe",
    );

    let p = alloc::alloc_slots(&mut vm, t_klass);
    p.set_body_oop(0, SmallInt::new(3).oop());
    p.set_body_oop(1, SmallInt::new(4).oop());
    let p_original_bits = p.oop().raw();

    // Pre-fill eden honestly so iteration 1's inline bump overflows
    // immediately (same idiom as every other slow-edge test here).
    let array_klass = vm.universe.array_klass;
    let chunk_elems = 4096usize;
    let chunk_bytes = (HEADER_WORDS + chunk_elems) * 8;
    while vm.universe.eden.end - vm.universe.eden.top >= chunk_bytes {
        alloc::alloc_indexable_oops(&mut vm, array_klass, chunk_elems);
    }
    while vm.universe.eden.end - vm.universe.eden.top >= (CHUNK_WORDS as usize) * 8 {
        alloc::alloc_slots(&mut vm, chunk_klass);
    }

    // Root the klass + p first, THEN run one real interpreted call, THEN
    // push the receiver copy on top. Ordering is load-bearing:
    // `enter_compiled`'s TierLink records `vm.stack.fp`, and `walk_frames`
    // (run by every scavenge under the compiled frame) crosses the
    // CallStub boundary into THAT frame and reads its `saved_fp`,
    // expecting a genuine entry-frame layout with the
    // ENTRY_FRAME_SENTINEL — exactly what every production compiled call
    // has (compiled code is only ever entered from a real interpreted
    // send, whose caller frame is LIVE above everything else). A bare
    // test VM only gets the DEAD REMNANT of a completed warm-up run's
    // entry frame — which works, but only if nothing overwrites its
    // slots: pushing test roots AFTER the warm-up plants them exactly
    // where that dead frame's own header lives (sp rewound to 0), and
    // the walker then reads a pushed mem oop as `saved_fp` (found the
    // hard way, twice: "Frame::saved_fp: not a smi" from inside the
    // first scavenge's own walk). Pushing them FIRST moves the warm-up
    // frame — and its intact sentinel — safely above them.
    vm.stack.push(t_klass.oop());
    vm.stack.push(p.oop());
    let idle_sel = vm.universe.intern(b"idleWarm");
    let mut ib = BytecodeBuilder::new();
    ib.ret_self();
    let idle_method = ib.finish(&mut vm, idle_sel, 0, 0);
    let _ = macvm::interpreter::run_method(&mut vm, idle_method, SmallInt::new(0).oop(), &[]);
    vm.stack.push(p.oop()); // the receiver `enter_compiled` reads from the top
    let scav_before = vm.universe.gc_stats.scavenge_count;
    let under_before = vm.universe.gc_stats.gc_under_compiled;

    assert_eq!(enter_compiled(&mut vm, nm_id, 0), EnterResult::Completed);

    let result = vm.stack.pop();
    let _p_root = vm.stack.pop(); // the slot-1 root copy (live, but done with)
    let t_klass_now = KlassOop::try_from(vm.stack.pop()).expect("rooted klass survived");
    assert!(
        vm.universe.gc_stats.scavenge_count >= scav_before + 2,
        "200 iterations x 4080 bytes through a 256 KiB eden must scavenge repeatedly \
         (got {} new scavenges)",
        vm.universe.gc_stats.scavenge_count - scav_before
    );
    assert!(
        vm.universe.gc_stats.gc_under_compiled >= under_before + 2,
        "every one of those scavenges ran UNDER the live compiled frame (Alloc safepoint)"
    );
    assert_ne!(result.raw(), p_original_bits, "p must have moved");
    assert_eq!(
        klass_of(&vm, result).oop().raw(),
        t_klass_now.oop().raw(),
        "the returned object must still be a T"
    );
    let result_mem = MemOop::try_from(result).expect("result must be a mem oop");
    assert_eq!(result_mem.body_oop(0), SmallInt::new(3).oop());
    assert_eq!(result_mem.body_oop(1), SmallInt::new(4).oop());
    let _ = Oop::from_raw(result.raw()); // shape-checkable to the end
}
