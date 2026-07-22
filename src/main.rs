//! MACVM entry point (placeholder).
//!
//! The VM is at the scaffold stage; this just proves the crate builds and
//! links. Hidden test hooks observed via a real subprocess (integration
//! tests can't otherwise see this process's own exit code or exhaustive
//! stderr output): `--selftest-alloc-loop` allocates rooted objects until
//! the heap is genuinely exhausted (`tests/it_memory.rs::eden_exhaustion_aborts`);
//! `--selftest-stack-overflow` pushes until the process stack is exhausted
//! (`tests/it_interp.rs::process_stack_overflow_exits_cleanly`);
//! `--selftest-trace-diamond` runs the k_diamond kernel under
//! `MACVM_TRACE=bytecode` so the caller can count emitted trace lines
//! (`tests/it_interp.rs::trace_mode_line_count`); `--selftest-dnu-fallback`
//! sends an unrecognized selector with no `doesNotUnderstand:` installed
//! anywhere, exercising `runtime::error::dnu_fallback`'s pinned stdout
//! format and its real `exit(1)`.

use std::io::{BufRead, Write as _};
use std::path::{Path, PathBuf};

use macvm::bytecode::BytecodeBuilder;
use macvm::memory::alloc;
use macvm::oops::smi::SmallInt;
use macvm::oops::Oop;
use macvm::runtime::{VmOptions, VmState};

fn main() {
    #[cfg(windows)]
    bench_pin::maybe_pin_benchmark_cpu();
    if std::env::args().any(|a| a == "--selftest-alloc-loop") {
        selftest_alloc_loop();
    }
    if std::env::args().any(|a| a == "--selftest-stack-overflow") {
        selftest_stack_overflow();
    }
    if std::env::args().any(|a| a == "--selftest-trace-diamond") {
        selftest_trace_diamond();
    }
    if std::env::args().any(|a| a == "--selftest-dnu-fallback") {
        selftest_dnu_fallback();
    }
    if std::env::args().any(|a| a == "--selftest-probe-assert") {
        selftest_probe_crash(ProbeCrashKind::Assert);
    }
    if std::env::args().any(|a| a == "--selftest-probe-segv") {
        selftest_probe_crash(ProbeCrashKind::Segv);
    }
    if std::env::args().any(|a| a == "--selftest-probe-foreign") {
        selftest_probe_foreign();
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("run") => cmd_run(&args[1..], false),
        Some("debug") => cmd_run(&args[1..], true),
        Some("repl") => cmd_repl(&args[1..]),
        Some("rusttcl") => cmd_rusttcl(&args[1..]),
        _ => println!("MACVM — Self/Strongtalk-lineage research VM (arm64). Scaffold only."),
    }
}

/// `--world <dir>` parsing shared by `run`/`repl`; any other args are
/// returned as the positional leftovers (`run`'s `<file.mst>`).
fn parse_world_flag(args: &[String]) -> (Option<PathBuf>, Vec<String>) {
    let mut world_dir = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--world" {
            i += 1;
            world_dir = args.get(i).map(PathBuf::from);
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    (world_dir, rest)
}

fn load_world_with_warning(vm: &mut VmState, world_dir: &Path) {
    match macvm::frontend::world::load_world(vm, world_dir) {
        Ok(true) => {}
        Ok(false) => eprintln!(
            "warning: no world.list found at {} — continuing without a world",
            world_dir.display()
        ),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// `macvm run <file.mst> [--world <dir>]` (SPEC §3.2, `sprint_s05_detail.md`
/// §Design "CLI"). Exit 0 unless a compile error / uncaught VM error.
/// `debug = true` (the `macvm debug` subcommand, DBG1 — docs/DEBUGGER.md
/// §5): arm `vm.debug.active` before running, and honor `MACVM_DEBUG`'s
/// `break:Class>>sel@bci[,…]` grammar after the world loads (the scripted
/// gates' entry). `MACVM_DEBUG=1`/`break:` also arms plain `run`.
fn cmd_run(args: &[String], debug: bool) {
    let (world_dir, rest) = parse_world_flag(args);
    let Some(file) = rest.first() else {
        eprintln!("usage: macvm run|debug <file.mst> [--world <dir>]");
        std::process::exit(2);
    };
    let mut vm = VmState::new();
    load_world_with_warning(
        &mut vm,
        &world_dir.unwrap_or_else(|| PathBuf::from("world")),
    );

    // DBG5 §D3: arm the interactive step-call auditor from the environment.
    // Independent of MACVM_DEBUG — it stops at compiled SEND boundaries, not
    // bytecode breakpoints, and force-colds ICs the same way MACVM_TRACE=calls
    // does (observation-only; the guest's answer is unchanged).
    if std::env::var("MACVM_STEP_CALLS").is_ok() {
        vm.debug.step_calls = true;
    }

    let debug_spec = std::env::var("MACVM_DEBUG").unwrap_or_default();
    let pin_spec = std::env::var("MACVM_PIN").unwrap_or_default();
    if debug || !debug_spec.is_empty() || !pin_spec.is_empty() {
        vm.debug.active = true;
        for (class, sel, bci) in macvm::runtime::debug::parse_debug_spec(&debug_spec) {
            match macvm::runtime::debug::set_breakpoint_by_name(&mut vm, &class, &sel, bci) {
                Ok(msg) => eprintln!("{msg}"),
                // The class usually lives in the very file about to run —
                // park the spec; install_method lands it the moment the
                // method exists (debug::on_method_installed).
                Err(_) => vm.debug.pending.push((class, sel, bci)),
            }
        }
        // MACVM_PIN: force methods to tier-0 (differential diagnosis —
        // "does interpreting THIS method change the result?").
        for (class, sel) in macvm::runtime::debug::parse_pin_spec(&pin_spec) {
            match macvm::runtime::debug::pin_by_name(&mut vm, &class, &sel) {
                Ok(msg) => eprintln!("{msg}"),
                Err(_) => vm.debug.pending_pins.push((class, sel)),
            }
        }
    }

    let result = macvm::frontend::world::load_file(&mut vm, Path::new(file));
    print_bytecode_count(&mut vm);
    print_gc_bridge_stats(&vm);
    print_vm_stats(&vm);
    match result {
        Ok(()) => std::process::exit(vm.exit_code.unwrap_or(0)),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// `MACVM_TRACE=count` (S6 PERF procedure) — printed to stderr so golden
/// stdout transcripts (fib/sieve/point_demo) stay exact regardless of
/// whether the flag is set.
fn print_bytecode_count(vm: &mut VmState) {
    if vm.options.trace.is_enabled("count") {
        eprintln!("bytecodes: {}", vm.bytecode_count);
        // S24 A1 (design §4 gate 2): the interpreted tail, attributed.
        // Flush the still-open run first (`count_cur`'s doc), then one
        // line per method, descending, with cumulative share — the top of
        // this list IS the answer to "what still interprets, and why".
        if let Some((_, label, run)) = vm.count_cur.take() {
            *vm.count_by_method.entry(label).or_insert(0) += run;
        }
        let mut rows: Vec<(&String, &u64)> = vm.count_by_method.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let total = vm.bytecode_count.max(1) as f64;
        let mut cum = 0u64;
        for (label, n) in rows.iter().take(40) {
            cum += **n;
            eprintln!(
                "bytecodes-by-method: {n:>12} {:5.1}% (cum {:5.1}%) {label}",
                100.0 * **n as f64 / total,
                100.0 * cum as f64 / total
            );
        }
        if rows.len() > 40 {
            let rest: u64 = rows[40..].iter().map(|(_, n)| **n).sum();
            eprintln!(
                "bytecodes-by-method: {rest:>12} {:5.1}% (cum 100.0%) ... {} more methods",
                100.0 * rest as f64 / total,
                rows.len() - 40
            );
        }
    }
}

/// `MACVM_TRACE=stats` (S15 A8): the full counter dump at process exit, to
/// stderr (golden stdout transcripts stay exact), grep-friendly one line per
/// counter. Code-cache byte totals are computed here from the live tables
/// rather than counted incrementally (they are exact by construction and
/// cost nothing off this path).
fn print_vm_stats(vm: &VmState) {
    if !vm.options.trace.is_enabled("stats") {
        return;
    }
    eprintln!("{}", macvm::runtime::vm_state::format_vm_stats(vm));
}

/// `MACVM_TRACE=gc`: a grep-friendly one-line counter summary printed to
/// stderr at process exit, mirroring `print_bytecode_count`'s own
/// convention. S12 step 7 inverted its meaning (P10): under S11's D8
/// bridge a shell recipe asserted `gc_under_compiled=0` (the bridge
/// held); with the bridge deleted the same counter is the proof the hard
/// case — a collection with live compiled frames on the native stack —
/// genuinely ran (`just bridge-stats-s11` now asserts it is > 0 under the
/// combined stress gate). `bridge_old_allocs` is gone with the bridge.
fn print_gc_bridge_stats(vm: &VmState) {
    if vm.options.trace.is_enabled("gc") {
        eprintln!(
            "gc: gc_under_compiled={}",
            vm.universe.gc_stats.gc_under_compiled
        );
    }
}

/// `macvm repl [--world <dir>]`: prompts `mst> `, accumulates lines until a
/// complete statement parses (an "unexpected EOF" parse error keeps
/// reading; any other error reports and resets the buffer), executes each
/// complete doIt, and prints its result via `printString` if understood,
/// else the Rust `print_oop` fallback (pre-S6 worlds).
fn cmd_repl(args: &[String]) {
    let (world_dir, _rest) = parse_world_flag(args);
    let mut vm = VmState::new();
    load_world_with_warning(
        &mut vm,
        &world_dir.unwrap_or_else(|| PathBuf::from("world")),
    );

    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        print!("{}", if buf.is_empty() { "mst> " } else { "...> " });
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        buf.push_str(&line);

        match macvm::frontend::parser::parse_one_top_item(&buf) {
            Ok(None) => buf.clear(),
            Ok(Some(item)) => {
                buf.clear();
                match macvm::frontend::classdef::execute_top_item(&mut vm, item) {
                    Ok(Some(result)) => println!("{}", print_result(&mut vm, result)),
                    Ok(None) => {}
                    Err(e) => println!("{e}"),
                }
            }
            Err(e) if e.eof => {} // keep buffering
            Err(e) => {
                println!("{e}");
                buf.clear();
            }
        }
    }
}

/// `macvm rusttcl [--world <dir>] [script.tcl]`: the live VM-introspection
/// shell (see `macvm::rusttcl`'s module doc) — `disasm`/`methods`/
/// `nmethods`/`ic`/`stats`/`trace`/`load`/`help`, plus the full vendored
/// Tcl language for scripting them. A positional script path runs
/// non-interactively (one shell invocation replaying a saved diagnostic
/// recipe); with none, it's an interactive `rusttcl> ` prompt.
fn cmd_rusttcl(args: &[String]) {
    let (world_dir, rest) = parse_world_flag(args);
    let mut ctx =
        macvm::rusttcl::RusttclCtx::new(world_dir.unwrap_or_else(|| PathBuf::from("world")));
    match rest.first() {
        Some(script) => {
            if let Err(e) = macvm::rusttcl::run_script(&mut ctx, Path::new(script)) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        None => macvm::rusttcl::run_repl(&mut ctx),
    }
}

fn print_result(vm: &mut VmState, result: Oop) -> String {
    let klass = macvm::runtime::lookup::klass_of(vm, result);
    let sel = vm.universe.intern(b"printString");
    if let Some(m) = macvm::runtime::lookup::lookup(vm, klass, sel) {
        let s = macvm::interpreter::run_method(vm, m, result, &[]);
        if let Some(b) = macvm::oops::wrappers::ByteArrayOop::try_from(s) {
            let mut bytes = Vec::new();
            b.copy_bytes_out(&mut bytes);
            return String::from_utf8_lossy(&bytes).into_owned();
        }
    }
    macvm::memory::print_oop(&vm.universe, result)
}

/// Allocates rooted (process-stack-pushed) arrays until the heap is
/// genuinely exhausted (S7-10: with a real scavenger wired into the
/// allocation choke point, unrooted garbage would just get reclaimed
/// forever and this would hang instead of exiting — `klass` is re-read
/// from `vm.universe` every iteration rather than captured once outside
/// the loop, since a bare local can go stale across the scavenges this
/// loop now triggers).
fn selftest_alloc_loop() -> ! {
    let mut vm = VmState::new();
    loop {
        let klass = vm.universe.array_klass;
        let arr = alloc::alloc_indexable_oops(&mut vm, klass, 1000);
        vm.stack.push(arr.oop());
    }
}

fn selftest_stack_overflow() -> ! {
    let mut vm = VmState::new();
    let v = SmallInt::new(0).oop();
    loop {
        vm.stack.push(v);
    }
}

fn selftest_trace_diamond() -> ! {
    let mut vm = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: macvm::runtime::TraceFlags::parse("bytecode"),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Off,
    });
    let mut b = BytecodeBuilder::new();
    let l1 = b.new_label();
    let l2 = b.new_label();
    b.push_self();
    b.br_false_fwd(l1);
    b.push_smi_i8(1);
    b.jump_fwd(l2);
    b.bind(l1);
    b.push_smi_i8(2);
    b.bind(l2);
    b.ret_tos();
    let sel = vm.universe.intern(b"diamond");
    let m = b.finish(&mut vm, sel, 0, 0);
    let true_obj = vm.universe.true_obj;
    let _ = macvm::interpreter::run_method(&mut vm, m, true_obj, &[]);
    std::process::exit(0)
}

fn selftest_dnu_fallback() -> ! {
    let mut vm = VmState::new();
    let object_klass = vm.universe.object_klass;
    let sel = vm.universe.intern(b"bar");
    let mut b = BytecodeBuilder::new();
    b.push_temp(0);
    b.send(&mut vm, sel, 0);
    b.ret_tos();
    let caller_sel = vm.universe.intern(b"caller");
    let caller = b.finish(&mut vm, caller_sel, 1, 0);
    let recv = alloc::alloc_slots(&mut vm, object_klass).oop();
    let nil = vm.universe.nil_obj;
    let _ = macvm::interpreter::run_method(&mut vm, caller, nil, &[recv]);
    unreachable!("dnu_fallback must have exited the process");
}

/// DBG0 gates (docs/DEBUGGER.md §6): which planted crash a
/// `--selftest-probe-*` flag drives through the PROBE dossier machinery.
enum ProbeCrashKind {
    /// A hand-published blob whose first instruction is `brk #0xDE02` —
    /// the compiled-assert trigger.
    Assert,
    /// A hand-published blob that loads from address 0 — a SIGSEGV whose
    /// pc is inside the registered code cache.
    Segv,
}

/// Publish a tiny crashing blob into a JIT VM's code cache and invoke it
/// through the real call stub (which establishes the x28 = &VmState
/// invariant the PROBE handlers rely on). The dossier exits 70; reaching
/// the end of this function is the failure mode.
fn selftest_probe_crash(kind: ProbeCrashKind) -> ! {
    use macvm::compiler::assembler::{imm, mem, x, Assembler};
    use macvm::compiler::jasm_assembler::JasmAssembler;

    let mut vm = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Threshold(1),
    });

    let mut a = JasmAssembler::new();
    match kind {
        ProbeCrashKind::Assert => {
            macvm::codecache::deopt_trap::emit_brk(
                &mut a,
                macvm::codecache::deopt_trap::TRAP_ASSERT,
            );
        }
        ProbeCrashKind::Segv => {
            a.emit("movz", &[x(16), imm(0)]);
            a.emit("ldr", &[x(0), mem(16, 0)]); // load from address 0 → SIGSEGV
        }
    }
    let blob = a.finish();
    let h = vm
        .code_cache
        .alloc(blob.code.len())
        .expect("selftest-probe: code cache alloc");
    vm.code_cache.publish(h, &blob);
    let entry = h.base as u64;
    let nil = vm.universe.nil_obj.raw();
    let stubs = vm.stubs;
    let _ = stubs.invoke(entry, &mut vm, &[nil]);
    eprintln!("selftest-probe: crash did not fire (BUG)");
    std::process::exit(1);
}

/// A fault whose pc is OUTSIDE every registered code cache — plain Rust
/// null-page read. PROBE must print only the one-line FOREIGN verdict and
/// let the default disposition kill the process (killed-by-signal, not
/// exit 70). The JIT VM exists solely to arm the handlers.
fn selftest_probe_foreign() -> ! {
    let vm = VmState::with_options(VmOptions {
        heap_mib: 64,
        trace: Default::default(),
        gc_stress: false,
        gc_stress_full_period: None,
        eden_kb: None,
        jit: macvm::runtime::JitMode::Threshold(1),
    });
    let _ = &vm;
    // SAFETY deliberately violated — this selftest IS the crash. The
    // address is computed through a volatile read of a runtime value so
    // neither rustc nor clippy can prove (or lint) the dereference away.
    unsafe {
        let addr: usize = std::ptr::read_volatile(&8usize);
        std::ptr::read_volatile(addr as *const u64);
    }
    eprintln!("selftest-probe-foreign: read of address 8 did not fault (BUG)");
    std::process::exit(1);
}

/// `MACVM_BENCH_CPU` — pin the VM to one PERFORMANCE core for benchmarking.
///
/// This machine (i7-12700, and hybrid Intel parts generally) mixes P- and
/// E-cores and throttles on temperature; Windows freely migrates the
/// process between core classes mid-run, which showed up as bench numbers
/// drifting 2x between sessions (PERF.md 2026-07-22: Cog's own arith read
/// 48, 116, and 50 ms across three sessions of identical code). Pinning
/// one P-core removes the core-class lottery; the thermal drift remains,
/// which is why PERF.md's same-session rule still stands.
///
/// Values: `perf` — detect the highest `EfficiencyClass` cores via
/// `GetLogicalProcessorInformationEx(RelationProcessorCore)` and pin to
/// the SECOND such core (core 0 eats a disproportionate share of system
/// interrupts); or an explicit logical-CPU index. Either form also raises
/// the process to HIGH_PRIORITY_CLASS. Off by default — an ordinary run
/// should share the machine like any other process.
#[cfg(windows)]
mod bench_pin {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn GetCurrentProcess() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        fn SetPriorityClass(process: isize, class: u32) -> i32;
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
    }
    const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;
    const RELATION_PROCESSOR_CORE: u32 = 0;

    /// Every physical core's `(efficiency_class, group0_mask)`, decoded from
    /// the variable-length `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records.
    fn cores() -> Vec<(u8, usize)> {
        let mut len: u32 = 0;
        unsafe {
            GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len)
        };
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let ok = unsafe {
            GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len)
        };
        if ok == 0 {
            return Vec::new();
        }
        // Record layout: u32 Relationship, u32 Size, then for
        // RelationProcessorCore a PROCESSOR_RELATIONSHIP: u8 Flags,
        // u8 EfficiencyClass, u8 Reserved[20], u16 GroupCount, then
        // GroupCount GROUP_AFFINITY entries (usize Mask, u16 Group,
        // u16 Reserved[3]). Only group 0 is read — this machine (and any
        // machine under 64 logical CPUs) has exactly one group.
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 8 <= len as usize {
            let rel = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let size = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap()) as usize;
            if size == 0 {
                break;
            }
            if rel == RELATION_PROCESSOR_CORE {
                let eff = buf[off + 9];
                // PROCESSOR_RELATIONSHIP: Flags(1) EfficiencyClass(1)
                // Reserved(20) GroupCount(2) then — already 8-aligned at
                // +32 from the record start — GROUP_AFFINITY.Mask.
                let mask_off = off + 32;
                if mask_off + 8 <= off + size {
                    let mask =
                        usize::from_le_bytes(buf[mask_off..mask_off + 8].try_into().unwrap());
                    out.push((eff, mask));
                }
            }
            off += size;
        }
        out
    }

    pub fn maybe_pin_benchmark_cpu() {
        let Ok(spec) = std::env::var("MACVM_BENCH_CPU") else {
            return;
        };
        let mask: usize = if spec == "perf" {
            let cores = cores();
            let best = cores.iter().map(|&(e, _)| e).max().unwrap_or(0);
            let perf: Vec<usize> = cores
                .iter()
                .filter(|&&(e, _)| e == best)
                .map(|&(_, m)| m)
                .collect();
            // Second P-core when there is one (core 0 takes the system's
            // interrupt load); one logical CPU only — its own hyperthread
            // sibling is excluded by taking the lowest bit.
            match perf.get(1).or_else(|| perf.first()) {
                Some(&m) if m != 0 => m & m.wrapping_neg(),
                _ => {
                    eprintln!("MACVM_BENCH_CPU=perf: no cores detected; not pinning");
                    return;
                }
            }
        } else {
            match spec.parse::<u8>() {
                Ok(n) if n < 64 => 1usize << n,
                _ => {
                    eprintln!("MACVM_BENCH_CPU={spec}: expected `perf` or a logical CPU index");
                    return;
                }
            }
        };
        unsafe {
            if SetThreadAffinityMask(GetCurrentThread(), mask) == 0 {
                eprintln!("MACVM_BENCH_CPU: SetThreadAffinityMask({mask:#x}) failed; not pinned");
                return;
            }
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        }
        eprintln!("[bench] pinned to logical CPU mask {mask:#x}, HIGH priority");
    }
}
