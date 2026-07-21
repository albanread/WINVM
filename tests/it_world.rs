//! World + SUnit-lite integration tests (`tests_s06.md` "Integration /
//! golden tests" and "Stress / negative tests"). Most of these run
//! in-process (`Smalltalk quit:` just sets `vm.exit_code`, it never calls
//! `std::process::exit` — only the `error:` primitive does that) so only
//! the DNU/`error:` case needs a real subprocess.

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use macvm::frontend::world;
use macvm::runtime::vm_state::OutputBuffer;

fn world_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("world")
}

/// Loads `world/tests/tests.list` (NOT named `world.list`, so
/// `frontend::world::load_world` doesn't apply directly) relative to
/// `world/tests/`, in order, stopping early if a file requests exit.
fn load_tests_list(vm: &mut macvm::runtime::VmState) {
    let dir = world_dir().join("tests");
    let list_src = std::fs::read_to_string(dir.join("tests.list")).expect("read tests.list");
    for raw_line in list_src.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // WINVM: a `posix-only:` entry exercises POSIX syscalls with no
        // Windows equivalent (`30_ffi_alien_tests` calls `getpid` and
        // `mmap` through the FFI). It is excluded here rather than left
        // to fail, because on Windows the FFI stub raises a GUEST FATAL —
        // which `fatal_exit` turns into `process::exit`, killing the test
        // harness outright. The whole 5891-assertion suite was
        // unreportable because of seven POSIX assertions.
        //
        // Announced, never silent: a skipped file that says nothing is
        // how a suite quietly stops covering something.
        let line = match line.strip_prefix("posix-only:") {
            Some(rest) if cfg!(windows) => {
                eprintln!("[tests.list] SKIP {rest} — POSIX-only, not available on this host");
                continue;
            }
            Some(rest) => rest,
            None => line,
        };
        // The mirror image: a suite that exercises Win32/COM has nothing
        // to run against on macOS. Both tags exist so neither platform's
        // coverage is expressed as the absence of the other's.
        let line = match line.strip_prefix("win32-only:") {
            Some(rest) if !cfg!(windows) => {
                eprintln!("[tests.list] SKIP {rest} — Win32-only, not available on this host");
                continue;
            }
            Some(rest) => rest,
            None => line,
        };
        world::load_file(vm, &dir.join(line)).unwrap_or_else(|e| panic!("{line}: {e}"));
        if vm.exit_requested {
            break;
        }
    }
}

#[test]
fn boots_clean() {
    let mut vm = common::test_vm();
    let buf = OutputBuffer::new();
    vm.out = Box::new(buf.clone());
    let loaded = world::load_world(&mut vm, &world_dir()).expect("load_world");
    assert!(loaded);
    assert_eq!(
        buf.as_string(),
        "",
        "world.list alone must produce no output"
    );
}

/// Boots the real world, runs `src`'s top-level doIts, and returns what they
/// printed to the Transcript — the end-to-end path a `.mst` file / the REPL
/// take (needs the full world: `Array`, `at:put:`, `printString`).
fn eval_world_transcript(src: &str) -> String {
    let mut vm = common::test_vm();
    let buf = OutputBuffer::new();
    vm.out = Box::new(buf.clone());
    world::load_world(&mut vm, &world_dir()).expect("load_world");
    for item in macvm::frontend::parser::parse_file(src).expect("parse") {
        macvm::frontend::classdef::execute_top_item(&mut vm, item).expect("execute");
    }
    buf.as_string()
}

/// Squeak/Pharo brace (dynamic) arrays `{ e1. e2. … }`: elements are
/// runtime EXPRESSIONS (not `#( … )`'s compile-time literals), a fresh
/// `Array` per evaluation. Codegen desugars to `(Array new: n)` + `at:put:`,
/// so this rides the real world's primitives end to end — arithmetic,
/// string concat, `size`, nesting, and a genuinely runtime element (a block
/// parameter) that a literal array could not express.
#[test]
fn brace_arrays_build_runtime_arrays() {
    let out = eval_world_transcript(
        "Transcript showCr: { 1 + 1. 2 * 3. 'a', 'b' } printString.\n\
         Transcript showCr: { 10. 20. 30 } size printString.\n\
         Transcript showCr: {} printString.\n\
         Transcript showCr: { 1. { 2. 3 }. 4 } printString.\n\
         Transcript showCr: ([ :x | { x. x * x } ] value: 9) printString.\n",
    );
    assert_eq!(out, "#(2 6 'ab')\n3\n#()\n#(1 #(2 3) 4)\n#(9 81)\n");
}

/// Gate item 1 (`tests_s06.md`): boot, load world.list + tests.list, run
/// SUnit-lite, exit 0 with a `/^\d+ run, 0 failed$/` report line, total
/// assertions >= 200.
#[test]
fn suite_green() {
    let mut vm = common::test_vm();
    let buf = OutputBuffer::new();
    vm.out = Box::new(buf.clone());
    world::load_world(&mut vm, &world_dir()).expect("load_world");
    load_tests_list(&mut vm);

    assert_eq!(vm.exit_code, Some(0), "stdout so far:\n{}", buf.as_string());
    let out = buf.as_string();
    let report_line = out
        .lines()
        .find(|l| l.ends_with("failed"))
        .unwrap_or_else(|| panic!("no '... failed' report line in:\n{out}"));
    assert!(
        report_line.ends_with(", 0 failed"),
        "expected 0 failures, got: {report_line}"
    );
    let run_count: u64 = report_line
        .split(' ')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("couldn't parse run count from: {report_line}"));
    assert!(
        run_count >= 200,
        "expected >= 200 assertions run, got {run_count}"
    );
}

/// Load-order torture: swapping files 12 (Dictionary, needs OrderedCollection
/// from 15... actually here we swap 12_string.mst and 19_printing.mst, which
/// both genuinely depend on earlier files) must fail fast with an
/// "undeclared variable" error — proves the ordering law is real.
#[test]
fn order_is_load_bearing() {
    let tmp = std::env::temp_dir().join(format!("macvm_order_torture_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).ok();
    // Copy the real world files, but list 19_printing.mst (which needs
    // WriteStream from 18 and OrderedCollection from 15) before them.
    for entry in std::fs::read_dir(world_dir()).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().and_then(|e| e.to_str()) == Some("mst") {
            std::fs::copy(entry.path(), tmp.join(entry.file_name())).unwrap();
        }
    }
    std::fs::write(
        tmp.join("world.list"),
        "19_printing.mst\n01_object.mst\n02_nil_boolean.mst\n04a_blockclosure.mst\n\
         04_transcript.mst\n03_behavior.mst\n05_magnitude.mst\n06_smallinteger.mst\n\
         07_largeinteger.mst\n08_double.mst\n10_array.mst\n09_character.mst\n\
         11_bytearray.mst\n12_string.mst\n13_symbol.mst\n14_association.mst\n\
         15_ordered.mst\n16_dictionary.mst\n17_interval.mst\n18_writestream.mst\n\
         20_system.mst\n",
    )
    .unwrap();

    let mut vm = common::test_vm();
    let err = world::load_world(&mut vm, &tmp).expect_err("swapped load order must fail");
    assert!(
        err.msg.contains("undeclared variable") || err.msg.contains("not understand"),
        "expected an undeclared-variable-shaped failure, got: {}",
        err.msg
    );

    std::fs::remove_dir_all(&tmp).ok();
}

/// Reopen misuse: declaring an existing class's superclass differently on
/// reopen must fail with a shape/superclass-mismatch error, not silently
/// succeed or corrupt the klass.
#[test]
fn reopen_misuse_wrong_superclass_fails() {
    let mut vm = common::test_vm();
    world::load_world(&mut vm, &world_dir()).expect("load_world");
    let items = macvm::frontend::parser::parse_file("Object subclass: Integer [ ]\n").unwrap();
    let mut saw_error = false;
    for item in items {
        if let Err(e) = macvm::frontend::classdef::execute_top_item(&mut vm, item) {
            saw_error = true;
            assert!(
                e.msg.contains("shape") || e.msg.contains("superclass"),
                "expected a shape/superclass-mismatch message, got: {}",
                e.msg
            );
        }
    }
    assert!(
        saw_error,
        "reopening Integer under the wrong superclass must fail"
    );
}

/// A deliberately-failing test class proves the suite can actually fail
/// (guards against a vacuously-green runner) — exit code 1, failure line
/// names the test.
#[test]
fn deliberately_failing_assertion_fails_the_run() {
    let mut vm = common::test_vm();
    let buf = OutputBuffer::new();
    vm.out = Box::new(buf.clone());
    world::load_world(&mut vm, &world_dir()).expect("load_world");
    let dir = world_dir().join("tests");
    world::load_file(&mut vm, &dir.join("00_sunit.mst")).expect("load 00_sunit.mst");

    let src = "TestCase subclass: DeliberatelyFailingTests [\n\
        runAll [ self runTest: #testBoom do: [ self testBoom ] ]\n\
        testBoom [ self assert: 1 = 2 ]\n\
    ]\n\
    TestRunner start.\n\
    TestRunner run: DeliberatelyFailingTests.\n\
    TestRunner report.\n";
    let items = macvm::frontend::parser::parse_file(src).unwrap();
    for item in items {
        macvm::frontend::classdef::execute_top_item(&mut vm, item).expect("execute");
        if vm.exit_requested {
            break;
        }
    }
    assert_eq!(vm.exit_code, Some(1));
    let out = buf.as_string();
    assert!(
        out.contains("testBoom"),
        "failure line must name the failing test, got:\n{out}"
    );
}

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_macvm"))
}

/// `error:`'s primitive calls `std::process::exit` directly — the only
/// case in this file that genuinely needs a subprocess rather than
/// in-process `VmState` inspection.
#[test]
fn error_kills_with_trace() {
    let dir = std::env::temp_dir().join(format!("macvm_error_trace_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("boom.mst");
    std::fs::write(&script, "nil foo.\n").unwrap();

    let out = Command::new(bin_path())
        .args([
            "run",
            script.to_str().unwrap(),
            "--world",
            world_dir().to_str().unwrap(),
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn macvm");

    assert_ne!(out.status.code().unwrap_or(-1), 0);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("foo"),
        "expected the selector name in the error, got stdout:\n{stdout}"
    );
    assert!(
        stdout.lines().count() >= 2,
        "expected the message line plus >=1 stack-trace line, got:\n{stdout}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Regression: raising an error from a frame that was entered from COMPILED
/// code must not crash the VM's own error reporter. `print_stack_trace` chases
/// `saved_fp` frame-to-frame; at the interpreter/compiled boundary that slot
/// isn't a valid interpreter word, and the old panicking readers aborted the
/// whole process (`Frame::saved_fp: not a smi`, SIGABRT — a GUI bug report). A
/// clean guest error exits with a normal code; a SIGABRT is a SIGNAL kill, so
/// `status.code()` is `None` — that (not the exit VALUE) is what this asserts.
#[test]
fn error_from_compiled_frame_does_not_abort() {
    let dir = std::env::temp_dir().join(format!("macvm_compiled_err_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("boom.mst");
    // `work:` is called ~100x so it tiers up; at i>40 it raises from inside the
    // now-compiled frame — the exact shape of the reported crash.
    std::fs::write(
        &script,
        "Object subclass: BoomC [\n\
         \x20 BoomC class >> work: i [ i > 40 ifTrue: [ ^self error: 'boom' ]. ^i + 1 ]\n\
         \x20 BoomC class >> run [ | s | s := 0. 1 to: 100 do: [:i | s := s + (self work: i) ]. ^s ]\n\
         ]\n\
         BoomC run.\n",
    )
    .unwrap();

    let out = Command::new(bin_path())
        .args([
            "run",
            script.to_str().unwrap(),
            "--world",
            world_dir().to_str().unwrap(),
        ])
        .env("MACVM_JIT", "threshold=10")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn macvm");

    // The crash was a SIGABRT → killed by a signal → `code()` is None. A clean
    // guest error exits with an actual code. So: it must have exited normally.
    assert!(
        out.status.code().is_some(),
        "VM was killed by a signal (the print_stack_trace abort), status: {:?}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("not a smi"),
        "the error reporter panicked, stderr:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("boom"),
        "expected the error message in stdout, got:\n{stdout}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// S24 A1 (adversarial-review BLOCKER 2): a compiled NLR block whose home
/// frame already returned must deliver `#cannotReturn:` to the ORIGINATING
/// closure — `rt_nlr_originate` parks it in `NlrState.closure`; the block's
/// own native frame is gone by liveness-check time, so reading the current
/// frame's receiver slot would name the value-sender's receiver instead.
/// The world defines no `cannotReturn:`, so the correct observable is the
/// fatal DNU cascade NAMING the closure's class — identically under both
/// tiers. (Repro ledger: tests/repros/closure_dead_home_cannot_return.mst.)
#[test]
fn dead_home_nlr_names_the_closure_both_tiers() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/repros/closure_dead_home_cannot_return.mst");
    for jit in ["off", "threshold=1"] {
        let out = Command::new(bin_path())
            .args([
                "run",
                script.to_str().unwrap(),
                "--world",
                world_dir().to_str().unwrap(),
            ])
            .env("MACVM_JIT", jit)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("spawn macvm");

        assert_ne!(
            out.status.code().unwrap_or(-1),
            0,
            "dead-home NLR must be fatal (MACVM_JIT={jit})"
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("invoking"),
            "the doit must reach the invocation first (MACVM_JIT={jit}), got:\n{stdout}"
        );
        assert!(
            stdout.contains("cannotReturn:") && stdout.contains("BlockClosure"),
            "expected the DNU to name #cannotReturn: on the closure's class \
             (MACVM_JIT={jit}), got:\n{stdout}"
        );
        assert!(
            !stdout.contains("unreachable"),
            "execution must not continue past the dead-home NLR (MACVM_JIT={jit})"
        );
    }
}
