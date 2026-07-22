//! M7 (`docs/package_aware_editing_design.md` §4.5/§6): `macvm-gui run
//! <file> [--from-image]` boots a program either `.mst`-direct (the default,
//! same as the bare `macvm run`) or from the SQLite image, and the two must
//! produce IDENTICAL output — the whole point of the image being a faithful
//! projection of the `.mst` world. `image_store`'s own
//! `db_booted_world_matches_mst_booted_world` proves the boot equivalence at
//! the library level; this proves it end to end through the real `macvm-gui`
//! binary, exercising the actual CLI both ways.

use std::path::PathBuf;
use std::process::Command;

fn world_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../world")
}

/// A small program that touches enough of the world (arithmetic, a
/// collection, string building) that a broken or partial boot would diverge
/// — printed to stdout via `Transcript`, the CLI's own output surface.
const PROGRAM: &str = "\
Transcript showCr: (3 + 4) printString.
Transcript showCr: ((1 to: 5) inject: 0 into: [:a :b | a + b]) printString.
Transcript showCr: (OrderedCollection new add: #alpha; add: #beta; yourself) printString.
Transcript showCr: 'winvm-gui run: done'.
";

#[test]
fn run_from_image_boots_identically_to_mst_direct() {
    let bin = env!("CARGO_BIN_EXE_winvm-gui");
    let tag = std::process::id();
    let prog = std::env::temp_dir().join(format!("macvm_m7_prog_{tag}.mst"));
    let image = std::env::temp_dir().join(format!("macvm_m7_image_{tag}.sqlite3"));
    std::fs::write(&prog, PROGRAM).expect("write the test program");
    std::fs::remove_file(&image).ok();

    // Run the real binary; capture stdout. JIT off for a deterministic run
    // (the boot-source equivalence is independent of the execution tier).
    let run = |from_image: bool| -> String {
        let mut cmd = Command::new(bin);
        cmd.arg("run")
            .arg(&prog)
            .arg("--world")
            .arg(world_dir())
            .env("MACVM_JIT", "off");
        if from_image {
            cmd.arg("--from-image").env("MACVM_IMAGE_PATH", &image);
        }
        let out = cmd.output().expect("spawn macvm-gui run");
        assert!(
            out.status.success(),
            "macvm-gui run (from_image={from_image}) failed: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let mst_direct = run(false);
    // Sanity: the default path actually ran the program (not an empty world).
    assert!(
        mst_direct.contains("winvm-gui run: done") && mst_direct.contains('7'),
        "the .mst-direct run produced unexpected output:\n{mst_direct}"
    );

    let from_image = run(true); // seeds the image on first open, then boots from it
    assert_eq!(
        mst_direct, from_image,
        "--from-image must boot identically to the .mst-direct default"
    );

    std::fs::remove_file(&prog).ok();
    std::fs::remove_file(&image).ok();
}
