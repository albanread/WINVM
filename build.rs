//! Build script — exists solely to compile the Cocoa bridge's Objective-C
//! exception shim (docs/cocoa_bridge_design.md §5): an `NSException`
//! unwinding through Rust/JIT frames is undefined behavior, so every
//! bridged `objc_msgSend` goes through a tiny `.m` that CALLS the send
//! inside `@try` and reports a caught exception as a status + description
//! instead of unwinding.
fn main() {
    // WINVM: the Objective-C shim is macOS-only; on Windows there is no Cocoa
    // bridge to protect, so the build script is a no-op.
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .file("src/runtime/objc_shim.m")
            .flag("-fobjc-exceptions")
            .compile("macvm_objc_shim");
        println!("cargo:rustc-link-lib=objc");
    }
    println!("cargo:rerun-if-changed=src/runtime/objc_shim.m");
}
