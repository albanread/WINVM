//! **Requires the `gamepane` feature** (macOS + a MacGamePane checkout beside
//! this repo). It is off by default: `../MIGRATION.md` §1 defers the game
//! demos on Windows, and the MacGamePane path dependencies are not declared in
//! this checkout at all (see `Cargo.toml`'s `gamepane` feature for how to
//! restore them). Without the gate this file would fail to compile on every
//! default build rather than simply not running.
#![cfg(feature = "gamepane")]

//! M0 (docs/gamepane_design.md, "Milestone ladder"): prove the metal-crate
//! coupling. This test exists to retire the single biggest integration unknown
//! *before* any Smalltalk surface is built: that MACVM's `gui` crate can link
//! `metal = "0.33"` + the MacGamePane `graphics` crate (across the sibling-repo
//! path dependency) and drive one `IndexedPane` frame headless, end to end
//! through the GPU pipeline, into an offscreen texture.
//!
//! It is deliberately a `gui`-crate integration test (not in MacGamePane),
//! because the thing under test is *the coupling from MACVM's side* — the
//! boundary types matching, the pipeline compiling, the render completing —
//! not the engine, which has its own tests.

use macgamepane_graphics::indexed_pane::IndexedPane;

#[test]
fn indexed_pane_renders_one_frame_headless_into_an_offscreen_texture() {
    // A Metal device is required. On a headless box without one, skip rather
    // than fail: the coupling (linking metal 0.33 + the graphics crate + the
    // matching boundary types) is already proven by the fact this compiled.
    let device = match metal::Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!(
                "no Metal device on this machine; skipping the headless render \
                 (metal-crate coupling already proven by successful linking)"
            );
            return;
        }
    };

    let (vw, vh) = (320u32, 240u32);
    let mut pane =
        IndexedPane::new(&device, 640, 480, vw, vh).expect("IndexedPane::new should succeed");

    // Palette index 16 = a known opaque colour. `set_rgb` asserts index >= 16
    // (index 0 is transparent), so 16 is the lowest legal user colour. Clear
    // the FRONT buffer to it and upload CPU -> GPU.
    let (r, g, b) = (200u8, 128u8, 64u8);
    pane.set_rgb(16, r, g, b);
    pane.cls(16);
    pane.upload();

    // Offscreen render target in the pane pipeline's colour-attachment format
    // (BGRA8Unorm), with Shared storage so we can read the pixels back on
    // Apple Silicon's unified memory.
    let tex_desc = metal::TextureDescriptor::new();
    tex_desc.set_texture_type(metal::MTLTextureType::D2);
    tex_desc.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    tex_desc.set_width(vw as u64);
    tex_desc.set_height(vh as u64);
    tex_desc.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    tex_desc.set_storage_mode(metal::MTLStorageMode::Shared);
    let target = device.new_texture(&tex_desc);

    let queue = device.new_command_queue();
    let cb = queue.new_command_buffer();
    pane.render(cb, &target, metal::MTLLoadAction::Clear);
    cb.commit();
    cb.wait_until_completed();

    assert_eq!(
        cb.status(),
        metal::MTLCommandBufferStatus::Completed,
        "the render command buffer must complete without error"
    );

    // Read the top-left pixel back and confirm the pane painted our palette
    // colour all the way through the GPU pipeline. BGRA8 byte order is B,G,R,A.
    let mut px = [0u8; 4];
    target.get_bytes(
        px.as_mut_ptr() as *mut std::ffi::c_void,
        4, // bytes per row for the single-pixel region we read
        metal::MTLRegion {
            origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal::MTLSize { width: 1, height: 1, depth: 1 },
        },
        0,
    );

    assert_eq!(px[2], r, "R channel (BGRA[2])");
    assert_eq!(px[1], g, "G channel (BGRA[1])");
    assert_eq!(px[0], b, "B channel (BGRA[0])");
    assert_eq!(px[3], 255, "opaque alpha — palette index 16 is not transparent");
}
