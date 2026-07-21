# WINVM — Migration Design: MACVM → x86-64 Windows

**Goal.** Produce WINVM, an equivalent of [MACVM](https://github.com/albanread/MACVM)
(two-tier Smalltalk VM: baseline interpreter + adaptive optimizing JIT, generational
GC, direct-pointer object model, `.mst` source worlds) running natively on Windows
x86-64.

**The decisive advantage.** The hard platform work already exists in this workshop:

- **E:\JASM** — the *Windows x86-64* JIT assembler this whole family grew from.
  MACVM's vendored `src/vendor/wfasm/native_macos.rs` describes itself as *"the
  AArch64 sibling of the Windows `NativeJit` (`crate::native`)"* — i.e. the Windows
  loader, the pure-Rust x64 encoder (`rasm/encode.rs`, incl. SSE2 scalar-double:
  `movsd/addsd/mulsd/ucomisd/cvtsi2sd`), Win32 API bindings, and an SEH crash
  dumper (`seh.rs`) with int3-continue semantics are **already written and
  MIT-licensed by the same author**.
- **E:\WF66** — a shipping Windows-x64 JIT VM (Forth) built on that exact substrate
  (~1.5 MB Rust + MASM kernel). It proves the encoder, W^X handling, code-cache
  discipline, and the interpreter-as-oracle differential test methodology on this
  exact machine.

So this is not a from-scratch port: it is *re-vendoring the other sibling* and
rewriting the architecture-specific back half of the compiler.

---

## 1. Component disposition (measured from the MACVM checkout)

| Module | LoC (Rust) | Disposition |
|---|---|---|
| `src/bytecode/` | 1,641 | **Portable.** Bytecode ISA is arch-neutral by design (docs/ISA.md, 31bytecodes.md). |
| `src/frontend/` | 5,965 | **Portable.** `.mst` parser/loader. |
| `src/interpreter/` | 5,471 | **Portable** minus 1 `compiled_call.rs` seam (tier-0→tier-1 call convention). |
| `src/oops/` | 2,626 | **Portable.** 2-bit tagging, 16-byte alignment — identical on x64. |
| `src/memory/` | 7,426 | **Portable logic**; swap `mmap` → `VirtualAlloc/VirtualFree` behind the existing native layer. |
| `src/runtime/` | 18,665 | **Mostly portable.** Arch seams: `frames.rs` (FP-chain walking → RBP chains), `deopt.rs`, `vm_state.rs` (pinned-register mirror), `simd_kernels.rs` (NEON `asm!` → SSE intrinsics). |
| `src/compiler/` front/mid: `ir.rs` (385 KB), `driver.rs`, `inline.rs`, `escape.rs`, `feedback.rs`, `scopes.rs`, `decode.rs` | ~15 k | **Portable.** IR, inlining, type feedback, escape analysis are arch-neutral. |
| `src/compiler/` back end: `assembler.rs`, `emit.rs` (142 KB), `regalloc.rs` (70 KB), `oopmap.rs`, `jasm_assembler.rs`, `disasm_a64.rs` | ~6 k+ | **Rewrite for x64.** This is the core of the port. |
| `src/codecache/` | 9,061 | **Heavy arch content.** `stubs.rs` (157 KB) and `deopt_trap.rs` (116 KB) emit AArch64 and use Mach signal traps; `pics.rs`, `adapters.rs`, `nmethod.rs` patch fixed-width A64 sites; `flush.rs` does icache maintenance. All need x64 equivalents — but each has a design doc in `docs/`. |
| `src/vendor/wfasm/` | 8,239 | **Re-vendor from E:\JASM** — swap `a64/` encoder for `rasm/`, `native_macos.rs` for `native.rs`/`win32.rs`, keep `relocpatch.rs` shape with x64 reloc kinds. The vendor header explicitly says "keep the diff against upstream minimal so re-vendoring stays mechanical." |
| `gui/` (WKWebView) | — | **Port shell to WebView2**; the HTML/CSS/JS Strongtalk environment itself is portable. |
| `cocoa_gui/` (AppKit) | — | **Defer / replace** with a Win32-native shell later (JASM's generated Win32 bindings + WF66's `igui` experience are the raw material). |
| Metal/AVFoundation game demos | — | **Defer.** D3D11/XAudio2 stretch goal. |
| `QBEJIT/` (1.5 MB C) | — | Vendored experiment; **exclude** from the Windows build unless `cargo build` proves otherwise. |
| `world/`, `image_store/` (SQLite), `rusttcl/`, tests | — | **Portable.** The 107-class/1,269-method world is plain `.mst` text + SQLite. |

Roughly: **~75 % of the VM ports with cfg-gating and a native-layer swap; ~25 %
(compiler back end + codecache) is a genuine x64 rewrite** — with every subsystem
already specified by a `docs/*.md` design note and covered by an existing test
corpus.

---

## 2. Architecture mapping decisions

### 2.1 Pinned VM registers (docs/VMregisters.md → Win64)

Windows x64 callee-saved: `RBX RBP RSI RDI R12–R15`, `XMM6–15`. Volatile:
`RAX RCX RDX R8–R11`, `XMM0–5`. Args: `RCX RDX R8 R9` + 32-byte shadow space.

| MACVM (A64) | Role | WINVM (x64) |
|---|---|---|
| `x28` | VM/thread state | `R15` |
| `x27` | receiver cache | `R14` |
| `x26` | bytecode ptr (tier 0) | `R13` |
| `x25` | method/frame info | `R12` |
| `x29` | frame pointer | `RBP` (keep real frame chains — GC/debugger walk them) |
| `x30` | link register | *(none — x64 uses stack return addresses; frame layout shifts by one slot)* |
| `x16`/`x17` | scratch | `R10`/`R11` (canonical Win64 scratch, never args) |
| allocatable pool (~18 regs) | JIT vregs | `RBX RSI RDI RCX RDX R8 R9` (+`RAX` as result/scratch) — **~7 regs** |

**Consequences to design for, not discover later:**

- **Register pressure.** 7 allocatable vs ~18. Expect more spills; consider
  demoting one pinned register (receiver cache `R14` is the best candidate —
  measure first) to reclaim a callee-saved reg for the allocator.
- **Two-address form.** x64 `dst = dst op src` vs A64's three-address form. This
  is the deepest semantic change in `regalloc.rs`/`emit.rs` — the allocator must
  coalesce dst/src1 or emit `mov` fixups. The self-aliasing hazards documented in
  `emit.rs`'s header (spilled operands landing in scratch regs that the same
  sequence clobbers) get *more* frequent on x64; carry that analysis forward
  deliberately.
- **Fixed-register ops.** `div`/`idiv` pin RAX:RDX; variable shifts pin CL;
  `imul` overflow is just the OF flag (simpler than A64's `smulh` dance).
- **Flags.** x64 compare→branch is a natural fusion (WF66 already does
  `0< if` → one `jcc`); the A64 `SmiCmpBr` patterns map cleanly.

### 2.2 W^X, code cache, and traps — *simpler* on Windows

| Concern | macOS/A64 (current) | Windows/x64 (target) |
|---|---|---|
| Executable memory | `mmap(MAP_JIT)` RWX + per-thread `pthread_jit_write_protect_np` toggle | `VirtualAlloc` one large `MEM_RESERVE` region, commit as `PAGE_EXECUTE_READWRITE` (or RW→RX `VirtualProtect` flips if W^X hygiene is wanted). No per-thread toggle exists or is needed. |
| I-cache | `sys_icache_invalidate` mandatory (split I/D) | x86 is I/D-coherent; call `FlushInstructionCache` pro forma. `flush.rs` becomes nearly trivial. |
| Deopt traps | `brk #imm` + Mach exception/signal | `int3` (single byte 0xCC) + **Vectored Exception Handler**. `E:\JASM\rust\src\seh.rs` already demonstrates int3-dump-and-continue — direct substrate for `deopt_trap.rs`. |
| Far calls | `movz/movk x16; br x16` veneers beyond ±128 MB | Reserve the whole code cache in one ≤2 GB region → all intra-cache calls are `rel32`; host externs via `movabs rax; call rax` (the veneer analogue `native_macos.rs` itself cites). |
| Patchable PIC sites | fixed-width 4-byte instructions, patch in place | variable-width: **plan patch-site shapes up front** — e.g. always emit `call rel32` (5 bytes, rel32 field aligned or patched via single 8-byte aligned write / two-step int3 protocol). Single-threaded execution (MACVM model) makes this easy; document the invariant. |
| Stack walking | FP chain via `x29` | RBP chain (identical walk). OS-level SEH unwind info (`RtlAddFunctionTable`) **not required** in phase 1 since VEH + RBP chains cover traps and GC; add unwind registration later only if C-FFI callbacks must unwind through JIT frames. |

### 2.3 SIMD & floats

- Float regions (unboxed doubles): scalar SSE2 (`movsd/addsd/mulsd/ucomisd`) —
  already in `rasm`'s encoder. XMM6–15 are callee-saved on Win64: nice for
  float-loop residency, but prologs must save them if used.
- `Float64x2` → SSE2 `movapd/addpd/...` 1:1 with NEON `v2d`. `Float32x4`/`Int32x4`
  → SSE/SSE2 likewise. `runtime/simd_kernels.rs` `asm!` blocks →
  `core::arch::x86_64` intrinsics (no inline-asm needed).

### 2.4 Disassembler

Don't hand-write `disasm_x64.rs` (the A64 one is 50 KB). Use the **`iced-x86`**
crate (pure Rust, no deps) for trace/debug disassembly, or reuse JASM's
LLVM-MC-based difftest oracle when LLVM is present.

---

## 3. Repository strategy

Follow the WF65→WF66 house pattern: **WINVM is a new repo seeded from MACVM**, with
MACVM kept as an `upstream` remote so portable-layer fixes cherry-pick both ways.

- Keep module *names and layout* identical (`src/compiler/emit.rs` stays
  `emit.rs`, but emits x64) rather than growing `cfg(target_arch)` forests — the
  two repos are siblings sharing docs and world, like the vendored-JASM
  discipline already in place.
- The **truly shared, arch-neutral core** (bytecode ISA, `.mst` world, IR,
  frontend) should drift toward being byte-identical between repos; a later
  unification into one dual-target repo is possible but is explicitly *not* a
  phase-1 goal.

---

## 4. Phases and milestones

Each phase ends with a runnable acceptance test. The Mac build is the behavioral
oracle throughout: same `.mst` world, same bytecode ISA, same test suite —
outputs must match.

### Phase 0 — Seed and compile-out (days)
1. `git clone` MACVM → `E:\WINVM`, add `upstream` remote, branch `windows-port`.
2. Make `cargo check` pass on Windows with the JIT compiled out:
   cfg-gate `native_macos.rs`, `codecache`'s trap/stub emitters, `simd_kernels`
   asm, GUI shells. Stub the tier-1 entry points to "always interpret".
3. Exclude `QBEJIT`, `gui/`, `cocoa_gui/` from the Windows build.

**Milestone M0:** `cargo build` succeeds; unit tests of portable modules pass.

### Phase 1 — Interpreter-only Smalltalk on Windows (1–2 weeks)
1. Native memory layer: `mmap/munmap` → `VirtualAlloc/VirtualFree` for heap
   spaces (generational GC logic is untouched).
2. Boot the world: 107 classes / 1,269 methods load from `.mst` + SQLite.
3. Run the full test suite and Richards/DeltaBlue **interpreted**; diff results
   against the Mac build.

**Milestone M1:** `winvm run world/bench/deltablue.mst --world world` gives
correct results with `MACVM_JIT=off` semantics. *This is already a working
Smalltalk on Windows.*

### Phase 2 — JIT substrate: re-vendor the Windows sibling (1 week)
1. Re-vendor from `E:\JASM`: `rasm` x64 encoder, Windows `NativeJit`
   (`native.rs`), `relocpatch` with x64 reloc kinds (rel32, abs64), `win32.rs`.
   Mirror the existing `VENDOR.md` + `// MACVM:` marker discipline.
2. Port `CodeCache` region management: one reserved ≤2 GB region, segment
   publish protocol, `FlushInstructionCache` in `flush.rs`.
3. Stand up the VEH: adapt `seh.rs`'s int3 dump-and-continue into the
   `deopt_trap` entry path; crash dumper doubles as the JIT debugging story.
4. Encoder confidence: run JASM's difftest / `rasm-diff` oracle over the
   instruction subset MACVM's emitter will use.

**Milestone M2:** hand-built x64 blob JITted into the code cache, called from
Rust, patched, and an `int3` deopt round-trips through the VEH back to the
interpreter.

### Phase 3 — Compiler back end (the core effort, 4–8 weeks)
1. `assembler.rs`: x64 operand DSL (`rax()`, `xmm(n)`, `mem(base, disp)`,
   labels, reloc kinds) over the vendored `rasm` encoder — mirror of the A64 one.
2. `regalloc.rs`: new register file (7-reg pool), two-address coalescing,
   fixed-reg constraints (div/shift), spill slots at `[rbp − 8·(i+1)]`
   (same scheme, RBP-relative).
3. `emit.rs`: port op-by-op in the order the IR docs define. Each A64 sequence
   in D5.3 gets an x64 counterpart; re-derive the spilled-operand aliasing
   analysis for two-address form. Simplifications: `SmiArith::Mul` overflow =
   `imul` + `jo`; tag checks = `test reg, 3`.
4. `oopmap.rs`: x64 register numbering for GC maps.
5. `stubs.rs` / `adapters.rs`: allocation, send, adapter stubs in x64.
6. Wire tier-up: hot-method counter → compile → `compiled_call.rs` entry.
7. **Differential testing from day one**: interpreter-vs-JIT oracle per method
   (the WF66 methodology), plus the vendored `corpus_replay.rs` pattern for
   encoder-level golden tests.

**Milestone M3:** Richards and DeltaBlue run tier-1 compiled with correct
results; ≥90 % of hot methods compile without bailout.

### Phase 4 — Adaptive machinery (2–4 weeks)
1. `pics.rs`: PIC patch protocol for x64 patch-site shapes (§2.2).
2. `deopt_trap.rs`: full deopt metadata → VEH → frame reconstruction; OSR
   entry/exit for x64 frame layout (`nmethod.rs` frame descriptors).
3. Mixed-tier GC: frame walking over RBP chains across interpreter and
   compiled frames; oop-map lookup at safepoints; derived-pointer handling.
4. Inlining + block splicing + non-local returns re-validated (logic is IR-level
   and ports; the *landing pads* are arch-specific).

**Milestone M4:** full benchmark suite matches Mac results with JIT on; GC
stress test (forced scavenges during compiled execution) passes; deopt storms
absent per `dual_arm_design.md`'s tiered-trap policy.

### Phase 5 — Floats & SIMD (1–2 weeks)
1. Float regions on XMM; measure Mandelbrot against the Mac's ~5.5×-off-C figure.
2. `Float64x2`/`Float32x4`/`Int32x4` value classes on SSE2; port
   `simd_kernels.rs` to intrinsics.

**Milestone M5:** float/SIMD benchmarks pass and are within ~2× of the Mac's
relative-to-C ratios.

### Phase 6 — Environment (parallel/ongoing)
1. **HTML GUI on WebView2** (`webview2-com` or `wry`): the Strongtalk hypertext
   environment, class browser, SQLite-indexed find tools, workspace, metrics
   dashboard — the web assets port as-is; only the shell changes.
2. Win32-native shell (the `cocoa_gui` analogue, UI written in Smalltalk driving
   real Win32 controls) as a second wave.
3. Game demos on D3D11/XAudio2 — stretch.

**Milestone M6:** class browser + workspace usable on Windows.

---

## 5. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Register pressure (7-reg pool) degrades JIT quality | Perf gap vs Mac | Measure before pinning all four VM regs; demote receiver-cache first; XMM residency for float loops; WF66's `promote_hot_cells` experience applies. |
| Two-address aliasing bugs in `emit.rs` | Silent corruption | Port the header's aliasing analysis as a checklist per op; interpreter-differential oracle on every method; keep `tst`-style non-writing checks (`test`) wherever possible. |
| Patching non-atomic on variable-width ISA | Rare crashes | Single-threaded execution invariant documented + enforced; fixed patch-site shapes; int3 two-step protocol if threading arrives. |
| `rasm` encoder gaps (instructions MACVM needs that WF66 didn't) | Blocked emit work | Enumerate emit.rs's mnemonic set in Phase 3 step 1; extend `rasm` + difftest against LLVM-MC per JASM's existing methodology. |
| Windows unwind interop (C FFI callbacks through JIT frames) | FFI crashes later | Out of scope until `docs/FFI.md` port; add `RtlAddFunctionTable` registration then. |
| WebView2 runtime availability | GUI phase | Ships with Win11 (this machine); Evergreen bootstrap for distribution. |

---

## 6. What deliberately does *not* change

- Bytecode ISA, `.mst` format, world sources, SQLite image store — **identical**,
  so worlds and tests are shared verbatim with MACVM.
- The no-`become:`, rebuild-from-source philosophy — unchanged; sub-second world
  rebuild is the compatibility guarantee between the two VMs.
- IR, type feedback, inlining, escape analysis, deopt *policy* — arch-neutral by
  construction; only their machine-level *rendering* is new.
