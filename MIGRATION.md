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

## 6. Status log

- **2026-07-21 — M0 done.** `cargo build` clean (zero warnings) on
  x86-64 Windows. OS seams ported: `memory/reservation.rs` (VirtualAlloc),
  `runtime/probe.rs` (thread stack bounds), `runtime/vm_state.rs`
  (ExitThread), `codecache/deopt_trap.rs` (Mach/signal layer gated to
  macOS; portable thread ids; recovery-jump stubs pending the Phase-2
  VEH), new `vendor/wfasm/native_windows.rs` (`WinJit`), Cocoa bridge +
  prims gated with clean-fail Windows stubs. The entire compiler/JIT
  middle end compiles unmodified — it is pure Rust emitting A64 bytes.
- **2026-07-21 — M1 done.** World boots (107 classes) from `.mst` +
  SQLite on Windows; DeltaBlue + Richards produce correct results
  interpreted; world test suite minus the four FFI-dependent files
  (ffi_alien, posix_io, socket, accel): **5891 run, 0 failed**.
  Interpreted release timings on this machine: DeltaBlue ×10 = 126 ms,
  Richards ×10 = 1135 ms — the tier-1 x64 backend (Phase 3) is where the
  Mac's 30–53× lives. FFI `dispatch_ffi_primitive` guest-fatals cleanly
  on non-ARM64 until then.
- **2026-07-21 — M2 done (Phase 2, the JIT substrate).** Three commits:
  - **Encoder.** Vendored JASM's native x86-64 `rasm` encoder into
    `src/vendor/wfasm/rasm/` (byte-identical to upstream modulo header +
    crate paths); 20 encoder tests pass.
  - **Loader + relocations.** `WinJit` (the `native_windows.rs` twin of
    `MacJit`) now assembles via `rasm` and relocates via new
    `relocpatch::patch_relocs_x64` (rel32 in place; far branches through
    `movabs rax ; jmp rax` stubs; `Abs64`). It **executes native x64** —
    proven by leaf, internal-call, and host-extern-callback tests.
  - **Trap layer.** An x86-64 Vectored Exception Handler
    (`deopt_trap::veh_trap_handler`) mirrors the macOS SIGTRAP path:
    trap site = `int3` + imm16, trap-pc stashed in **R10** (the x16
    analogue), classify→redirect→resume. `veh_redirect_smoke` round-trips
    `int3 → VEH → trampoline → resume` — **the M2 acceptance gate.**
  - **Windows is simpler here, as predicted:** no `MAP_JIT`, no per-thread
    W^X toggle, no icache invalidation; foreign `int3` passes through the
    VEH for free (no `SIG_DFL` restore dance).
  - `cargo test --lib` is **green on Windows: 655 passed, 0 failed.** The
    world interpreter is unchanged (5891/0). A `.gitattributes` pins LF so
    scripted edits stay byte-clean.
  - **Test gating (all macOS/aarch64-only, re-enabled per phase):** tier-1
    compile+execute tests (`target_arch = "aarch64"`); signal-based fault
    recovery + real-FFI (`mmap`/`getpid`) tests and the embedded-VmHandle
    integration suite (`embed::tests`) (`target_os = "macos"`); the 16 KiB
    Apple-page commit assertion.
  - **Known Phase-2 follow-up:** Windows guest-fatal recovery. My
    `sigsetjmp`/`siglongjmp` are stubs, so an embedded `VmHandle` cannot yet
    catch a guest `error:`/DNU/FFI-fault as a message (it would `process::
    exit`). The macOS path uses `siglongjmp` specifically to cross JIT
    frames; the interpreter-only Windows build has none yet, so a Rust
    `catch_unwind`-based recovery is viable and is the intended fix before
    tier-1 lands.
- **2026-07-21 — Phase 3 begun; the x64 back end executes.** Three commits:
  - **`assembler_x64.rs`** — the `JasmAssembler` sibling, producing the
    same `CodeBlob` so `codecache`/`nmethod`/GC are untouched. Branches
    always take the rel32 form (width known before displacement, so no
    relaxation pass); symbol operands *are* used, because the x64 encoder
    returns structured `Rel32`/`RipRel32` fixups — the exact hook a
    structured emitter wants (the A64 side's P6 rule doesn't apply);
    literals are RIP-relative, so every fixup records `insn_end`.
    13 tests, verified by decoding output with **iced-x86** rather than
    asserting hand-copied bytes.
  - **`regalloc.rs` register file** — the scan, spill policy, and oop-map
    bookkeeping stay shared verbatim; only the file is target-conditional.
    Pools became explicit lists of architectural numbers (x64's pool has
    holes — RSP/RBP sit mid-numbering). x64 GPR pool is **7** registers
    (RCX RDX RBX RSI RDI R8 R9) against AArch64's 16. Residency has no
    disjoint callee-saved base on x64 (every callee-saved register is
    pinned or already allocatable), so it draws only on scan leftovers
    among RBX/RSI/RDI — sound because residency only claims
    `!crosses_call` intervals. A new test pins that no reserved register
    (RSP/RBP/R10–R15/RAX) can ever be allocated.
  - **`emit_x64.rs`** — the vertical slice: ConstSmi, Move, Param,
    LoadField, SmiArith, SmiCmpBr, Jump, Ret, Bailout. **Its tests
    execute**, not inspect: all six arithmetic ops, guard bailout, real
    `SMI_MAX` overflow through the OF flag, compare-and-branch across
    zero, and the two-address `dst == b` aliasing hazard.
  - **Three findings worth carrying forward.** (1) `SMI_SHIFT == 2` with
    `INT_TAG == 0` makes tagged add/sub/bitwise need *no* untag and makes
    `jo` an exact overflow test — x64 gets this cheaper than AArch64's
    `smulh` dance. (2) `SmiOp` has **no division**, so the `idiv`
    RAX:RDX precoloring the plan feared is simply not needed. (3) The
    existing spill-all-at-safepoints policy means nothing is live in
    registers across a call, so emit may clobber freely at call
    boundaries — no ABI precoloring needed either.
  - 676 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3 continued: guards, traps, and the call stub.**
  - **`GuardKlass`/`LoadKlass`/`ConstPool`/`UncommonTrap`** in
    `emit_x64`, plus literal-pool interning so `PoolLit(i)` indexes
    `literal_ids[i]` 1:1 — the contract `codecache::read_pool_oop` relies
    on when a deopt reads a pool word back by index. `emit_x64` now
    returns `Emitted { blob, trap_sites }`; a trap site is keyed by its
    **own** offset, because the trapping pc *is* the `int3`.
    The klass guard rejects smis *before* loading the header — a smi has
    no header, so the reverse order would dereference a small integer as
    an address.
  - **The Phase-2/Phase-3 seam is closed.**
    `emitted_uncommon_trap_round_trips_through_the_veh` compiles a method
    containing a trap, registers its range with a capture trampoline, arms
    the **real** VEH, and calls it: the handler decodes the emitted site,
    stashes the trap pc in R10, and redirects. The emitter and the handler
    were written days apart against a written contract; this test proves
    they actually meet.
  - **`stubs_x64.rs` — the x64 call stub**, `call_stub(entry, vm, argv,
    argc)`. Saves `RBX RSI RDI R12–R15`; subtracts **40** (32 shadow + 8
    realignment, since `RSP % 16 == 8` after the return address, `push
    rbp`, and seven pushes); moves `entry`/`argv` to scratch before the
    argument registers — which are its own parameter registers — are
    overwritten. Verified by a machine-code harness that plants a sentinel
    in every callee-saved register and XORs them after the call, by an
    R15-dereferencing callee, and by a coupling assertion that fires if
    Phase 5 ever adds a callee-saved XMM to the FP pool without teaching
    this stub to save it.
  - 684 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3 continued: the call-free op set is complete.**
  `StoreField` with the generational card-marking write barrier (three
  early-outs, cheapest first: young `obj`, smi `val`, old `val`),
  `SmiCmpVal` via branchless `cmovcc` (the `csel` analogue), `BoolBr`
  against the canonical true/false oops with a `not_bool` edge for
  everything else, and `RetSelf`. `assembler_x64` gained `Cond::cmov()`
  and `mem_byte()`.
  - **A real bug, caught by a test rather than by review:** `LoadField`
    was not applying the `MEM_TAG` displacement bias that `StoreField`
    and `LoadKlass` apply, so every compiled field read was one byte off.
    The store-then-read-back test returned 1 instead of 308 — a value
    shifted by a single byte. Worth recording as evidence for the
    execution-testing discipline: the op had been "supported" and
    compiling cleanly since Phase 3c.
  - The write-barrier test asserts the three *negative* cases as well as
    the positive one. A barrier that marked unconditionally would still
    be functionally correct and would quietly destroy scavenge
    performance, so the non-marking cases are the ones worth pinning.
  - 688 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3g: `Poll` and `CallRuntime`.** Plus the shared
  `emit_runtime_call` helper and a `RuntimeAddrs` struct carrying the
  entry points (pooled as `RuntimeAddr` words, so the GC leaves them
  alone). `Emitted` now also reports `safepoints` — runtime-call *return*
  addresses, contrasting with `trap_sites`, which key on the trapping
  instruction itself.
  - **Why clobbering volatiles is safe** (documented at the helper):
    a Rust callee destroys `RAX RCX RDX R8–R11`, which includes four
    *allocatable* registers. That is sound only because every op reaching
    this helper is a safepoint, and the spill-all policy has already
    spilled anything live across one. This is the third time the
    spill-all invariant has paid for itself in this port — it also
    removed the need for `idiv` and ABI precoloring.
  - **Two different shadow-space numbers, both correct.** Compiled code
    reserves 32; the call stub reserves 40. They sit at different
    alignment phases (the stub has pushed an odd number of registers).
    Noted explicitly so it doesn't read as an inconsistency later.
  - `Poll` is tested in **both** directions against a call-counting probe
    stub: a poll that never fired would hang the collector at a
    safepoint, and one that always fired would call into the runtime on
    every loop iteration, so neither direction alone is sufficient.
  - 691 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3h: inline allocation.** `Alloc` bumps the live
  eden top, bounds-checks against `eden_end`, stamps `[mark][klass]`,
  nils the body, and `MEM_TAG`s the result; overflow calls
  `rt_alloc_slow(klass, size_bytes)`, whose return address is a safepoint
  (a real allocation may scavenge).
  - **The double indirection is load-bearing.** The VM register block
    holds the *address of* `eden.top`, not a copy — a value copy would go
    stale the moment a nested allocation or a GC beneath this frame moved
    the real pointer. `eden_end`, by contrast, is a genesis-fixed bound
    and *is* safe to read as a value. Both facts are recorded at the site.
  - The test asserts on the **heap**, not just the return value, because
    neither path is checkable from the result alone: header and nil'd
    body read back off the heap, the published eden top, a second bump,
    then a third allocation past `eden_end` that calls the slow path
    exactly once, receives the size in bytes, and provably does not bump
    eden itself.
  - 692 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3i: `CallSend`, inline caches, and near-host code
  placement.**
  - **`CallSend`** does parallel-move argument marshalling (a source
    register may be another argument's destination, so safe moves go
    first and a genuine cycle is broken through a scratch; spilled
    sources are never in a cycle), emits the patchable 5-byte
    `call rel32` site, records the safepoint and IC metadata, and checks
    the NLR sentinel so a non-local return propagates one native frame at
    a time instead of being read as a result.
  - **Near-host placement — §2.2 finally honoured.** The first `WinJit`
    copied the macOS loader's "allocate anywhere" shape, which silently
    cost every host-runtime call an absolute veneer (`mov r10,[rip+pool];
    call r10` — a load plus an indirect branch). It now asks
    `VirtualAlloc2` for the region within ±1.75 GB of this image, so
    those become direct `call rel32`. Falls back to anywhere when
    `VirtualAlloc2` is missing or the window is crowded; that path stays
    correct through `relocpatch`'s stubs, just slower. The test measures
    the distance from **both ends** of the region, since worst case is
    what decides whether a veneer is needed.
  - **A flaky test of my own making, fixed.** The write-barrier test
    derived `old_start` from two separate stack arrays, so whether they
    straddled a 512-byte card boundary decided whether a card index came
    out negative and wrapped when cast to `usize`. It passed in isolation
    and failed in the full parallel run. Rewritten around one arena with
    fixed offsets; the suite is now stable across 5 consecutive runs.
    Worth recording alongside the `LoadField` bias bug: both were caught
    by running things, neither by reading them.
  - 696 lib tests pass; world interpreter still 5891/0.
- **2026-07-21 — Phase 3j: array ops; `oopmap.rs` needed nothing.**
  - `ArrayAt`/`ArrayAtPut` with their four guards. x86 addressing makes
    these markedly tighter than the A64 originals: the element address is
    **one operand** (`[arr + idx*2 + base]` — a tagged index `i<<2`
    scaled by 2 is exactly `i*8`, one stride, where AArch64 needs two
    `add`s), and the klass check is register-free via
    `cmp reg, [rip+lit]` where AArch64 must load the literal into a
    scratch. Together those let the whole guard sequence run on `RAX`
    alone, leaving both scratches holding the array and index.
  - The bounds check is a single **unsigned** compare of tagged values:
    `idx − 4` is `(i−1) << 2`, so comparing unsigned against the tagged
    length rejects `i < 1` and `i > length` in one instruction, because a
    zero or negative index wraps to a huge unsigned value. Tests cover
    index 0 and −1 explicitly — a signed compare would pass both and read
    outside the object.
  - **`oopmap.rs` required no changes at all.** It never inspects
    registers, only spill slots — because spill-all-at-safepoints means
    nothing is live in a register at a safepoint. That is the fourth time
    this one invariant has removed work from the port (after `idiv`
    precoloring, ABI precoloring, and safe volatile clobbering at runtime
    calls).
  - 698 lib tests pass; world interpreter still 5891/0.
- **The IR surface is now complete except floats/SIMD/OSR** (`FUnbox`,
  `FBox`, `FArith`, `FCmpBr`, `FCmpVal`, `FConst`, `VecArith`,
  `NlrReturn`), which are Phase 5 work.
- **2026-07-21 — Phase 3k: entry guard, block PCs, verified entry.**
  `emit_entry_guard_x64` emits the per-klass customization guard
  (receiver in the first Win64 argument register; a heap key needs no
  smi-klass literal at all, since a smi can never match it; a miss
  **tail-jumps** to the resolve stub, sound because the guard runs before
  any prologue so the stub returns straight to the original caller).
  `emit_x64` now also reports `block_pcs` and `verified_entry_off`. The
  test enters both ways — through the guard and directly at
  `verified_entry_off` — and requires the same answer, which is what
  makes that offset a genuine entry point rather than a number.
  699 lib tests pass; world interpreter still 5891/0.

## 8. The remaining gap to a firing JIT — measured, not estimated

The emitter side of Phase 3 is essentially complete: every IR op except
floats/SIMD/OSR lowers, and each is proven by an execution test. **But no
running Smalltalk program reaches any of it**, and the reason is not the
emitter — it is that tier-1 needs a *runtime environment* of hand-written
stubs, all of which still exist only as AArch64.

Surveyed rather than guessed:

| Component | A64 generators to port | Notes |
|---|---|---|
| `codecache/stubs.rs` | **13** `build_*` functions | **4 done** (`call_stub`, `stub_poll`, `must_be_boolean`, `alloc_slow` — `stubs_x64.rs`); still needed: `stub_resolve`, `not_entrant`, `deopt_return_trampoline`, `mega_shared`, `dnu`, `box_double`, `call_primitive`, `nlr_originate`, `value_dispatch` |
| `codecache/deopt_trap.rs` | **3** trampolines | `uncommon`, `assert`, `probe` — the VEH already redirects to them; they just need x64 bodies |
| `codecache/pics.rs`, `mega.rs`, `adapters.rs` | PIC/megamorphic/adapter emitters | patch-site shapes already fixed by `call_patchable` |
| `compiler/driver.rs` | back-end selection | the `emit::emit` call site takes 15 parameters and returns a 6-tuple; `emit_x64` returns an `Emitted` struct. Needs a seam, plus `prim_shim` and OSR support, and `SafepointPc`-vs-`TrapSite` reconciliation for `build_deopt_metadata` |
| `compiler/disasm_a64.rs` | trace/debug disassembly | replace with `iced-x86` (already a dev-dependency) |

So the honest position: **the hard, novel work is done and tested; what
remains is a substantial amount of mechanical-but-careful stub porting**,
none of it conceptually new, but all of it load-bearing — a wrong stub is
a silent crash inside compiled code.

**"Mechanical" does not mean safe to translate literally.** The first
three stubs turned up a genuine ABI divergence: `rt_poll` returns a
16-byte `PollOutcome` struct, which AAPCS64 hands back in `x0:x1` but
**Win64 returns through a hidden pointer** — an implicit first argument
that shifts every real argument one register right. An
instruction-for-instruction port would have read `RAX`/`RDX` as the two
fields and gotten a pointer plus garbage, with nothing failing at the
point of the error. That is now pinned by a test that calls a real Rust
`extern "C"` returning such a struct and requires both fields intact,
rather than by my reading of the spec. **Expect at least one more of
these** among the remaining nine stubs — struct returns, varargs, and
by-value aggregates are exactly where the two ABIs disagree.

Only after that do the `target_arch = "aarch64"`-gated tier-1 tests come
back — and those are the real differential check against the Mac, worth
far more than the hand-built IR tests written so far.

## 7. What deliberately does *not* change

- Bytecode ISA, `.mst` format, world sources, SQLite image store — **identical**,
  so worlds and tests are shared verbatim with MACVM.
- The no-`become:`, rebuild-from-source philosophy — unchanged; sub-second world
  rebuild is the compatibility guarantee between the two VMs.
- IR, type feedback, inlining, escape analysis, deopt *policy* — arch-neutral by
  construction; only their machine-level *rendering* is new.
