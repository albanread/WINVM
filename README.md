# WINVM — a Windows x86-64 Smalltalk VM, inspired by Strongtalk

## Motivation

A from-scratch Windows x86-64 compiler for Smalltalk — the most complex
compiler project in my repos, and like the others, it may take a while
before it turns into a useful system.

This isn't a history lesson, just my own experience of one. Strongtalk was
released to the public in 2002 — first as documentation I thoroughly
enjoyed reading, then as full C++ source. At the time it executed Smalltalk
at high speed, and the released repo was fascinating, ambitious, and richly
engineered. I spent many happy hours exploring it and came away impressed
by the design: Strongtalk — and Self before it — pioneered adaptive
optimization (polymorphic inline caches, type feedback, deoptimization),
the ideas that went on to power the Java HotSpot VM, and added on top of
that an optional static type system and a live, hypertext programming
environment. There's a great deal of brilliant engineering there to learn
from and build on.

Decades later, software technology and AI have made life far simpler — it's
much easier to write compilers now, and I find re-implementing a strong,
well-documented design one of the most rewarding ways to work. So WINVM is
built to a large extent on Strongtalk's own design and documentation. I'm
cheating to the maximum extent possible: the bytecode interpreter and
compiler are written in Rust, my own **x86-64 assembler** is reused in the
compiler, and only the GC had to be entirely new. It also carries the almost
absurd level of introspection, debugging, and testing a project this complex
needs, in the hope it adds up to reliability.

WINVM is a research virtual machine for **Windows on x86-64**, in the
**Self → Strongtalk** lineage: a **class-based object model** with an
**adaptive optimizing compiler** driven by type feedback. It takes the
adaptive-optimization machinery those VMs share (inline caches, PICs, type
feedback, deoptimization) and Strongtalk's representation (classes + direct
pointers, no object table), reimplemented in Rust for 64-bit Windows. It is
the Windows sibling of [MACVM](https://github.com/albanread/MACVM), sharing
the entire portable front and middle end and re-vendoring the x86-64 JIT
substrate (the `E:\JASM` encoder and the `E:\WF66` shipping Windows JIT VM)
for the architecture-specific back half. The migration design is in
[`MIGRATION.md`](MIGRATION.md).

## Status — working, and it compiles

WINVM boots a real Smalltalk object world and runs programs on a **two-tier
engine**: a simple dispatch-based bytecode interpreter plus a **tier-1
optimizing JIT** that recompiles hot code with type feedback and deoptimizes
safely, all under a moving generational collector. The whole system is gated
against a single invariant — **compiled output must be byte-identical to
interpreted output** — enforced by a differential test suite of **5,860
in-language tests run four ways** (interpreter, JIT, JIT + GC-stress, JIT +
deopt-stress) that must all agree.

### Measured against Cog — a yardstick, not a competition

WINVM does not compete with Squeak, Pharo, or Cog in any way. Those are
mature production systems with decades of engineering and real communities
behind them; this is a from-scratch Windows VM exploring the Strongtalk
lineage. But a JIT still needs an honest yardstick — not its own interpreter
(ours is deliberately simple) — and Cog, the OpenSmalltalk JIT that powers
Squeak and Pharo, is the meaningful one: same language, same benchmarks, and
it sets a high bar.

So WINVM is measured against **Cog** (Pharo 13, the x86-64 OpenSmalltalk
JIT), on the same machine (i7-12700), both processes pinned to one
performance core and timed with a microsecond clock:

| benchmark | WINVM (JIT) | Cog | |
|-----------|-------------|-----|---|
| arith     | 36 ms | 48 ms | **faster** |
| sieve     | 3 ms  | 4 ms  | **faster** |
| alloc     | 14 ms | 18 ms | **faster** |
| dict      | 12 ms | 11 ms | parity |
| deltablue | 4 ms  | 4 ms  | parity |
| richards  | 33 ms | 29 ms | ~1.15× behind |
| fib       | ~207 ms | ~180 ms | ~1.2× behind |

(warm, ×10 inner reps, checksum-verified equal work on both VMs). WINVM
matches or beats Cog on five of seven and trails only on the two
deepest-recursion / send-heavy micro-benchmarks — a known, scoped codegen
gap ([`docs/x64_codegen_perf.md`](docs/x64_codegen_perf.md)), not a
correctness or GC problem. See [`docs/PERF.md`](docs/PERF.md) for the full
measured record, including how the comparison is kept fair (core pinning; a
microsecond clock, because Windows' millisecond timer quantizes to 15.6 ms
and made Cog's sub-tick numbers look artificially like zero).

### What's implemented

- **Object model** — Strongtalk-style classes, direct tagged pointers, **no
  object table**, a 2-word `[mark][klass]` header. Arch-neutral: identical on
  x86-64.
- **Garbage collection** — generational scavenge + a full compacting collector,
  both running **under live, moving compiled frames** via precise oop-maps and a
  mixed-tier frame walker (RBP-chain walking on x64).
- **Interpreter** — a simple dispatch-based bytecode baseline tier (a
  fetch-decode-`match` loop) with inline caches. It is also the **differential
  oracle** and the deoptimization target, so it is kept plain and obviously
  correct on purpose.
- **Tier-1 optimizing JIT** — a vendored pure-Rust **x86-64 encoder** behind
  the `Assembler` trait; PICs and type feedback; method + block inlining;
  per-klass **customization** with self-send and block-arg **devirtualization**;
  **deoptimization**, **on-stack replacement (OSR)**, and recompile-on-trap.
  Uncommon traps are `int3` sites recovered through a **Vectored Exception
  Handler** (the Windows counterpart of MACVM's Mach signal traps).
- **Windows codegen quality** — identity-move elision, spill-to-spill
  coalescing, constant rematerialisation, provably-dead write-barrier removal,
  a loop-weighted callee-saved **residency** tier, and inline lowering of the
  identity (`==`/`~~`) and boolean-`not` special selectors — all validated
  byte-identical under the four-way differential
  ([`docs/x64_codegen_perf.md`](docs/x64_codegen_perf.md)).
- **Closure compilation** — literal blocks compile and splice inline, including
  multi-basic-block conditional-`^` (non-local-return) blocks, with `Context`
  elision / materialization / adoption across the tier boundary. A recursive
  call never heap-allocates its activation (`contexts_allocated == 0` on the
  benchmarks).
- **Inline allocation** — `basicNew` and class-side constructors fuse to an
  inline eden bump (no Rust crossing per object); the nursery is sized so the
  allocation benchmark beats Cog ([`docs/gc_alloc_gap.md`](docs/gc_alloc_gap.md)).
- **Scalar float regions** — a mono-`Double` send site compiles to a guarded
  unbox, native SSE2 `movsd`/`addsd`/`mulsd`/`ucomisd`, and a box only where a
  boxed value is actually observed; inside a region there is no allocation, no
  GC interaction, and no message send
  ([`docs/float_fastpath_design.md`](docs/float_fastpath_design.md)).
- **Win32 FFI** — native calls resolved through `GetProcAddress` +
  shape-keyed native-call trampolines + an `Alien` raw-memory type
  ([`docs/FFI.md`](docs/FFI.md)). The `Platform` global (`#windows`) lets shared
  world source select the right OS surface at load time — e.g. `Time` reads
  the wall clock via `GetSystemTimePreciseAsFileTime` over `VirtualAlloc`
  scratch, where the Mac line used `clock_gettime`.
- **COM + the web GUI** — see below.
- **Multi-VM workers** — share-nothing parallelism driven from Smalltalk:
  `Worker spawn:` boots **worker VMs** (each its own heap, JIT, and GC on its
  own OS thread) that communicate with the primary by **deep-copy message
  passing** (the MOP pickle) — Erlang-style, no shared state, no identity across
  heaps, fully asynchronous (`send:onReply:` continuations; a send wakes the
  sleeping receiver, so nothing polls). A crashed worker dies alone and is
  reported as an ordinary `#workerDied` message
  ([`docs/multi-smalltalk-worker.md`](docs/multi-smalltalk-worker.md)).
- **Image store** — offline SQLite image editing + a DB→VM boot loader that
  reconstructs the world byte-identically to a `.mst` boot ([`docs/IMAGE.md`](docs/IMAGE.md)).
- **The object world** — 100+ classes / 1,200+ methods of hand-written and
  Strongtalk-ported library (`world/*.mst`): full collections + streams
  protocol, Dictionary/Set/OrderedCollection, String/Character text utilities,
  Fraction and LargeInteger arithmetic, an in-language test suite, and the
  Richards / DeltaBlue / Stanford benchmark ports in `world/bench/`.
- **Scripting** — an embedded RUSTTCL console for driving the VM and its
  debugger ([`docs/RUSTTCL.md`](docs/RUSTTCL.md)).
- **Debugger** — crash-dossier (PROBE) via an SEH/VEH dumper, breakpoints,
  mixed-tier backtrace, an x86-64 disassembler, IR dumps, and step-between-calls
  ([`docs/DEBUGGER.md`](docs/DEBUGGER.md)).

### COM and the web GUI

The programming environment is a **web GUI**: a faithful recreation of the
1996 **Strongtalk hypertext programming environment**, rendered as HTML and
driven live from the running VM. On Windows it is hosted in **WebView2** — the
Chromium/Edge runtime — inside a native **Win32** window, and the whole
integration goes through **COM**: `CoInitializeEx`, the `ICoreWebView2`
controller and view, and the `webview2-com` bindings
([`gui/src/shell/win.rs`](gui/src/shell/win.rs)). The page and the VM talk
over `window.chrome.webview.postMessage`; a virtual host name serves the GUI
tree, and every send from the page runs between doits on the VM thread.

COM is to WINVM what the Objective-C bridge was to the Mac line: the way the
language reaches the host platform. WebView2 is itself a COM component tree,
and the same `webview2-com` / `windows` crate bindings are the substrate a
broader **COM-from-Smalltalk** bridge builds on — the Windows analogue of
treating host objects as ordinary Smalltalk receivers.

The GUI ships the same core toolset as the reference environment — a live
**class browser** whose accepts compile into the running VM *and* persist to
the image, an outliner, **find tools** (definitions, implementors, senders —
SQLite-indexed), a **Workspace** with do-it/print-it, a **Canvas** drawing
widget, and a live **VM/GC metrics dashboard**
([`docs/vm_handle.md`](docs/vm_handle.md), [`gui/PLAN.md`](gui/PLAN.md)).

### Replace, don't mutate — there is no persistent image

WINVM never mutates a persistent image. Where classic Smalltalk carries one
long-lived heap snapshot forward across years of in-place modification, WINVM
keeps its truth in a **source-code database** — the `.mst` world files and the
SQLite image they seed — and spins up VMs from it in well under a second. VMs
are plural and disposable: the system would always rather **throw a VM away and
rebuild it from source than mutate one in place**.

You can feel the difference in the tools. When you file development code into
the GUI, it is not patched into the long-running VM: a **fresh VM is recreated
from the world and your file loads on top of it**. Filing in the same file
twenty times just works — there is no accumulated state to collide with,
because there is no accumulated state at all. To a Smalltalker raised on the
image this reads as less dynamic, almost static. In operation, though, WINVM is
a true Smalltalk system — live objects, live compilation, everything inspectable
while it runs. The difference is only in how change lands: **replacement instead
of mutation**, with every piece of state visible in source you can read, diff,
and version.

### Why there's no `become:`

WINVM has no `become:` — the Smalltalk primitive that swaps one object's
identity for another's, redirecting every reference in the system at once. This
is a deliberate omission. WINVM — like Strongtalk and Self before it —
represents an object reference as the **raw machine address of the object
body**, not as an index into an object table. That is the choice that makes a
field access a single load and lets the JIT cache classes at send sites, build
PICs, and inline — the whole basis of the adaptive optimizer. But it also means
"redirect every reference to A so it points to B" has no cheap implementation:
there is no table slot to swap, only every pointer in both heap generations,
every root, every live stack frame, and every machine register to find and
rewrite. `become:` fights everything the compiler is built to do.

We can afford to skip it because WINVM is **not image-based**: there is no
persistent snapshot of the live object heap (the SQLite image WINVM boots from
is a database of class/method *source*, not a heap dump). WINVM rebuilds its
entire world from source on every boot, in well under a second, so the dominant
use of `become:` — evolving a class whose instances you can't afford to lose —
is answered by editing the source and restarting, not by mutating a live heap.
Class redefinition itself already goes through the deoptimize-and-recompile path
the VM has for exactly this. The full reasoning, including what is genuinely
lost, is in [`docs/DESIGN.md`](docs/DESIGN.md).

### Not yet on Windows

WINVM shares the portable front/middle end with the Mac line but is a younger
port; a few Mac-specific capabilities have **not** been carried over and are not
claimed here: the NEON SIMD value-classes / bulk kernels (an x86-64 SSE/AVX
rewrite is future work), the native Metal game pane, and the native AppKit
("Cocoa") GUI. The web GUI above is the Windows front-end.

## Layout

| Path | Contents |
|------|----------|
| `src/oops/` | Object model — tagged pointers, 2-word headers, classes |
| `src/memory/` | Object memory, allocation (`VirtualAlloc`), generational + full GC |
| `src/interpreter/` | Dispatch-based interpreter (the baseline tier + differential oracle) |
| `src/bytecode/` | Bytecode format, decoder, CFG |
| `src/compiler/` | Tier-1 optimizing compiler + the x86-64 backend (`emit_x64.rs`, `regalloc.rs`, `oopmap.rs`) |
| `src/codecache/` | Native code cache, stubs, VEH deopt-trap machinery (`stubs_x64.rs`, `deopt_trap.rs`) |
| `src/runtime/` | Dispatch, frames (RBP chains), deopt materializer, OSR, recompile, debugger |
| `src/frontend/` | `.mst` parser + class-definition loader |
| `src/vendor/wfasm/` | Vendored pure-Rust x86-64 encoder + Win32 native loader (from `E:\JASM`) |
| `src/embed.rs` | `VmHandle` embedding API + multi-VM workers |
| `src/rusttcl/` | Embedded RUSTTCL console |
| `world/` | The object world / image sources, tests, benchmarks |
| `gui/` | The Strongtalk-style web GUI — rendered in **WebView2** via COM (`gui/src/shell/win.rs`) |
| `image_store/` | The versioned SQLite class/method source database (importer, exporter, send-index) |
| `docs/` | Design notes, specs, per-sprint guidance, the migration design (`MIGRATION.md`) |

## Building & running

```sh
cargo build --release
target/release/macvm run world/bench/deltablue.mst --world world   # runs it
set MACVM_JIT=off & target/release/macvm run <prog>.mst --world world   # interpreter only
set MACVM_JIT=threshold=20 & ...                                        # JIT gate
set MACVM_TRACE=stats & ...                                             # jit|deopt|count instrumentation
set MACVM_BENCH_CPU=perf & ...                                         # pin to a P-core for benchmarking
```

The JIT is on by default. `MACVM_JIT=off` selects the interpreter, which is the
differential oracle every JIT change is gated against (compiled output must be
byte-identical to interpreted output). Tests: `cargo test`; the stress matrix
(GC / deopt) and world differentials are in `tests/`.

The web GUI launches with a release build (`gui/`, hosted in WebView2 — the
Edge/Chromium runtime must be present, as it is on current Windows). It boots
its whole interface from a SQLite **image** (`world/image.sqlite3`) rebuilt from
the `world/*.mst` source; after changing a world class, rebuild the image with
the reseed workflow ([`docs/managingtheworld.md`](docs/managingtheworld.md)).

## Lineage & licensing

Self and Strongtalk were released under BSD-style licenses. Code adapted from
them retains its original notices; new WINVM code is under the license in
[`LICENSE`](LICENSE). See `docs/DESIGN.md` for provenance tracking. WINVM is the
Windows x86-64 sibling of [MACVM](https://github.com/albanread/MACVM); the two
share the portable front and middle end and diverge in the architecture-specific
back half (x86-64 vs AArch64) and the host-integration layer (COM/WebView2 vs
Cocoa/WKWebView).

## Further reading

WINVM's technical origin is **Strongtalk** — the original system lives on at
[strongtalk.org](https://strongtalk.org/) and
[talksmall/Strongtalk](https://github.com/talksmall/Strongtalk).

**Cog** is the other great branch of the Self family tree — the production
JIT that, from a deliberately simpler baseline design, still keeps pace with
Strongtalk's — and it is by far the best-documented. Eliot Miranda's
[Cog Blog](http://www.mirandabanda.org/cogblog/) is the clearest published
explanation anywhere of the machinery a Smalltalk VM actually needs
(remarkably, Cog is itself written in Smalltalk, translated to C for the
build). If the internals here interest you, read him:

- [About Cog](http://www.mirandabanda.org/cogblog/about-cog/) — what Cog is
  and how its pieces fit
- [Closures Part I](http://www.mirandabanda.org/cogblog/2008/06/07/closures-part-i/),
  [Part II — the Bytecodes](http://www.mirandabanda.org/cogblog/2008/07/22/closures-part-ii-the-bytecodes/),
  [Part III — the Compiler](http://www.mirandabanda.org/cogblog/2008/07/24/closures-part-iii-the-compiler/)
- [Under Cover Contexts and the Big Frame-Up](http://www.mirandabanda.org/cogblog/2009/01/14/under-cover-contexts-and-the-big-frame-up/)
  — mapping contexts to stack frames, the heart of making Smalltalk fast
- [Build me a JIT as fast as you can](http://www.mirandabanda.org/cogblog/2011/03/01/build-me-a-jit-as-fast-as-you-can/)
- [A Spur gear for Cog](http://www.mirandabanda.org/cogblog/2013/09/05/a-spur-gear-for-cog/)
  — the Spur object representation

Cog itself lives at
[OpenSmalltalk/opensmalltalk-vm](https://github.com/OpenSmalltalk/opensmalltalk-vm).
