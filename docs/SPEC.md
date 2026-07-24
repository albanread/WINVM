# MACVM Engineering Specification

**The full design for a Strongtalk-on-Apple-Silicon reimplementation in Rust.**

This document pins down everything [`DESIGN.md`](DESIGN.md) left open and adds
the missing subsystems (language, bytecode, bootstrap, primitives, testing). It
is the contract the sprint plan ([`SPRINTS.md`](SPRINTS.md)) builds against.
Evidence base: [`reference-vm-analysis.md`](reference-vm-analysis.md) (Self,
Strongtalk, JASM — file-anchored); machine-level detail: [`arm64.md`](arm64.md).

Where this spec and the reference VMs differ, the difference is deliberate and
noted with **Δ**. Where a number is a starting tunable, it is marked *(tunable)*.

---

## 0. System overview

```
                 .mst source files
                        │
                 ┌──────▼──────┐
                 │ Source      │  Rust: lexer/parser/codegen
                 │ compiler    │  (§1, §4)
                 └──────┬──────┘
                        │ CompiledMethod (bytecode + literals + IC table)
                 ┌──────▼──────────────────────────────────┐
                 │ Interpreter (tier 0)          §5        │
                 │  threaded loop, send-site ICs,          │
                 │  invocation + loop counters             │
                 └──────┬──────────────────────────▲───────┘
              counter   │                          │ deoptimization
              overflow  │       type feedback      │ (uncommon trap,
                 ┌──────▼──────────────────────────┴───────┐   invalidation)
                 │ Optimizing compiler (tier 1)  §8        │
                 │  IC-driven inlining, customization,     │
                 │  SSA-lite IR, linear-scan, JASM emit    │
                 └──────┬──────────────────────────────────┘
                        │ nmethod (code + relocs + oop maps + scope descs)
                 ┌──────▼──────┐        ┌──────────────────┐
                 │ Code cache  │        │ Heap: eden+surv, │
                 │ (MacJit)    │        │ old gen, cards   │
                 └─────────────┘        └──────────────────┘
```

Two tiers, exactly like Strongtalk: **interpreter + one optimizing compiler**
(with recompilation levels), no baseline JIT. The interpreter's send-site
inline caches are the type-feedback source.

Concurrency model: **one Smalltalk execution thread** (green processes are a
stretch goal, §11). GC runs only at well-defined points in that thread
(allocation sites, loop polls), which makes every collection trivially
safepointed until compiled code arrives (§8.8).

---

## 1. The language — MACVM Smalltalk

A classic Smalltalk-80 dialect, Strongtalk-flavored. The optional type system
is **out of scope for the runtime** (Strongtalk's rule: annotations must never
change dynamic behavior); the parser *accepts and discards* Strongtalk-style
type annotations so annotated sources still load.

### 1.1 Expressions (standard Smalltalk-80)
- **Literals**: integers (`42`, `-7`, `16rFF`), floats (`3.14`, `1e10` → boxed
  `Double`), characters (`$a`), strings (`'hi'`), symbols (`#foo`,
  `#at:put:`), literal arrays (`#(1 2 #three 'four')`), byte arrays
  (`#[1 2 255]`), `nil true false`.
- **Brace (dynamic) arrays** `{ e1. e2. … }` (Squeak/Pharo): a `.`-separated
  list of arbitrary EXPRESSIONS, evaluated left to right into a **fresh
  `Array`** at runtime — contrast `#( … )`, a compile-time constant of
  literals only. Optional trailing `.`; `{}` is the empty array. Desugared by
  the compiler to `(Array new: n)` + one `at:put:` per element (referencing
  the `Array` class directly, so a shadowing local can't hijack it) — no
  dedicated bytecode, so it runs identically interpreted and JIT-compiled.
- **Sends**: unary > binary > keyword precedence; parentheses; cascades (`;`).
- **Assignment** `:=`, **return** `^`, statement separator `.`.
- **Temporaries** `| a b |`; **blocks** `[:x :y | ...]` with block-local temps,
  full closures, non-local return (`^` inside a block returns from the home
  method).
- **Pragmas**: `<primitive: N>` as the first element of a method body binds the
  method to VM primitive N (§10). Other pragmas (incl. type annotations
  `<:T>` etc.) are parsed and ignored.

### 1.2 Class definitions — `.mst` world files
No image-based browser initially; the world is built from source files with a
brace syntax (GST-style, chosen for clean parsing):

```smalltalk
Object subclass: Point [
  | x y |                              "instance variables"
  Point class >> x: ax y: ay [ ^self new setX: ax y: ay ]
  x [ ^x ]
  setX: ax y: ay [ x := ax. y := ay ]
  + p [ ^Point x: x + p x y: y + p y ]
  printOn: s [ s << x printString << '@' << y printString ]
]
```

- `Superclass subclass: Name [ ... ]` defines/extends a class;
  `Name class >> sel [...]` defines a class-side method (metaclass).
- Indexable classes: `Object subclass: Array [ <indexable: oops> ... ]`,
  `<indexable: bytes>` — a class-level pragma sets the format (§2.4).
- Files load in order; a class may be reopened to add methods.
- **Δ from Strongtalk**: no `.dlt` chunk format, no mixin declarations in v1
  (mixin support is a stretch goal; the klass layout reserves a slot, §2.4).

### 1.3 Core semantics commitments
- `SmallInteger` arithmetic overflows into `LargePositiveInteger`/
  `LargeNegativeInteger` via primitive failure → Smalltalk fallback code (§10).
- `Double` is heap-boxed (decision pinned; NaN-boxing shelved — `arm64.md` §1.3).
- `Character`: flyweight — 256 preallocated instances for Latin-1; characters
  above U+00FF allocated on demand. Strings are byte arrays holding UTF-8;
  `at:` is byte-indexed in v1 (documented limitation).
- `doesNotUnderstand:` receives a `Message` (selector + argument array).
- `ensure:`/`ifCurtailed:` supported once NLR unwinding exists (Sprint 4).
- No `thisContext` reflection in v1 (contexts are VM-internal, §5.4).
- No exceptions in v1 beyond DNU + `error:`; ANSI exception classes are a
  library-level stretch goal built on `ensure:` + NLR.

---

## 2. Object model — exact representation

64-bit words throughout. All bit layouts below are **pinned** (they close
`DESIGN.md`'s open questions).

### 2.1 Tagged oops — Strongtalk's 2-bit scheme, widened
```
bits[1:0]  kind
   00      smi           value = (word as i64) >> 2   (62-bit signed)
   01      heap oop      address = word - 1           (objects 8-byte aligned)
   10      reserved      (future immediate: Character; unused in v1)
   11      mark word     (only ever the first header word of a heap object)
```
- `Int_Tag=0` ⇒ tagged smis add/subtract/compare directly; multiply untags one
  operand; shift-right by 2 recovers the value.
- `Mem_Tag=1` ⇒ untag by `sub x, x, #1`; field offsets are pre-biased by −1 so
  loads are single instructions (`ldr xd, [xoop, #off-1]`), exactly Strongtalk's
  `_byte_offset()` trick.
- `nil`, `true`, `false` are ordinary heap objects (well-known oops, §3.1), as
  in Strongtalk. **Δ from Self**: no immediate floats.

```rust
#[repr(transparent)] #[derive(Copy, Clone, PartialEq, Eq)]
pub struct Oop(u64);           // the only type that crosses the unsafe boundary
```

### 2.2 Heap object header — 2 words
```
offset 0:  mark  : u64      (tag 11)
offset 8:  klass : Oop      (tag 01, direct pointer to the class object)
offset 16: body...
```

**Mark word (64-bit, explicit fields — Δ: no 32-bit age/size punning):**
```
bit  1:0   tag            = 11
bit  2     sentinel       = 1  (distinguishes a real mark from a forwarding oop)
bit  3     near_death        (weak-ref support, later)
bit  4     tagged_contents   (body is all oops — GC may skip format dispatch)
bit 11:5   age (7 bits)      (scavenge survival count)
bit 43:12  hash (32 bits)    (identity hash; 0 = not yet assigned)
bit 44     gc_mark           (full-GC mark bit; always clear outside a collection)
bit 63:45  reserved          (future: lock/pin bits)
```
- **Forwarding** (scavenge and compaction): the mark word is overwritten with
  the forwardee oop; `is_forwarded ⇔ (mark & 3) == 01`. Same trick as both
  reference VMs. During full-GC compaction, original marks that carry state
  (hash ≠ 0 or flags set) are saved in a temporary side map and restored after
  the slide; pristine marks are just re-initialized. **Δ**: side map instead of
  Strongtalk's age-field size stash.
- Identity hash is assigned lazily from a per-VM counter on first `identityHash`.

### 2.3 Object formats
The klass's `format` field (a smi enum) tells the VM how to size and scan an
instance:

| Format | Body layout after header | Examples |
|---|---|---|
| `Slots` | `non_indexable_size − 2` named oop fields | Point, Association |
| `IndexableOops` | named fields, then `[size: smi][elem0..]` | Array, MethodDictionary |
| `IndexableBytes` | named fields, then `[size: smi][bytes…pad]` | String, Symbol, ByteArray, LargeInteger digits |
| `Method` | CompiledMethod layout (§4.4) | CompiledMethod, CompiledBlock |
| `Klass` | class layout (§2.4) | every class & metaclass |
| `Double` | 8 raw bytes (f64) | Double |
| `Closure` | method, home, `[ncopied: smi][copied…]` | BlockClosure |
| `Context` | home hint + `[size][slot…]` heap temps | (VM-internal, §5.4) |
| `Process` | VM stack handle + state | Process (later) |

Instance size of a `Slots` object = `klass.non_indexable_size` (in words,
including header). Indexable objects add `size` from their own body. All sizes
rounded to 8-byte multiples.

### 2.4 Class objects (klass)
A class is a heap object of format `Klass` (its own `klass` field points to its
metaclass; the metaclass's klass points to `Metaclass` — standard Smalltalk).
Body (all oops unless noted):

```
format              smi   (§2.3 enum + flags: has_untagged_contents)
non_indexable_size  smi   (words incl. header, for Slots part)
superclass          klassOop | nil
methods             MethodDictionary (selector → CompiledMethod)
name                Symbol
inst_var_names      Array of Symbol      (compiler/reflection)
class_vars          Array of Association (shared pools folded in here, v1)
mixin_reserved      nil                  (future mixin link — Δ deferred)
```
**Δ from Strongtalk**: no embedded C++ vtable word — Rust dispatches on
`format` with a `match`. No vtable fix-up on snapshot load, ever.

MethodDictionary v1: an `IndexableOops` open-addressed table of
(Symbol, CompiledMethod) pairs, power-of-two capacity, linear probing on the
symbol's identity hash. Growable by rehash-into-new.

### 2.5 The `objects` module surface (Rust)
All raw access lives behind typed wrappers: `SmallInt`, `MemOop`, `KlassOop`,
`ArrayOop`, `ByteArrayOop`, `MethodOop`, `ClosureOop`, `DoubleOop` — each a
`#[repr(transparent)]` newtype over `Oop` with checked constructors
(`try_from(oop)`) and `unsafe` raw accessors used only inside the module.
Everything above this layer (interpreter, compiler, runtime) handles oops
through these wrappers plus `Handle`s (§7.6).

---

## 3. Universe & bootstrap

### 3.1 Universe
A single `Universe` struct owns the heap, the well-known oops, and the tables:

- **Well-known oops**: `nil_obj true_obj false_obj`, the core klasses
  (`smi_klass double_klass string_klass symbol_klass array_klass …` — a
  macro-generated list like Strongtalk's `APPLY_TO_VM_OOPS`), and
  `smalltalk` (the global namespace: a Dictionary of Association).
- **Symbol table**: open-addressed hash set of Symbols (interning). Strong in
  v1; weak later.
- **Lookup cache** (§6.1) and **code table** (§8.2) — Rust-side, rebuilt or
  flushed on GC as needed.

### 3.2 Bootstrap sequence (no image file in v1)
1. `Universe::genesis()` hand-builds the metaobject knot in Rust: allocate
   `Metaclass`, `Class`, `Object`'s klass chain, nil/true/false, the format
   klasses — the classic circular bootstrap, done once in ~200 lines.
2. Load `world/*.mst` in a declared order (`world/world.list`): the source
   compiler (§1, §4.5) compiles each class and installs methods. **Amendment
   A22**: an `image_store` SQLite container is an equivalent, optional
   source for this step — ordered (class, source) pairs either way, still
   compiled, never a heap/pointer load (`docs/IMAGE.md`).
3. Run `Smalltalk startUp` (a normal send) → the world is live; the REPL or a
   script file runs.

**Snapshot/image save-load is deferred** to a late sprint (§SPRINTS): the
design keeps it possible (no Rust vtables in heap objects; all heap references
are oops; Universe well-known list is the root schema) but v1 always boots
from source. *(This replaces Strongtalk's `.bst` snapshot for now.)*

---

## 4. Bytecode — the MACVM instruction set

**Δ from Strongtalk (major, deliberate):** Strongtalk's 256-opcode set encodes
IC state *in the opcode* and patches bytecode at runtime. MACVM bytecode is
**immutable**; every send carries an index into a per-method **IC side table**
(§4.3). We also drop Strongtalk's dozens of frequency-specialized opcodes
(`push_temp_0..3` etc.) in v1 — the optimizing tier, not bytecode packing, is
our performance story. ~45 opcodes total.

Encoding: 1 opcode byte + fixed little-endian operands as listed. `u8`/`u16`
operand widths; wide variants only where needed.

### 4.1 Stack & variable access
```
00  push_self
01  push_nil            02  push_true           03  push_false
04  push_smi_i8   <i8>                          (small integer constant)
05  push_literal  <u8>                          (literal frame index)
06  push_temp     <u8>                          (unified arg/temp slot)
07  store_temp    <u8>       08  store_temp_pop <u8>
09  push_instvar  <u8>       0A  store_instvar_pop <u8>
0B  push_global   <u8>                          (literal = Association; pushes value)
0C  store_global_pop <u8>
0D  pop                      0E  dup
0F  push_ctx_temp <u8 depth> <u8 idx>           (heap-context slot, §5.4)
10  store_ctx_temp_pop <u8 depth> <u8 idx>
11  push_closure  <u8 lit>  <u8 ncopied>        (lit = CompiledBlock; pops ncopied)
12  push_literal_w <u16>                        (wide literal index)
```

### 4.2 Sends, control, returns
```
20  send        <u8 ic>       21  send_super  <u8 ic>
22  send_w      <u16 ic>      23  send_super_w <u16 ic>

30  jump_fwd    <u16>         31  jump_back   <u16>    (loop poll + counter, §5.5)
32  br_true_fwd <u16>         33  br_false_fwd <u16>   (pop; non-boolean → #mustBeBoolean send)

40  return_tos               (method return, unwinds to caller)
41  return_self
42  block_return_tos         (value of block activation)
43  nlr_tos                  (non-local return to home, §5.4)
```
All jump offsets are byte distances relative to the pc **immediately after the
whole instruction** (next_bci); offset 0 = fall through.

`ifTrue:/ifFalse:/and:/or:/whileTrue:/whileFalse:/to:do:` are **compiled
inline** to jumps by the source compiler when the argument is a literal block
(as in every Smalltalk); otherwise they are real sends.

The full 31-opcode table (hex, mnemonic, operands, description) is in
`docs/ISA.md`.

### 4.3 Send-site IC table (side table — replaces self-modifying bytecode)
Each CompiledMethod references `ics: Array` with stride 4 per site:
```
[sel: Symbol][argc: smi][guard: klassOop|nil|smi(POLY|MEGA)][target]
```
States (same lattice as Strongtalk's interpretedIC):
- **empty**: guard = nil.
- **monomorphic**: guard = klassOop, target = CompiledMethod **or** nmethod id
  (smi handle into the code cache — lets an interpreter IC point at compiled
  code, Strongtalk's `compiled_send` state).
- **polymorphic**: guard = POLY marker, target = Array
  `[k1,m1,k2,m2,…, c1..c4]` — max 4 pairs *(tunable)*, ordered first-seen,
  followed by a smi COUNT TAIL (dart124 lessons items 2+3; `nil` reads 0)
  bumped only by the interpreter's poly hit — the unoptimized tier is the
  profiler, compiled code never counts. The optimizer's dominant-case
  picker reads cases count-descending and trusts a dominant only past an
  evidence floor (16 samples, 34% share); compiled-PIC-sourced counts on
  recompile remain a later step.
- **megamorphic**: guard = MEGA marker; always consults the lookup cache (§6.1).

Because the table is a normal heap Array, **GC scans ICs for free** and the
optimizer reads type feedback from it without touching code (§8.4). Miss
handling and state transitions: §5.3.

### 4.4 CompiledMethod layout (format `Method`)
Named part (oops/smis), then byte-indexable bytecode:
```
selector      Symbol
holder        klassOop            (class it's installed in)
flags         smi   (argc:4 | ntemps:8 | nctx:8 | has_ctx:1 | captures_ctx:1
                     | is_block:1 | prim_fails:1)
primitive     smi   (0 = none; else primitive id, §10)
counters      smi   (invocation:16 | loop:16 | reserved) — Strongtalk's _counters
literals      Array               (constants, Associations, CompiledBlocks)
ics           Array               (§4.3)
[size][bytecode bytes…]
```
`CompiledBlock` is the same format with `is_block=1`; its `holder` is the home
method. Blocks compile to their own bytecode object referenced from the home
method's literal frame.

### 4.5 Source compiler (Rust)
`src/frontend/`: hand-written lexer + recursive-descent parser (Smalltalk
grammar is small) → AST → single-pass code generator per method:
- resolves names (temp → slot; instvar → index via klass `inst_var_names`;
  global → Association literal),
- decides **captured temps** → heap Context slots (§5.4) by scanning block
  usage (simple escape analysis: any temp referenced by a nested block is
  context-allocated; `Δ` v1 has no "clean block" optimization),
- emits inlined control flow for literal-block control selectors,
- builds literal frame, IC table, and the CompiledMethod object.

Testable in isolation: AST → bytecode golden tests (§12).

---

## 5. Interpreter (tier 0)

### 5.1 Execution stack — VM-managed, precisely scannable
Smalltalk activations do **not** live on the Rust call stack. Each process owns
a contiguous `Vec<Oop>`-backed stack region; the interpreter is one Rust
function looping over frames on that stack. Consequences: GC root-scanning is
exact and trivial (§7.5), NLR is stack-slicing, and deopt (§8.7) can
materialize frames directly.

Frame layout (grows upward; FP-relative slots):
```
FP-1  … caller's operand stack (incl. receiver + args pushed by caller) …
FP+0  method        (CompiledMethod oop)
FP+1  saved_fp      (smi-encoded caller FP index)
FP+2  saved_bci     (smi: caller's resume bytecode index)
FP+3  context       (heap Context oop | nil)
FP+4  receiver      (copied from arg area for fast access)
FP+5  serial        (smi: monotonic frame serial — dead-home detection, §5.4)
FP+6  marker        (nil | UnwindToken | ensure/ifCurtailed marker, §5.4)
FP+7… temps[ntemps] then operand stack…
```
`sp` always indexes the **next free slot** (top of stack = `stack[sp-1]`).

Receiver+args stay where the caller pushed them (callee addresses args as
negative offsets), Smalltalk-style; `return` pops down to the receiver slot and
overwrites it with the result.

### 5.2 Dispatch loop
A single `match` on the opcode byte in a hot loop (`src/interpreter.rs`).
Rust/LLVM compiles this to a jump table; no computed-goto needed at tier-0
performance targets (§13). Debug builds add bytecode tracing and an
every-N-bytecode GC-poll stress hook (§12.3).

### 5.3 Send protocol & IC transitions
```
send <ic>:
  rcvr  = stack[sp - 1 - argc]        (sp = next free slot, §5.1)
  k     = klass_of(rcvr)                     (smi → smi_klass, else header)
  IC hit (guard == k)          → activate target (method or nmethod)
  IC empty                     → full lookup (§6.1); install mono(k, m)
  IC mono, wrong k             → promote: POLY array [k0,m0,k,m]
  IC poly, miss, arity < 4     → append pair
  IC poly, full                → MEGA
  IC mega                      → lookup cache probe (§6.1)
  lookup failure               → doesNotUnderstand: (build Message, re-send)
```
Activating a CompiledMethod: run primitive if `primitive ≠ 0` (§10; on success
return value immediately; on failure fall into bytecode). Then push a frame,
allocate a Context if `has_ctx`, bump `counters` and check
`InvocationCounterLimit = 10_000` *(tunable — Strongtalk's value)* → §8.1.

`send_super` looks up starting at `method.holder.superclass` and caches in the
same IC structure (guard = receiver klass still).

### 5.4 Blocks, contexts, non-local return
- **Closure creation** (`push_closure`): allocate `BlockClosure { method,
  home_frame_ref, copied[] }`. `home_frame_ref` = (process id, frame FP index,
  frame serial) — a *safe* reference that can detect a dead home frame.
  Captured temps live in the home frame's heap **Context** (format `Context`),
  created at method entry when `has_ctx`; the closure copies the Context oop
  (so both see updates). **Pinned convention:** `push_closure` implicitly
  captures the home **receiver** as `copied[0]` and, iff the enclosing scope
  owns a Context, that Context as `copied[1]`; the bytecode's `ncopied` operand
  counts only *additional* by-value captures and is **always 0 in v1** — the
  compiler routes every explicit capture (read-only included) through the
  Context. By-value read-only captures are a deferred optimization (moot once
  S14's Context elision lands). *(Amended per S4/S5 review.)*
- **Block activation**: `value/value:…` are primitives that push a frame for
  the CompiledBlock; `push_ctx_temp <depth>` walks `Context.home_hint` links
  for nested blocks.
- **NLR** (`nlr_tos`): find the home frame (via `home_frame_ref`); if dead →
  send `cannotReturn:`. Otherwise unwind: walk frames from current down to
  home, running `ensure:` handlers (frames carry an ensure-marker flag set by
  the `ensure:` primitive), then return from the home frame with TOS.

### 5.5 Loop poll (`jump_back`)
Every backward jump: (a) increments the method's loop counter share (folded
into `counters`), triggering compilation at `LoopCounterLimit = 10_000`
*(tunable)* → later OSR (§8.7); (b) is a **GC/interrupt poll point** — checks
a `pending` flag in the VM state (interrupt, stack overflow check, future
safepoint).

---

## 6. Runtime services

### 6.1 Lookup
`lookup(klass, selector)`: probe `MethodDictionary` up the superclass chain.
Backed by a global **lookup cache**: open-addressed table of
`(klassOop, Symbol) → CompiledMethod`, 4K entries *(tunable)*, direct-mapped
with 2-way victim slot — Strongtalk's `lookupCache` shape, keyed exactly like
its `LookupKey (_klass, _selector_or_method)`. Flushed on: method
(re)definition, class hierarchy change, full GC (klass addresses move).

### 6.2 Method installation & invalidation
Defining a method: install in MethodDictionary, flush lookup cache; flush any
IC entry naming that (klass, selector) lazily (ICs self-heal on next guard
miss — guards compare klass identity, and a stale mono target for the same
klass is replaced because installation bumps a per-klass **dependency
version**; compiled-code dependencies: §8.6).

### 6.3 Errors
VM-level panics only for VM bugs. All user-level errors are sends: DNU,
`mustBeBoolean`, `cannotReturn:`, `error:`, primitive-failure fallbacks.
`Object>>error:` in v1 prints the message + a Smalltalk stack trace (walk
frames, read method selectors) and terminates the process.

---

## 7. Memory & GC

Strongtalk's collector, modernized (Δ: explicit worklist, no pointer
reversal; 64-bit mark word; side-map mark saving).

### 7.1 Address space
One `mmap` **reservation** of 8 GiB *(tunable)* `PROT_NONE`, committed as
needed. Layout (low → high): `[eden][from][to][old segments…]`. A single
boundary check `addr >= old_start` implements `is_old`/`is_new` (the trick
Strongtalk's `Mem_Tag=1` boundary compare enables works on tagged oops too).
Defaults *(tunable)*: eden 4 MiB, survivors 512 KiB each, old segment 16 MiB.

### 7.2 Allocation
- Fast path: bump `eden_top` (fields in the VM state; in compiled code, kept in
  the reserved VM register block, `arm64.md` §3): `new_top = top + size;
  if new_top > eden_end { slow path } else commit`.
- Slow path: scavenge; retry; if still no fit (or size > 256 KiB *(tunable)*)
  allocate directly in old gen (with card-table growth), possibly triggering
  full GC / segment growth.
- All allocation goes through one Rust choke point → the GC-stress test mode
  (§12.3) can collect on **every** allocation.

### 7.3 Scavenger (minor GC — Cheney copy)
1. Roots: Universe well-known oops, symbol table, Handles (§7.6), **process
   stacks** (exact: walk frames — every slot is an oop except the smi-encoded
   saved_fp/bci, which are valid smis anyway), dirty cards (§7.4), nmethod
   embedded oops + IC-table arrays reachable from code (§8.5).
2. Copy new-gen survivors eden+from → to (forwarding via mark word);
   `age += 1`; objects with `age ≥ tenuring_threshold` are promoted to old
   (their card range dirtied via `record_multistores`).
3. Cheney scan `to` and promoted-region until fixed point; swap from/to.
4. **Adaptive tenuring**: ageTable histogram of surviving bytes per age;
   `tenuring_threshold` = smallest age keeping survivors ≤ ½ survivor size
   (Strongtalk/HotSpot policy).

### 7.4 Write barrier & remembered set
Card table over the old-gen range, `card_shift = 9` (512-byte cards,
Strongtalk's wired-in value), one byte per card, **dirty = 0**:
```
store(obj, field_offset, val):
    *slot = val
    if obj >= old_start && val is heap_oop_in_new:   // value-conditional (Self's cheaper form)
        cards[(slot - old_start) >> 9] = 0
```
Compiled fast path: `lsr xtmp, xslot, #9; strb wzr, [xcards, xtmp]` behind a
new-gen check (2–4 instructions). Each old `Space` keeps an **object-start
offset table** per card so scavenging a dirty card can find the first object
header (Strongtalk's scheme).

### 7.5 Full GC (major — mark + slide compact)
1. Mark from the same root set with an **explicit worklist** (`Vec<Oop>`), mark
   bit in the mark word. `tagged_contents` objects scan without format
   dispatch.
2. Sweep-compute forwarding: slide live objects toward each space's bottom;
   forwarding oop written into the mark, displaced marks saved in a side map
   (§2.2).
3. Update phase: rewrite all root and heap references; rewrite nmethod
   embedded oops via reloc records (§8.5); fix interpreter stacks.
4. Restore saved marks; reset card table (re-dirty cards containing old→new
   after promotion churn — conservatively, dirty all cards of moved regions).
Lookup cache and any Rust-side address-keyed tables are flushed (klass oops
moved).

### 7.6 Handles — the Rust/GC contract
Any Rust runtime code holding an oop **across a possible allocation** must hold
it in a `Handle`. `HandleScope` is a stack-discipline arena registered in the
VM state and scanned/updated as a root set. Enforced by construction: the
allocator's Rust signature takes `&mut VmState`, so no `&`-borrows of heap
memory survive an allocation call, and in debug builds every vacated space is
**poison-filled** after a collection so a stale raw `Oop` dereference fails
fast — combined with the GC-stress mode (§12.3), violations surface
deterministically. *(Amended per S7 review: Rust cannot rewrite locals; the
enforcement is borrow-shape + space poisoning + stress.)*

**Post-mortem amendment (S7-11, 2026-07-02 — see A21): the above is an
incomplete description of the actual risk, and following it as written
produces recurring bugs, not a one-off oversight.** Two findings from
actually running the enforcement this paragraph promises:

1. **The poison-filling this section specifies was never implemented.**
   `sprint_s07_detail.md`'s own SPEC-QUESTION flagged the exact wording
   change ("the spaces a stale oop points into are poison-filled") back
   when S7 was written, but no code ever wrote a `POISON` pattern into
   vacated eden/from-space. Every stale-handle bug in the S7-9/S7-10/S7-11
   line was instead being found (when it was found at all) as a confusing,
   action-at-a-distance panic — a `to`-space bounds guard tripping cycles
   after the actual misuse, with no trace back to the responsible call
   site. This is a plain implementation gap against this section, now
   closed (`oops::layout::POISON`, `memory::scavenge::poison_range`).

2. **The deeper problem, which poisoning only makes legible rather than
   fixes: `Handle<T>` is not actually used as this section implies.** As of
   this writing it appears in exactly **one** `pub fn` signature in the
   entire codebase — its own constructor. Every other use, including in
   the very functions this section is trying to protect (`install_method`,
   `MethodDictOop::insert`, `BytecodeBuilder::finish`, `alloc_words` and
   its callers), creates and consumes a `Handle` entirely *inside* one
   function body. Not one allocating function's public signature takes or
   returns `Handle<T>` — they take/return bare `KlassOop`/`SymbolOop`/
   `MethodOop`/`Oop`, which are `Copy` and carry no signal, syntactic or
   type-level, that holding one across a call is dangerous. A function
   like `install_method` wraps its *own* parameters in handles internally
   (with a comment explaining exactly why), which protects them for the
   duration of that function's body — and does nothing about the identical
   danger one level up, at the call site, for the expression that computed
   the argument. This inverts how the prior art this design is modeled on
   (V8's `HandleScope`, JNI local refs) actually works: there, the *safe*
   handle type is the default currency for anything crossing a safepoint,
   and the raw/unsafe pointer is the deliberately awkward, tightly-scoped
   exception. Here it is backwards — the unsafe type is the pervasive
   default, including at allocating functions' boundaries, and `Handle` is
   an opt-in exception that only the callee remembers to reach for, too
   late to help the caller.
   
   This is why the same bug class — a bare oop-wrapper local read after an
   intervening allocation — kept being independently reintroduced across
   S7-9, S7-10, and S7-11, *including inside code written specifically to
   fix earlier instances of it*, and including in the GC's own unit test
   suite (`memory::scavenge::tests::copy_exact_fill_survivor`), not just
   hastily written test helpers. It is not a diligence failure by whoever
   wrote that code sprint over sprint: no amount of care fixes a bug class
   the API shape itself continuously regenerates. Detection (poisoning +
   `MACVM_GC_STRESS=1`) is a necessary backstop for genuine one-off
   mistakes, but it cannot be the *only* defense against a source that
   manufactures new instances at a rate proportional to how much allocating
   code gets written — a backstop with no upstream prevention just means
   the same class of bug fires forever, one newly written function at a
   time.
   
   **Corrective direction:** the two structural gaps above (the unsafe type
   is the default currency; handles carry no validation) are resolved by the
   revised handle contract specified in **§7.6.1**, adopted from the Locus
   sister VM's proven handle layer. That section is the design of record;
   this paragraph's earlier "not yet implemented" wording is superseded by
   it.

### 7.6.1 Revised handle contract — generational, validated, currency-level
*(S7-11, 2026-07-02 — the design of record; supersedes §7.6's "opt-in,
internal-only, unvalidated `Handle`" as the target. Adopted from the **Locus**
sister VM, `../locus/locus-gc`, same author, whose handle layer solves this
exact problem in production.)*

**Provenance and scope of the import.** Locus's collector *engine* is
`newgc-core` — the page-based, cohort-promoting, conservatively-pinning
mark-evacuate GC already assessed and **rejected** for MACVM in
[`reference-vm-analysis.md`](reference-vm-analysis.md) §5 (five still-valid
mismatches: page/cohort geometry vs. contiguous eden+survivors+7-bit age; no
slide compactor; header-self-describing vs. MACVM's klassOop-chase sizing; no
single `is_old` boundary). **That rejection stands — MACVM keeps its own
collector (§7.3, §7.5).** What transfers is strictly the *handle layer* that
Locus builds *above* the collector, which is engine-independent. The mistake
being corrected was never "MACVM shouldn't use newgc-core"; it was
"having (correctly) rolled its own collector, MACVM then rolled a *weak* handle
layer instead of copying the strong one sitting in the author's own Locus
repo." Three concrete changes bring MACVM's handle layer to Locus's model:

**Which change fixes what — stated honestly, because the first draft of this
section overclaimed it (adversarial review, 2026-07-02).** The bug class has
two halves and they need different fixes:
- The bugs actually hit in S7-9/10/11 (`mk` returning a bare `SymbolOop` past
  its scope; `install_prims` caching bare `KlassOop` locals across allocations;
  `copy_exact_fill_survivor` rooting via `vm.stack.push`) were **bare oops,
  never `Handle`s**. Generation validation (change 1) fires only on `Handle`
  access, so it catches *none* of these. **Change 2 is the actual fix for that
  half** — once a boundary takes `Handle<T>`, the bare oop is unspellable there.
- Change 1 catches the *other* half: misuse of an existing `Handle` after its
  scope dropped (the arena-staleness we also hit). It is a soundness backstop,
  not the primary fix, and it is what upholds change 2's type-safety (below).
- Everything not yet reshaped by change 2 stays covered **only** by poison +
  stress. That is the honest coverage map; do not read change 1 as "makes every
  latent instance loud" — it only makes *handle* misuse loud.

1. **Handles carry a generation; the arena validates every access.**
   `HandleArena` gains a `gen: Vec<u32>` parallel to `slots` but
   **persistent** — never truncated (only `slots` is, on scope `leave`).
   `Handle<T>` gains a `generation: u32` stamped at mint. **`get`/`set`/`oop`
   apply Locus's *three* guards in order** (`lib.rs:500` `resolve`), not just
   the generation compare: (i) index `< slots.len()` (in the live region);
   (ii) the slot is not vacated; (iii) `gen[index] == handle.generation`.
   All three matter — the generation compare alone is unsound, because a bare
   generation field with no bounds/vacancy guard still index-panics or reads a
   stale `gen` entry out of the live region.
   
   **Divergence from Locus, required because MACVM drops Locus's free-list +
   tombstone:** Locus tombstones a freed slot and its `resolve` rejects a
   tombstoned slot outright, so its generation only needs to bump on *re-push*.
   MACVM's pure-LIFO arena has no tombstone — a truncated slot just retains its
   last value until overwritten — so bumping only on re-push leaves a
   **false-negative**: a handle whose scope dropped, but whose exact index is
   never re-pushed before the stale use, would still validate and read a
   foreign oop. MACVM therefore **bumps `gen[i]` on *vacate*** — on scope drop,
   `for i in saved_len..len { gen[i] = gen[i].wrapping_add(1) }` (~2 lines in
   `HandleScope::drop`). Then *any* handle outliving its scope is stale the
   instant the scope drops, regardless of later push traffic, and guard (i)
   alone catches the common case while (iii) catches index-in-range reuse. This
   is the property Locus gets from tombstones; MACVM gets it from vacate-bump.
   
   `gen` is `u32`; bump is `wrapping_add`. Aliasing a *specific* live-but-stale
   handle requires 2^32 reuses of one slot between that handle's mint and its
   stale use — accepted (Locus accepts the same at 2^22). The collector **never
   touches `gen`** (it rewrites `slots[..len]` in place), so a *correctly live*
   handle across a scavenge still resolves — same index, same generation, slot
   repointed. Relocation stays invisible; only staleness is caught. The assert
   is gated `#[cfg(debug_assertions)]` (like the poison machinery): handles are
   on the **send hot path** (`activate_method` enters a scope + mints a handle
   every activation, `send.rs:42`), so release builds pay only the extra field,
   not the compare.

2. **`Handle<T>` is the currency at allocating-function boundaries — the
   primary structural fix.** Any function that (a) allocates internally, or (b)
   returns a value the caller holds across a further allocation, **takes and/or
   returns `Handle<T>`**, never a bare `Oop`/`KlassOop`/`SymbolOop`/`MethodOop`.
   `install_method`, `MethodDictOop::insert`, `alloc_*`, and `BytecodeBuilder`'s
   public surface reshape accordingly. This pushes rooting to where the caller
   *computes* the value, enforced by the signature.
   
   **Ergonomics — the friction relocates to the caller, it does not vanish
   (review M2).** `Handle<T>` being lifetime-free avoids the borrow-checker
   *wall* that killed the `Handle<'s,T>` sketch, but the E0503 seen in practice
   (`install_method(vm, vm.universe.closure_klass, …)` — reading `vm.universe.X`
   while `vm` is `&mut`-borrowed by the call) comes from the *argument
   expression*, and reshaping the signature just moves the mint up one frame:
   ```rust
   // before:  install_method(vm, vm.universe.closure_klass, sel, m);
   // after:
   let s  = HandleScope::enter(vm);
   let k  = s.handle(vm, vm.universe.closure_klass);   // read happens before the &mut call
   install_method(vm, k, sel_h, m_h);
   ```
   That is *more* ceremony per call site and pushes a `HandleScope` into callers
   that lack one — including the send path. To keep this from being resisted the
   way opt-in `Handle` already was, S7.5 must ship an ergonomic minting helper
   (e.g. a `Universe`/`VmState` accessor returning a `Handle<KlassOop>` for
   well-known klasses directly, so "root a well-known klass" is one call, not a
   three-line prologue). Budget the reshape as touching **every** `install_method`
   call site (`interpreter/ic.rs`, `send.rs`, `unwind.rs`, `blocks.rs`) plus the
   allocator and builder callers — this is the invasive bulk of S7.5, not a
   type-swap.
   
   **Type-safety note (review m4):** once `Handle<KlassOop>` etc. cross public
   boundaries, `Handle::get`'s `from_oop_unchecked` (`handles.rs:179`) trusts the
   slot was last written as `T`. The three-guard + vacate-bump validation from
   change 1 is what upholds that contract — without it, a `Handle<KlassOop>`
   aliasing a slot later reused for a `Handle<MethodOop>` is an unchecked
   wrong-type transmute, a soundness hole. Change 1 is therefore a *prerequisite*
   for change 2, not merely a backstop.

3. **The handle arena IS the precise Rust-side root set — stated as
   load-bearing, not incidental.** (Already true in `scavenge_handle_arena_
   roots`; §7.3 step 1 and this section now name it as a designed invariant so
   future code treats "in the arena ⇔ rooted" as the contract, and the scoped
   stack — `enter`/`handle`/scope-drop-truncates, already implemented — as the
   sole lifetime model. A LIFO scope stack means every slot below the top is a
   live root by construction, so scanning the whole arena *is* the precise
   collection; no per-handle liveness marking is needed.)

4. **Primitives are a *third* rooting surface — the "confined to" boundary
   below does NOT cover them, and they are currently unprotected (review C2).**
   A primitive body is Rust runtime code holding oops across an allocation
   (`prim_basic_new` reads `klass` from `args[0]` and holds it across
   `alloc_slots`, `primitives.rs:641`), but `args` is a copy in a Rust-local
   `[Oop; 6]` buffer (`send.rs:52`), *not* the operand stack — so a scavenge
   inside the alloc rewrites the stack and `vm.regs` but not `buf`. It is safe
   today only by the unstated accident that klass oops are old-gen/immobile
   under a scavenge; the first `can_allocate` primitive holding a *young* heap
   arg across its allocation breaks silently. The rule (make it explicit here
   and in §10): **a primitive re-reads live arguments via `prim_arg(i)`
   (`vm_state.rs:248`, which reads the scanned operand-stack slot) after any
   allocating call, and never touches the `args` copy afterward.** Better,
   S7.5 should hand `can_allocate` primitives `Handle`s (or `&[Handle]`) so
   they share the same validated currency; §10's existing "needs
   handles/safepoint" label for `can_allocate` becomes true rather than
   aspirational.

**Deliberately NOT imported from Locus** (recorded so the reviewer doesn't
flag their absence as an oversight):
- **`newgc-core` (the collector engine)** — see provenance above.
- **Locus's 16-bit self-identifying `MAGIC` tag on handles.** That exists so a
  *conservative* scan of a GC-oblivious compiled machine stack can tell a
  handle from an int/pointer. MACVM's compiled-frame plan (S12+) is *precise*
  oop-maps at safepoints — strictly better (no false retention), so the magic
  is unnecessary. MACVM's on-stack / in-object interpreter currency stays the
  raw `Oop` (walked exactly as roots via frame maps, §7.3); the `Handle`
  currency covers *Rust runtime code holding oops between allocations* — with
  the explicit exception of primitive bodies (change 4 above), which root by
  live-arg re-read, not handles, until S7.5 optionally unifies them.
  (`is_well_formed`, which Locus derives from the magic to answer "is this
  value a live handle?", goes too; MACVM never needs that at runtime. If it
  ever does — a bare index can't self-identify — revisit.)
- **Locus's `leave_with`/escape idiom** is optional — adopt only if a
  reshaped allocating function needs to return an in-scope freshly-built object
  (the common `alloc → return Handle` case); otherwise plain scope drop
  suffices.

**Scheduling.** This lands as a dedicated hardening pass (**S7.5**) *before*
S8 builds more moving-GC machinery on the current model. Order within it:
1. **Change 1 first** — small, isolated to `handles.rs` (persistent `gen`,
   three-guard access, vacate-bump, debug-gated assert). A *soundness backstop*
   and the prerequisite for change 2's type-safety; do **not** expect it to
   surface the cited bare-oop bugs (it can't — see the honesty note above).
2. **A regression test that reproduces the actual failure**, both directions:
   one that reshapes a helper like `mk` to return `Handle` and asserts the old
   bare-oop return no longer type-checks; one that deterministically trips the
   stale-handle assert. The whole premise of A21 is that tests+stress were an
   insufficient backstop — shipping the fix without a test pinning the fixed
   behavior repeats the pattern.
3. **Change 2** — the invasive currency reshape. **Migration ordering matters**
   (review): the allocator/builder surface change 2 reshapes is called *by*
   `handles.rs`'s own machinery, so reshape leaf allocators **last**, or the
   ripple deadlocks on itself.
4. **Change 4** — bring primitives into the contract (or at minimum a
   `debug_assert`/lint that a `can_allocate` primitive never reads `args[k]`
   after an alloc).

See amendment **A21**. *(This scheduling and the honesty note reflect an
adversarial design review of the first draft of §7.6.1, 2026-07-02, which
caught that the draft (a) overclaimed change 1, (b) specified an unsound
generation-only guard, and (c) wrongly excluded primitives from the bug
surface. All three are corrected above.)*

---

## 8. Adaptive compilation (tier 1)

### 8.1 Trigger & policy
- Interpreter bumps `counters` on activation; overflow of
  `InvocationCounterLimit` (10 000) or `LoopCounterLimit` (10 000) calls
  `Recompilation::trigger(rcvr, method)`.
- Policy (Strongtalk's `RecompilationPolicy`, simplified for v1): walk the
  interpreter stack up to 10 frames *(tunable)*, prefer compiling the **caller**
  of a trivially-small hot method (so it gets inlined), else the hot method
  itself. Levels: `MaxRecompilationLevels = 4`, `MaxVersions = 3` (Strongtalk's
  values) — a level-k nmethod that re-triggers compiles at k+1 with larger
  inlining budgets; version cap stops thrash. **Effectiveness check** (Self's
  `checkEffectiveness`, added per S14 review): each nmethod stores a
  `profile_hash` of its send-site type profile; if a recompilation produces an
  identical profile, counting is disabled on the new nmethod (an
  uncommon-trap overflow re-enables it).
- Compilation runs **synchronously on the VM thread** in v1 (no background
  compile thread → no `MacJit` threading issue, `arm64.md` §2).

### 8.2 nmethod & code table
nmethod = one code-cache allocation:
```
header { key: (klassOop, Symbol),          // customization key (LookupKey)
         entry_off, verified_entry_off, osr_entry_off,   // 0 = no OSR entry
         level, version, state {alive, not_entrant, zombie},
         counter, counter_limit,           // compiled-side re-trigger (§8.1)
         profile_hash,                     // effectiveness check (§8.1, per
                                           //  Self's checkEffectiveness)
         code_len, reloc_off, oopmap_off, scopes_off, pcdesc_off,
         osrmap_off, deps_off }
// Per-site uncommon-trap counters live Rust-side (keyed nmethod id × trap
// imm), not in the header. (Amended per S13–S15 review.)
code…  reloc[]  oopmaps  scopedescs  pcdescs  deps[]
```
- **Customization**: keyed by *(receiver klass, selector)* — one nmethod per
  concrete receiver class (Self/Strongtalk model). The **code table** (Rust
  hash map key → nmethod id) resolves them; interpreter ICs may cache nmethod
  ids directly (§4.3).
- `verified_entry`: receiver klass already guaranteed by the caller's IC guard.
  `entry`: prologue loads receiver klass, compares against the customized
  klass, and falls back to lookup on mismatch.

### 8.3 Compiler pipeline
```
bytecode ──decode──► CFG of basic blocks, stack→SSA-lite conversion
        ──feedback─► IC-driven receiver-type annotation (§8.4)
        ──inline───► inlining + customization + block inlining (§8.4)
        ──opt──────► constant fold, smi strength-reduce, guard motion/dedup,
                     redundant klass-test elimination (dominator-local),
                     escape-lite: eliminate Contexts for fully-inlined blocks
        ──lower────► machine-level IR (2-address, AArch64-shaped)
        ──regalloc─► linear scan over virtual regs; oop-ness tracked per vreg
        ──emit─────► JASM encode::encode(mnemonic, operands) via Assembler trait
                     + reloc records + oop maps at safepoints + scope descs
```
IR: SSA-lite — basic blocks with phi-less join handling via stack-slot merges
in v1 (block-argument form is a refinement) *(Δ from Strongtalk's PReg graph:
simpler, register-allocation-friendly)*. Every IR value carries an `is_oop`
bit — the register allocator's spill slots and live sets are therefore oop-
precise by construction, which is what makes oop maps (§8.5) cheap to emit.

### 8.4 Type feedback & inlining
- Feedback source: the method's **IC side tables** (interpreter) and PIC stubs
  (compiled callers). Reading is trivial — they're heap arrays (§4.3).
- For each send site: mono → inline the target if its cost ≤ budget
  (cost model: bytecode length with cheap-op discounts; budgets grow with
  recompilation level); poly with a dominant class → inline dominant case
  behind a klass guard + slow-path send; poly flat / mega → leave a call.
- **Uncommon branches**: an unreached IC state or an unlikely guard failure
  compiles to `brk #imm` (uncommon trap, §8.7) instead of code — Self's lazy
  cold path.
- **Block inlining**: literal blocks passed to inlined methods are inlined into
  the caller (this is where Smalltalk performance comes from — `do:`,
  `ifTrue:`, `whileTrue:` chains collapse); their Context allocation is
  elided when no captured temp escapes the inlined extent.
- **Δ**: Self-style *splitting* deferred to a stretch sprint; v1 relies on
  guards + uncommon traps.

### 8.5 Compiled sends, PICs, GC interaction
- Monomorphic compiled send: `bl target_entry` (the **guarded** entry — its
  prologue checks the receiver klass; `verified_entry` is only for callers that
  have already proved the klass: PIC cases and customized inlined calls) with an
  `InlineCache` reloc; miss protocol patches the branch (Branch26; > ±128 MB →
  x16 veneer) then `sys_icache_invalidate`. *(Amended per S11 review — the
  original text wrongly said `verified_entry` here.)*
- Polymorphic: patch to a **PIC stub** allocated in the code cache: sequence of
  `ldr klass; cmp; b.eq target` up to 4 entries, tail → lookup stub.
- Megamorphic: call the shared lookup-stub (probes lookup cache, falls back to
  full lookup, never patches back).
- **GC**: embedded oops via `Oop`-kind reloc records over a **literal pool** at
  the end of each nmethod (GC updates aligned data words — no instruction
  re-encoding, no icache flush on GC; `arm64.md` §4.1). Oop maps at every
  safepoint (calls, allocation, loop polls) enumerate oop registers/spill
  slots. Weak treatment of nmethod `key.klass` + dependency-based flushing on
  class death (v1: full GC flushes nmethods whose key klass died).

### 8.6 Dependencies & invalidation
Each nmethod records the (klass, selector) pairs it inlined or guard-assumed
(`deps[]`). Redefining a method / changing a hierarchy walks a Rust-side
dependency index → mark dependent nmethods **not_entrant** (patch `entry` and
`verified_entry` with a branch to the deopt stub; icache flush), and
deoptimize any activations of them found on the stack (§8.7). Not-entrant →
zombie → freed once no frames reference them (frame sweep at each full GC).

### 8.7 Deoptimization & OSR (the all-new arm64 machinery)
- **Scope descs** (per safepoint & trap): the chain of virtual frames
  (method, bci, receiver/temp/stack-value locations: constant | register |
  frame slot), LEB128-packed. **PcDescs**: sorted (code offset → scope desc)
  index. This is Self/Strongtalk's `ScopeDesc/PcDesc`, emitted by our
  compiler's emit stage.
- **Uncommon trap**: `brk #imm` → Mach exception/`SIGTRAP` handler defers to a
  runtime routine that: reads the scope descs for the trap pc, **materializes
  interpreter frames** on the process stack (our frames are plain oop arrays —
  §5.1 pays off here), pops the compiled physical frame, resumes the
  interpreter at the recorded bci. Trap counters per site; past
  `UncommonTrapLimit` *(tunable, Self used 1000)* → recompile without that
  assumption.
- **Invalidation deopt**: not_entrant nmethods with live activations get their
  return addresses redirected to a deopt trampoline (patched in the frames'
  saved-LR slots), converting lazily on return — Strongtalk's approach.
- **OSR** (interpreter → compiled, on loop-counter overflow): compile with an
  OSR entry for the hot loop's bci; build the compiled frame from the
  interpreter frame (reverse of deopt), jump in. Scheduled late (Sprint 15);
  deopt direction ships first (correctness > peak).
- PAC: v1 does not sign VM return addresses (`arm64.md` §5) — deopt frame
  surgery stays plain.

### 8.8 Safepoints (compiled code)
Interpreter-only VM is trivially safepointed (§0). Compiled code polls at
method entry + loop back-edges: `ldr wzr, [x28, #poll_off]` against a guard
page in the VM-state block — flipping the page's protection stops all compiled
code at oop-map'd pcs *(the page-poll variant is v2; v1 compiled code checks a
flag byte + conditional branch, simpler and fast enough single-threaded)*.

---

## 9. Code cache

`MacJit`-backed (vendored, `reference-vm-analysis.md` §4.3): one RWX `MAP_JIT`
region, 64 MiB *(tunable)*, bump-allocated segments per nmethod/PIC/stub with
a free list on flush; no compaction in v1 (zombie space is reclaimed by the
free list; a compacting code GC is a stretch goal). Write protocol per
publish: `pthread_jit_write_protect_np(0)` → write/patch →
`(1)` → `sys_icache_invalidate(range)`. Shared runtime **stubs** generated at
startup: lookup stub, megamorphic stub, deopt trampoline, allocation slow
path, DNU thunk, NLR unwind helper.

---

## 10. Primitives

Rust functions `fn(&mut VmState, args: &[Oop]) -> PrimResult` with three
variants: `Ok(oop)` (return value), `Fail` (fall into the method's bytecode
fallback), and `Activated` (the primitive pushed a guest frame itself — the
`value` family, `ensure:`, `ifCurtailed:` — and the interpreter simply
continues dispatching). Registered in a static table; bound by
`<primitive: N>`; metadata per primitive: `can_allocate` (needs
handles/safepoint), `can_fail`. *(Amended per S3/S4 review: `Activated` added.)* Initial set (~40):

| Group | Primitives |
|---|---|
| smi | `+ - * // \\ bitAnd bitOr bitXor bitShift < <= > >= = ~=` (fail on overflow/non-smi → Large/Double fallback code) |
| Double | `+ - * / < = sqrt floor asDouble fromSmi printString-support` |
| oops | `identityHash class == basicNew basicNew: instVarAt: at: at:put: size` (at:/at:put: klass-format checked) |
| ByteArray/String | `at: at:put: size replaceFrom:to:with: hash compare:` |
| BlockClosure | `value value: value:value: … valueWithArguments:` |
| Symbol | `intern` |
| control | `ensure: ifCurtailed:` (set frame markers) |
| system | `quit gcScavenge gcFull gcStats millisecondClock printOnStdout: sourceCompile: error:` (dev hooks; `gcStats` returns a pinned Array of counters — soak tests and S14's Context-elision gate read it; `error:` prints message + Smalltalk stack trace and terminates, §6.3) |
| game (ids 200–215) | `GamePane` drawing/palette/present + frame-loop (`run`/`stepWithKeys:`), `Sprite` define/move/colour, `Sound`/`Tune` audio — each validates its args, emits a `GameCommand` over the VM→GUI game sink, and returns `self` (`docs/gamepane_design.md`) |
| workers (ids 220–228) | `Worker` spawn/send/poll/await/terminate + the MOP pickle (`pickle:`/`unpickle:`) — multi-VM message passing, deep-copy only (`docs/multi-smalltalk-worker.md`) |
| Cocoa (ids 230–245) | ObjC sends (fixed-arity, auto-shape, and the main-thread hop), NSString bridging, autorelease pools, target/action callbacks (`docs/cocoa_bridge_design.md`) |
| reflection (ids 246–248) | `respondsTo:` (method-dict chain lookup), `shallowCopy` (same-klass, same-size, slots copied verbatim), `numArgs` (a block's argc) — the three that cannot be written in Smalltalk; `copy` = `shallowCopy` + `postCopy` sits on top in `world/59_reflection.mst` (`docs/ISA.md`) |

`sourceCompile:` exposes the Rust source compiler to Smalltalk (later enables
an in-language `compile:` and eventually a self-hosted tools story).

Later passes added the SIMD (`docs/SIMD.md`), unboxed-float (`docs/float_fastpath_design.md`),
FFI (`docs/FFI.md`), and game (`docs/gamepane_design.md`) groups, so the table is
now larger. The full, current id-by-id table — cross-checked against
`primitives.rs`'s own `prim_ids_frozen` regression test — is in `docs/ISA.md`;
this section's groupings are the design intent, that document is the exhaustive,
load-bearing reference.

## 11. Concurrency — deferred by design

v1 is single-process (one green process, no Scheduler, no preemption). The
design keeps the door open: per-process stacks are already first-class objects,
loop polls exist, and the VM state is one struct (no globals) — adding green
processes + a scheduler is additive. Native threads / parallel GC are out of
scope for this research VM.

---

## 12. Testing strategy (cross-cutting; per-sprint gates in SPRINTS.md)

1. **Unit (Rust `cargo test`)** — tags/mark-word bit invariants, allocator
   arithmetic, symbol interning, MethodDictionary, lookup cache, card-index
   math, LEB128 scope-desc round-trip, encoder wrappers.
2. **Golden tests** — (a) source compiler: `.mst` method → disassembled
   bytecode listing compared to checked-in `.expected`; (b) interpreter:
   program → stdout transcript golden files.
3. **In-language suite** — `world/tests/*.mst`, a ~100-line SUnit-lite written
   in MACVM Smalltalk; run as part of `cargo test` via the embedded VM. Grows
   every sprint; this is the primary regression net.
4. **GC torture** — `MACVM_GC_STRESS=1`: scavenge on *every* allocation;
   `=full`: full-GC every N allocations. Entire in-language suite must pass
   under both. Handle-discipline violations surface here deterministically.
5. **Differential execution** — once tier 1 exists: run the in-language suite
   with `MACVM_JIT=off|threshold=1|normal` and diff transcripts. `threshold=1`
   (compile everything immediately) is the JIT's torture mode; combined with
   GC stress it is the moving-GC-through-compiled-frames gate.
6. **Deopt torture** — `MACVM_DEOPT_STRESS=1`: every uncommon-trap-eligible
   guard fires once; every nmethod is invalidated after M calls. Suite must
   pass.
7. **Encoder trust** — vendored JASM corpus test (1,181-form frozen LLVM-MC
   oracle) runs in CI unchanged.
8. **Benchmarks** (tracking, not gating): fib, sieve, nested-send dispatch
   micro, OrderedCollection churn; later Richards & DeltaBlue ports (the
   classic Self/Strongtalk suite).

## 13. Performance targets (research-grade, honest)

| Milestone | Target |
|---|---|
| Interpreter | ≥ 50M bytecodes/s on M-series; fib(30) < 2 s |
| Tier 1, no inlining | ≥ 5× interpreter on send-free arithmetic |
| Tier 1 + inlining | ≥ 10× interpreter on fib/sieve; ≥ 5× on Richards |
| Scavenge pause | < 1 ms at default eden |
| Full GC pause | < 50 ms at 100 MB live |

These are sanity bars to catch architectural mistakes, not HotSpot ambitions.

---

## 14. Decisions closed by this spec
- Tag scheme: **2-bit, Int=00/Mem=01/Mark=11**, 62-bit smis (§2.1).
- Mark word: explicit 64-bit fields, no punning (§2.2).
- Floats: **boxed Double** (NaN-boxing shelved) (§1.3, §2.3).
- Precise GC from day 1 — interpreter stacks are VM-managed and exact;
  compiled-frame oop maps arrive with tier 1 (§5.1, §7.3, §8.5).
- Bytecode: immutable ~45-opcode set + IC side tables (§4).
- Bootstrap: Rust source compiler + `.mst` world files; no image in v1 (§3.2).
- Tiers: interpreter + one optimizing compiler, synchronous compilation (§8.1).
- Language: Smalltalk-80 dialect, GST-style class braces, type annotations
  parsed-and-ignored, mixins deferred (§1).

---

## 15. Amendment log — sprint-doc review (2026-07)

The sprint detail docs (`sprints/`) were written against this spec by
independent reviewers; their `SPEC-QUESTION` flags were adjudicated as follows.
Inline amendments already made above are marked *(Amended per …)* at the site
(§2.2 gc_mark bit 44; §4.2 jump base = next_bci; §4.3 PIC ordering/counts;
§4.4 flags nctx/captures_ctx + counter split; §5.1 sp convention + frame slots
FP+5 serial / FP+6 marker; §5.3 rcvr index; §5.4 closure copied[] convention;
§7.6 poison semantics; §8.1 effectiveness check; §8.2 header fields;
§8.5 mono-send `entry` not `verified_entry`; §10 PrimResult::Activated +
`gcStats`/`error:`). Additional pins, resolved in the named sprint doc:

| # | Pin | Where |
|---|-----|-------|
| A1 | Instance-size rule extended per-format (beyond Slots) | s01 §Design |
| A2 | S2 decodes all opcodes, executes the non-send/non-closure subset; `mustBeBoolean` pre-S3 = VM panic with S3 obligation | s02 |
| A3 | Association field order, metaclass naming `#'X class'`, v1 `Class superclass == Object` Δ | s01/s02 |
| A4 | Klass dependency versioning coarsened to a global 24-bit install epoch packed in the IC meta slot (no per-klass slot) | s03 |
| A5 | Full genesis klass skeleton (incl. Behavior, Number, Integer, Large±, collection hierarchy); `Smalltalk` is a VM-layout SystemDictionary | s06 |
| A6 | Class-variable declaration syntax = `<classVars: A B>` class-level pragma | s05 |
| A7 | Scavenge klass-sizing protocol: read-only forward-chase during copy, klass-slot-first during scan (Metaclass 2-cycle) | s07 §A5 |
| A8 | `MACVM_EDEN=<KiB>` env knob; `MACVM_TRACE` channel list is open-ended (`deopt`, `stats`, `count` added) | s07/CONVENTIONS §3 |
| A9 | Full-GC slide is per-space including new gen (ages preserved via side map) — the empty-new-gen alternative noted, not taken | s08 |
| A10 | nmethod literal-pool `oops_do` root hook ships in S10 (first oop-bearing nmethod), S12 keeps integration + gates | s10 |
| A11 | v1 collapses §8.3's separate lowering stage (IR is AArch64-shaped from birth); revisit at S14 | s10 |
| A12 | S10/S11 "≥5×" figures are tripwires (warn <5×, fail <2×), not gates — SPRINTS rule 3 stands | tests_s10/s11 |
| A13 | S11 ships a temporary GC bridge (old-direct alloc while compiled frames live); S12's first commit deletes it | s11/s12 |
| A14 | `brk` imm namespace `#0xDE00–02` for uncommon traps; foreign brk re-raised | s13 |
| A15 | v1 capture rule: all explicit captures via Context (`ncopied` = 0); by-value copies deferred | s05 (matches §5.4 as amended) |
| A16 | Method-source retention via `Smalltalk methodSources` IdentityDictionary (no CompiledMethod layout change); populated by the S5 frontend, `MACVM_KEEP_SOURCE` default on | §16.4 |
| A17 | Embedding API: `VmHandle` (boot/eval/mirrors) + `TranscriptSink` routing of §10 `printOnStdout:` | §16.2–16.3 |
| A18 | Repo becomes a cargo workspace (`macvm` + `gui/` = `macvm-gui`); VM on worker thread, AppKit on main, channel bridge | §16.1 |
| A19 | §16.3 revised: mirrors are **Smalltalk-side** over VM primitive groups R1–R5; HtmlWriter fragment rendering is Smalltalk-side; gui crate = shell + transport only | APPS.md §1–§3, §6 |
| A20 | Eval/accept error contract: compile-service errors return **(message, character position)** for in-editor display; `evaluate:receiver:ifError:` and the deferred `blockToEvaluateFor:` forms pinned | APPS.md §4 |
| A21 | **Design flaw, not just a bug backlog**: §7.6's poison-fill was never implemented (now fixed, S7-11); more importantly, `Handle<T>` is used as a purely function-internal convenience everywhere (one `pub fn` signature total, its own constructor) instead of as API-boundary currency, so the "hold an oop across an allocation" bug this section warns about is structurally regenerated by every new allocating function, not a diligence lapse. **Resolved by a specified revised design (§7.6.1)** adopting the Locus sister VM's handle layer: (1) generation-validated handles → use-after-scope panics at the misuse site; (2) `Handle<T>` as boundary currency for allocating fns; (3) arena-as-precise-root-set stated as invariant. Explicitly does **not** adopt Locus's `newgc-core` collector (already rejected, ref-analysis §5) or its conservative-scan MAGIC tag (MACVM uses precise oop-maps). Scheduled as pass **S7.5**, before S8 | §7.6.1 |
| A22 | `world/*.mst` + `world.list` gain an **optional** SQLite container (`image_store` crate) as an alternative source for §3.2 step 2 — versioned class/method **text** plus a recompile-avoidance bytecode cache (always re-derivable from source, never authoritative). **Not** the S16 snapshot: no oops, no heap pointers, still "boot by compiling source," per §3.2 — a different source *container*, not a different bootstrap model | `docs/IMAGE.md` |
| A23 | S20 (Cocoa + POSIX/BSD guest-language FFI) designed as a non-disruptive side track: two tiers (dynamic `doesNotUnderstand:`-based Cocoa dispatch reusing S11's PIC caching; a direct compiler-primitive path for POSIX calls) over an `Alien`-style byte-array representation, ABI-driven entirely by `cocoa_data` (a sibling repo's already-derived AAPCS64 register classification of the whole Obj-C + curated POSIX surface) rather than re-deriving `@encode`/C-declarator facts in MACVM itself. A working, tested offline generator crate (`ffi_gen`) emits real `.mst` bindings now, forward-declared against a VM-side primitive/JIT trampoline the real S20 sprint still has to build | `docs/FFI.md` |
| A24 | S23 (hand-written ASM methods) designed as a non-disruptive side track: a new `<asm: 'text'>` pragma (string-wrapped to keep ARM64 addressing-mode `[`/`]` out of `.mst`'s bracket-matcher), a leaf-routine-only v1 contract (no allocation/calls/safepoints, so no oop-map obligation) built directly on the register convention `docs/arm64.md` §3 already defines, reusing the existing `Nmethod`/`CodeTable` install path rather than adding a new one. No Strongtalk precedent exists (checked directly against source) — genuinely new for this lineage, unlike A23's FFI work. A working, tested preview tool (`asm_preview`) proves the mechanism via JASM's real text assembler (vendored S9, otherwise unused); the frontend/installer integration is deferred to the real S23 sprint | `docs/ASM.md` |
| A25 | Canvas widget (Smalltalk-drawable HTML5 canvas) designed as a non-disruptive side track: a Canvas view (`gui/src/canvas_render.rs`) plus VM-request/response plumbing (`CanvasCreate`/`CanvasRunDemo`/`CanvasClear` ↔ `CanvasCreated`/`CanvasDraw`/`CanvasCleared`) whose wire format is a JSON array of `[opName, ...args]`, checked against two explicit allowlists (`CANVAS_METHODS`/`CANVAS_PROPERTIES`) before touching `CanvasRenderingContext2D`, rather than blindly indexing into it. Deliberately reuses the existing ordered/batched response channel instead of `LatestSlot` — canvas commands are cumulative (order-dependent), not stale-frame-droppable like a redrawing pixel buffer; "fast channel" is one `VmResponse::CanvasDraw` batch per flush, not a drop-semantics optimization. Precedent checked directly against Strongtalk's real `Canvas.str` (Win32 GDI `bitBlt:`/`palette:`, not transferable — Layer C, replaced by WKWebView) — greenfield, not a port. A working, tested GUI-side implementation (view + plumbing + JS interpreter + "Run Demo"/"Clear") is built and visually verified now; the VM-side `Canvas` class and its `<primitive: N>` bodies are deferred to when S12/S13 compiler work lands | `docs/CANVAS.md` |

---

## 16. Embedding & GUI — the `GuiHost` seam

MACVM's user interface is a recreation of the **Strongtalk live-HTML
programming environment** in a native Cocoa window (WKWebView), developed as a
parallel track — plan of record: [`../gui/PLAN.md`](../gui/PLAN.md) (decisions
D-G1…D-G5), phases G0–G5 in `SPRINTS.md` Phase G. The GUI never reaches into
VM internals; everything crosses one seam. What the *core* must provide:

### 16.1 Crate & threading model
- The repo root becomes a cargo **workspace**: the existing `macvm` package
  (lib + bin) plus member `gui/` = `macvm-gui` (Rust shell via a hand-rolled
  `dlopen`/`objc_msgSend` bridge, not the `objc2` crate ecosystem originally
  sketched here — see `../gui/PLAN.md` D-G2 for why), which depends on the
  `macvm` lib.
- AppKit owns the **main thread**; the VM runs on a dedicated **worker
  thread** (the VM is single-threaded internally, §11 — this adds no VM-level
  concurrency). GUI→VM requests (doits, mirror queries, accepts) are queued
  over a channel and executed between interpreter turns; VM→GUI output
  (transcript, rendered fragments) returns via a channel drained on the main
  thread into `evaluateJavaScript`.

### 16.2 `VmHandle` — the embedding API (Rust) *(REVISED — S21, `src/embed.rs`)*
```rust
pub struct VmHandle { /* owns VmState + world, lives on the VM thread */ }
impl VmHandle {
    pub fn boot(opts: VmOptions, world_dir: &Path) -> Result<VmHandle, VmError>;
        // genesis + world; world_dir explicit (not hardcoded) for testability —
        // see boot's own doc comment
    pub fn eval(&mut self, source: &str) -> Result<String, GuestError>;
        // compile as a doit (S5 REPL machinery), run, answer printString
    pub fn set_transcript(&mut self, sink: Box<dyn TranscriptSink>);
    pub fn mirrors(&mut self) -> Mirrors<'_>;                    // §16.3, NOT YET BUILT
}
pub enum GuestError { Compile(CompileError), NativeFault { sig: i32, pc: u64, far: u64 } }
pub trait TranscriptSink: Send { fn show(&mut self, text: &str); }
```
`Transcript`'s primitive path (§10 `printOnStdout:`) routes through the
`VmState`-held sink; the default sink is stdout, the GUI installs a channel
sink. Guest errors surface as `GuestError` values, never as Rust panics —
`Compile` for a source that failed to lex/parse/codegen, `NativeFault` for a
genuinely recovered SIGSEGV/SIGBUS in ordinary (non-JIT) native code (S20's
`Alien` raw pointer accessors are the only source of these today).

**Thread-safety model (S21, not in the original sketch above):** a
`VmHandle` MUST be driven from a dedicated thread the caller is prepared to
see disappear out from under it. `boot` arms `FatalMode::ExitThread`
(`runtime::vm_state`) so every guest-fatal condition (`error:`, DNU, stack
overflow, heap exhaustion, a genesis-time mmap failure) terminates only the
calling thread (`libc::pthread_exit`, which does not unwind — safe
regardless of any JIT-compiled frames on the stack) rather than the whole
process; the JIT is fully supported in this mode, unrestricted. The caller
must never call `.join()`/`.is_finished()` on that thread's `JoinHandle` —
see `src/embed.rs`'s module doc for why (a `pthread_exit`-terminated thread
never completes `JoinHandle`'s own bookkeeping). See
`src/codecache/deopt_trap.rs`'s `arm_foreign_fault_handler` for the
mechanism, which works regardless of `opts.jit`.

### 16.3 Mirrors — the reflection surface (needed by G3/G4) *(REVISED — A19)*
**Mirrors are Smalltalk-side objects backed by small, dumb VM primitives** —
exactly Strongtalk's architecture (its whole tool suite is image-side over a
primitive floor; survey in `APPS.md`). The VM provides the primitive groups
**R1–R5** pinned in `APPS.md` §3 (structure reads; method reads incl.
referenced-selector scans for find tools; the compile service; heap queries;
debugger/activation access — the first two plus R3 are what G2–G4 need). The
Smalltalk mirror library (`ObjectMirror`, `ClassMirror`, `MethodMirror`, …)
and the tools built on it are Phase W (W2–W4); **rendering fragments is
Smalltalk-side too** (`HtmlWriter`, APPS.md §6) — the gui crate keeps only
the shell and transport. The original Rust-side-queries sketch is withdrawn.

### 16.4 Method-source retention *(Amendment A16)*
The GUI's browsers/editors need method source; CompiledMethod's pinned layout
(§4.4) is **not** changed. Instead the frontend records source in a VM-known
world-side registry: `Smalltalk methodSources`, an IdentityDictionary
(CompiledMethod → String). GC-safe by construction (it's an ordinary heap
object; identity hashes are stable across compaction — the S8 gate proves it).
Controlled by `MACVM_KEEP_SOURCE` (default **on**; `=0` for memory-lean runs).

### 16.5 Redefinition caveat
Interpreter-tier method redefinition is safe from S3 (IC self-heal + lookup
flush). **Compiled-tier** redefinition is now safe via S13's dependency
invalidation (landed), so the GUI's accept path (G4) runs with the JIT enabled
(default on); no `MACVM_JIT=off` workaround is required.
