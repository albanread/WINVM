//! Class-definition execution and top-level `do_it` execution
//! (`sprint_s05_detail.md` §Algorithms "Class-definition execution").
//! `classdef.rs` is the only module that creates/reopens klasses from
//! source (SPEC §3.2 step 2's `subclass:` equivalent, S5's compile-time
//! version — the bytecode-level `subclass:` primitive is a later sprint).

use crate::memory::alloc;
use crate::memory::handles::HandleScope;
use crate::oops::wrappers::{ArrayOop, KlassOop, MemOop, SymbolOop};
use crate::oops::{Format, Oop};
use crate::runtime::vm_state::VmState;

use super::ast::{ClassDefNode, Indexable, TopItem};
use super::codegen::find_class_var;
use super::lexer::Span;
use super::CompileError;

fn err(span: Span, msg: impl Into<String>) -> CompileError {
    CompileError {
        path: None,
        span,
        msg: msg.into(),
        eof: false,
    }
}

fn create_klass(vm: &mut VmState, super_klass: KlassOop, node: &ClassDefNode) -> KlassOop {
    let format = match node.indexable {
        Some(Indexable::Oops) => Format::IndexableOops,
        Some(Indexable::Bytes) => Format::IndexableBytes,
        None => super_klass.format(),
    };
    let untagged = match node.indexable {
        Some(Indexable::Oops) => false,
        Some(Indexable::Bytes) => true,
        None => super_klass.has_untagged_contents(),
    };
    let nis = super_klass.non_indexable_size() + node.inst_vars.len();
    let klass = vm
        .universe
        .new_klass(super_klass, &node.name, format, untagged, nis);

    // `klass` is fresh eden garbage — not yet linked into any global, so
    // not reachable from any GC root — until `install_class_def` declares
    // it. Every allocation below can move it, so it's handle-protected for
    // the rest of this function (and threaded into `append_class_var` the
    // same way).
    let scope = HandleScope::enter(vm);
    let klass_h = scope.handle(vm, klass);

    // `vm.universe.array_klass` is re-read before EACH allocation, never
    // cached across one: a universe klass field is kept current by the
    // root scan, but a Rust local copy of it is not — under GC_STRESS the
    // first alloc below moves the klass and a cached local hands the
    // second alloc a one-cycle-stale address (S7-10: this exact line was
    // the double-copy panic's origin).
    // Interned directly into `ivn` rather than collected into a `Vec<Oop>`
    // first: `Universe::intern` never allocates through the scavenge-aware
    // choke point (SPEC §7.2's genesis-era raw eden path), so nothing here
    // needs its own handle as long as each symbol is consumed before the
    // next allocating call.
    let array_klass = vm.universe.array_klass;
    let ivn = alloc::alloc_indexable_oops(vm, array_klass, node.inst_vars.len());
    for (i, n) in node.inst_vars.iter().enumerate() {
        let s = vm.universe.intern(n.as_bytes());
        ivn.at_put(i, s.oop());
    }
    klass_h.get(vm).set_inst_var_names(ivn.oop());

    let array_klass = vm.universe.array_klass;
    let cvs = alloc::alloc_indexable_oops(vm, array_klass, 0);
    klass_h.get(vm).set_class_vars(cvs.oop());
    for name in &node.class_vars {
        let sym = vm.universe.intern(name.as_bytes());
        append_class_var(vm, klass_h.get(vm), sym);
    }
    klass_h.get(vm)
}

fn append_class_var(vm: &mut VmState, klass: KlassOop, sym: SymbolOop) {
    if find_class_var(klass, sym.oop()).is_some() {
        return; // re-declaring an existing class var is a no-op
    }
    let scope = HandleScope::enter(vm);
    // `klass` may itself still be unlinked eden garbage (see
    // `create_klass`'s caller-side handle) and `sym`, though reachable via
    // the symbol table root, is a separate unrooted copy in this bare
    // parameter — both need protecting across the allocations below. The
    // universe fields (`association_klass`/`array_klass`/`nil_obj`) are
    // instead re-read fresh before each use — a Rust local copy of a
    // root-scanned field is NOT kept current by the root scan (S7-10).
    let klass_h = scope.handle(vm, klass);
    let sym_h = scope.handle(vm, sym);

    let association_klass = vm.universe.association_klass;
    let assoc = alloc::alloc_slots(vm, association_klass);
    assoc.set_body_oop(0, sym_h.get(vm).oop());
    assoc.set_body_oop(1, vm.universe.nil_obj);
    let assoc_h = scope.handle(vm, assoc);

    let old_len = ArrayOop::try_from(klass_h.get(vm).class_vars())
        .map(|a| a.len())
        .unwrap_or(0);
    let array_klass = vm.universe.array_klass;
    let new_arr = alloc::alloc_indexable_oops(vm, array_klass, old_len + 1);
    if let Some(old) = ArrayOop::try_from(klass_h.get(vm).class_vars()) {
        for i in 0..old_len {
            new_arr.at_put(i, old.at(i));
        }
    }
    new_arr.at_put(old_len, assoc_h.get(vm).oop());
    let klass = klass_h.get(vm);
    klass.set_class_vars(new_arr.oop());
    // On a reopen `klass` is old while `new_arr` is young (S7-10).
    crate::memory::store::post_write_barrier(vm, klass.as_mem());
}

fn reopen_klass(
    vm: &mut VmState,
    klass: KlassOop,
    declared_super: Oop,
    node: &ClassDefNode,
) -> Result<(), CompileError> {
    if klass.superclass().raw() != declared_super.raw() {
        return Err(err(
            node.span,
            format!(
                "cannot change shape of existing class {}: superclass differs",
                node.name
            ),
        ));
    }
    if !node.inst_vars.is_empty() || node.indexable.is_some() {
        return Err(err(
            node.span,
            format!("cannot change shape of existing class {}", node.name),
        ));
    }
    // `klass` is reachable from the globals root, but this loop's own
    // local copy isn't updated when a scavenge inside `append_class_var`
    // moves it — a later iteration would call in with a stale address
    // without re-reading through a handle each time (same hazard
    // `create_klass`'s loop has).
    let scope = HandleScope::enter(vm);
    let klass_h = scope.handle(vm, klass);
    for name in &node.class_vars {
        let sym = vm.universe.intern(name.as_bytes());
        append_class_var(vm, klass_h.get(vm), sym);
    }
    Ok(())
}

/// SPEC §3.2 step 2 / `sprint_s05_detail.md` §Algorithms "Class-definition
/// execution": create-or-reopen `node.name`, then compile+install every
/// method (instance-side into `klass`'s own `MethodDictionary`, class-side
/// into its metaclass's — `runtime::lookup::install_method` already flushes
/// the lookup cache and bumps `ic_epoch`, SPEC §6.2).
/// Resolves `name` as an already-bound klass global. `"nil"` is accepted
/// specially — `Object`'s REAL superclass is the nil oop itself (it is the
/// root of the hierarchy, not a klass), so a reopen naming its true
/// superclass must be able to say `nil subclass: Object [...]` and have
/// that compare equal to `Object.superclass()` (which is nil, not a
/// `KlassOop`). Only meaningful for reopen's superclass-match check —
/// `create_klass` never receives a nil target (creating a NEW root class
/// alongside Object is out of scope for source-level `subclass:`).
fn resolve_super_target(vm: &mut VmState, node: &ClassDefNode) -> Result<Oop, CompileError> {
    if node.superclass == "nil" {
        return Ok(vm.universe.nil_obj);
    }
    let super_sym = vm.universe.intern(node.superclass.as_bytes());
    let super_assoc = crate::runtime::globals::global_lookup(vm, super_sym).ok_or_else(|| {
        err(
            node.span,
            format!("superclass '{}' not found", node.superclass),
        )
    })?;
    Ok(MemOop::try_from(super_assoc)
        .expect("global association is a mem oop")
        .body_oop(1))
}

pub fn install_class_def(vm: &mut VmState, node: &mut ClassDefNode) -> Result<(), CompileError> {
    // The scope is entered — and `name_sym` handled — BEFORE the
    // create/reopen below: both arms allocate heavily, and a bare
    // `name_sym` held across them can end up aliasing a DIFFERENT symbol
    // that later gets interned into the recycled space (S7-10's nastiest
    // GC_STRESS find: `OrderedCollection`'s stale name symbol aliased the
    // freshly interned `"firstIndex"` instvar name, so the class was
    // silently global-declared under the wrong key — every pointer valid,
    // nothing for the heap verifier to object to, pure semantic
    // corruption).
    let scope = HandleScope::enter(vm);
    let name_sym = vm.universe.intern(node.name.as_bytes());
    let name_sym_h = scope.handle(vm, name_sym);

    let klass = match crate::runtime::globals::global_lookup(vm, name_sym) {
        None => {
            let super_target = resolve_super_target(vm, node)?;
            let super_klass = KlassOop::try_from(super_target)
                .ok_or_else(|| err(node.span, format!("'{}' is not a class", node.superclass)))?;
            create_klass(vm, super_klass, node)
        }
        Some(assoc) => {
            let val = MemOop::try_from(assoc)
                .expect("global association is a mem oop")
                .body_oop(1);
            let existing = KlassOop::try_from(val).ok_or_else(|| {
                err(
                    node.span,
                    format!("'{}' is already bound to a non-class value", node.name),
                )
            })?;
            // `reopen_klass` allocates (class-var appends); `existing`
            // must ride in a handle across it.
            let existing_h = scope.handle(vm, existing);
            let super_target = resolve_super_target(vm, node)?;
            reopen_klass(vm, existing, super_target, node)?;
            existing_h.get(vm)
        }
    };

    // `klass` (freshly created, in the `None` arm) isn't linked into the
    // globals until `global_declare` below runs, and both that call and
    // every `compile_method` in the loop allocate — protect it for the
    // rest of this function.
    let klass_h = scope.handle(vm, klass);

    let assoc = crate::runtime::globals::global_declare(vm, name_sym_h.get(vm));
    let assoc_mem = MemOop::try_from(assoc).expect("global association is a mem oop");
    // Barriered store: on a REOPEN the association is long-lived (old gen)
    // while the klass value may still be young (S7-10).
    crate::memory::store::store(vm, assoc_mem, 1, klass_h.get(vm).oop());

    for method in &mut node.methods {
        let compiled =
            super::codegen::compile_method(vm, klass_h.get(vm), method.class_side, method)?;
        let sel = vm.universe.intern(method.pattern_selector.as_bytes());
        let target = if method.class_side {
            klass_h.get(vm).klass()
        } else {
            klass_h.get(vm)
        };
        crate::runtime::lookup::install_method(vm, target, sel, compiled);
    }
    Ok(())
}

/// A top-level `do_it` (SPEC §3.2 step 3's REPL/script path): compiles and
/// executes `stmt` as an anonymous `#doIt` method (receiver `nil`, holder
/// `UndefinedObject`'s klass) and returns its result. `frontend/` never
/// calls `interpreter/` directly — this goes through `runtime::execute_doit`.
pub fn execute_do_it(vm: &mut VmState, stmt: super::ast::Expr) -> Result<Oop, CompileError> {
    let span = stmt.span();
    let holder = vm.universe.undefined_object_klass;
    let m = super::codegen::compile_doit(vm, holder, stmt)?;
    crate::runtime::execute_doit(vm, m).map_err(|e| err(span, e.msg))
}

/// Executes one parsed top-level item — a class definition (installed, no
/// value) or a `do_it` (its result). Shared by `world::load_file` and the
/// REPL.
pub fn execute_top_item(vm: &mut VmState, item: TopItem) -> Result<Option<Oop>, CompileError> {
    match item {
        TopItem::ClassDef(mut c) => {
            install_class_def(vm, &mut c)?;
            Ok(None)
        }
        TopItem::DoIt(stmt) => {
            let r = execute_do_it(vm, stmt)?;
            // Cocoa bridge pool hygiene (docs/cocoa_bridge_design.md §3.1):
            // a doit boundary is the natural quiescent point to drain +
            // renew this thread's bottom autorelease pool. A cheap
            // thread-local no-op on threads that never touched Cocoa.
            #[cfg(target_os = "macos")]
            crate::runtime::objc_bridge::drain_pool_at_doit_boundary();
            Ok(Some(r))
        }
    }
}
