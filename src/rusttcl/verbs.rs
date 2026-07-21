//! The MACVM introspection verbs registered on top of `rust_tcl::Registry
//! ::with_core()`'s ordinary Tcl verb set. Each closure's first line is
//! `bridge::active_ctx()` — see `bridge.rs` for why that's the only way
//! to reach the live `VmState`.
//!
//! There's no way to enumerate an already-built `Registry`'s verbs from
//! the outside (`registry.rs`'s `names`/`verbs` fields are private, by
//! design — see its own module doc on why this tree stays byte-identical
//! to upstream). So `TABLE` below is this file's own record of what it
//! registers, used both to build the `Registry` and to answer `help`.

use crate::rusttcl::bridge;
use crate::vendor::rust_tcl::{Arity, Error as TclError, Registry, Result as TclResult, Value, Vm};

struct VerbDoc {
    name: &'static str,
    usage: &'static str,
    help: &'static str,
}

const TABLE: &[VerbDoc] = &[
    VerbDoc {
        name: "disasm",
        usage: "disasm <Class> <selector>",
        help: "Disassemble one compiled method's bytecode.",
    },
    VerbDoc {
        name: "methods",
        usage: "methods <Class>",
        help: "List a class's own method dictionary: selector, argc, ntemps, primitive.",
    },
    VerbDoc {
        name: "nmethods",
        usage: "nmethods",
        help: "List every nmethod in the JIT code table: id, state, version, key klass>>selector, trap count, frame slots, code size.",
    },
    VerbDoc {
        name: "ic",
        usage: "ic <Class> <selector>",
        help: "Dump every send site's inline-cache state in one method: bci, selector, Empty/Mono/Poly/Mega, and the resolved klass(es).",
    },
    VerbDoc {
        name: "stats",
        usage: "stats",
        help: "Print the full VM counter dump on demand (same counters as MACVM_TRACE=stats' exit-time dump).",
    },
    VerbDoc {
        name: "trace",
        usage: "trace [channel] [on|off]",
        help: "No args: list enabled MACVM_TRACE channels. One arg: report whether it's on. Two args: enable/disable it live.",
    },
    VerbDoc {
        name: "flag",
        usage: "flag [name] [value]",
        help: "No args: list every VM flag and its current value. One arg: report one flag. Two args: set it live — same grammar as its MACVM_* env var. Flags: jit (off|threshold=N), gc_stress (0|1|full|full:N), deopt_stress (0|N), dbg_oop (0x<hex>|off).",
    },
    VerbDoc {
        name: "load",
        usage: "load <file.mst>",
        help: "Compile and run a .mst file into the current VM (its classes become visible to disasm/methods/ic/nmethods afterward).",
    },
    VerbDoc {
        name: "dbg",
        usage: "dbg [on|off]",
        help: "No args: report whether the HALT debugger is armed. on/off: arm/disarm it live (docs/DEBUGGER.md DBG1). While armed, breakpoints, `error:`, and DNU halt into the (halt) command loop instead of dying.",
    },
    VerbDoc {
        name: "bp",
        usage: "bp <Class> <selector> <bci>",
        help: "Set a HALT breakpoint (side-table, method pinned to tier-0; existing nmethods invalidated through the redefinition path). Use \"Class class\" for the metaclass side. Arms dbg.",
    },
    VerbDoc {
        name: "bp-clear",
        usage: "bp-clear <Class> <selector> <bci>",
        help: "Clear one breakpoint; the method's tier-up eligibility is restored once its last breakpoint is gone.",
    },
    VerbDoc {
        name: "bp-list",
        usage: "bp-list",
        help: "List every live breakpoint as Class>>selector @bci.",
    },
    VerbDoc {
        name: "ring",
        usage: "ring",
        help: "Dump the PROBE recent-history ring (DBG0): the last N compile / deopt / invalidate events, oldest first — the crash dossier's step 9, on demand.",
    },
    VerbDoc {
        name: "disasm-native",
        usage: "disasm-native <Class> <selector>",
        help: "Disassemble a compiled nmethod's MACHINE CODE (DBG3): one line per instruction, annotated with entry/verified_entry, IC send sites, safepoints, deopt traps (int3 + imm16, shown by name), and resolved literal-pool loads with well-known oops tagged (=true/=false/=nil). Host ISA: AArch64 or x86-64. Requires the method to have compiled (see nmethods). Contrast `disasm`, which shows bytecode.",
    },
    VerbDoc {
        name: "pin",
        usage: "pin <Class> <selector>",
        help: "Force a method to run interpreted (pin to tier-0, invalidate any nmethod) WITHOUT halting — the differential-diagnosis lever: 'does interpreting THIS method change the result?'. The fastest way to localize a wrong-value-from-compiled-code bug.",
    },
    VerbDoc {
        name: "unpin",
        usage: "unpin <Class> <selector>",
        help: "Restore tier-up eligibility for a `pin`-ed method.",
    },
    VerbDoc {
        name: "gui",
        usage: "gui connect ?port? | ping | eval <src> | doit <src> | view <name> | snap <path> | sleep <ms>",
        help: "Drive a running macvm-cocoa over its MACVM_COCOA_CTL control channel: run doits on its UI worker, switch views, capture window PNGs.",
    },
    VerbDoc {
        name: "help",
        usage: "help [verb]",
        help: "List every RUSTTCL verb, or show one verb's full help text.",
    },
    VerbDoc {
        name: "quit",
        usage: "quit | exit",
        help: "End the RUSTTCL session.",
    },
];

pub fn register_macvm_verbs(registry: &mut Registry) {
    registry.register("disasm", Arity::exact(2), verb_disasm);
    registry.register("methods", Arity::exact(1), verb_methods);
    registry.register("nmethods", Arity::exact(0), verb_nmethods);
    registry.register("ic", Arity::exact(2), verb_ic);
    registry.register("stats", Arity::exact(0), verb_stats);
    registry.register("trace", Arity::range(0, 2), verb_trace);
    registry.register("flag", Arity::range(0, 2), verb_flag);
    registry.register("load", Arity::exact(1), verb_load);
    registry.register("dbg", Arity::range(0, 1), verb_dbg);
    registry.register("bp", Arity::exact(3), verb_bp);
    registry.register("bp-clear", Arity::exact(3), verb_bp_clear);
    registry.register("bp-list", Arity::exact(0), verb_bp_list);
    registry.register("ring", Arity::exact(0), verb_ring);
    registry.register("disasm-native", Arity::exact(2), verb_disasm_native);
    registry.register("pin", Arity::exact(2), verb_pin);
    registry.register("unpin", Arity::exact(2), verb_unpin);
    registry.register("gui", Arity::range(1, 2), verb_gui);
    registry.register("help", Arity::range(0, 1), verb_help);
    registry.register("quit", Arity::exact(0), verb_quit);
    registry.register("exit", Arity::exact(0), verb_quit);
}

fn verb_disasm(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let klass = super::resolve_klass(ctx, args[0].as_str()).map_err(TclError::runtime)?;
    let method = super::resolve_method(ctx, klass, args[1].as_str()).map_err(TclError::runtime)?;
    Ok(Value::new(crate::bytecode::disassemble(
        &ctx.vm.universe,
        method,
    )))
}

fn verb_methods(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let klass = super::resolve_klass(ctx, args[0].as_str()).map_err(TclError::runtime)?;
    let dict =
        crate::oops::method_dict::MethodDictOop::try_from(klass.methods()).ok_or_else(|| {
            TclError::runtime(format!("{} has no method dictionary", args[0].as_str()))
        })?;
    let mut rows: Vec<String> = Vec::new();
    dict.each_pair(&ctx.vm, |sel, m| {
        rows.push(format!(
            "{} argc={} ntemps={} primitive={}",
            sel.as_string(),
            m.argc(),
            m.ntemps(),
            m.primitive()
        ));
    });
    rows.sort();
    if rows.is_empty() {
        rows.push("(no methods)".to_string());
    }
    Ok(Value::new(rows.join("\n")))
}

fn verb_nmethods(_vm: &mut Vm<'_>, _args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let mut rows: Vec<String> = Vec::new();
    for nm in ctx.vm.code_table.iter_all() {
        let klass_name = crate::memory::print_oop(&ctx.vm.universe, nm.key_klass.name());
        rows.push(format!(
            "nm={} state={:?} v{} {klass_name}>>{} trap_count={} frame_slots={} code_bytes={}",
            nm.id.0,
            nm.state,
            nm.version,
            nm.key_selector.as_string(),
            nm.trap_count,
            nm.frame_slots,
            nm.code.len
        ));
    }
    if rows.is_empty() {
        rows.push("(no nmethods compiled yet)".to_string());
    }
    Ok(Value::new(rows.join("\n")))
}

fn verb_ic(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let klass = super::resolve_klass(ctx, args[0].as_str()).map_err(TclError::runtime)?;
    let method = super::resolve_method(ctx, klass, args[1].as_str()).map_err(TclError::runtime)?;

    let mut rows: Vec<String> = Vec::new();
    let mut bci = 0usize;
    while bci < method.bytecode_len() {
        let (instr, next) = crate::bytecode::decode_at(method, bci);
        if let crate::bytecode::Instr::Send { ic, super_ } = instr {
            let icv = crate::interpreter::ic::InterpreterIc::at(method, ic);
            let sel_str = icv.selector().as_string();
            let state = crate::interpreter::ic::ic_state(method, ic);
            let mut row = format!("@{bci} ic={ic} super={super_} sel={sel_str} state={state:?}");
            match state {
                crate::interpreter::ic::IcState::Mono => {
                    if let Some(k) = crate::oops::wrappers::KlassOop::try_from(icv.guard()) {
                        row.push_str(&format!(
                            " klass={}",
                            crate::memory::print_oop(&ctx.vm.universe, k.name())
                        ));
                    }
                }
                crate::interpreter::ic::IcState::Poly(n) => {
                    if let Some(pairs) = crate::oops::wrappers::ArrayOop::try_from(icv.target()) {
                        for i in 0..n as usize {
                            if let Some(k) =
                                crate::oops::wrappers::KlassOop::try_from(pairs.at(2 * i))
                            {
                                row.push_str(&format!(
                                    " {}",
                                    crate::memory::print_oop(&ctx.vm.universe, k.name())
                                ));
                            }
                        }
                    }
                }
                crate::interpreter::ic::IcState::Empty | crate::interpreter::ic::IcState::Mega => {}
            }
            rows.push(row);
        }
        bci = next;
    }
    if rows.is_empty() {
        rows.push("(no send sites)".to_string());
    }
    Ok(Value::new(rows.join("\n")))
}

fn verb_stats(_vm: &mut Vm<'_>, _args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    Ok(Value::new(crate::runtime::vm_state::format_vm_stats(
        &ctx.vm,
    )))
}

fn verb_trace(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    match args {
        [] => {
            let channels = ctx.vm.options.trace.list();
            if channels.is_empty() {
                Ok(Value::new("(no trace channels enabled)"))
            } else {
                Ok(Value::new(channels.join(" ")))
            }
        }
        [channel] => {
            let on = ctx.vm.options.trace.is_enabled(channel.as_str());
            Ok(Value::new(if on { "on" } else { "off" }))
        }
        [channel, setting] => match setting.as_str() {
            "on" => {
                ctx.vm.options.trace.enable(channel.as_str());
                Ok(Value::empty())
            }
            "off" => {
                ctx.vm.options.trace.disable(channel.as_str());
                Ok(Value::empty())
            }
            other => Err(TclError::runtime(format!(
                "trace: expected on|off, got {other:?}"
            ))),
        },
        _ => unreachable!("Arity::range(0, 2) already rejects more than 2 args"),
    }
}

/// Every flag `flag` knows how to get/set. One name per operationally
/// live-tunable `VmOptions`/`VmState` field — deliberately NOT
/// `heap_mib`/`eden_kb` (sized once at `VmState::new()`; mutating the
/// field after construction wouldn't resize anything already allocated).
const FLAG_NAMES: &[&str] = &["jit", "gc_stress", "deopt_stress", "dbg_oop"];

fn verb_flag(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    match args {
        [] => {
            let rows: Vec<String> = FLAG_NAMES
                .iter()
                .map(|n| {
                    format!(
                        "{n}={}",
                        flag_get(ctx, n).expect("name drawn from FLAG_NAMES")
                    )
                })
                .collect();
            Ok(Value::new(rows.join(" ")))
        }
        [name] => flag_get(ctx, name.as_str()).map(Value::new).ok_or_else(|| {
            TclError::runtime(format!(
                "unknown flag {:?} — try `flag` for the list",
                name.as_str()
            ))
        }),
        [name, value] => {
            flag_set(ctx, name.as_str(), value.as_str())?;
            Ok(Value::empty())
        }
        _ => unreachable!("Arity::range(0, 2) already rejects more than 2 args"),
    }
}

fn flag_get(ctx: &mut super::RusttclCtx, name: &str) -> Option<String> {
    Some(match name {
        "jit" => match ctx.vm.options.jit {
            crate::runtime::JitMode::Off => "off".to_string(),
            crate::runtime::JitMode::Threshold(n) => format!("threshold={n}"),
        },
        "gc_stress" => match (
            ctx.vm.options.gc_stress,
            ctx.vm.options.gc_stress_full_period,
        ) {
            (true, _) => "1".to_string(),
            (false, Some(n)) => format!("full:{n}"),
            (false, None) => "0".to_string(),
        },
        "deopt_stress" => {
            if ctx.vm.deopt_stress {
                ctx.vm.stress_period.to_string()
            } else {
                "0".to_string()
            }
        }
        "dbg_oop" => match ctx.vm.dbg_oop {
            Some(addr) => format!("{addr:#x}"),
            None => "off".to_string(),
        },
        _ => return None,
    })
}

/// Same grammar as each flag's `MACVM_*` env var (`VmOptions::parse_jit`/
/// `parse_gc_stress`/`VmState::parse_deopt_stress` — the exact parse
/// functions `from_env`/`new` call, `pub(crate)` for this reuse) — so
/// `flag jit threshold=1` reads identically to `MACVM_JIT=threshold=1`.
fn flag_set(ctx: &mut super::RusttclCtx, name: &str, value: &str) -> TclResult<()> {
    match name {
        "jit" => {
            ctx.vm.options.jit = crate::runtime::vm_state::VmOptions::parse_jit(Some(value));
            Ok(())
        }
        "gc_stress" => {
            let (on, period) = crate::runtime::vm_state::VmOptions::parse_gc_stress(Some(value));
            ctx.vm.options.gc_stress = on;
            ctx.vm.options.gc_stress_full_period = period;
            Ok(())
        }
        "deopt_stress" => {
            let (on, period) = crate::runtime::vm_state::VmState::parse_deopt_stress(Some(value));
            ctx.vm.deopt_stress = on;
            ctx.vm.stress_period = period;
            ctx.vm.stress_countdown = period; // re-arm the live counter too, not just the config
            Ok(())
        }
        "dbg_oop" => {
            if value == "off" || value == "none" {
                ctx.vm.dbg_oop = None;
            } else {
                let s = value.trim();
                let s = s.strip_prefix("0x").unwrap_or(s);
                let addr = usize::from_str_radix(s, 16).map_err(|_| {
                    TclError::runtime(format!("dbg_oop: not a hex address: {value:?}"))
                })?;
                ctx.vm.dbg_oop = Some(addr);
            }
            Ok(())
        }
        other => Err(TclError::runtime(format!(
            "unknown flag {other:?} — try `flag` for the list"
        ))),
    }
}

fn verb_load(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let path = std::path::PathBuf::from(args[0].as_str());
    crate::frontend::world::load_file(&mut ctx.vm, &path)
        .map_err(|e| TclError::runtime(e.to_string()))?;
    Ok(Value::new(format!("loaded {}", path.display())))
}

fn verb_help(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    match args {
        [] => {
            let mut out = String::from("RUSTTCL verbs (`help <verb>` for detail; core Tcl verbs like set/if/while/proc are also available):\n");
            let mut seen = std::collections::HashSet::new();
            for v in TABLE {
                if !seen.insert(v.usage) {
                    continue;
                }
                out.push_str(&format!("  {:<28} {}\n", v.usage, v.help));
            }
            Ok(Value::new(out.trim_end().to_string()))
        }
        [name] => {
            // "exit" is `quit`'s alias (see `register_macvm_verbs`) and
            // isn't its own `TABLE` row — its usage text already reads
            // "quit | exit", so look that entry up under either name.
            let key = if name.as_str() == "exit" {
                "quit"
            } else {
                name.as_str()
            };
            match TABLE.iter().find(|v| v.name == key) {
                Some(v) => Ok(Value::new(format!("{}\n  {}", v.usage, v.help))),
                None => Err(TclError::runtime(format!(
                    "unknown verb {:?} — `help` lists them all",
                    name.as_str()
                ))),
            }
        }
        _ => unreachable!("Arity::range(0, 1) already rejects more than 1 arg"),
    }
}

/// `gui` — drive a RUNNING `macvm-cocoa` over its control channel
/// (`MACVM_COCOA_CTL=<port>`), so on-screen states are scriptable and
/// snapshot-inspectable from a Tcl session instead of needing a human at the
/// window. Subcommands:
///   gui connect ?port?   — connect (default 7644)
///   gui ping             — round-trip check
///   gui rebuild          — request a UI-worker rebuild-in-place (CG9)
///   gui eval <src>       — evaluate Smalltalk on the app's UI worker, answer printString
///   gui doit <src>       — execute Smalltalk on the app's UI worker
///   gui view <name>      — switch the app's content view (sugar for a doit)
///   gui snap <path>      — capture the window's client area to a PNG
///   gui sleep <ms>       — pause app-side (lets async replies land)
fn verb_gui(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let sub = args[0].as_str();
    let arg = args.get(1).map(|v| v.as_str().to_string());
    match sub {
        "connect" => {
            let port: u16 = arg
                .as_deref()
                .unwrap_or("7644")
                .trim()
                .parse()
                .map_err(|_| TclError::runtime("gui connect: port must be a number"))?;
            let conn = std::net::TcpStream::connect(("127.0.0.1", port))
                .map_err(|e| TclError::runtime(format!("gui connect 127.0.0.1:{port}: {e}")))?;
            ctx.gui_conn = Some(conn);
            Ok(Value::new(format!("connected 127.0.0.1:{port}")))
        }
        "ping" => gui_request(ctx, "ping").map(Value::new),
        "rebuild" => gui_request(ctx, "rebuild").map(Value::new),
        "game" => {
            let entry = arg.ok_or_else(|| TclError::runtime("usage: gui game <entry doit>"))?;
            gui_request(ctx, &format!("game {entry}")).map(Value::new)
        }
        "stopgame" => gui_request(ctx, "stopgame").map(Value::new),
        "gameclose" => gui_request(ctx, "gameclose").map(Value::new),
        "eval" => {
            let src = arg.ok_or_else(|| TclError::runtime("usage: gui eval <smalltalk>"))?;
            gui_request(ctx, &format!("eval {src}")).map(Value::new)
        }
        "doit" => {
            let src = arg.ok_or_else(|| TclError::runtime("usage: gui doit <smalltalk>"))?;
            gui_request(ctx, &format!("doit {src}")).map(Value::new)
        }
        "view" => {
            let name = arg.ok_or_else(|| TclError::runtime("usage: gui view <name>"))?;
            gui_request(ctx, &format!("doit CocoaUI switchToView: #{name}.")).map(Value::new)
        }
        "snap" => {
            let path = arg.ok_or_else(|| TclError::runtime("usage: gui snap <path>"))?;
            gui_request(ctx, &format!("snap {path}")).map(Value::new)
        }
        "sleep" => {
            let ms = arg.ok_or_else(|| TclError::runtime("usage: gui sleep <ms>"))?;
            gui_request(ctx, &format!("sleep {ms}")).map(Value::new)
        }
        other => Err(TclError::runtime(format!(
            "gui: unknown subcommand '{other}' (connect/ping/rebuild/game/stopgame/gameclose/eval/doit/view/snap/sleep)"
        ))),
    }
}

/// One framed request/reply on the gui connection (`<len>\n<bytes>` both
/// ways, matching `cocoa_gui/src/control.rs`). `OK`-prefixed replies answer
/// their payload; `ERR` becomes a Tcl error.
fn gui_request(ctx: &mut super::RusttclCtx, cmd: &str) -> TclResult<String> {
    use std::io::{Read, Write};
    let conn = ctx
        .gui_conn
        .as_mut()
        .ok_or_else(|| TclError::runtime("gui: not connected — run `gui connect ?port?`"))?;
    let io_err = |e: std::io::Error| TclError::runtime(format!("gui: connection error: {e}"));
    conn.write_all(format!("{}\n", cmd.len()).as_bytes())
        .map_err(io_err)?;
    conn.write_all(cmd.as_bytes()).map_err(io_err)?;
    conn.flush().map_err(io_err)?;
    let mut len_line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = conn.read(&mut byte).map_err(io_err)?;
        if n == 0 {
            return Err(TclError::runtime("gui: the app closed the connection"));
        }
        if byte[0] == b'\n' {
            break;
        }
        len_line.push(byte[0]);
    }
    let len: usize = String::from_utf8_lossy(&len_line)
        .trim()
        .parse()
        .map_err(|_| TclError::runtime("gui: bad reply frame"))?;
    let mut buf = vec![0u8; len];
    conn.read_exact(&mut buf).map_err(io_err)?;
    let reply = String::from_utf8_lossy(&buf).into_owned();
    if let Some(rest) = reply.strip_prefix("OK") {
        Ok(rest.trim_start().to_string())
    } else {
        Err(TclError::runtime(format!("gui: {reply}")))
    }
}

fn verb_quit(_vm: &mut Vm<'_>, _args: &[Value]) -> TclResult<Value> {
    bridge::active_ctx().quit = true;
    Ok(Value::empty())
}

// ── DBG1 (docs/DEBUGGER.md): the debugger's TCL surface ─────────────────

fn verb_dbg(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    match args {
        [] => Ok(Value::new(if ctx.vm.debug.active { "on" } else { "off" })),
        [setting] => match setting.as_str() {
            "on" => {
                ctx.vm.debug.active = true;
                Ok(Value::empty())
            }
            "off" => {
                ctx.vm.debug.active = false;
                Ok(Value::empty())
            }
            other => Err(TclError::runtime(format!(
                "dbg: expected on|off, got {other:?}"
            ))),
        },
        _ => unreachable!("Arity::range(0, 1) already rejects more than 1 arg"),
    }
}

fn verb_bp(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let bci: u16 = args[2].as_str().parse().map_err(|_| {
        TclError::runtime(format!("bp: bci must be a number: {:?}", args[2].as_str()))
    })?;
    crate::runtime::debug::set_breakpoint_by_name(
        &mut ctx.vm,
        args[0].as_str(),
        args[1].as_str(),
        bci,
    )
    .map(Value::new)
    .map_err(TclError::runtime)
}

fn verb_bp_clear(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let bci: u16 = args[2].as_str().parse().map_err(|_| {
        TclError::runtime(format!(
            "bp-clear: bci must be a number: {:?}",
            args[2].as_str()
        ))
    })?;
    crate::runtime::debug::clear_breakpoint_by_name(
        &mut ctx.vm,
        args[0].as_str(),
        args[1].as_str(),
        bci,
    )
    .map(Value::new)
    .map_err(TclError::runtime)
}

fn verb_bp_list(_vm: &mut Vm<'_>, _args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let rows = ctx.vm.debug.list();
    if rows.is_empty() {
        Ok(Value::new("(no breakpoints)"))
    } else {
        Ok(Value::new(rows.join("\n")))
    }
}

fn verb_disasm_native(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    let klass = super::resolve_klass(ctx, args[0].as_str()).map_err(TclError::runtime)?;
    let sel = ctx.vm.universe.intern(args[1].as_str().as_bytes());
    let id = ctx.vm.code_table.lookup(klass, sel).ok_or_else(|| {
        TclError::runtime(format!(
            "{}>>{} has no compiled nmethod (see `nmethods`)",
            args[0].as_str(),
            args[1].as_str()
        ))
    })?;
    let nm = ctx
        .vm
        .code_table
        .get(id)
        .expect("lookup returned a live id");
    // ic-site + safepoint offsets, for annotating the listing lines.
    let ic_offs: std::collections::HashSet<usize> =
        nm.ic_sites.iter().map(|s| s.off as usize).collect();
    let sp_offs: std::collections::HashSet<usize> =
        nm.pcdescs.iter().map(|p| p.pc_off as usize).collect();
    // Disassemble the CODE region only (up to the literal pool).
    let code = &nm.code.as_bytes()[..nm.literal_off as usize];
    let mut rows: Vec<String> = vec![format!(
        "nmethod #{} v{} entry+{:#x} verified+{:#x} ({} code bytes, {} pool)",
        id.0,
        nm.version,
        nm.entry_off,
        nm.verified_entry_off,
        nm.literal_off,
        nm.code.len - nm.literal_off as usize
    )];
    // Whole-blob bytes: a pool load's target lands PAST literal_off.
    let all = nm.code.as_bytes();
    let tag = |raw: u64| -> &'static str {
        if raw == ctx.vm.universe.true_obj.raw() {
            " =true"
        } else if raw == ctx.vm.universe.false_obj.raw() {
            " =false"
        } else if raw == ctx.vm.universe.nil_obj.raw() {
            " =nil"
        } else {
            ""
        }
    };
    // Shared annotation for one instruction at `off`, whatever the ISA.
    let annotate = |off: usize, pool_target: Option<usize>| -> String {
        let mut a = String::new();
        if off == nm.entry_off as usize {
            a.push_str("  ; entry");
        }
        if off == nm.verified_entry_off as usize && nm.verified_entry_off != nm.entry_off {
            a.push_str("  ; verified_entry");
        }
        if ic_offs.contains(&off) {
            a.push_str("  ; IC send site");
        }
        if sp_offs.contains(&off) {
            a.push_str("  ; safepoint");
        }
        if let Some(t) = pool_target {
            if t + 8 <= all.len() {
                let raw = u64::from_le_bytes(all[t..t + 8].try_into().expect("8 bytes"));
                a.push_str(&format!("  ; pool[{t:#x}]={raw:#x}{}", tag(raw)));
            }
        }
        a
    };

    // WINVM: variable-length instructions mean the listing cannot be
    // indexed by `i * 4`. Offsets come from the decoder itself, which is
    // also what supplies the RIP-relative pool target.
    #[cfg(not(target_arch = "aarch64"))]
    {
        let base = nm.code.base as u64;
        // A trap site is three bytes, only the first of which is an
        // instruction. Everything inside one is data and must be SKIPPED,
        // not rendered — the decoder happily reads `00 de` as `add dh,bl`.
        let mut skip_until = 0usize;
        for insn in
            crate::compiler::disasm_x64::decode_all(code, base, Some(nm.literal_off as usize))
        {
            if insn.off < skip_until {
                continue;
            }
            // A deopt trap is `int3` + a raw imm16 that is DATA; decoding
            // those two bytes as an instruction is faithful and useless.
            if let Some(imm) = crate::compiler::disasm_x64::trap_imm_at(all, insn.off) {
                skip_until = insn.off + 3;
                rows.push(format!(
                    "+{:#06x}  int3 {imm:#06x}{}  ; deopt trap",
                    insn.off,
                    annotate(insn.off, None)
                ));
                continue;
            }
            let pool = insn
                .ip_rel_target
                .map(|t| t.wrapping_sub(base) as usize)
                .filter(|t| *t >= nm.literal_off as usize);
            rows.push(format!(
                "+{:#06x}  {}{}",
                insn.off,
                insn.text,
                annotate(insn.off, pool)
            ));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let listing = crate::compiler::disasm_a64::disasm_slice(code, None);
        for (i, line) in listing.lines().enumerate() {
            let off = i * 4;
            // Resolve ldr-literal pool loads: word 0x58xxxxxx, imm19 = bits[23:5].
            let mut pool = None;
            if off + 4 <= all.len() {
                let w = u32::from_le_bytes([all[off], all[off + 1], all[off + 2], all[off + 3]]);
                if w & 0xff00_0000 == 0x5800_0000 {
                    let imm19 = ((w >> 5) & 0x7ffff) as i64;
                    let disp = if imm19 & (1 << 18) != 0 { imm19 - (1 << 19) } else { imm19 };
                    let target = off as i64 + disp * 4;
                    if target >= 0 {
                        pool = Some(target as usize);
                    }
                }
            }
            rows.push(format!("{line}{}", annotate(off, pool)));
        }
    }
    Ok(Value::new(rows.join("\n")))
}

fn verb_pin(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    ctx.vm.debug.active = true;
    crate::runtime::debug::pin_by_name(&mut ctx.vm, args[0].as_str(), args[1].as_str())
        .map(Value::new)
        .map_err(TclError::runtime)
}

fn verb_unpin(_vm: &mut Vm<'_>, args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    crate::runtime::debug::unpin_by_name(&mut ctx.vm, args[0].as_str(), args[1].as_str())
        .map(Value::new)
        .map_err(TclError::runtime)
}

fn verb_ring(_vm: &mut Vm<'_>, _args: &[Value]) -> TclResult<Value> {
    let ctx = bridge::active_ctx();
    use crate::runtime::vm_state::ProbeEvent;
    let mut rows: Vec<String> = vec![format!(
        "showing last {} of {} events (oldest first)",
        ctx.vm.probe_ring.iter_oldest_first().count(),
        ctx.vm.probe_ring.total
    )];
    for e in ctx.vm.probe_ring.iter_oldest_first() {
        rows.push(match e {
            ProbeEvent::Compile { nm, version } => format!("compile   nm={nm} v{version}"),
            ProbeEvent::Deopt { nm, bci, reexecute } => {
                format!("deopt     nm={nm} bci={bci} reexecute={reexecute}")
            }
            ProbeEvent::Invalidate { nm } => format!("invalidate nm={nm}"),
        });
    }
    Ok(Value::new(rows.join("\n")))
}
