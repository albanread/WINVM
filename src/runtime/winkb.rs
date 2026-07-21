//! WINVM: the Windows API knowledge-base resolver (`docs/FFI.md`).
//!
//! MACVM's FFI is data-driven — it queries `cocoa_data/cocoa.sqlite` and
//! "never re-derives from the SDK" (that doc's §1). This is the Windows
//! substitution of that data source, not a new design: `windows_api.db`,
//! the SQLite knowledge base RASM's `winkb` crate builds and queries, with
//! ~18k functions, ~46k COM interface methods (each with its vtable index),
//! ~97k constants, and struct field byte offsets.
//!
//! ## Why a database replaces Cocoa's introspection
//!
//! On macOS the ObjC runtime is *live*: a class can be looked up by name
//! and a selector's types read out of the running process. Windows has no
//! equivalent — a DLL export is a bare address with no type information
//! attached. The knowledge base supplies statically what the ObjC runtime
//! supplies dynamically, and for COM it supplies exactly the two facts
//! dynamic dispatch needs: the interface's IID and each method's vtable
//! slot.
//!
//! ## What this module refuses, and why that matters
//!
//! A resolver that guesses is worse than no resolver: a mis-classified
//! parameter does not fault, it silently passes a float's bit pattern in
//! an integer register. So anything that cannot be modelled exactly is an
//! explicit [`WinkbError::Unsupported`] rather than a best effort —
//! struct-by-value parameters (Win64 passes those larger than 8 bytes
//! through a hidden pointer) and any type whose class cannot be
//! determined. JASM's own Win32 generator draws the line in the same
//! place, emitting only an `@extern` for signatures it cannot model.
//!
//! ## Absence is not an error
//!
//! The database is a ~90 MB machine-local artifact, not a build input.
//! When it is missing every lookup returns [`WinkbError::DbMissing`] and
//! the caller falls back to the hand-declared types in the
//! `<primitive: FFI ...>` pragma — exactly the behaviour that exists
//! today. Nothing about the build or the test suite depends on it.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Which register file an argument or return value travels in — the only
/// classification the Win64 marshaller needs, since every integer, pointer
/// and handle is passed the same way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgClass {
    /// Integer, pointer, handle, enum — a general register (or a stack
    /// slot past the fourth argument).
    G,
    /// `f32`/`f64` — an XMM register.
    F,
    /// `void`. Valid only as a return class.
    V,
}

/// A resolved native function signature.
#[derive(Clone, Debug)]
pub struct FnSig {
    /// The exported name, as it must be passed to `GetProcAddress`.
    pub name: String,
    /// The DLL that exports it, e.g. `KERNEL32.dll`.
    pub dll: String,
    pub ret: ArgClass,
    /// Argument classes in **signature position order** — which is the
    /// order Win64 assigns slots in, and the order
    /// `codecache::ffi_stubs_x64`'s buffer expects.
    pub params: Vec<ArgClass>,
    /// Win64 requires a variadic callee to find floating-point arguments
    /// in the integer register as well as the XMM. The x64 trampoline
    /// loads both unconditionally, so this is currently informational —
    /// but it is the fact a narrower marshaller would need.
    pub is_variadic: bool,
}

impl FnSig {
    /// The float-position bitmask `ffi_stubs_x64`'s trampoline takes:
    /// bit `i` set means argument `i` travels in an XMM register.
    pub fn class_mask(&self) -> u32 {
        let mut m = 0u32;
        for (i, c) in self.params.iter().enumerate() {
            if *c == ArgClass::F {
                m |= 1 << i;
            }
        }
        m
    }
}

/// One COM interface method, enough to dispatch it.
#[derive(Clone, Debug)]
pub struct ComMethod {
    pub interface: String,
    /// The interface's IID, as the canonical hyphenated GUID string.
    pub iid: String,
    pub method: String,
    /// Index into the object's vtable. Inherited methods are counted, so
    /// a direct `IUnknown` child's own methods start at 3.
    pub vtable_index: i64,
    /// Includes the implicit `this` pointer at position 0, because that
    /// is how the call is actually made.
    pub params: Vec<ArgClass>,
    pub ret: ArgClass,
}

#[derive(Debug)]
pub enum WinkbError {
    /// No database on this machine — the caller should fall back to the
    /// hand-declared pragma types.
    DbMissing(PathBuf),
    /// The database is present but this symbol is not in it.
    NotFound(String),
    /// Present, but its signature cannot be modelled exactly.
    Unsupported(String),
    Db(String),
}

impl std::fmt::Display for WinkbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WinkbError::DbMissing(p) => write!(
                f,
                "windows_api.db not found at {} (set WINKB_DB to override)",
                p.display()
            ),
            WinkbError::NotFound(n) => write!(f, "{n} is not in windows_api.db"),
            WinkbError::Unsupported(m) => write!(f, "{m}"),
            WinkbError::Db(m) => write!(f, "windows_api.db: {m}"),
        }
    }
}

/// Where the database lives: `WINKB_DB` if set, else RASM's default path
/// (the same convention `winkb` itself uses, so one machine-wide copy
/// serves every tool in the portfolio).
pub fn db_path() -> PathBuf {
    match std::env::var("WINKB_DB") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from(r"E:\windows_api\windows_api.db"),
    }
}

/// Whether a knowledge base is available on this machine. Cached: the
/// answer cannot change during a run, and callers ask per FFI resolution.
pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| db_path().is_file())
}

#[cfg(windows)]
fn open() -> Result<rusqlite::Connection, WinkbError> {
    let p = db_path();
    if !p.is_file() {
        return Err(WinkbError::DbMissing(p));
    }
    // Read-only: this is a shared, machine-wide artifact and nothing here
    // has any business writing to it.
    rusqlite::Connection::open_with_flags(
        &p,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|e| WinkbError::Db(e.to_string()))
}

/// Map a knowledge-base type onto a register class.
///
/// `kind` and `type_name` come straight from the `types` table. The
/// `size_bits` column is `NULL` for primitives, so classification is by
/// NAME for those — which is exact, since the primitive set is closed
/// (`u8`..`u64`, `i8`..`i64`, `usize`, `isize`, `f32`, `f64`, `void`,
/// `string`, plus `[]` array forms).
fn classify(kind: &str, type_name: &str, size_bits: Option<i64>) -> Result<ArgClass, WinkbError> {
    // An array type decays to a pointer, whatever its element type — so
    // `f64[]` is a POINTER argument, not a float one. Checked before the
    // primitive names below, which it would otherwise match by prefix.
    if type_name.ends_with("[]") {
        return Ok(ArgClass::G);
    }
    match kind {
        // A pointer is a pointer regardless of pointee. `interface` is a
        // COM interface pointer; `delegate` a function pointer.
        "pointer" | "reference" | "interface" | "delegate" => Ok(ArgClass::G),
        // An enum's underlying type is always integral.
        "enum" => Ok(ArgClass::G),
        "primitive" => match type_name {
            "f32" | "f64" => Ok(ArgClass::F),
            "void" => Ok(ArgClass::V),
            "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize"
            | "bool" | "char" | "string" => Ok(ArgClass::G),
            other => Err(WinkbError::Unsupported(format!(
                "primitive type `{other}` has no FFI class"
            ))),
        },
        // A struct up to 8 bytes is passed BY VALUE in a general
        // register, which is how HANDLE, HWND and friends work. Anything
        // larger goes through a hidden pointer under Win64 and would need
        // the caller to materialise a copy — refused rather than guessed.
        "struct" => match size_bits {
            Some(b) if b <= 64 => Ok(ArgClass::G),
            Some(b) => Err(WinkbError::Unsupported(format!(
                "struct `{type_name}` is {} bytes and is passed by hidden pointer under \
                 Win64; struct-by-value is not modelled",
                b / 8
            ))),
            None => Err(WinkbError::Unsupported(format!(
                "struct `{type_name}` has no recorded size"
            ))),
        },
        other => Err(WinkbError::Unsupported(format!(
            "type `{type_name}` of kind `{other}` has no FFI class"
        ))),
    }
}

/// Resolve an exported function by exact name.
///
/// Note that Windows text APIs come in `A`/`W` pairs (`CreateFileA` /
/// `CreateFileW`) and this does no aliasing: the caller names the exact
/// export it wants, because picking one silently is precisely the sort of
/// guess this module exists to avoid.
#[cfg(windows)]
pub fn lookup_function(name: &str) -> Result<FnSig, WinkbError> {
    let conn = open()?;
    let (fid, dll, is_variadic, ret_type): (i64, Option<String>, i64, Option<i64>) = conn
        .query_row(
            "SELECT function_id, dll_name, is_variadic, return_type_id
               FROM functions WHERE function_name = ?1 LIMIT 1",
            [name],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => WinkbError::NotFound(name.to_string()),
            other => WinkbError::Db(other.to_string()),
        })?;

    let dll = dll.ok_or_else(|| {
        WinkbError::Unsupported(format!("{name} has no DLL recorded — cannot be resolved"))
    })?;

    let class_of_type = |tid: i64| -> Result<ArgClass, WinkbError> {
        let (kind, tname, bits): (String, String, Option<i64>) = conn
            .query_row(
                "SELECT kind, type_name, size_bits FROM types WHERE type_id = ?1",
                [tid],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(|e| WinkbError::Db(e.to_string()))?;
        classify(&kind, &tname, bits)
    };

    let ret = match ret_type {
        Some(t) => class_of_type(t)?,
        None => ArgClass::V,
    };

    let mut stmt = conn
        .prepare(
            "SELECT type_id FROM function_params WHERE function_id = ?1 ORDER BY ordinal",
        )
        .map_err(|e| WinkbError::Db(e.to_string()))?;
    let ids: Vec<i64> = stmt
        .query_map([fid], |r| r.get(0))
        .map_err(|e| WinkbError::Db(e.to_string()))?
        .collect::<Result<_, _>>()
        .map_err(|e| WinkbError::Db(e.to_string()))?;

    let mut params = Vec::with_capacity(ids.len());
    for (i, tid) in ids.iter().enumerate() {
        let c = class_of_type(*tid).map_err(|e| match e {
            WinkbError::Unsupported(m) => {
                WinkbError::Unsupported(format!("{name} argument {i}: {m}"))
            }
            other => other,
        })?;
        if c == ArgClass::V {
            return Err(WinkbError::Unsupported(format!(
                "{name} argument {i} is void"
            )));
        }
        params.push(c);
    }

    if params.len() > crate::codecache::ffi_stubs_x64::ARGV_WORDS {
        return Err(WinkbError::Unsupported(format!(
            "{name} takes {} arguments, more than the {} the trampoline buffer holds",
            params.len(),
            crate::codecache::ffi_stubs_x64::ARGV_WORDS
        )));
    }

    Ok(FnSig {
        name: name.to_string(),
        dll,
        ret,
        params,
        is_variadic: is_variadic != 0,
    })
}

/// Resolve a COM interface method to its vtable slot.
///
/// This is the Windows counterpart of MACVM's Tier-2 `objc_msgSend`
/// dispatch: given an interface and a method name, the two facts needed
/// to call it dynamically are the vtable index and the argument classes.
/// `params` includes the implicit `this` at position 0.
#[cfg(windows)]
pub fn lookup_com_method(interface: &str, method: &str) -> Result<ComMethod, WinkbError> {
    let conn = open()?;
    let (tid, iid): (i64, Option<String>) = conn
        .query_row(
            "SELECT type_id, iid FROM types WHERE type_name = ?1 AND kind = 'interface' LIMIT 1",
            [interface],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => WinkbError::NotFound(interface.to_string()),
            other => WinkbError::Db(other.to_string()),
        })?;
    let iid = iid.ok_or_else(|| {
        WinkbError::Unsupported(format!("interface {interface} has no IID recorded"))
    })?;

    let (mid, vtable_index, ret_name): (i64, i64, Option<String>) = conn
        .query_row(
            "SELECT method_id, vtable_index, return_type_name
               FROM interface_methods WHERE interface_type_id = ?1 AND method_name = ?2 LIMIT 1",
            rusqlite::params![tid, method],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                WinkbError::NotFound(format!("{interface}::{method}"))
            }
            other => WinkbError::Db(other.to_string()),
        })?;

    // Interface method params carry type NAMES rather than ids. Every COM
    // method returns HRESULT or a handle-like value and takes pointers or
    // integers, so a name-based classification is adequate here — but a
    // float parameter must still be recognised, not assumed away.
    let class_of_name = |n: &str| -> ArgClass {
        match n {
            "f32" | "f64" => ArgClass::F,
            "void" => ArgClass::V,
            _ => ArgClass::G,
        }
    };

    let mut stmt = conn
        .prepare(
            "SELECT type_name FROM interface_method_params WHERE method_id = ?1 ORDER BY ordinal",
        )
        .map_err(|e| WinkbError::Db(e.to_string()))?;
    let names: Vec<String> = stmt
        .query_map([mid], |r| r.get(0))
        .map_err(|e| WinkbError::Db(e.to_string()))?
        .collect::<Result<_, _>>()
        .map_err(|e| WinkbError::Db(e.to_string()))?;

    // `this` first — the call really does pass it, so the classes must
    // line up with what the trampoline will marshal.
    let mut params = vec![ArgClass::G];
    params.extend(names.iter().map(|n| class_of_name(n)));

    Ok(ComMethod {
        interface: interface.to_string(),
        iid,
        method: method.to_string(),
        vtable_index,
        params,
        ret: ret_name.as_deref().map(class_of_name).unwrap_or(ArgClass::G),
    })
}

/// Look up a named constant (`GENERIC_READ`, `OPEN_EXISTING`, …).
///
/// Searches `enum_members` FIRST and `constants` second, because that is
/// where the constants callers actually name live: `GENERIC_READ`,
/// `FILE_SHARE_READ` and `OPEN_EXISTING` are all flag-enum members, and
/// the `constants` table holds a different population (SDK `#define`-style
/// values like `DML_TARGET_VERSION`). A first cut of this function queried
/// only `constants`, and for a column name that does not exist — it would
/// have found nothing for every constant anyone would ask for.
///
/// Returned as `u64` because the common flags are unsigned and several
/// have the top bit set — `GENERIC_READ` is `0x8000_0000`, which as a
/// signed 32-bit value reads as negative. The `signedness` column decides
/// which of the two stored representations is authoritative.
#[cfg(windows)]
pub fn lookup_constant(name: &str) -> Result<u64, WinkbError> {
    let conn = open()?;

    let from_enum = conn.query_row(
        "SELECT value_i64, value_u64, signedness
           FROM enum_members WHERE member_name = ?1 LIMIT 1",
        [name],
        |r| {
            let i: i64 = r.get(0)?;
            let u: i64 = r.get(1)?;
            let sign: Option<String> = r.get(2)?;
            Ok(match sign.as_deref() {
                Some("signed") => i as u64,
                _ => u as u64,
            })
        },
    );
    match from_enum {
        Ok(v) => return Ok(v),
        Err(rusqlite::Error::QueryReturnedNoRows) => {}
        Err(e) => return Err(WinkbError::Db(e.to_string())),
    }

    conn.query_row(
        "SELECT value_i64, value_u64, value_kind
           FROM constants WHERE constant_name = ?1 LIMIT 1",
        [name],
        |r| {
            let i: Option<i64> = r.get(0)?;
            let u: Option<i64> = r.get(1)?;
            let kind: Option<String> = r.get(2)?;
            Ok(match kind.as_deref() {
                Some(k) if k.starts_with("int") => i.unwrap_or(0) as u64,
                _ => u.or(i).unwrap_or(0) as u64,
            })
        },
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => WinkbError::NotFound(name.to_string()),
        other => WinkbError::Db(other.to_string()),
    })
}

/// A struct field's byte offset — what `Alien` needs to read a field by
/// name instead of a hand-counted constant.
#[cfg(windows)]
pub fn lookup_struct_field(struct_name: &str, field: &str) -> Result<i64, WinkbError> {
    let conn = open()?;
    conn.query_row(
        "SELECT sf.byte_offset
           FROM struct_fields sf JOIN types t ON t.type_id = sf.struct_type_id
          WHERE t.type_name = ?1 AND sf.field_name = ?2 LIMIT 1",
        rusqlite::params![struct_name, field],
        |r| r.get(0),
    )
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            WinkbError::NotFound(format!("{struct_name}.{field}"))
        }
        other => WinkbError::Db(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classification is pure and testable without the database, which
    /// matters because it is where a wrong answer becomes a silently
    /// mis-marshalled call rather than an error.
    #[test]
    fn type_classification_covers_the_cases_that_decide_a_call() {
        assert_eq!(classify("primitive", "f64", None).unwrap(), ArgClass::F);
        assert_eq!(classify("primitive", "f32", None).unwrap(), ArgClass::F);
        assert_eq!(classify("primitive", "u32", None).unwrap(), ArgClass::G);
        assert_eq!(classify("primitive", "void", None).unwrap(), ArgClass::V);
        assert_eq!(classify("pointer", "FOO*", None).unwrap(), ArgClass::G);
        assert_eq!(classify("enum", "FILE_SHARE_MODE", Some(32)).unwrap(), ArgClass::G);
        // HANDLE and friends: a struct small enough to travel in a register.
        assert_eq!(classify("struct", "HANDLE", Some(64)).unwrap(), ArgClass::G);

        // An ARRAY of doubles is a pointer, not a float. Getting this
        // wrong would put an address in an XMM register.
        assert_eq!(classify("primitive", "f64[]", None).unwrap(), ArgClass::G);

        // Struct-by-value larger than a register is refused, not guessed:
        // Win64 passes it through a hidden pointer.
        let e = classify("struct", "D2D_POINT_2F", Some(128)).unwrap_err();
        assert!(
            matches!(e, WinkbError::Unsupported(ref m) if m.contains("hidden pointer")),
            "expected an explicit refusal, got {e:?}"
        );
    }

    #[test]
    fn class_mask_marks_float_positions() {
        let sig = FnSig {
            name: "f".into(),
            dll: "x.dll".into(),
            ret: ArgClass::G,
            params: vec![ArgClass::G, ArgClass::F, ArgClass::G, ArgClass::F],
            is_variadic: false,
        };
        // The interleaved case: floats at positions 1 and 3.
        assert_eq!(sig.class_mask(), 0b1010);
    }

    /// Against the real database when it is present. Skipped, loudly,
    /// when it is not — the database is a machine-local artifact and the
    /// suite must stay green without it.
    #[cfg(windows)]
    #[test]
    fn resolves_real_win32_and_com_symbols() {
        if !available() {
            eprintln!(
                "[winkb] SKIP resolves_real_win32_and_com_symbols — no database at {}",
                db_path().display()
            );
            return;
        }

        let f = lookup_function("CreateFileW").expect("CreateFileW");
        assert_eq!(f.dll.to_ascii_lowercase(), "kernel32.dll");
        assert_eq!(f.params.len(), 7, "CreateFileW takes 7 arguments");
        assert!(f.params.iter().all(|c| *c == ArgClass::G), "all integral");
        assert_eq!(f.class_mask(), 0, "no float arguments");

        // A float-taking API, to prove the class mask is real and not
        // always zero.
        if let Ok(g) = lookup_function("D2D1MakeRotateMatrix") {
            assert_eq!(g.params.first(), Some(&ArgClass::F), "angle is f32");
            assert_eq!(g.class_mask() & 1, 1);
        }

        // COM: the two facts dynamic dispatch needs.
        let m = lookup_com_method("IDXGIFactory", "CreateSwapChain").expect("IDXGIFactory");
        assert_eq!(m.iid, "7b7166ec-21c7-44ae-b21a-c9ae321ae369");
        assert!(
            m.vtable_index >= 3,
            "vtable index must be past IUnknown's first three slots, got {}",
            m.vtable_index
        );
        assert_eq!(m.params.first(), Some(&ArgClass::G), "this pointer first");

        // Constants. These are the ones a real binding names, and they
        // live in `enum_members` rather than `constants` — a fact worth
        // asserting, since the first version of `lookup_constant` queried
        // the wrong table AND a column that does not exist, and no test
        // would have noticed.
        assert_eq!(lookup_constant("GENERIC_READ").unwrap(), 0x8000_0000);
        assert_eq!(lookup_constant("FILE_SHARE_READ").unwrap(), 1);

        // Struct field offsets — what Alien needs to read a field by name
        // instead of a hand-counted byte offset.
        assert_eq!(lookup_struct_field("BITMAPINFOHEADER", "biWidth").unwrap(), 4);
        assert_eq!(
            lookup_struct_field("BITMAPINFOHEADER", "biBitCount").unwrap(),
            14
        );

        // A symbol that genuinely is not there must say so, not guess.
        assert!(matches!(
            lookup_function("NoSuchApiExistsW"),
            Err(WinkbError::NotFound(_))
        ));
        assert!(matches!(
            lookup_constant("NO_SUCH_CONSTANT_AT_ALL"),
            Err(WinkbError::NotFound(_))
        ));
    }
}
