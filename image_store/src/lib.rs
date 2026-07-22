//! A versioned SQLite class/method source database — see
//! `../../docs/IMAGE.md` for the full design and, critically, §0's
//! distinction from the S16 heap snapshot: this stores **text** (source,
//! plus small metadata) and an optional, always-re-derivable bytecode
//! cache, never oops or heap pointers. Booting from it is still "boot by
//! compiling source," per `docs/SPEC.md` §3.2 — a different source
//! *container* than `world/*.mst` flat files, not a different bootstrap
//! model.
//!
//! Two SQL views (`latest_class_versions`, `latest_method_versions`) do the
//! "latest version" lookup once, in the schema, rather than repeating a
//! correlated `MAX(version_number)` subquery in every Rust-side query
//! below — see `docs/IMAGE.md` §4 for why "latest" is always computed, not
//! a stored pointer that could drift out of sync.

pub mod export;
pub mod flows;
pub mod import;
pub mod mst;
pub mod world_boot;

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// `docs/IMAGE.md` §4 — keep the last 100 versions per method/class.
pub const RETENTION_LIMIT: i64 = 100;
/// `docs/IMAGE.md` §3 — sparse gap numbering for `load_order`.
pub const LOAD_ORDER_START: i64 = 1000;
pub const LOAD_ORDER_STEP: i64 = 100;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Instance,
    Class,
}

impl Side {
    fn as_str(self) -> &'static str {
        match self {
            Side::Instance => "instance",
            Side::Class => "class",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassSummary {
    pub name: String,
    pub superclass: Option<String>,
    pub load_order: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodSummary {
    pub selector: String,
    pub category: String,
}

/// A class's complete latest-version definition — for callers that need
/// *everything*, not one field at a time: the GUI's mock-world mirror
/// (`gui/src/vm_host.rs`) and, eventually, an image→`.mst` exporter
/// (`docs/IMAGE.md` §6/§9, not built yet).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FullClass {
    pub name: String,
    pub superclass: Option<String>,
    pub category: String,
    pub comment: String,
    pub instance_vars: String,
    pub class_vars: String,
    pub load_order: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FullMethod {
    pub selector: String,
    pub side: Side,
    pub category: String,
    pub source: String,
}

/// Result of [`Image::create_or_reopen_class`] — see its doc comment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassCreateOutcome {
    Created,
    Reopened,
    AlreadyLive,
}

/// Whether a `(class, side, selector)` method is currently defined — lets the
/// exporter tell an intentional deletion (a tombstone) apart from a method that
/// was simply never in the image, so it only ever *removes* the former from a
/// world file (never a benignly-unknown method).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodPresence {
    /// The method exists and its latest version is live; carries that source.
    Live(String),
    /// The method exists but its latest version is a deletion tombstone.
    Deleted,
    /// No such method record at all.
    Absent,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS classes (
    class_id    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    load_order  INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS class_versions (
    version_id      INTEGER PRIMARY KEY,
    class_id        INTEGER NOT NULL REFERENCES classes(class_id),
    version_number  INTEGER NOT NULL,
    superclass_name TEXT,
    category        TEXT NOT NULL DEFAULT '',
    comment         TEXT NOT NULL DEFAULT '',
    instance_vars   TEXT NOT NULL DEFAULT '',
    class_vars      TEXT NOT NULL DEFAULT '',
    edited_at       INTEGER NOT NULL,
    deleted         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(class_id, version_number)
);
CREATE INDEX IF NOT EXISTS idx_class_versions_latest ON class_versions(class_id, version_number DESC);
CREATE TABLE IF NOT EXISTS methods (
    method_id  INTEGER PRIMARY KEY,
    class_id   INTEGER NOT NULL REFERENCES classes(class_id),
    selector   TEXT NOT NULL,
    side       TEXT NOT NULL CHECK(side IN ('instance','class')),
    source_file TEXT,
    UNIQUE(class_id, selector, side)
);
CREATE TABLE IF NOT EXISTS method_versions (
    version_id     INTEGER PRIMARY KEY,
    method_id      INTEGER NOT NULL REFERENCES methods(method_id),
    version_number INTEGER NOT NULL,
    category       TEXT NOT NULL DEFAULT 'as yet unclassified',
    source         TEXT NOT NULL,
    edited_at      INTEGER NOT NULL,
    deleted        INTEGER NOT NULL DEFAULT 0,
    UNIQUE(method_id, version_number)
);
CREATE INDEX IF NOT EXISTS idx_method_versions_latest ON method_versions(method_id, version_number DESC);
CREATE TABLE IF NOT EXISTS method_bytecode (
    method_version_id INTEGER PRIMARY KEY REFERENCES method_versions(version_id),
    bytecode           BLOB NOT NULL,
    literals_json      TEXT,
    compiler_tag       TEXT NOT NULL,
    compiled_at        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS method_sends (
    method_version_id INTEGER NOT NULL REFERENCES method_versions(version_id),
    selector          TEXT NOT NULL,
    PRIMARY KEY (method_version_id, selector)
);
CREATE INDEX IF NOT EXISTS idx_method_sends_selector ON method_sends(selector);
CREATE TABLE IF NOT EXISTS package_lists (
    list_id  INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS package_list_members (
    list_id  INTEGER NOT NULL REFERENCES package_lists(list_id),
    package  TEXT NOT NULL,
    UNIQUE(list_id, package)
);
CREATE VIEW IF NOT EXISTS latest_class_versions AS
    SELECT cv.* FROM class_versions cv
    WHERE cv.version_number = (SELECT MAX(version_number) FROM class_versions WHERE class_id = cv.class_id);
CREATE VIEW IF NOT EXISTS latest_method_versions AS
    SELECT mv.* FROM method_versions mv
    WHERE mv.version_number = (SELECT MAX(version_number) FROM method_versions WHERE method_id = mv.method_id);
";

pub struct Image {
    conn: Connection,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Upgrade an already-existing database file created before the `deleted`
/// tombstone column existed (`CREATE TABLE IF NOT EXISTS` in `SCHEMA` above
/// only adds it to *brand-new* tables) — checked via `PRAGMA table_info`
/// rather than just trying the `ALTER TABLE` and swallowing a "duplicate
/// column" error, so a real failure doesn't get masked the same way. Safe
/// to call on a fresh database too: `SCHEMA` already created the column, so
/// this is a no-op.
fn migrate_add_deleted_columns(conn: &Connection) -> rusqlite::Result<()> {
    for table in ["class_versions", "method_versions"] {
        let has_deleted = conn
            .prepare(&format!("PRAGMA table_info({table})"))?
            .query_map([], |r| r.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .any(|name| name == "deleted");
        if !has_deleted {
            conn.execute(
                &format!("ALTER TABLE {table} ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0"),
                [],
            )?;
        }
    }
    Ok(())
}

/// Add the `methods.source_file` provenance column to a pre-existing database
/// (the world file each method came from — used by `export` to write an edit,
/// or a deletion, back into the right `.mst`). Same PRAGMA-guarded shape as
/// [`migrate_add_deleted_columns`]; a no-op on a fresh DB where `SCHEMA` already
/// created it.
fn migrate_add_source_file(conn: &Connection) -> rusqlite::Result<()> {
    let has_col = conn
        .prepare("PRAGMA table_info(methods)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|name| name == "source_file");
    if !has_col {
        conn.execute("ALTER TABLE methods ADD COLUMN source_file TEXT", [])?;
    }
    Ok(())
}

impl Image {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        migrate_add_deleted_columns(&conn)?;
        migrate_add_source_file(&conn)?;
        Ok(Self { conn })
    }

    /// Open for READING only — no schema creation, no migrations, no write
    /// lock ever taken. For pure readers of an image another actor owns (the
    /// Cocoa GUI's source pane, CG7): a concurrent writer can't be blocked by
    /// us, we can't mutate a file we don't own, and a busy/missing image is an
    /// immediate `Err` the caller degrades on — never a main-thread stall.
    pub fn open_read_only(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        Ok(Self { conn })
    }

    pub fn open_in_memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        migrate_add_deleted_columns(&conn)?;
        migrate_add_source_file(&conn)?;
        Ok(Self { conn })
    }

    fn class_id_of(&self, class_name: &str) -> rusqlite::Result<Option<i64>> {
        self.conn
            .query_row(
                "SELECT class_id FROM classes WHERE name = ?1",
                params![class_name],
                |r| r.get(0),
            )
            .optional()
    }

    /// Whether a class by this name already exists — for callers (like the
    /// `.mst` importer, `src/bin/import_world.rs`) that need to distinguish
    /// "define a new class" from "reopen an existing one to add more
    /// methods," which real `.mst` files legitimately do (confirmed against
    /// the corpus: `01_object.mst`'s own header comment says Boolean- and
    /// printing-dependent `Object` methods land in *later* files, reopening
    /// the same class).
    pub fn class_exists(&self, class_name: &str) -> rusqlite::Result<bool> {
        Ok(self.class_id_of(class_name)?.is_some())
    }

    fn method_id_of(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<Option<i64>> {
        self.conn
            .query_row(
                "SELECT m.method_id FROM classes c JOIN methods m ON m.class_id = c.class_id \
                 WHERE c.name = ?1 AND m.side = ?2 AND m.selector = ?3",
                params![class_name, side.as_str(), selector],
                |r| r.get(0),
            )
            .optional()
    }

    // ── Read queries — mirrors macvm-mock-vm's MockWorld API shape ────────

    /// Package/category names, in first-load-order-appearance order.
    pub fn packages(&self) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT lcv.category, MIN(c.load_order) AS first_lo \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.deleted = 0 \
             GROUP BY lcv.category ORDER BY first_lo",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect()
    }

    /// Named package-list names (`docs/package_aware_editing_design.md` §4.5)
    /// — a boot request names one or more of these, e.g. `["world"]` or
    /// `["world", "cocoaui"]`. `list_id` order, which is creation order (the
    /// importer processes `world.list` before any other `*.list` file, so
    /// this is also a reasonable "most-foundational-first" default).
    pub fn package_lists(&self) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT name FROM package_lists ORDER BY list_id")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect()
    }

    /// The packages `list_name` contains (empty if the list doesn't exist —
    /// not an error, matching `packages()`'s own "empty is a legitimate
    /// answer" shape).
    pub fn packages_in_list(&self, list_name: &str) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT plm.package FROM package_list_members plm \
             JOIN package_lists pl ON pl.list_id = plm.list_id \
             WHERE pl.name = ?1 ORDER BY plm.package",
        )?;
        let rows = stmt.query_map(params![list_name], |r| r.get::<_, String>(0))?;
        rows.collect()
    }

    /// Record that `list_name` contains `package` — idempotent (a re-import
    /// or an incremental reseed calls this again for the same pair). Creates
    /// `list_name`'s own `package_lists` row on first mention; no separate
    /// "create the list" step exists or is needed.
    pub fn ensure_package_list_member(&self, list_name: &str, package: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO package_lists (name) VALUES (?1)",
            params![list_name],
        )?;
        self.conn.execute(
            "INSERT OR IGNORE INTO package_list_members (list_id, package) \
             SELECT list_id, ?2 FROM package_lists WHERE name = ?1",
            params![list_name, package],
        )?;
        Ok(())
    }

    /// Every class whose package belongs to any of `list_names`, in the
    /// same dependency-safe `load_order` `all_classes()` uses — a single
    /// global sequence assigned once at import time, so it sorts correctly
    /// across any subset of packages without a separate per-list ordering
    /// column. Boot's entry point for a selective (non-`all_classes()`) load
    /// (`docs/package_aware_editing_design.md` §4.5) — not yet called by any
    /// boot path as of this writing (that's a later milestone); exists now
    /// so the schema and its query are provably correct together.
    pub fn classes_for_lists(&self, list_names: &[&str]) -> rusqlite::Result<Vec<FullClass>> {
        if list_names.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = list_names.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT c.name, lcv.superclass_name, lcv.category, lcv.comment, \
                    lcv.instance_vars, lcv.class_vars, c.load_order \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.deleted = 0 AND lcv.category IN ( \
                 SELECT DISTINCT plm.package FROM package_list_members plm \
                 JOIN package_lists pl ON pl.list_id = plm.list_id \
                 WHERE pl.name IN ({placeholders}) \
             ) ORDER BY c.load_order"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(list_names.iter()), |r| {
            Ok(FullClass {
                name: r.get(0)?,
                superclass: r.get(1)?,
                category: r.get(2)?,
                comment: r.get(3)?,
                instance_vars: r.get(4)?,
                class_vars: r.get(5)?,
                load_order: r.get(6)?,
            })
        })?;
        rows.collect()
    }

    /// Classes in `package` whose latest superclass is absent, removed, or
    /// lives in a different package — `docs/IMAGE.md`'s package-roots rule,
    /// same as `macvm-mock-vm`'s `package_roots`. A removed superclass
    /// counts the same as a missing one (not `slcv.deleted = 0`, i.e.
    /// excluded from the "does a live same-package superclass exist"
    /// check): removing a class is allowed even with subclasses still
    /// pointing at it (the GUI warns first but doesn't block), so those
    /// subclasses need to re-root visually rather than vanish or crash.
    pub fn package_roots(&self, package: &str) -> rusqlite::Result<Vec<ClassSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, lcv.superclass_name, c.load_order \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.category = ?1 AND lcv.deleted = 0 \
               AND (lcv.superclass_name IS NULL OR NOT EXISTS ( \
                     SELECT 1 FROM classes sc JOIN latest_class_versions slcv ON slcv.class_id = sc.class_id \
                     WHERE sc.name = lcv.superclass_name AND slcv.category = lcv.category AND slcv.deleted = 0)) \
             ORDER BY c.load_order",
        )?;
        let rows = stmt.query_map(params![package], |r| {
            Ok(ClassSummary {
                name: r.get(0)?,
                superclass: r.get(1)?,
                load_order: r.get(2)?,
            })
        })?;
        rows.collect()
    }

    pub fn subclasses_of(&self, class_name: &str) -> rusqlite::Result<Vec<ClassSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, lcv.superclass_name, c.load_order \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.superclass_name = ?1 AND lcv.deleted = 0 ORDER BY c.load_order",
        )?;
        let rows = stmt.query_map(params![class_name], |r| {
            Ok(ClassSummary {
                name: r.get(0)?,
                superclass: r.get(1)?,
                load_order: r.get(2)?,
            })
        })?;
        rows.collect()
    }

    pub fn categories(&self, class_name: &str, side: Side) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT lmv.category \
             FROM classes c JOIN methods m ON m.class_id = c.class_id \
             JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
             WHERE c.name = ?1 AND m.side = ?2 AND lmv.deleted = 0 ORDER BY m.method_id",
        )?;
        let rows = stmt.query_map(params![class_name, side.as_str()], |r| {
            r.get::<_, String>(0)
        })?;
        rows.collect()
    }

    pub fn methods_in(
        &self,
        class_name: &str,
        side: Side,
        category: &str,
    ) -> rusqlite::Result<Vec<MethodSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.selector, lmv.category \
             FROM classes c JOIN methods m ON m.class_id = c.class_id \
             JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
             WHERE c.name = ?1 AND m.side = ?2 AND lmv.category = ?3 AND lmv.deleted = 0 ORDER BY m.selector",
        )?;
        let rows = stmt.query_map(params![class_name, side.as_str(), category], |r| {
            Ok(MethodSummary {
                selector: r.get(0)?,
                category: r.get(1)?,
            })
        })?;
        rows.collect()
    }

    pub fn method_source(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT lmv.source \
                 FROM classes c JOIN methods m ON m.class_id = c.class_id \
                 JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
                 WHERE c.name = ?1 AND m.side = ?2 AND m.selector = ?3 AND lmv.deleted = 0",
                params![class_name, side.as_str(), selector],
                |r| r.get(0),
            )
            .optional()
    }

    pub fn class_comment(&self, class_name: &str) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT lcv.comment FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
                 WHERE c.name = ?1 AND lcv.deleted = 0",
                params![class_name],
                |r| r.get(0),
            )
            .optional()
    }

    /// Every class's complete latest-version definition, in `load_order` —
    /// enough to rebuild an equivalent world elsewhere (the GUI's mock-world
    /// mirror; eventually an image→`.mst` exporter).
    pub fn all_classes(&self) -> rusqlite::Result<Vec<FullClass>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, lcv.superclass_name, lcv.category, lcv.comment, lcv.instance_vars, lcv.class_vars, c.load_order \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.deleted = 0 \
             ORDER BY c.load_order",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(FullClass {
                name: r.get(0)?,
                superclass: r.get(1)?,
                category: r.get(2)?,
                comment: r.get(3)?,
                instance_vars: r.get(4)?,
                class_vars: r.get(5)?,
                load_order: r.get(6)?,
            })
        })?;
        rows.collect()
    }

    /// One class at its latest version, or `None` if unknown/deleted — the
    /// single-class companion to [`all_classes`].
    pub fn class_named(&self, class_name: &str) -> rusqlite::Result<Option<FullClass>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, lcv.superclass_name, lcv.category, lcv.comment, lcv.instance_vars, lcv.class_vars, c.load_order \
             FROM classes c JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE c.name = ?1 AND lcv.deleted = 0",
        )?;
        let mut rows = stmt.query_map(params![class_name], |r| {
            Ok(FullClass {
                name: r.get(0)?,
                superclass: r.get(1)?,
                category: r.get(2)?,
                comment: r.get(3)?,
                instance_vars: r.get(4)?,
                class_vars: r.get(5)?,
                load_order: r.get(6)?,
            })
        })?;
        rows.next().transpose()
    }

    /// The whole class rendered as one `.mst` text block — the text editor's
    /// fetch (`docs/editor_design.md` §5). `None` if the class is unknown.
    ///
    /// This is deliberately [`export::class_block`] itself, not a second
    /// renderer: what the editor shows is exactly what the exporter would
    /// write, and — the property the accept path depends on — a re-parse of
    /// this text yields each method's stored source *verbatim*, so opening a
    /// class and accepting it unchanged is a no-op rather than a churn of
    /// version bumps. (That idempotence is `class_block`'s own documented
    /// contract; `editor_class_source_round_trips` pins it for every class in
    /// the image.)
    ///
    /// Two things are deliberately NOT in the rendering, because
    /// `class_block` does not emit them: the class **comment** and per-method
    /// **categories**. They are therefore neither shown nor editable here —
    /// and, since an accept only writes back what the text contains, they are
    /// never clobbered either. A comment-aware round trip would need a
    /// renderer/parser pair that agree on where a comment lives; that is not
    /// this.
    pub fn class_source(&self, class_name: &str) -> rusqlite::Result<Option<String>> {
        let Some(class) = self.class_named(class_name)? else {
            return Ok(None);
        };
        let sources: Vec<String> = self
            .all_methods_of(class_name)?
            .into_iter()
            .map(|m| m.source)
            .collect();
        Ok(Some(crate::export::class_block(
            class.superclass.as_deref().unwrap_or("Object"),
            &class.name,
            &class.instance_vars,
            &class.class_vars,
            &sources,
            true,
        )))
    }

    /// Every method (both sides, every category) belonging to `class_name`,
    /// at its latest version — the per-class companion to [`all_classes`].
    pub fn all_methods_of(&self, class_name: &str) -> rusqlite::Result<Vec<FullMethod>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.selector, m.side, lmv.category, lmv.source \
             FROM classes c JOIN methods m ON m.class_id = c.class_id \
             JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
             WHERE c.name = ?1 AND lmv.deleted = 0 ORDER BY m.method_id",
        )?;
        let rows = stmt.query_map(params![class_name], |r| {
            let side_str: String = r.get(1)?;
            Ok(FullMethod {
                selector: r.get(0)?,
                side: if side_str == "class" {
                    Side::Class
                } else {
                    Side::Instance
                },
                category: r.get(2)?,
                source: r.get(3)?,
            })
        })?;
        rows.collect()
    }

    /// Every class name (latest, non-deleted) in load order — the option list
    /// for the find views' class combobox, and the corpus for a class-name
    /// substring search (Find Definition), done in SQL rather than a VM round
    /// trip now that the image is kept in sync with the running VM.
    pub fn class_names(&self) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name FROM classes c \
             JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
             WHERE lcv.deleted = 0 ORDER BY c.load_order",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.collect()
    }

    /// Every distinct selector implemented anywhere (either side), sorted — the
    /// option list for the implementors/senders combobox.
    pub fn all_selectors(&self) -> rusqlite::Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT m.selector FROM methods m \
             JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
             WHERE lmv.deleted = 0 ORDER BY m.selector",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.collect()
    }

    /// Every class implementing `selector` (either side) at its latest version —
    /// the Implementors query in SQL instead of a VM reflection round trip.
    /// Returns `(class_name, side)` sorted for a stable result list.
    pub fn implementors_of(&self, selector: &str) -> rusqlite::Result<Vec<(String, Side)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, m.side FROM classes c \
             JOIN methods m ON m.class_id = c.class_id \
             JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
             WHERE m.selector = ?1 AND lmv.deleted = 0 ORDER BY c.name, m.side",
        )?;
        let rows = stmt.query_map(params![selector], |r| {
            let side: String = r.get(1)?;
            Ok((
                r.get::<_, String>(0)?,
                if side == "class" {
                    Side::Class
                } else {
                    Side::Instance
                },
            ))
        })?;
        rows.collect()
    }

    /// Every method whose LATEST source SENDS `selector` — the Senders query,
    /// answered from the `method_sends` index (populated by
    /// [`Image::backfill_method_sends`] / on every method write) instead of a VM
    /// IC-table scan. Returns `(class_name, sending_method_selector, side)`.
    pub fn senders_of(&self, selector: &str) -> rusqlite::Result<Vec<(String, String, Side)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, m.selector, m.side \
             FROM method_sends ms \
             JOIN latest_method_versions lmv ON lmv.version_id = ms.method_version_id \
             JOIN methods m ON m.method_id = lmv.method_id \
             JOIN classes c ON c.class_id = m.class_id \
             WHERE ms.selector = ?1 AND lmv.deleted = 0 \
             ORDER BY c.name, m.selector, m.side",
        )?;
        let rows = stmt.query_map(params![selector], |r| {
            let side: String = r.get(2)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                if side == "class" {
                    Side::Class
                } else {
                    Side::Instance
                },
            ))
        })?;
        rows.collect()
    }

    /// Populate `method_sends` for every live method version that has no rows
    /// yet — a one-time backfill so an image seeded before senders-indexing
    /// gains it without a full re-import. Idempotent (re-run is cheap; a method
    /// that genuinely sends nothing is simply re-scanned to no effect). Returns
    /// the number of (version, selector) edges inserted.
    pub fn backfill_method_sends(&self) -> rusqlite::Result<usize> {
        let pending: Vec<(i64, String)> = {
            let mut stmt = self.conn.prepare(
                "SELECT mv.version_id, mv.source FROM method_versions mv \
                 WHERE mv.deleted = 0 \
                 AND mv.version_id NOT IN (SELECT DISTINCT method_version_id FROM method_sends)",
            )?;
            let it =
                stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            it.collect::<rusqlite::Result<Vec<_>>>()?
        };
        if pending.is_empty() {
            return Ok(0);
        }
        // ONE transaction and ONE prepared statement for the whole backfill.
        //
        // This is not a micro-optimisation. Outside an explicit transaction
        // SQLite commits every `INSERT` separately, and a commit is an fsync:
        // the world's ~1,300 methods send tens of thousands of selectors, so
        // the unbatched form asks the OS to flush the DB to disk tens of
        // thousands of times. On macOS that is merely slow (fsync there does
        // not force a device-level flush); on Windows/NTFS — with a real
        // flush per commit, and a virus scanner watching the file — the same
        // loop takes minutes, which is indistinguishable from a hang.
        //
        // That is exactly how it presented: the GUI's VM worker calls this on
        // the boot path (`vm_host::open_or_seed_image`), so the whole
        // environment appeared to start and then never serve a single
        // request. Batching makes it one commit, and helps macOS too.
        let mut inserted = 0usize;
        self.conn.execute_batch("BEGIN")?;
        let result = (|| -> rusqlite::Result<()> {
            let mut stmt = self.conn.prepare(
                "INSERT OR IGNORE INTO method_sends (method_version_id, selector) VALUES (?1, ?2)",
            )?;
            for (version_id, source) in pending {
                for selector in crate::mst::sent_selectors(&source) {
                    inserted += stmt.execute(params![version_id, selector])?;
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.conn.execute_batch("COMMIT")?;
                Ok(inserted)
            }
            Err(e) => {
                // Leave no open transaction behind on this connection — a
                // later write would otherwise fail or silently join it.
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Whether a method is live / a deletion tombstone / absent
    /// ([`MethodPresence`]) — the exporter uses it to remove a method from a
    /// world file only when it was intentionally deleted, never when it's merely
    /// unknown to the image.
    pub fn method_presence(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<MethodPresence> {
        let row: Option<(String, i64)> = self
            .conn
            .query_row(
                "SELECT lmv.source, lmv.deleted FROM classes c \
                 JOIN methods m ON m.class_id = c.class_id \
                 JOIN latest_method_versions lmv ON lmv.method_id = m.method_id \
                 WHERE c.name = ?1 AND m.side = ?2 AND m.selector = ?3",
                params![class_name, side.as_str(), selector],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(match row {
            None => MethodPresence::Absent,
            Some((_, deleted)) if deleted != 0 => MethodPresence::Deleted,
            Some((source, _)) => MethodPresence::Live(source),
        })
    }

    /// Record which world file a method lives in (its export home) — set by the
    /// importer as it reads each file, and by the GUI when a method is created.
    /// A no-op if the method row doesn't exist.
    pub fn set_method_home_file(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
        file: &str,
    ) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE methods SET source_file = ?4 \
             WHERE class_id = (SELECT class_id FROM classes WHERE name = ?1) \
             AND side = ?2 AND selector = ?3",
            params![class_name, side.as_str(), selector, file],
        )?;
        Ok(())
    }

    /// THIS method's own `source_file`, `None` if it has none yet (or the
    /// method itself doesn't exist) — distinct from [`Self::class_home_file`],
    /// which answers a majority vote across the whole class's methods and
    /// would wrongly attribute a brand-new, not-yet-homed method to whatever
    /// file most of its *siblings* came from. `flows::save_method` uses this
    /// (not `class_home_file`) to decide whether a method already has real
    /// provenance worth preserving.
    pub fn method_source_file(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT m.source_file FROM classes c \
                 JOIN methods m ON m.class_id = c.class_id \
                 WHERE c.name = ?1 AND m.side = ?2 AND m.selector = ?3",
                params![class_name, side.as_str(), selector],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()
            .map(|opt| opt.flatten())
    }

    /// The world file most of `class_name`'s methods live in — where `export`
    /// puts a newly-created method instead of the catch-all additions file.
    /// `None` if none of the class's methods has a recorded home yet.
    pub fn class_home_file(&self, class_name: &str) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT m.source_file FROM classes c \
                 JOIN methods m ON m.class_id = c.class_id \
                 WHERE c.name = ?1 AND m.source_file IS NOT NULL \
                 GROUP BY m.source_file ORDER BY COUNT(*) DESC LIMIT 1",
                params![class_name],
                |r| r.get(0),
            )
            .optional()
    }

    // ── Mutations — every save is an INSERT, never an UPDATE ──────────────

    /// Create a class with its first version. For the importer (§6) and
    /// for future "new class" UI; returns the new `class_id`.
    pub fn add_class(
        &self,
        name: &str,
        superclass: Option<&str>,
        category: &str,
        comment: &str,
        instance_vars: &str,
        class_vars: &str,
        load_order: i64,
    ) -> rusqlite::Result<i64> {
        self.conn.execute(
            "INSERT INTO classes (name, load_order) VALUES (?1, ?2)",
            params![name, load_order],
        )?;
        let class_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO class_versions (class_id, version_number, superclass_name, category, comment, instance_vars, class_vars, edited_at) \
             VALUES (?1, 1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![class_id, superclass, category, comment, instance_vars, class_vars, now_secs()],
        )?;
        Ok(class_id)
    }

    /// Create a method (on an existing class) with its first version.
    pub fn add_method(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
        category: &str,
        source: &str,
    ) -> rusqlite::Result<Option<i64>> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(None);
        };
        self.conn.execute(
            "INSERT INTO methods (class_id, selector, side) VALUES (?1, ?2, ?3)",
            params![class_id, selector, side.as_str()],
        )?;
        let method_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO method_versions (method_id, version_number, category, source, edited_at) VALUES (?1, 1, ?2, ?3, ?4)",
            params![method_id, category, source, now_secs()],
        )?;
        Ok(Some(method_id))
    }

    /// "Accept" edited source for an *existing* method — `docs/IMAGE.md` §4:
    /// insert a new version (carrying its category forward unchanged), then
    /// prune to `RETENTION_LIMIT`. Returns `false` (no auto-create) if
    /// `class_name`/`side`/`selector` doesn't already name a method — same
    /// contract as `macvm-mock-vm::MockWorld::set_method_source`.
    /// A class's current superclass name (`None` if the class is unknown or
    /// its superclass is `nil`, e.g. `Object`). Used to reconstruct a
    /// method-reopen (`<super> subclass: <Class> [ ... ]`) for live-compile.
    pub fn superclass_of(&self, class_name: &str) -> rusqlite::Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT lcv.superclass_name FROM classes c \
                 JOIN latest_class_versions lcv ON lcv.class_id = c.class_id \
                 WHERE c.name = ?1 AND lcv.deleted = 0",
                params![class_name],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()
            .map(|opt| opt.flatten())
    }

    /// How many versions of this method are retained (each accepted edit adds
    /// one, capped at [`RETENTION_LIMIT`]). `0` if the method is unknown.
    /// Lets callers confirm an edit was versioned, not overwritten, and backs
    /// a future version-history view.
    pub fn method_version_count(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<i64> {
        let Some(method_id) = self.method_id_of(class_name, side, selector)? else {
            return Ok(0);
        };
        self.conn.query_row(
            "SELECT COUNT(*) FROM method_versions WHERE method_id = ?1",
            params![method_id],
            |r| r.get(0),
        )
    }

    pub fn set_method_source(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
        new_source: &str,
    ) -> rusqlite::Result<bool> {
        let Some(method_id) = self.method_id_of(class_name, side, selector)? else {
            return Ok(false);
        };
        let category: String = self.conn.query_row(
            "SELECT category FROM latest_method_versions WHERE method_id = ?1",
            params![method_id],
            |r| r.get(0),
        )?;
        self.insert_method_version(method_id, &category, new_source, false)?;
        self.prune_method_versions(method_id)?;
        Ok(true)
    }

    /// "Accept" an edited class comment — same shape as `set_method_source`,
    /// carrying every other field of the class definition forward unchanged.
    pub fn set_class_comment(&self, class_name: &str, new_comment: &str) -> rusqlite::Result<bool> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(false);
        };
        let (superclass, category, ivars, cvars): (Option<String>, String, String, String) = self.conn.query_row(
            "SELECT superclass_name, category, instance_vars, class_vars FROM latest_class_versions WHERE class_id = ?1",
            params![class_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        self.insert_class_version(
            class_id,
            superclass.as_deref(),
            &category,
            new_comment,
            &ivars,
            &cvars,
            false,
        )?;
        self.prune_class_versions(class_id)?;
        Ok(true)
    }

    /// "Accept" an edited class *definition* — the superclass and instance
    /// variables, i.e. the part of a class header `.mst`'s real grammar
    /// (`mst.rs`) actually has syntax for. Carries `category` (package) and
    /// `comment` forward unchanged: reassigning a class's package isn't
    /// exposed through this path, and `class_vars` is left alone too, since
    /// there's no real `.mst` syntax for class variables to round-trip
    /// through yet (see `mst.rs`'s doc comment) — inventing GUI-only syntax
    /// for a field the file format can't express felt like the wrong kind
    /// of shortcut to take silently.
    pub fn set_class_definition(
        &self,
        class_name: &str,
        new_superclass: Option<&str>,
        new_instance_vars: &str,
    ) -> rusqlite::Result<bool> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(false);
        };
        let (category, comment, cvars): (String, String, String) = self.conn.query_row(
            "SELECT category, comment, class_vars FROM latest_class_versions WHERE class_id = ?1",
            params![class_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        self.insert_class_version(
            class_id,
            new_superclass,
            &category,
            &comment,
            new_instance_vars,
            &cvars,
            false,
        )?;
        self.prune_class_versions(class_id)?;
        Ok(true)
    }

    /// Re-import a live class's declared vars from a re-parsed `.mst`, **merging**
    /// (unioning) the parsed instance/class vars into what's stored rather than
    /// replacing — a fresh version only if that adds something. This is what the
    /// world importer ([`crate::import`]) needs for a class that already exists:
    /// unlike [`Self::set_class_definition`] it can add `class_vars`, and unlike
    /// [`Self::create_or_reopen_class`] it applies to a live class.
    ///
    /// Merge (not overwrite) is essential because a class is often *reopened* in
    /// the same file to add methods WITHOUT restating its `<classVars: …>` — an
    /// overwrite would wipe them (this bit `Character`, whose `Table` was lost on
    /// its second definition, breaking image boot). Adding a var incrementally
    /// still works; only *removing* one needs a full reseed. Superclass/category/
    /// comment are left as first defined. Returns `false` if the class doesn't
    /// exist or nothing new was added.
    pub fn reimport_class_shell(
        &self,
        class_name: &str,
        instance_vars: &str,
        class_vars: &str,
    ) -> rusqlite::Result<bool> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(false);
        };
        let (sc, cat, com, cur_iv, cur_cv): (Option<String>, String, String, String, String) =
            self.conn.query_row(
                "SELECT superclass_name, category, comment, instance_vars, class_vars \
                 FROM latest_class_versions WHERE class_id = ?1",
                params![class_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )?;
        let merged_iv = union_space_separated(&cur_iv, instance_vars);
        let merged_cv = union_space_separated(&cur_cv, class_vars);
        if merged_iv == cur_iv && merged_cv == cur_cv {
            return Ok(false); // nothing new — no version churn
        }
        self.insert_class_version(
            class_id,
            sc.as_deref(),
            &cat,
            &com,
            &merged_iv,
            &merged_cv,
            false,
        )?;
        self.prune_class_versions(class_id)?;
        Ok(true)
    }

    /// Soft-delete: insert one more version identical to the latest except
    /// `deleted=1` — reuses the exact insert/prune path every other edit
    /// above uses, so [`undo_method`](Self::undo_method) needs *no changes
    /// at all* to "undo" a removal: it already just reverts to the version
    /// before this one (`docs/IMAGE.md` §4's revert-as-new-version rule).
    /// Returns `false` if the method doesn't exist or is already removed.
    pub fn remove_method(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<bool> {
        let Some(method_id) = self.method_id_of(class_name, side, selector)? else {
            return Ok(false);
        };
        let (category, source, deleted): (String, String, i64) = self.conn.query_row(
            "SELECT category, source, deleted FROM latest_method_versions WHERE method_id = ?1",
            params![method_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        if deleted != 0 {
            return Ok(false);
        }
        self.insert_method_version(method_id, &category, &source, true)?;
        self.prune_method_versions(method_id)?;
        Ok(true)
    }

    /// Soft-delete a class — mirrors [`remove_method`](Self::remove_method).
    /// Subclasses still naming this class as their superclass are left
    /// exactly as they are (`superclass_name` is a plain text field, not a
    /// foreign key): `package_roots` treats a removed superclass the same
    /// as a missing one, so they simply re-root visually rather than
    /// vanishing or erroring. Deliberately unconditional — the browser is
    /// expected to warn about affected subclasses *before* sending this,
    /// not have the store refuse or auto-reparent on its behalf.
    pub fn remove_class(&self, class_name: &str) -> rusqlite::Result<bool> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(false);
        };
        let (superclass, category, comment, ivars, cvars, deleted): (Option<String>, String, String, String, String, i64) = self.conn.query_row(
            "SELECT superclass_name, category, comment, instance_vars, class_vars, deleted FROM latest_class_versions WHERE class_id = ?1",
            params![class_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
        )?;
        if deleted != 0 {
            return Ok(false);
        }
        self.insert_class_version(
            class_id,
            superclass.as_deref(),
            &category,
            &comment,
            &ivars,
            &cvars,
            true,
        )?;
        self.prune_class_versions(class_id)?;
        Ok(true)
    }

    /// The class-browser "New Class" path (`../../gui/src/vm_host.rs`) —
    /// distinct from [`add_class`](Self::add_class), which the `.mst`
    /// importer uses and which assumes the caller already checked
    /// [`class_exists`](Self::class_exists) itself. This instead
    /// distinguishes three cases in one call: a genuinely new name
    /// (`Created`); a name that only exists as a removed tombstone
    /// (`Reopened` — inserts a fresh non-deleted version on the *same*
    /// `class_id`, the identical trick `src/bin/import_world.rs` already
    /// uses for legitimate class-reopening); and a name that's already live
    /// (`AlreadyLive`, refused — silently overwriting an unrelated live
    /// class by name collision would be surprising and destructive in a way
    /// reopening a removed one isn't).
    pub fn create_or_reopen_class(
        &self,
        name: &str,
        superclass: Option<&str>,
        category: &str,
        comment: &str,
        instance_vars: &str,
    ) -> rusqlite::Result<ClassCreateOutcome> {
        match self.class_id_of(name)? {
            None => {
                let load_order = self.next_load_order()?;
                self.add_class(
                    name,
                    superclass,
                    category,
                    comment,
                    instance_vars,
                    "",
                    load_order,
                )?;
                Ok(ClassCreateOutcome::Created)
            }
            Some(class_id) => {
                let (deleted, cvars): (i64, String) = self.conn.query_row(
                    "SELECT deleted, class_vars FROM latest_class_versions WHERE class_id = ?1",
                    params![class_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                if deleted == 0 {
                    return Ok(ClassCreateOutcome::AlreadyLive);
                }
                self.insert_class_version(
                    class_id,
                    superclass,
                    category,
                    comment,
                    instance_vars,
                    &cvars,
                    false,
                )?;
                self.prune_class_versions(class_id)?;
                Ok(ClassCreateOutcome::Reopened)
            }
        }
    }

    /// The class-browser "New Method" path. Ensures the method row exists
    /// (creating it if this selector/side is new on this class), then
    /// always inserts a fresh non-deleted version — collapsing "create,"
    /// "reopen a removed method," and "redefine a live method under the
    /// same selector" into one operation, since all three are the same
    /// ordinary Smalltalk action (accepting a method under some selector)
    /// and none of them should be an error, unlike the class-name-collision
    /// case `create_or_reopen_class` refuses. Returns `None` only if
    /// `class_name` doesn't exist at all.
    pub fn create_or_reopen_method(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
        category: &str,
        source: &str,
    ) -> rusqlite::Result<Option<i64>> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(None);
        };
        let method_id = match self.method_id_of(class_name, side, selector)? {
            Some(id) => id,
            None => {
                self.conn.execute(
                    "INSERT INTO methods (class_id, selector, side) VALUES (?1, ?2, ?3)",
                    params![class_id, selector, side.as_str()],
                )?;
                self.conn.last_insert_rowid()
            }
        };
        self.insert_method_version(method_id, category, source, false)?;
        self.prune_method_versions(method_id)?;
        Ok(Some(method_id))
    }

    /// Revert-as-new-version (`docs/IMAGE.md` §4) — restores the
    /// second-to-latest version's source as a *new* latest version, rather
    /// than deleting the latest. Returns `false` if there's nothing to undo
    /// (unknown method, or only one version on record). Carries that
    /// version's own `deleted` flag forward too (not hardcoded `false`) —
    /// this is what makes "undo" double as "un-remove" for free: undoing
    /// past a removal restores whichever `deleted` state the version being
    /// restored actually had.
    pub fn undo_method(
        &self,
        class_name: &str,
        side: Side,
        selector: &str,
    ) -> rusqlite::Result<bool> {
        let Some(method_id) = self.method_id_of(class_name, side, selector)? else {
            return Ok(false);
        };
        let previous: Option<(String, String, i64)> = self
            .conn
            .query_row(
                "SELECT category, source, deleted FROM method_versions WHERE method_id = ?1 \
                 ORDER BY version_number DESC LIMIT 1 OFFSET 1",
                params![method_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((category, source, deleted)) = previous else {
            return Ok(false);
        };
        self.insert_method_version(method_id, &category, &source, deleted != 0)?;
        self.prune_method_versions(method_id)?;
        Ok(true)
    }

    /// See `undo_method`'s doc comment — same "carry the restored version's
    /// own `deleted` flag forward" rule, so undo doubles as un-remove here
    /// too.
    pub fn undo_class(&self, class_name: &str) -> rusqlite::Result<bool> {
        let Some(class_id) = self.class_id_of(class_name)? else {
            return Ok(false);
        };
        let previous: Option<(Option<String>, String, String, String, String, i64)> = self
            .conn
            .query_row(
                "SELECT superclass_name, category, comment, instance_vars, class_vars, deleted FROM class_versions \
                 WHERE class_id = ?1 ORDER BY version_number DESC LIMIT 1 OFFSET 1",
                params![class_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
            )
            .optional()?;
        let Some((superclass, category, comment, ivars, cvars, deleted)) = previous else {
            return Ok(false);
        };
        self.insert_class_version(
            class_id,
            superclass.as_deref(),
            &category,
            &comment,
            &ivars,
            &cvars,
            deleted != 0,
        )?;
        self.prune_class_versions(class_id)?;
        Ok(true)
    }

    fn insert_method_version(
        &self,
        method_id: i64,
        category: &str,
        source: &str,
        deleted: bool,
    ) -> rusqlite::Result<()> {
        let next: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM method_versions WHERE method_id = ?1",
            params![method_id],
            |r| r.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO method_versions (method_id, version_number, category, source, edited_at, deleted) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![method_id, next, category, source, now_secs(), deleted as i64],
        )?;
        let version_id = self.conn.last_insert_rowid();
        // Persist the selectors this version SENDS, so "senders" is an accurate
        // SQL query, not a VM IC-scan (crate::mst::sent_selectors parses them
        // out of the source, keyword parts regrouped). A deleted tombstone
        // sends nothing.
        if !deleted {
            for selector in crate::mst::sent_selectors(source) {
                self.conn.execute(
                    "INSERT OR IGNORE INTO method_sends (method_version_id, selector) VALUES (?1, ?2)",
                    params![version_id, selector],
                )?;
            }
        }
        Ok(())
    }

    fn insert_class_version(
        &self,
        class_id: i64,
        superclass: Option<&str>,
        category: &str,
        comment: &str,
        ivars: &str,
        cvars: &str,
        deleted: bool,
    ) -> rusqlite::Result<()> {
        let next: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM class_versions WHERE class_id = ?1",
            params![class_id],
            |r| r.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO class_versions (class_id, version_number, superclass_name, category, comment, instance_vars, class_vars, edited_at, deleted) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![class_id, next, superclass, category, comment, ivars, cvars, now_secs(), deleted as i64],
        )?;
        Ok(())
    }

    /// `docs/IMAGE.md` §4 — sliding-window retention, applied after every
    /// insert (not a separate GC pass).
    fn prune_method_versions(&self, method_id: i64) -> rusqlite::Result<()> {
        self.conn.execute(
            "DELETE FROM method_versions WHERE method_id = ?1 AND version_number <= \
             (SELECT MAX(version_number) FROM method_versions WHERE method_id = ?1) - ?2",
            params![method_id, RETENTION_LIMIT],
        )?;
        Ok(())
    }

    fn prune_class_versions(&self, class_id: i64) -> rusqlite::Result<()> {
        self.conn.execute(
            "DELETE FROM class_versions WHERE class_id = ?1 AND version_number <= \
             (SELECT MAX(version_number) FROM class_versions WHERE class_id = ?1) - ?2",
            params![class_id, RETENTION_LIMIT],
        )?;
        Ok(())
    }

    // ── Load order (`docs/IMAGE.md` §3) ────────────────────────────────────

    /// `load_order` one step past the current maximum — for appending a
    /// class at the end of the load sequence (the common case: the
    /// importer loading `world.list` in order, or a brand new class).
    pub fn next_load_order(&self) -> rusqlite::Result<i64> {
        let max: Option<i64> =
            self.conn
                .query_row("SELECT MAX(load_order) FROM classes", [], |r| r.get(0))?;
        Ok(max.map(|m| m + LOAD_ORDER_STEP).unwrap_or(LOAD_ORDER_START))
    }

    /// Midpoint between two `load_order` values, for inserting a class
    /// between two existing ones. Returns `None` if the gap has been
    /// exhausted (adjacent values differing by ≤ 1) — call
    /// [`rebalance_load_order`] first in that case.
    pub fn load_order_between(before: i64, after: i64) -> Option<i64> {
        let mid = before + (after - before) / 2;
        if mid == before || mid == after {
            None
        } else {
            Some(mid)
        }
    }

    /// Renumber every class back to clean multiples of `LOAD_ORDER_STEP`,
    /// preserving current relative order — the maintenance operation
    /// `docs/IMAGE.md` §3 describes for when gaps run out. Not needed per
    /// edit; safe to call any time (idempotent on an already-clean table).
    pub fn rebalance_load_order(&self) -> rusqlite::Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT class_id FROM classes ORDER BY load_order")?;
        let ids: Vec<i64> = stmt
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for (i, class_id) in ids.into_iter().enumerate() {
            let new_order = LOAD_ORDER_START + (i as i64) * LOAD_ORDER_STEP;
            self.conn.execute(
                "UPDATE classes SET load_order = ?1 WHERE class_id = ?2",
                params![new_order, class_id],
            )?;
        }
        Ok(())
    }

    pub fn reorder_class(&self, class_name: &str, new_load_order: i64) -> rusqlite::Result<bool> {
        let changed = self.conn.execute(
            "UPDATE classes SET load_order = ?1 WHERE name = ?2",
            params![new_load_order, class_name],
        )?;
        Ok(changed > 0)
    }
}

/// Union two space-separated variable lists: `existing` order preserved, plus
/// any names in `additions` not already present. The world importer uses this
/// to MERGE (not replace) a reopened class's declared vars.
fn union_space_separated(existing: &str, additions: &str) -> String {
    let mut out: Vec<&str> = existing.split_whitespace().collect();
    for a in additions.split_whitespace() {
        if !out.contains(&a) {
            out.push(a);
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cvars_of(img: &Image, name: &str) -> String {
        img.all_classes()
            .unwrap()
            .into_iter()
            .find(|c| c.name == name)
            .unwrap()
            .class_vars
    }

    #[test]
    fn reimport_class_shell_merges_class_vars_without_wiping() {
        let img = Image::open_in_memory().unwrap();
        let lo = img.next_load_order().unwrap();
        // First seeded WITHOUT class vars (the M3 GamePane situation).
        img.add_class("Widget", Some("Object"), "GUI", "", "", "", lo)
            .unwrap();

        // A reopen that adds `<classVars: …>` ADDS them (the GamePane fix).
        assert!(img.reimport_class_shell("Widget", "", "State Count").unwrap());
        assert_eq!(cvars_of(&img, "Widget"), "State Count");

        // A method-only reopen (no pragma -> empty vars) must NOT wipe them —
        // the Character regression. An empty addition merges to a no-op.
        assert!(!img.reimport_class_shell("Widget", "", "").unwrap());
        assert_eq!(cvars_of(&img, "Widget"), "State Count");

        // Adding a further var unions it in (order preserved).
        assert!(img.reimport_class_shell("Widget", "", "Total").unwrap());
        assert_eq!(cvars_of(&img, "Widget"), "State Count Total");
    }

    fn seeded() -> Image {
        let img = Image::open_in_memory().unwrap();
        let lo = img.next_load_order().unwrap();
        img.add_class(
            "Object",
            None,
            "Kernel",
            "The root of the hierarchy.",
            "",
            "",
            lo,
        )
        .unwrap();
        let lo = img.next_load_order().unwrap();
        img.add_class(
            "Collection",
            Some("Object"),
            "Collections",
            "Abstract collection.",
            "",
            "",
            lo,
        )
        .unwrap();
        img.add_method(
            "Object",
            Side::Instance,
            "printString",
            "printing",
            "printString\n\t^'an Object'",
        )
        .unwrap();
        img.add_method(
            "Object",
            Side::Instance,
            "hash",
            "comparing",
            "hash\n\t^self identityHash",
        )
        .unwrap();
        img
    }

    #[test]
    fn packages_and_roots_round_trip() {
        let img = seeded();
        assert_eq!(
            img.packages().unwrap(),
            vec!["Kernel".to_string(), "Collections".to_string()]
        );
        let roots = img.package_roots("Kernel").unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].name, "Object");
    }

    #[test]
    fn subclasses_and_categories_and_methods() {
        let img = seeded();
        let subs = img.subclasses_of("Object").unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].name, "Collection");

        let cats = img.categories("Object", Side::Instance).unwrap();
        assert!(cats.contains(&"printing".to_string()));
        assert!(cats.contains(&"comparing".to_string()));

        let methods = img
            .methods_in("Object", Side::Instance, "printing")
            .unwrap();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].selector, "printString");
    }

    #[test]
    fn set_method_source_versions_instead_of_overwriting() {
        let img = seeded();
        assert!(img
            .set_method_source("Object", Side::Instance, "hash", "hash\n\t^42")
            .unwrap());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            "hash\n\t^42"
        );
        // Unknown selector: no auto-create.
        assert!(!img
            .set_method_source("Object", Side::Instance, "nope", "x")
            .unwrap());
    }

    #[test]
    fn undo_method_restores_previous_source_as_a_new_version() {
        let img = seeded();
        img.set_method_source("Object", Side::Instance, "hash", "hash\n\t^42")
            .unwrap();
        assert!(img.undo_method("Object", Side::Instance, "hash").unwrap());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            "hash\n\t^self identityHash"
        );
        // The "undo" is itself a new version — undo-the-undo should work too.
        assert!(img.undo_method("Object", Side::Instance, "hash").unwrap());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            "hash\n\t^42"
        );
    }

    #[test]
    fn undo_with_only_one_version_reports_false() {
        let img = seeded();
        assert!(!img.undo_method("Object", Side::Instance, "hash").unwrap());
    }

    #[test]
    fn set_class_comment_carries_other_fields_forward() {
        let img = seeded();
        assert!(img
            .set_class_comment("Collection", "Updated comment.")
            .unwrap());
        assert_eq!(
            img.class_comment("Collection").unwrap().unwrap(),
            "Updated comment."
        );
        // superclass/category must survive the edit unchanged.
        let roots = img.package_roots("Collections").unwrap();
        assert_eq!(roots[0].superclass.as_deref(), Some("Object"));
    }

    #[test]
    fn retention_keeps_at_most_the_limit() {
        let img = seeded();
        for i in 0..(RETENTION_LIMIT + 5) {
            img.set_method_source("Object", Side::Instance, "hash", &format!("hash\n\t^{i}"))
                .unwrap();
        }
        let count: i64 = img
            .conn
            .query_row(
                "SELECT COUNT(*) FROM method_versions WHERE method_id = (SELECT method_id FROM methods WHERE selector='hash')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, RETENTION_LIMIT);
        // And the latest surviving value is still correct.
        let last = RETENTION_LIMIT + 4;
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            format!("hash\n\t^{last}")
        );
    }

    #[test]
    fn load_order_gap_numbering() {
        let img = Image::open_in_memory().unwrap();
        assert_eq!(img.next_load_order().unwrap(), LOAD_ORDER_START);
        img.add_class("A", None, "P", "", "", "", 1000).unwrap();
        assert_eq!(img.next_load_order().unwrap(), 1100);
        img.add_class("B", None, "P", "", "", "", 1100).unwrap();
        assert_eq!(Image::load_order_between(1000, 1100), Some(1050));
        img.add_class("C", None, "P", "", "", "", 1050).unwrap();
        assert_eq!(Image::load_order_between(1000, 1050), Some(1025));
        // Exhausted gap.
        assert_eq!(Image::load_order_between(1000, 1001), None);
    }

    #[test]
    fn rebalance_restores_clean_gaps() {
        let img = Image::open_in_memory().unwrap();
        img.add_class("A", None, "P", "", "", "", 1000).unwrap();
        img.add_class("B", None, "P", "", "", "", 1001).unwrap();
        img.rebalance_load_order().unwrap();
        assert_eq!(Image::load_order_between(1000, 1100), Some(1050));
    }

    #[test]
    fn reorder_class_updates_load_order() {
        let img = seeded();
        assert!(img.reorder_class("Collection", 500).unwrap());
        let roots = img.package_roots("Kernel").unwrap();
        // Object is still Kernel's only root; just confirm reorder didn't error
        // and the class is still queryable at its new position.
        assert_eq!(roots[0].name, "Object");
        assert!(!img.reorder_class("Nonexistent", 999).unwrap());
    }

    #[test]
    fn remove_method_hides_it_and_undo_restores_it() {
        let img = seeded();
        assert!(img.remove_method("Object", Side::Instance, "hash").unwrap());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash").unwrap(),
            None
        );
        assert!(!img
            .categories("Object", Side::Instance)
            .unwrap()
            .contains(&"comparing".to_string()));
        // Already removed: a second remove is a no-op-false, not an error.
        assert!(!img.remove_method("Object", Side::Instance, "hash").unwrap());
        // Unknown selector.
        assert!(!img.remove_method("Object", Side::Instance, "nope").unwrap());

        // Undo un-removes it — no dedicated "unremove" needed.
        assert!(img.undo_method("Object", Side::Instance, "hash").unwrap());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            "hash\n\t^self identityHash"
        );
    }

    #[test]
    fn remove_class_hides_it_and_reroots_subclasses() {
        let img = seeded();
        assert!(img.remove_class("Object").unwrap());
        assert!(
            !img.packages().unwrap().contains(&"Kernel".to_string()),
            "Kernel had only Object"
        );
        assert_eq!(img.class_comment("Object").unwrap(), None);
        // Collection's superclass_name still literally says "Object", but a
        // removed superclass counts as absent for the package-roots check —
        // Collection re-roots in "Collections" instead of vanishing.
        let roots = img.package_roots("Collections").unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].name, "Collection");

        assert!(!img.remove_class("Object").unwrap()); // already removed
        assert!(!img.remove_class("Nonexistent").unwrap());

        assert!(img.undo_class("Object").unwrap());
        assert!(img.packages().unwrap().contains(&"Kernel".to_string()));
    }

    #[test]
    fn create_or_reopen_class_distinguishes_new_live_and_removed() {
        let img = seeded();
        assert_eq!(
            img.create_or_reopen_class("Stream", Some("Object"), "Streams", "A stream.", "")
                .unwrap(),
            ClassCreateOutcome::Created
        );
        assert!(img.class_comment("Stream").unwrap().is_some());

        // Already live: refused, not silently overwritten.
        assert_eq!(
            img.create_or_reopen_class("Object", None, "Kernel", "Hijacked!", "")
                .unwrap(),
            ClassCreateOutcome::AlreadyLive
        );
        assert_eq!(
            img.class_comment("Object").unwrap().unwrap(),
            "The root of the hierarchy."
        );

        // Removed, then recreated under the same name: reopened, not a
        // UNIQUE(name) constraint error.
        img.remove_class("Collection").unwrap();
        assert_eq!(
            img.create_or_reopen_class(
                "Collection",
                Some("Object"),
                "Collections",
                "Reborn.",
                "elements"
            )
            .unwrap(),
            ClassCreateOutcome::Reopened
        );
        assert_eq!(img.class_comment("Collection").unwrap().unwrap(), "Reborn.");
    }

    #[test]
    fn create_or_reopen_method_creates_redefines_and_reopens() {
        let img = seeded();
        // Brand new selector.
        assert!(img
            .create_or_reopen_method(
                "Object",
                Side::Instance,
                "printOn:",
                "printing",
                "printOn: s\n\t^s"
            )
            .unwrap()
            .is_some());
        assert_eq!(
            img.method_source("Object", Side::Instance, "printOn:")
                .unwrap()
                .unwrap(),
            "printOn: s\n\t^s"
        );

        // Redefining an already-live selector via this path is normal, not
        // an error (unlike the class case).
        assert!(img
            .create_or_reopen_method("Object", Side::Instance, "hash", "comparing", "hash\n\t^0")
            .unwrap()
            .is_some());
        assert_eq!(
            img.method_source("Object", Side::Instance, "hash")
                .unwrap()
                .unwrap(),
            "hash\n\t^0"
        );

        // Removed, then reopened under the same selector.
        img.remove_method("Object", Side::Instance, "printString")
            .unwrap();
        assert!(img
            .create_or_reopen_method(
                "Object",
                Side::Instance,
                "printString",
                "printing",
                "printString\n\t^'reborn'"
            )
            .unwrap()
            .is_some());
        assert_eq!(
            img.method_source("Object", Side::Instance, "printString")
                .unwrap()
                .unwrap(),
            "printString\n\t^'reborn'"
        );

        // Unknown class.
        assert!(img
            .create_or_reopen_method("Nonexistent", Side::Instance, "foo", "cat", "foo")
            .unwrap()
            .is_none());
    }

    #[test]
    fn set_class_definition_changes_superclass_and_ivars_only() {
        let img = seeded();
        assert!(img
            .set_class_definition("Collection", Some("Object"), "elements size")
            .unwrap());
        let roots = img.package_roots("Collections").unwrap();
        assert_eq!(roots[0].name, "Collection");
        // comment and category must survive the definition edit unchanged.
        assert_eq!(
            img.class_comment("Collection").unwrap().unwrap(),
            "Abstract collection."
        );
        assert!(img.packages().unwrap().contains(&"Collections".to_string()));
        assert!(!img.set_class_definition("Nonexistent", None, "").unwrap());
    }

    #[test]
    fn class_named_reads_one_class_or_none() {
        let img = seeded();
        let c = img.class_named("Collection").unwrap().expect("live class");
        assert_eq!(c.superclass.as_deref(), Some("Object"));
        assert_eq!(c.category, "Collections");
        assert!(img.class_named("Nonexistent").unwrap().is_none());
    }

    /// The text editor's fetch contract (`docs/editor_design.md` §5): the
    /// rendered class must PARSE BACK to exactly the method sources the image
    /// holds. This is what makes "open a class and accept it unchanged" a
    /// no-op instead of a churn of version bumps — the accept path diffs
    /// parsed-vs-stored, so any drift here would rewrite every method on every
    /// accept, invalidating every nmethod and polluting the version history.
    #[test]
    fn class_source_round_trips_through_the_parser() {
        // Real MACVM `.mst` shape — a BRACKETED method body, indented and
        // captured verbatim, exactly as `import_world_dir` stores it (a real
        // one from the image: `escapeAtRe: cr im: ci [\n        | zr .. |\n
        // ..\n    ]`). The `seeded()` fixture's bare `selector\n\tbody` form
        // is Squeak chunk format and is not a shape this world ever holds, so
        // it cannot exercise the round trip.
        let img = Image::open_in_memory().unwrap();
        let lo = img.next_load_order().unwrap();
        img.add_class("Object", None, "Kernel", "The root.", "", "", lo)
            .unwrap();
        let lo = img.next_load_order().unwrap();
        img.add_class("Collection", Some("Object"), "Collections", "Abstract.", "", "", lo)
            .unwrap();
        img.add_method(
            "Collection",
            Side::Instance,
            "do:",
            "enumerating",
            "do: aBlock [\n        \"iterate — a comment with a ] bracket\"\n        ^self subclassResponsibility\n    ]",
        )
        .unwrap();
        img.add_method(
            "Collection",
            Side::Class,
            "new:",
            "instance creation",
            "Collection class >> new: n [\n        ^self basicNew\n    ]",
        )
        .unwrap();
        img.set_class_definition("Collection", Some("Object"), "elements size")
            .unwrap();

        let text = img.class_source("Collection").unwrap().expect("live class");
        let parsed = crate::mst::parse_mst_source(&text);
        assert_eq!(parsed.len(), 1, "one class block, got: {text}");
        let pc = &parsed[0];
        assert_eq!(pc.name, "Collection");
        assert_eq!(pc.superclass.as_deref(), Some("Object"));

        // Every stored method reappears with a BYTE-IDENTICAL source.
        for stored in img.all_methods_of("Collection").unwrap() {
            let want_class_side = stored.side == Side::Class;
            let got = pc
                .methods
                .iter()
                .find(|m| m.selector == stored.selector && m.is_class_side == want_class_side)
                .unwrap_or_else(|| panic!("{} missing from:\n{text}", stored.selector));
            assert_eq!(
                got.source, stored.source,
                "{} did not round-trip", stored.selector
            );
        }
        assert_eq!(
            pc.methods.len(),
            img.all_methods_of("Collection").unwrap().len(),
            "no extra or dropped methods"
        );

        assert!(img.class_source("Nonexistent").unwrap().is_none());
    }
}
