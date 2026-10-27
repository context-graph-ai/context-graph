//! Public store methods leave the placeholder signature, blueprint and approval tables unused.
//!
//! The source rule scans public methods on Store, ScopedStore and ConsumerGraphView. Free
//! functions and other types are outside its entry-point scope. It follows identifier references
//! transitively into functions/constants in the same module and every store module; duplicate
//! names are conservatively all followed. Comments and cfg(test) items are excluded. Normal,
//! byte and raw string literals are checked for SQL FROM, JOIN, INTO, UPDATE or USING naming
//! exactly signatures, blueprints or approvals (not similarly prefixed table names). CREATE
//! TABLE declarations and generic caller-supplied SQL do not constitute a named table API.
//! Rust syntax is parsed in process; macro bodies retain conservative lexical references.
//! Dynamic identifiers, macro expansion and dependencies outside this source tree are outside
//! this lexical rule; the source scan is not an SQL authorization boundary.

use std::path::Path;
#[path = "common/placeholder_api_scan.rs"]
mod placeholder_api_scan;

/// No public store API or its local helpers names a placeholder table in read or write SQL.
#[test]
fn no_public_store_method_names_the_signature_blueprint_or_approval_tables() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    // Positive and negative controls use the same scanner before scanning the shipped API.
    let fixture = tempfile::tempdir().expect("source scanner fixture");
    let src = fixture.path().join("crates/context-graph/src/store");
    std::fs::create_dir_all(&src).expect("fixture source directory");
    std::fs::write(
        src.join("mod.rs"),
        r###"
impl Store {
    pub fn reads_signature(&self) { private_read(); }
    pub fn harmless(&self) { execute("SELECT id FROM signature_aggregates"); }
}
impl<'s> ScopedStore<'s> {
    pub fn writes_blueprint(&self) { execute(r#"INSERT INTO blueprints (id) VALUES (1)"#); }
}
impl<'reader> crate::ConsumerGraphView<'reader> {
    pub fn reads_approval(&self) { execute("SELECT id FROM approvals"); }
}
fn private_read() { execute(SQL); }
const SQL: &str = "SELECT id FROM signatures";
pub fn free_function() { execute("SELECT id FROM signatures"); }
impl Other { pub fn other_type(&self) { execute("SELECT id FROM signatures"); } }
// pub fn comment_only() { execute("SELECT id FROM signatures"); }
#[cfg(test)]
mod tests { impl Store { pub fn test_only(&self) { execute("SELECT id FROM signatures"); } } }
"###,
    )
    .expect("fixture public and private items");
    let hits = placeholder_api_scan::scan(fixture.path());
    assert_eq!(hits.len(), 3, "exactly three named-table APIs: {hits:?}");
    assert_eq!(
        hits.iter()
            .map(|hit| hit["table"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["approvals", "signatures", "blueprints"]
    );
    assert_eq!(
        hits[1]["chain"],
        serde_json::json!(["reads_signature", "private_read", "SQL"])
    );
    // Include every Unicode White_Space character and the extra separators Python's
    // whitespace rule recognizes. The ordinary source control retains literal U+00A0.
    let unicode_control = "impl Store { pub fn read_signature(&self) { execute(\"SELECT id FROM\u{a0}signatures\"); } }\n";
    std::fs::write(src.join("mod.rs"), unicode_control).expect("Unicode whitespace source control");
    let hits = placeholder_api_scan::scan(fixture.path());
    assert_eq!(
        hits.len(),
        1,
        "the literal U+00A0 reference must be detected: {hits:?}"
    );
    assert_eq!(hits[0]["table"], "signatures");
    for whitespace in [
        '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ',
        '\u{85}', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}',
        '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}',
        '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}',
    ] {
        for keyword in ["FROM", "JOIN", "INTO", "UPDATE", "USING"] {
            let sql = format!("{keyword}{whitespace}signatures");
            std::fs::write(
                src.join("mod.rs"),
                format!("impl Store {{ pub fn read_signature(&self) {{ execute({sql:?}); }} }}\n"),
            )
            .expect("whitespace variant source");
            let hits = placeholder_api_scan::scan(fixture.path());
            assert_eq!(
                hits.len(),
                1,
                "{keyword} with U+{:04X}: {hits:?}",
                u32::from(whitespace)
            );
            assert_eq!(hits[0]["api"], "read_signature");
            assert_eq!(hits[0]["table"], "signatures");
        }
    }
    println!(
        "The U+00A0 source control and all 29 whitespace characters match each of five SQL keywords."
    );
    let violations = placeholder_api_scan::scan(root);
    assert!(
        violations.is_empty(),
        "public store methods name placeholder tables:\n{}",
        serde_json::to_string_pretty(&violations).unwrap()
    );
}
