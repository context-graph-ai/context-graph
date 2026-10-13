//! Public memory APIs leave the placeholder signature, blueprint and approval tables unused.
//!
//! The source rule scans every public function, constant and static in the core, including
//! Store, ScopedStore, ConsumerGraphView and free functions. It follows identifier references
//! transitively into functions/constants in the same module and every store module; duplicate
//! names are conservatively all followed. Comments and cfg(test) items are excluded. Normal,
//! byte and raw string literals are checked for SQL FROM, JOIN, INTO, UPDATE or USING naming
//! exactly signatures, blueprints or approvals (not similarly prefixed table names). CREATE
//! TABLE declarations and generic caller-supplied SQL do not constitute a named table API.
//! Dynamic identifiers, macro expansion and dependencies outside this source tree are outside
//! this lexical rule; the source scan is not an SQL authorization boundary.

use std::path::Path;
use std::process::Command;

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
pub fn reads_signature() { private_read(); }
fn private_read() { execute(SQL); }
const SQL: &str = "SELECT id FROM signatures";
pub fn writes_blueprint() { execute(r#"INSERT INTO blueprints (id) VALUES (1)"#); }
pub fn reads_approval() { execute("SELECT id FROM approvals"); }
pub fn harmless() { execute("SELECT id FROM signature_aggregates"); }
// pub fn comment_only() { execute("SELECT id FROM signatures"); }
#[cfg(test)]
mod tests { pub fn test_only() { execute("SELECT id FROM signatures"); } }
"###,
    )
    .expect("fixture public and private items");
    let control = Command::new("python3")
        .arg(root.join("crates/context-graph/tests/fixtures/placeholder_api_scan.py"))
        .arg(fixture.path())
        .output()
        .expect("run source scanner controls");
    assert!(
        control.status.success(),
        "{}",
        String::from_utf8_lossy(&control.stderr)
    );
    let hits: Vec<serde_json::Value> =
        serde_json::from_slice(&control.stdout).expect("control hits");
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
    let output = Command::new("python3")
        .arg(root.join("crates/context-graph/tests/fixtures/placeholder_api_scan.py"))
        .arg(root)
        .output()
        .expect("scan public Rust items and their local dependencies");
    assert!(
        output.status.success(),
        "source scanner failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let violations: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)
        .expect("source scan returns exact API, source line, table and call chain");
    assert!(
        violations.is_empty(),
        "public APIs name placeholder tables:\n{}",
        serde_json::to_string_pretty(&violations).unwrap()
    );
}
