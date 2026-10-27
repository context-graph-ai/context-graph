//! Fabric  — the frozen source rule: no transport name in ledger/payload/job/apply code.
//!
//! The ledger treats cg's payload as opaque and knows no transport. The authenticated endpoint
//! belongs to the `[fabric]` config surface and the contextdb adapter — never to cg's submission,
//! payload, job, apply, or sync-declaration sources. That path contains no transport identifiers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use context_graph::ingest::{
    RuntimeModelInfo, changed_since_context_cursor, register_runtime_model_info,
};
use context_graph::schema::TABLES;
use context_graph::{CreateContext, StoreConfig};

use crate::fake_extractor::open_store_with_lexical_embedder;

const DUPLICATE_SYNC_AUTHORITY_MARKERS: &[&str] = &[
    "context_graph_sync_table_contract",
    "ContextGraphSyncTablePolicy",
    "sync_policy_for_table",
    "context_graph_sync_conflict_policies",
    "register_context_graph_sync_client",
    "ContextGraphSyncRegistration",
    "work_ledger_conflict_policy_entries",
    "LOCAL_ONLY_RUNTIME_SYNC_TABLES",
    "ConflictPolicies",
    ".set_default_conflict_policy(",
    ".set_table_direction(",
    ".set_conflict_policy(",
];

#[test]
fn no_transport_name_in_ledger_or_payload_sources() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    // The ledger/payload/job/apply layer of cg — the code that must name no transport.
    let scanned = [
        "fabric.rs",
        "tier0.rs",
        "understanding.rs",
        "harness.rs",
        "capture_sync_status.rs",
    ];
    let removed_transport = ["na", "ts"].concat();
    let mut offenders = Vec::new();
    for file in scanned {
        let path = src.join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .to_ascii_lowercase();
        for name in ["iroh", removed_transport.as_str()] {
            if text.contains(name) {
                offenders.push(format!("{file} names transport `{name}`"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "transport identifiers leaked into ledger/payload/job/apply sources: {offenders:?}"
    );
}

#[test]
fn context_graph_tables_declare_the_frozen_sync_contract() {
    const KEEP_LATEST: &[&str] = &[
        "intentions",
        "decisions",
        "entities",
        "reindex_cursors",
        "outcomes",
        "invalidations",
        "entity_current_snapshots",
        "intention_current_snapshots",
        "cg_learned_skills",
    ];
    const KEEP_FIRST: &[&str] = &[
        "cg_schema_version",
        "contexts",
        "observations",
        "evidence",
        "embedding_spaces",
        "embedding_space_bindings",
        "evidence_embedding_handles",
        "entity_reference_embedding_handles",
        "entity_reference_embeddings_128",
        "entity_reference_embeddings_256",
        "entity_reference_embeddings_512",
        "entity_reference_embeddings_768",
        "evidence_availability_log",
        "signatures",
        "signature_aggregates",
        "signature_producer_lineage",
        "digests",
        "digest_embedding_handles",
        "blueprints",
        "edges",
        "entity_snapshots",
        "entity_snapshot_observations",
        "decision_basis_snapshots",
        "entity_snapshot_links",
        "intention_snapshots",
        "intention_snapshot_links",
        "decision_intention_citations",
        "approvals",
        "conflicts",
        "precedent_matches",
        "cg_teaching_events",
        "cg_overlay_entries",
        "cg_compensation_events",
    ];
    // This fixture opens the default LocalOnly Store. The approved event-placement policy
    // resolves these immutable shared rows to SYNC OFF locally while retaining their
    // first-writer arbitration; authenticated TenantHub binding is the only route that
    // changes their direction to SYNC PUSH ONLY.
    const LOCAL_ONLY_KEEP_FIRST: &[&str] = &["events", "event_anchors"];
    const SYNC_OFF: &[&str] = &[
        "audit_log",
        "cg_skill_firing_trace",
        "cg_review_cursor",
        "cg_pending_review_items",
        "cg_source_chunks",
        "cg_retrieval_touches",
        "cg_understanding_completions",
        "cg_local_settings",
        "cg_changed_cursors",
        "cg_runtime_models",
        "context_bindings",
        "event_placement",
        "event_local_commits",
        "event_source_checkpoints",
        "event_source_locators",
        "event_source_ordering",
    ];
    const ENGINE_OWNED: &[&str] = &[
        "work_jobs",
        "work_claims",
        "work_results",
        "work_failures",
        "work_cancellations",
        "work_inputs",
        "work_capabilities",
    ];

    assert_eq!(KEEP_LATEST.len(), 9);
    assert_eq!(KEEP_FIRST.len(), 33);
    assert_eq!(LOCAL_ONLY_KEEP_FIRST.len(), 2);
    assert_eq!(SYNC_OFF.len(), 16);
    assert_eq!(ENGINE_OWNED.len(), 7);

    let mut expected = BTreeMap::new();
    for table in KEEP_LATEST {
        assert_eq!(
            expected.insert(*table, ("SYNC TWO WAY", Some("SYNC CONFLICT KEEP LATEST")),),
            None,
            "table {table} must have one declaration"
        );
    }
    for table in KEEP_FIRST {
        assert_eq!(
            expected.insert(*table, ("SYNC TWO WAY", Some("SYNC CONFLICT KEEP FIRST")),),
            None,
            "table {table} must have one declaration"
        );
    }
    for table in LOCAL_ONLY_KEEP_FIRST {
        assert_eq!(
            expected.insert(*table, ("SYNC OFF", Some("SYNC CONFLICT KEEP FIRST")),),
            None,
            "table {table} must have one declaration"
        );
    }
    for table in SYNC_OFF {
        assert_eq!(
            expected.insert(*table, ("SYNC OFF", None)),
            None,
            "table {table} must have one declaration"
        );
    }

    let schema_tables = TABLES
        .iter()
        .map(|table| table.name)
        .collect::<BTreeSet<_>>();
    let expected_schema_tables = expected
        .keys()
        .copied()
        .filter(|table| !matches!(*table, "cg_changed_cursors" | "cg_runtime_models"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        schema_tables, expected_schema_tables,
        "the frozen mapping must classify every Context Graph schema table exactly once"
    );
    for table in ENGINE_OWNED {
        assert!(
            !schema_tables.contains(table),
            "{table} belongs to the engine and must not acquire a Context Graph declaration"
        );
    }

    let dir = tempfile::tempdir().expect("sync-contract tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("contract.db"),
        ..Default::default()
    });
    let context = store
        .create_context(CreateContext {
            name: "contract-inspection".to_string(),
            ..Default::default()
        })
        .expect("create context for runtime-table installation");
    changed_since_context_cursor(&store, context.id)
        .expect("install and use the changed-cursor table");
    register_runtime_model_info(
        &store,
        RuntimeModelInfo {
            context_label: "contract-inspection".to_string(),
            model_key: "hosted-test-model".to_string(),
            advertised_context_tokens: 1,
            default_for_context: true,
        },
    )
    .expect("install and use the runtime-model table");

    let db = store.sync_database();
    let mut mismatches = Vec::new();
    for (table, (expected_direction, expected_conflict)) in expected {
        let Some(meta) = db.table_meta(table) else {
            mismatches.push(format!("{table}: table missing"));
            continue;
        };
        let actual_direction = meta.sync_direction.map(|direction| direction.sql());
        let actual_conflict = meta
            .conflict_policy
            .and_then(|policy| policy.declared_clause());
        if actual_direction != Some(expected_direction) || actual_conflict != expected_conflict {
            mismatches.push(format!(
                "{table}: expected {expected_direction} + {expected_conflict:?}, got {actual_direction:?} + {actual_conflict:?}"
            ));
        }
    }
    for table in ENGINE_OWNED {
        assert!(
            db.table_meta(table).is_some(),
            "{table} must still be installed by ContextDB"
        );
    }

    let manifest_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source_root = manifest_root.join("src");
    let cli_source_root = manifest_root.join("../cg-cli/src");
    let mut production_sources = rust_sources_under(&source_root);
    production_sources.extend(rust_sources_under(&cli_source_root));
    for removed_symbol in DUPLICATE_SYNC_AUTHORITY_MARKERS {
        let offenders = production_sources
            .iter()
            .filter(|(_, source)| source_uses_removed_symbol(source, removed_symbol))
            .map(|(path, _)| path.display().to_string())
            .collect::<Vec<_>>();
        if !offenders.is_empty() {
            mismatches.push(format!(
                "production still contains duplicate sync authority `{removed_symbol}` in {}",
                offenders.join(", ")
            ));
        }
    }
    let status_source = production_sources
        .iter()
        .find(|(path, _)| path.ends_with("capture_sync_status.rs"))
        .map(|(_, source)| source.as_str())
        .expect("capture_sync_status.rs is in the production source inventory");
    let pending_count = rust_function_body(
        status_source,
        "pub fn context_graph_pending_push_change_count",
    );
    for obsolete_map_mechanic in [
        "filter_by_direction",
        "context_graph_sync_table_contract",
        "LOCAL_ONLY_RUNTIME_SYNC_TABLES",
        "work_ledger_conflict_policy_entries",
        "directions",
    ] {
        if pending_count.contains(obsolete_map_mechanic) {
            mismatches.push(format!(
                "pending-count production code still rebuilds a direction override through `{obsolete_map_mechanic}`"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "Context Graph sync declarations differ from the frozen table contract:\n{}",
        mismatches.join("\n")
    );
}

/// Does production source use an unsupported sync-authority symbol?
///
/// Identifier-shaped entries count only on identifier boundaries, so they do not fire inside a
/// larger identifier that merely ends in the same letters. Call-shaped entries
/// (`.set_conflict_policy(`) keep the plain substring scan.
fn source_uses_removed_symbol(source: &str, removed_symbol: &str) -> bool {
    fn is_identifier_char(character: char) -> bool {
        character.is_ascii_alphanumeric() || character == '_'
    }

    let identifier_shaped = removed_symbol.chars().all(is_identifier_char);
    if !identifier_shaped {
        return source.contains(removed_symbol);
    }
    source.match_indices(removed_symbol).any(|(offset, _)| {
        let preceded_by_identifier = source[..offset]
            .chars()
            .next_back()
            .is_some_and(is_identifier_char);
        let followed_by_identifier = source[offset + removed_symbol.len()..]
            .chars()
            .next()
            .is_some_and(is_identifier_char);
        !preceded_by_identifier && !followed_by_identifier
    })
}

#[test]
fn sync_authority_scanner_allows_engine_metadata_reads_and_rejects_duplicate_authority() {
    assert!(
        !DUPLICATE_SYNC_AUTHORITY_MARKERS.iter().any(|marker| {
            source_uses_removed_symbol(
                "meta.sync_direction == Some(SyncDirection::Push) && meta.conflict_policy == Some(ConflictPolicy::KEEP_FIRST)",
                marker,
            )
        }),
        "ContextDB metadata comparisons inspect the declared policy; they do not create a local authority"
    );
    assert!(
        DUPLICATE_SYNC_AUTHORITY_MARKERS.iter().any(|marker| {
            source_uses_removed_symbol(
                "builder.set_conflict_policy(ConflictPolicy::KEEP_FIRST)",
                marker,
            )
        }),
        "a local setter remains duplicate sync authority"
    );
    assert!(
        DUPLICATE_SYNC_AUTHORITY_MARKERS.iter().any(|marker| {
            source_uses_removed_symbol("struct ContextGraphSyncTablePolicy;", marker)
        }),
        "a local policy definition remains duplicate sync authority"
    );
}

fn rust_sources_under(root: &Path) -> Vec<(PathBuf, String)> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(directory)
            .unwrap_or_else(|err| panic!("read production source directory {directory:?}: {err}"))
        {
            let path = entry.expect("read production source entry").path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    visit(root, &mut files);
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read production source {path:?}: {err}"));
            (path, source)
        })
        .collect()
}

fn rust_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("production function `{signature}` must exist"));
    let body_start = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function body starts");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("production function `{signature}` has an unterminated body")
}
