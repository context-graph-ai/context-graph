use context_graph::ingest::{
    CaptureRequest, HarvestCounts, dry_run_import, import_conversation_text_with_extractor,
    record_conversation, record_conversation_with_extractor,
};
use context_graph::{DecisionId, DecisionStatus, DiscoveryFilter, IntentionId, Store, StoreConfig};
use contextdb_core::Value;
use contextdb_engine::Database;

use super::fake_extractor::{LabeledTranscriptExtractor, open_store_with_lexical_embedder};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImportCounts {
    contexts: usize,
    digests: usize,
    intentions: usize,
    decisions: usize,
    observations: usize,
    extracted_from_edges: usize,
}

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn persisted_counts(db_path: &std::path::Path) -> ImportCounts {
    let db = Database::open(db_path).expect("raw db");
    let snapshot = db.snapshot();
    let extracted_from_edges = db
        .scan("edges", snapshot)
        .expect("scan edges")
        .into_iter()
        .filter(|row| {
            matches!(
                row.values.get("edge_type"),
                Some(Value::Text(edge_type)) if edge_type == "EXTRACTED_FROM"
            )
        })
        .count();
    ImportCounts {
        contexts: db.scan("contexts", snapshot).expect("scan contexts").len(),
        digests: db.scan("digests", snapshot).expect("scan digests").len(),
        intentions: db
            .scan("intentions", snapshot)
            .expect("scan intentions")
            .len(),
        decisions: db
            .scan("decisions", snapshot)
            .expect("scan decisions")
            .len(),
        observations: db
            .scan("observations", snapshot)
            .expect("scan observations")
            .len(),
        extracted_from_edges,
    }
}

fn text_values(db_path: &std::path::Path, table: &str, column: &str) -> Vec<String> {
    let db = Database::open(db_path).expect("raw db");
    let mut values = db
        .scan(table, db.snapshot())
        .expect("scan table")
        .into_iter()
        .filter_map(|row| match row.values.get(column) {
            Some(Value::Text(value)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn stable_uuid(report: &context_graph::ingest::CaptureReport, keys: &[&str]) -> Option<uuid::Uuid> {
    keys.iter()
        .find_map(|key| report.stable_ids.get(*key))
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
}

#[test]
fn ingest_one_time_batch_import() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let path = dir.path().join("session.txt");
    std::fs::write(
        &path,
        [
            "Context: acme-import",
            "Goal: ship password-less login for acme.",
            "Decision: auth module at src/auth.py.",
            "Decision: use JWT with 15-min access tokens.",
            "Observation: auth module file observed at src/auth.py.",
        ]
        .join("\n"),
    )
    .expect("write fixture");

    let transcript = std::fs::read_to_string(&path).expect("read fixture");
    let report =
        import_conversation_text_with_extractor(&store, &transcript, &LabeledTranscriptExtractor)
            .expect("import seam returns");
    let goal_id =
        stable_uuid(&report, &["ship password-less login for acme", "G1"]).map(IntentionId::from);
    let auth_decision_id =
        stable_uuid(&report, &["auth module at src/auth.py", "D_old"]).map(DecisionId::from);
    drop(store);
    let reopened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen imported store");
    let walk_rows = match goal_id {
        Some(goal_id) => reopened
            .by_intention(goal_id, DiscoveryFilter::default())
            .expect("walk imported intention"),
        None => Vec::new(),
    };
    let impact_rows = match auth_decision_id {
        Some(decision_id) => reopened
            .list_invalidations(
                walk_rows
                    .first()
                    .map(|decision| decision.context_id)
                    .unwrap_or_else(context_graph::ContextId::new_v7),
                Some(decision_id),
                None,
            )
            .unwrap_or_default(),
        None => Vec::new(),
    };
    drop(reopened);

    assert_eq!(
        (
            report.last_harvest,
            report.wrote_digest,
            goal_id.is_some(),
            auth_decision_id.is_some(),
            persisted_counts(&db_path),
            text_values(&db_path, "contexts", "name"),
            text_values(&db_path, "digests", "summary"),
            text_values(&db_path, "intentions", "description"),
            text_values(&db_path, "decisions", "description"),
            text_values(&db_path, "observations", "observation_type"),
            walk_rows
                .iter()
                .map(|decision| (decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
            impact_rows
                .iter()
                .map(|notice| notice.affected_decision_id)
                .collect::<Vec<_>>(),
        ),
        (
            HarvestCounts {
                goals: 1,
                decisions: 2,
                observations: 1,
            },
            true,
            true,
            true,
            ImportCounts {
                contexts: 1,
                digests: 1,
                intentions: 1,
                decisions: 2,
                observations: 1,
                extracted_from_edges: 5,
            },
            vec!["acme-import".to_string()],
            vec![
                "Goal: ship password-less login for acme. Decision: auth module at src/auth.py. Decision: use JWT with 15-min access tokens. Observation: auth module file observed at src/auth.py."
                    .to_string(),
            ],
            vec!["ship password-less login for acme".to_string()],
            vec![
                "auth module at src/auth.py".to_string(),
                "use JWT with 15-min access tokens".to_string(),
            ],
            vec!["auth module file observed at src/auth.py".to_string()],
            vec![
                ("use JWT with 15-min access tokens", DecisionStatus::Active),
                ("auth module at src/auth.py", DecisionStatus::Active),
            ],
            Vec::<DecisionId>::new(),
        )
    );
}

#[test]
fn import_generic_auth_replacement_does_not_take_record_shorthand() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let report = import_conversation_text_with_extractor(
        &store,
        &[
            "Context: generic-auth-import",
            "Goal: migrate auth handler.",
            "Assistant: Decision: replace auth handler with src/authentication.py.",
        ]
        .join("\n"),
        &LabeledTranscriptExtractor,
    )
    .expect("generic import returns");
    drop(store);

    assert_eq!(
        (
            report.last_harvest,
            persisted_counts(&db_path),
            text_values(&db_path, "contexts", "name"),
            text_values(&db_path, "intentions", "description"),
            text_values(&db_path, "decisions", "description"),
        ),
        (
            HarvestCounts {
                goals: 1,
                decisions: 1,
                observations: 0,
            },
            ImportCounts {
                contexts: 1,
                digests: 1,
                intentions: 1,
                decisions: 1,
                observations: 0,
                extracted_from_edges: 3,
            },
            vec!["generic-auth-import".to_string()],
            vec!["migrate auth handler".to_string()],
            vec!["replace auth handler with src/authentication.py".to_string()],
        )
    );
}

#[test]
fn ingest_dry_run_reports_without_writing() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    drop(store);
    let before = persisted_counts(&db_path);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen for dry-run");
    let dry = dry_run_import(&store, "set up CI pipeline").expect("dry-run seam returns");
    drop(store);
    let after_dry_run = persisted_counts(&db_path);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen for status");
    let after = record_conversation(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: String::new(),
            model_available: true,
        },
    )
    .expect("status seam returns");
    drop(store);
    let after_status = persisted_counts(&db_path);

    assert_eq!(
        (
            dry.last_harvest,
            dry.wrote_digest,
            before.clone(),
            after_dry_run,
            after.total_harvest,
            after_status,
        ),
        (
            HarvestCounts::default(),
            false,
            before.clone(),
            before,
            HarvestCounts::default(),
            ImportCounts {
                contexts: 0,
                digests: 0,
                intentions: 0,
                decisions: 0,
                observations: 0,
                extracted_from_edges: 0,
            },
        )
    );
}

#[test]
fn ingest_dedup_on_second_run() {
    let (_dir, store) = fresh_store();
    let first = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript:
                "Goal: ship password-less login for acme. Decision: auth module at src/auth.py."
                    .to_string(),
            model_available: true,
        },
        &LabeledTranscriptExtractor,
    )
    .expect("first capture");
    let second = record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript:
                "Goal: ship password-less login for acme. Decision: auth module at src/auth.py."
                    .to_string(),
            model_available: true,
        },
        &LabeledTranscriptExtractor,
    )
    .expect("second capture");

    assert_eq!(
        (first.total_harvest, second.total_harvest),
        (
            HarvestCounts {
                goals: 1,
                decisions: 1,
                observations: 0,
            },
            HarvestCounts {
                goals: 1,
                decisions: 1,
                observations: 0,
            },
        )
    );
    assert_eq!(
        first.stable_ids.get("D_old"),
        second.stable_ids.get("D_old")
    );
}
