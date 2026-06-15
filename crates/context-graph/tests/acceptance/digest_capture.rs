use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use context_graph::embedding_space::default_text_embedding_space;
use context_graph::ingest::{
    CaptureImpactSignal, CaptureMode, CaptureRequest, DedupCandidate, DedupClassification,
    HarvestCounts, classify_dedup_candidate, record_conversation, standing_impact,
};
use context_graph::schema::DIGEST_VECTOR_DIMENSION;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDigest, DecisionId, DecisionStatus,
    DigestSearchFilter, DiscoveryFilter, Embedder, EmbedderRegistration, EmbeddingId,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingTarget, EvidenceKind,
    EvidenceProducer, IntentionId, PrecomputedEmbedding, Store, StoreConfig,
};
use contextdb_engine::Database;

const G1: &str = "ship password-less login for acme";
const D_OLD: &str = "auth module at src/auth.py";
const D_NEW: &str = "auth module at src/authentication.py";

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_recall_embedder(dir.path().join("cg.db"));
    (dir, store)
}

fn open_store_with_recall_embedder(db_path: PathBuf) -> Store {
    let embedder = Arc::new(RecallProbeEmbedder);
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open store with recall probe")
}

struct RecallProbeEmbedder;

impl Embedder for RecallProbeEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let text = match input {
            EmbeddingInput::Text(text) => text,
            _ => String::new(),
        };
        Ok(EmbeddingOutput {
            embedding_space_id: default_text_embedding_space().embedding_space_id,
            dimension: DIGEST_VECTOR_DIMENSION,
            producer: EvidenceProducer::default(),
            vector: recall_vector_for(&text),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        default_text_embedding_space()
    }
}

fn capture(
    store: &Store,
    transcript: &str,
    model_available: bool,
) -> context_graph::ingest::CaptureReport {
    record_conversation(
        store,
        CaptureRequest {
            context_label: "acme-coding-assistant".to_string(),
            transcript: transcript.to_string(),
            model_available,
        },
    )
    .expect("compile-only capture seam returns a value")
}

fn digest_vector(seed: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    vector[seed % DIGEST_VECTOR_DIMENSION] = 1.0;
    vector
}

fn recall_vector_for(text: &str) -> Vec<f32> {
    if text.contains("durable raw transcript") || text.contains("raw auth recall") {
        digest_vector(0)
    } else {
        digest_vector(1)
    }
}

fn raw_count(db_path: &Path, table: &str) -> usize {
    let db = Database::open(db_path).expect("raw db");
    db.scan(table, db.snapshot()).expect("raw table scan").len()
}

fn raw_digest_summaries(db_path: &Path) -> Vec<String> {
    let db = Database::open(db_path).expect("raw db");
    db.execute(
        "SELECT summary FROM digests ORDER BY created_at ASC",
        &HashMap::new(),
    )
    .expect("raw digest summaries")
    .rows
    .into_iter()
    .filter_map(|row| match row.first() {
        Some(contextdb_core::Value::Text(summary)) => Some(summary.clone()),
        _ => None,
    })
    .collect()
}

fn context_id_by_name(store: &Store, name: &str) -> Option<ContextId> {
    store
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|context| context.name == name)
        .map(|context| context.id)
}

fn stable_decision_id(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<DecisionId> {
    stable_uuid(reports, keys).map(DecisionId::from)
}

fn stable_intention_id(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<IntentionId> {
    stable_uuid(reports, keys).map(IntentionId::from)
}

fn stable_uuid(
    reports: &[&context_graph::ingest::CaptureReport],
    keys: &[&str],
) -> Option<uuid::Uuid> {
    reports.iter().find_map(|report| {
        keys.iter()
            .find_map(|key| report.stable_ids.get(*key))
            .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
    })
}

fn decision_row(
    store: &Store,
    id: Option<DecisionId>,
) -> Option<(DecisionId, String, DecisionStatus)> {
    id.and_then(|id| {
        store
            .get_decision(id)
            .ok()
            .flatten()
            .map(|decision| (decision.id, decision.description, decision.status))
    })
}

fn create_digest_input(context_id: ContextId, summary: &str, kind: EvidenceKind) -> CreateDigest {
    let id = context_graph::DigestId::new_v7();
    CreateDigest {
        id,
        source: "claude-cli".to_string(),
        session_ref: Some("session-1".to_string()),
        summary: summary.to_string(),
        kind,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
            vector: digest_vector(summary.len()),
            target: EmbeddingTarget::Digest(id),
            produced_by: EvidenceProducer::default(),
        },
        context_id,
        properties: BTreeMap::new(),
    }
}

#[test]
fn digest_create_and_retrieve_persist() {
    let (_dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "digest-persist".to_string(),
            ..Default::default()
        })
        .expect("context");
    let created = store
        .create_digest(create_digest_input(
            context.id,
            "Tuesday session: auth module at src/auth.py serves password-less login",
            EvidenceKind::Text,
        ))
        .expect("create digest");
    let db_path = store.db_path().to_path_buf();
    drop(store);

    let reopened = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen");
    let fetched = reopened
        .get_digest(created.id)
        .expect("get digest")
        .expect("digest exists");

    assert_eq!(fetched.id, created.id);
    assert_eq!(fetched.source, "claude-cli");
    assert_eq!(fetched.session_ref.as_deref(), Some("session-1"));
    assert_eq!(
        fetched.summary,
        "Tuesday session: auth module at src/auth.py serves password-less login"
    );
    assert_eq!(fetched.context_id, context.id);
    assert_eq!(fetched.dimension, DIGEST_VECTOR_DIMENSION);
    assert_eq!(
        fetched.vector_text,
        Some(digest_vector(created.summary.len()))
    );
}

#[test]
fn digest_extract_pipeline_produces_derived_entities() {
    let (_dir, store) = fresh_store();
    let report = capture(
        &store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py.",
        true,
    );
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let derived = report
        .derived_items
        .iter()
        .map(|item| {
            (
                item.id.as_str(),
                item.kind.as_str(),
                item.text.as_str(),
                item.status.as_str(),
                item.source_digest_id,
                item.context_label.as_str(),
            )
        })
        .collect::<Vec<_>>();

    assert!(source_digest.is_some());
    assert_eq!(
        derived,
        vec![
            (
                "G1",
                "intention",
                G1,
                "active",
                source_digest,
                "acme-coding-assistant",
            ),
            (
                "D_old",
                "decision",
                D_OLD,
                "current",
                source_digest,
                "acme-coding-assistant",
            ),
            (
                "O_auth_path",
                "observation",
                "auth module file observed at src/auth.py",
                "current",
                source_digest,
                "acme-coding-assistant",
            ),
        ]
    );
}

#[test]
fn digest_condensed_vs_raw_ingester() {
    let (_dir, store) = fresh_store();
    let condensed = capture(&store, "auth module lives at src/auth.py", true);
    let raw = capture(&store, "auth module lives at src/auth.py", false);

    assert_eq!(
        (
            condensed.mode,
            condensed.digest_summary.as_str(),
            condensed
                .derived_items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            raw.mode,
            raw.verbatim_text.as_str(),
            raw.derived_items.len(),
        ),
        (
            CaptureMode::Condensed,
            "condensed: password-less login uses auth module at src/auth.py",
            vec![G1, D_OLD],
            CaptureMode::RawVerbatim,
            "auth module lives at src/auth.py",
            0,
        )
    );
}

#[test]
fn digest_dedup_four_way_classification() {
    let observed = [
        (
            DedupClassification::NewInsertion,
            "cache policy",
            "",
            "store refresh tokens in Redis",
        ),
        (
            DedupClassification::NewInsertion,
            "auth module file",
            "auth module at src/auth.py",
            "authentication.py test fixture is mentioned only in a changelog comment",
        ),
        (
            DedupClassification::ExactDuplicate,
            "auth module file",
            D_OLD,
            D_OLD,
        ),
        (
            DedupClassification::ExactDuplicate,
            "passwordless login goal",
            "users sign in without passwords",
            "passkeys complete account access",
        ),
        (
            DedupClassification::RefinesExisting,
            "jwt token lifetime",
            "use JWT tokens",
            "use JWT with 15-min access tokens",
        ),
        (
            DedupClassification::RefinesExisting,
            "auth module owner",
            "platform team owns authentication",
            "platform team owns authentication and reviews auth.py changes",
        ),
        (
            DedupClassification::ConflictsExisting,
            "auth module file",
            D_OLD,
            D_NEW,
        ),
        (
            DedupClassification::ConflictsExisting,
            "token storage",
            "refresh tokens never leave Postgres",
            "refresh tokens are stored in Redis",
        ),
    ]
    .into_iter()
    .map(|(expected, entity, prior_value, candidate_value)| {
        (
            classify_dedup_candidate(DedupCandidate {
                entity: entity.to_string(),
                prior_value: prior_value.to_string(),
                candidate_value: candidate_value.to_string(),
            }),
            expected,
        )
    })
    .collect::<Vec<_>>();

    assert_eq!(
        observed,
        vec![
            (
                DedupClassification::NewInsertion,
                DedupClassification::NewInsertion,
            ),
            (
                DedupClassification::NewInsertion,
                DedupClassification::NewInsertion,
            ),
            (
                DedupClassification::ExactDuplicate,
                DedupClassification::ExactDuplicate,
            ),
            (
                DedupClassification::ExactDuplicate,
                DedupClassification::ExactDuplicate,
            ),
            (
                DedupClassification::RefinesExisting,
                DedupClassification::RefinesExisting,
            ),
            (
                DedupClassification::RefinesExisting,
                DedupClassification::RefinesExisting,
            ),
            (
                DedupClassification::ConflictsExisting,
                DedupClassification::ConflictsExisting,
            ),
            (
                DedupClassification::ConflictsExisting,
                DedupClassification::ConflictsExisting,
            ),
        ]
    );
}

#[test]
fn dedup_high_text_similarity_contradiction_not_folded_into_duplicate() {
    let class = classify_dedup_candidate(DedupCandidate {
        entity: "auth module file".to_string(),
        prior_value: "auth module at src/auth.py".to_string(),
        candidate_value:
            "auth module is no longer at src/auth.py; it moved to src/authentication.py".to_string(),
    });

    assert_eq!(class, DedupClassification::ConflictsExisting);
}

#[test]
fn digest_is_text_evidence_not_universal_observation() {
    let (dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "typed-text-digest".to_string(),
            ..Default::default()
        })
        .expect("context");
    let digest = store
        .create_digest(create_digest_input(
            context.id,
            "digest is typed text evidence, not an Observation row",
            EvidenceKind::Text,
        ))
        .expect("create digest");
    drop(store);

    let db = Database::open(dir.path().join("cg.db")).expect("raw db");
    let snapshot = db.snapshot();
    assert_eq!(db.scan("digests", snapshot).expect("digests").len(), 1);
    assert_eq!(
        db.scan("observations", snapshot)
            .expect("observations")
            .len(),
        0
    );
    assert_ne!(digest.id.as_uuid(), uuid::Uuid::nil());
}

#[test]
fn digest_does_not_ingest_camera_streams() {
    let (_dir, store) = fresh_store();
    let context = store
        .create_context(CreateContext {
            name: "camera-digest-rejection".to_string(),
            ..Default::default()
        })
        .expect("context");
    let result = store.create_digest(create_digest_input(
        context.id,
        "camera frame should not become a Digest",
        EvidenceKind::ImageFrame,
    ));

    assert!(matches!(
        result,
        Err(CgError::UnsupportedEvidenceKind {
            kind: EvidenceKind::ImageFrame
        })
    ));
}

#[test]
fn raw_ingester_floor_stores_verbatim_no_derived_graph_and_says_so() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let raw_text = "durable raw transcript: auth module remains at src/auth.py";
    let report = capture(&store, raw_text, false);
    let context_id = context_id_by_name(&store, "acme-coding-assistant");
    drop(store);

    let persisted_counts = (
        raw_count(&db_path, "digests"),
        raw_count(&db_path, "intentions"),
        raw_count(&db_path, "decisions"),
        raw_count(&db_path, "observations"),
    );
    let persisted_summaries = raw_digest_summaries(&db_path);
    let reopened = open_store_with_recall_embedder(db_path.clone());
    let hits = reopened
        .find_via_digests(
            "raw auth recall",
            DigestSearchFilter {
                context_id,
                ..Default::default()
            },
        )
        .expect("raw digest recall query");
    let standing = context_id
        .map(|context_id| standing_impact(&reopened, context_id).expect("standing impact"))
        .unwrap_or_default();

    assert_eq!(report.mode, CaptureMode::RawVerbatim);
    assert_eq!(report.verbatim_text.as_str(), raw_text);
    assert_eq!(report.derived_items.len(), 0);
    assert!(report.wrote_digest);
    assert_eq!(
        report.impact_signal,
        Some(CaptureImpactSignal::VerbatimOnlyNoGraph)
    );
    assert_eq!(
        report.message.as_str(),
        "model-free capture: stored verbatim text; no derived graph or impact analysis"
    );
    assert!(context_id.is_some());
    assert_eq!(persisted_counts, (1, 0, 0, 0));
    assert_eq!(persisted_summaries, vec![raw_text.to_string()]);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.summary.as_str())
            .collect::<Vec<_>>(),
        vec![raw_text]
    );
    assert_eq!(standing, Vec::new());

    let (_condensed_dir, condensed_store) = fresh_store();
    let condensed = capture(
        &condensed_store,
        "Goal: ship password-less login for acme. Decision: auth module at src/auth.py.",
        true,
    );
    assert_eq!(
        condensed.impact_signal,
        Some(CaptureImpactSignal::DerivedGraph)
    );
}

#[test]
fn every_derived_item_has_extracted_from_edge_to_source_digest() {
    let (_dir, store) = fresh_store();
    let report = capture(&store, "derive G1 and D_old from one digest", true);
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let edges = report
        .provenance_edges
        .iter()
        .map(|edge| {
            (
                edge.relationship.as_str(),
                edge.item_id.as_str(),
                Some(edge.digest_id),
            )
        })
        .collect::<Vec<_>>();
    let item_sources = report
        .derived_items
        .iter()
        .map(|item| (item.id.as_str(), item.source_digest_id))
        .collect::<Vec<_>>();

    assert!(source_digest.is_some());
    assert_eq!(
        edges,
        vec![
            ("EXTRACTED_FROM", "G1", source_digest),
            ("EXTRACTED_FROM", "D_old", source_digest),
            ("EXTRACTED_FROM", "O_auth_path", source_digest),
        ]
    );
    assert_eq!(
        item_sources,
        vec![
            ("G1", source_digest),
            ("D_old", source_digest),
            ("O_auth_path", source_digest),
        ]
    );
}

#[test]
fn record_empty_or_garbage_input_extracts_nothing_and_says_so() {
    let (_dir, store) = fresh_store();
    let report = capture(&store, "}{ definitely not a conversation", true);

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(report.total_harvest, HarvestCounts::default());
    assert_eq!(
        report.message,
        "nothing extractable: no goals, decisions, or observations found"
    );
    assert!(!report.wrote_digest);
}

#[test]
fn capture_with_model_zero_harvest_is_honest_not_error() {
    let (_dir, store) = fresh_store();
    let report = capture(&store, "Thanks, sounds good.", true);

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(
        report.message,
        "model completed: zero structured items extracted from this capture"
    );
    assert!(report.wrote_digest);
}

#[test]
fn value_replacement_supersedes_decision_not_invalidates() {
    let (_dir, store) = fresh_store();
    let db_path = store.db_path().to_path_buf();
    let first = capture(
        &store,
        "Goal G1: ship password-less login for acme. Decision D_old: auth module at src/auth.py serves G1.",
        true,
    );
    let second = capture(
        &store,
        "Replace auth module decision D_old with D_new: auth module at src/authentication.py serves G1.",
        true,
    );
    drop(store);

    let reopened = open_store_with_recall_embedder(db_path);
    let reports = [&first, &second];
    let old_decision = stable_decision_id(&reports, &["D_old", D_OLD]);
    let new_decision = stable_decision_id(&reports, &["D_new", D_NEW]);
    let goal = stable_intention_id(&reports, &["G1", G1]);
    let decision_statuses = second
        .derived_items
        .iter()
        .filter(|item| item.kind == "decision")
        .map(|item| (item.text.as_str(), item.status.as_str()))
        .collect::<Vec<_>>();
    let dedup_events = second
        .dedup_events
        .iter()
        .map(|event| {
            (
                event.entity.as_str(),
                event.prior_value.as_str(),
                event.candidate_value.as_str(),
                event.classification,
            )
        })
        .collect::<Vec<_>>();
    let default_active = goal
        .map(|goal| {
            reopened
                .by_intention(goal, DiscoveryFilter::default())
                .expect("default-active by-intention")
        })
        .unwrap_or_default();
    let include_inactive = goal
        .map(|goal| {
            reopened
                .by_intention(
                    goal,
                    DiscoveryFilter {
                        include_decision_statuses: Some(vec![
                            DecisionStatus::Active,
                            DecisionStatus::Superseded,
                            DecisionStatus::Invalidated,
                        ]),
                        ..Default::default()
                    },
                )
                .expect("include-inactive by-intention")
        })
        .unwrap_or_default();

    assert_eq!(
        decision_statuses,
        vec![(D_OLD, "superseded"), (D_NEW, "current")]
    );
    assert_eq!(
        dedup_events,
        vec![(
            "auth module file",
            "src/auth.py",
            "src/authentication.py",
            DedupClassification::ConflictsExisting,
        )]
    );
    assert_eq!(
        (
            decision_row(&reopened, old_decision),
            decision_row(&reopened, new_decision),
            default_active
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
            include_inactive
                .iter()
                .map(|decision| (decision.id, decision.description.as_str(), decision.status))
                .collect::<Vec<_>>(),
        ),
        (
            old_decision.map(|id| (id, D_OLD.to_string(), DecisionStatus::Superseded)),
            new_decision.map(|id| (id, D_NEW.to_string(), DecisionStatus::Active)),
            new_decision
                .map(|id| vec![(id, D_NEW, DecisionStatus::Active)])
                .unwrap_or_default(),
            old_decision
                .zip(new_decision)
                .map(|(old, new)| {
                    vec![
                        (new, D_NEW, DecisionStatus::Active),
                        (old, D_OLD, DecisionStatus::Superseded),
                    ]
                })
                .unwrap_or_default(),
        )
    );
}
