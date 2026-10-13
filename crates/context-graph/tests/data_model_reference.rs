//! The data-model reference is derived from shipped Rust types and installed table metadata.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use chrono::{DateTime, Utc};
use context_graph::*;
use serde_json::Value;

macro_rules! fields {
    ($page:expr, $name:ident, { $($field:ident: $ty:ty => $column:literal),+ $(,)? }) => {{
        // Exhaustive patterns and typed borrows detect both field and field-type changes.
        let _check: fn(&$name) = |value| {
            let $name { $($field),+ } = value;
            $(let _: &$ty = $field;)+
        };
        writeln!($page, "\n## {}\n", stringify!($name)).unwrap();
        writeln!($page, "| Rust field | Rust type | Stored column or relation | Presence |\n|---|---|---|---|").unwrap();
        $(
            let spelling = stringify!($ty);
            let presence = if spelling.starts_with("Option") { "optional" } else { "required" };
            writeln!($page, "| `{}` | `{spelling}` | {} | {presence} |", stringify!($field), $column).unwrap();
        )+
    }};
}

fn enum_values<T: std::fmt::Debug>(page: &mut String, name: &str, values: &[T]) {
    writeln!(page, "\n### {name}\n\n| Rust value |\n|---|").unwrap();
    for value in values {
        writeln!(page, "| `{value:?}` |").unwrap();
    }
}

fn render_reference() -> String {
    let mut page = String::from(
        "# Data model\n\n<!-- concept -->\nGenerated from public Rust types and the installed schema; field presence describes Rust values, while table column nullability describes storage.\n",
    );
    fields!(page, Context, {
        id: ContextId => "contexts.id",
        name: String => "contexts.name",
        labels: Vec<String> => "contexts.labels",
        properties: BTreeMap<String, Value> => "contexts.properties",
        created_at: TxId => "contexts.created_at",
    });
    fields!(page, Intention, {
        id: IntentionId => "intentions.id",
        description: String => "intentions.description",
        status: IntentionStatus => "intentions.status",
        origin: IntentionOrigin => "intentions.origin",
        context_id: ContextId => "intentions.context_id",
        properties: BTreeMap<String, Value> => "intentions.properties",
        tags: Vec<String> => "intentions.tags",
        created_at: TxId => "intentions.created_at",
        blueprint_catalog_id: Option<String> => "intentions.blueprint_catalog_id",
    });
    fields!(page, Decision, {
        id: DecisionId => "decisions.id",
        decision_type: String => "decisions.decision_type",
        description: String => "decisions.description",
        reasoning: Vec<ReasoningStep> => "decisions.reasoning",
        confidence: Option<f32> => "decisions.confidence",
        status: DecisionStatus => "decisions.status",
        agent_id: Option<AgentId> => "decisions.agent_id",
        context_id: ContextId => "decisions.context_id",
        tags: Vec<String> => "decisions.tags",
        properties: BTreeMap<String, Value> => "decisions.properties",
        created_at: TxId => "decisions.created_at",
        intention_ids: Vec<IntentionId> => "edges (SERVES) and decision_intention_citations",
        based_on_snapshots: Vec<EntitySnapshotId> => "decision_basis_snapshots.snapshot_id",
        outcome: Option<Outcome> => "outcomes.decision_id",
        precedent_ids: Vec<DecisionId> => "edges (CITES)",
    });
    fields!(page, ReasoningStep, {
        operation: String => "decisions.reasoning[].operation",
        details: String => "decisions.reasoning[].details",
        result: String => "decisions.reasoning[].result",
        agent_id: Option<AgentId> => "decisions.reasoning[].agent_id",
        timestamp: DateTime<Utc> => "decisions.reasoning[].timestamp",
    });
    fields!(page, Entity, {
        id: EntityId => "entities.id",
        entity_type: EntityType => "entities.entity_type",
        name: String => "entities.name",
        properties: BTreeMap<String, Value> => "entities.properties",
        tags: Vec<String> => "entities.tags",
        context_id: ContextId => "entities.context_id",
        status: EntityStatus => "entities.status",
        created_at: TxId => "entities.created_at",
    });
    fields!(page, EntitySnapshot, {
        id: EntitySnapshotId => "entity_snapshots.id",
        entity_id: EntityId => "entity_snapshots.entity_id",
        entity_type: EntityType => "entity_snapshots.data.entity_type",
        name: String => "entity_snapshots.data.name",
        properties: BTreeMap<String, Value> => "entity_snapshots.data.properties",
        evidenced_by: Vec<ObservationId> => "entity_snapshot_observations.observation_id",
        captured_at: TxId => "entity_snapshots.data.captured_at",
        valid_from: TxId => "entity_snapshots.valid_from",
        valid_to: Option<TxId> => "entity_snapshots.valid_to",
    });
    fields!(page, Observation, {
        id: ObservationId => "observations.id",
        observation_type: String => "observations.observation_type",
        evidence: Vec<EvidenceRef> => "evidence.observation_id",
        observed_properties: BTreeMap<String, Value> => "observations.observed_properties",
        state_delta: BTreeMap<String, Option<Value>> => "observations.state_delta",
        properties: BTreeMap<String, Value> => "observations.properties",
        source: String => "observations.source",
        entity_id: EntityId => "observations.entity_id",
        context_id: ContextId => "observations.context_id",
        observed_at: DateTime<Utc> => "observations.timestamp",
        created_at: TxId => "observations.created_at",
    });
    fields!(page, EvidenceRef, {
        id: EvidenceId => "evidence.id",
        observation_id: ObservationId => "evidence.observation_id",
        context_id: ContextId => "evidence.context_id",
        kind: EvidenceKind => "evidence.kind",
        source_ref: String => "evidence.source_ref",
        mime_type: Option<String> => "evidence.mime_type",
        content_hash: Option<String> => "evidence.content_hash",
        captured_at: Option<DateTime<Utc>> => "evidence.captured_at + captured_at_nanos",
        byte_range: Option<(u64, u64)> => "evidence.byte_range",
        time_range_ms: Option<(u64, u64)> => "evidence.time_range_ms",
        frame_index: Option<u64> => "evidence.frame_index",
        region: Option<Region> => "evidence.region",
        producer: EvidenceProducer => "evidence.producer",
        text_projection: Option<String> => "evidence.text_projection",
        derived_from: Option<EvidenceId> => "evidence.derived_from",
        retention_status: RetentionStatus => "evidence.retention_status + evidence_availability_log.status",
        media_retention_until: Option<DateTime<Utc>> => "evidence.media_retention_until + media_retention_until_nanos",
        properties: BTreeMap<String, Value> => "evidence.properties",
    });
    fields!(page, EvidenceProducer, {
        system: String => "producer JSON.system",
        model_name: String => "producer JSON.model_name",
        model_version: String => "producer JSON.model_version",
        pipeline_version: String => "producer JSON.pipeline_version",
    });
    fields!(page, Outcome, {
        id: OutcomeId => "outcomes.id",
        decision_id: DecisionId => "outcomes.decision_id",
        success: bool => "outcomes.success",
        notes: Option<String> => "outcomes.notes",
        recorded_at: TxId => "outcomes.recorded_at",
    });
    fields!(page, Invalidation, {
        id: InvalidationId => "invalidations.id",
        status: InvalidationStatus => "invalidations.status",
        context_id: ContextId => "invalidations.context_id",
        affected_decision_id: DecisionId => "invalidations.affected_decision_id",
        trigger_observation_id: ObservationId => "invalidations.trigger_observation_id",
        reason: Option<String> => "invalidations.reason",
        basis_diff: BasisDiffResult => "invalidations.basis_diff",
        cascade_from: Option<DecisionId> => "invalidations.cascade_from",
        severity: InvalidationSeverity => "invalidations.severity",
        threatened_intention_ids: Vec<IntentionId> => "invalidations.threatened_intention_ids",
        intention_now_un_served: bool => "invalidations.intention_now_un_served",
        detected_at: TxId => "invalidations.detected_at",
        resolved_at: Option<TxId> => "invalidations.resolved_at",
        resolution_action: Option<String> => "invalidations.resolution_action",
        resolution_decision_id: Option<DecisionId> => "invalidations.resolution_decision_id",
    });
    fields!(page, Digest, {
        id: DigestId => "digests.id",
        source: String => "digests.source",
        summary: String => "digests.summary",
        vector_text: Option<Vec<f32>> => "digests.vector_text",
        dimension: usize => "digests.dimension",
        context_id: ContextId => "digests.context_id",
        session_ref: Option<String> => "digests.session_ref",
        properties: BTreeMap<String, Value> => "digests.properties",
        created_at: TxId => "digests.created_at",
    });
    fields!(page, EmbeddingSpace, {
        embedding_space_id: String => "embedding_spaces.embedding_space_id",
        dimension: usize => "embedding_spaces.dimension",
        metric: DistanceMetric => "embedding_spaces.metric",
        model_family: String => "embedding_spaces.model_family",
        model_name: String => "embedding_spaces.model_name",
        model_version: String => "embedding_spaces.model_version",
        supported_evidence_kinds: Vec<EvidenceKind> => "embedding_spaces.supported_evidence_kinds",
        default_for_evidence_kind: Option<EvidenceKind> => "embedding_spaces.default_for_evidence_kind",
        bindings: Vec<EmbeddingSpaceBinding> => "embedding_space_bindings and embedding_spaces.bindings",
    });
    fields!(page, StoredEmbedding, {
        id: EmbeddingId => "embedding handle id",
        context_id: ContextId => "embedding handle context_id",
        embedding_space_id: String => "embedding handle embedding_space_id",
        vector: Vec<f32> => "target vector column",
        dimension: usize => "embedding_spaces.dimension",
        produced_by: EvidenceProducer => "embedding handle produced_by",
        target: EmbeddingTarget => "embedding handle target_id and target table",
        created_at: TxId => "embedding handle created_at",
    });

    page.push_str("\n## Enum values\n");
    enum_values(&mut page, "EntityType", &EntityType::ALL);
    enum_values(&mut page, "RelationshipType", &RelationshipType::ALL);
    enum_values(&mut page, "IntentionOrigin", &IntentionOrigin::ALL);
    enum_values(&mut page, "DecisionStatus", &DecisionStatus::ALL);
    enum_values(&mut page, "IntentionStatus", &IntentionStatus::ALL);
    enum_values(&mut page, "InvalidationStatus", &InvalidationStatus::ALL);
    enum_values(
        &mut page,
        "InvalidationSeverity",
        &InvalidationSeverity::ALL,
    );
    enum_values(&mut page, "EvidenceKind", &EvidenceKind::ALL);
    enum_values(&mut page, "RetentionStatus", &RetentionStatus::ALL);
    enum_values(&mut page, "DistanceMetric", &DistanceMetric::ALL);
    enum_values(&mut page, "ResolveAction", &ResolveAction::ALL);

    page.push_str("\n## Relationships\n\n| Rust value | How it exists | Written by | Read by |\n|---|---|---|---|\n");
    for value in RelationshipType::ALL {
        let (shape, writer, reader) = match value {
            RelationshipType::Serves => (
                "edge and intention citation",
                "create_decision, supersede_decision, route_decision_to_intention",
                "by_intention, impact analysis",
            ),
            RelationshipType::BasedOn => (
                "edge with basis_fields and decision_basis_snapshots",
                "create_decision",
                "impact analysis, by_entity_decisions, replay",
            ),
            RelationshipType::Supersedes => (
                "edge",
                "supersede_decision",
                "supersede_history, incoming_supersedes",
            ),
            RelationshipType::HasOutcome => (
                "outcomes.decision_id",
                "record_outcome",
                "has_outcome_relationship, Decision.outcome",
            ),
            RelationshipType::DependsOn | RelationshipType::RelatesTo => {
                ("edge", "add_relationship", "traverse")
            }
            RelationshipType::Cites => (
                "DAG edge",
                "create_decision precedent_ids",
                "precedent_chain, impact analysis",
            ),
            RelationshipType::About => (
                "edge type",
                "no typed writer",
                "by_entity_decisions, by_entity_intentions",
            ),
            RelationshipType::ObservedOn => (
                "observations.entity_id",
                "record_observation",
                "impact analysis",
            ),
            RelationshipType::Invalidates => (
                "invalidations.affected_decision_id",
                "observation invalidation",
                "list_invalidations",
            ),
            RelationshipType::TriggeredBy => (
                "invalidations.trigger_observation_id",
                "observation invalidation",
                "list_invalidations",
            ),
            RelationshipType::ExtractedFrom => ("edge", "capture", "find_via_digests"),
            RelationshipType::MatchedBy
            | RelationshipType::Generated
            | RelationshipType::Instantiates
            | RelationshipType::Refines => ("enum only", "none", "none"),
        };
        writeln!(page, "| `{value:?}` | {shape} | {writer} | {reader} |").unwrap();
    }
    page.push_str("\n## Tables, nullability, status transitions and locality\n");
    let store = Store::open(StoreConfig {
        db_path: ":memory:".into(),
        ..StoreConfig::default()
    })
    .expect("install shipped schema");
    let db = store.sync_database();
    for table in context_graph::schema::TABLES {
        assert!(
            db.table_meta(table.name).is_some(),
            "published table {} is installed",
            table.name
        );
    }
    // Include local event and search tables as well as the public TABLES declarations.
    let mut names = db.table_names();
    names.sort();
    for name in names {
        let meta = db.table_meta(&name).expect("installed table metadata");
        let sync = meta
            .sync_direction
            .unwrap_or(contextdb_core::SyncDirection::Both)
            .sql();
        let conflict = meta
            .conflict_policy
            .unwrap_or(contextdb_core::DEFAULT_CONFLICT_POLICY)
            .declared_clause()
            .expect("public conflict policy");
        writeln!(
            page,
            "\n### `{name}`\n\n| Sync | Conflict policy |\n|---|---|\n| {sync} | {conflict} |\n"
        )
        .unwrap();
        page.push_str(
            "| Stored column | Storage type | Nullability | Default |\n|---|---|---|---|\n",
        );
        for column in &meta.columns {
            writeln!(
                page,
                "| `{}` | `{:?}` | {} | `{}` |",
                column.name,
                column.column_type,
                if column.nullable {
                    "optional"
                } else {
                    "required"
                },
                column.default.as_deref().unwrap_or("none")
            )
            .unwrap();
        }
        if let Some(machine) = &meta.state_machine {
            page.push_str("\n| Status column | From | Allowed successors |\n|---|---|---|\n");
            let mut transitions: Vec<_> = machine.transitions.iter().collect();
            transitions.sort_by_key(|(from, _)| *from);
            for (from, to) in transitions {
                writeln!(
                    page,
                    "| `{}` | `{from}` | `{}` |",
                    machine.column,
                    to.join(", ")
                )
                .unwrap();
            }
        }
    }
    page.push_str("\n## Source references\n\n| Scheme | Acceptance |\n|---|---|\n| vigil, edge, s3, gs, azure, cas, webhook | unconditional |\n| https | host in configured allowlist, case insensitive |\n| file | file_uri_allowed is true |\n");
    page.push_str("\n## Reserved declarations\n\n| Declaration | Typed API |\n|---|---|\n| signatures, blueprints, approvals | none |\n| intentions.origin_ref, blueprint_id, bindings, blueprint_proposal | none |\n| IntentionOrigin::Human, IntentionOrigin::Agent | produced values |\n| IntentionOrigin::Signature, IntentionOrigin::Blueprint | reserved values |\n| invalidations.resolution_action | new_decision_created, basis_still_valid, dismissed_as_irrelevant |\n");
    page
}

/// The data-model page exactly matches shipped fields, enum values and installed tables.
#[test]
fn the_data_model_page_matches_the_shipped_types_and_tables() {
    let expected = render_reference();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let actual = std::fs::read_to_string(root.join("docs/data-model.md")).unwrap_or_default();
    assert_eq!(
        actual, expected,
        "docs/data-model.md differs; expected page:\n{expected}"
    );
}
