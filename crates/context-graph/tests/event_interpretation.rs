//! Independent RED contracts for released event interpretation.
//!
//! Fixture values are authored directly in JSON.  The mapper, preservation validator, and every
//! expected assertion below deliberately avoid production mapping and hash helpers.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::{
    AdapterCoverageGapState, AppendEventBatch, CapturedEventAndAnchors, CgError,
    ConnectorInterpretationRunner, ContextSelector, CreateContext, EffectiveEventFields,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole,
    InstalledInterpretationContract, InterpretationBuildRequest, Page,
    ReleasedConnectorInterpretation, ReleasedInterpretationDeclaration, Result, Store, StoreConfig,
    admit_interpretation,
};
use serde_json::{Value, json};

// The activation fixtures read persisted heads through the internal-only SQL test seam.
#[cfg(context_graph_internal_test_support)]
use {
    context_graph::embedder::{
        EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
        PreparedEventSearchQuery,
    },
    context_graph::{
        DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput,
        EmbeddingSpace, EmbeddingSpaceBinding, EventSearchProfile, EventSearchRebuildRequest,
        EvidenceKind, EvidenceProducer, InterpretationActivationRequest,
    },
    contextdb_core::Value as DbValue,
    std::collections::HashMap,
    std::sync::atomic::{AtomicUsize, Ordering},
};

const CAPTURED: &str = include_str!("fixtures/interpretation/captured-event-and-anchors.json");
const EXPECTED: &str = include_str!("fixtures/interpretation/expected-effective-event.json");

fn captured() -> CapturedEventAndAnchors {
    serde_json::from_str(CAPTURED).expect("independently authored captured fixture")
}

fn expected() -> EffectiveEventFields {
    serde_json::from_str(EXPECTED).expect("independently authored expected effective fixture")
}

fn declaration() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: "fixture-connector".into(),
        target_interpretation_version: "released-v1".into(),
        target_adapter_version: "capture-v1".into(),
        target_anchor_declaration_version: "anchors-v1".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-v1".into(),
            anchor_declaration_version: "anchors-v1".into(),
        }],
        mapping_contract_id: "fixture-preservation-contract".into(),
        mapping_contract_sha256: "fixture-contract-sha256".into(),
    }
}

#[derive(Clone)]
struct FixtureMapper {
    declaration: ReleasedInterpretationDeclaration,
    answer: EffectiveEventFields,
}

impl ReleasedConnectorInterpretation for FixtureMapper {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn interpret_event(
        &self,
        _: &context_graph::ConversationEvent,
    ) -> Result<EffectiveEventFields> {
        Ok(self.answer.clone())
    }
}

struct FixtureContract {
    declaration: ReleasedInterpretationDeclaration,
    expected_captured: CapturedEventAndAnchors,
    required: EffectiveEventFields,
}

impl InstalledInterpretationContract for FixtureContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn validate(
        &self,
        input: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        assert_eq!(
            input, &self.expected_captured,
            "validator consumes the complete capture expected for this journey"
        );
        if candidate == &self.required {
            Ok(())
        } else {
            Err(CgError::NotFound {
                id: None,
                missing: Some("fixture preservation mismatch"),
            })
        }
    }
}

fn installed() -> (Arc<FixtureMapper>, Arc<FixtureContract>) {
    installed_for_capture(captured())
}

fn installed_for_capture(
    expected_captured: CapturedEventAndAnchors,
) -> (Arc<FixtureMapper>, Arc<FixtureContract>) {
    let declaration = declaration();
    let answer = expected();
    (
        Arc::new(FixtureMapper {
            declaration: declaration.clone(),
            answer: answer.clone(),
        }),
        Arc::new(FixtureContract {
            declaration,
            expected_captured,
            required: answer,
        }),
    )
}

/// A deterministic event-search test embedder. It has a distinct test profile, so it never
/// claims the canonical Nomic event-history identity reserved for the authorized wrapper.
#[cfg(context_graph_internal_test_support)]
struct InterpretationProjectionEmbedder {
    document_prepare_calls: AtomicUsize,
    query_prepare_calls: AtomicUsize,
}

#[cfg(context_graph_internal_test_support)]
impl InterpretationProjectionEmbedder {
    fn new() -> Self {
        Self {
            document_prepare_calls: AtomicUsize::new(0),
            query_prepare_calls: AtomicUsize::new(0),
        }
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self, inference: &EventSearchInference) -> EmbeddingOutput {
        let mut vector = vec![0.0; 768];
        vector[0] = inference.token_count() as f32;
        EmbeddingOutput {
            embedding_space_id: self.encoding_profile_id(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "event-interpretation-red".into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }
}

#[cfg(context_graph_internal_test_support)]
impl Embedder for InterpretationProjectionEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search must use explicit document/query preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "event-interpretation-red-deterministic-768".into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "deterministic-768".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "event-interpretation-red-deterministic-768".into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(1_536)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        self.document_prepare_calls.fetch_add(1, Ordering::SeqCst);
        let payload: Vec<u32> = text.bytes().map(u32::from).collect();
        let boundaries = (0..text.len()).map(|byte| (byte, byte + 1)).collect();
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries(boundaries)
            .payload(payload)
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        self.query_prepare_calls.fetch_add(1, Ordering::SeqCst);
        seal.query_builder()
            .token_count(text.len())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }
}

#[cfg(context_graph_internal_test_support)]
fn expected_persisted_capture(event: context_graph::ConversationEvent) -> CapturedEventAndAnchors {
    let file = event.anchors["file"]
        .as_str()
        .expect("persisted fixture event retains its file anchor")
        .to_owned();
    CapturedEventAndAnchors {
        anchors: vec![context_graph::CapturedEventAnchor {
            context_id: event.context_id,
            event_id: event.id,
            declaration_version: event.anchor_declaration_version.clone(),
            connector_id: event.connector_id.clone(),
            source_instance_ref: event.source_instance_ref.clone(),
            session_ref: event.session_ref.clone(),
            render_role: event.render_role,
            interaction_ref: event.interaction_ref.clone(),
            key: "file".into(),
            canonical_value: file.clone(),
            identity_value: format!("\"{file}\""),
            original_value: Value::String(file),
        }],
        event,
    }
}

fn assert_admission_reason(
    result: Result<context_graph::AdmittedEffectiveEvent>,
    reason: context_graph::InterpretationAdmissionRefusalReason,
) {
    assert!(
        matches!(result, Err(CgError::InterpretationAdmissionRefused { reason: actual, .. }) if actual == reason),
        "expected typed admission refusal {reason:?}"
    );
}

#[test]
fn released_connector_interpretation_is_admission_capable_in_memory() {
    let captured = captured();
    let (mapper, contract) = installed();
    let candidate = mapper
        .interpret_event(&captured.event)
        .expect("fixture mapper result");
    contract
        .validate(&captured, &candidate)
        .expect("fixture contract accepts authored candidate");

    assert!(
        admit_interpretation(contract.as_ref(), &captured, candidate).is_ok(),
        "a complete independently released mapping is admission-capable"
    );
}

#[test]
fn admission_refuses_every_changed_effective_field_and_invalid_anchor_with_typed_reasons() {
    let captured = captured();
    let (_, contract) = installed();
    let mutations: [(&str, Value); 9] = [
        ("/occurred_at", Value::Null),
        ("/author", json!("other-author")),
        ("/kind", json!("other-kind")),
        ("/render_role", json!("human_message")),
        ("/interaction_ref", Value::Null),
        ("/text", Value::Null),
        ("/payload", json!({"released": false})),
        ("/anchors", json!([])),
        ("/coverage", json!("opaque")),
    ];

    for (pointer, replacement) in mutations {
        let mut value: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
        *value
            .pointer_mut(pointer)
            .expect("declared mutable effective field") = replacement;
        let candidate: EffectiveEventFields =
            serde_json::from_value(value).expect("valid effective fixture mutation");
        assert_admission_reason(
            admit_interpretation(contract.as_ref(), &captured, candidate),
            context_graph::InterpretationAdmissionRefusalReason::PreservationContractRefused,
        );
    }

    let mut renamed: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
    *renamed
        .pointer_mut("/anchors/0/key")
        .expect("declared anchor key") = json!("mutated-anchor");
    assert_admission_reason(
        admit_interpretation(
            contract.as_ref(),
            &captured,
            serde_json::from_value(renamed).expect("valid renamed-anchor fixture"),
        ),
        context_graph::InterpretationAdmissionRefusalReason::PreservationContractRefused,
    );

    for pointer in ["/anchors/0/canonical_value", "/anchors/0/original_value"] {
        let mut value: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
        *value.pointer_mut(pointer).expect("declared anchor field") = json!("mutated-anchor");
        let candidate: EffectiveEventFields =
            serde_json::from_value(value).expect("valid anchor mutation");
        assert_admission_reason(
            admit_interpretation(contract.as_ref(), &captured, candidate),
            context_graph::InterpretationAdmissionRefusalReason::AnchorRefused,
        );
    }
    let mut removed: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
    removed["anchors"]
        .as_array_mut()
        .expect("fixture anchor array")
        .remove(0);
    assert_admission_reason(
        admit_interpretation(
            contract.as_ref(),
            &captured,
            serde_json::from_value(removed).expect("valid missing-anchor fixture"),
        ),
        context_graph::InterpretationAdmissionRefusalReason::PreservationContractRefused,
    );
}

#[test]
fn build_refuses_unpublished_unauthorized_and_declaration_mismatched_releases_by_type() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: root.path().join("interpretation.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let (mapper, contract) = installed();
    let request =
        |target: &str,
         mapper: Arc<dyn ReleasedConnectorInterpretation>,
         contract: Arc<dyn InstalledInterpretationContract>| InterpretationBuildRequest {
            store: config.clone(),
            context_id: captured().event.context_id,
            connector_id: "fixture-connector".into(),
            target_interpretation_version: target.into(),
            mapper,
            contract,
            max_events_per_batch: NonZeroU32::new(2).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4096).unwrap(),
            lease_duration_ms: NonZeroU64::new(1000).unwrap(),
        };
    assert!(matches!(
        ConnectorInterpretationRunner::build(request(
            "other-release",
            mapper.clone(),
            contract.clone()
        )),
        Err(CgError::InterpretationAdmissionRefused {
            reason: context_graph::InterpretationAdmissionRefusalReason::ReleasedTargetUnpublished,
            ..
        })
    ));

    let mut unauthorized = declaration();
    unauthorized.connector_id = "other-connector".into();
    let unauthorized_mapper: Arc<dyn ReleasedConnectorInterpretation> = Arc::new(FixtureMapper {
        declaration: unauthorized,
        answer: expected(),
    });
    assert!(matches!(
        ConnectorInterpretationRunner::build(request(
            "released-v1",
            unauthorized_mapper,
            contract.clone()
        )),
        Err(CgError::InterpretationAdmissionRefused {
            reason: context_graph::InterpretationAdmissionRefusalReason::ReleasedTargetUnauthorized,
            ..
        })
    ));

    let mut mismatch = declaration();
    mismatch.mapping_contract_sha256 = "other-contract-sha256".into();
    let mismatch_contract: Arc<dyn InstalledInterpretationContract> = Arc::new(FixtureContract {
        declaration: mismatch,
        expected_captured: captured(),
        required: expected(),
    });
    assert!(matches!(
        ConnectorInterpretationRunner::build(request("released-v1", mapper, mismatch_contract)),
        Err(CgError::InterpretationAdmissionRefused {
            reason: context_graph::InterpretationAdmissionRefusalReason::DeclarationMismatch,
            ..
        })
    ));
}

fn store_with_structurally_equivalent_remainders()
-> (tempfile::TempDir, Store, context_graph::ContextId) {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("gaps.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("open store");
    let context_id = store
        .create_context(CreateContext {
            name: "interpretation gaps".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let declaration = EventAdapterDeclaration {
        connector_id: "fixture-connector".into(),
        adapter_version: "capture-v1".into(),
        anchor_declaration_version: "anchors-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/line".into()],
    };
    let draft = |key: &str, private: &str| EventDraft {
        session_ref: "fixture-session".into(),
        source_event_key: key.into(),
        source_position: json!({"line": if key == "one" { 1 } else { 2 }}),
        occurred_at: None,
        author: "capture".into(),
        kind: "unknown".into(),
        render_role: EventRenderRole::Opaque,
        interaction_ref: None,
        text: None,
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({"unsupported": {"shape": "same", "private": private}}),
        coverage: EventCoverage::Opaque,
    };
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration,
            "fixture-source",
            vec![draft("one", "first"), draft("two", "second")],
        ))
        .expect("retain captured events");
    (root, store, context_id)
}

#[test]
fn equivalent_unsupported_shapes_converge_on_one_value_free_adapter_coverage_gap() {
    let (_root, store, context_id) = store_with_structurally_equivalent_remainders();
    let page = store
        .list_adapter_coverage_gaps(
            ContextSelector::one(context_id),
            "fixture-connector".into(),
            Page {
                after: None,
                limit: NonZeroU32::new(1).unwrap(),
                affected_event_handle_limit: NonZeroU32::new(2).unwrap(),
            },
        )
        .expect("value-free gap page");
    assert_eq!(page.gaps.len(), 1);
    assert_eq!(page.gaps[0].state, AdapterCoverageGapState::Uncovered);
    assert_eq!(page.gaps[0].occurrence_count, 2);
    assert_eq!(page.gaps[0].affected_event_ids.len(), 2);
    assert!(page.next_after.is_none());
    assert!(!format!("{:?}", page.gaps[0]).contains("first"));
    assert!(!format!("{:?}", page.gaps[0]).contains("second"));
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn positive_activation_retry_is_idempotent() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(InterpretationProjectionEmbedder::new());
    let registration = embedder.registration();
    let config = StoreConfig {
        db_path: root.path().join("dual-head.db"),
        extra_embedding_spaces: vec![embedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store = Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
        .expect("open the database-only projection fixture");
    let context_id = store
        .create_context(CreateContext {
            name: "dual head interpretation".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create context")
        .id;
    let captured = captured();
    let retained = EventDraft {
        session_ref: captured.event.session_ref.clone(),
        source_event_key: captured.event.source_event_key.clone(),
        source_position: captured.event.source_position.clone(),
        occurred_at: captured.event.occurred_at,
        author: captured.event.author.clone(),
        kind: captured.event.kind.clone(),
        render_role: captured.event.render_role,
        interaction_ref: captured.event.interaction_ref.clone(),
        text: captured.event.text.clone(),
        payload: captured.event.payload.clone(),
        anchors: json!({"file": "src/lib.rs"}),
        source_remainder: captured.event.source_remainder.clone(),
        coverage: captured.event.coverage,
    };
    let appended = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "fixture-connector".into(),
                adapter_version: "capture-v1".into(),
                anchor_declaration_version: "anchors-v1".into(),
                anchor_schema: json!({"type": "object"}),
                source_position_order: vec!["/line".into()],
            },
            captured.event.source_instance_ref.clone(),
            vec![retained],
        ))
        .expect("retained event is sufficient without a source-file reread");

    let actual_event = appended
        .events
        .into_iter()
        .next()
        .expect("one retained event")
        .event;
    let (fixture_mapper, contract) =
        installed_for_capture(expected_persisted_capture(actual_event));
    let mapper: Arc<dyn ReleasedConnectorInterpretation> = fixture_mapper;
    drop(store);

    let interpretation = ConnectorInterpretationRunner::build(InterpretationBuildRequest {
        store: config.clone(),
        context_id,
        connector_id: "fixture-connector".into(),
        target_interpretation_version: "released-v1".into(),
        mapper: mapper.clone(),
        contract: contract.clone(),
        max_events_per_batch: NonZeroU32::new(1).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4096).unwrap(),
        lease_duration_ms: NonZeroU64::new(1000).unwrap(),
    })
    .expect("build derives from retained events");
    let search_store =
        Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
            .expect("open caller store for search projection");
    let search = search_store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: Some(interpretation.generation_id),
            embedder: registration.clone(),
            profile: EventSearchProfile::default(),
            max_events_per_batch: NonZeroU32::new(1).unwrap(),
            max_passages_per_batch: NonZeroU32::new(1).unwrap(),
            max_bytes_per_batch: NonZeroUsize::new(4096).unwrap(),
            lease_duration_ms: NonZeroU64::new(1000).unwrap(),
        })
        .expect("explicit staged search rebuild produces the activation generation");
    assert_eq!(
        search.interpretation_generation_ids,
        vec![interpretation.generation_id],
        "the replacement search generation is sealed to the requested interpretation"
    );
    assert!(
        !search.active_generation_changed,
        "a staged replacement stays invisible until dual-head activation"
    );
    drop(search_store);
    let request = InterpretationActivationRequest {
        store: config.clone(),
        context_id,
        connector_id: "fixture-connector".into(),
        target_interpretation_version: "released-v1".into(),
        mapper,
        contract,
        interpretation_generation_id: interpretation.generation_id,
        replacement_search_generation_id: search.generation_id,
    };
    let activated = ConnectorInterpretationRunner::activate(request.clone())
        .expect("matching complete public build receipts activate both heads atomically");
    assert!(activated.changed);
    let visible_store =
        Store::open_with_embedder_registrations(config.clone(), vec![registration.clone()])
            .expect("open store to inspect the atomically visible replacement");
    let interpretation_head = visible_store
        .execute_sql_for_test(
            "SELECT generation_id, activated_at FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
            &HashMap::from([
                ("context_id".into(), DbValue::Uuid(context_id.as_uuid())),
                ("connector_id".into(), DbValue::Text("fixture-connector".into())),
            ]),
        )
        .expect("inspect interpretation head");
    assert_eq!(
        interpretation_head.rows,
        vec![vec![
            DbValue::Uuid(interpretation.generation_id.as_uuid()),
            DbValue::TxId(activated.transaction_id),
        ]],
        "the interpretation head records the activation transaction"
    );
    let search_head = visible_store
        .execute_sql_for_test(
            "SELECT generation_id, activated_at FROM event_search_heads WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), DbValue::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect search head");
    assert_eq!(
        search_head.rows,
        vec![vec![
            DbValue::Uuid(search.generation_id.as_uuid()),
            DbValue::TxId(activated.transaction_id),
        ]],
        "the search head records the same activation transaction"
    );
    let active_exact_anchor = visible_store
        .execute_sql_for_test(
            "SELECT canonical_value FROM event_interpretation_anchors WHERE context_id = $context_id AND generation_id = $generation_id AND key = $key",
            &HashMap::from([
                ("context_id".into(), DbValue::Uuid(context_id.as_uuid())),
                (
                    "generation_id".into(),
                    DbValue::Uuid(interpretation.generation_id.as_uuid()),
                ),
                ("key".into(), DbValue::Text("file".into())),
            ]),
        )
        .expect("inspect the exact-anchor projection selected by the active head");
    assert_eq!(
        active_exact_anchor.rows,
        vec![vec![DbValue::Text("src/lib.rs".into())]],
        "the activated interpretation's exact anchor is visible"
    );
    let active_search = visible_store
        .find_event_history(
            ContextSelector::one(context_id),
            "released text",
            context_graph::EventSearchFilter::default(),
        )
        .expect("the replacement search generation is publicly visible");
    assert_eq!(active_search.hits.len(), 1);
    assert_eq!(active_search.hits[0].generation_id, search.generation_id);
    assert_eq!(
        active_search.hits[0].text,
        "Assistant [author=\"released-author\"]: released text"
    );
    let retry_snapshot = (
        interpretation_head.rows,
        search_head.rows,
        active_exact_anchor.rows,
        active_search.hits,
    );
    drop(visible_store);
    let retry = ConnectorInterpretationRunner::activate(request)
        .expect("retry of an already active complete pair is an idempotent receipt");
    assert!(!retry.changed);
    assert_eq!(
        retry.interpretation_generation_id,
        activated.interpretation_generation_id
    );
    assert_eq!(retry.search_generation_id, activated.search_generation_id);
    let retry_store = Store::open_with_embedder_registrations(config, vec![registration])
        .expect("reopen store after idempotent retry");
    let after_retry = (
        retry_store
            .execute_sql_for_test(
                "SELECT generation_id, activated_at FROM event_interpretation_heads WHERE context_id = $context_id AND connector_id = $connector_id",
                &HashMap::from([
                    ("context_id".into(), DbValue::Uuid(context_id.as_uuid())),
                    ("connector_id".into(), DbValue::Text("fixture-connector".into())),
                ]),
            )
            .expect("inspect interpretation head after retry")
            .rows,
        retry_store
            .execute_sql_for_test(
                "SELECT generation_id, activated_at FROM event_search_heads WHERE context_id = $context_id",
                &HashMap::from([("context_id".into(), DbValue::Uuid(context_id.as_uuid()))]),
            )
            .expect("inspect search head after retry")
            .rows,
        retry_store
            .execute_sql_for_test(
                "SELECT canonical_value FROM event_interpretation_anchors WHERE context_id = $context_id AND generation_id = $generation_id AND key = $key",
                &HashMap::from([
                    ("context_id".into(), DbValue::Uuid(context_id.as_uuid())),
                    (
                        "generation_id".into(),
                        DbValue::Uuid(interpretation.generation_id.as_uuid()),
                    ),
                    ("key".into(), DbValue::Text("file".into())),
                ]),
            )
            .expect("inspect exact-anchor projection after retry")
            .rows,
        retry_store
            .find_event_history(
                ContextSelector::one(context_id),
                "released text",
                context_graph::EventSearchFilter::default(),
            )
            .expect("search after idempotent retry")
            .hits,
    );
    assert_eq!(
        after_retry, retry_snapshot,
        "idempotent retry writes no replacement rows"
    );
}
