//! Regressions for the released-interpretation persistence foundation.

use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::{
    CapturedEventAndAnchors, CgError, EffectiveEventFields, EventCoverage,
    InstalledInterpretationContract, InterpretationAdmissionRefusalReason,
    ReleasedInterpretationDeclaration, Result, admit_interpretation,
};
use serde_json::json;

const CAPTURED: &str = include_str!("fixtures/interpretation/captured-event-and-anchors.json");

fn captured() -> CapturedEventAndAnchors {
    serde_json::from_str(CAPTURED).expect("independently authored captured fixture")
}

fn declaration() -> ReleasedInterpretationDeclaration {
    ReleasedInterpretationDeclaration {
        connector_id: "fixture-connector".into(),
        target_interpretation_version: "released-anchor-remap-v1".into(),
        target_adapter_version: "capture-v2".into(),
        target_anchor_declaration_version: "effective-anchors-v2".into(),
        accepted_source_declarations: vec![context_graph::EventDeclarationVersion {
            adapter_version: "capture-v1".into(),
            anchor_declaration_version: "anchors-v1".into(),
        }],
        mapping_contract_id: "authorized-anchor-remap".into(),
        mapping_contract_sha256: "authorized-anchor-remap-sha256".into(),
    }
}

fn authorized_replacement() -> EffectiveEventFields {
    EffectiveEventFields {
        occurred_at: Some(
            "2024-01-02T03:04:05.678Z"
                .parse()
                .expect("millisecond timestamp"),
        ),
        author: "released-author".into(),
        kind: "released-kind".into(),
        render_role: context_graph::EventRenderRole::AgentMessage,
        interaction_ref: Some("released-interaction-7".into()),
        text: Some("released text".into()),
        payload: json!({"released": true, "source": "fixture"}),
        anchors: json!([{
            "key": "repository_path",
            "canonical_value": "{\"path\":\"src/remapped.rs\"}",
            "original_value": {"path": "src/remapped.rs"}
        }]),
        coverage: EventCoverage::Complete,
    }
}

struct RemappingContract {
    declaration: ReleasedInterpretationDeclaration,
    authorized: EffectiveEventFields,
    validations: AtomicUsize,
}

impl InstalledInterpretationContract for RemappingContract {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration {
        &self.declaration
    }

    fn validate(
        &self,
        _captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        if candidate == &self.authorized {
            Ok(())
        } else {
            Err(CgError::NotFound {
                id: None,
                missing: Some("candidate is outside the installed released mapping"),
            })
        }
    }
}

#[test]
fn installed_validator_authorizes_complete_anchor_replacement_and_refuses_other_candidates() {
    let captured = captured();
    let immutable_capture = captured.clone();
    let authorized = authorized_replacement();
    let contract = RemappingContract {
        declaration: declaration(),
        authorized: authorized.clone(),
        validations: AtomicUsize::new(0),
    };

    admit_interpretation(&contract, &captured, authorized)
        .expect("the installed release may rename and remap the complete effective anchor set");

    let mut unauthorized = authorized_replacement();
    unauthorized.anchors = json!([{
        "key": "repository_path",
        "canonical_value": "src/unpublished.rs",
        "original_value": "src/unpublished.rs"
    }]);
    assert!(matches!(
        admit_interpretation(&contract, &captured, unauthorized),
        Err(CgError::InterpretationAdmissionRefused {
            reason: InterpretationAdmissionRefusalReason::PreservationContractRefused,
            ..
        })
    ));

    assert_eq!(contract.validations.load(Ordering::SeqCst), 2);
    assert_eq!(
        captured, immutable_capture,
        "effective anchor replacement must not mutate captured event or anchor facts"
    );
}

#[cfg(context_graph_internal_test_support)]
mod bounded_status {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use context_graph::{
        ContextId, ContextSelector, CreateContext, InterpretationGenerationState, Store,
        StoreConfig, TxId,
    };
    use contextdb_core::Value;
    use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
    use uuid::Uuid;

    #[derive(Default)]
    struct QueryRecorder {
        enabled: AtomicBool,
        queries: Mutex<Vec<String>>,
    }

    impl QueryRecorder {
        fn enable(&self) {
            self.queries.lock().expect("query recorder lock").clear();
            self.enabled.store(true, Ordering::SeqCst);
        }

        fn queries(&self) -> Vec<String> {
            self.queries.lock().expect("query recorder lock").clone()
        }
    }

    impl DatabasePlugin for QueryRecorder {
        fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
            if self.enabled.load(Ordering::SeqCst) {
                self.queries
                    .lock()
                    .expect("query recorder lock")
                    .push(sql.to_owned());
            }
            Ok(())
        }

        fn post_query(&self, _sql: &str, _duration: std::time::Duration, _outcome: &QueryOutcome) {}
    }

    struct Generation<'a> {
        id: Uuid,
        target: &'a str,
        started_at: u64,
        event_count: u64,
        interpreted_event_count: u64,
        state: &'a str,
        failure: Option<&'a str>,
    }

    fn insert_generation(store: &Store, context_id: ContextId, generation: Generation<'_>) {
        let params = HashMap::from([
            ("id".into(), Value::Uuid(generation.id)),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            (
                "connector_id".into(),
                Value::Text("fixture-connector".into()),
            ),
            (
                "accepted".into(),
                Value::Json(serde_json::json!([{
                    "adapter_version": "capture-v1",
                    "anchor_declaration_version": "anchors-v1"
                }])),
            ),
            ("target".into(), Value::Text(generation.target.into())),
            ("target_adapter".into(), Value::Text("capture-v2".into())),
            (
                "target_anchors".into(),
                Value::Text("effective-anchors-v2".into()),
            ),
            ("contract".into(), Value::Text("fixture-contract".into())),
            (
                "contract_sha".into(),
                Value::Text("fixture-contract-sha256".into()),
            ),
            (
                "event_count".into(),
                Value::Int64(i64::try_from(generation.event_count).expect("fixture count")),
            ),
            (
                "watermark_sha".into(),
                Value::Text(format!("watermark-{}", generation.id)),
            ),
            (
                "source_sha".into(),
                Value::Text(format!("source-{}", generation.id)),
            ),
            (
                "interpreted".into(),
                Value::Int64(
                    i64::try_from(generation.interpreted_event_count).expect("fixture count"),
                ),
            ),
            ("state".into(), Value::Text(generation.state.into())),
            (
                "failure".into(),
                generation
                    .failure
                    .map_or(Value::Null, |failure| Value::Text(failure.into())),
            ),
            (
                "started_at".into(),
                Value::TxId(TxId::from_raw_wire(generation.started_at)),
            ),
        ]);
        store
            .execute_sql_for_test(
                "INSERT INTO event_interpretation_generations (id, context_id, connector_id, accepted_source_declarations, target_interpretation_version, target_adapter_version, target_anchor_declaration_version, mapping_contract_id, mapping_contract_sha256, event_count, ordered_source_set_sha256, source_content_sha256, interpreted_event_count, state, sanitized_failure, started_at) VALUES ($id, $context_id, $connector_id, $accepted, $target, $target_adapter, $target_anchors, $contract, $contract_sha, $event_count, $watermark_sha, $source_sha, $interpreted, $state, $failure, $started_at)",
                &params,
            )
            .expect("insert interpretation generation fixture");
    }

    #[test]
    fn status_without_a_head_pages_generation_history_and_selects_a_stable_latest_tie() {
        const HISTORY_ROWS: u64 = 257;
        let directory = tempfile::tempdir().expect("tempdir");
        let recorder = Arc::new(QueryRecorder::default());
        let store = Store::open_with_database_plugin(
            StoreConfig {
                db_path: directory.path().join("bounded-interpretation-status.db"),
                default_text_embedder: None,
                ..StoreConfig::default()
            },
            recorder.clone(),
        )
        .expect("open instrumented Store");
        let context_id = store
            .create_context(CreateContext {
                name: "bounded interpretation status".into(),
                labels: vec![],
                properties: Default::default(),
            })
            .expect("create Context")
            .id;

        for ordinal in 0..HISTORY_ROWS {
            insert_generation(
                &store,
                context_id,
                Generation {
                    id: Uuid::from_u128(ordinal as u128 + 0x100),
                    target: "history",
                    started_at: ordinal + 1,
                    event_count: ordinal,
                    interpreted_event_count: ordinal,
                    state: "retired",
                    failure: None,
                },
            );
        }

        let stable_latest = Uuid::from_u128(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe);
        insert_generation(
            &store,
            context_id,
            Generation {
                id: stable_latest,
                target: "stable-latest",
                started_at: HISTORY_ROWS,
                event_count: 4_242,
                interpreted_event_count: 4_200,
                state: "failed",
                failure: Some("stable-latest-failure"),
            },
        );
        insert_generation(
            &store,
            context_id,
            Generation {
                id: Uuid::from_u128(1),
                target: "same-transaction-lower-id",
                started_at: HISTORY_ROWS,
                event_count: 3_131,
                interpreted_event_count: 3_100,
                state: "building",
                failure: None,
            },
        );

        recorder.enable();
        let answer = store
            .interpretation_status(ContextSelector::one(context_id), "fixture-connector".into())
            .expect("status reads a bounded latest generation without an active head");
        let status = answer.contexts.first().expect("selected Context status");
        assert!(status.active_generation_id.is_none());
        assert_eq!(
            status
                .source_watermark
                .as_ref()
                .map(|watermark| watermark.event_count),
            Some(4_242)
        );
        assert_eq!(status.interpreted_event_count, 4_200);
        assert_eq!(status.state, Some(InterpretationGenerationState::Failed));
        assert_eq!(
            status.sanitized_failure.as_deref(),
            Some("stable-latest-failure")
        );

        let queries = recorder
            .queries()
            .into_iter()
            .filter(|sql| sql.contains("FROM event_interpretation_generations"))
            .collect::<Vec<_>>();
        assert_eq!(
            queries.len(),
            1,
            "status issues one generation-history read"
        );
        let query = queries[0].to_ascii_uppercase();
        assert!(
            !query.contains("ORDER BY") && !query.contains("LIMIT"),
            "generation history must be paged and folded by the caller, never fully materialized and sorted before LIMIT: {}",
            queries[0]
        );
    }
}
