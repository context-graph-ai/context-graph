use context_graph::{
    CgError, ContextBindingKey, ContextId, EventPlacementPolicy, SourceHistoryDivergence,
};
use serde_json::json;

fn assert_human_display(error: CgError, required: &[&str], enum_name: &str) {
    let display = error.to_string();
    for fragment in required {
        assert!(
            display.contains(fragment),
            "Display output must contain {fragment:?}; got {display:?}"
        );
    }
    assert!(
        !display.contains(enum_name),
        "Display output must not expose a Rust enum dump; got {display:?}"
    );
}

#[test]
fn typed_import_errors_explain_the_failure_boundary_and_next_step() {
    assert_human_display(
        CgError::SourceHistoryDiverged {
            source_instance_ref: "session-42".into(),
            cause: SourceHistoryDivergence::PrefixHashMismatch {
                checkpoint_offset: 18,
                expected_sha256: "old-hash".into(),
                actual_sha256: "new-hash".into(),
            },
        },
        &[
            "source history for `session-42` no longer matches its committed checkpoint",
            "this source was not changed by this attempt",
            "Do not retry it until you restore the previously imported history",
        ],
        "SourceHistoryDiverged",
    );

    assert_human_display(
        CgError::SourceRecordMalformed {
            connector_id: "claude".into(),
            session_ref: "session-42".into(),
            position: json!({"line": 3}),
            reason: "invalid JSON".into(),
        },
        &[
            "record at {\"line\":3} in claude session `session-42` is malformed",
            "Fix the record before importing this source again",
            "earlier committed batches remain available",
        ],
        "SourceRecordMalformed",
    );

    assert_human_display(
        CgError::TenantPolicyBindingRequired {
            expected: EventPlacementPolicy::TenantHub,
        },
        &[
            "authenticated tenant-hub binding",
            "no Context binding, event schema, or event row was written",
            "Finish tenant enrollment, then retry import",
        ],
        "TenantPolicyBindingRequired",
    );

    assert_human_display(
        CgError::EventPlacementPolicyMismatch {
            expected: EventPlacementPolicy::TenantHub,
            actual: Some(EventPlacementPolicy::LocalOnly),
            reason: "existing store".into(),
        },
        &[
            "requires tenant hub, but the store is configured for local only",
            "made no placement change",
            "and retry the import",
        ],
        "EventPlacementPolicyMismatch",
    );

    assert_human_display(
        CgError::ContextBindingConflict {
            key: Box::new(ContextBindingKey {
                namespace: "project".into(),
                opaque_reference: "context-graph".into(),
            }),
            requested_context_id: ContextId::default(),
        },
        &[
            "project reference `project/context-graph` is already linked to a different Context",
            "this binding was not changed",
            "Use the Context already linked to this reference",
        ],
        "ContextBindingConflict",
    );

    assert_human_display(
        CgError::AppendContention {
            context_id: ContextId::default(),
            connector_id: "claude".into(),
            source_instance_ref: "session-42".into(),
        },
        &[
            "another writer changed source `session-42` while this import was appending",
            "no event batch was committed for this contention",
            "Wait for the other writer to exit, then retry this batch",
        ],
        "AppendContention",
    );

    assert_human_display(
        CgError::EmbedderModelNotAvailable {
            source: "local cache".into(),
            path: "/tmp/models/nomic-embed-text-v1.5".into(),
        },
        &[
            "text-embedding model from local cache is not available",
            "/tmp/models/nomic-embed-text-v1.5",
            "cg model fetch --defaults",
        ],
        "EmbedderModelNotAvailable",
    );
}
