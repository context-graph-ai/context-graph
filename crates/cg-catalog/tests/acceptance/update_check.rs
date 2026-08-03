use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use cg_catalog::{CG_CATALOG_CONTEXT_ID, CgCatalogError, UpdateCheckObservation, UpdatePolicy};
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DecisionStatus, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    IntentionId, InvalidationDetectedPayload, ObservationId, RecordObservation, RetentionStatus,
    Store, SubscriptionFilter,
};
use serde_json::json;

use crate::common::{
    GEMMA_ID, NOMIC_BYTES, NOMIC_ID, NOMIC_REVISION, TEST_NEW_NOMIC_REVISION, catalog_at,
    fixture_dir, separate_store, write_nomic_cache, write_source_override,
    write_valid_mini_gemma_acceptance,
};

const CG_BINARY_ID: &str = "cg-binary";
const TEST_NEW_CG_BINARY_RELEASE: &str = "cg-2026-05-28-test-release";

fn json_props(entries: &[(&str, serde_json::Value)]) -> BTreeMap<String, serde_json::Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn ensure_catalog_context(store: &Store) {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: EntityId::new_v7(),
            context_id: *CG_CATALOG_CONTEXT_ID,
            observation_type: "catalog_context_bootstrap".to_string(),
            source: "cg-catalog-acceptance".to_string(),
            observed_at: chrono::Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id: *CG_CATALOG_CONTEXT_ID,
                kind: EvidenceKind::Text,
                source_ref: "cas:cg-catalog-context-bootstrap".to_string(),
                mime_type: Some("text/plain".to_string()),
                producer: EvidenceProducer {
                    system: "cg-catalog".to_string(),
                    model_name: "catalog-context".to_string(),
                    model_version: "bootstrap".to_string(),
                    pipeline_version: "acceptance-test".to_string(),
                },
                text_projection: Some("bootstrap catalog context".to_string()),
                retention_status: RetentionStatus::Ephemeral,
                ..EvidenceRef::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("catalog context bootstrap observation");
}

fn setup_revision_decision(
    store: &Store,
) -> (
    ContextId,
    EntityId,
    DecisionId,
    IntentionId,
    Arc<Mutex<Vec<InvalidationDetectedPayload>>>,
) {
    ensure_catalog_context(store);
    let context_id = *CG_CATALOG_CONTEXT_ID;
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Configuration,
            name: NOMIC_ID.to_string(),
            properties: json_props(&[
                ("model_id", json!(NOMIC_ID)),
                ("current_revision", json!("old-revision")),
            ]),
            context_id,
            ..Default::default()
        })
        .expect("entity");
    let intention = store
        .create_intention(CreateIntention {
            description: "keep model revision decisions fresh".to_string(),
            context_id,
            ..Default::default()
        })
        .expect("intention");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(entity.id, vec!["current_revision".to_string()]);
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "model-lifecycle".to_string(),
            description: "use current Nomic pinned revision".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: Some(basis_fields),
            context_id,
            ..Default::default()
        })
        .expect("decision");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock()
                .expect("payload collector")
                .push(payload.clone());
            Ok(())
        })
        .expect("subscribe invalidation");
    (context_id, entity.id, decision.id, intention.id, payloads)
}

fn stub_base_url(server: &tiny_http::Server) -> String {
    let addr = server.server_addr().to_ip().expect("stub server TCP addr");
    format!("http://{}:{}", addr.ip(), addr.port())
}

struct UpdateFixtureServer {
    shutdown: Arc<AtomicBool>,
    server_thread: Option<std::thread::JoinHandle<()>>,
}

impl UpdateFixtureServer {
    fn stop_and_join(&mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        match self.server_thread.take() {
            Some(server_thread) => server_thread.join(),
            None => Ok(()),
        }
    }

    fn shutdown_and_join(mut self) {
        self.stop_and_join().expect("server thread joins");
    }
}

impl Drop for UpdateFixtureServer {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn serve_nomic_fixture_until_idle_counted(
    server: tiny_http::Server,
    revision: &'static str,
    request_count: Arc<AtomicUsize>,
) -> UpdateFixtureServer {
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match server.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(Some(request)) => {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    let url = request.url().to_string();
                    let fixture = fixture_dir("nomic");
                    let bytes = if url.ends_with("REVISION") {
                        revision.as_bytes().to_vec()
                    } else if url.ends_with("model.safetensors") {
                        std::fs::read(fixture.join("model.safetensors")).expect("fixture model")
                    } else if url.ends_with("tokenizer.json") {
                        std::fs::read(fixture.join("tokenizer.json")).expect("fixture tokenizer")
                    } else {
                        std::fs::read(fixture.join("config.json")).expect("fixture config")
                    };
                    let response = tiny_http::Response::from_data(bytes);
                    let _ = request.respond(response);
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    UpdateFixtureServer {
        shutdown,
        server_thread: Some(server_thread),
    }
}

fn serve_nomic_fixture_until_idle(server: tiny_http::Server) -> UpdateFixtureServer {
    serve_nomic_fixture_until_idle_counted(
        server,
        TEST_NEW_NOMIC_REVISION,
        Arc::new(AtomicUsize::new(0)),
    )
}

fn serve_revision_source_until_idle_counted(
    server: tiny_http::Server,
    revision: &'static str,
    request_count: Arc<AtomicUsize>,
    request_urls: Arc<Mutex<Vec<String>>>,
) -> UpdateFixtureServer {
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match server.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(Some(request)) => {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    let url = request.url().to_string();
                    request_urls
                        .lock()
                        .expect("request url log")
                        .push(url.clone());
                    let bytes = if url.ends_with("REVISION") {
                        revision.as_bytes().to_vec()
                    } else {
                        Vec::new()
                    };
                    let _ = request.respond(tiny_http::Response::from_data(bytes));
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    UpdateFixtureServer {
        shutdown,
        server_thread: Some(server_thread),
    }
}

fn start_update_revision_source(
    root: &std::path::Path,
    configured_revision: &'static str,
    upstream_revision: &'static str,
) -> (Arc<AtomicUsize>, UpdateFixtureServer) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, configured_revision);
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_thread = serve_nomic_fixture_until_idle_counted(
        server,
        upstream_revision,
        Arc::clone(&request_count),
    );
    (request_count, server_thread)
}

fn start_empty_revision_source(
    root: &std::path::Path,
) -> (
    Arc<AtomicUsize>,
    Arc<Mutex<Vec<String>>>,
    UpdateFixtureServer,
) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(root, NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(AtomicUsize::new(0));
    let request_paths = Arc::new(Mutex::new(Vec::new()));
    let counter = Arc::clone(&request_count);
    let paths = Arc::clone(&request_paths);
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let server_thread = std::thread::spawn(move || {
        while !server_shutdown.load(Ordering::SeqCst) {
            match server.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(Some(request)) => {
                    counter.fetch_add(1, Ordering::SeqCst);
                    let url = request.url().to_string();
                    paths.lock().expect("request paths").push(url.clone());
                    let response = if request.url().ends_with("REVISION") {
                        tiny_http::Response::from_string("")
                    } else {
                        tiny_http::Response::from_string("unexpected non-revision request")
                            .with_status_code(500)
                    };
                    let _ = request.respond(response);
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    (
        request_count,
        request_paths,
        UpdateFixtureServer {
            shutdown,
            server_thread: Some(server_thread),
        },
    )
}

fn wait_for_payload(payloads: &Arc<Mutex<Vec<InvalidationDetectedPayload>>>) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        if !payloads.lock().expect("payload collector").is_empty() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn cg_update_check_empty_source_revision_fails_loudly_without_observation_or_cache_mutation() {
    // A configured source is the authority for update checks; an empty REVISION is not a no-op.
    let temp = tempfile::tempdir().expect("tempdir");
    let (request_count, request_paths, server_thread) = start_empty_revision_source(temp.path());
    let (_store_temp, store) = separate_store();
    let result = catalog_at(temp.path(), "host-A").update_check(&store);
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "update_check should consult the configured REVISION source before reporting success"
    );
    assert!(
        request_paths
            .lock()
            .expect("request paths")
            .iter()
            .any(|path| path.ends_with("REVISION")),
        "update_check must request the upstream REVISION artifact from the configured source"
    );
    assert!(matches!(
        result,
        Err(CgCatalogError::SourceUnavailable { model_id, message })
            if model_id == NOMIC_ID && message.contains("REVISION")
    ));
    assert_eq!(
        store
            .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
            .expect("observations readable"),
        Vec::new(),
        "invalid upstream REVISION must not fabricate update observations"
    );
    assert!(
        !temp.path().join(NOMIC_ID).join("catalog.toml").exists(),
        "update_check must not publish cache state when the source revision is invalid"
    );
}

#[test]
fn cg_update_check_produces_observation_on_per_model_entity_with_new_revision_and_release_notes_url()
 {
    // Update checks should create observations through the existing Store observation path.
    let temp = tempfile::tempdir().expect("tempdir");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_thread = serve_nomic_fixture_until_idle_counted(
        server,
        TEST_NEW_NOMIC_REVISION,
        Arc::clone(&request_count),
    );
    let (_store_temp, store) = separate_store();
    let returned = catalog_at(temp.path(), "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "update_check should consult the configured upstream revision source"
    );
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(
        rows.iter()
            .all(|observation| observation.context_id == *CG_CATALOG_CONTEXT_ID)
    );
    assert!(rows.iter().any(|observation| {
        returned.iter().any(|candidate| {
            candidate.entity_id == observation.entity_id.to_string()
                && candidate.state_delta.get("new_revision")
                    == Some(&serde_json::json!(TEST_NEW_NOMIC_REVISION))
        }) && observation.observation_type == "model_update_available"
            && observation.state_delta.get("model_id") == Some(&Some(serde_json::json!(NOMIC_ID)))
            && observation.state_delta.get("new_revision")
                == Some(&Some(serde_json::json!(TEST_NEW_NOMIC_REVISION)))
            && observation.state_delta.get("current_revision")
                == Some(&Some(serde_json::json!(TEST_NEW_NOMIC_REVISION)))
            && observation.state_delta.contains_key("release_notes_url")
    }));
    let user_context = store
        .create_context(CreateContext {
            name: "ordinary-user-context".to_string(),
            ..Default::default()
        })
        .expect("user context");
    assert_eq!(
        store
            .list_observations(Some(user_context.id))
            .expect("ordinary context observations"),
        Vec::new(),
        "catalog update checks should not write rows into arbitrary user contexts"
    );
}

#[test]
fn cg_update_check_with_no_upstream_changes_writes_zero_observations_returns_empty_vec() {
    // No-op update checks should be quiet and return an empty result set.
    let temp = tempfile::tempdir().expect("tempdir");
    write_nomic_cache(temp.path());
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, TEST_NEW_NOMIC_REVISION);
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_thread =
        serve_nomic_fixture_until_idle_counted(server, NOMIC_REVISION, Arc::clone(&request_count));
    let (_store_temp, store) = separate_store();
    let observations = catalog_at(temp.path(), "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    assert!(
        request_count.load(Ordering::SeqCst) > 0,
        "no-change update_check should still consult the configured upstream revision source"
    );
    assert_eq!(
        observations,
        Vec::<UpdateCheckObservation>::new(),
        "no upstream revision change should not fabricate update observations"
    );
    assert_eq!(
        store
            .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
            .expect("observations readable"),
        Vec::new()
    );
}

#[test]
fn cg_update_check_observation_consumed_by_impact_analysis_invalidates_decisions_whose_basis_cites_current_revision()
 {
    // A model revision observation should invalidate decisions whose basis cites the old revision.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store();
    let (context_id, entity_id, decision_id, _intention_id, _payloads) =
        setup_revision_decision(&store);
    assert_eq!(context_id, *CG_CATALOG_CONTEXT_ID);
    let (_request_count, server_thread) =
        start_update_revision_source(temp.path(), NOMIC_REVISION, TEST_NEW_NOMIC_REVISION);
    let _observations = catalog_at(temp.path(), "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    let persisted = rows.iter().find(|observation| {
        observation.entity_id == entity_id
            && observation.observation_type == "model_update_available"
    });
    assert!(
        persisted.is_some(),
        "update check should persist a Store::record_observation row"
    );
    let persisted = persisted.unwrap();
    assert_eq!(
        persisted.state_delta.get("current_revision"),
        Some(&Some(serde_json::json!(TEST_NEW_NOMIC_REVISION))),
        "impact analysis should see the same current_revision field named by basis_fields"
    );
    let invalidations = store
        .list_invalidations(*CG_CATALOG_CONTEXT_ID, Some(decision_id), None)
        .expect("invalidations readable");
    assert!(
        invalidations
            .iter()
            .any(|invalidation| invalidation.trigger_observation_id == persisted.id),
        "impact analysis should consume the persisted observation row, not a returned property stub"
    );
    let decision = store
        .get_decision(decision_id)
        .expect("decision readable")
        .expect("decision exists");
    assert_eq!(
        decision.status,
        DecisionStatus::Invalidated,
        "the shipped impact-analysis path should transition the affected decision"
    );
}

#[test]
fn cg_update_check_observation_consumed_by_subscriptions_pushes_intention_anchored_message_to_configured_sink()
 {
    // Subscribers should receive an intention-anchored invalidation event from the observation.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store();
    let (context_id, _entity_id, decision_id, intention_id, payloads) =
        setup_revision_decision(&store);
    assert_eq!(context_id, *CG_CATALOG_CONTEXT_ID);
    let (_request_count, server_thread) =
        start_update_revision_source(temp.path(), NOMIC_REVISION, TEST_NEW_NOMIC_REVISION);
    let _observations = catalog_at(temp.path(), "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    wait_for_payload(&payloads);
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    let persisted = rows
        .iter()
        .find(|observation| observation.observation_type == "model_update_available");
    assert!(
        persisted.is_some(),
        "update check should persist a Store::record_observation row before subscription delivery"
    );
    let persisted = persisted.unwrap();
    assert_eq!(
        persisted.state_delta.get("current_revision"),
        Some(&Some(serde_json::json!(TEST_NEW_NOMIC_REVISION))),
        "subscription delivery should be driven by the same state field the decision cited"
    );
    let payloads = payloads.lock().expect("payload collector");
    let payload = payloads.iter().find(|payload| {
        payload.invalidation.affected_decision_id == decision_id
            && payload.invalidation.trigger_observation_id == persisted.id
    });
    assert!(
        payload.is_some(),
        "subscription delivery should be driven by the persisted invalidation"
    );
    let payload = payload.unwrap();
    assert_eq!(
        payload.threatened_intention_descriptions.get(&intention_id),
        Some(&"keep model revision decisions fresh".to_string())
    );
}

#[test]
fn cg_update_check_default_is_report_only_does_not_apply_updates() {
    // Checking for updates should not mutate the cache or active default unless auto is requested.
    let temp = tempfile::tempdir().expect("tempdir");
    let cache_root = temp.path().join("cache");
    let model_dir = write_nomic_cache(&cache_root);
    let (_request_count, server_thread) =
        start_update_revision_source(&cache_root, NOMIC_REVISION, TEST_NEW_NOMIC_REVISION);
    let (_store_temp, store) = separate_store();
    let before_revision =
        std::fs::read_to_string(model_dir.join("REVISION")).expect("revision readable");
    let observations = catalog_at(&cache_root, "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    let after_revision =
        std::fs::read_to_string(model_dir.join("REVISION")).expect("revision readable");
    assert_eq!(after_revision, before_revision);
    assert!(
        observations
            .iter()
            .any(
                |observation| observation.observation_type == "model_update_available"
                    && observation.state_delta.get("applied") == Some(&serde_json::json!(false))
            ),
        "report-only update checks should emit an unapplied update observation"
    );
}

#[test]
fn cg_update_check_auto_flag_applies_apache_updates_and_refuses_to_apply_license_gated_updates_silently()
 {
    // Auto update can advance Apache models but must refuse gated models with a visible record.
    let temp = tempfile::tempdir().expect("tempdir");
    let nomic_dir = write_nomic_cache(temp.path());
    let (gemma_dir, record) = write_valid_mini_gemma_acceptance(temp.path(), "host-A");
    let catalog = catalog_at(temp.path(), "host-A");
    catalog
        .record_acceptance(GEMMA_ID, record)
        .expect("pre-existing acceptance for old gated revision is recorded");
    let before_nomic_revision =
        std::fs::read_to_string(nomic_dir.join("REVISION")).expect("nomic revision readable");
    let before_gemma_revision =
        std::fs::read_to_string(gemma_dir.join("REVISION")).expect("gemma revision readable");
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server");
    let base_url = stub_base_url(&server);
    write_source_override(temp.path(), NOMIC_ID, &base_url, NOMIC_REVISION);
    write_source_override(
        temp.path(),
        GEMMA_ID,
        &base_url,
        "embeddinggemma-fixture-revision",
    );
    let server_thread = serve_nomic_fixture_until_idle(server);
    let (_store_temp, store) = separate_store();
    let observations = catalog
        .update(&store, UpdatePolicy::Auto)
        .expect("update returns");
    server_thread.shutdown_and_join();
    let after_nomic_revision =
        std::fs::read_to_string(nomic_dir.join("REVISION")).expect("nomic revision readable");
    let after_gemma_revision =
        std::fs::read_to_string(gemma_dir.join("REVISION")).expect("gemma revision readable");
    assert_ne!(after_nomic_revision, before_nomic_revision);
    assert_eq!(after_nomic_revision.trim(), TEST_NEW_NOMIC_REVISION);
    assert_eq!(
        after_gemma_revision, before_gemma_revision,
        "license-gated model revision must not be advanced by Auto"
    );
    assert!(
        observations
            .iter()
            .any(|observation| observation.observation_type == "manual_acceptance_required")
    );
    let rows = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(rows.iter().any(|observation| {
        observation.observation_type == "model_update_available"
            && observation.state_delta.get("model_id") == Some(&Some(serde_json::json!(NOMIC_ID)))
            && observation.state_delta.get("applied") == Some(&Some(serde_json::json!(true)))
    }));
    assert!(rows.iter().any(|observation| {
        observation.observation_type == "manual_acceptance_required"
            && observation.state_delta.get("model_id") == Some(&Some(serde_json::json!(GEMMA_ID)))
    }));
}

#[test]
fn cg_update_check_includes_test_fixtures_requiring_re_baseline_section_per_affected_model() {
    // Revision-coupled fixture output tells maintainers what test baselines must move with a model.
    let temp = tempfile::tempdir().expect("tempdir");
    let cache_root = temp.path().join("cache");
    let (_request_count, server_thread) =
        start_update_revision_source(&cache_root, NOMIC_REVISION, TEST_NEW_NOMIC_REVISION);
    let (_store_temp, store) = separate_store();
    let observations = catalog_at(&cache_root, "host-A")
        .update_check(&store)
        .expect("update check returns");
    server_thread.shutdown_and_join();
    assert!(observations.iter().any(|observation| {
        observation
            .revision_coupled_fixtures
            .contains(&format!("NOMIC_BYTE_SIZE={NOMIC_BYTES}"))
    }));
}

#[test]
fn cg_update_check_writes_observation_on_per_binary_cg_entity_for_cg_binary_itself() {
    // The cg binary update path is tracked as its own entity, not as graph catalog state.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store();
    let nomic_server = tiny_http::Server::http("127.0.0.1:0").expect("nomic stub server");
    let nomic_base_url = stub_base_url(&nomic_server);
    write_source_override(temp.path(), NOMIC_ID, &nomic_base_url, NOMIC_REVISION);
    let binary_server = tiny_http::Server::http("127.0.0.1:0").expect("binary stub server");
    let binary_base_url = format!("{}/cg-binary-release-source", stub_base_url(&binary_server));
    write_source_override(
        temp.path(),
        CG_BINARY_ID,
        &binary_base_url,
        env!("CARGO_PKG_VERSION"),
    );
    let nomic_request_count = Arc::new(AtomicUsize::new(0));
    let nomic_request_urls = Arc::new(Mutex::new(Vec::new()));
    let nomic_server_thread = serve_revision_source_until_idle_counted(
        nomic_server,
        TEST_NEW_NOMIC_REVISION,
        Arc::clone(&nomic_request_count),
        Arc::clone(&nomic_request_urls),
    );
    let binary_request_count = Arc::new(AtomicUsize::new(0));
    let binary_request_urls = Arc::new(Mutex::new(Vec::new()));
    let binary_server_thread = serve_revision_source_until_idle_counted(
        binary_server,
        TEST_NEW_CG_BINARY_RELEASE,
        Arc::clone(&binary_request_count),
        Arc::clone(&binary_request_urls),
    );
    let returned_observations = catalog_at(temp.path(), "host-A")
        .update_check(&store)
        .expect("update check returns");
    nomic_server_thread.shutdown_and_join();
    binary_server_thread.shutdown_and_join();
    assert!(
        binary_request_count.load(Ordering::SeqCst) > 0,
        "binary update check should poll its configured release source, not infer from model updates"
    );
    let binary_urls = binary_request_urls.lock().expect("binary request urls");
    assert!(
        binary_urls
            .iter()
            .any(|url| url.contains("/cg-binary-release-source/") && url.ends_with("REVISION")),
        "binary update check must request REVISION from the configured cg-binary source path, urls={binary_urls:?}"
    );
    assert!(
        returned_observations.iter().any(|observation| {
            observation.observation_type == "binary_update_available"
                && observation.state_delta.get("new_release")
                    == Some(&serde_json::json!(TEST_NEW_CG_BINARY_RELEASE))
        }),
        "returned binary update must carry the release tag read from the cg-binary source, observations={returned_observations:?}"
    );
    let observations = store
        .list_observations(Some(*CG_CATALOG_CONTEXT_ID))
        .expect("observations readable");
    assert!(observations.iter().any(|observation| {
        observation.observation_type == "binary_update_available"
            && observation.properties.get("binary") == Some(&serde_json::json!("cg"))
            && observation.state_delta.get("new_release")
                == Some(&Some(serde_json::json!(TEST_NEW_CG_BINARY_RELEASE)))
    }));
}
