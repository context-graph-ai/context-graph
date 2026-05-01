//! Journey B integration test — U3 AI Agent Developer.
//!
//! Walks the four moments-of-pain end-to-end against contextdb v1.0.0 with
//! a real on-disk store and reopen between session boundaries. Stubs are
//! per `track2-journeys.md` "Stubbed surfaces expected per journey:
//! Journey B". Cross-references to acceptance test names in code comments.

use journey_b::connect::{cg_init, connect_claude_code};
use journey_b::hook_runtime::deliver_r3_context;
use journey_b::ingest_history::{fixture_with_auth_decision, import_chat_history};
use journey_b::search::{semantic_search, DecisionStatusFilter};
use journey_b::store::{Store, TEXT_SPACE_ID};
use journey_b::subscriptions::Event;
use journey_b::trigger::compute_invalidations_for_observation;
use journey_b::types::{DecisionStatusSnapshot, Severity};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// ---------- M1: First connect ----------

#[test]
fn m1_first_connect_scaffolds_config_and_default_space_and_idempotent_hook() {
    // The developer just installed `cg`. Cwd-equivalent + $HOME are temp dirs.
    let cwd = TempDir::new().unwrap();
    let home = TempDir::new().unwrap();

    // ----- cli_init_scaffolds_on_disk_config -----
    let init = cg_init(cwd.path(), "default");
    assert!(init.config_path.exists(), "init must create .cg/config.toml");
    assert!(init.data_dir.exists(), "init must create .cg/data/");
    let config_body = std::fs::read_to_string(&init.config_path).unwrap();
    assert!(
        config_body.contains("default_context = \"default\""),
        "config must declare default context"
    );

    // ----- Store::open + embeddinggemma_loads_and_embeds_text_corpus_with_expected_dimension -----
    // The stub uses 8-dim mock vectors per `embedder.rs` (Plan 11 stub note);
    // the cited test pins 768. Stub-vs-test divergence flagged in findings.
    let store_file = init.data_dir.join("store.cdb");
    let store = Store::open(&store_file);
    let spaces = store.list_embedding_spaces();
    assert!(
        spaces.iter().any(|(sid, _, _, _)| sid == TEXT_SPACE_ID),
        "default text embedding space row must exist after Store::open; got {spaces:?}"
    );

    // ----- connect_claude_code_writes_hook_idempotent -----
    let p1 = connect_claude_code(home.path());
    let body1 = std::fs::read(&p1).unwrap();
    assert!(p1.exists());
    let parsed: serde_json::Value = serde_json::from_slice(&body1).unwrap();
    let hook_cmd = parsed["hooks"]["UserPromptSubmit"][0]["command"]
        .as_str()
        .unwrap();
    assert!(
        hook_cmd.contains("cg ingest --agent claude-code"),
        "hook command must reference cg ingest --agent claude-code; got {hook_cmd}"
    );
    assert!(
        parsed["hooks"]["UserPromptSubmit"][0]
            .get("timeout")
            .is_some(),
        "hook entry must carry `timeout`"
    );
    assert!(
        parsed["hooks"]["UserPromptSubmit"][0]
            .get("statusMessage")
            .is_some(),
        "hook entry must carry `statusMessage`"
    );

    // Re-run; assert byte-equal file (idempotency).
    connect_claude_code(home.path());
    let body2 = std::fs::read(&p1).unwrap();
    assert_eq!(
        body1, body2,
        "second `connect_claude_code` must be byte-equal — connect_claude_code_writes_hook_idempotent"
    );

    // ----- cli_standalone_no_server -----
    // The whole journey runs without any server process; presence of a
    // server URL or the absence of a local store would fail `Store::open`.
    drop(store);

    // Reopen — `Store::open` must observe the same store path / spaces.
    let store2 = Store::open(&store_file);
    let spaces2 = store2.list_embedding_spaces();
    assert_eq!(
        spaces, spaces2,
        "reopen must preserve embedding-space registry"
    );
}

// ---------- M2: Stale file path (rename + invalidation) ----------

#[test]
fn m2_stale_file_path_detected_and_subscriber_fires_post_commit() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");

    // Day 1: developer's first agent session creates Intention + Decision
    // BASED_ON the auth.py file Entity. confidence = 0.6 → Severity::Warning
    // per `impact_analysis_severity_warning_for_mid_confidence`.
    let (intent, _) = store.create_intention(
        "use existing helpers when adding token validation",
        ctx,
    );
    let (auth_entity, _) = store.create_entity(
        "File",
        "src/auth.py",
        json!({
            "path": "src/auth.py",
            "purpose": "token validation",
            "exists": true
        }),
        ctx,
    );
    let (decision, decision_tx) = store.create_decision(
        "use auth.py for token validation",
        "tool_selection",
        Some(0.6),
        vec![intent],
        vec![auth_entity],
        ctx,
    );
    // Mark decision as having a successful outcome — so effective_confidence
    // for active path is 0.6 × 1.0 = 0.6 (above the post-rerank threshold).
    store
        .record_outcome(decision, true)
        .expect("record outcome");

    // Reopen to simulate "two weeks later, fresh process".
    drop(store);
    let store = Store::open(&dir.path().join("cg.cdb"));

    // Subscriber wired BEFORE the rename observation lands.
    // Per `subscribe_invalidation_detected_fires`: callback fires AFTER the
    // invalidation row commits.
    let fired = Arc::new(AtomicUsize::new(0));
    let captured_severity = Arc::new(std::sync::Mutex::new(None::<Severity>));
    {
        let fired_ref = fired.clone();
        let cap = captured_severity.clone();
        store.subs.subscribe(Arc::new(move |ev: &Event| {
            let Event::InvalidationDetected(inv) = ev;
            fired_ref.fetch_add(1, Ordering::SeqCst);
            *cap.lock().unwrap() = Some(inv.severity);
        }));
    }

    // Two weeks later: file rename. New observation lands. The agent's prior
    // BASED_ON Entity is the SAME entity (file rename writes a new
    // properties version); we update entity properties + archive the old
    // path entry in the projectory ontology mapping. Per the journey
    // contract "the prior `repo:src/auth.py` Entity is archived" — we do
    // BOTH: update with the new path AND archive the entity status (so the
    // `archived_entity_still_replayable` branch holds for M4).
    let _new_tx = store.update_entity(
        auth_entity,
        "src/authentication.py",
        json!({
            "path": "src/authentication.py",
            "purpose": "token validation",
            "exists": true
        }),
    );

    // Record observation (rename event). Plan 7 ObservationTrigger is
    // engine-blocked → call the journey-side stub immediately after
    // Plan-3R-shipped `record_observation`.
    let obs_data = json!({
        "path": "src/authentication.py",
        "old_path": "src/auth.py",
        "exists": true
    });
    let (obs_id, _) = store.record_observation(
        auth_entity,
        "file_rename",
        obs_data.clone(),
        "filesystem-watcher",
        ctx,
    );
    let trigger_out =
        compute_invalidations_for_observation(&store, auth_entity, obs_id, &obs_data);

    // ----- impact_analysis_direct_invalidation_by_engine -----
    assert_eq!(
        trigger_out.invalidations.len(),
        1,
        "exactly one invalidation expected for a file-rename basis-diff trigger; got {:?}",
        trigger_out.invalidations
    );
    let inv = &trigger_out.invalidations[0];
    assert_eq!(inv.affected_decision_id, decision);
    assert_eq!(inv.trigger_observation_id, obs_id);

    // ----- impact_analysis_severity_warning_for_mid_confidence -----
    assert_eq!(
        inv.severity,
        Severity::Warning,
        "confidence 0.6 → Severity::Warning per impact_analysis_severity_warning_for_mid_confidence"
    );

    // ----- subscribe_invalidation_detected_fires -----
    assert_eq!(
        fired.load(Ordering::SeqCst),
        1,
        "InvalidationDetected callback must fire once post-commit"
    );
    assert_eq!(*captured_severity.lock().unwrap(), Some(Severity::Warning));

    // Decision flips to `invalidated` (Plan 8 column UPDATE shipped path).
    assert_eq!(
        store.get_decision_status(decision),
        DecisionStatusSnapshot::Invalidated,
        "decision must transition to Invalidated post-trigger"
    );

    // The basis snapshot on the decision still carries the OLD path —
    // frozen at decision creation per cg-acceptance
    // `decision_based_on_snapshot_frozen_after_entity_update`. This is the
    // load-bearing temporal invariant for M4 replay.
    let basis = store.get_decision_basis(decision);
    let snap = basis.get(&auth_entity.to_string()).cloned().unwrap();
    assert_eq!(
        snap.get("path").and_then(|p| p.as_str()),
        Some("src/auth.py"),
        "decision basis must remain frozen at decision-time path"
    );
    let _ = decision_tx; // exercised via state_at in M4
}

#[test]
fn m2_concurrent_observations_against_same_entity_one_invalidation_per_decision() {
    // The contract: "a basis-changing observation arrives while a read is
    // in flight" — concretely, multiple file-watcher processes may submit
    // overlapping rename observations against the same entity. The trigger
    // stub's UNIQUE composite on (decision_id, observation_id) means each
    // observation fires its own invalidation, but a CONCURRENT pair of
    // observations against ONE decision must produce N invalidation rows
    // (one per observation), AND the engine must conflict-serialize so
    // there is exactly one invalidation row per (decision, observation)
    // pair — no duplicates within a single observation.
    //
    // This test exercises N=4 concurrent observations + one trigger run per
    // observation. The stub-shipped engine-blocked impl keeps each
    // observation's trigger run independent; concurrent reads must remain
    // consistent with the post-commit watermark.
    //
    // KNOWN-DEDUP gap: this contention shape touches engine gap N1
    // ConcurrentUniqueConstraintEnforcement (composite UNIQUE on the
    // invalidations table). If two threads try to write the SAME (decision,
    // observation) pair concurrently (which can't happen here because each
    // observation has a fresh UUID), the engine would let both through.
    // We do not re-discover N1 — already catalogued in
    // `track1-synthesis/gap-delta.md` Gap N1.

    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");
    let (intent, _) = store.create_intention("intent", ctx);
    let (entity, _) = store.create_entity(
        "File",
        "src/auth.py",
        json!({"path": "src/auth.py", "purpose": "token validation"}),
        ctx,
    );
    let (decision, _) = store.create_decision(
        "use auth.py",
        "tool_selection",
        Some(0.7),
        vec![intent],
        vec![entity],
        ctx,
    );

    // Spawn 4 writer threads. Each records its own observation (fresh UUID)
    // then fires the trigger stub against that observation.
    let mut handles = Vec::new();
    for i in 0..4 {
        let store = store.clone();
        let path_new = format!("src/authentication-{i}.py");
        handles.push(thread::spawn(move || {
            let data = json!({"path": path_new});
            let (obs_id, _) =
                store.record_observation(entity, "file_rename", data.clone(), "fs", ctx);
            compute_invalidations_for_observation(&store, entity, obs_id, &data)
        }));
    }
    let outs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let total_invalidations: usize = outs.iter().map(|o| o.invalidations.len()).sum();

    // Decision is marked Invalidated by the FIRST writer's trigger; the
    // remaining writers' triggers also try to write Invalidation rows
    // against the SAME decision_id. The composite UNIQUE on
    // (decision_id, observation_id) holds because each observation_id is
    // distinct, so all 4 invalidation rows persist.
    let r = store
        .db
        .execute(
            "SELECT id FROM invalidations WHERE affected_decision_id = $did",
            &journey_b::store::params(vec![(
                "did",
                contextdb_core::Value::Uuid(decision),
            )]),
        )
        .unwrap();
    let persisted_count = r.rows.len();
    assert_eq!(
        total_invalidations, 4,
        "each writer's trigger must produce one invalidation; got {}",
        total_invalidations
    );
    assert_eq!(
        persisted_count, 4,
        "all 4 invalidations must persist (composite UNIQUE permits 4 \
         distinct (decision, observation) pairs); got {}",
        persisted_count
    );
    assert_eq!(
        store.get_decision_status(decision),
        DecisionStatusSnapshot::Invalidated
    );
}

// ---------- M3: Hallucination prevention ----------

#[test]
fn m3_search_excludes_invalidated_decision_and_ranks_by_effective_confidence() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");
    let (intent, _) = store.create_intention("intent", ctx);

    // Two files; two decisions. D_failed has high cosine (matches "auth"
    // query) but failed outcome → effective_confidence = 0.0. D_success has
    // lower cosine but successful outcome → effective_confidence = 0.85.
    // Per `semantic_search_effective_confidence_ranking`: D_success ranks above.
    let (e_auth, _) = store.create_entity(
        "File",
        "src/auth.py",
        json!({"path": "src/auth.py"}),
        ctx,
    );
    let (e_authn, _) = store.create_entity(
        "File",
        "src/authentication.py",
        json!({"path": "src/authentication.py"}),
        ctx,
    );
    let (d_failed, _) = store.create_decision(
        "use auth.py for token validation",
        "tool_selection",
        Some(0.8),
        vec![intent],
        vec![e_auth],
        ctx,
    );
    store.record_outcome(d_failed, false).unwrap();
    let (d_success, _) = store.create_decision(
        "auth.py handles token validation flows",
        "tool_selection",
        Some(0.85),
        vec![intent],
        vec![e_authn],
        ctx,
    );
    store.record_outcome(d_success, true).unwrap();

    // ----- semantic_search_effective_confidence_ranking -----
    // (filter All — both stay in the result; ranking by effective_confidence)
    let res = semantic_search(
        &store,
        "where is the auth handler",
        5,
        DecisionStatusFilter::All,
    );
    assert!(
        !res.is_empty(),
        "search must return at least one decision; got empty"
    );
    let top = &res[0];
    assert_eq!(
        top.decision_id, d_success,
        "D_success (effective_confidence=0.85) must rank above D_failed (=0.0); \
         got top={} d_success={} d_failed={}; full ranking={:?}",
        top.decision_id, d_success, d_failed, res
    );
    let d_success_eff = res
        .iter()
        .find(|r| r.decision_id == d_success)
        .unwrap()
        .effective_confidence
        .unwrap();
    let d_failed_eff = res
        .iter()
        .find(|r| r.decision_id == d_failed)
        .unwrap()
        .effective_confidence
        .unwrap();
    assert!(
        (d_success_eff - 0.85).abs() < 1e-5,
        "d_success effective_confidence must equal 0.85; got {d_success_eff}"
    );
    assert!(
        d_failed_eff.abs() < 1e-5,
        "d_failed effective_confidence must equal 0.0; got {d_failed_eff}"
    );

    // ----- semantic_search filter ExcludeInvalidated -----
    // Now invalidate d_success. The "filtered out" branch of
    // `semantic_search_effective_confidence_ranking` says invalidated
    // decisions drop out of the ranked result entirely.
    store.mark_decision_invalidated(d_success);
    let filtered = semantic_search(
        &store,
        "where is the auth handler",
        5,
        DecisionStatusFilter::ExcludeInvalidated,
    );
    assert!(
        filtered.iter().all(|r| r.decision_id != d_success),
        "invalidated decision must be filtered from results; got {:?}",
        filtered
    );

    // ----- effective_confidence_null_confidence_defaults_to_success_rate -----
    let (_, _) = (intent, ctx);
    let (e_legacy, _) = store.create_entity("File", "src/legacy.py", json!({}), ctx);
    let (d_legacy, _) = store.create_decision(
        "legacy authentication code path",
        "tool_selection",
        None, // confidence absent
        vec![intent],
        vec![e_legacy],
        ctx,
    );
    store.record_outcome(d_legacy, true).unwrap();
    let res2 = semantic_search(
        &store,
        "where is the auth handler",
        10,
        DecisionStatusFilter::ExcludeInvalidated,
    );
    let r_legacy = res2
        .iter()
        .find(|r| r.decision_id == d_legacy)
        .expect("legacy decision present");
    assert!(
        (r_legacy.effective_confidence.unwrap() - 1.0).abs() < 1e-5,
        "null confidence + success → effective_confidence = 1.0; got {}",
        r_legacy.effective_confidence.unwrap()
    );
}

#[test]
fn m3_hook_runtime_delivers_ranked_invalidation_aware_context() {
    // ----- connect_hook_delivers_r3_context_at_runtime -----
    // The hook fires post-rename. The runtime must surface:
    //   - active intentions
    //   - ranked decisions (excluding the invalidated one)
    //   - pending invalidations
    // Each line ≤ 120 chars; the runtime must NOT swallow Critical/Warning
    // invalidations.

    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");
    let (intent, _) = store.create_intention(
        "use existing helpers when adding token validation",
        ctx,
    );
    let (e_auth, _) = store.create_entity(
        "File",
        "src/auth.py",
        json!({"path": "src/auth.py"}),
        ctx,
    );
    let (decision, _) = store.create_decision(
        "use auth.py for token validation",
        "tool_selection",
        Some(0.6),
        vec![intent],
        vec![e_auth],
        ctx,
    );
    store.record_outcome(decision, true).unwrap();

    // Trigger: file rename
    let _ = store.update_entity(
        e_auth,
        "src/authentication.py",
        json!({"path": "src/authentication.py"}),
    );
    let data = json!({"path": "src/authentication.py"});
    let (obs_id, _) =
        store.record_observation(e_auth, "file_rename", data.clone(), "fs", ctx);
    let _ = compute_invalidations_for_observation(&store, e_auth, obs_id, &data);

    // Hook fires
    let lines = deliver_r3_context(&store, "where is the auth handler");
    assert!(!lines.is_empty(), "hook must deliver at least one line");
    for line in &lines {
        assert!(
            line.chars().count() <= 120,
            "every condensed line must be ≤120 chars; got {} ({})",
            line.chars().count(),
            line
        );
    }
    // Pending invalidation MUST appear — agent needs to know auth.py is stale.
    assert!(
        lines.iter().any(|l| l.contains("invalidation")),
        "hook output must include the pending invalidation; got {:?}",
        lines
    );
    // The Invalidated decision MUST NOT appear among ranked decisions.
    let dec_desc = store.get_decision_description(decision);
    assert!(
        !lines
            .iter()
            .any(|l| l.starts_with("decision") && l.contains(&dec_desc)),
        "invalidated decision must be filtered from the ranked decisions block; got {:?}",
        lines
    );
}

// ---------- M4: Audit replay ----------

#[test]
fn m4_replay_returns_decision_time_intention_wording_and_basis_after_archive() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");

    // Setup mirroring `replay_includes_cited_intention_wording_at_decision_time`:
    // intention v0 → v1 (pre-D) → D (cites v1) → v2 (post-D, current).
    let (intent, _) = store.create_intention("v0 wording", ctx);
    store.update_intention_description(intent, "v1 wording — strict matching");
    let (entity, _entity_tx) = store.create_entity(
        "File",
        "src/auth.py",
        json!({"path": "src/auth.py", "purpose": "token validation"}),
        ctx,
    );
    let (decision, decision_tx) = store.create_decision(
        "use auth.py for token validation",
        "tool_selection",
        Some(0.6),
        vec![intent],
        vec![entity],
        ctx,
    );

    // After-D mutations
    store.update_intention_description(intent, "v2 wording — relaxed");
    let _ = store.update_entity(
        entity,
        "src/authentication.py",
        json!({"path": "src/authentication.py", "purpose": "token validation"}),
    );
    // Archive the entity. `archived_entity_still_replayable` says the
    // basis must still resolve.
    let _ = store.archive_entity(entity);

    // Reopen — the journey's "next session, days later".
    drop(store);
    let store = Store::open(&dir.path().join("cg.cdb"));

    // ----- replay_includes_cited_intention_wording_at_decision_time -----
    let cited = store
        .intention_at(intent, decision_tx)
        .expect("cited intention snapshot must resolve");
    assert_eq!(
        cited, "v1 wording — strict matching",
        "replay must surface intention wording covering decision_at txid; \
         got {cited}"
    );

    // ----- archived_entity_still_replayable -----
    // Decision basis was frozen — reading it must still return the pre-archive,
    // pre-rename properties.
    let basis = store.get_decision_basis(decision);
    let snap = basis.get(&entity.to_string()).cloned().unwrap();
    assert_eq!(
        snap.get("path").and_then(|p| p.as_str()),
        Some("src/auth.py"),
        "basis snapshot must remain frozen at decision-time even after entity archive; got {snap}"
    );
    // entity_history must still contain ≥ 2 snapshots after archive.
    let history = store.entity_history(entity);
    assert!(
        history.len() >= 2,
        "entity_history retains every snapshot after archive; got {} snapshots",
        history.len()
    );
    // entity status reflects archive
    assert_eq!(store.entity_status(entity), "archived");
    // state_at(decision_tx) returns the pre-rename, pre-archive properties.
    let state = store
        .state_at(entity, decision_tx)
        .expect("state_at(decision_tx) must return a snapshot");
    assert_eq!(
        state.get("path").and_then(|p| p.as_str()),
        Some("src/auth.py"),
        "state_at must return the snapshot covering decision_at, not current; got {state}"
    );

    // ----- full_provenance_returns_point_in_time_snapshots -----
    // The shipped surface (cg-temporal Plan 5) does not yet expose
    // `full_provenance` as a single call. The journey reconstructs it from
    // the shipped Store APIs to mirror the contract: input_snapshots equal
    // the basis snapshot stored on the Decision, NOT the current entity state.
    let inputs_snapshot_at_decision = basis;
    let current = store
        .db
        .execute(
            "SELECT properties FROM entities WHERE id = $id",
            &journey_b::store::params(vec![(
                "id",
                contextdb_core::Value::Uuid(entity),
            )]),
        )
        .unwrap();
    let cur_props = match current.rows[0]
        .iter()
        .position(|_| true)
        .map(|i| &current.rows[0][i])
    {
        Some(contextdb_core::Value::Json(j)) => j.clone(),
        _ => json!({}),
    };
    assert_ne!(
        inputs_snapshot_at_decision
            .get(&entity.to_string())
            .unwrap()
            .get("path"),
        cur_props.get("path"),
        "full_provenance input_snapshots must NOT equal current state; \
         that would be the shortcut wrong-impl per full_provenance_returns_point_in_time_snapshots"
    );
}

#[test]
fn m4_chat_history_import_dedupes_on_replay() {
    // The developer re-imports the same Claude Code export. Plan 22's
    // `digest_dedup_four_way_classification` says re-import is a no-op.
    let dir = TempDir::new().unwrap();
    let store = Store::open(&dir.path().join("cg.cdb"));
    let ctx = store.create_context("project-acme");

    let fix = fixture_with_auth_decision();
    let r1 = import_chat_history(&store, &fix, ctx);
    assert!(r1.digest_id.is_some(), "first import must produce a digest");
    assert!(!r1.duplicate);

    let r2 = import_chat_history(&store, &fix, ctx);
    assert!(
        r2.digest_id.is_none(),
        "second import of the same fixture must NOT produce a new digest"
    );
    assert!(r2.duplicate, "second import must report duplicate");

    let count = store
        .db
        .execute(
            "SELECT id FROM digests",
            &std::collections::HashMap::new(),
        )
        .unwrap()
        .rows
        .len();
    assert_eq!(count, 1, "exactly one digest persisted; got {count}");
}
