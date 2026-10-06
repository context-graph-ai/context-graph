//! The `cg` event-history journey: import a project's Claude Code history, then find, show, status
//! and forget it — all through the compiled binary against isolated synthetic history.

mod event_support;

use std::fs;
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use std::process::Child;
use std::process::{Command, Stdio};
use std::time::Duration;

use event_support::{
    Machine, Output, fingerprint, str_at, u64_at, wait_for_state, write_helper_transcript,
    write_session,
};
use serde_json::Value;
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
use serde_json::json;

/// Execute a retry string through POSIX `sh` only after this test has constructed every source
/// path and Context literal itself. This is a shell-tokenization proof, not an evaluator for
/// arbitrary fixture data or user input.
fn run_controlled_posix_retry(machine: &Machine, cwd: &std::path::Path, retry: &str) -> Output {
    let bin = machine.root.join("controlled-retry-bin");
    fs::create_dir_all(&bin).expect("create controlled retry PATH directory");
    match std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_cg"), bin.join("cg")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("place the compiled cg binary on the controlled PATH: {error}"),
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let controlled_path =
        std::env::join_paths(std::iter::once(bin.clone()).chain(std::env::split_paths(&path)))
            .expect("construct controlled retry PATH");
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(retry)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .env("PATH", controlled_path)
        .env("CG_DATA_ROOT", &machine.data_root)
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
        .env_remove("CG_NO_UNDERSTAND_KICK")
        .output()
        .expect("run controlled POSIX retry");
    Output {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
struct HandoffWorker {
    child: Option<Child>,
    release: std::path::PathBuf,
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
impl HandoffWorker {
    fn release_pass(&self, pass: usize) {
        fs::write(
            self.release.with_file_name(format!("release.{pass}")),
            b"continue\n",
        )
        .expect("release one completed worker pass");
    }

    fn release_and_join(&mut self) {
        fs::write(&self.release, b"continue\n").expect("release the owned handoff worker");
        let mut child = self.child.take().expect("owned handoff worker");
        match wait_for_state(Duration::from_secs(120), Duration::from_millis(20), || {
            child.try_wait()
        }) {
            Ok(Some(status)) => assert!(status.success(), "owned handoff worker exited {status}"),
            Ok(None) => {
                child.kill().expect("stop timed-out owned handoff worker");
                let status = child.wait().expect("join stopped handoff worker");
                panic!("owned handoff worker did not exit after release; stopped with {status}");
            }
            Err(error) => panic!("poll owned handoff worker: {error}"),
        }
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
impl Drop for HandoffWorker {
    fn drop(&mut self) {
        let _ = fs::write(&self.release, b"cleanup\n");
        let Some(mut child) = self.child.take() else {
            return;
        };
        match wait_for_state(Duration::from_secs(120), Duration::from_millis(20), || {
            child.try_wait()
        }) {
            Ok(Some(_)) | Err(_) => {}
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn wait_for_handoff_barrier(path: &std::path::Path) {
    wait_for_worker_marker(&path.join("entered"));
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn wait_for_worker_marker(path: &std::path::Path) {
    let reached = wait_for_state(Duration::from_secs(60), Duration::from_millis(10), || {
        Ok::<_, ()>(path.exists().then_some(()))
    })
    .expect("marker observation cannot fail");
    assert!(
        reached.is_some(),
        "worker never reached the post-catch-up handoff barrier at {}",
        path.display()
    );
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn spawn_handoff_worker(
    machine: &Machine,
    project: &std::path::Path,
    barrier: &std::path::Path,
) -> HandoffWorker {
    let launch_profile = serde_json::to_string(&json!({
        "profile": context_graph::EventSearchProfile::default(),
        "embedding_workers": 1,
        "ignored_publication": null,
    }))
    .expect("default worker envelope serializes");
    let child = Command::new(env!("CARGO_BIN_EXE_cg"))
        .args(["__event-search-worker", &launch_profile])
        .current_dir(project)
        .env("CG_DATA_ROOT", &machine.data_root)
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER", barrier)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
        .env_remove("CG_NO_UNDERSTAND_KICK")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn owned event-search worker");
    HandoffWorker {
        child: Some(child),
        release: barrier.join("release"),
    }
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn spawn_worker_before_first_assessment(
    machine: &Machine,
    project: &std::path::Path,
    barrier: &std::path::Path,
) -> Child {
    let launch_profile = serde_json::to_string(&json!({
        "profile": context_graph::EventSearchProfile::default(),
        "embedding_workers": 1,
        "ignored_publication": null,
    }))
    .expect("default worker envelope serializes");
    Command::new(env!("CARGO_BIN_EXE_cg"))
        .args(["__event-search-worker", &launch_profile])
        .current_dir(project)
        .env("CG_DATA_ROOT", &machine.data_root)
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env("CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER", barrier)
        .env("CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER", barrier)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
        .env_remove("CG_NO_UNDERSTAND_KICK")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn default event-search worker")
}

/// Install one live lease for an existing generation after every CLI process has released the
/// store. The worker under test sees a durable lease exactly as it would after an interrupted
/// process, while the fixture's short lifetime keeps the journey bounded.
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn seed_unexpired_event_search_lease(
    machine: &Machine,
    context_id: &str,
    generation_id: &str,
    lifetime: Duration,
) {
    use std::collections::HashMap;

    use contextdb_core::{Value as DbValue, Wallclock};
    use contextdb_engine::Database;

    let context_id = uuid::Uuid::parse_str(context_id).expect("Context id from import receipt");
    let generation_id =
        uuid::Uuid::parse_str(generation_id).expect("generation id from ready status receipt");
    let expires_at = i64::try_from(Wallclock::now().0.saturating_add(
        u64::try_from(lifetime.as_millis()).expect("fixture lease duration fits u64"),
    ))
    .expect("fixture lease expiry fits timestamp");
    let db = Database::open(machine.db_path()).expect("open durable fixture store");
    db.execute(
        "INSERT INTO event_search_leases (context_id, generation_id, owner_token, lease_epoch, claimed_connector_id, claimed_source_instance_ref, claimed_start_sequence, claimed_end_sequence, cursor_sha256, expires_at) VALUES ($context_id, $generation_id, $owner_token, $lease_epoch, $connector_id, $source, $start, $end, $cursor_sha256, $expires_at)",
        &HashMap::from([
            ("context_id".into(), DbValue::Uuid(context_id)),
            ("generation_id".into(), DbValue::Uuid(generation_id)),
            ("owner_token".into(), DbValue::Text("interrupted-worker".into())),
            ("lease_epoch".into(), DbValue::Int64(1)),
            ("connector_id".into(), DbValue::Null),
            ("source".into(), DbValue::Null),
            ("start".into(), DbValue::Null),
            ("end".into(), DbValue::Null),
            ("cursor_sha256".into(), DbValue::Text("fixture-orphan-lease".into())),
            ("expires_at".into(), DbValue::Timestamp(expires_at)),
        ]),
    )
    .expect("persist unexpired orphan lease");
}

fn hits(value: &Value) -> Vec<Value> {
    value
        .get("hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
#[derive(Debug)]
struct CliEventSearchEmbedder;

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
impl context_graph::Embedder for CliEventSearchEmbedder {
    fn embed(
        &self,
        _input: context_graph::EmbeddingInput,
    ) -> context_graph::Result<context_graph::EmbeddingOutput> {
        unreachable!("event search uses the role-aware entrypoints")
    }

    fn supported_inputs(&self) -> Vec<context_graph::EvidenceKind> {
        vec![context_graph::EvidenceKind::Text]
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        context_graph::EmbeddingSpace {
            embedding_space_id: "cli-test-event-search-768".into(),
            dimension: 768,
            metric: context_graph::DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "cli-event-search".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![context_graph::EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![context_graph::EmbeddingSpaceBinding {
                embedding_space_id: "cli-test-event-search-768".into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn encoding_profile_id(&self) -> String {
        "cli-test-event-search-v1".into()
    }

    fn event_search_capability(&self) -> Option<context_graph::EventSearchCapability> {
        context_graph::EventSearchCapability::new(2_048)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &context_graph::embedder::EventSearchSeal,
    ) -> context_graph::Result<context_graph::embedder::PreparedEventSearchDocument> {
        let mut token_ids = Vec::new();
        let mut boundaries = Vec::new();
        for (start, scalar) in text.char_indices() {
            token_ids.push(scalar as u32);
            boundaries.push((start, start + scalar.len_utf8()));
        }
        seal.document_builder()
            .token_count(token_ids.len())
            .byte_boundaries(boundaries)
            .payload(token_ids)
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &context_graph::embedder::EventSearchSeal,
    ) -> context_graph::Result<context_graph::embedder::PreparedEventSearchQuery> {
        let token_ids = text.chars().map(|scalar| scalar as u32).collect::<Vec<_>>();
        seal.query_builder()
            .token_count(token_ids.len())
            .payload(token_ids)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &context_graph::EventSearchInference,
    ) -> context_graph::Result<context_graph::EmbeddingOutput> {
        let mut vector = vec![0.0; 768];
        for (index, token) in inference.token_ids().iter().enumerate() {
            vector[index % 768] += (*token % 251) as f32 + 1.0;
        }
        Ok(context_graph::EmbeddingOutput {
            embedding_space_id: "cli-test-event-search-768".into(),
            dimension: 768,
            producer: context_graph::EvidenceProducer {
                system: "cli-test".into(),
                model_name: "cli-event-search".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        })
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &context_graph::EventSearchInference,
    ) -> context_graph::Result<context_graph::EmbeddingOutput> {
        self.embed_prepared_document_hook(inference)
    }
}

#[test]
fn cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history() {
    let machine = Machine::without_model();
    let project = machine.project("ledger");
    let other = machine.project("other-ledger");
    let history = machine.history_root();
    write_session(
        &history,
        &project,
        "sess-one",
        &[("alpha question", "alpha answer")],
    );
    write_session(
        &history,
        &project,
        "sess-two",
        &[
            ("beta question", "beta answer"),
            ("gamma question", "gamma answer"),
        ],
    );
    write_session(
        &history,
        &other,
        "sess-other",
        &[("other project question", "other project answer")],
    );
    let helper = write_helper_transcript(&history, &project, "sess-one", "helper1", "helper text");
    fs::write(
        project.join("stray.jsonl"),
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user",
                "uuid": "00000000-0000-4000-8000-0000000000aa",
                "sessionId": "stray",
                "cwd": project.display().to_string(),
                "message": {"role": "user", "content": "working tree file"}
            })
        ),
    )
    .expect("stray working-tree jsonl");

    let imported = machine.run(
        &project,
        &["import", ".", "--context", "ledger", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_deferred");
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(str_at(&receipt, "/selection/source"), "explicit");
    assert_eq!(str_at(&receipt, "/context/name"), "ledger");
    assert_eq!(
        u64_at(&receipt, "/sources/files"),
        2,
        "only this project's two session files"
    );
    assert_eq!(u64_at(&receipt, "/sources/sessions"), 2);
    assert_eq!(
        u64_at(&receipt, "/sources/helper_transcripts_not_imported"),
        1,
        "the helper transcript is counted, not silently dropped"
    );
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 6);
    assert_eq!(u64_at(&receipt, "/capture/local_embedding_calls"), 0);
    assert_eq!(u64_at(&receipt, "/capture/understanding_calls"), 0);
    assert_eq!(u64_at(&receipt, "/capture/hosted_provider_calls"), 0);
    assert_eq!(str_at(&receipt, "/search/status"), "pending");
    assert!(
        !imported.stdout.contains("working tree file")
            && !imported.stdout.contains("other project")
            && !imported.stdout.contains(&project.display().to_string()),
        "the receipt carries neither source text nor the project path: {}",
        imported.stdout
    );
    let context_id = str_at(&receipt, "/context/id").to_owned();

    // A symlink alias and an absolute path resolve the same project and the same binding.
    let link = machine.root.join("work").join("ledger-link");
    std::os::unix::fs::symlink(&project, &link).expect("project symlink");
    for target in [link.display().to_string(), project.display().to_string()] {
        let again = machine.run(&machine.root, &["import", &target, "--search", "defer"]);
        assert_eq!(again.code, 0, "{}", again.describe());
        let again = again.json();
        assert_eq!(str_at(&again, "/selection/source"), "project_binding");
        assert_eq!(str_at(&again, "/context/id"), context_id);
        assert_eq!(u64_at(&again, "/capture/events/inserted"), 0);
        assert_eq!(u64_at(&again, "/sources/files"), 2);
    }

    // Reads from a descendant directory reuse the project binding.
    let deep = project.join("src").join("deep");
    fs::create_dir_all(&deep).expect("descendant directory");
    let status = machine.run(&deep, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(str_at(&status, "/selection/source"), "project_binding");
    assert_eq!(str_at(&status, "/context/name"), "ledger");
    assert_eq!(u64_at(&status, "/history/captured_events"), 6);
    assert_eq!(u64_at(&status, "/history/sources/sessions"), 2);

    // A named helper transcript imports on its own, under its parent session.
    let named = machine.run(
        &machine.root,
        &[
            "import",
            &helper.display().to_string(),
            "--context",
            "ledger",
            "--search",
            "defer",
        ],
    );
    assert_eq!(named.code, 0, "{}", named.describe());
    let named = named.json();
    assert_eq!(u64_at(&named, "/capture/events/inserted"), 1);
    assert_eq!(str_at(&named, "/selection/source"), "explicit");
    let status = machine.run(&deep, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 7);
    assert_eq!(
        u64_at(&status, "/history/sources/sessions"),
        2,
        "the helper stream belongs to its parent session"
    );
}

#[test]
fn cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found()
 {
    let machine = Machine::without_model();
    let project = machine.project("no-history");
    let before = fingerprint(&machine.root);
    let output = machine.run(&project, &["import", ".", "--search", "defer"]);
    assert_eq!(output.code, 2, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(str_at(&receipt, "/state"), "project_history_not_found");
    assert!(
        !machine.data_root.exists(),
        "no data root or database was created"
    );
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "no write anywhere on the machine"
    );
}

#[test]
fn noninteractive_unbound_project_import_returns_the_retry_without_writing() {
    let machine = Machine::without_model();
    let project = machine.project("harbor");
    write_session(
        &machine.history_root(),
        &project,
        "sess-harbor",
        &[("harbor question", "harbor answer")],
    );
    let output = machine.run(&project, &["import", "."]);
    assert_eq!(output.code, 2, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "context_binding_required");
    assert_eq!(str_at(&receipt, "/suggested_context"), "harbor");
    let retry = str_at(&receipt, "/retry");
    assert!(
        retry.contains("cg import") && retry.contains("--context harbor"),
        "the retry is the exact command: {retry}"
    );
    assert!(
        receipt.get("capture").is_none(),
        "no fabricated capture progress"
    );
    assert!(
        !output.stdout.contains("Context name") && !output.stderr.contains("Context name"),
        "no prompt is written without a controlling terminal"
    );
    assert!(!machine.data_root.exists(), "nothing was written");
}

#[test]
fn a_bound_project_refuses_a_different_explicit_context_without_changing_either() {
    let machine = Machine::without_model();
    let project = machine.project("quarry");
    write_session(
        &machine.history_root(),
        &project,
        "sess-quarry",
        &[("quarry question", "quarry answer")],
    );
    let first = machine.run(
        &project,
        &["import", ".", "--context", "quarry", "--search", "defer"],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let same = machine.run(
        &project,
        &["import", ".", "--context", "quarry", "--search", "defer"],
    );
    assert_eq!(
        same.code,
        0,
        "an explicit same name is accepted: {}",
        same.describe()
    );
    let conflict = machine.run(
        &project,
        &["import", ".", "--context", "mine", "--search", "defer"],
    );
    assert_eq!(conflict.code, 2, "{}", conflict.describe());
    assert_eq!(
        str_at(&conflict.json(), "/state"),
        "context_binding_conflict"
    );
    let other = machine.run(&project, &["status", "--context", "mine"]);
    assert_ne!(
        other.code,
        0,
        "the conflicting Context was never created: {}",
        other.describe()
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(str_at(&status, "/context/name"), "quarry");
}

#[test]
fn retry_commands_quote_actual_paths_contexts_and_refusals() {
    let machine = Machine::without_model();
    let sentinel = machine.root.join("retry-command-injection-sentinel");
    let sentinel_token = sentinel
        .file_name()
        .and_then(|name| name.to_str())
        .expect("controlled sentinel file name");
    let project = machine
        .root
        .join("work")
        .join(format!("notes space 'quoted' $(touch {})", sentinel_token));
    fs::create_dir_all(&project).expect("create controlled special-character project");
    let project = fs::canonicalize(&project).expect("canonical special-character project");
    write_session(
        &machine.history_root(),
        &project,
        "quoted-retry",
        &[("what survives a pasted retry?", "each original argv token")],
    );
    let target = project.to_str().expect("UTF-8 controlled project path");

    let required = machine.run(&machine.root, &["import", target]);
    assert_eq!(required.code, 2, "{}", required.describe());
    let required = required.json();
    assert_eq!(str_at(&required, "/state"), "context_binding_required");
    let suggested = str_at(&required, "/suggested_context");
    assert_eq!(
        suggested,
        &format!("notes space 'quoted' $(touch {sentinel_token})")
    );
    let retry = str_at(&required, "/retry");
    assert!(
        retry.contains("'\"'\"'"),
        "apostrophe was not POSIX quoted: {retry}"
    );
    assert!(retry.contains("--search-sessions 10 --search-order newest"));

    let replayed = run_controlled_posix_retry(&machine, &machine.root, retry);
    assert_eq!(replayed.code, 1, "{}", replayed.describe());
    let replayed = replayed.json();
    assert_eq!(str_at(&replayed, "/state"), "captured_search_behind");
    assert_eq!(str_at(&replayed, "/context/name"), suggested);
    assert!(
        !sentinel.exists(),
        "the controlled metacharacter fixture escaped the printed retry"
    );

    let conflict = machine.run(
        &machine.root,
        &["import", target, "--context", "different Context"],
    );
    assert_eq!(conflict.code, 2, "{}", conflict.describe());
    let conflict = conflict.json();
    assert_eq!(str_at(&conflict, "/state"), "context_binding_conflict");
    assert!(
        str_at(&conflict, "/retry").contains("'\"'\"'"),
        "binding conflict lost the literal apostrophe in its target: {conflict}"
    );

    let tenant = Machine::without_model();
    let tenant_sentinel = tenant.root.join("tenant-retry-command-injection-sentinel");
    let tenant_sentinel_token = tenant_sentinel
        .file_name()
        .and_then(|name| name.to_str())
        .expect("controlled tenant sentinel file name");
    let tenant_project = tenant.root.join("work").join(format!(
        "tenant space 'quoted' $(touch {})",
        tenant_sentinel_token
    ));
    fs::create_dir_all(&tenant_project).expect("create controlled tenant project");
    let tenant_project = fs::canonicalize(&tenant_project).expect("canonical tenant project");
    let tenant_session_source = write_session(
        &tenant.history_root(),
        &tenant_project,
        "quoted-tenant-retry",
        &[(
            "what is the tenant retry?",
            "the original path as one shell token",
        )],
    );
    let tenant_session = tenant_session_source
        .with_file_name("tenant 'quoted' $(touch tenant-retry-command-injection-sentinel).jsonl");
    fs::rename(&tenant_session_source, &tenant_session)
        .expect("rename controlled tenant session to an injected exact-file target");
    fs::create_dir_all(tenant.data_root.join("cg")).expect("create tenant config directory");
    fs::write(
        tenant.data_root.join("cg").join("fabric.toml"),
        "[fabric]\nevent_placement = \"tenant_hub\"\n",
    )
    .expect("configure tenant-hub placement");
    let tenant_target = tenant_session
        .to_str()
        .expect("UTF-8 controlled tenant session path");
    let blocked = tenant.run(
        &tenant.root,
        &["import", tenant_target, "--context", "tenant Context"],
    );
    assert_eq!(blocked.code, 1, "{}", blocked.describe());
    let blocked = blocked.json();
    assert_eq!(str_at(&blocked, "/state"), "tenant_policy_blocked");
    assert!(
        str_at(&blocked, "/retry").contains("'\"'\"'"),
        "tenant refusal lost the literal target token: {blocked}"
    );
    assert!(
        str_at(&blocked, "/retry").contains("--context 'tenant Context'"),
        "tenant retry dropped the explicit Context required by an exact-file import: {blocked}"
    );
    let tenant_replayed =
        run_controlled_posix_retry(&tenant, &tenant.root, str_at(&blocked, "/retry"));
    assert_eq!(tenant_replayed.code, 1, "{}", tenant_replayed.describe());
    assert_eq!(
        str_at(&tenant_replayed.json(), "/state"),
        "tenant_policy_blocked"
    );
    assert!(
        !tenant_sentinel.exists(),
        "the tenant retry rendered its controlled metacharacter as shell syntax"
    );
}

#[test]
fn receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope() {
    let machine = Machine::without_model();
    let project = machine.project("receipt-placeholder");
    let session = write_session(
        &machine.history_root(),
        &project,
        "receipt-placeholder-session",
        &[("what is substituted?", "only the documented placeholder")],
    );
    let session_target = session.to_str().expect("UTF-8 controlled session path");
    let receipt = machine.run(
        &project,
        &[
            "import",
            session_target,
            "--context",
            "receipt Context",
            "--search",
            "defer",
            "--search-sessions",
            "all",
            "--search-order",
            "oldest",
        ],
    );
    assert_eq!(receipt.code, 0, "{}", receipt.describe());
    let receipt = receipt.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_deferred");
    let template = str_at(&receipt, "/retry");
    assert!(template.contains("'<session.jsonl>'"), "{template}");
    assert!(
        template.contains("--context 'receipt Context'"),
        "{template}"
    );
    let replay = template.replace("'<session.jsonl>'", &format!("'{session_target}'"));
    let replayed = run_controlled_posix_retry(&machine, &project, &replay);
    assert_eq!(replayed.code, 1, "{}", replayed.describe());
    let replayed = replayed.json();
    assert_eq!(str_at(&replayed, "/state"), "captured_search_behind");
    assert_eq!(u64_at(&replayed, "/capture/events/inserted"), 0);
    assert_eq!(
        str_at(&replayed, "/context/name"),
        "receipt Context",
        "the separately quoted exact-file Context did not reach the CLI"
    );

    let retained = machine.project("persisted-all-oldest");
    write_session(
        &machine.history_root(),
        &retained,
        "persisted-all-oldest-session",
        &[(
            "what does retry preserve?",
            "the persisted all oldest policy",
        )],
    );
    let imported = machine.run(
        &retained,
        &[
            "import",
            ".",
            "--context",
            "persisted-all-oldest",
            "--search",
            "defer",
            "--search-sessions",
            "all",
            "--search-order",
            "oldest",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    fs::create_dir_all(machine.data_root.join("cg")).expect("create changed-config directory");
    fs::write(
        machine.data_root.join("cg").join("fabric.toml"),
        "[search]\nsessions = 10\norder = \"newest\"\n",
    )
    .expect("change only the current invocation defaults");

    let status = machine.run(&retained, &["status", "--context", "persisted-all-oldest"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    let status_retry = str_at(&status, "/history/search/next");
    assert!(
        status_retry.contains("--search-sessions all --search-order oldest"),
        "status silently shrank the persisted policy: {status_retry}"
    );

    let found = machine.run(
        &retained,
        &[
            "find",
            "what does retry preserve?",
            "--context",
            "persisted-all-oldest",
        ],
    );
    assert_eq!(found.code, 1, "{}", found.describe());
    let found = found.json();
    assert_eq!(str_at(&found, "/state"), "history_search_not_ready");
    assert!(
        str_at(&found, "/retry").contains("--search-sessions all --search-order oldest"),
        "find silently shrank the persisted policy: {found}"
    );

    let newer = machine.project("persisted-newest-default");
    write_session(
        &machine.history_root(),
        &newer,
        "persisted-newest-session",
        &[("what differs?", "the configured newest bounded policy")],
    );
    let imported = machine.run(
        &newer,
        &[
            "import",
            ".",
            "--context",
            "persisted-newest-default",
            "--search",
            "defer",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let mixed = machine.run(
        &retained,
        &[
            "find",
            "what differs?",
            "--context",
            "persisted-all-oldest,persisted-newest-default",
        ],
    );
    assert_eq!(mixed.code, 1, "{}", mixed.describe());
    let mixed = mixed.json();
    assert!(
        mixed["retry"].is_null(),
        "mixed policies must not pick one: {mixed}"
    );
    let actions = mixed["retry_actions"]
        .as_array()
        .expect("per-Context actions");
    assert_eq!(actions.len(), 2, "{mixed}");
    assert!(actions.iter().any(|action| {
        action["retry"]
            .as_str()
            .is_some_and(|retry| retry.contains("--search-sessions all --search-order oldest"))
    }));
    assert!(actions.iter().any(|action| {
        action["retry"]
            .as_str()
            .is_some_and(|retry| retry.contains("--search-sessions 10 --search-order newest"))
    }));
}

#[test]
fn import_usage_errors_name_the_token_and_do_no_work() {
    let machine = Machine::without_model();
    let project = machine.project("usage");
    let session = write_session(
        &machine.history_root(),
        &project,
        "sess-usage",
        &[("usage question", "usage answer")],
    );
    let session = session.display().to_string();
    for (args, token) in [
        (vec!["import", ".", "--bogus"], "--bogus"),
        (vec!["import", ".", "--search", "later"], "later"),
        (vec!["import", ".", "another"], "another"),
        (vec!["import", ".", "--context"], "--context"),
        (vec!["import", ".", "--understand-now"], "--understand-now"),
        (
            vec!["import", ".", "--chunk-tokens", "10"],
            "--chunk-tokens",
        ),
        (vec!["import", &session, "--search", "defer"], "--context"),
        (vec!["import", &session, "--context", "a,b"], "a,b"),
        (vec!["import", &session, "--all-contexts"], "--all-contexts"),
        (vec!["import", ".", "--session", "sess-usage"], "--session"),
    ] {
        let output = machine.run(&project, &args);
        assert_eq!(output.code, 2, "{args:?}: {}", output.describe());
        assert!(
            output.stdout.contains(token) || output.stderr.contains(token),
            "the usage error for {args:?} names {token}: {}",
            output.describe()
        );
    }
    assert!(
        !machine.data_root.exists(),
        "a usage error performs no work"
    );
}

#[test]
fn cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file()
 {
    let machine = Machine::without_model();
    let project = machine.project("partial");
    let history = machine.history_root();
    let bad = write_session(&history, &project, "sess-a-bad", &[("first", "second")]);
    let mut bytes = fs::read(&bad).expect("read bad session");
    bytes.extend_from_slice(b"{\"type\": \"user\", not json\n");
    fs::write(&bad, bytes).expect("corrupt the session tail");
    write_session(
        &history,
        &project,
        "sess-b-good",
        &[("good question", "good answer")],
    );

    let output = machine.run(
        &project,
        &["import", ".", "--context", "partial", "--search", "defer"],
    );
    assert_eq!(output.code, 5, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "source_partial");
    assert_eq!(receipt.get("ok"), Some(&Value::Bool(false)));
    let failed = receipt
        .pointer("/capture/failed_sources")
        .and_then(Value::as_array)
        .expect("failed sources");
    assert_eq!(failed.len(), 1);
    assert_eq!(str_at(&failed[0], "/file"), "sess-a-bad.jsonl");
    assert_eq!(u64_at(&failed[0], "/position/line"), 3);
    assert!(!str_at(&failed[0], "/reason").is_empty());
    assert_eq!(str_at(&receipt, "/search/worker"), "not_started");
    assert_eq!(
        u64_at(&receipt, "/capture/events/inserted"),
        2,
        "every other file is captured; the failed file keeps only batches it committed"
    );
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);
}

#[test]
fn with_no_data_root_override_the_store_lives_at_dot_cg_under_home() {
    let machine = Machine::without_model();
    let project = machine.project("unrooted");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cg"))
        .args(["status"])
        .current_dir(&project)
        .env_remove("CG_DATA_ROOT")
        .env("HOME", &machine.home)
        .env("CG_MODEL_CACHE_ROOT", &machine.model_cache)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run the compiled cg binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    let value: Value = serde_json::from_str(&stdout).expect("one JSON object");
    assert_eq!(str_at(&value, "/state"), "store_not_found");
    assert_eq!(
        str_at(&value, "/store/db_path"),
        machine
            .home
            .join(".cg")
            .join("context-graph.db")
            .display()
            .to_string()
    );
    assert!(
        !machine.home.join(".cg").exists(),
        "a status read never creates the default data root"
    );
}

#[test]
fn find_in_a_context_with_no_imported_history_answers_no_matching_history() {
    let machine = Machine::without_model();
    let history = machine.history_root();
    let project = machine.project("pantry");
    write_session(
        &history,
        &project,
        "sess-pantry",
        &[(
            "Where do the spare jars go?",
            "Top shelf, behind the flour.",
        )],
    );
    let imported = machine.run(
        &project,
        &["import", ".", "--context", "pantry", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let forgotten = machine.run(&project, &["forget", "--context", "pantry", "--yes"]);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());

    let found = machine.run(&project, &["find", "spare jars"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    let found = found.json();
    assert_eq!(str_at(&found, "/state"), "no_matching_history");
    assert!(hits(&found).is_empty());
}

#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
#[test]
fn cli_status_reports_bounded_session_scope_from_the_persisted_worker_policy() {
    let machine = Machine::without_model();
    let project = machine.project("bounded-policy");
    use context_graph::Embedder;
    let embedder = std::sync::Arc::new(CliEventSearchEmbedder);
    fs::create_dir_all(&machine.data_root).expect("create isolated data root");
    let store_config = context_graph::StoreConfig {
        db_path: machine.db_path(),
        default_text_embedder: None,
        extra_embedding_spaces: vec![embedder.embedding_space()],
        ..context_graph::StoreConfig::default()
    };
    let store = context_graph::Store::open_with_embedder_registrations(
        store_config.clone(),
        vec![context_graph::EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder: embedder.clone(),
        }],
    )
    .expect("open isolated CLI status fixture");
    let context_id = store
        .create_context(context_graph::CreateContext {
            name: "bounded-policy".into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create bounded Context")
        .id;
    let declaration = context_graph::EventAdapterDeclaration {
        connector_id: "cli-bounded-policy".into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: serde_json::json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    };
    for index in 0..12 {
        store
            .append_event_batch(context_graph::AppendEventBatch::live(
                context_id,
                declaration.clone(),
                format!("bounded-source-{index:02}"),
                vec![context_graph::EventDraft {
                    session_ref: format!("bounded-session-{index:02}"),
                    source_event_key: format!("bounded-event-{index:02}"),
                    source_position: serde_json::json!({"ordinal": 1}),
                    occurred_at: None,
                    author: "fixture".into(),
                    kind: "message".into(),
                    render_role: context_graph::EventRenderRole::HumanMessage,
                    interaction_ref: None,
                    text: Some(format!("bounded answer {index:02}")),
                    payload: serde_json::json!({}),
                    anchors: serde_json::json!({}),
                    source_remainder: serde_json::json!({}),
                    coverage: context_graph::EventCoverage::Complete,
                }],
            ))
            .expect("append bounded session");
    }
    drop(store);
    context_graph::EventSearchProjectionRunner::catch_up_observed_with_policy(
        context_graph::EventSearchRunRequest {
            store: store_config,
            contexts: context_graph::ContextSelector::one(context_id),
            embedder: context_graph::EmbedderRegistration {
                space: embedder.embedding_space(),
                embedder,
            },
            profile: context_graph::EventSearchProfile::default(),
            embedding_workers: std::num::NonZeroUsize::MIN,
            max_events_per_batch: std::num::NonZeroU32::new(256).expect("positive batch"),
            max_passages_per_batch: std::num::NonZeroU32::new(64).expect("positive batch"),
            max_bytes_per_batch: std::num::NonZeroUsize::new(1 << 20).expect("positive byte bound"),
            lease_duration_ms: std::num::NonZeroU64::new(60_000).expect("positive lease"),
        },
        context_graph::EventSearchWorkPolicy::default(),
    )
    .result
    .expect("bounded lexical projection succeeds");

    let status = machine.run(&project, &["status", "--context", "bounded-policy"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        str_at(&status, "/history/search/status"),
        "ready",
        "{status}"
    );
    assert_eq!(u64_at(&status, "/history/search/covered_events"), 10);
    assert_eq!(u64_at(&status, "/history/search/pending_events"), 0);
    assert_eq!(u64_at(&status, "/history/search/outside_scope_events"), 2);
    assert_eq!(
        u64_at(&status, "/history/search/no_searchable_text_events"),
        0
    );
}

#[test]
fn cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either()
 {
    let machine = Machine::without_model();
    let project = machine.project("absent");
    let db_path = machine.db_path().display().to_string();
    for prepare in [false, true] {
        if prepare {
            fs::create_dir_all(&machine.data_root).expect("an existing data root with no database");
        }
        let before = fingerprint(&machine.root);
        let json = machine.run(&project, &["status"]);
        assert_eq!(json.code, 1, "{}", json.describe());
        assert_eq!(
            json.json(),
            serde_json::json!({
                "ok": false,
                "state": "store_not_found",
                "store": {"db_path": db_path},
                "retry": "create_store_with_write_command"
            })
        );
        let text = machine.run(&project, &["status", "--format", "text"]);
        assert_eq!(text.code, 1);
        assert_eq!(text.stdout, "");
        assert_eq!(
            text.stderr,
            format!(
                "error: Context Graph store not found: {db_path}\nnext: create it with the intended write command; for project history run `cg import <project-directory>`\n"
            )
        );
        for args in [
            vec!["find", "anything"],
            vec!["show", "0190a3b2-0000-7000-8000-000000000000"],
        ] {
            let read = machine.run(&project, &args);
            assert_eq!(read.code, 1, "{args:?}: {}", read.describe());
            let value = read.json();
            assert_eq!(str_at(&value, "/state"), "store_not_found");
            assert!(value.get("hits").is_none(), "no fabricated hits");
            assert!(str_at(&value, "/next").contains("cg import"));
        }
        assert_eq!(
            fingerprint(&machine.root),
            before,
            "status, find and show created nothing"
        );
    }
}

#[test]
fn missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight()
{
    let machine = Machine::without_model();
    let project = machine.project("offline");
    write_session(
        &machine.history_root(),
        &project,
        "sess-offline",
        &[("offline question", "offline answer")],
    );
    let background = machine.run(&project, &["import", ".", "--context", "offline"]);
    assert_eq!(background.code, 1, "{}", background.describe());
    let receipt = background.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(str_at(&receipt, "/search/status"), "blocked");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(u64_at(&receipt, "/search/pending_events"), 2);
    assert!(str_at(&receipt, "/retry").contains("cg model fetch --defaults"));
    assert!(str_at(&receipt, "/retry").contains("--search wait"));
    assert!(
        fs::read_dir(&machine.model_cache)
            .expect("model cache")
            .next()
            .is_none(),
        "no model was downloaded"
    );

    let wait = machine.run(&project, &["import", ".", "--search", "wait"]);
    assert_eq!(wait.code, 1, "{}", wait.describe());
    assert_eq!(str_at(&wait.json(), "/state"), "captured_search_behind");

    let defer = machine.run(&project, &["import", ".", "--search", "defer"]);
    assert_eq!(defer.code, 0, "{}", defer.describe());
    assert_eq!(str_at(&defer.json(), "/state"), "captured_search_deferred");

    let status = machine.run(&project, &["status"]).json();
    assert_eq!(str_at(&status, "/history/search/status"), "blocked");
    assert_eq!(u64_at(&status, "/history/search/pending_events"), 2);
    let text = machine.run(&project, &["status", "--format", "text"]);
    assert!(
        text.stdout
            .contains("blocked — local Nomic artifact is not present"),
        "{}",
        text.describe()
    );
    assert!(text.stdout.contains("cg model fetch --defaults"));
}

#[test]
fn cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status()
{
    let mut machine = Machine::without_model();
    // An installed-looking artifact whose weights cannot load: the preflight accepts it, the
    // projection fails when it loads the model.
    let real = event_support::existing_nomic_cache_root().join(event_support::NOMIC_MODEL_ID);
    let broken = machine
        .root
        .join("broken-cache")
        .join(event_support::NOMIC_MODEL_ID);
    fs::create_dir_all(&broken).expect("broken cache");
    for file in ["config.json", "tokenizer.json", "REVISION"] {
        fs::copy(real.join(file), broken.join(file)).expect("copy artifact metadata");
    }
    fs::write(broken.join("model.safetensors"), vec![7_u8; 2_000_000]).expect("broken weights");
    machine.model_cache = machine.root.join("broken-cache");
    let project = machine.project("broken");
    write_session(
        &machine.history_root(),
        &project,
        "sess-broken",
        &[("broken question", "broken answer")],
    );
    let output = machine.run(
        &project,
        &["import", ".", "--context", "broken", "--search", "wait"],
    );
    assert_eq!(output.code, 1, "{}", output.describe());
    let receipt = output.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(str_at(&receipt, "/search/status"), "failed");
    assert!(!str_at(&receipt, "/search/reason").is_empty());
    let status = machine.run(&project, &["status"]).json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_ne!(str_at(&status, "/history/search/status"), "ready");
}

#[test]
fn find_and_show_parse_their_arguments_before_reading() {
    let machine = Machine::without_model();
    let project = machine.project("args");
    for (args, token) in [
        (vec!["find"], "find"),
        (vec!["find", ""], "query"),
        (vec!["find", "q", "--limit", "0"], "--limit"),
        (
            vec!["find", "q", "--context", "a", "--all-contexts"],
            "--all-contexts",
        ),
        (vec!["find", "q", "--bogus"], "--bogus"),
        (vec!["show"], "show"),
        (vec!["show", "not-a-passage-id"], "not-a-passage-id"),
        (vec!["forget"], "--session"),
        (vec!["forget", "--session", "x", "--bogus"], "--bogus"),
    ] {
        let output = machine.run(&project, &args);
        assert_eq!(output.code, 2, "{args:?}: {}", output.describe());
        assert!(
            output.stdout.contains(token) || output.stderr.contains(token),
            "{args:?} names {token}: {}",
            output.describe()
        );
    }
}

#[test]
fn event_history_journey_with_real_nomic_imports_finds_shows_and_forgets() {
    let machine = Machine::with_nomic();
    let history = machine.history_root();
    let atlas = machine.project("atlas");
    let garden = machine.project("garden");
    write_session(
        &history,
        &atlas,
        "sess-transport",
        &[(
            "Why did we pick Iroh instead of NATS for the transport?",
            "Iroh keeps a direct peer-to-peer path; NATS would add a broker deployment we must operate.",
        )],
    );
    write_session(
        &history,
        &atlas,
        "sess-ledger",
        &[(
            "How should the ledger reconcile foreign currency entries?",
            "Convert at the booking-date rate and keep the original amount beside it.",
        )],
    );
    write_session(
        &history,
        &garden,
        "sess-garden",
        &[(
            "When should the tomato seedlings go outside?",
            "After the last frost, once nights stay above ten degrees.",
        )],
    );

    let imported = machine.run(&atlas, &["import", ".", "--context", "atlas"]);
    assert_eq!(imported.code, 0, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_scheduled");
    assert_eq!(str_at(&receipt, "/search/status"), "scheduled");
    assert_eq!(u64_at(&receipt, "/search/pending_events"), 4);
    machine.wait_for_search_ready(&atlas, "atlas");
    // The detached worker catches up every pending Context on the data root; wait for it to exit
    // after its final re-check before capturing a Context that must stay unsearchable.
    machine.wait_for_search_worker_exit();
    let garden_import = machine.run(
        &garden,
        &["import", ".", "--context", "garden", "--search", "defer"],
    );
    assert_eq!(garden_import.code, 0, "{}", garden_import.describe());

    // An unchanged, caught-up rerun only verifies the committed prefixes.
    let rerun = machine.run(&atlas, &["import", ".", "--search", "wait"]);
    assert_eq!(rerun.code, 0, "{}", rerun.describe());
    let rerun = rerun.json();
    assert_eq!(str_at(&rerun, "/state"), "complete");
    assert_eq!(u64_at(&rerun, "/capture/events/inserted"), 0);
    assert_eq!(str_at(&rerun, "/search/worker"), "not_started");
    assert_eq!(u64_at(&rerun, "/search/local_embedding_calls"), 0);
    let text_receipt = machine.run(&atlas, &["import", ".", "--format", "text"]);
    assert_eq!(text_receipt.code, 0, "{}", text_receipt.describe());
    for internal in ["generation", "vector", "embedding"] {
        assert!(
            !text_receipt.stdout.to_lowercase().contains(internal),
            "human import output hides {internal}: {}",
            text_receipt.stdout
        );
    }
    assert!(text_receipt.stdout.contains("atlas"));

    let found = machine.run(&atlas, &["find", "why iroh rather than nats"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    let found = found.json();
    assert_eq!(str_at(&found, "/selection/source"), "project_binding");
    let found_hits = hits(&found);
    assert!(
        !found_hits.is_empty(),
        "the transport discussion is found: {found}"
    );
    let top = &found_hits[0];
    assert!(str_at(top, "/text").contains("Iroh"), "{top}");
    assert_eq!(str_at(top, "/context_name"), "atlas");
    assert!(top.get("cosine_similarity").is_some());
    assert!(top.get("embedding_space_id").is_some());
    let passage_id = str_at(top, "/passage_id").to_owned();
    let linked = top
        .get("source_links")
        .and_then(Value::as_array)
        .expect("source links")
        .iter()
        .map(|link| str_at(link, "/event_id").to_owned())
        .collect::<Vec<_>>();
    assert!(!linked.is_empty());

    let text = machine.run(
        &atlas,
        &["find", "why iroh rather than nats", "--format", "text"],
    );
    assert_eq!(text.code, 0, "{}", text.describe());
    assert!(text.stdout.contains(&passage_id) && text.stdout.contains("atlas"));
    for internal in ["generation", "embedding", "vector", "cosine"] {
        assert!(
            !text.stdout.to_lowercase().contains(internal),
            "human find output hides {internal}: {}",
            text.stdout
        );
    }

    fs::remove_dir_all(&history).expect("remove the source history");
    let shown = machine.run(&atlas, &["show", &passage_id]);
    assert_eq!(shown.code, 0, "{}", shown.describe());
    let shown = shown.json();
    let events = shown
        .get("events")
        .and_then(Value::as_array)
        .expect("shown events");
    assert_eq!(
        events
            .iter()
            .map(|event| str_at(event, "/id").to_owned())
            .collect::<Vec<_>>(),
        linked,
        "show expands exactly the linked events"
    );
    assert!(events.iter().any(|event| {
        event
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("broker deployment"))
    }));
    for event in events {
        for field in [
            "seq",
            "author",
            "kind",
            "render_role",
            "payload",
            "anchors",
            "coverage",
            "source_remainder",
        ] {
            assert!(event.get(field).is_some(), "show exposes {field}: {event}");
        }
    }
    let shown_text = machine.run(&atlas, &["show", &passage_id, "--format", "text"]);
    assert_eq!(shown_text.code, 0);
    assert!(shown_text.stdout.contains("broker deployment"));

    let foreign = machine.run(&atlas, &["show", &passage_id, "--context", "garden"]);
    assert_eq!(foreign.code, 1, "{}", foreign.describe());
    assert_eq!(str_at(&foreign.json(), "/state"), "passage_not_found");
    assert!(
        !foreign.stdout.contains("Iroh") && !foreign.stdout.contains("atlas"),
        "a foreign handle leaks no text or Context: {}",
        foreign.stdout
    );

    let wider = machine.run(&garden, &["show", &passage_id, "--all-contexts"]);
    assert_eq!(
        wider.code,
        0,
        "an explicit wider selection shows it: {}",
        wider.describe()
    );

    let mixed = machine.run(&atlas, &["find", "iroh nats", "--context", "atlas,garden"]);
    assert_eq!(mixed.code, 0, "{}", mixed.describe());
    let mixed = mixed.json();
    assert_eq!(str_at(&mixed, "/selection/source"), "explicit");
    assert!(
        hits(&mixed)
            .iter()
            .all(|hit| str_at(hit, "/context_name") == "atlas")
    );
    let omitted = mixed
        .get("omitted_contexts")
        .and_then(Value::as_array)
        .expect("omitted contexts");
    assert!(
        omitted
            .iter()
            .any(|context| str_at(context, "/name") == "garden")
    );

    let not_ready = machine.run(&garden, &["find", "tomato"]);
    assert_eq!(not_ready.code, 1, "{}", not_ready.describe());
    assert_eq!(
        str_at(&not_ready.json(), "/state"),
        "history_search_not_ready"
    );
    let still_pending = machine.run(&garden, &["status"]).json();
    assert_eq!(
        str_at(&still_pending, "/history/search/status"),
        "pending",
        "find never starts projection"
    );

    let unknown = machine.run(&atlas, &["find", "iroh", "--context", "atlas,nowhere"]);
    assert_ne!(unknown.code, 0);
    assert!(hits(&unknown.json()).is_empty());
    assert_eq!(str_at(&unknown.json(), "/state"), "unknown_context");

    let all = machine.run(&machine.root, &["find", "iroh nats", "--all-contexts"]);
    assert_eq!(all.code, 0, "{}", all.describe());
    assert_eq!(str_at(&all.json(), "/selection/source"), "all_authorized");

    let multi_status = machine.run(&atlas, &["status", "--all-contexts"]);
    assert_eq!(multi_status.code, 0, "{}", multi_status.describe());
    let multi_status = multi_status.json();
    assert_eq!(
        multi_status
            .get("contexts")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(2),
        "one tagged status per Context"
    );

    let unconfirmed = machine.run(&atlas, &["forget", "--session", "sess-transport"]);
    assert_eq!(unconfirmed.code, 2, "{}", unconfirmed.describe());
    let unconfirmed = unconfirmed.json();
    assert_eq!(str_at(&unconfirmed, "/state"), "confirmation_required");
    assert!(str_at(&unconfirmed, "/retry").contains("--yes"));
    assert_eq!(
        u64_at(
            &machine.run(&atlas, &["status"]).json(),
            "/history/captured_events"
        ),
        4,
        "no confirmation, no change"
    );

    let forgotten = machine.run(&atlas, &["forget", "--session", "sess-transport", "--yes"]);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    let forgotten = forgotten.json();
    assert_eq!(str_at(&forgotten, "/state"), "forgotten");
    assert_eq!(u64_at(&forgotten, "/counts/events"), 2);
    assert!(u64_at(&forgotten, "/counts/passages") > 0);
    assert_eq!(
        str_at(&forgotten, "/search/status"),
        "rebuild_scheduled",
        "forget re-indexes the remaining history itself: {forgotten}"
    );
    assert_eq!(u64_at(&forgotten, "/search/pending_events"), 2);
    assert!(
        !forgotten.to_string().contains("cg import"),
        "the user is never told to re-import: {forgotten}"
    );

    // The remaining session becomes searchable again without another import.
    machine.wait_for_search_ready(&atlas, "atlas");
    let remaining = machine.run(&atlas, &["find", "ledger foreign currency"]);
    assert_eq!(remaining.code, 0, "{}", remaining.describe());
    assert!(
        hits(&remaining.json())
            .iter()
            .any(|hit| str_at(hit, "/text").contains("booking-date rate")),
        "the remaining session is searchable after the rebuild: {}",
        remaining.describe()
    );

    let after = machine.run(&atlas, &["find", "why iroh rather than nats"]);
    assert!(
        hits(&after.json())
            .iter()
            .all(|hit| !str_at(hit, "/text").contains("Iroh")),
        "find returns nothing from forgotten history: {}",
        after.describe()
    );
    let old_handle = machine.run(&atlas, &["show", &passage_id]);
    assert_eq!(old_handle.code, 1, "{}", old_handle.describe());
    let status = machine.run(&atlas, &["status"]).json();
    assert_eq!(u64_at(&status, "/history/captured_events"), 2);
    assert_eq!(u64_at(&status, "/history/sources/sessions"), 1);

    let nothing = machine.run(&atlas, &["forget", "--session", "sess-transport", "--yes"]);
    assert_eq!(nothing.code, 0, "{}", nothing.describe());
    assert_eq!(str_at(&nothing.json(), "/state"), "nothing_to_forget");

    let whole = machine.run(&atlas, &["forget", "--context", "atlas", "--yes"]);
    assert_eq!(whole.code, 0, "{}", whole.describe());
    assert_eq!(str_at(&whole.json(), "/state"), "forgotten");
    let emptied = machine.run(&atlas, &["status"]);
    assert_eq!(
        emptied.code,
        0,
        "the Context and its binding remain: {}",
        emptied.describe()
    );
    let emptied = emptied.json();
    assert_eq!(str_at(&emptied, "/context/name"), "atlas");
    assert_eq!(u64_at(&emptied, "/history/captured_events"), 0);
}

#[test]
fn import_dry_run_verifies_committed_prefixes_reports_what_would_be_captured_and_writes_nothing() {
    let machine = Machine::without_model();
    let project = machine.project("rehearsal");
    let history = machine.history_root();
    let unchanged = write_session(&history, &project, "sess-unchanged", &[("one", "two")]);
    let appended = write_session(&history, &project, "sess-appended", &[("three", "four")]);
    let rewritten = write_session(&history, &project, "sess-rewritten", &[("five", "six")]);

    // Before anything is stored, every selected file is new.
    let before = fingerprint(&machine.root);
    let fresh = machine.run(
        &project,
        &["import", ".", "--context", "rehearsal", "--dry-run"],
    );
    assert_eq!(fresh.code, 0, "{}", fresh.describe());
    let fresh = fresh.json();
    assert_eq!(str_at(&fresh, "/state"), "dry_run");
    assert_eq!(u64_at(&fresh, "/sources/files"), 3);
    assert_eq!(u64_at(&fresh, "/capture/new_sources"), 3);
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "a dry run writes nothing"
    );
    assert!(!machine.data_root.exists());

    let imported = machine.run(
        &project,
        &["import", ".", "--context", "rehearsal", "--search", "defer"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    // One file grows, one is rewritten, one is new, one stays as it was.
    let mut grown = fs::read(&appended).expect("read appended session");
    let extra = fs::read(write_session(
        &history,
        &project,
        "sess-scratch",
        &[("seven", "eight")],
    ))
    .expect("scratch records");
    fs::remove_file(
        history
            .join(event_support::claude_project_key(&project))
            .join("sess-scratch.jsonl"),
    )
    .expect("remove scratch session");
    grown.extend_from_slice(&extra);
    fs::write(&appended, &grown).expect("append records");
    // Change one committed byte inside the recorded text, keeping every record valid JSON so the
    // file is still this project's session.
    let mut changed = fs::read(&rewritten).expect("read rewritten session");
    let at = changed
        .windows(4)
        .position(|window| window == b"five")
        .expect("the rewritten session carries its text");
    changed[at + 3] = b'x';
    fs::write(&rewritten, &changed).expect("rewrite a committed byte");
    write_session(&history, &project, "sess-new", &[("nine", "ten")]);

    let before = fingerprint(&machine.root);
    let rehearsed = machine.run(&project, &["import", ".", "--dry-run"]);
    assert_eq!(rehearsed.code, 0, "{}", rehearsed.describe());
    let rehearsed = rehearsed.json();
    assert_eq!(str_at(&rehearsed, "/state"), "dry_run");
    assert_eq!(str_at(&rehearsed, "/selection/source"), "project_binding");
    assert_eq!(u64_at(&rehearsed, "/capture/unchanged_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/appended_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/diverged_sources"), 1);
    assert_eq!(u64_at(&rehearsed, "/capture/new_sources"), 1);
    assert_eq!(
        u64_at(&rehearsed, "/capture/bytes_to_capture"),
        (extra.len()
            + fs::metadata(
                history
                    .join(event_support::claude_project_key(&project))
                    .join("sess-new.jsonl")
            )
            .expect("new file")
            .len() as usize) as u64,
        "a real run would read the appended tail and the whole new file"
    );
    let sources = rehearsed
        .pointer("/capture/sources")
        .and_then(Value::as_array)
        .expect("per-source plan");
    let status_of = |file: &str| {
        sources
            .iter()
            .find(|source| str_at(source, "/file") == file)
            .map(|source| str_at(source, "/prefix").to_owned())
            .unwrap_or_else(|| panic!("{file} is planned: {rehearsed}"))
    };
    assert_eq!(status_of("sess-unchanged.jsonl"), "unchanged");
    assert_eq!(status_of("sess-appended.jsonl"), "appended");
    assert_eq!(status_of("sess-rewritten.jsonl"), "diverged");
    assert_eq!(status_of("sess-new.jsonl"), "new");
    assert_eq!(
        fingerprint(&machine.root),
        before,
        "a dry run writes nothing"
    );
    let _ = unchanged;
}

#[test]
fn cg_import_background_spawn_failure_leaves_capture_durable_and_returns_typed_behind() {
    let machine = Machine::with_nomic();
    let project = machine.project("unspawnable");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unspawnable",
        &[("unspawnable question", "unspawnable answer")],
    );
    // A private copy of the binary that is removed while the import waits at its Context
    // question, so the executable path it would launch its background worker from is gone.
    let binary = machine.root.join("bin").join("cg");
    fs::create_dir_all(binary.parent().expect("bin dir")).expect("bin dir");
    fs::copy(env!("CARGO_BIN_EXE_cg"), &binary).expect("copy the cg binary");
    let (imported, transcript) = machine.run_interactive_using(
        &binary,
        &project,
        &["import", "."],
        &[("Context name [unspawnable]: ", "\r")],
        &mut |_| fs::remove_file(&binary).expect("remove the worker executable"),
    );
    assert_eq!(
        imported.code,
        1,
        "{} transcript={transcript:?}",
        imported.describe()
    );
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(u64_at(&receipt, "/capture/events/inserted"), 2);
    assert_eq!(str_at(&receipt, "/search/worker"), "spawn_failed");
    assert!(!str_at(&receipt, "/search/reason").is_empty());
    assert!(str_at(&receipt, "/retry").contains("--search wait"));

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_eq!(str_at(&status, "/history/search/status"), "pending");
}

/// A read must never make the background projection give up.
///
/// `cg status` is read-only and reads through ContextDB's read route, so it is never refused while
/// the worker holds the Store; the converse must hold too — readers arriving throughout the build
/// must not cost the worker the work it was started for. Three readers poll here at the journey's
/// own cadence, which is faster than any human would, for as long as the build takes.
#[test]
fn search_reaches_ready_while_three_readers_poll_the_store_throughout_the_build() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let machine = Machine::with_nomic();
    let history = machine.history_root();
    let atlas = machine.project("atlas");
    write_session(
        &history,
        &atlas,
        "sess-transport",
        &[(
            "Why did we pick Iroh instead of NATS for the transport?",
            "Iroh keeps a direct peer-to-peer path; NATS would add a broker deployment we must operate.",
        )],
    );
    write_session(
        &history,
        &atlas,
        "sess-ledger",
        &[(
            "How should the ledger reconcile foreign currency entries?",
            "Convert at the booking-date rate and keep the original amount beside it.",
        )],
    );

    let imported = machine.run(&atlas, &["import", ".", "--context", "atlas"]);
    assert_eq!(imported.code, 0, "{}", imported.describe());
    assert_eq!(str_at(&imported.json(), "/search/status"), "scheduled");

    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let machine = &machine;
            let atlas = &atlas;
            let stop = &stop;
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let status = machine.run(atlas, &["status", "--context", "atlas"]);
                    assert_eq!(
                        status.code,
                        0,
                        "a read is never refused while the projection runs: {}",
                        status.describe()
                    );
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            });
        }
        let ready = machine.wait_for_search_ready(&atlas, "atlas");
        stop.store(true, Ordering::Relaxed);
        assert_eq!(u64_at(&ready, "/history/search/pending_events"), 0);
        assert_eq!(u64_at(&ready, "/history/search/covered_events"), 4);
    });
}

/// A worker lock this user cannot open is a failure to report, not a worker already running.
///
/// `already_running` means another worker holds the lock and will do the work, so the import is
/// honest to say the search was scheduled. Every other reason the lock cannot be taken means
/// nothing was scheduled at all, and reporting `captured_search_scheduled` there would leave the
/// user waiting for a worker that was never started.
#[test]
fn cg_import_reports_search_behind_when_the_worker_lock_cannot_be_opened() {
    use std::os::unix::fs::PermissionsExt;

    let machine = Machine::with_nomic();
    let project = machine.project("unlockable");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unlockable",
        &[("unlockable question", "unlockable answer")],
    );

    // Capture first with search deferred: the data root and its store exist, search has pending
    // work, and no worker has run.
    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "unlockable",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());
    assert_eq!(
        str_at(&deferred.json(), "/state"),
        "captured_search_deferred"
    );

    // The lock file is present but unopenable by this user: not a lock somebody holds, an
    // unusable one.
    let lock_path = machine.data_root.join("event-search-worker.lock");
    fs::write(&lock_path, b"").expect("create the worker lock file");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o000))
        .expect("make the worker lock file unopenable");

    let imported = machine.run(&project, &["import", "."]);
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))
        .expect("restore the worker lock file");
    assert_eq!(imported.code, 1, "{}", imported.describe());
    let receipt = imported.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert_eq!(str_at(&receipt, "/search/worker"), "spawn_failed");
    assert!(
        !str_at(&receipt, "/search/reason").is_empty(),
        "the receipt discloses why nothing was started: {receipt}"
    );
    assert!(str_at(&receipt, "/retry").contains("--search wait"));

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let status = status.json();
    assert_eq!(
        u64_at(&status, "/history/captured_events"),
        2,
        "capture stays durable"
    );
    assert_eq!(str_at(&status, "/history/search/status"), "pending");
}

/// `--search wait` reports an unusable worker lock instead of waiting for a turn that never comes.
///
/// Waiting is right for a lock another worker holds: that worker finishes and lets go. A lock that
/// cannot be opened at all is not a turn that is coming, so a foreground import that waits for it
/// hangs forever rather than telling the user their search did not run.
#[test]
fn cg_import_search_wait_reports_behind_when_the_worker_lock_cannot_be_opened() {
    use std::os::unix::fs::PermissionsExt;

    let machine = Machine::with_nomic();
    let project = machine.project("unlockable-wait");
    write_session(
        &machine.history_root(),
        &project,
        "sess-unlockable-wait",
        &[("unlockable wait question", "unlockable wait answer")],
    );

    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "unlockable-wait",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());

    let lock_path = machine.data_root.join("event-search-worker.lock");
    fs::write(&lock_path, b"").expect("create the worker lock file");
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o000))
        .expect("make the worker lock file unopenable");

    let waited = machine.run(&project, &["import", ".", "--search", "wait"]);
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))
        .expect("restore the worker lock file");
    assert_eq!(waited.code, 1, "{}", waited.describe());
    let receipt = waited.json();
    assert_eq!(str_at(&receipt, "/state"), "captured_search_behind");
    assert_eq!(str_at(&receipt, "/capture/status"), "complete");
    assert!(
        !str_at(&receipt, "/search/reason").is_empty(),
        "the receipt discloses why the turn could not be taken: {receipt}"
    );

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        2,
        "capture stays durable"
    );
}

// A store whose last writer died before its committed image settled refuses every read-only open.
// Pure import configuration must still refuse before that store is recovered; once configuration
// is valid, `cg import` takes the writable recovery open and read-only status names that command.
#[test]
fn import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command() {
    use contextdb_engine::persistence::read_persistence_test_scaffold::{
        DurableStoreDamage, prepare_durable_store_damage_for_test,
    };

    let machine = Machine::without_model();
    let project = machine.project("kaval");
    write_session(
        &machine.history_root(),
        &project,
        "sess-kaval",
        &[("why esp32", "optical out was dropped")],
    );
    let first = machine.run(
        &project,
        &["import", ".", "--context", "kaval", "--search", "defer"],
    );
    assert_eq!(first.code, 0, "{}", first.describe());

    prepare_durable_store_damage_for_test(
        &machine.db_path(),
        DurableStoreDamage::NonMonotonicCommitIndex,
    )
    .expect("leave the store needing a writable open, as a killed writer does");

    write_placement_override(&machine, "somewhere_else");
    let damaged_store = fs::read(machine.db_path()).expect("fingerprint damaged store bytes");
    for args in [
        &["import", ".", "--search", "defer"][..],
        &["import", ".", "--dry-run"][..],
    ] {
        let refused = machine.run(&project, args);
        assert_eq!(refused.code, 1, "{}", refused.describe());
        assert_eq!(str_at(&refused.json(), "/state"), "config_error");
        assert_eq!(
            fs::read(machine.db_path()).expect("read store after placement refusal"),
            damaged_store,
            "invalid placement must not recover the interrupted store"
        );
    }

    write_passage_override(&machine, 256, 10_000);
    for args in [
        &["import", ".", "--search", "defer"][..],
        &["import", ".", "--dry-run"][..],
    ] {
        let refused = machine.run(&project, args);
        assert_eq!(refused.code, 1, "{}", refused.describe());
        assert_eq!(str_at(&refused.json(), "/state"), "config_error");
        assert_eq!(
            fs::read(machine.db_path()).expect("read store after profile refusal"),
            damaged_store,
            "invalid passage profile must not recover the interrupted store"
        );
    }

    write_passage_override(&machine, 256, 384);

    let status = machine.run(&project, &["status"]);
    assert_eq!(
        status.code, 1,
        "the import-only recovery exit does not change status"
    );
    assert_eq!(
        str_at(&status.json(), "/state"),
        "store_needs_recovery",
        "{}",
        status.describe()
    );
    assert!(
        str_at(&status.json(), "/message").contains("cg import <project-directory> --search defer"),
        "a read-only refusal names the command that recovers the store: {}",
        status.describe()
    );

    let recovered = machine.run(&project, &["import", ".", "--search", "defer"]);
    assert_eq!(
        recovered.code,
        0,
        "import recovers the store: {}",
        recovered.describe()
    );
    assert_eq!(str_at(&recovered.json(), "/context/name"), "kaval");

    let after = machine.run(&project, &["status"]);
    assert_eq!(
        after.code,
        0,
        "the recovered store reads again: {}",
        after.describe()
    );
}

// A dry-run is a read-only planning operation even when a normal import would recover a store
// before planning. Both Context-selection forms must surface the typed recovery refusal and leave
// the interrupted store bytes untouched.
#[test]
fn import_dry_run_refuses_a_recovery_needing_store_without_writing() {
    use contextdb_engine::persistence::read_persistence_test_scaffold::{
        DurableStoreDamage, prepare_durable_store_damage_for_test,
    };

    let machine = Machine::without_model();
    let project = machine.project("dry-run-recovery");
    write_session(
        &machine.history_root(),
        &project,
        "sess-dry-run-recovery",
        &[("why dry run", "it must leave the store unchanged")],
    );
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "dry-run-recovery",
            "--search",
            "defer",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    prepare_durable_store_damage_for_test(
        &machine.db_path(),
        DurableStoreDamage::NonMonotonicCommitIndex,
    )
    .expect("leave the store needing a writable open, as a killed writer does");
    let before = fingerprint(&machine.data_root);

    for args in [
        &["import", ".", "--dry-run"][..],
        &["import", ".", "--context", "dry-run-recovery", "--dry-run"][..],
    ] {
        let refused = machine.run(&project, args);
        assert_eq!(refused.code, 6, "{}", refused.describe());
        assert_eq!(
            str_at(&refused.json(), "/state"),
            "store_needs_recovery",
            "{}",
            refused.describe()
        );
        assert_eq!(
            fingerprint(&machine.data_root),
            before,
            "dry-run must not settle or otherwise change the recovery-needing store: {}",
            refused.describe()
        );
    }
}

fn write_passage_override(machine: &Machine, target: i64, hard: i64) {
    let config = machine.data_root.join("cg");
    fs::create_dir_all(&config).expect("cg config directory");
    fs::write(
        config.join("fabric.toml"),
        format!("[search.passages]\ntarget_tokens = {target}\nhard_tokens = {hard}\n"),
    )
    .expect("write the passage override");
}

fn write_placement_override(machine: &Machine, placement: &str) {
    let config = machine.data_root.join("cg");
    fs::create_dir_all(&config).expect("cg config directory");
    fs::write(
        config.join("fabric.toml"),
        format!("[fabric]\nevent_placement = \"{placement}\"\n"),
    )
    .expect("write the placement override");
}

#[test]
fn dry_run_refuses_invalid_placement_before_new_or_existing_store_writes_or_model_access() {
    let machine = Machine::without_model();
    let project = machine.project("invalid-placement-preview");
    write_session(
        &machine.history_root(),
        &project,
        "sess-invalid-placement",
        &[("preview the import", "without changing the machine")],
    );

    write_placement_override(&machine, "somewhere_else");
    let before_new_root = fingerprint(&machine.root);
    let before_new_model_cache = fingerprint(&machine.model_cache);
    let preview = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "invalid-placement-preview",
            "--dry-run",
        ],
    );
    let real = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "invalid-placement-preview",
            "--search",
            "defer",
        ],
    );
    for refused in [&preview, &real] {
        assert_eq!(refused.code, 1, "{}", refused.describe());
        assert_eq!(str_at(&refused.json(), "/state"), "config_error");
    }
    assert_eq!(
        str_at(&preview.json(), "/message"),
        str_at(&real.json(), "/message"),
        "preview and real import must report the same placement refusal"
    );
    assert_eq!(fingerprint(&machine.root), before_new_root);
    assert_eq!(fingerprint(&machine.model_cache), before_new_model_cache);
    assert!(!machine.db_path().exists());

    write_placement_override(&machine, "local_only");
    let seeded = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "invalid-placement-preview",
            "--search",
            "defer",
        ],
    );
    assert_eq!(seeded.code, 0, "{}", seeded.describe());
    write_placement_override(&machine, "somewhere_else");
    let before_existing_root = fingerprint(&machine.data_root);
    let before_existing_model_cache = fingerprint(&machine.model_cache);
    let preview = machine.run(&project, &["import", ".", "--dry-run"]);
    let real = machine.run(&project, &["import", ".", "--search", "defer"]);
    for refused in [&preview, &real] {
        assert_eq!(refused.code, 1, "{}", refused.describe());
        assert_eq!(str_at(&refused.json(), "/state"), "config_error");
    }
    assert_eq!(
        str_at(&preview.json(), "/message"),
        str_at(&real.json(), "/message")
    );
    assert_eq!(fingerprint(&machine.data_root), before_existing_root);
    assert_eq!(
        fingerprint(&machine.model_cache),
        before_existing_model_cache
    );
}

// A passage override whose hard limit is below its target cannot describe a passage, so import
// refuses it as a configuration error before it writes a store, a Context or an event.
#[test]
fn an_invalid_passage_override_is_refused_before_any_write() {
    let machine = Machine::without_model();
    let project = machine.project("kaval");
    write_session(
        &machine.history_root(),
        &project,
        "sess-kaval",
        &[("why esp32", "optical out was dropped")],
    );
    write_passage_override(&machine, 300, 200);

    let refused = machine.run(
        &project,
        &["import", ".", "--context", "kaval", "--search", "defer"],
    );
    assert_eq!(refused.code, 1, "{}", refused.describe());
    assert_eq!(str_at(&refused.json(), "/state"), "config_error");
    assert!(
        str_at(&refused.json(), "/message").contains("hard_tokens"),
        "the refusal names the setting: {}",
        refused.describe()
    );
    assert!(
        !machine.db_path().exists(),
        "no store was written for a refused override"
    );
}

#[test]
fn a_hard_passage_ceiling_refuses_each_import_mode_before_new_or_existing_store_writes() {
    let machine = Machine::without_model();
    let project = machine.project("passage-ceiling");
    write_session(
        &machine.history_root(),
        &project,
        "sess-ceiling",
        &[(
            "capture must remain untouched",
            "when hard passage ceiling exceeds Nomic",
        )],
    );
    write_passage_override(&machine, 256, 10_000);
    let before_new_store = fingerprint(&machine.root);
    let before_new_model_cache = fingerprint(&machine.model_cache);
    for search in ["background", "wait", "defer"] {
        let refused = machine.run(
            &project,
            &[
                "import",
                ".",
                "--context",
                "passage-ceiling",
                "--search",
                search,
            ],
        );
        assert_eq!(refused.code, 1, "{search}: {}", refused.describe());
        let receipt = refused.json();
        assert_eq!(str_at(&receipt, "/state"), "config_error");
        assert!(
            str_at(&receipt, "/message").contains("10000")
                && str_at(&receipt, "/message").contains("8192"),
            "{search}: {}",
            refused.describe()
        );
        assert_eq!(
            fingerprint(&machine.root),
            before_new_store,
            "{search} must refuse before creating any store or capture"
        );
        assert_eq!(fingerprint(&machine.model_cache), before_new_model_cache);
    }
    let preview = machine.run(
        &project,
        &["import", ".", "--context", "passage-ceiling", "--dry-run"],
    );
    assert_eq!(preview.code, 1, "{}", preview.describe());
    assert_eq!(str_at(&preview.json(), "/state"), "config_error");
    assert_eq!(fingerprint(&machine.root), before_new_store);
    assert_eq!(fingerprint(&machine.model_cache), before_new_model_cache);

    // A small declared profile remains accepted: deferred capture is asset-free and does not
    // invent a global 32/40 minimum beyond the profile's existing relations.
    write_passage_override(&machine, 32, 40);
    let cache_before_defer = fingerprint(&machine.model_cache);
    let deferred = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "passage-ceiling",
            "--search",
            "defer",
        ],
    );
    assert_eq!(deferred.code, 0, "{}", deferred.describe());
    assert_eq!(
        str_at(&deferred.json(), "/state"),
        "captured_search_deferred"
    );
    assert_eq!(
        fingerprint(&machine.model_cache),
        cache_before_defer,
        "valid deferred capture must not inspect or create model assets"
    );

    write_passage_override(&machine, 256, 10_000);
    let before_existing_store = fingerprint(&machine.data_root);
    let before_existing_model_cache = fingerprint(&machine.model_cache);
    for search in ["background", "wait", "defer"] {
        let refused = machine.run(&project, &["import", ".", "--search", search]);
        assert_eq!(refused.code, 1, "{search}: {}", refused.describe());
        assert_eq!(str_at(&refused.json(), "/state"), "config_error");
        assert_eq!(
            fingerprint(&machine.data_root),
            before_existing_store,
            "{search} must not alter an existing store when admission refuses"
        );
        assert_eq!(
            fingerprint(&machine.model_cache),
            before_existing_model_cache
        );
    }
    let preview = machine.run(&project, &["import", ".", "--dry-run"]);
    assert_eq!(preview.code, 1, "{}", preview.describe());
    assert_eq!(str_at(&preview.json(), "/state"), "config_error");
    assert_eq!(fingerprint(&machine.data_root), before_existing_store);
    assert_eq!(
        fingerprint(&machine.model_cache),
        before_existing_model_cache
    );
}

#[test]
fn a_hard_limit_equal_to_the_oversized_overlap_is_refused_before_any_write() {
    let machine = Machine::without_model();
    let project = machine.project("overlap");
    write_session(
        &machine.history_root(),
        &project,
        "sess-overlap",
        &[("why esp32", "optical out was dropped")],
    );
    write_passage_override(&machine, 16, 32);

    let refused = machine.run(
        &project,
        &["import", ".", "--context", "overlap", "--search", "defer"],
    );
    assert_eq!(refused.code, 1, "{}", refused.describe());
    assert_eq!(str_at(&refused.json(), "/state"), "config_error");
    let refusal = refused.json();
    let message = str_at(&refusal, "/message");
    assert!(
        message.contains("hard_tokens") && message.contains("overlap"),
        "the refusal names the forward-progress boundary: {}",
        refused.describe()
    );
    assert!(
        !machine.db_path().exists(),
        "no store was written for a profile that cannot make oversized-event progress"
    );
}

#[test]
fn forget_reports_a_configured_profile_refusal_after_erasing_without_starting_a_worker() {
    let machine = Machine::with_nomic();
    let project = machine.project("forget-invalid-profile");
    write_session(
        &machine.history_root(),
        &project,
        "sess-erased",
        &[("why esp32", "optical out was dropped")],
    );
    write_session(
        &machine.history_root(),
        &project,
        "sess-remaining",
        &[("why stm32", "the timer needs a capture channel")],
    );
    let imported = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "forget-invalid-profile",
            "--search",
            "wait",
        ],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());
    write_passage_override(&machine, 16, 32);

    let forgotten = machine.run(&project, &["forget", "--session", "sess-erased", "--yes"]);
    assert_eq!(forgotten.code, 0, "{}", forgotten.describe());
    let forgotten = forgotten.json();
    assert_eq!(str_at(&forgotten, "/state"), "forgotten");
    assert_eq!(u64_at(&forgotten, "/counts/events"), 2);
    assert_eq!(
        str_at(&forgotten, "/search/status"),
        "rebuild_not_started",
        "the receipt does not promise a worker that configuration refused"
    );
    assert!(
        str_at(&forgotten, "/search/reason").contains("overlap"),
        "the receipt carries the config refusal"
    );
    assert!(
        forgotten.pointer("/search/worker").is_none(),
        "a configuration refusal happens before the worker kick"
    );

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    assert_eq!(
        u64_at(&status.json(), "/history/captured_events"),
        2,
        "the successful erasure remains committed despite the rebuild refusal"
    );
}

// The passage profile is declared policy: an override in the cg configuration file is what the
// search generation is built with, and status reports the profile the active generation used.
#[test]
fn a_passage_override_is_recorded_on_the_generation_and_reported_by_status() {
    let machine = Machine::with_nomic();
    let project = machine.project("kaval");
    write_session(
        &machine.history_root(),
        &project,
        "sess-kaval",
        &[("why esp32", "optical out was dropped")],
    );
    write_passage_override(&machine, 128, 200);

    let imported = machine.run(
        &project,
        &["import", ".", "--context", "kaval", "--search", "wait"],
    );
    assert_eq!(imported.code, 0, "{}", imported.describe());

    let status = machine.run(&project, &["status"]);
    assert_eq!(status.code, 0, "{}", status.describe());
    let profile = status
        .json()
        .pointer("/history/search/renderer/profile")
        .cloned()
        .unwrap_or(Value::Null);
    assert_eq!(
        profile.get("target_passage_tokens"),
        Some(&Value::from(128)),
        "status reports the overridden target: {}",
        status.describe()
    );
    assert_eq!(profile.get("hard_passage_tokens"), Some(&Value::from(200)));
}

// A profile is part of a generation's identity. Changing it after the default generation is ready
// makes the next wait import rebuild that same captured history and then switch the active head.
// The core projection proof covers the old head continuing to serve during that replacement.
#[test]
fn a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import() {
    let machine = Machine::with_nomic();
    let project = machine.project("changed-passage-profile");
    write_session(
        &machine.history_root(),
        &project,
        "sess-profile",
        &[(
            "where is the release record",
            "the release record is in the changelog",
        )],
    );
    let first = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "changed-passage-profile",
            "--search",
            "wait",
        ],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let before = machine.run(&project, &["status"]);
    assert_eq!(before.code, 0, "{}", before.describe());
    let old_head = str_at(&before.json(), "/history/search/active_generation_id").to_owned();
    assert_eq!(
        before
            .json()
            .pointer("/history/search/renderer/profile/target_passage_tokens"),
        Some(&Value::from(256))
    );

    write_passage_override(&machine, 128, 200);
    let rebuilt = machine.run(&project, &["import", ".", "--search", "wait"]);
    assert_eq!(rebuilt.code, 0, "{}", rebuilt.describe());
    assert_eq!(str_at(&rebuilt.json(), "/state"), "complete");
    assert!(
        u64_at(&rebuilt.json(), "/search/local_embedding_calls") > 0,
        "the changed profile rebuilt captured history: {}",
        rebuilt.describe()
    );

    let after = machine.run(&project, &["status"]);
    assert_eq!(after.code, 0, "{}", after.describe());
    assert_eq!(
        after
            .json()
            .pointer("/history/search/renderer/profile/target_passage_tokens"),
        Some(&Value::from(128))
    );
    assert_ne!(
        str_at(&after.json(), "/history/search/active_generation_id"),
        old_head,
        "the replacement generation becomes the active head"
    );
    let found = machine.run(&project, &["find", "release record"]);
    assert_eq!(found.code, 0, "{}", found.describe());
    assert!(
        !hits(&found.json()).is_empty(),
        "the replacement head answers search"
    );
}

#[test]
fn a_background_import_schedules_a_ready_head_rebuild_for_the_changed_profile() {
    let machine = Machine::with_nomic();
    let project = machine.project("background-changed-passage-profile");
    write_session(
        &machine.history_root(),
        &project,
        "sess-background-profile",
        &[(
            "which file owns the cache",
            "the catalog owns the cache manifest",
        )],
    );
    let first = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "background-changed-passage-profile",
            "--search",
            "wait",
        ],
    );
    assert_eq!(first.code, 0, "{}", first.describe());

    let before = machine.run(&project, &["status"]);
    assert_eq!(before.code, 0, "{}", before.describe());
    let old_head = str_at(&before.json(), "/history/search/active_generation_id").to_owned();
    write_passage_override(&machine, 128, 200);
    let scheduled = machine.run(&project, &["import", "."]);
    assert_eq!(scheduled.code, 0, "{}", scheduled.describe());
    assert_eq!(
        str_at(&scheduled.json(), "/state"),
        "captured_search_scheduled"
    );
    // An unlocked lock can precede the detached child's first acquisition. Only the requested
    // ready profile AND a different head establish completion; status never rescues the worker.
    let mut last_status = None;
    let replacement = wait_for_state(Duration::from_secs(120), Duration::from_millis(100), || {
        let after = machine.run(&project, &["status"]);
        let replacement = if after.code == 0 {
            let value = after.json();
            if str_at(&value, "/history/search/status") == "ready"
                && value.pointer("/history/search/renderer/profile/target_passage_tokens")
                    == Some(&Value::from(128))
                && str_at(&value, "/history/search/active_generation_id") != old_head
            {
                Some(())
            } else {
                None
            }
        } else {
            None
        };
        last_status = Some(after);
        Ok::<_, ()>(replacement)
    })
    .expect("replacement observation cannot fail");
    assert!(
        replacement.is_some(),
        "replacement never activated: {}",
        last_status
            .expect("at least one status observation")
            .describe()
    );
}

// The detached worker keeps its lock after catch_up returns, but it no longer owns the Store. This
// forces a second default import into that interval, then proves the same worker's next pass uses
// the newly valid profile rather than recapturing history or requiring a second worker.
#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn a_default_import_with_a_changed_profile_reuses_the_live_workers_next_pass() {
    let machine = Machine::with_nomic();
    let project = machine.project("live-worker-profile-handoff");
    write_session(
        &machine.history_root(),
        &project,
        "sess-live-worker-profile",
        &[(
            "where is the release record",
            "the release record is in the changelog",
        )],
    );
    let captured = machine.run(
        &project,
        &[
            "import",
            ".",
            "--context",
            "live-worker-profile-handoff",
            "--search",
            "defer",
        ],
    );
    assert_eq!(captured.code, 0, "{}", captured.describe());

    let barrier = tempfile::tempdir().expect("handoff barrier directory");
    let mut worker = spawn_handoff_worker(&machine, &project, barrier.path());
    wait_for_handoff_barrier(barrier.path());
    assert!(
        cg_node_fs::acquire_worker_lock(&machine.data_root.join("event-search-worker.lock"))
            .is_err(),
        "the live worker must still hold its lock at the post-catch-up barrier"
    );

    let old_ready = machine.run(&project, &["status"]);
    assert_eq!(
        old_ready.code,
        0,
        "the barrier is after Store access, not merely after lock acquisition: {}",
        old_ready.describe()
    );
    let old_ready = old_ready.json();
    assert_eq!(str_at(&old_ready, "/history/search/status"), "ready");
    assert_eq!(
        old_ready.pointer("/history/search/renderer/profile/target_passage_tokens"),
        Some(&Value::from(256))
    );
    let old_head = str_at(&old_ready, "/history/search/active_generation_id").to_owned();

    write_passage_override(&machine, 128, 200);
    let scheduled = machine.run(&project, &["import", "."]);
    assert_eq!(scheduled.code, 0, "{}", scheduled.describe());
    let scheduled = scheduled.json();
    assert_eq!(str_at(&scheduled, "/state"), "captured_search_scheduled");
    assert_eq!(str_at(&scheduled, "/search/status"), "scheduled");
    assert_eq!(str_at(&scheduled, "/search/worker"), "already_running");
    assert_eq!(
        u64_at(&scheduled, "/capture/events/inserted"),
        0,
        "the second default import must not recapture already stored history"
    );
    assert_eq!(u64_at(&scheduled, "/capture/local_embedding_calls"), 0);

    worker.release_and_join();
    let new_ready = machine.run(&project, &["status"]);
    assert_eq!(new_ready.code, 0, "{}", new_ready.describe());
    let new_ready = new_ready.json();
    assert_eq!(str_at(&new_ready, "/history/search/status"), "ready");
    assert_eq!(
        new_ready.pointer("/history/search/renderer/profile/target_passage_tokens"),
        Some(&Value::from(128)),
        "the live worker's next pass reads the valid replacement profile"
    );
    assert_ne!(
        str_at(&new_ready, "/history/search/active_generation_id"),
        old_head,
        "the replacement profile activates a new search head"
    );
}

// A killed worker leaves its lease durable. The next default worker must keep serving another
// Context, then recheck the leased one until it can resume the captured work without another
// import.
#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn a_default_worker_waits_for_an_orphaned_lease_and_does_not_starve_other_contexts() {
    let machine = Machine::with_nomic();
    let orphaned_project = machine.project("orphaned-lease");
    let eligible_project = machine.project("eligible-while-orphaned");
    write_session(
        &machine.history_root(),
        &orphaned_project,
        "orphaned-first",
        &[("where is the release record", "the changelog owns it")],
    );
    let first = machine.run(
        &orphaned_project,
        &[
            "import",
            ".",
            "--context",
            "orphaned-lease",
            "--search",
            "wait",
        ],
    );
    assert_eq!(first.code, 0, "{}", first.describe());
    let orphaned_context_id = str_at(&first.json(), "/context/id").to_owned();
    let first_ready = machine.run(&orphaned_project, &["status"]);
    assert_eq!(first_ready.code, 0, "{}", first_ready.describe());
    let generation_id =
        str_at(&first_ready.json(), "/history/search/active_generation_id").to_owned();

    write_session(
        &machine.history_root(),
        &orphaned_project,
        "orphaned-second",
        &[(
            "what resumed after interruption",
            "the prior search work resumed",
        )],
    );
    let captured_orphaned = machine.run(
        &orphaned_project,
        &[
            "import",
            ".",
            "--context",
            "orphaned-lease",
            "--search",
            "defer",
        ],
    );
    assert_eq!(
        captured_orphaned.code,
        0,
        "{}",
        captured_orphaned.describe()
    );
    assert_eq!(
        u64_at(&captured_orphaned.json(), "/capture/events/inserted"),
        2
    );

    write_session(
        &machine.history_root(),
        &eligible_project,
        "eligible-first",
        &[(
            "which Context can still run",
            "the eligible Context can run",
        )],
    );
    let captured_eligible = machine.run(
        &eligible_project,
        &[
            "import",
            ".",
            "--context",
            "eligible-while-orphaned",
            "--search",
            "defer",
        ],
    );
    assert_eq!(
        captured_eligible.code,
        0,
        "{}",
        captured_eligible.describe()
    );

    let barrier = tempfile::tempdir().expect("pre-assessment barrier directory");
    let mut worker =
        spawn_worker_before_first_assessment(&machine, &orphaned_project, barrier.path());
    wait_for_worker_marker(&barrier.path().join("pre-catch-up-entered"));
    seed_unexpired_event_search_lease(
        &machine,
        &orphaned_context_id,
        &generation_id,
        Duration::from_secs(5),
    );
    fs::write(barrier.path().join("pre-catch-up-release"), b"continue\n")
        .expect("release worker into durable lease");

    // The handoff barrier runs after this service pass. It holds the worker before its next
    // assessment, making service order observable without requiring the eligible model build to
    // beat the orphan lease's five-second TTL.
    wait_for_handoff_barrier(barrier.path());
    let eligible_ready = machine.run(&eligible_project, &["status"]);
    assert_eq!(eligible_ready.code, 0, "{}", eligible_ready.describe());
    assert_eq!(
        eligible_ready.json().pointer("/history/search/status"),
        Some(&Value::from("ready")),
        "the eligible Context was not served in the worker's first pass"
    );
    let orphaned_before_next_assessment = machine.run(&orphaned_project, &["status"]);
    assert_eq!(
        orphaned_before_next_assessment.code,
        0,
        "{}",
        orphaned_before_next_assessment.describe()
    );
    assert_ne!(
        orphaned_before_next_assessment
            .json()
            .pointer("/history/search/status"),
        Some(&Value::from("ready")),
        "the leased Context was served before the eligible Context"
    );
    fs::write(
        barrier.path().join("release"),
        b"continue after ordered service\n",
    )
    .expect("release worker after observing ordered service");

    let resumed = machine.wait_for_search_ready(&orphaned_project, "orphaned-lease");
    assert_eq!(u64_at(&resumed, "/history/search/pending_events"), 0);
    assert_eq!(u64_at(&resumed, "/history/search/covered_events"), 4);
    machine.wait_for_search_worker_exit();
    let status = worker
        .wait()
        .expect("join default worker after lease expiry");
    assert!(status.success(), "default worker exited {status}");
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn an_adopted_valid_profile_survives_invalid_config_and_late_work_beyond_three_passes() {
    // Separate machines prove both the total-pass and per-Context-success cases.
    for same_context in [false, true] {
        let machine = Machine::with_nomic();
        let first_project = machine.project("late-pass-1");
        write_session(
            &machine.history_root(),
            &first_project,
            "late-session-1",
            &[("first captured question", "first captured answer")],
        );
        let capture = machine.run(
            &first_project,
            &[
                "import",
                ".",
                "--context",
                "late-pass-1",
                "--search",
                "defer",
            ],
        );
        assert_eq!(capture.code, 0, "{}", capture.describe());
        let barrier = tempfile::tempdir().expect("numbered worker barrier");
        fs::write(barrier.path().join("numbered-passes"), b"5\n").unwrap();
        let mut worker = spawn_handoff_worker(&machine, &first_project, barrier.path());
        wait_for_worker_marker(&barrier.path().join("entered.1"));
        let launch = machine.run(&first_project, &["status"]);
        assert_eq!(launch.code, 0, "{}", launch.describe());
        let launch_head = str_at(&launch.json(), "/history/search/active_generation_id").to_owned();

        write_passage_override(&machine, 128, 200);
        worker.release_pass(1);
        wait_for_worker_marker(&barrier.path().join("entered.2"));
        let adopted = machine.run(&first_project, &["status"]);
        assert_eq!(adopted.code, 0, "{}", adopted.describe());
        let adopted = adopted.json();
        assert_eq!(str_at(&adopted, "/history/search/status"), "ready");
        assert_eq!(
            adopted.pointer("/history/search/renderer/profile/target_passage_tokens"),
            Some(&json!(128))
        );
        assert_ne!(
            str_at(&adopted, "/history/search/active_generation_id"),
            launch_head
        );

        let mut projects = vec![first_project.clone()];
        // Each capture occurs after the preceding pass has completed. Invalid configuration is
        // installed AFTER capture (imports validate config) but BEFORE releasing the worker.
        for capture_number in 2..=4 {
            assert!(
                cg_node_fs::acquire_worker_lock(
                    &machine.data_root.join("event-search-worker.lock")
                )
                .is_err()
            );
            let name = format!("late-pass-{capture_number}");
            let project = if same_context {
                first_project.clone()
            } else {
                machine.project(&name)
            };
            write_session(
                &machine.history_root(),
                &project,
                &format!("late-session-{capture_number}"),
                &[("late captured question", "late captured answer")],
            );
            write_passage_override(&machine, 128, 200);
            let context = if same_context { "late-pass-1" } else { &name };
            let capture = machine.run(&project, &["import", ".", "--context", context]);
            assert_eq!(capture.code, 0, "{}", capture.describe());
            let capture = capture.json();
            assert_eq!(str_at(&capture, "/search/worker"), "already_running");
            assert_eq!(u64_at(&capture, "/capture/events/inserted"), 2);
            assert_eq!(u64_at(&capture, "/capture/local_embedding_calls"), 0);
            write_passage_override(&machine, 300, 200);
            worker.release_pass(capture_number);
            wait_for_worker_marker(
                &barrier
                    .path()
                    .join(format!("entered.{}", capture_number + 1)),
            );
            let ready = machine.run(&project, &["status"]);
            assert_eq!(ready.code, 0, "{}", ready.describe());
            let ready = ready.json();
            assert_eq!(str_at(&ready, "/history/search/status"), "ready");
            assert_eq!(
                ready.pointer("/history/search/renderer/profile/target_passage_tokens"),
                Some(&json!(128))
            );
            assert_eq!(
                u64_at(&ready, "/history/search/covered_events"),
                if same_context {
                    2 * capture_number as u64
                } else {
                    2
                }
            );
            assert_eq!(u64_at(&ready, "/history/search/pending_events"), 0);
            projects.push(project);
        }
        // The fourth capture was made after pass four; no import follows that final capture.
        worker.release_and_join();
        for project in projects {
            let ready = machine.run(&project, &["status"]);
            assert_eq!(ready.code, 0, "{}", ready.describe());
            assert_eq!(str_at(&ready.json(), "/history/search/status"), "ready");
        }
    }
}
