//! Independent acceptance proofs for the configured consumer-read allowances.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, Store,
    StoreConfig, owner_control::context_graph_read_session_options,
};
use contextdb_core::{Value, read_contract::ReadRoute};
use contextdb_engine::read_session::ReadSession;

const ROWS_DDL: &str =
    "CREATE TABLE proof_rows (id TEXT PRIMARY KEY, body TEXT) SYNC CONFLICT KEEP LATEST";
const ADEQUATE_BYTES: usize = 64 * 1024 * 1024;
const CUSTOM_ACTIVE_MS: u64 = 12_345;

fn budget(bytes: usize) -> NonZeroUsize {
    NonZeroUsize::new(bytes).expect("proof budgets are positive")
}

fn declaration() -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "proof".into(),
            tables: vec![ConsumerTable {
                name: "proof_rows".into(),
                ddl: ROWS_DDL.into(),
            }],
        },
        scope_labels: Default::default(),
    }
}

fn config(path: PathBuf, memory_limit_bytes: NonZeroUsize) -> StoreConfig {
    StoreConfig {
        db_path: path,
        read_memory_limit_bytes: memory_limit_bytes,
        read_active_time_ms: NonZeroU64::new(CUSTOM_ACTIVE_MS).expect("positive active allowance"),
        ..StoreConfig::default()
    }
}

fn seed(path: &Path, memory_limit_bytes: NonZeroUsize) -> Store {
    let store = Store::open_with_consumer_schema(
        config(path.to_path_buf(), memory_limit_bytes),
        Vec::new(),
        declaration().schema,
    )
    .expect("create the public consumer table");
    for ordinal in 0..=600 {
        store
            .sync_database()
            .execute(
                "INSERT INTO proof_rows (id, body) VALUES ($id, $body)",
                &HashMap::from([
                    ("id".into(), Value::Text(format!("row-{ordinal:04}"))),
                    // Each row is large enough that the 1-byte policy has no plausible successful
                    // read path, while the adequate policy remains a small deterministic fixture.
                    ("body".into(), Value::Text("x".repeat(8 * 1024))),
                ]),
            )
            .expect("seed one declared row");
    }
    store
}

fn every_id(reader: &ConsumerReader) -> Vec<String> {
    let mut cursor = reader
        .open_cursor("SELECT id FROM proof_rows ORDER BY id", &HashMap::new())
        .expect("open the declared table as a cursor");
    let (mut ids, mut has_more) = {
        let page = cursor
            .first_page()
            .expect("the opened cursor has its first page");
        (
            page.rows
                .iter()
                .map(|row| match &row[0] {
                    Value::Text(id) => id.clone(),
                    other => panic!("id column is text, got {other:?}"),
                })
                .collect::<Vec<_>>(),
            page.has_more,
        )
    };
    while has_more {
        let page = cursor.fetch(None).expect("fetch the next bounded page");
        has_more = page.has_more;
        ids.extend(page.rows.iter().map(|row| match &row[0] {
            Value::Text(id) => id.clone(),
            other => panic!("id column is text, got {other:?}"),
        }));
    }
    cursor.close().expect("release the cursor snapshot");
    ids
}

fn assert_memory_limit_refusal(error: CgError, route: &str) {
    let CgError::Engine(message) = error else {
        panic!(
            "the {route} one-byte read must preserve ContextDB's mapped memory-limit refusal, got {error:?}"
        );
    };
    assert!(
        message.starts_with("the read went past the memory it is allowed: ")
            && message.ends_with(" bytes"),
        "the {route} one-byte read must be ContextDB's memory-limit failure class, not an arbitrary engine error: {message}"
    );
}

fn low_budget_refusal(path: &Path, route: &str) {
    match ConsumerReader::open_with_memory_limit(path, declaration(), budget(1)) {
        Err(error) => assert_memory_limit_refusal(error, route),
        Ok(reader) => {
            let refusal = reader.execute(
                "SELECT id, body FROM proof_rows ORDER BY id",
                &HashMap::new(),
            );
            reader
                .close()
                .expect("release the one-byte session after its refusal");
            assert_memory_limit_refusal(
                refusal.expect_err("one byte cannot materialize the declared rows"),
                route,
            );
        }
    }
}

#[test]
fn default_store_config_declares_the_one_gib_read_memory_ceiling() {
    assert_eq!(StoreConfig::default().read_active_time_ms.get(), 5_000);
    assert_eq!(context_graph_read_session_options().limits.active_ms, 5_000);
    let custom = context_graph::owner_control::context_graph_read_session_options_with_budgets(
        budget(ADEQUATE_BYTES),
        NonZeroU64::new(12_345).expect("positive custom time allowance"),
    );
    assert_eq!(custom.limits.active_ms, 12_345);
    assert_eq!(custom.memory_limit, Some(ADEQUATE_BYTES));
    assert_eq!(custom.limits.cursor_idle_ms, 300_000);
    assert_eq!(custom.limits.cursor_lifetime_ms, 1_800_000);
    let mut saved = serde_json::to_value(StoreConfig::default()).expect("serialize config");
    saved
        .as_object_mut()
        .expect("config object")
        .remove("read_active_time_ms");
    let restored: StoreConfig = serde_json::from_value(saved).expect("time omission defaults");
    assert_eq!(restored.read_active_time_ms.get(), 5_000);
    let mut invalid = serde_json::to_value(restored).expect("serialize restored config");
    invalid["read_active_time_ms"] = serde_json::json!(0);
    assert!(serde_json::from_value::<StoreConfig>(invalid).is_err());
    assert_eq!(
        context_graph_read_session_options().memory_limit,
        Some(1_073_741_824),
        "the shared file and owner session options default to the one-GiB consumer policy"
    );
    assert_eq!(
        StoreConfig::default().read_memory_limit_bytes,
        budget(1_073_741_824),
        "omitting an application override is still an explicit one-GiB consumer policy"
    );
}

#[test]
fn file_and_live_owner_reads_honor_per_instance_memory_budgets_without_losing_a_601_row_result() {
    let scratch = tempfile::tempdir().expect("isolated store directory");
    let path = scratch.path().join("read-policy.db");
    let adequate = budget(ADEQUATE_BYTES);

    // The held Store owns its live route. Its adequate configured ReadLimits must serve all 601
    // rows, including the second cursor page, through the owner rather than a direct file open.
    let holder = seed(&path, adequate);
    let options = context_graph::owner_control::context_graph_read_session_options_with_budgets(
        adequate,
        NonZeroU64::new(CUSTOM_ACTIVE_MS).expect("positive active allowance"),
    );
    let report = ReadSession::owner_report_in_runtime_dir(
        &path,
        options.clone(),
        context_graph::owner_control::owner_read_runtime_dir().as_deref(),
    )
    .expect("ask the actual live owner for its configured ceilings");
    assert_eq!(
        report
            .serving
            .expect("the held store serves live owner reads")
            .effective_limits
            .active_ms
            .value,
        CUSTOM_ACTIVE_MS,
        "the live owner publishes the configured active-time ceiling, not the default"
    );
    let owner_reader = ConsumerReader::open_with_read_limits(
        &path,
        declaration(),
        adequate,
        NonZeroU64::new(CUSTOM_ACTIVE_MS).expect("positive active allowance"),
    )
    .expect("the active owner serves a reader with the configured adequate budget");
    let owner_session = ReadSession::open_in_runtime_dir(
        &path,
        options.clone(),
        context_graph::owner_control::owner_read_runtime_dir(),
    )
    .expect("open an actual owner-routed session with the configured allowances");
    let owner_cursor = owner_session
        .open_cursor("SELECT id FROM proof_rows ORDER BY id", &HashMap::new())
        .expect("open a real cursor through the live owner");
    assert_eq!(owner_cursor.route(), ReadRoute::Owner);
    owner_cursor
        .close()
        .expect("release the owner route witness");
    drop(owner_session);
    let expected: Vec<String> = (0..=600)
        .map(|ordinal| format!("row-{ordinal:04}"))
        .collect();
    assert_eq!(
        every_id(&owner_reader),
        expected,
        "the owner route preserves every row"
    );
    owner_reader
        .close()
        .expect("release the owner-routed session");

    low_budget_refusal(&path, "live owner route");
    let owner_neighbor = ConsumerReader::open_with_memory_limit(&path, declaration(), adequate)
        .expect("the one-byte owner refusal cannot alter another reader's adequate budget");
    assert_eq!(every_id(&owner_neighbor), expected);
    owner_neighbor
        .close()
        .expect("release the adequate owner neighbor");

    drop(holder);

    let file_session = ReadSession::open_in_runtime_dir(
        &path,
        options,
        context_graph::owner_control::owner_read_runtime_dir(),
    )
    .expect("open a committed-file session under the configured allowances");
    let file_cursor = file_session
        .open_cursor("SELECT id FROM proof_rows ORDER BY id", &HashMap::new())
        .expect("the configured file session executes a real read");
    assert_eq!(file_cursor.route(), ReadRoute::File);
    assert_eq!(file_session.options().limits.active_ms, CUSTOM_ACTIVE_MS);
    file_cursor.close().expect("release the file route witness");
    drop(file_session);

    // With no holder, the same store is loaded from its committed file. This independent one-byte
    // session must report the same mapped memory-limit refusal, rather than any unrelated error.
    low_budget_refusal(&path, "committed file route");

    let file_reader = ConsumerReader::open_with_read_limits(
        &path,
        declaration(),
        adequate,
        NonZeroU64::new(CUSTOM_ACTIVE_MS).expect("positive active allowance"),
    )
    .expect("a separate adequate reader opens the committed file after the low-budget refusal");
    assert_eq!(
        every_id(&file_reader),
        expected,
        "the file route remains complete after refusal"
    );
    file_reader.close().expect("release the file session");
}
