//! The receipt fold requires the declared ordinal index to preserve order.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::NonZeroUsize;

use context_graph::{CreateContext, Store, StoreConfig};
use contextdb_core::{TxId, Value, read_contract::ReadLimits};
use contextdb_engine::read_session::{ReadSession, ReadSessionOptions};
use sha2::{Digest as _, Sha256};

const RECEIPT_ORDER_PROBE_SQL: &str = "SELECT passage_ordinal, content_sha256 FROM event_search_passages WHERE generation_id = $generation_id AND is_current = $current ORDER BY generation_id ASC, passage_ordinal ASC";
const RECEIPT_INDEX: &str = "event_search_passages_generation_ordinal_idx";

fn params(generation_id: uuid::Uuid) -> HashMap<String, Value> {
    HashMap::from([
        ("generation_id".into(), Value::Uuid(generation_id)),
        ("current".into(), Value::Bool(true)),
    ])
}

fn receipt_hash(rows: impl IntoIterator<Item = (u64, String)>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"context-graph/event-search-generation/v1");
    for (ordinal, content_sha256) in rows {
        hash_part(&mut digest, &ordinal.to_be_bytes());
        hash_part(&mut digest, content_sha256.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn hash_part(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}

fn small_limits() -> ReadLimits {
    ReadLimits {
        result_rows: 16,
        result_bytes: 16 * 1024,
        work: 1_000,
        active_ms: 5_000,
        memory: 1024 * 1024,
        cursor_page_rows: 3,
        cursor_page_bytes: 4 * 1024,
        cursor_idle_ms: 30_000,
        cursor_lifetime_ms: 60_000,
    }
}

fn insert_generation(store: &Store, context_id: uuid::Uuid, id: uuid::Uuid) {
    store
        .execute_sql_for_test(
            "INSERT INTO event_search_generations (id, context_id, interpretation_snapshot, interpretation_snapshot_sha256, event_count, ordered_source_set_sha256, source_content_sha256, capture_ordinal, renderer_id, renderer_version, renderer_profile, embedding_space_id, encoding_profile, state, eligible_event_count, rendered_event_count, embedded_event_count, excluded_event_count, failed_event_count, passage_count, started_at) VALUES ($id, $context_id, $snapshot, $snapshot_sha256, $zero, $ordered_sha256, $source_sha256, $zero, $renderer_id, $renderer_version, $profile, $space, $encoding, $state, $zero, $zero, $zero, $zero, $zero, $zero, $started_at)",
            &HashMap::from([
                ("id".into(), Value::Uuid(id)),
                ("context_id".into(), Value::Uuid(context_id)),
                ("snapshot".into(), Value::Json(serde_json::json!({}))),
                ("snapshot_sha256".into(), Value::Text("snapshot".into())),
                ("ordered_sha256".into(), Value::Text("ordered".into())),
                ("source_sha256".into(), Value::Text("source".into())),
                ("zero".into(), Value::Int64(0)),
                ("renderer_id".into(), Value::Text("receipt-order".into())),
                ("renderer_version".into(), Value::Text("v1".into())),
                ("profile".into(), Value::Json(serde_json::json!({}))),
                ("space".into(), Value::Text("text_default_nomic_v1_5_768".into())),
                ("encoding".into(), Value::Text("receipt-order-v1".into())),
                ("state".into(), Value::Text("ready".into())),
                ("started_at".into(), Value::TxId(TxId(1))),
            ]),
        )
        .expect("seed derived generation through the declared test SQL seam");
}

fn insert_passage(
    store: &Store,
    context_id: uuid::Uuid,
    generation_id: uuid::Uuid,
    ordinal: u64,
    content_sha256: String,
    is_current: bool,
) {
    store
        .execute_sql_for_test(
            "INSERT INTO event_search_passages (id, context_id, generation_id, connector_id, session_ref, source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, rendered_text, content_sha256, vector_text, dimension, is_current, created_at) VALUES ($id, $context_id, $generation_id, $connector_id, $session_ref, $source_instance_ref, $ordinal, $ordinal, $ordinal, $text, $content_sha256, $vector_text, $dimension, $is_current, $created_at)",
            &HashMap::from([
                ("id".into(), Value::Uuid(uuid::Uuid::now_v7())),
                ("context_id".into(), Value::Uuid(context_id)),
                ("generation_id".into(), Value::Uuid(generation_id)),
                ("connector_id".into(), Value::Text("receipt-order".into())),
                ("session_ref".into(), Value::Text("receipt-order-session".into())),
                ("source_instance_ref".into(), Value::Text("receipt-order-source".into())),
                ("ordinal".into(), Value::Int64(i64::try_from(ordinal).expect("ordinal fits"))),
                ("text".into(), Value::Text(format!("passage {ordinal}"))),
                ("content_sha256".into(), Value::Text(content_sha256)),
                ("vector_text".into(), Value::Null),
                ("dimension".into(), Value::Int64(768)),
                ("is_current".into(), Value::Bool(is_current)),
                ("created_at".into(), Value::TxId(TxId(1))),
            ]),
        )
        .expect("seed derived passage through the declared test SQL seam");
}

fn receipt_rows(columns: &[String], rows: &[Vec<Value>]) -> Vec<(u64, String)> {
    let ordinal = columns
        .iter()
        .position(|column| column == "passage_ordinal")
        .expect("receipt ordinal column");
    let content_sha256 = columns
        .iter()
        .position(|column| column == "content_sha256")
        .expect("receipt hash column");
    rows.iter()
        .map(|row| {
            let ordinal = match &row[ordinal] {
                Value::Int64(value) => u64::try_from(*value).expect("non-negative ordinal"),
                other => panic!("receipt ordinal is integer, got {other:?}"),
            };
            let content_sha256 = match &row[content_sha256] {
                Value::Text(value) => value.clone(),
                other => panic!("receipt hash is text, got {other:?}"),
            };
            (ordinal, content_sha256)
        })
        .collect()
}

#[test]
fn receipt_fold_keeps_ordinal_hash_and_count_across_small_cursor_pages() {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("receipt-order.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("open Store");
    let context_id = store
        .create_context(CreateContext {
            name: "receipt order".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
        .as_uuid();
    let generation_id = uuid::Uuid::now_v7();
    let other_generation_id = uuid::Uuid::now_v7();
    insert_generation(&store, context_id, generation_id);
    insert_generation(&store, context_id, other_generation_id);

    let expected = (0..9)
        .map(|ordinal| (ordinal, format!("active-{ordinal:02}")))
        .collect::<Vec<_>>();
    for (ordinal, content_sha256) in &expected {
        insert_passage(
            &store,
            context_id,
            generation_id,
            *ordinal,
            content_sha256.clone(),
            true,
        );
    }
    for ordinal in [21, 22] {
        insert_passage(
            &store,
            context_id,
            generation_id,
            ordinal,
            format!("retired-{ordinal}"),
            false,
        );
    }
    for ordinal in 0..6 {
        insert_passage(
            &store,
            context_id,
            other_generation_id,
            ordinal,
            format!("other-{ordinal}"),
            true,
        );
    }

    let session = ReadSession::open_with_options(
        store.db_path(),
        ReadSessionOptions {
            limits: small_limits(),
            ..ReadSessionOptions::default()
        },
    )
    .expect("open bounded engine read session");
    let planned = session
        .execute(RECEIPT_ORDER_PROBE_SQL, &params(generation_id))
        .expect("execute equality-prefixed receipt order probe");
    let mut cursor = session
        .open_cursor(RECEIPT_ORDER_PROBE_SQL, &params(generation_id))
        .expect("open equality-prefixed receipt order probe cursor");
    let mut rows = receipt_rows(&cursor.first_page().columns, &cursor.first_page().rows);
    let mut more = cursor.first_page().has_more;
    while more {
        let page = cursor
            .fetch(NonZeroUsize::new(3))
            .expect("fetch bounded receipt cursor page");
        more = page.has_more;
        rows.extend(receipt_rows(&page.columns, &page.rows));
    }
    cursor.close().expect("close receipt cursor");

    assert_eq!(
        planned.trace.physical_plan, "IndexScan",
        "{:#?}",
        planned.trace
    );
    assert_eq!(
        planned.trace.index_used.as_deref(),
        Some(RECEIPT_INDEX),
        "{:#?}",
        planned.trace
    );
    assert!(planned.trace.sort_elided, "{:#?}", planned.trace);
    assert_eq!(planned.rows.len(), expected.len());
    assert_eq!(rows, expected);
    assert_eq!(rows.len(), expected.len());
    assert_eq!(receipt_hash(rows), receipt_hash(expected));
}
