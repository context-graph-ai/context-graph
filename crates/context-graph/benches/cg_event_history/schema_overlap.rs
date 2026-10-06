//! Complete read-only event-history answers during startup replay and dependency-wide migration.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use context_graph::test_support::{
    record_event_history_read_cost_for_test, take_event_history_read_cost_for_test,
};
use context_graph::{ConsumerReader, ContextSelector, EmbedderRegistration, EventSearchFilter};
use contextdb_core::{Lsn, Value};
use contextdb_engine::Database;
use contextdb_engine::sync_types::{ChangeSet, ConflictPolicies, ConflictPolicy};
use serde_json::{Value as Json, json};

const NAME: &str = "event_find_show_status_during_schema_replay_and_largest_migration";
const KINDS: [&str; 3] = ["find", "show", "status"];

#[derive(Default)]
struct Log {
    idle_ns: Vec<u64>,
    overlapped_ns: Vec<u64>,
    refusals: u64,
    wrong_answers: u64,
    charged_bytes: u64,
    retries: u64,
    writable_open_attempts: u64,
}

fn percentile(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1)]
}

fn query(
    kind: &str,
    reader: &ConsumerReader,
    fixture: &super::fixture::Fixture,
) -> context_graph::Result<Json> {
    let selector = ContextSelector::new(fixture.contexts.iter().copied())?;
    match kind {
        "find" => {
            let answer = reader.find_event_history(
                selector,
                super::QUESTIONS[0],
                EventSearchFilter::default(),
            )?;
            assert!(
                !answer.hits.is_empty(),
                "find must return complete passage hits"
            );
            assert!(
                answer.hits.iter().all(|hit| !hit.sources.is_empty()),
                "every hit carries exact source links"
            );
            Ok(serde_json::to_value(answer).expect("find answer"))
        }
        "show" => {
            let (context_id, passage_id) = fixture.sample_passages[0];
            let passage = reader
                .get_event_search_passage(ContextSelector::one(context_id), passage_id)?
                .expect("fixture passage exists");
            let events = passage
                .sources
                .iter()
                .map(|source| {
                    reader
                        .graph()
                        .get_event(source.context_id, source.event_id)
                        .map(|event| event.expect("linked event exists"))
                })
                .collect::<context_graph::Result<Vec<_>>>()?;
            Ok(json!({"passage": passage, "events": events}))
        }
        "status" => {
            let answer = reader.event_history_status(selector)?;
            assert_eq!(answer.contexts.len(), fixture.contexts.len());
            for context in &answer.contexts {
                assert_eq!(
                    context
                        .source_streams
                        .iter()
                        .map(|stream| stream.event_count)
                        .sum::<u64>(),
                    context.captured_events,
                    "status reports exact captured stream counts"
                );
                assert_eq!(
                    context.search.pending_events, 0,
                    "the sealed fixture is fully projected"
                );
            }
            Ok(serde_json::to_value(answer).expect("status answer"))
        }
        _ => unreachable!(),
    }
}

fn open_reader(
    fixture: &super::fixture::Fixture,
    registration: &EmbedderRegistration,
) -> ConsumerReader {
    let mut reader = ConsumerReader::open(
        &fixture.db_path,
        context_graph::event_history_reader_declaration(),
    )
    .expect("read-only event-history reader");
    reader
        .attach_event_search_embedder_registrations(vec![registration.clone()])
        .expect("share the real Nomic model");
    reader
}

/// Every round uses a fresh copy of the same fixture. Short migrations accumulate genuine
/// overlaps across fresh receivers; the active interval ends as soon as apply_changes returns.
fn overlap(
    fixture: &super::fixture::Fixture,
    registration: &EmbedderRegistration,
    db: &Database,
    changes: ChangeSet,
) -> [Log; 3] {
    let active = AtomicBool::new(false);
    let epoch = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let counters: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    std::thread::scope(|scope| {
        let handles = KINDS
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let (active, epoch, stop, counters) = (&active, &epoch, &stop, &counters);
                scope.spawn(move || {
                    record_event_history_read_cost_for_test();
                    let reader = open_reader(fixture, registration);
                    let baseline = query(kind, &reader, fixture).expect("idle baseline succeeds");
                    let opening = take_event_history_read_cost_for_test();
                    let mut log = Log {
                        retries: opening.read_retries,
                        writable_open_attempts: opening.writable_open_attempts,
                        charged_bytes: opening.charged_bytes,
                        ..Log::default()
                    };
                    while !stop.load(Ordering::Acquire) {
                        let started_active = active.load(Ordering::Acquire);
                        let started_epoch = epoch.load(Ordering::Acquire);
                        record_event_history_read_cost_for_test();
                        let begun = Instant::now();
                        let result = query(KINDS[index], &reader, fixture);
                        let elapsed = begun.elapsed().as_nanos() as u64;
                        let during_work =
                            started_active || epoch.load(Ordering::Acquire) != started_epoch;
                        let cost = take_event_history_read_cost_for_test();
                        log.retries += cost.read_retries;
                        log.writable_open_attempts += cost.writable_open_attempts;
                        log.charged_bytes = log
                            .charged_bytes
                            .max(cost.charged_bytes)
                            .max(reader.charged_bytes().expect("actual reader charges"));
                        match result {
                            Err(_) => log.refusals += 1,
                            Ok(answer) if answer != baseline => log.wrong_answers += 1,
                            Ok(_) => {}
                        }
                        if during_work {
                            log.overlapped_ns.push(elapsed);
                        } else {
                            log.idle_ns.push(elapsed);
                        }
                        counters[index].fetch_add(1, Ordering::Release);
                    }
                    log
                })
            })
            .collect::<Vec<_>>();
        let deadline = Instant::now();
        while counters
            .iter()
            .any(|count| count.load(Ordering::Acquire) < 20)
        {
            assert!(
                deadline.elapsed().as_secs() < 600,
                "reader warmup must finish"
            );
            std::thread::yield_now();
        }
        epoch.fetch_add(1, Ordering::AcqRel);
        active.store(true, Ordering::Release);
        let applied = db.apply_changes(
            changes,
            &ConflictPolicies::uniform(ConflictPolicy::LatestWins),
        );
        active.store(false, Ordering::Release);
        epoch.fetch_add(1, Ordering::AcqRel);
        stop.store(true, Ordering::Release);
        applied.expect("real dependency-complete schema apply succeeds");
        handles
            .into_iter()
            .map(|handle| handle.join().expect("reader thread"))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    })
}

fn vector(ordinal: usize) -> Value {
    let mut values = vec![0.0; 768];
    values[0] = 1.0;
    values[1] = ordinal as f32;
    Value::Vector(values)
}

/// The unchanged largest affected-set fixture: 100 existing FK-linked tables, vectors on every
/// table, a final-table including-sync trigger, and a single root ADD COLUMN.
fn prepare_chain(db: &Database) -> Arc<AtomicU64> {
    let calls = Arc::new(AtomicU64::new(0));
    let empty = HashMap::new();
    for ordinal in 0..100_usize {
        let columns = if ordinal == 0 {
            "id INTEGER PRIMARY KEY, body TEXT".into()
        } else {
            format!(
                "id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL REFERENCES schema_{:03}(id), body TEXT, FOREIGN KEY (parent_id) REFERENCES schema_{:03}(id)",
                ordinal - 1,
                ordinal - 1
            )
        };
        db.execute(
            &format!("CREATE TABLE schema_{ordinal:03} ({columns}) SYNC OFF"),
            &empty,
        )
        .expect("FK chain table");
    }
    db.execute(
        "CREATE EVENT TYPE schema_099_insert WHEN INSERT ON schema_099",
        &empty,
    )
    .unwrap();
    db.execute(
        "CREATE TRIGGER schema_099_insert_trigger ON schema_099 WHEN INSERT INCLUDING SYNC",
        &empty,
    )
    .unwrap();
    db.register_trigger_callback("schema_099_insert_trigger", {
        let calls = calls.clone();
        move |_db, _context| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    })
    .unwrap();
    for ordinal in 0..100_usize {
        db.execute(
            &format!("ALTER TABLE schema_{ordinal:03} ADD COLUMN embedding VECTOR(768)"),
            &empty,
        )
        .unwrap();
    }
    let tx = db.begin().unwrap();
    for ordinal in 0..100_usize {
        let params = HashMap::from([("v".into(), vector(ordinal))]);
        let sql = if ordinal == 0 {
            "INSERT INTO schema_000 (id, body, embedding) VALUES (1, 'kept', $v)".into()
        } else {
            format!(
                "INSERT INTO schema_{ordinal:03} (id, parent_id, body, embedding) VALUES (1, 1, 'kept', $v)"
            )
        };
        db.execute_in_tx(tx, &sql, &params).unwrap();
    }
    db.commit(tx).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    calls
}

fn assert_chain_kept(db: &Database, calls: &AtomicU64) {
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "schema migration must not replay the committed insert's trigger"
    );
    for ordinal in 0..100 {
        assert_eq!(
            db.execute(
                &format!("SELECT id, body, embedding FROM schema_{ordinal:03} ORDER BY id"),
                &HashMap::new()
            )
            .unwrap()
            .rows,
            vec![vec![
                Value::Int64(1),
                Value::Text("kept".into()),
                vector(ordinal)
            ]]
        );
        assert_eq!(db.execute(&format!("SELECT id FROM schema_{ordinal:03} ORDER BY embedding <=> $q USE VECTOR EXACT LIMIT 1"), &HashMap::from([("q".into(), vector(ordinal))])).unwrap().rows,
            vec![vec![Value::Int64(1)]]);
    }
    assert!(
        db.execute(
            "INSERT INTO schema_099 (id, parent_id, body) VALUES (3, 999, 'invalid')",
            &HashMap::new()
        )
        .is_err(),
        "last dependent still enforces its FK"
    );
    assert!(
        db.list_triggers()
            .iter()
            .any(|trigger| trigger.name == "schema_099_insert_trigger" && trigger.including_sync)
    );
    let before = calls.load(Ordering::Relaxed);
    db.execute(
        "INSERT INTO schema_099 (id, parent_id, body) VALUES (2, 1, 'after-replay')",
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        before + 1,
        "the actual retained trigger callback still runs after migration"
    );
}

fn scenario(
    original: &super::fixture::Fixture,
    registration: &EmbedderRegistration,
    migration: bool,
) -> Json {
    let mut logs: [Log; 3] = std::array::from_fn(|_| Log::default());
    let mut rounds = 0;
    let mut writer_peak = 0;
    while logs.iter().any(|log| log.overlapped_ns.len() < 20)
        && logs
            .iter()
            .all(|log| log.refusals == 0 && log.wrong_answers == 0)
    {
        if rounds == 100 {
            break;
        }
        rounds += 1;
        let directory = tempfile::tempdir().expect("fresh replay receiver");
        let path = directory.path().join("receiver.db");
        std::fs::copy(&original.db_path, &path).expect("copy the closed sealed fixture");
        let store = context_graph::Store::open(super::store_config(path.clone()))
            .expect("held receiver Store");
        let db = store.sync_database();
        let fixture = super::fixture::Fixture::sealed(
            None,
            path,
            original.embedding,
            original.total_passages,
            original.contexts.clone(),
            original.generations.clone(),
            original.total_events,
            original.sample_passages.clone(),
        );
        let mut calls = None;
        let changes = if migration {
            calls = Some(prepare_chain(&db));
            let source = Database::open_memory();
            prepare_chain(&source);
            let since = source.current_lsn();
            source
                .execute(
                    "ALTER TABLE schema_000 ADD COLUMN revised TEXT",
                    &HashMap::new(),
                )
                .unwrap();
            let change = source
                .changes_since(since)
                .expect("admit engine read memory");
            assert_eq!(
                change.ddl.len(),
                1,
                "one root alteration carries its 99 FK dependents"
            );
            change
        } else {
            let mut change = db.changes_since(Lsn(0)).expect("admit engine read memory");
            change.rows.clear();
            change.edges.clear();
            change.vectors.clear();
            assert!(!change.ddl.is_empty(), "replay the real startup schema");
            change
        };
        let measured = overlap(&fixture, registration, &db, changes);
        writer_peak = writer_peak.max(db.write_memory_high_water_for_test());
        if migration {
            assert_eq!(
                db.execute("SELECT revised FROM schema_000", &HashMap::new())
                    .unwrap()
                    .rows,
                vec![vec![Value::Null]]
            );
            assert_chain_kept(&db, calls.as_ref().unwrap());
        }
        for (into, measured) in logs.iter_mut().zip(measured) {
            into.idle_ns.extend(measured.idle_ns);
            into.overlapped_ns.extend(measured.overlapped_ns);
            into.retries += measured.retries;
            into.writable_open_attempts += measured.writable_open_attempts;
            into.refusals += measured.refusals;
            into.wrong_answers += measured.wrong_answers;
            into.charged_bytes = into.charged_bytes.max(measured.charged_bytes);
        }
    }
    let mut result = serde_json::Map::new();
    for (kind, log) in KINDS.into_iter().zip(logs) {
        result.insert(kind.into(), json!({
            "overlapped_queries": log.overlapped_ns.len(), "overlapped_p95_ns": percentile(&log.overlapped_ns),
            "idle_queries": log.idle_ns.len(), "idle_p95_ns": percentile(&log.idle_ns),
            "overlapped_durations_ns": log.overlapped_ns, "refusals": log.refusals,
            "wrong_answers": log.wrong_answers, "retries": log.retries,
            "opened_writable": log.writable_open_attempts > 0,
            "writable_open_attempts": log.writable_open_attempts,
            "charged_memory_high_water_bytes": log.charged_bytes,
        }));
    }
    result.insert("fresh_receivers".into(), json!(rounds));
    result.insert(
        "writer_charged_memory_high_water_bytes".into(),
        json!(writer_peak),
    );
    Json::Object(result)
}

pub fn run(_criterion: &mut criterion::Criterion) {
    if !std::env::args().skip(1).any(|arg| arg.contains(NAME)) {
        return;
    }
    let fixture = super::fixture::get(10_000, 8);
    let registration = super::reader::registration();
    let receipt = json!({
        "schema": "schema-overlap-receipt/1",
        "parameters": {"resident_rows": fixture.total_passages, "contexts": fixture.contexts.len(), "tables": 100},
        "schema_work_entry": "direct replay through the real apply_changes path",
        "read_paths": "ConsumerReader find with real Nomic query embedding and source links, show with linked events, event-history status",
        "identical_schema_replay": scenario(&fixture, &registration, false),
        "largest_affected_migration": scenario(&fixture, &registration, true),
    });
    let path = std::env::var_os("CG_SCHEMA_OVERLAP_RECEIPT")
        .map(std::path::PathBuf::from)
        .expect("CG_SCHEMA_OVERLAP_RECEIPT names the standalone receipt");
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt).unwrap())
        .expect("write schema overlap receipt");
    println!("SCHEMA_OVERLAP_RECEIPT {}", path.display());
    for section in ["identical_schema_replay", "largest_affected_migration"] {
        for kind in KINDS {
            assert_eq!(
                receipt[section][kind]["refusals"].as_u64().unwrap(),
                0,
                "{section}/{kind}: no retries or refusals"
            );
            assert_eq!(
                receipt[section][kind]["retries"].as_u64().unwrap(),
                0,
                "{section}/{kind}: read paths do not retry"
            );
            assert_eq!(
                receipt[section][kind]["writable_open_attempts"]
                    .as_u64()
                    .unwrap(),
                0,
                "{section}/{kind}: read paths do not open writable stores"
            );
            assert_eq!(
                receipt[section][kind]["wrong_answers"].as_u64().unwrap(),
                0,
                "{section}/{kind}: complete idle answer preserved"
            );
            assert!(
                receipt[section][kind]["overlapped_queries"]
                    .as_u64()
                    .unwrap()
                    >= 20,
                "{section}/{kind}: at least twenty actual overlaps"
            );
        }
        assert!(
            receipt[section]["find"]["overlapped_p95_ns"]
                .as_u64()
                .unwrap()
                <= 200_000_000,
            "{section}: find p95 exceeds 200 ms"
        );
        assert!(
            receipt[section]["show"]["overlapped_p95_ns"]
                .as_u64()
                .unwrap()
                <= 50_000_000,
            "{section}: show p95 exceeds 50 ms"
        );
    }
}
