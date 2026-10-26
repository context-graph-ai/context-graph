//! A fixed new source advances without rereading completed capture or passage history.
use super::*;
use context_graph::test_support::{
    record_event_history_read_cost_for_test, take_event_history_read_cost_for_test,
};

#[test]
fn one_new_source_keeps_prior_passages_and_reads_only_the_incremental_suffix() {
    let _clock = Wallclock::test_clock_guard(|| 1_800_000_000_000);
    let policy = EventSearchWorkPolicy {
        order: EventSearchOrder::Oldest,
        sessions: EventSearchSessionScope::All,
    };
    let measure = |history: usize| {
        let root = tempfile::tempdir().unwrap();
        let cfg = config(root.path().join("incremental-source.db"));
        let store = open(&cfg);
        let context = create_context(&store);
        let capture = |store: &Store, session: &str| {
            let events = (1..=6)
                .map(|ordinal| EventDraft {
                    session_ref: session.into(),
                    ..draft(
                        &format!("{session}-{ordinal}"),
                        ordinal,
                        if ordinal % 2 == 0 {
                            EventRenderRole::AgentMessage
                        } else {
                            EventRenderRole::HumanMessage
                        },
                        &format!("{session} turn {ordinal} about source advancement"),
                    )
                })
                .collect();
            store
                .append_event_batch(AppendEventBatch::live(
                    context,
                    declaration(),
                    format!("{session}-source"),
                    events,
                ))
                .unwrap();
        };
        for index in 0..history {
            capture(&store, &format!("completed-{index:03}"));
        }
        drop(store);
        let embedder = ControlledEmbedder::new();
        let before = EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(&cfg, context, &embedder, profile(), 8, OBSERVED_LEASE_MS),
            policy,
        )
        .result
        .expect("complete the captured prefix without retries");
        assert!(before.completed_at.is_some());
        let store = open(&cfg);
        let prefix = current_passages(&store, context);
        assert!(!prefix.is_empty(), "the fixture has a searchable prefix");
        capture(&store, "new-session");
        drop(store);
        let embedded_before = embedder.embedded().len();

        record_event_history_read_cost_for_test();
        let after = EventSearchProjectionRunner::catch_up_observed_with_policy(
            run_request(&cfg, context, &embedder, profile(), 8, OBSERVED_LEASE_MS),
            policy,
        )
        .result
        .expect("complete the new source without retries");
        let cost = take_event_history_read_cost_for_test();
        assert!(after.completed_at.is_some());
        assert_eq!(after.failed_event_count, 0);
        assert_eq!(after.generation_id, before.generation_id);
        assert!(
            !cost.reads.is_empty(),
            "the real projection reads were recorded"
        );
        let mut by_sql = std::collections::BTreeMap::<String, u64>::new();
        for read in cost.reads {
            *by_sql.entry(read.sql).or_default() += read.rows_examined.unwrap_or(0);
        }
        let examined: u64 = by_sql.values().sum();
        assert!(
            examined > 0,
            "the engine examined-entry meter is not vacuous"
        );
        let embedded = embedder.embedded().len() - embedded_before;
        assert!(embedded > 0, "the new source becomes searchable");
        let store = open(&cfg);
        let current = current_passages(&store, context);
        for passage in &prefix {
            assert!(
                current.contains(passage),
                "a completed prefix passage keeps its id and text"
            );
        }
        assert!(
            current.iter().any(|(_, text)| text.contains("new-session")),
            "the newly captured session is included"
        );

        // Independently recompute the published fold from all current passage
        // ordinals and content hashes, outside the measured projection work.
        use sha2::{Digest, Sha256};
        let rows = store.execute_sql_for_test(
            "SELECT passage_ordinal, content_sha256 FROM event_search_passages WHERE generation_id = $generation_id AND is_current = true ORDER BY passage_ordinal ASC, id ASC",
            &HashMap::from([("generation_id".into(), Value::Uuid(after.generation_id.as_uuid()))]),
        ).unwrap();
        let mut digest = Sha256::new();
        digest.update(b"context-graph/event-search-generation/v1");
        for row in rows.rows() {
            let [Value::Int64(ordinal), Value::Text(content)] = row.as_slice() else {
                panic!("typed passage fold input: {row:?}");
            };
            let ordinal = u64::try_from(*ordinal).unwrap().to_be_bytes();
            for bytes in [ordinal.as_slice(), content.as_bytes()] {
                digest.update(u64::try_from(bytes.len()).unwrap().to_be_bytes());
                digest.update(bytes);
            }
        }
        assert_eq!(
            after.generation_content_sha256,
            format!("{:x}", digest.finalize()),
            "the carried prefix and new suffix produce the complete published fold"
        );
        assert_eq!(after.passage_count, rows.rows().len() as u64);

        let verified = store
            .verify_event_search_generation(context, after.generation_id.as_uuid())
            .expect("the full verifier accepts the completed prefix and new source");
        assert_eq!(
            verified.member_count,
            u64::try_from((history + 1) * 6).unwrap(),
            "full verification includes all six captured events of every fixture source"
        );
        assert_eq!(
            verified.ordered_source_set_sha256, after.source_watermark.ordered_source_set_sha256,
            "the full membership walk agrees with the published source identities"
        );
        assert_eq!(
            verified.source_content_sha256, after.source_content_sha256,
            "the full membership walk agrees with the published source content"
        );
        assert_eq!(verified.ordered_source_set_sha256.len(), 64);
        assert_eq!(verified.source_content_sha256.len(), 64);
        eprintln!("full generation verification: {verified:?}");
        (examined, embedded, by_sql)
    };
    let small = measure(3);
    let large = measure(12);
    eprintln!(
        "completed_sources=3: examined={}, embedded={}; completed_sources=12: examined={}, embedded={}",
        small.0, small.1, large.0, large.1
    );
    assert_eq!(
        large.1, small.1,
        "only the same new source is embedded at either history size"
    );
    assert_eq!(
        large.0, small.0,
        "a fixed newly captured source must not rescan completed sources or passage folds; small={small:#?}; large={large:#?}"
    );
}
