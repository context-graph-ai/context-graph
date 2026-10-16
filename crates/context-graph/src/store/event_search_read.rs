//! Shared database-only reads for event-history search.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use contextdb_core::Value;
use contextdb_engine::QueryResult;

use crate::embedder::{Embedder, EventSearchEmbedder};
use crate::{
    CgError, ConsumerReader, ContextId, ContextSelector, EmbedderRegistration, EmbeddingId,
    EventCoverageCounts, EventId, EventSearchContextCoverage, EventSearchContextState,
    EventSearchFilter, EventSearchGenerationId, EventSearchHit, EventSearchPassage,
    EventSearchPassageId, EventSearchResult, EventWatermark, PassageEventSource, Result, Store,
    TxId,
};

use super::convert::map_engine_err;

type EventSearchRowVisitor<'a> = dyn FnMut(&[String], &[Value]) -> Result<()> + 'a;

pub(crate) trait EventSearchRows {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult>;
    fn stream_rows(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
        visit: &mut EventSearchRowVisitor<'_>,
    ) -> Result<()>;
    fn event_search_embedder(&self, embedding_space_id: &str) -> Option<Arc<dyn Embedder>>;
    /// Whether this door may read `event_context_counts`. A read-only consumer that declared only
    /// the events table recounts from the events it declared.
    fn reads_event_context_counts(&self) -> bool;
}

// ---------------------------------------------------------------------------------------------
// Statement recording (test-only)
// ---------------------------------------------------------------------------------------------

// Every statement the event-history reads issue goes through `EventSearchRows`, so recording here
// records the whole shape of a `find`, a `status` or a projection pass: how many statements it
// costs and what each one is. A proof that a read does not fan out per Context, or never falls back
// to a recount, is a statement about this list and nothing else -- never about elapsed time.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
thread_local! {
    static RECORDED_STATEMENTS: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// Start recording event-history statements on this thread, discarding anything already recorded.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn record_event_history_statements_for_test() {
    RECORDED_STATEMENTS.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
}

/// Take what has been recorded on this thread and stop recording.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn take_recorded_event_history_statements_for_test() -> Vec<String> {
    RECORDED_STATEMENTS.with(|slot| slot.borrow_mut().take().unwrap_or_default())
}

#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
fn record_statement(sql: &str) {
    RECORDED_STATEMENTS.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut()
            && let Some(recorded) = slot.as_mut()
        {
            recorded.push(sql.to_string());
        }
    });
}

/// Compiled out of every build that does not activate the test seam, so a shipped read pays
/// nothing for it.
#[cfg(not(any(context_graph_internal_test_support, feature = "test-support")))]
#[inline(always)]
fn record_statement(_sql: &str) {}

impl EventSearchRows for Store {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        record_statement(sql);
        self.require_application_ready()?;
        self.internal_db()
            .execute(sql, params)
            .map_err(map_engine_err)
    }

    fn stream_rows(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
        visit: &mut EventSearchRowVisitor<'_>,
    ) -> Result<()> {
        record_statement(sql);
        self.require_application_ready()?;
        let session = self
            .internal_db()
            .read_session(self.read_limits())
            .map_err(map_engine_err)?;
        let mut cursor = session.open_cursor(sql, params).map_err(map_engine_err)?;
        let read = (|| -> Result<()> {
            let first = cursor.first_page();
            let mut has_more = first.has_more;
            for row in &first.rows {
                visit(&first.columns, row)?;
            }
            while has_more {
                let page = cursor.fetch(None).map_err(map_engine_err)?;
                has_more = page.has_more;
                for row in &page.rows {
                    visit(&page.columns, row)?;
                }
            }
            Ok(())
        })();
        let closed = cursor.close().map_err(map_engine_err);
        read?;
        closed
    }

    fn event_search_embedder(&self, embedding_space_id: &str) -> Option<Arc<dyn Embedder>> {
        self.embedders.get(embedding_space_id).cloned()
    }

    fn reads_event_context_counts(&self) -> bool {
        true
    }
}

impl EventSearchRows for ConsumerReader {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        record_statement(sql);
        let mut cursor = self.open_cursor(sql, params)?;
        let paged = (|| -> Result<(Vec<String>, Vec<Vec<Value>>)> {
            let first = cursor.first_page()?;
            let mut columns = first.columns.clone();
            let mut rows = first.rows.clone();
            let mut has_more = first.has_more;
            while has_more {
                let page = cursor.fetch(None)?;
                if columns.is_empty() {
                    columns = page.columns;
                }
                rows.extend(page.rows);
                has_more = page.has_more;
            }
            Ok((columns, rows))
        })();
        let closed = cursor.close();
        let (columns, rows) = paged?;
        closed?;
        Ok(QueryResult {
            columns,
            rows,
            ..QueryResult::empty()
        })
    }

    fn stream_rows(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
        visit: &mut EventSearchRowVisitor<'_>,
    ) -> Result<()> {
        record_statement(sql);
        let mut cursor = self.open_cursor(sql, params)?;
        let read = (|| -> Result<()> {
            let first = cursor.first_page()?;
            let mut has_more = first.has_more;
            for row in &first.rows {
                visit(&first.columns, row)?;
            }
            while has_more {
                let page = cursor.fetch(None)?;
                has_more = page.has_more;
                for row in &page.rows {
                    visit(&page.columns, row)?;
                }
            }
            Ok(())
        })();
        let closed = cursor.close();
        read?;
        closed
    }

    fn event_search_embedder(&self, embedding_space_id: &str) -> Option<Arc<dyn Embedder>> {
        self.attached_event_search_embedder(embedding_space_id)
    }

    fn reads_event_context_counts(&self) -> bool {
        self.declaration()
            .schema
            .tables
            .iter()
            .any(|table| table.name == EVENT_CONTEXT_COUNTS_TABLE)
    }
}

/// The per-Context captured-event count table (see `schema.rs`).
pub(crate) const EVENT_CONTEXT_COUNTS_TABLE: &str = "event_context_counts";

/// One Context's captured-event totals: every captured event, and the same events by coverage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CapturedEventCounts {
    pub(crate) captured: u64,
    pub(crate) coverage: EventCoverageCounts,
}

impl CapturedEventCounts {
    /// The count-row columns and their values, in one fixed order.
    pub(crate) fn columns(&self) -> [(&'static str, Value); 5] {
        let int = |value: u64| Value::Int64(i64::try_from(value).unwrap_or(i64::MAX));
        [
            ("captured_events", int(self.captured)),
            ("complete_events", int(self.coverage.complete)),
            ("opaque_events", int(self.coverage.opaque)),
            ("payload_refused_events", int(self.coverage.payload_refused)),
            (
                "source_declared_missing_events",
                int(self.coverage.source_declared_missing),
            ),
        ]
    }

    pub(crate) fn add(&self, other: &CapturedEventCounts) -> CapturedEventCounts {
        CapturedEventCounts {
            captured: self.captured.saturating_add(other.captured),
            coverage: EventCoverageCounts {
                complete: self
                    .coverage
                    .complete
                    .saturating_add(other.coverage.complete),
                opaque: self.coverage.opaque.saturating_add(other.coverage.opaque),
                payload_refused: self
                    .coverage
                    .payload_refused
                    .saturating_add(other.coverage.payload_refused),
                source_declared_missing: self
                    .coverage
                    .source_declared_missing
                    .saturating_add(other.coverage.source_declared_missing),
            },
        }
    }
}

/// The count rows stored for `context_ids`; a Context without a row is absent from the answer.
pub(crate) fn stored_event_context_counts<R: EventSearchRows + ?Sized>(
    rows: &R,
    context_ids: &[ContextId],
) -> Result<HashMap<ContextId, CapturedEventCounts>> {
    let mut stored = HashMap::new();
    if context_ids.is_empty() {
        return Ok(stored);
    }
    let mut params = HashMap::new();
    let filter = bind_contexts(&mut params, "count_context", context_ids.iter().copied());
    let answer = rows.query_rows(
        &format!(
            "SELECT context_id, captured_events, complete_events, opaque_events, \
             payload_refused_events, source_declared_missing_events \
             FROM {EVENT_CONTEXT_COUNTS_TABLE} WHERE {filter}"
        ),
        &params,
    )?;
    for row in &answer.rows {
        stored.insert(
            ContextId::from(get_uuid(row, &answer.columns, "context_id")?),
            CapturedEventCounts {
                captured: get_u64(row, &answer.columns, "captured_events")?,
                coverage: EventCoverageCounts {
                    complete: get_u64(row, &answer.columns, "complete_events")?,
                    opaque: get_u64(row, &answer.columns, "opaque_events")?,
                    payload_refused: get_u64(row, &answer.columns, "payload_refused_events")?,
                    source_declared_missing: get_u64(
                        row,
                        &answer.columns,
                        "source_declared_missing_events",
                    )?,
                },
            },
        );
    }
    Ok(stored)
}

/// Count one Context's captured events from the events table itself. Only a missing count row (a
/// Context never captured, or a forget interrupted before it rewrote the row) and the rewrite
/// after a forget come here; ordinary reads use the row.
pub(crate) fn count_captured_events<R: EventSearchRows + ?Sized>(
    rows: &R,
    context_id: ContextId,
) -> Result<CapturedEventCounts> {
    let count = |sql: &str, params: &HashMap<String, Value>| -> Result<u64> {
        let answer = rows.query_rows(sql, params)?;
        let row = answer
            .rows
            .first()
            .ok_or_else(|| CgError::Engine("an event count returned no aggregate row".into()))?;
        get_u64(row, &answer.columns, "n")
    };
    let params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]);
    let by_coverage = |coverage: &str| -> Result<u64> {
        let mut with = params.clone();
        with.insert("coverage".into(), Value::Text(coverage.into()));
        count(
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id AND coverage = $coverage",
            &with,
        )
    };
    Ok(CapturedEventCounts {
        captured: count(
            "SELECT COUNT(*) AS n FROM events WHERE context_id = $context_id",
            &params,
        )?,
        coverage: EventCoverageCounts {
            complete: by_coverage("complete")?,
            opaque: by_coverage("opaque")?,
            payload_refused: by_coverage("payload_refused")?,
            source_declared_missing: by_coverage("source_declared_missing")?,
        },
    })
}

/// Captured-event totals for every Context in `context_ids`: the count row where it exists,
/// otherwise a recount from events.
pub(crate) fn captured_event_counts<R: EventSearchRows + ?Sized>(
    rows: &R,
    context_ids: &[ContextId],
) -> Result<HashMap<ContextId, CapturedEventCounts>> {
    // A reader that omits the count table is refused by table name before any recount.
    if !rows.reads_event_context_counts() {
        return Err(CgError::UndeclaredTableRefused {
            table: EVENT_CONTEXT_COUNTS_TABLE.to_string(),
        });
    }
    let mut counts = stored_event_context_counts(rows, context_ids)?;
    // A Context with no row has captured nothing, or had its row taken away by a `forget` that did
    // not finish rewriting it. The second is why this recounts rather than answering zero: a
    // Context whose events are still there must not read as empty.
    for context_id in context_ids {
        if !counts.contains_key(context_id) {
            counts.insert(*context_id, count_captured_events(rows, *context_id)?);
        }
    }
    Ok(counts)
}

pub(crate) fn registration_map(
    registrations: Vec<EmbedderRegistration>,
) -> Result<HashMap<String, Arc<dyn Embedder>>> {
    let mut attached = HashMap::new();
    for registration in registrations {
        let declared = registration.embedder.embedding_space();
        if declared.embedding_space_id != registration.space.embedding_space_id {
            return Err(CgError::EmbeddingSpaceIdMismatch {
                embedder_declared: declared.embedding_space_id,
                registry_declared: registration.space.embedding_space_id,
            });
        }
        if registration.embedder.supported_inputs().is_empty() {
            return Err(CgError::EmbedderDeclaresNoSupportedInputs {
                embedding_space_id: registration.space.embedding_space_id,
            });
        }
        crate::embedder::verify_event_history_authority(
            registration.embedder.as_ref(),
            &registration.space.embedding_space_id,
        )?;
        attached.insert(
            registration.space.embedding_space_id,
            Arc::clone(&registration.embedder),
        );
    }
    Ok(attached)
}

#[derive(Debug, Clone)]
struct ReadyHead {
    context_id: ContextId,
    generation_id: EventSearchGenerationId,
    source_sha256: String,
    embedding_space_id: String,
    encoding_profile: String,
}

struct ClassifiedContexts {
    selected_context_ids: Vec<ContextId>,
    coverage: Vec<EventSearchContextCoverage>,
    ready_heads: Vec<ReadyHead>,
}

pub(crate) fn find_event_history(
    rows: &dyn EventSearchRows,
    selector: ContextSelector,
    query: &str,
    filter: EventSearchFilter,
) -> Result<EventSearchResult> {
    if query.trim().is_empty() {
        return Err(CgError::EmptyEventSearchQuery);
    }
    if filter.limit.get() > crate::EVENT_SEARCH_MAX_LIMIT {
        return Err(CgError::EventSearchLimitExceeded {
            requested: filter.limit.get(),
            maximum: crate::EVENT_SEARCH_MAX_LIMIT,
        });
    }

    let classified = classify_contexts(rows, selector)?;
    let searched_context_ids = classified
        .ready_heads
        .iter()
        .map(|head| head.context_id)
        .collect::<Vec<_>>();
    if classified.ready_heads.is_empty() {
        if classified
            .coverage
            .iter()
            .any(|coverage| coverage.state == EventSearchContextState::NotReady)
        {
            return Err(CgError::HistorySearchNotReady {
                context_ids: classified.selected_context_ids,
            });
        }
        return Ok(EventSearchResult {
            selected_context_ids: classified.selected_context_ids,
            searched_context_ids,
            coverage: classified.coverage,
            hits: Vec::new(),
        });
    }

    let (embedding_space_id, encoding_profile) = comparable_profile(&classified.ready_heads)?;
    let embedder = rows
        .event_search_embedder(&embedding_space_id)
        .ok_or_else(|| CgError::UnknownEmbeddingSpace {
            id: embedding_space_id.clone(),
        })?;
    let embedder_space = embedder.embedding_space();
    if embedder_space.embedding_space_id != embedding_space_id {
        return Err(CgError::EmbeddingSpaceIdMismatch {
            embedder_declared: embedder_space.embedding_space_id,
            registry_declared: embedding_space_id,
        });
    }
    let actual_profile = embedder.encoding_profile_id();
    if actual_profile != encoding_profile {
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "encoding_profile",
            expected: encoding_profile,
            actual: actual_profile,
        });
    }
    let prepared = embedder.prepare_query(query)?;
    let output = embedder.embed_prepared_query(&prepared)?;
    if output.embedding_space_id != embedder_space.embedding_space_id {
        return Err(CgError::EmbeddingSpaceIdMismatch {
            embedder_declared: output.embedding_space_id,
            registry_declared: embedder_space.embedding_space_id,
        });
    }
    if output.dimension != embedder_space.dimension || output.vector.len() != output.dimension {
        return Err(CgError::EmbeddingDimensionMismatch {
            space_id: output.embedding_space_id,
            expected: u32::try_from(embedder_space.dimension).unwrap_or(u32::MAX),
            found: u32::try_from(output.vector.len()).unwrap_or(u32::MAX),
        });
    }
    if output.vector.iter().any(|value| !value.is_finite()) {
        return Err(CgError::InvalidEmbeddingVectorFloat);
    }

    let mut params = HashMap::from([
        ("query".to_string(), Value::Vector(output.vector)),
        ("is_current".to_string(), Value::Bool(true)),
        ("is_searchable".to_string(), Value::Bool(true)),
    ]);
    // One statement selects every ready (Context, active generation) partition of the vector
    // column, named as the pairs themselves: the standard row-value list, so the engine plans one
    // index run per active pair rather than one per combination of the two id lists. ContextDB
    // merges the selected partitions before the single top-k, and `is_current` stays a filter
    // inside them.
    let pair_filter = bind_context_generation_pairs(
        &mut params,
        "search",
        classified
            .ready_heads
            .iter()
            .map(|head| (head.context_id, head.generation_id)),
    );
    let mut predicates = vec![
        pair_filter,
        "is_current = $is_current".to_string(),
        "is_searchable = $is_searchable".to_string(),
    ];
    if let Some(connector_id) = &filter.connector_id {
        params.insert("connector_filter".into(), Value::Text(connector_id.clone()));
        predicates.push("connector_id = $connector_filter".into());
    }
    if let Some(session_ref) = &filter.session_ref {
        params.insert("session_filter".into(), Value::Text(session_ref.clone()));
        predicates.push("session_ref = $session_filter".into());
    }
    if let Some(source_instance_ref) = &filter.source_instance_ref {
        params.insert(
            "source_filter".into(),
            Value::Text(source_instance_ref.clone()),
        );
        predicates.push("source_instance_ref = $source_filter".into());
    }
    let sql = format!(
        "SELECT id, context_id, generation_id, connector_id, session_ref, source_instance_ref, \
         first_event_sequence, last_event_sequence, rendered_text, dimension, score \
         FROM event_search_passages WHERE {} \
         ORDER BY vector_text <=> $query LIMIT {}",
        predicates.join(" AND "),
        filter.limit.get(),
    );
    let ranked = rows.query_rows(&sql, &params)?;
    let active_pairs = classified
        .ready_heads
        .iter()
        .map(|head| {
            (
                (head.context_id, head.generation_id),
                head.embedding_space_id.clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut hits = Vec::with_capacity(ranked.rows.len());
    for row in &ranked.rows {
        let context_id = ContextId::from(get_uuid(row, &ranked.columns, "context_id")?);
        let generation_id =
            EventSearchGenerationId::from(get_uuid(row, &ranked.columns, "generation_id")?);
        let Some(space_id) = active_pairs.get(&(context_id, generation_id)) else {
            return Err(CgError::Engine(
                "event-history ranking returned a passage outside the active selected heads".into(),
            ));
        };
        let dimension = get_u32(row, &ranked.columns, "dimension")?;
        if dimension as usize != output.dimension {
            return Err(CgError::EmbeddingDimensionMismatch {
                space_id: space_id.clone(),
                expected: u32::try_from(output.dimension).unwrap_or(u32::MAX),
                found: dimension,
            });
        }
        let score = get_f32(row, &ranked.columns, "score")?;
        if !score.is_finite() {
            return Err(CgError::Engine(
                "event-history ranking returned a non-finite cosine similarity".into(),
            ));
        }
        hits.push(EventSearchHit {
            passage_id: EventSearchPassageId::from(get_uuid(row, &ranked.columns, "id")?),
            context_id,
            connector_id: get_text(row, &ranked.columns, "connector_id")?,
            session_ref: get_text(row, &ranked.columns, "session_ref")?,
            source_instance_ref: get_text(row, &ranked.columns, "source_instance_ref")?,
            first_event_seq: get_u64(row, &ranked.columns, "first_event_sequence")?,
            last_event_seq: get_u64(row, &ranked.columns, "last_event_sequence")?,
            first_occurred_at: None,
            last_occurred_at: None,
            text: get_text(row, &ranked.columns, "rendered_text")?,
            cosine_similarity: score,
            applied_filter: filter.clone(),
            embedding_space_id: space_id.clone(),
            generation_id,
            sources: Vec::new(),
        });
    }

    let requested = hits
        .iter()
        .map(|hit| (hit.context_id, hit.passage_id))
        .collect::<Vec<_>>();
    let mut sources = load_sources(rows, &requested)?;
    let event_times = load_event_times(rows, sources.values().flatten())?;
    for hit in &mut hits {
        hit.sources = sources.remove(&hit.passage_id).ok_or_else(|| {
            CgError::Engine(format!(
                "event-search passage {} has no persisted source links",
                hit.passage_id
            ))
        })?;
        (hit.first_occurred_at, hit.last_occurred_at) =
            source_time_range(&hit.sources, &event_times);
    }
    hits.sort_by(|left, right| {
        right
            .cosine_similarity
            .total_cmp(&left.cosine_similarity)
            .then_with(|| left.passage_id.cmp(&right.passage_id))
    });

    Ok(EventSearchResult {
        selected_context_ids: classified.selected_context_ids,
        searched_context_ids,
        coverage: classified.coverage,
        hits,
    })
}

pub(crate) fn get_event_search_passage(
    rows: &dyn EventSearchRows,
    selector: ContextSelector,
    passage_id: EventSearchPassageId,
) -> Result<Option<EventSearchPassage>> {
    let selected = selector.into_context_ids();
    let _ = verify_contexts(rows, &selected)?;
    let mut params = HashMap::from([("passage_id".into(), Value::Uuid(passage_id.as_uuid()))]);
    let context_filter = bind_contexts(&mut params, "passage_context", selected.iter().copied());
    let passage = rows.query_rows(
        &format!(
            "SELECT id, context_id, generation_id, connector_id, session_ref, \
             source_instance_ref, passage_ordinal, first_event_sequence, last_event_sequence, \
             rendered_text, content_sha256, dimension, is_current, created_at \
             FROM event_search_passages WHERE id = $passage_id AND is_searchable = $is_searchable AND {context_filter} LIMIT 1"
        ),
        &{
            params.insert("is_searchable".into(), Value::Bool(true));
            params
        },
    )?;
    let Some(row) = passage.rows.first() else {
        return Ok(None);
    };
    let context_id = ContextId::from(get_uuid(row, &passage.columns, "context_id")?);
    let generation_id =
        EventSearchGenerationId::from(get_uuid(row, &passage.columns, "generation_id")?);

    let generation = rows.query_rows(
        "SELECT embedding_space_id FROM event_search_generations \
         WHERE id = $generation_id AND context_id = $context_id LIMIT 1",
        &HashMap::from([
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ]),
    )?;
    let generation_row = generation.rows.first().ok_or_else(|| {
        CgError::Engine(format!(
            "event-search passage {passage_id} names a missing generation"
        ))
    })?;
    let embedding_space_id = get_text(generation_row, &generation.columns, "embedding_space_id")?;
    let handle = rows.query_rows(
        "SELECT id, embedding_space_id FROM event_search_embedding_handles \
         WHERE passage_id = $passage_id AND context_id = $context_id \
         AND generation_id = $generation_id AND embedding_space_id = $embedding_space_id LIMIT 1",
        &HashMap::from([
            ("passage_id".into(), Value::Uuid(passage_id.as_uuid())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("generation_id".into(), Value::Uuid(generation_id.as_uuid())),
            (
                "embedding_space_id".into(),
                Value::Text(embedding_space_id.clone()),
            ),
        ]),
    )?;
    let handle_row = handle.rows.first().ok_or_else(|| {
        CgError::Engine(format!(
            "event-search passage {passage_id} has no matching embedding handle"
        ))
    })?;

    let mut sources = load_sources(rows, &[(context_id, passage_id)])?
        .remove(&passage_id)
        .ok_or_else(|| {
            CgError::Engine(format!(
                "event-search passage {passage_id} has no persisted source links"
            ))
        })?;
    sources.sort_by_key(|source| source.source_order);
    let event_times = load_event_times(rows, sources.iter())?;
    let (first_occurred_at, last_occurred_at) = source_time_range(&sources, &event_times);

    Ok(Some(EventSearchPassage {
        id: passage_id,
        context_id,
        generation_id,
        connector_id: get_text(row, &passage.columns, "connector_id")?,
        session_ref: get_text(row, &passage.columns, "session_ref")?,
        source_instance_ref: get_text(row, &passage.columns, "source_instance_ref")?,
        ordinal: get_u64(row, &passage.columns, "passage_ordinal")?,
        first_event_seq: get_u64(row, &passage.columns, "first_event_sequence")?,
        last_event_seq: get_u64(row, &passage.columns, "last_event_sequence")?,
        first_occurred_at,
        last_occurred_at,
        text: get_text(row, &passage.columns, "rendered_text")?,
        content_sha256: get_text(row, &passage.columns, "content_sha256")?,
        embedding_id: EmbeddingId::from(get_uuid(handle_row, &handle.columns, "id")?),
        embedding_space_id,
        dimension: get_u32(row, &passage.columns, "dimension")?,
        is_current: get_bool(row, &passage.columns, "is_current")?,
        created_at: get_txid(row, &passage.columns, "created_at")?,
        sources,
    }))
}

pub(crate) fn list_events_for_passage(
    rows: &dyn EventSearchRows,
    selector: ContextSelector,
    passage_id: EventSearchPassageId,
) -> Result<Vec<PassageEventSource>> {
    let selected = selector.into_context_ids();
    let _ = verify_contexts(rows, &selected)?;
    Ok(load_sources(
        rows,
        &selected
            .into_iter()
            .map(|context_id| (context_id, passage_id))
            .collect::<Vec<_>>(),
    )?
    .remove(&passage_id)
    .unwrap_or_default())
}

fn classify_contexts(
    rows: &dyn EventSearchRows,
    selector: ContextSelector,
) -> Result<ClassifiedContexts> {
    let selected_context_ids = verify_contexts(rows, &selector.into_context_ids())?;

    let captured = captured_event_counts(rows, &selected_context_ids)?;

    let mut head_params = HashMap::new();
    let head_filter = bind_contexts(
        &mut head_params,
        "head_context",
        selected_context_ids.iter().copied(),
    );
    let heads = rows.query_rows(
        &format!("SELECT context_id, generation_id FROM event_search_heads WHERE {head_filter}"),
        &head_params,
    )?;
    let mut generation_for_context = HashMap::new();
    for row in &heads.rows {
        generation_for_context.insert(
            ContextId::from(get_uuid(row, &heads.columns, "context_id")?),
            EventSearchGenerationId::from(get_uuid(row, &heads.columns, "generation_id")?),
        );
    }

    let mut ready_by_context = HashMap::new();
    if !generation_for_context.is_empty() {
        let mut generation_params = HashMap::new();
        let generation_filter = bind_uuid_values(
            &mut generation_params,
            "id",
            "active_generation",
            generation_for_context
                .values()
                .map(|generation_id| generation_id.as_uuid()),
        );
        let generations = rows.query_rows(
            &format!(
                "SELECT id, context_id, event_count, ordered_source_set_sha256, \
                 embedding_space_id, encoding_profile, state FROM event_search_generations \
                 WHERE {generation_filter}"
            ),
            &generation_params,
        )?;
        for row in &generations.rows {
            let state = get_text(row, &generations.columns, "state")?;
            if state != "ready" && state != "building" {
                continue;
            }
            let context_id = ContextId::from(get_uuid(row, &generations.columns, "context_id")?);
            let generation_id =
                EventSearchGenerationId::from(get_uuid(row, &generations.columns, "id")?);
            if generation_for_context.get(&context_id) != Some(&generation_id) {
                return Err(CgError::Engine(
                    "event-search head and ready generation disagree on Context".into(),
                ));
            }
            ready_by_context.insert(
                context_id,
                ReadyHead {
                    context_id,
                    generation_id,
                    source_sha256: get_text(
                        row,
                        &generations.columns,
                        "ordered_source_set_sha256",
                    )?,
                    embedding_space_id: get_text(row, &generations.columns, "embedding_space_id")?,
                    encoding_profile: get_text(row, &generations.columns, "encoding_profile")?,
                },
            );
        }
    }

    // Coverage bookkeeping for the WHOLE selected set, in a statement count that does not grow
    // with the Context count: one policy read, two session reads, and -- when anything is ready --
    // one generation read and one session-progress read. A read per Context here would put four
    // statements at one Context and 256 at sixty-four on the critical find path.
    let policies =
        super::event_search_projection::persisted_work_policies(rows, &selected_context_ids)?;
    let session_work = super::event_search_projection::selected_captured_session_work_for(
        rows,
        &selected_context_ids,
        &policies,
    )?;
    let captured_totals = selected_context_ids
        .iter()
        .map(|context_id| {
            (
                *context_id,
                captured
                    .get(context_id)
                    .map(|counts| counts.captured)
                    .unwrap_or(0),
            )
        })
        .collect::<HashMap<_, _>>();
    let ready_pairs = selected_context_ids
        .iter()
        .filter_map(|context_id| {
            ready_by_context
                .get(context_id)
                .map(|head| (*context_id, head.generation_id))
        })
        .collect::<Vec<_>>();
    let mut policy_coverages = super::event_search_projection::policy_coverage_for(
        rows,
        &ready_pairs,
        &session_work,
        &captured_totals,
    )?;

    let mut coverage = Vec::with_capacity(selected_context_ids.len());
    let mut ready_heads = Vec::new();
    for context_id in &selected_context_ids {
        let captured_event_count = captured_totals.get(context_id).copied().unwrap_or(0);
        let selected_events = session_work
            .get(context_id)
            .map(super::event_search_projection::SessionWorkSet::selected_events)
            .unwrap_or(0);
        if let Some(head) = ready_by_context.remove(context_id) {
            let policy_coverage = policy_coverages.remove(context_id).ok_or_else(|| {
                CgError::Engine("event-search coverage generation returned no row".into())
            })?;
            coverage.push(EventSearchContextCoverage {
                context_id: *context_id,
                state: EventSearchContextState::Ready,
                generation_id: Some(head.generation_id),
                captured_event_count,
                searchable_event_count: policy_coverage.covered_events,
                pending_event_count: policy_coverage.pending_events,
                outside_scope_event_count: policy_coverage.outside_scope_events,
                outside_scope_indexed_event_count: policy_coverage.outside_scope_indexed_events,
                no_searchable_text_event_count: policy_coverage.no_searchable_text_events,
                source_watermark: Some(EventWatermark {
                    event_count: policy_coverage.covered_events,
                    ordered_source_set_sha256: head.source_sha256.clone(),
                }),
            });
            ready_heads.push(head);
        } else {
            coverage.push(EventSearchContextCoverage {
                context_id: *context_id,
                state: if captured_event_count == 0 {
                    EventSearchContextState::Empty
                } else {
                    EventSearchContextState::NotReady
                },
                generation_id: None,
                captured_event_count,
                searchable_event_count: 0,
                pending_event_count: selected_events,
                outside_scope_event_count: captured_event_count.saturating_sub(selected_events),
                outside_scope_indexed_event_count: 0,
                no_searchable_text_event_count: 0,
                source_watermark: None,
            });
        }
    }
    Ok(ClassifiedContexts {
        selected_context_ids,
        coverage,
        ready_heads,
    })
}

fn comparable_profile(heads: &[ReadyHead]) -> Result<(String, String)> {
    let spaces = heads
        .iter()
        .map(|head| head.embedding_space_id.clone())
        .collect::<BTreeSet<_>>();
    if spaces.len() != 1 {
        return Err(CgError::IncomparableEmbeddingSpaces {
            embedding_space_ids: spaces.into_iter().collect(),
        });
    }
    let profiles = heads
        .iter()
        .map(|head| head.encoding_profile.clone())
        .collect::<BTreeSet<_>>();
    let embedding_space_id = spaces
        .into_iter()
        .next()
        .expect("ready heads are non-empty");
    if profiles.len() != 1 {
        let mut profiles = profiles.into_iter();
        let expected = profiles.next().expect("more than one profile exists");
        let actual = profiles.next().expect("more than one profile exists");
        return Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "encoding_profile",
            expected,
            actual,
        });
    }
    Ok((
        embedding_space_id,
        profiles
            .into_iter()
            .next()
            .expect("ready heads are non-empty"),
    ))
}

fn verify_contexts(rows: &dyn EventSearchRows, selected: &[ContextId]) -> Result<Vec<ContextId>> {
    for context_id in selected {
        if !context_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ContextId",
            });
        }
    }
    let mut params = HashMap::new();
    let filter = bind_uuid_values(
        &mut params,
        "id",
        "selected_context",
        selected.iter().map(|context_id| context_id.as_uuid()),
    );
    let answer = rows.query_rows(
        &format!("SELECT id FROM contexts WHERE {filter} ORDER BY created_at ASC"),
        &params,
    )?;
    let ordered = answer
        .rows
        .iter()
        .map(|row| get_uuid(row, &answer.columns, "id").map(ContextId::from))
        .collect::<Result<Vec<_>>>()?;
    let found = ordered.iter().copied().collect::<HashSet<_>>();
    if let Some(missing) = selected
        .iter()
        .find(|context_id| !found.contains(context_id))
    {
        return Err(CgError::NotFound {
            id: Some(missing.to_string()),
            missing: Some("context"),
        });
    }
    Ok(ordered)
}

/// Source links for each requested passage, one primary-key-prefix read per passage
/// (`event_search_passage_sources` is keyed `(context_id, passage_id, source_order)`), so the cost
/// follows the hits, never the Context's link count.
fn load_sources(
    rows: &dyn EventSearchRows,
    requested: &[(ContextId, EventSearchPassageId)],
) -> Result<HashMap<EventSearchPassageId, Vec<PassageEventSource>>> {
    let mut by_passage = HashMap::<EventSearchPassageId, Vec<PassageEventSource>>::new();
    let mut seen = HashSet::new();
    for (context_id, passage_id) in requested {
        if !seen.insert((*context_id, *passage_id)) {
            continue;
        }
        let answer = rows.query_rows(
            "SELECT context_id, generation_id, passage_id, event_id, connector_id, session_ref, \
             source_instance_ref, source_order, event_start_byte, event_end_byte, \
             passage_start_byte, passage_end_byte FROM event_search_passage_sources \
             WHERE context_id = $context_id AND passage_id = $passage_id ORDER BY source_order",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("passage_id".into(), Value::Uuid(passage_id.as_uuid())),
            ]),
        )?;
        for row in &answer.rows {
            let row_context = ContextId::from(get_uuid(row, &answer.columns, "context_id")?);
            let row_passage =
                EventSearchPassageId::from(get_uuid(row, &answer.columns, "passage_id")?);
            if (row_context, row_passage) != (*context_id, *passage_id) {
                return Err(CgError::Engine(
                    "event-search source expansion returned an unrequested passage".into(),
                ));
            }
            by_passage
                .entry(row_passage)
                .or_default()
                .push(PassageEventSource {
                    context_id: row_context,
                    passage_id: row_passage,
                    event_id: EventId::from(get_uuid(row, &answer.columns, "event_id")?),
                    connector_id: get_text(row, &answer.columns, "connector_id")?,
                    session_ref: get_text(row, &answer.columns, "session_ref")?,
                    source_instance_ref: get_text(row, &answer.columns, "source_instance_ref")?,
                    source_order: get_u64(row, &answer.columns, "source_order")?,
                    event_start_byte: get_optional_u64(row, &answer.columns, "event_start_byte")?,
                    event_end_byte: get_optional_u64(row, &answer.columns, "event_end_byte")?,
                    passage_start_byte: get_u64(row, &answer.columns, "passage_start_byte")?,
                    passage_end_byte: get_u64(row, &answer.columns, "passage_end_byte")?,
                });
        }
    }
    for sources in by_passage.values_mut() {
        sources.sort_by_key(|source| source.source_order);
    }
    Ok(by_passage)
}

fn load_event_times<'a>(
    rows: &dyn EventSearchRows,
    sources: impl IntoIterator<Item = &'a PassageEventSource>,
) -> Result<HashMap<EventId, Option<DateTime<Utc>>>> {
    let sources = sources.into_iter().collect::<Vec<_>>();
    if sources.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::new();
    let context_filter = bind_contexts(
        &mut params,
        "event_context",
        sources.iter().map(|source| source.context_id),
    );
    let event_filter = bind_events(
        &mut params,
        "source_event",
        sources.iter().map(|source| source.event_id),
    );
    let answer = rows.query_rows(
        &format!("SELECT id, occurred_at FROM events WHERE {context_filter} AND {event_filter}"),
        &params,
    )?;
    let mut times = HashMap::new();
    for row in &answer.rows {
        times.insert(
            EventId::from(get_uuid(row, &answer.columns, "id")?),
            get_optional_timestamp(row, &answer.columns, "occurred_at")?,
        );
    }
    if let Some(missing) = sources
        .iter()
        .find(|source| !times.contains_key(&source.event_id))
    {
        return Err(CgError::Engine(format!(
            "event-search source link names missing event {}",
            missing.event_id
        )));
    }
    Ok(times)
}

fn source_time_range(
    sources: &[PassageEventSource],
    times: &HashMap<EventId, Option<DateTime<Utc>>>,
) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
    let first = sources
        .first()
        .and_then(|source| times.get(&source.event_id).cloned().flatten());
    let last = sources
        .last()
        .and_then(|source| times.get(&source.event_id).cloned().flatten());
    (first, last)
}

pub(crate) fn bind_contexts(
    params: &mut HashMap<String, Value>,
    prefix: &str,
    values: impl IntoIterator<Item = ContextId>,
) -> String {
    bind_uuid_values(
        params,
        "context_id",
        prefix,
        values.into_iter().map(|id| id.as_uuid()),
    )
}

pub(crate) fn bind_generations(
    params: &mut HashMap<String, Value>,
    prefix: &str,
    values: impl IntoIterator<Item = EventSearchGenerationId>,
) -> String {
    bind_uuid_values(
        params,
        "generation_id",
        prefix,
        values.into_iter().map(|id| id.as_uuid()),
    )
}

/// The (Context, generation) pairs a statement selects, as one row-value list:
/// `(context_id, generation_id) IN (($c0, $g0), ($c1, $g1), ...)`. One pair is written as two
/// equalities.
pub(crate) fn bind_context_generation_pairs(
    params: &mut HashMap<String, Value>,
    prefix: &str,
    pairs: impl IntoIterator<Item = (ContextId, EventSearchGenerationId)>,
) -> String {
    let pairs = pairs
        .into_iter()
        .map(|(context, generation)| (context.as_uuid(), generation.as_uuid()))
        .collect::<BTreeSet<_>>();
    debug_assert!(!pairs.is_empty());
    let mut rows = Vec::with_capacity(pairs.len());
    for (index, (context, generation)) in pairs.into_iter().enumerate() {
        let context_name = format!("{prefix}_context_{index}");
        let generation_name = format!("{prefix}_generation_{index}");
        params.insert(context_name.clone(), Value::Uuid(context));
        params.insert(generation_name.clone(), Value::Uuid(generation));
        rows.push((context_name, generation_name));
    }
    if let [(context, generation)] = rows.as_slice() {
        return format!("context_id = ${context} AND generation_id = ${generation}");
    }
    let listed = rows
        .iter()
        .map(|(context, generation)| format!("(${context}, ${generation})"))
        .collect::<Vec<_>>();
    format!("(context_id, generation_id) IN ({})", listed.join(", "))
}

fn bind_events(
    params: &mut HashMap<String, Value>,
    prefix: &str,
    values: impl IntoIterator<Item = EventId>,
) -> String {
    bind_uuid_values(
        params,
        "id",
        prefix,
        values.into_iter().map(|id| id.as_uuid()),
    )
}

pub(crate) fn bind_uuid_values(
    params: &mut HashMap<String, Value>,
    column: &str,
    prefix: &str,
    values: impl IntoIterator<Item = uuid::Uuid>,
) -> String {
    let values = values.into_iter().collect::<BTreeSet<_>>();
    debug_assert!(!values.is_empty());
    let mut placeholders = Vec::with_capacity(values.len());
    for (index, value) in values.into_iter().enumerate() {
        let name = format!("{prefix}_{index}");
        params.insert(name.clone(), Value::Uuid(value));
        placeholders.push(format!("${name}"));
    }
    if placeholders.len() == 1 {
        format!("{column} = {}", placeholders[0])
    } else {
        format!("{column} IN ({})", placeholders.join(", "))
    }
}

fn column_index(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("event-history read is missing column {name}")))
}

fn get_uuid(row: &[Value], columns: &[String], name: &str) -> Result<uuid::Uuid> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected UUID, got {other:?}"
        ))),
    }
}

fn get_text(row: &[Value], columns: &[String], name: &str) -> Result<String> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Text(value)) => Ok(value.clone()),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text, got {other:?}"
        ))),
    }
}

fn get_u64(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map_err(|_| CgError::Engine(format!("column {name}: expected non-negative integer"))),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer, got {other:?}"
        ))),
    }
}

fn get_u32(row: &[Value], columns: &[String], name: &str) -> Result<u32> {
    u32::try_from(get_u64(row, columns, name)?)
        .map_err(|_| CgError::Engine(format!("column {name}: integer exceeds u32")))
}

fn get_optional_u64(row: &[Value], columns: &[String], name: &str) -> Result<Option<u64>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map(Some)
            .map_err(|_| CgError::Engine(format!("column {name}: expected non-negative integer"))),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer|null, got {other:?}"
        ))),
    }
}

fn get_bool(row: &[Value], columns: &[String], name: &str) -> Result<bool> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Bool(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected boolean, got {other:?}"
        ))),
    }
}

fn get_f32(row: &[Value], columns: &[String], name: &str) -> Result<f32> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Float64(value)) => Ok(*value as f32),
        other => Err(CgError::Engine(format!(
            "column {name}: expected floating point score, got {other:?}"
        ))),
    }
}

fn get_optional_timestamp(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<DateTime<Utc>>> {
    match row.get(column_index(columns, name)?) {
        Some(Value::Timestamp(value)) => DateTime::from_timestamp_millis(*value)
            .map(Some)
            .ok_or_else(|| CgError::Engine(format!("column {name}: invalid timestamp"))),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp|null, got {other:?}"
        ))),
    }
}

fn get_txid(row: &[Value], columns: &[String], name: &str) -> Result<TxId> {
    match row.get(column_index(columns, name)?) {
        Some(Value::TxId(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid, got {other:?}"
        ))),
    }
}
