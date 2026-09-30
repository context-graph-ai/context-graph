#[cfg(context_graph_internal_test_support)]
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
#[cfg(context_graph_internal_test_support)]
use std::sync::Arc;

use contextdb_core::Value;
use contextdb_engine::ReadSession;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::read_session::ReadKernelTestObserver;
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::{
    CorrelationResolution, EventAdapterDeclaration, EventCorrelationResolver,
    EventCorrelationScope, EventDraft, EventRenderRole, PriorRequestAnchorQuery,
};

use super::Store;
use super::convert::map_engine_err;
use super::event::{EventAnchorCorrelationScopeHashInput, event_anchor_correlation_scope_hash};

const COMMITTED_INTERACTION_QUERY: &str = concat!(
    "SELECT event_id, context_id, connector_id, source_instance_ref, session_ref, ",
    "declaration_version, render_role, interaction_ref, key, original_value ",
    "FROM event_anchors WHERE correlation_scope_hash = $scope_hash ",
    "AND context_id = $context_id",
);

const COMMITTED_PROJECTED_QUERY: &str = concat!(
    "SELECT event_id, context_id, connector_id, source_instance_ref, session_ref, ",
    "declaration_version, render_role, interaction_ref, key, original_value ",
    "FROM event_anchors WHERE correlation_scope_hash = $scope_hash ",
    "AND context_id = $context_id AND event_id = $event_id",
);

const COMMITTED_ANCHOR_COLUMNS: &[&str] = &[
    "event_id",
    "context_id",
    "connector_id",
    "source_instance_ref",
    "session_ref",
    "declaration_version",
    "render_role",
    "interaction_ref",
    "key",
    "original_value",
];

/// Observation emitted only while an internal test watches the public Store resolver.
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub trait EventCorrelationTestObserver: Send + Sync {
    /// A page returned by the resolver's actual bounded cursor.
    fn cursor_page(&self, _is_continuation: bool, _rows: usize, _has_more: bool) {}

    /// The resolver's actual distinct-value accumulator after one candidate is considered.
    fn candidate_state(
        &self,
        _candidate: &JsonValue,
        _distinct_values: usize,
        _retained_json_bytes: usize,
    ) {
    }
}

#[cfg(context_graph_internal_test_support)]
#[derive(Clone)]
struct EventCorrelationTestObservers {
    kernel: Arc<dyn ReadKernelTestObserver>,
    correlation: Arc<dyn EventCorrelationTestObserver>,
}

#[cfg(context_graph_internal_test_support)]
thread_local! {
    static TEST_OBSERVERS: RefCell<Option<EventCorrelationTestObservers>> = const { RefCell::new(None) };
}

/// Attaches observations to the existing public [`Store::event_correlation_resolver`] on this
/// test thread. The resolver and SQL route remain the production implementation.
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub fn observe_public_store_resolver_for_test(
    kernel: Arc<dyn ReadKernelTestObserver>,
    correlation: Arc<dyn EventCorrelationTestObserver>,
) -> EventCorrelationObservationGuard {
    let previous = TEST_OBSERVERS.with(|slot| {
        slot.replace(Some(EventCorrelationTestObservers {
            kernel,
            correlation,
        }))
    });
    EventCorrelationObservationGuard { previous }
}

/// Restores the test thread's prior resolver observation after a proof finishes.
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub struct EventCorrelationObservationGuard {
    previous: Option<EventCorrelationTestObservers>,
}

#[cfg(context_graph_internal_test_support)]
impl Drop for EventCorrelationObservationGuard {
    fn drop(&mut self) {
        TEST_OBSERVERS.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

impl Store {
    /// Bind one adapter read to its declared source scope and current bounded batch. Import is the
    /// only production caller; the method is public only to the internal-support proofs.
    #[cfg(context_graph_internal_test_support)]
    #[doc(hidden)]
    pub fn event_correlation_resolver<'a>(
        &'a self,
        scope: EventCorrelationScope,
        declaration: &'a EventAdapterDeclaration,
        pending: &'a [EventDraft],
    ) -> impl EventCorrelationResolver + 'a {
        self.bind_event_correlation_resolver(scope, declaration, pending)
    }

    /// Bind one adapter read to its declared source scope and current bounded batch.
    #[cfg(not(context_graph_internal_test_support))]
    pub(crate) fn event_correlation_resolver<'a>(
        &'a self,
        scope: EventCorrelationScope,
        declaration: &'a EventAdapterDeclaration,
        pending: &'a [EventDraft],
    ) -> impl EventCorrelationResolver + 'a {
        self.bind_event_correlation_resolver(scope, declaration, pending)
    }

    fn bind_event_correlation_resolver<'a>(
        &'a self,
        scope: EventCorrelationScope,
        declaration: &'a EventAdapterDeclaration,
        pending: &'a [EventDraft],
    ) -> impl EventCorrelationResolver + 'a {
        #[cfg(context_graph_internal_test_support)]
        let test_observers = TEST_OBSERVERS.with(|slot| slot.borrow().clone());
        StoreEventCorrelationResolver {
            store: self,
            scope,
            declaration,
            pending,
            #[cfg(context_graph_internal_test_support)]
            test_observers,
        }
    }
}

struct StoreEventCorrelationResolver<'a> {
    store: &'a Store,
    scope: EventCorrelationScope,
    declaration: &'a EventAdapterDeclaration,
    pending: &'a [EventDraft],
    #[cfg(context_graph_internal_test_support)]
    test_observers: Option<EventCorrelationTestObservers>,
}

impl EventCorrelationResolver for StoreEventCorrelationResolver<'_> {
    fn resolve_prior_request_anchor(
        &mut self,
        query: &PriorRequestAnchorQuery,
    ) -> Result<CorrelationResolution> {
        self.validate_binding(query)?;

        let mut candidates = DistinctCandidates::default();
        for draft in self.pending {
            if draft.session_ref != self.scope.session_ref
                || draft.render_role != EventRenderRole::ToolRequest
                || draft.interaction_ref.as_deref() != Some(query.interaction_ref.as_str())
            {
                continue;
            }
            let Some(anchors) = draft.anchors.as_object() else {
                return Err(self.malformed("pending event anchors must be a JSON object"));
            };
            let Some(interaction_anchor) = anchors.get(&query.interaction_anchor_key) else {
                continue;
            };
            if !anchor_contains_interaction_ref(interaction_anchor, &query.interaction_ref) {
                continue;
            }
            let Some(projected_anchor) = anchors.get(&query.projected_anchor_key) else {
                continue;
            };
            let projected = visit_anchor_values(projected_anchor, |candidate| {
                require_scalar(candidate, &query.projected_anchor_key)?;
                candidates.admit(candidate.clone());
                #[cfg(context_graph_internal_test_support)]
                self.observe_candidate_state(candidate, &candidates);
                Ok(if candidates.is_ambiguous() {
                    CursorVisit::Stop
                } else {
                    CursorVisit::Continue
                })
            })?;
            if projected == CursorVisit::Stop {
                return Ok(CorrelationResolution::Ambiguous);
            }
        }

        let mut limits = self.store.read_limits();
        limits.cursor_page_rows = 1;
        #[cfg(context_graph_internal_test_support)]
        let session = match self.test_observers.as_ref() {
            Some(observers) => self
                .store
                .internal_db()
                .read_session_with_kernel_observer_for_test(limits, observers.kernel.clone())
                .map_err(map_engine_err)?,
            None => self
                .store
                .internal_db()
                .read_session(limits)
                .map_err(map_engine_err)?,
        };
        #[cfg(not(context_graph_internal_test_support))]
        let session = self
            .store
            .internal_db()
            .read_session(limits)
            .map_err(map_engine_err)?;
        let interaction_params =
            scoped_anchor_params(&self.scope, query, query.interaction_anchor_key.as_str());
        visit_cursor_rows(
            &session,
            COMMITTED_INTERACTION_QUERY,
            &interaction_params,
            COMMITTED_ANCHOR_COLUMNS,
            #[cfg(context_graph_internal_test_support)]
            self.test_observers.as_ref(),
            |row| {
                let Some((event_id, interaction_anchor)) = scoped_anchor_row(
                    row,
                    &self.scope,
                    &query.interaction_ref,
                    &query.interaction_anchor_key,
                    None,
                )?
                else {
                    return Ok(CursorVisit::Continue);
                };
                if !interaction_anchor_matches(interaction_anchor, &query.interaction_ref) {
                    return Ok(CursorVisit::Continue);
                }

                let mut projected_params =
                    scoped_anchor_params(&self.scope, query, query.projected_anchor_key.as_str());
                projected_params.insert("event_id".into(), Value::Uuid(event_id));
                let projected = visit_cursor_rows(
                    &session,
                    COMMITTED_PROJECTED_QUERY,
                    &projected_params,
                    COMMITTED_ANCHOR_COLUMNS,
                    #[cfg(context_graph_internal_test_support)]
                    self.test_observers.as_ref(),
                    |row| {
                        let Some((_, candidate)) = scoped_anchor_row(
                            row,
                            &self.scope,
                            &query.interaction_ref,
                            &query.projected_anchor_key,
                            Some(event_id),
                        )?
                        else {
                            return Ok(CursorVisit::Continue);
                        };
                        require_scalar(candidate, &query.projected_anchor_key)?;
                        candidates.admit(candidate.clone());
                        #[cfg(context_graph_internal_test_support)]
                        self.observe_candidate_state(candidate, &candidates);
                        Ok(if candidates.is_ambiguous() {
                            CursorVisit::Stop
                        } else {
                            CursorVisit::Continue
                        })
                    },
                )?;
                Ok(projected)
            },
        )?;

        Ok(candidates.resolve())
    }
}

fn scoped_anchor_params(
    scope: &EventCorrelationScope,
    query: &PriorRequestAnchorQuery,
    anchor_key: &str,
) -> HashMap<String, Value> {
    HashMap::from([
        ("context_id".into(), Value::Uuid(scope.context_id.as_uuid())),
        (
            "scope_hash".into(),
            Value::Text(event_anchor_correlation_scope_hash(
                EventAnchorCorrelationScopeHashInput {
                    context_id: scope.context_id,
                    connector_id: &scope.connector_id,
                    source_instance_ref: &scope.source_instance_ref,
                    session_ref: &scope.session_ref,
                    declaration_version: &scope.anchor_declaration_version,
                    render_role: EventRenderRole::ToolRequest,
                    interaction_ref: Some(&query.interaction_ref),
                    key: anchor_key,
                },
            )),
        ),
    ])
}

fn scoped_anchor_row<'a>(
    row: &'a [Value],
    scope: &EventCorrelationScope,
    interaction_ref: &str,
    anchor_key: &str,
    expected_event_id: Option<uuid::Uuid>,
) -> Result<Option<(uuid::Uuid, &'a JsonValue)>> {
    let [
        Value::Uuid(event_id),
        Value::Uuid(context_id),
        Value::Text(connector_id),
        Value::Text(source_instance_ref),
        Value::Text(session_ref),
        Value::Text(declaration_version),
        Value::Text(render_role),
        persisted_interaction_ref,
        Value::Text(key),
        Value::Json(original_value),
    ] = row
    else {
        return Err(CgError::Engine(format!(
            "event correlation expected complete anchor metadata and one JSON value, got {row:?}"
        )));
    };
    let persisted_interaction_ref = match persisted_interaction_ref {
        Value::Text(value) => Some(value.as_str()),
        Value::Null => None,
        other => {
            return Err(CgError::Engine(format!(
                "event correlation expected text or null interaction metadata, got {other:?}"
            )));
        }
    };
    let matches_scope = *context_id == scope.context_id.as_uuid()
        && connector_id == &scope.connector_id
        && source_instance_ref == &scope.source_instance_ref
        && session_ref == &scope.session_ref
        && declaration_version == &scope.anchor_declaration_version
        && render_role == "tool_request"
        && persisted_interaction_ref == Some(interaction_ref)
        && key == anchor_key
        && expected_event_id.is_none_or(|expected| expected == *event_id);
    Ok(matches_scope.then_some((*event_id, original_value)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorVisit {
    Continue,
    Stop,
}

fn visit_cursor_rows(
    session: &ReadSession,
    sql: &str,
    params: &HashMap<String, Value>,
    expected_columns: &[&str],
    #[cfg(context_graph_internal_test_support)] test_observers: Option<
        &EventCorrelationTestObservers,
    >,
    mut visit: impl FnMut(&[Value]) -> Result<CursorVisit>,
) -> Result<CursorVisit> {
    let mut cursor = session.open_cursor(sql, params).map_err(map_engine_err)?;
    let visited = (|| -> Result<CursorVisit> {
        let (mut has_more, first_result) = {
            let page = cursor.first_page();
            (
                page.has_more,
                visit_cursor_page(
                    &page.columns,
                    &page.rows,
                    expected_columns,
                    false,
                    page.has_more,
                    #[cfg(context_graph_internal_test_support)]
                    test_observers,
                    &mut visit,
                )?,
            )
        };
        if first_result == CursorVisit::Stop {
            return Ok(CursorVisit::Stop);
        }
        while has_more {
            let page = cursor
                .fetch(Some(NonZeroUsize::MIN))
                .map_err(map_engine_err)?;
            has_more = page.has_more;
            if visit_cursor_page(
                &page.columns,
                &page.rows,
                expected_columns,
                true,
                page.has_more,
                #[cfg(context_graph_internal_test_support)]
                test_observers,
                &mut visit,
            )? == CursorVisit::Stop
            {
                return Ok(CursorVisit::Stop);
            }
        }
        Ok(CursorVisit::Continue)
    })();
    let closed = cursor.close().map_err(map_engine_err);
    let visited = visited?;
    closed?;
    Ok(visited)
}

fn visit_cursor_page(
    columns: &[String],
    rows: &[Vec<Value>],
    expected_columns: &[&str],
    _is_continuation: bool,
    _has_more: bool,
    #[cfg(context_graph_internal_test_support)] test_observers: Option<
        &EventCorrelationTestObservers,
    >,
    visit: &mut impl FnMut(&[Value]) -> Result<CursorVisit>,
) -> Result<CursorVisit> {
    if rows.len() > 1 {
        return Err(CgError::Engine(
            "event correlation cursor returned more than its one-row page".into(),
        ));
    }
    if !columns.is_empty()
        && !columns
            .iter()
            .map(String::as_str)
            .eq(expected_columns.iter().copied())
    {
        return Err(CgError::Engine(
            "event correlation query returned an unexpected projection".into(),
        ));
    }
    #[cfg(context_graph_internal_test_support)]
    if let Some(observers) = test_observers {
        observers
            .correlation
            .cursor_page(_is_continuation, rows.len(), _has_more);
    }
    match rows.first() {
        Some(row) => visit(row),
        None => Ok(CursorVisit::Continue),
    }
}

impl StoreEventCorrelationResolver<'_> {
    #[cfg(context_graph_internal_test_support)]
    fn observe_candidate_state(&self, candidate: &JsonValue, candidates: &DistinctCandidates) {
        if let Some(observers) = self.test_observers.as_ref() {
            observers.correlation.candidate_state(
                candidate,
                candidates.distinct_len(),
                candidates.retained_json_bytes(),
            );
        }
    }

    fn validate_binding(&self, query: &PriorRequestAnchorQuery) -> Result<()> {
        if self.scope.connector_id != self.declaration.connector_id {
            return Err(self.malformed(
                "correlation scope connector differs from the bound adapter declaration",
            ));
        }
        if self.scope.anchor_declaration_version != self.declaration.anchor_declaration_version {
            return Err(self
                .malformed("correlation scope version differs from the bound anchor declaration"));
        }
        let properties = self
            .declaration
            .anchor_schema
            .get("properties")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                self.malformed("correlation requires explicitly declared anchor properties")
            })?;
        for key in [
            query.interaction_anchor_key.as_str(),
            query.projected_anchor_key.as_str(),
        ] {
            if !properties.contains_key(key) {
                return Err(
                    self.malformed(format!("correlation anchor key `{key}` is not declared"))
                );
            }
        }
        Ok(())
    }

    fn malformed(&self, reason: impl Into<String>) -> CgError {
        CgError::SourceRecordMalformed {
            connector_id: self.scope.connector_id.clone(),
            session_ref: self.scope.session_ref.clone(),
            position: JsonValue::Null,
            reason: reason.into(),
        }
    }
}

fn anchor_contains_interaction_ref(anchor: &JsonValue, interaction_ref: &str) -> bool {
    let expected = JsonValue::String(interaction_ref.to_owned());
    match anchor {
        JsonValue::Array(values) => values.iter().any(|value| value == &expected),
        value => value == &expected,
    }
}

fn interaction_anchor_matches(anchor: &JsonValue, interaction_ref: &str) -> bool {
    anchor == &JsonValue::String(interaction_ref.to_owned())
}

fn visit_anchor_values(
    anchor: &JsonValue,
    mut visit: impl FnMut(&JsonValue) -> Result<CursorVisit>,
) -> Result<CursorVisit> {
    match anchor {
        JsonValue::Array(values) => {
            for value in values {
                if visit(value)? == CursorVisit::Stop {
                    return Ok(CursorVisit::Stop);
                }
            }
            Ok(CursorVisit::Continue)
        }
        value => visit(value),
    }
}

fn require_scalar(value: &JsonValue, key: &str) -> Result<()> {
    if value.is_array() || value.is_object() {
        return Err(CgError::SourceRecordMalformed {
            connector_id: String::new(),
            session_ref: String::new(),
            position: JsonValue::Null,
            reason: format!("correlation anchor `{key}` must project scalar values"),
        });
    }
    Ok(())
}

#[derive(Default)]
struct DistinctCandidates {
    first: Option<JsonValue>,
    second: Option<JsonValue>,
}

impl DistinctCandidates {
    fn admit(&mut self, candidate: JsonValue) {
        if self.first.as_ref() == Some(&candidate) || self.second.as_ref() == Some(&candidate) {
            return;
        }
        if self.first.is_none() {
            self.first = Some(candidate);
        } else if self.second.is_none() {
            self.second = Some(candidate);
        }
    }

    fn is_ambiguous(&self) -> bool {
        self.second.is_some()
    }

    #[cfg(context_graph_internal_test_support)]
    fn distinct_len(&self) -> usize {
        usize::from(self.first.is_some()) + usize::from(self.second.is_some())
    }

    #[cfg(context_graph_internal_test_support)]
    fn retained_json_bytes(&self) -> usize {
        [&self.first, &self.second]
            .into_iter()
            .flatten()
            .map(|value| {
                serde_json::to_string(value)
                    .expect("JSON value serializes")
                    .len()
            })
            .sum()
    }

    fn resolve(self) -> CorrelationResolution {
        match (self.first, self.second) {
            (None, None) => CorrelationResolution::Missing,
            (Some(candidate), None) => CorrelationResolution::Unique(candidate),
            (Some(_), Some(_)) => CorrelationResolution::Ambiguous,
            (None, Some(_)) => unreachable!("the second candidate is filled only after the first"),
        }
    }
}
