use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Weak};

use contextdb_core::{Lsn, Value};
use contextdb_engine::sync_types::DdlChange;
use contextdb_engine::{Database, SinkError, SinkEvent};
use uuid::Uuid;

use crate::error::CgError;
use crate::types::{
    ConflictDetectedPayload, ContextId, DecisionId, EntityId, IntentionId,
    InvalidationDetectedPayload, InvalidationId, NewDecisionPayload, ObservationId,
    PrecedentMatchPayload, SubscriptionError, SubscriptionFilter, SubscriptionId,
    SubscriptionMetrics,
};

use super::convert::map_engine_err;
use super::encode::intention_ids_from_json;
use super::{ScopedStore, Store};

type SubscriptionResult<T> = std::result::Result<T, SubscriptionError>;

impl ScopedStore<'_> {
    pub fn subscribe_new_decision<F>(
        &self,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&NewDecisionPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "new_decision",
            None,
            filter,
            handler,
            build_new_decision_payload,
        )
    }

    pub fn resume_new_decision<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&NewDecisionPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "new_decision",
            Some(subscription_id),
            filter,
            handler,
            build_new_decision_payload,
        )
    }

    pub fn subscribe_invalidation_detected<F>(
        &self,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&InvalidationDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "invalidation_detected",
            None,
            filter,
            handler,
            build_invalidation_detected_payload,
        )
    }

    pub fn resume_invalidation_detected<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&InvalidationDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "invalidation_detected",
            Some(subscription_id),
            filter,
            handler,
            build_invalidation_detected_payload,
        )
    }

    pub fn subscribe_conflict_detected<F>(
        &self,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&ConflictDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "conflict_detected",
            None,
            filter,
            handler,
            build_conflict_detected_payload,
        )
    }

    pub fn resume_conflict_detected<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&ConflictDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "conflict_detected",
            Some(subscription_id),
            filter,
            handler,
            build_conflict_detected_payload,
        )
    }

    pub fn subscribe_precedent_match<F>(
        &self,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&PrecedentMatchPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "precedent_match",
            None,
            filter,
            handler,
            build_precedent_match_payload,
        )
    }

    pub fn resume_precedent_match<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&PrecedentMatchPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.register_subscription(
            "precedent_match",
            Some(subscription_id),
            filter,
            handler,
            build_precedent_match_payload,
        )
    }

    pub fn unsubscribe(&self, subscription_id: SubscriptionId) -> SubscriptionResult<()> {
        let ddl_db = self.store.resource_owner.as_ref().unwrap_or(&self.store.db);
        if !route_is_active(ddl_db, subscription_id.route_name()) {
            return Err(SubscriptionError::UnknownSubscription(subscription_id));
        }
        ddl_db
            .execute(
                &format!("DROP ROUTE {}", subscription_id.route_name()),
                &HashMap::new(),
            )
            .map_err(subscription_engine_err)?;
        Ok(())
    }

    pub fn subscription_metrics(
        &self,
        subscription_id: &SubscriptionId,
    ) -> SubscriptionResult<SubscriptionMetrics> {
        let ddl_db = self.store.resource_owner.as_ref().unwrap_or(&self.store.db);
        if !sink_is_declared(ddl_db, subscription_id.sink_name()) {
            return Err(SubscriptionError::UnknownSubscription(
                subscription_id.clone(),
            ));
        }
        let metrics = self
            .store
            .db
            .sink_metrics_for_test(subscription_id.sink_name());
        Ok(SubscriptionMetrics {
            delivered: metrics.delivered,
            queued: metrics.queued,
            retried: metrics.retried,
            permanent_failures: metrics.permanent_failures,
        })
    }

    fn register_subscription<P, F, B>(
        &self,
        event_name: &'static str,
        existing_subscription_id: Option<SubscriptionId>,
        filter: SubscriptionFilter,
        handler: F,
        build_payload: B,
    ) -> SubscriptionResult<SubscriptionId>
    where
        P: SubscriptionPayload,
        F: Fn(&P) -> SubscriptionResult<()> + Send + Sync + 'static,
        B: Fn(&crate::store::Store, &SinkEvent) -> SubscriptionResult<P> + Send + Sync + 'static,
    {
        let effective_contexts = self.effective_contexts(&filter)?;
        let ddl_db = self.store.resource_owner.as_ref().unwrap_or(&self.store.db);
        let subscription_id = match existing_subscription_id {
            Some(subscription_id) => {
                ensure_subscription_can_resume(ddl_db, event_name, &subscription_id)?;
                subscription_id
            }
            None => {
                let id = Uuid::now_v7();
                let suffix = id.simple().to_string();
                let sink_name = format!("cg_subscription_{event_name}_{suffix}");
                let route_name = format!("cg_subscription_{event_name}_route_{suffix}");
                ddl_db
                    .execute(
                        &format!("CREATE SINK {sink_name} TYPE callback"),
                        &HashMap::new(),
                    )
                    .map_err(subscription_engine_err)?;
                ddl_db
                    .execute(
                        &format!("CREATE ROUTE {route_name} EVENT {event_name} TO {sink_name}"),
                        &HashMap::new(),
                    )
                    .map_err(subscription_engine_err)?;
                SubscriptionId::new(sink_name, route_name)
            }
        };

        let handler = Arc::new(handler);
        let callback_store = CallbackStoreParts::from_store(self.store);
        let filter_for_callback = filter.clone();
        let deliver = move |event: &SinkEvent| {
            let Some(store) = callback_store.to_store() else {
                return Err(SinkError::Transient(
                    "context graph store is closed".to_string(),
                ));
            };
            let payload = build_payload(&store, event)
                .map_err(|err| SinkError::Permanent(err.to_string()))?;
            if !payload
                .matches_filter(&store, &filter_for_callback)
                .map_err(|err| SinkError::Permanent(err.to_string()))?
            {
                return Ok(());
            }
            handler(&payload).map_err(subscription_error_to_sink_error)
        };
        match &effective_contexts {
            Some(contexts) if Some(contexts) != self.store.authorized_contexts.as_ref() => {
                let scoped = scoped_database_for_contexts(&self.store.db, contexts);
                scoped
                    .register_sink(subscription_id.sink_name(), None, deliver)
                    .map_err(subscription_engine_err)?;
            }
            _ => {
                self.store
                    .db
                    .register_sink(subscription_id.sink_name(), None, deliver)
                    .map_err(subscription_engine_err)?;
            }
        };
        Ok(subscription_id)
    }

    fn effective_contexts(
        &self,
        filter: &SubscriptionFilter,
    ) -> SubscriptionResult<Option<BTreeSet<ContextId>>> {
        let requested = filter
            .context_ids
            .as_ref()
            .map(|contexts| contexts.iter().copied().collect::<BTreeSet<_>>());
        if let (Some(authorized), Some(requested)) =
            (&self.store.authorized_contexts, requested.as_ref())
            && let Some(offending) = requested
                .iter()
                .find(|context_id| !authorized.contains(context_id))
                .copied()
        {
            return Err(SubscriptionError::ContextNotAuthorized(offending));
        }
        Ok(requested.or_else(|| self.store.authorized_contexts.clone()))
    }
}

#[derive(Clone)]
struct CallbackStoreParts {
    db: Weak<Database>,
    build: Arc<dyn Fn(Arc<Database>) -> Store + Send + Sync>,
}

impl CallbackStoreParts {
    fn from_store(store: &Store) -> Self {
        let db_path = store.db_path.clone();
        let https_allowed_hosts = store.https_allowed_hosts.clone();
        let file_uri_allowed = store.file_uri_allowed;
        let trace = Arc::clone(&store.trace);
        let outcome_mutex = Arc::clone(&store.outcome_mutex);
        let authorized_contexts = store.authorized_contexts.clone();
        let build = Arc::new(move |db| Store {
            db_path: db_path.clone(),
            db,
            resource_owner: None,
            https_allowed_hosts: https_allowed_hosts.clone(),
            file_uri_allowed,
            trace: Arc::clone(&trace),
            outcome_mutex: Arc::clone(&outcome_mutex),
            authorized_contexts: authorized_contexts.clone(),
        });
        Self {
            db: Arc::downgrade(&store.db),
            build,
        }
    }

    fn to_store(&self) -> Option<Store> {
        Some((self.build)(self.db.upgrade()?))
    }
}

trait SubscriptionPayload {
    fn decision_ids(&self) -> Vec<DecisionId>;
    fn intention_ids(&self) -> Vec<IntentionId>;
    fn entity_ids(&self, store: &crate::store::Store) -> SubscriptionResult<Vec<EntityId>>;

    fn matches_filter(
        &self,
        store: &crate::store::Store,
        filter: &SubscriptionFilter,
    ) -> SubscriptionResult<bool> {
        if let Some(decision_id) = filter.decision_id
            && !self.decision_ids().contains(&decision_id)
        {
            return Ok(false);
        }
        if let Some(intention_id) = filter.intention_id
            && !self.intention_ids().contains(&intention_id)
        {
            return Ok(false);
        }
        if let Some(entity_id) = filter.entity_id
            && !self.entity_ids(store)?.contains(&entity_id)
        {
            return Ok(false);
        }
        Ok(true)
    }
}

impl SubscriptionPayload for NewDecisionPayload {
    fn decision_ids(&self) -> Vec<DecisionId> {
        vec![self.decision.id]
    }

    fn intention_ids(&self) -> Vec<IntentionId> {
        self.served_intention_ids.clone()
    }

    fn entity_ids(&self, store: &crate::store::Store) -> SubscriptionResult<Vec<EntityId>> {
        based_on_entity_ids(store, self.decision.id)
    }
}

impl SubscriptionPayload for InvalidationDetectedPayload {
    fn decision_ids(&self) -> Vec<DecisionId> {
        vec![self.invalidation.affected_decision_id]
    }

    fn intention_ids(&self) -> Vec<IntentionId> {
        self.invalidation.threatened_intention_ids.clone()
    }

    fn entity_ids(&self, store: &crate::store::Store) -> SubscriptionResult<Vec<EntityId>> {
        Ok(vec![observation_entity_id(
            store,
            self.invalidation.trigger_observation_id,
        )?])
    }
}

impl SubscriptionPayload for ConflictDetectedPayload {
    fn decision_ids(&self) -> Vec<DecisionId> {
        vec![self.older_decision_id, self.newer_decision_id]
    }

    fn intention_ids(&self) -> Vec<IntentionId> {
        self.threatened_intention_ids.clone()
    }

    fn entity_ids(&self, store: &crate::store::Store) -> SubscriptionResult<Vec<EntityId>> {
        let mut ids = based_on_entity_ids(store, self.older_decision_id)?;
        ids.extend(based_on_entity_ids(store, self.newer_decision_id)?);
        ids.sort();
        ids.dedup();
        Ok(ids)
    }
}

impl SubscriptionPayload for PrecedentMatchPayload {
    fn decision_ids(&self) -> Vec<DecisionId> {
        vec![self.new_decision_id, self.precedent_decision_id]
    }

    fn intention_ids(&self) -> Vec<IntentionId> {
        self.served_intention_ids.clone()
    }

    fn entity_ids(&self, store: &crate::store::Store) -> SubscriptionResult<Vec<EntityId>> {
        based_on_entity_ids(store, self.new_decision_id)
    }
}

fn scoped_database_for_contexts(db: &Arc<Database>, contexts: &BTreeSet<ContextId>) -> Database {
    db.scoped_with_contexts(
        contexts
            .iter()
            .map(|ctx| contextdb_core::types::ContextId::new(ctx.as_uuid()))
            .collect(),
    )
}

fn route_is_active(db: &Database, route_name: &str) -> bool {
    let mut active = false;
    for ddl in db.changes_since(Lsn(0)).ddl {
        match ddl {
            DdlChange::CreateRoute { name, .. } if name == route_name => active = true,
            DdlChange::DropRoute { name, .. } if name == route_name => active = false,
            _ => {}
        }
    }
    active
}

fn sink_is_declared(db: &Database, sink_name: &str) -> bool {
    db.changes_since(Lsn(0)).ddl.iter().any(|ddl| {
        matches!(
            ddl,
            DdlChange::CreateSink { name, .. } if name == sink_name
        )
    })
}

fn ensure_subscription_can_resume(
    db: &Database,
    event_name: &str,
    subscription_id: &SubscriptionId,
) -> SubscriptionResult<()> {
    if !sink_is_declared(db, subscription_id.sink_name()) {
        return Err(SubscriptionError::SinkDdlMissing(format!(
            "sink not found: {}",
            subscription_id.sink_name()
        )));
    }
    let mut active = false;
    for ddl in db.changes_since(Lsn(0)).ddl {
        match ddl {
            DdlChange::CreateRoute {
                name,
                event_type,
                sink,
                ..
            } if name == subscription_id.route_name()
                && event_type == event_name
                && sink == subscription_id.sink_name() =>
            {
                active = true;
            }
            DdlChange::CreateRoute { name, .. } if name == subscription_id.route_name() => {
                active = false;
            }
            DdlChange::DropRoute { name, .. } if name == subscription_id.route_name() => {
                active = false;
            }
            _ => {}
        }
    }
    if active {
        Ok(())
    } else {
        Err(SubscriptionError::UnknownSubscription(
            subscription_id.clone(),
        ))
    }
}

fn build_new_decision_payload(
    store: &crate::store::Store,
    event: &SinkEvent,
) -> SubscriptionResult<NewDecisionPayload> {
    let decision_id = event_uuid(event, "id").map(DecisionId::from)?;
    let decision = store
        .get_decision(decision_id)
        .map_err(subscription_cg_err)?
        .ok_or_else(|| SubscriptionError::Engine(format!("decision not found: {decision_id}")))?;
    let served_intention_ids = decision.intention_ids.clone();
    let served_intention_descriptions = intention_descriptions(store, &served_intention_ids)?;
    Ok(NewDecisionPayload {
        decision,
        served_intention_ids,
        served_intention_descriptions,
        at_lsn: event.at_lsn,
    })
}

fn build_invalidation_detected_payload(
    store: &crate::store::Store,
    event: &SinkEvent,
) -> SubscriptionResult<InvalidationDetectedPayload> {
    let invalidation_id = event_uuid(event, "id").map(InvalidationId::from)?;
    let invalidation = store
        .with_system_principal()
        .get_invalidation(invalidation_id)
        .map_err(subscription_cg_err)?;
    let decision = store
        .get_decision(invalidation.affected_decision_id)
        .map_err(subscription_cg_err)?
        .ok_or_else(|| {
            SubscriptionError::Engine(format!(
                "decision not found: {}",
                invalidation.affected_decision_id
            ))
        })?;
    let entity_id = observation_entity_id(store, invalidation.trigger_observation_id)?;
    let entity_name = store
        .get_entity(entity_id)
        .map_err(subscription_cg_err)?
        .ok_or_else(|| SubscriptionError::Engine(format!("entity not found: {entity_id}")))?
        .name;
    let threatened_intention_descriptions =
        intention_descriptions(store, &invalidation.threatened_intention_ids)?;
    Ok(InvalidationDetectedPayload {
        cascade_from: invalidation.cascade_from,
        invalidation,
        decision_description: decision.description,
        entity_name,
        threatened_intention_descriptions,
        at_lsn: event.at_lsn,
    })
}

fn build_conflict_detected_payload(
    store: &crate::store::Store,
    event: &SinkEvent,
) -> SubscriptionResult<ConflictDetectedPayload> {
    let context_id = event_uuid(event, "context_id").map(ContextId::from)?;
    let older_decision_id = event_uuid(event, "older_decision_id").map(DecisionId::from)?;
    let newer_decision_id = event_uuid(event, "newer_decision_id").map(DecisionId::from)?;
    let threatened_intention_ids = event_intention_ids(event, "threatened_intention_ids")?;
    let intention_in_tension = event_bool(event, "intention_in_tension")?;
    let threatened_intention_descriptions =
        intention_descriptions(store, &threatened_intention_ids)?;
    Ok(ConflictDetectedPayload {
        context_id,
        older_decision_id,
        newer_decision_id,
        threatened_intention_ids,
        intention_in_tension,
        threatened_intention_descriptions,
        at_lsn: event.at_lsn,
    })
}

fn build_precedent_match_payload(
    store: &crate::store::Store,
    event: &SinkEvent,
) -> SubscriptionResult<PrecedentMatchPayload> {
    let context_id = event_uuid(event, "context_id").map(ContextId::from)?;
    let new_decision_id = event_uuid(event, "new_decision_id").map(DecisionId::from)?;
    let precedent_decision_id = event_uuid(event, "precedent_decision_id").map(DecisionId::from)?;
    let served_intention_ids = event_intention_ids(event, "served_intention_ids")?;
    let served_intention_descriptions = intention_descriptions(store, &served_intention_ids)?;
    let matched_precedent_outcome_success = event_bool(event, "matched_precedent_outcome_success")?;
    Ok(PrecedentMatchPayload {
        context_id,
        new_decision_id,
        precedent_decision_id,
        served_intention_ids,
        served_intention_descriptions,
        matched_precedent_outcome_success,
        at_lsn: event.at_lsn,
    })
}

fn intention_descriptions(
    store: &crate::store::Store,
    intention_ids: &[IntentionId],
) -> SubscriptionResult<BTreeMap<IntentionId, String>> {
    let mut descriptions = BTreeMap::new();
    for intention_id in intention_ids {
        if let Some(intention) = store
            .get_intention(*intention_id)
            .map_err(subscription_cg_err)?
        {
            descriptions.insert(*intention_id, intention.description);
        } else {
            return Err(SubscriptionError::Engine(format!(
                "intention not found: {intention_id}"
            )));
        }
    }
    Ok(descriptions)
}

fn based_on_entity_ids(
    store: &crate::store::Store,
    decision_id: DecisionId,
) -> SubscriptionResult<Vec<EntityId>> {
    let result = store
        .db
        .execute(
            "SELECT target_id FROM edges WHERE source_id = $decision_id AND edge_type = $edge_type",
            &HashMap::from([
                (
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                ),
                ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
            ]),
        )
        .map_err(subscription_engine_err)?;
    let mut ids = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        match row.first() {
            Some(Value::Uuid(value)) => ids.push(EntityId::from(*value)),
            other => {
                return Err(SubscriptionError::Engine(format!(
                    "BASED_ON edge target expected uuid for decision {decision_id}, got {other:?}"
                )));
            }
        }
    }
    Ok(ids)
}

fn observation_entity_id(
    store: &crate::store::Store,
    observation_id: ObservationId,
) -> SubscriptionResult<EntityId> {
    let result = store
        .db
        .execute(
            "SELECT entity_id FROM observations WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(observation_id.as_uuid()))]),
        )
        .map_err(subscription_engine_err)?;
    let row = result.rows.first().ok_or_else(|| {
        SubscriptionError::Engine(format!("observation not found: {observation_id}"))
    })?;
    match row.first() {
        Some(Value::Uuid(value)) => Ok(EntityId::from(*value)),
        other => Err(SubscriptionError::Engine(format!(
            "observation entity_id expected uuid for {observation_id}, got {other:?}"
        ))),
    }
}

fn event_uuid(event: &SinkEvent, column: &str) -> SubscriptionResult<Uuid> {
    match event.row_values.get(column) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(SubscriptionError::Engine(format!(
            "event {} expected {column} uuid, got {other:?}",
            event.event_type
        ))),
    }
}

fn event_bool(event: &SinkEvent, column: &str) -> SubscriptionResult<bool> {
    match event.row_values.get(column) {
        Some(Value::Bool(value)) => Ok(*value),
        other => Err(SubscriptionError::Engine(format!(
            "event {} expected {column} bool, got {other:?}",
            event.event_type
        ))),
    }
}

fn event_intention_ids(event: &SinkEvent, column: &str) -> SubscriptionResult<Vec<IntentionId>> {
    match event.row_values.get(column) {
        Some(Value::Json(value)) => intention_ids_from_json(value).map_err(subscription_cg_err),
        other => Err(SubscriptionError::Engine(format!(
            "event {} expected {column} json, got {other:?}",
            event.event_type
        ))),
    }
}

fn subscription_error_to_sink_error(err: SubscriptionError) -> SinkError {
    match err {
        SubscriptionError::HandlerFailed(message) => SinkError::Permanent(message),
        other => SinkError::Permanent(other.to_string()),
    }
}

fn subscription_cg_err(err: CgError) -> SubscriptionError {
    SubscriptionError::Engine(err.to_string())
}

fn subscription_engine_err(err: contextdb_core::Error) -> SubscriptionError {
    match err {
        contextdb_core::Error::Other(message) if message.starts_with("sink not found") => {
            SubscriptionError::SinkDdlMissing(message)
        }
        other => SubscriptionError::Engine(map_engine_err(other).to_string()),
    }
}
