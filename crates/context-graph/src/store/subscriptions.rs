use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::{
    ConflictDetectedPayload, ContextId, InvalidationDetectedPayload, NewDecisionPayload,
    PrecedentMatchPayload, SubscriptionError, SubscriptionFilter, SubscriptionId,
    SubscriptionMetrics,
};

use super::ScopedStore;

type SubscriptionResult<T> = std::result::Result<T, SubscriptionError>;

static NEXT_SUBSCRIPTION_ID: AtomicU64 = AtomicU64::new(1);

impl ScopedStore<'_> {
    pub fn subscribe_new_decision<F>(
        &self,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&NewDecisionPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.stub_subscription("new_decision", &filter)
    }

    pub fn resume_new_decision<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&NewDecisionPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.ensure_contexts_authorized(&filter)?;
        Ok(subscription_id)
    }

    pub fn subscribe_invalidation_detected<F>(
        &self,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&InvalidationDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.stub_subscription("invalidation_detected", &filter)
    }

    pub fn resume_invalidation_detected<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&InvalidationDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.ensure_contexts_authorized(&filter)?;
        Ok(subscription_id)
    }

    pub fn subscribe_conflict_detected<F>(
        &self,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&ConflictDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.stub_subscription("conflict_detected", &filter)
    }

    pub fn resume_conflict_detected<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&ConflictDetectedPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.ensure_contexts_authorized(&filter)?;
        Ok(subscription_id)
    }

    pub fn subscribe_precedent_match<F>(
        &self,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&PrecedentMatchPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.stub_subscription("precedent_match", &filter)
    }

    pub fn resume_precedent_match<F>(
        &self,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        _handler: F,
    ) -> SubscriptionResult<SubscriptionId>
    where
        F: Fn(&PrecedentMatchPayload) -> SubscriptionResult<()> + Send + Sync + 'static,
    {
        self.ensure_contexts_authorized(&filter)?;
        Ok(subscription_id)
    }

    pub fn unsubscribe(&self, _subscription_id: SubscriptionId) -> SubscriptionResult<()> {
        Ok(())
    }

    pub fn subscription_metrics(
        &self,
        _subscription_id: &SubscriptionId,
    ) -> SubscriptionResult<SubscriptionMetrics> {
        Ok(SubscriptionMetrics::default())
    }

    fn stub_subscription(
        &self,
        event_name: &str,
        filter: &SubscriptionFilter,
    ) -> SubscriptionResult<SubscriptionId> {
        self.ensure_contexts_authorized(filter)?;
        let id = NEXT_SUBSCRIPTION_ID.fetch_add(1, Ordering::SeqCst);
        Ok(SubscriptionId::new(
            format!("cg_{event_name}_sink_{id}"),
            format!("cg_{event_name}_route_{id}"),
        ))
    }

    fn ensure_contexts_authorized(&self, filter: &SubscriptionFilter) -> SubscriptionResult<()> {
        let Some(requested) = &filter.context_ids else {
            return Ok(());
        };
        let Some(authorized) = &self.store.authorized_contexts else {
            return Ok(());
        };
        for context_id in requested {
            if !authorized.contains(context_id) {
                return Err(SubscriptionError::ContextNotAuthorized(ContextId::from(
                    context_id.as_uuid(),
                )));
            }
        }
        Ok(())
    }
}
