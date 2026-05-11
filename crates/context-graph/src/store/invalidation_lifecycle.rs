use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;

pub struct ScopedStore<'s> {
    store: &'s Store,
    principal: Principal,
}

impl<'s> ScopedStore<'s> {
    pub(super) fn new(store: &'s Store, principal: Principal) -> Self {
        Self { store, principal }
    }

    pub fn principal(&self) -> &Principal {
        &self.principal
    }

    pub fn get_invalidation(&self, id: InvalidationId) -> Result<Invalidation> {
        let _ = self.store.db_path();
        let _ = id;
        Err(CgError::NotImplemented {
            symbol: "ScopedStore::get_invalidation",
        })
    }

    pub fn acknowledge_invalidation(&self, id: InvalidationId) -> Result<Invalidation> {
        let _ = self.store.db_path();
        let _ = id;
        Err(CgError::NotImplemented {
            symbol: "ScopedStore::acknowledge_invalidation",
        })
    }

    pub fn resolve_invalidation(
        &self,
        id: InvalidationId,
        _action: ResolveAction,
        _resolution_decision_id: Option<DecisionId>,
    ) -> Result<Invalidation> {
        let _ = self.store.db_path();
        let _ = id;
        Err(CgError::NotImplemented {
            symbol: "ScopedStore::resolve_invalidation",
        })
    }

    pub fn dismiss_invalidation(
        &self,
        id: InvalidationId,
        _reason: Option<String>,
    ) -> Result<Invalidation> {
        let _ = self.store.db_path();
        let _ = id;
        Err(CgError::NotImplemented {
            symbol: "ScopedStore::dismiss_invalidation",
        })
    }
}
