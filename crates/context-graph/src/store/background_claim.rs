//! The one claim mechanism every kind of background work uses; see [`crate::background_work`].
//!
//! A row in `background_claims` names one unit's owner token and epoch. The process that owns the
//! unit holds contextdb's holder-presence lock on that token until it publishes, releases or exits,
//! so "is the owner alive?" is a question the kernel answers, not a timestamp. This module is the
//! only code that takes that lock or writes a claim row; each kind of background work calls it with
//! its own [`ClaimKey`] inside its own transactions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use contextdb_core::Value;
use contextdb_engine::QueryResult;
use contextdb_engine::persistence::holder_presence::{self, Holder, HolderPresence};
use uuid::Uuid;

use super::convert::map_engine_err;
use super::encode::get_uuid;
use super::event_search_read::EventSearchRows;
use crate::{BackgroundWork, BackgroundWorkState, CgError, ContextId, Result, Store, TxId};

pub(crate) const CLAIMS_TABLE: &str = "background_claims";

const SELECT_CLAIM: &str = "SELECT owner_token, owner_epoch FROM background_claims WHERE work = $work AND context_id = $context_id AND unit_id = $unit_id";
const INSERT_CLAIM: &str = "INSERT INTO background_claims (work, context_id, unit_id, owner_token, owner_epoch) VALUES ($work, $context_id, $unit_id, $owner_token, $owner_epoch)";
const TAKE_OVER_CLAIM: &str = "UPDATE background_claims SET owner_token = $owner_token, owner_epoch = $owner_epoch WHERE work = $work AND context_id = $context_id AND unit_id = $unit_id";
const DELETE_OWN_CLAIM: &str = "DELETE FROM background_claims WHERE work = $work AND context_id = $context_id AND unit_id = $unit_id AND owner_token = $owner_token AND owner_epoch = $owner_epoch";
const DELETE_TOKEN_CLAIM: &str = "DELETE FROM background_claims WHERE work = $work AND context_id = $context_id AND unit_id = $unit_id AND owner_token = $owner_token";
const SELECT_CONTEXT_CLAIMS: &str = "SELECT unit_id, owner_token FROM background_claims WHERE work = $work AND context_id = $context_id";

/// Where status reads claim rows: a writable Store or a read-only consumer of the same store.
pub(crate) trait ClaimRows {
    fn claim_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult>;
    fn claim_store_path(&self) -> &Path;
}

impl<T: EventSearchRows + ?Sized> ClaimRows for T {
    fn claim_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        self.query_rows(sql, params)
    }

    fn claim_store_path(&self) -> &Path {
        self.store_path()
    }
}

/// One unit of background work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaimKey {
    pub(crate) work: BackgroundWork,
    pub(crate) context_id: ContextId,
    pub(crate) unit_id: Uuid,
}

/// The owner a claim row names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClaimRow {
    token: Uuid,
    epoch: i64,
}

/// The outcome of asking for a unit: this process now holds it, a live process already does, or
/// another runner changed the unit's claim while this transaction was reading it.
pub(crate) enum Taken {
    Held(HeldClaim),
    Live(LiveHolder),
    /// Nothing was staged; the caller reads again and finds whoever moved it.
    Moved,
}

/// Whether a unit may be worked on without a claim of its own: nobody holds it, a live process
/// does, or its claim changed while this transaction was reading it.
pub(crate) enum Unheld {
    Free,
    Live(LiveHolder),
    /// The caller reads again and finds whoever moved it.
    Moved,
}

/// A live process holds the unit. Waiting releases nothing and changes nothing.
#[derive(Debug, Clone)]
pub(crate) struct LiveHolder {
    key: ClaimKey,
    store_path: PathBuf,
    token: Uuid,
}

impl LiveHolder {
    /// Block in the kernel until this holder releases its claim or exits. The caller must hold no
    /// Store: the holder needs it to publish.
    pub(crate) fn wait(self) -> Result<()> {
        #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
        super::event_search_projection::observe_background_claim_wait(
            self.key.work,
            self.key.context_id,
        );
        wait_released(&self.store_path, self.token)
    }

    /// The typed answer for a caller that keeps its Store open and so must not wait.
    pub(crate) fn refusal(&self) -> CgError {
        CgError::BackgroundWorkHeldByLiveWorker {
            work: self.key.work,
            context_id: self.key.context_id,
            holder: self.token,
        }
    }
}

/// A claim this process owns. Dropping it gives up the presence lock but never deletes the row,
/// so a dropped claim reads as interrupted and the next runner takes it over at once.
pub(crate) struct HeldClaim {
    key: ClaimKey,
    token: Uuid,
    epoch: i64,
    /// The kernel lock that proves this process owns the claim: held while the claim lives, given
    /// up when it drops, or kept past the call that took it by [`HeldClaim::keep`].
    presence: HolderPresence,
}

impl std::fmt::Debug for HeldClaim {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HeldClaim")
            .field("key", &self.key)
            .field("token", &self.token)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl ClaimKey {
    pub(crate) fn new(work: BackgroundWork, context_id: ContextId, unit_id: Uuid) -> Self {
        Self {
            work,
            context_id,
            unit_id,
        }
    }

    fn params(&self) -> HashMap<String, Value> {
        HashMap::from([
            ("work".into(), Value::Text(self.work.as_str().into())),
            ("context_id".into(), Value::Uuid(self.context_id.as_uuid())),
            ("unit_id".into(), Value::Uuid(self.unit_id)),
        ])
    }

    /// Claim this unit inside the caller's transaction, taking over a claim whose holder has
    /// exited. The presence lock is taken before the row is staged, so a crash between the two
    /// leaves no claim behind; a takeover raises the epoch and is guarded on the exact row it
    /// replaces, so of two concurrent takers exactly one commits. A live holder is reported, never
    /// displaced.
    pub(crate) fn take_in_tx(&self, store: &Store, tx: TxId) -> Result<Taken> {
        let current = self.read_in_tx(store, tx)?;
        if let Some(row) = current
            && probe(store.db_path(), row.token)? == Holder::Present
        {
            return Ok(Taken::Live(self.live_holder(store.db_path(), row.token)));
        }
        let token = Uuid::now_v7();
        let presence = hold(store.db_path(), token)?;
        let epoch = match current {
            Some(row) => {
                if !self.guard_in_tx(store, tx, row)? {
                    return Ok(Taken::Moved);
                }
                let epoch = row.epoch.checked_add(1).ok_or_else(|| {
                    CgError::Engine("background claim epoch exhausted INTEGER".into())
                })?;
                let mut params = self.params();
                params.insert("owner_token".into(), Value::Uuid(token));
                params.insert("owner_epoch".into(), Value::Int64(epoch));
                let updated = store
                    .internal_db()
                    .execute_in_tx(tx, TAKE_OVER_CLAIM, &params)
                    .map_err(map_engine_err)?;
                if updated.rows_affected != 1 {
                    return Ok(Taken::Moved);
                }
                epoch
            }
            None => {
                let mut params = self.params();
                params.insert("owner_token".into(), Value::Uuid(token));
                params.insert("owner_epoch".into(), Value::Int64(1));
                match store.internal_db().execute_in_tx(tx, INSERT_CLAIM, &params) {
                    Ok(_) => {}
                    Err(contextdb_core::Error::UniqueViolation { .. }) => return Ok(Taken::Moved),
                    Err(error) => return Err(map_engine_err(error)),
                }
                1
            }
        };
        #[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
        if let Some(hook) =
            super::event_search_projection::take_before_background_claim_commit_hook()
        {
            hook(store);
        }
        Ok(Taken::Held(HeldClaim {
            key: *self,
            token,
            epoch,
            presence,
        }))
    }

    /// Let this transaction go ahead unless a live process holds the unit. An existing claim whose
    /// holder has exited is guarded on its exact row, so a concurrent takeover conflicts at commit.
    /// Used before work that must never displace a live worker, such as recording an attempt.
    pub(crate) fn guard_unheld_in_tx(&self, store: &Store, tx: TxId) -> Result<Unheld> {
        let Some(row) = self.read_in_tx(store, tx)? else {
            return Ok(Unheld::Free);
        };
        if probe(store.db_path(), row.token)? == Holder::Present {
            return Ok(Unheld::Live(self.live_holder(store.db_path(), row.token)));
        }
        if !self.guard_in_tx(store, tx, row)? {
            return Ok(Unheld::Moved);
        }
        Ok(Unheld::Free)
    }

    /// Whether this unit is idle, running or interrupted. Reads only.
    pub(crate) fn state<R: ClaimRows + ?Sized>(&self, rows: &R) -> Result<BackgroundWorkState> {
        let answer = rows.claim_rows(SELECT_CLAIM, &self.params())?;
        match answer.rows.first() {
            None => Ok(BackgroundWorkState::Idle),
            Some(row) => {
                let token = get_uuid(row, &answer.columns, "owner_token")?;
                Ok(match probe(rows.claim_store_path(), token)? {
                    Holder::Present => BackgroundWorkState::Running,
                    Holder::Absent => BackgroundWorkState::Interrupted,
                })
            }
        }
    }

    fn read_in_tx(&self, store: &Store, tx: TxId) -> Result<Option<ClaimRow>> {
        let answer = store
            .internal_db()
            .execute_in_tx(tx, SELECT_CLAIM, &self.params())
            .map_err(map_engine_err)?;
        let Some(row) = answer.rows.first() else {
            return Ok(None);
        };
        Ok(Some(ClaimRow {
            token: get_uuid(row, &answer.columns, "owner_token")?,
            epoch: epoch_of(row, &answer.columns)?,
        }))
    }

    /// Register a commit-time guard on the exact claim row: a takeover, a release or a forget
    /// committed after this transaction read it makes this transaction's commit fail.
    fn guard_in_tx(&self, store: &Store, tx: TxId, row: ClaimRow) -> Result<bool> {
        store
            .internal_db()
            .guard_row_conditions_in_tx(
                tx,
                CLAIMS_TABLE,
                "unit_id",
                &Value::Uuid(self.unit_id),
                &[
                    ("work".into(), Value::Text(self.work.as_str().into())),
                    ("context_id".into(), Value::Uuid(self.context_id.as_uuid())),
                    ("owner_token".into(), Value::Uuid(row.token)),
                    ("owner_epoch".into(), Value::Int64(row.epoch)),
                ],
            )
            .map_err(map_engine_err)
    }

    fn live_holder(&self, store_path: &Path, token: Uuid) -> LiveHolder {
        LiveHolder {
            key: *self,
            store_path: store_path.to_path_buf(),
            token,
        }
    }
}

impl HeldClaim {
    /// Keep this claim's presence held once the call that took it returns, for work whose
    /// ownership spans calls, such as one digest of the understanding drain. It lasts until the
    /// unit's result commits ([`finish_kept_in_tx`], then [`release_kept`]), the unit is given up
    /// ([`abandon_kept`]), or the process exits. contextdb keeps the lock, so this process holds no
    /// claim state of its own.
    pub(crate) fn keep(self) -> Result<()> {
        self.presence
            .keep(*self.key.unit_id.as_bytes())
            .map_err(map_engine_err)
    }

    /// The owner token this claim was taken under.
    pub(crate) fn token(&self) -> Uuid {
        self.token
    }

    /// Fence the caller's publishing transaction on this claim's exact token and epoch. `false`
    /// means the claim is gone or belongs to someone else, and the caller must not publish.
    pub(crate) fn fence_in_tx(&self, store: &Store, tx: TxId) -> Result<bool> {
        self.key.guard_in_tx(
            store,
            tx,
            ClaimRow {
                token: self.token,
                epoch: self.epoch,
            },
        )
    }

    /// Fence the caller's publishing transaction on this claim and delete the claim in it, so the
    /// publication and the end of ownership commit together. `false` means the claim is gone or
    /// belongs to someone else, and the caller must not publish.
    pub(crate) fn finish_in_tx(&self, store: &Store, tx: TxId) -> Result<bool> {
        if !self.fence_in_tx(store, tx)? {
            return Ok(false);
        }
        let deleted = store
            .internal_db()
            .execute_in_tx(tx, DELETE_OWN_CLAIM, &self.own_params())
            .map_err(map_engine_err)?;
        Ok(deleted.rows_affected == 1)
    }

    /// Give the unit up without publishing: delete this claim's exact row, then drop the presence.
    /// A row already taken over or forgotten is left alone, and that is still success.
    pub(crate) fn release(self, store: &Store) -> Result<()> {
        store
            .internal_db()
            .execute(DELETE_OWN_CLAIM, &self.own_params())
            .map(|_| ())
            .map_err(map_engine_err)
    }

    fn own_params(&self) -> HashMap<String, Value> {
        let mut params = self.key.params();
        params.insert("owner_token".into(), Value::Uuid(self.token));
        params.insert("owner_epoch".into(), Value::Int64(self.epoch));
        params
    }
}

/// Whether `work` in this Context is running, interrupted or idle, across every unit it has.
/// A live unit wins over an interrupted one.
pub(crate) fn context_state<R: ClaimRows + ?Sized>(
    rows: &R,
    work: BackgroundWork,
    context_id: ContextId,
) -> Result<BackgroundWorkState> {
    let mut state = BackgroundWorkState::Idle;
    for (_, token) in context_claims(rows, work, context_id)? {
        match probe(rows.claim_store_path(), token)? {
            Holder::Present => return Ok(BackgroundWorkState::Running),
            Holder::Absent => state = BackgroundWorkState::Interrupted,
        }
    }
    Ok(state)
}

/// Wait, one live holder at a time, until no live process holds any unit of `work` in this
/// Context. Each wait blocks in the kernel; a claim whose holder has exited is not waited on.
pub(crate) fn wait_until_no_live_holder<R: ClaimRows>(
    mut rows_for: impl FnMut() -> Result<R>,
    work: BackgroundWork,
    context_id: ContextId,
) -> Result<()> {
    loop {
        let rows = rows_for()?;
        let store_path = rows.claim_store_path().to_path_buf();
        let mut live = None;
        for (unit_id, token) in context_claims(&rows, work, context_id)? {
            if probe(&store_path, token)? == Holder::Present {
                live = Some(LiveHolder {
                    key: ClaimKey::new(work, context_id, unit_id),
                    store_path: store_path.clone(),
                    token,
                });
                break;
            }
        }
        drop(rows);
        match live {
            Some(holder) => holder.wait()?,
            None => return Ok(()),
        }
    }
}

/// The token of the claim this process kept for `unit_id` beside the store, if it keeps one.
pub(crate) fn kept_token(store_path: &Path, unit_id: Uuid) -> Option<Uuid> {
    holder_presence::kept_token(store_path, *unit_id.as_bytes()).map(Uuid::from_bytes)
}

/// Fence the caller's transaction on the claim this process kept for `key`'s unit. `None` when
/// this process keeps no claim of it (a synced-in result being applied, say); `Some(false)` when
/// it kept one but the claim was taken over, released or forgotten since.
pub(crate) fn fence_kept_in_tx(key: &ClaimKey, store: &Store, tx: TxId) -> Result<Option<bool>> {
    let Some(token) = kept_token(store.db_path(), key.unit_id) else {
        return Ok(None);
    };
    match key.read_in_tx(store, tx)? {
        Some(row) if row.token == token => key.guard_in_tx(store, tx, row).map(Some),
        _ => Ok(Some(false)),
    }
}

/// [`fence_kept_in_tx`], and delete the claim in the same transaction, so the unit's result and
/// the end of its claim commit together.
pub(crate) fn finish_kept_in_tx(key: &ClaimKey, store: &Store, tx: TxId) -> Result<Option<bool>> {
    let Some(token) = kept_token(store.db_path(), key.unit_id) else {
        return Ok(None);
    };
    let Some(row) = key.read_in_tx(store, tx)? else {
        return Ok(Some(false));
    };
    if row.token != token || !key.guard_in_tx(store, tx, row)? {
        return Ok(Some(false));
    }
    let mut params = key.params();
    params.insert("owner_token".into(), Value::Uuid(row.token));
    params.insert("owner_epoch".into(), Value::Int64(row.epoch));
    let deleted = store
        .internal_db()
        .execute_in_tx(tx, DELETE_OWN_CLAIM, &params)
        .map_err(map_engine_err)?;
    Ok(Some(deleted.rows_affected == 1))
}

/// Let the presence this process kept for `unit_id` go, once the transaction that ended its claim
/// committed.
pub(crate) fn release_kept(store_path: &Path, unit_id: Uuid) {
    let _ = holder_presence::release_kept(store_path, *unit_id.as_bytes());
}

/// Give up a unit this process kept without a result: delete its claim row while it still names
/// this process, then let the presence go. A row already taken over or forgotten is left alone.
pub(crate) fn abandon_kept(key: &ClaimKey, store: &Store) -> Result<()> {
    if let Some(token) = kept_token(store.db_path(), key.unit_id) {
        let mut params = key.params();
        params.insert("owner_token".into(), Value::Uuid(token));
        store
            .internal_db()
            .execute(DELETE_TOKEN_CLAIM, &params)
            .map_err(map_engine_err)?;
    }
    release_kept(store.db_path(), key.unit_id);
    Ok(())
}

/// Every claim of `work` in this Context: (unit id, owner token).
pub(crate) fn context_claims<R: ClaimRows + ?Sized>(
    rows: &R,
    work: BackgroundWork,
    context_id: ContextId,
) -> Result<Vec<(Uuid, Uuid)>> {
    let answer = rows.claim_rows(
        SELECT_CONTEXT_CLAIMS,
        &HashMap::from([
            ("work".into(), Value::Text(work.as_str().into())),
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
        ]),
    )?;
    answer
        .rows
        .iter()
        .map(|row| {
            Ok((
                get_uuid(row, &answer.columns, "unit_id")?,
                get_uuid(row, &answer.columns, "owner_token")?,
            ))
        })
        .collect()
}

/// Wait in the kernel for the live holder a [`CgError::BackgroundWorkHeldByLiveWorker`] refusal
/// names; return at once for any other condition. The caller must hold no Store.
pub(crate) fn wait_for_refusal(store_path: &Path, refusal: &CgError) -> Result<()> {
    match refusal {
        CgError::BackgroundWorkHeldByLiveWorker {
            work,
            context_id,
            holder,
        } => LiveHolder {
            key: ClaimKey::new(*work, *context_id, Uuid::nil()),
            store_path: store_path.to_path_buf(),
            token: *holder,
        }
        .wait(),
        _ => Ok(()),
    }
}

/// Block in the kernel until `token`'s holder releases its presence or exits.
pub(crate) fn wait_released(store_path: &Path, token: Uuid) -> Result<()> {
    holder_presence::wait_released(store_path, *token.as_bytes()).map_err(map_engine_err)
}

/// Prove the filesystem under the store can take the holder lock before a run writes anything, so a
/// store that cannot prove its workers alive refuses every claim with nothing written.
pub(crate) fn require_presence(store_path: &Path) -> Result<()> {
    hold(store_path, Uuid::now_v7()).map(drop)
}

/// Whether a live process holds `token`'s presence. Reads only.
pub(crate) fn holder_present(store_path: &Path, token: Uuid) -> Result<bool> {
    Ok(probe(store_path, token)? == Holder::Present)
}

fn probe(store_path: &Path, token: Uuid) -> Result<Holder> {
    holder_presence::probe(store_path, *token.as_bytes()).map_err(map_engine_err)
}

fn hold(store_path: &Path, token: Uuid) -> Result<HolderPresence> {
    holder_presence::hold(store_path, *token.as_bytes()).map_err(map_engine_err)
}

fn epoch_of(row: &[Value], columns: &[String]) -> Result<i64> {
    let index = columns
        .iter()
        .position(|column| column == "owner_epoch")
        .ok_or_else(|| CgError::Engine("background claim row has no owner_epoch".into()))?;
    match row.get(index) {
        Some(Value::Int64(epoch)) => Ok(*epoch),
        other => Err(CgError::Engine(format!(
            "background claim owner_epoch: expected INTEGER, got {other:?}"
        ))),
    }
}

/// Run `hook` once inside the next claim or takeover transaction on this thread, after the holder
/// lock is taken and the claim row is staged, before the transaction commits.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn before_next_background_claim_commit_for_test(hook: impl FnOnce(&Store) + 'static) {
    super::event_search_projection::set_before_background_claim_commit_hook(Box::new(hook));
}

/// Tell `observer` each time a runner on this thread starts waiting on a live holder, immediately
/// before it blocks.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn observe_background_claim_waits_for_test(
    observer: impl FnMut(BackgroundWork, ContextId) + 'static,
) {
    super::event_search_projection::set_background_claim_wait_observer(Box::new(observer));
}

/// Release the holder locks of every claim this process holds on `store_path`, as a lost lock
/// would, while the claims themselves stay in memory. Returns how many were released.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn lose_held_claim_presence_for_test(store_path: &Path) -> usize {
    holder_presence::lose_held_for_test(store_path)
}

/// Refuse the next read a write on `store_path` makes after its commit, on this thread, with
/// `error`: the load-dependent read refusal that must never turn a committed write into a failure.
#[cfg(context_graph_internal_test_support)]
#[doc(hidden)]
pub fn refuse_next_post_commit_readback_for_test(store_path: &Path, error: crate::CgError) {
    super::event_search_projection::arm_post_commit_readback_refusal(
        store_path.to_path_buf(),
        error,
    );
}

/// Make the next holder-lock acquisition on this thread fail with `raw_os_error`, the way a
/// filesystem that cannot lock refuses it.
#[cfg(any(context_graph_internal_test_support, feature = "test-support"))]
#[doc(hidden)]
pub fn fail_next_background_claim_presence_for_test(raw_os_error: i32) {
    holder_presence::fail_next_hold_for_test(raw_os_error);
}
