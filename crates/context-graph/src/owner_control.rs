//! The administrative owner plane: asking the process that owns a store a question.
//!
//! A live owner holds the store's writable handle, so nobody else can read the file behind it.
//! Anything that needs an answer out of a running context-graph — a status probe, an
//! administrative frame — asks the owner through the database itself: the owner registers its
//! [`ControlHandler`] on the writable open, and a client reaches it with
//! [`request_owner`], naming only the STORE PATH. There is no socket path in this API and no
//! second transport in this crate: the channel, its framing, its deadlines and its refusals all
//! belong to contextdb's bounded read surface, and context-graph declares only two things of its
//! own — the namespace its frames travel under, and the [`ReadLimits`] one of its reads may spend.
//!
//! A caller that must fall back to doing the work itself distinguishes
//! [`OwnerControlError::OwnerNotRunning`] — nobody is holding the store, so the caller may open it
//! — from every other variant, which all mean an owner IS there and did not answer.
//!
//! The ask is paid for by the running owner and never by the store's file: [`request_owner`] takes
//! contextdb's owner-only door, so a store nobody holds is reported owner-absent without the file
//! being hydrated, locked, inspected or created.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use contextdb_core::Error as DbError;
use contextdb_core::read_contract::{
    OwnerReadLimits, OwnerRequestHandler, ReadFailureDetail, ReadFailureKind, ReadLimits,
};
use contextdb_engine::{OwnerReadConfig, ReadSession, ReadSessionOptions};

/// The stop token a caller uses to end a wait it started.
///
/// Re-exported so a consumer of this crate constructs one without naming the substrate: vigil and
/// anything else above context-graph builds against context-graph's vocabulary, the same way
/// [`ReaderIdentity`](crate::ReaderIdentity) spares them the substrate's reader record.
pub use contextdb_core::read_contract::OwnerReadCancellation;

/// The namespace every context-graph administrative frame travels under. An owner request that
/// names anything else is not this crate's and is refused by the handler.
pub const CONTROL_NAMESPACE: &str = "context-graph/control";

/// The substrate's own bootstrap variable for the directory the owner channel lives in.
///
/// Read here under the substrate's name and with the substrate's meaning — it names a DIRECTORY
/// and nothing else. It never says whether owner inspection runs, and never what a read may spend;
/// those stay `owner_read_config` and [`context_graph_read_limits`]. cg invents no name of its
/// own for this, because one directory has to serve BOTH sides: the owner puts its channel there,
/// and a session given the same root looks for it exactly there. Unset, contextdb's per-user
/// runtime directory applies as it always has.
const OWNER_READ_RUNTIME_DIR_ENV: &str = "CONTEXTDB_OWNER_READ_RUNTIME_DIR";

/// Where an operator has said this machine's owner channel lives, if anywhere.
///
/// A container or a packaged service has no per-user runtime directory to fall back on, so it
/// provisions this ONE variable and both halves of the owner plane line up on it. Empty is the
/// same as unset: an operator who exported nothing has said nothing.
pub(crate) fn owner_read_runtime_dir() -> Option<PathBuf> {
    std::env::var_os(OWNER_READ_RUNTIME_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// What the live owner runs on an administrative frame, holding the store while it answers.
pub type ControlHandler = Arc<dyn Fn(String) -> String + Send + Sync + 'static>;

/// What one context-graph read of a context-graph store may spend.
///
/// Public contract: these ceilings are declared once and nowhere else. Every session this crate
/// opens — the administrative client, the bounded status totals — reads under them, so "what a cg
/// read costs" has a single answer to read and a single place to change. They are the shipped
/// contextdb ceilings written out rather than borrowed, because a read that does not fit is paged
/// through a cursor; the limits are never raised to make a scan fit.
pub const fn context_graph_read_limits() -> ReadLimits {
    ReadLimits {
        result_rows: 500,
        result_bytes: ReadLimits::SHIPPED_RESULT_BYTES,
        work: 50_000,
        active_ms: 5_000,
        memory: ReadLimits::SHIPPED_MEMORY,
        cursor_page_rows: 100,
        cursor_page_bytes: ReadLimits::SHIPPED_CURSOR_PAGE_BYTES,
        cursor_idle_ms: 300_000,
        cursor_lifetime_ms: 1_800_000,
    }
}

/// The session options every context-graph read opens with.
pub fn context_graph_read_session_options() -> ReadSessionOptions {
    ReadSessionOptions {
        limits: context_graph_read_limits(),
        ..ReadSessionOptions::default()
    }
}

/// Why an administrative request did not come back with an answer.
///
/// [`Self::OwnerNotRunning`] is the one variant that says the store is FREE: nobody holds it, so a
/// caller whose fallback is to do the work itself may open the store directly. Every other variant
/// says an owner is there and the request did not get through, which is never a licence to open the
/// store behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnerControlError {
    /// No process holds this store, so there is nobody to ask. The store is free for the caller
    /// to open itself, whatever condition its file is in.
    OwnerNotRunning { store_path: PathBuf },
    /// There is no store at this path at all. [`request_owner`] never produces this, because the
    /// owner-only route it takes never looks at the file and so cannot know; it stays part of the
    /// vocabulary a caller matches on for routes that do.
    NoStore { store_path: PathBuf },
    /// A process holds this store but does not serve owner requests.
    OwnerNotServing { store_path: PathBuf },
    /// The owner is already serving as many requests as it admits.
    OwnerAtCapacity { store_path: PathBuf },
    /// The owner did not answer inside the time this request allows.
    OwnerTimedOut { store_path: PathBuf },
    /// The owner answered, and the answer was not the text a control frame carries.
    AnswerNotText { store_path: PathBuf, reason: String },
    /// The request reached a live owner and came back refused, in the store's own words.
    Refused { store_path: PathBuf, reason: String },
    /// A process holds this store, but its channel does not serve the inspection this request
    /// asked for. Unlike every other variant here, this one names something a caller MAY still
    /// get: a direct file session, once no writer holds the store. `inspection` is the substrate's
    /// own stable word for what went unanswered, so a caller licenses that fallback on this TYPE
    /// rather than on the words in a message. This stays a variant distinct from
    /// [`Self::OwnerNotServing`] -- whose claim-window contract must keep licensing nothing
    /// downstream -- and [`Self::owner_absent`] stays false for it, because somebody really holds
    /// the store.
    OwnerRouteUnsupported {
        store_path: PathBuf,
        inspection: String,
    },
}

impl OwnerControlError {
    /// The store this request was about.
    pub fn store_path(&self) -> &Path {
        match self {
            Self::OwnerNotRunning { store_path }
            | Self::NoStore { store_path }
            | Self::OwnerNotServing { store_path }
            | Self::OwnerAtCapacity { store_path }
            | Self::OwnerTimedOut { store_path }
            | Self::AnswerNotText { store_path, .. }
            | Self::Refused { store_path, .. }
            | Self::OwnerRouteUnsupported { store_path, .. } => store_path.as_path(),
        }
    }

    /// Whether the store is free for this caller to open itself.
    pub const fn owner_absent(&self) -> bool {
        matches!(self, Self::OwnerNotRunning { .. } | Self::NoStore { .. })
    }
}

impl fmt::Display for OwnerControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OwnerNotRunning { store_path } => write!(
                formatter,
                "no process is holding the store at {}",
                store_path.display()
            ),
            Self::NoStore { store_path } => {
                write!(formatter, "there is no store at {}", store_path.display())
            }
            Self::OwnerNotServing { store_path } => write!(
                formatter,
                "the process holding the store at {} does not answer owner requests",
                store_path.display()
            ),
            Self::OwnerAtCapacity { store_path } => write!(
                formatter,
                "the process holding the store at {} is already serving as many requests as it admits",
                store_path.display()
            ),
            Self::OwnerTimedOut { store_path } => write!(
                formatter,
                "the process holding the store at {} did not answer inside the time this request allows",
                store_path.display()
            ),
            Self::AnswerNotText { store_path, reason } => write!(
                formatter,
                "the owner of the store at {} answered with bytes that are not text: {reason}",
                store_path.display()
            ),
            Self::Refused { store_path, reason } => write!(
                formatter,
                "the owner of the store at {} refused the request: {reason}",
                store_path.display()
            ),
            Self::OwnerRouteUnsupported {
                store_path,
                inspection,
            } => write!(
                formatter,
                "the process holding the store at {} does not serve the {} inspection over its \
                 channel; a direct file session answers it once no writer holds the store",
                store_path.display(),
                inspection.replace('_', " ")
            ),
        }
    }
}

impl std::error::Error for OwnerControlError {}

/// Ask the process that owns the store at `store_path` an administrative question, and give back
/// what its handler answered.
///
/// The store path is the whole address: route selection, the local channel, framing and deadlines
/// are contextdb's, and this function adds only the namespace and this crate's declared limits.
///
/// This is the OWNER-ONLY door — [`ReadSession::open_owner_only`] — and never the general open.
/// The question is about a running process, and the committed file has no answer to it. Reading
/// the file publishes a reader that a writer starting beside the store must wait for, so a probe
/// that fell through to the file would stand in the way of the very process it is asking about;
/// and a file a direct read refuses would answer with the file's condition instead of the plain
/// fact that nobody owns the store. Both are avoided by never looking at the file: a store nobody
/// owns answers [`OwnerControlError::OwnerNotRunning`] whatever shape its file is in, including a
/// crash-dirty one, and nothing is created at a path that has no store.
///
/// A consequence worth stating plainly: because the file is never consulted, this route cannot
/// tell a store that was given up from a path that never had one, so it never answers
/// [`OwnerControlError::NoStore`]. A caller that wants the store's committed rows opens a session
/// of its own, which reads the file exactly as it always has.
pub fn request_owner(store_path: &Path, frame: &str) -> Result<String, OwnerControlError> {
    // The same one root both sides of this plane meet on: the owner puts its channel where
    // `owner_read_config` was told to, and this asks in exactly that directory. A client in a
    // container or a packaged service, where there is no per-user runtime directory to fall back
    // on, would otherwise be told nobody owns the store — about the very owner it was pointed at.
    let session = ReadSession::open_owner_only_in_runtime_dir(
        store_path,
        context_graph_read_session_options(),
        owner_read_runtime_dir(),
    )
    .map_err(|error| classify(store_path, error))?;
    let answer = session
        .request_owner(CONTROL_NAMESPACE, frame.as_bytes())
        .map_err(|error| classify(store_path, error))?;
    String::from_utf8(answer).map_err(|error| OwnerControlError::AnswerNotText {
        store_path: store_path.to_path_buf(),
        reason: error.to_string(),
    })
}

/// The owner-request handler to attach at a writable open so `handler` answers this crate's
/// administrative frames.
pub fn owner_request_handler(handler: ControlHandler) -> Arc<dyn OwnerRequestHandler> {
    Arc::new(ControlNamespaceHandler { handler })
}

/// The owner-read configuration a writable open takes to serve `handler`.
///
/// The ceilings the owner enforces are the ones this crate declares, so what a client is told the
/// owner will spend and what the owner actually spends are the same numbers from the same place.
///
/// Where the channel is put comes from `CONTEXTDB_OWNER_READ_RUNTIME_DIR`, so an operator who has to
/// place it — a container, a packaged service — places it once for the whole owner plane rather
/// than once per side.
pub fn owner_read_config(handler: ControlHandler) -> OwnerReadConfig {
    OwnerReadConfig {
        enabled: true,
        limits: OwnerReadLimits {
            limits: context_graph_read_limits(),
            ..OwnerReadLimits::default()
        },
        runtime_dir: owner_read_runtime_dir(),
        handler: Some(owner_request_handler(handler)),
        ..OwnerReadConfig::default()
    }
}

/// How a wait for a store's direct readers to let go of it ended.
///
/// `Unobservable` carries the reason as text rather than a substrate error, so a consumer branches
/// on this crate's vocabulary and nothing below it — and so an answer nobody could check is never
/// dressed up as `Released`.
#[derive(Debug)]
pub enum ReaderRelease {
    /// No direct reader is holding the store any more. A writer may take it.
    Released,
    /// The caller asked to stop waiting, and is answered at once.
    Stopped,
    /// Who is holding the store could not be observed, so this wait says nothing about it. The
    /// caller decides what to do rather than being handed a reassuring answer nobody checked.
    Unobservable(String),
}

/// Wait until no direct reader is holding the store at `path`.
///
/// A writer refused by direct readers ([`crate::CgError::StoreHeldByReaders`]) is not looking at a
/// broken machine: the readers finish and the next attempt succeeds. This is how a caller waits for
/// that instead of retrying on a timer of its own invention.
///
/// EVENT-DRIVEN, with no timer anywhere in it. The readers themselves answer this wait: every
/// direct reader takes a SHARED advisory range hold on the companion file beside the store for
/// exactly its load window, so asking for that same hold exclusively is the caller's real question
/// — "may I take this store" — put to the only thing that can answer it. The kernel puts the waiter
/// to sleep until the last holder lets go, or dies, which it treats as the same event. The wake IS
/// the release; nothing polls, and no interval decides how quickly a waiter learns. A store with
/// several readers answers only when the last of them has gone. `stop` ends the wait the moment it
/// is cancelled, through the token's own listener, so an interrupt is as prompt as a release.
///
/// The breadcrumbs readers write down are OPTIONAL, naming-only diagnosis kept in the default
/// per-user runtime location: they let an `Unobservable` wait name who is holding on. They never
/// decide the verdict. A reader whose breadcrumb could not be published is still a reader and still
/// holds the wait, and a breadcrumb directory that cannot be read says nothing about whether the
/// store is free.
///
/// WHAT A DIRECT READER IS, and the load window: a reader publishes itself for the span in which it
/// is LOADING a committed image, not for the whole life of a reading session. So this waits out the
/// loads in flight; it is not a lease on the store, and a reader that starts loading after this
/// answers `Released` is a new holder a later open will meet.
///
/// The owner-channel runtime root this crate takes — `CONTEXTDB_OWNER_READ_RUNTIME_DIR`, see
/// [`owner_read_config`] — is passed to the substrate for addressing COMPATIBILITY only, and
/// chooses nothing here. The readers are found by their hold on the companion beside the store, so
/// where the owner channel lives and whether this store's readers have let go are two unrelated
/// facts. A root that is absent, malformed, or unusable therefore does not affect the release
/// verdict at all: it is an owner-channel misconfiguration, and it surfaces on its own as
/// `owner_not_serving` when the channel fails to start.
pub fn wait_for_reader_release(path: &Path, stop: &OwnerReadCancellation) -> ReaderRelease {
    // The operator's pathname goes down the root-taking door so this crate keeps ONE owner-channel
    // input, but the door neither resolves nor validates it: the verdict comes from the readers'
    // hold on the store's companion. An unusable root is an owner-channel fault answered on its own
    // as `owner_not_serving`, never a reason to give up on a wait that would have ended by itself.
    match contextdb_engine::persistence::wait_for_reader_release_in_runtime_dir(
        path,
        owner_read_runtime_dir().as_deref(),
        stop,
    ) {
        contextdb_engine::persistence::ReaderReleaseWait::Released => ReaderRelease::Released,
        contextdb_engine::persistence::ReaderReleaseWait::Stopped => ReaderRelease::Stopped,
        contextdb_engine::persistence::ReaderReleaseWait::Unobservable(error) => {
            ReaderRelease::Unobservable(error.to_string())
        }
    }
}

struct ControlNamespaceHandler {
    handler: ControlHandler,
}

impl OwnerRequestHandler for ControlNamespaceHandler {
    fn handle(
        &self,
        namespace: &str,
        request: &[u8],
        _cancel: &OwnerReadCancellation,
    ) -> contextdb_core::Result<Vec<u8>> {
        if namespace != CONTROL_NAMESPACE {
            return Err(DbError::Other(format!(
                "`{namespace}` is not a context-graph owner request namespace"
            )));
        }
        let frame = std::str::from_utf8(request).map_err(|error| {
            DbError::Other(format!(
                "a context-graph control frame must be text: {error}"
            ))
        })?;
        Ok((self.handler)(frame.to_string()).into_bytes())
    }
}

/// Turn what the read surface refused into what an administrative caller decides on.
///
/// The wording each variant renders is part of this: an answer produced when NO owner is there
/// must never attribute the outcome to a live one. An operator told "the owner of the store
/// refused the request" goes hunting for a process that does not exist and leaves a store they
/// were free to open sitting idle. Because the owner-only route decides before the file is
/// touched, a file condition can no longer arrive here dressed as a live owner's refusal:
/// nobody-holds-it answers [`OwnerControlError::OwnerNotRunning`], which says exactly that.
fn classify(store_path: &Path, error: DbError) -> OwnerControlError {
    let store_path = store_path.to_path_buf();
    let DbError::ReadFailure(failure) = &error else {
        return OwnerControlError::Refused {
            store_path,
            reason: error.to_string(),
        };
    };
    match failure.kind() {
        // Nobody holds the store — decided without opening the file, so this is what a
        // crash-dirty store, an idle one and a path with no store alike are told, and it is
        // exactly the answer a caller with its own fallback needs.
        ReadFailureKind::OwnerNotRunning => OwnerControlError::OwnerNotRunning { store_path },
        ReadFailureKind::StoreNotFound => OwnerControlError::NoStore { store_path },
        ReadFailureKind::OwnerNotServing => OwnerControlError::OwnerNotServing { store_path },
        ReadFailureKind::OwnerAtCapacity => OwnerControlError::OwnerAtCapacity { store_path },
        ReadFailureKind::OwnerTimeout => OwnerControlError::OwnerTimedOut { store_path },
        // The substrate's own construction guard pairs this kind with exactly this detail, so a
        // store nobody built by hand always carries the inspection name here.
        ReadFailureKind::OwnerRouteUnsupported => {
            let inspection = match failure.detail() {
                ReadFailureDetail::OwnerRouteUnsupported(detail) => detail.inspection.clone(),
                _ => String::new(),
            };
            OwnerControlError::OwnerRouteUnsupported {
                store_path,
                inspection,
            }
        }
        _ => OwnerControlError::Refused {
            store_path,
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_read_limits_are_a_valid_read_contract() {
        context_graph_read_limits()
            .validate()
            .expect("context-graph read limits are a valid read contract");
    }

    #[test]
    fn a_session_reports_the_limits_this_crate_declares() {
        assert_eq!(
            context_graph_read_session_options().limits,
            context_graph_read_limits()
        );
    }

    /// The substrate's typed `OwnerRouteUnsupported` refusal must classify to a typed variant of
    /// its own, carrying the inspection the substrate named, rather than falling into the
    /// catch-all `Refused { reason: error.to_string() }` that wears live-owner-refusal wording a
    /// caller cannot license a fallback on.
    #[test]
    fn owner_route_unsupported_classifies_to_a_typed_variant_naming_the_inspection() {
        let store_path = PathBuf::from("/tmp/some-store");
        let failure = contextdb_core::read_contract::ReadFailure::owner_route_unsupported(
            "custom_namespace".to_string(),
        );

        let classified = classify(&store_path, DbError::from(failure));

        assert_eq!(
            classified,
            OwnerControlError::OwnerRouteUnsupported {
                store_path: store_path.clone(),
                inspection: "custom_namespace".to_string(),
            }
        );
        // Somebody really holds the store -- its channel merely does not serve this inspection --
        // so this is never a licence for a caller to open the store as though nobody were there.
        assert!(!classified.owner_absent());
    }

    /// `OwnerNotServing` and `OwnerRouteUnsupported` must stay distinct variants with distinct
    /// classification arms: the claim-window contract pins on `OwnerNotServing` and it must keep
    /// licensing nothing downstream, so the two can never collapse into one classification.
    #[test]
    fn owner_route_unsupported_stays_distinct_from_owner_not_serving() {
        let store_path = PathBuf::from("/tmp/some-store");

        let not_serving_failure = contextdb_core::read_contract::ReadFailure::new(
            ReadFailureKind::OwnerNotServing,
            contextdb_core::read_contract::ReadFailureDetail::None,
        )
        .expect("owner-not-serving pairs with no specialized detail");
        let route_unsupported_failure =
            contextdb_core::read_contract::ReadFailure::owner_route_unsupported(
                "custom_namespace".to_string(),
            );

        let not_serving = classify(&store_path, DbError::from(not_serving_failure));
        let route_unsupported = classify(&store_path, DbError::from(route_unsupported_failure));

        assert!(matches!(
            not_serving,
            OwnerControlError::OwnerNotServing { .. }
        ));
        assert!(matches!(
            route_unsupported,
            OwnerControlError::OwnerRouteUnsupported { .. }
        ));
        assert_ne!(not_serving, route_unsupported);
    }
}
