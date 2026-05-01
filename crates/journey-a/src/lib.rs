//! Journey A — U1 Physical Security Operator vertical-slice spike.
//!
//! This crate implements the Track 2.1 R0 journey for the U1 ICP "stranger
//! picks up the children" scenario. It walks the named moments of pain
//! M1 (detection) → M2 (notification) → M3 (audit replay) → M4 (cross-tenant
//! signature reuse) end-to-end against a real on-disk contextdb v1.0.0
//! engine instance with concurrent contention where the journey hits it.
//!
//! ----
//!
//! ## Stubbed surfaces — re-walk this journey when these plans ship
//!
//! Per Track 2 honesty rule (R0 plan §"Track 2 stub policy"), every stub
//! the spike introduces is named here with the originating acceptance
//! test that pins the behaviour. Re-walk the journey when the owning
//! plan ships and replace the stub with the real call.
//!
//! Plan 25 cg-server-observe — STUB. Webhook ingress is bypassed; the
//! spike calls `record_observation_inline` directly. The stub does NOT
//! exercise HMAC verification (`observe_webhook_signature_verification`)
//! or HTTP routing — those are application-surface and not the journey's
//! load-bearing contract.
//! Re-walk: when Plan 25 ships its `Server::observe` HTTP handler.
//!
//! Plan 12 cg-server-correlate — STUB. Cross-camera correlation within
//! the same `embedding_space_id` runs as a `semantic_search` against the
//! shipped engine vector index. Stub mirrors `correlate_observations_cross_context`.
//! Re-walk: when Plan 12 ships `Store::correlate_observations`.
//!
//! Plan 13 cg-server-signature-detect — STUB. Cluster aggregation runs
//! by computing centroid from the observation embeddings the spike just
//! recorded; the stub persists a Signature row with member_count + variance
//! but NOT the per-member vectors (per `signature_persists_cluster_aggregate_not_member_vectors`).
//! Modality-agnostic clustering within a single space mirrors
//! `signature_detection_modality_agnostic_within_embedding_space`.
//! Space mismatch uses a different `embedding_space_id` per
//! `signature_detection_space_mismatch_not_clustered`.
//! Re-walk: when Plan 13 ships `Store::detect_signatures`.
//!
//! Plan 14 cg-server-signature-propagate — STUB. Edge registration walks
//! existing signatures and records "would_match" events when centroid
//! distance + scope_label overlap pass. Mirrors `signature_match_at_fresh_edge_uses_centroid_not_member_lookup`
//! and `signature_distribution_respects_scope_labels`.
//! Re-walk: when Plan 14 ships `Store::register_edge` + `Store::match_signature`.
//!
//! Plan 22b cg-notify — STUB. `would_send_to_sink` events are recorded
//! to an in-app sink registry; the spike asserts:
//!   - `notify_signature_match_routes_to_configured_sink` — sink with
//!     `on: signature_match` receives the event; an `on: invalidation`
//!     sink does NOT receive the same event.
//!   - `notify_severity_filter_respected` — only Critical-severity alerts
//!     route to the WhatsApp sink; Info routes to the quieter sink.
//!   - `notify_sink_failure_does_not_block_engine` — the engine commit
//!     returns BEFORE sink delivery is attempted; a failing sink does
//!     not wedge the engine.
//!   - `cross_tenant_signature_match_no_raw_observation_leak` — the sink
//!     payload contains scope-label-match metadata + centroid distance
//!     only, never the other tenant's raw frame URIs.
//! Re-walk: when Plan 22b ships `cg-notify` with NotificationSink.
//!
//! Plan 7 ObservationTrigger (engine-blocked, gap §3) — STUB. After
//! `record_observation`, the spike runs an in-app `basis_diff` walk on
//! the affected Decision and emits an Invalidation row. Severity is
//! mapped from confidence per `notify_severity_filter_respected` (Critical
//! ≥ 0.8, Warning ≥ 0.5, Info < 0.5). The stub does NOT participate in
//! the engine's transaction — gap §3 (ObservationTrigger) blocks atomic
//! invalidation generation by the engine.
//! Re-walk: when contextdb §3 ObservationTrigger ships AND Plan 7 ships
//! `Store::invalidate_decision` integration.
//!
//! Plan 10 EventBus / Plan 8 cg-subscriptions (engine-blocked, gap §5) —
//! STUB. SignatureMatched + InvalidationDetected events route via an
//! in-app callback registry, not via NATS. Mirrors `subscribe_signature_matched_fires_on_edge`.
//! Re-walk: when contextdb §5 EventBus ships AND Plan 10 ships
//! `Store::subscribe`.
//!
//! Plan 23 cg-server-core — wrapper without binary; not exercised. The
//! journey walks the user-observable contract; server lifecycle (NATS
//! wiring, graceful drain) surfaces at per-plan execution.
//!
//! Plan 22a cg-auth — wrapper, the journey uses a single owner principal;
//! ContextAcl gating is not exercised. Engine-blocked subset is gap §9.
//!
//! Plan 11 cg-embedding-spaces — STUB. The journey declares a `face_512`
//! and `face_512_v2` embedding-space metadata row; dimension is enforced
//! by the engine VECTOR(N) column type. The full embedding-space registry
//! API (declare / list / reindex) is not exercised.
//! Re-walk: when Plan 11 ships `Store::list_embedding_spaces`.
//!
//! Plan 3R cg-evidence-retrofit — STUB. Typed `EvidenceRef::ImageFrame`
//! is encoded as a JSON column on the observation row; the spike asserts
//! the URI scheme `vigil-edge://` round-trips through INSERT/SELECT.
//! `record_observation` atomic across observation + evidence + embedding
//! is exercised — the spike commits all three in a single tx.
//! Re-walk: when Plan 3R ships the typed EvidenceRef + EmbeddingSpace
//! schema.

pub mod journey;
