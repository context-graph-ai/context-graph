mod capture_cost_estimate;
mod capture_cost_visibility;
mod capture_progress;
#[path = "../common/capture_sync_common.rs"]
mod capture_sync_common;
mod cg_main_dep_audit;
mod context_default_is_tenant;
mod context_isolation;
mod crud;
mod digest_capture;
mod discovery;
mod edge_sync_convergence;
mod edge_sync_offline_and_merge;
mod evidence;
#[path = "../common/fake_extractor.rs"]
mod fake_extractor;
mod find_via_digests;
mod graph_context_isolation;
mod graph_entity_neighborhood;
mod graph_intention_tree;
mod graph_precedent_chain;
mod graph_traverse;
mod host_session_substrate_guards;
mod impact_analysis;
mod impact_cascade_consolidation;
mod import_parser;
mod ingest_batch;
mod invalidation_lifecycle;
mod llm_runtime;
#[cfg(context_graph_internal_test_support)]
mod local_entity_face_match_override;
mod model_connection;
mod nomic_loads;
mod producer_lineage_audit_render;
mod school_gate_entity_alert_render;
#[cfg(context_graph_internal_test_support)]
mod screenshot_memory_distinguishability;
#[cfg(context_graph_internal_test_support)]
mod screenshot_memory_review;
mod subscriptions;
mod sync_status_reporting;
mod temporal;
mod usage_is_non_basis;
