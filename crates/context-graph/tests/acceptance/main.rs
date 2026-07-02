mod capture_cost_estimate;
mod capture_cost_visibility;
mod capture_progress;
#[path = "../common/capture_sync_common.rs"]
mod capture_sync_common;
mod cg_main_dep_audit;
mod connected_by_default_context;
mod context_default_is_tenant;
mod context_isolation;
mod correction_learning_loop;
mod correction_promotion_production;
mod crud;
mod digest_capture;
mod discovery;
mod edge_sync_convergence;
mod edge_sync_offline_and_merge;
mod evidence;
#[path = "../common/fake_extractor.rs"]
mod fake_extractor;
mod find_via_digests;
mod full_ontology_extraction;
mod graph_context_isolation;
mod graph_entity_neighborhood;
mod graph_intention_tree;
mod graph_precedent_chain;
mod graph_traverse;
mod grounding_silence;
mod hitl_review_surface;
mod host_session_substrate_guards;
mod impact_analysis;
mod impact_cascade_consolidation;
mod import_parser;
mod ingest_batch;
mod invalidation_lifecycle;
mod learned_skill_store;
mod learning_sync_split;
mod llm_runtime;
#[cfg(context_graph_internal_test_support)]
mod local_entity_face_match_override;
mod model_connection;
mod nomic_loads;
mod pre_extraction_steering;
mod producer_lineage_audit_render;
#[path = "../common/refactor_extractor.rs"]
mod refactor_extractor;
mod relevance_retrieval;
mod school_gate_entity_alert_render;
#[cfg(context_graph_internal_test_support)]
mod screenshot_memory_distinguishability;
#[cfg(context_graph_internal_test_support)]
mod screenshot_memory_review;
mod skill_drift_safeguard;
#[path = "../common/skill_fixtures.rs"]
mod skill_fixtures;
mod structured_harvest_parser;
mod subscriptions;
mod sync_status_reporting;
mod temporal;
mod tier0_capture;
mod two_layer_reversibility;
mod understanding_queue;
mod usage_is_non_basis;
mod usage_lane_determinism;
