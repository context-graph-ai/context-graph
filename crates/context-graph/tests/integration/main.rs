#[path = "../common/fake_extractor.rs"]
mod fake_extractor;
#[path = "../common/invalidation_fixtures.rs"]
mod invalidation_fixtures;
#[path = "../common/refactor_extractor.rs"]
mod refactor_extractor;

mod cluster_aggregate_substrate;
#[cfg(context_graph_internal_test_support)]
mod composite_query_edge_cases;
mod concurrent_probes_evidence;
#[cfg(context_graph_internal_test_support)]
mod cross_context_label_filter;
#[cfg(context_graph_internal_test_support)]
mod cross_space_refusal;
mod decision_reasoning_chain_metadata;
mod embedder_metadata;
mod entity_reference_edge_cases;
mod entity_reference_embeddings;
mod four_stages_three_sources;
mod harness_contract_single_source;
mod provider_schema_coverage;
#[cfg(context_graph_internal_test_support)]
mod engine_probes {
    pub mod embedder_concurrency;
}
mod capture_pipeline;
#[cfg(context_graph_internal_test_support)]
mod explain_introspection;
mod extraction_prompt_config;
mod impact_analysis_engine_capabilities;
mod impact_cascade_probes;
mod import_tool_output;
mod ingest_core_streaming;
#[cfg(context_graph_internal_test_support)]
mod intra_space_modality_agnostic;
mod invalidation_audit_log;
mod invalidation_concurrency;
mod pass_focus;
mod precomputed_embedding_edge_cases;
mod precomputed_embedding_ingest;
mod producer_diagnostic;
mod registry_edge_cases;
mod reindex;
#[cfg(context_graph_internal_test_support)]
mod reindex_concurrency;
mod reliability_wrappers;
mod schema;
mod staged_extraction;
#[cfg(context_graph_internal_test_support)]
mod store_open_runtime_config;
mod subscriptions_engine_capabilities;
mod visibility_boundary;
