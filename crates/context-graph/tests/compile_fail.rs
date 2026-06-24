// Proves that immutability of Observation and Digest is enforced at the type system, not by runtime convention: the mutation symbols must fail to compile.
#[test]
fn compile_fail_ui_tests() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/no_create_digest_embedding_field.rs");
    trybuild.compile_fail("tests/ui/no_digest_embedding_field.rs");
    trybuild.compile_fail("tests/ui/no_observation_data_field.rs");
    trybuild.compile_fail("tests/ui/no_record_observation_data_field.rs");
    trybuild.compile_fail("tests/ui/no_update_digest.rs");
    trybuild.compile_fail("tests/ui/no_update_observation.rs");
    trybuild.compile_fail("tests/ui/cluster_aggregate_member_vectors_private.rs");
    trybuild.compile_fail("tests/ui/test_support_embedding_not_public.rs");
    trybuild.compile_fail("tests/ui/composite_query_without_fusion_policy.rs");
    trybuild.compile_fail("tests/ui/composite_query_single_clause_without_fusion_policy.rs");
    trybuild.compile_fail("tests/ui/embedding_module_is_private.rs");
    trybuild.compile_fail("tests/ui/vector_search_primitives_are_private.rs");
    trybuild.compile_fail("tests/ui/byok_reply_phrasing_is_not_a_basis.rs");
}

#[test]
fn enforced_output_cannot_become_decision_basis_barrier_holds() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/byok_reply_phrasing_is_not_a_basis.rs");
}

#[test]
fn extracted_field_is_not_a_basis() {
    // The .stderr declares the compile error that the ModelExtractedField barrier will produce.
    // Until the impl adds the barrier type to src/, the .rs file compiles (no error) and
    // trybuild reports a mismatch.
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/extracted_field_is_not_a_basis.rs");
}

#[test]
fn usage_is_not_a_basis_input_compile_fail() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/usage_is_not_a_basis_input.rs");
}

#[test]
fn bare_store_invalidation_op_does_not_compile() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/compile_fail/invalidation_requires_principal.rs");
}

#[test]
fn bare_store_list_decisions_does_not_compile_from_userland() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/compile_fail/list_decisions_is_not_public.rs");
    trybuild.compile_fail("tests/compile_fail/list_decisions_method_item_is_not_public.rs");
    trybuild.compile_fail("tests/compile_fail/list_decision_filter_is_not_public.rs");
    trybuild.compile_fail("tests/compile_fail/list_decision_filter_module_is_not_public.rs");
}
