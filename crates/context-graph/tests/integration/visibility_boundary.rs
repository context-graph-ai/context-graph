#[test]
fn vector_search_evidence_search_semantic_search_not_pub_from_crate_facade() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/vector_search_primitives_are_private.rs");
}

#[test]
fn embedding_primitives_module_not_re_exported_via_pub_use() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/embedding_module_is_private.rs");
}
