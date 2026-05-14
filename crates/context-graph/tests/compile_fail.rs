// Proves that immutability of Observation and Digest is enforced at the type system, not by runtime convention: the mutation symbols must fail to compile.
#[test]
fn compile_fail_ui_tests() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/*.rs");
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
