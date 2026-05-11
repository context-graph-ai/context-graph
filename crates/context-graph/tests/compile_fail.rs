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
