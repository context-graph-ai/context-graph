// Proves that immutability of Observation and Digest is enforced at the type system, not by runtime convention: the mutation symbols must fail to compile.
#[test]
fn compile_fail_ui_tests() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/ui/*.rs");
}
