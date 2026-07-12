//! RED contract test for library-owned prompt framing: the production extraction
//! prompt framing lives in the LIBRARY (the flows-doc "no surface without a library API beneath
//! it" rule), so the bench and production render from the same source and cannot drift. The CLI
//! may only delegate. Authored and committed BEFORE implementation; implementation commits may
//! not touch this file.
//!
//! Source-scan guard, same class as the sanctioned cg-core purity guards: it reads the CLI
//! source and asserts the framing is not defined there.

use std::path::Path;

#[test]
fn production_extraction_prompt_framing_lives_in_the_library_not_the_cli() {
    let main_rs = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&main_rs).expect("cg-cli main.rs readable");

    assert!(
        !source.contains("Extract context-graph memory from this conversation"),
        "the extraction prompt framing must not be defined in cg-cli — it belongs to the \
         library assembly (CaptureHarnessConfig::render_extraction_prompt) so the bench and \
         production cannot drift"
    );
    assert!(
        source.contains("render_extraction_prompt("),
        "cg-cli must DELEGATE prompt assembly to the library render"
    );
}
