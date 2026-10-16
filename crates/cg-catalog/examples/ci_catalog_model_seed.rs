// Internal CI entry point; omitted from the library package because it shares private CLI source.
#[path = "../../cg-cli/src/roots.rs"]
mod roots;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = cg_catalog::Catalog::open_with_scope_against_roots(
        roots::data_root(),
        cg_catalog::default_cache_root(),
        roots::CONNECTION_SCOPE,
    )?;
    let summary = catalog.prefetch_defaults()?;
    for fetched in summary.fetched {
        println!("fetched\t{}\t{}", fetched.model_id, fetched.artifact_sha);
    }
    for skipped in summary.skipped {
        println!("skipped\t{}\t{:?}", skipped.model_id, skipped.reason);
    }
    Ok(())
}
