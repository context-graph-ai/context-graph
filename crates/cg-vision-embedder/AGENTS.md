# cg-vision-embedder

Purpose: Loads and runs the bundled SigLIP-family joint image/text embedder.

Owns: Image preprocessing, tokenizer/towers, Candle inference and normalized vector output.
Must not own: Text-default selection, store policies, CLI rendering or a second embedding abstraction.

## Seams

Implements core `Embedder`; consumers register its declared space.

## Where a change lives

`src/lib.rs` for load, preprocessing, inference and core trait implementation.

## Invariants and guarding tests

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

Missing artifacts and unsupported kinds return typed errors; same-space output dimension and normalization remain stable.

Guard: `tests/siglip.rs`: `load_from_a_missing_directory_is_a_typed_error_not_a_panic`.
Review-only in automated runs: `weight_backed_dimension_and_normalized_distinct_vectors` and `weight_backed_unsupported_kinds_are_typed_errors` return without asserting unless `CG_SIGLIP_TEST_WEIGHTS` names a local checkpoint, and neither the gate nor CI sets it.

Fast check: `cargo test -p cg-vision-embedder`. Full gate: `scripts/gate.sh`.
