# candle-embeddinggemma

Purpose: Loads and runs the optional EmbeddingGemma text embedder.

Owns: Candle forward execution, tokenizer and weight validation, head dimensions and local artifact loading.
Must not own: The bundled default Nomic model, CLI output, store policies or a second embedding trait.

## Seams

Implements core `Embedder`; register its explicit space in the core Store.

## Where a change lives

`src/embeddinggemma.rs`; configurations and exports in `src/lib.rs`.

## Invariants and guarding tests

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

Valid artifacts are required before first vector; dimensional heads declare matching spaces.

Guards: `tests/acceptance/embeddinggemma_loads.rs`: `embeddinggemma_requires_valid_candle_tokenizer_safetensors_artifacts_before_first_vector`, `embeddinggemma_declares_768_dim_vectors_for_768_head`.

Fast check: `cargo test -p candle-embeddinggemma --test acceptance`. Full gate: `scripts/gate.sh`.
