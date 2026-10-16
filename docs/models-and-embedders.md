# Models and embedders

The default text space is `text_default_nomic_v1_5_768`. <!-- enforced by: `text_default_space_id_resolves_to_text_default_nomic_v1_5_768_after_default_store_open` -->

The offline Nomic embedder loads model artifacts from the selected catalog cache. <!-- enforced by: `catalog_and_offline_embedder_agree_under_cg_model_cache_root_override` -->

An embedder whose `ModelSource::LocalPath` names an empty directory returns typed `EmbedderModelNotAvailable` for embedding text and for preparing a document for event search. <!-- enforced by: `embedding_over_an_isolated_empty_local_cache_refuses_locally_with_a_typed_error_and_never_reaches_the_network` -->

Disabling default features removes the bundled embedder dependencies. <!-- enforced by: `the_core_library_builds_with_default_features_off_and_drops_every_bundled_embedder_dependency` -->

The core builds with default features disabled. <!-- enforced by: `the_core_library_builds_with_default_features_off_and_drops_every_bundled_embedder_dependency` -->

## Choosing an embedder

The default model is fetched from [`nomic-ai/nomic-embed-text-v1.5` on Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). <!-- enforced by: `first_touch_lazy_fetch_observer_event_payload_carries_resolved_source_and_byte_budget` -->

The download is 546,938,168 bytes (about 547MB). <!-- enforced by: `first_touch_lazy_fetch_observer_event_payload_carries_resolved_source_and_byte_budget` -->

<!-- concept -->
At 25Mbps that is about three minutes of transfer time, before protocol overhead and setup.

See [registration](register-an-embedder.md) for opt-in `candle-embeddinggemma`.

| Related entry point | Guide |
| --- | --- |
| `default_embeddinggemma_offline_config` | [Registration](register-an-embedder.md) |
| `Store::open_with_embedder_registrations` | [Registration](register-an-embedder.md) |
| `reindex_space` | [Registration](register-an-embedder.md) |

## Encoding event history

Document preparation counts exactly one `search_document: ` instruction plus untouched content and special tokens. <!-- enforced by: `nomic_document_preparer_adds_exactly_one_system_task_prefix` -->

Embedding that prepared document matches plain embedding of the once-prefixed input. <!-- enforced by: `nomic_document_preparer_adds_exactly_one_system_task_prefix` -->

Query preparation counts exactly one `search_query: ` instruction plus untouched content and special tokens. <!-- enforced by: `nomic_query_preparer_adds_exactly_one_system_task_prefix` -->

Embedding that prepared query matches plain embedding of the once-prefixed input. <!-- enforced by: `nomic_query_preparer_adds_exactly_one_system_task_prefix` -->

Document preparation reports the exact content token byte boundaries. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Literal content beginning with `search_document:` is preserved behind one system-added document prefix. <!-- enforced by: `literal_source_and_query_text_starting_with_nomic_prefix_words_is_preserved_and_searchable` -->

Literal content beginning with `search_query:` is preserved behind one system-added query prefix. <!-- enforced by: `literal_source_and_query_text_starting_with_nomic_prefix_words_is_preserved_and_searchable` -->

Oversized document preparation reports the full final token count. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

It reports the complete content token boundaries. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Embedding a still-oversized document returns `PreparedInputExceedsModelLimit` with the exact count and limit. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Oversized query preparation reports the full final token count. <!-- enforced by: `an_oversized_event_search_query_is_prepared_in_full_and_refused_only_at_embed_time` -->

Embedding a still-oversized query returns the same typed count-and-limit refusal. <!-- enforced by: `an_oversized_event_search_query_is_prepared_in_full_and_refused_only_at_embed_time` -->

An embedder with no event-search capability returns `EventSearchPreparationUnsupported` from document preparation. <!-- enforced by: `embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

Its query preparation returns the same refusal. <!-- enforced by: `embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

Embedder authors: see the [event-search slice bound](library-api.md#event-search-slice-bound) and [registration guide](register-an-embedder.md).

`EventHistoryTextEmbedder::registration` declares the canonical event-history embedding space. <!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

The wrapper shares the caller's loaded model Arc. <!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

Document and query embedding retain the shared model handle. <!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

An ordinary default Store open registers the default text space and no event-history space. <!-- enforced by: `text_default_space_id_resolves_to_text_default_nomic_v1_5_768_after_default_store_open` -->

Event-history queries refuse text-default candidates even when both dimensions are 768. <!-- enforced by: `dedicated_event_search_space_refuses_old_unlabelled_space_comparison` -->

Text-default queries also refuse event-history candidates. <!-- enforced by: `dedicated_event_search_space_refuses_old_unlabelled_space_comparison` -->

The ordinary Nomic embedding path retains exact model, tokenizer and fixture bindings; real vectors are compared within an explicit roundoff envelope across hosts, and same-host role-path vectors remain exactly equal. <!-- enforced by: `event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged` -->

An embedder that declares event-search capability but does not implement preparation returns `EventSearchPreparationUnsupported` for document preparation. <!-- enforced by: `an_embedder_that_declares_the_capability_without_implementing_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

It returns the same refusal for query preparation. <!-- enforced by: `an_embedder_that_declares_the_capability_without_implementing_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

With or without a declared capability, failed preparation is never delegated to plain embedding. <!-- enforced by: `embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed`, `an_embedder_that_declares_the_capability_without_implementing_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

For opt-in EmbeddingGemma and other consumer-declared spaces, follow [embedder registration](register-an-embedder.md).

Concrete, borrowed trait-object and shared trait-object routes produce byte-identical prepared document embeddings. <!-- enforced by: `the_four_role_methods_give_byte_identical_results_across_every_embedder_handle_route` -->

They produce byte-identical prepared query embeddings. <!-- enforced by: `the_four_role_methods_give_byte_identical_results_across_every_embedder_handle_route` -->

A document prepared by the raw embedder is accepted by its wrapper with the same vector. <!-- enforced by: `document_prepared_by_the_raw_embedder_is_accepted_by_the_wrapper_and_produces_the_same_vector` -->

An external embedder cannot borrow canonical event-history authority. <!-- enforced by: `external_embedder_cannot_borrow_canonical_event_history_authority` -->

## Cache layout

`CG_MODEL_CACHE_ROOT` overrides the catalog model cache root. <!-- enforced by: `catalog_and_offline_embedder_agree_under_cg_model_cache_root_override` -->

Without `CG_MODEL_CACHE_ROOT`, `XDG_CACHE_HOME` selects `$XDG_CACHE_HOME/cg/models`. <!-- enforced by: `the_installer_seeds_the_model_cache_root_the_library_reads_when_the_cache_home_is_relocated` -->

With neither variable set, the repository's `scripts/install.sh` uses `~/.cache/cg/models` on Linux and macOS. <!-- enforced by: `install_script_runs_on_linux_and_macos_and_emits_correct_cache_root_per_platform` -->

Explicit fetch writes a `catalog.toml` manifest when it adopts a complete, validated artifact-only cache: a model directory holding `model.safetensors`, `tokenizer.json`, `config.json` and `REVISION` but no manifest. <!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

The adopted model then reports `Cached`. <!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

Corrupt artifact-only caches return a typed checksum refusal. <!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

That refusal creates no manifest. <!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

A default import downloads no missing model. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Each model has its own directory under the cache root, named by model id: `<cache-root>/nomic-embed-text-v1.5/`. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

Default prefetch writes `model.safetensors` and a `catalog.toml` manifest there. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

The written `model.safetensors` matches the catalog checksum. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

Prefetch skips the license-gated EmbeddingGemma model with `ManualAcceptanceRequired`. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

It creates no gated-model cache directory. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

`cg model show` on a license-gated model prints its manual installation instructions and URL. <!-- enforced by: `cli_model_show_license_gated_renders_manual_instructions` -->

See [fetch recipe](../skills/cg-model-fetch/SKILL.md) and [embedder registration](register-an-embedder.md).
