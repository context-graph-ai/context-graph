# Troubleshoot an import or read

## Captured but not searchable

The missing-model retry names `cg model fetch --defaults`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

It then names `--search wait`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Capture remains complete. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A default background import downloads no model. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Find returns `history_search_not_ready` for captured but unsearchable history. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

It leaves that history's search pending. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A Context with no imported history returns `no_matching_history`. <!-- enforced by: `find_in_a_context_with_no_imported_history_answers_no_matching_history` -->

That empty result exits successfully. <!-- enforced by: `find_in_a_context_with_no_imported_history_answers_no_matching_history` -->

## A project has no binding

An unlinked project without a terminal returns `context_binding_required`. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

It writes nothing. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

The retry supplies `--context NAME`. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

A project with no matching provider history writes nothing. <!-- enforced by: `cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found` -->

## A source file failed

With `--search defer`, a malformed file produces `source_partial`. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The receipt names the failed line. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

## A read was refused

Malformed read configuration returns `config_error`. <!-- enforced by: `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

`cg status` returns that refusal before it creates or changes any file in the data directory. <!-- enforced by: `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

An insufficient read memory budget returns a mapped memory refusal. <!-- enforced by: `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget` -->

Status names `cg import <project-directory> --search defer` for writable recovery. <!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->

That import restores readable status. <!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->

Dry-run import leaves the recovery-needing store unchanged. <!-- enforced by: `import_dry_run_refuses_a_recovery_needing_store_without_writing` -->

## Another process holds the store

A reader-held-store refusal identifies each verified reader's process id. <!-- enforced by: `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

It identifies each verified reader's process name. <!-- enforced by: `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

## Provider diagnostics

Doctor spends no completion call. <!-- enforced by: `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` -->

Selftest spends one live call per invocation. <!-- enforced by: `selftest_spends_exactly_one_live_call` -->

Billing failures have the `billing` bucket. <!-- enforced by: `selftest_classifies_billing_and_rate_limit_buckets_end_to_end` -->

Rate-limit failures have the distinct `rate_limit` bucket. <!-- enforced by: `selftest_classifies_billing_and_rate_limit_buckets_end_to_end` -->

See [command recipes](cli.md#recipes) and [configuration](configuration.md).

## Remove local data

<!-- concept -->
For full removal, first stop processes using the store and back up what you need. Then delete the chosen data root.

<!-- concept -->
Consult [storage locations](privacy-and-data-movement.md#where-your-data-lives) and the [independent model cache](models-and-embedders.md#cache-layout). Handle provider-side deletion under your provider's terms.
