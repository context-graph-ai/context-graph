# CLI reference

Top-level help presents import, find, show, status, forget, model connect/fetch/list/show/status/auth login, doctor and selftest. <!-- enforced by: `help_presents_exactly_the_public_command_list` -->

## History arguments

| History command | Contract |
| --- | --- |
| `cg import` | [Import](#cg-import) |
| `cg find`, `cg show` | [Find and show](#cg-find-and-cg-show) |
| `cg status` | [Status](#cg-status) |
| `cg forget` | [Forget](#cg-forget) |

`cg find --format text` prints the passage id and Context name. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

`--write-memory-limit-bytes BYTES` overrides the configured writer budget reported by status. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

Use each command’s `--help` for argument syntax; settings are in [configuration](configuration.md).

History search returns JSON by default. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

`--format text` selects the human search rendering. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

See [how it works](how-it-works.md) for the architecture.

### `cg import`

```text
cd ~/projects/my-app
cg import .
```

A directory import selects the Claude Code session files recorded for that project. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

JSONL files in the working tree and another project's history are excluded from capture. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

Helper-agent transcripts are counted separately from imported project sessions. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

An explicitly imported helper transcript belongs to its parent session. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

Enter at the first interactive project import accepts the suggested Context name. <!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes` -->

The Context prompt does not enter JSON stdout. <!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes` -->

Enter declines linking to an existing same-name Context. <!-- enforced by: `an_existing_same_name_context_requires_confirmation_and_enter_means_no` -->

An explicit affirmative answer links that Context. <!-- enforced by: `an_existing_same_name_context_requires_confirmation_and_enter_means_no` -->

Later `status` calls and imports from a project subdirectory reuse its Context binding. <!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes` -->

An unlinked project without a controlling terminal returns `context_binding_required`. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

The receipt includes the suggested Context name. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

The retry command supplies that name with `--context`. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

The refused import writes nothing. <!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

A project without matching history returns `project_history_not_found`. <!-- enforced by: `cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found` -->

That refusal writes nothing. <!-- enforced by: `cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found` -->

A bound project refuses a different explicit Context with `context_binding_conflict`. <!-- enforced by: `a_bound_project_refuses_a_different_explicit_context_without_changing_either` -->

The conflicting Context is not created. <!-- enforced by: `a_bound_project_refuses_a_different_explicit_context_without_changing_either` -->

A second path, unknown flag, missing value or invalid search mode returns a usage error. <!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work` -->

The usage error names the offending token. <!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work` -->

The usage error creates no data root. <!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work` -->

Deferred capture remains readable in a later status call. <!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

A deferred import builds no searchable passages: a later status reports every captured event as pending. <!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

The default background import reports `captured_search_scheduled`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

An unchanged wait import reports `complete`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Deferred import reports `captured_search_deferred`. <!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

Missing local artifacts make a background import return `captured_search_behind`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A wait import with missing artifacts returns the same state. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Capture still completes. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

The retry names `cg model fetch --defaults`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

The retry includes `--search wait`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A default background import downloads no model. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A foreground embedding failure also returns `captured_search_behind`. <!-- enforced by: `cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status` -->

The captured events remain durable. <!-- enforced by: `cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status` -->

Changing the configured passage profile makes the next wait import embed captured history again. <!-- enforced by: `a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import` -->

The replacement generation becomes the active search head. <!-- enforced by: `a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import` -->

The replacement head answers search. <!-- enforced by: `a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import` -->

Search scope and processing controls are in [configuration](configuration.md#search-selection).

An unchanged caught-up import reports zero `search.local_embedding_calls`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

With `--search defer`, a malformed selected file produces `source_partial`. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file`, `malformed_input_reports_source_partial_at_the_failed_position_without_silently_skipping_it` -->

The command exits 1. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The receipt names the failed file. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The receipt identifies the failing line. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The receipt includes a reason. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The resulting capture is durable. <!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

An import receipt includes `ok`. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

The receipt distinguishes explicit selection from project binding selection. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

The receipt names the selected Context. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

Capture status is separate from search status. <!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

The receipt omits source text and source paths. <!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

An interrupted all-sessions oldest-first import prints a retry preserving that scope, order and explicit worker cap. <!-- enforced by: `interrupted_all_scope_retry_resumes_all_sessions_in_the_same_oldest_order` -->

Exact-file retry templates contain the quoted `'<session.jsonl>'` placeholder. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

Substituting the original separately quoted file path preserves the explicit Context selection. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

Status retries preserve the saved session scope and order. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

Unavailable search retries preserve the saved session scope and order. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

A multi-Context refusal with mixed policies has no single shared retry. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

It provides a separate action for each selected Context. <!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

### `cg find` and `cg show`

```text
cg find "why did we choose that transport?" --format text
cg show <passage-id> --format text
```

A find hit includes the `cosine_similarity` field. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Without an explicit Context flag, find uses the current project's binding. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

`--context a,b` supplies an explicit multi-Context selection. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

`--all-contexts` uses all authorized Contexts. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A hit carries its Context name. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Its `passage_id` can be passed to show to expand the linked events. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

An unknown Context name returns `unknown_context`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

That refusal returns no hits. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Captured but unsearchable Contexts appear in `omitted_contexts`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A selection with no searchable history returns `history_search_not_ready`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Find leaves pending search pending. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Library find reports published searchable-event coverage separately from pending-event coverage. <!-- enforced by: `status_and_find_report_published_coverage_while_the_selected_scope_is_mid_build` -->

Library find reports events outside the selected work scope separately. <!-- enforced by: `status_and_find_report_published_coverage_while_the_selected_scope_is_mid_build` -->

`cg show` returns the exact stored events linked by find. <!-- enforced by: `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

The events appear in source sequence order. <!-- enforced by: `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

The original source file is not required. <!-- enforced by: `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

A passage outside the selected Context returns `passage_not_found`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

That refusal exposes neither the passage text nor its Context name. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

### `cg status`

`cg status` can read a store held by a writer. <!-- enforced by: `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte` -->

That read leaves the data-root bytes unchanged. <!-- enforced by: `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte` -->

Status reports `store_needs_recovery` for an interrupted store. <!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->

The message names `cg import <project-directory> --search defer` as the recovery command. <!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->

That import restores readable status. <!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->

Library history status reports captured events per Context. <!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context` -->

It reports session counts. <!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context` -->

Graph memory totals are distinct from captured history. <!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context` -->

Unprojected history reports pending search. <!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context` -->

Completed projection reports ready search. <!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context` -->

CLI status reports blocked search when the local model is missing. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Library status keeps the active generation id visible during a replacement build. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

The building generation id is distinct from the active generation id. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

The building profile reports the replacement's actual passage profile. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

Building pending coverage is separate from active pending coverage. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

An incomplete first build can have no active generation. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

Its building generation remains observable. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

See [projection and worker coordination](how-it-works.md#projection-and-worker-coordination) for activation and failure ownership.

Status, find and show create nothing when the store is missing. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Missing-store status exits 1. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Its JSON is `{"ok":false,"state":"store_not_found","store":{"db_path":"…"},"retry":"create_store_with_write_command"}`. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Text status writes no stdout. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

It writes an error and a next action on stderr. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Find and show also return `store_not_found`. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Their next action names `cg import`. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

For model connections, see [model status](#models-and-diagnostics).

### `cg forget`

```text
cg forget --session <ref>
cg forget --context my-app --yes
```

Session forget erases the selected session's events. <!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

It erases derived search passages. <!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

The project binding remains. <!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

The Context remains. <!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

Its search progress is purged before the retained history is rebuilt. <!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

CLI session forget schedules a rebuild of remaining history. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A missing local model reports a blocked rebuild. <!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

An invalid configured passage profile reports `rebuild_not_started` after erasure. <!-- enforced by: `forget_reports_a_configured_profile_refusal_after_erasing_without_starting_a_worker` -->

Without a terminal, forget requires confirmation. <!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

Its retry supplies `--yes`. <!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

Enter at the terminal leaves history unchanged. <!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

Whitespace-only session forget erases its checkpoint and source locator. <!-- enforced by: `whitespace_only_session_forget_erases_checkpoint_and_locator` -->

Whitespace-only Context forget erases its checkpoint and source locator. <!-- enforced by: `whitespace_only_context_forget_erases_checkpoint_and_locator` -->

An empty history selection returns `nothing_to_forget`. <!-- enforced by: `empty_history_can_be_forgotten_with_a_small_container_writer_share` -->

Forgetting history in a store placed as a tenant hub returns the typed `EventHistoryForgetRequiresEdgeDiscard` refusal. <!-- enforced by: `a_tenant_hub_store_is_refused_rather_than_purged_from_an_edge` -->

### When a writing command meets a held store

A reader-held-store refusal names the store. <!-- enforced by: `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

It names each verified reader's process id. <!-- enforced by: `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

It names each verified reader's process name. <!-- enforced by: `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

An observed reader without a verified identity is disclosed. <!-- enforced by: `a_store_held_by_two_readers_of_which_one_is_named_presents_both_holders` -->

## Models and diagnostics

| Model or diagnostic command | Recipe |
| --- | --- |
| `cg model connect` | [Connect](../skills/cg-model-connect/SKILL.md) |
| `cg model fetch` | [Fetch](../skills/cg-model-fetch/SKILL.md) |
| `cg model list` | [List](../skills/cg-model-list/SKILL.md) |
| `cg model show` | [Show](../skills/cg-model-show/SKILL.md) |
| `cg model status` | [Model status](../skills/cg-model-status/SKILL.md) |
| `cg model auth login` | [Login](../skills/cg-model-auth-login/SKILL.md) |
| `cg doctor` | [Doctor](../skills/cg-doctor/SKILL.md) |
| `cg selftest` | [Selftest](../skills/cg-selftest/SKILL.md) |

Model connect records the chosen provider. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

It records the chosen endpoint. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

It records the chosen model. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

It stores an environment-variable credential reference. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

`--consent` records authorization. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

Connect makes no model request. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

Model list reports the registered local and connected models. <!-- enforced by: `cli_model_list` -->

Model show includes model kind. <!-- enforced by: `cli_model_show` -->

It includes the connection. <!-- enforced by: `cli_model_show` -->

It includes license information. <!-- enforced by: `cli_model_show` -->

License-gated entries display manual installation instructions. <!-- enforced by: `cli_model_show_license_gated_renders_manual_instructions` -->

Model auth login records the selected existing host session. <!-- enforced by: `model_auth_login_consent_names_account_and_spend_before_anything_spends` -->

The consent prompt names the candidate account, and the login output names the account and its quota spend. <!-- enforced by: `model_auth_login_consent_names_account_and_spend_before_anything_spends` -->

Declining leaves consent unrecorded. <!-- enforced by: `model_auth_login_declined_consent_leaves_no_consent_and_spends_nothing` -->

Declining spends no completion call. <!-- enforced by: `model_auth_login_declined_consent_leaves_no_consent_and_spends_nothing` -->

Model status reports the connection name. <!-- enforced by: `cli_status_connection_slice_shows_name_quota_kind_of_spend_and_expectation` -->

It reports the spend owner. <!-- enforced by: `cli_status_connection_slice_shows_name_quota_kind_of_spend_and_expectation` -->

It reports quota or subscription spend. <!-- enforced by: `cli_status_connection_slice_shows_name_quota_kind_of_spend_and_expectation` -->

Without a connection it says “no model connection.” <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

It provides login guidance. <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

Doctor spends no completion call. <!-- enforced by: `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` -->

`--json` returns structured findings. <!-- enforced by: `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` -->

Selftest spends one live completion call per invocation. <!-- enforced by: `selftest_spends_exactly_one_live_call` -->

Selftest does not persist the reply. <!-- enforced by: `selftest_persists_nothing_the_model_said` -->

Selftest without recorded consent returns `consent_not_recorded`. <!-- enforced by: `cli_selftest_without_consent_is_a_distinct_consent_refused_condition_and_zero_requests` -->

It makes no request. <!-- enforced by: `cli_selftest_without_consent_is_a_distinct_consent_refused_condition_and_zero_requests` -->

## Recipes

| Command | Recipe |
| --- | --- |
| `import` | [cg-import](../skills/cg-import/SKILL.md) |
| `find` | [cg-find](../skills/cg-find/SKILL.md) |
| `show` | [cg-show](../skills/cg-show/SKILL.md) |
| `status` | [cg-status](../skills/cg-status/SKILL.md) |
| `forget` | [cg-forget](../skills/cg-forget/SKILL.md) |
| `model connect` | [cg-model-connect](../skills/cg-model-connect/SKILL.md) |
| `model fetch` | [cg-model-fetch](../skills/cg-model-fetch/SKILL.md) |
| `model list` | [cg-model-list](../skills/cg-model-list/SKILL.md) |
| `model show` | [cg-model-show](../skills/cg-model-show/SKILL.md) |
| `model status` | [cg-model-status](../skills/cg-model-status/SKILL.md) |
| `model auth login` | [cg-model-auth-login](../skills/cg-model-auth-login/SKILL.md) |
| `doctor` | [cg-doctor](../skills/cg-doctor/SKILL.md) |
| `selftest` | [cg-selftest](../skills/cg-selftest/SKILL.md) |
