# Configuration reference

With no `CG_DATA_ROOT` override, the store path is `~/.cg/context-graph.db`. <!-- enforced by: `with_no_data_root_override_the_store_lives_at_dot_cg_under_home` -->

The binary reads `CG_DATA_ROOT` from the environment. <!-- enforced by: `the_binary_reads_exactly_the_environment_it_declares` -->

Merge settings into an existing file rather than replacing unrelated tables.

Writer settings are read from `<data-root>/cg/fabric.toml`. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

## Search selection

The CLI search policy defaults to newest order. <!-- enforced by: `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

It defaults to ten recent sessions. <!-- enforced by: `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

`[search].sessions` selects the configured session count. <!-- enforced by: `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

`--search-sessions all` overrides the configured session count. <!-- enforced by: `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

`--search-order newest` overrides configured oldest order. <!-- enforced by: `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

The default library projection builds the ten newest sessions first. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Oldest-order projection builds the ten oldest sessions first. <!-- enforced by: `oldest_order_builds_the_ten_oldest_sessions_oldest_first` -->

Widening to all sessions embeds only the additional sessions. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Narrowing leaves indexed sessions searchable. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Outside-scope indexed events are reported separately from pending work. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Invalid saved search configuration is refused even when valid CLI values are supplied. <!-- enforced by: `invalid_search_config_is_refused_even_when_cli_values_are_present` -->

```toml
[search]
order = "newest"
sessions = 10 # or "all"
```

### Embedding worker control

The default CLI embedding worker count is Rust's available CPU parallelism. <!-- enforced by: `omitted_worker_policy_resolves_to_rust_available_parallelism` -->

Zero, negative and nonnumeric CLI worker limits are refused. <!-- enforced by: `resolver_refuses_zero_negative_and_non_number_values_from_each_input` -->

Zero, negative and nonnumeric configured limits are refused. <!-- enforced by: `resolver_refuses_zero_negative_and_non_number_values_from_each_input` -->

```text
cg import <project-directory> --context <name> --search wait --embedding-workers 4
```

Without a CLI override, `[search].embedding_workers` selects the worker limit. <!-- enforced by: `configured_worker_policy_is_used_when_the_cli_is_silent` -->

```toml
[search]
embedding_workers = 4
```

The command-line worker limit overrides `[search].embedding_workers`. <!-- enforced by: `explicit_cli_worker_policy_wins_over_configuration_in_both_directions` -->

A limit of one prevents overlapping passage embedding calls. <!-- enforced by: `embedding_worker_cap_one_keeps_the_second_call_out_until_the_first_returns` -->

Serial and concurrent builds commit identical passages, vectors, order and coverage. <!-- enforced by: `serial_and_concurrent_runs_commit_identical_passages_vectors_order_and_coverage` -->

For library worker settings, see [projection workers](library-api.md#library-projection-workers).

See [projection and worker coordination](how-it-works.md#projection-and-worker-coordination) for handoffs, admission and retry behavior.

### Passage configuration

The default CLI passage target is 256 tokens. <!-- enforced by: `a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import` -->

Configured passage targets appear in generation status. <!-- enforced by: `a_passage_override_is_recorded_on_the_generation_and_reported_by_status` -->

```toml
[search.passages]
target_tokens = 256
hard_tokens = 384
```

A passage hard limit above the embedder's 8,192-token capability is refused. <!-- enforced by: `passage_profile_capability_ceiling_is_pure_and_preserves_admitted_values` -->

A hard limit equal to the 32-token oversized-event overlap is refused. <!-- enforced by: `a_hard_limit_equal_to_the_oversized_overlap_is_refused_before_any_write` -->

That refusal creates no store. <!-- enforced by: `a_hard_limit_equal_to_the_oversized_overlap_is_refused_before_any_write` -->

An excessive hard limit returns `config_error` in every import mode. <!-- enforced by: `a_hard_passage_ceiling_refuses_each_import_mode_before_new_or_existing_store_writes` -->

It changes no store or captured history. <!-- enforced by: `a_hard_passage_ceiling_refuses_each_import_mode_before_new_or_existing_store_writes` -->

Changing passage settings makes the next wait import rebuild search. <!-- enforced by: `a_changed_passage_override_rebuilds_the_ready_default_generation_on_the_next_wait_import` -->

The prior ready head remains active during the replacement build. <!-- enforced by: `changed_profile_replacement_keeps_the_old_head_until_the_required_union_is_ready` -->

The completed replacement becomes the active head. <!-- enforced by: `changed_profile_replacement_keeps_the_old_head_until_the_required_union_is_ready` -->

### Writer memory limit

Set `[write] memory_limit_bytes` in `cg/fabric.toml` to choose the writer ceiling reported by status. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

The CLI writer ceiling overrides the configured writer ceiling. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

Status reports the configured writer bytes. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

Without an explicit setting, status identifies a machine-derived writer ceiling. <!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

A small writer setting still permits status. <!-- enforced by: `a_small_writer_setting_does_not_block_help_status_or_show` -->

It still permits top-level help. <!-- enforced by: `a_small_writer_setting_does_not_block_help_status_or_show` -->

It still permits a show lookup. <!-- enforced by: `a_small_writer_setting_does_not_block_help_status_or_show` -->

## Read budgets

Set `[read].memory_limit_bytes` in `cg/fabric.toml` to choose the CLI read memory allowance. <!-- enforced by: `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget` -->

`[read].active_time_ms` accepts positive integer milliseconds. <!-- enforced by: `read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds` -->

The default library read memory limit is 1 GiB. <!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling` -->

The default CLI read active-time allowance is 5,000 ms. <!-- enforced by: `read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds` -->

A valid configured read memory limit permits status. <!-- enforced by: `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget` -->

An insufficient read memory limit is refused. <!-- enforced by: `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget` -->

Malformed read configuration returns `config_error`. <!-- enforced by: `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

That refusal leaves the store unchanged. <!-- enforced by: `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

For library entry points and cursor limits, see [read budgets](library-api.md#library-read-budgets).

## Local runtime directory

`CONTEXTDB_OWNER_READ_RUNTIME_DIR` locates the owner's channel directory. <!-- enforced by: `one_provisioned_root_reaches_the_owner_and_a_separate_read_only_command` -->

A separate reader provisioned with the same directory reaches the live owner. <!-- enforced by: `one_provisioned_root_reaches_the_owner_and_a_separate_read_only_command` -->

## Model artifacts

See [models and embedders](models-and-embedders.md#cache-layout) for the independent model cache root.
