# `cg-node-fs`

`cg-node-fs` keeps node-only filesystem work outside the embeddable `context-graph` core.

When the node configuration file cannot be read, `cg status` stops with a configuration error that
names the file and leaves the data root unchanged.
<!-- enforced by: `unreadable_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

While another holder has the data root's worker lock, a background `cg import` reports the search
worker as `already_running`.
<!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff` -->

When the worker lock file cannot be opened, `cg import` reports the search worker as `spawn_failed`
with a reason.
<!-- enforced by: `cg_import_reports_search_behind_when_the_worker_lock_cannot_be_opened` -->

The latest event-search worker request lives at a fixed data-root-relative path. Before any write a
read returns nothing; each write replaces the record and reads back exactly.
<!-- enforced by: `missing_policy_is_absent_then_writes_and_replacements_round_trip_at_the_fixed_path` -->

A record larger than 4 KiB is refused, and the last record stays in place.
<!-- enforced by: `policy_record_refuses_more_than_four_kibibytes_without_replacing_the_last_value` -->

Under the update guard, a caller reads the record and replaces it.
<!-- enforced by: `update_guard_reads_and_replaces_one_bounded_record` -->
