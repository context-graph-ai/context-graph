# `cg-node-fs`

`cg-node-fs` keeps node-only filesystem work outside the embeddable `context-graph` core. It
loads a node configuration, owns the per-data-root worker lock, and stores the latest event-search
worker execution request at a fixed data-root-relative path.<!-- enforced by: `missing_policy_is_absent_then_writes_and_replacements_round_trip_at_the_fixed_path` -->

The event-search worker-policy record is UTF-8 text no larger than 4 KiB. A replacement is written
through a synced temporary file and atomically persisted, so a reader sees either the prior complete
record or the replacement; oversize input leaves the last record in place. The record carries a
latest execution request only. Parsing it and deciding what a worker does remain with the CLI.
<!-- enforced by: `missing_policy_is_absent_then_writes_and_replacements_round_trip_at_the_fixed_path`, `policy_record_refuses_more_than_four_kibibytes_without_replacing_the_last_value` -->
