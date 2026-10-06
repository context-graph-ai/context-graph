# Event-history maintenance measurement

`event_fresh_graph_build` measures fresh construction and durable publication of one graph from
exactly 10,000 real Nomic vectors in the sealed `P10000-K1-sessions` fixture. Run the compiled
`cg_event_history` benchmark with an absent output directory:

```sh
CG_EVENT_BENCH_FRESH_GRAPH_SOURCE=/path/to/realmodel-fixtures/P10000-K1-sessions \
CG_EVENT_BENCH_FRESH_GRAPH_DIR=/path/to/new-fresh-graph \
CG_EVENT_BENCH_EVIDENCE=/path/to/fresh-graph.jsonl \
/path/to/cg_event_history --bench event_fresh_graph_build
```

Build with `RUSTFLAGS='--cfg context_graph_internal_test_support'` and
`cargo bench --no-run -p context-graph --bench cg_event_history`. The selector runs only when
explicitly named; flag-only dispatch and ordinary benchmark runs do not seed its store.

The source opens read-only at 512 MiB and is paged in UUID order. Its stored content must match the
sealed binding, with exactly 10,000 current searchable 768-dimensional vectors and one Context and
search generation. The fresh writer uses caller-driven maintenance at a fixed 1.5 GiB ceiling;
the input collection and source reader are released before timing. The start record asserts one
partition, 10,000 live vectors, zero retained vectors, no graph or base generation, no build in
flight, and 10,000 pending inserts. One production maintenance call is timed through durable
publication. Success requires one built partition, zero remaining partitions, no failure record,
and indexed search after closing and reopening at the same ceiling.

The successful final record emits `fixture.fresh_graph_build_seconds` and includes the asserted
precondition and restart evidence. A refusal or failed assertion exits nonzero.

The sessions fixture retains its final maintenance duration as
`fixture.post_projection_maintenance_seconds`: projection has already maintained the graph before
that span starts. Synthetic fixture construction retains `fixture.vector_maintenance_seconds`.
The fresh graph measurement has a narrow row schema. Capture, projection, recall, latency and
resource measurements are separate.

## Recall floor per query

Every query in a required scope and mode must have recall@10 against EXACT of at least 0.95.
The result retains each value as `query_recalls_at_10_against_exact`, reports
`minimum_recall_at_10_against_exact` and `queries_below_recall_floor`, and keeps
`aggregate_recall_at_10_against_exact` as a diagnostic. A pooled 0.99 from 99 perfect queries and
one zero-recall query is below the floor. Distance work continues to compare the mean of the paired
query set within each scope and mode.

`--bench event_recall_gate_controls` runs without a fixture or model. Its controls include the
99-perfect/one-zero case, the exact 0.95 boundary, a value immediately below the floor, and a
missing per-query recall. The minimum recall must be at least 0.95 and
`queries_below_recall_floor` must be zero.

## Event-history reads during schema replay

The explicit selector below uses the 10,000-passage, eight-Context synthetic fixture and the real
Nomic query embedder. It calls the same read-only library methods as find, show (including linked
events), and event-history status while the held database applies its actual startup ChangeSet
and a root migration across 100 existing FK-linked tables with vectors and an including-sync
trigger. Schema work enters through direct replay of the real apply path. Short migrations repeat
on fresh copies of the same fixture until every reader has at least 20 actual overlap samples.

```sh
CG_SCHEMA_OVERLAP_RECEIPT=/path/to/schema-overlap.json \
/path/to/cg_event_history --bench event_find_show_status_during_schema_replay_and_largest_migration
```

Build with the internal-support configuration shown above. Install the Nomic artifacts before
running; the reader does not download a model. The standalone schema-overlap-receipt/1 includes
idle and overlap p95, wrong answers, refusals, retries, actual page-boundary read charges, writer
charges, and whether any reader attempted to open writable state. Retry counters are sampled
at the real consumer transition/re-execution branches; writable-open attempts are observed at
the Store admission entry on each reader thread, including reader setup and baseline reads. Answers must match their complete idle
baseline, including source links. The process writes the receipt before checking the 200 ms find
and 50 ms show p95 limits.
