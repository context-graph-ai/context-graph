#!/usr/bin/env python3
"""Verify raw paired observations before applying existing product acceptance."""
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import subprocess

RECALL = 'event_find_recall_at_10_against_exact_on_the_real_model_fixture'
FRESH = 'event_fresh_graph_build'
IDENTITIES = ('embedding', 'corpus_content_sha256', 'held_out_sha256', 'model_artifacts_sha256', 'source_sha256')
REQUIRED = {('scoped', 'one'), ('scoped', 'few'), ('scoped', 'all'), ('single_context', 'one_context_all_passages')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def content(binding):
    require(all(binding.get(k) for k in IDENTITIES), 'incomplete fixture identity')
    require(binding['embedding'] == 'nomic-sessions', 'fixture is not real sessions')
    require(all(isinstance(binding[key], str) and re.fullmatch(r'[0-9a-f]{64}', binding[key]) for key in IDENTITIES[1:]), 'invalid fixture content digest')
    return {key: binding[key] for key in IDENTITIES}


def build_identity(variant, env):
    prefix = variant.upper()
    receipt = json.loads(pathlib.Path(env[prefix + '_BUILD_RECEIPT']).read_text())
    require(receipt.get('schema') == 1 and receipt.get('build_exit_code') == 0, 'invalid build receipt')
    for key in ('cg_sha', 'contextdb_sha'):
        require(receipt.get(key) == env[prefix + '_' + key.upper()], 'build receipt tuple mismatch')
    require(receipt.get('binary_sha256') == digest(env[prefix + '_BENCH_BINARY']), 'build executable identity mismatch')
    require(receipt.get('toolchain') == '1.94.1', 'compiler toolchain mismatch')
    compiler = receipt.get('rustc_vv', '')
    require(isinstance(compiler, str) and compiler.startswith('rustc 1.94.1 ')
            and all(re.search(r'^' + field + r': \S+', compiler, re.M) for field in ('host', 'commit-hash', 'LLVM version'))
            and re.search(r'^release: 1\.94\.1$', compiler, re.M), 'missing or wrong compiler identity')
    require(receipt.get('profile') == 'bench', 'build profile mismatch')
    require(receipt.get('cargo_incremental') == '0', 'build incremental setting mismatch')
    require(isinstance(receipt.get('features'), list) and 'default' in receipt['features'] and all(isinstance(value, str) for value in receipt['features']), 'build feature identity mismatch')
    require(receipt.get('rustflags') == '--cfg context_graph_internal_test_support', 'build flags mismatch')
    require(receipt.get('cargo_command') == ['bench', '--locked', '--no-run', '-p', 'context-graph', '--bench', 'cg_event_history'], 'build command mismatch')
    inputs = receipt.get('inputs', {})
    repository = env.get('PERFORMANCE_SOURCE_REPOSITORY', str(pathlib.Path.cwd()))
    revision = receipt['cg_sha']
    expected = {'source_tree': subprocess.check_output(
        ['git', '-C', repository, 'rev-parse', revision + '^{tree}'], text=True).strip()}
    for name in ('Cargo.lock', 'Cargo.toml', 'rust-toolchain.toml'):
        blob = subprocess.check_output(['git', '-C', repository, 'show', revision + ':' + name])
        expected[name] = hashlib.sha256(blob).hexdigest()
    require(inputs == expected, 'build input identity mismatch against selected revision')
    require(receipt.get('profile_overrides') == {}, 'unrecorded Cargo profile overrides')
    return receipt


def fixtures(env):
    # Requested dimensions locate fixtures; eligibility is assessed on stored corpora below.
    require(all(int(env[key]) > 0 for key in ('CG_EVENT_BENCH_TOTALS', 'CG_EVENT_BENCH_CONTEXTS', 'CG_EVENT_BENCH_RECALL_QUERIES')), 'invalid requested workload')
    pair = {}
    roots = []
    for variant in ('baseline', 'candidate'):
        prefix = variant.upper()
        root = pathlib.Path(env[prefix + '_FIXTURE_DIR']).resolve()
        roots.append(root)
        binary_digest = digest(env[prefix + '_BENCH_BINARY'])
        build_identity(variant, env)
        identities = {}
        names = {f"P{env['CG_EVENT_BENCH_TOTALS']}-K{env['CG_EVENT_BENCH_CONTEXTS']}-sessions", f"P{env['CG_EVENT_BENCH_TOTALS']}-K1-sessions", 'P10000-K1-sessions'}
        for name in names:
            binding = json.loads((root / name / 'binding.json').read_text())
            require(binding.get('build_executable_sha256') == binary_digest, 'sealed executable mismatch')
            identities[name] = content(binding)
        pair[variant] = identities
    require(roots[0] != roots[1], 'paired fixtures must have separate roots')
    require(pair['baseline'] == pair['candidate'], 'paired corpus, model or query identities differ')
    return pair


def raw_result(path, variant, env, sealed):
    records = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    builds = [r['build_identity'] for r in records if 'build_identity' in r]
    require(builds == [build_identity(variant, env)], 'missing or mismatched raw build identity')
    summaries = [r for r in records if r.get('bench') == RECALL and 'receipt' in r]
    require(len(summaries) == 1, 'missing or repeated recall summary')
    summary = summaries[0]
    receipt = summary['receipt']
    require(receipt.get('consumer_sha') == env[variant.upper() + '_CG_SHA'], 'embedded source tuple mismatch')
    require(receipt.get('engine_sha') == env[variant.upper() + '_CONTEXTDB_SHA'], 'embedded dependency tuple mismatch')
    require(receipt.get('binary_sha256') == digest(env[variant.upper() + '_BENCH_BINARY']), 'embedded executable digest mismatch')
    canonical = json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    require(summary.get('receipt_sha256') == hashlib.sha256(canonical.encode()).hexdigest(), 'embedded receipt digest mismatch')
    require(receipt.get('mode') == 'acceptance', 'missing acceptance selector')
    require(receipt.get('gate_eligible') is True and summary.get('gate_eligible') is True, 'raw result is ineligible')
    for key in ('process_failures', 'failures', 'not_gate_because'):
        require(summary.get(key) == [], 'raw failures or incomplete eligibility: ' + key)
    cells = [r for r in records if r.get('bench') == RECALL and 'scope' in r]
    require(len(cells) == len(REQUIRED) and {(r.get('cell'), r.get('scope')) for r in cells} == REQUIRED, 'missing or repeated selector cells')
    require(summary.get('cells') == len(cells), 'cell count mismatch')
    actual = []
    for cell in cells:
        key = f"P{env['CG_EVENT_BENCH_TOTALS']}-K{env['CG_EVENT_BENCH_CONTEXTS'] if cell['cell'] == 'scoped' else '1'}-sessions"
        require(content(cell['binding']) == sealed[key], 'measured fixture identity mismatch')
        require(cell['binding'].get('build_executable_sha256') == receipt['binary_sha256'], 'cell executable mismatch')
        require(cell.get('verdict') == {'pass': True, 'failures': []}, 'cell verdict failed')
        held = cell.get('held_out_queries', {})
        require(held.get('count') == int(env['CG_EVENT_BENCH_RECALL_QUERIES']) and held.get('in_corpus') == 0, 'query workload incomplete')
        require(set(cell.get('modes', {})) == ({'EXACT', 'AUTO', 'INDEXED'} if cell['cell'] == 'single_context' else {'EXACT', 'AUTO'}), 'query modes incomplete')
        count = int(env['CG_EVENT_BENCH_RECALL_QUERIES'])
        for name, mode in cell['modes'].items():
            require(mode.get('queries_answered') == count, 'mode query workload incomplete')
            require(mode.get('error_count') == 0 and mode.get('errors') == [], 'mode process errors')
            statement = mode.get('statement') or {}
            require(statement.get('samples') == count, 'mode latency samples incomplete')
            for metric in ('p50_ms', 'p95_ms', 'p99_ms', 'max_ms', 'mean_ms'):
                value = statement.get(metric)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'missing measured latency: ' + metric)
            recalls = mode.get('query_recalls_at_10_against_exact')
            require(isinstance(recalls, list) and len(recalls) == (0 if name == 'EXACT' else count), 'mode recall workload incomplete')
            require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in recalls), 'invalid measured recalls')
            work = mode.get('distance_evaluations') or {}
            require(all(type(work.get(key)) in (int, float) and math.isfinite(work[key]) and work[key] >= 0
                        for key in ('graph_total', 'native_total', 'per_query_mean', 'graph_per_query_max', 'native_per_query_max')), 'distance workload incomplete')
            if name != 'EXACT':
                require(mode.get('work_against_exact', {}).get('paired_queries') == count, 'paired query workload incomplete')
        actual.append({k: cell[k] for k in ('cell', 'scope', 'binding')})
    require(receipt.get('cells') == actual, 'embedded selector inventory mismatch')
    corpora = summary.get('stored_corpora', [])
    require(len(corpora) == 2 and {r.get('cell') for r in corpora} == {'scoped', 'single_context'}, 'stored workload incomplete')
    for corpus in corpora:
        expected_scopes = {s for c, s in REQUIRED if c == corpus['cell']}
        require(set(corpus.get('scopes', [])) == expected_scopes, 'stored scopes incomplete')
        require(corpus.get('queries') == int(env['CG_EVENT_BENCH_RECALL_QUERIES']) and corpus['queries'] >= 100, 'stored query count mismatch')
        require(corpus.get('requested_passages') == int(env['CG_EVENT_BENCH_TOTALS']), 'requested workload mismatch')
        contexts = int(env['CG_EVENT_BENCH_CONTEXTS']) if corpus['cell'] == 'scoped' else 1
        require(corpus.get('requested_contexts') == contexts and corpus.get('stored_contexts') == (8 if corpus['cell'] == 'scoped' else 1), 'stored contexts mismatch')
        require(corpus.get('stored_passages', 0) >= 10000, 'stored passages incomplete')
    fresh = [r for r in records if r.get('bench') == FRESH and 'fixture' in r]
    sources = [r['source'] for r in records if r.get('bench') == FRESH and 'source' in r]
    require(len(fresh) == 1 and len(sources) == 1, 'fresh graph workload incomplete')
    source = sources[0]
    require(source.get('binding_verified') is True and source.get('vectors') == 10000, 'fresh source incomplete')
    require(source.get('corpus_content_sha256') == sealed['P10000-K1-sessions']['corpus_content_sha256'], 'fresh source identity mismatch')
    require(pathlib.Path(source['directory']).resolve() == pathlib.Path(env[variant.upper() + '_FIXTURE_DIR']).resolve() / 'P10000-K1-sessions', 'fresh source path mismatch')
    fixture = fresh[0]['fixture']
    require(fixture.get('source_binding_verified') is True and fixture.get('restart_indexed') is True, 'fresh graph verification incomplete')
    require(fixture.get('current_passages') == 10000 and fixture.get('contexts') == 1 and fixture.get('first_failure') is None and fixture.get('remaining_partitions') == 0 and fixture.get('built_partitions') == 1, 'fresh graph build incomplete')
    require(fixture.get('fresh_graph_precondition', {}).get('asserted') is True, 'fresh precondition missing')
    seconds = fixture.get('fresh_graph_build_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'fresh graph measurement missing')
    return {(c['cell'], c['scope']): content(c['binding']) for c in cells}


def validate(acceptance_path, result_dir, env):
    for key in ('BASELINE_CG_SHA', 'CANDIDATE_CG_SHA', 'BASELINE_CONTEXTDB_SHA', 'CANDIDATE_CONTEXTDB_SHA'):
        require(re.fullmatch(r'[0-9a-f]{40}', env[key]) is not None, 'selected source tuple is not exact')
    sealed = fixtures(env)
    receipt = json.loads(pathlib.Path(acceptance_path).read_text())
    identities = []
    for variant in ('baseline', 'candidate'):
        path = pathlib.Path(result_dir) / (variant + '.jsonl')
        require(receipt.get(variant + '_sha256') == digest(path), 'paired acceptance does not bind raw results')
        identities.append(raw_result(path, variant, env, sealed[variant]))
        invocation = json.loads((pathlib.Path(result_dir) / (variant + '-process.json')).read_text())
        require(invocation == {'binary_sha256': digest(env[variant.upper() + '_BENCH_BINARY']), 'selectors': [RECALL, FRESH], 'exit_codes': [0, 0]}, 'selected process receipt mismatch')
    builds = [build_identity(variant, env) for variant in ('baseline', 'candidate')]
    require(all(builds[0][key] == builds[1][key] for key in ('toolchain', 'rustc_vv', 'profile', 'features', 'rustflags', 'profile_overrides', 'cargo_incremental', 'cargo_command')), 'paired compiler or build identity mismatch')
    require(identities[0] == identities[1], 'measured pair identities differ')
    for key in ('baseline_cg_sha', 'candidate_cg_sha', 'baseline_contextdb_sha', 'candidate_contextdb_sha'):
        require(receipt.get(key) == env[key.upper()], 'paired acceptance tuple mismatch')
    for key in ('workload_complete', 'paired_latency_accepted', 'distance_work_accepted', 'recall_accepted', 'fresh_graph_accepted'):
        require(receipt.get(key) is True, 'product acceptance incomplete: ' + key)


if __name__ == '__main__':
    try:
        if sys.argv[1] == '--fixtures':
            fixtures(os.environ)
        else:
            validate(sys.argv[1], sys.argv[2], os.environ)
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        raise SystemExit('PENDING: ' + str(error)) from error
