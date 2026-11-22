"""Independent numerical reconciliation; no scorer, inference, or raw archive imports."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/v3'
OUT = Path(__file__).resolve().parent
ORIG = ROOT / 'benchmarks/v3-full-run-20261015/completed_v3'
inputs = {}
checks = Counter()
findings = []
details = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def check(ok, label):
    checks['checks'] += 1
    if not ok:
        findings.append(label)


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=1e-10, rel_tol=2e-11)
    return a == b


def eq(a, b, label):
    check(same(a, b), label + ('' if same(a, b) else f': recomputed {a!r}, recorded {b!r}'))


def valid(v):
    return type(v) in (float, int) and math.isfinite(v)


def quantile(v, percent):
    v = sorted(v)
    i = (len(v) - 1) * percent / 100
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def describe(vals):
    good = [v for v in vals if valid(v)]
    d = dict(total=math.fsum(good), mean=statistics.fmean(good),
             median=statistics.median(good), p95=quantile(good, 95),
             stdev=statistics.stdev(good) if len(good) > 1 else None,
             min=min(good), max=max(good)) if good else dict.fromkeys(
                 ['total', 'mean', 'median', 'p95', 'stdev', 'min', 'max'])
    return len(good), len(vals) - len(good), d


def bootstrap(vals, patients):
    # Independent grouping/aggregation; same frozen RNG and multinomial recipe.
    groups = defaultdict(list)
    for value, patient in zip(vals, patients):
        groups[patient].append(value)
    # Full FHIR identifiers sort lexically; archived development uses numeric
    # local cluster labels and its retained derivation sorts these numerically.
    order = sorted(groups)
    sums = np.array([sum(groups[p]) for p in order], dtype=float)
    counts = np.array([len(groups[p]) for p in order], dtype=float)
    weights = np.random.default_rng(5106).multinomial(
        len(order), np.full(len(order), 1 / len(order)), size=10000)
    means = (weights @ sums) / (weights @ counts)
    low, high = np.percentile(means, [2.500000000000002, 97.5])
    return dict(n=len(vals), patients=len(order), estimate=statistics.fmean(vals),
                ci_low=float(low), ci_high=float(high))


def ci_check(calc, stored, label):
    for key in ['n', 'patients', 'estimate', 'ci_low', 'ci_high']:
        eq(calc[key], stored[key], label + '/' + key)


MEASURES = ('latency_seconds', 'estimated_cost_usd', 'reported_cost_usd',
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'cached_prompt_tokens', 'uncached_prompt_tokens', 'repeated_prompt_tokens',
            'llm_calls', 'tool_calls_executed', 'tool_calls_recorded', 'evidence_count',
            'evidence_valid', 'valid_evidence_fraction', 'evidence_precision', 'evidence_recall')


def full_stats(data, summary, label):
    eq(len(data), summary['questions'], label + '/questions')
    eq(sum(r['correct'] for r in data), summary['correct'], label + '/correct')
    eq(sum(r['abstained'] for r in data), summary['abstentions'], label + '/empty')
    for metric in MEASURES:
        n, missing, calc = describe([r[metric] for r in data])
        recorded = summary['metrics'][metric]
        eq(n, recorded['observed_n'], label + '/' + metric + '/observed')
        eq(missing, recorded['missing_n'], label + '/' + metric + '/missing')
        for k, val in calc.items():
            eq(val, recorded['observed_only'][k], label + '/' + metric + '/observed_' + k)
            eq(None if missing else val, recorded[k], label + '/' + metric + '/' + k)
    patient = [r['patient_fhir_id'] for r in data]
    ci_check(bootstrap([int(r['correct']) for r in data], patient), summary['accuracy'], label + '/accuracy_CI')
    for metric in ['latency_seconds', 'estimated_cost_usd', 'total_tokens']:
        ci_check(bootstrap([r[metric] for r in data], patient), summary['metrics'][metric]['mean_patient_bootstrap'], label + '/' + metric + '_CI')
    for r in data:
        q = label + '/' + r['question_id']
        p, c, k = [r[key] for key in ['prompt_tokens', 'completion_tokens', 'cached_prompt_tokens']]
        eq(p + c, r['total_tokens'], q + '/token_balance')
        check(0 <= k <= p, q + '/cached_subset')
        eq(p - k, r['uncached_prompt_tokens'], q + '/uncached_balance')
        eq(((p-k)*.1 + k*.01 + c*.5)/1e6, r['estimated_cost_usd'], q + '/cost')
        calls = r['per_call_usage']
        eq(len(calls), r['llm_calls'], q + '/call_count')
        for key, agg in [('prompt', p), ('completion', c), ('cached', k)]:
            eq(sum(x[key] for x in calls), agg, q + '/call_' + key)
        for x in calls:
            check(x['usage_available'] is True and 0 <= x['cached'] <= x['prompt'], q + '/usage_balance')
    return Counter((r['correct'], r['abstained']) for r in data)


def dev_stats(data, pair_data, stats):
    arms = {arm: [r for r in data if r['arm'] == arm] for arm in ['single_agent', 'multi_agent_v3']}
    for arm, rr in arms.items():
        eq(len(rr), 300, 'development/' + arm + '/n')
        for r in rr:
            p, c, k = [r[x] for x in ['prompt_tokens_measured', 'completion_tokens_measured', 'cached_prompt_tokens_measured']]
            eq(p+c, r['total_tokens_measured'], 'development/token_balance')
            check(0 <= k <= p, 'development/cached_subset')
            expected = ((p-k)*.1+k*.01+c*.5)/1e6
            eq(expected, r['known_usage_proxy_lower_bound_usd'], 'development/known_cost')
            eq(expected if r['provider_usage_complete'] else None, r['proxy_cost_usd'], 'development/complete_cost')
        for row in stats['accuracy']:
            if row['arm'] != arm:
                continue
            group = [r for r in rr if row['family'] == 'all' or r['family'] == row['family']]
            eq(len(group), row['n'], 'development/accuracy_n')
            eq(sum(r['correct'] for r in group), row['correct'], 'development/accuracy_correct')
            eq(statistics.fmean(int(r['correct']) for r in group), row['estimate'], 'development/accuracy_estimate')
            ci_check(bootstrap([int(r['correct']) for r in group], [r['patient_cluster'] for r in group]),
                     row, 'development/frozen_accuracy_CI')
    for row in stats['measurement_statistics']:
        group = [r for r in arms[row['arm']] if row['family'] == 'all' or r['family'] == row['family']]
        vals = [r[row['metric']] for r in group]
        n, missing, calc = describe(vals)
        for k, value in [('n_expected', len(vals)), ('n_observed', n), ('n_missing', missing),
                         ('sum_observed', calc['total']), ('total_complete', None if missing else calc['total'])]:
            eq(value, row[k], 'development/' + row['metric'] + '/' + k)
        for key in ['mean', 'median', 'p95', 'min', 'max', 'stdev']:
            eq(calc[key], row['sample_sd' if key == 'stdev' else key], 'development/' + row['metric'] + '/' + key)
        good = [v for v in vals if valid(v)]
        for key, pct in [('p25', 25), ('p75', 75), ('p90', 90), ('p99', 99)]:
            eq(quantile(good, pct) if good else None, row[key], 'development/' + row['metric'] + '/' + key)
    lookup = {arm: {r['question_id']: r for r in rr} for arm, rr in arms.items()}
    eq(set(lookup['single_agent']), set(lookup['multi_agent_v3']), 'development/matched_ids')
    eq(len(pair_data), 300, 'development/pairs_n')
    cross = Counter()
    for r in pair_data:
        a, b = lookup['multi_agent_v3'][r['question_id']], lookup['single_agent'][r['question_id']]
        eq(a['patient_cluster'], b['patient_cluster'], 'development/matched_patient')
        eq(r['team_correct'], a['correct'], 'development/team_correct')
        eq(r['single_correct'], b['correct'], 'development/single_correct')
        eq(r['correct_difference'], int(a['correct'])-int(b['correct']), 'development/correct_sign')
        cross[a['correct'], b['correct']] += 1
        for metric in ['latency_seconds', 'total_tokens_measured', 'llm_calls', 'tool_calls_run', 'proxy_cost_usd']:
            expected = a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None
            eq(r['difference_' + metric], expected, 'development/pair_sign/' + metric)
    for row in stats['paired_measurement_statistics']:
        complete = [r for r in pair_data if r['difference_' + row['metric']] is not None]
        vals = [r['difference_' + row['metric']] for r in complete]
        n, _, calc = describe(vals)
        for key in ['mean', 'median', 'p95', 'min', 'max', 'stdev']:
            eq(calc[key], row['sample_sd' if key == 'stdev' else key], 'development/paired/' + row['metric'] + '/' + key)
        eq(n, row['n_observed'], 'development/paired/n')
        c = bootstrap(vals, [r['patient_cluster'] for r in complete])
        stored = row['patient_block_mean_ci95']
        ci_check(c, {**stored, 'estimate': stored['mean']}, 'development/pairedCI/' + row['metric'])
    # Verify, but do not replace, the historical frozen accuracy endpoints.
    details['development_crosstab'] = {str(k): v for k, v in cross.items()}
    expected = stats['paired_accuracy']
    for k, val in [('both_correct', cross[True, True]), ('both_incorrect', cross[False, False]),
                   ('wins', cross[True, False]), ('losses', cross[False, True])]:
        eq(val, expected[k], 'development/paired_accuracy/' + k)
    ci_check(bootstrap([r['correct_difference'] for r in pair_data], [r['patient_cluster'] for r in pair_data]),
             expected, 'development/frozen_paired_accuracy_CI')
    return lookup


def main():
    single = rows(BASE/'full_single_agent/per_question_measurements_and_scores.jsonl')
    matched = rows(BASE/'full_single_agent/matched_v3_measurements_and_scores.jsonl')
    original_raw = rows(ORIG/'per_question.jsonl')
    # Keep only numeric/identity fields from the original full table; no answer values used.
    original = [{k: r[k] for k in (*MEASURES, 'question_id', 'patient_fhir_id', 'correct', 'abstained', 'per_call_usage')} for r in original_raw]
    del original_raw
    summaries = {'single': load(BASE/'full_single_agent/single_agent_summary.json'),
                 'matched_v3': load(BASE/'full_single_agent/matched_v3_summary.json'),
                 'original_v3': load(ORIG/'summary.json')['arms']['multi_agent']}
    for label, data in [('single', single), ('matched_v3', matched), ('original_v3', original)]:
        ct = full_stats(data, summaries[label], label)
        details[label] = {'n': len(data), 'correct': sum(r['correct'] for r in data),
                          'patients': len(set(r['patient_fhir_id'] for r in data)),
                          'correctness_empty_flag_crosstab': {str(k): v for k, v in ct.items()}}
    ct = load(BASE/'full_single_agent/correctness_empty_flag_crosstab.json')
    expected = Counter((r['correct'], r['abstained']) for r in single)
    for r in ct:
        eq(expected[r['correct'], r['syntactic_empty_no_value_flag']], r['questions'], 'single/empty_crosstab')
    aa, bb, oo = [{r['question_id']: r for r in data} for data in [single, matched, original]]
    eq(len(aa), 2929, 'single/unique_ids')
    eq(len(bb), 2929, 'matched_v3/unique_ids')
    eq(len(oo), 2931, 'original_v3/unique_ids')
    eq(set(aa), set(bb), 'full/matched_ids')
    eq(set(oo)-set(bb), {'05a8454a6a71435a557db2cc','0eb69f6e47ab63c055e2df90'}, 'full/excluded_ids')
    for q in bb:
        eq(aa[q]['patient_fhir_id'], bb[q]['patient_fhir_id'], 'full/matched_patient')
        for key in (*MEASURES, 'correct', 'patient_fhir_id', 'abstained'):
            eq(bb[q][key], oo[q][key], 'full/matched_original/' + key)
    pair = load(BASE/'full_single_agent/paired_comparison.json')
    ids = sorted(aa)
    cross = Counter((aa[q]['correct'], bb[q]['correct']) for q in ids)
    for key, value in [('left_wins', cross[True,False]), ('right_wins', cross[False,True]),
                       ('both_correct', cross[True,True]), ('both_wrong', cross[False,False])]:
        eq(value, pair[key], 'full/pair/' + key)
    for key in ['correct', 'latency_seconds', 'estimated_cost_usd', 'total_tokens']:
        vals = [int(aa[q][key])-int(bb[q][key]) if key == 'correct' else aa[q][key]-bb[q][key] for q in ids]
        calc = bootstrap(vals, [aa[q]['patient_fhir_id'] for q in ids])
        recorded = pair['accuracy_difference' if key == 'correct' else key + '_difference_observed_pairs']
        ci_check(calc, recorded, 'full/pair_CI/' + key)
        details['full_pair_' + key] = calc
    details['full_pair_crosstab'] = {str(k): v for k, v in cross.items()}
    scope = load(BASE/'full_single_agent/draft_final_scope_reconciliation.json')['arms']
    for label, rr in [('single_agent', single), ('multi_agent_v3_matched', matched)]:
        calc = {
            'all_final_correct':sum(r['correct'] for r in rr),
            'draft_present_questions':sum(r['draft_present'] for r in rr),
            'draft_present_final_correct':sum(r['draft_present'] and r['correct'] for r in rr),
            'draft_absent_final_correct':sum(not r['draft_present'] and r['correct'] for r in rr),
            'draft_present_draft_correct':sum(r['draft_present'] and r['draft_correct'] is True for r in rr),
            'draft_present_wrong_to_right':sum(r['draft_present'] and r['draft_correct'] is False and r['correct'] for r in rr),
            'draft_present_right_to_wrong':sum(r['draft_present'] and r['draft_correct'] is True and not r['correct'] for r in rr),
            'draft_absent_questions':sum(not r['draft_present'] for r in rr),
        }
        eq(calc, scope[label], 'full/draft_scope/' + label)
    families = load(BASE/'full_single_agent/paired_by_family.json')
    for f in families:
        subset = [q for q in ids if aa[q]['family']==f['group']]
        eq(len(subset), f['matched_questions'], 'full/family_n/' + f['group'])
        calc = bootstrap([int(aa[q]['correct'])-int(bb[q]['correct']) for q in subset], [aa[q]['patient_fhir_id'] for q in subset])
        ci_check(calc, f['accuracy_difference'], 'full/family_accuracy_CI/' + f['group'])
    sensitivity = load(BASE/'cost_cache_sensitivity.json')
    for label, summary in [('single_agent', summaries['single']), ('multi_agent_v3', summaries['matched_v3'])]:
        m = summary['metrics']; a = sensitivity['arms'][label]
        p, k, c = [m[x]['total'] for x in ['prompt_tokens','cached_prompt_tokens','completion_tokens']]
        eq(k/p, a['cached_prompt_fraction'], 'sensitivity/cached_fraction/' + label)
        eq((p*.1+c*.5)/1e6, a['no_cache_total_usd'], 'sensitivity/no_cache_total/' + label)
    eq(sensitivity['arms']['multi_agent_v3']['no_cache_total_usd']/sensitivity['arms']['single_agent']['no_cache_total_usd'],
       sensitivity['v3_to_single_no_cache_cost_ratio'], 'sensitivity/no_cache_ratio')
    development = rows(BASE/'development/per_question.jsonl')
    devpairs = rows(BASE/'development/paired_measurements.jsonl')
    devlookup = dev_stats(development, devpairs, load(BASE/'development/statistics.json'))
    labels = rows(BASE/'full_single_agent/exploratory_development_exposure_labels.jsonl')
    overlap = load(BASE/'full_single_agent/exploratory_development_exposure_overlap.json')
    eq({r['question_id'] for r in labels}, set(aa), 'exposure/matched_ids')
    fresh_ids = set(devlookup['single_agent'])
    eq(sum(r['in_fresh300_question_set'] for r in labels), len(fresh_ids & set(aa)), 'exposure/fresh_questions')
    eq(sum(r['in_fresh300_question_set'] for r in labels), 300, 'exposure/fresh_questions300')
    seen = {r['patient_fhir_id'] for r in labels if r['patient_seen_in_fresh300']}
    original_exposed = {r['patient_fhir_id'] for r in labels if r['frozen_source_split_exposure']=='development_exposed_patient'}
    union = seen | original_exposed
    eq(len(seen), 58, 'exposure/fresh_patients')
    eq(len(original_exposed), 35, 'exposure/original_patients')
    eq(len(union), 93, 'exposure/union_patients')
    eq(len(union), overlap['known_union_patients'], 'exposure/recorded_union')
    eq(sum(r['patient_seen_in_fresh300'] for r in labels), 1678, 'exposure/question_count')
    check(all(r['frozen_source_split_exposure']=='held_out_patient_descriptive' for r in labels if r['patient_seen_in_fresh300']), 'exposure/fresh_all_original_heldout')
    details['exposure'] = {'known_union_patients': len(union), 'total_patients':94, 'fresh_patients':len(seen), 'original_patients':len(original_exposed), 'fresh_question_overlap':300}
    source_comparison = load(BASE/'development/source_comparison.json')
    source_bytes = {}
    for row in source_comparison['comparison_files']:
        name = row['path']
        eq(row['full_sha256']==row['development_sha256'], row['equal'], 'source/equality_flag/'+name)
        check(name.startswith(('agent/','tools/','utils/','scripts/')) or name in ('run_agent.py','fhir_client.py','requirements.txt'), 'source/allowlisted_path')
        if not (name.startswith(('agent/','tools/','utils/','scripts/')) or name in ('run_agent.py','fhir_client.py','requirements.txt')):
            continue
        for label, field in [('full','full_source_commit'), ('development','development_source_commit')]:
            ref = source_comparison[field]
            data = subprocess.run(['git','show',ref+':'+name], cwd=ROOT, check=True, capture_output=True).stdout
            source_bytes[(ref,name)] = data
            eq(hashlib.sha256(data).hexdigest(), row[label+'_sha256'], 'source/'+label+'/'+name)
    differing = [r['path'] for r in source_comparison['comparison_files'] if not r['equal']]
    eq(len(source_comparison['comparison_files']), 57, 'source/declared_scope57')
    eq(differing, ['agent/query_contract_agent.py'], 'source/one_shared_difference')
    for ref, recorded in source_comparison['reproduced_fingerprints'].items():
        digest = hashlib.sha256()
        for name in sorted(source_comparison['fingerprint_source_files']):
            data = source_bytes.get((ref,name))
            if data is None:
                data = subprocess.run(['git','show',ref+':'+name], cwd=ROOT, check=True, capture_output=True).stdout
            digest.update(name.encode()); digest.update(b'\0'); digest.update(data); digest.update(b'\0')
        digest.update(json.dumps(source_comparison['same_dependency_version_map'],sort_keys=True).encode())
        eq(digest.hexdigest(), recorded, 'source/reproduced_fingerprint/'+ref)
    details['source_comparison'] = {'shared_files':57,'equal':56,'different':differing,
                                    'fingerprint_source_files':42,'recorded_dependency_versions':21,
                                    'both_recorded_fingerprints_reproduced':True}
    # No archives are read while packaging is in progress. Check tracked implementation delta only.
    diff = subprocess.run(['git','diff','--name-only','f71a8d9'], cwd=ROOT, check=True, capture_output=True, text=True)
    details['tracked_diff_from_f71a8d9'] = diff.stdout.splitlines()
    check(not diff.stdout.strip(), 'No tracked inference/source changes from f71a8d9')
    for name, digest in inputs.items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Input changed during review: '+name)
    output = {'status':'PASS' if not findings else 'FINDINGS', 'checks':checks['checks'],
              'findings':findings, 'details':details, 'input_sha256':inputs,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime':{'numpy':np.__version__, 'bootstrap_replicates':10000,'seed':5106},
              'limits':['No scorer is rerun; retained correctness flags are independently counted and cross-tabulated.',
                        'Patient-cluster bootstrap independently aggregated with the frozen multinomial/default_rng recipe; not repeated-run or provider uncertainty.',
                        'Development accuracy intervals are unchanged copied historical results; both accuracy and measurement intervals were independently verified using numerically ordered local cluster labels.',
                        'Cache numeric balance and cost formula checks do not prove original cache-field availability; missing/cache provenance limits remain.',
                        'Raw archives and final package manifest are owned by other agents and were not read or validated while being written.',
                        'No provider/model calls, config/secret reads, inference changes, commits or pushes.']}
    (OUT/'numeric_qa.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'status':output['status'],'checks':checks['checks'],'findings':findings[:20]}))


if __name__ == '__main__':
    main()
