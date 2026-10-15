"""Exposed fixed300 only; pure typed scoring, no broader QA/SQL or config."""
import os
os.environ['FHIR_LOCAL_AZURE_CONFIG'] = ''
os.environ['FHIR_LOCAL_AZURE_DIAGNOSTICS_DIR'] = ''
import csv
import hashlib
import importlib.util
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
OUT = Path(__file__).parent
RUNS = [('original', 'midterm_20261012_full_v1', 223, 162),
        ('V2', 'verified_team_v2_300q', 231, 198),
        ('V3d', 'query_contract_v3d_300q', 238, 232),
        ('V4', 'plan_conditioned_v4_300q', 239, 218),
        ('V5', 'executable_critic_v5_300q', 237, 225),
        ('V6', 'independent_recomputation_v6_300q', 234, 231),
        ('V7', 'residual_trajectories_v7_300q', 238, 233)]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(path):
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buf = ''
        started = False
        ended = False
        while True:
            chunk = stream.read(1024 * 1024)
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not started:
                    if not buf:
                        break
                    assert buf[0] == '['
                    buf = buf[1:]
                    started = True
                    continue
                if buf.startswith(','):
                    buf = buf[1:]
                    continue
                if buf.startswith(']'):
                    ended = True
                    buf = buf[1:]
                    break
                if not buf:
                    break
                try:
                    row, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                yield row
                buf = buf[end:]
            if ended:
                assert not buf.strip() and not stream.read().strip()
                return
            if not chunk:
                raise ValueError('Incomplete result array')


def failed(row):
    return row.get('error') is not None and str(row['error']).strip() not in ('', 'nan')


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


spec = importlib.util.spec_from_file_location('pure_cross_version_scorer', REPO / 'utils/answer_scorer.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
assert sha(REPO / 'utils/answer_scorer.py') == 'ca02e5a8758f5bcef3eee08ab70f69fddf23917d267df6301e83275f337f3964'
expected = json.loads((REPO / 'manifests/split.json').read_text())['splits']['held_out']['question_ids']
assert len(expected) == len(set(expected)) == 300
families = {}
with (REPO / 'manifests/question_families.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        if row['question_id'] in set(expected):
            families[row['question_id']] = {'family': row['family'], 'template': row['template']}
assert set(families) == set(expected)
references = {}
matrix = {qid: {'question_id': qid, **families[qid], 'outcomes': {}} for qid in expected}
summary = {}
sources = {}
for version, directory, expected_single, expected_team in RUNS:
    for arm, target in [('single', expected_single), ('team', expected_team)]:
        filename = 'ordinary_single_agent.json' if arm == 'single' else 'ordinary_multi_agent.json'
        path = REPO / 'output/experiments/gpt-6-luna' / directory / filename
        label = version + '_' + arm
        counters = Counter()
        qids = []
        for row in iter_rows(path):
            qid = row['question_id']
            assert qid in matrix
            qids.append(qid)
            assert not row.get('attempt_history')
            reference = {k: row.get(k) for k in ('question', 'true_answer', 'assumption', 'patient_fhir_id', 'question_with_context')}
            if qid not in references:
                references[qid] = reference
            else:
                # Original context can differ after field-semantics improvements;
                # the exposed question/reference/patient cohort must still agree.
                for key in ('question', 'true_answer', 'patient_fhir_id'):
                    assert exact(reference[key], references[qid][key]), (label, key)
            typed = scorer.score_answer(row.get('agent_answer'), row['true_answer'], row['question'], families[qid]['template'])
            error = failed(row)
            correct = bool(not error and typed['typed_correct'])
            null = row.get('agent_answer') is None
            revision = row.get('revision') if isinstance(row.get('revision'), dict) else {}
            final_issues = revision.get('final_check_issues')
            error_type = None
            if error:
                text = str(row['error'])
                error_type = ('provider_timeout' if 'Timeout' in text or '408' in text else
                              'provider_connection' if 'Connection' in text else
                              'accounting' if 'accounting' in text.lower() or 'usage' in text.lower() else
                              'other_terminal')
            matrix[qid]['outcomes'][label] = {
                'correct': correct, 'terminal_error': error, 'error_category_heuristic': error_type,
                'answer': row.get('agent_answer'), 'null': null,
                'null_credit': correct and null, 'reference_type': typed.get('answer_type'),
                'limitation': row.get('limitation'), 'budget_stop': bool((row.get('budget') or {}).get('stop_reason')),
                'revision_final_issues_recorded': final_issues, 'evidence_items': len(row.get('evidence') or []),
                'revised_flag_not_mechanism': revision.get('revised')}
            counters.update({'correct': correct, 'wrong': not correct, 'errors': error,
                             'nonerror_wrong': not correct and not error, 'null': null,
                             'null_credit': correct and null, 'nonnull_wrong': not correct and not null,
                             'nonnull_correct': correct and not null,
                             'wrong_with_recorded_final_issues': not correct and bool(final_issues)})
        assert qids == expected
        assert counters['correct'] == target, (label, counters['correct'], target)
        sources[label] = {'path': str(path.relative_to(REPO)), 'sha256': sha(path), 'n': len(qids)}
        summary[label] = dict(counters)
        print(json.dumps({'arm': label, 'correct': counters['correct'], 'errors': counters['errors']}), flush=True)

team_labels = [v + '_team' for v, *_ in RUNS]
single_labels = [v + '_single' for v, *_ in RUNS]
late_labels = [v + '_team' for v in ['V3d', 'V4', 'V5', 'V6', 'V7']]
def wrongs(labels):
    return [qid for qid in expected if all(not matrix[qid]['outcomes'][label]['correct'] for label in labels)]
def rights(labels):
    return [qid for qid in expected if all(matrix[qid]['outcomes'][label]['correct'] for label in labels)]
sets = {'all7teams_wrong': wrongs(team_labels), 'all7singles_wrong': wrongs(single_labels),
        'all14arms_wrong': wrongs(team_labels + single_labels), 'all5recentteams_wrong': wrongs(late_labels),
        'all7teams_correct': rights(team_labels), 'all7singles_correct': rights(single_labels),
        'all14arms_correct': rights(team_labels + single_labels)}
histograms = {kind: dict(sorted(Counter(sum(matrix[qid]['outcomes'][label]['correct'] for label in labels) for qid in expected).items()))
              for kind, labels in [('team_correct_versions', team_labels), ('single_correct_versions', single_labels), ('all14_correct', team_labels + single_labels)]}
cross = {}
for left, right in itertools.combinations(team_labels, 2):
    n = Counter(('right' if matrix[qid]['outcomes'][left]['correct'] else 'wrong') + '_' +
                ('right' if matrix[qid]['outcomes'][right]['correct'] else 'wrong') for qid in expected)
    cross[left + ' -> ' + right] = dict(n)
paired = {}
for version, *_ in RUNS:
    a, b = version + '_single', version + '_team'
    c = Counter(('right' if matrix[qid]['outcomes'][a]['correct'] else 'wrong') + '_' +
                ('right' if matrix[qid]['outcomes'][b]['correct'] else 'wrong') for qid in expected)
    paired[version] = dict(c)
family_summary = {}
for family in sorted({v['family'] for v in families.values()}):
    qids = [qid for qid in expected if families[qid]['family'] == family]
    family_summary[family] = {'n': len(qids), 'all14_wrong': len(set(qids) & set(sets['all14arms_wrong'])),
                              'all5recentteams_wrong': len(set(qids) & set(sets['all5recentteams_wrong'])),
                              'correct': {label: sum(matrix[qid]['outcomes'][label]['correct'] for qid in qids) for label in team_labels + single_labels}}
result = {'scope': 'Same exposed fixed300, seven completed versions; all firsts/errors retained, adaptive descriptive comparisons only',
          'source_files': sources, 'scorer_sha256': sha(REPO / 'utils/answer_scorer.py'),
          'cohort_sha256': hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
          'summaries': summary, 'set_counts': {k: len(v) for k, v in sets.items()},
          'question_sets': sets, 'histograms': histograms, 'same_version_single_team': paired,
          'cross_team_versions': cross, 'family_summary': family_summary,
          'limits': ['Correctness persistence is not a root-cause label.', 'Cross-version changes bundle architecture, budgets, shared tools/prompts/source gates and stochastic draws.',
                     'Original/V2 audits do not retain all raw returned workspace resources; later gzip provenance is richer.',
                     'Saved outputs contain questions/references but no SQL field; no broader QA/SQL file was opened.',
                     'Unknown/error handling changed prospectively for both V6/V7 arms; errors remain incorrect under current unchanged typed scorer.',
                     'Current fixed300 is repeatedly inspected development; no new generalization or protected confirmation claim.']}
(OUT / 'cross_version_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
(OUT / 'exposed300_matrix.json').write_text(json.dumps({'references': references, 'questions': matrix}, indent=2, allow_nan=False) + '\n')
print(json.dumps({'set_counts': result['set_counts'], 'histograms': histograms, 'paired': paired, 'family_summary': family_summary}, indent=2))
