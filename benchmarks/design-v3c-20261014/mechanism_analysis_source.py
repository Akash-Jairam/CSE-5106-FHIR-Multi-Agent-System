"""Completed 300-question V3 mechanism ledger. Never imported by benchmark agents.

Scoring is post-hoc only, using the unchanged source scorer and the first
recorded attempt. Incomplete cohorts are rejected before any scoring/output.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

RESULT_FIELDS = ('agent_answer', 'evidence', 'limitation', 'tokens', 'tool_calls',
                 'latency', 'agent_fhir_resources', 'trace', 'error', 'usage',
                 'revision', 'budget', 'audit_artifacts', 'retrieved_fhir_resources')
PHASES = ('contract', 'solver', 'audit', 'targeted')
VERDICTS = ('PASS', 'FAIL', 'UNKNOWN', 'N/A')
CLEAN_POLICIES = {'source_valid_primary_preserved', 'source_checked_targeted_local_constraint'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_rows(path, chunk_chars=1024 * 1024):
    """Stream the same top-level result-array format as the benchmark runner."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer, position, ended = '', 0, False
        def fill():
            nonlocal buffer, position, ended
            fragment = stream.read(chunk_chars)
            buffer, position = buffer[position:] + fragment, 0
            ended = not fragment
        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or ended:
                    return
                fill()
        whitespace()
        if position >= len(buffer) or buffer[position] != '[':
            raise ValueError('Result must be a JSON array')
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == ']':
            position += 1
        else:
            while True:
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if ended:
                            raise ValueError('Incomplete result JSON') from None
                        fill()
                position = end
                yield record
                whitespace()
                if position < len(buffer) and buffer[position] == ']':
                    position += 1
                    break
                if position >= len(buffer) or buffer[position] != ',':
                    raise ValueError('Invalid result array separator')
                position += 1
                whitespace()
        whitespace()
        if position < len(buffer):
            raise ValueError('Trailing result JSON')


def failed(row):
    error = row.get('error')
    return error is not None and str(error).strip() not in ('', 'nan')


def first_attempt(row):
    history = row.get('attempt_history')
    if history is None:
        history = []
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError('Malformed attempt history')
    selected = history[0] if history else row
    # Match the existing first-attempt rule without inheriting later result data.
    result = {**row, **{key: selected.get(key) for key in RESULT_FIELDS}}
    result.pop('attempt_history', None)
    return result, len(history)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def issue_types(items):
    if items is not None and not isinstance(items, list):
        return ['malformed_issue_list']
    result = []
    for item in items or []:
        if isinstance(item, dict):
            result.append(item.get('type', 'malformed_issue'))
        elif isinstance(item, str):
            result.append(item.partition(':')[0].strip() or 'unstructured_issue')
        else:
            result.append('malformed_issue')
    return sorted(set(result))


def same_json(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False


def artifact(value):
    return isinstance(value, dict) and not value.get('error') and all(
        key in value for key in ('answer', 'evidence', 'operation', 'contributing_ids', 'limitation'))


def audit_status(revision):
    audit = revision.get('audit')
    error = revision.get('audit_error') or revision.get('contract_error')
    if isinstance(audit, dict) and isinstance(audit.get('checks'), list):
        checks = audit['checks']
        if all(isinstance(item, dict) and item.get('verdict') in VERDICTS for item in checks):
            return 'valid_json', Counter(item['verdict'] for item in checks)
        return 'malformed_record', Counter()
    if error:
        text = str(error).lower()
        if 'requested tools' in text:
            return 'forbidden_tool', Counter()
        if 'program exceeds' in text:
            return 'program_cap', Counter()
        if any(word in text for word in ('budget', 'input tokens', 'token ceiling')):
            return 'budget_or_guard', Counter()
        if any(word in text for word in ('provider', 'unavailable', 'timeout', 'retries', 'badrequest')):
            return 'unavailable', Counter()
        return 'malformed_or_failed', Counter()
    if revision.get('contract') is None:
        return 'contract_unavailable_or_not_run', Counter()
    return 'not_run', Counter()


def terminal_kind(row):
    if not failed(row):
        return None
    text = str(row.get('error')).lower()
    for words, name in [ (('timeout', 'timed out'), 'timeout'),
                         (('budget', 'token ceiling', 'model-call allocation'), 'budget'),
                         (('input tokens', 'context'), 'input_guard'),
                         (('json serializable', 'serialization', 'serialize'), 'transport'),
                         (('badrequest', 'sdk', 'retries', 'provider'), 'provider'),
                         (('source', 'scope', 'field'), 'source'),
                         (('contract', 'final computation'), 'execution_contract') ]:
        if any(word in text for word in words):
            return name
    return 'other'


def trace_stats(trace):
    counts = Counter()
    arguments = {}
    if not isinstance(trace, list):
        return {'trace_malformed': 1}
    for message in trace:
        if not isinstance(message, dict):
            continue
        if message.get('model_call') is False and message.get('phase') == 'solver_boundary':
            counts['host_final_boundaries'] += 1
        for call in message.get('tool_calls') or []:
            if isinstance(call, dict):
                function = call.get('function') or {}
                try:
                    args = json.loads(function.get('arguments') or '{}')
                except (ValueError, TypeError):
                    args = {}
                arguments[call.get('id')] = (function.get('name'), args)
        if message.get('role') != 'tool':
            continue
        name = message.get('name')
        if name == 'check_fields':
            counts['source_check_replies'] += 1
        if name != 'execute_python_code':
            continue
        try:
            result = json.loads(message.get('content') or '{}')
        except (ValueError, TypeError):
            counts['python_reply_non_json'] += 1
            continue
        if not isinstance(result, dict):
            continue
        mode = result.get('mode') or (arguments.get(message.get('tool_call_id'), (None, {}))[1]).get('mode')
        if mode == 'final' and 'answer' in result and not result.get('error'):
            counts['worker_valid_final_replies'] += 1
        if result.get('error_type') == 'final_contract':
            counts['worker_final_contract_errors'] += 1
    return dict(counts)


def phase_record(revision, phase_caps):
    entries = revision.get('phases') or []
    if not isinstance(entries, list):
        return {}, ['malformed_phase_records']
    result, anomalies = {}, []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('phase') not in PHASES:
            anomalies.append('malformed_phase_record')
            continue
        name = entry['phase']
        if name in result:
            anomalies.append('duplicate_phase:' + name)
        result[name] = entry
        if name in phase_caps and (entry.get('token_cap'), entry.get('tool_cap')) != tuple(phase_caps[name]):
            anomalies.append('phase_cap_mismatch:' + name)
        for measured, cap in (('tokens', 'token_cap'), ('tool_calls', 'tool_cap')):
            if not finite(entry.get(measured)) or not finite(entry.get(cap)):
                anomalies.append('unknown_phase_measure:' + name + ':' + measured)
            elif entry[measured] > entry[cap]:
                anomalies.append('phase_cap_exceeded:' + name + ':' + measured)
    return result, anomalies


def row_record(row, qid, family, score, phase_caps, identity):
    revision = row.get('revision') or {}
    if not isinstance(revision, dict):
        revision = {}
    draft, primary, final = (revision.get(name) for name in ('draft', 'primary', 'final'))
    correct_candidate = lambda candidate: bool(score(candidate.get('answer'), qid)['typed_correct']) if isinstance(candidate, dict) and 'answer' in candidate else None
    final_score = score(row.get('agent_answer'), qid)
    terminal = failed(row)
    final_correct = bool(not terminal and final_score['typed_correct'])
    policy = revision.get('release_policy')
    clean = not terminal and revision.get('final_check_issues') == []
    release_matches = (isinstance(final, dict) and 'answer' in final and 'evidence' in final
        and same_json(row.get('agent_answer'), final['answer'])
        and same_json(row.get('evidence'), final['evidence'])
        and same_json(row.get('limitation'), final.get('limitation')))
    source_valid = clean and release_matches and (policy in CLEAN_POLICIES and artifact(final) if revision.get('protocol') else True)
    audit_kind, verdict_counts = audit_status(revision)
    phases, anomalies = phase_record(revision, phase_caps if revision.get('protocol') else {})
    if clean and (policy in CLEAN_POLICIES or not revision.get('protocol')) and not release_matches:
        anomalies.append('released_artifact_mismatch')
    tokens, budget = row.get('tokens') or {}, row.get('budget') or {}
    if finite(tokens.get('total')) and finite(identity.get('max_total_tokens')) and tokens['total'] > identity['max_total_tokens']:
        anomalies.append('global_token_cap_exceeded')
    if finite(budget.get('tool_calls_run')) and finite(identity.get('max_tool_calls')) and budget['tool_calls_run'] > identity['max_tool_calls']:
        anomalies.append('global_tool_cap_exceeded')
    if phases and all(finite(entry.get('tokens')) for entry in phases.values()) and finite(tokens.get('total')):
        if sum(entry['tokens'] for entry in phases.values()) != tokens['total']:
            anomalies.append('phase_token_sum_mismatch')
    if phases and all(finite(entry.get('tool_calls')) for entry in phases.values()) and finite(budget.get('tool_calls_run')):
        if sum(entry['tool_calls'] for entry in phases.values()) != budget['tool_calls_run']:
            anomalies.append('phase_tool_sum_mismatch')
    proof = revision.get('targeted_proof') or {}
    if not isinstance(proof, dict):
        proof = {'status': 'malformed'}
    targeted = bool(revision.get('targeted_revised'))
    primary_correct, draft_correct = correct_candidate(primary), correct_candidate(draft)
    unresolved = revision.get('unresolved_scope') or {}
    if not isinstance(unresolved, dict):
        anomalies.append('malformed_unresolved_scope')
        unresolved = {}
    reason_counts = Counter(str(value) for value in unresolved.values())
    primary_revision = revision.get('primary_revision') or {}
    if not isinstance(primary_revision, dict):
        primary_revision = {}
    budget_scope = ('fatal_accounting' if any(bool(entry.get('fatal')) for entry in phases.values()) else
                    'aggregate' if budget.get('stopped') else
                    'local_phase' if any(bool(entry.get('errors')) for entry in phases.values()) else 'none')
    record = {'question_id': qid, 'family': family['family'], 'protocol': revision.get('protocol'),
        'terminal_error': terminal, 'terminal_error_kind': terminal_kind(row),
        'correct': final_correct, 'typed_reason': final_score['typed_reason'],
        'reference_empty': final_score['answer_type'] == 'empty',
        'draft_correct': draft_correct, 'primary_correct': primary_correct,
        'final_candidate_correct': correct_candidate(final),
        'draft_artifact_present': artifact(draft), 'primary_artifact_present': artifact(primary),
        'released_artifact_present': artifact(final), 'missing_final_artifact': bool(revision.get('protocol')) and not artifact(primary),
        'primary_source_valid': (primary_revision.get('final_check_issues') == [] if artifact(primary) and 'final_check_issues' in primary_revision else None),
        'clean_final_source_check': clean, 'released_artifact_matches_output': release_matches,
        'source_valid_release': source_valid, 'release_policy': policy,
        'answer_is_null': row.get('agent_answer') is None,
        'scored_null_credit': final_correct and row.get('agent_answer') is None,
        'unverified_null_credit': final_correct and row.get('agent_answer') is None and not source_valid,
        'credited_without_source_valid_release': final_correct and not source_valid,
        'solver_source_revised': bool(revision.get('solver_source_revised')),
        'targeted_revised': targeted, 'revised': bool(revision.get('revised')),
        'source_revision_helped': bool(revision.get('solver_source_revised')) and draft_correct is False and primary_correct is True,
        'source_revision_harmed': bool(revision.get('solver_source_revised')) and draft_correct is True and primary_correct is False,
        'targeted_helped': targeted and primary_correct is False and final_correct,
        'targeted_harmed': targeted and primary_correct is True and not final_correct,
        'audit_status': audit_kind, 'audit_verdict_counts': dict(verdict_counts),
        'audit_error': revision.get('audit_error') or revision.get('contract_error'),
        'targeted_proposed': isinstance((revision.get('audit') or {}).get('repair'), dict) if isinstance(revision.get('audit'), dict) else False,
        'targeted_status': proof.get('status'), 'targeted_denial_reason': proof.get('reason'),
        'targeted_issue_types': issue_types(proof.get('issues')),
        'initial_issue_types': issue_types(revision.get('check_issues')),
        'final_issue_types': issue_types(revision.get('final_check_issues')),
        'unresolved_scope_count': len(unresolved), 'unresolved_scope_reason_counts': dict(reason_counts),
        'budget_stop': budget.get('stopped'), 'exhaustion_scope': budget_scope, 'phase_records': phases,
        'phase_fatal': any(bool(entry.get('fatal')) for entry in phases.values()),
        'phase_local_refusal': any(bool(entry.get('errors')) for entry in phases.values()),
        'accounting_anomalies': sorted(set(anomalies)),
        'tokens': tokens.get('total'), 'tool_calls': budget.get('tool_calls_run'),
        'provider_usage_unknown': any(call.get('usage_available') is False for call in tokens.get('per_call') or [] if isinstance(call, dict)),
        'context_compactions': len(revision.get('context_compactions') or []),
        'has_limitation': bool(row.get('limitation')), 'trace_stats': trace_stats(row.get('trace'))}
    return record


def summary(records):
    count = lambda predicate: sum(bool(predicate(row)) for row in records)
    fields = ('terminal_error', 'correct', 'reference_empty', 'source_valid_release', 'scored_null_credit',
              'unverified_null_credit', 'credited_without_source_valid_release', 'solver_source_revised',
              'targeted_revised', 'source_revision_helped', 'source_revision_harmed', 'targeted_helped',
              'targeted_harmed', 'phase_fatal', 'phase_local_refusal', 'provider_usage_unknown', 'missing_final_artifact', 'primary_source_valid')
    result = {'questions': len(records), **{field: count(lambda row, field=field: row[field]) for field in fields}}
    result.update(correct_draft_lost=count(lambda row: row['draft_correct'] is True and not row['correct']),
                  correct_primary_lost=count(lambda row: row['primary_correct'] is True and not row['correct']),
                  correct_primary_terminal_error=count(lambda row: row['primary_correct'] is True and row['terminal_error']),
                  wrong_primary_recovered=count(lambda row: row['primary_correct'] is False and row['correct']),
                  unresolved_scope_questions=count(lambda row: row['unresolved_scope_count'] > 0),
                  primary_source_validation_unknown=count(lambda row: row['primary_artifact_present'] and row['primary_source_valid'] is None),
                  budget_stops=count(lambda row: bool(row['budget_stop'])),
                  accounting_anomaly_questions=count(lambda row: bool(row['accounting_anomalies'])))
    for field in ('audit_status', 'release_policy', 'targeted_status', 'targeted_denial_reason', 'terminal_error_kind', 'exhaustion_scope'):
        result[field + '_counts'] = dict(Counter(row[field] for row in records if row[field] is not None))
    for field in ('audit_verdict_counts', 'unresolved_scope_reason_counts', 'trace_stats'):
        counts = Counter()
        for row in records:
            counts.update(row[field])
        result[field] = dict(counts)
    for field in ('initial_issue_types', 'final_issue_types', 'targeted_issue_types', 'accounting_anomalies'):
        result[field] = dict(Counter(item for row in records for item in row[field]))
    result['phases'] = {}
    for name in PHASES:
        entries = [row['phase_records'][name] for row in records if name in row['phase_records']]
        result['phases'][name] = {'observed': len(entries), 'fatal': sum(bool(item.get('fatal')) for item in entries),
            'local_refusals': sum(bool(item.get('errors')) for item in entries),
            'tokens_recorded': sum(item['tokens'] for item in entries if finite(item.get('tokens'))),
            'tool_calls_recorded': sum(item['tool_calls'] for item in entries if finite(item.get('tool_calls'))),
            'errors': dict(Counter(str(error) for item in entries for error in item.get('errors') or []))}
    return result


def source_caps(path):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PHASE_CAPS', 'PROTOCOL'):
                    values[target.id] = ast.literal_eval(node.value)
    return values


def run(root, arms, output, source_manifest, subset=None):
    root, output, manifest_path = Path(root).resolve(), Path(output).resolve(), Path(source_manifest).resolve()
    source_path = root / 'final_dataset/questions_answers_sql_fhir.csv'
    family_path = root / 'manifests/question_families.csv'
    split_path = root / 'manifests/split.json'
    cohort_path = Path(subset).resolve() if subset else split_path
    manifest = json.loads(manifest_path.read_text())
    hashes = {'source_manifest': sha256(manifest_path), 'dataset': sha256(source_path),
              'families': sha256(family_path), 'cohort': sha256(cohort_path), 'scorer': sha256(root / 'utils/answer_scorer.py'),
              'diagnostics_script': sha256(__file__)}
    for key, actual in (('dataset_sha256', hashes['dataset']), ('split_sha256', sha256(split_path))):
        if manifest.get(key) and manifest[key] != actual:
            raise ValueError('Frozen source input differs: ' + key)
    frozen_files = manifest.get('source_sha256') or {}
    if not isinstance(frozen_files, dict) or not frozen_files.get('utils/answer_scorer.py'):
        raise ValueError('A frozen source manifest with the unchanged scorer hash is required')
    if frozen_files.get('utils/answer_scorer.py') and frozen_files['utils/answer_scorer.py'] != hashes['scorer']:
        raise ValueError('The frozen unchanged scorer differs from the analysis scorer')
    if (manifest_path.parent / 'source.zip').exists():
        hashes['source_archive'] = sha256(manifest_path.parent / 'source.zip')
        if manifest.get('source_archive_sha256') and manifest['source_archive_sha256'] != hashes['source_archive']:
            raise ValueError('Frozen source archive differs')
    cohort = json.loads(cohort_path.read_text())
    ids = cohort.get('question_ids') if subset else cohort['splits']['held_out']['question_ids']
    if not isinstance(ids, list) or len(ids) != 300 or len(set(ids)) != 300 or not all(isinstance(qid, str) for qid in ids):
        raise ValueError('Only the complete fixed 300-question cohort is allowed')
    expected = set(ids)
    with source_path.open(newline='') as stream:
        source = {row['question_id']: row for row in csv.DictReader(stream)}
    with family_path.open(newline='') as stream:
        families = {row['question_id']: row for row in csv.DictReader(stream)}
    if not expected.issubset(source) or not expected.issubset(families):
        raise ValueError('Cohort question metadata is incomplete')
    source_files = {}
    for relative in ('agent/query_contract_agent.py', 'agent/single_agent.py', 'agent/base_agent.py', 'agent/framework.py',
                     'tools/field_check.py', 'tools/python_worker.py', 'tools/resource_tools.py', 'utils/core_utils.py',
                     'utils/answer_scorer.py', 'scripts/audit_store.py', 'run_agent.py'):
        path = root / relative
        if path.exists():
            actual = sha256(path)
            source_files[relative] = {'sha256': actual, 'frozen_sha256': frozen_files.get(relative)}
            if frozen_files.get(relative) and frozen_files[relative] != actual:
                raise ValueError('Frozen source module differs: ' + relative)
    hashes['source_modules'] = source_files
    cap_path = root / 'agent/query_contract_agent.py'
    caps = source_caps(cap_path) if cap_path.exists() else {'PHASE_CAPS': {}, 'PROTOCOL': None}
    scorer = load_module(root / 'utils/answer_scorer.py', '_fhir_posthoc_unchanged_scorer')
    audit_module = None
    prepared = []
    names = set()
    # Complete and stable cohort validation precedes every score and output write.
    for spec in arms:
        name, separator, filename = spec.partition('=')
        if not separator or not name or name in names:
            raise ValueError('Arms require unique NAME=PATH specifications')
        names.add(name)
        raw_path = Path(filename).resolve()
        sidecar = Path(str(raw_path) + '.run.json')
        identity = json.loads(sidecar.read_text())
        raw_hash, sidecar_hash = sha256(raw_path), sha256(sidecar)
        if identity.get('split_manifest_hash') and identity['split_manifest_hash'] != sha256(split_path):
            raise ValueError(name + ': split identity differs')
        if subset and identity.get('subset_hash') and identity['subset_hash'] != hashes['cohort']:
            raise ValueError(name + ': subset identity differs')
        seen = set()
        for original in iter_rows(raw_path):
            if not isinstance(original, dict):
                raise ValueError(name + ': malformed scheduled row')
            qid = original.get('question_id')
            if qid not in expected or qid in seen:
                raise ValueError(name + ': duplicate or unexpected question')
            seen.add(qid)
            if original.get('patient_fhir_id') != source[qid]['patient_fhir_id']:
                raise ValueError(name + ': patient identity differs')
            selected, _ = first_attempt(original)
            if not finite(selected.get('latency')) and not failed(selected):
                raise ValueError(name + ': incomplete first attempt; no partial scoring permitted')
            if identity.get('audit_storage') == 'gzip' and not isinstance(selected.get('audit_artifacts'), dict):
                raise ValueError(name + ': first gzip attempt lacks an audit artifact')
        if seen != expected:
            raise ValueError(name + ': incomplete scheduled cohort; no partial scoring permitted')
        prepared.append((name, raw_path, sidecar, identity, raw_hash, sidecar_hash))
    score = lambda answer, qid: scorer.score_answer(answer, source[qid]['true_answer'], source[qid]['question'], families[qid]['template'])
    all_records, summaries, arm_hashes = [], {}, {}
    for name, raw_path, sidecar, identity, raw_hash, sidecar_hash in prepared:
        records, artifact_hashes = [], []
        for original in iter_rows(raw_path):
            qid = original['question_id']
            selected, retries = first_attempt(original)
            reference = selected.get('audit_artifacts')
            if reference is not None:
                if audit_module is None:
                    audit_module = load_module(root / 'scripts/audit_store.py', '_fhir_posthoc_audit_store')
                selected = audit_module.restore_result_audit(selected, raw_path.parent, question_id=qid, restore_history=False)
                artifact_path = (raw_path.parent / reference['path']).resolve()
                artifact_hashes.append({'question_id': qid, 'path': reference['path'],
                    'uncompressed_sha256': reference['sha256'], 'compressed_sha256': sha256(artifact_path)})
            record = row_record(selected, qid, families[qid], score, caps['PHASE_CAPS'], identity)
            record.update(arm=name, recorded_retries=retries)
            records.append(record)
        if sha256(raw_path) != raw_hash or sha256(sidecar) != sidecar_hash:
            raise ValueError(name + ': raw run changed during analysis; report not written')
        summaries[name] = summary(records)
        arm_hashes[name] = {'raw_path': str(raw_path), 'raw_sha256': raw_hash, 'sidecar_sha256': sidecar_hash,
            'identity': {key: identity.get(key) for key in ('agent_strategy', 'model', 'code_hash', 'input_hash',
                       'max_total_tokens', 'max_tool_calls', 'audit_storage', 'split', 'repeat')},
            'first_attempt_artifacts': artifact_hashes,
            'artifact_index_sha256': hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest()}
        all_records.extend(records)
    for key, path in (('source_manifest', manifest_path), ('dataset', source_path), ('families', family_path), ('cohort', cohort_path)):
        if sha256(path) != hashes[key]:
            raise ValueError('Input changed during analysis: ' + key)
    for relative, record in source_files.items():
        if sha256(root / relative) != record['sha256']:
            raise ValueError('Source input changed during analysis: ' + relative)
    report = {'schema_version': 3, 'scope': 'Completed fixed 300; post-hoc descriptive mechanisms only; no runtime gold feedback.',
        'scoring': 'Unchanged typed scorer; all first attempts; terminal errors incorrect; later retries cannot replace outcomes.',
        'caveats': ['Source field validation and local AST witnesses do not prove global clinical semantics or complete retrieval.',
                    'The scorer can credit a null/absence on an empty reference despite no source-valid release.',
                    'The static auditor sees code and bounded source samples; it is not an independent clinical recomputation.'],
        'input_hashes': hashes, 'frozen_protocol': caps, 'raw_inputs': arm_hashes, 'arms': summaries}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mechanisms.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'mechanisms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        for record in all_records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})
    (output / 'mechanisms_rows.json').write_text(json.dumps(all_records, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', action='append', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--subset', help='Optional exact 300-question subset manifest; default held_out split')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.root, args.arm, args.output, args.source_manifest, args.subset)
    print(json.dumps(report['arms'], indent=2))


if __name__ == '__main__':
    main()
