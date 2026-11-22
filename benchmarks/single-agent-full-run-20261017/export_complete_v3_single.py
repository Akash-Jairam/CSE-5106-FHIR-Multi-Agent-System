"""Private additive delivery of complete matched V3 + Single Agent cohorts.

No inference/launch/configuration functions execute. The original three-arm
analyzer and its final gate stay unchanged. Both 2,931-question first-clean
cohorts must validate before the analyzer/scorer is imported. Direct LLM is
omitted entirely from the detached analysis snapshot and final paired report.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import csv
import importlib.util
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ARMS = ('single_agent', 'multi_agent')
N = 2931
FINGERPRINT = '65d2bbfc7da05f97e646acb22d20f0367d7f7a4e1a2a7c8b9c244fd77ef8d409'
STATUS = 'V3_SINGLE_COMPLETE_DIRECT_PENDING'
CLASSIFIER_NAMES = {'CLASSIFIER_VERSION', 'RUNTIME', 'INTRINSIC', 'PROVIDER_JWT_REJECTION',
                    'classify_attempt', 'diagnostic_errors', 'intrinsic_tool_validation', 'provider_jwt_paths'}
PUBLIC_PINS = {'baseline_decision.txt', 'preflight.json', 'protocol.txt', 'provider_profile.json',
               'pricing.json', 'split.json', 'runtime/azure.py', 'runtime/azure_adapter.py', 'runtime/sitecustomize.py'}
SNAPSHOT_OMIT_PUBLIC = PUBLIC_PINS - {'pricing.json', 'split.json'}
V4_RUNNER_SHA256 = 'd13f52fa060acbdd1aec73b985b8857dba6664c060ad3228dc2283b3295768da'
V4_POLICY = {'version': 'azure-prompt-rejection-v1', 'category': 'provider_policy_rejection',
             'max_rejections_per_question': 2, 'retry_request': 'unchanged',
             'persistent_action': 'park_pending_then_attention', 'scope': ['single_agent']}
V4_CLASSIFIER_NAMES = CLASSIFIER_NAMES | {'is_observed_provider_prompt_rejection', 'SDK_ERROR_PREFIX', 'MAX_ERROR_CHARS'}


def load_utilities(root):
    source = root / 'benchmarks/v3-full-run-20261015/export_complete_v3.py'
    specification = importlib.util.spec_from_file_location('original_complete_v3_utilities', source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def classifier_ast(path, semantic_hash):
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in CLASSIFIER_NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in CLASSIFIER_NAMES for target in node.targets):
            nodes.append(node)
    return semantic_hash(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))


def load_v4_classifier(path, u):
    """Execute the reviewed pure classifier dependency closure only."""
    u.require(u.sha256(path) == V4_RUNNER_SHA256, 'Unapproved provider classifier source')
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    definitions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    selected, pending = set(), ['classify_attempt', 'CLASSIFIER_VERSION']
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        selected.add(name)
        pending.extend(child.id for child in ast.walk(definitions[name])
                       if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                       and child.id in definitions and child.id not in selected)
    u.require(selected == V4_CLASSIFIER_NAMES, 'Provider classifier dependency set changed')
    nodes = [node for node in tree.body if any(definitions[name] is node for name in selected)]
    u.require(not any(isinstance(node, ast.FunctionDef) and node.decorator_list for node in nodes), 'Classifier decorators forbidden')
    scope = {'ast': ast, 're': re, 'math': math, 'hashlib': hashlib, 'json': json, 'restore_result_audit': u.restore_result_audit}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    u.require(scope['CLASSIFIER_VERSION'] == 'v3-infrastructure-v4', 'Provider classifier version changed')
    digest = u.semantic_hash(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))
    return scope['classify_attempt'], scope['CLASSIFIER_VERSION'], digest


class PairedGate:
    def __init__(self, root, run, historical_export, amendment, provider_amendment=None):
        self.root, self.run, self.historical_export, self.amendment = map(Path, (root, run, historical_export, amendment))
        self.u = load_utilities(self.root)
        self.inventory = {}
        self.canonical_hashes = {}
        self.row_hashes = {}
        self.ledger_slices = {}
        self.replay = {}
        self.rounds = {}
        self.provider_amendment = Path(provider_amendment) if provider_amendment else None
        self.classifier_transition = None
        self.transition_replay = {arm: {'attempted': 0, 'unchanged': 0, 'provider_rejections': []} for arm in ARMS}

    def remember(self, path):
        path = Path(path).resolve()
        digest = self.u.sha256(path)
        self.inventory[path] = {'sha256': digest, 'bytes': path.stat().st_size}
        return digest

    def slice(self, ledger, arm):
        return {'selections': (ledger.get('arms') or {}).get(arm) or {},
                'counts': (ledger.get('counts') or {}).get(arm) or {}}

    def validate(self):
        u = self.u
        self.manifest_path, self.ledger_path = self.run / 'run_manifest.json', self.run / 'acceptance.json'
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.manifest_hash = self.remember(self.manifest_path)
        self.ledger_bytes = self.ledger_path.read_bytes()
        ledger = json.loads(self.ledger_bytes)
        u.require(ledger.get('protocol') == 'v3-first-runtime-clean-terminal' and ledger.get('selection') == 'first_runtime_clean_terminal', 'Acceptance protocol changed')
        u.require(self.manifest.get('questions_per_arm') == N, 'Full 2,931-question cohort required')
        self.source = Path(self.manifest['input']['path'])
        u.require(self.remember(self.source) == self.manifest['input']['sha256'], 'Dataset hash changed')
        patients, source_rows = {}, {}
        with self.source.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                qid, patient = row['question_id'], row['patient_fhir_id']
                u.require(qid and patient and qid not in patients, 'Dataset identities missing or duplicated')
                patients[qid], source_rows[qid] = patient, u.semantic_hash(row)
        u.require(len(patients) == N, 'Dataset does not have 2,931 unique questions')
        # Gate BOTH cohorts before importing the analyzer or evaluating correctness.
        for arm in ARMS:
            self.ledger_slices[arm] = self.slice(ledger, arm)
            u.require(set(self.ledger_slices[arm]['selections']) == set(patients), arm + ': incomplete acceptance cohort; no partial scoring permitted')
            path = self.run / f'full_{arm}.json'
            self.canonical_hashes[arm] = self.remember(path)
            indexed = {}
            for row in u.iter_result_rows(path):
                qid = row.get('question_id')
                u.require(qid in patients and qid not in indexed and row.get('patient_fhir_id') == patients[qid], arm + ': canonical identity changed')
                u.require(not row.get('attempt_history') and 'agent_answer' in row, arm + ': retry history/missing answer field')
                indexed[qid] = u.semantic_hash(row)
            u.require(set(indexed) == set(patients), arm + ': incomplete canonical cohort')
            self.row_hashes[arm] = indexed
            identity_path = Path(str(path) + '.run.json')
            self.remember(identity_path)
            expected = self.manifest['run_identities'][arm]
            u.require(u.read_json(identity_path) == expected and expected.get('code_hash') == FINGERPRINT,
                      arm + ': frozen inference identity changed')
            u.require(expected.get('agent_strategy') == ('multi_agent_v3' if arm == 'multi_agent' else arm), 'Strategy identity changed')
        self.validate_lineage()
        full_input_hash = u.input_identity_hash(self.source)
        for arm in ARMS:
            u.require(self.manifest['run_identities'][arm]['input_hash'] == full_input_hash, 'Full input identity changed')
        self.validate_sources()
        if self.provider_amendment:
            classifier, version, current_ast = load_v4_classifier(self.root / 'scripts/v3_full_run.py', u)
            u.require(version == self.manifest['classifier_version'], 'Provider classifier manifest version differs')
            self.classifier_transition['current_AST_sha256'] = current_ast
            self.old_classifier, old_version = u.load_classifier(self.amendment / 'pre_apply_source/scripts/v3_full_run.py')
            u.require(old_version == 'v3-infrastructure-v3', 'Historical classifier version changed')
            active = lambda row, directory: classifier(row, directory, allow_provider_rejection=True)
        else:
            classifier, version = u.load_classifier(self.root / 'scripts/v3_full_run.py')
            u.require(version == 'v3-infrastructure-v3' == self.manifest['classifier_version'], 'Runtime selection policy changed')
            active = classifier
        self.classifier_version = version
        for arm in ARMS:
            self.replay_arm(arm, active, patients, source_rows)
        self.check_stable()
        self.verify_inventory()
        return self

    def validate_lineage(self):
        u = self.u
        receipt_path = self.historical_export / 'V3_COMPLETE_CONTROLS_PENDING.json'
        old_manifest_path = self.historical_export / 'validated_v3_input_snapshot/run_manifest.json'
        self.historical_receipt = u.read_json(receipt_path)
        old = u.read_json(old_manifest_path)
        self.remember(receipt_path)
        self.utility_hash = self.remember(Path(u.__file__))
        u.require(self.utility_hash == self.historical_receipt['helper_sha256'], 'Historical validation utility source changed')
        self.historical_manifest_hash = self.remember(old_manifest_path)
        u.require(self.historical_receipt['status'] == 'V3_COMPLETE_CONTROLS_PENDING'
                  and self.historical_receipt['questions'] == N and self.historical_receipt['v3_complete'] is True,
                  'Validated historical complete V3 receipt required')
        u.require(self.historical_manifest_hash == self.historical_receipt['source_manifest_sha256'], 'Historical V3 manifest changed')
        archived_manifest = self.amendment / 'pre_apply_metadata/run_manifest.json'
        archived_runner = self.amendment / 'pre_apply_source/scripts/v3_full_run.py'
        u.require(self.remember(archived_manifest) == self.historical_manifest_hash, 'Selected-arm amendment is not based on original V3 manifest')
        u.require(self.remember(archived_runner) == old['source_sha256']['scripts/v3_full_run.py'], 'Historical operational runner changed')
        # Only runner scheduling source and an optional explicit selected scope may differ.
        excluded = {'source_sha256', 'selected_arms'}
        if self.provider_amendment:
            excluded |= {'classifier_version', 'provider_rejection_policy'}
        u.require({k: v for k, v in self.manifest.items() if k not in excluded}
                  == {k: v for k, v in old.items() if k not in excluded}, 'Non-operational frozen manifest fields changed')
        u.require(self.manifest.get('selected_arms') == ['single_agent'], 'Operational selection is not explicitly single_agent only')
        before, after = old['source_sha256'], self.manifest['source_sha256']
        u.require(set(before) == set(after), 'Source pin set changed')
        differences = {name for name in before if before[name] != after[name]}
        u.require(differences <= {'scripts/v3_full_run.py'}, 'Inference/analyzer/launcher source changed')
        self.operational_source_changes = {name: {'before': before[name], 'after': after[name]} for name in sorted(differences)}
        self.classifier_ast_hash = classifier_ast(archived_runner, u.semantic_hash)
        if self.provider_amendment:
            self.validate_provider_lineage(old)
        else:
            u.require(self.classifier_ast_hash == classifier_ast(self.root / 'scripts/v3_full_run.py', u.semantic_hash),
                      'Scheduling amendment changed the historical runtime classifier semantics')
        u.require(self.canonical_hashes['multi_agent'] == self.historical_receipt['v3_canonical_sha256_after'], 'Historical V3 canonical changed')
        u.require(u.semantic_hash(self.ledger_slices['multi_agent']) == self.historical_receipt['v3_ledger_slice_sha256_after'], 'Historical V3 selections changed')
        self.old_manifest = old

    def validate_provider_lineage(self, original):
        u = self.u
        amendment = self.provider_amendment.resolve()
        u.require(amendment.is_relative_to((self.run / 'amendments').resolve()), 'Provider amendment outside run')
        before_path = amendment / 'pre_apply_metadata/run_manifest.json'
        before = u.read_json(before_path)
        self.remember(before_path)
        runner = amendment / 'pre_apply_source/scripts/v3_full_run.py'
        u.require(self.remember(runner) == before['source_sha256']['scripts/v3_full_run.py'], 'Pre-provider source changed')
        u.require(classifier_ast(runner, u.semantic_hash) == self.classifier_ast_hash, 'Prior selected scheduler changed classifier semantics')
        excluded = {'source_sha256', 'selected_arms'}
        u.require({k:v for k,v in before.items() if k not in excluded} == {k:v for k,v in original.items() if k not in excluded}, 'Prior selected manifest inference differs')
        u.require(before.get('selected_arms') == ['single_agent'], 'Prior scope changed')
        u.require(set(before['source_sha256']) == set(original['source_sha256']) and all(before['source_sha256'][k] == v for k,v in original['source_sha256'].items() if k != 'scripts/v3_full_run.py'), 'Prior inference source changed')
        expected = json.loads(json.dumps(before))
        expected['source_sha256']['scripts/v3_full_run.py'] = V4_RUNNER_SHA256
        expected['classifier_version'] = 'v3-infrastructure-v4'
        expected['provider_rejection_policy'] = V4_POLICY
        u.require(self.manifest == expected, 'Provider amendment changed unapproved manifest fields')
        approval_path = amendment / 'root_approval.json'
        approval = u.read_json(approval_path)
        self.remember(approval_path)
        u.require(approval.get('status') == 'APPROVED_OPERATIONAL_RECOVERY_ONLY'
                  and approval.get('runner_sha256') == V4_RUNNER_SHA256
                  and approval.get('manifest_sha256') == self.manifest_hash
                  and approval.get('inference_unchanged') is True
                  and approval.get('historical_accepted_choices_unchanged') is True,
                  'Provider amendment lacks reviewed provenance')
        self.classifier_transition = {'amendment_directory': str(amendment), 'approval_sha256': u.sha256(approval_path),
                                     'historical_version': 'v3-infrastructure-v3', 'historical_AST_sha256': self.classifier_ast_hash,
                                     'current_version': 'v3-infrastructure-v4', 'policy': V4_POLICY,
                                     'scope': 'Single-agent provider rejection before usable completion; all V3 decisions unchanged'}

    def validate_sources(self):
        u = self.u
        for relative, expected in self.manifest['source_sha256'].items():
            path = (self.root / relative).resolve()
            u.require(path.is_relative_to(self.root) and (path.suffix == '.py' or relative == 'requirements.txt'), 'Unexpected source pin path')
            u.require(self.remember(path) == expected, 'Frozen current source changed')
        for key in ('families', 'split_manifest'):
            entry = self.manifest[key]
            u.require(self.remember(entry['path']) == entry['sha256'], 'Frozen analysis input changed')
        pins = self.manifest.get('pinned_files_sha256') or {}
        u.require(set(pins) == PUBLIC_PINS, 'Public pinned-file allowlist changed')
        for name in sorted(PUBLIC_PINS):
            path = self.run / name
            u.require(not path.is_symlink() and path.resolve().is_relative_to(self.run)
                      and not (name.startswith('runtime/') and (self.run / 'runtime').is_symlink()),
                      'Public pinned-file path must be regular and owned by the run')
            u.require(self.remember(path) == pins[name], 'Frozen public pinned file changed: ' + name)
        self.pricing = u.read_json(self.run / 'pricing.json')
        u.require(self.pricing == self.historical_receipt['pricing'], 'Matched-arm official rates differ')
        reference = self.manifest['run_identities']['multi_agent']
        single = self.manifest['run_identities']['single_agent']
        u.require({key: value for key, value in reference.items() if key != 'agent_strategy'}
                  == {key: value for key, value in single.items() if key != 'agent_strategy'}, 'Matched model/input/budgets/storage identity differs')

    def replay_arm(self, arm, classifier, patients, source_rows):
        u = self.u
        selected, choices, retries, provider_counts = {}, {}, Counter(), Counter()
        counts = {'raw_attempts': 0, 'runtime_failures': 0, 'intrinsic_errors': 0}
        batches = sorted((self.run / 'attempts' / arm).glob('round-*.batch.json'), key=lambda path: int(path.name.split('-')[1].split('.')[0]))
        u.require(bool(batches), arm + ': no sealed source batches')
        rounds = []
        expected = self.manifest['run_identities'][arm]
        for batch_path in batches:
            seal_path = batch_path.with_suffix('.seal.json')
            u.require(seal_path.is_file(), arm + ': unsealed batch remains')
            batch, seal = u.read_json(batch_path), u.read_json(seal_path)
            number = int(batch_path.name.split('-')[1].split('.')[0])
            u.require(batch.get('arm') == arm and batch.get('round') == number, 'Batch ownership changed')
            u.require(self.remember(batch_path) == seal['batch_sha256'], 'Batch seal mismatch')
            self.remember(seal_path)
            output, shard = Path(batch['output']).resolve(), Path(batch['input']).resolve()
            u.require(output.is_relative_to(self.run) and shard.is_relative_to(self.run), 'Batch path escapes run')
            u.require(self.remember(shard) == batch['input_sha256'], 'Shard hash changed')
            ids = []
            with shard.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    qid = row['question_id']
                    u.require(qid in patients and u.semantic_hash(row) == source_rows[qid], 'Shard differs from authoritative input')
                    ids.append(qid)
            u.require(len(set(ids)) == len(ids) and ids == batch['question_ids'], 'Shard question order/cohort changed')
            if self.provider_amendment and arm == 'single_agent':
                u.require(all(provider_counts[qid] < V4_POLICY['max_rejections_per_question'] for qid in ids),
                          'Historical batch scheduled a persistently rejected request beyond its unchanged-retry limit')
            rounds.append(number)
            if not output.exists():
                u.require(seal.get('result_sha256') is None and seal.get('identity_sha256') is None, 'Missing sealed result')
                continue
            identity_path = Path(str(output) + '.run.json')
            u.require(self.remember(output) == seal['result_sha256'] and self.remember(identity_path) == seal['identity_sha256'], 'Raw result/identity seal mismatch')
            identity = u.read_json(identity_path)
            u.require(identity == batch['run_identity'] and set(identity) == set(expected), 'Raw identity changed')
            u.require(all(value == expected[key] for key, value in identity.items() if key != 'input_hash')
                      and identity['input_hash'] == u.input_identity_hash(shard), 'Raw model/input/inference identity differs')
            seen = set()
            for row in u.iter_result_rows(output):
                qid = row.get('question_id')
                u.require(qid in ids and qid not in seen and row.get('patient_fhir_id') == patients[qid], 'Raw question/patient ownership changed')
                seen.add(qid)
                ref = row.get('audit_artifacts')
                if ref is not None:
                    u.require(isinstance(ref, dict) and ref.get('version') == 2 and ref.get('codec') == 'json+gzip'
                              and ref.get('question_id') == qid and u.ARTIFACT_PATTERN.fullmatch(ref.get('path', '')), 'Raw audit owner/descriptor changed')
                    path = (output.parent / ref['path']).resolve()
                    u.require(path.is_relative_to(output.parent), 'Raw audit path escapes owner')
                    audit_hash = self.remember(path)
                full = u.restore_result_audit(row, output.parent, restore_history=False)
                decision = classifier(full, None)
                if self.provider_amendment and u.attempted(row):
                    previous = self.old_classifier(full, None)
                    replay = self.transition_replay[arm]
                    replay['attempted'] += 1
                    if decision == previous:
                        replay['unchanged'] += 1
                    else:
                        u.require(arm == 'single_agent' and previous['status'] == 'attention'
                                  and decision == {'status':'runtime', 'reason':'provider_policy_rejection',
                                                   'category':'provider_policy_rejection', 'diagnostic_paths':['error']},
                                  'Classifier transition changed a non-provider decision')
                        replay['provider_rejections'].append({'question_id': qid, 'round': number,
                                                             'before': previous, 'after': decision})
                if not u.attempted(row):
                    u.require(decision['status'] == 'pending', 'Unattempted placeholder classified terminal')
                    continue
                u.require(qid not in selected, 'Accepted question was rerun')
                counts['raw_attempts'] += 1
                if decision['status'] == 'runtime':
                    counts['runtime_failures'] += 1
                    retries[qid] += 1
                    if decision.get('category') == 'provider_policy_rejection':
                        provider_counts[qid] += 1
                elif decision['status'] == 'accepted':
                    u.require(ref is not None, 'Accepted row lacks owned audit')
                    selected[qid] = u.semantic_hash({key: value for key, value in row.items() if key != 'attempt_history'})
                    choices[qid] = {'attempt_file': str(output.relative_to(self.run)), 'round': number,
                                    'runtime_failures_before': retries[qid], 'selection': decision['reason'], 'runtime_clean': True}
                    counts['intrinsic_errors'] += int(bool(row.get('error')))
                    u.require(self.remember(self.run / ref['path']) == audit_hash, 'Copied canonical audit differs')
                else:
                    raise ValueError(arm + ': unresolved sealed attention or invalid attempt')
            u.require(seen == set(ids), 'Scheduled placeholders omitted')
        u.require(selected == self.row_hashes[arm], arm + ': canonical is not the first clean selection')
        u.require(choices == self.ledger_slices[arm]['selections'] and counts == self.ledger_slices[arm]['counts'], arm + ': replay/ledger mismatch')
        self.replay[arm], self.rounds[arm] = counts, rounds

    def check_stable(self):
        u = self.u
        u.require(u.sha256(self.manifest_path) == self.manifest_hash, 'Operational manifest changed during export')
        latest = u.read_json(self.ledger_path)
        for arm in ARMS:
            u.require(u.sha256(self.run / f'full_{arm}.json') == self.canonical_hashes[arm]
                      and self.slice(latest, arm) == self.ledger_slices[arm], arm + ': accepted cohort/slice changed during export')
        return latest

    def verify_inventory(self):
        for path, metadata in self.inventory.items():
            self.u.require(self.u.sha256(path) == metadata['sha256'], 'Verified immutable provenance changed')

    def snapshot(self, directory):
        directory.mkdir()
        for source, metadata in self.inventory.items():
            if not source.is_relative_to(self.run):
                continue
            relative = source.relative_to(self.run)
            # Historical amendment lineage is retained externally in receipt/hash records.
            if relative.parts[0] == 'amendments' or relative.as_posix() in SNAPSHOT_OMIT_PUBLIC:
                continue
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.as_posix() in {'run_manifest.json', 'pricing.json', 'split.json'} or ('audit' not in relative.parts and metadata['bytes'] <= 1024 * 1024):
                shutil.copyfile(source, target)
            else:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            self.u.require(self.u.sha256(target) == metadata['sha256'], 'Snapshot hash mismatch')
        (directory / 'run_manifest.json').write_bytes(self.manifest_bytes)
        (directory / 'acceptance.json').write_bytes(self.ledger_bytes)
        if (self.run / 'supervisor_state.json').exists():
            shutil.copyfile(self.run / 'supervisor_state.json', directory / 'supervisor_state.json')
        if (self.run / 'operations.jsonl').exists():
            raw = (self.run / 'operations.jsonl').read_bytes()
            prefix = raw[:raw.rfind(b'\n') + 1] if b'\n' in raw else b''
            for line in prefix.splitlines():
                json.loads(line)
            (directory / 'operations.jsonl').write_bytes(prefix)
        self.check_stable()
        self.verify_inventory()


def deliver(root, run, historical_export, amendment, output, provider_amendment=None):
    root, run, historical_export, amendment, output = [Path(path).resolve() for path in (root, run, historical_export, amendment, output)]
    if output.exists() or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Output must be NEW and outside the live run')
    gate = PairedGate(root, run, historical_export, amendment, provider_amendment).validate()
    u = gate.u
    output.mkdir(parents=True)
    snapshot = output / 'validated_paired_input_snapshot'
    gate.snapshot(snapshot)
    # No analyzer/scorer import before both full cohorts and their provenance pass.
    analyzer = u.load_analyzer()
    report, records = analyzer.analyze(snapshot, partial=True, n_boot=10000)
    u.require(report['status'] == 'PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL', 'Original three-arm partial label changed')
    u.require(len(records) == N * 2 and Counter(row['arm'] for row in records) == Counter({arm: N for arm in ARMS})
              and all(row['runtime_clean_selection_verified'] for row in records), 'Paired export contains incomplete/foreign arm')
    u.require(all(report['completeness'][arm]['complete'] and report['completeness'][arm]['missing'] == 0 for arm in ARMS), 'Paired cohorts incomplete')
    analyzer.write_report(report, records, output / 'global_partial')
    projected = {key: value for key, value in report.items() if key not in {'all_arms_accepted_metrics', 'arms', 'completeness',
                  'accepted_results', 'groups', 'comparisons', 'mechanisms', 'phase_usage', 'operational_context'}}
    comparisons = [item for item in report['comparisons'] if {item['left'], item['right']} == set(ARMS)]
    u.require(len(comparisons) == 1 and comparisons[0]['matched_questions'] == N
              and comparisons[0]['left_only'] == comparisons[0]['right_only'] == 0, 'Matched paired comparison incomplete')
    state_path = snapshot / 'supervisor_state.json'
    state = u.read_json(state_path) if state_path.exists() else {}
    scope_started, scope_finished = state.get('scope_started_at'), state.get('finished_at')
    scope_wall_seconds = ((datetime.fromisoformat(scope_finished) - datetime.fromisoformat(scope_started)).total_seconds()
                          if scope_started and scope_finished else None)
    selected_scope_timing = {'started_at': scope_started, 'finished_at': scope_finished,
                             'wall_clock_seconds': scope_wall_seconds,
                             'note': 'Selected-resume interval; separate from summed accepted task time and from the historical first-to-last batch span.'}
    projected.update(status=STATUS, benchmark_complete=False, paired_scope_complete=True, scope='matched_v3_single_agent',
                     arms={arm: report['arms'][arm] for arm in ARMS}, completeness={arm: report['completeness'][arm] for arm in ARMS},
                     accepted_results={arm: report['accepted_results'][arm] for arm in ARMS},
                     groups={key: [item for item in table if item['arm'] in ARMS] for key, table in report['groups'].items()},
                     comparisons=comparisons, mechanisms={arm: report['mechanisms'][arm] for arm in ARMS},
                     phase_usage=[item for item in report['phase_usage'] if item['arm'] in ARMS],
                     operational_context={'per_arm': {arm: report['operational_context']['per_arm'][arm] for arm in ARMS},
                                          'selected_single_scope': selected_scope_timing,
                                          'note': 'Historical first-to-last per-arm spans can include earlier partial batches, explicit user deferral and recovery. They are not per-task latency. Accepted task metrics exclude runtime failures.'})
    analyzer.write_report(projected, records, output / 'paired_complete')
    gate.check_stable()
    gate.verify_inventory()
    receipt = {'schema_version': 1, 'status': STATUS, 'benchmark_complete': False, 'paired_scope_complete': True,
               'generated_at': datetime.now(timezone.utc).isoformat(), 'participating_arms': list(ARMS), 'questions_per_arm': N,
               'inference_fingerprint': FINGERPRINT, 'current_manifest_sha256': gate.manifest_hash,
               'historical_v3_manifest_sha256': gate.historical_manifest_hash,
               'historical_v3_receipt_sha256': u.sha256(historical_export / 'V3_COMPLETE_CONTROLS_PENDING.json'),
               'selected_arm_operational_amendment_directory': str(amendment), 'operational_source_changes': gate.operational_source_changes,
               'historical_classifier_AST_sha256': gate.classifier_ast_hash, 'classifier_version': gate.classifier_version,
               'classifier_transition': gate.classifier_transition, 'classifier_transition_actual_replay': gate.transition_replay,
               'canonical_sha256': gate.canonical_hashes,
               'accepted_ledger_slice_sha256': {arm: u.semantic_hash(gate.ledger_slices[arm]) for arm in ARMS},
               'sealed_rounds': gate.rounds, 'first_clean_replay_counts': gate.replay,
               'selected_single_scope_timing': selected_scope_timing,
               'missing_measurements': {arm: {key: metric['missing_n'] for key, metric in report['arms'][arm]['metrics'].items()} for arm in ARMS},
               'paired_difference_convention': 'Unchanged frozen analyzer convention: left minus right; see left/right in paired_comparisons.csv.',
               'direct_llm': {'included_or_scored': False, 'status': 'DEFERRED_NOT_COMPARED'},
               'global_partial_export': {'directory': 'global_partial', 'status': report['status'], 'direct_llm_omitted_from_snapshot': True},
               'original_three_arm_final_gate_changed': False, 'scoring': report['scoring'], 'statistics': report['statistics'], 'pricing': report['pricing'],
               'snapshot': {'directory': 'validated_paired_input_snapshot', 'self_contained_or_portable': False,
                            'verified_public_pins_omitted': sorted(SNAPSHOT_OMIT_PUBLIC),
                            'note': 'Direct LLM and provider/runtime/preflight/protocol/baseline files omitted. All nine public pins verified before analysis and after export. Original manifest/ledger bytes copied. Audits and large results can be writable hardlinks; do not edit/chmod snapshot files. Absolute input and batch paths remain historical references.'},
               'helper_sha256': u.sha256(Path(__file__))}
    receipt['reused_validation_utility_sha256'] = gate.utility_hash
    (output / 'V3_SINGLE_COMPLETE_DIRECT_PENDING.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    (output / 'verified_provenance.json').write_text(json.dumps({str(path): metadata for path, metadata in gate.inventory.items()}, indent=2) + '\n')
    (output / 'README.txt').write_text('V3_SINGLE_COMPLETE_DIRECT_PENDING\n\nBoth matched 2,931-question arms are complete and independently first-clean validated. '
        'Use paired_complete/ for per-question data, all per-arm metrics, mechanisms and the matched paired comparison. '
        'Direct LLM is omitted and is not scored or compared. The original three-arm benchmark is not marked complete.\n'
        'global_partial/ preserves unchanged frozen analyzer output from a two-arm snapshot. '
        'The frozen inference fingerprint is unchanged. The receipt records any reviewed provider-rejection classifier extension and full actual-attempt replay; V3 decisions remain unchanged. '
        'Comparison signs follow left minus right. Treat input snapshot hardlinks as read-only; this is not a portable archive.\n')
    (output / 'delivery_artifact_hashes.json').write_text(json.dumps(u.file_manifest(output), indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--historical-v3-export', required=True, type=Path)
    parser.add_argument('--operational-amendment-dir', required=True, type=Path)
    parser.add_argument('--provider-rejection-amendment-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    receipt = deliver(args.project_root, args.run_dir, args.historical_v3_export, args.operational_amendment_dir, args.output_dir, args.provider_rejection_amendment_dir)
    print(json.dumps({'status': receipt['status'], 'questions_per_arm': N, 'benchmark_complete': False,
                      'receipt': str(args.output_dir / 'V3_SINGLE_COMPLETE_DIRECT_PENDING.json')}))


if __name__ == '__main__':
    main()
