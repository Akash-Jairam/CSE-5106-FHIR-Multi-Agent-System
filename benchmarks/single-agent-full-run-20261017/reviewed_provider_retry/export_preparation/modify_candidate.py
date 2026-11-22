from pathlib import Path
p=Path('/private/tmp/single-reviewed-retry-export-20261017/export_complete_v3_single.py')
s=p.read_text()
s=s.replace("V4_CLASSIFIER_NAMES = CLASSIFIER_NAMES |", """REVIEWED_QUESTION_IDS = ('05a8454a6a71435a557db2cc', '0eb69f6e47ab63c055e2df90')
REVIEWED_RUNNER_SHA256 = 'b1543ea655e005af6b14c280a0695b877084fc14f19024c569a4f70ebc583f4a'
REVIEWED_MANIFEST_SHA256 = 'a76fcc081d88c7293e69713cf6f95c4b7b721aeaca62704ce10e3fb2ce435f10'
REVIEWED_PARENT_MANIFEST_SHA256 = '4630c7c9a41badad1b1bba9e800aa9b5e5296de34dc96aef10c441769e21543c'
REVIEWED_PREFIX_SHA256 = '745aa44dc7f447f14dd767ec40aa317967b57bcf422eb7df53af3ef9d7c83b89'
REVIEWED_AUTHORIZATION_SHA256 = '2ebcc3400ba37674d5d8ddb555ed613ec917a63124010236e9ce921bca345947'
REVIEWED_BINDING_SHA256 = 'dfbfc3aee9901e8bf6885857665c954b6269b373fd75bebcac2818fda25bdea5'
# Fail closed until the reviewed approval artifact is finalized and pinned.
REVIEWED_APPROVAL_SHA256 = None
REVIEWED_POLICY = {'version': 'azure-prompt-rejection-v2-reviewed-retry', 'category': 'provider_policy_rejection',
                   'max_rejections_per_question': 3, 'prior_max_rejections_per_question': 2,
                   'reviewed_question_ids': list(REVIEWED_QUESTION_IDS), 'first_reviewed_round': 32,
                   'retry_request': 'unchanged', 'persistent_action': 'park_pending_then_attention', 'scope': ['single_agent']}
V4_CLASSIFIER_NAMES = CLASSIFIER_NAMES |""")
s=s.replace('def load_v4_classifier(path, u):','def load_v4_classifier(path, u, expected_sha256=V4_RUNNER_SHA256):')
s=s.replace('u.sha256(path) == V4_RUNNER_SHA256', 'u.sha256(path) == expected_sha256')
s=s.replace('def __init__(self, root, run, historical_export, amendment, provider_amendment=None):','def __init__(self, root, run, historical_export, amendment, provider_amendment=None, reviewed_provider_retry_amendment=None):')
s=s.replace('        self.classifier_transition = None','''        self.reviewed_provider_retry_amendment = Path(reviewed_provider_retry_amendment) if reviewed_provider_retry_amendment else None
        self.u.require(not self.reviewed_provider_retry_amendment or self.provider_amendment, 'Reviewed retry requires the prior provider amendment')
        self.reviewed_retry_lineage = None
        self.reviewed_prefix_checkpoint = None
        self.classifier_transition = None''')
s=s.replace("classifier, version, current_ast = load_v4_classifier(self.root / 'scripts/v3_full_run.py', u)","classifier, version, current_ast = load_v4_classifier(self.root / 'scripts/v3_full_run.py', u, REVIEWED_RUNNER_SHA256 if self.reviewed_provider_retry_amendment else V4_RUNNER_SHA256)\n            if self.reviewed_provider_retry_amendment:\n                u.require(current_ast == self.reviewed_retry_lineage['historical_v4_AST_sha256'], 'Reviewed retry changed classifier semantics')")
s=s.replace("        u.require(self.manifest == expected, 'Provider amendment changed unapproved manifest fields')", """        target_manifest, target_hash = self.manifest, self.manifest_hash
        if self.reviewed_provider_retry_amendment:
            target_path = self.reviewed_provider_retry_amendment / 'pre_apply_metadata/run_manifest.json'
            target_manifest, target_hash = u.read_json(target_path), self.remember(target_path)
        u.require(target_manifest == expected, 'Provider amendment changed unapproved manifest fields')""")
s=s.replace("and approval.get('manifest_sha256') == self.manifest_hash", "and approval.get('manifest_sha256') == target_hash")
needle="                                     'scope': 'Single-agent provider rejection before usable completion; all V3 decisions unchanged'}\n"
s=s.replace(needle,needle+"        if self.reviewed_provider_retry_amendment:\n            self.validate_reviewed_retry_lineage(target_manifest, target_hash)\n")
insert='''    def validate_reviewed_retry_lineage(self, before, before_hash):
        """Bind the exact reviewed third retry to the stopped 31-seal prefix."""
        u = self.u
        amendment = self.reviewed_provider_retry_amendment.resolve()
        u.require(amendment.is_relative_to((self.run / 'amendments').resolve())
                  and amendment.name == '0005-user-reviewed-provider-retry', 'Reviewed retry amendment outside its exact run scope')
        u.require(REVIEWED_APPROVAL_SHA256 is not None, 'Reviewed retry approval SHA is not finalized')
        u.require(before_hash == REVIEWED_PARENT_MANIFEST_SHA256 and self.manifest_hash == REVIEWED_MANIFEST_SHA256,
                  'Reviewed retry manifest pin changed')
        expected = json.loads(json.dumps(before))
        expected['source_sha256']['scripts/v3_full_run.py'] = REVIEWED_RUNNER_SHA256
        expected['provider_rejection_policy'] = REVIEWED_POLICY
        u.require(self.manifest == expected, 'Reviewed retry changed unapproved manifest fields')
        archived_runner = amendment / 'pre_apply_source/scripts/v3_full_run.py'
        self.remember(archived_runner)
        _, version, old_ast = load_v4_classifier(archived_runner, u)
        u.require(version == self.manifest['classifier_version'] == 'v3-infrastructure-v4', 'Reviewed retry changed classifier version')
        authorization_path, prefix_path = amendment / 'authorization.json', amendment / 'sealed_prefix_inventory.json'
        binding_path, approval_path = amendment / 'pre_apply_binding.json', amendment / 'root_approval.json'
        for path, expected_hash in ((authorization_path, REVIEWED_AUTHORIZATION_SHA256), (prefix_path, REVIEWED_PREFIX_SHA256),
                                    (binding_path, REVIEWED_BINDING_SHA256), (approval_path, REVIEWED_APPROVAL_SHA256)):
            u.require(self.remember(path) == expected_hash, 'Reviewed retry approval/archive pin changed: ' + path.name)
        approval, authorization, binding = map(u.read_json, (approval_path, authorization_path, binding_path))
        u.require(approval.get('status') == 'APPROVED_OPERATIONAL_RECOVERY_ONLY'
                  and approval.get('runner_sha256') == REVIEWED_RUNNER_SHA256
                  and approval.get('manifest_sha256') == self.manifest_hash
                  and approval.get('inference_unchanged') is True
                  and approval.get('historical_accepted_choices_unchanged') is True
                  and approval.get('sealed_prefix_inventory_sha256') == REVIEWED_PREFIX_SHA256
                  and approval.get('authorization_sha256') == REVIEWED_AUTHORIZATION_SHA256,
                  'Reviewed retry lacks exact approved provenance')
        u.require(authorization.get('question_ids') == list(REVIEWED_QUESTION_IDS)
                  and authorization.get('prior_provider_rejections_each') == 2
                  and authorization.get('first_reviewed_round') == 32
                  and authorization.get('additional_provider_rejection_allowance_each') == 1
                  and authorization.get('accepted_results_must_not_be_rerun') == N - 2
                  and authorization.get('provider_safeguards_prompts_model_budgets_tools_and_pricing_unchanged') is True
                  and authorization.get('generic_infrastructure_recovery') == 'unchanged original first-clean rules'
                  and authorization.get('request') == 'unchanged frozen single-agent benchmark execution'
                  and bool(authorization.get('human_request')), 'Reviewed retry authorization scope changed')
        u.require(binding.get('status') == 'STOPPED_PRE_RETRY_STATE_SNAPSHOTTED'
                  and binding.get('parent_manifest_sha256') == before_hash
                  and binding.get('parent_runner_sha256') == V4_RUNNER_SHA256
                  and binding.get('selected_accepted') == N - 2
                  and binding.get('remaining_question_ids') == list(REVIEWED_QUESTION_IDS)
                  and binding.get('sealed_rounds') == 31 and binding.get('sealed_prefix_files') == 155
                  and binding.get('sealed_prefix_inventory_sha256') == REVIEWED_PREFIX_SHA256,
                  'Reviewed retry stopped-prefix binding changed')
        u.require(set(binding.get('core_metadata_sha256') or {}) == {'acceptance.json', 'operations.jsonl', 'run_manifest.json', 'supervisor_state.json'},
                  'Reviewed retry core metadata inventory changed')
        for name, expected_hash in binding['core_metadata_sha256'].items():
            u.require(self.remember(amendment / 'pre_apply_metadata' / name) == expected_hash, 'Reviewed retry archived metadata changed')
        for name, expected_hash in before['source_sha256'].items():
            u.require(self.remember(amendment / 'pre_apply_source' / name) == expected_hash, 'Reviewed retry archived frozen source changed')
        u.require(set(before.get('pinned_files_sha256') or {}) == PUBLIC_PINS, 'Reviewed retry public pins changed')
        for name, expected_hash in before['pinned_files_sha256'].items():
            u.require(self.remember(amendment / 'pre_apply_public' / name) == expected_hash, 'Reviewed retry archived public pin changed')
        prefix = u.read_json(prefix_path)
        expected_paths = {f'attempts/single_agent/round-{number:04}{suffix}' for number in range(1, 32)
                          for suffix in ('.batch.json', '.batch.seal.json', '.json', '.json.run.json', '.csv')}
        u.require(set(prefix) == expected_paths, 'Reviewed retry sealed-prefix inventory scope changed')
        for relative, metadata in prefix.items():
            path = self.run / relative
            u.require(set(metadata) == {'sha256', 'bytes'} and self.remember(path) == metadata['sha256']
                      and path.stat().st_size == metadata['bytes'], 'Reviewed retry historical sealed-prefix artifact changed')
        ledger = u.read_json(amendment / 'pre_apply_metadata/acceptance.json')
        u.require(ledger.get('protocol') == 'v3-first-runtime-clean-terminal' and ledger.get('selection') == 'first_runtime_clean_terminal'
                  and ledger.get('complete') is False, 'Reviewed retry archived selection protocol changed')
        prior = self.slice(ledger, 'single_agent')
        current = self.ledger_slices['single_agent']
        u.require(len(prior['selections']) == N - 2 and set(current['selections']) - set(prior['selections']) == set(REVIEWED_QUESTION_IDS)
                  and all(current['selections'].get(qid) == choice for qid, choice in prior['selections'].items()),
                  'Reviewed retry changed historical accepted selections')
        self.reviewed_prior_rows = {}
        prior_canonical = amendment / 'pre_apply_metadata/full_single_agent.json'
        self.remember(prior_canonical)
        for row in u.iter_result_rows(prior_canonical):
            qid = row.get('question_id')
            u.require(qid in prior['selections'] and qid not in self.reviewed_prior_rows and not row.get('attempt_history'),
                      'Reviewed retry archived canonical cohort changed')
            digest = u.semantic_hash(row)
            u.require(self.row_hashes['single_agent'].get(qid) == digest, 'Reviewed retry changed historical accepted rows')
            self.reviewed_prior_rows[qid] = digest
        u.require(set(self.reviewed_prior_rows) == set(prior['selections']), 'Reviewed retry archived canonical incomplete')
        prior_identity = amendment / 'pre_apply_metadata/full_single_agent.json.run.json'
        self.remember(prior_identity)
        u.require(u.read_json(prior_identity) == self.manifest['run_identities']['single_agent'], 'Reviewed retry archived identity changed')
        state_path, provider_path = amendment / 'pre_apply_metadata/supervisor_state.json', amendment / 'pre_apply_metadata/provider_rejections.json'
        state, provider = u.read_json(state_path), u.read_json(provider_path)
        self.remember(provider_path)
        u.require(state.get('phase') == 'attention' and state.get('attention') is True
                  and state.get('reason') == 'provider_policy_review_required' and state.get('active') == {}
                  and state.get('selected_arms') == ['single_agent'] and state.get('accepted', {}).get('single_agent') == N - 2
                  and state.get('accepted', {}).get('multi_agent') == N
                  and state.get('counts', {}).get('single_agent') == prior['counts'], 'Reviewed retry prior terminal state changed')
        issues = state.get('issues') or []
        u.require(len(issues) == 2 and {item.get('question_id') for item in issues} == set(REVIEWED_QUESTION_IDS)
                  and all(item.get('arm') == 'single_agent' and item.get('provider_rejections') == 2
                          and item.get('status') == 'pending' and item.get('action') == 'external_policy_review_required' for item in issues),
                  'Reviewed retry prior terminal issues changed')
        expected_counts = {qid: 2 for qid in REVIEWED_QUESTION_IDS}
        u.require(provider.get('classifier_version') == 'v3-infrastructure-v4' and provider.get('policy') == V4_POLICY
                  and provider.get('source') == 'reconstructed_from_sealed_immutable_attempts'
                  and provider.get('rejections') == {'single_agent': expected_counts}
                  and provider.get('parked_question_ids') == {'single_agent': list(REVIEWED_QUESTION_IDS)}
                  and provider.get('pending_question_ids') == {'single_agent': list(REVIEWED_QUESTION_IDS)},
                  'Reviewed retry prior provider counters changed')
        self.reviewed_prior_slice = prior
        self.reviewed_retry_lineage = {'amendment_directory': str(amendment), 'approval_sha256': REVIEWED_APPROVAL_SHA256,
            'authorization_sha256': REVIEWED_AUTHORIZATION_SHA256, 'pre_apply_binding_sha256': REVIEWED_BINDING_SHA256,
            'sealed_prefix_inventory_sha256': REVIEWED_PREFIX_SHA256, 'sealed_prefix_rounds': list(range(1, 32)),
            'parent_manifest_sha256': before_hash, 'current_manifest_sha256': self.manifest_hash,
            'historical_v4_AST_sha256': old_ast, 'classifier_changed': False, 'policy': REVIEWED_POLICY,
            'historical_single_accepted': N - 2, 'historical_accepted_choices_preserved': True,
            'scope': 'One additional unchanged provider-rejection retry for the two human-reviewed IDs from round32; generic runtime recovery unchanged.'}

    def provider_limit(self, question_id, round_number):
        if (self.reviewed_provider_retry_amendment and question_id in REVIEWED_QUESTION_IDS and round_number >= 32):
            return REVIEWED_POLICY['max_rejections_per_question']
        return V4_POLICY['max_rejections_per_question']

    def validate_reviewed_prefix_checkpoint(self, selected, choices, counts, provider_counts):
        u = self.u
        u.require(selected == self.reviewed_prior_rows and choices == self.reviewed_prior_slice['selections']
                  and counts == self.reviewed_prior_slice['counts']
                  and dict(provider_counts) == {qid: 2 for qid in REVIEWED_QUESTION_IDS},
                  'Reviewed retry historical first-clean replay/counters changed')
        self.reviewed_prefix_checkpoint = {'through_round': 31, 'accepted': len(selected), 'counts': dict(counts),
            'provider_rejections': dict(provider_counts), 'canonical_rows_sha256': u.semantic_hash(selected),
            'accepted_ledger_slice_sha256': u.semantic_hash({'selections': choices, 'counts': counts})}

'''
s=s.replace('    def validate_sources(self):',insert+'    def validate_sources(self):')
s=s.replace("u.require(all(provider_counts[qid] < V4_POLICY['max_rejections_per_question'] for qid in ids),", "u.require(all(provider_counts[qid] < self.provider_limit(qid, number) for qid in ids),")
s=s.replace("            rounds.append(number)","""            if self.reviewed_provider_retry_amendment and arm == 'single_agent' and number >= 32:
                u.require(self.reviewed_prefix_checkpoint is not None and set(ids) <= set(REVIEWED_QUESTION_IDS),
                          'Reviewed retry scheduled outside the approved post-prefix cohort')
            rounds.append(number)""")
s=s.replace("            u.require(seen == set(ids), 'Scheduled placeholders omitted')", """            u.require(seen == set(ids), 'Scheduled placeholders omitted')
            if self.reviewed_provider_retry_amendment and arm == 'single_agent' and number == 31:
                self.validate_reviewed_prefix_checkpoint(selected, choices, counts, provider_counts)""")
s=s.replace("        u.require(selected == self.row_hashes[arm]", "        if self.reviewed_provider_retry_amendment and arm == 'single_agent':\n            u.require(rounds[:31] == list(range(1, 32)) and self.reviewed_prefix_checkpoint is not None, 'Reviewed retry historical round prefix changed')\n        u.require(selected == self.row_hashes[arm]")
s=s.replace('def deliver(root, run, historical_export, amendment, output, provider_amendment=None):','def deliver(root, run, historical_export, amendment, output, provider_amendment=None, reviewed_provider_retry_amendment=None):')
s=s.replace('gate = PairedGate(root, run, historical_export, amendment, provider_amendment).validate()', 'gate = PairedGate(root, run, historical_export, amendment, provider_amendment, reviewed_provider_retry_amendment).validate()')
s=s.replace("'classifier_transition': gate.classifier_transition, 'classifier_transition_actual_replay': gate.transition_replay,", "'classifier_transition': gate.classifier_transition, 'classifier_transition_actual_replay': gate.transition_replay,\n               'reviewed_provider_retry_lineage': gate.reviewed_retry_lineage, 'reviewed_provider_retry_prefix_replay': gate.reviewed_prefix_checkpoint,")
s=s.replace("    parser.add_argument('--output-dir', required=True, type=Path)", "    parser.add_argument('--reviewed-provider-retry-amendment-dir', type=Path)\n    parser.add_argument('--output-dir', required=True, type=Path)")
s=s.replace('args.output_dir, args.provider_rejection_amendment_dir)', 'args.output_dir, args.provider_rejection_amendment_dir, args.reviewed_provider_retry_amendment_dir)')
p.write_text(s)
