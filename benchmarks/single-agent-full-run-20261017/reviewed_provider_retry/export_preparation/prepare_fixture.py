from pathlib import Path
repo=Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
s=(repo/'benchmarks/single-agent-full-run-20261017/provider_recovery/validation_sources/test_v4_paired_export.py').read_text()
s=s.replace("/private/tmp/v3-paired-provider-amendment/export_complete_v3_single.py", "/private/tmp/single-reviewed-retry-export-20261017/export_complete_v3_single.py")
s=s.replace("Path('/private/tmp/v3-provider-rejection-candidate/v3_full_run.py')", "REAL / 'output/experiments/gpt-6-luna/v3_full_2931_20261015/amendments/0005-user-reviewed-provider-retry/pre_apply_source/scripts/v3_full_run.py'")
s=s.replace("runner.write_bytes(REAL / 'output/experiments/gpt-6-luna/v3_full_2931_20261015/amendments/0005-user-reviewed-provider-retry/pre_apply_source/scripts/v3_full_run.py'.read_bytes())", "runner.write_bytes((REAL / 'output/experiments/gpt-6-luna/v3_full_2931_20261015/amendments/0005-user-reviewed-provider-retry/pre_apply_source/scripts/v3_full_run.py').read_bytes())")
s=s.replace("        source, families = run / 'source.csv', run / 'families.csv'", "        for i, qid in enumerate(candidate.REVIEWED_QUESTION_IDS): inputs[i]['question_id'] = qid\n        source, families = run / 'source.csv', run / 'families.csv'")
a=s.index("        for arm in ('multi_agent', 'single_agent'):")
b=s.index("        write_json(run / 'full_direct_llm.json'",a)
s=s[:a]+'''        def link_rows(arm, rows):
            for row in rows:
                origin = run / 'attempts' / arm / row['audit_artifacts']['path']
                dest = run / row['audit_artifacts']['path']
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists(): os.link(origin, dest)

        def reject_row(index, transport=False):
            row = raw_row('single_agent', index)
            payload = {'error': {'code':'content_filter','param':'prompt','status':400,'innererror':{'code':'ResponsibleAIPolicyViolation','content_filter_result':{'jailbreak':{'detected':True,'filtered':True}}}}}
            row.update(error='Max retries exceeded: Error code: 400 - ' + repr(payload), latency=99,
                       tokens={'total':999,'per_call':[{'usage_available':False}]})
            if transport: row['error']='Max retries exceeded: request timed out'
            return row

        multi_rows = batch('multi_agent', 1, inputs, [raw_row('multi_agent', i) for i in range(2931)])
        link_rows('multi_agent', multi_rows)
        write_json(run / 'full_multi_agent.json', multi_rows)
        write_json(run / 'full_multi_agent.json.run.json', identities['multi_agent'])
        ledger['arms']['multi_agent']={row['question_id']:{'attempt_file':'attempts/multi_agent/round-0001.json','round':1,
             'runtime_failures_before':0,'selection':'runtime_clean_terminal','runtime_clean':True} for row in multi_rows}
        ledger['counts']['multi_agent']={'raw_attempts':2931,'runtime_failures':0,'intrinsic_errors':0}
        prior_rows=[]
        for number in range(1,30):
            indices=list(range(2+(number-1)*101,min(2931,2+number*101)))
            rows=batch('single_agent',number,[inputs[i] for i in indices],[raw_row('single_agent',i) for i in indices])
            link_rows('single_agent',rows);prior_rows.extend(rows)
            for row in rows:
                ledger['arms']['single_agent'][row['question_id']]={'attempt_file':f'attempts/single_agent/round-{number:04}.json','round':number,
                     'runtime_failures_before':0,'selection':'runtime_clean_terminal','runtime_clean':True}
        for number in (30,31): batch('single_agent',number,inputs[:2],[reject_row(0),reject_row(1)])
        ledger['counts']['single_agent']={'raw_attempts':2933,'runtime_failures':4,'intrinsic_errors':0}
        write_json(run/'full_single_agent.json',prior_rows)
        write_json(run/'full_single_agent.json.run.json',identities['single_agent'])
''' +s[b:]
b=s.index('        checks = []')
s=s[:b]+'''        reviewed = run/'amendments/0005-user-reviewed-provider-retry'
        metadata = reviewed/'pre_apply_metadata'
        state={'phase':'attention','attention':True,'reason':'provider_policy_review_required','active':{},'selected_arms':['single_agent'],
               'accepted':{'single_agent':2929,'multi_agent':2931,'direct_llm':0},'counts':ledger['counts'],
               'issues':[{'arm':'single_agent','question_id':qid,'provider_rejections':2,'status':'pending','action':'external_policy_review_required'} for qid in candidate.REVIEWED_QUESTION_IDS]}
        for name in ('run_manifest.json','acceptance.json','operations.jsonl','full_single_agent.json','full_single_agent.json.run.json'):
            dest=metadata/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/name,dest)
        write_json(metadata/'supervisor_state.json',state)
        write_json(metadata/'provider_rejections.json',{'classifier_version':'v3-infrastructure-v4','policy':candidate.V4_POLICY,
            'source':'reconstructed_from_sealed_immutable_attempts','rejections':{'single_agent':{qid:2 for qid in candidate.REVIEWED_QUESTION_IDS}},
            'parked_question_ids':{'single_agent':list(candidate.REVIEWED_QUESTION_IDS)},'pending_question_ids':{'single_agent':list(candidate.REVIEWED_QUESTION_IDS)}})
        for name in current['source_sha256']:
            dst=reviewed/'pre_apply_source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/name,dst)
        for name in candidate.PUBLIC_PINS:
            dst=reviewed/'pre_apply_public'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/name,dst)
        prefix={str(path.relative_to(run)):{'sha256':u.sha256(path),'bytes':path.stat().st_size}
                for path in (run/'attempts/single_agent').glob('round-*') if path.is_file()}
        assert len(prefix)==155
        write_json(reviewed/'sealed_prefix_inventory.json',prefix)
        authorization={'question_ids':list(candidate.REVIEWED_QUESTION_IDS),'prior_provider_rejections_each':2,'first_reviewed_round':32,
            'additional_provider_rejection_allowance_each':1,'accepted_results_must_not_be_rerun':2929,
            'provider_safeguards_prompts_model_budgets_tools_and_pricing_unchanged':True,'generic_infrastructure_recovery':'unchanged original first-clean rules',
            'request':'unchanged frozen single-agent benchmark execution','human_request':'Synthetic explicit unchanged retry authorization'}
        write_json(reviewed/'authorization.json',authorization)
        binding={'status':'STOPPED_PRE_RETRY_STATE_SNAPSHOTTED','parent_manifest_sha256':u.sha256(metadata/'run_manifest.json'),
            'parent_runner_sha256':candidate.V4_RUNNER_SHA256,'selected_accepted':2929,'remaining_question_ids':list(candidate.REVIEWED_QUESTION_IDS),
            'sealed_rounds':31,'sealed_prefix_files':155,'sealed_prefix_inventory_sha256':u.sha256(reviewed/'sealed_prefix_inventory.json'),
            'core_metadata_sha256':{name:u.sha256(metadata/name) for name in ('acceptance.json','operations.jsonl','run_manifest.json','supervisor_state.json')}}
        write_json(reviewed/'pre_apply_binding.json',binding)
        runner.write_bytes(Path('/private/tmp/single-reviewed-retry-20261017/v3_full_run.py').read_bytes())
        current['source_sha256']['scripts/v3_full_run.py']=u.sha256(runner)
        current['provider_rejection_policy']=candidate.REVIEWED_POLICY
        write_json(run/'run_manifest.json',current)
        approval={'status':'APPROVED_OPERATIONAL_RECOVERY_ONLY','runner_sha256':u.sha256(runner),'manifest_sha256':u.sha256(run/'run_manifest.json'),
            'inference_unchanged':True,'historical_accepted_choices_unchanged':True,'sealed_prefix_inventory_sha256':u.sha256(reviewed/'sealed_prefix_inventory.json'),
            'authorization_sha256':u.sha256(reviewed/'authorization.json')}
        write_json(reviewed/'root_approval.json',approval)
        for constant,path in [('REVIEWED_PARENT_MANIFEST_SHA256',metadata/'run_manifest.json'),('REVIEWED_MANIFEST_SHA256',run/'run_manifest.json'),
            ('REVIEWED_PREFIX_SHA256',reviewed/'sealed_prefix_inventory.json'),('REVIEWED_AUTHORIZATION_SHA256',reviewed/'authorization.json'),
            ('REVIEWED_BINDING_SHA256',reviewed/'pre_apply_binding.json'),('REVIEWED_APPROVAL_SHA256',reviewed/'root_approval.json')]:
            setattr(candidate,constant,u.sha256(path))
        clean_rows=batch('single_agent',32,inputs[:2],[raw_row('single_agent',0),raw_row('single_agent',1)])
        link_rows('single_agent',clean_rows)
        for row in clean_rows:
            ledger['arms']['single_agent'][row['question_id']]={'attempt_file':'attempts/single_agent/round-0032.json','round':32,
                'runtime_failures_before':2,'selection':'runtime_clean_terminal','runtime_clean':True}
        ledger['counts']['single_agent']={'raw_attempts':2935,'runtime_failures':4,'intrinsic_errors':0}
        write_json(run/'acceptance.json',ledger)
        write_json(run/'full_single_agent.json',prior_rows+clean_rows)
        checks=[]
        def refusal(expected):
            output=base/f'must_not_exist_{len(checks)}'
            try: candidate.deliver(project,run,historical,amendment,output,provider_amendment,reviewed)
            except ValueError as error: assert expected in str(error),str(error)
            else: raise AssertionError('Invalid reviewed export accepted')
            assert not output.exists()
            assert 'scripts.v3_full_analysis' not in sys.modules and 'utils.answer_scorer' not in sys.modules
            checks.append(expected)
        incomplete=json.loads(json.dumps(ledger));incomplete['arms']['single_agent'].pop(candidate.REVIEWED_QUESTION_IDS[0])
        write_json(run/'acceptance.json',incomplete);refusal('incomplete acceptance cohort');write_json(run/'acceptance.json',ledger)
        saved=candidate.REVIEWED_APPROVAL_SHA256;candidate.REVIEWED_APPROVAL_SHA256=None
        refusal('approval SHA is not finalized');candidate.REVIEWED_APPROVAL_SHA256=saved
        oldapproval=provider_amendment/'root_approval.json';savedapproval=oldapproval.read_bytes()
        wrong=json.loads(savedapproval);wrong['manifest_sha256']=u.sha256(run/'run_manifest.json');write_json(oldapproval,wrong)
        refusal('lacks reviewed provenance');oldapproval.write_bytes(savedapproval)
        prefixpath=run/'attempts/single_agent/round-0001.batch.seal.json';savedprefix=prefixpath.read_bytes()
        prefixpath.write_bytes(savedprefix+b' ');refusal('historical sealed-prefix artifact changed');prefixpath.write_bytes(savedprefix)
        origin=metadata/'full_single_agent.json';savedrows=origin.read_bytes()
        changedrows=json.loads(savedrows);changedrows[0]['agent_answer']='changed historical answer';write_json(origin,changedrows)
        refusal('changed historical accepted rows');origin.write_bytes(savedrows)
        gate=candidate.PairedGate(project,run,historical,amendment,provider_amendment,reviewed)
        assert gate.provider_limit(candidate.REVIEWED_QUESTION_IDS[0],31)==2
        assert gate.provider_limit(candidate.REVIEWED_QUESTION_IDS[0],32)==3
        assert gate.provider_limit('unreviewed',32)==2
        checks.append('effective_cap_prior_round_and_unreviewed_boundaries')
        # Four provider rejections cannot be followed by an accepted retry; block scheduling at the third count.
        # Exercise the exact replay scheduling predicate without importing scoring or changing real data.
        from collections import Counter
        counters=Counter({candidate.REVIEWED_QUESTION_IDS[0]:3,candidate.REVIEWED_QUESTION_IDS[1]:2})
        assert not all(counters[qid] < gate.provider_limit(qid,33) for qid in candidate.REVIEWED_QUESTION_IDS)
        assert counters[candidate.REVIEWED_QUESTION_IDS[1]] < gate.provider_limit(candidate.REVIEWED_QUESTION_IDS[1],33)
        checks.append('third_provider_rejection_stops_only_exhausted_question')
        protected=['run_manifest.json','acceptance.json','full_multi_agent.json','full_single_agent.json','full_direct_llm.json']
        before={name:u.sha256(run/name) for name in protected}
        output=base/'complete_pair'
        receipt=candidate.deliver(project,run,historical,amendment,output,provider_amendment,reviewed)
        report=u.read_json(output/'paired_complete/summary.json')
        assert receipt['reviewed_provider_retry_prefix_replay']['accepted']==2929
        assert receipt['reviewed_provider_retry_prefix_replay']['through_round']==31
        assert receipt['reviewed_provider_retry_lineage']['classifier_changed'] is False
        assert receipt['reviewed_provider_retry_lineage']['policy']==candidate.REVIEWED_POLICY
        assert receipt['first_clean_replay_counts']['single_agent']=={'raw_attempts':2935,'runtime_failures':4,'intrinsic_errors':0}
        assert receipt['status']=='V3_SINGLE_COMPLETE_DIRECT_PENDING' and not receipt['benchmark_complete']
        assert set(report['arms'])==set(candidate.ARMS) and 'all_arms_accepted_metrics' not in report
        assert not (output/'validated_paired_input_snapshot/full_direct_llm.json').exists()
        assert all(not (output/'validated_paired_input_snapshot'/name).exists() for name in candidate.SNAPSHOT_OMIT_PUBLIC)
        assert report['arms']['single_agent']['metrics']['latency_seconds']['total']==5862
        assert report['arms']['single_agent']['metrics']['total_tokens']['total']==2931*140
        assert math.isclose(report['arms']['single_agent']['metrics']['estimated_cost_usd']['total'],2931*.0000193,rel_tol=1e-12)
        pair=report['comparisons'][0];assert pair['matched_questions']==2931 and pair['left_only']==pair['right_only']==0
        assert pair['latency_seconds_difference_observed_pairs']['estimate']==1
        assert u.read_json(output/'global_partial/summary.json')['status']=='PARTIAL_PROGRESS_DO_NOT_REPORT_AS_FINAL'
        csv.field_size_limit(sys.maxsize)
        for path in (output/'paired_complete').glob('*.csv'):
            with path.open(newline='') as stream:
                rows=list(csv.DictReader(stream));assert all(None not in row for row in rows)
                if path.name=='per_question.csv':assert len(rows)==5862
        assert len((output/'paired_complete/per_question.jsonl').read_text().splitlines())==5862
        assert before=={name:u.sha256(run/name) for name in before}
        checks.extend(['full2931x2_positive_0005_export','historical31seal2929_checkpoint','unchanged_v4_classifier_AST',
            'runtime_measurements_excluded','paired_only_metrics_and_CIs','csv_jsonl5862_parse','fixture_protected_hashes_unchanged'])
        evidence={'status':'PASS','checks':checks,'helper_sha256':u.sha256(CANDIDATE),'test_sha256':u.sha256(Path(__file__)),
            'actual_live_data_scored':False,'synthetic_questions_per_arm':2931,'private_fixture_only':True}
        write_json(Path('/private/tmp/single-reviewed-retry-export-20261017/test_results.json'),evidence)
        print(json.dumps(evidence))

if __name__=='__main__':main()
'''
Path('/private/tmp/single-reviewed-retry-export-20261017/test_reviewed_export.py').write_text(s)
