"""Transform private full-cohort fixture to gate-only testing; never import scorer."""
import json
from pathlib import Path
source=Path('/private/tmp/v3-paired-provider-amendment/test_v4_paired_export.py').read_text()
needle="        before = {name: u.sha256(run / name) for name in ['run_manifest.json', 'acceptance.json', 'full_multi_agent.json', 'full_single_agent.json', 'full_direct_llm.json']}"
insert='''        # Independent gate-only positive replay; both cohorts complete, no scorer.
        gate = candidate.PairedGate(project, run, historical, amendment, provider_amendment).validate()
        assert gate.transition_replay['multi_agent'] == {'attempted':2931,'unchanged':2931,'provider_rejections':[]}
        assert gate.transition_replay['single_agent']['attempted'] == 2932
        assert gate.transition_replay['single_agent']['unchanged'] == 2931
        assert len(gate.transition_replay['single_agent']['provider_rejections']) == 1
        assert 'scripts.v3_full_analysis' not in sys.modules and 'utils.answer_scorer' not in sys.modules
        checks.append('authorized_second_clean_full2931each_gate_only_no_scorer')
        # Two provider rejections then a third clean cohort must fail historical scheduling gate.
        rejected = raw_row('single_agent',0)
        rejection = {'error':{'code':'content_filter','param':'prompt','status':400,'innererror':{'code':'ResponsibleAIPolicyViolation','content_filter_result':{'jailbreak':{'detected':True,'filtered':True}}}}}
        rejected.update(error='Max retries exceeded: Error code: 400 - ' + repr(rejection),latency=1,tokens={'total':0,'per_call':[{'usage_available':False}]})
        batch('single_agent',2,inputs[:1],[rejected])
        third = batch('single_agent',3,inputs,[raw_row('single_agent',i) for i in range(2931)])
        write_json(run/'full_single_agent.json',third)
        ledger['arms']['single_agent']={row['question_id']:{'attempt_file':'attempts/single_agent/round-0003.json','round':3,'runtime_failures_before':2 if i==0 else 0,'selection':'runtime_clean_terminal','runtime_clean':True} for i,row in enumerate(third)}
        ledger['counts']['single_agent']={'raw_attempts':2933,'runtime_failures':2,'intrinsic_errors':0}
        write_json(run/'acceptance.json',ledger)
        refusal('Historical batch scheduled a persistently rejected request beyond its unchanged-retry limit')
        evidence={'status':'PASS','checks':checks,'helper_sha256':u.sha256(CANDIDATE),'runner_sha256':u.sha256(runner),'synthetic_questions_per_arm':2931,'actual_live_data_scored':False,'scorer_imported':False,'private_fixture_only':True,'gate_only_no_scoring':True}
        write_json(Path('/private/tmp/v3-independent-provider-export-review/paired_historical_limit_gate_results.json'),evidence)
        print(json.dumps(evidence))
        return
'''
assert source.count(needle)==1
source=source.replace(needle,insert+needle)
path=Path('/private/tmp/v3-independent-provider-export-review/transformed_paired_guard_review.py')
path.write_text(source)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
