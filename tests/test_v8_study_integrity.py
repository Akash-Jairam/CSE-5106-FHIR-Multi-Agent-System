"""Independent, entirely synthetic V8 completion-integrity fixtures.

Only stdlib, the pure validator and existing audit-store serialization are used.
No model/core/scorer/configuration/QA module or live result is imported.
"""
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace

from scripts import audit_store as store
from scripts import v8_study_integrity as guard


def fp(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def compact(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def bills(count=1,unknown=False):
    calls=[{'prompt':10,'completion':2,'cached':0,'usage_available':not unknown} for _ in range(count)]
    totals={'prompt_tokens':10*count,'completion_tokens':2*count,'total_tokens':12*count,
            'llm_calls':count,'cost':None if unknown else 0.0}
    return totals,calls


def make_candidate(*,eligible=True,phase='natural_primary',label='A'):
    workspace={'Observation':[{'resourceType':'Observation','id':'o1',
        'subject':{'reference':'Patient/fixture-patient'},'valueQuantity':{'value':7}}]}
    five={'answer':7,'evidence':[{'resource_id':'Observation/o1','field_path':'valueQuantity.value','value':7}],
          'operation':'Read the original numeric source value','contributing_ids':['Observation/o1'],'limitation':None}
    code="r=retrieved_resources['Observation'][0]\nanswer=r['valueQuantity']['value']\nevidence=[{'resource_id':'Observation/o1','field_path':'valueQuantity.value','value':answer}]\noperation='Read the original numeric source value'\ncontributing_ids=['Observation/o1']\nlimitation=None\n"
    check={'issues':[],'n_valid':len(five['evidence']) if eligible else 0,
           'results':[{'citation':deepcopy(c),'status':'valid','warnings':[]} for c in five['evidence']] if eligible else []}
    render={key:deepcopy(five[key]) for key in ('answer','evidence','limitation')} if eligible else {'answer':None,'evidence':[],'limitation':'No executable measurement computation'}
    trace=[{'role':'system','content':'Synthetic source policy'},
           {'role':'user','content':'Synthetic question/context'}]
    if eligible:
        trace += [{'role':'assistant','content':None,'tool_calls':[{'id':'final','type':'function','function':{
            'name':'execute_python_code','arguments':json.dumps({'code':code,'mode':'final'})}}]},
                  {'role':'tool','tool_call_id':'final','name':'execute_python_code','content':json.dumps({**deepcopy(five),'mode':'final'})}]
    trace += [{'role':'assistant','content':json.dumps(render),'tool_calls':None},
              {'role':'assistant','content':None,'tool_calls':[{'id':'automatic_field_check_1','type':'function','function':{
                  'name':'check_fields','arguments':json.dumps({'citations':render['evidence']})}}]},
              {'role':'tool','tool_call_id':'automatic_field_check_1','name':'check_fields','content':json.dumps(check)}]
    total,calls=bills(2 if eligible else 1)
    tool_count=2 if eligible else 1
    limits=(200000,30,30) if label=='A' else (60000,6,6)
    diagnostic={'phase':phase,'token_cap':limits[0],'tool_cap':limits[1],'sdk_cap':limits[2],
        'tokens':total['total_tokens'],'tool_calls':tool_count,'model_calls':len(calls),
        'fatal':False,'eligible':eligible,'usage_known':True,'errors':[],'budget_stop':None,
        'skip_reason':None if eligible else 'no surviving explicit final artifact'}
    tagged=[{**deepcopy(m),'phase':phase} for m in trace]
    artifact={**deepcopy(five),'code':code,'mode':'final','workspace_sha256':fp(workspace)} if eligible else None
    binding={'execution_tool_call_id':'final','execution_trace_index':3,
        'source_check_tool_call_id':'automatic_field_check_1','source_check_trace_index':6,
        'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'workspace_sha256':fp(workspace)} if eligible else {}
    snapshot={'version':1,'kind':'candidate_snapshot','candidate':label,'phase':phase,
        'released':{**render,'error':None,'revision':{'draft':deepcopy(render),'final':deepcopy(render),
                    'final_check_issues':[]},'budget':{'max_total_tokens':limits[0],'max_tool_calls':limits[1],
                    'tool_calls_run':tool_count,'stopped':None}},
        'artifact':artifact,'workspace':workspace,'source_check':check if eligible else None,'binding':binding,
        'workspace_sha256':fp(workspace),'trace_sha256':fp(trace),'tagged_trace_sha256':fp(tagged),
        'usage':{'total_usage':total,'per_call':calls,'raw_accounting':{'total_usage':deepcopy(total),
            'call_usage':deepcopy(calls),'tool_calls_made':tool_count},'observer':{}},'diagnostics':diagnostic}
    return snapshot,tagged


def make_policy(mode='A',*,eligible=False,qid='synthetic-case'):
    a,trace=make_candidate(eligible=eligible)
    row={'question_id':qid,'patient_fhir_id':'fixture-patient','question':'What is the recorded synthetic value?',
         'assumption':'','true_answer':'[[7]]','template':'fixture','study_arm':mode,'study_terminal':True,
         'physical_primary_inference_executed':mode=='A'}
    question='Question: '+row['question']+'\nContext:\nPatient FHIR ID is fixture-patient.\n'
    payload={'question_sha256':fp(question),'model':'synthetic-model','base_url_sha256':fp(None),'perturbation_sha256':fp(None),
        'caps':{'max_total_tokens':200000,'max_tool_calls':30},'source_identity':{},'candidate':a,
        'trace':trace,'model_messages':[{k:v for k,v in m.items() if k!='phase'} for m in trace],
        'context_compactions':[]}
    envelope={'version':1,'kind':'v8_primary_envelope','payload':payload,'sha256':fp(payload)}
    status='primary_only' if mode=='A' else 'primary_ineligible'
    mechanism={'released_source_bound_B':None,'selected_B':False,'auxiliary_fatal':False,
               'status':status,'witness_bound':False,'probe_interactions':0}
    revision={'protocol':guard.PROTOCOL,'mode':mode,'status':status,'selected':'A','fatal':False,
        'candidate_snapshot_sha256':{'A':fp(a)},'phases':[deepcopy(a['diagnostics'])],
        'admission':None,'observations':[],'witness':None,'selection':None,'messages':[deepcopy(a)],
        'alternative_selected':False,'mechanism':mechanism}
    total=a['usage']['total_usage'];calls=deepcopy(a['usage']['per_call'])
    physical=calls if mode=='A' else []
    row.update(agent_answer=a['released']['answer'],evidence=deepcopy(a['released']['evidence']),
        limitation=a['released']['limitation'],error=None,latency=0.0,
        tokens={'prompt':total['prompt_tokens'],'completion':total['completion_tokens'],
            'total':total['total_tokens'],'llm_calls':total['llm_calls'],'cost':total['cost'],'per_call':calls},
        budget={'max_total_tokens':200000,'max_tool_calls':30,'tool_calls_run':a['diagnostics']['tool_calls'],'stopped':None},
        trace=deepcopy(trace),retrieved_fhir_resources=deepcopy(a['workspace']),revision=revision,
        mechanism=deepcopy(mechanism),physical_usage={'available':True,'observed_invocations':len(physical),
            'known_usage_invocations':len(physical),'unknown_usage_invocations':0,
            'known_tokens':{k:sum(c[k] for c in physical) for k in ('prompt','completion','cached')}},
        primary_envelope_ref='primaries/'+qid+'.json.gz',primary_envelope_sha256=hashlib.sha256(compact(envelope)).hexdigest())
    return row,envelope


def write_case(directory,qid='synthetic-case'):
    output=Path(directory);output.mkdir(parents=True,exist_ok=True)
    rows=[];envelope=None
    for mode in ('A','W','R','C'):
        row,envelope=make_policy(mode,qid=qid)
        rows.append(store.externalize_result_audit(row,output,question_id=qid))
    (output/'primaries').mkdir(exist_ok=True)
    with gzip.open(output/'primaries'/f'{qid}.json.gz','wb') as stream:stream.write(compact(envelope))
    cache=output/'case_cache'/qid
    for mode in ('post_A_base','W','R','C'):
        path=cache/mode;path.mkdir(parents=True)
        (path/'original.cache').write_bytes(b'opaque synthetic cache bytes')
    raw=(cache/'post_A_base'/'original.cache').read_bytes()
    inventory={'original.cache':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
    proof={'policy':'cold_private_A_then_identical_physical_copies_no_hardlinks','files':1,'bytes':len(raw),
           'inventory_sha256':hashlib.sha256(compact(inventory)).hexdigest()}
    for row in rows:
        row['cache_isolation']=deepcopy(proof)
        if row['study_arm']!='A':row['cache_isolation'].update(base_unchanged=True,followup=row['study_arm'],overlay_files_after=1,new_overlay_files=0)
    return rows,envelope


class CandidateIntegrityIndependentTests(unittest.TestCase):
    def test_eligible_B_retains_source_after_exact_known_billed_provider_failure(self):
        b,trace=make_candidate(label='B',phase='counterexample_investigation')
        b['usage']['observer']={'calls':[{'status':'responded'},{'status':'responded'}]}
        guard.validate_candidate(b,trace)
        total,calls=bills(3)
        b['usage'].update(total_usage=total,per_call=calls,
            raw_accounting={'total_usage':deepcopy(total),'call_usage':deepcopy(calls),'tool_calls_made':2})
        b['diagnostics'].update(tokens=36,model_calls=3)
        b['usage']['observer']['calls'].append({'status':'provider_failed','provider_error_present':True})
        counts=guard.validate_candidate(b,trace)
        self.assertEqual(counts['charged_minus_observed_model_attempts'],1)
        for mutation in ('generic_gap','missing_flag','extra_admission','A_strict'):
            altered=deepcopy(b);tagged=deepcopy(trace)
            if mutation=='generic_gap':altered['usage']['observer']['calls'][-1]['status']='responded'
            elif mutation=='missing_flag':altered['usage']['observer']['calls'][-1].pop('provider_error_present')
            elif mutation=='extra_admission':altered['usage']['observer']['calls'].append({'status':'responded'})
            else:
                altered['candidate']='A';altered['phase']='natural_primary'
                altered['diagnostics'].update(phase='natural_primary',token_cap=200000,tool_cap=30,sdk_cap=30)
                for message in tagged:message['phase']='natural_primary'
                altered['trace_sha256']=fp([{k:v for k,v in m.items() if k!='phase'} for m in tagged])
                altered['tagged_trace_sha256']=fp(tagged)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):guard.validate_candidate(altered,tagged)

    def test_oversized_native_probe_keeps_diagnostic_despite_audit_model_content(self):
        row,envelope=make_policy('R',eligible=True)
        a=envelope['payload']['candidate'];helper=guard.shared()
        b,_=make_candidate(label='B',phase='counterexample_investigation',eligible=False)
        packet={'context':helper.context_from_row({**row,'question_with_context':guard.input_text(row)}),
            'A':helper.candidate_entry(a,envelope['payload']['trace']),
            'workspace_inventory':[{'resource_type':'Observation','held_count':1}], 'retrieval_permitted':True}
        arguments={'code':'print("large diagnostic")','mode':'inspect'}
        reply={'role':'tool','name':'execute_python_code','tool_call_id':'probe',
               'content':json.dumps({'answer':'x'*9000})}
        trace=[{'role':'system','content':'Synthetic investigator','phase':b['phase']},
            {'role':'user','content':json.dumps(packet),'phase':b['phase']},
            {'role':'assistant','content':None,'tool_calls':[{'id':'probe','type':'function','function':{
                'name':'execute_python_code','arguments':json.dumps(arguments)}}],'phase':b['phase']},
            {**deepcopy(reply),'phase':b['phase'],'model_content':'Host omitted oversized probe body'}]
        b['trace_sha256']=fp([{k:v for k,v in m.items() if k!='phase'} for m in trace]);b['tagged_trace_sha256']=fp(trace)
        observation={'observation_id':'observation_1','tool_call_id':'probe','tool':'execute_python_code',
            'arguments':arguments,'actual_reply':reply,'A_program_sha256':a['binding']['code_sha256'],
            'A_workspace_sha256':a['workspace_sha256'],'workspace_before_sha256':a['workspace_sha256'],
            'workspace_after_sha256':a['workspace_sha256'],'witness_eligible':False}
        revision=row['revision'];revision['messages'].append(b)
        revision['candidate_snapshot_sha256']['B']=fp(b);revision['observations']=[observation]
        revision['status']='probe_ineligible';revision['mechanism'].update(status='probe_ineligible',probe_interactions=1)
        total,calls=deepcopy(b['usage']['total_usage']),deepcopy(b['usage']['per_call'])
        aux={'phase':'counterexample_auxiliary','token_cap':60000,'tool_cap':6,'sdk_cap':6,
            'fatal':False,'tokens':12,'model_calls':1,'tool_calls':1,'probe_interactions':1,'automatic_checks':0,
            'calls':[{'status':'responded'}],'raw_accounting':{'total_usage':total,'call_usage':calls,'tool_calls_made':1}}
        revision['phases'].append(aux);revision['admission']={'aux_token_cap':60000,'aux_tool_cap':6,'aux_sdk_cap':6}
        row['trace'].extend(trace);row['mechanism']=deepcopy(revision['mechanism'])
        row['tokens'].update(prompt=30,completion=6,total=36,llm_calls=3,cost=0.0,
                            per_call=deepcopy(a['usage']['per_call'])+calls)
        row['budget']['tool_calls_run']=3
        row['physical_usage']={'available':True,'observed_invocations':1,'known_usage_invocations':1,
            'unknown_usage_invocations':0,'known_tokens':{'prompt':10,'completion':2,'cached':0}}
        proof=guard.validate_policy(row,row,envelope)
        self.assertEqual(proof['counts']['eligible_B'],0)
        self.assertIsNone(revision['witness']);self.assertIsNone(revision['mechanism']['released_source_bound_B'])
        for mutation in ('false_witness_eligibility','different_native_reply'):
            altered=deepcopy(row)
            if mutation=='false_witness_eligibility':altered['revision']['observations'][0]['witness_eligible']=True
            else:altered['revision']['observations'][0]['actual_reply']['content']='{}'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):guard.validate_policy(altered,altered,envelope)

    def test_handwritten_eligible_and_prose_A_baselines(self):
        for eligible in (True,False):
            with self.subTest(eligible=eligible):
                snapshot,trace=make_candidate(eligible=eligible)
                counts=guard.validate_candidate(snapshot,trace)
                self.assertEqual(counts['charged_minus_observed_model_attempts'],0)
                self.assertEqual(counts['charged_minus_observed_tool_calls'],0)

    def test_eligible_candidate_rejects_stale_source_execution_or_workspace(self):
        variants=[]
        a,t=make_candidate();a['artifact']['answer']=99;variants.append((a,t))
        a,t=make_candidate();a['workspace']['Observation'][0]['valueQuantity']['value']=99;variants.append((a,t))
        a,t=make_candidate();a['binding']['source_check_trace_index']=3;variants.append((a,t))
        a,t=make_candidate();a['source_check']['results'][0]['warnings']=['unresolved time'];variants.append((a,t))
        a,t=make_candidate();a['diagnostics']['model_calls']+=1;variants.append((a,t))
        for a,t in variants:
            with self.subTest(variant=len(t)),self.assertRaises(ValueError):guard.validate_candidate(a,t)

    def test_clean_policy_counts_A_once_physically_and_reuses_conceptual_prefix(self):
        physical=[]
        for mode in ('A','W','R','C'):
            row,envelope=make_policy(mode)
            result=guard.validate_policy(row,row,envelope)
            physical.extend(result['physical_calls'])
            self.assertEqual(row['tokens']['llm_calls'],1)
        self.assertEqual(len(physical),1)

    def test_followup_cannot_change_A_prefix_or_bill_it_physically_twice(self):
        for mutation in ('prefix','physical','snapshot'):
            row,envelope=make_policy('R')
            if mutation=='prefix':row['tokens']['per_call'][0]['prompt']+=1
            elif mutation=='physical':row['physical_primary_inference_executed']=True
            else:row['revision']['messages'][0]['released']['answer']=999
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):guard.validate_policy(row,row,envelope)

    def test_unknown_and_reported_overshoot_cannot_claim_nonfatal_accounting(self):
        totals,calls=bills(unknown=True)
        with self.assertRaises(ValueError):guard.ledger(totals,calls,False,(200000,30,30),1)
        self.assertEqual(guard.ledger(totals,calls,True,(200000,30,30),1)['unknown_calls'],1)
        totals,calls=bills();totals['total_tokens']=200001
        with self.assertRaises(ValueError):guard.ledger(totals,calls,False,(200000,30,30),1)
        self.assertEqual(guard.ledger(totals,calls,True,(200000,30,30),1)['reported_phase_overshoots'],1)


class OwnedCaseIntegrityIndependentTests(unittest.TestCase):
    def test_four_policy_case_restores_owned_gzip_and_shared_A_without_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,envelope=write_case(tmp)
            proof=guard.verify_case(rows,tmp,store)
            self.assertEqual(proof['counts']['policies'],4)
            self.assertEqual(len(proof['physical_calls']),1)
            self.assertFalse(proof['fatal_physical'])

    def test_missing_duplicate_or_reference_changed_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,_=write_case(tmp)
            altered=deepcopy(rows);altered[1]['template']='another template'
            for group in (rows[:-1],rows+[deepcopy(rows[-1])],altered):
                with self.subTest(n=len(group)),self.assertRaises(ValueError):guard.verify_case(group,tmp,store)

    def test_audit_question_ownership_and_compressed_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,_=write_case(tmp)
            wrong=deepcopy(rows);wrong[1]['audit_artifacts']['question_id']='another-question'
            with self.assertRaises(ValueError):guard.verify_case(wrong,tmp,store)
            artifact=Path(tmp)/rows[0]['audit_artifacts']['path']
            artifact.write_bytes(artifact.read_bytes()[:-5])
            with self.assertRaises((ValueError,EOFError,OSError)):guard.verify_case(rows,tmp,store)

    def test_same_byte_hardlinked_cache_fork_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,_=write_case(tmp)
            base=Path(tmp)/'case_cache/synthetic-case/post_A_base/original.cache'
            fork=Path(tmp)/'case_cache/synthetic-case/R/original.cache'
            fork.unlink();fork.hardlink_to(base)
            with self.assertRaises(ValueError):guard.verify_case(rows,tmp,store)


def journal_events(calls):
    from datetime import timedelta
    client={'sdk':{'class':'openai.lib.azure.AzureOpenAI','mro':['openai.lib.azure.AzureOpenAI']},
        'http':{'class':'openai._base_client.SyncHttpxClientWrapper','mro':['openai._base_client.SyncHttpxClientWrapper','httpx2.Client']},
        'transport':{'class':'httpx2.HTTPTransport','mro':['httpx2.HTTPTransport']},'pool':None,
        'sdk_closed':False,'http_closed':False,'client_id':42,'versions':{'openai':'3.26.1','httpx2':'2.13.1'}}
    result=[]
    initial=datetime(2026,10,15,tzinfo=timezone.utc)
    for index,call in enumerate(calls,1):
        started=initial+timedelta(seconds=2*index)
        base={'version':1,'pid':123,'ppid':1,'counter':index,'process_session':'1'*32,
              'started_utc':started.isoformat(),'client':deepcopy(client),'fd_count':10}
        result.append({**deepcopy(base),'event':'start'})
        usage={'prompt_tokens':call['prompt'],'completion_tokens':call['completion'],
               'total_tokens':call['prompt']+call['completion'],'cached_prompt_tokens':call.get('cached',0)} if call['usage_available'] else {
                   'prompt_tokens':None,'completion_tokens':None,'total_tokens':None,'cached_prompt_tokens':None}
        result.append({**deepcopy(base),'event':'end','finished_utc':(started+timedelta(seconds=1)).isoformat(),
            'duration_seconds':1.0,'outcome':'success' if call['usage_available'] else 'error',
            'exception_chain':[],'usage':usage,'http_status':None,'request_ids':{}})
    return result


def write_journal(output,events,tail=''):
    path=Path(output)/'provider-events'/('azure-diagnostics-123-'+'1'*32+'.jsonl')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(event)+'\n' for event in events)+tail)
    return path


class JournalIntegrityIndependentTests(unittest.TestCase):
    def test_known_cached_and_unknown_calls_reconcile_A_once(self):
        _,known=bills();known[0]['cached']=4
        _,unknown=bills(unknown=True)
        physical=known+unknown
        with tempfile.TemporaryDirectory() as tmp:
            write_journal(tmp,journal_events(physical))
            proof=guard.validate_journals(tmp,physical,fatal_physical=True)
            self.assertEqual((proof['physical_ledger_calls'],proof['observed_sdk_ends']), (2,2))
            self.assertEqual(proof['unknown_usage_calls'],1)
            self.assertEqual(proof['journal_usage_unavailable'],1)
            with self.assertRaises(ValueError):guard.validate_journals(tmp,physical,fatal_physical=False)

    def test_unmatched_duplicate_or_partial_events_fail_even_on_fatal_paths(self):
        _,calls=bills();events=journal_events(calls)
        variants=[(events[:-1],''),(events+[deepcopy(events[-1])],''),(events,'{"event":')]
        for event_list,tail in variants:
            with self.subTest(events=len(event_list),tail=bool(tail)),tempfile.TemporaryDirectory() as tmp:
                write_journal(tmp,event_list,tail)
                with self.assertRaises(ValueError):guard.validate_journals(tmp,calls,fatal_physical=True)

    def test_known_journal_usage_cannot_change_or_claim_an_extra_send(self):
        _,calls=bills();events=journal_events(calls)
        wrong=deepcopy(events);wrong[-1]['usage']['prompt_tokens']+=1
        for event_list in (wrong,journal_events(calls+calls)):
            with self.subTest(events=len(event_list)),tempfile.TemporaryDirectory() as tmp:
                write_journal(tmp,event_list)
                with self.assertRaises(ValueError):guard.validate_journals(tmp,calls)

    def test_nonfatal_known_calls_require_complete_journal_coverage(self):
        _,calls=bills()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):guard.validate_journals(tmp,calls)
        # An admitted fatal accounting placeholder may precede the observer;
        # its absence is reported as a gap, never as a free successful request.
        _,unknown=bills(unknown=True)
        with tempfile.TemporaryDirectory() as tmp:
            proof=guard.validate_journals(tmp,unknown,fatal_physical=True)
            self.assertEqual(proof['ledger_minus_observed_sdk_calls'],1)
            self.assertEqual(proof['unknown_usage_calls'],1)

    def test_cached_usage_must_be_numeric_and_cannot_exceed_prompt(self):
        totals,calls=bills();calls[0]['cached']=11
        with self.assertRaises(ValueError):guard.ledger(totals,calls,False,(200000,30,30),1)
        totals,calls=bills();calls[0]['cached']=True
        with self.assertRaises(ValueError):guard.ledger(totals,calls,False,(200000,30,30),1)


class TerminalGateIndependentTests(unittest.TestCase):
    def terminal(self,output):
        return {'version':1,'owned_study_process_terminal':True,'owned_session_id':123,'exit_code':0,
            'observed_terminal_utc':'2026-10-15T12:00:00+00:00','expected_policy_outcomes':480,
            'saved_policy_outcomes':480,'accuracy_scored':False,'output_directory':str(Path(output).resolve())}

    def test_invalid_owned_terminal_proof_is_rejected_before_any_provenance_or_artifact_read(self):
        mutations=[{'owned_study_process_terminal':False},{'owned_session_id':0},{'exit_code':True},
                   {'observed_terminal_utc':'2026-10-15T12:00:00'},{'saved_policy_outcomes':479},
                   {'accuracy_scored':True},{'output_directory':'/private/tmp/unrelated-synthetic-output'}]
        for changed in mutations:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);output=root/'raw';output.mkdir();terminal=root/'terminal.json'
                terminal.write_text(json.dumps({**self.terminal(output),**changed}))
                with patch.object(guard,'provenance') as provenance,self.assertRaises(ValueError):
                    guard.run_check(root,root/'record',root/'pins','0'*64,output,terminal,
                                    hashlib.sha256(terminal.read_bytes()).hexdigest(),root/'report.json')
                provenance.assert_not_called()
                self.assertFalse((root/'report.json').exists())

    def test_479_outcomes_never_reach_case_or_journal_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'raw';output.mkdir();terminal=root/'terminal.json'
            terminal.write_text(json.dumps(self.terminal(output)))
            (output/'study_identity.json').write_text('{}')
            (output/'study_state.json').write_text(json.dumps({'saved_policy_outcomes':480,
                'expected_policy_outcomes':480,'correctness_scored':False,'phase':'generation_complete'}))
            (output/'study_outcomes.json').write_text(json.dumps([{} for _ in range(479)]))
            with patch.object(guard,'provenance',return_value=({'output_directory':str(output)},{},{},{})), \
                    patch.object(guard,'verify_case') as cases,patch.object(guard,'validate_journals') as journals, \
                    self.assertRaisesRegex(ValueError,'All declared first outcomes'):
                guard.run_check(root,root/'record',root/'pins','0'*64,output,terminal,
                                hashlib.sha256(terminal.read_bytes()).hexdigest(),root/'report.json')
            cases.assert_not_called();journals.assert_not_called()
            self.assertFalse((root/'report.json').exists())


class ManifestBindingIndependentTests(unittest.TestCase):
    """Only the manifest-path lifecycle; policy/audit checks stay tested above."""
    NEW = 'manifests/v8_mechanism_120_protocol_repair_20261015.json'

    def fixture(self, root, name=NEW):
        root=Path(root);record=root/'record';record.mkdir()
        sources={}
        for relative in guard.REQUIRED:
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('synthetic public source '+relative)
            sources[relative]=guard.sha(path)
        cases=[{'question_id':'synthetic-'+str(i),'followup_order':['W','R','C']} for i in range(120)]
        policy={'aggregate':{'tokens':200000,'tools':30,'sdk_attempts':30},
                'auxiliary':{'tokens':60000,'tools':6,'sdk_attempts':6,'probes':2}}
        study={'source_sha256':{},'policy':policy,'cases':cases}
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(study))
        sources[name]=guard.sha(path)
        prospective={'study_manifest':name,'study_manifest_sha256':guard.sha(path),'study_policy':policy}
        (record/'prospective_protocol.json').write_text(json.dumps(prospective))
        runtime={}
        for relative in guard.RUNTIME:
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('synthetic public runtime '+relative);runtime[relative]=guard.sha(target)
        binding={'provider_profile_sha256':'0'*64,
                 'adapter_sha256':runtime['.local-setup/azure_adapter.py'],
                 'bootstrap_sha256':runtime['.local-setup/sitecustomize.py']}
        (record/'runtime_adapter_manifest.json').write_text(json.dumps({'source_sha256':runtime,'archive_sha256':'1'*64}))
        (record/'expected_public_binding.json').write_text(json.dumps(binding))
        (record/'launch_commands.json').write_text('{}')
        (record/'design.txt').write_text('Synthetic design record; no inference')
        manifest={'model':'synthetic','output_directory':str(root/'raw'),
                  'source_sha256':sources,'study_manifest_sha256':guard.sha(path),
                  'source_archive_sha256':'2'*64,'design_sha256':guard.sha(record/'design.txt')}
        (record/'manifest.json').write_text(json.dumps(manifest))
        helper=SimpleNamespace(actual_source_inventory=lambda _root:set(sources),
            public_path=lambda _root,relative:Path(_root)/relative,
            B=SimpleNamespace(source_zip=Mock(),load_public_store=Mock(return_value=object())))
        return record,sources,prospective,manifest,study,runtime,binding,helper

    def test_distinct_prospective_selection_is_pinned_and_verified_without_old_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record,sources,_,_,study,_,_,helper=self.fixture(root)
            self.assertFalse((root/guard.LEGACY_STUDY_MANIFEST).exists())
            pins_path=root/'pins.json';pins=guard.prepare_pins(root,record,pins_path)
            self.assertEqual(pins['study_manifest_path'],self.NEW)
            self.assertEqual(pins['study_manifest_sha256'],sources[self.NEW])
            with patch.object(guard,'shared',return_value=helper):
                actual=guard.provenance(root,record,pins_path,guard.sha(pins_path))
            self.assertEqual(actual[1],study)
            self.assertEqual(helper.B.source_zip.call_count,2)

    def test_original_prospective_key_and_legacy_pin_path_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record,sources,_,_,_,_,_,helper=self.fixture(root,guard.LEGACY_STUDY_MANIFEST)
            pins_path=root/'pins.json';pins=guard.prepare_pins(root,record,pins_path)
            del pins['study_manifest_path'];pins_path.write_text(json.dumps(pins))
            with patch.object(guard,'shared',return_value=helper):
                guard.provenance(root,record,pins_path,guard.sha(pins_path))
            self.assertEqual(guard.bound_study_manifest(root,record,sources,pins)[0],guard.LEGACY_STUDY_MANIFEST)

    def test_missing_or_mismatched_pin_path_cannot_select_distinct_manifest(self):
        for changed in ({'study_manifest_path':guard.LEGACY_STUDY_MANIFEST},
                        {'study_manifest_sha256':'f'*64},{'study_manifest_path':None}):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);record,sources,_,_,_,_,_,_=self.fixture(root)
                pins={'study_manifest_path':self.NEW,'study_manifest_sha256':sources[self.NEW],**changed}
                with self.assertRaises(ValueError):guard.bound_study_manifest(root,record,sources,pins)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record,sources,_,_,_,_,_,_=self.fixture(root)
            with self.assertRaises(ValueError):
                guard.bound_study_manifest(root,record,sources,{'study_manifest_sha256':sources[self.NEW]})

    def test_unsafe_missing_or_ambiguous_prospective_path_fails_before_pin_creation(self):
        variants=[{'study_manifest':'../outside.json'},{'study_manifest':'/tmp/outside.json'},
                  {'study_manifest':'docs/selection.json'},{'study_manifest':'manifests/../outside.json'},
                  {'study_manifest':'manifests//selection.json'},{'study_manifest':'manifests/selection.txt'},
                  {'study_manifest':None},{'study_manifest_path':guard.LEGACY_STUDY_MANIFEST},
                  {'study_manifest_sha256':'f'*64}]
        for changed in variants:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);record,_,prospective,_,_,_,_,_=self.fixture(root)
                (record/'prospective_protocol.json').write_text(json.dumps({**prospective,**changed}))
                with self.assertRaises(ValueError):guard.prepare_pins(root,record,root/'pins.json')
                self.assertFalse((root/'pins.json').exists())

    def test_source_pin_removal_change_or_symlink_is_rejected(self):
        for mutation in ('missing','changed','symlink'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);record,sources,_,_,_,_,_,_=self.fixture(root)
                if mutation=='missing':del sources[self.NEW]
                elif mutation=='changed':sources[self.NEW]='f'*64
                else:
                    target=root/self.NEW;other=root/'elsewhere.json';target.rename(other);target.symlink_to(other)
                with self.assertRaises(ValueError):guard.bound_study_manifest(root,record,sources)

    def test_record_digest_or_pinned_protocol_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record,_,prospective,manifest,_,_,_,helper=self.fixture(root)
            pins_path=root/'pins.json';guard.prepare_pins(root,record,pins_path)
            (record/'prospective_protocol.json').write_text(json.dumps({**prospective,'study_manifest':guard.LEGACY_STUDY_MANIFEST}))
            with patch.object(guard,'shared',return_value=helper),self.assertRaisesRegex(ValueError,'Frozen record metadata'):
                guard.provenance(root,record,pins_path,guard.sha(pins_path))
            (record/'prospective_protocol.json').write_text(json.dumps(prospective))
            (record/'manifest.json').write_text(json.dumps({**manifest,'study_manifest_sha256':'f'*64}))
            with self.assertRaisesRegex(ValueError,'Record study manifest SHA'):
                guard.prepare_pins(root,record,root/'new-pins.json')

    def test_completion_identity_and_report_use_the_prospective_distinct_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record,_,_,manifest,study,runtime,binding,helper=self.fixture(root)
            pins_path=root/'pins.json';guard.prepare_pins(root,record,pins_path)
            output=root/'raw';output.mkdir();cases=study['cases']
            identity={'development_interface_only':False,'expected_policy_outcomes':480,
                'manifest_sha256':guard.sha(root/self.NEW),'model':manifest['model'],'policy':study['policy'],
                'runtime_case_metadata':cases,'freeze':{'public_provider_binding':binding,
                'source_sha256':manifest['source_sha256'],'runtime_sha256':runtime,
                'freeze_manifest_sha256':guard.sha(record/'manifest.json')}}
            (output/'study_identity.json').write_text(json.dumps(identity))
            (output/'study_state.json').write_text(json.dumps({'saved_policy_outcomes':480,
                'expected_policy_outcomes':480,'correctness_scored':False,'phase':'generation_complete'}))
            (output/'study_outcomes.json').write_text(json.dumps([{'question_id':case['question_id'],'study_arm':mode}
                for case in cases for mode in ('A',*case['followup_order'])]))
            terminal=root/'terminal.json';terminal.write_text(json.dumps(TerminalGateIndependentTests().terminal(output)))
            proof={'counts':{},'physical_calls':[],'files':{},'fatal_physical':False}
            with patch.object(guard,'shared',return_value=helper), \
                    patch.object(guard,'verify_case',return_value=proof) as verify, \
                    patch.object(guard,'validate_journals',return_value={'journals':[]}):
                result=guard.run_check(root,record,pins_path,guard.sha(pins_path),output,
                    terminal,guard.sha(terminal),root/'completion.json')
                self.assertEqual(verify.call_count,120)
                self.assertEqual(result['study_manifest_path'],self.NEW)
                self.assertEqual(result['study_manifest_sha256'],guard.sha(root/self.NEW))
                (output/'study_identity.json').write_text(json.dumps({**identity,'manifest_sha256':'f'*64}))
                with self.assertRaisesRegex(ValueError,'Actual study identity'):
                    guard.run_check(root,record,pins_path,guard.sha(pins_path),output,
                        terminal,guard.sha(terminal),root/'rejected.json')
                self.assertFalse((root/'rejected.json').exists())
